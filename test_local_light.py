#!/usr/bin/env python3
"""
Lightweight local test for CIFAR-10 training pipeline.
Uses CPU, small subset, few epochs to verify the flow works.
"""
import os
import sys
import csv

# Mock torchvision to avoid download/install issues on local Mac
class MockModule:
    def __getattr__(self, name): return self
sys.modules['torchvision'] = MockModule()
sys.modules['torchvision.datasets'] = MockModule()
sys.modules['torchvision.transforms'] = MockModule()

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# ── Inline architectures (no imports needed) ────────────────────────────────────
class SkipConnection(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        if in_ch != out_ch or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch))
        else:
            self.skip = nn.Identity()
    def forward(self, x, residual): return x + self.skip(residual)

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, activation='gelu', use_norm=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=not use_norm)
        self.norm = nn.LayerNorm([out_ch, 32, 32]) if use_norm else nn.Identity()
        if activation == 'gelu': self.act = nn.GELU()
        elif activation == 'tga': self.act = nn.Tanh()
        elif activation == 'relu': self.act = nn.ReLU(inplace=True)
    def forward(self, x): return self.act(self.norm(self.conv(x)))

class ThermoNet5(nn.Module):
    """Best G1 architecture: [64,128,256,128,64] with Skip"""
    def __init__(self, num_classes=10):
        super().__init__()
        channels = [3, 64, 128, 256, 128, 64]
        self.blocks = nn.ModuleList()
        self.skip_ops = nn.ModuleList()
        for i in range(len(channels)-1):
            self.blocks.append(ConvBlock(channels[i], channels[i+1], 'gelu', True))
            self.skip_ops.append(SkipConnection(channels[i], channels[i+1]) if i > 0 else None)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(64, num_classes)
    def forward(self, x):
        residual = None
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i > 0 and self.skip_ops[i] is not None:
                x = self.skip_ops[i](x, residual)
            residual = x.detach()
        return self.classifier(self.pool(x).flatten(1))

class ResNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Identity() if in_ch==out_ch and stride==1 else \
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm2d(out_ch))
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + self.shortcut(x))

class ResNet18CIFAR(nn.Module):
    """Simplified ResNet-18 for CIFAR"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, 2, 1)
        self.layer2 = self._make_layer(64, 128, 2, 2)
        self.layer3 = self._make_layer(128, 256, 2, 2)
        self.layer4 = self._make_layer(256, 512, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)
    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = [ResNetBlock(in_ch, out_ch, stride)]
        for _ in range(1, blocks): layers.append(ResNetBlock(out_ch, out_ch))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        return self.fc(self.avgpool(x).flatten(1))

MODELS = {
    'ThermoNet-5': ThermoNet5,
    'ResNet-18-CIFAR': ResNet18CIFAR,
}

# ── Fake CIFAR-10 data ─────────────────────────────────────────────────────────
def get_fake_cifar_loader(batch_size=32, n_samples=128):
    """Generate fake random data to test the pipeline."""
    X = torch.randn(n_samples, 3, 32, 32)
    y = torch.randint(0, 10, (n_samples,))
    loader = [(X[:batch_size], y[:batch_size])]  # single batch for speed test
    return loader

# ── Training loop (minimal) ────────────────────────────────────────────────────
def train_model(model_name, model_class, epochs=3, batch_size=32, device='cpu'):
    print(f"\n{'='*60}")
    print(f"Training {model_name} ({epochs} epochs, batch={batch_size}, device={device})")
    print(f"{'='*60}")
    
    model = model_class(num_classes=10).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    loader = get_fake_cifar_loader(batch_size=batch_size)
    
    results = []
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            correct += (output.argmax(1) == y_batch).sum().item()
            total += y_batch.size(0)
        
        acc = 100 * correct / total
        avg_loss = total_loss / max(len(loader), 1)
        
        print(f"  Epoch {epoch}/{epochs}: loss={avg_loss:.4f}, acc={acc:.1f}%")
        results.append({'epoch': epoch, 'loss': avg_loss, 'acc': acc})
    
    return results

# ── Main test ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Lightweight CIFAR-10 Training Pipeline Test (CPU)")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    os.makedirs('experiments/lift_test/results', exist_ok=True)
    
    for name, cls in MODELS.items():
        try:
            results = train_model(name, cls, epochs=3, batch_size=32, device=device)
            
            # Save CSV
            csv_path = f'experiments/lift_test/results/{name}/phase_a_metrics.csv'
            os.makedirs(f'experiments/lift_test/results/{name}', exist_ok=True)
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['epoch', 'loss', 'acc'])
                writer.writeheader()
                for r in results:
                    writer.writerow({'epoch': r['epoch'], 'loss': r['loss'], 'acc': r['acc']})
            print(f"  💾 Saved: {csv_path}")
            print(f"  ✅ {name} PASSED")
        except Exception as e:
            print(f"  ❌ {name} FAILED: {e}")
            import traceback; traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Lightweight test complete!")
    print("=" * 60)

if __name__ == '__main__':
    main()
