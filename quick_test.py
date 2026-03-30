#!/usr/bin/env python3
"""Quick sanity check: 1 forward + 1 backward pass on all 15 architectures."""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Inline minimal building blocks (mirrors architectures.py) ────────────────

class TGAActivation(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 1.0) * 0.7 + x * 0.3

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, activation='gelu', use_norm=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=not use_norm)
        self.norm = nn.LayerNorm([out_ch, 32, 32]) if use_norm else nn.Identity()
        if activation == 'gelu':   self.act = nn.GELU()
        elif activation == 'tga': self.act = TGAActivation()
        elif activation == 'relu': self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class SkipConnection(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels))
        else:
            self.skip = nn.Identity()
    def forward(self, x, residual):
        return x + self.skip(residual)

class BottleneckBlock(nn.Module):
    def __init__(self, in_ch, out_ch, bottleneck_dim, activation='gelu', use_norm=True):
        super().__init__()
        self.compress = nn.Conv2d(in_ch, bottleneck_dim, kernel_size=1, bias=not use_norm)
        self.norm1 = nn.LayerNorm([bottleneck_dim, 32, 32]) if use_norm else nn.Identity()
        self.conv = nn.Conv2d(bottleneck_dim, bottleneck_dim, kernel_size=3, padding=1, bias=not use_norm)
        self.norm2 = nn.LayerNorm([bottleneck_dim, 32, 32]) if use_norm else nn.Identity()
        self.expand = nn.Conv2d(bottleneck_dim, out_ch, kernel_size=1, bias=not use_norm)
        self.norm3 = nn.LayerNorm([out_ch, 32, 32]) if use_norm else nn.Identity()
        if activation == 'gelu':   self.act = nn.GELU()
        elif activation == 'tga':  self.act = TGAActivation()
        elif activation == 'relu': self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.act(self.norm1(self.compress(x)))
        x = self.act(self.norm2(self.conv(x)))
        return self.act(self.norm3(self.expand(x)))

# ── Inline all 15 architectures (from architectures.py) ───────────────────────

class ThermoNet3(nn.Module):
    """G1-1: Thermogeometric Optimal 3-layer network."""
    def __init__(self, num_classes=10):
        super().__init__()
        channels = [3, 64, 64, 128, 128]
        self.blocks = nn.ModuleList()
        self.skip_ops = nn.ModuleList()
        for i in range(len(channels)-1):
            self.blocks.append(ConvBlock(channels[i], channels[i+1], activation='gelu', use_norm=True))
            self.skip_ops.append(SkipConnection(channels[i], channels[i+1]) if i > 0 else None)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(128, num_classes)
    def forward(self, x):
        residual = None
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i > 0 and self.skip_ops[i] is not None:
                x = self.skip_ops[i](x, residual)
            residual = x.detach()
        return self.classifier(self.pool(x).flatten(1))

class ThermoNet5(nn.Module):
    """G1-2: Thermogeometric Optimal 5-layer network."""
    def __init__(self, num_classes=10):
        super().__init__()
        channels = [3, 64, 128, 256, 128, 64]
        self.blocks = nn.ModuleList()
        self.skip_ops = nn.ModuleList()
        for i in range(len(channels)-1):
            self.blocks.append(ConvBlock(channels[i], channels[i+1], activation='gelu', use_norm=True))
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

class ThermoNet7(nn.Module):
    """G1-3: Thermogeometric Optimal 7-layer network."""
    def __init__(self, num_classes=10):
        super().__init__()
        channels = [3, 64, 64, 128, 128, 256, 128, 64]
        self.blocks = nn.ModuleList()
        self.skip_ops = nn.ModuleList()
        for i in range(len(channels)-1):
            self.blocks.append(ConvBlock(channels[i], channels[i+1], activation='gelu', use_norm=True))
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

class ThermoNet9(nn.Module):
    """G1-4: Thermogeometric Optimal 9-layer network."""
    def __init__(self, num_classes=10):
        super().__init__()
        channels = [3] + [64]*8
        self.blocks = nn.ModuleList()
        for i in range(len(channels)-1):
            self.blocks.append(ConvBlock(channels[i], channels[i+1], activation='gelu', use_norm=True))
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(64, num_classes)
    def forward(self, x):
        residuals = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i % 2 == 0 and i > 0:
                skip_idx = (i//2)-1
                if skip_idx < len(residuals):
                    residual = residuals[skip_idx]
                    if residual.shape[1] == x.shape[1]:
                        x = x + residual
            residuals.append(x.detach())
        return self.classifier(self.pool(x).flatten(1))

class ThermoBot3(nn.Module):
    """G2-1: Topology Destroyer - ThermoNet-3 with 8x bottleneck."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.block1 = ConvBlock(3, 64, activation='gelu', use_norm=True)
        self.block2 = ConvBlock(64, 64, activation='gelu', use_norm=True)
        self.skip1 = SkipConnection(64, 64)
        self.bottleneck = BottleneckBlock(64, 128, bottleneck_dim=8, activation='gelu', use_norm=True)
        self.block3 = ConvBlock(128, 128, activation='gelu', use_norm=True)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(128, num_classes)
    def forward(self, x):
        residual = x
        x = self.block1(x)
        residual = x.detach()
        x = self.block2(x)
        x = self.skip1(x, residual)
        x = self.bottleneck(x)
        x = self.block3(x)
        return self.classifier(self.pool(x).flatten(1))

class ThermoBot5(nn.Module):
    """G2-2: Topology Destroyer - ThermoNet-5 with 8x bottleneck."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.block1 = ConvBlock(3, 64, activation='gelu', use_norm=True)
        self.block2 = ConvBlock(64, 128, activation='gelu', use_norm=True)
        self.skip1 = SkipConnection(64, 128)
        self.bottleneck = BottleneckBlock(128, 128, bottleneck_dim=16, activation='gelu', use_norm=True)
        self.block3 = ConvBlock(128, 128, activation='gelu', use_norm=True)
        self.block4 = ConvBlock(128, 64, activation='gelu', use_norm=True)
        self.skip2 = SkipConnection(128, 64)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(64, num_classes)
    def forward(self, x):
        residual = x
        x = self.block1(x)
        residual = x.detach()
        x = self.block2(x)
        x = self.skip1(x, residual)
        x = self.bottleneck(x)
        residual = x.detach()
        x = self.block3(x)
        x = self.block4(x)
        x = self.skip2(x, residual)
        return self.classifier(self.pool(x).flatten(1))

class ThermoBot7(nn.Module):
    """G2-3: Topology Destroyer - ThermoNet-7 with two bottlenecks."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.block1 = ConvBlock(3, 64, activation='tga', use_norm=True)
        self.block2 = ConvBlock(64, 64, activation='tga', use_norm=True)
        self.skip1 = SkipConnection(64, 64)
        self.bottleneck1 = BottleneckBlock(64, 128, bottleneck_dim=8, activation='tga', use_norm=True)
        self.block3 = ConvBlock(128, 128, activation='tga', use_norm=True)
        self.block4 = ConvBlock(128, 256, activation='tga', use_norm=True)
        self.skip2 = SkipConnection(128, 256)
        self.bottleneck2 = BottleneckBlock(256, 128, bottleneck_dim=16, activation='tga', use_norm=True)
        self.block5 = ConvBlock(128, 64, activation='tga', use_norm=True)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(64, num_classes)
    def forward(self, x):
        residual = x
        x = self.block1(x)
        residual = x.detach()
        x = self.block2(x)
        x = self.skip1(x, residual)
        x = self.bottleneck1(x)
        residual = x.detach()
        x = self.block3(x)
        x = self.block4(x)
        x = self.skip2(x, residual)
        x = self.bottleneck2(x)
        x = self.block5(x)
        return self.classifier(self.pool(x).flatten(1))

class ThermoBot9(nn.Module):
    """G2-4: Topology Destroyer - ThermoNet-9 with bottlenecks."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.block1 = ConvBlock(3, 64, activation='tga', use_norm=True)
        self.block2 = ConvBlock(64, 64, activation='tga', use_norm=True)
        self.skip1 = SkipConnection(64, 64)
        self.bottleneck1 = BottleneckBlock(64, 128, bottleneck_dim=8, activation='tga', use_norm=True)
        self.block3 = ConvBlock(128, 128, activation='tga', use_norm=True)
        self.block4 = ConvBlock(128, 256, activation='tga', use_norm=True)
        self.skip2 = SkipConnection(128, 256)
        self.bottleneck2 = BottleneckBlock(256, 128, bottleneck_dim=16, activation='tga', use_norm=True)
        self.block5 = ConvBlock(128, 128, activation='tga', use_norm=True)
        self.block6 = ConvBlock(128, 64, activation='tga', use_norm=True)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(64, num_classes)
    def forward(self, x):
        residual = x
        x = self.block1(x)
        residual = x.detach()
        x = self.block2(x)
        x = self.skip1(x, residual)
        x = self.bottleneck1(x)
        residual = x.detach()
        x = self.block3(x)
        x = self.block4(x)
        x = self.skip2(x, residual)
        x = self.bottleneck2(x)
        x = self.block5(x)
        x = self.block6(x)
        return self.classifier(self.pool(x).flatten(1))

class ReLUFurnace3(nn.Module):
    """G3-1: ReLU ablation of ThermoNet-3 (no Norm, no Skip)."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
        ])
        self.activations = nn.ModuleList([nn.ReLU(inplace=True) for _ in range(4)])
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(128, num_classes)
    def forward(self, x):
        for block, act in zip(self.blocks, self.activations):
            x = act(block(x))
        return self.classifier(self.pool(x).flatten(1))

class ReLUFurnace5(nn.Module):
    """G3-2: ReLU ablation of ThermoNet-5 (no Norm, no Skip)."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
        ])
        self.activations = nn.ModuleList([nn.ReLU(inplace=True) for _ in range(5)])
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(64, num_classes)
    def forward(self, x):
        for block, act in zip(self.blocks, self.activations):
            x = act(block(x))
        return self.classifier(self.pool(x).flatten(1))

class ReLUFurnace7(nn.Module):
    """G3-3: ReLU ablation of ThermoNet-7 (no Norm, no Skip)."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
        ])
        self.activations = nn.ModuleList([nn.ReLU(inplace=True) for _ in range(7)])
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(64, num_classes)
    def forward(self, x):
        for block, act in zip(self.blocks, self.activations):
            x = act(block(x))
        return self.classifier(self.pool(x).flatten(1))

class ReLUFurnace9(nn.Module):
    """G3-4: ReLU ablation of ThermoNet-9 (no Norm, no Skip)."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Conv2d(3, 64, kernel_size=3, padding=1)] +
            [nn.Conv2d(64, 64, kernel_size=3, padding=1) for _ in range(8)])
        self.activations = nn.ModuleList([nn.ReLU(inplace=True) for _ in range(9)])
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(64, num_classes)
    def forward(self, x):
        for block, act in zip(self.blocks, self.activations):
            x = act(block(x))
        return self.classifier(self.pool(x).flatten(1))

# ── Standard baselines (ResNet/VGG/DenseNet) ─────────────────────────────────

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Identity() if in_ch==out_ch and stride==1 else \
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm2d(out_ch))
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))

class ResNet18CIFAR(nn.Module):
    """G4-1: ResNet-18 adapted for CIFAR (32x32 input)."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)
    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = [BasicBlock(in_ch, out_ch, stride)]
        for _ in range(1, blocks): layers.append(BasicBlock(out_ch, out_ch))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x)
        return self.fc(x.flatten(1))

class VGG11CIFAR(nn.Module):
    """G4-2: VGG-11 adapted for CIFAR."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32→16
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16→8
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 8→4
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 4→2
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 2→1
        )
        self.classifier = nn.Linear(512, num_classes)
    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1)
        return self.classifier(x.flatten(1))
        return self.classifier(x.flatten(1))


# ── DenseNet-40 (bottleneck: BN→1x1 conv→3x3 conv) ──────────────────────────
class _DenseLayer(nn.Module):
    def __init__(self, in_ch, growth_rate, bn_size=4):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, bn_size * growth_rate, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, 3, padding=1, bias=False)
    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return torch.cat([x, out], 1)

class _DenseBlock(nn.Sequential):
    def __init__(self, in_ch, num_layers, growth_rate):
        super().__init__()
        for i in range(num_layers):
            super(_DenseBlock, self).add_module(
                f'denselayer{i}', _DenseLayer(in_ch + i * growth_rate, growth_rate))

class _Transition(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.AvgPool2d(2, 2))

class DenseNet40CIFAR(nn.Module):
    def __init__(self, num_classes=10, growth_rate=12):
        super().__init__()
        num_layers = (40 - 4) // 3  # 12 layers per block
        nChannels = 2 * growth_rate  # = 24
        self.features = nn.Sequential(
            nn.Conv2d(3, nChannels, 3, padding=1, bias=False),
            nn.BatchNorm2d(nChannels), nn.ReLU(inplace=True),
        )
        for i, bl in enumerate([num_layers]*3):
            self.features.add_module(f'denseblock{i+1}', _DenseBlock(nChannels, bl, growth_rate))
            nChannels += bl * growth_rate
            if i != 2:
                self.features.add_module(f'transition{i+1}', _Transition(nChannels, nChannels // 2))
                nChannels = nChannels // 2
        self.features.add_module('norm_last', nn.BatchNorm2d(nChannels))
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(nChannels, num_classes)
    def forward(self, x):
        return self.classifier(
            self.avgpool(F.relu(self.features(x))).flatten(1))

# ── Test runner ──────────────────────────────────────────────────────────────
fake_input = torch.randn(2, 3, 32, 32, requires_grad=True)
loss_fn = nn.CrossEntropyLoss()

archs = [
    ("ThermoNet-3",       ThermoNet3(num_classes=10)),
    ("ThermoNet-5",       ThermoNet5(num_classes=10)),
    ("ThermoNet-7",       ThermoNet7(num_classes=10)),
    ("ThermoNet-9",       ThermoNet9(num_classes=10)),
    ("ThermoBot-3",       ThermoBot3(num_classes=10)),
    ("ThermoBot-5",       ThermoBot5(num_classes=10)),
    ("ThermoBot-7",       ThermoBot7(num_classes=10)),
    ("ThermoBot-9",       ThermoBot9(num_classes=10)),
    ("ReLUFurnace-3",     ReLUFurnace3(num_classes=10)),
    ("ReLUFurnace-5",     ReLUFurnace5(num_classes=10)),
    ("ReLUFurnace-7",     ReLUFurnace7(num_classes=10)),
    ("ReLUFurnace-9",     ReLUFurnace9(num_classes=10)),
    ("ResNet-18-CIFAR",   ResNet18CIFAR(num_classes=10)),
    ("VGG-11-CIFAR",      VGG11CIFAR(num_classes=10)),
    ("DenseNet-40-CIFAR", DenseNet40CIFAR(num_classes=10)),
]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Input: {fake_input.shape} | Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
print("=" * 65)
all_passed = True
for name, model in archs:
    model.train()
    params = count_parameters(model)
    try:
        output = model(fake_input)
        assert output.shape == (2, 10), f"Expected (2,10), got {output.shape}"
        loss = loss_fn(output, torch.randint(0, 10, (2,)))
        loss.backward()
        has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_grad, "No gradients!"
        print(f"✅ {name:22s} | {params/1e6:5.2f}M | out={output.shape} | backward=✓")
    except Exception as e:
        print(f"❌ {name:22s} | ERROR: {e}")
        all_passed = False
    model.zero_grad()

print("=" * 65)
print("🎉 ALL PASSED" if all_passed else "🚨 FAILED — fix before Kaggle!")
