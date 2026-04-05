#!/usr/bin/env python3
"""
Phase S1 Medium: Cooling Theory Validation (Balanced)
======================================================
Moderate complexity simulation to test BatchNorm and Skip cooling effects.
"""

import json, math, time, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import linalg

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device('cpu')

# ──────────────────────────────────────────────────────────────────────────────
# ConvNet with configurable norm and skip
# ──────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm_type='none', use_skip=False, stride=1):
        super().__init__()
        self.use_skip = use_skip
        self.norm_type = norm_type

        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=stride, bias=False)

        if norm_type == 'layernorm':
            self.norm = nn.LayerNorm([out_ch, 32, 32])
        elif norm_type == 'batchnorm':
            self.norm = nn.BatchNorm2d(out_ch)
        else:
            self.norm = nn.Identity()

        self.act = nn.GELU()

        if use_skip and in_ch == out_ch and stride == 1:
            self.skip = nn.Identity()
        elif use_skip:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.skip = None

    def forward(self, x):
        out = self.norm(self.conv(x))
        out = self.act(out)
        if self.use_skip and self.skip is not None:
            out = out + self.skip(x)
        return out


class MediumNet(nn.Module):
    """Medium ConvNet: [3, 64, 128, 128] channels"""

    def __init__(self, base_ch=32, norm_type='none', use_skip=False, n_classes=10):
        super().__init__()
        channels = [3, base_ch, base_ch*2, base_ch*2]

        self.blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.blocks.append(ConvBlock(
                channels[i], channels[i+1],
                norm_type=norm_type,
                use_skip=(i > 0 and use_skip),
                stride=1
            ))

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[-1], n_classes)

    def get_weights(self):
        weights = []
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                weights.append(m.weight.data)
        return weights

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ──────────────────────────────────────────────────────────────────────────────
# J_topo computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_D_eff(W):
    """D_eff = ||W||_F^2 / ||W||_2^2"""
    if W.dim() == 4:
        W = W.view(W.size(0), -1)
    fro_sq = (W ** 2).sum().item()
    S = linalg.svd(W).S
    spec_sq = S[0].item() ** 2 + 1e-12
    return fro_sq / spec_sq


def compute_J_topo(weights):
    """J_topo = exp(-|Σlog η_l| / L)"""
    if not weights:
        return 0.0
    eta_vals = []
    d_prev = 3.0  # input channels
    for W in weights:
        if W.dim() == 4:
            W_flat = W.view(W.size(0), -1)
            D_eff = compute_D_eff(W_flat)
            eta = D_eff / max(d_prev, 1e-8)
        else:
            D_eff = compute_D_eff(W)
            eta = D_eff / max(d_prev, 1e-8)
        eta_vals.append(max(eta, 1e-8))
        d_prev = D_eff
    L = len(eta_vals)
    log_sum = sum(abs(math.log(e)) for e in eta_vals)
    return math.exp(-log_sum / L) if L > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic CIFAR-like task
# ──────────────────────────────────────────────────────────────────────────────

def generate_task(n_samples=5000, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Simple synthetic images: random patterns + labels
    X = torch.randn(n_samples, 3, 32, 32)
    # Labels based on simple feature: sum of first channel > threshold
    y = ((X[:, 0, :, :].mean(dim=(1, 2)) > 0).long() % 10)
    # Mix with random to make it harder
    y = (y + torch.randint(0, 10, (n_samples,))) % 10

    return X, y


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_model(model, X_train, y_train, X_val, y_val, epochs=150, lr=0.01):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_epoch = 0

    batch_size = 128
    n_samples = X_train.shape[0]

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_samples)
        total_loss = 0
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i+batch_size]
            X_batch = X_train[idx]
            y_batch = y_train[idx]

            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_out = model(X_val)
            val_loss = criterion(val_out, y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1

    return best_val_loss, best_epoch


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Phase S1 Medium: Cooling Theory Validation")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    # 6 configurations
    configs = [
        ('None_NoSkip', 'none', False),
        ('LN_NoSkip', 'layernorm', False),
        ('BN_NoSkip', 'batchnorm', False),
        ('None_Skip', 'none', True),
        ('LN_Skip', 'layernorm', True),
        ('BN_Skip', 'batchnorm', True),
    ]

    # D values via base_ch
    D_configs = [
        ('small', 16),   # ~0.2M params
        ('medium', 32),  # ~0.8M params
        ('large', 48),   # ~1.8M params
    ]

    results = {'configs': []}

    # Generate data
    X_all, y_all = generate_task(n_samples=5000, seed=42)
    X_train, y_train = X_all[:4000], y_all[:4000]
    X_val, y_val = X_all[4000:], y_all[4000:]

    SEEDS = [42, 123]

    for config_name, norm_type, use_skip in configs:
        print(f"\n[{config_name}]")
        cfg_result = {'name': config_name, 'norm': norm_type, 'skip': use_skip, 'D': {}}

        # J_topo at init
        model_init = MediumNet(base_ch=32, norm_type=norm_type, use_skip=use_skip)
        weights = model_init.get_weights()
        J_topo = compute_J_topo(weights)
        cfg_result['J_topo_init'] = round(J_topo, 4)
        print(f"  J_topo(init) = {J_topo:.4f}")

        for d_name, base_ch in D_configs:
            print(f"  Training {d_name} (ch={base_ch})...", end=" ", flush=True)
            d_result = {'ch': base_ch, 'seeds': []}

            for seed in SEEDS:
                torch.manual_seed(seed)
                np.random.seed(seed)

                model = MediumNet(base_ch=base_ch, norm_type=norm_type,
                                use_skip=use_skip)
                best_loss, best_ep = train_model(
                    model, X_train, y_train, X_val, y_val,
                    epochs=150, lr=0.01
                )
                d_result['seeds'].append({'seed': seed, 'loss': round(best_loss, 4), 'epoch': best_ep})
                print(f".", end="", flush=True)

            avg_loss = np.mean([s['loss'] for s in d_result['seeds']])
            d_result['avg_loss'] = round(avg_loss, 4)
            print(f" avg_loss={avg_loss:.4f}")
            cfg_result['D'][d_name] = d_result

        # Estimate β from scaling
        losses = [cfg_result['D'][d]['avg_loss'] for d in ['small', 'medium', 'large']]
        # Simple β estimate: log-log slope
        log_losses = np.log(np.array(losses) + 0.1)
        log_Ds = np.log(np.array([0.2, 0.8, 1.8]))  # approximate params in M
        beta = (log_losses[-1] - log_losses[0]) / (log_Ds[-1] - log_Ds[0])
        beta = max(0.05, min(abs(beta), 3.0))
        cfg_result['beta_est'] = round(beta, 3)
        cfg_result['E_est'] = round(min(losses), 4)
        print(f"  → β≈{beta:.3f}, E≈{min(losses):.4f}")

        results['configs'].append(cfg_result)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)

    baseline = results['configs'][0]  # None_NoSkip
    print(f"\n{'Config':<15} {'J_topo':<8} {'β':<8} {'φ(β)':<8} {'E':<8}")
    print("-" * 60)
    for cfg in results['configs']:
        phi = cfg['beta_est'] / max(baseline['beta_est'], 0.01)
        print(f"{cfg['name']:<15} {cfg['J_topo_init']:<8.4f} {cfg['beta_est']:<8.3f} "
              f"{phi:<8.3f} {cfg['E_est']:<8.4f}")

    # Cooling analysis
    print("\n" + "=" * 60)
    print("COOLING THEORY ANALYSIS")
    print("=" * 60)

    def get_cfg(name):
        return next(c for c in results['configs'] if c['name'] == name)

    bn = get_cfg('BN_NoSkip')
    ln = get_cfg('LN_NoSkip')
    none = get_cfg('None_NoSkip')

    # Normalization effect
    print(f"\n1. Normalization Cooling:")
    print(f"   LN vs None: β={ln['beta_est']:.3f} vs {none['beta_est']:.3f} → φ={ln['beta_est']/none['beta_est']:.3f}")
    print(f"   BN vs None: β={bn['beta_est']:.3f} vs {none['beta_est']:.3f} → φ={bn['beta_est']/none['beta_est']:.3f}")
    print(f"   BN vs LN:   β={bn['beta_est']:.3f} vs {ln['beta_est']:.3f} → φ={bn['beta_est']/max(ln['beta_est'],0.01):.3f}")

    # Skip effect
    skip_ln = get_cfg('LN_Skip')
    noskip_ln = get_cfg('LN_NoSkip')
    print(f"\n2. Skip Cooling (LN):")
    print(f"   Skip vs NoSkip: β={skip_ln['beta_est']:.3f} vs {noskip_ln['beta_est']:.3f} → φ={skip_ln['beta_est']/max(noskip_ln['beta_est'],0.01):.3f}")

    # Combined
    bn_skip = get_cfg('BN_Skip')
    print(f"\n3. Combined (BN+Skip) vs Baseline:")
    print(f"   φ_Combined = {bn_skip['beta_est']/max(baseline['beta_est'],0.01):.3f}")
    print(f"   φ_Additivity = φ_BN × φ_Skip = {(bn['beta_est']/max(ln['beta_est'],0.01)) * (skip_ln['beta_est']/max(noskip_ln['beta_est'],0.01)):.3f}")

    # Save
    out_file = 'phase_s1_medium_results.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_file}")

    return results


if __name__ == '__main__':
    main()
