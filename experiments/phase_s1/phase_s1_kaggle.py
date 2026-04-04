#!/usr/bin/env python3
"""
ThermoRG Phase S1 — Cooling Theory Validation
============================================

Purpose: Validate BatchNorm and Skip connection as "cooling mechanisms"

Theory predictions:
    β(J, γ) = β₀(J) · φ(γ)
    E(J, γ) = E₀(J) · exp(-κγ)
    φ(γ) = (γ_c/(γ_c+γ))·exp(-γ/γ_c)
    γ_total = γ_norm + γ_skip

Key test:
    - BatchNorm should reduce β by φ ≈ 0.66 compared to LayerNorm
    - Skip should reduce β by φ ≈ 0.93-0.98 compared to no skip
    - Combined: φ_total = φ_BN × φ_skip

Experiment:
    - 6 configs: norm (none/LN/BN) × skip (no/yes)
    - 4 D values via width scaling: base_ch ∈ {32, 48, 64, 96}
    - CIFAR-10, 200 epochs, 2 seeds
    - Track activation variances for γ_norm calculation
"""

import json, math, os, sys, time, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

# 6 configurations: norm_type × use_skip
CONFIGS = [
    ('None_NoSkip',  'none',       False),
    ('LN_NoSkip',    'layernorm',  False),
    ('BN_NoSkip',    'batchnorm',  False),
    ('None_Skip',   'none',       True),
    ('LN_Skip',      'layernorm',  True),
    ('BN_Skip',      'batchnorm',  True),
]

# D values via base channel width
D_VALUES = [32, 48, 64, 96]  # ~{0.2M, 0.5M, 0.9M, 2M} params
SEEDS = [42, 123]
EPOCHS = 200
BATCH_SIZE = 128
LR = 0.01
WD = 5e-4
MOMENTUM = 0.9

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_DIR = Path('./phase_s1_results')
OUT_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# NETWORK
# ──────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Single conv block with configurable norm and skip."""

    def __init__(self, in_ch, out_ch, norm_type='none', use_skip=False, stride=1):
        super().__init__()
        self.norm_type = norm_type
        self.use_skip = use_skip

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


class ValidationNet(nn.Module):
    """Medium ConvNet: [3, ch, 2*ch, 2*ch]"""

    def __init__(self, base_ch=64, norm_type='none', use_skip=False, n_classes=10):
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

    def get_conv_weights(self):
        return [m.weight.data for m in self.modules() if isinstance(m, nn.Conv2d)]

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ──────────────────────────────────────────────────────────────────────────────
# J_TOPO COMPUTATION
# ──────────────────────────────────────────────────────────────────────────────

def compute_D_eff(W):
    """D_eff = ||W||_F² / ||W||_2²"""
    if W.dim() == 4:
        W = W.view(W.size(0), -1)
    fro_sq = (W ** 2).sum().item()
    S = linalg.svd(W.to('cpu')).S
    spec_sq = S[0].item() ** 2 + 1e-12
    return fro_sq / spec_sq


def compute_J_topo(weights, d_input=3.0):
    """J_topo = exp(-|Σlog η_l| / L)"""
    eta_vals = []
    d_prev = d_input
    for W in weights:
        if W.dim() == 4:
            W = W.view(W.size(0), -1)
        D_eff = compute_D_eff(W)
        eta = D_eff / max(d_prev, 1e-8)
        eta_vals.append(max(eta, 1e-8))
        d_prev = D_eff
    L = len(eta_vals)
    log_sum = sum(abs(math.log(e)) for e in eta_vals)
    return math.exp(-log_sum / L) if L > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────────────────────────────────────

def get_dataloaders():
    """CIFAR-10 dataloaders."""
    import torchvision.transforms as T
    from torchvision.datasets import CIFAR10
    from torch.utils.data import DataLoader

    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
    ])
    transform_val = T.Compose([
        T.ToTensor(),
        T.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
    ])

    train_ds = CIFAR10(root='./data', train=True, transform=transform_train, download=True)
    val_ds   = CIFAR10(root='./data', train=False, transform=transform_val, download=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=2, pin_memory=True)
    return train_loader, val_loader


def train_model(model, train_loader, val_loader, epochs, lr, wd, momentum):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(epochs):
        # Train
        model.train()
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Evaluate
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                out = model(X)
                val_loss += criterion(out, y).item() * X.size(0)
                correct += (out.argmax(1) == y).sum().item()
                total += X.size(0)

        val_loss /= total
        val_acc = correct / total

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch + 1

    return {
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'final_val_loss': val_loss,
        'final_val_acc': val_acc,
    }


# ──────────────────────────────────────────────────────────────────────────────
# SCALING LAW FIT
# ──────────────────────────────────────────────────────────────────────────────

def fit_scaling_law(losses_by_d, d_values):
    """Fit L(D) = α·D^(-β) + E"""
    from scipy.optimize import curve_fit

    def power_law(D, alpha, beta, E):
        return alpha * np.array(D) ** (-beta) + E

    Ds = np.array(d_values)
    losses = np.array([losses_by_d[d] for d in d_values])

    try:
        popt, _ = curve_fit(power_law, Ds, losses,
                           p0=[10.0, 0.5, 0.5],
                           bounds=([0, 0, 0], [1000, 5, 10]),
                           maxfev=10000)
        alpha, beta, E = popt

        preds = power_law(Ds, alpha, beta, E)
        ss_res = ((losses - preds) ** 2).sum()
        ss_tot = ((losses - losses.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return {'alpha': float(alpha), 'beta': float(beta),
                'E': float(E), 'R2': float(r2)}
    except Exception as e:
        return {'alpha': None, 'beta': None, 'E': None, 'R2': 0.0,
                'error': str(e)}


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Phase S1: Cooling Theory Validation")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Configs: {len(CONFIGS)}")
    print(f"D values: {D_VALUES}")
    print(f"Seeds: {SEEDS}")
    print(f"Epochs: {EPOCHS}")

    train_loader, val_loader = get_dataloaders()
    print("Data loaded.")

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'epochs': EPOCHS, 'batch_size': BATCH_SIZE,
            'lr': LR, 'wd': WD, 'momentum': MOMENTUM,
            'd_values': D_VALUES, 'seeds': SEEDS
        },
        'configs': []
    }

    total_start = time.time()

    for config_name, norm_type, use_skip in CONFIGS:
        print(f"\n{'='*60}")
        print(f"[{config_name}] norm={norm_type}, skip={use_skip}")
        print(f"{'='*60}")

        cfg_start = time.time()
        cfg_result = {
            'name': config_name,
            'norm': norm_type,
            'skip': use_skip,
            'D_results': {}
        }

        # J_topo at initialization
        model_init = ValidationNet(base_ch=64, norm_type=norm_type, use_skip=use_skip).to(DEVICE)
        weights_init = model_init.get_conv_weights()
        J_topo_init = compute_J_topo(weights_init)
        cfg_result['J_topo_init'] = J_topo_init
        print(f"J_topo(init) = {J_topo_init:.4f}")

        del model_init
        torch.cuda.empty_cache()

        # Train for each D value
        for base_ch in D_VALUES:
            print(f"\n  base_ch={base_ch} ({'约' + str(estimate_params(base_ch)) + 'M params'})...", end=' ', flush=True)

            d_result = {'base_ch': base_ch, 'seeds': {}}
            losses = []

            for seed in SEEDS:
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)

                model = ValidationNet(base_ch=base_ch, norm_type=norm_type,
                                   use_skip=use_skip).to(DEVICE)

                result = train_model(model, train_loader, val_loader,
                                   epochs=EPOCHS, lr=LR, wd=WD, momentum=MOMENTUM)

                d_result['seeds'][seed] = result
                losses.append(result['best_val_loss'])
                print('.', end='', flush=True)

                del model
                torch.cuda.empty_cache()

            avg_loss = float(np.mean(losses))
            d_result['avg_val_loss'] = avg_loss
            print(f" avg_loss={avg_loss:.4f}")
            cfg_result['D_results'][str(base_ch)] = d_result

        # Fit scaling law
        losses_by_d = {str(ch): cfg_result['D_results'][str(ch)]['avg_val_loss']
                       for ch in D_VALUES}
        fit = fit_scaling_law(losses_by_d, D_VALUES)
        cfg_result['scaling_fit'] = fit

        print(f"\n  Scaling fit: α={fit['alpha']:.2f}, β={fit['beta']:.4f}, "
              f"E={fit['E']:.4f}, R²={fit['R2']:.4f}")

        cfg_time = (time.time() - cfg_start) / 60
        cfg_result['wall_time_min'] = cfg_time
        print(f"  Time: {cfg_time:.1f} min")

        all_results['configs'].append(cfg_result)

    total_time = (time.time() - total_start) / 60
    all_results['total_time_min'] = total_time

    # Save
    out_file = OUT_DIR / 'phase_s1_results.json'
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_file}")
    print(f"Total runtime: {total_time:.1f} min")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    baseline = next(c for c in all_results['configs'] if c['name'] == 'None_NoSkip')
    base_beta = baseline['scaling_fit']['beta'] or 0.4

    print(f"\n{'Config':<15} {'J_topo':<8} {'Beta':<10} {'φ(beta)':<10} {'E':<10}")
    print("-" * 70)

    for cfg in all_results['configs']:
        fit = cfg['scaling_fit']
        beta = fit['beta'] or 0
        phi = beta / base_beta if base_beta > 0 else 0
        print(f"{cfg['name']:<15} {cfg['J_topo_init']:<8.4f} "
              f"{beta:<10.4f} {phi:<10.3f} {fit['E']:<10.4f}")

    # Analysis
    print("\n" + "=" * 70)
    print("COOLING ANALYSIS")
    print("=" * 70)

    def get_cfg(name):
        return next(c for c in all_results['configs'] if c['name'] == name)

    bn = get_cfg('BN_NoSkip')
    ln = get_cfg('LN_NoSkip')
    none = get_cfg('None_NoSkip')
    skip_ln = get_cfg('LN_Skip')
    noskip_ln = get_cfg('LN_NoSkip')

    # Normalization effect
    beta_none = none['scaling_fit']['beta'] or 0.4
    beta_ln = ln['scaling_fit']['beta'] or 0.4
    beta_bn = bn['scaling_fit']['beta'] or 0.4

    print(f"\n1. Normalization Cooling:")
    print(f"   None vs LN: φ = {beta_ln/beta_none:.3f}")
    print(f"   None vs BN: φ = {beta_bn/beta_none:.3f}")
    print(f"   LN vs BN:   φ = {beta_bn/beta_ln:.3f}")
    print(f"   Prediction: φ_BN ≈ 0.66")

    # Skip effect
    beta_skip = skip_ln['scaling_fit']['beta'] or 0.4
    beta_noskip = noskip_ln['scaling_fit']['beta'] or 0.4
    print(f"\n2. Skip Cooling (LN baseline):")
    print(f"   Skip/NoSkip: φ = {beta_skip/beta_noskip:.3f}")
    print(f"   Prediction: φ_skip ≈ 0.93-0.98")

    # Combined
    bn_skip = get_cfg('BN_Skip')
    beta_bnskip = bn_skip['scaling_fit']['beta'] or 0.4
    phi_combined = beta_bnskip / beta_none
    phi_add = (beta_bn/beta_none) * (beta_skip/beta_noskip)
    print(f"\n3. Combined (BN+Skip):")
    print(f"   φ_combined = {phi_combined:.3f}")
    print(f"   φ_additivity = φ_BN × φ_skip = {phi_add:.3f}")

    return all_results


def estimate_params(base_ch):
    """Rough param estimate in millions."""
    # [3, ch, 2ch, 2ch] conv layers + fc
    conv_params = 3*base_ch*9 + base_ch*base_ch*2*9 + base_ch*base_ch*2*9
    fc_params = 2*base_ch * 10
    return (conv_params + fc_params) / 1e6


if __name__ == '__main__':
    main()
