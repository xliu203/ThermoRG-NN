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
    - 4 configs: norm (none/LN/BN) × skip (no/yes)
    - 4 D values via width scaling: base_ch ∈ {32, 48, 64, 96}
    - CIFAR-10, 200 epochs, 2 seeds
    - **Track activation variances for γ calculation**
"""

import json, math, os, sys, time, warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

CONFIGS = [
    ('None_NoSkip',  'none',       False),
    ('LN_NoSkip',    'layernorm',  False),
    ('BN_NoSkip',    'batchnorm',  False),
    ('None_Skip',    'none',       True),
]

D_VALUES = [32, 48, 64, 96]
SEEDS = [42, 123]
EPOCHS = 200
BATCH_SIZE = 128
LR = 0.01
WD = 5e-4
MOMENTUM = 0.9

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_DIR = Path('./phase_s1_results')
OUT_DIR.mkdir(exist_ok=True)

GAMMA_TRACK_EVERY = 50  # track gamma at these epochs

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
# ACTIVATION VARIANCE TRACKING (γ calculation)
# ──────────────────────────────────────────────────────────────────────────────

class VarianceTracker:
    """Track activation variances through the network for γ calculation."""

    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks on all conv and linear layers."""
        def get_activation(name):
            def hook(module, input, output):
                # Capture post-activation variance
                self.activations[name] = output.detach()
            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                handle = module.register_forward_hook(get_activation(name))
                self.handles.append(handle)

    def get_variances(self):
        """Return dict of layer_name -> variance."""
        variances = {}
        for name, acts in self.activations.items():
            # Global variance across batch, height, width
            variances[name] = acts.var().item()
        return variances

    def compute_gamma(self, init_variances):
        """
        Compute γ = Σ_l |log(σ_l / σ_l^init)| / L
        γ measures how much the activation variance has shifted from init.
        """
        final_variances = self.get_variances()
        gamma_total = 0.0
        count = 0

        for name in init_variances:
            if name in final_variances:
                sigma_init = math.sqrt(init_variances[name])
                sigma_final = math.sqrt(final_variances[name])
                if sigma_init > 1e-8 and sigma_final > 1e-8:
                    gamma_total += abs(math.log(sigma_final / sigma_init))
                    count += 1

        return gamma_total / max(count, 1)

    def close(self):
        for handle in self.handles:
            handle.remove()


def measure_init_variance(model, batch_size=32):
    """Measure initial activation variances on a dummy batch."""
    model.eval()
    tracker = VarianceTracker(model)

    # Dummy forward pass
    dummy_input = torch.randn(batch_size, 3, 32, 32).to(DEVICE)
    with torch.no_grad():
        model(dummy_input)

    init_variances = tracker.get_variances()
    tracker.close()
    return init_variances


# ──────────────────────────────────────────────────────────────────────────────
# DATALOADER
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
                             num_workers=2)
    return train_loader, val_loader


def train_model(model, train_loader, val_loader, epochs, lr, wd, momentum,
                init_variances=None, track_gamma=False):
    """
    Train model. If track_gamma=True and init_variances provided,
    measure γ at epochs defined by GAMMA_TRACK_EVERY.
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_epoch = 0
    gamma_history = []

    # Setup gamma tracking
    tracker = None
    if track_gamma and init_variances is not None:
        tracker = VarianceTracker(model)

    t0 = time.time()

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

        # Track gamma at specified epochs
        if tracker and (epoch + 1) in GAMMA_TRACK_EVERY or \
           (tracker and epoch == epochs - 1):
            model.eval()
            # Forward pass to capture activations
            dummy_x = torch.randn(64, 3, 32, 32).to(DEVICE)
            with torch.no_grad():
                model(dummy_x)
            gamma = tracker.compute_gamma(init_variances)
            gamma_history.append({'epoch': epoch + 1, 'gamma': gamma})

        # Periodic logging
        if (epoch + 1) % 50 == 0 or epoch == epochs - 1:
            elapsed = (time.time() - t0) / 60
            epochs_left = epochs - epoch - 1
            eta = elapsed / (epoch + 1) * epochs_left if epoch > 0 else 0
            gamma_str = f", γ={gamma_history[-1]['gamma']:.4f}" if gamma_history else ""
            print(f"  Epoch {epoch+1}/{epochs}: loss={val_loss:.4f}, "
                  f"best={best_val_loss:.4f}, elapsed={elapsed:.1f}min, ETA={eta:.1f}min{gamma_str}")

    if tracker:
        tracker.close()

    return {
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'final_val_loss': val_loss,
        'final_val_acc': val_acc,
        'gamma_history': gamma_history,
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
    print(f"Track gamma: True")

    train_loader, val_loader = get_dataloaders()
    print("Data loaded.")

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'epochs': EPOCHS, 'batch_size': BATCH_SIZE,
            'lr': LR, 'wd': WD, 'momentum': MOMENTUM,
            'd_values': D_VALUES, 'seeds': SEEDS,
            'gamma_tracked': True
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
            n_params = estimate_params(base_ch)
            print(f"\n  base_ch={base_ch} (~{n_params:.1f}M params)...", end=' ', flush=True)

            d_result = {'base_ch': base_ch, 'n_params_M': n_params, 'seeds': {}}
            losses = []
            gamma_all = []

            for seed in SEEDS:
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)

                model = ValidationNet(base_ch=base_ch, norm_type=norm_type,
                                     use_skip=use_skip).to(DEVICE)

                # Measure initial variance for gamma tracking
                init_vars = measure_init_variance(model, batch_size=64)

                result = train_model(model, train_loader, val_loader,
                                   epochs=EPOCHS, lr=LR, wd=WD, momentum=MOMENTUM,
                                   init_variances=init_vars, track_gamma=True)

                d_result['seeds'][seed] = result
                losses.append(result['best_val_loss'])

                # Average gamma across tracking epochs
                if result['gamma_history']:
                    avg_gamma = np.mean([g['gamma'] for g in result['gamma_history']])
                    gamma_all.append(avg_gamma)

                print('.', end='', flush=True)

                del model
                torch.cuda.empty_cache()

            avg_loss = float(np.mean(losses))
            avg_gamma = float(np.mean(gamma_all)) if gamma_all else None
            d_result['avg_val_loss'] = avg_loss
            d_result['avg_gamma'] = avg_gamma
            print(f" avg_loss={avg_loss:.4f}" + (f", γ={avg_gamma:.4f}" if avg_gamma else ""))

            cfg_result['D_results'][str(base_ch)] = d_result

        # Fit scaling law
        losses_by_d = {ch: cfg_result['D_results'][str(ch)]['avg_val_loss']
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

    print(f"\n{'Config':<15} {'J_topo':<8} {'Beta':<10} {'φ(beta)':<10} {'γ':<10} {'E':<10}")
    print("-" * 70)

    for cfg in all_results['configs']:
        fit = cfg['scaling_fit']
        beta = fit['beta'] or 0
        phi = beta / base_beta if base_beta > 0 else 0
        # Get average gamma across all D values
        gammas = [cfg['D_results'][str(ch)]['avg_gamma']
                  for ch in D_VALUES if cfg['D_results'][str(ch)]['avg_gamma'] is not None]
        avg_gamma = sum(gammas)/len(gammas) if gammas else 0
        print(f"{cfg['name']:<15} {cfg['J_topo_init']:<8.4f} "
              f"{beta:<10.4f} {phi:<10.3f} {avg_gamma:<10.4f} {fit['E']:<10.4f}")

    # Analysis
    print("\n" + "=" * 70)
    print("COOLING ANALYSIS")
    print("=" * 70)

    def get_cfg(name):
        return next(c for c in all_results['configs'] if c['name'] == name)

    bn = get_cfg('BN_NoSkip')
    ln = get_cfg('LN_NoSkip')
    none = get_cfg('None_NoSkip')
    skip = get_cfg('None_Skip')

    beta_none = none['scaling_fit']['beta'] or 0.4
    beta_ln = ln['scaling_fit']['beta'] or 0.4
    beta_bn = bn['scaling_fit']['beta'] or 0.4
    beta_skip = skip['scaling_fit']['beta'] or 0.4

    print(f"\n1. Normalization Cooling:")
    print(f"   None vs LN: φ = {beta_ln/beta_none:.3f}")
    print(f"   None vs BN: φ = {beta_bn/beta_none:.3f}")
    print(f"   LN vs BN:   φ = {beta_bn/beta_ln:.3f}")
    print(f"   Prediction: φ_BN ≈ 0.66")

    print(f"\n2. Skip Cooling:")
    print(f"   Skip/NoSkip: φ = {beta_skip/beta_none:.3f}")
    print(f"   Prediction: φ_skip ≈ 0.93-0.98")

    print(f"\n3. γ (activation variance shift):")
    def get_gamma(cfg_name):
        cfg = get_cfg(cfg_name)
        gammas = [cfg['D_results'][str(ch)]['avg_gamma']
                  for ch in D_VALUES if cfg['D_results'][str(ch)]['avg_gamma'] is not None]
        return sum(gammas)/len(gammas) if gammas else 0

    gamma_none = get_gamma('None_NoSkip')
    gamma_ln = get_gamma('LN_NoSkip')
    gamma_bn = get_gamma('BN_NoSkip')
    gamma_skip = get_gamma('None_Skip')

    print(f"   None:    γ = {gamma_none:.4f}")
    print(f"   LN:      γ = {gamma_ln:.4f}")
    print(f"   BN:      γ = {gamma_bn:.4f}")
    print(f"   Skip:    γ = {gamma_skip:.4f}")
    print(f"   Theory: γ_c ≈ 0.22 (critical cooling point)")

    return all_results


def estimate_params(base_ch):
    """Rough param estimate in millions."""
    conv_params = 3*base_ch*9 + base_ch*base_ch*2*9 + base_ch*base_ch*2*9
    fc_params = 2*base_ch * 10
    return (conv_params + fc_params) / 1e6


if __name__ == '__main__':
    main()
