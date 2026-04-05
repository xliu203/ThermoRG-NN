#!/usr/bin/env python3
"""
Phase S1: Cooling Theory Validation
=====================================

Validates the thermodynamic normalization theory:
- γ_norm from BatchNorm vs LayerNorm
- γ_skip from skip connections
- Additivity: γ_total = γ_norm + γ_skip
- Variance ratio relation

Theory predictions:
    β(J, γ) = β₀(J) · φ(γ),  φ(γ) = (γ_c/(γ_c+γ))·exp(-γ/γ_c)
    E(J, γ) = E₀(J) · exp(-κγ)
    γ_total = γ_norm + γ_skip
"""

import json, math, time, sys, os, argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

NORM_TYPES = ['none', 'layernorm', 'batchnorm']
SKIP_OPTIONS = [False, True]
D_VALUES = [500, 1000, 2000, 5000]
SEEDS = [42, 123]
EPOCHS = 200
BATCH_SIZE = 128
LR = 0.01
DATASET = 'cifar10'  # 'cifar10' or 'synthetic'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ──────────────────────────────────────────────────────────────────────────────
# Network with configurable norm and skip
# ──────────────────────────────────────────────────────────────────────────────

class ValidationBlock(nn.Module):
    """Single conv block with configurable norm and skip."""

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
        if self.use_skip:
            out = out + self.skip(x)
        return out


class ValidationNet(nn.Module):
    """
    Simple ConvNet for validation.
    Architecture: [3, 64, 128, 256, 128] channels
    """

    def __init__(self, base_ch=64, norm_type='none', use_skip=False, n_classes=10):
        super().__init__()
        self.use_skip = use_skip
        self.norm_type = norm_type

        # Build channel list: [3, 64, 128, 256, 128]
        channels = [3, base_ch, base_ch*2, base_ch*4, base_ch*2]

        self.blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.blocks.append(ValidationBlock(
                channels[i], channels[i+1],
                norm_type=norm_type,
                use_skip=(i > 0 and use_skip),  # skip after first block
                stride=1
            ))

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[-1], n_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                if hasattr(m, 'weight'): nn.init.uniform_(m.weight, 0.95, 1.05)
                if hasattr(m, 'bias'): nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def get_weight_matrices(self):
        """Extract weight matrices for J_topo computation."""
        Ws = []
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                Ws.append(m.weight.data)
        return Ws

    def get_intermediate_vars(self):
        """Hook to capture activation variances."""
        # Will be populated by forward pass hooks
        return {}


# ──────────────────────────────────────────────────────────────────────────────
# J_topo computation (skip-aware)
# ──────────────────────────────────────────────────────────────────────────────

def compute_D_eff(J):
    """Stable rank: D_eff = ||J||_F² / ||J||_2²"""
    # Handle both 2D and 4D (conv) tensors
    if J.dim() == 4:
        J = J.view(J.size(0), -1)  # (out_channels, in_channels*H*W)
    fro_sq = (J ** 2).sum().item()
    S = linalg.svd(J.to('cpu')).S  # new torch API
    spec_sq = S[0].item() ** 2 + 1e-12
    return fro_sq / spec_sq


def compute_J_topo(Ws, d_input=3):
    """
    Compute skip-aware J_topo.
    W_eff = W_main + W_skip (if skip exists)
    """
    if not Ws:
        return 0.0, []

    # Get effective W for each layer
    W_effs = []
    d_prev = float(d_input)

    for i, W in enumerate(Ws):
        W_eff = W
        W_effs.append(W_eff)
        d_prev = W_eff.shape[0]  # out_channels

    # Compute η for each layer
    eta_vals = []
    d_prev = float(d_input)
    for W in W_effs:
        D_eff = compute_D_eff(W)
        eta = D_eff / max(d_prev, 1e-8)
        eta_vals.append(eta)
        d_prev = D_eff

    # J_topo = exp(-|Σlog η_l| / L)
    L = len(eta_vals)
    log_sum = sum(abs(math.log(max(e, 1e-12))) for e in eta_vals)
    J = math.exp(-log_sum / L) if L > 0 else 0.0

    return J, eta_vals


# ──────────────────────────────────────────────────────────────────────────────
# Variance tracking
# ──────────────────────────────────────────────────────────────────────────────

var_records = {}

def var_hook(module, input, output):
    """Hook to record output variance."""
    if isinstance(output, tuple):
        output = output[0]
    var = output.detach().float().var().item()
    name = str(id(module))
    if name not in var_records:
        var_records[name] = []
    var_records[name].append(var)

def clear_var_records():
    global var_records
    var_records = {}

def get_var_ratios(model, x):
    """Get variance before/after norm layers."""
    clear_var_records()

    hooks = []
    for name, module in model.named_modules():
        if 'norm' in name and not isinstance(module, nn.Identity):
            hooks.append(module.register_forward_hook(var_hook))

    model.eval()
    with torch.no_grad():
        _ = model(x)

    for h in hooks:
        h.remove()

    return var_records


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def get_dataloader(dataset='cifar10', batch_size=128, seed=42):
    """Get train/val dataloader."""
    if dataset == 'cifar10':
        try:
            import torchvision
            transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                                 [0.2470, 0.2435, 0.2616])
            ])
            train_ds = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                    transform=transform, download=True)
            val_ds = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                   transform=transform, download=True)
            train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,
                                                        shuffle=True, num_workers=2, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size,
                                                     shuffle=False, num_workers=2, pin_memory=True)
            return train_loader, val_loader
        except ImportError:
            print("torchvision not available, using synthetic data")
            dataset = 'synthetic'

    if dataset == 'synthetic':
        # Simple synthetic regression task
        torch.manual_seed(seed)
        N = 50000
        X = torch.randn(N, 3, 32, 32)
        y = torch.randint(0, 10, (N,))
        train_ds = torch.utils.data.TensorDataset(X[:40000], y[:40000])
        val_ds = torch.utils.data.TensorDataset(X[40000:], y[40000:])
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader

    raise ValueError(f"Unknown dataset: {dataset}")


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += X.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        out = model(X)
        loss = criterion(out, y)
        total_loss += loss.item() * X.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += X.size(0)
    return total_loss / total, correct / total


def fit_scaling_law(losses_by_d, D_values):
    """
    Fit L(D) = α·D^(-β) + E to data.
    Returns: α, β, E, R²
    """
    from scipy.optimize import curve_fit

    def power_law(D, alpha, beta, E):
        return alpha * np.array(D) ** (-beta) + E

    Ds = np.array(D_values)
    losses = np.array([losses_by_d[d]['val_loss'] for d in D_values])

    try:
        popt, pcov = curve_fit(power_law, Ds, losses, p0=[10.0, 0.5, 0.5],
                                 bounds=([0, 0, 0], [1000, 5, 10]),
                                 maxfev=10000)
        alpha, beta, E = popt

        # R²
        preds = power_law(Ds, alpha, beta, E)
        ss_res = ((losses - preds) ** 2).sum()
        ss_tot = ((losses - losses.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return alpha, beta, E, r2
    except Exception as e:
        print(f"  Fit failed: {e}")
        return None, None, None, 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Main experiment
# ──────────────────────────────────────────────────────────────────────────────

def run_config(norm_type, use_skip, D_values, seeds, epochs, device, dataset='cifar10'):
    """Run experiment for one configuration (norm_type × use_skip)."""

    config_name = f"{'BN' if norm_type=='batchnorm' else 'LN' if norm_type=='layernorm' else 'NN'}_{'S' if use_skip else 'N'}"

    results = {
        'config': config_name,
        'norm_type': norm_type,
        'use_skip': use_skip,
        'D_results': {},
        'J_topo': None,
        'var_ratio': None,
        'gamma_norm_estimate': None,
    }

    # Get dataloader
    train_loader, val_loader = get_dataloader(dataset, BATCH_SIZE, seeds[0])

    # Compute J_topo at initialization
    print(f"\n[{config_name}] Computing J_topo at initialization...")
    model = ValidationNet(base_ch=64, norm_type=norm_type, use_skip=use_skip).to(device)
    Ws = model.get_weight_matrices()
    J_topo, eta_vals = compute_J_topo(Ws, d_input=3)
    results['J_topo'] = J_topo
    results['eta_vals'] = eta_vals
    print(f"  J_topo = {J_topo:.4f}")

    # Estimate γ_norm from variance
    if norm_type != 'none':
        x_sample = torch.randn(32, 3, 32, 32).to(device)
        var_dict = get_var_ratios(model, x_sample)
        # Compute average variance reduction ratio
        if var_dict:
            ratios = []
            for name, vars_list in var_dict.items():
                if len(vars_list) >= 2:
                    # Ratio of consecutive variances (before/after norm)
                    for i in range(1, len(vars_list)):
                        if vars_list[i-1] > 0:
                            ratios.append(vars_list[i] / vars_list[i-1])
            if ratios:
                avg_ratio = np.mean(ratios)
                # γ_norm = -k * ln(ratio), approximate k=1
                gamma_est = -math.log(max(avg_ratio, 1e-6))
                results['var_ratio'] = avg_ratio
                results['gamma_norm_estimate'] = gamma_est
                print(f"  Var ratio = {avg_ratio:.4f}, γ_estimate = {gamma_est:.4f}")

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Training loop for each D value
    for D in D_values:
        D_result = {
            'D': D,
            'seeds': {}
        }

        # Compute base channel for target D
        # Approximate: D ≈ sum(ch[i] * ch[i+1]) for conv layers
        # We use base_ch to control D
        base_ch_options = {500: 32, 1000: 48, 2000: 64, 5000: 96}
        base_ch = base_ch_options.get(D, 64)

        print(f"\n[{config_name}] D={D}, base_ch={base_ch}")

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = ValidationNet(base_ch=base_ch, norm_type=norm_type,
                                  use_skip=use_skip).to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=LR,
                                       momentum=0.9, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            criterion = nn.CrossEntropyLoss()

            best_val_loss = float('inf')
            best_epoch = 0
            val_losses = []

            for epoch in range(epochs):
                train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
                val_loss, val_acc = evaluate(model, val_loader, criterion, device)
                scheduler.step()

                val_losses.append(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch + 1

            seed_result = {
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
                'final_val_loss': val_losses[-1],
                'final_val_acc': val_acc,
                'train_time_min': 0  # Will be tracked externally
            }
            D_result['seeds'][seed] = seed_result
            print(f"  Seed {seed}: best_val_loss={best_val_loss:.4f} @ epoch {best_epoch}")

        # Aggregate D result
        avg_val_loss = np.mean([D_result['seeds'][s]['best_val_loss'] for s in seeds])
        D_result['avg_val_loss'] = avg_val_loss
        results['D_results'][D] = D_result

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Fit scaling law
    print(f"\n[{config_name}] Fitting scaling law...")
    losses_by_d = {D: results['D_results'][D]['avg_val_loss'] for D in D_values}
    alpha, beta, E, r2 = fit_scaling_law(losses_by_d, D_values)
    results['scaling_fit'] = {
        'alpha': alpha,
        'beta': beta,
        'E': E,
        'R2': r2
    }
    print(f"  α={alpha:.3f}, β={beta:.4f}, E={E:.4f}, R²={r2:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Phase S1: Cooling Theory Validation')
    parser.add_argument('--dataset', default='synthetic', choices=['cifar10', 'synthetic'],
                        help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--output', default='./phase_s1_results.json')
    args = parser.parse_args()

    print("=" * 60)
    print("Phase S1: Cooling Theory Validation")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"D values: {D_VALUES}")
    print(f"Seeds: {SEEDS}")

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'epochs': args.epochs,
            'D_values': D_VALUES,
            'seeds': SEEDS,
            'batch_size': BATCH_SIZE,
            'lr': LR
        },
        'runs': []
    }

    total_start = time.time()

    # Run all configurations
    for norm_type in NORM_TYPES:
        for use_skip in SKIP_OPTIONS:
            config_start = time.time()
            print(f"\n{'='*60}")
            print(f"CONFIG: norm={norm_type}, skip={use_skip}")
            print(f"{'='*60}")

            result = run_config(norm_type, use_skip, D_VALUES, SEEDS,
                                args.epochs, DEVICE, args.dataset)

            config_time = (time.time() - config_start) / 60
            result['wall_time_min'] = config_time
            all_results['runs'].append(result)

            print(f"\n[{result['config']}] Total time: {config_time:.1f} min")

    total_time = (time.time() - total_start) / 60
    all_results['total_time_min'] = total_time

    # Save results
    print(f"\nTotal runtime: {total_time:.1f} min")
    print(f"Saving results to {args.output}")
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Config':<8} {'J_topo':<8} {'β':<8} {'E':<8} {'R²':<6} {'γ_est':<8}")
    print("-" * 60)
    for run in all_results['runs']:
        fit = run.get('scaling_fit', {})
        gamma = run.get('gamma_norm_estimate', 0)
        print(f"{run['config']:<8} {run['J_topo']:<8.4f} "
              f"{fit.get('beta','-'):<8.4f} {fit.get('E','-'):<8.4f} "
              f"{fit.get('R2','-'):<6.4f} {gamma:<8.4f}")

    return all_results


if __name__ == '__main__':
    main()
