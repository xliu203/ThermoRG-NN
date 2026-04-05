#!/usr/bin/env python3
"""
ThermoRG Phase S1 v3 — Cooling Theory Validation (Compact)
==========================================================

Design: Validate BatchNorm and Skip as "cooling mechanisms"
- None_NoSkip: baseline (D=32,48,64,96 × 2 seeds = 8 runs)
  -> Already completed: D=32,48 × 2 seeds (4 runs), remaining: D=64,96 × 2 seeds (4 runs)
- BN_NoSkip: test BatchNorm cooling (D=32,48,64,96 × 2 seeds = 8 runs)
- LN_NoSkip: control for BN (D=32,48,64,96 × 1 seed = 4 runs)

Total: 16 runs × ~45min = ~12 hours (within 17h quota)
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

try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG — v3: drop None_Skip, LN uses 1 seed
# ──────────────────────────────────────────────────────────────────────────────

CONFIGS = [
    ('None_NoSkip',  'none',       [42, 123]),    # baseline, 2 seeds
    ('BN_NoSkip',    'batchnorm',  [42, 123]),    # test BatchNorm cooling, 2 seeds
    ('LN_NoSkip',    'layernorm',  [42]),         # control, 1 seed
]

D_VALUES = [32, 48, 64, 96]
EPOCHS = 200
BATCH_SIZE = 128
LR = 0.01
WD = 5e-4
MOMENTUM = 0.9

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_DIR = Path('./phase_s1_results_v3')
OUT_DIR.mkdir(exist_ok=True)
CKPT_DIR = OUT_DIR / 'checkpoints'
CKPT_DIR.mkdir(exist_ok=True)

GAMMA_TRACK_EVERY = [50, 100, 150, 200]  # track gamma at these epochs

# ──────────────────────────────────────────────────────────────────────────────
# NETWORK
# ──────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
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
# J_TOPO + D_EFF
# ──────────────────────────────────────────────────────────────────────────────

def compute_D_eff(W):
    if W.dim() == 4:
        W = W.view(W.size(0), -1)
    fro_sq = (W ** 2).sum().item()
    S = linalg.svd(W.to('cpu')).S
    spec_sq = S[0].item() ** 2 + 1e-12
    return fro_sq / spec_sq


def compute_J_topo(weights, d_input=3.0):
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
# ACTIVATION VARIANCE TRACKING
# ──────────────────────────────────────────────────────────────────────────────

class VarianceTracker:
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        def get_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self.handles.append(module.register_forward_hook(get_activation(name)))

    def get_variances(self):
        return {name: acts.var().item() for name, acts in self.activations.items()}

    def compute_gamma(self, init_variances):
        final_vars = self.get_variances()
        gamma_total = 0.0
        count = 0
        for name in init_variances:
            if name in final_vars:
                sigma_init = math.sqrt(init_variances[name])
                sigma_final = math.sqrt(final_vars[name])
                if sigma_init > 1e-8 and sigma_final > 1e-8:
                    gamma_total += abs(math.log(sigma_final / sigma_init))
                    count += 1
        return gamma_total / max(count, 1)

    def close(self):
        for h in self.handles:
            h.remove()


def measure_init_variance(model, batch_size=32):
    model = model.to(DEVICE)
    model.eval()
    tracker = VarianceTracker(model)
    dummy = torch.randn(batch_size, 3, 32, 32).to(DEVICE)
    with torch.no_grad():
        model(dummy)
    init_vars = tracker.get_variances()
    tracker.close()
    return init_vars


# ──────────────────────────────────────────────────────────────────────────────
# DATALOADER
# ──────────────────────────────────────────────────────────────────────────────

def get_dataloaders():
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


# ──────────────────────────────────────────────────────────────────────────────
# TRAIN (fixed: checkpoint resume actually works)
# ──────────────────────────────────────────────────────────────────────────────

def train_model(model, train_loader, val_loader, epochs, lr, wd, momentum,
                init_variances=None, track_gamma=False, start_epoch=0):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_epoch = 0
    gamma_history = []

    tracker = None
    if track_gamma and init_variances is not None:
        tracker = VarianceTracker(model)

    t0 = time.time()
    pbar = tqdm(range(epochs), desc="Training", leave=False) if TQDM_AVAILABLE else range(epochs)

    for epoch in pbar:
        if epoch < start_epoch:
            # Fast-forward scheduler without training
            scheduler.step()
            continue

        model.train()
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
        scheduler.step()

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

        # Gamma tracking
        if tracker and ((epoch + 1) in GAMMA_TRACK_EVERY or epoch == epochs - 1):
            model.eval()
            dummy_x = torch.randn(64, 3, 32, 32).to(DEVICE)
            with torch.no_grad():
                model(dummy_x)
            gamma = tracker.compute_gamma(init_variances)
            gamma_history.append({'epoch': epoch + 1, 'gamma': gamma})

        if TQDM_AVAILABLE and isinstance(pbar, tqdm):
            elapsed = (time.time() - t0) / 60
            pbar.set_postfix({'loss': f'{val_loss:.4f}', 'best': f'{best_val_loss:.4f}',
                              'elapsed': f'{elapsed:.1f}m'})

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
    from scipy.optimize import curve_fit

    def power_law(D, alpha, beta, E):
        return alpha * np.array(D) ** (-beta) + E

    Ds = np.array(d_values)
    losses = np.array([losses_by_d[d] for d in d_values])

    try:
        popt, _ = curve_fit(power_law, Ds, losses, p0=[10.0, 0.5, 0.5],
                           bounds=([0, 0, 0], [1000, 5, 10]), maxfev=10000)
        alpha, beta, E = popt
        preds = power_law(Ds, *popt)
        ss_res = ((losses - preds) ** 2).sum()
        ss_tot = ((losses - losses.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return {'alpha': float(alpha), 'beta': float(beta),
                'E': float(E), 'R2': float(r2)}
    except Exception as e:
        return {'alpha': None, 'beta': None, 'E': None, 'R2': 0.0, 'error': str(e)}


def estimate_params(base_ch):
    conv_params = 3*base_ch*9 + base_ch*base_ch*2*9 + base_ch*base_ch*2*9
    fc_params = 2*base_ch * 10
    return (conv_params + fc_params) / 1e6


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Phase S1 v3: Cooling Theory Validation")
    print(f"Device: {DEVICE} | Epochs: {EPOCHS} | D: {D_VALUES}")

    # Load metadata for checkpoint/resume
    meta_file = OUT_DIR / 'metadata.json'
    if meta_file.exists():
        with open(meta_file) as f:
            metadata = json.load(f)
    else:
        metadata = {'completed_runs': []}

    completed = set(metadata['completed_runs'])
    total_runs = sum(len(cfg[2]) * len(D_VALUES) for cfg in CONFIGS)
    remaining = total_runs - len([r for r in completed if is_valid_run(r)])

    train_loader, val_loader = get_dataloaders()
    print("Data loaded.")

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'epochs': EPOCHS,
        'total_runs': total_runs,
        'configs': []
    }

    t_start = time.time()

    # Print clean header
    print(f"Training {total_runs} runs ({len(completed)} done, {remaining} remaining)")
    print(f"Epochs: {EPOCHS} | D: {D_VALUES} | Checkpoint dir: {CKPT_DIR}")
    print("-" * 60)

    for config_name, norm_type, seeds in CONFIGS:
        cfg_start = time.time()

        # J_topo at init (silent)
        model_init = ValidationNet(base_ch=64, norm_type=norm_type, use_skip=False).to(DEVICE)
        weights = model_init.get_conv_weights()
        J_topo = compute_J_topo(weights)
        del model_init
        torch.cuda.empty_cache()

        cfg_result = {
            'name': config_name,
            'norm': norm_type,
            'J_topo_init': float(J_topo),
            'D_results': {}
        }

        for base_ch in D_VALUES:
            n_params = estimate_params(base_ch)
            d_result = {'base_ch': base_ch, 'n_params_M': float(n_params),
                        'seeds': {}, 'avg_val_loss': None, 'avg_gamma': None}

            losses = []
            gammas = []

            for seed in seeds:
                run_name = f"{config_name}_ch{base_ch}_s{seed}"
                ckpt_path = CKPT_DIR / f"{run_name}.pt"

                # Check if completed — silent load
                if run_name in completed:
                    if ckpt_path.exists():
                        ckpt = torch.load(ckpt_path, map_location=DEVICE)
                        result = ckpt['result']
                        losses.append(result['best_val_loss'])
                        if result.get('gamma_history'):
                            gammas.append(np.mean([g['gamma'] for g in result['gamma_history']]))
                    continue

                # Actually train
                print(f"  [{config_name}] D={base_ch} seed={seed} ({n_params:.1f}M) ... ", end='', flush=True)

                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                model = ValidationNet(base_ch=base_ch, norm_type=norm_type,
                                     use_skip=False).to(DEVICE)

                init_vars = measure_init_variance(model, batch_size=64)

                # Check for checkpoint to resume
                start_epoch = 0
                if ckpt_path.exists():
                    ckpt = torch.load(ckpt_path, map_location=DEVICE)
                    model.load_state_dict(ckpt['model_state'])
                    start_epoch = ckpt.get('epoch', 0) + 1

                t0 = time.time()

                result = train_model(
                    model, train_loader, val_loader,
                    epochs=EPOCHS, lr=LR, wd=WD, momentum=MOMENTUM,
                    init_variances=init_vars, track_gamma=True,
                    start_epoch=start_epoch
                )

                elapsed = (time.time() - t0) / 60

                # Save checkpoint
                torch.save({
                    'epoch': EPOCHS - 1,
                    'model_state': model.state_dict(),
                    'result': result
                }, ckpt_path)

                avg_g = np.mean([g['gamma'] for g in result['gamma_history']]) if result.get('gamma_history') else 0
                print(f"loss={result['best_val_loss']:.4f} acc={result['best_val_acc']:.4f} "
                      f"γ={avg_g:.3f} [{elapsed:.0f}m]")

                gammas.append(avg_g)
                losses.append(result['best_val_loss'])
                d_result['seeds'][str(seed)] = result

                # Mark completed
                completed.add(run_name)
                metadata['completed_runs'] = list(completed)
                with open(meta_file, 'w') as f:
                    json.dump(metadata, f)

                del model
                torch.cuda.empty_cache()

            if losses:
                d_result['avg_val_loss'] = float(np.mean(losses))
                d_result['avg_gamma'] = float(np.mean(gammas)) if gammas else None

            cfg_result['D_results'][str(base_ch)] = d_result

        # Fit scaling law
        losses_by_d = {ch: cfg_result['D_results'][str(ch)]['avg_val_loss']
                       for ch in D_VALUES if cfg_result['D_results'][str(ch)]['avg_val_loss'] is not None}
        if losses_by_d:
            fit = fit_scaling_law(losses_by_d, list(losses_by_d.keys()))
            cfg_result['scaling_fit'] = fit
            print(f"  [{config_name}] β={fit['beta']:.4f} R²={fit['R2']:.4f} "
                  f"(wall: {(time.time()-cfg_start)/60:.0f}min)")

        cfg_result['wall_time_min'] = (time.time() - cfg_start) / 60
        all_results['configs'].append(cfg_result)

    # Save results
    out_file = OUT_DIR / 'phase_s1_results.json'
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    base_beta = None
    for cfg in all_results['configs']:
        fit = cfg.get('scaling_fit', {})
        if fit.get('beta') and base_beta is None and cfg['name'] == 'None_NoSkip':
            base_beta = fit['beta']

    print(f"\n{'Config':<15} {'β':<10} {'φ(β)':<10} {'γ':<10}")
    print("-" * 50)
    for cfg in all_results['configs']:
        fit = cfg.get('scaling_fit', {})
        beta = fit.get('beta') or 0
        phi = beta / base_beta if base_beta and base_beta > 0 else 0
        gammas = [cfg['D_results'][str(ch)]['avg_gamma']
                  for ch in D_VALUES if cfg['D_results'][str(ch)]['avg_gamma'] is not None]
        avg_gamma = sum(gammas)/len(gammas) if gammas else 0
        print(f"{cfg['name']:<15} {beta:<10.4f} {phi:<10.3f} {avg_gamma:<10.4f}")

    print("\n" + "=" * 70)
    print("COOLING ANALYSIS")
    print("=" * 70)
    if base_beta:
        bn_cfg = next((c for c in all_results['configs'] if c['name'] == 'BN_NoSkip'), None)
        ln_cfg = next((c for c in all_results['configs'] if c['name'] == 'LN_NoSkip'), None)
        if bn_cfg and bn_cfg.get('scaling_fit', {}).get('beta'):
            phi_bn = bn_cfg['scaling_fit']['beta'] / base_beta
            print(f"  φ_BN = {phi_bn:.3f}  (theory: ≈ 0.66)")
        if ln_cfg and ln_cfg.get('scaling_fit', {}).get('beta'):
            phi_ln = ln_cfg['scaling_fit']['beta'] / base_beta
            print(f"  φ_LN = {phi_ln:.3f}  (theory: ≈ 0.85-0.95)")

    total_time = (time.time() - t_start) / 60
    print(f"\nTotal runtime: {total_time:.1f} min")
    print(f"Results: {out_file}")

    return all_results


def is_valid_run(run_name):
    """Check if run_name matches v3 config pattern."""
    import re
    return bool(re.match(r'^(None|BN|LN)_NoSkip_ch(32|48|64|96)_s\d+$', run_name))


if __name__ == '__main__':
    main()
