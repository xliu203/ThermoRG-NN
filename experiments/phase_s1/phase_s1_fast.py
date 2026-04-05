#!/usr/bin/env python3
"""
Phase S1 Fast: Quick Cooling Theory Validation
===============================================
Minimal simulation to test cooling theory predictions.
Uses simple MLP on synthetic task, fewer epochs.
"""

import json, math, time, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device('cpu')

# ──────────────────────────────────────────────────────────────────────────────
# Simple MLP with configurable norm and skip
# ──────────────────────────────────────────────────────────────────────────────

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, n_layers=3, output_dim=10,
                 norm_type='none', use_skip=False):
        super().__init__()
        self.norm_type = norm_type
        self.use_skip = use_skip

        layers = []
        for i in range(n_layers):
            in_d = input_dim if i == 0 else hidden_dim
            out_d = output_dim if i == n_layers - 1 else hidden_dim

            layers.append(nn.Linear(in_d, out_d, bias=False))

            if norm_type == 'layernorm' and i < n_layers - 1:
                layers.append(nn.LayerNorm(out_d))
            elif norm_type == 'batchnorm' and i < n_layers - 1:
                layers.append(nn.BatchNorm1d(out_d))

            if i < n_layers - 1:
                layers.append(nn.GELU())

            if use_skip and i > 0 and in_d == out_d:
                # Skip connection (identity)
                self.skip_weight = nn.Parameter(torch.zeros(1))
            else:
                self.skip_weight = None

        self.net = nn.Sequential(*layers)
        self.skip_weight = nn.Parameter(torch.zeros(1)) if use_skip else None

    def forward(self, x):
        for i, layer in enumerate(self.net):
            x = layer(x)
            if self.use_skip and hasattr(self, 'skip_weight') and i == len(self.net) // 2:
                x = x + self.skip_weight * x  # Simplified skip
        return x

    def get_weights(self):
        weights = []
        for m in self.modules():
            if isinstance(m, nn.Linear):
                weights.append(m.weight.data)
        return weights


# ──────────────────────────────────────────────────────────────────────────────
# J_topo computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_D_eff(W):
    """D_eff = ||W||_F^2 / ||W||_2^2"""
    if W.dim() > 2:
        W = W.view(W.size(0), -1)
    fro_sq = (W ** 2).sum().item()
    # Use power iteration for top singular value
    x = torch.randn(W.shape[1], device=W.device)
    for _ in range(20):
        x = W @ x
        x = W.T @ x
        x = x / (x.norm() + 1e-8)
    spec_sq = (W @ x).norm().item() ** 2 + 1e-12
    return fro_sq / spec_sq


def compute_J_topo(weights):
    """J_topo = exp(-|Σlog η_l| / L)"""
    if not weights:
        return 0.0
    eta_vals = []
    d_prev = weights[0].shape[1] if weights[0].dim() > 1 else weights[0].shape[0]
    for W in weights:
        if W.dim() > 2:
            W = W.view(W.size(0), -1)
        D_eff = compute_D_eff(W)
        eta = D_eff / max(d_prev, 1e-8)
        eta_vals.append(max(eta, 1e-8))
        d_prev = D_eff
    L = len(eta_vals)
    log_sum = sum(abs(math.log(e)) for e in eta_vals)
    return math.exp(-log_sum / L) if L > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic task
# ──────────────────────────────────────────────────────────────────────────────

def generate_synthetic_task(n_samples=5000, input_dim=64, output_dim=10, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Simple polynomial task
    X = torch.randn(n_samples, input_dim)
    # True function: y = softmax(W1 * ReLU(W2 * x))
    W1 = torch.randn(input_dim, 32)
    W2 = torch.randn(32, output_dim)
    h = torch.relu(X @ W1)
    logits = h @ W2
    y = logits.argmax(dim=1)

    # Add some noise
    logits += 0.1 * torch.randn_like(logits)
    y = logits.argmax(dim=1)

    return X, y


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_model(model, X_train, y_train, X_val, y_val, epochs=100, lr=0.01):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    best_epoch = 0
    val_losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_out = model(X_val)
            val_loss = criterion(val_out, y_val).item()
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1

    return best_val_loss, best_epoch, val_losses[-1]


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Phase S1 Fast: Cooling Theory Validation")
    print("=" * 60)

    # Configs to test (minimal set)
    configs = [
        ('None_NoSkip', 'none', False),
        ('LN_NoSkip', 'layernorm', False),
        ('BN_NoSkip', 'batchnorm', False),
        ('None_Skip', 'none', True),
        ('LN_Skip', 'layernorm', True),
        ('BN_Skip', 'batchnorm', True),
    ]

    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'epochs': 100,
        'configs': []
    }

    # Generate data once
    X_all, y_all = generate_synthetic_task(n_samples=5000, input_dim=64, output_dim=10)
    X_train, y_train = X_all[:4000], y_all[:4000]
    X_val, y_val = X_all[4000:], y_all[4000:]

    D_VALUES = [128, 256, 512]
    SEEDS = [42, 123]

    for config_name, norm_type, use_skip in configs:
        print(f"\n[{config_name}] Training...")
        config_result = {
            'name': config_name,
            'norm': norm_type,
            'skip': use_skip,
            'D_results': {}
        }

        # Compute J_topo at init
        model_init = SimpleMLP(input_dim=64, hidden_dim=128, n_layers=3,
                              norm_type=norm_type, use_skip=use_skip).to(DEVICE)
        weights = model_init.get_weights()
        J_topo = compute_J_topo(weights)
        config_result['J_topo_init'] = J_topo
        print(f"  J_topo(init) = {J_topo:.4f}")

        for D in D_VALUES:
            D_result = {'D': D, 'seeds': {}}
            losses = []

            for seed in SEEDS:
                torch.manual_seed(seed)
                np.random.seed(seed)

                model = SimpleMLP(input_dim=64, hidden_dim=D//2, n_layers=3,
                                 output_dim=10, norm_type=norm_type, use_skip=use_skip).to(DEVICE)

                best_loss, best_ep, final_loss = train_model(
                    model, X_train, y_train, X_val, y_val,
                    epochs=100, lr=0.01
                )
                losses.append(best_loss)
                D_result['seeds'][seed] = {
                    'best_val_loss': best_loss,
                    'best_epoch': best_ep
                }
                print(f"  D={D}, seed={seed}: loss={best_loss:.4f} @ epoch {best_ep}")

            D_result['avg_loss'] = np.mean(losses)
            config_result['D_results'][D] = D_result

        # Fit scaling law L(D) = α·D^(-β) + E
        Ds = list(config_result['D_results'].keys())
        losses_arr = [config_result['D_results'][D]['avg_loss'] for D in Ds]

        # Simple log-log fit for β
        log_Ds = np.log(Ds)
        log_losses = np.log(np.array(losses_arr) + 0.01)
        beta = (log_losses[-1] - log_losses[0]) / (log_Ds[-1] - log_Ds[0])
        beta = max(0.1, min(beta, 2.0))  # Bound

        # Estimate E as asymptotic (use largest D as proxy)
        E_est = losses_arr[-1]

        config_result['beta_est'] = beta
        config_result['E_est'] = E_est
        print(f"  → β ≈ {beta:.3f}, E ≈ {E_est:.4f}")

        results['configs'].append(config_result)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Find baseline (no norm, no skip)
    baseline = next(c for c in results['configs'] if c['name'] == 'None_NoSkip')
    baseline_beta = baseline['beta_est']
    baseline_E = baseline['E_est']

    print(f"\n{'Config':<15} {'J_topo':<8} {'β':<8} {'φ(β)':<8} {'E':<8}")
    print("-" * 60)
    for cfg in results['configs']:
        phi = cfg['beta_est'] / baseline_beta
        print(f"{cfg['name']:<15} {cfg['J_topo_init']:<8.4f} {cfg['beta_est']:<8.3f} "
              f"{phi:<8.3f} {cfg['E_est']:<8.4f}")

    # Analysis
    print("\n" + "=" * 60)
    print("COOLING THEORY VALIDATION")
    print("=" * 60)

    # BatchNorm effect
    bn_cfg = next(c for c in results['configs'] if c['name'] == 'BN_NoSkip')
    ln_cfg = next(c for c in results['configs'] if c['name'] == 'LN_NoSkip')
    phi_bn = bn_cfg['beta_est'] / ln_cfg['beta_est']
    print(f"\nBatchNorm cooling (BN vs LN):")
    print(f"  φ_BN = β_BN / β_LN = {bn_cfg['beta_est']:.3f} / {ln_cfg['beta_est']:.3f} = {phi_bn:.3f}")
    print(f"  Prediction: φ ≈ 0.66 (from theory)")

    # Skip effect
    skip_cfg = next(c for c in results['configs'] if c['name'] == 'LN_Skip')
    noskip_cfg = next(c for c in results['configs'] if c['name'] == 'LN_NoSkip')
    phi_skip = skip_cfg['beta_est'] / noskip_cfg['beta_est']
    print(f"\nSkip cooling (Skip vs NoSkip, LN):")
    print(f"  φ_Skip = β_Skip / β_NoSkip = {skip_cfg['beta_est']:.3f} / {noskip_cfg['beta_est']:.3f} = {phi_skip:.3f}")
    print(f"  Prediction: φ ≈ 0.93-0.98")

    # Combined effect
    bn_skip_cfg = next(c for c in results['configs'] if c['name'] == 'BN_Skip')
    phi_combined = bn_skip_cfg['beta_est'] / baseline_beta
    print(f"\nCombined (BN+Skip vs baseline):")
    print(f"  φ_Combined = {phi_combined:.3f}")
    print(f"  φ_Additivity = φ_BN × φ_Skip = {phi_bn * phi_skip:.3f}")

    # Save
    with open('phase_s1_fast_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to phase_s1_fast_results.json")

    return results


if __name__ == '__main__':
    main()
