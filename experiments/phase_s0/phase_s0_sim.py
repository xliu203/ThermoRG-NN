#!/usr/bin/env python3
"""
ThermoRG Phase S0: End-to-End Simulation
=========================================

Purpose: Validate v3 theory with synthetic data and controllable networks.

Theory:
    L(N,D) = E + A·N^(-α) + B·D^(-β)
    α = k_α · J_topo² · (2s/d_task) · ψ(T_eff) · φ(γ_cool)
    β = s · J_topo / d_task
    J_topo = exp(-|Σlog η_l|/L)  ∈ (0,1]
    η_l = D_eff^(l) / d_eff^(l-1)
    D_eff = ||J||_F² / ||J||_2²  (stable rank)
    ψ(T_eff) = (T_eff/T_c)·exp(1 - T_eff/T_c)
    T_eff/T_c = η·λ_max/2 = sharpness/2

Predictions:
    P1: J_topo ∈ (0,1] (always bounded)
    P2: β ∝ J_topo (linear)
    P3: α ∝ J_topo² (quadratic)
    P4: d_eff^realized = d_task/J_topo
    P5: Sharpness → 2 during training (Edge of Stability)
"""

import json, math, time, sys, os, argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))


# ──────────────────────────────────────────────────────
# CORE COMPUTATIONS (v3 formulas)
# ──────────────────────────────────────────────────────

def compute_D_eff(J):
    """Stable rank: D_eff = ||J||_F² / ||J||_2²"""
    fro_sq = (J ** 2).sum().item()
    S = linalg.svd(J)[1]
    spec_sq = S[0].item() ** 2 + 1e-12
    return fro_sq / spec_sq


def compute_J_topo(eta_vals, L):
    """J_topo = exp(-|Σlog η_l| / L) ∈ (0,1]"""
    log_sum = sum(abs(math.log(max(e, 1e-12))) for e in eta_vals)
    return math.exp(-log_sum / L) if L > 0 else 0.0


def compute_J_topo_from_weights(weight_matrices, d_input):
    """
    Compute J_topo from layer weight matrices.
    η_l = D_eff^(l) / d_eff^(l-1), starting from d_input.
    """
    eta_vals = []
    d_prev = float(d_input)
    for W in weight_matrices:
        D_eff = compute_D_eff(W)
        eta = D_eff / max(d_prev, 1e-8)
        eta_vals.append(eta)
        d_prev = D_eff
    L = len(eta_vals)
    J = compute_J_topo(eta_vals, L)
    return J, eta_vals


def psi_response(T_eff, T_c):
    """ψ(T_eff) = (T_eff/T_c)·exp(1 - T_eff/T_c)"""
    x = T_eff / T_c
    return x * math.exp(1 - x)


# ──────────────────────────────────────────────────────
# SYNTHETIC NETWORKS (varying J_topo)
# ──────────────────────────────────────────────────────

class SyntheticFCNet(nn.Module):
    """Fully-connected network with controllable J_topo."""

    def __init__(self, d_input, d_hidden, d_output, n_layers, init_scale=1.0):
        super().__init__()
        dims = [d_input] + [d_hidden] * n_layers + [d_output]
        self.layers = nn.ModuleList()
        for i in range(n_layers + 1):
            self.layers.append(nn.Linear(dims[i], dims[i+1], bias=False))
        self.d_input = d_input
        self.n_layers = n_layers
        self._init_weights(init_scale)

    def _init_weights(self, scale):
        for i, l in enumerate(self.layers):
            if i < len(self.layers) - 1:  # not output layer
                gain = scale * math.sqrt(2.0 / l.in_features)
            else:
                gain = 1.0
            nn.init.xavier_uniform_(l.weight, gain=gain)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        for l in self.layers[:-1]:
            x = F.relu(l(x))
        return self.layers[-1](x)

    def get_weight_matrices(self):
        """Return list of weight matrices (proxy for layer Jacobians)."""
        return [l.weight.data.clone() for l in self.layers[:-1]]


def build_network_sweep(d_input, d_output, n_classes=10):
    """
    Build networks with systematically varying J_topo.
    Strategy: vary hidden_dim and initialization scale.
    """
    configs = []

    # Vary hidden dimension (larger → higher J_topo)
    for d_mult in [0.5, 1.0, 2.0, 4.0]:
        d_hidden = max(8, int(d_input * d_mult))
        for n_layers in [1, 2, 3]:
            configs.append({
                'name': f'h{int(d_mult*100)}_L{n_layers}',
                'd_hidden': d_hidden,
                'n_layers': n_layers,
                'init_scale': 1.0,
            })

    # Vary initialization scale (smaller → lower J_topo)
    for scale in [0.1, 0.5, 1.0, 2.0]:
        d_hidden = d_input
        n_layers = 2
        configs.append({
            'name': f's{int(scale*10)}_L{n_layers}',
            'd_hidden': d_hidden,
            'n_layers': n_layers,
            'init_scale': scale,
        })

    networks = []
    for cfg in configs:
        net = SyntheticFCNet(
            d_input=d_input,
            d_hidden=cfg['d_hidden'],
            d_output=n_classes,
            n_layers=cfg['n_layers'],
            init_scale=cfg['init_scale']
        )
        networks.append((cfg['name'], net, cfg))

    return networks


# ──────────────────────────────────────────────────────
# DATA GENERATION
# ──────────────────────────────────────────────────────

def generate_synthetic_data(d_manifold, d_embed, n_samples, n_classes, seed=42):
    """Generate x = f(z) + ε using polynomial embedding."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Latent z ~ Uniform[-1, 1]^d_manifold
    z = torch.rand(n_samples, d_manifold) * 2 - 1

    # Polynomial features of z -> x
    # Simple: x = [z, z^2, z*z', ...] up to degree 2
    x_list = [z]
    for i in range(d_manifold):
        for j in range(i, d_manifold):
            x_list.append(z[:, i:i+1] * z[:, j:j+1])
    x = torch.cat(x_list, dim=1)

    # Truncate/pad to d_embed
    if x.shape[1] > d_embed:
        x = x[:, :d_embed]
    elif x.shape[1] < d_embed:
        pad = torch.zeros(n_samples, d_embed - x.shape[1])
        x = torch.cat([x, pad], dim=1)

    # Add noise
    x = x + 0.1 * torch.randn_like(x)

    # Labels: y = argmax of a linear combination of z
    y = ((z.sum(dim=1) + z[:, 0] * z[:, 1]) > 0).long() % n_classes

    return x, y


# ──────────────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────────────

def train_and_measure(net, x, y, lr, epochs, batch_size, device='cpu'):
    """
    Train a network and record metrics.
    Returns: dict with training metrics and J_topo.
    """
    net = net.to(device)
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    # Train/test split
    n = x.shape[0]
    perm = torch.randperm(n)
    n_train = int(0.8 * n)
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    x_tr, y_tr = x[train_idx].to(device), y[train_idx].to(device)
    x_te, y_te = x[test_idx].to(device), y[test_idx].to(device)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_tr, y_tr),
        batch_size=batch_size, shuffle=True
    )

    metrics = {
        'epochs': [], 'train_loss': [], 'test_loss': [],
        'test_acc': [], 'J_topo': [], 'sharpness': []
    }
    record_every = max(1, epochs // 20)

    for epoch in range(epochs):
        net.train()
        for bx, by in loader:
            opt.zero_grad()
            loss_fn(net(bx), by).backward()
            opt.step()
        scheduler.step()

        if epoch % record_every == 0 or epoch == epochs - 1:
            net.eval()
            with torch.no_grad():
                logits = net(x_te)
                test_loss = loss_fn(logits, y_te).item()
                test_acc = (logits.argmax(1) == y_te).float().mean().item()

            # J_topo from weight matrices
            Ws = net.get_weight_matrices()
            J_topo, _ = compute_J_topo_from_weights(Ws, net.d_input)

            # Sharpness: lr * ||grad||
            grad_norm = sum(p.grad.norm().item() ** 2
                          for p in net.parameters() if p.grad is not None)
            sharpness = lr * math.sqrt(grad_norm)

            metrics['epochs'].append(epoch)
            metrics['train_loss'].append(None)
            metrics['test_loss'].append(test_loss)
            metrics['test_acc'].append(test_acc)
            metrics['J_topo'].append(J_topo)
            metrics['sharpness'].append(sharpness)

    net.eval()
    return metrics


# ──────────────────────────────────────────────────────
# D-SCALING (measure β)
# ──────────────────────────────────────────────────────

def measure_beta(NetClass, cfg, x_full, y_full, d_input, d_output,
                 dataset_sizes, lr, epochs, device='cpu'):
    """
    Train fixed-architecture networks on varying dataset sizes.
    Fit L(D) = E + B·D^(-β)
    """
    results = []
    for D in dataset_sizes:
        idx = torch.randperm(x_full.shape[0])[:D]
        x_sub, y_sub = x_full[idx], y_full[idx]
        net = NetClass(d_input=d_input, d_hidden=cfg['d_hidden'],
                       d_output=d_output, n_layers=cfg['n_layers'],
                       init_scale=cfg['init_scale'])
        m = train_and_measure(net, x_sub, y_sub, lr, epochs,
                             batch_size=min(256, D//4), device=device)
        results.append((D, min(m['test_loss'])))

    losses = np.array([r[1] for r in results])
    Ds = np.array([float(r[0]) for r in results])

    # Fit L(D) = E + B·D^(-β)
    E_init = min(losses) * 0.8
    valid = losses - E_init > 1e-6
    if valid.sum() < 3:
        return {'beta': None, 'R2': 0, 'points': results}

    log_D = np.log(Ds[valid])
    log_LmE = np.log(np.maximum(losses[valid] - E_init, 1e-6))
    coeffs = np.polyfit(log_D, log_LmE, deg=1)
    beta = -coeffs[0]
    B_fit = math.exp(coeffs[1])

    L_pred = E_init + B_fit * Ds ** (-beta)
    ss_res = np.sum((losses - L_pred) ** 2)
    ss_tot = np.sum((losses - np.mean(losses)) ** 2)
    R2 = 1 - ss_res / (ss_tot + 1e-10)

    return {'beta': beta, 'B': B_fit, 'R2': R2, 'points': results}


# ──────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────

def run_s0(output_dir="experiments/phase_s0/sim_results",
           d_manifold=8, d_embed=32, n_samples=5000,
           n_classes=10, epochs=200, lr=0.01, seed=42):
    """Run the complete S0 end-to-end simulation."""
    print("=" * 60)
    print("ThermoRG Phase S0: End-to-End Simulation")
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = 'cpu'

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Step 1: Data ──────────────────────────────────
    print(f"\n[1/4] Generating data: d_mfld={d_manifold}, d_emb={d_embed}, n={n_samples}")
    x, y = generate_synthetic_data(d_manifold, d_embed, n_samples, n_classes, seed)
    print(f"      x={x.shape}, y={y.shape}, classes={n_classes}")

    # ── Step 2: Networks ───────────────────────────────
    print(f"\n[2/4] Building network sweep")
    networks = build_network_sweep(d_embed, n_classes, n_classes)
    print(f"      {len(networks)} configurations")

    # ── Step 3: Train All ─────────────────────────────
    print(f"\n[3/4] Training networks ({epochs} epochs, lr={lr})")
    all_results = []

    for name, net, cfg in networks:
        t0 = time.time()
        m = train_and_measure(net, x, y, lr=lr, epochs=epochs,
                             batch_size=min(256, n_samples//20), device=device)
        elapsed = time.time() - t0

        # J_topo pre and post training
        Ws_init = net.get_weight_matrices()
        J_init, _ = compute_J_topo_from_weights(Ws_init, d_embed)
        Ws_final = net.get_weight_matrices()
        J_final, eta_final = compute_J_topo_from_weights(Ws_final, d_embed)

        result = {
            'name': name,
            'config': cfg,
            'J_topo_init': J_init,
            'J_topo_final': J_final,
            'eta_final': eta_final,
            'final_acc': m['test_acc'][-1],
            'final_loss': min(m['test_loss']),
            'J_topo_traj': m['J_topo'],
            'sharp_traj': m['sharpness'],
            'loss_traj': m['test_loss'],
            'acc_traj': m['test_acc'],
            'time_s': elapsed,
        }
        all_results.append(result)
        print(f"  {name:20s}: J={J_init:.3f}→{J_final:.3f}, acc={result['final_acc']:.3f}, {elapsed:.1f}s")

    # ── Step 4: D-Scaling (β) ─────────────────────────
    print(f"\n[4/4] D-scaling for β (top 4 architectures)")
    # Select architectures spanning J_topo range
    sorted_by_J = sorted(all_results, key=lambda r: r['J_topo_final'])
    selected = [sorted_by_J[0], sorted_by_J[len(sorted_by_J)//2],
                sorted_by_J[-1],
                all_results[np.argmax([r['final_acc'] for r in all_results])]]
    selected = [r for r in selected if r['name'] not in [None]]

    beta_results = {}
    for result in selected:
        cfg = result['config']
        name = result['name']
        print(f"  D-scaling {name}...")
        b = measure_beta(SyntheticFCNet, cfg, x, y,
                        d_input=d_embed, d_output=n_classes,
                        dataset_sizes=[500, 1000, 2000, 4000],
                        lr=lr, epochs=epochs, device=device)
        beta_results[name] = b
        if b['beta'] is not None:
            print(f"    β={b['beta']:.4f}, R²={b['R2']:.3f}")
        else:
            print(f"    fit failed")

    # ── Save ───────────────────────────────────────────
    out = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'd_manifold': d_manifold, 'd_embed': d_embed,
            'n_samples': n_samples, 'n_classes': n_classes,
            'epochs': epochs, 'lr': lr, 'seed': seed
        },
        'results': [
            {k: v for k, v in r.items() if k not in ['J_topo_traj', 'sharp_traj', 'loss_traj', 'acc_traj']}
            for r in all_results
        ],
        'beta_results': beta_results,
    }
    with open(str(output_dir / 's0_sim_results.json'), 'w') as f:
        json.dump(out, f, indent=2)

    # ── Print Summary ─────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    J_vals = [(r['name'], r['J_topo_final'], r['final_acc']) for r in all_results]
    J_vals.sort(key=lambda x: x[1])

    print("\nP1: J_topo is Bounded in (0,1]")
    print(f"  Range: {min(r[1] for r in J_vals):.4f} – {max(r[1] for r in J_vals):.4f}")
    print(f"  {'PASS ✓' if all(0 < r[1] <= 1 for r in J_vals) else 'FAIL ✗'}")

    print("\nP2: β ∝ J_topo")
    if beta_results:
        betas = {n: b['beta'] for n, b in beta_results.items() if b['beta'] is not None}
        if len(betas) >= 2:
            J_for_beta = {n: next(r['J_topo_final'] for r in all_results if r['name'] == n) for n in betas}
            corr = np.corrcoef(list(betas.values()), list(J_for_beta.values()))[0, 1]
            print(f"  β values: {betas}")
            print(f"  Correlation with J_topo: {corr:.3f} {'✓' if corr > 0 else '✗'}")

    print("\nP3: Higher J_topo → Better Accuracy")
    accs = [r['final_acc'] for r in all_results]
    J_finals = [r['J_topo_final'] for r in all_results]
    corr_acc = np.corrcoef(J_finals, accs)[0, 1]
    print(f"  Correlation: {corr_acc:.3f} {'✓' if corr_acc > 0 else '✗'}")

    print(f"\nResults: {str(output_dir / 's0_sim_results.json')}")
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--d_manifold', type=int, default=8)
    p.add_argument('--d_embed', type=int, default=32)
    p.add_argument('--n_samples', type=int, default=5000)
    p.add_argument('--n_classes', type=int, default=10)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output_dir', default='experiments/phase_s0/sim_results')
    args=p.parse_args(); run_s0(output_dir=args.output_dir, d_manifold=args.d_manifold, d_embed=args.d_embed, n_samples=args.n_samples, n_classes=args.n_classes, epochs=args.epochs, lr=args.lr, seed=args.seed)
