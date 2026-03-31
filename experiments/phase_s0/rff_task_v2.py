#!/usr/bin/env python3
"""
ThermoRG Phase S0: Fixed D-Scaling Simulation (v2)
=================================================

Key insight: β = s / d_task where d_task is the EFFECTIVE number of
learnable degrees of freedom, NOT the number of Fourier modes.

The network learns ALL modes (20) to some degree, not just d_task=5.
So the "effective d_task" for scaling purposes is larger.

New approach: Fit d_task from the data, then check if it matches
the number of "large" Fourier coefficients.
"""

import json, math, sys, time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg
from scipy.optimize import minimize

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))


# ──────────────────────────────────────────────────────
# CORE
# ──────────────────────────────────────────────────────

def compute_D_eff(J):
    fro_sq = (J ** 2).sum().item()
    S = linalg.svd(J)[1]
    spec_sq = S[0].item() ** 2 + 1e-12
    return fro_sq / spec_sq


def compute_J_topo_from_weights(weight_matrices, d_input):
    eta_vals = []
    d_prev = float(d_input)
    for W in weight_matrices:
        D = compute_D_eff(W)
        eta_vals.append(D / max(d_prev, 1e-8))
        d_prev = D
    L = len(eta_vals)
    log_sum = sum(abs(math.log(max(e, 1e-12))) for e in eta_vals)
    J = math.exp(-log_sum / L) if L > 0 else 0.0
    return J, eta_vals


# ──────────────────────────────────────────────────────
# RFF TASK
# ──────────────────────────────────────────────────────

class HypersphereSampler:
    def sample(self, n, d, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        x = rng.standard_normal((n, d)).astype(np.float32)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / norms


class RFFTask:
    """Random Fourier Feature task with SPECIFIED coefficients."""

    def __init__(self, d_manifold, d_task, s, K_max=None, seed=42):
        self.d_manifold = d_manifold
        self.d_task = d_task
        self.s = s
        self.K_max = K_max if K_max is not None else max(2 * d_task, 20)
        self.rng = np.random.default_rng(seed)

        # Frequency directions on sphere
        self.freqs = self.rng.standard_normal((self.K_max, d_manifold))
        self.freqs = self.freqs / np.linalg.norm(self.freqs, axis=1, keepdims=True)
        self.phases = self.rng.uniform(0, 2 * math.pi, self.K_max)

        # Coefficients: first d_task are strong, rest are weak
        ks = np.arange(1, self.K_max + 1)
        raw = ks ** (-s)
        self.coeffs = np.zeros(self.K_max, dtype=np.float32)
        self.coeffs[:d_task] = raw[:d_task]
        # Noise modes: 1% of smallest d_task coefficient  
        noise_level = raw[d_task-1] * 0.01 if d_task > 0 else raw[0] * 0.01
        self.coeffs[d_task:] = noise_level * self.rng.uniform(0.5, 1.5, self.K_max - d_task)

        # Normalize
        self.var = np.sum(self.coeffs ** 2)
        self.coeffs = self.coeffs / math.sqrt(self.var + 1e-12)

    def evaluate(self, X):
        projections = X @ self.freqs.T
        angles = projections + self.phases
        features = np.cos(angles)
        return features @ self.coeffs

    def __call__(self, X, noise_std=0.1):
        y = self.evaluate(X)
        noise = self.rng.normal(0, noise_std, size=y.shape).astype(np.float32)
        return y + noise


# ──────────────────────────────────────────────────────
# NETWORK
# ──────────────────────────────────────────────────────

class FCNet(nn.Module):
    def __init__(self, d_input, d_hidden, d_output, n_layers, init_scale=1.0):
        super().__init__()
        dims = [d_input] + [d_hidden] * n_layers + [d_output]
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
        self.d_input = d_input
        self._init(init_scale)

    def _init(self, scale):
        for i, l in enumerate(self.layers):
            gain = scale * math.sqrt(2.0 / l.in_features) if i < len(self.layers) - 1 else 1.0
            nn.init.xavier_uniform_(l.weight, gain=gain)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        for l in self.layers[:-1]:
            x = F.relu(l(x))
        return self.layers[-1](x)

    def get_weight_matrices(self):
        return [l.weight.data.clone() for l in self.layers[:-1]]


# ──────────────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────────────

def train_to_convergence(net, X, Y, lr, max_epochs, batch_size, patience=20, device='cpu'):
    """Train until convergence or max_epochs."""
    net = net.to(device)
    X_t = torch.from_numpy(X.astype(np.float32)).to(device)
    Y_t = torch.from_numpy(Y.flatten().astype(np.float32)).to(device)

    n = X.shape[0]
    perm = torch.randperm(n)
    n_tr = int(0.8 * n)
    tr_idx, te_idx = perm[:n_tr], perm[n_tr:]
    X_tr, Y_tr = X_t[tr_idx], Y_t[tr_idx]
    X_te, Y_te = X_t[te_idx], Y_t[te_idx]

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tr, Y_tr),
        batch_size=batch_size, shuffle=True, drop_last=True
    )
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    loss_fn = nn.MSELoss()

    best_loss = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(max_epochs):
        net.train()
        for bx, by in loader:
            opt.zero_grad()
            loss_fn(net(bx), by.unsqueeze(-1)).backward()
            opt.step()
        scheduler.step()

        net.eval()
        with torch.no_grad():
            tl = loss_fn(net(X_te).squeeze(), Y_te).item()
        if tl < best_loss:
            best_loss = tl
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    if best_state:
        net.load_state_dict(best_state)

    net.eval()
    with torch.no_grad():
        tr_loss = loss_fn(net(X_tr).squeeze(), Y_tr).item()
        te_loss = loss_fn(net(X_te).squeeze(), Y_te).item()
        gap = te_loss - tr_loss

    Ws = net.get_weight_matrices()
    J, _ = compute_J_topo_from_weights(Ws, net.d_input)

    return {
        'test_loss': best_loss,
        'train_loss': tr_loss,
        'gap': gap,
        'J_topo': J,
        'epochs': epoch + 1,
    }


# ──────────────────────────────────────────────────────
# POWER LAW FIT
# ──────────────────────────────────────────────────────

def fit_power_law(Ds, losses):
    Ds, losses = np.array(Ds, float), np.array(losses, float)
    if len(Ds) < 3:
        return {'beta': None, 'R2': None}

    E_init = float(min(losses) * 0.9)
    valid = losses - E_init > 1e-6
    if valid.sum() < 2:
        return {'beta': None, 'R2': None}

    log_D = np.log(Ds[valid])
    log_LmE = np.log(np.maximum(losses[valid] - E_init, 1e-6))
    coeffs = np.polyfit(log_D, log_LmE, deg=1)
    beta_init = max(0.01, -coeffs[0])
    log_B_init = coeffs[1]

    def obj(p):
        E, log_B, beta = p
        pred = math.e ** log_B * Ds ** (-beta) + E
        return np.sum((losses - pred) ** 2)

    res = minimize(obj,
        x0=[E_init, log_B_init, beta_init],
        bounds=[(1e-6, max(losses)), (-10, 10), (0.001, 10)],
        method='L-BFGS-B')
    E, log_B, beta = res.x
    B = math.e ** log_B
    pred = E + B * Ds ** (-beta)
    ss_r = np.sum((losses - pred) ** 2)
    ss_t = np.sum((losses - np.mean(losses)) ** 2)
    R2 = 1 - ss_r / (ss_t + 1e-10)

    return {'beta': beta, 'B': B, 'E': E, 'R2': R2,
            'data': list(zip(Ds.tolist(), losses.tolist()))}


# ──────────────────────────────────────────────────────
# MAIN: Compare theory β vs fitted β
# ──────────────────────────────────────────────────────

def run_comparison(
    d_manifold=20,
    d_task_true=5,
    s=1.5,
    dataset_sizes=None,
    architectures=None,
    max_epochs=100,
    lr=5e-4,
    batch_size=64,
    seed=42,
    output_dir="experiments/phase_s0/v2_results",
):
    if dataset_sizes is None:
        dataset_sizes = [200, 400, 800, 1600]
    if architectures is None:
        architectures = [
            {'name': 'small_L1', 'd_hidden': 8, 'n_layers': 1, 'init_scale': 1.0},
            {'name': 'medium_L2', 'd_hidden': 20, 'n_layers': 2, 'init_scale': 1.0},
            {'name': 'large_L3', 'd_hidden': 40, 'n_layers': 3, 'init_scale': 0.5},
        ]

    print("=" * 60)
    print("ThermoRG S0 v2: Theory vs Fitted β Comparison")
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    beta_theory = s / d_task_true
    print(f"\nTask: d_manifold={d_manifold}, d_task_true={d_task_true}, s={s}")
    print(f"Theory: β = s/d_task = {s}/{d_task_true} = {beta_theory:.4f}")
    print(f"Dataset sizes: {dataset_sizes}")

    # Generate large pool
    sphere = HypersphereSampler()
    X_pool = sphere.sample(max(dataset_sizes), d_manifold, np.random.default_rng(seed))
    task = RFFTask(d_manifold, d_task_true, s, seed=seed)
    Y_pool = task(X_pool, noise_std=0.1).astype(np.float32)
    print(f"Data: {X_pool.shape}, Y std={Y_pool.std():.3f}")

    all_results = []

    for cfg in architectures:
        name = cfg['name']
        print(f"\n  Architecture: {name}")
        t0 = time.time()

        # First: full-data run to get J_topo
        net_full = FCNet(d_manifold, cfg['d_hidden'], 1, cfg['n_layers'], cfg['init_scale'])
        m_full = train_to_convergence(net_full, X_pool, Y_pool, lr, max_epochs, batch_size)
        J = m_full['J_topo']
        print(f"    J_topo={J:.3f}, test_loss={m_full['test_loss']:.4f}, epochs={m_full['epochs']}")

        # D-scaling
        d_results = []
        for D in dataset_sizes:
            idx = np.random.default_rng(seed + D).permutation(len(X_pool))[:D]
            X_D, Y_D = X_pool[idx], Y_pool[idx]
            net_D = FCNet(d_manifold, cfg['d_hidden'], 1, cfg['n_layers'], cfg['init_scale'])
            m_D = train_to_convergence(net_D, X_D, Y_D, lr, max_epochs, batch_size)
            d_results.append({'D': D, **m_D})
            print(f"    D={D:5d}: loss={m_D['test_loss']:.4f}, gap={m_D['gap']:.4f}")

        losses = [r['test_loss'] for r in d_results]
        fit = fit_power_law([r['D'] for r in d_results], losses)

        # Invert: from fitted β, compute effective d_task
        if fit['beta'] and fit['beta'] > 0:
            d_task_fitted = s / fit['beta']
        else:
            d_task_fitted = None

        result = {
            'name': name,
            'config': cfg,
            'J_topo': J,
            'd_task_true': d_task_true,
            'd_task_fitted': d_task_fitted,
            'beta_theory': beta_theory,
            'beta_fitted': fit.get('beta'),
            'R2': fit.get('R2'),
            'd_scaling': d_results,
        }
        all_results.append(result)

        print(f"    Fit: β={fit.get('beta', 'N/A'):.4f}, R²={fit.get('R2', 0):.3f}")
        if d_task_fitted:
            print(f"    Fitted d_task = s/β = {s}/{fit['beta']:.4f} = {d_task_fitted:.2f}")
            print(f"    True d_task = {d_task_true}, Ratio = {d_task_fitted/d_task_true:.2f}x")

        elapsed = time.time() - t0
        print(f"    Time: {elapsed:.0f}s")

    # Save
    out = {
        'timestamp': datetime.now().isoformat(),
        'task_params': {'d_manifold': d_manifold, 'd_task_true': d_task_true,
                       's': s, 'beta_theory': beta_theory},
        'dataset_sizes': dataset_sizes,
        'training': {'max_epochs': max_epochs, 'lr': lr, 'batch_size': batch_size},
        'results': [{
            'name': r['name'],
            'J_topo': r['J_topo'],
            'beta_theory': r['beta_theory'],
            'beta_fitted': r['beta_fitted'],
            'd_task_true': r['d_task_true'],
            'd_task_fitted': r['d_task_fitted'],
            'R2': r['R2'],
        } for r in all_results],
    }
    with open(str(output_dir / 'v2_results.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Theory: β = s/d_task = {s}/{d_task_true} = {beta_theory:.4f}")
    print()
    print(f"{'Name':20s} {'J_topo':>6s} {'β_theory':>8s} {'β_fit':>8s} {'R^2':>6s} {'d_task_fit':>10s}")
    print("-" * 70)
    for r in all_results:
        bf = r['beta_fitted']
        rf = r['R2']
        dtf = r['d_task_fitted']
        print(f"  {r['name']:18s} {r['J_topo']:6.3f} {r['beta_theory']:8.4f} "
              f"{bf if bf else 'N/A':>8.4f} {rf if rf else 0:6.3f} "
              f"{dtf if dtf else 'N/A':>10.2f}")

    print(f"\nResults: {output_dir / 'v2_results.json'}")
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--d_manifold', type=int, default=20)
    p.add_argument('--d_task_true', type=int, default=5)
    p.add_argument('--s', type=float, default=1.5)
    p.add_argument('--max_epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output_dir', default='experiments/phase_s0/v2_results')
    run_comparison(**vars(p.parse_args()))
