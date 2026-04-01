#!/usr/bin/env python3
"""
ThermoRG Phase S0: Participation Ratio Theory Verification (v3)
=============================================================

KEY THEORY: d_task defined via Participation Ratio
-------------------------------------------------
Covering number theory assumes isotropy → β_worst = s/d_task
Real networks exploit spectral bias → β_eff = s/d_task_PR

where:
  d_task_PR = (Σλ_i)² / (Σλ_i²)  [participation ratio]
  λ_i = energy in i-th mode (coefficient² for RFF)

For RFF with coefficients decaying as k^(-s):
  - Isotropic (worst-case): d_task_PR = d_task = 5 → β = s/5 = 0.30
  - Anisotropic (real): d_task_PR ≈ 1.38 → β = s/1.38 ≈ 1.09

Simulation should show β_fit ≈ s/d_task_PR (not s/d_task).
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


def participation_ratio(coeffs):
    """d_task_PR = (Σλ)² / (Σλ²), λ = coefficient²."""
    lambdas = np.array(coeffs, dtype=float) ** 2
    return (np.sum(lambdas) ** 2) / (np.sum(lambdas ** 2) + 1e-12)


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
    """
    Random Fourier Feature task with SPECIFIED coefficients.
    
    Key property: coefficients k^(-s) give participation ratio << d_task.
    This captures the spectral bias of real-world data.
    """

    def __init__(self, d_manifold, d_task, s, K_max=None, seed=42):
        self.d_manifold = d_manifold
        self.d_task = d_task
        self.s = s
        self.K_max = K_max if K_max else max(2 * d_task, 20)
        self.rng = np.random.default_rng(seed)

        self.freqs = self.rng.standard_normal((self.K_max, d_manifold))
        self.freqs = self.rng.standard_normal((self.K_max, d_manifold))
        self.freqs = self.freqs / np.linalg.norm(self.freqs, axis=1, keepdims=True)
        self.phases = self.rng.uniform(0, 2 * math.pi, self.K_max)

        # Coefficients: k^(-s) decay
        ks = np.arange(1, self.K_max + 1)
        raw = ks ** (-s)
        self.coeffs = np.zeros(self.K_max, dtype=np.float32)
        self.coeffs[:d_task] = raw[:d_task]
        noise_level = raw[d_task-1] * 0.01 if d_task > 0 else raw[0] * 0.01
        self.coeffs[d_task:] = noise_level * self.rng.uniform(0.5, 1.5, self.K_max - d_task)

        # Normalize
        self.var = np.sum(self.coeffs ** 2)
        self.coeffs = self.coeffs / math.sqrt(self.var + 1e-12)

        # Compute participation ratio
        self.d_task_PR = participation_ratio(self.coeffs[:d_task])
        self.d_task_iso = float(d_task)  # if all modes equal

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

def train_to_convergence(net, X, Y, lr, max_epochs, batch_size, patience=15, device='cpu'):
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
        batch_size=batch_size, shuffle=True, drop_last=True)

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

    Ws = net.get_weight_matrices()
    J, _ = compute_J_topo_from_weights(Ws, net.d_input)

    return {
        'test_loss': best_loss,
        'train_loss': tr_loss,
        'gap': best_loss - tr_loss,
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
    beta_init, log_B_init = max(0.01, -coeffs[0]), coeffs[1]

    def obj(p):
        E, log_B, beta = p
        return np.sum((losses - (math.e ** log_B * Ds ** (-beta) + E)) ** 2)

    res = minimize(obj, x0=[E_init, log_B_init, beta_init],
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
# MAIN
# ──────────────────────────────────────────────────────

def run_verification(
    d_manifold=20,
    d_task=5,
    s=1.5,
    dataset_sizes=None,
    architectures=None,
    max_epochs=100,
    lr=5e-4,
    batch_size=64,
    seed=42,
    output_dir="experiments/phase_s0/v3_results",
):
    if dataset_sizes is None:
        dataset_sizes = [200, 400, 800, 1600]
    if architectures is None:
        architectures = [
            {'name': 'small_L1',  'd_hidden': 8,  'n_layers': 1, 'init_scale': 1.0},
            {'name': 'medium_L2', 'd_hidden': 20, 'n_layers': 2, 'init_scale': 1.0},
            {'name': 'large_L3',  'd_hidden': 40, 'n_layers': 3, 'init_scale': 0.5},
        ]

    print("=" * 65)
    print("ThermoRG S0 v3: Participation Ratio Theory Verification")
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 65)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Compute theory predictions ────────────────────────
    sphere = HypersphereSampler()
    task = RFFTask(d_manifold, d_task, s, seed=seed)

    beta_worst = s / task.d_task_iso        # isotropic (covering number bound)
    beta_eff   = s / task.d_task_PR         # anisotropic (participation ratio)

    print(f"\nTask: RFF with k^(-{s}) coefficient decay")
    print(f"  d_manifold={d_manifold}, d_task={d_task}")
    print(f"  Participation Ratio: d_task_PR = {task.d_task_PR:.2f}")
    print()
    print(f"  β_worst_case (isotropic, s/d_task):   s/{task.d_task_iso:.0f} = {beta_worst:.4f}")
    print(f"  β_effective  (anisotropic, s/d_task_PR): s/{task.d_task_PR:.2f} = {beta_eff:.4f}")

    # Generate data
    X_pool = sphere.sample(max(dataset_sizes), d_manifold, np.random.default_rng(seed))
    Y_pool = task(X_pool, noise_std=0.1).astype(np.float32)
    print(f"\n  Data: {X_pool.shape}, Y std={Y_pool.std():.3f}")

    # ── D-scaling for each architecture ────────────────
    print(f"\nDataset sizes: {dataset_sizes}")
    print(f"Architectures: {[a['name'] for a in architectures]}")

    results = []
    for cfg in architectures:
        name = cfg['name']
        print(f"\n  [{name}] J_topo init...", end=" ", flush=True)
        net_survey = FCNet(d_manifold, cfg['d_hidden'], 1, cfg['n_layers'], cfg['init_scale'])
        Ws0 = net_survey.get_weight_matrices()
        J0, _ = compute_J_topo_from_weights(Ws0, d_manifold)
        print(f"{J0:.3f}")

        d_results = []
        for D in dataset_sizes:
            idx = np.random.default_rng(seed + D).permutation(len(X_pool))[:D]
            net = FCNet(d_manifold, cfg['d_hidden'], 1, cfg['n_layers'], cfg['init_scale'])
            m = train_to_convergence(net, X_pool[idx], Y_pool[idx], lr, max_epochs, batch_size)
            d_results.append({'D': D, **m})
            print(f"    D={D:5d}: loss={m['test_loss']:.4f}, J={m['J_topo']:.3f}, ep={m['epochs']}")

        losses = [r['test_loss'] for r in d_results]
        fit = fit_power_law([r['D'] for r in d_results], losses)

        results.append({
            'name': name,
            'config': cfg,
            'J_topo_init': J0,
            'J_topo_final': d_results[-1]['J_topo'],
            'beta_fit': fit.get('beta'),
            'beta_worst': beta_worst,
            'beta_eff': beta_eff,
            'd_task_iso': task.d_task_iso,
            'd_task_PR': task.d_task_PR,
            'R2': fit.get('R2'),
            'd_scaling': d_results,
        })
        print(f"    Fit: β={fit.get('beta', 0):.4f}, R²={fit.get('R2', 0):.3f}")

    # ── Save ──────────────────────────────────────────
    out = {
        'timestamp': datetime.now().isoformat(),
        'task': {
            'd_manifold': d_manifold, 'd_task': d_task, 's': s,
            'd_task_iso': task.d_task_iso,
            'd_task_PR': task.d_task_PR,
            'beta_worst': beta_worst,
            'beta_eff': beta_eff,
        },
        'dataset_sizes': dataset_sizes,
        'results': [{
            'name': r['name'],
            'J_topo_init': r['J_topo_init'],
            'J_topo_final': r['J_topo_final'],
            'beta_fit': r['beta_fit'],
            'beta_worst': r['beta_worst'],
            'beta_eff': r['beta_eff'],
            'R2': r['R2'],
            'd_task_PR': r['d_task_PR'],
        } for r in results],
    }
    with open(str(output_dir / 'v3_results.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    # ── Summary ───────────────────────────────────────
    print("\n" + "=" * 65)
    print("VERIFICATION SUMMARY")
    print("=" * 65)
    print(f"\nTheory predictions:")
    print(f"  β_worst (isotropic):  {beta_worst:.4f}")
    print(f"  β_eff   (anisotropic): {beta_eff:.4f}")
    print()
    print(f"{'Arch':20s} {'J_topo':>6s} {'β_fit':>8s} {'R^2':>6s}  vs β_eff={beta_eff:.4f}")
    print("-" * 60)
    for r in results:
        bf = r['beta_fit']
        print(f"  {r['name']:18s} {r['J_topo_init']:6.3f} {bf if bf else 0:8.4f} "
              f"{r['R2'] if r['R2'] else 0:6.3f}  "
              f"{'✓ CLOSER to β_eff' if bf and abs(bf - beta_eff) < abs(bf - beta_worst) else 'closer to worst-case'}")

    print(f"\nResults: {output_dir / 'v3_results.json'}")
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--d_manifold', type=int, default=20)
    p.add_argument('--d_task', type=int, default=5)
    p.add_argument('--s', type=float, default=1.5)
    p.add_argument('--max_epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output_dir', default='experiments/phase_s0/v3_results')
    run_verification(**vars(p.parse_args()))
