#!/usr/bin/env python3
"""
ThermoRG Phase S0: RFF-based Synthetic Task
============================================

Teacher-Student Framework with Random Fourier Features.

Theory: L(D) = E + B·D^(-β),  β = s / d_task

Key fixes from analysis:
- Harder task: d_task=18, s=0.5 → β=0.028 (MUCH harder)
- All D get SAME epochs (500) — no reduction for small D
- Consistent train/test split across all D (no resampling per D)
- Lower LR for stability at large D
- Larger minibatch for better gradients
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


class RandomFourierFeatureFunction:
    """
    Y = f*(X) + noise, f* is a random Fourier series on the hypersphere.
    Coefficients decay as k^(-s). First d_task modes are informative.

    With d_task=18, d_manifold=20, s=0.5:
    - 18 informative modes (k=1..18) with coefficients k^(-0.5)
    - 2 non-informative modes (k=19,20) with tiny coefficients
    - β = s/d_task = 0.5/18 ≈ 0.028
    """

    def __init__(self, d_manifold, d_task, s, K_max=None, seed=42):
        self.d_manifold = d_manifold
        self.d_task = d_task
        self.s = s
        self.K_max = K_max or max(2 * d_task, 20)
        self.rng = np.random.default_rng(seed)

        self.freqs = self.rng.standard_normal((self.K_max, self.d_manifold))
        self.freqs = self.freqs / np.linalg.norm(self.freqs, axis=1, keepdims=True)
        self.phases = self.rng.uniform(0, 2 * math.pi, self.K_max)

        ks = np.arange(1, self.K_max + 1)
        raw = ks ** (-s)
        self.coeffs = np.zeros(self.K_max, dtype=np.float32)
        self.coeffs[:d_task] = raw[:d_task]
        # Non-informative modes: very small random coefficients
        self.coeffs[d_task:] = raw[d_task:] * 0.01 * self.rng.uniform(0.5, 1.5, self.K_max - d_task)
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
# STUDENT NETWORK
# ──────────────────────────────────────────────────────

class FCStudentNet(nn.Module):
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
            if i < len(self.layers) - 1:
                nn.init.xavier_uniform_(l.weight, gain=scale * math.sqrt(2.0 / l.in_features))
            else:
                nn.init.xavier_uniform_(l.weight, gain=1.0)

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

def train_network(net, X, Y, lr, epochs, batch_size, device='cpu',
                  X_te=None, Y_te=None, verbose=False):
    """
    Train network with fixed train/test split.
    If X_te/Y_te provided, use them as the test set.
    """
    net = net.to(device)
    X_t = torch.from_numpy(X.astype(np.float32)).to(device)
    Y_t = torch.from_numpy(Y.flatten().astype(np.float32)).to(device)

    n = X.shape[0]
    if X_te is None:
        perm = torch.randperm(n)
        n_train = int(0.8 * n)
        tr_idx, te_idx = perm[:n_train], perm[n_train:]
        X_tr, Y_tr = X_t[tr_idx], Y_t[tr_idx]
        X_te, Y_te = X_t[te_idx], Y_t[te_idx]
    else:
        X_te_t = torch.from_numpy(X_te.astype(np.float32)).to(device)
        Y_te_t = torch.from_numpy(Y_te.flatten().astype(np.float32)).to(device)
        X_tr, Y_tr = X_t, Y_t
        X_te, Y_te = X_te_t, Y_te_t

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tr, Y_tr),
        batch_size=batch_size, shuffle=True, drop_last=True
    )
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()

    metrics = {'test_loss': [], 'test_r2': [], 'train_loss': [], 'J_topo': []}
    record_every = max(1, epochs // 20)

    for epoch in range(epochs):
        net.train()
        for bx, by in loader:
            opt.zero_grad()
            loss_fn(net(bx), by.unsqueeze(-1)).backward()
            opt.step()
        scheduler.step()

        if verbose and (epoch % record_every == 0 or epoch == epochs - 1):
            net.eval()
            with torch.no_grad():
                pred_tr = net(X_tr).squeeze()
                tr_loss = loss_fn(pred_tr, Y_tr).item()
                pred_te = net(X_te).squeeze()
                te_loss = loss_fn(pred_te, Y_te).item()
                ss_r = ((pred_te - Y_te) ** 2).sum().item()
                ss_t = ((Y_te - Y_te.mean()) ** 2).sum().item()
                r2 = 1 - ss_r / (ss_t + 1e-8)
            Ws = net.get_weight_matrices()
            J, _ = compute_J_topo_from_weights(Ws, net.d_input)
            if verbose:
                print(f"    epoch {epoch:4d}: train={tr_loss:.4f} test={te_loss:.4f} r2={r2:.3f} J={J:.3f}")

        if epoch % record_every == 0 or epoch == epochs - 1:
            net.eval()
            with torch.no_grad():
                pred_te = net(X_te).squeeze()
                te_loss = loss_fn(pred_te, Y_te).item()
                ss_r = ((pred_te - Y_te) ** 2).sum().item()
                ss_t = ((Y_te - Y_te.mean()) ** 2).sum().item()
                r2 = 1 - ss_r / (ss_t + 1e-8)
            Ws = net.get_weight_matrices()
            J, _ = compute_J_topo_from_weights(Ws, net.d_input)
            metrics['test_loss'].append(te_loss)
            metrics['test_r2'].append(r2)
            metrics['train_loss'].append(tr_loss if 'tr_loss' in dir() else float('nan'))
            metrics['J_topo'].append(J)

    return metrics


# ──────────────────────────────────────────────────────
# POWER LAW FIT
# ──────────────────────────────────────────────────────

def fit_power_law(Ds, losses):
    Ds, losses = np.array(Ds, float), np.array(losses, float)
    if len(Ds) < 3:
        return {'beta': None, 'R2': None, 'E': None, 'B': None}
    E_init = float(min(losses) * 0.95)
    valid = losses - E_init > 1e-6
    if valid.sum() < 3:
        return {'beta': None, 'R2': None, 'E': E_init, 'B': None}

    log_D = np.log(Ds[valid])
    log_LmE = np.log(np.maximum(losses[valid] - E_init, 1e-6))
    coeffs = np.polyfit(log_D, log_LmE, deg=1)
    beta_init, log_B_init = -coeffs[0], coeffs[1]

    def obj(p):
        E, log_B, beta = p
        pred = math.e ** log_B * Ds ** (-beta) + E
        return np.sum((losses - pred) ** 2)

    res = minimize(obj, x0=[E_init, log_B_init, max(0.01, beta_init)],
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

def run_rff_simulation(
    d_manifold=20, d_task=18, s_smoothness=0.5, noise_std=0.1,
    dataset_sizes=None,
    epochs=500, lr=1e-3, batch_size=64,
    n_arch_full=4, seed=42,
    output_dir="experiments/phase_s0/rff_results",
    test_fraction=0.2,
):
    if dataset_sizes is None:
        dataset_sizes = [500, 1000, 2000, 4000, 8000]

    print("=" * 60)
    print("ThermoRG S0: RFF Teacher-Student Simulation (FIXED)")
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    beta_theory = s_smoothness / d_task
    print(f"\nTask: d_manifold={d_manifold}, d_task={d_task}, s={s_smoothness}")
    print(f"Theory: β = s/d_task = {s_smoothness}/{d_task} = {beta_theory:.4f}")
    print(f"Dataset sizes: {dataset_sizes}")
    print(f"Epochs: {epochs} (FIXED for all D)", f", LR: {lr}, Batch: {batch_size}")

    # ── Generate data ────────────────────────────────────
    print("\n[1/4] Generating data...")
    sphere = HypersphereSampler()
    X_all = sphere.sample(max(dataset_sizes), d_manifold, np.random.default_rng(seed))
    task = RandomFourierFeatureFunction(d_manifold, d_task, s_smoothness, seed=seed)
    Y_all = task(X_all, noise_std=noise_std).astype(np.float32)
    print(f"  X={X_all.shape}, Y std={Y_all.std():.3f}")

    # ── Architecture sweep ────────────────────────────────
    print("\n[2/4] Architecture survey...")
    arch_configs = []
    # Vary hidden dim and depth
    for d_mult in [0.5, 1.0, 2.0, 4.0]:
        d_hidden = max(8, int(d_manifold * d_mult))
        for n_layers in [1, 2, 3]:
            arch_configs.append({'d_hidden': d_hidden, 'n_layers': n_layers, 'init_scale': 1.0})
    # Vary init scale
    for scale in [0.1, 0.5, 2.0]:
        arch_configs.append({'d_hidden': d_manifold, 'n_layers': 2, 'init_scale': scale})
    # Deduplicate
    seen = set()
    unique_configs = []
    for c in arch_configs:
        key = (c['d_hidden'], c['n_layers'], c['init_scale'])
        if key not in seen:
            seen.add(key)
            unique_configs.append(c)
    arch_configs = unique_configs

    arch_survey = []
    for cfg in arch_configs:
        net = FCStudentNet(d_manifold, cfg['d_hidden'], 1, cfg['n_layers'], cfg['init_scale'])
        Ws = net.get_weight_matrices()
        J, _ = compute_J_topo_from_weights(Ws, d_manifold)
        arch_survey.append({**cfg, 'J_init': J})

    arch_survey.sort(key=lambda x: x['J_init'])
    print(f"  {len(arch_survey)} architectures, J range: [{arch_survey[0]['J_init']:.3f}, {arch_survey[-1]['J_init']:.3f}]")

    # Pick extremes + middle
    n_sel = min(n_arch_full, len(arch_survey))
    indices = list(np.linspace(0, len(arch_survey)-1, n_sel).astype(int))
    selected = [arch_survey[i] for i in indices]
    j_strs = ['J=%.3f' % a['J_init'] for a in selected]; print(f'  Selected: {j_strs}')

    # ── Generate fixed train/test split for each D ───────
    # CRITICAL FIX: Use the SAME test set for all D measurements
    # (generated from the FULL dataset, not from the D-subsampled data)
    print("\n[3/4] Generating fixed train/test splits...")
    rng_global = np.random.default_rng(seed + 999)  # Different seed from data gen
    n_full = len(X_all)
    perm_full = rng_global.permutation(n_full)
    n_test_full = int(test_fraction * n_full)
    test_indices = perm_full[:n_test_full]
    train_indices = perm_full[n_test_full:]

    X_te_full = X_all[test_indices]
    Y_te_full = Y_all[test_indices]
    print(f"  Fixed test set: {len(test_indices)} samples")
    print(f"  Fixed train pool: {len(train_indices)} samples")

    # ── D-scaling ────────────────────────────────────────
    print(f"\n[4/4] D-scaling ({len(dataset_sizes)} points, {epochs} epochs each)...")
    results = []

    for cfg in selected:
        name = f"h{cfg['d_hidden']}_L{cfg['n_layers']}_s{cfg['init_scale']}"
        print(f"\n  Architecture: {name} (J_init={cfg['J_init']:.3f})")

        # Quick sanity check: train on full data
        t0 = time.time()
        net_full = FCStudentNet(d_manifold, cfg['d_hidden'], 1, cfg['n_layers'], cfg['init_scale'])
        m_full = train_network(net_full, X_all, Y_all, lr=lr, epochs=epochs,
                               batch_size=batch_size, device=device,
                               X_te=X_te_full, Y_te=Y_te_full, verbose=False)
        Ws = net_full.get_weight_matrices()
        J_final, eta_final = compute_J_topo_from_weights(Ws, d_manifold)
        print(f"    Full-data train: final_loss={m_full['test_loss'][-1]:.4f}, J={J_final:.3f}, {time.time()-t0:.0f}s")

        # D-scaling: each D gets SAME epochs
        print(f"    D-scaling...")
        d_results = []
        for D in dataset_sizes:
            # Use first D samples from the train pool
            X_D = X_all[train_indices[:D]]
            Y_D = Y_all[train_indices[:D]]
            net_D = FCStudentNet(d_manifold, cfg['d_hidden'], 1, cfg['n_layers'], cfg['init_scale'])
            m_D = train_network(net_D, X_D, Y_D, lr=lr, epochs=epochs,
                               batch_size=batch_size, device=device,
                               X_te=X_te_full, Y_te=Y_te_full, verbose=False)
            best_loss = min(m_D['test_loss'])
            final_loss = m_D['test_loss'][-1]
            d_results.append({'D': D, 'loss': best_loss, 'final_loss': final_loss,
                            'losses': m_D['test_loss'], 'r2': max(m_D['test_r2'])})
            print(f"      D={D:5d}: best_loss={best_loss:.4f} final_loss={final_loss:.4f} r2_max={max(m_D['test_r2']):.3f}")

        losses_D = [r['loss'] for r in d_results]
        fit = fit_power_law([r['D'] for r in d_results], losses_D)
        print(f"      Fit: β={fit.get('beta', 'N/A'):.4f}, R²={fit.get('R2', 0):.3f}, E={fit.get('E', 0):.4f}, B={fit.get('B', 0):.4f}")

        results.append({
            'name': name,
            'config': cfg,
            'J_init': cfg['J_init'],
            'J_final': J_final,
            'eta_final': eta_final,
            'final_loss': min(m_full['test_loss']),
            'd_scaling': d_results,
            'beta_fit': fit,
        })

    # ── Save ──────────────────────────────────────────────
    out = {
        'timestamp': datetime.now().isoformat(),
        'task_params': {'d_manifold': d_manifold, 'd_task': d_task,
                       's_smoothness': s_smoothness, 'noise_std': noise_std,
                       'beta_theory': beta_theory},
        'dataset_sizes': dataset_sizes,
        'training': {'epochs': epochs, 'lr': lr, 'batch_size': batch_size},
        'results': [
            {k: v for k, v in r.items()
             if k not in ['d_scaling', 'eta_final', 'config']}
            for r in results
        ],
        'full_results': results,
    }
    with open(str(output_dir / 'rff_results.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    # ── Summary ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Theory: β = s/d_task = {s_smoothness}/{d_task} = {beta_theory:.4f}")
    print(f"\n{'Name':25s} {'J_init':>6s} {'J_final':>7s} {'beta':>7s} {'R^2':>6s}")
    print("-" * 60)
    for r in results:
        beta = r['beta_fit'].get('beta')
        R2 = r['beta_fit'].get('R2', 0)
        bn = f"{beta:.4f}" if beta else "FAIL"
        rn = f"{R2:.3f}" if R2 else "N/A"
        print(f"  {r['name']:23s} {r['J_init']:6.3f} {r['J_final']:7.3f} {bn:>7s} {rn:>6s}")

    # Check if D-scaling is working
    print("\n--- D-scaling check ---")
    first_arch = results[0]
    ds = [r['D'] for r in first_arch['d_scaling']]
    ls = [r['loss'] for r in first_arch['d_scaling']]
    print(f"  D values: {ds}")
    print(f"  Losses:   {[f'{l:.4f}' for l in ls]}")
    decreasing = all(ls[i] >= ls[i+1] for i in range(len(ls)-1))
    print(f"  Monotonically decreasing: {decreasing}")

    print(f"\nResults: {output_dir / 'rff_results.json'}")
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--d_manifold', type=int, default=20)
    p.add_argument('--d_task', type=int, default=18)
    p.add_argument('--s', dest='s_smoothness', type=float, default=0.5)
    p.add_argument('--noise', dest='noise_std', type=float, default=0.1)
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output_dir', default='experiments/phase_s0/rff_results')
    p.add_argument('--n_arch', dest='n_arch_full', type=int, default=4)
    run_rff_simulation(**vars(p.parse_args()))
