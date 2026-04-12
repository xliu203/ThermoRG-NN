#!/usr/bin/env python3
"""Gap 2: RMT + GELU Eigenvalue Measurement — standalone runner."""

import os, sys, json, math, time

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, '/home/node/.openclaw/workspace/github_staging/ThermoRG-NN')

WORK_DIR = '/home/node/.openclaw/workspace/github_staging/ThermoRG-NN/experiments/phase_gap2'
os.makedirs(WORK_DIR, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# ── ThermoNet ──────────────────────────────────────────────────────────────
class ThermoNet(nn.Module):
    def __init__(self, width=128, depth=4, in_channels=3, num_classes=10,
                 kernel_size=3, padding=1):
        super().__init__()
        self.width = width
        self.depth = depth
        layers = []
        c = in_channels
        for i in range(depth):
            layers.append(nn.Conv2d(c, width, kernel_size=kernel_size, padding=padding))
            layers.append(nn.GELU())
            c = width
        layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(width, num_classes)]
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, mean=0.0, std=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

# Smoke test
test_net = ThermoNet(width=64, depth=3).to(device)
x = torch.randn(2, 3, 32, 32, device=device)
out = test_net(x)
print(f"ThermoNet(64,3) output: {out.shape}")

# ── Power iteration ──────────────────────────────────────────────────────────
def power_iteration(W, n_iterations=20):
    W_flat = W.reshape(W.shape[0], -1)
    v = torch.randn(W_flat.shape[1], device=W_flat.device)
    v = v / (v.norm() + 1e-10)
    for _ in range(n_iterations):
        Wv = torch.matmul(W_flat, v)
        v_new = torch.matmul(W_flat.T, Wv)
        v_norm = v_new.norm() + 1e-10
        v_new = v_new / v_norm
        v = v_new
    return float(torch.matmul(W_flat, v).norm().item())

def svd_lambda_max(W):
    W_flat = W.reshape(W.shape[0], -1)
    return float(torch.linalg.svdvals(W_flat).max().item())

# Accuracy check
W_test = torch.randn(256, 128, 3, 3)
lm_pi = power_iteration(W_test, n_iterations=20)
lm_sv = svd_lambda_max(W_test)
print(f"Power iter λ_max={lm_pi:.4f}  SVD={lm_sv:.4f}  RelErr={abs(lm_pi-lm_sv)/lm_sv*100:.2f}%")

# ── D_eff ──────────────────────────────────────────────────────────────────
def compute_deff(W, n_iter=20):
    W_flat = W.reshape(W.shape[0], -1)
    lambda_max = power_iteration(W, n_iterations=n_iter)
    fro_sq = float((W_flat ** 2).sum().item())
    deff = fro_sq / (lambda_max ** 2 + 1e-10)
    return {'lambda_max': lambda_max, 'D_eff': deff, 'fro_sq': fro_sq}

# Quick check
rc = compute_deff(torch.randn(128, 64, 3, 3))
print(f"D_eff quick check: {rc}")

# ── Width sweep ─────────────────────────────────────────────────────────────
WIDTHS = [32, 64, 128, 256, 512]
DEPTH = 4
N_SEEDS = 5
N_ITER = 20

results_all = {}
start_time = time.time()

for width in WIDTHS:
    print(f"\n{'='*60}\nWidth = {width}\n{'='*60}")
    per_seed_lm = []
    per_seed_de = []

    for seed in range(N_SEEDS):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = ThermoNet(width=width, depth=DEPTH).to(device)

        seed_lm, seed_de = [], []
        for _, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                res = compute_deff(module.weight.data, n_iter=N_ITER)
                seed_lm.append(res['lambda_max'])
                seed_de.append(res['D_eff'])

        mean_lm = float(np.mean(seed_lm))
        mean_de = float(np.mean(seed_de))
        per_seed_lm.append(mean_lm)
        per_seed_de.append(mean_de)
        print(f"  seed {seed}: λ_max={mean_lm:.4f}, D_eff={mean_de:.4f}")

    results_all[width] = {
        'lambda_max_mean': float(np.mean(per_seed_lm)),
        'lambda_max_std':  float(np.std(per_seed_lm)),
        'D_eff_mean':      float(np.mean(per_seed_de)),
        'D_eff_std':       float(np.std(per_seed_de)),
        'per_seed_lambda_max': per_seed_lm,
        'per_seed_D_eff':  per_seed_de,
    }
    print(f"  >> λ_max = {results_all[width]['lambda_max_mean']:.4f} ± {results_all[width]['lambda_max_std']:.4f}")
    print(f"  >> D_eff = {results_all[width]['D_eff_mean']:.4f} ± {results_all[width]['D_eff_std']:.4f}")

elapsed = time.time() - start_time
print(f"\n✓ Width sweep complete in {elapsed:.1f}s")

# ── Power law fitting ────────────────────────────────────────────────────────
D_vals = np.array(WIDTHS, dtype=float)
log_D = np.log(D_vals)

lm_means = np.array([results_all[w]['lambda_max_mean'] for w in WIDTHS])
lm_stds  = np.array([results_all[w]['lambda_max_std']  for w in WIDTHS])
de_means = np.array([results_all[w]['D_eff_mean']     for w in WIDTHS])
de_stds  = np.array([results_all[w]['D_eff_std']       for w in WIDTHS])

log_lm = np.log(lm_means)
log_de = np.log(de_means)

def fit_power_law(x, y):
    log_x, log_y = np.log(x), np.log(y)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
    a = math.exp(intercept)
    return {'a': a, 'b': slope, 'R2': r_value**2, 'std_err': std_err, 'intercept': intercept}

fit_lm = fit_power_law(D_vals, lm_means)
fit_de = fit_power_law(D_vals, de_means)

print("\n" + "="*65)
print("POWER LAW FITS")
print("="*65)
print(f"\nλ_max = a · D^b")
print(f"  a = {fit_lm['a']:.6f}")
print(f"  b = {fit_lm['b']:.4f}  (expected ≈ 0.26)")
print(f"  R² = {fit_lm['R2']:.6f}")
print(f"  std_err(b) = {fit_lm['std_err']:.4f}")
print(f"\nD_eff = c · D^d")
print(f"  c = {fit_de['a']:.6f}")
print(f"  d = {fit_de['b']:.4f}  (expected ≈ 0.48)")
print(f"  R² = {fit_de['R2']:.6f}")
print(f"  std_err(d) = {fit_de['std_err']:.4f}")
print("="*65)

# ── Plotting ─────────────────────────────────────────────────────────────────
plt.rcParams.update({'font.size': 13, 'axes.labelsize': 14, 'axes.titlesize': 14,
                     'figure.figsize': (6, 4.5), 'figure.dpi': 130})

x_fit = np.linspace(log_D.min(), log_D.max(), 200)

# Plot 1: λ_max
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.errorbar(log_D, log_lm, yerr=lm_stds/lm_means, fmt='o', capsize=4, color='steelblue',
            label='Measured (mean ± std)')
y_fit_lm = fit_lm['intercept'] + fit_lm['b'] * x_fit
ax.plot(x_fit, y_fit_lm, 'r--', lw=2, label=f"Fit: D^{fit_lm['b']:.3f} (R²={fit_lm['R2']:.4f})")
ax.plot(x_fit, fit_lm['intercept'] + 0.26*x_fit, 'g:', lw=1.5, alpha=0.7, label='Theory: D^0.26')
ax.set_xlabel('log(D)'); ax.set_ylabel('log(λ_max)')
ax.set_title('λ_max vs Width D — Conv2d+GELU'); ax.legend(fontsize=10); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, 'lambda_max_vs_D.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: lambda_max_vs_D.png")

# Plot 2: D_eff
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.errorbar(log_D, log_de, yerr=de_stds/de_means, fmt='s', capsize=4, color='darkorange',
            label='Measured (mean ± std)')
y_fit_de = fit_de['intercept'] + fit_de['b'] * x_fit
ax.plot(x_fit, y_fit_de, 'r--', lw=2, label=f"Fit: D^{fit_de['b']:.3f} (R²={fit_de['R2']:.4f})")
ax.plot(x_fit, fit_de['intercept'] + 0.48*x_fit, 'g:', lw=1.5, alpha=0.7, label='Theory: D^0.48')
ax.set_xlabel('log(D)'); ax.set_ylabel('log(D_eff)')
ax.set_title('D_eff vs Width D — Conv2d+GELU'); ax.legend(fontsize=10); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, 'D_eff_vs_D.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: D_eff_vs_D.png")

# Combined figure
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ax = axes[0]
ax.errorbar(log_D, log_lm, yerr=lm_stds/lm_means, fmt='o', capsize=4, color='steelblue')
ax.plot(x_fit, y_fit_lm, 'r--', lw=2, label=f"D^{fit_lm['b']:.3f} (R²={fit_lm['R2']:.4f})")
ax.plot(x_fit, fit_lm['intercept']+0.26*x_fit, 'g:', lw=1.5, alpha=0.7, label='D^0.26 theory')
ax.set_xlabel('log(D)'); ax.set_ylabel('log(λ_max)'); ax.set_title('λ_max Scaling')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax = axes[1]
ax.errorbar(log_D, log_de, yerr=de_stds/de_means, fmt='s', capsize=4, color='darkorange')
ax.plot(x_fit, y_fit_de, 'r--', lw=2, label=f"D^{fit_de['b']:.3f} (R²={fit_de['R2']:.4f})")
ax.plot(x_fit, fit_de['intercept']+0.48*x_fit, 'g:', lw=1.5, alpha=0.7, label='D^0.48 theory')
ax.set_xlabel('log(D)'); ax.set_ylabel('log(D_eff)'); ax.set_title('D_eff Scaling')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.suptitle('ThermoRG Gap 2: RMT + GELU Eigenvalue Measurement', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, 'gap2_combined.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: gap2_combined.png")

# ── Save results ─────────────────────────────────────────────────────────────
output = {
    'experiment': 'gap2_rmt_eigenvalue',
    'description': 'RMT+GELU eigenvalue measurement at initialization',
    'widths': WIDTHS, 'depth': DEPTH, 'n_seeds': N_SEEDS, 'n_iterations': N_ITER,
    'fit_lambda_max': {
        'a': fit_lm['a'], 'b': fit_lm['b'], 'R2': fit_lm['R2'],
        'std_err': fit_lm['std_err'], 'expected_b': 0.26,
        'relative_error_pct': abs(fit_lm['b']-0.26)/0.26*100,
    },
    'fit_D_eff': {
        'c': fit_de['a'], 'd': fit_de['b'], 'R2': fit_de['R2'],
        'std_err': fit_de['std_err'], 'expected_d': 0.48,
        'relative_error_pct': abs(fit_de['b']-0.48)/0.48*100,
    },
    'per_width': {str(w): results_all[w] for w in WIDTHS},
}

results_path = os.path.join(WORK_DIR, 'results.json')
with open(results_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to: {results_path}")

# Summary table
print("\n" + "="*70)
print(f"{'D':>6} | {'λ_max mean':>12} | {'λ_max std':>10} | {'D_eff mean':>12} | {'D_eff std':>10}")
print("-"*70)
for w in WIDTHS:
    r = results_all[w]
    print(f"{w:>6} | {r['lambda_max_mean']:>12.4f} | {r['lambda_max_std']:>10.4f} | "
          f"{r['D_eff_mean']:>12.4f} | {r['D_eff_std']:>10.4f}")
print("="*70)

print("\nFINAL RESULTS")
print(f"  λ_max ∝ D^b  →  b = {fit_lm['b']:.4f}  (expected 0.26)  R² = {fit_lm['R2']:.6f}")
print(f"  D_eff ∝ D^d  →  d = {fit_de['b']:.4f}  (expected 0.48)  R² = {fit_de['R2']:.6f}")
print(f"\nTotal runtime: {time.time()-start_time:.1f}s")
