#!/usr/bin/env python3
"""
Gap 2b: λ_max During Training
ThermoRG — Phase Gap 2 Extension

Measure how λ_max scales with width D at different training checkpoints.
Key question: does the GELU saturation effect (λ_max ∝ D^0.26) emerge during training?

Expected behavior:
- At init: b ≈ 0.5 (random weights, Marchenko-Pastur)
- After training: b may decrease toward 0.26 (GELU saturation effect)
"""

import os, sys, json, math, time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, '/home/node/.openclaw/workspace/github_staging/ThermoRG-NN')

plt.rcParams.update({'font.size': 13, 'axes.labelsize': 14, 'axes.titlesize': 14,
                     'figure.figsize': (6, 4.5), 'figure.dpi': 130})

WORK_DIR = '/home/node/.openclaw/workspace/github_staging/ThermoRG-NN/experiments/phase_gap2'
os.makedirs(WORK_DIR, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")


# ── Synthetic dataset ────────────────────────────────────────────────────────
class SyntheticDataset(torch.utils.data.Dataset):
    """Simple synthetic dataset for training speed (no torchvision needed)."""
    def __init__(self, n_samples=1000, n_classes=10, img_size=32, n_channels=3, seed=42):
        rng = np.random.default_rng(seed)
        self.data = rng.normal(0, 1, (n_samples, n_channels, img_size, img_size)).astype(np.float32)
        self.data = np.clip(self.data, -3, 3) / 3  # normalize to [-1, 1]
        self.labels = rng.integers(0, n_classes, size=n_samples)
        self.n_samples = n_samples
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]), self.labels[idx]


# ── Config ──────────────────────────────────────────────────────────────────
WIDTHS = [128, 256, 512]
DEPTH = 4
CHECKPOINT_EPOCHS = [0, 1, 10, 50, 100]
N_EPOCHS = 100
N_TRAIN_SUBSET = 1000
BATCH_SIZE = 128
LR = 0.001
N_ITER_PI = 20


# ── ThermoNet ──────────────────────────────────────────────────────────────
class ThermoNet(nn.Module):
    """Conv2d + GELU network with configurable width and depth."""
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


def power_iteration(W, n_iterations=20):
    """Estimate λ_max (largest singular value) via power iteration."""
    W_flat = W.reshape(W.shape[0], -1)
    v = torch.randn(W_flat.shape[1], device=W_flat.device)
    v = v / (v.norm() + 1e-10)
    for _ in range(n_iterations):
        Wv = torch.matmul(W_flat, v)
        v_new = torch.matmul(W_flat.T, Wv)
        v_norm = v_new.norm() + 1e-10
        v = v_new / v_norm
    return float(torch.matmul(W_flat, v).norm().item())


def compute_layer_stats(model):
    """Compute λ_max and D_eff for all Conv2d/Linear layers in model.
    
    Handles Edge of Stability regime where λ_max may approach zero.
    """
    layer_lm, layer_de = [], []
    for _, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            W_flat = module.weight.data.reshape(module.weight.shape[0], -1)
            lambda_max = power_iteration(module.weight.data, n_iterations=20)
            # Clamp λ_max to prevent numerical explosion in EoS regime
            lambda_max = max(lambda_max, 1e-6)
            fro_sq = float((W_flat ** 2).sum().item())
            # D_eff = frobenius_norm^2 / lambda_max^2, capped to avoid overflow
            D_eff = min(fro_sq / (lambda_max ** 2), 1e6)
            layer_lm.append(lambda_max)
            layer_de.append(D_eff)
    return {
        'layer_lambda_max': layer_lm,
        'layer_D_eff': layer_de,
        'mean_lambda_max': float(np.mean(layer_lm)) if layer_lm else 0.0,
        'mean_D_eff': float(np.mean(layer_de)) if layer_de else 0.0,
    }


def fit_power_law(x, y):
    """Fit y = a * x^b via OLS on log-log. Returns {a, b, R2, std_err, intercept}."""
    log_x = np.log(np.array(x, dtype=float))
    log_y = np.log(np.array(y, dtype=float))
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
    a = math.exp(intercept)
    return {'a': a, 'b': slope, 'R2': r_value**2, 'std_err': std_err, 'intercept': intercept}


def train_one_epoch(model, loader, criterion, optimizer):
    """Train for one epoch, return loss."""
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader):
    """Compute accuracy."""
    model.eval()
    correct = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            correct += (out.argmax(1) == y).sum().item()
    return correct / len(loader.dataset)


# ── Load data ───────────────────────────────────────────────────────────────
print("Using synthetic dataset...")
train_subset = SyntheticDataset(n_samples=N_TRAIN_SUBSET, n_classes=10, seed=42)
test_ds = SyntheticDataset(n_samples=1000, n_classes=10, seed=99)
train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"Train: {len(train_subset)} samples, Test: {len(test_ds)} samples")

# ── Smoke test ─────────────────────────────────────────────────────────────
test_net = ThermoNet(width=128, depth=4).to(device)
x = torch.randn(2, 3, 32, 32, device=device)
out = test_net(x)
print(f"ThermoNet(128, 4) output shape: {out.shape}")
stats = compute_layer_stats(test_net)
print(f"Mean λ_max: {stats['mean_lambda_max']:.4f}, Mean D_eff: {stats['mean_D_eff']:.4f}")

# ── Training loop ───────────────────────────────────────────────────────────
training_results = {}
total_start = time.time()

for width in WIDTHS:
    print(f"\n{'='*70}")
    print(f"Training ThermoNet width={width}, depth={DEPTH}")
    print(f"{'='*70}")

    torch.manual_seed(42)
    np.random.seed(42)
    
    model = ThermoNet(width=width, depth=DEPTH, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    epoch_stats = {}
    
    for epoch in range(N_EPOCHS + 1):  # +1 to include epoch 0
        # Checkpoint: measure λ_max
        if epoch in CHECKPOINT_EPOCHS:
            model.eval()
            layer_stats = compute_layer_stats(model)
            acc = evaluate(model, test_loader)
            
            epoch_stats[epoch] = {
                'lambda_max': layer_stats['mean_lambda_max'],
                'D_eff': layer_stats['mean_D_eff'],
                'layer_lambda_max': layer_stats['layer_lambda_max'],
                'layer_D_eff': layer_stats['layer_D_eff'],
                'accuracy': acc,
            }
            print(f"  Epoch {epoch:3d}: λ_max={layer_stats['mean_lambda_max']:.4f}, "
                  f"D_eff={layer_stats['mean_D_eff']:.2f}, acc={acc:.4f}")
        
        # Train one epoch (skip epoch 0)
        if epoch < N_EPOCHS:
            loss = train_one_epoch(model, train_loader, criterion, optimizer)
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1:3d}/{N_EPOCHS}: loss={loss:.4f}")
    
    training_results[width] = epoch_stats
    print(f"\n  ✓ Width {width} training complete")

total_time = time.time() - total_start
print(f"\n{'='*70}")
print(f"Total training time: {total_time:.1f}s ({total_time/60:.1f} min)")
print(f"{'='*70}")

# ── Fitting ─────────────────────────────────────────────────────────────────
D_vals = np.array(WIDTHS, dtype=float)
fitting_results = {}

print("\n" + "="*75)
print(f"{'Epoch':>6} | {'b (λ_max)':>12} | {'R²':>8} | {'d (D_eff)':>12} | {'R²':>8} | {'b stderr':>10}")
print("-"*75)

for epoch in CHECKPOINT_EPOCHS:
    lm_vals = [training_results[w][epoch]['lambda_max'] for w in WIDTHS]
    de_vals = [training_results[w][epoch]['D_eff'] for w in WIDTHS]
    
    fit_lm = fit_power_law(D_vals, lm_vals)
    fit_de = fit_power_law(D_vals, de_vals)
    
    fitting_results[epoch] = {
        'b_lambda_max': fit_lm['b'],
        'R2_lambda_max': fit_lm['R2'],
        'b_lambda_max_std_err': fit_lm['std_err'],
        'd_D_eff': fit_de['b'],
        'R2_D_eff': fit_de['R2'],
        'per_width': {w: {'lambda_max': training_results[w][epoch]['lambda_max'],
                          'D_eff': training_results[w][epoch]['D_eff']} for w in WIDTHS},
    }
    
    print(f"{epoch:>6} | {fit_lm['b']:>12.4f} | {fit_lm['R2']:>8.4f} | "
          f"{fit_de['b']:>12.4f} | {fit_de['R2']:>8.4f} | {fit_lm['std_err']:>10.4f}")

print("="*75)
print("\nInterpretation:")
print("  b ≈ 0.50: Random matrix (Marchenko-Pastur) behavior")
print("  b ≈ 0.26: GELU saturation effect (theoretical prediction)")
print("  b ≈ 0.00: Dense network with saturating activations")

# ── Plotting ─────────────────────────────────────────────────────────────────
epochs = CHECKPOINT_EPOCHS
b_vals = [fitting_results[e]['b_lambda_max'] for e in epochs]
b_errs = [fitting_results[e]['b_lambda_max_std_err'] for e in epochs]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
markers = ['o', 's', '^', 'D', 'v']

# Plot 1: Exponent b vs Epoch
fig, ax = plt.subplots(figsize=(7, 5))
ax.errorbar(epochs, b_vals, yerr=b_errs, fmt='o-', capsize=5, 
            color='steelblue', markersize=8, linewidth=2,
            label='Measured b (λ_max ∝ D^b)')
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Random init (b=0.5)')
ax.axhline(y=0.26, color='green', linestyle=':', alpha=0.7, label='Theory prediction (b=0.26)')
ax.set_xlabel('Epoch')
ax.set_ylabel('Exponent b')
ax.set_title('λ_max Scaling Exponent b vs Training Epoch')
ax.set_xlim(-5, 105)
ax.set_ylim(0, 0.7)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, 'exponent_b_vs_epoch.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: exponent_b_vs_epoch.png")

# Plot 2: λ_max vs Width at Different Epochs
fig, ax = plt.subplots(figsize=(7, 5))
for i, epoch in enumerate(CHECKPOINT_EPOCHS):
    lm_vals = [training_results[w][epoch]['lambda_max'] for w in WIDTHS]
    ax.plot(WIDTHS, lm_vals, marker=markers[i], color=colors[i], 
            linewidth=2, markersize=7, label=f'Epoch {epoch}')
ax.set_xlabel('Width D')
ax.set_ylabel('Mean λ_max')
ax.set_title('λ_max vs Width at Different Training Epochs')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, 'lambda_max_vs_width_by_epoch.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: lambda_max_vs_width_by_epoch.png")

# Plot 3: log-log λ_max vs D with fits
fig, ax = plt.subplots(figsize=(7, 5))
log_D = np.log(D_vals)
for i, epoch in enumerate(CHECKPOINT_EPOCHS):
    lm_vals = [training_results[w][epoch]['lambda_max'] for w in WIDTHS]
    log_lm = np.log(lm_vals)
    ax.scatter(log_D, log_lm, marker=markers[i], color=colors[i], s=60,
               label=f'Epoch {epoch}', zorder=5)
    fit_lm = fit_power_law(D_vals, lm_vals)
    x_fit = np.linspace(log_D.min(), log_D.max(), 100)
    y_fit = fit_lm['intercept'] + fit_lm['b'] * x_fit
    ax.plot(x_fit, y_fit, '--', color=colors[i], alpha=0.6, linewidth=1.5)
ax.set_xlabel('log(D)')
ax.set_ylabel('log(λ_max)')
ax.set_title('log(λ_max) vs log(D) — Evolution During Training')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, 'log_lambda_max_vs_log_D.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: log_lambda_max_vs_log_D.png")

# Plot 4: Accuracy vs Epoch
fig, ax = plt.subplots(figsize=(7, 5))
for i, width in enumerate(WIDTHS):
    accs = [training_results[width][e]['accuracy'] for e in CHECKPOINT_EPOCHS]
    ax.plot(CHECKPOINT_EPOCHS, accs, marker=markers[i], color=colors[i],
            linewidth=2, markersize=7, label=f'D={width}')
ax.set_xlabel('Epoch')
ax.set_ylabel('Test Accuracy')
ax.set_title('Test Accuracy vs Training Epoch')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(-5, 105)
plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, 'accuracy_vs_epoch.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: accuracy_vs_epoch.png")

# Combined figure
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax = axes[0, 0]
ax.errorbar(epochs, b_vals, yerr=b_errs, fmt='o-', capsize=5, 
            color='steelblue', markersize=8, linewidth=2)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Init (b=0.5)')
ax.axhline(y=0.26, color='green', linestyle=':', alpha=0.7, label='Theory (b=0.26)')
ax.set_xlabel('Epoch')
ax.set_ylabel('Exponent b')
ax.set_title('(a) λ_max Scaling Exponent vs Epoch')
ax.set_xlim(-5, 105)
ax.set_ylim(0, 0.7)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
for i, epoch in enumerate(CHECKPOINT_EPOCHS):
    lm_vals = [training_results[w][epoch]['lambda_max'] for w in WIDTHS]
    ax.plot(WIDTHS, lm_vals, marker=markers[i], color=colors[i], 
            linewidth=2, markersize=6, label=f'Epoch {epoch}')
ax.set_xlabel('Width D')
ax.set_ylabel('Mean λ_max')
ax.set_title('(b) λ_max vs Width')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, which='both')

ax = axes[1, 0]
for i, epoch in enumerate(CHECKPOINT_EPOCHS):
    lm_vals = [training_results[w][epoch]['lambda_max'] for w in WIDTHS]
    log_lm = np.log(lm_vals)
    ax.scatter(log_D, log_lm, marker=markers[i], color=colors[i], s=50, label=f'E{epoch}')
    fit_lm = fit_power_law(D_vals, lm_vals)
    x_fit = np.linspace(log_D.min(), log_D.max(), 100)
    y_fit = fit_lm['intercept'] + fit_lm['b'] * x_fit
    ax.plot(x_fit, y_fit, '--', color=colors[i], alpha=0.5, linewidth=1.5)
ax.set_xlabel('log(D)')
ax.set_ylabel('log(λ_max)')
ax.set_title('(c) log-log Scaling')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
for i, width in enumerate(WIDTHS):
    accs = [training_results[width][e]['accuracy'] for e in CHECKPOINT_EPOCHS]
    ax.plot(CHECKPOINT_EPOCHS, accs, marker=markers[i], color=colors[i],
            linewidth=2, markersize=6, label=f'D={width}')
ax.set_xlabel('Epoch')
ax.set_ylabel('Test Accuracy')
ax.set_title('(d) Test Accuracy vs Epoch')
ax.set_xlim(-5, 105)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle('Gap 2b: λ_max During Training — GELU Saturation Effect', 
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, 'gap2b_combined.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: gap2b_combined.png")

# ── Save results ─────────────────────────────────────────────────────────────
output = {
    'experiment': 'gap2b_lambda_max_during_training',
    'description': 'Measure λ_max scaling exponent b during training',
    'widths': WIDTHS,
    'depth': DEPTH,
    'checkpoint_epochs': CHECKPOINT_EPOCHS,
    'n_epochs_total': N_EPOCHS,
    'train_subset_size': N_TRAIN_SUBSET,
    'batch_size': BATCH_SIZE,
    'learning_rate': LR,
    'power_iteration_iterations': N_ITER_PI,
    'training_time_seconds': total_time,
    'fitting_results': fitting_results,
    'per_epoch_per_width': {
        str(epoch): {
            str(width): training_results[width][epoch] 
            for width in WIDTHS
        } for epoch in CHECKPOINT_EPOCHS
    },
}

results_path = os.path.join(WORK_DIR, 'training_time_results.json')
with open(results_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to: {results_path}")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "="*75)
print("SUMMARY: λ_max Scaling Exponent b During Training")
print("="*75)
print(f"\n{'Epoch':>6} | {'b (λ_max)':>10} | {'R²':>8} | {'d (D_eff)':>10} | {'R²':>8}")
print("-"*75)
for epoch in CHECKPOINT_EPOCHS:
    fr = fitting_results[epoch]
    print(f"{epoch:>6} | {fr['b_lambda_max']:>10.4f} | {fr['R2_lambda_max']:>8.4f} | "
          f"{fr['d_D_eff']:>10.4f} | {fr['R2_D_eff']:>8.4f}")
print("="*75)

print("\nKey observations:")
print(f"  - b at init (epoch 0): {fitting_results[0]['b_lambda_max']:.4f}")
print(f"  - b at epoch 100:      {fitting_results[100]['b_lambda_max']:.4f}")
print(f"  - Change in b:         {fitting_results[100]['b_lambda_max'] - fitting_results[0]['b_lambda_max']:.4f}")

print("\nExpected behavior:")
print("  - b ≈ 0.50 at init: random matrix Marchenko-Pastur scaling")
print("  - b ≈ 0.26 after training: GELU saturation effect (if it emerges)")
print("  - Any intermediate value suggests partial saturation")

print(f"\nTotal runtime: {time.time()-total_start:.1f}s")
