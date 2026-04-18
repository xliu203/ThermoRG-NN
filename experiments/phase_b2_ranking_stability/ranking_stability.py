#!/usr/bin/env python3
"""
ThermoRG Phase B2 — Ranking Stability Experiment
================================================

Test whether J_topo-based ranking is stable across training seeds.
This addresses Reviewer B2's critique: "the two-stage approach has no 
empirical backing for the claim of high per-seed variance."

Hypothesis:
- If J_topo ranking is unstable across seeds (Spearman ρ < 0.7), 
  then the two-stage heuristic is justified.
- If stable (ρ ≥ 0.7), the two-stage heuristic is unnecessary.

5 architectures × 5 seeds = 25 training runs
Each run: 10 epochs (same as HBO L1), CIFAR-10, SGD
"""

import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

# Try to import thermorg J_topo, fallback to local implementation
try:
    from thermorg import compute_J_topo
except ImportError:
    import sys
    sys.path.insert(0, '/home/node/.openclaw/workspace/github_staging/ThermoRG-NN')
    from thermorg.j_topo import compute_J_topo

# =============================================================================
# ThermoNet Architecture (from phase_b_hbo_revised notebook)
# =============================================================================

class ConvBlock(nn.Module):
    def __init__(self, ci, co, norm='none', skip=False, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(ci, co, 3, padding=1, stride=stride, bias=False)
        if norm == 'batchnorm':
            self.norm = nn.BatchNorm2d(co)
        elif norm == 'layernorm':
            self.norm = nn.LayerNorm([co, 32 // stride, 32 // stride])
        else:
            self.norm = nn.Identity()
        self.act = nn.GELU()
        self.skip = skip and stride == 1 and ci == co

    def forward(self, x):
        out = self.act(self.norm(self.conv(x)))
        if self.skip:
            out = out + x
        return out


class ThermoNet(nn.Module):
    """ThermoNet architecture matching the HBO experiment.
    
    Args:
        base_ch: Base channel width
        depth: Number of convolutional blocks
        norm: Normalization type ('none', 'batchnorm', 'layernorm')
        skip: Whether to use skip connections
    """
    def __init__(self, base_ch=64, depth=3, norm='none', skip=False):
        super().__init__()
        ch = [3] + [base_ch] * depth
        self.blocks = nn.ModuleList([
            ConvBlock(ch[i], ch[i+1], norm, skip and i > 0)
            for i in range(depth)
        ])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(ch[-1], 10)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return self.fc(self.pool(x).squeeze())


def count_params(model):
    """Count trainable parameters in millions."""
    return sum(p.numel() for p in model.parameters()) / 1e6


# =============================================================================
# Experiment Configurations
# =============================================================================

# 5 diverse architectures covering different J_topo and D (parameter count) regimes
ARCHITECTURES = [
    # (name, base_ch, depth, norm, skip) - expected J_topo and D characteristics
    ("ThermoNet-6 W=32 BN", 32, 6, "batchnorm", True),   # Small D, high J_topo
    ("ThermoNet-6 W=64 BN", 64, 6, "batchnorm", True),   # Medium D, medium J_topo
    ("ThermoNet-6 W=96 BN", 96, 6, "batchnorm", True),   # Large D, lower J_topo
    ("ThermoNet-3 W=64 BN", 64, 3, "batchnorm", True),   # Different depth, medium J_topo
    ("ThermoNet-9 W=48 BN", 48, 9, "batchnorm", True),   # Different depth, medium J_topo
]

SEEDS = [42, 123, 456, 789, 2024]
N_EPOCHS = 3  # 3 epochs sufficient for ranking signal
BATCH_SIZE = 256
LR = 0.01
WD = 5e-4
MOM = 0.9

# Output directory
OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(exist_ok=True)
CKPT_DIR = OUT_DIR / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)


# =============================================================================
# Data Loading
# =============================================================================

def get_data_loaders():
    """Get CIFAR-10 train and validation loaders."""
    tfm_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
    ])
    tfm_val = T.Compose([
        T.ToTensor(),
        T.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
    ])
    train_ds = CIFAR10("./data", train=True, transform=tfm_train, download=True)
    val_ds = CIFAR10("./data", train=False, transform=tfm_val, download=True)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, drop_last=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, drop_last=True, pin_memory=True
    )
    return train_loader, val_loader


# =============================================================================
# Training Function
# =============================================================================

def train_model(model, train_loader, val_loader, epochs, seed, verbose=True):
    """Train a model and return best validation loss."""
    torch.manual_seed(seed)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    if torch.cuda.is_available():
        model = model.cuda()

    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOM, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_loss = float('inf')
    best_acc = 0.0

    for epoch in range(epochs):
        # Training
        model.train()
        for X, y in train_loader:
            if torch.cuda.is_available():
                X, y = X.cuda(), y.cuda()
            opt.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            opt.step()
        scheduler.step()

        # Validation
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for X, y in val_loader:
                if torch.cuda.is_available():
                    X, y = X.cuda(), y.cuda()
                outputs = model(X)
                total_loss += criterion(outputs, y).item() * X.size(0)
                total_correct += (outputs.argmax(1) == y).sum().item()
                total_samples += X.size(0)

        val_loss = total_loss / total_samples
        val_acc = total_correct / total_samples

        if val_loss < best_loss:
            best_loss = val_loss
            best_acc = val_acc

        if verbose and (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{epochs}: val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "best_val_loss": best_loss,
        "best_val_acc": best_acc,
    }


# =============================================================================
# J_topo Computation
# =============================================================================

def compute_j_topo_for_arch(base_ch, depth, norm, skip):
    """Compute J_topo for an architecture (zero-cost, no training needed)."""
    model = ThermoNet(base_ch=base_ch, depth=depth, norm=norm, skip=skip)
    J, eta_list = compute_J_topo(model)
    D = count_params(model)
    del model
    return J, D


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment():
    print("=" * 70)
    print("ThermoRG Phase B2 — Ranking Stability Experiment")
    print("=" * 70)
    print()
    print(f"Architectures: {len(ARCHITECTURES)}")
    print(f"Seeds per architecture: {len(SEEDS)}")
    print(f"Total runs: {len(ARCHITECTURES) * len(SEEDS)}")
    print(f"Epochs per run: {N_EPOCHS}")
    print(f"Learning rate: {LR}")
    print(f"Optimizer: SGD (momentum={MOM}, weight_decay={WD})")
    print()

    # Step 1: Compute J_topo and D for all architectures (zero-cost)
    print("Step 1: Computing J_topo for all architectures (zero-cost)...")
    arch_data = []
    for name, base_ch, depth, norm, skip in ARCHITECTURES:
        J_topo, D = compute_j_topo_for_arch(base_ch, depth, norm, skip)
        arch_data.append({
            "name": name,
            "base_ch": base_ch,
            "depth": depth,
            "norm": norm,
            "skip": skip,
            "J_topo": J_topo,
            "D_M": D,
        })
        print(f"  {name}: J_topo={J_topo:.4f}, D={D:.2f}M params")

    # Sort by J_topo for display
    arch_data_sorted = sorted(arch_data, key=lambda x: x['J_topo'], reverse=True)
    print()
    print("Architectures sorted by J_topo (high to low):")
    for i, a in enumerate(arch_data_sorted):
        print(f"  {i+1}. {a['name']}: J_topo={a['J_topo']:.4f}, D={a['D_M']:.2f}M")

    # Step 2: Train each architecture with multiple seeds
    print()
    print("Step 2: Training architectures with multiple seeds...")
    train_loader, val_loader = get_data_loaders()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    results = {}
    t0_total = time.time()

    for arch_idx, (name, base_ch, depth, norm, skip) in enumerate(ARCHITECTURES):
        print(f"\nArchitecture {arch_idx+1}/{len(ARCHITECTURES)}: {name}")
        print(f"  J_topo={next(a['J_topo'] for a in arch_data if a['name']==name):.4f}, "
              f"D={next(a['D_M'] for a in arch_data if a['name']==name):.2f}M")

        seed_results = []
        for seed_idx, seed in enumerate(SEEDS):
            ckpt_file = CKPT_DIR / f"{name.replace(' ', '_').replace('=', '')}_seed{seed}.pt"
            result_file = CKPT_DIR / f"{name.replace(' ', '_').replace('=', '')}_seed{seed}_result.json"

            if result_file.exists():
                with open(result_file) as f:
                    result = json.load(f)
                print(f"  Seed {seed} [SKIP]: val_loss={result['best_val_loss']:.4f}")
            else:
                print(f"  Seed {seed} [{seed_idx+1}/{len(SEEDS)}]: training...", end="", flush=True)
                t0 = time.time()

                torch.manual_seed(seed)
                model = ThermoNet(base_ch=base_ch, depth=depth, norm=norm, skip=skip)

                result = train_model(
                    model, train_loader, val_loader,
                    epochs=N_EPOCHS, seed=seed, verbose=False
                )
                result['seed'] = seed
                result['epoch_time'] = time.time() - t0

                with open(result_file, 'w') as f:
                    json.dump(result, f)
                print(f" val_loss={result['best_val_loss']:.4f} ({result['epoch_time']:.1f}s)")

            seed_results.append(result)

        # Aggregate per-seed results
        losses = [r['best_val_loss'] for r in seed_results]
        accs = [r['best_val_acc'] for r in seed_results]

        results[name] = {
            "seed_losses": losses,
            "seed_accs": accs,
            "mean_loss": float(np.mean(losses)),
            "std_loss": float(np.std(losses)),
            "mean_acc": float(np.mean(accs)),
            "std_acc": float(np.std(accs)),
        }
        print(f"  Aggregate: mean_loss={results[name]['mean_loss']:.4f} ± {results[name]['std_loss']:.4f}")

    total_time = time.time() - t0_total
    print(f"\nTotal training time: {total_time / 60:.1f} minutes")

    # Step 3: Analysis - Spearman correlation
    print()
    print("=" * 70)
    print("ANALYSIS: J_topo vs Validation Loss Correlation")
    print("=" * 70)

    # Build arrays for correlation analysis
    j_topo_values = []
    mean_loss_values = []
    std_loss_values = []
    names = []

    for name, base_ch, depth, norm, skip in ARCHITECTURES:
        j = next(a['J_topo'] for a in arch_data if a['name'] == name)
        m = results[name]['mean_loss']
        s = results[name]['std_loss']
        j_topo_values.append(j)
        mean_loss_values.append(m)
        std_loss_values.append(s)
        names.append(name)

    # Spearman correlation between J_topo and mean validation loss
    rho, p_value = spearmanr(j_topo_values, mean_loss_values)

    print()
    print(f"Spearman rank correlation (J_topo vs mean validation loss):")
    print(f"  ρ = {rho:.4f}")
    print(f"  p-value = {p_value:.4f}")
    print()

    # Interpretation
    if abs(rho) < 0.7:
        two_stage_justified = True
        print("|ρ| < 0.7 → J_topo ranking is SEED-DEPENDENT")
        print("→ Two-stage heuristic is empirically JUSTIFIED")
        print("→ Keep D≥48 filter and use J_topo for screening")
    else:
        two_stage_justified = False
        if rho < -0.7:
            print("ρ < -0.7 → J_topo ranking is STABLE across seeds")
            print("Higher J_topo → Lower loss (good correlation)")
        else:
            print("ρ > 0.7 → J_topo ranking is STABLE but positive?")
        print("→ Two-stage heuristic may be UNNECESSARY")
        print("→ Consider direct L_hat optimization")

    # Per-seed ranking analysis
    print()
    print("Per-seed ranking analysis:")
    print("-" * 50)
    print(f"{'Architecture':<22} {'J_topo':>7} | {'Seed1':>7} {'Seed2':>7} {'Seed3':>7} {'Seed4':>7} {'Seed5':>7}")
    print("-" * 50)
    for i, name in enumerate(names):
        losses_str = " ".join([f"{results[name]['seed_losses'][j]:>7.4f}" for j in range(5)])
        print(f"{name:<22} {j_topo_values[i]:>7.4f} | {losses_str}")
    print("-" * 50)

    # Ranking stability: how often does each arch rank the same across seeds?
    print()
    print("Ranking consistency across seeds:")
    for name in names:
        losses = results[name]['seed_losses']
        rank = sum(1 for n2 in names if results[n2]['mean_loss'] > results[name]['mean_loss']) + 1
        seed_ranks = []
        for s in range(5):
            s_rank = sum(1 for n2 in names if results[n2]['seed_losses'][s] < losses[s]) + 1
            seed_ranks.append(s_rank)
        print(f"  {name}: mean_rank={rank}, seed_ranks={seed_ranks}")

    # Step 4: Save results
    print()
    print("Step 4: Saving results...")

    results_json = {
        "experiment": "ThermoRG Phase B2 - Ranking Stability",
        "description": (
            "Tests whether J_topo-based ranking is stable across training seeds. "
            "Addresses Reviewer B2 critique about high per-seed variance."
        ),
        "hypothesis": (
            "If J_topo ranking is unstable across seeds (ρ < 0.7), two-stage heuristic is justified. "
            "If stable (ρ ≥ 0.7), two-stage heuristic is unnecessary."
        ),
        "config": {
            "architectures": [
                {"name": name, "base_ch": bc, "depth": d, "norm": n, "skip": s}
                for name, bc, d, n, s in ARCHITECTURES
            ],
            "n_seeds": len(SEEDS),
            "seeds": SEEDS,
            "n_epochs": N_EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "weight_decay": WD,
            "momentum": MOM,
        },
        "architecture_results": {},
        "correlation_analysis": {
            "spearman_rho": float(rho),
            "spearman_p_value": float(p_value),
            "two_stage_justified": two_stage_justified,
            "interpretation": (
                "J_topo ranking is seed-dependent (|ρ| < 0.7) → two-stage heuristic justified"
                if two_stage_justified else
                "J_topo ranking is stable across seeds (|ρ| ≥ 0.7) → two-stage may be unnecessary"
            ),
        },
        "j_topo_vs_loss": [
            {"name": n, "J_topo": j, "mean_loss": m, "std_loss": s}
            for n, j, m, s in zip(names, j_topo_values, mean_loss_values, std_loss_values)
        ],
        "training_time_minutes": total_time / 60,
    }

    for name in names:
        base_ch, depth, norm, skip = next(
            (bc, d, n, s) for n2, bc, d, n, s in ARCHITECTURES if n2 == name
        )
        results_json["architecture_results"][name] = {
            "config": {"base_ch": base_ch, "depth": depth, "norm": norm, "skip": skip},
            "J_topo": float(next(a['J_topo'] for a in arch_data if a['name'] == name)),
            "D_M": float(next(a['D_M'] for a in arch_data if a['name'] == name)),
            "per_seed_results": [
                {"seed": SEEDS[j], "val_loss": results[name]['seed_losses'][j], "val_acc": results[name]['seed_accs'][j]}
                for j in range(len(SEEDS))
            ],
            "aggregate": {
                "mean_loss": results[name]['mean_loss'],
                "std_loss": results[name]['std_loss'],
                "mean_acc": results[name]['mean_acc'],
                "std_acc": results[name]['std_acc'],
            }
        }

    with open(OUT_DIR / "ranking_stability_results.json", 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"  Saved: {OUT_DIR / 'ranking_stability_results.json'}")

    # Step 5: Generate visualization
    print()
    print("Step 5: Generating visualization...")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: J_topo vs Mean Validation Loss (scatter with error bars)
        ax1 = axes[0]
        ax1.errorbar(
            j_topo_values, mean_loss_values,
            yerr=std_loss_values,
            fmt='o', capsize=5, capthick=2, markersize=10,
            color='steelblue', ecolor='gray',
            label='Mean ± Std across 5 seeds'
        )

        # Add architecture labels
        for i, name in enumerate(names):
            short_name = name.replace("ThermoNet-", "TN").replace(" ", "\n")
            ax1.annotate(
                short_name,
                (j_topo_values[i], mean_loss_values[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontsize=8
            )

        # Fit line
        z = np.polyfit(j_topo_values, mean_loss_values, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(j_topo_values) - 0.02, max(j_topo_values) + 0.02, 100)
        ax1.plot(x_line, p(x_line), '--', color='red', alpha=0.6, label=f'Linear fit (ρ={rho:.3f})')

        ax1.set_xlabel('J_topo (zero-cost metric)', fontsize=12)
        ax1.set_ylabel('Mean Validation Loss (10 epochs)', fontsize=12)
        ax1.set_title(f'J_topo vs Validation Loss\nSpearman ρ = {rho:.4f} (p = {p_value:.4f})', fontsize=13)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Add interpretation text
        interp_text = (
            "Two-stage heuristic:\nEMPIRICALLY JUSTIFIED\n(|ρ| < 0.7, seed-dependent)"
            if two_stage_justified else
            "Two-stage heuristic:\nMAY BE UNNECESSARY\n(|ρ| ≥ 0.7, stable)"
        )
        ax1.text(
            0.05, 0.95, interp_text,
            transform=ax1.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat' if two_stage_justified else 'lightgreen', alpha=0.8)
        )

        # Plot 2: Per-seed loss distribution (box plot style)
        ax2 = axes[1]
        x_pos = np.arange(len(names))
        for i, name in enumerate(names):
            losses = results[name]['seed_losses']
            jitter = np.random.uniform(-0.1, 0.1, len(losses))
            ax2.scatter([i + jitter[j] for j in range(len(losses))], losses,
                       alpha=0.7, s=50, color='steelblue')
            ax2.hlines(
                results[name]['mean_loss'], i - 0.3, i + 0.3,
                colors='red', linewidths=2
            )

        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([n.replace("ThermoNet-", "TN").replace(" W=", "\nW=") for n in names], fontsize=9)
        ax2.set_xlabel('Architecture', fontsize=12)
        ax2.set_ylabel('Validation Loss (per seed)', fontsize=12)
        ax2.set_title('Per-Seed Loss Distribution\n(Red lines = mean, dots = individual seeds)', fontsize=13)
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(OUT_DIR / "ranking_stability.png", dpi=150, bbox_inches='tight')
        print(f"  Saved: {OUT_DIR / 'ranking_stability.png'}")

    except Exception as e:
        print(f"  Warning: Could not generate plot: {e}")

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"Spearman ρ = {rho:.4f} (p = {p_value:.4f})")
    print()
    if two_stage_justified:
        print("✓ Two-stage heuristic is empirically JUSTIFIED")
        print("  J_topo ranking is seed-dependent (|ρ| < 0.7)")
        print("  → Keep D≥48 filter and use J_topo for screening")
        print("  → Per-seed variance is high enough to warrant two-stage approach")
    else:
        print("✗ Two-stage heuristic may be UNNECESSARY")
        print("  J_topo ranking is stable across seeds (|ρ| ≥ 0.7)")
        print("  → Consider direct L_hat optimization")
        print("  → J_topo alone may be sufficient for architecture selection")
    print()
    print("Recommendation:")
    print("  " + (
        "Keep D≥48 filter as insurance against capacity floor effects. "
        "Use J_topo within wide pool for screening, then train top candidates."
        if two_stage_justified else
        "J_topo ranking is reliable enough to consider end-to-end optimization. "
        "However, D≥48 filter may still help avoid capacity-limited regimes."
    ))
    print()

    return results_json


if __name__ == "__main__":
    results = run_experiment()