#!/usr/bin/env python3
"""
ThermoRG Phase B — Stepwise Algorithm Training Phase Simulation
=========================================================

Simulates normal training phases to validate the stepwise algorithm.

Key验证目标:
1. Three training phases: rapid → slow → plateau
2. Distinguish overfitting vs honest learning
3. Effect of adding modules (BN, skip) on γ and β

Phase 1: Rapid descent (easy patterns)
Phase 2: Slow descent (complex manifold)
Phase 3: Plateau (honest learning OR overfitting)
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional
import sys

# ─── GROUND TRUTH MODEL ────────────────────────────────────────────────────────

# From Phase A / S1 v3 empirical values
J_TOPO_INTERCEPT = 0.84
J_TOPO_SLOPE = 0.83

BETA_NONE = 0.180
BETA_BN = 0.368
BETA_LN = 0.370

ALPHA_BASE = 0.93
ALPHA_BN = 1.71

E_FLOOR_NONE = 0.276
E_FLOOR_BN = 0.181

# Training phase parameters
GAMMA_NONE = 3.36
GAMMA_BN = 2.36


# ─── SIMULATION MODELS ────────────────────────────────────────────────────────

@dataclass
class Architecture:
    width: int
    depth: int
    skip: bool
    norm: str  # 'none', 'bn', 'ln'

    def compute_j_topo(self, noise=0.02):
        j = 0.35
        if self.skip:
            j += 0.15
        j += (self.depth - 5) * 0.03
        j -= (self.width / 64) * 0.10
        if self.norm == 'bn':
            j += 0.10
        j += np.random.normal(0, noise)
        return float(np.clip(j, 0.05, 0.95))

    def compute_e_floor(self, j_topo):
        base = J_TOPO_INTERCEPT + J_TOPO_SLOPE * (j_topo - 0.35)
        if self.norm == 'bn':
            base -= 0.20
        elif self.norm == 'ln':
            base += 0.05
        return float(np.clip(base + np.random.normal(0, 0.02), 0.05, 1.5))

    def compute_beta(self):
        return {'none': BETA_NONE, 'bn': BETA_BN, 'ln': BETA_LN}[self.norm]

    def compute_gamma(self):
        return {'none': GAMMA_NONE, 'bn': GAMMA_BN, 'ln': GAMMA_NONE}[self.norm]


def compute_loss(D_max, arch, epoch, training_phase='normal', noise_scale=1.0):
    """Simulate loss at given epoch.

    training_phase options:
    - 'normal': honest learning (three phases)
    - 'overfitting': val loss increases after epoch ~50
    - 'difficulty': both decrease slowly (manifold learning)
    """
    beta = arch.compute_beta()
    e_floor = arch.compute_e_floor(arch.compute_j_topo())
    alpha = ALPHA_BN if arch.norm == 'bn' else ALPHA_BASE

    # Base loss trajectory
    t = epoch / 200.0  # normalized time

    if training_phase == 'normal':
        # Three phases
        if t < 0.1:  # Phase 1: rapid descent
            rate = 2.0 * beta
        elif t < 0.5:  # Phase 2: slow descent
            rate = 0.5 * beta
        else:  # Phase 3: plateau
            rate = 0.05 * beta

        loss = e_floor + alpha * np.exp(-rate * epoch)
        loss *= (1 + np.random.normal(0, 0.01 * noise_scale))

    elif training_phase == 'overfitting':
        # Train keeps decreasing, val increases after epoch ~50
        if epoch < 50:
            loss = e_floor + alpha * np.exp(-1.5 * beta * epoch)
        else:
            train_contribution = alpha * np.exp(-1.5 * beta * 50) * np.exp(-0.01 * (epoch - 50))
            val_increase = 0.05 * (1 - np.exp(-0.05 * (epoch - 50)))
            loss = e_floor + train_contribution + val_increase
        loss *= (1 + np.random.normal(0, 0.01 * noise_scale))

    elif training_phase == 'difficulty':
        # Both decrease slowly - learning difficult manifold
        loss = e_floor + alpha * np.exp(-0.1 * beta * epoch)
        loss *= (1 + np.random.normal(0, 0.02 * noise_scale))

    return float(np.clip(loss, e_floor * 0.9, 2.0))


def simulate_training(arch, n_epochs=200, training_phase='normal'):
    """Simulate full training trajectory."""
    losses = []
    for epoch in range(n_epochs):
        loss = compute_loss(96, arch, epoch, training_phase)
        losses.append(loss)
    return np.array(losses)


def simulate_with_gamma(arch, n_epochs=200, training_phase='normal'):
    """Simulate training with gamma tracking."""
    losses = []
    gammas = []

    # Initial gamma at epoch 0
    gamma = arch.compute_gamma()
    base_gamma = gamma

    for epoch in range(n_epochs):
        loss = compute_loss(96, arch, epoch, training_phase)
        losses.append(loss)

        # Gamma evolves during training
        # Phase 1: gamma decreases (cooling from initialization)
        # Phase 2: gamma stable or slight increase
        # Phase 3: gamma increases (network "heats up" near convergence)
        t = epoch / 200.0

        if t < 0.1:  # Phase 1: cooling
            gamma = base_gamma * (1 - 0.3 * t / 0.1)
        elif t < 0.5:  # Phase 2: stable
            gamma = base_gamma * 0.7
        else:  # Phase 3: slight increase
            gamma = base_gamma * (0.7 + 0.2 * (t - 0.5) / 0.5)

        gamma += np.random.normal(0, 0.02)
        gammas.append(float(gamma))

    return np.array(losses), np.array(gammas)


# ─── PLATEAU DETECTION ────────────────────────────────────────────────────────

def detect_plateau_type(train_losses, val_losses, window=20):
    """Detect what type of plateau we're in.

    Returns:
        'honest': train↓ val↓ slowly (continue training)
        'overfitting': train↓ val↑ (add regularization)
        'difficulty': train↓ val plateau (may need more capacity)
        None: not in plateau
    """
    if len(train_losses) < window or len(val_losses) < window:
        return None

    train_recent = np.mean(train_losses[-window:])
    train_earlier = np.mean(train_losses[-2*window:-window])
    val_recent = np.mean(val_losses[-window:])
    val_earlier = np.mean(val_losses[-2*window:-window])

    train_change = (train_earlier - train_recent) / train_earlier
    val_change = (val_earlier - val_recent) / val_earlier

    # Not in plateau if either is still improving >1%
    if train_change > 0.01 or val_change > 0.01:
        return None

    # Overfitting: val getting worse while train improving
    if val_change < -0.01 and train_change > 0:
        return 'overfitting'

    # Honest plateau: both decreasing (slowly)
    if val_change > 0 and train_change > 0:
        return 'honest'

    # Difficulty: train decreasing, val plateau
    if train_change > 0 and abs(val_change) < 0.01:
        return 'difficulty'

    return None


# ─── SIMULATION EXPERIMENTS ──────────────────────────────────────────────────

def experiment_three_phases():
    """Show three training phases for different architectures."""
    print("=" * 70)
    print("EXPERIMENT 1: Three Training Phases")
    print("=" * 70)
    print()

    archs = [
        Architecture(width=64, depth=5, skip=False, norm='none'),
        Architecture(width=64, depth=5, skip=True, norm='bn'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for idx, arch in enumerate(archs):
        losses = simulate_training(arch, n_epochs=200, training_phase='normal')

        epochs = np.arange(200)

        # Mark phases
        axes[idx].plot(epochs, losses, 'b-', label='Loss')
        axes[idx].axvspan(0, 20, alpha=0.2, color='green', label='Phase 1: Rapid')
        axes[idx].axvspan(20, 100, alpha=0.2, color='yellow', label='Phase 2: Slow')
        axes[idx].axvspan(100, 200, alpha=0.2, color='red', label='Phase 3: Plateau')

        axes[idx].set_xlabel('Epoch')
        axes[idx].set_ylabel('Loss')
        axes[idx].set_title(f'{arch.norm} + skip={arch.skip}\nJ_topo={arch.compute_j_topo():.3f}, β={arch.compute_beta():.3f}')
        axes[idx].legend(fontsize=8)
        axes[idx].grid(True, alpha=0.3)

        print(f"{'Config':<20} {'J_topo':<10} {'β':<10} {'Final Loss':<12}")
        print("-" * 50)
        print(f"{arch.norm+'_skip='+str(arch.skip):<20} {arch.compute_j_topo():<10.3f} {arch.compute_beta():<10.3f} {losses[-1]:<12.4f}")

    plt.tight_layout()
    plt.savefig('/home/node/.openclaw/workspace/github_staging/ThermoRG-NN/experiments/phase_b/stepwise_phase1_three_phases.png', dpi=150)
    print("\nSaved: stepwise_phase1_three_phases.png")
    return fig


def experiment_plateau_types():
    """Show different plateau types."""
    print()
    print("=" * 70)
    print("EXPERIMENT 2: Plateau Types")
    print("=" * 70)
    print()

    arch = Architecture(width=64, depth=5, skip=True, norm='bn')

    phases = ['normal', 'overfitting', 'difficulty']
    phase_names = ['Normal (Honest)', 'Overfitting', 'Difficulty (Manifold)']

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    plateau_types_detected = []

    for idx, (phase, name) in enumerate(zip(phases, phase_names)):
        # Simulate with train and val
        np.random.seed(42)
        train_losses = simulate_training(arch, n_epochs=200, training_phase=phase)

        # Val loss is same for honest/difficulty, worse for overfitting
        if phase == 'overfitting':
            val_losses = train_losses + np.array([
                0 if e < 50 else 0.03 * (1 - np.exp(-0.05 * (e - 50)))
                for e in range(200)
            ])
        else:
            val_losses = train_losses + np.random.normal(0.01, 0.005, 200)

        epochs = np.arange(200)
        axes[idx].plot(epochs, train_losses, 'b-', label='Train')
        axes[idx].plot(epochs, val_losses, 'r--', label='Val')
        axes[idx].set_xlabel('Epoch')
        axes[idx].set_ylabel('Loss')
        axes[idx].set_title(f'{name}')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

        # Detect plateau
        detected = detect_plateau_type(train_losses[-50:], val_losses[-50:])
        plateau_types_detected.append(detected)
        print(f"{name:<20} → Detected: {detected}")

    plt.tight_layout()
    plt.savefig('/home/node/.openclaw/workspace/github_staging/ThermoRG-NN/experiments/phase_b/stepwise_phase2_plateau_types.png', dpi=150)
    print("\nSaved: stepwise_phase2_plateau_types.png")
    return fig


def experiment_bn_effect():
    """Show how BN affects training phases."""
    print()
    print("=" * 70)
    print("EXPERIMENT 3: BN Effect on Training")
    print("=" * 70)
    print()

    configs = [
        ('None, No Skip', Architecture(width=64, depth=5, skip=False, norm='none')),
        ('BN, No Skip', Architecture(width=64, depth=5, skip=False, norm='bn')),
        ('None, Skip', Architecture(width=64, depth=5, skip=True, norm='none')),
        ('BN, Skip', Architecture(width=64, depth=5, skip=True, norm='bn')),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    results = []

    for idx, (name, arch) in enumerate(configs):
        losses, gammas = simulate_with_gamma(arch, n_epochs=200, training_phase='normal')
        epochs = np.arange(200)

        row, col = idx // 2, idx % 2
        axes[row, col].plot(epochs, losses, 'b-', label='Loss')
        ax2 = axes[row, col].twinx()
        ax2.plot(epochs, gammas, 'r-', alpha=0.5, label='γ')
        axes[row, col].set_xlabel('Epoch')
        axes[row, col].set_ylabel('Loss', color='b')
        ax2.set_ylabel('γ', color='r')
        axes[row, col].set_title(f'{name}\nJ={arch.compute_j_topo():.3f}, β={arch.compute_beta():.3f}, γ={arch.compute_gamma():.3f}')
        axes[row, col].grid(True, alpha=0.3)

        results.append({
            'name': name,
            'j_topo': arch.compute_j_topo(),
            'beta': arch.compute_beta(),
            'gamma': arch.compute_gamma(),
            'final_loss': losses[-1],
            'loss_at_50': losses[49],
        })

    plt.tight_layout()
    plt.savefig('/home/node/.openclaw/workspace/github_staging/ThermoRG-NN/experiments/phase_b/stepwise_phase3_bn_effect.png', dpi=150)
    print("\nSaved: stepwise_phase3_bn_effect.png")

    print()
    print(f"{'Config':<20} {'J_topo':<10} {'β':<10} {'γ':<10} {'Loss@50':<10} {'Final':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<20} {r['j_topo']:<10.3f} {r['beta']:<10.3f} {r['gamma']:<10.3f} {r['loss_at_50']:<10.4f} {r['final_loss']:<10.4f}")

    return fig


def experiment_j_topo_range():
    """Show that J_topo has optimal range, not minimized."""
    print()
    print("=" * 70)
    print("EXPERIMENT 4: J_topo Optimal Range")
    print("=" * 70)
    print()

    # Compare architectures with different J_topo
    archs = [
        Architecture(width=8, depth=3, skip=False, norm='none'),   # Low J
        Architecture(width=64, depth=3, skip=False, norm='none'), # Medium J
        Architecture(width=64, depth=5, skip=True, norm='bn'),    # High J + skip + BN
        Architecture(width=64, depth=9, skip=True, norm='bn'),    # Very high J
    ]

    print(f"{'Config':<25} {'J_topo':<10} {'β':<10} {'Final Loss':<12}")
    print("-" * 60)

    for arch in archs:
        losses = simulate_training(arch, n_epochs=200, training_phase='normal')
        j = arch.compute_j_topo()
        beta = arch.compute_beta()
        final = losses[-1]
        print(f"w={arch.width} d={arch.depth} skip={arch.skip} norm={arch.norm:<5} "
              f"{j:<10.3f} {beta:<10.3f} {final:<12.4f}")

    print()
    print("Key observation: High J_topo + skip + BN can achieve similar/better")
    print("results than low J_topo alone (ResNet-18 style)")


# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("ThermoRG Stepwise Algorithm — Training Phase Simulation")
    print("=" * 70)
    print()

    # Run experiments
    experiment_three_phases()
    experiment_plateau_types()
    experiment_bn_effect()
    experiment_j_topo_range()

    print()
    print("=" * 70)
    print("All experiments complete. Check experiments/phase_b/ for plots.")


if __name__ == '__main__':
    main()
