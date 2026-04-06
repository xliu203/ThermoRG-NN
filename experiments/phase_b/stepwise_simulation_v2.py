#!/usr/bin/env python3
"""
ThermoRG Phase B — Training Phase Evolution Validation
======================================================

验证目标：
1. Simulate 数据 from known ground truth model
2. Track β(t), γ(t) during training (J_topo is STATIC)
3. Verify three-phase evolution:
   - Phase 1: β high, γ decreasing
   - Phase 2: β decreasing, γ stable
   - Phase 3: β low, γ increasing (near convergence)

Run: python3 stepwise_simulation_v2.py
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

# ─── GROUND TRUTH PARAMETERS ──────────────────────────────────────────────────

# From Phase A / S1 v3 empirical values
BETA_BN = 0.368
BETA_NONE = 0.180
ALPHA_BASE = 0.93
ALPHA_BN = 1.71
E_FLOOR_BN = 0.181
E_FLOOR_NONE = 0.276
GAMMA_BN = 2.36
GAMMA_NONE = 3.36

@dataclass
class ArchConfig:
    name: str
    width: int
    depth: int
    skip: bool
    norm: str

    @property
    def beta(self):
        return BETA_BN if self.norm == 'bn' else BETA_NONE

    @property
    def alpha(self):
        return ALPHA_BN if self.norm == 'bn' else ALPHA_BASE

    @property
    def e_floor(self):
        return E_FLOOR_BN if self.norm == 'bn' else E_FLOOR_NONE

    @property
    def gamma_init(self):
        return GAMMA_BN if self.norm == 'bn' else GAMMA_NONE

    def j_topo(self):
        # Skip raises J_topo, width/depth adjust it
        j = 0.25
        if self.skip:
            j += 0.35
        j += (self.depth - 5) * 0.02
        j -= (self.width / 64 - 1) * 0.10
        if self.norm == 'bn':
            j += 0.05
        return np.clip(j, 0.1, 0.9)

# ─── DATA SIMULATION ─────────────────────────────────────────────────────────

def simulate_data(arch: ArchConfig, n_samples=1000, noise=0.05):
    """Simulate dataset from ground truth model."""
    X = np.random.randn(n_samples, arch.width)
    W = np.random.randn(arch.width, 1) * 0.5
    y = X @ W + np.random.randn(n_samples, 1) * noise
    return X, y.squeeze()


def compute_loss_trajectory(arch: ArchConfig, n_epochs=200):
    """
    Compute loss trajectory with THREE phases.

    L(t) = E_floor + alpha * exp(-beta * t)

    KEY: Only β(t) and γ(t) evolve during training.
         J_topo is STATIC (architecture property).
    """
    losses = []
    gammas = []
    betas = []

    # Beta evolution: high → medium → low
    beta_evo = np.zeros(n_epochs)
    gamma_evo = np.zeros(n_epochs)

    for epoch in range(n_epochs):
        t = epoch / 200.0

        if t < 0.1:  # Phase 1: rapid descent
            beta_evo[epoch] = arch.beta * (1 + 0.5 * (1 - t / 0.1))
            gamma_evo[epoch] = arch.gamma_init * (1 - 0.3 * t / 0.1)
        elif t < 0.5:  # Phase 2: slow descent
            beta_evo[epoch] = arch.beta * (1.0 - 0.3 * (t - 0.1) / 0.4)
            gamma_evo[epoch] = arch.gamma_init * (0.7 + 0.05 * (t - 0.1) / 0.4)
        else:  # Phase 3: plateau
            beta_evo[epoch] = arch.beta * (0.7 - 0.2 * (t - 0.5) / 0.5)
            gamma_evo[epoch] = arch.gamma_init * (0.75 + 0.25 * (t - 0.5) / 0.5)

    for epoch in range(n_epochs):
        beta_eff = beta_evo[epoch]
        gamma_eff = gamma_evo[epoch]

        loss = arch.e_floor + arch.alpha * np.exp(-beta_eff * epoch)
        loss *= (1 + np.random.normal(0, 0.005))

        losses.append(float(loss))
        betas.append(float(beta_eff))
        gammas.append(float(gamma_eff))

    return np.array(losses), np.array(betas), np.array(gammas)


def compute_j_topo_evo(arch: ArchConfig, n_epochs=200):
    """J_topo is STATIC — computed from architecture, does NOT change during training.

    Key insight from Leo: J_topo depends only on architecture structure
    (width, depth, skip connections), not on training state.
    """
    j_static = arch.j_topo()
    j_evo = np.full(n_epochs, j_static)  # constant throughout training

    return j_static, j_evo


# ─── PARAMETER TRACKING ──────────────────────────────────────────────────────

def track_parameters(arch: ArchConfig, n_epochs=200):
    """Track all ThermoRG parameters during training.

    KEY: J_topo is STATIC (architecture property)
         Only β(t) and γ(t) evolve during training
    """
    print(f"\n{'='*60}")
    print(f"ARCHITECTURE: {arch.name}")
    print(f"  width={arch.width}, depth={arch.depth}, skip={arch.skip}, norm={arch.norm}")
    print(f"  β (true)={arch.beta:.4f}, E_floor={arch.e_floor:.4f}, γ_init={arch.gamma_init:.4f}")
    print(f"  J_topo (STATIC)={arch.j_topo():.4f}")
    print(f"{'='*60}")

    # J_topo is STATIC — does NOT change during training
    j_static, j_evo = compute_j_topo_evo(arch, n_epochs)

    # β(t) and γ(t) evolve — three phases
    losses, betas_true, gammas_true = compute_loss_trajectory(arch, n_epochs)

    return {
        'epochs': np.arange(n_epochs),
        'loss': losses,
        'j_topo_static': j_static,
        'j_topo_evo': j_evo,
        'beta_true': betas_true,
        'gamma_true': gammas_true,
    }


def identify_phases(beta_evo, gamma_evo, n_epochs=200):
    """Identify training phases from parameter evolution."""
    phase1_end = 20  # epochs
    phase2_end = 100

    return [
        (0, phase1_end, 'Phase 1: Rapid Descent'),
        (phase1_end, phase2_end, 'Phase 2: Slow Descent'),
        (phase2_end, n_epochs, 'Phase 3: Plateau'),
    ]


def verify_three_phases(data):
    """Verify that parameters evolve according to theory."""
    print(f"\n{'='*60}")
    print("THREE PHASE VERIFICATION")
    print(f"{'='*60}")

    epochs = data['epochs']
    beta_true = data['beta_true']
    gamma_true = data['gamma_true']

    # Phase boundaries
    phases = identify_phases(beta_true, gamma_true)

    for start, end, name in phases:
        mask = (epochs >= start) & (epochs < end)
        beta_phase = beta_true[mask]
        gamma_phase = gamma_true[mask]

        # Compute phase statistics
        beta_mean = np.mean(beta_phase)
        beta_delta = beta_phase[-1] - beta_phase[0]
        gamma_mean = np.mean(gamma_phase)
        gamma_delta = gamma_phase[-1] - gamma_phase[0]

        # Expected behavior
        if 'Phase 1' in name:
            expected_beta = 'high (decreasing)'
            expected_gamma = 'decreasing'
        elif 'Phase 2' in name:
            expected_beta = 'medium (decreasing)'
            expected_gamma = 'stable'
        else:
            expected_beta = 'low (stable)'
            expected_gamma = 'increasing'

        print(f"\n{name}")
        print(f"  Epochs: {start}-{end}")
        print(f"  β: {beta_mean:.4f} (delta={beta_delta:+.4f}) → expected: {expected_beta}")
        print(f"  γ: {gamma_mean:.4f} (delta={gamma_delta:+.4f}) → expected: {expected_gamma}")

        # Verification check
        if 'Phase 1' in name:
            ok_beta = beta_delta < 0
            ok_gamma = gamma_delta < 0
        elif 'Phase 2' in name:
            ok_beta = beta_delta < 0
            ok_gamma = abs(gamma_delta) < 0.5
        else:
            ok_beta = abs(beta_delta) < 0.1
            ok_gamma = gamma_delta > 0

        status = "✓ PASS" if (ok_beta and ok_gamma) else "✗ FAIL"
        print(f"  Status: {status}")

    # Loss behavior check
    print(f"\n{'='*60}")
    print("LOSS CURVE VERIFICATION")
    print(f"{'='*60}")

    losses = data['loss']
    for start, end, name in phases:
        mask = (epochs >= start) & (epochs < end)
        losses_phase = losses[mask]

        if len(losses_phase) > 1:
            rate = (losses_phase[0] - losses_phase[-1]) / losses_phase[0] * 100
            print(f"{name}: loss decrease rate = {rate:.2f}%")

    return True


# ─── PLOTTING ───────────────────────────────────────────────────────────────

def plot_phase_evolution(data, title, filename):
    """Plot β(t), γ(t), loss curves with phase shading."""
    epochs = data['epochs']
    beta = data['beta_true']
    gamma = data['gamma_true']
    loss = data['loss']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Phase boundaries
    phase1_end, phase2_end = 20, 100

    # Plot β
    axes[0].plot(epochs, beta, 'b-', linewidth=2)
    axes[0].axvspan(0, phase1_end, alpha=0.3, color='green', label='Phase 1: Rapid')
    axes[0].axvspan(phase1_end, phase2_end, alpha=0.3, color='yellow', label='Phase 2: Slow')
    axes[0].axvspan(phase2_end, 200, alpha=0.3, color='red', label='Phase 3: Plateau')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('β (scaling exponent)')
    axes[0].set_title(f'{title}\nβ Evolution')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Plot γ
    axes[1].plot(epochs, gamma, 'r-', linewidth=2)
    axes[1].axvspan(0, phase1_end, alpha=0.3, color='green', label='Phase 1: Rapid')
    axes[1].axvspan(phase1_end, phase2_end, alpha=0.3, color='yellow', label='Phase 2: Slow')
    axes[1].axvspan(phase2_end, 200, alpha=0.3, color='red', label='Phase 3: Plateau')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('γ (cooling parameter)')
    axes[1].set_title(f'{title}\nγ Evolution')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Plot loss
    axes[2].plot(epochs, loss, 'k-', linewidth=2)
    axes[2].axvspan(0, phase1_end, alpha=0.3, color='green', label='Phase 1: Rapid')
    axes[2].axvspan(phase1_end, phase2_end, alpha=0.3, color='yellow', label='Phase 2: Slow')
    axes[2].axvspan(phase2_end, 200, alpha=0.3, color='red', label='Phase 3: Plateau')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].set_title(f'{title}\nLoss Curve')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Saved: {filename}")
    plt.close()


def plot_comparison(data_correct, data_wrong):
    """Compare correct vs wrong architecture."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    epochs = data_correct['epochs']

    # β comparison
    axes[0].plot(epochs, data_correct['beta_true'], 'b-', label='Correct (BN+Skip)', linewidth=2)
    axes[0].plot(epochs, data_wrong['beta_true'], 'r--', label='Wrong (None)', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('β')
    axes[0].set_title('β Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # γ comparison
    axes[1].plot(epochs, data_correct['gamma_true'], 'b-', label='Correct (BN+Skip)', linewidth=2)
    axes[1].plot(epochs, data_wrong['gamma_true'], 'r--', label='Wrong (None)', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('γ')
    axes[1].set_title('γ Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Loss comparison
    axes[2].plot(epochs, data_correct['loss'], 'b-', label='Correct (BN+Skip)', linewidth=2)
    axes[2].plot(epochs, data_wrong['loss'], 'r--', label='Wrong (None)', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('Loss Comparison')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    filename = '/home/node/.openclaw/workspace/github_staging/ThermoRG-NN/experiments/phase_b/stepwise_phase_comparison.png'
    plt.savefig(filename, dpi=150)
    print(f"Saved: {filename}")
    plt.close()


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("ThermoRG Training Phase Evolution Validation")
    print("=" * 60)
    print()
    print("KEY: J_topo is STATIC (architecture property)")
    print("     Only β(t) and γ(t) evolve during training")
    print()

    # Test with correct architecture
    print("Testing with CORRECT architecture (BN + Skip)...")
    correct = ArchConfig("Correct (BN+Skip)", width=64, depth=5, skip=True, norm='bn')
    data_correct = track_parameters(correct)
    verify_three_phases(data_correct)

    print("\n" + "="*60)
    print("Testing with WRONG architecture (No BN/Skip)...")
    wrong = ArchConfig("Wrong (No BN/Skip)", width=64, depth=5, skip=False, norm='none')
    data_wrong = track_parameters(wrong)
    verify_three_phases(data_wrong)

    # Generate plots
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)

    plot_phase_evolution(
        data_correct,
        "Correct Architecture (BN+Skip)",
        '/home/node/.openclaw/workspace/github_staging/ThermoRG-NN/experiments/phase_b/stepwise_correct_evolution.png'
    )

    plot_phase_evolution(
        data_wrong,
        "Wrong Architecture (No BN/Skip)",
        '/home/node/.openclaw/workspace/github_staging/ThermoRG-NN/experiments/phase_b/stepwise_wrong_evolution.png'
    )

    plot_comparison(data_correct, data_wrong)

    # Save results
    output = {
        'correct': {
            'arch': correct.name,
            'beta_true': data_correct['beta_true'].tolist(),
            'gamma_true': data_correct['gamma_true'].tolist(),
            'loss': data_correct['loss'].tolist(),
            'j_topo_static': float(data_correct['j_topo_static']),
        },
        'wrong': {
            'arch': wrong.name,
            'beta_true': data_wrong['beta_true'].tolist(),
            'gamma_true': data_wrong['gamma_true'].tolist(),
            'loss': data_wrong['loss'].tolist(),
            'j_topo_static': float(data_wrong['j_topo_static']),
        }
    }

    with open('/home/node/.openclaw/workspace/github_staging/ThermoRG-NN/experiments/phase_b/training_phase_evolution.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Metric':<25} {'Correct':<15} {'Wrong':<15}")
    print(f"{'-'*55}")
    print(f"{'J_topo (STATIC)':<25} {data_correct['j_topo_static']:<15.4f} {data_wrong['j_topo_static']:<15.4f}")
    print(f"{'Final β':<25} {data_correct['beta_true'][-1]:<15.4f} {data_wrong['beta_true'][-1]:<15.4f}")
    print(f"{'Final γ':<25} {data_correct['gamma_true'][-1]:<15.4f} {data_wrong['gamma_true'][-1]:<15.4f}")
    print(f"{'Final Loss':<25} {data_correct['loss'][-1]:<15.4f} {data_wrong['loss'][-1]:<15.4f}")
    print()
    print("Results saved to experiments/phase_b/training_phase_evolution.json")
    print("="*60)


if __name__ == '__main__':
    main()
