#!/usr/bin/env python3
"""
ThermoRG Phase B — Utility Function Optimization
================================================

验证目标：
1. 实现效用函数优化：U(A) = -E_floor(J_topo) + λ·β(γ, J_topo)
2. 模块添加作为 (J_topo, γ) 空间的转移
3. 选择使 ΔU 最大的 action
4. 对比效用函数 vs 规则判断

Run: python3 stepwise_simulation_v4_utility.py
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

# ─── GROUND TRUTH PARAMETERS ──────────────────────────────────────────────────

BETA_BN = 0.368
BETA_NONE = 0.180
ALPHA_BASE = 0.93
ALPHA_BN = 1.71
E_FLOOR_BN = 0.181
E_FLOOR_NONE = 0.276
GAMMA_BN = 2.36
GAMMA_NONE = 3.36
GAMMA_C = 2.0

# From Phase A correlation: E_floor = 0.84 + 0.83 * (J_topo - 0.35)
E_FLOOR_INTERCEPT = 0.84
E_FLOOR_SLOPE = 0.83

# Weighting factor for β in utility
LAMBDA_BETA = 10.0


@dataclass
class ArchConfig:
    name: str
    width: int
    depth: int
    skip: bool
    norm: str

    @property
    def beta_0(self):
        return BETA_BN if self.norm == 'bn' else BETA_NONE

    @property
    def gamma_init(self):
        return GAMMA_BN if self.norm == 'bn' else GAMMA_NONE

    def j_topo(self):
        j = 0.25
        if self.skip:
            j += 0.35
        j += (self.depth - 5) * 0.02
        j -= (self.width / 64 - 1) * 0.10
        if self.norm == 'bn':
            j += 0.05
        return np.clip(j, 0.1, 0.9)

    def e_floor(self):
        """E_floor from J_topo correlation."""
        return E_FLOOR_INTERCEPT + E_FLOOR_SLOPE * (self.j_topo() - 0.35)

    def cooling_factor(self, gamma):
        return GAMMA_C / (GAMMA_C + gamma) * np.exp(-gamma / GAMMA_C)

    def compute_beta_eff(self, gamma):
        return self.beta_0 * self.cooling_factor(gamma)


@dataclass
class Action:
    name: str
    description: str
    delta_j: float  # Change in J_topo
    delta_gamma: float  # Change in gamma
    cost: float = 1.0  # Computational cost


# ─── ACTION LIBRARY ──────────────────────────────────────────────────────────

def get_available_actions(arch: ArchConfig) -> List[Action]:
    """Get available actions for current architecture."""
    actions = []

    # Add BN/LN (cooling — reduces gamma)
    if arch.norm == 'none':
        actions.append(Action(
            name='add_bn',
            description='Add BatchNorm (cooling)',
            delta_j=0.05,  # BN slightly increases J_topo
            delta_gamma=-1.0,  # Strong cooling
        ))
        actions.append(Action(
            name='add_ln',
            description='Add LayerNorm (moderate cooling)',
            delta_j=0.02,
            delta_gamma=-0.7,
        ))
    elif arch.norm in ['bn', 'ln']:
        actions.append(Action(
            name='remove_norm',
            description='Remove normalization',
            delta_j=-0.05,
            delta_gamma=0.8,
        ))

    # Add skip (increases J_topo, helps gradient flow)
    if not arch.skip:
        actions.append(Action(
            name='add_skip',
            description='Add skip connection',
            delta_j=0.35,
            delta_gamma=-0.2,  # Skip also helps cooling slightly
        ))
    else:
        actions.append(Action(
            name='remove_skip',
            description='Remove skip connection',
            delta_j=-0.35,
            delta_gamma=0.2,
        ))

    # Adjust width (affects J_topo)
    if arch.width < 128:
        actions.append(Action(
            name='increase_width',
            description='Increase width (reduce J_topo)',
            delta_j=-0.15,
            delta_gamma=0.0,
        ))
    if arch.width > 16:
        actions.append(Action(
            name='decrease_width',
            description='Decrease width (increase J_topo)',
            delta_j=0.15,
            delta_gamma=0.0,
        ))

    return actions


# ─── UTILITY FUNCTION ────────────────────────────────────────────────────────

def compute_utility(arch: ArchConfig, gamma_current: float) -> float:
    """
    Compute utility U(A) = -E_floor(J_topo) + λ·β_eff(γ, J_topo)

    Higher is better:
    - Lower E_floor → higher utility
    - Higher β_eff → higher utility
    """
    j = arch.j_topo()
    e_floor = arch.e_floor()
    beta_eff = arch.compute_beta_eff(gamma_current)

    utility = -e_floor + LAMBDA_BETA * beta_eff
    return utility


def compute_delta_utility(arch: ArchConfig, action: Action, gamma_current: float) -> float:
    """
    Compute ΔU = U(A_after) - U(A_before) for taking an action.
    """
    # Current state
    u_before = compute_utility(arch, gamma_current)

    # Create hypothetical new architecture
    new_arch = ArchConfig(
        name=arch.name + '+' + action.name,
        width=max(16, min(128, arch.width + (20 if 'width' in action.name and 'increase' in action.name else -20 if 'width' in action.name else 0))),
        depth=arch.depth,
        skip=arch.skip if 'skip' not in action.name else (action.name == 'add_skip'),
        norm=arch.norm if 'norm' not in action.name else ('bn' if action.name == 'add_bn' else 'ln' if action.name == 'add_ln' else 'none'),
    )

    # New gamma
    new_gamma = max(0.5, gamma_current + action.delta_gamma)

    # New utility
    u_after = compute_utility(new_arch, new_gamma)

    return u_after - u_before


# ─── SIMULATION ─────────────────────────────────────────────────────────────

def simulate_training(arch: ArchConfig, n_epochs=50) -> dict:
    """Simulate short training to get gamma trajectory."""
    gamma_evo = np.zeros(n_epochs)
    gamma_init = arch.gamma_init

    for epoch in range(n_epochs):
        t = epoch / 50.0
        if t < 0.2:  # Phase 1: cooling
            gamma_evo[epoch] = gamma_init * (1 - 0.3 * t / 0.2)
        elif t < 0.6:  # Phase 2: stable
            gamma_evo[epoch] = gamma_init * (0.7 + 0.1 * (t - 0.2) / 0.4)
        else:  # Phase 3: slight increase
            gamma_evo[epoch] = gamma_init * (0.8 + 0.2 * (t - 0.6) / 0.4)

    return {
        'gamma_evo': gamma_evo,
        'gamma_init': gamma_init,
        'gamma_final': gamma_evo[-1],
    }


def utility_based_search(initial_arch: ArchConfig, max_steps=5) -> dict:
    """
    Utility-based architecture search.

    Algorithm:
    1. Start with minimal baseline
    2. For each step:
       a. Simulate short training, get current gamma
       b. Get available actions
       c. Compute ΔU for each action
       d. Select action with max ΔU (if positive)
       e. Apply action
    3. Stop when no positive ΔU or max steps reached
    """
    print(f"\n{'='*70}")
    print(f"UTILITY-BASED SEARCH: {initial_arch.name}")
    print(f"{'='*70}")

    arch = initial_arch
    history = []
    step = 0

    while step < max_steps:
        # Simulate short training
        sim = simulate_training(arch)
        gamma_current = np.mean(sim['gamma_evo'][:10])  # Early gamma

        # Compute current utility
        u_current = compute_utility(arch, gamma_current)

        print(f"\nStep {step}: {arch.name}")
        print(f"  J_topo = {arch.j_topo():.4f}")
        print(f"  γ (early) = {gamma_current:.4f}")
        print(f"  E_floor = {arch.e_floor():.4f}")
        print(f"  β_eff = {arch.compute_beta_eff(gamma_current):.4f}")
        print(f"  U = {u_current:.4f}")

        # Get available actions
        actions = get_available_actions(arch)

        # Compute ΔU for each action
        action_scores = []
        for action in actions:
            delta_u = compute_delta_utility(arch, action, gamma_current)
            action_scores.append((delta_u, action))

        # Sort by ΔU (descending)
        action_scores.sort(key=lambda x: x[0], reverse=True)

        print(f"\n  Available actions (sorted by ΔU):")
        for delta_u, action in action_scores:
            sign = '+' if delta_u > 0 else ''
            print(f"    {action.name:<20} ΔU={sign}{delta_u:.4f} ({action.description})")

        # Select best action (if positive)
        if action_scores and action_scores[0][0] > 0:
            best_delta_u, best_action = action_scores[0]
            print(f"\n  → Selected: {best_action.name} (ΔU={best_delta_u:+.4f})")

            # Apply action (simplified — just update arch)
            if best_action.name == 'add_bn':
                arch = ArchConfig(arch.name + '+BN', arch.width, arch.depth, arch.skip, 'bn')
            elif best_action.name == 'add_ln':
                arch = ArchConfig(arch.name + '+LN', arch.width, arch.depth, arch.skip, 'ln')
            elif best_action.name == 'remove_norm':
                arch = ArchConfig(arch.name.replace('+BN', '').replace('+LN', ''), arch.width, arch.depth, arch.skip, 'none')
            elif best_action.name == 'add_skip':
                arch = ArchConfig(arch.name + '+Skip', arch.width, arch.depth, True, arch.norm)
            elif best_action.name == 'remove_skip':
                arch = ArchConfig(arch.name.replace('+Skip', ''), arch.width, arch.depth, False, arch.norm)
            elif best_action.name == 'increase_width':
                arch = ArchConfig(arch.name, min(128, arch.width + 32), arch.depth, arch.skip, arch.norm)
            elif best_action.name == 'decrease_width':
                arch = ArchConfig(arch.name, max(16, arch.width - 32), arch.depth, arch.skip, arch.norm)

            history.append({
                'step': step,
                'action': best_action.name,
                'delta_u': best_delta_u,
                'arch': arch,
                'utility': u_current,
            })
            step += 1
        else:
            print(f"\n  → No positive ΔU found. Stopping.")
            break

    return {
        'initial_arch': initial_arch,
        'final_arch': arch,
        'history': history,
    }


def rule_based_search(initial_arch: ArchConfig, max_steps=5) -> dict:
    """
    Rule-based architecture search (heuristic thresholds).
    """
    print(f"\n{'='*70}")
    print(f"RULE-BASED SEARCH: {initial_arch.name}")
    print(f"{'='*70}")

    arch = initial_arch
    history = []
    step = 0

    # Thresholds
    GAMMA_HIGH = 3.0
    BETA_LOW = 0.1
    J_TOPO_LOW = 0.3
    J_TOPO_HIGH = 0.6

    while step < max_steps:
        sim = simulate_training(arch)
        gamma_current = np.mean(sim['gamma_evo'][:10])
        beta_eff = arch.compute_beta_eff(gamma_current)
        j_topo = arch.j_topo()

        print(f"\nStep {step}: {arch.name}")
        print(f"  J_topo = {j_topo:.4f}, γ = {gamma_current:.4f}, β_eff = {beta_eff:.4f}")

        # Apply rules (priority order)
        action_taken = None

        if gamma_current > GAMMA_HIGH:
            action_taken = 'add_bn'
            arch = ArchConfig(arch.name + '+BN', arch.width, arch.depth, arch.skip, 'bn')
            print(f"  → Rule: γ high ({gamma_current:.3f} > {GAMMA_HIGH}) → Add BN")

        elif j_topo < J_TOPO_LOW and beta_eff < BETA_LOW:
            action_taken = 'add_skip'
            arch = ArchConfig(arch.name + '+Skip', arch.width, arch.depth, True, arch.norm)
            print(f"  → Rule: J_topo low + β low → Add Skip")

        elif beta_eff < BETA_LOW and j_topo > J_TOPO_HIGH:
            action_taken = 'add_ln'
            arch = ArchConfig(arch.name + '+LN', arch.width, arch.depth, arch.skip, 'ln')
            print(f"  → Rule: β low + J_topo high → Add LN")

        elif j_topo > J_TOPO_HIGH:
            action_taken = 'decrease_width'
            arch = ArchConfig(arch.name, max(16, arch.width - 32), arch.depth, arch.skip, arch.norm)
            print(f"  → Rule: J_topo high → Decrease Width")

        else:
            print(f"  → No rule triggered. Stopping.")
            break

        if action_taken:
            history.append({'step': step, 'action': action_taken, 'arch': arch})
            step += 1

    return {
        'initial_arch': initial_arch,
        'final_arch': arch,
        'history': history,
    }


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("ThermoRG Phase B — Utility Function vs Rule-Based Search")
    print("="*70)

    # Start with minimal baseline (Conv-ReLU-Conv, no skip, no norm)
    baseline = ArchConfig("Baseline", width=32, depth=3, skip=False, norm='none')

    print(f"\nStarting architecture: {baseline.name}")
    print(f"  J_topo = {baseline.j_topo():.4f}")
    print(f"  γ_init = {baseline.gamma_init:.4f}")
    print(f"  β_0 = {baseline.beta_0:.4f}")
    print(f"  E_floor = {baseline.e_floor():.4f}")

    # Run both search methods
    utility_result = utility_based_search(baseline, max_steps=5)
    rule_result = rule_based_search(baseline, max_steps=5)

    # Compare results
    print("\n" + "="*70)
    print("COMPARISON: Utility-Based vs Rule-Based")
    print("="*70)

    print(f"\n{'Metric':<25} {'Utility-Based':<30} {'Rule-Based':<30}")
    print("-" * 85)
    print(f"{'Initial Arch':<25} {baseline.name:<30} {baseline.name:<30}")
    print(f"{'Final Arch':<25} {utility_result['final_arch'].name:<30} {rule_result['final_arch'].name:<30}")
    print(f"{'Final J_topo':<25} {utility_result['final_arch'].j_topo():<30.4f} {rule_result['final_arch'].j_topo():<30.4f}")
    print(f"{'Final E_floor':<25} {utility_result['final_arch'].e_floor():<30.4f} {rule_result['final_arch'].e_floor():<30.4f}")

    # Compute final utilities
    gamma_utility = simulate_training(utility_result['final_arch'])['gamma_init']
    gamma_rule = simulate_training(rule_result['final_arch'])['gamma_init']

    u_utility = compute_utility(utility_result['final_arch'], gamma_utility)
    u_rule = compute_utility(rule_result['final_arch'], gamma_rule)

    print(f"{'Final Utility U':<25} {u_utility:<30.4f} {u_rule:<30.4f}")
    print(f"{'Steps taken':<25} {len(utility_result['history']):<30} {len(rule_result['history']):<30}")

    # History
    print(f"\n{'='*70}")
    print("SEARCH HISTORY")
    print("="*70)

    print("\nUtility-Based:")
    for h in utility_result['history']:
        print(f"  Step {h['step']}: {h['action']} (ΔU={h['delta_u']:+.4f})")

    print("\nRule-Based:")
    for h in rule_result['history']:
        print(f"  Step {h['step']}: {h['action']}")

    # Summary
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    print("""
1. Utility-based uses ΔU to quantitatively select the BEST action
2. Rule-based uses fixed thresholds (GAMMA_HIGH=3.0, etc.)
3. Utility-based adapts to current state without thresholds
4. Both should converge to similar architectures if well-calibrated

5. Advantages of Utility-Based:
   - No manual threshold tuning
   - Naturally balances E_floor vs β trade-off via λ
   - Quantitatively optimal (max ΔU)

6. Advantages of Rule-Based:
   - More interpretable ("γ high → add BN")
   - Easier to debug
   - Works when actions have known effects
""")


if __name__ == '__main__':
    main()
