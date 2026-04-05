#!/usr/bin/env python3
"""
ThermoRG Phase B — Hierarchical BO Simulation (CPU-only)
========================================================

Validates HBO approach on synthetic architecture space before GPU deployment.

Ground truth (from Phase A + Phase S1 v3):
- J_topo → E_floor: r = -0.83 (lower J -> lower E_floor)
- norm_type → β: None=0.18, BN=0.37, LN=0.37
- φ_BN = 2.05

Simulates:
- L0: J_topo (zero-cost, ~1ms)
- L1: 5-epoch loss (noisy proxy for E_floor)
- L2: 50-epoch loss (rough β estimate)
- L3: 200-epoch loss (full scaling law fit)

Compares:
- Random search
- Greedy (J_topo only)
- Full HBO (J_topo + multi-fidelity GP)
"""

import numpy as np
import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod
import random

np.random.seed(42)
random.seed(42)

# ──────────────────────────────────────────────────────────────────────────────
# GROUND TRUTH MODEL (from empirical data)
# ──────────────────────────────────────────────────────────────────────────────

# Phase A ThermoNet: J_topo → E_floor
# Linear regression: E_floor = 0.458 + 1.018 * J_topo (r=0.79, p=0.02)
# Higher J_topo → HIGHER E_floor → WORSE performance
# Therefore: lower J_topo → lower E_floor → BETTER performance
J_TOPO_TO_E_INTERCEPT = 0.458
J_TOPO_TO_E_SLOPE = 1.018

# Phase S1 v3: normalization → β
BETA_NONE = 0.180
BETA_BN = 0.368
BETA_LN = 0.370

# Scaling law params
ALPHA_BASE = 0.93   # from None config fit
ALPHA_BN = 1.71    # from BN config fit
E_FLOOR_NONE = 0.276
E_FLOOR_BN = 0.181

# Noise levels (empirical from Phase A/S1)
NOISE_J_TOPO = 0.02   # J_topo measurement noise
NOISE_L1 = 0.05       # 5-epoch loss noise (relative)
NOISE_L2 = 0.02       # 50-epoch loss noise
NOISE_L3 = 0.005      # 200-epoch loss noise

# Model size range
D_VALUES = [32, 48, 64, 96]


def compute_ground_truth(j_topo: float, norm: str) -> Dict:
    """Compute ground truth scaling law params for an architecture."""
    # E_floor from J_topo (POSITIVE correlation: higher J -> higher E)
    e_floor = J_TOPO_TO_E_INTERCEPT + J_TOPO_TO_E_SLOPE * j_topo
    e_floor = np.clip(e_floor, 0.05, 1.5)

    # β from normalization type
    if norm == 'none':
        beta = BETA_NONE
        alpha = ALPHA_BASE
    elif norm == 'bn':
        beta = BETA_BN
        alpha = ALPHA_BN
    else:  # ln
        beta = BETA_LN
        alpha = ALPHA_BASE  # approximate

    # Jitter for architecture-specific variation
    beta += np.random.normal(0, 0.01)
    e_floor += np.random.normal(0, 0.03)

    return {'alpha': alpha, 'beta': beta, 'e_floor': e_floor}


def compute_loss(D: int, params: Dict, noise: float = 0.0) -> float:
    """Compute loss at given D with optional noise."""
    alpha, beta, e_floor = params['alpha'], params['beta'], params['e_floor']
    loss = alpha * (D ** (-beta)) + e_floor
    if noise > 0:
        loss *= (1 + np.random.normal(0, noise))
    return max(loss, e_floor * 0.9)


# ──────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE SPACE
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Architecture:
    """Synthetic architecture."""
    arch_id: int
    width: int          # hidden dim multiplier: 8, 16, 32, 64
    depth: int          # 3, 5, 7, 9
    skip: bool          # skip connections
    norm: str           # 'none', 'bn', 'ln'

    @property
    def n_params_M(self) -> float:
        """Approximate params in millions."""
        channels = [3, self.width, self.width*2, self.width*2]
        conv_params = sum(
            channels[i] * channels[i+1] * 9
            for i in range(len(channels)-1)
        )
        fc_params = channels[-1] * 10
        return (conv_params + fc_params) / 1e6

    def compute_j_topo(self, noise: float = NOISE_J_TOPO) -> float:
        """Compute J_topo from architecture features (with noise)."""
        # Match ThermoNet data range: roughly 0.15 to 0.65
        j = 0.25
        if self.skip:
            j += 0.08  # skip increases J_topo
        j += (self.depth - 5) * 0.025
        j -= (self.width / 64) * 0.15
        j += np.random.normal(0, noise)
        return np.clip(j, 0.12, 0.65)


def generate_arch_space(n: int = 200) -> List[Architecture]:
    """Generate random architecture space."""
    widths = [8, 16, 32, 64]
    depths = [3, 5, 7, 9]
    norms = ['none', 'bn', 'ln']

    archs = []
    for i in range(n):
        archs.append(Architecture(
            arch_id=i,
            width=random.choice(widths),
            depth=random.choice(depths),
            skip=random.choice([True, False]),
            norm=random.choice(norms)
        ))
    return archs


# ──────────────────────────────────────────────────────────────────────────────
# SURROGATE MODEL (Multi-fidelity GP — simplified)
# ──────────────────────────────────────────────────────────────────────────────

class GPRegressor:
    """Simplified Gaussian Process for architecture search.

    Uses J_topo as primary feature, norm type as secondary.
    """

    def __init__(self):
        self.observations = []  # List of (j_topo, norm, fidelity, loss)
        self.j_topo_observations = []  # L0 observations
        self.loss_observations = []     # L1+ observations

    def add_j_topo(self, arch: Architecture, j_topo: float):
        """Add zero-cost J_topo observation."""
        self.j_topo_observations.append({
            'arch': arch,
            'j_topo': j_topo
        })
        self.observations.append({
            'arch': arch,
            'j_topo': j_topo,
            'fidelity': 0,
            'loss': None
        })

    def add_loss_observation(self, arch: Architecture, j_topo: float,
                            fidelity: int, loss: float):
        """Add loss observation at given fidelity."""
        self.loss_observations.append({
            'arch': arch,
            'j_topo': j_topo,
            'fidelity': fidelity,
            'loss': loss
        })
        self.observations.append({
            'arch': arch,
            'j_topo': j_topo,
            'fidelity': fidelity,
            'loss': loss
        })

    def predict_e_floor(self, arch: Architecture, j_topo: float) -> Tuple[float, float]:
        """Predict E_floor and uncertainty from J_topo."""
        # Linear regression: E_floor = a + b * J_topo
        # Using empirical values
        e_pred = J_TOPO_TO_E_INTERCEPT + J_TOPO_TO_E_SLOPE * (j_topo - 0.35)

        # Uncertainty decreases with more observations
        n_obs = len(self.j_topo_observations)
        sigma = 0.15 / np.sqrt(n_obs + 1)

        return e_pred, sigma

    def predict_loss(self, arch: Architecture, j_topo: float,
                     fidelity: int) -> Tuple[float, float]:
        """Predict loss at given fidelity."""
        # Base prediction from E_floor
        e_pred, e_sigma = self.predict_e_floor(arch, j_topo)

        # Add β contribution from norm
        if arch.norm == 'bn':
            beta_pred = BETA_BN
        elif arch.norm == 'ln':
            beta_pred = BETA_LN
        else:
            beta_pred = BETA_NONE

        # Loss at D_max with scaling term
        D_max = max(D_VALUES)
        alpha = ALPHA_BN if arch.norm == 'bn' else ALPHA_BASE
        loss_pred = alpha * (D_max ** (-beta_pred)) + e_pred

        # Uncertainty decreases with higher fidelity
        noise_levels = {1: NOISE_L1, 2: NOISE_L2, 3: NOISE_L3}
        sigma = loss_pred * noise_levels.get(fidelity, 0.1)

        return loss_pred, sigma

    def acquisition_score(self, arch: Architecture, j_topo: float,
                         fidelity: int, lambda_explore: float = 0.1) -> float:
        """Compute acquisition score for an architecture."""
        loss_pred, sigma = self.predict_loss(arch, j_topo, fidelity)

        # Score: lower loss + exploration bonus
        score = -loss_pred + lambda_explore * sigma
        return score


# ──────────────────────────────────────────────────────────────────────────────
# SEARCH STRATEGIES
# ──────────────────────────────────────────────────────────────────────────────

class SearchStrategy(ABC):
    """Base class for search strategies."""

    @abstractmethod
    def select_next(self, candidates: List[Architecture],
                    gp: GPRegressor,
                    budget_remaining: float) -> Optional[Tuple[Architecture, int]]:
        """Select next architecture and fidelity level. Returns None if done."""
        pass


class RandomSearch(SearchStrategy):
    """Pure random search."""

    def select_next(self, candidates, gp, budget_remaining):
        if budget_remaining < 0.5 or not candidates:
            return None
        arch = random.choice(candidates)
        return arch, 3  # Full fidelity


class GreedyJTopo(SearchStrategy):
    """Greedy by J_topo only (zero-cost)."""

    def select_next(self, candidates, gp, budget_remaining):
        if budget_remaining < 0.5 or not candidates:
            return None

        # Compute J_topo for all candidates (simulated zero-cost)
        for arch in candidates:
            arch._cached_j_topo = arch.compute_j_topo(noise=0)

        # Select LOWEST J_topo (theory: lower J -> lower E_floor, r=+0.83)
        candidates.sort(key=lambda a: a._cached_j_topo, reverse=False)
        return candidates[0], 3


class HierarchicalBO(SearchStrategy):
    """Full hierarchical BO with multi-fidelity."""

    def __init__(self):
        self.round = 0

    def select_next(self, candidates, gp, budget_remaining=999):
        if budget_remaining < 0.5 or not candidates:
            return None

        self.round += 1

        # Round 1-3: L1 (5 epochs) on diverse candidates
        if self.round <= 3:
            fidelity = 1
            # Select most uncertain (highest variance from GP)
            if gp.loss_observations:
                arch = max(candidates,
                           key=lambda a: gp.predict_loss(a, a.compute_j_topo(), 1)[1])
            else:
                arch = random.choice(candidates)
            return arch, fidelity

        # Round 4-6: L2 (50 epochs) refinement
        if self.round <= 6:
            fidelity = 2
        else:
            fidelity = 3

        # Select by acquisition score
        candidates.sort(
            key=lambda a: gp.acquisition_score(a, a.compute_j_topo(), fidelity),
            reverse=True
        )
        return candidates[0], fidelity


# ──────────────────────────────────────────────────────────────────────────────
# SIMULATION ENGINE
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SimulationResult:
    """Result of a simulation run."""
    strategy_name: str
    final_losses: List[float]
    final_e_floors: List[float]
    best_arch_id: int
    best_loss: float
    total_budget: float
    architectures_evaluated: int
    by_fidelity: Dict[int, int] = field(default_factory=dict)
    history: List[Dict] = field(default_factory=list)


def run_simulation(archs: List[Architecture],
                    strategy: SearchStrategy,
                    budget: float = 100.0,  # GPU-minutes
                    name: str = "strategy") -> SimulationResult:
    """Run a simulation with given strategy."""

    gp = GPRegressor()
    remaining = budget
    evaluated = set()
    history = []

    # Phase 0: Compute J_topo for all (zero-cost)
    for arch in archs:
        j_topo = arch.compute_j_topo()
        arch._cached_j_topo = j_topo
        gp.add_j_topo(arch, j_topo)

    # Active loop
    candidates = list(archs)

    while remaining > 0:
        selection = strategy.select_next(candidates, gp, budget_remaining=remaining)
        if selection is None:
            break

        arch, fidelity = selection

        # Cost per fidelity level
        cost_map = {1: 0.5, 2: 5.0, 3: 30.0}
        cost = cost_map[fidelity]

        if remaining < cost:
            break

        # Compute ground truth
        gt = compute_ground_truth(arch._cached_j_topo, arch.norm)
        noise_map = {1: NOISE_L1, 2: NOISE_L2, 3: NOISE_L3}
        loss = compute_loss(max(D_VALUES), gt, noise=noise_map[fidelity])

        # Add observation
        gp.add_loss_observation(arch, arch._cached_j_topo, fidelity, loss)
        evaluated.add(arch.arch_id)
        remaining -= cost

        # Record
        history.append({
            'round': len(history) + 1,
            'arch_id': arch.arch_id,
            'fidelity': fidelity,
            'cost': cost,
            'loss': loss,
            'remaining': remaining
        })

        # Remove evaluated from candidates
        if fidelity == 3:  # Only remove full evaluations from candidates
            if arch in candidates:
                candidates.remove(arch)

    # Evaluate best architecture at full fidelity
    best_arch = min(archs, key=lambda a: a._final_loss if hasattr(a, '_final_loss') else 999)

    # Compute final losses for all evaluated archs
    final_losses = []
    final_e_floors = []
    for arch in archs:
        if arch.arch_id in evaluated:
            gt = compute_ground_truth(arch._cached_j_topo, arch.norm)
            final_losses.append(compute_loss(max(D_VALUES), gt, noise=NOISE_L3))
            final_e_floors.append(gt['e_floor'])

    # Find best
    best_idx = np.argmin(final_losses) if final_losses else 0
    best_arch_id = list(evaluated)[best_idx] if evaluated else -1
    best_loss = final_losses[best_idx] if final_losses else 999

    return SimulationResult(
        strategy_name=name,
        final_losses=final_losses,
        final_e_floors=final_e_floors,
        best_arch_id=best_arch_id,
        best_loss=best_loss,
        total_budget=budget - remaining,
        architectures_evaluated=len(evaluated),
        history=history
    )


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("ThermoRG Phase B — HBO Simulation (CPU-only)")
    print("=" * 70)
    print()

    # Generate architecture space
    N_ARCHS = 200
    archs = generate_arch_space(N_ARCHS)
    print(f"Generated {N_ARCHS} architectures")
    print()

    # Pre-compute J_topo and ground truth for all archs
    for arch in archs:
        arch._cached_j_topo = arch.compute_j_topo(noise=0)
        gt = compute_ground_truth(arch._cached_j_topo, arch.norm)
        arch._gt = gt
        arch._final_loss = compute_loss(max(D_VALUES), gt, noise=NOISE_L3)

    # Sort by true final loss
    true_best = min(archs, key=lambda a: a._final_loss)
    print(f"True best architecture: ID={true_best.arch_id}")
    print(f"  width={true_best.width}, depth={true_best.depth}, "
          f"skip={true_best.skip}, norm={true_best.norm}")
    print(f"  J_topo={true_best._cached_j_topo:.3f}, "
          f"E_floor={true_best._gt['e_floor']:.3f}, "
          f"loss={true_best._final_loss:.4f}")
    print()

    # Compute oracle ranking
    sorted_archs = sorted(archs, key=lambda a: a._final_loss)
    oracle_top5 = [a.arch_id for a in sorted_archs[:5]]
    oracle_top10 = [a.arch_id for a in sorted_archs[:10]]
    print(f"Oracle top-5 IDs: {oracle_top5}")
    print(f"Oracle top-10 IDs: {oracle_top10}")
    print()

    # Run simulations
    BUDGET = 100.0  # GPU-minutes

    strategies = [
        ("Random Search", RandomSearch()),
        ("Greedy (J_topo)", GreedyJTopo()),
        ("HBO (Full)", HierarchicalBO()),
    ]

    results = []
    for name, strategy in strategies:
        print(f"Running: {name} ...", end=" ", flush=True)
        t0 = time.time()
        result = run_simulation(archs, strategy, budget=BUDGET, name=name)
        elapsed = time.time() - t0
        print(f"done in {elapsed:.1f}s")
        results.append(result)

    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Strategy':<20} {'Archs Eval':<12} {'Budget Used':<12} {'Best Loss':<10} {'Top-5 Hit':<10}")
    print("-" * 65)

    for result in results:
        # Count full-fidelity evaluations
        full_evals = sum(1 for h in result.history if h['fidelity'] == 3)

        # Check if oracle top-5 was found
        eval_ids = set(h['arch_id'] for h in result.history if h['fidelity'] == 3)
        top5_found = len(eval_ids & set(oracle_top5))

        print(f"{result.strategy_name:<20} "
              f"{result.architectures_evaluated:<12} "
              f"{result.total_budget:<12.1f} "
              f"{result.best_loss:<10.4f} "
              f"{top5_found}/5")

    print()
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Compute efficiency
    random_result = results[0]
    hbo_result = results[2]

    # How much budget does HBO need to match random?
    print()
    print("Key question: How does HBO efficiency compare to random?")
    print()
    print("Expected:")
    print("  - Random: evaluates many architectures at full fidelity (expensive)")
    print("  - HBO: evaluates many at L1 (cheap) + few at L3 (expensive)")
    print("  - HBO should find comparable or better architectures at lower budget")
    print()
    print("Next steps:")
    print("  1. Implement full GP with proper kernel (sklearn GPy)")
    print("  2. Run on real CIFAR-10 when GPU available")
    print("  3. Calibrate with Phase B Session 2 data")

    return results


if __name__ == '__main__':
    results = main()
