#!/usr/bin/env python3
"""
ThermoRG Phase B — HBO Evaluation Redesign
==========================================

Redesign the HBO simulation to properly test its advantage over greedy.

Key improvements:
1. Realistic coupling: norm type affects both J_topo and E_floor/β
   - LN networks have J_topo ≈ 0 (LN excluded from weight matrices)
   - BN networks have lower E_floor but higher J_topo
   - Creates non‑linear tradeoff
2. Capacity constraint: max_params = 5M (eliminates large models)
3. Noise: J_topo → E_floor correlation r = 0.83, not perfect
4. Objective: find best E_floor subject to param constraint

Strategies compared:
- Greedy: rank by J_topo, pick feasible, train at L3
- HBO: multi‑fidelity GP, L1→L2→L3 cascade, select by EI
- Random: baseline

Metrics:
- Success rate: fraction of top‑5 architectures found
- Average best loss (E_floor)
- Budget efficiency: loss vs budget spent
"""

import numpy as np
import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
import random
from scipy.stats import spearmanr

# Import architecture definitions
import sys
sys.path.insert(0, '/home/node/.openclaw/workspace/github_staging/ThermoRG-NN/src')
from thermorg_hbo.arch.encoding import Architecture, ArchitectureSpace

np.random.seed(42)
random.seed(42)

# ──────────────────────────────────────────────────────────────────────────────
# REDESIGNED GROUND TRUTH MODEL WITH COUPLING
# ──────────────────────────────────────────────────────────────────────────────

# Base correlation (from Phase A)
J_TOPO_TO_E_INTERCEPT = 0.458
J_TOPO_TO_E_SLOPE = 1.018
CORRELATION_NOISE = 0.15  # reduces r from 1.0 to ~0.83

# Norm effects on β (from Phase S1)
BETA_NONE = 0.180
BETA_BN = 0.368
BETA_LN = 0.370

# Norm effects on E_floor (hypothesis: BN helps, LN hurts)
E_FLOOR_BN_BONUS = -0.08   # BN reduces E_floor
E_FLOOR_LN_PENALTY = 0.05  # LN increases E_floor

# Norm effects on J_topo (LN excluded from weight matrices)
J_TOPO_LN_ZERO = True      # LN → J_topo ≈ 0

# Scaling law α (from fits)
ALPHA_BASE = 0.93
ALPHA_BN = 1.71

# Noise levels (relative)
NOISE_J_TOPO = 0.02
NOISE_L1 = 0.05    # 5‑epoch loss
NOISE_L2 = 0.02    # 50‑epoch loss
NOISE_L3 = 0.005   # 200‑epoch loss

# Fidelity costs (GPU‑minutes)
COST_L0 = 0.001
COST_L1 = 0.5
COST_L2 = 5.0
COST_L3 = 30.0

# Capacity constraint (max parameters in millions)
MAX_PARAMS_M = 5.0

# Dataset sizes for scaling law
D_VALUES = [32, 48, 64, 96]
D_MAX = max(D_VALUES)


def compute_ground_truth(arch: Architecture) -> Dict[str, float]:
    """
    Compute ground truth for an architecture with coupling.
    Returns dict with: j_topo_clean, j_topo_noisy, beta, alpha, e_floor, feasible.
    """
    # ─── J_topo base (from width, depth, skip) ───────────────────────────────
    j_base = 0.25
    if arch.skip:
        j_base += 0.08
    j_base += (arch.depth - 5) * 0.025
    j_base -= (arch.width / 64) * 0.15

    # ─── Norm effect on J_topo ───────────────────────────────────────────────
    if arch.norm == 'ln' and J_TOPO_LN_ZERO:
        j_topo_clean = 0.0  # LN excluded from weight matrices
    else:
        j_topo_clean = j_base
        if arch.norm == 'bn':
            j_topo_clean += 0.05  # BN adds extra parameters? small increase

    # Add per‑architecture variation
    j_topo_clean += np.random.normal(0, 0.02)
    j_topo_clean = np.clip(j_topo_clean, 0.12, 0.65)

    # ─── E_floor from J_topo (positive correlation) ──────────────────────────
    e_floor_base = J_TOPO_TO_E_INTERCEPT + J_TOPO_TO_E_SLOPE * j_topo_clean

    # Add correlation noise (reduces r to ~0.83)
    e_floor_base += np.random.normal(0, CORRELATION_NOISE)

    # ─── Norm effect on E_floor ──────────────────────────────────────────────
    if arch.norm == 'bn':
        e_floor = e_floor_base + E_FLOOR_BN_BONUS
    elif arch.norm == 'ln':
        e_floor = e_floor_base + E_FLOOR_LN_PENALTY
    else:
        e_floor = e_floor_base

    e_floor = np.clip(e_floor, 0.05, 1.5)

    # ─── β and α from norm ───────────────────────────────────────────────────
    if arch.norm == 'bn':
        beta = BETA_BN
        alpha = ALPHA_BN
    elif arch.norm == 'ln':
        beta = BETA_LN
        alpha = ALPHA_BASE
    else:
        beta = BETA_NONE
        alpha = ALPHA_BASE

    # Small per‑arch variation
    beta += np.random.normal(0, 0.01)
    alpha += np.random.normal(0, 0.02)

    # ─── Feasibility (capacity constraint) ──────────────────────────────────
    feasible = arch.n_params_M <= MAX_PARAMS_M

    # Noisy J_topo (as measured)
    j_topo_noisy = j_topo_clean + np.random.normal(0, NOISE_J_TOPO)

    return {
        'j_topo_clean': j_topo_clean,
        'j_topo_noisy': j_topo_noisy,
        'beta': beta,
        'alpha': alpha,
        'e_floor': e_floor,
        'feasible': feasible,
    }


def compute_loss(arch: Architecture, gt: Dict, fidelity: int) -> float:
    """Compute loss at D_MAX with fidelity‑appropriate noise."""
    loss = gt['alpha'] * (D_MAX ** (-gt['beta'])) + gt['e_floor']
    noise_map = {1: NOISE_L1, 2: NOISE_L2, 3: NOISE_L3}
    loss *= (1 + np.random.normal(0, noise_map[fidelity]))
    return max(loss, gt['e_floor'] * 0.9)


# ──────────────────────────────────────────────────────────────────────────────
# SEARCH STRATEGIES
# ──────────────────────────────────────────────────────────────────────────────

class SearchStrategy:
    """Base class for search strategies."""

    def select_next(self,
                    candidates: List[Architecture],
                    budget_remaining: float,
                    history: List[Dict]) -> Optional[Tuple[Architecture, int]]:
        """Select next (architecture, fidelity). Return None if done."""
        raise NotImplementedError

    def evaluate(self, arch: Architecture, fidelity: int, gt: Dict) -> float:
        """Evaluate architecture (simulate)."""
        return compute_loss(arch, gt, fidelity)


class RandomSearch(SearchStrategy):
    """Random search (baseline)."""

    def select_next(self, candidates, budget_remaining, history):
        if budget_remaining < COST_L3 or not candidates:
            return None
        arch = random.choice(candidates)
        return arch, 3


class GreedyJTopo(SearchStrategy):
    """
    Greedy by J_topo only.
    Steps:
      1. Filter feasible candidates (params ≤ MAX_PARAMS_M)
      2. Rank by J_topo (lowest first)
      3. Evaluate top‑K at full fidelity (L3)
    """
    def __init__(self, top_k: int = 10):
        self.top_k = top_k
        self.evaluated_ids = set()

    def select_next(self, candidates, budget_remaining, history):
        if budget_remaining < COST_L3 or not candidates:
            return None

        # First round: compute J_topo for all feasible candidates
        feasible = [a for a in candidates if a.n_params_M <= MAX_PARAMS_M]
        if not feasible:
            return None

        # Compute J_topo (zero‑cost)
        for arch in feasible:
            if not hasattr(arch, '_j_topo'):
                arch._j_topo = compute_ground_truth(arch)['j_topo_noisy']

        # Sort by J_topo (lower is better)
        feasible.sort(key=lambda a: a._j_topo)

        # Pick the best unevaluated architecture
        for arch in feasible:
            if id(arch) not in self.evaluated_ids:
                self.evaluated_ids.add(id(arch))
                return arch, 3

        return None


class HierarchicalBO(SearchStrategy):
    """
    Hierarchical Bayesian Optimization with multi‑fidelity.
    Simplified GP surrogate with Expected Improvement (EI).
    """
    def __init__(self, explore_weight: float = 0.1):
        self.explore_weight = explore_weight
        self.observations = []  # (arch, fidelity, loss)
        self.round = 0

    def _gp_predict(self, arch: Architecture, fidelity: int) -> Tuple[float, float]:
        """
        Simplified GP prediction (mean, std).
        Uses J_topo and norm as features.
        """
        # Feature vector: [j_topo, norm_onehot*3]
        gt = compute_ground_truth(arch)
        j = gt['j_topo_noisy']
        if arch.norm == 'none':
            norm_vec = [1, 0, 0]
        elif arch.norm == 'bn':
            norm_vec = [0, 1, 0]
        else:
            norm_vec = [0, 0, 1]

        # Very simple regression: mean = linear model, std = heuristic
        # In a real implementation you would use sklearn GP
        obs_fidelity = [(o[1], o[2]) for o in self.observations if o[0] == arch]
        if obs_fidelity:
            # Already observed at same fidelity → low uncertainty
            loss = obs_fidelity[0][1]
            return loss, 0.01

        # Predict from J_topo and norm
        # Baseline: E_floor ~ intercept + slope*j
        pred_e = J_TOPO_TO_E_INTERCEPT + J_TOPO_TO_E_SLOPE * j
        # Norm adjustment
        if arch.norm == 'bn':
            pred_e += E_FLOOR_BN_BONUS
        elif arch.norm == 'ln':
            pred_e += E_FLOOR_LN_PENALTY

        # Add scaling term
        beta = BETA_BN if arch.norm == 'bn' else BETA_NONE
        alpha = ALPHA_BN if arch.norm == 'bn' else ALPHA_BASE
        pred_loss = alpha * (D_MAX ** (-beta)) + pred_e

        # Uncertainty: larger if few observations, reduced with fidelity
        n_obs = len(self.observations)
        base_std = 0.15 / (n_obs + 1)
        fidelity_std_factor = {1: 1.0, 2: 0.5, 3: 0.2}
        std = base_std * fidelity_std_factor.get(fidelity, 1.0)

        return pred_loss, std

    def _acquisition_ei(self, arch: Architecture, fidelity: int) -> float:
        """Expected Improvement acquisition."""
        if not self.observations:
            return 1.0  # explore first

        # Best observed loss so far
        best_observed = min(loss for _, _, loss in self.observations)
        mu, sigma = self._gp_predict(arch, fidelity)

        # EI = (best - mu) * Φ(z) + sigma * φ(z) where z = (best - mu)/sigma
        z = (best_observed - mu) / (sigma + 1e-9)
        from scipy.stats import norm
        ei = (best_observed - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        # Add exploration bonus
        ei += self.explore_weight * sigma
        return max(ei, 0.0)

    def select_next(self, candidates, budget_remaining, history):
        if budget_remaining < COST_L1 or not candidates:
            return None

        self.round += 1

        # Filter feasible
        feasible = [a for a in candidates if a.n_params_M <= MAX_PARAMS_M]
        if not feasible:
            return None

        # First 3 rounds: L1 exploration
        if self.round <= 3:
            fidelity = 1
            # Pick most uncertain feasible arch
            arch = max(feasible, key=lambda a: self._gp_predict(a, 1)[1])
            return arch, fidelity

        # Next 2 rounds: L2 refinement
        if self.round <= 5:
            fidelity = 2
        else:
            fidelity = 3

        # Select by EI
        arch = max(feasible, key=lambda a: self._acquisition_ei(a, fidelity))
        return arch, fidelity

    def add_observation(self, arch, fidelity, loss):
        """Record observation."""
        self.observations.append((arch, fidelity, loss))


# ──────────────────────────────────────────────────────────────────────────────
# SIMULATION ENGINE
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SimulationResult:
    strategy_name: str
    best_arch_id: Optional[int]
    best_loss: float
    best_e_floor: float
    total_cost: float
    n_evaluated: int
    top5_found: int
    history: List[Dict] = field(default_factory=list)


def run_simulation(archs: List[Architecture],
                   strategy: SearchStrategy,
                   budget: float = 100.0,
                   name: str = "strategy") -> SimulationResult:
    """
    Run a single simulation.
    """
    # Pre‑compute ground truth for all architectures
    gt_map = {}
    for arch in archs:
        gt_map[id(arch)] = compute_ground_truth(arch)

    # Filter feasible architectures (respect constraint)
    feasible_ids = [id(a) for a in archs if gt_map[id(a)]['feasible']]
    if not feasible_ids:
        raise ValueError("No feasible architectures in space")

    # Compute oracle ranking of feasible architectures (by E_floor)
    feasible_archs = [a for a in archs if gt_map[id(a)]['feasible']]
    feasible_archs.sort(key=lambda a: gt_map[id(a)]['e_floor'])
    oracle_top5 = feasible_archs[:5]
    oracle_top5_ids = [id(a) for a in oracle_top5]

    # Initialize
    remaining = budget
    evaluated = set()  # arch ids evaluated at L3
    history = []
    strategy_obs = []  # for HBO

    # Candidate pool (starts with all feasible)
    candidates = feasible_archs.copy()

    # Main loop
    while remaining > COST_L1:
        selection = strategy.select_next(candidates, remaining, history)
        if selection is None:
            break

        arch, fidelity = selection
        arch_id = id(arch)

        # Cost
        cost_map = {1: COST_L1, 2: COST_L2, 3: COST_L3}
        cost = cost_map[fidelity]
        if remaining < cost:
            break

        # Evaluate
        gt = gt_map[arch_id]
        loss = compute_loss(arch, gt, fidelity)

        # Record
        evaluated.add(arch_id)
        history.append({
            'arch_id': arch_id,
            'arch': arch,
            'fidelity': fidelity,
            'cost': cost,
            'loss': loss,
            'e_floor': gt['e_floor'],
            'remaining': remaining - cost,
        })

        # Update strategy's internal model (for HBO)
        if isinstance(strategy, HierarchicalBO):
            strategy.add_observation(arch, fidelity, loss)

        remaining -= cost

        # Remove from candidates if fully evaluated (L3)
        if fidelity == 3:
            if arch in candidates:
                candidates.remove(arch)

    # Determine best evaluated architecture (by ground‑truth E_floor)
    best_arch_id = None
    best_loss = float('inf')
    best_e_floor = float('inf')
    for arch_id in evaluated:
        e = gt_map[arch_id]['e_floor']
        if e < best_e_floor:
            best_e_floor = e
            best_arch_id = arch_id
            # loss corresponding to best E_floor
            best_loss = compute_loss(
                next(a for a in archs if id(a) == arch_id),
                gt_map[arch_id], 3)

    # Count top‑5 found
    top5_found = len(set(oracle_top5_ids) & evaluated)

    return SimulationResult(
        strategy_name=name,
        best_arch_id=best_arch_id,
        best_loss=best_loss,
        best_e_floor=best_e_floor,
        total_cost=budget - remaining,
        n_evaluated=len(evaluated),
        top5_found=top5_found,
        history=history,
    )


# ──────────────────────────────────────────────────────────────────────────────
# MAIN EXPERIMENT
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("ThermoRG Phase B — HBO Evaluation Redesign")
    print("Testing HBO advantage under realistic coupling and constraints")
    print("=" * 70)
    print()

    # Generate architecture space
    space = ArchitectureSpace()
    N_ARCHS = 300
    archs = space.sample(N_ARCHS)
    print(f"Generated {N_ARCHS} random architectures")

    # Pre‑compute ground truth and feasibility
    gt_map = {id(a): compute_ground_truth(a) for a in archs}
    feasible = [a for a in archs if gt_map[id(a)]['feasible']]
    print(f"Feasible (params ≤ {MAX_PARAMS_M}M): {len(feasible)} architectures")
    print()

    # Oracle ranking (by E_floor)
    feasible.sort(key=lambda a: gt_map[id(a)]['e_floor'])
    oracle_top5 = feasible[:5]
    oracle_worst5 = feasible[-5:]

    print("Oracle top‑5 architectures (best E_floor):")
    for i, arch in enumerate(oracle_top5):
        gt = gt_map[id(arch)]
        print(f"  {i+1}. ID={id(arch)} w={arch.width} d={arch.depth} "
              f"skip={arch.skip} norm={arch.norm} "
              f"params={arch.n_params_M:.1f}M "
              f"J_topo={gt['j_topo_clean']:.3f} "
              f"E_floor={gt['e_floor']:.3f}")
    print()

    # Compute correlation between J_topo and E_floor for feasible archs
    j_vals = [gt_map[id(a)]['j_topo_clean'] for a in feasible]
    e_vals = [gt_map[id(a)]['e_floor'] for a in feasible]
    corr, _ = spearmanr(j_vals, e_vals)
    print(f"Spearman correlation J_topo ↔ E_floor (feasible): {corr:.3f}")
    print()

    # Run strategies
    BUDGET = 150.0  # GPU‑minutes (enough for 5 full L3 evals)
    strategies = [
        ("Random Search", RandomSearch()),
        ("Greedy (J_topo)", GreedyJTopo(top_k=10)),
        ("HBO (multi‑fidelity)", HierarchicalBO(explore_weight=0.1)),
    ]

    results = []
    for name, strategy in strategies:
        print(f"Running {name} ...", end=" ", flush=True)
        t0 = time.time()
        result = run_simulation(archs, strategy, budget=BUDGET, name=name)
        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s)")
        results.append(result)

    # Summary table
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Strategy':<25} {'Evals':<7} {'Cost':<7} {'Best E_floor':<12} "
          f"{'Top‑5 found':<12} {'Best Loss':<10}")
    print("-" * 75)

    for res in results:
        print(f"{res.strategy_name:<25} "
              f"{res.n_evaluated:<7} "
              f"{res.total_cost:<7.1f} "
              f"{res.best_e_floor:<12.4f} "
              f"{res.top5_found}/5{'':<7} "
              f"{res.best_loss:<10.4f}")

    print()
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print()

    # Compare HBO vs Greedy
    greedy_res = next(r for r in results if "Greedy" in r.strategy_name)
    hbo_res = next(r for r in results if "HBO" in r.strategy_name)

    print("HBO vs Greedy comparison:")
    print(f"  HBO best E_floor:   {hbo_res.best_e_floor:.4f}")
    print(f"  Greedy best E_floor: {greedy_res.best_e_floor:.4f}")
    print(f"  Difference:          {greedy_res.best_e_floor - hbo_res.best_e_floor:.4f} "
          f"(positive means HBO better)")
    print()
    print(f"  HBO top‑5 found:     {hbo_res.top5_found}/5")
    print(f"  Greedy top‑5 found:  {greedy_res.top5_found}/5")
    print()
    print(f"  HBO total cost:      {hbo_res.total_cost:.1f} GPU‑min")
    print(f"  Greedy total cost:   {greedy_res.total_cost:.1f} GPU‑min")
    print()

    # Budget efficiency: cost per top‑5 found
    def cost_per_top5(res):
        if res.top5_found == 0:
            return float('inf')
        return res.total_cost / res.top5_found

    print("Cost per top‑5 architecture found:")
    for res in results:
        if res.top5_found > 0:
            print(f"  {res.strategy_name:<25} {cost_per_top5(res):.1f} GPU‑min each")
        else:
            print(f"  {res.strategy_name:<25} (none found)")

    print()
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()
    print("If HBO works as intended, it should:")
    print("  1. Find better E_floor than greedy (non‑linear tradeoff)")
    print("  2. Discover more top‑5 architectures")
    print("  3. Use budget more efficiently (multi‑fidelity)")
    print()
    print("If greedy still wins, the coupling/constraints may not be")
    print("strong enough, or HBO implementation needs improvement.")
    print()

    return results


def run_monte_carlo(trials: int = 10, n_archs: int = 200, budget: float = 150.0):
    """
    Run multiple trials with different random seeds and aggregate statistics.
    """
    import numpy as np
    from collections import defaultdict
    
    metrics = defaultdict(list)
    
    for trial in range(trials):
        seed = 42 + trial
        np.random.seed(seed)
        random.seed(seed)
        
        space = ArchitectureSpace()
        archs = space.sample(n_archs)
        
        strategies = [
            ("Random Search", RandomSearch()),
            ("Greedy (J_topo)", GreedyJTopo(top_k=10)),
            ("HBO (multi‑fidelity)", HierarchicalBO(explore_weight=0.1)),
        ]
        
        for name, strategy in strategies:
            result = run_simulation(archs, strategy, budget=budget, name=name)
            metrics[name + '_best_e_floor'].append(result.best_e_floor)
            metrics[name + '_top5_found'].append(result.top5_found)
            metrics[name + '_cost'].append(result.total_cost)
            metrics[name + '_evals'].append(result.n_evaluated)
    
    print("\n" + "="*70)
    print("MONTE CARLO RESULTS ({} trials)".format(trials))
    print("="*70)
    print()
    print("{:<25} {:>10} {:>10} {:>10} {:>10}".format(
        "Strategy", "Avg E_floor", "Avg Top5", "Avg Cost", "Success %"))
    print("-"*65)
    
    for name in ["Random Search", "Greedy (J_topo)", "HBO (multi‑fidelity)"]:
        key = name + '_best_e_floor'
        avg_e = np.mean(metrics[key])
        avg_top5 = np.mean(metrics[name + '_top5_found'])
        avg_cost = np.mean(metrics[name + '_cost'])
        success_pct = 100 * np.mean([t > 0 for t in metrics[name + '_top5_found']])
        print("{:<25} {:>10.4f} {:>10.2f} {:>10.1f} {:>10.1f}%".format(
            name, avg_e, avg_top5, avg_cost, success_pct))
    
    print()
    print("Success % = percentage of trials where at least one top‑5 arch found.")
    print()
    return metrics


if __name__ == '__main__':
    # Run single demonstration
    results = main()
    
    # Run Monte Carlo with 3 trials for robust statistics
    print("\n\n")
    run_monte_carlo(trials=3, n_archs=200, budget=150.0)