"""
Synthetic simulation environment for ThermoRG-HBO.

This simulates architecture evaluation with ground-truth loss
based on the validated theory (Phase A + S1 v3).

Ground truth:
- E_floor ≈ 0.84 + 0.83 * (J_topo - 0.35)
- β from normalization: None=0.18, BN=0.37, LN=0.37
- Loss = α * D^(-β) + E_floor + noise
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np

from ..arch.encoding import Architecture


# Ground truth parameters (from Phase A + S1 v3)
J_TOPO_INTERCEPT = 0.84
J_TOPO_SLOPE = 0.83

BETA_NONE = 0.180
BETA_BN = 0.368
BETA_LN = 0.370

ALPHA_BASE = 0.93
ALPHA_BN = 1.71

E_FLOOR_NONE = 0.276
E_FLOOR_BN = 0.181

# Noise levels (relative)
NOISE_J_TOPO = 0.02
NOISE_L1 = 0.05    # 5-epoch loss
NOISE_L2 = 0.02    # 50-epoch loss
NOISE_L3 = 0.005   # 200-epoch loss

# Fidelity costs (GPU-minutes)
FIDELITY_COST = {1: 0.5, 2: 5.0, 3: 30.0}

D_VALUES = [32, 48, 64, 96]
D_MAX = max(D_VALUES)


@dataclass
class GroundTruth:
    """Ground truth parameters for an architecture."""
    j_topo: float
    beta: float
    alpha: float
    e_floor: float


class SimulationEnv:
    """Synthetic environment for architecture evaluation.

    Simulates the full pipeline:
    1. Compute J_topo from architecture (zero-cost)
    2. Evaluate at various fidelities (additive noise)
    3. Return loss observations
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self._arch_gt: Dict[int, GroundTruth] = {}

    def _compute_j_topo(self, arch: Architecture, noise: float = NOISE_J_TOPO) -> float:
        """Compute J_topo from architecture features.

        Simplified model: skip raises J, depth raises J, width lowers J.
        """
        j = 0.35
        if arch.skip:
            j += 0.15
        j += (arch.depth - 5) * 0.03
        j -= (arch.width / 64) * 0.10
        j += self.rng.normal(0, noise)
        return float(np.clip(j, 0.1, 0.9))

    def _compute_ground_truth(self, arch: Architecture) -> GroundTruth:
        """Compute ground truth parameters for architecture."""
        # J_topo (stored without noise)
        j_topo_clean = 0.35
        if arch.skip:
            j_topo_clean += 0.15
        j_topo_clean += (arch.depth - 5) * 0.03
        j_topo_clean -= (arch.width / 64) * 0.10
        j_topo_clean = float(np.clip(j_topo_clean, 0.1, 0.9))

        # E_floor from J_topo (r=0.83)
        e_floor = J_TOPO_INTERCEPT + J_TOPO_SLOPE * (j_topo_clean - 0.35)
        e_floor = float(np.clip(e_floor, 0.05, 1.5))

        # Beta from normalization
        if arch.norm == 'bn':
            beta = BETA_BN
            alpha = ALPHA_BN
        elif arch.norm == 'ln':
            beta = BETA_LN
            alpha = ALPHA_BASE
        else:
            beta = BETA_NONE
            alpha = ALPHA_BASE

        # Add per-arch noise
        beta += self.rng.normal(0, 0.01)
        e_floor += self.rng.normal(0, 0.02)

        return GroundTruth(
            j_topo=j_topo_clean,
            beta=beta,
            alpha=alpha,
            e_floor=e_floor,
        )

    def get_j_topo(self, arch: Architecture) -> float:
        """Get J_topo for architecture (zero-cost)."""
        arch_id = id(arch)
        if arch_id not in self._arch_gt:
            self._arch_gt[arch_id] = self._compute_ground_truth(arch)
        # Return noisy version (simulating measurement)
        return self._compute_j_topo(arch, noise=NOISE_J_TOPO)

    def evaluate(self, arch: Architecture, fidelity: int) -> Tuple[float, GroundTruth]:
        """Evaluate architecture at given fidelity.

        Args:
            arch: Architecture to evaluate
            fidelity: 1 (L1, 5ep), 2 (L2, 50ep), 3 (L3, 200ep)

        Returns:
            (loss, ground_truth)
        """
        arch_id = id(arch)
        if arch_id not in self._arch_gt:
            self._arch_gt[arch_id] = self._compute_ground_truth(arch)

        gt = self._arch_gt[arch_id]

        # Compute loss at D_max
        loss = gt.alpha * (D_MAX ** (-gt.beta)) + gt.e_floor

        # Add noise based on fidelity
        noise_map = {1: NOISE_L1, 2: NOISE_L2, 3: NOISE_L3}
        noise = noise_map.get(fidelity, NOISE_L3)
        loss *= (1 + self.rng.normal(0, noise))

        return max(float(loss), gt.e_floor * 0.9), gt

    def get_ground_truth(self, arch: Architecture) -> GroundTruth:
        """Get ground truth for architecture (no noise)."""
        arch_id = id(arch)
        if arch_id not in self._arch_gt:
            self._arch_gt[arch_id] = self._compute_ground_truth(arch)
        return self._arch_gt[arch_id]


def create_test_env() -> SimulationEnv:
    """Create a simulation environment with fixed seed for testing."""
    return SimulationEnv(seed=42)
