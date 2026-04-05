"""
Gaussian Process Surrogate for ThermoRG-HBO.

Implements a GP that uses J_topo as a prior for E_floor,
with normalization type informing β.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ..arch.encoding import Architecture, arch_to_features


# Theoretical priors from Phase A / S1 v3
J_TOPO_TO_E_INTERCEPT = 0.84
J_TOPO_TO_E_SLOPE = 0.83  # per unit J_topo deviation from 0.35

BETA_NORM = {
    'none': 0.180,
    'bn': 0.368,
    'ln': 0.370,
}


@dataclass
class Observation:
    """A single observation."""
    arch: Architecture
    j_topo: float
    fidelity: int      # 0=J_topo only, 1=L1(5ep), 2=L2(50ep), 3=L3(200ep)
    loss: Optional[float] = None


class GPSurrogate:
    """GP surrogate for architecture search.

    Uses J_topo as a cheap prior for E_floor (r=0.83 from Phase A).
    Normalization type informs β prior.

    For Phase B, the GP models E_floor, not early loss.
    """

    def __init__(self, length_scale: float = 1.0, noise_var: float = 0.01):
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn required for GPSurrogate. Install: pip install scikit-learn")

        self.observations: List[Observation] = []
        self.j_topo_observations: List[Observation] = []
        self.loss_observations: List[Observation] = []

        # GP kernel: Matern-5/2 + white noise
        # Note: kernel dimension set dynamically in fit()
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3)) *
            Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
            + WhiteKernel(noise_var, (1e-5, 1e-1))
        )
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            alpha=1e-6,
            normalize_y=True,
        )
        self._fitted = False

    def add_j_topo(self, arch: Architecture, j_topo: float):
        """Add a zero-cost J_topo observation."""
        obs = Observation(arch=arch, j_topo=j_topo, fidelity=0, loss=None)
        self.observations.append(obs)
        self.j_topo_observations.append(obs)

    def add_loss(self, arch: Architecture, j_topo: float,
                 fidelity: int, loss: float):
        """Add a loss observation at given fidelity."""
        obs = Observation(arch=arch, j_topo=j_topo, fidelity=fidelity, loss=loss)
        self.observations.append(obs)
        self.loss_observations.append(obs)

    def _j_topo_prior(self, j_topo: float) -> Tuple[float, float]:
        """J_topo → E_floor prior (r=0.83 from Phase A).

        Returns (mean, std) for E_floor.
        """
        e_mean = J_TOPO_TO_E_INTERCEPT + J_TOPO_TO_E_SLOPE * (j_topo - 0.35)
        # Uncertainty: σ ≈ 0.15 at low J_topo observations, decreases with more data
        n_j = len(self.j_topo_observations)
        e_std = 0.15 / np.sqrt(n_j + 1)
        return e_mean, e_std

    def _norm_beta_prior(self, norm: str) -> float:
        """Normalization type → β prior (from Phase S1 v3)."""
        return BETA_NORM.get(norm, 0.180)

    def fit(self):
        """Fit GP to loss observations."""
        if not self.loss_observations:
            return

        X = arch_to_features([o.arch for o in self.loss_observations])
        y = np.array([o.loss for o in self.loss_observations])

        self.gp.fit(X, y)
        self._fitted = True

    def predict(self, archs: List[Architecture],
                j_topos: Optional[List[float]] = None,
                return_std: bool = True
                ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict E_floor for architectures.

        Combines:
        1. GP posterior (if fitted)
        2. J_topo prior (theoretical, r=0.83)
        3. Norm-informed β prior (empirical)
        """
        X = arch_to_features(archs)
        n = len(archs)

        if j_topos is None:
            j_topos = [0.35] * n  # fallback

        # GP prediction (if fitted)
        if self._fitted:
            gp_mean, gp_std = self.gp.predict(X, return_std=True)
        else:
            # Use J_topo prior only
            gp_mean = np.array([self._j_topo_prior(j)[0] for j in j_topos])
            gp_std = np.array([self._j_topo_prior(j)[1] for j in j_topos])

        return gp_mean, gp_std

    def predict_loss(self, arch: Architecture, j_topo: float,
                     fidelity: int) -> Tuple[float, float]:
        """Predict loss at given fidelity for a single architecture.

        Returns (mean, std).
        """
        mean, std = self.predict([arch], [j_topo], return_std=True)
        return float(mean[0]), float(std[0])

    def acquisition_score(self, arch: Architecture, j_topo: float,
                          fidelity: int, best_loss: float,
                          lambda_explore: float = 0.1) -> float:
        """Expected Improvement acquisition score.

        EI = (best_loss - mean) * Φ(z) + σ * φ(z)
        where z = (best_loss - mean) / σ
        """
        mean, std = self.predict_loss(arch, j_topo, fidelity)

        if std < 1e-6:
            return best_loss - mean

        z = (best_loss - mean) / std
        from scipy.stats import norm
        ei = (best_loss - mean) * norm.cdf(z) + std * norm.pdf(z)
        return ei

    def best_observation(self) -> Optional[Observation]:
        """Return the best (lowest loss) observation."""
        losses = [o.loss for o in self.loss_observations if o.loss is not None]
        if not losses:
            return None
        best_idx = np.argmin(losses)
        return self.loss_observations[best_idx]
