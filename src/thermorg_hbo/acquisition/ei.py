"""
Expected Improvement acquisition function for ThermoRG-HBO.
"""

from typing import List, Tuple
import numpy as np

try:
    from scipy.stats import norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from ..arch.encoding import Architecture
from ..surrogate.gp import GPSurrogate


class EIAcquisition:
    """Expected Improvement acquisition function.

    EI(x) = E[max(0, f_best - f(x))]

    This prioritizes candidates that are:
    1. Predicted to have lower loss (exploitation)
    2. Have high uncertainty (exploration)
    """

    def __init__(self, gp: GPSurrogate, xi: float = 0.01):
        """
        Args:
            gp: GP surrogate model
            xi: exploration parameter (higher = more exploration)
        """
        self.gp = gp
        self.xi = xi

    def score(self, arch: Architecture, j_topo: float,
              fidelity: int) -> float:
        """Compute EI score for one architecture."""
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy required for EIAcquisition")

        mean, std = self.gp.predict_loss(arch, j_topo, fidelity)
        best = self.gp.best_observation()

        if best is None or best.loss is None:
            # No best yet — use maximum uncertainty
            return std

        best_loss = best.loss

        # EI with exploration parameter
        if std < 1e-8:
            return max(0, best_loss - mean)

        z = (best_loss - mean - self.xi) / std
        ei = (best_loss - mean - self.xi) * norm.cdf(z) + std * norm.pdf(z)
        return max(0.0, ei)

    def score_batch(self, archs: List[Architecture],
                    j_topos: List[float],
                    fidelities: List[int]) -> np.ndarray:
        """Compute EI scores for a batch."""
        scores = []
        for arch, j_topo, fid in zip(archs, j_topos, fidelities):
            scores.append(self.score(arch, j_topo, fid))
        return np.array(scores)

    def select_top_k(self, archs: List[Architecture],
                     j_topos: List[float],
                     fidelities: List[int],
                     k: int = 5) -> List[Tuple[int, float]]:
        """Select top-k architectures by EI score.

        Returns list of (index, score) tuples sorted by score descending.
        """
        scores = self.score_batch(archs, j_topos, fidelities)
        top_k_idx = np.argsort(scores)[::-1][:k]
        return [(int(idx), float(scores[idx])) for idx in top_k_idx]
