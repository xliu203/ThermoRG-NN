# SPDX-License-Identifier: Apache-2.0

"""Manifold dimension estimation using Levina-Bickel maximum likelihood method.

Phase 1: Data Geometric Profiling - Estimate d_manifold via Maximum-Likelihood
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple


class ManifoldEstimator:
    """Estimates intrinsic manifold dimension using Levina-Bickel method.
    
    The Levina-Bickel method estimates the intrinsic dimension d of a manifold
    using a maximum likelihood estimator based on nearest neighbor distances.
    
    The method works by analyzing the distribution of k-nearest neighbor distances
    and fitting the expected distribution under a uniform distribution on a 
    d-dimensional manifold.
    """
    
    def __init__(self, k_max: int = 20):
        """Initialize ManifoldEstimator.
        
        Args:
            k_max: Maximum number of nearest neighbors to consider for estimation.
                   Higher values give smoother estimates but are more computationally
                   expensive.
        """
        self.k_max = k_max
        self._d_estimate: Optional[float] = None
        
    def estimate_d(self, X: NDArray[np.floating], k: Optional[int] = None) -> float:
        """Estimate intrinsic dimension d_manifold.
        
        Uses the Levina-Bickel maximum likelihood estimator which computes
        the ratio of average log distances to estimate d.
        
        Args:
            X: Data matrix of shape (n_samples, n_features)
            k: Number of nearest neighbors to use (defaults to k_max)
            
        Returns:
            Estimated intrinsic dimension d_manifold
        """
        n, d_features = X.shape
        k = k or self.k_max
        
        # Need at least k+1 samples
        if n <= k + 1:
            return float(d_features)
        
        # Compute pairwise distances efficiently using sklearn
        from sklearn.neighbors import NearestNeighbors
        
        # Use k+1 because the first neighbor is the point itself
        nbrs = NearestNeighbors(n_neighbors=min(k + 1, n), algorithm='ball_tree').fit(X)
        distances, _ = nbrs.kneighbors(X)
        
        # Remove self-distances (first column)
        distances = distances[:, 1:k + 1]
        
        # Levina-Bickel MLE formula:
        # d_hat = [1 / (n * k)] * sum_{i=1}^n sum_{j=1}^k [log(r_{ij}) - log(k)]
        # where r_{ij} is the j-th nearest neighbor distance of point i
        
        # Actually, the standard formula is:
        # d = (1 / n) * sum_{i=1}^n d_i
        # where d_i = [1 / (k-1)] * sum_{j=1}^{k-1} log(r_{ij} / r_{ik})
        
        # Use the improved Levina-Bickel formula with ratio correction
        d_estimates = []
        for i in range(n):
            d_i = 0.0
            for j in range(k - 1):
                # Ratio of consecutive distances
                ratio = distances[i, j] / distances[i, k - 1]
                if ratio > 0:
                    d_i += np.log(ratio)
            d_estimates.append(-d_i / (k - 1) if k > 1 else d_features)
        
        d_manifold = np.mean(d_estimates)
        
        # Clamp to valid range [1, d_features]
        d_manifold = float(np.clip(d_manifold, 1.0, float(d_features)))
        
        self._d_estimate = d_manifold
        return d_manifold
    
    def estimate_d_with_confidence(
        self, X: NDArray[np.floating], k: Optional[int] = None
    ) -> Tuple[float, float]:
        """Estimate d_manifold with confidence interval.
        
        Args:
            X: Data matrix
            k: Number of nearest neighbors
            
        Returns:
            Tuple of (d_estimate, standard_error)
        """
        n, d_features = X.shape
        k = k or self.k_max
        
        if n <= k + 1:
            return float(d_features), 0.0
        
        from sklearn.neighbors import NearestNeighbors
        
        nbrs = NearestNeighbors(n_neighbors=min(k + 1, n), algorithm='ball_tree').fit(X)
        distances, _ = nbrs.kneighbors(X)
        distances = distances[:, 1:k + 1]
        
        d_estimates = []
        for i in range(n):
            d_i = 0.0
            for j in range(k - 1):
                ratio = distances[i, j] / distances[i, k - 1]
                if ratio > 0:
                    d_i += np.log(ratio)
            d_estimates.append(-d_i / (k - 1) if k > 1 else d_features)
        
        d_estimates = np.array(d_estimates)
        d_manifold = float(np.clip(np.mean(d_estimates), 1.0, float(d_features)))
        std_error = float(np.std(d_estimates) / np.sqrt(n))
        
        self._d_estimate = d_manifold
        return d_manifold, std_error
    
    @property
    def d_estimate(self) -> Optional[float]:
        """Return last estimated d_manifold value."""
        return self._d_estimate
