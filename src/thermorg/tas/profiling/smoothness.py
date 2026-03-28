# SPDX-License-Identifier: Apache-2.0

"""Sobolev smoothness estimation via kNN Graph Laplacian eigenfunction decay.

Phase 1: Data Geometric Profiling - Estimate s via kNN Graph Laplacian eigenfunction decay
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional
from scipy.sparse import csr_matrix

from ...utils.math import get_knn_graph, compute_graph_laplacian, estimate_eigendecay


class SmoothnessEstimator:
    """Estimates Sobolev smoothness s using graph Laplacian eigenfunction decay.
    
    The smoothness s characterizes how rapidly functions on the manifold vary.
    It's estimated by analyzing the decay rate of Laplacian eigenfunction powers,
    where smoother functions correspond to slower decay.
    
    The relationship is: λ_j ~ j^{2s/d} for eigenfunctions with regularity s.
    """
    
    def __init__(self, k_neighbors: int = 10, n_eigenvalues: int = 50):
        """Initialize SmoothnessEstimator.
        
        Args:
            k_neighbors: Number of nearest neighbors for graph construction
            n_eigenvalues: Number of eigenvalues to compute for decay estimation
        """
        self.k_neighbors = k_neighbors
        self.n_eigenvalues = n_eigenvalues
        self._s_estimate: Optional[float] = None
        
    def estimate_s(
        self, 
        y: NDArray[np.floating], 
        X: NDArray[np.floating],
        normalized: bool = True
    ) -> float:
        """Estimate Sobolev smoothness s.
        
        Uses the relationship between eigenfunction decay and smoothness.
        For a function y on the manifold, the smoothness s relates to how
        quickly the graph Laplacian eigenfunctions decay.
        
        The power spectrum of the graph Fourier transform decays as:
            |ŷ(λ_j)|² ~ λ_j^{-s}
        
        Args:
            y: Function values on manifold, shape (n_samples,)
            X: Data points, shape (n_samples, n_features)
            normalized: Whether to use normalized Laplacian
            
        Returns:
            Estimated smoothness s
        """
        n = X.shape[0]
        
        if n <= self.k_neighbors + 1:
            return 1.0  # Default smoothness
        
        # Build k-NN graph
        W = get_knn_graph(X, k=self.k_neighbors, symmetric=True)
        
        # Compute graph Laplacian
        L = compute_graph_laplacian(W, normalized=normalized)
        
        # Compute eigenvalues using sparse eigenvalue solver
        # For normalized Laplacian, eigenvalues are in [0, 2]
        n_eig = min(self.n_eigenvalues, n - 2)
        
        try:
            from scipy.sparse.linalg import eigsh
            eigenvalues, _ = eigsh(L.astype(float), k=n_eig + 1, which='SM')
            # Smallest eigenvalue should be ~0 (constant function kernel)
            eigenvalues = np.sort(eigenvalues)
        except Exception:
            # Fallback: use dense eigenvalue decomposition
            L_dense = L.toarray()
            eigenvalues = np.linalg.eigvalsh(L_dense)
            eigenvalues = np.sort(eigenvalues)[:n_eig + 1]
        
        # Filter out near-zero eigenvalues (numerical artifacts)
        eigenvalues = eigenvalues[eigenvalues > 1e-8]
        
        if len(eigenvalues) < 5:
            return 1.0  # Not enough eigenvalues
        
        # Compute smoothness from decay rate
        # For normalized Laplacian: eigenvalues ~ j^{2s/d}
        # So s ~ (d/2) * decay_rate
        decay_rate = estimate_eigendecay(eigenvalues[1:])  # Skip smallest (should be ~0)
        
        # Estimate d for scaling (use heuristic if needed)
        d_manifold = min(20, int(np.sqrt(n)))  # Rough estimate
        
        # s relates to decay rate: decay ~ j^{2s/d} means log(decay) ~ (2s/d) * log(j)
        # We already computed decay_rate = -slope of log(eigenvalues) vs index
        # So: s = decay_rate * d / 2
        s = decay_rate * d_manifold / 2.0
        
        # Clamp to reasonable range
        s = float(np.clip(s, 0.1, 5.0))
        
        self._s_estimate = s
        return s
    
    def estimate_s_from_signal(
        self,
        y: NDArray[np.floating],
        X: NDArray[np.floating],
        d_manifold: float,
        normalized: bool = True
    ) -> float:
        """Estimate smoothness using graph Fourier transform power spectrum.
        
        More direct estimation using the relationship between smoothness s
        and the decay of graph Fourier coefficients.
        
        Args:
            y: Signal/function values on manifold
            X: Data points
            d_manifold: Known or estimated manifold dimension
            normalized: Whether to use normalized Laplacian
            
        Returns:
            Estimated smoothness s
        """
        n = X.shape[0]
        
        if n <= self.k_neighbors + 1:
            return 1.0
        
        # Build graph and Laplacian
        W = get_knn_graph(X, k=self.k_neighbors, symmetric=True)
        L = compute_graph_laplacian(W, normalized=normalized)
        
        # Compute Laplacian eigenbasis
        n_eig = min(self.n_eigenvalues, n - 1)
        
        try:
            from scipy.sparse.linalg import eigsh
            eigenvalues, eigenfunctions = eigsh(L.astype(float), k=n_eig, which='SM')
        except Exception:
            L_dense = L.toarray()
            eigenvalues, eigenfunctions = np.linalg.eigh(L_dense)
            idx = np.argsort(eigenvalues)[:n_eig]
            eigenvalues = eigenvalues[idx]
            eigenfunctions = eigenfunctions[:, idx]
        
        # Compute graph Fourier transform of y
        # y_hat = U^T * y (where U is eigenfunction matrix)
        y_hat = eigenfunctions.T @ y
        
        # Power spectrum: |y_hat(λ_j)|²
        power_spectrum = y_hat ** 2
        
        # Sort by eigenvalue magnitude (skip zero eigenvalue)
        sort_idx = np.argsort(eigenvalues[1:])  # Skip smallest (constant)
        eigenvalues_sorted = eigenvalues[1:][sort_idx]
        power_sorted = power_spectrum[1:][sort_idx]
        
        # Fit decay: power ~ λ^{-s}
        # log(power) ~ -s * log(λ)
        valid = (eigenvalues_sorted > 1e-8) & (power_sorted > 1e-10)
        if np.sum(valid) < 5:
            return 1.0
        
        log_lambda = np.log(eigenvalues_sorted[valid])
        log_power = np.log(power_sorted[valid])
        
        # Linear regression: slope = -s
        from numpy.polynomial import polynomial as P
        s, _ = P.polyfit(log_lambda, log_power, 1)
        s = -s  # Flip sign
        
        # Clamp to reasonable range
        s = float(np.clip(s, 0.1, 5.0))
        
        self._s_estimate = s
        return s
    
    @property
    def s_estimate(self) -> Optional[float]:
        """Return last estimated smoothness value."""
        return self._s_estimate
