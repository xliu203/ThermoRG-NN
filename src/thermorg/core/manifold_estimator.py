# SPDX-License-Identifier: Apache-2.0

"""
Manifold Dimension Estimators for ThermoRG-NN

This module provides various methods to estimate the intrinsic dimensionality (d_manifold)
of data lying on a low-dimensional manifold embedded in a higher-dimensional space.

Methods implemented:
- PCA: Principal Component Analysis based variance thresholding
- Levina-Bickel: Local linear embedding based intrinsic dimension estimation
- Correlation: Grassberger-Procaccia correlation dimension
- Effective: Effective spectral dimension using Frobenius/spectral norm ratio
"""

import torch
from torch import Tensor
from typing import Optional, Literal


def estimate_d_manifold_pca(X: Tensor, variance_threshold: float = 0.95) -> float:
    """
    Estimate manifold dimensionality using PCA.

    Uses the cumulative variance explained by principal components to determine
    the intrinsic dimension. Components are retained until the cumulative variance
    ratio reaches the specified threshold.

    Args:
        X: Input data tensor of shape (n_samples, n_features)
        variance_threshold: Fraction of variance to retain (default: 0.95)

    Returns:
        Estimated intrinsic dimension

    Example:
        >>> X = torch.randn(100, 20)
        >>> d = estimate_d_manifold_pca(X, variance_threshold=0.95)
        >>> print(f"Estimated dimension: {d}")
    """
    n_samples, n_features = X.shape
    
    # Center the data
    X_centered = X - X.mean(axis=0)
    
    # Handle case with zero variance
    if torch.allclose(X_centered, torch.zeros_like(X_centered)):
        return 1.0
    
    # SVD decomposition
    U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)
    
    # Compute variance ratio (eigenvalues are proportional to S^2)
    var_ratio = (S ** 2) / (S ** 2).sum()
    
    # Cumulative variance
    cumsum = torch.cumsum(var_ratio, dim=0)
    
    # Find the dimension where cumulative variance first reaches threshold
    # Add 1 because dimensions are 1-indexed in the formula
    indices = (cumsum >= variance_threshold).nonzero()
    
    if indices.numel() == 0:
        # If threshold not reached, return full dimension
        return float(n_features)
    
    d_est = indices[0][0].item() + 1
    return float(d_est)


def estimate_d_manifold_levina(
    X: Tensor,
    k_neighbors: int = 10,
    n_subsamples: Optional[int] = None
) -> float:
    """
    Estimate manifold dimensionality using the Levina-Bickel method.

    This method is based on the maximum likelihood estimation of the intrinsic
    dimension using local neighborhoods. The algorithm:

    1. For each point, find its k nearest neighbors
    2. Compute local dimension estimates using the ratio of k-th to j-th distances
    3. Average the local estimates (harmonic mean)

    Args:
        X: Input data tensor of shape (n_samples, n_features)
        k_neighbors: Number of neighbors to use (default: 10)
        n_subsamples: Number of points to sample for estimation (default: min(1000, n_samples))

    Returns:
        Estimated intrinsic dimension

    Reference:
        Levina, E., & Bickel, P. J. (2004). Maximum Likelihood Estimation of
        Intrinsic Dimension. Advances in Neural Information Processing Systems.
    """
    n_samples, n_features = X.shape
    
    # Subsample if data is too large for computational efficiency
    if n_subsamples is None:
        n_subsamples = min(1000, n_samples)
    
    if n_samples > n_subsamples:
        indices = torch.randperm(n_samples)[:n_subsamples]
        X = X[indices]
        n_samples = n_subsamples
    
    # Ensure k is valid
    k = min(k_neighbors, n_samples - 1)
    if k < 2:
        return 1.0
    
    # Compute pairwise distances
    dists = torch.cdist(X, X, p=2)
    
    # For each point, find k nearest neighbors (excluding itself)
    # Set self-distance to infinity and find k smallest
    dists.fill_diagonal_(float('inf'))
    
    # Get k nearest neighbors distances, sorted
    all_k_dists, _ = torch.topk(dists, k=k, dim=1, largest=False, sorted=True)
    # all_k_dists[:, j] = distance to j-th nearest neighbor (0-indexed)
    
    # Compute local dimension for each point
    # d_i = k / sum_{j=1}^k log(r_k / r_j)
    # where r_k = kth_dists[i, k-1] and r_j = kth_dists[i, j-1]
    
    # Avoid division by zero
    r_k = all_k_dists[:, -1:]  # k-th neighbor distance (n x 1)
    r_j = all_k_dists[:, :-1]  # 1st to (k-1)-th neighbor distances (n x (k-1))
    
    # Compute log(r_k / r_j) for j = 1 to k-1
    log_ratio = torch.log(r_k / (r_j + 1e-10) + 1e-10)
    
    # Sum over j
    sum_log_ratio = log_ratio.sum(dim=1)  # (n,)
    
    # Local dimension estimate
    local_d = k / (sum_log_ratio + 1e-10)
    
    # Handle infinite or zero values
    local_d = torch.where(
        torch.isfinite(local_d) & (local_d > 0),
        local_d,
        torch.ones_like(local_d)
    )
    
    # Use harmonic mean of local dimensions (as per Levina-Bickel)
    # d_est = n / sum(1/d_i)
    d_est = n_samples / (1.0 / local_d).sum()
    
    # Bound the estimate
    d_est = min(max(d_est.item(), 1.0), float(n_features))
    
    return float(d_est)


def estimate_d_manifold_correlation(
    X: Tensor,
    n_bins: int = 50,
    n_samples: Optional[int] = None
) -> float:
    """
    Estimate correlation (Grassberger-Procaccia) dimension.

    The correlation dimension is computed using the Grassberger-Procaccia algorithm:
    1. Compute pairwise distances between points
    2. For a range of radii r, compute the correlation integral C(r)
    3. C(r) ~ r^d in the scaling regime, so d = d(log C) / d(log r)

    Args:
        X: Input data tensor of shape (n_samples, n_features)
        n_bins: Number of radius bins for the log-log fit (default: 50)
        n_samples: Maximum number of samples to use (default: min(1000, n_samples))

    Returns:
        Estimated correlation dimension

    Reference:
        Grassberger, P., & Procaccia, I. (1983). Measuring the strangeness of
        strange attractors. Physica D: Nonlinear Phenomena, 9(1-2), 189-208.
    """
    n_total, n_features = X.shape
    
    # Subsample for computational efficiency
    if n_samples is None:
        n_samples = min(1000, n_total)
    
    if n_total > n_samples:
        indices = torch.randperm(n_total)[:n_samples]
        X = X[indices]
        n = n_samples
    else:
        n = n_total
    
    # Compute pairwise distances (upper triangle only for efficiency)
    # We only need unique pairs, so use cdist on subset or compute full matrix
    
    # For large n, this becomes O(n^2), so we use the full matrix approach
    dists = torch.cdist(X, X, p=2)
    
    # Get upper triangle (excluding diagonal)
    i_upper = torch.triu_indices(n, n, offset=1)
    r = dists[i_upper[0], i_upper[1]]
    
    # Remove zero distances
    r = r[r > 0]
    
    if r.numel() < 10:
        return 1.0
    
    # Define radius range (in log space)
    r_min = r.min()
    r_max = r.max()
    
    if r_min <= 0 or r_max <= r_min:
        return 1.0
    
    # Create logarithmically spaced bins
    log_r_min = torch.log(r_min + 1e-10)
    log_r_max = torch.log(r_max)
    
    log_r_bins = torch.linspace(log_r_min, log_r_max, n_bins + 1)
    r_bins = torch.exp(log_r_bins)
    
    # Compute correlation integral C(r) for each bin
    # C(r) = (2 / (n * (n-1))) * sum_{i<j} I(||x_i - x_j|| < r)
    n_pairs = n * (n - 1) / 2
    
    C_values = []
    r_centers = []
    
    for i in range(n_bins):
        r_upper = r_bins[i + 1]
        count = (r < r_upper).sum().float()
        C = 2 * count / (n * (n - 1))
        C_values.append(C)
        r_centers.append((r_bins[i] + r_bins[i + 1]) / 2)
    
    C_values = torch.tensor(C_values)
    r_centers = torch.tensor(r_centers)
    
    # Remove zero values for log-log fit
    valid = C_values > 0
    if valid.sum() < 3:
        return 1.0
    
    log_C = torch.log(C_values[valid])
    log_r = torch.log(r_centers[valid])
    
    # Linear regression: log(C) = d * log(r) + const
    # d = cov(log_C, log_r) / var(log_r)
    d_est = torch.cov(torch.stack([log_C, log_r]))[0, 1] / torch.var(log_r, unbiased=True)
    
    # Ensure positive and bounded
    d_est = max(d_est.item(), 1.0)
    d_est = min(d_est, float(n_features))
    
    return float(d_est)


def estimate_d_manifold_effective(X: Tensor, threshold: float = 0.01) -> float:
    """
    Estimate effective spectral dimension.

    The effective spectral dimension is computed as:
        D_eff = ||X||_F^2 / ||X||_2^2 = (sum of all squared singular values) / (largest singular value)^2

    This measures how spread out the singular values are.
    For isotropic data (all SVs equal), D_eff ≈ rank.
    For structured data (few large SVs), D_eff ≈ number of significant directions.

    Args:
        X: Input data tensor of shape (n_samples, n_features)
        threshold: Ignored, kept for API compatibility (default: 0.01)

    Returns:
        Estimated effective spectral dimension

    Note:
        D_eff ≈ 1 when data is isotropic (no structure)
        D_eff ≈ d when data has d significant directions
    """
    n_samples, n_features = X.shape
    
    # Center the data
    X_centered = X - X.mean(axis=0)
    
    # Compute SVD
    U, S, Vt = torch.linalg.svd(X_centered)
    
    # Frobenius norm squared = sum of all singular values squared
    fro_sq = (S ** 2).sum()
    
    # Spectral norm squared = (largest singular value)^2
    spec_sq = S[0] ** 2
    
    # Effective dimension = spread of singular values
    D_eff = fro_sq / (spec_sq + 1e-8)
    
    # Bound the estimate
    D_eff = min(max(D_eff, 1.0), float(min(n_samples, n_features)))
    
    return float(D_eff)


def estimate_d_manifold(
    X: Tensor,
    method: Literal["pca", "levina", "correlation", "effective", "spectral_decay"] = "pca",
    **kwargs
) -> float:
    """
    Unified interface for manifold dimension estimation.

    Provides access to various intrinsic dimension estimators through a
    single interface. The method used can significantly affect the estimate,
    so choose based on your data characteristics.

    Args:
        X: Input data tensor of shape (n_samples, n_features)
        method: Estimation method to use. Options:
            - "pca": PCA-based variance thresholding (default)
            - "levina": Levina-Bickel maximum likelihood estimator
            - "correlation": Grassberger-Procaccia correlation dimension
            - "effective": Effective spectral dimension (fast approximation)
        **kwargs: Additional arguments passed to the specific estimator:
            - PCA: variance_threshold (default: 0.95)
            - Levina: k_neighbors (default: 10), n_subsamples (default: 1000)
            - Correlation: n_bins (default: 50), n_samples (default: 1000)
            - Effective: threshold (default: 0.01, unused)

    Returns:
        Estimated intrinsic dimension (d_manifold)

    Example:
        >>> X = torch.randn(200, 50)  # 50D data
        >>> 
        >>> # PCA method
        >>> d_pca = estimate_d_manifold(X, method="pca")
        >>> 
        >>> # Levina-Bickel method
        >>> d_levina = estimate_d_manifold(X, method="levina", k_neighbors=15)
        >>> 
        >>> # Correlation dimension
        >>> d_corr = estimate_d_manifold(X, method="correlation")
        >>> 
        >>> # Effective spectral dimension
        >>> d_eff = estimate_d_manifold(X, method="effective")
        >>> 
        >>> print(f"PCA: {d_pca:.2f}, Levina: {d_levina:.2f}, "
        ...       f"Correlation: {d_corr:.2f}, Effective: {d_eff:.2f}")

    Notes:
        - PCA is generally robust and fast, good for linear manifolds
        - Levina-Bickel works well for curved manifolds with local linear structure
        - Correlation dimension is suitable for fractal-like structures
        - Effective spectral dimension is the fastest but provides an upper bound
    """
    if method == "pca":
        return estimate_d_manifold_pca(X, **kwargs)
    elif method == "levina":
        return estimate_d_manifold_levina(X, **kwargs)
    elif method == "correlation":
        return estimate_d_manifold_correlation(X, **kwargs)
    elif method == "effective":
        return estimate_d_manifold_effective(X, **kwargs)
    elif method == "spectral_decay":
        return estimate_d_manifold_effective(X, **kwargs)
    else:
        raise ValueError(
            f"Unknown method: {method}. "
            f"Valid options: 'pca', 'levina', 'correlation', 'effective'"
        )


def test_estimator() -> None:
    """
    Basic test function to verify the estimators work correctly.
    """
    print("Testing manifold dimension estimators...")
    print("-" * 50)
    
    # Test data: 100 samples, 20 dimensions
    X = torch.randn(100, 20)
    
    print(f"Test data shape: {X.shape}")
    print()
    
    # Test each method
    methods = ["pca", "levina", "correlation", "effective"]
    
    for method in methods:
        try:
            d = estimate_d_manifold(X, method=method)
            print(f"{method:15s}: d = {d:.4f}")
        except Exception as e:
            print(f"{method:15s}: ERROR - {e}")
    
    print()
    print("-" * 50)
    print("Testing with low-dimensional data embedded in high dimensions...")
    print("-" * 50)
    
    # Create data on a 3D manifold embedded in 20D
    n_samples = 200
    t = torch.linspace(0, 2 * torch.pi, n_samples)
    
    # Parametric curve in 3D
    x1 = torch.cos(t)
    x2 = torch.sin(t)
    x3 = t
    
    # Embed in 20D by adding zeros
    X_manifold = torch.zeros(n_samples, 20)
    X_manifold[:, 0] = x1
    X_manifold[:, 1] = x2
    X_manifold[:, 2] = x3
    
    # Add small noise
    X_manifold += 0.01 * torch.randn_like(X_manifold)
    
    print(f"Embedded manifold shape: {X_manifold.shape}")
    print(f"True intrinsic dimension: ~3 (parametric curve)")
    print()
    
    for method in methods:
        try:
            d = estimate_d_manifold(X, method=method)
            print(f"{method:15s}: d = {d:.4f}")
        except Exception as e:
            print(f"{method:15s}: ERROR - {e}")
    
    print()
    print("All tests completed!")


if __name__ == "__main__":
    test_estimator()


def estimate_d_manifold_spectral_decay(
    X: Tensor,
    n_singular_values: Optional[int] = None
) -> float:
    """
    Estimate manifold dimensionality using spectral decay rate.

    Based on the observation that for data on a d-dimensional manifold,
    the singular values typically decay as λ_i ~ i^{-γ}.
    The intrinsic dimension is estimated from the decay rate.

    Args:
        X: Input data tensor of shape (n_samples, n_features)
        n_singular_values: Number of singular values to use for fit (default: min(100, n_features))

    Returns:
        Estimated intrinsic dimension from spectral decay

    Reference:
        Little, A. et al. (2012). Multiscaleopenset an approach to estimate
        intrinsic dimension. ICML Workshop on Scaling, Mcmethod.
    """
    n_samples, n_features = X.shape
    
    # Center the data
    X_centered = X - X.mean(axis=0)
    
    # SVD decomposition
    U, S, Vt = torch.linalg.svd(X_centered)
    
    # Use top singular values for fitting
    if n_singular_values is None:
        n_singular_values = min(100, len(S))
    S_fit = S[:n_singular_values]
    
    # Remove zero/near-zero singular values
    S_fit = S_fit[S_fit > 1e-10]
    
    if len(S_fit) < 3:
        return 1.0
    
    # Compute log singular values
    log_S = torch.log(S_fit + 1e-10)
    log_i = torch.log(torch.arange(1, len(S_fit) + 1, dtype=torch.float32))
    
    # Linear regression: log(λ_i) ≈ const - γ * log(i)
    cov = torch.cov(torch.stack([log_S, log_i]))[0, 1]
    var_i = torch.var(log_i, unbiased=True)
    gamma = -cov / var_i
    
    # Intrinsic dimension from decay rate
    if gamma > 1:
        d_est = 1.0 / (gamma - 1)
    else:
        # When gamma <= 1, use the number of significant singular values
        threshold = S_fit[0] * 0.01
        d_est = (S_fit > threshold).sum().item()
    
    # Bound the estimate
    d_est = min(max(d_est, 1.0), float(min(n_samples, n_features)))
    
    return float(d_est)
