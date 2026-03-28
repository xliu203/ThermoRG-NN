# SPDX-License-Identifier: Apache-2.0

"""Manifold dimension estimation module.

Implements intrinsic dimension estimation and dynamic manifold
dimension evolution based on compression efficiency.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from typing import Callable


def levina_bickel_estimator(data: np.ndarray, k: int = 10) -> float:
    """Levina-Bickel intrinsic dimension estimator.
    
    Args:
        data: Data matrix of shape (n_samples, n_features)
        k: Number of neighbors to use
        k: Maximum distance exponent to consider
        
    Returns:
        Estimated intrinsic dimension
    """
    n_samples, n_features = data.shape
    
    # Compute k-nearest neighbor distances
    from scipy.spatial import KDTree
    tree = KDTree(data)
    distances, _ = tree.query(data, k=k + 1)
    
    # distances[:, 0] is zero (self), use distances[:, 1:k+1]
    distances = distances[:, 1:k + 1]  # (n_samples, k)
    
    # Compute log ratios
    log_ratios = np.log(distances[:, k - 1:] + 1e-10) - np.log(distances[:, :1] + 1e-10)
    
    # Sum over samples
    S = np.sum(log_ratios, axis=1)  # (n_samples,)
    
    # MLE estimator
    denom = n_samples * np.log(k) - np.sum(S)
    d_hat = denom / (np.sum(S) + 1e-10)
    
    return max(d_hat, 1.0)


def dynamic_manifold演化(
    prev_dim: float,
    compression_eff: float,
) -> float:
    """Compute dynamic manifold dimension evolution.
    
    d_manifold^(l) = η_l * d_manifold^(l-1)
    
    Args:
        prev_dim: Previous layer manifold dimension
        compression_eff: Compression efficiency η_l
        
    Returns:
        Updated manifold dimension
    """
    return compression_eff * prev_dim


def estimate_from_jacobian(
    jacobian: Tensor,
    method: str = "svd",
    svd_thresh: float = 0.99,
) -> float:
    """Estimate manifold dimension from Jacobian singular values.
    
    Args:
        jacobian: Jacobian matrix J
        method: Method to use ('svd', 'threshold')
        svd_thresh: Cumulative variance threshold for SVD method
        
    Returns:
        Estimated manifold dimension
    """
    # Compute SVD
    s = torch.linalg.svdvals(jacobian)
    s = s[s > 1e-8]  # Filter near-zero singular values
    
    if s.numel() == 0:
        return 1.0
    
    if method == "svd":
        # Count singular values needed to explain svd_thresh variance
        variances = s ** 2
        cumvar = torch.cumsum(variances, dim=0) / torch.sum(variances)
        d = (cumvar < svd_thresh).sum() + 1
        return float(d)
    
    elif method == "threshold":
        # Count singular values above threshold
        max_s = s[0]
        thresh = max_s * 1e-3
        d = (s > thresh).sum().item()
        return float(d)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def batch_manifold_tracking(
    jacobians: list[Tensor],
    initial_dim: float = 1.0,
) -> list[float]:
    """Track manifold dimension across layers.
    
    Args:
        jacobians: List of layer Jacobians
        initial_dim: Initial manifold dimension
        
    Returns:
        List of manifold dimensions for each layer
    """
    from .smc import compute_smc_metrics
    
    dims = [initial_dim]
    current_dim = initial_dim
    
    for j in jacobians:
        metrics = compute_smc_metrics(j, current_dim)
        current_dim = dynamic_manifold演化(current_dim, metrics["compression_eff"].item())
        dims.append(current_dim)
    
    return dims
