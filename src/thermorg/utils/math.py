# SPDX-License-Identifier: Apache-2.0

"""Utility functions for ThermoRG package."""

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from typing import Optional


def compute_pairwise_distances(X: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute pairwise Euclidean distances.
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
        
    Returns:
        Distance matrix of shape (n_samples, n_samples)
    """
    # Use scipy for efficiency with large matrices
    from scipy.spatial.distance import pdist, squareform
    return squareform(pdist(X, metric='euclidean'))


def get_knn_graph(X: NDArray[np.floating], k: int, symmetric: bool = True) -> csr_matrix:
    """Build k-nearest-neighbors graph.
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
        k: Number of nearest neighbors
        symmetric: If True, make graph symmetric (undirected)
        
    Returns:
        Sparse adjacency matrix
    """
    from sklearn.neighbors import NearestNeighbors
    
    n = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(X)
    indices = nbrs.kneighbors(X, return_distance=False)
    
    # Remove self-loops (first column)
    indices = indices[:, 1:]
    
    # Build sparse matrix
    rows = np.repeat(np.arange(n), k)
    cols = indices.flatten()
    data = np.ones(n * k)
    
    W = csr_matrix((data, (rows, cols)), shape=(n, n))
    
    if symmetric:
        W = W + W.T
        W = (W > 0).astype(float)
    
    return W


def compute_graph_laplacian(W: csr_matrix, normalized: bool = True) -> csr_matrix:
    """Compute graph Laplacian.
    
    Args:
        W: Adjacency matrix
        normalized: If True, use normalized Laplacian
        
    Returns:
        Graph Laplacian matrix
    """
    from scipy.sparse import diags
    
    degrees = np.array(W.sum(axis=1)).flatten()
    
    if normalized:
        # L_norm = I - D^{-1/2} W D^{-1/2}
        d_inv_sqrt = diags(1.0 / np.sqrt(degrees + 1e-10))
        L = diags([1.0] * len(degrees)) - d_inv_sqrt @ W @ d_inv_sqrt
    else:
        # L = D - W
        D = diags(degrees)
        L = D - W
    
    return L


def estimate_eigendecay(eigenvalues: NDArray[np.floating], tol: float = 1e-6) -> float:
    """Estimate decay rate of eigenvalues.
    
    Fits exponential decay: λ_i ~ exp(-ρ * i)
    
    Args:
        eigenvalues: Sorted eigenvalues (descending)
        tol: Tolerance for numerical stability
        
    Returns:
        Decay rate ρ
    """
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues, tol)
    
    # Use first portion of spectrum for fitting
    n_fit = min(len(eigenvalues), 50)
    idx = np.arange(1, n_fit + 1)
    
    log_lambda = np.log(eigenvalues[:n_fit])
    
    # Linear regression in log space: log(λ) = -ρ * i + const
    rho, _ = np.polyfit(idx, log_lambda, 1)
    
    return max(-rho, tol)


def safe_divide(a: float, b: float, eps: float = 1e-10) -> float:
    """Safe division with epsilon fallback."""
    return a / (b + eps)


def safe_log(x: float, eps: float = 1e-10) -> float:
    """Safe logarithm with epsilon offset."""
    return np.log(max(x, eps))


def product_log(eta_ls: list[float]) -> float:
    """Compute product of η_l values."""
    if not eta_ls:
        return 1.0
    return np.prod(eta_ls)
