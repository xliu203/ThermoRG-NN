#!/usr/bin/env python3
"""
ThermoRG Utilities Module
=========================

Common utilities for ThermoRG computations.

Includes:
- Manifold dimension estimation
- Capacity bound computation
- Model architecture utilities
- Logging helpers
"""

from __future__ import annotations

import math
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn


# ============================================================================
# Manifold Estimation
# ============================================================================

def estimate_d_manifold(X: np.ndarray, variance_threshold: float = 0.95) -> float:
    """
    Estimate intrinsic dimension of data manifold using PCA.
    
    Args:
        X: Data array of shape (n_samples, n_features)
        variance_threshold: Cumulative variance to explain (default 0.95)
        
    Returns:
        Estimated manifold dimension
        
    Example:
        >>> X = np.random.randn(1000, 784)
        >>> d = estimate_d_manifold(X)
        >>> print(f"Manifold dimension: {d}")
    """
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=variance_threshold)
        pca.fit(X)
        return float(pca.n_components_)
    except ImportError:
        # Fallback: simple eigenvalue-based estimate
        X_centered = X - X.mean(axis=0)
        cov = np.dot(X_centered.T, X_centered) / X.shape[0]
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues[::-1]  # Sort descending
        total = eigenvalues.sum()
        cumsum = np.cumsum(eigenvalues) / total
        n_components = np.searchsorted(cumsum, variance_threshold) + 1
        return float(n_components)


def compute_capacity_bound(
    d_manifold: float,
    n_samples: int,
    safety_factor: float = 2.0
) -> float:
    """
    Theory-grounded D_eff upper bound from covering numbers.
    
    Bound: D_eff <= d_manifold * (log n_samples + 1) * safety_factor
    
    Derived from:
      - Covering number theory: ε-covering of d-manifold needs ~(R/ε)^d cells
      - Setting ε ~ 1/√n_samples gives log(1/ε) = O(log n_samples)
    
    Args:
        d_manifold: Intrinsic manifold dimension
        n_samples: Number of training samples
        safety_factor: Relaxation factor (default 2.0)
        
    Returns:
        Upper bound on total D_eff
    """
    base_bound = d_manifold * (math.log(n_samples) + 1)
    return base_bound * safety_factor


# ============================================================================
# Model Architecture Utilities
# ============================================================================

def get_layer_info(model: nn.Module) -> List[Dict[str, Any]]:
    """
    Get information about all layers in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        List of layer info dicts
    """
    layers = []
    for name, module in model.named_modules():
        info = {'name': name, 'type': type(module).__name__}
        
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            info['params'] = sum(p.numel() for p in module.parameters())
            
            if isinstance(module, nn.Conv2d):
                info['in_channels'] = module.in_channels
                info['out_channels'] = module.out_channels
                info['kernel_size'] = module.kernel_size
                info['stride'] = module.stride
            elif isinstance(module, nn.Linear):
                info['in_features'] = module.in_features
                info['out_features'] = module.out_features
                
        layers.append(info)
    return layers


def count_stride2_layers(model: nn.Module) -> int:
    """
    Count number of stride-2 convolution layers.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of stride-2 conv layers
    """
    count = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            if module.stride[0] == 2:
                count += 1
    return count


def count_maxpool_layers(model: nn.Module) -> int:
    """
    Count number of max pooling layers.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of MaxPool2d layers
    """
    count = 0
    for module in model.modules():
        if isinstance(module, nn.MaxPool2d):
            count += 1
    return count


# ============================================================================
# Logging Utilities
# ============================================================================

def setup_logger(
    name: str = 'thermorg',
    level: int = logging.INFO,
    log_file: Optional[Path] = None
) -> logging.Logger:
    """
    Setup ThermoRG logger.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file to write logs
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    console = logging.StreamHandler()
    console.setLevel(level)
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console.setFormatter(fmt)
    logger.addHandler(console)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    
    return logger


# ============================================================================
# JSON/Serialization Utilities
# ============================================================================

def save_results(
    results: Dict[str, Any],
    path: Path,
    append: bool = False
) -> None:
    """
    Save results to JSON file.
    
    Args:
        results: Results dictionary
        path: Output file path
        append: If True, append to existing file
    """
    import json
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if append and path.exists():
        with open(path, 'r') as f:
            existing = json.load(f)
        existing.update(results)
        results = existing
    
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)


def load_results(path: Path) -> Dict[str, Any]:
    """
    Load results from JSON file.
    
    Args:
        path: Input file path
        
    Returns:
        Results dictionary
    """
    import json
    with open(path, 'r') as f:
        return json.load(f)


# ============================================================================
# Math Utilities
# ============================================================================

def clamp(x: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp value to range."""
    return max(min_val, min(max_val, x))


def safe_log(x: float, eps: float = 1e-10) -> float:
    """Safe logarithm that handles near-zero values."""
    return math.log(max(x, eps))


def geometric_mean(values: List[float], eps: float = 1e-10) -> float:
    """Compute geometric mean of values."""
    if not values:
        return 1.0
    product = math.prod(max(v, eps) for v in values)
    return math.pow(product, 1.0 / len(values))
