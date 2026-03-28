# SPDX-License-Identifier: Apache-2.0

"""Spectral Momentum Conservation (SMC) module.

Implements compression efficiency and spectral momentum operators
based on the SMC theory.
"""

from __future__ import annotations

import torch
from torch import Tensor


def compute_frobenius_norm(jacobian: Tensor) -> Tensor:
    """Compute Frobenius norm of Jacobian matrix.
    
    Args:
        jacobian: Jacobian matrix J of shape (d_out, d_in)
        
    Returns:
        Frobenius norm ||J||_F
    """
    return torch.linalg.norm(jacobian, ord="fro")


def compute_spectral_norm(jacobian: Tensor, n_power_iter: int = 10) -> Tensor:
    """Compute spectral norm (operator norm) of Jacobian via power iteration.
    
    Args:
        jacobian: Jacobian matrix J of shape (d_out, d_in)
        n_power_iter: Number of power iterations for estimation
        
    Returns:
        Spectral norm ||J||_2
    """
    d_out, d_in = jacobian.shape
    # Initialize with random vector
    u = torch.randn(d_out, device=jacobian.device, dtype=jacobian.dtype)
    u = u / torch.linalg.norm(u)
    v = torch.randn(d_in, device=jacobian.device, dtype=jacobian.dtype)
    v = v / torch.linalg.norm(v)
    
    for _ in range(n_power_iter):
        # Power iteration step
        jv = jacobian @ v
        sigma = torch.linalg.norm(jv)
        if sigma == 0:
            break
        u = jv / sigma
        
        ju = jacobian.T @ u
        v = ju / torch.linalg.norm(ju)
    
    final_norm = torch.linalg.norm(jacobian @ v)
    return final_norm


def effective_dimension(fro_norm: Tensor, spec_norm: Tensor) -> Tensor:
    """Compute effective dimension D_eff = ||J||_F^2 / ||J||_2^2.
    
    Args:
        fro_norm: Frobenius norm ||J||_F
        spec_norm: Spectral norm ||J||_2
        
    Returns:
        Effective dimension D_eff
    """
    return (fro_norm ** 2) / (spec_norm ** 2 + 1e-8)


def compression_efficiency(eff_dim: Tensor, manifold_dim: float) -> Tensor:
    """Compute compression efficiency η_l = D_eff / d_manifold.
    
    Args:
        eff_dim: Effective dimension D_eff
        manifold_dim: Manifold dimension d_manifold
        
    Returns:
        Compression efficiency η_l
    """
    return eff_dim / (manifold_dim + 1e-8)


def spectral_momentum_operator(jacobian: Tensor) -> Tensor:
    """Compute spectral momentum operator Π = J^T J.
    
    Args:
        jacobian: Jacobian matrix J of shape (d_out, d_in)
        
    Returns:
        Spectral momentum operator Π of shape (d_in, d_in)
    """
    return jacobian.T @ jacobian


def compute_smc_metrics(
    jacobian: Tensor,
    manifold_dim: float,
    n_power_iter: int = 10,
) -> dict[str, Tensor]:
    """Compute all SMC metrics from a Jacobian.
    
    Args:
        jacobian: Jacobian matrix J
        manifold_dim: Intrinsic manifold dimension
        n_power_iter: Power iteration steps for spectral norm
        
    Returns:
        Dictionary containing:
            - frobenius_norm: ||J||_F
            - spectral_norm: ||J||_2
            - effective_dim: D_eff
            - compression_eff: η_l
            - momentum_op: Π = J^T J
    """
    fro_norm = compute_frobenius_norm(jacobian)
    spec_norm = compute_spectral_norm(jacobian, n_power_iter)
    eff_dim = effective_dimension(fro_norm, spec_norm)
    comp_eff = compression_efficiency(eff_dim, manifold_dim)
    momentum_op = spectral_momentum_operator(jacobian)
    
    return {
        "frobenius_norm": fro_norm,
        "spectral_norm": spec_norm,
        "effective_dim": eff_dim,
        "compression_eff": comp_eff,
        "momentum_op": momentum_op,
    }
