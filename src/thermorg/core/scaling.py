# SPDX-License-Identifier: Apache-2.0

"""Scaling law predictions module.

Implements unified scaling laws and optimal temperature theorem
based on the SMC theory framework.
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import Optional


def unified_scaling_law(
    compression_effs: list[float],
    s: float,
    d_manifold: float,
    psi: float = 1.0,
    phi: float = 1.0,
    epsilon: float = 1.0,
) -> float:
    """Compute scaling exponent α using unified scaling law.
    
    α = (2s/d_manifold) * ∏η_l * ψ * φ * ε
    
    Args:
        compression_effs: List of compression efficiencies η_l for each layer
        s: Scaling parameter (e.g., compute or data scale)
        d_manifold: Manifold dimension
        psi: Architecture efficiency factor
        phi: Optimization efficiency factor
        epsilon: Generalization efficiency factor
        
    Returns:
        Scaling exponent α
    """
    product_eta = torch.prod(torch.tensor(compression_effs)).item()
    alpha = (2 * s / d_manifold) * product_eta * psi * phi * epsilon
    return alpha


def optimal_temperature(
    critical_temp: float,
    fraction: float = 2.0 / 3.0,
) -> float:
    """Compute optimal effective temperature.
    
    T_eff* = (2/3) * T_c
    
    Args:
        critical_temp: Critical temperature T_c
        fraction: Fractional multiplier (default 2/3)
        
    Returns:
        Optimal effective temperature
    """
    return fraction * critical_temp


def compute_scaling_prediction(
    compression_effs: list[float],
    base_error: float,
    s: Tensor,
    d_manifold: float,
    psi: float = 1.0,
    phi: float = 1.0,
    epsilon: float = 1.0,
) -> Tensor:
    """Predict error using scaling law.
    
    Args:
        compression_effs: Compression efficiencies per layer
        base_error: Base error at initial scale
        s: Scale parameter (can be tensor for batch prediction)
        d_manifold: Manifold dimension
        psi: Architecture efficiency factor
        phi: Optimization efficiency factor
        epsilon: Generalization efficiency factor
        
    Returns:
        Predicted error
    """
    alpha = unified_scaling_law(
        compression_effs, s.item() if torch.is_tensor(s) else s,
        d_manifold, psi, phi, epsilon
    )
    
    if torch.is_tensor(s):
        return base_error * (s ** alpha)
    else:
        return base_error * (s ** alpha)


def temperature_effect(
    temp_ratio: float,
    beta: float = 1.0,
) -> float:
    """Compute temperature effect on generalization.
    
    Args:
        temp_ratio: T_eff / T_c ratio
        beta: Sensitivity parameter
        
    Returns:
        Temperature scaling factor
    """
    return torch.exp(-beta * (temp_ratio - 2.0 / 3.0) ** 2)


class ScalingLawPredictor:
    """Predictor for neural scaling laws using SMC theory."""
    
    def __init__(
        self,
        compression_effs: list[float],
        d_manifold: float,
        base_error: float = 1.0,
    ):
        self.compression_effs = compression_effs
        self.d_manifold = d_manifold
        self.base_error = base_error
        self.alpha = None
    
    def fit(self, scales: Tensor, errors: Tensor) -> float:
        """Fit scaling exponent from observed data.
        
        Args:
            scales: Observed scales
            errors: Observed errors
            
        Returns:
            Fitted scaling exponent α
        """
        log_ratio = torch.log(errors / self.base_error)
        log_scales = torch.log(scales)
        
        self.alpha = (log_ratio / log_scales).mean().item()
        return self.alpha
    
    def predict(self, scale: float) -> float:
        """Predict error at given scale.
        
        Args:
            scale: Target scale
            
        Returns:
            Predicted error
        """
        if self.alpha is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.base_error * (scale ** self.alpha)
    
    def optimal_temp(self, critical_temp: float) -> float:
        """Compute optimal temperature for given critical temperature.
        
        Args:
            critical_temp: Critical temperature
            
        Returns:
            Optimal effective temperature
        """
        return optimal_temperature(critical_temp)
