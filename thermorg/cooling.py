#!/usr/bin/env python3
"""
ThermoRG Cooling Factor Module
==============================

Computes cooling factor φ(γ) for thermal regulation in network training.

The cooling factor describes how effective temperature decreases with training:
- γ: Training progress (0 = start, 1 = convergence)
- φ(γ): Remaining "thermal energy" fraction

Reference: ThermoRG Theory Framework v5-v8
"""

from __future__ import annotations

import math
from typing import Callable


def cooling_factor_linear(gamma: float) -> float:
    """
    Linear cooling: φ(γ) = 1 - γ
    
    Args:
        gamma: Training progress in [0, 1]
        
    Returns:
        Cooling factor in (0, 1]
    """
    return max(0.0, 1.0 - gamma)


def cooling_factor_exponential(
    gamma: float,
    rate: float = 2.0
) -> float:
    """
    Exponential cooling: φ(γ) = exp(-γ · rate)
    
    Faster initial cooling, slower later.
    
    Args:
        gamma: Training progress in [0, 1]
        rate: Cooling rate parameter
        
    Returns:
        Cooling factor in (0, 1]
    """
    return math.exp(-gamma * rate)


def cooling_factor_power_law(
    gamma: float,
    exponent: float = 0.5
) -> float:
    """
    Power-law cooling: φ(γ) = (1 - γ)^exponent
    
    Slower initial cooling, faster later.
    
    Args:
        gamma: Training progress in [0, 1]
        exponent: Power-law exponent (> 0)
        
    Returns:
        Cooling factor in (0, 1]
    """
    return math.pow(max(0.0, 1.0 - gamma), exponent)


def cooling_factor_cosine(
    gamma: float,
    T_max: float = 1.0,
    T_min: float = 0.1
) -> float:
    """
    Cosine annealing: φ(γ) = T_min + 0.5*(T_max - T_min)*(1 + cos(πγ))
    
    Smooth cosine schedule from T_max to T_min.
    
    Args:
        gamma: Training progress in [0, 1]
        T_max: Maximum temperature
        T_min: Minimum temperature
        
    Returns:
        Temperature in [T_min, T_max]
    """
    return T_min + 0.5 * (T_max - T_min) * (1 + math.cos(math.pi * gamma))


def get_cooling_factor(
    gamma: float,
    schedule: str = 'exponential',
    **kwargs
) -> float:
    """
    Compute cooling factor using specified schedule.
    
    Args:
        gamma: Training progress in [0, 1]
        schedule: One of 'linear', 'exponential', 'power_law', 'cosine'
        **kwargs: Additional parameters for specific schedule
        
    Returns:
        Cooling factor
        
    Example:
        >>> get_cooling_factor(0.5, 'exponential', rate=2.0)
        0.1353352832366127
        >>> get_cooling_factor(0.5, 'cosine', T_max=1.0, T_min=0.1)
        0.35
    """
    if schedule == 'linear':
        return cooling_factor_linear(gamma)
    elif schedule == 'exponential':
        rate = kwargs.get('rate', 2.0)
        return cooling_factor_exponential(gamma, rate)
    elif schedule == 'power_law':
        exponent = kwargs.get('exponent', 0.5)
        return cooling_factor_power_law(gamma, exponent)
    elif schedule == 'cosine':
        T_max = kwargs.get('T_max', 1.0)
        T_min = kwargs.get('T_min', 0.1)
        return cooling_factor_cosine(gamma, T_max, T_min)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


def phi_from_delta(
    n_s: int,
    phi_per_layer: float = 0.87
) -> float:
    """
    Compute total cooling factor from number of stride-2 layers.
    
    φ_total = φ^n_s
    
    Where φ ≈ 0.87 per stride-2 layer (from Δ ≈ 0.20 scaling dimension).
    
    Args:
        n_s: Number of stride-2 layers
        phi_per_layer: Cooling factor per stride-2 layer
        
    Returns:
        Total cooling factor
        
    Example:
        >>> phi_from_delta(0)  # No stride-2
        1.0
        >>> phi_from_delta(1)  # One stride-2
        0.87
        >>> phi_from_delta(2)  # Two stride-2s
        0.7569
    """
    return math.pow(phi_per_layer, n_s)


# Backward compatibility alias
phi = phi_from_delta
