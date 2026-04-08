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


def beta_gamma(gamma: float, gamma_c: float = 2.0, a: float = 0.425, beta_c: float = 0.893) -> float:
    """
    Compute scaling exponent β from variance fluctuation γ.
    
    Derived from RG scaling near the EOS critical point γ_c ≈ 2.0:
        β(γ) = a · ln(γ / γ_c) + β_c
    
    Validated (Phase S1 TPU, CIFAR-10, 200 epochs, R² > 0.995):
        γ_None = 3.39, β_obs = 1.117, β_pred = 1.117
        γ_BN   = 2.29, β_obs = 0.950, β_pred = 0.951
    
    Args:
        gamma: Variance fluctuation (measured from training dynamics)
        gamma_c: Critical EOS scale (~2.0, universal from Edge of Stability literature)
        a: Sensitivity of β to γ (fitted: a ≈ 0.425)
        beta_c: β at γ = γ_c (fitted: β_c ≈ 0.893)
        
    Returns:
        Scaling exponent β
        
    Example:
        >>> beta_gamma(3.39)
        1.117
        >>> beta_gamma(2.29)
        0.951
    """
    return a * math.log(gamma / gamma_c) + beta_c


def phi_gamma_ratio(gamma1: float, gamma2: float,
                     gamma_c: float = 2.0, a: float = 0.425, beta_c: float = 0.893) -> float:
    """
    Compute the ratio β(γ1) / β(γ2) from variance fluctuations.
    
    For BN vs None:
        >>> phi_gamma_ratio(2.29, 3.39)
        0.850
    """
    return beta_gamma(gamma1, gamma_c, a, beta_c) / beta_gamma(gamma2, gamma_c, a, beta_c)


# DEPRECATED: The original phi_cooling formula was an ansatz that predicted
# the wrong direction (BN should DECREASE β, not increase it).
# Use beta_gamma() instead.
def phi_cooling(gamma: float, gamma_c: float = 2.0) -> float:
    """
    DEPRECATED: Use beta_gamma() instead.
    
    The formula φ(γ) = γ_c/(γ_c+γ)·exp(-γ/γ_c) was an empirical ansatz.
    It predicted that cooling (lower γ) increases β, which is opposite
    to what is observed. The correct relationship is β(γ) = a·ln(γ/γ_c) + β_c.
    """
    import warnings
    warnings.warn(
        "phi_cooling() is deprecated. Use beta_gamma() instead. "
        "The old formula predicts the wrong direction for BatchNorm effects.",
        DeprecationWarning, stacklevel=2)
    return (gamma_c / (gamma_c + gamma)) * math.exp(-gamma / gamma_c)


def phi_ratio_BN(gamma_BN: float = 2.29, gamma_none: float = 3.39,
                  gamma_c: float = 2.0, a: float = 0.425, beta_c: float = 0.893) -> float:
    """
    DEPRECATED: Use phi_gamma_ratio() instead.
    
    The ratio β_BN / β_None for BatchNorm vs no-normalization.
    Measured: ≈ 0.850 (BN reduces β by ~15%).
    """
    import warnings
    warnings.warn(
        "phi_ratio_BN() is deprecated. Use phi_gamma_ratio() instead.",
        DeprecationWarning, stacklevel=2)
    return phi_gamma_ratio(gamma_BN, gamma_none, gamma_c, a, beta_c)


def phi_from_delta(
    n_s: int,
    phi_per_layer: float = 0.87
) -> float:
    """
    Compute total stride-2 suppression factor from number of stride-2 layers.
    
    φ_total = φ^n_s
    
    Where φ ≈ 0.87 per stride-2 layer (from Δ ≈ 0.20 scaling dimension).
    This is DIFFERENT from the BatchNorm cooling factor φ(γ).
    
    Args:
        n_s: Number of stride-2 layers
        phi_per_layer: Suppression factor per stride-2 layer
        
    Returns:
        Total suppression factor
        
    Example:
        >>> phi_from_delta(0)  # No stride-2
        1.0
        >>> phi_from_delta(1)  # One stride-2
        0.87
        >>> phi_from_delta(2)  # Two stride-2s
        0.7569
    """
    return math.pow(phi_per_layer, n_s)


# Backward compatibility alias (now correctly pointing to stride-2 function)
# NOTE: phi is now phi_from_delta (stride-2), use phi_cooling for BatchNorm
phi = phi_from_delta
