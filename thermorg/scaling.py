#!/usr/bin/env python3
"""
ThermoRG Scaling Laws Module
=============================

Implements D-scaling law fitting and predictions.

D-scaling law: L(D) = α · D^(-β) + E

Where:
- D: Training set size
- α: Pre-asymptotic coefficient (initial complexity penalty)
- β: Scaling exponent (learning efficiency)
- E: Asymptotic floor (irreducible error)

Reference: ThermoRG Theory Framework v5-v8
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit


def scaling_law(D: np.ndarray, alpha: float, beta: float, epsilon: float) -> np.ndarray:
    """
    D-scaling law: L(D) = α · D^(-β) + E
    
    Args:
        D: Training set size (array)
        alpha: Pre-asymptotic coefficient
        beta: Scaling exponent
        epsilon: Asymptotic floor
        
    Returns:
        Predicted loss values
    """
    return alpha * np.power(D, -beta) + epsilon


def fit_scaling_law(
    D: np.ndarray,
    L: np.ndarray,
    alpha_bounds: Tuple[float, float] = (1.0, 500.0),
    beta_bounds: Tuple[float, float] = (0.01, 2.0),  # Extended to 2.0 for heating regime (β > 1)
    epsilon_bounds: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[float, float, float, float]:
    """
    Fit D-scaling law to data.
    
    Args:
        D: Training set sizes
        L: Observed losses
        alpha_bounds: Bounds for alpha
        beta_bounds: Bounds for beta
        epsilon_bounds: Bounds for epsilon
        
    Returns:
        Tuple of (alpha, beta, epsilon, rmse)
        
    Example:
        >>> D = np.array([100, 500, 1000, 5000])
        >>> L = np.array([0.8, 0.4, 0.25, 0.15])
        >>> alpha, beta, eps, rmse = fit_scaling_law(D, L)
        >>> print(f"α={alpha:.2f}, β={beta:.3f}, E={eps:.3f}, RMSE={rmse:.4f}")
    """
    p0 = [100.0, 0.5, 0.5]  # Initial guess (β often > 0.3 in heating regime)
    
    bounds = (
        [alpha_bounds[0], beta_bounds[0], epsilon_bounds[0]],
        [alpha_bounds[1], beta_bounds[1], epsilon_bounds[1]]
    )
    
    try:
        popt, pcov = curve_fit(
            scaling_law, D, L,
            p0=p0,
            bounds=bounds,
            maxfev=10000
        )
        alpha, beta, epsilon = popt
        
        # Compute RMSE
        L_pred = scaling_law(D, alpha, beta, epsilon)
        rmse = np.sqrt(np.mean((L - L_pred) ** 2))
        
        return alpha, beta, epsilon, rmse
        
    except RuntimeError:
        # Fallback: simple linear regression on log-log
        log_D = np.log(D)
        log_L = np.log(L - np.min(L) + 0.01)
        
        coeffs = np.polyfit(log_D, log_L, 1)
        beta = -coeffs[0]
        alpha = np.exp(coeffs[1])
        epsilon = np.min(L)
        
        L_pred = scaling_law(D, alpha, beta, epsilon)
        rmse = np.sqrt(np.mean((L - L_pred) ** 2))
        
        return alpha, beta, epsilon, rmse


def predict_loss(D: float, alpha: float, beta: float, epsilon: float) -> float:
    """
    Predict loss at given dataset size using fitted parameters.
    
    Args:
        D: Training set size
        alpha: Fitted alpha
        beta: Fitted beta
        epsilon: Fitted epsilon
        
    Returns:
        Predicted loss
    """
    return scaling_law(np.array([D]), alpha, beta, epsilon)[0]


def compute_optimal_temperature(
    beta: float,
    fraction: float = 2.0 / 3.0
) -> float:
    """
    Compute optimal effective temperature from scaling exponent.
    
    T_eff* = (2/3) · T_c
    
    The fraction 2/3 comes from the thermodynamic optimality principle.
    
    Args:
        beta: Scaling exponent
        fraction: Fraction of critical temperature (default 2/3)
        
    Returns:
        Optimal temperature
    """
    # Critical temperature is inversely related to beta
    T_c = 1.0 / beta
    return fraction * T_c


# Alias for backward compatibility
unified_scaling_law = scaling_law


# ──────────────────────────────────────────────────────────────────────────────
# COOLING THEORY: β(γ) Relationship
# ──────────────────────────────────────────────────────────────────────────────

# EOS critical point (universal, from Edge of Stability literature)
GAMMA_C = 2.0  # Universal EOS critical variance fluctuation

def beta_gamma(gamma: float, gamma_c: float = GAMMA_C, 
               a: float = 0.425, beta_c: float = 0.893) -> float:
    """
    Compute scaling exponent β from variance fluctuation γ.
    
    Derived from RG scaling near the EOS critical point γ_c ≈ 2.0:
        β(γ) = a · ln(γ / γ_c) + β_c
    
    Validated (Phase S1 TPU, CIFAR-10, 200 epochs, R² > 0.995):
        γ_None = 3.39, β_obs = 1.117, β_pred = 1.117
        γ_BN   = 2.29, β_obs = 0.950, β_pred = 0.951
    
    Physical interpretation:
        - Higher γ (heating) → higher β (better width scaling efficiency)
        - Lower γ (cooling) → lower β (worse width scaling, better E_floor)
        - γ_c ≈ 2.0 is the EOS critical point (universal across architectures)
    
    Args:
        gamma: Variance fluctuation γ = (1/L)∑|ln(σ_final/σ_init)|
        gamma_c: Critical EOS scale (~2.0, universal constant)
        a: Sensitivity of β to γ (a ≈ 0.425)
        beta_c: β at γ = γ_c (β_c ≈ 0.893)
        
    Returns:
        Scaling exponent β
        
    Example:
        >>> beta_gamma(3.39)   # None (heating)
        1.117
        >>> beta_gamma(2.29)   # BatchNorm (cooling)
        0.951
    """
    import math
    return a * math.log(gamma / gamma_c) + beta_c


def gamma_ratio_effect(gamma1: float, gamma2: float,
                       gamma_c: float = GAMMA_C, 
                       a: float = 0.425, beta_c: float = 0.893) -> float:
    """
    Compute the ratio β(γ1) / β(γ2).
    
    For BN vs None (heating vs cooling):
        >>> gamma_ratio_effect(2.29, 3.39)  # BN/None
        0.850  # BN reduces β by ~15%
    
    Args:
        gamma1: First variance fluctuation (e.g., γ_BN)
        gamma2: Second variance fluctuation (e.g., γ_None)
        
    Returns:
        Ratio β(γ1) / β(γ2)
    """
    return beta_gamma(gamma1, gamma_c, a, beta_c) / beta_gamma(gamma2, gamma_c, a, beta_c)


def compute_gamma_critical() -> float:
    """
    Return the universal EOS critical point γ_c ≈ 2.0.
    
    This value is measured from Edge of Stability training dynamics
    and is universal across architectures and tasks.
    """
    return GAMMA_C


