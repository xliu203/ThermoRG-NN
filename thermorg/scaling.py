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
    beta_bounds: Tuple[float, float] = (0.01, 1.0),
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
    p0 = [100.0, 0.3, 0.1]  # Initial guess
    
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
