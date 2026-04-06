"""
Utility functions for SU-HBO based on ThermoRG theory.

Core equations:
- E_floor(J, γ) = k·γ / (1 - B·J·γ)
- φ(γ) = γ_c / (γ_c + γ) · exp(-γ/γ_c)
- β(γ) = β_0 · φ(γ)
- U(A) = -E_floor(J, γ) + λ · β(γ)
"""

import numpy as np
from typing import Optional


# Default parameters (from Phase A / S1)
DEFAULT_K = 0.06
DEFAULT_B = 0.15
DEFAULT_GAMMA_C = 2.0
DEFAULT_LAMBDA = 10.0

# Base beta values
BETA_BN = 0.368
BETA_NONE = 0.180
BETA_LN = 0.370

# Base gamma values
GAMMA_BN = 2.36
GAMMA_NONE = 3.36
GAMMA_LN = 2.80


def compute_e_floor(j_topo: float, gamma: float,
                    k: float = DEFAULT_K,
                    B: float = DEFAULT_B) -> float:
    """
    Compute E_floor from J_topo and gamma.

    E_floor(J, γ) = k·γ / (1 - B·J·γ)

    Args:
        j_topo: Topological quality (0-1)
        gamma: Variance instability parameter
        k: Coupling constant (default 0.06)
        B: Topology-temperature coupling (default 0.15)

    Returns:
        Asymptotic error floor
    """
    denominator = 1.0 - B * j_topo * gamma

    # Critical condition check
    if denominator <= 0:
        # Return large value indicating divergence
        return 10.0

    return k * gamma / denominator


def compute_cooling_factor(gamma: float,
                           gamma_c: float = DEFAULT_GAMMA_C) -> float:
    """
    Compute cooling factor φ(γ).

    φ(γ) = γ_c / (γ_c + γ) · exp(-γ/γ_c)

    Args:
        gamma: Variance instability parameter
        gamma_c: Critical gamma (default 2.0)

    Returns:
        Cooling factor (0-1)
    """
    phi = (gamma_c / (gamma_c + gamma)) * np.exp(-gamma / gamma_c)
    return float(phi)


def compute_beta(gamma: float,
                 norm_type: str = 'none',
                 gamma_c: float = DEFAULT_GAMMA_C) -> float:
    """
    Compute effective beta (learning speed).

    β(γ) = β_0 · φ(γ)

    Args:
        gamma: Variance instability parameter
        norm_type: 'none', 'bn', or 'ln'
        gamma_c: Critical gamma

    Returns:
        Effective beta
    """
    # Base beta
    if norm_type == 'bn':
        beta_0 = BETA_BN
    elif norm_type == 'ln':
        beta_0 = BETA_LN
    else:
        beta_0 = BETA_NONE

    # Apply cooling factor
    phi = compute_cooling_factor(gamma, gamma_c)

    return beta_0 * phi


def compute_utility(j_topo: float,
                     gamma: float,
                     norm_type: str = 'none',
                     lambda_param: float = DEFAULT_LAMBDA,
                     k: float = DEFAULT_K,
                     B: float = DEFAULT_B,
                     gamma_c: float = DEFAULT_GAMMA_C) -> float:
    """
    Compute utility U(A) = -E_floor(J, γ) + λ · β(γ)

    Args:
        j_topo: Topological quality
        gamma: Variance instability
        norm_type: 'none', 'bn', or 'ln'
        lambda_param: Trade-off weight (default 10.0)
        k: E_floor parameter
        B: E_floor parameter
        gamma_c: Critical gamma

    Returns:
        Utility value (higher is better)
    """
    e_floor = compute_e_floor(j_topo, gamma, k, B)
    beta = compute_beta(gamma, norm_type, gamma_c)

    utility = -e_floor + lambda_param * beta
    return float(utility)


def compute_delta_utility(j_before: float,
                          gamma_before: float,
                          norm_before: str,
                          j_after: float,
                          gamma_after: float,
                          norm_after: str,
                          lambda_param: float = DEFAULT_LAMBDA,
                          k: float = DEFAULT_K,
                          B: float = DEFAULT_B,
                          gamma_c: float = DEFAULT_GAMMA_C) -> float:
    """
    Compute change in utility from an action.

    ΔU = U(after) - U(before)

    Args:
        j_before, gamma_before, norm_before: State before action
        j_after, gamma_after, norm_after: State after action
        lambda_param: Trade-off weight

    Returns:
        Change in utility
    """
    u_before = compute_utility(j_before, gamma_before, norm_before,
                                lambda_param, k, B, gamma_c)
    u_after = compute_utility(j_after, gamma_after, norm_after,
                              lambda_param, k, B, gamma_c)
    return u_after - u_before


def is_stable(j_topo: float, gamma: float, B: float = DEFAULT_B) -> bool:
    """
    Check if architecture is in stable regime.

    Stability condition: γ < 1 / (B · J)

    Args:
        j_topo: Topological quality
        gamma: Variance instability
        B: Coupling parameter

    Returns:
        True if stable, False if diverging
    """
    critical_gamma = 1.0 / (B * j_topo) if j_topo > 0 else float('inf')
    return gamma < critical_gamma


def get_stability_margin(j_topo: float, gamma: float,
                         B: float = DEFAULT_B) -> float:
    """
    Get margin to instability.

    Returns:
        Fraction of critical gamma used (lower is more stable)
    """
    critical_gamma = 1.0 / (B * j_topo) if j_topo > 0 else float('inf')
    return gamma / critical_gamma if critical_gamma > 0 else 0.0
