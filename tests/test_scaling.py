# SPDX-License-Identifier: Apache-2.0
"""Unit tests for scaling.py and cooling.py."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from thermorg.scaling import (
    scaling_law,
    fit_scaling_law,
    predict_loss,
    beta_gamma,
    compute_gamma_critical,
)
from thermorg.cooling import cooling_factor_linear


def test_scaling_law_shape():
    """scaling_law output shape should match input shape."""
    D = np.array([100, 500, 1000, 5000])
    alpha, beta, epsilon = 10.0, 0.5, 0.3
    L = scaling_law(D, alpha, beta, epsilon)
    assert L.shape == D.shape, f"Output shape {L.shape} != input shape {D.shape}"
    # Larger D → smaller L
    assert L[0] > L[-1], f"L(D) should decrease with D: {L}"
    print(f"  D={D}, L={L} ✓")


def test_scaling_law_monotonic():
    """L(D) should be strictly decreasing in D."""
    D = np.linspace(100, 10000, 50)
    L = scaling_law(D, alpha=20.0, beta=0.5, epsilon=0.5)
    diffs = np.diff(L)
    assert all(d < 0 for d in diffs), "L(D) should be monotonically decreasing"
    print(f"  Monotonically decreasing ✓")


def test_fit_scaling_law_synthetic():
    """fit_scaling_law should recover parameters from synthetic data."""
    # True parameters
    alpha_true, beta_true, epsilon_true = 15.0, 0.4, 0.6
    D = np.array([200, 500, 1000, 2000, 5000])
    L_true = scaling_law(D, alpha_true, beta_true, epsilon_true)
    # Add tiny noise
    L_noisy = L_true + np.random.randn(len(D)) * 0.01
    
    alpha_fit, beta_fit, epsilon_fit, rmse = fit_scaling_law(D, L_noisy)
    
    assert abs(alpha_fit - alpha_true) / alpha_true < 0.3, \
        f"alpha: {alpha_fit:.3f} vs true {alpha_true:.3f}"
    assert abs(beta_fit - beta_true) / beta_true < 0.3, \
        f"beta: {beta_fit:.3f} vs true {beta_true:.3f}"
    assert abs(epsilon_fit - epsilon_true) / epsilon_true < 0.3, \
        f"epsilon: {epsilon_fit:.3f} vs true {epsilon_true:.3f}"
    print(f"  True: α={alpha_true}, β={beta_true}, ε={epsilon_true}")
    print(f"  Fitted: α={alpha_fit:.3f}, β={beta_fit:.3f}, ε={epsilon_fit:.3f} ✓")


def test_predict_loss():
    """predict_loss should return positive values."""
    loss = predict_loss(D=1000.0, alpha=10.0, beta=0.5, epsilon=0.5)
    assert loss > 0, f"Loss should be positive, got {loss}"
    assert loss < 10.0 + 0.5, f"Loss={loss} exceeds max expected {10.0+0.5}"
    print(f"  predict_loss(D=1000)={loss:.4f} ✓")


def test_beta_gamma_known_values():
    """beta_gamma should match known validated values."""
    # From Phase S1 (validated on real training data):
    # BN: γ=2.29 → β=0.950
    # None: γ=3.39 → β=1.117
    gamma_c = 2.0  # critical point
    beta_bn = beta_gamma(2.29, gamma_c=gamma_c)
    beta_none = beta_gamma(3.39, gamma_c=gamma_c)
    
    assert abs(beta_bn - 0.950) < 0.05, f"BN beta: {beta_bn:.3f} vs expected 0.950"
    assert abs(beta_none - 1.117) < 0.05, f"None beta: {beta_none:.3f} vs expected 1.117"
    print(f"  β(γ=2.29)={beta_bn:.3f} (expected 0.950) ✓")
    print(f"  β(γ=3.39)={beta_none:.3f} (expected 1.117) ✓")


def test_beta_gamma_subcritical():
    """In sub-critical regime (γ < γ_c), beta should be small."""
    gamma_c = 2.0
    # LN: γ≈0.41 → β≈0.219
    beta_ln = beta_gamma(0.41, gamma_c=gamma_c)
    assert 0.1 < beta_ln < 0.3, f"LN beta: {beta_ln:.3f} should be ~0.219"
    print(f"  β(γ=0.41)={beta_ln:.3f} (sub-critical) ✓")


def test_beta_gamma_monotonic():
    """beta(gamma) should increase with gamma."""
    gammas = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    betas = [beta_gamma(g, gamma_c=2.0) for g in gammas]
    diffs = np.diff(betas)
    assert all(d >= 0 for d in diffs), f"beta(gamma) should be monotonic: {betas}"
    print(f"  β(γ): monotonic increasing ✓")


def test_cooling_factor_linear():
    """Linear cooling: φ(0)=1, φ(1)=0."""
    assert abs(cooling_factor_linear(0.0) - 1.0) < 1e-6, "φ(0) should be 1"
    assert abs(cooling_factor_linear(1.0) - 0.0) < 1e-6, "φ(1) should be 0"
    assert cooling_factor_linear(0.5) == 0.5, "φ(0.5) should be 0.5"
    print(f"  φ(0)={cooling_factor_linear(0.0):.1f}, φ(0.5)={cooling_factor_linear(0.5):.1f}, φ(1)={cooling_factor_linear(1.0):.1f} ✓")


def test_gamma_critical():
    """Critical gamma should be positive."""
    gamma_c = compute_gamma_critical()
    assert gamma_c > 0, f"γ_c should be positive, got {gamma_c}"
    print(f"  γ_c={gamma_c:.4f} ✓")


if __name__ == '__main__':
    print("=== test_scaling + cooling ===")
    test_scaling_law_shape()
    test_scaling_law_monotonic()
    test_fit_scaling_law_synthetic()
    test_predict_loss()
    test_beta_gamma_known_values()
    test_beta_gamma_subcritical()
    test_beta_gamma_monotonic()
    test_cooling_factor_linear()
    test_gamma_critical()
    print("\nAll tests passed ✓")
