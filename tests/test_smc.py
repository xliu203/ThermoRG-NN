# SPDX-License-Identifier: Apache-2.0

"""Tests for SMC module."""

import torch
import pytest
from thermorg.core.smc import (
    compute_frobenius_norm,
    compute_spectral_norm,
    effective_dimension,
    compression_efficiency,
    spectral_momentum_operator,
    compute_smc_metrics,
)


def test_frobenius_norm():
    """Test Frobenius norm computation."""
    jacobian = torch.randn(10, 5)
    norm = compute_frobenius_norm(jacobian)
    expected = torch.linalg.norm(jacobian, ord="fro")
    assert torch.isclose(norm, expected)


def test_spectral_norm():
    """Test spectral norm computation."""
    jacobian = torch.randn(10, 5)
    norm = compute_spectral_norm(jacobian, n_power_iter=20)
    assert norm >= 0
    # For rank-deficient matrix, spectral norm should be bounded
    assert norm <= compute_frobenius_norm(jacobian)


def test_effective_dimension():
    """Test effective dimension calculation."""
    fro_norm = torch.tensor(10.0)
    spec_norm = torch.tensor(5.0)
    eff_dim = effective_dimension(fro_norm, spec_norm)
    assert eff_dim == pytest.approx(4.0, rel=1e-5)


def test_compression_efficiency():
    """Test compression efficiency."""
    eff_dim = torch.tensor(4.0)
    manifold_dim = 10.0
    eta = compression_efficiency(eff_dim, manifold_dim)
    assert eta == pytest.approx(0.4, rel=1e-5)


def test_spectral_momentum_operator():
    """Test spectral momentum operator."""
    jacobian = torch.randn(10, 5)
    pi = spectral_momentum_operator(jacobian)
    assert pi.shape == (5, 5)
    # Should be symmetric
    assert torch.allclose(pi, pi.T)


def test_compute_smc_metrics():
    """Test full SMC metrics computation."""
    jacobian = torch.randn(10, 5)
    manifold_dim = 3.0
    metrics = compute_smc_metrics(jacobian, manifold_dim)
    
    assert "frobenius_norm" in metrics
    assert "spectral_norm" in metrics
    assert "effective_dim" in metrics
    assert "compression_eff" in metrics
    assert "momentum_op" in metrics
    
    assert metrics["compression_eff"] >= 0
    assert metrics["compression_eff"] <= metrics["effective_dim"]
