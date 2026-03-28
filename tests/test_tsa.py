# SPDX-License-Identifier: Apache-2.0

"""Tests for TSA modules."""

import torch
import pytest
from thermorg.tsa.rmt import (
    marchenko_pastur_cdf,
    spacing_distribution,
    wigner_surmise,
    compute_spectral_entropy,
    rmt_analytical_path,
    level_statistics,
)
from thermorg.tsa.gradient import (
    OptimizationConfig,
    compute_gradient_path,
    temperature_gradient_flow,
    optimize_spectral_path,
)
from thermorg.tsa.greedy import (
    RoutingConfig,
    compute_local_reward,
    greedy_route_step,
    greedy_routing,
    routing_summary,
)


def test_marchenko_pastur_cdf():
    """Test Marchenko-Pastur CDF."""
    # At edge of support, CDF should be 0 or 1
    val = marchenko_pastur_cdf(0.01, sigma=1.0)
    assert 0.0 <= val <= 1.0


def test_spacing_distribution():
    """Test spacing distribution computation."""
    eigenvalues = torch.sort(torch.randn(100)).values
    spacings = spacing_distribution(eigenvalues)
    assert len(spacings) > 0
    assert torch.all(spacings >= 0)


def test_wigner_surmise():
    """Test Wigner surmise."""
    s = torch.linspace(0.1, 3.0, 10)
    p = wigner_surmise(s, beta=1)
    assert len(p) == len(s)
    assert torch.all(p >= 0)


def test_spectral_entropy():
    """Test spectral entropy computation."""
    eigenvalues = torch.tensor([1.0, 2.0, 3.0, 4.0])
    entropy = compute_spectral_entropy(eigenvalues)
    assert entropy >= 0


def test_rmt_analytical_path():
    """Test RMT analytical path."""
    matrix = torch.randn(10, 10)
    matrix = matrix @ matrix.T  # Make symmetric
    
    result = rmt_analytical_path(matrix, temperature=1.0)
    
    assert "free_energy" in result
    assert "entropy" in result
    assert "gap_ratio" in result
    assert "temperature_eff" in result


def test_level_statistics():
    """Test level statistics classification."""
    # GOE-like spectrum
    eigenvalues = torch.cumsum(torch.randn(100), dim=0)
    stats = level_statistics(eigenvalues)
    assert stats in ["poisson", "goe", "gue", "gse"]


def test_compute_gradient_path():
    """Test gradient-based optimization path."""
    matrix = torch.randn(10, 10)
    matrix = matrix @ matrix.T
    
    config = OptimizationConfig(max_iterations=10)
    result = compute_gradient_path(matrix, lambda x: x, config)
    
    assert "optimal_temp" in result
    assert "loss_history" in result


def test_temperature_gradient_flow():
    """Test temperature gradient flow."""
    temps = temperature_gradient_flow(1.0, 0.5, n_steps=10)
    assert len(temps) == 10
    assert temps[0] == pytest.approx(1.0, abs=0.1)
    assert temps[-1] == pytest.approx(0.5, abs=0.1)


def test_greedy_routing():
    """Test greedy routing."""
    jacobians = [torch.randn(10, 5) for _ in range(3)]
    route = greedy_routing(jacobians)
    
    assert len(route) == 3
    summary = routing_summary(route)
    assert summary["n_steps"] == 3


def test_routing_summary():
    """Test routing summary statistics."""
    from thermorg.tsa.greedy import RouteStep
    
    route = [
        RouteStep(0, 1.0, 0.8, 0.8),
        RouteStep(1, 0.9, 0.7, 1.5),
        RouteStep(2, 0.85, 0.6, 2.1),
    ]
    
    summary = routing_summary(route)
    assert summary["n_steps"] == 3
    assert summary["mean_temp"] == pytest.approx(0.917, abs=0.01)
