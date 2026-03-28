# SPDX-License-Identifier: Apache-2.0

"""Tests for manifold module."""

import torch
import numpy as np
import pytest
from thermorg.core.manifold import (
    levina_bickel_estimator,
    dynamic_manifold演化,
    estimate_from_jacobian,
    batch_manifold_tracking,
)


def test_dynamic_manifold_evolution():
    """Test manifold dimension evolution."""
    prev_dim = 10.0
    comp_eff = 0.8
    new_dim = dynamic_manifold演化(prev_dim, comp_eff)
    assert new_dim == pytest.approx(8.0, rel=1e-5)


def test_estimate_from_jacobian_svd():
    """Test manifold dimension estimation from Jacobian (SVD method)."""
    jacobian = torch.randn(10, 5)
    dim = estimate_from_jacobian(jacobian, method="svd")
    assert dim >= 1.0
    assert dim <= 5.0


def test_estimate_from_jacobian_threshold():
    """Test manifold dimension estimation from Jacobian (threshold method)."""
    jacobian = torch.randn(10, 5)
    dim = estimate_from_jacobian(jacobian, method="threshold")
    assert dim >= 1.0
    assert dim <= 5.0


def test_batch_manifold_tracking():
    """Test manifold dimension tracking across layers."""
    jacobians = [torch.randn(10, 5) for _ in range(3)]
    dims = batch_manifold_tracking(jacobians, initial_dim=1.0)
    
    assert len(dims) == 4  # Initial + 3 layers
    assert dims[0] == 1.0
    # Each subsequent dim should be compressed
    for i in range(1, len(dims)):
        assert dims[i] <= dims[i-1]


def test_levina_bickel():
    """Test Levina-Bickel estimator."""
    np.random.seed(42)
    # Generate data from low-dimensional manifold
    n_samples, n_features = 100, 20
    data = np.random.randn(n_samples, 5) @ np.random.randn(5, n_features)
    data += 0.1 * np.random.randn(n_samples, n_features)
    
    dim = levina_bickel_estimator(data, k=10)
    assert dim >= 1.0
    assert dim <= n_features
