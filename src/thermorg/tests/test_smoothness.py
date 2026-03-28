# SPDX-License-Identifier: Apache-2.0

"""Tests for smoothness estimation."""

import pytest
import numpy as np
from numpy.testing import assert_allclose

import sys
sys.path.insert(0, '/home/node/.openclaw/workspace/agents/coding/work/codebase/ThermoRG-NN/src')

from thermorg.tas.profiling.smoothness import SmoothnessEstimator


class TestSmoothnessEstimator:
    """Test suite for SmoothnessEstimator."""
    
    def test_estimate_s_basic(self):
        """Test basic smoothness estimation."""
        np.random.seed(42)
        n = 300
        d = 10
        
        # Generate smooth manifold data
        t = np.random.randn(n, 3)
        X = np.column_stack([
            np.sin(t[:, 0]) + 0.1 * np.random.randn(n),
            np.cos(t[:, 1]) + 0.1 * np.random.randn(n),
            t[:, 2] + 0.1 * np.random.randn(n),
            np.random.randn(n, d - 3)
        ])
        
        # Generate smooth function on manifold
        y = np.sin(t[:, 0]) * np.cos(t[:, 1]) + 0.1 * np.random.randn(n)
        
        estimator = SmoothnessEstimator(k_neighbors=10, n_eigenvalues=30)
        s = estimator.estimate_s(y, X)
        
        # Should return reasonable smoothness value
        assert 0.1 <= s <= 5.0
        assert isinstance(s, float)
    
    def test_estimate_s_small_sample(self):
        """Test with small sample size."""
        np.random.seed(123)
        X = np.random.randn(10, 5)
        y = np.sin(X[:, 0]) + 0.1 * np.random.randn(10)
        
        estimator = SmoothnessEstimator(k_neighbors=5)
        s = estimator.estimate_s(y, X)
        
        # Should return default or estimated value in range
        assert 0.1 <= s <= 5.0
    
    def test_estimate_s_from_signal(self):
        """Test signal-based smoothness estimation."""
        np.random.seed(456)
        n = 200
        X = np.random.randn(n, 10)
        
        # Smooth sinusoidal signal
        y = np.sin(X[:, 0]) * np.cos(X[:, 1])
        
        estimator = SmoothnessEstimator(k_neighbors=10)
        s = estimator.estimate_s_from_signal(y, X, d_manifold=5.0)
        
        assert 0.1 <= s <= 5.0
    
    def test_s_estimate_property(self):
        """Test the s_estimate property."""
        np.random.seed(789)
        X = np.random.randn(100, 8)
        y = X[:, 0] ** 2 + 0.1 * np.random.randn(100)
        
        estimator = SmoothnessEstimator()
        assert estimator.s_estimate is None
        
        estimator.estimate_s(y, X)
        assert estimator.s_estimate is not None
    
    def test_different_k_neighbors(self):
        """Test with different k values."""
        np.random.seed(111)
        X = np.random.randn(200, 12)
        y = np.sin(X[:, 0]) + np.cos(X[:, 1])
        
        for k in [5, 10, 15]:
            est = SmoothnessEstimator(k_neighbors=k)
            s = est.estimate_s(y, X)
            assert 0.1 <= s <= 5.0
    
    def test_normalized_vs_unnormalized(self):
        """Test normalized and unnormalized Laplacian."""
        np.random.seed(222)
        X = np.random.randn(150, 10)
        y = X[:, 0] * X[:, 1]
        
        est_norm = SmoothnessEstimator(k_neighbors=10)
        est_unorm = SmoothnessEstimator(k_neighbors=10)
        
        s_norm = est_norm.estimate_s(y, X, normalized=True)
        s_unorm = est_unorm.estimate_s(y, X, normalized=False)
        
        # Both should be in valid range
        assert 0.1 <= s_norm <= 5.0
        assert 0.1 <= s_unorm <= 5.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
