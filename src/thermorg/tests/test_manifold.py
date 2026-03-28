# SPDX-License-Identifier: Apache-2.0

"""Tests for manifold dimension estimation."""

import pytest
import numpy as np
from numpy.testing import assert_allclose

import sys
sys.path.insert(0, '/home/node/.openclaw/workspace/agents/coding/work/codebase/ThermoRG-NN/src')

from thermorg.tas.profiling.manifold import ManifoldEstimator


class TestManifoldEstimator:
    """Test suite for ManifoldEstimator."""
    
    def test_estimate_d_basic(self):
        """Test basic dimension estimation."""
        # Generate random data in known dimension
        np.random.seed(42)
        n_samples = 500
        d_true = 5
        X = np.random.randn(n_samples, d_true) @ np.random.randn(d_true, 20)
        
        estimator = ManifoldEstimator(k_max=10)
        d_est = estimator.estimate_d(X)
        
        # Should be reasonably close to true dimension
        assert 1 <= d_est <= 20
        assert isinstance(d_est, float)
    
    def test_estimate_d_high_dim(self):
        """Test on high-dimensional data."""
        np.random.seed(123)
        n = 300
        d = 50
        X = np.random.randn(n, d)
        
        estimator = ManifoldEstimator(k_max=15)
        d_est = estimator.estimate_d(X)
        
        assert d_est >= 1.0
        assert d_est <= d
    
    def test_estimate_d_with_confidence(self):
        """Test confidence interval estimation."""
        np.random.seed(456)
        X = np.random.randn(200, 10)
        
        estimator = ManifoldEstimator(k_max=10)
        d_est, std_err = estimator.estimate_d_with_confidence(X)
        
        assert d_est >= 1.0
        assert std_err >= 0.0
        assert std_err < d_est  # Standard error should be smaller than estimate
    
    def test_small_sample(self):
        """Test behavior with small sample size."""
        np.random.seed(789)
        X = np.random.randn(5, 10)  # Very small
        
        estimator = ManifoldEstimator(k_max=3)
        d_est = estimator.estimate_d(X)
        
        # Should not crash and return valid value
        assert d_est >= 1.0
    
    def test_d_estimate_property(self):
        """Test the d_estimate property."""
        np.random.seed(111)
        X = np.random.randn(100, 15)
        
        estimator = ManifoldEstimator()
        assert estimator.d_estimate is None
        
        estimator.estimate_d(X)
        assert estimator.d_estimate is not None
        assert isinstance(estimator.d_estimate, float)
    
    def test_deterministic_with_seed(self):
        """Test that same seed gives same result."""
        np.random.seed(42)
        X1 = np.random.randn(100, 10)
        
        np.random.seed(42)
        X2 = np.random.randn(100, 10)
        
        est1 = ManifoldEstimator()
        est2 = ManifoldEstimator()
        
        d1 = est1.estimate_d(X1)
        d2 = est2.estimate_d(X2)
        
        assert_allclose(d1, d2, rtol=1e-5)
    
    def test_k_parameter(self):
        """Test different k values."""
        np.random.seed(222)
        X = np.random.randn(200, 12)
        
        for k in [5, 10, 15]:
            est = ManifoldEstimator(k_max=k)
            d = est.estimate_d(X)
            assert d >= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
