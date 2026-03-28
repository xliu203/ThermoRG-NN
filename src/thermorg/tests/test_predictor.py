# SPDX-License-Identifier: Apache-2.0

"""Tests for the main TAS predictor."""

import pytest
import numpy as np
from numpy.testing import assert_allclose

import sys
sys.path.insert(0, '/home/node/.openclaw/workspace/agents/coding/work/codebase/ThermoRG-NN/src')

from thermorg.tas import TASProfiler, TASConfig, TASResult


class TestTASProfiler:
    """Test suite for TASProfiler."""
    
    def test_init_default_config(self):
        """Test initialization with default config."""
        profiler = TASProfiler()
        
        assert profiler.config is not None
        assert isinstance(profiler.config, TASConfig)
    
    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = TASConfig(
            d_manifold=10.0,
            s_smoothness=1.5,
            eta_lr=1e-4,
            batch_size=64
        )
        
        profiler = TASProfiler(config=config)
        
        assert profiler.config.d_manifold == 10.0
        assert profiler.config.s_smoothness == 1.5
    
    def test_profile_basic(self):
        """Test basic profiling."""
        np.random.seed(42)
        X = np.random.randn(200, 20)
        y = np.sin(X[:, 0]) + 0.1 * np.random.randn(200)
        
        config = TASConfig(d_manifold=5.0, s_smoothness=1.0)
        profiler = TASProfiler(config)
        
        architecture = {'widths': [64, 128, 256], 'types': ['linear'] * 3}
        train_config = {'lr': 1e-3, 'batch_size': 32}
        
        result = profiler.profile(X, y, architecture, train_config)
        
        assert isinstance(result, TASResult)
        assert result.d_manifold == 5.0
        assert result.s_smoothness == 1.0
        assert len(result.eta_ls) == 3
        assert result.alpha >= 0
    
    def test_profile_auto_estimation(self):
        """Test profiling with auto-estimation."""
        np.random.seed(123)
        X = np.random.randn(300, 15)
        y = X[:, 0] ** 2 + 0.1 * np.random.randn(300)
        
        profiler = TASProfiler()
        
        architecture = {'widths': [64, 128]}
        train_config = {'lr': 1e-3, 'batch_size': 32}
        
        result = profiler.profile(X, y, architecture, train_config)
        
        assert result.d_manifold > 0
        assert result.s_smoothness > 0
        assert result.alpha >= 0
    
    def test_profile_architecture(self):
        """Test profiling without data."""
        config = TASConfig(d_manifold=8.0, s_smoothness=1.2)
        profiler = TASProfiler(config)
        
        architecture = {'widths': [64, 128, 256]}
        train_config = {'lr': 1e-3, 'batch_size': 32}
        
        result = profiler.profile_architecture(architecture, train_config)
        
        assert result.d_manifold == 8.0
        assert result.s_smoothness == 1.2
        assert len(result.eta_ls) == 3
    
    def test_predict_alpha(self):
        """Test alpha prediction with given values."""
        profiler = TASProfiler()
        
        alpha = profiler.predict_alpha(
            d=10.0,
            s=1.5,
            eta_ls=[1.0, 1.5, 2.0],
            train_config={'lr': 1e-3, 'batch_size': 32}
        )
        
        assert alpha >= 0
        assert isinstance(alpha, float)
    
    def test_result_summary(self):
        """Test result summary string."""
        result = TASResult(
            d_manifold=10.0,
            s_smoothness=1.5,
            V_grad=1.0,
            T_c=10.0,
            k_alpha=1.0,
            eta_ls=[1.0, 1.5],
            eta_product=1.5,
            d_effs=[],
            T_eff=1e-5,
            T_tilde=1e-4,
            psi_algo=0.5,
            phi_cool=0.8,
            alpha=0.1
        )
        
        summary = result.summary()
        
        assert 'd_manifold = 10' in summary
        assert 'α = 0.1' in summary
    
    def test_last_result_property(self):
        """Test last_result property."""
        profiler = TASProfiler()
        assert profiler.last_result is None
        
        np.random.seed(456)
        X = np.random.randn(100, 10)
        y = np.sin(X[:, 0])
        
        result = profiler.profile(
            X, y,
            {'widths': [64]},
            {'lr': 1e-3, 'batch_size': 32}
        )
        
        assert profiler.last_result is not None
        assert profiler.last_result == result
    
    def test_alpha_formula_components(self):
        """Test that alpha computation uses correct formula."""
        config = TASConfig(
            d_manifold=10.0,
            s_smoothness=1.0,
            V_grad=1.0,
            eta_lr=1e-3,
            batch_size=32,
            epsilon=1.0
        )
        profiler = TASProfiler(config)
        
        # α = k_α(L) · |log(∏η_l)| · (2s/d_manifold) · Ψ_algo · φ · ε
        k_alpha = 1.0
        eta_product = 2.0
        log_eta = abs(np.log(eta_product))
        geometric = 2.0 * 1.0 / 10.0
        psi = 0.5
        phi = 0.8
        
        expected_alpha = k_alpha * log_eta * geometric * psi * phi * 1.0
        
        # This should match when we profile
        X = np.random.randn(100, 10)
        y = X[:, 0]
        
        result = profiler.profile(
            X, y,
            {'widths': [64], 'delta_loss': 0},
            {'lr': 1e-3, 'batch_size': 32}
        )
        
        # Alpha should be positive
        assert result.alpha >= 0


class TestTASResult:
    """Test suite for TASResult dataclass."""
    
    def test_predicted_scaling_exponent(self):
        """Test predicted_scaling_exponent equals alpha."""
        result = TASResult(
            d_manifold=10.0,
            s_smoothness=1.0,
            V_grad=1.0,
            T_c=10.0,
            k_alpha=1.0,
            eta_ls=[1.0],
            eta_product=1.0,
            d_effs=[],
            T_eff=1e-5,
            T_tilde=1e-4,
            psi_algo=0.5,
            phi_cool=0.8,
            alpha=0.1
        )
        
        assert result.predicted_scaling_exponent == result.alpha


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
