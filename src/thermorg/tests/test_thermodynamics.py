# SPDX-License-Identifier: Apache-2.0

"""Tests for thermodynamics components."""

import pytest
import numpy as np
from numpy.testing import assert_allclose

import sys
sys.path.insert(0, '/home/node/.openclaw/workspace/agents/coding/work/codebase/ThermoRG-NN/src')

from thermorg.tas.thermodynamics import (
    TemperatureEstimator,
    ThermalPhaseComputer,
    CoolingPhaseComputer,
)


class TestTemperatureEstimator:
    """Test suite for TemperatureEstimator."""
    
    def test_estimate_T_eff(self):
        """Test T_eff estimation."""
        est = TemperatureEstimator()
        
        T_eff = est.estimate_T_eff(
            eta_lr=1e-3,
            noise_variance=0.1,
            batch_size=32
        )
        
        expected = (1e-3 * 0.1) / 32
        assert_allclose(T_eff, expected, rtol=1e-5)
    
    def test_estimate_T_c(self):
        """Test T_c estimation."""
        est = TemperatureEstimator()
        
        T_c = est.estimate_T_c(d_manifold=10.0, V_grad=1.0)
        
        assert_allclose(T_c, 10.0, rtol=1e-5)
    
    def test_estimate_from_gradient_stats(self):
        """Test estimation from gradient statistics."""
        np.random.seed(42)
        gradients = np.random.randn(100, 50) * 0.1
        
        est = TemperatureEstimator()
        result = est.estimate_from_gradient_stats(
            gradients, d_manifold=5.0, eta_lr=1e-3
        )
        
        assert 'T_eff' in result
        assert 'T_c' in result
        assert 'V_grad' in result
        # T_c = d / V_grad (should be approximately 5 given gradient std of 0.1)
        assert result['T_c'] > 0
    
    def test_compute_scaling_temperature(self):
        """Test scaled temperature computation."""
        est = TemperatureEstimator()
        
        eta_ls = [1.0, 1.5, 2.0]
        eta_product = np.prod(eta_ls)
        d_manifold = 4.0
        
        T_eff = 0.01
        T_scaled = est.compute_scaling_temperature(eta_ls, T_eff, d_manifold)
        
        scaling_factor = np.power(eta_product, 2.0 / d_manifold)
        expected = T_eff / scaling_factor
        
        assert_allclose(T_scaled, expected, rtol=1e-5)
    
    def test_zero_batch_size(self):
        """Test handling of zero batch size."""
        est = TemperatureEstimator()
        
        T_eff = est.estimate_T_eff(eta_lr=1e-3, noise_variance=0.1, batch_size=0)
        
        # Should handle gracefully (defaults to batch_size=1)
        assert T_eff >= 0


class TestThermalPhaseComputer:
    """Test suite for ThermalPhaseComputer."""
    
    def test_psi_basic(self):
        """Test basic Ψ_algo computation."""
        computer = ThermalPhaseComputer(gamma_T=1.0)
        
        psi = computer.compute_psi(
            T_tilde=0.5,  # Below T_c
            T_c=1.0,
            delta_loss=0.0,
            T_eff=0.1
        )
        
        # When T_tilde < T_c and delta_loss = 0:
        # psi = T_tilde * (1 - T_tilde/T_c)^gamma_T * 1 * 1
        expected = 0.5 * (1 - 0.5) ** 1.0
        assert_allclose(psi, expected, rtol=1e-5)
    
    def test_psi_above_critical(self):
        """Test when T_tilde >= T_c (should be 0)."""
        computer = ThermalPhaseComputer()
        
        psi = computer.compute_psi(
            T_tilde=1.5,  # Above T_c
            T_c=1.0,
            delta_loss=0.0
        )
        
        assert psi == 0.0
    
    def test_psi_with_loss_barrier(self):
        """Test Ψ_algo with loss barrier."""
        computer = ThermalPhaseComputer()
        
        psi = computer.compute_psi(
            T_tilde=0.5,
            T_c=1.0,
            delta_loss=0.1,
            T_eff=0.1
        )
        
        # psi = T_tilde * (1 - 0.5)^1 * exp(-0.1/0.1) * 1
        #     = 0.5 * 0.5 * exp(-1)
        expected = 0.5 * 0.5 * np.exp(-1)
        assert_allclose(psi, expected, rtol=1e-5)
    
    def test_psi_gamma_exponent(self):
        """Test different gamma_T values."""
        for gamma in [0.5, 1.0, 2.0]:
            computer = ThermalPhaseComputer(gamma_T=gamma)
            
            psi = computer.compute_psi(
                T_tilde=0.5,
                T_c=1.0,
                delta_loss=0.0
            )
            
            expected = 0.5 * (0.5) ** gamma
            assert_allclose(psi, expected, rtol=1e-5)
    
    def test_psi_vectorized(self):
        """Test vectorized computation."""
        computer = ThermalPhaseComputer()
        
        T_tildes = np.array([0.3, 0.5, 0.7])
        T_c = 1.0
        
        psis = computer.compute_psi_vectorized(T_tildes, T_c)
        
        assert len(psis) == 3
        assert all(p >= 0 for p in psis)


class TestCoolingPhaseComputer:
    """Test suite for CoolingPhaseComputer."""
    
    def test_phi_zero_gamma(self):
        """Test φ(0) = 1."""
        computer = CoolingPhaseComputer()
        
        phi = computer.compute_phi(gamma=0.0, gamma_c=1.0)
        
        assert_allclose(phi, 1.0, rtol=1e-5)
    
    def test_phi_at_critical(self):
        """Test φ(γ_c) = exp(-1) / 2."""
        computer = CoolingPhaseComputer()
        
        phi = computer.compute_phi(gamma=1.0, gamma_c=1.0)
        
        expected = np.exp(-1) / 2
        assert_allclose(phi, expected, rtol=1e-5)
    
    def test_phi_large_gamma(self):
        """Test φ(∞) → 0."""
        computer = CoolingPhaseComputer()
        
        phi = computer.compute_phi(gamma=100.0, gamma_c=1.0)
        
        assert phi < 0.01
    
    def test_phi_in_range(self):
        """Test that φ is always in [0, 1]."""
        computer = CoolingPhaseComputer()
        
        for gamma in np.linspace(0, 10, 100):
            phi = computer.compute_phi(gamma, gamma_c=1.0)
            assert 0 <= phi <= 1
    
    def test_estimate_gamma_c(self):
        """Test γ_c estimation."""
        computer = CoolingPhaseComputer()
        
        gamma_c = computer.estimate_gamma_c(
            eta_lr=1e-3,
            V_grad=1.0,
            d_manifold=10.0
        )
        
        expected = (1e-3 * 1.0) / 10.0
        assert_allclose(gamma_c, expected, rtol=1e-5)
    
    def test_derivative(self):
        """Test derivative computation."""
        computer = CoolingPhaseComputer()
        
        # Numerical derivative check
        gamma = 0.5
        gamma_c = 1.0
        h = 1e-5
        
        phi = computer.compute_phi(gamma, gamma_c)
        phi_plus = computer.compute_phi(gamma + h, gamma_c)
        numerical_deriv = (phi_plus - phi) / h
        
        analytical_deriv = computer.compute_phi_derivative(gamma, gamma_c)
        
        assert_allclose(analytical_deriv, numerical_deriv, rtol=1e-3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
