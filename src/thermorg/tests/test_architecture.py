# SPDX-License-Identifier: Apache-2.0

"""Tests for architecture analysis."""

import pytest
import numpy as np
from numpy.testing import assert_allclose

import sys
sys.path.insert(0, '/home/node/.openclaw/workspace/agents/coding/work/codebase/ThermoRG-NN/src')

from thermorg.tas.architecture import ArchitectureAnalyzer, JacobianAnalyzer


class TestArchitectureAnalyzer:
    """Test suite for ArchitectureAnalyzer."""
    
    def test_compute_heuristic_eta_basic(self):
        """Test basic heuristic η_l computation."""
        analyzer = ArchitectureAnalyzer()
        
        eta_ls = analyzer.compute_heuristic_eta(
            layer_widths=[64, 128, 256],
            layer_types=['linear', 'linear', 'linear'],
            d_manifold=10.0
        )
        
        assert len(eta_ls) == 3
        assert all(0 < eta < 1 for eta in eta_ls)
    
    def test_compute_heuristic_eta_width_factor(self):
        """Test that width factor scales correctly."""
        analyzer = ArchitectureAnalyzer()
        
        # Narrow layer (should have low w_factor)
        eta_narrow = analyzer.compute_heuristic_eta(
            layer_widths=[5],
            d_manifold=10.0
        )
        
        # Wide layer (should have w_factor = 1)
        eta_wide = analyzer.compute_heuristic_eta(
            layer_widths=[100],
            d_manifold=10.0
        )
        
        # Wide layer should have higher or equal η
        assert eta_wide[0] >= eta_narrow[0]
    
    def test_compute_heuristic_eta_product(self):
        """Test η_l product property."""
        analyzer = ArchitectureAnalyzer()
        
        # Same architecture computed twice should give same product
        eta_ls1 = analyzer.compute_heuristic_eta(
            layer_widths=[64, 128],
            d_manifold=10.0
        )
        
        eta_ls2 = analyzer.compute_heuristic_eta(
            layer_widths=[64, 128],
            d_manifold=10.0
        )
        
        assert_allclose(np.prod(eta_ls1), np.prod(eta_ls2), rtol=1e-5)
    
    def test_eta_ls_property(self):
        """Test eta_ls property."""
        analyzer = ArchitectureAnalyzer()
        
        assert analyzer.eta_ls is None
        
        analyzer.compute_heuristic_eta([64, 128], d_manifold=10.0)
        
        assert analyzer.eta_ls is not None
        assert len(analyzer.eta_ls) == 2
    
    def test_d_effs_property(self):
        """Test d_effs property (initially empty without model)."""
        analyzer = ArchitectureAnalyzer()
        
        assert analyzer.d_effs is None
    
    def test_different_layer_types(self):
        """Test with different layer types."""
        analyzer = ArchitectureAnalyzer()
        
        eta_ls = analyzer.compute_heuristic_eta(
            layer_widths=[64, 64, 64],
            layer_types=['linear', 'conv2d', 'conv1d'],
            d_manifold=10.0
        )
        
        assert len(eta_ls) == 3
        assert all(0 < eta < 1 for eta in eta_ls)


class TestJacobianAnalyzer:
    """Test suite for JacobianAnalyzer."""
    
    def test_init(self):
        """Test initialization."""
        analyzer = JacobianAnalyzer()
        assert analyzer.device == 'cpu'
        
        analyzer_cuda = JacobianAnalyzer(device='cuda')
        assert analyzer_cuda.device == 'cuda'
    
    def test_compute_d_eff(self):
        """Test D_eff computation from Jacobian."""
        analyzer = JacobianAnalyzer()
        
        # Create a simple Jacobian
        J = np.random.randn(10, 5)
        
        d_eff = analyzer.compute_d_eff(J)
        
        # D_eff should be between 1 and n_outputs * n_inputs
        n_out, n_in = J.shape
        assert 1 <= d_eff <= n_out * n_in
        
        # For random matrix, D_eff should be relatively large
        assert d_eff > 1
    
    def test_compute_d_eff_low_rank(self):
        """Test D_eff for low-rank matrix."""
        analyzer = JacobianAnalyzer()
        
        # Create low-rank matrix
        u = np.random.randn(10, 1)
        v = np.random.randn(1, 5)
        J_lowrank = u @ v
        
        d_eff = analyzer.compute_d_eff(J_lowrank)
        
        # Low-rank should have small D_eff
        assert d_eff < 10
    
    def test_compute_d_eff_identity(self):
        """Test D_eff for identity-like matrix."""
        analyzer = JacobianAnalyzer()
        
        J = np.eye(5)
        
        d_eff = analyzer.compute_d_eff(J)
        
        # For identity, ||J||_F^2 = n and ||J||_2 = 1
        # So D_eff = n
        assert_allclose(d_eff, 5.0, rtol=1e-5)
    
    def test_compute_d_eff_zero(self):
        """Test D_eff for zero matrix."""
        analyzer = JacobianAnalyzer()
        
        J = np.zeros((5, 5))
        
        d_eff = analyzer.compute_d_eff(J)
        
        # Should return minimum value of 1.0
        assert d_eff == 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
