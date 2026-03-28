# SPDX-License-Identifier: Apache-2.0

"""Tests for utility functions."""

import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy.sparse import csr_matrix

import sys
sys.path.insert(0, '/home/node/.openclaw/workspace/agents/coding/work/codebase/ThermoRG-NN/src')

from thermorg.utils import (
    compute_pairwise_distances,
    get_knn_graph,
    compute_graph_laplacian,
    estimate_eigendecay,
    safe_divide,
    safe_log,
    product_log,
)


class TestMathUtils:
    """Test suite for math utilities."""
    
    def test_safe_divide(self):
        """Test safe division."""
        assert_allclose(safe_divide(1.0, 2.0), 0.5, rtol=1e-9)
        # Default eps is 1e-10, so 1/1e-10 = 1e10
        assert_allclose(safe_divide(1.0, 0.0), 1e10, rtol=1e-9)
        assert_allclose(safe_divide(1.0, 0.0, eps=1e-5), 1e5, rtol=1e-9)
    
    def test_safe_log(self):
        """Test safe logarithm."""
        assert_allclose(safe_log(1.0), 0.0, atol=1e-10)
        assert_allclose(safe_log(np.e), 1.0, rtol=1e-5)
        assert safe_log(0.0) == safe_log(1e-10)  # Clipped to eps
    
    def test_product_log_empty(self):
        """Test product_log with empty list."""
        assert product_log([]) == 1.0
    
    def test_product_log_single(self):
        """Test product_log with single element."""
        assert product_log([2.5]) == 2.5
    
    def test_product_log_multiple(self):
        """Test product_log with multiple elements."""
        assert product_log([2.0, 3.0, 4.0]) == 24.0
    
    def test_product_log_with_zeros(self):
        """Test product_log handles zeros."""
        result = product_log([2.0, 0.0, 3.0])
        assert result == 0.0


class TestGraphUtils:
    """Test suite for graph construction utilities."""
    
    def test_compute_pairwise_distances(self):
        """Test pairwise distance computation."""
        X = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0]
        ])
        
        D = compute_pairwise_distances(X)
        
        assert D.shape == (3, 3)
        assert_allclose(D, D.T)  # Symmetric
        assert_allclose(np.diag(D), 0.0)  # Zero diagonal
        
        # Distance from (0,0) to (1,0) should be 1
        assert_allclose(D[0, 1], 1.0, atol=1e-10)
    
    def test_get_knn_graph(self):
        """Test k-NN graph construction."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        
        W = get_knn_graph(X, k=5)
        
        assert isinstance(W, csr_matrix)
        assert W.shape == (50, 50)
        assert W.nnz > 0  # Has non-zero entries
        
        # Should be symmetric for undirected graph
        assert_allclose(W.toarray(), W.T.toarray())
    
    def test_get_knn_graph_no_self_loops(self):
        """Test that k-NN graph has no self-loops."""
        X = np.random.randn(30, 5)
        
        W = get_knn_graph(X, k=5, symmetric=False)
        
        # Check no self-loops (diagonal should be 0)
        diag = np.array(W.diagonal()).flatten()
        assert np.all(diag == 0)
    
    def test_get_knn_graph_symmetric(self):
        """Test symmetric k-NN graph."""
        np.random.seed(123)
        X = np.random.randn(40, 5)
        
        W = get_knn_graph(X, k=5, symmetric=True)
        
        # Check symmetry
        assert_allclose(W.toarray(), W.T.toarray())
    
    def test_compute_graph_laplacian_normalized(self):
        """Test normalized Laplacian computation."""
        np.random.seed(456)
        X = np.random.randn(30, 5)
        
        W = get_knn_graph(X, k=5)
        L = compute_graph_laplacian(W, normalized=True)
        
        assert isinstance(L, csr_matrix)
        assert L.shape == W.shape
        
        # Check L is symmetric
        assert_allclose(L.toarray(), L.T.toarray())
    
    def test_compute_graph_laplacian_unnormalized(self):
        """Test unnormalized Laplacian computation."""
        np.random.seed(789)
        X = np.random.randn(30, 5)
        
        W = get_knn_graph(X, k=5)
        L = compute_graph_laplacian(W, normalized=False)
        
        # For unnormalized: L = D - W
        # Row sums should be 0 (or close for numerical)
        row_sums = np.array(L.sum(axis=1)).flatten()
        assert_allclose(row_sums, 0.0, atol=1e-10)


class TestEigenDecay:
    """Test suite for eigenvalue decay estimation."""
    
    def test_estimate_eigendecay_fast(self):
        """Test decay estimation for fast decay."""
        # λ_i = exp(-ρ * i) with ρ = 2
        n = 30
        rho = 2.0
        eigenvalues = np.exp(-rho * np.arange(1, n + 1))
        
        rho_est = estimate_eigendecay(eigenvalues)
        
        assert 0 < rho_est < 10  # Should be in reasonable range
        assert rho_est > 0.1  # Should detect some decay
    
    def test_estimate_eigendecay_slow(self):
        """Test decay estimation for slow decay."""
        # λ_i = exp(-ρ * i) with ρ = 0.1
        n = 30
        rho = 0.1
        eigenvalues = np.exp(-rho * np.arange(1, n + 1))
        
        rho_est = estimate_eigendecay(eigenvalues)
        
        assert rho_est > 0
        assert rho_est < 1  # Should detect slow decay
    
    def test_estimate_eigendecay_constant(self):
        """Test decay estimation for constant eigenvalues."""
        eigenvalues = np.ones(20)
        
        rho_est = estimate_eigendecay(eigenvalues)
        
        # Should return small positive value (essentially no decay)
        assert rho_est > 0
    
    def test_estimate_eigendecay_order_invariance(self):
        """Test that order doesn't matter."""
        eigenvalues = np.array([10, 5, 2, 1, 0.5, 0.1])
        
        rho1 = estimate_eigendecay(eigenvalues)
        rho2 = estimate_eigendecay(np.sort(eigenvalues)[::-1])
        
        assert_allclose(rho1, rho2, rtol=0.1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
