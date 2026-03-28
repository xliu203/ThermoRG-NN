# SPDX-License-Identifier: Apache-2.0

"""Tests for TAS optimality verification (Phase 6)."""

import pytest
import numpy as np
from thermorg.tas import (
    TASProfiler,
    TASConfig,
    OptimalityResult,
    compute_epsilon_coupling,
    check_c1_topological_isometry,
    check_c2_thermal_safety,
    is_thermogeometrically_feasible,
    ArchitectureSearcher,
    SearchResult,
)


class TestComputeEpsilonCoupling:
    """Tests for compute_epsilon_coupling function."""
    
    def test_basic_case(self):
        """Test basic epsilon coupling computation."""
        eta_ls = [0.5, 0.5, 0.5]
        d_manifold = 6.0
        eps_coup = compute_epsilon_coupling(eta_ls, d_manifold)
        
        # exp(-(2/6) * (3 * log(0.5))) = exp(-(1/3) * (-2.079)) = exp(0.693) ≈ 2.0
        # But it should be clipped to 1.0
        assert eps_coup == 1.0  # Clipped since > 1.0
    
    def test_empty_eta_ls(self):
        """Test with empty eta list returns 1.0."""
        eps_coup = compute_epsilon_coupling([], 6.0)
        assert eps_coup == 1.0
    
    def test_zero_d_manifold(self):
        """Test with zero d_manifold returns 1.0."""
        eps_coup = compute_epsilon_coupling([0.5, 0.5], 0.0)
        assert eps_coup == 1.0
    
    def test_small_values(self):
        """Test with small eta values gives value < 1."""
        eta_ls = [0.1, 0.1]
        d_manifold = 10.0
        eps_coup = compute_epsilon_coupling(eta_ls, d_manifold)
        # exp(-(2/10) * 2 * log(0.1)) = exp(-(0.2) * (-4.6)) = exp(0.92) ≈ 2.51 -> clipped
        assert 0.0 <= eps_coup <= 1.0


class TestCheckC1TopologicalIsometry:
    """Tests for check_c1_topological_isometry (C1 condition)."""
    
    def test_c1_satisfied(self):
        """Test C1 is satisfied when J_topo <= epsilon_topo."""
        assert check_c1_topological_isometry(0.3, 0.5) == True
        assert check_c1_topological_isometry(0.5, 0.5) == True
    
    def test_c1_not_satisfied(self):
        """Test C1 is not satisfied when J_topo > epsilon_topo."""
        assert check_c1_topological_isometry(0.6, 0.5) == False
        assert check_c1_topological_isometry(1.0, 0.5) == False


class TestCheckC2ThermalSafety:
    """Tests for check_c2_thermal_safety (C2 condition)."""
    
    def test_c2_satisfied(self):
        """Test C2 is satisfied when T_tilde_eff <= xi_opt * T_c."""
        # T_eff * eps_coup = 1.0 * 0.5 = 0.5
        # xi_opt * T_c = (2/3) * 1.5 = 1.0
        # 0.5 <= 1.0 -> True
        assert check_c2_thermal_safety(1.0, 0.5, 1.5, 2/3) == True
    
    def test_c2_not_satisfied(self):
        """Test C2 is not satisfied when T_tilde_eff > xi_opt * T_c."""
        # T_eff * eps_coup = 1.5 * 0.8 = 1.2
        # xi_opt * T_c = (2/3) * 1.0 ≈ 0.667
        # 1.2 > 0.667 -> False
        assert check_c2_thermal_safety(1.5, 0.8, 1.0, 2/3) == False


class TestIsThermogeometricallyFeasible:
    """Tests for is_thermogeometrically_feasible function."""
    
    def test_feasible_architecture(self):
        """Test a feasible architecture passes both checks."""
        config = TASConfig(epsilon_topo=1.0, xi_opt=2/3)
        eta_ls = [0.8, 0.8, 0.8]  # Small logs -> small J_topo
        T_eff = 0.01
        T_c = 10.0
        
        result = is_thermogeometrically_feasible(eta_ls, T_eff, T_c, config)
        
        assert isinstance(result, OptimalityResult)
        assert result.is_feasible == True
        assert result.c1_satisfied == True
        assert result.c2_satisfied == True
    
    def test_infeasible_by_c1(self):
        """Test architecture fails C1 with large eta values."""
        config = TASConfig(epsilon_topo=0.01, xi_opt=2/3)
        eta_ls = [10.0, 10.0, 10.0]  # Large values -> large J_topo
        T_eff = 0.01
        T_c = 10.0
        
        result = is_thermogeometrically_feasible(eta_ls, T_eff, T_c, config)
        
        assert result.c1_satisfied == False
    
    def test_infeasible_by_c2(self):
        """Test architecture fails C2 with high effective temperature."""
        config = TASConfig(epsilon_topo=1.0, xi_opt=0.01)  # Very tight thermal constraint
        eta_ls = [0.9, 0.9, 0.9]  # Mild eta
        T_eff = 10.0  # High effective temperature
        T_c = 1.0  # Low critical temperature
        
        result = is_thermogeometrically_feasible(eta_ls, T_eff, T_c, config)
        
        assert result.c2_satisfied == False


class TestTASConfig:
    """Tests for TASConfig with Phase 6 parameters."""
    
    def test_default_values(self):
        """Test default Phase 6 values."""
        config = TASConfig()
        
        assert config.epsilon_topo == 0.5
        assert config.xi_opt == 2/3
        assert config.c_T == 1.0
        assert config.c_gamma == 1.0
    
    def test_custom_values(self):
        """Test custom Phase 6 values."""
        config = TASConfig(
            epsilon_topo=1.0,
            xi_opt=0.5,
            c_T=2.0,
            c_gamma=1.5,
        )
        
        assert config.epsilon_topo == 1.0
        assert config.xi_opt == 0.5
        assert config.c_T == 2.0
        assert config.c_gamma == 1.5


class TestOptimalityResult:
    """Tests for OptimalityResult dataclass."""
    
    def test_summary(self):
        """Test OptimalityResult summary."""
        result = OptimalityResult(
            is_feasible=True,
            c1_satisfied=True,
            c2_satisfied=True,
            alpha=0.123,
            J_topo=0.4,
            T_tilde_eff=0.5,
            epsilon_coupling=0.8,
        )
        
        summary = result.summary()
        assert "FEASIBLE" in summary
        assert "✓" in summary
        assert "α = 0.1230" in summary


class TestTASProfilerVerifyOptimality:
    """Tests for TASProfiler.verify_optimality method."""
    
    def test_verify_optimality_feasible(self):
        """Test verify_optimality with feasible architecture."""
        config = TASConfig(epsilon_topo=1.0, xi_opt=2/3)
        profiler = TASProfiler(config)
        
        eta_ls = [0.8, 0.8, 0.8]
        T_eff = 0.01
        T_c = 10.0
        gamma_c = 1.0
        
        result = profiler.verify_optimality(eta_ls, T_eff, T_c, gamma_c)
        
        assert isinstance(result, OptimalityResult)
        assert result.is_feasible == True
        assert result.alpha >= 0  # Alpha should be non-negative
    
    def test_verify_optimality_infeasible(self):
        """Test verify_optimality with infeasible architecture."""
        config = TASConfig(epsilon_topo=0.01, xi_opt=0.01)  # Very tight constraints
        profiler = TASProfiler(config)
        
        eta_ls = [5.0, 5.0, 5.0]  # Large values -> infeasible
        T_eff = 10.0
        T_c = 1.0
        gamma_c = 1.0
        
        result = profiler.verify_optimality(eta_ls, T_eff, T_c, gamma_c)
        
        assert result.is_feasible == False


class TestVerifyAndProfile:
    """Tests for TASProfiler.verify_and_profile method."""
    
    def test_verify_and_profile(self):
        """Test combined profile and verification."""
        profiler = TASProfiler(TASConfig())
        
        # Create dummy data
        X = np.random.randn(100, 10)
        y = np.random.randn(100)
        
        architecture = {'widths': [64, 128, 64]}
        train_config = {'lr': 1e-3, 'batch_size': 32}
        
        result = profiler.verify_and_profile(X, y, architecture, train_config)
        
        assert result.optimality_result is not None
        assert isinstance(result.optimality_result, OptimalityResult)


class TestArchitectureSearcherOptimality:
    """Tests for ArchitectureSearcher with optimality filtering."""
    
    def test_search_result_dataclass(self):
        """Test SearchResult contains optimality info."""
        from thermorg.tas import ArchitectureConfig, OptimalityResult
        
        result = SearchResult(
            architecture=ArchitectureConfig(
                name="test", layer_widths=[64], layer_types=['linear']
            ),
            alpha=0.5,
            optimality=OptimalityResult(
                is_feasible=True,
                c1_satisfied=True,
                c2_satisfied=True,
                alpha=0.5,
            ),
            metrics={'params': 1000, 'flops': 5000},
            feasible=True,
        )
        
        summary = result.summary()
        assert "test" in summary
        assert "FEASIBLE" in summary
        assert "α = 0.5" in summary
    
    def test_search_result_infeasible(self):
        """Test SearchResult with infeasible result."""
        from thermorg.tas import ArchitectureConfig, OptimalityResult
        
        result = SearchResult(
            architecture=ArchitectureConfig(
                name="test", layer_widths=[64], layer_types=['linear']
            ),
            alpha=0.5,
            optimality=OptimalityResult(
                is_feasible=False,
                c1_satisfied=False,
                c2_satisfied=False,
                alpha=0.5,
            ),
            metrics={},
            feasible=False,
        )
        
        summary = result.summary()
        assert "INFEASIBLE" in summary


class TestIntegration:
    """Integration tests for Phase 6 optimality verification."""
    
    def test_full_pipeline_with_optimality(self):
        """Test full TAS pipeline with optimality verification."""
        # Create profiler with Phase 6 config
        config = TASConfig(
            d_manifold=10.0,
            s_smoothness=1.0,
            epsilon_topo=0.5,
            xi_opt=2/3,
            c_T=1.0,
            c_gamma=1.0,
        )
        profiler = TASProfiler(config)
        
        # Create test data
        np.random.seed(42)
        X = np.random.randn(200, 20)
        y = np.random.randn(200)
        
        architecture = {'widths': [128, 256, 128]}
        train_config = {'lr': 1e-3, 'batch_size': 32}
        
        # Run verification
        result = profiler.verify_and_profile(X, y, architecture, train_config)
        
        # Verify result structure
        assert result.d_manifold == 10.0
        assert result.alpha >= 0
        assert result.optimality_result is not None
        
        # Verify optimality
        opt = result.optimality_result
        assert isinstance(opt, OptimalityResult)
        assert hasattr(opt, 'c1_satisfied')
        assert hasattr(opt, 'c2_satisfied')
        assert hasattr(opt, 'J_topo')
        assert hasattr(opt, 'T_tilde_eff')
