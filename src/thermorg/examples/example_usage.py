#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Example usage of the ThermoRG TAS pipeline.

This script demonstrates how to use the Thermogeometric Architecture Search
pipeline to analyze neural network architectures and predict scaling exponents.
"""

import numpy as np
import sys

sys.path.insert(0, '/home/node/.openclaw/workspace/agents/coding/work/codebase/ThermoRG-NN/src')

from thermorg.tas import (
    TASProfiler, TASConfig,
    ManifoldEstimator, SmoothnessEstimator,
    ArchitectureAnalyzer,
    TemperatureEstimator, ThermalPhaseComputer, CoolingPhaseComputer,
)


def example_basic_profiling():
    """Basic example of TAS profiling."""
    print("=" * 60)
    print("Example 1: Basic TAS Profiling")
    print("=" * 60)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 500
    n_features = 20
    
    # Create manifold-structured data
    t = np.random.randn(n_samples, 3)
    X = np.column_stack([
        np.sin(t[:, 0]) * 2 + 0.1 * np.random.randn(n_samples),
        np.cos(t[:, 1]) * 2 + 0.1 * np.random.randn(n_samples),
        t[:, 2] + 0.1 * np.random.randn(n_samples),
        np.random.randn(n_samples, n_features - 3)
    ])
    
    # Create target function on manifold
    y = np.sin(t[:, 0]) * np.cos(t[:, 1]) + 0.1 * np.random.randn(n_samples)
    
    # Define architecture
    architecture = {
        'widths': [64, 128, 256, 128],
        'types': ['linear', 'linear', 'linear', 'linear']
    }
    
    # Define training configuration
    train_config = {
        'lr': 1e-3,
        'batch_size': 32,
    }
    
    # Create profiler with custom config
    config = TASConfig(
        eta_lr=1e-3,
        batch_size=32,
        gamma_T=1.0,
        gamma_cool=0.0,
        epsilon=1.0
    )
    
    profiler = TASProfiler(config)
    
    # Run profiling
    result = profiler.profile(X, y, architecture, train_config)
    
    print(f"\nManifold dimension (d_manifold): {result.d_manifold:.4f}")
    print(f"Smoothness (s): {result.s_smoothness:.4f}")
    print(f"Gradient variance (V_grad): {result.V_grad:.4f}")
    print(f"Critical temperature (T_c): {result.T_c:.4f}")
    print(f"η_l values: {[f'{eta:.4f}' for eta in result.eta_ls]}")
    print(f"η_product: {result.eta_product:.4f}")
    print(f"Effective temperature (T_eff): {result.T_eff:.6f}")
    print(f"Scaled temperature (T_tilde): {result.T_tilde:.6f}")
    print(f"Thermal phase (Ψ_algo): {result.psi_algo:.4f}")
    print(f"Cooling phase (φ_cool): {result.phi_cool:.4f}")
    print(f"\n>>> Predicted Scaling Exponent (α): {result.alpha:.6f}")
    
    print("\n" + result.summary())
    return result


def example_manual_components():
    """Example using components manually."""
    print("\n" + "=" * 60)
    print("Example 2: Manual Component Usage")
    print("=" * 60)
    
    # Generate data
    np.random.seed(123)
    X = np.random.randn(300, 15)
    y = np.sin(X[:, 0]) * X[:, 1] + 0.1 * np.random.randn(300)
    
    # Step 1: Estimate manifold dimension
    manifold_est = ManifoldEstimator(k_max=15)
    d_manifold = manifold_est.estimate_d(X)
    print(f"\nManifold dimension: {d_manifold:.4f}")
    
    # Step 2: Estimate smoothness
    smoothness_est = SmoothnessEstimator(k_neighbors=10)
    s = smoothness_est.estimate_s(y, X)
    print(f"Smoothness: {s:.4f}")
    
    # Step 3: Compute architecture efficiency
    arch_analyzer = ArchitectureAnalyzer()
    eta_ls = arch_analyzer.compute_heuristic_eta(
        layer_widths=[64, 128, 256],
        d_manifold=d_manifold
    )
    print(f"η_l values: {[f'{eta:.4f}' for eta in eta_ls]}")
    
    # Step 4: Thermodynamic factors
    temp_est = TemperatureEstimator()
    T_eff = temp_est.estimate_T_eff(eta_lr=1e-3, noise_variance=0.1, batch_size=32)
    T_c = temp_est.estimate_T_c(d_manifold, V_grad=1.0)
    T_tilde = temp_est.compute_scaling_temperature(eta_ls, T_eff, d_manifold)
    
    print(f"\nT_eff: {T_eff:.6f}")
    print(f"T_c: {T_c:.4f}")
    print(f"T_tilde: {T_tilde:.6f}")
    
    # Step 5: Thermal and cooling phases
    psi_computer = ThermalPhaseComputer(gamma_T=1.0)
    psi = psi_computer.compute_psi(T_tilde, T_c, delta_loss=0.0, T_eff=T_eff)
    
    phi_computer = CoolingPhaseComputer()
    gamma_c = phi_computer.estimate_gamma_c(eta_lr=1e-3, V_grad=1.0, d_manifold=d_manifold)
    phi = phi_computer.compute_phi(gamma=0.0, gamma_c=gamma_c)
    
    print(f"\nΨ_algo: {psi:.4f}")
    print(f"φ_cool: {phi:.4f}")
    print(f"γ_c: {gamma_c:.6f}")
    
    # Compute final alpha
    eta_product = np.prod(eta_ls)
    k_alpha = 1.0
    alpha = k_alpha * abs(np.log(eta_product)) * (2 * s / d_manifold) * psi * phi
    
    print(f"\n>>> Final α: {alpha:.6f}")


def example_compare_architectures():
    """Example comparing different architectures."""
    print("\n" + "=" * 60)
    print("Example 3: Architecture Comparison")
    print("=" * 60)
    
    # Generate data
    np.random.seed(456)
    X = np.random.randn(400, 20)
    y = X[:, 0] ** 2 + np.sin(X[:, 1]) + 0.1 * np.random.randn(400)
    
    # Different architectures to compare
    architectures = [
        {'name': 'Narrow Deep', 'widths': [32, 32, 64, 64, 128]},
        {'name': 'Wide Shallow', 'widths': [512, 256]},
        {'name': 'Balanced', 'widths': [64, 128, 128, 64]},
        {'name': 'Pyramid', 'widths': [256, 128, 64, 32]},
    ]
    
    train_config = {'lr': 1e-3, 'batch_size': 32}
    
    profiler = TASProfiler(TASConfig(d_manifold=None, s_smoothness=None))
    
    results = []
    
    print("\nComparing architectures:")
    print("-" * 50)
    
    for arch in architectures:
        result = profiler.profile_architecture(
            architecture={'widths': arch['widths']},
            train_config=train_config,
            X=X, y=y
        )
        
        results.append({
            'name': arch['name'],
            'widths': arch['widths'],
            'alpha': result.alpha,
            'd_manifold': result.d_manifold,
            'eta_product': result.eta_product
        })
        
        print(f"\n{arch['name']}:")
        print(f"  Widths: {arch['widths']}")
        print(f"  d_manifold: {result.d_manifold:.4f}")
        print(f"  η_product: {result.eta_product:.4f}")
        print(f"  α: {result.alpha:.6f}")
    
    # Sort by alpha
    results.sort(key=lambda x: x['alpha'], reverse=True)
    
    print("\n" + "-" * 50)
    print("Ranking by α:")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['name']}: α = {r['alpha']:.6f}")


def example_jacobian_analysis():
    """Example demonstrating Jacobian analysis."""
    print("\n" + "=" * 60)
    print("Example 4: Jacobian Analysis")
    print("=" * 60)
    
    from thermorg.tas.architecture import JacobianAnalyzer
    
    analyzer = JacobianAnalyzer()
    
    # Test with different Jacobian matrices
    test_cases = [
        ('Random', np.random.randn(10, 5)),
        ('Low-rank', np.random.randn(10, 1) @ np.random.randn(1, 5)),
        ('Identity', np.eye(5)),
        ('Orthogonal', np.linalg.qr(np.random.randn(10, 10))[0][:, :5]),
    ]
    
    print("\nD_eff for different Jacobian types:")
    print("-" * 40)
    
    for name, J in test_cases:
        d_eff = analyzer.compute_d_eff(J)
        print(f"  {name}: D_eff = {d_eff:.4f}")


def example_search_integration():
    """Example integrating with architecture search."""
    print("\n" + "=" * 60)
    print("Example 5: Architecture Search Integration")
    print("=" * 60)
    
    from thermorg.tas import ArchitectureSearcher, TASConfig, ConstraintBounds
    
    # Create searcher with constraints
    constraints = ConstraintBounds(
        params_max=1e6,  # 1M parameters
        flops_max=1e9     # 1B FLOPs
    )
    
    searcher = ArchitectureSearcher(
        constraints=constraints,
        max_candidates=50
    )
    
    # Define alpha predictor function
    def alpha_predictor(arch_config, **kwargs):
        profiler = kwargs.get('profiler')
        arch_dict = {'widths': arch_config.layer_widths}
        train_config = kwargs.get('train_config', {'lr': 1e-3, 'batch_size': 32})
        
        result = profiler.profile_architecture(arch_dict, train_config)
        return result.alpha
    
    # Create profiler
    profiler = TASProfiler(TASConfig(d_manifold=10.0, s_smoothness=1.0))
    
    # Run search
    print("\nRunning grid search...")
    best_arch, best_alpha, metadata = searcher.search_grid(
        width_options=[32, 64, 128, 256],
        depth_options=[2, 3, 4],
        alpha_predictor=alpha_predictor,
        input_dim=20,
        profiler=profiler,
        train_config={'lr': 1e-3, 'batch_size': 32}
    )
    
    print(f"\nBest Architecture Found:")
    print(f"  Name: {best_arch.name}")
    print(f"  Widths: {best_arch.layer_widths}")
    print(f"  Layers: {best_arch.n_layers}")
    print(f"  α: {best_alpha:.6f}")
    print(f"  Candidates evaluated: {metadata.get('candidates', 'N/A')}")


def main():
    """Run all examples."""
    try:
        example_basic_profiling()
        example_manual_components()
        example_compare_architectures()
        example_jacobian_analysis()
        example_search_integration()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
