# SPDX-License-Identifier: Apache-2.0

"""Example: Phase 6 Optimality Verification for ThermoRG-NN

This example demonstrates how to use the Phase 6 optimality verification
to check whether an architecture satisfies thermogeometric feasibility
conditions C1 and C2.

C1 (Topological Isometry): J_topo = |∑log η_l| ≤ ε_topo
C2 (Thermal Safety): T̃_eff = T_eff · ε_coupling ≤ ξ_opt · T_c
"""

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


def example_basic_verification():
    """Basic example of optimality verification."""
    print("=" * 60)
    print("Example 1: Basic Optimality Verification")
    print("=" * 60)
    
    # Create configuration with Phase 6 parameters
    config = TASConfig(
        d_manifold=10.0,
        s_smoothness=1.0,
        epsilon_topo=0.5,  # Geometric tolerance
        xi_opt=2/3,         # Thermal safety margin
        c_T=1.0,           # T_c calibration constant
        c_gamma=1.0,       # γ_c calibration constant
    )
    
    profiler = TASProfiler(config)
    
    # Define architecture
    architecture = {'widths': [128, 256, 128, 64]}
    train_config = {'lr': 1e-3, 'batch_size': 32}
    
    # Create dummy data
    np.random.seed(42)
    X = np.random.randn(500, 20)
    y = np.random.randn(500)
    
    # Run full profiling with optimality verification
    result = profiler.verify_and_profile(X, y, architecture, train_config)
    
    print(f"\nArchitecture: {architecture['widths']}")
    print(f"Alpha: {result.alpha:.4f}")
    print(f"\nOptimality Verification:")
    print(f"  ε_topo (tolerance): {config.epsilon_topo}")
    print(f"  ξ_opt (safety margin): {config.xi_opt:.4f}")
    
    if result.optimality_result:
        opt = result.optimality_result
        print(f"\n  C1 (Topological Isometry): {'✓ SATISFIED' if opt.c1_satisfied else '✗ NOT SATISFIED'}")
        print(f"    J_topo = |∑log η_l| = {opt.J_topo:.4f} (threshold: {config.epsilon_topo})")
        
        print(f"\n  C2 (Thermal Safety): {'✓ SATISFIED' if opt.c2_satisfied else '✗ NOT SATISFIED'}")
        print(f"    T̃_eff = {opt.T_tilde_eff:.4f} (threshold: {config.xi_opt * result.T_c:.4f})")
        print(f"    ε_coupling = {opt.epsilon_coupling:.4f}")
        
        print(f"\n  Overall: {'✓ THERMOGEOMETRICALLY FEASIBLE' if opt.is_feasible else '✗ INFEASIBLE'}")
    
    return result


def example_manual_functions():
    """Example using the Phase 6 functions directly."""
    print("\n" + "=" * 60)
    print("Example 2: Manual Optimality Verification Functions")
    print("=" * 60)
    
    # Sample architecture parameters
    eta_ls = [0.5, 0.6, 0.7, 0.8]
    d_manifold = 8.0
    T_eff = 0.001
    T_c = 5.0
    
    config = TASConfig(epsilon_topo=0.5, xi_opt=2/3)
    
    # Step 1: Compute ε_coupling
    epsilon_coupling = compute_epsilon_coupling(eta_ls, d_manifold)
    print(f"\nε_coupling = exp(-(2/{d_manifold}) * ∑log η_l) = {epsilon_coupling:.4f}")
    
    # Step 2: Compute J_topo
    J_topo = abs(sum(np.log(eta) for eta in eta_ls))
    print(f"J_topo = |∑log η_l| = {J_topo:.4f}")
    
    # Step 3: Check C1
    c1_satisfied = check_c1_topological_isometry(J_topo, config.epsilon_topo)
    print(f"\nC1 Check: J_topo ≤ ε_topo")
    print(f"  {J_topo:.4f} ≤ {config.epsilon_topo} → {'✓ PASS' if c1_satisfied else '✗ FAIL'}")
    
    # Step 4: Compute T_tilde_eff and check C2
    T_tilde_eff = T_eff * epsilon_coupling
    threshold = config.xi_opt * T_c
    c2_satisfied = check_c2_thermal_safety(T_eff, epsilon_coupling, T_c, config.xi_opt)
    print(f"\nC2 Check: T_eff * ε_coupling ≤ ξ_opt * T_c")
    print(f"  {T_eff:.4f} * {epsilon_coupling:.4f} = {T_tilde_eff:.6f}")
    print(f"  ≤ {config.xi_opt:.4f} * {T_c:.4f} = {threshold:.4f} → {'✓ PASS' if c2_satisfied else '✗ FAIL'}")
    
    # Step 5: Combined check
    result = is_thermogeometrically_feasible(eta_ls, T_eff, T_c, config)
    print(f"\nOverall: {'✓ FEASIBLE' if result.is_feasible else '✗ INFEASIBLE'}")
    
    return result


def example_search_with_optimality():
    """Example architecture search with optimality filtering."""
    print("\n" + "=" * 60)
    print("Example 3: Architecture Search with Optimality Filtering")
    print("=" * 60)
    
    # Create profiler
    config = TASConfig(
        d_manifold=10.0,
        epsilon_topo=0.5,
        xi_opt=2/3,
    )
    profiler = TASProfiler(config)
    
    # Create searcher
    searcher = ArchitectureSearcher(max_candidates=20)
    
    # Define predictor and verifier functions
    def alpha_predictor(arch, **kwargs):
        result = profiler.profile_architecture(
            {
                'widths': arch.layer_widths,
                'types': arch.layer_types,
            },
            {'lr': 1e-3, 'batch_size': 32}
        )
        return result.alpha
    
    def optimality_verifier(arch, **kwargs):
        result = profiler.profile_architecture(
            {
                'widths': arch.layer_widths,
                'types': arch.layer_types,
            },
            {'lr': 1e-3, 'batch_size': 32}
        )
        return profiler.verify_optimality(
            eta_ls=result.eta_ls,
            T_eff=result.T_eff,
            T_c=result.T_c,
            gamma_c=1.0,
        )
    
    # Run search with optimality filtering
    search_result = searcher.search_with_optimality(
        alpha_predictor=alpha_predictor,
        optimality_verifier=optimality_verifier,
        input_dim=20,
    )
    
    print(f"\nBest Architecture: {search_result.architecture.name}")
    print(f"Layer Widths: {search_result.architecture.layer_widths}")
    print(f"Alpha: {search_result.alpha:.4f}")
    print(f"Feasible: {'✓' if search_result.feasible else '✗'}")
    
    if search_result.optimality:
        print(f"  C1: {'✓' if search_result.optimality.c1_satisfied else '✗'}")
        print(f"  C2: {'✓' if search_result.optimality.c2_satisfied else '✗'}")
    
    params = search_result.metrics.get('params')
    flops = search_result.metrics.get('flops')
    print(f"Params: {params if params is None else f'{params:,}'}")
    print(f"FLOPs: {flops if flops is None else f'{flops:,}'}")
    
    return search_result


def example_compare_architectures():
    """Example comparing multiple architectures by optimality."""
    print("\n" + "=" * 60)
    print("Example 4: Comparing Architectures by Optimality")
    print("=" * 60)
    
    config = TASConfig(epsilon_topo=0.5, xi_opt=2/3)
    profiler = TASProfiler(config)
    
    architectures = [
        {'name': 'small', 'widths': [32, 32]},
        {'name': 'medium', 'widths': [128, 128]},
        {'name': 'wide', 'widths': [512, 512]},
        {'name': 'deep', 'widths': [64, 64, 64, 64, 64]},
    ]
    
    train_config = {'lr': 1e-3, 'batch_size': 32}
    
    print(f"\n{'Architecture':<15} {'Alpha':>8} {'J_topo':>8} {'ε_coup':>8} {'C1':>4} {'C2':>4} {'Feasible':>10}")
    print("-" * 70)
    
    for arch in architectures:
        result = profiler.profile_architecture(arch, train_config)
        opt = profiler.verify_optimality(
            eta_ls=result.eta_ls,
            T_eff=result.T_eff,
            T_c=result.T_c,
            gamma_c=1.0,
        )
        
        feasible_str = "✓ YES" if opt.is_feasible else "✗ NO"
        print(f"{arch['name']:<15} {result.alpha:>8.4f} {opt.J_topo:>8.4f} "
              f"{opt.epsilon_coupling:>8.4f} {'✓' if opt.c1_satisfied else '✗':>4} "
              f"{'✓' if opt.c2_satisfied else '✗':>4} {feasible_str:>10}")


if __name__ == "__main__":
    # Run all examples
    example_basic_verification()
    example_manual_functions()
    example_search_with_optimality()
    example_compare_architectures()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
