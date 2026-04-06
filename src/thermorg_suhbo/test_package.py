#!/usr/bin/env python3
"""
Test script for SU-HBO package.
"""

import sys
import numpy as np

# Add src to path
sys.path.insert(0, '/home/node/.openclaw/workspace/github_staging/ThermoRG-NN/src')

from thermorg_suhbo import (
    SUHBO, SUHBOConfig,
    Architecture, ArchConfig, get_baseline,
    Action, ActionLibrary,
    compute_utility, compute_e_floor, compute_beta, compute_cooling_factor,
    is_stable, get_stability_margin,
    PlateauDetector,
    GPSurrogate,
    expected_improvement,
    DEFAULT_K, DEFAULT_B, DEFAULT_GAMMA_C, DEFAULT_LAMBDA,
    GAMMA_BN, GAMMA_NONE, BETA_BN, BETA_NONE,
)


def test_utility():
    """Test utility function computations."""
    print("Testing utility functions...")

    # Test E_floor
    e = compute_e_floor(j_topo=0.3, gamma=2.5)
    print(f"  E_floor(J=0.3, γ=2.5) = {e:.4f}")

    # Test cooling factor
    phi = compute_cooling_factor(gamma=2.5, gamma_c=2.0)
    print(f"  φ(γ=2.5) = {phi:.4f}")

    # Test beta
    beta = compute_beta(gamma=2.5, norm_type='bn')
    print(f"  β(γ=2.5, BN) = {beta:.4f}")

    # Test utility
    u = compute_utility(j_topo=0.3, gamma=2.5, norm_type='bn')
    print(f"  U(J=0.3, γ=2.5, BN) = {u:.4f}")

    # Test stability
    stable = is_stable(j_topo=0.3, gamma=2.5)
    print(f"  Stable(J=0.3, γ=2.5) = {stable}")

    margin = get_stability_margin(j_topo=0.3, gamma=2.5)
    print(f"  Stability margin = {margin:.4f}")

    print("  ✓ Utility functions work\n")


def test_architecture():
    """Test architecture representation."""
    print("Testing architecture...")

    config = ArchConfig(name="test", width=64, depth=5, skip=True, norm='bn')
    arch = Architecture(config)

    j = arch.compute_j_topo()
    print(f"  J_topo for W64-D5-Skip-BN = {j:.4f}")

    features = arch.to_feature_vector()
    print(f"  Feature vector length = {len(features)}")

    baseline = get_baseline('image')
    print(f"  Baseline image arch = {baseline.config}")

    print("  ✓ Architecture works\n")


def test_action_library():
    """Test action library."""
    print("Testing action library...")

    config = ArchConfig(name="test", width=64, depth=5, skip=False, norm='none')
    library = ActionLibrary()

    actions = library.get_available_actions(config)
    print(f"  Available actions for baseline: {len(actions)}")
    for action in actions:
        print(f"    - {action.name}: ΔJ={action.delta_j:.2f}, Δγ={action.delta_gamma:.2f}")

    # Apply action
    add_bn = [a for a in actions if a.name == 'add_bn'][0]
    new_config = add_bn.apply(config)
    print(f"  After add_bn: norm={new_config.norm}")

    print("  ✓ Action library works\n")


def test_plateau():
    """Test plateau detection."""
    print("Testing plateau detection...")

    detector = PlateauDetector()

    # Simulate training
    for epoch in range(30):
        beta = 0.3 - 0.01 * epoch / 30 + np.random.normal(0, 0.005)
        gamma = 2.5 + 0.1 * np.sin(epoch / 10) + np.random.normal(0, 0.02)
        loss = 0.5 * np.exp(-0.05 * epoch) + 0.2 + np.random.normal(0, 0.01)
        detector.update(epoch, beta, gamma, loss)

    is_plateau = detector.is_plateau()
    phase = detector.get_phase()
    print(f"  Is plateau: {is_plateau}")
    print(f"  Phase: {phase}")

    print("  ✓ Plateau detection works\n")


def test_surrogate():
    """Test GP surrogate."""
    print("Testing GP surrogate...")

    # Create dummy data
    X_train = np.random.randn(10, 8)
    y_train = np.random.randn(10)

    surrogate = GPSurrogate(
        lambda_param=10.0,
        k=0.06,
        B=0.15,
        gamma_c=2.0
    )

    surrogate.fit(X_train, y_train)

    # Predict
    X_test = np.random.randn(3, 8)
    mu, sigma = surrogate.predict(X_test)
    print(f"  Predictions shape: {mu.shape}")

    print("  ✓ GP surrogate works\n")


def test_acquisition():
    """Test acquisition function."""
    print("Testing acquisition function...")

    mu = np.array([0.5, 0.6, 0.4])
    sigma = np.array([0.1, 0.2, 0.05])
    f_best = 0.55

    ei = expected_improvement(mu, sigma, f_best)
    print(f"  EI values: {ei}")

    print("  ✓ Acquisition function works\n")


def test_suhbo_init():
    """Test SU-HBO initialization."""
    print("Testing SU-HBO...")

    config = SUHBOConfig(
        lambda_param=10.0,
        max_iterations=10,
    )

    algo = SUHBO(config=config)
    print(f"  Config: λ={algo.config.lambda_param}")

    algo.initialize(task_type='image')
    print(f"  Initialized with: {algo.current_arch.config}")

    print("  ✓ SU-HBO works\n")


def main():
    print("=" * 60)
    print("SU-HBO Package Test")
    print("=" * 60)
    print()

    try:
        test_utility()
        test_architecture()
        test_action_library()
        test_plateau()
        test_surrogate()
        test_acquisition()
        test_suhbo_init()

        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
