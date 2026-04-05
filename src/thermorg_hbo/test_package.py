#!/usr/bin/env python3
"""
Tests for thermorg_hbo package.
Run: python -m thermorg_hbo.test_package
"""

import sys
import numpy as np

def test_import():
    """Test that package imports correctly."""
    print("test_import...", end=" ")
    from thermorg_hbo import (
        Architecture,
        ArchitectureSpace,
        arch_to_features,
        GPSurrogate,
        EIAcquisition,
        SimulationEnv,
    )
    print("OK")
    return True

def test_architecture():
    """Test Architecture class."""
    print("test_architecture...", end=" ")
    from thermorg_hbo.arch.encoding import Architecture

    arch = Architecture(width=32, depth=5, skip=True, norm='bn')
    assert arch.width == 32
    assert arch.depth == 5
    assert arch.skip == True
    assert arch.norm == 'bn'
    assert arch.n_params_M > 0
    assert len(arch.norm_onehot) == 3
    assert arch.norm_onehot.tolist() == [0, 1, 0]  # bn

    # Feature vector (6 dims: width, depth, skip, 3-norm-onehot)
    fv = arch.to_features()
    assert fv.shape == (6,), f"Expected (6,), got {fv.shape}"
    assert fv.dtype == np.float32

    print("OK")
    return True

def test_architecture_space():
    """Test ArchitectureSpace class."""
    print("test_architecture_space...", end=" ")
    from thermorg_hbo.arch.encoding import ArchitectureSpace, Architecture

    space = ArchitectureSpace()

    # Sample
    archs = space.sample(10)
    assert len(archs) == 10
    for a in archs:
        assert isinstance(a, Architecture)

    # Grid
    grid = space.grid()
    assert len(grid) == 4 * 4 * 2 * 3  # 96 configs

    # To matrix
    X = space.to_matrix(archs[:5])
    assert X.shape == (5, 6), f"Expected (5, 6), got {X.shape}"

    print("OK")
    return True

def test_simulation_env():
    """Test SimulationEnv."""
    print("test_simulation_env...", end=" ")
    from thermorg_hbo import SimulationEnv
    from thermorg_hbo.arch.encoding import Architecture

    env = SimulationEnv(seed=42)

    arch = Architecture(width=32, depth=5, skip=True, norm='bn')

    # J_topo (zero-cost)
    j1 = env.get_j_topo(arch)
    j2 = env.get_j_topo(arch)
    assert isinstance(j1, float)
    assert 0.1 <= j1 <= 0.9

    # Ground truth
    gt = env.get_ground_truth(arch)
    assert hasattr(gt, 'j_topo')
    assert hasattr(gt, 'beta')
    assert hasattr(gt, 'e_floor')
    assert 0.05 < gt.e_floor < 1.5
    assert 0.1 < gt.beta < 0.5

    # Evaluate at different fidelities
    loss1, gt1 = env.evaluate(arch, fidelity=1)
    loss2, gt2 = env.evaluate(arch, fidelity=2)
    loss3, gt3 = env.evaluate(arch, fidelity=3)

    # Higher fidelity = lower noise = closer to true value
    gt_loss = gt.alpha * (96 ** (-gt.beta)) + gt.e_floor
    assert abs(loss1 - gt_loss) < 0.3  # L1 has 5% noise
    assert abs(loss2 - gt_loss) < 0.15  # L2 has 2% noise
    assert abs(loss3 - gt_loss) < 0.05  # L3 has 0.5% noise

    print("OK")
    return True

def test_gp_surrogate():
    """Test GPSurrogate."""
    print("test_gp_surrogate...", end=" ")
    from thermorg_hbo import ArchitectureSpace, GPSurrogate, SimulationEnv

    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
    except ImportError:
        print("SKIP (sklearn not available)")
        return True

    space = ArchitectureSpace()
    env = SimulationEnv(seed=42)

    # Sample architectures
    archs = space.sample(20)

    # Create GP
    gp = GPSurrogate()

    # Add J_topo observations (zero-cost)
    for arch in archs[:10]:
        j_topo = env.get_j_topo(arch)
        gp.add_j_topo(arch, j_topo)

    # Add loss observations
    for arch in archs[:5]:
        j_topo = env.get_j_topo(arch)
        loss, _ = env.evaluate(arch, fidelity=2)
        gp.add_loss(arch, j_topo, fidelity=2, loss=loss)

    # Fit GP
    gp.fit()

    # Predict
    mean, std = gp.predict([archs[0]], [env.get_j_topo(archs[0])])
    assert isinstance(mean[0], (float, np.floating))
    assert isinstance(std[0], (float, np.floating))

    # Best observation
    best = gp.best_observation()
    assert best is not None
    assert best.loss is not None

    print("OK")
    return True

def test_ei_acquisition():
    """Test EIAcquisition."""
    print("test_ei_acquisition...", end=" ")
    from thermorg_hbo import ArchitectureSpace, GPSurrogate, EIAcquisition, SimulationEnv

    try:
        from scipy.stats import norm
    except ImportError:
        print("SKIP (scipy not available)")
        return True

    space = ArchitectureSpace()
    env = SimulationEnv(seed=42)

    archs = space.sample(10)
    gp = GPSurrogate()
    acq = EIAcquisition(gp, xi=0.01)

    # Add some observations
    for arch in archs[:3]:
        j_topo = env.get_j_topo(arch)
        loss, _ = env.evaluate(arch, fidelity=2)
        gp.add_loss(arch, j_topo, fidelity=2, loss=loss)

    gp.fit()

    # Score batch
    j_topos = [env.get_j_topo(a) for a in archs]
    scores = acq.score_batch(archs, j_topos, [2]*len(archs))
    assert scores.shape == (len(archs),)
    assert all(s >= 0 for s in scores)

    # Select top-k
    top_k = acq.select_top_k(archs, j_topos, [2]*len(archs), k=3)
    assert len(top_k) == 3
    assert all(isinstance(idx, int) and isinstance(score, float) for idx, score in top_k)

    print("OK")
    return True

def test_j_topo_e_floor_correlation():
    """Test that J_topo → E_floor relationship is correct."""
    print("test_j_topo_e_floor_correlation...", end=" ")
    from thermorg_hbo import ArchitectureSpace, SimulationEnv

    space = ArchitectureSpace()
    env = SimulationEnv(seed=42)
    archs = space.sample(50)

    j_topos = []
    e_floors = []
    for arch in archs:
        j = env.get_j_topo(arch)
        gt = env.get_ground_truth(arch)
        j_topos.append(j)
        e_floors.append(gt.e_floor)

    r = np.corrcoef(j_topos, e_floors)[0, 1]
    assert r > 0.5, f"J_topo-E_floor correlation too low: {r:.3f}"
    assert r < 1.0, f"J_topo-E_floor correlation > 1: {r:.3f}"

    print(f"OK (r={r:.3f})")
    return True

def test_full_pipeline():
    """Test full HBO pipeline: sample → J_topo → GP → EI → select → evaluate."""
    print("test_full_pipeline...", end=" ")
    from thermorg_hbo import ArchitectureSpace, GPSurrogate, EIAcquisition, SimulationEnv

    space = ArchitectureSpace(width_range=[8, 32, 64], depth_range=[3, 7])
    env = SimulationEnv(seed=42)

    # Sample 50 candidates
    candidates = space.sample(50)

    # Compute J_topo for all (zero-cost)
    j_topos = [env.get_j_topo(a) for a in candidates]

    # Initialize GP
    gp = GPSurrogate()
    for arch, j in zip(candidates, j_topos):
        gp.add_j_topo(arch, j)

    # Initial evaluation: train top-3 at L1
    acq = EIAcquisition(gp, xi=0.01)
    top3_idx, _ = zip(*acq.select_top_k(candidates, j_topos, [1]*len(candidates), k=3))

    losses = []
    for idx in top3_idx:
        loss, _ = env.evaluate(candidates[idx], fidelity=2)
        losses.append(loss)
        gp.add_loss(candidates[idx], j_topos[idx], fidelity=2, loss=loss)

    gp.fit()

    # Second round: select by EI
    top5_idx, _ = zip(*acq.select_top_k(candidates, j_topos, [2]*len(candidates), k=5))

    # Best found
    best_obs = gp.best_observation()
    assert best_obs is not None
    assert best_obs.loss is not None

    print("OK")
    return True

def run_all():
    """Run all tests."""
    print("=" * 60)
    print("ThermoRG-HBO Package Tests")
    print("=" * 60)
    print()

    tests = [
        test_import,
        test_architecture,
        test_architecture_space,
        test_simulation_env,
        test_gp_surrogate,
        test_ei_acquisition,
        test_j_topo_e_floor_correlation,
        test_full_pipeline,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"FAIL: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0

if __name__ == '__main__':
    success = run_all()
    sys.exit(0 if success else 1)
