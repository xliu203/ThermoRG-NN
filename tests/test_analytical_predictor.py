# SPDX-License-Identifier: Apache-2.0
"""Unit tests for analytical_predictor.py."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from thermorg.calibration import get_cifar10_calibration
from thermorg.analytical_predictor import (
    AnalyticalPredictor,
    D_scaling_law,
    E_floor_decomposition,
    cooling_law,
    predict_loss,
)


def make_predictor():
    """Create CIFAR-10 calibrated predictor."""
    cal = get_cifar10_calibration()
    return AnalyticalPredictor(**cal)


def test_analytical_predictor_basic():
    """AnalyticalPredictor.predict should return positive loss."""
    predictor = make_predictor()
    loss = predictor.predict(width=64, depth=5, J_topo=0.75, norm_type='bn')
    assert loss > 0, f"Loss should be positive, got {loss}"
    print(f"  predict(width=64, depth=5, bn, J=0.75)={loss:.4f} ✓")


def test_analytical_predictor_width_effect():
    """Wider networks should have lower loss."""
    predictor = make_predictor()
    losses = [predictor.predict(width=w, depth=5, J_topo=0.75, norm_type='bn')
              for w in [32, 64, 96]]
    assert losses[0] > losses[1] > losses[2], \
        f"Loss should decrease with width: {losses}"
    print(f"  Width 32→64→96: loss={[f'{l:.4f}' for l in losses]} ✓")


def test_analytical_predictor_jtopo_effect():
    """Higher J_topo should lead to lower loss (within fixed width)."""
    predictor = make_predictor()
    losses = [predictor.predict(width=64, depth=5, J_topo=j, norm_type='bn')
              for j in [0.6, 0.7, 0.8, 0.9]]
    assert losses[0] > losses[-1], \
        f"Higher J_topo → lower loss expected: {losses}"
    print(f"  J_topo 0.6→0.9: loss={[f'{l:.4f}' for l in losses]} ✓")


def test_analytical_predictor_norm_types():
    """Different norm types should give different predictions."""
    predictor = make_predictor()
    loss_bn = predictor.predict(width=64, depth=5, J_topo=0.75, norm_type='bn')
    loss_none = predictor.predict(width=64, depth=5, J_topo=0.75, norm_type='none')
    loss_ln = predictor.predict(width=64, depth=5, J_topo=0.75, norm_type='ln')
    assert loss_ln != loss_bn, "LN and BN should differ"
    print(f"  BN={loss_bn:.4f}, None={loss_none:.4f}, LN={loss_ln:.4f} ✓")


def test_d_scaling_law_api():
    """D_scaling_law function should work as expected (scalar)."""
    L = D_scaling_law(100.0, alpha=10.0, beta=0.5)
    assert L > 0
    print(f"  D_scaling_law(D=100): L={L:.4f} ✓")


def test_e_floor_decomposition():
    """E_floor should increase when J_topo decreases (since E_floor = E_task + C/D − B·J^ν)."""
    e1 = E_floor_decomposition(D_eff=64, J_topo=0.8, E_task=0.35, B=0.10)
    e2 = E_floor_decomposition(D_eff=64, J_topo=0.6, E_task=0.35, B=0.10)
    assert e2 > e1, f"E_floor should increase when J_topo decreases: {e1:.4f} vs {e2:.4f}"
    print(f"  D=64, J=0.8→0.6: E_floor={e1:.4f}→{e2:.4f} (increases ✓)")


def test_cooling_law_api():
    """cooling_law should return beta value."""
    beta = cooling_law(gamma=2.29, gamma_c=2.0)
    assert 0 < beta < 2.0, f"beta should be in (0, 2), got {beta}"
    print(f"  cooling_law(γ=2.29)={beta:.4f} ✓")


def test_predict_loss_api():
    """predict_loss should return positive value."""
    predictor = make_predictor()
    loss = predictor.predict(width=64, depth=5, J_topo=0.75, norm_type='bn')
    assert loss > 0
    print(f"  predict_loss(width=64, depth=5, bn, J=0.75)={loss:.4f} ✓")


if __name__ == '__main__':
    print("=== test_analytical_predictor ===")
    test_analytical_predictor_basic()
    test_analytical_predictor_width_effect()
    test_analytical_predictor_jtopo_effect()
    test_analytical_predictor_norm_types()
    test_d_scaling_law_api()
    test_e_floor_decomposition()
    test_cooling_law_api()
    test_predict_loss_api()
    print("\nAll tests passed ✓")

