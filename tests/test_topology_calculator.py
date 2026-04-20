# SPDX-License-Identifier: Apache-2.0
"""Unit tests for topology_calculator (J_topo computation)."""

import torch
import torch.nn as nn
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from thermorg.topology_calculator import (
    compute_J_topo,
    compute_D_eff_power_iteration,
    compute_resblock_eff_W,
)


class SimpleConvNet(nn.Module):
    """Simple conv net for testing."""
    def __init__(self, width=64, depth=3):
        super().__init__()
        layers = []
        c_in = 3
        for i in range(depth):
            layers.append(nn.Conv2d(c_in, width, 3, padding=1))
            layers.append(nn.BatchNorm2d(width))
            c_in = width
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(width, 10)

    def forward(self, x):
        return self.fc(self.pool(self.features(x)).flatten(1))


def test_compute_D_eff_power_iteration():
    """D_eff should be in range [1, C_out]."""
    torch.manual_seed(42)
    W = torch.randn(64, 32)  # C_out=64, C_in=32
    d_eff = compute_D_eff_power_iteration(W, n_iter=20)
    assert 1.0 <= d_eff <= 64.0, f"D_eff={d_eff} out of expected range [1, 64]"
    print(f"  D_eff={d_eff:.2f} (expected [1, 64]) ✓")


def test_compute_D_eff_reproducible():
    """D_eff should be reproducible with fixed seed."""
    torch.manual_seed(42)
    W = torch.randn(64, 32)
    d1 = compute_D_eff_power_iteration(W, n_iter=20)
    d2 = compute_D_eff_power_iteration(W, n_iter=20)
    assert abs(d1 - d2) < 1e-5, "D_eff should be reproducible with fixed seed"
    print(f"  Reproducible: d1={d1:.4f}, d2={d2:.4f} ✓")


def test_J_topo_in_range():
    """J_topo should be in (0, 1]."""
    torch.manual_seed(42)
    model = SimpleConvNet(width=32, depth=3)
    J, etas = compute_J_topo(model)
    assert 0.0 < J <= 1.0, f"J_topo={J} out of expected range (0, 1]"
    assert len(etas) >= 3, f"Expected at least 3 eta values, got {len(etas)}"
    print(f"  J_topo={J:.4f}, etas={[f'{e:.3f}' for e in etas]} ✓")


def test_J_topo_width_dependence():
    """Wider networks should have lower J_topo (GELU saturation effect)."""
    torch.manual_seed(42)
    Js = []
    for width in [16, 32, 64]:
        model = SimpleConvNet(width=width, depth=3)
        J, _ = compute_J_topo(model)
        Js.append(J)
    # J should decrease with width (GELU saturation, α_GELU ≈ 0.45)
    assert Js[0] > Js[1] > Js[2], f"J_topo should decrease with width: {Js}"
    print(f"  Width 16→32→64: J={[f'{j:.4f}' for j in Js]} (decreasing ✓)")


def test_J_topo_depth_dependence():
    """Deeper networks should have higher J_topo (more layers → less variance)."""
    torch.manual_seed(42)
    Js = []
    for depth in [3, 5, 7]:
        model = SimpleConvNet(width=32, depth=depth)
        J, _ = compute_J_topo(model)
        Js.append(J)
    assert Js[0] < Js[1] < Js[2], f"J_topo should increase with depth: {Js}"
    print(f"  Depth 3→5→7: J={[f'{j:.4f}' for j in Js]} (increasing ✓)")


def test_J_topo_zero_cost():
    """J_topo should be zero-cost (no training, no loss)."""
    import time
    torch.manual_seed(42)
    model = SimpleConvNet(width=64, depth=5)
    
    start = time.time()
    J, etas = compute_J_topo(model)
    elapsed = time.time() - start
    
    # Should complete in well under 1 second for small network
    assert elapsed < 1.0, f"compute_J_topo took {elapsed:.3f}s (> 1s)"
    print(f"  Zero-cost: J={J:.4f} in {elapsed*1000:.1f}ms ✓")


if __name__ == '__main__':
    print("=== test_topology_calculator ===")
    test_compute_D_eff_power_iteration()
    test_compute_D_eff_reproducible()
    test_J_topo_in_range()
    test_J_topo_width_dependence()
    test_J_topo_depth_dependence()
    test_J_topo_zero_cost()
    print("\nAll tests passed ✓")
