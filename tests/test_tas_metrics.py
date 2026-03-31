#!/usr/bin/env python3
"""tests/test_tas_metrics.py - Verify TAS alpha computation"""
import sys; sys.path.insert(0, 'src'); sys.path.insert(0, '.')
import torch

from experiments.phase_a.phase_a_analysis import (
    compute_alpha_from_features, FeatureExtractor,
    build_TN3, build_ResNet18,
)

def test_alpha_computation():
    for name, builder in [("TN3", build_TN3), ("ResNet18", build_ResNet18)]:
        model = builder().eval()
        ext = FeatureExtractor(model)
        x = torch.randn(8, 3, 32, 32)
        feats = ext.run(x)
        metrics = compute_alpha_from_features(feats)
        assert metrics is not None, f"{name}: metrics should not be None"
        assert 'alpha' in metrics, f"{name}: missing alpha key"
        assert 'J_topo' in metrics, f"{name}: missing J_topo key"
        assert metrics['num_layers'] == len(feats), f"{name}: num_layers mismatch"
        print(f"  {name}: alpha={metrics['alpha']:.4f}, J_topo={metrics['J_topo']:.4f}, layers={metrics['num_layers']}")
        ext.remove()

if __name__ == "__main__":
    print("Testing TAS metrics...")
    test_alpha_computation()
    print("\n✅ TAS metrics passed!")
