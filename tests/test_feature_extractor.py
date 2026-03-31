#!/usr/bin/env python3
"""tests/test_feature_extractor.py - Verify hook-based feature extraction"""
import sys; sys.path.insert(0, 'src'); sys.path.insert(0, '.')
import torch

from experiments.phase_a.phase_a_analysis import (
    FeatureExtractor, build_TB3, build_TN5,
)

def test_feature_extractor():
    for name, builder in [("TB3", build_TB3), ("TN5", build_TN5)]:
        model = builder().eval()
        ext = FeatureExtractor(model)
        x = torch.randn(4, 3, 32, 32)
        feats = ext.run(x)
        assert len(feats) > 0, f"{name}: no features extracted"
        assert all(f.dim() == 2 for f in feats), f"{name}: features should all be 2D"
        print(f"  {name}: {len(feats)} features extracted, shapes={[f.shape for f in feats[:3]]}...")
        ext.remove()

if __name__ == "__main__":
    print("Testing feature extractor...")
    test_feature_extractor()
    print("\n✅ Feature extractor passed!")
