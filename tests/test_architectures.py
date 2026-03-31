#!/usr/bin/env python3
"""tests/test_architectures.py - Verify all 15 architectures produce correct output shapes"""
import sys; sys.path.insert(0, 'src'); sys.path.insert(0, '.')
import torch

from experiments.phase_a.phase_a_analysis import (
    ARCHITECTURES, generic_forward, build_TN3, build_TN5, build_TN7, build_TN9,
    build_TB3, build_TB5, build_TB7, build_TB9,
    build_RF3, build_RF5, build_RF7, build_RF9,
    build_ResNet18, build_VGG11, build_DenseNet40,
)

def test_all_architectures():
    x = torch.randn(4, 3, 32, 32)
    for name, (builder, group) in ARCHITECTURES.items():
        model = builder().eval()
        out = generic_forward(model, x)
        assert out.shape == (4, 10), f"{name}: expected (4, 10), got {out.shape}"
        print(f"  {name}: OK → {out.shape}")

if __name__ == "__main__":
    print("Testing all 15 architectures...")
    test_all_architectures()
    print("\n✅ All architectures passed!")
