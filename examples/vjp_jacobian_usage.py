#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0

"""Example script demonstrating memory-efficient VJP Jacobian estimation.

This script shows how to use the VJP-based estimators to compute D_eff
without materializing the full Jacobian, which would OOM on GPUs like T4.

Run with: python examples/vjp_jacobian_usage.py
"""

import torch
import torch.nn as nn
from torch import Tensor

from thermorg.core.jacobian import (
    trace_JTJ_vjp,
    spectral_norm_vjp,
    compute_d_eff_vjp,
    VJPJacobianEstimator,
    compute_jacobian_naive,
)


# =============================================================================
# Example Models
# =============================================================================

class LargeMLP(nn.Module):
    """Large MLP that would OOM with naive Jacobian computation."""
    
    def __init__(self, input_dim=2048, hidden_dim=1024, output_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ConvNetLarge(nn.Module):
    """Large conv net for image inputs."""
    
    def __init__(self, in_channels=3, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# =============================================================================
# Memory Comparison
# =============================================================================

def compare_memory_usage():
    """Compare memory usage between naive and VJP approaches."""
    print("=" * 70)
    print("MEMORY COMPARISON: Naive vs VJP Jacobian Estimation")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Model dimensions that would OOM with naive on T4
    input_dim = 2048
    hidden_dim = 1024
    output_dim = 512
    
    model = LargeMLP(input_dim, hidden_dim, output_dim).to(device)
    x = torch.randn(8, input_dim, device=device)
    
    print(f"\nModel: LargeMLP({input_dim}, {hidden_dim}, {output_dim})")
    print(f"Input shape: {x.shape}")
    print(f"Jacobian size if naive: {output_dim} x {input_dim} = {output_dim * input_dim:,} elements")
    
    # Memory estimates
    naive_memory = output_dim * input_dim * 4  # float32
    vjp_footprint = 4 * x.numel() * 4  # 4 tensors of input size
    
    print(f"\nEstimated memory:")
    print(f"  Naive Jacobian: {naive_memory / 1024 / 1024:.1f} MB")
    print(f"  VJP-based:      {vjp_footprint / 1024 / 1024:.1f} MB")
    print(f"  Savings:         {naive_memory / vjp_footprint:.1f}x")
    
    # Actually measure VJP memory
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        
        d_eff = compute_d_eff_vjp(model, x, n_trace_samples=50, n_power_iter=25)
        
        peak = torch.cuda.max_memory_allocated()
        used = peak - before
        print(f"\nActual VJP peak memory: {used / 1024 / 1024:.1f} MB")
        print(f"D_eff = {d_eff.item():.2f}")
    else:
        d_eff = compute_d_eff_vjp(model, x, n_trace_samples=50, n_power_iter=25)
        print(f"\nD_eff = {d_eff.item():.2f}")


# =============================================================================
# Accuracy vs Sample Count
# =============================================================================

def accuracy_vs_samples():
    """Show how accuracy improves with more samples."""
    print("\n" + "=" * 70)
    print("ACCURACY vs SAMPLE COUNT")
    print("=" * 70)
    
    # Small model for ground truth comparison
    small_model = LargeMLP(input_dim=32, hidden_dim=64, output_dim=16)
    x = torch.randn(2, 32)
    
    # Compute ground truth with naive method
    print("\nComputing ground truth (naive Jacobian)...")
    J = compute_jacobian_naive(small_model, x)
    trace_true = (J ** 2).sum()
    _, s, _ = torch.linalg.svd(J, full_matrices=False)
    spec_norm_true = s[0]
    d_eff_true = trace_true / (spec_norm_true ** 2)
    
    print(f"Ground truth:")
    print(f"  trace(J^T J) = {trace_true.item():.4f}")
    print(f"  spectral norm = {spec_norm_true.item():.4f}")
    print(f"  D_eff = {d_eff_true.item():.4f}")
    
    # Compare with different sample counts
    print("\nVJP estimates with different sample counts:")
    print("-" * 50)
    print(f"{'Samples':<10} {'trace':<12} {'spec_norm':<12} {'D_eff':<10} {'rel_err':<10}")
    print("-" * 50)
    
    for n_samples in [10, 50, 100, 200, 500]:
        trace_est = trace_JTJ_vjp(small_model, x, n_samples=n_samples)
        spec_est = spectral_norm_vjp(small_model, x, n_iter=n_samples)
        d_eff_est = trace_est / (spec_est ** 2)
        
        trace_err = abs(trace_est - trace_true) / trace_true
        spec_err = abs(spec_est - spec_norm_true) / spec_norm_true
        d_eff_err = abs(d_eff_est - d_eff_true) / d_eff_true
        
        print(f"{n_samples:<10} {trace_est.item():<12.4f} {spec_est.item():<12.4f} "
              f"{d_eff_est.item():<10.4f} {d_eff_err.item():<10.4f}")


# =============================================================================
# Class-based Usage
# =============================================================================

def class_based_usage():
    """Demonstrate the VJPJacobianEstimator class."""
    print("\n" + "=" * 70)
    print("CLASS-BASED USAGE: VJPJacobianEstimator")
    print("=" * 70)
    
    # Create estimator with desired precision
    estimator = VJPJacobianEstimator(
        n_trace_samples=100,
        n_power_iter=50,
    )
    
    # Large model
    model = LargeMLP(input_dim=1024, hidden_dim=512, output_dim=256)
    x = torch.randn(4, 1024)
    
    print(f"\nModel: LargeMLP(1024, 512, 256)")
    print(f"Input shape: {x.shape}")
    
    # Single call to get all estimates
    print("\nEstimating all Jacobian properties...")
    results = estimator.estimate_all(model, x)
    
    print("\nResults:")
    print(f"  trace(J^T J)    = {results['trace']:.4f}")
    print(f"  spectral norm   = {results['spectral_norm']:.4f}")
    print(f"  D_eff           = {results['d_eff']:.4f}")
    
    # Memory footprint
    footprint = estimator.memory_footprint(x)
    print(f"\nMemory footprint:")
    print(f"  VJP-based:      {footprint['estimated_total_bytes'] / 1024 / 1024:.2f} MB")
    print(f"  Naive would be: {footprint['equivalent_naive_bytes'] / 1024 / 1024:.2f} MB")


# =============================================================================
# ConvNet Example
# =============================================================================

def convnet_example():
    """Show usage with convolutional network."""
    print("\n" + "=" * 70)
    print("CONVNET EXAMPLE")
    print("=" * 70)
    
    model = ConvNetLarge(in_channels=3, num_classes=100)
    x = torch.randn(2, 3, 32, 32)
    
    print(f"\nModel: ConvNetLarge")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {model(x).shape}")
    
    estimator = VJPJacobianEstimator(n_trace_samples=50, n_power_iter=30)
    results = estimator.estimate_all(model, x)
    
    print(f"\nResults:")
    print(f"  trace(J^T J)    = {results['trace']:.4f}")
    print(f"  spectral norm   = {results['spectral_norm']:.4f}")
    print(f"  D_eff           = {results['d_eff']:.4f}")


# =============================================================================
# Timing Comparison
# =============================================================================

def timing_comparison():
    """Compare timing between naive and VJP."""
    print("\n" + "=" * 70)
    print("TIMING COMPARISON")
    print("=" * 70)
    
    # Small enough for naive, large enough to measure
    model = LargeMLP(input_dim=128, hidden_dim=256, output_dim=64)
    x = torch.randn(4, 128)
    
    import time
    
    # Naive timing
    print("\nTiming naive Jacobian (may be slow for larger models)...")
    start = time.time()
    J = compute_jacobian_naive(model, x)
    naive_time = time.time() - start
    print(f"Naive: {naive_time:.4f} seconds")
    
    # VJP timing with different sample counts
    print("\nTiming VJP with different sample counts...")
    for n_samples in [10, 50, 100, 200]:
        start = time.time()
        trace = trace_JTJ_vjp(model, x, n_samples=n_samples)
        spec_norm = spectral_norm_vjp(model, x, n_iter=n_samples)
        vjp_time = time.time() - start
        print(f"  n_samples={n_samples}: {vjp_time:.4f} seconds, D_eff={trace / spec_norm**2:.2f}")
    
    # Note: VJP is slower per-call but avoids OOM and is parallelizable


# =============================================================================
# Main
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("VJP-BASED JACOBIAN ESTIMATION EXAMPLES")
    print("=" * 70)
    print("""
This example demonstrates memory-efficient Jacobian property estimation
using Vector-Jacobian Products (VJPs) instead of materializing the full
Jacobian matrix.

Key benefits:
- Memory: O(batch_size × feature_dim) instead of O(N × M)
- Scalable: Works on GPUs with limited memory (T4, etc.)
- Flexible: Trade off accuracy vs speed with sample count
""")
    
    compare_memory_usage()
    accuracy_vs_samples()
    class_based_usage()
    convnet_example()
    timing_comparison()
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
