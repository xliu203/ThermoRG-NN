#!/usr/bin/env python3
"""
Compute intrinsic dimensionality (d_manifold) of CIFAR-10 dataset.

Key insight: CIFAR-10 raw dimension is 3×32×32 = 3072,
but the TRUE intrinsic dimension (degrees of freedom that distinguish 10 classes)
is much smaller — probably in the range of 50-200.

This script uses:
1. PCA-based variance thresholding (d_95 = dimensions needed for 95% variance)
2. Levina-Bickel MLE (maximum likelihood local neighborhood estimation)
3. Correlation dimension (Grassberger-Procaccia)
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import sys, os

# ── Add TAS package to path ───────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from thermorg.core.manifold_estimator import (
    estimate_d_manifold_pca,
    estimate_d_manifold_levina,
)
try:
    from thermorg.core.manifold_estimator import estimate_d_manifold_correlation
    HAS_CORRELATION = True
except ImportError:
    HAS_CORRELATION = False

# ── Load CIFAR-10 ─────────────────────────────────────────────────────────────
def load_cifar10(n_samples=1000, batch_size=256):
    """Load CIFAR-10 and flatten images to vectors.
    
    NOTE: On Kaggle, run this with GPU. On local Mac CPU, n_samples=500 is
    fast enough for PCA but Levina-Bickel may be slower.
    """
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    
    # Sample subset for faster computation
    indices = torch.randperm(len(trainset))[:n_samples]
    subset = torch.utils.data.Subset(trainset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
    
    all_images = []
    all_labels = []
    for images, labels in loader:
        # Flatten: (B, 3, 32, 32) → (B, 3072)
        all_images.append(images.view(images.size(0), -1))
        all_labels.append(labels)
    
    X = torch.cat(all_images, dim=0)  # (n_samples, 3072)
    y = torch.cat(all_labels, dim=0)  # (n_samples,)
    
    return X, y

# ── Per-class d_manifold ──────────────────────────────────────────────────────
def compute_per_class_dimensions(X, y, n_classes=10):
    """Compute d_manifold separately for each class."""
    results = {}
    for c in range(n_classes):
        mask = (y == c)
        X_c = X[mask]
        if len(X_c) < 100:
            continue
        
        try:
            d_pca_95 = estimate_d_manifold_pca(X_c, variance_threshold=0.95)
            d_pca_99 = estimate_d_manifold_pca(X_c, variance_threshold=0.99)
            d_levina = estimate_d_manifold_levina(X_c, k_neighbors=min(20, len(X_c)//2))
            
            results[f"class_{c}"] = {
                "n_samples": len(X_c),
                "d_pca_95": d_pca_95,
                "d_pca_99": d_pca_99,
                "d_levina": d_levina,
            }
        except Exception as e:
            print(f"  Warning: class {c} failed: {e}")
    
    return results

# ── Full dataset d_manifold ───────────────────────────────────────────────────
def compute_full_dataset_dimensions(X):
    """Compute d_manifold for the entire dataset."""
    results = {}
    try:
        results["d_pca_90"] = estimate_d_manifold_pca(X, variance_threshold=0.90)
        results["d_pca_95"] = estimate_d_manifold_pca(X, variance_threshold=0.95)
        results["d_pca_99"] = estimate_d_manifold_pca(X, variance_threshold=0.99)
        results["d_levina_10"] = estimate_d_manifold_levina(X, k_neighbors=10)
        results["d_levina_20"] = estimate_d_manifold_levina(X, k_neighbors=20)
        results["d_levina_50"] = estimate_d_manifold_levina(X, k_neighbors=50)
    except Exception as e:
        print(f"Error computing full dataset dimensions: {e}")
    
    return results

# ── Class-separability dimension ──────────────────────────────────────────────
def compute_separability_dimension(X, y, n_classes=10):
    """
    The dimension that MATTERS for classification:
    NOT the raw dimension of data, but the dimension of the subspace
    that SEPARATES the classes.
    
    Approximate: PCA on class centroids difference vectors.
    """
    centroids = []
    for c in range(n_classes):
        mask = (y == c)
        centroids.append(X[mask].mean(dim=0))
    
    centroids = torch.stack(centroids, dim=0)  # (10, 3072)
    
    # Compute covariance of centroids (how do class means vary?)
    centroid_centered = centroids - centroids.mean(dim=0)
    cov = torch.cov(centroid_centered.T)  # (3072, 3072)
    
    eigenvalues = torch.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues.flip(0)  # Descending
    
    # Variance explained by top components
    var_ratio = eigenvalues / eigenvalues.sum()
    cumsum = var_ratio.cumsum(dim=0)
    
    d_90 = (cumsum >= 0.90).nonzero()[0][0].item() + 1
    d_95 = (cumsum >= 0.95).nonzero()[0][0].item() + 1
    d_99 = (cumsum >= 0.99).nonzero()[0][0].item() + 1
    
    return {
        "d_separable_90": d_90,
        "d_separable_95": d_95,
        "d_separable_99": d_99,
        "top_eigenvalues": eigenvalues[:10].cpu().numpy().tolist(),
    }

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    
    print("=" * 60)
    print(f"CIFAR-10 Intrinsic Dimension Analysis (n={n_samples})")
    print("=" * 60)
    print(f"\nRaw dimension: 3 × 32 × 32 = 3072")
    print(f"Num classes: 10")
    print()
    
    print("Loading CIFAR-10...")
    X, y = load_cifar10(n_samples=n_samples)
    print(f"Loaded: X.shape = {X.shape}, y.shape = {y.shape}")
    
    # 1. Full dataset
    print("\n[1] Full dataset intrinsic dimension")
    print("-" * 40)
    full_results = compute_full_dataset_dimensions(X)
    for k, v in full_results.items():
        print(f"  {k}: {v:.1f}")
    
    # 2. Per-class
    print("\n[2] Per-class intrinsic dimension (Levina-Bickel)")
    print("-" * 40)
    per_class = compute_per_class_dimensions(X, y)
    levina_values = []
    pca95_values = []
    for cls, vals in sorted(per_class.items()):
        print(f"  {cls:12s}: n={vals['n_samples']:5d} | PCA_95={vals['d_pca_95']:5.1f} | Levina={vals['d_levina']:5.1f}")
        levina_values.append(vals['d_levina'])
        pca95_values.append(vals['d_pca_95'])
    
    avg_levina = np.mean(levina_values)
    avg_pca95 = np.mean(pca95_values)
    print(f"\n  Average per-class: PCA_95={avg_pca95:.1f}, Levina={avg_levina:.1f}")
    
    # 3. Class-separability dimension
    print("\n[3] Class-separability dimension (dimension of centroid differences)")
    print("-" * 40)
    sep_results = compute_separability_dimension(X, y)
    for k, v in sep_results.items():
        if k != "top_eigenvalues":
            print(f"  {k}: {v:.1f}")
    print(f"  Top eigenvalues: {[f'{e:.2f}' for e in sep_results['top_eigenvalues'][:5]]}")
    
    # 4. Summary
    print("\n" + "=" * 60)
    print("SUMMARY: CIFAR-10 Intrinsic Dimensionality")
    print("=" * 60)
    print(f"  Raw dimension:                         {X.shape[1]}")
    print(f"  PCA 95% variance (full dataset):       {full_results.get('d_pca_95', 'N/A')}")
    print(f"  Levina-Bickel (k=20, full dataset):    {full_results.get('d_levina_20', 'N/A')}")
    print(f"  Average per-class Levina-Bickel:       {avg_levina:.1f}")
    print(f"  Average per-class PCA 95%:             {avg_pca95:.1f}")
    print(f"  Class-separability dimension (95%):     {sep_results['d_separable_95']}")
    print()
    print("  → The manifold dimension is FAR smaller than 3072!")
    print(f"  → ~{avg_levina:.0f}-{avg_pca95:.0f} intrinsic dimensions for full data")
    print(f"  → Only ~{sep_results['d_separable_95']} dimensions separate 10 classes")
    print()
    print("  TAS implication: If d_manifold ≈ 50-200, then")
    print("  BatchNorm compression (η_l < 1) may NOT hurt accuracy")
    print("  because most of the 3072D space is irrelevant noise.")

if __name__ == "__main__":
    main()
