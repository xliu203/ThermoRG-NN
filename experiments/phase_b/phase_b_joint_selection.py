#!/usr/bin/env python3
"""
ThermoRG Phase B — Joint J_topo + Capacity Selection
===================================================

General architecture selection algorithm using theoretically-grounded constraints.

Key features (dataset-general, no empirical thresholds):
  1. J_topo > 0.5  (information flow filter)
  2. D_eff_total <= d_manifold * (log N + 1)  (capacity bound)

Reference: ThermoRG Theory Framework v5
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional


# ============================================================================
# 1. Dataset-General Parameters
# ============================================================================

def estimate_d_manifold(X: np.ndarray, method: str = "pca") -> float:
    """
    Estimate intrinsic dimension of data manifold using PCA.
    
    Args:
        X: Data array of shape (N, D)
        method: "pca" or "levina" (default: pca for speed)
    
    Returns:
        d_manifold: Estimated intrinsic dimension
    """
    from sklearn.decomposition import PCA
    
    # Use PCA to find dimension explaining 95% variance
    pca = PCA(n_components=0.95)
    pca.fit(X)
    d_manifold = float(pca.n_components_)
    return d_manifold


def compute_capacity_bound(d_manifold: float, n_samples: int) -> float:
    """
    Compute theory-grounded D_eff upper bound from covering numbers.
    
    Bound: D_eff <= d_manifold * (log n_samples + 1)
    
    Derived from:
      - Covering number theory: ε-covering of d-manifold needs ~(R/ε)^d cells
      - Setting ε ~ 1/√n_samples gives log(1/ε) = O(log n_samples)
    
    Args:
        d_manifold: Intrinsic data dimension
        n_samples: Number of training samples
    
    Returns:
        D_eff_max: Maximum D_eff to avoid overparameterization
    """
    import math
    return d_manifold * (math.log(n_samples) + 1)


# ============================================================================
# 2. J_topo Computation (from phase_a_dscaling.py)
# ============================================================================

def compute_D_eff_power_iteration(W: torch.Tensor, n_iter: int = 20) -> float:
    """
    Estimate D_eff = ||W||_F^2 / λ_max^2 via Power Iteration.
    ~23× faster than full SVD, ~2.5% D_eff error.
    """
    d = W.shape[0]
    if W.numel() == 0:
        return 1.0
    
    # Flatten for FC, keep (out, in, h, w) for conv
    W_flat = W.reshape(W.shape[0], -1)
    if min(W_flat.shape) == 0:
        return 1.0
    
    # Power iteration
    v = torch.randn(W_flat.shape[1], device=W_flat.device)
    v = v / (v.norm() + 1e-10)
    
    for _ in range(n_iter):
        # y = W^T W v
        Wv = torch.matmul(W_flat.T, torch.matmul(W_flat, v))
        v_new = Wv / (Wv.norm() + 1e-10)
        if torch.abs(v - v_new).sum() < 1e-8:
            break
        v = v_new
    
    # λ_max² ≈ ||W^T W v|| / v
    lambda_max_sq = torch.matmul(W_flat.T, torch.matmul(W_flat, v)).norm()**2 / (v.norm()**2 + 1e-10)
    
    # D_eff = ||W||_F² / λ_max²
    fro_sq = (W_flat ** 2).sum()
    D_eff = fro_sq / (lambda_max_sq + 1e-10)
    return float(D_eff.clamp(min=1.0))


def compute_D_eff_svd(W: torch.Tensor) -> float:
    """Full SVD D_eff (ground truth, slower)."""
    W_flat = W.reshape(W.shape[0], -1)
    if min(W_flat.shape) == 0:
        return 1.0
    try:
        _, s, _ = torch.svd(W_flat)
        fro_sq = (W_flat ** 2).sum()
        lambda_max_sq = s[0] ** 2
        return float((fro_sq / (lambda_max_sq + 1e-10)).clamp(min=1.0))
    except:
        return 1.0


def compute_J_topo(model: torch.nn.Module, 
                   skip_layers: Optional[List[str]] = None,
                   exclude_layers: Optional[List[str]] = None) -> Tuple[float, List[float]]:
    """
    Compute J_topo = exp(-mean|log η_l|) from initialized weights.
    
    Args:
        model: Neural network with initialized weights
        skip_layers: Layer names to skip (e.g., LayerNorm)
        exclude_layers: Layer names to treat as η=1 (e.g., LayerNorm)
    
    Returns:
        J_topo: Geometric mean of per-layer compression ratios
        eta_list: Per-layer η_l values
    """
    import re
    
    if skip_layers is None:
        skip_layers = []
    if exclude_layers is None:
        exclude_layers = []
    
    skip_patterns = [re.compile(p) for p in skip_layers]
    exclude_patterns = [re.compile(p) for p in exclude_layers]
    
    D_effs = []
    eta_list = []
    prev_D_eff = None
    
    for name, module in model.named_modules():
        # Skip patterns
        if any(p.search(name) for p in skip_patterns):
            continue
        
        # Check if it has weights
        if not hasattr(module, 'weight') or module.weight is None:
            continue
        
        W = module.weight.data
        
        # Handle LayerNorm: η = 1 (no compression)
        if any(p.search(name) for p in exclude_patterns):
            eta_list.append(1.0)
            continue
        
        # Compute D_eff
        D_eff = compute_D_eff_power_iteration(W, n_iter=20)
        D_effs.append(D_eff)
        
        if prev_D_eff is not None:
            eta = D_eff / max(prev_D_eff, 1.0)
            eta_list.append(float(eta))
        
        prev_D_eff = D_eff
    
    # J_topo = exp(-mean|log η_l|)
    if not eta_list:
        return 1.0, [1.0]
    
    log_etas = [abs(np.log(max(eta, 1e-10))) for eta in eta_list]
    J_topo = np.exp(-np.mean(log_etas))
    
    return float(J_toto), eta_list


def compute_D_eff_total(model: torch.nn.Module) -> float:
    """
    Compute total effective degrees of freedom.
    Sum of per-layer D_eff.
    """
    total = 0.0
    for _, module in model.named_modules():
        if not hasattr(module, 'weight') or module.weight is None:
            continue
        W = module.weight.data
        D_eff = compute_D_eff_power_iteration(W, n_iter=20)
        total += D_eff
    return float(total)


# ============================================================================
# 3. Architecture Evaluation
# ============================================================================

def evaluate_architecture(model: torch.nn.Module,
                         d_manifold: float,
                         n_samples: int,
                         skip_layers: Optional[List[str]] = None,
                         exclude_layers: Optional[List[str]] = None) -> Dict:
    """
    Evaluate a single architecture using the two-metric filter.
    
    Returns:
        dict with J_topo, D_eff_total, capacity_bound, passes_filter
    """
    J_topo, _ = compute_J_topo(model, skip_layers, exclude_layers)
    D_eff_total = compute_D_eff_total(model)
    capacity_bound = compute_capacity_bound(d_manifold, n_samples)
    
    passes_filter = bool(
        J_topo > 0.5 and
        D_eff_total <= capacity_bound
    )
    
    return {
        "J_topo": J_topo,
        "D_eff_total": D_eff_total,
        "capacity_bound": capacity_bound,
        "J_topo_passes": J_topo > 0.5,
        "capacity_passes": D_eff_total <= capacity_bound,
        "passes_filter": passes_filter,
        "J_topo_margin": J_topo - 0.5,
        "capacity_margin": capacity_bound - D_eff_total,
    }


def rank_architectures(candidates: List[Dict],
                       use_filter: bool = True,
                       secondary_metric: str = "J_topo") -> List[Dict]:
    """
    Rank architectures by J_topo (higher = better information flow).
    
    Args:
        candidates: List of evaluation results
        use_filter: If True, only return architectures passing the filter
        secondary_metric: Secondary sort key (for tiebreaking)
    
    Returns:
        Sorted list of candidate dicts
    """
    if use_filter:
        filtered = [c for c in candidates if c["passes_filter"]]
    else:
        filtered = candidates
    
    # Sort by J_topo descending (higher = better)
    filtered.sort(key=lambda x: (x["J_topo"], x.get(secondary_metric, 0)), reverse=True)
    
    return filtered


# ============================================================================
# 4. ThermoNet Candidate Configurations
# ============================================================================

def build_thermonet(depth: int, width_mult: float, use_skip: bool) -> torch.nn.Module:
    """
    Build a ThermoNet model.
    
    Args:
        depth: Number of layers
        width_mult: Width multiplier
        use_skip: Whether to use skip connections
    
    Returns:
        ThermoNet model
    """
    # This is a placeholder — actual implementation depends on your codebase
    raise NotImplementedError("Use your actual ThermoNet definition from phase_a_dscaling.py")


# ============================================================================
# 5. Main Experiment Pipeline
# ============================================================================

def run_phase_b_experiment(d_manifold: float,
                          n_samples: int,
                          candidate_configs: List[Dict],
                          use_filter: bool = True,
                          verbose: bool = True) -> List[Dict]:
    """
    Run the complete Phase B selection pipeline.
    
    Args:
        d_manifold: Intrinsic data dimension
        n_samples: Number of training samples
        candidate_configs: List of dicts with 'depth', 'width_mult', 'use_skip'
        use_filter: Whether to apply theory-based filtering
        verbose: Print progress
    
    Returns:
        Ranked list of evaluated candidates
    """
    import math
    
    capacity_bound = compute_capacity_bound(d_manifold, n_samples)
    
    if verbose:
        print(f"=== ThermoRG Phase B Selection ===")
        print(f"d_manifold: {d_manifold:.1f}")
        print(f"n_samples: {n_samples}")
        print(f"Capacity bound (D_eff_total): {capacity_bound:.1f}")
        print(f"J_topo threshold: 0.5")
        print(f"Filter enabled: {use_filter}")
        print()
    
    results = []
    
    for i, config in enumerate(candidate_configs):
        depth = config["depth"]
        wm = config["width_mult"]
        skip = config["skip"]
        
        try:
            model = build_thermonet(depth, wm, skip)
            eval_result = evaluate_architecture(
                model, d_manifold, n_samples,
                skip_layers=["ln", "norm"],
                exclude_layers=["ln", "norm"]
            )
            eval_result.update({
                "id": config.get("id", f"arch_{i}"),
                "depth": depth,
                "width_mult": wm,
                "use_skip": skip,
                "config": config,
            })
            results.append(eval_result)
            
            if verbose:
                status = "✓" if eval_result["passes_filter"] else "✗"
                print(f"{status} {eval_result['id']}: "
                      f"J_topo={eval_result['J_topo']:.4f}, "
                      f"D_eff={eval_result['D_eff_total']:.1f}/{capacity_bound:.1f}")
        
        except Exception as e:
            if verbose:
                print(f"✗ {config.get('id', f'arch_{i}')}: ERROR — {e}")
    
    # Rank
    ranked = rank_architectures(results, use_filter=use_filter)
    
    if verbose:
        print(f"\n=== Results ({len(ranked)} passing) ===")
        for i, r in enumerate(ranked[:10]):
            print(f"  {i+1}. {r['id']}: J_topo={r['J_topo']:.4f}, "
                  f"D_eff={r['D_eff_total']:.1f}")
    
    return ranked


if __name__ == "__main__":
    # CIFAR-10 parameters
    d_manifold_cifar10 = 50.0  # conservative PCA estimate
    n_samples_cifar10 = 50000
    
    capacity_bound = compute_capacity_bound(d_manifold_cifar10, n_samples_cifar10)
    print(f"CIFAR-10: d_manifold={d_manifold_cifar10}, n={n_samples_cifar10}")
    print(f"Capacity bound: D_eff_total <= {capacity_bound:.1f}")
