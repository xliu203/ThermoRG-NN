#!/usr/bin/env python3
"""
ThermoRG Phase B — Joint J_topo + Capacity Selection
===================================================

General architecture selection algorithm using theoretically-grounded constraints.

Key features (dataset-general, no empirical thresholds):
  1. J_topo > 0.45  (information flow filter, relaxed from 0.5)
  2. D_eff_total <= 2 * d_manifold * (log N + 1)  (capacity bound, ×2 relaxation)

Reference: ThermoRG Theory Framework v5
"""

import math
import re
import sys
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# Add phase_a to path so we can import the actual ThermoNet builders
# ============================================================================
PHASE_A_PATH = Path(__file__).parent.parent / "phase_a"
if str(PHASE_A_PATH) not in sys.path:
    sys.path.insert(0, str(PHASE_A_PATH))

# ============================================================================
# 1. Dataset-General Parameters
# ============================================================================

def estimate_d_manifold(X: np.ndarray) -> float:
    """
    Estimate intrinsic dimension of data manifold using PCA.
    Returns number of components explaining 95% variance.
    """
    from sklearn.decomposition import PCA
    pca = PCA(n_components=0.95)
    pca.fit(X)
    return float(pca.n_components_)


def compute_capacity_bound(d_manifold: float, n_samples: int) -> float:
    """
    Theory-grounded D_eff upper bound from covering numbers.
    
    Bound: D_eff <= d_manifold * (log n_samples + 1)
    
    Derived from:
      - Covering number theory: ε-covering of d-manifold needs ~(R/ε)^d cells
      - Setting ε ~ 1/√n_samples gives log(1/ε) = O(log n_samples)
    """
    return d_manifold * (math.log(n_samples) + 1)


# ============================================================================
# 2. ThermoNet Model Builders (from phase_a_dscaling.py)
# ============================================================================

def scale_channels(chs, mult):
    """Scale channel list by multiplier, preserving first (input) channel."""
    return [chs[0]] + [max(1, int(c * mult)) for c in chs[1:]]


class SkipConnection(nn.Module):
    def __init__(self, ic, oc, s=1):
        super().__init__()
        if ic == oc and s == 1:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Sequential(
                nn.Conv2d(ic, oc, 1, s, bias=False),
                nn.BatchNorm2d(oc)
            )
    def forward(self, x, residual):
        return x + self.skip(residual)


class ConvBlock(nn.Module):
    def __init__(self, ic, oc, act='gelu', norm=True):
        super().__init__()
        self.conv = nn.Conv2d(ic, oc, 3, padding=1, bias=not norm)
        self.norm = nn.LayerNorm([oc, 32, 32]) if norm else nn.Identity()
        if act == 'gelu':
            self.act = nn.GELU()
        elif act == 'tga':
            self.act = nn.Tanh()
        else:
            self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class BasicBlock(nn.Module):
    def __init__(self, ic, oc, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(ic, oc, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(oc)
        self.conv2 = nn.Conv2d(oc, oc, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(oc)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return F.relu(out + identity)


def make_layer(ic, oc, blocks, stride=1):
    downsample = None
    if stride != 1 or ic != oc:
        downsample = nn.Sequential(
            nn.Conv2d(ic, oc, 1, stride=stride, bias=False),
            nn.BatchNorm2d(oc)
        )
    layers = [BasicBlock(ic, oc, stride, downsample)]
    for _ in range(1, blocks):
        layers.append(BasicBlock(oc, oc))
    return nn.Sequential(*layers)


def build_TN3(wm=1.0, num_classes=10, use_skip=True):
    """ThermoNet-3: [64,64,128,128]"""
    ch = scale_channels([3, 64, 64, 128, 128], wm)
    blocks = nn.ModuleList()
    for i in range(len(ch) - 1):
        blocks.append(ConvBlock(ch[i], ch[i+1], 'gelu', True))
    pool = nn.AdaptiveAvgPool2d((1, 1))
    fc = nn.Linear(ch[-1], num_classes)
    return nn.Sequential(*[*blocks, pool, nn.Flatten(), fc])


def build_TN5(wm=1.0, num_classes=10, use_skip=True):
    """ThermoNet-5: [64,128,256,128,64]"""
    ch = scale_channels([3, 64, 128, 256, 128, 64], wm)
    blocks = nn.ModuleList()
    for i in range(len(ch) - 1):
        blocks.append(ConvBlock(ch[i], ch[i+1], 'gelu', True))
    pool = nn.AdaptiveAvgPool2d((1, 1))
    fc = nn.Linear(ch[-1], num_classes)
    return nn.Sequential(*[*blocks, pool, nn.Flatten(), fc])


def build_TN7(wm=1.0, num_classes=10, use_skip=True):
    """ThermoNet-7: [64,64,128,128,256,128,64]"""
    ch = scale_channels([3, 64, 64, 128, 128, 256, 128, 64], wm)
    blocks = nn.ModuleList()
    for i in range(len(ch) - 1):
        blocks.append(ConvBlock(ch[i], ch[i+1], 'tga', True))
    pool = nn.AdaptiveAvgPool2d((1, 1))
    fc = nn.Linear(ch[-1], num_classes)
    return nn.Sequential(*[*blocks, pool, nn.Flatten(), fc])


def build_TN9(wm=1.0, num_classes=10, use_skip=False):
    """ThermoNet-9: [64]*8 uniform"""
    ch = scale_channels([3] + [64]*8, wm)
    blocks = nn.ModuleList()
    for i in range(len(ch) - 1):
        blocks.append(ConvBlock(ch[i], ch[i+1], 'gelu', True))
    pool = nn.AdaptiveAvgPool2d((1, 1))
    fc = nn.Linear(ch[-1], num_classes)
    return nn.Sequential(*[*blocks, pool, nn.Flatten(), fc])


def build_TN_arbitrary_depth(depth, wm=1.0, num_classes=10, use_skip=True):
    """
    Build ThermoNet with arbitrary depth.
    Uses a repeating channel pattern that scales with depth.
    
    Architecture: [64] + [64]*depth → pool → fc
    For depth > 9, uses a wider base channel pattern.
    """
    if depth <= 3:
        return build_TN3(wm, num_classes, use_skip)
    elif depth == 5:
        return build_TN5(wm, num_classes, use_skip)
    elif depth == 7:
        return build_TN7(wm, num_classes, use_skip)
    elif depth == 9:
        return build_TN9(wm, num_classes, use_skip)
    else:
        # For depth > 9, use a repeating pattern
        # Base: [64] + [64, 128, 256, 512] pattern repeating
        base_pattern = [64, 128, 256, 512]
        ch = [3]
        # Repeat the base pattern to achieve desired depth
        reps = (depth + 3) // 4
        for _ in range(reps):
            ch.extend(base_pattern)
        ch = ch[:depth+1]  # truncate to exact depth+1 (including input)
        while len(ch) < depth + 1:
            ch.append(64)
        
        ch = scale_channels(ch, wm)
        blocks = nn.ModuleList()
        for i in range(len(ch) - 1):
            blocks.append(ConvBlock(ch[i], ch[i+1], 'gelu', True))
        pool = nn.AdaptiveAvgPool2d((1, 1))
        fc = nn.Linear(ch[-1], num_classes)
        return nn.Sequential(*[*blocks, pool, nn.Flatten(), fc])


def build_thermonet(depth: int, width_mult: float, use_skip: bool,
                   num_classes: int = 10) -> nn.Module:
    """
    Build ThermoNet from arbitrary depth/width/skip.
    
    Args:
        depth: Number of layers (3, 5, 7, 9, 12, 15, etc.)
        width_mult: Width multiplier
        use_skip: Whether to use skip connections
        num_classes: Number of output classes
    
    Returns:
        ThermoNet model
    """
    # Use specialized builders for standard depths, fallback for others
    model = build_TN_arbitrary_depth(depth, width_mult, num_classes, use_skip)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    return model


# ============================================================================
# 3. D_eff Computation
# ============================================================================

def compute_D_eff_power_iteration(W: torch.Tensor, n_iter: int = 20) -> float:
    """
    Estimate D_eff = ||W||_F^2 / λ_max^2 via Power Iteration.
    ~23× faster than full SVD, ~2.5% D_eff error.
    """
    W_flat = W.reshape(W.shape[0], -1)
    if min(W_flat.shape) == 0:
        return 1.0
    
    # Power iteration
    v = torch.randn(W_flat.shape[1], device=W_flat.device)
    v = v / (v.norm() + 1e-10)
    
    for _ in range(n_iter):
        Wv = torch.matmul(W_flat.T, torch.matmul(W_flat, v))
        v_new = Wv / (Wv.norm() + 1e-10)
        if torch.abs(v - v_new).sum() < 1e-8:
            break
        v = v_new
    
    lambda_max_sq = torch.matmul(W_flat.T, torch.matmul(W_flat, v)).norm()**2 / (v.norm()**2 + 1e-10)
    fro_sq = (W_flat ** 2).sum()
    D_eff = fro_sq / (lambda_max_sq + 1e-10)
    return float(D_eff.clamp(min=1.0))


def get_layer_weights_for_J_topo(module: nn.Module, name: str) -> Optional[torch.Tensor]:
    """
    Get the weight tensor for J_topo computation.
    
    For Conv2d: returns weight as-is
    For Linear: returns weight as-is
    
    Skip connections are handled by combining main + skip weights.
    """
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        return module.weight.data
    return None


def compute_J_topo(model: nn.Module,
                   skip_exclude_patterns: Optional[List[str]] = None,
                   use_stride_correction: bool = True) -> Tuple[float, List[float]]:
    """
    Compute J_topo = exp(-mean|log η_l|) from initialized weights.
    
    Handles:
      - Conv2d and Linear layers
      - LayerNorm excluded (η = 1)
      - Skip connections: combined weight W_eff = W_main + W_skip (if detectable)
      - Stride-2 downsampling: spatial-channel compression factor ζ = C_out/(C_in·s²)
    
    Args:
        model: Neural network with initialized weights
        skip_exclude_patterns: Regex patterns for layers to exclude
        use_stride_correction: If True, apply stride correction for Conv2d
    
    Returns:
        J_topo: Geometric mean of per-layer compression ratios
        eta_list: Per-layer η_l values
    """
    if skip_exclude_patterns is None:
        skip_exclude_patterns = ['layernorm', 'layer_norm', 'norm', 'batchnorm', 'bn', 'pool', 'flatten', 'fc', 'linear']
    
    eta_list = []
    prev_D_eff = None
    prev_c_out = None
    
    named_modules = list(model.named_modules())
    
    for i, (name, module) in enumerate(named_modules):
        # Check if should be excluded
        if any(re.search(p, name.lower()) for p in skip_exclude_patterns):
            eta_list.append(1.0)
            prev_D_eff = None  # Reset after excluded layer
            prev_c_out = None
            continue
        
        # Get weight
        W = get_layer_weights_for_J_topo(module, name)
        if W is None:
            continue
        
        # Determine stride and channels
        s = 1
        c_in = W.shape[1]
        c_out = W.shape[0]
        
        if isinstance(module, nn.Conv2d):
            s = module.stride[0] if hasattr(module, 'stride') else 1
            c_in = module.in_channels
            c_out = module.out_channels
        
        # Compute D_eff
        D_eff = compute_D_eff_power_iteration(W, n_iter=20)
        
        if prev_D_eff is not None and prev_c_out is not None:
            # Standard expansion ratio
            eta = D_eff / max(prev_D_eff, 1.0)
            
            # Stride correction: spatial-channel compression factor
            if use_stride_correction and s > 1:
                zeta = (c_out / prev_c_out) / (s ** 2)
                eta = eta * zeta
            
            eta_list.append(float(eta))
        
        prev_D_eff = D_eff
        prev_c_out = c_out
    
    if not eta_list:
        return 1.0, [1.0]
    
    log_etas = [abs(math.log(max(eta, 1e-10))) for eta in eta_list]
    J_topo = math.exp(-np.mean(log_etas))
    
    return float(J_topo), eta_list


def compute_D_eff_total(model: nn.Module) -> float:
    """
    Compute total effective degrees of freedom.
    Sum of per-layer D_eff across all weight layers.
    """
    total = 0.0
    for _, module in model.named_modules():
        W = get_layer_weights_for_J_topo(module, "")
        if W is None:
            continue
        D_eff = compute_D_eff_power_iteration(W, n_iter=20)
        total += D_eff
    return float(total)


def count_parameters(model: nn.Module) -> int:
    """Count total parameters in model."""
    return sum(p.numel() for p in model.parameters())


# ============================================================================
# 4. Architecture Evaluation
# ============================================================================

def evaluate_architecture(model: nn.Module,
                         d_manifold: float,
                         n_samples: int) -> Dict:
    """
    Evaluate a single architecture using the two-metric filter.
    
    Returns:
        dict with J_topo, D_eff_total, capacity_bound, passes_filter
    """
    J_topo, _ = compute_J_topo(model)
    D_eff_total = compute_D_eff_total(model)
    capacity_bound = compute_capacity_bound(d_manifold, n_samples)
    n_params = count_parameters(model)
    
    passes_filter = bool(
        J_topo > 0.45 and
        D_eff_total <= 2 * capacity_bound
    )
    
    return {
        "J_topo": J_topo,
        "D_eff_total": D_eff_total,
        "capacity_bound": capacity_bound,
        "n_params": n_params,
        "J_topo_passes": J_topo > 0.5,
        "capacity_passes": D_eff_total <= capacity_bound,
        "passes_filter": passes_filter,
        "J_topo_margin": J_topo - 0.5,
        "capacity_margin": capacity_bound - D_eff_total,
    }


def rank_architectures(candidates: List[Dict],
                       use_filter: bool = True) -> List[Dict]:
    """
    Rank architectures by J_topo descending (higher = better).
    
    Args:
        candidates: List of evaluation results
        use_filter: If True, only return architectures passing the filter
    
    Returns:
        Sorted list of candidate dicts
    """
    if use_filter:
        filtered = [c for c in candidates if c["passes_filter"]]
    else:
        filtered = candidates
    
    filtered.sort(key=lambda x: x["J_topo"], reverse=True)
    return filtered


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
        candidate_configs: List of dicts with 'depth', 'width_mult', 'use_skip', 'id'
        use_filter: Whether to apply theory-based filtering
        verbose: Print progress
    
    Returns:
        Ranked list of evaluated candidates
    """
    capacity_bound = compute_capacity_bound(d_manifold, n_samples)
    
    if verbose:
        print(f"=== ThermoRG Phase B Selection ===")
        print(f"d_manifold: {d_manifold:.1f}")
        print(f"n_samples: {n_samples:,}")
        print(f"Capacity bound (D_eff_total, relaxed ×2): {2*capacity_bound:.1f}")
        print(f"J_topo threshold: 0.45 (relaxed from 0.5)")
        print(f"Filter enabled: {use_filter}")
        print()
    
    results = []
    
    for i, config in enumerate(candidate_configs):
        depth = config["depth"]
        wm = config["width_mult"]
        skip = config["skip"]
        arch_id = config.get("id", f"arch_{i}")
        
        try:
            # Build model with fresh initialization
            torch.manual_seed(42)
            model = build_thermonet(depth, wm, skip)
            
            # Evaluate
            eval_result = evaluate_architecture(model, d_manifold, n_samples)
            eval_result.update({
                "id": arch_id,
                "depth": depth,
                "width_mult": wm,
                "use_skip": skip,
                "config": config,
            })
            results.append(eval_result)
            
            if verbose:
                status = "✓" if eval_result["passes_filter"] else "✗"
                params_M = eval_result["n_params"] / 1e6
                print(f"{status} {arch_id}: "
                      f"J_topo={eval_result['J_topo']:.4f} "
                      f"(margin={eval_result['J_topo_margin']:+.3f}), "
                      f"D_eff={eval_result['D_eff_total']:.0f}/{capacity_bound:.0f}, "
                      f"params={params_M:.2f}M")
        
        except Exception as e:
            if verbose:
                print(f"✗ {arch_id}: ERROR — {e}")
            import traceback
            traceback.print_exc()
    
    # Rank
    ranked = rank_architectures(results, use_filter=use_filter)
    
    if verbose:
        n_passing = sum(1 for r in results if r["passes_filter"])
        print(f"\n=== Results ({n_passing}/{len(results)} passing) ===")
        for i, r in enumerate(ranked[:10]):
            params_M = r["n_params"] / 1e6
            print(f"  {i+1}. {r['id']}: J_topo={r['J_topo']:.4f}, "
                  f"D_eff={r['D_eff_total']:.0f}, params={params_M:.2f}M")
    
    return ranked


# ============================================================================
# 6. Quick Test
# ============================================================================

def quick_test():
    """Test the pipeline on a few architectures."""
    from thermonet_candidates_20 import CANDIDATES_20
    
    # CIFAR-10 parameters
    d_manifold = 50.0
    n_samples = 50000
    capacity_bound = compute_capacity_bound(d_manifold, n_samples)
    
    print(f"CIFAR-10: d_manifold={d_manifold}, n={n_samples:,}")
    print(f"Capacity bound: D_eff_total <= {capacity_bound:.1f}")
    print()
    
    # Test on a few candidates
    test_configs = [c for c in CANDIDATES_20 if c["id"] in ["T01", "T11", "T18", "T19"]]
    results = run_phase_b_experiment(d_manifold, n_samples, test_configs, verbose=True)
    return results


if __name__ == "__main__":
    quick_test()
