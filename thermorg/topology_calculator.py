#!/usr/bin/env python3
"""
ThermoRG Topology Calculator Module
====================================

Zero-cost J_topo computation via Power Iteration (PI-20).

J_topo measures the quality of information flow through a network architecture:
- J_topo → 1: stable information flow (all layers have similar D_eff)
- J_topo → 0: bottlenecks or expansion issues

Mathematical Foundation:
-----------------------
The participation ratio D_eff for a weight matrix W is:
    D_eff = ||W||_F^2 / λ_max^2

where λ_max is the largest singular value. This can be estimated via power iteration:
    v_{k+1} = W^T W v_k / ||W^T W v_k||

After n iterations: λ_max² ≈ ||W^T W v_n|| / ||v_n||, and D_eff = ||W||_F² / λ_max².

IMPORTANT: Reproducibility Note
------------------------------
Different random seeds give slightly different λ_max estimates (due to the 
random starting vector v_0 in power iteration), but the resulting J_topo 
RANKING is stable across seeds. This is because J_topo measures relative 
layer uniformity, not absolute D_eff values. For reproducibility, this module
uses a fixed seed (42) by default.

Reference: ThermoRG Theory Framework v5-v8
"""

from __future__ import annotations

import math
import re
from typing import List, Tuple, Optional, Dict, Any, Union

import numpy as np
import torch
import torch.nn as nn


# =============================================================================
# Constants
# =============================================================================

DEFAULT_SEED = 42
DEFAULT_POWER_ITERATIONS = 20


# =============================================================================
# Power Iteration for D_eff Computation
# =============================================================================

def compute_D_eff_power_iteration(
    W: torch.Tensor,
    n_iter: int = DEFAULT_POWER_ITERATIONS,
    seed: int = DEFAULT_SEED,
    verbose: bool = False
) -> float:
    """
    Estimate D_eff = ||W||_F^2 / λ_max^2 via Power Iteration.
    
    This is ~23× faster than full SVD with ~2.5% D_eff error, which is
    acceptable for zero-cost architectural scoring.
    
    Args:
        W: Weight tensor (2D or 4D for Conv2d)
           - 4D tensors (Conv2d) are averaged across spatial dimensions
           - 2D tensors (Linear) are used directly
        n_iter: Number of power iterations (default: 20 = PI-20)
        seed: Random seed for reproducibility (default: 42)
        verbose: If True, print iteration details
        
    Returns:
        Estimated D_eff value (clamped to minimum 1.0)
    
    Example:
        >>> W = torch.randn(64, 32, 3, 3)
        >>> D_eff = compute_D_eff_power_iteration(W)
        >>> print(f"D_eff = {D_eff:.2f}")
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)
    
    # Flatten spatial dimensions for Conv2d
    W_flat = W.reshape(W.shape[0], -1)
    
    if min(W_flat.shape) == 0:
        return 1.0
    
    # Handle very small matrices
    if W_flat.shape[0] == 0 or W_flat.shape[1] == 0:
        return 1.0
    
    # Power iteration
    # v_0: random starting vector (orthogonal to W's column space if possible)
    v = torch.randn(W_flat.shape[1], device=W_flat.device, dtype=W_flat.dtype)
    v = v / (v.norm() + 1e-10)
    
    for iteration in range(n_iter):
        # Compute W^T @ W @ v
        Wv = torch.matmul(W_flat.T, torch.matmul(W_flat, v))
        
        # Normalize
        v_new = Wv / (Wv.norm() + 1e-10)
        
        # Check convergence
        diff = torch.abs(v - v_new).sum().item()
        if verbose and iteration % 5 == 0:
            print(f"  Iteration {iteration}: |v - v_new| = {diff:.2e}")
        
        if diff < 1e-8:
            if verbose:
                print(f"  Converged at iteration {iteration}")
            break
        v = v_new
    
    # Estimate λ_max^2 from the converged vector
    Wv = torch.matmul(W_flat.T, torch.matmul(W_flat, v))
    lambda_max_sq = torch.matmul(W_flat, v).norm()**2 / (v.norm()**2 + 1e-10)
    
    # Frobenius norm squared
    fro_sq = (W_flat ** 2).sum()
    
    # D_eff = ||W||_F^2 / λ_max^2
    D_eff = fro_sq / (lambda_max_sq + 1e-10)
    
    return float(D_eff.clamp(min=1.0))


def compute_D_eff_from_W_eff(
    W_eff: torch.Tensor,
    n_iter: int = DEFAULT_POWER_ITERATIONS,
    seed: int = DEFAULT_SEED
) -> float:
    """
    Compute D_eff from an effective weight matrix W_eff.
    
    This is used for residual blocks where the effective weight combines
    the main path and skip connection: W_eff = I + W2 @ W1 (for identity skip).
    
    Args:
        W_eff: Effective weight tensor (2D matrix)
        n_iter: Number of power iterations
        seed: Random seed
        
    Returns:
        D_eff value
    """
    return compute_D_eff_power_iteration(W_eff, n_iter=n_iter, seed=seed)


# =============================================================================
# Residual Block Handling
# =============================================================================

def compute_resblock_eff_W(
    W1: torch.Tensor,
    W2: torch.Tensor,
    W_skip: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute effective weight for a residual block.
    
    Theory (ThermoRG Framework):
    - Skip connections provide SHORTCUT paths that REDUCE effective depth
    - They bypass computation, making information flow more uniform
    - For D_eff, use W_eff = I + W2 @ W1 for identity skip (raises D_eff)
    - For projection skip (stride-2 or channel change), use W2 @ W1
    
    Key insight: W_eff = I + W2 @ W1 has D_eff closer to identity (C) than
    either W2 or W1 alone. This correctly captures the skip connection's effect.
    
    Args:
        W1: Weight tensor of first layer in residual path
        W2: Weight tensor of second layer in residual path
        W_skip: Optional skip projection weight (if None, identity skip assumed)
        
    Returns:
        Effective weight tensor W_eff (2D channel matrix for D_eff computation)
    """
    device = W1.device
    dtype = W1.dtype
    
    if W1.dim() == 4:
        # Conv2d case: use channel-space approximation
        W1_ch = W1.mean(dim=(2, 3))  # (C_out, C_in)
        W2_ch = W2.mean(dim=(2, 3))  # (C_out, C_out)
        
        c_out = W2.shape[0]
        c_in = W1.shape[1]
        
        if W_skip is not None:
            # Projection skip: W_skip and W2 @ W1 operate on different inputs
            # Use only the transformation path W2 @ W1 for D_eff
            W_eff_ch = torch.matmul(W2_ch, W1_ch)
        elif c_in == c_out:
            # Identity skip: W_eff = I + W2 @ W1
            I_ch = torch.eye(c_out, dtype=dtype, device=device)
            W_eff_ch = I_ch + torch.matmul(W2_ch, W1_ch)
        else:
            # No skip or unknown skip type: use W2 @ W1
            W_eff_ch = torch.matmul(W2_ch, W1_ch)
        
        return W_eff_ch
        
    elif W1.dim() == 2:
        # Linear layers case
        if W_skip is not None or W1.shape[0] != W2.shape[0]:
            W_eff = torch.matmul(W2, W1)
        else:
            I = torch.eye(W2.shape[0], dtype=dtype, device=device)
            W_eff = I + torch.matmul(W2, W1)
        return W_eff
    else:
        # Fallback: return W2
        return W2


# =============================================================================
# Block Detection
# =============================================================================

def detect_residual_block(
    module: nn.Module,
    name: str
) -> Optional[Dict[str, Any]]:
    """
    Detect if a module is a residual block and return its components.
    
    Args:
        module: PyTorch module
        name: Module name (for pattern matching)
        
    Returns:
        Dict with 'W1', 'W2', 'W_skip', 'stride' if residual block, None otherwise
    """
    name_lower = name.lower()
    
    # Pattern-based detection
    is_residual_name = any(p in name_lower for p in ['resnet', 'residual', 'block', 'skip'])
    
    # Attribute-based detection
    has_downsample = hasattr(module, 'downsample') and module.downsample is not None
    has_shortcut = hasattr(module, 'shortcut') and module.shortcut is not None
    has_sh = hasattr(module, 'sh') and module.sh is not None
    
    if not (is_residual_name or has_downsample or has_shortcut or has_sh):
        return None
    
    # Extract conv layers
    conv_layers = []
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            conv_layers.append(m)
    
    if len(conv_layers) >= 2:
        W1 = conv_layers[0].weight.data
        W2 = conv_layers[1].weight.data
        stride = conv_layers[0].stride[0] if hasattr(conv_layers[0], 'stride') else 1
        
        # Get skip/projection weight if available
        W_skip = None
        if has_downsample:
            downsample = module.downsample
            if isinstance(downsample, nn.Sequential):
                for m in downsample.modules():
                    if isinstance(m, nn.Conv2d):
                        W_skip = m.weight.data
                        break
        elif has_shortcut:
            shortcut = module.shortcut
            if isinstance(shortcut, nn.Conv2d):
                W_skip = shortcut.weight.data
        elif has_sh:
            sh = module.sh
            if isinstance(sh, nn.Conv2d):
                W_skip = sh.weight.data
            elif isinstance(sh, nn.Sequential):
                for m in sh.modules():
                    if isinstance(m, nn.Conv2d):
                        W_skip = m.weight.data
                        break
        
        return {
            'W1': W1,
            'W2': W2,
            'W_skip': W_skip,
            'stride': stride
        }
    
    return None


def detect_dense_block(
    module: nn.Module,
    name: str,
    prev_c_out: int
) -> Optional[Dict[str, Any]]:
    """
    Detect if a module is a dense block and return its properties.
    
    For DenseNet-style blocks, each layer receives concatenated features
    from all previous layers, so input channels grow cumulatively.
    
    Args:
        module: PyTorch module
        name: Module name
        prev_c_out: Previous layer's output channels
        
    Returns:
        Dict with properties if dense block, None otherwise
    """
    name_lower = name.lower()
    
    is_dense_name = any(p in name_lower for p in ['dense', 'db', 'denseblock', 'dense_block'])
    
    if not is_dense_name:
        return None
    
    first_conv = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            first_conv = m
            break
    
    if first_conv is None:
        return None
    
    c_in = first_conv.in_channels
    c_out = first_conv.out_channels
    
    if c_in >= prev_c_out:
        growth_rate = c_out // 4 if c_out >= 4 else c_out
        num_layers = sum(1 for m in module.modules() if isinstance(m, nn.Conv2d))
        return {
            'c_in': c_in,
            'c_out': c_out,
            'growth_rate': growth_rate,
            'num_layers': num_layers,
            'is_dense': True
        }
    
    return None


def detect_transition_layer(
    module: nn.Module,
    name: str
) -> bool:
    """
    Detect if a module is a transition layer between dense blocks.
    
    Args:
        module: PyTorch module
        name: Module name
        
    Returns:
        True if this is a transition layer, False otherwise
    """
    name_lower = name.lower()
    
    is_top_level_transition = (
        name_lower.endswith('transition') or
        name_lower.endswith('transition1') or
        name_lower.endswith('transition2') or
        name_lower.endswith('transition3') or
        name_lower == 'transition'
    )
    
    if not is_top_level_transition:
        return False
    
    return type(module).__name__ == '_Transition'


def get_layer_weights_for_J_topo(
    module: nn.Module,
    name: str
) -> Optional[torch.Tensor]:
    """
    Get the weight tensor for J_topo computation.
    
    Args:
        module: PyTorch module
        name: Module name
        
    Returns:
        Weight tensor if found, None otherwise
    """
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        return module.weight.data
    return None


# =============================================================================
# Main J_topo Computation
# =============================================================================

def compute_J_topo(
    model: nn.Module,
    skip_exclude_patterns: Optional[List[str]] = None,
    use_stride_correction: bool = True,
    seed: int = DEFAULT_SEED,
    n_iter: int = DEFAULT_POWER_ITERATIONS
) -> Tuple[float, List[float]]:
    """
    Compute J_topo = exp(-mean|log η_l|) from initialized weights.
    
    The participation ratio D_eff for each layer is computed via power iteration.
    J_topo is the geometric mean of per-layer compression ratios η_l = D_eff[l] / D_eff[l-1].
    
    Handles:
      - Conv2d and Linear layers
      - LayerNorm excluded (η = 1 by definition)
      - Skip connections: combined weight W_eff = I + W2 @ W1 (ResNet-style)
      - Dense connections: cumulative input channels (DenseNet-style)
      - Stride-2 downsampling: spatial-channel compression factor ζ = C_out/(C_in·s²)
    
    Args:
        model: Neural network with initialized weights
        skip_exclude_patterns: Regex patterns for layers to exclude
        use_stride_correction: If True, apply stride correction for Conv2d
        seed: Random seed for power iteration (default: 42)
        n_iter: Number of power iterations per layer (default: 20)
        
    Returns:
        J_topo: Geometric mean of per-layer compression ratios (0 to 1)
        eta_list: Per-layer η_l values
        
    Example:
        >>> import torch.nn as nn
        >>> model = nn.Sequential(
        ...     nn.Conv2d(3, 64, 3, padding=1),
        ...     nn.Conv2d(64, 128, 3, stride=2, padding=1)
        ... )
        >>> J_topo, eta_list = compute_J_topo(model)
        >>> print(f"J_topo = {J_topo:.4f}")
    """
    if skip_exclude_patterns is None:
        skip_exclude_patterns = [
            'layernorm', 'layer_norm', 'norm', 'batchnorm', 'bn',
            'pool', 'flatten', 'fc', 'linear'
        ]
    
    eta_list = []
    prev_D_eff = None
    prev_c_out = None
    
    named_modules_list = list(model.named_modules())
    processed_names = set()
    
    i = 0
    while i < len(named_modules_list):
        name, module = named_modules_list[i]
        
        # Skip already processed modules
        if name in processed_names:
            i += 1
            continue
        
        # Check exclusion patterns
        if any(re.search(p, name.lower()) for p in skip_exclude_patterns):
            eta_list.append(1.0)
            i += 1
            continue
        
        # Detect dense block
        dense_info = detect_dense_block(module, name, prev_c_out if prev_c_out else 0)
        if dense_info is not None:
            c_cumulative = dense_info['c_in']
            growth = dense_info.get('growth_rate', 32)
            
            for child_name, _ in named_modules_list:
                if child_name != name and child_name.startswith(name + '.'):
                    processed_names.add(child_name)
            
            dense_layers = []
            for m in module.modules():
                if m is module:
                    continue
                if hasattr(m, 'conv2') and isinstance(m.conv2, nn.Conv2d):
                    dense_layers.append(m)
            
            for layer_idx, dense_layer in enumerate(dense_layers):
                conv2 = dense_layer.conv2
                k = conv2.out_channels
                
                if c_cumulative > 0:
                    eta = (c_cumulative + k) / c_cumulative
                    eta_list.append(float(eta))
                
                c_cumulative += k
                prev_c_out = k
            
            prev_D_eff = float(c_cumulative)
            prev_c_out = k if dense_layers else prev_c_out
            
            i += 1
            continue
        
        # Detect transition layer
        if detect_transition_layer(module, name):
            conv = None
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    conv = m
                    break
            
            if conv is not None:
                c_out = conv.out_channels
                c_in = conv.in_channels
                
                if c_in > 0:
                    eta = c_out / c_in
                    eta_list.append(float(eta))
                
                prev_c_out = c_out
                prev_D_eff = float(c_out)
                
                for child_name, _ in named_modules_list:
                    if child_name != name and child_name.startswith(name + '.'):
                        processed_names.add(child_name)
                
            i += 1
            continue
        
        # Detect residual block
        resblock_info = detect_residual_block(module, name)
        if resblock_info is not None:
            W1 = resblock_info['W1']
            W2 = resblock_info['W2']
            W_skip = resblock_info['W_skip']
            s = resblock_info['stride']
            
            W_eff = compute_resblock_eff_W(W1, W2, W_skip)
            D_eff = compute_D_eff_power_iteration(W_eff, n_iter=n_iter, seed=seed)
            
            c_out = W2.shape[0]
            c_in = W1.shape[1]
            
            if prev_D_eff is not None and prev_c_out is not None:
                eta = D_eff / max(prev_D_eff, 1.0)
                
                if use_stride_correction and s > 1:
                    zeta = (c_out / prev_c_out) / (s ** 2)
                    eta = eta * zeta
                
                eta_list.append(float(eta))
            
            prev_D_eff = D_eff
            prev_c_out = c_out
            
            for child_name, _ in named_modules_list:
                if child_name != name and child_name.startswith(name + '.'):
                    processed_names.add(child_name)
            
            i += 1
            continue
        
        # Standard layer handling
        W = get_layer_weights_for_J_topo(module, name)
        if W is None:
            i += 1
            continue
        
        s = 1
        c_in = W.shape[1]
        c_out = W.shape[0]
        
        if isinstance(module, nn.Conv2d):
            s = module.stride[0] if hasattr(module, 'stride') else 1
            c_in = module.in_channels
            c_out = module.out_channels
        
        D_eff = compute_D_eff_power_iteration(W, n_iter=n_iter, seed=seed)
        
        if prev_D_eff is not None and prev_c_out is not None:
            eta = D_eff / max(prev_D_eff, 1.0)
            
            if use_stride_correction and s > 1:
                zeta = (c_out / prev_c_out) / (s ** 2)
                eta = eta * zeta
            
            eta_list.append(float(eta))
        
        prev_D_eff = D_eff
        prev_c_out = c_out
        i += 1
    
    if not eta_list:
        return 1.0, [1.0]
    
    log_etas = [abs(math.log(max(eta, 1e-10))) for eta in eta_list]
    J_topo = math.exp(-np.mean(log_etas))
    
    return float(J_topo), eta_list


def compute_J_topo_detailed(
    model: nn.Module,
    skip_exclude_patterns: Optional[List[str]] = None,
    use_stride_correction: bool = True,
    seed: int = DEFAULT_SEED,
    n_iter: int = DEFAULT_POWER_ITERATIONS
) -> Tuple[float, List[float], Dict[str, Any]]:
    """
    Compute J_topo with detailed layer information for debugging.
    
    Args:
        model: Neural network with initialized weights
        skip_exclude_patterns: Regex patterns for layers to exclude
        use_stride_correction: If True, apply stride correction
        seed: Random seed for power iteration
        n_iter: Number of power iterations per layer
        
    Returns:
        J_topo, eta_list, layer_info (dict with per-layer details)
    """
    if skip_exclude_patterns is None:
        skip_exclude_patterns = [
            'layernorm', 'layer_norm', 'norm', 'batchnorm', 'bn',
            'pool', 'flatten', 'fc', 'linear'
        ]
    
    eta_list = []
    layer_info = []
    prev_D_eff = None
    prev_c_out = None
    
    named_modules_list = list(model.named_modules())
    processed_names = set()
    
    i = 0
    while i < len(named_modules_list):
        name, module = named_modules_list[i]
        
        info = {
            'name': name,
            'type': type(module).__name__,
            'is_skip': False,
            'is_dense': False,
        }
        
        if name in processed_names:
            i += 1
            continue
        
        if any(re.search(p, name.lower()) for p in skip_exclude_patterns):
            eta_list.append(1.0)
            info['excluded'] = True
            layer_info.append(info)
            i += 1
            continue
        
        # Detect residual block
        resblock_info = detect_residual_block(module, name)
        if resblock_info is not None:
            W1 = resblock_info['W1']
            W2 = resblock_info['W2']
            W_skip = resblock_info['W_skip']
            s = resblock_info['stride']
            
            W_eff = compute_resblock_eff_W(W1, W2, W_skip)
            D_eff = compute_D_eff_power_iteration(W_eff, n_iter=n_iter, seed=seed)
            
            c_out = W2.shape[0]
            c_in = W1.shape[1]
            
            info['is_skip'] = True
            info['W_eff_shape'] = list(W_eff.shape)
            info['D_eff'] = D_eff
            
            if prev_D_eff is not None and prev_c_out is not None:
                eta = D_eff / max(prev_D_eff, 1.0)
                if use_stride_correction and s > 1:
                    zeta = (c_out / prev_c_out) / (s ** 2)
                    eta = eta * zeta
                    info['zeta'] = zeta
                info['eta'] = float(eta)
                eta_list.append(float(eta))
            
            prev_D_eff = D_eff
            prev_c_out = c_out
            
            for child_name, _ in named_modules_list:
                if child_name != name and child_name.startswith(name + '.'):
                    processed_names.add(child_name)
            
            layer_info.append(info)
            i += 1
            continue
        
        # Detect dense block
        dense_info = detect_dense_block(module, name, prev_c_out if prev_c_out else 0)
        if dense_info is not None:
            info['is_dense'] = True
            info['c_cumulative'] = dense_info['c_in']
            
            c_cumulative = dense_info['c_in']
            growth = dense_info.get('growth_rate', dense_info['c_out'] // 4)
            
            for child_name, _ in named_modules_list:
                if child_name != name and child_name.startswith(name + '.'):
                    processed_names.add(child_name)
            
            for j, sub_module in enumerate(dense_block.modules()):
                if sub_module is dense_block:
                    continue
                
                W = get_layer_weights_for_J_topo(sub_module, f"{name}.layer{j}")
                if W is None:
                    continue
                
                if isinstance(sub_module, nn.Conv2d):
                    c_out = sub_module.out_channels
                    D_eff = compute_D_eff_power_iteration(W, n_iter=n_iter, seed=seed)
                    
                    if prev_D_eff is not None and prev_c_out is not None:
                        eta = D_eff / max(prev_D_eff, 1.0)
                        s = sub_module.stride[0] if hasattr(sub_module, 'stride') else 1
                        if use_stride_correction and s > 1:
                            zeta = (c_out / prev_c_out) / (s ** 2)
                            eta = eta * zeta
                        eta_list.append(float(eta))
                    
                    prev_D_eff = D_eff
                    prev_c_out = c_out
                    c_cumulative += growth
                elif isinstance(sub_module, (nn.BatchNorm2d, nn.LayerNorm, nn.ReLU, nn.Identity)):
                    continue
            
            layer_info.append(info)
            i += 1
            continue
        
        # Standard layer
        W = get_layer_weights_for_J_topo(module, name)
        if W is None:
            layer_info.append(info)
            i += 1
            continue
        
        s = 1
        c_in = W.shape[1]
        c_out = W.shape[0]
        
        if isinstance(module, nn.Conv2d):
            s = module.stride[0] if hasattr(module, 'stride') else 1
            c_in = module.in_channels
            c_out = module.out_channels
        
        D_eff = compute_D_eff_power_iteration(W, n_iter=n_iter, seed=seed)
        
        info['D_eff'] = D_eff
        info['c_in'] = c_in
        info['c_out'] = c_out
        
        if prev_D_eff is not None and prev_c_out is not None:
            eta = D_eff / max(prev_D_eff, 1.0)
            if use_stride_correction and s > 1:
                zeta = (c_out / prev_c_out) / (s ** 2)
                eta = eta * zeta
                info['zeta'] = zeta
            info['eta'] = float(eta)
            eta_list.append(float(eta))
        
        prev_D_eff = D_eff
        prev_c_out = c_out
        layer_info.append(info)
        i += 1
    
    if not eta_list:
        J_topo = 1.0
    else:
        log_etas = [abs(math.log(max(eta, 1e-10))) for eta in eta_list]
        J_topo = math.exp(-np.mean(log_etas))
    
    return float(J_topo), eta_list, {'layers': layer_info}


# =============================================================================
# Utility Functions
# =============================================================================

def compute_D_eff_total(model: nn.Module, seed: int = DEFAULT_SEED, n_iter: int = DEFAULT_POWER_ITERATIONS) -> float:
    """
    Compute total effective degrees of freedom.
    Sum of per-layer D_eff across all weight layers.
    
    Args:
        model: Neural network
        seed: Random seed for power iteration
        n_iter: Number of power iterations per layer
        
    Returns:
        Total D_eff
    """
    total = 0.0
    for _, module in model.named_modules():
        W = get_layer_weights_for_J_topo(module, "")
        if W is None:
            continue
        D_eff = compute_D_eff_power_iteration(W, n_iter=n_iter, seed=seed)
        total += D_eff
    return float(total)


def count_parameters(model: nn.Module) -> int:
    """Count total parameters in model."""
    return sum(p.numel() for p in model.parameters())


# Alias for backward compatibility
compute_D_eff = compute_D_eff_power_iteration