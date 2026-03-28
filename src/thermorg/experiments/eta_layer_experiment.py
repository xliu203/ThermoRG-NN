# SPDX-License-Identifier: Apache-2.0

"""Stage 1: Verify compression efficiency η_l ≤ 1

Generates complex nonlinear data, computes activation Jacobian for each layer,
and measures η_l to verify the compression efficiency bound.
"""

import time
import json
import torch
import sys
from pathlib import Path

sys.path.insert(0, 'src')

from thermorg.simulations.manifold_data import ManifoldDataGenerator
from thermorg.simulations.networks.mlp import MLP
from thermorg.utils.logging import setup_logger


def compute_activation_jacobian_per_layer(net, x, layer_idx):
    """Compute activation Jacobian ∂h_l/∂h_{l-1} for layer l.
    
    Args:
        net: MLP network
        x: Input tensor
        layer_idx: Layer index (0-indexed)
        
    Returns:
        Jacobian matrix of shape (hidden_dim, hidden_dim)
    """
    d_hidden = net.hidden_dim
    
    # Clone input and enable gradient
    x_clean = x.detach().clone().requires_grad_(True)
    
    # Forward pass to capture intermediate activations
    h_prev = x_clean
    
    # Register hooks to capture pre-activation values
    pre_activations = {}
    handles = []
    
    def make_hook(idx):
        def hook(module, input, output):
            # Capture input to this module (pre-activation)
            pre_activations[idx] = input[0].detach().requires_grad_(True)
        return hook
    
    # Attach hooks to capture h_{l-1} (input to layer l)
    linear_idx = 0
    for i, module in enumerate(net.network):
        if isinstance(module, torch.nn.Linear):
            handle = module.register_forward_hook(make_hook(linear_idx))
            handles.append(handle)
            linear_idx += 1
    
    # Run forward pass to capture activations
    with torch.no_grad():
        _ = net(x_clean)
    
    # Remove hooks
    for h in handles:
        h.remove()
    
    # Now compute Jacobian for the target layer
    # For layer_idx, we want ∂h_l/∂h_{l-1}
    # h_l = activation(W_l @ h_{l-1} + b_l)
    
    # Get the linear layer corresponding to layer_idx
    linear_idx = 0
    target_module = None
    target_pre_act = None
    
    for i, module in enumerate(net.network):
        if isinstance(module, torch.nn.Linear):
            if linear_idx == layer_idx:
                target_module = module
                target_pre_act = pre_activations[linear_idx]
                break
            linear_idx += 1
    
    if target_module is None:
        raise ValueError(f"Layer {layer_idx} not found")
    
    # Compute h_l given h_{l-1}
    h_l = target_module(target_pre_act)
    
    # Apply activation if present (ReLU, etc.)
    activation_fn = None
    next_idx = list(net.network).index(target_module) + 1
    if next_idx < len(net.network):
        next_module = list(net.network)[next_idx]
        if isinstance(next_module, torch.nn.ReLU):
            activation_fn = torch.nn.ReLU()
        elif isinstance(next_module, torch.nn.Tanh):
            activation_fn = torch.nn.Tanh()
    
    if activation_fn is not None:
        h_l = activation_fn(h_l)
    
    # Compute Jacobian ∂h_l/∂h_{l-1}
    d_out, d_in = h_l.shape[-1], target_pre_act.shape[-1]
    jacobian = torch.zeros(d_out, d_in, device=x.device, dtype=x.dtype)
    
    for i in range(d_out):
        grad = torch.zeros_like(h_l)
        grad[..., i] = 1.0
        (jac,) = torch.autograd.grad(outputs=h_l, inputs=target_pre_act, 
                                     grad_outputs=grad, retain_graph=True)
        jacobian[i] = jac.reshape(-1, d_in)
    
    return jacobian


def compute_jacobian_simple(net, x, layer_idx):
    """Compute activation Jacobian for layer_idx.
    
    Returns Jacobian of layer's post-activation output w.r.t. 
    its pre-activation input.
    """
    x = x.detach().requires_grad_(True)
    
    # Get all modules excluding the Sequential container
    modules_list = [m for m in net.network.modules()][1:]  # Skip Sequential itself
    
    # Find the linear layer at layer_idx
    # The network structure is: Linear, ReLU, Linear, ReLU, ...
    # layer_idx=0 corresponds to the first Linear layer
    # layer_idx=1 corresponds to the second Linear layer, etc.
    
    linear_count = 0
    linear_layer = None
    linear_layer_global_idx = None
    
    for i, module in enumerate(modules_list):
        if isinstance(module, torch.nn.Linear):
            if linear_count == layer_idx:
                linear_layer = module
                linear_layer_global_idx = i
                break
            linear_count += 1
    
    if linear_layer is None:
        raise ValueError(f"Layer {layer_idx} not found")
    
    # The activation after this linear layer is at linear_layer_global_idx + 1
    activation_layer = None
    if linear_layer_global_idx + 1 < len(modules_list):
        maybe_activation = modules_list[linear_layer_global_idx + 1]
        if not isinstance(maybe_activation, torch.nn.Linear):
            activation_layer = maybe_activation
    
    # Prepare input for the linear layer (use first sample)
    if layer_idx == 0:
        h_prev_sample = x[0].detach().requires_grad_(True)
    else:
        # Run forward pass to get input to this layer
        h = x
        for i, module in enumerate(modules_list):
            h = module(h)
            if i == linear_layer_global_idx:
                break
        h_prev_sample = h[0].detach().requires_grad_(True)
    
    # Compute output after linear layer
    h_linear = linear_layer(h_prev_sample)
    
    # Apply activation if present
    if activation_layer is not None:
        h_out = activation_layer(h_linear)
    else:
        h_out = h_linear
    
    d_out_layer = h_out.shape[-1]
    d_in_layer = h_prev_sample.shape[-1]
    
    # Compute Jacobian ∂h_out/∂h_prev_sample
    jacobian = torch.zeros(d_out_layer, d_in_layer, device=x.device, dtype=x.dtype)
    
    for i in range(d_out_layer):
        grad = torch.zeros(d_out_layer, device=x.device)
        grad[i] = 1.0
        (jac,) = torch.autograd.grad(outputs=h_out, inputs=h_prev_sample, 
                                     grad_outputs=grad, retain_graph=True)
        jacobian[i] = jac
    
    return jacobian


def main():
    logger = setup_logger("eta_layer_experiment")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    start_time = time.time()
    last_heartbeat = start_time
    
    # Configuration
    d_manifold = 10
    hidden_dim = 32
    n_layers = 4
    n_samples = 100
    
    logger.info("=" * 60)
    logger.info("STAGE 1: Compression Efficiency η_l < 1 Verification")
    logger.info("=" * 60)
    logger.info(f"Configuration: d_manifold={d_manifold}, hidden_dim={hidden_dim}, "
                f"n_layers={n_layers}, n_samples={n_samples}")
    
    # Generate complex nonlinear data
    logger.info("Generating complex nonlinear data...")
    gen = ManifoldDataGenerator(seed=42)
    X, y = gen.generate(n_samples=n_samples, d_manifold=d_manifold, 
                        d_embed=hidden_dim, mode='nonlinear')
    # Use y as target since it's embedded in the same space
    logger.info(f"Generated data: X.shape={X.shape}, y.shape={y.shape}")
    
    # Create network
    logger.info("Creating MLP network...")
    net = MLP(input_dim=d_manifold, hidden_dim=hidden_dim, 
              output_dim=d_manifold, n_layers=n_layers, activation='relu')
    net.eval()
    
    logger.info(f"Network structure: {net.network}")
    
    # Compute η_l for each layer
    eta_per_layer = []
    
    for layer_idx in range(n_layers):
        elapsed = time.time() - start_time
        elapsed_min = elapsed / 60
        
        # Heartbeat every 5 minutes
        if time.time() - last_heartbeat >= 300:
            logger.info(f"[HEARTBEAT {elapsed_min:.1f}min] Stage 1: Computing layer {layer_idx}/{n_layers}")
            last_heartbeat = time.time()
        
        logger.info(f"Computing Jacobian for layer {layer_idx}...")
        
        # Compute Jacobian
        jac = compute_jacobian_simple(net, X, layer_idx)
        
        # Compute η_l = D_eff / d_in = ||J||_F² / (||J||_2² × d_in)
        fro_sq = (jac ** 2).sum()
        spec_norm = jac.norm(2) ** 2 + 1e-8
        D_eff = fro_sq / spec_norm
        eta_l = D_eff / jac.shape[1]
        
        eta_per_layer.append(eta_l.item())
        logger.info(f"  Layer {layer_idx}: η_l = {eta_l:.4f} (D_eff = {D_eff:.2f}, "
                    f"||J||_F² = {fro_sq:.2f}, ||J||_2² = {spec_norm:.2f})")
    
    # Compute average η
    avg_eta = sum(eta_per_layer) / len(eta_per_layer)
    passed = all(e < 1.0 for e in eta_per_layer)
    
    logger.info("-" * 60)
    logger.info(f"Stage 1 Results:")
    logger.info(f"  η per layer: {[f'{e:.4f}' for e in eta_per_layer]}")
    logger.info(f"  Average η: {avg_eta:.4f}")
    logger.info(f"  All η_l < 1: {passed}")
    
    # Save progress
    stage1_results = {
        "stage": 1,
        "eta_per_layer": eta_per_layer,
        "avg_eta": avg_eta,
        "passed": passed,
        "config": {
            "d_manifold": d_manifold,
            "hidden_dim": hidden_dim,
            "n_layers": n_layers,
            "n_samples": n_samples
        }
    }
    
    progress_file = results_dir / "stage1_progress.json"
    with open(progress_file, 'w') as f:
        json.dump(stage1_results, f, indent=2)
    logger.info(f"Saved progress to {progress_file}")
    
    return stage1_results


if __name__ == "__main__":
    results = main()
    print(f"\nStage 1 Final: avg_eta={results['avg_eta']:.4f}, passed={results['passed']}")
