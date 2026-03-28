# SPDX-License-Identifier: Apache-2.0

"""Stage 2: Verify α vs ∏η_l relationship

Scans d_manifold = [5, 10, 15, 20, 25], computes total compression efficiency
η_total = ∏η_l, and verifies α ∝ η_total / d_manifold.
"""

import time
import json
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, 'src')

from thermorg.simulations.manifold_data import ManifoldDataGenerator
from thermorg.simulations.networks.mlp import MLP
from thermorg.utils.logging import setup_logger


def compute_layer_jacobian(net, x, layer_idx):
    """Compute activation Jacobian for layer_idx.
    
    Returns Jacobian of layer's post-activation output w.r.t. 
    its pre-activation input.
    """
    x = x.detach().requires_grad_(True)
    
    # Get all modules excluding the Sequential container
    modules_list = [m for m in net.network.modules()][1:]
    
    # Find the linear layer at layer_idx
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
    
    # The activation after this linear layer
    activation_layer = None
    if linear_layer_global_idx + 1 < len(modules_list):
        maybe_activation = modules_list[linear_layer_global_idx + 1]
        if not isinstance(maybe_activation, torch.nn.Linear):
            activation_layer = maybe_activation
    
    # Prepare input for the linear layer
    if layer_idx == 0:
        h_prev_sample = x[0].detach().requires_grad_(True)
    else:
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
    
    # Compute Jacobian
    jacobian = torch.zeros(d_out_layer, d_in_layer, device=x.device, dtype=x.dtype)
    
    for i in range(d_out_layer):
        grad = torch.zeros(d_out_layer, device=x.device)
        grad[i] = 1.0
        (jac,) = torch.autograd.grad(outputs=h_out, inputs=h_prev_sample, 
                                     grad_outputs=grad, retain_graph=True)
        jacobian[i] = jac
    
    return jacobian


def compute_eta_for_network(net, X, n_layers):
    """Compute compression efficiency for each layer."""
    eta_values = []
    for layer_idx in range(n_layers):
        jac = compute_layer_jacobian(net, X, layer_idx)
        fro_sq = (jac ** 2).sum()
        spec_norm = jac.norm(2) ** 2
        D_eff = fro_sq / (spec_norm + 1e-8)
        eta_l = D_eff / jac.shape[1]
        eta_values.append(eta_l.item())
    return eta_values


def compute_alpha_measured(net, X, d_manifold, n_layers):
    """Compute measured α via scaling law fitting.
    
    α = slope of log(error) vs log(scale)
    We vary the network scale and measure error.
    """
    errors = []
    scales = []
    
    # Vary hidden_dim to change scale
    for hidden_dim in [16, 32, 64]:
        net_test = MLP(input_dim=d_manifold, hidden_dim=hidden_dim,
                       output_dim=d_manifold, n_layers=n_layers, activation='relu')
        net_test.eval()
        
        with torch.no_grad():
            output = net_test(X)
            # Use reconstruction error (X is the target since output_dim = d_manifold)
            error = torch.mean((output - X) ** 2).item()
        
        scales.append(hidden_dim)
        errors.append(error)
    
    # Fit α: error ∝ scale^(-α) => log(error) = log(C) - α*log(scale)
    log_errors = np.log(np.array(errors))
    log_scales = np.log(np.array(scales))
    
    # Linear regression
    n = len(log_scales)
    mean_x = np.mean(log_scales)
    mean_y = np.mean(log_errors)
    
    numerator = np.sum((log_scales - mean_x) * (log_errors - mean_y))
    denominator = np.sum((log_scales - mean_x) ** 2)
    
    alpha = -numerator / denominator if denominator != 0 else 0
    
    return alpha, errors, scales


def main():
    logger = setup_logger("scaling_verification")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    start_time = time.time()
    last_heartbeat = start_time
    
    # Configuration
    d_manifold_list = [5, 10, 15, 20, 25]
    hidden_dim = 32
    n_layers = 4
    n_samples = 100
    
    logger.info("=" * 60)
    logger.info("STAGE 2: α vs ∏η_l Relationship Verification")
    logger.info("=" * 60)
    logger.info(f"Configuration: d_manifold_list={d_manifold_list}, "
                f"hidden_dim={hidden_dim}, n_layers={n_layers}")
    
    eta_totals = []
    alpha_predicted_list = []
    alpha_measured_list = []
    eta_per_d = []
    
    for d_manifold in d_manifold_list:
        elapsed = time.time() - start_time
        elapsed_min = elapsed / 60
        
        # Heartbeat every 5 minutes
        if time.time() - last_heartbeat >= 300:
            logger.info(f"[HEARTBEAT {elapsed_min:.1f}min] Stage 2: d_manifold={d_manifold}")
            last_heartbeat = time.time()
        
        logger.info(f"\n--- Testing d_manifold = {d_manifold} ---")
        
        # Generate data
        gen = ManifoldDataGenerator(seed=42)
        X, y = gen.generate(n_samples=n_samples, d_manifold=d_manifold,
                            d_embed=hidden_dim, mode='nonlinear')
        
        # Create network
        net = MLP(input_dim=d_manifold, hidden_dim=hidden_dim,
                  output_dim=d_manifold, n_layers=n_layers, activation='relu')
        net.eval()
        
        # Compute η for each layer
        eta_values = compute_eta_for_network(net, X, n_layers)
        eta_per_d.append(eta_values)
        
        # Compute η_total = ∏η_l
        eta_total = np.prod(eta_values)
        eta_totals.append(eta_total)
        
        logger.info(f"  η per layer: {[f'{e:.4f}' for e in eta_values]}")
        logger.info(f"  η_total = ∏η_l = {eta_total:.6f}")
        
        # Predicted α ∝ η_total / d_manifold
        alpha_predicted = eta_total / d_manifold
        alpha_predicted_list.append(alpha_predicted)
        
        # Measured α
        alpha_measured, errors, scales = compute_alpha_measured(net, X, d_manifold, n_layers)
        alpha_measured_list.append(alpha_measured)
        
        logger.info(f"  α_predicted = η_total / d_manifold = {alpha_predicted:.6f}")
        logger.info(f"  α_measured = {alpha_measured:.6f}")
    
    # Compute R² for the relationship
    alpha_pred_arr = np.array(alpha_predicted_list)
    alpha_meas_arr = np.array(alpha_measured_list)
    
    # Normalize for comparison
    if np.std(alpha_pred_arr) > 1e-8:
        alpha_pred_norm = (alpha_pred_arr - np.mean(alpha_pred_arr)) / np.std(alpha_pred_arr)
        alpha_meas_norm = (alpha_meas_arr - np.mean(alpha_meas_arr)) / np.std(alpha_meas_arr)
        correlation = np.corrcoef(alpha_pred_norm, alpha_meas_norm)[0, 1]
        r_squared = correlation ** 2
    else:
        r_squared = 0.0
    
    logger.info("-" * 60)
    logger.info(f"Stage 2 Results:")
    logger.info(f"  d_manifold values: {d_manifold_list}")
    logger.info(f"  η_total values: {[f'{e:.6f}' for e in eta_totals]}")
    logger.info(f"  α_predicted: {[f'{e:.6f}' for e in alpha_predicted_list]}")
    logger.info(f"  α_measured: {[f'{e:.6f}' for e in alpha_measured_list]}")
    logger.info(f"  R² = {r_squared:.4f}")
    
    # Save progress
    stage2_results = {
        "stage": 2,
        "d_manifold_list": d_manifold_list,
        "eta_totals": eta_totals,
        "eta_per_layer_per_d": eta_per_d,
        "alpha_predicted": alpha_predicted_list,
        "alpha_measured": alpha_measured_list,
        "r_squared": float(r_squared),
        "config": {
            "hidden_dim": hidden_dim,
            "n_layers": n_layers,
            "n_samples": n_samples
        }
    }
    
    progress_file = results_dir / "stage2_progress.json"
    with open(progress_file, 'w') as f:
        json.dump(stage2_results, f, indent=2, default=str)
    logger.info(f"Saved progress to {progress_file}")
    
    return stage2_results


if __name__ == "__main__":
    results = main()
    print(f"\nStage 2 Final: R² = {results['r_squared']:.4f}")
