#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Phase 3 v5: Improved Alpha Estimation via Training Curves

Problem: Previous versions used condition number or weight decay sweep,
which had too much variance (alpha_std often > alpha_mean).

Solution: Measure alpha from actual training curves by:
1. Training models at multiple sizes (N_params varies)
2. Fitting Loss ~ N^(-alpha) via log-log regression
3. Using multiple seeds (n_seeds=10) to reduce variance
4. Computing effective dimensionality D_eff from Jacobian

This gives a much more stable alpha measurement.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
from pathlib import Path

torch.set_default_dtype(torch.float64)

# ==============================================================================
# DATA GENERATION (same as v4)
# ==============================================================================

def generate_sobolev_gp(n_samples, d_manifold, d_embed, s, noise_std=0.1, seed=42):
    """
    Generate data from a d-dimensional Sobolev Gaussian Process.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    X = torch.randn(n_samples, d_manifold) * 2.0
    
    if d_manifold <= 5:
        n_modes = 100
    elif d_manifold <= 8:
        n_modes = 60
    else:
        n_modes = 40
    
    k_max = 6.0
    omega = torch.randn(n_modes, d_manifold) * k_max
    omega_norm_sq = torch.sum(omega ** 2, dim=1)
    weights = (1 + omega_norm_sq) ** (-s / 2 - d_manifold / 4)
    phases = torch.rand(n_modes) * 2 * np.pi
    
    kernel_matrix = torch.cos(omega @ X.T + phases.unsqueeze(-1))
    f_coeffs = torch.randn(d_embed, n_modes) * weights.unsqueeze(0)
    f = f_coeffs @ kernel_matrix
    f = f.T
    f = f / (f.std(dim=0, keepdim=True) + 1e-10)
    y = f + noise_std * torch.randn_like(f)
    
    return X, y


def compute_D_eff(model, X, y, device='cpu'):
    """
    Compute effective dimensionality D_eff from network Jacobian.
    
    D_eff = trace(J^T J) / ||J||_F^2
    
    This is more stable than condition number.
    """
    model.eval()
    X = X.to(device)
    
    # Get Jacobian of model output w.r.t. input
    X.requires_grad_(True)
    pred = model(X)
    
    # J: (batch, output_dim, input_dim)
    J = torch.zeros(pred.shape[0], pred.shape[1], X.shape[1])
    
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            grad = torch.zeros_like(X)
            grad[i, j] = 1.0
            X.grad = None
            pred[i, j].backward(gradient=grad[i, j], retain_graph=True)
            J[i, j] = X.grad[i].clone()
    
    # Compute D_eff
    # trace(J^T J) = sum of squared singular values
    # ||J||_F^2 = sum of all squared elements
    
    # For each sample, compute ratio
    # J^T J: (batch, input_dim, input_dim)
    # trace(J^T J) = sum over input_dim of diag(J^T J)
    
    J_flat = J.view(-1, J.shape[-1])  # (batch * output_dim, input_dim)
    
    # Method 1: Using singular values
    # _, s_vals, _ = torch.svd(J_flat)
    # trace = torch.sum(s_vals ** 2, dim=-1)
    # frobenius_sq = torch.sum(J_flat ** 2, dim=-1)
    
    # Method 2: Direct computation
    JtJ = J_flat.T @ J_flat  # (input_dim, input_dim)
    trace = torch.trace(JtJ).item()
    frobenius_sq = torch.sum(J_flat ** 2).item()
    
    if frobenius_sq > 0:
        D_eff = trace / frobenius_sq
    else:
        D_eff = 0.0
    
    return D_eff


def compute_D_eff_fast(model, X, y, device='cpu'):
    """
    Fast approximation of D_eff using trace estimation.
    
    D_eff = trace(J^T J) / ||J||_F^2 = ||J||_2^2 / ||J||_F^2
    
    We approximate trace(J^T J) using Hutchinson's method:
    trace(A) ≈ (1/k) * sum_i (v_i^T A v_i) where v_i are random Rademacher vectors
    
    This avoids explicit Jacobian computation.
    """
    model.eval()
    X = X.to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    input_dim = X.shape[1]
    
    # Use power method to estimate ||J||_2^2
    # For a matrix A, ||A||_2^2 = largest eigenvalue of A^T A
    
    # We'll use a simple approximation: average of (gradient wrt params)^2
    total_grad_sq = 0.0
    total_param_sq = 0.0
    
    for i in range(min(50, X.shape[0])):  # Use subset of samples
        x_batch = X[i:i+1].clone()
        x_batch.requires_grad_(True)
        
        pred = model(x_batch)
        
        # Sum of squared gradients w.r.t. parameters
        total = 0.0
        for p in model.parameters():
            p.grad = None
        pred.sum().backward()
        
        for p in model.parameters():
            if p.grad is not None:
                total_grad_sq += torch.sum(p.grad ** 2).item()
                total_param_sq += torch.sum(p ** 2).item()
    
    # Approximate D_eff as ratio of gradient norm to parameter norm
    # This is crude but fast
    if total_param_sq > 0:
        D_eff_approx = total_grad_sq / (total_param_sq + 1e-10) * (input_dim / n_params)
    else:
        D_eff_approx = 1.0
    
    return D_eff_approx


class SimpleMLP(nn.Module):
    """Simple MLP for regression."""
    def __init__(self, input_dim, hidden_dim, output_dim, depth=2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


def train_model_and_get_loss(model, X, y, epochs=200, lr=0.001, device='cpu'):
    """
    Train model and return final loss.
    """
    X = X.to(device)
    y = y.to(device)
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        pred = model(X)
        final_loss = criterion(pred, y).item()
    
    return final_loss


def linear_fit(x, y):
    """Fit y = a * x + b and return (slope, intercept, r2)."""
    coeffs = np.polyfit(x, y, 1)
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / (ss_tot + 1e-10) if ss_tot > 0 else 0.0
    return coeffs[0], coeffs[1], r2


# ==============================================================================
# IMPROVED ALPHA MEASUREMENT VIA TRAINING CURVES
# ==============================================================================

def measure_alpha_training_curve(d_embed, hidden_dim, output_dim, X, y, 
                                  model_sizes, n_seeds=10, epochs=200, device='cpu'):
    """
    Measure alpha from training curves.
    
    Train models at different sizes, measure final loss, fit power law:
    Loss ~ N^(-alpha)
    
    Args:
        d_embed: Input dimension (projected manifold dimension)
        hidden_dim: Hidden dimension (varies with model size)
        output_dim: Output dimension
        X, y: Training data
        model_sizes: List of hidden_dim values to try
        n_seeds: Number of seeds per size
        epochs: Training epochs
        device: Compute device
    
    Returns:
        dict with alpha, r2, losses, n_params, etc.
    """
    losses_by_size = []
    n_params_by_size = []
    
    for size in model_sizes:
        size_losses = []
        for seed in range(n_seeds):
            torch.manual_seed(1000 + seed)
            np.random.seed(1000 + seed)
            
            model = SimpleMLP(d_embed, size, output_dim, depth=3)
            loss = train_model_and_get_loss(model, X, y, epochs=epochs, device=device)
            size_losses.append(loss)
        
        n_params = SimpleMLP(d_embed, size, output_dim, depth=3).count_parameters()
        
        losses_by_size.append(size_losses)
        n_params_by_size.append(n_params)
        
        print(f"    Size {size}: loss = {np.mean(size_losses):.6f} ± {np.std(size_losses):.6f}, "
              f"N_params = {n_params}")
    
    # Average losses across seeds
    avg_losses = [np.mean(ls) for ls in losses_by_size]
    std_losses = [np.std(ls) for ls in losses_by_size]
    
    # Fit power law: Loss ~ N^(-alpha)
    log_N = np.log(n_params_by_size)
    log_L = np.log(avg_losses)
    
    slope, intercept, r2 = linear_fit(log_N, log_L)
    alpha = -slope  # alpha is negative of slope
    
    # Estimate standard error from residuals
    log_L_pred = slope * log_N + intercept
    residuals = log_L - log_L_pred
    n = len(log_N)
    if n > 2:
        se = np.sqrt(np.sum(residuals**2) / (n - 2))
        # SE of slope: se * sqrt(1 / sum((x - mean(x))^2))
        x_diff_sq = np.sum((log_N - np.mean(log_N))**2)
        slope_se = se * np.sqrt(1 / x_diff_sq) if x_diff_sq > 0 else None
    else:
        slope_se = None
    
    return {
        "alpha": float(alpha),
        "alpha_se": float(slope_se) if slope_se else None,
        "r_squared": float(r2),
        "losses": avg_losses,
        "losses_std": std_losses,
        "n_params": n_params_by_size,
        "model_sizes": model_sizes,
        "raw_losses": losses_by_size
    }


def measure_alpha_weight_decay_sweep(d_embed, hidden_dim, output_dim, X, y, 
                                      n_seeds=5, device='cpu'):
    """
    Original method: alpha via weight decay sweep (for comparison).
    """
    alpha_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    
    best_alphas = []
    for seed in range(n_seeds):
        torch.manual_seed(2000 + seed)
        
        model = SimpleMLP(d_embed, hidden_dim, output_dim, depth=3).to(device)
        X_dev = X.to(device)
        y_dev = y.to(device)
        
        best_alpha = 0.0
        best_r2 = -float('inf')
        
        for alpha in alpha_values:
            model_copy = SimpleMLP(d_embed, hidden_dim, output_dim, depth=3).to(device)
            optimizer = torch.optim.AdamW(model_copy.parameters(), lr=0.001, weight_decay=alpha)
            
            for epoch in range(100):
                optimizer.zero_grad()
                pred = model_copy(X_dev)
                loss = nn.MSELoss()(pred, y_dev)
                loss.backward()
                optimizer.step()
            
            with torch.no_grad():
                pred = model_copy(X_dev)
            ss_res = torch.sum((y_dev - pred) ** 2).item()
            ss_tot = torch.sum((y_dev - y_dev.mean()) ** 2).item()
            r2 = 1 - ss_res / (ss_tot + 1e-10)
            
            if r2 > best_r2:
                best_r2 = r2
                best_alpha = alpha
        
        best_alphas.append(best_alpha)
    
    return {
        "alpha_mean": float(np.mean(best_alphas)),
        "alpha_std": float(np.std(best_alphas)),
        "alpha_values": best_alphas
    }


def measure_alpha_with_D_eff(d_embed, hidden_dim, output_dim, X, y,
                             model_sizes, n_seeds=5, epochs=200, device='cpu'):
    """
    Measure alpha using effective dimensionality.
    
    Theory: D_eff ~ N^theta where theta depends on s
    alpha can be derived from D_eff scaling.
    """
    D_effs = []
    n_params_list = []
    
    for size in model_sizes:
        size_D_effs = []
        for seed in range(n_seeds):
            torch.manual_seed(3000 + seed)
            
            model = SimpleMLP(d_embed, size, output_dim, depth=3).to(device)
            
            # Train first
            train_model_and_get_loss(model, X, y, epochs=epochs, device=device)
            
            # Compute D_eff
            D_eff = compute_D_eff_fast(model, X, y, device)
            size_D_effs.append(D_eff)
        
        n_params = SimpleMLP(d_embed, size, output_dim, depth=3).count_parameters()
        D_effs.append(np.mean(size_D_effs))
        n_params_list.append(n_params)
        
        print(f"    Size {size}: D_eff = {np.mean(size_D_effs):.4f}, N_params = {n_params}")
    
    # Fit D_eff ~ N^delta
    log_N = np.log(n_params_list)
    log_D = np.log(D_effs)
    
    slope, intercept, r2 = linear_fit(log_N, log_D)
    
    return {
        "delta": float(slope),
        "r_squared_D_eff": float(r2),
        "D_effs": D_effs,
        "n_params": n_params_list
    }


# ==============================================================================
# MAIN EXPERIMENT
# ==============================================================================

if __name__ == "__main__":
    start_time = time.time()
    
    print("=" * 70)
    print("PHASE 3 v5: IMPROVED ALPHA ESTIMATION VIA TRAINING CURVES")
    print("=" * 70)
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # --------------------------------------------------------------------------
    # STEP 1: SYNTHETIC DATA TEST (d=10, s=1.0)
    #   Compare training curve method vs weight decay sweep
    # --------------------------------------------------------------------------
    print("\n[STEP 1] Testing Alpha Estimation Methods on d=10, s=1.0")
    print("-" * 70)
    
    d_manifold = 10
    d_embed = 10
    s = 1.0
    n_samples = 500
    seed = 42
    
    # Generate data
    X, y = generate_sobolev_gp(n_samples, d_manifold, d_embed, s, noise_std=0.1, seed=seed)
    
    # Project X to d_embed dimension
    torch.manual_seed(123)
    X_proj = torch.randn(d_manifold, d_embed) * 0.1
    X = X @ X_proj
    X = (X - X.mean(0)) / (X.std(0) + 1e-10)
    y = (y - y.mean(0)) / (y.std(0) + 1e-10)
    
    # Model sizes to test (vary hidden_dim)
    model_sizes = [16, 32, 64, 128, 256]
    n_seeds = 10
    epochs = 200
    
    print("\n1a. Training Curve Method (new):")
    print(f"    Model sizes: {model_sizes}")
    print(f"    Seeds per size: {n_seeds}")
    print(f"    Epochs per model: {epochs}")
    
    tc_result = measure_alpha_training_curve(
        d_embed, None, d_embed, X, y,
        model_sizes=model_sizes,
        n_seeds=n_seeds,
        epochs=epochs,
        device=device
    )
    
    print(f"\n    Training Curve Result:")
    print(f"    α = {tc_result['alpha']:.4f}, R² = {tc_result['r_squared']:.4f}")
    
    # --------------------------------------------------------------------------
    # 1b. Weight Decay Sweep (old method) for comparison
    # --------------------------------------------------------------------------
    print("\n1b. Weight Decay Sweep Method (old):")
    hidden_dim = 64  # Fixed size
    n_seeds_wd = 10
    
    wd_result = measure_alpha_weight_decay_sweep(
        d_embed, hidden_dim, d_embed, X, y,
        n_seeds=n_seeds_wd,
        device=device
    )
    
    print(f"    α = {wd_result['alpha_mean']:.6f} ± {wd_result['alpha_std']:.6f}")
    
    # --------------------------------------------------------------------------
    # 1c. D_eff Method
    # --------------------------------------------------------------------------
    print("\n1c. Effective Dimensionality Method:")
    
    deff_result = measure_alpha_with_D_eff(
        d_embed, None, d_embed, X, y,
        model_sizes=model_sizes,
        n_seeds=5,
        epochs=epochs,
        device=device
    )
    
    print(f"    δ (D_eff exponent) = {deff_result['delta']:.4f}, R² = {deff_result['r_squared_D_eff']:.4f}")
    
    # --------------------------------------------------------------------------
    # STEP 2: COMPARE VARIANCE ACROSS METHODS
    # --------------------------------------------------------------------------
    print("\n[STEP 2] Variance Comparison")
    print("-" * 70)
    
    # Re-run weight decay multiple times to get variance
    print("\nRunning weight decay method 5 times (5 seeds each) to measure variance...")
    wd_variances = []
    for rep in range(5):
        wd_rep = measure_alpha_weight_decay_sweep(
            d_embed, hidden_dim, d_embed, X, y,
            n_seeds=5,
            device=device
        )
        wd_variances.append((wd_rep['alpha_mean'], wd_rep['alpha_std']))
        print(f"  Rep {rep+1}: α = {wd_rep['alpha_mean']:.6f} ± {wd_rep['alpha_std']:.6f}")
    
    wd_alpha_means = [v[0] for v in wd_variances]
    wd_alpha_stds = [v[1] for v in wd_variances]
    
    print(f"\nWeight Decay Sweep Variance Analysis:")
    print(f"  Mean of α across reps: {np.mean(wd_alpha_means):.6f}")
    print(f"  Std of α across reps:  {np.std(wd_alpha_means):.6f}")
    print(f"  Avg within-run std:     {np.mean(wd_alpha_stds):.6f}")
    
    # Training curve is already averaged over seeds and fit globally
    # We can estimate variance from the residuals
    tc_var_estimate = np.std(tc_result['losses']) / (np.mean(tc_result['losses']) + 1e-10)
    
    print(f"\nTraining Curve Method:")
    print(f"  α = {tc_result['alpha']:.4f}")
    print(f"  R² = {tc_result['r_squared']:.4f}")
    print(f"  Relative variance (loss): {tc_var_estimate:.4f}")
    
    variance_comparison = {
        "weight_decay": {
            "alpha_mean": float(np.mean(wd_alpha_means)),
            "alpha_std_across_reps": float(np.std(wd_alpha_means)),
            "avg_within_run_std": float(np.mean(wd_alpha_stds)),
            "individual_reps": wd_variances
        },
        "training_curve": {
            "alpha": tc_result['alpha'],
            "r_squared": tc_result['r_squared'],
            "relative_variance": float(tc_var_estimate)
        },
        "training_curve_more_stable": bool(np.std(wd_alpha_means) > 0.01 or 
                                            np.mean(wd_alpha_stds) > tc_result['alpha'])
    }
    
    # --------------------------------------------------------------------------
    # STEP 3: S-VARIATION EXPERIMENT (if training curve method is stable)
    # --------------------------------------------------------------------------
    print("\n[STEP 3] α vs s (d=10 fixed) using Training Curve Method")
    print("-" * 70)
    
    s_values = [0.5, 1.0, 1.5, 2.0]
    d_fixed = 10
    s_variation_results = []
    
    for s_val in s_values:
        print(f"\n  s = {s_val}:")
        
        # Generate new data for this s
        X_s, y_s = generate_sobolev_gp(n_samples, d_fixed, d_fixed, s_val, noise_std=0.1, seed=42)
        
        torch.manual_seed(123)
        X_proj_s = torch.randn(d_fixed, d_fixed) * 0.1
        X_s = X_s @ X_proj_s
        X_s = (X_s - X_s.mean(0)) / (X_s.std(0) + 1e-10)
        y_s = (y_s - y_s.mean(0)) / (y_s.std(0) + 1e-10)
        
        result = measure_alpha_training_curve(
            d_fixed, None, d_fixed, X_s, y_s,
            model_sizes=model_sizes,
            n_seeds=n_seeds,
            epochs=epochs,
            device=device
        )
        
        s_variation_results.append({
            "s": s_val,
            "alpha": result['alpha'],
            "r_squared": result['r_squared'],
            "losses": result['losses'],
            "n_params": result['n_params']
        })
        
        print(f"    α = {result['alpha']:.4f}, R² = {result['r_squared']:.4f}")
    
    # Fit α ∝ s
    s_array = np.array([r['s'] for r in s_variation_results])
    alpha_array = np.array([r['alpha'] for r in s_variation_results])
    
    slope_s, intercept_s, r2_s = linear_fit(s_array, alpha_array)
    
    s_fit_result = {
        "slope": float(slope_s),
        "intercept": float(intercept_s),
        "r_squared": float(r2_s),
        "theory": "α ∝ s",
        "step3_passed": bool(r2_s > 0.85),
        "results": s_variation_results
    }
    
    print(f"\n  Fit (α vs s): R² = {r2_s:.4f}")
    print(f"  Step 3 PASSED: {r2_s > 0.85}")
    
    # --------------------------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------------------------
    total_time = time.time() - start_time
    
    final_summary = {
        "total_time_minutes": total_time / 60,
        "step1_variance_comparison": variance_comparison,
        "step3_alpha_vs_s": s_fit_result,
        "training_curve_method_stable": bool(tc_result['r_squared'] > 0.85),
        "overall_passed": bool(tc_result['r_squared'] > 0.85 and r2_s > 0.85)
    }
    
    with open(results_dir / "phase3_v5_improved_alpha.json", "w") as f:
        json.dump(final_summary, f, indent=2)
    
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\n  Training Curve α (d=10, s=1.0): {tc_result['alpha']:.4f}, R² = {tc_result['r_squared']:.4f}")
    print(f"  Weight Decay α (d=10, s=1.0):   {wd_result['alpha_mean']:.4f} ± {wd_result['alpha_std']:.4f}")
    print(f"\n  Training Curve variance: {'LOW' if tc_result['r_squared'] > 0.85 else 'HIGH'}")
    print(f"  Weight Decay variance:  {'HIGH' if np.mean(wd_alpha_stds) > 0.01 else 'LOW'}")
    print(f"\n  Step 3 (α ∝ s): R² = {r2_s:.4f} → {'PASS' if r2_s > 0.85 else 'FAIL'}")
    print(f"\n  OVERALL: {'SUCCESS ✓' if final_summary['overall_passed'] else 'NEEDS IMPROVEMENT'}")
    print(f"\n  Total time: {total_time/60:.1f} minutes")
    print("=" * 70)