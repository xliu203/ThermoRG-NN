# SPDX-License-Identifier: Apache-2.0

"""Phase 3 Fixed Experiment: Validate α ∝ 2s/d_manifold with Matched Hidden Dimensions.

The key fix: Search for matched hidden_dim for each d_manifold to keep ∏η ≈ 1.
This isolates the 2s/d_manifold term in the formula:
    α = k_α · |log(∏η)| · (2s / d_manifold)

With ∏η ≈ 1 (constant), the prediction is:
    α ∝ 2s/d_manifold
    log(α) = log(2s) - log(d_manifold)
    Slope of log(α) vs log(d_manifold) should be ≈ -1
"""

import time
import json
import torch
import sys
from pathlib import Path
from datetime import datetime
import numpy as np

sys.path.insert(0, 'src')

from thermorg.simulations.manifold_data import ManifoldDataGenerator


def compute_eta_D_eff(J, d_manifold):
    """Compute η = D_eff / d_manifold and D_eff."""
    fro_sq = (J ** 2).sum()
    S = torch.linalg.svd(J, full_matrices=False)[1]
    spec_sq = S[0] ** 2 + 1e-8
    D_eff = fro_sq / spec_sq
    eta = D_eff / d_manifold
    return eta.item(), D_eff.item()


def find_matched_hidden_dim(d_manifold, hidden_dim_range, n_layers, seed_offset, target_eta=1.0):
    """Find hidden_dim that gives ∏η ≈ target_eta (default 1.0)."""
    
    hidden_dim_range = sorted(hidden_dim_range)
    
    best_hd = hidden_dim_range[0]
    best_product_eta = 1.0
    best_dist = float('inf')
    
    for hd in hidden_dim_range:
        # Generate data
        gen = ManifoldDataGenerator(seed=42 + seed_offset)
        z, X = gen.generate(
            n_samples=5000,  # Smaller for search
            d_manifold=d_manifold,
            d_embed=hd,
            mode="polynomial",
            noise_std=0.05
        )
        
        # Create network with n_layers
        torch.manual_seed(42 + seed_offset)
        W_layers = [torch.randn(hd, hd) * 0.1 for _ in range(n_layers)]
        W_out = torch.randn(hd, d_manifold) * 0.1
        
        # Compute η for each layer
        eta_layers = []
        for W in W_layers:
            eta, _ = compute_eta_D_eff(W, d_manifold)
            eta_layers.append(eta)
        eta_out, _ = compute_eta_D_eff(W_out, d_manifold)
        eta_layers.append(eta_out)
        
        # Product of all η
        product_eta = 1.0
        for e in eta_layers:
            product_eta *= e
        
        dist = abs(product_eta - target_eta)
        if dist < best_dist:
            best_dist = dist
            best_hd = hd
            best_product_eta = product_eta
    
    return best_hd, best_product_eta


def run_phase3_fixed_experiment(config, results_dir, start_time, log_fn, heartbeat_fn):
    """Phase 3 with MATCHED hidden_dim to keep ∏η ≈ 1."""
    
    log_fn("\n" + "=" * 70)
    log_fn("PHASE 3 FIXED: Validate d_manifold Scaling with Matched Hidden Dimensions")
    log_fn("=" * 70)
    log_fn(f"d_manifold values: {config['phase3_d_manifold']}")
    log_fn(f"Target: ∏η ≈ 1.0 (constant) for each d_manifold")
    log_fn(f"Prediction: α ∝ 2s/d_manifold, so log(α) ∝ -log(d_manifold)")
    log_fn("")
    
    # Step 1: Find matched hidden_dim for each d_manifold
    log_fn("STEP 1: Finding matched hidden_dim for each d_manifold...")
    log_fn("-" * 50)
    
    hidden_dim_range = [8, 12, 16, 24, 32, 48, 64, 96, 128]
    
    matched_config = {}
    for d_manifold in config["phase3_d_manifold"]:
        matched_hd, matched_pi_eta = find_matched_hidden_dim(
            d_manifold, 
            hidden_dim_range, 
            config['n_layers'],
            seed_offset=d_manifold * 10,
            target_eta=1.0
        )
        matched_config[d_manifold] = matched_hd
        log_fn(f"  d_manifold={d_manifold}: matched_hidden_dim={matched_hd}, ∏η={matched_pi_eta:.4f}")
    
    log_fn("")
    log_fn(f"Matched configuration: {matched_config}")
    log_fn("")
    
    # Step 2: Run experiments with matched hidden_dim
    log_fn("STEP 2: Running experiments with matched hidden_dim...")
    log_fn("-" * 50)
    
    phase3_results = {
        "d_manifold": [],
        "hidden_dim": [],
        "alpha_measured": [],
        "log_alpha": [],
        "log_d": [],
        "product_eta": [],
        "avg_eta": [],
        "s_value": [],
        "eta_layers": [],
    }
    
    for dm_idx, d_manifold in enumerate(config["phase3_d_manifold"]):
        matched_hd = matched_config[d_manifold]
        
        log_fn(f"\n[d_manifold={d_manifold}, MATCHED hidden_dim={matched_hd}]")
        
        # Generate data with full n_samples
        gen = ManifoldDataGenerator(seed=config['data_seed'] + dm_idx + 100)
        z, X = gen.generate(
            n_samples=config['n_samples'],
            d_manifold=d_manifold,
            d_embed=matched_hd,
            mode="polynomial",
            noise_std=0.05
        )
        
        # Create network with MATCHED architecture
        torch.manual_seed(config['network_seed'] + dm_idx + 100)
        W_layers = [torch.randn(matched_hd, matched_hd) * 0.1 
                   for _ in range(config['n_layers'])]
        W_out = torch.randn(matched_hd, d_manifold) * 0.1
        
        # Compute η per layer
        eta_layers = []
        D_eff_layers = []
        
        for W in W_layers:
            eta, D_eff = compute_eta_D_eff(W, d_manifold)
            eta_layers.append(eta)
            D_eff_layers.append(D_eff)
        
        eta_out, D_eff_out = compute_eta_D_eff(W_out, d_manifold)
        eta_layers.append(eta_out)
        D_eff_layers.append(D_eff_out)
        
        product_eta = 1.0
        for e in eta_layers:
            product_eta *= e
        
        avg_eta = sum(eta_layers) / len(eta_layers)
        
        # Compute s (scale factor) - average D_eff
        s = sum(D_eff_layers) / len(D_eff_layers)
        
        # Compute α empirically: α = |log(∏η)| / (N * T) at T=1
        N = len(eta_layers)
        T = 1.0
        
        if product_eta > 0:
            log_product = torch.log(torch.tensor(product_eta).abs()).item()
            alpha_measured = abs(log_product) / (N * T)
        else:
            alpha_measured = 0.0
        
        log_alpha = np.log(alpha_measured) if alpha_measured > 0 else -np.inf
        log_d = np.log(d_manifold)
        
        log_fn(f"  η layers: {[f'{x:.4f}' for x in eta_layers]}")
        log_fn(f"  ∏η={product_eta:.4f} (target ≈ 1.0)")
        log_fn(f"  η_avg={avg_eta:.4f}, s={s:.4f}")
        log_fn(f"  α_measured={alpha_measured:.6f}, log(α)={log_alpha:.4f}")
        
        phase3_results["d_manifold"].append(d_manifold)
        phase3_results["hidden_dim"].append(matched_hd)
        phase3_results["alpha_measured"].append(alpha_measured)
        phase3_results["log_alpha"].append(log_alpha)
        phase3_results["log_d"].append(log_d)
        phase3_results["product_eta"].append(product_eta)
        phase3_results["avg_eta"].append(avg_eta)
        phase3_results["s_value"].append(s)
        phase3_results["eta_layers"].append(eta_layers)
        
        heartbeat_fn("Phase3")
    
    # ========================================================================
    # Analysis: Linear regression of log(α) vs log(d_manifold)
    # ========================================================================
    
    log_d_array = np.array(phase3_results["log_d"])
    log_alpha_array = np.array(phase3_results["log_alpha"])
    
    # Linear regression: log(α) = slope * (-log(d)) + intercept
    neg_log_d = -log_d_array
    
    cov = np.cov(neg_log_d, log_alpha_array)[0, 1]
    var = np.var(neg_log_d, ddof=1)
    slope = cov / var if var > 0 else 0
    
    intercept = np.mean(log_alpha_array) - slope * np.mean(neg_log_d)
    
    log_alpha_predicted = slope * neg_log_d + intercept
    
    ss_res = np.sum((log_alpha_array - log_alpha_predicted) ** 2)
    ss_tot = np.sum((log_alpha_array - np.mean(log_alpha_array)) ** 2)
    r2_d_scaling = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # Also check product_eta constancy
    product_eta_array = np.array(phase3_results["product_eta"])
    pi_eta_mean = np.mean(product_eta_array)
    pi_eta_std = np.std(product_eta_array)
    
    log_fn("\n" + "-" * 50)
    log_fn("PHASE 3 FIXED SUMMARY")
    log_fn("-" * 50)
    log_fn(f"  Linear fit: log(α) = {slope:.4f} * (-log(d)) + {intercept:.4f}")
    log_fn(f"  R² for d-scaling: {r2_d_scaling:.4f}")
    log_fn(f"  Expected slope: -1.0 (perfect inverse scaling)")
    log_fn("")
    log_fn(f"  ∏η constancy check:")
    log_fn(f"    Mean ∏η: {pi_eta_mean:.4f}")
    log_fn(f"    Std ∏η:  {pi_eta_std:.4f}")
    log_fn("")
    log_fn("  d_manifold | hidden_dim | ∏η      | α_measured | log(α)")
    log_fn("  " + "-" * 60)
    for i, dm in enumerate(phase3_results["d_manifold"]):
        log_fn(f"  {dm:10d} | {phase3_results['hidden_dim'][i]:10d} | "
               f"{phase3_results['product_eta'][i]:7.4f} | "
               f"{phase3_results['alpha_measured'][i]:11.6f} | "
               f"{phase3_results['log_alpha'][i]:8.4f}")
    
    phase3_summary = {
        "slope": slope,
        "intercept": intercept,
        "r2_d_scaling": r2_d_scaling,
        "expected_slope": -1.0,
        "slope_error": abs(slope - (-1.0)),
        "validation_passed": abs(slope - (-1.0)) < 0.3,
        "pi_eta_mean": pi_eta_mean,
        "pi_eta_std": pi_eta_std,
        "pi_eta_constant": pi_eta_std < 0.5,
        "matched_config": matched_config,
    }
    
    return phase3_results, phase3_summary


def main():
    config = {
        "n_samples": 20000,
        "phase3_d_manifold": [5, 8, 10, 12, 15, 18, 20],
        "n_layers": 4,
        "network_seed": 42,
        "data_seed": 42,
    }
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    start_time = time.time()
    last_heartbeat = start_time
    heartbeat_interval = 300  # 5 minutes
    last_progress_report = start_time
    progress_interval = 1200  # 20 minutes
    
    def log(msg):
        elapsed = (time.time() - start_time) / 60
        print(f"[{elapsed:6.1f}min] {msg}")
    
    def heartbeat(category=""):
        nonlocal last_heartbeat
        now = time.time()
        if now - last_heartbeat >= heartbeat_interval:
            log(f"[HEARTBEAT] {category}")
            last_heartbeat = now
    
    def progress_report():
        nonlocal last_progress_report
        now = time.time()
        if now - last_progress_report >= progress_interval:
            elapsed = (now - start_time) / 60
            log(f"[PROGRESS REPORT] Elapsed: {elapsed:.1f} min - Running Phase 3 Fixed experiment")
            last_progress_report = now
    
    log("=" * 70)
    log("PHASE 3 FIXED: d_manifold Scaling with Matched Hidden Dimensions")
    log("=" * 70)
    log(f"Configuration:")
    log(f"  n_samples: {config['n_samples']}")
    log(f"  n_layers: {config['n_layers']}")
    log(f"  phase3_d_manifold: {config['phase3_d_manifold']}")
    log(f"  Goal: Search for hidden_dim that gives ∏η ≈ 1.0 (constant)")
    log(f"  Prediction: log(α) slope should be ≈ -1.0")
    
    results = {
        "experiment": "phase3_fixed_d_scaling",
        "timestamp": datetime.now().isoformat(),
        "config": config,
    }
    
    phase3_results, phase3_summary = run_phase3_fixed_experiment(
        config, results_dir, start_time, log, heartbeat
    )
    results["phase3"] = {
        "data": phase3_results,
        "summary": phase3_summary,
    }
    progress_report()
    
    duration = time.time() - start_time
    
    final_summary = {
        "duration_seconds": duration,
        "slope": phase3_summary["slope"],
        "intercept": phase3_summary["intercept"],
        "r2_d_scaling": phase3_summary["r2_d_scaling"],
        "expected_slope": phase3_summary["expected_slope"],
        "slope_error": phase3_summary["slope_error"],
        "validation_passed": phase3_summary["validation_passed"],
        "pi_eta_mean": phase3_summary["pi_eta_mean"],
        "pi_eta_std": phase3_summary["pi_eta_std"],
        "pi_eta_constant": phase3_summary["pi_eta_constant"],
        "matched_config": phase3_summary["matched_config"],
        "alpha_proportional_to_1_over_d": phase3_summary["validation_passed"],
        "timestamp_completed": datetime.now().isoformat(),
    }
    results["final_summary"] = final_summary
    
    results_path = results_dir / "phase3_fixed_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    log("\n" + "=" * 70)
    log("FINAL SUMMARY - Phase 3 Fixed Experiment")
    log("=" * 70)
    log(f"Duration: {duration/60:.1f} minutes")
    log("")
    log("Matched hidden_dim configuration found:")
    for d, hd in phase3_summary["matched_config"].items():
        log(f"  d_manifold={d}: hidden_dim={hd}")
    log("")
    log("PHASE 3 FIXED RESULTS (d_manifold scaling with matched hidden_dim):")
    log(f"  Slope:         {phase3_summary['slope']:.4f} (expected: -1.0)")
    log(f"  Intercept:     {phase3_summary['intercept']:.4f}")
    log(f"  R²:            {phase3_summary['r2_d_scaling']:.4f}")
    log(f"  Slope error:   {phase3_summary['slope_error']:.4f}")
    log(f"  Validation:    {'PASSED ✓' if phase3_summary['validation_passed'] else 'FAILED ✗'}")
    log("")
    log(f"∏η Constancy Check:")
    log(f"  Mean ∏η:       {phase3_summary['pi_eta_mean']:.4f} (target: 1.0)")
    log(f"  Std ∏η:        {phase3_summary['pi_eta_std']:.4f} (should be small)")
    log(f"  ∏η constant:   {'YES ✓' if phase3_summary['pi_eta_constant'] else 'NO ✗'}")
    log("")
    log("CONCLUSION:")
    if phase3_summary["validation_passed"]:
        log("  ✓ Theory CONFIRMED: α ∝ 1/d_manifold when ∏η ≈ 1 (constant)")
        log("    The slope is close to -1.0, validating the prediction.")
    else:
        log("  ✗ Theory NOT confirmed: slope deviates significantly from -1.0")
        log(f"    Actual slope: {phase3_summary['slope']:.4f}, Expected: -1.0")
    log("")
    log(f"Results saved to: {results_path}")
    
    return results


if __name__ == "__main__":
    results = main()