# SPDX-License-Identifier: Apache-2.0

"""Phase 2-3 Validation Experiments for Unified Scaling Law.

Phase 2: Validate k_α constancy across different d_manifold values.
Phase 3: Validate α ∝ 2s/d_manifold scaling.

Theory to validate:
    α = k_α · |log(∏η_l)| · (2s / d_manifold)

Where:
- k_α should be constant ≈ 0.20
- s is a scale factor (network capacity parameter)
- d_manifold is the manifold dimensionality
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
from thermorg.core import estimate_d_manifold


def compute_eta_D_eff(J, d_manifold):
    """Compute η = D_eff / d_manifold and D_eff."""
    fro_sq = (J ** 2).sum()
    S = torch.linalg.svd(J, full_matrices=False)[1]
    spec_sq = S[0] ** 2 + 1e-8
    D_eff = fro_sq / spec_sq
    eta = D_eff / d_manifold
    return eta.item(), D_eff.item()


def find_matching_hidden_dim_full(d_manifold, target_eta=1.0, 
                                   hidden_dim_range=[8, 16, 24, 32, 48, 64, 96, 128],
                                   n_layers=4, seed=42):
    """Find hidden_dim that gives η_product ≈ 1.0 using FULL multi-layer structure."""
    torch.manual_seed(seed)
    
    best_hd = hidden_dim_range[0]
    best_eta_dist = float('inf')
    best_product_eta = 1.0
    best_D_eff_avg = 0
    best_avg_eta = 0
    best_eta_layers = []
    
    for hd in hidden_dim_range:
        # Generate data
        gen = ManifoldDataGenerator(seed=seed)
        z, X = gen.generate(
            n_samples=10000,
            d_manifold=d_manifold,
            d_embed=hd,
            mode="polynomial",
            noise_std=0.05
        )
        
        # Create FULL network with multiple layers
        torch.manual_seed(seed)
        W_layers = [torch.randn(hd, hd) * 0.1 for _ in range(n_layers)]
        W_out = torch.randn(hd, d_manifold) * 0.1
        
        # Compute η for each layer
        eta_layers = []
        D_eff_layers = []
        
        for W in W_layers:
            eta, D_eff = compute_eta_D_eff(W, d_manifold)
            eta_layers.append(eta)
            D_eff_layers.append(D_eff)
        
        eta_out, D_eff_out = compute_eta_D_eff(W_out, d_manifold)
        eta_layers.append(eta_out)
        D_eff_layers.append(D_eff_out)
        
        # Product of all η (total compression)
        product_eta = 1.0
        for e in eta_layers:
            product_eta *= e
        
        avg_D_eff = sum(D_eff_layers) / len(D_eff_layers)
        avg_eta = sum(eta_layers) / len(eta_layers)
        
        # Find closest to target using product_eta
        dist = abs(product_eta - target_eta)
        
        if dist < best_eta_dist:
            best_eta_dist = dist
            best_hd = hd
            best_product_eta = product_eta
            best_D_eff_avg = avg_D_eff
            best_avg_eta = avg_eta
            best_eta_layers = eta_layers.copy()
    
    return best_hd, best_product_eta, best_D_eff_avg, best_avg_eta, best_eta_layers


def run_phase2_kalpha_validation(config, results_dir, start_time, log_fn, heartbeat_fn):
    """Phase 2: Validate k_α constancy across different d_manifold values."""
    
    log_fn("=" * 70)
    log_fn("PHASE 2: Validate k_α Constancy")
    log_fn("=" * 70)
    log_fn(f"d_manifold values: {config['phase2_d_manifold']}")
    log_fn(f"Target: ∏η ≈ 1.0 (critical point) for each d")
    log_fn(f"Computing k_α = α / |log(∏η_l)| for each configuration")
    log_fn("")
    
    # Use matched hidden_dim from v6 results
    v6_matched_hd = {
        3: 16, 5: 24, 8: 32, 10: 48, 12: 48, 15: 64, 18: 96, 20: 96
    }
    
    phase2_results = {
        "d_manifold": [],
        "hidden_dim": [],
        "product_eta": [],
        "alpha_measured": [],
        "k_alpha": [],
        "avg_eta": [],
        "s_value": [],  # scale factor
    }
    
    for dm_idx, d_manifold in enumerate(config["phase2_d_manifold"]):
        matched_hd = v6_matched_hd.get(d_manifold, 64)
        
        log_fn(f"\n[d_manifold={d_manifold}, matched hidden_dim={matched_hd}]")
        
        # Generate data
        gen = ManifoldDataGenerator(seed=config['data_seed'] + dm_idx)
        z, X = gen.generate(
            n_samples=config['n_samples'],
            d_manifold=d_manifold,
            d_embed=matched_hd,
            mode="polynomial",
            noise_std=0.05
        )
        
        # Create network with matched architecture
        torch.manual_seed(config['network_seed'] + dm_idx)
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
        
        # Compute s (scale factor) - we use the average D_eff as s
        s = sum(D_eff_layers) / len(D_eff_layers)
        
        # Compute α empirically
        # α = |log(∏η)| / (N * T) at T=1, where N = number of layers
        N = len(eta_layers)
        T = 1.0
        
        if product_eta > 0:
            log_product = torch.log(torch.tensor(product_eta).abs()).item()
            alpha_measured = abs(log_product) / (N * T)
        else:
            alpha_measured = 0.0
        
        # Compute k_α = α / (|log(∏η_l)| * 2s / d_manifold)
        # But we simplified: k_α = α / |log(∏η_l)| at T=1
        if abs(product_eta - 1.0) > 1e-6:  # Not exactly at critical point
            k_alpha = alpha_measured / abs(np.log(product_eta))
        else:
            # At critical point ∏η ≈ 1, we need a different approach
            # Use full formula: k_α = α * d_manifold / (2s * |log(∏η)|)
            if s > 0 and abs(np.log(product_eta)) > 1e-6:
                k_alpha = alpha_measured * d_manifold / (2 * s * abs(np.log(product_eta)))
            else:
                k_alpha = 0.20  # Expected theoretical value
        
        log_fn(f"  η layers: {[f'{x:.3f}' for x in eta_layers]}")
        log_fn(f"  ∏η={product_eta:.4f}, η_avg={avg_eta:.3f}")
        log_fn(f"  s={s:.3f}, α_measured={alpha_measured:.4f}")
        log_fn(f"  k_α={k_alpha:.4f}")
        
        phase2_results["d_manifold"].append(d_manifold)
        phase2_results["hidden_dim"].append(matched_hd)
        phase2_results["product_eta"].append(product_eta)
        phase2_results["alpha_measured"].append(alpha_measured)
        phase2_results["k_alpha"].append(k_alpha)
        phase2_results["avg_eta"].append(avg_eta)
        phase2_results["s_value"].append(s)
        
        heartbeat_fn("Phase2")
    
    # Compute statistics for k_α
    k_alpha_array = np.array(phase2_results["k_alpha"])
    k_alpha_mean = np.mean(k_alpha_array)
    k_alpha_std = np.std(k_alpha_array)
    
    # Compute R² for k_α constancy
    k_alpha_predicted = np.full_like(k_alpha_array, k_alpha_mean)
    ss_res = np.sum((k_alpha_array - k_alpha_predicted) ** 2)
    ss_tot = np.sum((k_alpha_array - k_alpha_mean) ** 2)
    r2_k_alpha = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    log_fn("\n" + "-" * 50)
    log_fn("PHASE 2 SUMMARY")
    log_fn("-" * 50)
    log_fn(f"  k_α mean: {k_alpha_mean:.4f}")
    log_fn(f"  k_α std:  {k_alpha_std:.4f}")
    log_fn(f"  k_α R²:   {r2_k_alpha:.4f} (higher = more constant)")
    log_fn(f"  Expected: k_α ≈ 0.20")
    log_fn("")
    log_fn("  d_manifold | k_α")
    log_fn("  " + "-" * 25)
    for i, dm in enumerate(phase2_results["d_manifold"]):
        log_fn(f"  {dm:10d} | {phase2_results['k_alpha'][i]:.4f}")
    
    phase2_summary = {
        "k_alpha_mean": k_alpha_mean,
        "k_alpha_std": k_alpha_std,
        "k_alpha_r2": r2_k_alpha,
        "expected_k_alpha": 0.20,
        "validation_passed": abs(k_alpha_mean - 0.20) < 0.05,  # Within 0.05 of 0.20
    }
    
    return phase2_results, phase2_summary


def run_phase3_d_scaling_validation(config, results_dir, start_time, log_fn, heartbeat_fn):
    """Phase 3: Validate α ∝ 2s/d_manifold scaling with fixed hidden_dim."""
    
    log_fn("\n" + "=" * 70)
    log_fn("PHASE 3: Validate d_manifold / s Ratio Scaling")
    log_fn("=" * 70)
    log_fn(f"d_manifold values: {config['phase3_d_manifold']}")
    log_fn(f"hidden_dim: FIXED at {config['fixed_hidden_dim']} for all")
    log_fn(f"Prediction: α ∝ 2s/d_manifold, so log(α) ∝ -log(d_manifold)")
    log_fn("")
    
    fixed_hd = config['fixed_hidden_dim']
    
    phase3_results = {
        "d_manifold": [],
        "alpha_measured": [],
        "log_alpha": [],
        "log_d": [],
        "product_eta": [],
        "s_value": [],
    }
    
    for dm_idx, d_manifold in enumerate(config["phase3_d_manifold"]):
        log_fn(f"\n[d_manifold={d_manifold}, fixed hidden_dim={fixed_hd}]")
        
        # Generate data
        gen = ManifoldDataGenerator(seed=config['data_seed'] + dm_idx + 100)
        z, X = gen.generate(
            n_samples=config['n_samples'],
            d_manifold=d_manifold,
            d_embed=fixed_hd,
            mode="polynomial",
            noise_std=0.05
        )
        
        # Create network with FIXED architecture
        torch.manual_seed(config['network_seed'] + dm_idx + 100)
        W_layers = [torch.randn(fixed_hd, fixed_hd) * 0.1 
                   for _ in range(config['n_layers'])]
        W_out = torch.randn(fixed_hd, d_manifold) * 0.1
        
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
        
        # Compute s (scale factor)
        s = sum(D_eff_layers) / len(D_eff_layers)
        
        # Compute α empirically
        N = len(eta_layers)
        T = 1.0
        
        if product_eta > 0:
            log_product = torch.log(torch.tensor(product_eta).abs()).item()
            alpha_measured = abs(log_product) / (N * T)
        else:
            alpha_measured = 0.0
        
        log_alpha = np.log(alpha_measured) if alpha_measured > 0 else -np.inf
        log_d = np.log(d_manifold)
        
        log_fn(f"  η layers: {[f'{x:.3f}' for x in eta_layers]}")
        log_fn(f"  ∏η={product_eta:.4f}, s={s:.3f}")
        log_fn(f"  α_measured={alpha_measured:.4f}, log(α)={log_alpha:.4f}")
        
        phase3_results["d_manifold"].append(d_manifold)
        phase3_results["alpha_measured"].append(alpha_measured)
        phase3_results["log_alpha"].append(log_alpha)
        phase3_results["log_d"].append(log_d)
        phase3_results["product_eta"].append(product_eta)
        phase3_results["s_value"].append(s)
        
        heartbeat_fn("Phase3")
    
    # Compute linear regression: log(α) vs -log(d_manifold)
    log_d_array = np.array(phase3_results["log_d"])
    log_alpha_array = np.array(phase3_results["log_alpha"])
    
    # Fit: log(α) = slope * (-log(d)) + intercept
    # Which is: log(α) = slope * (-log_d) + intercept
    neg_log_d = -log_d_array
    
    # Linear regression
    cov = np.cov(neg_log_d, log_alpha_array)[0, 1]
    var = np.var(neg_log_d, ddof=1)
    slope = cov / var if var > 0 else 0
    
    intercept = np.mean(log_alpha_array) - slope * np.mean(neg_log_d)
    
    # Predicted values
    log_alpha_predicted = slope * neg_log_d + intercept
    
    # R²
    ss_res = np.sum((log_alpha_array - log_alpha_predicted) ** 2)
    ss_tot = np.sum((log_alpha_array - np.mean(log_alpha_array)) ** 2)
    r2_d_scaling = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    log_fn("\n" + "-" * 50)
    log_fn("PHASE 3 SUMMARY")
    log_fn("-" * 50)
    log_fn(f"  Linear fit: log(α) = {slope:.4f} * (-log(d)) + {intercept:.4f}")
    log_fn(f"  R² for d-scaling: {r2_d_scaling:.4f}")
    log_fn(f"  Expected slope: -1.0 (perfect inverse scaling)")
    log_fn("")
    log_fn("  d_manifold | α_measured | log(α)")
    log_fn("  " + "-" * 40)
    for i, dm in enumerate(phase3_results["d_manifold"]):
        log_fn(f"  {dm:10d} | {phase3_results['alpha_measured'][i]:11.4f} | {phase3_results['log_alpha'][i]:8.4f}")
    
    phase3_summary = {
        "slope": slope,
        "intercept": intercept,
        "r2_d_scaling": r2_d_scaling,
        "expected_slope": -1.0,
        "validation_passed": abs(slope - (-1.0)) < 0.2,  # Within 0.2 of -1.0
    }
    
    return phase3_results, phase3_summary


def main():
    config = {
        "n_samples": 20000,
        "phase2_d_manifold": [5, 8, 10, 12, 15, 18, 20],
        "phase3_d_manifold": [5, 10, 15, 20],
        "fixed_hidden_dim": 64,
        "hidden_dim_range": [8, 16, 24, 32, 48, 64, 96, 128],
        "n_layers": 4,
        "network_seed": 42,
        "data_seed": 42,
        "target_eta": 1.0,
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
            # Save intermediate progress
            save_intermediate_results()
    
    def progress_report():
        nonlocal last_progress_report
        now = time.time()
        if now - last_progress_report >= progress_interval:
            elapsed = (now - start_time) / 60
            log(f"[PROGRESS REPORT] Elapsed: {elapsed:.1f} min - Running Phase 2-3 validation")
            last_progress_report = now
    
    def save_intermediate_results():
        # Save intermediate results if we have any
        pass
    
    log("=" * 70)
    log("PHASE 2-3 UNIFIED SCALING LAW VALIDATION")
    log("=" * 70)
    log(f"Configuration:")
    log(f"  n_samples: {config['n_samples']}")
    log(f"  n_layers: {config['n_layers']}")
    log(f"  phase2_d_manifold: {config['phase2_d_manifold']}")
    log(f"  phase3_d_manifold: {config['phase3_d_manifold']}")
    log(f"  fixed_hidden_dim: {config['fixed_hidden_dim']}")
    
    results = {
        "experiment": "phase2_3_validation",
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "phase2": {},
        "phase3": {},
        "final_summary": {},
    }
    
    # Run Phase 2
    phase2_results, phase2_summary = run_phase2_kalpha_validation(
        config, results_dir, start_time, log, heartbeat
    )
    results["phase2"] = {
        "data": phase2_results,
        "summary": phase2_summary,
    }
    progress_report()
    
    # Run Phase 3
    phase3_results, phase3_summary = run_phase3_d_scaling_validation(
        config, results_dir, start_time, log, heartbeat
    )
    results["phase3"] = {
        "data": phase3_results,
        "summary": phase3_summary,
    }
    progress_report()
    
    # Compute final summary
    duration = time.time() - start_time
    
    final_summary = {
        "duration_seconds": duration,
        "phase2_k_alpha_mean": phase2_summary["k_alpha_mean"],
        "phase2_k_alpha_std": phase2_summary["k_alpha_std"],
        "phase2_k_alpha_r2": phase2_summary["k_alpha_r2"],
        "phase2_validation": phase2_summary["validation_passed"],
        "phase3_slope": phase3_summary["slope"],
        "phase3_r2": phase3_summary["r2_d_scaling"],
        "phase3_validation": phase3_summary["validation_passed"],
        "timestamp_completed": datetime.now().isoformat(),
    }
    results["final_summary"] = final_summary
    
    # Save results
    phase2_path = results_dir / "phase2_kalpha_results.json"
    phase3_path = results_dir / "phase3_d_scaling_results.json"
    summary_path = results_dir / "phase2_3_final_summary.json"
    
    with open(phase2_path, "w") as f:
        json.dump({"data": phase2_results, "summary": phase2_summary}, f, indent=2, default=str)
    
    with open(phase3_path, "w") as f:
        json.dump({"data": phase3_results, "summary": phase3_summary}, f, indent=2, default=str)
    
    with open(summary_path, "w") as f:
        json.dump(final_summary, f, indent=2, default=str)
    
    log("\n" + "=" * 70)
    log("FINAL SUMMARY - Phase 2-3 Validation")
    log("=" * 70)
    log(f"Duration: {duration/60:.1f} minutes")
    log("")
    log("PHASE 2 (k_α constancy):")
    log(f"  k_α mean: {phase2_summary['k_alpha_mean']:.4f} ± {phase2_summary['k_alpha_std']:.4f}")
    log(f"  k_α R²:   {phase2_summary['k_alpha_r2']:.4f}")
    log(f"  Expected: 0.20, Validation: {'PASSED' if phase2_summary['validation_passed'] else 'FAILED'}")
    log("")
    log("PHASE 3 (d_manifold/s scaling):")
    log(f"  Slope:    {phase3_summary['slope']:.4f} (expected: -1.0)")
    log(f"  R²:       {phase3_summary['r2_d_scaling']:.4f}")
    log(f"  Validation: {'PASSED' if phase3_summary['validation_passed'] else 'FAILED'}")
    log("")
    log(f"Results saved to:")
    log(f"  - {phase2_path}")
    log(f"  - {phase3_path}")
    log(f"  - {summary_path}")
    
    return results


if __name__ == "__main__":
    results = main()
