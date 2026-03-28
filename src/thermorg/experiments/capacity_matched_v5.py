# SPDX-License-Identifier: Apache-2.0

"""Capacity-Matched Scaling Experiment v5.

Key insight from Leo:
- For d=15, hidden_dim=64 works well because network capacity ≈ data complexity
- For other d values, the mismatch causes poor η/d_est estimation
- We need to MATCH network capacity to d_manifold for each d

Design:
- For each d_manifold, find appropriate hidden_dim such that η ≈ 1
- This ensures the network has "right-sized" capacity for the data
- Then we can properly test if α vs ∏η theory holds

Stage 1: For each d, tune hidden_dim to get η ≈ 1
Stage 2: With matched capacities, measure α vs ∏η
"""

import time
import json
import torch
import sys
from pathlib import Path
from datetime import datetime

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


def find_matching_hidden_dim(d_manifold, target_eta=1.0, hidden_dim_range=[16, 32, 64, 128, 256], seed=42):
    """Find hidden_dim that gives η closest to target (usually 1.0)."""
    torch.manual_seed(seed)
    
    best_hd = hidden_dim_range[0]
    best_eta = float('inf')
    best_D_eff = 0
    
    for hd in hidden_dim_range:
        # Generate data
        gen = ManifoldDataGenerator(seed=seed)
        z, X = gen.generate(
            n_samples=10000,  # Quick estimation
            d_manifold=d_manifold,
            d_embed=hd,
            mode="polynomial",
            noise_std=0.05
        )
        
        # Create network weights
        torch.manual_seed(seed)
        W = torch.randn(hd, hd) * 0.1  # Single layer for quick test
        
        # Compute η
        eta, D_eff = compute_eta_D_eff(W, d_manifold)
        
        # Also estimate d_est
        d_est = estimate_d_manifold(X, method="effective")
        
        # Check distance from target
        dist = abs(eta - target_eta)
        
        print(f"      hd={hd}: η={eta:.3f}, D_eff={D_eff:.1f}, d_est={d_est:.1f}")
        
        if dist < best_eta:
            best_eta = eta
            best_hd = hd
            best_D_eff = D_eff
    
    return best_hd, best_eta, best_D_eff


def main():
    # Configuration
    config = {
        "n_samples": 20000,
        "d_manifold_values": [3, 5, 8, 10, 12, 15, 18, 20],  # Up to 20
        "hidden_dim_range": [8, 16, 24, 32, 48, 64, 96, 128],  # Search space
        "n_layers": 4,
        "network_seed": 42,
        "data_seed": 42,
        "target_eta": 1.0,  # We want η ≈ 1 for proper capacity matching
    }
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    start_time = time.time()
    last_heartbeat = start_time
    heartbeat_interval = 300  # 5 min
    
    def log(msg):
        elapsed = (time.time() - start_time) / 60
        print(f"[{elapsed:6.1f}min] {msg}")
    
    def heartbeat(category=""):
        nonlocal last_heartbeat
        now = time.time()
        if now - last_heartbeat >= heartbeat_interval:
            log(f"[HEARTBEAT] {category}")
            last_heartbeat = now
    
    total_exp = len(config["d_manifold_values"])
    
    log("=" * 70)
    log("CAPACITY-MATCHED SCALING EXPERIMENT v5")
    log("=" * 70)
    log(f"Target: η ≈ 1.0 for each d_manifold (proper capacity match)")
    log(f"d_manifold values: {config['d_manifold_values']}")
    log(f"hidden_dim search: {config['hidden_dim_range']}")
    log(f"Total experiments: {total_exp}")
    
    results = {
        "experiment": "capacity_matched_v5",
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "stage1_matching": {},  # Find best hidden_dim for each d
        "stage1_full": {},      # Full measurement with matched network
        "stage2": {},           # α vs ∏η with matched networks
        "data": [],
    }
    
    # Stage 1a: Find matching hidden_dim for each d_manifold
    log("\n" + "=" * 50)
    log("STAGE 1a: Find matching hidden_dim for each d")
    log("=" * 50)
    
    matching_results = {}
    
    for dm_idx, d_manifold in enumerate(config["d_manifold_values"]):
        log(f"\n[d_manifold={d_manifold}] Searching for matching hidden_dim...")
        
        best_hd, best_eta, best_D_eff = find_matching_hidden_dim(
            d_manifold,
            target_eta=config["target_eta"],
            hidden_dim_range=config["hidden_dim_range"],
            seed=config["network_seed"] + dm_idx
        )
        
        matching_results[d_manifold] = {
            "matched_hidden_dim": best_hd,
            "matched_eta": best_eta,
            "matched_D_eff": best_D_eff,
        }
        
        log(f"  → Best: hidden_dim={best_hd}, η={best_eta:.3f}")
        
        heartbeat(f"Matching d={d_manifold}")
    
    results["stage1_matching"] = matching_results
    
    # Stage 1b: Full measurement with matched networks
    log("\n" + "=" * 50)
    log("STAGE 1b: Full η measurement with matched networks")
    log("=" * 50)
    
    full_results = {}
    
    for dm_idx, d_manifold in enumerate(config["d_manifold_values"]):
        matched_hd = matching_results[d_manifold]["matched_hidden_dim"]
        
        log(f"\n[d_manifold={d_manifold}, matched hidden_dim={matched_hd}]")
        
        # Generate data with matched hidden_dim
        gen = ManifoldDataGenerator(seed=config["data_seed"] + dm_idx)
        z, X = gen.generate(
            n_samples=config["n_samples"],
            d_manifold=d_manifold,
            d_embed=matched_hd,
            mode="polynomial",
            noise_std=0.05
        )
        
        # Estimate d_manifold
        d_est_eff = estimate_d_manifold(X, method="effective")
        d_est_pca = estimate_d_manifold(X, method="pca")
        d_est_levina = estimate_d_manifold(X, method="levina")
        
        # Create matched network
        torch.manual_seed(config["network_seed"] + dm_idx)
        W_layers = [torch.randn(matched_hd, matched_hd) * 0.1 
                   for _ in range(config["n_layers"])]
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
        avg_D_eff = sum(D_eff_layers) / len(D_eff_layers)
        
        log(f"  d_est: eff={d_est_eff:.1f}, pca={d_est_pca:.1f}, levina={d_est_levina:.1f}")
        log(f"  η layers: {[f'{x:.3f}' for x in eta_layers]}")
        log(f"  avg_η={avg_eta:.3f}, ∏η={product_eta:.4e}")
        
        full_results[d_manifold] = {
            "d_manifold_true": d_manifold,
            "matched_hidden_dim": matched_hd,
            "d_est": {
                "effective": d_est_eff,
                "pca": d_est_pca,
                "levina": d_est_levina,
            },
            "eta_layers": eta_layers,
            "D_eff_layers": D_eff_layers,
            "avg_eta": avg_eta,
            "avg_D_eff": avg_D_eff,
            "product_eta": product_eta,
        }
        
        heartbeat(f"Full measurement d={d_manifold}")
    
    results["stage1_full"] = full_results
    
    # Stage 2: Compute α from matched ∏η
    log("\n" + "=" * 50)
    log("STAGE 2: α from matched ∏η (with proper capacity)")
    log("=" * 50)
    
    # Using formula: α = (1/T) * log(∏η) / N at T=1
    # With matched capacity, we expect ∏η ≈ 1 for all d
    
    stage2_data = {"d_manifold": [], "product_eta": [], "alpha": [], "avg_eta": []}
    
    for d_manifold in config["d_manifold_values"]:
        res = full_results[d_manifold]
        product_eta = res["product_eta"]
        avg_eta = res["avg_eta"]
        
        # α = log(∏η) / (N * T) where N = n_layers + 1
        N = len(res["eta_layers"])
        T = 1.0
        
        if product_eta > 0:
            alpha = abs(torch.log(torch.tensor(product_eta).abs()).item()) / (N * T)
        else:
            alpha = 0.0
        
        stage2_data["d_manifold"].append(d_manifold)
        stage2_data["product_eta"].append(product_eta)
        stage2_data["alpha"].append(alpha)
        stage2_data["avg_eta"].append(avg_eta)
        
        log(f"  d={d_manifold}: ∏η={product_eta:.4f}, α={alpha:.4f}, η_avg={avg_eta:.3f}")
    
    results["stage2"] = stage2_data
    
    # Compute correlation
    import numpy as np
    d_vals = np.array(stage2_data["d_manifold"])
    eta_vals = np.array(stage2_data["avg_eta"])
    
    # Check if η ≈ 1 for all d (with matched capacity)
    eta_deviation = np.abs(eta_vals - 1.0)
    avg_deviation = np.mean(eta_deviation)
    
    log(f"\n  Average deviation from η=1: {avg_deviation:.3f}")
    log(f"  (Lower is better - means capacity matching worked)")
    
    # Duration
    duration = time.time() - start_time
    results["summary"] = {
        "duration_seconds": duration,
        "avg_eta_deviation_from_1": avg_deviation,
        "timestamp_completed": datetime.now().isoformat(),
    }
    
    # Save results
    with open(results_dir / "v5_final_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    log("\n" + "=" * 70)
    log("FINAL SUMMARY")
    log("=" * 70)
    log(f"Duration: {duration/60:.1f} minutes")
    log(f"\nCapacity matching results:")
    log(f"  d | matched_hd | η (should≈1)")
    log(f"  " + "-" * 30)
    for dm in config["d_manifold_values"]:
        hd = matching_results[dm]["matched_hidden_dim"]
        eta = full_results[dm]["avg_eta"]
        log(f"  {dm:2d} | {hd:10d} | {eta:.3f}")
    log(f"\nAverage deviation from η=1: {avg_deviation:.3f}")
    log(f"\nResults saved to results/v5_final_results.json")
    
    return results


if __name__ == "__main__":
    results = main()
