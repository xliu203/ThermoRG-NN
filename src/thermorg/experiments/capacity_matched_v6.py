# SPDX-License-Identifier: Apache-2.0

"""Capacity-Matched Scaling Experiment v6 - FIXED.

Key fix: Search for matching hidden_dim using FULL multi-layer structure
(4 hidden layers + output layer), not just a single layer.

This ensures the product η_total = η_1 × η_2 × ... × η_5 is properly accounted for.
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


def find_matching_hidden_dim_full(d_manifold, target_eta=1.0, 
                                   hidden_dim_range=[8, 16, 24, 32, 48, 64, 96, 128],
                                   n_layers=4, seed=42):
    """Find hidden_dim that gives η_≈ 1.0 using FULL multi-layer structure."""
    torch.manual_seed(seed)
    
    best_hd = hidden_dim_range[0]
    best_eta = float('inf')
    best_product_eta = 1.0
    best_D_eff_avg = 0
    
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
        D_eff_layers.append(eta_out)
        
        # Product of all η (total compression)
        product_eta = 1.0
        for e in eta_layers:
            product_eta *= e
        
        avg_D_eff = sum(D_eff_layers) / len(D_eff_layers)
        avg_eta = sum(eta_layers) / len(eta_layers)
        
        print(f"      hd={hd:3d}: η_layers={[f'{x:.2f}' for x in eta_layers]}, "
              f"∏η={product_eta:.3f}, η_avg={avg_eta:.2f}")
        
        # Find closest to target using product_eta
        dist = abs(product_eta - target_eta)
        
        if dist < best_eta:
            best_eta = dist
            best_hd = hd
            best_product_eta = product_eta
            best_D_eff_avg = avg_D_eff
    
    return best_hd, best_product_eta, best_D_eff_avg, avg_eta


def main():
    config = {
        "n_samples": 20000,
        "d_manifold_values": [3, 5, 8, 10, 12, 15, 18, 20],
        "hidden_dim_range": [8, 16, 24, 32, 48, 64, 96, 128],
        "n_layers": 4,
        "network_seed": 42,
        "data_seed": 42,
        "target_eta": 1.0,  # We want ∏η ≈ 1 (balanced compression)
    }
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    start_time = time.time()
    last_heartbeat = start_time
    heartbeat_interval = 300
    
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
    log("CAPACITY-MATCHED SCALING EXPERIMENT v6 (FIXED)")
    log("=" * 70)
    log(f"Target: ∏η ≈ 1.0 for each d_manifold (FULL multi-layer search)")
    log(f"d_manifold values: {config['d_manifold_values']}")
    log(f"hidden_dim search: {config['hidden_dim_range']}")
    log(f"n_layers: {config['n_layers']} + 1 output layer")
    
    results = {
        "experiment": "capacity_matched_v6",
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "stage1_matching": {},
        "stage1_full": {},
        "stage2": {},
        "data": [],
    }
    
    # Stage 1a: Find matching hidden_dim using FULL structure
    log("\n" + "=" * 50)
    log("STAGE 1a: Find matching hidden_dim (FULL multi-layer)")
    log("=" * 50)
    
    matching_results = {}
    
    for dm_idx, d_manifold in enumerate(config["d_manifold_values"]):
        log(f"\n[d_manifold={d_manifold}] Searching with FULL structure...")
        
        best_hd, best_prod_eta, best_D_eff, avg_eta = find_matching_hidden_dim_full(
            d_manifold,
            target_eta=config["target_eta"],
            hidden_dim_range=config["hidden_dim_range"],
            n_layers=config["n_layers"],
            seed=config["network_seed"] + dm_idx
        )
        
        matching_results[d_manifold] = {
            "matched_hidden_dim": best_hd,
            "matched_product_eta": best_prod_eta,
            "matched_D_eff": best_D_eff,
        }
        
        log(f"  → Best: hidden_dim={best_hd}, ∏η={best_prod_eta:.3f}")
        
        heartbeat(f"Matching d={d_manifold}")
    
    results["stage1_matching"] = matching_results
    
    # Stage 1b: Full measurement with matched networks
    log("\n" + "=" * 50)
    log("STAGE 1b: Full measurement with matched networks")
    log("=" * 50)
    
    full_results = {}
    
    for dm_idx, d_manifold in enumerate(config["d_manifold_values"]):
        matched_hd = matching_results[d_manifold]["matched_hidden_dim"]
        
        log(f"\n[d_manifold={d_manifold}, matched hidden_dim={matched_hd}]")
        
        # Generate data
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
        
        log(f"  d_est: eff={d_est_eff:.1f}, pca={d_est_pca:.1f}")
        log(f"  η layers: {[f'{x:.3f}' for x in eta_layers]}")
        log(f"  ∏η={product_eta:.3f}, η_avg={avg_eta:.3f}")
        
        full_results[d_manifold] = {
            "d_manifold_true": d_manifold,
            "matched_hidden_dim": matched_hd,
            "d_est": {
                "effective": d_est_eff,
                "pca": d_est_pca,
            },
            "eta_layers": eta_layers,
            "D_eff_layers": D_eff_layers,
            "avg_eta": avg_eta,
            "avg_D_eff": avg_D_eff,
            "product_eta": product_eta,
        }
        
        heartbeat(f"Full measurement d={d_manifold}")
    
    results["stage1_full"] = full_results
    
    # Stage 2: α from matched ∏η
    log("\n" + "=" * 50)
    log("STAGE 2: α from matched ∏η")
    log("=" * 50)
    
    # α = |log(∏η)| / (N * T) at T=1
    stage2_data = {"d_manifold": [], "product_eta": [], "alpha": [], "avg_eta": []}
    
    for d_manifold in config["d_manifold_values"]:
        res = full_results[d_manifold]
        product_eta = res["product_eta"]
        avg_eta = res["avg_eta"]
        
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
        
        log(f"  d={d_manifold}: ∏η={product_eta:.3f}, α={alpha:.4f}, η_avg={avg_eta:.3f}")
    
    results["stage2"] = stage2_data
    
    # Compute statistics
    import numpy as np
    prod_etas = np.array(stage2_data["product_eta"])
    avg_etas = np.array(stage2_data["avg_eta"])
    
    # Deviation from ∏η = 1
    log_deviation = np.abs(np.log(prod_etas + 1e-10))
    avg_log_deviation = np.mean(log_deviation)
    
    # Deviation from η_avg = 1
    eta_deviation = np.abs(avg_etas - 1.0)
    avg_eta_deviation = np.mean(eta_deviation)
    
    duration = time.time() - start_time
    results["summary"] = {
        "duration_seconds": duration,
        "avg_log_deviation_from_1": avg_log_deviation,
        "avg_eta_deviation_from_1": avg_eta_deviation,
        "timestamp_completed": datetime.now().isoformat(),
    }
    
    # Save
    with open(results_dir / "v6_final_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    log("\n" + "=" * 70)
    log("FINAL SUMMARY")
    log("=" * 70)
    log(f"Duration: {duration/60:.1f} minutes")
    log(f"\nCapacity matching results (∏η should ≈ 1):")
    log(f"  d | hd | ∏η | η_avg | α")
    log(f"  " + "-" * 45)
    for dm in config["d_manifold_values"]:
        hd = matching_results[dm]["matched_hidden_dim"]
        prod = full_results[dm]["product_eta"]
        avg = full_results[dm]["avg_eta"]
        alp = stage2_data["alpha"][stage2_data["d_manifold"].index(dm)]
        log(f"  {dm:2d} | {hd:3d} | {prod:6.3f} | {avg:5.3f} | {alp:.4f}")
    log(f"\nAverage |log(∏η)| deviation from 0: {avg_log_deviation:.3f}")
    log(f"Average η_avg deviation from 1: {avg_eta_deviation:.3f}")
    log(f"\nResults saved to results/v6_final_results.json")
    
    return results


if __name__ == "__main__":
    results = main()
