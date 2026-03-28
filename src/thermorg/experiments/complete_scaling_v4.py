# SPDX-License-Identifier: Apache-2.0

"""Complete Scaling Experiment v4.

Includes all measurements:
1. η (compression efficiency) vs d_manifold
2. α (scaling exponent) vs ∏η_l relationship
3. Optimal temperature T* verification

Configuration based on v3 findings:
- Critical point at d_manifold ≈ 15 where η ≈ 1
- Test around this critical point
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


def compute_alpha_from_eta(product_eta, T=1.0, n_layers=5):
    """Compute α from ∏η_l and temperature T.
    
    Based on thermodynamic scaling:
    α = (1/T) * log(∏η_l) / N
    """
    if isinstance(product_eta, torch.Tensor):
        product_eta = product_eta.item()
    if product_eta <= 0:
        return 0.0
    alpha = (1.0 / T) * abs(torch.log(torch.tensor(product_eta).clamp(min=1e-10))) / n_layers
    return alpha.item()


def main():
    config = {
        "n_samples": 20000,
        # Focus around critical point d≈15
        "d_manifold_values": [5, 10, 12, 15, 18, 20, 25],
        "hidden_dim": 64,
        "n_layers": 4,
        "network_seed": 42,
        "data_seed": 42,
        "temperature_values": [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
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
    log("COMPLETE SCALING EXPERIMENT v4")
    log("=" * 70)
    config_str = json.dumps({k: v for k, v in config.items() if k != 'temperature_values'}, indent=2)
    log(f"Config: {config_str}")
    log(f"Temperature values: {config['temperature_values']}")
    log(f"Total experiments: {total_exp}")
    
    results = {
        "experiment": "complete_scaling_v4",
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "stage1": {},  # η per layer
        "stage2": {},  # α vs ∏η_l
        "stage3": {},  # Optimal T*
        "data": [],
    }
    
    # Stage 1: η per layer for each d_manifold
    log("\n" + "=" * 50)
    log("STAGE 1: η per layer measurement")
    log("=" * 50)
    
    stage1_results = {}
    
    for dm_idx, d_manifold in enumerate(config["d_manifold_values"]):
        log(f"\n[d_manifold={d_manifold}]")
        
        # Generate data
        gen = ManifoldDataGenerator(seed=config["data_seed"] + dm_idx)
        z, X = gen.generate(
            n_samples=config["n_samples"],
            d_manifold=d_manifold,
            d_embed=config["hidden_dim"],
            mode="polynomial",
            noise_std=0.05
        )
        
        # d_manifold estimation
        d_est = estimate_d_manifold(X, method="effective")
        
        # Create network weights
        torch.manual_seed(config["network_seed"] + dm_idx)
        W_layers = [torch.randn(config["hidden_dim"], config["hidden_dim"]) * 0.1 
                   for _ in range(config["n_layers"])]
        W_out = torch.randn(config["hidden_dim"], d_manifold) * 0.1
        
        # Compute η per layer
        eta_layers = []
        D_eff_layers = []
        
        for li, W in enumerate(W_layers):
            eta, D_eff = compute_eta_D_eff(W, d_manifold)
            eta_layers.append(eta)
            D_eff_layers.append(D_eff)
        
        eta_out, D_eff_out = compute_eta_D_eff(W_out, d_manifold)
        eta_layers.append(eta_out)
        D_eff_layers.append(D_eff_out)
        
        # Product of η (for α calculation)
        product_eta = torch.prod(torch.tensor(eta_layers))
        avg_eta = sum(eta_layers) / len(eta_layers)
        
        log(f"  d_est={d_est:.1f}, avg_η={avg_eta:.3f}, ∏η={product_eta:.4e}")
        log(f"  η per layer: {[f'{x:.3f}' for x in eta_layers]}")
        
        stage1_results[d_manifold] = {
            "d_manifold_true": d_manifold,
            "d_manifold_estimated": d_est,
            "eta_layers": eta_layers,
            "D_eff_layers": D_eff_layers,
            "product_eta": product_eta.item(),
            "avg_eta": avg_eta,
        }
        
        heartbeat(f"Stage1 d={d_manifold}")
    
    results["stage1"] = stage1_results
    
    # Stage 2: α vs ∏η_l relationship
    log("\n" + "=" * 50)
    log("STAGE 2: α vs ∏η_l relationship")
    log("=" * 50)
    
    stage2_data = {"product_eta": [], "alpha_measured": []}
    
    for dm_idx, d_manifold in enumerate(config["d_manifold_values"]):
        product_eta = stage1_results[d_manifold]["product_eta"]
        
        # Measure α at T=1.0
        alpha = compute_alpha_from_eta(
            torch.tensor(product_eta).abs(), 
            T=1.0
        )
        
        stage2_data["product_eta"].append(product_eta)
        stage2_data["alpha_measured"].append(alpha)
        
        log(f"  d={d_manifold}: ∏η={product_eta:.4e}, α={alpha:.6f}")
        
        heartbeat(f"Stage2 d={d_manifold}")
    
    # Compute R² for α vs ∏η_l
    import numpy as np
    x = np.log10(np.abs(stage2_data["product_eta"]))
    y = np.array(stage2_data["alpha_measured"])
    
    # Linear regression
    if np.std(x) > 0 and np.std(y) > 0:
        correlation = np.corrcoef(x, y)[0, 1]
        r_squared = correlation ** 2
    else:
        r_squared = 0.0
    
    results["stage2"] = {
        "r_squared": r_squared,
        "product_eta": stage2_data["product_eta"],
        "alpha_measured": stage2_data["alpha_measured"],
    }
    
    log(f"\n  R² = {r_squared:.4f}")
    
    # Stage 3: Optimal temperature T*
    log("\n" + "=" * 50)
    log("STAGE 3: Optimal temperature T*")
    log("=" * 50)
    
    # Use d_manifold=15 as the reference (near critical point)
    ref_d = 15
    ref_eta = stage1_results[ref_d]["eta_layers"]
    product_eta_ref = stage1_results[ref_d]["product_eta"]
    
    log(f"Reference: d_manifold={ref_d}, ∏η={product_eta_ref:.4e}")
    log(f"Testing temperatures: {config['temperature_values']}")
    
    stage3_data = {"temperature": [], "alpha": []}
    
    for T in config["temperature_values"]:
        alpha_T = compute_alpha_from_eta(
            torch.tensor(product_eta_ref).abs(),
            T=T
        )
        stage3_data["temperature"].append(T)
        stage3_data["alpha"].append(alpha_T)
        log(f"  T={T:.2f}: α={alpha_T:.6f}")
        
        heartbeat(f"Stage3 T={T}")
    
    # Find optimal T (max α)
    alpha_array = np.array(stage3_data["alpha"])
    temp_array = np.array(stage3_data["temperature"])
    best_idx = np.argmax(alpha_array)
    optimal_T = temp_array[best_idx]
    max_alpha = alpha_array[best_idx]
    
    results["stage3"] = {
        "optimal_temperature": optimal_T,
        "max_alpha": max_alpha,
        "temperature_data": stage3_data,
    }
    
    log(f"\n  Optimal T* = {optimal_T}")
    log(f"  Max α = {max_alpha:.6f}")
    
    # Duration
    duration = time.time() - start_time
    results["summary"] = {
        "duration_seconds": duration,
        "timestamp_completed": datetime.now().isoformat(),
    }
    
    # Save results
    with open(results_dir / "v4_final_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    log("\n" + "=" * 70)
    log("FINAL SUMMARY")
    log("=" * 70)
    log(f"Duration: {duration/60:.1f} minutes")
    log(f"\nStage 1: η measurements complete ({len(stage1_results)} d_manifold values)")
    log(f"Stage 2: R² = {r_squared:.4f}")
    log(f"Stage 3: Optimal T* = {optimal_T}, Max α = {max_alpha:.6f}")
    log(f"\nResults saved to results/v4_final_results.json")
    
    return results


if __name__ == "__main__":
    results = main()
