# SPDX-License-Identifier: Apache-2.0

"""Systematic Scaling Experiment v3.

Based on DeepSeek Reasoner recommendations:
- Use Effective/Spectral methods as primary d_manifold estimation
- Vary hidden_dim: 32, 64, 128, 256
- Systematically scan d_manifold: [1, 2, 5, 10, 15, 20, 25, 30]
- Use 4 layers
- Proper heartbeat (5 min) and progress reports (20 min)
"""

import time
import json
import torch
import sys
from pathlib import Path
from datetime import datetime
from itertools import product

sys.path.insert(0, 'src')

from thermorg.simulations.manifold_data import ManifoldDataGenerator
from thermorg.core import estimate_d_manifold


def compute_eta_from_jacobian(J, d_manifold):
    """Compute η = D_eff / d_manifold from Jacobian matrix.
    
    D_eff = ||J||_F^2 / ||J||_2^2
    """
    fro_sq = (J ** 2).sum()
    S = torch.linalg.svd(J, full_matrices=False)[1]
    spec_sq = S[0] ** 2
    D_eff = fro_sq / (spec_sq + 1e-8)
    eta = D_eff / d_manifold
    return eta.item(), D_eff.item()


def main():
    # Configuration - Systematic scan
    config = {
        "n_samples": 20000,
        "d_manifold_values": [1, 2, 5, 10, 15, 20, 25, 30],
        "hidden_dim_values": [32, 64, 128, 256],
        "n_layers": 4,
        "network_seed": 42,
        "data_seed": 42,
        "noise_std": 0.05,
        "mode": "polynomial",
    }
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Timing
    start_time = time.time()
    last_heartbeat = start_time
    last_progress = start_time
    heartbeat_interval = 300  # 5 minutes
    progress_interval = 1200  # 20 minutes
    experiment_start = start_time
    
    def log(msg):
        elapsed = (time.time() - start_time) / 60
        print(f"[{elapsed:6.1f}min] {msg}")
    
    def heartbeat(category=""):
        nonlocal last_heartbeat
        now = time.time()
        if now - last_heartbeat >= heartbeat_interval:
            elapsed = (now - experiment_start) / 60
            log(f"[HEARTBEAT {elapsed:.0f}min] {category}")
            last_heartbeat = now
            return True
        return False
    
    total_experiments = len(config["d_manifold_values"]) * len(config["hidden_dim_values"])
    current_exp = 0
    
    log("=" * 70)
    log("Systematic Scaling Experiment v3")
    log("=" * 70)
    log(f"Config: {json.dumps(config, indent=2)}")
    log(f"Total experiments: {total_experiments}")
    
    # Results storage
    results = {
        "experiment": "systematic_scaling_v3",
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "data": [],
        "summary": {},
    }
    
    # Iterate over all combinations
    for dm_idx, d_manifold in enumerate(config["d_manifold_values"]):
        for hd_idx, hidden_dim in enumerate(config["hidden_dim_values"]):
            current_exp += 1
            elapsed_total = (time.time() - experiment_start) / 60
            
            # Progress report every 20 minutes
            now = time.time()
            if now - last_progress >= progress_interval:
                eta_progress = elapsed_total / current_exp
                eta_remaining = eta_progress * (total_experiments - current_exp)
                log(f"\n{'='*60}")
                log(f"[PROGRESS REPORT - {elapsed_total:.0f}min]")
                log(f"  Completed: {current_exp}/{total_experiments}")
                log(f"  ETA: ~{eta_remaining:.0f} min")
                log(f"  Current: d_manifold={d_manifold}, hidden_dim={hidden_dim}")
                log("=" * 60)
                last_progress = now
            
            log(f"\n[{current_exp}/{total_experiments}] d_manifold={d_manifold}, hidden_dim={hidden_dim}")
            
            # Generate data
            torch.manual_seed(config["data_seed"] + dm_idx * 100 + hd_idx)
            gen = ManifoldDataGenerator(seed=config["data_seed"] + dm_idx * 100 + hd_idx)
            z, X_embedded = gen.generate(
                n_samples=config["n_samples"],
                d_manifold=d_manifold,
                d_embed=hidden_dim,
                mode=config["mode"],
                noise_std=config["noise_std"]
            )
            
            # Estimate d_manifold from embedded data (use Effective as primary)
            d_effective = estimate_d_manifold(X_embedded, method="effective")
            d_spectral = estimate_d_manifold(X_embedded, method="spectral_decay")
            d_pca = estimate_d_manifold(X_embedded, method="pca")
            d_levina = estimate_d_manifold(X_embedded, method="levina")
            
            heartbeat(f"Data generated for d={d_manifold}, hd={hidden_dim}")
            
            # Create network weights
            torch.manual_seed(config["network_seed"] + dm_idx * 100 + hd_idx)
            
            W_layers = []
            for layer_idx in range(config["n_layers"]):
                W = torch.randn(hidden_dim, hidden_dim) * 0.1
                W_layers.append(W)
            
            # Output weight
            W_out = torch.randn(hidden_dim, d_manifold) * 0.1
            
            # Compute η per layer
            eta_layers = []
            D_eff_layers = []
            
            for layer_idx, W in enumerate(W_layers):
                eta_l, D_eff_l = compute_eta_from_jacobian(W, d_manifold)
                eta_layers.append(eta_l)
                D_eff_layers.append(D_eff_l)
            
            # Output layer
            eta_out, D_eff_out = compute_eta_from_jacobian(W_out, d_manifold)
            eta_layers.append(eta_out)
            D_eff_layers.append(D_eff_out)
            
            avg_eta = sum(eta_layers) / len(eta_layers)
            avg_D_eff = sum(D_eff_layers) / len(D_eff_layers)
            
            # Store result
            result_entry = {
                "d_manifold_true": d_manifold,
                "hidden_dim": hidden_dim,
                "d_manifold_estimated": {
                    "effective": d_effective,
                    "spectral": d_spectral,
                    "pca": d_pca,
                    "levina": d_levina,
                },
                "eta_per_layer": eta_layers,
                "D_eff_per_layer": D_eff_layers,
                "avg_eta": avg_eta,
                "avg_D_eff": avg_D_eff,
            }
            results["data"].append(result_entry)
            
            log(f"  d_est: eff={d_effective:.1f}, pca={d_pca:.1f}, levina={d_levina:.1f}")
            log(f"  η layers: {[f'{x:.2f}' for x in eta_layers]}, avg={avg_eta:.3f}")
            log(f"  D_eff: {avg_D_eff:.1f}")
            
            heartbeat(f"Completed d={d_manifold}, hd={hidden_dim}")
            
            # Save progress every 10 experiments
            if current_exp % 10 == 0:
                with open(results_dir / "v3_progress.json", "w") as f:
                    json.dump(results, f, indent=2, default=str)
    
    # Compute final statistics
    duration = time.time() - experiment_start
    
    # Summary analysis
    log("\n" + "=" * 70)
    log("FINAL RESULTS SUMMARY")
    log("=" * 70)
    
    # Group by d_manifold
    by_d_manifold = {}
    for entry in results["data"]:
        dm = entry["d_manifold_true"]
        if dm not in by_d_manifold:
            by_d_manifold[dm] = []
        by_d_manifold[dm].append(entry)
    
    log("\n[η vs d_manifold (fixed hidden_dim=64)]")
    log("-" * 50)
    for dm in config["d_manifold_values"]:
        entries = [e for e in by_d_manifold[dm] if e["hidden_dim"] == 64]
        if entries:
            avg_eta = entries[0]["avg_eta"]
            d_est = entries[0]["d_manifold_estimated"]["effective"]
            log(f"  d={dm:2d}: η={avg_eta:.3f}, d_eff={d_est:.1f}")
    
    # Group by hidden_dim
    by_hidden_dim = {}
    for entry in results["data"]:
        hd = entry["hidden_dim"]
        if hd not in by_hidden_dim:
            by_hidden_dim[hd] = []
        by_hidden_dim[hd].append(entry)
    
    log("\n[D_eff vs hidden_dim (fixed d_manifold=10)]")
    log("-" * 50)
    for hd in config["hidden_dim_values"]:
        entries = [e for e in by_hidden_dim[hd] if e["d_manifold_true"] == 10]
        if entries:
            avg_D_eff = entries[0]["avg_D_eff"]
            avg_eta = entries[0]["avg_eta"]
            log(f"  hd={hd:3d}: D_eff={avg_D_eff:.1f}, η={avg_eta:.3f}")
    
    # Save final results
    results["summary"] = {
        "duration_seconds": duration,
        "total_experiments": total_experiments,
        "timestamp_completed": datetime.now().isoformat(),
    }
    
    with open(results_dir / "v3_final_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    log(f"\nDuration: {duration/60:.1f} minutes")
    log(f"Results saved to results/v3_final_results.json")
    
    return results


if __name__ == "__main__":
    results = main()
