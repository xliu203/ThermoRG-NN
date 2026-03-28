"""Evaluation and metrics for CIFAR-10 Lift Test.

This module provides functions for:
- Computing metrics (α/FLOPs, generalization gap, grokking timing)
- Correlation analysis (predicted α vs actual accuracy)
- Generating comparison tables and visualizations
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr, pearsonr


@dataclass
class ArchitectureMetrics:
    """Metrics for a single architecture."""
    name: str
    group: int
    phase: str
    
    # Model properties
    total_params: int
    total_flops: int
    
    # Accuracy metrics
    train_acc: float
    val_acc: float
    test_acc: float
    best_test_acc: float
    
    # Loss metrics
    train_loss: float
    val_loss: float
    test_loss: float
    
    # Generalization
    generalization_gap: float  # train_acc - test_acc
    
    # Grokking
    grokking_epoch: Optional[int]
    grokking_improvement: float
    
    # Computed α (thermal conductivity proxy)
    alpha: float  # = accuracy / log(FLOPs + 1)
    
    # Training time
    training_time: float


def load_results(results_dir: str) -> List[ArchitectureMetrics]:
    """Load training results from directory.
    
    Args:
        results_dir: Directory containing result.json files
        
    Returns:
        List of ArchitectureMetrics
    """
    results_dir = Path(results_dir)
    metrics_list = []
    
    for arch_dir in results_dir.iterdir():
        if not arch_dir.is_dir():
            continue
        
        result_file = arch_dir / "result.json"
        if not result_file.exists():
            continue
        
        with open(result_file) as f:
            data = json.load(f)
        
        # Compute generalization gap
        generalization_gap = data.get("final_train_acc", 0) - data.get("final_test_acc", 0)
        
        # Compute α = accuracy / log(FLOPs + 1)
        flops = data.get("total_flops", 1)
        test_acc = data.get("final_test_acc", 0)
        alpha = test_acc / np.log(flops + 1)
        
        metrics = ArchitectureMetrics(
            name=data["arch_name"],
            group=_get_group_from_name(data["arch_name"]),
            phase=data.get("phase", "A"),
            total_params=data.get("total_params", 0),
            total_flops=flops,
            train_acc=data.get("final_train_acc", 0),
            val_acc=data.get("final_val_acc", 0),
            test_acc=test_acc,
            best_test_acc=data.get("best_test_acc", 0),
            train_loss=data.get("final_train_loss", 0),
            val_loss=data.get("final_val_loss", 0),
            test_loss=data.get("final_test_loss", 0),
            generalization_gap=generalization_gap,
            grokking_epoch=data.get("grokking_epoch"),
            grokking_improvement=data.get("grokking_improvement", 0),
            alpha=alpha,
            training_time=data.get("training_time", 0)
        )
        metrics_list.append(metrics)
    
    return metrics_list


def _get_group_from_name(name: str) -> int:
    """Get group number from architecture name."""
    if "ThermoNet" in name:
        return 1
    elif "ThermoBot" in name:
        return 2
    elif "ReLUFurnace" in name:
        return 3
    elif "ResNet" in name or "VGG" in name or "DenseNet" in name:
        return 4
    return 0


def compute_metrics(metrics_list: List[ArchitectureMetrics]) -> Dict:
    """Compute summary metrics across all architectures.
    
    Args:
        metrics_list: List of ArchitectureMetrics
        
    Returns:
        Dictionary with summary statistics
    """
    if not metrics_list:
        return {}
    
    # Group by phase
    phase_a = [m for m in metrics_list if m.phase == "A"]
    phase_b = [m for m in metrics_list if m.phase == "B"]
    
    # Group by architecture group
    group_metrics = {}
    for m in metrics_list:
        if m.group not in group_metrics:
            group_metrics[m.group] = []
        group_metrics[m.group].append(m)
    
    summary = {
        "num_architectures": len(metrics_list),
        "phase_a_count": len(phase_a),
        "phase_b_count": len(phase_b),
    }
    
    # Overall statistics
    test_accs = [m.test_acc for m in metrics_list]
    summary["overall"] = {
        "mean_test_acc": np.mean(test_accs),
        "std_test_acc": np.std(test_accs),
        "max_test_acc": np.max(test_accs),
        "min_test_acc": np.min(test_accs),
    }
    
    # Per-group statistics
    for group, group_list in group_metrics.items():
        test_accs = [m.test_acc for m in group_list]
        summary[f"group_{group}"] = {
            "count": len(group_list),
            "mean_test_acc": np.mean(test_accs),
            "std_test_acc": np.std(test_accs),
            "max_test_acc": np.max(test_accs),
            "min_test_acc": np.min(test_accs),
        }
    
    # Grokking statistics
    grokking_archs = [m for m in metrics_list if m.grokking_epoch is not None]
    summary["grokking"] = {
        "count": len(grokking_archs),
        "percentage": 100 * len(grokking_archs) / len(metrics_list) if metrics_list else 0,
        "mean_epoch": np.mean([m.grokking_epoch for m in grokking_archs]) if grokking_archs else None,
    }
    
    return summary


def correlation_analysis(metrics_list: List[ArchitectureMetrics]) -> Dict:
    """Compute correlation between predicted α and actual accuracy.
    
    Also computes correlations between other model properties and accuracy.
    
    Args:
        metrics_list: List of ArchitectureMetrics
        
    Returns:
        Dictionary with correlation results
    """
    if not metrics_list:
        return {}
    
    results = {}
    
    # Extract data
    alphas = np.array([m.alpha for m in metrics_list])
    test_accs = np.array([m.test_acc for m in metrics_list])
    params = np.array([m.total_params for m in metrics_list])
    flops = np.array([m.total_flops for m in metrics_list])
    gen_gaps = np.array([m.generalization_gap for m in metrics_list])
    
    # α vs Test Accuracy
    if len(alphas) > 2:
        spearman_r, spearman_p = spearmanr(alphas, test_accs)
        pearson_r, pearson_p = pearsonr(alphas, test_accs)
        results["alpha_vs_test_acc"] = {
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
        }
    
    # FLOPs vs Test Accuracy
    if len(flops) > 2:
        spearman_r, spearman_p = spearmanr(flops, test_accs)
        pearson_r, pearson_p = pearsonr(flops, test_accs)
        results["flops_vs_test_acc"] = {
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
        }
    
    # Parameters vs Test Accuracy
    if len(params) > 2:
        spearman_r, spearman_p = spearmanr(params, test_accs)
        pearson_r, pearson_p = pearsonr(params, test_accs)
        results["params_vs_test_acc"] = {
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
        }
    
    # Generalization gap analysis
    if len(gen_gaps) > 2:
        spearman_r, spearman_p = spearmanr(gen_gaps, test_accs)
        results["gen_gap_vs_test_acc"] = {
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
        }
    
    return results


def generate_comparison_table(metrics_list: List[ArchitectureMetrics], sort_by: str = "test_acc") -> str:
    """Generate ASCII comparison table of architectures.
    
    Args:
        metrics_list: List of ArchitectureMetrics
        sort_by: Column to sort by (test_acc, alpha, params, etc.)
        
    Returns:
        ASCII table as string
    """
    if not metrics_list:
        return "No metrics to display"
    
    # Sort
    if sort_by == "test_acc":
        sorted_metrics = sorted(metrics_list, key=lambda m: m.test_acc, reverse=True)
    elif sort_by == "alpha":
        sorted_metrics = sorted(metrics_list, key=lambda m: m.alpha, reverse=True)
    elif sort_by == "params":
        sorted_metrics = sorted(metrics_list, key=lambda m: m.total_params, reverse=True)
    else:
        sorted_metrics = sorted(metrics_list, key=lambda m: m.name)
    
    # Table header
    header = (
        f"{'Arch':<20} {'Grp':>3} {'Test Acc':>8} {'Best':>8} {'Gen Gap':>8} "
        f"{'Params':>10} {'FLOPs':>12} {'α×1000':>10} {'Grok':>5} {'Time':>8}"
    )
    
    lines = [header]
    lines.append("-" * len(header))
    
    for m in sorted_metrics:
        grok_str = f"{m.grokking_epoch}" if m.grokking_epoch else "-"
        alpha_scaled = m.alpha * 1000  # Scale for display
        
        line = (
            f"{m.name:<20} {m.group:>3} {m.test_acc:>7.2f}% {m.best_test_acc:>7.2f}% "
            f"{m.generalization_gap:>7.2f}% {m.total_params:>10,} {m.total_flops:>12,} "
            f"{alpha_scaled:>10.4f} {grok_str:>5} {m.training_time:>7.1f}s"
        )
        lines.append(line)
    
    return "\n".join(lines)


def rank_architectures(metrics_list: List[ArchitectureMetrics], by: str = "test_acc") -> List[Tuple[str, float]]:
    """Rank architectures by a metric.
    
    Args:
        metrics_list: List of ArchitectureMetrics
        by: Metric to rank by (test_acc, alpha, etc.)
        
    Returns:
        List of (name, value) tuples sorted by rank
    """
    if by == "test_acc":
        return [(m.name, m.test_acc) for m in sorted(metrics_list, key=lambda x: x.test_acc, reverse=True)]
    elif by == "alpha":
        return [(m.name, m.alpha) for m in sorted(metrics_list, key=lambda x: x.alpha, reverse=True)]
    elif by == "generalization_gap":
        return [(m.name, m.generalization_gap) for m in sorted(metrics_list, key=lambda x: x.generalization_gap)]
    elif by == "grokking_epoch":
        grokking = [(m.name, m.grokking_epoch if m.grokking_epoch else float('inf')) 
                    for m in metrics_list]
        return sorted(grokking, key=lambda x: x[1])
    else:
        return [(m.name, 0) for m in metrics_list]


def print_summary(metrics_list: List[ArchitectureMetrics]):
    """Print summary of results.
    
    Args:
        metrics_list: List of ArchitectureMetrics
    """
    if not metrics_list:
        print("No results to summarize")
        return
    
    print("\n" + "=" * 80)
    print("CIFAR-10 LIFT TEST RESULTS SUMMARY")
    print("=" * 80)
    
    # Compute summary statistics
    summary = compute_metrics(metrics_list)
    
    # Overall
    print(f"\nOverall ({summary['num_architectures']} architectures):")
    print(f"  Mean Test Accuracy: {summary['overall']['mean_test_acc']:.2f}% ± {summary['overall']['std_test_acc']:.2f}%")
    print(f"  Best Test Accuracy: {summary['overall']['max_test_acc']:.2f}%")
    print(f"  Worst Test Accuracy: {summary['overall']['min_test_acc']:.2f}%")
    
    # Per-group
    print("\nPer-Group Performance:")
    for group in sorted([k for k in summary.keys() if k.startswith("group_")]):
        group_num = group.split("_")[1]
        group_data = summary[group]
        group_names = {
            1: "Thermogeometric Optimal",
            2: "Topology Destroyer",
            3: "Thermal Boiling Furnace",
            4: "Traditional Baselines"
        }
        print(f"  G{group_num} ({group_names.get(int(group_num), 'Unknown')}): "
              f"{group_data['mean_test_acc']:.2f}% ± {group_data['std_test_acc']:.2f}%")
    
    # Grokking
    grokking = summary.get("grokking", {})
    if grokking.get("count", 0) > 0:
        print(f"\nGrokking: {grokking['count']} architectures ({grokking['percentage']:.1f}%)")
        print(f"  Mean Grokking Epoch: {grokking['mean_epoch']:.1f}")
    
    # Correlation analysis
    correlations = correlation_analysis(metrics_list)
    if "alpha_vs_test_acc" in correlations:
        print("\nCorrelation: α vs Test Accuracy")
        print(f"  Spearman ρ = {correlations['alpha_vs_test_acc']['spearman_r']:.4f} (p={correlations['alpha_vs_test_acc']['spearman_p']:.4f})")
    
    # Full table
    print("\n" + "=" * 80)
    print(generate_comparison_table(metrics_list))
    print("=" * 80)


def main():
    """Main evaluation entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CIFAR-10 Lift Test Evaluation")
    parser.add_argument("--results_dir", type=str, 
                        default="./experiments/lift_test/results",
                        help="Directory containing training results")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output file for results (JSON)")
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.results_dir}...")
    metrics_list = load_results(args.results_dir)
    
    if not metrics_list:
        print("No results found!")
        return
    
    print(f"Loaded {len(metrics_list)} architecture results")
    
    # Print summary
    print_summary(metrics_list)
    
    # Save if output specified
    if args.output_file:
        correlations = correlation_analysis(metrics_list)
        summary = compute_metrics(metrics_list)
        
        output = {
            "summary": summary,
            "correlations": correlations,
            "architectures": [
                {
                    "name": m.name,
                    "group": m.group,
                    "phase": m.phase,
                    "test_acc": m.test_acc,
                    "alpha": m.alpha,
                    "generalization_gap": m.generalization_gap,
                }
                for m in metrics_list
            ]
        }
        
        with open(args.output_file, "w") as f:
            json.dump(output, f, indent=2)
        
        print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
