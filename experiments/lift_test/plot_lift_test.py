"""
plot_lift_test.py - Money Plots Generator for CIFAR-10 Thermogeometric Lift Test
==================================================================================
Generates three key publication-quality figures:

1. Spearman Correlation Scatter: predicted α vs actual test accuracy
2. Grokking Dynamics: Test loss curves comparing G1 vs G3
3. Pareto Front: (T_c, γ_c) space with feasible region

Author: Leo Liu / ThermoRG-NN Team
Date: 2026-03-28
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import stats
from scipy.spatial import ConvexHull
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import warnings

# =============================================================================
# Style Configuration
# =============================================================================

# Publication-quality plot settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Group colors (colorblind-friendly palette)
GROUP_COLORS = {
    1: '#2ecc71',  # Green - Thermogeometric Optimal
    2: '#e74c3c',  # Red - Topology Destroyer
    3: '#f39c12',  # Orange - Thermal Furnace
    4: '#3498db',  # Blue - Traditional Baselines
}

GROUP_MARKERS = {
    1: 'o',   # Circle
    2: 's',   # Square
    3: '^',   # Triangle
    4: 'D',   # Diamond
}

GROUP_LABELS = {
    1: 'G1: Thermo. Optimal',
    2: 'G2: Topology Destroyer',
    3: 'G3: Thermal Furnace',
    4: 'G4: Traditional',
}


# =============================================================================
# Helper Functions
# =============================================================================

def linear_fit(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Fit a linear model y = mx + b and return fitted y, slope, intercept.
    """
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    y_fit = slope * x + intercept
    return y_fit, slope, intercept


def get_group_stats(
    df: pd.DataFrame,
    group_col: str = 'group'
) -> Dict[int, Dict]:
    """Compute per-group statistics for error bands."""
    stats_dict = {}
    for group_id in df[group_col].unique():
        group_df = df[df[group_col] == group_id]
        stats_dict[group_id] = {
            'mean': group_df.groupby('epoch')['test_loss'].mean(),
            'std': group_df.groupby('epoch')['test_loss'].std(),
            'count': len(group_df),
        }
    return stats_dict


# =============================================================================
# Plot 1: Spearman Correlation Scatter
# =============================================================================

def plot_correlation(
    df: pd.DataFrame,
    save_path: str,
    x_col: str = 'predicted_alpha',
    y_col: str = 'final_test_acc',
    title: str = "Predicted Complexity Exponent vs. Test Accuracy",
    xlabel: str = r"Predicted $\alpha$ (Complexity Exponent)",
    ylabel: str = "Final Test Accuracy",
    figsize: Tuple[float, float] = (7, 5.5)
) -> Dict:
    """
    Generate Spearman correlation scatter plot.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with predicted_alpha and final_test_acc columns.
    save_path : str
        Path to save the figure.
    x_col : str
        Column name for x-axis.
    y_col : str
        Column name for y-axis.
    title : str
        Plot title.
    xlabel, ylabel : str
        Axis labels.
    figsize : Tuple
        Figure size in inches.
    
    Returns
    -------
    Dict
        Correlation statistics.
    """
    # Filter valid data
    valid_df = df.dropna(subset=[x_col, y_col, 'group'])
    
    if len(valid_df) < 3:
        warnings.warn("Insufficient data for correlation plot")
        return {}
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each group
    for group_id in sorted(valid_df['group'].unique()):
        group_df = valid_df[valid_df['group'] == group_id]
        color = GROUP_COLORS.get(group_id, 'gray')
        marker = GROUP_MARKERS.get(group_id, 'o')
        
        ax.scatter(
            group_df[x_col],
            group_df[y_col],
            c=color,
            marker=marker,
            s=80,
            alpha=0.8,
            edgecolors='white',
            linewidths=0.5,
            label=GROUP_LABELS.get(group_id, f'Group {group_id}'),
            zorder=3
        )
    
    # Compute correlation statistics
    spearman_rho, spearman_p = stats.spearmanr(valid_df[x_col], valid_df[y_col])
    pearson_r, pearson_p = stats.pearsonr(valid_df[x_col], valid_df[y_col])
    
    # Linear fit
    x_vals = valid_df[x_col].values
    y_vals = valid_df[y_col].values
    y_fit, slope, intercept = linear_fit(x_vals, y_vals)
    
    # Sort for plotting
    sort_idx = np.argsort(x_vals)
    x_sorted = x_vals[sort_idx]
    y_fit_sorted = y_fit[sort_idx]
    
    # Plot fit line
    ax.plot(x_sorted, y_fit_sorted, 'k--', linewidth=1.5, alpha=0.7, zorder=2)
    
    # R² annotation
    r_squared = pearson_r ** 2
    stats_text = (
        r"$\rho_s = {:.3f}$".format(spearman_rho) + "\n"
        r"$R^2 = {:.3f}$".format(r_squared) + "\n"
        r"$p = {:.2e}$".format(spearman_p)
    )
    
    # Position text box
    text_x, text_y = 0.05, 0.95
    ax.text(
        text_x, text_y,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='gray')
    )
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='lower right', framealpha=0.9, edgecolor='gray')
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_axisbelow(True)
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved correlation plot: {save_path}")
    
    return {
        'spearman_rho': spearman_rho,
        'spearman_pvalue': spearman_p,
        'pearson_r': pearson_r,
        'pearson_pvalue': pearson_p,
        'r_squared': r_squared,
        'slope': slope,
        'intercept': intercept,
    }


# =============================================================================
# Plot 2: Grokking Dynamics
# =============================================================================

def plot_grokking_dynamics(
    df: pd.DataFrame,
    save_path: str,
    groups_to_compare: List[int] = [1, 3],
    x_col: str = 'epoch',
    y_col: str = 'test_loss',
    title: str = "Grokking Dynamics: G1 (Optimal) vs. G3 (Furnace)",
    xlabel: str = "Epoch",
    ylabel: str = "Test Loss",
    figsize: Tuple[float, float] = (8, 5.5)
) -> None:
    """
    Generate grokking dynamics plot comparing test loss curves.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    compare_colors = {
        groups_to_compare[0]: GROUP_COLORS[groups_to_compare[0]],
        groups_to_compare[1]: GROUP_COLORS[groups_to_compare[1]],
    }
    
    for group_id in groups_to_compare:
        group_df = df[df['group'] == group_id]
        
        if len(group_df) == 0:
            warnings.warn(f"No data for group {group_id}")
            continue
        
        epoch_stats = group_df.groupby(x_col)[y_col].agg(['mean', 'std', 'count'])
        epoch_stats = epoch_stats.reset_index()
        
        epochs = epoch_stats[x_col].values
        mean_loss = epoch_stats['mean'].values
        std_loss = epoch_stats['std'].values
        n_seeds = epoch_stats['count'].values
        se_loss = std_loss / np.sqrt(n_seeds)
        
        color = compare_colors[group_id]
        
        ax.plot(epochs, mean_loss, color=color, linewidth=2,
                label=f"{GROUP_LABELS[group_id]}", zorder=3)
        
        ax.fill_between(epochs, mean_loss - se_loss, mean_loss + se_loss,
                        color=color, alpha=0.2, zorder=1)
        
        # Mark grokking transition
        if len(epochs) > 5:
            gradients = np.gradient(mean_loss, epochs)
            early_mask = epochs < 50
            if early_mask.any():
                early_gradients = np.abs(gradients[early_mask])
                early_epochs = epochs[early_mask]
                max_grad_idx = np.argmax(early_gradients)
                transition_epoch = early_epochs[max_grad_idx]
                transition_loss = mean_loss[early_mask][max_grad_idx]
                
                ax.annotate(
                    'Grokking\nTransition',
                    xy=(transition_epoch, transition_loss),
                    xytext=(transition_epoch + 15, transition_loss + 0.1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5,
                                   connectionstyle='arc3,rad=0.2'),
                    fontsize=9, color=color, ha='left'
                )
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_axisbelow(True)
    
    ax.axvspan(0, 30, alpha=0.05, color='gray')
    ax.axvspan(30, 150, alpha=0.03, color='blue')
    ax.text(15, ax.get_ylim()[1] * 0.98, 'Phase A\n(30 epochs)', 
            ha='center', va='top', fontsize=8, color='gray', style='italic')
    ax.text(90, ax.get_ylim()[1] * 0.98, 'Phase B', 
            ha='center', va='top', fontsize=8, color='gray', style='italic')
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved grokking dynamics plot: {save_path}")


# =============================================================================
# Plot 3: Pareto Front
# =============================================================================

def plot_pareto_front(
    df: pd.DataFrame,
    save_path: str,
    x_col: str = 'T_c',
    y_col: str = 'gamma_c',
    title: str = "Pareto Front in Thermogeometric Space",
    xlabel: str = r"$T_c$ (Landscape Flatness $\uparrow$)",
    ylabel: str = r"$\gamma_c$ (Kinetic Mobility $\uparrow$)",
    figsize: Tuple[float, float] = (7, 5.5)
) -> pd.DataFrame:
    """
    Generate Pareto front plot in (T_c, γ_c) space.
    """
    valid_df = df.dropna(subset=[x_col, y_col, 'C1_feasible', 'C2_feasible', 'group'])
    
    if len(valid_df) == 0:
        warnings.warn("Insufficient data for Pareto plot")
        return pd.DataFrame()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    def is_pareto(costs: np.ndarray) -> np.ndarray:
        """Find Pareto-optimal points (minimize both costs)."""
        is_pareto = np.ones(costs.shape[0], dtype=bool)
        for i in range(costs.shape[0]):
            for j in range(costs.shape[0]):
                if i != j:
                    if (costs[j, 0] <= costs[i, 0] and 
                        costs[j, 1] <= costs[i, 1] and
                        (costs[j, 0] < costs[i, 0] or costs[j, 1] < costs[i, 1])):
                        is_pareto[i] = False
                        break
        return is_pareto
    
    # Costs: [T_c, -gamma_c] (lower is better)
    costs = np.column_stack([valid_df[x_col].values, -valid_df[y_col].values])
    pareto_mask = is_pareto(costs)
    pareto_df = valid_df[pareto_mask].copy()
    
    # Shade feasible region
    feasible_df = valid_df[valid_df['C1_feasible'] & valid_df['C2_feasible']]
    
    if len(feasible_df) >= 3:
        try:
            hull_points = np.column_stack([feasible_df[x_col], feasible_df[y_col]])
            hull = ConvexHull(hull_points)
            hull_vertices = np.append(hull.vertices, hull.vertices[0])
            ax.fill(hull_points[hull_vertices, 0], hull_points[hull_vertices, 1],
                    color='green', alpha=0.1, label='Feasible Region (C1 ∧ C2)')
        except Exception as e:
            warnings.warn(f"Could not compute convex hull: {e}")
    
    # Plot Pareto frontier line
    if len(pareto_df) > 1:
        pareto_sorted = pareto_df.sort_values(x_col)
        ax.plot(pareto_sorted[x_col], pareto_sorted[y_col], 'k--',
                linewidth=2, alpha=0.7, label='Pareto Frontier', zorder=2)
    
    # Scatter all architectures
    for group_id in sorted(valid_df['group'].unique()):
        group_df = valid_df[valid_df['group'] == group_id]
        color = GROUP_COLORS.get(group_id, 'gray')
        marker = GROUP_MARKERS.get(group_id, 'o')
        sizes = 100 * (group_df.get('final_test_acc', 0.5) + 0.5)
        
        ax.scatter(group_df[x_col], group_df[y_col], c=color, marker=marker,
                   s=sizes, alpha=0.8, edgecolors='white', linewidths=0.5,
                   label=GROUP_LABELS.get(group_id, f'Group {group_id}'), zorder=3)
        
        pareto_group = pareto_df[pareto_df['group'] == group_id]
        ax.scatter(pareto_group[x_col], pareto_group[y_col], facecolors='none',
                   edgecolors='black', linewidths=2, s=200, zorder=4)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_axisbelow(True)
    
    ax.text(0.02, 0.98, f'Pareto-optimal: {len(pareto_df)}/{len(valid_df)} architectures',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved Pareto front plot: {save_path}")
    
    return pareto_df


# =============================================================================
# Main Figure Generator
# =============================================================================

def generate_all_plots(
    df: pd.DataFrame,
    output_dir: str = ".",
    prefix: str = "lift_test"
) -> Dict[str, str]:
    """
    Generate all three money plots.
    
    Parameters
    ----------
    df : pd.DataFrame
        Combined DataFrame with all required columns.
    output_dir : str
        Directory to save figures.
    prefix : str
        Filename prefix for figures.
    
    Returns
    -------
    Dict[str, str]
        Mapping of plot name to save path.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Plot 1: Spearman Correlation
    print("\n[Plot 1/3] Generating Spearman correlation scatter...")
    corr_path = output_path / f"{prefix}_correlation.png"
    corr_stats = plot_correlation(df, str(corr_path))
    results['correlation'] = str(corr_path)
    
    # Plot 2: Grokking Dynamics
    print("[Plot 2/3] Generating grokking dynamics...")
    grok_path = output_path / f"{prefix}_grokking_dynamics.png"
    plot_grokking_dynamics(df, str(grok_path))
    results['grokking_dynamics'] = str(grok_path)
    
    # Plot 3: Pareto Front
    print("[Plot 3/3] Generating Pareto front...")
    pareto_path = output_path / f"{prefix}_pareto_front.png"
    pareto_df = plot_pareto_front(df, str(pareto_path))
    results['pareto_front'] = str(pareto_path)
    
    # Combined 3-panel figure
    print("\n[Bonus] Generating combined 3-panel figure...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Panel A: Correlation
    ax = axes[0]
    valid_df = df.dropna(subset=['predicted_alpha', 'final_test_acc'])
    for group_id in sorted(valid_df['group'].unique()):
        gdf = valid_df[valid_df['group'] == group_id]
        ax.scatter(gdf['predicted_alpha'], gdf['final_test_acc'],
                   c=GROUP_COLORS[group_id], marker=GROUP_MARKERS[group_id],
                   s=60, alpha=0.8, label=GROUP_LABELS[group_id])
    x_vals = valid_df['predicted_alpha'].values
    y_vals = valid_df['final_test_acc'].values
    y_fit, slope, _ = linear_fit(x_vals, y_vals)
    ax.plot(np.sort(x_vals), y_fit[np.argsort(x_vals)], 'k--', alpha=0.7)
    rho, p = stats.spearmanr(valid_df['predicted_alpha'], valid_df['final_test_acc'])
    ax.text(0.05, 0.95, f'ρ = {rho:.3f}', transform=ax.transAxes,
            va='top', fontsize=9, bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    ax.set_xlabel(r'Predicted $\alpha$')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('(A) Correlation Analysis')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(alpha=0.3)
    
    # Panel B: Grokking
    ax = axes[1]
    for gid in [1, 3]:
        gdf = df[df['group'] == gid].groupby('epoch')['test_loss'].agg(['mean', 'std']).reset_index()
        color = GROUP_COLORS[gid]
        ax.plot(gdf['epoch'], gdf['mean'], color=color, label=GROUP_LABELS[gid], linewidth=1.5)
        ax.fill_between(gdf['epoch'], gdf['mean'] - gdf['std'], gdf['mean'] + gdf['std'],
                        color=color, alpha=0.2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Loss')
    ax.set_title('(B) Grokking Dynamics')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax.axvspan(0, 30, alpha=0.05, color='gray')
    
    # Panel C: Pareto
    ax = axes[2]
    pareto_pts = df.dropna(subset=['T_c', 'gamma_c'])
    if len(pareto_pts) > 0:
        ax.scatter(pareto_pts['T_c'], pareto_pts['gamma_c'],
                   c=[GROUP_COLORS.get(g, 'gray') for g in pareto_pts['group']],
                   marker='o', s=80, alpha=0.8)
        ax.set_xlabel(r'$T_c$')
        ax.set_ylabel(r'$\gamma_c$')
        ax.set_title('(C) Pareto Front')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    combined_path = output_path / f"{prefix}_combined.png"
    plt.savefig(str(combined_path), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    results['combined'] = str(combined_path)
    
    print("\n" + "=" * 50)
    print("All plots generated successfully!")
    for name, path in results.items():
        print(f"  {name}: {path}")
    print("=" * 50)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate lift test plots")
    parser.add_argument("--data_csv", type=str, required=True,
                        help="Path to cleaned CSV data from parse_results.py")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Output directory for plots")
    parser.add_argument("--prefix", type=str, default="lift_test",
                        help="Filename prefix")
    
    args = parser.parse_args()
    df = pd.read_csv(args.data_csv)
    print(f"Loaded {len(df)} rows from {args.data_csv}")
    results = generate_all_plots(df, args.output_dir, args.prefix)
