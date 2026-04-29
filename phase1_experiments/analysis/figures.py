"""
Figure generation for ThermoRG v4 experiments.

Key figures:
- Figure 1: β vs ln(γ) parallel lines plot
- Figure 2: D-scaling collapse
- Figure 3: EOS verification (λ_max · η scatter)
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
import os


# Color scheme for normalization types
NORM_COLORS = {
    'batchnorm': '#E63946',   # Red
    'layernorm': '#2A9D8F',  # Teal
    'groupnorm': '#E9C46A',   # Gold
    'none': '#264653',       # Dark blue
}

NORM_MARKERS = {
    'batchnorm': 'o',
    'layernorm': 's',
    'groupnorm': '^',
    'none': 'd',
}


def generate_figure_1_parallel_lines(
    regression_results: Dict,
    aggregated: List[Dict],
    output_path: str,
    title: str = "β vs ln(γ) Parallel Lines Test"
) -> str:
    """
    Generate Figure 1: β vs ln(γ) parallel lines plot.
    
    Each normalization type should have its own line with same slope.
    
    Args:
        regression_results: Dict of norm_type -> regression dict with m, c, r_squared
        aggregated: List of aggregated results
        output_path: Where to save the figure
        title: Plot title
        
    Returns:
        Path to saved figure
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot data points
    for norm_type in aggregated:
        nt = norm_type['norm_type']
        gamma = norm_type.get('gamma_mean') or norm_type.get('gamma')
        beta = norm_type.get('beta_mean') or norm_type.get('beta')
        
        if gamma is None or beta is None:
            continue
        
        color = NORM_COLORS.get(nt, '#888888')
        marker = NORM_MARKERS.get(nt, 'o')
        
        ax.scatter(np.log(gamma), beta, color=color, marker=marker,
                  s=80, alpha=0.7, label=nt, edgecolors='white', linewidth=0.5)
    
    # Plot regression lines
    x_range = np.linspace(-3, 1, 100)  # ln(γ) range
    
    for norm_type, result in regression_results.items():
        if result['m'] is None or result['c'] is None:
            continue
        
        color = NORM_COLORS.get(norm_type, '#888888')
        m, c = result['m'], result['c']
        
        # Plot line
        y_line = m * x_range + c
        ax.plot(x_range, y_line, color=color, linewidth=2, alpha=0.8)
        
        # Add R² annotation
        r2 = result['r_squared']
        ax.text(0.05, 0.95 - 0.08 * list(regression_results.keys()).index(norm_type),
               f'{norm_type}: m={m:.3f}, R²={r2:.3f}',
               transform=ax.transAxes, fontsize=10, color=color,
               verticalalignment='top')
    
    ax.set_xlabel('ln(γ)', fontsize=12)
    ax.set_ylabel('β', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add horizontal line at β=0
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def generate_figure_2_d_scaling_collapse(
    results_all: List[Dict],
    output_path: str,
    title: str = "D-Scaling Collapse"
) -> str:
    """
    Generate Figure 2: D-scaling collapse.
    
    Loss ℒ vs D, with D-scaling law ℒ = α·D^(-β) + E_floor overlay.
    
    Args:
        results_all: List of all run results
        output_path: Where to save
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Group by norm_type
    by_norm = {}
    for r in results_all:
        nt = r['config']['norm_type']
        D = r['config']['D']
        if nt not in by_norm:
            by_norm[nt] = {'D': [], 'loss': []}
        by_norm[nt]['D'].append(D)
        by_norm[nt]['loss'].append(r['loss_history'][-1])
    
    # Plot each norm type
    for norm_type, data in by_norm.items():
        color = NORM_COLORS.get(norm_type, '#888888')
        marker = NORM_MARKERS.get(norm_type, 'o')
        
        # Average over same D values
        unique_D = sorted(set(data['D']))
        avg_losses = []
        for D in unique_D:
            losses_at_D = [l for d, l in zip(data['D'], data['loss']) if d == D]
            avg_losses.append(np.mean(losses_at_D))
        
        ax.scatter(unique_D, avg_losses, color=color, marker=marker,
                  s=100, alpha=0.8, label=norm_type)
        
        # Fit D-scaling law
        try:
            from scipy.optimize import curve_fit
            def d_scaling(D, alpha, beta, E_floor):
                return alpha * np.power(D, -beta) + E_floor
            
            popt, _ = curve_fit(d_scaling, unique_D, avg_losses,
                               p0=[1.0, 0.5, 0.5], bounds=([0, 0, 0], [np.inf, 5, np.inf]))
            alpha, beta, E_floor = popt
            
            # Plot fit
            D_fit = np.linspace(min(unique_D), max(unique_D), 100)
            ax.plot(D_fit, d_scaling(D_fit, alpha, beta, E_floor),
                   color=color, linestyle='--', alpha=0.6,
                   label=f'{norm_type}: β={beta:.3f}')
        except:
            pass
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('D (channel width)', fontsize=12)
    ax.set_ylabel('Final Loss ℒ', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def generate_figure_3_eos_verification(
    eos_results: Dict,
    output_path: str,
    title: str = "EOS Verification: λ_max · η"
) -> str:
    """
    Generate Figure 3: EOS verification.
    
    Scatter plot of λ_max · η vs η, showing convergence boundary.
    Expected: λ_max · η ≈ 2 for critical learning rate.
    
    Args:
        eos_results: Dict from Phase 0.1 with EOS data
        output_path: Where to save
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for norm_type, lr_data in eos_results.items():
        lrs = []
        eos_ratios = []
        converged = []
        
        for lr, seed_data in lr_data.items():
            for seed, data in seed_data.items():
                lrs.append(lr)
                eos_ratios.append(data['eos_ratio'])
                converged.append(data['is_converged'])
        
        color = NORM_COLORS.get(norm_type, '#888888')
        marker = NORM_MARKERS.get(norm_type, 'o')
        
        for i in range(len(lrs)):
            ax.scatter(lrs[i], eos_ratios[i], color=color, marker=marker,
                      s=60, alpha=0.7 if converged[i] else 0.3,
                      edgecolors='white' if converged[i] else 'none')
        
        # Add legend entry
        ax.scatter([], [], color=color, marker=marker, label=norm_type, s=80)
    
    # Expected EOS ratio
    ax.axhline(y=2.0, color='red', linestyle='--', alpha=0.7,
              label='EOS target (λ_max · η = 2)')
    
    # Tolerance band
    ax.axhspan(1.5, 3.0, alpha=0.1, color='green', label='Tolerance [1.5, 3.0]')
    
    ax.set_xscale('log')
    ax.set_xlabel('Learning Rate η', fontsize=12)
    ax.set_ylabel('λ_max · η', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def generate_all_figures(
    regression_results: Dict,
    aggregated: List[Dict],
    eos_results: Optional[Dict],
    results_all: List[Dict],
    output_dir: str
) -> Dict[str, str]:
    """
    Generate all three key figures.
    
    Returns:
        Dict mapping figure name to file path
    """
    os.makedirs(output_dir, exist_ok=True)
    
    paths = {}
    
    # Figure 1: Parallel lines
    fig1_path = os.path.join(output_dir, 'figure1_parallel_lines.png')
    paths['figure1'] = generate_figure_1_parallel_lines(
        regression_results, aggregated, fig1_path
    )
    
    # Figure 2: D-scaling collapse
    fig2_path = os.path.join(output_dir, 'figure2_d_scaling_collapse.png')
    paths['figure2'] = generate_figure_2_d_scaling_collapse(
        results_all, fig2_path
    )
    
    # Figure 3: EOS verification
    if eos_results is not None:
        fig3_path = os.path.join(output_dir, 'figure3_eos_verification.png')
        paths['figure3'] = generate_figure_3_eos_verification(
            eos_results, fig3_path
        )
    
    return paths
