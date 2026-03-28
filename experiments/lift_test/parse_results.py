"""
parse_results.py - Data Harvester for CIFAR-10 Thermogeometric Lift Test
=========================================================================
Parses and cleans CSV training logs, merges with theoretical predictions,
and computes correlations.

Author: Leo Liu / ThermoRG-NN Team
Date: 2026-03-28
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy import stats
import warnings

# Suppress pandas performance warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


# =============================================================================
# Group Definitions
# =============================================================================

# Mapping of architecture names to their control groups
ARCH_TO_GROUP: Dict[str, int] = {
    # Group 1: Thermogeometric Optimal
    'ResNet-18-TGR': 1,
    'ResNet-34-TGR': 1,
    'WideResNet-28-10-TGR': 1,
    'DenseNet-40-TGR': 1,
    
    # Group 2: Topology Destroyer (narrow bottlenecks)
    'ResNet-18-Narrow': 2,
    'ResNet-34-Narrow': 2,
    'WideResNet-28-10-Narrow': 2,
    'DenseNet-40-Narrow': 2,
    
    # Group 3: Thermal Boiling Furnace (raw ReLU, no norm)
    'ResNet-18-Furnace': 3,
    'ResNet-34-Furnace': 3,
    'WideResNet-28-10-Furnace': 3,
    'DenseNet-40-Furnace': 3,
    
    # Group 4: Traditional Baselines
    'ResNet-18': 4,
    'VGG-11': 4,
    'DenseNet-40': 4,
}

GROUP_LABELS: Dict[int, str] = {
    1: 'G1 (Thermo. Optimal)',
    2: 'G2 (Topology Destroyer)',
    3: 'G3 (Thermal Furnace)',
    4: 'G4 (Traditional)',
}


# =============================================================================
# Core Functions
# =============================================================================

def load_training_logs(results_dir: str) -> pd.DataFrame:
    """
    Load all CSV training logs from results folder.
    
    Parameters
    ----------
    results_dir : str
        Path to directory containing CSV training logs.
        Expected naming: {arch_name}_seed{seed}.csv
    
    Returns
    -------
    pd.DataFrame
        Combined DataFrame with columns:
        arch_name, seed, epoch, train_loss, val_loss, test_loss,
        train_acc, val_acc, test_acc
    
    Raises
    ------
    FileNotFoundError
        If no CSV files found in results_dir.
    """
    results_path = Path(results_dir)
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    csv_files = list(results_path.glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {results_dir}")
    
    dfs = []
    for csv_file in csv_files:
        # Parse architecture name and seed from filename
        # Expected format: {arch_name}_seed{seed}.csv
        filename = csv_file.stem
        parts = filename.rsplit('_seed', 1)
        
        if len(parts) == 2:
            arch_name = parts[0]
            try:
                seed = int(parts[1])
            except ValueError:
                seed = 0
                arch_name = filename  # Fallback: whole filename is arch name
        else:
            arch_name = filename
            seed = 0
        
        # Load CSV
        df = pd.read_csv(csv_file)
        df['arch_name'] = arch_name
        df['seed'] = seed
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Validate required columns
    required_cols = ['epoch', 'train_loss', 'val_loss', 'test_loss',
                     'train_acc', 'val_acc', 'test_acc']
    missing = set(required_cols) - set(combined_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    return combined_df


def extract_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract key metrics from training logs.
    
    Computes:
    - Final test loss and accuracy (at max epoch)
    - Best test accuracy across all epochs
    - Grokking epoch (first epoch where test_acc > 50%)
    - Learning speed (epochs to reach 80% of best acc)
    
    Parameters
    ----------
    df : pd.DataFrame
        Training logs DataFrame from load_training_logs().
    
    Returns
    -------
    pd.DataFrame
        Aggregated metrics per (arch_name, seed).
    """
    results = []
    
    for (arch_name, seed), group_df in df.groupby(['arch_name', 'seed']):
        group_df = group_df.sort_values('epoch')
        
        # Final epoch metrics
        final_row = group_df.iloc[-1]
        
        # Best test accuracy
        best_test_acc = group_df['test_acc'].max()
        best_test_acc_epoch = group_df.loc[group_df['test_acc'].idxmax(), 'epoch']
        
        # Grokking epoch: first epoch where test_acc > 50%
        grokking_mask = group_df['test_acc'] > 0.50
        grokking_epoch = group_df.loc[grokking_mask, 'epoch'].min() if grokking_mask.any() else np.nan
        
        # Early-phase metrics (epoch 30)
        epoch_30 = group_df[group_df['epoch'] <= 30]
        if len(epoch_30) > 0:
            early_test_acc = epoch_30.iloc[-1]['test_acc']
            early_test_loss = epoch_30.iloc[-1]['test_loss']
        else:
            early_test_acc = np.nan
            early_test_loss = np.nan
        
        results.append({
            'arch_name': arch_name,
            'seed': seed,
            'final_epoch': final_row['epoch'],
            'final_test_loss': final_row['test_loss'],
            'final_test_acc': final_row['test_acc'],
            'best_test_acc': best_test_acc,
            'best_test_acc_epoch': best_test_acc_epoch,
            'grokking_epoch': grokking_epoch,
            'early_test_acc_30': early_test_acc,
            'early_test_loss_30': early_test_loss,
        })
    
    return pd.DataFrame(results)


def merge_with_predictions(
    df: pd.DataFrame,
    predictions: Dict[str, Dict]
) -> pd.DataFrame:
    """
    Merge experimental results with theoretical predictions.
    
    Parameters
    ----------
    df : pd.DataFrame
        Metrics DataFrame from extract_metrics().
    predictions : Dict[str, Dict]
        Theoretical predictions per architecture.
        Format: {
            'arch_name': {
                'predicted_alpha': float,
                'T_tilde_eff': float,
                'C1_feasible': bool,
                'C2_feasible': bool,
                'T_c': float,          # landscape flatness
                'gamma_c': float,       # kinetic mobility
            }
        }
    
    Returns
    -------
    pd.DataFrame
        Merged DataFrame with both experimental and theoretical columns.
    """
    pred_records = []
    for arch_name, pred in predictions.items():
        record = {'arch_name': arch_name, **pred}
        pred_records.append(record)
    
    pred_df = pd.DataFrame(pred_records)
    
    merged_df = df.merge(pred_df, on='arch_name', how='left')
    
    # Flag architectures missing predictions
    missing_pred = merged_df['predicted_alpha'].isna()
    if missing_pred.any():
        archs = merged_df.loc[missing_pred, 'arch_name'].unique()
        warnings.warn(f"Missing predictions for architectures: {archs}")
    
    return merged_df


def add_group_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add group labels (G1/G2/G3/G4) to the DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with arch_name column.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added 'group' and 'group_label' columns.
    """
    df = df.copy()
    
    df['group'] = df['arch_name'].map(ARCH_TO_GROUP).fillna(0).astype(int)
    df['group_label'] = df['group'].map(GROUP_LABELS).fillna('Unknown')
    
    return df


def compute_correlations(df: pd.DataFrame) -> Dict:
    """
    Compute Spearman rank correlation between predicted α and actual accuracy.
    
    Also computes:
    - Pearson correlation
    - P-values
    - Group-wise correlations
    
    Parameters
    ----------
    df : pd.DataFrame
        Merged DataFrame with predicted_alpha and test accuracy columns.
    
    Returns
    -------
    Dict
        Dictionary of correlation results:
        {
            'spearman_rho': float,
            'spearman_pvalue': float,
            'pearson_r': float,
            'pearson_pvalue': float,
            'group_correlations': Dict[int, Tuple[float, float]],
        }
    """
    # Filter to rows with both predicted_alpha and final_test_acc
    valid_df = df.dropna(subset=['predicted_alpha', 'final_test_acc'])
    
    if len(valid_df) < 3:
        warnings.warn("Insufficient data for correlation analysis (< 3 points)")
        return {}
    
    # Overall Spearman correlation
    spearman_rho, spearman_p = stats.spearmanr(
        valid_df['predicted_alpha'],
        valid_df['final_test_acc']
    )
    
    # Overall Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(
        valid_df['predicted_alpha'],
        valid_df['final_test_acc']
    )
    
    # Group-wise correlations
    group_corrs = {}
    for group_id in sorted(df['group'].unique()):
        group_df = valid_df[valid_df['group'] == group_id]
        if len(group_df) >= 3:
            rho, p = stats.spearmanr(
                group_df['predicted_alpha'],
                group_df['final_test_acc']
            )
            group_corrs[group_id] = (rho, p)
    
    return {
        'spearman_rho': spearman_rho,
        'spearman_pvalue': spearman_p,
        'pearson_r': pearson_r,
        'pearson_pvalue': pearson_p,
        'group_correlations': group_corrs,
        'n_samples': len(valid_df),
    }


def export_clean_data(
    df: pd.DataFrame,
    output_path: str,
    include_columns: Optional[List[str]] = None
) -> None:
    """
    Export cleaned data for plotting.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to export.
    output_path : str
        Path to output CSV file.
    include_columns : List[str], optional
        Specific columns to export. If None, exports all.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if include_columns:
        export_df = df[[c for c in include_columns if c in df.columns]]
    else:
        export_df = df
    
    export_df.to_csv(output_path, index=False)
    print(f"Exported clean data to: {output_path}")


# =============================================================================
# Convenience Functions
# =============================================================================

def compute_pareto_frontier(
    df: pd.DataFrame,
    x_col: str = 'T_c',
    y_col: str = 'gamma_c'
) -> pd.DataFrame:
    """
    Compute Pareto-optimal frontier in (x, y) space.
    
    Maximizes y while minimizing x (lower T_c, higher gamma_c is better).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with x and y columns.
    x_col : str
        Column name for x-axis (T_c - lower is better).
    y_col : str
        Column name for y-axis (gamma_c - higher is better).
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing only Pareto-optimal points.
    """
    pareto_points = []
    
    for idx, row in df.iterrows():
        is_pareto = True
        for _, other in df.iterrows():
            # Check if 'other' dominates 'row'
            # (lower x AND higher y is better)
            if (other[x_col] <= row[x_col] and 
                other[y_col] >= row[y_col] and
                (other[x_col] < row[x_col] or other[y_col] > row[y_col])):
                is_pareto = False
                break
        if is_pareto:
            pareto_points.append(row)
    
    return pd.DataFrame(pareto_points)


# =============================================================================
# Main Pipeline
# =============================================================================

def run_pipeline(
    results_dir: str,
    predictions: Dict[str, Dict],
    output_dir: str = "."
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run the full data harvesting pipeline.
    
    Parameters
    ----------
    results_dir : str
        Path to directory containing CSV training logs.
    predictions : Dict[str, Dict]
        Theoretical predictions per architecture.
    output_dir : str
        Directory to save cleaned data CSV.
    
    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        (final DataFrame, correlation results)
    """
    print("=" * 60)
    print("CIFAR-10 Lift Test - Data Harvesting Pipeline")
    print("=" * 60)
    
    # Step 1: Load training logs
    print("\n[1/5] Loading training logs...")
    raw_df = load_training_logs(results_dir)
    print(f"      Loaded {len(raw_df)} rows from {raw_df['arch_name'].nunique()} architectures")
    
    # Step 2: Extract metrics
    print("\n[2/5] Extracting metrics...")
    metrics_df = extract_metrics(raw_df)
    print(f"      Extracted metrics for {len(metrics_df)} (arch, seed) pairs")
    
    # Step 3: Merge with predictions
    print("\n[3/5] Merging with theoretical predictions...")
    merged_df = merge_with_predictions(metrics_df, predictions)
    print(f"      Merged {merged_df['predicted_alpha'].notna().sum()} architectures with predictions")
    
    # Step 4: Add group labels
    print("\n[4/5] Adding group labels...")
    labeled_df = add_group_labels(merged_df)
    group_counts = labeled_df['group'].value_counts().sort_index()
    for g, count in group_counts.items():
        print(f"      {GROUP_LABELS.get(g, g)}: {count} runs")
    
    # Step 5: Compute correlations
    print("\n[5/5] Computing correlations...")
    corr_results = compute_correlations(labeled_df)
    if corr_results:
        print(f"      Spearman ρ = {corr_results['spearman_rho']:.4f} (p = {corr_results['spearman_pvalue']:.4e})")
        print(f"      Pearson r   = {corr_results['pearson_r']:.4f} (p = {corr_results['pearson_pvalue']:.4e})")
    
    # Export cleaned data
    output_path = Path(output_dir) / "cleaned_lift_test_data.csv"
    export_clean_data(labeled_df, str(output_path))
    
    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)
    
    return labeled_df, corr_results


if __name__ == "__main__":
    # Example usage (requires data)
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse CIFAR-10 lift test results")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing CSV training logs")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Output directory for cleaned CSV")
    
    args = parser.parse_args()
    
    # Example predictions (replace with actual theoretical values)
    example_predictions = {
        arch: {
            'predicted_alpha': np.random.uniform(0.5, 1.5),
            'T_tilde_eff': np.random.uniform(0.1, 1.0),
            'C1_feasible': np.random.choice([True, False]),
            'C2_feasible': np.random.choice([True, False]),
            'T_c': np.random.uniform(0.5, 2.0),
            'gamma_c': np.random.uniform(0.1, 1.0),
        }
        for arch in ARCH_TO_GROUP.keys()
    }
    
    # Run pipeline
    df, corrs = run_pipeline(args.results_dir, example_predictions, args.output_dir)
    
    print("\nSample of cleaned data:")
    print(df.head())
