"""
Data analysis utilities for ThermoRG v4 experiments.

Functions for:
- Aggregating results across seeds
- Fitting β vs ln(γ) linear regression
- F-test for equal slopes
- Generating key figures
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import json


def aggregate_results(results_all: List[Dict]) -> List[Dict]:
    """
    Aggregate results across seeds, computing mean and std.
    
    For each (norm_type, D, lr), computes:
    - mean and std of β, γ, λ_max
    - R² of D-scaling fit
    - number of valid runs
    
    Args:
        results_all: List of result dicts from training
        
    Returns:
        List of aggregated results
    """
    # Group by (norm_type, D, lr)
    groups = {}
    for r in results_all:
        cfg = r.get('config', {})
        key = (cfg.get('norm_type'), cfg.get('D'), cfg.get('lr'))
        if key not in groups:
            groups[key] = []
        groups[key].append(r)
    
    aggregated = []
    
    for key, runs in groups.items():
        norm_type, D, lr = key
        
        # Extract metrics
        betas = [r.get('beta') for r in runs if r.get('beta') is not None]
        gammas = [r.get('gamma') for r in runs if r.get('gamma') is not None]
        lambdas = [r.get('lambda_max_final') for r in runs if r.get('lambda_max_final') is not None]
        losses_final = [r['loss_history'][-1] for r in runs]
        
        agg = {
            'norm_type': norm_type,
            'D': D,
            'lr': lr,
            'n_runs': len(runs),
            'n_valid_beta': len(betas),
            'n_valid_gamma': len(gammas),
        }
        
        if betas:
            agg['beta_mean'] = np.mean(betas)
            agg['beta_std'] = np.std(betas)
        else:
            agg['beta_mean'] = None
            agg['beta_std'] = None
            
        if gammas:
            agg['gamma_mean'] = np.mean(gammas)
            agg['gamma_std'] = np.std(gammas)
            # Use mean gamma for regression
            agg['gamma'] = np.mean(gammas)
        else:
            agg['gamma_mean'] = None
            agg['gamma_std'] = None
            agg['gamma'] = None
            
        if lambdas:
            agg['lambda_max_mean'] = np.mean(lambdas)
            agg['lambda_max_std'] = np.std(lambdas)
        else:
            agg['lambda_max_mean'] = None
            agg['lambda_max_std'] = None
            
        agg['loss_final_mean'] = np.mean(losses_final)
        agg['loss_final_std'] = np.std(losses_final)
        
        # Use beta from D-scaling fit if available (from train result)
        betas_direct = [r.get('beta', r.get('beta_direct')) for r in runs]
        betas_direct = [b for b in betas_direct if b is not None]
        if betas_direct:
            agg['beta'] = np.mean(betas_direct)
        
        # Aggregate gamma_init (zero-point fluctuation indicator)
        gammas_init = [r.get('gamma_init') for r in runs if r.get('gamma_init') is not None]
        if gammas_init:
            agg['gamma_init_mean'] = np.mean(gammas_init)
            agg['gamma_init_std'] = np.std(gammas_init)
        else:
            agg['gamma_init_mean'] = None
            agg['gamma_init_std'] = None
        
        aggregated.append(agg)
    
    return aggregated


def fit_beta_vs_ln_gamma(beta_gamma_pairs: List[Tuple[float, float]],
                         norm_type: str) -> Dict:
    """
    Fit β = m * ln(γ) + c linear regression.
    
    Args:
        beta_gamma_pairs: List of (beta, gamma) tuples
        norm_type: Name of normalization type
        
    Returns:
        Dict with m, c, r_squared, n_points, std_err
    """
    betas = np.array([p[0] for p in beta_gamma_pairs])
    gammas = np.array([p[1] for p in beta_gamma_pairs])
    
    # Need γ > 0 for log
    valid_mask = gammas > 0
    if np.sum(valid_mask) < 3:
        return {'m': None, 'c': None, 'r_squared': 0, 'n_points': len(gammas),
                'p_value': None, 'std_err': None}
    
    ln_gamma = np.log(gammas[valid_mask])
    betas_valid = betas[valid_mask]
    
    # Linear regression: β = m * ln(γ) + c
    slope, intercept, r_value, p_value, std_err = stats.linregress(ln_gamma, betas_valid)
    
    return {
        'm': slope,
        'c': intercept,
        'r_squared': r_value ** 2,
        'p_value': p_value,
        'std_err': std_err,
        'n_points': int(np.sum(valid_mask)),
        'norm_type': norm_type
    }


def f_test_equal_slopes(regression_results: Dict[str, Dict],
                        alpha: float = 0.05) -> Dict:
    """
    F-test to check if all normalization types have equal slopes.
    
    H0: All slopes are equal (m_BN = m_LN = m_GN = m_None)
    H1: At least one slope is different
    
    Uses reduced model (equal slopes) vs full model (free slopes).
    
    Args:
        regression_results: Dict of norm_type -> regression dict with 'm', 'c', 'n_points'
        alpha: Significance level
        
    Returns:
        Dict with f_statistic, p_value, reject_null, etc.
    """
    norm_types = list(regression_results.keys())
    k = len(norm_types)
    
    if k < 2:
        return {'f_statistic': None, 'p_value': None, 'reject_null': None,
                'message': 'Need at least 2 groups for F-test'}
    
    # Collect all data points
    all_slopes = np.array([regression_results[nt]['m'] for nt in norm_types])
    all_ns = np.array([regression_results[nt]['n_points'] for nt in norm_types])
    all_intercepts = np.array([regression_results[nt]['c'] for nt in norm_types])
    
    # Total sample size
    N = np.sum(all_ns)
    
    # Degrees of freedom
    df_reduced = N - 2 * k  # Equal slopes model
    df_full = N - 2 * k     # Different slopes model (same df but more params)
    
    # Actually for F-test we need raw data. Using approximation:
    # Variance of slope estimates from standard errors
    # F = (SS_reduced - SS_full) / (df_reduced - df_full) / (SS_full / df_full)
    
    # Weighted mean slope
    weights = all_ns / np.sum(all_ns)
    pooled_slope = np.sum(weights * all_slopes)
    
    # Between-group SS (due to slope differences)
    ss_between = np.sum(all_ns * (all_slopes - pooled_slope)**2)
    
    # Estimate within-group variance from regression standard errors
    # MSE_i = std_err_i² * S_xx_i, approximated from R² and slope magnitude
    # Typical observed range of ln(γ) is ~[-7, 5] (γ from ~0.001 to ~150)
    ln_gamma_range_estimate = 12.0
    mse_estimates = []
    for nt in norm_types:
        res = regression_results[nt]
        if (res.get('m') is not None and res.get('r_squared') is not None
                and res['r_squared'] < 1 and np.isfinite(res['m'])):
            # Var(β) ≈ (m · ln(γ)_range / 4)²   (uniform spread assumption)
            var_beta = (res['m'] * ln_gamma_range_estimate / 4) ** 2
            # MSE = (1 - R²) · Var(β)
            mse = (1 - res['r_squared']) * var_beta
            mse_estimates.append(mse)
    ms_within = float(np.mean(mse_estimates)) if mse_estimates else 0.01
    
    if ms_within <= 0:
        return {'f_statistic': None, 'p_value': None, 'reject_null': None,
                'message': 'Cannot estimate within-group variance'}
    
    f_stat = ss_between / ms_within
    df1 = k - 1  # Number of slope differences tested
    df2 = N - 2 * k
    
    if df2 <= 0:
        return {'f_statistic': None, 'p_value': None, 'reject_null': None,
                'message': 'Insufficient data for F-test'}
    
    p_value = 1 - stats.f.cdf(f_stat, df1, df2)
    
    return {
        'f_statistic': f_stat,
        'p_value': p_value,
        'df_between': df1,
        'df_within': df2,
        'reject_null': p_value < alpha,
        'norm_types': norm_types,
        'pooled_slope': pooled_slope,
        'message': 'F-test for equal slopes'
    }


def ancova_test_equal_slopes(raw_data: List[Dict], alpha: float = 0.05) -> Dict:
    """
    ANCOVA test for equal slopes across normalization types.

    Full model: β = β_0 + β_spec · ln(γ) + Σ_j δ_j · I(norm_type=j)
                                         + Σ_j α_j · [ln(γ) × I(norm_type=j)] + ε

    H0: All interaction terms α_j = 0 (equal slopes)
    H1: At least one α_j ≠ 0 (slopes differ)

    Uses Type III ANOVA to test the interaction term jointly.

    Args:
        raw_data: List of dicts with keys 'beta', 'gamma', 'norm_type'
                  (one entry per run, not aggregated)
        alpha: Significance level

    Returns:
        Dict with:
        - beta_spec: shared slope estimate (main effect of ln(gamma))
        - interaction_pvalue: p-value for H0 (equal slopes)
        - reject_null: True if slopes differ significantly
        - model: dict with coefficients and R²
        - anova_table: Type III ANOVA results
    """
    try:
        import pandas as pd
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except ImportError:
        return {
            'beta_spec': None,
            'interaction_pvalue': None,
            'reject_null': None,
            'message': 'statsmodels required for ANCOVA. Install: pip install statsmodels',
            'fallback_to_f_test': True
        }

    # Build DataFrame
    df = pd.DataFrame(raw_data)

    # Filter to valid data points
    df = df[df['gamma'] > 0].copy()
    df['ln_gamma'] = np.log(df['gamma'])

    if len(df) < 10:
        return {
            'beta_spec': None,
            'interaction_pvalue': None,
            'reject_null': None,
            'message': f'Insufficient data points ({len(df)}) for ANCOVA',
            'n_points': len(df)
        }

    norm_types = df['norm_type'].unique()
    if len(norm_types) < 2:
        return {
            'beta_spec': None,
            'interaction_pvalue': None,
            'reject_null': None,
            'message': 'Need at least 2 norm types for ANCOVA',
            'n_points': len(df)
        }

    # Fit full ANCOVA model with interaction
    # C(norm_type) creates indicator variables for each norm type
    # C(norm_type):ln_gamma creates the interaction terms
    # Note: ln_gamma is pre-computed above (line ~275) to avoid np.log namespace issues
    formula = 'beta ~ C(norm_type) * ln_gamma'

    try:
        model = smf.ols(formula, data=df).fit()
    except Exception as e:
        return {
            'beta_spec': None,
            'interaction_pvalue': None,
            'reject_null': None,
            'message': f'ANCOVA model fitting failed: {e}',
            'fallback_to_f_test': True
        }

    # Type III ANOVA for the interaction term
    try:
        anova_table = sm.stats.anova_lm(model, typ=3)
    except Exception as e:
        return {
            'beta_spec': None,
            'interaction_pvalue': None,
            'reject_null': None,
            'message': f'Type III ANOVA failed: {e}',
            'fallback_to_f_test': True
        }

    # Extract interaction p-value
    interaction_label = 'C(norm_type):np.log(gamma)'
    if interaction_label not in anova_table.index:
        # Try alternative label format
        for idx in anova_table.index:
            if 'norm_type' in idx and 'ln_gamma' in idx:
                interaction_label = idx
                break
        else:
            return {
                'beta_spec': None,
                'interaction_pvalue': None,
                'reject_null': None,
                'message': f'Interaction term not found in ANOVA table. Available: {list(anova_table.index)}',
                'fallback_to_f_test': True
            }

    interaction_pvalue = anova_table.loc[interaction_label, 'PR(>F)']

    # Extract β_spec (main effect of ln(gamma))
    ln_gamma_label = 'np.log(gamma)'
    if ln_gamma_label not in anova_table.index:
        for idx in anova_table.index:
            if 'ln_gamma' in idx or 'log' in idx.lower():
                ln_gamma_label = idx
                break

    beta_spec = model.params.get(ln_gamma_label) or model.params.get('np.log(gamma)')

    # Extract intercepts per norm type (for A_norm computation)
    intercepts = {}
    for nt in norm_types:
        col = f'C(norm_type)[T.{nt}]'
        if col in model.params:
            intercepts[nt] = model.params[col]
        else:
            intercepts[nt] = None

    # Compute A_norm per type: A_norm = exp(-c_norm / beta_spec)
    # The full model is: β = (intercept) + β_spec * ln(γ) + offset
    # For norm_type=j: β = (intercept + offset_j) + β_spec * ln(γ)
    # So c_j = intercept + offset_j
    base_intercept = model.params.get('Intercept', 0)

    A_norm = {}
    for nt, offset in intercepts.items():
        if offset is not None and beta_spec is not None and beta_spec != 0:
            c_j = base_intercept + offset
            A_norm[nt] = float(np.exp(-c_j / beta_spec))
        else:
            A_norm[nt] = None

    return {
        'beta_spec': float(beta_spec) if beta_spec is not None else None,
        'interaction_pvalue': float(interaction_pvalue) if interaction_pvalue is not None else None,
        'reject_null': bool(interaction_pvalue < alpha) if interaction_pvalue is not None else None,
        'r_squared': float(model.rsquared),
        'r_squared_adj': float(model.rsquared_adj),
        'n_points': int(len(df)),
        'n_norm_types': int(len(norm_types)),
        'A_norm': A_norm,
        'base_intercept': float(base_intercept),
        'norm_type_offsets': {k: float(v) if v is not None else None for k, v in intercepts.items()},
        'anova_table': anova_table.to_dict(),
        'model_summary': {
            'nobs': int(model.nobs),
            'df_model': int(model.df_model),
            'df_resid': int(model.df_resid),
        },
        'message': 'ANCOVA completed successfully',
        'fallback_to_f_test': False
    }


def prepare_raw_data_for_ancova(results_all: List[Dict]) -> List[Dict]:
    """
    Prepare raw run data for ANCOVA analysis.

    ANCOVA needs individual run data points (not aggregated) to properly
    estimate residual variance and interactions.

    Args:
        results_all: List of result dicts from training runs

    Returns:
        List of dicts with 'beta', 'gamma', 'norm_type', 'D', 'lr', 'seed'
        Only includes runs with valid beta and gamma > 0
    """
    raw_data = []
    for r in results_all:
        cfg = r.get('config', {})
        norm_type = cfg.get('norm_type')
        beta = r.get('beta') or r.get('beta_direct')
        gamma = r.get('gamma')

        if beta is not None and gamma is not None and gamma > 0 and norm_type:
            raw_data.append({
                'beta': beta,
                'gamma': gamma,
                'norm_type': norm_type,
                'D': cfg.get('D'),
                'lr': cfg.get('lr'),
                'seed': cfg.get('seed'),
                'gamma_init': r.get('gamma_init'),
            })

    return raw_data


def extract_physics_constants(regression_results: Dict[str, Dict]) -> Dict:
    """
    Extract physical constants from regression results.
    
    From β = β_spec · ln(γ / A_norm):
    - β_spec = shared slope
    - A_norm = exp(-c_norm / β_spec)
    
    Args:
        regression_results: Dict of norm_type -> regression result
        
    Returns:
        Dict with β_spec, A_norm per normalization type
    """
    # Pooled slope (β_spec)
    slopes = np.array([r['m'] for r in regression_results.values() if r['m'] is not None])
    ns = np.array([r['n_points'] for r in regression_results.values() if r['m'] is not None])
    
    if len(slopes) == 0:
        return {'beta_spec': None, 'A_norm': {}, 'message': 'No valid slopes'}
    
    # Weighted average as β_spec estimate
    weights = ns / np.sum(ns)
    beta_spec = np.sum(weights * slopes)
    
    # Compute A_norm per type
    A_norm = {}
    for norm_type, result in regression_results.items():
        if result['m'] is not None and result['c'] is not None:
            A_norm[norm_type] = np.exp(-result['c'] / beta_spec)
    
    return {
        'beta_spec': beta_spec,
        'A_norm': A_norm,
        'slopes_by_type': {nt: r['m'] for nt, r in regression_results.items() if r['m'] is not None},
        'intercepts_by_type': {nt: r['c'] for nt, r in regression_results.items() if r['c'] is not None},
    }


def check_parallel_lines(regression_results: Dict[str, Dict],
                         slope_tolerance: float = 0.3) -> Dict:
    """
    Check if β vs ln(γ) lines are parallel (same slope).
    
    Args:
        regression_results: Dict of norm_type -> regression result
        slope_tolerance: Maximum allowed slope difference (default 0.3)
        
    Returns:
        Dict with check results
    """
    slopes = {nt: r['m'] for nt, r in regression_results.items() if r['m'] is not None}
    
    if len(slopes) < 2:
        return {'is_parallel': None, 'message': 'Need at least 2 norm types'}
    
    slope_values = list(slopes.values())
    slope_range = max(slope_values) - min(slope_values)
    
    # Relative difference normalized by mean slope
    mean_slope = np.mean(slope_values)
    relative_diff = slope_range / abs(mean_slope) if mean_slope != 0 else float('inf')
    
    is_parallel = relative_diff < slope_tolerance
    
    return {
        'is_parallel': is_parallel,
        'slope_range': slope_range,
        'relative_difference': relative_diff,
        'slopes': slopes,
        'slope_tolerance': slope_tolerance,
        'message': f"Slopes differ by {relative_diff:.2%} (tolerance: {slope_tolerance:.0%})"
    }


def save_aggregated_results(aggregated: List[Dict], filepath: str):
    """Save aggregated results to JSON."""
    # Convert numpy types to Python types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    with open(filepath, 'w') as f:
        json.dump(aggregated, f, indent=2, default=convert)


def load_aggregated_results(filepath: str) -> List[Dict]:
    """Load aggregated results from JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)
