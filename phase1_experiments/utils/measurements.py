"""
ThermoRG v4 Experiment - Unified Measurement Utilities (Kaggle-Adapted)

Core measurement functions for:
- γ (representational shift): (1/L) * Σ |ln(σ_final / σ_init)| using normalized activations
- λ_max: maximum singular value via power iteration
- β: D-scaling exponent from ℒ = α·D^(-β) + E_floor fit
- Stationarity detection: 85% monotonicity over 20 epochs

Kaggle-specific features:
- Per-run JSON files (result_{run_id}.json)
- Incremental results saving
- Resume-friendly format

All measurements follow the protocol definitions in phase1_v4_experiment_protocol.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from typing import Dict, Tuple, Optional, List
import json
import os


# =============================================================================
# γ (REPRESENTATIONAL SHIFT) MEASUREMENT
# =============================================================================

def get_normalized_activations(x: torch.Tensor, norm_type: str, 
                                running_mean: Optional[torch.Tensor] = None,
                                running_var: Optional[torch.Tensor] = None,
                                num_groups: int = 4) -> torch.Tensor:
    """
    Extract normalized activations (WITHOUT affine transform) for γ measurement.
    """
    eps = 1e-5
    
    if norm_type == 'batchnorm':
        if running_mean is not None and running_var is not None:
            mu = running_mean
            sigma = torch.sqrt(running_var + eps)
        else:
            mu = x.mean(dim=(0, 2, 3), keepdim=True) if x.dim() == 4 else x.mean(dim=0, keepdim=True)
            sigma = x.std(dim=(0, 2, 3), keepdim=True) if x.dim() == 4 else x.std(dim=0, keepdim=True)
        return (x - mu) / sigma
        
    elif norm_type == 'layernorm':
        if x.dim() == 4:
            x_flat = x.flatten(start_dim=1)
        else:
            x_flat = x
        norm = x_flat.norm(p=2, dim=1, keepdim=True)
        return x_flat / (norm + eps)
        
    elif norm_type == 'groupnorm':
        N, C, H, W = x.shape
        assert C % num_groups == 0
        G = num_groups
        C_per_g = C // G
        x_grouped = x.view(N, G, C_per_g, H, W)
        mu = x_grouped.mean(dim=(2, 3, 4), keepdim=True)
        sigma = x_grouped.std(dim=(2, 3, 4), keepdim=True)
        return ((x_grouped - mu) / sigma).view(N, C, H, W)
        
    elif norm_type == 'none':
        return x
    
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")


def compute_activation_l2_norms(activations: List[torch.Tensor]) -> torch.Tensor:
    """Compute ℓ₂ norm of activations for each layer."""
    norms = []
    for act in activations:
        act_flat = act.flatten(start_dim=1)
        l2_per_sample = act_flat.norm(p=2, dim=1)
        mean_l2 = l2_per_sample.mean()
        norms.append(mean_l2.item())
    return torch.tensor(norms)


def measure_gamma(activations_init: List[torch.Tensor],
                  activations_final: List[torch.Tensor],
                  norm_type: str) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Measure representational shift γ = (1/L) * Σ |ln(σ_final / σ_init)|
    and zero-point fluctuation indicator γ_init = (1/L) * Σ |ln(σ_init / σ_ref)|.

    Returns:
        gamma: (1/L) * Σ |ln(σ_final / σ_init)|
        gamma_init: (1/L) * Σ |ln(σ_init / σ_ref)| where σ_ref = 1 (for LN, BN, GN)
                    Returns None for norm_type='none' (no natural σ_ref)
        sigma_init: per-layer ℓ₂ norms at initialization
        sigma_final: per-layer ℓ₂ norms at final state

    γ_init 物理意义：
        - LN: σ_ref = 1（单位球面定义），γ_init = (1/L)Σ|ln(σ_init)| ≈ 0
        - BN: σ_ref = 1（running var 初始化为1），γ_init = (1/L)Σ|ln(σ_init)| ≥ 0
        - GN: σ_ref = 1（归一化定义），γ_init ≈ 0
        - None: 无自然 σ_ref，返回 None
    """
    L = len(activations_init)
    assert L == len(activations_final), "Layer count mismatch"
    
    sigma_init = compute_activation_l2_norms(activations_init)
    sigma_final = compute_activation_l2_norms(activations_final)
    
    # γ: representational shift
    log_ratios = torch.log(sigma_final / sigma_init)
    gamma = torch.abs(log_ratios).mean().item()
    
    # γ_init: zero-point fluctuation indicator
    # σ_ref = 1 for all normalized types (LN, BN, GN)
    # For 'none': no natural reference, return None
    if norm_type in ('batchnorm', 'layernorm', 'groupnorm'):
        # σ_ref = 1 by normalization definition
        sigma_ref = 1.0
        log_init = torch.log(sigma_init / sigma_ref)
        gamma_init = torch.abs(log_init).mean().item()
    else:
        gamma_init = None
        sigma_ref = None
    
    return gamma, gamma_init, sigma_init.numpy(), sigma_final.numpy()


# =============================================================================
# λ_max (MAXIMUM SINGULAR VALUE) VIA POWER ITERATION
# =============================================================================

def power_iteration_single_layer(W: torch.Tensor, 
                                  num_iterations: int = 20,
                                  tol: float = 1e-6) -> float:
    """Compute λ_max (largest singular value) via power iteration.
    
    For Conv2d weights (outC, inC, kH, kW), reshapes to (outC, inC*kH*kW).
    Uses W.T @ W (square matrix) for power iteration, then takes sqrt.
    This handles rectangular matrices correctly.
    """
    if W.dim() == 4:  # Conv2d
        W_mat = W.reshape(W.shape[0], -1)  # (outC, inC*kH*kW)
    elif W.dim() == 2:  # Linear
        W_mat = W
    else:
        raise ValueError(f"Unsupported weight shape: {W.shape}")
    
    # For rectangular matrices, use W.T @ W (square) and take sqrt
    # This gives eigenvalue = singular_value^2
    if W_mat.shape[0] != W_mat.shape[1]:
        # Rectangular case: use W.T @ W for power iteration
        M = W_mat.T @ W_mat  # (inC*kH*kW, inC*kH*kW) - square matrix
        d = M.shape[0]
    else:
        # Square case: use W directly
        M = W_mat
        d = M.shape[0]
    
    torch.manual_seed(42)
    b = torch.randn(d, device=M.device)
    b = b / b.norm()
    
    lambda_prev = 0.0
    
    for _ in range(num_iterations):
        Mb = M @ b
        lambda_curr = Mb.norm().item()
        b = Mb / lambda_curr
        
        if abs(lambda_curr - lambda_prev) < tol:
            break
        lambda_prev = lambda_curr
    
    # lambda_curr is eigenvalue of M = W.T @ W = singular_value^2
    # So singular_value = sqrt(lambda_curr)
    return torch.sqrt(torch.tensor(lambda_curr, device=M.device)).item()


def measure_lambda_max(model: nn.Module, 
                       device: torch.device,
                       num_iterations: int = 20) -> Dict[str, float]:
    """Measure λ_max for all weight layers in a model."""
    model.eval()
    lambda_max_dict = {}
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            W = module.weight.data.to(device)
            lambda_max = power_iteration_single_layer(W, num_iterations)
            lambda_max_dict[name] = lambda_max
    
    return lambda_max_dict


def measure_lambda_max_mean(model: nn.Module,
                           device: torch.device,
                           num_iterations: int = 20) -> float:
    """Measure mean λ_max across all layers."""
    lambda_dict = measure_lambda_max(model, device, num_iterations)
    return np.mean(list(lambda_dict.values()))


# =============================================================================
# β (D-SCALING EXPONENT) FITTING
# =============================================================================

def d_scaling_model(D: np.ndarray, alpha: float, beta: float, 
                    E_floor: float) -> np.ndarray:
    """D-scaling law model: ℒ = α · D^(-β) + E_floor"""
    return alpha * np.power(D, -beta) + E_floor


def fit_beta(losses: np.ndarray, 
             D_values: np.ndarray,
             r2_threshold: float = 0.995) -> Tuple[Optional[float], Optional[float], float]:
    """Fit D-scaling law ℒ = α·D^(-β) + E_floor to extract β."""
    try:
        E_floor_guess = np.min(losses)
        alpha_guess = np.max(losses) - E_floor_guess
        beta_guess = 0.5
        
        popt, pcov = curve_fit(
            d_scaling_model,
            D_values,
            losses,
            p0=[alpha_guess, beta_guess, E_floor_guess],
            bounds=([0, 0, 0], [np.inf, 5, np.max(losses)]),
            maxfev=10000
        )
        
        alpha, beta, E_floor = popt
        
        y_pred = d_scaling_model(D_values, alpha, beta, E_floor)
        ss_res = np.sum((losses - y_pred) ** 2)
        ss_tot = np.sum((losses - np.mean(losses)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        if r_squared >= r2_threshold:
            return beta, alpha, r_squared
        else:
            return None, None, r_squared
            
    except Exception as e:
        return None, None, 0.0


# =============================================================================
# STATIONARITY DETECTION
# =============================================================================

def compute_loss_monotonicity_ratio(losses: np.ndarray, 
                                    window_size: int = 20) -> float:
    """Compute monotonicity ratio for a loss window."""
    if len(losses) < window_size:
        return 0.0
    
    window = losses[-window_size:]
    deltas = np.diff(window)
    
    if len(deltas) < 2:
        return 1.0
    
    signs = np.sign(deltas)
    sign_changes = np.sum(signs[:-1] != signs[1:])
    same_sign_count = len(signs) - 1 - sign_changes
    
    monotonicity_ratio = same_sign_count / (len(signs) - 1)
    
    return monotonicity_ratio


def is_stationary(loss_history: np.ndarray, 
                  window_size: int = 20,
                  monotonicity_threshold: float = 0.85) -> Tuple[bool, int]:
    """Check if training has reached stationarity."""
    if len(loss_history) < window_size:
        return False, -1
    
    for start_idx in range(len(loss_history) - window_size + 1):
        window = loss_history[start_idx:start_idx + window_size]
        mono_ratio = compute_loss_monotonicity_ratio(window, window_size)
        
        if mono_ratio >= monotonicity_threshold:
            stationary_epoch = start_idx + window_size - 1
            return True, stationary_epoch
    
    return False, -1


def find_stationary_epoch(losses_per_epoch: List[float],
                          window_size: int = 20,
                          monotonicity_threshold: float = 0.85) -> Tuple[bool, int, float]:
    """Find the epoch at which training becomes stationary."""
    losses = np.array(losses_per_epoch)
    
    is_stat, stat_epoch = is_stationary(losses, window_size, monotonicity_threshold)
    
    if is_stat:
        window = losses[stat_epoch - window_size + 1:stat_epoch + 1]
        return True, stat_epoch, float(np.mean(window))
    
    return False, -1, float(np.mean(losses[-window_size:]))


# =============================================================================
# η_crit (CRITICAL LEARNING RATE) VIA BINARY SEARCH
# =============================================================================

def find_eta_crit(model_factory, 
                 train_loader,
                 D: int,
                 norm_type: str,
                 device: torch.device,
                 eta_grid: List[float],
                 convergence_threshold: float = 0.3,
                 r2_threshold: float = 0.995) -> Tuple[Optional[float], Dict]:
    """Find η_crit: maximum convergent learning rate via binary search."""
    results = {}
    
    for eta in eta_grid:
        model = model_factory(D=D, norm_type=norm_type).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=eta, momentum=0.9)
        
        losses = []
        for epoch in range(20):
            epoch_loss = 0.0
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss / len(train_loader))
        
        final_loss = losses[-1]
        init_loss = losses[0]
        loss_reduction = (init_loss - final_loss) / init_loss
        
        converged = loss_reduction >= convergence_threshold
        
        results[eta] = {
            'converged': converged,
            'loss_reduction': loss_reduction,
            'final_loss': final_loss,
            'init_loss': init_loss,
            'losses': losses
        }
    
    eta_crit = None
    for eta in sorted(results.keys(), reverse=True):
        if results[eta]['converged']:
            eta_crit = eta
            break
    
    return eta_crit, results


# =============================================================================
# β vs ln(γ) LINEAR REGRESSION & F-TEST
# =============================================================================

def fit_beta_vs_ln_gamma(beta_gamma_pairs: List[Tuple[float, float]],
                         norm_type: str) -> Dict:
    """Fit β = m * ln(γ) + c linear regression."""
    betas = np.array([p[0] for p in beta_gamma_pairs])
    gammas = np.array([p[1] for p in beta_gamma_pairs])
    
    valid_mask = gammas > 0
    if np.sum(valid_mask) < 3:
        return {'s': None, 'c': None, 'r_squared': 0, 'n_points': len(gammas)}
    
    ln_gamma = np.log(gammas[valid_mask])
    betas_valid = betas[valid_mask]
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(ln_gamma, betas_valid)
    
    return {
        's': slope,
        'c': intercept,
        'r_squared': r_value ** 2,
        'p_value': p_value,
        'std_err': std_err,
        'n_points': int(np.sum(valid_mask)),
        'norm_type': norm_type
    }


def f_test_equal_slopes(regression_results: Dict[str, Dict],
                        alpha: float = 0.05) -> Dict:
    """F-test for equal slopes — delegates to analysis.statistics implementation."""
    from analysis.statistics import f_test_equal_slopes as _f_test
    return _f_test(regression_results, alpha)


# =============================================================================
# SAVE/LOAD UTILITIES (KAGGLE-ADAPTED)
# =============================================================================

def _json_serializer(obj):
    """JSON serializer for numpy types and other non-serializable objects."""
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_results(results: Dict, filepath: str):
    """
    Save results dict to JSON file.
    
    Args:
        results: Dictionary to save
        filepath: Output file path
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=_json_serializer)


def load_results(filepath: str) -> Dict:
    """Load results dict from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_run_result(result: Dict, output_dir: str):
    """
    Save a single run's result to a dedicated JSON file.
    
    This enables:
    - Incremental saving (one file per run)
    - Easy resume (check file existence)
    - Kaggle timeout safety (partial results preserved)
    
    File format: {output_dir}/result_{run_id}.json
    
    Args:
        result: Result dictionary from train_and_measure
        output_dir: Directory to save in
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract run_id from result
    if 'config' in result:
        cfg = result['config']
        from experiments.train import get_run_id
        run_id = get_run_id(cfg.get('D'), cfg.get('norm_type'), cfg.get('lr'), cfg.get('seed'))
    else:
        # Generate a hash-based ID if config not available
        run_id = f"run_{hash(str(result))[:16]}"
    
    filepath = os.path.join(output_dir, f"result_{run_id}.json")
    save_results(result, filepath)


def load_run_results(output_dir: str) -> List[Dict]:
    """
    Load all run results from a directory.
    
    Scans for files matching pattern: result_*.json
    
    Args:
        output_dir: Directory containing result files
        
    Returns:
        List of result dictionaries
    """
    results = []
    
    if not os.path.exists(output_dir):
        return results
    
    for fname in os.listdir(output_dir):
        if fname.startswith('result_') and fname.endswith('.json'):
            filepath = os.path.join(output_dir, fname)
            try:
                with open(filepath, 'r') as f:
                    result = json.load(f)
                    # Validate it has required fields
                    if isinstance(result, dict) and 'config' in result:
                        results.append(result)
            except (json.JSONDecodeError, IOError) as e:
                print(f"[Warning] Could not load {fname}: {e}")
                continue
    
    return results


def append_to_results(results: List[Dict], output_dir: str, filename: str = 'results_all.json'):
    """
    Append results to a cumulative JSON file.
    
    For non-Kaggle use where append-style saves are preferred.
    
    Args:
        results: List of result dicts to append
        output_dir: Directory to save in
        filename: Name of cumulative results file
    """
    filepath = os.path.join(output_dir, filename)
    
    # Load existing if present
    existing = []
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                existing = json.load(f).get('runs', [])
        except (json.JSONDecodeError, IOError):
            existing = []
    
    # Append new results
    existing.extend(results)
    
    # Save
    save_results({'runs': existing}, filepath)


# =============================================================================
# PROTOCOL-SPECIFIC CONSTANTS
# =============================================================================

PROTOCOL_CONSTANTS = {
    # Stationarity detection
    'stationary_window': 20,
    'stationary_threshold': 0.85,
    
    # D-scaling fit
    'r2_threshold': 0.995,
    
    # Power iteration
    'power_iter_steps': 20,
    
    # EOS verification
    'eos_lambda_eta_tolerance': (1.5, 3.0),
    
    # Convergence
    'loss_reduction_threshold': 0.3,
    
    # Phase specific
    'phase_0_epochs': 200,
    'phase_1_epochs': 40,
    
    # Learning rate grid (Phase 1)
    'lr_grid': [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0],
    
    # D values (Phase 1)
    'D_values': [32, 64, 128, 256, 512, 1024],
}
