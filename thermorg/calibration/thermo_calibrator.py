#!/usr/bin/env python3
"""
ThermoRG Thermodynamic Calibrator
==================================

Calibrates the thermodynamic equation of state parameters from observed training data.

The scaling law:
    L(D) = α · D^(-β) + E_floor

where:
    E_floor = E_task + C/D − B · J_topo^ν

Input:
    - 8 calibration architectures with their early training losses
    - J_topo values computed via topology_calculator.py

Output:
    - α_type: α parameter for different norm types
    - β: depth exponent (cooling law parameter)
    - C, B: E_floor decomposition parameters

The calibration process:
1. Fit E_floor from observed losses (subtract α·D^(-β) contribution)
2. Fit C and B from E_floor vs J_topo relationship
3. Fit β from the D-scaling across depths

Reference: ThermoRG Theory Framework v5-v8
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.stats import pearsonr


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CalibrationResult:
    """Result of thermodynamic calibration."""
    
    # α parameters for different norm types
    alpha_bn: float  # α for BatchNorm networks
    alpha_none: float  # α for networks without normalization
    
    # β parameter (depth exponent / cooling law)
    beta: float
    
    # E_floor decomposition: E_floor = E_task + C/D − B·J_topo^ν
    C: float
    B: float
    nu: float = 1.0  # exponent on J_topo
    n_architectures: int = 8
    calibration_epochs: int = 200
    
    # R² goodness of fit (must come after fields with defaults)
    r2_E_floor: float = 0.0
    r2_D_scaling: float = 0.0
    
    def __repr__(self) -> str:
        return (
            f"CalibrationResult(\n"
            f"  α_bn={self.alpha_bn:.4f}, α_none={self.alpha_none:.4f}\n"
            f"  β={self.beta:.4f}\n"
            f"  E_floor: C={self.C:.4f}, B={self.B:.4f}, ν={self.nu:.1f}\n"
            f"  R²: E_floor={self.r2_E_floor:.4f}, D_scaling={self.r2_D_scaling:.4f}\n"
            f")"
        )
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            'alpha_bn': self.alpha_bn,
            'alpha_none': self.alpha_none,
            'beta': self.beta,
            'C': self.C,
            'B': self.B,
            'nu': self.nu,
            'r2_E_floor': self.r2_E_floor,
            'r2_D_scaling': self.r2_D_scaling,
            'n_architectures': self.n_architectures,
            'calibration_epochs': self.calibration_epochs,
        }


@dataclass
class ArchitectureSpec:
    """Specification for a single architecture."""
    width: int
    depth: int
    norm_type: str  # 'bn' or 'none'
    skip: bool
    D_eff: Optional[float] = None  # Computed from topology
    J_topo: Optional[float] = None  # Computed from topology
    early_loss: Optional[float] = None  # Observed training loss
    final_loss: Optional[float] = None  # Observed final loss


# =============================================================================
# Calibrator Class
# =============================================================================

class ThermoCalibrator:
    """
    Calibrates thermodynamic equation of state from training data.
    
    The calibrator takes observed losses from a set of calibration architectures
    and fits the parameters of the scaling law:
    
        L(D) = α · D^(-β) + E_floor
        
    where:
        E_floor = C/D + B·J_topo^ν
        
    Usage:
        >>> calibrator = ThermoCalibrator()
        >>> result = calibrator.calibrate(calibration_data)
        >>> print(result)
    """
    
    # Default initial guesses for fitting
    DEFAULT_ALPHA_0 = 0.5
    DEFAULT_BETA_0 = 0.8
    DEFAULT_C_0 = 0.1
    DEFAULT_B_0 = 0.5
    DEFAULT_NU_0 = 1.0
    
    # Bounds for parameters
    BOUNDS = {
        'alpha': (0.01, 5.0),
        'beta': (0.1, 2.0),
        'C': (0.0, 2.0),
        'B': (0.0, 5.0),
        'nu': (0.5, 2.0),
    }
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the calibrator.
        
        Args:
            verbose: If True, print fitting progress
        """
        self.verbose = verbose
        self.result: Optional[CalibrationResult] = None
    
    def fit_E_floor(
        self,
        architectures: List[ArchitectureSpec],
        early_losses: Dict[str, float]
    ) -> np.ndarray:
        """
        Fit E_floor values from observed early losses.
        
        E_floor is estimated by removing the D-scaling contribution:
            E_floor ≈ L_obs - α · D^(-β)
            
        This requires knowing β, so this is typically called after fit_beta.
        
        Args:
            architectures: List of architecture specifications
            early_losses: Dict mapping architecture name to observed loss
            
        Returns:
            Array of fitted E_floor values
        """
        E_floors = []
        
        for arch in architectures:
            loss = early_losses.get(f"{arch.width}/{arch.depth}/{arch.norm_type}/{arch.skip}", None)
            if loss is None and arch.early_loss is not None:
                loss = arch.early_loss
            
            if loss is not None and arch.D_eff is not None:
                # E_floor ≈ L_obs - α · D^(-β)
                # We use a preliminary estimate of α and β
                alpha = self.DEFAULT_ALPHA_0 if arch.norm_type == 'bn' else self.DEFAULT_ALPHA_0 * 1.5
                beta = self.DEFAULT_BETA_0
                D_term = alpha * (arch.D_eff ** (-beta))
                E_floor = loss - D_term
                E_floors.append(max(E_floor, 0.01))  # Floor at small positive value
            else:
                E_floors.append(np.nan)
        
        return np.array(E_floors)
    
    def fit_E_floor_decomposition(
        self,
        J_topo_values: np.ndarray,
        E_floor_values: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Fit the E_floor decomposition: E_floor = C/D + B·J_topo^ν
        
        Args:
            J_topo_values: Array of J_topo values
            E_floor_values: Array of fitted E_floor values
            
        Returns:
            Tuple of (C, B, nu)
        """
        # Filter out NaN values
        mask = ~np.isnan(J_topo_values) & ~np.isnan(E_floor_values) & (J_topo_values > 0)
        if mask.sum() < 3:
            return self.DEFAULT_C_0, self.DEFAULT_B_0, self.DEFAULT_NU_0
        
        J = J_topo_values[mask]
        E = E_floor_values[mask]
        
        # Simple linear fit for initial guess
        # E_floor ≈ B·J_topo (ignoring C/D for simplicity)
        from numpy.polynomial import polynomial as P
        try:
            # Fit: E = B * J^nu
            # Take log: log(E) = log(B) + nu * log(J)
            log_J = np.log(J + 1e-10)
            log_E = np.log(E + 1e-10)
            
            # Linear regression in log space
            coeffs = np.polyfit(log_J, log_E, 1)
            nu_0 = max(min(coeffs[0], 2.0), 0.5)
            log_B_0 = coeffs[1]
            B_0 = np.exp(log_B_0)
        except:
            nu_0 = self.DEFAULT_NU_0
            B_0 = self.DEFAULT_B_0
        
        # More careful fit using scipy
        def model(x, B, nu):
            return B * (x ** nu)
        
        try:
            popt, _ = curve_fit(
                model, J, E,
                p0=[B_0, nu_0],
                bounds=([0, 0.5], [5.0, 2.0]),
                maxfev=5000
            )
            B_fit, nu_fit = popt
        except:
            B_fit = B_0
            nu_fit = nu_0
        
        # C is estimated from the intercept (when D → ∞, J_topo → 0)
        # For now, set C=0 (C/D term becomes negligible for large D)
        C_fit = 0.0
        
        return C_fit, B_fit, nu_fit
    
    def fit_beta(
        self,
        architectures: List[ArchitectureSpec],
        losses: Dict[str, float],
        alpha_bn: float,
        alpha_none: float
    ) -> float:
        """
        Fit the β parameter from depth variation.
        
        For architectures with same width but different depth,
        the D-scaling law L(D) = α·D^(-β) should hold.
        
        Taking ratio of losses for two depths:
            L1/L2 = (D2/D1)^β
            
        So: β = log(L1/L2) / log(D2/D1)
        
        Args:
            architectures: List of architecture specifications
            losses: Dict mapping arch name to observed loss
            alpha_bn: α for BatchNorm networks
            alpha_none: α for no-norm networks
            
        Returns:
            Fitted β value
        """
        # Group by width and norm_type, vary depth
        depth_groups: Dict[str, List[Tuple[int, float, float]]] = {}
        
        for arch in architectures:
            key = f"{arch.width}/{arch.norm_type}"
            if key not in depth_groups:
                depth_groups[key] = []
            
            loss = losses.get(f"{arch.width}/{arch.depth}/{arch.norm_type}/{arch.skip}", arch.early_loss)
            if loss is not None and arch.D_eff is not None:
                depth_groups[key].append((arch.depth, loss, arch.D_eff))
        
        beta_estimates = []
        
        for key, items in depth_groups.items():
            if len(items) < 2:
                continue
            
            # Sort by depth
            items.sort(key=lambda x: x[0])
            
            # Compute β from consecutive depth pairs
            for i in range(len(items) - 1):
                d1, l1, D1 = items[i]
                d2, l2, D2 = items[i + 1]
                
                if l1 > 0 and l2 > 0 and D1 > 0 and D2 > 0:
                    # β = log(l1/l2) / log(D2/D1)
                    # But we need to account for different α
                    norm_type = key.split('/')[1]
                    alpha = alpha_bn if norm_type == 'bn' else alpha_none
                    
                    # L = α·D^(-β) + E_floor
                    # For same architecture family, E_floor should be similar
                    # So: ΔL ≈ α·(D2^(-β) - D1^(-β))
                    # This is nonlinear, use optimization
                    
                    def residual(beta):
                        pred1 = alpha * (D1 ** (-beta))
                        pred2 = alpha * (D2 ** (-beta))
                        return (l1 - pred1)**2 + (l2 - pred2)**2
                    
                    result = minimize(residual, 0.8, bounds=[(0.1, 2.0)], method='L-BFGS-B')
                    if result.success:
                        beta_estimates.append(result.x[0])
        
        if not beta_estimates:
            return self.DEFAULT_BETA_0
        
        return float(np.median(beta_estimates))
    
    def fit_alpha(
        self,
        architectures: List[ArchitectureSpec],
        losses: Dict[str, float],
        beta: float,
        E_floor_approx: float = 0.2
    ) -> Tuple[float, float]:
        """
        Fit α parameters for BN and no-norm networks.
        
        L(D) = α · D^(-β) + E_floor
        
        For each architecture:
            α ≈ (L_obs - E_floor) · D^β
            
        Args:
            architectures: List of architecture specifications
            losses: Dict mapping arch name to observed loss
            beta: Fitted β parameter
            E_floor_approx: Approximate E_floor value
            
        Returns:
            Tuple of (alpha_bn, alpha_none)
        """
        alpha_bn_estimates = []
        alpha_none_estimates = []
        
        for arch in architectures:
            loss = losses.get(f"{arch.width}/{arch.depth}/{arch.norm_type}/{arch.skip}", arch.early_loss)
            if loss is None:
                loss = arch.early_loss
            
            if loss is not None and arch.D_eff is not None:
                alpha_est = (loss - E_floor_approx) * (arch.D_eff ** beta)
                
                if arch.norm_type == 'bn':
                    alpha_bn_estimates.append(alpha_est)
                else:
                    alpha_none_estimates.append(alpha_est)
        
        alpha_bn = np.median(alpha_bn_estimates) if alpha_bn_estimates else self.DEFAULT_ALPHA_0
        alpha_none = np.median(alpha_none_estimates) if alpha_none_estimates else self.DEFAULT_ALPHA_0 * 1.5
        
        return float(alpha_bn), float(alpha_none)
    
    def calibrate(
        self,
        calibration_data: List[Dict[str, Any]]
    ) -> CalibrationResult:
        """
        Main calibration method.
        
        Args:
            calibration_data: List of dicts with keys:
                - width, depth, norm_type, skip
                - D_eff, J_topo
                - early_loss (observed training loss at ~200 epochs)
                
        Returns:
            CalibrationResult with fitted parameters
        """
        # Convert to ArchitectureSpec list
        architectures = []
        for data in calibration_data:
            arch = ArchitectureSpec(
                width=data['width'],
                depth=data['depth'],
                norm_type=data['norm_type'],
                skip=data.get('skip', False),
                D_eff=data.get('D_eff'),
                J_topo=data.get('J_topo'),
                early_loss=data.get('early_loss'),
            )
            architectures.append(arch)
        
        # Build losses dict
        losses = {}
        for data in calibration_data:
            key = f"{data['width']}/{data['depth']}/{data['norm_type']}/{data.get('skip', False)}"
            losses[key] = data.get('early_loss')
        
        # Step 1: Preliminary α fit
        alpha_bn_0, alpha_none_0 = self.fit_alpha(
            architectures, losses,
            beta=self.DEFAULT_BETA_0,
            E_floor_approx=0.2
        )
        
        if self.verbose:
            print(f"Step 1: Preliminary α: bn={alpha_bn_0:.4f}, none={alpha_none_0:.4f}")
        
        # Step 2: Fit β from depth variation
        beta = self.fit_beta(architectures, losses, alpha_bn_0, alpha_none_0)
        
        if self.verbose:
            print(f"Step 2: Fitted β = {beta:.4f}")
        
        # Step 3: Refine α with fitted β
        alpha_bn, alpha_none = self.fit_alpha(architectures, losses, beta, E_floor_approx=0.2)
        
        if self.verbose:
            print(f"Step 3: Refined α: bn={alpha_bn:.4f}, none={alpha_none:.4f}")
        
        # Step 4: Fit E_floor decomposition
        J_topo_arr = np.array([a.J_topo if a.J_topo else np.nan for a in architectures])
        E_floors = self.fit_E_floor(architectures, losses)
        
        C, B, nu = self.fit_E_floor_decomposition(J_topo_arr, E_floors)
        
        if self.verbose:
            print(f"Step 4: E_floor decomposition: C={C:.4f}, B={B:.4f}, ν={nu:.1f}")
        
        # Compute R² for quality assessment
        r2_E_floor = self._compute_r2_E_floor(architectures, losses, alpha_bn, alpha_none, beta, C, B, nu)
        r2_D_scaling = self._compute_r2_D_scaling(architectures, losses, alpha_bn, alpha_none, beta)
        
        if self.verbose:
            print(f"R² E_floor = {r2_E_floor:.4f}, R² D_scaling = {r2_D_scaling:.4f}")
        
        self.result = CalibrationResult(
            alpha_bn=alpha_bn,
            alpha_none=alpha_none,
            beta=beta,
            C=C,
            B=B,
            nu=nu,
            r2_E_floor=r2_E_floor,
            r2_D_scaling=r2_D_scaling,
            n_architectures=len(architectures),
        )
        
        return self.result
    
    def _compute_r2_E_floor(
        self,
        architectures: List[ArchitectureSpec],
        losses: Dict[str, float],
        alpha_bn: float,
        alpha_none: float,
        beta: float,
        C: float,
        B: float,
        nu: float
    ) -> float:
        """Compute R² for E_floor decomposition."""
        predictions = []
        observations = []
        
        for arch in architectures:
            if arch.D_eff is None or arch.J_topo is None:
                continue
            
            loss = losses.get(f"{arch.width}/{arch.depth}/{arch.norm_type}/{arch.skip}", arch.early_loss)
            if loss is None:
                continue
            
            alpha = alpha_bn if arch.norm_type == 'bn' else alpha_none
            D_term = alpha * (arch.D_eff ** (-beta))
            E_floor_pred = C / arch.D_eff + B * (arch.J_topo ** nu)
            E_floor_obs = loss - D_term
            
            predictions.append(E_floor_pred)
            observations.append(E_floor_obs)
        
        if len(predictions) < 2:
            return 0.0
        
        predictions = np.array(predictions)
        observations = np.array(observations)
        
        ss_res = np.sum((observations - predictions) ** 2)
        ss_tot = np.sum((observations - np.mean(observations)) ** 2)
        
        if ss_tot < 1e-10:
            return 1.0
        
        return 1.0 - ss_res / ss_tot
    
    def _compute_r2_D_scaling(
        self,
        architectures: List[ArchitectureSpec],
        losses: Dict[str, float],
        alpha_bn: float,
        alpha_none: float,
        beta: float
    ) -> float:
        """Compute R² for D-scaling law."""
        predictions = []
        observations = []
        
        for arch in architectures:
            if arch.D_eff is None:
                continue
            
            loss = losses.get(f"{arch.width}/{arch.depth}/{arch.norm_type}/{arch.skip}", arch.early_loss)
            if loss is None:
                continue
            
            alpha = alpha_bn if arch.norm_type == 'bn' else alpha_none
            pred = alpha * (arch.D_eff ** (-beta))
            
            predictions.append(pred)
            observations.append(loss)
        
        if len(predictions) < 2:
            return 0.0
        
        predictions = np.array(predictions)
        observations = np.array(observations)
        
        ss_res = np.sum((observations - predictions) ** 2)
        ss_tot = np.sum((observations - np.mean(observations)) ** 2)
        
        if ss_tot < 1e-10:
            return 1.0
        
        return 1.0 - ss_res / ss_tot


# =============================================================================
# Calibration Data Example
# =============================================================================

def get_default_calibration_data() -> List[Dict[str, Any]]:
    """
    Return default calibration data (8 architectures from Phase B2).
    
    These are representative architectures used for calibrating
    the thermodynamic scaling law.
    
    Returns:
        List of calibration data dicts
    """
    return [
        # Width=48, Depth=3, BN
        {'width': 48, 'depth': 3, 'norm_type': 'bn', 'skip': False,
         'D_eff': 48, 'J_topo': 0.85, 'early_loss': 1.50},
        # Width=48, Depth=5, BN
        {'width': 48, 'depth': 5, 'norm_type': 'bn', 'skip': False,
         'D_eff': 80, 'J_topo': 0.78, 'early_loss': 1.25},
        # Width=48, Depth=6, BN
        {'width': 48, 'depth': 6, 'norm_type': 'bn', 'skip': False,
         'D_eff': 96, 'J_topo': 0.72, 'early_loss': 1.15},
        # Width=64, Depth=5, BN
        {'width': 64, 'depth': 5, 'norm_type': 'bn', 'skip': False,
         'D_eff': 100, 'J_topo': 0.75, 'early_loss': 1.18},
        # Width=64, Depth=6, BN
        {'width': 64, 'depth': 6, 'norm_type': 'bn', 'skip': False,
         'D_eff': 120, 'J_topo': 0.70, 'early_loss': 1.10},
        # Width=96, Depth=6, BN
        {'width': 96, 'depth': 6, 'norm_type': 'bn', 'skip': False,
         'D_eff': 150, 'J_topo': 0.68, 'early_loss': 1.05},
        # Width=48, Depth=5, no norm
        {'width': 48, 'depth': 5, 'norm_type': 'none', 'skip': False,
         'D_eff': 85, 'J_topo': 0.80, 'early_loss': 1.80},
        # Width=64, Depth=5, no norm
        {'width': 64, 'depth': 5, 'norm_type': 'none', 'skip': False,
         'D_eff': 105, 'J_topo': 0.77, 'early_loss': 1.70},
    ]


def create_calibrator_and_calibrate() -> CalibrationResult:
    """
    Convenience function to create calibrator and run with default data.
    
    Returns:
        Calibrated CalibrationResult
    """
    calibrator = ThermoCalibrator(verbose=True)
    calibration_data = get_default_calibration_data()
    result = calibrator.calibrate(calibration_data)
    return result

# =============================================================================
# CIFAR-10 Calibration Preset
# =============================================================================

def get_cifar10_calibration() -> dict:
    """
    Return CIFAR-10 calibrated parameters for AnalyticalPredictor.
    
    These values were calibrated on Phase B2 CIFAR-10 data and should
    ONLY be used for CIFAR-10 experiments. For other datasets, use
    ThermoCalibrator to calibrate per-dataset.
    
    Usage:
        >>> from thermorg.calibration import get_cifar10_calibration
        >>> cal = get_cifar10_calibration()
        >>> predictor = AnalyticalPredictor(**cal)
        >>> loss = predictor.predict(width=64, depth=5, J_topo=0.75, norm_type='bn')
    
    Returns:
        Dict with E_task, B, C, alpha_bn, alpha_none, beta, gamma_c, nu, dataset
    """
    return {
        'E_task': 0.35,     # Task-intrinsic irreducible error for CIFAR-10
        'B': 0.10,          # J_topo sensitivity (calibrated on CIFAR-10)
        'C': 0.0,           # Finite-width correction
        'alpha_bn': 0.45,  # α for BatchNorm networks
        'alpha_none': 0.68, # α for no-normalization networks
        'beta': 0.85,       # Depth exponent
        'gamma_c': 1.0,     # Critical initialization quality
        'nu': 1.0,         # J_topo exponent in E_floor
        'dataset': 'cifar10',
    }
