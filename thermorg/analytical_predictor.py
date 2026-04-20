#!/usr/bin/env python3
"""
ThermoRG Analytical Predictor
==============================

Pure mathematical prediction of network training loss.

This module contains ONLY mathematical formulas - NO torch training,
NO backward pass, NO DataLoader. It computes the predicted loss from
architecture parameters using the calibrated thermodynamic scaling law.

Mathematical Framework:
-----------------------
The loss L for a network with depth D and topology quality J_topo is:

    L(D) = α · D^(-β) + E_floor

where the floor energy decomposes as:

    E_floor = C/D − B · J_topo^ν

The depth exponent β follows a "cooling law" depending on the
network's initialization quality γ:

    β = 0.425 · ln(γ/γ_c) + 0.893

where γ_c ≈ 1.0 is the critical initialization quality.

Usage:
------
    >>> from thermorg.analytical_predictor import AnalyticalPredictor
    >>> predictor = AnalyticalPredictor()
    >>> loss = predictor.predict(width=64, depth=5, norm_type='bn', J_topo=0.75)
    >>> print(f"Predicted loss: {loss:.4f}")

Reference: ThermoRG Theory Framework v5-v8
"""

from __future__ import annotations

from typing import Optional, Dict, Any

import numpy as np


# =============================================================================
# Default Calibrated Parameters
# =============================================================================

# Default parameters for CIFAR-10 (calibrated on Phase B2 data).
# WARNING: These are dataset-specific. For other datasets, use ThermoCalibrator
# to calibrate per-dataset. See thermorg.calibration.get_default_calibration_data().
DEFAULT_PARAMS = {
    'alpha_bn': 0.45,     # α for BatchNorm networks
    'alpha_none': 0.68,   # α for networks without normalization
    'beta': 0.85,         # depth exponent
    'gamma_c': 1.0,       # critical initialization quality
    'C': 0.0,             # C in E_floor = E_task + C/D − B·J_topo^ν
    'E_task': 0.35,       # E_task (dataset-specific, CIFAR-10 calibrated)
    'B': 0.10,             # B coefficient (dataset-specific, CIFAR-10 calibrated)
    'nu': 1.0,            # exponent on J_topo
    'dataset': 'cifar10',  # Dataset identifier (for warning)
}


# =============================================================================
# D-Scaling Law Functions
# =============================================================================

def D_scaling_law(
    D_eff: float,
    alpha: float,
    beta: float
) -> float:
    """
    Compute the D-scaling contribution: α · D^(-β).
    
    Args:
        D_eff: Effective degrees of freedom (proportional to depth × width)
        alpha: Prefactor (depends on norm type)
        beta: Depth exponent (cooling law parameter)
        
    Returns:
        D-scaling contribution to loss
    """
    if D_eff <= 0:
        return float('inf')
    return alpha * (D_eff ** (-beta))


def E_floor_decomposition(
    D_eff: float,
    J_topo: float,
    E_task: float = DEFAULT_PARAMS['E_task'],
    C: float = DEFAULT_PARAMS['C'],
    B: float = DEFAULT_PARAMS['B'],
    nu: float = DEFAULT_PARAMS['nu']
) -> float:
    """
    Compute the floor energy: E_task + C/D − B·J_topo^ν.
    
    Args:
        D_eff: Effective degrees of freedom
        J_topo: Topology quality (0 to 1, higher is better)
        E_task: Task-intrinsic irreducible error
        C: Coefficient for 1/D term
        B: Coefficient for J_topo term (negative contribution)
        nu: Exponent on J_topo
        
    Returns:
        Floor energy value
    """
    if D_eff <= 0 or J_topo <= 0:
        return float('inf')
    
    term1 = C / D_eff if D_eff > 0 else 0.0
    term2 = -B * (J_topo ** nu)
    
    return E_task + term1 + term2


def cooling_law(
    gamma: float,
    gamma_c: float = DEFAULT_PARAMS['gamma_c'],
    slope: float = 0.425,
    intercept: float = 0.893
) -> float:
    """
    Compute the depth exponent β from initialization quality γ.
    
    β(γ) = 0.425 · ln(γ/γ_c) + 0.893
    
    Args:
        gamma: Initialization quality metric (γ = J_topo at initialization)
        gamma_c: Critical initialization quality (default: 1.0)
        slope: Slope of the cooling law (default: 0.425)
        intercept: Intercept of the cooling law (default: 0.893)
        
    Returns:
        Depth exponent β
    """
    if gamma <= 0:
        gamma = 1e-10  # Avoid log(0)
    
    beta = slope * np.log(gamma / gamma_c) + intercept
    
    # Clamp to reasonable range
    return float(np.clip(beta, 0.1, 2.0))


# =============================================================================
# Main Predictor Class
# =============================================================================

class AnalyticalPredictor:
    """
    Predicts training loss from architecture parameters using thermodynamic scaling law.
    
    This predictor implements the calibrated thermodynamic equation of state:
    
        L(D) = α · D^(-β) + E_floor
        
    where:
        E_floor = E_task + C/D − B·J_topo^ν
        
    Args:
        E_task: Task-intrinsic irreducible error (dataset-specific, required)
        B: Sensitivity of E_floor to J_topo (dataset-specific, required)
        C: Finite-width correction coefficient
        J_topo: Topology quality (required, no default — compute via topology_calculator)
    
    Example:
        >>> from thermorg import AnalyticalPredictor, compute_J_topo
        >>> from thermorg.calibration import get_cifar10_calibration
        >>> cal = get_cifar10_calibration()
        >>> predictor = AnalyticalPredictor(E_task=cal['E_task'], B=cal['B'])
        >>> model = MyNet()
        >>> J, _ = compute_J_topo(model)
        >>> loss = predictor.predict(width=64, depth=5, norm_type='bn', J_topo=J)
    """
    
    def __init__(
        self,
        E_task: float,
        B: float,
        C: float = 0.0,
        alpha_bn: float = 0.45,
        alpha_none: float = 0.68,
        beta: float = 0.85,
        gamma_c: float = 1.0,
        nu: float = 1.0,
        dataset: str = 'unknown',
        use_cooling_law: bool = False
    ):
        """
        Initialize the predictor with dataset-specific calibration.
        
        Args:
            E_task: Task-intrinsic irreducible error (from calibration).
                   Dataset-specific — MUST be calibrated per dataset.
            B: Sensitivity of E_floor to J_topo (from calibration).
               Dataset-specific — MUST be calibrated per dataset.
            C: Finite-width correction coefficient (default: 0.0)
            alpha_bn: α for BatchNorm networks (default: 0.45)
            alpha_none: α for no-normalization networks (default: 0.68)
            beta: Depth exponent / cooling law parameter (default: 0.85)
            gamma_c: Critical initialization quality (default: 1.0)
            nu: Exponent on J_topo in E_floor (default: 1.0)
            dataset: Dataset name for reference (e.g., 'cifar10')
            use_cooling_law: If True, compute β from γ via cooling_law()
        """
        self.E_task = E_task
        self.B = B
        self.C = C
        self.alpha_bn = alpha_bn
        self.alpha_none = alpha_none
        self.beta = beta
        self.gamma_c = gamma_c
        self.nu = nu
        self.dataset = dataset
        self.use_cooling_law = use_cooling_law
    
    def compute_D_eff(
        self,
        width: int,
        depth: int,
        kernel_size: int = 3,
        expansion_factor: float = 1.0
    ) -> float:
        """
        Compute effective degrees of freedom.
        
        D_eff ≈ width × depth for plain ConvNets.
        For more complex architectures, use the topology_calculator module.
        
        Args:
            width: Number of channels per layer
            depth: Number of layers
            kernel_size: Convolution kernel size (default: 3)
            expansion_factor: Multiplier for skip connections, etc.
            
        Returns:
            Estimated D_eff
        """
        # For plain ConvNets: D_eff ≈ width × depth
        # The kernel size and other factors contribute as multiplicative corrections
        base_D = width * depth
        
        # Kernel size correction (larger kernels → more parameters per layer)
        kernel_correction = (kernel_size ** 2) / (3 ** 2)
        
        return base_D * kernel_correction * expansion_factor
    
    def get_alpha(self, norm_type: str) -> float:
        """
        Get α parameter for normalization type.
        
        Args:
            norm_type: 'bn' for BatchNorm, 'none' or anything else for no norm
            
        Returns:
            α value
        """
        if norm_type.lower() == 'bn':
            return self.alpha_bn
        else:
            return self.alpha_none
    
    def get_beta(self, gamma: Optional[float] = None) -> float:
        """
        Get β parameter (depth exponent).
        
        If use_cooling_law is True and gamma is provided, compute β from γ.
        Otherwise, return the fixed β.
        
        Args:
            gamma: Initialization quality γ (used if use_cooling_law=True)
            
        Returns:
            β value
        """
        if self.use_cooling_law and gamma is not None:
            return cooling_law(gamma, gamma_c=self.gamma_c)
        else:
            return self.beta
    
    def predict(
        self,
        width: int,
        depth: int,
        J_topo: float,
        norm_type: str = 'bn',
        gamma: Optional[float] = None,
        kernel_size: int = 3,
        skip: bool = False,
        return_components: bool = False
    ) -> float:
        """
        Predict training loss for an architecture.
        
        Args:
            width: Number of channels per layer
            depth: Number of layers
            J_topo: Topology quality metric (0 to 1). REQUIRED — compute via
                    thermorg.topology_calculator.compute_J_topo() from initialized weights.
            norm_type: 'bn' for BatchNorm, 'none' for no normalization
            gamma: Initialization quality. If None and use_cooling_law=True,
                   uses J_topo as proxy.
            kernel_size: Convolution kernel size (default: 3)
            skip: Whether architecture has skip connections
            return_components: If True, return (L, D_scaling, E_floor) tuple
                             If False, return just L
                               
        Returns:
            Predicted loss L (scalar), or (L, D_scaling, E_floor) if return_components=True
        """
        # Compute D_eff
        expansion = 1.2 if skip else 1.0
        D_eff = self.compute_D_eff(width, depth, kernel_size, expansion)
        
        # Get α and β
        alpha = self.get_alpha(norm_type)
        
        if gamma is not None:
            beta = self.get_beta(gamma)
        elif J_topo is not None and self.use_cooling_law:
            beta = self.get_beta(J_topo)
        else:
            beta = self.get_beta()
        
        # Compute components
        D_term = D_scaling_law(D_eff, alpha, beta)
        E_floor = E_floor_decomposition(
            D_eff, J_topo,
            E_task=self.E_task,
            C=self.C,
            B=self.B,
            nu=self.nu
        )
        
        # Total loss
        L = D_term + E_floor
        
        if return_components:
            return L, D_term, E_floor
        else:
            return L
    
    def predict_from_D_eff(
        self,
        D_eff: float,
        J_topo: float,
        norm_type: str = 'bn',
        gamma: Optional[float] = None
    ) -> float:
        """
        Predict loss directly from D_eff (skipping D_eff computation).
        
        Useful when D_eff is already computed via topology_calculator.
        
        Args:
            D_eff: Effective degrees of freedom (from topology_calculator)
            J_topo: Topology quality (0 to 1). REQUIRED.
            norm_type: 'bn' or 'none'
            gamma: Initialization quality (for cooling law)
            
        Returns:
            Predicted loss
        """
        alpha = self.get_alpha(norm_type)
        
        if gamma is not None:
            beta = self.get_beta(gamma)
        elif self.use_cooling_law:
            beta = self.get_beta(J_topo)
        else:
            beta = self.get_beta()
        
        D_term = D_scaling_law(D_eff, alpha, beta)
        E_floor = E_floor_decomposition(
            D_eff, J_topo,
            E_task=self.E_task,
            C=self.C,
            B=self.B,
            nu=self.nu
        )
        
        return D_term + E_floor
    
    def rank_architectures(
        self,
        architectures: list,
        key: str = 'name'
    ) -> list:
        """
        Rank architectures by predicted loss (ascending = better).
        
        Args:
            architectures: List of dicts with keys: width, depth, norm_type, J_topo
            key: Key name for architecture identifier
            
        Returns:
            Sorted list of architectures with predicted losses added
        """
        for arch in architectures:
            loss = self.predict(
                width=arch['width'],
                depth=arch['depth'],
                norm_type=arch.get('norm_type', 'bn'),
                J_topo=arch.get('J_topo', 0.75)
            )
            arch['predicted_loss'] = loss
        
        return sorted(architectures, key=lambda x: x['predicted_loss'])


# =============================================================================
# Convenience Functions
# =============================================================================

def predict_loss(
    width: int,
    depth: int,
    norm_type: str = 'bn',
    J_topo: float = 0.75,
    params: Optional[Dict[str, float]] = None
) -> float:
    """
    Convenience function to predict loss for a single architecture.
    
    Args:
        width: Number of channels per layer
        depth: Number of layers
        norm_type: 'bn' or 'none'
        J_topo: Topology quality metric
        params: Optional calibrated parameters
        
    Returns:
        Predicted loss
    """
    predictor = AnalyticalPredictor(params=params)
    return predictor.predict(width=width, depth=depth, norm_type=norm_type, J_topo=J_topo)


# =============================================================================
# Example Usage
# =============================================================================

def main():
    """
    Demonstrate the analytical predictor with example architectures.
    """
    print("=" * 60)
    print("ThermoRG Analytical Predictor - Demo")
    print("=" * 60)
    
    predictor = AnalyticalPredictor()
    
    # Example architectures
    architectures = [
        {'width': 48, 'depth': 3, 'norm_type': 'bn', 'J_topo': 0.85},
        {'width': 48, 'depth': 5, 'norm_type': 'bn', 'J_topo': 0.78},
        {'width': 48, 'depth': 6, 'norm_type': 'bn', 'J_topo': 0.72},
        {'width': 64, 'depth': 5, 'norm_type': 'bn', 'J_topo': 0.75},
        {'width': 64, 'depth': 6, 'norm_type': 'bn', 'J_topo': 0.70},
        {'width': 96, 'depth': 6, 'norm_type': 'bn', 'J_topo': 0.68},
        {'width': 48, 'depth': 5, 'norm_type': 'none', 'J_topo': 0.80},
        {'width': 64, 'depth': 5, 'norm_type': 'none', 'J_topo': 0.77},
    ]
    
    print("\nPredicted losses for calibration architectures:")
    print("-" * 60)
    print(f"{'Config':<20} {'D_eff':<8} {'D_term':<8} {'E_floor':<8} {'L_pred':<8}")
    print("-" * 60)
    
    ranked = predictor.rank_architectures(architectures)
    
    for arch in ranked:
        L, D_term, E_floor = predictor.predict(
            width=arch['width'],
            depth=arch['depth'],
            norm_type=arch['norm_type'],
            J_topo=arch['J_topo'],
            return_components=True
        )
        D_eff = predictor.compute_D_eff(arch['width'], arch['depth'])
        name = f"W{arch['width']}-D{arch['depth']}-{arch['norm_type']}"
        print(f"{name:<20} {D_eff:<8.1f} {D_term:<8.4f} {E_floor:<8.4f} {L:<8.4f}")
    
    print("-" * 60)
    print("\nRanking (best to worst):")
    for i, arch in enumerate(ranked):
        name = f"W{arch['width']}-D{arch['depth']}-{arch['norm_type']}"
        print(f"  {i+1}. {name}: L = {arch['predicted_loss']:.4f}")


if __name__ == '__main__':
    main()