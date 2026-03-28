# SPDX-License-Identifier: Apache-2.0

"""Main TAS Predictor - Orchestrates all phases.

This is the main entry point for the Thermogeometric Architecture Search pipeline.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Dict, Any, List, Union, TYPE_CHECKING
from dataclasses import dataclass, field

# Import submodules
from .profiling import ManifoldEstimator, SmoothnessEstimator
from .architecture import ArchitectureAnalyzer, JacobianAnalyzer
from .thermodynamics import TemperatureEstimator, ThermalPhaseComputer, CoolingPhaseComputer
from .optimization import ArchitectureSearcher, ConstraintBounds

# Modality and module support
from .modality import BaseModality, ModalityConfig, TabularModality, EmbeddingModality, ModalityRegistry
from .modules import BaseModule, ModuleRegistry

from ..utils.math import product_log, safe_log, safe_divide

if TYPE_CHECKING:
    import torch


@dataclass
class TASConfig:
    """Configuration for TAS profiling.
    
    Attributes:
        d_manifold: Intrinsic manifold dimension (auto-estimate if None)
        s_smoothness: Sobolev smoothness (auto-estimate if None)
        V_grad: Gradient variance (auto-estimate if None)
        eta_lr: Learning rate for thermodynamic computation
        batch_size: Batch size for training
        noise_variance: Noise variance (auto-estimate if None)
        k_neighbors: k for k-NN graph construction
        gamma_T: Critical exponent for thermal phase (shape parameter for ψ)
        gamma_cool: Cooling rate
        epsilon: Entropy factor (default 1.0)
        epsilon_topo: Geometric tolerance for topological isometry check (Phase 6)
        xi_opt: Thermal safety margin for optimality check (Phase 6, default 2/3)
        c_T: Calibration constant for T_c (default 1.0)
        c_gamma: Calibration constant for γ_c (default 1.0)
    """
    d_manifold: Optional[float] = None
    s_smoothness: Optional[float] = None
    V_grad: Optional[float] = None
    eta_lr: float = 1e-3
    batch_size: int = 32
    noise_variance: Optional[float] = None
    k_neighbors: int = 10
    gamma_T: float = 1.0
    gamma_cool: float = 0.0
    epsilon: float = 1.0
    epsilon_topo: float = 0.5  # Phase 6: geometric tolerance
    xi_opt: float = 2/3  # Phase 6: thermal safety margin
    c_T: float = 1.0  # Phase 6: T_c calibration constant
    c_gamma: float = 1.0  # Phase 6: γ_c calibration constant
    

@dataclass 
class TASResult:
    """Results from TAS profiling.
    
    Contains all intermediate and final computed values from the TAS pipeline.
    """
    # Phase 1: Data Geometric Profiling
    d_manifold: float
    s_smoothness: float
    
    # Phase 2: Loss Landscape Profiling
    V_grad: float
    T_c: float
    k_alpha: float
    
    # Phase 3-4: Architecture
    eta_ls: List[float]
    eta_product: float
    d_effs: List[float]
    
    # Phase 5: Thermodynamic Factors
    T_eff: float
    T_tilde: float
    psi_algo: float
    phi_cool: float
    alpha: float
    
    # Phase 6: Optimality Verification
    optimality_result: Optional['OptimalityResult'] = None
    
    # Final prediction
    predicted_scaling_exponent: float = field(init=False)
    
    def __post_init__(self):
        self.predicted_scaling_exponent = self.alpha
        
    def summary(self) -> str:
        """Return human-readable summary of results."""
        base = (
            f"TAS Results:\n"
            f"  d_manifold = {self.d_manifold:.4f}\n"
            f"  s_smoothness = {self.s_smoothness:.4f}\n"
            f"  V_grad = {self.V_grad:.4f}\n"
            f"  T_c = {self.T_c:.4f}\n"
            f"  η_product = {self.eta_product:.4f}\n"
            f"  T_eff = {self.T_eff:.4f}\n"
            f"  T_tilde = {self.T_tilde:.4f}\n"
            f"  Ψ_algo = {self.psi_algo:.4f}\n"
            f"  φ_cool = {self.phi_cool:.4f}\n"
            f"  α = {self.alpha:.4f}"
        )
        if self.optimality_result:
            opt = self.optimality_result
            feasibility = "FEASIBLE" if opt.is_feasible else "INFEASIBLE"
            base += (
                f"\n  Optimality (Phase 6):\n"
                f"    Status: {feasibility}\n"
                f"    C1 (Topological Isometry): {'✓' if opt.c1_satisfied else '✗'}\n"
                f"    C2 (Thermal Safety): {'✓' if opt.c2_satisfied else '✗'}\n"
                f"    J_topo = {opt.J_topo:.4f}\n"
                f"    T_tilde_eff = {opt.T_tilde_eff:.4f}"
            )
        return base


@dataclass
class OptimalityResult:
    """Result from Phase 6: Optimality Verification.
    
    Contains the results of checking thermogeometric feasibility
    conditions C1 and C2.
    
    Attributes:
        is_feasible: True if both C1 and C2 are satisfied
        c1_satisfied: True if topological isometry condition is met
        c2_satisfied: True if thermal safety condition is met
        alpha: Scaling exponent α for this architecture
        J_topo: Topological isometry metric |∑log η_l|
        T_tilde_eff: Effective scaled temperature T̃_eff = T_eff · ε_coupling
        epsilon_coupling: Coupling factor ε_coupling
    """
    is_feasible: bool
    c1_satisfied: bool
    c2_satisfied: bool
    alpha: float
    J_topo: float = 0.0
    T_tilde_eff: float = 0.0
    epsilon_coupling: float = 1.0
    
    def summary(self) -> str:
        """Return human-readable summary."""
        status = "FEASIBLE" if self.is_feasible else "INFEASIBLE"
        return (
            f"OptimalityResult: {status}\n"
            f"  C1 (|∑log η_l| ≤ ε_topo): {'✓' if self.c1_satisfied else '✗'}\n"
            f"  C2 (T̃_eff ≤ ξ_opt · T_c): {'✓' if self.c2_satisfied else '✗'}\n"
            f"  J_topo = {self.J_topo:.4f}\n"
            f"  T_tilde_eff = {self.T_tilde_eff:.4f}\n"
            f"  ε_coupling = {self.epsilon_coupling:.4f}\n"
            f"  α = {self.alpha:.4f}"
        )


# =============================================================================
# Phase 6: Optimality Verification Functions
# =============================================================================

def compute_epsilon_coupling(eta_ls: List[float], d_manifold: float) -> float:
    """Compute coupling factor ε_coupling.
    
    ε_coupling = exp(-(2/d_manifold) · ∑log η_l)
    
    This factor characterizes the coupling between layers through the
    manifold geometry.
    
    Args:
        eta_ls: List of η_l values (layer efficiency factors)
        d_manifold: Intrinsic manifold dimension
        
    Returns:
        Coupling factor ε_coupling (in range [0, 1])
    """
    if d_manifold <= 0 or not eta_ls:
        return 1.0
    
    sum_log_eta = sum(safe_log(eta) for eta in eta_ls)
    epsilon_coupling = np.exp(-(2.0 / d_manifold) * sum_log_eta)
    
    return float(np.clip(epsilon_coupling, 0.0, 1.0))


def check_c1_topological_isometry(J_topo: float, epsilon_topo: float) -> bool:
    """Check C1: Topological Isometry condition.
    
    C1: J_topo = |∑log η_l| ≤ ε_topo
    
    This checks whether the architecture preserves the manifold structure
    within the geometric tolerance.
    
    Args:
        J_topo: Topological isometry metric |∑log η_l|
        epsilon_topo: Geometric tolerance threshold
        
    Returns:
        True if C1 is satisfied (J_topo ≤ ε_topo)
    """
    return J_topo <= epsilon_topo


def check_c2_thermal_safety(
    T_eff: float,
    epsilon_coupling: float,
    T_c: float,
    xi_opt: float,
) -> bool:
    """Check C2: Thermal Safety condition.
    
    C2: T̃_eff = T_eff · ε_coupling ≤ ξ_opt · T_c
    
    This checks whether the effective temperature is within safe bounds
    relative to the critical temperature.
    
    Args:
        T_eff: Effective temperature
        epsilon_coupling: Coupling factor ε_coupling
        T_c: Critical temperature
        xi_opt: Thermal safety margin (default 2/3)
        
    Returns:
        True if C2 is satisfied (T̃_eff ≤ ξ_opt · T_c)
    """
    T_tilde_eff = T_eff * epsilon_coupling
    threshold = xi_opt * T_c
    
    return T_tilde_eff <= threshold


def is_thermogeometrically_feasible(
    eta_ls: List[float],
    T_eff: float,
    T_c: float,
    config: TASConfig,
) -> OptimalityResult:
    """Check thermogeometric feasibility (both C1 and C2).
    
    An architecture is thermogeometrically feasible if:
        C1: |∑log η_l| ≤ ε_topo (topological isometry)
        C2: T_eff · ε_coupling ≤ ξ_opt · T_c (thermal safety)
    
    Args:
        eta_ls: List of η_l values
        T_eff: Effective temperature
        T_c: Critical temperature
        config: TAS configuration with tolerance parameters
        
    Returns:
        OptimalityResult with feasibility status and diagnostics
    """
    # Compute J_topo = |∑log η_l|
    J_topo = abs(sum(safe_log(eta) for eta in eta_ls))
    
    # Compute ε_coupling
    d_manifold = config.d_manifold or 10.0
    epsilon_coupling = compute_epsilon_coupling(eta_ls, d_manifold)
    
    # Check C1
    c1_satisfied = check_c1_topological_isometry(J_topo, config.epsilon_topo)
    
    # Compute T_tilde_eff for C2
    T_tilde_eff = T_eff * epsilon_coupling
    
    # Check C2
    c2_satisfied = check_c2_thermal_safety(
        T_eff, epsilon_coupling, T_c, config.xi_opt
    )
    
    # Overall feasibility
    is_feasible = c1_satisfied and c2_satisfied
    
    return OptimalityResult(
        is_feasible=is_feasible,
        c1_satisfied=c1_satisfied,
        c2_satisfied=c2_satisfied,
        alpha=0.0,  # Alpha computed separately
        J_topo=J_topo,
        T_tilde_eff=T_tilde_eff,
        epsilon_coupling=epsilon_coupling,
    )


class TASProfiler:
    """Main class for Thermogeometric Architecture Search.
    
    This class orchestrates all six phases of the TAS pipeline to predict
    the optimal scaling exponent α for a given architecture and training config.
    
    Supports multiple data modalities (tabular, text, video, audio) and
    custom neural network modules with thermogeometric activation functions.
    
    Example:
        >>> profiler = TASProfiler()
        >>> result = profiler.profile(X, y, architecture={'widths': [64, 128, 256]}, 
        ...                           train_config={'lr': 1e-3, 'batch_size': 32})
        >>> print(result.alpha)
        
        # With modality and modules:
        >>> profiler = TASProfiler()
        >>> profiler.set_modality('text', encoder='bert')
        >>> modules = [
        ...     ModuleRegistry.embedding(vocab_size=30000, embed_dim=512),
        ...     ModuleRegistry.conv1d(512, 768, kernel_size=3),
        ...     ModuleRegistry.attention(num_heads=12, head_dim=64),
        ... ]
        >>> result = profiler.profile_architecture(modules, text_data, train_config)
    """
    
    def __init__(self, config: Optional[TASConfig] = None):
        """Initialize TASProfiler.
        
        Args:
            config: TAS configuration (uses defaults if None)
        """
        self.config = config or TASConfig()
        
        # Initialize estimators
        self.manifold_estimator = ManifoldEstimator(k_max=20)
        self.smoothness_estimator = SmoothnessEstimator(k_neighbors=self.config.k_neighbors)
        self.architecture_analyzer = ArchitectureAnalyzer()
        self.temp_estimator = TemperatureEstimator()
        self.psi_computer = ThermalPhaseComputer(gamma_T=self.config.gamma_T)
        self.phi_computer = CoolingPhaseComputer()
        
        # Modality support (default: tabular)
        self.modality: BaseModality = TabularModality()
        
        # Module registry for convenience
        self.module_registry = ModuleRegistry()
        
        self._last_result: Optional[TASResult] = None
    
    def set_modality(self, modality: str, **kwargs) -> None:
        """Set the data modality.
        
        Args:
            modality: Modality type ('tabular', 'text', 'video', 'audio', 'image')
            **kwargs: Additional arguments for the modality constructor
                     For 'tabular': scale (bool)
                     For embedding modalities: encoder (str)
            
        Example:
            >>> profiler.set_modality('tabular', scale=True)
            >>> profiler.set_modality('text', encoder='bert')
            >>> profiler.set_modality('video', encoder='clip')
            >>> profiler.set_modality('audio', encoder='wav2vec')
        """
        if modality == 'tabular':
            self.modality = ModalityRegistry.create_tabular(**kwargs)
        elif modality in ['text', 'video', 'audio', 'image']:
            encoder = kwargs.pop('encoder', 'clip' if modality in ['video', 'image'] else 'bert')
            self.modality = ModalityRegistry.create(modality, encoder=encoder, **kwargs)
        else:
            raise ValueError(
                f"Unknown modality: {modality}. "
                f"Available: tabular, text, video, audio, image"
            )
    
    def set_modality_instance(self, modality: BaseModality) -> None:
        """Set a custom modality instance.
        
        Args:
            modality: BaseModality instance
        """
        self.modality = modality
        
    def profile(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        architecture: Dict[str, Any],
        train_config: Dict[str, Any],
    ) -> TASResult:
        """Run full TAS profiling pipeline.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            y: Target labels of shape (n_samples,)
            architecture: Architecture specification dict with keys like
                         'widths', 'types', 'depth', etc.
            train_config: Training configuration with keys like
                         'lr', 'batch_size', etc.
                         
        Returns:
            TASResult with all computed values
        """
        # Override config from train_config if provided
        eta_lr = train_config.get('lr', self.config.eta_lr)
        batch_size = train_config.get('batch_size', self.config.batch_size)
        
        # ========== PHASE 1: Data Geometric Profiling ==========
        # Extract features using modality
        X_features = self.modality.extract_features(X)
        
        # Estimate d_manifold
        if self.config.d_manifold is not None:
            d_manifold = self.config.d_manifold
        else:
            d_manifold = self.manifold_estimator.estimate_d(X_features)
        
        # Estimate s (Sobolev smoothness)
        if self.config.s_smoothness is not None:
            s = self.config.s_smoothness
        else:
            s = self.smoothness_estimator.estimate_s(y, X_features)
        
        # ========== PHASE 2: Loss Landscape Profiling ==========
        # Compute V_grad
        if self.config.V_grad is not None:
            V_grad = self.config.V_grad
        else:
            # Estimate from data variance
            V_grad = np.var(y) if len(y) > 0 else 1.0
        
        # Compute T_c
        T_c = self.temp_estimator.estimate_T_c(d_manifold, V_grad)
        
        # Compute k_alpha(L) - loss landscape ruggedness
        k_alpha = self._estimate_k_alpha(y)
        
        # ========== PHASE 3-4: Architecture Surrogate & Thermodynamic Probing ==========
        # Get architecture parameters
        layer_widths = architecture.get('widths', [64])
        layer_types = architecture.get('types', ['linear'] * len(layer_widths))
        
        # Compute η_l using heuristic (Phase 3)
        eta_ls_heuristic = self.architecture_analyzer.compute_heuristic_eta(
            layer_widths, layer_types, d_manifold=d_manifold
        )
        
        # Use product of eta_ls
        eta_product = product_log(eta_ls_heuristic)
        
        # ========== PHASE 5: Unified Scaling Prediction ==========
        # Effective temperature
        noise_var = self.config.noise_variance or np.var(y)
        T_eff = self.temp_estimator.estimate_T_eff(eta_lr, noise_var, batch_size)
        
        # Scaled temperature
        T_tilde = self.temp_estimator.compute_scaling_temperature(
            eta_ls_heuristic, T_eff, d_manifold
        )
        
        # Thermal phase factor Ψ_algo
        delta_loss = architecture.get('delta_loss', 0.0)
        psi_algo = self.psi_computer.compute_psi(T_tilde, T_c, delta_loss, T_eff)
        
        # Cooling phase factor φ(γ_cool)
        gamma_c = self.phi_computer.estimate_gamma_c(eta_lr, V_grad, d_manifold)
        phi_cool = self.phi_computer.compute_phi(self.config.gamma_cool, gamma_c)
        
        # ========== Compute final alpha ==========
        # α = k_α(L) · |log(∏η_l)| · (2s/d_manifold) · Ψ_algo · φ · ε
        log_eta_product = abs(safe_log(eta_product))
        alpha = (
            k_alpha *
            log_eta_product *
            (2.0 * s / d_manifold) *
            psi_algo *
            phi_cool *
            self.config.epsilon
        )
        
        # Create result
        result = TASResult(
            d_manifold=d_manifold,
            s_smoothness=s,
            V_grad=V_grad,
            T_c=T_c,
            k_alpha=k_alpha,
            eta_ls=eta_ls_heuristic,
            eta_product=eta_product,
            d_effs=[],  # Could be filled with exact-track analysis
            T_eff=T_eff,
            T_tilde=T_tilde,
            psi_algo=psi_algo,
            phi_cool=phi_cool,
            alpha=alpha,
        )
        
        self._last_result = result
        return result
    
    def _estimate_k_alpha(self, y: NDArray[np.floating]) -> float:
        """Estimate loss landscape ruggedness k_alpha.
        
        k_alpha characterizes how rugged the loss landscape is.
        Higher values indicate more complex/rugged landscapes.
        
        Args:
            y: Target values
            
        Returns:
            k_alpha estimate in range [0.1, 10]
        """
        if len(y) < 2:
            return 1.0
        
        # Estimate from target variance and distribution
        var_y = np.var(y)
        range_y = np.max(y) - np.min(y) + 1e-10
        
        # Normalize and scale
        k_alpha = 1.0 + var_y / range_y
        
        return float(np.clip(k_alpha, 0.1, 10.0))
    
    def predict_alpha(
        self,
        d: float,
        s: float,
        eta_ls: List[float],
        train_config: Dict[str, Any],
    ) -> float:
        """Predict scaling exponent α given computed values.
        
        Useful when you already have profiled quantities and want
        to predict alpha for different configurations.
        
        Args:
            d: Manifold dimension
            s: Smoothness
            eta_ls: List of η_l values
            train_config: Training configuration
            
        Returns:
            Predicted alpha
        """
        eta_lr = train_config.get('lr', self.config.eta_lr)
        batch_size = train_config.get('batch_size', self.config.batch_size)
        V_grad = self.config.V_grad or 1.0
        
        T_eff = self.temp_estimator.estimate_T_eff(eta_lr, 1.0, batch_size)
        T_c = self.temp_estimator.estimate_T_c(d, V_grad)
        T_tilde = self.temp_estimator.compute_scaling_temperature(eta_ls, T_eff, d)
        psi_algo = self.psi_computer.compute_psi(T_tilde, T_c, 0.0, T_eff)
        
        gamma_c = self.phi_computer.estimate_gamma_c(eta_lr, V_grad, d)
        phi_cool = self.phi_computer.compute_phi(self.config.gamma_cool, gamma_c)
        
        eta_product = product_log(eta_ls)
        log_eta_product = abs(safe_log(eta_product))
        
        k_alpha = 1.0  # Default
        alpha = (
            k_alpha *
            log_eta_product *
            (2.0 * s / d) *
            psi_algo *
            phi_cool *
            self.config.epsilon
        )
        
        return alpha
    
    def profile_architecture(
        self,
        architecture: Union[Dict[str, Any], List[BaseModule]],
        train_config: Dict[str, Any],
        X: Optional[NDArray[np.floating]] = None,
        y: Optional[NDArray[np.floating]] = None,
    ) -> TASResult:
        """Profile a specific architecture.
        
        Supports two modes:
        1. Architecture dict (backward compatible): {'widths': [...], 'types': [...]}
        2. List of BaseModule objects for precise module-level control
        
        Args:
            architecture: Architecture specification dict or list of modules
            train_config: Training configuration
            X: Input data (optional, required for modality-based extraction)
            y: Target data (optional)
            
        Returns:
            TASResult with predicted values
            
        Example:
            # Using architecture dict (legacy)
            >>> result = profiler.profile_architecture(
            ...     {'widths': [64, 128, 256], 'types': ['linear', 'linear', 'linear']},
            ...     {'lr': 1e-3}
            ... )
            
            # Using module list (new)
            >>> modules = [
            ...     ModuleRegistry.conv2d(3, 64, 3),
            ...     ModuleRegistry.attention(8, 64),
            ...     ModuleRegistry.residual(ModuleRegistry.conv2d(64, 64, 3)),
            ...     ModuleRegistry.pooling('max', 2),
            ... ]
            >>> result = profiler.profile_architecture(modules, {'lr': 1e-3})
        """
        # Check if architecture is a list of modules
        if isinstance(architecture, list) and len(architecture) > 0 and isinstance(architecture[0], BaseModule):
            return self._profile_with_modules(architecture, train_config, X, y)
        
        # Legacy dict-based profiling
        return self._profile_with_dict(architecture, train_config, X, y)
    
    def _profile_with_dict(
        self,
        architecture: Dict[str, Any],
        train_config: Dict[str, Any],
        X: Optional[NDArray[np.floating]],
        y: Optional[NDArray[np.floating]],
    ) -> TASResult:
        """Profile architecture using dict specification (legacy mode)."""
        d = self.config.d_manifold or 10.0
        s = self.config.s_smoothness or 1.0
        
        layer_widths = architecture.get('widths', [64])
        layer_types = architecture.get('types', ['linear'] * len(layer_widths))
        
        eta_ls = self.architecture_analyzer.compute_heuristic_eta(
            layer_widths, layer_types, d_manifold=d
        )
        
        eta_lr = train_config.get('lr', self.config.eta_lr)
        batch_size = train_config.get('batch_size', self.config.batch_size)
        V_grad = self.config.V_grad or 1.0
        
        T_eff = self.temp_estimator.estimate_T_eff(eta_lr, 1.0, batch_size)
        T_c = self.temp_estimator.estimate_T_c(d, V_grad)
        T_tilde = self.temp_estimator.compute_scaling_temperature(eta_ls, T_eff, d)
        psi_algo = self.psi_computer.compute_psi(T_tilde, T_c, 0.0, T_eff)
        
        gamma_c = self.phi_computer.estimate_gamma_c(eta_lr, V_grad, d)
        phi_cool = self.phi_computer.compute_phi(self.config.gamma_cool, gamma_c)
        
        eta_product = product_log(eta_ls)
        k_alpha = 1.0
        
        alpha = (
            k_alpha *
            abs(safe_log(eta_product)) *
            (2.0 * s / d) *
            psi_algo *
            phi_cool *
            self.config.epsilon
        )
        
        return TASResult(
            d_manifold=d,
            s_smoothness=s,
            V_grad=V_grad,
            T_c=T_c,
            k_alpha=k_alpha,
            eta_ls=eta_ls,
            eta_product=eta_product,
            d_effs=[],
            T_eff=T_eff,
            T_tilde=T_tilde,
            psi_algo=psi_algo,
            phi_cool=phi_cool,
            alpha=alpha,
        )
    
    def _profile_with_modules(
        self,
        modules: List[BaseModule],
        train_config: Dict[str, Any],
        X: Optional[NDArray[np.floating]],
        y: Optional[NDArray[np.floating]],
    ) -> TASResult:
        """Profile architecture using list of BaseModule objects.
        
        This method provides precise module-level control and uses the
        module's built-in η_l computation formulas.
        
        Args:
            modules: List of BaseModule objects
            train_config: Training configuration
            X: Input data (optional)
            y: Target data (optional)
            
        Returns:
            TASResult with predicted values
        """
        # Extract features using modality if X is provided
        if X is not None:
            X_features = self.modality.extract_features(X)
            d_manifold = self.config.d_manifold or self.manifold_estimator.estimate_d(X_features)
        else:
            d_manifold = self.config.d_manifold or 10.0
        
        s = self.config.s_smoothness or 1.0
        
        # Compute η_l for each module using module-specific formulas
        eta_ls = []
        d_effs = []
        d_current = d_manifold
        
        for module in modules:
            # Module computes η based on its type and previous dimension
            eta_l = module.compute_eta(d_current)
            eta_ls.append(eta_l)
            
            # Track D_eff for exact track
            if module.d_eff is not None:
                d_effs.append(module.d_eff)
            else:
                d_effs.append(d_current * eta_l)
            
            # Update current dimension
            d_current = eta_l * d_current
        
        eta_product = product_log(eta_ls)
        k_alpha = 1.0  # Default when no data
        
        eta_lr = train_config.get('lr', self.config.eta_lr)
        batch_size = train_config.get('batch_size', self.config.batch_size)
        V_grad = self.config.V_grad or 1.0
        
        T_eff = self.temp_estimator.estimate_T_eff(eta_lr, 1.0, batch_size)
        T_c = self.temp_estimator.estimate_T_c(d_manifold, V_grad)
        T_tilde = self.temp_estimator.compute_scaling_temperature(eta_ls, T_eff, d_manifold)
        psi_algo = self.psi_computer.compute_psi(T_tilde, T_c, 0.0, T_eff)
        
        gamma_c = self.phi_computer.estimate_gamma_c(eta_lr, V_grad, d_manifold)
        phi_cool = self.phi_computer.compute_phi(self.config.gamma_cool, gamma_c)
        
        alpha = (
            k_alpha *
            abs(safe_log(eta_product)) *
            (2.0 * s / d_manifold) *
            psi_algo *
            phi_cool *
            self.config.epsilon
        )
        
        return TASResult(
            d_manifold=d_manifold,
            s_smoothness=s,
            V_grad=V_grad,
            T_c=T_c,
            k_alpha=k_alpha,
            eta_ls=eta_ls,
            eta_product=eta_product,
            d_effs=d_effs,
            T_eff=T_eff,
            T_tilde=T_tilde,
            psi_algo=psi_algo,
            phi_cool=phi_cool,
            alpha=alpha,
        )
    
    def verify_optimality(
        self,
        eta_ls: List[float],
        T_eff: float,
        T_c: float,
        gamma_c: float,
    ) -> OptimalityResult:
        """Phase 6: Verify thermogeometric optimality conditions C1 and C2.
        
        C1: J_topo = |∑log η_l| ≤ ε_topo (topological isometry)
        C2: T̃_eff = T_eff · ε_coupling ≤ ξ_opt · T_c (thermal safety)
        
        Args:
            eta_ls: List of η_l values (layer efficiency factors)
            T_eff: Effective temperature
            T_c: Critical temperature
            gamma_c: Critical cooling rate (not directly used, for API consistency)
            
        Returns:
            OptimalityResult with feasibility status and diagnostics
        """
        # Compute J_topo = |∑log η_l|
        J_topo = abs(sum(safe_log(eta) for eta in eta_ls))
        
        # Compute ε_coupling
        d_manifold = self.config.d_manifold or 10.0
        epsilon_coupling = compute_epsilon_coupling(eta_ls, d_manifold)
        
        # Check C1
        c1_satisfied = check_c1_topological_isometry(J_topo, self.config.epsilon_topo)
        
        # Compute T_tilde_eff for C2
        T_tilde_eff = T_eff * epsilon_coupling
        
        # Check C2
        c2_satisfied = check_c2_thermal_safety(
            T_eff, epsilon_coupling, T_c, self.config.xi_opt
        )
        
        # Compute alpha for this architecture
        eta_product = product_log(eta_ls)
        log_eta_product = abs(safe_log(eta_product))
        s = self.config.s_smoothness or 1.0
        
        k_alpha = 1.0  # Default when not computed from data
        psi_computer = ThermalPhaseComputer(gamma_T=self.config.gamma_T)
        phi_computer = CoolingPhaseComputer()
        
        T_tilde = self.temp_estimator.compute_scaling_temperature(eta_ls, T_eff, d_manifold)
        psi_algo = psi_computer.compute_psi(T_tilde, T_c, 0.0, T_eff)
        phi_cool = phi_computer.compute_phi(self.config.gamma_cool, gamma_c)
        
        alpha = (
            k_alpha *
            log_eta_product *
            (2.0 * s / d_manifold) *
            psi_algo *
            phi_cool *
            self.config.epsilon
        )
        
        # Overall feasibility
        is_feasible = c1_satisfied and c2_satisfied
        
        return OptimalityResult(
            is_feasible=is_feasible,
            c1_satisfied=c1_satisfied,
            c2_satisfied=c2_satisfied,
            alpha=alpha,
            J_topo=J_topo,
            T_tilde_eff=T_tilde_eff,
            epsilon_coupling=epsilon_coupling,
        )
    
    def verify_and_profile(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        architecture: Dict[str, Any],
        train_config: Dict[str, Any],
    ) -> TASResult:
        """Run full TAS profiling with Phase 6 optimality verification.
        
        This combines profile() with verify_optimality() to produce
        a TASResult that includes optimality verification.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            y: Target labels of shape (n_samples,)
            architecture: Architecture specification dict
            train_config: Training configuration dict
            
        Returns:
            TASResult with optimality_result populated
        """
        result = self.profile(X, y, architecture, train_config)
        
        # Get gamma_c for verification
        gamma_c = self.phi_computer.estimate_gamma_c(
            train_config.get('lr', self.config.eta_lr),
            result.V_grad,
            result.d_manifold,
        )
        
        # Verify optimality
        optimality = self.verify_optimality(
            eta_ls=result.eta_ls,
            T_eff=result.T_eff,
            T_c=result.T_c,
            gamma_c=gamma_c,
        )
        
        # Update result with optimality
        result.optimality_result = optimality
        
        self._last_result = result
        return result
    
    @property
    def last_result(self) -> Optional[TASResult]:
        """Return the last TAS result."""
        return self._last_result
