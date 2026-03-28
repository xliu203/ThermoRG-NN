# SPDX-License-Identifier: Apache-2.0

"""Cooling phase factor φ(γ_cool) computation.

Phase 5: Unified Scaling Prediction
    φ(γ_cool) = exp(-γ/γ_c) / (1 + γ/γ_c)
"""

import numpy as np
from typing import Optional

from ...utils.math import safe_divide


class CoolingPhaseComputer:
    """Computes the cooling phase factor φ(γ_cool).
    
    φ(γ_cool) characterizes the cooling schedule effectiveness:
        φ(γ) = exp(-γ/γ_c) / (1 + γ/γ_c)
    
    where γ is the current cooling rate and γ_c is the critical cooling rate.
    
    Properties:
    - φ(0) = 1 (no cooling, maximum phase factor)
    - φ(γ_c) = exp(-1) / 2 ≈ 0.184
    - φ(∞) = 0 (full cooling)
    """
    
    def __init__(self, c_gamma: float = 1.0):
        """Initialize CoolingPhaseComputer.
        
        Args:
            c_gamma: Calibration constant for γ_c (default 1.0)
        """
        self.c_gamma = c_gamma
        self._gamma_c: Optional[float] = None
        self._phi: Optional[float] = None
        
    def compute_phi(
        self,
        gamma: float,
        gamma_c: Optional[float] = None,
    ) -> float:
        """Compute cooling phase factor φ(γ).
        
        Args:
            gamma: Current cooling rate
            gamma_c: Critical cooling rate (computed from η_lr, V_∇, d_manifold)
            
        Returns:
            φ(γ) value in range [0, 1]
        """
        if gamma_c is None:
            gamma_c = self._gamma_c or 1.0
            
        if gamma_c <= 0:
            gamma_c = 1.0
            
        ratio = safe_divide(gamma, gamma_c)
        
        # φ(γ) = exp(-γ/γ_c) / (1 + γ/γ_c)
        numerator = np.exp(-ratio)
        denominator = 1.0 + ratio
        
        phi = safe_divide(numerator, denominator)
        
        self._phi = phi
        return phi
    
    def compute_phi_derivative(
        self,
        gamma: float,
        gamma_c: Optional[float] = None,
    ) -> float:
        """Compute derivative of φ with respect to γ.
        
        dφ/dγ = [-exp(-γ/γ_c) * (1 + γ/γ_c) - exp(-γ/γ_c) * (1/γ_c)] / (1 + γ/γ_c)²
               = -exp(-γ/γ_c) * (1 + γ/γ_c + 1/γ_c) / (1 + γ/γ_c)²
        
        Args:
            gamma: Current cooling rate
            gamma_c: Critical cooling rate
            
        Returns:
            dφ/dγ
        """
        if gamma_c is None:
            gamma_c = self._gamma_c or 1.0
            
        if gamma_c <= 0:
            return 0.0
            
        ratio = safe_divide(gamma, gamma_c)
        exp_neg = np.exp(-ratio)
        denom = (1.0 + ratio) ** 2
        
        if denom < 1e-10:
            return 0.0
        
        d_phi = -exp_neg * (1 + ratio + 1.0/gamma_c) / denom
        
        return d_phi
    
    def estimate_gamma_c(
        self,
        eta_lr: float,
        V_grad: float,
        d_manifold: float,
    ) -> float:
        """Estimate critical cooling rate γ_c.
        
        γ_c = c_γ · (η_lr · V_∇) / d_manifold
        
        Args:
            eta_lr: Learning rate
            V_grad: Gradient variance E[||∇_θ L(x)||²]
            d_manifold: Manifold dimension
            
        Returns:
            Estimated γ_c
        """
        if d_manifold <= 0:
            d_manifold = 1.0
            
        gamma_c = self.c_gamma * safe_divide(eta_lr * V_grad, d_manifold)
        
        self._gamma_c = gamma_c
        return gamma_c
    
    def compute_phi_schedule(
        self,
        gamma_start: float,
        gamma_end: float,
        n_steps: int,
        gamma_c: Optional[float] = None,
    ) -> np.ndarray:
        """Compute cooling phase factor over a schedule.
        
        Args:
            gamma_start: Starting cooling rate
            gamma_end: Ending cooling rate
            n_steps: Number of schedule steps
            gamma_c: Critical cooling rate
            
        Returns:
            Array of φ values over the schedule
        """
        gammas = np.linspace(gamma_start, gamma_end, n_steps)
        phis = np.array([self.compute_phi(g, gamma_c) for g in gammas])
        
        return phis
    
    def find_optimal_gamma(
        self,
        gamma_c: Optional[float] = None,
        alpha_target: float = 0.5,
    ) -> float:
        """Find the cooling rate that achieves a target φ value.
        
        Solves φ(γ) = alpha_target for γ.
        
        Args:
            gamma_c: Critical cooling rate
            alpha_target: Target φ value (in (0, 1))
            
        Returns:
            γ that gives φ(γ) = alpha_target
        """
        gamma_c = gamma_c or self._gamma_c or 1.0
        
        if alpha_target <= 0 or alpha_target >= 1:
            raise ValueError("alpha_target must be in (0, 1)")
        
        # φ(γ) = exp(-γ/γ_c) / (1 + γ/γ_c) = alpha_target
        # This requires numerical solution
        
        from scipy.optimize import brentq
        
        def objective(gamma):
            return self.compute_phi(gamma, gamma_c) - alpha_target
        
        # Search in reasonable range
        gamma_opt = brentq(objective, 0.01, 10.0 * gamma_c)
        
        return gamma_opt
    
    @property
    def gamma_c(self) -> Optional[float]:
        """Return last estimated γ_c value."""
        return self._gamma_c
    
    @property
    def phi(self) -> Optional[float]:
        """Return last computed φ value."""
        return self._phi
