# SPDX-License-Identifier: Apache-2.0

"""Thermal phase factor Ψ_algo computation.

Phase 5: Unified Scaling Prediction
    Ψ_algo = T̃ · (1 - T̃/T_c)^γ_T · exp(-Δ_loss/T_eff) · I(T̃ < T_c)
"""

import numpy as np
from typing import Optional

from ...utils.math import safe_divide, safe_log


class ThermalPhaseComputer:
    """Computes the thermal phase factor Ψ_algo.
    
    Ψ_algo characterizes the thermal regime of the optimization:
    - When T̃ < T_c: system is in "cold" regime, good convergence expected
    - When T̃ > T_c: system is in "hot" regime, convergence may struggle
    
    The formula combines:
    1. Scaled temperature T̃
    2. Proximity to criticality (1 - T̃/T_c)^γ_T
    3. Loss barrier factor exp(-Δ_loss/T_eff)
    4. Indicator I(T̃ < T_c)
    """
    
    def __init__(self, gamma_T: float = 1.0):
        """Initialize ThermalPhaseComputer.
        
        Args:
            gamma_T: Critical exponent for thermal phase (default 1.0)
        """
        self.gamma_T = gamma_T
        self._psi: Optional[float] = None
        
    def compute_psi(
        self,
        T_tilde: float,
        T_c: float,
        delta_loss: float = 0.0,
        T_eff: Optional[float] = None,
    ) -> float:
        """Compute thermal phase factor Ψ_algo.
        
        Args:
            T_tilde: Scaled effective temperature
            T_c: Critical temperature
            delta_loss: Loss barrier Δ_loss
            T_eff: Effective temperature (used in exponent)
            
        Returns:
            Ψ_algo value
        """
        T_eff = T_eff or 1.0
        
        # Indicator function: I(T̃ < T_c)
        indicator = 1.0 if T_tilde < T_c else 0.0
        
        # Base temperature factor
        psi = T_tilde
        
        # Critical proximity factor: (1 - T̃/T_c)^γ_T
        if T_c > 0 and T_tilde < T_c:
            proximity = 1.0 - (T_tilde / T_c)
            proximity_factor = np.power(proximity, self.gamma_T)
            psi *= proximity_factor
        elif T_tilde >= T_c:
            psi = 0.0
            return psi
        
        # Loss barrier factor: exp(-Δ_loss/T_eff)
        if delta_loss > 0 and T_eff > 0:
            barrier_factor = np.exp(-safe_divide(delta_loss, T_eff))
            psi *= barrier_factor
        
        psi *= indicator
        
        self._psi = psi
        return psi
    
    def compute_psi_vectorized(
        self,
        T_tildes: np.ndarray,
        T_c: float,
        delta_losses: Optional[np.ndarray] = None,
        T_eff: Optional[float] = None,
    ) -> np.ndarray:
        """Compute Ψ_algo for multiple samples.
        
        Args:
            T_tildes: Array of scaled temperatures
            T_c: Critical temperature
            delta_losses: Array of loss barriers (optional)
            T_eff: Effective temperature
            
        Returns:
            Array of Ψ_algo values
        """
        T_eff = T_eff or 1.0
        n = len(T_tildes)
        
        if delta_losses is None:
            delta_losses = np.zeros(n)
        
        psis = np.zeros(n)
        
        for i in range(n):
            psis[i] = self.compute_psi(
                T_tildes[i], T_c, delta_losses[i], T_eff
            )
        
        return psis
    
    def compute_psi_derivative(
        self,
        T_tilde: float,
        T_c: float,
        T_eff: float = 1.0,
        delta_loss: float = 0.0,
    ) -> float:
        """Compute derivative of Ψ_algo with respect to T_tilde.
        
        Useful for gradient-based optimization.
        
        Args:
            T_tilde: Scaled temperature
            T_c: Critical temperature
            T_eff: Effective temperature
            delta_loss: Loss barrier
            
        Returns:
            dΨ/dT_tilde
        """
        if T_tilde >= T_c:
            return 0.0
        
        # d/dT_tilde [T_tilde * (1 - T_tilde/T_c)^γ_T]
        proximity = 1.0 - (T_tilde / T_c)
        
        d_proximity_dt = -1.0 / T_c
        
        d_psi_dt = (
            proximity ** self.gamma_T + 
            T_tilde * self.gamma_T * (proximity ** (self.gamma_T - 1)) * d_proximity_dt
        )
        
        # Include loss barrier factor derivative
        if delta_loss > 0 and T_eff > 0:
            barrier = np.exp(-delta_loss / T_eff)
            d_psi_dt *= barrier
        
        return d_psi_dt
    
    def find_critical_point(
        self,
        eta_ls: list,
        T_eff: float,
        d_manifold: float,
        V_grad: float = 1.0,
    ) -> float:
        """Find the learning rate at which T_tilde = T_c.
        
        This is the critical learning rate where the system transitions
        from convergent to non-convergent behavior.
        
        Args:
            eta_ls: List of η_l values
            T_eff: Effective temperature
            d_manifold: Manifold dimension
            V_grad: Gradient variance
            
        Returns:
            Critical learning rate η_c
        """
        # At criticality: T_tilde = T_c
        # T_tilde = (η_lr * σ² / B) / (∏η_l)^{2/d_manifold}
        # T_c = d_manifold / V_grad
        
        # Solving for η_lr:
        # η_lr * σ² / B = T_c * (∏η_l)^{2/d_manifold}
        
        eta_product = np.prod(eta_ls)
        scaling = np.power(eta_product, 2.0 / d_manifold) if eta_product > 0 else 1.0
        
        T_c = safe_divide(d_manifold, V_grad)
        sigma_sq = T_eff  # Assuming T_eff was computed with unit batch size
        
        # This is approximate - actual critical point depends on batch size
        eta_c = T_c * scaling / sigma_sq if sigma_sq > 0 else 1.0
        
        return eta_c
    
    @property
    def psi(self) -> Optional[float]:
        """Return last computed Ψ_algo value."""
        return self._psi
