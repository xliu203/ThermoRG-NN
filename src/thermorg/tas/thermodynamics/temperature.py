# SPDX-License-Identifier: Apache-2.0

"""Temperature estimation for thermodynamic factors.

Phase 5: Unified Scaling Prediction - T_eff, T_c estimation
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Dict, Any

from ...utils.math import safe_divide, product_log


class TemperatureEstimator:
    """Estimates effective temperature and critical temperature.
    
    Phase 5:
        T_eff = η_lr · σ² / B
        T_c = c_T · d_manifold / V_∇
    
    where:
        η_lr = learning rate
        σ² = noise variance
        B = batch size
        d_manifold = intrinsic dimension
        V_∇ = gradient variance E[||∇_θ L(x)||²]
        c_T = calibration constant (default 1.0)
    """
    
    def __init__(self, c_T: float = 1.0):
        """Initialize TemperatureEstimator.
        
        Args:
            c_T: Calibration constant for T_c (default 1.0)
        """
        self.c_T = c_T
        self._T_eff: Optional[float] = None
        self._T_c: Optional[float] = None
        
    def estimate_T_eff(
        self,
        eta_lr: float,
        noise_variance: float,
        batch_size: int,
    ) -> float:
        """Estimate effective temperature T_eff.
        
        T_eff = η_lr · σ² / B
        
        Args:
            eta_lr: Learning rate
            noise_variance: Noise variance σ²
            batch_size: Batch size B
            
        Returns:
            Effective temperature T_eff
        """
        if batch_size <= 0:
            batch_size = 1
            
        T_eff = safe_divide(eta_lr * noise_variance, batch_size)
        
        self._T_eff = T_eff
        return T_eff
    
    def estimate_T_c(
        self,
        d_manifold: float,
        V_grad: float,
    ) -> float:
        """Estimate critical temperature T_c.
        
        T_c = c_T · d_manifold / V_∇
        
        Args:
            d_manifold: Intrinsic manifold dimension
            V_grad: Gradient variance E[||∇_θ L(x)||²]
            
        Returns:
            Critical temperature T_c
        """
        if V_grad <= 0:
            V_grad = 1.0
            
        T_c = self.c_T * safe_divide(d_manifold, V_grad)
        
        self._T_c = T_c
        return T_c
    
    def estimate_from_gradient_stats(
        self,
        gradients: NDArray[np.floating],
        d_manifold: float,
        eta_lr: float = 1e-3,
        batch_size: Optional[int] = None,
    ) -> Dict[str, float]:
        """Estimate temperatures from gradient statistics.
        
        Args:
            gradients: Gradient vectors, shape (n_samples, n_params)
            d_manifold: Manifold dimension
            eta_lr: Learning rate
            batch_size: Batch size (defaults to n_samples)
            
        Returns:
            Dictionary with T_eff, T_c, and V_grad estimates
        """
        batch_size = batch_size or gradients.shape[0]
        
        # Estimate V_grad = E[||∇_θ L(x)||²]
        grad_norms_sq = np.sum(gradients ** 2, axis=1)
        V_grad = np.mean(grad_norms_sq)
        
        # Estimate noise variance from gradient fluctuations
        noise_variance = np.var(grad_norms_sq)
        
        T_eff = self.estimate_T_eff(eta_lr, noise_variance, batch_size)
        T_c = self.estimate_T_c(d_manifold, V_grad)
        
        return {
            'T_eff': T_eff,
            'T_c': T_c,
            'V_grad': V_grad,
            'noise_variance': noise_variance,
        }
    
    def compute_scaling_temperature(
        self,
        eta_ls: list,
        T_eff: float,
        d_manifold: float,
    ) -> float:
        """Compute scaled effective temperature.
        
        T̃_eff = T_eff / (∏η_l)^{2/d_manifold}
        
        Args:
            eta_ls: List of η_l values
            T_eff: Effective temperature
            d_manifold: Manifold dimension
            
        Returns:
            Scaled temperature T̃_eff
        """
        eta_product = product_log(eta_ls)
        
        if eta_product <= 0 or d_manifold <= 0:
            return T_eff
            
        scaling_factor = np.power(eta_product, 2.0 / d_manifold)
        T_scaled = safe_divide(T_eff, scaling_factor)
        
        return T_scaled
    
    @property
    def T_eff(self) -> Optional[float]:
        """Return last estimated T_eff."""
        return self._T_eff
    
    @property
    def T_c(self) -> Optional[float]:
        """Return last estimated T_c."""
        return self._T_c
