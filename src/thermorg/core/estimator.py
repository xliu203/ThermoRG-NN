# SPDX-License-Identifier: Apache-2.0

"""Unified measurement interface for ThermoRG-NN experiments.

Provides a single interface for computing all relevant metrics:
- Jacobian-based metrics (SMC theory)
- Spectral metrics (eigenvalue distributions)
- Scaling law metrics (α, D_eff, T_eff)
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from typing import Optional, Literal, Any
from dataclasses import dataclass, field

from .smc import (
    compute_frobenius_norm,
    compute_spectral_norm,
    effective_dimension,
    compression_efficiency,
    spectral_momentum_operator,
    compute_smc_metrics,
)
from .jacobian import activation_jacobian, power_iteration


@dataclass
class MeasurementResult:
    """Container for measurement results.
    
    Attributes:
        frobenius_norm: Frobenius norm of Jacobian
        spectral_norm: Spectral norm (largest singular value)
        effective_dim: Effective dimension D_eff = ||J||_F² / ||J||_2²
        compression_eff: Compression efficiency η = D_eff / d_manifold
        momentum_op: Spectral momentum operator Π = J^T J
        singular_values: All singular values of Jacobian
        eigenvalues: Eigenvalues of momentum operator (if computed)
        trace: Trace of momentum operator
        timestamp: Computation timestamp
    """
    frobenius_norm: Tensor
    spectral_norm: Tensor
    effective_dim: Tensor
    compression_eff: Tensor
    momentum_op: Tensor
    singular_values: Tensor
    eigenvalues: Optional[Tensor] = None
    trace: Optional[Tensor] = None
    timestamp: float = field(default_factory=lambda: 0.0)
    
    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary with scalar values."""
        result = {
            "frobenius_norm": self.frobenius_norm.item(),
            "spectral_norm": self.spectral_norm.item(),
            "effective_dim": self.effective_dim.item(),
            "compression_eff": self.compression_eff.item(),
            "trace": self.trace.item() if self.trace is not None else None,
        }
        return result


class MeasurementEstimator:
    """Unified interface for computing network metrics.
    
    Provides a single entry point for all measurement computations
    needed in ThermoRG-NN experiments.
    
    Example:
        >>> estimator = MeasurementEstimator()
        >>> result = estimator.measure_jacobian(model, x, manifold_dim=8)
        >>> print(f"D_eff: {result.effective_dim.item():.3f}")
        >>> print(f"η: {result.compression_eff.item():.3f}")
    """
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        n_power_iter: int = 10,
        compute_eigenvalues: bool = False,
    ) -> None:
        """Initialize measurement estimator.
        
        Args:
            device: Device for computation
            n_power_iter: Power iteration steps for spectral norm
            compute_eigenvalues: Whether to compute eigenvalues of momentum op
        """
        self.device = device or torch.device("cpu")
        self.n_power_iter = n_power_iter
        self.compute_eigenvalues = compute_eigenvalues
    
    def measure_jacobian(
        self,
        module: nn.Module,
        x: Tensor,
        manifold_dim: float,
        layer_idx: Optional[int] = None,
    ) -> MeasurementResult:
        """Measure Jacobian-based metrics for a module.
        
        Args:
            module: Neural network module or single layer
            x: Input tensor (batch_size, input_dim)
            manifold_dim: Intrinsic manifold dimension
            layer_idx: If module is Sequential, measure specific layer's Jacobian
            
        Returns:
            MeasurementResult with all metrics
        """
        # Compute Jacobian based on module type
        if isinstance(module, nn.Sequential):
            if layer_idx is not None:
                jacobian = self._compute_layer_jacobian(module, x, layer_idx)
            else:
                jacobian = self._compute_full_jacobian(module, x)
        elif isinstance(module, nn.Linear):
            jacobian = module.weight.data
        else:
            jacobian = self._compute_activation_jacobian(module, x)
        
        jacobian = jacobian.to(self.device)
        
        # Compute SMC metrics
        fro_norm = compute_frobenius_norm(jacobian)
        spec_norm = compute_spectral_norm(jacobian, self.n_power_iter)
        eff_dim = effective_dimension(fro_norm, spec_norm)
        comp_eff = compression_efficiency(eff_dim, manifold_dim)
        momentum_op = spectral_momentum_operator(jacobian)
        singular_values = torch.linalg.svdvals(jacobian)
        
        # Optional eigenvalue computation
        eigenvalues = None
        if self.compute_eigenvalues:
            eigenvalues = torch.linalg.eigvals(momentum_op)
        
        # Compute trace
        trace = torch.trace(momentum_op)
        
        return MeasurementResult(
            frobenius_norm=fro_norm,
            spectral_norm=spec_norm,
            effective_dim=eff_dim,
            compression_eff=comp_eff,
            momentum_op=momentum_op,
            singular_values=singular_values,
            eigenvalues=eigenvalues,
            trace=trace,
        )
    
    def _compute_activation_jacobian(
        self,
        module: nn.Module,
        x: Tensor,
    ) -> Tensor:
        """Compute Jacobian of module output w.r.t. input.
        
        Args:
            module: Module to compute Jacobian for
            x: Input tensor
            
        Returns:
            Jacobian matrix
        """
        x = x.detach().requires_grad_(True)
        output = module(x)
        
        # For vector output, compute full Jacobian
        d_out, d_in = output.shape[-1], x.shape[-1]
        jacobian = torch.zeros(d_out, d_in, device=x.device, dtype=x.dtype)
        
        for i in range(min(d_out, 10)):  # Limit for efficiency
            grad = torch.zeros_like(output)
            grad[..., i] = 1.0
            (jac,) = torch.autograd.grad(outputs=output, inputs=x, grad_outputs=grad)
            jacobian[i] = jac.reshape(-1, d_in)
        
        return jacobian
    
    def _compute_layer_jacobian(
        self,
        sequential: nn.Sequential,
        x: Tensor,
        layer_idx: int,
    ) -> Tensor:
        """Compute Jacobian for a specific layer in Sequential.
        
        Args:
            sequential: Sequential module
            x: Input tensor
            layer_idx: Index of layer to measure
            
        Returns:
            Jacobian matrix for that layer
        """
        # Get output at specified layer
        h = x.detach().requires_grad_(True)
        
        for i, layer in enumerate(sequential):
            h = layer(h)
            if i == layer_idx:
                break
        
        # Backprop to get input gradient at this layer
        d_out, d_in = h.shape[-1], x.shape[-1]
        jacobian = torch.zeros(d_out, d_in, device=x.device, dtype=x.dtype)
        
        for i in range(min(d_out, 10)):
            grad = torch.zeros_like(h)
            grad[..., i] = 1.0
            (jac,) = torch.autograd.grad(outputs=h, inputs=x, grad_outputs=grad)
            jacobian[i] = jac.reshape(-1, d_in)
        
        return jacobian
    
    def _compute_full_jacobian(
        self,
        module: nn.Module,
        x: Tensor,
    ) -> Tensor:
        """Compute full network Jacobian.
        
        Args:
            module: Network module
            x: Input tensor
            
        Returns:
            Jacobian matrix
        """
        return self._compute_activation_jacobian(module, x)
    
    def measure_scaling_law(
        self,
        model: nn.Module,
        x: Tensor,
        manifold_dims: list[float],
    ) -> dict[str, list[float]]:
        """Measure scaling law relationship.
        
        Args:
            model: Neural network model
            x: Input tensor
            manifold_dims: List of manifold dimensions to sweep
            
        Returns:
            Dictionary mapping metric names to values across manifold_dims
        """
        results = {
            "manifold_dim": [],
            "effective_dim": [],
            "compression_eff": [],
            "spectral_norm": [],
        }
        
        for d_m in manifold_dims:
            measurement = self.measure_jacobian(model, x, manifold_dim=d_m)
            results["manifold_dim"].append(d_m)
            results["effective_dim"].append(measurement.effective_dim.item())
            results["compression_eff"].append(measurement.compression_eff.item())
            results["spectral_norm"].append(measurement.spectral_norm.item())
        
        return results
    
    def measure_thermal_transition(
        self,
        model: nn.Module,
        x: Tensor,
        temperature_range: list[float],
        manifold_dim: float,
    ) -> dict[str, list[float]]:
        """Measure thermal phase transition.
        
        Args:
            model: Neural network model (should have temperature parameter)
            x: Input tensor
            temperature_range: List of temperatures to sweep
            manifold_dim: Manifold dimension
            
        Returns:
            Dictionary with metrics across temperatures
        """
        results = {
            "temperature": [],
            "effective_dim": [],
            "compression_eff": [],
            "loss": [],
        }
        
        for T in temperature_range:
            # If model supports temperature, apply it
            if hasattr(model, "temperature"):
                model.temperature = T
            
            # Measure
            measurement = self.measure_jacobian(model, x, manifold_dim=manifold_dim)
            
            # Compute loss if model has forward
            with torch.no_grad():
                try:
                    output = model(x)
                    loss = torch.mean(output ** 2).item()
                except:
                    loss = float('nan')
            
            results["temperature"].append(T)
            results["effective_dim"].append(measurement.effective_dim.item())
            results["compression_eff"].append(measurement.compression_eff.item())
            results["loss"].append(loss)
        
        return results


def estimate_phase_transition(
    compression_effs: list[float],
    temperatures: list[float],
    threshold: float = 0.5,
) -> tuple[float, float]:
    """Estimate critical temperature from compression efficiency curve.
    
    Args:
        compression_effs: Compression efficiency values
        temperatures: Corresponding temperatures
        threshold: Threshold for critical point detection
        
    Returns:
        Tuple of (T_c estimate, confidence)
    """
    # Find point where derivative is maximum (sharp transition)
    deps = torch.diff(torch.tensor(compression_effs))
    dtemps = torch.diff(torch.tensor(temperatures))
    
    if len(deps) == 0:
        return temperatures[0], 0.0
    
    derivatives = deps / dtemps
    max_idx = torch.argmax(torch.abs(derivatives))
    
    T_c = temperatures[max_idx]
    
    # Confidence based on how sharp the transition is
    max_deriv = torch.abs(derivatives[max_idx]).item()
    confidence = min(max_deriv * 10, 1.0)
    
    return T_c, confidence
