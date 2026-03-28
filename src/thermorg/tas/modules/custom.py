# SPDX-License-Identifier: Apache-2.0

"""Custom neural network module support with TGA activation."""

from typing import Optional, Dict, Any, Callable, TYPE_CHECKING
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad

from .base import BaseModule, ModuleConfig
from .activations import get_activation, TGAActivation

if TYPE_CHECKING:
    from ...utils.math import safe_divide


class CustomModule(BaseModule):
    """User-defined module with custom activation support.
    
    This class wraps a PyTorch module and provides thermogeometric
    analysis capabilities including Jacobian computation for exact-track
    efficiency calculation.
    
    The activation function can be any of the supported types including
    the Thermogeometric Activation (TGA).
    
    Attributes:
        module: The wrapped PyTorch module
        activation_fn: The activation function module
        _jacobian_cache: Cache for computed Jacobians
        
    Example:
        >>> import torch.nn as nn
        >>> linear = nn.Linear(512, 256)
        >>> custom = CustomModule(linear, activation='tga', t_adiab=0.5)
        >>> eta = custom.compute_eta(d_prev=512)
        >>> x = torch.randn(1, 512)
        >>> J = custom.compute_jacobian(x)
    """
    
    def __init__(
        self,
        module: 'nn.Module',
        activation: str = 'tga',
        t_adiab: float = 1.0,
        config: Optional[ModuleConfig] = None,
        safe_divide_fn: Optional[Callable] = None,
    ):
        """Initialize custom module.
        
        Args:
            module: PyTorch module to wrap
            activation: Activation function name ('tga', 'relu', 'gelu', etc.)
            t_adiab: Adiabatic temperature for TGA activation
            config: Optional module configuration
            safe_divide_fn: Optional safe divide function (for dependency injection)
        """
        super().__init__(config)
        self.module = module
        self.activation_name = activation
        self.t_adiab = t_adiab
        
        # Get activation function
        if activation.lower() == 'tga':
            self.activation_fn = TGAActivation(t_adiab=t_adiab)
        else:
            self.activation_fn = get_activation(activation, t_adiab=t_adiab)
        
        # Cache for jacobians
        self._jacobian_cache: Dict[int, torch.Tensor] = {}
        
        # Try to infer dimensions from module
        self._infer_dimensions()
        
        # Safe divide function (lazy import to avoid circular dependency)
        self._safe_divide = safe_divide_fn
    
    def _infer_dimensions(self):
        """Infer input/output dimensions from the wrapped module."""
        if isinstance(self.module, nn.Linear):
            self._input_dim = self.module.in_features
            self._output_dim = self.module.out_features
        elif isinstance(self.module, nn.Conv1d):
            self._input_dim = self.module.in_channels
            self._output_dim = self.module.out_channels
        elif isinstance(self.module, nn.Conv2d):
            self._input_dim = self.module.in_channels
            self._output_dim = self.module.out_channels
        elif isinstance(self.module, nn.LayerNorm):
            self._input_dim = self.module.normalized_shape[0]
            self._output_dim = self.module.normalized_shape[0]
        elif isinstance(self.module, nn.BatchNorm1d):
            self._input_dim = self.module.num_features
            self._output_dim = self.module.num_features
        elif isinstance(self.module, nn.BatchNorm2d):
            self._input_dim = self.module.num_features
            self._output_dim = self.module.num_features
        else:
            # Default inference from config
            self._input_dim = self.config.in_channels
            self._output_dim = self.config.out_channels
    
    def _get_safe_divide(self):
        """Get safe divide function (lazy import)."""
        if self._safe_divide is None:
            try:
                from ...utils.math import safe_divide
                self._safe_divide = safe_divide
            except ImportError:
                # Fallback implementation
                def safe_divide(a, b, default=1.0):
                    return a / b if b != 0 else default
                self._safe_divide = safe_divide
        return self._safe_divide
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through module and activation.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after module and activation
        """
        x = self.module(x)
        x = self.activation_fn(x)
        return x
    
    def compute_eta(self, d_prev: float) -> float:
        """Compute η_l using Jacobian analysis.
        
        For custom modules, we use the exact Jacobian when possible
        or fall back to a heuristic based on the module's structure.
        
        η_l = D_eff / d^(l-1)
        where D_eff = ||J||_F² / ||J||²
        
        Args:
            d_prev: Previous layer dimension d^(l-1)
            
        Returns:
            η_l value
        """
        safe_divide = self._get_safe_divide()
        
        # Try exact computation if we have a cache
        if self._d_eff is not None:
            return safe_divide(self._d_eff, d_prev)
        
        # Use heuristic based on dimension ratio
        out_dim = self._output_dim or d_prev
        in_dim = self._input_dim or d_prev
        
        # For information-preserving modules
        if isinstance(self.module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            self._eta = 1.0
        elif isinstance(self.module, nn.Dropout):
            self._eta = 1.0
        elif isinstance(self.module, nn.Linear):
            # Linear transformation with activation
            # Approximate: D_eff ≈ out_dim (activation brings non-linearity)
            self._d_eff = out_dim
            self._eta = safe_divide(out_dim, d_prev)
        else:
            # Conservative estimate
            eta = safe_divide(out_dim, d_prev)
            self._eta = min(eta, 1.0)
        
        return max(self._eta, 1e-6)
    
    def compute_jacobian(
        self,
        x: torch.Tensor,
        use_activation: bool = True,
    ) -> torch.Tensor:
        """Compute Jacobian for exact-track analysis.
        
        Uses autograd to compute the exact Jacobian of the module
        (with or without activation).
        
        For large inputs, consider using compute_jacobian_approximate().
        
        Args:
            x: Input tensor of shape (batch_size, in_dim)
            use_activation: Whether to include activation in Jacobian
            
        Returns:
            Jacobian tensor of shape (batch_size, out_dim, batch_size, in_dim)
            or flattened version if flatten=True
        """
        x = x.detach().requires_grad_(True)
        
        # Forward pass
        output = self.module(x)
        
        if use_activation:
            output = self.activation_fn(output)
        
        # Compute Jacobian for each output element
        batch_size = output.shape[0]
        out_dim = output.shape[1] if output.ndim > 1 else 1
        
        jacobians = []
        for i in range(min(batch_size * out_dim, 100)):  # Limit for large outputs
            b = i // out_dim if out_dim > 1 else 0
            j = i % out_dim if out_dim > 1 else 0
            
            if output.ndim > 1:
                grad_output = torch.zeros_like(output)
                grad_output[b, j] = 1.0
            else:
                grad_output = torch.zeros_like(output)
                grad_output[b] = 1.0
            
            grad_input = grad(
                output,
                x,
                grad_output,
                retain_graph=True,
            )[0]
            
            jacobians.append(grad_input[b] if batch_size > 1 else grad_input)
        
        if jacobians:
            J = torch.stack(jacobians)  # (out_dim, in_dim) or (out_dim, batch, in_dim)
            if J.ndim == 3:
                J = J.mean(dim=0)  # Average over batch if needed
        else:
            J = torch.eye(x.shape[1], device=x.device)
        
        return J
    
    def compute_jacobian_approximate(
        self,
        x: torch.Tensor,
        method: str = 'diagonal',
    ) -> torch.Tensor:
        """Compute approximate Jacobian for efficiency.
        
        When exact Jacobian is too expensive, use approximations.
        
        Args:
            x: Input tensor
            method: Approximation method ('diagonal', 'frobenius', 'spectral')
            
        Returns:
            Approximate Jacobian
        """
        if method == 'diagonal':
            # Return identity-like matrix (conservative)
            dim = self._output_dim or x.shape[-1]
            return torch.eye(dim, device=x.device)
        elif method == 'frobenius':
            # Estimate using weight norms
            if hasattr(self.module, 'weight'):
                weight_norm = self.module.weight.data.norm()
                return torch.eye(self._output_dim or 1) * weight_norm
            return torch.eye(x.shape[-1])
        else:
            return torch.eye(x.shape[-1])
    
    def compute_d_eff(self, x: torch.Tensor) -> float:
        """Compute effective dimension D_eff from Jacobian.
        
        D_eff = ||J||_F² / ||J||²
        
        Args:
            x: Input tensor
            
        Returns:
            Effective dimension D_eff
        """
        J = self.compute_jacobian(x)
        
        # Frobenius norm squared
        frob_sq = torch.sum(J ** 2).item()
        
        # Spectral norm squared (max eigenvalue)
        try:
            eigenvalues = torch.linalg.eigvalsh(J)
            spectral_sq = eigenvalues[-1].item() ** 2 if len(eigenvalues) > 0 else frob_sq
        except:
            spectral_sq = frob_sq
        
        safe_divide = self._get_safe_divide()
        d_eff = safe_divide(frob_sq, spectral_sq) if spectral_sq > 0 else 1.0
        
        self._d_eff = d_eff
        return d_eff
    
    def reset_cache(self):
        """Reset cached values."""
        self._jacobian_cache.clear()
        self._eta = None
        self._d_eff = None
    
    @property
    def in_features(self) -> int:
        """Return input features."""
        return self._input_dim or 0
    
    @property
    def out_features(self) -> int:
        """Return output features."""
        return self._output_dim or 0
    
    def __repr__(self) -> str:
        return f"CustomModule({self.module.__class__.__name__}, activation={self.activation_name})"