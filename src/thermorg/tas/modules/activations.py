# SPDX-License-Identifier: Apache-2.0

"""Activation functions including Thermogeometric Activation (TGA)."""

import torch
import torch.nn as nn
from typing import Dict, Callable, Optional

# Try to import scipy for normal distribution
try:
    from scipy.stats import norm
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class TGAActivation(nn.Module):
    """Thermogeometric Activation (TGA).
    
    σ(z) = z · Φ(z / T_adiab)
    
    where Φ is the Gaussian CDF and T_adiab is the adiabatic temperature
    that depends on the manifold dimension.
    
    T_adiab ∝ 1 / d_manifold
    
    For T_adiab = 1, this reduces to a form similar to GELU.
    
    Attributes:
        t_adiab: Adiabatic temperature parameter
    """
    
    def __init__(self, t_adiab: float = 1.0):
        """Initialize TGA activation.
        
        Args:
            t_adiab: Adiabatic temperature (default 1.0)
        """
        super().__init__()
        self.t_adiab = t_adiab
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            z: Input tensor
            
        Returns:
            Activated tensor
        """
        # Gaussian CDF approximation using error function
        # Φ(z) ≈ 0.5 * (1 + erf(z / sqrt(2)))
        # σ(z) = z * Φ(z / T_adiab)
        sqrt2 = 1.4142135623730951
        phi = 0.5 * (1 + torch.erf(z / (self.t_adiab * sqrt2)))
        return z * phi
    
    def extra_repr(self) -> str:
        return f't_adiab={self.t_adiab}'


class GELU(nn.Module):
    """Gaussian Error Linear Unit (GELU).
    
    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    
    This is approximately equal to TGA with T_adiab = 1.0.
    
    Note:
        This is the approximate GELU used in transformers.
        The exact GELU uses torch.nn.functional.gelu.
    """
    
    def __init__(self):
        super().__init__()
        self.sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            z: Input tensor
            
        Returns:
            Activated tensor
        """
        return 0.5 * z * (1 + torch.tanh(self.sqrt_2_over_pi * (z + 0.044715 * z ** 3)))


class ExactGELU(nn.Module):
    """Exact GELU using the error function.
    
    GELU(x) = x * Φ(x) where Φ is the standard normal CDF.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass using exact GELU.
        
        Args:
            z: Input tensor
            
        Returns:
            Activated tensor
        """
        return torch.nn.functional.gelu(z)


class Swish(nn.Module):
    """Swish activation.
    
    σ(x) = x * sigmoid(β * x)
    
    where β is a learnable parameter (default 1.0).
    
    Note:
        When β = 1, this is identical to SiLU (Sigmoid Linear Unit).
    """
    
    def __init__(self, beta: float = 1.0):
        """Initialize Swish activation.
        
        Args:
            beta: Beta parameter (default 1.0)
        """
        super().__init__()
        self.beta = beta
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            z: Input tensor
            
        Returns:
            Activated tensor
        """
        return z * torch.sigmoid(self.beta * z)


class TGAWithLearnableBeta(nn.Module):
    """TGA with learnable beta parameter.
    
    σ(z) = z * Φ(z / (beta * T_adiab))
    
    where beta is a learnable parameter.
    """
    
    def __init__(self, t_adiab: float = 1.0, init_beta: float = 1.0):
        """Initialize.
        
        Args:
            t_adiab: Base adiabatic temperature
            init_beta: Initial beta value
        """
        super().__init__()
        self.t_adiab = t_adiab
        self.beta = nn.Parameter(torch.tensor(init_beta))
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            z: Input tensor
            
        Returns:
            Activated tensor
        """
        sqrt2 = 1.4142135623730951
        effective_t = self.t_adiab * self.beta
        phi = 0.5 * (1 + torch.erf(z / (effective_t * sqrt2)))
        return z * phi


# Registry of activation functions
ACTIVATION_REGISTRY: Dict[str, Callable[..., nn.Module]] = {
    'tga': TGAActivation,
    'gelu': GELU,
    'exact_gelu': ExactGELU,
    'swish': Swish,
    'relu': nn.ReLU,
    'silu': nn.SiLU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh,
    'leaky_relu': nn.LeakyReLU,
    'elu': nn.ELU,
    'selu': nn.SELU,
    'softplus': nn.Softplus,
    ' hardswish': nn.Hardswish,
}


def get_activation(name: str, **kwargs) -> nn.Module:
    """Get activation function by name.
    
    Args:
        name: Activation name (case-insensitive)
        **kwargs: Additional arguments for the activation constructor
        
    Returns:
        Activation module
        
    Raises:
        ValueError: If activation name is not recognized
    """
    name_lower = name.lower()
    
    if name_lower not in ACTIVATION_REGISTRY:
        available = list(ACTIVATION_REGISTRY.keys())
        raise ValueError(
            f"Unknown activation: {name}. Available: {available}"
        )
    
    return ACTIVATION_REGISTRY[name_lower](**kwargs)


def register_activation(name: str, cls: Callable[..., nn.Module]):
    """Register a custom activation function.
    
    Args:
        name: Activation name
        cls: Activation class (must be nn.Module subclass)
    """
    ACTIVATION_REGISTRY[name.lower()] = cls


def is_tga_compatible(activation_name: str) -> bool:
    """Check if an activation is TGA-compatible (has similar form).
    
    TGA-compatible activations have the form σ(z) = z * f(z)
    where f(z) is a sigmoid-like function.
    
    Args:
        activation_name: Name of activation
        
    Returns:
        True if TGA-compatible
    """
    compatible = {'tga', 'gelu', 'swish', 'silu', 'softplus'}
    return activation_name.lower() in compatible


class AdaptiveTGAActivation(nn.Module):
    """Adaptive TGA that adjusts T_adiab based on manifold dimension.
    
    T_adiab = base_t_adiab / d_manifold
    
    This ensures the activation adapts to the intrinsic data dimension.
    """
    
    def __init__(self, base_t_adiab: float = 10.0):
        """Initialize.
        
        Args:
            base_t_adiab: Base adiabatic temperature (will be divided by d_manifold)
        """
        super().__init__()
        self.base_t_adiab = base_t_adiab
    
    def forward(self, z: torch.Tensor, d_manifold: float) -> torch.Tensor:
        """Forward pass with manifold-aware temperature.
        
        Args:
            z: Input tensor
            d_manifold: Manifold dimension
            
        Returns:
            Activated tensor
        """
        t_adiab = self.base_t_adiab / d_manifold if d_manifold > 0 else self.base_t_adiab
        sqrt2 = 1.4142135623730951
        phi = 0.5 * (1 + torch.erf(z / (t_adiab * sqrt2)))
        return z * phi
