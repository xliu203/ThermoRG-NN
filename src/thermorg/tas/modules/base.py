# SPDX-License-Identifier: Apache-2.0

"""Base module classes for neural network architecture components."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    import torch
    import torch.nn as nn


@dataclass
class ModuleConfig:
    """Configuration for a neural network module.
    
    Attributes:
        module_type: Type identifier (conv2d, attention, residual, pooling, etc.)
        in_channels: Input channel dimension
        out_channels: Output channel dimension
        kernel_size: Kernel size (for conv/pooling)
        stride: Stride (for conv/pooling)
        padding: Padding (for conv)
        num_heads: Number of attention heads
        head_dim: Dimension per attention head
        dropout: Dropout rate
        activation: Activation function name
        t_adiab: Adiabatic temperature for TGA activation
        seq_len: Sequence length (for attention)
        pool_type: Pooling type ('max', 'avg')
        inner: Inner module (for residual)
    """
    module_type: str = 'linear'
    in_channels: Optional[int] = None
    out_channels: Optional[int] = None
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1
    num_heads: int = 8
    head_dim: int = 64
    dropout: float = 0.1
    activation: str = 'relu'
    t_adiab: float = 1.0
    seq_len: Optional[int] = None
    pool_type: str = 'max'
    inner: Optional['BaseModule'] = None
    # Additional metadata
    name: str = ''
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ModuleConfig':
        """Create from dictionary."""
        return cls(**d)


class BaseModule(ABC):
    """Abstract base class for architecture modules.
    
    A module represents a layer or block in a neural network architecture.
    Each module can compute its layer efficiency η_l based on the
    thermogeometric theory.
    
    Attributes:
        config: Module configuration
        _eta: Cached η_l value
        _d_eff: Cached D_eff value
    """
    
    def __init__(self, config: Optional[ModuleConfig] = None):
        """Initialize base module.
        
        Args:
            config: Module configuration
        """
        self.config = config or ModuleConfig()
        self._eta: Optional[float] = None
        self._d_eff: Optional[float] = None
        self._input_dim: Optional[int] = None
        self._output_dim: Optional[int] = None
    
    @abstractmethod
    def compute_eta(self, d_prev: float) -> float:
        """Compute η_l for this module given previous dimension.
        
        The layer efficiency η_l = D_eff / d^(l-1)
        where D_eff = ||J||_F² / ||J||² is the effective dimension.
        
        Args:
            d_prev: Previous layer dimension d^(l-1)
            
        Returns:
            η_l value
        """
        pass
    
    @abstractmethod
    def compute_jacobian(self, x: 'torch.Tensor') -> 'torch.Tensor':
        """Compute Jacobian for exact-track analysis.
        
        Args:
            x: Input tensor
            
        Returns:
            Jacobian tensor
        """
        pass
    
    @property
    def activation(self) -> 'nn.Module':
        """Return activation function module.
        
        Returns:
            Activation module
        """
        from .activations import get_activation
        return get_activation(self.config.activation, t_adiab=self.config.t_adiab)
    
    @property
    def eta(self) -> Optional[float]:
        """Return cached η_l value."""
        return self._eta
    
    @property
    def d_eff(self) -> Optional[float]:
        """Return cached D_eff value."""
        return self._d_eff
    
    @property
    def input_dim(self) -> Optional[int]:
        """Return input dimension."""
        return self._input_dim or self.config.in_channels
    
    @property
    def output_dim(self) -> Optional[int]:
        """Return output dimension."""
        return self._output_dim or self.config.out_channels
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.config.module_type})"
