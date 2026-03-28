# SPDX-License-Identifier: Apache-2.0

"""Preset neural network modules with known efficiency formulas."""

from typing import Optional, List, Dict, Any, TYPE_CHECKING
import math

import numpy as np

from .base import BaseModule, ModuleConfig
from .activations import get_activation

if TYPE_CHECKING:
    import torch
    import torch.nn as nn


class PresetModule(BaseModule):
    """Preset module with known efficiency formulas.
    
    Preset modules have well-characterized η_l formulas based on
    their structural properties (kernel size, heads, etc.).
    
    Supported types:
        - conv1d, conv2d: Convolutional layers
        - attention: Multi-head self-attention
        - residual: Residual connection wrapper
        - pooling: Max/Avg pooling
        - linear: Fully connected layers
        - embedding: Token embedding
        - layernorm, batchnorm: Normalization layers
        - dropout: Regularization
    
    Example:
        >>> module = PresetModule('conv2d', {
        ...     'in_channels': 3,
        ...     'out_channels': 64,
        ...     'kernel_size': 3,
        ...     'stride': 1,
        ...     'padding': 1,
        ...     'activation': 'relu'
        ... })
        >>> eta = module.compute_eta(d_prev=100)
    """
    
    # Known embedding dimensions for common encoders
    ENCODER_DIMS = {
        'bert': 768,
        'clip': 512,
        'wav2vec': 768,
        'vit': 768,
        'gpt2': 768,
        't5': 512,
    }
    
    def __init__(self, module_type: str, config: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize preset module.
        
        Args:
            module_type: Type of preset module
            config: Configuration dictionary
            **kwargs: Additional config overrides
        """
        module_config = ModuleConfig(module_type=module_type)
        if config:
            for key, value in config.items():
                if hasattr(module_config, key):
                    setattr(module_config, key, value)
        
        # Apply kwargs overrides
        for key, value in kwargs.items():
            if hasattr(module_config, key):
                setattr(module_config, key, value)
        
        super().__init__(module_config)
        
        self.module_type = module_type.lower()
    
    def compute_eta(self, d_prev: float) -> float:
        """Compute η_l using preset formula.
        
        Args:
            d_prev: Previous layer dimension d^(l-1)
            
        Returns:
            η_l value
        """
        return self._compute_eta_heuristic(d_prev)
    
    def _compute_eta_heuristic(self, d_prev: float) -> float:
        """Compute η_l using heuristic formula based on module type.
        
        Formulas:
            - conv1d/conv2d: η ≈ (k × in_ch) / d_prev (capped at 1.0)
            - attention: η ≈ (num_heads × head_dim) / d_prev (capped at 1.0)
            - residual: η = 1.0 (perfect preservation)
            - pooling: η ≈ stride^(-2) (compression)
            - linear: η ≈ min(out_dim / d_prev, 1.0)
            - embedding: η = 1.0
            - layernorm/batchnorm: η = 1.0
            - dropout: η = 1.0
        """
        config = self.config
        
        if self.module_type in ['conv1d', 'conv2d']:
            kernel_prod = config.kernel_size ** 2 if config.kernel_size > 1 else 1
            in_channels = config.in_channels or d_prev
            out_channels = config.out_channels or in_channels
            
            # For conv: effective width = out_channels * kernel_prod
            effective_width = out_channels * kernel_prod
            eta = min(1.0, effective_width / d_prev)
            
        elif self.module_type == 'attention':
            num_heads = config.num_heads
            head_dim = config.head_dim
            effective_width = num_heads * head_dim
            eta = min(1.0, effective_width / d_prev)
            
        elif self.module_type == 'residual':
            # Perfect preservation through shortcut
            eta = 1.0
            
        elif self.module_type == 'pooling':
            # Compression factor
            stride = config.stride
            eta = stride ** -2
            
        elif self.module_type == 'linear':
            out_dim = config.out_channels or d_prev
            in_dim = config.in_channels or d_prev
            # Information bottleneck factor
            eta = min(1.0, out_dim / in_dim)
            
        elif self.module_type == 'embedding':
            # Embedding doesn't change effective dimension directly
            eta = 1.0
            
        elif self.module_type in ['layernorm', 'batchnorm', 'ln', 'bn']:
            # Normalization preserves dimension
            eta = 1.0
            
        elif self.module_type == 'dropout':
            # Dropout doesn't change dimension during inference
            eta = 1.0
            
        elif self.module_type == 'flatten':
            # Flatten reduces dimension to 1 in spatial sense
            eta = 1.0
            
        elif self.module_type == 'reshape':
            eta = 1.0
            
        else:
            # Default: conservative estimate
            eta = min(1.0, (config.out_channels or d_prev) / d_prev)
        
        # Ensure minimum for numerical stability
        self._eta = max(eta, 1e-6)
        return self._eta
    
    def compute_jacobian(self, x: 'torch.Tensor') -> 'torch.Tensor':
        """Compute Jacobian for exact-track analysis.
        
        For preset modules, we use module-specific approximations.
        
        Args:
            x: Input tensor
            
        Returns:
            Jacobian tensor (diagonal approximation)
        """
        import torch
        
        # Get output dimension
        out_dim = self.config.out_channels or x.shape[-1]
        
        # Create identity-like Jacobian (diagonal)
        # This is an approximation - real Jacobian would require autograd
        jacobian = torch.eye(min(x.numel(), out_dim), device=x.device)
        
        # Pad or trim to match
        if jacobian.shape[0] < x.numel():
            padding = torch.zeros(
                x.numel() - jacobian.shape[0],
                jacobian.shape[1],
                device=jacobian.device
            )
            jacobian = torch.cat([jacobian, padding], dim=0)
        
        return jacobian
    
    def __repr__(self) -> str:
        config_str = ', '.join(f'{k}={v}' for k, v in self.config.to_dict().items() if v is not None)
        return f"PresetModule({self.module_type}, {config_str})"


class ModuleRegistry:
    """Registry for creating preset modules with convenience methods.
    
    Provides factory methods for common module types.
    
    Example:
        >>> conv = ModuleRegistry.conv2d(3, 64, kernel_size=3, stride=1)
        >>> attn = ModuleRegistry.attention(num_heads=8, head_dim=64)
        >>> residual = ModuleRegistry.residual(conv)
        >>> pool = ModuleRegistry.pooling('max', kernel_size=2)
    """
    
    @staticmethod
    def conv2d(
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        activation: str = 'relu',
    ) -> PresetModule:
        """Create a 2D convolutional module.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride
            padding: Padding size
            activation: Activation function name
            
        Returns:
            PresetModule for conv2d
        """
        return PresetModule('conv2d', {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'activation': activation,
        })
    
    @staticmethod
    def conv1d(
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        activation: str = 'relu',
    ) -> PresetModule:
        """Create a 1D convolutional module.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride
            padding: Padding size
            activation: Activation function name
            
        Returns:
            PresetModule for conv1d
        """
        return PresetModule('conv1d', {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'activation': activation,
        })
    
    @staticmethod
    def attention(
        num_heads: int,
        head_dim: int = 64,
        seq_len: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = 'relu',
    ) -> PresetModule:
        """Create a multi-head self-attention module.
        
        Args:
            num_heads: Number of attention heads
            head_dim: Dimension per head
            seq_len: Sequence length (optional)
            dropout: Dropout rate
            activation: Activation function name
            
        Returns:
            PresetModule for attention
        """
        return PresetModule('attention', {
            'num_heads': num_heads,
            'head_dim': head_dim,
            'seq_len': seq_len,
            'dropout': dropout,
            'activation': activation,
        })
    
    @staticmethod
    def residual(inner_module: PresetModule) -> PresetModule:
        """Create a residual connection wrapper.
        
        Args:
            inner_module: Inner module to wrap
            
        Returns:
            PresetModule for residual connection
        """
        return PresetModule('residual', {
            'inner': inner_module,
        })
    
    @staticmethod
    def pooling(
        pool_type: str = 'max',
        kernel_size: int = 2,
        stride: int = 2,
    ) -> PresetModule:
        """Create a pooling module.
        
        Args:
            pool_type: Pooling type ('max' or 'avg')
            kernel_size: Pooling kernel size
            stride: Pooling stride
            
        Returns:
            PresetModule for pooling
        """
        return PresetModule('pooling', {
            'type': pool_type,
            'kernel_size': kernel_size,
            'stride': stride,
        })
    
    @staticmethod
    def linear(
        in_features: int,
        out_features: int,
        activation: str = 'relu',
    ) -> PresetModule:
        """Create a linear/fully-connected module.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension
            activation: Activation function name
            
        Returns:
            PresetModule for linear
        """
        return PresetModule('linear', {
            'in_channels': in_features,
            'out_channels': out_features,
            'activation': activation,
        })
    
    @staticmethod
    def embedding(
        vocab_size: int,
        embed_dim: int,
    ) -> PresetModule:
        """Create an embedding module.
        
        Args:
            vocab_size: Vocabulary size
            embed_dim: Embedding dimension
            
        Returns:
            PresetModule for embedding
        """
        return PresetModule('embedding', {
            'in_channels': vocab_size,
            'out_channels': embed_dim,
        })
    
    @staticmethod
    def layernorm(
        normalized_shape: int,
    ) -> PresetModule:
        """Create a layer normalization module.
        
        Args:
            normalized_shape: Shape to normalize
            
        Returns:
            PresetModule for layernorm
        """
        return PresetModule('layernorm', {
            'in_channels': normalized_shape,
            'out_channels': normalized_shape,
        })
    
    @staticmethod
    def batchnorm(
        num_features: int,
    ) -> PresetModule:
        """Create a batch normalization module.
        
        Args:
            num_features: Number of features
            
        Returns:
            PresetModule for batchnorm
        """
        return PresetModule('batchnorm', {
            'in_channels': num_features,
            'out_channels': num_features,
        })
    
    @staticmethod
    def dropout(
        p: float = 0.5,
    ) -> PresetModule:
        """Create a dropout module.
        
        Args:
            p: Dropout probability
            
        Returns:
            PresetModule for dropout
        """
        return PresetModule('dropout', {
            'dropout': p,
        })
    
    @staticmethod
    def flatten() -> PresetModule:
        """Create a flatten module.
        
        Returns:
            PresetModule for flatten
        """
        return PresetModule('flatten', {})
    
    @staticmethod
    def from_dict(config: Dict[str, Any]) -> PresetModule:
        """Create a preset module from dictionary config.
        
        Args:
            config: Configuration dictionary with 'type' key
            
        Returns:
            PresetModule instance
        """
        module_type = config.pop('type')
        return PresetModule(module_type, config)
    
    @staticmethod
    def to_dict(module: BaseModule) -> Dict[str, Any]:
        """Convert a module to dictionary config.
        
        Args:
            module: Module to convert
            
        Returns:
            Configuration dictionary
        """
        config = module.config.to_dict()
        config['type'] = module.config.module_type
        return config
