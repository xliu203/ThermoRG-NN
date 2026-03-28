# SPDX-License-Identifier: Apache-2.0

"""MLP network architecture with nonlinear activations.

Standard multi-layer perceptron with configurable activation functions.
Used to study the effect of nonlinearity on scaling laws.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from typing import Optional, Literal


class MLP(nn.Module):
    """Multi-layer perceptron with configurable activation functions.
    
    Architecture: y = f_L(W_L f_{L-1}(... f_1(W_1 x)...))
    
    Supports different activation functions per layer or shared across layers.
    Optional layer normalization and dropout for regularization.
    
    Example:
        >>> net = MLP(input_dim=64, hidden_dim=128, output_dim=10, n_layers=3, activation='relu')
        >>> x = torch.randn(32, 64)
        >>> y = net(x)
        >>> print(f"Output shape: {y.shape}")  # (32, 10)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int = 3,
        *,
        activation: Literal["relu", "tanh", "gelu", "sigmoid", "none"] = "relu",
        bias: bool = True,
        layer_norm: bool = False,
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initialize MLP network.
        
        Args:
            input_dim: Input dimension D_in
            hidden_dim: Hidden layer dimension D_h
            output_dim: Output dimension D_out
            n_layers: Number of hidden layers
            activation: Activation function type ('relu', 'tanh', 'gelu', 'sigmoid', 'none')
            bias: Whether to include bias terms
            layer_norm: Whether to apply layer normalization
            dropout: Dropout probability (0 to disable)
            device: Target device
            dtype: Data type
        """
        super().__init__()
        
        factory_kwargs = {"device": device, "dtype": dtype}
        
        # Parse activation function
        self.activation_name = activation
        activation_fn = self._get_activation_fn(activation)
        
        layers = []
        
        # First layer: input -> hidden
        layers.append(nn.Linear(input_dim, hidden_dim, bias=bias, **factory_kwargs))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim, **factory_kwargs))
        if activation_fn is not None:
            layers.append(activation_fn)
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        # Hidden layers: hidden -> hidden
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias, **factory_kwargs))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim, **factory_kwargs))
            if activation_fn is not None:
                layers.append(activation_fn)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        # Output layer: hidden -> output (no activation)
        layers.append(nn.Linear(hidden_dim, output_dim, bias=bias, **factory_kwargs))
        
        self.network = nn.Sequential(*layers)
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self._activation_name = activation
    
    def _get_activation_fn(self, name: str) -> Optional[nn.Module]:
        """Get activation function module by name.
        
        Args:
            name: Activation name
            
        Returns:
            Activation module or None
        """
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "gelu": nn.GELU(),
            "sigmoid": nn.Sigmoid(),
            "none": None,
        }
        return activations.get(name, nn.ReLU())
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through MLP.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        return self.network(x)
    
    def get_layer_output(self, x: Tensor, layer_idx: int) -> Tensor:
        """Get output after a specific hidden layer.
        
        Args:
            x: Input tensor
            layer_idx: Layer index (0-indexed)
            
        Returns:
            Output after specified layer's activation
        """
        for i, module in enumerate(self.network):
            x = module(x)
            if i == layer_idx:
                break
        return x
    
    def get_all_layer_outputs(self, x: Tensor) -> list[Tensor]:
        """Get outputs after each layer.
        
        Args:
            x: Input tensor
            
        Returns:
            List of outputs after each layer
        """
        outputs = []
        for module in self.network:
            x = module(x)
            outputs.append(x)
        return outputs
    
    @property
    def weights(self) -> list[Tensor]:
        """Return list of weight matrices for linear layers."""
        return [m.weight for m in self.network.modules() if isinstance(m, nn.Linear)]
    
    @property
    def biases(self) -> list[Optional[Tensor]]:
        """Return list of bias vectors for linear layers."""
        return [m.bias for m in self.network.modules() if isinstance(m, nn.Linear)]
