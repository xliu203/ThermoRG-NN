# SPDX-License-Identifier: Apache-2.0

"""Linear network architecture (no activations).

A linear network represents: y = W_L · ... · W_1 · x
with no activation functions between layers.
Used as baseline for studying the effect of nonlinearity.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from typing import Optional


class LinearNetwork(nn.Module):
    """Linear network without activation functions.
    
    Architecture: y = W_L ... W_1 x
    
    All layers are linear transformations with no biases (for simplicity).
    The network maintains constant width across layers.
    
    Example:
        >>> net = LinearNetwork(input_dim=64, hidden_dim=128, output_dim=10, n_layers=3)
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
        bias: bool = False,
        weight_scale: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initialize linear network.
        
        Args:
            input_dim: Input dimension D_in
            hidden_dim: Hidden layer dimension D_h (same for all layers)
            output_dim: Output dimension D_out
            n_layers: Number of hidden layers
            bias: Whether to include bias terms (default False)
            weight_scale: Scale factor for weight initialization
            device: Target device
            dtype: Data type
        """
        super().__init__()
        
        factory_kwargs = {"device": device, "dtype": dtype}
        
        layers = []
        
        # First layer: input -> hidden
        layers.append(nn.Linear(input_dim, hidden_dim, bias=bias, **factory_kwargs))
        
        # Hidden layers: hidden -> hidden
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias, **factory_kwargs))
        
        # Output layer: hidden -> output
        layers.append(nn.Linear(hidden_dim, output_dim, bias=bias, **factory_kwargs))
        
        self.layers = nn.ModuleList(layers)
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        # Initialize weights with small scale for stability
        self._init_weights(weight_scale)
    
    def _init_weights(self, scale: float) -> None:
        """Initialize network weights.
        
        Args:
            scale: Scale factor for weight initialization
        """
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=scale / (layer.weight.shape[1] ** 0.5))
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through linear network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        for layer in self.layers:
            x = layer(x)
        return x
    
    def get_layer_output(self, x: Tensor, layer_idx: int) -> Tensor:
        """Get output of a specific hidden layer.
        
        Args:
            x: Input tensor
            layer_idx: Layer index (0-indexed, 0 is first hidden layer)
            
        Returns:
            Output of specified layer before activation
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == layer_idx:
                break
        return x
    
    def get_all_layer_outputs(self, x: Tensor) -> list[Tensor]:
        """Get outputs of all layers.
        
        Args:
            x: Input tensor
            
        Returns:
            List of layer outputs [h_1, h_2, ..., h_L, y]
        """
        outputs = []
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        return outputs
    
    @property
    def weight_matrices(self) -> list[Tensor]:
        """Return list of weight matrices for each layer.
        
        Returns:
            List of weight tensors [W_1, W_2, ..., W_L]
        """
        return [layer.weight for layer in self.layers]
    
    def get_effective_singular_values(self) -> list[Tensor]:
        """Compute singular values for each layer's weight matrix.
        
        Returns:
            List of singular value tensors for each layer
        """
        return [torch.linalg.svdvals(w) for w in self.weight_matrices]
