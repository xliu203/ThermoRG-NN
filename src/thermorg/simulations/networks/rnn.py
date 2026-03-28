# SPDX-License-Identifier: Apache-2.0

"""Recurrent Neural Network (RNN) architecture.

Simple RNN with variable sequence length support.
Used for studying temporal dependencies and hidden state dynamics.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from typing import Optional, Literal


class RNNNetwork(nn.Module):
    """Simple RNN network with configurable hidden dimension.
    
    Implements standard RNN:
    h_t = tanh(W_ih x_t + W_hh h_{t-1} + b_h)
    y_t = W_oh h_t + b_o
    
    Supports variable sequence lengths via pack_padded_sequence.
    
    Example:
        >>> net = RNNNetwork(input_dim=64, hidden_dim=128, output_dim=10, n_layers=2)
        >>> x = torch.randn(32, 10, 64)  # (batch, seq_len, input_dim)
        >>> y = net(x)
        >>> print(f"Output shape: {y.shape}")  # (32, 10, 10)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int = 1,
        *,
        activation: Literal["tanh", "relu"] = "tanh",
        bias: bool = True,
        bidirectional: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initialize RNN network.
        
        Args:
            input_dim: Input dimension D_in
            hidden_dim: Hidden state dimension D_h
            output_dim: Output dimension D_out
            n_layers: Number of RNN layers
            activation: Activation function ('tanh' or 'relu')
            bias: Whether to include biases
            bidirectional: Whether to use bidirectional RNN
            device: Target device
            dtype: Data type
        """
        super().__init__()
        
        factory_kwargs = {"device": device, "dtype": dtype}
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # RNN layer
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            nonlinearity=activation,
            bias=bias,
            batch_first=True,
            bidirectional=bidirectional,
            **factory_kwargs,
        )
        
        # Output projection
        fc_input_dim = hidden_dim * self.num_directions
        self.fc = nn.Linear(fc_input_dim, output_dim, bias=bias, **factory_kwargs)
        
        self._init_hidden_state = None
    
    def forward(
        self,
        x: Tensor,
        h0: Optional[Tensor] = None,
        lengths: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass through RNN.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            h0: Initial hidden state of shape (n_layers * num_directions, batch, hidden_dim)
               If None, initialized to zeros.
            lengths: Actual sequence lengths for packed sequences (optional)
            
        Returns:
            Tuple of (output, hidden_state) where:
                - output: Shape (batch_size, seq_len, output_dim * num_directions)
                - hidden_state: Shape (n_layers * num_directions, batch, hidden_dim)
        """
        # Process RNN
        if lengths is not None:
            # Pack padded sequence for efficiency
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            output_packed, h_n = self.rnn(x_packed, h0)
            output, _ = nn.utils.rnn.pad_packed_sequence(output_packed, batch_first=True)
        else:
            output, h_n = self.rnn(x, h0)
        
        # Project to output dimension
        y = self.fc(output)
        
        return y, h_n
    
    def forward_single_step(self, x: Tensor, h: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        """Forward pass for single time step.
        
        Args:
            x: Input of shape (batch_size, input_dim)
            h: Hidden state of shape (n_layers * num_directions, batch, hidden_dim)
            
        Returns:
            Tuple of (output, new_hidden_state)
        """
        # Add sequence dimension
        x = x.unsqueeze(1)  # (batch, 1, input_dim)
        
        if h is None:
            h = self.init_hidden(batch_size=x.shape[0], device=x.device)
        
        output, h_new = self.rnn(x, h)
        y = self.fc(output)  # (batch, 1, output_dim)
        
        return y.squeeze(1), h_new
    
    def init_hidden(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """Initialize hidden state to zeros.
        
        Args:
            batch_size: Batch dimension
            device: Device for tensor
            
        Returns:
            Initial hidden state of zeros
        """
        device = device or next(self.parameters()).device
        return torch.zeros(
            self.n_layers * self.num_directions,
            batch_size,
            self.hidden_dim,
            device=device,
        )
    
    def get_hidden_dynamics(
        self,
        x: Tensor,
        h0: Optional[Tensor] = None,
    ) -> list[Tensor]:
        """Get hidden state at each time step.
        
        Args:
            x: Input sequence of shape (batch_size, seq_len, input_dim)
            h0: Initial hidden state
            
        Returns:
            List of hidden states for each time step
        """
        if h0 is None:
            h0 = self.init_hidden(batch_size=x.shape[0], device=x.device)
        
        hidden_states = []
        h = h0
        
        for t in range(x.shape[1]):
            # Single step forward
            output, h = self.forward_single_step(x[:, t], h)
            hidden_states.append(h)
        
        return hidden_states
    
    @property
    def weight_ih(self) -> list[Tensor]:
        """Get input-to-hidden weights for each layer."""
        return [getattr(self.rnn, f'weight_ih_l{l}') for l in range(self.n_layers)]
    
    @property
    def weight_hh(self) -> list[Tensor]:
        """Get hidden-to-hidden weights for each layer."""
        return [getattr(self.rnn, f'weight_hh_l{l}') for l in range(self.n_layers)]
    
    @property
    def biases_ih(self) -> list[Optional[Tensor]]:
        """Get input-to-hidden biases for each layer."""
        return [getattr(self.rnn, f'bias_ih_l{l}', None) for l in range(self.n_layers)]
    
    @property
    def biases_hh(self) -> list[Optional[Tensor]]:
        """Get hidden-to-hidden biases for each layer."""
        return [getattr(self.rnn, f'bias_hh_l{l}', None) for l in range(self.n_layers)]
