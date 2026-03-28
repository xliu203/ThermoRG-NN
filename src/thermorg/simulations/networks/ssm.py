# SPDX-License-Identifier: Apache-2.0

"""State Space Model (SSM) network with controllable Tr(A).

Structured State Space Model (S4-like) with explicit control over
the trace of the state transition matrix A. Used to study the
relationship between Tr(A) and thermal phase transitions.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from typing import Optional, Literal


class SSMNetwork(nn.Module):
    """State Space Model network with controllable Tr(A).
    
    Implements a discretized state space model:
    
    h_{t+1} = A h_t + B x_t
    y_t = C h_t + D x_t
    
    where A ∈ R^{N×N} is the state transition matrix.
    The trace of A (Tr(A)) can be explicitly controlled to study
    thermal phase transitions near T_c.
    
    Example:
        >>> net = SSMNetwork(state_dim=64, n_states=16, output_dim=10, trace_mode='zero')
        >>> x = torch.randn(32, 64)  # (batch, input_dim)
        >>> y = net(x)
        >>> print(f"Output shape: {y.shape}")  # (32, 10)
    """
    
    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        output_dim: int,
        n_layers: int = 1,
        *,
        trace_mode: Literal["zero", "negative", "positive", "free"] = "zero",
        target_trace: float = 0.0,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initialize SSM network.
        
        Args:
            input_dim: Input dimension D_in
            state_dim: State dimension N
            output_dim: Output dimension D_out
            n_layers: Number of SSM layers (stacked)
            trace_mode: How to set Tr(A):
                - 'zero': Tr(A) = 0 (critical point)
                - 'negative': Tr(A) < 0 (below T_c)
                - 'positive': Tr(A) > 0 (above T_c)
                - 'free': Unconstrained (learnable)
            target_trace: Target trace value for constrained modes
            bias: Whether to include input/output biases
            device: Target device
            dtype: Data type
        """
        super().__init__()
        
        factory_kwargs = {"device": device, "dtype": dtype}
        
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.trace_mode = trace_mode
        self.target_trace = target_trace
        
        # Build SSM layers
        self.layers = nn.ModuleList()
        
        for layer_idx in range(n_layers):
            input_d = input_dim if layer_idx == 0 else state_dim
            
            # State transition matrix A (discretized)
            A = self._initialize_A(trace_mode, target_trace, state_dim, **factory_kwargs)
            
            # Input-to-state projection B
            B = nn.Parameter(torch.randn(state_dim, input_d, **factory_kwargs) * 0.5)
            
            # State-to-output projection C
            C = nn.Parameter(torch.randn(output_dim, state_dim, **factory_kwargs) * 0.5)
            
            # Direct feedthrough D (optional)
            D = nn.Parameter(torch.randn(output_dim, input_d, **factory_kwargs) * 0.01) if bias else None
            
            self.layers.append(nn.ModuleDict({
                "A": A,
                "B": B,
                "C": C,
                "D": D,
            }))
        
        # For sequence processing, aggregate outputs
        self.use_sequence = True
    
    def _initialize_A(
        self,
        trace_mode: str,
        target_trace: float,
        state_dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> nn.Parameter:
        """Initialize state transition matrix A.
        
        Args:
            trace_mode: Trace constraint mode
            target_trace: Target trace value
            state_dim: Dimension of state matrix
            device: Target device
            dtype: Data type
            
        Returns:
            Parameterized matrix A
        """
        if trace_mode == "free":
            # Learnable unconstrained matrix
            A = nn.Parameter(torch.randn(state_dim, state_dim, device=device, dtype=dtype) * 0.1)
        else:
            # Initialize with specific trace constraint
            A = torch.randn(state_dim, state_dim, device=device, dtype=dtype) * 0.1
            
            # Make A approximately orthogonal (for stability)
            A = A + A.T  # Symmetric
            
            # Adjust trace
            current_trace = torch.trace(A)
            if trace_mode == "zero":
                # Set trace to zero by adjusting diagonal
                trace_adjustment = -current_trace / state_dim
                A = A + trace_adjustment * torch.eye(state_dim, device=device, dtype=dtype)
            elif trace_mode == "negative":
                # Ensure negative trace
                if current_trace >= 0:
                    trace_adjustment = -(current_trace + abs(target_trace)) / state_dim
                    A = A + trace_adjustment * torch.eye(state_dim, device=device, dtype=dtype)
            elif trace_mode == "positive":
                # Ensure positive trace
                if current_trace <= 0:
                    trace_adjustment = (abs(target_trace) - current_trace) / state_dim
                    A = A + trace_adjustment * torch.eye(state_dim, device=device, dtype=dtype)
            
            A = nn.Parameter(A)
        
        return A
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through SSM network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim) or 
               (batch_size, seq_len, input_dim) for sequences
               
        Returns:
            Output tensor of shape (batch_size, output_dim) or
            (batch_size, seq_len, output_dim) for sequences
        """
        is_sequence = x.dim() == 3
        
        if not is_sequence:
            # Add sequence dimension: (batch, dim) -> (batch, 1, dim)
            x = x.unsqueeze(1)
        
        batch_size, seq_len, _ = x.shape
        
        # Process each SSM layer
        h = torch.zeros(batch_size, self.state_dim, device=x.device, dtype=x.dtype)
        
        outputs = []
        for layer in self.layers:
            A = layer["A"]
            B = layer["B"]
            C = layer["C"]
            D = layer["D"]
            
            hs = []  # State outputs
            for t in range(seq_len):
                # h_{t+1} = A h_t + B x_t
                h = torch.matmul(A, h.unsqueeze(-1)).squeeze(-1) + torch.matmul(B, x[:, t].unsqueeze(-1)).squeeze(-1)
                hs.append(h)
            
            # y_t = C h_t + D x_t
            h = torch.stack(hs, dim=1)  # (batch, seq, state)
            if D is not None:
                y = torch.matmul(C, h.transpose(-1, -2)).transpose(-1, -2) + torch.matmul(D, x.transpose(-1, -2)).transpose(-1, -2)
            else:
                y = torch.matmul(C, h.transpose(-1, -2)).transpose(-1, -2)
            outputs.append(y)
        
        # Return last layer output
        y = outputs[-1]
        
        if not is_sequence:
            y = y.squeeze(1)  # (batch, output_dim)
        
        return y
    
    def get_A_trace(self) -> Tensor:
        """Get trace of state transition matrix A for first layer.
        
        Returns:
            Trace of A as tensor
        """
        A = self.layers[0]["A"]
        return torch.trace(A)
    
    def get_all_A_traces(self) -> list[Tensor]:
        """Get traces of A matrices for all layers.
        
        Returns:
            List of traces for each layer's A matrix
        """
        return [torch.trace(layer["A"]) for layer in self.layers]
    
    def get_spectral_radius(self, layer_idx: int = 0) -> Tensor:
        """Get spectral radius (largest eigenvalue magnitude) of A.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Spectral radius of A
        """
        A = self.layers[layer_idx]["A"]
        eigenvalues = torch.linalg.eigvals(A)
        return torch.max(torch.abs(eigenvalues))
    
    @property
    def state_matrices(self) -> list[Tensor]:
        """Return list of A matrices for each layer."""
        return [layer["A"].data for layer in self.layers]
