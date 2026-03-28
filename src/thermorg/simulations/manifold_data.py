# SPDX-License-Identifier: Apache-2.0

"""Manifold data generator for ThermoRG-NN experiments.

Generates data x = f(z) + ε where:
- z ∈ R^{d_manifold} is the latent manifold coordinate
- f: R^{d_manifold} → R^{d_embed} is a (non)linear embedding
- ε ~ N(0, σ²) is additive Gaussian noise
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import Literal, Optional


class ManifoldDataGenerator:
    """Generate manifold-embedded data for scaling law experiments.
    
    Data generation follows: x = f(z) + ε
    
    Supports three embedding modes:
    - 'linear': Random Gaussian projection matrix
    - 'nonlinear': Neural network-based embedding with tanh activations
    - 'polynomial': Polynomial feature expansion
    
    Example:
        >>> generator = ManifoldDataGenerator(seed=42)
        >>> z, x = generator.generate(n_samples=1000, d_manifold=8, d_embed=64)
        >>> print(f"Latent shape: {z.shape}, Observed shape: {x.shape}")
    """
    
    def __init__(
        self,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize the manifold data generator.
        
        Args:
            seed: Random seed for reproducibility
            device: Torch device for computation
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        self.device = device or torch.device("cpu")
        self._embedding_weights: Optional[Tensor] = None
        self._nonlinear_net: Optional[torch.nn.Sequential] = None
    
    def generate(
        self,
        n_samples: int,
        d_manifold: int,
        d_embed: int,
        noise_std: float = 0.1,
        mode: Literal["linear", "nonlinear", "polynomial"] = "linear",
    ) -> tuple[Tensor, Tensor]:
        """Generate manifold-embedded data.
        
        Args:
            n_samples: Number of data points to generate
            d_manifold: Intrinsic manifold dimension (dim of z)
            d_embed: Embedding space dimension (dim of x)
            noise_std: Standard deviation of additive Gaussian noise
            mode: Embedding mode ('linear', 'nonlinear', 'polynomial')
            
        Returns:
            Tuple of (z, x) where:
                - z: Latent coordinates in R^{d_manifold}, shape (n_samples, d_manifold)
                - x: Observed data in R^{d_embed}, shape (n_samples, d_embed)
        """
        # Generate latent manifold coordinates z ~ Uniform(-1, 1)
        z = torch.rand(n_samples, d_manifold, device=self.device) * 2 - 1
        
        # Apply embedding f: R^{d_manifold} → R^{d_embed}
        if mode == "linear":
            x = self._linear_embedding(z, d_embed)
        elif mode == "nonlinear":
            x = self._nonlinear_embedding(z, d_embed)
        elif mode == "polynomial":
            x = self._polynomial_embedding(z, d_embed)
        else:
            raise ValueError(f"Unknown mode: {mode}. Choose from 'linear', 'nonlinear', 'polynomial'.")
        
        # Add Gaussian noise
        noise = torch.randn_like(x) * noise_std
        x = x + noise
        
        return z, x
    
    def _linear_embedding(self, z: Tensor, d_embed: int) -> Tensor:
        """Apply random linear projection embedding.
        
        Args:
            z: Input tensor of shape (n_samples, d_manifold)
            d_embed: Target embedding dimension
            
        Returns:
            Embedded tensor of shape (n_samples, d_embed)
        """
        d_manifold = z.shape[1]
        
        # Initialize projection matrix if needed or changed
        if self._embedding_weights is None or self._embedding_weights.shape != (d_embed, d_manifold):
            # Orthogonal random projection (preserves distances better)
            W = torch.randn(d_embed, d_manifold, device=self.device)
            # Orthogonalize columns
            Q, _ = torch.linalg.qr(W)
            self._embedding_weights = Q * (d_embed ** 0.5)
        
        return z @ self._embedding_weights.T
    
    def _nonlinear_embedding(self, z: Tensor, d_embed: int) -> Tensor:
        """Apply nonlinear embedding via small MLP.
        
        Args:
            z: Input tensor of shape (n_samples, d_manifold)
            d_embed: Target embedding dimension
            
        Returns:
            Embedded tensor of shape (n_samples, d_embed)
        """
        d_manifold = z.shape[1]
        
        # Create a simple nonlinear network
        hidden_dim = max(d_manifold, d_embed // 2)
        
        if self._nonlinear_net is None:
            self._nonlinear_net = torch.nn.Sequential(
                torch.nn.Linear(d_manifold, hidden_dim),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden_dim, d_embed),
            ).to(self.device)
        
        return self._nonlinear_net(z)
    
    def _polynomial_embedding(self, z: Tensor, d_embed: int) -> Tensor:
        """Apply polynomial feature expansion embedding.
        
        Args:
            z: Input tensor of shape (n_samples, d_manifold)
            d_embed: Target embedding dimension (approximate)
            
        Returns:
            Embedded tensor of shape (n_samples, d_embed)
        """
        d_manifold = z.shape[1]
        degree = 3
        
        # Compute polynomial features iteratively
        poly_features = [z]
        for d in range(2, degree + 1):
            # Element-wise power
            poly_features.append(z ** d)
        
        # Concatenate all polynomial terms
        poly = torch.cat(poly_features, dim=1)  # (n_samples, d_manifold * degree)
        
        # Project to desired embedding dimension via random linear map
        d_poly = poly.shape[1]
        W = torch.randn(d_poly, d_embed, device=self.device) / (d_poly ** 0.5)
        
        return poly @ W
    
    def generate_batch(
        self,
        batch_size: int,
        d_manifold: int,
        d_embed: int,
        noise_std: float = 0.1,
        mode: Literal["linear", "nonlinear", "polynomial"] = "linear",
    ) -> tuple[Tensor, Tensor]:
        """Generate a single batch of manifold data.
        
        Wrapper for generate() with batch_size as n_samples.
        
        Args:
            batch_size: Number of samples per batch
            d_manifold: Intrinsic manifold dimension
            d_embed: Embedding dimension
            noise_std: Noise standard deviation
            mode: Embedding mode
            
        Returns:
            Tuple of (z, x) tensors
        """
        return self.generate(
            n_samples=batch_size,
            d_manifold=d_manifold,
            d_embed=d_embed,
            noise_std=noise_std,
            mode=mode,
        )
    
    def reset(self) -> None:
        """Reset cached embedding parameters."""
        self._embedding_weights = None
        self._nonlinear_net = None
