# SPDX-License-Identifier: Apache-2.0

"""Regression task datasets for ThermoRG-NN experiments.

Implements smooth function regression tasks:
1. Polynomial functions
2. Trigonometric functions  
3. Mixed smooth functions (combinations)
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import Literal, Optional, Callable
from torch.utils.data import Dataset


class RegressionTaskDataset(Dataset):
    """Dataset for 1D function regression tasks.
    
    Tasks:
    - 'polynomial': Sum of monomials with random coefficients
    - 'trigonometric': Combinations of sin/cos with different frequencies
    - 'mixed': Polynomial + trigonometric combinations
    
    Example:
        >>> dataset = RegressionTaskDataset(task='polynomial', n_dims=4, size=1000)
        >>> x, y = dataset[0]
        >>> print(f"Input shape: {x.shape}, Target: {y.shape}")
    """
    
    def __init__(
        self,
        task: Literal["polynomial", "trigonometric", "mixed"],
        n_dims: int,
        size: int,
        *,
        noise_std: float = 0.0,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize regression task dataset.
        
        Args:
            task: Task type ('polynomial', 'trigonometric', 'mixed')
            n_dims: Input dimension (number of variables)
            size: Number of samples
            noise_std: Additive Gaussian noise std (0 for noiseless)
            seed: Random seed for reproducibility
            device: Torch device for computation
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        self.task = task
        self.n_dims = n_dims
        self.size = size
        self.noise_std = noise_std
        self.device = device or torch.device("cpu")
        
        # Generate random coefficients/parameters
        self._init_parameters()
        
        # Generate all data at once
        self.x = torch.rand(size, n_dims, device=self.device) * 2 - 1  # [-1, 1]^n_dims
        self.y = self._compute_targets()
    
    def _init_parameters(self) -> None:
        """Initialize function parameters based on task type."""
        if self.task == "polynomial":
            # Random coefficients for polynomial terms
            self.degree = 4
            self.coeffs = torch.randn(self.degree, self.n_dims, device=self.device) * 0.5
        
        elif self.task == "trigonometric":
            # Random frequencies and phase shifts
            self.n_freqs = 4
            self.frequencies = torch.randint(1, 5, (self.n_freqs, self.n_dims), device=self.device).float()
            self.phases = torch.rand(self.n_freqs, self.n_dims, device=self.device) * 2 * torch.pi
        
        elif self.task == "mixed":
            # Combination of polynomial and trigonometric
            self.degree = 3
            self.poly_coeffs = torch.randn(self.degree, self.n_dims, device=self.device) * 0.3
            self.n_freqs = 3
            self.frequencies = torch.randint(1, 4, (self.n_freqs, self.n_dims), device=self.device).float()
            self.phases = torch.rand(self.n_freqs, self.n_dims, device=self.device) * 2 * torch.pi
    
    def _compute_targets(self) -> Tensor:
        """Compute target values based on task type."""
        if self.task == "polynomial":
            return self._polynomial_func(self.x)
        elif self.task == "trigonometric":
            return self._trigonometric_func(self.x)
        elif self.task == "mixed":
            return self._mixed_func(self.x)
        
        raise RuntimeError(f"Unknown task: {self.task}")
    
    def _polynomial_func(self, x: Tensor) -> Tensor:
        """Evaluate polynomial function.
        
        f(x) = Σ_{d=1}^{degree} Σ_{i=1}^{n_dims} coeff[d,i] * x_i^d
        
        Args:
            x: Input tensor of shape (batch, n_dims)
            
        Returns:
            Target values of shape (batch,)
        """
        y = torch.zeros(x.shape[0], device=self.device)
        
        for d in range(1, self.degree + 1):
            y += (self.coeffs[d - 1] * (x ** d)).sum(dim=1)
        
        if self.noise_std > 0:
            y = y + torch.randn_like(y) * self.noise_std
        
        return y
    
    def _trigonometric_func(self, x: Tensor) -> Tensor:
        """Evaluate trigonometric function.
        
        f(x) = Σ_{k=1}^{n_freqs} sin(2π freq_k · x + phase_k)
        
        Args:
            x: Input tensor of shape (batch, n_dims)
            
        Returns:
            Target values of shape (batch,)
        """
        y = torch.zeros(x.shape[0], device=self.device)
        
        for k in range(self.n_freqs):
            arg = 2 * torch.pi * (self.frequencies[k] * x).sum(dim=1) + self.phases[k].sum()
            y += torch.sin(arg)
        
        if self.noise_std > 0:
            y = y + torch.randn_like(y) * self.noise_std
        
        return y
    
    def _mixed_func(self, x: Tensor) -> Tensor:
        """Evaluate mixed polynomial-trigonometric function.
        
        f(x) = polynomial(x) + trigonometric(x)
        
        Args:
            x: Input tensor of shape (batch, n_dims)
            
        Returns:
            Target values of shape (batch,)
        """
        y = torch.zeros(x.shape[0], device=self.device)
        
        # Polynomial component
        for d in range(1, self.degree + 1):
            y += (self.poly_coeffs[d - 1] * (x ** d)).sum(dim=1)
        
        # Trigonometric component
        for k in range(self.n_freqs):
            arg = 2 * torch.pi * (self.frequencies[k] * x).sum(dim=1) + self.phases[k].sum()
            y += torch.sin(arg) * 0.5
        
        if self.noise_std > 0:
            y = y + torch.randn_like(y) * self.noise_std
        
        return y
    
    def __len__(self) -> int:
        """Return dataset size."""
        return self.size
    
    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Get a single data point.
        
        Returns:
            Tuple of (input, target) tensors
        """
        return self.x[idx], self.y[idx]
    
    @property
    def input_dim(self) -> int:
        """Return input dimension."""
        return self.n_dims
    
    @property
    def output_dim(self) -> int:
        """Return output dimension (scalar regression)."""
        return 1


def create_polynomial_task(
    n_dims: int = 4,
    train_size: int = 5000,
    test_size: int = 1000,
    noise_std: float = 0.1,
    seed: int = 42,
):
    """Factory function for polynomial regression task.
    
    Args:
        n_dims: Input dimension
        train_size: Training set size
        test_size: Test set size
        noise_std: Target noise std
        seed: Random seed
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    train = RegressionTaskDataset(
        task="polynomial", n_dims=n_dims, size=train_size, noise_std=noise_std, seed=seed
    )
    test = RegressionTaskDataset(
        task="polynomial", n_dims=n_dims, size=test_size, noise_std=noise_std, seed=seed + 1
    )
    return train, test


def create_trigonometric_task(
    n_dims: int = 4,
    train_size: int = 5000,
    test_size: int = 1000,
    noise_std: float = 0.0,
    seed: int = 42,
):
    """Factory function for trigonometric regression task.
    
    Args:
        n_dims: Input dimension
        train_size: Training set size
        test_size: Test set size
        noise_std: Target noise std
        seed: Random seed
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    train = RegressionTaskDataset(
        task="trigonometric", n_dims=n_dims, size=train_size, noise_std=noise_std, seed=seed
    )
    test = RegressionTaskDataset(
        task="trigonometric", n_dims=n_dims, size=test_size, noise_std=noise_std, seed=seed + 1
    )
    return train, test


def create_mixed_task(
    n_dims: int = 4,
    train_size: int = 5000,
    test_size: int = 1000,
    noise_std: float = 0.1,
    seed: int = 42,
):
    """Factory function for mixed (polynomial + trigonometric) regression task.
    
    Args:
        n_dims: Input dimension
        train_size: Training set size
        test_size: Test set size
        noise_std: Target noise std
        seed: Random seed
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    train = RegressionTaskDataset(
        task="mixed", n_dims=n_dims, size=train_size, noise_std=noise_std, seed=seed
    )
    test = RegressionTaskDataset(
        task="mixed", n_dims=n_dims, size=test_size, noise_std=noise_std, seed=seed + 1
    )
    return train, test
