# SPDX-License-Identifier: Apache-2.0

"""Algorithmic task datasets for ThermoRG-NN experiments.

Implements three fundamental algorithmic tasks:
1. Modular arithmetic (mod p)
2. Parity detection (odd/even)
3. Sequence reversal
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import Literal, Optional, Callable
from torch.utils.data import Dataset


class AlgorithmicTaskDataset(Dataset):
    """Dataset for algorithmic tasks used in ThermoRG-NN experiments.
    
    Tasks:
    - 'mod': Modular arithmetic (x + y) mod p
    - 'parity': Parity detection (odd/even)
    - 'reverse': Sequence reversal
    
    Example:
        >>> dataset = AlgorithmicTaskDataset(task='mod', p=7, size=1000)
        >>> x, y = dataset[0]
        >>> print(f"Input: {x}, Target: {y}")
    """
    
    def __init__(
        self,
        task: Literal["mod", "parity", "reverse"],
        size: int,
        *,
        p: Optional[int] = None,
        seq_len: int = 10,
        modulus_range: tuple[int, int] = (2, 13),
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize algorithmic task dataset.
        
        Args:
            task: Task type ('mod', 'parity', 'reverse')
            size: Number of samples in dataset
            p: Modulus for 'mod' task (required if task='mod')
            seq_len: Sequence length for 'reverse' task
            modulus_range: Range (min, max) for random p selection
            seed: Random seed for reproducibility
            device: Torch device for computation
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        self.task = task
        self.size = size
        self.device = device or torch.device("cpu")
        
        if task == "mod":
            if p is None:
                raise ValueError("p must be specified for 'mod' task")
            self.p = p
            self._generate_mod_data()
        elif task == "parity":
            self._generate_parity_data()
        elif task == "reverse":
            self.seq_len = seq_len
            self._generate_reverse_data()
        else:
            raise ValueError(f"Unknown task: {task}. Choose from 'mod', 'parity', 'reverse'.")
    
    def _generate_mod_data(self) -> None:
        """Generate modular arithmetic data (x + y) mod p."""
        # Random inputs in range [0, p-1]
        x = torch.randint(0, self.p, (self.size,), device=self.device)
        y = torch.randint(0, self.p, (self.size,), device=self.device)
        
        # Target: (x + y) mod p
        target = (x + y) % self.p
        
        # One-hot encode inputs and target
        self.x = torch.zeros(self.size, self.p, device=self.device)
        self.x.scatter_(1, x.unsqueeze(1), 1.0)
        
        self.y = torch.zeros(self.size, self.p, device=self.device)
        self.y.scatter_(1, y.unsqueeze(1), 1.0)
        
        self.target = torch.zeros(self.size, self.p, device=self.device)
        self.target.scatter_(1, target.unsqueeze(1), 1.0)
    
    def _generate_parity_data(self) -> None:
        """Generate parity detection data (odd/even)."""
        # Generate random bitstrings
        self.x = torch.randint(0, 2, (self.size, 8), device=self.device).float()
        
        # Target: parity bit (1 if odd number of 1s, 0 if even)
        parity = (self.x.sum(dim=1) % 2).long()
        self.target = torch.zeros(self.size, 2, device=self.device)
        self.target.scatter_(1, parity.unsqueeze(1), 1.0)
    
    def _generate_reverse_data(self) -> None:
        """Generate sequence reversal data."""
        vocab_size = 10
        
        # Generate random sequences
        self.x = torch.randint(0, vocab_size, (self.size, self.seq_len), device=self.device).float()
        
        # Target: reversed sequence (flattened for regression)
        reversed_seq = torch.flip(self.x, dims=[1])
        self.target = reversed_seq.reshape(self.size, -1).float()
    
    def __len__(self) -> int:
        """Return dataset size."""
        return self.size
    
    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Get a single data point.
        
        Returns:
            Tuple of (input, target) tensors
        """
        if self.task == "mod":
            # Concatenate one-hot encoded x and y
            inp = torch.cat([self.x[idx], self.y[idx]], dim=0)
            return inp, self.target[idx]
        elif self.task == "parity":
            return self.x[idx], self.target[idx]
        elif self.task == "reverse":
            return self.x[idx], self.target[idx]
        
        raise RuntimeError(f"Unknown task: {self.task}")
    
    @property
    def input_dim(self) -> int:
        """Return input dimension."""
        if self.task == "mod":
            return self.p * 2
        elif self.task == "parity":
            return 8
        elif self.task == "reverse":
            return self.seq_len
        
        raise RuntimeError(f"Unknown task: {self.task}")
    
    @property
    def output_dim(self) -> int:
        """Return output dimension."""
        if self.task == "mod":
            return self.p
        elif self.task == "parity":
            return 2
        elif self.task == "reverse":
            return self.seq_len
        
        raise RuntimeError(f"Unknown task: {self.task}")


def create_mod_task(p: int = 7, train_size: int = 5000, test_size: int = 1000, seed: int = 42):
    """Factory function for modular arithmetic task.
    
    Args:
        p: Modulus value
        train_size: Training set size
        test_size: Test set size
        seed: Random seed
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    train = AlgorithmicTaskDataset(task="mod", size=train_size, p=p, seed=seed)
    test = AlgorithmicTaskDataset(task="mod", size=test_size, p=p, seed=seed + 1)
    return train, test


def create_parity_task(train_size: int = 5000, test_size: int = 1000, seed: int = 42):
    """Factory function for parity detection task.
    
    Args:
        train_size: Training set size
        test_size: Test set size
        seed: Random seed
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    train = AlgorithmicTaskDataset(task="parity", size=train_size, seed=seed)
    test = AlgorithmicTaskDataset(task="parity", size=test_size, seed=seed + 1)
    return train, test


def create_reverse_task(seq_len: int = 10, train_size: int = 5000, test_size: int = 1000, seed: int = 42):
    """Factory function for sequence reversal task.
    
    Args:
        seq_len: Sequence length
        train_size: Training set size
        test_size: Test set size
        seed: Random seed
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    train = AlgorithmicTaskDataset(task="reverse", size=train_size, seq_len=seq_len, seed=seed)
    test = AlgorithmicTaskDataset(task="reverse", size=test_size, seq_len=seq_len, seed=seed + 1)
    return train, test
