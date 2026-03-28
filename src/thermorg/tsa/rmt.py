# SPDX-License-Identifier: Apache-2.0

"""Path 1: Random Matrix Theory (RMT) analytical method.

Analytical approach to Thermal Spectral Analysis using RMT
for direct computation of spectral properties.
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class RMTParams:
    """Parameters for RMT analysis."""
    beta: float = 1.0  # Dyson index (1=GOE, 2=GUE, 4=GSE)
    n: int = 100  # Matrix dimension
    temperature: float = 1.0  # Temperature


def marchenko_pastur_cdf(x: float, sigma: float = 1.0, n: int = 1) -> float:
    """Marchenko-Pastur cumulative distribution function.
    
    For sample covariance matrices with ratio γ = n/p.
    
    Args:
        x: Position to evaluate
        sigma: Standard deviation of entries
        n: Sample size
        
    Returns:
        CDF value at x
    """
    gamma = 1.0  # Assuming n=p for simplicity
    lambda_minus = sigma ** 2 * (1 - np.sqrt(gamma)) ** 2
    lambda_plus = sigma ** 2 * (1 + np.sqrt(gamma)) ** 2
    
    if x < lambda_minus or x > lambda_plus:
        return 0.0 if x < lambda_minus else 1.0
    
    # Simplified CDF computation
    t = np.sqrt((lambda_plus - x) * (x - lambda_minus))
    return 0.5 + np.sign(x - sigma ** 2) * 0.5 * np.sqrt(1 - ((x - sigma ** 2) / t) ** 2)


def spectral_density_goe(eigenvalues: Tensor, n_bins: int = 100) -> tuple[Tensor, Tensor]:
    """Compute spectral density from eigenvalues (histogram method).
    
    Args:
        eigenvalues: Eigenvalue spectrum
        n_bins: Number of histogram bins
        
    Returns:
        Tuple of (density, bin_centers)
    """
    eig = eigenvalues.detach().cpu().numpy()
    
    density, edges = np.histogram(eig, bins=n_bins, density=True)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    
    return torch.tensor(density), torch.tensor(bin_centers)


def spacing_distribution(
    eigenvalues: Tensor,
    normalization: str = "gamma1",
) -> Tensor:
    """Compute level spacing distribution.
    
    Args:
        eigenvalues: Sorted eigenvalues
        normalization: Spacing normalization ('gamma1' or 'gamma2')
        
    Returns:
        Normalized spacing sequence
    """
    eig = eigenvalues.detach().cpu().numpy()
    eig = np.sort(eig)
    
    # Compute spacings
    spacings = np.diff(eig)
    
    # Normalize by mean spacing (Wigner surmise approximation)
    s_mean = np.mean(spacings)
    normalized = spacings / s_mean
    
    return torch.tensor(normalized)


def wigner_surmise(s: Tensor, beta: int = 1) -> Tensor:
    """Wigner surmise for level spacing distribution.
    
    p(s) ∝ s^β exp(-γ s²)
    
    Args:
        s: Spacing values
        beta: Dyson index
        
    Returns:
        Probability density
    """
    if beta == 1:  # GOE
        const = np.pi / 2
    elif beta == 2:  # GUE
        const = 32 / np.pi ** 2
    else:
        const = 2.0
    
    return const * s ** beta * torch.exp(-const / 4 * s ** 2)


def compute_spectral_entropy(eigenvalues: Tensor) -> float:
    """Compute spectral entropy from eigenvalues.
    
    H = -∑ λ_i log λ_i
    
    Args:
        eigenvalues: Eigenvalue spectrum
        
    Returns:
        Spectral entropy
    """
    eig = torch.abs(eigenvalues)
    eig = eig / (torch.sum(eig) + 1e-10)  # Normalize
    
    # Avoid log(0)
    entropy = -torch.sum(eig * torch.log(eig + 1e-10))
    
    return entropy.item()


def rmt_analytical_path(
    matrix: Tensor,
    temperature: float = 1.0,
) -> dict[str, float]:
    """RMT analytical path for Thermal Spectral Analysis.
    
    Args:
        matrix: Input matrix (e.g., Jacobian or Hamiltonian)
        temperature: Temperature parameter
        
    Returns:
        Dictionary with spectral properties:
            - free_energy: Analytic free energy estimate
            - entropy: Spectral entropy
            - gap_ratio: Average level spacing ratio
            - temperature_eff: Effective temperature
    """
    # Compute eigenvalues
    eigenvalues = torch.linalg.eigvalsh(matrix)
    
    # Compute spectral gap ratio
    eig_sorted = torch.sort(eigenvalues).values
    spacings = torch.diff(eig_sorted)
    mean_spacing = torch.mean(spacings)
    
    # Gap ratio <r> (oscillator repulsion parameter)
    r_values = []
    for i in range(len(spacings) - 1):
        s1 = spacings[i].item()
        s2 = spacings[i + 1].item()
        if s1 + s2 > 0:
            r_values.append(min(s1, s2) / max(s1, s2))
    
    gap_ratio = np.mean(r_values) if r_values else 0.0
    
    # Temperature rescaling
    temp_eff = temperature * (1 + 0.1 * gap_ratio)
    
    # Free energy from trace (partition function approximation)
    beta = 1.0 / (temp_eff + 1e-10)
    z = torch.sum(torch.exp(-beta * torch.abs(eigenvalues)))
    free_energy = -temp_eff * torch.log(z + 1e-10)
    
    return {
        "free_energy": free_energy.item(),
        "entropy": compute_spectral_entropy(eigenvalues),
        "gap_ratio": gap_ratio,
        "temperature_eff": temp_eff,
        "n_eigenvalues": len(eigenvalues),
    }


def level_statistics(eigenvalues: Tensor) -> str:
    """Determine level statistics from eigenvalue spectrum.
    
    Args:
        eigenvalues: Eigenvalue spectrum
        
    Returns:
        'poisson', 'goe', 'gue', or 'gse' based on statistics
    """
    spacing = spacing_distribution(eigenvalues)
    
    # Compute delta statistics
    # Simplified: use variance of spacings
    s = spacing.cpu().numpy()
    
    # Poisson: variance = 1
    # GOE: variance ≈ 0.286
    # GUE: variance ≈ 0.179
    
    var = np.var(s)
    
    if var > 0.5:
        return "poisson"
    elif var > 0.23:
        return "goe"
    elif var > 0.15:
        return "gue"
    else:
        return "gse"
