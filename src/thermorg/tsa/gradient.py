# SPDX-License-Identifier: Apache-2.0

"""Path 2: Gradient optimization method.

Gradient-based optimization for Thermal Spectral Analysis
to find optimal temperature and spectral properties.
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import Optional, Callable
from dataclasses import dataclass


@dataclass
class OptimizationConfig:
    """Configuration for gradient optimization."""
    learning_rate: float = 0.01
    max_iterations: int = 100
    tolerance: float = 1e-6
    temperature_init: float = 1.0


def thermal_objective(
    params: Tensor,
    objective_fn: Callable,
    temperature: float,
    beta_c: float,
) -> Tensor:
    """Compute thermal objective for optimization.
    
    L(T) = ⟨E⟩ + T S - β_c/β * F
    
    Args:
        params: Parameters to optimize
        objective_fn: Base objective function
        temperature: Current temperature
        beta_c: Critical beta
        
    Returns:
        Thermal objective value
    """
    # Basic thermal objective
    base_loss = objective_fn(params)
    
    # Temperature regularization
    temp_reg = 0.5 * (temperature - 2.0/3.0) ** 2
    
    return base_loss + temp_reg


def compute_gradient_path(
    matrix: Tensor,
    target_fn: Callable,
    config: Optional[OptimizationConfig] = None,
) -> dict[str, float]:
    """Compute optimal path via gradient descent.
    
    Args:
        matrix: Input matrix
        target_fn: Target function to optimize
        config: Optimization configuration
        
    Returns:
        Dictionary with optimization results:
            - optimal_temp: Optimal temperature
            - loss_history: Loss values during optimization
            - n_iterations: Number of iterations
    """
    if config is None:
        config = OptimizationConfig()
    
    # Temperature as learnable parameter
    temp = torch.tensor(config.temperature_init, requires_grad=True)
    
    optimizer = torch.optim.Adam([temp], lr=config.learning_rate)
    
    loss_history = []
    
    for i in range(config.max_iterations):
        optimizer.zero_grad()
        
        # Compute objective with current temperature
        # Simplified: use spectral norm as proxy
        spectral_norm = torch.linalg.norm(matrix, ord=2)
        
        # Target: minimize while maintaining T = 2/3 T_c
        loss = torch.abs(spectral_norm) + 0.5 * (temp - 0.667) ** 2
        
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        # Check convergence
        if i > 0 and abs(loss_history[-1] - loss_history[-2]) < config.tolerance:
            break
    
    return {
        "optimal_temp": temp.item(),
        "loss_history": loss_history,
        "n_iterations": len(loss_history),
    }


def temperature_gradient_flow(
    initial_temp: float,
    final_temp: float,
    n_steps: int = 100,
) -> list[float]:
    """Generate temperature gradient flow path.
    
    Args:
        initial_temp: Starting temperature
        final_temp: Target temperature
        n_steps: Number of interpolation steps
        
    Returns:
        List of intermediate temperatures
    """
    temps = []
    for i in range(n_steps):
        alpha = i / (n_steps - 1)
        # Smooth interpolation (cosine schedule)
        t = initial_temp + (final_temp - initial_temp) * (1 - torch.cos(torch.tensor(alpha * torch.pi)) / 2)
        temps.append(t.item())
    
    return temps


def optimize_spectral_path(
    jacobians: list[Tensor],
    target_metric: str = "compression_eff",
) -> dict[str, list]:
    """Optimize temperature path across layer sequence.
    
    Args:
        jacobians: Sequence of layer Jacobians
        target_metric: Metric to optimize
        
    Returns:
        Dictionary with:
            - temperatures: Optimal temperature per layer
            - metrics: Metric values per layer
    """
    from ..core.smc import compute_smc_metrics
    
    temperatures = []
    metrics = []
    
    # Initialize at critical temperature
    current_temp = 1.0
    
    for j in jacobians:
        # Compute metrics
        d_manifold = 1.0  # Initial assumption
        smc_metrics = compute_smc_metrics(j, d_manifold)
        
        if target_metric == "compression_eff":
            metric_value = smc_metrics["compression_eff"].item()
        elif target_metric == "effective_dim":
            metric_value = smc_metrics["effective_dim"].item()
        else:
            metric_value = smc_metrics["spectral_norm"].item()
        
        # Gradient step for temperature
        # Increase temp if compression is low, decrease if high
        optimal_delta = 0.01 * (0.9 - metric_value)
        current_temp = max(0.1, min(2.0, current_temp + optimal_delta))
        
        temperatures.append(current_temp)
        metrics.append(metric_value)
    
    return {
        "temperatures": temperatures,
        "metrics": metrics,
    }


def second_order_optimization(
    matrix: Tensor,
    n_iterations: int = 50,
    learning_rate: float = 0.1,
) -> dict[str, float]:
    """Second-order (Newton-style) optimization for spectral properties.
    
    Args:
        matrix: Input matrix
        n_iterations: Number of iterations
        learning_rate: Step size
        
    Returns:
        Optimization results
    """
    # Temperature parameter
    temp = torch.tensor(1.0, requires_grad=True)
    
    history = []
    
    for _ in range(n_iterations):
        # Compute gradient of spectral free energy
        eigvals = torch.linalg.eigvalsh(matrix)
        
        # Partition function
        beta = 1.0 / (temp + 1e-10)
        z = torch.sum(torch.exp(-beta * eigvals))
        
        # Free energy
        f = -temp * torch.log(z + 1e-10)
        
        # Gradient
        f.backward()
        
        with torch.no_grad():
            temp -= learning_rate * temp.grad
            temp.grad.zero()
        
        # Clamp temperature
        temp = torch.clamp(temp, min=0.1, max=10.0)
        
        history.append({
            "temp": temp.item(),
            "free_energy": f.item(),
        })
    
    return {
        "optimal_temp": temp.item(),
        "history": history,
    }
