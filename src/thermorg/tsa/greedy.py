# SPDX-License-Identifier: Apache-2.0

"""Path 3: Greedy routing method.

Greedy algorithm for Thermal Spectral Analysis routing
to find optimal path through layer sequence.
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import Optional, Callable
from dataclasses import dataclass, field


@dataclass
class RouteStep:
    """Single step in routing decision."""
    layer_index: int
    temperature: float
    compression_eff: float
    cumulated_reward: float


@dataclass 
class RoutingConfig:
    """Configuration for greedy routing."""
    temperature_min: float = 0.1
    temperature_max: float = 2.0
    temperature_step: float = 0.1
    lookahead: int = 3  # Number of steps to lookahead


def compute_local_reward(
    jacobian: Tensor,
    temperature: float,
    prev_manifold_dim: float,
) -> float:
    """Compute local reward for a routing decision.
    
    Args:
        jacobian: Layer Jacobian
        temperature: Current temperature
        prev_manifold_dim: Previous manifold dimension
        
    Returns:
        Local reward value
    """
    from ..core.smc import compute_smc_metrics
    from ..core.scaling import optimal_temperature, temperature_effect
    
    # Compute SMC metrics
    metrics = compute_smc_metrics(jacobian, prev_manifold_dim)
    comp_eff = metrics["compression_eff"].item()
    
    # Compute temperature factor
    T_opt = optimal_temperature(temperature, fraction=2.0/3.0)
    temp_effect = temperature_effect(torch.tensor(temperature / T_opt)).item()
    
    # Reward: compression efficiency modulated by temperature
    reward = comp_eff * temp_effect
    
    return reward


def greedy_route_step(
    jacobians: list[Tensor],
    current_idx: int,
    current_temp: float,
    config: RoutingConfig,
) -> RouteStep:
    """Take single greedy routing step.
    
    Args:
        jacobians: Available Jacobians
        current_idx: Current layer index
        current_temp: Current temperature
        config: Routing configuration
        
    Returns:
        RouteStep with decision
    """
    if current_idx >= len(jacobians):
        raise ValueError("Index out of bounds")
    
    jacobian = jacobians[current_idx]
    
    # Evaluate all possible temperature actions
    temps = torch.arange(
        config.temperature_min,
        config.temperature_max,
        config.temperature_step,
    )
    
    best_reward = -float('inf')
    best_temp = current_temp
    
    for t in temps:
        reward = compute_local_reward(jacobian, t.item(), 1.0)
        if reward > best_reward:
            best_reward = reward
            best_temp = t.item()
    
    return RouteStep(
        layer_index=current_idx,
        temperature=best_temp,
        compression_eff=best_reward,
        cumulated_reward=best_reward,
    )


def greedy_routing(
    jacobians: list[Tensor],
    config: Optional[RoutingConfig] = None,
) -> list[RouteStep]:
    """Greedy routing through layer sequence.
    
    Args:
        jacobians: Sequence of layer Jacobians
        config: Routing configuration
        
    Returns:
        List of RouteStep decisions
    """
    if config is None:
        config = RoutingConfig()
    
    n_layers = len(jacobians)
    route = []
    cumulated_reward = 0.0
    current_temp = 1.0
    
    for idx in range(n_layers):
        step = greedy_route_step(jacobians, idx, current_temp, config)
        cumulated_reward += step.compression_eff
        step = RouteStep(
            layer_index=step.layer_index,
            temperature=step.temperature,
            compression_eff=step.compression_eff,
            cumulated_reward=cumulated_reward,
        )
        route.append(step)
        current_temp = step.temperature
    
    return route


def lookahead_routing(
    jacobians: list[Tensor],
    start_idx: int,
    current_temp: float,
    lookahead: int,
    config: RoutingConfig,
) -> tuple[float, float]:
    """Lookahead routing decision.
    
    Evaluates future rewards before making decision.
    
    Args:
        jacobians: Available Jacobians
        start_idx: Starting layer index
        current_temp: Current temperature
        lookahead: Number of steps to lookahead
        config: Routing configuration
        
    Returns:
        Tuple of (best_temp, best_cumulative_reward)
    """
    if start_idx >= len(jacobians):
        return current_temp, 0.0
    
    # Evaluate each possible temperature action
    temps = torch.arange(
        config.temperature_min,
        config.temperature_max,
        config.temperature_step,
    )
    
    best_temp = current_temp
    best_cum_reward = -float('inf')
    
    for t in temps:
        cum_reward = 0.0
        temp = t.item()
        prev_dim = 1.0
        
        for step in range(lookahead):
            idx = start_idx + step
            if idx >= len(jacobians):
                break
            
            reward = compute_local_reward(jacobians[idx], temp, prev_dim)
            cum_reward += reward
            
            # Update for next step
            from ..core.smc import compute_smc_metrics
            metrics = compute_smc_metrics(jacobians[idx], prev_dim)
            prev_dim = metrics["compression_eff"].item()
        
        if cum_reward > best_cum_reward:
            best_cum_reward = cum_reward
            best_temp = t.item()
    
    return best_temp, best_cum_reward


def optimal_routing(
    jacobians: list[Tensor],
    lookahead: int = 3,
) -> list[float]:
    """Optimal routing with lookahead.
    
    Args:
        jacobians: Sequence of layer Jacobians
        lookahead: Number of steps to lookahead
        
    Returns:
        List of optimal temperatures per layer
    """
    config = RoutingConfig(lookahead=lookahead)
    temperatures = []
    current_temp = 1.0
    prev_dim = 1.0
    
    for idx in range(len(jacobians)):
        # Lookahead to evaluate all options
        best_temp, _ = lookahead_routing(
            jacobians, idx, current_temp, lookahead, config
        )
        temperatures.append(best_temp)
        current_temp = best_temp
        
        # Update manifold dimension for next step
        from ..core.smc import compute_smc_metrics
        metrics = compute_smc_metrics(jacobians[idx], prev_dim)
        prev_dim = metrics["compression_eff"].item()
    
    return temperatures


def routing_summary(route: list[RouteStep]) -> dict[str, float]:
    """Generate summary statistics for routing result.
    
    Args:
        route: List of RouteStep decisions
        
    Returns:
        Summary dictionary
    """
    if not route:
        return {
            "n_steps": 0,
            "mean_temp": 0.0,
            "mean_compression": 0.0,
            "final_reward": 0.0,
        }
    
    temps = [step.temperature for step in route]
    comps = [step.compression_eff for step in route]
    
    return {
        "n_steps": len(route),
        "mean_temp": sum(temps) / len(temps),
        "std_temp": (sum((t - sum(temps)/len(temps))**2 for t in temps) / len(temps)) ** 0.5,
        "mean_compression": sum(comps) / len(comps),
        "final_reward": route[-1].cumulated_reward if route else 0.0,
    }
