# SPDX-License-Identifier: Apache-2.0

"""Theory validation module.

Provides tools for validating SMC theory predictions against
empirical observations.
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import Callable, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class ValidationMetrics:
    """Metrics from theory validation."""
    mae: float  # Mean Absolute Error
    mse: float  # Mean Squared Error
    correlation: float  # Pearson correlation
    within_error: float  # Fraction of predictions within error bounds


def validate_scaling_prediction(
    predicted: Tensor,
    observed: Tensor,
    error_bars: Optional[Tensor] = None,
) -> ValidationMetrics:
    """Validate scaling law predictions against observations.
    
    Args:
        predicted: Theory predictions
        observed: Observed values
        error_bars: Optional uncertainty estimates
        
    Returns:
        ValidationMetrics
    """
    pred = predicted.detach().cpu().numpy()
    obs = observed.detach().cpu().numpy()
    
    mae = np.mean(np.abs(pred - obs))
    mse = np.mean((pred - obs) ** 2)
    
    # Pearson correlation
    if np.std(pred) > 0 and np.std(obs) > 0:
        corr = np.corrcoef(pred, obs)[0, 1]
    else:
        corr = 0.0
    
    # Fraction within error bars
    if error_bars is not None:
        err = error_bars.detach().cpu().numpy()
        within = np.mean(np.abs(pred - obs) <= err)
    else:
        within = 0.0
    
    return ValidationMetrics(
        mae=float(mae),
        mse=float(mse),
        correlation=float(corr),
        within_error=float(within),
    )


def validate_manifold_evolution(
    predicted_dims: list[float],
    observed_jacobians: list[Tensor],
    tolerance: float = 0.1,
) -> dict[str, float]:
    """Validate manifold dimension evolution predictions.
    
    Args:
        predicted_dims: Predicted manifold dimensions per layer
        observed_jacobians: Observed Jacobians per layer
        tolerance: Relative tolerance for validation
        
    Returns:
        Dictionary with validation results
    """
    from ..core.manifold import estimate_from_jacobian
    
    observed_dims = [
        estimate_from_jacobian(j).item() 
        for j in observed_jacobians
    ]
    
    if len(predicted_dims) != len(observed_dims):
        raise ValueError("Length mismatch between predicted and observed dimensions")
    
    errors = [
        abs(pred - obs) / (abs(obs) + 1e-8)
        for pred, obs in zip(predicted_dims, observed_dims)
    ]
    
    return {
        "mean_relative_error": np.mean(errors),
        "max_relative_error": np.max(errors),
        "fraction_within_tolerance": np.mean([e < tolerance for e in errors]),
    }


def validate_compression_efficiency(
    compression_effs: list[float],
    model: torch.nn.Module,
    test_data: Tensor,
) -> dict[str, float]:
    """Validate compression efficiency predictions.
    
    Args:
        compression_effs: Predicted compression efficiencies
        model: Neural network model
        test_data: Test data tensor
        
    Returns:
        Validation results dictionary
    """
    from ..core.smc import compute_smc_metrics
    from ..core.jacobian import extract_jacobians
    
    jacobians = extract_jacobians(model, test_data)
    
    if len(jacobians) != len(compression_effs):
        raise ValueError("Jacobian count doesn't match compression efficiency list")
    
    errors = []
    for j, pred_eff in zip(jacobians, compression_effs):
        # Compute actual efficiency from Jacobian
        d_manifold = 1.0  # Initial assumption
        metrics = compute_smc_metrics(j, d_manifold)
        actual_eff = metrics["compression_eff"].item()
        
        rel_error = abs(pred_eff - actual_eff) / (abs(actual_eff) + 1e-8)
        errors.append(rel_error)
    
    return {
        "mean_error": np.mean(errors),
        "max_error": np.max(errors),
        "correlation": np.corrcoef(compression_effs, 
                                   [1 - e for e in errors])[0, 1],
    }


class TheoryValidator:
    """Main class for validating SMC theory predictions."""
    
    def __init__(self):
        self.validation_results: list[ValidationMetrics] = []
        self.history: list[dict] = []
    
    def validate(
        self,
        name: str,
        predicted: Tensor,
        observed: Tensor,
        error_bars: Optional[Tensor] = None,
    ) -> ValidationMetrics:
        """Run validation for a specific prediction.
        
        Args:
            name: Name of the validation test
            predicted: Predicted values
            observed: Observed values
            error_bars: Optional uncertainty estimates
            
        Returns:
            ValidationMetrics
        """
        metrics = validate_scaling_prediction(predicted, observed, error_bars)
        self.validation_results.append(metrics)
        self.history.append({
            "name": name,
            "metrics": metrics,
        })
        return metrics
    
    def summary(self) -> dict[str, float]:
        """Get summary of all validations.
        
        Returns:
            Dictionary with aggregated metrics
        """
        if not self.validation_results:
            return {}
        
        return {
            "mean_mae": np.mean([m.mae for m in self.validation_results]),
            "mean_corr": np.mean([m.correlation for m in self.validation_results]),
            "total_tests": len(self.validation_results),
        }
