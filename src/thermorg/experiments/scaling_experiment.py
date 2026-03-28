# SPDX-License-Identifier: Apache-2.0

"""Scaling law experiment framework.

Verifies the fundamental scaling law: α ∝ 1/d_manifold
where α is the generalization error exponent and d_manifold is the
intrinsic manifold dimension.
"""

from __future__ import annotations

import sys
from pathlib import Path
import time
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset
from typing import Optional, Literal, Callable
from dataclasses import dataclass, field
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ..core.estimator import MeasurementEstimator, MeasurementResult
from ..simulations.manifold_data import ManifoldDataGenerator
from .base import BaseExperiment
from ..utils.logging import setup_logger


@dataclass
class ScalingLawResult:
    """Results from a scaling law experiment.
    
    Attributes:
        manifold_dims: List of manifold dimensions tested
        effective_dims: List of measured effective dimensions
        compression_effs: List of compression efficiencies
        errors: List of test errors for each manifold dimension
        alpha: Fitted scaling exponent
        r_squared: Goodness of fit for power law
    """
    manifold_dims: list[float]
    effective_dims: list[float]
    compression_effs: list[float]
    errors: list[float]
    alpha: float = 0.0
    r_squared: float = 0.0
    metadata: dict = field(default_factory=dict)


class ScalingExperiment(BaseExperiment):
    """Experiment to verify α ∝ 1/d_manifold scaling law.
    
    Tests the hypothesis that generalization error scales inversely
    with manifold dimension.
    
    Example:
        >>> exp = ScalingExperiment(d_manifold_range=[4, 8, 16, 32])
        >>> results = exp.run(network_factory, train_data, test_data)
        >>> print(f"α = {results.alpha:.3f}")
    """
    
    def __init__(
        self,
        d_manifold_range: list[int],
        d_embed: int = 64,
        n_train: int = 1000,
        n_test: int = 500,
        noise_std: float = 0.1,
        seed: int = 42,
        device: Optional[torch.device] = None,
        results_dir: str = "results",
    ) -> None:
        """Initialize scaling experiment.
        
        Args:
            d_manifold_range: List of manifold dimensions to test
            d_embed: Embedding dimension (fixed across experiments)
            n_train: Number of training samples
            n_test: Number of test samples
            noise_std: Noise standard deviation
            seed: Random seed for reproducibility
            device: Computation device
            results_dir: Directory to save results
        """
        self.d_manifold_range = d_manifold_range
        self.d_embed = d_embed
        self.n_train = n_train
        self.n_test = n_test
        self.noise_std = noise_std
        self.seed = seed
        self.device = device or torch.device("cpu")
        
        super().__init__("scaling_experiment", results_dir)
        
        # Log configuration
        self.logger.info(f"=== ScalingExperiment Configuration ===")
        self.logger.info(f"d_manifold_range: {d_manifold_range}")
        self.logger.info(f"d_embed: {d_embed}, n_train: {n_train}, n_test: {n_test}")
        self.logger.info(f"noise_std: {noise_std}, seed: {seed}, device: {self.device}")
        self.logger.info(f"==========================================")
        
        self.data_generator = ManifoldDataGenerator(seed=seed, device=self.device)
        self.estimator = MeasurementEstimator(device=self.device)
        
        # Pre-generate test data for all manifold dimensions
        self.test_data = {}
        for d_m in d_manifold_range:
            _, x_test = self.data_generator.generate(
                n_samples=n_test,
                d_manifold=d_m,
                d_embed=d_embed,
                noise_std=noise_std,
            )
            self.test_data[d_m] = x_test
    
    def run(
        self,
        network_factory: Callable[[int, int], nn.Module],
        loss_fn: Optional[Callable] = None,
        lr: float = 0.001,
        epochs: int = 100,
        batch_size: int = 64,
        verbose: bool = True,
    ) -> ScalingLawResult:
        """Run scaling law experiment.
        
        Args:
            network_factory: Function that creates a network given (input_dim, output_dim)
            loss_fn: Loss function (default: MSE)
            lr: Learning rate
            epochs: Number of training epochs
            batch_size: Training batch size
            verbose: Whether to print progress
            
        Returns:
            ScalingLawResult with experimental findings
        """
        if loss_fn is None:
            loss_fn = nn.MSELoss()
        
        manifold_dims = []
        effective_dims = []
        compression_effs = []
        errors = []
        
        for d_m in self.d_manifold_range:
            self.heartbeat(f"Starting d_manifold={d_m}")
            if verbose:
                self.logger.info(f"Testing d_manifold = {d_m}...")
            
            # Generate training data
            torch.manual_seed(self.seed)
            z_train, x_train = self.data_generator.generate(
                n_samples=self.n_train,
                d_manifold=d_m,
                d_embed=self.d_embed,
                noise_std=self.noise_std,
            )
            
            # Create network
            model = network_factory(self.d_embed, self.d_embed).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            # Train
            model.train()
            n_batches = (self.n_train + batch_size - 1) // batch_size
            
            for epoch in range(epochs):
                indices = torch.randperm(self.n_train, device=self.device)
                epoch_loss = 0.0
                
                for i in range(n_batches):
                    batch_idx = indices[i * batch_size:(i + 1) * batch_size]
                    x_batch = x_train[batch_idx]
                    
                    optimizer.zero_grad()
                    output = model(x_batch)
                    loss = loss_fn(output, x_batch)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
            
            # Evaluate on test set
            model.eval()
            with torch.no_grad():
                x_test = self.test_data[d_m].to(self.device)
                output = model(x_test)
                error = loss_fn(output, x_test).item()
            
            # Measure Jacobian metrics
            with torch.no_grad():
                # Use a batch for more stable Jacobian estimate
                x_sample = x_train[:batch_size].to(self.device)
                measurement = self.estimator.measure_jacobian(
                    model, x_sample, manifold_dim=float(d_m)
                )
            
            manifold_dims.append(float(d_m))
            effective_dims.append(measurement.effective_dim.item())
            compression_effs.append(measurement.compression_eff.item())
            errors.append(error)
            
            if verbose:
                self.logger.info(f"  D_eff = {measurement.effective_dim.item():.3f}, "
                      f"η = {measurement.compression_eff.item():.3f}, "
                      f"error = {error:.5f}")
        
        # Fit scaling law: error ∝ d_manifold^(-alpha)
        alpha, r_sq = self._fit_scaling_law(manifold_dims, errors)
        
        result = ScalingLawResult(
            manifold_dims=manifold_dims,
            effective_dims=effective_dims,
            compression_effs=compression_effs,
            errors=errors,
            alpha=alpha,
            r_squared=r_sq,
            metadata={
                "d_embed": self.d_embed,
                "n_train": self.n_train,
                "n_test": self.n_test,
                "noise_std": self.noise_std,
                "epochs": epochs,
            }
        )
        
        duration = time.time() - self.start_time
        self.logger.info(f"Experiment completed in {duration/60:.1f} minutes")
        self.logger.info(f"Fitted α = {alpha:.3f}, R² = {r_sq:.3f}")
        
        # Save results
        results_dict = {
            "experiment_name": "scaling_experiment",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "duration_seconds": duration,
            "config": {
                "d_manifold_range": self.d_manifold_range,
                "d_embed": self.d_embed,
                "n_train": self.n_train,
                "n_test": self.n_test,
                "noise_std": self.noise_std,
                "seed": self.seed,
                "lr": lr,
                "epochs": epochs,
                "batch_size": batch_size,
            },
            "metrics": {
                "alpha": alpha,
                "r_squared": r_sq,
            },
            "data": {
                "d_manifold_values": manifold_dims,
                "effective_dims": effective_dims,
                "compression_effs": compression_effs,
                "errors": errors,
            }
        }
        self.save_results(results_dict)
        
        return result
    
    def _fit_scaling_law(
        self,
        x: list[float],
        y: list[float],
    ) -> tuple[float, float]:
        """Fit power law: y = c * x^(-alpha)
        
        Args:
            x: Input values (manifold dimensions)
            y: Output values (errors)
            
        Returns:
            Tuple of (alpha, r_squared)
        """
        import numpy as np
        
        x_arr = np.array(x, dtype=np.float64)
        y_arr = np.array(y, dtype=np.float64)
        
        # Log-transform for linear regression
        log_x = np.log(x_arr)
        log_y = np.log(y_arr)
        
        # Linear fit: log(y) = log(c) - alpha * log(x)
        coeffs = np.polyfit(log_x, log_y, deg=1)
        alpha = -coeffs[0]  # Negative because y ∝ x^(-alpha)
        intercept = coeffs[1]
        
        # Compute R-squared
        y_pred = np.exp(intercept) * (x_arr ** (-alpha))
        ss_res = np.sum((y_arr - y_pred) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return alpha, r_squared
    
    def save_results(self, result: ScalingLawResult, path: str | Path) -> None:
        """Save results to JSON file.
        
        Args:
            result: ScalingLawResult to save
            path: Output file path
        """
        data = {
            "manifold_dims": result.manifold_dims,
            "effective_dims": result.effective_dims,
            "compression_effs": result.compression_effs,
            "errors": result.errors,
            "alpha": result.alpha,
            "r_squared": result.r_squared,
            "metadata": result.metadata,
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def load_results(path: str | Path) -> ScalingLawResult:
        """Load results from JSON file.
        
        Args:
            path: Input file path
            
        Returns:
            Loaded ScalingLawResult
        """
        with open(path, 'r') as f:
            data = json.load(f)
        
        return ScalingLawResult(
            manifold_dims=data["manifold_dims"],
            effective_dims=data["effective_dims"],
            compression_effs=data["compression_effs"],
            errors=data["errors"],
            alpha=data["alpha"],
            r_squared=data["r_squared"],
            metadata=data.get("metadata", {}),
        )
