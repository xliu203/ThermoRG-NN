# SPDX-License-Identifier: Apache-2.0

"""Thermodynamic phase transition experiment framework.

Verifies the critical temperature prediction: T_eff* ≈ 0.667 T_c
where T_c is the critical temperature for the phase transition.
"""

from __future__ import annotations

import sys
from pathlib import Path
import time
import torch
from torch import Tensor, nn
from typing import Optional, Callable, Literal
from dataclasses import dataclass, field
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ..core.estimator import MeasurementEstimator, estimate_phase_transition
from .base import BaseExperiment
from ..utils.logging import setup_logger


@dataclass
class ThermodynamicResult:
    """Results from a thermodynamic experiment.
    
    Attributes:
        temperatures: List of effective temperatures tested
        effective_dims: List of measured effective dimensions
        compression_effs: List of compression efficiencies
        free_energies: List of computed free energies
        t_c_estimated: Estimated critical temperature
        t_eff_star: Effective temperature at optimal performance
        transition_width: Width of phase transition
    """
    temperatures: list[float]
    effective_dims: list[float]
    compression_effs: list[float]
    free_energies: list[float]
    t_c_estimated: float = 0.0
    t_eff_star: float = 0.0
    transition_width: float = 0.0
    metadata: dict = field(default_factory=dict)


class ThermodynamicExperiment(BaseExperiment):
    """Experiment to verify T_eff* ≈ 0.667 T_c thermodynamic prediction.
    
    Studies the relationship between effective temperature and
    generalization performance near the critical point.
    
    Example:
        >>> exp = ThermodynamicExperiment(temperature_range=torch.linspace(0.1, 2.0, 20))
        >>> results = exp.run_with_temperature(model, x, y)
        >>> print(f"T_c ≈ {results.t_c_estimated:.3f}")
        >>> print(f"T_eff* ≈ {results.t_eff_star:.3f}")
    """
    
    def __init__(
        self,
        temperature_range: Optional[list[float]] = None,
        n_steps: int = 20,
        t_min: float = 0.1,
        t_max: float = 2.0,
        seed: int = 42,
        device: Optional[torch.device] = None,
        results_dir: str = "results",
    ) -> None:
        """Initialize thermodynamic experiment.
        
        Args:
            temperature_range: Explicit list of temperatures (overrides n_steps/t_min/t_max)
            n_steps: Number of temperature points if range not given
            t_min: Minimum temperature
            t_max: Maximum temperature
            seed: Random seed
            device: Computation device
            results_dir: Directory to save results
        """
        self.device = device or torch.device("cpu")
        self.seed = seed
        
        if temperature_range is not None:
            self.temperatures = temperature_range
        else:
            self.temperatures = torch.linspace(t_min, t_max, n_steps).tolist()
        
        super().__init__("thermodynamic_experiment", results_dir)
        
        # Log configuration
        self.logger.info(f"=== ThermodynamicExperiment Configuration ===")
        self.logger.info(f"temperature_range: {self.temperatures}")
        self.logger.info(f"t_min: {t_min}, t_max: {t_max}, n_steps: {n_steps}")
        self.logger.info(f"seed: {seed}, device: {self.device}")
        self.logger.info(f"=============================================")
        
        self.estimator = MeasurementEstimator(device=self.device)
    
    def run_with_temperature(
        self,
        model: nn.Module,
        x_train: Tensor,
        y_train: Tensor,
        x_test: Tensor,
        y_test: Tensor,
        manifold_dim: float,
        loss_fn: Optional[Callable] = None,
        lr: float = 0.001,
        epochs_per_temp: int = 50,
        batch_size: int = 64,
        verbose: bool = True,
    ) -> ThermodynamicResult:
        """Run thermodynamic experiment with temperature scaling.
        
        Args:
            model: Neural network model (should support temperature via .temperature attribute)
            x_train: Training inputs
            y_train: Training targets
            x_test: Test inputs
            y_test: Test targets
            manifold_dim: Intrinsic manifold dimension
            loss_fn: Loss function
            lr: Learning rate
            epochs_per_temp: Training epochs at each temperature
            batch_size: Batch size
            verbose: Print progress
            
        Returns:
            ThermodynamicResult with findings
        """
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss() if y_test.dim() > 1 and y_test.shape[-1] > 1 else nn.MSELoss()
        
        temperatures = []
        effective_dims = []
        compression_effs = []
        free_energies = []
        test_errors = []
        
        x_train = x_train.to(self.device)
        y_train = y_train.to(self.device)
        x_test = x_test.to(self.device)
        y_test = y_test.to(self.device)
        
        n_train = x_train.shape[0]
        n_batches = (n_train + batch_size - 1) // batch_size
        
        for T in self.temperatures:
            self.heartbeat(f"Starting temperature T={T:.3f}")
            if verbose:
                self.logger.info(f"Testing T = {T:.3f}...")
            
            # Reset model to initial state
            if hasattr(model, 'reset_parameters'):
                model.reset_parameters()
            
            # Set temperature if model supports it
            if hasattr(model, 'temperature'):
                model.temperature = T
            
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            # Train at this temperature
            model.train()
            for epoch in range(epochs_per_temp):
                indices = torch.randperm(n_train, device=self.device)
                for i in range(n_batches):
                    batch_idx = indices[i * batch_size:(i + 1) * batch_size]
                    x_batch = x_train[batch_idx]
                    y_batch = y_train[batch_idx]
                    
                    optimizer.zero_grad()
                    output = model(x_batch)
                    loss = loss_fn(output, y_batch)
                    loss.backward()
                    optimizer.step()
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                # Test error
                output = model(x_test)
                error = loss_fn(output, y_test).item()
                
                # Jacobian metrics (use a sample from training)
                x_sample = x_train[:min(batch_size, n_train)]
                measurement = self.estimator.measure_jacobian(
                    model, x_sample, manifold_dim=manifold_dim
                )
                
                # Estimate free energy (F = -T * log(Z), approximated via loss)
                free_energy = T * error if error > 0 else 0.0
            
            temperatures.append(T)
            effective_dims.append(measurement.effective_dim.item())
            compression_effs.append(measurement.compression_eff.item())
            free_energies.append(free_energy)
            test_errors.append(error)
            
            if verbose:
                self.logger.info(f"  D_eff = {measurement.effective_dim.item():.3f}, "
                      f"η = {measurement.compression_eff.item():.3f}, "
                      f"error = {error:.5f}")
        
        # Find critical temperature from compression efficiency transition
        t_c, confidence = estimate_phase_transition(compression_effs, temperatures)
        
        # Find T_eff* (temperature with minimum test error)
        min_error_idx = min(range(len(test_errors)), key=lambda i: test_errors[i])
        t_eff_star = temperatures[min_error_idx]
        
        # Estimate transition width
        transition_width = self._estimate_transition_width(compression_effs, temperatures)
        
        result = ThermodynamicResult(
            temperatures=temperatures,
            effective_dims=effective_dims,
            compression_effs=compression_effs,
            free_energies=free_energies,
            t_c_estimated=t_c,
            t_eff_star=t_eff_star,
            transition_width=transition_width,
            metadata={
                "manifold_dim": manifold_dim,
                "epochs_per_temp": epochs_per_temp,
                "confidence": confidence,
            }
        )
        
        duration = time.time() - self.start_time
        self.logger.info(f"Experiment completed in {duration/60:.1f} minutes")
        self.logger.info(f"T_c ≈ {t_c:.3f}, T_eff* ≈ {t_eff_star:.3f}, transition_width ≈ {transition_width:.3f}")
        
        # Save results
        results_dict = {
            "experiment_name": "thermodynamic_experiment",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "duration_seconds": duration,
            "config": {
                "temperatures": self.temperatures,
                "manifold_dim": manifold_dim,
                "epochs_per_temp": epochs_per_temp,
                "lr": lr,
                "batch_size": batch_size,
                "seed": self.seed,
            },
            "metrics": {
                "t_c_estimated": t_c,
                "t_eff_star": t_eff_star,
                "transition_width": transition_width,
                "confidence": confidence,
            },
            "data": {
                "temperatures": temperatures,
                "effective_dims": effective_dims,
                "compression_effs": compression_effs,
                "free_energies": free_energies,
                "test_errors": test_errors,
            }
        }
        self.save_results(results_dict)
        
        return result
    
    def run_with_noise(
        self,
        model_factory: Callable[[], nn.Module],
        x: Tensor,
        y: Tensor,
        manifold_dim: float,
        noise_range: Optional[list[float]] = None,
        n_noise_levels: int = 10,
        noise_min: float = 0.01,
        noise_max: float = 1.0,
        loss_fn: Optional[Callable] = None,
        lr: float = 0.001,
        epochs: int = 100,
        batch_size: int = 64,
        verbose: bool = True,
    ) -> ThermodynamicResult:
        """Run thermodynamic experiment using noise as temperature proxy.
        
        Uses label noise level as an effective temperature:
        T_eff ∝ noise_std
        
        Args:
            model_factory: Function to create fresh model for each noise level
            x: Input data
            y: Target data
            manifold_dim: Intrinsic manifold dimension
            noise_range: Explicit noise levels
            n_noise_levels: Number of noise levels if range not given
            noise_min: Minimum noise level
            noise_max: Maximum noise level
            loss_fn: Loss function
            lr: Learning rate
            epochs: Training epochs
            batch_size: Batch size
            verbose: Print progress
            
        Returns:
            ThermodynamicResult
        """
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss() if y.dim() > 1 and y.shape[-1] > 1 else nn.MSELoss()
        
        if noise_range is not None:
            noise_levels = noise_range
        else:
            noise_levels = torch.linspace(noise_min, noise_max, n_noise_levels).tolist()
        
        temperatures = []
        effective_dims = []
        compression_effs = []
        free_energies = []
        test_errors = []
        
        n_samples = x.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for noise_std in noise_levels:
            self.heartbeat(f"Starting noise_std={noise_std:.3f}")
            if verbose:
                self.logger.info(f"Testing noise_std = {noise_std:.3f} (T_eff ∝ {noise_std:.3f})...")
            
            # Create fresh model
            model = model_factory().to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            # Add noise to labels (effective temperature)
            y_noisy = y + torch.randn_like(y) * noise_std
            
            # Train
            model.train()
            indices = torch.randperm(n_samples, device=self.device)
            
            for epoch in range(epochs):
                perm_indices = torch.randperm(n_samples, device=self.device)
                for i in range(n_batches):
                    batch_idx = perm_indices[i * batch_size:(i + 1) * batch_size]
                    x_batch = x[batch_idx].to(self.device)
                    y_batch = y_noisy[batch_idx].to(self.device)
                    
                    optimizer.zero_grad()
                    output = model(x_batch)
                    loss = loss_fn(output, y_batch)
                    loss.backward()
                    optimizer.step()
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                output = model(x.to(self.device))
                error = loss_fn(output, y.to(self.device)).item()
                
                x_sample = x[:batch_size].to(self.device)
                measurement = self.estimator.measure_jacobian(
                    model, x_sample, manifold_dim=manifold_dim
                )
                
                free_energy = noise_std * error
            
            temperatures.append(noise_std)
            effective_dims.append(measurement.effective_dim.item())
            compression_effs.append(measurement.compression_eff.item())
            free_energies.append(free_energy)
            test_errors.append(error)
            
            if verbose:
                self.logger.info(f"  D_eff = {measurement.effective_dim.item():.3f}, "
                      f"η = {measurement.compression_eff.item():.3f}, "
                      f"error = {error:.5f}")
        
        t_c, confidence = estimate_phase_transition(compression_effs, temperatures)
        min_idx = min(range(len(test_errors)), key=lambda i: test_errors[i])
        t_eff_star = temperatures[min_idx]
        transition_width = self._estimate_transition_width(compression_effs, temperatures)
        
        duration = time.time() - self.start_time
        self.logger.info(f"Experiment completed in {duration/60:.1f} minutes")
        self.logger.info(f"T_c ≈ {t_c:.3f}, T_eff* ≈ {t_eff_star:.3f}")
        
        # Save results
        results_dict = {
            "experiment_name": "thermodynamic_experiment",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "duration_seconds": duration,
            "config": {
                "noise_levels": noise_levels,
                "manifold_dim": manifold_dim,
                "epochs": epochs,
                "lr": lr,
                "batch_size": batch_size,
                "seed": self.seed,
            },
            "metrics": {
                "t_c_estimated": t_c,
                "t_eff_star": t_eff_star,
                "transition_width": transition_width,
                "confidence": confidence,
            },
            "data": {
                "temperatures": temperatures,
                "effective_dims": effective_dims,
                "compression_effs": compression_effs,
                "free_energies": free_energies,
                "test_errors": test_errors,
            }
        }
        self.save_results(results_dict)
        
        return ThermodynamicResult(
            temperatures=temperatures,
            effective_dims=effective_dims,
            compression_effs=compression_effs,
            free_energies=free_energies,
            t_c_estimated=t_c,
            t_eff_star=t_eff_star,
            transition_width=transition_width,
            metadata={
                "manifold_dim": manifold_dim,
                "epochs": epochs,
                "noise_range": "proxy",
                "confidence": confidence,
            }
        )
    
    def _estimate_transition_width(
        self,
        compression_effs: list[float],
        temperatures: list[float],
        fraction: float = 0.5,
    ) -> float:
        """Estimate width of phase transition.
        
        Args:
            compression_effs: Compression efficiency values
            temperatures: Corresponding temperatures
            fraction: Fraction of total change to define width
            
        Returns:
            Transition width
        """
        import numpy as np
        
        eff_arr = np.array(compression_effs)
        temp_arr = np.array(temperatures)
        
        # Normalize efficiency
        eff_norm = (eff_arr - eff_arr.min()) / (eff_arr.max() - eff_arr.min() + 1e-8)
        
        # Find width where efficiency crosses fraction of range
        above = np.where(eff_norm >= fraction)[0]
        if len(above) > 0:
            width = temp_arr[above[-1]] - temp_arr[above[0]]
            return float(width)
        
        return 0.0
    
    def save_results(self, result: ThermodynamicResult, path: str | Path) -> None:
        """Save results to JSON file."""
        data = {
            "temperatures": result.temperatures,
            "effective_dims": result.effective_dims,
            "compression_effs": result.compression_effs,
            "free_energies": result.free_energies,
            "t_c_estimated": result.t_c_estimated,
            "t_eff_star": result.t_eff_star,
            "transition_width": result.transition_width,
            "metadata": result.metadata,
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def load_results(path: str | Path) -> ThermodynamicResult:
        """Load results from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        return ThermodynamicResult(
            temperatures=data["temperatures"],
            effective_dims=data["effective_dims"],
            compression_effs=data["compression_effs"],
            free_energies=data["free_energies"],
            t_c_estimated=data["t_c_estimated"],
            t_eff_star=data["t_eff_star"],
            transition_width=data["transition_width"],
            metadata=data.get("metadata", {}),
        )
