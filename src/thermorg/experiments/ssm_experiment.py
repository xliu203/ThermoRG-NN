# SPDX-License-Identifier: Apache-2.0

"""SSM (State Space Model) trace control experiment.

Verifies the role of Tr(A) in the thermal phase transition by comparing
SSMs with Tr(A) = 0 vs Tr(A) ≠ 0.
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
from ..core.estimator import MeasurementEstimator
from ..simulations.networks.ssm import SSMNetwork
from .base import BaseExperiment
from ..utils.logging import setup_logger


@dataclass
class SSMExperimentResult:
    """Results from SSM trace experiment.
    
    Attributes:
        trace_modes: List of trace modes tested
        traces: Actual Tr(A) values for each mode
        spectral_radii: Spectral radius of A for each mode
        effective_dims: Effective dimensions for each mode
        compression_effs: Compression efficiencies for each mode
        test_errors: Test errors for each mode
        critical_difference: Quantified difference between Tr(A)=0 and Tr(A)≠0
    """
    trace_modes: list[str]
    traces: list[float]
    spectral_radii: list[float]
    effective_dims: list[float]
    compression_effs: list[float]
    test_errors: list[float]
    critical_difference: float = 0.0
    metadata: dict = field(default_factory=dict)


class SSMExperiment(BaseExperiment):
    """Experiment to verify Tr(A) = 0 vs Tr(A) ≠ 0 behavior.
    
    Compares SSM networks with:
    - Tr(A) = 0 (critical point, expected to show phase transition)
    - Tr(A) < 0 (below T_c, ordered phase)
    - Tr(A) > 0 (above T_c, disordered phase)
    
    Example:
        >>> exp = SSMExperiment(trace_modes=['zero', 'negative', 'positive'])
        >>> results = exp.run(input_dim=64, state_dim=32, output_dim=10, data=data)
        >>> print(f"Critical difference: {results.critical_difference:.3f}")
    """
    
    def __init__(
        self,
        trace_modes: Optional[list[Literal["zero", "negative", "positive", "free"]]] = None,
        target_traces: Optional[list[float]] = None,
        seed: int = 42,
        device: Optional[torch.device] = None,
        results_dir: str = "results",
    ) -> None:
        """Initialize SSM experiment.
        
        Args:
            trace_modes: List of trace modes to compare
            target_traces: Target trace values for constrained modes
            seed: Random seed
            device: Computation device
            results_dir: Directory to save results
        """
        if trace_modes is None:
            trace_modes = ["zero", "negative", "positive"]
        
        self.trace_modes = trace_modes
        self.target_traces = target_traces or [-1.0, 1.0]
        self.seed = seed
        self.device = device or torch.device("cpu")
        
        super().__init__("ssm_experiment", results_dir)
        
        # Log configuration
        self.logger.info(f"=== SSMExperiment Configuration ===")
        self.logger.info(f"trace_modes: {trace_modes}")
        self.logger.info(f"target_traces: {self.target_traces}")
        self.logger.info(f"seed: {seed}, device: {self.device}")
        self.logger.info(f"===================================")
        
        self.estimator = MeasurementEstimator(device=self.device)
    
    def run(
        self,
        input_dim: int,
        state_dim: int,
        output_dim: int,
        x_train: Tensor,
        y_train: Tensor,
        x_test: Tensor,
        y_test: Tensor,
        manifold_dim: float,
        n_layers: int = 1,
        loss_fn: Optional[Callable] = None,
        lr: float = 0.001,
        epochs: int = 100,
        batch_size: int = 64,
        verbose: bool = True,
    ) -> SSMExperimentResult:
        """Run SSM trace experiment.
        
        Args:
            input_dim: Input dimension
            state_dim: SSM state dimension
            output_dim: Output dimension
            x_train: Training inputs
            y_train: Training targets
            x_test: Test inputs
            y_test: Test targets
            manifold_dim: Intrinsic manifold dimension
            n_layers: Number of SSM layers
            loss_fn: Loss function
            lr: Learning rate
            epochs: Training epochs
            batch_size: Batch size
            verbose: Print progress
            
        Returns:
            SSMExperimentResult with findings
        """
        if loss_fn is None:
            loss_fn = nn.MSELoss()
        
        trace_modes_out = []
        traces = []
        spectral_radii = []
        effective_dims = []
        compression_effs = []
        test_errors = []
        
        x_train = x_train.to(self.device)
        y_train = y_train.to(self.device)
        x_test = x_test.to(self.device)
        y_test = y_test.to(self.device)
        
        n_samples = x_train.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for i, mode in enumerate(self.trace_modes):
            self.heartbeat(f"Starting trace_mode={mode}")
            target_trace = self.target_traces[i] if i < len(self.target_traces) else 0.0
            
            if verbose:
                self.logger.info(f"Testing trace_mode = '{mode}' (target Tr(A) = {target_trace:.2f})...")
            
            # Create SSM with specified trace mode
            torch.manual_seed(self.seed)
            model = SSMNetwork(
                input_dim=input_dim,
                state_dim=state_dim,
                output_dim=output_dim,
                n_layers=n_layers,
                trace_mode=mode,
                target_trace=target_trace,
                device=self.device,
            ).to(self.device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            # Train
            model.train()
            for epoch in range(epochs):
                indices = torch.randperm(n_samples, device=self.device)
                for j in range(n_batches):
                    batch_idx = indices[j * batch_size:(j + 1) * batch_size]
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
                output = model(x_test)
                error = loss_fn(output, y_test).item()
                
                # Get Tr(A) and spectral radius
                trace = model.get_A_trace().item()
                spectral_radius = model.get_spectral_radius().item()
                
                # Jacobian metrics
                x_sample = x_train[:min(batch_size, n_samples)]
                measurement = self.estimator.measure_jacobian(
                    model, x_sample, manifold_dim=manifold_dim
                )
            
            trace_modes_out.append(mode)
            traces.append(trace)
            spectral_radii.append(spectral_radius)
            effective_dims.append(measurement.effective_dim.item())
            compression_effs.append(measurement.compression_eff.item())
            test_errors.append(error)
            
            if verbose:
                self.logger.info(f"  Tr(A) = {trace:.3f}, |λ_max| = {spectral_radius:.3f}, "
                      f"D_eff = {measurement.effective_dim.item():.3f}, "
                      f"η = {measurement.compression_eff.item():.3f}, "
                      f"error = {error:.5f}")
        
        # Compute critical difference between zero-trace and non-zero-trace modes
        crit_diff = self._compute_critical_difference(
            trace_modes_out, compression_effs, test_errors
        )
        
        result = SSMExperimentResult(
            trace_modes=trace_modes_out,
            traces=traces,
            spectral_radii=spectral_radii,
            effective_dims=effective_dims,
            compression_effs=compression_effs,
            test_errors=test_errors,
            critical_difference=crit_diff,
            metadata={
                "state_dim": state_dim,
                "n_layers": n_layers,
                "input_dim": input_dim,
                "output_dim": output_dim,
                "manifold_dim": manifold_dim,
                "epochs": epochs,
            }
        )
        
        duration = time.time() - self.start_time
        self.logger.info(f"Experiment completed in {duration/60:.1f} minutes")
        self.logger.info(f"Critical difference: {crit_diff:.3f}")
        
        # Save results
        results_dict = {
            "experiment_name": "ssm_experiment",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "duration_seconds": duration,
            "config": {
                "trace_modes": trace_modes_out,
                "target_traces": self.target_traces,
                "state_dim": state_dim,
                "n_layers": n_layers,
                "input_dim": input_dim,
                "output_dim": output_dim,
                "manifold_dim": manifold_dim,
                "epochs": epochs,
                "lr": lr,
                "batch_size": batch_size,
                "seed": self.seed,
            },
            "metrics": {
                "critical_difference": crit_diff,
            },
            "data": {
                "traces": traces,
                "spectral_radii": spectral_radii,
                "effective_dims": effective_dims,
                "compression_effs": compression_effs,
                "test_errors": test_errors,
            }
        }
        self.save_results(results_dict)
        
        return result
    
    def _compute_critical_difference(
        self,
        trace_modes: list[str],
        compression_effs: list[float],
        test_errors: list[float],
    ) -> float:
        """Compute quantified difference between Tr(A)=0 and Tr(A)≠0.
        
        Args:
            trace_modes: List of trace modes
            compression_effs: Compression efficiencies
            test_errors: Test errors
            
        Returns:
            Critical difference metric
        """
        import numpy as np
        
        zero_idx = None
        nonzero_compression = []
        nonzero_errors = []
        
        for i, mode in enumerate(trace_modes):
            if mode == "zero":
                zero_idx = i
            else:
                nonzero_compression.append(compression_effs[i])
                nonzero_errors.append(test_errors[i])
        
        if zero_idx is None or len(nonzero_compression) == 0:
            return 0.0
        
        zero_compression = compression_effs[zero_idx]
        zero_error = test_errors[zero_idx]
        
        # Average difference
        avg_nonzero_compression = np.mean(nonzero_compression)
        avg_nonzero_error = np.mean(nonzero_errors)
        
        compression_diff = abs(zero_compression - avg_nonzero_compression)
        error_diff = abs(zero_error - avg_nonzero_error)
        
        # Combined metric
        return compression_diff + error_diff
    
    def save_results(self, result: SSMExperimentResult, path: str | Path) -> None:
        """Save results to JSON file."""
        data = {
            "trace_modes": result.trace_modes,
            "traces": result.traces,
            "spectral_radii": result.spectral_radii,
            "effective_dims": result.effective_dims,
            "compression_effs": result.compression_effs,
            "test_errors": result.test_errors,
            "critical_difference": result.critical_difference,
            "metadata": result.metadata,
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def load_results(path: str | Path) -> SSMExperimentResult:
        """Load results from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        return SSMExperimentResult(
            trace_modes=data["trace_modes"],
            traces=data["traces"],
            spectral_radii=data["spectral_radii"],
            effective_dims=data["effective_dims"],
            compression_effs=data["compression_effs"],
            test_errors=data["test_errors"],
            critical_difference=data["critical_difference"],
            metadata=data.get("metadata", {}),
        )
