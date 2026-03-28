# SPDX-License-Identifier: Apache-2.0

"""Experiment results recording and analysis.

Provides classes and functions for storing, saving, loading,
and analyzing experimental results.
"""

from __future__ import annotations

import json
import torch
from pathlib import Path
from typing import Optional, Any, Literal
from dataclasses import dataclass, field, asdict
from datetime import datetime
import numpy as np


@dataclass
class ExperimentResults:
    """Container for experiment results with metadata.
    
    Attributes:
        experiment_name: Name identifier for the experiment
        timestamp: When the experiment was run
        config: Experiment configuration parameters
        metrics: Dictionary of measured metrics
        artifacts: Optional paths to saved artifacts (models, plots, etc.)
        tags: Optional tags for categorization
        
    Example:
        >>> results = ExperimentResults(
        ...     experiment_name="scaling_law_v1",
        ...     config={"d_manifold": [4, 8, 16], "epochs": 100},
        ...     metrics={"alpha": 0.67, "r_squared": 0.95}
        ... )
        >>> results.save("results/exp_001.json")
    """
    experiment_name: str
    config: dict[str, Any]
    metrics: dict[str, float]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    artifacts: dict[str, str] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert non-serializable types
        for k, v in data.get("metrics", {}).items():
            if isinstance(v, (torch.Tensor, np.ndarray)):
                data["metrics"][k] = v.tolist() if hasattr(v, 'tolist') else float(v)
        return data
    
    def save(self, path: str | Path, include_timestamp: bool = True) -> None:
        """Save results to JSON file.
        
        Args:
            path: Output file path
            include_timestamp: Whether to append timestamp to filename
        """
        path = Path(path)
        
        if include_timestamp and "{ts}" not in str(path):
            stem = path.stem
            suffix = path.suffix
            path = path.parent / f"{stem}_{self.timestamp.replace(':', '-')}{suffix}"
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str | Path) -> "ExperimentResults":
        """Load results from JSON file.
        
        Args:
            path: Input file path
            
        Returns:
            Loaded ExperimentResults instance
        """
        with open(path, 'r') as f:
            data = json.load(f)
        
        return cls(
            experiment_name=data["experiment_name"],
            timestamp=data["timestamp"],
            config=data["config"],
            metrics=data["metrics"],
            artifacts=data.get("artifacts", {}),
            tags=data.get("tags", []),
            notes=data.get("notes", ""),
        )
    
    def add_metric(self, name: str, value: float) -> None:
        """Add a metric to results.
        
        Args:
            name: Metric name
            value: Metric value
        """
        self.metrics[name] = float(value)
    
    def add_artifact(self, name: str, path: str) -> None:
        """Add an artifact path.
        
        Args:
            name: Artifact name
            path: Artifact file path
        """
        self.artifacts[name] = str(path)
    
    def summary(self) -> str:
        """Generate human-readable summary.
        
        Returns:
            Summary string
        """
        lines = [
            f"Experiment: {self.experiment_name}",
            f"Timestamp: {self.timestamp}",
            f"Config: {json.dumps(self.config, indent=2)}",
            "Metrics:",
        ]
        for k, v in self.metrics.items():
            lines.append(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
        
        if self.tags:
            lines.append(f"Tags: {', '.join(self.tags)}")
        
        return "\n".join(lines)


class ResultsCollection:
    """Collection of experiment results with analysis utilities.
    
    Manages multiple ExperimentResults and provides aggregation,
    filtering, and comparison functionality.
    
    Example:
        >>> collection = ResultsCollection()
        >>> collection.add(results1)
        >>> collection.add(results2)
        >>> avg_metrics = collection.average_metrics(group_by="config.epochs")
    """
    
    def __init__(self) -> None:
        """Initialize empty collection."""
        self.results: list[ExperimentResults] = []
    
    def add(self, result: ExperimentResults) -> None:
        """Add a result to the collection.
        
        Args:
            result: ExperimentResults to add
        """
        self.results.append(result)
    
    def filter(self, tag: Optional[str] = None, experiment_name: Optional[str] = None) -> "ResultsCollection":
        """Filter results by tag or experiment name.
        
        Args:
            tag: Filter by tag
            experiment_name: Filter by experiment name (partial match)
            
        Returns:
            New ResultsCollection with filtered results
        """
        filtered = ResultsCollection()
        
        for r in self.results:
            if tag and tag not in r.tags:
                continue
            if experiment_name and experiment_name not in r.experiment_name:
                continue
            filtered.add(r)
        
        return filtered
    
    def average_metrics(self, metric_names: Optional[list[str]] = None) -> dict[str, float]:
        """Compute average of metrics across all results.
        
        Args:
            metric_names: Specific metrics to average (None = all)
            
        Returns:
            Dictionary of averaged metrics
        """
        if not self.results:
            return {}
        
        if metric_names is None:
            metric_names = list(self.results[0].metrics.keys())
        
        averages = {}
        for name in metric_names:
            values = [r.metrics.get(name) for r in self.results if name in r.metrics]
            if values:
                averages[name] = float(np.mean(values))
        
        return averages
    
    def std_metrics(self, metric_names: Optional[list[str]] = None) -> dict[str, float]:
        """Compute standard deviation of metrics.
        
        Args:
            metric_names: Specific metrics (None = all)
            
        Returns:
            Dictionary of standard deviations
        """
        if not self.results:
            return {}
        
        if metric_names is None:
            metric_names = list(self.results[0].metrics.keys())
        
        stds = {}
        for name in metric_names:
            values = [r.metrics.get(name) for r in self.results if name in r.metrics]
            if values:
                stds[name] = float(np.std(values))
        
        return stds
    
    def save_all(self, directory: str | Path) -> list[str]:
        """Save all results to directory.
        
        Args:
            directory: Output directory
            
        Returns:
            List of saved file paths
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        saved = []
        for r in self.results:
            path = directory / f"{r.experiment_name}.json"
            r.save(path)
            saved.append(str(path))
        
        return saved
    
    @classmethod
    def load_all(cls, directory: str | Path, pattern: str = "*.json") -> "ResultsCollection":
        """Load all results from directory.
        
        Args:
            directory: Input directory
            pattern: File pattern to match
            
        Returns:
            ResultsCollection with loaded results
        """
        directory = Path(directory)
        collection = cls()
        
        for path in directory.glob(pattern):
            try:
                result = ExperimentResults.load(path)
                collection.add(result)
            except Exception as e:
                print(f"Warning: Could not load {path}: {e}")
        
        return collection
    
    def __len__(self) -> int:
        """Return number of results."""
        return len(self.results)
    
    def __getitem__(self, idx: int) -> ExperimentResults:
        """Get result by index."""
        return self.results[idx]


def plot_results(
    results: ExperimentResults | ResultsCollection,
    metric_x: str,
    metric_y: str,
    output_path: Optional[str | Path] = None,
) -> dict:
    """Plot results using matplotlib (if available).
    
    Args:
        results: Single result or collection
        metric_x: Metric name for x-axis
        metric_y: Metric name for y-axis
        output_path: Optional path to save plot
        
    Returns:
        Plot data dictionary
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return {}
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if isinstance(results, ExperimentResults):
        # Single result - can't plot relationship without more data
        ax.scatter([results.metrics.get(metric_x, 0)], [results.metrics.get(metric_y, 0)])
    else:
        # Collection - plot all points
        x_vals = [r.metrics.get(metric_x, 0) for r in results.results if metric_x in r.metrics]
        y_vals = [r.metrics.get(metric_y, 0) for r in results.results if metric_y in r.metrics]
        ax.scatter(x_vals, y_vals)
    
    ax.set_xlabel(metric_x)
    ax.set_ylabel(metric_y)
    ax.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return {"x": metric_x, "y": metric_y}
