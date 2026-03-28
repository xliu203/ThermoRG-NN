# SPDX-License-Identifier: Apache-2.0

"""Tests for experiment modules.

Tests scaling, thermodynamic, and SSM experiments.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from thermorg.experiments.scaling_experiment import ScalingExperiment, ScalingLawResult
from thermorg.experiments.thermodynamic_experiment import ThermodynamicExperiment, ThermodynamicResult
from thermorg.experiments.ssm_experiment import SSMExperiment, SSMExperimentResult
from thermorg.analysis.results import ExperimentResults, ResultsCollection
from thermorg.simulations.networks import LinearNetwork, MLP


class TestScalingExperiment:
    """Tests for ScalingExperiment."""
    
    @pytest.fixture
    def experiment(self):
        """Create experiment fixture."""
        return ScalingExperiment(
            d_manifold_range=[4, 8],
            d_embed=32,
            n_train=100,
            n_test=50,
            seed=42,
        )
    
    def test_initialization(self, experiment):
        """Test experiment initialization."""
        assert experiment.d_manifold_range == [4, 8]
        assert experiment.d_embed == 32
        assert experiment.n_train == 100
    
    def test_run_linear_network(self, experiment, tmp_path):
        """Test running scaling experiment with linear network."""
        def network_factory(input_dim, output_dim):
            return LinearNetwork(
                input_dim=input_dim,
                hidden_dim=64,
                output_dim=output_dim,
                n_layers=2,
            )
        
        results = experiment.run(
            network_factory=network_factory,
            epochs=10,
            batch_size=32,
            verbose=False,
        )
        
        assert isinstance(results, ScalingLawResult)
        assert len(results.manifold_dims) == 2
        assert len(results.effective_dims) == 2
        assert results.alpha > 0  # Scaling exponent should be positive
    
    def test_save_load_results(self, experiment, tmp_path):
        """Test saving and loading results."""
        result = ScalingLawResult(
            manifold_dims=[4.0, 8.0],
            effective_dims=[2.5, 4.2],
            compression_effs=[0.6, 0.5],
            errors=[0.1, 0.05],
            alpha=0.67,
            r_squared=0.95,
        )
        
        save_path = tmp_path / "scaling_test.json"
        experiment.save_results(result, save_path)
        
        loaded = ScalingExperiment.load_results(save_path)
        
        assert loaded.manifold_dims == result.manifold_dims
        assert loaded.alpha == result.alpha
        assert loaded.r_squared == result.r_squared


class TestThermodynamicExperiment:
    """Tests for ThermodynamicExperiment."""
    
    @pytest.fixture
    def experiment(self):
        """Create experiment fixture."""
        return ThermodynamicExperiment(
            temperature_range=[0.5, 1.0, 1.5],
            seed=42,
        )
    
    def test_initialization(self, experiment):
        """Test experiment initialization."""
        assert len(experiment.temperatures) == 3
        assert 0.5 in experiment.temperatures
    
    def test_run_with_temperature(self, experiment):
        """Test temperature sweep experiment."""
        # Create simple data
        n = 100
        x_train = torch.randn(n, 32)
        y_train = torch.randn(n, 10)
        x_test = torch.randn(50, 32)
        y_test = torch.randn(50, 10)
        
        # Simple model factory
        model = LinearNetwork(input_dim=32, hidden_dim=64, output_dim=10, n_layers=2)
        
        results = experiment.run_with_temperature(
            model=model,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            manifold_dim=8.0,
            epochs_per_temp=5,
            verbose=False,
        )
        
        assert isinstance(results, ThermodynamicResult)
        assert len(results.temperatures) == 3
        assert results.t_c_estimated > 0
    
    def test_transition_width_estimation(self, experiment):
        """Test transition width estimation."""
        # Synthetic compression efficiency curve
        compression_effs = [0.2, 0.4, 0.6, 0.8, 0.9]
        temperatures = [0.5, 0.75, 1.0, 1.25, 1.5]
        
        width = experiment._estimate_transition_width(compression_effs, temperatures, fraction=0.5)
        
        assert width > 0


class TestSSMExperiment:
    """Tests for SSMExperiment."""
    
    @pytest.fixture
    def experiment(self):
        """Create experiment fixture."""
        return SSMExperiment(
            trace_modes=["zero", "negative", "positive"],
            target_traces=[0.0, -1.0, 1.0],
            seed=42,
        )
    
    def test_initialization(self, experiment):
        """Test experiment initialization."""
        assert len(experiment.trace_modes) == 3
        assert experiment.trace_modes == ["zero", "negative", "positive"]
    
    def test_run(self, experiment):
        """Test SSM trace comparison experiment."""
        # Create simple data
        n = 100
        x_train = torch.randn(n, 32)
        y_train = torch.randn(n, 10)
        x_test = torch.randn(50, 32)
        y_test = torch.randn(50, 10)
        
        results = experiment.run(
            input_dim=32,
            state_dim=16,
            output_dim=10,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            manifold_dim=8.0,
            epochs=10,
            verbose=False,
        )
        
        assert isinstance(results, SSMExperimentResult)
        assert len(results.trace_modes) == 3
        assert all(tr in results.traces for tr in [0.0, -1.0, 1.0])
        assert results.critical_difference >= 0
    
    def test_critical_difference(self, experiment):
        """Test critical difference computation."""
        result = experiment._compute_critical_difference(
            trace_modes=["zero", "negative"],
            compression_effs=[0.8, 0.4],
            test_errors=[0.1, 0.2],
        )
        
        assert result >= 0


class TestExperimentResults:
    """Tests for ExperimentResults."""
    
    def test_creation(self):
        """Test creating experiment results."""
        results = ExperimentResults(
            experiment_name="test_exp",
            config={"epochs": 100, "lr": 0.001},
            metrics={"accuracy": 0.95, "loss": 0.05},
        )
        
        assert results.experiment_name == "test_exp"
        assert results.metrics["accuracy"] == 0.95
    
    def test_add_metric(self):
        """Test adding metrics."""
        results = ExperimentResults(
            experiment_name="test",
            config={},
            metrics={},
        )
        
        results.add_metric("new_metric", 0.99)
        assert results.metrics["new_metric"] == 0.99
    
    def test_add_artifact(self):
        """Test adding artifacts."""
        results = ExperimentResults(
            experiment_name="test",
            config={},
            metrics={},
        )
        
        results.add_artifact("model", "/path/to/model.pt")
        assert results.artifacts["model"] == "/path/to/model.pt"
    
    def test_save_load(self, tmp_path):
        """Test saving and loading."""
        results = ExperimentResults(
            experiment_name="save_test",
            config={"test": True},
            metrics={"val": 1.23},
            tags=["test"],
        )
        
        save_path = tmp_path / "test_results.json"
        results.save(save_path)
        
        loaded = ExperimentResults.load(save_path)
        
        assert loaded.experiment_name == results.experiment_name
        assert loaded.metrics == results.metrics
    
    def test_summary(self):
        """Test summary generation."""
        results = ExperimentResults(
            experiment_name="summary_test",
            config={"epochs": 50},
            metrics={"acc": 0.9},
            tags=["quick"],
        )
        
        summary = results.summary()
        
        assert "summary_test" in summary
        assert "epochs" in summary
        assert "0.9" in summary


class TestResultsCollection:
    """Tests for ResultsCollection."""
    
    def test_add_and_length(self):
        """Test adding results and checking length."""
        collection = ResultsCollection()
        
        for i in range(3):
            results = ExperimentResults(
                experiment_name=f"exp_{i}",
                config={},
                metrics={"val": i * 0.1},
            )
            collection.add(results)
        
        assert len(collection) == 3
    
    def test_filter_by_tag(self):
        """Test filtering by tag."""
        collection = ResultsCollection()
        
        for i in range(4):
            results = ExperimentResults(
                experiment_name=f"exp_{i}",
                config={},
                metrics={"val": i * 0.1},
                tags=["group_a"] if i < 2 else ["group_b"],
            )
            collection.add(results)
        
        filtered = collection.filter(tag="group_a")
        
        assert len(filtered) == 2
    
    def test_average_metrics(self):
        """Test averaging metrics."""
        collection = ResultsCollection()
        
        for i in range(3):
            results = ExperimentResults(
                experiment_name=f"exp_{i}",
                config={},
                metrics={"val": 1.0 + i * 0.1, "other": 5.0},
            )
            collection.add(results)
        
        averages = collection.average_metrics()
        
        assert abs(averages["val"] - 1.1) < 0.01
        assert averages["other"] == 5.0
    
    def test_save_and_load_all(self, tmp_path):
        """Test saving and loading all results."""
        collection = ResultsCollection()
        
        for i in range(2):
            results = ExperimentResults(
                experiment_name=f"multi_{i}",
                config={"i": i},
                metrics={"val": float(i)},
            )
            collection.add(results)
        
        saved_paths = collection.save_all(tmp_path)
        assert len(saved_paths) == 2
        
        loaded_collection = ResultsCollection.load_all(tmp_path)
        assert len(loaded_collection) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
