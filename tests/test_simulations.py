# SPDX-License-Identifier: Apache-2.0

"""Tests for simulation modules.

Tests data generators, tasks, and network architectures.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from thermorg.simulations.manifold_data import ManifoldDataGenerator
from thermorg.simulations.algorithmic_tasks import (
    AlgorithmicTaskDataset,
    create_mod_task,
    create_parity_task,
    create_reverse_task,
)
from thermorg.simulations.regression_tasks import (
    RegressionTaskDataset,
    create_polynomial_task,
    create_trigonometric_task,
    create_mixed_task,
)
from thermorg.simulations.networks import (
    LinearNetwork,
    MLP,
    SSMNetwork,
    RNNNetwork,
)


class TestManifoldDataGenerator:
    """Tests for ManifoldDataGenerator."""
    
    def test_linear_embedding(self):
        """Test linear embedding generation."""
        generator = ManifoldDataGenerator(seed=42)
        z, x = generator.generate(n_samples=100, d_manifold=8, d_embed=64, mode="linear")
        
        assert z.shape == (100, 8)
        assert x.shape == (100, 64)
        assert z.min() >= -1 and z.max() <= 1
    
    def test_nonlinear_embedding(self):
        """Test nonlinear embedding generation."""
        generator = ManifoldDataGenerator(seed=42)
        z, x = generator.generate(n_samples=50, d_manifold=4, d_embed=32, mode="nonlinear")
        
        assert z.shape == (50, 4)
        assert x.shape == (50, 32)
    
    def test_polynomial_embedding(self):
        """Test polynomial embedding generation."""
        generator = ManifoldDataGenerator(seed=42)
        z, x = generator.generate(n_samples=50, d_manifold=4, d_embed=32, mode="polynomial")
        
        assert z.shape == (50, 4)
        assert x.shape == (50, 32)
    
    def test_reproducibility(self):
        """Test that same seed gives same results."""
        gen1 = ManifoldDataGenerator(seed=123)
        gen2 = ManifoldDataGenerator(seed=123)
        
        z1, x1 = gen1.generate(n_samples=100, d_manifold=8, d_embed=32)
        z2, x2 = gen2.generate(n_samples=100, d_manifold=8, d_embed=32)
        
        assert torch.allclose(z1, z2)
        assert torch.allclose(x1, x2)
    
    def test_reset(self):
        """Test reset clears cached parameters."""
        generator = ManifoldDataGenerator(seed=42)
        generator.generate(n_samples=100, d_manifold=8, d_embed=32, mode="nonlinear")
        
        assert generator._nonlinear_net is not None
        generator.reset()
        assert generator._nonlinear_net is None


class TestAlgorithmicTaskDataset:
    """Tests for AlgorithmicTaskDataset."""
    
    def test_mod_task(self):
        """Test modular arithmetic task."""
        train, test = create_mod_task(p=7, train_size=1000, test_size=200)
        
        assert len(train) == 1000
        assert len(test) == 1000
        assert train.input_dim == 14  # p * 2 for x and y
        assert train.output_dim == 7
        
        x, y = train[0]
        assert x.shape[0] == 14
        assert y.shape[0] == 7
    
    def test_parity_task(self):
        """Test parity detection task."""
        train, test = create_parity_task(train_size=1000, test_size=200)
        
        assert len(train) == 1000
        assert train.input_dim == 8
        assert train.output_dim == 2
        
        x, y = train[0]
        assert x.shape == (8,)
        assert y.shape == (2,)
    
    def test_reverse_task(self):
        """Test sequence reversal task."""
        train, test = create_reverse_task(seq_len=10, train_size=500, test_size=100)
        
        assert len(train) == 500
        assert train.input_dim == 10
        assert train.output_dim == 10
        
        x, y = train[0]
        assert x.shape == (10,)
        assert y.shape == (10,)
    
    def test_mod_explicit_p(self):
        """Test mod task with explicit p parameter."""
        dataset = AlgorithmicTaskDataset(task="mod", size=100, p=11, seed=42)
        
        assert len(dataset) == 100
        assert dataset.p == 11


class TestRegressionTaskDataset:
    """Tests for RegressionTaskDataset."""
    
    def test_polynomial_task(self):
        """Test polynomial regression task."""
        train, test = create_polynomial_task(n_dims=4, train_size=500, noise_std=0.0)
        
        assert len(train) == 500
        assert train.input_dim == 4
        assert train.output_dim == 1
        
        x, y = train[0]
        assert x.shape == (4,)
        assert y.dim() == 0  # Scalar target
    
    def test_trigonometric_task(self):
        """Test trigonometric regression task."""
        train, test = create_trigonometric_task(n_dims=3, train_size=200, noise_std=0.0)
        
        assert len(train) == 200
        assert train.input_dim == 3
        
        x, y = train[0]
        assert x.shape == (3,)
    
    def test_mixed_task(self):
        """Test mixed regression task."""
        train, test = create_mixed_task(n_dims=5, train_size=300, noise_std=0.1)
        
        assert len(train) == 300
        assert train.input_dim == 5
    
    def test_noisy_targets(self):
        """Test that noise is properly added to targets."""
        dataset_noiseless = RegressionTaskDataset(
            task="polynomial", n_dims=4, size=100, noise_std=0.0, seed=42
        )
        dataset_noisy = RegressionTaskDataset(
            task="polynomial", n_dims=4, size=100, noise_std=0.5, seed=42
        )
        
        # Same seed should give same x values
        assert torch.allclose(dataset_noiseless.x, dataset_noisy.x)
        
        # Noiseless targets should be deterministic
        y1 = dataset_noiseless.y[0]
        y2 = dataset_noiseless.y[0]
        assert torch.allclose(y1, y2)


class TestLinearNetwork:
    """Tests for LinearNetwork."""
    
    def test_initialization(self):
        """Test network initialization."""
        net = LinearNetwork(input_dim=64, hidden_dim=128, output_dim=10, n_layers=3)
        
        assert len(net.layers) == 4  # n_layers hidden + 1 output
        assert net.hidden_dim == 128
    
    def test_forward(self):
        """Test forward pass."""
        net = LinearNetwork(input_dim=64, hidden_dim=128, output_dim=10)
        x = torch.randn(32, 64)
        y = net(x)
        
        assert y.shape == (32, 10)
    
    def test_weight_matrices(self):
        """Test weight matrix access."""
        net = LinearNetwork(input_dim=64, hidden_dim=128, output_dim=10, n_layers=2)
        weights = net.weight_matrices
        
        assert len(weights) == 3  # 2 hidden + 1 output
        assert weights[0].shape == (128, 64)
    
    def test_layer_outputs(self):
        """Test getting intermediate layer outputs."""
        net = LinearNetwork(input_dim=64, hidden_dim=128, output_dim=10, n_layers=2)
        x = torch.randn(16, 64)
        outputs = net.get_all_layer_outputs(x)
        
        assert len(outputs) == 3
        assert outputs[-1].shape == (16, 10)


class TestMLP:
    """Tests for MLP."""
    
    def test_initialization(self):
        """Test MLP initialization with different activations."""
        net = MLP(input_dim=64, hidden_dim=128, output_dim=10, activation="relu")
        assert net._activation_name == "relu"
        
        net_tanh = MLP(input_dim=64, hidden_dim=128, output_dim=10, activation="tanh")
        assert net_tanh._activation_name == "tanh"
    
    def test_forward(self):
        """Test MLP forward pass."""
        net = MLP(input_dim=64, hidden_dim=128, output_dim=10, n_layers=3)
        x = torch.randn(32, 64)
        y = net(x)
        
        assert y.shape == (32, 10)
    
    def test_none_activation(self):
        """Test MLP with no activation (linear network behavior)."""
        net = MLP(input_dim=64, hidden_dim=128, output_dim=10, activation="none")
        x = torch.randn(32, 64)
        y = net(x)
        
        assert y.shape == (32, 10)


class TestSSMNetwork:
    """Tests for SSMNetwork."""
    
    def test_initialization(self):
        """Test SSM initialization."""
        net = SSMNetwork(input_dim=64, state_dim=32, output_dim=10)
        
        assert net.input_dim == 64
        assert net.state_dim == 32
        assert net.output_dim == 10
    
    def test_forward_vector(self):
        """Test SSM with vector input (no sequence)."""
        net = SSMNetwork(input_dim=64, state_dim=32, output_dim=10)
        x = torch.randn(32, 64)
        y = net(x)
        
        assert y.shape == (32, 10)
    
    def test_forward_sequence(self):
        """Test SSM with sequence input."""
        net = SSMNetwork(input_dim=64, state_dim=32, output_dim=10)
        x = torch.randn(32, 10, 64)  # (batch, seq, dim)
        y = net(x)
        
        assert y.shape == (32, 10, 10)
    
    def test_trace_modes(self):
        """Test different trace modes."""
        for mode in ["zero", "negative", "positive", "free"]:
            net = SSMNetwork(input_dim=32, state_dim=16, output_dim=8, trace_mode=mode)
            
            trace = net.get_A_trace()
            assert isinstance(trace, torch.Tensor)
    
    def test_spectral_radius(self):
        """Test spectral radius computation."""
        net = SSMNetwork(input_dim=32, state_dim=16, output_dim=8)
        radius = net.get_spectral_radius()
        
        assert isinstance(radius, torch.Tensor)
        assert radius >= 0


class TestRNNNetwork:
    """Tests for RNNNetwork."""
    
    def test_initialization(self):
        """Test RNN initialization."""
        net = RNNNetwork(input_dim=64, hidden_dim=128, output_dim=10, n_layers=2)
        
        assert net.input_dim == 64
        assert net.hidden_dim == 128
        assert net.output_dim == 10
        assert net.n_layers == 2
    
    def test_forward(self):
        """Test RNN forward pass."""
        net = RNNNetwork(input_dim=64, hidden_dim=128, output_dim=10)
        x = torch.randn(32, 15, 64)  # (batch, seq_len, input_dim)
        y, h = net(x)
        
        assert y.shape == (32, 15, 10)
        assert h.shape == (2, 32, 128)  # n_layers=2, bidirectional=False
    
    def test_single_step(self):
        """Test single time step forward."""
        net = RNNNetwork(input_dim=64, hidden_dim=128, output_dim=10)
        x = torch.randn(32, 64)
        y, h = net.forward_single_step(x)
        
        assert y.shape == (32, 10)
        assert h.shape == (1, 32, 128)
    
    def test_init_hidden(self):
        """Test hidden state initialization."""
        net = RNNNetwork(input_dim=64, hidden_dim=128, output_dim=10)
        h = net.init_hidden(batch_size=16)
        
        assert h.shape == (1, 16, 128)
        assert torch.allclose(h, torch.zeros_like(h))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
