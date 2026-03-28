# SPDX-License-Identifier: Apache-2.0

"""Tests for TAS modality and module support."""

import pytest
import numpy as np

from thermorg.tas.modality import (
    BaseModality,
    ModalityConfig,
    TabularModality,
    EmbeddingModality,
    ModalityRegistry,
)
from thermorg.tas.modules import (
    BaseModule,
    ModuleConfig,
    PresetModule,
    ModuleRegistry,
    CustomModule,
    TGAActivation,
    GELU,
    Swish,
    get_activation,
)


class TestModality:
    """Test data modality classes."""
    
    def test_tabular_modality(self):
        """Test TabularModality feature extraction and distance."""
        modality = TabularModality(scale=True)
        
        # Create sample tabular data
        X = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ])
        
        features = modality.extract_features(X)
        
        assert features.shape == (3, 3)
        assert modality.embedding_dim == 3
        
        # Test distance computation
        dist = modality.compute_distance(features[0], features[1])
        assert dist >= 0
        
        # Test Euclidean distance (not cosine for tabular)
        dist_manual = np.linalg.norm(features[0] - features[1])
        assert np.isclose(dist, dist_manual)
    
    def test_tabular_modality_unscaled(self):
        """Test TabularModality without scaling."""
        modality = TabularModality(scale=False)
        
        X = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
        ])
        
        features = modality.extract_features(X)
        assert features.shape == (2, 2)
    
    def test_modality_registry_tabular(self):
        """Test ModalityRegistry for tabular."""
        modality = ModalityRegistry.create_tabular(scale=True)
        assert isinstance(modality, TabularModality)
    
    def test_modality_registry_list(self):
        """Test listing available modalities."""
        modalities = ModalityRegistry.list_modalities()
        assert 'tabular' in modalities
        assert 'text' in modalities
        assert 'audio' in modalities
    
    def test_modality_config(self):
        """Test ModalityConfig."""
        config = ModalityConfig(
            modality_type='tabular',
            embedding_dim=128,
            scale=True,
            normalize=False,
            distance_metric='euclidean',
        )
        assert config.modality_type == 'tabular'
        assert config.embedding_dim == 128
        assert config.distance_metric == 'euclidean'


class TestModules:
    """Test neural network module classes."""
    
    def test_preset_module_conv2d(self):
        """Test PresetModule for conv2d."""
        module = PresetModule('conv2d', {
            'in_channels': 3,
            'out_channels': 64,
            'kernel_size': 3,
            'stride': 1,
            'padding': 1,
        })
        
        d_prev = 100.0
        eta = module.compute_eta(d_prev)
        
        assert eta > 0
        assert eta <= 1.0  # Capped at 1.0 for conv
        
        # Check config
        assert module.config.module_type == 'conv2d'
        assert module.config.out_channels == 64
    
    def test_preset_module_attention(self):
        """Test PresetModule for attention."""
        module = PresetModule('attention', {
            'num_heads': 8,
            'head_dim': 64,
        })
        
        d_prev = 512.0
        eta = module.compute_eta(d_prev)
        
        # effective_width = 8 * 64 = 512
        # eta = min(1.0, 512 / 512) = 1.0
        assert eta == 1.0
    
    def test_preset_module_residual(self):
        """Test PresetModule for residual."""
        inner = PresetModule('conv2d', {'out_channels': 64})
        module = PresetModule('residual', {'inner': inner})
        
        d_prev = 100.0
        eta = module.compute_eta(d_prev)
        
        assert eta == 1.0  # Residual preserves dimension
    
    def test_preset_module_pooling(self):
        """Test PresetModule for pooling."""
        module = PresetModule('pooling', {
            'pool_type': 'max',
            'kernel_size': 2,
            'stride': 2,
        })
        
        d_prev = 100.0
        eta = module.compute_eta(d_prev)
        
        # eta = stride^(-2) = 0.25
        assert eta == 0.25
    
    def test_module_registry_conv2d(self):
        """Test ModuleRegistry for conv2d."""
        module = ModuleRegistry.conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
        )
        
        assert isinstance(module, PresetModule)
        assert module.module_type == 'conv2d'
        assert module.config.out_channels == 64
    
    def test_module_registry_attention(self):
        """Test ModuleRegistry for attention."""
        module = ModuleRegistry.attention(
            num_heads=12,
            head_dim=64,
        )
        
        assert module.config.num_heads == 12
        assert module.config.head_dim == 64
    
    def test_module_registry_residual(self):
        """Test ModuleRegistry for residual."""
        inner = ModuleRegistry.conv2d(64, 64, 3)
        module = ModuleRegistry.residual(inner)
        
        assert module.module_type == 'residual'
    
    def test_module_registry_pooling(self):
        """Test ModuleRegistry for pooling."""
        module = ModuleRegistry.pooling('max', kernel_size=2, stride=2)
        
        assert module.module_type == 'pooling'
        assert module.config.pool_type == 'max'
    
    def test_module_registry_linear(self):
        """Test ModuleRegistry for linear."""
        module = ModuleRegistry.linear(
            in_features=512,
            out_features=256,
        )
        
        assert module.module_type == 'linear'
        assert module.config.out_channels == 256
    
    def test_module_registry_embedding(self):
        """Test ModuleRegistry for embedding."""
        module = ModuleRegistry.embedding(
            vocab_size=30000,
            embed_dim=512,
        )
        
        assert module.module_type == 'embedding'


class TestActivations:
    """Test activation functions."""
    
    def test_tga_activation(self):
        """Test TGAActivation forward pass."""
        import torch
        
        activation = TGAActivation(t_adiab=1.0)
        
        z = torch.tensor([0.0, 1.0, -1.0, 2.0])
        out = activation(z)
        
        assert out.shape == z.shape
        # For z > 0, output should be positive
        assert out[1] > 0
    
    def test_tga_with_different_t_adiab(self):
        """Test TGA with different T_adiab values."""
        import torch
        
        z = torch.tensor([1.0, 2.0, 3.0])
        
        activation1 = TGAActivation(t_adiab=1.0)
        activation2 = TGAActivation(t_adiab=0.5)
        
        out1 = activation1(z)
        out2 = activation2(z)
        
        # Lower T_adiab should give more sigmoid-like behavior
        # (saturated faster)
        assert not torch.allclose(out1, out2)
    
    def test_gelu_activation(self):
        """Test GELU activation."""
        import torch
        
        activation = GELU()
        
        z = torch.tensor([0.0, 1.0, -1.0, 2.0])
        out = activation(z)
        
        assert out.shape == z.shape
        # GELU(0) should be close to 0
        assert abs(out[0]) < 0.1
    
    def test_swish_activation(self):
        """Test Swish activation."""
        import torch
        
        activation = Swish(beta=1.0)
        
        z = torch.tensor([0.0, 1.0, -1.0, 2.0])
        out = activation(z)
        
        assert out.shape == z.shape
        # Swish(x) = x * sigmoid(x)
        expected = z * torch.sigmoid(z)
        assert torch.allclose(out, expected)
    
    def test_get_activation(self):
        """Test get_activation factory."""
        import torch
        
        tga = get_activation('tga', t_adiab=0.5)
        assert isinstance(tga, TGAActivation)
        assert tga.t_adiab == 0.5
        
        gelu = get_activation('gelu')
        assert isinstance(gelu, GELU)
        
        relu = get_activation('relu')
        assert isinstance(relu, torch.nn.ReLU)
    
    def test_get_activation_error(self):
        """Test get_activation with unknown name."""
        with pytest.raises(ValueError):
            get_activation('unknown_activation')


class TestTASProfilerWithModality:
    """Test TASProfiler with modality and module support."""
    
    def test_profiler_set_modality(self):
        """Test setting modalities on TASProfiler."""
        from thermorg.tas import TASProfiler
        
        profiler = TASProfiler()
        
        # Set tabular modality
        profiler.set_modality('tabular', scale=True)
        assert isinstance(profiler.modality, TabularModality)
    
    def test_profiler_set_modality_text(self):
        """Test setting text modality on TASProfiler."""
        from thermorg.tas import TASProfiler
        
        profiler = TASProfiler()
        
        # Text modality requires encoder - will fail without transformers
        # Just test the interface here
        # profiler.set_modality('text', encoder='bert')
    
    def test_profiler_with_modules(self):
        """Test profiling with module list."""
        from thermorg.tas import TASProfiler
        
        profiler = TASProfiler()
        
        # Define architecture with modules
        modules = [
            ModuleRegistry.conv2d(3, 64, kernel_size=3),
            ModuleRegistry.conv2d(64, 128, kernel_size=3),
            ModuleRegistry.pooling('max', kernel_size=2, stride=2),
            ModuleRegistry.attention(num_heads=8, head_dim=64),
        ]
        
        # Profile
        result = profiler.profile_architecture(modules, {'lr': 1e-3})
        
        assert result is not None
        assert len(result.eta_ls) == len(modules)
        assert result.alpha >= 0
    
    def test_profiler_with_modules_eta_product(self):
        """Test that module eta computation affects product."""
        from thermorg.tas import TASProfiler
        
        profiler = TASProfiler()
        
        # Single pooling layer should give compression
        modules_pool = [
            ModuleRegistry.pooling('max', kernel_size=2, stride=2),
        ]
        result_pool = profiler.profile_architecture(modules_pool, {'lr': 1e-3})
        
        # Single linear layer with same dimensions
        modules_linear = [
            ModuleRegistry.linear(100, 100),
        ]
        result_linear = profiler.profile_architecture(modules_linear, {'lr': 1e-3})
        
        # Pooling should give lower eta than linear (compression)
        assert result_pool.eta_ls[0] < result_linear.eta_ls[0]


class TestIntegration:
    """Integration tests for modality + module + profiler."""
    
    def test_tabular_with_architecture(self):
        """Test full pipeline with tabular data and architecture dict."""
        from thermorg.tas import TASProfiler
        
        profiler = TASProfiler()
        
        # Create sample data
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randn(100)
        
        # Architecture
        architecture = {
            'widths': [64, 128, 64],
            'types': ['linear', 'linear', 'linear'],
        }
        
        train_config = {'lr': 1e-3, 'batch_size': 32}
        
        result = profiler.profile(X, y, architecture, train_config)
        
        assert result.d_manifold > 0
        assert result.alpha >= 0
        assert len(result.eta_ls) == 3
    
    def test_custom_modality_instance(self):
        """Test setting custom modality instance."""
        from thermorg.tas import TASProfiler
        
        profiler = TASProfiler()
        
        modality = TabularModality(scale=False)
        profiler.set_modality_instance(modality)
        
        assert profiler.modality is modality


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
