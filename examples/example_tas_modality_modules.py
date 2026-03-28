# SPDX-License-Identifier: Apache-2.0

"""Example usage of TAS with modality and module support.

This example demonstrates:
1. Setting up TASProfiler with different data modalities
2. Using preset modules to define architectures
3. Using TGA (Thermogeometric Activation) in custom modules
4. Profiling architectures with both legacy dict and new module list formats
"""

import numpy as np

# Set random seed for reproducibility
np.random.seed(42)


def example_tabular_modality():
    """Example with tabular data modality."""
    print("=" * 60)
    print("Example 1: Tabular Data Modality")
    print("=" * 60)
    
    from thermorg.tas import TASProfiler, TabularModality
    
    # Create profiler with tabular modality (default)
    profiler = TASProfiler()
    
    # Or explicitly set tabular modality
    profiler.set_modality('tabular', scale=True)
    
    # Generate sample tabular data
    X = np.random.randn(500, 20)  # 500 samples, 20 features
    y = np.sum(X[:, :5] ** 2, axis=1)  # Simple quadratic function
    
    # Define architecture using dict (legacy format)
    architecture = {
        'widths': [64, 128, 256, 128],
        'types': ['linear', 'linear', 'linear', 'linear'],
    }
    
    train_config = {
        'lr': 1e-3,
        'batch_size': 32,
    }
    
    result = profiler.profile(X, y, architecture, train_config)
    
    print(f"\nResults:")
    print(f"  d_manifold = {result.d_manifold:.4f}")
    print(f"  s_smoothness = {result.s_smoothness:.4f}")
    print(f"  α (scaling exponent) = {result.alpha:.4f}")
    print(f"  η_product = {result.eta_product:.4f}")
    
    return result


def example_module_list():
    """Example with module list (new format)."""
    print("\n" + "=" * 60)
    print("Example 2: Module List Format")
    print("=" * 60)
    
    from thermorg.tas import TASProfiler, ModuleRegistry
    
    profiler = TASProfiler()
    profiler.set_modality('tabular')
    
    # Define architecture using module list (new format)
    modules = [
        ModuleRegistry.linear(20, 64),      # Input: 20 features -> 64
        ModuleRegistry.linear(64, 128),     # 64 -> 128
        ModuleRegistry.linear(128, 256),    # 128 -> 256
        ModuleRegistry.linear(256, 128),    # 256 -> 128
        ModuleRegistry.linear(128, 64),      # 128 -> 64
    ]
    
    train_config = {'lr': 1e-3, 'batch_size': 32}
    
    result = profiler.profile_architecture(modules, train_config)
    
    print(f"\nArchitecture: 5-layer linear network")
    print(f"  η_ls = {[f'{eta:.4f}' for eta in result.eta_ls]}")
    print(f"  η_product = {result.eta_product:.4f}")
    print(f"  α = {result.alpha:.4f}")
    
    return result


def example_conv_network():
    """Example with convolutional modules."""
    print("\n" + "=" * 60)
    print("Example 3: Convolutional Network with Attention")
    print("=" * 60)
    
    from thermorg.tas import TASProfiler, ModuleRegistry
    
    profiler = TASProfiler()
    
    # Define CNN + Attention architecture
    modules = [
        # Conv block 1: 3 channels -> 64
        ModuleRegistry.conv2d(3, 64, kernel_size=3, stride=1, padding=1),
        ModuleRegistry.pooling('max', kernel_size=2, stride=2),  # Compress
        ModuleRegistry.conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        ModuleRegistry.pooling('max', kernel_size=2, stride=2),  # Compress
        # Attention
        ModuleRegistry.attention(num_heads=8, head_dim=64),
        # Residual wrapper
        ModuleRegistry.residual(ModuleRegistry.linear(512, 512)),
    ]
    
    train_config = {'lr': 1e-3, 'batch_size': 64}
    
    result = profiler.profile_architecture(modules, train_config)
    
    print(f"\nArchitecture: CNN + Attention")
    print(f"  Module types: {[m.module_type for m in modules]}")
    print(f"  η_ls = {[f'{eta:.4f}' for eta in result.eta_ls]}")
    print(f"  η_product = {result.eta_product:.4f}")
    print(f"  α = {result.alpha:.4f}")
    
    return result


def example_custom_module_with_tga():
    """Example with custom module using TGA activation."""
    print("\n" + "=" * 60)
    print("Example 4: Custom Module with TGA Activation")
    print("=" * 60)
    
    from thermorg.tas import TASProfiler, CustomModule, TGAActivation
    import torch
    import torch.nn as nn
    
    profiler = TASProfiler()
    
    # Create custom PyTorch module
    linear = nn.Linear(512, 256)
    
    # Wrap with custom module using TGA
    custom_mod = CustomModule(
        module=linear,
        activation='tga',
        t_adiab=0.5,  # T_adiab controls sigmoid steepness
    )
    
    print(f"\nCustom module: {custom_mod}")
    print(f"  Input dim: {custom_mod.in_features}")
    print(f"  Output dim: {custom_mod.out_features}")
    print(f"  Activation: {custom_mod.activation_name}")
    print(f"  T_adiab: {custom_mod.t_adiab}")
    
    # Compute eta
    d_prev = 512.0
    eta = custom_mod.compute_eta(d_prev)
    print(f"  η (with d_prev={d_prev}) = {eta:.4f}")
    
    # Test forward pass
    x = torch.randn(1, 512)
    out = custom_mod.forward(x)
    print(f"  Output shape: {out.shape}")
    
    return custom_mod


def example_modality_registry():
    """Example using ModalityRegistry."""
    print("\n" + "=" * 60)
    print("Example 5: Modality Registry")
    print("=" * 60)
    
    from thermorg.tas import ModalityRegistry
    
    # List available modalities
    modalities = ModalityRegistry.list_modalities()
    print("\nAvailable modalities:")
    for name, desc in modalities.items():
        print(f"  {name}: {desc}")
    
    # Create modalities
    tabular = ModalityRegistry.create_tabular(scale=True)
    print(f"\nCreated: {tabular.__class__.__name__}")
    
    # Note: Embedding modalities require additional libraries
    # Uncomment to test (requires transformers):
    # text = ModalityRegistry.create_text(encoder='bert')
    # print(f"Created: {text.__class__.__name__}, encoder={text.encoder}")


def example_module_registry():
    """Example using ModuleRegistry."""
    print("\n" + "=" * 60)
    print("Example 6: Module Registry")
    print("=" * 60)
    
    from thermorg.tas import ModuleRegistry
    
    # List available modules
    print("\nCreating modules via ModuleRegistry:")
    
    conv = ModuleRegistry.conv2d(3, 64, kernel_size=3)
    print(f"  conv2d: {conv}")
    
    attn = ModuleRegistry.attention(num_heads=12, head_dim=64)
    print(f"  attention: {attn}")
    
    pool = ModuleRegistry.pooling('max', kernel_size=2, stride=2)
    print(f"  pooling: {pool}")
    
    residual = ModuleRegistry.residual(ModuleRegistry.conv2d(64, 64, 3))
    print(f"  residual: {residual}")


def example_activation_functions():
    """Example demonstrating different activation functions."""
    print("\n" + "=" * 60)
    print("Example 7: Activation Functions Comparison")
    print("=" * 60)
    
    import torch
    from thermorg.tas import TGAActivation, GELU, Swish, get_activation
    
    # Test input
    x = torch.linspace(-3, 3, 100).reshape(-1, 1)
    
    # TGA with different T_adiab values
    tga_05 = TGAActivation(t_adiab=0.5)
    tga_10 = TGAActivation(t_adiab=1.0)
    tga_20 = TGAActivation(t_adiab=2.0)
    
    out_05 = tga_05(x).numpy().flatten()
    out_10 = tga_10(x).numpy().flatten()
    out_20 = tga_20(x).numpy().flatten()
    
    print("\nTGA activation with different T_adiab:")
    print(f"  T_adiab=0.5: output range = [{out_05.min():.3f}, {out_05.max():.3f}]")
    print(f"  T_adiab=1.0: output range = [{out_10.min():.3f}, {out_10.max():.3f}]")
    print(f"  T_adiab=2.0: output range = [{out_20.min():.3f}, {out_20.max():.3f}]")
    
    # Compare with GELU
    gelu = GELU()
    gelu_out = gelu(x).numpy().flatten()
    
    # TGA with T_adiab=1 is similar to GELU
    print(f"\nGELU: output range = [{gelu_out.min():.3f}, {gelu_out.max():.3f}]")
    
    diff = np.abs(out_10 - gelu_out).mean()
    print(f"Mean |TGA(T=1) - GELU| = {diff:.4f}")
    
    # Swish
    swish = Swish(beta=1.0)
    swish_out = swish(x).numpy().flatten()
    print(f"\nSwish(β=1): output range = [{swish_out.min():.3f}, {swish_out.max():.3f}]")


def example_compare_architectures():
    """Example comparing different architectures."""
    print("\n" + "=" * 60)
    print("Example 8: Architecture Comparison")
    print("=" * 60)
    
    from thermorg.tas import TASProfiler, ModuleRegistry
    
    profiler = TASProfiler()
    
    architectures = {
        'shallow_wide': [
            ModuleRegistry.linear(100, 1024),
            ModuleRegistry.linear(1024, 512),
        ],
        'deep_narrow': [
            ModuleRegistry.linear(100, 64),
            ModuleRegistry.linear(64, 64),
            ModuleRegistry.linear(64, 64),
            ModuleRegistry.linear(64, 64),
            ModuleRegistry.linear(64, 512),
        ],
        'bottleneck': [
            ModuleRegistry.linear(100, 256),
            ModuleRegistry.linear(256, 64),
            ModuleRegistry.linear(64, 64),
            ModuleRegistry.linear(64, 256),
            ModuleRegistry.linear(256, 512),
        ],
        'with_attention': [
            ModuleRegistry.linear(100, 512),
            ModuleRegistry.attention(num_heads=8, head_dim=64),
            ModuleRegistry.linear(512, 512),
        ],
    }
    
    train_config = {'lr': 1e-3, 'batch_size': 32}
    
    print("\nArchitecture comparison:")
    print("-" * 60)
    
    results = {}
    for name, modules in architectures.items():
        result = profiler.profile_architecture(modules, train_config)
        results[name] = result
        
        print(f"{name:20s}: α = {result.alpha:8.4f}, "
              f"η_product = {result.eta_product:8.4f}, "
              f"layers = {len(modules)}")
    
    # Find best
    best = max(results.items(), key=lambda x: x[1].alpha)
    print(f"\nBest architecture: {best[0]} (α = {best[1].alpha:.4f})")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("TAS MODULE AND MODALITY SUPPORT EXAMPLES")
    print("=" * 60)
    
    # Run examples
    example_tabular_modality()
    example_module_list()
    example_conv_network()
    example_custom_module_with_tga()
    example_modality_registry()
    example_module_registry()
    example_activation_functions()
    example_compare_architectures()
    
    print("\n" + "=" * 60)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 60)


if __name__ == '__main__':
    main()
