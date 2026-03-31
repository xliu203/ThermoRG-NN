"""All 15 architectures for CIFAR-10 Lift Test.

This module implements the architectures defined in constants.py:
- Group 1: Thermogeometric Optimal (4 architectures)
- Group 2: Topology Destroyer (4 architectures with bottlenecks)
- Group 3: Thermal Boiling Furnace (4 architectures - ReLU ablation)
- Group 4: Traditional Baselines (3 architectures)

Each architecture accepts (batch, 3, 32, 32) input and outputs
logits of shape (batch, 10) for CIFAR-10 classification.

All ThermoNet/ThermoBot models use:
- Conv layers with kernel_size=3, padding=1 (preserves spatial dimensions)
- Skip connections with 1x1 conv when channel dimensions differ
- LayerNorm after each block (except ReLUFurnace)
- GELU or TGA activation (except ReLUFurnace uses ReLU)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from thermorg.tas.modules.activations import TGAActivation
from .constants import ALL_SPECS, ARCH_MAP, CIFAR10_NUM_CLASSES


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_flops(model: nn.Module, input_size: tuple = (3, 32, 32)) -> int:
    """Estimate FLOPs for a model using convolution FLOP formula.
    
    For Conv2d: 2 * output_size * kernel_size^2 * in_channels * out_channels
    For Linear: 2 * in_features * out_features
    
    Note: This is an approximation as it doesn't account for all operations.
    """
    total_flops = 0
    batch_size = 1
    
    def _conv_flops(kernel, in_ch, out_ch, H, W):
        return 2 * H * W * kernel * kernel * in_ch * out_ch
    
    x = torch.zeros(batch_size, *input_size)
    hooks = []
    
    def hook_fn(module, input, output):
        if isinstance(module, nn.Conv2d):
            out_h, out_w = output.shape[2], output.shape[3]
            total_flops += _conv_flops(
                module.kernel_size[0],
                module.in_channels,
                module.out_channels,
                out_h, out_w
            ) * batch_size
        elif isinstance(module, nn.Linear):
            total_flops += 2 * module.in_features * module.out_features * batch_size
    
    hooks = [model.apply(hook_fn) for _ in range(1)]
    
    # Simple FLOP estimation based on conv operations
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and 'downsample' not in name:
            # Estimate output size (assuming same padding)
            if hasattr(module, 'computed_padding'):
                pad = module.computed_padding
            else:
                pad = module.padding
            h = (x.shape[1] + 2 * pad[0] - module.kernel_size[0]) // module.stride[0] + 1
            w = (x.shape[2] + 2 * pad[1] - module.kernel_size[1]) // module.stride[1] + 1
            total_flops += _conv_flops(
                module.kernel_size[0],
                module.in_channels,
                module.out_channels,
                h, w
            ) * batch_size
    
    return total_flops


# =============================================================================
# Base Building Blocks
# =============================================================================

class ConvBlock(nn.Module):
    """Basic convolutional block with conv + norm + activation.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        activation: Activation type ('gelu', 'tga', 'relu')
        use_norm: Whether to use LayerNorm
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = 'gelu',
        use_norm: bool = True
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, padding=1, bias=not use_norm
        )
        self.norm = nn.LayerNorm([out_channels, 32, 32]) if use_norm else nn.Identity()
        
        if activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'tga':
            self.act = TGAActivation(t_adiab=1.0)
        elif activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class BottleneckBlock(nn.Module):
    """Bottleneck block with 1x1 compression, 3x3 conv, 1x1 expansion.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        bottleneck_dim: Compression dimension (typically in_channels // 8)
        activation: Activation type
        use_norm: Whether to use LayerNorm
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bottleneck_dim: int,
        activation: str = 'gelu',
        use_norm: bool = True
    ):
        super().__init__()
        # Compress
        self.compress = nn.Conv2d(in_channels, bottleneck_dim, kernel_size=1, bias=not use_norm)
        self.norm1 = nn.LayerNorm([bottleneck_dim, 32, 32]) if use_norm else nn.Identity()
        
        # 3x3 conv (main)
        self.conv = nn.Conv2d(bottleneck_dim, bottleneck_dim, kernel_size=3, padding=1, bias=not use_norm)
        self.norm2 = nn.LayerNorm([bottleneck_dim, 32, 32]) if use_norm else nn.Identity()
        
        # Expand
        self.expand = nn.Conv2d(bottleneck_dim, out_channels, kernel_size=1, bias=not use_norm)
        self.norm3 = nn.LayerNorm([out_channels, 32, 32]) if use_norm else nn.Identity()
        
        if activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'tga':
            self.act = TGAActivation(t_adiab=1.0)
        elif activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.compress(x)
        x = self.norm1(x)
        x = self.act(x)
        
        x = self.conv(x)
        x = self.norm2(x)
        x = self.act(x)
        
        x = self.expand(x)
        x = self.norm3(x)
        x = self.act(x)
        return x


class SkipConnection(nn.Module):
    """Skip connection with 1x1 conv projection when dimensions differ.
    
    Handles both channel mismatch AND spatial downsampling (stride > 1).
    The projection is always applied to the RESIDUAL to match x's dimensions.
    
    Args:
        in_channels: Number of channels in the residual (from earlier layer)
        out_channels: Number of channels in x (current layer output)
        stride: Stride of current layer (for spatial downsampling)
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        if in_channels != out_channels or stride != 1:
            # Project residual to match x's channel and spatial dimensions
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return x + self.skip(residual)


# =============================================================================
# Group 1: Thermogeometric Optimal Architectures
# =============================================================================

class ThermoNet3(nn.Module):
    """G1-1: Thermogeometric Optimal 3-layer network.
    
    Architecture: [64, 64, 128, 128]
    - 4 conv blocks with skip connections
    - GELU activation
    - LayerNorm after each block
    
    Input: (batch, 3, 32, 32)
    Output: (batch, 10)
    """
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        channels = [3, 64, 64, 128, 128]
        
        self.blocks = nn.ModuleList()
        self.skip_ops = nn.ModuleList()
        
        for i in range(len(channels) - 1):
            self.blocks.append(ConvBlock(
                channels[i], channels[i + 1],
                activation='gelu', use_norm=True
            ))
            if i > 0:  # Skip after first block
                self.skip_ops.append(SkipConnection(channels[i], channels[i + 1]))
            else:
                self.skip_ops.append(None)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = None
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i > 0 and self.skip_ops[i] is not None:
                x = self.skip_ops[i](x, residual)
            residual = x.detach()
        
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ThermoNet5(nn.Module):
    """G1-2: Thermogeometric Optimal 5-layer network.
    
    Architecture: [64, 128, 256, 128, 64]
    - 5 conv blocks with skip connections
    - GELU activation
    - LayerNorm after each block
    
    Design note: Widening-then-narrowing channel profile (矮宽型 / wide-then-narrow).
    Peak at 256ch. Compare with ThermoNet-9 which uses uniform 64ch (瘦长型).
    
    Input: (batch, 3, 32, 32)
    Output: (batch, 10)
    """
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        channels = [3, 64, 128, 256, 128, 64]
        
        self.blocks = nn.ModuleList()
        self.skip_ops = nn.ModuleList()
        
        for i in range(len(channels) - 1):
            self.blocks.append(ConvBlock(
                channels[i], channels[i + 1],
                activation='gelu', use_norm=True
            ))
            if i > 0:
                self.skip_ops.append(SkipConnection(channels[i], channels[i + 1]))
            else:
                self.skip_ops.append(None)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = None
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i > 0 and self.skip_ops[i] is not None:
                x = self.skip_ops[i](x, residual)
            residual = x.detach()
        
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ThermoNet7(nn.Module):
    """G1-3: Thermogeometric Optimal 7-layer network.
    
    Architecture: [64, 64, 128, 128, 256, 128, 64]
    - 7 conv blocks with skip connections
    - TGA activation
    - LayerNorm after each block
    
    Design note: Widening-then-narrowing channel profile (矮宽型 / wide-then-narrow).
    Peak at 256ch. Compare with ThermoNet-9 which uses uniform 64ch (瘦长型).
    
    Input: (batch, 3, 32, 32)
    Output: (batch, 10)
    """
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        channels = [3, 64, 64, 128, 128, 256, 128, 64]
        
        self.blocks = nn.ModuleList()
        self.skip_ops = nn.ModuleList()
        
        for i in range(len(channels) - 1):
            self.blocks.append(ConvBlock(
                channels[i], channels[i + 1],
                activation='tga', use_norm=True
            ))
            if i > 0:
                self.skip_ops.append(SkipConnection(channels[i], channels[i + 1]))
            else:
                self.skip_ops.append(None)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = None
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i > 0 and self.skip_ops[i] is not None:
                x = self.skip_ops[i](x, residual)
            residual = x.detach()
        
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ThermoNet9(nn.Module):
    """G1-4: Thermogeometric Optimal 9-layer network.
    
    Architecture: [64] × 8 with skip every 2 layers
    - 8 conv blocks with skip connections every 2 layers
    - GELU activation
    - LayerNorm after each block
    
    Design note: Uses uniform 64ch width (瘦长型 / slender architecture).
    Unlike ThermoNet-5/-7 which widen to 256ch then narrow (矮宽型),
    ThermoNet-9 stays at 64ch throughout for a deeper narrower profile.
    Naming reflects topology (skip interval pattern), not channel width.
    
    Input: (batch, 3, 32, 32)
    Output: (batch, 10)
    """
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        channels = [3] + [64] * 8
        
        self.blocks = nn.ModuleList()
        self.skip_interval = 2
        
        for i in range(len(channels) - 1):
            self.blocks.append(ConvBlock(
                channels[i], channels[i + 1],
                activation='gelu', use_norm=True
            ))
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residuals = []
        
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i % self.skip_interval == 0 and i > 0:
                # Apply skip from block i-2
                skip_idx = (i // self.skip_interval) - 1
                if skip_idx < len(residuals):
                    residual = residuals[skip_idx]
                    if residual.shape[1] == x.shape[1]:
                        x = x + residual
            residuals.append(x.detach())
        
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# =============================================================================
# Group 2: Topology Destroyer Architectures (with bottlenecks)
# =============================================================================

class ThermoBot3(nn.Module):
    """G2-1: Topology Destroyer - ThermoNet-3 with 8x bottleneck.
    
    Architecture: [64, 64, 8, 128, 128] where 8 is the bottleneck
    - Same structure as ThermoNet-3 but with 8x compression bottleneck
    
    Input: (batch, 3, 32, 32)
    Output: (batch, 10)
    """
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # First two blocks without bottleneck
        self.block1 = ConvBlock(3, 64, activation='gelu', use_norm=True)
        self.block2 = ConvBlock(64, 64, activation='gelu', use_norm=True)
        self.skip1 = SkipConnection(64, 64)
        
        # Bottleneck block
        self.bottleneck = BottleneckBlock(64, 128, bottleneck_dim=8, activation='gelu', use_norm=True)
        
        # Final block
        self.block3 = ConvBlock(128, 128, activation='gelu', use_norm=True)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.block1(x)
        residual = x.detach()  # Now residual = block1 output (64ch)
        x = self.block2(x)
        x = self.skip1(x, residual)
        
        x = self.bottleneck(x)
        x = self.block3(x)
        
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ThermoBot5(nn.Module):
    """G2-2: Topology Destroyer - ThermoNet-5 with 8x bottleneck.
    
    Architecture: [64, 128, 16, 128, 64] where 16 is the bottleneck
    - Same structure as ThermoNet-5 but with 8x compression bottleneck
    
    Input: (batch, 3, 32, 32)
    Output: (batch, 10)
    """
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.block1 = ConvBlock(3, 64, activation='gelu', use_norm=True)
        self.block2 = ConvBlock(64, 128, activation='gelu', use_norm=True)
        self.skip1 = SkipConnection(64, 128)
        
        # Bottleneck block
        self.bottleneck = BottleneckBlock(128, 128, bottleneck_dim=16, activation='gelu', use_norm=True)
        
        self.block3 = ConvBlock(128, 128, activation='gelu', use_norm=True)
        self.block4 = ConvBlock(128, 64, activation='gelu', use_norm=True)
        self.skip2 = SkipConnection(128, 64)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.block1(x)
        residual = x.detach()  # Now residual = block1 output (64ch)
        x = self.block2(x)
        x = self.skip1(x, residual)
        
        x = self.bottleneck(x)
        
        residual = x.detach()
        x = self.block3(x)
        x = self.block4(x)
        x = self.skip2(x, residual)
        
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ThermoBot7(nn.Module):
    """G2-3: Topology Destroyer - ThermoNet-7 with two 8x bottlenecks.
    
    Architecture: [64, 64, 8, 128, 128, 16, 64] with two bottlenecks
    - Same structure as ThermoNet-7 but with two 8x compression bottlenecks
    
    Input: (batch, 3, 32, 32)
    Output: (batch, 10)
    """
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.block1 = ConvBlock(3, 64, activation='tga', use_norm=True)
        self.block2 = ConvBlock(64, 64, activation='tga', use_norm=True)
        self.skip1 = SkipConnection(64, 64)
        
        # First bottleneck
        self.bottleneck1 = BottleneckBlock(64, 128, bottleneck_dim=8, activation='tga', use_norm=True)
        
        self.block3 = ConvBlock(128, 128, activation='tga', use_norm=True)
        self.block4 = ConvBlock(128, 256, activation='tga', use_norm=True)
        self.skip2 = SkipConnection(128, 256)
        
        # Second bottleneck
        self.bottleneck2 = BottleneckBlock(256, 128, bottleneck_dim=16, activation='tga', use_norm=True)
        
        self.block5 = ConvBlock(128, 64, activation='tga', use_norm=True)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.block1(x)
        residual = x.detach()  # Now residual = block1 output (64ch)
        x = self.block2(x)
        x = self.skip1(x, residual)
        
        x = self.bottleneck1(x)
        
        residual = x.detach()
        x = self.block3(x)
        x = self.block4(x)
        x = self.skip2(x, residual)
        
        x = self.bottleneck2(x)
        x = self.block5(x)
        
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ThermoBot9(nn.Module):
    """G2-4: Topology Destroyer - ThermoNet-9 with bottlenecks every 4 layers.
    
    Architecture: [64] × 8 with bottlenecks every 4 layers
    - Same structure as ThermoNet-9 but with bottlenecks inserted
    
    Input: (batch, 3, 32, 32)
    Output: (batch, 10)
    """
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        channels = [3] + [64] * 8
        
        self.blocks = nn.ModuleList()
        
        for i in range(len(channels) - 1):
            if i == 3:  # Insert bottleneck at layer 4
                self.blocks.append(BottleneckBlock(
                    channels[i], channels[i + 1],
                    bottleneck_dim=16, activation='gelu', use_norm=True
                ))
            else:
                self.blocks.append(ConvBlock(
                    channels[i], channels[i + 1],
                    activation='gelu', use_norm=True
                ))
        
        self.skip_interval = 2
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residuals = []
        
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i % self.skip_interval == 0 and i > 0:
                # Apply skip from block i-2
                skip_idx = (i // self.skip_interval) - 1
                if skip_idx < len(residuals):
                    residual = residuals[skip_idx]
                    if residual.shape[1] == x.shape[1]:
                        x = x + residual
            residuals.append(x.detach())
        
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# =============================================================================
# Group 3: Thermal Boiling Furnace (ReLU ablation - no Norm, no Skip)
# =============================================================================

class ReLUFurnace3(nn.Module):
    """G3-1: Thermal Boiling Furnace - ThermoNet-3 structure, ReLU, no Norm, no Skip.
    
    Architecture: [64, 64, 128, 128]
    - Same structure as ThermoNet-3 but:
    - ReLU activation instead of GELU
    - No LayerNorm
    - No skip connections
    
    Input: (batch, 3, 32, 32)
    Output: (batch, 10)
    """
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        channels = [3, 64, 64, 128, 128]
        
        self.blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.blocks.append(nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, padding=1))
        
        self.activations = nn.ModuleList([nn.ReLU(inplace=True) for _ in range(4)])
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, block in enumerate(self.blocks):
            x = block(x)
            x = self.activations[i](x)
        
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ReLUFurnace5(nn.Module):
    """G3-2: Thermal Boiling Furnace - ThermoNet-5 structure, ReLU, no Norm, no Skip.
    
    Architecture: [64, 128, 256, 128, 64]
    - Same structure as ThermoNet-5 but:
    - ReLU activation instead of GELU
    - No LayerNorm
    - No skip connections
    
    Input: (batch, 3, 32, 32)
    Output: (batch, 10)
    """
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        channels = [3, 64, 128, 256, 128, 64]
        
        self.blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.blocks.append(nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, padding=1))
        
        self.activations = nn.ModuleList([nn.ReLU(inplace=True) for _ in range(5)])
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, block in enumerate(self.blocks):
            x = block(x)
            x = self.activations[i](x)
        
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ReLUFurnace7(nn.Module):
    """G3-3: Thermal Boiling Furnace - ThermoNet-7 structure, ReLU, no Norm, no Skip.
    
    Architecture: [64, 64, 128, 128, 256, 128, 64]
    - Same structure as ThermoNet-7 but:
    - ReLU activation instead of TGA
    - No LayerNorm
    - No skip connections
    
    Input: (batch, 3, 32, 32)
    Output: (batch, 10)
    """
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        channels = [3, 64, 64, 128, 128, 256, 128, 64]
        
        self.blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.blocks.append(nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, padding=1))
        
        self.activations = nn.ModuleList([nn.ReLU(inplace=True) for _ in range(7)])
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, block in enumerate(self.blocks):
            x = block(x)
            x = self.activations[i](x)
        
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ReLUFurnace9(nn.Module):
    """G3-4: Thermal Boiling Furnace - ThermoNet-9 structure, ReLU, no Norm, no Skip.
    
    Architecture: [64] × 8
    - Same structure as ThermoNet-9 but:
    - ReLU activation instead of GELU
    - No LayerNorm
    - No skip connections
    
    Design note: Uses uniform 64ch width (瘦长型 / slender architecture).
    Naming reflects topology (skip interval pattern), not channel width.
    See ThermoNet9 docstring for design rationale comparison.
    
    Input: (batch, 3, 32, 32)
    Output: (batch, 10)
    """
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        channels = [3] + [64] * 8
        
        self.blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.blocks.append(nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, padding=1))
        
        self.activations = nn.ModuleList([nn.ReLU(inplace=True) for _ in range(8)])
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, block in enumerate(self.blocks):
            x = block(x)
            x = self.activations[i](x)
        
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# =============================================================================
# Group 4: Traditional Baselines
# =============================================================================

class ResNet18CIFAR(nn.Module):
    """G4-1: ResNet-18 adapted for CIFAR-10.
    
    Standard ResNet-18 but:
    - Initial conv is 3x3 (not 7x7 with maxpool)
    - No max pooling layer
    - Adapted for 32x32 input
    
    Input: (batch, 3, 32, 32)
    Output: (batch, 10)
    """
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        # Initial convolution (CIFAR-style, no maxpool)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual layers
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)  # 32 -> 16
        self.layer3 = self._make_layer(128, 256, 2, stride=2)  # 16 -> 8
        self.layer4 = self._make_layer(256, 512, 2, stride=2)  # 8 -> 4
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int):
        layers = []
        
        # First block may have downsample
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample))
        
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class BasicBlock(nn.Module):
    """Basic residual block for ResNet."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out)
        return out


class VGG11CIFAR(nn.Module):
    """G4-2: VGG-11 adapted for CIFAR-10.
    
    Standard VGG-11 with:
    - Reduced channel dimensions for CIFAR-10
    - BatchNorm (standard for modern VGG)
    
    Input: (batch, 3, 32, 32)
    Output: (batch, 10)
    """
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        channels = [64, 128, 256, 256, 512, 512]
        
        layers = []
        in_ch = 3
        
        for idx, out_ch in enumerate(channels):
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            # Only pool for first 5 groups (spatial: 32→16→8→4→2→1, no 6th pool)
            if idx < len(channels) - 1:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_ch = out_ch
        
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class DenseNet40CIFAR(nn.Module):
    """G4-3: DenseNet-40 adapted for CIFAR-10.
    
    DenseNet with:
    - Growth rate = 12
    - 3 dense blocks (12 layers each = 40 total)
    - Compression factor = 0.5 for transition layers
    
    Input: (batch, 3, 32, 32)
    Output: (batch, 10)
    """
    
    def __init__(self, num_classes: int = 10, growth_rate: int = 12, block_layers: int = 12, compression: float = 0.5):
        super().__init__()
        
        self.growth_rate = growth_rate
        num_channels = 2 * growth_rate  # Initial channels
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=3, padding=1, bias=False)
        
        # Dense blocks
        self.dense1 = self._make_dense_block(num_channels, block_layers)
        num_channels += block_layers * growth_rate
        out_channels = int(num_channels * compression)
        self.trans1 = self._make_transition(num_channels, out_channels)
        
        num_channels = out_channels
        self.dense2 = self._make_dense_block(num_channels, block_layers)
        num_channels += block_layers * growth_rate
        out_channels = int(num_channels * compression)
        self.trans2 = self._make_transition(num_channels, out_channels)
        
        num_channels = out_channels
        self.dense3 = self._make_dense_block(num_channels, block_layers)
        num_channels += block_layers * growth_rate
        
        self.bn = nn.BatchNorm2d(num_channels)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(num_channels, num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_dense_block(self, in_channels: int, layers: int):
        block = []
        for _ in range(layers):
            block.append(_DenseLayer(in_channels + _ * self.growth_rate, self.growth_rate))
        return nn.Sequential(*block)
    
    def _make_transition(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        
        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.trans2(x)
        x = self.dense3(x)
        
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class _DenseLayer(nn.Module):
    """Single layer in DenseNet."""
    
    def __init__(self, in_channels: int, growth_rate: int):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return torch.cat([x, out], 1)


# =============================================================================
# Model Registry
# =============================================================================

MODEL_REGISTRY = {
    # Group 1: Thermogeometric Optimal
    "ThermoNet-3": ThermoNet3,
    "ThermoNet-5": ThermoNet5,
    "ThermoNet-7": ThermoNet7,
    "ThermoNet-9": ThermoNet9,
    
    # Group 2: Topology Destroyer
    "ThermoBot-3": ThermoBot3,
    "ThermoBot-5": ThermoBot5,
    "ThermoBot-7": ThermoBot7,
    "ThermoBot-9": ThermoBot9,
    
    # Group 3: Thermal Boiling Furnace
    "ReLUFurnace-3": ReLUFurnace3,
    "ReLUFurnace-5": ReLUFurnace5,
    "ReLUFurnace-7": ReLUFurnace7,
    "ReLUFurnace-9": ReLUFurnace9,
    
    # Group 4: Traditional Baselines
    "ResNet-18-CIFAR": ResNet18CIFAR,
    "VGG-11-CIFAR": VGG11CIFAR,
    "DenseNet-40-CIFAR": DenseNet40CIFAR,
}


def get_model(name: str, num_classes: int = 10) -> nn.Module:
    """Get model by name.
    
    Args:
        name: Model name (must be in MODEL_REGISTRY)
        num_classes: Number of output classes
        
    Returns:
        Instantiated model
        
    Raises:
        ValueError: If model name not found
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](num_classes=num_classes)


def list_models() -> list:
    """List all available model names."""
    return list(MODEL_REGISTRY.keys())


def estimate_model_flops(model: nn.Module) -> int:
    """Estimate FLOPs for a model.
    
    Uses a simple convolution FLOP counting heuristic.
    
    Args:
        model: PyTorch model
        
    Returns:
        Estimated FLOPs as integer
    """
    total_flops = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Output spatial dimensions
            h = 32 // (module.stride[0] or 1)
            w = 32 // (module.stride[1] or 1)
            # Conv FLOPs: 2 * output_size * kernel_size^2 * in_channels * out_channels
            kernel_prod = module.kernel_size[0] * module.kernel_size[1]
            total_flops += 2 * h * w * kernel_prod * module.in_channels * module.out_channels
        elif isinstance(module, nn.Linear):
            total_flops += 2 * module.in_features * module.out_features
    
    return total_flops


def get_model_info(name: str) -> dict:
    """Get model information (params, FLOPs, group).
    
    Args:
        name: Model name
        
    Returns:
        Dictionary with model info
    """
    model = get_model(name)
    params = count_parameters(model)
    flops = estimate_model_flops(model)
    
    from .constants import ARCH_MAP
    spec = ARCH_MAP.get(name)
    
    return {
        "name": name,
        "group": spec.group if spec else None,
        "parameters": params,
        "flops": flops,
        "channels": spec.channels if spec else None,
        "activation": spec.activation if spec else None,
        "has_skip": spec.has_skip if spec else None,
        "has_norm": spec.has_norm if spec else None,
    }