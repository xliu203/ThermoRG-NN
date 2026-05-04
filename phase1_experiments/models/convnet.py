"""
ConvNet L=5 with configurable normalization for ThermoRG v4 experiments.

Architecture:
- L=5 layers, kernel=3x3, no skip connections
- Normalization: configurable (None, BatchNorm, LayerNorm, GroupNorm)
- Activation: GELU
- Initialization: Kaiming normal

Key design:
- Hooks to extract activation ℓ₂ norms per layer
- Easy access to weight matrices for λ_max computation
- Normalized output (no affine) accessible for γ measurement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple
import copy


class ConvNetL5(nn.Module):
    """
    ConvNet with L=5 layers, configurable normalization.
    
    Architecture per layer:
    Conv2d -> Normalization -> Activation (GELU)
    
    No skip connections (per protocol specification).
    """
    
    def __init__(
        self,
        D: int = 64,  # Channel width
        num_classes: int = 10,
        norm_type: str = 'batchnorm',  # 'none', 'batchnorm', 'layernorm', 'groupnorm'
        group_size: int = 4,  # For GroupNorm
        input_channels: int = 3,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        super().__init__()
        
        self.D = D
        self.num_classes = num_classes
        self.norm_type = norm_type
        self.L = 5  # Fixed per protocol
        self.group_size = group_size
        
        # Build layers
        self.conv1 = nn.Conv2d(input_channels, D, kernel_size, padding=padding)
        self.norm1 = self._make_norm(D)
        
        self.conv2 = nn.Conv2d(D, D, kernel_size, padding=padding)
        self.norm2 = self._make_norm(D)
        
        self.conv3 = nn.Conv2d(D, D, kernel_size, padding=padding)
        self.norm3 = self._make_norm(D)
        
        self.conv4 = nn.Conv2d(D, D, kernel_size, padding=padding)
        self.norm4 = self._make_norm(D)
        
        self.conv5 = nn.Conv2d(D, D, kernel_size, padding=padding)
        self.norm5 = self._make_norm(D)
        
        # Global pooling + classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(D, num_classes)
        
        # Activation function (fixed to GELU per protocol)
        self.activation = nn.GELU()
        
        # Initialize weights with Kaiming normal
        self._initialize_weights()
        
        # Storage for activation hooks
        self._activation_hooks = []
        self._stored_activations = {}
        
        # Per-layer normalization stats (for γ measurement)
        self._norm_stats = {}
        
    def _make_norm(self, num_channels: int) -> Optional[nn.Module]:
        """Create normalization layer based on type."""
        if self.norm_type == 'batchnorm':
            return nn.BatchNorm2d(num_channels, affine=False)  # No affine for γ measurement
        elif self.norm_type == 'layernorm':
            # Use GroupNorm(1, C) which is mathematically equivalent to LayerNorm(C)
            # for 4D tensors (N,C,H,W), but works correctly at any spatial size
            return nn.GroupNorm(1, num_channels, affine=False)
        elif self.norm_type == 'groupnorm':
            return nn.GroupNorm(self.group_size, num_channels, affine=False)
        elif self.norm_type == 'none':
            return None
        else:
            raise ValueError(f"Unknown norm_type: {self.norm_type}")
    
    def _initialize_weights(self):
        """Initialize weights with Kaiming normal (for GELU)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                # Normal initialization for normalization layers
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
    
    def _get_norm_output(self, x: torch.Tensor, norm_layer: Optional[nn.Module]) -> torch.Tensor:
        """Get normalized output without affine transform."""
        if norm_layer is None:
            return x
        
        if isinstance(norm_layer, nn.BatchNorm2d):
            # BatchNorm: use running stats during eval, batch stats during train
            if self.training:
                return norm_layer(x)
            else:
                # Use running mean/var for eval
                return (x - norm_layer.running_mean.view(1, -1, 1, 1)) / \
                       torch.sqrt(norm_layer.running_var.view(1, -1, 1, 1) + norm_layer.eps)
        
        # GroupNorm (includes GroupNorm(1, C) for layernorm replacement)
        return norm_layer(x)
    
    def forward(self, x: torch.Tensor, store_activations: bool = False) -> torch.Tensor:
        """
        Forward pass with optional activation storage for γ measurement.
        
        Args:
            x: Input tensor (N, 3, 32, 32) for CIFAR-10
            store_activations: If True, store normalized activations per layer
            
        Returns:
            Logits (N, num_classes)
        """
        # Layer 1: Conv -> Norm -> Act
        x = self.conv1(x)
        norm_out_1 = self._get_norm_output(x, self.norm1)
        x = self.activation(norm_out_1)
        if store_activations:
            self._stored_activations['layer1'] = norm_out_1.detach()
        
        # Layer 2
        x = self.conv2(x)
        norm_out_2 = self._get_norm_output(x, self.norm2)
        x = self.activation(norm_out_2)
        if store_activations:
            self._stored_activations['layer2'] = norm_out_2.detach()
        
        # Layer 3
        x = self.conv3(x)
        norm_out_3 = self._get_norm_output(x, self.norm3)
        x = self.activation(norm_out_3)
        if store_activations:
            self._stored_activations['layer3'] = norm_out_3.detach()
        
        # Layer 4
        x = self.conv4(x)
        norm_out_4 = self._get_norm_output(x, self.norm4)
        x = self.activation(norm_out_4)
        if store_activations:
            self._stored_activations['layer4'] = norm_out_4.detach()
        
        # Layer 5
        x = self.conv5(x)
        norm_out_5 = self._get_norm_output(x, self.norm5)
        x = self.activation(norm_out_5)
        if store_activations:
            self._stored_activations['layer5'] = norm_out_5.detach()
        
        # Global pooling and classification
        x = self.global_pool(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        
        return x
    
    def get_stored_activations(self) -> Dict[str, torch.Tensor]:
        """Get stored activations from last forward pass."""
        return self._stored_activations
    
    def clear_stored_activations(self):
        """Clear stored activations."""
        self._stored_activations = {}
    
    def get_weight_matrices(self) -> Dict[str, torch.Tensor]:
        """
        Get weight matrices for all conv layers for λ_max computation.
        
        Returns:
            Dict mapping layer name -> weight tensor reshaped for power iteration
        """
        weights = {}
        for i in range(1, 6):
            conv = getattr(self, f'conv{i}')
            # Reshape to (out_channels, in_channels * kH * kW)
            W = conv.weight.data
            W_mat = W.reshape(W.shape[0], -1)
            weights[f'conv{i}'] = W_mat
        return weights
    
    def get_all_weights(self) -> List[torch.Tensor]:
        """Get all layer weights as list."""
        return [getattr(self, f'conv{i}').weight.data for i in range(1, 6)]
    
    def get_layer_names(self) -> List[str]:
        """Get list of layer names."""
        return [f'conv{i}' for i in range(1, 6)]
    
    def num_parameters(self) -> int:
        """Total number of parameters."""
        return sum(p.numel() for p in self.parameters())


def create_model(D: int, norm_type: str, **kwargs) -> ConvNetL5:
    """
    Factory function to create a ConvNetL5 model.
    
    Args:
        D: Channel width
        norm_type: 'none', 'batchnorm', 'layernorm', 'groupnorm'
        **kwargs: Additional arguments to ConvNetL5
        
    Returns:
        ConvNetL5 model
    """
    return ConvNetL5(D=D, norm_type=norm_type, **kwargs)


class ModelFactory:
    """
    Factory for creating models with consistent interface.
    
    Usage:
        factory = ModelFactory(D=256, norm_type='batchnorm')
        model = factory()  # Creates new model with fresh initialization
    """
    
    def __init__(self, D: int, norm_type: str, **model_kwargs):
        self.D = D
        self.norm_type = norm_type
        self.model_kwargs = model_kwargs
    
    def __call__(self) -> ConvNetL5:
        """Create a new model with fresh initialization."""
        return create_model(self.D, self.norm_type, **self.model_kwargs)
    
    def create_multiple(self, n: int) -> List[ConvNetL5]:
        """Create multiple models with different random initializations."""
        return [self() for _ in range(n)]


# =============================================================================
# COMPATIBILITY WITH EXISTING MEASUREMENT CODE
# =============================================================================

def get_model_for_measurement(model: ConvNetL5) -> Dict:
    """
    Prepare model for measurement utilities.
    
    Returns dict with:
    - weights: List of weight matrices
    - layer_names: List of layer names
    - norm_stats: Running stats if BN
    """
    return {
        'weights': model.get_all_weights(),
        'layer_names': model.get_layer_names(),
        'norm_type': model.norm_type,
        'D': model.D,
        'get_weight_matrices': model.get_weight_matrices,
    }
