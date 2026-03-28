# SPDX-License-Identifier: Apache-2.0

"""Architecture layer efficiency η_l computation.

Phase 3: Architecture Surrogate (Fast-Track) - Heuristic η_l = w_l · ν(σ) · c_conn
Phase 4: Thermodynamic Probing (Exact-Track) - η_l = D_eff / d^(l-1)

Extended Module Support:
    - Preset modules with known formulas (conv, attention, pooling, etc.)
    - Custom modules with user-defined activations (including TGA)
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional, List, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from .jacobian import JacobianAnalyzer
from ...utils.math import safe_divide


class ArchitectureAnalyzer:
    """Computes layer-wise efficiency η_l for neural network architectures.
    
    Phase 3 (Fast-Track): Heuristic computation
        η_l = w_l · ν(σ) · c_conn
    
    Phase 4 (Exact-Track): Thermodynamic probing
        Initialize d^(0) = d_manifold
        For each layer: η_l = D_eff / d^(l-1)
    """
    
    def __init__(self, device: str = 'cpu'):
        """Initialize ArchitectureAnalyzer.
        
        Args:
            device: Computation device
        """
        self.device = device
        self.jacobian_analyzer = JacobianAnalyzer(device=device)
        self._eta_ls: Optional[List[float]] = None
        self._d_effs: Optional[List[float]] = None
        
    def compute_heuristic_eta(
        self,
        layer_widths: List[int],
        layer_types: Optional[List[str]] = None,
        connection_sparsity: Optional[List[float]] = None,
        weight_variances: Optional[List[float]] = None,
        d_manifold: float = 10.0,
    ) -> List[float]:
        """Compute heuristic η_l using layer properties.
        
        Fast-Track Phase 3:
            η_l = w_l · ν(σ) · c_conn
        
        where:
            w_l = min(1, width_l / d_manifold) - width factor
            ν(σ) = 1 / (1 + std(weights)) - variance factor  
            c_conn = connection density (for sparse connections)
        
        Args:
            layer_widths: List of layer widths
            layer_types: List of layer types (linear, conv1d, conv2d, etc.)
            connection_sparsity: List of sparsity ratios [0, 1] per layer
            weight_variances: List of weight variances per layer
            d_manifold: Manifold dimension
            
        Returns:
            List of η_l values per layer
        """
        if layer_types is None:
            layer_types = ['linear'] * len(layer_widths)
        if connection_sparsity is None:
            connection_sparsity = [1.0] * len(layer_widths)
        if weight_variances is None:
            weight_variances = [1.0] * len(layer_widths)
        
        eta_ls = []
        
        for l, (width, ltype, sparsity, wvar) in enumerate(
            zip(layer_widths, layer_types, connection_sparsity, weight_variances)
        ):
            # Width factor: w_l = min(1, width / d_manifold)
            w_factor = min(1.0, width / d_manifold)
            
            # Variance factor: ν(σ) = 1 / (1 + std(weights))
            std_weights = np.sqrt(wvar)
            nu_factor = 1.0 / (1.0 + std_weights)
            
            # Connection density factor
            c_conn = sparsity
            
            # Compute η_l
            eta_l = w_factor * nu_factor * c_conn
            
            # Ensure minimum value for numerical stability
            eta_l = max(eta_l, 1e-6)
            
            eta_ls.append(eta_l)
        
        self._eta_ls = eta_ls
        return eta_ls
    
    def compute_thermodynamic_eta(
        self,
        model: 'torch.nn.Module',
        X: NDArray[np.floating],
        d_manifold: float,
        gamma_c: Optional[float] = None,
        eta_lr: Optional[float] = None,
        V_grad: Optional[float] = None,
        layer_indices: Optional[List[int]] = None,
    ) -> List[float]:
        """Compute thermodynamic η_l using Jacobian analysis.
        
        Phase 4 (Exact-Track):
            Initialize d^(0) = d_manifold
            γ_c ≈ (η_lr · V_∇) / d_manifold
            For each layer:
                D_eff = ||J||_F² / ||J||²
                η_l = D_eff / d^(l-1)
                d^(l) = η_l · d^(l-1)
        
        Args:
            model: PyTorch model
            X: Input data
            d_manifold: Manifold dimension
            gamma_c: Critical cooling rate (computed if not provided)
            eta_lr: Learning rate for critical computation
            V_grad: Gradient variance
            layer_indices: Indices of layers to analyze
            
        Returns:
            List of η_l values per layer
        """
        # Compute gamma_c if not provided
        if gamma_c is None:
            eta_lr = eta_lr or 1e-3
            V_grad = V_grad or 1.0
            gamma_c = safe_divide(eta_lr * V_grad, d_manifold)
        
        d_effs = self._compute_d_effs_per_layer(model, X, layer_indices)
        self._d_effs = d_effs
        
        # Compute η_l iteratively
        eta_ls = []
        d_current = d_manifold
        
        for d_eff in d_effs:
            # η_l = D_eff / d^(l-1)
            eta_l = safe_divide(d_eff, d_current)
            
            # Update d for next layer
            d_current = eta_l * d_current
            
            eta_ls.append(eta_l)
        
        self._eta_ls = eta_ls
        return eta_ls
    
    def _compute_d_effs_per_layer(
        self,
        model: 'torch.nn.Module',
        X: NDArray[np.floating],
        layer_indices: Optional[List[int]] = None,
    ) -> List[float]:
        """Compute D_eff for each layer.
        
        Args:
            model: PyTorch model
            X: Input data
            layer_indices: Specific layer indices to compute
            
        Returns:
            List of D_eff values per layer
        """
        try:
            import torch
        except ImportError:
            return [1.0]
        
        model.eval()
        device = torch.device(self.device)
        X_tensor = torch.tensor(X[:10], dtype=torch.float32, device=device)  # Use subset for speed
        
        d_effs = []
        d_prev = None
        
        # Hook to capture activations
        activations = {}
        
        def get_hook(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook
        
        handles = []
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv1d)):
                handles.append(module.register_forward_hook(get_hook(name)))
        
        with torch.no_grad():
            _ = model(X_tensor)
        
        for h in handles:
            h.remove()
        
        # Compute D_eff from activation shapes
        prev_dim = X_tensor.shape[-1]
        for name, act in activations.items():
            if len(act.shape) > 1:
                out_dim = act.shape[-1]
            else:
                out_dim = act.shape[-1]
            
            # Approximate D_eff as ratio of output to input dimension
            d_eff = safe_divide(out_dim, prev_dim) if prev_dim > 0 else 1.0
            d_effs.append(d_eff)
            
            prev_dim = out_dim
        
        return d_effs if d_effs else [1.0]
    
    def compute_eta_from_model(
        self,
        model: 'torch.nn.Module',
        X: NDArray[np.floating],
        d_manifold: float,
        use_thermodynamic: bool = True,
        **kwargs
    ) -> List[float]:
        """Compute η_l from a PyTorch model.
        
        Automatically extracts layer information and computes η_l.
        
        Args:
            model: PyTorch model
            X: Input data
            d_manifold: Manifold dimension
            use_thermodynamic: If True, use exact-track; else heuristic
            **kwargs: Additional arguments for specific methods
            
        Returns:
            List of η_l values per layer
        """
        if use_thermodynamic:
            return self.compute_thermodynamic_eta(model, X, d_manifold, **kwargs)
        else:
            # Extract layer info for heuristic
            layer_widths = []
            layer_types = []
            
            for module in model.modules():
                if isinstance(module, torch.nn.Linear):
                    layer_widths.append(module.out_features)
                    layer_types.append('linear')
                elif isinstance(module, torch.nn.Conv2d):
                    layer_widths.append(module.out_channels)
                    layer_types.append('conv2d')
                elif isinstance(module, torch.nn.Conv1d):
                    layer_widths.append(module.out_channels)
                    layer_types.append('conv1d')
            
            # Remove input/output layers, keep hidden
            if len(layer_widths) > 2:
                layer_widths = layer_widths[1:-1]
                layer_types = layer_types[1:-1]
            
            return self.compute_heuristic_eta(
                layer_widths, layer_types, d_manifold=d_manifold, **kwargs
            )
    
    def compute_eta_from_modules(
        self,
        modules: List['BaseModule'],
        d_manifold: float,
    ) -> List[float]:
        """Compute η_l from a list of BaseModule objects.
        
        Uses each module's built-in compute_eta method for precise
        module-specific efficiency calculation.
        
        Args:
            modules: List of BaseModule objects
            d_manifold: Manifold dimension
            
        Returns:
            List of η_l values per module
        """
        eta_ls = []
        d_current = d_manifold
        
        for module in modules:
            # Use module's own compute_eta method
            eta_l = module.compute_eta(d_current)
            eta_ls.append(eta_l)
            
            # Update d_current for next layer
            d_current = eta_l * d_current
        
        self._eta_ls = eta_ls
        return eta_ls
    
    def compute_module_eta(
        self,
        module_type: str,
        config: dict,
        d_prev: float,
    ) -> float:
        """Compute η for a single module type.
        
        Provides module-type specific η_l computation for:
            - conv1d, conv2d: Based on kernel size and channels
            - attention: Based on num_heads and head_dim
            - pooling: Based on stride (compression)
            - residual: Always 1.0 (preserves dimension)
            - linear: Based on output/input ratio
            - embedding: Always 1.0
            - layernorm, batchnorm: Always 1.0
            - dropout: Always 1.0
        
        Args:
            module_type: Type of module
            config: Module configuration dict
            d_prev: Previous layer dimension
            
        Returns:
            η_l value
        """
        module_type = module_type.lower()
        
        if module_type in ['conv1d', 'conv2d']:
            kernel_prod = config.get('kernel_size', 3) ** 2
            in_channels = config.get('in_channels', d_prev)
            out_channels = config.get('out_channels', in_channels)
            effective_width = out_channels * kernel_prod
            eta = min(1.0, effective_width / d_prev)
            
        elif module_type == 'attention':
            num_heads = config.get('num_heads', 8)
            head_dim = config.get('head_dim', 64)
            effective_width = num_heads * head_dim
            eta = min(1.0, effective_width / d_prev)
            
        elif module_type == 'residual':
            eta = 1.0
            
        elif module_type == 'pooling':
            stride = config.get('stride', 2)
            eta = stride ** -2  # Compression factor
            
        elif module_type == 'linear':
            out_dim = config.get('out_channels', config.get('out_features', d_prev))
            in_dim = config.get('in_channels', config.get('in_features', d_prev))
            eta = min(1.0, out_dim / in_dim)
            
        elif module_type in ['embedding', 'layernorm', 'batchnorm', 'ln', 'bn', 'dropout']:
            eta = 1.0
            
        elif module_type == 'flatten':
            eta = 1.0
            
        else:
            # Default: conservative
            out_dim = config.get('out_channels', d_prev)
            eta = min(1.0, out_dim / d_prev)
        
        return max(eta, 1e-6)
    
    def filter_top_k(
        self,
        eta_ls: List[float],
        architectures: List[dict],
        k: int = 10
    ) -> List[dict]:
        """Filter top-K architectures based on η_l.
        
        Args:
            eta_ls: List of η_l values for current architecture
            architectures: List of architecture configurations
            k: Number of top architectures to keep
            
        Returns:
            Top-K architectures sorted by η_l product
        """
        # Compute product of η_l
        eta_product = np.prod(eta_ls)
        
        # Score each architecture
        scored = []
        for arch, eta in zip(architectures, eta_ls) if len(architectures) == len(eta_ls) else []:
            arch['eta_product'] = np.prod(eta) if isinstance(eta, list) else eta
            arch['eta_ls'] = eta if isinstance(eta, list) else [eta]
            scored.append(arch)
        
        # Sort by score and return top-K
        scored.sort(key=lambda x: x['eta_product'], reverse=True)
        return scored[:k]
    
    @property
    def eta_ls(self) -> Optional[List[float]]:
        """Return last computed η_l values."""
        return self._eta_ls
    
    @property
    def d_effs(self) -> Optional[List[float]]:
        """Return last computed D_eff values."""
        return self._d_effs
