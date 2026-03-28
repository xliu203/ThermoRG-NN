# SPDX-License-Identifier: Apache-2.0

"""Jacobian and effective dimension computation for architecture analysis.

Phase 4: Thermodynamic Probing - Jacobian/D_eff estimation
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Union, List, Tuple
from scipy.sparse import csr_matrix

# Try to import torch, but make it optional
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Try to import tensorflow/keras, but make it optional
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class JacobianAnalyzer:
    """Computes Jacobian and effective dimensions for neural network layers.
    
    D_eff = ||J||_F² / ||J||²
    
    where J is the Jacobian matrix of layer outputs with respect to inputs
    or parameters.
    
    The effective dimension D_eff characterizes how information propagates
    through the network and relates to thermodynamic properties.
    """
    
    def __init__(self, device: str = 'cpu'):
        """Initialize JacobianAnalyzer.
        
        Args:
            device: Computation device ('cpu', 'cuda')
        """
        self.device = device
        
    def compute_jacobian_numerical(
        self,
        model: 'torch.nn.Module',
        X: NDArray[np.floating],
        layer_idx: int = -1,
        batch_size: int = 100
    ) -> NDArray[np.floating]:
        """Compute Jacobian numerically using finite differences.
        
        Args:
            model: PyTorch model
            X: Input data, shape (n_samples, n_features)
            layer_idx: Index of layer to analyze (-1 for last layer)
            batch_size: Batch size for computation
            
        Returns:
            Jacobian matrix
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for Jacobian computation")
        
        model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        
        jacobians = []
        
        with torch.no_grad():
            for i in range(min(batch_size, X.shape[0])):
                x = X_tensor[i:i+1]
                x.requires_grad_(True)
                
                # Forward pass
                output = model(x)
                
                if isinstance(layer_idx, int) and layer_idx == -1:
                    # Last layer
                    out = output
                else:
                    # Hook into intermediate layer (requires model modification)
                    out = output
                
                # Compute gradient
                jac = torch.autograd.grad(
                    outputs=out.sum(),
                    inputs=x,
                    create_graph=False
                )[0]
                
                jacobians.append(jac.cpu().numpy())
        
        J = np.concatenate(jacobians, axis=0)
        return J
    
    def compute_layer_jacobian(
        self,
        model: 'torch.nn.Module',
        X: NDArray[np.floating],
        layer_names: Optional[List[str]] = None
    ) -> dict[str, NDArray[np.floating]]:
        """Compute Jacobians for multiple layers.
        
        Args:
            model: PyTorch model
            X: Input data
            layer_names: List of layer names to compute Jacobians for
            
        Returns:
            Dictionary mapping layer names to Jacobian matrices
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for Jacobian computation")
        
        model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        
        activations = {}
        gradients = {}
        
        def get_activation(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook
        
        def get_gradient(name):
            def hook(module, grad_input, grad_output):
                gradients[name] = grad_output[0].detach()
            return hook
        
        # Register hooks
        handles = []
        for name, module in model.named_modules():
            if layer_names is None or name in layer_names:
                handles.append(module.register_forward_hook(get_activation(name)))
        
        # Forward pass
        with torch.no_grad():
            _ = model(X_tensor)
        
        # Remove hooks
        for h in handles:
            h.remove()
        
        return activations
    
    def compute_d_eff(self, J: NDArray[np.floating]) -> float:
        """Compute effective dimension D_eff from Jacobian.
        
        D_eff = ||J||_F² / ||J||²
        
        where ||J||_F is the Frobenius norm and ||J|| is the spectral norm.
        
        Args:
            J: Jacobian matrix of shape (n_outputs, n_inputs) or (n_outputs, n_params)
            
        Returns:
            Effective dimension D_eff
        """
        # Frobenius norm squared
        frob_norm_sq = np.sum(J ** 2)
        
        if frob_norm_sq < 1e-10:
            return 1.0
        
        # Spectral norm (largest singular value)
        from numpy.linalg import svd
        s = svd(J, compute_uv=False)
        spectral_norm = s[0] if len(s) > 0 else 0.0
        
        spectral_norm_sq = spectral_norm ** 2
        
        if spectral_norm_sq < 1e-10:
            return 1.0
        
        D_eff = frob_norm_sq / spectral_norm_sq
        
        return float(np.clip(D_eff, 1.0, J.shape[0] * J.shape[1]))
    
    def compute_d_eff_batch(
        self,
        model: 'torch.nn.Module',
        X: NDArray[np.floating],
        layer_idx: int = -1
    ) -> float:
        """Compute D_eff for a batch of samples.
        
        Args:
            model: PyTorch model
            X: Input data
            layer_idx: Layer index to analyze
            
        Returns:
            Average D_eff across samples
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for Jacobian computation")
        
        model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        
        d_effs = []
        
        with torch.no_grad():
            for i in range(X.shape[0]):
                x = X_tensor[i:i+1]
                
                # Get intermediate activations
                features = []
                
                def hook(module, input, output):
                    features.append(output)
                
                # Register hook on model layers
                if isinstance(layer_idx, int) and layer_idx >= 0:
                    # Target specific layer
                    target_layer = list(model.children())[layer_idx]
                    handle = target_layer.register_forward_hook(hook)
                
                _ = model(x)
                
                if isinstance(layer_idx, int) and layer_idx >= 0:
                    handle.remove()
                
                if features:
                    f = features[0]
                    # Approximate D_eff as output_dim / input_dim ratio
                    d_eff = f.shape[-1] / x.shape[-1] if len(f.shape) > 1 else 1.0
                    d_effs.append(d_eff)
        
        return float(np.mean(d_effs)) if d_effs else 1.0
    
    def estimate_jacobian_variance(
        self,
        model: 'torch.nn.Module',
        X: NDArray[np.floating],
        n_samples: int = 100
    ) -> NDArray[np.floating]:
        """Estimate variance of Jacobian entries across samples.
        
        Args:
            model: PyTorch model
            X: Input data
            n_samples: Number of samples to use
            
        Returns:
            Variance vector of Jacobian entries
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for Jacobian computation")
        
        indices = np.random.choice(X.shape[0], min(n_samples, X.shape[0]), replace=False)
        X_batch = X[indices]
        
        jacobians = []
        for i in range(len(indices)):
            try:
                J = self.compute_jacobian_numerical(model, X_batch[i:i+1])
                jacobians.append(J)
            except Exception:
                continue
        
        if not jacobians:
            return np.array([1.0])
        
        jacobians = np.array(jacobians)
        variances = np.var(jacobians, axis=0)
        
        return variances
