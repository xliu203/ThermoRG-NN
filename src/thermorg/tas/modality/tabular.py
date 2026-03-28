# SPDX-License-Identifier: Apache-2.0

"""Tabular data modality support."""

from typing import Optional, Union
import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler

from .base import BaseModality, ModalityConfig


class TabularModality(BaseModality):
    """Native support for tabular data.
    
    Tabular data works directly with raw features. Supports optional
    scaling for normalization.
    
    Distance metric: Euclidean (default for raw features)
    
    Example:
        >>> modality = TabularModality(scale=True)
        >>> features = modality.extract_features(tabular_data)
        >>> dist = modality.compute_distance(features[0], features[1])
    """
    
    def __init__(self, config: Optional[ModalityConfig] = None, scale: bool = True):
        """Initialize tabular modality.
        
        Args:
            config: Modality configuration
            scale: Whether to scale features using StandardScaler
        """
        super().__init__(config)
        self.scale = scale
        self.scaler: Optional[StandardScaler] = None
        
        # Override distance metric for tabular
        if self.config.distance_metric == 'euclidean' or self.config.distance_metric not in ['cosine', 'euclidean']:
            self.config.distance_metric = 'euclidean'
    
    def extract_features(self, data) -> NDArray[np.floating]:
        """Extract features from tabular data.
        
        Args:
            data: Tabular data as numpy array, pandas DataFrame, or similar
                  Shape: (n_samples, n_features)
                  
        Returns:
            Feature array of shape (n_samples, n_features)
        """
        # Handle different input types
        if hasattr(data, 'values'):  # pandas DataFrame/Series
            X = np.asarray(data.values, dtype=np.float64)
        elif hasattr(data, 'numpy'):  # tensorflow tensor
            X = np.asarray(data.numpy(), dtype=np.float64)
        elif hasattr(data, 'cpu'):  # pytorch tensor
            X = np.asarray(data.cpu().numpy(), dtype=np.float64)
        else:
            X = np.asarray(data, dtype=np.float64)
        
        # Ensure 2D
        X = self._ensure_2d(X)
        
        # Scale if requested
        if self.scale:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        
        # Set embedding dimension
        self._embedding_dim = X.shape[1]
        
        return X
    
    def compute_distance(self, x1: NDArray, x2: NDArray) -> float:
        """Compute Euclidean distance between two tabular samples.
        
        Args:
            x1: First sample features
            x2: Second sample features
            
        Returns:
            Euclidean distance
        """
        return float(np.linalg.norm(x1 - x2))
    
    def transform(self, data) -> NDArray[np.floating]:
        """Transform new tabular data using fitted scaler.
        
        Args:
            data: New tabular data
            
        Returns:
            Transformed feature array
        """
        if hasattr(data, 'values'):
            X = np.asarray(data.values, dtype=np.float64)
        else:
            X = np.asarray(data, dtype=np.float64)
        
        X = self._ensure_2d(X)
        
        if self.scale and self.scaler is not None:
            X = self.scaler.transform(X)
        
        return X
