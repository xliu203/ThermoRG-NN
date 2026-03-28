# SPDX-License-Identifier: Apache-2.0

"""Base modality classes for data-agnostic feature extraction."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union
import numpy as np
from numpy.typing import NDArray


@dataclass
class ModalityConfig:
    """Configuration for data modality.
    
    Attributes:
        modality_type: Type of modality ('tabular', 'text', 'video', 'audio')
        embedding_dim: Dimension of embedding space (auto-computed if None)
        encoder_name: Name of pre-trained encoder (for embedding modalities)
        scale: Whether to scale features (for tabular)
        normalize: Whether to normalize embeddings
        distance_metric: Distance metric to use ('cosine', 'euclidean')
    """
    modality_type: str = 'tabular'
    embedding_dim: Optional[int] = None
    encoder_name: Optional[str] = None
    scale: bool = True
    normalize: bool = True
    distance_metric: str = 'euclidean'


class BaseModality(ABC):
    """Abstract base class for data modality support.
    
    This class defines the interface for extracting features from different
    data modalities and computing appropriate distance metrics.
    
    Attributes:
        config: Modality configuration
        _embedding_dim: Cached embedding dimension
    """
    
    def __init__(self, config: Optional[ModalityConfig] = None):
        """Initialize base modality.
        
        Args:
            config: Modality configuration (uses defaults if None)
        """
        self.config = config or ModalityConfig()
        self._embedding_dim: Optional[int] = None
        
    @abstractmethod
    def extract_features(self, data) -> NDArray[np.floating]:
        """Extract features from raw data.
        
        Args:
            data: Raw data in native format
            
        Returns:
            Feature array of shape (n_samples, embedding_dim)
        """
        pass
    
    @abstractmethod
    def compute_distance(self, x1: NDArray, x2: NDArray) -> float:
        """Compute distance between two samples.
        
        Args:
            x1: First sample features
            x2: Second sample features
            
        Returns:
            Distance value
        """
        pass
    
    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension.
        
        Returns:
            Embedding dimension
        """
        if self._embedding_dim is not None:
            return self._embedding_dim
        if self.config.embedding_dim is not None:
            return self.config.embedding_dim
        raise NotImplementedError(
            "embedding_dim not set. Call extract_features first or set config.embedding_dim."
        )
    
    @embedding_dim.setter
    def embedding_dim(self, value: int):
        """Set the embedding dimension."""
        self._embedding_dim = value
    
    def _ensure_2d(self, x: NDArray) -> NDArray:
        """Ensure feature array is 2D.
        
        Args:
            x: Feature array
            
        Returns:
            2D array of shape (n_samples, n_features)
        """
        if x.ndim == 1:
            return x.reshape(1, -1)
        return x
