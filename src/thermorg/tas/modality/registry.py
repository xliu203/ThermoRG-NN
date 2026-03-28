# SPDX-License-Identifier: Apache-2.0

"""Modality registry for easy access to different data modalities."""

from typing import Dict, Type, Optional, Callable
from dataclasses import dataclass

from .base import BaseModality, ModalityConfig
from .tabular import TabularModality
from .embedding import EmbeddingModality


@dataclass
class ModalityInfo:
    """Information about a registered modality.
    
    Attributes:
        name: Modality name
        cls: Modality class
        default_config: Default configuration
        description: Human-readable description
    """
    name: str
    cls: Type[BaseModality]
    default_config: Optional[ModalityConfig] = None
    description: str = ""


class ModalityRegistry:
    """Registry for data modalities.
    
    Provides a centralized way to access and create modality instances.
    
    Example:
        >>> registry = ModalityRegistry()
        >>> registry.register('tabular', TabularModality)
        >>> modality = registry.create('tabular', scale=True)
        >>> 
        >>> # Or use convenience methods
        >>> modality = ModalityRegistry.create_tabular()
        >>> modality = ModalityRegistry.create_embedding('text', encoder='bert')
    """
    
    _registry: Dict[str, ModalityInfo] = {}
    _initialized: bool = False
    
    @classmethod
    def _init_registry(cls):
        """Initialize registry with default modalities."""
        if cls._initialized:
            return
        
        # Register default modalities
        cls._registry['tabular'] = ModalityInfo(
            name='tabular',
            cls=TabularModality,
            default_config=ModalityConfig(
                modality_type='tabular',
                distance_metric='euclidean',
                scale=True,
                normalize=False,
            ),
            description="Native tabular data with optional scaling",
        )
        
        cls._registry['text'] = ModalityInfo(
            name='text',
            cls=EmbeddingModality,
            default_config=ModalityConfig(
                modality_type='text',
                encoder_name='bert',
                distance_metric='cosine',
                normalize=True,
            ),
            description="Text data via BERT or sentence embeddings",
        )
        
        cls._registry['video'] = ModalityInfo(
            name='video',
            cls=EmbeddingModality,
            default_config=ModalityConfig(
                modality_type='video',
                encoder_name='clip',
                distance_metric='cosine',
                normalize=True,
            ),
            description="Video data via CLIP or VideoMAE embeddings",
        )
        
        cls._registry['audio'] = ModalityInfo(
            name='audio',
            cls=EmbeddingModality,
            default_config=ModalityConfig(
                modality_type='audio',
                encoder_name='wav2vec',
                distance_metric='cosine',
                normalize=True,
            ),
            description="Audio data via Wav2Vec embeddings",
        )
        
        cls._registry['image'] = ModalityInfo(
            name='image',
            cls=EmbeddingModality,
            default_config=ModalityConfig(
                modality_type='image',
                encoder_name='clip',
                distance_metric='cosine',
                normalize=True,
            ),
            description="Image data via CLIP embeddings",
        )
        
        cls._initialized = True
    
    @classmethod
    def register(
        cls,
        name: str,
        modality_cls: Type[BaseModality],
        config: Optional[ModalityConfig] = None,
        description: str = "",
    ):
        """Register a new modality.
        
        Args:
            name: Modality name
            modality_cls: Modality class
            config: Default configuration
            description: Human-readable description
        """
        cls._init_registry()
        
        cls._registry[name] = ModalityInfo(
            name=name,
            cls=modality_cls,
            default_config=config,
            description=description,
        )
    
    @classmethod
    def get(cls, name: str) -> Optional[ModalityInfo]:
        """Get modality info by name.
        
        Args:
            name: Modality name
            
        Returns:
            ModalityInfo or None if not found
        """
        cls._init_registry()
        return cls._registry.get(name)
    
    @classmethod
    def create(cls, name: str, **kwargs) -> BaseModality:
        """Create a modality instance by name.
        
        Args:
            name: Modality name
            **kwargs: Additional arguments for modality constructor
            
        Returns:
            Modality instance
            
        Raises:
            ValueError: If modality not found
        """
        cls._init_registry()
        
        info = cls._registry.get(name)
        if info is None:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Unknown modality: {name}. Available: {available}"
            )
        
        # Merge default config with provided kwargs
        config = info.default_config or ModalityConfig()
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return info.cls(config=config, **kwargs)
    
    @classmethod
    def list_modalities(cls) -> Dict[str, str]:
        """List all registered modalities.
        
        Returns:
            Dict mapping modality name to description
        """
        cls._init_registry()
        return {name: info.description for name, info in cls._registry.items()}
    
    @classmethod
    def create_tabular(cls, **kwargs) -> TabularModality:
        """Convenience method to create tabular modality.
        
        Args:
            **kwargs: Arguments for TabularModality
            
        Returns:
            TabularModality instance
        """
        return cls.create('tabular', **kwargs)
    
    @classmethod
    def create_embedding(cls, modality: str = 'text', encoder: str = 'clip', **kwargs) -> EmbeddingModality:
        """Convenience method to create embedding modality.
        
        Args:
            modality: Modality type ('text', 'video', 'audio', 'image')
            encoder: Encoder name
            **kwargs: Additional arguments
            
        Returns:
            EmbeddingModality instance
        """
        return cls.create(modality, encoder=encoder, **kwargs)
    
    @classmethod
    def create_text(cls, encoder: str = 'bert', **kwargs) -> EmbeddingModality:
        """Convenience method to create text modality.
        
        Args:
            encoder: Encoder name ('bert', 'clip', 'sentence_transformer')
            **kwargs: Additional arguments
            
        Returns:
            EmbeddingModality instance
        """
        return cls.create('text', encoder=encoder, **kwargs)
    
    @classmethod
    def create_video(cls, encoder: str = 'clip', **kwargs) -> EmbeddingModality:
        """Convenience method to create video modality.
        
        Args:
            encoder: Encoder name
            **kwargs: Additional arguments
            
        Returns:
            EmbeddingModality instance
        """
        return cls.create('video', encoder=encoder, **kwargs)
    
    @classmethod
    def create_audio(cls, encoder: str = 'wav2vec', **kwargs) -> EmbeddingModality:
        """Convenience method to create audio modality.
        
        Args:
            encoder: Encoder name
            **kwargs: Additional arguments
            
        Returns:
            EmbeddingModality instance
        """
        return cls.create('audio', encoder=encoder, **kwargs)
