# SPDX-License-Identifier: Apache-2.0

"""Module registry for easy access to architecture modules."""

from typing import Dict, Type, Optional, List, Any
from dataclasses import dataclass

from .base import BaseModule, ModuleConfig
from .preset import PresetModule, ModuleRegistry
from .custom import CustomModule


@dataclass
class ModuleInfo:
    """Information about a registered module type.
    
    Attributes:
        name: Module name
        cls: Module class
        description: Human-readable description
        required_params: Required parameters for creation
        optional_params: Optional parameters with defaults
    """
    name: str
    cls: Type[BaseModule]
    description: str = ""
    required_params: List[str] = []
    optional_params: Dict[str, Any] = {}


class ArchitectureModuleRegistry:
    """Registry for architecture modules.
    
    Provides centralized access to all module types.
    
    Example:
        >>> registry = ArchitectureModuleRegistry()
        >>> 
        >>> # Create modules via registry
        >>> conv = registry.create('conv2d', in_channels=3, out_channels=64)
        >>> attn = registry.create('attention', num_heads=8)
        >>> 
        >>> # List available modules
        >>> print(registry.list_modules())
    """
    
    _registry: Dict[str, ModuleInfo] = {}
    _initialized: bool = False
    
    @classmethod
    def _init_registry(cls):
        """Initialize registry with default module types."""
        if cls._initialized:
            return
        
        # Preset modules
        cls._registry['conv2d'] = ModuleInfo(
            name='conv2d',
            cls=PresetModule,
            description="2D Convolutional layer",
            required_params=['in_channels', 'out_channels'],
            optional_params={'kernel_size': 3, 'stride': 1, 'padding': 1, 'activation': 'relu'},
        )
        
        cls._registry['conv1d'] = ModuleInfo(
            name='conv1d',
            cls=PresetModule,
            description="1D Convolutional layer",
            required_params=['in_channels', 'out_channels'],
            optional_params={'kernel_size': 3, 'stride': 1, 'padding': 1, 'activation': 'relu'},
        )
        
        cls._registry['attention'] = ModuleInfo(
            name='attention',
            cls=PresetModule,
            description="Multi-head self-attention",
            required_params=['num_heads'],
            optional_params={'head_dim': 64, 'dropout': 0.1, 'activation': 'relu'},
        )
        
        cls._registry['residual'] = ModuleInfo(
            name='residual',
            cls=PresetModule,
            description="Residual connection",
            required_params=[],
            optional_params={'inner': None},
        )
        
        cls._registry['pooling'] = ModuleInfo(
            name='pooling',
            cls=PresetModule,
            description="Pooling layer",
            required_params=[],
            optional_params={'pool_type': 'max', 'kernel_size': 2, 'stride': 2},
        )
        
        cls._registry['linear'] = ModuleInfo(
            name='linear',
            cls=PresetModule,
            description="Linear/Fully-connected layer",
            required_params=['in_features', 'out_features'],
            optional_params={'activation': 'relu'},
        )
        
        cls._registry['embedding'] = ModuleInfo(
            name='embedding',
            cls=PresetModule,
            description="Token embedding",
            required_params=['vocab_size', 'embed_dim'],
            optional_params={},
        )
        
        cls._registry['layernorm'] = ModuleInfo(
            name='layernorm',
            cls=PresetModule,
            description="Layer normalization",
            required_params=['normalized_shape'],
            optional_params={},
        )
        
        cls._registry['batchnorm'] = ModuleInfo(
            name='batchnorm',
            cls=PresetModule,
            description="Batch normalization",
            required_params=['num_features'],
            optional_params={},
        )
        
        cls._registry['dropout'] = ModuleInfo(
            name='dropout',
            cls=PresetModule,
            description="Dropout layer",
            required_params=[],
            optional_params={'p': 0.5},
        )
        
        cls._registry['flatten'] = ModuleInfo(
            name='flatten',
            cls=PresetModule,
            description="Flatten layer",
            required_params=[],
            optional_params={},
        )
        
        # Custom module
        cls._registry['custom'] = ModuleInfo(
            name='custom',
            cls=CustomModule,
            description="User-defined custom module",
            required_params=['module'],
            optional_params={'activation': 'tga', 't_adiab': 1.0},
        )
        
        cls._initialized = True
    
    @classmethod
    def register(
        cls,
        name: str,
        module_cls: Type[BaseModule],
        description: str = "",
        required_params: Optional[List[str]] = None,
        optional_params: Optional[Dict[str, Any]] = None,
    ):
        """Register a new module type.
        
        Args:
            name: Module name
            module_cls: Module class
            description: Human-readable description
            required_params: Required parameters
            optional_params: Optional parameters with defaults
        """
        cls._init_registry()
        
        cls._registry[name] = ModuleInfo(
            name=name,
            cls=module_cls,
            description=description,
            required_params=required_params or [],
            optional_params=optional_params or {},
        )
    
    @classmethod
    def get(cls, name: str) -> Optional[ModuleInfo]:
        """Get module info by name.
        
        Args:
            name: Module name
            
        Returns:
            ModuleInfo or None if not found
        """
        cls._init_registry()
        return cls._registry.get(name)
    
    @classmethod
    def create(cls, name: str, **kwargs) -> BaseModule:
        """Create a module instance by name.
        
        Args:
            name: Module name
            **kwargs: Module parameters
            
        Returns:
            Module instance
            
        Raises:
            ValueError: If module type not found
        """
        cls._init_registry()
        
        info = cls._registry.get(name)
        if info is None:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Unknown module type: {name}. Available: {available}"
            )
        
        # Merge optional params with kwargs
        config = {**info.optional_params, **kwargs}
        
        if info.cls == PresetModule:
            return PresetModule(name, config)
        elif info.cls == CustomModule:
            return info.cls(**config)
        else:
            return info.cls(ModuleConfig(module_type=name, **config))
    
    @classmethod
    def list_modules(cls) -> Dict[str, str]:
        """List all registered module types.
        
        Returns:
            Dict mapping module name to description
        """
        cls._init_registry()
        return {name: info.description for name, info in cls._registry.items()}
    
    @classmethod
    def get_required_params(cls, name: str) -> List[str]:
        """Get required parameters for a module type.
        
        Args:
            name: Module name
            
        Returns:
            List of required parameter names
        """
        cls._init_registry()
        info = cls._registry.get(name)
        return info.required_params if info else []
    
    @classmethod
    def get_optional_params(cls, name: str) -> Dict[str, Any]:
        """Get optional parameters for a module type.
        
        Args:
            name: Module name
            
        Returns:
            Dict of optional parameter names to defaults
        """
        cls._init_registry()
        info = cls._registry.get(name)
        return info.optional_params if info else {}


# Convenience function
def create_module(module_type: str, **kwargs) -> BaseModule:
    """Create a module by type name.
    
    Args:
        module_type: Type of module
        **kwargs: Module parameters
        
    Returns:
        Module instance
    """
    return ArchitectureModuleRegistry.create(module_type, **kwargs)
