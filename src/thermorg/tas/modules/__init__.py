# SPDX-License-Identifier: Apache-2.0

"""Architecture module support for TAS.

This module provides preset and custom neural network modules with
thermogeometric activation functions.
"""

from .base import BaseModule, ModuleConfig
from .preset import PresetModule, ModuleRegistry
from .custom import CustomModule
from .activations import (
    TGAActivation,
    GELU,
    Swish,
    get_activation,
    ACTIVATION_REGISTRY,
)

__all__ = [
    # Base
    'BaseModule',
    'ModuleConfig',
    # Preset
    'PresetModule',
    'ModuleRegistry',
    # Custom
    'CustomModule',
    # Activations
    'TGAActivation',
    'GELU',
    'Swish',
    'get_activation',
    'ACTIVATION_REGISTRY',
]
