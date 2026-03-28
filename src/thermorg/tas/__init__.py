# SPDX-License-Identifier: Apache-2.0

"""Thermogeometric Architecture Search (TAS) module.

This module implements the full TAS pipeline for predicting optimal
neural network architectures based on thermodynamic principles.
"""

from .profiling import ManifoldEstimator, SmoothnessEstimator
from .architecture import ArchitectureAnalyzer, JacobianAnalyzer
from .thermodynamics import TemperatureEstimator, ThermalPhaseComputer, CoolingPhaseComputer
from .optimization import ArchitectureSearcher, ArchitectureConfig, ConstraintBounds, SearchResult
from .predictor import TASProfiler, TASConfig, TASResult, OptimalityResult
from .predictor import (
    compute_epsilon_coupling,
    check_c1_topological_isometry,
    check_c2_thermal_safety,
    is_thermogeometrically_feasible,
)

# Modality support
from .modality import (
    BaseModality,
    ModalityConfig,
    TabularModality,
    EmbeddingModality,
    ModalityRegistry,
)

# Module support
from .modules import (
    BaseModule,
    ModuleConfig,
    PresetModule,
    ModuleRegistry,
    CustomModule,
    TGAActivation,
    GELU,
    Swish,
    get_activation,
    ACTIVATION_REGISTRY,
)

__all__ = [
    # Profiling
    'ManifoldEstimator',
    'SmoothnessEstimator',
    # Architecture
    'ArchitectureAnalyzer',
    'JacobianAnalyzer',
    # Thermodynamics
    'TemperatureEstimator',
    'ThermalPhaseComputer',
    'CoolingPhaseComputer',
    # Optimization
    'ArchitectureSearcher',
    'ArchitectureConfig',
    'ConstraintBounds',
    'SearchResult',
    # Main
    'TASProfiler',
    'TASConfig',
    'TASResult',
    'OptimalityResult',
    # Phase 6 Functions
    'compute_epsilon_coupling',
    'check_c1_topological_isometry',
    'check_c2_thermal_safety',
    'is_thermogeometrically_feasible',
    # Modality
    'BaseModality',
    'ModalityConfig',
    'TabularModality',
    'EmbeddingModality',
    'ModalityRegistry',
    # Modules
    'BaseModule',
    'ModuleConfig',
    'PresetModule',
    'ModuleRegistry',
    'CustomModule',
    'TGAActivation',
    'GELU',
    'Swish',
    'get_activation',
    'ACTIVATION_REGISTRY',
]
