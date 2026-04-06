#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
ThermoRG - Thermodynamic Theory of Neural Architecture Scaling
==============================================================

A core library for analyzing neural network architectures using
thermodynamic principles and manifold geometry.

Core Modules:
-------------
- j_topo:    J_topo computation with stride correction
- scaling:   D-scaling law fitting and predictions
- cooling:   Cooling factor φ(γ) computation
- utils:     Common utilities (manifold estimation, etc.)

Quick Start:
-----------
    >>> from thermorg import compute_J_topo, fit_scaling_law
    >>> import torch.nn as nn
    >>> model = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1),
    ...                      nn.Conv2d(64, 128, 3, stride=2, padding=1))
    >>> J, etas = compute_J_topo(model)
    >>> print(f"J_topo = {J:.4f}")

Reference: ThermoRG Theory Framework v8
"""

from __future__ import annotations

__version__ = '0.2.0'
__author__ = 'ThermoRG Team'

# Core API
from .j_topo import (
    compute_J_topo,
    compute_D_eff,
    compute_D_eff_power_iteration,
    compute_D_eff_total,
    count_parameters,
    get_layer_weights_for_J_topo,
)

from .scaling import (
    scaling_law,
    fit_scaling_law,
    predict_loss,
    compute_optimal_temperature,
    unified_scaling_law,
)

from .cooling import (
    cooling_factor_linear,
    cooling_factor_exponential,
    cooling_factor_power_law,
    cooling_factor_cosine,
    get_cooling_factor,
    phi_cooling,
    phi_ratio_BN,
    phi_from_delta,
    phi,
)

from .utils import (
    estimate_d_manifold,
    compute_capacity_bound,
    get_layer_info,
    count_stride2_layers,
    count_maxpool_layers,
    setup_logger,
    save_results,
    load_results,
    clamp,
    safe_log,
    geometric_mean,
)

__all__ = [
    # j_topo
    'compute_J_topo',
    'compute_D_eff',
    'compute_D_eff_power_iteration',
    'compute_D_eff_total',
    'count_parameters',
    'get_layer_weights_for_J_topo',
    # scaling
    'scaling_law',
    'fit_scaling_law',
    'predict_loss',
    'compute_optimal_temperature',
    'unified_scaling_law',
    # cooling
    'cooling_factor_linear',
    'cooling_factor_exponential',
    'cooling_factor_power_law',
    'cooling_factor_cosine',
    'get_cooling_factor',
    'phi_cooling',
    'phi_ratio_BN',
    'phi_from_delta',
    'phi',
    # utils
    'estimate_d_manifold',
    'compute_capacity_bound',
    'get_layer_info',
    'count_stride2_layers',
    'count_maxpool_layers',
    'setup_logger',
    'save_results',
    'load_results',
    'clamp',
    'safe_log',
    'geometric_mean',
]
