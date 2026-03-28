# SPDX-License-Identifier: Apache-2.0

"""Core modules for SMC theory implementation."""

from .manifold_estimator import (
    estimate_d_manifold,
    estimate_d_manifold_pca,
    estimate_d_manifold_levina,
    estimate_d_manifold_correlation,
    estimate_d_manifold_effective,
    estimate_d_manifold_spectral_decay,
)

from .smc import (
    compression_efficiency,
    compute_smc_metrics,
    spectral_momentum_operator,
)

from .jacobian import (
    activation_jacobian,
    hutchinson_estimator,
    power_iteration,
)

from .scaling import (
    unified_scaling_law,
    optimal_temperature,
)

__all__ = [
    # Estimators
    "estimate_d_manifold",
    "estimate_d_manifold_pca",
    "estimate_d_manifold_levina",
    "estimate_d_manifold_correlation",
    "estimate_d_manifold_effective",
    "estimate_d_manifold_spectral_decay",
    # SMC
    "compression_efficiency",
    "compute_smc_metrics",
    "spectral_momentum_operator",
    # Jacobian
    "activation_jacobian",
    "hutchinson_estimator",
    "power_iteration",
    # Scaling
    "unified_scaling_law",
    "optimal_temperature",
]
