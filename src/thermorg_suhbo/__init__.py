"""
SU-HBO: Stepwise Utility-guided Hierarchical Bayesian Optimization
============================================================

A ThermoRG-based algorithm for automated neural architecture design.

Usage:
    from thermorg_suhbo import SUHBO

    algo = SUHBO(
        lambda_param=10.0,
        gamma_c=2.0,
        k=0.06,
        B=0.15,
    )
    best_arch = algo.run(dataset='CIFAR10')
"""

from .su_hbo import SUHBO, SUHBOConfig
from .architecture import Architecture, ArchConfig, get_baseline
from .action_library import Action, ActionLibrary, ActionType
from .utility import (
    compute_utility, compute_e_floor, compute_beta,
    compute_cooling_factor, compute_delta_utility,
    is_stable, get_stability_margin,
    DEFAULT_K, DEFAULT_B, DEFAULT_GAMMA_C, DEFAULT_LAMBDA,
    GAMMA_BN, GAMMA_NONE, GAMMA_LN,
    BETA_BN, BETA_NONE, BETA_LN,
)
from .plateau import PlateauDetector, PlateauConfig, AdaptivePlateauDetector
from .surrogate import GPSurrogate, SurrogateConfig, MultiFidelitySurrogate
from .acquisition import (
    expected_improvement, upper_confidence_bound,
    probability_of_improvement, AcquisitionFunction, select_best_candidate
)

__version__ = '1.0.0'

__all__ = [
    # Main class
    'SUHBO',
    'SUHBOConfig',

    # Architecture
    'Architecture',
    'ArchConfig',
    'get_baseline',

    # Actions
    'Action',
    'ActionLibrary',
    'ActionType',

    # Utility
    'compute_utility',
    'compute_e_floor',
    'compute_beta',
    'compute_cooling_factor',
    'compute_delta_utility',
    'is_stable',
    'get_stability_margin',
    'DEFAULT_K',
    'DEFAULT_B',
    'DEFAULT_GAMMA_C',
    'DEFAULT_LAMBDA',
    'GAMMA_BN',
    'GAMMA_NONE',
    'GAMMA_LN',
    'BETA_BN',
    'BETA_NONE',
    'BETA_LN',

    # Plateau
    'PlateauDetector',
    'PlateauConfig',
    'AdaptivePlateauDetector',

    # Surrogate
    'GPSurrogate',
    'SurrogateConfig',
    'MultiFidelitySurrogate',

    # Acquisition
    'expected_improvement',
    'upper_confidence_bound',
    'probability_of_improvement',
    'AcquisitionFunction',
    'select_best_candidate',
]
