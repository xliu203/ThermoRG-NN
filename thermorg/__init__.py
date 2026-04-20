"""
ThermoRG: Thermodynamic Theory of Neural Architecture Scaling
============================================================

A clean 3-module framework for zero-cost architecture scoring and loss prediction.

Modules:
- topology_calculator: J_topo computation via Power Iteration
- analytical_predictor: Pure mathematical loss prediction
- calibration: Parameter calibration from training data
"""

from thermorg.topology_calculator import (
    compute_J_topo,
    compute_D_eff_power_iteration,
    compute_resblock_eff_W,
)

from thermorg.analytical_predictor import (
    AnalyticalPredictor,
    D_scaling_law,
    E_floor_decomposition,
    cooling_law,
    predict_loss,
)

__version__ = '0.1.0'

__all__ = [
    'compute_J_topo',
    'compute_D_eff_power_iteration',
    'compute_resblock_eff_W',
    'AnalyticalPredictor',
    'D_scaling_law',
    'E_floor_decomposition',
    'cooling_law',
    'predict_loss',
]