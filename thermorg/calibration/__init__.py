"""
ThermoRG Calibration Module
============================

Calibrates thermodynamic equation of state parameters from observed training data.
"""

from thermorg.calibration.thermo_calibrator import (
    ThermoCalibrator,
    CalibrationResult,
    ArchitectureSpec,
    get_default_calibration_data,
    create_calibrator_and_calibrate,
)

__all__ = [
    'ThermoCalibrator',
    'CalibrationResult',
    'ArchitectureSpec',
    'get_default_calibration_data',
    'create_calibrator_and_calibrate',
]