"""
ThermoRG Hierarchical Bayesian Optimization Package
================================================

A principled approach to neural architecture search using
thermodynamic theory (J_topo → E_floor) and multi-fidelity BO.

Example usage:
    from thermorg_hbo import ArchitectureSpace, GP Surrogate, EIAcquisition

    space = ArchitectureSpace()
    candidates = space.sample(100)
    gp = GPSurrogate()
    acq = EIAcquisition(gp)
"""

from .arch.encoding import Architecture, ArchitectureSpace, arch_to_features
from .surrogate.gp import GPSurrogate
from .acquisition.ei import EIAcquisition
from .sim.environment import SimulationEnv

__all__ = [
    'Architecture',
    'ArchitectureSpace',
    'arch_to_features',
    'GPSurrogate',
    'EIAcquisition',
    'SimulationEnv',
]
