"""Architecture specifications and FLOPs constants for CIFAR-10 Lift Test.

This module defines the 15 architectures across 4 groups:
- Group 1: Thermogeometric Optimal (4 architectures)
- Group 2: Topology Destroyer (4 architectures with bottlenecks)
- Group 3: Thermal Boiling Furnace (4 architectures - baseline ablation)
- Group 4: Traditional Baselines (3 architectures)

Each architecture is characterized by:
- name: Unique identifier
- group: Group number (1-4)
- channels: Channel dimensions for each layer
- has_skip: Whether skip connections are used
- activation: Activation function type ('gelu', 'tga', 'relu')
- has_norm: Whether layer normalization is used
- bottleneck_dim: Bottleneck dimension (None if no bottleneck)
- skip_interval: Interval for skip connections (for ThermoNet-9 style)
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class ArchSpec:
    """Specification for a single architecture."""
    name: str
    group: int
    channels: List[int]
    has_skip: bool
    activation: str  # 'gelu', 'tga', or 'relu'
    has_norm: bool
    bottleneck_dim: Optional[int] = None
    skip_interval: int = 1  # For skip_every_N style
    description: str = ""


# Group 1: Thermogeometric Optimal - optimal thermogeometric design
GROUP1_SPECS = [
    ArchSpec(
        name="ThermoNet-3",
        group=1,
        channels=[64, 64, 128, 128],
        has_skip=True,
        activation='gelu',
        has_norm=True,
        description="G1-1: Thermogeometric Optimal 3-layer with skip connections"
    ),
    ArchSpec(
        name="ThermoNet-5",
        group=1,
        channels=[64, 128, 256, 128, 64],
        has_skip=True,
        activation='gelu',
        has_norm=True,
        description="G1-2: Thermogeometric Optimal 5-layer with skip connections"
    ),
    ArchSpec(
        name="ThermoNet-7",
        group=1,
        channels=[64, 64, 128, 128, 256, 128, 64],
        has_skip=True,
        activation='tga',
        has_norm=True,
        description="G1-3: Thermogeometric Optimal 7-layer with TGA activation"
    ),
    ArchSpec(
        name="ThermoNet-9",
        group=1,
        channels=[64, 64, 64, 64, 64, 64, 64, 64],
        has_skip=True,
        activation='gelu',
        has_norm=True,
        skip_interval=2,
        description="G1-4: Thermogeometric Optimal 9-layer with skip every 2 layers"
    ),
]

# Group 2: Topology Destroyer - same params but with 8x compression bottlenecks
GROUP2_SPECS = [
    ArchSpec(
        name="ThermoBot-3",
        group=2,
        channels=[64, 64, 8, 128, 128],  # 8x bottleneck at layer 3
        has_skip=True,
        activation='gelu',
        has_norm=True,
        bottleneck_dim=8,
        description="G2-1: Topology Destroyer - ThermoNet-3 with 8x bottleneck"
    ),
    ArchSpec(
        name="ThermoBot-5",
        group=2,
        channels=[64, 128, 16, 128, 64],  # 8x bottleneck at layer 3
        has_skip=True,
        activation='gelu',
        has_norm=True,
        bottleneck_dim=16,
        description="G2-2: Topology Destroyer - ThermoNet-5 with 8x bottleneck"
    ),
    ArchSpec(
        name="ThermoBot-7",
        group=2,
        channels=[64, 64, 8, 128, 128, 16, 64],  # Two 8x bottlenecks
        has_skip=True,
        activation='tga',
        has_norm=True,
        bottleneck_dim=8,
        description="G2-3: Topology Destroyer - ThermoNet-7 with two 8x bottlenecks"
    ),
    ArchSpec(
        name="ThermoBot-9",
        group=2,
        channels=[64, 64, 64, 64, 16, 64, 64, 64],  # Bottleneck every 4 layers
        has_skip=True,
        activation='gelu',
        has_norm=True,
        bottleneck_dim=16,
        skip_interval=2,
        description="G2-4: Topology Destroyer - ThermoNet-9 with bottlenecks every 4 layers"
    ),
]

# Group 3: Thermal Boiling Furnace - ReLU, no Norm, no Skip (ablation)
GROUP3_SPECS = [
    ArchSpec(
        name="ReLUFurnace-3",
        group=3,
        channels=[64, 64, 128, 128],
        has_skip=False,
        activation='relu',
        has_norm=False,
        description="G3-1: Thermal Boiling Furnace - ThermoNet-3 structure, ReLU, no Norm, no Skip"
    ),
    ArchSpec(
        name="ReLUFurnace-5",
        group=3,
        channels=[64, 128, 256, 128, 64],
        has_skip=False,
        activation='relu',
        has_norm=False,
        description="G3-2: Thermal Boiling Furnace - ThermoNet-5 structure, ReLU, no Norm, no Skip"
    ),
    ArchSpec(
        name="ReLUFurnace-7",
        group=3,
        channels=[64, 64, 128, 128, 256, 128, 64],
        has_skip=False,
        activation='relu',
        has_norm=False,
        description="G3-3: Thermal Boiling Furnace - ThermoNet-7 structure, ReLU, no Norm, no Skip"
    ),
    ArchSpec(
        name="ReLUFurnace-9",
        group=3,
        channels=[64, 64, 64, 64, 64, 64, 64, 64],
        has_skip=False,
        activation='relu',
        has_norm=False,
        skip_interval=1,
        description="G3-4: Thermal Boiling Furnace - ThermoNet-9 structure, ReLU, no Norm, no Skip"
    ),
]

# Group 4: Traditional Baselines
GROUP4_SPECS = [
    ArchSpec(
        name="ResNet-18-CIFAR",
        group=4,
        channels=[64, 64, 128, 256, 512],  # Standard ResNet-18 channels
        has_skip=True,
        activation='relu',
        has_norm=True,
        skip_interval=2,
        description="G4-1: ResNet-18 adapted for CIFAR-10 (3x3 conv, no maxpool)"
    ),
    ArchSpec(
        name="VGG-11-CIFAR",
        group=4,
        channels=[64, 128, 256, 256, 512, 512],  # VGG-11 with reduced channels
        has_skip=False,
        activation='relu',
        has_norm=True,
        description="G4-2: VGG-11 adapted for CIFAR-10 (reduced channels)"
    ),
    ArchSpec(
        name="DenseNet-40-CIFAR",
        group=4,
        channels=[32, 32, 32, 32],  # DenseNet-40 with growth rate 12
        has_skip=True,
        activation='relu',
        has_norm=True,
        skip_interval=1,
        description="G4-3: DenseNet-40 adapted for CIFAR-10 (growth rate 12)"
    ),
]

# All architectures combined
ALL_SPECS = GROUP1_SPECS + GROUP2_SPECS + GROUP3_SPECS + GROUP4_SPECS

# Architecture name to spec mapping
ARCH_MAP = {spec.name: spec for spec in ALL_SPECS}


# Training hyperparameters
PHASE_A_EPOCHS = 30
PHASE_B_EPOCHS = 150
PHASE_B_ARCHITECTURES = [
    "ThermoNet-3", "ThermoNet-7",  # Top 2 from Phase A
    "ReLUFurnace-3", "ReLUFurnace-7",  # Bottom 2 from Phase A
    "ResNet-18-CIFAR"  # G4-1 baseline
]

# CIFAR-10 constants
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR10_INPUT_SIZE = (3, 32, 32)
CIFAR10_NUM_CLASSES = 10

# Default training hyperparameters
DEFAULT_BATCH_SIZE = 128
DEFAULT_LR = 0.001
DEFAULT_WEIGHT_DECAY = 1e-4


def get_group_name(group: int) -> str:
    """Get group name from group number."""
    names = {
        1: "Thermogeometric Optimal",
        2: "Topology Destroyer",
        3: "Thermal Boiling Furnace",
        4: "Traditional Baselines"
    }
    return names.get(group, f"Group {group}")


def get_architectures_by_group(group: int) -> List[ArchSpec]:
    """Get all architecture specs for a given group."""
    return [spec for spec in ALL_SPECS if spec.group == group]
