"""
Action library for architecture modifications.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
from enum import Enum

from .architecture import ArchConfig, Architecture


class ActionType(Enum):
    """Types of architecture modifications."""
    ADD_NORM = "add_norm"
    REMOVE_NORM = "remove_norm"
    ADD_SKIP = "add_skip"
    REMOVE_SKIP = "remove_skip"
    INCREASE_WIDTH = "increase_width"
    DECREASE_WIDTH = "decrease_width"
    INCREASE_DEPTH = "increase_depth"
    DECREASE_DEPTH = "decrease_depth"


@dataclass
class Action:
    """Represents an architecture modification."""
    name: str
    action_type: ActionType
    description: str

    # Effect on J_topo (delta)
    delta_j: float

    # Effect on gamma (delta)
    delta_gamma: float

    # Computational cost (relative to baseline)
    cost: float = 1.0

    # Parameters for width/depth changes
    width_delta: int = 0
    depth_delta: int = 0

    def apply(self, config: ArchConfig) -> ArchConfig:
        """Apply this action to an architecture config."""
        if self.action_type == ActionType.ADD_NORM:
            new_norm = 'bn' if config.norm == 'none' else config.norm
            return ArchConfig(
                name=f"{config.name}+{self.name}",
                width=config.width,
                depth=config.depth,
                skip=config.skip,
                norm=new_norm
            )

        elif self.action_type == ActionType.REMOVE_NORM:
            return ArchConfig(
                name=config.name.replace('+bn', '').replace('+ln', ''),
                width=config.width,
                depth=config.depth,
                skip=config.skip,
                norm='none'
            )

        elif self.action_type == ActionType.ADD_SKIP:
            return ArchConfig(
                name=f"{config.name}+Skip",
                width=config.width,
                depth=config.depth,
                skip=True,
                norm=config.norm
            )

        elif self.action_type == ActionType.REMOVE_SKIP:
            return ArchConfig(
                name=config.name.replace('+Skip', ''),
                width=config.width,
                depth=config.depth,
                skip=False,
                norm=config.norm
            )

        elif self.action_type == ActionType.INCREASE_WIDTH:
            return ArchConfig(
                name=f"{config.name}",
                width=min(256, config.width + self.width_delta),
                depth=config.depth,
                skip=config.skip,
                norm=config.norm
            )

        elif self.action_type == ActionType.DECREASE_WIDTH:
            return ArchConfig(
                name=f"{config.name}",
                width=max(8, config.width - self.width_delta),
                depth=config.depth,
                skip=config.skip,
                norm=config.norm
            )

        elif self.action_type == ActionType.INCREASE_DEPTH:
            return ArchConfig(
                name=f"{config.name}",
                width=config.width,
                depth=min(20, config.depth + self.depth_delta),
                skip=config.skip,
                norm=config.norm
            )

        elif self.action_type == ActionType.DECREASE_DEPTH:
            return ArchConfig(
                name=f"{config.name}",
                width=config.width,
                depth=max(2, config.depth - self.depth_delta),
                skip=config.skip,
                norm=config.norm
            )

        return config


class ActionLibrary:
    """
    Library of available architecture modifications.
    """

    # Default action effects (calibrated from Phase A)
    DEFAULT_EFFECTS = {
        'add_bn': {'delta_j': 0.05, 'delta_gamma': -1.0},
        'add_ln': {'delta_j': 0.02, 'delta_gamma': -0.7},
        'remove_norm': {'delta_j': -0.05, 'delta_gamma': 0.8},
        'add_skip': {'delta_j': 0.35, 'delta_gamma': -0.2},
        'remove_skip': {'delta_j': -0.35, 'delta_gamma': 0.2},
        'increase_width': {'delta_j': -0.15, 'delta_gamma': 0.0},
        'decrease_width': {'delta_j': 0.15, 'delta_gamma': 0.0},
        'increase_depth': {'delta_j': 0.04, 'delta_gamma': 0.1},
        'decrease_depth': {'delta_j': -0.04, 'delta_gamma': -0.1},
    }

    def __init__(self, effects: Optional[Dict] = None):
        """
        Initialize action library.

        Args:
            effects: Dictionary of action effects. If None, uses defaults.
        """
        self.effects = effects or self.DEFAULT_EFFECTS

    def get_available_actions(self, config: ArchConfig) -> List[Action]:
        """
        Get list of valid actions for current architecture.

        Args:
            config: Current architecture configuration

        Returns:
            List of available actions
        """
        actions = []

        # Norm modifications
        if config.norm == 'none':
            actions.append(Action(
                name='add_bn',
                action_type=ActionType.ADD_NORM,
                description='Add BatchNorm (strong cooling)',
                delta_j=self.effects['add_bn']['delta_j'],
                delta_gamma=self.effects['add_bn']['delta_gamma'],
            ))
            actions.append(Action(
                name='add_ln',
                action_type=ActionType.ADD_NORM,
                description='Add LayerNorm (moderate cooling)',
                delta_j=self.effects['add_ln']['delta_j'],
                delta_gamma=self.effects['add_ln']['delta_gamma'],
            ))
        elif config.norm in ['bn', 'ln']:
            actions.append(Action(
                name='remove_norm',
                action_type=ActionType.REMOVE_NORM,
                description='Remove normalization',
                delta_j=self.effects['remove_norm']['delta_j'],
                delta_gamma=self.effects['remove_norm']['delta_gamma'],
            ))

        # Skip connections
        if not config.skip:
            actions.append(Action(
                name='add_skip',
                action_type=ActionType.ADD_SKIP,
                description='Add skip connection',
                delta_j=self.effects['add_skip']['delta_j'],
                delta_gamma=self.effects['add_skip']['delta_gamma'],
            ))
        else:
            actions.append(Action(
                name='remove_skip',
                action_type=ActionType.REMOVE_SKIP,
                description='Remove skip connection',
                delta_j=self.effects['remove_skip']['delta_j'],
                delta_gamma=self.effects['remove_skip']['delta_gamma'],
            ))

        # Width modifications
        if config.width < 128:
            actions.append(Action(
                name='increase_width',
                action_type=ActionType.INCREASE_WIDTH,
                description='Increase width (reduce J_topo)',
                delta_j=self.effects['increase_width']['delta_j'],
                delta_gamma=self.effects['increase_width']['delta_gamma'],
                width_delta=32,
            ))
        if config.width > 16:
            actions.append(Action(
                name='decrease_width',
                action_type=ActionType.DECREASE_WIDTH,
                description='Decrease width (increase J_topo)',
                delta_j=self.effects['decrease_width']['delta_j'],
                delta_gamma=self.effects['decrease_width']['delta_gamma'],
                width_delta=32,
            ))

        # Depth modifications
        if config.depth < 12:
            actions.append(Action(
                name='increase_depth',
                action_type=ActionType.INCREASE_DEPTH,
                description='Increase depth',
                delta_j=self.effects['increase_depth']['delta_j'],
                delta_gamma=self.effects['increase_depth']['delta_gamma'],
                depth_delta=2,
            ))
        if config.depth > 2:
            actions.append(Action(
                name='decrease_depth',
                action_type=ActionType.DECREASE_DEPTH,
                description='Decrease depth',
                delta_j=self.effects['decrease_depth']['delta_j'],
                delta_gamma=self.effects['decrease_depth']['delta_gamma'],
                depth_delta=2,
            ))

        return actions

    def update_effects(self, action_name: str, delta_j: float, delta_gamma: float):
        """
        Update effect for an action (after calibration).
        """
        self.effects[action_name] = {
            'delta_j': delta_j,
            'delta_gamma': delta_gamma
        }
