"""
Architecture representation for SU-HBO.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class ArchConfig:
    """Configuration for a neural network architecture."""
    name: str
    width: int
    depth: int
    skip: bool = False
    norm: str = 'none'  # 'none', 'bn', 'ln', 'gn'

    def __str__(self):
        parts = [f"W{self.width}", f"D{self.depth}"]
        if self.skip:
            parts.append("Skip")
        if self.norm != 'none':
            parts.append(self.norm.upper())
        return "-".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'width': self.width,
            'depth': self.depth,
            'skip': self.skip,
            'norm': self.norm,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ArchConfig':
        return cls(**d)


class Architecture:
    """Represents a trainable neural network architecture."""

    def __init__(self, config: ArchConfig):
        self.config = config
        self.j_topo: Optional[float] = None
        self.gamma: Optional[float] = None
        self.beta: Optional[float] = None
        self.loss_history = []
        self.trained_epochs = 0

    def compute_j_topo(self) -> float:
        """Compute J_topo from architecture using PI-20 approximation."""
        # J_topo = exp(-mean|log eta_l|)
        # Simplified: based on width, depth, skip

        # Base J from width/depth
        j = 0.35

        # Skip raises J (more paths = better flow)
        if self.config.skip:
            j += 0.35

        # Depth adjustment
        j += (self.config.depth - 5) * 0.03

        # Width adjustment (wider = lower J typically)
        j -= (self.config.width / 64 - 1) * 0.10

        # Norm adjustment (BN slightly raises J)
        if self.config.norm == 'bn':
            j += 0.05

        # Clip to valid range
        self.j_topo = float(max(0.05, min(0.95, j)))
        return self.j_topo

    def to_feature_vector(self) -> list:
        """Convert architecture to feature vector for GP."""
        return [
            self.j_topo if self.j_topo is not None else 0.5,
            self.gamma if self.gamma is not None else 3.0,
            self.config.width / 128.0,  # normalized
            self.config.depth / 10.0,   # normalized
            1.0 if self.config.skip else 0.0,
            1.0 if self.config.norm == 'bn' else 0.0,
            1.0 if self.config.norm == 'ln' else 0.0,
            1.0 if self.config.norm == 'gn' else 0.0,
        ]

    def __repr__(self):
        return f"Architecture({self.config})"


# Standard baselines for different tasks
def get_baseline(task_type: str = 'image') -> Architecture:
    """Get minimal baseline architecture for task type."""
    if task_type == 'image':
        config = ArchConfig(
            name="baseline",
            width=32,
            depth=3,
            skip=False,
            norm='none'
        )
    elif task_type == 'language':
        config = ArchConfig(
            name="baseline",
            width=64,
            depth=2,
            skip=False,
            norm='ln'
        )
    else:
        config = ArchConfig(
            name="baseline",
            width=64,
            depth=3,
            skip=False,
            norm='none'
        )
    return Architecture(config)
