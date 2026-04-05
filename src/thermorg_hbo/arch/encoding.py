"""
Architecture encoding for ThermoRG-HBO.
=====================================

Defines the architecture feature space and conversion to vectors.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np


@dataclass
class Architecture:
    """Neural network architecture specification.

    Attributes:
        width: Hidden dimension multiplier. Base channels = [3, 64*w, 128*w, 128*w].
        depth: Number of conv blocks (3, 5, 7, 9).
        skip: Use residual skip connections.
        norm: Normalization type: 'none', 'bn' (BatchNorm), 'ln' (LayerNorm).
    """
    width: int      # 8, 16, 32, 64
    depth: int      # 3, 5, 7, 9
    skip: bool      # True/False
    norm: str       # 'none', 'bn', 'ln'

    @property
    def n_params_M(self) -> float:
        """Approximate parameters in millions."""
        ch = [3, self.width, self.width*2, self.width*2]
        conv = sum(ch[i] * ch[i+1] * 9 for i in range(len(ch)-1))
        fc = ch[-1] * 10
        return (conv + fc) / 1e6

    @property
    def norm_onehot(self) -> np.ndarray:
        """One-hot encoding for norm type."""
        mapping = {'none': [1, 0, 0], 'bn': [0, 1, 0], 'ln': [0, 0, 1]}
        return np.array(mapping[self.norm], dtype=np.float32)

    def to_features(self) -> np.ndarray:
        """Convert to feature vector.

        Returns:
            Feature vector: [width_norm, depth_norm, skip, norm_onehot*3]
            Shape: (8,)
        """
        return np.array([
            self.width / 64.0,                              # normalized width
            self.depth / 9.0,                               # normalized depth
            float(self.skip),                                # skip connection
            *self.norm_onehot,                              # 3 one-hot
        ], dtype=np.float32)

    def __repr__(self):
        return f"Arch(w={self.width}, d={self.depth}, skip={self.skip}, norm={self.norm})"


class ArchitectureSpace:
    """Search space for architectures."""

    WIDTHS = [8, 16, 32, 64]
    DEPTHS = [3, 5, 7, 9]
    NORMS = ['none', 'bn', 'ln']

    def __init__(self, width_range=None, depth_range=None, norm_range=None):
        self.width_range = width_range or self.WIDTHS
        self.depth_range = depth_range or self.DEPTHS
        self.norm_range = norm_range or self.NORMS

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> List[Architecture]:
        """Sample n random architectures."""
        if rng is None:
            rng = np.random.default_rng()

        archs = []
        for _ in range(n):
            archs.append(Architecture(
                width=rng.choice(self.width_range),
                depth=rng.choice(self.depth_range),
                skip=rng.choice([True, False]),
                norm=rng.choice(self.norm_range),
            ))
        return archs

    def grid(self) -> List[Architecture]:
        """Generate full grid of architectures."""
        archs = []
        for w in self.width_range:
            for d in self.depth_range:
                for skip in [True, False]:
                    for norm in self.norm_range:
                        archs.append(Architecture(width=w, depth=d, skip=skip, norm=norm))
        return archs

    def to_matrix(self, archs: List[Architecture]) -> np.ndarray:
        """Convert list of architectures to feature matrix."""
        return np.stack([a.to_features() for a in archs])


def arch_to_features(archs: List[Architecture]) -> np.ndarray:
    """Convert architectures to feature matrix. Alias for ArchitectureSpace.to_matrix."""
    return np.stack([a.to_features() for a in archs])
