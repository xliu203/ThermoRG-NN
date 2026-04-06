"""
Plateau detection for SU-HBO.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PlateauConfig:
    """Configuration for plateau detection."""
    # Window size for rolling statistics
    window_size: int = 10

    # Thresholds for plateau detection
    beta_var_threshold: float = 0.01
    gamma_var_threshold: float = 0.05
    loss_improvement_threshold: float = 0.001

    # Minimum epochs before triggering
    min_epochs: int = 20


class PlateauDetector:
    """
    Detects training plateaus based on β and γ evolution.
    """

    def __init__(self, config: Optional[PlateauConfig] = None):
        """
        Initialize plateau detector.

        Args:
            config: Plateau detection configuration
        """
        self.config = config or PlateauConfig()

        self.beta_history: List[float] = []
        self.gamma_history: List[float] = []
        self.loss_history: List[float] = []
        self.epoch_history: List[int] = []

    def update(self, epoch: int, beta: float, gamma: float, loss: float):
        """
        Update with new observation.

        Args:
            epoch: Current epoch number
            beta: Current beta estimate
            gamma: Current gamma estimate
            loss: Current loss value
        """
        self.beta_history.append(beta)
        self.gamma_history.append(gamma)
        self.loss_history.append(loss)
        self.epoch_history.append(epoch)

    def is_plateau(self) -> bool:
        """
        Detect if training is in a plateau.

        Returns:
            True if plateau detected, False otherwise
        """
        n = len(self.beta_history)
        if n < self.config.window_size:
            return False

        if self.epoch_history[-1] < self.config.min_epochs:
            return False

        # Get recent window
        start = max(0, n - self.config.window_size)
        recent_beta = self.beta_history[start:]
        recent_gamma = self.gamma_history[start:]
        recent_loss = self.loss_history[start:]

        # Check variance of beta and gamma
        beta_var = np.var(recent_beta)
        gamma_var = np.var(recent_gamma)

        # Check loss improvement rate
        if len(recent_loss) >= 2:
            initial_loss = recent_loss[0]
            final_loss = recent_loss[-1]
            if initial_loss > 0:
                improvement_rate = (initial_loss - final_loss) / initial_loss
            else:
                improvement_rate = 0.0
        else:
            improvement_rate = 1.0

        # Plateau if:
        # 1. Beta and gamma are stable (low variance)
        # 2. Loss improvement is slow
        beta_stable = beta_var < self.config.beta_var_threshold
        gamma_stable = gamma_var < self.config.gamma_var_threshold
        loss_slow = improvement_rate < self.config.loss_improvement_threshold

        return beta_stable and gamma_stable and loss_slow

    def get_phase(self) -> str:
        """
        Get current training phase.

        Returns:
            'rapid', 'slow', or 'plateau'
        """
        n = len(self.beta_history)
        if n < 5:
            return 'unknown'

        # Detect phase from beta trend
        recent_beta = self.beta_history[-min(20, n):]

        if len(recent_beta) >= 2:
            # Fit trend
            x = np.arange(len(recent_beta))
            slope = np.polyfit(x, recent_beta, 1)[0]

            if slope > 0.01:
                return 'rapid'
            elif slope > -0.005:
                return 'slow'
            else:
                return 'plateau'

        return 'unknown'

    def reset(self):
        """Reset history."""
        self.beta_history.clear()
        self.gamma_history.clear()
        self.loss_history.clear()
        self.epoch_history.clear()


class AdaptivePlateauDetector:
    """
    Plateau detector with adaptive thresholds.
    """

    def __init__(self, base_config: Optional[PlateauConfig] = None):
        self.base_config = base_config or PlateauConfig()
        self.detector = PlateauDetector(self.base_config)
        self.stable_count = 0
        self.unstable_count = 0

    def update(self, epoch: int, beta: float, gamma: float, loss: float):
        """Update with new observation."""
        was_plateau = self.is_plateau()
        self.detector.update(epoch, beta, gamma, loss)
        is_plateau_now = self.is_plateau()

        # Track plateau state changes
        if is_plateau_now:
            self.stable_count += 1
            self.unstable_count = 0
        else:
            self.unstable_count += 1
            self.stable_count = 0

    def is_plateau(self) -> bool:
        """Check if currently in plateau."""
        return self.detector.is_plateau()

    def should_trigger_search(self) -> bool:
        """
        Determine if architecture search should be triggered.

        Returns:
            True if should trigger search
        """
        # Trigger if plateau for several consecutive checks
        return self.stable_count >= 3

    def get_phase(self) -> str:
        """Get current phase."""
        return self.detector.get_phase()

    def reset(self):
        """Reset."""
        self.detector.reset()
        self.stable_count = 0
        self.unstable_count = 0
