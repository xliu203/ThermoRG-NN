"""
SU-HBO: Stepwise Utility-guided Hierarchical Bayesian Optimization
============================================================

Main algorithm class.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
import numpy as np
import logging

from .architecture import Architecture, ArchConfig, get_baseline
from .action_library import ActionLibrary, Action
from .utility import (
    compute_utility, compute_e_floor, compute_beta,
    DEFAULT_K, DEFAULT_B, DEFAULT_GAMMA_C, DEFAULT_LAMBDA
)
from .plateau import PlateauDetector, PlateauConfig
from .surrogate import GPSurrogate, SurrogateConfig, MultiFidelitySurrogate
from .acquisition import AcquisitionFunction, expected_improvement, select_best_candidate


@dataclass
class SUHBOConfig:
    """Configuration for SU-HBO."""
    # Utility function parameters
    lambda_param: float = DEFAULT_LAMBDA
    k: float = DEFAULT_K
    B: float = DEFAULT_B
    gamma_c: float = DEFAULT_GAMMA_C

    # Plateau detection
    plateau_config: PlateauConfig = field(default_factory=PlateauConfig)

    # Surrogate
    surrogate_config: SurrogateConfig = field(default_factory=SurrogateConfig)

    # Acquisition
    acquisition: str = 'ei'  # 'ei', 'ucb', 'pi'
    exploration_param: float = 0.01

    # Budget
    max_iterations: int = 50
    max_evaluations: int = 100

    # Thresholds
    utility_threshold: float = 0.01
    no_improvement_limit: int = 3

    # Fidelity levels
    eval_L1_epochs: int = 5
    eval_L2_epochs: int = 50
    eval_L3_epochs: int = 200
    eval_L3_interval: int = 5


class SUHBO:
    """
    Stepwise Utility-guided Hierarchical Bayesian Optimization.

    Main algorithm that combines:
    - Stepwise architecture composition
    - Utility function based on ThermoRG theory
    - Multi-fidelity Bayesian optimization
    """

    def __init__(self, config: Optional[SUHBOConfig] = None):
        """
        Initialize SU-HBO.

        Args:
            config: Algorithm configuration
        """
        self.config = config or SUHBOConfig()

        # Initialize components
        self.action_library = ActionLibrary()
        self.plateau_detector = PlateauDetector(self.config.plateau_config)
        self.surrogate = GPSurrogate(
            config=self.config.surrogate_config,
            lambda_param=self.config.lambda_param,
            k=self.config.k,
            B=self.config.B,
            gamma_c=self.config.gamma_c
        )
        self.acquisition_func = AcquisitionFunction(
            name=self.config.acquisition,
            xi=self.config.exploration_param
        )

        # State
        self.current_arch: Optional[Architecture] = None
        self.best_arch: Optional[Architecture] = None
        self.best_utility: float = float('-inf')
        self.iteration: int = 0
        self.evaluation_count: int = 0
        self.no_improvement_count: int = 0

        # History
        self.history: List[Dict[str, Any]] = []

        # Logger
        self.logger = logging.getLogger(__name__)

    def initialize(self, baseline: Optional[Architecture] = None,
                   task_type: str = 'image'):
        """
        Initialize with baseline architecture.

        Args:
            baseline: Custom baseline, or None for default
            task_type: 'image', 'language', or 'other'
        """
        if baseline is None:
            baseline = get_baseline(task_type)

        self.current_arch = baseline
        self.current_arch.compute_j_topo()
        self.best_arch = baseline
        self.best_utility = float('-inf')

        self.logger.info(f"Initialized with baseline: {baseline.config}")

    def train_epoch(self, arch: Architecture,
                    epochs: int = 5,
                    data_fraction: float = 0.1) -> Dict[str, float]:
        """
        Train architecture and measure β, γ.

        This is a placeholder - actual implementation would use PyTorch.

        Args:
            arch: Architecture to train
            epochs: Number of epochs
            data_fraction: Fraction of data to use

        Returns:
            Dict with 'loss', 'beta', 'gamma'
        """
        # Placeholder - actual implementation would:
        # 1. Build model from arch.config
        # 2. Train for epochs
        # 3. Measure gamma from activation variance
        # 4. Estimate beta from loss curve

        # For now, return mock values
        return {
            'loss': 0.5,
            'beta': 0.2,
            'gamma': 3.0,
        }

    def evaluate_candidate(self, config: ArchConfig,
                          fidelity: int = 1) -> Dict[str, float]:
        """
        Evaluate a candidate architecture.

        Args:
            config: Architecture configuration
            fidelity: 1=L1 (5ep), 2=L2 (50ep), 3=L3 (200ep)

        Returns:
            Dict with metrics
        """
        arch = Architecture(config)
        arch.compute_j_topo()

        # Determine epochs based on fidelity
        epochs = {1: self.config.eval_L1_epochs,
                   2: self.config.eval_L2_epochs,
                   3: self.config.eval_L3_epochs}[fidelity]

        # Train
        metrics = self.train_epoch(arch, epochs=epochs)
        arch.gamma = metrics['gamma']
        arch.beta = metrics['beta']
        arch.trained_epochs = epochs

        # Compute utility
        utility = compute_utility(
            j_topo=arch.j_topo,
            gamma=arch.gamma,
            norm_type=config.norm,
            lambda_param=self.config.lambda_param,
            k=self.config.k,
            B=self.config.B,
            gamma_c=self.config.gamma_c
        )

        self.evaluation_count += 1

        return {
            'arch': arch,
            'utility': utility,
            'loss': metrics['loss'],
            'beta': metrics['beta'],
            'gamma': metrics['gamma'],
            'fidelity': fidelity,
            'j_topo': arch.j_topo,
        }

    def generate_candidates(self) -> List[tuple]:
        """
        Generate candidate actions.

        Returns:
            List of (action, new_config, delta_utility) tuples
        """
        if self.current_arch is None:
            raise ValueError("Not initialized")

        actions = self.action_library.get_available_actions(self.current_arch.config)
        candidates = []

        for action in actions:
            new_config = action.apply(self.current_arch.config)

            # Predict new J_topo
            new_arch = Architecture(new_config)
            j_new = new_arch.compute_j_topo()

            # Predict new gamma
            gamma_new = max(0.5, self.current_arch.gamma + action.delta_gamma)

            # Compute predicted utility
            u_new = compute_utility(
                j_topo=j_new,
                gamma=gamma_new,
                norm_type=new_config.norm,
                lambda_param=self.config.lambda_param,
                k=self.config.k,
                B=self.config.B,
                gamma_c=self.config.gamma_c
            )

            u_current = compute_utility(
                j_topo=self.current_arch.j_topo,
                gamma=self.current_arch.gamma,
                norm_type=self.current_arch.config.norm,
                lambda_param=self.config.lambda_param,
                k=self.config.k,
                B=self.config.B,
                gamma_c=self.config.gamma_c
            )

            delta_u = u_new - u_current

            candidates.append((action, new_config, j_new, gamma_new, delta_u))

        # Sort by predicted delta utility
        candidates.sort(key=lambda x: x[4], reverse=True)
        return candidates

    def select_best_candidate_acquisition(self, candidates: List) -> Optional[tuple]:
        """
        Select best candidate using acquisition function.

        Args:
            candidates: List from generate_candidates

        Returns:
            Best candidate or None
        """
        if not candidates or self.surrogate.model is None:
            # Fallback to greedy
            return candidates[0] if candidates else None

        # Get features for all candidates
        X_cand = []
        for action, config, j_new, gamma_new, delta_u in candidates:
            arch = Architecture(config)
            X_cand.append(arch.to_feature_vector())

        X_cand = np.array(X_cand)

        # Predict with GP
        mu, sigma = self.surrogate.predict(X_cand)

        # Compute acquisition values
        acq_values = expected_improvement(
            mu=mu,
            sigma=sigma,
            f_best=-self.best_utility,  # Negative because we predict loss
            xi=self.config.exploration_param
        )

        # Select best
        best_idx = np.argmax(acq_values)
        return candidates[best_idx]

    def step(self) -> bool:
        """
        Execute one iteration of SU-HBO.

        Returns:
            True if should continue, False if should stop
        """
        self.iteration += 1
        self.logger.info(f"Iteration {self.iteration}")

        # 1. Monitor current architecture
        if self.current_arch.trained_epochs < self.config.eval_L1_epochs:
            metrics = self.evaluate_candidate(self.current_arch.config, fidelity=1)
            self.current_arch.gamma = metrics['gamma']
            self.current_arch.beta = metrics['beta']
            self.plateau_detector.update(
                epoch=self.current_arch.trained_epochs,
                beta=self.current_arch.beta,
                gamma=self.current_arch.gamma,
                loss=metrics['loss']
            )

        # 2. Check for plateau
        if self.plateau_detector.is_plateau():
            self.logger.info("Plateau detected, generating candidates...")

            # Generate candidates
            candidates = self.generate_candidates()

            if candidates:
                # Select using acquisition function if available
                best = self.select_best_candidate_acquisition(candidates)

                if best is None:
                    best = candidates[0]

                action, new_config, j_new, gamma_new, delta_u = best

                self.logger.info(
                    f"Candidate: {new_config}, "
                    f"ΔU={delta_u:.4f}"
                )

                # Evaluate at L1
                metrics = self.evaluate_candidate(new_config, fidelity=1)

                # Decision
                if metrics['utility'] > self.best_utility + self.config.utility_threshold:
                    self.logger.info(f"Accepting: {new_config}")
                    self.current_arch = metrics['arch']
                    self.no_improvement_count = 0
                else:
                    self.logger.info("Not accepting - insufficient improvement")
                    self.no_improvement_count += 1

        # 3. Update best
        u_current = compute_utility(
            j_topo=self.current_arch.j_topo,
            gamma=self.current_arch.gamma,
            norm_type=self.current_arch.config.norm,
            lambda_param=self.config.lambda_param,
            k=self.config.k,
            B=self.config.B,
            gamma_c=self.config.gamma_c
        )

        if u_current > self.best_utility:
            self.best_utility = u_current
            self.best_arch = self.current_arch
            self.logger.info(f"New best: U={self.best_utility:.4f}")

        # 4. Periodic L3 evaluation
        if self.iteration % self.config.eval_L3_interval == 0:
            self.logger.info("Periodic L3 evaluation...")
            metrics = self.evaluate_candidate(self.current_arch.config, fidelity=3)
            # Update surrogate with L3 data
            self.surrogate.update(
                np.array([self.current_arch.to_feature_vector()]),
                metrics['loss'],
                fidelity_new=3
            )

        # 5. Check stopping
        if self.no_improvement_count >= self.config.no_improvement_limit:
            self.logger.info("No improvement limit reached")
            return False

        if self.iteration >= self.config.max_iterations:
            self.logger.info("Max iterations reached")
            return False

        if self.evaluation_count >= self.config.max_evaluations:
            self.logger.info("Max evaluations reached")
            return False

        return True

    def run(self, baseline: Optional[Architecture] = None,
            task_type: str = 'image') -> Architecture:
        """
        Run SU-HBO.

        Args:
            baseline: Starting architecture
            task_type: 'image', 'language', or 'other'

        Returns:
            Best architecture found
        """
        self.initialize(baseline, task_type)

        # Initial evaluation
        self.evaluate_candidate(self.current_arch.config, fidelity=1)

        # Main loop
        while self.step():
            pass

        return self.best_arch
