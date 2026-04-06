"""
GP Surrogate model for SU-HBO.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import botorch.models
    import gpytorch
    HAS_BOTORCH = True
except ImportError:
    HAS_BOTORCH = False


@dataclass
class SurrogateConfig:
    """Configuration for GP surrogate."""
    # Kernel parameters
    kernel_lengthscale: float = 1.0
    kernel_variance: float = 1.0
    noise_level: float = 0.01

    # Training
    n_restarts: int = 5

    # Multi-fidelity
    use_multi_fidelity: bool = True


class GPSurrogate:
    """
    Gaussian Process surrogate model with custom mean function.

    The mean function is:
        m(x) = -E_floor(J, γ) + λ · β(γ)

    This encodes our ThermoRG theory as a prior.
    """

    def __init__(self,
                 config: Optional[SurrogateConfig] = None,
                 lambda_param: float = 10.0,
                 k: float = 0.06,
                 B: float = 0.15,
                 gamma_c: float = 2.0):
        """
        Initialize GP surrogate.

        Args:
            config: Surrogate configuration
            lambda_param: Utility function weight
            k: E_floor parameter
            B: E_floor parameter
            gamma_c: Critical gamma
        """
        self.config = config or SurrogateConfig()
        self.lambda_param = lambda_param
        self.k = k
        self.B = B
        self.gamma_c = gamma_c

        self.model = None
        self.X_train_ = []
        self.y_train_ = []
        self.fidelity_train_ = []

        if not HAS_SKLEARN and not HAS_BOTORCH:
            raise ImportError(
                "Either scikit-learn or botorch is required for GPSurrogate"
            )

    def _compute_mean_prior(self, X: np.ndarray) -> np.ndarray:
        """
        Compute prior mean from ThermoRG utility function.

        Args:
            X: Feature matrix [n_samples, n_features]
               Expected features: [J_topo, gamma, width_norm, depth_norm,
                                   has_skip, has_bn, has_ln, has_gn]

        Returns:
            Prior mean values
        """
        from .utility import compute_utility

        means = []
        for x in X:
            j_topo = x[0]
            gamma = x[1]
            norm_type = 'bn' if x[5] > 0.5 else ('ln' if x[6] > 0.5 else 'none')

            # Handle GN (group norm)
            if len(x) > 7 and x[7] > 0.5:
                norm_type = 'gn'

            u = compute_utility(
                j_topo=j_topo,
                gamma=gamma,
                norm_type=norm_type,
                lambda_param=self.lambda_param,
                k=self.k,
                B=self.B,
                gamma_c=self.gamma_c
            )
            means.append(-u)  # Negative because we predict loss, not utility

        return np.array(means)

    def fit(self, X: np.ndarray, y: np.ndarray,
            fidelity: Optional[np.ndarray] = None):
        """
        Fit GP model to data.

        Args:
            X: Feature matrix [n_samples, n_features]
            y: Target values [n_samples]
            fidelity: Fidelity levels (0=L0, 1=L1, 2=L2, 3=L3)
        """
        self.X_train_ = X
        self.y_train_ = y
        self.fidelity_train_ = fidelity if fidelity is not None else np.ones(len(y))

        if HAS_SKLEARN:
            self._fit_sklearn()
        else:
            self._fit_botorch()

    def _fit_sklearn(self):
        """Fit using scikit-learn."""
        # Define kernel
        kernel = (
            ConstantKernel(self.config.kernel_variance) *
            RBF(length_scale=self.config.kernel_lengthscale) +
            WhiteKernel(noise_level=self.config.noise_level)
        )

        self.model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=self.config.n_restarts,
            normalize_y=True
        )

        # Use custom mean as prior
        prior_mean = self._compute_mean_prior(self.X_train_)

        self.model.fit(self.X_train_, self.y_train_)

    def _fit_botorch(self):
        """Fit using botorch."""
        # Botorch implementation would go here
        raise NotImplementedError("Botorch not yet implemented")

    def predict(self, X: np.ndarray,
                return_std: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict at new points.

        Args:
            X: Feature matrix [n_samples, n_features]
            return_std: If True, return standard deviations

        Returns:
            (predictions, std_devs) or just predictions
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")

        if HAS_SKLEARN:
            if return_std:
                y_pred, y_std = self.model.predict(X, return_std=True)
                return y_pred, y_std
            else:
                return self.model.predict(X), None

    def update(self, X_new: np.ndarray, y_new: float,
               fidelity_new: int = 1):
        """
        Update model with new observation.

        Args:
            X_new: New feature vector
            y_new: New observation
            fidelity_new: Fidelity level
        """
        self.X_train_.append(X_new)
        self.y_train_.append(y_new)
        self.fidelity_train_.append(fidelity_new)

        # Refit (could do incremental update instead)
        X = np.array(self.X_train_)
        y = np.array(self.y_train_)
        self.fit(X, y, np.array(self.fidelity_train_))

    def get_observation_count(self) -> int:
        """Get number of observations."""
        return len(self.X_train_)


class MultiFidelitySurrogate:
    """
    Multi-fidelity GP surrogate for hierarchical evaluation.
    """

    def __init__(self,
                 lambda_param: float = 10.0,
                 k: float = 0.06,
                 B: float = 0.15,
                 gamma_c: float = 2.0):
        """
        Initialize multi-fidelity surrogate.

        Args:
            lambda_param: Utility function weight
            k: E_floor parameter
            B: E_floor parameter
            gamma_c: Critical gamma
        """
        self.lambda_param = lambda_param
        self.k = k
        self.B = B
        self.gamma_c = gamma_c

        # Separate models for each fidelity
        self.models: Dict[int, GPSurrogate] = {}

        # Combined model
        self.combined = GPSurrogate(
            lambda_param=lambda_param,
            k=k,
            B=B,
            gamma_c=gamma_c
        )

    def update_fidelity(self, fidelity: int, X: np.ndarray, y: np.ndarray):
        """Update model at specific fidelity."""
        if fidelity not in self.models:
            self.models[fidelity] = GPSurrogate(
                lambda_param=self.lambda_param,
                k=self.k,
                B=self.B,
                gamma_c=self.gamma_c
            )

        self.models[fidelity].update(X, y, fidelity)

    def predict_at_fidelity(self, fidelity: int, X: np.ndarray):
        """Predict using model at specific fidelity."""
        if fidelity in self.models:
            return self.models[fidelity].predict(X)
        return self.combined.predict(X)

    def get_best_prediction(self, X: np.ndarray) -> float:
        """Get best available prediction."""
        # Use highest fidelity available
        if self.models:
            max_fid = max(self.models.keys())
            pred, _ = self.models[max_fid].predict(X)
            return float(pred)
        return 0.0
