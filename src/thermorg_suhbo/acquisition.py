"""
Acquisition functions for SU-HBO.
"""

import numpy as np
from typing import Optional, Tuple


def expected_improvement(mu: np.ndarray,
                        sigma: np.ndarray,
                        f_best: float,
                        xi: float = 0.01) -> np.ndarray:
    """
    Compute Expected Improvement acquisition function.

    EI = E[max(0, f(x) - f_best - xi)]

    For Gaussian predictive distribution:
        EI = (mu - f_best - xi) * Phi(z) + sigma * phi(z)
        where z = (mu - f_best - xi) / sigma

    Args:
        mu: Predictive mean [n_samples]
        sigma: Predictive std [n_samples]
        f_best: Best observed value
        xi: Exploration parameter (default 0.01)

    Returns:
        Expected improvement values [n_samples]
    """
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)

    # Avoid division by zero
    sigma = np.maximum(sigma, 1e-10)

    # Compute z
    z = (mu - f_best - xi) / sigma

    # Normal CDF and PDF
    from scipy.stats import norm
    Phi = norm.cdf(z)
    phi = norm.pdf(z)

    # EI formula
    ei = (mu - f_best - xi) * Phi + sigma * phi

    # Set EI to 0 where sigma is very small
    ei[sigma < 1e-10] = 0.0

    return ei


def upper_confidence_bound(mu: np.ndarray,
                          sigma: np.ndarray,
                          beta: float = 2.0) -> np.ndarray:
    """
    Compute Upper Confidence Bound acquisition function.

    UCB = mu + beta * sigma

    Args:
        mu: Predictive mean [n_samples]
        sigma: Predictive std [n_samples]
        beta: Exploration parameter (default 2.0)

    Returns:
        UCB values [n_samples]
    """
    return mu + beta * sigma


def probability_of_improvement(mu: np.ndarray,
                               sigma: np.ndarray,
                               f_best: float,
                               xi: float = 0.01) -> np.ndarray:
    """
    Compute Probability of Improvement acquisition function.

    PI = P(f(x) > f_best + xi) = Phi((mu - f_best - xi) / sigma)

    Args:
        mu: Predictive mean [n_samples]
        sigma: Predictive std [n_samples]
        f_best: Best observed value
        xi: Threshold (default 0.01)

    Returns:
        Probability values [n_samples]
    """
    sigma = np.maximum(sigma, 1e-10)
    z = (mu - f_best - xi) / sigma
    from scipy.stats import norm
    return norm.cdf(z)


def expected_improvement_per_second(mu: np.ndarray,
                                   sigma: np.ndarray,
                                   f_best: float,
                                   cost: np.ndarray,
                                   xi: float = 0.01) -> np.ndarray:
    """
    Compute cost-adjusted Expected Improvement.

    EI/s where s is the cost (e.g., training time).

    Args:
        mu: Predictive mean
        sigma: Predictive std
        f_best: Best observed value
        cost: Cost for each sample [n_samples]
        xi: Exploration parameter

    Returns:
        EI per unit cost
    """
    ei = expected_improvement(mu, sigma, f_best, xi)
    cost = np.maximum(cost, 1e-10)
    return ei / cost


class AcquisitionFunction:
    """
    Wrapper for acquisition functions.
    """

    def __init__(self, name: str = 'ei', **kwargs):
        """
        Initialize acquisition function.

        Args:
            name: 'ei', 'ucb', or 'pi'
            **kwargs: Additional parameters
        """
        self.name = name
        self.kwargs = kwargs

        if name == 'ei':
            self.func = expected_improvement
        elif name == 'ucb':
            self.func = upper_confidence_bound
        elif name == 'pi':
            self.func = probability_of_improvement
        elif name == 'ei_per_cost':
            self.func = expected_improvement_per_second
        else:
            raise ValueError(f"Unknown acquisition function: {name}")

    def __call__(self, mu: np.ndarray, sigma: np.ndarray,
                 **kwargs) -> np.ndarray:
        """
        Compute acquisition values.

        Args:
            mu: Predictive mean
            sigma: Predictive std
            **kwargs: Additional arguments

        Returns:
            Acquisition values
        """
        all_kwargs = {**self.kwargs, **kwargs}

        if self.name == 'ei':
            return self.func(mu, sigma, **all_kwargs)
        elif self.name == 'ucb':
            return self.func(mu, sigma, **all_kwargs)
        elif self.name == 'pi':
            return self.func(mu, sigma, **all_kwargs)
        elif self.name == 'ei_per_cost':
            return self.func(mu, sigma, **all_kwargs)


def select_best_candidate(acquisition_values: np.ndarray,
                          candidates: list,
                          n_select: int = 1) -> list:
    """
    Select best candidate(s) based on acquisition values.

    Args:
        acquisition_values: Acquisition function values [n_candidates]
        candidates: List of candidate objects
        n_select: Number to select

    Returns:
        Selected candidates
    """
    # Sort by acquisition value (descending)
    sorted_indices = np.argsort(acquisition_values)[::-1]
    return [candidates[i] for i in sorted_indices[:n_select]]
