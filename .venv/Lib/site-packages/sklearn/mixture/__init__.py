"""
The :mod:`sklearn.mixture` module implements mixture modeling algorithms.
"""

from ._bayesian_mixture import BayesianGaussianMixture
from ._gaussian_mixture import GaussianMixture

__all__ = ["GaussianMixture", "BayesianGaussianMixture"]
