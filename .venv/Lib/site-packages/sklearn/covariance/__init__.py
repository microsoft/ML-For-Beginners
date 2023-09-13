"""
The :mod:`sklearn.covariance` module includes methods and algorithms to
robustly estimate the covariance of features given a set of points. The
precision matrix defined as the inverse of the covariance is also estimated.
Covariance estimation is closely related to the theory of Gaussian Graphical
Models.
"""

from ._elliptic_envelope import EllipticEnvelope
from ._empirical_covariance import (
    EmpiricalCovariance,
    empirical_covariance,
    log_likelihood,
)
from ._graph_lasso import GraphicalLasso, GraphicalLassoCV, graphical_lasso
from ._robust_covariance import MinCovDet, fast_mcd
from ._shrunk_covariance import (
    OAS,
    LedoitWolf,
    ShrunkCovariance,
    ledoit_wolf,
    ledoit_wolf_shrinkage,
    oas,
    shrunk_covariance,
)

__all__ = [
    "EllipticEnvelope",
    "EmpiricalCovariance",
    "GraphicalLasso",
    "GraphicalLassoCV",
    "LedoitWolf",
    "MinCovDet",
    "OAS",
    "ShrunkCovariance",
    "empirical_covariance",
    "fast_mcd",
    "graphical_lasso",
    "ledoit_wolf",
    "ledoit_wolf_shrinkage",
    "log_likelihood",
    "oas",
    "shrunk_covariance",
]
