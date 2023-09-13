"""
The :mod:`sklearn.linear_model` module implements a variety of linear models.
"""

# See http://scikit-learn.sourceforge.net/modules/sgd.html and
# http://scikit-learn.sourceforge.net/modules/linear_model.html for
# complete documentation.

from ._base import LinearRegression
from ._bayes import ARDRegression, BayesianRidge
from ._coordinate_descent import (
    ElasticNet,
    ElasticNetCV,
    Lasso,
    LassoCV,
    MultiTaskElasticNet,
    MultiTaskElasticNetCV,
    MultiTaskLasso,
    MultiTaskLassoCV,
    enet_path,
    lasso_path,
)
from ._glm import GammaRegressor, PoissonRegressor, TweedieRegressor
from ._huber import HuberRegressor
from ._least_angle import (
    Lars,
    LarsCV,
    LassoLars,
    LassoLarsCV,
    LassoLarsIC,
    lars_path,
    lars_path_gram,
)
from ._logistic import LogisticRegression, LogisticRegressionCV
from ._omp import (
    OrthogonalMatchingPursuit,
    OrthogonalMatchingPursuitCV,
    orthogonal_mp,
    orthogonal_mp_gram,
)
from ._passive_aggressive import PassiveAggressiveClassifier, PassiveAggressiveRegressor
from ._perceptron import Perceptron
from ._quantile import QuantileRegressor
from ._ransac import RANSACRegressor
from ._ridge import Ridge, RidgeClassifier, RidgeClassifierCV, RidgeCV, ridge_regression
from ._sgd_fast import Hinge, Huber, Log, ModifiedHuber, SquaredLoss
from ._stochastic_gradient import SGDClassifier, SGDOneClassSVM, SGDRegressor
from ._theil_sen import TheilSenRegressor

__all__ = [
    "ARDRegression",
    "BayesianRidge",
    "ElasticNet",
    "ElasticNetCV",
    "Hinge",
    "Huber",
    "HuberRegressor",
    "Lars",
    "LarsCV",
    "Lasso",
    "LassoCV",
    "LassoLars",
    "LassoLarsCV",
    "LassoLarsIC",
    "LinearRegression",
    "Log",
    "LogisticRegression",
    "LogisticRegressionCV",
    "ModifiedHuber",
    "MultiTaskElasticNet",
    "MultiTaskElasticNetCV",
    "MultiTaskLasso",
    "MultiTaskLassoCV",
    "OrthogonalMatchingPursuit",
    "OrthogonalMatchingPursuitCV",
    "PassiveAggressiveClassifier",
    "PassiveAggressiveRegressor",
    "Perceptron",
    "QuantileRegressor",
    "Ridge",
    "RidgeCV",
    "RidgeClassifier",
    "RidgeClassifierCV",
    "SGDClassifier",
    "SGDRegressor",
    "SGDOneClassSVM",
    "SquaredLoss",
    "TheilSenRegressor",
    "enet_path",
    "lars_path",
    "lars_path_gram",
    "lasso_path",
    "orthogonal_mp",
    "orthogonal_mp_gram",
    "ridge_regression",
    "RANSACRegressor",
    "PoissonRegressor",
    "GammaRegressor",
    "TweedieRegressor",
]
