"""
The :mod:`imblearn.ensemble` module include methods generating
under-sampled subsets combined inside an ensemble.
"""

from ._bagging import BalancedBaggingClassifier
from ._easy_ensemble import EasyEnsembleClassifier
from ._forest import BalancedRandomForestClassifier
from ._weight_boosting import RUSBoostClassifier

__all__ = [
    "BalancedBaggingClassifier",
    "BalancedRandomForestClassifier",
    "EasyEnsembleClassifier",
    "RUSBoostClassifier",
]
