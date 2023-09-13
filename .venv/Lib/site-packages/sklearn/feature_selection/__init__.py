"""
The :mod:`sklearn.feature_selection` module implements feature selection
algorithms. It currently includes univariate filter selection methods and the
recursive feature elimination algorithm.
"""

from ._base import SelectorMixin
from ._from_model import SelectFromModel
from ._mutual_info import mutual_info_classif, mutual_info_regression
from ._rfe import RFE, RFECV
from ._sequential import SequentialFeatureSelector
from ._univariate_selection import (
    GenericUnivariateSelect,
    SelectFdr,
    SelectFpr,
    SelectFwe,
    SelectKBest,
    SelectPercentile,
    chi2,
    f_classif,
    f_oneway,
    f_regression,
    r_regression,
)
from ._variance_threshold import VarianceThreshold

__all__ = [
    "GenericUnivariateSelect",
    "SequentialFeatureSelector",
    "RFE",
    "RFECV",
    "SelectFdr",
    "SelectFpr",
    "SelectFwe",
    "SelectKBest",
    "SelectFromModel",
    "SelectPercentile",
    "VarianceThreshold",
    "chi2",
    "f_classif",
    "f_oneway",
    "f_regression",
    "r_regression",
    "mutual_info_classif",
    "mutual_info_regression",
    "SelectorMixin",
]
