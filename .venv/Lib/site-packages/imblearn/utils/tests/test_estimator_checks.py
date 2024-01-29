import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.utils.multiclass import check_classification_targets

from imblearn.base import BaseSampler
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.utils import check_target_type as target_check
from imblearn.utils.estimator_checks import (
    check_samplers_fit,
    check_samplers_nan,
    check_samplers_one_label,
    check_samplers_preserve_dtype,
    check_samplers_sparse,
    check_samplers_string,
    check_target_type,
)


class BaseBadSampler(BaseEstimator):
    """Sampler without inputs checking."""

    _sampling_type = "bypass"

    def fit(self, X, y):
        return self

    def fit_resample(self, X, y):
        check_classification_targets(y)
        self.fit(X, y)
        return X, y


class SamplerSingleClass(BaseSampler):
    """Sampler that would sample even with a single class."""

    _sampling_type = "bypass"

    def fit_resample(self, X, y):
        return self._fit_resample(X, y)

    def _fit_resample(self, X, y):
        return X, y


class NotFittedSampler(BaseBadSampler):
    """Sampler without target checking."""

    def fit(self, X, y):
        X, y = self._validate_data(X, y)
        return self


class NoAcceptingSparseSampler(BaseBadSampler):
    """Sampler which does not accept sparse matrix."""

    def fit(self, X, y):
        X, y = self._validate_data(X, y)
        self.sampling_strategy_ = "sampling_strategy_"
        return self


class NotPreservingDtypeSampler(BaseSampler):
    _sampling_type = "bypass"

    _parameter_constraints: dict = {"sampling_strategy": "no_validation"}

    def _fit_resample(self, X, y):
        return X.astype(np.float64), y.astype(np.int64)


class IndicesSampler(BaseOverSampler):
    def _check_X_y(self, X, y):
        y, binarize_y = target_check(y, indicate_one_vs_all=True)
        X, y = self._validate_data(
            X,
            y,
            reset=True,
            dtype=None,
            force_all_finite=False,
        )
        return X, y, binarize_y

    def _fit_resample(self, X, y):
        n_max_count_class = np.bincount(y).max()
        indices = np.random.choice(np.arange(X.shape[0]), size=n_max_count_class * 2)
        return X[indices], y[indices]


def test_check_samplers_string():
    sampler = IndicesSampler()
    check_samplers_string(sampler.__class__.__name__, sampler)


def test_check_samplers_nan():
    sampler = IndicesSampler()
    check_samplers_nan(sampler.__class__.__name__, sampler)


mapping_estimator_error = {
    "BaseBadSampler": (AssertionError, "ValueError not raised by fit"),
    "SamplerSingleClass": (AssertionError, "Sampler can't balance when only"),
    "NotFittedSampler": (AssertionError, "No fitted attribute"),
    "NoAcceptingSparseSampler": (TypeError, "dense data is required"),
    "NotPreservingDtypeSampler": (AssertionError, "X dtype is not preserved"),
}


def _test_single_check(Estimator, check):
    estimator = Estimator()
    name = estimator.__class__.__name__
    err_type, err_msg = mapping_estimator_error[name]
    with pytest.raises(err_type, match=err_msg):
        check(name, estimator)


def test_all_checks():
    _test_single_check(BaseBadSampler, check_target_type)
    _test_single_check(SamplerSingleClass, check_samplers_one_label)
    _test_single_check(NotFittedSampler, check_samplers_fit)
    _test_single_check(NoAcceptingSparseSampler, check_samplers_sparse)
    _test_single_check(NotPreservingDtypeSampler, check_samplers_preserve_dtype)
