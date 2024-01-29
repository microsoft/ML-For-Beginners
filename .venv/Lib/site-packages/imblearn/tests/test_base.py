"""Test for miscellaneous samplers objects."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

import numpy as np
import pytest
from scipy import sparse
from sklearn.datasets import load_iris, make_regression
from sklearn.linear_model import LinearRegression
from sklearn.utils import _safe_indexing
from sklearn.utils._testing import assert_allclose_dense_sparse, assert_array_equal
from sklearn.utils.multiclass import type_of_target

from imblearn import FunctionSampler
from imblearn.datasets import make_imbalance
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler

iris = load_iris()
X, y = make_imbalance(
    iris.data, iris.target, sampling_strategy={0: 10, 1: 25}, random_state=0
)


def test_function_sampler_reject_sparse():
    X_sparse = sparse.csr_matrix(X)
    sampler = FunctionSampler(accept_sparse=False)
    err_msg = "dense data is required"
    with pytest.raises(
        TypeError,
        match=err_msg,
    ):
        sampler.fit_resample(X_sparse, y)


@pytest.mark.parametrize(
    "X, y", [(X, y), (sparse.csr_matrix(X), y), (sparse.csc_matrix(X), y)]
)
def test_function_sampler_identity(X, y):
    sampler = FunctionSampler()
    X_res, y_res = sampler.fit_resample(X, y)
    assert_allclose_dense_sparse(X_res, X)
    assert_array_equal(y_res, y)


@pytest.mark.parametrize(
    "X, y", [(X, y), (sparse.csr_matrix(X), y), (sparse.csc_matrix(X), y)]
)
def test_function_sampler_func(X, y):
    def func(X, y):
        return X[:10], y[:10]

    sampler = FunctionSampler(func=func)
    X_res, y_res = sampler.fit_resample(X, y)
    assert_allclose_dense_sparse(X_res, X[:10])
    assert_array_equal(y_res, y[:10])


@pytest.mark.parametrize(
    "X, y", [(X, y), (sparse.csr_matrix(X), y), (sparse.csc_matrix(X), y)]
)
def test_function_sampler_func_kwargs(X, y):
    def func(X, y, sampling_strategy, random_state):
        rus = RandomUnderSampler(
            sampling_strategy=sampling_strategy, random_state=random_state
        )
        return rus.fit_resample(X, y)

    sampler = FunctionSampler(
        func=func, kw_args={"sampling_strategy": "auto", "random_state": 0}
    )
    X_res, y_res = sampler.fit_resample(X, y)
    X_res_2, y_res_2 = RandomUnderSampler(random_state=0).fit_resample(X, y)
    assert_allclose_dense_sparse(X_res, X_res_2)
    assert_array_equal(y_res, y_res_2)


def test_function_sampler_validate():
    # check that we can let a pass a regression variable by turning down the
    # validation
    X, y = make_regression()

    def dummy_sampler(X, y):
        indices = np.random.choice(np.arange(X.shape[0]), size=100)
        return _safe_indexing(X, indices), _safe_indexing(y, indices)

    sampler = FunctionSampler(func=dummy_sampler, validate=False)
    pipeline = make_pipeline(sampler, LinearRegression())
    y_pred = pipeline.fit(X, y).predict(X)

    assert type_of_target(y_pred) == "continuous"


def test_function_resampler_fit():
    # Check that the validation is bypass when calling `fit`
    # Non-regression test for:
    # https://github.com/scikit-learn-contrib/imbalanced-learn/issues/782
    X = np.array([[1, np.nan], [2, 3], [np.inf, 4]])
    y = np.array([0, 1, 1])

    def func(X, y):
        return X[:1], y[:1]

    sampler = FunctionSampler(func=func, validate=False)
    sampler.fit(X, y)
    sampler.fit_resample(X, y)
