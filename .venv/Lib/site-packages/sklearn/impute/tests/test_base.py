import numpy as np
import pytest

from sklearn.impute._base import _BaseImputer
from sklearn.impute._iterative import _assign_where
from sklearn.utils._mask import _get_mask
from sklearn.utils._testing import _convert_container, assert_allclose


@pytest.fixture
def data():
    X = np.random.randn(10, 2)
    X[::2] = np.nan
    return X


class NoFitIndicatorImputer(_BaseImputer):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return self._concatenate_indicator(X, self._transform_indicator(X))


class NoTransformIndicatorImputer(_BaseImputer):
    def fit(self, X, y=None):
        mask = _get_mask(X, value_to_mask=np.nan)
        super()._fit_indicator(mask)
        return self

    def transform(self, X, y=None):
        return self._concatenate_indicator(X, None)


class NoPrecomputedMaskFit(_BaseImputer):
    def fit(self, X, y=None):
        self._fit_indicator(X)
        return self

    def transform(self, X):
        return self._concatenate_indicator(X, self._transform_indicator(X))


class NoPrecomputedMaskTransform(_BaseImputer):
    def fit(self, X, y=None):
        mask = _get_mask(X, value_to_mask=np.nan)
        self._fit_indicator(mask)
        return self

    def transform(self, X):
        return self._concatenate_indicator(X, self._transform_indicator(X))


def test_base_imputer_not_fit(data):
    imputer = NoFitIndicatorImputer(add_indicator=True)
    err_msg = "Make sure to call _fit_indicator before _transform_indicator"
    with pytest.raises(ValueError, match=err_msg):
        imputer.fit(data).transform(data)
    with pytest.raises(ValueError, match=err_msg):
        imputer.fit_transform(data)


def test_base_imputer_not_transform(data):
    imputer = NoTransformIndicatorImputer(add_indicator=True)
    err_msg = (
        "Call _fit_indicator and _transform_indicator in the imputer implementation"
    )
    with pytest.raises(ValueError, match=err_msg):
        imputer.fit(data).transform(data)
    with pytest.raises(ValueError, match=err_msg):
        imputer.fit_transform(data)


def test_base_no_precomputed_mask_fit(data):
    imputer = NoPrecomputedMaskFit(add_indicator=True)
    err_msg = "precomputed is True but the input data is not a mask"
    with pytest.raises(ValueError, match=err_msg):
        imputer.fit(data)
    with pytest.raises(ValueError, match=err_msg):
        imputer.fit_transform(data)


def test_base_no_precomputed_mask_transform(data):
    imputer = NoPrecomputedMaskTransform(add_indicator=True)
    err_msg = "precomputed is True but the input data is not a mask"
    imputer.fit(data)
    with pytest.raises(ValueError, match=err_msg):
        imputer.transform(data)
    with pytest.raises(ValueError, match=err_msg):
        imputer.fit_transform(data)


@pytest.mark.parametrize("X1_type", ["array", "dataframe"])
def test_assign_where(X1_type):
    """Check the behaviour of the private helpers `_assign_where`."""
    rng = np.random.RandomState(0)

    n_samples, n_features = 10, 5
    X1 = _convert_container(rng.randn(n_samples, n_features), constructor_name=X1_type)
    X2 = rng.randn(n_samples, n_features)
    mask = rng.randint(0, 2, size=(n_samples, n_features)).astype(bool)

    _assign_where(X1, X2, mask)

    if X1_type == "dataframe":
        X1 = X1.to_numpy()
    assert_allclose(X1[mask], X2[mask])
