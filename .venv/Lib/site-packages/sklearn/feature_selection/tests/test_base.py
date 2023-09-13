import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy import sparse as sp

from sklearn.base import BaseEstimator
from sklearn.feature_selection._base import SelectorMixin


class StepSelector(SelectorMixin, BaseEstimator):
    """Retain every `step` features (beginning with 0).

    If `step < 1`, then no features are selected.
    """

    def __init__(self, step=2):
        self.step = step

    def fit(self, X, y=None):
        X = self._validate_data(X, accept_sparse="csc")
        return self

    def _get_support_mask(self):
        mask = np.zeros(self.n_features_in_, dtype=bool)
        if self.step >= 1:
            mask[:: self.step] = True
        return mask


support = [True, False] * 5
support_inds = [0, 2, 4, 6, 8]
X = np.arange(20).reshape(2, 10)
Xt = np.arange(0, 20, 2).reshape(2, 5)
Xinv = X.copy()
Xinv[:, 1::2] = 0
y = [0, 1]
feature_names = list("ABCDEFGHIJ")
feature_names_t = feature_names[::2]
feature_names_inv = np.array(feature_names)
feature_names_inv[1::2] = ""


def test_transform_dense():
    sel = StepSelector()
    Xt_actual = sel.fit(X, y).transform(X)
    Xt_actual2 = StepSelector().fit_transform(X, y)
    assert_array_equal(Xt, Xt_actual)
    assert_array_equal(Xt, Xt_actual2)

    # Check dtype matches
    assert np.int32 == sel.transform(X.astype(np.int32)).dtype
    assert np.float32 == sel.transform(X.astype(np.float32)).dtype

    # Check 1d list and other dtype:
    names_t_actual = sel.transform([feature_names])
    assert_array_equal(feature_names_t, names_t_actual.ravel())

    # Check wrong shape raises error
    with pytest.raises(ValueError):
        sel.transform(np.array([[1], [2]]))


def test_transform_sparse():
    sparse = sp.csc_matrix
    sel = StepSelector()
    Xt_actual = sel.fit(sparse(X)).transform(sparse(X))
    Xt_actual2 = sel.fit_transform(sparse(X))
    assert_array_equal(Xt, Xt_actual.toarray())
    assert_array_equal(Xt, Xt_actual2.toarray())

    # Check dtype matches
    assert np.int32 == sel.transform(sparse(X).astype(np.int32)).dtype
    assert np.float32 == sel.transform(sparse(X).astype(np.float32)).dtype

    # Check wrong shape raises error
    with pytest.raises(ValueError):
        sel.transform(np.array([[1], [2]]))


def test_inverse_transform_dense():
    sel = StepSelector()
    Xinv_actual = sel.fit(X, y).inverse_transform(Xt)
    assert_array_equal(Xinv, Xinv_actual)

    # Check dtype matches
    assert np.int32 == sel.inverse_transform(Xt.astype(np.int32)).dtype
    assert np.float32 == sel.inverse_transform(Xt.astype(np.float32)).dtype

    # Check 1d list and other dtype:
    names_inv_actual = sel.inverse_transform([feature_names_t])
    assert_array_equal(feature_names_inv, names_inv_actual.ravel())

    # Check wrong shape raises error
    with pytest.raises(ValueError):
        sel.inverse_transform(np.array([[1], [2]]))


def test_inverse_transform_sparse():
    sparse = sp.csc_matrix
    sel = StepSelector()
    Xinv_actual = sel.fit(sparse(X)).inverse_transform(sparse(Xt))
    assert_array_equal(Xinv, Xinv_actual.toarray())

    # Check dtype matches
    assert np.int32 == sel.inverse_transform(sparse(Xt).astype(np.int32)).dtype
    assert np.float32 == sel.inverse_transform(sparse(Xt).astype(np.float32)).dtype

    # Check wrong shape raises error
    with pytest.raises(ValueError):
        sel.inverse_transform(np.array([[1], [2]]))


def test_get_support():
    sel = StepSelector()
    sel.fit(X, y)
    assert_array_equal(support, sel.get_support())
    assert_array_equal(support_inds, sel.get_support(indices=True))


def test_output_dataframe():
    """Check output dtypes for dataframes is consistent with the input dtypes."""
    pd = pytest.importorskip("pandas")

    X = pd.DataFrame(
        {
            "a": pd.Series([1.0, 2.4, 4.5], dtype=np.float32),
            "b": pd.Series(["a", "b", "a"], dtype="category"),
            "c": pd.Series(["j", "b", "b"], dtype="category"),
            "d": pd.Series([3.0, 2.4, 1.2], dtype=np.float64),
        }
    )

    for step in [2, 3]:
        sel = StepSelector(step=step).set_output(transform="pandas")
        sel.fit(X)

        output = sel.transform(X)
        for name, dtype in output.dtypes.items():
            assert dtype == X.dtypes[name]

    # step=0 will select nothing
    sel0 = StepSelector(step=0).set_output(transform="pandas")
    sel0.fit(X, y)

    msg = "No features were selected"
    with pytest.warns(UserWarning, match=msg):
        output0 = sel0.transform(X)

    assert_array_equal(output0.index, X.index)
    assert output0.shape == (X.shape[0], 0)
