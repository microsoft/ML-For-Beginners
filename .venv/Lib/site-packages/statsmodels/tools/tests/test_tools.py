"""
Test functions for models.tools
"""
from statsmodels.compat.pandas import assert_frame_equal, assert_series_equal
from statsmodels.compat.python import lrange

import string

import numpy as np
from numpy.random import standard_normal
from numpy.testing import (
    assert_almost_equal,
    assert_array_equal,
    assert_equal,
    assert_string_equal,
)
import pandas as pd
import pytest

from statsmodels.datasets import longley
from statsmodels.tools import tools
from statsmodels.tools.tools import pinv_extended


@pytest.fixture(scope="module")
def string_var():
    string_var = [
        string.ascii_lowercase[0:5],
        string.ascii_lowercase[5:10],
        string.ascii_lowercase[10:15],
        string.ascii_lowercase[15:20],
        string.ascii_lowercase[20:25],
    ]
    string_var *= 5
    string_var = np.asarray(sorted(string_var))
    series = pd.Series(string_var, name="string_var")
    return series


class TestTools:
    def test_add_constant_list(self):
        x = lrange(1, 5)
        x = tools.add_constant(x)
        y = np.asarray([[1, 1, 1, 1], [1, 2, 3, 4.0]]).T
        assert_equal(x, y)

    def test_add_constant_1d(self):
        x = np.arange(1, 5)
        x = tools.add_constant(x)
        y = np.asarray([[1, 1, 1, 1], [1, 2, 3, 4.0]]).T
        assert_equal(x, y)

    def test_add_constant_has_constant1d(self):
        x = np.ones(5)
        x = tools.add_constant(x, has_constant="skip")
        assert_equal(x, np.ones((5, 1)))

        with pytest.raises(ValueError):
            tools.add_constant(x, has_constant="raise")

        assert_equal(
            tools.add_constant(x, has_constant="add"), np.ones((5, 2))
        )

    def test_add_constant_has_constant2d(self):
        x = np.asarray([[1, 1, 1, 1], [1, 2, 3, 4.0]]).T
        y = tools.add_constant(x, has_constant="skip")
        assert_equal(x, y)

        with pytest.raises(ValueError):
            tools.add_constant(x, has_constant="raise")

        assert_equal(
            tools.add_constant(x, has_constant="add"),
            np.column_stack((np.ones(4), x)),
        )

    def test_add_constant_series(self):
        s = pd.Series([1.0, 2.0, 3.0])
        output = tools.add_constant(s)
        expected = pd.Series([1.0, 1.0, 1.0], name="const")
        assert_series_equal(expected, output["const"])

    def test_add_constant_dataframe(self):
        df = pd.DataFrame([[1.0, "a", 4], [2.0, "bc", 9], [3.0, "def", 16]])
        output = tools.add_constant(df)
        expected = pd.Series([1.0, 1.0, 1.0], name="const")
        assert_series_equal(expected, output["const"])
        dfc = df.copy()
        dfc.insert(0, "const", np.ones(3))
        assert_frame_equal(dfc, output)

    def test_add_constant_zeros(self):
        a = np.zeros(100)
        output = tools.add_constant(a)
        assert_equal(output[:, 0], np.ones(100))

        s = pd.Series([0.0, 0.0, 0.0])
        output = tools.add_constant(s)
        expected = pd.Series([1.0, 1.0, 1.0], name="const")
        assert_series_equal(expected, output["const"])

        df = pd.DataFrame([[0.0, "a", 4], [0.0, "bc", 9], [0.0, "def", 16]])
        output = tools.add_constant(df)
        dfc = df.copy()
        dfc.insert(0, "const", np.ones(3))
        assert_frame_equal(dfc, output)

        df = pd.DataFrame([[1.0, "a", 0], [0.0, "bc", 0], [0.0, "def", 0]])
        output = tools.add_constant(df)
        dfc = df.copy()
        dfc.insert(0, "const", np.ones(3))
        assert_frame_equal(dfc, output)

    def test_recipr(self):
        X = np.array([[2, 1], [-1, 0]])
        Y = tools.recipr(X)
        assert_almost_equal(Y, np.array([[0.5, 1], [0, 0]]))

    def test_recipr0(self):
        X = np.array([[2, 1], [-4, 0]])
        Y = tools.recipr0(X)
        assert_almost_equal(Y, np.array([[0.5, 1], [-0.25, 0]]))

    def test_extendedpinv(self):
        X = standard_normal((40, 10))
        np_inv = np.linalg.pinv(X)
        np_sing_vals = np.linalg.svd(X, 0, 0)
        sm_inv, sing_vals = pinv_extended(X)
        assert_almost_equal(np_inv, sm_inv)
        assert_almost_equal(np_sing_vals, sing_vals)

    def test_extendedpinv_singular(self):
        X = standard_normal((40, 10))
        X[:, 5] = X[:, 1] + X[:, 3]
        np_inv = np.linalg.pinv(X)
        np_sing_vals = np.linalg.svd(X, 0, 0)
        sm_inv, sing_vals = pinv_extended(X)
        assert_almost_equal(np_inv, sm_inv)
        assert_almost_equal(np_sing_vals, sing_vals)

    def test_fullrank(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X = standard_normal((40, 10))
            X[:, 0] = X[:, 1] + X[:, 2]

            Y = tools.fullrank(X)
            assert_equal(Y.shape, (40, 9))

            X[:, 5] = X[:, 3] + X[:, 4]
            Y = tools.fullrank(X)
            assert_equal(Y.shape, (40, 8))
            warnings.simplefilter("ignore")


def test_estimable():
    rng = np.random.RandomState(20120713)
    N, P = (40, 10)
    X = rng.normal(size=(N, P))
    C = rng.normal(size=(1, P))
    isestimable = tools.isestimable
    assert isestimable(C, X)
    assert isestimable(np.eye(P), X)
    for row in np.eye(P):
        assert isestimable(row, X)
    X = np.ones((40, 2))
    assert isestimable([1, 1], X)
    assert not isestimable([1, 0], X)
    assert not isestimable([0, 1], X)
    assert not isestimable(np.eye(2), X)
    halfX = rng.normal(size=(N, 5))
    X = np.hstack([halfX, halfX])
    assert not isestimable(np.hstack([np.eye(5), np.zeros((5, 5))]), X)
    assert not isestimable(np.hstack([np.zeros((5, 5)), np.eye(5)]), X)
    assert isestimable(np.hstack([np.eye(5), np.eye(5)]), X)
    # Test array_like for design
    XL = X.tolist()
    assert isestimable(np.hstack([np.eye(5), np.eye(5)]), XL)
    # Test ValueError for incorrect number of columns
    X = rng.normal(size=(N, 5))
    for n in range(1, 4):
        with pytest.raises(ValueError):
            isestimable(np.ones((n,)), X)
    with pytest.raises(ValueError):
        isestimable(np.eye(4), X)


def test_pandas_const_series():
    dta = longley.load_pandas()
    series = dta.exog["GNP"]
    series = tools.add_constant(series, prepend=False)
    assert_string_equal("const", series.columns[1])
    assert_equal(series.var(0)[1], 0)


def test_pandas_const_series_prepend():
    dta = longley.load_pandas()
    series = dta.exog["GNP"]
    series = tools.add_constant(series, prepend=True)
    assert_string_equal("const", series.columns[0])
    assert_equal(series.var(0)[0], 0)


def test_pandas_const_df():
    dta = longley.load_pandas().exog
    dta = tools.add_constant(dta, prepend=False)
    assert_string_equal("const", dta.columns[-1])
    assert_equal(dta.var(0)[-1], 0)


def test_pandas_const_df_prepend():
    dta = longley.load_pandas().exog
    # regression test for #1025
    dta["UNEMP"] /= dta["UNEMP"].std()
    dta = tools.add_constant(dta, prepend=True)
    assert_string_equal("const", dta.columns[0])
    assert_equal(dta.var(0)[0], 0)


class TestNanDot:
    @classmethod
    def setup_class(cls):
        nan = np.nan
        cls.mx_1 = np.array([[nan, 1.0], [2.0, 3.0]])
        cls.mx_2 = np.array([[nan, nan], [2.0, 3.0]])
        cls.mx_3 = np.array([[0.0, 0.0], [0.0, 0.0]])
        cls.mx_4 = np.array([[1.0, 0.0], [1.0, 0.0]])
        cls.mx_5 = np.array([[0.0, 1.0], [0.0, 1.0]])
        cls.mx_6 = np.array([[1.0, 2.0], [3.0, 4.0]])

    def test_11(self):
        test_res = tools.nan_dot(self.mx_1, self.mx_1)
        expected_res = np.array([[np.nan, np.nan], [np.nan, 11.0]])
        assert_array_equal(test_res, expected_res)

    def test_12(self):
        nan = np.nan
        test_res = tools.nan_dot(self.mx_1, self.mx_2)
        expected_res = np.array([[nan, nan], [nan, nan]])
        assert_array_equal(test_res, expected_res)

    def test_13(self):
        nan = np.nan
        test_res = tools.nan_dot(self.mx_1, self.mx_3)
        expected_res = np.array([[0.0, 0.0], [0.0, 0.0]])
        assert_array_equal(test_res, expected_res)

    def test_14(self):
        nan = np.nan
        test_res = tools.nan_dot(self.mx_1, self.mx_4)
        expected_res = np.array([[nan, 0.0], [5.0, 0.0]])
        assert_array_equal(test_res, expected_res)

    def test_41(self):
        nan = np.nan
        test_res = tools.nan_dot(self.mx_4, self.mx_1)
        expected_res = np.array([[nan, 1.0], [nan, 1.0]])
        assert_array_equal(test_res, expected_res)

    def test_23(self):
        nan = np.nan
        test_res = tools.nan_dot(self.mx_2, self.mx_3)
        expected_res = np.array([[0.0, 0.0], [0.0, 0.0]])
        assert_array_equal(test_res, expected_res)

    def test_32(self):
        nan = np.nan
        test_res = tools.nan_dot(self.mx_3, self.mx_2)
        expected_res = np.array([[0.0, 0.0], [0.0, 0.0]])
        assert_array_equal(test_res, expected_res)

    def test_24(self):
        nan = np.nan
        test_res = tools.nan_dot(self.mx_2, self.mx_4)
        expected_res = np.array([[nan, 0.0], [5.0, 0.0]])
        assert_array_equal(test_res, expected_res)

    def test_25(self):
        nan = np.nan
        test_res = tools.nan_dot(self.mx_2, self.mx_5)
        expected_res = np.array([[0.0, nan], [0.0, 5.0]])
        assert_array_equal(test_res, expected_res)

    def test_66(self):
        nan = np.nan
        test_res = tools.nan_dot(self.mx_6, self.mx_6)
        expected_res = np.array([[7.0, 10.0], [15.0, 22.0]])
        assert_array_equal(test_res, expected_res)


class TestEnsure2d:
    @classmethod
    def setup_class(cls):
        x = np.arange(400.0).reshape((100, 4))
        cls.df = pd.DataFrame(x, columns=["a", "b", "c", "d"])
        cls.series = cls.df.iloc[:, 0]
        cls.ndarray = x

    def test_enfore_numpy(self):
        results = tools._ensure_2d(self.df, True)
        assert_array_equal(results[0], self.ndarray)
        assert_array_equal(results[1], self.df.columns)
        results = tools._ensure_2d(self.series, True)
        assert_array_equal(results[0], self.ndarray[:, [0]])
        assert_array_equal(results[1], self.df.columns[0])

    def test_pandas(self):
        results = tools._ensure_2d(self.df, False)
        assert_frame_equal(results[0], self.df)
        assert_array_equal(results[1], self.df.columns)

        results = tools._ensure_2d(self.series, False)
        assert_frame_equal(results[0], self.df.iloc[:, [0]])
        assert_equal(results[1], self.df.columns[0])

    def test_numpy(self):
        results = tools._ensure_2d(self.ndarray)
        assert_array_equal(results[0], self.ndarray)
        assert_equal(results[1], None)

        results = tools._ensure_2d(self.ndarray[:, 0])
        assert_array_equal(results[0], self.ndarray[:, [0]])
        assert_equal(results[1], None)
