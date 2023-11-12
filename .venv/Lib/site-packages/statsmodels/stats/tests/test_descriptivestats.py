import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
import pandas as pd
import scipy.stats
import pytest

from statsmodels.iolib.table import SimpleTable
from statsmodels.stats.descriptivestats import (
    Description,
    describe,
    sign_test,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore::DeprecationWarning:statsmodels.stats.descriptivestats"
)


@pytest.fixture(scope="function")
def df():
    a = np.random.RandomState(0).standard_normal(100)
    b = pd.Series(np.arange(100) % 10, dtype="category")
    return pd.DataFrame({"a": a, "b": b})


def test_sign_test():
    x = [7.8, 6.6, 6.5, 7.4, 7.3, 7.0, 6.4, 7.1, 6.7, 7.6, 6.8]
    M, p = sign_test(x, mu0=6.5)
    # from R SIGN.test(x, md=6.5)
    # from R
    assert_almost_equal(p, 0.02148, 5)
    # not from R, we use a different convention
    assert_equal(M, 4)


data5 = [
    [25, "Bob", True, 1.2],
    [41, "John", False, 0.5],
    [30, "Alice", True, 0.3],
]

data1 = np.array(
    [(1, 2, "a", "aa"), (2, 3, "b", "bb"), (2, 4, "b", "cc")],
    dtype=[
        ("alpha", float),
        ("beta", int),
        ("gamma", "|S1"),
        ("delta", "|S2"),
    ],
)
data2 = np.array(
    [(1, 2), (2, 3), (2, 4)], dtype=[("alpha", float), ("beta", float)]
)

data3 = np.array([[1, 2, 4, 4], [2, 3, 3, 3], [2, 4, 4, 3]], dtype=float)

data4 = np.array([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1], [9, 9, 9, 9, 9, 9]])


def test_description_exceptions():
    df = pd.DataFrame(
        {"a": np.empty(100), "b": pd.Series(np.arange(100) % 10)},
        dtype="category",
    )
    with pytest.raises(ValueError):
        Description(df, stats=["unknown"])
    with pytest.raises(ValueError):
        Description(df, alpha=-0.3)
    with pytest.raises(ValueError):
        Description(df, percentiles=[0, 100])
    with pytest.raises(ValueError):
        Description(df, percentiles=[10, 20, 30, 10])
    with pytest.raises(ValueError):
        Description(df, ntop=-3)
    with pytest.raises(ValueError):
        Description(df, numeric=False, categorical=False)


def test_description_basic(df):
    res = Description(df)
    assert isinstance(res.frame, pd.DataFrame)
    assert isinstance(res.numeric, pd.DataFrame)
    assert isinstance(res.categorical, pd.DataFrame)
    assert isinstance(res.summary(), SimpleTable)
    assert isinstance(res.summary().as_text(), str)
    assert "Descriptive" in str(res)

    res = Description(df.a)
    assert isinstance(res.frame, pd.DataFrame)
    assert isinstance(res.numeric, pd.DataFrame)
    assert isinstance(res.categorical, pd.DataFrame)
    assert isinstance(res.summary(), SimpleTable)
    assert isinstance(res.summary().as_text(), str)
    assert "Descriptive" in str(res)

    res = Description(df.b)
    assert isinstance(res.frame, pd.DataFrame)
    assert isinstance(res.numeric, pd.DataFrame)
    assert isinstance(res.categorical, pd.DataFrame)
    assert isinstance(res.summary(), SimpleTable)
    assert isinstance(res.summary().as_text(), str)
    assert "Descriptive" in str(res)


def test_odd_percentiles(df):
    percentiles = np.linspace(7.0, 93.0, 13)
    res = Description(df, percentiles=percentiles)
    stats = [
        'nobs', 'missing', 'mean', 'std_err', 'upper_ci', 'lower_ci', 'std',
        'iqr', 'iqr_normal', 'mad', 'mad_normal', 'coef_var', 'range', 'max',
        'min', 'skew', 'kurtosis', 'jarque_bera', 'jarque_bera_pval', 'mode',
        'mode_freq', 'median', 'distinct', 'top_1', 'top_2', 'top_3', 'top_4',
        'top_5', 'freq_1', 'freq_2', 'freq_3', 'freq_4', 'freq_5', '7.0%',
        '14.1%', '21.3%', '28.5%', '35.6%', '42.8%', '50.0%', '57.1%', '64.3%',
        '71.5%', '78.6%', '85.8%', '93.0%']
    assert_equal(res.frame.index.tolist(), stats)


def test_large_ntop(df):
    res = Description(df, ntop=15)
    assert "top_15" in res.frame.index


def test_use_t(df):
    res = Description(df)
    res_t = Description(df, use_t=True)
    assert res_t.frame.a.lower_ci < res.frame.a.lower_ci
    assert res_t.frame.a.upper_ci > res.frame.a.upper_ci


SPECIAL = (
    ("ci", ("lower_ci", "upper_ci")),
    ("jarque_bera", ("jarque_bera", "jarque_bera_pval")),
    ("mode", ("mode", "mode_freq")),
    ("top", tuple([f"top_{i}" for i in range(1, 6)])),
    ("freq", tuple([f"freq_{i}" for i in range(1, 6)])),
)


@pytest.mark.parametrize("stat", SPECIAL, ids=[s[0] for s in SPECIAL])
def test_special_stats(df, stat):
    all_stats = [st for st in Description.default_statistics]
    all_stats.remove(stat[0])
    res = Description(df, stats=all_stats)
    for val in stat[1]:
        assert val not in res.frame.index


def test_empty_columns(df):
    df["c"] = np.nan
    res = Description(df)
    dropped = res.frame.c.dropna()
    assert dropped.shape[0] == 2
    assert "missing" in dropped
    assert "nobs" in dropped

    df["c"] = np.nan
    res = Description(df.c)
    dropped = res.frame.dropna()
    assert dropped.shape[0] == 2


@pytest.mark.skipif(not hasattr(pd, "NA"), reason="Must support NA")
def test_extension_types(df):
    df["c"] = pd.Series(np.arange(100.0))
    df["d"] = pd.Series(np.arange(100), dtype=pd.Int64Dtype())
    df.loc[df.index[::2], "c"] = np.nan
    df.loc[df.index[::2], "d"] = pd.NA
    res = Description(df)
    np.testing.assert_allclose(res.frame.c, res.frame.d)


def test_std_err(df):
    """
    Test the standard error of the mean matches result from scipy.stats.sem
    """
    np.testing.assert_allclose(
        Description(df["a"]).frame.loc["std_err"],
        scipy.stats.sem(df["a"])
    )


def test_describe(df):
    pd.testing.assert_frame_equal(describe(df), Description(df).frame)
