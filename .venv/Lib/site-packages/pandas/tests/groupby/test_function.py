import builtins
from io import StringIO
import re

import numpy as np
import pytest

from pandas._libs import lib
from pandas.errors import UnsupportedFunctionCall

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    Timestamp,
    date_range,
)
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
from pandas.util import _test_decorators as td


@pytest.fixture(
    params=[np.int32, np.int64, np.float32, np.float64, "Int64", "Float64"],
    ids=["np.int32", "np.int64", "np.float32", "np.float64", "Int64", "Float64"],
)
def dtypes_for_minmax(request):
    """
    Fixture of dtypes with min and max values used for testing
    cummin and cummax
    """
    dtype = request.param

    np_type = dtype
    if dtype == "Int64":
        np_type = np.int64
    elif dtype == "Float64":
        np_type = np.float64

    min_val = (
        np.iinfo(np_type).min
        if np.dtype(np_type).kind == "i"
        else np.finfo(np_type).min
    )
    max_val = (
        np.iinfo(np_type).max
        if np.dtype(np_type).kind == "i"
        else np.finfo(np_type).max
    )

    return (dtype, min_val, max_val)


def test_intercept_builtin_sum():
    s = Series([1.0, 2.0, np.nan, 3.0])
    grouped = s.groupby([0, 1, 2, 2])

    msg = "using SeriesGroupBy.sum"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        # GH#53425
        result = grouped.agg(builtins.sum)
    msg = "using np.sum"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        # GH#53425
        result2 = grouped.apply(builtins.sum)
    expected = grouped.sum()
    tm.assert_series_equal(result, expected)
    tm.assert_series_equal(result2, expected)


@pytest.mark.parametrize("f", [max, min, sum])
@pytest.mark.parametrize("keys", ["jim", ["jim", "joe"]])  # Single key  # Multi-key
def test_builtins_apply(keys, f):
    # see gh-8155
    rs = np.random.default_rng(2)
    df = DataFrame(rs.integers(1, 7, (10, 2)), columns=["jim", "joe"])
    df["jolie"] = rs.standard_normal(10)

    gb = df.groupby(keys)

    fname = f.__name__

    warn = None if f is not sum else FutureWarning
    msg = "The behavior of DataFrame.sum with axis=None is deprecated"
    with tm.assert_produces_warning(
        warn, match=msg, check_stacklevel=False, raise_on_extra_warnings=False
    ):
        # Also warns on deprecation GH#53425
        result = gb.apply(f)
    ngroups = len(df.drop_duplicates(subset=keys))

    assert_msg = f"invalid frame shape: {result.shape} (expected ({ngroups}, 3))"
    assert result.shape == (ngroups, 3), assert_msg

    npfunc = lambda x: getattr(np, fname)(x, axis=0)  # numpy's equivalent function
    expected = gb.apply(npfunc)
    tm.assert_frame_equal(result, expected)

    with tm.assert_produces_warning(None):
        expected2 = gb.apply(lambda x: npfunc(x))
    tm.assert_frame_equal(result, expected2)

    if f != sum:
        expected = gb.agg(fname).reset_index()
        expected.set_index(keys, inplace=True, drop=False)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    tm.assert_series_equal(getattr(result, fname)(axis=0), getattr(df, fname)(axis=0))


class TestNumericOnly:
    # make sure that we are passing thru kwargs to our agg functions

    @pytest.fixture
    def df(self):
        # GH3668
        # GH5724
        df = DataFrame(
            {
                "group": [1, 1, 2],
                "int": [1, 2, 3],
                "float": [4.0, 5.0, 6.0],
                "string": list("abc"),
                "category_string": Series(list("abc")).astype("category"),
                "category_int": [7, 8, 9],
                "datetime": date_range("20130101", periods=3),
                "datetimetz": date_range("20130101", periods=3, tz="US/Eastern"),
                "timedelta": pd.timedelta_range("1 s", periods=3, freq="s"),
            },
            columns=[
                "group",
                "int",
                "float",
                "string",
                "category_string",
                "category_int",
                "datetime",
                "datetimetz",
                "timedelta",
            ],
        )
        return df

    @pytest.mark.parametrize("method", ["mean", "median"])
    def test_averages(self, df, method):
        # mean / median
        expected_columns_numeric = Index(["int", "float", "category_int"])

        gb = df.groupby("group")
        expected = DataFrame(
            {
                "category_int": [7.5, 9],
                "float": [4.5, 6.0],
                "timedelta": [pd.Timedelta("1.5s"), pd.Timedelta("3s")],
                "int": [1.5, 3],
                "datetime": [
                    Timestamp("2013-01-01 12:00:00"),
                    Timestamp("2013-01-03 00:00:00"),
                ],
                "datetimetz": [
                    Timestamp("2013-01-01 12:00:00", tz="US/Eastern"),
                    Timestamp("2013-01-03 00:00:00", tz="US/Eastern"),
                ],
            },
            index=Index([1, 2], name="group"),
            columns=[
                "int",
                "float",
                "category_int",
            ],
        )

        result = getattr(gb, method)(numeric_only=True)
        tm.assert_frame_equal(result.reindex_like(expected), expected)

        expected_columns = expected.columns

        self._check(df, method, expected_columns, expected_columns_numeric)

    @pytest.mark.parametrize("method", ["min", "max"])
    def test_extrema(self, df, method):
        # TODO: min, max *should* handle
        # categorical (ordered) dtype

        expected_columns = Index(
            [
                "int",
                "float",
                "string",
                "category_int",
                "datetime",
                "datetimetz",
                "timedelta",
            ]
        )
        expected_columns_numeric = expected_columns

        self._check(df, method, expected_columns, expected_columns_numeric)

    @pytest.mark.parametrize("method", ["first", "last"])
    def test_first_last(self, df, method):
        expected_columns = Index(
            [
                "int",
                "float",
                "string",
                "category_string",
                "category_int",
                "datetime",
                "datetimetz",
                "timedelta",
            ]
        )
        expected_columns_numeric = expected_columns

        self._check(df, method, expected_columns, expected_columns_numeric)

    @pytest.mark.parametrize("method", ["sum", "cumsum"])
    def test_sum_cumsum(self, df, method):
        expected_columns_numeric = Index(["int", "float", "category_int"])
        expected_columns = Index(
            ["int", "float", "string", "category_int", "timedelta"]
        )
        if method == "cumsum":
            # cumsum loses string
            expected_columns = Index(["int", "float", "category_int", "timedelta"])

        self._check(df, method, expected_columns, expected_columns_numeric)

    @pytest.mark.parametrize("method", ["prod", "cumprod"])
    def test_prod_cumprod(self, df, method):
        expected_columns = Index(["int", "float", "category_int"])
        expected_columns_numeric = expected_columns

        self._check(df, method, expected_columns, expected_columns_numeric)

    @pytest.mark.parametrize("method", ["cummin", "cummax"])
    def test_cummin_cummax(self, df, method):
        # like min, max, but don't include strings
        expected_columns = Index(
            ["int", "float", "category_int", "datetime", "datetimetz", "timedelta"]
        )

        # GH#15561: numeric_only=False set by default like min/max
        expected_columns_numeric = expected_columns

        self._check(df, method, expected_columns, expected_columns_numeric)

    def _check(self, df, method, expected_columns, expected_columns_numeric):
        gb = df.groupby("group")

        # object dtypes for transformations are not implemented in Cython and
        # have no Python fallback
        exception = NotImplementedError if method.startswith("cum") else TypeError

        if method in ("min", "max", "cummin", "cummax", "cumsum", "cumprod"):
            # The methods default to numeric_only=False and raise TypeError
            msg = "|".join(
                [
                    "Categorical is not ordered",
                    f"Cannot perform {method} with non-ordered Categorical",
                    re.escape(f"agg function failed [how->{method},dtype->object]"),
                    # cumsum/cummin/cummax/cumprod
                    "function is not implemented for this dtype",
                ]
            )
            with pytest.raises(exception, match=msg):
                getattr(gb, method)()
        elif method in ("sum", "mean", "median", "prod"):
            msg = "|".join(
                [
                    "category type does not support sum operations",
                    re.escape(f"agg function failed [how->{method},dtype->object]"),
                ]
            )
            with pytest.raises(exception, match=msg):
                getattr(gb, method)()
        else:
            result = getattr(gb, method)()
            tm.assert_index_equal(result.columns, expected_columns_numeric)

        if method not in ("first", "last"):
            msg = "|".join(
                [
                    "Categorical is not ordered",
                    "category type does not support",
                    "function is not implemented for this dtype",
                    f"Cannot perform {method} with non-ordered Categorical",
                    re.escape(f"agg function failed [how->{method},dtype->object]"),
                ]
            )
            with pytest.raises(exception, match=msg):
                getattr(gb, method)(numeric_only=False)
        else:
            result = getattr(gb, method)(numeric_only=False)
            tm.assert_index_equal(result.columns, expected_columns)


class TestGroupByNonCythonPaths:
    # GH#5610 non-cython calls should not include the grouper
    # Tests for code not expected to go through cython paths.

    @pytest.fixture
    def df(self):
        df = DataFrame(
            [[1, 2, "foo"], [1, np.nan, "bar"], [3, np.nan, "baz"]],
            columns=["A", "B", "C"],
        )
        return df

    @pytest.fixture
    def gb(self, df):
        gb = df.groupby("A")
        return gb

    @pytest.fixture
    def gni(self, df):
        gni = df.groupby("A", as_index=False)
        return gni

    def test_describe(self, df, gb, gni):
        # describe
        expected_index = Index([1, 3], name="A")
        expected_col = MultiIndex(
            levels=[["B"], ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]],
            codes=[[0] * 8, list(range(8))],
        )
        expected = DataFrame(
            [
                [1.0, 2.0, np.nan, 2.0, 2.0, 2.0, 2.0, 2.0],
                [0.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            ],
            index=expected_index,
            columns=expected_col,
        )
        result = gb.describe()
        tm.assert_frame_equal(result, expected)

        expected = expected.reset_index()
        result = gni.describe()
        tm.assert_frame_equal(result, expected)


def test_cython_api2():
    # this takes the fast apply path

    # cumsum (GH5614)
    df = DataFrame([[1, 2, np.nan], [1, np.nan, 9], [3, 4, 9]], columns=["A", "B", "C"])
    expected = DataFrame([[2, np.nan], [np.nan, 9], [4, 9]], columns=["B", "C"])
    result = df.groupby("A").cumsum()
    tm.assert_frame_equal(result, expected)

    # GH 5755 - cumsum is a transformer and should ignore as_index
    result = df.groupby("A", as_index=False).cumsum()
    tm.assert_frame_equal(result, expected)

    # GH 13994
    msg = "DataFrameGroupBy.cumsum with axis=1 is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby("A").cumsum(axis=1)
    expected = df.cumsum(axis=1)
    tm.assert_frame_equal(result, expected)

    msg = "DataFrameGroupBy.cumprod with axis=1 is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby("A").cumprod(axis=1)
    expected = df.cumprod(axis=1)
    tm.assert_frame_equal(result, expected)


def test_cython_median():
    arr = np.random.default_rng(2).standard_normal(1000)
    arr[::2] = np.nan
    df = DataFrame(arr)

    labels = np.random.default_rng(2).integers(0, 50, size=1000).astype(float)
    labels[::17] = np.nan

    result = df.groupby(labels).median()
    msg = "using DataFrameGroupBy.median"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        exp = df.groupby(labels).agg(np.nanmedian)
    tm.assert_frame_equal(result, exp)

    df = DataFrame(np.random.default_rng(2).standard_normal((1000, 5)))
    msg = "using DataFrameGroupBy.median"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        rs = df.groupby(labels).agg(np.median)
    xp = df.groupby(labels).median()
    tm.assert_frame_equal(rs, xp)


def test_median_empty_bins(observed):
    df = DataFrame(np.random.default_rng(2).integers(0, 44, 500))

    grps = range(0, 55, 5)
    bins = pd.cut(df[0], grps)

    result = df.groupby(bins, observed=observed).median()
    expected = df.groupby(bins, observed=observed).agg(lambda x: x.median())
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "dtype", ["int8", "int16", "int32", "int64", "float32", "float64", "uint64"]
)
@pytest.mark.parametrize(
    "method,data",
    [
        ("first", {"df": [{"a": 1, "b": 1}, {"a": 2, "b": 3}]}),
        ("last", {"df": [{"a": 1, "b": 2}, {"a": 2, "b": 4}]}),
        ("min", {"df": [{"a": 1, "b": 1}, {"a": 2, "b": 3}]}),
        ("max", {"df": [{"a": 1, "b": 2}, {"a": 2, "b": 4}]}),
        ("count", {"df": [{"a": 1, "b": 2}, {"a": 2, "b": 2}], "out_type": "int64"}),
    ],
)
def test_groupby_non_arithmetic_agg_types(dtype, method, data):
    # GH9311, GH6620
    df = DataFrame(
        [{"a": 1, "b": 1}, {"a": 1, "b": 2}, {"a": 2, "b": 3}, {"a": 2, "b": 4}]
    )

    df["b"] = df.b.astype(dtype)

    if "args" not in data:
        data["args"] = []

    if "out_type" in data:
        out_type = data["out_type"]
    else:
        out_type = dtype

    exp = data["df"]
    df_out = DataFrame(exp)

    df_out["b"] = df_out.b.astype(out_type)
    df_out.set_index("a", inplace=True)

    grpd = df.groupby("a")
    t = getattr(grpd, method)(*data["args"])
    tm.assert_frame_equal(t, df_out)


@pytest.mark.parametrize(
    "i",
    [
        (
            Timestamp("2011-01-15 12:50:28.502376"),
            Timestamp("2011-01-20 12:50:28.593448"),
        ),
        (24650000000000001, 24650000000000002),
    ],
)
def test_groupby_non_arithmetic_agg_int_like_precision(i):
    # see gh-6620, gh-9311
    df = DataFrame([{"a": 1, "b": i[0]}, {"a": 1, "b": i[1]}])

    grp_exp = {
        "first": {"expected": i[0]},
        "last": {"expected": i[1]},
        "min": {"expected": i[0]},
        "max": {"expected": i[1]},
        "nth": {"expected": i[1], "args": [1]},
        "count": {"expected": 2},
    }

    for method, data in grp_exp.items():
        if "args" not in data:
            data["args"] = []

        grouped = df.groupby("a")
        res = getattr(grouped, method)(*data["args"])

        assert res.iloc[0].b == data["expected"]


@pytest.mark.parametrize(
    "func, values",
    [
        ("idxmin", {"c_int": [0, 2], "c_float": [1, 3], "c_date": [1, 2]}),
        ("idxmax", {"c_int": [1, 3], "c_float": [0, 2], "c_date": [0, 3]}),
    ],
)
@pytest.mark.parametrize("numeric_only", [True, False])
def test_idxmin_idxmax_returns_int_types(func, values, numeric_only):
    # GH 25444
    df = DataFrame(
        {
            "name": ["A", "A", "B", "B"],
            "c_int": [1, 2, 3, 4],
            "c_float": [4.02, 3.03, 2.04, 1.05],
            "c_date": ["2019", "2018", "2016", "2017"],
        }
    )
    df["c_date"] = pd.to_datetime(df["c_date"])
    df["c_date_tz"] = df["c_date"].dt.tz_localize("US/Pacific")
    df["c_timedelta"] = df["c_date"] - df["c_date"].iloc[0]
    df["c_period"] = df["c_date"].dt.to_period("W")
    df["c_Integer"] = df["c_int"].astype("Int64")
    df["c_Floating"] = df["c_float"].astype("Float64")

    result = getattr(df.groupby("name"), func)(numeric_only=numeric_only)

    expected = DataFrame(values, index=Index(["A", "B"], name="name"))
    if numeric_only:
        expected = expected.drop(columns=["c_date"])
    else:
        expected["c_date_tz"] = expected["c_date"]
        expected["c_timedelta"] = expected["c_date"]
        expected["c_period"] = expected["c_date"]
    expected["c_Integer"] = expected["c_int"]
    expected["c_Floating"] = expected["c_float"]

    tm.assert_frame_equal(result, expected)


def test_idxmin_idxmax_axis1():
    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)), columns=["A", "B", "C", "D"]
    )
    df["A"] = [1, 2, 3, 1, 2, 3, 1, 2, 3, 4]

    gb = df.groupby("A")

    warn_msg = "DataFrameGroupBy.idxmax with axis=1 is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=warn_msg):
        res = gb.idxmax(axis=1)

    alt = df.iloc[:, 1:].idxmax(axis=1)
    indexer = res.index.get_level_values(1)

    tm.assert_series_equal(alt[indexer], res.droplevel("A"))

    df["E"] = date_range("2016-01-01", periods=10)
    gb2 = df.groupby("A")

    msg = "'>' not supported between instances of 'Timestamp' and 'float'"
    with pytest.raises(TypeError, match=msg):
        with tm.assert_produces_warning(FutureWarning, match=warn_msg):
            gb2.idxmax(axis=1)


@pytest.mark.parametrize("numeric_only", [True, False, None])
def test_axis1_numeric_only(request, groupby_func, numeric_only):
    if groupby_func in ("idxmax", "idxmin"):
        pytest.skip("idxmax and idx_min tested in test_idxmin_idxmax_axis1")
    if groupby_func in ("corrwith", "skew"):
        msg = "GH#47723 groupby.corrwith and skew do not correctly implement axis=1"
        request.node.add_marker(pytest.mark.xfail(reason=msg))

    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)), columns=["A", "B", "C", "D"]
    )
    df["E"] = "x"
    groups = [1, 2, 3, 1, 2, 3, 1, 2, 3, 4]
    gb = df.groupby(groups)
    method = getattr(gb, groupby_func)
    args = get_groupby_method_args(groupby_func, df)
    kwargs = {"axis": 1}
    if numeric_only is not None:
        # when numeric_only is None we don't pass any argument
        kwargs["numeric_only"] = numeric_only

    # Functions without numeric_only and axis args
    no_args = ("cumprod", "cumsum", "diff", "fillna", "pct_change", "rank", "shift")
    # Functions with axis args
    has_axis = (
        "cumprod",
        "cumsum",
        "diff",
        "pct_change",
        "rank",
        "shift",
        "cummax",
        "cummin",
        "idxmin",
        "idxmax",
        "fillna",
    )
    warn_msg = f"DataFrameGroupBy.{groupby_func} with axis=1 is deprecated"
    if numeric_only is not None and groupby_func in no_args:
        msg = "got an unexpected keyword argument 'numeric_only'"
        if groupby_func in ["cumprod", "cumsum"]:
            with pytest.raises(TypeError, match=msg):
                with tm.assert_produces_warning(FutureWarning, match=warn_msg):
                    method(*args, **kwargs)
        else:
            with pytest.raises(TypeError, match=msg):
                method(*args, **kwargs)
    elif groupby_func not in has_axis:
        msg = "got an unexpected keyword argument 'axis'"
        with pytest.raises(TypeError, match=msg):
            method(*args, **kwargs)
    # fillna and shift are successful even on object dtypes
    elif (numeric_only is None or not numeric_only) and groupby_func not in (
        "fillna",
        "shift",
    ):
        msgs = (
            # cummax, cummin, rank
            "not supported between instances of",
            # cumprod
            "can't multiply sequence by non-int of type 'float'",
            # cumsum, diff, pct_change
            "unsupported operand type",
        )
        with pytest.raises(TypeError, match=f"({'|'.join(msgs)})"):
            with tm.assert_produces_warning(FutureWarning, match=warn_msg):
                method(*args, **kwargs)
    else:
        with tm.assert_produces_warning(FutureWarning, match=warn_msg):
            result = method(*args, **kwargs)

        df_expected = df.drop(columns="E").T if numeric_only else df.T
        expected = getattr(df_expected, groupby_func)(*args).T
        if groupby_func == "shift" and not numeric_only:
            # shift with axis=1 leaves the leftmost column as numeric
            # but transposing for expected gives us object dtype
            expected = expected.astype(float)

        tm.assert_equal(result, expected)


def test_groupby_cumprod():
    # GH 4095
    df = DataFrame({"key": ["b"] * 10, "value": 2})

    actual = df.groupby("key")["value"].cumprod()
    expected = df.groupby("key", group_keys=False)["value"].apply(lambda x: x.cumprod())
    expected.name = "value"
    tm.assert_series_equal(actual, expected)

    df = DataFrame({"key": ["b"] * 100, "value": 2})
    df["value"] = df["value"].astype(float)
    actual = df.groupby("key")["value"].cumprod()
    expected = df.groupby("key", group_keys=False)["value"].apply(lambda x: x.cumprod())
    expected.name = "value"
    tm.assert_series_equal(actual, expected)


def test_groupby_cumprod_overflow():
    # GH#37493 if we overflow we return garbage consistent with numpy
    df = DataFrame({"key": ["b"] * 4, "value": 100_000})
    actual = df.groupby("key")["value"].cumprod()
    expected = Series(
        [100_000, 10_000_000_000, 1_000_000_000_000_000, 7766279631452241920],
        name="value",
    )
    tm.assert_series_equal(actual, expected)

    numpy_result = df.groupby("key", group_keys=False)["value"].apply(
        lambda x: x.cumprod()
    )
    numpy_result.name = "value"
    tm.assert_series_equal(actual, numpy_result)


def test_groupby_cumprod_nan_influences_other_columns():
    # GH#48064
    df = DataFrame(
        {
            "a": 1,
            "b": [1, np.nan, 2],
            "c": [1, 2, 3.0],
        }
    )
    result = df.groupby("a").cumprod(numeric_only=True, skipna=False)
    expected = DataFrame({"b": [1, np.nan, np.nan], "c": [1, 2, 6.0]})
    tm.assert_frame_equal(result, expected)


def scipy_sem(*args, **kwargs):
    from scipy.stats import sem

    return sem(*args, ddof=1, **kwargs)


@pytest.mark.parametrize(
    "op,targop",
    [
        ("mean", np.mean),
        ("median", np.median),
        ("std", np.std),
        ("var", np.var),
        ("sum", np.sum),
        ("prod", np.prod),
        ("min", np.min),
        ("max", np.max),
        ("first", lambda x: x.iloc[0]),
        ("last", lambda x: x.iloc[-1]),
        ("count", np.size),
        pytest.param("sem", scipy_sem, marks=td.skip_if_no_scipy),
    ],
)
def test_ops_general(op, targop):
    df = DataFrame(np.random.default_rng(2).standard_normal(1000))
    labels = np.random.default_rng(2).integers(0, 50, size=1000).astype(float)

    result = getattr(df.groupby(labels), op)()
    warn = None if op in ("first", "last", "count", "sem") else FutureWarning
    msg = f"using DataFrameGroupBy.{op}"
    with tm.assert_produces_warning(warn, match=msg):
        expected = df.groupby(labels).agg(targop)
    tm.assert_frame_equal(result, expected)


def test_max_nan_bug():
    raw = """,Date,app,File
-04-23,2013-04-23 00:00:00,,log080001.log
-05-06,2013-05-06 00:00:00,,log.log
-05-07,2013-05-07 00:00:00,OE,xlsx"""

    with tm.assert_produces_warning(UserWarning, match="Could not infer format"):
        df = pd.read_csv(StringIO(raw), parse_dates=[0])
    gb = df.groupby("Date")
    r = gb[["File"]].max()
    e = gb["File"].max().to_frame()
    tm.assert_frame_equal(r, e)
    assert not r["File"].isna().any()


def test_nlargest():
    a = Series([1, 3, 5, 7, 2, 9, 0, 4, 6, 10])
    b = Series(list("a" * 5 + "b" * 5))
    gb = a.groupby(b)
    r = gb.nlargest(3)
    e = Series(
        [7, 5, 3, 10, 9, 6],
        index=MultiIndex.from_arrays([list("aaabbb"), [3, 2, 1, 9, 5, 8]]),
    )
    tm.assert_series_equal(r, e)

    a = Series([1, 1, 3, 2, 0, 3, 3, 2, 1, 0])
    gb = a.groupby(b)
    e = Series(
        [3, 2, 1, 3, 3, 2],
        index=MultiIndex.from_arrays([list("aaabbb"), [2, 3, 1, 6, 5, 7]]),
    )
    tm.assert_series_equal(gb.nlargest(3, keep="last"), e)


def test_nlargest_mi_grouper():
    # see gh-21411
    npr = np.random.default_rng(2)

    dts = date_range("20180101", periods=10)
    iterables = [dts, ["one", "two"]]

    idx = MultiIndex.from_product(iterables, names=["first", "second"])
    s = Series(npr.standard_normal(20), index=idx)

    result = s.groupby("first").nlargest(1)

    exp_idx = MultiIndex.from_tuples(
        [
            (dts[0], dts[0], "one"),
            (dts[1], dts[1], "one"),
            (dts[2], dts[2], "one"),
            (dts[3], dts[3], "two"),
            (dts[4], dts[4], "one"),
            (dts[5], dts[5], "one"),
            (dts[6], dts[6], "one"),
            (dts[7], dts[7], "one"),
            (dts[8], dts[8], "one"),
            (dts[9], dts[9], "one"),
        ],
        names=["first", "first", "second"],
    )

    exp_values = [
        0.18905338179353307,
        -0.41306354339189344,
        1.799707382720902,
        0.7738065867276614,
        0.28121066979764925,
        0.9775674511260357,
        -0.3288239040579627,
        0.45495807124085547,
        0.5452887139646817,
        0.12682784711186987,
    ]

    expected = Series(exp_values, index=exp_idx)
    tm.assert_series_equal(result, expected, check_exact=False, rtol=1e-3)


def test_nsmallest():
    a = Series([1, 3, 5, 7, 2, 9, 0, 4, 6, 10])
    b = Series(list("a" * 5 + "b" * 5))
    gb = a.groupby(b)
    r = gb.nsmallest(3)
    e = Series(
        [1, 2, 3, 0, 4, 6],
        index=MultiIndex.from_arrays([list("aaabbb"), [0, 4, 1, 6, 7, 8]]),
    )
    tm.assert_series_equal(r, e)

    a = Series([1, 1, 3, 2, 0, 3, 3, 2, 1, 0])
    gb = a.groupby(b)
    e = Series(
        [0, 1, 1, 0, 1, 2],
        index=MultiIndex.from_arrays([list("aaabbb"), [4, 1, 0, 9, 8, 7]]),
    )
    tm.assert_series_equal(gb.nsmallest(3, keep="last"), e)


@pytest.mark.parametrize(
    "data, groups",
    [([0, 1, 2, 3], [0, 0, 1, 1]), ([0], [0])],
)
@pytest.mark.parametrize("dtype", [None, *tm.ALL_INT_NUMPY_DTYPES])
@pytest.mark.parametrize("method", ["nlargest", "nsmallest"])
def test_nlargest_and_smallest_noop(data, groups, dtype, method):
    # GH 15272, GH 16345, GH 29129
    # Test nlargest/smallest when it results in a noop,
    # i.e. input is sorted and group size <= n
    if dtype is not None:
        data = np.array(data, dtype=dtype)
    if method == "nlargest":
        data = list(reversed(data))
    ser = Series(data, name="a")
    result = getattr(ser.groupby(groups), method)(n=2)
    expidx = np.array(groups, dtype=np.int_) if isinstance(groups, list) else groups
    expected = Series(data, index=MultiIndex.from_arrays([expidx, ser.index]), name="a")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("func", ["cumprod", "cumsum"])
def test_numpy_compat(func):
    # see gh-12811
    df = DataFrame({"A": [1, 2, 1], "B": [1, 2, 3]})
    g = df.groupby("A")

    msg = "numpy operations are not valid with groupby"

    with pytest.raises(UnsupportedFunctionCall, match=msg):
        getattr(g, func)(1, 2, 3)
    with pytest.raises(UnsupportedFunctionCall, match=msg):
        getattr(g, func)(foo=1)


def test_cummin(dtypes_for_minmax):
    dtype = dtypes_for_minmax[0]
    min_val = dtypes_for_minmax[1]

    # GH 15048
    base_df = DataFrame({"A": [1, 1, 1, 1, 2, 2, 2, 2], "B": [3, 4, 3, 2, 2, 3, 2, 1]})
    expected_mins = [3, 3, 3, 2, 2, 2, 2, 1]

    df = base_df.astype(dtype)

    expected = DataFrame({"B": expected_mins}).astype(dtype)
    result = df.groupby("A").cummin()
    tm.assert_frame_equal(result, expected)
    result = df.groupby("A", group_keys=False).B.apply(lambda x: x.cummin()).to_frame()
    tm.assert_frame_equal(result, expected)

    # Test w/ min value for dtype
    df.loc[[2, 6], "B"] = min_val
    df.loc[[1, 5], "B"] = min_val + 1
    expected.loc[[2, 3, 6, 7], "B"] = min_val
    expected.loc[[1, 5], "B"] = min_val + 1  # should not be rounded to min_val
    result = df.groupby("A").cummin()
    tm.assert_frame_equal(result, expected, check_exact=True)
    expected = (
        df.groupby("A", group_keys=False).B.apply(lambda x: x.cummin()).to_frame()
    )
    tm.assert_frame_equal(result, expected, check_exact=True)

    # Test nan in some values
    # Explicit cast to float to avoid implicit cast when setting nan
    base_df = base_df.astype({"B": "float"})
    base_df.loc[[0, 2, 4, 6], "B"] = np.nan
    expected = DataFrame({"B": [np.nan, 4, np.nan, 2, np.nan, 3, np.nan, 1]})
    result = base_df.groupby("A").cummin()
    tm.assert_frame_equal(result, expected)
    expected = (
        base_df.groupby("A", group_keys=False).B.apply(lambda x: x.cummin()).to_frame()
    )
    tm.assert_frame_equal(result, expected)

    # GH 15561
    df = DataFrame({"a": [1], "b": pd.to_datetime(["2001"])})
    expected = Series(pd.to_datetime("2001"), index=[0], name="b")

    result = df.groupby("a")["b"].cummin()
    tm.assert_series_equal(expected, result)

    # GH 15635
    df = DataFrame({"a": [1, 2, 1], "b": [1, 2, 2]})
    result = df.groupby("a").b.cummin()
    expected = Series([1, 2, 1], name="b")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("method", ["cummin", "cummax"])
@pytest.mark.parametrize("dtype", ["UInt64", "Int64", "Float64", "float", "boolean"])
def test_cummin_max_all_nan_column(method, dtype):
    base_df = DataFrame({"A": [1, 1, 1, 1, 2, 2, 2, 2], "B": [np.nan] * 8})
    base_df["B"] = base_df["B"].astype(dtype)
    grouped = base_df.groupby("A")

    expected = DataFrame({"B": [np.nan] * 8}, dtype=dtype)
    result = getattr(grouped, method)()
    tm.assert_frame_equal(expected, result)

    result = getattr(grouped["B"], method)().to_frame()
    tm.assert_frame_equal(expected, result)


def test_cummax(dtypes_for_minmax):
    dtype = dtypes_for_minmax[0]
    max_val = dtypes_for_minmax[2]

    # GH 15048
    base_df = DataFrame({"A": [1, 1, 1, 1, 2, 2, 2, 2], "B": [3, 4, 3, 2, 2, 3, 2, 1]})
    expected_maxs = [3, 4, 4, 4, 2, 3, 3, 3]

    df = base_df.astype(dtype)

    expected = DataFrame({"B": expected_maxs}).astype(dtype)
    result = df.groupby("A").cummax()
    tm.assert_frame_equal(result, expected)
    result = df.groupby("A", group_keys=False).B.apply(lambda x: x.cummax()).to_frame()
    tm.assert_frame_equal(result, expected)

    # Test w/ max value for dtype
    df.loc[[2, 6], "B"] = max_val
    expected.loc[[2, 3, 6, 7], "B"] = max_val
    result = df.groupby("A").cummax()
    tm.assert_frame_equal(result, expected)
    expected = (
        df.groupby("A", group_keys=False).B.apply(lambda x: x.cummax()).to_frame()
    )
    tm.assert_frame_equal(result, expected)

    # Test nan in some values
    # Explicit cast to float to avoid implicit cast when setting nan
    base_df = base_df.astype({"B": "float"})
    base_df.loc[[0, 2, 4, 6], "B"] = np.nan
    expected = DataFrame({"B": [np.nan, 4, np.nan, 4, np.nan, 3, np.nan, 3]})
    result = base_df.groupby("A").cummax()
    tm.assert_frame_equal(result, expected)
    expected = (
        base_df.groupby("A", group_keys=False).B.apply(lambda x: x.cummax()).to_frame()
    )
    tm.assert_frame_equal(result, expected)

    # GH 15561
    df = DataFrame({"a": [1], "b": pd.to_datetime(["2001"])})
    expected = Series(pd.to_datetime("2001"), index=[0], name="b")

    result = df.groupby("a")["b"].cummax()
    tm.assert_series_equal(expected, result)

    # GH 15635
    df = DataFrame({"a": [1, 2, 1], "b": [2, 1, 1]})
    result = df.groupby("a").b.cummax()
    expected = Series([2, 1, 2], name="b")
    tm.assert_series_equal(result, expected)


def test_cummax_i8_at_implementation_bound():
    # the minimum value used to be treated as NPY_NAT+1 instead of NPY_NAT
    #  for int64 dtype GH#46382
    ser = Series([pd.NaT._value + n for n in range(5)])
    df = DataFrame({"A": 1, "B": ser, "C": ser.view("M8[ns]")})
    gb = df.groupby("A")

    res = gb.cummax()
    exp = df[["B", "C"]]
    tm.assert_frame_equal(res, exp)


@pytest.mark.parametrize("method", ["cummin", "cummax"])
@pytest.mark.parametrize("dtype", ["float", "Int64", "Float64"])
@pytest.mark.parametrize(
    "groups,expected_data",
    [
        ([1, 1, 1], [1, None, None]),
        ([1, 2, 3], [1, None, 2]),
        ([1, 3, 3], [1, None, None]),
    ],
)
def test_cummin_max_skipna(method, dtype, groups, expected_data):
    # GH-34047
    df = DataFrame({"a": Series([1, None, 2], dtype=dtype)})
    orig = df.copy()
    gb = df.groupby(groups)["a"]

    result = getattr(gb, method)(skipna=False)
    expected = Series(expected_data, dtype=dtype, name="a")

    # check we didn't accidentally alter df
    tm.assert_frame_equal(df, orig)

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("method", ["cummin", "cummax"])
def test_cummin_max_skipna_multiple_cols(method):
    # Ensure missing value in "a" doesn't cause "b" to be nan-filled
    df = DataFrame({"a": [np.nan, 2.0, 2.0], "b": [2.0, 2.0, 2.0]})
    gb = df.groupby([1, 1, 1])[["a", "b"]]

    result = getattr(gb, method)(skipna=False)
    expected = DataFrame({"a": [np.nan, np.nan, np.nan], "b": [2.0, 2.0, 2.0]})

    tm.assert_frame_equal(result, expected)


@td.skip_if_32bit
@pytest.mark.parametrize("method", ["cummin", "cummax"])
@pytest.mark.parametrize(
    "dtype,val", [("UInt64", np.iinfo("uint64").max), ("Int64", 2**53 + 1)]
)
def test_nullable_int_not_cast_as_float(method, dtype, val):
    data = [val, pd.NA]
    df = DataFrame({"grp": [1, 1], "b": data}, dtype=dtype)
    grouped = df.groupby("grp")

    result = grouped.transform(method)
    expected = DataFrame({"b": data}, dtype=dtype)

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "in_vals, out_vals",
    [
        # Basics: strictly increasing (T), strictly decreasing (F),
        # abs val increasing (F), non-strictly increasing (T)
        ([1, 2, 5, 3, 2, 0, 4, 5, -6, 1, 1], [True, False, False, True]),
        # Test with inf vals
        (
            [1, 2.1, np.inf, 3, 2, np.inf, -np.inf, 5, 11, 1, -np.inf],
            [True, False, True, False],
        ),
        # Test with nan vals; should always be False
        (
            [1, 2, np.nan, 3, 2, np.nan, np.nan, 5, -np.inf, 1, np.nan],
            [False, False, False, False],
        ),
    ],
)
def test_is_monotonic_increasing(in_vals, out_vals):
    # GH 17015
    source_dict = {
        "A": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"],
        "B": ["a", "a", "a", "b", "b", "b", "c", "c", "c", "d", "d"],
        "C": in_vals,
    }
    df = DataFrame(source_dict)
    result = df.groupby("B").C.is_monotonic_increasing
    index = Index(list("abcd"), name="B")
    expected = Series(index=index, data=out_vals, name="C")
    tm.assert_series_equal(result, expected)

    # Also check result equal to manually taking x.is_monotonic_increasing.
    expected = df.groupby(["B"]).C.apply(lambda x: x.is_monotonic_increasing)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "in_vals, out_vals",
    [
        # Basics: strictly decreasing (T), strictly increasing (F),
        # abs val decreasing (F), non-strictly increasing (T)
        ([10, 9, 7, 3, 4, 5, -3, 2, 0, 1, 1], [True, False, False, True]),
        # Test with inf vals
        (
            [np.inf, 1, -np.inf, np.inf, 2, -3, -np.inf, 5, -3, -np.inf, -np.inf],
            [True, True, False, True],
        ),
        # Test with nan vals; should always be False
        (
            [1, 2, np.nan, 3, 2, np.nan, np.nan, 5, -np.inf, 1, np.nan],
            [False, False, False, False],
        ),
    ],
)
def test_is_monotonic_decreasing(in_vals, out_vals):
    # GH 17015
    source_dict = {
        "A": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"],
        "B": ["a", "a", "a", "b", "b", "b", "c", "c", "c", "d", "d"],
        "C": in_vals,
    }

    df = DataFrame(source_dict)
    result = df.groupby("B").C.is_monotonic_decreasing
    index = Index(list("abcd"), name="B")
    expected = Series(index=index, data=out_vals, name="C")
    tm.assert_series_equal(result, expected)


# describe
# --------------------------------


def test_apply_describe_bug(mframe):
    grouped = mframe.groupby(level="first")
    grouped.describe()  # it works!


def test_series_describe_multikey():
    ts = tm.makeTimeSeries()
    grouped = ts.groupby([lambda x: x.year, lambda x: x.month])
    result = grouped.describe()
    tm.assert_series_equal(result["mean"], grouped.mean(), check_names=False)
    tm.assert_series_equal(result["std"], grouped.std(), check_names=False)
    tm.assert_series_equal(result["min"], grouped.min(), check_names=False)


def test_series_describe_single():
    ts = tm.makeTimeSeries()
    grouped = ts.groupby(lambda x: x.month)
    result = grouped.apply(lambda x: x.describe())
    expected = grouped.describe().stack(future_stack=True)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("keys", ["key1", ["key1", "key2"]])
def test_series_describe_as_index(as_index, keys):
    # GH#49256
    df = DataFrame(
        {
            "key1": ["one", "two", "two", "three", "two"],
            "key2": ["one", "two", "two", "three", "two"],
            "foo2": [1, 2, 4, 4, 6],
        }
    )
    gb = df.groupby(keys, as_index=as_index)["foo2"]
    result = gb.describe()
    expected = DataFrame(
        {
            "key1": ["one", "three", "two"],
            "count": [1.0, 1.0, 3.0],
            "mean": [1.0, 4.0, 4.0],
            "std": [np.nan, np.nan, 2.0],
            "min": [1.0, 4.0, 2.0],
            "25%": [1.0, 4.0, 3.0],
            "50%": [1.0, 4.0, 4.0],
            "75%": [1.0, 4.0, 5.0],
            "max": [1.0, 4.0, 6.0],
        }
    )
    if len(keys) == 2:
        expected.insert(1, "key2", expected["key1"])
    if as_index:
        expected = expected.set_index(keys)
    tm.assert_frame_equal(result, expected)


def test_series_index_name(df):
    grouped = df.loc[:, ["C"]].groupby(df["A"])
    result = grouped.agg(lambda x: x.mean())
    assert result.index.name == "A"


def test_frame_describe_multikey(tsframe):
    grouped = tsframe.groupby([lambda x: x.year, lambda x: x.month])
    result = grouped.describe()
    desc_groups = []
    for col in tsframe:
        group = grouped[col].describe()
        # GH 17464 - Remove duplicate MultiIndex levels
        group_col = MultiIndex(
            levels=[[col], group.columns],
            codes=[[0] * len(group.columns), range(len(group.columns))],
        )
        group = DataFrame(group.values, columns=group_col, index=group.index)
        desc_groups.append(group)
    expected = pd.concat(desc_groups, axis=1)
    tm.assert_frame_equal(result, expected)

    msg = "DataFrame.groupby with axis=1 is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        groupedT = tsframe.groupby({"A": 0, "B": 0, "C": 1, "D": 1}, axis=1)
    result = groupedT.describe()
    expected = tsframe.describe().T
    # reverting the change from https://github.com/pandas-dev/pandas/pull/35441/
    expected.index = MultiIndex(
        levels=[[0, 1], expected.index],
        codes=[[0, 0, 1, 1], range(len(expected.index))],
    )
    tm.assert_frame_equal(result, expected)


def test_frame_describe_tupleindex():
    # GH 14848 - regression from 0.19.0 to 0.19.1
    df1 = DataFrame(
        {
            "x": [1, 2, 3, 4, 5] * 3,
            "y": [10, 20, 30, 40, 50] * 3,
            "z": [100, 200, 300, 400, 500] * 3,
        }
    )
    df1["k"] = [(0, 0, 1), (0, 1, 0), (1, 0, 0)] * 5
    df2 = df1.rename(columns={"k": "key"})
    msg = "Names should be list-like for a MultiIndex"
    with pytest.raises(ValueError, match=msg):
        df1.groupby("k").describe()
    with pytest.raises(ValueError, match=msg):
        df2.groupby("key").describe()


def test_frame_describe_unstacked_format():
    # GH 4792
    prices = {
        Timestamp("2011-01-06 10:59:05", tz=None): 24990,
        Timestamp("2011-01-06 12:43:33", tz=None): 25499,
        Timestamp("2011-01-06 12:54:09", tz=None): 25499,
    }
    volumes = {
        Timestamp("2011-01-06 10:59:05", tz=None): 1500000000,
        Timestamp("2011-01-06 12:43:33", tz=None): 5000000000,
        Timestamp("2011-01-06 12:54:09", tz=None): 100000000,
    }
    df = DataFrame({"PRICE": prices, "VOLUME": volumes})
    result = df.groupby("PRICE").VOLUME.describe()
    data = [
        df[df.PRICE == 24990].VOLUME.describe().values.tolist(),
        df[df.PRICE == 25499].VOLUME.describe().values.tolist(),
    ]
    expected = DataFrame(
        data,
        index=Index([24990, 25499], name="PRICE"),
        columns=["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.filterwarnings(
    "ignore:"
    "indexing past lexsort depth may impact performance:"
    "pandas.errors.PerformanceWarning"
)
@pytest.mark.parametrize("as_index", [True, False])
@pytest.mark.parametrize("keys", [["a1"], ["a1", "a2"]])
def test_describe_with_duplicate_output_column_names(as_index, keys):
    # GH 35314
    df = DataFrame(
        {
            "a1": [99, 99, 99, 88, 88, 88],
            "a2": [99, 99, 99, 88, 88, 88],
            "b": [1, 2, 3, 4, 5, 6],
            "c": [10, 20, 30, 40, 50, 60],
        },
        columns=["a1", "a2", "b", "b"],
        copy=False,
    )
    if keys == ["a1"]:
        df = df.drop(columns="a2")

    expected = (
        DataFrame.from_records(
            [
                ("b", "count", 3.0, 3.0),
                ("b", "mean", 5.0, 2.0),
                ("b", "std", 1.0, 1.0),
                ("b", "min", 4.0, 1.0),
                ("b", "25%", 4.5, 1.5),
                ("b", "50%", 5.0, 2.0),
                ("b", "75%", 5.5, 2.5),
                ("b", "max", 6.0, 3.0),
                ("b", "count", 3.0, 3.0),
                ("b", "mean", 5.0, 2.0),
                ("b", "std", 1.0, 1.0),
                ("b", "min", 4.0, 1.0),
                ("b", "25%", 4.5, 1.5),
                ("b", "50%", 5.0, 2.0),
                ("b", "75%", 5.5, 2.5),
                ("b", "max", 6.0, 3.0),
            ],
        )
        .set_index([0, 1])
        .T
    )
    expected.columns.names = [None, None]
    if len(keys) == 2:
        expected.index = MultiIndex(
            levels=[[88, 99], [88, 99]], codes=[[0, 1], [0, 1]], names=["a1", "a2"]
        )
    else:
        expected.index = Index([88, 99], name="a1")

    if not as_index:
        expected = expected.reset_index()

    result = df.groupby(keys, as_index=as_index).describe()

    tm.assert_frame_equal(result, expected)


def test_describe_duplicate_columns():
    # GH#50806
    df = DataFrame([[0, 1, 2, 3]])
    df.columns = [0, 1, 2, 0]
    gb = df.groupby(df[1])
    result = gb.describe(percentiles=[])

    columns = ["count", "mean", "std", "min", "50%", "max"]
    frames = [
        DataFrame([[1.0, val, np.nan, val, val, val]], index=[1], columns=columns)
        for val in (0.0, 2.0, 3.0)
    ]
    expected = pd.concat(frames, axis=1)
    expected.columns = MultiIndex(
        levels=[[0, 2], columns],
        codes=[6 * [0] + 6 * [1] + 6 * [0], 3 * list(range(6))],
    )
    expected.index.names = [1]
    tm.assert_frame_equal(result, expected)


def test_groupby_mean_no_overflow():
    # Regression test for (#22487)
    df = DataFrame(
        {
            "user": ["A", "A", "A", "A", "A"],
            "connections": [4970, 4749, 4719, 4704, 18446744073699999744],
        }
    )
    assert df.groupby("user")["connections"].mean()["A"] == 3689348814740003840


@pytest.mark.parametrize(
    "values",
    [
        {
            "a": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "b": [1, pd.NA, 2, 1, pd.NA, 2, 1, pd.NA, 2],
        },
        {"a": [1, 1, 2, 2, 3, 3], "b": [1, 2, 1, 2, 1, 2]},
    ],
)
@pytest.mark.parametrize("function", ["mean", "median", "var"])
def test_apply_to_nullable_integer_returns_float(values, function):
    # https://github.com/pandas-dev/pandas/issues/32219
    output = 0.5 if function == "var" else 1.5
    arr = np.array([output] * 3, dtype=float)
    idx = Index([1, 2, 3], name="a", dtype="Int64")
    expected = DataFrame({"b": arr}, index=idx).astype("Float64")

    groups = DataFrame(values, dtype="Int64").groupby("a")

    result = getattr(groups, function)()
    tm.assert_frame_equal(result, expected)

    result = groups.agg(function)
    tm.assert_frame_equal(result, expected)

    result = groups.agg([function])
    expected.columns = MultiIndex.from_tuples([("b", function)])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("min_count", [0, 10])
def test_groupby_sum_mincount_boolean(min_count):
    b = True
    a = False
    na = np.nan
    dfg = pd.array([b, b, na, na, a, a, b], dtype="boolean")

    df = DataFrame({"A": [1, 1, 2, 2, 3, 3, 1], "B": dfg})
    result = df.groupby("A").sum(min_count=min_count)
    if min_count == 0:
        expected = DataFrame(
            {"B": pd.array([3, 0, 0], dtype="Int64")},
            index=Index([1, 2, 3], name="A"),
        )
        tm.assert_frame_equal(result, expected)
    else:
        expected = DataFrame(
            {"B": pd.array([pd.NA] * 3, dtype="Int64")},
            index=Index([1, 2, 3], name="A"),
        )
        tm.assert_frame_equal(result, expected)


def test_groupby_sum_below_mincount_nullable_integer():
    # https://github.com/pandas-dev/pandas/issues/32861
    df = DataFrame({"a": [0, 1, 2], "b": [0, 1, 2], "c": [0, 1, 2]}, dtype="Int64")
    grouped = df.groupby("a")
    idx = Index([0, 1, 2], name="a", dtype="Int64")

    result = grouped["b"].sum(min_count=2)
    expected = Series([pd.NA] * 3, dtype="Int64", index=idx, name="b")
    tm.assert_series_equal(result, expected)

    result = grouped.sum(min_count=2)
    expected = DataFrame({"b": [pd.NA] * 3, "c": [pd.NA] * 3}, dtype="Int64", index=idx)
    tm.assert_frame_equal(result, expected)


def test_mean_on_timedelta():
    # GH 17382
    df = DataFrame({"time": pd.to_timedelta(range(10)), "cat": ["A", "B"] * 5})
    result = df.groupby("cat")["time"].mean()
    expected = Series(
        pd.to_timedelta([4, 5]), name="time", index=Index(["A", "B"], name="cat")
    )
    tm.assert_series_equal(result, expected)


def test_groupby_sum_timedelta_with_nat():
    # GH#42659
    df = DataFrame(
        {
            "a": [1, 1, 2, 2],
            "b": [pd.Timedelta("1d"), pd.Timedelta("2d"), pd.Timedelta("3d"), pd.NaT],
        }
    )
    td3 = pd.Timedelta(days=3)

    gb = df.groupby("a")

    res = gb.sum()
    expected = DataFrame({"b": [td3, td3]}, index=Index([1, 2], name="a"))
    tm.assert_frame_equal(res, expected)

    res = gb["b"].sum()
    tm.assert_series_equal(res, expected["b"])

    res = gb["b"].sum(min_count=2)
    expected = Series([td3, pd.NaT], dtype="m8[ns]", name="b", index=expected.index)
    tm.assert_series_equal(res, expected)


@pytest.mark.parametrize(
    "kernel, has_arg",
    [
        ("all", False),
        ("any", False),
        ("bfill", False),
        ("corr", True),
        ("corrwith", True),
        ("cov", True),
        ("cummax", True),
        ("cummin", True),
        ("cumprod", True),
        ("cumsum", True),
        ("diff", False),
        ("ffill", False),
        ("fillna", False),
        ("first", True),
        ("idxmax", True),
        ("idxmin", True),
        ("last", True),
        ("max", True),
        ("mean", True),
        ("median", True),
        ("min", True),
        ("nth", False),
        ("nunique", False),
        ("pct_change", False),
        ("prod", True),
        ("quantile", True),
        ("sem", True),
        ("skew", True),
        ("std", True),
        ("sum", True),
        ("var", True),
    ],
)
@pytest.mark.parametrize("numeric_only", [True, False, lib.no_default])
@pytest.mark.parametrize("keys", [["a1"], ["a1", "a2"]])
def test_numeric_only(kernel, has_arg, numeric_only, keys):
    # GH#46072
    # drops_nuisance: Whether the op drops nuisance columns even when numeric_only=False
    # has_arg: Whether the op has a numeric_only arg
    df = DataFrame({"a1": [1, 1], "a2": [2, 2], "a3": [5, 6], "b": 2 * [object]})

    args = get_groupby_method_args(kernel, df)
    kwargs = {} if numeric_only is lib.no_default else {"numeric_only": numeric_only}

    gb = df.groupby(keys)
    method = getattr(gb, kernel)
    if has_arg and numeric_only is True:
        # Cases where b does not appear in the result
        result = method(*args, **kwargs)
        assert "b" not in result.columns
    elif (
        # kernels that work on any dtype and have numeric_only arg
        kernel in ("first", "last")
        or (
            # kernels that work on any dtype and don't have numeric_only arg
            kernel in ("any", "all", "bfill", "ffill", "fillna", "nth", "nunique")
            and numeric_only is lib.no_default
        )
    ):
        result = method(*args, **kwargs)
        assert "b" in result.columns
    elif has_arg:
        assert numeric_only is not True
        # kernels that are successful on any dtype were above; this will fail

        # object dtypes for transformations are not implemented in Cython and
        # have no Python fallback
        exception = NotImplementedError if kernel.startswith("cum") else TypeError

        msg = "|".join(
            [
                "not allowed for this dtype",
                "cannot be performed against 'object' dtypes",
                # On PY39 message is "a number"; on PY310 and after is "a real number"
                "must be a string or a.* number",
                "unsupported operand type",
                "function is not implemented for this dtype",
                re.escape(f"agg function failed [how->{kernel},dtype->object]"),
            ]
        )
        if kernel == "idxmin":
            msg = "'<' not supported between instances of 'type' and 'type'"
        elif kernel == "idxmax":
            msg = "'>' not supported between instances of 'type' and 'type'"
        with pytest.raises(exception, match=msg):
            method(*args, **kwargs)
    elif not has_arg and numeric_only is not lib.no_default:
        with pytest.raises(
            TypeError, match="got an unexpected keyword argument 'numeric_only'"
        ):
            method(*args, **kwargs)
    else:
        assert kernel in ("diff", "pct_change")
        assert numeric_only is lib.no_default
        # Doesn't have numeric_only argument and fails on nuisance columns
        with pytest.raises(TypeError, match=r"unsupported operand type"):
            method(*args, **kwargs)


@pytest.mark.parametrize("dtype", [bool, int, float, object])
def test_deprecate_numeric_only_series(dtype, groupby_func, request):
    # GH#46560
    grouper = [0, 0, 1]

    ser = Series([1, 0, 0], dtype=dtype)
    gb = ser.groupby(grouper)

    if groupby_func == "corrwith":
        # corrwith is not implemented on SeriesGroupBy
        assert not hasattr(gb, groupby_func)
        return

    method = getattr(gb, groupby_func)

    expected_ser = Series([1, 0, 0])
    expected_gb = expected_ser.groupby(grouper)
    expected_method = getattr(expected_gb, groupby_func)

    args = get_groupby_method_args(groupby_func, ser)

    fails_on_numeric_object = (
        "corr",
        "cov",
        "cummax",
        "cummin",
        "cumprod",
        "cumsum",
        "quantile",
    )
    # ops that give an object result on object input
    obj_result = (
        "first",
        "last",
        "nth",
        "bfill",
        "ffill",
        "shift",
        "sum",
        "diff",
        "pct_change",
        "var",
        "mean",
        "median",
        "min",
        "max",
        "prod",
        "skew",
    )

    # Test default behavior; kernels that fail may be enabled in the future but kernels
    # that succeed should not be allowed to fail (without deprecation, at least)
    if groupby_func in fails_on_numeric_object and dtype is object:
        if groupby_func == "quantile":
            msg = "cannot be performed against 'object' dtypes"
        else:
            msg = "is not supported for object dtype"
        with pytest.raises(TypeError, match=msg):
            method(*args)
    elif dtype is object:
        result = method(*args)
        expected = expected_method(*args)
        if groupby_func in obj_result:
            expected = expected.astype(object)
        tm.assert_series_equal(result, expected)

    has_numeric_only = (
        "first",
        "last",
        "max",
        "mean",
        "median",
        "min",
        "prod",
        "quantile",
        "sem",
        "skew",
        "std",
        "sum",
        "var",
        "cummax",
        "cummin",
        "cumprod",
        "cumsum",
    )
    if groupby_func not in has_numeric_only:
        msg = "got an unexpected keyword argument 'numeric_only'"
        with pytest.raises(TypeError, match=msg):
            method(*args, numeric_only=True)
    elif dtype is object:
        msg = "|".join(
            [
                "SeriesGroupBy.sem called with numeric_only=True and dtype object",
                "Series.skew does not allow numeric_only=True with non-numeric",
                "cum(sum|prod|min|max) is not supported for object dtype",
                r"Cannot use numeric_only=True with SeriesGroupBy\..* and non-numeric",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            method(*args, numeric_only=True)
    elif dtype == bool and groupby_func == "quantile":
        msg = "Allowing bool dtype in SeriesGroupBy.quantile"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            # GH#51424
            result = method(*args, numeric_only=True)
            expected = method(*args, numeric_only=False)
        tm.assert_series_equal(result, expected)
    else:
        result = method(*args, numeric_only=True)
        expected = method(*args, numeric_only=False)
        tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("dtype", [int, float, object])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"percentiles": [0.10, 0.20, 0.30], "include": "all", "exclude": None},
        {"percentiles": [0.10, 0.20, 0.30], "include": None, "exclude": ["int"]},
        {"percentiles": [0.10, 0.20, 0.30], "include": ["int"], "exclude": None},
    ],
)
def test_groupby_empty_dataset(dtype, kwargs):
    # GH#41575
    df = DataFrame([[1, 2, 3]], columns=["A", "B", "C"], dtype=dtype)
    df["B"] = df["B"].astype(int)
    df["C"] = df["C"].astype(float)

    result = df.iloc[:0].groupby("A").describe(**kwargs)
    expected = df.groupby("A").describe(**kwargs).reset_index(drop=True).iloc[:0]
    tm.assert_frame_equal(result, expected)

    result = df.iloc[:0].groupby("A").B.describe(**kwargs)
    expected = df.groupby("A").B.describe(**kwargs).reset_index(drop=True).iloc[:0]
    expected.index = Index([])
    tm.assert_frame_equal(result, expected)


def test_corrwith_with_1_axis():
    # GH 47723
    df = DataFrame({"a": [1, 1, 2], "b": [3, 7, 4]})
    gb = df.groupby("a")

    msg = "DataFrameGroupBy.corrwith with axis=1 is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = gb.corrwith(df, axis=1)
    index = Index(
        data=[(1, 0), (1, 1), (1, 2), (2, 2), (2, 0), (2, 1)],
        name=("a", None),
    )
    expected = Series([np.nan] * 6, index=index)
    tm.assert_series_equal(result, expected)


def test_multiindex_group_all_columns_when_empty(groupby_func):
    # GH 32464
    df = DataFrame({"a": [], "b": [], "c": []}).set_index(["a", "b", "c"])
    gb = df.groupby(["a", "b", "c"], group_keys=False)
    method = getattr(gb, groupby_func)
    args = get_groupby_method_args(groupby_func, df)

    result = method(*args).index
    expected = df.index
    tm.assert_index_equal(result, expected)


def test_duplicate_columns(request, groupby_func, as_index):
    # GH#50806
    if groupby_func == "corrwith":
        msg = "GH#50845 - corrwith fails when there are duplicate columns"
        request.node.add_marker(pytest.mark.xfail(reason=msg))
    df = DataFrame([[1, 3, 6], [1, 4, 7], [2, 5, 8]], columns=list("abb"))
    args = get_groupby_method_args(groupby_func, df)
    gb = df.groupby("a", as_index=as_index)
    result = getattr(gb, groupby_func)(*args)

    expected_df = df.set_axis(["a", "b", "c"], axis=1)
    expected_args = get_groupby_method_args(groupby_func, expected_df)
    expected_gb = expected_df.groupby("a", as_index=as_index)
    expected = getattr(expected_gb, groupby_func)(*expected_args)
    if groupby_func not in ("size", "ngroup", "cumcount"):
        expected = expected.rename(columns={"c": "b"})
    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "op",
    [
        "sum",
        "prod",
        "min",
        "max",
        "median",
        "mean",
        "skew",
        "std",
        "var",
        "sem",
    ],
)
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("skipna", [True, False])
@pytest.mark.parametrize("sort", [True, False])
def test_regression_allowlist_methods(op, axis, skipna, sort):
    # GH6944
    # GH 17537
    # explicitly test the allowlist methods
    raw_frame = DataFrame([0])
    if axis == 0:
        frame = raw_frame
        msg = "The 'axis' keyword in DataFrame.groupby is deprecated and will be"
    else:
        frame = raw_frame.T
        msg = "DataFrame.groupby with axis=1 is deprecated"

    with tm.assert_produces_warning(FutureWarning, match=msg):
        grouped = frame.groupby(level=0, axis=axis, sort=sort)

    if op == "skew":
        # skew has skipna
        result = getattr(grouped, op)(skipna=skipna)
        expected = frame.groupby(level=0).apply(
            lambda h: getattr(h, op)(axis=axis, skipna=skipna)
        )
        if sort:
            expected = expected.sort_index(axis=axis)
        tm.assert_frame_equal(result, expected)
    else:
        result = getattr(grouped, op)()
        expected = frame.groupby(level=0).apply(lambda h: getattr(h, op)(axis=axis))
        if sort:
            expected = expected.sort_index(axis=axis)
        tm.assert_frame_equal(result, expected)
