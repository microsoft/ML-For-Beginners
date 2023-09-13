import datetime
import re

import numpy as np
import pytest

from pandas._libs.tslibs import Timestamp
from pandas.compat import is_platform_windows

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    Series,
    _testing as tm,
    bdate_range,
    read_hdf,
)
from pandas.tests.io.pytables.common import (
    _maybe_remove,
    ensure_clean_store,
)
from pandas.util import _test_decorators as td

pytestmark = pytest.mark.single_cpu


def test_conv_read_write():
    with tm.ensure_clean() as path:

        def roundtrip(key, obj, **kwargs):
            obj.to_hdf(path, key, **kwargs)
            return read_hdf(path, key)

        o = tm.makeTimeSeries()
        tm.assert_series_equal(o, roundtrip("series", o))

        o = tm.makeStringSeries()
        tm.assert_series_equal(o, roundtrip("string_series", o))

        o = tm.makeDataFrame()
        tm.assert_frame_equal(o, roundtrip("frame", o))

        # table
        df = DataFrame({"A": range(5), "B": range(5)})
        df.to_hdf(path, "table", append=True)
        result = read_hdf(path, "table", where=["index>2"])
        tm.assert_frame_equal(df[df.index > 2], result)


def test_long_strings(setup_path):
    # GH6166
    df = DataFrame({"a": tm.makeStringIndex(10)}, index=tm.makeStringIndex(10))

    with ensure_clean_store(setup_path) as store:
        store.append("df", df, data_columns=["a"])

        result = store.select("df")
        tm.assert_frame_equal(df, result)


def test_api(tmp_path, setup_path):
    # GH4584
    # API issue when to_hdf doesn't accept append AND format args
    path = tmp_path / setup_path

    df = tm.makeDataFrame()
    df.iloc[:10].to_hdf(path, "df", append=True, format="table")
    df.iloc[10:].to_hdf(path, "df", append=True, format="table")
    tm.assert_frame_equal(read_hdf(path, "df"), df)

    # append to False
    df.iloc[:10].to_hdf(path, "df", append=False, format="table")
    df.iloc[10:].to_hdf(path, "df", append=True, format="table")
    tm.assert_frame_equal(read_hdf(path, "df"), df)


def test_api_append(tmp_path, setup_path):
    path = tmp_path / setup_path

    df = tm.makeDataFrame()
    df.iloc[:10].to_hdf(path, "df", append=True)
    df.iloc[10:].to_hdf(path, "df", append=True, format="table")
    tm.assert_frame_equal(read_hdf(path, "df"), df)

    # append to False
    df.iloc[:10].to_hdf(path, "df", append=False, format="table")
    df.iloc[10:].to_hdf(path, "df", append=True)
    tm.assert_frame_equal(read_hdf(path, "df"), df)


def test_api_2(tmp_path, setup_path):
    path = tmp_path / setup_path

    df = tm.makeDataFrame()
    df.to_hdf(path, "df", append=False, format="fixed")
    tm.assert_frame_equal(read_hdf(path, "df"), df)

    df.to_hdf(path, "df", append=False, format="f")
    tm.assert_frame_equal(read_hdf(path, "df"), df)

    df.to_hdf(path, "df", append=False)
    tm.assert_frame_equal(read_hdf(path, "df"), df)

    df.to_hdf(path, "df")
    tm.assert_frame_equal(read_hdf(path, "df"), df)

    with ensure_clean_store(setup_path) as store:
        df = tm.makeDataFrame()

        _maybe_remove(store, "df")
        store.append("df", df.iloc[:10], append=True, format="table")
        store.append("df", df.iloc[10:], append=True, format="table")
        tm.assert_frame_equal(store.select("df"), df)

        # append to False
        _maybe_remove(store, "df")
        store.append("df", df.iloc[:10], append=False, format="table")
        store.append("df", df.iloc[10:], append=True, format="table")
        tm.assert_frame_equal(store.select("df"), df)

        # formats
        _maybe_remove(store, "df")
        store.append("df", df.iloc[:10], append=False, format="table")
        store.append("df", df.iloc[10:], append=True, format="table")
        tm.assert_frame_equal(store.select("df"), df)

        _maybe_remove(store, "df")
        store.append("df", df.iloc[:10], append=False, format="table")
        store.append("df", df.iloc[10:], append=True, format=None)
        tm.assert_frame_equal(store.select("df"), df)


def test_api_invalid(tmp_path, setup_path):
    path = tmp_path / setup_path
    # Invalid.
    df = tm.makeDataFrame()

    msg = "Can only append to Tables"

    with pytest.raises(ValueError, match=msg):
        df.to_hdf(path, "df", append=True, format="f")

    with pytest.raises(ValueError, match=msg):
        df.to_hdf(path, "df", append=True, format="fixed")

    msg = r"invalid HDFStore format specified \[foo\]"

    with pytest.raises(TypeError, match=msg):
        df.to_hdf(path, "df", append=True, format="foo")

    with pytest.raises(TypeError, match=msg):
        df.to_hdf(path, "df", append=False, format="foo")

    # File path doesn't exist
    path = ""
    msg = f"File {path} does not exist"

    with pytest.raises(FileNotFoundError, match=msg):
        read_hdf(path, "df")


def test_get(setup_path):
    with ensure_clean_store(setup_path) as store:
        store["a"] = tm.makeTimeSeries()
        left = store.get("a")
        right = store["a"]
        tm.assert_series_equal(left, right)

        left = store.get("/a")
        right = store["/a"]
        tm.assert_series_equal(left, right)

        with pytest.raises(KeyError, match="'No object named b in the file'"):
            store.get("b")


def test_put_integer(setup_path):
    # non-date, non-string index
    df = DataFrame(np.random.default_rng(2).standard_normal((50, 100)))
    _check_roundtrip(df, tm.assert_frame_equal, setup_path)


def test_table_values_dtypes_roundtrip(setup_path):
    with ensure_clean_store(setup_path) as store:
        df1 = DataFrame({"a": [1, 2, 3]}, dtype="f8")
        store.append("df_f8", df1)
        tm.assert_series_equal(df1.dtypes, store["df_f8"].dtypes)

        df2 = DataFrame({"a": [1, 2, 3]}, dtype="i8")
        store.append("df_i8", df2)
        tm.assert_series_equal(df2.dtypes, store["df_i8"].dtypes)

        # incompatible dtype
        msg = re.escape(
            "invalid combination of [values_axes] on appending data "
            "[name->values_block_0,cname->values_block_0,"
            "dtype->float64,kind->float,shape->(1, 3)] vs "
            "current table [name->values_block_0,"
            "cname->values_block_0,dtype->int64,kind->integer,"
            "shape->None]"
        )
        with pytest.raises(ValueError, match=msg):
            store.append("df_i8", df1)

        # check creation/storage/retrieval of float32 (a bit hacky to
        # actually create them thought)
        df1 = DataFrame(np.array([[1], [2], [3]], dtype="f4"), columns=["A"])
        store.append("df_f4", df1)
        tm.assert_series_equal(df1.dtypes, store["df_f4"].dtypes)
        assert df1.dtypes.iloc[0] == "float32"

        # check with mixed dtypes
        df1 = DataFrame(
            {
                c: Series(np.random.default_rng(2).integers(5), dtype=c)
                for c in ["float32", "float64", "int32", "int64", "int16", "int8"]
            }
        )
        df1["string"] = "foo"
        df1["float322"] = 1.0
        df1["float322"] = df1["float322"].astype("float32")
        df1["bool"] = df1["float32"] > 0
        df1["time1"] = Timestamp("20130101")
        df1["time2"] = Timestamp("20130102")

        store.append("df_mixed_dtypes1", df1)
        result = store.select("df_mixed_dtypes1").dtypes.value_counts()
        result.index = [str(i) for i in result.index]
        expected = Series(
            {
                "float32": 2,
                "float64": 1,
                "int32": 1,
                "bool": 1,
                "int16": 1,
                "int8": 1,
                "int64": 1,
                "object": 1,
                "datetime64[ns]": 2,
            },
            name="count",
        )
        result = result.sort_index()
        expected = expected.sort_index()
        tm.assert_series_equal(result, expected)


@pytest.mark.filterwarnings("ignore::pandas.errors.PerformanceWarning")
def test_series(setup_path):
    s = tm.makeStringSeries()
    _check_roundtrip(s, tm.assert_series_equal, path=setup_path)

    ts = tm.makeTimeSeries()
    _check_roundtrip(ts, tm.assert_series_equal, path=setup_path)

    ts2 = Series(ts.index, Index(ts.index, dtype=object))
    _check_roundtrip(ts2, tm.assert_series_equal, path=setup_path)

    ts3 = Series(ts.values, Index(np.asarray(ts.index, dtype=object), dtype=object))
    _check_roundtrip(
        ts3, tm.assert_series_equal, path=setup_path, check_index_type=False
    )


def test_float_index(setup_path):
    # GH #454
    index = np.random.default_rng(2).standard_normal(10)
    s = Series(np.random.default_rng(2).standard_normal(10), index=index)
    _check_roundtrip(s, tm.assert_series_equal, path=setup_path)


def test_tuple_index(setup_path):
    # GH #492
    col = np.arange(10)
    idx = [(0.0, 1.0), (2.0, 3.0), (4.0, 5.0)]
    data = np.random.default_rng(2).standard_normal(30).reshape((3, 10))
    DF = DataFrame(data, index=idx, columns=col)

    with tm.assert_produces_warning(pd.errors.PerformanceWarning):
        _check_roundtrip(DF, tm.assert_frame_equal, path=setup_path)


@pytest.mark.filterwarnings("ignore::pandas.errors.PerformanceWarning")
def test_index_types(setup_path):
    values = np.random.default_rng(2).standard_normal(2)

    func = lambda lhs, rhs: tm.assert_series_equal(lhs, rhs, check_index_type=True)

    ser = Series(values, [0, "y"])
    _check_roundtrip(ser, func, path=setup_path)

    ser = Series(values, [datetime.datetime.today(), 0])
    _check_roundtrip(ser, func, path=setup_path)

    ser = Series(values, ["y", 0])
    _check_roundtrip(ser, func, path=setup_path)

    ser = Series(values, [datetime.date.today(), "a"])
    _check_roundtrip(ser, func, path=setup_path)

    ser = Series(values, [0, "y"])
    _check_roundtrip(ser, func, path=setup_path)

    ser = Series(values, [datetime.datetime.today(), 0])
    _check_roundtrip(ser, func, path=setup_path)

    ser = Series(values, ["y", 0])
    _check_roundtrip(ser, func, path=setup_path)

    ser = Series(values, [datetime.date.today(), "a"])
    _check_roundtrip(ser, func, path=setup_path)

    ser = Series(values, [1.23, "b"])
    _check_roundtrip(ser, func, path=setup_path)

    ser = Series(values, [1, 1.53])
    _check_roundtrip(ser, func, path=setup_path)

    ser = Series(values, [1, 5])
    _check_roundtrip(ser, func, path=setup_path)

    ser = Series(values, [datetime.datetime(2012, 1, 1), datetime.datetime(2012, 1, 2)])
    _check_roundtrip(ser, func, path=setup_path)


def test_timeseries_preepoch(setup_path, request):
    dr = bdate_range("1/1/1940", "1/1/1960")
    ts = Series(np.random.default_rng(2).standard_normal(len(dr)), index=dr)
    try:
        _check_roundtrip(ts, tm.assert_series_equal, path=setup_path)
    except OverflowError:
        if is_platform_windows():
            request.node.add_marker(
                pytest.mark.xfail("known failure on some windows platforms")
            )
        raise


@pytest.mark.parametrize(
    "compression", [False, pytest.param(True, marks=td.skip_if_windows)]
)
def test_frame(compression, setup_path):
    df = tm.makeDataFrame()

    # put in some random NAs
    df.iloc[0, 0] = np.nan
    df.iloc[5, 3] = np.nan

    _check_roundtrip_table(
        df, tm.assert_frame_equal, path=setup_path, compression=compression
    )
    _check_roundtrip(
        df, tm.assert_frame_equal, path=setup_path, compression=compression
    )

    tdf = tm.makeTimeDataFrame()
    _check_roundtrip(
        tdf, tm.assert_frame_equal, path=setup_path, compression=compression
    )

    with ensure_clean_store(setup_path) as store:
        # not consolidated
        df["foo"] = np.random.default_rng(2).standard_normal(len(df))
        store["df"] = df
        recons = store["df"]
        assert recons._mgr.is_consolidated()

    # empty
    _check_roundtrip(df[:0], tm.assert_frame_equal, path=setup_path)


def test_empty_series_frame(setup_path):
    s0 = Series(dtype=object)
    s1 = Series(name="myseries", dtype=object)
    df0 = DataFrame()
    df1 = DataFrame(index=["a", "b", "c"])
    df2 = DataFrame(columns=["d", "e", "f"])

    _check_roundtrip(s0, tm.assert_series_equal, path=setup_path)
    _check_roundtrip(s1, tm.assert_series_equal, path=setup_path)
    _check_roundtrip(df0, tm.assert_frame_equal, path=setup_path)
    _check_roundtrip(df1, tm.assert_frame_equal, path=setup_path)
    _check_roundtrip(df2, tm.assert_frame_equal, path=setup_path)


@pytest.mark.parametrize("dtype", [np.int64, np.float64, object, "m8[ns]", "M8[ns]"])
def test_empty_series(dtype, setup_path):
    s = Series(dtype=dtype)
    _check_roundtrip(s, tm.assert_series_equal, path=setup_path)


def test_can_serialize_dates(setup_path):
    rng = [x.date() for x in bdate_range("1/1/2000", "1/30/2000")]
    frame = DataFrame(
        np.random.default_rng(2).standard_normal((len(rng), 4)), index=rng
    )

    _check_roundtrip(frame, tm.assert_frame_equal, path=setup_path)


def test_store_hierarchical(setup_path, multiindex_dataframe_random_data):
    frame = multiindex_dataframe_random_data

    _check_roundtrip(frame, tm.assert_frame_equal, path=setup_path)
    _check_roundtrip(frame.T, tm.assert_frame_equal, path=setup_path)
    _check_roundtrip(frame["A"], tm.assert_series_equal, path=setup_path)

    # check that the names are stored
    with ensure_clean_store(setup_path) as store:
        store["frame"] = frame
        recons = store["frame"]
        tm.assert_frame_equal(recons, frame)


@pytest.mark.parametrize(
    "compression", [False, pytest.param(True, marks=td.skip_if_windows)]
)
def test_store_mixed(compression, setup_path):
    def _make_one():
        df = tm.makeDataFrame()
        df["obj1"] = "foo"
        df["obj2"] = "bar"
        df["bool1"] = df["A"] > 0
        df["bool2"] = df["B"] > 0
        df["int1"] = 1
        df["int2"] = 2
        return df._consolidate()

    df1 = _make_one()
    df2 = _make_one()

    _check_roundtrip(df1, tm.assert_frame_equal, path=setup_path)
    _check_roundtrip(df2, tm.assert_frame_equal, path=setup_path)

    with ensure_clean_store(setup_path) as store:
        store["obj"] = df1
        tm.assert_frame_equal(store["obj"], df1)
        store["obj"] = df2
        tm.assert_frame_equal(store["obj"], df2)

    # check that can store Series of all of these types
    _check_roundtrip(
        df1["obj1"],
        tm.assert_series_equal,
        path=setup_path,
        compression=compression,
    )
    _check_roundtrip(
        df1["bool1"],
        tm.assert_series_equal,
        path=setup_path,
        compression=compression,
    )
    _check_roundtrip(
        df1["int1"],
        tm.assert_series_equal,
        path=setup_path,
        compression=compression,
    )


def _check_roundtrip(obj, comparator, path, compression=False, **kwargs):
    options = {}
    if compression:
        options["complib"] = "blosc"

    with ensure_clean_store(path, "w", **options) as store:
        store["obj"] = obj
        retrieved = store["obj"]
        comparator(retrieved, obj, **kwargs)


def _check_roundtrip_table(obj, comparator, path, compression=False):
    options = {}
    if compression:
        options["complib"] = "blosc"

    with ensure_clean_store(path, "w", **options) as store:
        store.put("obj", obj, format="table")
        retrieved = store["obj"]

        comparator(retrieved, obj)


def test_unicode_index(setup_path):
    unicode_values = ["\u03c3", "\u03c3\u03c3"]

    s = Series(
        np.random.default_rng(2).standard_normal(len(unicode_values)),
        unicode_values,
    )
    _check_roundtrip(s, tm.assert_series_equal, path=setup_path)


def test_unicode_longer_encoded(setup_path):
    # GH 11234
    char = "\u0394"
    df = DataFrame({"A": [char]})
    with ensure_clean_store(setup_path) as store:
        store.put("df", df, format="table", encoding="utf-8")
        result = store.get("df")
        tm.assert_frame_equal(result, df)

    df = DataFrame({"A": ["a", char], "B": ["b", "b"]})
    with ensure_clean_store(setup_path) as store:
        store.put("df", df, format="table", encoding="utf-8")
        result = store.get("df")
        tm.assert_frame_equal(result, df)


def test_store_datetime_mixed(setup_path):
    df = DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0], "c": ["a", "b", "c"]})
    ts = tm.makeTimeSeries()
    df["d"] = ts.index[:3]
    _check_roundtrip(df, tm.assert_frame_equal, path=setup_path)


def test_round_trip_equals(tmp_path, setup_path):
    # GH 9330
    df = DataFrame({"B": [1, 2], "A": ["x", "y"]})

    path = tmp_path / setup_path
    df.to_hdf(path, "df", format="table")
    other = read_hdf(path, "df")
    tm.assert_frame_equal(df, other)
    assert df.equals(other)
    assert other.equals(df)
