import numpy as np
import pytest

import pandas as pd
from pandas import (
    Index,
    Interval,
    IntervalIndex,
    Timedelta,
    Timestamp,
    date_range,
    timedelta_range,
)
import pandas._testing as tm
from pandas.core.arrays import IntervalArray


@pytest.fixture(
    params=[
        (Index([0, 2, 4]), Index([1, 3, 5])),
        (Index([0.0, 1.0, 2.0]), Index([1.0, 2.0, 3.0])),
        (timedelta_range("0 days", periods=3), timedelta_range("1 day", periods=3)),
        (date_range("20170101", periods=3), date_range("20170102", periods=3)),
        (
            date_range("20170101", periods=3, tz="US/Eastern"),
            date_range("20170102", periods=3, tz="US/Eastern"),
        ),
    ],
    ids=lambda x: str(x[0].dtype),
)
def left_right_dtypes(request):
    """
    Fixture for building an IntervalArray from various dtypes
    """
    return request.param


class TestAttributes:
    @pytest.mark.parametrize(
        "left, right",
        [
            (0, 1),
            (Timedelta("0 days"), Timedelta("1 day")),
            (Timestamp("2018-01-01"), Timestamp("2018-01-02")),
            (
                Timestamp("2018-01-01", tz="US/Eastern"),
                Timestamp("2018-01-02", tz="US/Eastern"),
            ),
        ],
    )
    @pytest.mark.parametrize("constructor", [IntervalArray, IntervalIndex])
    def test_is_empty(self, constructor, left, right, closed):
        # GH27219
        tuples = [(left, left), (left, right), np.nan]
        expected = np.array([closed != "both", False, False])
        result = constructor.from_tuples(tuples, closed=closed).is_empty
        tm.assert_numpy_array_equal(result, expected)


class TestMethods:
    @pytest.mark.parametrize("new_closed", ["left", "right", "both", "neither"])
    def test_set_closed(self, closed, new_closed):
        # GH 21670
        array = IntervalArray.from_breaks(range(10), closed=closed)
        result = array.set_closed(new_closed)
        expected = IntervalArray.from_breaks(range(10), closed=new_closed)
        tm.assert_extension_array_equal(result, expected)

    @pytest.mark.parametrize(
        "other",
        [
            Interval(0, 1, closed="right"),
            IntervalArray.from_breaks([1, 2, 3, 4], closed="right"),
        ],
    )
    def test_where_raises(self, other):
        # GH#45768 The IntervalArray methods raises; the Series method coerces
        ser = pd.Series(IntervalArray.from_breaks([1, 2, 3, 4], closed="left"))
        mask = np.array([True, False, True])
        match = "'value.closed' is 'right', expected 'left'."
        with pytest.raises(ValueError, match=match):
            ser.array._where(mask, other)

        res = ser.where(mask, other=other)
        expected = ser.astype(object).where(mask, other)
        tm.assert_series_equal(res, expected)

    def test_shift(self):
        # https://github.com/pandas-dev/pandas/issues/31495, GH#22428, GH#31502
        a = IntervalArray.from_breaks([1, 2, 3])
        result = a.shift()
        # int -> float
        expected = IntervalArray.from_tuples([(np.nan, np.nan), (1.0, 2.0)])
        tm.assert_interval_array_equal(result, expected)

        msg = "can only insert Interval objects and NA into an IntervalArray"
        with pytest.raises(TypeError, match=msg):
            a.shift(1, fill_value=pd.NaT)

    def test_shift_datetime(self):
        # GH#31502, GH#31504
        a = IntervalArray.from_breaks(date_range("2000", periods=4))
        result = a.shift(2)
        expected = a.take([-1, -1, 0], allow_fill=True)
        tm.assert_interval_array_equal(result, expected)

        result = a.shift(-1)
        expected = a.take([1, 2, -1], allow_fill=True)
        tm.assert_interval_array_equal(result, expected)

        msg = "can only insert Interval objects and NA into an IntervalArray"
        with pytest.raises(TypeError, match=msg):
            a.shift(1, fill_value=np.timedelta64("NaT", "ns"))


class TestSetitem:
    def test_set_na(self, left_right_dtypes):
        left, right = left_right_dtypes
        left = left.copy(deep=True)
        right = right.copy(deep=True)
        result = IntervalArray.from_arrays(left, right)

        if result.dtype.subtype.kind not in ["m", "M"]:
            msg = "'value' should be an interval type, got <.*NaTType'> instead."
            with pytest.raises(TypeError, match=msg):
                result[0] = pd.NaT
        if result.dtype.subtype.kind in ["i", "u"]:
            msg = "Cannot set float NaN to integer-backed IntervalArray"
            # GH#45484 TypeError, not ValueError, matches what we get with
            # non-NA un-holdable value.
            with pytest.raises(TypeError, match=msg):
                result[0] = np.nan
            return

        result[0] = np.nan

        expected_left = Index([left._na_value] + list(left[1:]))
        expected_right = Index([right._na_value] + list(right[1:]))
        expected = IntervalArray.from_arrays(expected_left, expected_right)

        tm.assert_extension_array_equal(result, expected)

    def test_setitem_mismatched_closed(self):
        arr = IntervalArray.from_breaks(range(4))
        orig = arr.copy()
        other = arr.set_closed("both")

        msg = "'value.closed' is 'both', expected 'right'"
        with pytest.raises(ValueError, match=msg):
            arr[0] = other[0]
        with pytest.raises(ValueError, match=msg):
            arr[:1] = other[:1]
        with pytest.raises(ValueError, match=msg):
            arr[:0] = other[:0]
        with pytest.raises(ValueError, match=msg):
            arr[:] = other[::-1]
        with pytest.raises(ValueError, match=msg):
            arr[:] = list(other[::-1])
        with pytest.raises(ValueError, match=msg):
            arr[:] = other[::-1].astype(object)
        with pytest.raises(ValueError, match=msg):
            arr[:] = other[::-1].astype("category")

        # empty list should be no-op
        arr[:0] = []
        tm.assert_interval_array_equal(arr, orig)


def test_repr():
    # GH 25022
    arr = IntervalArray.from_tuples([(0, 1), (1, 2)])
    result = repr(arr)
    expected = (
        "<IntervalArray>\n"
        "[(0, 1], (1, 2]]\n"
        "Length: 2, dtype: interval[int64, right]"
    )
    assert result == expected


class TestReductions:
    def test_min_max_invalid_axis(self, left_right_dtypes):
        left, right = left_right_dtypes
        left = left.copy(deep=True)
        right = right.copy(deep=True)
        arr = IntervalArray.from_arrays(left, right)

        msg = "`axis` must be fewer than the number of dimensions"
        for axis in [-2, 1]:
            with pytest.raises(ValueError, match=msg):
                arr.min(axis=axis)
            with pytest.raises(ValueError, match=msg):
                arr.max(axis=axis)

        msg = "'>=' not supported between"
        with pytest.raises(TypeError, match=msg):
            arr.min(axis="foo")
        with pytest.raises(TypeError, match=msg):
            arr.max(axis="foo")

    def test_min_max(self, left_right_dtypes, index_or_series_or_array):
        # GH#44746
        left, right = left_right_dtypes
        left = left.copy(deep=True)
        right = right.copy(deep=True)
        arr = IntervalArray.from_arrays(left, right)

        # The expected results below are only valid if monotonic
        assert left.is_monotonic_increasing
        assert Index(arr).is_monotonic_increasing

        MIN = arr[0]
        MAX = arr[-1]

        indexer = np.arange(len(arr))
        np.random.default_rng(2).shuffle(indexer)
        arr = arr.take(indexer)

        arr_na = arr.insert(2, np.nan)

        arr = index_or_series_or_array(arr)
        arr_na = index_or_series_or_array(arr_na)

        for skipna in [True, False]:
            res = arr.min(skipna=skipna)
            assert res == MIN
            assert type(res) == type(MIN)

            res = arr.max(skipna=skipna)
            assert res == MAX
            assert type(res) == type(MAX)

        res = arr_na.min(skipna=False)
        assert np.isnan(res)
        res = arr_na.max(skipna=False)
        assert np.isnan(res)

        res = arr_na.min(skipna=True)
        assert res == MIN
        assert type(res) == type(MIN)
        res = arr_na.max(skipna=True)
        assert res == MAX
        assert type(res) == type(MAX)


# ----------------------------------------------------------------------------
# Arrow interaction


def test_arrow_extension_type():
    pa = pytest.importorskip("pyarrow")

    from pandas.core.arrays.arrow.extension_types import ArrowIntervalType

    p1 = ArrowIntervalType(pa.int64(), "left")
    p2 = ArrowIntervalType(pa.int64(), "left")
    p3 = ArrowIntervalType(pa.int64(), "right")

    assert p1.closed == "left"
    assert p1 == p2
    assert p1 != p3
    assert hash(p1) == hash(p2)
    assert hash(p1) != hash(p3)


def test_arrow_array():
    pa = pytest.importorskip("pyarrow")

    from pandas.core.arrays.arrow.extension_types import ArrowIntervalType

    intervals = pd.interval_range(1, 5, freq=1).array

    result = pa.array(intervals)
    assert isinstance(result.type, ArrowIntervalType)
    assert result.type.closed == intervals.closed
    assert result.type.subtype == pa.int64()
    assert result.storage.field("left").equals(pa.array([1, 2, 3, 4], type="int64"))
    assert result.storage.field("right").equals(pa.array([2, 3, 4, 5], type="int64"))

    expected = pa.array([{"left": i, "right": i + 1} for i in range(1, 5)])
    assert result.storage.equals(expected)

    # convert to its storage type
    result = pa.array(intervals, type=expected.type)
    assert result.equals(expected)

    # unsupported conversions
    with pytest.raises(TypeError, match="Not supported to convert IntervalArray"):
        pa.array(intervals, type="float64")

    with pytest.raises(TypeError, match="Not supported to convert IntervalArray"):
        pa.array(intervals, type=ArrowIntervalType(pa.float64(), "left"))


def test_arrow_array_missing():
    pa = pytest.importorskip("pyarrow")

    from pandas.core.arrays.arrow.extension_types import ArrowIntervalType

    arr = IntervalArray.from_breaks([0.0, 1.0, 2.0, 3.0])
    arr[1] = None

    result = pa.array(arr)
    assert isinstance(result.type, ArrowIntervalType)
    assert result.type.closed == arr.closed
    assert result.type.subtype == pa.float64()

    # fields have missing values (not NaN)
    left = pa.array([0.0, None, 2.0], type="float64")
    right = pa.array([1.0, None, 3.0], type="float64")
    assert result.storage.field("left").equals(left)
    assert result.storage.field("right").equals(right)

    # structarray itself also has missing values on the array level
    vals = [
        {"left": 0.0, "right": 1.0},
        {"left": None, "right": None},
        {"left": 2.0, "right": 3.0},
    ]
    expected = pa.StructArray.from_pandas(vals, mask=np.array([False, True, False]))
    assert result.storage.equals(expected)


@pytest.mark.parametrize(
    "breaks",
    [[0.0, 1.0, 2.0, 3.0], date_range("2017", periods=4, freq="D")],
    ids=["float", "datetime64[ns]"],
)
def test_arrow_table_roundtrip(breaks):
    pa = pytest.importorskip("pyarrow")

    from pandas.core.arrays.arrow.extension_types import ArrowIntervalType

    arr = IntervalArray.from_breaks(breaks)
    arr[1] = None
    df = pd.DataFrame({"a": arr})

    table = pa.table(df)
    assert isinstance(table.field("a").type, ArrowIntervalType)
    result = table.to_pandas()
    assert isinstance(result["a"].dtype, pd.IntervalDtype)
    tm.assert_frame_equal(result, df)

    table2 = pa.concat_tables([table, table])
    result = table2.to_pandas()
    expected = pd.concat([df, df], ignore_index=True)
    tm.assert_frame_equal(result, expected)

    # GH-41040
    table = pa.table(
        [pa.chunked_array([], type=table.column(0).type)], schema=table.schema
    )
    result = table.to_pandas()
    tm.assert_frame_equal(result, expected[0:0])


@pytest.mark.parametrize(
    "breaks",
    [[0.0, 1.0, 2.0, 3.0], date_range("2017", periods=4, freq="D")],
    ids=["float", "datetime64[ns]"],
)
def test_arrow_table_roundtrip_without_metadata(breaks):
    pa = pytest.importorskip("pyarrow")

    arr = IntervalArray.from_breaks(breaks)
    arr[1] = None
    df = pd.DataFrame({"a": arr})

    table = pa.table(df)
    # remove the metadata
    table = table.replace_schema_metadata()
    assert table.schema.metadata is None

    result = table.to_pandas()
    assert isinstance(result["a"].dtype, pd.IntervalDtype)
    tm.assert_frame_equal(result, df)


def test_from_arrow_from_raw_struct_array():
    # in case pyarrow lost the Interval extension type (eg on parquet roundtrip
    # with datetime64[ns] subtype, see GH-45881), still allow conversion
    # from arrow to IntervalArray
    pa = pytest.importorskip("pyarrow")

    arr = pa.array([{"left": 0, "right": 1}, {"left": 1, "right": 2}])
    dtype = pd.IntervalDtype(np.dtype("int64"), closed="neither")

    result = dtype.__from_arrow__(arr)
    expected = IntervalArray.from_breaks(
        np.array([0, 1, 2], dtype="int64"), closed="neither"
    )
    tm.assert_extension_array_equal(result, expected)

    result = dtype.__from_arrow__(pa.chunked_array([arr]))
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize("timezone", ["UTC", "US/Pacific", "GMT"])
def test_interval_index_subtype(timezone, inclusive_endpoints_fixture):
    # GH 46999
    dates = date_range("2022", periods=3, tz=timezone)
    dtype = f"interval[datetime64[ns, {timezone}], {inclusive_endpoints_fixture}]"
    result = IntervalIndex.from_arrays(
        ["2022-01-01", "2022-01-02"],
        ["2022-01-02", "2022-01-03"],
        closed=inclusive_endpoints_fixture,
        dtype=dtype,
    )
    expected = IntervalIndex.from_arrays(
        dates[:-1], dates[1:], closed=inclusive_endpoints_fixture
    )
    tm.assert_index_equal(result, expected)
