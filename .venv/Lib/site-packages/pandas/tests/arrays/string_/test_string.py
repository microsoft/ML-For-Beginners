"""
This module tests the functionality of StringArray and ArrowStringArray.
Tests for the str accessors are in pandas/tests/strings/test_string_array.py
"""
import operator

import numpy as np
import pytest

from pandas.compat.pyarrow import pa_version_under12p0

from pandas.core.dtypes.common import is_dtype_equal

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.string_arrow import (
    ArrowStringArray,
    ArrowStringArrayNumpySemantics,
)


def na_val(dtype):
    if dtype.storage == "pyarrow_numpy":
        return np.nan
    else:
        return pd.NA


@pytest.fixture
def dtype(string_storage):
    """Fixture giving StringDtype from parametrized 'string_storage'"""
    return pd.StringDtype(storage=string_storage)


@pytest.fixture
def cls(dtype):
    """Fixture giving array type from parametrized 'dtype'"""
    return dtype.construct_array_type()


def test_repr(dtype):
    df = pd.DataFrame({"A": pd.array(["a", pd.NA, "b"], dtype=dtype)})
    if dtype.storage == "pyarrow_numpy":
        expected = "     A\n0    a\n1  NaN\n2    b"
    else:
        expected = "      A\n0     a\n1  <NA>\n2     b"
    assert repr(df) == expected

    if dtype.storage == "pyarrow_numpy":
        expected = "0      a\n1    NaN\n2      b\nName: A, dtype: string"
    else:
        expected = "0       a\n1    <NA>\n2       b\nName: A, dtype: string"
    assert repr(df.A) == expected

    if dtype.storage == "pyarrow":
        arr_name = "ArrowStringArray"
        expected = f"<{arr_name}>\n['a', <NA>, 'b']\nLength: 3, dtype: string"
    elif dtype.storage == "pyarrow_numpy":
        arr_name = "ArrowStringArrayNumpySemantics"
        expected = f"<{arr_name}>\n['a', nan, 'b']\nLength: 3, dtype: string"
    else:
        arr_name = "StringArray"
        expected = f"<{arr_name}>\n['a', <NA>, 'b']\nLength: 3, dtype: string"
    assert repr(df.A.array) == expected


def test_none_to_nan(cls, dtype):
    a = cls._from_sequence(["a", None, "b"], dtype=dtype)
    assert a[1] is not None
    assert a[1] is na_val(a.dtype)


def test_setitem_validates(cls, dtype):
    arr = cls._from_sequence(["a", "b"], dtype=dtype)

    if cls is pd.arrays.StringArray:
        msg = "Cannot set non-string value '10' into a StringArray."
    else:
        msg = "Scalar must be NA or str"
    with pytest.raises(TypeError, match=msg):
        arr[0] = 10

    if cls is pd.arrays.StringArray:
        msg = "Must provide strings."
    else:
        msg = "Scalar must be NA or str"
    with pytest.raises(TypeError, match=msg):
        arr[:] = np.array([1, 2])


def test_setitem_with_scalar_string(dtype):
    # is_float_dtype considers some strings, like 'd', to be floats
    # which can cause issues.
    arr = pd.array(["a", "c"], dtype=dtype)
    arr[0] = "d"
    expected = pd.array(["d", "c"], dtype=dtype)
    tm.assert_extension_array_equal(arr, expected)


def test_setitem_with_array_with_missing(dtype):
    # ensure that when setting with an array of values, we don't mutate the
    # array `value` in __setitem__(self, key, value)
    arr = pd.array(["a", "b", "c"], dtype=dtype)
    value = np.array(["A", None])
    value_orig = value.copy()
    arr[[0, 1]] = value

    expected = pd.array(["A", pd.NA, "c"], dtype=dtype)
    tm.assert_extension_array_equal(arr, expected)
    tm.assert_numpy_array_equal(value, value_orig)


def test_astype_roundtrip(dtype):
    ser = pd.Series(pd.date_range("2000", periods=12))
    ser[0] = None

    casted = ser.astype(dtype)
    assert is_dtype_equal(casted.dtype, dtype)

    result = casted.astype("datetime64[ns]")
    tm.assert_series_equal(result, ser)

    # GH#38509 same thing for timedelta64
    ser2 = ser - ser.iloc[-1]
    casted2 = ser2.astype(dtype)
    assert is_dtype_equal(casted2.dtype, dtype)

    result2 = casted2.astype(ser2.dtype)
    tm.assert_series_equal(result2, ser2)


def test_add(dtype):
    a = pd.Series(["a", "b", "c", None, None], dtype=dtype)
    b = pd.Series(["x", "y", None, "z", None], dtype=dtype)

    result = a + b
    expected = pd.Series(["ax", "by", None, None, None], dtype=dtype)
    tm.assert_series_equal(result, expected)

    result = a.add(b)
    tm.assert_series_equal(result, expected)

    result = a.radd(b)
    expected = pd.Series(["xa", "yb", None, None, None], dtype=dtype)
    tm.assert_series_equal(result, expected)

    result = a.add(b, fill_value="-")
    expected = pd.Series(["ax", "by", "c-", "-z", None], dtype=dtype)
    tm.assert_series_equal(result, expected)


def test_add_2d(dtype, request, arrow_string_storage):
    if dtype.storage in arrow_string_storage:
        reason = "Failed: DID NOT RAISE <class 'ValueError'>"
        mark = pytest.mark.xfail(raises=None, reason=reason)
        request.applymarker(mark)

    a = pd.array(["a", "b", "c"], dtype=dtype)
    b = np.array([["a", "b", "c"]], dtype=object)
    with pytest.raises(ValueError, match="3 != 1"):
        a + b

    s = pd.Series(a)
    with pytest.raises(ValueError, match="3 != 1"):
        s + b


def test_add_sequence(dtype):
    a = pd.array(["a", "b", None, None], dtype=dtype)
    other = ["x", None, "y", None]

    result = a + other
    expected = pd.array(["ax", None, None, None], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)

    result = other + a
    expected = pd.array(["xa", None, None, None], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)


def test_mul(dtype):
    a = pd.array(["a", "b", None], dtype=dtype)
    result = a * 2
    expected = pd.array(["aa", "bb", None], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)

    result = 2 * a
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.xfail(reason="GH-28527")
def test_add_strings(dtype):
    arr = pd.array(["a", "b", "c", "d"], dtype=dtype)
    df = pd.DataFrame([["t", "y", "v", "w"]], dtype=object)
    assert arr.__add__(df) is NotImplemented

    result = arr + df
    expected = pd.DataFrame([["at", "by", "cv", "dw"]]).astype(dtype)
    tm.assert_frame_equal(result, expected)

    result = df + arr
    expected = pd.DataFrame([["ta", "yb", "vc", "wd"]]).astype(dtype)
    tm.assert_frame_equal(result, expected)


@pytest.mark.xfail(reason="GH-28527")
def test_add_frame(dtype):
    arr = pd.array(["a", "b", np.nan, np.nan], dtype=dtype)
    df = pd.DataFrame([["x", np.nan, "y", np.nan]])

    assert arr.__add__(df) is NotImplemented

    result = arr + df
    expected = pd.DataFrame([["ax", np.nan, np.nan, np.nan]]).astype(dtype)
    tm.assert_frame_equal(result, expected)

    result = df + arr
    expected = pd.DataFrame([["xa", np.nan, np.nan, np.nan]]).astype(dtype)
    tm.assert_frame_equal(result, expected)


def test_comparison_methods_scalar(comparison_op, dtype):
    op_name = f"__{comparison_op.__name__}__"
    a = pd.array(["a", None, "c"], dtype=dtype)
    other = "a"
    result = getattr(a, op_name)(other)
    if dtype.storage == "pyarrow_numpy":
        expected = np.array([getattr(item, op_name)(other) for item in a])
        if comparison_op == operator.ne:
            expected[1] = True
        else:
            expected[1] = False
        tm.assert_numpy_array_equal(result, expected.astype(np.bool_))
    else:
        expected_dtype = "boolean[pyarrow]" if dtype.storage == "pyarrow" else "boolean"
        expected = np.array([getattr(item, op_name)(other) for item in a], dtype=object)
        expected = pd.array(expected, dtype=expected_dtype)
        tm.assert_extension_array_equal(result, expected)


def test_comparison_methods_scalar_pd_na(comparison_op, dtype):
    op_name = f"__{comparison_op.__name__}__"
    a = pd.array(["a", None, "c"], dtype=dtype)
    result = getattr(a, op_name)(pd.NA)

    if dtype.storage == "pyarrow_numpy":
        if operator.ne == comparison_op:
            expected = np.array([True, True, True])
        else:
            expected = np.array([False, False, False])
        tm.assert_numpy_array_equal(result, expected)
    else:
        expected_dtype = "boolean[pyarrow]" if dtype.storage == "pyarrow" else "boolean"
        expected = pd.array([None, None, None], dtype=expected_dtype)
        tm.assert_extension_array_equal(result, expected)
        tm.assert_extension_array_equal(result, expected)


def test_comparison_methods_scalar_not_string(comparison_op, dtype):
    op_name = f"__{comparison_op.__name__}__"

    a = pd.array(["a", None, "c"], dtype=dtype)
    other = 42

    if op_name not in ["__eq__", "__ne__"]:
        with pytest.raises(TypeError, match="Invalid comparison|not supported between"):
            getattr(a, op_name)(other)

        return

    result = getattr(a, op_name)(other)

    if dtype.storage == "pyarrow_numpy":
        expected_data = {
            "__eq__": [False, False, False],
            "__ne__": [True, True, True],
        }[op_name]
        expected = np.array(expected_data)
        tm.assert_numpy_array_equal(result, expected)
    else:
        expected_data = {"__eq__": [False, None, False], "__ne__": [True, None, True]}[
            op_name
        ]
        expected_dtype = "boolean[pyarrow]" if dtype.storage == "pyarrow" else "boolean"
        expected = pd.array(expected_data, dtype=expected_dtype)
        tm.assert_extension_array_equal(result, expected)


def test_comparison_methods_array(comparison_op, dtype):
    op_name = f"__{comparison_op.__name__}__"

    a = pd.array(["a", None, "c"], dtype=dtype)
    other = [None, None, "c"]
    result = getattr(a, op_name)(other)
    if dtype.storage == "pyarrow_numpy":
        if operator.ne == comparison_op:
            expected = np.array([True, True, False])
        else:
            expected = np.array([False, False, False])
            expected[-1] = getattr(other[-1], op_name)(a[-1])
        tm.assert_numpy_array_equal(result, expected)

        result = getattr(a, op_name)(pd.NA)
        if operator.ne == comparison_op:
            expected = np.array([True, True, True])
        else:
            expected = np.array([False, False, False])
        tm.assert_numpy_array_equal(result, expected)

    else:
        expected_dtype = "boolean[pyarrow]" if dtype.storage == "pyarrow" else "boolean"
        expected = np.full(len(a), fill_value=None, dtype="object")
        expected[-1] = getattr(other[-1], op_name)(a[-1])
        expected = pd.array(expected, dtype=expected_dtype)
        tm.assert_extension_array_equal(result, expected)

        result = getattr(a, op_name)(pd.NA)
        expected = pd.array([None, None, None], dtype=expected_dtype)
        tm.assert_extension_array_equal(result, expected)


def test_constructor_raises(cls):
    if cls is pd.arrays.StringArray:
        msg = "StringArray requires a sequence of strings or pandas.NA"
    else:
        msg = "Unsupported type '<class 'numpy.ndarray'>' for ArrowExtensionArray"

    with pytest.raises(ValueError, match=msg):
        cls(np.array(["a", "b"], dtype="S1"))

    with pytest.raises(ValueError, match=msg):
        cls(np.array([]))

    if cls is pd.arrays.StringArray:
        # GH#45057 np.nan and None do NOT raise, as they are considered valid NAs
        #  for string dtype
        cls(np.array(["a", np.nan], dtype=object))
        cls(np.array(["a", None], dtype=object))
    else:
        with pytest.raises(ValueError, match=msg):
            cls(np.array(["a", np.nan], dtype=object))
        with pytest.raises(ValueError, match=msg):
            cls(np.array(["a", None], dtype=object))

    with pytest.raises(ValueError, match=msg):
        cls(np.array(["a", pd.NaT], dtype=object))

    with pytest.raises(ValueError, match=msg):
        cls(np.array(["a", np.datetime64("NaT", "ns")], dtype=object))

    with pytest.raises(ValueError, match=msg):
        cls(np.array(["a", np.timedelta64("NaT", "ns")], dtype=object))


@pytest.mark.parametrize("na", [np.nan, np.float64("nan"), float("nan"), None, pd.NA])
def test_constructor_nan_like(na):
    expected = pd.arrays.StringArray(np.array(["a", pd.NA]))
    tm.assert_extension_array_equal(
        pd.arrays.StringArray(np.array(["a", na], dtype="object")), expected
    )


@pytest.mark.parametrize("copy", [True, False])
def test_from_sequence_no_mutate(copy, cls, dtype):
    nan_arr = np.array(["a", np.nan], dtype=object)
    expected_input = nan_arr.copy()
    na_arr = np.array(["a", pd.NA], dtype=object)

    result = cls._from_sequence(nan_arr, dtype=dtype, copy=copy)

    if cls in (ArrowStringArray, ArrowStringArrayNumpySemantics):
        import pyarrow as pa

        expected = cls(pa.array(na_arr, type=pa.string(), from_pandas=True))
    else:
        expected = cls(na_arr)

    tm.assert_extension_array_equal(result, expected)
    tm.assert_numpy_array_equal(nan_arr, expected_input)


def test_astype_int(dtype):
    arr = pd.array(["1", "2", "3"], dtype=dtype)
    result = arr.astype("int64")
    expected = np.array([1, 2, 3], dtype="int64")
    tm.assert_numpy_array_equal(result, expected)

    arr = pd.array(["1", pd.NA, "3"], dtype=dtype)
    if dtype.storage == "pyarrow_numpy":
        err = ValueError
        msg = "cannot convert float NaN to integer"
    else:
        err = TypeError
        msg = (
            r"int\(\) argument must be a string, a bytes-like "
            r"object or a( real)? number"
        )
    with pytest.raises(err, match=msg):
        arr.astype("int64")


def test_astype_nullable_int(dtype):
    arr = pd.array(["1", pd.NA, "3"], dtype=dtype)

    result = arr.astype("Int64")
    expected = pd.array([1, pd.NA, 3], dtype="Int64")
    tm.assert_extension_array_equal(result, expected)


def test_astype_float(dtype, any_float_dtype):
    # Don't compare arrays (37974)
    ser = pd.Series(["1.1", pd.NA, "3.3"], dtype=dtype)
    result = ser.astype(any_float_dtype)
    expected = pd.Series([1.1, np.nan, 3.3], dtype=any_float_dtype)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("skipna", [True, False])
@pytest.mark.xfail(reason="Not implemented StringArray.sum")
def test_reduce(skipna, dtype):
    arr = pd.Series(["a", "b", "c"], dtype=dtype)
    result = arr.sum(skipna=skipna)
    assert result == "abc"


@pytest.mark.parametrize("skipna", [True, False])
@pytest.mark.xfail(reason="Not implemented StringArray.sum")
def test_reduce_missing(skipna, dtype):
    arr = pd.Series([None, "a", None, "b", "c", None], dtype=dtype)
    result = arr.sum(skipna=skipna)
    if skipna:
        assert result == "abc"
    else:
        assert pd.isna(result)


@pytest.mark.parametrize("method", ["min", "max"])
@pytest.mark.parametrize("skipna", [True, False])
def test_min_max(method, skipna, dtype):
    arr = pd.Series(["a", "b", "c", None], dtype=dtype)
    result = getattr(arr, method)(skipna=skipna)
    if skipna:
        expected = "a" if method == "min" else "c"
        assert result == expected
    else:
        assert result is na_val(arr.dtype)


@pytest.mark.parametrize("method", ["min", "max"])
@pytest.mark.parametrize("box", [pd.Series, pd.array])
def test_min_max_numpy(method, box, dtype, request, arrow_string_storage):
    if dtype.storage in arrow_string_storage and box is pd.array:
        if box is pd.array:
            reason = "'<=' not supported between instances of 'str' and 'NoneType'"
        else:
            reason = "'ArrowStringArray' object has no attribute 'max'"
        mark = pytest.mark.xfail(raises=TypeError, reason=reason)
        request.applymarker(mark)

    arr = box(["a", "b", "c", None], dtype=dtype)
    result = getattr(np, method)(arr)
    expected = "a" if method == "min" else "c"
    assert result == expected


def test_fillna_args(dtype, arrow_string_storage):
    # GH 37987

    arr = pd.array(["a", pd.NA], dtype=dtype)

    res = arr.fillna(value="b")
    expected = pd.array(["a", "b"], dtype=dtype)
    tm.assert_extension_array_equal(res, expected)

    res = arr.fillna(value=np.str_("b"))
    expected = pd.array(["a", "b"], dtype=dtype)
    tm.assert_extension_array_equal(res, expected)

    if dtype.storage in arrow_string_storage:
        msg = "Invalid value '1' for dtype string"
    else:
        msg = "Cannot set non-string value '1' into a StringArray."
    with pytest.raises(TypeError, match=msg):
        arr.fillna(value=1)


def test_arrow_array(dtype):
    # protocol added in 0.15.0
    pa = pytest.importorskip("pyarrow")
    import pyarrow.compute as pc

    data = pd.array(["a", "b", "c"], dtype=dtype)
    arr = pa.array(data)
    expected = pa.array(list(data), type=pa.large_string(), from_pandas=True)
    if dtype.storage in ("pyarrow", "pyarrow_numpy") and pa_version_under12p0:
        expected = pa.chunked_array(expected)
    if dtype.storage == "python":
        expected = pc.cast(expected, pa.string())
    assert arr.equals(expected)


@pytest.mark.filterwarnings("ignore:Passing a BlockManager:DeprecationWarning")
def test_arrow_roundtrip(dtype, string_storage2, request, using_infer_string):
    # roundtrip possible from arrow 1.0.0
    pa = pytest.importorskip("pyarrow")

    if using_infer_string and string_storage2 != "pyarrow_numpy":
        request.applymarker(
            pytest.mark.xfail(
                reason="infer_string takes precedence over string storage"
            )
        )

    data = pd.array(["a", "b", None], dtype=dtype)
    df = pd.DataFrame({"a": data})
    table = pa.table(df)
    if dtype.storage == "python":
        assert table.field("a").type == "string"
    else:
        assert table.field("a").type == "large_string"
    with pd.option_context("string_storage", string_storage2):
        result = table.to_pandas()
    assert isinstance(result["a"].dtype, pd.StringDtype)
    expected = df.astype(f"string[{string_storage2}]")
    tm.assert_frame_equal(result, expected)
    # ensure the missing value is represented by NA and not np.nan or None
    assert result.loc[2, "a"] is na_val(result["a"].dtype)


@pytest.mark.filterwarnings("ignore:Passing a BlockManager:DeprecationWarning")
def test_arrow_load_from_zero_chunks(
    dtype, string_storage2, request, using_infer_string
):
    # GH-41040
    pa = pytest.importorskip("pyarrow")

    if using_infer_string and string_storage2 != "pyarrow_numpy":
        request.applymarker(
            pytest.mark.xfail(
                reason="infer_string takes precedence over string storage"
            )
        )

    data = pd.array([], dtype=dtype)
    df = pd.DataFrame({"a": data})
    table = pa.table(df)
    if dtype.storage == "python":
        assert table.field("a").type == "string"
    else:
        assert table.field("a").type == "large_string"
    # Instantiate the same table with no chunks at all
    table = pa.table([pa.chunked_array([], type=pa.string())], schema=table.schema)
    with pd.option_context("string_storage", string_storage2):
        result = table.to_pandas()
    assert isinstance(result["a"].dtype, pd.StringDtype)
    expected = df.astype(f"string[{string_storage2}]")
    tm.assert_frame_equal(result, expected)


def test_value_counts_na(dtype):
    if getattr(dtype, "storage", "") == "pyarrow":
        exp_dtype = "int64[pyarrow]"
    elif getattr(dtype, "storage", "") == "pyarrow_numpy":
        exp_dtype = "int64"
    else:
        exp_dtype = "Int64"
    arr = pd.array(["a", "b", "a", pd.NA], dtype=dtype)
    result = arr.value_counts(dropna=False)
    expected = pd.Series([2, 1, 1], index=arr[[0, 1, 3]], dtype=exp_dtype, name="count")
    tm.assert_series_equal(result, expected)

    result = arr.value_counts(dropna=True)
    expected = pd.Series([2, 1], index=arr[:2], dtype=exp_dtype, name="count")
    tm.assert_series_equal(result, expected)


def test_value_counts_with_normalize(dtype):
    if getattr(dtype, "storage", "") == "pyarrow":
        exp_dtype = "double[pyarrow]"
    elif getattr(dtype, "storage", "") == "pyarrow_numpy":
        exp_dtype = np.float64
    else:
        exp_dtype = "Float64"
    ser = pd.Series(["a", "b", "a", pd.NA], dtype=dtype)
    result = ser.value_counts(normalize=True)
    expected = pd.Series([2, 1], index=ser[:2], dtype=exp_dtype, name="proportion") / 3
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "values, expected",
    [
        (["a", "b", "c"], np.array([False, False, False])),
        (["a", "b", None], np.array([False, False, True])),
    ],
)
def test_use_inf_as_na(values, expected, dtype):
    # https://github.com/pandas-dev/pandas/issues/33655
    values = pd.array(values, dtype=dtype)
    msg = "use_inf_as_na option is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        with pd.option_context("mode.use_inf_as_na", True):
            result = values.isna()
            tm.assert_numpy_array_equal(result, expected)

            result = pd.Series(values).isna()
            expected = pd.Series(expected)
            tm.assert_series_equal(result, expected)

            result = pd.DataFrame(values).isna()
            expected = pd.DataFrame(expected)
            tm.assert_frame_equal(result, expected)


def test_memory_usage(dtype, arrow_string_storage):
    # GH 33963

    if dtype.storage in arrow_string_storage:
        pytest.skip(f"not applicable for {dtype.storage}")

    series = pd.Series(["a", "b", "c"], dtype=dtype)

    assert 0 < series.nbytes <= series.memory_usage() < series.memory_usage(deep=True)


@pytest.mark.parametrize("float_dtype", [np.float16, np.float32, np.float64])
def test_astype_from_float_dtype(float_dtype, dtype):
    # https://github.com/pandas-dev/pandas/issues/36451
    ser = pd.Series([0.1], dtype=float_dtype)
    result = ser.astype(dtype)
    expected = pd.Series(["0.1"], dtype=dtype)
    tm.assert_series_equal(result, expected)


def test_to_numpy_returns_pdna_default(dtype):
    arr = pd.array(["a", pd.NA, "b"], dtype=dtype)
    result = np.array(arr)
    expected = np.array(["a", na_val(dtype), "b"], dtype=object)
    tm.assert_numpy_array_equal(result, expected)


def test_to_numpy_na_value(dtype, nulls_fixture):
    na_value = nulls_fixture
    arr = pd.array(["a", pd.NA, "b"], dtype=dtype)
    result = arr.to_numpy(na_value=na_value)
    expected = np.array(["a", na_value, "b"], dtype=object)
    tm.assert_numpy_array_equal(result, expected)


def test_isin(dtype, fixed_now_ts):
    s = pd.Series(["a", "b", None], dtype=dtype)

    result = s.isin(["a", "c"])
    expected = pd.Series([True, False, False])
    tm.assert_series_equal(result, expected)

    result = s.isin(["a", pd.NA])
    expected = pd.Series([True, False, True])
    tm.assert_series_equal(result, expected)

    result = s.isin([])
    expected = pd.Series([False, False, False])
    tm.assert_series_equal(result, expected)

    result = s.isin(["a", fixed_now_ts])
    expected = pd.Series([True, False, False])
    tm.assert_series_equal(result, expected)


def test_setitem_scalar_with_mask_validation(dtype):
    # https://github.com/pandas-dev/pandas/issues/47628
    # setting None with a boolean mask (through _putmaks) should still result
    # in pd.NA values in the underlying array
    ser = pd.Series(["a", "b", "c"], dtype=dtype)
    mask = np.array([False, True, False])

    ser[mask] = None
    assert ser.array[1] is na_val(ser.dtype)

    # for other non-string we should also raise an error
    ser = pd.Series(["a", "b", "c"], dtype=dtype)
    if type(ser.array) is pd.arrays.StringArray:
        msg = "Cannot set non-string value"
    else:
        msg = "Scalar must be NA or str"
    with pytest.raises(TypeError, match=msg):
        ser[mask] = 1


def test_from_numpy_str(dtype):
    vals = ["a", "b", "c"]
    arr = np.array(vals, dtype=np.str_)
    result = pd.array(arr, dtype=dtype)
    expected = pd.array(vals, dtype=dtype)
    tm.assert_extension_array_equal(result, expected)


def test_tolist(dtype):
    vals = ["a", "b", "c"]
    arr = pd.array(vals, dtype=dtype)
    result = arr.tolist()
    expected = vals
    tm.assert_equal(result, expected)
