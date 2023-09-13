from __future__ import annotations

import collections
from collections import Counter
from datetime import datetime
from decimal import Decimal
import operator
import os
import re
import string
from sys import byteorder
from typing import (
    TYPE_CHECKING,
    Callable,
    ContextManager,
    cast,
)

import numpy as np

from pandas._config.localization import (
    can_set_locale,
    get_locales,
    set_locale,
)

from pandas.compat import pa_version_under7p0

from pandas.core.dtypes.common import (
    is_float_dtype,
    is_sequence,
    is_signed_integer_dtype,
    is_unsigned_integer_dtype,
    pandas_dtype,
)

import pandas as pd
from pandas import (
    ArrowDtype,
    Categorical,
    CategoricalIndex,
    DataFrame,
    DatetimeIndex,
    Index,
    IntervalIndex,
    MultiIndex,
    RangeIndex,
    Series,
    bdate_range,
)
from pandas._testing._io import (
    round_trip_localpath,
    round_trip_pathlib,
    round_trip_pickle,
    write_to_compressed,
)
from pandas._testing._warnings import (
    assert_produces_warning,
    maybe_produces_warning,
)
from pandas._testing.asserters import (
    assert_almost_equal,
    assert_attr_equal,
    assert_categorical_equal,
    assert_class_equal,
    assert_contains_all,
    assert_copy,
    assert_datetime_array_equal,
    assert_dict_equal,
    assert_equal,
    assert_extension_array_equal,
    assert_frame_equal,
    assert_index_equal,
    assert_indexing_slices_equivalent,
    assert_interval_array_equal,
    assert_is_sorted,
    assert_is_valid_plot_return_object,
    assert_metadata_equivalent,
    assert_numpy_array_equal,
    assert_period_array_equal,
    assert_series_equal,
    assert_sp_array_equal,
    assert_timedelta_array_equal,
    raise_assert_detail,
)
from pandas._testing.compat import (
    get_dtype,
    get_obj,
)
from pandas._testing.contexts import (
    decompress_file,
    ensure_clean,
    raises_chained_assignment_error,
    set_timezone,
    use_numexpr,
    with_csv_dialect,
)
from pandas.core.arrays import (
    BaseMaskedArray,
    ExtensionArray,
    NumpyExtensionArray,
)
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas.core.construction import extract_array

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pandas._typing import (
        Dtype,
        Frequency,
        NpDtype,
    )

    from pandas import (
        PeriodIndex,
        TimedeltaIndex,
    )
    from pandas.core.arrays import ArrowExtensionArray

_N = 30
_K = 4

UNSIGNED_INT_NUMPY_DTYPES: list[NpDtype] = ["uint8", "uint16", "uint32", "uint64"]
UNSIGNED_INT_EA_DTYPES: list[Dtype] = ["UInt8", "UInt16", "UInt32", "UInt64"]
SIGNED_INT_NUMPY_DTYPES: list[NpDtype] = [int, "int8", "int16", "int32", "int64"]
SIGNED_INT_EA_DTYPES: list[Dtype] = ["Int8", "Int16", "Int32", "Int64"]
ALL_INT_NUMPY_DTYPES = UNSIGNED_INT_NUMPY_DTYPES + SIGNED_INT_NUMPY_DTYPES
ALL_INT_EA_DTYPES = UNSIGNED_INT_EA_DTYPES + SIGNED_INT_EA_DTYPES
ALL_INT_DTYPES: list[Dtype] = [*ALL_INT_NUMPY_DTYPES, *ALL_INT_EA_DTYPES]

FLOAT_NUMPY_DTYPES: list[NpDtype] = [float, "float32", "float64"]
FLOAT_EA_DTYPES: list[Dtype] = ["Float32", "Float64"]
ALL_FLOAT_DTYPES: list[Dtype] = [*FLOAT_NUMPY_DTYPES, *FLOAT_EA_DTYPES]

COMPLEX_DTYPES: list[Dtype] = [complex, "complex64", "complex128"]
STRING_DTYPES: list[Dtype] = [str, "str", "U"]

DATETIME64_DTYPES: list[Dtype] = ["datetime64[ns]", "M8[ns]"]
TIMEDELTA64_DTYPES: list[Dtype] = ["timedelta64[ns]", "m8[ns]"]

BOOL_DTYPES: list[Dtype] = [bool, "bool"]
BYTES_DTYPES: list[Dtype] = [bytes, "bytes"]
OBJECT_DTYPES: list[Dtype] = [object, "object"]

ALL_REAL_NUMPY_DTYPES = FLOAT_NUMPY_DTYPES + ALL_INT_NUMPY_DTYPES
ALL_REAL_EXTENSION_DTYPES = FLOAT_EA_DTYPES + ALL_INT_EA_DTYPES
ALL_REAL_DTYPES: list[Dtype] = [*ALL_REAL_NUMPY_DTYPES, *ALL_REAL_EXTENSION_DTYPES]
ALL_NUMERIC_DTYPES: list[Dtype] = [*ALL_REAL_DTYPES, *COMPLEX_DTYPES]

ALL_NUMPY_DTYPES = (
    ALL_REAL_NUMPY_DTYPES
    + COMPLEX_DTYPES
    + STRING_DTYPES
    + DATETIME64_DTYPES
    + TIMEDELTA64_DTYPES
    + BOOL_DTYPES
    + OBJECT_DTYPES
    + BYTES_DTYPES
)

NARROW_NP_DTYPES = [
    np.float16,
    np.float32,
    np.int8,
    np.int16,
    np.int32,
    np.uint8,
    np.uint16,
    np.uint32,
]

PYTHON_DATA_TYPES = [
    str,
    int,
    float,
    complex,
    list,
    tuple,
    range,
    dict,
    set,
    frozenset,
    bool,
    bytes,
    bytearray,
    memoryview,
]

ENDIAN = {"little": "<", "big": ">"}[byteorder]

NULL_OBJECTS = [None, np.nan, pd.NaT, float("nan"), pd.NA, Decimal("NaN")]
NP_NAT_OBJECTS = [
    cls("NaT", unit)
    for cls in [np.datetime64, np.timedelta64]
    for unit in [
        "Y",
        "M",
        "W",
        "D",
        "h",
        "m",
        "s",
        "ms",
        "us",
        "ns",
        "ps",
        "fs",
        "as",
    ]
]

if not pa_version_under7p0:
    import pyarrow as pa

    UNSIGNED_INT_PYARROW_DTYPES = [pa.uint8(), pa.uint16(), pa.uint32(), pa.uint64()]
    SIGNED_INT_PYARROW_DTYPES = [pa.int8(), pa.int16(), pa.int32(), pa.int64()]
    ALL_INT_PYARROW_DTYPES = UNSIGNED_INT_PYARROW_DTYPES + SIGNED_INT_PYARROW_DTYPES
    ALL_INT_PYARROW_DTYPES_STR_REPR = [
        str(ArrowDtype(typ)) for typ in ALL_INT_PYARROW_DTYPES
    ]

    # pa.float16 doesn't seem supported
    # https://github.com/apache/arrow/blob/master/python/pyarrow/src/arrow/python/helpers.cc#L86
    FLOAT_PYARROW_DTYPES = [pa.float32(), pa.float64()]
    FLOAT_PYARROW_DTYPES_STR_REPR = [
        str(ArrowDtype(typ)) for typ in FLOAT_PYARROW_DTYPES
    ]
    DECIMAL_PYARROW_DTYPES = [pa.decimal128(7, 3)]
    STRING_PYARROW_DTYPES = [pa.string()]
    BINARY_PYARROW_DTYPES = [pa.binary()]

    TIME_PYARROW_DTYPES = [
        pa.time32("s"),
        pa.time32("ms"),
        pa.time64("us"),
        pa.time64("ns"),
    ]
    DATE_PYARROW_DTYPES = [pa.date32(), pa.date64()]
    DATETIME_PYARROW_DTYPES = [
        pa.timestamp(unit=unit, tz=tz)
        for unit in ["s", "ms", "us", "ns"]
        for tz in [None, "UTC", "US/Pacific", "US/Eastern"]
    ]
    TIMEDELTA_PYARROW_DTYPES = [pa.duration(unit) for unit in ["s", "ms", "us", "ns"]]

    BOOL_PYARROW_DTYPES = [pa.bool_()]

    # TODO: Add container like pyarrow types:
    #  https://arrow.apache.org/docs/python/api/datatypes.html#factory-functions
    ALL_PYARROW_DTYPES = (
        ALL_INT_PYARROW_DTYPES
        + FLOAT_PYARROW_DTYPES
        + DECIMAL_PYARROW_DTYPES
        + STRING_PYARROW_DTYPES
        + BINARY_PYARROW_DTYPES
        + TIME_PYARROW_DTYPES
        + DATE_PYARROW_DTYPES
        + DATETIME_PYARROW_DTYPES
        + TIMEDELTA_PYARROW_DTYPES
        + BOOL_PYARROW_DTYPES
    )
else:
    FLOAT_PYARROW_DTYPES_STR_REPR = []
    ALL_INT_PYARROW_DTYPES_STR_REPR = []
    ALL_PYARROW_DTYPES = []


EMPTY_STRING_PATTERN = re.compile("^$")


arithmetic_dunder_methods = [
    "__add__",
    "__radd__",
    "__sub__",
    "__rsub__",
    "__mul__",
    "__rmul__",
    "__floordiv__",
    "__rfloordiv__",
    "__truediv__",
    "__rtruediv__",
    "__pow__",
    "__rpow__",
    "__mod__",
    "__rmod__",
]

comparison_dunder_methods = ["__eq__", "__ne__", "__le__", "__lt__", "__ge__", "__gt__"]


def reset_display_options() -> None:
    """
    Reset the display options for printing and representing objects.
    """
    pd.reset_option("^display.", silent=True)


# -----------------------------------------------------------------------------
# Comparators


def equalContents(arr1, arr2) -> bool:
    """
    Checks if the set of unique elements of arr1 and arr2 are equivalent.
    """
    return frozenset(arr1) == frozenset(arr2)


def box_expected(expected, box_cls, transpose: bool = True):
    """
    Helper function to wrap the expected output of a test in a given box_class.

    Parameters
    ----------
    expected : np.ndarray, Index, Series
    box_cls : {Index, Series, DataFrame}

    Returns
    -------
    subclass of box_cls
    """
    if box_cls is pd.array:
        if isinstance(expected, RangeIndex):
            # pd.array would return an IntegerArray
            expected = NumpyExtensionArray(np.asarray(expected._values))
        else:
            expected = pd.array(expected, copy=False)
    elif box_cls is Index:
        expected = Index(expected)
    elif box_cls is Series:
        expected = Series(expected)
    elif box_cls is DataFrame:
        expected = Series(expected).to_frame()
        if transpose:
            # for vector operations, we need a DataFrame to be a single-row,
            #  not a single-column, in order to operate against non-DataFrame
            #  vectors of the same length. But convert to two rows to avoid
            #  single-row special cases in datetime arithmetic
            expected = expected.T
            expected = pd.concat([expected] * 2, ignore_index=True)
    elif box_cls is np.ndarray or box_cls is np.array:
        expected = np.array(expected)
    elif box_cls is to_array:
        expected = to_array(expected)
    else:
        raise NotImplementedError(box_cls)
    return expected


def to_array(obj):
    """
    Similar to pd.array, but does not cast numpy dtypes to nullable dtypes.
    """
    # temporary implementation until we get pd.array in place
    dtype = getattr(obj, "dtype", None)

    if dtype is None:
        return np.asarray(obj)

    return extract_array(obj, extract_numpy=True)


# -----------------------------------------------------------------------------
# Others


def rands_array(
    nchars, size: int, dtype: NpDtype = "O", replace: bool = True
) -> np.ndarray:
    """
    Generate an array of byte strings.
    """
    chars = np.array(list(string.ascii_letters + string.digits), dtype=(np.str_, 1))
    retval = (
        np.random.default_rng(2)
        .choice(chars, size=nchars * np.prod(size), replace=replace)
        .view((np.str_, nchars))
        .reshape(size)
    )
    return retval.astype(dtype)


def getCols(k) -> str:
    return string.ascii_uppercase[:k]


# make index
def makeStringIndex(k: int = 10, name=None) -> Index:
    return Index(rands_array(nchars=10, size=k), name=name)


def makeCategoricalIndex(
    k: int = 10, n: int = 3, name=None, **kwargs
) -> CategoricalIndex:
    """make a length k index or n categories"""
    x = rands_array(nchars=4, size=n, replace=False)
    return CategoricalIndex(
        Categorical.from_codes(np.arange(k) % n, categories=x), name=name, **kwargs
    )


def makeIntervalIndex(k: int = 10, name=None, **kwargs) -> IntervalIndex:
    """make a length k IntervalIndex"""
    x = np.linspace(0, 100, num=(k + 1))
    return IntervalIndex.from_breaks(x, name=name, **kwargs)


def makeBoolIndex(k: int = 10, name=None) -> Index:
    if k == 1:
        return Index([True], name=name)
    elif k == 2:
        return Index([False, True], name=name)
    return Index([False, True] + [False] * (k - 2), name=name)


def makeNumericIndex(k: int = 10, *, name=None, dtype: Dtype | None) -> Index:
    dtype = pandas_dtype(dtype)
    assert isinstance(dtype, np.dtype)

    if dtype.kind in "iu":
        values = np.arange(k, dtype=dtype)
        if is_unsigned_integer_dtype(dtype):
            values += 2 ** (dtype.itemsize * 8 - 1)
    elif dtype.kind == "f":
        values = np.random.default_rng(2).random(k) - np.random.default_rng(2).random(1)
        values.sort()
        values = values * (10 ** np.random.default_rng(2).integers(0, 9))
    else:
        raise NotImplementedError(f"wrong dtype {dtype}")

    return Index(values, dtype=dtype, name=name)


def makeIntIndex(k: int = 10, *, name=None, dtype: Dtype = "int64") -> Index:
    dtype = pandas_dtype(dtype)
    if not is_signed_integer_dtype(dtype):
        raise TypeError(f"Wrong dtype {dtype}")
    return makeNumericIndex(k, name=name, dtype=dtype)


def makeUIntIndex(k: int = 10, *, name=None, dtype: Dtype = "uint64") -> Index:
    dtype = pandas_dtype(dtype)
    if not is_unsigned_integer_dtype(dtype):
        raise TypeError(f"Wrong dtype {dtype}")
    return makeNumericIndex(k, name=name, dtype=dtype)


def makeRangeIndex(k: int = 10, name=None, **kwargs) -> RangeIndex:
    return RangeIndex(0, k, 1, name=name, **kwargs)


def makeFloatIndex(k: int = 10, *, name=None, dtype: Dtype = "float64") -> Index:
    dtype = pandas_dtype(dtype)
    if not is_float_dtype(dtype):
        raise TypeError(f"Wrong dtype {dtype}")
    return makeNumericIndex(k, name=name, dtype=dtype)


def makeDateIndex(
    k: int = 10, freq: Frequency = "B", name=None, **kwargs
) -> DatetimeIndex:
    dt = datetime(2000, 1, 1)
    dr = bdate_range(dt, periods=k, freq=freq, name=name)
    return DatetimeIndex(dr, name=name, **kwargs)


def makeTimedeltaIndex(
    k: int = 10, freq: Frequency = "D", name=None, **kwargs
) -> TimedeltaIndex:
    return pd.timedelta_range(start="1 day", periods=k, freq=freq, name=name, **kwargs)


def makePeriodIndex(k: int = 10, name=None, **kwargs) -> PeriodIndex:
    dt = datetime(2000, 1, 1)
    pi = pd.period_range(start=dt, periods=k, freq="D", name=name, **kwargs)
    return pi


def makeMultiIndex(k: int = 10, names=None, **kwargs):
    N = (k // 2) + 1
    rng = range(N)
    mi = MultiIndex.from_product([("foo", "bar"), rng], names=names, **kwargs)
    assert len(mi) >= k  # GH#38795
    return mi[:k]


def index_subclass_makers_generator():
    make_index_funcs = [
        makeDateIndex,
        makePeriodIndex,
        makeTimedeltaIndex,
        makeRangeIndex,
        makeIntervalIndex,
        makeCategoricalIndex,
        makeMultiIndex,
    ]
    yield from make_index_funcs


def all_timeseries_index_generator(k: int = 10) -> Iterable[Index]:
    """
    Generator which can be iterated over to get instances of all the classes
    which represent time-series.

    Parameters
    ----------
    k: length of each of the index instances
    """
    make_index_funcs: list[Callable[..., Index]] = [
        makeDateIndex,
        makePeriodIndex,
        makeTimedeltaIndex,
    ]
    for make_index_func in make_index_funcs:
        yield make_index_func(k=k)


# make series
def make_rand_series(name=None, dtype=np.float64) -> Series:
    index = makeStringIndex(_N)
    data = np.random.default_rng(2).standard_normal(_N)
    with np.errstate(invalid="ignore"):
        data = data.astype(dtype, copy=False)
    return Series(data, index=index, name=name)


def makeFloatSeries(name=None) -> Series:
    return make_rand_series(name=name)


def makeStringSeries(name=None) -> Series:
    return make_rand_series(name=name)


def makeObjectSeries(name=None) -> Series:
    data = makeStringIndex(_N)
    data = Index(data, dtype=object)
    index = makeStringIndex(_N)
    return Series(data, index=index, name=name)


def getSeriesData() -> dict[str, Series]:
    index = makeStringIndex(_N)
    return {
        c: Series(np.random.default_rng(i).standard_normal(_N), index=index)
        for i, c in enumerate(getCols(_K))
    }


def makeTimeSeries(nper=None, freq: Frequency = "B", name=None) -> Series:
    if nper is None:
        nper = _N
    return Series(
        np.random.default_rng(2).standard_normal(nper),
        index=makeDateIndex(nper, freq=freq),
        name=name,
    )


def makePeriodSeries(nper=None, name=None) -> Series:
    if nper is None:
        nper = _N
    return Series(
        np.random.default_rng(2).standard_normal(nper),
        index=makePeriodIndex(nper),
        name=name,
    )


def getTimeSeriesData(nper=None, freq: Frequency = "B") -> dict[str, Series]:
    return {c: makeTimeSeries(nper, freq) for c in getCols(_K)}


def getPeriodData(nper=None) -> dict[str, Series]:
    return {c: makePeriodSeries(nper) for c in getCols(_K)}


# make frame
def makeTimeDataFrame(nper=None, freq: Frequency = "B") -> DataFrame:
    data = getTimeSeriesData(nper, freq)
    return DataFrame(data)


def makeDataFrame() -> DataFrame:
    data = getSeriesData()
    return DataFrame(data)


def getMixedTypeDict():
    index = Index(["a", "b", "c", "d", "e"])

    data = {
        "A": [0.0, 1.0, 2.0, 3.0, 4.0],
        "B": [0.0, 1.0, 0.0, 1.0, 0.0],
        "C": ["foo1", "foo2", "foo3", "foo4", "foo5"],
        "D": bdate_range("1/1/2009", periods=5),
    }

    return index, data


def makeMixedDataFrame() -> DataFrame:
    return DataFrame(getMixedTypeDict()[1])


def makePeriodFrame(nper=None) -> DataFrame:
    data = getPeriodData(nper)
    return DataFrame(data)


def makeCustomIndex(
    nentries,
    nlevels,
    prefix: str = "#",
    names: bool | str | list[str] | None = False,
    ndupe_l=None,
    idx_type=None,
) -> Index:
    """
    Create an index/multindex with given dimensions, levels, names, etc'

    nentries - number of entries in index
    nlevels - number of levels (> 1 produces multindex)
    prefix - a string prefix for labels
    names - (Optional), bool or list of strings. if True will use default
       names, if false will use no names, if a list is given, the name of
       each level in the index will be taken from the list.
    ndupe_l - (Optional), list of ints, the number of rows for which the
       label will repeated at the corresponding level, you can specify just
       the first few, the rest will use the default ndupe_l of 1.
       len(ndupe_l) <= nlevels.
    idx_type - "i"/"f"/"s"/"dt"/"p"/"td".
       If idx_type is not None, `idx_nlevels` must be 1.
       "i"/"f" creates an integer/float index,
       "s" creates a string
       "dt" create a datetime index.
       "td" create a datetime index.

        if unspecified, string labels will be generated.
    """
    if ndupe_l is None:
        ndupe_l = [1] * nlevels
    assert is_sequence(ndupe_l) and len(ndupe_l) <= nlevels
    assert names is None or names is False or names is True or len(names) is nlevels
    assert idx_type is None or (
        idx_type in ("i", "f", "s", "u", "dt", "p", "td") and nlevels == 1
    )

    if names is True:
        # build default names
        names = [prefix + str(i) for i in range(nlevels)]
    if names is False:
        # pass None to index constructor for no name
        names = None

    # make singleton case uniform
    if isinstance(names, str) and nlevels == 1:
        names = [names]

    # specific 1D index type requested?
    idx_func_dict: dict[str, Callable[..., Index]] = {
        "i": makeIntIndex,
        "f": makeFloatIndex,
        "s": makeStringIndex,
        "dt": makeDateIndex,
        "td": makeTimedeltaIndex,
        "p": makePeriodIndex,
    }
    idx_func = idx_func_dict.get(idx_type)
    if idx_func:
        idx = idx_func(nentries)
        # but we need to fill in the name
        if names:
            idx.name = names[0]
        return idx
    elif idx_type is not None:
        raise ValueError(
            f"{repr(idx_type)} is not a legal value for `idx_type`, "
            "use  'i'/'f'/'s'/'dt'/'p'/'td'."
        )

    if len(ndupe_l) < nlevels:
        ndupe_l.extend([1] * (nlevels - len(ndupe_l)))
    assert len(ndupe_l) == nlevels

    assert all(x > 0 for x in ndupe_l)

    list_of_lists = []
    for i in range(nlevels):

        def keyfunc(x):
            numeric_tuple = re.sub(r"[^\d_]_?", "", x).split("_")
            return [int(num) for num in numeric_tuple]

        # build a list of lists to create the index from
        div_factor = nentries // ndupe_l[i] + 1

        # Deprecated since version 3.9: collections.Counter now supports []. See PEP 585
        # and Generic Alias Type.
        cnt: Counter[str] = collections.Counter()
        for j in range(div_factor):
            label = f"{prefix}_l{i}_g{j}"
            cnt[label] = ndupe_l[i]
        # cute Counter trick
        result = sorted(cnt.elements(), key=keyfunc)[:nentries]
        list_of_lists.append(result)

    tuples = list(zip(*list_of_lists))

    # convert tuples to index
    if nentries == 1:
        # we have a single level of tuples, i.e. a regular Index
        name = None if names is None else names[0]
        index = Index(tuples[0], name=name)
    elif nlevels == 1:
        name = None if names is None else names[0]
        index = Index((x[0] for x in tuples), name=name)
    else:
        index = MultiIndex.from_tuples(tuples, names=names)
    return index


def makeCustomDataframe(
    nrows,
    ncols,
    c_idx_names: bool | list[str] = True,
    r_idx_names: bool | list[str] = True,
    c_idx_nlevels: int = 1,
    r_idx_nlevels: int = 1,
    data_gen_f=None,
    c_ndupe_l=None,
    r_ndupe_l=None,
    dtype=None,
    c_idx_type=None,
    r_idx_type=None,
) -> DataFrame:
    """
    Create a DataFrame using supplied parameters.

    Parameters
    ----------
    nrows,  ncols - number of data rows/cols
    c_idx_names, r_idx_names  - False/True/list of strings,  yields No names ,
            default names or uses the provided names for the levels of the
            corresponding index. You can provide a single string when
            c_idx_nlevels ==1.
    c_idx_nlevels - number of levels in columns index. > 1 will yield MultiIndex
    r_idx_nlevels - number of levels in rows index. > 1 will yield MultiIndex
    data_gen_f - a function f(row,col) which return the data value
            at that position, the default generator used yields values of the form
            "RxCy" based on position.
    c_ndupe_l, r_ndupe_l - list of integers, determines the number
            of duplicates for each label at a given level of the corresponding
            index. The default `None` value produces a multiplicity of 1 across
            all levels, i.e. a unique index. Will accept a partial list of length
            N < idx_nlevels, for just the first N levels. If ndupe doesn't divide
            nrows/ncol, the last label might have lower multiplicity.
    dtype - passed to the DataFrame constructor as is, in case you wish to
            have more control in conjunction with a custom `data_gen_f`
    r_idx_type, c_idx_type -  "i"/"f"/"s"/"dt"/"td".
        If idx_type is not None, `idx_nlevels` must be 1.
        "i"/"f" creates an integer/float index,
        "s" creates a string index
        "dt" create a datetime index.
        "td" create a timedelta index.

            if unspecified, string labels will be generated.

    Examples
    --------
    # 5 row, 3 columns, default names on both, single index on both axis
    >> makeCustomDataframe(5,3)

    # make the data a random int between 1 and 100
    >> mkdf(5,3,data_gen_f=lambda r,c:randint(1,100))

    # 2-level multiindex on rows with each label duplicated
    # twice on first level, default names on both axis, single
    # index on both axis
    >> a=makeCustomDataframe(5,3,r_idx_nlevels=2,r_ndupe_l=[2])

    # DatetimeIndex on row, index with unicode labels on columns
    # no names on either axis
    >> a=makeCustomDataframe(5,3,c_idx_names=False,r_idx_names=False,
                             r_idx_type="dt",c_idx_type="u")

    # 4-level multindex on rows with names provided, 2-level multindex
    # on columns with default labels and default names.
    >> a=makeCustomDataframe(5,3,r_idx_nlevels=4,
                             r_idx_names=["FEE","FIH","FOH","FUM"],
                             c_idx_nlevels=2)

    >> a=mkdf(5,3,r_idx_nlevels=2,c_idx_nlevels=4)
    """
    assert c_idx_nlevels > 0
    assert r_idx_nlevels > 0
    assert r_idx_type is None or (
        r_idx_type in ("i", "f", "s", "dt", "p", "td") and r_idx_nlevels == 1
    )
    assert c_idx_type is None or (
        c_idx_type in ("i", "f", "s", "dt", "p", "td") and c_idx_nlevels == 1
    )

    columns = makeCustomIndex(
        ncols,
        nlevels=c_idx_nlevels,
        prefix="C",
        names=c_idx_names,
        ndupe_l=c_ndupe_l,
        idx_type=c_idx_type,
    )
    index = makeCustomIndex(
        nrows,
        nlevels=r_idx_nlevels,
        prefix="R",
        names=r_idx_names,
        ndupe_l=r_ndupe_l,
        idx_type=r_idx_type,
    )

    # by default, generate data based on location
    if data_gen_f is None:
        data_gen_f = lambda r, c: f"R{r}C{c}"

    data = [[data_gen_f(r, c) for c in range(ncols)] for r in range(nrows)]

    return DataFrame(data, index, columns, dtype=dtype)


class SubclassedSeries(Series):
    _metadata = ["testattr", "name"]

    @property
    def _constructor(self):
        # For testing, those properties return a generic callable, and not
        # the actual class. In this case that is equivalent, but it is to
        # ensure we don't rely on the property returning a class
        # See https://github.com/pandas-dev/pandas/pull/46018 and
        # https://github.com/pandas-dev/pandas/issues/32638 and linked issues
        return lambda *args, **kwargs: SubclassedSeries(*args, **kwargs)

    @property
    def _constructor_expanddim(self):
        return lambda *args, **kwargs: SubclassedDataFrame(*args, **kwargs)


class SubclassedDataFrame(DataFrame):
    _metadata = ["testattr"]

    @property
    def _constructor(self):
        return lambda *args, **kwargs: SubclassedDataFrame(*args, **kwargs)

    @property
    def _constructor_sliced(self):
        return lambda *args, **kwargs: SubclassedSeries(*args, **kwargs)


class SubclassedCategorical(Categorical):
    pass


def _make_skipna_wrapper(alternative, skipna_alternative=None):
    """
    Create a function for calling on an array.

    Parameters
    ----------
    alternative : function
        The function to be called on the array with no NaNs.
        Only used when 'skipna_alternative' is None.
    skipna_alternative : function
        The function to be called on the original array

    Returns
    -------
    function
    """
    if skipna_alternative:

        def skipna_wrapper(x):
            return skipna_alternative(x.values)

    else:

        def skipna_wrapper(x):
            nona = x.dropna()
            if len(nona) == 0:
                return np.nan
            return alternative(nona)

    return skipna_wrapper


def convert_rows_list_to_csv_str(rows_list: list[str]) -> str:
    """
    Convert list of CSV rows to single CSV-formatted string for current OS.

    This method is used for creating expected value of to_csv() method.

    Parameters
    ----------
    rows_list : List[str]
        Each element represents the row of csv.

    Returns
    -------
    str
        Expected output of to_csv() in current OS.
    """
    sep = os.linesep
    return sep.join(rows_list) + sep


def external_error_raised(expected_exception: type[Exception]) -> ContextManager:
    """
    Helper function to mark pytest.raises that have an external error message.

    Parameters
    ----------
    expected_exception : Exception
        Expected error to raise.

    Returns
    -------
    Callable
        Regular `pytest.raises` function with `match` equal to `None`.
    """
    import pytest

    return pytest.raises(expected_exception, match=None)


cython_table = pd.core.common._cython_table.items()


def get_cython_table_params(ndframe, func_names_and_expected):
    """
    Combine frame, functions from com._cython_table
    keys and expected result.

    Parameters
    ----------
    ndframe : DataFrame or Series
    func_names_and_expected : Sequence of two items
        The first item is a name of a NDFrame method ('sum', 'prod') etc.
        The second item is the expected return value.

    Returns
    -------
    list
        List of three items (DataFrame, function, expected result)
    """
    results = []
    for func_name, expected in func_names_and_expected:
        results.append((ndframe, func_name, expected))
        results += [
            (ndframe, func, expected)
            for func, name in cython_table
            if name == func_name
        ]
    return results


def get_op_from_name(op_name: str) -> Callable:
    """
    The operator function for a given op name.

    Parameters
    ----------
    op_name : str
        The op name, in form of "add" or "__add__".

    Returns
    -------
    function
        A function performing the operation.
    """
    short_opname = op_name.strip("_")
    try:
        op = getattr(operator, short_opname)
    except AttributeError:
        # Assume it is the reverse operator
        rop = getattr(operator, short_opname[1:])
        op = lambda x, y: rop(y, x)

    return op


# -----------------------------------------------------------------------------
# Indexing test helpers


def getitem(x):
    return x


def setitem(x):
    return x


def loc(x):
    return x.loc


def iloc(x):
    return x.iloc


def at(x):
    return x.at


def iat(x):
    return x.iat


# -----------------------------------------------------------------------------


def shares_memory(left, right) -> bool:
    """
    Pandas-compat for np.shares_memory.
    """
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return np.shares_memory(left, right)
    elif isinstance(left, np.ndarray):
        # Call with reversed args to get to unpacking logic below.
        return shares_memory(right, left)

    if isinstance(left, RangeIndex):
        return False
    if isinstance(left, MultiIndex):
        return shares_memory(left._codes, right)
    if isinstance(left, (Index, Series)):
        return shares_memory(left._values, right)

    if isinstance(left, NDArrayBackedExtensionArray):
        return shares_memory(left._ndarray, right)
    if isinstance(left, pd.core.arrays.SparseArray):
        return shares_memory(left.sp_values, right)
    if isinstance(left, pd.core.arrays.IntervalArray):
        return shares_memory(left._left, right) or shares_memory(left._right, right)

    if isinstance(left, ExtensionArray) and left.dtype == "string[pyarrow]":
        # https://github.com/pandas-dev/pandas/pull/43930#discussion_r736862669
        left = cast("ArrowExtensionArray", left)
        if isinstance(right, ExtensionArray) and right.dtype == "string[pyarrow]":
            right = cast("ArrowExtensionArray", right)
            left_pa_data = left._pa_array
            right_pa_data = right._pa_array
            left_buf1 = left_pa_data.chunk(0).buffers()[1]
            right_buf1 = right_pa_data.chunk(0).buffers()[1]
            return left_buf1 == right_buf1

    if isinstance(left, BaseMaskedArray) and isinstance(right, BaseMaskedArray):
        # By convention, we'll say these share memory if they share *either*
        #  the _data or the _mask
        return np.shares_memory(left._data, right._data) or np.shares_memory(
            left._mask, right._mask
        )

    if isinstance(left, DataFrame) and len(left._mgr.arrays) == 1:
        arr = left._mgr.arrays[0]
        return shares_memory(arr, right)

    raise NotImplementedError(type(left), type(right))


__all__ = [
    "ALL_INT_EA_DTYPES",
    "ALL_INT_NUMPY_DTYPES",
    "ALL_NUMPY_DTYPES",
    "ALL_REAL_NUMPY_DTYPES",
    "all_timeseries_index_generator",
    "assert_almost_equal",
    "assert_attr_equal",
    "assert_categorical_equal",
    "assert_class_equal",
    "assert_contains_all",
    "assert_copy",
    "assert_datetime_array_equal",
    "assert_dict_equal",
    "assert_equal",
    "assert_extension_array_equal",
    "assert_frame_equal",
    "assert_index_equal",
    "assert_indexing_slices_equivalent",
    "assert_interval_array_equal",
    "assert_is_sorted",
    "assert_is_valid_plot_return_object",
    "assert_metadata_equivalent",
    "assert_numpy_array_equal",
    "assert_period_array_equal",
    "assert_produces_warning",
    "assert_series_equal",
    "assert_sp_array_equal",
    "assert_timedelta_array_equal",
    "at",
    "BOOL_DTYPES",
    "box_expected",
    "BYTES_DTYPES",
    "can_set_locale",
    "COMPLEX_DTYPES",
    "convert_rows_list_to_csv_str",
    "DATETIME64_DTYPES",
    "decompress_file",
    "EMPTY_STRING_PATTERN",
    "ENDIAN",
    "ensure_clean",
    "equalContents",
    "external_error_raised",
    "FLOAT_EA_DTYPES",
    "FLOAT_NUMPY_DTYPES",
    "getCols",
    "get_cython_table_params",
    "get_dtype",
    "getitem",
    "get_locales",
    "getMixedTypeDict",
    "get_obj",
    "get_op_from_name",
    "getPeriodData",
    "getSeriesData",
    "getTimeSeriesData",
    "iat",
    "iloc",
    "index_subclass_makers_generator",
    "loc",
    "makeBoolIndex",
    "makeCategoricalIndex",
    "makeCustomDataframe",
    "makeCustomIndex",
    "makeDataFrame",
    "makeDateIndex",
    "makeFloatIndex",
    "makeFloatSeries",
    "makeIntervalIndex",
    "makeIntIndex",
    "makeMixedDataFrame",
    "makeMultiIndex",
    "makeNumericIndex",
    "makeObjectSeries",
    "makePeriodFrame",
    "makePeriodIndex",
    "makePeriodSeries",
    "make_rand_series",
    "makeRangeIndex",
    "makeStringIndex",
    "makeStringSeries",
    "makeTimeDataFrame",
    "makeTimedeltaIndex",
    "makeTimeSeries",
    "makeUIntIndex",
    "maybe_produces_warning",
    "NARROW_NP_DTYPES",
    "NP_NAT_OBJECTS",
    "NULL_OBJECTS",
    "OBJECT_DTYPES",
    "raise_assert_detail",
    "reset_display_options",
    "raises_chained_assignment_error",
    "round_trip_localpath",
    "round_trip_pathlib",
    "round_trip_pickle",
    "setitem",
    "set_locale",
    "set_timezone",
    "shares_memory",
    "SIGNED_INT_EA_DTYPES",
    "SIGNED_INT_NUMPY_DTYPES",
    "STRING_DTYPES",
    "SubclassedCategorical",
    "SubclassedDataFrame",
    "SubclassedSeries",
    "TIMEDELTA64_DTYPES",
    "to_array",
    "UNSIGNED_INT_EA_DTYPES",
    "UNSIGNED_INT_NUMPY_DTYPES",
    "use_numexpr",
    "with_csv_dialect",
    "write_to_compressed",
]
