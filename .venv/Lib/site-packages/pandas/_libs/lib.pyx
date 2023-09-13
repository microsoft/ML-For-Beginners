from collections import abc
from decimal import Decimal
from enum import Enum
from sys import getsizeof
from typing import (
    Literal,
    _GenericAlias,
)

cimport cython
from cpython.datetime cimport (
    PyDate_Check,
    PyDateTime_Check,
    PyDelta_Check,
    PyTime_Check,
    date,
    datetime,
    import_datetime,
    time,
    timedelta,
)
from cpython.iterator cimport PyIter_Check
from cpython.number cimport PyNumber_Check
from cpython.object cimport (
    Py_EQ,
    PyObject,
    PyObject_RichCompareBool,
    PyTypeObject,
)
from cpython.ref cimport Py_INCREF
from cpython.sequence cimport PySequence_Check
from cpython.tuple cimport (
    PyTuple_New,
    PyTuple_SET_ITEM,
)
from cython cimport (
    Py_ssize_t,
    floating,
)

from pandas._config import using_pyarrow_string_dtype

from pandas._libs.missing import check_na_tuples_nonequal

import_datetime()

import numpy as np

cimport numpy as cnp
from numpy cimport (
    NPY_OBJECT,
    PyArray_Check,
    PyArray_GETITEM,
    PyArray_ITER_DATA,
    PyArray_ITER_NEXT,
    PyArray_IterNew,
    complex128_t,
    flatiter,
    float64_t,
    int32_t,
    int64_t,
    intp_t,
    ndarray,
    uint8_t,
    uint64_t,
)

cnp.import_array()

cdef extern from "Python.h":
    # Note: importing extern-style allows us to declare these as nogil
    # functions, whereas `from cpython cimport` does not.
    bint PyObject_TypeCheck(object obj, PyTypeObject* type) nogil

cdef extern from "numpy/arrayobject.h":
    # cython's numpy.dtype specification is incorrect, which leads to
    # errors in issubclass(self.dtype.type, np.bool_), so we directly
    # include the correct version
    # https://github.com/cython/cython/issues/2022

    ctypedef class numpy.dtype [object PyArray_Descr]:
        # Use PyDataType_* macros when possible, however there are no macros
        # for accessing some of the fields, so some are defined. Please
        # ask on cython-dev if you need more.
        cdef:
            int type_num
            int itemsize "elsize"
            char byteorder
            object fields
            tuple names

    PyTypeObject PySignedIntegerArrType_Type
    PyTypeObject PyUnsignedIntegerArrType_Type

cdef extern from "numpy/ndarrayobject.h":
    bint PyArray_CheckScalar(obj) nogil

cdef extern from "pandas/parser/pd_parser.h":
    int floatify(object, float64_t *result, int *maybe_int) except -1
    void PandasParser_IMPORT()

PandasParser_IMPORT

from pandas._libs cimport util
from pandas._libs.util cimport (
    INT64_MAX,
    INT64_MIN,
    UINT64_MAX,
    is_nan,
)

from pandas._libs.tslibs import (
    OutOfBoundsDatetime,
    OutOfBoundsTimedelta,
)
from pandas._libs.tslibs.period import Period

from pandas._libs.missing cimport (
    C_NA,
    checknull,
    is_matching_na,
    is_null_datetime64,
    is_null_timedelta64,
)
from pandas._libs.tslibs.conversion cimport (
    _TSObject,
    convert_to_tsobject,
)
from pandas._libs.tslibs.nattype cimport (
    NPY_NAT,
    c_NaT as NaT,
    checknull_with_nat,
)
from pandas._libs.tslibs.np_datetime cimport NPY_FR_ns
from pandas._libs.tslibs.offsets cimport is_offset_object
from pandas._libs.tslibs.period cimport is_period_object
from pandas._libs.tslibs.timedeltas cimport convert_to_timedelta64
from pandas._libs.tslibs.timezones cimport tz_compare

# constants that will be compared to potentially arbitrarily large
# python int
cdef:
    object oINT64_MAX = <int64_t>INT64_MAX
    object oINT64_MIN = <int64_t>INT64_MIN
    object oUINT64_MAX = <uint64_t>UINT64_MAX

    float64_t NaN = <float64_t>np.nan

# python-visible
i8max = <int64_t>INT64_MAX
u8max = <uint64_t>UINT64_MAX


cdef bint PYARROW_INSTALLED = False

try:
    import pyarrow as pa

    PYARROW_INSTALLED = True
except ImportError:
    pa = None


@cython.wraparound(False)
@cython.boundscheck(False)
def memory_usage_of_objects(arr: object[:]) -> int64_t:
    """
    Return the memory usage of an object array in bytes.

    Does not include the actual bytes of the pointers
    """
    cdef:
        Py_ssize_t i
        Py_ssize_t n
        int64_t size = 0

    n = len(arr)
    for i in range(n):
        size += getsizeof(arr[i])
    return size


# ----------------------------------------------------------------------


def is_scalar(val: object) -> bool:
    """
    Return True if given object is scalar.

    Parameters
    ----------
    val : object
        This includes:

        - numpy array scalar (e.g. np.int64)
        - Python builtin numerics
        - Python builtin byte arrays and strings
        - None
        - datetime.datetime
        - datetime.timedelta
        - Period
        - decimal.Decimal
        - Interval
        - DateOffset
        - Fraction
        - Number.

    Returns
    -------
    bool
        Return True if given object is scalar.

    Examples
    --------
    >>> import datetime
    >>> dt = datetime.datetime(2018, 10, 3)
    >>> pd.api.types.is_scalar(dt)
    True

    >>> pd.api.types.is_scalar([2, 3])
    False

    >>> pd.api.types.is_scalar({0: 1, 2: 3})
    False

    >>> pd.api.types.is_scalar((0, 2))
    False

    pandas supports PEP 3141 numbers:

    >>> from fractions import Fraction
    >>> pd.api.types.is_scalar(Fraction(3, 5))
    True
    """

    # Start with C-optimized checks
    if (cnp.PyArray_IsAnyScalar(val)
            # PyArray_IsAnyScalar is always False for bytearrays on Py3
            or PyDate_Check(val)
            or PyDelta_Check(val)
            or PyTime_Check(val)
            # We differ from numpy, which claims that None is not scalar;
            # see np.isscalar
            or val is C_NA
            or val is None):
        return True

    # Next use C-optimized checks to exclude common non-scalars before falling
    #  back to non-optimized checks.
    if PySequence_Check(val):
        # e.g. list, tuple
        # includes np.ndarray, Series which PyNumber_Check can return True for
        return False

    # Note: PyNumber_Check check includes Decimal, Fraction, numbers.Number
    return (PyNumber_Check(val)
            or is_period_object(val)
            or is_interval(val)
            or is_offset_object(val))


cdef int64_t get_itemsize(object val):
    """
    Get the itemsize of a NumPy scalar, -1 if not a NumPy scalar.

    Parameters
    ----------
    val : object

    Returns
    -------
    is_ndarray : bool
    """
    if PyArray_CheckScalar(val):
        return cnp.PyArray_DescrFromScalar(val).itemsize
    else:
        return -1


def is_iterator(obj: object) -> bool:
    """
    Check if the object is an iterator.

    This is intended for generators, not list-like objects.

    Parameters
    ----------
    obj : The object to check

    Returns
    -------
    is_iter : bool
        Whether `obj` is an iterator.

    Examples
    --------
    >>> import datetime
    >>> from pandas.api.types import is_iterator
    >>> is_iterator((x for x in []))
    True
    >>> is_iterator([1, 2, 3])
    False
    >>> is_iterator(datetime.datetime(2017, 1, 1))
    False
    >>> is_iterator("foo")
    False
    >>> is_iterator(1)
    False
    """
    return PyIter_Check(obj)


def item_from_zerodim(val: object) -> object:
    """
    If the value is a zerodim array, return the item it contains.

    Parameters
    ----------
    val : object

    Returns
    -------
    object

    Examples
    --------
    >>> item_from_zerodim(1)
    1
    >>> item_from_zerodim('foobar')
    'foobar'
    >>> item_from_zerodim(np.array(1))
    1
    >>> item_from_zerodim(np.array([1]))
    array([1])
    """
    if cnp.PyArray_IsZeroDim(val):
        return cnp.PyArray_ToScalar(cnp.PyArray_DATA(val), val)
    return val


@cython.wraparound(False)
@cython.boundscheck(False)
def fast_unique_multiple_list(lists: list, sort: bool | None = True) -> list:
    cdef:
        list buf
        Py_ssize_t k = len(lists)
        Py_ssize_t i, j, n
        list uniques = []
        dict table = {}
        object val, stub = 0

    for i in range(k):
        buf = lists[i]
        n = len(buf)
        for j in range(n):
            val = buf[j]
            if val not in table:
                table[val] = stub
                uniques.append(val)
    if sort:
        try:
            uniques.sort()
        except TypeError:
            pass

    return uniques


@cython.wraparound(False)
@cython.boundscheck(False)
def fast_unique_multiple_list_gen(object gen, bint sort=True) -> list:
    """
    Generate a list of unique values from a generator of lists.

    Parameters
    ----------
    gen : generator object
        Generator of lists from which the unique list is created.
    sort : bool
        Whether or not to sort the resulting unique list.

    Returns
    -------
    list of unique values
    """
    cdef:
        list buf
        Py_ssize_t j, n
        list uniques = []
        dict table = {}
        object val, stub = 0

    for buf in gen:
        n = len(buf)
        for j in range(n):
            val = buf[j]
            if val not in table:
                table[val] = stub
                uniques.append(val)
    if sort:
        try:
            uniques.sort()
        except TypeError:
            pass

    return uniques


@cython.wraparound(False)
@cython.boundscheck(False)
def dicts_to_array(dicts: list, columns: list):
    cdef:
        Py_ssize_t i, j, k, n
        ndarray[object, ndim=2] result
        dict row
        object col, onan = np.nan

    k = len(columns)
    n = len(dicts)

    result = np.empty((n, k), dtype="O")

    for i in range(n):
        row = dicts[i]
        for j in range(k):
            col = columns[j]
            if col in row:
                result[i, j] = row[col]
            else:
                result[i, j] = onan

    return result


def fast_zip(list ndarrays) -> ndarray[object]:
    """
    For zipping multiple ndarrays into an ndarray of tuples.
    """
    cdef:
        Py_ssize_t i, j, k, n
        ndarray[object, ndim=1] result
        flatiter it
        object val, tup

    k = len(ndarrays)
    n = len(ndarrays[0])

    result = np.empty(n, dtype=object)

    # initialize tuples on first pass
    arr = ndarrays[0]
    it = <flatiter>PyArray_IterNew(arr)
    for i in range(n):
        val = PyArray_GETITEM(arr, PyArray_ITER_DATA(it))
        tup = PyTuple_New(k)

        PyTuple_SET_ITEM(tup, 0, val)
        Py_INCREF(val)
        result[i] = tup
        PyArray_ITER_NEXT(it)

    for j in range(1, k):
        arr = ndarrays[j]
        it = <flatiter>PyArray_IterNew(arr)
        if len(arr) != n:
            raise ValueError("all arrays must be same length")

        for i in range(n):
            val = PyArray_GETITEM(arr, PyArray_ITER_DATA(it))
            PyTuple_SET_ITEM(result[i], j, val)
            Py_INCREF(val)
            PyArray_ITER_NEXT(it)

    return result


def get_reverse_indexer(const intp_t[:] indexer, Py_ssize_t length) -> ndarray:
    """
    Reverse indexing operation.

    Given `indexer`, make `indexer_inv` of it, such that::

        indexer_inv[indexer[x]] = x

    Parameters
    ----------
    indexer : np.ndarray[np.intp]
    length : int

    Returns
    -------
    np.ndarray[np.intp]

    Notes
    -----
    If indexer is not unique, only first occurrence is accounted.
    """
    cdef:
        Py_ssize_t i, n = len(indexer)
        ndarray[intp_t, ndim=1] rev_indexer
        intp_t idx

    rev_indexer = np.empty(length, dtype=np.intp)
    rev_indexer[:] = -1
    for i in range(n):
        idx = indexer[i]
        if idx != -1:
            rev_indexer[idx] = i

    return rev_indexer


@cython.wraparound(False)
@cython.boundscheck(False)
# TODO(cython3): Can add const once cython#1772 is resolved
def has_infs(floating[:] arr) -> bool:
    cdef:
        Py_ssize_t i, n = len(arr)
        floating inf, neginf, val
        bint ret = False

    inf = np.inf
    neginf = -inf
    with nogil:
        for i in range(n):
            val = arr[i]
            if val == inf or val == neginf:
                ret = True
                break
    return ret


@cython.boundscheck(False)
@cython.wraparound(False)
def has_only_ints_or_nan(floating[:] arr) -> bool:
    cdef:
        floating val
        intp_t i

    for i in range(len(arr)):
        val = arr[i]
        if (val != val) or (val == <int64_t>val):
            continue
        else:
            return False
    return True


def maybe_indices_to_slice(ndarray[intp_t, ndim=1] indices, int max_len):
    cdef:
        Py_ssize_t i, n = len(indices)
        intp_t k, vstart, vlast, v

    if n == 0:
        return slice(0, 0)

    vstart = indices[0]
    if vstart < 0 or max_len <= vstart:
        return indices

    if n == 1:
        return slice(vstart, <intp_t>(vstart + 1))

    vlast = indices[n - 1]
    if vlast < 0 or max_len <= vlast:
        return indices

    k = indices[1] - indices[0]
    if k == 0:
        return indices
    else:
        for i in range(2, n):
            v = indices[i]
            if v - indices[i - 1] != k:
                return indices

        if k > 0:
            return slice(vstart, <intp_t>(vlast + 1), k)
        else:
            if vlast == 0:
                return slice(vstart, None, k)
            else:
                return slice(vstart, <intp_t>(vlast - 1), k)


@cython.wraparound(False)
@cython.boundscheck(False)
def maybe_booleans_to_slice(ndarray[uint8_t, ndim=1] mask):
    cdef:
        Py_ssize_t i, n = len(mask)
        Py_ssize_t start = 0, end = 0
        bint started = False, finished = False

    for i in range(n):
        if mask[i]:
            if finished:
                return mask.view(np.bool_)
            if not started:
                started = True
                start = i
        else:
            if finished:
                continue

            if started:
                end = i
                finished = True

    if not started:
        return slice(0, 0)
    if not finished:
        return slice(start, None)
    else:
        return slice(start, end)


@cython.wraparound(False)
@cython.boundscheck(False)
def array_equivalent_object(ndarray left, ndarray right) -> bool:
    """
    Perform an element by element comparison on N-d object arrays
    taking into account nan positions.
    """
    # left and right both have object dtype, but we cannot annotate that
    #  without limiting ndim.
    cdef:
        Py_ssize_t i, n = left.size
        object x, y
        cnp.broadcast mi = cnp.PyArray_MultiIterNew2(left, right)

    # Caller is responsible for checking left.shape == right.shape

    for i in range(n):
        # Analogous to: x = left[i]
        x = <object>(<PyObject**>cnp.PyArray_MultiIter_DATA(mi, 0))[0]
        y = <object>(<PyObject**>cnp.PyArray_MultiIter_DATA(mi, 1))[0]

        # we are either not equal or both nan
        # I think None == None will be true here
        try:
            if PyArray_Check(x) and PyArray_Check(y):
                if x.shape != y.shape:
                    return False
                if x.dtype == y.dtype == object:
                    if not array_equivalent_object(x, y):
                        return False
                else:
                    # Circular import isn't great, but so it goes.
                    # TODO: could use np.array_equal?
                    from pandas.core.dtypes.missing import array_equivalent

                    if not array_equivalent(x, y):
                        return False

            elif (x is C_NA) ^ (y is C_NA):
                return False
            elif not (
                PyObject_RichCompareBool(x, y, Py_EQ)
                or is_matching_na(x, y, nan_matches_none=True)
            ):
                return False
        except (ValueError, TypeError):
            # Avoid raising ValueError when comparing Numpy arrays to other types
            if cnp.PyArray_IsAnyScalar(x) != cnp.PyArray_IsAnyScalar(y):
                # Only compare scalars to scalars and non-scalars to non-scalars
                return False
            elif (not (cnp.PyArray_IsPythonScalar(x) or cnp.PyArray_IsPythonScalar(y))
                  and not (isinstance(x, type(y)) or isinstance(y, type(x)))):
                # Check if non-scalars have the same type
                return False
            elif check_na_tuples_nonequal(x, y):
                # We have tuples where one Side has a NA and the other side does not
                # Only condition we may end up with a TypeError
                return False
            raise

        cnp.PyArray_MultiIter_NEXT(mi)

    return True


ctypedef fused int6432_t:
    int64_t
    int32_t


@cython.wraparound(False)
@cython.boundscheck(False)
def is_range_indexer(ndarray[int6432_t, ndim=1] left, Py_ssize_t n) -> bool:
    """
    Perform an element by element comparison on 1-d integer arrays, meant for indexer
    comparisons
    """
    cdef:
        Py_ssize_t i

    if left.size != n:
        return False

    for i in range(n):

        if left[i] != i:
            return False

    return True


ctypedef fused ndarr_object:
    ndarray[object, ndim=1]
    ndarray[object, ndim=2]

# TODO: get rid of this in StringArray and modify
#  and go through ensure_string_array instead


@cython.wraparound(False)
@cython.boundscheck(False)
def convert_nans_to_NA(ndarr_object arr) -> ndarray:
    """
    Helper for StringArray that converts null values that
    are not pd.NA(e.g. np.nan, None) to pd.NA. Assumes elements
    have already been validated as null.
    """
    cdef:
        Py_ssize_t i, m, n
        object val
        ndarr_object result
    result = np.asarray(arr, dtype="object")
    if arr.ndim == 2:
        m, n = arr.shape[0], arr.shape[1]
        for i in range(m):
            for j in range(n):
                val = arr[i, j]
                if not isinstance(val, str):
                    result[i, j] = <object>C_NA
    else:
        n = len(arr)
        for i in range(n):
            val = arr[i]
            if not isinstance(val, str):
                result[i] = <object>C_NA
    return result


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef ndarray[object] ensure_string_array(
        arr,
        object na_value=np.nan,
        bint convert_na_value=True,
        bint copy=True,
        bint skipna=True,
):
    """
    Returns a new numpy array with object dtype and only strings and na values.

    Parameters
    ----------
    arr : array-like
        The values to be converted to str, if needed.
    na_value : Any, default np.nan
        The value to use for na. For example, np.nan or pd.NA.
    convert_na_value : bool, default True
        If False, existing na values will be used unchanged in the new array.
    copy : bool, default True
        Whether to ensure that a new array is returned.
    skipna : bool, default True
        Whether or not to coerce nulls to their stringified form
        (e.g. if False, NaN becomes 'nan').

    Returns
    -------
    np.ndarray[object]
        An array with the input array's elements casted to str or nan-like.
    """
    cdef:
        Py_ssize_t i = 0, n = len(arr)
        bint already_copied = True

    if hasattr(arr, "to_numpy"):

        if hasattr(arr, "dtype") and arr.dtype.kind in "mM":
            # dtype check to exclude DataFrame
            # GH#41409 TODO: not a great place for this
            out = arr.astype(str).astype(object)
            out[arr.isna()] = na_value
            return out
        arr = arr.to_numpy()
    elif not util.is_array(arr):
        arr = np.array(arr, dtype="object")

    result = np.asarray(arr, dtype="object")

    if copy and result is arr:
        result = result.copy()
    elif not copy and result is arr:
        already_copied = False

    if issubclass(arr.dtype.type, np.str_):
        # short-circuit, all elements are str
        return result

    for i in range(n):
        val = arr[i]

        if isinstance(val, str):
            continue

        elif not already_copied:
            result = result.copy()
            already_copied = True

        if not checknull(val):
            if isinstance(val, bytes):
                # GH#49658 discussion of desired behavior here
                result[i] = val.decode()
            elif not util.is_float_object(val):
                # f"{val}" is faster than str(val)
                result[i] = f"{val}"
            else:
                # f"{val}" is not always equivalent to str(val) for floats
                result[i] = str(val)
        else:
            if convert_na_value:
                val = na_value
            if skipna:
                result[i] = val
            else:
                result[i] = f"{val}"

    return result


def is_all_arraylike(obj: list) -> bool:
    """
    Should we treat these as levels of a MultiIndex, as opposed to Index items?
    """
    cdef:
        Py_ssize_t i, n = len(obj)
        object val
        bint all_arrays = True

    for i in range(n):
        val = obj[i]
        if not (isinstance(val, list) or
                util.is_array(val) or hasattr(val, "_data")):
            # TODO: EA?
            # exclude tuples, frozensets as they may be contained in an Index
            all_arrays = False
            break

    return all_arrays


# ------------------------------------------------------------------------------
# Groupby-related functions

# TODO: could do even better if we know something about the data. eg, index has
# 1-min data, binner has 5-min data, then bins are just strides in index. This
# is a general, O(max(len(values), len(binner))) method.
@cython.boundscheck(False)
@cython.wraparound(False)
def generate_bins_dt64(ndarray[int64_t, ndim=1] values, const int64_t[:] binner,
                       object closed="left", bint hasnans=False):
    """
    Int64 (datetime64) version of generic python version in ``groupby.py``.
    """
    cdef:
        Py_ssize_t lenidx, lenbin, i, j, bc
        ndarray[int64_t, ndim=1] bins
        int64_t r_bin, nat_count
        bint right_closed = closed == "right"

    nat_count = 0
    if hasnans:
        mask = values == NPY_NAT
        nat_count = np.sum(mask)
        values = values[~mask]

    lenidx = len(values)
    lenbin = len(binner)

    if lenidx <= 0 or lenbin <= 0:
        raise ValueError("Invalid length for values or for binner")

    # check binner fits data
    if values[0] < binner[0]:
        raise ValueError("Values falls before first bin")

    if values[lenidx - 1] > binner[lenbin - 1]:
        raise ValueError("Values falls after last bin")

    bins = np.empty(lenbin - 1, dtype=np.int64)

    j = 0  # index into values
    bc = 0  # bin count

    # linear scan
    if right_closed:
        for i in range(0, lenbin - 1):
            r_bin = binner[i + 1]
            # count values in current bin, advance to next bin
            while j < lenidx and values[j] <= r_bin:
                j += 1
            bins[bc] = j
            bc += 1
    else:
        for i in range(0, lenbin - 1):
            r_bin = binner[i + 1]
            # count values in current bin, advance to next bin
            while j < lenidx and values[j] < r_bin:
                j += 1
            bins[bc] = j
            bc += 1

    if nat_count > 0:
        # shift bins by the number of NaT
        bins = bins + nat_count
        bins = np.insert(bins, 0, nat_count)

    return bins


@cython.boundscheck(False)
@cython.wraparound(False)
def get_level_sorter(
    ndarray[int64_t, ndim=1] codes, const intp_t[:] starts
) -> ndarray:
    """
    Argsort for a single level of a multi-index, keeping the order of higher
    levels unchanged. `starts` points to starts of same-key indices w.r.t
    to leading levels; equivalent to:
        np.hstack([codes[starts[i]:starts[i+1]].argsort(kind='mergesort')
            + starts[i] for i in range(len(starts) - 1)])

    Parameters
    ----------
    codes : np.ndarray[int64_t, ndim=1]
    starts : np.ndarray[intp, ndim=1]

    Returns
    -------
    np.ndarray[np.int, ndim=1]
    """
    cdef:
        Py_ssize_t i, l, r
        ndarray[intp_t, ndim=1] out = cnp.PyArray_EMPTY(1, codes.shape, cnp.NPY_INTP, 0)

    for i in range(len(starts) - 1):
        l, r = starts[i], starts[i + 1]
        out[l:r] = l + codes[l:r].argsort(kind="mergesort")

    return out


@cython.boundscheck(False)
@cython.wraparound(False)
def count_level_2d(ndarray[uint8_t, ndim=2, cast=True] mask,
                   const intp_t[:] labels,
                   Py_ssize_t max_bin,
                   ):
    cdef:
        Py_ssize_t i, j, k, n
        ndarray[int64_t, ndim=2] counts

    n, k = (<object>mask).shape

    counts = np.zeros((n, max_bin), dtype="i8")
    with nogil:
        for i in range(n):
            for j in range(k):
                if mask[i, j]:
                    counts[i, labels[j]] += 1

    return counts


@cython.wraparound(False)
@cython.boundscheck(False)
def generate_slices(const intp_t[:] labels, Py_ssize_t ngroups):
    cdef:
        Py_ssize_t i, group_size, n, start
        intp_t lab
        int64_t[::1] starts, ends

    n = len(labels)

    starts = np.zeros(ngroups, dtype=np.int64)
    ends = np.zeros(ngroups, dtype=np.int64)

    start = 0
    group_size = 0
    with nogil:
        for i in range(n):
            lab = labels[i]
            if lab < 0:
                start += 1
            else:
                group_size += 1
                if i == n - 1 or lab != labels[i + 1]:
                    starts[lab] = start
                    ends[lab] = start + group_size
                    start += group_size
                    group_size = 0

    return np.asarray(starts), np.asarray(ends)


def indices_fast(ndarray[intp_t, ndim=1] index, const int64_t[:] labels, list keys,
                 list sorted_labels) -> dict:
    """
    Parameters
    ----------
    index : ndarray[intp]
    labels : ndarray[int64]
    keys : list
    sorted_labels : list[ndarray[int64]]
    """
    cdef:
        Py_ssize_t i, j, k, lab, cur, start, n = len(labels)
        dict result = {}
        object tup

    k = len(keys)

    # Start at the first non-null entry
    j = 0
    for j in range(0, n):
        if labels[j] != -1:
            break
    else:
        return result
    cur = labels[j]
    start = j

    for i in range(j+1, n):
        lab = labels[i]

        if lab != cur:
            if lab != -1:
                if k == 1:
                    # When k = 1 we do not want to return a tuple as key
                    tup = keys[0][sorted_labels[0][i - 1]]
                else:
                    tup = PyTuple_New(k)
                    for j in range(k):
                        val = keys[j][sorted_labels[j][i - 1]]
                        PyTuple_SET_ITEM(tup, j, val)
                        Py_INCREF(val)
                result[tup] = index[start:i]
            start = i
        cur = lab

    if k == 1:
        # When k = 1 we do not want to return a tuple as key
        tup = keys[0][sorted_labels[0][n - 1]]
    else:
        tup = PyTuple_New(k)
        for j in range(k):
            val = keys[j][sorted_labels[j][n - 1]]
            PyTuple_SET_ITEM(tup, j, val)
            Py_INCREF(val)
    result[tup] = index[start:]

    return result


# core.common import for fast inference checks

def is_float(obj: object) -> bool:
    """
    Return True if given object is float.

    Returns
    -------
    bool

    Examples
    --------
    >>> pd.api.types.is_float(1.0)
    True

    >>> pd.api.types.is_float(1)
    False
    """
    return util.is_float_object(obj)


def is_integer(obj: object) -> bool:
    """
    Return True if given object is integer.

    Returns
    -------
    bool

    Examples
    --------
    >>> pd.api.types.is_integer(1)
    True

    >>> pd.api.types.is_integer(1.0)
    False
    """
    return util.is_integer_object(obj)


def is_int_or_none(obj) -> bool:
    """
    Return True if given object is integer or None.

    Returns
    -------
    bool
    """
    return obj is None or util.is_integer_object(obj)


def is_bool(obj: object) -> bool:
    """
    Return True if given object is boolean.

    Returns
    -------
    bool

    Examples
    --------
    >>> pd.api.types.is_bool(True)
    True

    >>> pd.api.types.is_bool(1)
    False
    """
    return util.is_bool_object(obj)


def is_complex(obj: object) -> bool:
    """
    Return True if given object is complex.

    Returns
    -------
    bool

    Examples
    --------
    >>> pd.api.types.is_complex(1 + 1j)
    True

    >>> pd.api.types.is_complex(1)
    False
    """
    return util.is_complex_object(obj)


cpdef bint is_decimal(object obj):
    return isinstance(obj, Decimal)


cpdef bint is_interval(object obj):
    return getattr(obj, "_typ", "_typ") == "interval"


def is_period(val: object) -> bool:
    """
    Return True if given object is Period.

    Returns
    -------
    bool
    """
    return is_period_object(val)


def is_list_like(obj: object, allow_sets: bool = True) -> bool:
    """
    Check if the object is list-like.

    Objects that are considered list-like are for example Python
    lists, tuples, sets, NumPy arrays, and Pandas Series.

    Strings and datetime objects, however, are not considered list-like.

    Parameters
    ----------
    obj : object
        Object to check.
    allow_sets : bool, default True
        If this parameter is False, sets will not be considered list-like.

    Returns
    -------
    bool
        Whether `obj` has list-like properties.

    Examples
    --------
    >>> import datetime
    >>> from pandas.api.types import is_list_like
    >>> is_list_like([1, 2, 3])
    True
    >>> is_list_like({1, 2, 3})
    True
    >>> is_list_like(datetime.datetime(2017, 1, 1))
    False
    >>> is_list_like("foo")
    False
    >>> is_list_like(1)
    False
    >>> is_list_like(np.array([2]))
    True
    >>> is_list_like(np.array(2))
    False
    """
    return c_is_list_like(obj, allow_sets)


cdef bint c_is_list_like(object obj, bint allow_sets) except -1:
    # first, performance short-cuts for the most common cases
    if util.is_array(obj):
        # exclude zero-dimensional numpy arrays, effectively scalars
        return not cnp.PyArray_IsZeroDim(obj)
    elif isinstance(obj, list):
        return True
    # then the generic implementation
    return (
        # equiv: `isinstance(obj, abc.Iterable)`
        getattr(obj, "__iter__", None) is not None and not isinstance(obj, type)
        # we do not count strings/unicode/bytes as list-like
        # exclude Generic types that have __iter__
        and not isinstance(obj, (str, bytes, _GenericAlias))
        # exclude zero-dimensional duck-arrays, effectively scalars
        and not (hasattr(obj, "ndim") and obj.ndim == 0)
        # exclude sets if allow_sets is False
        and not (allow_sets is False and isinstance(obj, abc.Set))
    )


def is_pyarrow_array(obj):
    """
    Return True if given object is a pyarrow Array or ChunkedArray.

    Returns
    -------
    bool
    """
    if PYARROW_INSTALLED:
        return isinstance(obj, (pa.Array, pa.ChunkedArray))
    return False


_TYPE_MAP = {
    "categorical": "categorical",
    "category": "categorical",
    "int8": "integer",
    "int16": "integer",
    "int32": "integer",
    "int64": "integer",
    "i": "integer",
    "uint8": "integer",
    "uint16": "integer",
    "uint32": "integer",
    "uint64": "integer",
    "u": "integer",
    "float32": "floating",
    "float64": "floating",
    "float128": "floating",
    "float256": "floating",
    "f": "floating",
    "complex64": "complex",
    "complex128": "complex",
    "complex256": "complex",
    "c": "complex",
    "string": "string",
    str: "string",
    "S": "bytes",
    "U": "string",
    "bool": "boolean",
    "b": "boolean",
    "datetime64[ns]": "datetime64",
    "M": "datetime64",
    "timedelta64[ns]": "timedelta64",
    "m": "timedelta64",
    "interval": "interval",
    Period: "period",
    datetime: "datetime64",
    date: "date",
    time: "time",
    timedelta: "timedelta64",
    Decimal: "decimal",
    bytes: "bytes",
}


@cython.internal
cdef class Seen:
    """
    Class for keeping track of the types of elements
    encountered when trying to perform type conversions.
    """

    cdef:
        bint int_             # seen_int
        bint nat_             # seen nat
        bint bool_            # seen_bool
        bint null_            # seen_null
        bint nan_             # seen_np.nan
        bint uint_            # seen_uint (unsigned integer)
        bint sint_            # seen_sint (signed integer)
        bint float_           # seen_float
        bint object_          # seen_object
        bint complex_         # seen_complex
        bint datetime_        # seen_datetime
        bint coerce_numeric   # coerce data to numeric
        bint timedelta_       # seen_timedelta
        bint datetimetz_      # seen_datetimetz
        bint period_          # seen_period
        bint interval_        # seen_interval
        bint str_             # seen_str

    def __cinit__(self, bint coerce_numeric=False):
        """
        Initialize a Seen instance.

        Parameters
        ----------
        coerce_numeric : bool, default False
            Whether or not to force conversion to a numeric data type if
            initial methods to convert to numeric fail.
        """
        self.int_ = False
        self.nat_ = False
        self.bool_ = False
        self.null_ = False
        self.nan_ = False
        self.uint_ = False
        self.sint_ = False
        self.float_ = False
        self.object_ = False
        self.complex_ = False
        self.datetime_ = False
        self.timedelta_ = False
        self.datetimetz_ = False
        self.period_ = False
        self.interval_ = False
        self.str_ = False
        self.coerce_numeric = coerce_numeric

    cdef bint check_uint64_conflict(self) except -1:
        """
        Check whether we can safely convert a uint64 array to a numeric dtype.

        There are two cases when conversion to numeric dtype with a uint64
        array is not safe (and will therefore not be performed)

        1) A NaN element is encountered.

           uint64 cannot be safely cast to float64 due to truncation issues
           at the extreme ends of the range.

        2) A negative number is encountered.

           There is no numerical dtype that can hold both negative numbers
           and numbers greater than INT64_MAX. Hence, at least one number
           will be improperly cast if we convert to a numeric dtype.

        Returns
        -------
        bool
            Whether or not we should return the original input array to avoid
            data truncation.

        Raises
        ------
        ValueError
            uint64 elements were detected, and at least one of the
            two conflict cases was also detected. However, we are
            trying to force conversion to a numeric dtype.
        """
        return (self.uint_ and (self.null_ or self.sint_)
                and not self.coerce_numeric)

    cdef saw_null(self):
        """
        Set flags indicating that a null value was encountered.
        """
        self.null_ = True
        self.float_ = True

    cdef saw_int(self, object val):
        """
        Set flags indicating that an integer value was encountered.

        In addition to setting a flag that an integer was seen, we
        also set two flags depending on the type of integer seen:

        1) sint_ : a signed numpy integer type or a negative (signed) number in the
                   range of [-2**63, 0) was encountered
        2) uint_ : an unsigned numpy integer type or a positive number in the range of
                   [2**63, 2**64) was encountered

        Parameters
        ----------
        val : Python int
            Value with which to set the flags.
        """
        self.int_ = True
        self.sint_ = (
            self.sint_
            or (oINT64_MIN <= val < 0)
            # Cython equivalent of `isinstance(val, np.signedinteger)`
            or PyObject_TypeCheck(val, &PySignedIntegerArrType_Type)
        )
        self.uint_ = (
            self.uint_
            or (oINT64_MAX < val <= oUINT64_MAX)
            # Cython equivalent of `isinstance(val, np.unsignedinteger)`
            or PyObject_TypeCheck(val, &PyUnsignedIntegerArrType_Type)
        )

    @property
    def numeric_(self):
        return self.complex_ or self.float_ or self.int_

    @property
    def is_bool(self):
        # i.e. not (anything but bool)
        return self.is_bool_or_na and not (self.nan_ or self.null_)

    @property
    def is_bool_or_na(self):
        # i.e. not (anything but bool or missing values)
        return self.bool_ and not (
            self.datetime_ or self.datetimetz_ or self.nat_ or self.timedelta_
            or self.period_ or self.interval_ or self.numeric_ or self.object_
        )


cdef object _try_infer_map(object dtype):
    """
    If its in our map, just return the dtype.
    """
    cdef:
        object val
        str attr
    for attr in ["type", "kind", "name", "base"]:
        # Checking type before kind matters for ArrowDtype cases
        val = getattr(dtype, attr, None)
        if val in _TYPE_MAP:
            return _TYPE_MAP[val]
    return None


def infer_dtype(value: object, skipna: bool = True) -> str:
    """
    Return a string label of the type of a scalar or list-like of values.

    Parameters
    ----------
    value : scalar, list, ndarray, or pandas type
    skipna : bool, default True
        Ignore NaN values when inferring the type.

    Returns
    -------
    str
        Describing the common type of the input data.
    Results can include:

    - string
    - bytes
    - floating
    - integer
    - mixed-integer
    - mixed-integer-float
    - decimal
    - complex
    - categorical
    - boolean
    - datetime64
    - datetime
    - date
    - timedelta64
    - timedelta
    - time
    - period
    - mixed
    - unknown-array

    Raises
    ------
    TypeError
        If ndarray-like but cannot infer the dtype

    Notes
    -----
    - 'mixed' is the catchall for anything that is not otherwise
      specialized
    - 'mixed-integer-float' are floats and integers
    - 'mixed-integer' are integers mixed with non-integers
    - 'unknown-array' is the catchall for something that *is* an array (has
      a dtype attribute), but has a dtype unknown to pandas (e.g. external
      extension array)

    Examples
    --------
    >>> from pandas.api.types import infer_dtype
    >>> infer_dtype(['foo', 'bar'])
    'string'

    >>> infer_dtype(['a', np.nan, 'b'], skipna=True)
    'string'

    >>> infer_dtype(['a', np.nan, 'b'], skipna=False)
    'mixed'

    >>> infer_dtype([b'foo', b'bar'])
    'bytes'

    >>> infer_dtype([1, 2, 3])
    'integer'

    >>> infer_dtype([1, 2, 3.5])
    'mixed-integer-float'

    >>> infer_dtype([1.0, 2.0, 3.5])
    'floating'

    >>> infer_dtype(['a', 1])
    'mixed-integer'

    >>> from decimal import Decimal
    >>> infer_dtype([Decimal(1), Decimal(2.0)])
    'decimal'

    >>> infer_dtype([True, False])
    'boolean'

    >>> infer_dtype([True, False, np.nan])
    'boolean'

    >>> infer_dtype([pd.Timestamp('20130101')])
    'datetime'

    >>> import datetime
    >>> infer_dtype([datetime.date(2013, 1, 1)])
    'date'

    >>> infer_dtype([np.datetime64('2013-01-01')])
    'datetime64'

    >>> infer_dtype([datetime.timedelta(0, 1, 1)])
    'timedelta'

    >>> infer_dtype(pd.Series(list('aabc')).astype('category'))
    'categorical'
    """
    cdef:
        Py_ssize_t i, n
        object val
        ndarray values
        bint seen_pdnat = False
        bint seen_val = False
        flatiter it

    if util.is_array(value):
        values = value
    elif hasattr(type(value), "inferred_type") and skipna is False:
        # Index, use the cached attribute if possible, populate the cache otherwise
        return value.inferred_type
    elif hasattr(value, "dtype"):
        inferred = _try_infer_map(value.dtype)
        if inferred is not None:
            return inferred
        elif not cnp.PyArray_DescrCheck(value.dtype):
            return "unknown-array"
        # Unwrap Series/Index
        values = np.asarray(value)
    else:
        if not isinstance(value, list):
            value = list(value)
        if not value:
            return "empty"

        from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
        values = construct_1d_object_array_from_listlike(value)

    inferred = _try_infer_map(values.dtype)
    if inferred is not None:
        # Anything other than object-dtype should return here.
        return inferred

    if values.descr.type_num != NPY_OBJECT:
        # i.e. values.dtype != np.object_
        # This should not be reached
        values = values.astype(object)

    n = cnp.PyArray_SIZE(values)
    if n == 0:
        return "empty"

    # Iterate until we find our first valid value. We will use this
    #  value to decide which of the is_foo_array functions to call.
    it = PyArray_IterNew(values)
    for i in range(n):
        # The PyArray_GETITEM and PyArray_ITER_NEXT are faster
        #  equivalents to `val = values[i]`
        val = PyArray_GETITEM(values, PyArray_ITER_DATA(it))
        PyArray_ITER_NEXT(it)

        # do not use checknull to keep
        # np.datetime64('nat') and np.timedelta64('nat')
        if val is None or util.is_nan(val) or val is C_NA:
            pass
        elif val is NaT:
            seen_pdnat = True
        else:
            seen_val = True
            break

    # if all values are nan/NaT
    if seen_val is False and seen_pdnat is True:
        return "datetime"
        # float/object nan is handled in latter logic
    if seen_val is False and skipna:
        return "empty"

    if util.is_datetime64_object(val):
        if is_datetime64_array(values, skipna=skipna):
            return "datetime64"

    elif is_timedelta(val):
        if is_timedelta_or_timedelta64_array(values, skipna=skipna):
            return "timedelta"

    elif util.is_integer_object(val):
        # ordering matters here; this check must come after the is_timedelta
        #  check otherwise numpy timedelta64 objects would come through here

        if is_integer_array(values, skipna=skipna):
            return "integer"
        elif is_integer_float_array(values, skipna=skipna):
            if is_integer_na_array(values, skipna=skipna):
                return "integer-na"
            else:
                return "mixed-integer-float"
        return "mixed-integer"

    elif PyDateTime_Check(val):
        if is_datetime_array(values, skipna=skipna):
            return "datetime"
        elif is_date_array(values, skipna=skipna):
            return "date"

    elif PyDate_Check(val):
        if is_date_array(values, skipna=skipna):
            return "date"

    elif PyTime_Check(val):
        if is_time_array(values, skipna=skipna):
            return "time"

    elif is_decimal(val):
        if is_decimal_array(values, skipna=skipna):
            return "decimal"

    elif util.is_complex_object(val):
        if is_complex_array(values):
            return "complex"

    elif util.is_float_object(val):
        if is_float_array(values):
            return "floating"
        elif is_integer_float_array(values, skipna=skipna):
            if is_integer_na_array(values, skipna=skipna):
                return "integer-na"
            else:
                return "mixed-integer-float"

    elif util.is_bool_object(val):
        if is_bool_array(values, skipna=skipna):
            return "boolean"

    elif isinstance(val, str):
        if is_string_array(values, skipna=skipna):
            return "string"

    elif isinstance(val, bytes):
        if is_bytes_array(values, skipna=skipna):
            return "bytes"

    elif is_period_object(val):
        if is_period_array(values, skipna=skipna):
            return "period"

    elif is_interval(val):
        if is_interval_array(values):
            return "interval"

    cnp.PyArray_ITER_RESET(it)
    for i in range(n):
        val = PyArray_GETITEM(values, PyArray_ITER_DATA(it))
        PyArray_ITER_NEXT(it)

        if util.is_integer_object(val):
            return "mixed-integer"

    return "mixed"


cdef bint is_timedelta(object o):
    return PyDelta_Check(o) or util.is_timedelta64_object(o)


@cython.internal
cdef class Validator:

    cdef:
        Py_ssize_t n
        dtype dtype
        bint skipna

    def __cinit__(self, Py_ssize_t n, dtype dtype=np.dtype(np.object_),
                  bint skipna=False):
        self.n = n
        self.dtype = dtype
        self.skipna = skipna

    cdef bint validate(self, ndarray values) except -1:
        if not self.n:
            return False

        if self.is_array_typed():
            # i.e. this ndarray is already of the desired dtype
            return True
        elif self.dtype.type_num == NPY_OBJECT:
            if self.skipna:
                return self._validate_skipna(values)
            else:
                return self._validate(values)
        else:
            return False

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef bint _validate(self, ndarray values) except -1:
        cdef:
            Py_ssize_t i
            Py_ssize_t n = values.size
            flatiter it = PyArray_IterNew(values)

        for i in range(n):
            # The PyArray_GETITEM and PyArray_ITER_NEXT are faster
            #  equivalents to `val = values[i]`
            val = PyArray_GETITEM(values, PyArray_ITER_DATA(it))
            PyArray_ITER_NEXT(it)
            if not self.is_valid(val):
                return False

        return True

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef bint _validate_skipna(self, ndarray values) except -1:
        cdef:
            Py_ssize_t i
            Py_ssize_t n = values.size
            flatiter it = PyArray_IterNew(values)

        for i in range(n):
            # The PyArray_GETITEM and PyArray_ITER_NEXT are faster
            #  equivalents to `val = values[i]`
            val = PyArray_GETITEM(values, PyArray_ITER_DATA(it))
            PyArray_ITER_NEXT(it)
            if not self.is_valid_skipna(val):
                return False

        return True

    cdef bint is_valid(self, object value) except -1:
        return self.is_value_typed(value)

    cdef bint is_valid_skipna(self, object value) except -1:
        return self.is_valid(value) or self.is_valid_null(value)

    cdef bint is_value_typed(self, object value) except -1:
        raise NotImplementedError(f"{type(self).__name__} child class "
                                  "must define is_value_typed")

    cdef bint is_valid_null(self, object value) except -1:
        return value is None or value is C_NA or util.is_nan(value)
        # TODO: include decimal NA?

    cdef bint is_array_typed(self) except -1:
        return False


@cython.internal
cdef class BoolValidator(Validator):
    cdef bint is_value_typed(self, object value) except -1:
        return util.is_bool_object(value)

    cdef bint is_array_typed(self) except -1:
        return issubclass(self.dtype.type, np.bool_)


cpdef bint is_bool_array(ndarray values, bint skipna=False):
    cdef:
        BoolValidator validator = BoolValidator(len(values),
                                                values.dtype,
                                                skipna=skipna)
    return validator.validate(values)


@cython.internal
cdef class IntegerValidator(Validator):
    cdef bint is_value_typed(self, object value) except -1:
        return util.is_integer_object(value)

    cdef bint is_array_typed(self) except -1:
        return issubclass(self.dtype.type, np.integer)


# Note: only python-exposed for tests
cpdef bint is_integer_array(ndarray values, bint skipna=True):
    cdef:
        IntegerValidator validator = IntegerValidator(len(values),
                                                      values.dtype,
                                                      skipna=skipna)
    return validator.validate(values)


@cython.internal
cdef class IntegerNaValidator(Validator):
    cdef bint is_value_typed(self, object value) except -1:
        return (util.is_integer_object(value)
                or (util.is_nan(value) and util.is_float_object(value)))


cdef bint is_integer_na_array(ndarray values, bint skipna=True):
    cdef:
        IntegerNaValidator validator = IntegerNaValidator(len(values),
                                                          values.dtype, skipna=skipna)
    return validator.validate(values)


@cython.internal
cdef class IntegerFloatValidator(Validator):
    cdef bint is_value_typed(self, object value) except -1:
        return util.is_integer_object(value) or util.is_float_object(value)

    cdef bint is_array_typed(self) except -1:
        return issubclass(self.dtype.type, np.integer)


cdef bint is_integer_float_array(ndarray values, bint skipna=True):
    cdef:
        IntegerFloatValidator validator = IntegerFloatValidator(len(values),
                                                                values.dtype,
                                                                skipna=skipna)
    return validator.validate(values)


@cython.internal
cdef class FloatValidator(Validator):
    cdef bint is_value_typed(self, object value) except -1:
        return util.is_float_object(value)

    cdef bint is_array_typed(self) except -1:
        return issubclass(self.dtype.type, np.floating)


# Note: only python-exposed for tests
cpdef bint is_float_array(ndarray values):
    cdef:
        FloatValidator validator = FloatValidator(len(values), values.dtype)
    return validator.validate(values)


@cython.internal
cdef class ComplexValidator(Validator):
    cdef bint is_value_typed(self, object value) except -1:
        return (
            util.is_complex_object(value)
            or (util.is_float_object(value) and is_nan(value))
        )

    cdef bint is_array_typed(self) except -1:
        return issubclass(self.dtype.type, np.complexfloating)


cdef bint is_complex_array(ndarray values):
    cdef:
        ComplexValidator validator = ComplexValidator(len(values), values.dtype)
    return validator.validate(values)


@cython.internal
cdef class DecimalValidator(Validator):
    cdef bint is_value_typed(self, object value) except -1:
        return is_decimal(value)


cdef bint is_decimal_array(ndarray values, bint skipna=False):
    cdef:
        DecimalValidator validator = DecimalValidator(
            len(values), values.dtype, skipna=skipna
        )
    return validator.validate(values)


@cython.internal
cdef class StringValidator(Validator):
    cdef bint is_value_typed(self, object value) except -1:
        return isinstance(value, str)

    cdef bint is_array_typed(self) except -1:
        return issubclass(self.dtype.type, np.str_)


cpdef bint is_string_array(ndarray values, bint skipna=False):
    cdef:
        StringValidator validator = StringValidator(len(values),
                                                    values.dtype,
                                                    skipna=skipna)
    return validator.validate(values)


@cython.internal
cdef class BytesValidator(Validator):
    cdef bint is_value_typed(self, object value) except -1:
        return isinstance(value, bytes)

    cdef bint is_array_typed(self) except -1:
        return issubclass(self.dtype.type, np.bytes_)


cdef bint is_bytes_array(ndarray values, bint skipna=False):
    cdef:
        BytesValidator validator = BytesValidator(len(values), values.dtype,
                                                  skipna=skipna)
    return validator.validate(values)


@cython.internal
cdef class TemporalValidator(Validator):
    cdef:
        bint all_generic_na

    def __cinit__(self, Py_ssize_t n, dtype dtype=np.dtype(np.object_),
                  bint skipna=False):
        self.n = n
        self.dtype = dtype
        self.skipna = skipna
        self.all_generic_na = True

    cdef bint is_valid(self, object value) except -1:
        return self.is_value_typed(value) or self.is_valid_null(value)

    cdef bint is_valid_null(self, object value) except -1:
        raise NotImplementedError(f"{type(self).__name__} child class "
                                  "must define is_valid_null")

    cdef bint is_valid_skipna(self, object value) except -1:
        cdef:
            bint is_typed_null = self.is_valid_null(value)
            bint is_generic_null = value is None or util.is_nan(value)
        if not is_generic_null:
            self.all_generic_na = False
        return self.is_value_typed(value) or is_typed_null or is_generic_null

    cdef bint _validate_skipna(self, ndarray values) except -1:
        """
        If we _only_ saw non-dtype-specific NA values, even if they are valid
        for this dtype, we do not infer this dtype.
        """
        return Validator._validate_skipna(self, values) and not self.all_generic_na


@cython.internal
cdef class DatetimeValidator(TemporalValidator):
    cdef bint is_value_typed(self, object value) except -1:
        return PyDateTime_Check(value)

    cdef bint is_valid_null(self, object value) except -1:
        return is_null_datetime64(value)


cpdef bint is_datetime_array(ndarray values, bint skipna=True):
    cdef:
        DatetimeValidator validator = DatetimeValidator(len(values),
                                                        skipna=skipna)
    return validator.validate(values)


@cython.internal
cdef class Datetime64Validator(DatetimeValidator):
    cdef bint is_value_typed(self, object value) except -1:
        return util.is_datetime64_object(value)


# Note: only python-exposed for tests
cpdef bint is_datetime64_array(ndarray values, bint skipna=True):
    cdef:
        Datetime64Validator validator = Datetime64Validator(len(values),
                                                            skipna=skipna)
    return validator.validate(values)


@cython.internal
cdef class AnyDatetimeValidator(DatetimeValidator):
    cdef bint is_value_typed(self, object value) except -1:
        return util.is_datetime64_object(value) or (
            PyDateTime_Check(value) and value.tzinfo is None
        )


cdef bint is_datetime_or_datetime64_array(ndarray values, bint skipna=True):
    cdef:
        AnyDatetimeValidator validator = AnyDatetimeValidator(len(values),
                                                              skipna=skipna)
    return validator.validate(values)


# Note: only python-exposed for tests
def is_datetime_with_singletz_array(values: ndarray) -> bool:
    """
    Check values have the same tzinfo attribute.
    Doesn't check values are datetime-like types.
    """
    cdef:
        Py_ssize_t i = 0, j, n = len(values)
        object base_val, base_tz, val, tz

    if n == 0:
        return False

    # Get a reference timezone to compare with the rest of the tzs in the array
    for i in range(n):
        base_val = values[i]
        if base_val is not NaT and base_val is not None and not util.is_nan(base_val):
            base_tz = getattr(base_val, "tzinfo", None)
            break

    for j in range(i, n):
        # Compare val's timezone with the reference timezone
        # NaT can coexist with tz-aware datetimes, so skip if encountered
        val = values[j]
        if val is not NaT and val is not None and not util.is_nan(val):
            tz = getattr(val, "tzinfo", None)
            if not tz_compare(base_tz, tz):
                return False

    # Note: we should only be called if a tzaware datetime has been seen,
    #  so base_tz should always be set at this point.
    return True


@cython.internal
cdef class TimedeltaValidator(TemporalValidator):
    cdef bint is_value_typed(self, object value) except -1:
        return PyDelta_Check(value)

    cdef bint is_valid_null(self, object value) except -1:
        return is_null_timedelta64(value)


@cython.internal
cdef class AnyTimedeltaValidator(TimedeltaValidator):
    cdef bint is_value_typed(self, object value) except -1:
        return is_timedelta(value)


# Note: only python-exposed for tests
cpdef bint is_timedelta_or_timedelta64_array(ndarray values, bint skipna=True):
    """
    Infer with timedeltas and/or nat/none.
    """
    cdef:
        AnyTimedeltaValidator validator = AnyTimedeltaValidator(len(values),
                                                                skipna=skipna)
    return validator.validate(values)


@cython.internal
cdef class DateValidator(Validator):
    cdef bint is_value_typed(self, object value) except -1:
        return PyDate_Check(value)


# Note: only python-exposed for tests
cpdef bint is_date_array(ndarray values, bint skipna=False):
    cdef:
        DateValidator validator = DateValidator(len(values), skipna=skipna)
    return validator.validate(values)


@cython.internal
cdef class TimeValidator(Validator):
    cdef bint is_value_typed(self, object value) except -1:
        return PyTime_Check(value)


# Note: only python-exposed for tests
cpdef bint is_time_array(ndarray values, bint skipna=False):
    cdef:
        TimeValidator validator = TimeValidator(len(values), skipna=skipna)
    return validator.validate(values)


# FIXME: actually use skipna
cdef bint is_period_array(ndarray values, bint skipna=True):
    """
    Is this an ndarray of Period objects (or NaT) with a single `freq`?
    """
    # values should be object-dtype, but ndarray[object] assumes 1D, while
    #  this _may_ be 2D.
    cdef:
        Py_ssize_t i, N = values.size
        int dtype_code = -10000  # i.e. c_FreqGroup.FR_UND
        object val
        flatiter it

    if N == 0:
        return False

    it = PyArray_IterNew(values)
    for i in range(N):
        # The PyArray_GETITEM and PyArray_ITER_NEXT are faster
        #  equivalents to `val = values[i]`
        val = PyArray_GETITEM(values, PyArray_ITER_DATA(it))
        PyArray_ITER_NEXT(it)

        if is_period_object(val):
            if dtype_code == -10000:
                dtype_code = val._dtype._dtype_code
            elif dtype_code != val._dtype._dtype_code:
                # mismatched freqs
                return False
        elif checknull_with_nat(val):
            pass
        else:
            # Not a Period or NaT-like
            return False

    if dtype_code == -10000:
        # we saw all-NaTs, no actual Periods
        return False
    return True


# Note: only python-exposed for tests
cpdef bint is_interval_array(ndarray values):
    """
    Is this an ndarray of Interval (or np.nan) with a single dtype?
    """
    cdef:
        Py_ssize_t i, n = len(values)
        str closed = None
        bint numeric = False
        bint dt64 = False
        bint td64 = False
        object val

    if len(values) == 0:
        return False

    for i in range(n):
        val = values[i]

        if is_interval(val):
            if closed is None:
                closed = val.closed
                numeric = (
                    util.is_float_object(val.left)
                    or util.is_integer_object(val.left)
                )
                td64 = is_timedelta(val.left)
                dt64 = PyDateTime_Check(val.left)
            elif val.closed != closed:
                # mismatched closedness
                return False
            elif numeric:
                if not (
                    util.is_float_object(val.left)
                    or util.is_integer_object(val.left)
                ):
                    # i.e. datetime64 or timedelta64
                    return False
            elif td64:
                if not is_timedelta(val.left):
                    return False
            elif dt64:
                if not PyDateTime_Check(val.left):
                    return False
            else:
                raise ValueError(val)
        elif util.is_nan(val) or val is None:
            pass
        else:
            return False

    if closed is None:
        # we saw all-NAs, no actual Intervals
        return False
    return True


@cython.boundscheck(False)
@cython.wraparound(False)
def maybe_convert_numeric(
    ndarray[object, ndim=1] values,
    set na_values,
    bint convert_empty=True,
    bint coerce_numeric=False,
    bint convert_to_masked_nullable=False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Convert object array to a numeric array if possible.

    Parameters
    ----------
    values : ndarray[object]
        Array of object elements to convert.
    na_values : set
        Set of values that should be interpreted as NaN.
    convert_empty : bool, default True
        If an empty array-like object is encountered, whether to interpret
        that element as NaN or not. If set to False, a ValueError will be
        raised if such an element is encountered and 'coerce_numeric' is False.
    coerce_numeric : bool, default False
        If initial attempts to convert to numeric have failed, whether to
        force conversion to numeric via alternative methods or by setting the
        element to NaN. Otherwise, an Exception will be raised when such an
        element is encountered.

        This boolean also has an impact on how conversion behaves when a
        numeric array has no suitable numerical dtype to return (i.e. uint64,
        int32, uint8). If set to False, the original object array will be
        returned. Otherwise, a ValueError will be raised.
    convert_to_masked_nullable : bool, default False
        Whether to return a mask for the converted values. This also disables
        upcasting for ints with nulls to float64.
    Returns
    -------
    np.ndarray
        Array of converted object values to numerical ones.

    Optional[np.ndarray]
        If convert_to_masked_nullable is True,
        returns a boolean mask for the converted values, otherwise returns None.
    """
    if len(values) == 0:
        return (np.array([], dtype="i8"), None)

    # fastpath for ints - try to convert all based on first value
    cdef:
        object val = values[0]

    if util.is_integer_object(val):
        try:
            maybe_ints = values.astype("i8")
            if (maybe_ints == values).all():
                return (maybe_ints, None)
        except (ValueError, OverflowError, TypeError):
            pass

    # Otherwise, iterate and do full inference.
    cdef:
        int maybe_int
        Py_ssize_t i, n = values.size
        Seen seen = Seen(coerce_numeric)
        ndarray[float64_t, ndim=1] floats = cnp.PyArray_EMPTY(
            1, values.shape, cnp.NPY_FLOAT64, 0
        )
        ndarray[complex128_t, ndim=1] complexes = cnp.PyArray_EMPTY(
            1, values.shape, cnp.NPY_COMPLEX128, 0
        )
        ndarray[int64_t, ndim=1] ints = cnp.PyArray_EMPTY(
            1, values.shape, cnp.NPY_INT64, 0
        )
        ndarray[uint64_t, ndim=1] uints = cnp.PyArray_EMPTY(
            1, values.shape, cnp.NPY_UINT64, 0
        )
        ndarray[uint8_t, ndim=1] bools = cnp.PyArray_EMPTY(
            1, values.shape, cnp.NPY_UINT8, 0
        )
        ndarray[uint8_t, ndim=1] mask = np.zeros(n, dtype="u1")
        float64_t fval
        bint allow_null_in_int = convert_to_masked_nullable

    for i in range(n):
        val = values[i]
        # We only want to disable NaNs showing as float if
        # a) convert_to_masked_nullable = True
        # b) no floats have been seen ( assuming an int shows up later )
        # However, if no ints present (all null array), we need to return floats
        allow_null_in_int = convert_to_masked_nullable and not seen.float_

        if val.__hash__ is not None and val in na_values:
            if allow_null_in_int:
                seen.null_ = True
                mask[i] = 1
            else:
                if convert_to_masked_nullable:
                    mask[i] = 1
                seen.saw_null()
            floats[i] = complexes[i] = NaN
        elif util.is_float_object(val):
            fval = val
            if fval != fval:
                seen.null_ = True
                if allow_null_in_int:
                    mask[i] = 1
                else:
                    if convert_to_masked_nullable:
                        mask[i] = 1
                    seen.float_ = True
            else:
                seen.float_ = True
            floats[i] = complexes[i] = fval
        elif util.is_integer_object(val):
            floats[i] = complexes[i] = val

            val = int(val)
            seen.saw_int(val)

            if val >= 0:
                if val <= oUINT64_MAX:
                    uints[i] = val
                else:
                    seen.float_ = True

            if oINT64_MIN <= val <= oINT64_MAX:
                ints[i] = val

            if val < oINT64_MIN or (seen.sint_ and seen.uint_):
                seen.float_ = True

        elif util.is_bool_object(val):
            floats[i] = uints[i] = ints[i] = bools[i] = val
            seen.bool_ = True
        elif val is None or val is C_NA:
            if allow_null_in_int:
                seen.null_ = True
                mask[i] = 1
            else:
                if convert_to_masked_nullable:
                    mask[i] = 1
                seen.saw_null()
            floats[i] = complexes[i] = NaN
        elif hasattr(val, "__len__") and len(val) == 0:
            if convert_empty or seen.coerce_numeric:
                seen.saw_null()
                floats[i] = complexes[i] = NaN
                mask[i] = 1
            else:
                raise ValueError("Empty string encountered")
        elif util.is_complex_object(val):
            complexes[i] = val
            seen.complex_ = True
        elif is_decimal(val):
            floats[i] = complexes[i] = val
            seen.float_ = True
        else:
            try:
                floatify(val, &fval, &maybe_int)

                if fval in na_values:
                    seen.saw_null()
                    floats[i] = complexes[i] = NaN
                    mask[i] = 1
                else:
                    if fval != fval:
                        seen.null_ = True
                        mask[i] = 1

                    floats[i] = fval

                if maybe_int:
                    as_int = int(val)

                    if as_int in na_values:
                        mask[i] = 1
                        seen.null_ = True
                        if not allow_null_in_int:
                            seen.float_ = True
                    else:
                        seen.saw_int(as_int)

                    if as_int not in na_values:
                        if as_int < oINT64_MIN or as_int > oUINT64_MAX:
                            if seen.coerce_numeric:
                                seen.float_ = True
                            else:
                                raise ValueError("Integer out of range.")
                        else:
                            if as_int >= 0:
                                uints[i] = as_int

                            if as_int <= oINT64_MAX:
                                ints[i] = as_int

                    seen.float_ = seen.float_ or (seen.uint_ and seen.sint_)
                else:
                    seen.float_ = True
            except (TypeError, ValueError) as err:
                if not seen.coerce_numeric:
                    raise type(err)(f"{err} at position {i}")

                mask[i] = 1

                if allow_null_in_int:
                    seen.null_ = True
                else:
                    seen.saw_null()
                    floats[i] = NaN

    if seen.check_uint64_conflict():
        return (values, None)

    # This occurs since we disabled float nulls showing as null in anticipation
    # of seeing ints that were never seen. So then, we return float
    if allow_null_in_int and seen.null_ and not seen.int_ and not seen.bool_:
        seen.float_ = True

    if seen.complex_:
        return (complexes, None)
    elif seen.float_:
        if seen.null_ and convert_to_masked_nullable:
            return (floats, mask.view(np.bool_))
        return (floats, None)
    elif seen.int_:
        if seen.null_ and convert_to_masked_nullable:
            if seen.uint_:
                return (uints, mask.view(np.bool_))
            else:
                return (ints, mask.view(np.bool_))
        if seen.uint_:
            return (uints, None)
        else:
            return (ints, None)
    elif seen.bool_:
        if allow_null_in_int:
            return (bools.view(np.bool_), mask.view(np.bool_))
        return (bools.view(np.bool_), None)
    elif seen.uint_:
        return (uints, None)
    return (ints, None)


@cython.boundscheck(False)
@cython.wraparound(False)
def maybe_convert_objects(ndarray[object] objects,
                          *,
                          bint try_float=False,
                          bint safe=False,
                          bint convert_numeric=True,  # NB: different default!
                          bint convert_to_nullable_dtype=False,
                          bint convert_non_numeric=False,
                          object dtype_if_all_nat=None) -> "ArrayLike":
    """
    Type inference function-- convert object array to proper dtype

    Parameters
    ----------
    objects : ndarray[object]
        Array of object elements to convert.
    try_float : bool, default False
        If an array-like object contains only float or NaN values is
        encountered, whether to convert and return an array of float dtype.
    safe : bool, default False
        Whether to upcast numeric type (e.g. int cast to float). If set to
        True, no upcasting will be performed.
    convert_numeric : bool, default True
        Whether to convert numeric entries.
    convert_to_nullable_dtype : bool, default False
        If an array-like object contains only integer or boolean values (and NaN) is
        encountered, whether to convert and return an Boolean/IntegerArray.
    convert_non_numeric : bool, default False
        Whether to convert datetime, timedelta, period, interval types.
    dtype_if_all_nat : np.dtype, ExtensionDtype, or None, default None
        Dtype to cast to if we have all-NaT.

    Returns
    -------
    np.ndarray or ExtensionArray
        Array of converted object values to more specific dtypes if applicable.
    """
    cdef:
        Py_ssize_t i, n, itemsize_max = 0
        ndarray[float64_t] floats
        ndarray[complex128_t] complexes
        ndarray[int64_t] ints
        ndarray[uint64_t] uints
        ndarray[uint8_t] bools
        Seen seen = Seen()
        object val
        _TSObject tsobj
        float64_t fnan = NaN

    if dtype_if_all_nat is not None:
        # in practice we don't expect to ever pass dtype_if_all_nat
        #  without both convert_non_numeric, so disallow
        #  it to avoid needing to handle it below.
        if not convert_non_numeric:
            raise ValueError(
                "Cannot specify 'dtype_if_all_nat' without convert_non_numeric=True"
            )

    n = len(objects)

    floats = cnp.PyArray_EMPTY(1, objects.shape, cnp.NPY_FLOAT64, 0)
    complexes = cnp.PyArray_EMPTY(1, objects.shape, cnp.NPY_COMPLEX128, 0)
    ints = cnp.PyArray_EMPTY(1, objects.shape, cnp.NPY_INT64, 0)
    uints = cnp.PyArray_EMPTY(1, objects.shape, cnp.NPY_UINT64, 0)
    bools = cnp.PyArray_EMPTY(1, objects.shape, cnp.NPY_UINT8, 0)
    mask = np.full(n, False)

    for i in range(n):
        val = objects[i]
        if itemsize_max != -1:
            itemsize = get_itemsize(val)
            if itemsize > itemsize_max or itemsize == -1:
                itemsize_max = itemsize

        if val is None:
            seen.null_ = True
            floats[i] = complexes[i] = fnan
            mask[i] = True
        elif val is NaT:
            seen.nat_ = True
            if not convert_non_numeric:
                seen.object_ = True
                break
        elif util.is_nan(val):
            seen.nan_ = True
            mask[i] = True
            if util.is_complex_object(val):
                floats[i] = fnan
                complexes[i] = val
                seen.complex_ = True
                if not convert_numeric:
                    break
            else:
                floats[i] = complexes[i] = val
        elif util.is_bool_object(val):
            seen.bool_ = True
            bools[i] = val
            if not convert_numeric:
                break
        elif util.is_float_object(val):
            floats[i] = complexes[i] = val
            seen.float_ = True
            if not convert_numeric:
                break
        elif is_timedelta(val):
            if convert_non_numeric:
                seen.timedelta_ = True
                try:
                    convert_to_timedelta64(val, "ns")
                except OutOfBoundsTimedelta:
                    seen.object_ = True
                    break
                break
            else:
                seen.object_ = True
                break
        elif util.is_integer_object(val):
            seen.int_ = True
            floats[i] = <float64_t>val
            complexes[i] = <double complex>val
            if not seen.null_ or convert_to_nullable_dtype:
                seen.saw_int(val)

                if ((seen.uint_ and seen.sint_) or
                        val > oUINT64_MAX or val < oINT64_MIN):
                    seen.object_ = True
                    break

                if seen.uint_:
                    uints[i] = val
                elif seen.sint_:
                    ints[i] = val
                else:
                    uints[i] = val
                    ints[i] = val
            if not convert_numeric:
                break

        elif util.is_complex_object(val):
            complexes[i] = val
            seen.complex_ = True
            if not convert_numeric:
                break
        elif PyDateTime_Check(val) or util.is_datetime64_object(val):

            # if we have an tz's attached then return the objects
            if convert_non_numeric:
                if getattr(val, "tzinfo", None) is not None:
                    seen.datetimetz_ = True
                    break
                else:
                    seen.datetime_ = True
                    try:
                        tsobj = convert_to_tsobject(val, None, None, 0, 0)
                        tsobj.ensure_reso(NPY_FR_ns)
                    except OutOfBoundsDatetime:
                        seen.object_ = True
                        break
            else:
                seen.object_ = True
                break
        elif is_period_object(val):
            if convert_non_numeric:
                seen.period_ = True
                break
            else:
                seen.object_ = True
                break
        elif try_float and not isinstance(val, str):
            # this will convert Decimal objects
            try:
                floats[i] = float(val)
                complexes[i] = complex(val)
                seen.float_ = True
            except (ValueError, TypeError):
                seen.object_ = True
                break
        elif is_interval(val):
            if convert_non_numeric:
                seen.interval_ = True
                break
            else:
                seen.object_ = True
                break
        elif isinstance(val, str):
            if convert_non_numeric:
                seen.str_ = True
                break
            else:
                seen.object_ = True
                break
        else:
            seen.object_ = True
            break

    # we try to coerce datetime w/tz but must all have the same tz
    if seen.datetimetz_:
        if is_datetime_with_singletz_array(objects):
            from pandas import DatetimeIndex

            try:
                dti = DatetimeIndex(objects)
            except OutOfBoundsDatetime:
                # e.g. test_to_datetime_cache_coerce_50_lines_outofbounds
                pass
            else:
                # unbox to DatetimeArray
                return dti._data
        seen.object_ = True

    elif seen.datetime_:
        if is_datetime_or_datetime64_array(objects):
            from pandas import DatetimeIndex

            try:
                dti = DatetimeIndex(objects)
            except OutOfBoundsDatetime:
                pass
            else:
                # unbox to ndarray[datetime64[ns]]
                return dti._data._ndarray
        seen.object_ = True

    elif seen.timedelta_:
        if is_timedelta_or_timedelta64_array(objects):
            from pandas import TimedeltaIndex

            try:
                tdi = TimedeltaIndex(objects)
            except OutOfBoundsTimedelta:
                pass
            else:
                # unbox to ndarray[timedelta64[ns]]
                return tdi._data._ndarray
        seen.object_ = True

    elif seen.period_:
        if is_period_array(objects):
            from pandas import PeriodIndex
            pi = PeriodIndex(objects)

            # unbox to PeriodArray
            return pi._data
        seen.object_ = True

    elif seen.str_:
        if using_pyarrow_string_dtype() and is_string_array(objects, skipna=True):
            from pandas.core.arrays.string_ import StringDtype

            dtype = StringDtype(storage="pyarrow_numpy")
            return dtype.construct_array_type()._from_sequence(objects, dtype=dtype)

        seen.object_ = True
    elif seen.interval_:
        if is_interval_array(objects):
            from pandas import IntervalIndex
            ii = IntervalIndex(objects)

            # unbox to IntervalArray
            return ii._data

        seen.object_ = True

    elif seen.nat_:
        if not seen.object_ and not seen.numeric_ and not seen.bool_:
            # all NaT, None, or nan (at least one NaT)
            # see GH#49340 for discussion of desired behavior
            dtype = dtype_if_all_nat
            if cnp.PyArray_DescrCheck(dtype):
                # i.e. isinstance(dtype, np.dtype)
                if dtype.kind not in "mM":
                    raise ValueError(dtype)
                else:
                    res = np.empty((<object>objects).shape, dtype=dtype)
                    res[:] = NPY_NAT
                    return res
            elif dtype is not None:
                # EA, we don't expect to get here, but _could_ implement
                raise NotImplementedError(dtype)
            else:
                # we don't guess
                seen.object_ = True
        else:
            seen.object_ = True

    if not convert_numeric:
        # Note: we count "bool" as numeric here. This is because
        #  np.array(list_of_items) will convert bools just like it will numeric
        #  entries.
        return objects

    if seen.bool_:
        if seen.is_bool:
            # is_bool property rules out everything else
            return bools.view(np.bool_)
        elif convert_to_nullable_dtype and seen.is_bool_or_na:
            from pandas.core.arrays import BooleanArray
            return BooleanArray(bools.view(np.bool_), mask)
        seen.object_ = True

    if not seen.object_:
        result = None
        if not safe:
            if seen.null_ or seen.nan_:
                if seen.complex_:
                    result = complexes
                elif seen.float_:
                    result = floats
                elif seen.int_ or seen.uint_:
                    if convert_to_nullable_dtype:
                        from pandas.core.arrays import IntegerArray
                        if seen.uint_:
                            result = IntegerArray(uints, mask)
                        else:
                            result = IntegerArray(ints, mask)
                    else:
                        result = floats
                elif seen.nan_:
                    result = floats
            else:
                if seen.complex_:
                    result = complexes
                elif seen.float_:
                    result = floats
                elif seen.int_:
                    if seen.uint_:
                        result = uints
                    else:
                        result = ints

        else:
            # don't cast int to float, etc.
            if seen.null_:
                if seen.complex_:
                    if not seen.int_:
                        result = complexes
                elif seen.float_ or seen.nan_:
                    if not seen.int_:
                        result = floats
            else:
                if seen.complex_:
                    if not seen.int_:
                        result = complexes
                elif seen.float_ or seen.nan_:
                    if not seen.int_:
                        result = floats
                elif seen.int_:
                    if seen.uint_:
                        result = uints
                    else:
                        result = ints

        if result is uints or result is ints or result is floats or result is complexes:
            # cast to the largest itemsize when all values are NumPy scalars
            if itemsize_max > 0 and itemsize_max != result.dtype.itemsize:
                result = result.astype(result.dtype.kind + str(itemsize_max))
            return result
        elif result is not None:
            return result

    return objects


class _NoDefault(Enum):
    # We make this an Enum
    # 1) because it round-trips through pickle correctly (see GH#40397)
    # 2) because mypy does not understand singletons
    no_default = "NO_DEFAULT"

    def __repr__(self) -> str:
        return "<no_default>"


# Note: no_default is exported to the public API in pandas.api.extensions
no_default = _NoDefault.no_default  # Sentinel indicating the default value.
NoDefault = Literal[_NoDefault.no_default]


@cython.boundscheck(False)
@cython.wraparound(False)
def map_infer_mask(ndarray arr, object f, const uint8_t[:] mask, bint convert=True,
                   object na_value=no_default, cnp.dtype dtype=np.dtype(object)
                   ) -> np.ndarray:
    """
    Substitute for np.vectorize with pandas-friendly dtype inference.

    Parameters
    ----------
    arr : ndarray
    f : function
    mask : ndarray
        uint8 dtype ndarray indicating values not to apply `f` to.
    convert : bool, default True
        Whether to call `maybe_convert_objects` on the resulting ndarray
    na_value : Any, optional
        The result value to use for masked values. By default, the
        input value is used
    dtype : numpy.dtype
        The numpy dtype to use for the result ndarray.

    Returns
    -------
    np.ndarray
    """
    cdef:
        Py_ssize_t i, n
        ndarray result
        object val

    n = len(arr)
    result = np.empty(n, dtype=dtype)
    for i in range(n):
        if mask[i]:
            if na_value is no_default:
                val = arr[i]
            else:
                val = na_value
        else:
            val = f(arr[i])

            if cnp.PyArray_IsZeroDim(val):
                # unbox 0-dim arrays, GH#690
                val = val.item()

        result[i] = val

    if convert:
        return maybe_convert_objects(result)
    else:
        return result


@cython.boundscheck(False)
@cython.wraparound(False)
def map_infer(
    ndarray arr, object f, bint convert=True, bint ignore_na=False
) -> np.ndarray:
    """
    Substitute for np.vectorize with pandas-friendly dtype inference.

    Parameters
    ----------
    arr : ndarray
    f : function
    convert : bint
    ignore_na : bint
        If True, NA values will not have f applied

    Returns
    -------
    np.ndarray
    """
    cdef:
        Py_ssize_t i, n
        ndarray[object] result
        object val

    n = len(arr)
    result = cnp.PyArray_EMPTY(1, arr.shape, cnp.NPY_OBJECT, 0)
    for i in range(n):
        if ignore_na and checknull(arr[i]):
            result[i] = arr[i]
            continue
        val = f(arr[i])

        if cnp.PyArray_IsZeroDim(val):
            # unbox 0-dim arrays, GH#690
            val = val.item()

        result[i] = val

    if convert:
        return maybe_convert_objects(result)
    else:
        return result


def to_object_array(rows: object, min_width: int = 0) -> ndarray:
    """
    Convert a list of lists into an object array.

    Parameters
    ----------
    rows : 2-d array (N, K)
        List of lists to be converted into an array.
    min_width : int
        Minimum width of the object array. If a list
        in `rows` contains fewer than `width` elements,
        the remaining elements in the corresponding row
        will all be `NaN`.

    Returns
    -------
    np.ndarray[object, ndim=2]
    """
    cdef:
        Py_ssize_t i, j, n, k, tmp
        ndarray[object, ndim=2] result
        list row

    rows = list(rows)
    n = len(rows)

    k = min_width
    for i in range(n):
        tmp = len(rows[i])
        if tmp > k:
            k = tmp

    result = np.empty((n, k), dtype=object)

    for i in range(n):
        row = list(rows[i])

        for j in range(len(row)):
            result[i, j] = row[j]

    return result


def tuples_to_object_array(ndarray[object] tuples):
    cdef:
        Py_ssize_t i, j, n, k
        ndarray[object, ndim=2] result
        tuple tup

    n = len(tuples)
    k = len(tuples[0])
    result = np.empty((n, k), dtype=object)
    for i in range(n):
        tup = tuples[i]
        for j in range(k):
            result[i, j] = tup[j]

    return result


def to_object_array_tuples(rows: object) -> np.ndarray:
    """
    Convert a list of tuples into an object array. Any subclass of
    tuple in `rows` will be casted to tuple.

    Parameters
    ----------
    rows : 2-d array (N, K)
        List of tuples to be converted into an array.

    Returns
    -------
    np.ndarray[object, ndim=2]
    """
    cdef:
        Py_ssize_t i, j, n, k, tmp
        ndarray[object, ndim=2] result
        tuple row

    rows = list(rows)
    n = len(rows)

    k = 0
    for i in range(n):
        tmp = 1 if checknull(rows[i]) else len(rows[i])
        if tmp > k:
            k = tmp

    result = np.empty((n, k), dtype=object)

    try:
        for i in range(n):
            row = rows[i]
            for j in range(len(row)):
                result[i, j] = row[j]
    except TypeError:
        # e.g. "Expected tuple, got list"
        # upcast any subclasses to tuple
        for i in range(n):
            row = (rows[i],) if checknull(rows[i]) else tuple(rows[i])
            for j in range(len(row)):
                result[i, j] = row[j]

    return result


@cython.wraparound(False)
@cython.boundscheck(False)
def fast_multiget(dict mapping, ndarray keys, default=np.nan) -> np.ndarray:
    cdef:
        Py_ssize_t i, n = len(keys)
        object val
        ndarray[object] output = np.empty(n, dtype="O")

    if n == 0:
        # kludge, for Series
        return np.empty(0, dtype="f8")

    for i in range(n):
        val = keys[i]
        if val in mapping:
            output[i] = mapping[val]
        else:
            output[i] = default

    return maybe_convert_objects(output)


def is_bool_list(obj: list) -> bool:
    """
    Check if this list contains only bool or np.bool_ objects.

    This is appreciably faster than checking `np.array(obj).dtype == bool`

    obj1 = [True, False] * 100
    obj2 = obj1 * 100
    obj3 = obj2 * 100
    obj4 = [True, None] + obj1

    for obj in [obj1, obj2, obj3, obj4]:
        %timeit is_bool_list(obj)
        %timeit np.array(obj).dtype.kind == "b"

    340 ns ± 8.22 ns
    8.78 µs ± 253 ns

    28.8 µs ± 704 ns
    813 µs ± 17.8 µs

    3.4 ms ± 168 µs
    78.4 ms ± 1.05 ms

    48.1 ns ± 1.26 ns
    8.1 µs ± 198 ns
    """
    cdef:
        object item

    for item in obj:
        if not util.is_bool_object(item):
            return False

    # Note: we return True for empty list
    return True


cpdef ndarray eq_NA_compat(ndarray[object] arr, object key):
    """
    Check for `arr == key`, treating all values as not-equal to pd.NA.

    key is assumed to have `not isna(key)`
    """
    cdef:
        ndarray[uint8_t, cast=True] result = cnp.PyArray_EMPTY(
            arr.ndim, arr.shape, cnp.NPY_BOOL, 0
        )
        Py_ssize_t i
        object item

    for i in range(len(arr)):
        item = arr[i]
        if item is C_NA:
            result[i] = False
        else:
            result[i] = item == key

    return result


def dtypes_all_equal(list types not None) -> bool:
    """
    Faster version for:

    first = types[0]
    all(is_dtype_equal(first, t) for t in types[1:])

    And assuming all elements in the list are np.dtype/ExtensionDtype objects

    See timings at https://github.com/pandas-dev/pandas/pull/44594
    """
    first = types[0]
    for t in types[1:]:
        if t is first:
            # Fastpath can provide a nice boost for EADtypes
            continue
        try:
            if not t == first:
                return False
        except (TypeError, AttributeError):
            return False
    else:
        return True


def is_np_dtype(object dtype, str kinds=None) -> bool:
    """
    Optimized check for `isinstance(dtype, np.dtype)` with
    optional `and dtype.kind in kinds`.

    dtype = np.dtype("m8[ns]")

    In [7]: %timeit isinstance(dtype, np.dtype)
    117 ns ± 1.91 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)

    In [8]: %timeit is_np_dtype(dtype)
    64 ns ± 1.51 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)

    In [9]: %timeit is_timedelta64_dtype(dtype)
    209 ns ± 6.96 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)

    In [10]: %timeit is_np_dtype(dtype, "m")
    93.4 ns ± 1.11 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
    """
    if not cnp.PyArray_DescrCheck(dtype):
        # i.e. not isinstance(dtype, np.dtype)
        return False
    if kinds is None:
        return True
    return dtype.kind in kinds
