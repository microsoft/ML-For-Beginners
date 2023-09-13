from decimal import Decimal
import numbers
from sys import maxsize

cimport cython
from cpython.datetime cimport (
    date,
    time,
    timedelta,
)
from cython cimport Py_ssize_t

import numpy as np

cimport numpy as cnp
from numpy cimport (
    flatiter,
    float64_t,
    int64_t,
    ndarray,
    uint8_t,
)

cnp.import_array()

from pandas._libs cimport util
from pandas._libs.tslibs.nattype cimport (
    c_NaT as NaT,
    checknull_with_nat,
    is_dt64nat,
    is_td64nat,
)
from pandas._libs.tslibs.np_datetime cimport (
    get_datetime64_unit,
    get_datetime64_value,
    get_timedelta64_value,
    import_pandas_datetime,
)

import_pandas_datetime()

from pandas._libs.ops_dispatch import maybe_dispatch_ufunc_to_dunder_op

cdef:
    float64_t INF = <float64_t>np.inf
    float64_t NEGINF = -INF

    int64_t NPY_NAT = util.get_nat()

    bint is_32bit = maxsize <= 2 ** 32

    type cDecimal = Decimal  # for faster isinstance checks


cpdef bint check_na_tuples_nonequal(object left, object right):
    """
    When we have NA in one of the tuples but not the other we have to check here,
    because our regular checks fail before with ambiguous boolean value.

    Parameters
    ----------
    left: Any
    right: Any

    Returns
    -------
    True if we are dealing with tuples that have NA on one side and non NA on
    the other side.

    """
    if not isinstance(left, tuple) or not isinstance(right, tuple):
        return False

    if len(left) != len(right):
        return False

    for left_element, right_element in zip(left, right):
        if left_element is C_NA and right_element is not C_NA:
            return True
        elif right_element is C_NA and left_element is not C_NA:
            return True

    return False


cpdef bint is_matching_na(object left, object right, bint nan_matches_none=False):
    """
    Check if two scalars are both NA of matching types.

    Parameters
    ----------
    left : Any
    right : Any
    nan_matches_none : bool, default False
        For backwards compatibility, consider NaN as matching None.

    Returns
    -------
    bool
    """
    if left is None:
        if nan_matches_none and util.is_nan(right):
            return True
        return right is None
    elif left is C_NA:
        return right is C_NA
    elif left is NaT:
        return right is NaT
    elif util.is_float_object(left):
        if nan_matches_none and right is None and util.is_nan(left):
            return True
        return (
            util.is_nan(left)
            and util.is_float_object(right)
            and util.is_nan(right)
        )
    elif util.is_complex_object(left):
        return (
            util.is_nan(left)
            and util.is_complex_object(right)
            and util.is_nan(right)
        )
    elif util.is_datetime64_object(left):
        return (
            get_datetime64_value(left) == NPY_NAT
            and util.is_datetime64_object(right)
            and get_datetime64_value(right) == NPY_NAT
            and get_datetime64_unit(left) == get_datetime64_unit(right)
        )
    elif util.is_timedelta64_object(left):
        return (
            get_timedelta64_value(left) == NPY_NAT
            and util.is_timedelta64_object(right)
            and get_timedelta64_value(right) == NPY_NAT
            and get_datetime64_unit(left) == get_datetime64_unit(right)
        )
    elif is_decimal_na(left):
        return is_decimal_na(right)
    return False


cpdef bint checknull(object val, bint inf_as_na=False):
    """
    Return boolean describing of the input is NA-like, defined here as any
    of:
     - None
     - nan
     - NaT
     - np.datetime64 representation of NaT
     - np.timedelta64 representation of NaT
     - NA
     - Decimal("NaN")

    Parameters
    ----------
    val : object
    inf_as_na : bool, default False
        Whether to treat INF and -INF as NA values.

    Returns
    -------
    bool
    """
    if val is None or val is NaT or val is C_NA:
        return True
    elif util.is_float_object(val) or util.is_complex_object(val):
        if val != val:
            return True
        elif inf_as_na:
            return val == INF or val == NEGINF
        return False
    elif util.is_timedelta64_object(val):
        return get_timedelta64_value(val) == NPY_NAT
    elif util.is_datetime64_object(val):
        return get_datetime64_value(val) == NPY_NAT
    else:
        return is_decimal_na(val)


cdef bint is_decimal_na(object val):
    """
    Is this a decimal.Decimal object Decimal("NAN").
    """
    return isinstance(val, cDecimal) and val != val


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef ndarray[uint8_t] isnaobj(ndarray arr, bint inf_as_na=False):
    """
    Return boolean mask denoting which elements of a 1-D array are na-like,
    according to the criteria defined in `checknull`:
     - None
     - nan
     - NaT
     - np.datetime64 representation of NaT
     - np.timedelta64 representation of NaT
     - NA
     - Decimal("NaN")

    Parameters
    ----------
    arr : ndarray

    Returns
    -------
    result : ndarray (dtype=np.bool_)
    """
    cdef:
        Py_ssize_t i, n = arr.size
        object val
        bint is_null
        ndarray result = np.empty((<object>arr).shape, dtype=np.uint8)
        flatiter it = cnp.PyArray_IterNew(arr)
        flatiter it2 = cnp.PyArray_IterNew(result)

    for i in range(n):
        # The PyArray_GETITEM and PyArray_ITER_NEXT are faster
        #  equivalents to `val = values[i]`
        val = cnp.PyArray_GETITEM(arr, cnp.PyArray_ITER_DATA(it))
        cnp.PyArray_ITER_NEXT(it)
        is_null = checknull(val, inf_as_na=inf_as_na)
        # Dereference pointer (set value)
        (<uint8_t *>(cnp.PyArray_ITER_DATA(it2)))[0] = <uint8_t>is_null
        cnp.PyArray_ITER_NEXT(it2)
    return result.view(np.bool_)


def isposinf_scalar(val: object) -> bool:
    return util.is_float_object(val) and val == INF


def isneginf_scalar(val: object) -> bool:
    return util.is_float_object(val) and val == NEGINF


cdef bint is_null_datetime64(v):
    # determine if we have a null for a datetime (or integer versions),
    # excluding np.timedelta64('nat')
    if checknull_with_nat(v) or is_dt64nat(v):
        return True
    return False


cdef bint is_null_timedelta64(v):
    # determine if we have a null for a timedelta (or integer versions),
    # excluding np.datetime64('nat')
    if checknull_with_nat(v) or is_td64nat(v):
        return True
    return False


cdef bint checknull_with_nat_and_na(object obj):
    # See GH#32214
    return checknull_with_nat(obj) or obj is C_NA


@cython.wraparound(False)
@cython.boundscheck(False)
def is_float_nan(values: ndarray) -> ndarray:
    """
    True for elements which correspond to a float nan

    Returns
    -------
    ndarray[bool]
    """
    cdef:
        ndarray[uint8_t] result
        Py_ssize_t i, N
        object val

    N = len(values)
    result = np.zeros(N, dtype=np.uint8)

    for i in range(N):
        val = values[i]
        if util.is_nan(val):
            result[i] = True
    return result.view(bool)


@cython.wraparound(False)
@cython.boundscheck(False)
def is_numeric_na(values: ndarray) -> ndarray:
    """
    Check for NA values consistent with IntegerArray/FloatingArray.

    Similar to a vectorized is_valid_na_for_dtype restricted to numeric dtypes.

    Returns
    -------
    ndarray[bool]
    """
    cdef:
        ndarray[uint8_t] result
        Py_ssize_t i, N
        object val

    N = len(values)
    result = np.zeros(N, dtype=np.uint8)

    for i in range(N):
        val = values[i]
        if checknull(val):
            if val is None or val is C_NA or util.is_nan(val) or is_decimal_na(val):
                result[i] = True
            else:
                raise TypeError(f"'values' contains non-numeric NA {val}")
    return result.view(bool)


# -----------------------------------------------------------------------------
# Implementation of NA singleton


def _create_binary_propagating_op(name, is_divmod=False):
    is_cmp = name.strip("_") in ["eq", "ne", "le", "lt", "ge", "gt"]

    def method(self, other):
        if (other is C_NA or isinstance(other, (str, bytes))
                or isinstance(other, (numbers.Number, np.bool_))
                or util.is_array(other) and not other.shape):
            # Need the other.shape clause to handle NumPy scalars,
            # since we do a setitem on `out` below, which
            # won't work for NumPy scalars.
            if is_divmod:
                return NA, NA
            else:
                return NA

        elif util.is_array(other):
            out = np.empty(other.shape, dtype=object)
            out[:] = NA

            if is_divmod:
                return out, out.copy()
            else:
                return out

        elif is_cmp and isinstance(other, (date, time, timedelta)):
            return NA

        elif isinstance(other, date):
            if name in ["__sub__", "__rsub__"]:
                return NA

        elif isinstance(other, timedelta):
            if name in ["__sub__", "__rsub__", "__add__", "__radd__"]:
                return NA

        return NotImplemented

    method.__name__ = name
    return method


def _create_unary_propagating_op(name: str):
    def method(self):
        return NA

    method.__name__ = name
    return method


cdef class C_NAType:
    pass


class NAType(C_NAType):
    """
    NA ("not available") missing value indicator.

    .. warning::

       Experimental: the behaviour of NA can still change without warning.

    The NA singleton is a missing value indicator defined by pandas. It is
    used in certain new extension dtypes (currently the "string" dtype).

    Examples
    --------
    >>> pd.NA
    <NA>

    >>> True | pd.NA
    True

    >>> True & pd.NA
    <NA>

    >>> pd.NA != pd.NA
    <NA>

    >>> pd.NA == pd.NA
    <NA>

    >>> True | pd.NA
    True
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if NAType._instance is None:
            NAType._instance = C_NAType.__new__(cls, *args, **kwargs)
        return NAType._instance

    def __repr__(self) -> str:
        return "<NA>"

    def __format__(self, format_spec) -> str:
        try:
            return self.__repr__().__format__(format_spec)
        except ValueError:
            return self.__repr__()

    def __bool__(self):
        raise TypeError("boolean value of NA is ambiguous")

    def __hash__(self):
        # GH 30013: Ensure hash is large enough to avoid hash collisions with integers
        exponent = 31 if is_32bit else 61
        return 2 ** exponent - 1

    def __reduce__(self):
        return "NA"

    # Binary arithmetic and comparison ops -> propagate

    __add__ = _create_binary_propagating_op("__add__")
    __radd__ = _create_binary_propagating_op("__radd__")
    __sub__ = _create_binary_propagating_op("__sub__")
    __rsub__ = _create_binary_propagating_op("__rsub__")
    __mul__ = _create_binary_propagating_op("__mul__")
    __rmul__ = _create_binary_propagating_op("__rmul__")
    __matmul__ = _create_binary_propagating_op("__matmul__")
    __rmatmul__ = _create_binary_propagating_op("__rmatmul__")
    __truediv__ = _create_binary_propagating_op("__truediv__")
    __rtruediv__ = _create_binary_propagating_op("__rtruediv__")
    __floordiv__ = _create_binary_propagating_op("__floordiv__")
    __rfloordiv__ = _create_binary_propagating_op("__rfloordiv__")
    __mod__ = _create_binary_propagating_op("__mod__")
    __rmod__ = _create_binary_propagating_op("__rmod__")
    __divmod__ = _create_binary_propagating_op("__divmod__", is_divmod=True)
    __rdivmod__ = _create_binary_propagating_op("__rdivmod__", is_divmod=True)
    # __lshift__ and __rshift__ are not implemented

    __eq__ = _create_binary_propagating_op("__eq__")
    __ne__ = _create_binary_propagating_op("__ne__")
    __le__ = _create_binary_propagating_op("__le__")
    __lt__ = _create_binary_propagating_op("__lt__")
    __gt__ = _create_binary_propagating_op("__gt__")
    __ge__ = _create_binary_propagating_op("__ge__")

    # Unary ops

    __neg__ = _create_unary_propagating_op("__neg__")
    __pos__ = _create_unary_propagating_op("__pos__")
    __abs__ = _create_unary_propagating_op("__abs__")
    __invert__ = _create_unary_propagating_op("__invert__")

    # pow has special
    def __pow__(self, other):
        if other is C_NA:
            return NA
        elif isinstance(other, (numbers.Number, np.bool_)):
            if other == 0:
                # returning positive is correct for +/- 0.
                return type(other)(1)
            else:
                return NA
        elif util.is_array(other):
            return np.where(other == 0, other.dtype.type(1), NA)

        return NotImplemented

    def __rpow__(self, other):
        if other is C_NA:
            return NA
        elif isinstance(other, (numbers.Number, np.bool_)):
            if other == 1:
                return other
            else:
                return NA
        elif util.is_array(other):
            return np.where(other == 1, other, NA)
        return NotImplemented

    # Logical ops using Kleene logic

    def __and__(self, other):
        if other is False:
            return False
        elif other is True or other is C_NA:
            return NA
        return NotImplemented

    __rand__ = __and__

    def __or__(self, other):
        if other is True:
            return True
        elif other is False or other is C_NA:
            return NA
        return NotImplemented

    __ror__ = __or__

    def __xor__(self, other):
        if other is False or other is True or other is C_NA:
            return NA
        return NotImplemented

    __rxor__ = __xor__

    __array_priority__ = 1000
    _HANDLED_TYPES = (np.ndarray, numbers.Number, str, np.bool_)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        types = self._HANDLED_TYPES + (NAType,)
        for x in inputs:
            if not isinstance(x, types):
                return NotImplemented

        if method != "__call__":
            raise ValueError(f"ufunc method '{method}' not supported for NA")
        result = maybe_dispatch_ufunc_to_dunder_op(
            self, ufunc, method, *inputs, **kwargs
        )
        if result is NotImplemented:
            # For a NumPy ufunc that's not a binop, like np.logaddexp
            index = [i for i, x in enumerate(inputs) if x is NA][0]
            result = np.broadcast_arrays(*inputs)[index]
            if result.ndim == 0:
                result = result.item()
            if ufunc.nout > 1:
                result = (NA,) * ufunc.nout

        return result


C_NA = NAType()   # C-visible
NA = C_NA         # Python-visible
