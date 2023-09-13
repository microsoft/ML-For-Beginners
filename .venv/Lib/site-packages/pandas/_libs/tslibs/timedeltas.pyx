import collections
import warnings

cimport cython
from cpython.object cimport (
    Py_EQ,
    Py_GE,
    Py_GT,
    Py_LE,
    Py_LT,
    Py_NE,
    PyObject,
    PyObject_RichCompare,
)

import numpy as np

cimport numpy as cnp
from numpy cimport (
    int64_t,
    ndarray,
)

cnp.import_array()

from cpython.datetime cimport (
    PyDateTime_Check,
    PyDelta_Check,
    import_datetime,
    timedelta,
)

import_datetime()


cimport pandas._libs.tslibs.util as util
from pandas._libs.missing cimport checknull_with_nat_and_na
from pandas._libs.tslibs.base cimport ABCTimestamp
from pandas._libs.tslibs.conversion cimport (
    cast_from_unit,
    precision_from_unit,
)
from pandas._libs.tslibs.dtypes cimport (
    get_supported_reso,
    is_supported_unit,
    npy_unit_to_abbrev,
)
from pandas._libs.tslibs.nattype cimport (
    NPY_NAT,
    c_NaT as NaT,
    c_nat_strings as nat_strings,
    checknull_with_nat,
)
from pandas._libs.tslibs.np_datetime cimport (
    NPY_DATETIMEUNIT,
    NPY_FR_ns,
    cmp_dtstructs,
    cmp_scalar,
    convert_reso,
    get_datetime64_unit,
    get_timedelta64_value,
    get_unit_from_dtype,
    import_pandas_datetime,
    npy_datetimestruct,
    pandas_datetime_to_datetimestruct,
    pandas_timedelta_to_timedeltastruct,
    pandas_timedeltastruct,
)

import_pandas_datetime()

from pandas._libs.tslibs.np_datetime import (
    OutOfBoundsDatetime,
    OutOfBoundsTimedelta,
)

from pandas._libs.tslibs.offsets cimport is_tick_object
from pandas._libs.tslibs.util cimport (
    is_array,
    is_datetime64_object,
    is_float_object,
    is_integer_object,
    is_timedelta64_object,
)

from pandas._libs.tslibs.fields import (
    RoundTo,
    round_nsint64,
)

# ----------------------------------------------------------------------
# Constants

# components named tuple
Components = collections.namedtuple(
    "Components",
    [
        "days",
        "hours",
        "minutes",
        "seconds",
        "milliseconds",
        "microseconds",
        "nanoseconds",
    ],
)

# This should be kept consistent with UnitChoices in pandas/_libs/tslibs/timedeltas.pyi
cdef dict timedelta_abbrevs = {
    "Y": "Y",
    "y": "Y",
    "M": "M",
    "W": "W",
    "w": "W",
    "D": "D",
    "d": "D",
    "days": "D",
    "day": "D",
    "hours": "h",
    "hour": "h",
    "hr": "h",
    "h": "h",
    "m": "m",
    "minute": "m",
    "min": "m",
    "minutes": "m",
    "t": "m",
    "s": "s",
    "seconds": "s",
    "sec": "s",
    "second": "s",
    "ms": "ms",
    "milliseconds": "ms",
    "millisecond": "ms",
    "milli": "ms",
    "millis": "ms",
    "l": "ms",
    "us": "us",
    "microseconds": "us",
    "microsecond": "us",
    "µs": "us",
    "micro": "us",
    "micros": "us",
    "u": "us",
    "ns": "ns",
    "nanoseconds": "ns",
    "nano": "ns",
    "nanos": "ns",
    "nanosecond": "ns",
    "n": "ns",
}

_no_input = object()

# ----------------------------------------------------------------------
# API


@cython.boundscheck(False)
@cython.wraparound(False)
def ints_to_pytimedelta(ndarray m8values, box=False):
    """
    convert an i8 repr to an ndarray of timedelta or Timedelta (if box ==
    True)

    Parameters
    ----------
    arr : ndarray[timedelta64]
    box : bool, default False

    Returns
    -------
    result : ndarray[object]
        array of Timedelta or timedeltas objects
    """
    cdef:
        NPY_DATETIMEUNIT reso = get_unit_from_dtype(m8values.dtype)
        Py_ssize_t i, n = m8values.size
        int64_t value
        object res_val

        # Note that `result` (and thus `result_flat`) is C-order and
        #  `it` iterates C-order as well, so the iteration matches
        #  See discussion at
        #  github.com/pandas-dev/pandas/pull/46886#discussion_r860261305
        ndarray result = cnp.PyArray_EMPTY(
            m8values.ndim, m8values.shape, cnp.NPY_OBJECT, 0
        )
        object[::1] res_flat = result.ravel()     # should NOT be a copy

        ndarray arr = m8values.view("i8")
        cnp.flatiter it = cnp.PyArray_IterNew(arr)

    for i in range(n):
        # Analogous to: value = arr[i]
        value = (<int64_t*>cnp.PyArray_ITER_DATA(it))[0]

        if value == NPY_NAT:
            res_val = <object>NaT
        else:
            if box:
                res_val = _timedelta_from_value_and_reso(Timedelta, value, reso=reso)
            elif reso == NPY_DATETIMEUNIT.NPY_FR_ns:
                res_val = timedelta(microseconds=int(value) / 1000)
            elif reso == NPY_DATETIMEUNIT.NPY_FR_us:
                res_val = timedelta(microseconds=value)
            elif reso == NPY_DATETIMEUNIT.NPY_FR_ms:
                res_val = timedelta(milliseconds=value)
            elif reso == NPY_DATETIMEUNIT.NPY_FR_s:
                res_val = timedelta(seconds=value)
            elif reso == NPY_DATETIMEUNIT.NPY_FR_m:
                res_val = timedelta(minutes=value)
            elif reso == NPY_DATETIMEUNIT.NPY_FR_h:
                res_val = timedelta(hours=value)
            elif reso == NPY_DATETIMEUNIT.NPY_FR_D:
                res_val = timedelta(days=value)
            elif reso == NPY_DATETIMEUNIT.NPY_FR_W:
                res_val = timedelta(weeks=value)
            else:
                # Month, Year, NPY_FR_GENERIC, pico, femto, atto
                raise NotImplementedError(reso)

        # Note: we can index result directly instead of using PyArray_MultiIter_DATA
        #  like we do for the other functions because result is known C-contiguous
        #  and is the first argument to PyArray_MultiIterNew2.  The usual pattern
        #  does not seem to work with object dtype.
        #  See discussion at
        #  github.com/pandas-dev/pandas/pull/46886#discussion_r860261305
        res_flat[i] = res_val

        cnp.PyArray_ITER_NEXT(it)

    return result


# ----------------------------------------------------------------------


cpdef int64_t delta_to_nanoseconds(
    delta,
    NPY_DATETIMEUNIT reso=NPY_FR_ns,
    bint round_ok=True,
) except? -1:
    # Note: this will raise on timedelta64 with Y or M unit

    cdef:
        NPY_DATETIMEUNIT in_reso
        int64_t n

    if is_tick_object(delta):
        n = delta.n
        in_reso = delta._creso

    elif isinstance(delta, _Timedelta):
        n = delta._value
        in_reso = delta._creso

    elif is_timedelta64_object(delta):
        in_reso = get_datetime64_unit(delta)
        if in_reso == NPY_DATETIMEUNIT.NPY_FR_Y or in_reso == NPY_DATETIMEUNIT.NPY_FR_M:
            raise ValueError(
                "delta_to_nanoseconds does not support Y or M units, "
                "as their duration in nanoseconds is ambiguous."
            )
        n = get_timedelta64_value(delta)

    elif PyDelta_Check(delta):
        in_reso = NPY_DATETIMEUNIT.NPY_FR_us
        try:
            n = (
                delta.days * 24 * 3600 * 1_000_000
                + delta.seconds * 1_000_000
                + delta.microseconds
                )
        except OverflowError as err:
            raise OutOfBoundsTimedelta(*err.args) from err

    else:
        raise TypeError(type(delta))

    try:
        return convert_reso(n, in_reso, reso, round_ok=round_ok)
    except (OutOfBoundsDatetime, OverflowError) as err:
        # Catch OutOfBoundsDatetime bc convert_reso can call check_dts_bounds
        #  for Y/M-resolution cases
        unit_str = npy_unit_to_abbrev(reso)
        raise OutOfBoundsTimedelta(
            f"Cannot cast {str(delta)} to unit={unit_str} without overflow."
        ) from err


@cython.overflowcheck(True)
cdef object ensure_td64ns(object ts):
    """
    Overflow-safe implementation of td64.astype("m8[ns]")

    Parameters
    ----------
    ts : np.timedelta64

    Returns
    -------
    np.timedelta64[ns]
    """
    cdef:
        NPY_DATETIMEUNIT td64_unit
        int64_t td64_value, mult
        str unitstr

    td64_unit = get_datetime64_unit(ts)
    if (
        td64_unit != NPY_DATETIMEUNIT.NPY_FR_ns
        and td64_unit != NPY_DATETIMEUNIT.NPY_FR_GENERIC
    ):
        unitstr = npy_unit_to_abbrev(td64_unit)

        td64_value = get_timedelta64_value(ts)

        mult = precision_from_unit(unitstr)[0]
        try:
            # NB: cython#1381 this cannot be *=
            td64_value = td64_value * mult
        except OverflowError as err:
            raise OutOfBoundsTimedelta(ts) from err

        return np.timedelta64(td64_value, "ns")

    return ts


cdef convert_to_timedelta64(object ts, str unit):
    """
    Convert an incoming object to a timedelta64 if possible.
    Before calling, unit must be standardized to avoid repeated unit conversion

    Handle these types of objects:
        - timedelta/Timedelta
        - timedelta64
        - an offset
        - np.int64 (with unit providing a possible modifier)
        - None/NaT

    Return an ns based int64
    """
    # Caller is responsible for checking unit not in ["Y", "y", "M"]
    if checknull_with_nat_and_na(ts):
        return np.timedelta64(NPY_NAT, "ns")
    elif isinstance(ts, _Timedelta):
        # already in the proper format
        if ts._creso != NPY_FR_ns:
            ts = ts.as_unit("ns").asm8
        else:
            ts = np.timedelta64(ts._value, "ns")
    elif is_timedelta64_object(ts):
        ts = ensure_td64ns(ts)
    elif is_integer_object(ts):
        if ts == NPY_NAT:
            return np.timedelta64(NPY_NAT, "ns")
        else:
            ts = _maybe_cast_from_unit(ts, unit)
    elif is_float_object(ts):
        ts = _maybe_cast_from_unit(ts, unit)
    elif isinstance(ts, str):
        if (len(ts) > 0 and ts[0] == "P") or (len(ts) > 1 and ts[:2] == "-P"):
            ts = parse_iso_format_string(ts)
        else:
            ts = parse_timedelta_string(ts)
        ts = np.timedelta64(ts, "ns")
    elif is_tick_object(ts):
        ts = np.timedelta64(ts.nanos, "ns")

    if PyDelta_Check(ts):
        ts = np.timedelta64(delta_to_nanoseconds(ts), "ns")
    elif not is_timedelta64_object(ts):
        raise TypeError(f"Invalid type for timedelta scalar: {type(ts)}")
    return ts.astype("timedelta64[ns]")


cdef _maybe_cast_from_unit(ts, str unit):
    # caller is responsible for checking
    #  assert unit not in ["Y", "y", "M"]
    try:
        ts = cast_from_unit(ts, unit)
    except OutOfBoundsDatetime as err:
        raise OutOfBoundsTimedelta(
            f"Cannot cast {ts} from {unit} to 'ns' without overflow."
        ) from err

    ts = np.timedelta64(ts, "ns")
    return ts


@cython.boundscheck(False)
@cython.wraparound(False)
def array_to_timedelta64(
    ndarray values, str unit=None, str errors="raise"
) -> ndarray:
    # values is object-dtype, may be 2D
    """
    Convert an ndarray to an array of timedeltas. If errors == 'coerce',
    coerce non-convertible objects to NaT. Otherwise, raise.

    Returns
    -------
    np.ndarray[timedelta64ns]
    """
    # Caller is responsible for checking
    assert unit not in ["Y", "y", "M"]

    cdef:
        Py_ssize_t i, n = values.size
        ndarray result = np.empty((<object>values).shape, dtype="m8[ns]")
        object item
        int64_t ival
        cnp.broadcast mi = cnp.PyArray_MultiIterNew2(result, values)
        cnp.flatiter it

    if values.descr.type_num != cnp.NPY_OBJECT:
        # raise here otherwise we segfault below
        raise TypeError("array_to_timedelta64 'values' must have object dtype")

    if errors not in {"ignore", "raise", "coerce"}:
        raise ValueError("errors must be one of {'ignore', 'raise', or 'coerce'}")

    if unit is not None and errors != "coerce":
        it = cnp.PyArray_IterNew(values)
        for i in range(n):
            # Analogous to: item = values[i]
            item = cnp.PyArray_GETITEM(values, cnp.PyArray_ITER_DATA(it))
            if isinstance(item, str):
                raise ValueError(
                    "unit must not be specified if the input contains a str"
                )
            cnp.PyArray_ITER_NEXT(it)

    # Usually, we have all strings. If so, we hit the fast path.
    # If this path fails, we try conversion a different way, and
    # this is where all of the error handling will take place.
    try:
        for i in range(n):
            # Analogous to: item = values[i]
            item = <object>(<PyObject**>cnp.PyArray_MultiIter_DATA(mi, 1))[0]

            ival = _item_to_timedelta64_fastpath(item)

            # Analogous to: iresult[i] = ival
            (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 0))[0] = ival

            cnp.PyArray_MultiIter_NEXT(mi)

    except (TypeError, ValueError):
        cnp.PyArray_MultiIter_RESET(mi)

        parsed_unit = parse_timedelta_unit(unit or "ns")
        for i in range(n):
            item = <object>(<PyObject**>cnp.PyArray_MultiIter_DATA(mi, 1))[0]

            ival = _item_to_timedelta64(item, parsed_unit, errors)

            (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 0))[0] = ival

            cnp.PyArray_MultiIter_NEXT(mi)

    return result


cdef int64_t _item_to_timedelta64_fastpath(object item) except? -1:
    """
    See array_to_timedelta64.
    """
    if item is NaT:
        # we allow this check in the fast-path because NaT is a C-object
        #  so this is an inexpensive check
        return NPY_NAT
    else:
        return parse_timedelta_string(item)


cdef int64_t _item_to_timedelta64(
    object item,
    str parsed_unit,
    str errors
) except? -1:
    """
    See array_to_timedelta64.
    """
    try:
        return get_timedelta64_value(convert_to_timedelta64(item, parsed_unit))
    except ValueError as err:
        if errors == "coerce":
            return NPY_NAT
        elif "unit abbreviation w/o a number" in str(err):
            # re-raise with more pertinent message
            msg = f"Could not convert '{item}' to NumPy timedelta"
            raise ValueError(msg) from err
        else:
            raise


@cython.cpow(True)
cdef int64_t parse_timedelta_string(str ts) except? -1:
    """
    Parse a regular format timedelta string. Return an int64_t (in ns)
    or raise a ValueError on an invalid parse.
    """

    cdef:
        unicode c
        bint neg = 0, have_dot = 0, have_value = 0, have_hhmmss = 0
        object current_unit = None
        int64_t result = 0, m = 0, r
        list number = [], frac = [], unit = []

    # neg : tracks if we have a leading negative for the value
    # have_dot : tracks if we are processing a dot (either post hhmmss or
    #            inside an expression)
    # have_value : track if we have at least 1 leading unit
    # have_hhmmss : tracks if we have a regular format hh:mm:ss

    if len(ts) == 0 or ts in nat_strings:
        return NPY_NAT

    for c in ts:

        # skip whitespace / commas
        if c == " " or c == ",":
            pass

        # positive signs are ignored
        elif c == "+":
            pass

        # neg
        elif c == "-":

            if neg or have_value or have_hhmmss:
                raise ValueError("only leading negative signs are allowed")

            neg = 1

        # number (ascii codes)
        elif ord(c) >= 48 and ord(c) <= 57:

            if have_dot:

                # we found a dot, but now its just a fraction
                if len(unit):
                    number.append(c)
                    have_dot = 0
                else:
                    frac.append(c)

            elif not len(unit):
                number.append(c)

            else:
                r = timedelta_from_spec(number, frac, unit)
                unit, number, frac = [], [c], []

                result += timedelta_as_neg(r, neg)

        # hh:mm:ss.
        elif c == ":":

            # we flip this off if we have a leading value
            if have_value:
                neg = 0

            # we are in the pattern hh:mm:ss pattern
            if len(number):
                if current_unit is None:
                    current_unit = "h"
                    m = 1000000000 * 3600
                elif current_unit == "h":
                    current_unit = "m"
                    m = 1000000000 * 60
                elif current_unit == "m":
                    current_unit = "s"
                    m = 1000000000
                r = <int64_t>int("".join(number)) * m
                result += timedelta_as_neg(r, neg)
                have_hhmmss = 1
            else:
                raise ValueError(f"expecting hh:mm:ss format, received: {ts}")

            unit, number = [], []

        # after the decimal point
        elif c == ".":

            if len(number) and current_unit is not None:

                # by definition we had something like
                # so we need to evaluate the final field from a
                # hh:mm:ss (so current_unit is 'm')
                if current_unit != "m":
                    raise ValueError("expected hh:mm:ss format before .")
                m = 1000000000
                r = <int64_t>int("".join(number)) * m
                result += timedelta_as_neg(r, neg)
                have_value = 1
                unit, number, frac = [], [], []

            have_dot = 1

        # unit
        else:
            unit.append(c)
            have_value = 1
            have_dot = 0

    # we had a dot, but we have a fractional
    # value since we have an unit
    if have_dot and len(unit):
        r = timedelta_from_spec(number, frac, unit)
        result += timedelta_as_neg(r, neg)

    # we have a dot as part of a regular format
    # e.g. hh:mm:ss.fffffff
    elif have_dot:

        if ((len(number) or len(frac)) and not len(unit)
                and current_unit is None):
            raise ValueError("no units specified")

        if len(frac) > 0 and len(frac) <= 3:
            m = 10**(3 -len(frac)) * 1000 * 1000
        elif len(frac) > 3 and len(frac) <= 6:
            m = 10**(6 -len(frac)) * 1000
        elif len(frac) > 6 and len(frac) <= 9:
            m = 10**(9 -len(frac))
        else:
            m = 1
            frac = frac[:9]
        r = <int64_t>int("".join(frac)) * m
        result += timedelta_as_neg(r, neg)

    # we have a regular format
    # we must have seconds at this point (hence the unit is still 'm')
    elif current_unit is not None:
        if current_unit != "m":
            raise ValueError("expected hh:mm:ss format")
        m = 1000000000
        r = <int64_t>int("".join(number)) * m
        result += timedelta_as_neg(r, neg)

    # we have a last abbreviation
    elif len(unit):
        if len(number):
            r = timedelta_from_spec(number, frac, unit)
            result += timedelta_as_neg(r, neg)
        else:
            raise ValueError("unit abbreviation w/o a number")

    # we only have symbols and no numbers
    elif len(number) == 0:
        raise ValueError("symbols w/o a number")

    # treat as nanoseconds
    # but only if we don't have anything else
    else:
        if have_value:
            raise ValueError("have leftover units")
        if len(number):
            r = timedelta_from_spec(number, frac, "ns")
            result += timedelta_as_neg(r, neg)

    return result


cdef int64_t timedelta_as_neg(int64_t value, bint neg):
    """

    Parameters
    ----------
    value : int64_t of the timedelta value
    neg : bool if the a negative value
    """
    if neg:
        return -value
    return value


cdef timedelta_from_spec(object number, object frac, object unit):
    """

    Parameters
    ----------
    number : a list of number digits
    frac : a list of frac digits
    unit : a list of unit characters
    """
    cdef:
        str n

    unit = "".join(unit)
    if unit in ["M", "Y", "y"]:
        raise ValueError(
            "Units 'M', 'Y' and 'y' do not represent unambiguous timedelta "
            "values and are not supported."
        )

    unit = parse_timedelta_unit(unit)

    n = "".join(number) + "." + "".join(frac)
    return cast_from_unit(float(n), unit)


cpdef inline str parse_timedelta_unit(str unit):
    """
    Parameters
    ----------
    unit : str or None

    Returns
    -------
    str
        Canonical unit string.

    Raises
    ------
    ValueError : on non-parseable input
    """
    if unit is None:
        return "ns"
    elif unit == "M":
        return unit
    try:
        return timedelta_abbrevs[unit.lower()]
    except KeyError:
        raise ValueError(f"invalid unit abbreviation: {unit}")

# ----------------------------------------------------------------------
# Timedelta ops utilities

cdef bint _validate_ops_compat(other):
    # return True if we are compat with operating
    if checknull_with_nat(other):
        return True
    elif is_any_td_scalar(other):
        return True
    elif isinstance(other, str):
        return True
    return False


def _op_unary_method(func, name):
    def f(self):
        new_value = func(self._value)
        return _timedelta_from_value_and_reso(Timedelta, new_value, self._creso)
    f.__name__ = name
    return f


def _binary_op_method_timedeltalike(op, name):
    # define a binary operation that only works if the other argument is
    # timedelta like or an array of timedeltalike
    def f(self, other):
        if other is NaT:
            return NaT

        elif is_datetime64_object(other) or (
            PyDateTime_Check(other) and not isinstance(other, ABCTimestamp)
        ):
            # this case is for a datetime object that is specifically
            # *not* a Timestamp, as the Timestamp case will be
            # handled after `_validate_ops_compat` returns False below
            from pandas._libs.tslibs.timestamps import Timestamp
            return op(self, Timestamp(other))
            # We are implicitly requiring the canonical behavior to be
            # defined by Timestamp methods.

        elif is_array(other):
            if other.ndim == 0:
                # see also: item_from_zerodim
                item = cnp.PyArray_ToScalar(cnp.PyArray_DATA(other), other)
                return f(self, item)

            elif other.dtype.kind in "mM":
                return op(self.to_timedelta64(), other)
            elif other.dtype.kind == "O":
                return np.array([op(self, x) for x in other])
            else:
                return NotImplemented

        elif not _validate_ops_compat(other):
            # Includes any of our non-cython classes
            return NotImplemented

        try:
            other = Timedelta(other)
        except ValueError:
            # failed to parse as timedelta
            return NotImplemented

        if other is NaT:
            # e.g. if original other was timedelta64('NaT')
            return NaT

        # Matching numpy, we cast to the higher resolution. Unlike numpy,
        #  we raise instead of silently overflowing during this casting.
        if self._creso < other._creso:
            self = (<_Timedelta>self)._as_creso(other._creso, round_ok=True)
        elif self._creso > other._creso:
            other = (<_Timedelta>other)._as_creso(self._creso, round_ok=True)

        res = op(self._value, other._value)
        if res == NPY_NAT:
            # e.g. test_implementation_limits
            # TODO: more generally could do an overflowcheck in op?
            return NaT

        return _timedelta_from_value_and_reso(Timedelta, res, reso=self._creso)

    f.__name__ = name
    return f


# ----------------------------------------------------------------------
# Timedelta Construction

cdef int64_t parse_iso_format_string(str ts) except? -1:
    """
    Extracts and cleanses the appropriate values from a match object with
    groups for each component of an ISO 8601 duration

    Parameters
    ----------
    ts: str
        ISO 8601 Duration formatted string

    Returns
    -------
    ns: int64_t
        Precision in nanoseconds of matched ISO 8601 duration

    Raises
    ------
    ValueError
        If ``ts`` cannot be parsed
    """

    cdef:
        unicode c
        int64_t result = 0, r
        int p = 0, sign = 1
        object dec_unit = "ms", err_msg
        bint have_dot = 0, have_value = 0, neg = 0
        list number = [], unit = []

    err_msg = f"Invalid ISO 8601 Duration format - {ts}"

    if ts[0] == "-":
        sign = -1
        ts = ts[1:]

    for c in ts:
        # number (ascii codes)
        if 48 <= ord(c) <= 57:

            have_value = 1
            if have_dot:
                if p == 3 and dec_unit != "ns":
                    unit.append(dec_unit)
                    if dec_unit == "ms":
                        dec_unit = "us"
                    elif dec_unit == "us":
                        dec_unit = "ns"
                    p = 0
                p += 1

            if not len(unit):
                number.append(c)
            else:
                r = timedelta_from_spec(number, "0", unit)
                result += timedelta_as_neg(r, neg)

                neg = 0
                unit, number = [], [c]
        else:
            if c == "P" or c == "T":
                pass  # ignore marking characters P and T
            elif c == "-":
                if neg or have_value:
                    raise ValueError(err_msg)
                else:
                    neg = 1
            elif c == "+":
                pass
            elif c in ["W", "D", "H", "M"]:
                if c in ["H", "M"] and len(number) > 2:
                    raise ValueError(err_msg)
                if c == "M":
                    c = "min"
                unit.append(c)
                r = timedelta_from_spec(number, "0", unit)
                result += timedelta_as_neg(r, neg)

                neg = 0
                unit, number = [], []
            elif c == ".":
                # append any seconds
                if len(number):
                    r = timedelta_from_spec(number, "0", "S")
                    result += timedelta_as_neg(r, neg)
                    unit, number = [], []
                have_dot = 1
            elif c == "S":
                if have_dot:  # ms, us, or ns
                    if not len(number) or p > 3:
                        raise ValueError(err_msg)
                    # pad to 3 digits as required
                    pad = 3 - p
                    while pad > 0:
                        number.append("0")
                        pad -= 1

                    r = timedelta_from_spec(number, "0", dec_unit)
                    result += timedelta_as_neg(r, neg)
                else:  # seconds
                    r = timedelta_from_spec(number, "0", "S")
                    result += timedelta_as_neg(r, neg)
            else:
                raise ValueError(err_msg)

    if not have_value:
        # Received string only - never parsed any values
        raise ValueError(err_msg)

    return sign*result


cdef _to_py_int_float(v):
    # Note: This used to be defined inside Timedelta.__new__
    # but cython will not allow `cdef` functions to be defined dynamically.
    if is_integer_object(v):
        return int(v)
    elif is_float_object(v):
        return float(v)
    raise TypeError(f"Invalid type {type(v)}. Must be int or float.")


def _timedelta_unpickle(value, reso):
    return _timedelta_from_value_and_reso(Timedelta, value, reso)


cdef _timedelta_from_value_and_reso(cls, int64_t value, NPY_DATETIMEUNIT reso):
    # Could make this a classmethod if/when cython supports cdef classmethods
    cdef:
        _Timedelta td_base

    assert value != NPY_NAT
    # For millisecond and second resos, we cannot actually pass int(value) because
    #  many cases would fall outside of the pytimedelta implementation bounds.
    #  We pass 0 instead, and override seconds, microseconds, days.
    #  In principle we could pass 0 for ns and us too.
    if reso == NPY_FR_ns:
        td_base = _Timedelta.__new__(cls, microseconds=int(value) // 1000)
    elif reso == NPY_DATETIMEUNIT.NPY_FR_us:
        td_base = _Timedelta.__new__(cls, microseconds=int(value))
    elif reso == NPY_DATETIMEUNIT.NPY_FR_ms:
        td_base = _Timedelta.__new__(cls, milliseconds=0)
    elif reso == NPY_DATETIMEUNIT.NPY_FR_s:
        td_base = _Timedelta.__new__(cls, seconds=0)
    # Other resolutions are disabled but could potentially be implemented here:
    # elif reso == NPY_DATETIMEUNIT.NPY_FR_m:
    #    td_base = _Timedelta.__new__(Timedelta, minutes=int(value))
    # elif reso == NPY_DATETIMEUNIT.NPY_FR_h:
    #    td_base = _Timedelta.__new__(Timedelta, hours=int(value))
    # elif reso == NPY_DATETIMEUNIT.NPY_FR_D:
    #    td_base = _Timedelta.__new__(Timedelta, days=int(value))
    else:
        raise NotImplementedError(
            "Only resolutions 's', 'ms', 'us', 'ns' are supported."
        )

    td_base._value = value
    td_base._is_populated = 0
    td_base._creso = reso
    return td_base


class MinMaxReso:
    """
    We need to define min/max/resolution on both the Timedelta _instance_
    and Timedelta class.  On an instance, these depend on the object's _reso.
    On the class, we default to the values we would get with nanosecond _reso.
    """
    def __init__(self, name):
        self._name = name

    def __get__(self, obj, type=None):
        if self._name == "min":
            val = np.iinfo(np.int64).min + 1
        elif self._name == "max":
            val = np.iinfo(np.int64).max
        else:
            assert self._name == "resolution"
            val = 1

        if obj is None:
            # i.e. this is on the class, default to nanos
            return Timedelta(val)
        else:
            return Timedelta._from_value_and_reso(val, obj._creso)

    def __set__(self, obj, value):
        raise AttributeError(f"{self._name} is not settable.")


# Similar to Timestamp/datetime, this is a construction requirement for
# timedeltas that we need to do object instantiation in python. This will
# serve as a C extension type that shadows the Python class, where we do any
# heavy lifting.
cdef class _Timedelta(timedelta):
    # cdef readonly:
    #    int64_t value      # nanoseconds
    #    bint _is_populated  # are my components populated
    #    int64_t _d, _h, _m, _s, _ms, _us, _ns
    #    NPY_DATETIMEUNIT _reso

    # higher than np.ndarray and np.matrix
    __array_priority__ = 100
    min = MinMaxReso("min")
    max = MinMaxReso("max")
    resolution = MinMaxReso("resolution")

    @property
    def value(self):
        try:
            return convert_reso(self._value, self._creso, NPY_FR_ns, False)
        except OverflowError:
            raise OverflowError(
                "Cannot convert Timedelta to nanoseconds without overflow. "
                "Use `.asm8.view('i8')` to cast represent Timedelta in its own "
                f"unit (here, {self.unit})."
            )

    @property
    def _unit(self) -> str:
        """
        The abbreviation associated with self._creso.
        """
        return npy_unit_to_abbrev(self._creso)

    @property
    def days(self) -> int:  # TODO(cython3): make cdef property
        """
        Returns the days of the timedelta.

        Returns
        -------
        int

        Examples
        --------
        >>> td = pd.Timedelta(1, "d")
        >>> td.days
        1

        >>> td = pd.Timedelta('4 min 3 us 42 ns')
        >>> td.days
        0
        """
        # NB: using the python C-API PyDateTime_DELTA_GET_DAYS will fail
        #  (or be incorrect)
        self._ensure_components()
        return self._d

    @property
    def seconds(self) -> int:  # TODO(cython3): make cdef property
        """
        Return the total hours, minutes, and seconds of the timedelta as seconds.

        Timedelta.seconds = hours * 3600 + minutes * 60 + seconds.

        Returns
        -------
        int
            Number of seconds.

        See Also
        --------
        Timedelta.components : Return all attributes with assigned values
            (i.e. days, hours, minutes, seconds, milliseconds, microseconds,
            nanoseconds).
        Timedelta.total_seconds : Express the Timedelta as total number of seconds.

        Examples
        --------
        **Using string input**

        >>> td = pd.Timedelta('1 days 2 min 3 us 42 ns')
        >>> td.seconds
        120

        **Using integer input**

        >>> td = pd.Timedelta(42, unit='s')
        >>> td.seconds
        42
        """
        # NB: using the python C-API PyDateTime_DELTA_GET_SECONDS will fail
        #  (or be incorrect)
        self._ensure_components()
        return self._h * 3600 + self._m * 60 + self._s

    @property
    def microseconds(self) -> int:  # TODO(cython3): make cdef property
        # NB: using the python C-API PyDateTime_DELTA_GET_MICROSECONDS will fail
        #  (or be incorrect)
        self._ensure_components()
        return self._ms * 1000 + self._us

    def total_seconds(self) -> float:
        """
        Total seconds in the duration.

        Examples
        --------
        >>> td = pd.Timedelta('1min')
        >>> td
        Timedelta('0 days 00:01:00')
        >>> td.total_seconds()
        60.0
        """
        # We need to override bc we overrode days/seconds/microseconds
        # TODO: add nanos/1e9?
        return self.days * 24 * 3600 + self.seconds + self.microseconds / 1_000_000

    @property
    def unit(self) -> str:
        return npy_unit_to_abbrev(self._creso)

    def __hash__(_Timedelta self):
        if self._has_ns():
            # Note: this does *not* satisfy the invariance
            #  td1 == td2 \\Rightarrow hash(td1) == hash(td2)
            #  if td1 and td2 have different _resos. timedelta64 also has this
            #  non-invariant behavior.
            #  see GH#44504
            return hash(self._value)
        elif self._is_in_pytimedelta_bounds() and (
            self._creso == NPY_FR_ns or self._creso == NPY_DATETIMEUNIT.NPY_FR_us
        ):
            # If we can defer to timedelta.__hash__, do so, as that
            #  ensures the hash is invariant to our _reso.
            # We can only defer for ns and us, as for these two resos we
            #  call _Timedelta.__new__ with the correct input in
            #  _timedelta_from_value_and_reso; so timedelta.__hash__
            #  will be correct
            return timedelta.__hash__(self)
        else:
            # We want to ensure that two equivalent Timedelta objects
            #  have the same hash.  So we try downcasting to the next-lowest
            #  resolution.
            try:
                obj = (<_Timedelta>self)._as_creso(<NPY_DATETIMEUNIT>(self._creso + 1))
            except OutOfBoundsTimedelta:
                # Doesn't fit, so we're off the hook
                return hash(self._value)
            else:
                return hash(obj)

    def __richcmp__(_Timedelta self, object other, int op):
        cdef:
            _Timedelta ots

        if isinstance(other, _Timedelta):
            ots = other
        elif is_any_td_scalar(other):
            try:
                ots = Timedelta(other)
            except OutOfBoundsTimedelta as err:
                # GH#49021 pytimedelta.max overflows
                if not PyDelta_Check(other):
                    # TODO: handle this case
                    raise
                ltup = (self.days, self.seconds, self.microseconds, self.nanoseconds)
                rtup = (other.days, other.seconds, other.microseconds, 0)
                if op == Py_EQ:
                    return ltup == rtup
                elif op == Py_NE:
                    return ltup != rtup
                elif op == Py_LT:
                    return ltup < rtup
                elif op == Py_LE:
                    return ltup <= rtup
                elif op == Py_GT:
                    return ltup > rtup
                elif op == Py_GE:
                    return ltup >= rtup

        elif other is NaT:
            return op == Py_NE

        elif util.is_array(other):
            if other.dtype.kind == "m":
                return PyObject_RichCompare(self.asm8, other, op)
            elif other.dtype.kind == "O":
                # operate element-wise
                return np.array(
                    [PyObject_RichCompare(self, x, op) for x in other],
                    dtype=bool,
                )
            if op == Py_EQ:
                return np.zeros(other.shape, dtype=bool)
            elif op == Py_NE:
                return np.ones(other.shape, dtype=bool)
            return NotImplemented  # let other raise TypeError

        else:
            return NotImplemented

        if self._creso == ots._creso:
            return cmp_scalar(self._value, ots._value, op)
        return self._compare_mismatched_resos(ots, op)

    # TODO: re-use/share with Timestamp
    cdef bint _compare_mismatched_resos(self, _Timedelta other, op):
        # Can't just dispatch to numpy as they silently overflow and get it wrong
        cdef:
            npy_datetimestruct dts_self
            npy_datetimestruct dts_other

        # dispatch to the datetimestruct utils instead of writing new ones!
        pandas_datetime_to_datetimestruct(self._value, self._creso, &dts_self)
        pandas_datetime_to_datetimestruct(other._value, other._creso, &dts_other)
        return cmp_dtstructs(&dts_self,  &dts_other, op)

    cdef bint _has_ns(self):
        if self._creso == NPY_FR_ns:
            return self._value % 1000 != 0
        elif self._creso < NPY_FR_ns:
            # i.e. seconds, millisecond, microsecond
            return False
        else:
            raise NotImplementedError(self._creso)

    cdef bint _is_in_pytimedelta_bounds(self):
        """
        Check if we are within the bounds of datetime.timedelta.
        """
        self._ensure_components()
        return -999999999 <= self._d and self._d <= 999999999

    cdef _ensure_components(_Timedelta self):
        """
        compute the components
        """
        if self._is_populated:
            return

        cdef:
            pandas_timedeltastruct tds

        pandas_timedelta_to_timedeltastruct(self._value, self._creso, &tds)
        self._d = tds.days
        self._h = tds.hrs
        self._m = tds.min
        self._s = tds.sec
        self._ms = tds.ms
        self._us = tds.us
        self._ns = tds.ns
        self._seconds = tds.seconds
        self._microseconds = tds.microseconds

        self._is_populated = 1

    cpdef timedelta to_pytimedelta(_Timedelta self):
        """
        Convert a pandas Timedelta object into a python ``datetime.timedelta`` object.

        Timedelta objects are internally saved as numpy datetime64[ns] dtype.
        Use to_pytimedelta() to convert to object dtype.

        Returns
        -------
        datetime.timedelta or numpy.array of datetime.timedelta

        See Also
        --------
        to_timedelta : Convert argument to Timedelta type.

        Notes
        -----
        Any nanosecond resolution will be lost.

        Examples
        --------
        >>> td = pd.Timedelta('3D')
        >>> td
        Timedelta('3 days 00:00:00')
        >>> td.to_pytimedelta()
        datetime.timedelta(days=3)
        """
        if self._creso == NPY_FR_ns:
            return timedelta(microseconds=int(self._value) / 1000)

        # TODO(@WillAyd): is this the right way to use components?
        self._ensure_components()
        return timedelta(
            days=self._d, seconds=self._seconds, microseconds=self._microseconds
        )

    def to_timedelta64(self) -> np.timedelta64:
        """
        Return a numpy.timedelta64 object with 'ns' precision.

        Examples
        --------
        >>> td = pd.Timedelta('3D')
        >>> td
        Timedelta('3 days 00:00:00')
        >>> td.to_timedelta64()
        numpy.timedelta64(259200000000000,'ns')
        """
        cdef:
            str abbrev = npy_unit_to_abbrev(self._creso)
        # TODO: way to create a np.timedelta64 obj with the reso directly
        #  instead of having to get the abbrev?
        return np.timedelta64(self._value, abbrev)

    def to_numpy(self, dtype=None, copy=False) -> np.timedelta64:
        """
        Convert the Timedelta to a NumPy timedelta64.

        This is an alias method for `Timedelta.to_timedelta64()`. The dtype and
        copy parameters are available here only for compatibility. Their values
        will not affect the return value.

        Returns
        -------
        numpy.timedelta64

        See Also
        --------
        Series.to_numpy : Similar method for Series.

        Examples
        --------
        >>> td = pd.Timedelta('3D')
        >>> td
        Timedelta('3 days 00:00:00')
        >>> td.to_numpy()
        numpy.timedelta64(259200000000000,'ns')
        """
        if dtype is not None or copy is not False:
            raise ValueError(
                "Timedelta.to_numpy dtype and copy arguments are ignored"
            )
        return self.to_timedelta64()

    def view(self, dtype):
        """
        Array view compatibility.

        Parameters
        ----------
        dtype : str or dtype
            The dtype to view the underlying data as.

        Examples
        --------
        >>> td = pd.Timedelta('3D')
        >>> td
        Timedelta('3 days 00:00:00')
        >>> td.view(int)
        259200000000000
        """
        return np.timedelta64(self._value).view(dtype)

    @property
    def components(self):
        """
        Return a components namedtuple-like.

        Examples
        --------
        >>> td = pd.Timedelta('2 day 4 min 3 us 42 ns')
        >>> td.components
        Components(days=2, hours=0, minutes=4, seconds=0, milliseconds=0,
            microseconds=3, nanoseconds=42)
        """
        self._ensure_components()
        # return the named tuple
        return Components(self._d, self._h, self._m, self._s,
                          self._ms, self._us, self._ns)

    @property
    def asm8(self) -> np.timedelta64:
        """
        Return a numpy timedelta64 array scalar view.

        Provides access to the array scalar view (i.e. a combination of the
        value and the units) associated with the numpy.timedelta64().view(),
        including a 64-bit integer representation of the timedelta in
        nanoseconds (Python int compatible).

        Returns
        -------
        numpy timedelta64 array scalar view
            Array scalar view of the timedelta in nanoseconds.

        Examples
        --------
        >>> td = pd.Timedelta('1 days 2 min 3 us 42 ns')
        >>> td.asm8
        numpy.timedelta64(86520000003042,'ns')

        >>> td = pd.Timedelta('2 min 3 s')
        >>> td.asm8
        numpy.timedelta64(123000000000,'ns')

        >>> td = pd.Timedelta('3 ms 5 us')
        >>> td.asm8
        numpy.timedelta64(3005000,'ns')

        >>> td = pd.Timedelta(42, unit='ns')
        >>> td.asm8
        numpy.timedelta64(42,'ns')
        """
        return self.to_timedelta64()

    @property
    def resolution_string(self) -> str:
        """
        Return a string representing the lowest timedelta resolution.

        Each timedelta has a defined resolution that represents the lowest OR
        most granular level of precision. Each level of resolution is
        represented by a short string as defined below:

        Resolution:     Return value

        * Days:         'D'
        * Hours:        'H'
        * Minutes:      'T'
        * Seconds:      'S'
        * Milliseconds: 'L'
        * Microseconds: 'U'
        * Nanoseconds:  'N'

        Returns
        -------
        str
            Timedelta resolution.

        Examples
        --------
        >>> td = pd.Timedelta('1 days 2 min 3 us 42 ns')
        >>> td.resolution_string
        'N'

        >>> td = pd.Timedelta('1 days 2 min 3 us')
        >>> td.resolution_string
        'U'

        >>> td = pd.Timedelta('2 min 3 s')
        >>> td.resolution_string
        'S'

        >>> td = pd.Timedelta(36, unit='us')
        >>> td.resolution_string
        'U'
        """
        self._ensure_components()
        if self._ns:
            return "N"
        elif self._us:
            return "U"
        elif self._ms:
            return "L"
        elif self._s:
            return "S"
        elif self._m:
            return "T"
        elif self._h:
            return "H"
        else:
            return "D"

    @property
    def nanoseconds(self):
        """
        Return the number of nanoseconds (n), where 0 <= n < 1 microsecond.

        Returns
        -------
        int
            Number of nanoseconds.

        See Also
        --------
        Timedelta.components : Return all attributes with assigned values
            (i.e. days, hours, minutes, seconds, milliseconds, microseconds,
            nanoseconds).

        Examples
        --------
        **Using string input**

        >>> td = pd.Timedelta('1 days 2 min 3 us 42 ns')

        >>> td.nanoseconds
        42

        **Using integer input**

        >>> td = pd.Timedelta(42, unit='ns')
        >>> td.nanoseconds
        42
        """
        self._ensure_components()
        return self._ns

    def _repr_base(self, format=None) -> str:
        """

        Parameters
        ----------
        format : None|all|sub_day|long

        Returns
        -------
        converted : string of a Timedelta

        """
        cdef:
            str sign, fmt
            dict comp_dict
            object subs

        self._ensure_components()

        if self._d < 0:
            sign = " +"
        else:
            sign = " "

        if format == "all":
            fmt = ("{days} days{sign}{hours:02}:{minutes:02}:{seconds:02}."
                   "{milliseconds:03}{microseconds:03}{nanoseconds:03}")
        else:
            # if we have a partial day
            subs = (self._h or self._m or self._s or
                    self._ms or self._us or self._ns)

            if self._ms or self._us or self._ns:
                seconds_fmt = "{seconds:02}.{milliseconds:03}{microseconds:03}"
                if self._ns:
                    # GH#9309
                    seconds_fmt += "{nanoseconds:03}"
            else:
                seconds_fmt = "{seconds:02}"

            if format == "sub_day" and not self._d:
                fmt = "{hours:02}:{minutes:02}:" + seconds_fmt
            elif subs or format == "long":
                fmt = "{days} days{sign}{hours:02}:{minutes:02}:" + seconds_fmt
            else:
                fmt = "{days} days"

        comp_dict = self.components._asdict()
        comp_dict["sign"] = sign

        return fmt.format(**comp_dict)

    def __repr__(self) -> str:
        repr_based = self._repr_base(format="long")
        return f"Timedelta('{repr_based}')"

    def __str__(self) -> str:
        return self._repr_base(format="long")

    def __bool__(self) -> bool:
        return self._value!= 0

    def isoformat(self) -> str:
        """
        Format the Timedelta as ISO 8601 Duration.

        ``P[n]Y[n]M[n]DT[n]H[n]M[n]S``, where the ``[n]`` s are replaced by the
        values. See https://en.wikipedia.org/wiki/ISO_8601#Durations.

        Returns
        -------
        str

        See Also
        --------
        Timestamp.isoformat : Function is used to convert the given
            Timestamp object into the ISO format.

        Notes
        -----
        The longest component is days, whose value may be larger than
        365.
        Every component is always included, even if its value is 0.
        Pandas uses nanosecond precision, so up to 9 decimal places may
        be included in the seconds component.
        Trailing 0's are removed from the seconds component after the decimal.
        We do not 0 pad components, so it's `...T5H...`, not `...T05H...`

        Examples
        --------
        >>> td = pd.Timedelta(days=6, minutes=50, seconds=3,
        ...                   milliseconds=10, microseconds=10, nanoseconds=12)

        >>> td.isoformat()
        'P6DT0H50M3.010010012S'
        >>> pd.Timedelta(hours=1, seconds=10).isoformat()
        'P0DT1H0M10S'
        >>> pd.Timedelta(days=500.5).isoformat()
        'P500DT12H0M0S'
        """
        components = self.components
        seconds = (f"{components.seconds}."
                   f"{components.milliseconds:0>3}"
                   f"{components.microseconds:0>3}"
                   f"{components.nanoseconds:0>3}")
        # Trim unnecessary 0s, 1.000000000 -> 1
        seconds = seconds.rstrip("0").rstrip(".")
        tpl = (f"P{components.days}DT{components.hours}"
               f"H{components.minutes}M{seconds}S")
        return tpl

    # ----------------------------------------------------------------
    # Constructors

    @classmethod
    def _from_value_and_reso(cls, int64_t value, NPY_DATETIMEUNIT reso):
        # exposing as classmethod for testing
        return _timedelta_from_value_and_reso(cls, value, reso)

    def as_unit(self, str unit, bint round_ok=True):
        """
        Convert the underlying int64 representation to the given unit.

        Parameters
        ----------
        unit : {"ns", "us", "ms", "s"}
        round_ok : bool, default True
            If False and the conversion requires rounding, raise.

        Returns
        -------
        Timedelta

        Examples
        --------
        >>> td = pd.Timedelta('1001ms')
        >>> td
        Timedelta('0 days 00:00:01.001000')
        >>> td.as_unit('s')
        Timedelta('0 days 00:00:01')
        """
        dtype = np.dtype(f"m8[{unit}]")
        reso = get_unit_from_dtype(dtype)
        return self._as_creso(reso, round_ok=round_ok)

    @cython.cdivision(False)
    cdef _Timedelta _as_creso(self, NPY_DATETIMEUNIT reso, bint round_ok=True):
        cdef:
            int64_t value

        if reso == self._creso:
            return self

        try:
            value = convert_reso(self._value, self._creso, reso, round_ok=round_ok)
        except OverflowError as err:
            unit = npy_unit_to_abbrev(reso)
            raise OutOfBoundsTimedelta(
                f"Cannot cast {self} to unit='{unit}' without overflow."
            ) from err

        return type(self)._from_value_and_reso(value, reso=reso)

    cpdef _maybe_cast_to_matching_resos(self, _Timedelta other):
        """
        If _resos do not match, cast to the higher resolution, raising on overflow.
        """
        if self._creso > other._creso:
            other = other._as_creso(self._creso)
        elif self._creso < other._creso:
            self = self._as_creso(other._creso)
        return self, other


# Python front end to C extension type _Timedelta
# This serves as the box for timedelta64

class Timedelta(_Timedelta):
    """
    Represents a duration, the difference between two dates or times.

    Timedelta is the pandas equivalent of python's ``datetime.timedelta``
    and is interchangeable with it in most cases.

    Parameters
    ----------
    value : Timedelta, timedelta, np.timedelta64, str, or int
    unit : str, default 'ns'
        Denote the unit of the input, if input is an integer.

        Possible values:

        * 'W', 'D', 'T', 'S', 'L', 'U', or 'N'
        * 'days' or 'day'
        * 'hours', 'hour', 'hr', or 'h'
        * 'minutes', 'minute', 'min', or 'm'
        * 'seconds', 'second', or 'sec'
        * 'milliseconds', 'millisecond', 'millis', or 'milli'
        * 'microseconds', 'microsecond', 'micros', or 'micro'
        * 'nanoseconds', 'nanosecond', 'nanos', 'nano', or 'ns'.

    **kwargs
        Available kwargs: {days, seconds, microseconds,
        milliseconds, minutes, hours, weeks}.
        Values for construction in compat with datetime.timedelta.
        Numpy ints and floats will be coerced to python ints and floats.

    Notes
    -----
    The constructor may take in either both values of value and unit or
    kwargs as above. Either one of them must be used during initialization

    The ``.value`` attribute is always in ns.

    If the precision is higher than nanoseconds, the precision of the duration is
    truncated to nanoseconds.

    Examples
    --------
    Here we initialize Timedelta object with both value and unit

    >>> td = pd.Timedelta(1, "d")
    >>> td
    Timedelta('1 days 00:00:00')

    Here we initialize the Timedelta object with kwargs

    >>> td2 = pd.Timedelta(days=1)
    >>> td2
    Timedelta('1 days 00:00:00')

    We see that either way we get the same result
    """

    _req_any_kwargs_new = {"weeks", "days", "hours", "minutes", "seconds",
                           "milliseconds", "microseconds", "nanoseconds"}

    def __new__(cls, object value=_no_input, unit=None, **kwargs):
        if value is _no_input:
            if not len(kwargs):
                raise ValueError("cannot construct a Timedelta without a "
                                 "value/unit or descriptive keywords "
                                 "(days,seconds....)")

            kwargs = {key: _to_py_int_float(kwargs[key]) for key in kwargs}

            unsupported_kwargs = set(kwargs)
            unsupported_kwargs.difference_update(cls._req_any_kwargs_new)
            if unsupported_kwargs or not cls._req_any_kwargs_new.intersection(kwargs):
                raise ValueError(
                    "cannot construct a Timedelta from the passed arguments, "
                    "allowed keywords are "
                    "[weeks, days, hours, minutes, seconds, "
                    "milliseconds, microseconds, nanoseconds]"
                )

            # GH43764, convert any input to nanoseconds first and then
            # create the timestamp. This ensures that any potential
            # nanosecond contributions from kwargs parsed as floats
            # are taken into consideration.
            seconds = int((
                (
                    (kwargs.get("days", 0) + kwargs.get("weeks", 0) * 7) * 24
                    + kwargs.get("hours", 0)
                ) * 3600
                + kwargs.get("minutes", 0) * 60
                + kwargs.get("seconds", 0)
                ) * 1_000_000_000
            )

            value = np.timedelta64(
                int(kwargs.get("nanoseconds", 0))
                + int(kwargs.get("microseconds", 0) * 1_000)
                + int(kwargs.get("milliseconds", 0) * 1_000_000)
                + seconds
            )
        if unit in {"Y", "y", "M"}:
            raise ValueError(
                "Units 'M', 'Y', and 'y' are no longer supported, as they do not "
                "represent unambiguous timedelta values durations."
            )

        # GH 30543 if pd.Timedelta already passed, return it
        # check that only value is passed
        if isinstance(value, _Timedelta):
            # 'unit' is benign in this case, but e.g. days or seconds
            #  doesn't make sense here.
            if len(kwargs):
                # GH#48898
                raise ValueError(
                    "Cannot pass both a Timedelta input and timedelta keyword "
                    "arguments, got "
                    f"{list(kwargs.keys())}"
                )
            return value
        elif isinstance(value, str):
            if unit is not None:
                raise ValueError("unit must not be specified if the value is a str")
            if (len(value) > 0 and value[0] == "P") or (
                len(value) > 1 and value[:2] == "-P"
            ):
                value = parse_iso_format_string(value)
            else:
                value = parse_timedelta_string(value)
            value = np.timedelta64(value)
        elif PyDelta_Check(value):
            # pytimedelta object -> microsecond resolution
            new_value = delta_to_nanoseconds(
                value, reso=NPY_DATETIMEUNIT.NPY_FR_us
            )
            return cls._from_value_and_reso(
                new_value, reso=NPY_DATETIMEUNIT.NPY_FR_us
            )
        elif is_timedelta64_object(value):
            # Retain the resolution if possible, otherwise cast to the nearest
            #  supported resolution.
            new_value = get_timedelta64_value(value)
            if new_value == NPY_NAT:
                # i.e. np.timedelta64("NaT")
                return NaT

            reso = get_datetime64_unit(value)
            if not (is_supported_unit(reso) or
                    reso in [NPY_DATETIMEUNIT.NPY_FR_m,
                             NPY_DATETIMEUNIT.NPY_FR_h,
                             NPY_DATETIMEUNIT.NPY_FR_D,
                             NPY_DATETIMEUNIT.NPY_FR_W,
                             NPY_DATETIMEUNIT.NPY_FR_GENERIC]):
                err = npy_unit_to_abbrev(reso)
                raise ValueError(
                    f"Unit {err} is not supported. "
                    "Only unambiguous timedelta values durations are supported. "
                    "Allowed units are 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns'")

            new_reso = get_supported_reso(reso)
            if reso != NPY_DATETIMEUNIT.NPY_FR_GENERIC:
                try:
                    new_value = convert_reso(
                        new_value,
                        reso,
                        new_reso,
                        round_ok=True,
                    )
                except (OverflowError, OutOfBoundsDatetime) as err:
                    raise OutOfBoundsTimedelta(value) from err
            return cls._from_value_and_reso(new_value, reso=new_reso)

        elif is_tick_object(value):
            new_reso = get_supported_reso(value._creso)
            new_value = delta_to_nanoseconds(value, reso=new_reso)
            return cls._from_value_and_reso(new_value, reso=new_reso)

        elif is_integer_object(value) or is_float_object(value):
            # unit=None is de-facto 'ns'
            unit = parse_timedelta_unit(unit)
            value = convert_to_timedelta64(value, unit)
        elif checknull_with_nat_and_na(value):
            return NaT
        else:
            raise ValueError(
                "Value must be Timedelta, string, integer, "
                f"float, timedelta or convertible, not {type(value).__name__}"
            )

        if is_timedelta64_object(value):
            value = value.view("i8")

        # nat
        if value == NPY_NAT:
            return NaT

        return _timedelta_from_value_and_reso(cls, value, NPY_FR_ns)

    def __setstate__(self, state):
        if len(state) == 1:
            # older pickle, only supported nanosecond
            value = state[0]
            reso = NPY_FR_ns
        else:
            value, reso = state
        self._value= value
        self._creso = reso

    def __reduce__(self):
        object_state = self._value, self._creso
        return (_timedelta_unpickle, object_state)

    @cython.cdivision(True)
    def _round(self, freq, mode):
        cdef:
            int64_t result, unit
            ndarray[int64_t] arr

        from pandas._libs.tslibs.offsets import to_offset

        to_offset(freq).nanos  # raises on non-fixed freq
        unit = delta_to_nanoseconds(to_offset(freq), self._creso)

        arr = np.array([self._value], dtype="i8")
        try:
            result = round_nsint64(arr, mode, unit)[0]
        except OverflowError as err:
            raise OutOfBoundsTimedelta(
                f"Cannot round {self} to freq={freq} without overflow"
            ) from err
        return Timedelta._from_value_and_reso(result, self._creso)

    def round(self, freq):
        """
        Round the Timedelta to the specified resolution.

        Parameters
        ----------
        freq : str
            Frequency string indicating the rounding resolution.

        Returns
        -------
        a new Timedelta rounded to the given resolution of `freq`

        Raises
        ------
        ValueError if the freq cannot be converted

        Examples
        --------
        >>> td = pd.Timedelta('1001ms')
        >>> td
        Timedelta('0 days 00:00:01.001000')
        >>> td.round('s')
        Timedelta('0 days 00:00:01')
        """
        return self._round(freq, RoundTo.NEAREST_HALF_EVEN)

    def floor(self, freq):
        """
        Return a new Timedelta floored to this resolution.

        Parameters
        ----------
        freq : str
            Frequency string indicating the flooring resolution.

        Examples
        --------
        >>> td = pd.Timedelta('1001ms')
        >>> td
        Timedelta('0 days 00:00:01.001000')
        >>> td.floor('s')
        Timedelta('0 days 00:00:01')
        """
        return self._round(freq, RoundTo.MINUS_INFTY)

    def ceil(self, freq):
        """
        Return a new Timedelta ceiled to this resolution.

        Parameters
        ----------
        freq : str
            Frequency string indicating the ceiling resolution.

        Examples
        --------
        >>> td = pd.Timedelta('1001ms')
        >>> td
        Timedelta('0 days 00:00:01.001000')
        >>> td.ceil('s')
        Timedelta('0 days 00:00:02')
        """
        return self._round(freq, RoundTo.PLUS_INFTY)

    # ----------------------------------------------------------------
    # Arithmetic Methods
    # TODO: Can some of these be defined in the cython class?

    __neg__ = _op_unary_method(lambda x: -x, "__neg__")
    __pos__ = _op_unary_method(lambda x: x, "__pos__")
    __abs__ = _op_unary_method(lambda x: abs(x), "__abs__")

    __add__ = _binary_op_method_timedeltalike(lambda x, y: x + y, "__add__")
    __radd__ = _binary_op_method_timedeltalike(lambda x, y: x + y, "__radd__")
    __sub__ = _binary_op_method_timedeltalike(lambda x, y: x - y, "__sub__")
    __rsub__ = _binary_op_method_timedeltalike(lambda x, y: y - x, "__rsub__")

    def __mul__(self, other):
        if is_integer_object(other) or is_float_object(other):
            if util.is_nan(other):
                # np.nan * timedelta -> np.timedelta64("NaT"), in this case NaT
                return NaT

            return _timedelta_from_value_and_reso(
                Timedelta,
                <int64_t>(other * self._value),
                reso=self._creso,
            )

        elif is_array(other):
            if other.ndim == 0:
                # see also: item_from_zerodim
                item = cnp.PyArray_ToScalar(cnp.PyArray_DATA(other), other)
                return self.__mul__(item)
            return other * self.to_timedelta64()

        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, other):
        if _should_cast_to_timedelta(other):
            # We interpret NaT as timedelta64("NaT")
            other = Timedelta(other)
            if other is NaT:
                return np.nan
            if other._creso != self._creso:
                self, other = self._maybe_cast_to_matching_resos(other)
            return self._value/ float(other._value)

        elif is_integer_object(other) or is_float_object(other):
            # integers or floats
            if util.is_nan(other):
                return NaT
            return Timedelta._from_value_and_reso(
                <int64_t>(self._value/ other), self._creso
            )

        elif is_array(other):
            if other.ndim == 0:
                # see also: item_from_zerodim
                item = cnp.PyArray_ToScalar(cnp.PyArray_DATA(other), other)
                return self.__truediv__(item)
            return self.to_timedelta64() / other

        return NotImplemented

    def __rtruediv__(self, other):
        if _should_cast_to_timedelta(other):
            # We interpret NaT as timedelta64("NaT")
            other = Timedelta(other)
            if other is NaT:
                return np.nan
            if self._creso != other._creso:
                self, other = self._maybe_cast_to_matching_resos(other)
            return float(other._value) / self._value

        elif is_array(other):
            if other.ndim == 0:
                # see also: item_from_zerodim
                item = cnp.PyArray_ToScalar(cnp.PyArray_DATA(other), other)
                return self.__rtruediv__(item)
            elif other.dtype.kind == "O":
                # GH#31869
                return np.array([x / self for x in other])

            # TODO: if other.dtype.kind == "m" and other.dtype != self.asm8.dtype
            #  then should disallow for consistency with scalar behavior; requires
            #  deprecation cycle. (or changing scalar behavior)
            return other / self.to_timedelta64()

        return NotImplemented

    def __floordiv__(self, other):
        # numpy does not implement floordiv for timedelta64 dtype, so we cannot
        # just defer
        if _should_cast_to_timedelta(other):
            # We interpret NaT as timedelta64("NaT")
            other = Timedelta(other)
            if other is NaT:
                return np.nan
            if self._creso != other._creso:
                self, other = self._maybe_cast_to_matching_resos(other)
            return self._value// other._value

        elif is_integer_object(other) or is_float_object(other):
            if util.is_nan(other):
                return NaT
            return type(self)._from_value_and_reso(self._value// other, self._creso)

        elif is_array(other):
            if other.ndim == 0:
                # see also: item_from_zerodim
                item = cnp.PyArray_ToScalar(cnp.PyArray_DATA(other), other)
                return self.__floordiv__(item)

            if other.dtype.kind == "m":
                # also timedelta-like
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        "invalid value encountered in floor_divide",
                        RuntimeWarning
                    )
                    result = self.asm8 // other
                mask = other.view("i8") == NPY_NAT
                if mask.any():
                    # We differ from numpy here
                    result = result.astype("f8")
                    result[mask] = np.nan
                return result

            elif other.dtype.kind in "iuf":
                if other.ndim == 0:
                    return self // other.item()
                else:
                    return self.to_timedelta64() // other

            raise TypeError(f"Invalid dtype {other.dtype} for __floordiv__")

        return NotImplemented

    def __rfloordiv__(self, other):
        # numpy does not implement floordiv for timedelta64 dtype, so we cannot
        # just defer
        if _should_cast_to_timedelta(other):
            # We interpret NaT as timedelta64("NaT")
            other = Timedelta(other)
            if other is NaT:
                return np.nan
            if self._creso != other._creso:
                self, other = self._maybe_cast_to_matching_resos(other)
            return other._value// self._value

        elif is_array(other):
            if other.ndim == 0:
                # see also: item_from_zerodim
                item = cnp.PyArray_ToScalar(cnp.PyArray_DATA(other), other)
                return self.__rfloordiv__(item)

            if other.dtype.kind == "m":
                # also timedelta-like
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        "invalid value encountered in floor_divide",
                        RuntimeWarning
                    )
                    result = other // self.asm8
                mask = other.view("i8") == NPY_NAT
                if mask.any():
                    # We differ from numpy here
                    result = result.astype("f8")
                    result[mask] = np.nan
                return result

            # Includes integer array // Timedelta, disallowed in GH#19761
            raise TypeError(f"Invalid dtype {other.dtype} for __floordiv__")

        return NotImplemented

    def __mod__(self, other):
        # Naive implementation, room for optimization
        return self.__divmod__(other)[1]

    def __rmod__(self, other):
        # Naive implementation, room for optimization
        return self.__rdivmod__(other)[1]

    def __divmod__(self, other):
        # Naive implementation, room for optimization
        div = self // other
        return div, self - div * other

    def __rdivmod__(self, other):
        # Naive implementation, room for optimization
        div = other // self
        return div, other - div * self


def truediv_object_array(ndarray left, ndarray right):
    cdef:
        ndarray[object] result = np.empty((<object>left).shape, dtype=object)
        object td64  # really timedelta64 if we find a way to declare that
        object obj, res_value
        _Timedelta td
        Py_ssize_t i

    for i in range(len(left)):
        td64 = left[i]
        obj = right[i]

        if get_timedelta64_value(td64) == NPY_NAT:
            # td here should be interpreted as a td64 NaT
            if _should_cast_to_timedelta(obj):
                res_value = np.nan
            else:
                # if its a number then let numpy handle division, otherwise
                #  numpy will raise
                res_value = td64 / obj
        else:
            td = Timedelta(td64)
            res_value = td / obj

        result[i] = res_value

    return result


def floordiv_object_array(ndarray left, ndarray right):
    cdef:
        ndarray[object] result = np.empty((<object>left).shape, dtype=object)
        object td64  # really timedelta64 if we find a way to declare that
        object obj, res_value
        _Timedelta td
        Py_ssize_t i

    for i in range(len(left)):
        td64 = left[i]
        obj = right[i]

        if get_timedelta64_value(td64) == NPY_NAT:
            # td here should be interpreted as a td64 NaT
            if _should_cast_to_timedelta(obj):
                res_value = np.nan
            else:
                # if its a number then let numpy handle division, otherwise
                #  numpy will raise
                res_value = td64 // obj
        else:
            td = Timedelta(td64)
            res_value = td // obj

        result[i] = res_value

    return result


cdef bint is_any_td_scalar(object obj):
    """
    Cython equivalent for `isinstance(obj, (timedelta, np.timedelta64, Tick))`

    Parameters
    ----------
    obj : object

    Returns
    -------
    bool
    """
    return (
        PyDelta_Check(obj) or is_timedelta64_object(obj) or is_tick_object(obj)
    )


cdef bint _should_cast_to_timedelta(object obj):
    """
    Should we treat this object as a Timedelta for the purpose of a binary op
    """
    return (
        is_any_td_scalar(obj) or obj is None or obj is NaT or isinstance(obj, str)
    )
