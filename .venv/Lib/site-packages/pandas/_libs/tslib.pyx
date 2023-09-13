import warnings

from pandas.util._exceptions import find_stack_level

cimport cython

from datetime import timezone

from cpython.datetime cimport (
    PyDate_Check,
    PyDateTime_Check,
    datetime,
    import_datetime,
    timedelta,
    tzinfo,
)
from cpython.object cimport PyObject

# import datetime C API
import_datetime()


cimport numpy as cnp
from numpy cimport (
    int64_t,
    ndarray,
)

import numpy as np

cnp.import_array()

from pandas._libs.tslibs.np_datetime cimport (
    NPY_DATETIMEUNIT,
    NPY_FR_ns,
    check_dts_bounds,
    import_pandas_datetime,
    npy_datetimestruct,
    npy_datetimestruct_to_datetime,
    pandas_datetime_to_datetimestruct,
    pydate_to_dt64,
    string_to_dts,
)

import_pandas_datetime()


from pandas._libs.tslibs.strptime cimport parse_today_now
from pandas._libs.util cimport (
    is_datetime64_object,
    is_float_object,
    is_integer_object,
)

from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime

from pandas._libs.tslibs.conversion cimport (
    _TSObject,
    cast_from_unit,
    convert_str_to_tsobject,
    convert_timezone,
    get_datetime64_nanos,
    parse_pydatetime,
)
from pandas._libs.tslibs.nattype cimport (
    NPY_NAT,
    c_NaT as NaT,
    c_nat_strings as nat_strings,
)
from pandas._libs.tslibs.timestamps cimport _Timestamp

from pandas._libs.tslibs import (
    Resolution,
    get_resolution,
)
from pandas._libs.tslibs.timestamps import Timestamp

# Note: this is the only non-tslibs intra-pandas dependency here

from pandas._libs.missing cimport checknull_with_nat_and_na
from pandas._libs.tslibs.tzconversion cimport tz_localize_to_utc_single


def _test_parse_iso8601(ts: str):
    """
    TESTING ONLY: Parse string into Timestamp using iso8601 parser. Used
    only for testing, actual construction uses `convert_str_to_tsobject`
    """
    cdef:
        _TSObject obj
        int out_local = 0, out_tzoffset = 0
        NPY_DATETIMEUNIT out_bestunit

    obj = _TSObject()

    string_to_dts(ts, &obj.dts, &out_bestunit, &out_local, &out_tzoffset, True)
    obj.value = npy_datetimestruct_to_datetime(NPY_FR_ns, &obj.dts)
    check_dts_bounds(&obj.dts)
    if out_local == 1:
        obj.tzinfo = timezone(timedelta(minutes=out_tzoffset))
        obj.value = tz_localize_to_utc_single(obj.value, obj.tzinfo)
        return Timestamp(obj.value, tz=obj.tzinfo)
    else:
        return Timestamp(obj.value)


@cython.wraparound(False)
@cython.boundscheck(False)
def format_array_from_datetime(
    ndarray values,
    tzinfo tz=None,
    str format=None,
    na_rep: str | float = "NaT",
    NPY_DATETIMEUNIT reso=NPY_FR_ns,
) -> np.ndarray:
    """
    return a np object array of the string formatted values

    Parameters
    ----------
    values : ndarray[int64_t], arbitrary ndim
    tz : tzinfo or None, default None
    format : str or None, default None
          a strftime capable string
    na_rep : optional, default is None
          a nat format
    reso : NPY_DATETIMEUNIT, default NPY_FR_ns

    Returns
    -------
    np.ndarray[object]
    """
    cdef:
        int64_t val, ns, N = values.size
        bint show_ms = False, show_us = False, show_ns = False
        bint basic_format = False, basic_format_day = False
        _Timestamp ts
        object res
        npy_datetimestruct dts

        # Note that `result` (and thus `result_flat`) is C-order and
        #  `it` iterates C-order as well, so the iteration matches
        #  See discussion at
        #  github.com/pandas-dev/pandas/pull/46886#discussion_r860261305
        ndarray result = cnp.PyArray_EMPTY(values.ndim, values.shape, cnp.NPY_OBJECT, 0)
        object[::1] res_flat = result.ravel()     # should NOT be a copy
        cnp.flatiter it = cnp.PyArray_IterNew(values)

    if tz is None:
        # if we don't have a format nor tz, then choose
        # a format based on precision
        basic_format = format is None
        if basic_format:
            reso_obj = get_resolution(values, tz=tz, reso=reso)
            show_ns = reso_obj == Resolution.RESO_NS
            show_us = reso_obj == Resolution.RESO_US
            show_ms = reso_obj == Resolution.RESO_MS

        elif format == "%Y-%m-%d %H:%M:%S":
            # Same format as default, but with hardcoded precision (s)
            basic_format = True
            show_ns = show_us = show_ms = False

        elif format == "%Y-%m-%d %H:%M:%S.%f":
            # Same format as default, but with hardcoded precision (us)
            basic_format = show_us = True
            show_ns = show_ms = False

        elif format == "%Y-%m-%d":
            # Default format for dates
            basic_format_day = True

    assert not (basic_format_day and basic_format)

    for i in range(N):
        # Analogous to: utc_val = values[i]
        val = (<int64_t*>cnp.PyArray_ITER_DATA(it))[0]

        if val == NPY_NAT:
            res = na_rep
        elif basic_format_day:

            pandas_datetime_to_datetimestruct(val, reso, &dts)
            res = f"{dts.year}-{dts.month:02d}-{dts.day:02d}"

        elif basic_format:

            pandas_datetime_to_datetimestruct(val, reso, &dts)
            res = (f"{dts.year}-{dts.month:02d}-{dts.day:02d} "
                   f"{dts.hour:02d}:{dts.min:02d}:{dts.sec:02d}")

            if show_ns:
                ns = dts.ps // 1000
                res += f".{ns + dts.us * 1000:09d}"
            elif show_us:
                res += f".{dts.us:06d}"
            elif show_ms:
                res += f".{dts.us // 1000:03d}"

        else:

            ts = Timestamp._from_value_and_reso(val, reso=reso, tz=tz)
            if format is None:
                # Use datetime.str, that returns ts.isoformat(sep=' ')
                res = str(ts)
            else:

                # invalid format string
                # requires dates > 1900
                try:
                    # Note: dispatches to pydatetime
                    res = ts.strftime(format)
                except ValueError:
                    # Use datetime.str, that returns ts.isoformat(sep=' ')
                    res = str(ts)

        # Note: we can index result directly instead of using PyArray_MultiIter_DATA
        #  like we do for the other functions because result is known C-contiguous
        #  and is the first argument to PyArray_MultiIterNew2.  The usual pattern
        #  does not seem to work with object dtype.
        #  See discussion at
        #  github.com/pandas-dev/pandas/pull/46886#discussion_r860261305
        res_flat[i] = res

        cnp.PyArray_ITER_NEXT(it)

    return result


def array_with_unit_to_datetime(
    ndarray[object] values,
    str unit,
    str errors="coerce"
):
    """
    Convert the ndarray to datetime according to the time unit.

    This function converts an array of objects into a numpy array of
    datetime64[ns]. It returns the converted array
    and also returns the timezone offset

    if errors:
      - raise: return converted values or raise OutOfBoundsDatetime
          if out of range on the conversion or
          ValueError for other conversions (e.g. a string)
      - ignore: return non-convertible values as the same unit
      - coerce: NaT for non-convertibles

    Parameters
    ----------
    values : ndarray
         Date-like objects to convert.
    unit : str
         Time unit to use during conversion.
    errors : str, default 'raise'
         Error behavior when parsing.

    Returns
    -------
    result : ndarray of m8 values
    tz : parsed timezone offset or None
    """
    cdef:
        Py_ssize_t i, n=len(values)
        bint is_ignore = errors == "ignore"
        bint is_coerce = errors == "coerce"
        bint is_raise = errors == "raise"
        ndarray[int64_t] iresult
        tzinfo tz = None
        float fval

    assert is_ignore or is_coerce or is_raise

    if unit == "ns":
        result, tz = array_to_datetime(
            values.astype(object, copy=False),
            errors=errors,
        )
        return result, tz

    result = np.empty(n, dtype="M8[ns]")
    iresult = result.view("i8")

    for i in range(n):
        val = values[i]

        try:
            if checknull_with_nat_and_na(val):
                iresult[i] = NPY_NAT

            elif is_integer_object(val) or is_float_object(val):

                if val != val or val == NPY_NAT:
                    iresult[i] = NPY_NAT
                else:
                    iresult[i] = cast_from_unit(val, unit)

            elif isinstance(val, str):
                if len(val) == 0 or val in nat_strings:
                    iresult[i] = NPY_NAT

                else:

                    try:
                        fval = float(val)
                    except ValueError:
                        raise ValueError(
                            f"non convertible value {val} with the unit '{unit}'"
                        )
                    warnings.warn(
                        "The behavior of 'to_datetime' with 'unit' when parsing "
                        "strings is deprecated. In a future version, strings will "
                        "be parsed as datetime strings, matching the behavior "
                        "without a 'unit'. To retain the old behavior, explicitly "
                        "cast ints or floats to numeric type before calling "
                        "to_datetime.",
                        FutureWarning,
                        stacklevel=find_stack_level(),
                    )

                    iresult[i] = cast_from_unit(fval, unit)

            else:
                # TODO: makes more sense as TypeError, but that would be an
                #  API change.
                raise ValueError(
                    f"unit='{unit}' not valid with non-numerical val='{val}'"
                )

        except (ValueError, OutOfBoundsDatetime, TypeError) as err:
            if is_raise:
                err.args = (f"{err}, at position {i}",)
                raise
            elif is_ignore:
                # we have hit an exception
                # and are in ignore mode
                # redo as object
                return _array_with_unit_to_datetime_object_fallback(values, unit)
            else:
                # is_coerce
                iresult[i] = NPY_NAT

    return result, tz


cdef _array_with_unit_to_datetime_object_fallback(ndarray[object] values, str unit):
    cdef:
        Py_ssize_t i, n = len(values)
        ndarray[object] oresult
        tzinfo tz = None

    # TODO: fix subtle differences between this and no-unit code
    oresult = cnp.PyArray_EMPTY(values.ndim, values.shape, cnp.NPY_OBJECT, 0)
    for i in range(n):
        val = values[i]

        if checknull_with_nat_and_na(val):
            oresult[i] = <object>NaT
        elif is_integer_object(val) or is_float_object(val):

            if val != val or val == NPY_NAT:
                oresult[i] = <object>NaT
            else:
                try:
                    oresult[i] = Timestamp(val, unit=unit)
                except OutOfBoundsDatetime:
                    oresult[i] = val

        elif isinstance(val, str):
            if len(val) == 0 or val in nat_strings:
                oresult[i] = <object>NaT

            else:
                oresult[i] = val

    return oresult, tz


@cython.wraparound(False)
@cython.boundscheck(False)
def first_non_null(values: ndarray) -> int:
    """Find position of first non-null value, return -1 if there isn't one."""
    cdef:
        Py_ssize_t n = len(values)
        Py_ssize_t i
    for i in range(n):
        val = values[i]
        if checknull_with_nat_and_na(val):
            continue
        if (
            isinstance(val, str)
            and
            (len(val) == 0 or val in nat_strings or val in ("now", "today"))
        ):
            continue
        return i
    else:
        return -1


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef array_to_datetime(
    ndarray values,  # object dtype, arbitrary ndim
    str errors="raise",
    bint dayfirst=False,
    bint yearfirst=False,
    bint utc=False,
):
    """
    Converts a 1D array of date-like values to a numpy array of either:
        1) datetime64[ns] data
        2) datetime.datetime objects, if OutOfBoundsDatetime or TypeError
           is encountered

    Also returns a fixed-offset tzinfo object if an array of strings with the same
    timezone offset is passed and utc=True is not passed. Otherwise, None
    is returned

    Handles datetime.date, datetime.datetime, np.datetime64 objects, numeric,
    strings

    Parameters
    ----------
    values : ndarray of object
         date-like objects to convert
    errors : str, default 'raise'
         error behavior when parsing
    dayfirst : bool, default False
         dayfirst parsing behavior when encountering datetime strings
    yearfirst : bool, default False
         yearfirst parsing behavior when encountering datetime strings
    utc : bool, default False
         indicator whether the dates should be UTC

    Returns
    -------
    np.ndarray
        May be datetime64[ns] or object dtype
    tzinfo or None
    """
    cdef:
        Py_ssize_t i, n = values.size
        object val, tz
        ndarray[int64_t] iresult
        npy_datetimestruct dts
        bint utc_convert = bool(utc)
        bint seen_datetime_offset = False
        bint is_raise = errors == "raise"
        bint is_ignore = errors == "ignore"
        bint is_coerce = errors == "coerce"
        bint is_same_offsets
        _TSObject _ts
        float tz_offset
        set out_tzoffset_vals = set()
        tzinfo tz_out = None
        bint found_tz = False, found_naive = False
        cnp.broadcast mi

    # specify error conditions
    assert is_raise or is_ignore or is_coerce

    result = np.empty((<object>values).shape, dtype="M8[ns]")
    mi = cnp.PyArray_MultiIterNew2(result, values)
    iresult = result.view("i8").ravel()

    for i in range(n):
        # Analogous to `val = values[i]`
        val = <object>(<PyObject**>cnp.PyArray_MultiIter_DATA(mi, 1))[0]

        try:
            if checknull_with_nat_and_na(val):
                iresult[i] = NPY_NAT

            elif PyDateTime_Check(val):
                if val.tzinfo is not None:
                    found_tz = True
                else:
                    found_naive = True
                tz_out = convert_timezone(
                    val.tzinfo,
                    tz_out,
                    found_naive,
                    found_tz,
                    utc_convert,
                )
                iresult[i] = parse_pydatetime(val, &dts, utc_convert)

            elif PyDate_Check(val):
                iresult[i] = pydate_to_dt64(val, &dts)
                check_dts_bounds(&dts)

            elif is_datetime64_object(val):
                iresult[i] = get_datetime64_nanos(val, NPY_FR_ns)

            elif is_integer_object(val) or is_float_object(val):
                # these must be ns unit by-definition

                if val != val or val == NPY_NAT:
                    iresult[i] = NPY_NAT
                else:
                    # we now need to parse this as if unit='ns'
                    iresult[i] = cast_from_unit(val, "ns")

            elif isinstance(val, str):
                # string
                if type(val) is not str:
                    # GH#32264 np.str_ object
                    val = str(val)

                if parse_today_now(val, &iresult[i], utc):
                    # We can't _quite_ dispatch this to convert_str_to_tsobject
                    #  bc there isn't a nice way to pass "utc"
                    cnp.PyArray_MultiIter_NEXT(mi)
                    continue

                _ts = convert_str_to_tsobject(
                    val, None, unit="ns", dayfirst=dayfirst, yearfirst=yearfirst
                )
                _ts.ensure_reso(NPY_FR_ns, val)

                iresult[i] = _ts.value

                tz = _ts.tzinfo
                if tz is not None:
                    # dateutil timezone objects cannot be hashed, so
                    # store the UTC offsets in seconds instead
                    nsecs = tz.utcoffset(None).total_seconds()
                    out_tzoffset_vals.add(nsecs)
                    # need to set seen_datetime_offset *after* the
                    #  potentially-raising timezone(timedelta(...)) call,
                    #  otherwise we can go down the is_same_offsets path
                    #  bc len(out_tzoffset_vals) == 0
                    seen_datetime_offset = True
                else:
                    # Add a marker for naive string, to track if we are
                    # parsing mixed naive and aware strings
                    out_tzoffset_vals.add("naive")

            else:
                raise TypeError(f"{type(val)} is not convertible to datetime")

            cnp.PyArray_MultiIter_NEXT(mi)

        except (TypeError, OverflowError, ValueError) as ex:
            ex.args = (f"{ex}, at position {i}",)
            if is_coerce:
                iresult[i] = NPY_NAT
                cnp.PyArray_MultiIter_NEXT(mi)
                continue
            elif is_raise:
                raise
            return values, None

    if seen_datetime_offset and not utc_convert:
        # GH#17697
        # 1) If all the offsets are equal, return one offset for
        #    the parsed dates to (maybe) pass to DatetimeIndex
        # 2) If the offsets are different, then force the parsing down the
        #    object path where an array of datetimes
        #    (with individual dateutil.tzoffsets) are returned
        is_same_offsets = len(out_tzoffset_vals) == 1
        if not is_same_offsets:
            return _array_to_datetime_object(values, errors, dayfirst, yearfirst)
        else:
            tz_offset = out_tzoffset_vals.pop()
            tz_out = timezone(timedelta(seconds=tz_offset))
    return result, tz_out


@cython.wraparound(False)
@cython.boundscheck(False)
cdef _array_to_datetime_object(
    ndarray[object] values,
    str errors,
    bint dayfirst=False,
    bint yearfirst=False,
):
    """
    Fall back function for array_to_datetime

    Attempts to parse datetime strings with dateutil to return an array
    of datetime objects

    Parameters
    ----------
    values : ndarray[object]
         date-like objects to convert
    errors : str
         error behavior when parsing
    dayfirst : bool, default False
         dayfirst parsing behavior when encountering datetime strings
    yearfirst : bool, default False
         yearfirst parsing behavior when encountering datetime strings

    Returns
    -------
    np.ndarray[object]
    Literal[None]
    """
    cdef:
        Py_ssize_t i, n = values.size
        object val
        bint is_ignore = errors == "ignore"
        bint is_coerce = errors == "coerce"
        bint is_raise = errors == "raise"
        ndarray oresult_nd
        ndarray[object] oresult
        npy_datetimestruct dts
        cnp.broadcast mi
        _TSObject tsobj

    assert is_raise or is_ignore or is_coerce

    oresult_nd = cnp.PyArray_EMPTY(values.ndim, values.shape, cnp.NPY_OBJECT, 0)
    mi = cnp.PyArray_MultiIterNew2(oresult_nd, values)
    oresult = oresult_nd.ravel()

    # We return an object array and only attempt to parse:
    # 1) NaT or NaT-like values
    # 2) datetime strings, which we return as datetime.datetime
    # 3) special strings - "now" & "today"
    unique_timezones = set()
    for i in range(n):
        # Analogous to: val = values[i]
        val = <object>(<PyObject**>cnp.PyArray_MultiIter_DATA(mi, 1))[0]

        if checknull_with_nat_and_na(val) or PyDateTime_Check(val):
            # GH 25978. No need to parse NaT-like or datetime-like vals
            oresult[i] = val
        elif isinstance(val, str):
            if type(val) is not str:
                # GH#32264 np.str_ objects
                val = str(val)

            if len(val) == 0 or val in nat_strings:
                oresult[i] = "NaT"
                cnp.PyArray_MultiIter_NEXT(mi)
                continue

            try:
                tsobj = convert_str_to_tsobject(
                    val, None, unit="ns", dayfirst=dayfirst, yearfirst=yearfirst
                )
                tsobj.ensure_reso(NPY_FR_ns, val)

                dts = tsobj.dts
                oresult[i] = datetime(
                    dts.year, dts.month, dts.day, dts.hour, dts.min, dts.sec, dts.us,
                    tzinfo=tsobj.tzinfo,
                    fold=tsobj.fold,
                )
                unique_timezones.add(tsobj.tzinfo)

            except (ValueError, OverflowError) as ex:
                ex.args = (f"{ex}, at position {i}", )
                if is_coerce:
                    oresult[i] = <object>NaT
                    cnp.PyArray_MultiIter_NEXT(mi)
                    continue
                if is_raise:
                    raise
                return values, None
        else:
            if is_raise:
                raise
            return values, None

        cnp.PyArray_MultiIter_NEXT(mi)

    if len(unique_timezones) > 1:
        warnings.warn(
            "In a future version of pandas, parsing datetimes with mixed time "
            "zones will raise a warning unless `utc=True`. "
            "Please specify `utc=True` to opt in to the new behaviour "
            "and silence this warning. To create a `Series` with mixed offsets and "
            "`object` dtype, please use `apply` and `datetime.datetime.strptime`",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
    return oresult_nd, None


def array_to_datetime_with_tz(ndarray values, tzinfo tz):
    """
    Vectorized analogue to pd.Timestamp(value, tz=tz)

    values has object-dtype, unrestricted ndim.

    Major differences between this and array_to_datetime with utc=True
        - np.datetime64 objects are treated as _wall_ times.
        - tznaive datetimes are treated as _wall_ times.
    """
    cdef:
        ndarray result = cnp.PyArray_EMPTY(values.ndim, values.shape, cnp.NPY_INT64, 0)
        cnp.broadcast mi = cnp.PyArray_MultiIterNew2(result, values)
        Py_ssize_t i, n = values.size
        object item
        int64_t ival
        datetime ts

    for i in range(n):
        # Analogous to `item = values[i]`
        item = <object>(<PyObject**>cnp.PyArray_MultiIter_DATA(mi, 1))[0]

        if checknull_with_nat_and_na(item):
            # this catches pd.NA which would raise in the Timestamp constructor
            ival = NPY_NAT

        else:
            ts = Timestamp(item)
            if ts is NaT:
                ival = NPY_NAT
            else:
                if ts.tzinfo is not None:
                    ts = ts.tz_convert(tz)
                else:
                    # datetime64, tznaive pydatetime, int, float
                    ts = ts.tz_localize(tz)
                ts = ts.as_unit("ns")
                ival = ts._value

        # Analogous to: result[i] = ival
        (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 0))[0] = ival

        cnp.PyArray_MultiIter_NEXT(mi)

    return result
