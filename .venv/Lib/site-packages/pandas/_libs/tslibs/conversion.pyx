import numpy as np

cimport numpy as cnp
from libc.math cimport log10
from numpy cimport (
    int32_t,
    int64_t,
)

cnp.import_array()

# stdlib datetime imports

from datetime import timezone

from cpython.datetime cimport (
    PyDate_Check,
    PyDateTime_Check,
    datetime,
    import_datetime,
    time,
    timedelta,
    tzinfo,
)

import_datetime()

from pandas._libs.missing cimport checknull_with_nat_and_na
from pandas._libs.tslibs.base cimport ABCTimestamp
from pandas._libs.tslibs.dtypes cimport (
    abbrev_to_npy_unit,
    get_supported_reso,
    periods_per_second,
)
from pandas._libs.tslibs.np_datetime cimport (
    NPY_DATETIMEUNIT,
    NPY_FR_ns,
    NPY_FR_us,
    check_dts_bounds,
    convert_reso,
    get_conversion_factor,
    get_datetime64_unit,
    get_datetime64_value,
    get_implementation_bounds,
    import_pandas_datetime,
    npy_datetime,
    npy_datetimestruct,
    npy_datetimestruct_to_datetime,
    pandas_datetime_to_datetimestruct,
    pydatetime_to_dt64,
    pydatetime_to_dtstruct,
    string_to_dts,
)

import_pandas_datetime()

from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime

from pandas._libs.tslibs.nattype cimport (
    NPY_NAT,
    c_nat_strings as nat_strings,
)
from pandas._libs.tslibs.parsing cimport parse_datetime_string
from pandas._libs.tslibs.timezones cimport (
    get_utcoffset,
    is_utc,
)
from pandas._libs.tslibs.tzconversion cimport (
    Localizer,
    tz_localize_to_utc_single,
)
from pandas._libs.tslibs.util cimport (
    is_datetime64_object,
    is_float_object,
    is_integer_object,
)

# ----------------------------------------------------------------------
# Constants

DT64NS_DTYPE = np.dtype("M8[ns]")
TD64NS_DTYPE = np.dtype("m8[ns]")


# ----------------------------------------------------------------------
# Unit Conversion Helpers

cdef int64_t cast_from_unit(
    object ts,
    str unit,
    NPY_DATETIMEUNIT out_reso=NPY_FR_ns
) except? -1:
    """
    Return a casting of the unit represented to nanoseconds
    round the fractional part of a float to our precision, p.

    Parameters
    ----------
    ts : int, float, or None
    unit : str

    Returns
    -------
    int64_t
    """
    cdef:
        int64_t m
        int p

    if unit in ["Y", "M"]:
        if is_float_object(ts) and not ts.is_integer():
            # GH#47267 it is clear that 2 "M" corresponds to 1970-02-01,
            #  but not clear what 2.5 "M" corresponds to, so we will
            #  disallow that case.
            raise ValueError(
                f"Conversion of non-round float with unit={unit} "
                "is ambiguous"
            )
        # GH#47266 go through np.datetime64 to avoid weird results e.g. with "Y"
        #  and 150 we'd get 2120-01-01 09:00:00
        if is_float_object(ts):
            ts = int(ts)
        dt64obj = np.datetime64(ts, unit)
        return get_datetime64_nanos(dt64obj, out_reso)

    m, p = precision_from_unit(unit, out_reso)

    # cast the unit, multiply base/frac separately
    # to avoid precision issues from float -> int
    try:
        base = <int64_t>ts
    except OverflowError as err:
        raise OutOfBoundsDatetime(
            f"cannot convert input {ts} with the unit '{unit}'"
        ) from err

    frac = ts - base
    if p:
        frac = round(frac, p)

    try:
        return <int64_t>(base * m) + <int64_t>(frac * m)
    except OverflowError as err:
        raise OutOfBoundsDatetime(
            f"cannot convert input {ts} with the unit '{unit}'"
        ) from err


cpdef inline (int64_t, int) precision_from_unit(
    str unit,
    NPY_DATETIMEUNIT out_reso=NPY_DATETIMEUNIT.NPY_FR_ns,
):
    """
    Return a casting of the unit represented to nanoseconds + the precision
    to round the fractional part.

    Notes
    -----
    The caller is responsible for ensuring that the default value of "ns"
    takes the place of None.
    """
    cdef:
        int64_t m
        int64_t multiplier
        int p
        NPY_DATETIMEUNIT reso = abbrev_to_npy_unit(unit)

    if reso == NPY_DATETIMEUNIT.NPY_FR_GENERIC:
        reso = NPY_DATETIMEUNIT.NPY_FR_ns
    if reso == NPY_DATETIMEUNIT.NPY_FR_Y:
        # each 400 years we have 97 leap years, for an average of 97/400=.2425
        #  extra days each year. We get 31556952 by writing
        #  3600*24*365.2425=31556952
        multiplier = periods_per_second(out_reso)
        m = multiplier * 31556952
    elif reso == NPY_DATETIMEUNIT.NPY_FR_M:
        # 2629746 comes from dividing the "Y" case by 12.
        multiplier = periods_per_second(out_reso)
        m = multiplier * 2629746
    else:
        # Careful: if get_conversion_factor raises, the exception does
        #  not propagate, instead we get a warning about an ignored exception.
        #  https://github.com/pandas-dev/pandas/pull/51483#discussion_r1115198951
        m = get_conversion_factor(reso, out_reso)

    p = <int>log10(m)  # number of digits in 'm' minus 1
    return m, p


cdef int64_t get_datetime64_nanos(object val, NPY_DATETIMEUNIT reso) except? -1:
    """
    Extract the value and unit from a np.datetime64 object, then convert the
    value to nanoseconds if necessary.
    """
    cdef:
        npy_datetimestruct dts
        NPY_DATETIMEUNIT unit
        npy_datetime ival

    ival = get_datetime64_value(val)
    if ival == NPY_NAT:
        return NPY_NAT

    unit = get_datetime64_unit(val)

    if unit != reso:
        pandas_datetime_to_datetimestruct(ival, unit, &dts)
        check_dts_bounds(&dts, reso)
        ival = npy_datetimestruct_to_datetime(reso, &dts)

    return ival


# ----------------------------------------------------------------------
# _TSObject Conversion

# lightweight C object to hold datetime & int64 pair
cdef class _TSObject:
    # cdef:
    #    npy_datetimestruct dts      # npy_datetimestruct
    #    int64_t value               # numpy dt64
    #    tzinfo tzinfo
    #    bint fold
    #    NPY_DATETIMEUNIT creso

    def __cinit__(self):
        # GH 25057. As per PEP 495, set fold to 0 by default
        self.fold = 0
        self.creso = NPY_FR_ns  # default value

    cdef int64_t ensure_reso(self, NPY_DATETIMEUNIT creso, str val=None) except? -1:
        if self.creso != creso:
            try:
                self.value = convert_reso(self.value, self.creso, creso, False)
            except OverflowError as err:
                if val is not None:
                    raise OutOfBoundsDatetime(
                        f"Out of bounds nanosecond timestamp: {val}"
                    ) from err
                raise OutOfBoundsDatetime from err

            self.creso = creso
        return self.value


cdef _TSObject convert_to_tsobject(object ts, tzinfo tz, str unit,
                                   bint dayfirst, bint yearfirst, int32_t nanos=0):
    """
    Extract datetime and int64 from any of:
        - np.int64 (with unit providing a possible modifier)
        - np.datetime64
        - a float (with unit providing a possible modifier)
        - python int or long object (with unit providing a possible modifier)
        - iso8601 string object
        - python datetime object
        - another timestamp object

    Raises
    ------
    OutOfBoundsDatetime : ts cannot be converted within implementation bounds
    """
    cdef:
        _TSObject obj
        NPY_DATETIMEUNIT reso

    obj = _TSObject()

    if isinstance(ts, str):
        return convert_str_to_tsobject(ts, tz, unit, dayfirst, yearfirst)

    if checknull_with_nat_and_na(ts):
        obj.value = NPY_NAT
    elif is_datetime64_object(ts):
        reso = get_supported_reso(get_datetime64_unit(ts))
        obj.creso = reso
        obj.value = get_datetime64_nanos(ts, reso)
        if obj.value != NPY_NAT:
            pandas_datetime_to_datetimestruct(obj.value, reso, &obj.dts)
    elif is_integer_object(ts):
        try:
            ts = <int64_t>ts
        except OverflowError:
            # GH#26651 re-raise as OutOfBoundsDatetime
            raise OutOfBoundsDatetime(f"Out of bounds nanosecond timestamp {ts}")
        if ts == NPY_NAT:
            obj.value = NPY_NAT
        else:
            if unit is None:
                unit = "ns"
            in_reso = abbrev_to_npy_unit(unit)
            reso = get_supported_reso(in_reso)
            ts = cast_from_unit(ts, unit, reso)
            obj.value = ts
            obj.creso = reso
            pandas_datetime_to_datetimestruct(ts, reso, &obj.dts)
    elif is_float_object(ts):
        if ts != ts or ts == NPY_NAT:
            obj.value = NPY_NAT
        else:
            ts = cast_from_unit(ts, unit)
            obj.value = ts
            pandas_datetime_to_datetimestruct(ts, NPY_FR_ns, &obj.dts)
    elif PyDateTime_Check(ts):
        if nanos == 0:
            if isinstance(ts, ABCTimestamp):
                reso = abbrev_to_npy_unit(ts.unit)  # TODO: faster way to do this?
            else:
                # TODO: what if user explicitly passes nanos=0?
                reso = NPY_FR_us
        else:
            reso = NPY_FR_ns
        return convert_datetime_to_tsobject(ts, tz, nanos, reso=reso)
    elif PyDate_Check(ts):
        # Keep the converter same as PyDateTime's
        # For date object we give the lowest supported resolution, i.e. "s"
        ts = datetime.combine(ts, time())
        return convert_datetime_to_tsobject(
            ts, tz, nanos=0, reso=NPY_DATETIMEUNIT.NPY_FR_s
        )
    else:
        from .period import Period
        if isinstance(ts, Period):
            raise ValueError("Cannot convert Period to Timestamp "
                             "unambiguously. Use to_timestamp")
        raise TypeError(f"Cannot convert input [{ts}] of type {type(ts)} to "
                        f"Timestamp")

    maybe_localize_tso(obj, tz, obj.creso)
    return obj


cdef maybe_localize_tso(_TSObject obj, tzinfo tz, NPY_DATETIMEUNIT reso):
    if tz is not None:
        _localize_tso(obj, tz, reso)

    if obj.value != NPY_NAT:
        # check_overflows needs to run after _localize_tso
        check_dts_bounds(&obj.dts, reso)
        check_overflows(obj, reso)


cdef _TSObject convert_datetime_to_tsobject(
    datetime ts,
    tzinfo tz,
    int32_t nanos=0,
    NPY_DATETIMEUNIT reso=NPY_FR_ns,
):
    """
    Convert a datetime (or Timestamp) input `ts`, along with optional timezone
    object `tz` to a _TSObject.

    The optional argument `nanos` allows for cases where datetime input
    needs to be supplemented with higher-precision information.

    Parameters
    ----------
    ts : datetime or Timestamp
        Value to be converted to _TSObject
    tz : tzinfo or None
        timezone for the timezone-aware output
    nanos : int32_t, default is 0
        nanoseconds supplement the precision of the datetime input ts
    reso : NPY_DATETIMEUNIT, default NPY_FR_ns

    Returns
    -------
    obj : _TSObject
    """
    cdef:
        _TSObject obj = _TSObject()
        int64_t pps

    obj.creso = reso
    obj.fold = ts.fold
    if tz is not None:

        if ts.tzinfo is not None:
            # Convert the current timezone to the passed timezone
            ts = ts.astimezone(tz)
            pydatetime_to_dtstruct(ts, &obj.dts)
            obj.tzinfo = ts.tzinfo
        elif not is_utc(tz):
            ts = _localize_pydatetime(ts, tz)
            pydatetime_to_dtstruct(ts, &obj.dts)
            obj.tzinfo = ts.tzinfo
        else:
            # UTC
            pydatetime_to_dtstruct(ts, &obj.dts)
            obj.tzinfo = tz
    else:
        pydatetime_to_dtstruct(ts, &obj.dts)
        obj.tzinfo = ts.tzinfo

    if isinstance(ts, ABCTimestamp):
        obj.dts.ps = ts.nanosecond * 1000

    if nanos:
        obj.dts.ps = nanos * 1000

    obj.value = npy_datetimestruct_to_datetime(reso, &obj.dts)

    if obj.tzinfo is not None and not is_utc(obj.tzinfo):
        offset = get_utcoffset(obj.tzinfo, ts)
        pps = periods_per_second(reso)
        obj.value -= int(offset.total_seconds() * pps)

    check_dts_bounds(&obj.dts, reso)
    check_overflows(obj, reso)
    return obj


cdef _TSObject _create_tsobject_tz_using_offset(npy_datetimestruct dts,
                                                int tzoffset, tzinfo tz=None,
                                                NPY_DATETIMEUNIT reso=NPY_FR_ns):
    """
    Convert a datetimestruct `dts`, along with initial timezone offset
    `tzoffset` to a _TSObject (with timezone object `tz` - optional).

    Parameters
    ----------
    dts : npy_datetimestruct
    tzoffset : int
    tz : tzinfo or None
        timezone for the timezone-aware output.
    reso : NPY_DATETIMEUNIT, default NPY_FR_ns

    Returns
    -------
    obj : _TSObject
    """
    cdef:
        _TSObject obj = _TSObject()
        int64_t value  # numpy dt64
        datetime dt
        Py_ssize_t pos

    value = npy_datetimestruct_to_datetime(reso, &dts)
    obj.dts = dts
    obj.tzinfo = timezone(timedelta(minutes=tzoffset))
    obj.value = tz_localize_to_utc_single(
        value, obj.tzinfo, ambiguous=None, nonexistent=None, creso=reso
    )
    obj.creso = reso
    if tz is None:
        check_overflows(obj, reso)
        return obj

    cdef:
        Localizer info = Localizer(tz, reso)

    # Infer fold from offset-adjusted obj.value
    # see PEP 495 https://www.python.org/dev/peps/pep-0495/#the-fold-attribute
    if info.use_utc:
        pass
    elif info.use_tzlocal:
        info.utc_val_to_local_val(obj.value, &pos, &obj.fold)
    elif info.use_dst and not info.use_pytz:
        # i.e. dateutil
        info.utc_val_to_local_val(obj.value, &pos, &obj.fold)

    # Keep the converter same as PyDateTime's
    dt = datetime(obj.dts.year, obj.dts.month, obj.dts.day,
                  obj.dts.hour, obj.dts.min, obj.dts.sec,
                  obj.dts.us, obj.tzinfo, fold=obj.fold)
    obj = convert_datetime_to_tsobject(
        dt, tz, nanos=obj.dts.ps // 1000)
    obj.ensure_reso(reso)  # TODO: more performant to get reso right up front?
    return obj


cdef _TSObject convert_str_to_tsobject(str ts, tzinfo tz, str unit,
                                       bint dayfirst=False,
                                       bint yearfirst=False):
    """
    Convert a string input `ts`, along with optional timezone object`tz`
    to a _TSObject.

    The optional arguments `dayfirst` and `yearfirst` are passed to the
    dateutil parser.

    Parameters
    ----------
    ts : str
        Value to be converted to _TSObject
    tz : tzinfo or None
        timezone for the timezone-aware output
    unit : str or None
    dayfirst : bool, default False
        When parsing an ambiguous date string, interpret e.g. "3/4/1975" as
        April 3, as opposed to the standard US interpretation March 4.
    yearfirst : bool, default False
        When parsing an ambiguous date string, interpret e.g. "01/05/09"
        as "May 9, 2001", as opposed to the default "Jan 5, 2009"

    Returns
    -------
    obj : _TSObject
    """
    cdef:
        npy_datetimestruct dts
        int out_local = 0, out_tzoffset = 0, string_to_dts_failed
        datetime dt
        int64_t ival
        NPY_DATETIMEUNIT out_bestunit, reso

    if len(ts) == 0 or ts in nat_strings:
        obj = _TSObject()
        obj.value = NPY_NAT
        obj.tzinfo = tz
        return obj
    elif ts == "now":
        # Issue 9000, we short-circuit rather than going
        # into np_datetime_strings which returns utc
        dt = datetime.now(tz)
    elif ts == "today":
        # Issue 9000, we short-circuit rather than going
        # into np_datetime_strings which returns a normalized datetime
        dt = datetime.now(tz)
        # equiv: datetime.today().replace(tzinfo=tz)
    else:
        string_to_dts_failed = string_to_dts(
            ts, &dts, &out_bestunit, &out_local,
            &out_tzoffset, False
        )
        if not string_to_dts_failed:
            reso = get_supported_reso(out_bestunit)
            check_dts_bounds(&dts, reso)
            if out_local == 1:
                return _create_tsobject_tz_using_offset(
                    dts, out_tzoffset, tz, reso
                )
            else:
                ival = npy_datetimestruct_to_datetime(reso, &dts)
                if tz is not None:
                    # shift for _localize_tso
                    ival = tz_localize_to_utc_single(
                        ival, tz, ambiguous="raise", nonexistent=None, creso=reso
                    )
                obj = _TSObject()
                obj.dts = dts
                obj.value = ival
                obj.creso = reso
                maybe_localize_tso(obj, tz, obj.creso)
                return obj

        dt = parse_datetime_string(
            ts, dayfirst=dayfirst, yearfirst=yearfirst, out_bestunit=&out_bestunit
        )
        reso = get_supported_reso(out_bestunit)
        return convert_datetime_to_tsobject(dt, tz, nanos=0, reso=reso)

    return convert_datetime_to_tsobject(dt, tz)


cdef check_overflows(_TSObject obj, NPY_DATETIMEUNIT reso=NPY_FR_ns):
    """
    Check that we haven't silently overflowed in timezone conversion

    Parameters
    ----------
    obj : _TSObject
    reso : NPY_DATETIMEUNIT, default NPY_FR_ns

    Returns
    -------
    None

    Raises
    ------
    OutOfBoundsDatetime
    """
    # GH#12677
    cdef:
        npy_datetimestruct lb, ub

    get_implementation_bounds(reso, &lb, &ub)

    if obj.dts.year == lb.year:
        if not (obj.value < 0):
            from pandas._libs.tslibs.timestamps import Timestamp
            fmt = (f"{obj.dts.year}-{obj.dts.month:02d}-{obj.dts.day:02d} "
                   f"{obj.dts.hour:02d}:{obj.dts.min:02d}:{obj.dts.sec:02d}")
            raise OutOfBoundsDatetime(
                f"Converting {fmt} underflows past {Timestamp.min}"
            )
    elif obj.dts.year == ub.year:
        if not (obj.value > 0):
            from pandas._libs.tslibs.timestamps import Timestamp
            fmt = (f"{obj.dts.year}-{obj.dts.month:02d}-{obj.dts.day:02d} "
                   f"{obj.dts.hour:02d}:{obj.dts.min:02d}:{obj.dts.sec:02d}")
            raise OutOfBoundsDatetime(
                f"Converting {fmt} overflows past {Timestamp.max}"
            )

# ----------------------------------------------------------------------
# Localization

cdef void _localize_tso(_TSObject obj, tzinfo tz, NPY_DATETIMEUNIT reso) noexcept:
    """
    Given the UTC nanosecond timestamp in obj.value, find the wall-clock
    representation of that timestamp in the given timezone.

    Parameters
    ----------
    obj : _TSObject
    tz : tzinfo
    reso : NPY_DATETIMEUNIT

    Returns
    -------
    None

    Notes
    -----
    Sets obj.tzinfo inplace, alters obj.dts inplace.
    """
    cdef:
        int64_t local_val
        Py_ssize_t outpos = -1
        Localizer info = Localizer(tz, reso)

    assert obj.tzinfo is None

    if info.use_utc:
        pass
    elif obj.value == NPY_NAT:
        pass
    else:
        local_val = info.utc_val_to_local_val(obj.value, &outpos, &obj.fold)

        if info.use_pytz:
            # infer we went through a pytz path, will have outpos!=-1
            tz = tz._tzinfos[tz._transition_info[outpos]]

        pandas_datetime_to_datetimestruct(local_val, reso, &obj.dts)

    obj.tzinfo = tz


cdef datetime _localize_pydatetime(datetime dt, tzinfo tz):
    """
    Take a datetime/Timestamp in UTC and localizes to timezone tz.

    NB: Unlike the public version, this treats datetime and Timestamp objects
        identically, i.e. discards nanos from Timestamps.
        It also assumes that the `tz` input is not None.
    """
    try:
        # datetime.replace with pytz may be incorrect result
        return tz.localize(dt)
    except AttributeError:
        return dt.replace(tzinfo=tz)


cpdef inline datetime localize_pydatetime(datetime dt, tzinfo tz):
    """
    Take a datetime/Timestamp in UTC and localizes to timezone tz.

    Parameters
    ----------
    dt : datetime or Timestamp
    tz : tzinfo or None

    Returns
    -------
    localized : datetime or Timestamp
    """
    if tz is None:
        return dt
    elif isinstance(dt, ABCTimestamp):
        return dt.tz_localize(tz)
    return _localize_pydatetime(dt, tz)


cdef tzinfo convert_timezone(
    tzinfo tz_in,
    tzinfo tz_out,
    bint found_naive,
    bint found_tz,
    bint utc_convert,
):
    """
    Validate that ``tz_in`` can be converted/localized to ``tz_out``.

    Parameters
    ----------
    tz_in : tzinfo or None
        Timezone info of element being processed.
    tz_out : tzinfo or None
        Timezone info of output.
    found_naive : bool
        Whether a timezone-naive element has been found so far.
    found_tz : bool
        Whether a timezone-aware element has been found so far.
    utc_convert : bool
        Whether to convert/localize to UTC.

    Returns
    -------
    tz_info
        Timezone info of output.

    Raises
    ------
    ValueError
        If ``tz_in`` can't be converted/localized to ``tz_out``.
    """
    if tz_in is not None:
        if utc_convert:
            pass
        elif found_naive:
            raise ValueError("Tz-aware datetime.datetime "
                             "cannot be converted to "
                             "datetime64 unless utc=True")
        elif tz_out is not None and not tz_compare(tz_out, tz_in):
            raise ValueError("Tz-aware datetime.datetime "
                             "cannot be converted to "
                             "datetime64 unless utc=True")
        else:
            tz_out = tz_in
    else:
        if found_tz and not utc_convert:
            raise ValueError("Cannot mix tz-aware with "
                             "tz-naive values")
    return tz_out


cdef int64_t parse_pydatetime(
    datetime val,
    npy_datetimestruct *dts,
    bint utc_convert,
) except? -1:
    """
    Convert pydatetime to datetime64.

    Parameters
    ----------
    val : datetime
        Element being processed.
    dts : *npy_datetimestruct
        Needed to use in pydatetime_to_dt64, which writes to it.
    utc_convert : bool
        Whether to convert/localize to UTC.

    Raises
    ------
    OutOfBoundsDatetime
    """
    cdef:
        _TSObject _ts
        int64_t result

    if val.tzinfo is not None:
        if utc_convert:
            _ts = convert_datetime_to_tsobject(val, None)
            _ts.ensure_reso(NPY_FR_ns)
            result = _ts.value
        else:
            _ts = convert_datetime_to_tsobject(val, None)
            _ts.ensure_reso(NPY_FR_ns)
            result = _ts.value
    else:
        if isinstance(val, ABCTimestamp):
            result = val.as_unit("ns")._value
        else:
            result = pydatetime_to_dt64(val, dts)
            check_dts_bounds(dts)
    return result
