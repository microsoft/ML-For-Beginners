"""Strptime-related classes and functions.

TimeRE, _calc_julian_from_U_or_W are vendored
from the standard library, see
https://github.com/python/cpython/blob/main/Lib/_strptime.py
The original module-level docstring follows.

Strptime-related classes and functions.
CLASSES:
    LocaleTime -- Discovers and stores locale-specific time information
    TimeRE -- Creates regexes for pattern matching a string of text containing
                time information
FUNCTIONS:
    _getlang -- Figure out what language is being used for the locale
    strptime -- Calculates the time struct represented by the passed-in string
"""
from datetime import timezone

from cpython.datetime cimport (
    PyDate_Check,
    PyDateTime_Check,
    date,
    import_datetime,
    timedelta,
    tzinfo,
)
from _strptime import (
    TimeRE as _TimeRE,
    _getlang,
)
from _strptime import LocaleTime  # no-cython-lint

import_datetime()

from _thread import allocate_lock as _thread_allocate_lock
import re

import numpy as np
import pytz

cimport numpy as cnp
from numpy cimport (
    int64_t,
    ndarray,
)

from pandas._libs.missing cimport checknull_with_nat_and_na
from pandas._libs.tslibs.conversion cimport (
    convert_timezone,
    get_datetime64_nanos,
)
from pandas._libs.tslibs.nattype cimport (
    NPY_NAT,
    c_nat_strings as nat_strings,
)
from pandas._libs.tslibs.np_datetime cimport (
    NPY_DATETIMEUNIT,
    NPY_FR_ns,
    check_dts_bounds,
    import_pandas_datetime,
    npy_datetimestruct,
    npy_datetimestruct_to_datetime,
    pydate_to_dt64,
    pydatetime_to_dt64,
    string_to_dts,
)

import_pandas_datetime()

from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime
from pandas._libs.tslibs.timestamps cimport _Timestamp
from pandas._libs.util cimport (
    is_datetime64_object,
    is_float_object,
    is_integer_object,
)

from pandas._libs.tslibs.timestamps import Timestamp

cnp.import_array()

cdef bint format_is_iso(f: str):
    """
    Does format match the iso8601 set that can be handled by the C parser?
    Generally of form YYYY-MM-DDTHH:MM:SS - date separator can be different
    but must be consistent.  Leading 0s in dates and times are optional.
    """
    iso_regex = re.compile(
        r"""
        ^                     # start of string
        %Y                    # Year
        (?:([-/ \\.]?)%m      # month with or without separators
        (?: \1%d              # day with same separator as for year-month
        (?:[ T]%H             # hour with separator
        (?:\:%M               # minute with separator
        (?:\:%S               # second with separator
        (?:%z|\.%f(?:%z)?     # timezone or fractional second
        )?)?)?)?)?)?          # optional
        $                     # end of string
        """,
        re.VERBOSE,
    )
    excluded_formats = ["%Y%m"]
    return re.match(iso_regex, f) is not None and f not in excluded_formats


def _test_format_is_iso(f: str) -> bool:
    """Only used in testing."""
    return format_is_iso(f)


cdef bint parse_today_now(str val, int64_t* iresult, bint utc):
    # We delay this check for as long as possible
    # because it catches relatively rare cases

    # Multiply by 1000 to convert to nanos, since these methods naturally have
    #  microsecond resolution
    if val == "now":
        if utc:
            iresult[0] = Timestamp.utcnow()._value * 1000
        else:
            # GH#18705 make sure to_datetime("now") matches Timestamp("now")
            # Note using Timestamp.now() is faster than Timestamp("now")
            iresult[0] = Timestamp.now()._value * 1000
        return True
    elif val == "today":
        iresult[0] = Timestamp.today()._value * 1000
        return True
    return False

cdef dict _parse_code_table = {"y": 0,
                               "Y": 1,
                               "m": 2,
                               "B": 3,
                               "b": 4,
                               "d": 5,
                               "H": 6,
                               "I": 7,
                               "M": 8,
                               "S": 9,
                               "f": 10,
                               "A": 11,
                               "a": 12,
                               "w": 13,
                               "j": 14,
                               "U": 15,
                               "W": 16,
                               "Z": 17,
                               "p": 18,  # an additional key, only with I
                               "z": 19,
                               "G": 20,
                               "V": 21,
                               "u": 22}


def array_strptime(
    ndarray[object] values,
    str fmt,
    bint exact=True,
    errors="raise",
    bint utc=False,
):
    """
    Calculates the datetime structs represented by the passed array of strings

    Parameters
    ----------
    values : ndarray of string-like objects
    fmt : string-like regex
    exact : matches must be exact if True, search if False
    errors : string specifying error handling, {'raise', 'ignore', 'coerce'}
    """

    cdef:
        Py_ssize_t i, n = len(values)
        npy_datetimestruct dts
        int64_t[::1] iresult
        object[::1] result_timezone
        int year, month, day, minute, hour, second, weekday, julian
        int week_of_year, week_of_year_start, parse_code, ordinal
        int iso_week, iso_year
        int64_t us, ns
        object val, group_key, ampm, found, tz
        bint is_raise = errors=="raise"
        bint is_ignore = errors=="ignore"
        bint is_coerce = errors=="coerce"
        bint found_naive = False
        bint found_tz = False
        tzinfo tz_out = None
        bint iso_format = format_is_iso(fmt)
        NPY_DATETIMEUNIT out_bestunit
        int out_local = 0, out_tzoffset = 0
        bint string_to_dts_succeeded = 0

    assert is_raise or is_ignore or is_coerce

    if "%W" in fmt or "%U" in fmt:
        if "%Y" not in fmt and "%y" not in fmt:
            raise ValueError("Cannot use '%W' or '%U' without day and year")
        if "%A" not in fmt and "%a" not in fmt and "%w" not in fmt:
            raise ValueError("Cannot use '%W' or '%U' without day and year")
    elif "%Z" in fmt and "%z" in fmt:
        raise ValueError("Cannot parse both %Z and %z")
    elif "%j" in fmt and "%G" in fmt:
        raise ValueError("Day of the year directive '%j' is not "
                         "compatible with ISO year directive '%G'. "
                         "Use '%Y' instead.")
    elif "%G" in fmt and (
        "%V" not in fmt
        or not (
            "%A" in fmt
            or "%a" in fmt
            or "%w" in fmt
            or "%u" in fmt
        )
    ):
        raise ValueError("ISO year directive '%G' must be used with "
                         "the ISO week directive '%V' and a weekday "
                         "directive '%A', '%a', '%w', or '%u'.")
    elif "%V" in fmt and "%Y" in fmt:
        raise ValueError("ISO week directive '%V' is incompatible with "
                         "the year directive '%Y'. Use the ISO year "
                         "'%G' instead.")
    elif "%V" in fmt and (
        "%G" not in fmt
        or not (
            "%A" in fmt
            or "%a" in fmt
            or "%w" in fmt
            or "%u" in fmt
        )
    ):
        raise ValueError("ISO week directive '%V' must be used with "
                         "the ISO year directive '%G' and a weekday "
                         "directive '%A', '%a', '%w', or '%u'.")

    global _TimeRE_cache, _regex_cache
    with _cache_lock:
        if _getlang() != _TimeRE_cache.locale_time.lang:
            _TimeRE_cache = TimeRE()
            _regex_cache.clear()
        if len(_regex_cache) > _CACHE_MAX_SIZE:
            _regex_cache.clear()
        locale_time = _TimeRE_cache.locale_time
        format_regex = _regex_cache.get(fmt)
        if not format_regex:
            try:
                format_regex = _TimeRE_cache.compile(fmt)
            # KeyError raised when a bad format is found; can be specified as
            # \\, in which case it was a stray % but with a space after it
            except KeyError, err:
                bad_directive = err.args[0]
                if bad_directive == "\\":
                    bad_directive = "%"
                del err
                raise ValueError(f"'{bad_directive}' is a bad directive "
                                 f"in format '{fmt}'")
            # IndexError only occurs when the format string is "%"
            except IndexError:
                raise ValueError(f"stray % in format '{fmt}'")
            _regex_cache[fmt] = format_regex

    result = np.empty(n, dtype="M8[ns]")
    iresult = result.view("i8")
    result_timezone = np.empty(n, dtype="object")

    dts.us = dts.ps = dts.as = 0

    for i in range(n):
        val = values[i]
        try:
            if isinstance(val, str):
                if len(val) == 0 or val in nat_strings:
                    iresult[i] = NPY_NAT
                    continue
            elif checknull_with_nat_and_na(val):
                iresult[i] = NPY_NAT
                continue
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
                    utc,
                )
                if isinstance(val, _Timestamp):
                    iresult[i] = val.tz_localize(None).as_unit("ns")._value
                else:
                    iresult[i] = pydatetime_to_dt64(val.replace(tzinfo=None), &dts)
                    check_dts_bounds(&dts)
                result_timezone[i] = val.tzinfo
                continue
            elif PyDate_Check(val):
                iresult[i] = pydate_to_dt64(val, &dts)
                check_dts_bounds(&dts)
                continue
            elif is_datetime64_object(val):
                iresult[i] = get_datetime64_nanos(val, NPY_FR_ns)
                continue
            elif (
                    (is_integer_object(val) or is_float_object(val))
                    and (val != val or val == NPY_NAT)
            ):
                iresult[i] = NPY_NAT
                continue
            else:
                val = str(val)

            if fmt == "ISO8601":
                string_to_dts_succeeded = not string_to_dts(
                    val, &dts, &out_bestunit, &out_local,
                    &out_tzoffset, False, None, False
                )
            elif iso_format:
                string_to_dts_succeeded = not string_to_dts(
                    val, &dts, &out_bestunit, &out_local,
                    &out_tzoffset, False, fmt, exact
                )
            if string_to_dts_succeeded:
                # No error reported by string_to_dts, pick back up
                # where we left off
                value = npy_datetimestruct_to_datetime(NPY_FR_ns, &dts)
                if out_local == 1:
                    # Store the out_tzoffset in seconds
                    # since we store the total_seconds of
                    # dateutil.tz.tzoffset objects
                    tz = timezone(timedelta(minutes=out_tzoffset))
                    result_timezone[i] = tz
                    out_local = 0
                    out_tzoffset = 0
                iresult[i] = value
                check_dts_bounds(&dts)
                continue

            if parse_today_now(val, &iresult[i], utc):
                continue

            # Some ISO formats can't be parsed by string_to_dts
            # For example, 6-digit YYYYMD. So, if there's an error, and a format
            # was specified, then try the string-matching code below. If the format
            # specified was 'ISO8601', then we need to error, because
            # only string_to_dts handles mixed ISO8601 formats.
            if not string_to_dts_succeeded and fmt == "ISO8601":
                raise ValueError(f"Time data {val} is not ISO8601 format")

            # exact matching
            if exact:
                found = format_regex.match(val)
                if not found:
                    raise ValueError(
                        f"time data \"{val}\" doesn't match format \"{fmt}\""
                    )
                if len(val) != found.end():
                    raise ValueError(
                        "unconverted data remains when parsing with "
                        f"format \"{fmt}\": \"{val[found.end():]}\""
                    )

            # search
            else:
                found = format_regex.search(val)
                if not found:
                    raise ValueError(
                        f"time data \"{val}\" doesn't match format \"{fmt}\""
                    )

            iso_year = -1
            year = 1900
            month = day = 1
            hour = minute = second = ns = us = 0
            tz = None
            # Default to -1 to signify that values not known; not critical to have,
            # though
            iso_week = week_of_year = -1
            week_of_year_start = -1
            # weekday and julian defaulted to -1 so as to signal need to calculate
            # values
            weekday = julian = -1
            found_dict = found.groupdict()
            for group_key in found_dict.iterkeys():
                # Directives not explicitly handled below:
                #   c, x, X
                #      handled by making out of other directives
                #   U, W
                #      worthless without day of the week
                parse_code = _parse_code_table[group_key]

                if parse_code == 0:
                    year = int(found_dict["y"])
                    # Open Group specification for strptime() states that a %y
                    # value in the range of [00, 68] is in the century 2000, while
                    # [69,99] is in the century 1900
                    if year <= 68:
                        year += 2000
                    else:
                        year += 1900
                elif parse_code == 1:
                    year = int(found_dict["Y"])
                elif parse_code == 2:
                    month = int(found_dict["m"])
                # elif group_key == 'B':
                elif parse_code == 3:
                    month = locale_time.f_month.index(found_dict["B"].lower())
                # elif group_key == 'b':
                elif parse_code == 4:
                    month = locale_time.a_month.index(found_dict["b"].lower())
                # elif group_key == 'd':
                elif parse_code == 5:
                    day = int(found_dict["d"])
                # elif group_key == 'H':
                elif parse_code == 6:
                    hour = int(found_dict["H"])
                elif parse_code == 7:
                    hour = int(found_dict["I"])
                    ampm = found_dict.get("p", "").lower()
                    # If there was no AM/PM indicator, we'll treat this like AM
                    if ampm in ("", locale_time.am_pm[0]):
                        # We're in AM so the hour is correct unless we're
                        # looking at 12 midnight.
                        # 12 midnight == 12 AM == hour 0
                        if hour == 12:
                            hour = 0
                    elif ampm == locale_time.am_pm[1]:
                        # We're in PM so we need to add 12 to the hour unless
                        # we're looking at 12 noon.
                        # 12 noon == 12 PM == hour 12
                        if hour != 12:
                            hour += 12
                elif parse_code == 8:
                    minute = int(found_dict["M"])
                elif parse_code == 9:
                    second = int(found_dict["S"])
                elif parse_code == 10:
                    s = found_dict["f"]
                    # Pad to always return nanoseconds
                    s += "0" * (9 - len(s))
                    us = long(s)
                    ns = us % 1000
                    us = us // 1000
                elif parse_code == 11:
                    weekday = locale_time.f_weekday.index(found_dict["A"].lower())
                elif parse_code == 12:
                    weekday = locale_time.a_weekday.index(found_dict["a"].lower())
                elif parse_code == 13:
                    weekday = int(found_dict["w"])
                    if weekday == 0:
                        weekday = 6
                    else:
                        weekday -= 1
                elif parse_code == 14:
                    julian = int(found_dict["j"])
                elif parse_code == 15 or parse_code == 16:
                    week_of_year = int(found_dict[group_key])
                    if group_key == "U":
                        # U starts week on Sunday.
                        week_of_year_start = 6
                    else:
                        # W starts week on Monday.
                        week_of_year_start = 0
                elif parse_code == 17:
                    tz = pytz.timezone(found_dict["Z"])
                elif parse_code == 19:
                    tz = parse_timezone_directive(found_dict["z"])
                elif parse_code == 20:
                    iso_year = int(found_dict["G"])
                elif parse_code == 21:
                    iso_week = int(found_dict["V"])
                elif parse_code == 22:
                    weekday = int(found_dict["u"])
                    weekday -= 1

            # If we know the wk of the year and what day of that wk, we can figure
            # out the Julian day of the year.
            if julian == -1 and weekday != -1:
                if week_of_year != -1:
                    week_starts_Mon = week_of_year_start == 0
                    julian = _calc_julian_from_U_or_W(year, week_of_year, weekday,
                                                      week_starts_Mon)
                elif iso_year != -1 and iso_week != -1:
                    year, julian = _calc_julian_from_V(iso_year, iso_week,
                                                       weekday + 1)
            # Cannot pre-calculate date() since can change in Julian
            # calculation and thus could have different value for the day of the wk
            # calculation.
            if julian == -1:
                # Need to add 1 to result since first day of the year is 1, not
                # 0.
                ordinal = date(year, month, day).toordinal()
                julian = ordinal - date(year, 1, 1).toordinal() + 1
            else:
                # Assume that if they bothered to include Julian day it will
                # be accurate.
                datetime_result = date.fromordinal(
                    (julian - 1) + date(year, 1, 1).toordinal())
                year = datetime_result.year
                month = datetime_result.month
                day = datetime_result.day
            if weekday == -1:
                weekday = date(year, month, day).weekday()

            dts.year = year
            dts.month = month
            dts.day = day
            dts.hour = hour
            dts.min = minute
            dts.sec = second
            dts.us = us
            dts.ps = ns * 1000

            iresult[i] = npy_datetimestruct_to_datetime(NPY_FR_ns, &dts)
            check_dts_bounds(&dts)

            result_timezone[i] = tz

        except (ValueError, OutOfBoundsDatetime) as ex:
            ex.args = (
                f"{str(ex)}, at position {i}. You might want to try:\n"
                "    - passing `format` if your strings have a consistent format;\n"
                "    - passing `format='ISO8601'` if your strings are "
                "all ISO8601 but not necessarily in exactly the same format;\n"
                "    - passing `format='mixed'`, and the format will be "
                "inferred for each element individually. "
                "You might want to use `dayfirst` alongside this.",
            )
            if is_coerce:
                iresult[i] = NPY_NAT
                continue
            elif is_raise:
                raise
            return values, []

    return result, result_timezone.base


class TimeRE(_TimeRE):
    """
    Handle conversion from format directives to regexes.

    Creates regexes for pattern matching a string of text containing
    time information
    """

    def __init__(self, locale_time=None):
        """
        Create keys/values.

        Order of execution is important for dependency reasons.
        """
        self._Z = None
        super().__init__(locale_time=locale_time)
        # GH 48767: Overrides for cpython's TimeRE
        #  1) Parse up to nanos instead of micros
        self.update({"f": r"(?P<f>[0-9]{1,9})"}),

    def __getitem__(self, key):
        if key == "Z":
            # lazy computation
            if self._Z is None:
                self._Z = self.__seqToRE(pytz.all_timezones, "Z")
            # Note: handling Z is the key difference vs using the stdlib
            # _strptime.TimeRE. test_to_datetime_parse_tzname_or_tzoffset with
            # fmt='%Y-%m-%d %H:%M:%S %Z' fails with the stdlib version.
            return self._Z
        return super().__getitem__(key)


_cache_lock = _thread_allocate_lock()
# DO NOT modify _TimeRE_cache or _regex_cache without acquiring the cache lock
# first!
_TimeRE_cache = TimeRE()
_CACHE_MAX_SIZE = 5  # Max number of regexes stored in _regex_cache
_regex_cache = {}


cdef int _calc_julian_from_U_or_W(int year, int week_of_year,
                                  int day_of_week, int week_starts_Mon):
    """
    Calculate the Julian day based on the year, week of the year, and day of
    the week, with week_start_day representing whether the week of the year
    assumes the week starts on Sunday or Monday (6 or 0).

    Parameters
    ----------
    year : int
        the year
    week_of_year : int
        week taken from format U or W
    week_starts_Mon : int
        represents whether the week of the year
        assumes the week starts on Sunday or Monday (6 or 0)

    Returns
    -------
    int
        converted julian day
    """

    cdef:
        int first_weekday, week_0_length, days_to_week

    first_weekday = date(year, 1, 1).weekday()
    # If we are dealing with the %U directive (week starts on Sunday), it's
    # easier to just shift the view to Sunday being the first day of the
    # week.
    if not week_starts_Mon:
        first_weekday = (first_weekday + 1) % 7
        day_of_week = (day_of_week + 1) % 7

    # Need to watch out for a week 0 (when the first day of the year is not
    # the same as that specified by %U or %W).
    week_0_length = (7 - first_weekday) % 7
    if week_of_year == 0:
        return 1 + day_of_week - first_weekday
    else:
        days_to_week = week_0_length + (7 * (week_of_year - 1))
        return 1 + days_to_week + day_of_week


cdef (int, int) _calc_julian_from_V(int iso_year, int iso_week, int iso_weekday):
    """
    Calculate the Julian day based on the ISO 8601 year, week, and weekday.

    ISO weeks start on Mondays, with week 01 being the week containing 4 Jan.
    ISO week days range from 1 (Monday) to 7 (Sunday).

    Parameters
    ----------
    iso_year : int
        the year taken from format %G
    iso_week : int
        the week taken from format %V
    iso_weekday : int
        weekday taken from format %u

    Returns
    -------
    (int, int)
        the iso year and the Gregorian ordinal date / julian date
    """

    cdef:
        int correction, ordinal

    correction = date(iso_year, 1, 4).isoweekday() + 3
    ordinal = (iso_week * 7) + iso_weekday - correction
    # ordinal may be negative or 0 now, which means the date is in the previous
    # calendar year
    if ordinal < 1:
        ordinal += date(iso_year, 1, 1).toordinal()
        iso_year -= 1
        ordinal -= date(iso_year, 1, 1).toordinal()
    return iso_year, ordinal


cdef tzinfo parse_timezone_directive(str z):
    """
    Parse the '%z' directive and return a datetime.timezone object.

    Parameters
    ----------
    z : string of the UTC offset

    Returns
    -------
    datetime.timezone

    Notes
    -----
    This is essentially similar to the cpython implementation
    https://github.com/python/cpython/blob/master/Lib/_strptime.py#L457-L479
    """

    cdef:
        int hours, minutes, seconds, pad_number, microseconds
        int total_minutes
        object gmtoff_remainder, gmtoff_remainder_padding

    if z == "Z":
        return timezone(timedelta(0))
    if z[3] == ":":
        z = z[:3] + z[4:]
        if len(z) > 5:
            if z[5] != ":":
                raise ValueError(f"Inconsistent use of : in {z}")
            z = z[:5] + z[6:]
    hours = int(z[1:3])
    minutes = int(z[3:5])
    seconds = int(z[5:7] or 0)

    # Pad to always return microseconds.
    gmtoff_remainder = z[8:]
    pad_number = 6 - len(gmtoff_remainder)
    gmtoff_remainder_padding = "0" * pad_number
    microseconds = int(gmtoff_remainder + gmtoff_remainder_padding)

    total_minutes = ((hours * 60) + minutes + (seconds // 60) +
                     (microseconds // 60_000_000))
    total_minutes = -total_minutes if z.startswith("-") else total_minutes
    return timezone(timedelta(minutes=total_minutes))
