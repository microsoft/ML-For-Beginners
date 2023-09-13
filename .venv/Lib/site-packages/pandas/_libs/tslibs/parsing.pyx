"""
Parsing functions for datetime and datetime-like strings.
"""
import re
import time
import warnings

from pandas.util._exceptions import find_stack_level

cimport cython
from cpython.datetime cimport (
    datetime,
    datetime_new,
    import_datetime,
    timedelta,
    tzinfo,
)

from datetime import timezone

from cpython.object cimport PyObject_Str
from cython cimport Py_ssize_t
from libc.string cimport strchr

import_datetime()

import numpy as np

cimport numpy as cnp
from numpy cimport (
    PyArray_GETITEM,
    PyArray_ITER_DATA,
    PyArray_ITER_NEXT,
    PyArray_IterNew,
    flatiter,
    float64_t,
)

cnp.import_array()

# dateutil compat

from decimal import InvalidOperation

from dateutil.parser import DEFAULTPARSER
from dateutil.tz import (
    tzlocal as _dateutil_tzlocal,
    tzoffset,
    tzutc as _dateutil_tzutc,
)

from pandas._config import get_option

from pandas._libs.tslibs.ccalendar cimport c_MONTH_NUMBERS
from pandas._libs.tslibs.dtypes cimport (
    attrname_to_npy_unit,
    npy_unit_to_attrname,
)
from pandas._libs.tslibs.nattype cimport (
    c_NaT as NaT,
    c_nat_strings as nat_strings,
)

from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime

from pandas._libs.tslibs.np_datetime cimport (
    NPY_DATETIMEUNIT,
    import_pandas_datetime,
    npy_datetimestruct,
    string_to_dts,
)

import_pandas_datetime()

from pandas._libs.tslibs.strptime import array_strptime

from pandas._libs.tslibs.util cimport (
    get_c_string_buf_and_size,
    is_array,
)


cdef extern from "pandas/portable.h":
    int getdigit_ascii(char c, int default) nogil

cdef extern from "pandas/parser/tokenizer.h":
    double xstrtod(const char *p, char **q, char decimal, char sci, char tsep,
                   int skip_trailing, int *error, int *maybe_int)


# ----------------------------------------------------------------------
# Constants


class DateParseError(ValueError):
    pass


_DEFAULT_DATETIME = datetime(1, 1, 1).replace(hour=0, minute=0,
                                              second=0, microsecond=0)

cdef:
    set _not_datelike_strings = {"a", "A", "m", "M", "p", "P", "t", "T"}

    # _timestamp_units -> units that we round to nanos
    set _timestamp_units = {
        NPY_DATETIMEUNIT.NPY_FR_ns,
        NPY_DATETIMEUNIT.NPY_FR_ps,
        NPY_DATETIMEUNIT.NPY_FR_fs,
        NPY_DATETIMEUNIT.NPY_FR_as,
    }

# ----------------------------------------------------------------------
cdef:
    const char* delimiters = " /-."
    int MAX_DAYS_IN_MONTH = 31, MAX_MONTH = 12


cdef bint _is_delimiter(const char ch):
    return strchr(delimiters, ch) != NULL


cdef int _parse_1digit(const char* s):
    cdef int result = 0
    result += getdigit_ascii(s[0], -10) * 1
    return result


cdef int _parse_2digit(const char* s):
    cdef int result = 0
    result += getdigit_ascii(s[0], -10) * 10
    result += getdigit_ascii(s[1], -100) * 1
    return result


cdef int _parse_4digit(const char* s):
    cdef int result = 0
    result += getdigit_ascii(s[0], -10) * 1000
    result += getdigit_ascii(s[1], -100) * 100
    result += getdigit_ascii(s[2], -1000) * 10
    result += getdigit_ascii(s[3], -10000) * 1
    return result


cdef datetime _parse_delimited_date(
    str date_string, bint dayfirst, NPY_DATETIMEUNIT* out_bestunit
):
    """
    Parse special cases of dates: MM/DD/YYYY, DD/MM/YYYY, MM/YYYY.

    At the beginning function tries to parse date in MM/DD/YYYY format, but
    if month > 12 - in DD/MM/YYYY (`dayfirst == False`).
    With `dayfirst == True` function makes an attempt to parse date in
    DD/MM/YYYY, if an attempt is wrong - in DD/MM/YYYY

    For MM/DD/YYYY, DD/MM/YYYY: delimiter can be a space or one of /-.
    For MM/YYYY: delimiter can be a space or one of /-
    If `date_string` can't be converted to date, then function returns
    None, None

    Parameters
    ----------
    date_string : str
    dayfirst : bool
    out_bestunit : NPY_DATETIMEUNIT*
        For specifying identified resolution.

    Returns:
    --------
    datetime or None
    """
    cdef:
        const char* buf
        Py_ssize_t length
        int day = 1, month = 1, year
        bint can_swap = 0

    buf = get_c_string_buf_and_size(date_string, &length)
    if length == 10 and _is_delimiter(buf[2]) and _is_delimiter(buf[5]):
        # parsing MM?DD?YYYY and DD?MM?YYYY dates
        month = _parse_2digit(buf)
        day = _parse_2digit(buf + 3)
        year = _parse_4digit(buf + 6)
        out_bestunit[0] = NPY_DATETIMEUNIT.NPY_FR_D
        can_swap = 1
    elif length == 9 and _is_delimiter(buf[1]) and _is_delimiter(buf[4]):
        # parsing M?DD?YYYY and D?MM?YYYY dates
        month = _parse_1digit(buf)
        day = _parse_2digit(buf + 2)
        year = _parse_4digit(buf + 5)
        out_bestunit[0] = NPY_DATETIMEUNIT.NPY_FR_D
        can_swap = 1
    elif length == 9 and _is_delimiter(buf[2]) and _is_delimiter(buf[4]):
        # parsing MM?D?YYYY and DD?M?YYYY dates
        month = _parse_2digit(buf)
        day = _parse_1digit(buf + 3)
        year = _parse_4digit(buf + 5)
        out_bestunit[0] = NPY_DATETIMEUNIT.NPY_FR_D
        can_swap = 1
    elif length == 8 and _is_delimiter(buf[1]) and _is_delimiter(buf[3]):
        # parsing M?D?YYYY and D?M?YYYY dates
        month = _parse_1digit(buf)
        day = _parse_1digit(buf + 2)
        year = _parse_4digit(buf + 4)
        out_bestunit[0] = NPY_DATETIMEUNIT.NPY_FR_D
        can_swap = 1
    elif length == 7 and _is_delimiter(buf[2]):
        # parsing MM?YYYY dates
        if buf[2] == b".":
            # we cannot reliably tell whether e.g. 10.2010 is a float
            # or a date, thus we refuse to parse it here
            return None
        month = _parse_2digit(buf)
        year = _parse_4digit(buf + 3)
        out_bestunit[0] = NPY_DATETIMEUNIT.NPY_FR_M
    else:
        return None

    if month < 0 or day < 0 or year < 1000:
        # some part is not an integer, so
        # date_string can't be converted to date, above format
        return None

    if 1 <= month <= MAX_DAYS_IN_MONTH and 1 <= day <= MAX_DAYS_IN_MONTH \
            and (month <= MAX_MONTH or day <= MAX_MONTH):
        if (month > MAX_MONTH or (day <= MAX_MONTH and dayfirst)) and can_swap:
            day, month = month, day
        # In Python <= 3.6.0 there is no range checking for invalid dates
        # in C api, thus we call faster C version for 3.6.1 or newer
        return datetime_new(year, month, day, 0, 0, 0, 0, None)

    raise DateParseError(f"Invalid date specified ({month}/{day})")


cdef bint _does_string_look_like_time(str parse_string):
    """
    Checks whether given string is a time: it has to start either from
    H:MM or from HH:MM, and hour and minute values must be valid.

    Parameters
    ----------
    parse_string : str

    Returns:
    --------
    bool
        Whether given string is potentially a time.
    """
    cdef:
        const char* buf
        Py_ssize_t length
        int hour = -1, minute = -1

    buf = get_c_string_buf_and_size(parse_string, &length)
    if length >= 4:
        if buf[1] == b":":
            # h:MM format
            hour = getdigit_ascii(buf[0], -1)
            minute = _parse_2digit(buf + 2)
        elif buf[2] == b":":
            # HH:MM format
            hour = _parse_2digit(buf)
            minute = _parse_2digit(buf + 3)

    return 0 <= hour <= 23 and 0 <= minute <= 59


def py_parse_datetime_string(
    str date_string, bint dayfirst=False, bint yearfirst=False
):
    # Python-accessible version for testing (we can't just make
    #  parse_datetime_string cpdef bc it has a pointer argument)
    cdef:
        NPY_DATETIMEUNIT out_bestunit

    return parse_datetime_string(date_string, dayfirst, yearfirst, &out_bestunit)


cdef datetime parse_datetime_string(
    # NB: This will break with np.str_ (GH#32264) even though
    #  isinstance(npstrobj, str) evaluates to True, so caller must ensure
    #  the argument is *exactly* 'str'
    str date_string,
    bint dayfirst,
    bint yearfirst,
    NPY_DATETIMEUNIT* out_bestunit
):
    """
    Parse datetime string, only returns datetime.
    Also cares special handling matching time patterns.

    Returns
    -------
    datetime

    Notes
    -----
    Does not handle "today" or "now", which caller is responsible for handling.
    """

    cdef:
        datetime dt
        bint is_quarter = 0

    if not _does_string_look_like_datetime(date_string):
        raise ValueError(f'Given date string "{date_string}" not likely a datetime')

    if _does_string_look_like_time(date_string):
        # time without date e.g. "01:01:01.111"
        # use current datetime as default, not pass _DEFAULT_DATETIME
        default = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        dt = dateutil_parse(date_string, default=default,
                            dayfirst=dayfirst, yearfirst=yearfirst,
                            ignoretz=False, out_bestunit=out_bestunit)
        return dt

    dt = _parse_delimited_date(date_string, dayfirst, out_bestunit)
    if dt is not None:
        return dt

    try:
        dt = _parse_dateabbr_string(
            date_string, _DEFAULT_DATETIME, None, out_bestunit, &is_quarter
        )
        return dt
    except DateParseError:
        raise
    except ValueError:
        pass

    dt = dateutil_parse(date_string, default=_DEFAULT_DATETIME,
                        dayfirst=dayfirst, yearfirst=yearfirst,
                        ignoretz=False, out_bestunit=out_bestunit)
    return dt


def parse_datetime_string_with_reso(
    str date_string, str freq=None, dayfirst=None, yearfirst=None
):
    # NB: This will break with np.str_ (GH#45580) even though
    #  isinstance(npstrobj, str) evaluates to True, so caller must ensure
    #  the argument is *exactly* 'str'
    """
    Try hard to parse datetime string, leveraging dateutil plus some extra
    goodies like quarter recognition.

    Parameters
    ----------
    date_string : str
    freq : str or None, default None
        Helps with interpreting time string if supplied
        Corresponds to `offset.rule_code`
    dayfirst : bool, default None
        If None uses default from print_config
    yearfirst : bool, default None
        If None uses default from print_config

    Returns
    -------
    datetime
    str
        Describing resolution of parsed string.

    Raises
    ------
    ValueError : preliminary check suggests string is not datetime
    DateParseError : error within dateutil
    """

    if dayfirst is None:
        dayfirst = get_option("display.date_dayfirst")
    if yearfirst is None:
        yearfirst = get_option("display.date_yearfirst")

    cdef:
        datetime parsed
        str reso
        bint string_to_dts_failed
        npy_datetimestruct dts
        NPY_DATETIMEUNIT out_bestunit
        int out_local = 0
        int out_tzoffset
        tzinfo tz
        bint is_quarter = 0

    if not _does_string_look_like_datetime(date_string):
        raise ValueError(f'Given date string "{date_string}" not likely a datetime')

    # Try iso8601 first, as it handles nanoseconds
    string_to_dts_failed = string_to_dts(
        date_string, &dts, &out_bestunit, &out_local,
        &out_tzoffset, False
    )
    if not string_to_dts_failed:
        # Match Timestamp and drop picoseconds, femtoseconds, attoseconds
        # The new resolution will just be nano
        # GH#50417
        if out_bestunit in _timestamp_units:
            out_bestunit = NPY_DATETIMEUNIT.NPY_FR_ns

        if out_bestunit == NPY_DATETIMEUNIT.NPY_FR_ns:
            # TODO: avoid circular import
            from pandas import Timestamp
            parsed = Timestamp(date_string)
        else:
            if out_local:
                tz = timezone(timedelta(minutes=out_tzoffset))
            else:
                tz = None
            parsed = datetime_new(
                dts.year, dts.month, dts.day, dts.hour, dts.min, dts.sec, dts.us, tz
            )

        reso = npy_unit_to_attrname[out_bestunit]
        return parsed, reso

    parsed = _parse_delimited_date(date_string, dayfirst, &out_bestunit)
    if parsed is not None:
        reso = npy_unit_to_attrname[out_bestunit]
        return parsed, reso

    try:
        parsed = _parse_dateabbr_string(
            date_string, _DEFAULT_DATETIME, freq, &out_bestunit, &is_quarter
        )
    except DateParseError:
        raise
    except ValueError:
        pass
    else:
        if is_quarter:
            reso = "quarter"
        else:
            reso = npy_unit_to_attrname[out_bestunit]
        return parsed, reso

    parsed = dateutil_parse(date_string, _DEFAULT_DATETIME,
                            dayfirst=dayfirst, yearfirst=yearfirst,
                            ignoretz=False, out_bestunit=&out_bestunit)
    reso = npy_unit_to_attrname[out_bestunit]
    return parsed, reso


cpdef bint _does_string_look_like_datetime(str py_string):
    """
    Checks whether given string is a datetime: it has to start with '0' or
    be greater than 1000.

    Parameters
    ----------
    py_string: str

    Returns
    -------
    bool
        Whether given string is potentially a datetime.
    """
    cdef:
        const char *buf
        char *endptr = NULL
        Py_ssize_t length = -1
        double converted_date
        char first
        int error = 0

    buf = get_c_string_buf_and_size(py_string, &length)
    if length >= 1:
        first = buf[0]
        if first == b"0":
            # Strings starting with 0 are more consistent with a
            # date-like string than a number
            return True
        elif py_string in _not_datelike_strings:
            return False
        else:
            # xstrtod with such parameters copies behavior of python `float`
            # cast; for example, " 35.e-1 " is valid string for this cast so,
            # for correctly xstrtod call necessary to pass these params:
            # b'.' - a dot is used as separator, b'e' - an exponential form of
            # a float number can be used, b'\0' - not to use a thousand
            # separator, 1 - skip extra spaces before and after,
            converted_date = xstrtod(buf, &endptr,
                                     b".", b"e", b"\0", 1, &error, NULL)
            # if there were no errors and the whole line was parsed, then ...
            if error == 0 and endptr == buf + length:
                return converted_date >= 1000

    return True


cdef datetime _parse_dateabbr_string(str date_string, datetime default,
                                     str freq, NPY_DATETIMEUNIT* out_bestunit,
                                     bint* is_quarter):
    # special handling for possibilities eg, 2Q2005, 2Q05, 2005Q1, 05Q1
    cdef:
        datetime ret
        # year initialized to prevent compiler warnings
        int year = -1, quarter = -1, month
        Py_ssize_t date_len
        const char* buf

    if date_string in nat_strings:
        # default to nanos, could also reasonably do NPY_FR_GENERIC
        out_bestunit[0] = NPY_DATETIMEUNIT.NPY_FR_ns
        return NaT

    date_string = date_string.upper()
    date_len = len(date_string)

    if date_len == 4:
        # parse year only like 2000
        try:
            ret = default.replace(year=int(date_string))
            out_bestunit[0] = NPY_DATETIMEUNIT.NPY_FR_Y
            return ret
        except ValueError:
            pass

    if 4 <= date_len <= 7:
        buf = get_c_string_buf_and_size(date_string, &date_len)
        try:
            i = date_string.index("Q", 1, 6)
            if i == 1:
                quarter = _parse_1digit(buf)  # i.e. int(date_string[0])
                if date_len == 4 or (date_len == 5
                                     and date_string[i + 1] == "-"):
                    # r'(\d)Q-?(\d\d)')
                    year = 2000 + int(date_string[-2:])
                elif date_len == 6 or (date_len == 7
                                       and date_string[i + 1] == "-"):
                    # r'(\d)Q-?(\d\d\d\d)')
                    year = int(date_string[-4:])
                else:
                    raise ValueError
            elif i == 2 or i == 3:
                # r'(\d\d)-?Q(\d)'
                if date_len == 4 or (date_len == 5
                                     and date_string[i - 1] == "-"):
                    # i.e. quarter = int(date_string[-1])
                    quarter = _parse_1digit(buf + date_len - 1)
                    year = 2000 + int(date_string[:2])
                else:
                    raise ValueError
            elif i == 4 or i == 5:
                if date_len == 6 or (date_len == 7
                                     and date_string[i - 1] == "-"):
                    # r'(\d\d\d\d)-?Q(\d)'
                    # i.e. quarter = int(date_string[-1])
                    quarter = _parse_1digit(buf + date_len - 1)
                    year = int(date_string[:4])
                else:
                    raise ValueError

            if not (1 <= quarter <= 4):
                raise DateParseError(f"Incorrect quarterly string is given, "
                                     f"quarter must be "
                                     f"between 1 and 4: {date_string}")

            try:
                # GH#1228
                year, month = quarter_to_myear(year, quarter, freq)
            except KeyError:
                raise DateParseError("Unable to retrieve month "
                                     "information from given "
                                     f"freq: {freq}")

            ret = default.replace(year=year, month=month)
            # Monthly is as close as we can get to a non-existent NPY_FR_Q
            out_bestunit[0] = NPY_DATETIMEUNIT.NPY_FR_M
            is_quarter[0] = 1
            return ret

        except DateParseError:
            raise
        except ValueError:
            # e.g. if "Q" is not in date_string and .index raised
            pass

    if date_len == 6 and freq == "M":
        year = int(date_string[:4])
        month = int(date_string[4:6])
        try:
            ret = default.replace(year=year, month=month)
            out_bestunit[0] = NPY_DATETIMEUNIT.NPY_FR_M
            return ret
        except ValueError as err:
            # We can infer that none of the patterns below will match
            raise ValueError(f"Unable to parse {date_string}") from err

    for pat in ["%Y-%m", "%b %Y", "%b-%Y"]:
        try:
            ret = datetime.strptime(date_string, pat)
            out_bestunit[0] = NPY_DATETIMEUNIT.NPY_FR_M
            return ret
        except ValueError:
            pass

    raise ValueError(f"Unable to parse {date_string}")


cpdef quarter_to_myear(int year, int quarter, str freq):
    """
    A quarterly frequency defines a "year" which may not coincide with
    the calendar-year.  Find the calendar-year and calendar-month associated
    with the given year and quarter under the `freq`-derived calendar.

    Parameters
    ----------
    year : int
    quarter : int
    freq : str or None

    Returns
    -------
    year : int
    month : int

    See Also
    --------
    Period.qyear
    """
    if quarter <= 0 or quarter > 4:
        raise ValueError("Quarter must be 1 <= q <= 4")

    if freq is not None:
        mnum = c_MONTH_NUMBERS[get_rule_month(freq)] + 1
        month = (mnum + (quarter - 1) * 3) % 12 + 1
        if month > mnum:
            year -= 1
    else:
        month = (quarter - 1) * 3 + 1

    return year, month


cdef datetime dateutil_parse(
    str timestr,
    datetime default,
    bint ignoretz,
    bint dayfirst,
    bint yearfirst,
    NPY_DATETIMEUNIT* out_bestunit
):
    """ lifted from dateutil to get resolution"""

    cdef:
        str attr
        datetime ret
        object res
        str reso = None
        dict repl = {}

    try:
        res, _ = DEFAULTPARSER._parse(timestr, dayfirst=dayfirst, yearfirst=yearfirst)
    except InvalidOperation:
        # GH#51157 dateutil can raise decimal.InvalidOperation
        res = None

    if res is None:
        raise DateParseError(
            f"Unknown datetime string format, unable to parse: {timestr}"
        )

    for attr in ["year", "month", "day", "hour",
                 "minute", "second", "microsecond"]:
        value = getattr(res, attr)
        if value is not None:
            repl[attr] = value
            reso = attr

    if reso is None:
        raise DateParseError(f"Unable to parse datetime string: {timestr}")

    if reso == "microsecond":
        if repl["microsecond"] == 0:
            reso = "second"
        elif repl["microsecond"] % 1000 == 0:
            reso = "millisecond"

    try:
        ret = default.replace(**repl)
    except ValueError as err:
        # e.g. "day is out of range for month"
        # we re-raise to match dateutil's exception message
        raise DateParseError(str(err) + ": " + timestr) from err
    except OverflowError as err:
        # with e.g. "08335394550" dateutil raises when trying to pass
        #  year=8335394550 to datetime.replace
        raise OutOfBoundsDatetime(
            f'Parsing "{timestr}" to datetime overflows'
        ) from err

    if res.weekday is not None and not res.day:
        # GH#52659
        raise ValueError(
            "Parsing datetimes with weekday but no day information is "
            "not supported"
        )
    if not ignoretz:
        if res.tzname and res.tzname in time.tzname:
            # GH#50791
            if res.tzname != "UTC":
                # If the system is localized in UTC (as many CI runs are)
                #  we get tzlocal, once the deprecation is enforced will get
                #  timezone.utc, not raise.
                warnings.warn(
                    f"Parsing '{res.tzname}' as tzlocal (dependent on system timezone) "
                    "is deprecated and will raise in a future version. Pass the 'tz' "
                    "keyword or call tz_localize after construction instead",
                    FutureWarning,
                    stacklevel=find_stack_level()
                )
            ret = ret.replace(tzinfo=_dateutil_tzlocal())
        elif res.tzoffset == 0:
            ret = ret.replace(tzinfo=_dateutil_tzutc())
        elif res.tzoffset:
            ret = ret.replace(tzinfo=tzoffset(res.tzname, res.tzoffset))

            # dateutil can return a datetime with a tzoffset outside of (-24H, 24H)
            #  bounds, which is invalid (can be constructed, but raises if we call
            #  str(ret)).  Check that and raise here if necessary.
            try:
                ret.utcoffset()
            except ValueError as err:
                # offset must be a timedelta strictly between -timedelta(hours=24)
                #  and timedelta(hours=24)
                raise ValueError(
                    f'Parsed string "{timestr}" gives an invalid tzoffset, '
                    "which must be between -timedelta(hours=24) and timedelta(hours=24)"
                )
        elif res.tzname is not None:
            # e.g. "1994 Jan 15 05:16 FOO" where FOO is not recognized
            # GH#18702
            warnings.warn(
                f'Parsed string "{timestr}" included an un-recognized timezone '
                f'"{res.tzname}". Dropping unrecognized timezones is deprecated; '
                "in a future version this will raise. Instead pass the string "
                "without the timezone, then use .tz_localize to convert to a "
                "recognized timezone.",
                FutureWarning,
                stacklevel=find_stack_level()
            )

    out_bestunit[0] = attrname_to_npy_unit[reso]
    return ret


# ----------------------------------------------------------------------
# Parsing for type-inference


def try_parse_dates(object[:] values, parser) -> np.ndarray:
    cdef:
        Py_ssize_t i, n
        object[::1] result

    n = len(values)
    result = np.empty(n, dtype="O")

    for i in range(n):
        if values[i] == "":
            result[i] = np.nan
        else:
            result[i] = parser(values[i])

    return result.base  # .base to access underlying ndarray


def try_parse_year_month_day(
    object[:] years, object[:] months, object[:] days
) -> np.ndarray:
    cdef:
        Py_ssize_t i, n
        object[::1] result

    n = len(years)
    # TODO(cython3): Use len instead of `shape[0]`
    if months.shape[0] != n or days.shape[0] != n:
        raise ValueError("Length of years/months/days must all be equal")
    result = np.empty(n, dtype="O")

    for i in range(n):
        result[i] = datetime(int(years[i]), int(months[i]), int(days[i]))

    return result.base  # .base to access underlying ndarray


# ----------------------------------------------------------------------
# Miscellaneous


# Class copied verbatim from https://github.com/dateutil/dateutil/pull/732
#
# We use this class to parse and tokenize date strings. However, as it is
# a private class in the dateutil library, relying on backwards compatibility
# is not practical. In fact, using this class issues warnings (xref gh-21322).
# Thus, we port the class over so that both issues are resolved.
#
# Copyright (c) 2017 - dateutil contributors
class _timelex:
    def __init__(self, instream):
        if getattr(instream, "decode", None) is not None:
            instream = instream.decode()

        if isinstance(instream, str):
            self.stream = instream
        elif getattr(instream, "read", None) is None:
            raise TypeError(
                "Parser must be a string or character stream, not "
                f"{type(instream).__name__}")
        else:
            self.stream = instream.read()

    def get_tokens(self):
        """
        This function breaks the time string into lexical units (tokens), which
        can be parsed by the parser. Lexical units are demarcated by changes in
        the character set, so any continuous string of letters is considered
        one unit, any continuous string of numbers is considered one unit.
        The main complication arises from the fact that dots ('.') can be used
        both as separators (e.g. "Sep.20.2009") or decimal points (e.g.
        "4:30:21.447"). As such, it is necessary to read the full context of
        any dot-separated strings before breaking it into tokens; as such, this
        function maintains a "token stack", for when the ambiguous context
        demands that multiple tokens be parsed at once.
        """
        cdef:
            Py_ssize_t n

        stream = self.stream.replace("\x00", "")

        # TODO: Change \s --> \s+ (this doesn't match existing behavior)
        # TODO: change the punctuation block to punc+ (does not match existing)
        # TODO: can we merge the two digit patterns?
        tokens = re.findall(r"\s|"
                            r"(?<![\.\d])\d+\.\d+(?![\.\d])"
                            r"|\d+"
                            r"|[a-zA-Z]+"
                            r"|[\./:]+"
                            r"|[^\da-zA-Z\./:\s]+", stream)

        # Re-combine token tuples of the form ["59", ",", "456"] because
        # in this context the "," is treated as a decimal
        # (e.g. in python's default logging format)
        for n, token in enumerate(tokens[:-2]):
            # Kludge to match ,-decimal behavior; it'd be better to do this
            # later in the process and have a simpler tokenization
            if (token is not None and token.isdigit() and
                    tokens[n + 1] == "," and tokens[n + 2].isdigit()):
                # Have to check None b/c it might be replaced during the loop
                # TODO: I _really_ don't faking the value here
                tokens[n] = token + "." + tokens[n + 2]
                tokens[n + 1] = None
                tokens[n + 2] = None

        tokens = [x for x in tokens if x is not None]
        return tokens

    @classmethod
    def split(cls, s):
        return cls(s).get_tokens()


_DATEUTIL_LEXER_SPLIT = _timelex.split


def guess_datetime_format(dt_str: str, bint dayfirst=False) -> str | None:
    """
    Guess the datetime format of a given datetime string.

    Parameters
    ----------
    dt_str : str
        Datetime string to guess the format of.
    dayfirst : bool, default False
        If True parses dates with the day first, eg 20/01/2005
        Warning: dayfirst=True is not strict, but will prefer to parse
        with day first (this is a known bug).

    Returns
    -------
    str or None : ret
        datetime format string (for `strftime` or `strptime`),
        or None if it can't be guessed.
    """
    cdef:
        NPY_DATETIMEUNIT out_bestunit
    day_attribute_and_format = (("day",), "%d", 2)

    # attr name, format, padding (if any)
    datetime_attrs_to_format = [
        (("year", "month", "day", "hour", "minute", "second"), "%Y%m%d%H%M%S", 0),
        (("year", "month", "day", "hour", "minute"), "%Y%m%d%H%M", 0),
        (("year", "month", "day", "hour"), "%Y%m%d%H", 0),
        (("year", "month", "day"), "%Y%m%d", 0),
        (("hour", "minute", "second"), "%H%M%S", 0),
        (("hour", "minute"), "%H%M", 0),
        (("year",), "%Y", 0),
        (("month",), "%B", 0),
        (("month",), "%b", 0),
        (("month",), "%m", 2),
        day_attribute_and_format,
        (("hour",), "%H", 2),
        (("minute",), "%M", 2),
        (("second",), "%S", 2),
        (("second", "microsecond"), "%S.%f", 0),
        (("tzinfo",), "%z", 0),
        (("tzinfo",), "%Z", 0),
        (("day_of_week",), "%a", 0),
        (("day_of_week",), "%A", 0),
        (("meridiem",), "%p", 0),
    ]

    if dayfirst:
        datetime_attrs_to_format.remove(day_attribute_and_format)
        datetime_attrs_to_format.insert(0, day_attribute_and_format)

    # same default used by dateutil
    default = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    try:
        parsed_datetime = dateutil_parse(
            dt_str,
            default=default,
            dayfirst=dayfirst,
            yearfirst=False,
            ignoretz=False,
            out_bestunit=&out_bestunit,
        )
    except (ValueError, OverflowError, InvalidOperation):
        # In case the datetime can't be parsed, its format cannot be guessed
        return None

    if parsed_datetime is None:
        return None

    # _DATEUTIL_LEXER_SPLIT from dateutil will never raise here
    tokens = _DATEUTIL_LEXER_SPLIT(dt_str)

    # Normalize offset part of tokens.
    # There are multiple formats for the timezone offset.
    # To pass the comparison condition between the output of `strftime` and
    # joined tokens, which is carried out at the final step of the function,
    # the offset part of the tokens must match the '%z' format like '+0900'
    # instead of ‘+09:00’.
    if parsed_datetime.tzinfo is not None:
        offset_index = None
        if len(tokens) > 0 and tokens[-1] == "Z":
            # the last 'Z' means zero offset
            offset_index = -1
        elif len(tokens) > 1 and tokens[-2] in ("+", "-"):
            # ex. [..., '+', '0900']
            offset_index = -2
        elif len(tokens) > 3 and tokens[-4] in ("+", "-"):
            # ex. [..., '+', '09', ':', '00']
            offset_index = -4

        if offset_index is not None:
            # If the input string has a timezone offset like '+0900',
            # the offset is separated into two tokens, ex. ['+', '0900’].
            # This separation will prevent subsequent processing
            # from correctly parsing the time zone format.
            # So in addition to the format nomalization, we rejoin them here.
            try:
                tokens[offset_index] = parsed_datetime.strftime("%z")
            except ValueError:
                # Invalid offset might not have raised in du_parse
                # https://github.com/dateutil/dateutil/issues/188
                return None
            tokens = tokens[:offset_index + 1 or None]

    format_guess = [None] * len(tokens)
    found_attrs = set()

    for attrs, attr_format, padding in datetime_attrs_to_format:
        # If a given attribute has been placed in the format string, skip
        # over other formats for that same underlying attribute (IE, month
        # can be represented in multiple different ways)
        if set(attrs) & found_attrs:
            continue

        if parsed_datetime.tzinfo is None and attr_format in ("%Z", "%z"):
            continue

        parsed_formatted = parsed_datetime.strftime(attr_format)
        for i, token_format in enumerate(format_guess):
            token_filled = _fill_token(tokens[i], padding)
            if token_format is None and token_filled == parsed_formatted:
                format_guess[i] = attr_format
                tokens[i] = token_filled
                found_attrs.update(attrs)
                break

    # Only consider it a valid guess if we have a year, month and day.
    # We make exceptions for %Y and %Y-%m (only with the `-` separator)
    # as they conform with ISO8601.
    if (
        len({"year", "month", "day"} & found_attrs) != 3
        and format_guess != ["%Y"]
        and not (
            format_guess == ["%Y", None, "%m"] and tokens[1] == "-"
        )
    ):
        return None

    output_format = []
    for i, guess in enumerate(format_guess):
        if guess is not None:
            # Either fill in the format placeholder (like %Y)
            output_format.append(guess)
        else:
            # Or just the token separate (IE, the dashes in "01-01-2013")
            try:
                # If the token is numeric, then we likely didn't parse it
                # properly, so our guess is wrong
                float(tokens[i])
                return None
            except ValueError:
                pass

            output_format.append(tokens[i])

    # if am/pm token present, replace 24-hour %H, with 12-hour %I
    if "%p" in output_format and "%H" in output_format:
        i = output_format.index("%H")
        output_format[i] = "%I"

    guessed_format = "".join(output_format)

    try:
        array_strptime(np.asarray([dt_str], dtype=object), guessed_format)
    except ValueError:
        # Doesn't parse, so this can't be the correct format.
        return None
    # rebuild string, capturing any inferred padding
    dt_str = "".join(tokens)
    if parsed_datetime.strftime(guessed_format) == dt_str:
        _maybe_warn_about_dayfirst(guessed_format, dayfirst)
        return guessed_format
    else:
        return None


cdef str _fill_token(token: str, padding: int):
    cdef str token_filled
    if re.search(r"\d+\.\d+", token) is None:
        # For example: 98
        token_filled = token.zfill(padding)
    else:
        # For example: 00.123
        seconds, nanoseconds = token.split(".")
        seconds = f"{int(seconds):02d}"
        # right-pad so we get nanoseconds, then only take
        # first 6 digits (microseconds) as stdlib datetime
        # doesn't support nanoseconds
        nanoseconds = nanoseconds.ljust(9, "0")[:6]
        token_filled = f"{seconds}.{nanoseconds}"
    return token_filled


cdef void _maybe_warn_about_dayfirst(format: str, bint dayfirst) noexcept:
    """Warn if guessed datetime format doesn't respect dayfirst argument."""
    cdef:
        int day_index = format.find("%d")
        int month_index = format.find("%m")

    if (day_index != -1) and (month_index != -1):
        if (day_index > month_index) and dayfirst:
            warnings.warn(
                f"Parsing dates in {format} format when dayfirst=True was specified. "
                "Pass `dayfirst=False` or specify a format to silence this warning.",
                UserWarning,
                stacklevel=find_stack_level(),
            )
        if (day_index < month_index) and not dayfirst:
            warnings.warn(
                f"Parsing dates in {format} format when dayfirst=False (the default) "
                "was specified. "
                "Pass `dayfirst=True` or specify a format to silence this warning.",
                UserWarning,
                stacklevel=find_stack_level(),
            )


@cython.wraparound(False)
@cython.boundscheck(False)
cdef object convert_to_unicode(object item, bint keep_trivial_numbers):
    """
    Convert `item` to str.

    Parameters
    ----------
    item : object
    keep_trivial_numbers : bool
        if True, then conversion (to string from integer/float zero)
        is not performed

    Returns
    -------
    str or int or float
    """
    cdef:
        float64_t float_item

    if keep_trivial_numbers:
        if isinstance(item, int):
            if <int>item == 0:
                return item
        elif isinstance(item, float):
            float_item = item
            if float_item == 0.0 or float_item != float_item:
                return item

    if not isinstance(item, str):
        item = PyObject_Str(item)

    return item


@cython.wraparound(False)
@cython.boundscheck(False)
def concat_date_cols(tuple date_cols) -> np.ndarray:
    """
    Concatenates elements from numpy arrays in `date_cols` into strings.

    Parameters
    ----------
    date_cols : tuple[ndarray]

    Returns
    -------
    arr_of_rows : ndarray[object]

    Examples
    --------
    >>> dates=np.array(['3/31/2019', '4/31/2019'], dtype=object)
    >>> times=np.array(['11:20', '10:45'], dtype=object)
    >>> result = concat_date_cols((dates, times))
    >>> result
    array(['3/31/2019 11:20', '4/31/2019 10:45'], dtype=object)
    """
    cdef:
        Py_ssize_t rows_count = 0, col_count = len(date_cols)
        Py_ssize_t col_idx, row_idx
        list list_to_join
        cnp.ndarray[object] iters
        object[::1] iters_view
        flatiter it
        cnp.ndarray[object] result
        object[::1] result_view

    if col_count == 0:
        return np.zeros(0, dtype=object)

    if not all(is_array(array) for array in date_cols):
        raise ValueError("not all elements from date_cols are numpy arrays")

    rows_count = min(len(array) for array in date_cols)
    result = np.zeros(rows_count, dtype=object)
    result_view = result

    if col_count == 1:
        array = date_cols[0]
        it = <flatiter>PyArray_IterNew(array)
        for row_idx in range(rows_count):
            item = PyArray_GETITEM(array, PyArray_ITER_DATA(it))
            result_view[row_idx] = convert_to_unicode(item, True)
            PyArray_ITER_NEXT(it)
    else:
        # create fixed size list - more efficient memory allocation
        list_to_join = [None] * col_count
        iters = np.zeros(col_count, dtype=object)

        # create memoryview of iters ndarray, that will contain some
        # flatiter's for each array in `date_cols` - more efficient indexing
        iters_view = iters
        for col_idx, array in enumerate(date_cols):
            iters_view[col_idx] = PyArray_IterNew(array)

        # array elements that are on the same line are converted to one string
        for row_idx in range(rows_count):
            for col_idx, array in enumerate(date_cols):
                # this cast is needed, because we did not find a way
                # to efficiently store `flatiter` type objects in ndarray
                it = <flatiter>iters_view[col_idx]
                item = PyArray_GETITEM(array, PyArray_ITER_DATA(it))
                list_to_join[col_idx] = convert_to_unicode(item, False)
                PyArray_ITER_NEXT(it)
            result_view[row_idx] = " ".join(list_to_join)

    return result


cpdef str get_rule_month(str source):
    """
    Return starting month of given freq, default is December.

    Parameters
    ----------
    source : str
        Derived from `freq.rule_code` or `freq.freqstr`.

    Returns
    -------
    rule_month: str

    Examples
    --------
    >>> get_rule_month('D')
    'DEC'

    >>> get_rule_month('A-JAN')
    'JAN'
    """
    source = source.upper()
    if "-" not in source:
        return "DEC"
    else:
        return source.split("-")[1]
