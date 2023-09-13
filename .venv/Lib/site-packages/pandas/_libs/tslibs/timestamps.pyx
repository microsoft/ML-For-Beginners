"""
_Timestamp is a c-defined subclass of datetime.datetime

_Timestamp is PITA. Because we inherit from datetime, which has very specific
construction requirements, we need to do object instantiation in python
(see Timestamp class below). This will serve as a C extension type that
shadows the python class, where we do any heavy lifting.
"""

import warnings

cimport cython

import numpy as np

cimport numpy as cnp
from numpy cimport (
    int64_t,
    ndarray,
    uint8_t,
)

cnp.import_array()

from cpython.datetime cimport (  # alias tzinfo_type bc `tzinfo` is a kwarg below
    PyDate_Check,
    PyDateTime_Check,
    PyDelta_Check,
    PyTZInfo_Check,
    datetime,
    import_datetime,
    time as dt_time,
    tzinfo as tzinfo_type,
)
from cpython.object cimport (
    Py_EQ,
    Py_GE,
    Py_GT,
    Py_LE,
    Py_LT,
    Py_NE,
    PyObject_RichCompare,
    PyObject_RichCompareBool,
)

import_datetime()

import datetime as dt

from pandas._libs.tslibs cimport ccalendar
from pandas._libs.tslibs.base cimport ABCTimestamp

from pandas.util._exceptions import find_stack_level

from pandas._libs.tslibs.conversion cimport (
    _TSObject,
    convert_datetime_to_tsobject,
    convert_to_tsobject,
    maybe_localize_tso,
)
from pandas._libs.tslibs.dtypes cimport (
    npy_unit_to_abbrev,
    periods_per_day,
    periods_per_second,
)
from pandas._libs.tslibs.util cimport (
    is_array,
    is_datetime64_object,
    is_integer_object,
)

from pandas._libs.tslibs.fields import (
    RoundTo,
    get_date_name_field,
    get_start_end_field,
    round_nsint64,
)

from pandas._libs.tslibs.nattype cimport (
    NPY_NAT,
    c_NaT as NaT,
)
from pandas._libs.tslibs.np_datetime cimport (
    NPY_DATETIMEUNIT,
    NPY_FR_ns,
    check_dts_bounds,
    cmp_dtstructs,
    cmp_scalar,
    convert_reso,
    get_datetime64_unit,
    get_datetime64_value,
    get_unit_from_dtype,
    import_pandas_datetime,
    npy_datetimestruct,
    npy_datetimestruct_to_datetime,
    pandas_datetime_to_datetimestruct,
    pydatetime_to_dtstruct,
)

import_pandas_datetime()

from pandas._libs.tslibs.np_datetime import (
    OutOfBoundsDatetime,
    OutOfBoundsTimedelta,
)

from pandas._libs.tslibs.offsets cimport to_offset
from pandas._libs.tslibs.timedeltas cimport (
    _Timedelta,
    delta_to_nanoseconds,
    is_any_td_scalar,
)

from pandas._libs.tslibs.timedeltas import Timedelta

from pandas._libs.tslibs.timezones cimport (
    get_timezone,
    is_utc,
    maybe_get_tz,
    treat_tz_as_pytz,
    utc_stdlib as UTC,
)
from pandas._libs.tslibs.tzconversion cimport (
    tz_convert_from_utc_single,
    tz_localize_to_utc_single,
)

# ----------------------------------------------------------------------
# Constants
_zero_time = dt_time(0, 0)
_no_input = object()

# ----------------------------------------------------------------------


cdef _Timestamp create_timestamp_from_ts(
    int64_t value,
    npy_datetimestruct dts,
    tzinfo tz,
    bint fold,
    NPY_DATETIMEUNIT reso=NPY_FR_ns,
):
    """ convenience routine to construct a Timestamp from its parts """
    cdef:
        _Timestamp ts_base
        int64_t pass_year = dts.year

    # We pass year=1970/1972 here and set year below because with non-nanosecond
    #  resolution we may have datetimes outside of the stdlib pydatetime
    #  implementation bounds, which would raise.
    # NB: this means the C-API macro PyDateTime_GET_YEAR is unreliable.
    if 1 <= pass_year <= 9999:
        # we are in-bounds for pydatetime
        pass
    elif ccalendar.is_leapyear(dts.year):
        pass_year = 1972
    else:
        pass_year = 1970

    ts_base = _Timestamp.__new__(Timestamp, pass_year, dts.month,
                                 dts.day, dts.hour, dts.min,
                                 dts.sec, dts.us, tz, fold=fold)

    ts_base._value = value
    ts_base.year = dts.year
    ts_base.nanosecond = dts.ps // 1000
    ts_base._creso = reso

    return ts_base


def _unpickle_timestamp(value, freq, tz, reso=NPY_FR_ns):
    # GH#41949 dont warn on unpickle if we have a freq
    ts = Timestamp._from_value_and_reso(value, reso, tz)
    return ts


# ----------------------------------------------------------------------

def integer_op_not_supported(obj):
    # GH#22535 add/sub of integers and int-arrays is no longer allowed
    # Note we return rather than raise the exception so we can raise in
    #  the caller; mypy finds this more palatable.
    cls = type(obj).__name__

    # GH#30886 using an fstring raises SystemError
    int_addsub_msg = (
        f"Addition/subtraction of integers and integer-arrays with {cls} is "
        "no longer supported.  Instead of adding/subtracting `n`, "
        "use `n * obj.freq`"
    )
    return TypeError(int_addsub_msg)


class MinMaxReso:
    """
    We need to define min/max/resolution on both the Timestamp _instance_
    and Timestamp class.  On an instance, these depend on the object's _reso.
    On the class, we default to the values we would get with nanosecond _reso.

    See also: timedeltas.MinMaxReso
    """
    def __init__(self, name):
        self._name = name

    def __get__(self, obj, type=None):
        cls = Timestamp
        if self._name == "min":
            val = np.iinfo(np.int64).min + 1
        elif self._name == "max":
            val = np.iinfo(np.int64).max
        else:
            assert self._name == "resolution"
            val = 1
            cls = Timedelta

        if obj is None:
            # i.e. this is on the class, default to nanos
            return cls(val)
        elif self._name == "resolution":
            return Timedelta._from_value_and_reso(val, obj._creso)
        else:
            return Timestamp._from_value_and_reso(val, obj._creso, tz=None)

    def __set__(self, obj, value):
        raise AttributeError(f"{self._name} is not settable.")


# ----------------------------------------------------------------------

cdef class _Timestamp(ABCTimestamp):

    # higher than np.ndarray and np.matrix
    __array_priority__ = 100
    dayofweek = _Timestamp.day_of_week
    dayofyear = _Timestamp.day_of_year

    min = MinMaxReso("min")
    max = MinMaxReso("max")
    resolution = MinMaxReso("resolution")  # GH#21336, GH#21365

    @property
    def value(self) -> int:
        try:
            return convert_reso(self._value, self._creso, NPY_FR_ns, False)
        except OverflowError:
            raise OverflowError(
                "Cannot convert Timestamp to nanoseconds without overflow. "
                "Use `.asm8.view('i8')` to cast represent Timestamp in its own "
                f"unit (here, {self.unit})."
            )

    @property
    def unit(self) -> str:
        """
        The abbreviation associated with self._creso.

        Examples
        --------
        >>> pd.Timestamp("2020-01-01 12:34:56").unit
        's'

        >>> pd.Timestamp("2020-01-01 12:34:56.123").unit
        'ms'

        >>> pd.Timestamp("2020-01-01 12:34:56.123456").unit
        'us'

        >>> pd.Timestamp("2020-01-01 12:34:56.123456789").unit
        'ns'
        """
        return npy_unit_to_abbrev(self._creso)

    # -----------------------------------------------------------------
    # Constructors

    @classmethod
    def _from_value_and_reso(cls, int64_t value, NPY_DATETIMEUNIT reso, tzinfo tz):
        cdef:
            _TSObject obj = _TSObject()

        if value == NPY_NAT:
            return NaT

        if reso < NPY_DATETIMEUNIT.NPY_FR_s or reso > NPY_DATETIMEUNIT.NPY_FR_ns:
            raise NotImplementedError(
                "Only resolutions 's', 'ms', 'us', 'ns' are supported."
            )

        obj.value = value
        obj.creso = reso
        pandas_datetime_to_datetimestruct(value, reso, &obj.dts)
        maybe_localize_tso(obj, tz, reso)

        return create_timestamp_from_ts(
            value, obj.dts, tz=obj.tzinfo, fold=obj.fold, reso=reso
        )

    @classmethod
    def _from_dt64(cls, dt64: np.datetime64):
        # construct a Timestamp from a np.datetime64 object, keeping the
        #  resolution of the input.
        # This is herely mainly so we can incrementally implement non-nano
        #  (e.g. only tznaive at first)
        cdef:
            int64_t value
            NPY_DATETIMEUNIT reso

        reso = get_datetime64_unit(dt64)
        value = get_datetime64_value(dt64)
        return cls._from_value_and_reso(value, reso, None)

    # -----------------------------------------------------------------

    def __hash__(_Timestamp self):
        if self.nanosecond:
            return hash(self._value)
        if not (1 <= self.year <= 9999):
            # out of bounds for pydatetime
            return hash(self._value)
        if self.fold:
            return datetime.__hash__(self.replace(fold=0))
        return datetime.__hash__(self)

    def __richcmp__(_Timestamp self, object other, int op):
        cdef:
            _Timestamp ots

        if isinstance(other, _Timestamp):
            ots = other
        elif other is NaT:
            return op == Py_NE
        elif is_datetime64_object(other):
            ots = Timestamp(other)
        elif PyDateTime_Check(other):
            if self.nanosecond == 0:
                val = self.to_pydatetime()
                return PyObject_RichCompareBool(val, other, op)

            try:
                ots = type(self)(other)
            except ValueError:
                return self._compare_outside_nanorange(other, op)

        elif is_array(other):
            # avoid recursion error GH#15183
            if other.dtype.kind == "M":
                if self.tz is None:
                    return PyObject_RichCompare(self.asm8, other, op)
                elif op == Py_NE:
                    return np.ones(other.shape, dtype=np.bool_)
                elif op == Py_EQ:
                    return np.zeros(other.shape, dtype=np.bool_)
                raise TypeError(
                    "Cannot compare tz-naive and tz-aware timestamps"
                )
            elif other.dtype.kind == "O":
                # Operate element-wise
                return np.array(
                    [PyObject_RichCompare(self, x, op) for x in other],
                    dtype=bool,
                )
            elif op == Py_NE:
                return np.ones(other.shape, dtype=np.bool_)
            elif op == Py_EQ:
                return np.zeros(other.shape, dtype=np.bool_)
            return NotImplemented

        elif PyDate_Check(other):
            # returning NotImplemented defers to the `date` implementation
            #  which incorrectly drops tz and normalizes to midnight
            #  before comparing
            # We follow the stdlib datetime behavior of never being equal
            if op == Py_EQ:
                return False
            elif op == Py_NE:
                return True
            raise TypeError(
                "Cannot compare Timestamp with datetime.date. "
                "Use ts == pd.Timestamp(date) or ts.date() == date instead."
            )
        else:
            return NotImplemented

        if not self._can_compare(ots):
            if op == Py_NE or op == Py_EQ:
                return NotImplemented
            raise TypeError(
                "Cannot compare tz-naive and tz-aware timestamps"
            )
        if self._creso == ots._creso:
            return cmp_scalar(self._value, ots._value, op)
        return self._compare_mismatched_resos(ots, op)

    # TODO: copied from Timedelta; try to de-duplicate
    cdef bint _compare_mismatched_resos(self, _Timestamp other, int op):
        # Can't just dispatch to numpy as they silently overflow and get it wrong
        cdef:
            npy_datetimestruct dts_self
            npy_datetimestruct dts_other

        # dispatch to the datetimestruct utils instead of writing new ones!
        pandas_datetime_to_datetimestruct(self._value, self._creso, &dts_self)
        pandas_datetime_to_datetimestruct(other._value, other._creso, &dts_other)
        return cmp_dtstructs(&dts_self,  &dts_other, op)

    cdef bint _compare_outside_nanorange(_Timestamp self, datetime other,
                                         int op) except -1:
        cdef:
            datetime dtval = self.to_pydatetime(warn=False)

        if not self._can_compare(other):
            return NotImplemented

        if self.nanosecond == 0:
            return PyObject_RichCompareBool(dtval, other, op)

        # otherwise we have dtval < self
        if op == Py_NE:
            return True
        if op == Py_EQ:
            return False
        if op == Py_LE or op == Py_LT:
            return self.year <= other.year
        if op == Py_GE or op == Py_GT:
            return self.year >= other.year

    cdef bint _can_compare(self, datetime other):
        if self.tzinfo is not None:
            return other.tzinfo is not None
        return other.tzinfo is None

    @cython.overflowcheck(True)
    def __add__(self, other):
        cdef:
            int64_t nanos = 0

        if is_any_td_scalar(other):
            other = Timedelta(other)

            # TODO: share this with __sub__, Timedelta.__add__
            # Matching numpy, we cast to the higher resolution. Unlike numpy,
            #  we raise instead of silently overflowing during this casting.
            if self._creso < other._creso:
                self = (<_Timestamp>self)._as_creso(other._creso, round_ok=True)
            elif self._creso > other._creso:
                other = (<_Timedelta>other)._as_creso(self._creso, round_ok=True)

            nanos = other._value

            try:
                new_value = self._value+ nanos
                result = type(self)._from_value_and_reso(
                    new_value, reso=self._creso, tz=self.tzinfo
                )
            except OverflowError as err:
                # TODO: don't hard-code nanosecond here
                new_value = int(self._value) + int(nanos)
                raise OutOfBoundsDatetime(
                    f"Out of bounds nanosecond timestamp: {new_value}"
                ) from err

            return result

        elif is_integer_object(other):
            raise integer_op_not_supported(self)

        elif is_array(other):
            if other.dtype.kind in "iu":
                raise integer_op_not_supported(self)
            if other.dtype.kind == "m":
                if self.tz is None:
                    return self.asm8 + other
                return np.asarray(
                    [self + other[n] for n in range(len(other))],
                    dtype=object,
                )

        elif not isinstance(self, _Timestamp):
            # cython semantics, args have been switched and this is __radd__
            # TODO(cython3): remove this it moved to __radd__
            return other.__add__(self)

        return NotImplemented

    def __radd__(self, other):
        # Have to duplicate checks to avoid infinite recursion due to NotImplemented
        if is_any_td_scalar(other) or is_integer_object(other) or is_array(other):
            return self.__add__(other)
        return NotImplemented

    def __sub__(self, other):
        if other is NaT:
            return NaT

        elif is_any_td_scalar(other) or is_integer_object(other):
            neg_other = -other
            return self + neg_other

        elif is_array(other):
            if other.dtype.kind in "iu":
                raise integer_op_not_supported(self)
            if other.dtype.kind == "m":
                if self.tz is None:
                    return self.asm8 - other
                return np.asarray(
                    [self - other[n] for n in range(len(other))],
                    dtype=object,
                )
            return NotImplemented

        # coerce if necessary if we are a Timestamp-like
        if (PyDateTime_Check(self)
                and (PyDateTime_Check(other) or is_datetime64_object(other))):
            # both_timestamps is to determine whether Timedelta(self - other)
            # should raise the OOB error, or fall back returning a timedelta.
            # TODO(cython3): clean out the bits that moved to __rsub__
            both_timestamps = (isinstance(other, _Timestamp) and
                               isinstance(self, _Timestamp))
            if isinstance(self, _Timestamp):
                other = type(self)(other)
            else:
                self = type(other)(self)

            if (self.tzinfo is None) ^ (other.tzinfo is None):
                raise TypeError(
                    "Cannot subtract tz-naive and tz-aware datetime-like objects."
                )

            # Matching numpy, we cast to the higher resolution. Unlike numpy,
            #  we raise instead of silently overflowing during this casting.
            if self._creso < other._creso:
                self = (<_Timestamp>self)._as_creso(other._creso, round_ok=True)
            elif self._creso > other._creso:
                other = (<_Timestamp>other)._as_creso(self._creso, round_ok=True)

            # scalar Timestamp/datetime - Timestamp/datetime -> yields a
            # Timedelta
            try:
                res_value = self._value- other._value
                return Timedelta._from_value_and_reso(res_value, self._creso)
            except (OverflowError, OutOfBoundsDatetime, OutOfBoundsTimedelta) as err:
                if isinstance(other, _Timestamp):
                    if both_timestamps:
                        raise OutOfBoundsDatetime(
                            "Result is too large for pandas.Timedelta. Convert inputs "
                            "to datetime.datetime with 'Timestamp.to_pydatetime()' "
                            "before subtracting."
                        ) from err
                # We get here in stata tests, fall back to stdlib datetime
                #  method and return stdlib timedelta object
                pass
        elif is_datetime64_object(self):
            # GH#28286 cython semantics for __rsub__, `other` is actually
            #  the Timestamp
            # TODO(cython3): remove this, this moved to __rsub__
            return type(other)(self) - other

        return NotImplemented

    def __rsub__(self, other):
        if PyDateTime_Check(other):
            try:
                return type(self)(other) - self
            except (OverflowError, OutOfBoundsDatetime) as err:
                # We get here in stata tests, fall back to stdlib datetime
                #  method and return stdlib timedelta object
                pass
        elif is_datetime64_object(other):
            return type(self)(other) - self
        return NotImplemented

    # -----------------------------------------------------------------

    cdef int64_t _maybe_convert_value_to_local(self):
        """Convert UTC i8 value to local i8 value if tz exists"""
        cdef:
            int64_t val
            tzinfo own_tz = self.tzinfo
            npy_datetimestruct dts

        if own_tz is not None and not is_utc(own_tz):
            pydatetime_to_dtstruct(self, &dts)
            val = npy_datetimestruct_to_datetime(self._creso, &dts) + self.nanosecond
        else:
            val = self._value
        return val

    @cython.boundscheck(False)
    cdef bint _get_start_end_field(self, str field, freq):
        cdef:
            int64_t val
            dict kwds
            ndarray[uint8_t, cast=True] out
            int month_kw

        if freq:
            kwds = freq.kwds
            month_kw = kwds.get("startingMonth", kwds.get("month", 12))
            freqstr = freq.freqstr
        else:
            month_kw = 12
            freqstr = None

        val = self._maybe_convert_value_to_local()

        out = get_start_end_field(np.array([val], dtype=np.int64),
                                  field, freqstr, month_kw, self._creso)
        return out[0]

    @property
    def is_month_start(self) -> bool:
        """
        Check if the date is the first day of the month.

        Returns
        -------
        bool
            True if the date is the first day of the month.

        See Also
        --------
        Timestamp.is_month_end : Similar property indicating the last day of the month.

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14)
        >>> ts.is_month_start
        False

        >>> ts = pd.Timestamp(2020, 1, 1)
        >>> ts.is_month_start
        True
        """
        return self.day == 1

    @property
    def is_month_end(self) -> bool:
        """
        Check if the date is the last day of the month.

        Returns
        -------
        bool
            True if the date is the last day of the month.

        See Also
        --------
        Timestamp.is_month_start : Similar property indicating month start.

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14)
        >>> ts.is_month_end
        False

        >>> ts = pd.Timestamp(2020, 12, 31)
        >>> ts.is_month_end
        True
        """
        return self.day == self.days_in_month

    @property
    def is_quarter_start(self) -> bool:
        """
        Check if the date is the first day of the quarter.

        Returns
        -------
        bool
            True if date is first day of the quarter.

        See Also
        --------
        Timestamp.is_quarter_end : Similar property indicating the quarter end.
        Timestamp.quarter : Return the quarter of the date.

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14)
        >>> ts.is_quarter_start
        False

        >>> ts = pd.Timestamp(2020, 4, 1)
        >>> ts.is_quarter_start
        True
        """
        return self.day == 1 and self.month % 3 == 1

    @property
    def is_quarter_end(self) -> bool:
        """
        Check if date is last day of the quarter.

        Returns
        -------
        bool
            True if date is last day of the quarter.

        See Also
        --------
        Timestamp.is_quarter_start : Similar property indicating the quarter start.
        Timestamp.quarter : Return the quarter of the date.

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14)
        >>> ts.is_quarter_end
        False

        >>> ts = pd.Timestamp(2020, 3, 31)
        >>> ts.is_quarter_end
        True
        """
        return (self.month % 3) == 0 and self.day == self.days_in_month

    @property
    def is_year_start(self) -> bool:
        """
        Return True if date is first day of the year.

        Returns
        -------
        bool

        See Also
        --------
        Timestamp.is_year_end : Similar property indicating the end of the year.

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14)
        >>> ts.is_year_start
        False

        >>> ts = pd.Timestamp(2020, 1, 1)
        >>> ts.is_year_start
        True
        """
        return self.day == self.month == 1

    @property
    def is_year_end(self) -> bool:
        """
        Return True if date is last day of the year.

        Returns
        -------
        bool

        See Also
        --------
        Timestamp.is_year_start : Similar property indicating the start of the year.

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14)
        >>> ts.is_year_end
        False

        >>> ts = pd.Timestamp(2020, 12, 31)
        >>> ts.is_year_end
        True
        """
        return self.month == 12 and self.day == 31

    @cython.boundscheck(False)
    cdef _get_date_name_field(self, str field, object locale):
        cdef:
            int64_t val
            object[::1] out

        val = self._maybe_convert_value_to_local()

        out = get_date_name_field(np.array([val], dtype=np.int64),
                                  field, locale=locale, reso=self._creso)
        return out[0]

    def day_name(self, locale=None) -> str:
        """
        Return the day name of the Timestamp with specified locale.

        Parameters
        ----------
        locale : str, default None (English locale)
            Locale determining the language in which to return the day name.

        Returns
        -------
        str

        Examples
        --------
        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651')
        >>> ts.day_name()
        'Saturday'

        Analogous for ``pd.NaT``:

        >>> pd.NaT.day_name()
        nan
        """
        return self._get_date_name_field("day_name", locale)

    def month_name(self, locale=None) -> str:
        """
        Return the month name of the Timestamp with specified locale.

        Parameters
        ----------
        locale : str, default None (English locale)
            Locale determining the language in which to return the month name.

        Returns
        -------
        str

        Examples
        --------
        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651')
        >>> ts.month_name()
        'March'

        Analogous for ``pd.NaT``:

        >>> pd.NaT.month_name()
        nan
        """
        return self._get_date_name_field("month_name", locale)

    @property
    def is_leap_year(self) -> bool:
        """
        Return True if year is a leap year.

        Returns
        -------
        bool

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14)
        >>> ts.is_leap_year
        True
        """
        return bool(ccalendar.is_leapyear(self.year))

    @property
    def day_of_week(self) -> int:
        """
        Return day of the week.

        Returns
        -------
        int

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14)
        >>> ts.day_of_week
        5
        """
        return self.weekday()

    @property
    def day_of_year(self) -> int:
        """
        Return the day of the year.

        Returns
        -------
        int

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14)
        >>> ts.day_of_year
        74
        """
        return ccalendar.get_day_of_year(self.year, self.month, self.day)

    @property
    def quarter(self) -> int:
        """
        Return the quarter of the year.

        Returns
        -------
        int

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14)
        >>> ts.quarter
        1
        """
        return ((self.month - 1) // 3) + 1

    @property
    def week(self) -> int:
        """
        Return the week number of the year.

        Returns
        -------
        int

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14)
        >>> ts.week
        11
        """
        return ccalendar.get_week_of_year(self.year, self.month, self.day)

    @property
    def days_in_month(self) -> int:
        """
        Return the number of days in the month.

        Returns
        -------
        int

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14)
        >>> ts.days_in_month
        31
        """
        return ccalendar.get_days_in_month(self.year, self.month)

    # -----------------------------------------------------------------
    # Transformation Methods

    def normalize(self) -> "Timestamp":
        """
        Normalize Timestamp to midnight, preserving tz information.

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14, 15, 30)
        >>> ts.normalize()
        Timestamp('2020-03-14 00:00:00')
        """
        cdef:
            local_val = self._maybe_convert_value_to_local()
            int64_t normalized
            int64_t ppd = periods_per_day(self._creso)
            _Timestamp ts

        normalized = normalize_i8_stamp(local_val, ppd)
        ts = type(self)._from_value_and_reso(normalized, reso=self._creso, tz=None)
        return ts.tz_localize(self.tzinfo)

    # -----------------------------------------------------------------
    # Pickle Methods

    def __reduce_ex__(self, protocol):
        # python 3.6 compat
        # https://bugs.python.org/issue28730
        # now __reduce_ex__ is defined and higher priority than __reduce__
        return self.__reduce__()

    def __setstate__(self, state):
        self._value= state[0]
        self.tzinfo = state[2]

        if len(state) == 3:
            # pre-non-nano pickle
            # TODO: no tests get here 2022-05-10
            reso = NPY_FR_ns
        else:
            reso = state[4]
        self._creso = reso

    def __reduce__(self):
        object_state = self._value, None, self.tzinfo, self._creso
        return (_unpickle_timestamp, object_state)

    # -----------------------------------------------------------------
    # Rendering Methods

    def isoformat(self, sep: str = "T", timespec: str = "auto") -> str:
        """
        Return the time formatted according to ISO 8601.

        The full format looks like 'YYYY-MM-DD HH:MM:SS.mmmmmmnnn'.
        By default, the fractional part is omitted if self.microsecond == 0
        and self.nanosecond == 0.

        If self.tzinfo is not None, the UTC offset is also attached, giving
        giving a full format of 'YYYY-MM-DD HH:MM:SS.mmmmmmnnn+HH:MM'.

        Parameters
        ----------
        sep : str, default 'T'
            String used as the separator between the date and time.

        timespec : str, default 'auto'
            Specifies the number of additional terms of the time to include.
            The valid values are 'auto', 'hours', 'minutes', 'seconds',
            'milliseconds', 'microseconds', and 'nanoseconds'.

        Returns
        -------
        str

        Examples
        --------
        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651')
        >>> ts.isoformat()
        '2020-03-14T15:32:52.192548651'
        >>> ts.isoformat(timespec='microseconds')
        '2020-03-14T15:32:52.192548'
        """
        base_ts = "microseconds" if timespec == "nanoseconds" else timespec
        base = super(_Timestamp, self).isoformat(sep=sep, timespec=base_ts)
        # We need to replace the fake year 1970 with our real year
        base = f"{self.year:04d}-" + base.split("-", 1)[1]

        if self.nanosecond == 0 and timespec != "nanoseconds":
            return base

        if self.tzinfo is not None:
            base1, base2 = base[:-6], base[-6:]
        else:
            base1, base2 = base, ""

        if timespec == "nanoseconds" or (timespec == "auto" and self.nanosecond):
            if self.microsecond or timespec == "nanoseconds":
                base1 += f"{self.nanosecond:03d}"
            else:
                base1 += f".{self.nanosecond:09d}"

        return base1 + base2

    def __repr__(self) -> str:
        stamp = self._repr_base
        zone = None

        if self.tzinfo is not None:
            try:
                stamp += self.strftime("%z")
            except ValueError:
                year2000 = self.replace(year=2000)
                stamp += year2000.strftime("%z")

            zone = get_timezone(self.tzinfo)
            try:
                stamp += zone.strftime(" %%Z")
            except AttributeError:
                # e.g. tzlocal has no `strftime`
                pass

        tz = f", tz='{zone}'" if zone is not None else ""

        return f"Timestamp('{stamp}'{tz})"

    @property
    def _repr_base(self) -> str:
        return f"{self._date_repr} {self._time_repr}"

    @property
    def _date_repr(self) -> str:
        # Ideal here would be self.strftime("%Y-%m-%d"), but
        # the datetime strftime() methods require year >= 1900 and is slower
        return f"{self.year}-{self.month:02d}-{self.day:02d}"

    @property
    def _time_repr(self) -> str:
        result = f"{self.hour:02d}:{self.minute:02d}:{self.second:02d}"

        if self.nanosecond != 0:
            result += f".{self.nanosecond + 1000 * self.microsecond:09d}"
        elif self.microsecond != 0:
            result += f".{self.microsecond:06d}"

        return result

    @property
    def _short_repr(self) -> str:
        # format a Timestamp with only _date_repr if possible
        # otherwise _repr_base
        if (self.hour == 0 and
                self.minute == 0 and
                self.second == 0 and
                self.microsecond == 0 and
                self.nanosecond == 0):
            return self._date_repr
        return self._repr_base

    # -----------------------------------------------------------------
    # Conversion Methods

    @cython.cdivision(False)
    cdef _Timestamp _as_creso(self, NPY_DATETIMEUNIT creso, bint round_ok=True):
        cdef:
            int64_t value

        if creso == self._creso:
            return self

        try:
            value = convert_reso(self._value, self._creso, creso, round_ok=round_ok)
        except OverflowError as err:
            unit = npy_unit_to_abbrev(creso)
            raise OutOfBoundsDatetime(
                f"Cannot cast {self} to unit='{unit}' without overflow."
            ) from err

        return type(self)._from_value_and_reso(value, reso=creso, tz=self.tzinfo)

    def as_unit(self, str unit, bint round_ok=True):
        """
        Convert the underlying int64 representaton to the given unit.

        Parameters
        ----------
        unit : {"ns", "us", "ms", "s"}
        round_ok : bool, default True
            If False and the conversion requires rounding, raise.

        Returns
        -------
        Timestamp

        Examples
        --------
        >>> ts = pd.Timestamp('2023-01-01 00:00:00.01')
        >>> ts
        Timestamp('2023-01-01 00:00:00.010000')
        >>> ts.unit
        'ms'
        >>> ts = ts.as_unit('s')
        >>> ts
        Timestamp('2023-01-01 00:00:00')
        >>> ts.unit
        's'
        """
        dtype = np.dtype(f"M8[{unit}]")
        reso = get_unit_from_dtype(dtype)
        try:
            return self._as_creso(reso, round_ok=round_ok)
        except OverflowError as err:
            raise OutOfBoundsDatetime(
                f"Cannot cast {self} to unit='{unit}' without overflow."
            ) from err

    @property
    def asm8(self) -> np.datetime64:
        """
        Return numpy datetime64 format in nanoseconds.

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14, 15)
        >>> ts.asm8
        numpy.datetime64('2020-03-14T15:00:00.000000')
        """
        return self.to_datetime64()

    def timestamp(self):
        """
        Return POSIX timestamp as float.

        Examples
        --------
        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548')
        >>> ts.timestamp()
        1584199972.192548
        """
        # GH 17329
        # Note: Naive timestamps will not match datetime.stdlib

        denom = periods_per_second(self._creso)

        return round(self._value/ denom, 6)

    cpdef datetime to_pydatetime(_Timestamp self, bint warn=True):
        """
        Convert a Timestamp object to a native Python datetime object.

        If warn=True, issue a warning if nanoseconds is nonzero.

        Examples
        --------
        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548')
        >>> ts.to_pydatetime()
        datetime.datetime(2020, 3, 14, 15, 32, 52, 192548)

        Analogous for ``pd.NaT``:

        >>> pd.NaT.to_pydatetime()
        NaT
        """
        if self.nanosecond != 0 and warn:
            warnings.warn("Discarding nonzero nanoseconds in conversion.",
                          UserWarning, stacklevel=find_stack_level())

        return datetime(self.year, self.month, self.day,
                        self.hour, self.minute, self.second,
                        self.microsecond, self.tzinfo, fold=self.fold)

    cpdef to_datetime64(self):
        """
        Return a numpy.datetime64 object with same precision.

        Examples
        --------
        >>> ts = pd.Timestamp(year=2023, month=1, day=1,
        ...                   hour=10, second=15)
        >>> ts
        Timestamp('2023-01-01 10:00:15')
        >>> ts.to_datetime64()
        numpy.datetime64('2023-01-01T10:00:15.000000')
        """
        # TODO: find a way to construct dt64 directly from _reso
        abbrev = npy_unit_to_abbrev(self._creso)
        return np.datetime64(self._value, abbrev)

    def to_numpy(self, dtype=None, copy=False) -> np.datetime64:
        """
        Convert the Timestamp to a NumPy datetime64.

        This is an alias method for `Timestamp.to_datetime64()`. The dtype and
        copy parameters are available here only for compatibility. Their values
        will not affect the return value.

        Returns
        -------
        numpy.datetime64

        See Also
        --------
        DatetimeIndex.to_numpy : Similar method for DatetimeIndex.

        Examples
        --------
        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651')
        >>> ts.to_numpy()
        numpy.datetime64('2020-03-14T15:32:52.192548651')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.to_numpy()
        numpy.datetime64('NaT')
        """
        if dtype is not None or copy is not False:
            raise ValueError(
                "Timestamp.to_numpy dtype and copy arguments are ignored."
            )
        return self.to_datetime64()

    def to_period(self, freq=None):
        """
        Return an period of which this timestamp is an observation.

        Examples
        --------
        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651')
        >>> # Year end frequency
        >>> ts.to_period(freq='Y')
        Period('2020', 'A-DEC')

        >>> # Month end frequency
        >>> ts.to_period(freq='M')
        Period('2020-03', 'M')

        >>> # Weekly frequency
        >>> ts.to_period(freq='W')
        Period('2020-03-09/2020-03-15', 'W-SUN')

        >>> # Quarter end frequency
        >>> ts.to_period(freq='Q')
        Period('2020Q1', 'Q-DEC')
        """
        from pandas import Period

        if self.tz is not None:
            # GH#21333
            warnings.warn(
                "Converting to Period representation will drop timezone information.",
                UserWarning,
                stacklevel=find_stack_level(),
            )

        return Period(self, freq=freq)


# ----------------------------------------------------------------------

# Python front end to C extension type _Timestamp
# This serves as the box for datetime64


class Timestamp(_Timestamp):
    """
    Pandas replacement for python datetime.datetime object.

    Timestamp is the pandas equivalent of python's Datetime
    and is interchangeable with it in most cases. It's the type used
    for the entries that make up a DatetimeIndex, and other timeseries
    oriented data structures in pandas.

    Parameters
    ----------
    ts_input : datetime-like, str, int, float
        Value to be converted to Timestamp.
    year, month, day : int
    hour, minute, second, microsecond : int, optional, default 0
    tzinfo : datetime.tzinfo, optional, default None
    nanosecond : int, optional, default 0
    tz : str, pytz.timezone, dateutil.tz.tzfile or None
        Time zone for time which Timestamp will have.
    unit : str
        Unit used for conversion if ts_input is of type int or float. The
        valid values are 'D', 'h', 'm', 's', 'ms', 'us', and 'ns'. For
        example, 's' means seconds and 'ms' means milliseconds.

        For float inputs, the result will be stored in nanoseconds, and
        the unit attribute will be set as ``'ns'``.
    fold : {0, 1}, default None, keyword-only
        Due to daylight saving time, one wall clock time can occur twice
        when shifting from summer to winter time; fold describes whether the
        datetime-like corresponds  to the first (0) or the second time (1)
        the wall clock hits the ambiguous time.

    Notes
    -----
    There are essentially three calling conventions for the constructor. The
    primary form accepts four parameters. They can be passed by position or
    keyword.

    The other two forms mimic the parameters from ``datetime.datetime``. They
    can be passed by either position or keyword, but not both mixed together.

    Examples
    --------
    Using the primary calling convention:

    This converts a datetime-like string

    >>> pd.Timestamp('2017-01-01T12')
    Timestamp('2017-01-01 12:00:00')

    This converts a float representing a Unix epoch in units of seconds

    >>> pd.Timestamp(1513393355.5, unit='s')
    Timestamp('2017-12-16 03:02:35.500000')

    This converts an int representing a Unix-epoch in units of seconds
    and for a particular timezone

    >>> pd.Timestamp(1513393355, unit='s', tz='US/Pacific')
    Timestamp('2017-12-15 19:02:35-0800', tz='US/Pacific')

    Using the other two forms that mimic the API for ``datetime.datetime``:

    >>> pd.Timestamp(2017, 1, 1, 12)
    Timestamp('2017-01-01 12:00:00')

    >>> pd.Timestamp(year=2017, month=1, day=1, hour=12)
    Timestamp('2017-01-01 12:00:00')
    """

    @classmethod
    def fromordinal(cls, ordinal, tz=None):
        """
        Construct a timestamp from a a proleptic Gregorian ordinal.

        Parameters
        ----------
        ordinal : int
            Date corresponding to a proleptic Gregorian ordinal.
        tz : str, pytz.timezone, dateutil.tz.tzfile or None
            Time zone for the Timestamp.

        Notes
        -----
        By definition there cannot be any tz info on the ordinal itself.

        Examples
        --------
        >>> pd.Timestamp.fromordinal(737425)
        Timestamp('2020-01-01 00:00:00')
        """
        return cls(datetime.fromordinal(ordinal), tz=tz)

    @classmethod
    def now(cls, tz=None):
        """
        Return new Timestamp object representing current time local to tz.

        Parameters
        ----------
        tz : str or timezone object, default None
            Timezone to localize to.

        Examples
        --------
        >>> pd.Timestamp.now()  # doctest: +SKIP
        Timestamp('2020-11-16 22:06:16.378782')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.now()
        NaT
        """
        if isinstance(tz, str):
            tz = maybe_get_tz(tz)
        return cls(datetime.now(tz))

    @classmethod
    def today(cls, tz=None):
        """
        Return the current time in the local timezone.

        This differs from datetime.today() in that it can be localized to a
        passed timezone.

        Parameters
        ----------
        tz : str or timezone object, default None
            Timezone to localize to.

        Examples
        --------
        >>> pd.Timestamp.today()    # doctest: +SKIP
        Timestamp('2020-11-16 22:37:39.969883')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.today()
        NaT
        """
        return cls.now(tz)

    @classmethod
    def utcnow(cls):
        """
        Timestamp.utcnow()

        Return a new Timestamp representing UTC day and time.

        Examples
        --------
        >>> pd.Timestamp.utcnow()   # doctest: +SKIP
        Timestamp('2020-11-16 22:50:18.092888+0000', tz='UTC')
        """
        return cls.now(UTC)

    @classmethod
    def utcfromtimestamp(cls, ts):
        """
        Timestamp.utcfromtimestamp(ts)

        Construct a timezone-aware UTC datetime from a POSIX timestamp.

        Notes
        -----
        Timestamp.utcfromtimestamp behavior differs from datetime.utcfromtimestamp
        in returning a timezone-aware object.

        Examples
        --------
        >>> pd.Timestamp.utcfromtimestamp(1584199972)
        Timestamp('2020-03-14 15:32:52+0000', tz='UTC')
        """
        # GH#22451
        return cls.fromtimestamp(ts, tz="UTC")

    @classmethod
    def fromtimestamp(cls, ts, tz=None):
        """
        Timestamp.fromtimestamp(ts)

        Transform timestamp[, tz] to tz's local time from POSIX timestamp.

        Examples
        --------
        >>> pd.Timestamp.fromtimestamp(1584199972)  # doctest: +SKIP
        Timestamp('2020-03-14 15:32:52')

        Note that the output may change depending on your local time.
        """
        tz = maybe_get_tz(tz)
        return cls(datetime.fromtimestamp(ts, tz))

    def strftime(self, format):
        """
        Return a formatted string of the Timestamp.

        Parameters
        ----------
        format : str
            Format string to convert Timestamp to string.
            See strftime documentation for more information on the format string:
            https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior.

        Examples
        --------
        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651')
        >>> ts.strftime('%Y-%m-%d %X')
        '2020-03-14 15:32:52'
        """
        try:
            _dt = datetime(self.year, self.month, self.day,
                           self.hour, self.minute, self.second,
                           self.microsecond, self.tzinfo, fold=self.fold)
        except ValueError as err:
            raise NotImplementedError(
                "strftime not yet supported on Timestamps which "
                "are outside the range of Python's standard library. "
                "For now, please call the components you need (such as `.year` "
                "and `.month`) and construct your string from there."
            ) from err
        return _dt.strftime(format)

    def ctime(self):
        """
        Return ctime() style string.

        Examples
        --------
        >>> ts = pd.Timestamp('2023-01-01 10:00:00.00')
        >>> ts
        Timestamp('2023-01-01 10:00:00')
        >>> ts.ctime()
        'Sun Jan  1 10:00:00 2023'
        """
        try:
            _dt = datetime(self.year, self.month, self.day,
                           self.hour, self.minute, self.second,
                           self.microsecond, self.tzinfo, fold=self.fold)
        except ValueError as err:
            raise NotImplementedError(
                "ctime not yet supported on Timestamps which "
                "are outside the range of Python's standard library. "
                "For now, please call the components you need (such as `.year` "
                "and `.month`) and construct your string from there."
            ) from err
        return _dt.ctime()

    def date(self):
        """
        Return date object with same year, month and day.

        Examples
        --------
        >>> ts = pd.Timestamp('2023-01-01 10:00:00.00')
        >>> ts
        Timestamp('2023-01-01 10:00:00')
        >>> ts.date()
        datetime.date(2023, 1, 1)
        """
        try:
            _dt = dt.date(self.year, self.month, self.day)
        except ValueError as err:
            raise NotImplementedError(
                "date not yet supported on Timestamps which "
                "are outside the range of Python's standard library. "
            ) from err
        return _dt

    def dst(self):
        """
        Return the daylight saving time (DST) adjustment.

        Examples
        --------
        >>> ts = pd.Timestamp('2000-06-01 00:00:00', tz='Europe/Brussels')
        >>> ts
        Timestamp('2000-06-01 00:00:00+0200', tz='Europe/Brussels')
        >>> ts.dst()
        datetime.timedelta(seconds=3600)
        """
        return super().dst()

    def isocalendar(self):
        """
        Return a named tuple containing ISO year, week number, and weekday.

        Examples
        --------
        >>> ts = pd.Timestamp('2023-01-01 10:00:00')
        >>> ts
        Timestamp('2023-01-01 10:00:00')
        >>> ts.isocalendar()
        datetime.IsoCalendarDate(year=2022, week=52, weekday=7)
        """
        try:
            _dt = datetime(self.year, self.month, self.day,
                           self.hour, self.minute, self.second,
                           self.microsecond, self.tzinfo, fold=self.fold)
        except ValueError as err:
            raise NotImplementedError(
                "isocalendar not yet supported on Timestamps which "
                "are outside the range of Python's standard library. "
            ) from err
        return _dt.isocalendar()

    def tzname(self):
        """
        Return time zone name.

        Examples
        --------
        >>> ts = pd.Timestamp('2023-01-01 10:00:00', tz='Europe/Brussels')
        >>> ts
        Timestamp('2023-01-01 10:00:00+0100', tz='Europe/Brussels')
        >>> ts.tzname()
        'CET'
        """
        return super().tzname()

    def utcoffset(self):
        """
        Return utc offset.

        Examples
        --------
        >>> ts = pd.Timestamp('2023-01-01 10:00:00', tz='Europe/Brussels')
        >>> ts
        Timestamp('2023-01-01 10:00:00+0100', tz='Europe/Brussels')
        >>> ts.utcoffset()
        datetime.timedelta(seconds=3600)
        """
        return super().utcoffset()

    def utctimetuple(self):
        """
        Return UTC time tuple, compatible with time.localtime().

        Examples
        --------
        >>> ts = pd.Timestamp('2023-01-01 10:00:00', tz='Europe/Brussels')
        >>> ts
        Timestamp('2023-01-01 10:00:00+0100', tz='Europe/Brussels')
        >>> ts.utctimetuple()
        time.struct_time(tm_year=2023, tm_mon=1, tm_mday=1, tm_hour=9,
        tm_min=0, tm_sec=0, tm_wday=6, tm_yday=1, tm_isdst=0)
        """
        return super().utctimetuple()

    def time(self):
        """
        Return time object with same time but with tzinfo=None.

        Examples
        --------
        >>> ts = pd.Timestamp('2023-01-01 10:00:00')
        >>> ts
        Timestamp('2023-01-01 10:00:00')
        >>> ts.time()
        datetime.time(10, 0)
        """
        return super().time()

    def timetuple(self):
        """
        Return time tuple, compatible with time.localtime().

        Examples
        --------
        >>> ts = pd.Timestamp('2023-01-01 10:00:00')
        >>> ts
        Timestamp('2023-01-01 10:00:00')
        >>> ts.timetuple()
        time.struct_time(tm_year=2023, tm_mon=1, tm_mday=1,
        tm_hour=10, tm_min=0, tm_sec=0, tm_wday=6, tm_yday=1, tm_isdst=-1)
        """
        try:
            _dt = datetime(self.year, self.month, self.day,
                           self.hour, self.minute, self.second,
                           self.microsecond, self.tzinfo, fold=self.fold)
        except ValueError as err:
            raise NotImplementedError(
                "timetuple not yet supported on Timestamps which "
                "are outside the range of Python's standard library. "
            ) from err
        return _dt.timetuple()

    def timetz(self):
        """
        Return time object with same time and tzinfo.

        Examples
        --------
        >>> ts = pd.Timestamp('2023-01-01 10:00:00', tz='Europe/Brussels')
        >>> ts
        Timestamp('2023-01-01 10:00:00+0100', tz='Europe/Brussels')
        >>> ts.timetz()
        datetime.time(10, 0, tzinfo=<DstTzInfo 'Europe/Brussels' CET+1:00:00 STD>)
        """
        return super().timetz()

    def toordinal(self):
        """
        Return proleptic Gregorian ordinal. January 1 of year 1 is day 1.

        Examples
        --------
        >>> ts = pd.Timestamp('2023-01-01 10:00:50')
        >>> ts
        Timestamp('2023-01-01 10:00:50')
        >>> ts.toordinal()
        738521
        """
        try:
            _dt = datetime(self.year, self.month, self.day,
                           self.hour, self.minute, self.second,
                           self.microsecond, self.tzinfo, fold=self.fold)
        except ValueError as err:
            raise NotImplementedError(
                "toordinal not yet supported on Timestamps which "
                "are outside the range of Python's standard library. "
            ) from err
        return _dt.toordinal()

    # Issue 25016.
    @classmethod
    def strptime(cls, date_string, format):
        """
        Timestamp.strptime(string, format)

        Function is not implemented. Use pd.to_datetime().

        Examples
        --------
        >>> pd.Timestamp.strptime("2023-01-01", "%d/%m/%y")
        Traceback (most recent call last):
        NotImplementedError
        """
        raise NotImplementedError(
            "Timestamp.strptime() is not implemented. "
            "Use to_datetime() to parse date strings."
        )

    @classmethod
    def combine(cls, date, time):
        """
        Timestamp.combine(date, time)

        Combine date, time into datetime with same date and time fields.

        Examples
        --------
        >>> from datetime import date, time
        >>> pd.Timestamp.combine(date(2020, 3, 14), time(15, 30, 15))
        Timestamp('2020-03-14 15:30:15')
        """
        return cls(datetime.combine(date, time))

    def __new__(
        cls,
        object ts_input=_no_input,
        year=None,
        month=None,
        day=None,
        hour=None,
        minute=None,
        second=None,
        microsecond=None,
        tzinfo_type tzinfo=None,
        *,
        nanosecond=None,
        tz=None,
        unit=None,
        fold=None,
    ):
        # The parameter list folds together legacy parameter names (the first
        # four) and positional and keyword parameter names from pydatetime.
        #
        # There are three calling forms:
        #
        # - In the legacy form, the first parameter, ts_input, is required
        #   and may be datetime-like, str, int, or float. The second
        #   parameter, offset, is optional and may be str or DateOffset.
        #
        # - ints in the first, second, and third arguments indicate
        #   pydatetime positional arguments. Only the first 8 arguments
        #   (standing in for year, month, day, hour, minute, second,
        #   microsecond, tzinfo) may be non-None. As a shortcut, we just
        #   check that the second argument is an int.
        #
        # - Nones for the first four (legacy) arguments indicate pydatetime
        #   keyword arguments. year, month, and day are required. As a
        #   shortcut, we just check that the first argument was not passed.
        #
        # Mixing pydatetime positional and keyword arguments is forbidden!

        cdef:
            _TSObject ts
            tzinfo_type tzobj

        _date_attributes = [year, month, day, hour, minute, second,
                            microsecond, nanosecond]

        if tzinfo is not None:
            # GH#17690 tzinfo must be a datetime.tzinfo object, ensured
            #  by the cython annotation.
            if tz is not None:
                raise ValueError("Can provide at most one of tz, tzinfo")

            # User passed tzinfo instead of tz; avoid silently ignoring
            tz, tzinfo = tzinfo, None

        # Allow fold only for unambiguous input
        if fold is not None:
            if fold not in [0, 1]:
                raise ValueError(
                    "Valid values for the fold argument are None, 0, or 1."
                )

            if (ts_input is not _no_input and not (
                    PyDateTime_Check(ts_input) and
                    getattr(ts_input, "tzinfo", None) is None)):
                raise ValueError(
                    "Cannot pass fold with possibly unambiguous input: int, "
                    "float, numpy.datetime64, str, or timezone-aware "
                    "datetime-like. Pass naive datetime-like or build "
                    "Timestamp from components."
                )

            if tz is not None and PyTZInfo_Check(tz) and treat_tz_as_pytz(tz):
                raise ValueError(
                    "pytz timezones do not support fold. Please use dateutil "
                    "timezones."
                )

            if hasattr(ts_input, "fold"):
                ts_input = ts_input.replace(fold=fold)

        # GH 30543 if pd.Timestamp already passed, return it
        # check that only ts_input is passed
        # checking verbosely, because cython doesn't optimize
        # list comprehensions (as of cython 0.29.x)
        if (isinstance(ts_input, _Timestamp) and
                tz is None and unit is None and year is None and
                month is None and day is None and hour is None and
                minute is None and second is None and
                microsecond is None and nanosecond is None and
                tzinfo is None):
            return ts_input
        elif isinstance(ts_input, str):
            # User passed a date string to parse.
            # Check that the user didn't also pass a date attribute kwarg.
            if any(arg is not None for arg in _date_attributes):
                raise ValueError(
                    "Cannot pass a date attribute keyword "
                    "argument when passing a date string; 'tz' is keyword-only"
                )

        elif ts_input is _no_input:
            # GH 31200
            # When year, month or day is not given, we call the datetime
            # constructor to make sure we get the same error message
            # since Timestamp inherits datetime
            datetime_kwargs = {
                "hour": hour or 0,
                "minute": minute or 0,
                "second": second or 0,
                "microsecond": microsecond or 0,
                "fold": fold or 0
            }
            if year is not None:
                datetime_kwargs["year"] = year
            if month is not None:
                datetime_kwargs["month"] = month
            if day is not None:
                datetime_kwargs["day"] = day

            ts_input = datetime(**datetime_kwargs)

        elif is_integer_object(year):
            # User passed positional arguments:
            # Timestamp(year, month, day[, hour[, minute[, second[,
            # microsecond[, tzinfo]]]]])
            ts_input = datetime(ts_input, year, month, day or 0,
                                hour or 0, minute or 0, second or 0, fold=fold or 0)
            unit = None

        if getattr(ts_input, "tzinfo", None) is not None and tz is not None:
            raise ValueError("Cannot pass a datetime or Timestamp with tzinfo with "
                             "the tz parameter. Use tz_convert instead.")

        tzobj = maybe_get_tz(tz)
        if tzobj is not None and is_datetime64_object(ts_input):
            # GH#24559, GH#42288 As of 2.0 we treat datetime64 as
            #  wall-time (consistent with DatetimeIndex)
            return cls(ts_input).tz_localize(tzobj)

        if nanosecond is None:
            nanosecond = 0
        elif not (999 >= nanosecond >= 0):
            raise ValueError("nanosecond must be in 0..999")

        ts = convert_to_tsobject(ts_input, tzobj, unit, 0, 0, nanosecond)

        if ts.value == NPY_NAT:
            return NaT

        return create_timestamp_from_ts(ts.value, ts.dts, ts.tzinfo, ts.fold, ts.creso)

    def _round(self, freq, mode, ambiguous="raise", nonexistent="raise"):
        cdef:
            int64_t nanos

        freq = to_offset(freq)
        freq.nanos  # raises on non-fixed freq
        nanos = delta_to_nanoseconds(freq, self._creso)
        if nanos == 0:
            if freq.nanos == 0:
                raise ValueError("Division by zero in rounding")

            # e.g. self.unit == "s" and sub-second freq
            return self

        # TODO: problem if nanos==0

        if self.tz is not None:
            value = self.tz_localize(None)._value
        else:
            value = self._value

        value = np.array([value], dtype=np.int64)

        # Will only ever contain 1 element for timestamp
        try:
            r = round_nsint64(value, mode, nanos)[0]
        except OverflowError as err:
            raise OutOfBoundsDatetime(
                f"Cannot round {self} to freq={freq} without overflow"
            ) from err

        result = Timestamp._from_value_and_reso(r, self._creso, None)
        if self.tz is not None:
            result = result.tz_localize(
                self.tz, ambiguous=ambiguous, nonexistent=nonexistent
            )
        return result

    def round(self, freq, ambiguous="raise", nonexistent="raise"):
        """
        Round the Timestamp to the specified resolution.

        Parameters
        ----------
        freq : str
            Frequency string indicating the rounding resolution.
        ambiguous : bool or {'raise', 'NaT'}, default 'raise'
            The behavior is as follows:

            * bool contains flags to determine if time is dst or not (note
              that this flag is only applicable for ambiguous fall dst dates).
            * 'NaT' will return NaT for an ambiguous time.
            * 'raise' will raise an AmbiguousTimeError for an ambiguous time.

        nonexistent : {'raise', 'shift_forward', 'shift_backward, 'NaT', \
timedelta}, default 'raise'
            A nonexistent time does not exist in a particular timezone
            where clocks moved forward due to DST.

            * 'shift_forward' will shift the nonexistent time forward to the
              closest existing time.
            * 'shift_backward' will shift the nonexistent time backward to the
              closest existing time.
            * 'NaT' will return NaT where there are nonexistent times.
            * timedelta objects will shift nonexistent times by the timedelta.
            * 'raise' will raise an NonExistentTimeError if there are
              nonexistent times.

        Returns
        -------
        a new Timestamp rounded to the given resolution of `freq`

        Raises
        ------
        ValueError if the freq cannot be converted

        Notes
        -----
        If the Timestamp has a timezone, rounding will take place relative to the
        local ("wall") time and re-localized to the same timezone. When rounding
        near daylight savings time, use ``nonexistent`` and ``ambiguous`` to
        control the re-localization behavior.

        Examples
        --------
        Create a timestamp object:

        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651')

        A timestamp can be rounded using multiple frequency units:

        >>> ts.round(freq='H') # hour
        Timestamp('2020-03-14 16:00:00')

        >>> ts.round(freq='T') # minute
        Timestamp('2020-03-14 15:33:00')

        >>> ts.round(freq='S') # seconds
        Timestamp('2020-03-14 15:32:52')

        >>> ts.round(freq='L') # milliseconds
        Timestamp('2020-03-14 15:32:52.193000')

        ``freq`` can also be a multiple of a single unit, like '5T' (i.e.  5 minutes):

        >>> ts.round(freq='5T')
        Timestamp('2020-03-14 15:35:00')

        or a combination of multiple units, like '1H30T' (i.e. 1 hour and 30 minutes):

        >>> ts.round(freq='1H30T')
        Timestamp('2020-03-14 15:00:00')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.round()
        NaT

        When rounding near a daylight savings time transition, use ``ambiguous`` or
        ``nonexistent`` to control how the timestamp should be re-localized.

        >>> ts_tz = pd.Timestamp("2021-10-31 01:30:00").tz_localize("Europe/Amsterdam")

        >>> ts_tz.round("H", ambiguous=False)
        Timestamp('2021-10-31 02:00:00+0100', tz='Europe/Amsterdam')

        >>> ts_tz.round("H", ambiguous=True)
        Timestamp('2021-10-31 02:00:00+0200', tz='Europe/Amsterdam')
        """
        return self._round(
            freq, RoundTo.NEAREST_HALF_EVEN, ambiguous, nonexistent
        )

    def floor(self, freq, ambiguous="raise", nonexistent="raise"):
        """
        Return a new Timestamp floored to this resolution.

        Parameters
        ----------
        freq : str
            Frequency string indicating the flooring resolution.
        ambiguous : bool or {'raise', 'NaT'}, default 'raise'
            The behavior is as follows:

            * bool contains flags to determine if time is dst or not (note
              that this flag is only applicable for ambiguous fall dst dates).
            * 'NaT' will return NaT for an ambiguous time.
            * 'raise' will raise an AmbiguousTimeError for an ambiguous time.

        nonexistent : {'raise', 'shift_forward', 'shift_backward, 'NaT', \
timedelta}, default 'raise'
            A nonexistent time does not exist in a particular timezone
            where clocks moved forward due to DST.

            * 'shift_forward' will shift the nonexistent time forward to the
              closest existing time.
            * 'shift_backward' will shift the nonexistent time backward to the
              closest existing time.
            * 'NaT' will return NaT where there are nonexistent times.
            * timedelta objects will shift nonexistent times by the timedelta.
            * 'raise' will raise an NonExistentTimeError if there are
              nonexistent times.

        Raises
        ------
        ValueError if the freq cannot be converted.

        Notes
        -----
        If the Timestamp has a timezone, flooring will take place relative to the
        local ("wall") time and re-localized to the same timezone. When flooring
        near daylight savings time, use ``nonexistent`` and ``ambiguous`` to
        control the re-localization behavior.

        Examples
        --------
        Create a timestamp object:

        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651')

        A timestamp can be floored using multiple frequency units:

        >>> ts.floor(freq='H') # hour
        Timestamp('2020-03-14 15:00:00')

        >>> ts.floor(freq='T') # minute
        Timestamp('2020-03-14 15:32:00')

        >>> ts.floor(freq='S') # seconds
        Timestamp('2020-03-14 15:32:52')

        >>> ts.floor(freq='N') # nanoseconds
        Timestamp('2020-03-14 15:32:52.192548651')

        ``freq`` can also be a multiple of a single unit, like '5T' (i.e.  5 minutes):

        >>> ts.floor(freq='5T')
        Timestamp('2020-03-14 15:30:00')

        or a combination of multiple units, like '1H30T' (i.e. 1 hour and 30 minutes):

        >>> ts.floor(freq='1H30T')
        Timestamp('2020-03-14 15:00:00')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.floor()
        NaT

        When rounding near a daylight savings time transition, use ``ambiguous`` or
        ``nonexistent`` to control how the timestamp should be re-localized.

        >>> ts_tz = pd.Timestamp("2021-10-31 03:30:00").tz_localize("Europe/Amsterdam")

        >>> ts_tz.floor("2H", ambiguous=False)
        Timestamp('2021-10-31 02:00:00+0100', tz='Europe/Amsterdam')

        >>> ts_tz.floor("2H", ambiguous=True)
        Timestamp('2021-10-31 02:00:00+0200', tz='Europe/Amsterdam')
        """
        return self._round(freq, RoundTo.MINUS_INFTY, ambiguous, nonexistent)

    def ceil(self, freq, ambiguous="raise", nonexistent="raise"):
        """
        Return a new Timestamp ceiled to this resolution.

        Parameters
        ----------
        freq : str
            Frequency string indicating the ceiling resolution.
        ambiguous : bool or {'raise', 'NaT'}, default 'raise'
            The behavior is as follows:

            * bool contains flags to determine if time is dst or not (note
              that this flag is only applicable for ambiguous fall dst dates).
            * 'NaT' will return NaT for an ambiguous time.
            * 'raise' will raise an AmbiguousTimeError for an ambiguous time.

        nonexistent : {'raise', 'shift_forward', 'shift_backward, 'NaT', \
timedelta}, default 'raise'
            A nonexistent time does not exist in a particular timezone
            where clocks moved forward due to DST.

            * 'shift_forward' will shift the nonexistent time forward to the
              closest existing time.
            * 'shift_backward' will shift the nonexistent time backward to the
              closest existing time.
            * 'NaT' will return NaT where there are nonexistent times.
            * timedelta objects will shift nonexistent times by the timedelta.
            * 'raise' will raise an NonExistentTimeError if there are
              nonexistent times.

        Raises
        ------
        ValueError if the freq cannot be converted.

        Notes
        -----
        If the Timestamp has a timezone, ceiling will take place relative to the
        local ("wall") time and re-localized to the same timezone. When ceiling
        near daylight savings time, use ``nonexistent`` and ``ambiguous`` to
        control the re-localization behavior.

        Examples
        --------
        Create a timestamp object:

        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651')

        A timestamp can be ceiled using multiple frequency units:

        >>> ts.ceil(freq='H') # hour
        Timestamp('2020-03-14 16:00:00')

        >>> ts.ceil(freq='T') # minute
        Timestamp('2020-03-14 15:33:00')

        >>> ts.ceil(freq='S') # seconds
        Timestamp('2020-03-14 15:32:53')

        >>> ts.ceil(freq='U') # microseconds
        Timestamp('2020-03-14 15:32:52.192549')

        ``freq`` can also be a multiple of a single unit, like '5T' (i.e.  5 minutes):

        >>> ts.ceil(freq='5T')
        Timestamp('2020-03-14 15:35:00')

        or a combination of multiple units, like '1H30T' (i.e. 1 hour and 30 minutes):

        >>> ts.ceil(freq='1H30T')
        Timestamp('2020-03-14 16:30:00')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.ceil()
        NaT

        When rounding near a daylight savings time transition, use ``ambiguous`` or
        ``nonexistent`` to control how the timestamp should be re-localized.

        >>> ts_tz = pd.Timestamp("2021-10-31 01:30:00").tz_localize("Europe/Amsterdam")

        >>> ts_tz.ceil("H", ambiguous=False)
        Timestamp('2021-10-31 02:00:00+0100', tz='Europe/Amsterdam')

        >>> ts_tz.ceil("H", ambiguous=True)
        Timestamp('2021-10-31 02:00:00+0200', tz='Europe/Amsterdam')
        """
        return self._round(freq, RoundTo.PLUS_INFTY, ambiguous, nonexistent)

    @property
    def tz(self):
        """
        Alias for tzinfo.

        Examples
        --------
        >>> ts = pd.Timestamp(1584226800, unit='s', tz='Europe/Stockholm')
        >>> ts.tz
        <DstTzInfo 'Europe/Stockholm' CET+1:00:00 STD>
        """
        return self.tzinfo

    @tz.setter
    def tz(self, value):
        # GH 3746: Prevent localizing or converting the index by setting tz
        raise AttributeError(
            "Cannot directly set timezone. "
            "Use tz_localize() or tz_convert() as appropriate"
        )

    def tz_localize(self, tz, ambiguous="raise", nonexistent="raise"):
        """
        Localize the Timestamp to a timezone.

        Convert naive Timestamp to local time zone or remove
        timezone from timezone-aware Timestamp.

        Parameters
        ----------
        tz : str, pytz.timezone, dateutil.tz.tzfile or None
            Time zone for time which Timestamp will be converted to.
            None will remove timezone holding local time.

        ambiguous : bool, 'NaT', default 'raise'
            When clocks moved backward due to DST, ambiguous times may arise.
            For example in Central European Time (UTC+01), when going from
            03:00 DST to 02:00 non-DST, 02:30:00 local time occurs both at
            00:30:00 UTC and at 01:30:00 UTC. In such a situation, the
            `ambiguous` parameter dictates how ambiguous times should be
            handled.

            The behavior is as follows:

            * bool contains flags to determine if time is dst or not (note
              that this flag is only applicable for ambiguous fall dst dates).
            * 'NaT' will return NaT for an ambiguous time.
            * 'raise' will raise an AmbiguousTimeError for an ambiguous time.

        nonexistent : 'shift_forward', 'shift_backward, 'NaT', timedelta, \
default 'raise'
            A nonexistent time does not exist in a particular timezone
            where clocks moved forward due to DST.

            The behavior is as follows:

            * 'shift_forward' will shift the nonexistent time forward to the
              closest existing time.
            * 'shift_backward' will shift the nonexistent time backward to the
              closest existing time.
            * 'NaT' will return NaT where there are nonexistent times.
            * timedelta objects will shift nonexistent times by the timedelta.
            * 'raise' will raise an NonExistentTimeError if there are
              nonexistent times.

        Returns
        -------
        localized : Timestamp

        Raises
        ------
        TypeError
            If the Timestamp is tz-aware and tz is not None.

        Examples
        --------
        Create a naive timestamp object:

        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651')
        >>> ts
        Timestamp('2020-03-14 15:32:52.192548651')

        Add 'Europe/Stockholm' as timezone:

        >>> ts.tz_localize(tz='Europe/Stockholm')
        Timestamp('2020-03-14 15:32:52.192548651+0100', tz='Europe/Stockholm')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.tz_localize()
        NaT
        """
        if not isinstance(ambiguous, bool) and ambiguous not in {"NaT", "raise"}:
            raise ValueError(
                        "'ambiguous' parameter must be one of: "
                        "True, False, 'NaT', 'raise' (default)"
                    )

        nonexistent_options = ("raise", "NaT", "shift_forward", "shift_backward")
        if nonexistent not in nonexistent_options and not PyDelta_Check(nonexistent):
            raise ValueError(
                "The nonexistent argument must be one of 'raise', "
                "'NaT', 'shift_forward', 'shift_backward' or a timedelta object"
            )

        if self.tzinfo is None:
            # tz naive, localize
            tz = maybe_get_tz(tz)
            if not isinstance(ambiguous, str):
                ambiguous = [ambiguous]
            value = tz_localize_to_utc_single(self._value, tz,
                                              ambiguous=ambiguous,
                                              nonexistent=nonexistent,
                                              creso=self._creso)
        elif tz is None:
            # reset tz
            value = tz_convert_from_utc_single(self._value, self.tz, creso=self._creso)

        else:
            raise TypeError(
                "Cannot localize tz-aware Timestamp, use tz_convert for conversions"
            )

        out = type(self)._from_value_and_reso(value, self._creso, tz=tz)
        return out

    def tz_convert(self, tz):
        """
        Convert timezone-aware Timestamp to another time zone.

        Parameters
        ----------
        tz : str, pytz.timezone, dateutil.tz.tzfile or None
            Time zone for time which Timestamp will be converted to.
            None will remove timezone holding UTC time.

        Returns
        -------
        converted : Timestamp

        Raises
        ------
        TypeError
            If Timestamp is tz-naive.

        Examples
        --------
        Create a timestamp object with UTC timezone:

        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651', tz='UTC')
        >>> ts
        Timestamp('2020-03-14 15:32:52.192548651+0000', tz='UTC')

        Change to Tokyo timezone:

        >>> ts.tz_convert(tz='Asia/Tokyo')
        Timestamp('2020-03-15 00:32:52.192548651+0900', tz='Asia/Tokyo')

        Can also use ``astimezone``:

        >>> ts.astimezone(tz='Asia/Tokyo')
        Timestamp('2020-03-15 00:32:52.192548651+0900', tz='Asia/Tokyo')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.tz_convert(tz='Asia/Tokyo')
        NaT
        """
        if self.tzinfo is None:
            # tz naive, use tz_localize
            raise TypeError(
                "Cannot convert tz-naive Timestamp, use tz_localize to localize"
            )
        else:
            # Same UTC timestamp, different time zone
            tz = maybe_get_tz(tz)
            out = type(self)._from_value_and_reso(self._value, reso=self._creso, tz=tz)
            return out

    astimezone = tz_convert

    def replace(
        self,
        year=None,
        month=None,
        day=None,
        hour=None,
        minute=None,
        second=None,
        microsecond=None,
        nanosecond=None,
        tzinfo=object,
        fold=None,
    ):
        """
        Implements datetime.replace, handles nanoseconds.

        Parameters
        ----------
        year : int, optional
        month : int, optional
        day : int, optional
        hour : int, optional
        minute : int, optional
        second : int, optional
        microsecond : int, optional
        nanosecond : int, optional
        tzinfo : tz-convertible, optional
        fold : int, optional

        Returns
        -------
        Timestamp with fields replaced

        Examples
        --------
        Create a timestamp object:

        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651', tz='UTC')
        >>> ts
        Timestamp('2020-03-14 15:32:52.192548651+0000', tz='UTC')

        Replace year and the hour:

        >>> ts.replace(year=1999, hour=10)
        Timestamp('1999-03-14 10:32:52.192548651+0000', tz='UTC')

        Replace timezone (not a conversion):

        >>> import pytz
        >>> ts.replace(tzinfo=pytz.timezone('US/Pacific'))
        Timestamp('2020-03-14 15:32:52.192548651-0700', tz='US/Pacific')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.replace(tzinfo=pytz.timezone('US/Pacific'))
        NaT
        """

        cdef:
            npy_datetimestruct dts
            int64_t value
            object k, v
            datetime ts_input
            tzinfo_type tzobj
            _TSObject ts

        # set to naive if needed
        tzobj = self.tzinfo
        value = self._value

        # GH 37610. Preserve fold when replacing.
        if fold is None:
            fold = self.fold

        if tzobj is not None:
            value = tz_convert_from_utc_single(value, tzobj, creso=self._creso)

        # setup components
        pandas_datetime_to_datetimestruct(value, self._creso, &dts)
        dts.ps = self.nanosecond * 1000

        # replace
        def validate(k, v):
            """ validate integers """
            if not is_integer_object(v):
                raise ValueError(
                    f"value must be an integer, received {type(v)} for {k}"
                )
            return v

        if year is not None:
            dts.year = validate("year", year)
        if month is not None:
            dts.month = validate("month", month)
        if day is not None:
            dts.day = validate("day", day)
        if hour is not None:
            dts.hour = validate("hour", hour)
        if minute is not None:
            dts.min = validate("minute", minute)
        if second is not None:
            dts.sec = validate("second", second)
        if microsecond is not None:
            dts.us = validate("microsecond", microsecond)
        if nanosecond is not None:
            dts.ps = validate("nanosecond", nanosecond) * 1000
        if tzinfo is not object:
            tzobj = tzinfo

        # reconstruct & check bounds
        if tzobj is None:
            # We can avoid going through pydatetime paths, which is robust
            #  to datetimes outside of pydatetime range.
            ts = _TSObject()
            check_dts_bounds(&dts, self._creso)
            ts.value = npy_datetimestruct_to_datetime(self._creso, &dts)
            ts.dts = dts
            ts.creso = self._creso
            ts.fold = fold
            return create_timestamp_from_ts(
                ts.value, dts, tzobj, fold, reso=self._creso
            )

        elif tzobj is not None and treat_tz_as_pytz(tzobj):
            # replacing across a DST boundary may induce a new tzinfo object
            # see GH#18319
            ts_input = tzobj.localize(datetime(dts.year, dts.month, dts.day,
                                               dts.hour, dts.min, dts.sec,
                                               dts.us),
                                      is_dst=not bool(fold))
            tzobj = ts_input.tzinfo
        else:
            kwargs = {"year": dts.year, "month": dts.month, "day": dts.day,
                      "hour": dts.hour, "minute": dts.min, "second": dts.sec,
                      "microsecond": dts.us, "tzinfo": tzobj,
                      "fold": fold}
            ts_input = datetime(**kwargs)

        ts = convert_datetime_to_tsobject(
            ts_input, tzobj, nanos=dts.ps // 1000, reso=self._creso
        )
        return create_timestamp_from_ts(
            ts.value, dts, tzobj, fold, reso=self._creso
        )

    def to_julian_date(self) -> np.float64:
        """
        Convert TimeStamp to a Julian Date.

        0 Julian date is noon January 1, 4713 BC.

        Examples
        --------
        >>> ts = pd.Timestamp('2020-03-14T15:32:52')
        >>> ts.to_julian_date()
        2458923.147824074
        """
        year = self.year
        month = self.month
        day = self.day
        if month <= 2:
            year -= 1
            month += 12
        return (day +
                np.fix((153 * month - 457) / 5) +
                365 * year +
                np.floor(year / 4) -
                np.floor(year / 100) +
                np.floor(year / 400) +
                1721118.5 +
                (self.hour +
                 self.minute / 60.0 +
                 self.second / 3600.0 +
                 self.microsecond / 3600.0 / 1e+6 +
                 self.nanosecond / 3600.0 / 1e+9
                 ) / 24.0)

    def isoweekday(self):
        """
        Return the day of the week represented by the date.

        Monday == 1 ... Sunday == 7.

        Examples
        --------
        >>> ts = pd.Timestamp('2023-01-01 10:00:00')
        >>> ts
        Timestamp('2023-01-01 10:00:00')
        >>> ts.isoweekday()
        7
        """
        # same as super().isoweekday(), but that breaks because of how
        #  we have overridden year, see note in create_timestamp_from_ts

        return self.weekday() + 1

    def weekday(self):
        """
        Return the day of the week represented by the date.

        Monday == 0 ... Sunday == 6.

        Examples
        --------
        >>> ts = pd.Timestamp('2023-01-01')
        >>> ts
        Timestamp('2023-01-01  00:00:00')
        >>> ts.weekday()
        6
        """
        # same as super().weekday(), but that breaks because of how
        #  we have overridden year, see note in create_timestamp_from_ts
        return ccalendar.dayofweek(self.year, self.month, self.day)


# Aliases
Timestamp.weekofyear = Timestamp.week
Timestamp.daysinmonth = Timestamp.days_in_month


# ----------------------------------------------------------------------
# Scalar analogues to functions in vectorized.pyx


@cython.cdivision(False)
cdef int64_t normalize_i8_stamp(int64_t local_val, int64_t ppd) noexcept nogil:
    """
    Round the localized nanosecond timestamp down to the previous midnight.

    Parameters
    ----------
    local_val : int64_t
    ppd : int64_t
        Periods per day in the Timestamp's resolution.

    Returns
    -------
    int64_t
    """
    return local_val - (local_val % ppd)
