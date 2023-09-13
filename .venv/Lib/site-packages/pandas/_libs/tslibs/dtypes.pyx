# period frequency constants corresponding to scikits timeseries
# originals
from enum import Enum

from pandas._libs.tslibs.np_datetime cimport (
    NPY_DATETIMEUNIT,
    get_conversion_factor,
    import_pandas_datetime,
)

import_pandas_datetime()


cdef class PeriodDtypeBase:
    """
    Similar to an actual dtype, this contains all of the information
    describing a PeriodDtype in an integer code.
    """
    # cdef readonly:
    #    PeriodDtypeCode _dtype_code
    #    int64_t _n

    def __cinit__(self, PeriodDtypeCode code, int64_t n):
        self._dtype_code = code
        self._n = n

    def __eq__(self, other):
        if not isinstance(other, PeriodDtypeBase):
            return False
        if not isinstance(self, PeriodDtypeBase):
            # cython semantics, this is a reversed op
            return False
        return self._dtype_code == other._dtype_code and self._n == other._n

    def __hash__(self) -> int:
        return hash((self._n, self._dtype_code))

    @property
    def _freq_group_code(self) -> int:
        # See also: libperiod.get_freq_group
        return (self._dtype_code // 1000) * 1000

    @property
    def _resolution_obj(self) -> "Resolution":
        fgc = self._freq_group_code
        freq_group = FreqGroup(fgc)
        abbrev = _reverse_period_code_map[freq_group.value].split("-")[0]
        if abbrev == "B":
            return Resolution.RESO_DAY
        attrname = _abbrev_to_attrnames[abbrev]
        return Resolution.from_attrname(attrname)

    @property
    def _freqstr(self) -> str:
        # Will be passed to to_offset in Period._maybe_convert_freq
        out = _reverse_period_code_map.get(self._dtype_code)
        if self._n == 1:
            return out
        return str(self._n) + out

    cpdef int _get_to_timestamp_base(self):
        """
        Return frequency code group used for base of to_timestamp against
        frequency code.

        Return day freq code against longer freq than day.
        Return second freq code against hour between second.

        Returns
        -------
        int
        """
        base = <c_FreqGroup>self._dtype_code
        if base < FR_BUS:
            return FR_DAY
        elif FR_HR <= base <= FR_SEC:
            return FR_SEC
        return base

    cpdef bint _is_tick_like(self):
        return self._dtype_code >= PeriodDtypeCode.D

    @property
    def _creso(self) -> int:
        return {
            PeriodDtypeCode.D: NPY_DATETIMEUNIT.NPY_FR_D,
            PeriodDtypeCode.H: NPY_DATETIMEUNIT.NPY_FR_h,
            PeriodDtypeCode.T: NPY_DATETIMEUNIT.NPY_FR_m,
            PeriodDtypeCode.S: NPY_DATETIMEUNIT.NPY_FR_s,
            PeriodDtypeCode.L: NPY_DATETIMEUNIT.NPY_FR_ms,
            PeriodDtypeCode.U: NPY_DATETIMEUNIT.NPY_FR_us,
            PeriodDtypeCode.N: NPY_DATETIMEUNIT.NPY_FR_ns,
        }[self._dtype_code]

    @property
    def _td64_unit(self) -> str:
        return npy_unit_to_abbrev(self._creso)


_period_code_map = {
    # Annual freqs with various fiscal year ends.
    # eg, 2005 for A-FEB runs Mar 1, 2004 to Feb 28, 2005
    "A-DEC": PeriodDtypeCode.A_DEC,  # Annual - December year end
    "A-JAN": PeriodDtypeCode.A_JAN,  # Annual - January year end
    "A-FEB": PeriodDtypeCode.A_FEB,  # Annual - February year end
    "A-MAR": PeriodDtypeCode.A_MAR,  # Annual - March year end
    "A-APR": PeriodDtypeCode.A_APR,  # Annual - April year end
    "A-MAY": PeriodDtypeCode.A_MAY,  # Annual - May year end
    "A-JUN": PeriodDtypeCode.A_JUN,  # Annual - June year end
    "A-JUL": PeriodDtypeCode.A_JUL,  # Annual - July year end
    "A-AUG": PeriodDtypeCode.A_AUG,  # Annual - August year end
    "A-SEP": PeriodDtypeCode.A_SEP,  # Annual - September year end
    "A-OCT": PeriodDtypeCode.A_OCT,  # Annual - October year end
    "A-NOV": PeriodDtypeCode.A_NOV,  # Annual - November year end

    # Quarterly frequencies with various fiscal year ends.
    # eg, Q42005 for Q-OCT runs Aug 1, 2005 to Oct 31, 2005
    "Q-DEC": PeriodDtypeCode.Q_DEC,    # Quarterly - December year end
    "Q-JAN": PeriodDtypeCode.Q_JAN,    # Quarterly - January year end
    "Q-FEB": PeriodDtypeCode.Q_FEB,    # Quarterly - February year end
    "Q-MAR": PeriodDtypeCode.Q_MAR,    # Quarterly - March year end
    "Q-APR": PeriodDtypeCode.Q_APR,    # Quarterly - April year end
    "Q-MAY": PeriodDtypeCode.Q_MAY,    # Quarterly - May year end
    "Q-JUN": PeriodDtypeCode.Q_JUN,    # Quarterly - June year end
    "Q-JUL": PeriodDtypeCode.Q_JUL,    # Quarterly - July year end
    "Q-AUG": PeriodDtypeCode.Q_AUG,    # Quarterly - August year end
    "Q-SEP": PeriodDtypeCode.Q_SEP,    # Quarterly - September year end
    "Q-OCT": PeriodDtypeCode.Q_OCT,    # Quarterly - October year end
    "Q-NOV": PeriodDtypeCode.Q_NOV,    # Quarterly - November year end

    "M": PeriodDtypeCode.M,        # Monthly

    "W-SUN": PeriodDtypeCode.W_SUN,    # Weekly - Sunday end of week
    "W-MON": PeriodDtypeCode.W_MON,    # Weekly - Monday end of week
    "W-TUE": PeriodDtypeCode.W_TUE,    # Weekly - Tuesday end of week
    "W-WED": PeriodDtypeCode.W_WED,    # Weekly - Wednesday end of week
    "W-THU": PeriodDtypeCode.W_THU,    # Weekly - Thursday end of week
    "W-FRI": PeriodDtypeCode.W_FRI,    # Weekly - Friday end of week
    "W-SAT": PeriodDtypeCode.W_SAT,    # Weekly - Saturday end of week

    "B": PeriodDtypeCode.B,        # Business days
    "D": PeriodDtypeCode.D,        # Daily
    "H": PeriodDtypeCode.H,        # Hourly
    "T": PeriodDtypeCode.T,        # Minutely
    "S": PeriodDtypeCode.S,        # Secondly
    "L": PeriodDtypeCode.L,       # Millisecondly
    "U": PeriodDtypeCode.U,       # Microsecondly
    "N": PeriodDtypeCode.N,       # Nanosecondly
}

_reverse_period_code_map = {
    _period_code_map[key]: key for key in _period_code_map}

# Yearly aliases; careful not to put these in _reverse_period_code_map
_period_code_map.update({"Y" + key[1:]: _period_code_map[key]
                         for key in _period_code_map
                         if key.startswith("A-")})

_period_code_map.update({
    "Q": 2000,   # Quarterly - December year end (default quarterly)
    "A": PeriodDtypeCode.A,   # Annual
    "W": 4000,   # Weekly
    "C": 5000,   # Custom Business Day
})

cdef set _month_names = {
    x.split("-")[-1] for x in _period_code_map.keys() if x.startswith("A-")
}

# Map attribute-name resolutions to resolution abbreviations
_attrname_to_abbrevs = {
    "year": "A",
    "quarter": "Q",
    "month": "M",
    "day": "D",
    "hour": "H",
    "minute": "T",
    "second": "S",
    "millisecond": "L",
    "microsecond": "U",
    "nanosecond": "N",
}
cdef dict attrname_to_abbrevs = _attrname_to_abbrevs
cdef dict _abbrev_to_attrnames = {v: k for k, v in attrname_to_abbrevs.items()}


class FreqGroup(Enum):
    # Mirrors c_FreqGroup in the .pxd file
    FR_ANN = c_FreqGroup.FR_ANN
    FR_QTR = c_FreqGroup.FR_QTR
    FR_MTH = c_FreqGroup.FR_MTH
    FR_WK = c_FreqGroup.FR_WK
    FR_BUS = c_FreqGroup.FR_BUS
    FR_DAY = c_FreqGroup.FR_DAY
    FR_HR = c_FreqGroup.FR_HR
    FR_MIN = c_FreqGroup.FR_MIN
    FR_SEC = c_FreqGroup.FR_SEC
    FR_MS = c_FreqGroup.FR_MS
    FR_US = c_FreqGroup.FR_US
    FR_NS = c_FreqGroup.FR_NS
    FR_UND = c_FreqGroup.FR_UND  # undefined

    @staticmethod
    def from_period_dtype_code(code: int) -> "FreqGroup":
        # See also: PeriodDtypeBase._freq_group_code
        code = (code // 1000) * 1000
        return FreqGroup(code)


class Resolution(Enum):
    RESO_NS = c_Resolution.RESO_NS
    RESO_US = c_Resolution.RESO_US
    RESO_MS = c_Resolution.RESO_MS
    RESO_SEC = c_Resolution.RESO_SEC
    RESO_MIN = c_Resolution.RESO_MIN
    RESO_HR = c_Resolution.RESO_HR
    RESO_DAY = c_Resolution.RESO_DAY
    RESO_MTH = c_Resolution.RESO_MTH
    RESO_QTR = c_Resolution.RESO_QTR
    RESO_YR = c_Resolution.RESO_YR

    def __lt__(self, other):
        return self.value < other.value

    def __ge__(self, other):
        return self.value >= other.value

    @property
    def attr_abbrev(self) -> str:
        # string that we can pass to to_offset
        return _attrname_to_abbrevs[self.attrname]

    @property
    def attrname(self) -> str:
        """
        Return datetime attribute name corresponding to this Resolution.

        Examples
        --------
        >>> Resolution.RESO_SEC.attrname
        'second'
        """
        return _reso_str_map[self.value]

    @classmethod
    def from_attrname(cls, attrname: str) -> "Resolution":
        """
        Return resolution str against resolution code.

        Examples
        --------
        >>> Resolution.from_attrname('second')
        <Resolution.RESO_SEC: 3>

        >>> Resolution.from_attrname('second') == Resolution.RESO_SEC
        True
        """
        return cls(_str_reso_map[attrname])

    @classmethod
    def get_reso_from_freqstr(cls, freq: str) -> "Resolution":
        """
        Return resolution code against frequency str.

        `freq` is given by the `offset.freqstr` for some DateOffset object.

        Examples
        --------
        >>> Resolution.get_reso_from_freqstr('H')
        <Resolution.RESO_HR: 5>

        >>> Resolution.get_reso_from_freqstr('H') == Resolution.RESO_HR
        True
        """
        try:
            attr_name = _abbrev_to_attrnames[freq]
        except KeyError:
            # For quarterly and yearly resolutions, we need to chop off
            #  a month string.
            split_freq = freq.split("-")
            if len(split_freq) != 2:
                raise
            if split_freq[1] not in _month_names:
                # i.e. we want e.g. "Q-DEC", not "Q-INVALID"
                raise
            attr_name = _abbrev_to_attrnames[split_freq[0]]

        return cls.from_attrname(attr_name)


class NpyDatetimeUnit(Enum):
    """
    Python-space analogue to NPY_DATETIMEUNIT.
    """
    NPY_FR_Y = NPY_DATETIMEUNIT.NPY_FR_Y
    NPY_FR_M = NPY_DATETIMEUNIT.NPY_FR_M
    NPY_FR_W = NPY_DATETIMEUNIT.NPY_FR_W
    NPY_FR_D = NPY_DATETIMEUNIT.NPY_FR_D
    NPY_FR_h = NPY_DATETIMEUNIT.NPY_FR_h
    NPY_FR_m = NPY_DATETIMEUNIT.NPY_FR_m
    NPY_FR_s = NPY_DATETIMEUNIT.NPY_FR_s
    NPY_FR_ms = NPY_DATETIMEUNIT.NPY_FR_ms
    NPY_FR_us = NPY_DATETIMEUNIT.NPY_FR_us
    NPY_FR_ns = NPY_DATETIMEUNIT.NPY_FR_ns
    NPY_FR_ps = NPY_DATETIMEUNIT.NPY_FR_ps
    NPY_FR_fs = NPY_DATETIMEUNIT.NPY_FR_fs
    NPY_FR_as = NPY_DATETIMEUNIT.NPY_FR_as
    NPY_FR_GENERIC = NPY_DATETIMEUNIT.NPY_FR_GENERIC


cpdef NPY_DATETIMEUNIT get_supported_reso(NPY_DATETIMEUNIT reso):
    # If we have an unsupported reso, return the nearest supported reso.
    if reso == NPY_DATETIMEUNIT.NPY_FR_GENERIC:
        # TODO: or raise ValueError? trying this gives unraisable errors, but
        #  "except? -1" breaks at compile-time for unknown reasons
        return NPY_DATETIMEUNIT.NPY_FR_ns
    if reso < NPY_DATETIMEUNIT.NPY_FR_s:
        return NPY_DATETIMEUNIT.NPY_FR_s
    elif reso > NPY_DATETIMEUNIT.NPY_FR_ns:
        return NPY_DATETIMEUNIT.NPY_FR_ns
    return reso


cpdef bint is_supported_unit(NPY_DATETIMEUNIT reso):
    return (
        reso == NPY_DATETIMEUNIT.NPY_FR_ns
        or reso == NPY_DATETIMEUNIT.NPY_FR_us
        or reso == NPY_DATETIMEUNIT.NPY_FR_ms
        or reso == NPY_DATETIMEUNIT.NPY_FR_s
    )


cpdef str npy_unit_to_abbrev(NPY_DATETIMEUNIT unit):
    if unit == NPY_DATETIMEUNIT.NPY_FR_ns or unit == NPY_DATETIMEUNIT.NPY_FR_GENERIC:
        # generic -> default to nanoseconds
        return "ns"
    elif unit == NPY_DATETIMEUNIT.NPY_FR_us:
        return "us"
    elif unit == NPY_DATETIMEUNIT.NPY_FR_ms:
        return "ms"
    elif unit == NPY_DATETIMEUNIT.NPY_FR_s:
        return "s"
    elif unit == NPY_DATETIMEUNIT.NPY_FR_m:
        return "m"
    elif unit == NPY_DATETIMEUNIT.NPY_FR_h:
        return "h"
    elif unit == NPY_DATETIMEUNIT.NPY_FR_D:
        return "D"
    elif unit == NPY_DATETIMEUNIT.NPY_FR_W:
        return "W"
    elif unit == NPY_DATETIMEUNIT.NPY_FR_M:
        return "M"
    elif unit == NPY_DATETIMEUNIT.NPY_FR_Y:
        return "Y"

    # Checks for not-really-supported units go at the end, as we don't expect
    #  to see these often
    elif unit == NPY_DATETIMEUNIT.NPY_FR_ps:
        return "ps"
    elif unit == NPY_DATETIMEUNIT.NPY_FR_fs:
        return "fs"
    elif unit == NPY_DATETIMEUNIT.NPY_FR_as:
        return "as"

    else:
        raise NotImplementedError(unit)


cpdef NPY_DATETIMEUNIT abbrev_to_npy_unit(str abbrev):
    if abbrev == "Y":
        return NPY_DATETIMEUNIT.NPY_FR_Y
    elif abbrev == "M":
        return NPY_DATETIMEUNIT.NPY_FR_M
    elif abbrev == "W":
        return NPY_DATETIMEUNIT.NPY_FR_W
    elif abbrev == "D" or abbrev == "d":
        return NPY_DATETIMEUNIT.NPY_FR_D
    elif abbrev == "h":
        return NPY_DATETIMEUNIT.NPY_FR_h
    elif abbrev == "m":
        return NPY_DATETIMEUNIT.NPY_FR_m
    elif abbrev == "s":
        return NPY_DATETIMEUNIT.NPY_FR_s
    elif abbrev == "ms":
        return NPY_DATETIMEUNIT.NPY_FR_ms
    elif abbrev == "us":
        return NPY_DATETIMEUNIT.NPY_FR_us
    elif abbrev == "ns":
        return NPY_DATETIMEUNIT.NPY_FR_ns
    elif abbrev == "ps":
        return NPY_DATETIMEUNIT.NPY_FR_ps
    elif abbrev == "fs":
        return NPY_DATETIMEUNIT.NPY_FR_fs
    elif abbrev == "as":
        return NPY_DATETIMEUNIT.NPY_FR_as
    elif abbrev is None:
        return NPY_DATETIMEUNIT.NPY_FR_GENERIC
    else:
        raise ValueError(f"Unrecognized unit {abbrev}")


cdef NPY_DATETIMEUNIT freq_group_code_to_npy_unit(int freq) noexcept nogil:
    """
    Convert the freq to the corresponding NPY_DATETIMEUNIT to pass
    to npy_datetimestruct_to_datetime.
    """
    if freq == FR_MTH:
        return NPY_DATETIMEUNIT.NPY_FR_M
    elif freq == FR_DAY:
        return NPY_DATETIMEUNIT.NPY_FR_D
    elif freq == FR_HR:
        return NPY_DATETIMEUNIT.NPY_FR_h
    elif freq == FR_MIN:
        return NPY_DATETIMEUNIT.NPY_FR_m
    elif freq == FR_SEC:
        return NPY_DATETIMEUNIT.NPY_FR_s
    elif freq == FR_MS:
        return NPY_DATETIMEUNIT.NPY_FR_ms
    elif freq == FR_US:
        return NPY_DATETIMEUNIT.NPY_FR_us
    elif freq == FR_NS:
        return NPY_DATETIMEUNIT.NPY_FR_ns
    elif freq == FR_UND:
        # Default to Day
        return NPY_DATETIMEUNIT.NPY_FR_D


# TODO: use in _matplotlib.converter?
cpdef int64_t periods_per_day(
    NPY_DATETIMEUNIT reso=NPY_DATETIMEUNIT.NPY_FR_ns
) except? -1:
    """
    How many of the given time units fit into a single day?
    """
    return get_conversion_factor(NPY_DATETIMEUNIT.NPY_FR_D, reso)


cpdef int64_t periods_per_second(NPY_DATETIMEUNIT reso) except? -1:
    return get_conversion_factor(NPY_DATETIMEUNIT.NPY_FR_s, reso)


cdef dict _reso_str_map = {
    Resolution.RESO_NS.value: "nanosecond",
    Resolution.RESO_US.value: "microsecond",
    Resolution.RESO_MS.value: "millisecond",
    Resolution.RESO_SEC.value: "second",
    Resolution.RESO_MIN.value: "minute",
    Resolution.RESO_HR.value: "hour",
    Resolution.RESO_DAY.value: "day",
    Resolution.RESO_MTH.value: "month",
    Resolution.RESO_QTR.value: "quarter",
    Resolution.RESO_YR.value: "year",
}

cdef dict _str_reso_map = {v: k for k, v in _reso_str_map.items()}

cdef dict npy_unit_to_attrname = {
    NPY_DATETIMEUNIT.NPY_FR_Y: "year",
    NPY_DATETIMEUNIT.NPY_FR_M: "month",
    NPY_DATETIMEUNIT.NPY_FR_D: "day",
    NPY_DATETIMEUNIT.NPY_FR_h: "hour",
    NPY_DATETIMEUNIT.NPY_FR_m: "minute",
    NPY_DATETIMEUNIT.NPY_FR_s: "second",
    NPY_DATETIMEUNIT.NPY_FR_ms: "millisecond",
    NPY_DATETIMEUNIT.NPY_FR_us: "microsecond",
    NPY_DATETIMEUNIT.NPY_FR_ns: "nanosecond",
}
cdef dict attrname_to_npy_unit = {v: k for k, v in npy_unit_to_attrname.items()}
