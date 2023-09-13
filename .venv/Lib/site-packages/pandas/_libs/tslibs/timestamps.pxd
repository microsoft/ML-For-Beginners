from cpython.datetime cimport (
    datetime,
    tzinfo,
)
from numpy cimport int64_t

from pandas._libs.tslibs.base cimport ABCTimestamp
from pandas._libs.tslibs.np_datetime cimport (
    NPY_DATETIMEUNIT,
    npy_datetimestruct,
)
from pandas._libs.tslibs.offsets cimport BaseOffset


cdef _Timestamp create_timestamp_from_ts(int64_t value,
                                         npy_datetimestruct dts,
                                         tzinfo tz,
                                         bint fold,
                                         NPY_DATETIMEUNIT reso=*)


cdef class _Timestamp(ABCTimestamp):
    cdef readonly:
        int64_t _value, nanosecond, year
        NPY_DATETIMEUNIT _creso

    cdef bint _get_start_end_field(self, str field, freq)
    cdef _get_date_name_field(self, str field, object locale)
    cdef int64_t _maybe_convert_value_to_local(self)
    cdef bint _can_compare(self, datetime other)
    cpdef to_datetime64(self)
    cpdef datetime to_pydatetime(_Timestamp self, bint warn=*)
    cdef bint _compare_outside_nanorange(_Timestamp self, datetime other,
                                         int op) except -1
    cdef bint _compare_mismatched_resos(_Timestamp self, _Timestamp other, int op)
    cdef _Timestamp _as_creso(_Timestamp self, NPY_DATETIMEUNIT creso, bint round_ok=*)
