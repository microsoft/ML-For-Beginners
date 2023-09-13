from cpython.datetime cimport tzinfo
from numpy cimport (
    int64_t,
    intp_t,
    ndarray,
)

from pandas._libs.tslibs.np_datetime cimport NPY_DATETIMEUNIT


cpdef int64_t tz_convert_from_utc_single(
    int64_t utc_val, tzinfo tz, NPY_DATETIMEUNIT creso=*
) except? -1
cdef int64_t tz_localize_to_utc_single(
    int64_t val,
    tzinfo tz,
    object ambiguous=*,
    object nonexistent=*,
    NPY_DATETIMEUNIT creso=*,
) except? -1


cdef class Localizer:
    cdef:
        tzinfo tz
        NPY_DATETIMEUNIT _creso
        bint use_utc, use_fixed, use_tzlocal, use_dst, use_pytz
        ndarray trans
        Py_ssize_t ntrans
        const int64_t[::1] deltas
        int64_t delta
        int64_t* tdata

    cdef int64_t utc_val_to_local_val(
        self,
        int64_t utc_val,
        Py_ssize_t* pos,
        bint* fold=?,
    ) except? -1
