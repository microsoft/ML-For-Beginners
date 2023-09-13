from cpython.datetime cimport timedelta
from numpy cimport int64_t

from .np_datetime cimport NPY_DATETIMEUNIT


# Exposed for tslib, not intended for outside use.
cpdef int64_t delta_to_nanoseconds(
    delta, NPY_DATETIMEUNIT reso=*, bint round_ok=*
) except? -1
cdef convert_to_timedelta64(object ts, str unit)
cdef bint is_any_td_scalar(object obj)


cdef class _Timedelta(timedelta):
    cdef readonly:
        int64_t _value      # nanoseconds
        bint _is_populated  # are my components populated
        int64_t _d, _h, _m, _s, _ms, _us, _ns
        NPY_DATETIMEUNIT _creso

    cpdef timedelta to_pytimedelta(_Timedelta self)
    cdef bint _has_ns(self)
    cdef bint _is_in_pytimedelta_bounds(self)
    cdef _ensure_components(_Timedelta self)
    cdef bint _compare_mismatched_resos(self, _Timedelta other, op)
    cdef _Timedelta _as_creso(self, NPY_DATETIMEUNIT reso, bint round_ok=*)
    cpdef _maybe_cast_to_matching_resos(self, _Timedelta other)
