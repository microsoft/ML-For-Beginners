cimport numpy as cnp
from cpython.datetime cimport (
    date,
    datetime,
)
from numpy cimport (
    int32_t,
    int64_t,
)


# TODO(cython3): most of these can be cimported directly from numpy
cdef extern from "numpy/ndarrayobject.h":
    ctypedef int64_t npy_timedelta
    ctypedef int64_t npy_datetime

cdef extern from "numpy/ndarraytypes.h":
    ctypedef struct PyArray_DatetimeMetaData:
        NPY_DATETIMEUNIT base
        int64_t num

cdef extern from "numpy/arrayscalars.h":
    ctypedef struct PyDatetimeScalarObject:
        # PyObject_HEAD
        npy_datetime obval
        PyArray_DatetimeMetaData obmeta

    ctypedef struct PyTimedeltaScalarObject:
        # PyObject_HEAD
        npy_timedelta obval
        PyArray_DatetimeMetaData obmeta

cdef extern from "numpy/ndarraytypes.h":
    ctypedef struct npy_datetimestruct:
        int64_t year
        int32_t month, day, hour, min, sec, us, ps, as

    ctypedef enum NPY_DATETIMEUNIT:
        NPY_FR_Y
        NPY_FR_M
        NPY_FR_W
        NPY_FR_D
        NPY_FR_B
        NPY_FR_h
        NPY_FR_m
        NPY_FR_s
        NPY_FR_ms
        NPY_FR_us
        NPY_FR_ns
        NPY_FR_ps
        NPY_FR_fs
        NPY_FR_as
        NPY_FR_GENERIC

    int64_t NPY_DATETIME_NAT  # elswhere we call this NPY_NAT


cdef extern from "pandas/datetime/pd_datetime.h":
    ctypedef struct pandas_timedeltastruct:
        int64_t days
        int32_t hrs, min, sec, ms, us, ns, seconds, microseconds, nanoseconds

    void pandas_datetime_to_datetimestruct(npy_datetime val,
                                           NPY_DATETIMEUNIT fr,
                                           npy_datetimestruct *result) nogil

    npy_datetime npy_datetimestruct_to_datetime(NPY_DATETIMEUNIT fr,
                                                npy_datetimestruct *d) nogil

    void pandas_timedelta_to_timedeltastruct(npy_timedelta val,
                                             NPY_DATETIMEUNIT fr,
                                             pandas_timedeltastruct *result
                                             ) nogil

    void PandasDateTime_IMPORT()

    ctypedef enum FormatRequirement:
        PARTIAL_MATCH
        EXACT_MATCH
        INFER_FORMAT

# You must call this before using the PandasDateTime CAPI functions
cdef inline void import_pandas_datetime() noexcept:
    PandasDateTime_IMPORT

cdef bint cmp_scalar(int64_t lhs, int64_t rhs, int op) except -1

cdef check_dts_bounds(npy_datetimestruct *dts, NPY_DATETIMEUNIT unit=?)

cdef int64_t pydatetime_to_dt64(
    datetime val, npy_datetimestruct *dts, NPY_DATETIMEUNIT reso=?
)
cdef void pydatetime_to_dtstruct(datetime dt, npy_datetimestruct *dts) noexcept
cdef int64_t pydate_to_dt64(
    date val, npy_datetimestruct *dts, NPY_DATETIMEUNIT reso=?
)
cdef void pydate_to_dtstruct(date val, npy_datetimestruct *dts) noexcept

cdef npy_datetime get_datetime64_value(object obj) noexcept nogil
cdef npy_timedelta get_timedelta64_value(object obj) noexcept nogil
cdef NPY_DATETIMEUNIT get_datetime64_unit(object obj) noexcept nogil

cdef int string_to_dts(
    str val,
    npy_datetimestruct* dts,
    NPY_DATETIMEUNIT* out_bestunit,
    int* out_local,
    int* out_tzoffset,
    bint want_exc,
    format: str | None = *,
    bint exact = *
) except? -1

cdef NPY_DATETIMEUNIT get_unit_from_dtype(cnp.dtype dtype)

cpdef cnp.ndarray astype_overflowsafe(
    cnp.ndarray values,  # ndarray[datetime64[anyunit]]
    cnp.dtype dtype,  # ndarray[datetime64[anyunit]]
    bint copy=*,
    bint round_ok=*,
    bint is_coerce=*,
)
cdef int64_t get_conversion_factor(
    NPY_DATETIMEUNIT from_unit,
    NPY_DATETIMEUNIT to_unit,
) except? -1

cdef bint cmp_dtstructs(npy_datetimestruct* left, npy_datetimestruct* right, int op)
cdef get_implementation_bounds(
    NPY_DATETIMEUNIT reso, npy_datetimestruct *lower, npy_datetimestruct *upper
)

cdef int64_t convert_reso(
    int64_t value,
    NPY_DATETIMEUNIT from_reso,
    NPY_DATETIMEUNIT to_reso,
    bint round_ok,
) except? -1
