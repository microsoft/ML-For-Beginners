cimport cython
from cpython.datetime cimport (
    PyDateTime_CheckExact,
    PyDateTime_DATE_GET_HOUR,
    PyDateTime_DATE_GET_MICROSECOND,
    PyDateTime_DATE_GET_MINUTE,
    PyDateTime_DATE_GET_SECOND,
    PyDateTime_GET_DAY,
    PyDateTime_GET_MONTH,
    PyDateTime_GET_YEAR,
    import_datetime,
)
from cpython.object cimport (
    Py_EQ,
    Py_GE,
    Py_GT,
    Py_LE,
    Py_LT,
    Py_NE,
)

import_datetime()
PandasDateTime_IMPORT

import numpy as np

cimport numpy as cnp

cnp.import_array()
from numpy cimport (
    int64_t,
    ndarray,
    uint8_t,
)

from pandas._libs.tslibs.util cimport get_c_string_buf_and_size


cdef extern from "pandas/datetime/pd_datetime.h":
    int cmp_npy_datetimestruct(npy_datetimestruct *a,
                               npy_datetimestruct *b)

    # AS, FS, PS versions exist but are not imported because they are not used.
    npy_datetimestruct _NS_MIN_DTS, _NS_MAX_DTS
    npy_datetimestruct _US_MIN_DTS, _US_MAX_DTS
    npy_datetimestruct _MS_MIN_DTS, _MS_MAX_DTS
    npy_datetimestruct _S_MIN_DTS, _S_MAX_DTS
    npy_datetimestruct _M_MIN_DTS, _M_MAX_DTS

    PyArray_DatetimeMetaData get_datetime_metadata_from_dtype(cnp.PyArray_Descr *dtype)

    int parse_iso_8601_datetime(const char *str, int len, int want_exc,
                                npy_datetimestruct *out,
                                NPY_DATETIMEUNIT *out_bestunit,
                                int *out_local, int *out_tzoffset,
                                const char *format, int format_len,
                                FormatRequirement exact)

# ----------------------------------------------------------------------
# numpy object inspection

cdef npy_datetime get_datetime64_value(object obj) noexcept nogil:
    """
    returns the int64 value underlying scalar numpy datetime64 object

    Note that to interpret this as a datetime, the corresponding unit is
    also needed.  That can be found using `get_datetime64_unit`.
    """
    return (<PyDatetimeScalarObject*>obj).obval


cdef npy_timedelta get_timedelta64_value(object obj) noexcept nogil:
    """
    returns the int64 value underlying scalar numpy timedelta64 object
    """
    return (<PyTimedeltaScalarObject*>obj).obval


cdef NPY_DATETIMEUNIT get_datetime64_unit(object obj) noexcept nogil:
    """
    returns the unit part of the dtype for a numpy datetime64 object.
    """
    return <NPY_DATETIMEUNIT>(<PyDatetimeScalarObject*>obj).obmeta.base


cdef NPY_DATETIMEUNIT get_unit_from_dtype(cnp.dtype dtype):
    # NB: caller is responsible for ensuring this is *some* datetime64 or
    #  timedelta64 dtype, otherwise we can segfault
    cdef:
        cnp.PyArray_Descr* descr = <cnp.PyArray_Descr*>dtype
        PyArray_DatetimeMetaData meta
    meta = get_datetime_metadata_from_dtype(descr)
    return meta.base


def py_get_unit_from_dtype(dtype):
    # for testing get_unit_from_dtype; adds 896 bytes to the .so file.
    return get_unit_from_dtype(dtype)


def is_unitless(dtype: cnp.dtype) -> bool:
    """
    Check if a datetime64 or timedelta64 dtype has no attached unit.
    """
    if dtype.type_num not in [cnp.NPY_DATETIME, cnp.NPY_TIMEDELTA]:
        raise ValueError("is_unitless dtype must be datetime64 or timedelta64")
    cdef:
        NPY_DATETIMEUNIT unit = get_unit_from_dtype(dtype)

    return unit == NPY_DATETIMEUNIT.NPY_FR_GENERIC


# ----------------------------------------------------------------------
# Comparison


cdef bint cmp_dtstructs(
    npy_datetimestruct* left, npy_datetimestruct* right, int op
):
    cdef:
        int cmp_res

    cmp_res = cmp_npy_datetimestruct(left, right)
    if op == Py_EQ:
        return cmp_res == 0
    if op == Py_NE:
        return cmp_res != 0
    if op == Py_GT:
        return cmp_res == 1
    if op == Py_LT:
        return cmp_res == -1
    if op == Py_GE:
        return cmp_res == 1 or cmp_res == 0
    else:
        # i.e. op == Py_LE
        return cmp_res == -1 or cmp_res == 0


cdef bint cmp_scalar(int64_t lhs, int64_t rhs, int op) except -1:
    """
    cmp_scalar is a more performant version of PyObject_RichCompare
    typed for int64_t arguments.
    """
    if op == Py_EQ:
        return lhs == rhs
    elif op == Py_NE:
        return lhs != rhs
    elif op == Py_LT:
        return lhs < rhs
    elif op == Py_LE:
        return lhs <= rhs
    elif op == Py_GT:
        return lhs > rhs
    elif op == Py_GE:
        return lhs >= rhs


class OutOfBoundsDatetime(ValueError):
    """
    Raised when the datetime is outside the range that can be represented.

    Examples
    --------
    >>> pd.to_datetime("08335394550")
    Traceback (most recent call last):
    OutOfBoundsDatetime: Parsing "08335394550" to datetime overflows,
    at position 0
    """
    pass


class OutOfBoundsTimedelta(ValueError):
    """
    Raised when encountering a timedelta value that cannot be represented.

    Representation should be within a timedelta64[ns].

    Examples
    --------
    >>> pd.date_range(start="1/1/1700", freq="B", periods=100000)
    Traceback (most recent call last):
    OutOfBoundsTimedelta: Cannot cast 139999 days 00:00:00
    to unit='ns' without overflow.
    """
    # Timedelta analogue to OutOfBoundsDatetime
    pass


cdef get_implementation_bounds(
    NPY_DATETIMEUNIT reso,
    npy_datetimestruct *lower,
    npy_datetimestruct *upper,
):
    if reso == NPY_FR_ns:
        upper[0] = _NS_MAX_DTS
        lower[0] = _NS_MIN_DTS
    elif reso == NPY_FR_us:
        upper[0] = _US_MAX_DTS
        lower[0] = _US_MIN_DTS
    elif reso == NPY_FR_ms:
        upper[0] = _MS_MAX_DTS
        lower[0] = _MS_MIN_DTS
    elif reso == NPY_FR_s:
        upper[0] = _S_MAX_DTS
        lower[0] = _S_MIN_DTS
    elif reso == NPY_FR_m:
        upper[0] = _M_MAX_DTS
        lower[0] = _M_MIN_DTS
    else:
        raise NotImplementedError(reso)


cdef check_dts_bounds(npy_datetimestruct *dts, NPY_DATETIMEUNIT unit=NPY_FR_ns):
    """Raises OutOfBoundsDatetime if the given date is outside the range that
    can be represented by nanosecond-resolution 64-bit integers."""
    cdef:
        bint error = False
        npy_datetimestruct cmp_upper, cmp_lower

    get_implementation_bounds(unit, &cmp_lower, &cmp_upper)

    if cmp_npy_datetimestruct(dts, &cmp_lower) == -1:
        error = True
    elif cmp_npy_datetimestruct(dts, &cmp_upper) == 1:
        error = True

    if error:
        fmt = (f"{dts.year}-{dts.month:02d}-{dts.day:02d} "
               f"{dts.hour:02d}:{dts.min:02d}:{dts.sec:02d}")
        # TODO: "nanosecond" in the message assumes NPY_FR_ns
        raise OutOfBoundsDatetime(f"Out of bounds nanosecond timestamp: {fmt}")


# ----------------------------------------------------------------------
# Conversion


# just exposed for testing at the moment
def py_td64_to_tdstruct(int64_t td64, NPY_DATETIMEUNIT unit):
    cdef:
        pandas_timedeltastruct tds
    pandas_timedelta_to_timedeltastruct(td64, unit, &tds)
    return tds  # <- returned as a dict to python


cdef void pydatetime_to_dtstruct(datetime dt, npy_datetimestruct *dts) noexcept:
    if PyDateTime_CheckExact(dt):
        dts.year = PyDateTime_GET_YEAR(dt)
    else:
        # We use dt.year instead of PyDateTime_GET_YEAR because with Timestamp
        #  we override year such that PyDateTime_GET_YEAR is incorrect.
        dts.year = dt.year

    dts.month = PyDateTime_GET_MONTH(dt)
    dts.day = PyDateTime_GET_DAY(dt)
    dts.hour = PyDateTime_DATE_GET_HOUR(dt)
    dts.min = PyDateTime_DATE_GET_MINUTE(dt)
    dts.sec = PyDateTime_DATE_GET_SECOND(dt)
    dts.us = PyDateTime_DATE_GET_MICROSECOND(dt)
    dts.ps = dts.as = 0


cdef int64_t pydatetime_to_dt64(datetime val,
                                npy_datetimestruct *dts,
                                NPY_DATETIMEUNIT reso=NPY_FR_ns):
    """
    Note we are assuming that the datetime object is timezone-naive.
    """
    pydatetime_to_dtstruct(val, dts)
    return npy_datetimestruct_to_datetime(reso, dts)


cdef void pydate_to_dtstruct(date val, npy_datetimestruct *dts) noexcept:
    dts.year = PyDateTime_GET_YEAR(val)
    dts.month = PyDateTime_GET_MONTH(val)
    dts.day = PyDateTime_GET_DAY(val)
    dts.hour = dts.min = dts.sec = dts.us = 0
    dts.ps = dts.as = 0
    return

cdef int64_t pydate_to_dt64(
    date val, npy_datetimestruct *dts, NPY_DATETIMEUNIT reso=NPY_FR_ns
):
    pydate_to_dtstruct(val, dts)
    return npy_datetimestruct_to_datetime(reso, dts)


cdef int string_to_dts(
    str val,
    npy_datetimestruct* dts,
    NPY_DATETIMEUNIT* out_bestunit,
    int* out_local,
    int* out_tzoffset,
    bint want_exc,
    format: str | None=None,
    bint exact=True,
) except? -1:
    cdef:
        Py_ssize_t length
        const char* buf
        Py_ssize_t format_length
        const char* format_buf
        FormatRequirement format_requirement

    buf = get_c_string_buf_and_size(val, &length)
    if format is None:
        format_buf = b""
        format_length = 0
        format_requirement = INFER_FORMAT
    else:
        format_buf = get_c_string_buf_and_size(format, &format_length)
        format_requirement = <FormatRequirement>exact
    return parse_iso_8601_datetime(buf, length, want_exc,
                                   dts, out_bestunit, out_local, out_tzoffset,
                                   format_buf, format_length,
                                   format_requirement)


cpdef ndarray astype_overflowsafe(
    ndarray values,
    cnp.dtype dtype,
    bint copy=True,
    bint round_ok=True,
    bint is_coerce=False,
):
    """
    Convert an ndarray with datetime64[X] to datetime64[Y]
    or timedelta64[X] to timedelta64[Y],
    raising on overflow.
    """
    if values.descr.type_num == dtype.type_num == cnp.NPY_DATETIME:
        # i.e. dtype.kind == "M"
        dtype_name = "datetime64"
    elif values.descr.type_num == dtype.type_num == cnp.NPY_TIMEDELTA:
        # i.e. dtype.kind == "m"
        dtype_name = "timedelta64"
    else:
        raise TypeError(
            "astype_overflowsafe values.dtype and dtype must be either "
            "both-datetime64 or both-timedelta64."
        )

    cdef:
        NPY_DATETIMEUNIT from_unit = get_unit_from_dtype(values.dtype)
        NPY_DATETIMEUNIT to_unit = get_unit_from_dtype(dtype)

    if from_unit == NPY_DATETIMEUNIT.NPY_FR_GENERIC:
        raise TypeError(f"{dtype_name} values must have a unit specified")

    if to_unit == NPY_DATETIMEUNIT.NPY_FR_GENERIC:
        # without raising explicitly here, we end up with a SystemError
        # built-in function [...] returned a result with an error
        raise ValueError(
            f"{dtype_name} dtype must have a unit specified"
        )

    if from_unit == to_unit:
        # Check this before allocating result for perf, might save some memory
        if copy:
            return values.copy()
        return values

    elif from_unit > to_unit:
        if round_ok:
            # e.g. ns -> us, so there is no risk of overflow, so we can use
            #  numpy's astype safely. Note there _is_ risk of truncation.
            return values.astype(dtype)
        else:
            iresult2 = astype_round_check(values.view("i8"), from_unit, to_unit)
            return iresult2.view(dtype)

    if (<object>values).dtype.byteorder == ">":
        # GH#29684 we incorrectly get OutOfBoundsDatetime if we dont swap
        values = values.astype(values.dtype.newbyteorder("<"))

    cdef:
        ndarray i8values = values.view("i8")

        # equiv: result = np.empty((<object>values).shape, dtype="i8")
        ndarray iresult = cnp.PyArray_EMPTY(
            values.ndim, values.shape, cnp.NPY_INT64, 0
        )

        cnp.broadcast mi = cnp.PyArray_MultiIterNew2(iresult, i8values)
        Py_ssize_t i, N = values.size
        int64_t value, new_value
        npy_datetimestruct dts
        bint is_td = dtype.type_num == cnp.NPY_TIMEDELTA

    for i in range(N):
        # Analogous to: item = values[i]
        value = (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 1))[0]

        if value == NPY_DATETIME_NAT:
            new_value = NPY_DATETIME_NAT
        else:
            pandas_datetime_to_datetimestruct(value, from_unit, &dts)

            try:
                check_dts_bounds(&dts, to_unit)
            except OutOfBoundsDatetime as err:
                if is_coerce:
                    new_value = NPY_DATETIME_NAT
                elif is_td:
                    from_abbrev = np.datetime_data(values.dtype)[0]
                    np_val = np.timedelta64(value, from_abbrev)
                    msg = (
                        "Cannot convert {np_val} to {dtype} without overflow"
                        .format(np_val=str(np_val), dtype=str(dtype))
                    )
                    raise OutOfBoundsTimedelta(msg) from err
                else:
                    raise
            else:
                new_value = npy_datetimestruct_to_datetime(to_unit, &dts)

        # Analogous to: iresult[i] = new_value
        (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 0))[0] = new_value

        cnp.PyArray_MultiIter_NEXT(mi)

    return iresult.view(dtype)


# TODO: try to upstream this fix to numpy
def compare_mismatched_resolutions(ndarray left, ndarray right, op):
    """
    Overflow-safe comparison of timedelta64/datetime64 with mismatched resolutions.

    >>> left = np.array([500], dtype="M8[Y]")
    >>> right = np.array([0], dtype="M8[ns]")
    >>> left < right  # <- wrong!
    array([ True])
    """

    if left.dtype.kind != right.dtype.kind or left.dtype.kind not in "mM":
        raise ValueError("left and right must both be timedelta64 or both datetime64")

    cdef:
        int op_code = op_to_op_code(op)
        NPY_DATETIMEUNIT left_unit = get_unit_from_dtype(left.dtype)
        NPY_DATETIMEUNIT right_unit = get_unit_from_dtype(right.dtype)

        # equiv: result = np.empty((<object>left).shape, dtype="bool")
        ndarray result = cnp.PyArray_EMPTY(
            left.ndim, left.shape, cnp.NPY_BOOL, 0
        )

        ndarray lvalues = left.view("i8")
        ndarray rvalues = right.view("i8")

        cnp.broadcast mi = cnp.PyArray_MultiIterNew3(result, lvalues, rvalues)
        int64_t lval, rval
        bint res_value

        Py_ssize_t i, N = left.size
        npy_datetimestruct ldts, rdts

    for i in range(N):
        # Analogous to: lval = lvalues[i]
        lval = (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 1))[0]

        # Analogous to: rval = rvalues[i]
        rval = (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 2))[0]

        if lval == NPY_DATETIME_NAT or rval == NPY_DATETIME_NAT:
            res_value = op_code == Py_NE

        else:
            pandas_datetime_to_datetimestruct(lval, left_unit, &ldts)
            pandas_datetime_to_datetimestruct(rval, right_unit, &rdts)

            res_value = cmp_dtstructs(&ldts, &rdts, op_code)

        # Analogous to: result[i] = res_value
        (<uint8_t*>cnp.PyArray_MultiIter_DATA(mi, 0))[0] = res_value

        cnp.PyArray_MultiIter_NEXT(mi)

    return result


import operator


cdef int op_to_op_code(op):
    # TODO: should exist somewhere?
    if op is operator.eq:
        return Py_EQ
    if op is operator.ne:
        return Py_NE
    if op is operator.le:
        return Py_LE
    if op is operator.lt:
        return Py_LT
    if op is operator.ge:
        return Py_GE
    if op is operator.gt:
        return Py_GT


cdef ndarray astype_round_check(
    ndarray i8values,
    NPY_DATETIMEUNIT from_unit,
    NPY_DATETIMEUNIT to_unit
):
    # cases with from_unit > to_unit, e.g. ns->us, raise if the conversion
    #  involves truncation, e.g. 1500ns->1us
    cdef:
        Py_ssize_t i, N = i8values.size

        # equiv: iresult = np.empty((<object>i8values).shape, dtype="i8")
        ndarray iresult = cnp.PyArray_EMPTY(
            i8values.ndim, i8values.shape, cnp.NPY_INT64, 0
        )
        cnp.broadcast mi = cnp.PyArray_MultiIterNew2(iresult, i8values)

        # Note the arguments to_unit, from unit are swapped vs how they
        #  are passed when going to a higher-frequency reso.
        int64_t mult = get_conversion_factor(to_unit, from_unit)
        int64_t value, mod

    for i in range(N):
        # Analogous to: item = i8values[i]
        value = (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 1))[0]

        if value == NPY_DATETIME_NAT:
            new_value = NPY_DATETIME_NAT
        else:
            new_value, mod = divmod(value, mult)
            if mod != 0:
                # TODO: avoid runtime import
                from pandas._libs.tslibs.dtypes import npy_unit_to_abbrev
                from_abbrev = npy_unit_to_abbrev(from_unit)
                to_abbrev = npy_unit_to_abbrev(to_unit)
                raise ValueError(
                    f"Cannot losslessly cast '{value} {from_abbrev}' to {to_abbrev}"
                )

        # Analogous to: iresult[i] = new_value
        (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 0))[0] = new_value

        cnp.PyArray_MultiIter_NEXT(mi)

    return iresult


@cython.overflowcheck(True)
cdef int64_t get_conversion_factor(
    NPY_DATETIMEUNIT from_unit,
    NPY_DATETIMEUNIT to_unit
) except? -1:
    """
    Find the factor by which we need to multiply to convert from from_unit to to_unit.
    """
    if (
        from_unit == NPY_DATETIMEUNIT.NPY_FR_GENERIC
        or to_unit == NPY_DATETIMEUNIT.NPY_FR_GENERIC
    ):
        raise ValueError("unit-less resolutions are not supported")
    if from_unit > to_unit:
        raise ValueError

    if from_unit == to_unit:
        return 1

    if from_unit == NPY_DATETIMEUNIT.NPY_FR_W:
        return 7 * get_conversion_factor(NPY_DATETIMEUNIT.NPY_FR_D, to_unit)
    elif from_unit == NPY_DATETIMEUNIT.NPY_FR_D:
        return 24 * get_conversion_factor(NPY_DATETIMEUNIT.NPY_FR_h, to_unit)
    elif from_unit == NPY_DATETIMEUNIT.NPY_FR_h:
        return 60 * get_conversion_factor(NPY_DATETIMEUNIT.NPY_FR_m, to_unit)
    elif from_unit == NPY_DATETIMEUNIT.NPY_FR_m:
        return 60 * get_conversion_factor(NPY_DATETIMEUNIT.NPY_FR_s, to_unit)
    elif from_unit == NPY_DATETIMEUNIT.NPY_FR_s:
        return 1000 * get_conversion_factor(NPY_DATETIMEUNIT.NPY_FR_ms, to_unit)
    elif from_unit == NPY_DATETIMEUNIT.NPY_FR_ms:
        return 1000 * get_conversion_factor(NPY_DATETIMEUNIT.NPY_FR_us, to_unit)
    elif from_unit == NPY_DATETIMEUNIT.NPY_FR_us:
        return 1000 * get_conversion_factor(NPY_DATETIMEUNIT.NPY_FR_ns, to_unit)
    elif from_unit == NPY_DATETIMEUNIT.NPY_FR_ns:
        return 1000 * get_conversion_factor(NPY_DATETIMEUNIT.NPY_FR_ps, to_unit)
    elif from_unit == NPY_DATETIMEUNIT.NPY_FR_ps:
        return 1000 * get_conversion_factor(NPY_DATETIMEUNIT.NPY_FR_fs, to_unit)
    elif from_unit == NPY_DATETIMEUNIT.NPY_FR_fs:
        return 1000 * get_conversion_factor(NPY_DATETIMEUNIT.NPY_FR_as, to_unit)
    else:
        raise ValueError("Converting from M or Y units is not supported.")


cdef int64_t convert_reso(
    int64_t value,
    NPY_DATETIMEUNIT from_reso,
    NPY_DATETIMEUNIT to_reso,
    bint round_ok,
) except? -1:
    cdef:
        int64_t res_value, mult, div, mod

    if from_reso == to_reso:
        return value

    elif to_reso < from_reso:
        # e.g. ns -> us, no risk of overflow, but can be lossy rounding
        mult = get_conversion_factor(to_reso, from_reso)
        div, mod = divmod(value, mult)
        if mod > 0 and not round_ok:
            raise ValueError("Cannot losslessly convert units")

        # Note that when mod > 0, we follow np.timedelta64 in always
        #  rounding down.
        res_value = div

    elif (
        from_reso == NPY_FR_Y
        or from_reso == NPY_FR_M
        or to_reso == NPY_FR_Y
        or to_reso == NPY_FR_M
    ):
        # Converting by multiplying isn't _quite_ right bc the number of
        #  seconds in a month/year isn't fixed.
        res_value = _convert_reso_with_dtstruct(value, from_reso, to_reso)

    else:
        # e.g. ns -> us, risk of overflow, but no risk of lossy rounding
        mult = get_conversion_factor(from_reso, to_reso)
        with cython.overflowcheck(True):
            # Note: caller is responsible for re-raising as OutOfBoundsTimedelta
            res_value = value * mult

    return res_value


cdef int64_t _convert_reso_with_dtstruct(
    int64_t value,
    NPY_DATETIMEUNIT from_unit,
    NPY_DATETIMEUNIT to_unit,
) except? -1:
    cdef:
        npy_datetimestruct dts

    pandas_datetime_to_datetimestruct(value, from_unit, &dts)
    check_dts_bounds(&dts, to_unit)
    return npy_datetimestruct_to_datetime(to_unit, &dts)
