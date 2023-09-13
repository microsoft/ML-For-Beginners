"""
timezone conversion
"""
cimport cython
from cpython.datetime cimport (
    PyDelta_Check,
    datetime,
    datetime_new,
    import_datetime,
    timedelta,
    tzinfo,
)
from cython cimport Py_ssize_t

import_datetime()

import numpy as np
import pytz

cimport numpy as cnp
from numpy cimport (
    int64_t,
    intp_t,
    ndarray,
    uint8_t,
)

cnp.import_array()

from pandas._libs.tslibs.dtypes cimport (
    periods_per_day,
    periods_per_second,
)
from pandas._libs.tslibs.nattype cimport NPY_NAT
from pandas._libs.tslibs.np_datetime cimport (
    NPY_DATETIMEUNIT,
    import_pandas_datetime,
    npy_datetimestruct,
    pandas_datetime_to_datetimestruct,
    pydatetime_to_dt64,
)

import_pandas_datetime()

from pandas._libs.tslibs.timezones cimport (
    get_dst_info,
    is_fixed_offset,
    is_tzlocal,
    is_utc,
    is_zoneinfo,
    utc_stdlib,
)


cdef const int64_t[::1] _deltas_placeholder = np.array([], dtype=np.int64)


@cython.freelist(16)
@cython.final
cdef class Localizer:
    # cdef:
    #    tzinfo tz
    #    NPY_DATETIMEUNIT _creso
    #    bint use_utc, use_fixed, use_tzlocal, use_dst, use_pytz
    #    ndarray trans
    #    Py_ssize_t ntrans
    #    const int64_t[::1] deltas
    #    int64_t delta
    #    int64_t* tdata

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    def __cinit__(self, tzinfo tz, NPY_DATETIMEUNIT creso):
        self.tz = tz
        self._creso = creso
        self.use_utc = self.use_tzlocal = self.use_fixed = False
        self.use_dst = self.use_pytz = False
        self.ntrans = -1  # placeholder
        self.delta = -1  # placeholder
        self.deltas = _deltas_placeholder
        self.tdata = NULL

        if is_utc(tz) or tz is None:
            self.use_utc = True

        elif is_tzlocal(tz) or is_zoneinfo(tz):
            self.use_tzlocal = True

        else:
            trans, deltas, typ = get_dst_info(tz)
            if creso != NPY_DATETIMEUNIT.NPY_FR_ns:
                # NB: using floordiv here is implicitly assuming we will
                #  never see trans or deltas that are not an integer number
                #  of seconds.
                # TODO: avoid these np.array calls
                if creso == NPY_DATETIMEUNIT.NPY_FR_us:
                    trans = np.array(trans) // 1_000
                    deltas = np.array(deltas) // 1_000
                elif creso == NPY_DATETIMEUNIT.NPY_FR_ms:
                    trans = np.array(trans) // 1_000_000
                    deltas = np.array(deltas) // 1_000_000
                elif creso == NPY_DATETIMEUNIT.NPY_FR_s:
                    trans = np.array(trans) // 1_000_000_000
                    deltas = np.array(deltas) // 1_000_000_000
                else:
                    raise NotImplementedError(creso)

            self.trans = trans
            self.ntrans = self.trans.shape[0]
            self.deltas = deltas

            if typ != "pytz" and typ != "dateutil":
                # static/fixed; in this case we know that len(delta) == 1
                self.use_fixed = True
                self.delta = deltas[0]
            else:
                self.use_dst = True
                if typ == "pytz":
                    self.use_pytz = True
                self.tdata = <int64_t*>cnp.PyArray_DATA(trans)

    @cython.boundscheck(False)
    cdef int64_t utc_val_to_local_val(
        self, int64_t utc_val, Py_ssize_t* pos, bint* fold=NULL
    ) except? -1:
        if self.use_utc:
            return utc_val
        elif self.use_tzlocal:
            return utc_val + _tz_localize_using_tzinfo_api(
                utc_val, self.tz, to_utc=False, creso=self._creso, fold=fold
            )
        elif self.use_fixed:
            return utc_val + self.delta
        else:
            pos[0] = bisect_right_i8(self.tdata, utc_val, self.ntrans) - 1
            if fold is not NULL:
                fold[0] = _infer_dateutil_fold(
                    utc_val, self.trans, self.deltas, pos[0]
                )

            return utc_val + self.deltas[pos[0]]


cdef int64_t tz_localize_to_utc_single(
    int64_t val,
    tzinfo tz,
    object ambiguous=None,
    object nonexistent=None,
    NPY_DATETIMEUNIT creso=NPY_DATETIMEUNIT.NPY_FR_ns,
) except? -1:
    """See tz_localize_to_utc.__doc__"""
    cdef:
        int64_t delta
        int64_t[::1] deltas

    if val == NPY_NAT:
        return val

    elif is_utc(tz) or tz is None:
        return val

    elif is_tzlocal(tz):
        return val - _tz_localize_using_tzinfo_api(val, tz, to_utc=True, creso=creso)

    elif is_fixed_offset(tz):
        _, deltas, _ = get_dst_info(tz)
        delta = deltas[0]
        # TODO: de-duplicate with Localizer.__init__
        if creso != NPY_DATETIMEUNIT.NPY_FR_ns:
            if creso == NPY_DATETIMEUNIT.NPY_FR_us:
                delta = delta // 1000
            elif creso == NPY_DATETIMEUNIT.NPY_FR_ms:
                delta = delta // 1_000_000
            elif creso == NPY_DATETIMEUNIT.NPY_FR_s:
                delta = delta // 1_000_000_000

        return val - delta

    else:
        return tz_localize_to_utc(
            np.array([val], dtype="i8"),
            tz,
            ambiguous=ambiguous,
            nonexistent=nonexistent,
            creso=creso,
        )[0]


@cython.boundscheck(False)
@cython.wraparound(False)
def tz_localize_to_utc(
    ndarray[int64_t] vals,
    tzinfo tz,
    object ambiguous=None,
    object nonexistent=None,
    NPY_DATETIMEUNIT creso=NPY_DATETIMEUNIT.NPY_FR_ns,
):
    """
    Localize tzinfo-naive i8 to given time zone (using pytz). If
    there are ambiguities in the values, raise AmbiguousTimeError.

    Parameters
    ----------
    vals : ndarray[int64_t]
    tz : tzinfo or None
    ambiguous : str, bool, or arraylike
        When clocks moved backward due to DST, ambiguous times may arise.
        For example in Central European Time (UTC+01), when going from 03:00
        DST to 02:00 non-DST, 02:30:00 local time occurs both at 00:30:00 UTC
        and at 01:30:00 UTC. In such a situation, the `ambiguous` parameter
        dictates how ambiguous times should be handled.

        - 'infer' will attempt to infer fall dst-transition hours based on
          order
        - bool-ndarray where True signifies a DST time, False signifies a
          non-DST time (note that this flag is only applicable for ambiguous
          times, but the array must have the same length as vals)
        - bool if True, treat all vals as DST. If False, treat them as non-DST
        - 'NaT' will return NaT where there are ambiguous times

    nonexistent : {None, "NaT", "shift_forward", "shift_backward", "raise", \
timedelta-like}
        How to handle non-existent times when converting wall times to UTC
    creso : NPY_DATETIMEUNIT, default NPY_FR_ns

    Returns
    -------
    localized : ndarray[int64_t]
    """

    if tz is None or is_utc(tz) or vals.size == 0:
        # Fastpath, avoid overhead of creating Localizer
        return vals.copy()

    cdef:
        ndarray[uint8_t, cast=True] ambiguous_array
        Py_ssize_t i, n = vals.shape[0]
        Py_ssize_t delta_idx_offset, delta_idx
        int64_t v, left, right, val, new_local, remaining_mins
        int64_t first_delta, delta
        int64_t shift_delta = 0
        ndarray[int64_t] result_a, result_b, dst_hours
        int64_t[::1] result
        bint is_zi = False
        bint infer_dst = False, is_dst = False, fill = False
        bint shift_forward = False, shift_backward = False
        bint fill_nonexist = False
        str stamp
        Localizer info = Localizer(tz, creso=creso)
        int64_t pph = periods_per_day(creso) // 24
        int64_t pps = periods_per_second(creso)
        npy_datetimestruct dts

    # Vectorized version of DstTzInfo.localize

    # silence false-positive compiler warning
    ambiguous_array = np.empty(0, dtype=bool)
    if isinstance(ambiguous, str):
        if ambiguous == "infer":
            infer_dst = True
        elif ambiguous == "NaT":
            fill = True
    elif isinstance(ambiguous, bool):
        is_dst = True
        if ambiguous:
            ambiguous_array = np.ones(len(vals), dtype=bool)
        else:
            ambiguous_array = np.zeros(len(vals), dtype=bool)
    elif hasattr(ambiguous, "__iter__"):
        is_dst = True
        if len(ambiguous) != len(vals):
            raise ValueError("Length of ambiguous bool-array must be "
                             "the same size as vals")
        ambiguous_array = np.asarray(ambiguous, dtype=bool)

    if nonexistent == "NaT":
        fill_nonexist = True
    elif nonexistent == "shift_forward":
        shift_forward = True
    elif nonexistent == "shift_backward":
        shift_backward = True
    elif PyDelta_Check(nonexistent):
        from .timedeltas import delta_to_nanoseconds
        shift_delta = delta_to_nanoseconds(nonexistent, reso=creso)
    elif nonexistent not in ("raise", None):
        msg = ("nonexistent must be one of {'NaT', 'raise', 'shift_forward', "
               "shift_backwards} or a timedelta object")
        raise ValueError(msg)

    result = cnp.PyArray_EMPTY(vals.ndim, vals.shape, cnp.NPY_INT64, 0)

    if info.use_tzlocal and not is_zoneinfo(tz):
        for i in range(n):
            v = vals[i]
            if v == NPY_NAT:
                result[i] = NPY_NAT
            else:
                result[i] = v - _tz_localize_using_tzinfo_api(
                    v, tz, to_utc=True, creso=creso
                )
        return result.base  # to return underlying ndarray

    elif info.use_fixed:
        delta = info.delta
        for i in range(n):
            v = vals[i]
            if v == NPY_NAT:
                result[i] = NPY_NAT
            else:
                result[i] = v - delta
        return result.base  # to return underlying ndarray

    # Determine whether each date lies left of the DST transition (store in
    # result_a) or right of the DST transition (store in result_b)
    if is_zoneinfo(tz):
        is_zi = True
        result_a, result_b =_get_utc_bounds_zoneinfo(
            vals, tz, creso=creso
        )
    else:
        result_a, result_b =_get_utc_bounds(
            vals, info.tdata, info.ntrans, info.deltas, creso=creso
        )

    # silence false-positive compiler warning
    dst_hours = np.empty(0, dtype=np.int64)
    if infer_dst:
        dst_hours = _get_dst_hours(vals, result_a, result_b, creso=creso)

    # Pre-compute delta_idx_offset that will be used if we go down non-existent
    #  paths.
    # Shift the delta_idx by if the UTC offset of
    # the target tz is greater than 0 and we're moving forward
    # or vice versa
    first_delta = info.deltas[0]
    if (shift_forward or shift_delta > 0) and first_delta > 0:
        delta_idx_offset = 1
    elif (shift_backward or shift_delta < 0) and first_delta < 0:
        delta_idx_offset = 1
    else:
        delta_idx_offset = 0

    for i in range(n):
        val = vals[i]
        left = result_a[i]
        right = result_b[i]
        if val == NPY_NAT:
            # TODO: test with non-nano
            result[i] = val
        elif left != NPY_NAT and right != NPY_NAT:
            if left == right:
                # TODO: test with non-nano
                result[i] = left
            else:
                if infer_dst and dst_hours[i] != NPY_NAT:
                    # TODO: test with non-nano
                    result[i] = dst_hours[i]
                elif is_dst:
                    if ambiguous_array[i]:
                        result[i] = left
                    else:
                        result[i] = right
                elif fill:
                    # TODO: test with non-nano; parametrize test_dt_round_tz_ambiguous
                    result[i] = NPY_NAT
                else:
                    stamp = _render_tstamp(val, creso=creso)
                    raise pytz.AmbiguousTimeError(
                        f"Cannot infer dst time from {stamp}, try using the "
                        "'ambiguous' argument"
                    )
        elif left != NPY_NAT:
            result[i] = left
        elif right != NPY_NAT:
            # TODO: test with non-nano
            result[i] = right
        else:
            # Handle nonexistent times
            if shift_forward or shift_backward or shift_delta != 0:
                # Shift the nonexistent time to the closest existing time
                remaining_mins = val % pph
                if shift_delta != 0:
                    # Validate that we don't relocalize on another nonexistent
                    # time
                    if -1 < shift_delta + remaining_mins < pph:
                        raise ValueError(
                            "The provided timedelta will relocalize on a "
                            f"nonexistent time: {nonexistent}"
                        )
                    new_local = val + shift_delta
                elif shift_forward:
                    new_local = val + (pph - remaining_mins)
                else:
                    # Subtract 1 since the beginning hour is _inclusive_ of
                    # nonexistent times
                    new_local = val - remaining_mins - 1

                if is_zi:
                    # use the same construction as in _get_utc_bounds_zoneinfo
                    pandas_datetime_to_datetimestruct(new_local, creso, &dts)
                    extra = (dts.ps // 1000) * (pps // 1_000_000_000)

                    dt = datetime_new(dts.year, dts.month, dts.day, dts.hour,
                                      dts.min, dts.sec, dts.us, None)

                    if shift_forward or shift_delta > 0:
                        dt = dt.replace(tzinfo=tz, fold=1)
                    else:
                        dt = dt.replace(tzinfo=tz, fold=0)
                    dt = dt.astimezone(utc_stdlib)
                    dt = dt.replace(tzinfo=None)
                    result[i] = pydatetime_to_dt64(dt, &dts, creso) + extra

                else:
                    delta_idx = bisect_right_i8(info.tdata, new_local, info.ntrans)

                    delta_idx = delta_idx - delta_idx_offset
                    result[i] = new_local - info.deltas[delta_idx]
            elif fill_nonexist:
                result[i] = NPY_NAT
            else:
                stamp = _render_tstamp(val, creso=creso)
                raise pytz.NonExistentTimeError(stamp)

    return result.base  # .base to get underlying ndarray


cdef Py_ssize_t bisect_right_i8(int64_t *data, int64_t val, Py_ssize_t n):
    # Caller is responsible for checking n > 0
    # This looks very similar to local_search_right in the ndarray.searchsorted
    #  implementation.
    cdef:
        Py_ssize_t pivot, left = 0, right = n

    # edge cases
    if val > data[n - 1]:
        return n

    # Caller is responsible for ensuring 'val >= data[0]'. This is
    #  ensured by the fact that 'data' comes from get_dst_info where data[0]
    #  is *always* NPY_NAT+1. If that ever changes, we will need to restore
    #  the following disabled check.
    # if val < data[0]:
    #    return 0

    while left < right:
        pivot = left + (right - left) // 2

        if data[pivot] <= val:
            left = pivot + 1
        else:
            right = pivot

    return left


cdef str _render_tstamp(int64_t val, NPY_DATETIMEUNIT creso):
    """ Helper function to render exception messages"""
    from pandas._libs.tslibs.timestamps import Timestamp
    ts = Timestamp._from_value_and_reso(val, creso, None)
    return str(ts)


cdef _get_utc_bounds(
    ndarray vals,
    int64_t* tdata,
    Py_ssize_t ntrans,
    const int64_t[::1] deltas,
    NPY_DATETIMEUNIT creso,
):
    # Determine whether each date lies left of the DST transition (store in
    # result_a) or right of the DST transition (store in result_b)

    cdef:
        ndarray result_a, result_b
        Py_ssize_t i, n = vals.size
        int64_t val, v_left, v_right
        Py_ssize_t isl, isr, pos_left, pos_right
        int64_t ppd = periods_per_day(creso)

    result_a = cnp.PyArray_EMPTY(vals.ndim, vals.shape, cnp.NPY_INT64, 0)
    result_b = cnp.PyArray_EMPTY(vals.ndim, vals.shape, cnp.NPY_INT64, 0)

    for i in range(n):
        # This loops resembles the "Find the two best possibilities" block
        #  in pytz's DstTZInfo.localize method.
        result_a[i] = NPY_NAT
        result_b[i] = NPY_NAT

        val = vals[i]
        if val == NPY_NAT:
            continue

        # TODO: be careful of overflow in val-ppd
        isl = bisect_right_i8(tdata, val - ppd, ntrans) - 1
        if isl < 0:
            isl = 0

        v_left = val - deltas[isl]
        pos_left = bisect_right_i8(tdata, v_left, ntrans) - 1
        # timestamp falls to the left side of the DST transition
        if v_left + deltas[pos_left] == val:
            result_a[i] = v_left

        # TODO: be careful of overflow in val+ppd
        isr = bisect_right_i8(tdata, val + ppd, ntrans) - 1
        if isr < 0:
            isr = 0

        v_right = val - deltas[isr]
        pos_right = bisect_right_i8(tdata, v_right, ntrans) - 1
        # timestamp falls to the right side of the DST transition
        if v_right + deltas[pos_right] == val:
            result_b[i] = v_right

    return result_a, result_b


cdef _get_utc_bounds_zoneinfo(ndarray vals, tz, NPY_DATETIMEUNIT creso):
    """
    For each point in 'vals', find the UTC time that it corresponds to if
    with fold=0 and fold=1. In non-ambiguous cases, these will match.

    Parameters
    ----------
    vals : ndarray[int64_t]
    tz : ZoneInfo
    creso : NPY_DATETIMEUNIT

    Returns
    -------
    ndarray[int64_t]
    ndarray[int64_t]
    """
    cdef:
        Py_ssize_t i, n = vals.size
        npy_datetimestruct dts
        datetime dt, rt, left, right, aware, as_utc
        int64_t val, pps = periods_per_second(creso)
        ndarray result_a, result_b

    result_a = cnp.PyArray_EMPTY(vals.ndim, vals.shape, cnp.NPY_INT64, 0)
    result_b = cnp.PyArray_EMPTY(vals.ndim, vals.shape, cnp.NPY_INT64, 0)

    for i in range(n):
        val = vals[i]
        if val == NPY_NAT:
            result_a[i] = NPY_NAT
            result_b[i] = NPY_NAT
            continue

        pandas_datetime_to_datetimestruct(val, creso, &dts)
        # casting to pydatetime drops nanoseconds etc, which we will
        #  need to re-add later as 'extra'
        extra = (dts.ps // 1000) * (pps // 1_000_000_000)

        dt = datetime_new(dts.year, dts.month, dts.day, dts.hour,
                          dts.min, dts.sec, dts.us, None)

        aware = dt.replace(tzinfo=tz)
        as_utc = aware.astimezone(utc_stdlib)
        rt = as_utc.astimezone(tz)
        if aware != rt:
            # AFAICT this means that 'aware' is non-existent
            # TODO: better way to check this?
            #  mail.python.org/archives/list/datetime-sig@python.org/
            #  thread/57Y3IQAASJOKHX4D27W463XTZIS2NR3M/
            result_a[i] = NPY_NAT
        else:
            left = as_utc.replace(tzinfo=None)
            result_a[i] = pydatetime_to_dt64(left, &dts, creso) + extra

        aware = dt.replace(fold=1, tzinfo=tz)
        as_utc = aware.astimezone(utc_stdlib)
        rt = as_utc.astimezone(tz)
        if aware != rt:
            result_b[i] = NPY_NAT
        else:
            right = as_utc.replace(tzinfo=None)
            result_b[i] = pydatetime_to_dt64(right, &dts, creso) + extra

    return result_a, result_b


@cython.boundscheck(False)
cdef ndarray[int64_t] _get_dst_hours(
    # vals, creso only needed here to potential render an exception message
    const int64_t[:] vals,
    ndarray[int64_t] result_a,
    ndarray[int64_t] result_b,
    NPY_DATETIMEUNIT creso,
):
    cdef:
        Py_ssize_t i, n = vals.shape[0]
        ndarray[uint8_t, cast=True] mismatch
        ndarray[int64_t] delta, dst_hours
        ndarray[intp_t] switch_idxs, trans_idx, grp, a_idx, b_idx, one_diff
        list trans_grp
        intp_t switch_idx
        int64_t left, right

    dst_hours = cnp.PyArray_EMPTY(result_a.ndim, result_a.shape, cnp.NPY_INT64, 0)
    dst_hours[:] = NPY_NAT

    mismatch = cnp.PyArray_ZEROS(result_a.ndim, result_a.shape, cnp.NPY_BOOL, 0)

    for i in range(n):
        left = result_a[i]
        right = result_b[i]

        # Get the ambiguous hours (given the above, these are the hours
        # where result_a != result_b and neither of them are NAT)
        if left != right and left != NPY_NAT and right != NPY_NAT:
            mismatch[i] = 1

    trans_idx = mismatch.nonzero()[0]

    if trans_idx.size == 1:
        # see test_tz_localize_to_utc_ambiguous_infer
        stamp = _render_tstamp(vals[trans_idx[0]], creso=creso)
        raise pytz.AmbiguousTimeError(
            f"Cannot infer dst time from {stamp} as there "
            "are no repeated times"
        )

    # Split the array into contiguous chunks (where the difference between
    # indices is 1).  These are effectively dst transitions in different
    # years which is useful for checking that there is not an ambiguous
    # transition in an individual year.
    if trans_idx.size > 0:
        one_diff = np.where(np.diff(trans_idx) != 1)[0] + 1
        trans_grp = np.array_split(trans_idx, one_diff)

        # Iterate through each day, if there are no hours where the
        # delta is negative (indicates a repeat of hour) the switch
        # cannot be inferred
        for grp in trans_grp:

            delta = np.diff(result_a[grp])
            if grp.size == 1 or np.all(delta > 0):
                # see test_tz_localize_to_utc_ambiguous_infer
                stamp = _render_tstamp(vals[grp[0]], creso=creso)
                raise pytz.AmbiguousTimeError(stamp)

            # Find the index for the switch and pull from a for dst and b
            # for standard
            switch_idxs = (delta <= 0).nonzero()[0]
            if switch_idxs.size > 1:
                # see test_tz_localize_to_utc_ambiguous_infer
                raise pytz.AmbiguousTimeError(
                    f"There are {switch_idxs.size} dst switches when "
                    "there should only be 1."
                )

            switch_idx = switch_idxs[0] + 1
            # Pull the only index and adjust
            a_idx = grp[:switch_idx]
            b_idx = grp[switch_idx:]
            dst_hours[grp] = np.hstack((result_a[a_idx], result_b[b_idx]))

    return dst_hours


# ----------------------------------------------------------------------
# Timezone Conversion

cpdef int64_t tz_convert_from_utc_single(
    int64_t utc_val, tzinfo tz, NPY_DATETIMEUNIT creso=NPY_DATETIMEUNIT.NPY_FR_ns
) except? -1:
    """
    Convert the val (in i8) from UTC to tz

    This is a single value version of tz_convert_from_utc.

    Parameters
    ----------
    utc_val : int64
    tz : tzinfo
    creso : NPY_DATETIMEUNIT, default NPY_FR_ns

    Returns
    -------
    converted: int64
    """
    cdef:
        Localizer info = Localizer(tz, creso=creso)
        Py_ssize_t pos

    # Note: caller is responsible for ensuring utc_val != NPY_NAT
    return info.utc_val_to_local_val(utc_val, &pos)


# OSError may be thrown by tzlocal on windows at or close to 1970-01-01
#  see https://github.com/pandas-dev/pandas/pull/37591#issuecomment-720628241
cdef int64_t _tz_localize_using_tzinfo_api(
    int64_t val,
    tzinfo tz,
    bint to_utc=True,
    NPY_DATETIMEUNIT creso=NPY_DATETIMEUNIT.NPY_FR_ns,
    bint* fold=NULL,
) except? -1:
    """
    Convert the i8 representation of a datetime from a general-case timezone to
    UTC, or vice-versa using the datetime/tzinfo API.

    Private, not intended for use outside of tslibs.tzconversion.

    Parameters
    ----------
    val : int64_t
    tz : tzinfo
    to_utc : bint
        True if converting _to_ UTC, False if going the other direction.
    creso : NPY_DATETIMEUNIT
    fold : bint*, default NULL
        pointer to fold: whether datetime ends up in a fold or not
        after adjustment.
        Only passed with to_utc=False.

    Returns
    -------
    delta : int64_t
        Value to add when converting from utc, subtract when converting to utc.

    Notes
    -----
    Sets fold by pointer
    """
    cdef:
        npy_datetimestruct dts
        datetime dt
        int64_t delta
        timedelta td
        int64_t pps = periods_per_second(creso)

    pandas_datetime_to_datetimestruct(val, creso, &dts)

    # datetime_new is cython-optimized constructor
    if not to_utc:
        # tz.utcoffset only makes sense if datetime
        # is _wall time_, so if val is a UTC timestamp convert to wall time
        dt = _astimezone(dts, tz)

        if fold is not NULL:
            # NB: fold is only passed with to_utc=False
            fold[0] = dt.fold
    else:
        dt = datetime_new(dts.year, dts.month, dts.day, dts.hour,
                          dts.min, dts.sec, dts.us, None)

    td = tz.utcoffset(dt)
    delta = int(td.total_seconds() * pps)
    return delta


cdef datetime _astimezone(npy_datetimestruct dts, tzinfo tz):
    """
    Optimized equivalent to:

    dt = datetime(dts.year, dts.month, dts.day, dts.hour,
                  dts.min, dts.sec, dts.us, utc_stdlib)
    dt = dt.astimezone(tz)

    Derived from the datetime.astimezone implementation at
    https://github.com/python/cpython/blob/main/Modules/_datetimemodule.c#L6187

    NB: we are assuming tz is not None.
    """
    cdef:
        datetime result

    result = datetime_new(dts.year, dts.month, dts.day, dts.hour,
                          dts.min, dts.sec, dts.us, tz)
    return tz.fromutc(result)


# NB: relies on dateutil internals, subject to change.
@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint _infer_dateutil_fold(
    int64_t value,
    const int64_t[::1] trans,
    const int64_t[::1] deltas,
    Py_ssize_t pos,
):
    """
    Infer _TSObject fold property from value by assuming 0 and then setting
    to 1 if necessary.

    Parameters
    ----------
    value : int64_t
    trans : ndarray[int64_t]
        ndarray of offset transition points in nanoseconds since epoch.
    deltas : int64_t[:]
        array of offsets corresponding to transition points in trans.
    pos : Py_ssize_t
        Position of the last transition point before taking fold into account.

    Returns
    -------
    bint
        Due to daylight saving time, one wall clock time can occur twice
        when shifting from summer to winter time; fold describes whether the
        datetime-like corresponds  to the first (0) or the second time (1)
        the wall clock hits the ambiguous time

    References
    ----------
    .. [1] "PEP 495 - Local Time Disambiguation"
           https://www.python.org/dev/peps/pep-0495/#the-fold-attribute
    """
    cdef:
        bint fold = 0
        int64_t fold_delta

    if pos > 0:
        fold_delta = deltas[pos - 1] - deltas[pos]
        if value - fold_delta < trans[pos]:
            fold = 1

    return fold
