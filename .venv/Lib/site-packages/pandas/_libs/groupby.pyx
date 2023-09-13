cimport cython
from cython cimport (
    Py_ssize_t,
    floating,
)
from libc.math cimport (
    NAN,
    sqrt,
)
from libc.stdlib cimport (
    free,
    malloc,
)

import numpy as np

cimport numpy as cnp
from numpy cimport (
    complex64_t,
    complex128_t,
    float32_t,
    float64_t,
    int8_t,
    int64_t,
    intp_t,
    ndarray,
    uint8_t,
    uint64_t,
)

cnp.import_array()

from pandas._libs cimport util
from pandas._libs.algos cimport (
    get_rank_nan_fill_val,
    kth_smallest_c,
)

from pandas._libs.algos import (
    groupsort_indexer,
    rank_1d,
    take_2d_axis1_bool_bool,
    take_2d_axis1_float64_float64,
)

from pandas._libs.dtypes cimport (
    numeric_object_t,
    numeric_t,
)
from pandas._libs.missing cimport checknull


cdef int64_t NPY_NAT = util.get_nat()

cdef float64_t NaN = <float64_t>np.nan

cdef enum InterpolationEnumType:
    INTERPOLATION_LINEAR,
    INTERPOLATION_LOWER,
    INTERPOLATION_HIGHER,
    INTERPOLATION_NEAREST,
    INTERPOLATION_MIDPOINT


cdef float64_t median_linear_mask(float64_t* a, int n, uint8_t* mask) noexcept nogil:
    cdef:
        int i, j, na_count = 0
        float64_t* tmp
        float64_t result

    if n == 0:
        return NaN

    # count NAs
    for i in range(n):
        if mask[i]:
            na_count += 1

    if na_count:
        if na_count == n:
            return NaN

        tmp = <float64_t*>malloc((n - na_count) * sizeof(float64_t))

        j = 0
        for i in range(n):
            if not mask[i]:
                tmp[j] = a[i]
                j += 1

        a = tmp
        n -= na_count

    result = calc_median_linear(a, n)

    if na_count:
        free(a)

    return result


cdef float64_t median_linear(float64_t* a, int n) noexcept nogil:
    cdef:
        int i, j, na_count = 0
        float64_t* tmp
        float64_t result

    if n == 0:
        return NaN

    # count NAs
    for i in range(n):
        if a[i] != a[i]:
            na_count += 1

    if na_count:
        if na_count == n:
            return NaN

        tmp = <float64_t*>malloc((n - na_count) * sizeof(float64_t))

        j = 0
        for i in range(n):
            if a[i] == a[i]:
                tmp[j] = a[i]
                j += 1

        a = tmp
        n -= na_count

    result = calc_median_linear(a, n)

    if na_count:
        free(a)

    return result


cdef float64_t calc_median_linear(float64_t* a, int n) noexcept nogil:
    cdef:
        float64_t result

    if n % 2:
        result = kth_smallest_c(a, n // 2, n)
    else:
        result = (kth_smallest_c(a, n // 2, n) +
                  kth_smallest_c(a, n // 2 - 1, n)) / 2

    return result


ctypedef fused int64float_t:
    int64_t
    uint64_t
    float32_t
    float64_t


@cython.boundscheck(False)
@cython.wraparound(False)
def group_median_float64(
    ndarray[float64_t, ndim=2] out,
    ndarray[int64_t] counts,
    ndarray[float64_t, ndim=2] values,
    ndarray[intp_t] labels,
    Py_ssize_t min_count=-1,
    const uint8_t[:, :] mask=None,
    uint8_t[:, ::1] result_mask=None,
) -> None:
    """
    Only aggregates on axis=0
    """
    cdef:
        Py_ssize_t i, j, N, K, ngroups, size
        ndarray[intp_t] _counts
        ndarray[float64_t, ndim=2] data
        ndarray[uint8_t, ndim=2] data_mask
        ndarray[intp_t] indexer
        float64_t* ptr
        uint8_t* ptr_mask
        float64_t result
        bint uses_mask = mask is not None

    assert min_count == -1, "'min_count' only used in sum and prod"

    ngroups = len(counts)
    N, K = (<object>values).shape

    indexer, _counts = groupsort_indexer(labels, ngroups)
    counts[:] = _counts[1:]

    data = np.empty((K, N), dtype=np.float64)
    ptr = <float64_t*>cnp.PyArray_DATA(data)

    take_2d_axis1_float64_float64(values.T, indexer, out=data)

    if uses_mask:
        data_mask = np.empty((K, N), dtype=np.uint8)
        ptr_mask = <uint8_t *>cnp.PyArray_DATA(data_mask)

        take_2d_axis1_bool_bool(mask.T, indexer, out=data_mask, fill_value=1)

        with nogil:

            for i in range(K):
                # exclude NA group
                ptr += _counts[0]
                ptr_mask += _counts[0]

                for j in range(ngroups):
                    size = _counts[j + 1]
                    result = median_linear_mask(ptr, size, ptr_mask)
                    out[j, i] = result

                    if result != result:
                        result_mask[j, i] = 1
                    ptr += size
                    ptr_mask += size

    else:
        with nogil:
            for i in range(K):
                # exclude NA group
                ptr += _counts[0]
                for j in range(ngroups):
                    size = _counts[j + 1]
                    out[j, i] = median_linear(ptr, size)
                    ptr += size


@cython.boundscheck(False)
@cython.wraparound(False)
def group_cumprod(
    int64float_t[:, ::1] out,
    ndarray[int64float_t, ndim=2] values,
    const intp_t[::1] labels,
    int ngroups,
    bint is_datetimelike,
    bint skipna=True,
    const uint8_t[:, :] mask=None,
    uint8_t[:, ::1] result_mask=None,
) -> None:
    """
    Cumulative product of columns of `values`, in row groups `labels`.

    Parameters
    ----------
    out : np.ndarray[np.float64, ndim=2]
        Array to store cumprod in.
    values : np.ndarray[np.float64, ndim=2]
        Values to take cumprod of.
    labels : np.ndarray[np.intp]
        Labels to group by.
    ngroups : int
        Number of groups, larger than all entries of `labels`.
    is_datetimelike : bool
        Always false, `values` is never datetime-like.
    skipna : bool
        If true, ignore nans in `values`.
    mask : np.ndarray[uint8], optional
        Mask of values
    result_mask : np.ndarray[int8], optional
        Mask of out array

    Notes
    -----
    This method modifies the `out` parameter, rather than returning an object.
    """
    cdef:
        Py_ssize_t i, j, N, K
        int64float_t val, na_val
        int64float_t[:, ::1] accum
        intp_t lab
        uint8_t[:, ::1] accum_mask
        bint isna_entry, isna_prev = False
        bint uses_mask = mask is not None

    N, K = (<object>values).shape
    accum = np.ones((ngroups, K), dtype=(<object>values).dtype)
    na_val = _get_na_val(<int64float_t>0, is_datetimelike)
    accum_mask = np.zeros((ngroups, K), dtype="uint8")

    with nogil:
        for i in range(N):
            lab = labels[i]

            if lab < 0:
                continue
            for j in range(K):
                val = values[i, j]

                if uses_mask:
                    isna_entry = mask[i, j]
                else:
                    isna_entry = _treat_as_na(val, False)

                if not isna_entry:
                    isna_prev = accum_mask[lab, j]
                    if isna_prev:
                        out[i, j] = na_val
                        if uses_mask:
                            result_mask[i, j] = True

                    else:
                        accum[lab, j] *= val
                        out[i, j] = accum[lab, j]

                else:
                    if uses_mask:
                        result_mask[i, j] = True
                        out[i, j] = 0
                    else:
                        out[i, j] = na_val

                    if not skipna:
                        accum[lab, j] = na_val
                        accum_mask[lab, j] = True


@cython.boundscheck(False)
@cython.wraparound(False)
def group_cumsum(
    int64float_t[:, ::1] out,
    ndarray[int64float_t, ndim=2] values,
    const intp_t[::1] labels,
    int ngroups,
    bint is_datetimelike,
    bint skipna=True,
    const uint8_t[:, :] mask=None,
    uint8_t[:, ::1] result_mask=None,
) -> None:
    """
    Cumulative sum of columns of `values`, in row groups `labels`.

    Parameters
    ----------
    out : np.ndarray[ndim=2]
        Array to store cumsum in.
    values : np.ndarray[ndim=2]
        Values to take cumsum of.
    labels : np.ndarray[np.intp]
        Labels to group by.
    ngroups : int
        Number of groups, larger than all entries of `labels`.
    is_datetimelike : bool
        True if `values` contains datetime-like entries.
    skipna : bool
        If true, ignore nans in `values`.
    mask : np.ndarray[uint8], optional
        Mask of values
    result_mask : np.ndarray[int8], optional
        Mask of out array

    Notes
    -----
    This method modifies the `out` parameter, rather than returning an object.
    """
    cdef:
        Py_ssize_t i, j, N, K
        int64float_t val, y, t, na_val
        int64float_t[:, ::1] accum, compensation
        uint8_t[:, ::1] accum_mask
        intp_t lab
        bint isna_entry, isna_prev = False
        bint uses_mask = mask is not None

    N, K = (<object>values).shape

    if uses_mask:
        accum_mask = np.zeros((ngroups, K), dtype="uint8")

    accum = np.zeros((ngroups, K), dtype=np.asarray(values).dtype)
    compensation = np.zeros((ngroups, K), dtype=np.asarray(values).dtype)

    na_val = _get_na_val(<int64float_t>0, is_datetimelike)

    with nogil:
        for i in range(N):
            lab = labels[i]

            if lab < 0:
                continue
            for j in range(K):
                val = values[i, j]

                if uses_mask:
                    isna_entry = mask[i, j]
                else:
                    isna_entry = _treat_as_na(val, is_datetimelike)

                if not skipna:
                    if uses_mask:
                        isna_prev = accum_mask[lab, j]
                    else:
                        isna_prev = _treat_as_na(accum[lab, j], is_datetimelike)

                    if isna_prev:
                        if uses_mask:
                            result_mask[i, j] = True
                            # Be deterministic, out was initialized as empty
                            out[i, j] = 0
                        else:
                            out[i, j] = na_val
                        continue

                if isna_entry:

                    if uses_mask:
                        result_mask[i, j] = True
                        # Be deterministic, out was initialized as empty
                        out[i, j] = 0
                    else:
                        out[i, j] = na_val

                    if not skipna:
                        if uses_mask:
                            accum_mask[lab, j] = True
                        else:
                            accum[lab, j] = na_val

                else:
                    # For floats, use Kahan summation to reduce floating-point
                    # error (https://en.wikipedia.org/wiki/Kahan_summation_algorithm)
                    if int64float_t == float32_t or int64float_t == float64_t:
                        y = val - compensation[lab, j]
                        t = accum[lab, j] + y
                        compensation[lab, j] = t - accum[lab, j] - y
                    else:
                        t = val + accum[lab, j]

                    accum[lab, j] = t
                    out[i, j] = t


@cython.boundscheck(False)
@cython.wraparound(False)
def group_shift_indexer(
    int64_t[::1] out,
    const intp_t[::1] labels,
    int ngroups,
    int periods,
) -> None:
    cdef:
        Py_ssize_t N, i, ii, lab
        int offset = 0, sign
        int64_t idxer, idxer_slot
        int64_t[::1] label_seen = np.zeros(ngroups, dtype=np.int64)
        int64_t[:, ::1] label_indexer

    N, = (<object>labels).shape

    if periods < 0:
        periods = -periods
        offset = N - 1
        sign = -1
    elif periods > 0:
        offset = 0
        sign = 1

    if periods == 0:
        with nogil:
            for i in range(N):
                out[i] = i
    else:
        # array of each previous indexer seen
        label_indexer = np.zeros((ngroups, periods), dtype=np.int64)
        with nogil:
            for i in range(N):
                # reverse iterator if shifting backwards
                ii = offset + sign * i
                lab = labels[ii]

                # Skip null keys
                if lab == -1:
                    out[ii] = -1
                    continue

                label_seen[lab] += 1

                idxer_slot = label_seen[lab] % periods
                idxer = label_indexer[lab, idxer_slot]

                if label_seen[lab] > periods:
                    out[ii] = idxer
                else:
                    out[ii] = -1

                label_indexer[lab, idxer_slot] = ii


@cython.wraparound(False)
@cython.boundscheck(False)
def group_fillna_indexer(
    ndarray[intp_t] out,
    ndarray[intp_t] labels,
    ndarray[intp_t] sorted_labels,
    ndarray[uint8_t] mask,
    int64_t limit,
    bint dropna,
) -> None:
    """
    Indexes how to fill values forwards or backwards within a group.

    Parameters
    ----------
    out : np.ndarray[np.intp]
        Values into which this method will write its results.
    labels : np.ndarray[np.intp]
        Array containing unique label for each group, with its ordering
        matching up to the corresponding record in `values`.
    sorted_labels : np.ndarray[np.intp]
        obtained by `np.argsort(labels, kind="mergesort")`
    values : np.ndarray[np.uint8]
        Containing the truth value of each element.
    mask : np.ndarray[np.uint8]
        Indicating whether a value is na or not.
    limit : Consecutive values to fill before stopping, or -1 for no limit
    dropna : Flag to indicate if NaN groups should return all NaN values

    Notes
    -----
    This method modifies the `out` parameter rather than returning an object
    """
    cdef:
        Py_ssize_t i, N, idx
        intp_t curr_fill_idx=-1
        int64_t filled_vals = 0

    N = len(out)

    # Make sure all arrays are the same size
    assert N == len(labels) == len(mask)

    with nogil:
        for i in range(N):
            idx = sorted_labels[i]
            if dropna and labels[idx] == -1:  # nan-group gets nan-values
                curr_fill_idx = -1
            elif mask[idx] == 1:  # is missing
                # Stop filling once we've hit the limit
                if filled_vals >= limit and limit != -1:
                    curr_fill_idx = -1
                filled_vals += 1
            else:  # reset items when not missing
                filled_vals = 0
                curr_fill_idx = idx

            out[idx] = curr_fill_idx

            # If we move to the next group, reset
            # the fill_idx and counter
            if i == N - 1 or labels[idx] != labels[sorted_labels[i + 1]]:
                curr_fill_idx = -1
                filled_vals = 0


@cython.boundscheck(False)
@cython.wraparound(False)
def group_any_all(
    int8_t[:, ::1] out,
    const int8_t[:, :] values,
    const intp_t[::1] labels,
    const uint8_t[:, :] mask,
    str val_test,
    bint skipna,
    uint8_t[:, ::1] result_mask,
) -> None:
    """
    Aggregated boolean values to show truthfulness of group elements. If the
    input is a nullable type (result_mask is not None), the result will be computed
    using Kleene logic.

    Parameters
    ----------
    out : np.ndarray[np.int8]
        Values into which this method will write its results.
    labels : np.ndarray[np.intp]
        Array containing unique label for each group, with its
        ordering matching up to the corresponding record in `values`
    values : np.ndarray[np.int8]
        Containing the truth value of each element.
    mask : np.ndarray[np.uint8]
        Indicating whether a value is na or not.
    val_test : {'any', 'all'}
        String object dictating whether to use any or all truth testing
    skipna : bool
        Flag to ignore nan values during truth testing
    result_mask : ndarray[bool, ndim=2], optional
        If not None, these specify locations in the output that are NA.
        Modified in-place.

    Notes
    -----
    This method modifies the `out` parameter rather than returning an object.
    The returned values will either be 0, 1 (False or True, respectively), or
    -1 to signify a masked position in the case of a nullable input.
    """
    cdef:
        Py_ssize_t i, j, N = len(labels), K = out.shape[1]
        intp_t lab
        int8_t flag_val, val
        bint uses_mask = result_mask is not None

    if val_test == "all":
        # Because the 'all' value of an empty iterable in Python is True we can
        # start with an array full of ones and set to zero when a False value
        # is encountered
        flag_val = 0
    elif val_test == "any":
        # Because the 'any' value of an empty iterable in Python is False we
        # can start with an array full of zeros and set to one only if any
        # value encountered is True
        flag_val = 1
    else:
        raise ValueError("'val_test' must be either 'any' or 'all'!")

    out[:] = 1 - flag_val

    with nogil:
        for i in range(N):
            lab = labels[i]
            if lab < 0:
                continue

            for j in range(K):
                if skipna and mask[i, j]:
                    continue

                if uses_mask and mask[i, j]:
                    # Set the position as masked if `out[lab] != flag_val`, which
                    # would indicate True/False has not yet been seen for any/all,
                    # so by Kleene logic the result is currently unknown
                    if out[lab, j] != flag_val:
                        result_mask[lab, j] = 1
                    continue

                val = values[i, j]

                # If True and 'any' or False and 'all', the result is
                # already determined
                if val == flag_val:
                    out[lab, j] = flag_val
                    if uses_mask:
                        result_mask[lab, j] = 0


# ----------------------------------------------------------------------
# group_sum, group_prod, group_var, group_mean, group_ohlc
# ----------------------------------------------------------------------

ctypedef fused mean_t:
    float64_t
    float32_t
    complex64_t
    complex128_t

ctypedef fused sum_t:
    mean_t
    int64_t
    uint64_t
    object


@cython.wraparound(False)
@cython.boundscheck(False)
def group_sum(
    sum_t[:, ::1] out,
    int64_t[::1] counts,
    ndarray[sum_t, ndim=2] values,
    const intp_t[::1] labels,
    const uint8_t[:, :] mask,
    uint8_t[:, ::1] result_mask=None,
    Py_ssize_t min_count=0,
    bint is_datetimelike=False,
) -> None:
    """
    Only aggregates on axis=0 using Kahan summation
    """
    cdef:
        Py_ssize_t i, j, N, K, lab, ncounts = len(counts)
        sum_t val, t, y
        sum_t[:, ::1] sumx, compensation
        int64_t[:, ::1] nobs
        Py_ssize_t len_values = len(values), len_labels = len(labels)
        bint uses_mask = mask is not None
        bint isna_entry

    if len_values != len_labels:
        raise ValueError("len(index) != len(labels)")

    nobs = np.zeros((<object>out).shape, dtype=np.int64)
    # the below is equivalent to `np.zeros_like(out)` but faster
    sumx = np.zeros((<object>out).shape, dtype=(<object>out).base.dtype)
    compensation = np.zeros((<object>out).shape, dtype=(<object>out).base.dtype)

    N, K = (<object>values).shape

    if sum_t is object:
        # NB: this does not use 'compensation' like the non-object track does.
        for i in range(N):
            lab = labels[i]
            if lab < 0:
                continue

            counts[lab] += 1
            for j in range(K):
                val = values[i, j]

                # not nan
                if not checknull(val):
                    nobs[lab, j] += 1

                    if nobs[lab, j] == 1:
                        # i.e. we haven't added anything yet; avoid TypeError
                        #  if e.g. val is a str and sumx[lab, j] is 0
                        t = val
                    else:
                        t = sumx[lab, j] + val
                    sumx[lab, j] = t

        for i in range(ncounts):
            for j in range(K):
                if nobs[i, j] < min_count:
                    out[i, j] = None

                else:
                    out[i, j] = sumx[i, j]
    else:
        with nogil:
            for i in range(N):
                lab = labels[i]
                if lab < 0:
                    continue

                counts[lab] += 1
                for j in range(K):
                    val = values[i, j]

                    if uses_mask:
                        isna_entry = mask[i, j]
                    else:
                        isna_entry = _treat_as_na(val, is_datetimelike)

                    if not isna_entry:
                        nobs[lab, j] += 1
                        y = val - compensation[lab, j]
                        t = sumx[lab, j] + y
                        compensation[lab, j] = t - sumx[lab, j] - y
                        if compensation[lab, j] != compensation[lab, j]:
                            # GH#53606
                            # If val is +/- infinity compensation is NaN
                            # which would lead to results being NaN instead
                            # of +/- infinity. We cannot use util.is_nan
                            # because of no gil
                            compensation[lab, j] = 0
                        sumx[lab, j] = t

            _check_below_mincount(
                out, uses_mask, result_mask, ncounts, K, nobs, min_count, sumx
            )


@cython.wraparound(False)
@cython.boundscheck(False)
def group_prod(
    int64float_t[:, ::1] out,
    int64_t[::1] counts,
    ndarray[int64float_t, ndim=2] values,
    const intp_t[::1] labels,
    const uint8_t[:, ::1] mask,
    uint8_t[:, ::1] result_mask=None,
    Py_ssize_t min_count=0,
) -> None:
    """
    Only aggregates on axis=0
    """
    cdef:
        Py_ssize_t i, j, N, K, lab, ncounts = len(counts)
        int64float_t val
        int64float_t[:, ::1] prodx
        int64_t[:, ::1] nobs
        Py_ssize_t len_values = len(values), len_labels = len(labels)
        bint isna_entry, uses_mask = mask is not None

    if len_values != len_labels:
        raise ValueError("len(index) != len(labels)")

    nobs = np.zeros((<object>out).shape, dtype=np.int64)
    prodx = np.ones((<object>out).shape, dtype=(<object>out).base.dtype)

    N, K = (<object>values).shape

    with nogil:
        for i in range(N):
            lab = labels[i]
            if lab < 0:
                continue

            counts[lab] += 1
            for j in range(K):
                val = values[i, j]

                if uses_mask:
                    isna_entry = mask[i, j]
                else:
                    isna_entry = _treat_as_na(val, False)

                if not isna_entry:
                    nobs[lab, j] += 1
                    prodx[lab, j] *= val

        _check_below_mincount(
            out, uses_mask, result_mask, ncounts, K, nobs, min_count, prodx
        )


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def group_var(
    floating[:, ::1] out,
    int64_t[::1] counts,
    ndarray[floating, ndim=2] values,
    const intp_t[::1] labels,
    Py_ssize_t min_count=-1,
    int64_t ddof=1,
    const uint8_t[:, ::1] mask=None,
    uint8_t[:, ::1] result_mask=None,
    bint is_datetimelike=False,
    str name="var",
) -> None:
    cdef:
        Py_ssize_t i, j, N, K, lab, ncounts = len(counts)
        floating val, ct, oldmean
        floating[:, ::1] mean
        int64_t[:, ::1] nobs
        Py_ssize_t len_values = len(values), len_labels = len(labels)
        bint isna_entry, uses_mask = mask is not None
        bint is_std = name == "std"
        bint is_sem = name == "sem"

    assert min_count == -1, "'min_count' only used in sum and prod"

    if len_values != len_labels:
        raise ValueError("len(index) != len(labels)")

    nobs = np.zeros((<object>out).shape, dtype=np.int64)
    mean = np.zeros((<object>out).shape, dtype=(<object>out).base.dtype)

    N, K = (<object>values).shape

    out[:, :] = 0.0

    with nogil:
        for i in range(N):
            lab = labels[i]
            if lab < 0:
                continue

            counts[lab] += 1

            for j in range(K):
                val = values[i, j]

                if uses_mask:
                    isna_entry = mask[i, j]
                elif is_datetimelike:
                    # With group_var, we cannot just use _treat_as_na bc
                    #  datetimelike dtypes get cast to float64 instead of
                    #  to int64.
                    isna_entry = val == NPY_NAT
                else:
                    isna_entry = _treat_as_na(val, is_datetimelike)

                if not isna_entry:
                    nobs[lab, j] += 1
                    oldmean = mean[lab, j]
                    mean[lab, j] += (val - oldmean) / nobs[lab, j]
                    out[lab, j] += (val - mean[lab, j]) * (val - oldmean)

        for i in range(ncounts):
            for j in range(K):
                ct = nobs[i, j]
                if ct <= ddof:
                    if uses_mask:
                        result_mask[i, j] = True
                    else:
                        out[i, j] = NAN
                else:
                    if is_std:
                        out[i, j] = sqrt(out[i, j] / (ct - ddof))
                    elif is_sem:
                        out[i, j] = sqrt(out[i, j] / (ct - ddof) / ct)
                    else:
                        # just "var"
                        out[i, j] /= (ct - ddof)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.cpow
def group_skew(
    float64_t[:, ::1] out,
    int64_t[::1] counts,
    ndarray[float64_t, ndim=2] values,
    const intp_t[::1] labels,
    const uint8_t[:, ::1] mask=None,
    uint8_t[:, ::1] result_mask=None,
    bint skipna=True,
) -> None:
    cdef:
        Py_ssize_t i, j, N, K, lab, ngroups = len(counts)
        int64_t[:, ::1] nobs
        Py_ssize_t len_values = len(values), len_labels = len(labels)
        bint isna_entry, uses_mask = mask is not None
        float64_t[:, ::1] M1, M2, M3
        float64_t delta, delta_n, term1, val
        int64_t n1, n
        float64_t ct

    if len_values != len_labels:
        raise ValueError("len(index) != len(labels)")

    nobs = np.zeros((<object>out).shape, dtype=np.int64)

    # M1, M2, and M3 correspond to 1st, 2nd, and third Moments
    M1 = np.zeros((<object>out).shape, dtype=np.float64)
    M2 = np.zeros((<object>out).shape, dtype=np.float64)
    M3 = np.zeros((<object>out).shape, dtype=np.float64)

    N, K = (<object>values).shape

    out[:, :] = 0.0

    with nogil:
        for i in range(N):
            lab = labels[i]
            if lab < 0:
                continue

            counts[lab] += 1

            for j in range(K):
                val = values[i, j]

                if uses_mask:
                    isna_entry = mask[i, j]
                else:
                    isna_entry = _treat_as_na(val, False)

                if not isna_entry:
                    # Based on RunningStats::Push from
                    #  https://www.johndcook.com/blog/skewness_kurtosis/
                    n1 = nobs[lab, j]
                    n = n1 + 1

                    nobs[lab, j] = n
                    delta = val - M1[lab, j]
                    delta_n = delta / n
                    term1 = delta * delta_n * n1

                    M1[lab, j] += delta_n
                    M3[lab, j] += term1 * delta_n * (n - 2) - 3 * delta_n * M2[lab, j]
                    M2[lab, j] += term1
                elif not skipna:
                    M1[lab, j] = NaN
                    M2[lab, j] = NaN
                    M3[lab, j] = NaN

        for i in range(ngroups):
            for j in range(K):
                ct = <float64_t>nobs[i, j]
                if ct < 3:
                    if result_mask is not None:
                        result_mask[i, j] = 1
                    out[i, j] = NaN
                elif M2[i, j] == 0:
                    out[i, j] = 0
                else:
                    out[i, j] = (
                        (ct * (ct - 1) ** 0.5 / (ct - 2))
                        * (M3[i, j] / M2[i, j] ** 1.5)
                    )


@cython.wraparound(False)
@cython.boundscheck(False)
def group_mean(
    mean_t[:, ::1] out,
    int64_t[::1] counts,
    ndarray[mean_t, ndim=2] values,
    const intp_t[::1] labels,
    Py_ssize_t min_count=-1,
    bint is_datetimelike=False,
    const uint8_t[:, ::1] mask=None,
    uint8_t[:, ::1] result_mask=None,
) -> None:
    """
    Compute the mean per label given a label assignment for each value.
    NaN values are ignored.

    Parameters
    ----------
    out : np.ndarray[floating]
        Values into which this method will write its results.
    counts : np.ndarray[int64]
        A zeroed array of the same shape as labels,
        populated by group sizes during algorithm.
    values : np.ndarray[floating]
        2-d array of the values to find the mean of.
    labels : np.ndarray[np.intp]
        Array containing unique label for each group, with its
        ordering matching up to the corresponding record in `values`.
    min_count : Py_ssize_t
        Only used in sum and prod. Always -1.
    is_datetimelike : bool
        True if `values` contains datetime-like entries.
    mask : ndarray[bool, ndim=2], optional
        Mask of the input values.
    result_mask : ndarray[bool, ndim=2], optional
        Mask of the out array

    Notes
    -----
    This method modifies the `out` parameter rather than returning an object.
    `counts` is modified to hold group sizes
    """

    cdef:
        Py_ssize_t i, j, N, K, lab, ncounts = len(counts)
        mean_t val, count, y, t, nan_val
        mean_t[:, ::1] sumx, compensation
        int64_t[:, ::1] nobs
        Py_ssize_t len_values = len(values), len_labels = len(labels)
        bint isna_entry, uses_mask = mask is not None

    assert min_count == -1, "'min_count' only used in sum and prod"

    if len_values != len_labels:
        raise ValueError("len(index) != len(labels)")

    # the below is equivalent to `np.zeros_like(out)` but faster
    nobs = np.zeros((<object>out).shape, dtype=np.int64)
    sumx = np.zeros((<object>out).shape, dtype=(<object>out).base.dtype)
    compensation = np.zeros((<object>out).shape, dtype=(<object>out).base.dtype)

    N, K = (<object>values).shape
    if uses_mask:
        nan_val = 0
    elif is_datetimelike:
        nan_val = NPY_NAT
    else:
        nan_val = NAN

    with nogil:
        for i in range(N):
            lab = labels[i]
            if lab < 0:
                continue

            counts[lab] += 1
            for j in range(K):
                val = values[i, j]

                if uses_mask:
                    isna_entry = mask[i, j]
                elif is_datetimelike:
                    # With group_mean, we cannot just use _treat_as_na bc
                    #  datetimelike dtypes get cast to float64 instead of
                    #  to int64.
                    isna_entry = val == NPY_NAT
                else:
                    isna_entry = _treat_as_na(val, is_datetimelike)

                if not isna_entry:
                    nobs[lab, j] += 1
                    y = val - compensation[lab, j]
                    t = sumx[lab, j] + y
                    compensation[lab, j] = t - sumx[lab, j] - y
                    if compensation[lab, j] != compensation[lab, j]:
                        # GH#50367
                        # If val is +/- infinity, compensation is NaN
                        # which would lead to results being NaN instead
                        # of +/-infinity. We cannot use util.is_nan
                        # because of no gil
                        compensation[lab, j] = 0.
                    sumx[lab, j] = t

        for i in range(ncounts):
            for j in range(K):
                count = nobs[i, j]
                if nobs[i, j] == 0:

                    if uses_mask:
                        result_mask[i, j] = True
                    else:
                        out[i, j] = nan_val

                else:
                    out[i, j] = sumx[i, j] / count


@cython.wraparound(False)
@cython.boundscheck(False)
def group_ohlc(
    int64float_t[:, ::1] out,
    int64_t[::1] counts,
    ndarray[int64float_t, ndim=2] values,
    const intp_t[::1] labels,
    Py_ssize_t min_count=-1,
    const uint8_t[:, ::1] mask=None,
    uint8_t[:, ::1] result_mask=None,
) -> None:
    """
    Only aggregates on axis=0
    """
    cdef:
        Py_ssize_t i, N, K, lab
        int64float_t val
        uint8_t[::1] first_element_set
        bint isna_entry, uses_mask = mask is not None

    assert min_count == -1, "'min_count' only used in sum and prod"

    if len(labels) == 0:
        return

    N, K = (<object>values).shape

    if out.shape[1] != 4:
        raise ValueError("Output array must have 4 columns")

    if K > 1:
        raise NotImplementedError("Argument 'values' must have only one dimension")

    if int64float_t is float32_t or int64float_t is float64_t:
        out[:] = NAN
    else:
        out[:] = 0

    first_element_set = np.zeros((<object>counts).shape, dtype=np.uint8)
    if uses_mask:
        result_mask[:] = True

    with nogil:
        for i in range(N):
            lab = labels[i]
            if lab == -1:
                continue

            counts[lab] += 1
            val = values[i, 0]

            if uses_mask:
                isna_entry = mask[i, 0]
            else:
                isna_entry = _treat_as_na(val, False)

            if isna_entry:
                continue

            if not first_element_set[lab]:
                out[lab, 0] = out[lab, 1] = out[lab, 2] = out[lab, 3] = val
                first_element_set[lab] = True
                if uses_mask:
                    result_mask[lab] = False
            else:
                out[lab, 1] = max(out[lab, 1], val)
                out[lab, 2] = min(out[lab, 2], val)
                out[lab, 3] = val


@cython.boundscheck(False)
@cython.wraparound(False)
def group_quantile(
    ndarray[float64_t, ndim=2] out,
    ndarray[numeric_t, ndim=1] values,
    ndarray[intp_t] labels,
    const uint8_t[:] mask,
    const float64_t[:] qs,
    ndarray[int64_t] starts,
    ndarray[int64_t] ends,
    str interpolation,
    uint8_t[:, ::1] result_mask,
    bint is_datetimelike,
) -> None:
    """
    Calculate the quantile per group.

    Parameters
    ----------
    out : np.ndarray[np.float64, ndim=2]
        Array of aggregated values that will be written to.
    values : np.ndarray
        Array containing the values to apply the function against.
    labels : ndarray[np.intp]
        Array containing the unique group labels.
    qs : ndarray[float64_t]
        The quantile values to search for.
    starts : ndarray[int64]
        Positions at which each group begins.
    ends : ndarray[int64]
        Positions at which each group ends.
    interpolation : {'linear', 'lower', 'highest', 'nearest', 'midpoint'}
    result_mask : ndarray[bool, ndim=2] or None
    is_datetimelike : bool
        Whether int64 values represent datetime64-like values.

    Notes
    -----
    Rather than explicitly returning a value, this function modifies the
    provided `out` parameter.
    """
    cdef:
        Py_ssize_t i, N=len(labels), ngroups, non_na_sz, k, nqs
        Py_ssize_t idx=0
        Py_ssize_t grp_size
        InterpolationEnumType interp
        float64_t q_val, q_idx, frac, val, next_val
        bint uses_result_mask = result_mask is not None
        Py_ssize_t start, end
        ndarray[numeric_t] grp
        intp_t[::1] sort_indexer
        const uint8_t[:] sub_mask

    assert values.shape[0] == N
    assert starts is not None
    assert ends is not None
    assert len(starts) == len(ends)

    if any(not (0 <= q <= 1) for q in qs):
        wrong = [x for x in qs if not (0 <= x <= 1)][0]
        raise ValueError(
            f"Each 'q' must be between 0 and 1. Got '{wrong}' instead"
        )

    inter_methods = {
        "linear": INTERPOLATION_LINEAR,
        "lower": INTERPOLATION_LOWER,
        "higher": INTERPOLATION_HIGHER,
        "nearest": INTERPOLATION_NEAREST,
        "midpoint": INTERPOLATION_MIDPOINT,
    }
    interp = inter_methods[interpolation]

    nqs = len(qs)
    ngroups = len(out)

    # TODO: get cnp.PyArray_ArgSort to work with nogil so we can restore the rest
    #  of this function as being `with nogil:`
    for i in range(ngroups):
        start = starts[i]
        end = ends[i]

        grp = values[start:end]

        # Figure out how many group elements there are
        sub_mask = mask[start:end]
        grp_size = sub_mask.size
        non_na_sz = 0
        for k in range(grp_size):
            if sub_mask[k] == 0:
                non_na_sz += 1

        # equiv: sort_indexer = grp.argsort()
        if is_datetimelike:
            # We need the argsort to put NaTs at the end, not the beginning
            sort_indexer = cnp.PyArray_ArgSort(grp.view("M8[ns]"), 0, cnp.NPY_QUICKSORT)
        else:
            sort_indexer = cnp.PyArray_ArgSort(grp, 0, cnp.NPY_QUICKSORT)

        if non_na_sz == 0:
            for k in range(nqs):
                if uses_result_mask:
                    result_mask[i, k] = 1
                else:
                    out[i, k] = NaN
        else:
            for k in range(nqs):
                q_val = qs[k]

                # Calculate where to retrieve the desired value
                # Casting to int will intentionally truncate result
                idx = <int64_t>(q_val * <float64_t>(non_na_sz - 1))

                val = grp[sort_indexer[idx]]
                # If requested quantile falls evenly on a particular index
                # then write that index's value out. Otherwise interpolate
                q_idx = q_val * (non_na_sz - 1)
                frac = q_idx % 1

                if frac == 0.0 or interp == INTERPOLATION_LOWER:
                    out[i, k] = val
                else:
                    next_val = grp[sort_indexer[idx + 1]]
                    if interp == INTERPOLATION_LINEAR:
                        out[i, k] = val + (next_val - val) * frac
                    elif interp == INTERPOLATION_HIGHER:
                        out[i, k] = next_val
                    elif interp == INTERPOLATION_MIDPOINT:
                        out[i, k] = (val + next_val) / 2.0
                    elif interp == INTERPOLATION_NEAREST:
                        if frac > .5 or (frac == .5 and q_val > .5):  # Always OK?
                            out[i, k] = next_val
                        else:
                            out[i, k] = val


# ----------------------------------------------------------------------
# group_nth, group_last, group_rank
# ----------------------------------------------------------------------

ctypedef fused numeric_object_complex_t:
    numeric_object_t
    complex64_t
    complex128_t


cdef bint _treat_as_na(numeric_object_complex_t val,
                       bint is_datetimelike) noexcept nogil:
    if numeric_object_complex_t is object:
        # Should never be used, but we need to avoid the `val != val` below
        #  or else cython will raise about gil acquisition.
        raise NotImplementedError

    elif numeric_object_complex_t is int64_t:
        return is_datetimelike and val == NPY_NAT
    elif (
        numeric_object_complex_t is float32_t
        or numeric_object_complex_t is float64_t
        or numeric_object_complex_t is complex64_t
        or numeric_object_complex_t is complex128_t
    ):
        return val != val
    else:
        # non-datetimelike integer
        return False


cdef numeric_object_t _get_min_or_max(
    numeric_object_t val,
    bint compute_max,
    bint is_datetimelike,
):
    """
    Find either the min or the max supported by numeric_object_t; 'val' is a
    placeholder to effectively make numeric_object_t an argument.
    """
    return get_rank_nan_fill_val(
        not compute_max,
        val=val,
        is_datetimelike=is_datetimelike,
    )


cdef numeric_t _get_na_val(numeric_t val, bint is_datetimelike):
    cdef:
        numeric_t na_val

    if numeric_t == float32_t or numeric_t == float64_t:
        na_val = NaN
    elif numeric_t is int64_t and is_datetimelike:
        na_val = NPY_NAT
    else:
        # Used in case of masks
        na_val = 0
    return na_val


ctypedef fused mincount_t:
    numeric_t
    complex64_t
    complex128_t


@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline void _check_below_mincount(
    mincount_t[:, ::1] out,
    bint uses_mask,
    uint8_t[:, ::1] result_mask,
    Py_ssize_t ncounts,
    Py_ssize_t K,
    int64_t[:, ::1] nobs,
    int64_t min_count,
    mincount_t[:, ::1] resx,
) noexcept nogil:
    """
    Check if the number of observations for a group is below min_count,
    and if so set the result for that group to the appropriate NA-like value.
    """
    cdef:
        Py_ssize_t i, j

    for i in range(ncounts):
        for j in range(K):

            if nobs[i, j] < min_count:
                # if we are integer dtype, not is_datetimelike, and
                #  not uses_mask, then getting here implies that
                #  counts[i] < min_count, which means we will
                #  be cast to float64 and masked at the end
                #  of WrappedCythonOp._call_cython_op. So we can safely
                #  set a placeholder value in out[i, j].
                if uses_mask:
                    result_mask[i, j] = True
                    # set out[i, j] to 0 to be deterministic, as
                    #  it was initialized with np.empty. Also ensures
                    #  we can downcast out if appropriate.
                    out[i, j] = 0
                elif (
                    mincount_t is float32_t
                    or mincount_t is float64_t
                    or mincount_t is complex64_t
                    or mincount_t is complex128_t
                ):
                    out[i, j] = NAN
                elif mincount_t is int64_t:
                    # Per above, this is a placeholder in
                    #  non-is_datetimelike cases.
                    out[i, j] = NPY_NAT
                else:
                    # placeholder, see above
                    out[i, j] = 0
            else:
                out[i, j] = resx[i, j]


# TODO(cython3): GH#31710 use memorviews once cython 0.30 is released so we can
#  use `const numeric_object_t[:, :] values`
@cython.wraparound(False)
@cython.boundscheck(False)
def group_last(
    numeric_object_t[:, ::1] out,
    int64_t[::1] counts,
    ndarray[numeric_object_t, ndim=2] values,
    const intp_t[::1] labels,
    const uint8_t[:, :] mask,
    uint8_t[:, ::1] result_mask=None,
    Py_ssize_t min_count=-1,
    bint is_datetimelike=False,
) -> None:
    """
    Only aggregates on axis=0
    """
    cdef:
        Py_ssize_t i, j, N, K, lab, ncounts = len(counts)
        numeric_object_t val
        numeric_object_t[:, ::1] resx
        int64_t[:, ::1] nobs
        bint uses_mask = mask is not None
        bint isna_entry

    # TODO(cython3):
    # Instead of `labels.shape[0]` use `len(labels)`
    if not len(values) == labels.shape[0]:
        raise AssertionError("len(index) != len(labels)")

    min_count = max(min_count, 1)
    nobs = np.zeros((<object>out).shape, dtype=np.int64)
    if numeric_object_t is object:
        resx = np.empty((<object>out).shape, dtype=object)
    else:
        resx = np.empty_like(out)

    N, K = (<object>values).shape

    if numeric_object_t is object:
        # TODO(cython3): De-duplicate once conditional-nogil is available
        for i in range(N):
            lab = labels[i]
            if lab < 0:
                continue

            counts[lab] += 1
            for j in range(K):
                val = values[i, j]

                if uses_mask:
                    isna_entry = mask[i, j]
                else:
                    isna_entry = checknull(val)

                if not isna_entry:
                    # TODO(cython3): use _treat_as_na here once
                    #  conditional-nogil is available.
                    nobs[lab, j] += 1
                    resx[lab, j] = val

        for i in range(ncounts):
            for j in range(K):
                if nobs[i, j] < min_count:
                    out[i, j] = None
                else:
                    out[i, j] = resx[i, j]
    else:
        with nogil:
            for i in range(N):
                lab = labels[i]
                if lab < 0:
                    continue

                counts[lab] += 1
                for j in range(K):
                    val = values[i, j]

                    if uses_mask:
                        isna_entry = mask[i, j]
                    else:
                        isna_entry = _treat_as_na(val, is_datetimelike)

                    if not isna_entry:
                        nobs[lab, j] += 1
                        resx[lab, j] = val

            _check_below_mincount(
                out, uses_mask, result_mask, ncounts, K, nobs, min_count, resx
            )


# TODO(cython3): GH#31710 use memorviews once cython 0.30 is released so we can
#  use `const numeric_object_t[:, :] values`
@cython.wraparound(False)
@cython.boundscheck(False)
def group_nth(
    numeric_object_t[:, ::1] out,
    int64_t[::1] counts,
    ndarray[numeric_object_t, ndim=2] values,
    const intp_t[::1] labels,
    const uint8_t[:, :] mask,
    uint8_t[:, ::1] result_mask=None,
    int64_t min_count=-1,
    int64_t rank=1,
    bint is_datetimelike=False,
) -> None:
    """
    Only aggregates on axis=0
    """
    cdef:
        Py_ssize_t i, j, N, K, lab, ncounts = len(counts)
        numeric_object_t val
        numeric_object_t[:, ::1] resx
        int64_t[:, ::1] nobs
        bint uses_mask = mask is not None
        bint isna_entry

    # TODO(cython3):
    # Instead of `labels.shape[0]` use `len(labels)`
    if not len(values) == labels.shape[0]:
        raise AssertionError("len(index) != len(labels)")

    min_count = max(min_count, 1)
    nobs = np.zeros((<object>out).shape, dtype=np.int64)
    if numeric_object_t is object:
        resx = np.empty((<object>out).shape, dtype=object)
    else:
        resx = np.empty_like(out)

    N, K = (<object>values).shape

    if numeric_object_t is object:
        # TODO(cython3): De-duplicate once conditional-nogil is available
        for i in range(N):
            lab = labels[i]
            if lab < 0:
                continue

            counts[lab] += 1
            for j in range(K):
                val = values[i, j]

                if uses_mask:
                    isna_entry = mask[i, j]
                else:
                    isna_entry = checknull(val)

                if not isna_entry:
                    # TODO(cython3): use _treat_as_na here once
                    #  conditional-nogil is available.
                    nobs[lab, j] += 1
                    if nobs[lab, j] == rank:
                        resx[lab, j] = val

        for i in range(ncounts):
            for j in range(K):
                if nobs[i, j] < min_count:
                    out[i, j] = None
                else:
                    out[i, j] = resx[i, j]

    else:
        with nogil:
            for i in range(N):
                lab = labels[i]
                if lab < 0:
                    continue

                counts[lab] += 1
                for j in range(K):
                    val = values[i, j]

                    if uses_mask:
                        isna_entry = mask[i, j]
                    else:
                        isna_entry = _treat_as_na(val, is_datetimelike)

                    if not isna_entry:
                        nobs[lab, j] += 1
                        if nobs[lab, j] == rank:
                            resx[lab, j] = val

            _check_below_mincount(
                out, uses_mask, result_mask, ncounts, K, nobs, min_count, resx
            )


@cython.boundscheck(False)
@cython.wraparound(False)
def group_rank(
    float64_t[:, ::1] out,
    ndarray[numeric_object_t, ndim=2] values,
    const intp_t[::1] labels,
    int ngroups,
    bint is_datetimelike,
    str ties_method="average",
    bint ascending=True,
    bint pct=False,
    str na_option="keep",
    const uint8_t[:, :] mask=None,
) -> None:
    """
    Provides the rank of values within each group.

    Parameters
    ----------
    out : np.ndarray[np.float64, ndim=2]
        Values to which this method will write its results.
    values : np.ndarray of numeric_object_t values to be ranked
    labels : np.ndarray[np.intp]
        Array containing unique label for each group, with its ordering
        matching up to the corresponding record in `values`
    ngroups : int
        This parameter is not used, is needed to match signatures of other
        groupby functions.
    is_datetimelike : bool
        True if `values` contains datetime-like entries.
    ties_method : {'average', 'min', 'max', 'first', 'dense'}, default 'average'
        * average: average rank of group
        * min: lowest rank in group
        * max: highest rank in group
        * first: ranks assigned in order they appear in the array
        * dense: like 'min', but rank always increases by 1 between groups
    ascending : bool, default True
        False for ranks by high (1) to low (N)
        na_option : {'keep', 'top', 'bottom'}, default 'keep'
    pct : bool, default False
        Compute percentage rank of data within each group
    na_option : {'keep', 'top', 'bottom'}, default 'keep'
        * keep: leave NA values where they are
        * top: smallest rank if ascending
        * bottom: smallest rank if descending
    mask : np.ndarray[bool] or None, default None

    Notes
    -----
    This method modifies the `out` parameter rather than returning an object
    """
    cdef:
        Py_ssize_t i, k, N
        ndarray[float64_t, ndim=1] result
        const uint8_t[:] sub_mask

    N = values.shape[1]

    for k in range(N):
        if mask is None:
            sub_mask = None
        else:
            sub_mask = mask[:, k]

        result = rank_1d(
            values=values[:, k],
            labels=labels,
            is_datetimelike=is_datetimelike,
            ties_method=ties_method,
            ascending=ascending,
            pct=pct,
            na_option=na_option,
            mask=sub_mask,
        )
        for i in range(len(result)):
            if labels[i] >= 0:
                out[i, k] = result[i]


# ----------------------------------------------------------------------
# group_min, group_max
# ----------------------------------------------------------------------


@cython.wraparound(False)
@cython.boundscheck(False)
cdef group_min_max(
    numeric_t[:, ::1] out,
    int64_t[::1] counts,
    ndarray[numeric_t, ndim=2] values,
    const intp_t[::1] labels,
    Py_ssize_t min_count=-1,
    bint is_datetimelike=False,
    bint compute_max=True,
    const uint8_t[:, ::1] mask=None,
    uint8_t[:, ::1] result_mask=None,
):
    """
    Compute minimum/maximum  of columns of `values`, in row groups `labels`.

    Parameters
    ----------
    out : np.ndarray[numeric_t, ndim=2]
        Array to store result in.
    counts : np.ndarray[int64]
        Input as a zeroed array, populated by group sizes during algorithm
    values : array
        Values to find column-wise min/max of.
    labels : np.ndarray[np.intp]
        Labels to group by.
    min_count : Py_ssize_t, default -1
        The minimum number of non-NA group elements, NA result if threshold
        is not met
    is_datetimelike : bool
        True if `values` contains datetime-like entries.
    compute_max : bint, default True
        True to compute group-wise max, False to compute min
    mask : ndarray[bool, ndim=2], optional
        If not None, indices represent missing values,
        otherwise the mask will not be used
    result_mask : ndarray[bool, ndim=2], optional
        If not None, these specify locations in the output that are NA.
        Modified in-place.

    Notes
    -----
    This method modifies the `out` parameter, rather than returning an object.
    `counts` is modified to hold group sizes
    """
    cdef:
        Py_ssize_t i, j, N, K, lab, ngroups = len(counts)
        numeric_t val
        numeric_t[:, ::1] group_min_or_max
        int64_t[:, ::1] nobs
        bint uses_mask = mask is not None
        bint isna_entry

    # TODO(cython3):
    # Instead of `labels.shape[0]` use `len(labels)`
    if not len(values) == labels.shape[0]:
        raise AssertionError("len(index) != len(labels)")

    min_count = max(min_count, 1)
    nobs = np.zeros((<object>out).shape, dtype=np.int64)

    group_min_or_max = np.empty_like(out)
    group_min_or_max[:] = _get_min_or_max(<numeric_t>0, compute_max, is_datetimelike)

    N, K = (<object>values).shape

    with nogil:
        for i in range(N):
            lab = labels[i]
            if lab < 0:
                continue

            counts[lab] += 1
            for j in range(K):
                val = values[i, j]

                if uses_mask:
                    isna_entry = mask[i, j]
                else:
                    isna_entry = _treat_as_na(val, is_datetimelike)

                if not isna_entry:
                    nobs[lab, j] += 1
                    if compute_max:
                        if val > group_min_or_max[lab, j]:
                            group_min_or_max[lab, j] = val
                    else:
                        if val < group_min_or_max[lab, j]:
                            group_min_or_max[lab, j] = val

        _check_below_mincount(
            out, uses_mask, result_mask, ngroups, K, nobs, min_count, group_min_or_max
        )


@cython.wraparound(False)
@cython.boundscheck(False)
def group_max(
    numeric_t[:, ::1] out,
    int64_t[::1] counts,
    ndarray[numeric_t, ndim=2] values,
    const intp_t[::1] labels,
    Py_ssize_t min_count=-1,
    bint is_datetimelike=False,
    const uint8_t[:, ::1] mask=None,
    uint8_t[:, ::1] result_mask=None,
) -> None:
    """See group_min_max.__doc__"""
    group_min_max(
        out,
        counts,
        values,
        labels,
        min_count=min_count,
        is_datetimelike=is_datetimelike,
        compute_max=True,
        mask=mask,
        result_mask=result_mask,
    )


@cython.wraparound(False)
@cython.boundscheck(False)
def group_min(
    numeric_t[:, ::1] out,
    int64_t[::1] counts,
    ndarray[numeric_t, ndim=2] values,
    const intp_t[::1] labels,
    Py_ssize_t min_count=-1,
    bint is_datetimelike=False,
    const uint8_t[:, ::1] mask=None,
    uint8_t[:, ::1] result_mask=None,
) -> None:
    """See group_min_max.__doc__"""
    group_min_max(
        out,
        counts,
        values,
        labels,
        min_count=min_count,
        is_datetimelike=is_datetimelike,
        compute_max=False,
        mask=mask,
        result_mask=result_mask,
    )


@cython.boundscheck(False)
@cython.wraparound(False)
cdef group_cummin_max(
    numeric_t[:, ::1] out,
    ndarray[numeric_t, ndim=2] values,
    const uint8_t[:, ::1] mask,
    uint8_t[:, ::1] result_mask,
    const intp_t[::1] labels,
    int ngroups,
    bint is_datetimelike,
    bint skipna,
    bint compute_max,
):
    """
    Cumulative minimum/maximum of columns of `values`, in row groups `labels`.

    Parameters
    ----------
    out : np.ndarray[numeric_t, ndim=2]
        Array to store cummin/max in.
    values : np.ndarray[numeric_t, ndim=2]
        Values to take cummin/max of.
    mask : np.ndarray[bool] or None
        If not None, indices represent missing values,
        otherwise the mask will not be used
    result_mask : ndarray[bool, ndim=2], optional
        If not None, these specify locations in the output that are NA.
        Modified in-place.
    labels : np.ndarray[np.intp]
        Labels to group by.
    ngroups : int
        Number of groups, larger than all entries of `labels`.
    is_datetimelike : bool
        True if `values` contains datetime-like entries.
    skipna : bool
        If True, ignore nans in `values`.
    compute_max : bool
        True if cumulative maximum should be computed, False
        if cumulative minimum should be computed

    Notes
    -----
    This method modifies the `out` parameter, rather than returning an object.
    """
    cdef:
        numeric_t[:, ::1] accum
        Py_ssize_t i, j, N, K
        numeric_t val, mval, na_val
        uint8_t[:, ::1] seen_na
        intp_t lab
        bint na_possible
        bint uses_mask = mask is not None
        bint isna_entry

    accum = np.empty((ngroups, (<object>values).shape[1]), dtype=values.dtype)
    accum[:] = _get_min_or_max(<numeric_t>0, compute_max, is_datetimelike)

    na_val = _get_na_val(<numeric_t>0, is_datetimelike)

    if uses_mask:
        na_possible = True
        # Will never be used, just to avoid uninitialized warning
        na_val = 0
    elif numeric_t is float64_t or numeric_t is float32_t:
        na_possible = True
    elif is_datetimelike:
        na_possible = True
    else:
        # Will never be used, just to avoid uninitialized warning
        na_possible = False

    if na_possible:
        seen_na = np.zeros((<object>accum).shape, dtype=np.uint8)

    N, K = (<object>values).shape
    with nogil:
        for i in range(N):
            lab = labels[i]
            if lab < 0:
                continue
            for j in range(K):

                if not skipna and na_possible and seen_na[lab, j]:
                    if uses_mask:
                        result_mask[i, j] = 1
                        # Set to 0 ensures that we are deterministic and can
                        #  downcast if appropriate
                        out[i, j] = 0

                    else:
                        out[i, j] = na_val
                else:
                    val = values[i, j]

                    if uses_mask:
                        isna_entry = mask[i, j]
                    else:
                        isna_entry = _treat_as_na(val, is_datetimelike)

                    if not isna_entry:
                        mval = accum[lab, j]
                        if compute_max:
                            if val > mval:
                                accum[lab, j] = mval = val
                        else:
                            if val < mval:
                                accum[lab, j] = mval = val
                        out[i, j] = mval
                    else:
                        seen_na[lab, j] = 1
                        out[i, j] = val


@cython.boundscheck(False)
@cython.wraparound(False)
def group_cummin(
    numeric_t[:, ::1] out,
    ndarray[numeric_t, ndim=2] values,
    const intp_t[::1] labels,
    int ngroups,
    bint is_datetimelike,
    const uint8_t[:, ::1] mask=None,
    uint8_t[:, ::1] result_mask=None,
    bint skipna=True,
) -> None:
    """See group_cummin_max.__doc__"""
    group_cummin_max(
        out=out,
        values=values,
        mask=mask,
        result_mask=result_mask,
        labels=labels,
        ngroups=ngroups,
        is_datetimelike=is_datetimelike,
        skipna=skipna,
        compute_max=False,
    )


@cython.boundscheck(False)
@cython.wraparound(False)
def group_cummax(
    numeric_t[:, ::1] out,
    ndarray[numeric_t, ndim=2] values,
    const intp_t[::1] labels,
    int ngroups,
    bint is_datetimelike,
    const uint8_t[:, ::1] mask=None,
    uint8_t[:, ::1] result_mask=None,
    bint skipna=True,
) -> None:
    """See group_cummin_max.__doc__"""
    group_cummin_max(
        out=out,
        values=values,
        mask=mask,
        result_mask=result_mask,
        labels=labels,
        ngroups=ngroups,
        is_datetimelike=is_datetimelike,
        skipna=skipna,
        compute_max=True,
    )
