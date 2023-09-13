cimport cython
from cython cimport Py_ssize_t
from libc.math cimport (
    fabs,
    sqrt,
)
from libc.stdlib cimport (
    free,
    malloc,
)
from libc.string cimport memmove

import numpy as np

cimport numpy as cnp
from numpy cimport (
    NPY_FLOAT64,
    NPY_INT8,
    NPY_INT16,
    NPY_INT32,
    NPY_INT64,
    NPY_OBJECT,
    NPY_UINT64,
    float32_t,
    float64_t,
    int8_t,
    int16_t,
    int32_t,
    int64_t,
    intp_t,
    ndarray,
    uint8_t,
    uint16_t,
    uint32_t,
    uint64_t,
)

cnp.import_array()

cimport pandas._libs.util as util
from pandas._libs.dtypes cimport (
    numeric_object_t,
    numeric_t,
)
from pandas._libs.khash cimport (
    kh_destroy_int64,
    kh_get_int64,
    kh_init_int64,
    kh_int64_t,
    kh_put_int64,
    kh_resize_int64,
    khiter_t,
)
from pandas._libs.missing cimport (
    checknull,
    isnaobj,
)
from pandas._libs.util cimport get_nat

cdef:
    float64_t FP_ERR = 1e-13
    float64_t NaN = <float64_t>np.nan
    int64_t NPY_NAT = get_nat()


tiebreakers = {
    "average": TIEBREAK_AVERAGE,
    "min": TIEBREAK_MIN,
    "max": TIEBREAK_MAX,
    "first": TIEBREAK_FIRST,
    "dense": TIEBREAK_DENSE,
}


cdef bint are_diff(object left, object right):
    try:
        return fabs(left - right) > FP_ERR
    except TypeError:
        return left != right


class Infinity:
    """
    Provide a positive Infinity comparison method for ranking.
    """
    def __lt__(self, other):
        return False

    def __le__(self, other):
        return isinstance(other, Infinity)

    def __eq__(self, other):
        return isinstance(other, Infinity)

    def __ne__(self, other):
        return not isinstance(other, Infinity)

    def __gt__(self, other):
        return (not isinstance(other, Infinity) and
                not checknull(other))

    def __ge__(self, other):
        return not checknull(other)


class NegInfinity:
    """
    Provide a negative Infinity comparison method for ranking.
    """
    def __lt__(self, other):
        return  (not isinstance(other, NegInfinity) and
                 not checknull(other))

    def __le__(self, other):
        return not checknull(other)

    def __eq__(self, other):
        return isinstance(other, NegInfinity)

    def __ne__(self, other):
        return not isinstance(other, NegInfinity)

    def __gt__(self, other):
        return False

    def __ge__(self, other):
        return isinstance(other, NegInfinity)


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef ndarray[int64_t, ndim=1] unique_deltas(const int64_t[:] arr):
    """
    Efficiently find the unique first-differences of the given array.

    Parameters
    ----------
    arr : ndarray[int64_t]

    Returns
    -------
    ndarray[int64_t]
        An ordered ndarray[int64_t]
    """
    cdef:
        Py_ssize_t i, n = len(arr)
        int64_t val
        khiter_t k
        kh_int64_t *table
        int ret = 0
        list uniques = []
        ndarray[int64_t, ndim=1] result

    table = kh_init_int64()
    kh_resize_int64(table, 10)
    for i in range(n - 1):
        val = arr[i + 1] - arr[i]
        k = kh_get_int64(table, val)
        if k == table.n_buckets:
            kh_put_int64(table, val, &ret)
            uniques.append(val)
    kh_destroy_int64(table)

    result = np.array(uniques, dtype=np.int64)
    result.sort()
    return result


@cython.wraparound(False)
@cython.boundscheck(False)
def is_lexsorted(list_of_arrays: list) -> bool:
    cdef:
        Py_ssize_t i
        Py_ssize_t n, nlevels
        int64_t k, cur, pre
        ndarray arr
        bint result = True

    nlevels = len(list_of_arrays)
    n = len(list_of_arrays[0])

    cdef int64_t **vecs = <int64_t**>malloc(nlevels * sizeof(int64_t*))
    for i in range(nlevels):
        arr = list_of_arrays[i]
        assert arr.dtype.name == "int64"
        vecs[i] = <int64_t*>cnp.PyArray_DATA(arr)

    # Assume uniqueness??
    with nogil:
        for i in range(1, n):
            for k in range(nlevels):
                cur = vecs[k][i]
                pre = vecs[k][i -1]
                if cur == pre:
                    continue
                elif cur > pre:
                    break
                else:
                    result = False
                    break
            if not result:
                break
    free(vecs)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def groupsort_indexer(const intp_t[:] index, Py_ssize_t ngroups):
    """
    Compute a 1-d indexer.

    The indexer is an ordering of the passed index,
    ordered by the groups.

    Parameters
    ----------
    index: np.ndarray[np.intp]
        Mappings from group -> position.
    ngroups: int64
        Number of groups.

    Returns
    -------
    ndarray[intp_t, ndim=1]
        Indexer
    ndarray[intp_t, ndim=1]
        Group Counts

    Notes
    -----
    This is a reverse of the label factorization process.
    """
    cdef:
        Py_ssize_t i, label, n
        intp_t[::1] indexer, where, counts

    counts = np.zeros(ngroups + 1, dtype=np.intp)
    n = len(index)
    indexer = np.zeros(n, dtype=np.intp)
    where = np.zeros(ngroups + 1, dtype=np.intp)

    with nogil:

        # count group sizes, location 0 for NA
        for i in range(n):
            counts[index[i] + 1] += 1

        # mark the start of each contiguous group of like-indexed data
        for i in range(1, ngroups + 1):
            where[i] = where[i - 1] + counts[i - 1]

        # this is our indexer
        for i in range(n):
            label = index[i] + 1
            indexer[where[label]] = i
            where[label] += 1

    return indexer.base, counts.base


cdef Py_ssize_t swap(numeric_t *a, numeric_t *b) noexcept nogil:
    cdef:
        numeric_t t

    # cython doesn't allow pointer dereference so use array syntax
    t = a[0]
    a[0] = b[0]
    b[0] = t
    return 0


cdef numeric_t kth_smallest_c(numeric_t* arr,
                              Py_ssize_t k, Py_ssize_t n) noexcept nogil:
    """
    See kth_smallest.__doc__. The additional parameter n specifies the maximum
    number of elements considered in arr, needed for compatibility with usage
    in groupby.pyx
    """
    cdef:
        Py_ssize_t i, j, left, m
        numeric_t x

    left = 0
    m = n - 1

    while left < m:
        x = arr[k]
        i = left
        j = m

        while 1:
            while arr[i] < x:
                i += 1
            while x < arr[j]:
                j -= 1
            if i <= j:
                swap(&arr[i], &arr[j])
                i += 1
                j -= 1

            if i > j:
                break

        if j < k:
            left = i
        if k < i:
            m = j
    return arr[k]


@cython.boundscheck(False)
@cython.wraparound(False)
def kth_smallest(numeric_t[::1] arr, Py_ssize_t k) -> numeric_t:
    """
    Compute the kth smallest value in arr. Note that the input
    array will be modified.

    Parameters
    ----------
    arr : numeric[::1]
        Array to compute the kth smallest value for, must be
        contiguous
    k : Py_ssize_t

    Returns
    -------
    numeric
        The kth smallest value in arr
    """
    cdef:
        numeric_t result

    with nogil:
        result = kth_smallest_c(&arr[0], k, arr.shape[0])

    return result


# ----------------------------------------------------------------------
# Pairwise correlation/covariance


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def nancorr(const float64_t[:, :] mat, bint cov=False, minp=None):
    cdef:
        Py_ssize_t i, xi, yi, N, K
        int64_t minpv
        float64_t[:, ::1] result
        ndarray[uint8_t, ndim=2] mask
        int64_t nobs = 0
        float64_t vx, vy, dx, dy, meanx, meany, divisor, ssqdmx, ssqdmy, covxy

    N, K = (<object>mat).shape

    if minp is None:
        minpv = 1
    else:
        minpv = <int64_t>minp

    result = np.empty((K, K), dtype=np.float64)
    mask = np.isfinite(mat).view(np.uint8)

    with nogil:
        for xi in range(K):
            for yi in range(xi + 1):
                # Welford's method for the variance-calculation
                # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
                nobs = ssqdmx = ssqdmy = covxy = meanx = meany = 0
                for i in range(N):
                    if mask[i, xi] and mask[i, yi]:
                        vx = mat[i, xi]
                        vy = mat[i, yi]
                        nobs += 1
                        dx = vx - meanx
                        dy = vy - meany
                        meanx += 1. / nobs * dx
                        meany += 1. / nobs * dy
                        ssqdmx += (vx - meanx) * dx
                        ssqdmy += (vy - meany) * dy
                        covxy += (vx - meanx) * dy

                if nobs < minpv:
                    result[xi, yi] = result[yi, xi] = NaN
                else:
                    divisor = (nobs - 1.0) if cov else sqrt(ssqdmx * ssqdmy)

                    if divisor != 0:
                        result[xi, yi] = result[yi, xi] = covxy / divisor
                    else:
                        result[xi, yi] = result[yi, xi] = NaN

    return result.base

# ----------------------------------------------------------------------
# Pairwise Spearman correlation


@cython.boundscheck(False)
@cython.wraparound(False)
def nancorr_spearman(ndarray[float64_t, ndim=2] mat, Py_ssize_t minp=1) -> ndarray:
    cdef:
        Py_ssize_t i, xi, yi, N, K
        ndarray[float64_t, ndim=2] result
        ndarray[float64_t, ndim=2] ranked_mat
        ndarray[float64_t, ndim=1] rankedx, rankedy
        float64_t[::1] maskedx, maskedy
        ndarray[uint8_t, ndim=2] mask
        int64_t nobs = 0
        bint no_nans
        float64_t vx, vy, sumx, sumxx, sumyy, mean, divisor

    N, K = (<object>mat).shape

    # Handle the edge case where we know all results will be nan
    # to keep conditional logic inside loop simpler
    if N < minp:
        result = np.full((K, K), np.nan, dtype=np.float64)
        return result

    result = np.empty((K, K), dtype=np.float64)
    mask = np.isfinite(mat).view(np.uint8)
    no_nans = mask.all()

    ranked_mat = np.empty((N, K), dtype=np.float64)

    # Note: we index into maskedx, maskedy in loops up to nobs, but using N is safe
    # here since N >= nobs and values are stored contiguously
    maskedx = np.empty(N, dtype=np.float64)
    maskedy = np.empty(N, dtype=np.float64)
    for i in range(K):
        ranked_mat[:, i] = rank_1d(mat[:, i])

    with nogil:
        for xi in range(K):
            for yi in range(xi + 1):
                sumx = sumxx = sumyy = 0

                # Fastpath for data with no nans/infs, allows avoiding mask checks
                # and array reassignments
                if no_nans:
                    mean = (N + 1) / 2.

                    # now the cov numerator
                    for i in range(N):
                        vx = ranked_mat[i, xi] - mean
                        vy = ranked_mat[i, yi] - mean

                        sumx += vx * vy
                        sumxx += vx * vx
                        sumyy += vy * vy
                else:
                    nobs = 0
                    # Keep track of whether we need to recompute ranks
                    all_ranks = True
                    for i in range(N):
                        all_ranks &= not (mask[i, xi] ^ mask[i, yi])
                        if mask[i, xi] and mask[i, yi]:
                            maskedx[nobs] = ranked_mat[i, xi]
                            maskedy[nobs] = ranked_mat[i, yi]
                            nobs += 1

                    if nobs < minp:
                        result[xi, yi] = result[yi, xi] = NaN
                        continue
                    else:
                        if not all_ranks:
                            with gil:
                                # We need to slice back to nobs because rank_1d will
                                # require arrays of nobs length
                                rankedx = rank_1d(np.asarray(maskedx)[:nobs])
                                rankedy = rank_1d(np.asarray(maskedy)[:nobs])
                            for i in range(nobs):
                                maskedx[i] = rankedx[i]
                                maskedy[i] = rankedy[i]

                        mean = (nobs + 1) / 2.

                        # now the cov numerator
                        for i in range(nobs):
                            vx = maskedx[i] - mean
                            vy = maskedy[i] - mean

                            sumx += vx * vy
                            sumxx += vx * vx
                            sumyy += vy * vy

                divisor = sqrt(sumxx * sumyy)

                if divisor != 0:
                    result[xi, yi] = result[yi, xi] = sumx / divisor
                else:
                    result[xi, yi] = result[yi, xi] = NaN

    return result


# ----------------------------------------------------------------------

def validate_limit(nobs: int | None, limit=None) -> int:
    """
    Check that the `limit` argument is a positive integer.

    Parameters
    ----------
    nobs : int
    limit : object

    Returns
    -------
    int
        The limit.
    """
    if limit is None:
        lim = nobs
    else:
        if not util.is_integer_object(limit):
            raise ValueError("Limit must be an integer")
        if limit < 1:
            raise ValueError("Limit must be greater than 0")
        lim = limit

    return lim


# TODO: overlap with libgroupby.group_fillna_indexer?
@cython.boundscheck(False)
@cython.wraparound(False)
def get_fill_indexer(const uint8_t[:] mask, limit=None):
    """
    Find an indexer to use for ffill to `take` on the array being filled.
    """
    cdef:
        ndarray[intp_t, ndim=1] indexer
        Py_ssize_t i, N = len(mask), last_valid
        int lim

        # fill_count is the number of consecutive NAs we have seen.
        #  If it exceeds the given limit, we stop padding.
        int fill_count = 0

    lim = validate_limit(N, limit)
    indexer = np.empty(N, dtype=np.intp)

    last_valid = -1  # haven't yet seen anything non-NA

    for i in range(N):
        if not mask[i]:
            indexer[i] = i
            last_valid = i
            fill_count = 0
        else:
            if fill_count < lim:
                indexer[i] = last_valid
            else:
                indexer[i] = -1
            fill_count += 1

    return indexer


@cython.boundscheck(False)
@cython.wraparound(False)
def pad(
    ndarray[numeric_object_t] old,
    ndarray[numeric_object_t] new,
    limit=None
) -> ndarray:
    # -> ndarray[intp_t, ndim=1]
    cdef:
        Py_ssize_t i, j, nleft, nright
        ndarray[intp_t, ndim=1] indexer
        numeric_object_t cur, next_val
        int lim, fill_count = 0

    nleft = len(old)
    nright = len(new)
    indexer = np.empty(nright, dtype=np.intp)
    indexer[:] = -1

    lim = validate_limit(nright, limit)

    if nleft == 0 or nright == 0 or new[nright - 1] < old[0]:
        return indexer

    i = j = 0

    cur = old[0]

    while j <= nright - 1 and new[j] < cur:
        j += 1

    while True:
        if j == nright:
            break

        if i == nleft - 1:
            while j < nright:
                if new[j] == cur:
                    indexer[j] = i
                elif new[j] > cur and fill_count < lim:
                    indexer[j] = i
                    fill_count += 1
                j += 1
            break

        next_val = old[i + 1]

        while j < nright and cur <= new[j] < next_val:
            if new[j] == cur:
                indexer[j] = i
            elif fill_count < lim:
                indexer[j] = i
                fill_count += 1
            j += 1

        fill_count = 0
        i += 1
        cur = next_val

    return indexer


@cython.boundscheck(False)
@cython.wraparound(False)
def pad_inplace(numeric_object_t[:] values, uint8_t[:] mask, limit=None):
    cdef:
        Py_ssize_t i, N
        numeric_object_t val
        uint8_t prev_mask
        int lim, fill_count = 0

    N = len(values)

    # GH#2778
    if N == 0:
        return

    lim = validate_limit(N, limit)

    val = values[0]
    prev_mask = mask[0]
    for i in range(N):
        if mask[i]:
            if fill_count >= lim:
                continue
            fill_count += 1
            values[i] = val
            mask[i] = prev_mask
        else:
            fill_count = 0
            val = values[i]
            prev_mask = mask[i]


@cython.boundscheck(False)
@cython.wraparound(False)
def pad_2d_inplace(numeric_object_t[:, :] values, uint8_t[:, :] mask, limit=None):
    cdef:
        Py_ssize_t i, j, N, K
        numeric_object_t val
        int lim, fill_count = 0

    K, N = (<object>values).shape

    # GH#2778
    if N == 0:
        return

    lim = validate_limit(N, limit)

    for j in range(K):
        fill_count = 0
        val = values[j, 0]
        for i in range(N):
            if mask[j, i]:
                if fill_count >= lim or i == 0:
                    continue
                fill_count += 1
                values[j, i] = val
                mask[j, i] = False
            else:
                fill_count = 0
                val = values[j, i]


@cython.boundscheck(False)
@cython.wraparound(False)
def backfill(
    ndarray[numeric_object_t] old,
    ndarray[numeric_object_t] new,
    limit=None
) -> ndarray:  # -> ndarray[intp_t, ndim=1]
    """
    Backfilling logic for generating fill vector

    Diagram of what's going on

    Old      New    Fill vector    Mask
            .        0               1
            .        0               1
            .        0               1
    A        A        0               1
            .        1               1
            .        1               1
            .        1               1
            .        1               1
            .        1               1
    B        B        1               1
            .        2               1
            .        2               1
            .        2               1
    C        C        2               1
            .                        0
            .                        0
    D
    """
    cdef:
        Py_ssize_t i, j, nleft, nright
        ndarray[intp_t, ndim=1] indexer
        numeric_object_t cur, prev
        int lim, fill_count = 0

    nleft = len(old)
    nright = len(new)
    indexer = np.empty(nright, dtype=np.intp)
    indexer[:] = -1

    lim = validate_limit(nright, limit)

    if nleft == 0 or nright == 0 or new[0] > old[nleft - 1]:
        return indexer

    i = nleft - 1
    j = nright - 1

    cur = old[nleft - 1]

    while j >= 0 and new[j] > cur:
        j -= 1

    while True:
        if j < 0:
            break

        if i == 0:
            while j >= 0:
                if new[j] == cur:
                    indexer[j] = i
                elif new[j] < cur and fill_count < lim:
                    indexer[j] = i
                    fill_count += 1
                j -= 1
            break

        prev = old[i - 1]

        while j >= 0 and prev < new[j] <= cur:
            if new[j] == cur:
                indexer[j] = i
            elif new[j] < cur and fill_count < lim:
                indexer[j] = i
                fill_count += 1
            j -= 1

        fill_count = 0
        i -= 1
        cur = prev

    return indexer


def backfill_inplace(numeric_object_t[:] values, uint8_t[:] mask, limit=None):
    pad_inplace(values[::-1], mask[::-1], limit=limit)


def backfill_2d_inplace(numeric_object_t[:, :] values,
                        uint8_t[:, :] mask,
                        limit=None):
    pad_2d_inplace(values[:, ::-1], mask[:, ::-1], limit)


@cython.boundscheck(False)
@cython.wraparound(False)
def is_monotonic(ndarray[numeric_object_t, ndim=1] arr, bint timelike):
    """
    Returns
    -------
    tuple
        is_monotonic_inc : bool
        is_monotonic_dec : bool
        is_strict_monotonic : bool
    """
    cdef:
        Py_ssize_t i, n
        numeric_object_t prev, cur
        bint is_monotonic_inc = 1
        bint is_monotonic_dec = 1
        bint is_unique = 1
        bint is_strict_monotonic = 1

    n = len(arr)

    if n == 1:
        if arr[0] != arr[0] or (numeric_object_t is int64_t and timelike and
                                arr[0] == NPY_NAT):
            # single value is NaN
            return False, False, True
        else:
            return True, True, True
    elif n < 2:
        return True, True, True

    if timelike and <int64_t>arr[0] == NPY_NAT:
        return False, False, True

    if numeric_object_t is not object:
        with nogil:
            prev = arr[0]
            for i in range(1, n):
                cur = arr[i]
                if timelike and <int64_t>cur == NPY_NAT:
                    is_monotonic_inc = 0
                    is_monotonic_dec = 0
                    break
                if cur < prev:
                    is_monotonic_inc = 0
                elif cur > prev:
                    is_monotonic_dec = 0
                elif cur == prev:
                    is_unique = 0
                else:
                    # cur or prev is NaN
                    is_monotonic_inc = 0
                    is_monotonic_dec = 0
                    break
                if not is_monotonic_inc and not is_monotonic_dec:
                    is_monotonic_inc = 0
                    is_monotonic_dec = 0
                    break
                prev = cur
    else:
        # object-dtype, identical to above except we cannot use `with nogil`
        prev = arr[0]
        for i in range(1, n):
            cur = arr[i]
            if timelike and <int64_t>cur == NPY_NAT:
                is_monotonic_inc = 0
                is_monotonic_dec = 0
                break
            if cur < prev:
                is_monotonic_inc = 0
            elif cur > prev:
                is_monotonic_dec = 0
            elif cur == prev:
                is_unique = 0
            else:
                # cur or prev is NaN
                is_monotonic_inc = 0
                is_monotonic_dec = 0
                break
            if not is_monotonic_inc and not is_monotonic_dec:
                is_monotonic_inc = 0
                is_monotonic_dec = 0
                break
            prev = cur

    is_strict_monotonic = is_unique and (is_monotonic_inc or is_monotonic_dec)
    return is_monotonic_inc, is_monotonic_dec, is_strict_monotonic


# ----------------------------------------------------------------------
# rank_1d, rank_2d
# ----------------------------------------------------------------------

cdef numeric_object_t get_rank_nan_fill_val(
    bint rank_nans_highest,
    numeric_object_t val,
    bint is_datetimelike=False,
):
    """
    Return the value we'll use to represent missing values when sorting depending
    on if we'd like missing values to end up at the top/bottom. (The second parameter
    is unused, but needed for fused type specialization)
    """
    if numeric_object_t is int64_t and is_datetimelike and not rank_nans_highest:
        return NPY_NAT + 1

    if rank_nans_highest:
        if numeric_object_t is object:
            return Infinity()
        elif numeric_object_t is int64_t:
            return util.INT64_MAX
        elif numeric_object_t is int32_t:
            return util.INT32_MAX
        elif numeric_object_t is int16_t:
            return util.INT16_MAX
        elif numeric_object_t is int8_t:
            return util.INT8_MAX
        elif numeric_object_t is uint64_t:
            return util.UINT64_MAX
        elif numeric_object_t is uint32_t:
            return util.UINT32_MAX
        elif numeric_object_t is uint16_t:
            return util.UINT16_MAX
        elif numeric_object_t is uint8_t:
            return util.UINT8_MAX
        else:
            return np.inf
    else:
        if numeric_object_t is object:
            return NegInfinity()
        elif numeric_object_t is int64_t:
            # Note(jbrockmendel) 2022-03-15 for reasons unknown, using util.INT64_MIN
            #  instead of NPY_NAT here causes build warnings and failure in
            #  test_cummax_i8_at_implementation_bound
            return NPY_NAT
        elif numeric_object_t is int32_t:
            return util.INT32_MIN
        elif numeric_object_t is int16_t:
            return util.INT16_MIN
        elif numeric_object_t is int8_t:
            return util.INT8_MIN
        elif numeric_object_t is uint64_t:
            return 0
        elif numeric_object_t is uint32_t:
            return 0
        elif numeric_object_t is uint16_t:
            return 0
        elif numeric_object_t is uint8_t:
            return 0
        else:
            return -np.inf


@cython.wraparound(False)
@cython.boundscheck(False)
def rank_1d(
    ndarray[numeric_object_t, ndim=1] values,
    const intp_t[:] labels=None,
    bint is_datetimelike=False,
    ties_method="average",
    bint ascending=True,
    bint pct=False,
    na_option="keep",
    const uint8_t[:] mask=None,
):
    """
    Fast NaN-friendly version of ``scipy.stats.rankdata``.

    Parameters
    ----------
    values : array of numeric_object_t values to be ranked
    labels : np.ndarray[np.intp] or None
        Array containing unique label for each group, with its ordering
        matching up to the corresponding record in `values`. If not called
        from a groupby operation, will be None.
    is_datetimelike : bool, default False
        True if `values` contains datetime-like entries.
    ties_method : {'average', 'min', 'max', 'first', 'dense'}, default
        'average'
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
    mask : np.ndarray[bool], optional, default None
        Specify locations to be treated as NA, for e.g. Categorical.
    """
    cdef:
        TiebreakEnumType tiebreak
        Py_ssize_t N
        int64_t[::1] grp_sizes
        intp_t[:] lexsort_indexer
        float64_t[::1] out
        ndarray[numeric_object_t, ndim=1] masked_vals
        numeric_object_t[:] masked_vals_memview
        bint keep_na, nans_rank_highest, check_labels, check_mask
        numeric_object_t nan_fill_val

    tiebreak = tiebreakers[ties_method]
    if tiebreak == TIEBREAK_FIRST:
        if not ascending:
            tiebreak = TIEBREAK_FIRST_DESCENDING

    keep_na = na_option == "keep"

    N = len(values)
    if labels is not None:
        # TODO(cython3): cast won't be necessary (#2992)
        assert <Py_ssize_t>len(labels) == N
    out = np.empty(N)
    grp_sizes = np.ones(N, dtype=np.int64)

    # If we don't care about labels, can short-circuit later label
    # comparisons
    check_labels = labels is not None

    # For cases where a mask is not possible, we can avoid mask checks
    check_mask = (
        numeric_object_t is float32_t
        or numeric_object_t is float64_t
        or numeric_object_t is object
        or (numeric_object_t is int64_t and is_datetimelike)
    )
    check_mask = check_mask or mask is not None

    # Copy values into new array in order to fill missing data
    # with mask, without obfuscating location of missing data
    # in values array
    if numeric_object_t is object and values.dtype != np.object_:
        masked_vals = values.astype("O")
    else:
        masked_vals = values.copy()

    if mask is not None:
        pass
    elif numeric_object_t is object:
        mask = isnaobj(masked_vals)
    elif numeric_object_t is int64_t and is_datetimelike:
        mask = (masked_vals == NPY_NAT).astype(np.uint8)
    elif numeric_object_t is float64_t or numeric_object_t is float32_t:
        mask = np.isnan(masked_vals).astype(np.uint8)
    else:
        mask = np.zeros(shape=len(masked_vals), dtype=np.uint8)

    # If `na_option == 'top'`, we want to assign the lowest rank
    # to NaN regardless of ascending/descending. So if ascending,
    # fill with lowest value of type to end up with lowest rank.
    # If descending, fill with highest value since descending
    # will flip the ordering to still end up with lowest rank.
    # Symmetric logic applies to `na_option == 'bottom'`
    nans_rank_highest = ascending ^ (na_option == "top")
    nan_fill_val = get_rank_nan_fill_val(nans_rank_highest, <numeric_object_t>0)
    if nans_rank_highest:
        order = [masked_vals, mask]
    else:
        order = [masked_vals, ~(np.asarray(mask))]

    if check_labels:
        order.append(labels)

    np.putmask(masked_vals, mask, nan_fill_val)
    # putmask doesn't accept a memoryview, so we assign as a separate step
    masked_vals_memview = masked_vals

    # lexsort using labels, then mask, then actual values
    # each label corresponds to a different group value,
    # the mask helps you differentiate missing values before
    # performing sort on the actual values
    lexsort_indexer = np.lexsort(order).astype(np.intp, copy=False)

    if not ascending:
        lexsort_indexer = lexsort_indexer[::-1]

    with nogil:
        rank_sorted_1d(
            out,
            grp_sizes,
            lexsort_indexer,
            masked_vals_memview,
            mask,
            check_mask=check_mask,
            N=N,
            tiebreak=tiebreak,
            keep_na=keep_na,
            pct=pct,
            labels=labels,
        )

    return np.asarray(out)


@cython.wraparound(False)
@cython.boundscheck(False)
cdef void rank_sorted_1d(
    float64_t[::1] out,
    int64_t[::1] grp_sizes,
    const intp_t[:] sort_indexer,
    # TODO(cython3): make const (https://github.com/cython/cython/issues/3222)
    numeric_object_t[:] masked_vals,
    const uint8_t[:] mask,
    bint check_mask,
    Py_ssize_t N,
    TiebreakEnumType tiebreak=TIEBREAK_AVERAGE,
    bint keep_na=True,
    bint pct=False,
    # https://github.com/cython/cython/issues/1630, only trailing arguments can
    # currently be omitted for cdef functions, which is why we keep this at the end
    const intp_t[:] labels=None,
) noexcept nogil:
    """
    See rank_1d.__doc__. Handles only actual ranking, so sorting and masking should
    be handled in the caller. Note that `out` and `grp_sizes` are modified inplace.

    Parameters
    ----------
    out : float64_t[::1]
        Array to store computed ranks
    grp_sizes : int64_t[::1]
        Array to store group counts, only used if pct=True. Should only be None
        if labels is None.
    sort_indexer : intp_t[:]
        Array of indices which sorts masked_vals
    masked_vals : numeric_object_t[:]
        The values input to rank_1d, with missing values replaced by fill values
    mask : uint8_t[:]
        Array where entries are True if the value is missing, False otherwise.
    check_mask : bool
        If False, assumes the mask is all False to skip mask indexing
    N : Py_ssize_t
        The number of elements to rank. Note: it is not always true that
        N == len(out) or N == len(masked_vals) (see `nancorr_spearman` usage for why)
    tiebreak : TiebreakEnumType, default TIEBREAK_AVERAGE
        See rank_1d.__doc__ for the different modes
    keep_na : bool, default True
        Whether or not to keep nulls
    pct : bool, default False
        Compute percentage rank of data within each group
    labels : See rank_1d.__doc__, default None. None implies all labels are the same.
    """

    cdef:
        Py_ssize_t i, j, dups=0, sum_ranks=0,
        Py_ssize_t grp_start=0, grp_vals_seen=1, grp_na_count=0
        bint at_end, next_val_diff, group_changed, check_labels
        int64_t grp_size

    check_labels = labels is not None

    # Loop over the length of the value array
    # each incremental i value can be looked up in the lexsort_indexer
    # array that we sorted previously, which gives us the location of
    # that sorted value for retrieval back from the original
    # values / masked_vals arrays
    # TODO(cython3): de-duplicate once cython supports conditional nogil
    if numeric_object_t is object:
        with gil:
            for i in range(N):
                at_end = i == N - 1

                # dups and sum_ranks will be incremented each loop where
                # the value / group remains the same, and should be reset
                # when either of those change. Used to calculate tiebreakers
                dups += 1
                sum_ranks += i - grp_start + 1

                next_val_diff = at_end or are_diff(masked_vals[sort_indexer[i]],
                                                   masked_vals[sort_indexer[i+1]])

                # We'll need this check later anyway to determine group size, so just
                # compute it here since shortcircuiting won't help
                group_changed = at_end or (check_labels and
                                           (labels[sort_indexer[i]]
                                            != labels[sort_indexer[i+1]]))

                # Update out only when there is a transition of values or labels.
                # When a new value or group is encountered, go back #dups steps(
                # the number of occurrence of current value) and assign the ranks
                # based on the starting index of the current group (grp_start)
                # and the current index
                if (next_val_diff or group_changed or (check_mask and
                                                       (mask[sort_indexer[i]]
                                                        ^ mask[sort_indexer[i+1]]))):

                    # If keep_na, check for missing values and assign back
                    # to the result where appropriate
                    if keep_na and check_mask and mask[sort_indexer[i]]:
                        grp_na_count = dups
                        for j in range(i - dups + 1, i + 1):
                            out[sort_indexer[j]] = NaN
                    elif tiebreak == TIEBREAK_AVERAGE:
                        for j in range(i - dups + 1, i + 1):
                            out[sort_indexer[j]] = sum_ranks / <float64_t>dups
                    elif tiebreak == TIEBREAK_MIN:
                        for j in range(i - dups + 1, i + 1):
                            out[sort_indexer[j]] = i - grp_start - dups + 2
                    elif tiebreak == TIEBREAK_MAX:
                        for j in range(i - dups + 1, i + 1):
                            out[sort_indexer[j]] = i - grp_start + 1

                    # With n as the previous rank in the group and m as the number
                    # of duplicates in this stretch, if TIEBREAK_FIRST and ascending,
                    # then rankings should be n + 1, n + 2 ... n + m
                    elif tiebreak == TIEBREAK_FIRST:
                        for j in range(i - dups + 1, i + 1):
                            out[sort_indexer[j]] = j + 1 - grp_start

                    # If TIEBREAK_FIRST and descending, the ranking should be
                    # n + m, n + (m - 1) ... n + 1. This is equivalent to
                    # (i - dups + 1) + (i - j + 1) - grp_start
                    elif tiebreak == TIEBREAK_FIRST_DESCENDING:
                        for j in range(i - dups + 1, i + 1):
                            out[sort_indexer[j]] = 2 * i - j - dups + 2 - grp_start
                    elif tiebreak == TIEBREAK_DENSE:
                        for j in range(i - dups + 1, i + 1):
                            out[sort_indexer[j]] = grp_vals_seen

                    # Look forward to the next value (using the sorting in
                    # lexsort_indexer). If the value does not equal the current
                    # value then we need to reset the dups and sum_ranks, knowing
                    # that a new value is coming up. The conditional also needs
                    # to handle nan equality and the end of iteration. If group
                    # changes we do not record seeing a new value in the group
                    if not group_changed and (next_val_diff or (check_mask and
                                              (mask[sort_indexer[i]]
                                               ^ mask[sort_indexer[i+1]]))):
                        dups = sum_ranks = 0
                        grp_vals_seen += 1

                    # Similar to the previous conditional, check now if we are
                    # moving to a new group. If so, keep track of the index where
                    # the new group occurs, so the tiebreaker calculations can
                    # decrement that from their position. Fill in the size of each
                    # group encountered (used by pct calculations later). Also be
                    # sure to reset any of the items helping to calculate dups
                    if group_changed:

                        # If not dense tiebreak, group size used to compute
                        # percentile will be # of non-null elements in group
                        if tiebreak != TIEBREAK_DENSE:
                            grp_size = i - grp_start + 1 - grp_na_count

                        # Otherwise, it will be the number of distinct values
                        # in the group, subtracting 1 if NaNs are present
                        # since that is a distinct value we shouldn't count
                        else:
                            grp_size = grp_vals_seen - (grp_na_count > 0)

                        for j in range(grp_start, i + 1):
                            grp_sizes[sort_indexer[j]] = grp_size

                        dups = sum_ranks = 0
                        grp_na_count = 0
                        grp_start = i + 1
                        grp_vals_seen = 1
    else:
        for i in range(N):
            at_end = i == N - 1

            # dups and sum_ranks will be incremented each loop where
            # the value / group remains the same, and should be reset
            # when either of those change. Used to calculate tiebreakers
            dups += 1
            sum_ranks += i - grp_start + 1

            next_val_diff = at_end or (masked_vals[sort_indexer[i]]
                                       != masked_vals[sort_indexer[i+1]])

            # We'll need this check later anyway to determine group size, so just
            # compute it here since shortcircuiting won't help
            group_changed = at_end or (check_labels and
                                       (labels[sort_indexer[i]]
                                        != labels[sort_indexer[i+1]]))

            # Update out only when there is a transition of values or labels.
            # When a new value or group is encountered, go back #dups steps(
            # the number of occurrence of current value) and assign the ranks
            # based on the starting index of the current group (grp_start)
            # and the current index
            if (next_val_diff or group_changed
                or (check_mask and
                    (mask[sort_indexer[i]] ^ mask[sort_indexer[i+1]]))):

                # If keep_na, check for missing values and assign back
                # to the result where appropriate
                if keep_na and check_mask and mask[sort_indexer[i]]:
                    grp_na_count = dups
                    for j in range(i - dups + 1, i + 1):
                        out[sort_indexer[j]] = NaN
                elif tiebreak == TIEBREAK_AVERAGE:
                    for j in range(i - dups + 1, i + 1):
                        out[sort_indexer[j]] = sum_ranks / <float64_t>dups
                elif tiebreak == TIEBREAK_MIN:
                    for j in range(i - dups + 1, i + 1):
                        out[sort_indexer[j]] = i - grp_start - dups + 2
                elif tiebreak == TIEBREAK_MAX:
                    for j in range(i - dups + 1, i + 1):
                        out[sort_indexer[j]] = i - grp_start + 1

                # With n as the previous rank in the group and m as the number
                # of duplicates in this stretch, if TIEBREAK_FIRST and ascending,
                # then rankings should be n + 1, n + 2 ... n + m
                elif tiebreak == TIEBREAK_FIRST:
                    for j in range(i - dups + 1, i + 1):
                        out[sort_indexer[j]] = j + 1 - grp_start

                # If TIEBREAK_FIRST and descending, the ranking should be
                # n + m, n + (m - 1) ... n + 1. This is equivalent to
                # (i - dups + 1) + (i - j + 1) - grp_start
                elif tiebreak == TIEBREAK_FIRST_DESCENDING:
                    for j in range(i - dups + 1, i + 1):
                        out[sort_indexer[j]] = 2 * i - j - dups + 2 - grp_start
                elif tiebreak == TIEBREAK_DENSE:
                    for j in range(i - dups + 1, i + 1):
                        out[sort_indexer[j]] = grp_vals_seen

                # Look forward to the next value (using the sorting in
                # lexsort_indexer). If the value does not equal the current
                # value then we need to reset the dups and sum_ranks, knowing
                # that a new value is coming up. The conditional also needs
                # to handle nan equality and the end of iteration. If group
                # changes we do not record seeing a new value in the group
                if not group_changed and (next_val_diff
                                          or (check_mask and
                                              (mask[sort_indexer[i]]
                                               ^ mask[sort_indexer[i+1]]))):
                    dups = sum_ranks = 0
                    grp_vals_seen += 1

                # Similar to the previous conditional, check now if we are
                # moving to a new group. If so, keep track of the index where
                # the new group occurs, so the tiebreaker calculations can
                # decrement that from their position. Fill in the size of each
                # group encountered (used by pct calculations later). Also be
                # sure to reset any of the items helping to calculate dups
                if group_changed:

                    # If not dense tiebreak, group size used to compute
                    # percentile will be # of non-null elements in group
                    if tiebreak != TIEBREAK_DENSE:
                        grp_size = i - grp_start + 1 - grp_na_count

                    # Otherwise, it will be the number of distinct values
                    # in the group, subtracting 1 if NaNs are present
                    # since that is a distinct value we shouldn't count
                    else:
                        grp_size = grp_vals_seen - (grp_na_count > 0)

                    for j in range(grp_start, i + 1):
                        grp_sizes[sort_indexer[j]] = grp_size

                    dups = sum_ranks = 0
                    grp_na_count = 0
                    grp_start = i + 1
                    grp_vals_seen = 1

    if pct:
        for i in range(N):
            if grp_sizes[i] != 0:
                out[i] = out[i] / grp_sizes[i]


def rank_2d(
    ndarray[numeric_object_t, ndim=2] in_arr,
    int axis=0,
    bint is_datetimelike=False,
    ties_method="average",
    bint ascending=True,
    na_option="keep",
    bint pct=False,
):
    """
    Fast NaN-friendly version of ``scipy.stats.rankdata``.
    """
    cdef:
        Py_ssize_t k, n, col
        float64_t[::1, :] out  # Column-major so columns are contiguous
        int64_t[::1] grp_sizes
        ndarray[numeric_object_t, ndim=2] values
        numeric_object_t[:, :] masked_vals
        intp_t[:, :] sort_indexer
        uint8_t[:, :] mask
        TiebreakEnumType tiebreak
        bint check_mask, keep_na, nans_rank_highest
        numeric_object_t nan_fill_val

    tiebreak = tiebreakers[ties_method]
    if tiebreak == TIEBREAK_FIRST:
        if not ascending:
            tiebreak = TIEBREAK_FIRST_DESCENDING

    keep_na = na_option == "keep"

    # For cases where a mask is not possible, we can avoid mask checks
    check_mask = (
        numeric_object_t is float32_t
        or numeric_object_t is float64_t
        or numeric_object_t is object
        or (numeric_object_t is int64_t and is_datetimelike)
    )

    if axis == 1:
        values = np.asarray(in_arr).T.copy()
    else:
        values = np.asarray(in_arr).copy()

    if numeric_object_t is object:
        if values.dtype != np.object_:
            values = values.astype("O")

    nans_rank_highest = ascending ^ (na_option == "top")
    if check_mask:
        nan_fill_val = get_rank_nan_fill_val(nans_rank_highest, <numeric_object_t>0)

        if numeric_object_t is object:
            mask = isnaobj(values).view(np.uint8)
        elif numeric_object_t is float64_t or numeric_object_t is float32_t:
            mask = np.isnan(values).view(np.uint8)
        else:
            # i.e. int64 and datetimelike
            mask = (values == NPY_NAT).view(np.uint8)
        np.putmask(values, mask, nan_fill_val)
    else:
        mask = np.zeros_like(values, dtype=np.uint8)

    if nans_rank_highest:
        order = (values, mask)
    else:
        order = (values, ~np.asarray(mask))

    n, k = (<object>values).shape
    out = np.empty((n, k), dtype="f8", order="F")
    grp_sizes = np.ones(n, dtype=np.int64)

    # lexsort is slower, so only use if we need to worry about the mask
    if check_mask:
        sort_indexer = np.lexsort(order, axis=0).astype(np.intp, copy=False)
    else:
        kind = "stable" if ties_method == "first" else None
        sort_indexer = values.argsort(axis=0, kind=kind).astype(np.intp, copy=False)

    if not ascending:
        sort_indexer = sort_indexer[::-1, :]

    # putmask doesn't accept a memoryview, so we assign in a separate step
    masked_vals = values
    with nogil:
        for col in range(k):
            rank_sorted_1d(
                out[:, col],
                grp_sizes,
                sort_indexer[:, col],
                masked_vals[:, col],
                mask[:, col],
                check_mask=check_mask,
                N=n,
                tiebreak=tiebreak,
                keep_na=keep_na,
                pct=pct,
            )

    if axis == 1:
        return np.asarray(out.T)
    else:
        return np.asarray(out)


ctypedef fused diff_t:
    float64_t
    float32_t
    int8_t
    int16_t
    int32_t
    int64_t

ctypedef fused out_t:
    float32_t
    float64_t
    int64_t


@cython.boundscheck(False)
@cython.wraparound(False)
def diff_2d(
    ndarray[diff_t, ndim=2] arr,  # TODO(cython3) update to "const diff_t[:, :] arr"
    ndarray[out_t, ndim=2] out,
    Py_ssize_t periods,
    int axis,
    bint datetimelike=False,
):
    cdef:
        Py_ssize_t i, j, sx, sy, start, stop
        bint f_contig = arr.flags.f_contiguous
        # bint f_contig = arr.is_f_contig()  # TODO(cython3)
        diff_t left, right

    # Disable for unsupported dtype combinations,
    #  see https://github.com/cython/cython/issues/2646
    if (out_t is float32_t
            and not (diff_t is float32_t or diff_t is int8_t or diff_t is int16_t)):
        raise NotImplementedError  # pragma: no cover
    elif (out_t is float64_t
          and (diff_t is float32_t or diff_t is int8_t or diff_t is int16_t)):
        raise NotImplementedError  # pragma: no cover
    elif out_t is int64_t and diff_t is not int64_t:
        # We only have out_t of int64_t if we have datetimelike
        raise NotImplementedError  # pragma: no cover
    else:
        # We put this inside an indented else block to avoid cython build
        #  warnings about unreachable code
        sx, sy = (<object>arr).shape
        with nogil:
            if f_contig:
                if axis == 0:
                    if periods >= 0:
                        start, stop = periods, sx
                    else:
                        start, stop = 0, sx + periods
                    for j in range(sy):
                        for i in range(start, stop):
                            left = arr[i, j]
                            right = arr[i - periods, j]
                            if out_t is int64_t and datetimelike:
                                if left == NPY_NAT or right == NPY_NAT:
                                    out[i, j] = NPY_NAT
                                else:
                                    out[i, j] = left - right
                            else:
                                out[i, j] = left - right
                else:
                    if periods >= 0:
                        start, stop = periods, sy
                    else:
                        start, stop = 0, sy + periods
                    for j in range(start, stop):
                        for i in range(sx):
                            left = arr[i, j]
                            right = arr[i, j - periods]
                            if out_t is int64_t and datetimelike:
                                if left == NPY_NAT or right == NPY_NAT:
                                    out[i, j] = NPY_NAT
                                else:
                                    out[i, j] = left - right
                            else:
                                out[i, j] = left - right
            else:
                if axis == 0:
                    if periods >= 0:
                        start, stop = periods, sx
                    else:
                        start, stop = 0, sx + periods
                    for i in range(start, stop):
                        for j in range(sy):
                            left = arr[i, j]
                            right = arr[i - periods, j]
                            if out_t is int64_t and datetimelike:
                                if left == NPY_NAT or right == NPY_NAT:
                                    out[i, j] = NPY_NAT
                                else:
                                    out[i, j] = left - right
                            else:
                                out[i, j] = left - right
                else:
                    if periods >= 0:
                        start, stop = periods, sy
                    else:
                        start, stop = 0, sy + periods
                    for i in range(sx):
                        for j in range(start, stop):
                            left = arr[i, j]
                            right = arr[i, j - periods]
                            if out_t is int64_t and datetimelike:
                                if left == NPY_NAT or right == NPY_NAT:
                                    out[i, j] = NPY_NAT
                                else:
                                    out[i, j] = left - right
                            else:
                                out[i, j] = left - right


# generated from template
include "algos_common_helper.pxi"
include "algos_take_helper.pxi"
