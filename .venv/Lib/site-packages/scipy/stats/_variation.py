import numpy as np
from numpy.core.multiarray import normalize_axis_index
from scipy._lib._util import _nan_allsame, _contains_nan
from ._stats_py import _chk_asarray


def _nanvariation(a, *, axis=0, ddof=0, keepdims=False):
    """
    Private version of `variation` that ignores nan.

    `a` must be a numpy array.
    `axis` is assumed to be normalized, i.e. 0 <= axis < a.ndim.
    """
    #
    # In theory, this should be as simple as something like
    #     nanstd(a, ddof=ddof, axis=axis, keepdims=keepdims) /
    #     nanmean(a, axis=axis, keepdims=keepdims)
    # In practice, annoying issues arise.  Specifically, numpy
    # generates warnings in certain edge cases that we don't want
    # to propagate to the user.  Unfortunately, there does not
    # appear to be a thread-safe way to filter out the warnings,
    # so we have to do the calculation in a way that doesn't
    # generate numpy warnings.
    #
    # Let N be the number of non-nan inputs in a slice.
    # Conditions that generate nan:
    #   * empty input (i.e. N = 0)
    #   * All non-nan values 0
    #   * N < ddof
    #   * N == ddof and the input is constant
    # Conditions that generate inf:
    #   * non-constant input and either
    #       * the mean is 0, or
    #       * N == ddof
    #
    a_isnan = np.isnan(a)
    all_nan = a_isnan.all(axis=axis, keepdims=True)
    all_nan_full = np.broadcast_to(all_nan, a.shape)
    all_zero = (a_isnan | (a == 0)).all(axis=axis, keepdims=True) & ~all_nan

    # ngood is the number of non-nan values in each slice.
    ngood = (a.shape[axis] -
             np.expand_dims(np.count_nonzero(a_isnan, axis=axis), axis))
    # The return value is nan where ddof > ngood.
    ddof_too_big = ddof > ngood
    # If ddof == ngood, the return value is nan where the input is constant and
    # inf otherwise.
    ddof_equal_n = ddof == ngood

    is_const = _nan_allsame(a, axis=axis, keepdims=True)

    a2 = a.copy()
    # If an entire slice is nan, `np.nanmean` will generate a warning,
    # so we replace those nan's with 1.0 before computing the mean.
    # We'll fix the corresponding output later.
    a2[all_nan_full] = 1.0
    mean_a = np.nanmean(a2, axis=axis, keepdims=True)

    # If ddof >= ngood (the number of non-nan values in the slice), `np.nanstd`
    # will generate a warning, so set all the values in such a slice to 1.0.
    # We'll fix the corresponding output later.
    a2[np.broadcast_to(ddof_too_big, a2.shape) | ddof_equal_n] = 1.0
    with np.errstate(invalid='ignore'):
        std_a = np.nanstd(a2, axis=axis, ddof=ddof, keepdims=True)
    del a2

    sum_zero = np.nansum(a, axis=axis, keepdims=True) == 0

    # Where the sum along the axis is 0, replace mean_a with 1.  This avoids
    # division by zero.  We'll fix the corresponding output later.
    mean_a[sum_zero] = 1.0

    # Here--finally!--is the calculation of the variation.
    result = std_a / mean_a

    # Now fix the values that were given fake data to avoid warnings.
    result[~is_const & sum_zero] = np.inf
    signed_inf_mask = ~is_const & ddof_equal_n
    result[signed_inf_mask] = np.sign(mean_a[signed_inf_mask]) * np.inf
    nan_mask = all_zero | all_nan | ddof_too_big | (ddof_equal_n & is_const)
    result[nan_mask] = np.nan

    if not keepdims:
        result = np.squeeze(result, axis=axis)
        if result.shape == ():
            result = result[()]

    return result


def variation(a, axis=0, nan_policy='propagate', ddof=0, *, keepdims=False):
    """
    Compute the coefficient of variation.

    The coefficient of variation is the standard deviation divided by the
    mean.  This function is equivalent to::

        np.std(x, axis=axis, ddof=ddof) / np.mean(x)

    The default for ``ddof`` is 0, but many definitions of the coefficient
    of variation use the square root of the unbiased sample variance
    for the sample standard deviation, which corresponds to ``ddof=1``.

    The function does not take the absolute value of the mean of the data,
    so the return value is negative if the mean is negative.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int or None, optional
        Axis along which to calculate the coefficient of variation.
        Default is 0. If None, compute over the whole array `a`.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains ``nan``.
        The following options are available:

          * 'propagate': return ``nan``
          * 'raise': raise an exception
          * 'omit': perform the calculation with ``nan`` values omitted

        The default is 'propagate'.
    ddof : int, optional
        Gives the "Delta Degrees Of Freedom" used when computing the
        standard deviation.  The divisor used in the calculation of the
        standard deviation is ``N - ddof``, where ``N`` is the number of
        elements.  `ddof` must be less than ``N``; if it isn't, the result
        will be ``nan`` or ``inf``, depending on ``N`` and the values in
        the array.  By default `ddof` is zero for backwards compatibility,
        but it is recommended to use ``ddof=1`` to ensure that the sample
        standard deviation is computed as the square root of the unbiased
        sample variance.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the
        result as dimensions with size one. With this option, the result
        will broadcast correctly against the input array.

    Returns
    -------
    variation : ndarray
        The calculated variation along the requested axis.

    Notes
    -----
    There are several edge cases that are handled without generating a
    warning:

    * If both the mean and the standard deviation are zero, ``nan``
      is returned.
    * If the mean is zero and the standard deviation is nonzero, ``inf``
      is returned.
    * If the input has length zero (either because the array has zero
      length, or all the input values are ``nan`` and ``nan_policy`` is
      ``'omit'``), ``nan`` is returned.
    * If the input contains ``inf``, ``nan`` is returned.

    References
    ----------
    .. [1] Zwillinger, D. and Kokoska, S. (2000). CRC Standard
       Probability and Statistics Tables and Formulae. Chapman & Hall: New
       York. 2000.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import variation
    >>> variation([1, 2, 3, 4, 5], ddof=1)
    0.5270462766947299

    Compute the variation along a given dimension of an array that contains
    a few ``nan`` values:

    >>> x = np.array([[  10.0, np.nan, 11.0, 19.0, 23.0, 29.0, 98.0],
    ...               [  29.0,   30.0, 32.0, 33.0, 35.0, 56.0, 57.0],
    ...               [np.nan, np.nan, 12.0, 13.0, 16.0, 16.0, 17.0]])
    >>> variation(x, axis=1, ddof=1, nan_policy='omit')
    array([1.05109361, 0.31428986, 0.146483  ])

    """
    a, axis = _chk_asarray(a, axis)
    axis = normalize_axis_index(axis, ndim=a.ndim)
    n = a.shape[axis]

    contains_nan, nan_policy = _contains_nan(a, nan_policy)
    if contains_nan and nan_policy == 'omit':
        return _nanvariation(a, axis=axis, ddof=ddof, keepdims=keepdims)

    if a.size == 0 or ddof > n:
        # Handle as a special case to avoid spurious warnings.
        # The return values, if any, are all nan.
        shp = list(a.shape)
        if keepdims:
            shp[axis] = 1
        else:
            del shp[axis]
        if len(shp) == 0:
            result = np.nan
        else:
            result = np.full(shp, fill_value=np.nan)

        return result

    mean_a = a.mean(axis, keepdims=True)

    if ddof == n:
        # Another special case.  Result is either inf or nan.
        std_a = a.std(axis=axis, ddof=0, keepdims=True)
        result = np.full_like(std_a, fill_value=np.nan)
        result.flat[std_a.flat > 0] = (np.sign(mean_a) * np.inf).flat
        if result.shape == ():
            result = result[()]
        return result

    with np.errstate(divide='ignore', invalid='ignore'):
        std_a = a.std(axis, ddof=ddof, keepdims=True)
        result = std_a / mean_a

    if not keepdims:
        result = np.squeeze(result, axis=axis)
        if result.shape == ():
            result = result[()]

    return result
