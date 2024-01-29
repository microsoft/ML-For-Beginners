import numpy as np
from scipy._lib._util import _get_nan
from ._axis_nan_policy import _axis_nan_policy_factory


@_axis_nan_policy_factory(
    lambda x: x, n_outputs=1, result_to_tuple=lambda x: (x,)
)
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
    # `nan_policy` and `keepdims` are handled by `_axis_nan_policy`
    n = a.shape[axis]
    NaN = _get_nan(a)

    if a.size == 0 or ddof > n:
        # Handle as a special case to avoid spurious warnings.
        # The return values, if any, are all nan.
        shp = np.asarray(a.shape)
        shp = np.delete(shp, axis)
        result = np.full(shp, fill_value=NaN)
        return result[()]

    mean_a = a.mean(axis)

    if ddof == n:
        # Another special case.  Result is either inf or nan.
        std_a = a.std(axis=axis, ddof=0)
        result = np.full_like(std_a, fill_value=NaN)
        i = std_a > 0
        result[i] = np.inf
        result[i] = np.copysign(result[i], mean_a[i])
        return result[()]

    with np.errstate(divide='ignore', invalid='ignore'):
        std_a = a.std(axis, ddof=ddof)
        result = std_a / mean_a

    return result[()]
