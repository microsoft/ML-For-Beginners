# -*- coding: utf-8 -*-
"""Anova k-sample comparison without and with trimming

Created on Sun Jun 09 23:51:34 2013

Author: Josef Perktold
"""

import numbers
import numpy as np

# the trimboth and trim_mean are taken from scipy.stats.stats
# and enhanced by axis


def trimboth(a, proportiontocut, axis=0):
    """
    Slices off a proportion of items from both ends of an array.

    Slices off the passed proportion of items from both ends of the passed
    array (i.e., with `proportiontocut` = 0.1, slices leftmost 10% **and**
    rightmost 10% of scores).  You must pre-sort the array if you want
    'proper' trimming.  Slices off less if proportion results in a
    non-integer slice index (i.e., conservatively slices off
    `proportiontocut`).

    Parameters
    ----------
    a : array_like
        Data to trim.
    proportiontocut : float or int
        Proportion of data to trim at each end.
    axis : int or None
        Axis along which the observations are trimmed. The default is to trim
        along axis=0. If axis is None then the array will be flattened before
        trimming.

    Returns
    -------
    out : array-like
        Trimmed version of array `a`.

    Examples
    --------
    >>> from scipy import stats
    >>> a = np.arange(20)
    >>> b = stats.trimboth(a, 0.1)
    >>> b.shape
    (16,)

    """
    a = np.asarray(a)
    if axis is None:
        a = a.ravel()
        axis = 0
    nobs = a.shape[axis]
    lowercut = int(proportiontocut * nobs)
    uppercut = nobs - lowercut
    if (lowercut >= uppercut):
        raise ValueError("Proportion too big.")

    sl = [slice(None)] * a.ndim
    sl[axis] = slice(lowercut, uppercut)
    return a[tuple(sl)]


def trim_mean(a, proportiontocut, axis=0):
    """
    Return mean of array after trimming observations from both tails.

    If `proportiontocut` = 0.1, slices off 'leftmost' and 'rightmost' 10% of
    scores. Slices off LESS if proportion results in a non-integer slice
    index (i.e., conservatively slices off `proportiontocut` ).

    Parameters
    ----------
    a : array_like
        Input array
    proportiontocut : float
        Fraction to cut off at each tail of the sorted observations.
    axis : int or None
        Axis along which the trimmed means are computed. The default is axis=0.
        If axis is None then the trimmed mean will be computed for the
        flattened array.

    Returns
    -------
    trim_mean : ndarray
        Mean of trimmed array.

    """
    newa = trimboth(np.sort(a, axis), proportiontocut, axis=axis)
    return np.mean(newa, axis=axis)


class TrimmedMean:
    """
    class for trimmed and winsorized one sample statistics

    axis is None, i.e. ravelling, is not supported

    Parameters
    ----------
    data : array-like
        The data, observations to analyze.
    fraction : float in (0, 0.5)
        The fraction of observations to trim at each tail.
        The number of observations trimmed at each tail is
        ``int(fraction * nobs)``
    is_sorted : boolean
        Indicator if data is already sorted. By default the data is sorted
        along ``axis``.
    axis : int
        The axis of reduce operations. By default axis=0, that is observations
        are along the zero dimension, i.e. rows if 2-dim.
    """

    def __init__(self, data, fraction, is_sorted=False, axis=0):
        self.data = np.asarray(data)
        # TODO: add pandas handling, maybe not if this stays internal

        self.axis = axis
        self.fraction = fraction
        self.nobs = nobs = self.data.shape[axis]
        self.lowercut = lowercut = int(fraction * nobs)
        self.uppercut = uppercut = nobs - lowercut
        if (lowercut >= uppercut):
            raise ValueError("Proportion too big.")
        self.nobs_reduced = nobs - 2 * lowercut

        self.sl = [slice(None)] * self.data.ndim
        self.sl[axis] = slice(self.lowercut, self.uppercut)
        # numpy requires now tuple for indexing, not list
        self.sl = tuple(self.sl)
        if not is_sorted:
            self.data_sorted = np.sort(self.data, axis=axis)
        else:
            self.data_sorted = self.data

        # this only works for axis=0
        self.lowerbound = np.take(self.data_sorted, lowercut, axis=axis)
        self.upperbound = np.take(self.data_sorted, uppercut - 1, axis=axis)
        # self.lowerbound = self.data_sorted[lowercut]
        # self.upperbound = self.data_sorted[uppercut - 1]

    @property
    def data_trimmed(self):
        """numpy array of trimmed and sorted data
        """
        # returns a view
        return self.data_sorted[self.sl]

    @property  # cache
    def data_winsorized(self):
        """winsorized data
        """
        lb = np.expand_dims(self.lowerbound, self.axis)
        ub = np.expand_dims(self.upperbound, self.axis)
        return np.clip(self.data_sorted, lb, ub)

    @property
    def mean_trimmed(self):
        """mean of trimmed data
        """
        return np.mean(self.data_sorted[tuple(self.sl)], self.axis)

    @property
    def mean_winsorized(self):
        """mean of winsorized data
        """
        return np.mean(self.data_winsorized, self.axis)

    @property
    def var_winsorized(self):
        """variance of winsorized data
        """
        # hardcoded ddof = 1
        return np.var(self.data_winsorized, ddof=1, axis=self.axis)

    @property
    def std_mean_trimmed(self):
        """standard error of trimmed mean
        """
        se = np.sqrt(self.var_winsorized / self.nobs_reduced)
        # trimming creates correlation across trimmed observations
        # trimming is based on order statistics of the data
        # wilcox 2012, p.61
        se *= np.sqrt(self.nobs / self.nobs_reduced)
        return se

    @property
    def std_mean_winsorized(self):
        """standard error of winsorized mean
        """
        # the following matches Wilcox, WRS2
        std_ = np.sqrt(self.var_winsorized / self.nobs)
        std_ *= (self.nobs - 1) / (self.nobs_reduced - 1)
        # old version
        # tm = self
        # formula from an old SAS manual page, simplified
        # std_ = np.sqrt(tm.var_winsorized / (tm.nobs_reduced - 1) *
        #               (tm.nobs - 1.) / tm.nobs)
        return std_

    def ttest_mean(self, value=0, transform='trimmed',
                   alternative='two-sided'):
        """
        One sample t-test for trimmed or Winsorized mean

        Parameters
        ----------
        value : float
            Value of the mean under the Null hypothesis
        transform : {'trimmed', 'winsorized'}
            Specified whether the mean test is based on trimmed or winsorized
            data.
        alternative : {'two-sided', 'larger', 'smaller'}


        Notes
        -----
        p-value is based on the approximate t-distribution of the test
        statistic. The approximation is valid if the underlying distribution
        is symmetric.
        """
        import statsmodels.stats.weightstats as smws
        df = self.nobs_reduced - 1
        if transform == 'trimmed':
            mean_ = self.mean_trimmed
            std_ = self.std_mean_trimmed
        elif transform == 'winsorized':
            mean_ = self.mean_winsorized
            std_ = self.std_mean_winsorized
        else:
            raise ValueError("transform can only be 'trimmed' or 'winsorized'")

        res = smws._tstat_generic(mean_, 0, std_,
                                  df, alternative=alternative, diff=value)
        return res + (df,)

    def reset_fraction(self, frac):
        """create a TrimmedMean instance with a new trimming fraction

        This reuses the sorted array from the current instance.
        """
        tm = TrimmedMean(self.data_sorted, frac, is_sorted=True,
                         axis=self.axis)
        tm.data = self.data
        # TODO: this will not work if there is processing of meta-information
        #       in __init__,
        #       for example storing a pandas DataFrame or Series index
        return tm


def scale_transform(data, center='median', transform='abs', trim_frac=0.2,
                    axis=0):
    """Transform data for variance comparison for Levene type tests

    Parameters
    ----------
    data : array_like
        Observations for the data.
    center : "median", "mean", "trimmed" or float
        Statistic used for centering observations. If a float, then this
        value is used to center. Default is median.
    transform : 'abs', 'square', 'identity' or a callable
        The transform for the centered data.
    trim_frac : float in [0, 0.5)
        Fraction of observations that are trimmed on each side of the sorted
        observations. This is only used if center is `trimmed`.
    axis : int
        Axis along which the data are transformed when centering.

    Returns
    -------
    res : ndarray
        transformed data in the same shape as the original data.

    """
    x = np.asarray(data)  # x is shorthand from earlier code

    if transform == 'abs':
        tfunc = np.abs
    elif transform == 'square':
        tfunc = lambda x: x * x  # noqa
    elif transform == 'identity':
        tfunc = lambda x: x  # noqa
    elif callable(transform):
        tfunc = transform
    else:
        raise ValueError('transform should be abs, square or exp')

    if center == 'median':
        res = tfunc(x - np.expand_dims(np.median(x, axis=axis), axis))
    elif center == 'mean':
        res = tfunc(x - np.expand_dims(np.mean(x, axis=axis), axis))
    elif center == 'trimmed':
        center = trim_mean(x, trim_frac, axis=axis)
        res = tfunc(x - np.expand_dims(center, axis))
    elif isinstance(center, numbers.Number):
        res = tfunc(x - center)
    else:
        raise ValueError('center should be median, mean or trimmed')

    return res
