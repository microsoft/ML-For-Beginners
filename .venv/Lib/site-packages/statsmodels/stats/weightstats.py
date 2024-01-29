"""Tests and descriptive statistics with weights


Created on 2010-09-18

Author: josef-pktd
License: BSD (3-clause)


References
----------
SPSS manual
SAS manual

This follows in large parts the SPSS manual, which is largely the same as
the SAS manual with different, simpler notation.

Freq, Weight in SAS seems redundant since they always show up as product, SPSS
has only weights.

Notes
-----

This has potential problems with ddof, I started to follow numpy with ddof=0
by default and users can change it, but this might still mess up the t-tests,
since the estimates for the standard deviation will be based on the ddof that
the user chooses.
- fixed ddof for the meandiff ttest, now matches scipy.stats.ttest_ind

Note: scipy has now a separate, pooled variance option in ttest, but I have not
compared yet.

"""

import numpy as np
from scipy import stats

from statsmodels.tools.decorators import cache_readonly


class DescrStatsW:
    """
    Descriptive statistics and tests with weights for case weights

    Assumes that the data is 1d or 2d with (nobs, nvars) observations in rows,
    variables in columns, and that the same weight applies to each column.

    If degrees of freedom correction is used, then weights should add up to the
    number of observations. ttest also assumes that the sum of weights
    corresponds to the sample size.

    This is essentially the same as replicating each observations by its
    weight, if the weights are integers, often called case or frequency weights.

    Parameters
    ----------
    data : array_like, 1-D or 2-D
        dataset
    weights : None or 1-D ndarray
        weights for each observation, with same length as zero axis of data
    ddof : int
        default ddof=0, degrees of freedom correction used for second moments,
        var, std, cov, corrcoef.
        However, statistical tests are independent of `ddof`, based on the
        standard formulas.

    Examples
    --------

    >>> import numpy as np
    >>> np.random.seed(0)
    >>> x1_2d = 1.0 + np.random.randn(20, 3)
    >>> w1 = np.random.randint(1, 4, 20)
    >>> d1 = DescrStatsW(x1_2d, weights=w1)
    >>> d1.mean
    array([ 1.42739844,  1.23174284,  1.083753  ])
    >>> d1.var
    array([ 0.94855633,  0.52074626,  1.12309325])
    >>> d1.std_mean
    array([ 0.14682676,  0.10878944,  0.15976497])

    >>> tstat, pval, df = d1.ttest_mean(0)
    >>> tstat; pval; df
    array([  9.72165021,  11.32226471,   6.78342055])
    array([  1.58414212e-12,   1.26536887e-14,   2.37623126e-08])
    44.0

    >>> tstat, pval, df = d1.ttest_mean([0, 1, 1])
    >>> tstat; pval; df
    array([ 9.72165021,  2.13019609,  0.52422632])
    array([  1.58414212e-12,   3.87842808e-02,   6.02752170e-01])
    44.0

    # if weights are integers, then asrepeats can be used

    >>> x1r = d1.asrepeats()
    >>> x1r.shape
    ...
    >>> stats.ttest_1samp(x1r, [0, 1, 1])
    ...

    """

    def __init__(self, data, weights=None, ddof=0):

        self.data = np.asarray(data)
        if weights is None:
            self.weights = np.ones(self.data.shape[0])
        else:
            self.weights = np.asarray(weights).astype(float)
            # TODO: why squeeze?
            if len(self.weights.shape) > 1 and len(self.weights) > 1:
                self.weights = self.weights.squeeze()
        self.ddof = ddof

    @cache_readonly
    def sum_weights(self):
        """Sum of weights"""
        return self.weights.sum(0)

    @cache_readonly
    def nobs(self):
        """alias for number of observations/cases, equal to sum of weights
        """
        return self.sum_weights

    @cache_readonly
    def sum(self):
        """weighted sum of data"""
        return np.dot(self.data.T, self.weights)

    @cache_readonly
    def mean(self):
        """weighted mean of data"""
        return self.sum / self.sum_weights

    @cache_readonly
    def demeaned(self):
        """data with weighted mean subtracted"""
        return self.data - self.mean

    @cache_readonly
    def sumsquares(self):
        """weighted sum of squares of demeaned data"""
        return np.dot((self.demeaned ** 2).T, self.weights)

    # need memoize instead of cache decorator
    def var_ddof(self, ddof=0):
        """variance of data given ddof

        Parameters
        ----------
        ddof : int, float
            degrees of freedom correction, independent of attribute ddof

        Returns
        -------
        var : float, ndarray
            variance with denominator ``sum_weights - ddof``
        """
        return self.sumsquares / (self.sum_weights - ddof)

    def std_ddof(self, ddof=0):
        """standard deviation of data with given ddof

        Parameters
        ----------
        ddof : int, float
            degrees of freedom correction, independent of attribute ddof

        Returns
        -------
        std : float, ndarray
            standard deviation with denominator ``sum_weights - ddof``
        """
        return np.sqrt(self.var_ddof(ddof=ddof))

    @cache_readonly
    def var(self):
        """variance with default degrees of freedom correction
        """
        return self.sumsquares / (self.sum_weights - self.ddof)

    @cache_readonly
    def _var(self):
        """variance without degrees of freedom correction

        used for statistical tests with controlled ddof
        """
        return self.sumsquares / self.sum_weights

    @cache_readonly
    def std(self):
        """standard deviation with default degrees of freedom correction
        """
        return np.sqrt(self.var)

    @cache_readonly
    def cov(self):
        """weighted covariance of data if data is 2 dimensional

        assumes variables in columns and observations in rows
        uses default ddof
        """
        cov_ = np.dot(self.weights * self.demeaned.T, self.demeaned)
        cov_ /= self.sum_weights - self.ddof
        return cov_

    @cache_readonly
    def corrcoef(self):
        """weighted correlation with default ddof

        assumes variables in columns and observations in rows
        """
        return self.cov / self.std / self.std[:, None]

    @cache_readonly
    def std_mean(self):
        """standard deviation of weighted mean
        """
        std = self.std
        if self.ddof != 0:
            # ddof correction,   (need copy of std)
            std = std * np.sqrt(
                (self.sum_weights - self.ddof) / self.sum_weights
            )

        return std / np.sqrt(self.sum_weights - 1)

    def quantile(self, probs, return_pandas=True):
        """
        Compute quantiles for a weighted sample.

        Parameters
        ----------
        probs : array_like
            A vector of probability points at which to calculate the
            quantiles.  Each element of `probs` should fall in [0, 1].
        return_pandas : bool
            If True, return value is a Pandas DataFrame or Series.
            Otherwise returns a ndarray.

        Returns
        -------
        quantiles : Series, DataFrame, or ndarray
            If `return_pandas` = True, returns one of the following:
              * data are 1d, `return_pandas` = True: a Series indexed by
                the probability points.
              * data are 2d, `return_pandas` = True: a DataFrame with
                the probability points as row index and the variables
                as column index.

            If `return_pandas` = False, returns an ndarray containing the
            same values as the Series/DataFrame.

        Notes
        -----
        To compute the quantiles, first, the weights are summed over
        exact ties yielding distinct data values y_1 < y_2 < ..., and
        corresponding weights w_1, w_2, ....  Let s_j denote the sum
        of the first j weights, and let W denote the sum of all the
        weights.  For a probability point p, if pW falls strictly
        between s_j and s_{j+1} then the estimated quantile is
        y_{j+1}.  If pW = s_j then the estimated quantile is (y_j +
        y_{j+1})/2.  If pW < p_1 then the estimated quantile is y_1.

        References
        ----------
        SAS documentation for weighted quantiles:

        https://support.sas.com/documentation/cdl/en/procstat/63104/HTML/default/viewer.htm#procstat_univariate_sect028.htm
        """

        import pandas as pd

        probs = np.asarray(probs)
        probs = np.atleast_1d(probs)

        if self.data.ndim == 1:
            rslt = self._quantile(self.data, probs)
            if return_pandas:
                rslt = pd.Series(rslt, index=probs)
        else:
            rslt = []
            for vec in self.data.T:
                rslt.append(self._quantile(vec, probs))
            rslt = np.column_stack(rslt)
            if return_pandas:
                columns = ["col%d" % (j + 1) for j in range(rslt.shape[1])]
                rslt = pd.DataFrame(data=rslt, columns=columns, index=probs)

        if return_pandas:
            rslt.index.name = "p"

        return rslt

    def _quantile(self, vec, probs):
        # Helper function to calculate weighted quantiles for one column.
        # Follows definition from SAS documentation.
        # Returns ndarray

        import pandas as pd

        # Aggregate over ties
        df = pd.DataFrame(index=np.arange(len(self.weights)))
        df["weights"] = self.weights
        df["vec"] = vec
        dfg = df.groupby("vec").agg("sum")
        weights = dfg.values[:, 0]
        values = np.asarray(dfg.index)

        cweights = np.cumsum(weights)
        totwt = cweights[-1]
        targets = probs * totwt
        ii = np.searchsorted(cweights, targets)

        rslt = values[ii]

        # Exact hits
        jj = np.flatnonzero(np.abs(targets - cweights[ii]) < 1e-10)
        jj = jj[ii[jj] < len(cweights) - 1]
        rslt[jj] = (values[ii[jj]] + values[ii[jj] + 1]) / 2

        return rslt

    def tconfint_mean(self, alpha=0.05, alternative="two-sided"):
        """two-sided confidence interval for weighted mean of data

        If the data is 2d, then these are separate confidence intervals
        for each column.

        Parameters
        ----------
        alpha : float
            significance level for the confidence interval, coverage is
            ``1-alpha``
        alternative : str
            This specifies the alternative hypothesis for the test that
            corresponds to the confidence interval.
            The alternative hypothesis, H1, has to be one of the following

              'two-sided': H1: mean not equal to value (default)
              'larger' :   H1: mean larger than value
              'smaller' :  H1: mean smaller than value

        Returns
        -------
        lower, upper : floats or ndarrays
            lower and upper bound of confidence interval

        Notes
        -----
        In a previous version, statsmodels 0.4, alpha was the confidence
        level, e.g. 0.95
        """
        # TODO: add asymmetric
        dof = self.sum_weights - 1
        ci = _tconfint_generic(
            self.mean, self.std_mean, dof, alpha, alternative
        )
        return ci

    def zconfint_mean(self, alpha=0.05, alternative="two-sided"):
        """two-sided confidence interval for weighted mean of data

        Confidence interval is based on normal distribution.
        If the data is 2d, then these are separate confidence intervals
        for each column.

        Parameters
        ----------
        alpha : float
            significance level for the confidence interval, coverage is
            ``1-alpha``
        alternative : str
            This specifies the alternative hypothesis for the test that
            corresponds to the confidence interval.
            The alternative hypothesis, H1, has to be one of the following

              'two-sided': H1: mean not equal to value (default)
              'larger' :   H1: mean larger than value
              'smaller' :  H1: mean smaller than value

        Returns
        -------
        lower, upper : floats or ndarrays
            lower and upper bound of confidence interval

        Notes
        -----
        In a previous version, statsmodels 0.4, alpha was the confidence
        level, e.g. 0.95
        """

        return _zconfint_generic(self.mean, self.std_mean, alpha, alternative)

    def ttest_mean(self, value=0, alternative="two-sided"):
        """ttest of Null hypothesis that mean is equal to value.

        The alternative hypothesis H1 is defined by the following

        - 'two-sided': H1: mean not equal to value
        - 'larger' :   H1: mean larger than value
        - 'smaller' :  H1: mean smaller than value

        Parameters
        ----------
        value : float or array
            the hypothesized value for the mean
        alternative : str
            The alternative hypothesis, H1, has to be one of the following:

              - 'two-sided': H1: mean not equal to value (default)
              - 'larger' :   H1: mean larger than value
              - 'smaller' :  H1: mean smaller than value

        Returns
        -------
        tstat : float
            test statistic
        pvalue : float
            pvalue of the t-test
        df : int or float

        """
        # TODO: check direction with R, smaller=less, larger=greater
        tstat = (self.mean - value) / self.std_mean
        dof = self.sum_weights - 1
        # TODO: use outsourced
        if alternative == "two-sided":
            pvalue = stats.t.sf(np.abs(tstat), dof) * 2
        elif alternative == "larger":
            pvalue = stats.t.sf(tstat, dof)
        elif alternative == "smaller":
            pvalue = stats.t.cdf(tstat, dof)
        else:
            raise ValueError("alternative not recognized")

        return tstat, pvalue, dof

    def ttost_mean(self, low, upp):
        """test of (non-)equivalence of one sample

        TOST: two one-sided t tests

        null hypothesis:  m < low or m > upp
        alternative hypothesis:  low < m < upp

        where m is the expected value of the sample (mean of the population).

        If the pvalue is smaller than a threshold, say 0.05, then we reject the
        hypothesis that the expected value of the sample (mean of the
        population) is outside of the interval given by thresholds low and upp.

        Parameters
        ----------
        low, upp : float
            equivalence interval low < mean < upp

        Returns
        -------
        pvalue : float
            pvalue of the non-equivalence test
        t1, pv1, df1 : tuple
            test statistic, pvalue and degrees of freedom for lower threshold
            test
        t2, pv2, df2 : tuple
            test statistic, pvalue and degrees of freedom for upper threshold
            test

        """

        t1, pv1, df1 = self.ttest_mean(low, alternative="larger")
        t2, pv2, df2 = self.ttest_mean(upp, alternative="smaller")
        return np.maximum(pv1, pv2), (t1, pv1, df1), (t2, pv2, df2)

    def ztest_mean(self, value=0, alternative="two-sided"):
        """z-test of Null hypothesis that mean is equal to value.

        The alternative hypothesis H1 is defined by the following
        'two-sided': H1: mean not equal to value
        'larger' :   H1: mean larger than value
        'smaller' :  H1: mean smaller than value

        Parameters
        ----------
        value : float or array
            the hypothesized value for the mean
        alternative : str
            The alternative hypothesis, H1, has to be one of the following

              'two-sided': H1: mean not equal to value (default)
              'larger' :   H1: mean larger than value
              'smaller' :  H1: mean smaller than value

        Returns
        -------
        tstat : float
            test statistic
        pvalue : float
            pvalue of the t-test

        Notes
        -----
        This uses the same degrees of freedom correction as the t-test in the
        calculation of the standard error of the mean, i.e it uses
        `(sum_weights - 1)` instead of `sum_weights` in the denominator.
        See Examples below for the difference.

        Examples
        --------

        z-test on a proportion, with 20 observations, 15 of those are our event

        >>> import statsmodels.api as sm
        >>> x1 = [0, 1]
        >>> w1 = [5, 15]
        >>> d1 = sm.stats.DescrStatsW(x1, w1)
        >>> d1.ztest_mean(0.5)
        (2.5166114784235836, 0.011848940928347452)

        This differs from the proportions_ztest because of the degrees of
        freedom correction:
        >>> sm.stats.proportions_ztest(15, 20.0, value=0.5)
        (2.5819888974716112, 0.009823274507519247).

        We can replicate the results from ``proportions_ztest`` if we increase
        the weights to have artificially one more observation:

        >>> sm.stats.DescrStatsW(x1, np.array(w1)*21./20).ztest_mean(0.5)
        (2.5819888974716116, 0.0098232745075192366)
        """
        tstat = (self.mean - value) / self.std_mean
        # TODO: use outsourced
        if alternative == "two-sided":
            pvalue = stats.norm.sf(np.abs(tstat)) * 2
        elif alternative == "larger":
            pvalue = stats.norm.sf(tstat)
        elif alternative == "smaller":
            pvalue = stats.norm.cdf(tstat)

        return tstat, pvalue

    def ztost_mean(self, low, upp):
        """test of (non-)equivalence of one sample, based on z-test

        TOST: two one-sided z-tests

        null hypothesis:  m < low or m > upp
        alternative hypothesis:  low < m < upp

        where m is the expected value of the sample (mean of the population).

        If the pvalue is smaller than a threshold, say 0.05, then we reject the
        hypothesis that the expected value of the sample (mean of the
        population) is outside of the interval given by thresholds low and upp.

        Parameters
        ----------
        low, upp : float
            equivalence interval low < mean < upp

        Returns
        -------
        pvalue : float
            pvalue of the non-equivalence test
        t1, pv1 : tuple
            test statistic and p-value for lower threshold test
        t2, pv2 : tuple
            test statistic and p-value for upper threshold test

        """

        t1, pv1 = self.ztest_mean(low, alternative="larger")
        t2, pv2 = self.ztest_mean(upp, alternative="smaller")
        return np.maximum(pv1, pv2), (t1, pv1), (t2, pv2)

    def get_compare(self, other, weights=None):
        """return an instance of CompareMeans with self and other

        Parameters
        ----------
        other : array_like or instance of DescrStatsW
            If array_like then this creates an instance of DescrStatsW with
            the given weights.
        weights : None or array
            weights are only used if other is not an instance of DescrStatsW

        Returns
        -------
        cm : instance of CompareMeans
            the instance has self attached as d1 and other as d2.

        See Also
        --------
        CompareMeans

        """
        if not isinstance(other, self.__class__):
            d2 = DescrStatsW(other, weights)
        else:
            d2 = other
        return CompareMeans(self, d2)

    def asrepeats(self):
        """get array that has repeats given by floor(weights)

        observations with weight=0 are dropped

        """
        w_int = np.floor(self.weights).astype(int)
        return np.repeat(self.data, w_int, axis=0)


def _tstat_generic(value1, value2, std_diff, dof, alternative, diff=0):
    """generic ttest based on summary statistic

    The test statistic is :
        tstat = (value1 - value2 - diff) / std_diff

    and is assumed to be t-distributed with ``dof`` degrees of freedom.

    Parameters
    ----------
    value1 : float or ndarray
        Value, for example mean, of the first sample.
    value2 : float or ndarray
        Value, for example mean, of the second sample.
    std_diff : float or ndarray
        Standard error of the difference value1 - value2
    dof : int or float
        Degrees of freedom
    alternative : str
        The alternative hypothesis, H1, has to be one of the following

           * 'two-sided' : H1: ``value1 - value2 - diff`` not equal to 0.
           * 'larger' :   H1: ``value1 - value2 - diff > 0``
           * 'smaller' :  H1: ``value1 - value2 - diff < 0``

    diff : float
        value of difference ``value1 - value2`` under the null hypothesis

    Returns
    -------
    tstat : float or ndarray
        Test statistic.
    pvalue : float or ndarray
        P-value of the hypothesis test assuming that the test statistic is
        t-distributed with ``df`` degrees of freedom.
    """

    tstat = (value1 - value2 - diff) / std_diff
    if alternative in ["two-sided", "2-sided", "2s"]:
        pvalue = stats.t.sf(np.abs(tstat), dof) * 2
    elif alternative in ["larger", "l"]:
        pvalue = stats.t.sf(tstat, dof)
    elif alternative in ["smaller", "s"]:
        pvalue = stats.t.cdf(tstat, dof)
    else:
        raise ValueError("invalid alternative")
    return tstat, pvalue


def _tconfint_generic(mean, std_mean, dof, alpha, alternative):
    """generic t-confint based on summary statistic

    Parameters
    ----------
    mean : float or ndarray
        Value, for example mean, of the first sample.
    std_mean : float or ndarray
        Standard error of the difference value1 - value2
    dof : int or float
        Degrees of freedom
    alpha : float
        Significance level for the confidence interval, coverage is
        ``1-alpha``.
    alternative : str
        The alternative hypothesis, H1, has to be one of the following

           * 'two-sided' : H1: ``value1 - value2 - diff`` not equal to 0.
           * 'larger' :   H1: ``value1 - value2 - diff > 0``
           * 'smaller' :  H1: ``value1 - value2 - diff < 0``

    Returns
    -------
    lower : float or ndarray
        Lower confidence limit. This is -inf for the one-sided alternative
        "smaller".
    upper : float or ndarray
        Upper confidence limit. This is inf for the one-sided alternative
        "larger".
    """

    if alternative in ["two-sided", "2-sided", "2s"]:
        tcrit = stats.t.ppf(1 - alpha / 2.0, dof)
        lower = mean - tcrit * std_mean
        upper = mean + tcrit * std_mean
    elif alternative in ["larger", "l"]:
        tcrit = stats.t.ppf(alpha, dof)
        lower = mean + tcrit * std_mean
        upper = np.inf
    elif alternative in ["smaller", "s"]:
        tcrit = stats.t.ppf(1 - alpha, dof)
        lower = -np.inf
        upper = mean + tcrit * std_mean
    else:
        raise ValueError("invalid alternative")

    return lower, upper


def _zstat_generic(value1, value2, std_diff, alternative, diff=0):
    """generic (normal) z-test based on summary statistic

    The test statistic is :
        tstat = (value1 - value2 - diff) / std_diff

    and is assumed to be normally distributed.

    Parameters
    ----------
    value1 : float or ndarray
        Value, for example mean, of the first sample.
    value2 : float or ndarray
        Value, for example mean, of the second sample.
    std_diff : float or ndarray
        Standard error of the difference value1 - value2
    alternative : str
        The alternative hypothesis, H1, has to be one of the following

           * 'two-sided' : H1: ``value1 - value2 - diff`` not equal to 0.
           * 'larger' :   H1: ``value1 - value2 - diff > 0``
           * 'smaller' :  H1: ``value1 - value2 - diff < 0``

    diff : float
        value of difference ``value1 - value2`` under the null hypothesis

    Returns
    -------
    tstat : float or ndarray
        Test statistic.
    pvalue : float or ndarray
        P-value of the hypothesis test assuming that the test statistic is
        t-distributed with ``df`` degrees of freedom.
    """

    zstat = (value1 - value2 - diff) / std_diff
    if alternative in ["two-sided", "2-sided", "2s"]:
        pvalue = stats.norm.sf(np.abs(zstat)) * 2
    elif alternative in ["larger", "l"]:
        pvalue = stats.norm.sf(zstat)
    elif alternative in ["smaller", "s"]:
        pvalue = stats.norm.cdf(zstat)
    else:
        raise ValueError("invalid alternative")
    return zstat, pvalue


def _zstat_generic2(value, std, alternative):
    """generic (normal) z-test based on summary statistic

    The test statistic is :
        zstat = value / std

    and is assumed to be normally distributed with standard deviation ``std``.

    Parameters
    ----------
    value : float or ndarray
        Value of a sample statistic, for example mean.
    value2 : float or ndarray
        Value, for example mean, of the second sample.
    std : float or ndarray
        Standard error of the sample statistic value.
    alternative : str
        The alternative hypothesis, H1, has to be one of the following

           * 'two-sided' : H1: ``value1 - value2 - diff`` not equal to 0.
           * 'larger' :   H1: ``value1 - value2 - diff > 0``
           * 'smaller' :  H1: ``value1 - value2 - diff < 0``

    Returns
    -------
    zstat : float or ndarray
        Test statistic.
    pvalue : float or ndarray
        P-value of the hypothesis test assuming that the test statistic is
        normally distributed.
    """

    zstat = value / std
    if alternative in ["two-sided", "2-sided", "2s"]:
        pvalue = stats.norm.sf(np.abs(zstat)) * 2
    elif alternative in ["larger", "l"]:
        pvalue = stats.norm.sf(zstat)
    elif alternative in ["smaller", "s"]:
        pvalue = stats.norm.cdf(zstat)
    else:
        raise ValueError("invalid alternative")
    return zstat, pvalue


def _zconfint_generic(mean, std_mean, alpha, alternative):
    """generic normal-confint based on summary statistic

    Parameters
    ----------
    mean : float or ndarray
        Value, for example mean, of the first sample.
    std_mean : float or ndarray
        Standard error of the difference value1 - value2
    alpha : float
        Significance level for the confidence interval, coverage is
        ``1-alpha``
    alternative : str
        The alternative hypothesis, H1, has to be one of the following

           * 'two-sided' : H1: ``value1 - value2 - diff`` not equal to 0.
           * 'larger' :   H1: ``value1 - value2 - diff > 0``
           * 'smaller' :  H1: ``value1 - value2 - diff < 0``

    Returns
    -------
    lower : float or ndarray
        Lower confidence limit. This is -inf for the one-sided alternative
        "smaller".
    upper : float or ndarray
        Upper confidence limit. This is inf for the one-sided alternative
        "larger".
    """

    if alternative in ["two-sided", "2-sided", "2s"]:
        zcrit = stats.norm.ppf(1 - alpha / 2.0)
        lower = mean - zcrit * std_mean
        upper = mean + zcrit * std_mean
    elif alternative in ["larger", "l"]:
        zcrit = stats.norm.ppf(alpha)
        lower = mean + zcrit * std_mean
        upper = np.inf
    elif alternative in ["smaller", "s"]:
        zcrit = stats.norm.ppf(1 - alpha)
        lower = -np.inf
        upper = mean + zcrit * std_mean
    else:
        raise ValueError("invalid alternative")

    return lower, upper


class CompareMeans:
    """class for two sample comparison

    The tests and the confidence interval work for multi-endpoint comparison:
    If d1 and d2 have the same number of rows, then each column of the data
    in d1 is compared with the corresponding column in d2.

    Parameters
    ----------
    d1, d2 : instances of DescrStatsW

    Notes
    -----
    The result for the statistical tests and the confidence interval are
    independent of the user specified ddof.

    TODO: Extend to any number of groups or write a version that works in that
    case, like in SAS and SPSS.

    """

    def __init__(self, d1, d2):
        """assume d1, d2 hold the relevant attributes

        """
        self.d1 = d1
        self.d2 = d2
        # assume nobs is available

    #        if not hasattr(self.d1, 'nobs'):
    #            d1.nobs1 = d1.sum_weights.astype(float)  #float just to make sure
    #        self.nobs2 = d2.sum_weights.astype(float)

    @classmethod
    def from_data(
        cls, data1, data2, weights1=None, weights2=None, ddof1=0, ddof2=0
    ):
        """construct a CompareMeans object from data

        Parameters
        ----------
        data1, data2 : array_like, 1-D or 2-D
            compared datasets
        weights1, weights2 : None or 1-D ndarray
            weights for each observation of data1 and data2 respectively,
            with same length as zero axis of corresponding dataset.
        ddof1, ddof2 : int
            default ddof1=0, ddof2=0, degrees of freedom for data1,
            data2 respectively.

        Returns
        -------
        A CompareMeans instance.

        """
        return cls(
            DescrStatsW(data1, weights=weights1, ddof=ddof1),
            DescrStatsW(data2, weights=weights2, ddof=ddof2),
        )

    def summary(self, use_t=True, alpha=0.05, usevar="pooled", value=0):
        """summarize the results of the hypothesis test

        Parameters
        ----------
        use_t : bool, optional
            if use_t is True, then t test results are returned
            if use_t is False, then z test results are returned
        alpha : float
            significance level for the confidence interval, coverage is
            ``1-alpha``
        usevar : str, 'pooled' or 'unequal'
            If ``pooled``, then the standard deviation of the samples is
            assumed to be the same. If ``unequal``, then the variance of
            Welch ttest will be used, and the degrees of freedom are those
            of Satterthwaite if ``use_t`` is True.
        value : float
            difference between the means under the Null hypothesis.

        Returns
        -------
        smry : SimpleTable

        """

        d1 = self.d1
        d2 = self.d2

        confint_percents = 100 - alpha * 100

        if use_t:
            tstat, pvalue, _ = self.ttest_ind(usevar=usevar, value=value)
            lower, upper = self.tconfint_diff(alpha=alpha, usevar=usevar)
        else:
            tstat, pvalue = self.ztest_ind(usevar=usevar, value=value)
            lower, upper = self.zconfint_diff(alpha=alpha, usevar=usevar)

        if usevar == "pooled":
            std_err = self.std_meandiff_pooledvar
        else:
            std_err = self.std_meandiff_separatevar

        std_err = np.atleast_1d(std_err)
        tstat = np.atleast_1d(tstat)
        pvalue = np.atleast_1d(pvalue)
        lower = np.atleast_1d(lower)
        upper = np.atleast_1d(upper)
        conf_int = np.column_stack((lower, upper))
        params = np.atleast_1d(d1.mean - d2.mean - value)

        title = "Test for equality of means"
        yname = "y"  # not used in params_frame
        xname = ["subset #%d" % (ii + 1) for ii in range(tstat.shape[0])]

        from statsmodels.iolib.summary import summary_params

        return summary_params(
            (None, params, std_err, tstat, pvalue, conf_int),
            alpha=alpha,
            use_t=use_t,
            yname=yname,
            xname=xname,
            title=title,
        )

    @cache_readonly
    def std_meandiff_separatevar(self):
        # this uses ``_var`` to use ddof=0 for formula
        d1 = self.d1
        d2 = self.d2
        return np.sqrt(d1._var / (d1.nobs - 1) + d2._var / (d2.nobs - 1))

    @cache_readonly
    def std_meandiff_pooledvar(self):
        """variance assuming equal variance in both data sets

        """
        # this uses ``_var`` to use ddof=0 for formula

        d1 = self.d1
        d2 = self.d2
        # could make var_pooled into attribute
        var_pooled = (
            (d1.sumsquares + d2.sumsquares)
            /
            # (d1.nobs - d1.ddof + d2.nobs - d2.ddof))
            (d1.nobs - 1 + d2.nobs - 1)
        )
        return np.sqrt(var_pooled * (1.0 / d1.nobs + 1.0 / d2.nobs))

    def dof_satt(self):
        """degrees of freedom of Satterthwaite for unequal variance
        """
        d1 = self.d1
        d2 = self.d2
        # this follows blindly the SPSS manual
        # except I use  ``_var`` which has ddof=0
        sem1 = d1._var / (d1.nobs - 1)
        sem2 = d2._var / (d2.nobs - 1)
        semsum = sem1 + sem2
        z1 = (sem1 / semsum) ** 2 / (d1.nobs - 1)
        z2 = (sem2 / semsum) ** 2 / (d2.nobs - 1)
        dof = 1.0 / (z1 + z2)
        return dof

    def ttest_ind(self, alternative="two-sided", usevar="pooled", value=0):
        """ttest for the null hypothesis of identical means

        this should also be the same as onewaygls, except for ddof differences

        Parameters
        ----------
        x1 : array_like, 1-D or 2-D
            first of the two independent samples, see notes for 2-D case
        x2 : array_like, 1-D or 2-D
            second of the two independent samples, see notes for 2-D case
        alternative : str
            The alternative hypothesis, H1, has to be one of the following
            'two-sided': H1: difference in means not equal to value (default)
            'larger' :   H1: difference in means larger than value
            'smaller' :  H1: difference in means smaller than value

        usevar : str, 'pooled' or 'unequal'
            If ``pooled``, then the standard deviation of the samples is assumed to be
            the same. If ``unequal``, then Welch ttest with Satterthwait degrees
            of freedom is used
        value : float
            difference between the means under the Null hypothesis.


        Returns
        -------
        tstat : float
            test statistic
        pvalue : float
            pvalue of the t-test
        df : int or float
            degrees of freedom used in the t-test

        Notes
        -----
        The result is independent of the user specified ddof.

        """
        d1 = self.d1
        d2 = self.d2

        if usevar == "pooled":
            stdm = self.std_meandiff_pooledvar
            dof = d1.nobs - 1 + d2.nobs - 1
        elif usevar == "unequal":
            stdm = self.std_meandiff_separatevar
            dof = self.dof_satt()
        else:
            raise ValueError('usevar can only be "pooled" or "unequal"')

        tstat, pval = _tstat_generic(
            d1.mean, d2.mean, stdm, dof, alternative, diff=value
        )

        return tstat, pval, dof

    def ztest_ind(self, alternative="two-sided", usevar="pooled", value=0):
        """z-test for the null hypothesis of identical means

        Parameters
        ----------
        x1 : array_like, 1-D or 2-D
            first of the two independent samples, see notes for 2-D case
        x2 : array_like, 1-D or 2-D
            second of the two independent samples, see notes for 2-D case
        alternative : str
            The alternative hypothesis, H1, has to be one of the following
            'two-sided': H1: difference in means not equal to value (default)
            'larger' :   H1: difference in means larger than value
            'smaller' :  H1: difference in means smaller than value

        usevar : str, 'pooled' or 'unequal'
            If ``pooled``, then the standard deviation of the samples is assumed to be
            the same. If ``unequal``, then the standard deviations of the samples may
            be different.
        value : float
            difference between the means under the Null hypothesis.

        Returns
        -------
        tstat : float
            test statistic
        pvalue : float
            pvalue of the z-test

        """
        d1 = self.d1
        d2 = self.d2

        if usevar == "pooled":
            stdm = self.std_meandiff_pooledvar
        elif usevar == "unequal":
            stdm = self.std_meandiff_separatevar
        else:
            raise ValueError('usevar can only be "pooled" or "unequal"')

        tstat, pval = _zstat_generic(
            d1.mean, d2.mean, stdm, alternative, diff=value
        )

        return tstat, pval

    def tconfint_diff(
        self, alpha=0.05, alternative="two-sided", usevar="pooled"
    ):
        """confidence interval for the difference in means

        Parameters
        ----------
        alpha : float
            significance level for the confidence interval, coverage is
            ``1-alpha``
        alternative : str
            This specifies the alternative hypothesis for the test that
            corresponds to the confidence interval.
            The alternative hypothesis, H1, has to be one of the following :

            'two-sided': H1: difference in means not equal to value (default)
            'larger' :   H1: difference in means larger than value
            'smaller' :  H1: difference in means smaller than value

        usevar : str, 'pooled' or 'unequal'
            If ``pooled``, then the standard deviation of the samples is assumed to be
            the same. If ``unequal``, then Welch ttest with Satterthwait degrees
            of freedom is used

        Returns
        -------
        lower, upper : floats
            lower and upper limits of the confidence interval

        Notes
        -----
        The result is independent of the user specified ddof.

        """
        d1 = self.d1
        d2 = self.d2
        diff = d1.mean - d2.mean
        if usevar == "pooled":
            std_diff = self.std_meandiff_pooledvar
            dof = d1.nobs - 1 + d2.nobs - 1
        elif usevar == "unequal":
            std_diff = self.std_meandiff_separatevar
            dof = self.dof_satt()
        else:
            raise ValueError('usevar can only be "pooled" or "unequal"')

        res = _tconfint_generic(
            diff, std_diff, dof, alpha=alpha, alternative=alternative
        )
        return res

    def zconfint_diff(
        self, alpha=0.05, alternative="two-sided", usevar="pooled"
    ):
        """confidence interval for the difference in means

        Parameters
        ----------
        alpha : float
            significance level for the confidence interval, coverage is
            ``1-alpha``
        alternative : str
            This specifies the alternative hypothesis for the test that
            corresponds to the confidence interval.
            The alternative hypothesis, H1, has to be one of the following :

            'two-sided': H1: difference in means not equal to value (default)
            'larger' :   H1: difference in means larger than value
            'smaller' :  H1: difference in means smaller than value

        usevar : str, 'pooled' or 'unequal'
            If ``pooled``, then the standard deviation of the samples is assumed to be
            the same. If ``unequal``, then Welch ttest with Satterthwait degrees
            of freedom is used

        Returns
        -------
        lower, upper : floats
            lower and upper limits of the confidence interval

        Notes
        -----
        The result is independent of the user specified ddof.

        """
        d1 = self.d1
        d2 = self.d2
        diff = d1.mean - d2.mean
        if usevar == "pooled":
            std_diff = self.std_meandiff_pooledvar
        elif usevar == "unequal":
            std_diff = self.std_meandiff_separatevar
        else:
            raise ValueError('usevar can only be "pooled" or "unequal"')

        res = _zconfint_generic(
            diff, std_diff, alpha=alpha, alternative=alternative
        )
        return res

    def ttost_ind(self, low, upp, usevar="pooled"):
        """
        test of equivalence for two independent samples, base on t-test

        Parameters
        ----------
        low, upp : float
            equivalence interval low < m1 - m2 < upp
        usevar : str, 'pooled' or 'unequal'
            If ``pooled``, then the standard deviation of the samples is assumed to be
            the same. If ``unequal``, then Welch ttest with Satterthwait degrees
            of freedom is used

        Returns
        -------
        pvalue : float
            pvalue of the non-equivalence test
        t1, pv1 : tuple of floats
            test statistic and pvalue for lower threshold test
        t2, pv2 : tuple of floats
            test statistic and pvalue for upper threshold test
        """
        tt1 = self.ttest_ind(alternative="larger", usevar=usevar, value=low)
        tt2 = self.ttest_ind(alternative="smaller", usevar=usevar, value=upp)
        # TODO: remove tuple return, use same as for function tost_ind
        return np.maximum(tt1[1], tt2[1]), (tt1, tt2)

    def ztost_ind(self, low, upp, usevar="pooled"):
        """
        test of equivalence for two independent samples, based on z-test

        Parameters
        ----------
        low, upp : float
            equivalence interval low < m1 - m2 < upp
        usevar : str, 'pooled' or 'unequal'
            If ``pooled``, then the standard deviation of the samples is assumed to be
            the same. If ``unequal``, then Welch ttest with Satterthwait degrees
            of freedom is used

        Returns
        -------
        pvalue : float
            pvalue of the non-equivalence test
        t1, pv1 : tuple of floats
            test statistic and pvalue for lower threshold test
        t2, pv2 : tuple of floats
            test statistic and pvalue for upper threshold test
        """
        tt1 = self.ztest_ind(alternative="larger", usevar=usevar, value=low)
        tt2 = self.ztest_ind(alternative="smaller", usevar=usevar, value=upp)
        # TODO: remove tuple return, use same as for function tost_ind
        return np.maximum(tt1[1], tt2[1]), tt1, tt2

    # tost.__doc__ = tost_ind.__doc__


# does not work for 2d, does not take weights into account
##    def test_equal_var(self):
##        """Levene test for independence
##
##        """
##        d1 = self.d1
##        d2 = self.d2
##        #rewrite this, for now just use scipy.stats
##        return stats.levene(d1.data, d2.data)


def ttest_ind(
    x1,
    x2,
    alternative="two-sided",
    usevar="pooled",
    weights=(None, None),
    value=0,
):
    """ttest independent sample

    Convenience function that uses the classes and throws away the intermediate
    results,
    compared to scipy stats: drops axis option, adds alternative, usevar, and
    weights option.

    Parameters
    ----------
    x1 : array_like, 1-D or 2-D
        first of the two independent samples, see notes for 2-D case
    x2 : array_like, 1-D or 2-D
        second of the two independent samples, see notes for 2-D case
    alternative : str
        The alternative hypothesis, H1, has to be one of the following

           * 'two-sided' (default): H1: difference in means not equal to value
           * 'larger' :   H1: difference in means larger than value
           * 'smaller' :  H1: difference in means smaller than value

    usevar : str, 'pooled' or 'unequal'
        If ``pooled``, then the standard deviation of the samples is assumed to be
        the same. If ``unequal``, then Welch ttest with Satterthwait degrees
        of freedom is used
    weights : tuple of None or ndarrays
        Case weights for the two samples. For details on weights see
        ``DescrStatsW``
    value : float
        difference between the means under the Null hypothesis.


    Returns
    -------
    tstat : float
        test statistic
    pvalue : float
        pvalue of the t-test
    df : int or float
        degrees of freedom used in the t-test

    """
    cm = CompareMeans(
        DescrStatsW(x1, weights=weights[0], ddof=0),
        DescrStatsW(x2, weights=weights[1], ddof=0),
    )
    tstat, pval, dof = cm.ttest_ind(
        alternative=alternative, usevar=usevar, value=value
    )

    return tstat, pval, dof


def ttost_ind(
    x1, x2, low, upp, usevar="pooled", weights=(None, None), transform=None
):
    """test of (non-)equivalence for two independent samples

    TOST: two one-sided t tests

    null hypothesis:  m1 - m2 < low or m1 - m2 > upp
    alternative hypothesis:  low < m1 - m2 < upp

    where m1, m2 are the means, expected values of the two samples.

    If the pvalue is smaller than a threshold, say 0.05, then we reject the
    hypothesis that the difference between the two samples is larger than the
    the thresholds given by low and upp.

    Parameters
    ----------
    x1 : array_like, 1-D or 2-D
        first of the two independent samples, see notes for 2-D case
    x2 : array_like, 1-D or 2-D
        second of the two independent samples, see notes for 2-D case
    low, upp : float
        equivalence interval low < m1 - m2 < upp
    usevar : str, 'pooled' or 'unequal'
        If ``pooled``, then the standard deviation of the samples is assumed to be
        the same. If ``unequal``, then Welch ttest with Satterthwait degrees
        of freedom is used
    weights : tuple of None or ndarrays
        Case weights for the two samples. For details on weights see
        ``DescrStatsW``
    transform : None or function
        If None (default), then the data is not transformed. Given a function,
        sample data and thresholds are transformed. If transform is log, then
        the equivalence interval is in ratio: low < m1 / m2 < upp

    Returns
    -------
    pvalue : float
        pvalue of the non-equivalence test
    t1, pv1 : tuple of floats
        test statistic and pvalue for lower threshold test
    t2, pv2 : tuple of floats
        test statistic and pvalue for upper threshold test

    Notes
    -----
    The test rejects if the 2*alpha confidence interval for the difference
    is contained in the ``(low, upp)`` interval.

    This test works also for multi-endpoint comparisons: If d1 and d2
    have the same number of columns, then each column of the data in d1 is
    compared with the corresponding column in d2. This is the same as
    comparing each of the corresponding columns separately. Currently no
    multi-comparison correction is used. The raw p-values reported here can
    be correction with the functions in ``multitest``.

    """

    if transform:
        if transform is np.log:
            # avoid hstack in special case
            x1 = transform(x1)
            x2 = transform(x2)
        else:
            # for transforms like rankdata that will need both datasets
            # concatenate works for stacking 1d and 2d arrays
            xx = transform(np.concatenate((x1, x2), 0))
            x1 = xx[: len(x1)]
            x2 = xx[len(x1) :]
        low = transform(low)
        upp = transform(upp)
    cm = CompareMeans(
        DescrStatsW(x1, weights=weights[0], ddof=0),
        DescrStatsW(x2, weights=weights[1], ddof=0),
    )
    pval, res = cm.ttost_ind(low, upp, usevar=usevar)
    return pval, res[0], res[1]


def ttost_paired(x1, x2, low, upp, transform=None, weights=None):
    """test of (non-)equivalence for two dependent, paired sample

    TOST: two one-sided t tests

    null hypothesis:  md < low or md > upp
    alternative hypothesis:  low < md < upp

    where md is the mean, expected value of the difference x1 - x2

    If the pvalue is smaller than a threshold,say 0.05, then we reject the
    hypothesis that the difference between the two samples is larger than the
    the thresholds given by low and upp.

    Parameters
    ----------
    x1 : array_like
        first of the two independent samples
    x2 : array_like
        second of the two independent samples
    low, upp : float
        equivalence interval low < mean of difference < upp
    weights : None or ndarray
        case weights for the two samples. For details on weights see
        ``DescrStatsW``
    transform : None or function
        If None (default), then the data is not transformed. Given a function
        sample data and thresholds are transformed. If transform is log the
        the equivalence interval is in ratio: low < x1 / x2 < upp

    Returns
    -------
    pvalue : float
        pvalue of the non-equivalence test
    t1, pv1, df1 : tuple
        test statistic, pvalue and degrees of freedom for lower threshold test
    t2, pv2, df2 : tuple
        test statistic, pvalue and degrees of freedom for upper threshold test

    """

    if transform:
        if transform is np.log:
            # avoid hstack in special case
            x1 = transform(x1)
            x2 = transform(x2)
        else:
            # for transforms like rankdata that will need both datasets
            # concatenate works for stacking 1d and 2d arrays
            xx = transform(np.concatenate((x1, x2), 0))
            x1 = xx[: len(x1)]
            x2 = xx[len(x1) :]
        low = transform(low)
        upp = transform(upp)
    dd = DescrStatsW(x1 - x2, weights=weights, ddof=0)
    t1, pv1, df1 = dd.ttest_mean(low, alternative="larger")
    t2, pv2, df2 = dd.ttest_mean(upp, alternative="smaller")
    return np.maximum(pv1, pv2), (t1, pv1, df1), (t2, pv2, df2)


def ztest(
    x1, x2=None, value=0, alternative="two-sided", usevar="pooled", ddof=1.0
):
    """test for mean based on normal distribution, one or two samples

    In the case of two samples, the samples are assumed to be independent.

    Parameters
    ----------
    x1 : array_like, 1-D or 2-D
        first of the two independent samples
    x2 : array_like, 1-D or 2-D
        second of the two independent samples
    value : float
        In the one sample case, value is the mean of x1 under the Null
        hypothesis.
        In the two sample case, value is the difference between mean of x1 and
        mean of x2 under the Null hypothesis. The test statistic is
        `x1_mean - x2_mean - value`.
    alternative : str
        The alternative hypothesis, H1, has to be one of the following

           'two-sided': H1: difference in means not equal to value (default)
           'larger' :   H1: difference in means larger than value
           'smaller' :  H1: difference in means smaller than value

    usevar : str, 'pooled' or 'unequal'
        If ``pooled``, then the standard deviation of the samples is assumed to be
        the same. If ``unequal``, then the standard deviation of the sample is
        assumed to be different.
    ddof : int
        Degrees of freedom use in the calculation of the variance of the mean
        estimate. In the case of comparing means this is one, however it can
        be adjusted for testing other statistics (proportion, correlation)

    Returns
    -------
    tstat : float
        test statistic
    pvalue : float
        pvalue of the t-test

    Notes
    -----
    usevar can be pooled or unequal in two sample case

    """
    # TODO: this should delegate to CompareMeans like ttest_ind
    #       However that does not implement ddof

    # usevar can be pooled or unequal

    if usevar not in {"pooled", "unequal"}:
        raise NotImplementedError('usevar can only be "pooled" or "unequal"')

    x1 = np.asarray(x1)
    nobs1 = x1.shape[0]
    x1_mean = x1.mean(0)
    x1_var = x1.var(0)

    if x2 is not None:
        x2 = np.asarray(x2)
        nobs2 = x2.shape[0]
        x2_mean = x2.mean(0)
        x2_var = x2.var(0)
        if usevar == "pooled":
            var = nobs1 * x1_var + nobs2 * x2_var
            var /= nobs1 + nobs2 - 2 * ddof
            var *= 1.0 / nobs1 + 1.0 / nobs2
        elif usevar == "unequal":
            var = x1_var / (nobs1 - ddof) + x2_var / (nobs2 - ddof)
    else:
        var = x1_var / (nobs1 - ddof)
        x2_mean = 0

    std_diff = np.sqrt(var)
    # stat = x1_mean - x2_mean - value
    return _zstat_generic(x1_mean, x2_mean, std_diff, alternative, diff=value)


def zconfint(
    x1,
    x2=None,
    value=0,
    alpha=0.05,
    alternative="two-sided",
    usevar="pooled",
    ddof=1.0,
):
    """confidence interval based on normal distribution z-test

    Parameters
    ----------
    x1 : array_like, 1-D or 2-D
        first of the two independent samples, see notes for 2-D case
    x2 : array_like, 1-D or 2-D
        second of the two independent samples, see notes for 2-D case
    value : float
        In the one sample case, value is the mean of x1 under the Null
        hypothesis.
        In the two sample case, value is the difference between mean of x1 and
        mean of x2 under the Null hypothesis. The test statistic is
        `x1_mean - x2_mean - value`.
    usevar : str, 'pooled'
        Currently, only 'pooled' is implemented.
        If ``pooled``, then the standard deviation of the samples is assumed to be
        the same. see CompareMeans.ztest_ind for different options.
    ddof : int
        Degrees of freedom use in the calculation of the variance of the mean
        estimate. In the case of comparing means this is one, however it can
        be adjusted for testing other statistics (proportion, correlation)

    Notes
    -----
    checked only for 1 sample case

    usevar not implemented, is always pooled in two sample case

    ``value`` shifts the confidence interval so it is centered at
    `x1_mean - x2_mean - value`

    See Also
    --------
    ztest
    CompareMeans

    """
    # usevar is not used, always pooled
    # mostly duplicate code from ztest

    if usevar != "pooled":
        raise NotImplementedError('only usevar="pooled" is implemented')
    x1 = np.asarray(x1)
    nobs1 = x1.shape[0]
    x1_mean = x1.mean(0)
    x1_var = x1.var(0)
    if x2 is not None:
        x2 = np.asarray(x2)
        nobs2 = x2.shape[0]
        x2_mean = x2.mean(0)
        x2_var = x2.var(0)
        var_pooled = nobs1 * x1_var + nobs2 * x2_var
        var_pooled /= nobs1 + nobs2 - 2 * ddof
        var_pooled *= 1.0 / nobs1 + 1.0 / nobs2
    else:
        var_pooled = x1_var / (nobs1 - ddof)
        x2_mean = 0

    std_diff = np.sqrt(var_pooled)
    ci = _zconfint_generic(
        x1_mean - x2_mean - value, std_diff, alpha, alternative
    )
    return ci


def ztost(x1, low, upp, x2=None, usevar="pooled", ddof=1.0):
    """Equivalence test based on normal distribution

    Parameters
    ----------
    x1 : array_like
        one sample or first sample for 2 independent samples
    low, upp : float
        equivalence interval low < m1 - m2 < upp
    x1 : array_like or None
        second sample for 2 independent samples test. If None, then a
        one-sample test is performed.
    usevar : str, 'pooled'
        If `pooled`, then the standard deviation of the samples is assumed to be
        the same. Only `pooled` is currently implemented.

    Returns
    -------
    pvalue : float
        pvalue of the non-equivalence test
    t1, pv1 : tuple of floats
        test statistic and pvalue for lower threshold test
    t2, pv2 : tuple of floats
        test statistic and pvalue for upper threshold test

    Notes
    -----
    checked only for 1 sample case

    """
    tt1 = ztest(
        x1, x2, alternative="larger", usevar=usevar, value=low, ddof=ddof
    )
    tt2 = ztest(
        x1, x2, alternative="smaller", usevar=usevar, value=upp, ddof=ddof
    )
    return (
        np.maximum(tt1[1], tt2[1]),
        tt1,
        tt2,
    )
