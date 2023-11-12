'''
Test for ratio of Poisson intensities in two independent samples

Author: Josef Perktold
License: BSD-3

'''

import numpy as np
import warnings

from scipy import stats, optimize

from statsmodels.stats.base import HolderTuple
from statsmodels.stats.weightstats import _zstat_generic2
from statsmodels.stats._inference_tools import _mover_confint

# shorthand
norm = stats.norm


method_names_poisson_1samp = {
    "test": [
        "wald",
        "score",
        "exact-c",
        "midp-c",
        "waldccv",
        "sqrt-a",
        "sqrt-v",
        "sqrt",
        ],
    "confint": [
        "wald",
        "score",
        "exact-c",
        "midp-c",
        "jeff",
        "waldccv",
        "sqrt-a",
        "sqrt-v",
        "sqrt",
        "sqrt-cent",
        "sqrt-centcc",
        ]
    }


def test_poisson(count, nobs, value, method=None, alternative="two-sided",
                 dispersion=1):
    """Test for one sample poisson mean or rate

    Parameters
    ----------
    count : array_like
        Observed count, number of events.
    nobs : arrat_like
        Currently this is total exposure time of the count variable.
        This will likely change.
    value : float, array_like
        This is the value of poisson rate under the null hypothesis.
    method : str
        Method to use for confidence interval.
        This is required, there is currently no default method.
        See Notes for available methods.
    alternative : {'two-sided', 'smaller', 'larger'}
        alternative hypothesis, which can be two-sided or either one of the
        one-sided tests.
    dispersion : float
        Dispersion scale coefficient for Poisson QMLE. Default is that the
        data follows a Poisson distribution. Dispersion different from 1
        correspond to excess-dispersion in Poisson quasi-likelihood (GLM).
        Dispersion coeffficient different from one is currently only used in
        wald and score method.

    Returns
    -------
    HolderTuple instance with test statistic, pvalue and other attributes.

    Notes
    -----
    The implementatio of the hypothesis test is mainly based on the references
    for the confidence interval, see confint_poisson.

    Available methods are:

    - "score" : based on score test, uses variance under null value
    - "wald" : based on wald test, uses variance base on estimated rate.
    - "waldccv" : based on wald test with 0.5 count added to variance
      computation. This does not use continuity correction for the center of
      the confidence interval.
    - "exact-c" central confidence interval based on gamma distribution
    - "midp-c" : based on midp correction of central exact confidence interval.
      this uses numerical inversion of the test function. not vectorized.
    - "sqrt" : based on square root transformed counts
    - "sqrt-a" based on Anscombe square root transformation of counts + 3/8.

    See Also
    --------
    confint_poisson

    """

    n = nobs  # short hand
    rate = count / n

    if method is None:
        msg = "method needs to be specified, currently no default method"
        raise ValueError(msg)

    if dispersion != 1:
        if method not in ["wald", "waldcc", "score"]:
            msg = "excess dispersion only supported in wald and score methods"
            raise ValueError(msg)

    dist = "normal"

    if method == "wald":
        std = np.sqrt(dispersion * rate / n)
        statistic = (rate - value) / std

    elif method == "waldccv":
        # WCC in Barker 2002
        # add 0.5 event, not 0.5 event rate as in waldcc
        # std = np.sqrt((rate + 0.5 / n) / n)
        # statistic = (rate + 0.5 / n - value) / std
        std = np.sqrt(dispersion * (rate + 0.5 / n) / n)
        statistic = (rate - value) / std

    elif method == "score":
        std = np.sqrt(dispersion * value / n)
        statistic = (rate - value) / std
        pvalue = stats.norm.sf(statistic)

    elif method.startswith("exact-c") or method.startswith("midp-c"):
        pv1 = stats.poisson.cdf(count, n * value)
        pv2 = stats.poisson.sf(count - 1, n * value)
        if method.startswith("midp-c"):
            pv1 = pv1 - 0.5 * stats.poisson.pmf(count, n * value)
            pv2 = pv2 - 0.5 * stats.poisson.pmf(count, n * value)
        if alternative == "two-sided":
            pvalue = 2 * np.minimum(pv1, pv2)
        elif alternative == "larger":
            pvalue = pv2
        elif alternative == "smaller":
            pvalue = pv1
        else:
            msg = 'alternative should be "two-sided", "larger" or "smaller"'
            raise ValueError(msg)

        statistic = np.nan
        dist = "Poisson"

    elif method == "sqrt":
        std = 0.5
        statistic = (np.sqrt(count) - np.sqrt(n * value)) / std

    elif method == "sqrt-a":
        # anscombe, based on Swift 2009 (with transformation to rate)
        std = 0.5
        statistic = (np.sqrt(count + 3 / 8) - np.sqrt(n * value + 3 / 8)) / std

    elif method == "sqrt-v":
        # vandenbroucke, based on Swift 2009 (with transformation to rate)
        std = 0.5
        crit = stats.norm.isf(0.025)
        statistic = (np.sqrt(count + (crit**2 + 2) / 12) -
                     # np.sqrt(n * value + (crit**2 + 2) / 12)) / std
                     np.sqrt(n * value)) / std

    else:
        raise ValueError("unknown method %s" % method)

    if dist == 'normal':
        statistic, pvalue = _zstat_generic2(statistic, 1, alternative)

    res = HolderTuple(
        statistic=statistic,
        pvalue=np.clip(pvalue, 0, 1),
        distribution=dist,
        method=method,
        alternative=alternative,
        rate=rate,
        nobs=n
        )
    return res


def confint_poisson(count, exposure, method=None, alpha=0.05):
    """Confidence interval for a Poisson mean or rate

    The function is vectorized for all methods except "midp-c", which uses
    an iterative method to invert the hypothesis test function.

    All current methods are central, that is the probability of each tail is
    smaller or equal to alpha / 2. The one-sided interval limits can be
    obtained by doubling alpha.

    Parameters
    ----------
    count : array_like
        Observed count, number of events.
    exposure : arrat_like
        Currently this is total exposure time of the count variable.
        This will likely change.
    method : str
        Method to use for confidence interval
        This is required, there is currently no default method
    alpha : float in (0, 1)
        Significance level, nominal coverage of the confidence interval is
        1 - alpha.

    Returns
    -------
    tuple (low, upp) : confidence limits.

    Notes
    -----
    Methods are mainly based on Barker (2002) [1]_ and Swift (2009) [3]_.

    Available methods are:

    - "exact-c" central confidence interval based on gamma distribution
    - "score" : based on score test, uses variance under null value
    - "wald" : based on wald test, uses variance base on estimated rate.
    - "waldccv" : based on wald test with 0.5 count added to variance
      computation. This does not use continuity correction for the center of
      the confidence interval.
    - "midp-c" : based on midp correction of central exact confidence interval.
      this uses numerical inversion of the test function. not vectorized.
    - "jeffreys" : based on Jeffreys' prior. computed using gamma distribution
    - "sqrt" : based on square root transformed counts
    - "sqrt-a" based on Anscombe square root transformation of counts + 3/8.
    - "sqrt-centcc" will likely be dropped. anscombe with continuity corrected
      center.
      (Similar to R survival cipoisson, but without the 3/8 right shift of
      the confidence interval).

    sqrt-cent is the same as sqrt-a, using a different computation, will be
    deleted.

    sqrt-v is a corrected square root method attributed to vandenbrouke, which
    might also be deleted.

    Todo:

    - missing dispersion,
    - maybe split nobs and exposure (? needed in NB). Exposure could be used
      to standardize rate.
    - modified wald, switch method if count=0.

    See Also
    --------
    test_poisson

    References
    ----------
    .. [1] Barker, Lawrence. 2002. “A Comparison of Nine Confidence Intervals
       for a Poisson Parameter When the Expected Number of Events Is ≤ 5.”
       The American Statistician 56 (2): 85–89.
       https://doi.org/10.1198/000313002317572736.
    .. [2] Patil, VV, and HV Kulkarni. 2012. “Comparison of Confidence
       Intervals for the Poisson Mean: Some New Aspects.”
       REVSTAT–Statistical Journal 10(2): 211–27.
    .. [3] Swift, Michael Bruce. 2009. “Comparison of Confidence Intervals for
       a Poisson Mean – Further Considerations.” Communications in Statistics -
       Theory and Methods 38 (5): 748–59.
       https://doi.org/10.1080/03610920802255856.

    """
    n = exposure  # short hand
    rate = count / exposure
    alpha = alpha / 2  # two-sided

    if method is None:
        msg = "method needs to be specified, currently no default method"
        raise ValueError(msg)

    if method == "wald":
        whalf = stats.norm.isf(alpha) * np.sqrt(rate / n)
        ci = (rate - whalf, rate + whalf)

    elif method == "waldccv":
        # based on WCC in Barker 2002
        # add 0.5 event, not 0.5 event rate as in BARKER waldcc
        whalf = stats.norm.isf(alpha) * np.sqrt((rate + 0.5 / n) / n)
        ci = (rate - whalf, rate + whalf)

    elif method == "score":
        crit = stats.norm.isf(alpha)
        center = count + crit**2 / 2
        whalf = crit * np.sqrt((count + crit**2 / 4))
        ci = ((center - whalf) / n, (center + whalf) / n)

    elif method == "midp-c":
        # note local alpha above is for one tail
        ci = _invert_test_confint(count, n, alpha=2 * alpha, method="midp-c",
                                  method_start="exact-c")

    elif method == "sqrt":
        # drop, wrong n
        crit = stats.norm.isf(alpha)
        center = rate + crit**2 / (4 * n)
        whalf = crit * np.sqrt(rate / n)
        ci = (center - whalf, center + whalf)

    elif method == "sqrt-cent":
        crit = stats.norm.isf(alpha)
        center = count + crit**2 / 4
        whalf = crit * np.sqrt((count + 3 / 8))
        ci = ((center - whalf) / n, (center + whalf) / n)

    elif method == "sqrt-centcc":
        # drop with cc, does not match cipoisson in R survival
        crit = stats.norm.isf(alpha)
        # avoid sqrt of negative value if count=0
        center_low = np.sqrt(np.maximum(count + 3 / 8 - 0.5, 0))
        center_upp = np.sqrt(count + 3 / 8 + 0.5)
        whalf = crit / 2
        # above is for ci of count
        ci = (((np.maximum(center_low - whalf, 0))**2 - 3 / 8) / n,
              ((center_upp + whalf)**2 - 3 / 8) / n)

        # crit = stats.norm.isf(alpha)
        # center = count
        # whalf = crit * np.sqrt((count + 3 / 8 + 0.5))
        # ci = ((center - whalf - 0.5) / n, (center + whalf + 0.5) / n)

    elif method == "sqrt-a":
        # anscombe, based on Swift 2009 (with transformation to rate)
        crit = stats.norm.isf(alpha)
        center = np.sqrt(count + 3 / 8)
        whalf = crit / 2
        # above is for ci of count
        ci = (((np.maximum(center - whalf, 0))**2 - 3 / 8) / n,
              ((center + whalf)**2 - 3 / 8) / n)

    elif method == "sqrt-v":
        # vandenbroucke, based on Swift 2009 (with transformation to rate)
        crit = stats.norm.isf(alpha)
        center = np.sqrt(count + (crit**2 + 2) / 12)
        whalf = crit / 2
        # above is for ci of count
        ci = (np.maximum(center - whalf, 0))**2 / n, (center + whalf)**2 / n

    elif method in ["gamma", "exact-c"]:
        # garwood exact, gamma
        low = stats.gamma.ppf(alpha, count) / exposure
        upp = stats.gamma.isf(alpha, count+1) / exposure
        if np.isnan(low).any():
            # case with count = 0
            if np.size(low) == 1:
                low = 0.0
            else:
                low[np.isnan(low)] = 0.0

        ci = (low, upp)

    elif method.startswith("jeff"):
        # jeffreys, gamma
        countc = count + 0.5
        ci = (stats.gamma.ppf(alpha, countc) / exposure,
              stats.gamma.isf(alpha, countc) / exposure)

    else:
        raise ValueError("unknown method %s" % method)

    ci = (np.maximum(ci[0], 0), ci[1])
    return ci


def tolerance_int_poisson(count, exposure, prob=0.95, exposure_new=1.,
                          method=None, alpha=0.05,
                          alternative="two-sided"):
    """tolerance interval for a poisson observation

    Parameters
    ----------
    count : array_like
        Observed count, number of events.
    exposure : arrat_like
        Currently this is total exposure time of the count variable.
    prob : float in (0, 1)
        Probability of poisson interval, often called "content".
        With known parameters, each tail would have at most probability
        ``1 - prob / 2`` in the two-sided interval.
    exposure_new : float
        Exposure of the new or predicted observation.
    method : str
        Method to used for confidence interval of the estimate of the
        poisson rate, used in `confint_poisson`.
        This is required, there is currently no default method.
    alpha : float in (0, 1)
        Significance level for the confidence interval of the estimate of the
        Poisson rate. Nominal coverage of the confidence interval is
        1 - alpha.
    alternative : {"two-sider", "larger", "smaller")
        The tolerance interval can be two-sided or one-sided.
        Alternative "larger" provides the upper bound of the confidence
        interval, larger counts are outside the interval.

    Returns
    -------
    tuple (low, upp) of limits of tolerance interval.
        The tolerance interval is a closed interval, that is both ``low`` and
        ``upp`` are in the interval.

    Notes
    -----
    verified against R package tolerance `poistol.int`

    See Also
    --------
    confint_poisson
    confint_quantile_poisson

    References
    ----------
    .. [1] Hahn, Gerald J., and William Q. Meeker. 1991. Statistical Intervals:
       A Guide for Practitioners. 1st ed. Wiley Series in Probability and
       Statistics. Wiley. https://doi.org/10.1002/9780470316771.
    .. [2] Hahn, Gerald J., and Ramesh Chandra. 1981. “Tolerance Intervals for
       Poisson and Binomial Variables.” Journal of Quality Technology 13 (2):
       100–110. https://doi.org/10.1080/00224065.1981.11980998.

    """
    prob_tail = 1 - prob
    alpha_ = alpha
    if alternative != "two-sided":
        # confint_poisson does not have one-sided alternatives
        alpha_ = alpha * 2
    low, upp = confint_poisson(count, exposure, method=method, alpha=alpha_)

    if exposure_new != 1:
        low *= exposure_new
        upp *= exposure_new

    if alternative == "two-sided":
        low_pred = stats.poisson.ppf(prob_tail / 2, low)
        upp_pred = stats.poisson.ppf(1 - prob_tail / 2, upp)
    elif alternative == "larger":
        low_pred = 0
        upp_pred = stats.poisson.ppf(1 - prob_tail, upp)
    elif alternative == "smaller":
        low_pred = stats.poisson.ppf(prob_tail, low)
        upp_pred = np.inf

    # clip -1 of ppf(0)
    low_pred = np.maximum(low_pred, 0)
    return low_pred, upp_pred


def confint_quantile_poisson(count, exposure, prob, exposure_new=1.,
                             method=None, alpha=0.05,
                             alternative="two-sided"):
    """confidence interval for quantile of poisson random variable

    Parameters
    ----------
    count : array_like
        Observed count, number of events.
    exposure : arrat_like
        Currently this is total exposure time of the count variable.
    prob : float in (0, 1)
        Probability for the quantile, e.g. 0.95 to get the upper 95% quantile.
        With known mean mu, the quantile would be poisson.ppf(prob, mu).
    exposure_new : float
        Exposure of the new or predicted observation.
    method : str
        Method to used for confidence interval of the estimate of the
        poisson rate, used in `confint_poisson`.
        This is required, there is currently no default method.
    alpha : float in (0, 1)
        Significance level for the confidence interval of the estimate of the
        Poisson rate. Nominal coverage of the confidence interval is
        1 - alpha.
    alternative : {"two-sider", "larger", "smaller")
        The tolerance interval can be two-sided or one-sided.
        Alternative "larger" provides the upper bound of the confidence
        interval, larger counts are outside the interval.

    Returns
    -------
    tuple (low, upp) of limits of tolerance interval.
    The confidence interval is a closed interval, that is both ``low`` and
    ``upp`` are in the interval.

    See Also
    --------
    confint_poisson
    tolerance_int_poisson

    References
    ----------
    Hahn, Gerald J, and William Q Meeker. 2010. Statistical Intervals: A Guide
    for Practitioners.
    """
    alpha_ = alpha
    if alternative != "two-sided":
        # confint_poisson does not have one-sided alternatives
        alpha_ = alpha * 2
    low, upp = confint_poisson(count, exposure, method=method, alpha=alpha_)
    if exposure_new != 1:
        low *= exposure_new
        upp *= exposure_new

    if alternative == "two-sided":
        low_pred = stats.poisson.ppf(prob, low)
        upp_pred = stats.poisson.ppf(prob, upp)
    elif alternative == "larger":
        low_pred = 0
        upp_pred = stats.poisson.ppf(prob, upp)
    elif alternative == "smaller":
        low_pred = stats.poisson.ppf(prob, low)
        upp_pred = np.inf

    # clip -1 of ppf(0)
    low_pred = np.maximum(low_pred, 0)
    return low_pred, upp_pred


def _invert_test_confint(count, nobs, alpha=0.05, method="midp-c",
                         method_start="exact-c"):
    """invert hypothesis test to get confidence interval
    """

    def func(r):
        v = (test_poisson(count, nobs, value=r, method=method)[1] -
             alpha)**2
        return v

    ci = confint_poisson(count, nobs, method=method_start)
    low = optimize.fmin(func, ci[0], xtol=1e-8, disp=False)
    upp = optimize.fmin(func, ci[1], xtol=1e-8, disp=False)
    assert np.size(low) == 1
    return low[0], upp[0]


def _invert_test_confint_2indep(
        count1, exposure1, count2, exposure2,
        alpha=0.05,
        method="score",
        compare="diff",
        method_start="wald"
        ):
    """invert hypothesis test to get confidence interval for 2indep
    """

    def func(r):
        v = (test_poisson_2indep(
             count1, exposure1, count2, exposure2,
             value=r, method=method, compare=compare
             )[1] - alpha)**2
        return v

    ci = confint_poisson_2indep(count1, exposure1, count2, exposure2,
                                method=method_start, compare=compare)
    low = optimize.fmin(func, ci[0], xtol=1e-8, disp=False)
    upp = optimize.fmin(func, ci[1], xtol=1e-8, disp=False)
    assert np.size(low) == 1
    return low[0], upp[0]


method_names_poisson_2indep = {
    "test": {
        "ratio": [
            "wald",
            "score",
            "score-log",
            "wald-log",
            "exact-cond",
            "cond-midp",
            "sqrt",
            "etest-score",
            "etest-wald"
            ],
        "diff": [
            "wald",
            "score",
            "waldccv",
            "etest-score",
            "etest-wald"
            ]
        },
    "confint": {
        "ratio": [
            "waldcc",
            "score",
            "score-log",
            "wald-log",
            "sqrtcc",
            "mover",
            ],
        "diff": [
            "wald",
            "score",
            "waldccv",
            "mover"
            ]
        }
    }


def test_poisson_2indep(count1, exposure1, count2, exposure2, value=None,
                        ratio_null=None,
                        method=None, compare='ratio',
                        alternative='two-sided', etest_kwds=None):
    '''Test for comparing two sample Poisson intensity rates.

    Rates are defined as expected count divided by exposure.

    The Null and alternative hypothesis for the rates, rate1 and rate2, of two
    independent Poisson samples are

    for compare = 'diff'

    - H0: rate1 - rate2 - value = 0
    - H1: rate1 - rate2 - value != 0  if alternative = 'two-sided'
    - H1: rate1 - rate2 - value > 0   if alternative = 'larger'
    - H1: rate1 - rate2 - value < 0   if alternative = 'smaller'

    for compare = 'ratio'

    - H0: rate1 / rate2 - value = 0
    - H1: rate1 / rate2 - value != 0  if alternative = 'two-sided'
    - H1: rate1 / rate2 - value > 0   if alternative = 'larger'
    - H1: rate1 / rate2 - value < 0   if alternative = 'smaller'

    Parameters
    ----------
    count1 : int
        Number of events in first sample, treatment group.
    exposure1 : float
        Total exposure (time * subjects) in first sample.
    count2 : int
        Number of events in second sample, control group.
    exposure2 : float
        Total exposure (time * subjects) in second sample.
    ratio_null: float
        Ratio of the two Poisson rates under the Null hypothesis. Default is 1.
        Deprecated, use ``value`` instead.

        .. deprecated:: 0.14.0

            Use ``value`` instead.

    value : float
        Value of the ratio or difference of 2 independent rates under the null
        hypothesis. Default is equal rates, i.e. 1 for ratio and 0 for diff.

        .. versionadded:: 0.14.0

            Replacement for ``ratio_null``.

    method : string
        Method for the test statistic and the p-value. Defaults to `'score'`.
        see Notes.

        ratio:

        - 'wald': method W1A, wald test, variance based on observed rates
        - 'score': method W2A, score test, variance based on estimate under
          the Null hypothesis
        - 'wald-log': W3A, uses log-ratio, variance based on observed rates
        - 'score-log' W4A, uses log-ratio, variance based on estimate under
          the Null hypothesis
        - 'sqrt': W5A, based on variance stabilizing square root transformation
        - 'exact-cond': exact conditional test based on binomial distribution
           This uses ``binom_test`` which is minlike in the two-sided case.
        - 'cond-midp': midpoint-pvalue of exact conditional test
        - 'etest' or 'etest-score: etest with score test statistic
        - 'etest-wald': etest with wald test statistic

        diff:

        - 'wald',
        - 'waldccv'
        - 'score'
        - 'etest-score' or 'etest: etest with score test statistic
        - 'etest-wald': etest with wald test statistic

    compare : {'diff', 'ratio'}
        Default is "ratio".
        If compare is `ratio`, then the hypothesis test is for the
        rate ratio defined by ratio = rate1 / rate2.
        If compare is `diff`, then the hypothesis test is for
        diff = rate1 - rate2.
    alternative : {"two-sided" (default), "larger", smaller}
        The alternative hypothesis, H1, has to be one of the following

        - 'two-sided': H1: ratio, or diff, of rates is not equal to value
        - 'larger' :   H1: ratio, or diff, of rates is larger than value
        - 'smaller' :  H1: ratio, or diff, of rates is smaller than value
    etest_kwds: dictionary
        Additional optional parameters to be passed to the etest_poisson_2indep
        function, namely y_grid.

    Returns
    -------
    results : instance of HolderTuple class
        The two main attributes are test statistic `statistic` and p-value
        `pvalue`.

    See Also
    --------
    tost_poisson_2indep
    etest_poisson_2indep

    Notes
    -----
    The hypothesis tests for compare="ratio" are based on Gu et al 2018.
    The e-tests are also based on ...

    - 'wald': method W1A, wald test, variance based on separate estimates
    - 'score': method W2A, score test, variance based on estimate under Null
    - 'wald-log': W3A, wald test for log transformed ratio
    - 'score-log' W4A, score test for log transformed ratio
    - 'sqrt': W5A, based on variance stabilizing square root transformation
    - 'exact-cond': exact conditional test based on binomial distribution
    - 'cond-midp': midpoint-pvalue of exact conditional test
    - 'etest': etest with score test statistic
    - 'etest-wald': etest with wald test statistic

    The hypothesis test for compare="diff" are mainly based on Ng et al 2007
    and ...

    - wald
    - score
    - etest-score
    - etest-wald

    Note the etests use the constraint maximum likelihood estimate (cmle) as
    parameters for the underlying Poisson probabilities. The constraint cmle
    parameters are the same as in the score test.
    The E-test in Krishnamoorty and Thomson uses a moment estimator instead of
    the score estimator.

    References
    ----------
    .. [1] Gu, Ng, Tang, Schucany 2008: Testing the Ratio of Two Poisson Rates,
       Biometrical Journal 50 (2008) 2, 2008

    .. [2] Ng, H. K. T., K. Gu, and M. L. Tang. 2007. “A Comparative Study of
       Tests for the Difference of Two Poisson Means.”
       Computational Statistics & Data Analysis 51 (6): 3085–99.
       https://doi.org/10.1016/j.csda.2006.02.004.

    '''

    # shortcut names
    y1, n1, y2, n2 = map(np.asarray, [count1, exposure1, count2, exposure2])
    d = n2 / n1
    rate1, rate2 = y1 / n1, y2 / n2
    rates_cmle = None

    if compare == 'ratio':
        if method is None:
            # default method
            method = 'score'

        if ratio_null is not None:
            warnings.warn("'ratio_null' is deprecated, use 'value' keyword",
                          FutureWarning)
            value = ratio_null
        if ratio_null is None and value is None:
            # default value
            value = ratio_null = 1
        else:
            # for results holder instance, it still contains ratio_null
            ratio_null = value

        r = value
        r_d = r / d   # r1 * n1 / (r2 * n2)

        if method in ['score']:
            stat = (y1 - y2 * r_d) / np.sqrt((y1 + y2) * r_d)
            dist = 'normal'
        elif method in ['wald']:
            stat = (y1 - y2 * r_d) / np.sqrt(y1 + y2 * r_d**2)
            dist = 'normal'
        elif method in ['score-log']:
            stat = (np.log(y1 / y2) - np.log(r_d))
            stat /= np.sqrt((2 + 1 / r_d + r_d) / (y1 + y2))
            dist = 'normal'
        elif method in ['wald-log']:
            stat = (np.log(y1 / y2) - np.log(r_d)) / np.sqrt(1 / y1 + 1 / y2)
            dist = 'normal'
        elif method in ['sqrt']:
            stat = 2 * (np.sqrt(y1 + 3 / 8.) - np.sqrt((y2 + 3 / 8.) * r_d))
            stat /= np.sqrt(1 + r_d)
            dist = 'normal'
        elif method in ['exact-cond', 'cond-midp']:
            from statsmodels.stats import proportion
            bp = r_d / (1 + r_d)
            y_total = y1 + y2
            stat = np.nan
            # TODO: why y2 in here and not y1, check definition of H1 "larger"
            pvalue = proportion.binom_test(y1, y_total, prop=bp,
                                           alternative=alternative)
            if method in ['cond-midp']:
                # not inplace in case we still want binom pvalue
                pvalue = pvalue - 0.5 * stats.binom.pmf(y1, y_total, bp)

            dist = 'binomial'
        elif method.startswith('etest'):
            if method.endswith('wald'):
                method_etest = 'wald'
            else:
                method_etest = 'score'
            if etest_kwds is None:
                etest_kwds = {}

            stat, pvalue = etest_poisson_2indep(
                count1, exposure1, count2, exposure2, value=value,
                method=method_etest, alternative=alternative, **etest_kwds)

            dist = 'poisson'
        else:
            raise ValueError(f'method "{method}" not recognized')

    elif compare == "diff":
        if value is None:
            value = 0
        if method in ['wald']:
            stat = (rate1 - rate2 - value) / np.sqrt(rate1 / n1 + rate2 / n2)
            dist = 'normal'
            "waldccv"
        elif method in ['waldccv']:
            stat = (rate1 - rate2 - value)
            stat /= np.sqrt((count1 + 0.5) / n1**2 + (count2 + 0.5) / n2**2)
            dist = 'normal'
        elif method in ['score']:
            # estimate rates with constraint MLE
            count_pooled = y1 + y2
            rate_pooled = count_pooled / (n1 + n2)
            dt = rate_pooled - value
            r2_cmle = 0.5 * (dt + np.sqrt(dt**2 + 4 * value * y2 / (n1 + n2)))
            r1_cmle = r2_cmle + value

            stat = ((rate1 - rate2 - value) /
                    np.sqrt(r1_cmle / n1 + r2_cmle / n2))
            rates_cmle = (r1_cmle, r2_cmle)
            dist = 'normal'
        elif method.startswith('etest'):
            if method.endswith('wald'):
                method_etest = 'wald'
            else:
                method_etest = 'score'
                if method == "etest":
                    method = method + "-score"

            if etest_kwds is None:
                etest_kwds = {}

            stat, pvalue = etest_poisson_2indep(
                count1, exposure1, count2, exposure2, value=value,
                method=method_etest, compare="diff",
                alternative=alternative, **etest_kwds)

            dist = 'poisson'
        else:
            raise ValueError(f'method "{method}" not recognized')
    else:
        raise NotImplementedError('"compare" needs to be ratio or diff')

    if dist == 'normal':
        stat, pvalue = _zstat_generic2(stat, 1, alternative)

    rates = (rate1, rate2)
    ratio = rate1 / rate2
    diff = rate1 - rate2
    res = HolderTuple(statistic=stat,
                      pvalue=pvalue,
                      distribution=dist,
                      compare=compare,
                      method=method,
                      alternative=alternative,
                      rates=rates,
                      ratio=ratio,
                      diff=diff,
                      value=value,
                      rates_cmle=rates_cmle,
                      ratio_null=ratio_null,
                      )
    return res


def _score_diff(y1, n1, y2, n2, value=0, return_cmle=False):
    """score test and cmle for difference of 2 independent poisson rates

    """
    count_pooled = y1 + y2
    rate1, rate2 = y1 / n1, y2 / n2
    rate_pooled = count_pooled / (n1 + n2)
    dt = rate_pooled - value
    r2_cmle = 0.5 * (dt + np.sqrt(dt**2 + 4 * value * y2 / (n1 + n2)))
    r1_cmle = r2_cmle + value
    eps = 1e-20  # avoid zero division in stat_func
    v = r1_cmle / n1 + r2_cmle / n2
    stat = (rate1 - rate2 - value) / np.sqrt(v + eps)

    if return_cmle:
        return stat, r1_cmle, r2_cmle
    else:
        return stat


def etest_poisson_2indep(count1, exposure1, count2, exposure2, ratio_null=None,
                         value=None, method='score', compare="ratio",
                         alternative='two-sided', ygrid=None,
                         y_grid=None):
    """
    E-test for ratio of two sample Poisson rates.

    Rates are defined as expected count divided by exposure. The Null and
    alternative hypothesis for the rates, rate1 and rate2, of two independent
    Poisson samples are:

    for compare = 'diff'

    - H0: rate1 - rate2 - value = 0
    - H1: rate1 - rate2 - value != 0  if alternative = 'two-sided'
    - H1: rate1 - rate2 - value > 0   if alternative = 'larger'
    - H1: rate1 - rate2 - value < 0   if alternative = 'smaller'

    for compare = 'ratio'

    - H0: rate1 / rate2 - value = 0
    - H1: rate1 / rate2 - value != 0  if alternative = 'two-sided'
    - H1: rate1 / rate2 - value > 0   if alternative = 'larger'
    - H1: rate1 / rate2 - value < 0   if alternative = 'smaller'

    Parameters
    ----------
    count1 : int
        Number of events in first sample
    exposure1 : float
        Total exposure (time * subjects) in first sample
    count2 : int
        Number of events in first sample
    exposure2 : float
        Total exposure (time * subjects) in first sample
    ratio_null: float
        Ratio of the two Poisson rates under the Null hypothesis. Default is 1.
        Deprecated, use ``value`` instead.

        .. deprecated:: 0.14.0

            Use ``value`` instead.

    value : float
        Value of the ratio or diff of 2 independent rates under the null
        hypothesis. Default is equal rates, i.e. 1 for ratio and 0 for diff.

        .. versionadded:: 0.14.0

            Replacement for ``ratio_null``.

    method : {"score", "wald"}
        Method for the test statistic that defines the rejection region.
    alternative : string
        The alternative hypothesis, H1, has to be one of the following

        - 'two-sided': H1: ratio of rates is not equal to ratio_null (default)
        - 'larger' :   H1: ratio of rates is larger than ratio_null
        - 'smaller' :  H1: ratio of rates is smaller than ratio_null

    y_grid : None or 1-D ndarray
        Grid values for counts of the Poisson distribution used for computing
        the pvalue. By default truncation is based on an upper tail Poisson
        quantiles.

    ygrid : None or 1-D ndarray
        Same as y_grid. Deprecated. If both y_grid and ygrid are provided,
        ygrid will be ignored.

        .. deprecated:: 0.14.0

            Use ``y_grid`` instead.

    Returns
    -------
    stat_sample : float
        test statistic for the sample
    pvalue : float

    References
    ----------
    Gu, Ng, Tang, Schucany 2008: Testing the Ratio of Two Poisson Rates,
    Biometrical Journal 50 (2008) 2, 2008
    Ng, H. K. T., K. Gu, and M. L. Tang. 2007. “A Comparative Study of Tests
    for the Difference of Two Poisson Means.” Computational Statistics & Data
    Analysis 51 (6): 3085–99. https://doi.org/10.1016/j.csda.2006.02.004.

    """
    y1, n1, y2, n2 = map(np.asarray, [count1, exposure1, count2, exposure2])
    d = n2 / n1

    eps = 1e-20  # avoid zero division in stat_func

    if compare == "ratio":
        if ratio_null is None and value is None:
            # default value
            value = 1
        elif ratio_null is not None:
            warnings.warn("'ratio_null' is deprecated, use 'value' keyword",
                          FutureWarning)
            value = ratio_null

        r = value  # rate1 / rate2
        r_d = r / d
        rate2_cmle = (y1 + y2) / n2 / (1 + r_d)
        rate1_cmle = rate2_cmle * r

        if method in ['score']:
            def stat_func(x1, x2):
                return (x1 - x2 * r_d) / np.sqrt((x1 + x2) * r_d + eps)
            # TODO: do I need these? return_results ?
            # rate2_cmle = (y1 + y2) / n2 / (1 + r_d)
            # rate1_cmle = rate2_cmle * r
            # rate1 = rate1_cmle
            # rate2 = rate2_cmle
        elif method in ['wald']:
            def stat_func(x1, x2):
                return (x1 - x2 * r_d) / np.sqrt(x1 + x2 * r_d**2 + eps)
            # rate2_mle = y2 / n2
            # rate1_mle = y1 / n1
            # rate1 = rate1_mle
            # rate2 = rate2_mle
        else:
            raise ValueError('method not recognized')

    elif compare == "diff":
        if value is None:
            value = 0
        tmp = _score_diff(y1, n1, y2, n2, value=value, return_cmle=True)
        _, rate1_cmle, rate2_cmle = tmp

        if method in ['score']:

            def stat_func(x1, x2):
                return _score_diff(x1, n1, x2, n2, value=value)

        elif method in ['wald']:

            def stat_func(x1, x2):
                rate1, rate2 = x1 / n1, x2 / n2
                stat = (rate1 - rate2 - value)
                stat /= np.sqrt(rate1 / n1 + rate2 / n2 + eps)
                return stat

        else:
            raise ValueError('method not recognized')

    # The sampling distribution needs to be based on the null hypotheis
    # use constrained MLE from 'score' calculation
    rate1 = rate1_cmle
    rate2 = rate2_cmle
    mean1 = n1 * rate1
    mean2 = n2 * rate2

    stat_sample = stat_func(y1, y2)

    if ygrid is not None:
        warnings.warn("ygrid is deprecated, use y_grid", FutureWarning)
    y_grid = y_grid if y_grid is not None else ygrid

    # The following uses a fixed truncation for evaluating the probabilities
    # It will currently only work for small counts, so that sf at truncation
    # point is small
    # We can make it depend on the amount of truncated sf.
    # Some numerical optimization or checks for large means need to be added.
    if y_grid is None:
        threshold = stats.poisson.isf(1e-13, max(mean1, mean2))
        threshold = max(threshold, 100)   # keep at least 100
        y_grid = np.arange(threshold + 1)
    else:
        y_grid = np.asarray(y_grid)
        if y_grid.ndim != 1:
            raise ValueError("y_grid needs to be None or 1-dimensional array")
    pdf1 = stats.poisson.pmf(y_grid, mean1)
    pdf2 = stats.poisson.pmf(y_grid, mean2)

    stat_space = stat_func(y_grid[:, None], y_grid[None, :])  # broadcasting
    eps = 1e-15   # correction for strict inequality check

    if alternative in ['two-sided', '2-sided', '2s']:
        mask = np.abs(stat_space) >= (np.abs(stat_sample) - eps)
    elif alternative in ['larger', 'l']:
        mask = stat_space >= (stat_sample - eps)
    elif alternative in ['smaller', 's']:
        mask = stat_space <= (stat_sample + eps)
    else:
        raise ValueError('invalid alternative')

    pvalue = ((pdf1[:, None] * pdf2[None, :])[mask]).sum()
    return stat_sample, pvalue


def tost_poisson_2indep(count1, exposure1, count2, exposure2, low, upp,
                        method='score', compare='ratio'):
    '''Equivalence test based on two one-sided `test_proportions_2indep`

    This assumes that we have two independent poisson samples.

    The Null and alternative hypothesis for equivalence testing are

    for compare = 'ratio'

    - H0: rate1 / rate2 <= low or upp <= rate1 / rate2
    - H1: low < rate1 / rate2 < upp

    for compare = 'diff'

    - H0: rate1 - rate2 <= low or upp <= rate1 - rate2
    - H1: low < rate - rate < upp

    Parameters
    ----------
    count1 : int
        Number of events in first sample
    exposure1 : float
        Total exposure (time * subjects) in first sample
    count2 : int
        Number of events in second sample
    exposure2 : float
        Total exposure (time * subjects) in second sample
    low, upp :
        equivalence margin for the ratio or difference of Poisson rates
    method: string
        TOST uses ``test_poisson_2indep`` and has the same methods.

        ratio:

        - 'wald': method W1A, wald test, variance based on observed rates
        - 'score': method W2A, score test, variance based on estimate under
          the Null hypothesis
        - 'wald-log': W3A, uses log-ratio, variance based on observed rates
        - 'score-log' W4A, uses log-ratio, variance based on estimate under
          the Null hypothesis
        - 'sqrt': W5A, based on variance stabilizing square root transformation
        - 'exact-cond': exact conditional test based on binomial distribution
           This uses ``binom_test`` which is minlike in the two-sided case.
        - 'cond-midp': midpoint-pvalue of exact conditional test
        - 'etest' or 'etest-score: etest with score test statistic
        - 'etest-wald': etest with wald test statistic

        diff:

        - 'wald',
        - 'waldccv'
        - 'score'
        - 'etest-score' or 'etest: etest with score test statistic
        - 'etest-wald': etest with wald test statistic

    Returns
    -------
    results : instance of HolderTuple class
        The two main attributes are test statistic `statistic` and p-value
        `pvalue`.

    References
    ----------
    Gu, Ng, Tang, Schucany 2008: Testing the Ratio of Two Poisson Rates,
    Biometrical Journal 50 (2008) 2, 2008

    See Also
    --------
    test_poisson_2indep
    confint_poisson_2indep
    '''

    tt1 = test_poisson_2indep(count1, exposure1, count2, exposure2,
                              value=low, method=method,
                              compare=compare,
                              alternative='larger')
    tt2 = test_poisson_2indep(count1, exposure1, count2, exposure2,
                              value=upp, method=method,
                              compare=compare,
                              alternative='smaller')

    # idx_max = 1 if t1.pvalue < t2.pvalue else 0
    idx_max = np.asarray(tt1.pvalue < tt2.pvalue, int)
    statistic = np.choose(idx_max, [tt1.statistic, tt2.statistic])
    pvalue = np.choose(idx_max, [tt1.pvalue, tt2.pvalue])

    res = HolderTuple(statistic=statistic,
                      pvalue=pvalue,
                      method=method,
                      compare=compare,
                      equiv_limits=(low, upp),
                      results_larger=tt1,
                      results_smaller=tt2,
                      title="Equivalence test for 2 independent Poisson rates"
                      )

    return res


def nonequivalence_poisson_2indep(count1, exposure1, count2, exposure2,
                                  low, upp, method='score', compare="ratio"):
    """Test for non-equivalence, minimum effect for poisson.

    This reverses null and alternative hypothesis compared to equivalence
    testing. The null hypothesis is that the effect, ratio (or diff), is in
    an interval that specifies a range of irrelevant or unimportant
    differences between the two samples.

    The Null and alternative hypothesis comparing the ratio of rates are

    for compare = 'ratio':

    - H0: low < rate1 / rate2 < upp
    - H1: rate1 / rate2 <= low or upp <= rate1 / rate2

    for compare = 'diff':

    - H0: rate1 - rate2 <= low or upp <= rate1 - rate2
    - H1: low < rate - rate < upp


    Notes
    -----
    This is implemented as two one-sided tests at the minimum effect boundaries
    (low, upp) with (nominal) size alpha / 2 each.
    The size of the test is the sum of the two one-tailed tests, which
    corresponds to an equal-tailed two-sided test.
    If low and upp are equal, then the result is the same as the standard
    two-sided test.

    The p-value is computed as `2 * min(pvalue_low, pvalue_upp)` in analogy to
    two-sided equal-tail tests.

    In large samples the nominal size of the test will be below alpha.

    References
    ----------
    .. [1] Hodges, J. L., Jr., and E. L. Lehmann. 1954. Testing the Approximate
       Validity of Statistical Hypotheses. Journal of the Royal Statistical
       Society, Series B (Methodological) 16: 261–68.

    .. [2] Kim, Jae H., and Andrew P. Robinson. 2019. “Interval-Based
       Hypothesis Testing and Its Applications to Economics and Finance.”
       Econometrics 7 (2): 21. https://doi.org/10.3390/econometrics7020021.

    """
    tt1 = test_poisson_2indep(count1, exposure1, count2, exposure2,
                              value=low, method=method, compare=compare,
                              alternative='smaller')
    tt2 = test_poisson_2indep(count1, exposure1, count2, exposure2,
                              value=upp, method=method, compare=compare,
                              alternative='larger')

    # idx_min = 0 if tt1.pvalue < tt2.pvalue else 1
    idx_min = np.asarray(tt1.pvalue < tt2.pvalue, int)
    pvalue = 2 * np.minimum(tt1.pvalue, tt2.pvalue)
    statistic = np.choose(idx_min, [tt1.statistic, tt2.statistic])
    res = HolderTuple(statistic=statistic,
                      pvalue=pvalue,
                      method=method,
                      results_larger=tt1,
                      results_smaller=tt2,
                      title="Equivalence test for 2 independent Poisson rates"
                      )

    return res


def confint_poisson_2indep(count1, exposure1, count2, exposure2,
                           method='score', compare='ratio', alpha=0.05,
                           method_mover="score",
                           ):
    """Confidence interval for ratio or difference of 2 indep poisson rates.

    Parameters
    ----------
    count1 : int
        Number of events in first sample.
    exposure1 : float
        Total exposure (time * subjects) in first sample.
    count2 : int
        Number of events in second sample.
    exposure2 : float
        Total exposure (time * subjects) in second sample.
    method : string
        Method for the test statistic and the p-value. Defaults to `'score'`.
        see Notes.

        ratio:

        - 'wald': NOT YET, method W1A, wald test, variance based on observed
          rates
        - 'waldcc' :
        - 'score': method W2A, score test, variance based on estimate under
          the Null hypothesis
        - 'wald-log': W3A, uses log-ratio, variance based on observed rates
        - 'score-log' W4A, uses log-ratio, variance based on estimate under
          the Null hypothesis
        - 'sqrt': W5A, based on variance stabilizing square root transformation
        - 'sqrtcc' :
        - 'exact-cond': NOT YET, exact conditional test based on binomial
          distribution
          This uses ``binom_test`` which is minlike in the two-sided case.
        - 'cond-midp': NOT YET, midpoint-pvalue of exact conditional test
        - 'mover' :

        diff:

        - 'wald',
        - 'waldccv'
        - 'score'
        - 'mover'

    compare : {'diff', 'ratio'}
        Default is "ratio".
        If compare is `diff`, then the hypothesis test is for
        diff = rate1 - rate2.
        If compare is `ratio`, then the hypothesis test is for the
        rate ratio defined by ratio = rate1 / rate2.
    alternative : string
        The alternative hypothesis, H1, has to be one of the following

        - 'two-sided': H1: ratio of rates is not equal to ratio_null (default)
        - 'larger' :   H1: ratio of rates is larger than ratio_null
        - 'smaller' :  H1: ratio of rates is smaller than ratio_null

    alpha : float in (0, 1)
        Significance level, nominal coverage of the confidence interval is
        1 - alpha.

    Returns
    -------
    tuple (low, upp) : confidence limits.

    """

    # shortcut names
    y1, n1, y2, n2 = map(np.asarray, [count1, exposure1, count2, exposure2])
    rate1, rate2 = y1 / n1, y2 / n2
    alpha = alpha / 2  # two-sided only

    if compare == "ratio":

        if method == "score":
            low, upp = _invert_test_confint_2indep(
                count1, exposure1, count2, exposure2,
                alpha=alpha * 2,   # check how alpha is defined
                method="score",
                compare="ratio",
                method_start="waldcc"
                )
            ci = (low, upp)

        elif method == "wald-log":
            crit = stats.norm.isf(alpha)
            c = 0
            center = (count1 + c) / (count2 + c) * n2 / n1
            std = np.sqrt(1 / (count1 + c) + 1 / (count2 + c))

            ci = (center * np.exp(- crit * std), center * np.exp(crit * std))

        elif method == "score-log":
            low, upp = _invert_test_confint_2indep(
                count1, exposure1, count2, exposure2,
                alpha=alpha * 2,   # check how alpha is defined
                method="score-log",
                compare="ratio",
                method_start="waldcc"
                )
            ci = (low, upp)

        elif method == "waldcc":
            crit = stats.norm.isf(alpha)
            center = (count1 + 0.5) / (count2 + 0.5) * n2 / n1
            std = np.sqrt(1 / (count1 + 0.5) + 1 / (count2 + 0.5))

            ci = (center * np.exp(- crit * std), center * np.exp(crit * std))

        elif method == "sqrtcc":
            # coded based on Price, Bonett 2000 equ (2.4)
            crit = stats.norm.isf(alpha)
            center = np.sqrt((count1 + 0.5) * (count2 + 0.5))
            std = 0.5 * np.sqrt(count1 + 0.5 + count2 + 0.5 - 0.25 * crit)
            denom = (count2 + 0.5 - 0.25 * crit**2)

            low_sqrt = (center - crit * std) / denom
            upp_sqrt = (center + crit * std) / denom

            ci = (low_sqrt**2, upp_sqrt**2)

        elif method == "mover":
            method_p = method_mover
            ci1 = confint_poisson(y1, n1, method=method_p, alpha=2*alpha)
            ci2 = confint_poisson(y2, n2, method=method_p, alpha=2*alpha)

            ci = _mover_confint(rate1, rate2, ci1, ci2, contrast="ratio")

        else:
            raise ValueError(f'method "{method}" not recognized')

        ci = (np.maximum(ci[0], 0), ci[1])

    elif compare == "diff":

        if method in ['wald']:
            crit = stats.norm.isf(alpha)
            center = rate1 - rate2
            half = crit * np.sqrt(rate1 / n1 + rate2 / n2)
            ci = center - half, center + half

        elif method in ['waldccv']:
            crit = stats.norm.isf(alpha)
            center = rate1 - rate2
            std = np.sqrt((count1 + 0.5) / n1**2 + (count2 + 0.5) / n2**2)
            half = crit * std
            ci = center - half, center + half

        elif method == "score":
            low, upp = _invert_test_confint_2indep(
                count1, exposure1, count2, exposure2,
                alpha=alpha * 2,   # check how alpha is defined
                method="score",
                compare="diff",
                method_start="waldccv"
                )
            ci = (low, upp)

        elif method == "mover":
            method_p = method_mover
            ci1 = confint_poisson(y1, n1, method=method_p, alpha=2*alpha)
            ci2 = confint_poisson(y2, n2, method=method_p, alpha=2*alpha)

            ci = _mover_confint(rate1, rate2, ci1, ci2, contrast="diff")
        else:
            raise ValueError(f'method "{method}" not recognized')
    else:
        raise NotImplementedError('"compare" needs to be ratio or diff')

    return ci


def power_poisson_ratio_2indep(
        rate1, rate2, nobs1,
        nobs_ratio=1,
        exposure=1,
        value=0,
        alpha=0.05,
        dispersion=1,
        alternative="smaller",
        method_var="alt",
        return_results=True,
        ):
    """Power of test of ratio of 2 independent poisson rates.

    This is based on Zhu and Zhu and Lakkis. It does not directly correspond
    to `test_poisson_2indep`.

    Parameters
    ----------
    rate1 : float
        Poisson rate for the first sample, treatment group, under the
        alternative hypothesis.
    rate2 : float
        Poisson rate for the second sample, reference group, under the
        alternative hypothesis.
    nobs1 : float or int
        Number of observations in sample 1.
    nobs_ratio : float
        Sample size ratio, nobs2 = nobs_ratio * nobs1.
    exposure : float
        Exposure for each observation. Total exposure is nobs1 * exposure
        and nobs2 * exposure.
    alpha : float in interval (0,1)
        Significance level, e.g. 0.05, is the probability of a type I
        error, that is wrong rejections if the Null Hypothesis is true.
    value : float
        Rate ratio, rate1 / rate2, under the null hypothesis.
    dispersion : float
        Dispersion coefficient for quasi-Poisson. Dispersion different from
        one can capture over or under dispersion relative to Poisson
        distribution.
    method_var : {"score", "alt"}
        The variance of the test statistic for the null hypothesis given the
        rates under the alternative can be either equal to the rates under the
        alternative ``method_var="alt"``, or estimated under the constrained
        of the null hypothesis, ``method_var="score"``.
    alternative : string, 'two-sided' (default), 'larger', 'smaller'
        Alternative hypothesis whether the power is calculated for a
        two-sided (default) or one sided test. The one-sided test can be
        either 'larger', 'smaller'.
    return_results : bool
        If true, then a results instance with extra information is returned,
        otherwise only the computed power is returned.

    Returns
    -------
    results : results instance or float
        If return_results is False, then only the power is returned.
        If return_results is True, then a results instance with the
        information in attributes is returned.

        power : float
            Power of the test, e.g. 0.8, is one minus the probability of a
            type II error. Power is the probability that the test correctly
            rejects the Null Hypothesis if the Alternative Hypothesis is true.

        Other attributes in results instance include :

        std_null
            standard error of difference under the null hypothesis (without
            sqrt(nobs1))
        std_alt
            standard error of difference under the alternative hypothesis
            (without sqrt(nobs1))

    References
    ----------
    .. [1] Zhu, Haiyuan. 2017. “Sample Size Calculation for Comparing Two
       Poisson or Negative Binomial Rates in Noninferiority or Equivalence
       Trials.” Statistics in Biopharmaceutical Research, March.
       https://doi.org/10.1080/19466315.2016.1225594
    .. [2] Zhu, Haiyuan, and Hassan Lakkis. 2014. “Sample Size Calculation for
       Comparing Two Negative Binomial Rates.” Statistics in Medicine 33 (3):
       376–87. https://doi.org/10.1002/sim.5947.
    .. [3] PASS documentation
    """
    # TODO: avoid possible circular import, check if needed
    from statsmodels.stats.power import normal_power_het

    rate1, rate2, nobs1 = map(np.asarray, [rate1, rate2, nobs1])

    nobs2 = nobs_ratio * nobs1
    v1 = dispersion / exposure * (1 / rate1 + 1 / (nobs_ratio * rate2))
    if method_var == "alt":
        v0 = v1
    elif method_var == "score":
        # nobs_ratio = 1 / nobs_ratio
        v0 = dispersion / exposure * (1 + value / nobs_ratio)**2
        v0 /= value / nobs_ratio * (rate1 + (nobs_ratio * rate2))
    else:
        raise NotImplementedError(f"method_var {method_var} not recognized")

    std_null = np.sqrt(v0)
    std_alt = np.sqrt(v1)
    es = np.log(rate1 / rate2) - np.log(value)

    pow_ = normal_power_het(es, nobs1, alpha, std_null=std_null,
                            std_alternative=std_alt,
                            alternative=alternative)

    p_pooled = None  # TODO: replace or remove

    if return_results:
        res = HolderTuple(
            power=pow_,
            p_pooled=p_pooled,
            std_null=std_null,
            std_alt=std_alt,
            nobs1=nobs1,
            nobs2=nobs2,
            nobs_ratio=nobs_ratio,
            alpha=alpha,
            tuple_=("power",),  # override default
            )
        return res

    return pow_


def power_equivalence_poisson_2indep(rate1, rate2, nobs1,
                                     low, upp, nobs_ratio=1,
                                     exposure=1, alpha=0.05, dispersion=1,
                                     method_var="alt",
                                     return_results=False):
    """Power of equivalence test of ratio of 2 independent poisson rates.

    Parameters
    ----------
    rate1 : float
        Poisson rate for the first sample, treatment group, under the
        alternative hypothesis.
    rate2 : float
        Poisson rate for the second sample, reference group, under the
        alternative hypothesis.
    nobs1 : float or int
        Number of observations in sample 1.
    low : float
        Lower equivalence margin for the rate ratio, rate1 / rate2.
    upp : float
        Upper equivalence margin for the rate ratio, rate1 / rate2.
    nobs_ratio : float
        Sample size ratio, nobs2 = nobs_ratio * nobs1.
    exposure : float
        Exposure for each observation. Total exposure is nobs1 * exposure
        and nobs2 * exposure.
    alpha : float in interval (0,1)
        Significance level, e.g. 0.05, is the probability of a type I
        error, that is wrong rejections if the Null Hypothesis is true.
    value : float
        Difference between rates 1 and 2 under the null hypothesis.
    method_var : {"score", "alt"}
        The variance of the test statistic for the null hypothesis given the
        rates uder the alternative, can be either equal to the rates under the
        alternative ``method_var="alt"``, or estimated under the constrained
        of the null hypothesis, ``method_var="score"``.
    alternative : string, 'two-sided' (default), 'larger', 'smaller'
        Alternative hypothesis whether the power is calculated for a
        two-sided (default) or one sided test. The one-sided test can be
        either 'larger', 'smaller'.
    return_results : bool
        If true, then a results instance with extra information is returned,
        otherwise only the computed power is returned.

    Returns
    -------
    results : results instance or float
        If return_results is False, then only the power is returned.
        If return_results is True, then a results instance with the
        information in attributes is returned.

        power : float
            Power of the test, e.g. 0.8, is one minus the probability of a
            type II error. Power is the probability that the test correctly
            rejects the Null Hypothesis if the Alternative Hypothesis is true.

        Other attributes in results instance include :

        std_null
            standard error of difference under the null hypothesis (without
            sqrt(nobs1))
        std_alt
            standard error of difference under the alternative hypothesis
            (without sqrt(nobs1))

    References
    ----------
    .. [1] Zhu, Haiyuan. 2017. “Sample Size Calculation for Comparing Two
       Poisson or Negative Binomial Rates in Noninferiority or Equivalence
       Trials.” Statistics in Biopharmaceutical Research, March.
       https://doi.org/10.1080/19466315.2016.1225594
    .. [2] Zhu, Haiyuan, and Hassan Lakkis. 2014. “Sample Size Calculation for
       Comparing Two Negative Binomial Rates.” Statistics in Medicine 33 (3):
       376–87. https://doi.org/10.1002/sim.5947.
    .. [3] PASS documentation
    """
    rate1, rate2, nobs1 = map(np.asarray, [rate1, rate2, nobs1])

    nobs2 = nobs_ratio * nobs1
    v1 = dispersion / exposure * (1 / rate1 + 1 / (nobs_ratio * rate2))

    if method_var == "alt":
        v0_low = v0_upp = v1
    elif method_var == "score":
        v0_low = dispersion / exposure * (1 + low * nobs_ratio)**2
        v0_low /= low * nobs_ratio * (rate1 + (nobs_ratio * rate2))
        v0_upp = dispersion / exposure * (1 + upp * nobs_ratio)**2
        v0_upp /= upp * nobs_ratio * (rate1 + (nobs_ratio * rate2))
    else:
        raise NotImplementedError(f"method_var {method_var} not recognized")

    es_low = np.log(rate1 / rate2) - np.log(low)
    es_upp = np.log(rate1 / rate2) - np.log(upp)
    std_null_low = np.sqrt(v0_low)
    std_null_upp = np.sqrt(v0_upp)
    std_alternative = np.sqrt(v1)

    pow_ = _power_equivalence_het(es_low, es_upp, nobs2, alpha=alpha,
                                  std_null_low=std_null_low,
                                  std_null_upp=std_null_upp,
                                  std_alternative=std_alternative)

    if return_results:
        res = HolderTuple(
            power=pow_[0],
            power_margins=pow[1:],
            std_null_low=std_null_low,
            std_null_upp=std_null_upp,
            std_alt=std_alternative,
            nobs1=nobs1,
            nobs2=nobs2,
            nobs_ratio=nobs_ratio,
            alpha=alpha,
            tuple_=("power",),  # override default
            )
        return res
    else:
        return pow_[0]


def _power_equivalence_het_v0(es_low, es_upp, nobs, alpha=0.05,
                              std_null_low=None,
                              std_null_upp=None,
                              std_alternative=None):
    """power for equivalence test

    """

    s0_low = std_null_low
    s0_upp = std_null_upp
    s1 = std_alternative

    crit = norm.isf(alpha)
    pow_ = (
        norm.cdf((np.sqrt(nobs) * es_low - crit * s0_low) / s1) +
        norm.cdf((np.sqrt(nobs) * es_upp - crit * s0_upp) / s1) - 1
        )
    return pow_


def _power_equivalence_het(es_low, es_upp, nobs, alpha=0.05,
                           std_null_low=None,
                           std_null_upp=None,
                           std_alternative=None):
    """power for equivalence test

    """

    s0_low = std_null_low
    s0_upp = std_null_upp
    s1 = std_alternative

    crit = norm.isf(alpha)

    # Note: rejection region is an interval [low, upp]
    # Here we compute the complement of the two tail probabilities
    p1 = norm.sf((np.sqrt(nobs) * es_low - crit * s0_low) / s1)
    p2 = norm.cdf((np.sqrt(nobs) * es_upp + crit * s0_upp) / s1)
    pow_ = 1 - (p1 + p2)
    return pow_, p1, p2


def _std_2poisson_power(
        rate1, rate2, nobs_ratio=1, alpha=0.05,
        exposure=1,
        dispersion=1,
        value=0,
        method_var="score",
        ):
    rates_pooled = (rate1 + rate2 * nobs_ratio) / (1 + nobs_ratio)
    # v1 = dispersion / exposure * (1 / rate2 + 1 / (nobs_ratio * rate1))
    if method_var == "alt":
        v0 = v1 = rate1 + rate2 / nobs_ratio
    else:
        # uaw n1 = 1 as normalization
        _, r1_cmle, r2_cmle = _score_diff(
            rate1, 1, rate2 * nobs_ratio, nobs_ratio, value=value,
            return_cmle=True)
        v1 = rate1 + rate2 / nobs_ratio
        v0 = r1_cmle + r2_cmle / nobs_ratio
    return rates_pooled, np.sqrt(v0), np.sqrt(v1)


def power_poisson_diff_2indep(rate1, rate2, nobs1, nobs_ratio=1, alpha=0.05,
                              value=0,
                              method_var="score",
                              alternative='two-sided',
                              return_results=True):
    """Power of ztest for the difference between two independent poisson rates.

    Parameters
    ----------
    rate1 : float
        Poisson rate for the first sample, treatment group, under the
        alternative hypothesis.
    rate2 : float
        Poisson rate for the second sample, reference group, under the
        alternative hypothesis.
    nobs1 : float or int
        Number of observations in sample 1.
    nobs_ratio : float
        Sample size ratio, nobs2 = nobs_ratio * nobs1.
    alpha : float in interval (0,1)
        Significance level, e.g. 0.05, is the probability of a type I
        error, that is wrong rejections if the Null Hypothesis is true.
    value : float
        Difference between rates 1 and 2 under the null hypothesis.
    method_var : {"score", "alt"}
        The variance of the test statistic for the null hypothesis given the
        rates uder the alternative, can be either equal to the rates under the
        alternative ``method_var="alt"``, or estimated under the constrained
        of the null hypothesis, ``method_var="score"``.
    alternative : string, 'two-sided' (default), 'larger', 'smaller'
        Alternative hypothesis whether the power is calculated for a
        two-sided (default) or one sided test. The one-sided test can be
        either 'larger', 'smaller'.
    return_results : bool
        If true, then a results instance with extra information is returned,
        otherwise only the computed power is returned.

    Returns
    -------
    results : results instance or float
        If return_results is False, then only the power is returned.
        If return_results is True, then a results instance with the
        information in attributes is returned.

        power : float
            Power of the test, e.g. 0.8, is one minus the probability of a
            type II error. Power is the probability that the test correctly
            rejects the Null Hypothesis if the Alternative Hypothesis is true.

        Other attributes in results instance include :

        std_null
            standard error of difference under the null hypothesis (without
            sqrt(nobs1))
        std_alt
            standard error of difference under the alternative hypothesis
            (without sqrt(nobs1))

    References
    ----------
    .. [1] Stucke, Kathrin, and Meinhard Kieser. 2013. “Sample Size
       Calculations for Noninferiority Trials with Poisson Distributed Count
       Data.” Biometrical Journal 55 (2): 203–16.
       https://doi.org/10.1002/bimj.201200142.
    .. [2] PASS manual chapter 436

    """
    # TODO: avoid possible circular import, check if needed
    from statsmodels.stats.power import normal_power_het

    rate1, rate2, nobs1 = map(np.asarray, [rate1, rate2, nobs1])

    diff = rate1 - rate2
    _, std_null, std_alt = _std_2poisson_power(
        rate1,
        rate2,
        nobs_ratio=nobs_ratio,
        alpha=alpha,
        value=value,
        method_var=method_var,
        )

    pow_ = normal_power_het(diff - value, nobs1, alpha, std_null=std_null,
                            std_alternative=std_alt,
                            alternative=alternative)

    if return_results:
        res = HolderTuple(
            power=pow_,
            rates_alt=(rate2 + diff, rate2),
            std_null=std_null,
            std_alt=std_alt,
            nobs1=nobs1,
            nobs2=nobs_ratio * nobs1,
            nobs_ratio=nobs_ratio,
            alpha=alpha,
            tuple_=("power",),  # override default
            )
        return res
    else:
        return pow_


def _var_cmle_negbin(rate1, rate2, nobs_ratio, exposure=1, value=1,
                     dispersion=0):
    """
    variance based on constrained cmle, for score test version

    for ratio comparison of two negative binomial samples

    value = rate1 / rate2 under the null
    """
    # definitions in Zhu
    # nobs_ratio = n1 / n0
    # value = ratio = r1 / r0
    rate0 = rate2  # control
    nobs_ratio = 1 / nobs_ratio

    a = - dispersion * exposure * value * (1 + nobs_ratio)
    b = (dispersion * exposure * (rate0 * value + nobs_ratio * rate1) -
         (1 + nobs_ratio * value))
    c = rate0 + nobs_ratio * rate1
    if dispersion == 0:
        r0 = -c / b
    else:
        r0 = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)
    r1 = r0 * value
    v = (1 / exposure / r0 * (1 + 1 / value / nobs_ratio) +
         (1 + nobs_ratio) / nobs_ratio * dispersion)

    r2 = r0
    return v * nobs_ratio, r1, r2


def power_negbin_ratio_2indep(
        rate1, rate2, nobs1,
        nobs_ratio=1,
        exposure=1,
        value=1,
        alpha=0.05,
        dispersion=0.01,
        alternative="two-sided",
        method_var="alt",
        return_results=True):
    """
    Power of test of ratio of 2 independent negative binomial rates.

    Parameters
    ----------
    rate1 : float
        Poisson rate for the first sample, treatment group, under the
        alternative hypothesis.
    rate2 : float
        Poisson rate for the second sample, reference group, under the
        alternative hypothesis.
    nobs1 : float or int
        Number of observations in sample 1.
    low : float
        Lower equivalence margin for the rate ratio, rate1 / rate2.
    upp : float
        Upper equivalence margin for the rate ratio, rate1 / rate2.
    nobs_ratio : float
        Sample size ratio, nobs2 = nobs_ratio * nobs1.
    exposure : float
        Exposure for each observation. Total exposure is nobs1 * exposure
        and nobs2 * exposure.
    value : float
        Rate ratio, rate1 / rate2, under the null hypothesis.
    alpha : float in interval (0,1)
        Significance level, e.g. 0.05, is the probability of a type I
        error, that is wrong rejections if the Null Hypothesis is true.
    dispersion : float >= 0.
        Dispersion parameter for Negative Binomial distribution.
        The Poisson limiting case corresponds to ``dispersion=0``.
    method_var : {"score", "alt"}
        The variance of the test statistic for the null hypothesis given the
        rates under the alternative, can be either equal to the rates under the
        alternative ``method_var="alt"``, or estimated under the constrained
        of the null hypothesis, ``method_var="score"``, or based on a moment
        constrained estimate, ``method_var="ftotal"``. see references.
    alternative : string, 'two-sided' (default), 'larger', 'smaller'
        Alternative hypothesis whether the power is calculated for a
        two-sided (default) or one sided test. The one-sided test can be
        either 'larger', 'smaller'.
    return_results : bool
        If true, then a results instance with extra information is returned,
        otherwise only the computed power is returned.

    Returns
    -------
    results : results instance or float
        If return_results is False, then only the power is returned.
        If return_results is True, then a results instance with the
        information in attributes is returned.

        power : float
            Power of the test, e.g. 0.8, is one minus the probability of a
            type II error. Power is the probability that the test correctly
            rejects the Null Hypothesis if the Alternative Hypothesis is true.

        Other attributes in results instance include :

        std_null
            standard error of difference under the null hypothesis (without
            sqrt(nobs1))
        std_alt
            standard error of difference under the alternative hypothesis
            (without sqrt(nobs1))

    References
    ----------
    .. [1] Zhu, Haiyuan. 2017. “Sample Size Calculation for Comparing Two
       Poisson or Negative Binomial Rates in Noninferiority or Equivalence
       Trials.” Statistics in Biopharmaceutical Research, March.
       https://doi.org/10.1080/19466315.2016.1225594
    .. [2] Zhu, Haiyuan, and Hassan Lakkis. 2014. “Sample Size Calculation for
       Comparing Two Negative Binomial Rates.” Statistics in Medicine 33 (3):
       376–87. https://doi.org/10.1002/sim.5947.
    .. [3] PASS documentation
    """
    # TODO: avoid possible circular import, check if needed
    from statsmodels.stats.power import normal_power_het

    rate1, rate2, nobs1 = map(np.asarray, [rate1, rate2, nobs1])

    nobs2 = nobs_ratio * nobs1
    v1 = ((1 / rate1 + 1 / (nobs_ratio * rate2)) / exposure +
          (1 + nobs_ratio) / nobs_ratio * dispersion)
    if method_var == "alt":
        v0 = v1
    elif method_var == "ftotal":
        v0 = (1 + value * nobs_ratio)**2 / (
             exposure * nobs_ratio * value * (rate1 + nobs_ratio * rate2))
        v0 += (1 + nobs_ratio) / nobs_ratio * dispersion
    elif method_var == "score":
        v0 = _var_cmle_negbin(rate1, rate2, nobs_ratio,
                              exposure=exposure, value=value,
                              dispersion=dispersion)[0]
    else:
        raise NotImplementedError(f"method_var {method_var} not recognized")

    std_null = np.sqrt(v0)
    std_alt = np.sqrt(v1)
    es = np.log(rate1 / rate2) - np.log(value)

    pow_ = normal_power_het(es, nobs1, alpha, std_null=std_null,
                            std_alternative=std_alt,
                            alternative=alternative)

    if return_results:
        res = HolderTuple(
            power=pow_,
            std_null=std_null,
            std_alt=std_alt,
            nobs1=nobs1,
            nobs2=nobs2,
            nobs_ratio=nobs_ratio,
            alpha=alpha,
            tuple_=("power",),  # override default
            )
        return res

    return pow_


def power_equivalence_neginb_2indep(rate1, rate2, nobs1,
                                    low, upp, nobs_ratio=1,
                                    exposure=1, alpha=0.05, dispersion=0,
                                    method_var="alt",
                                    return_results=False):
    """
    Power of equivalence test of ratio of 2 indep. negative binomial rates.

    Parameters
    ----------
    rate1 : float
        Poisson rate for the first sample, treatment group, under the
        alternative hypothesis.
    rate2 : float
        Poisson rate for the second sample, reference group, under the
        alternative hypothesis.
    nobs1 : float or int
        Number of observations in sample 1.
    low : float
        Lower equivalence margin for the rate ratio, rate1 / rate2.
    upp : float
        Upper equivalence margin for the rate ratio, rate1 / rate2.
    nobs_ratio : float
        Sample size ratio, nobs2 = nobs_ratio * nobs1.
    alpha : float in interval (0,1)
        Significance level, e.g. 0.05, is the probability of a type I
        error, that is wrong rejections if the Null Hypothesis is true.
    dispersion : float >= 0.
        Dispersion parameter for Negative Binomial distribution.
        The Poisson limiting case corresponds to ``dispersion=0``.
    method_var : {"score", "alt"}
        The variance of the test statistic for the null hypothesis given the
        rates under the alternative, can be either equal to the rates under the
        alternative ``method_var="alt"``, or estimated under the constrained
        of the null hypothesis, ``method_var="score"``, or based on a moment
        constrained estimate, ``method_var="ftotal"``. see references.
    alternative : string, 'two-sided' (default), 'larger', 'smaller'
        Alternative hypothesis whether the power is calculated for a
        two-sided (default) or one sided test. The one-sided test can be
        either 'larger', 'smaller'.
    return_results : bool
        If true, then a results instance with extra information is returned,
        otherwise only the computed power is returned.

    Returns
    -------
    results : results instance or float
        If return_results is False, then only the power is returned.
        If return_results is True, then a results instance with the
        information in attributes is returned.

        power : float
            Power of the test, e.g. 0.8, is one minus the probability of a
            type II error. Power is the probability that the test correctly
            rejects the Null Hypothesis if the Alternative Hypothesis is true.

        Other attributes in results instance include :

        std_null
            standard error of difference under the null hypothesis (without
            sqrt(nobs1))
        std_alt
            standard error of difference under the alternative hypothesis
            (without sqrt(nobs1))


    References
    ----------
    .. [1] Zhu, Haiyuan. 2017. “Sample Size Calculation for Comparing Two
       Poisson or Negative Binomial Rates in Noninferiority or Equivalence
       Trials.” Statistics in Biopharmaceutical Research, March.
       https://doi.org/10.1080/19466315.2016.1225594
    .. [2] Zhu, Haiyuan, and Hassan Lakkis. 2014. “Sample Size Calculation for
       Comparing Two Negative Binomial Rates.” Statistics in Medicine 33 (3):
       376–87. https://doi.org/10.1002/sim.5947.
    .. [3] PASS documentation
    """
    rate1, rate2, nobs1 = map(np.asarray, [rate1, rate2, nobs1])

    nobs2 = nobs_ratio * nobs1

    v1 = ((1 / rate2 + 1 / (nobs_ratio * rate1)) / exposure +
          (1 + nobs_ratio) / nobs_ratio * dispersion)
    if method_var == "alt":
        v0_low = v0_upp = v1
    elif method_var == "ftotal":
        v0_low = (1 + low * nobs_ratio)**2 / (
             exposure * nobs_ratio * low * (rate1 + nobs_ratio * rate2))
        v0_low += (1 + nobs_ratio) / nobs_ratio * dispersion
        v0_upp = (1 + upp * nobs_ratio)**2 / (
             exposure * nobs_ratio * upp * (rate1 + nobs_ratio * rate2))
        v0_upp += (1 + nobs_ratio) / nobs_ratio * dispersion
    elif method_var == "score":
        v0_low = _var_cmle_negbin(rate1, rate2, nobs_ratio,
                                  exposure=exposure, value=low,
                                  dispersion=dispersion)[0]
        v0_upp = _var_cmle_negbin(rate1, rate2, nobs_ratio,
                                  exposure=exposure, value=upp,
                                  dispersion=dispersion)[0]
    else:
        raise NotImplementedError(f"method_var {method_var} not recognized")

    es_low = np.log(rate1 / rate2) - np.log(low)
    es_upp = np.log(rate1 / rate2) - np.log(upp)
    std_null_low = np.sqrt(v0_low)
    std_null_upp = np.sqrt(v0_upp)
    std_alternative = np.sqrt(v1)

    pow_ = _power_equivalence_het(es_low, es_upp, nobs1, alpha=alpha,
                                  std_null_low=std_null_low,
                                  std_null_upp=std_null_upp,
                                  std_alternative=std_alternative)

    if return_results:
        res = HolderTuple(
            power=pow_[0],
            power_margins=pow[1:],
            std_null_low=std_null_low,
            std_null_upp=std_null_upp,
            std_alt=std_alternative,
            nobs1=nobs1,
            nobs2=nobs2,
            nobs_ratio=nobs_ratio,
            alpha=alpha,
            tuple_=("power",),  # override default
            )
        return res
    else:
        return pow_[0]
