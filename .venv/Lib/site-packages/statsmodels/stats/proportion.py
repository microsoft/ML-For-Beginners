# -*- coding: utf-8 -*-
"""
Tests and Confidence Intervals for Binomial Proportions

Created on Fri Mar 01 00:23:07 2013

Author: Josef Perktold
License: BSD-3
"""

from statsmodels.compat.python import lzip
from typing import Callable, Tuple
import numpy as np
import pandas as pd
from scipy import optimize, stats

from statsmodels.stats.base import AllPairsResults, HolderTuple
from statsmodels.stats.weightstats import _zstat_generic2
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
from statsmodels.tools.testing import Holder
from statsmodels.tools.validation import array_like

FLOAT_INFO = np.finfo(float)


def _bound_proportion_confint(
    func: Callable[[float], float], qi: float, lower: bool = True
) -> float:
    """
    Try hard to find a bound different from eps/1 - eps in proportion_confint

    Parameters
    ----------
    func : callable
        Callable function to use as the objective of the search
    qi : float
        The empirical success rate
    lower : bool
        Whether to fund a lower bound for the left side of the CI

    Returns
    -------
    float
        The coarse bound
    """
    default = FLOAT_INFO.eps if lower else 1.0 - FLOAT_INFO.eps

    def step(v):
        return v / 8 if lower else v + (1.0 - v) / 8

    x = step(qi)
    w = func(x)
    cnt = 1
    while w > 0 and cnt < 10:
        x = step(x)
        w = func(x)
        cnt += 1
    return x if cnt < 10 else default


def _bisection_search_conservative(
    func: Callable[[float], float], lb: float, ub: float, steps: int = 27
) -> Tuple[float, float]:
    """
    Private function used as a fallback by proportion_confint

    Used when brentq returns a non-conservative bound for the CI

    Parameters
    ----------
    func : callable
        Callable function to use as the objective of the search
    lb : float
        Lower bound
    ub : float
        Upper bound
    steps : int
        Number of steps to use in the bisection

    Returns
    -------
    est : float
        The estimated value.  Will always produce a negative value of func
    func_val : float
        The value of the function at the estimate
    """
    upper = func(ub)
    lower = func(lb)
    best = upper if upper < 0 else lower
    best_pt = ub if upper < 0 else lb
    if np.sign(lower) == np.sign(upper):
        raise ValueError("problem with signs")
    mp = (ub + lb) / 2
    mid = func(mp)
    if (mid < 0) and (mid > best):
        best = mid
        best_pt = mp
    for _ in range(steps):
        if np.sign(mid) == np.sign(upper):
            ub = mp
            upper = mid
        else:
            lb = mp
        mp = (ub + lb) / 2
        mid = func(mp)
        if (mid < 0) and (mid > best):
            best = mid
            best_pt = mp
    return best_pt, best


def proportion_confint(count, nobs, alpha:float=0.05, method="normal"):
    """
    Confidence interval for a binomial proportion

    Parameters
    ----------
    count : {int or float, array_like}
        number of successes, can be pandas Series or DataFrame. Arrays
        must contain integer values if method is "binom_test".
    nobs : {int or float, array_like}
        total number of trials.  Arrays must contain integer values if method
        is "binom_test".
    alpha : float
        Significance level, default 0.05. Must be in (0, 1)
    method : {"normal", "agresti_coull", "beta", "wilson", "binom_test"}
        default: "normal"
        method to use for confidence interval. Supported methods:

         - `normal` : asymptotic normal approximation
         - `agresti_coull` : Agresti-Coull interval
         - `beta` : Clopper-Pearson interval based on Beta distribution
         - `wilson` : Wilson Score interval
         - `jeffreys` : Jeffreys Bayesian Interval
         - `binom_test` : Numerical inversion of binom_test

    Returns
    -------
    ci_low, ci_upp : {float, ndarray, Series DataFrame}
        lower and upper confidence level with coverage (approximately) 1-alpha.
        When a pandas object is returned, then the index is taken from `count`.

    Notes
    -----
    Beta, the Clopper-Pearson exact interval has coverage at least 1-alpha,
    but is in general conservative. Most of the other methods have average
    coverage equal to 1-alpha, but will have smaller coverage in some cases.

    The "beta" and "jeffreys" interval are central, they use alpha/2 in each
    tail, and alpha is not adjusted at the boundaries. In the extreme case
    when `count` is zero or equal to `nobs`, then the coverage will be only
    1 - alpha/2 in the case of "beta".

    The confidence intervals are clipped to be in the [0, 1] interval in the
    case of "normal" and "agresti_coull".

    Method "binom_test" directly inverts the binomial test in scipy.stats.
    which has discrete steps.

    TODO: binom_test intervals raise an exception in small samples if one
       interval bound is close to zero or one.

    References
    ----------
    .. [*] https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval

    .. [*] Brown, Lawrence D.; Cai, T. Tony; DasGupta, Anirban (2001).
       "Interval Estimation for a Binomial Proportion", Statistical
       Science 16 (2): 101–133. doi:10.1214/ss/1009213286.
    """
    is_scalar = np.isscalar(count) and np.isscalar(nobs)
    is_pandas = isinstance(count, (pd.Series, pd.DataFrame))
    count_a = array_like(count, "count", optional=False, ndim=None)
    nobs_a = array_like(nobs, "nobs", optional=False, ndim=None)

    def _check(x: np.ndarray, name: str) -> np.ndarray:
        if np.issubdtype(x.dtype, np.integer):
            return x
        y = x.astype(np.int64, casting="unsafe")
        if np.any(y != x):
            raise ValueError(
                f"{name} must have an integral dtype. Found data with "
                f"dtype {x.dtype}"
            )
        return y

    if method == "binom_test":
        count_a = _check(np.asarray(count_a), "count")
        nobs_a = _check(np.asarray(nobs_a), "count")

    q_ = count_a / nobs_a
    alpha_2 = 0.5 * alpha

    if method == "normal":
        std_ = np.sqrt(q_ * (1 - q_) / nobs_a)
        dist = stats.norm.isf(alpha / 2.0) * std_
        ci_low = q_ - dist
        ci_upp = q_ + dist
    elif method == "binom_test":
        # inverting the binomial test
        def func_factory(count: int, nobs: int) -> Callable[[float], float]:
            if hasattr(stats, "binomtest"):

                def func(qi):
                    return stats.binomtest(count, nobs, p=qi).pvalue - alpha

            else:
                # Remove after min SciPy >= 1.7
                def func(qi):
                    return stats.binom_test(count, nobs, p=qi) - alpha

            return func

        bcast = np.broadcast(count_a, nobs_a)
        ci_low = np.zeros(bcast.shape)
        ci_upp = np.zeros(bcast.shape)
        index = bcast.index
        for c, n in bcast:
            # Enforce symmetry
            reverse = False
            _q = q_.flat[index]
            if c > n // 2:
                c = n - c
                reverse = True
                _q = 1 - _q
            func = func_factory(c, n)
            if c == 0:
                ci_low.flat[index] = 0.0
            else:
                lower_bnd = _bound_proportion_confint(func, _q, lower=True)
                val, _z = optimize.brentq(
                    func, lower_bnd, _q, full_output=True
                )
                if func(val) > 0:
                    power = 10
                    new_lb = val - (val - lower_bnd) / 2**power
                    while func(new_lb) > 0 and power >= 0:
                        power -= 1
                        new_lb = val - (val - lower_bnd) / 2**power
                    val, _ = _bisection_search_conservative(func, new_lb, _q)
                ci_low.flat[index] = val
            if c == n:
                ci_upp.flat[index] = 1.0
            else:
                upper_bnd = _bound_proportion_confint(func, _q, lower=False)
                val, _z = optimize.brentq(
                    func, _q, upper_bnd, full_output=True
                )
                if func(val) > 0:
                    power = 10
                    new_ub = val + (upper_bnd - val) / 2**power
                    while func(new_ub) > 0 and power >= 0:
                        power -= 1
                        new_ub = val - (upper_bnd - val) / 2**power
                    val, _ = _bisection_search_conservative(func, _q, new_ub)
                ci_upp.flat[index] = val
            if reverse:
                temp = ci_upp.flat[index]
                ci_upp.flat[index] = 1 - ci_low.flat[index]
                ci_low.flat[index] = 1 - temp
            index = bcast.index
    elif method == "beta":
        ci_low = stats.beta.ppf(alpha_2, count_a, nobs_a - count_a + 1)
        ci_upp = stats.beta.isf(alpha_2, count_a + 1, nobs_a - count_a)

        if np.ndim(ci_low) > 0:
            ci_low.flat[q_.flat == 0] = 0
            ci_upp.flat[q_.flat == 1] = 1
        else:
            ci_low = 0 if q_ == 0 else ci_low
            ci_upp = 1 if q_ == 1 else ci_upp
    elif method == "agresti_coull":
        crit = stats.norm.isf(alpha / 2.0)
        nobs_c = nobs_a + crit**2
        q_c = (count_a + crit**2 / 2.0) / nobs_c
        std_c = np.sqrt(q_c * (1.0 - q_c) / nobs_c)
        dist = crit * std_c
        ci_low = q_c - dist
        ci_upp = q_c + dist
    elif method == "wilson":
        crit = stats.norm.isf(alpha / 2.0)
        crit2 = crit**2
        denom = 1 + crit2 / nobs_a
        center = (q_ + crit2 / (2 * nobs_a)) / denom
        dist = crit * np.sqrt(
            q_ * (1.0 - q_) / nobs_a + crit2 / (4.0 * nobs_a**2)
        )
        dist /= denom
        ci_low = center - dist
        ci_upp = center + dist
    # method adjusted to be more forgiving of misspellings or incorrect option name
    elif method[:4] == "jeff":
        ci_low, ci_upp = stats.beta.interval(
            1 - alpha, count_a + 0.5, nobs_a - count_a + 0.5
        )
    else:
        raise NotImplementedError(f"method {method} is not available")
    if method in ["normal", "agresti_coull"]:
        ci_low = np.clip(ci_low, 0, 1)
        ci_upp = np.clip(ci_upp, 0, 1)
    if is_pandas:
        container = pd.Series if isinstance(count, pd.Series) else pd.DataFrame
        ci_low = container(ci_low, index=count.index)
        ci_upp = container(ci_upp, index=count.index)
    if is_scalar:
        return float(ci_low), float(ci_upp)
    return ci_low, ci_upp


def multinomial_proportions_confint(counts, alpha=0.05, method='goodman'):
    """
    Confidence intervals for multinomial proportions.

    Parameters
    ----------
    counts : array_like of int, 1-D
        Number of observations in each category.
    alpha : float in (0, 1), optional
        Significance level, defaults to 0.05.
    method : {'goodman', 'sison-glaz'}, optional
        Method to use to compute the confidence intervals; available methods
        are:

         - `goodman`: based on a chi-squared approximation, valid if all
           values in `counts` are greater or equal to 5 [2]_
         - `sison-glaz`: less conservative than `goodman`, but only valid if
           `counts` has 7 or more categories (``len(counts) >= 7``) [3]_

    Returns
    -------
    confint : ndarray, 2-D
        Array of [lower, upper] confidence levels for each category, such that
        overall coverage is (approximately) `1-alpha`.

    Raises
    ------
    ValueError
        If `alpha` is not in `(0, 1)` (bounds excluded), or if the values in
        `counts` are not all positive or null.
    NotImplementedError
        If `method` is not kown.
    Exception
        When ``method == 'sison-glaz'``, if for some reason `c` cannot be
        computed; this signals a bug and should be reported.

    Notes
    -----
    The `goodman` method [2]_ is based on approximating a statistic based on
    the multinomial as a chi-squared random variable. The usual recommendation
    is that this is valid if all the values in `counts` are greater than or
    equal to 5. There is no condition on the number of categories for this
    method.

    The `sison-glaz` method [3]_ approximates the multinomial probabilities,
    and evaluates that with a maximum-likelihood estimator. The first
    approximation is an Edgeworth expansion that converges when the number of
    categories goes to infinity, and the maximum-likelihood estimator converges
    when the number of observations (``sum(counts)``) goes to infinity. In
    their paper, Sison & Glaz demo their method with at least 7 categories, so
    ``len(counts) >= 7`` with all values in `counts` at or above 5 can be used
    as a rule of thumb for the validity of this method. This method is less
    conservative than the `goodman` method (i.e. it will yield confidence
    intervals closer to the desired significance level), but produces
    confidence intervals of uniform width over all categories (except when the
    intervals reach 0 or 1, in which case they are truncated), which makes it
    most useful when proportions are of similar magnitude.

    Aside from the original sources ([1]_, [2]_, and [3]_), the implementation
    uses the formulas (though not the code) presented in [4]_ and [5]_.

    References
    ----------
    .. [1] Levin, Bruce, "A representation for multinomial cumulative
           distribution functions," The Annals of Statistics, Vol. 9, No. 5,
           1981, pp. 1123-1126.

    .. [2] Goodman, L.A., "On simultaneous confidence intervals for multinomial
           proportions," Technometrics, Vol. 7, No. 2, 1965, pp. 247-254.

    .. [3] Sison, Cristina P., and Joseph Glaz, "Simultaneous Confidence
           Intervals and Sample Size Determination for Multinomial
           Proportions," Journal of the American Statistical Association,
           Vol. 90, No. 429, 1995, pp. 366-369.

    .. [4] May, Warren L., and William D. Johnson, "A SAS® macro for
           constructing simultaneous confidence intervals  for multinomial
           proportions," Computer methods and programs in Biomedicine, Vol. 53,
           No. 3, 1997, pp. 153-162.

    .. [5] May, Warren L., and William D. Johnson, "Constructing two-sided
           simultaneous confidence intervals for multinomial proportions for
           small counts in a large number of cells," Journal of Statistical
           Software, Vol. 5, No. 6, 2000, pp. 1-24.
    """
    if alpha <= 0 or alpha >= 1:
        raise ValueError('alpha must be in (0, 1), bounds excluded')
    counts = np.array(counts, dtype=float)
    if (counts < 0).any():
        raise ValueError('counts must be >= 0')

    n = counts.sum()
    k = len(counts)
    proportions = counts / n
    if method == 'goodman':
        chi2 = stats.chi2.ppf(1 - alpha / k, 1)
        delta = chi2 ** 2 + (4 * n * proportions * chi2 * (1 - proportions))
        region = ((2 * n * proportions + chi2 +
                   np.array([- np.sqrt(delta), np.sqrt(delta)])) /
                  (2 * (chi2 + n))).T
    elif method[:5] == 'sison':  # We accept any name starting with 'sison'
        # Define a few functions we'll use a lot.
        def poisson_interval(interval, p):
            """
            Compute P(b <= Z <= a) where Z ~ Poisson(p) and
            `interval = (b, a)`.
            """
            b, a = interval
            prob = stats.poisson.cdf(a, p) - stats.poisson.cdf(b - 1, p)
            return prob

        def truncated_poisson_factorial_moment(interval, r, p):
            """
            Compute mu_r, the r-th factorial moment of a poisson random
            variable of parameter `p` truncated to `interval = (b, a)`.
            """
            b, a = interval
            return p ** r * (1 - ((poisson_interval((a - r + 1, a), p) -
                                   poisson_interval((b - r, b - 1), p)) /
                                  poisson_interval((b, a), p)))

        def edgeworth(intervals):
            """
            Compute the Edgeworth expansion term of Sison & Glaz's formula
            (1) (approximated probability for multinomial proportions in a
            given box).
            """
            # Compute means and central moments of the truncated poisson
            # variables.
            mu_r1, mu_r2, mu_r3, mu_r4 = [
                np.array([truncated_poisson_factorial_moment(interval, r, p)
                          for (interval, p) in zip(intervals, counts)])
                for r in range(1, 5)
            ]
            mu = mu_r1
            mu2 = mu_r2 + mu - mu ** 2
            mu3 = mu_r3 + mu_r2 * (3 - 3 * mu) + mu - 3 * mu ** 2 + 2 * mu ** 3
            mu4 = (mu_r4 + mu_r3 * (6 - 4 * mu) +
                   mu_r2 * (7 - 12 * mu + 6 * mu ** 2) +
                   mu - 4 * mu ** 2 + 6 * mu ** 3 - 3 * mu ** 4)

            # Compute expansion factors, gamma_1 and gamma_2.
            g1 = mu3.sum() / mu2.sum() ** 1.5
            g2 = (mu4.sum() - 3 * (mu2 ** 2).sum()) / mu2.sum() ** 2

            # Compute the expansion itself.
            x = (n - mu.sum()) / np.sqrt(mu2.sum())
            phi = np.exp(- x ** 2 / 2) / np.sqrt(2 * np.pi)
            H3 = x ** 3 - 3 * x
            H4 = x ** 4 - 6 * x ** 2 + 3
            H6 = x ** 6 - 15 * x ** 4 + 45 * x ** 2 - 15
            f = phi * (1 + g1 * H3 / 6 + g2 * H4 / 24 + g1 ** 2 * H6 / 72)
            return f / np.sqrt(mu2.sum())


        def approximated_multinomial_interval(intervals):
            """
            Compute approximated probability for Multinomial(n, proportions)
            to be in `intervals` (Sison & Glaz's formula (1)).
            """
            return np.exp(
                np.sum(np.log([poisson_interval(interval, p)
                               for (interval, p) in zip(intervals, counts)])) +
                np.log(edgeworth(intervals)) -
                np.log(stats.poisson._pmf(n, n))
            )

        def nu(c):
            """
            Compute interval coverage for a given `c` (Sison & Glaz's
            formula (7)).
            """
            return approximated_multinomial_interval(
                [(np.maximum(count - c, 0), np.minimum(count + c, n))
                 for count in counts])

        # Find the value of `c` that will give us the confidence intervals
        # (solving nu(c) <= 1 - alpha < nu(c + 1).
        c = 1.0
        nuc = nu(c)
        nucp1 = nu(c + 1)
        while not (nuc <= (1 - alpha) < nucp1):
            if c > n:
                raise Exception("Couldn't find a value for `c` that "
                                "solves nu(c) <= 1 - alpha < nu(c + 1)")
            c += 1
            nuc = nucp1
            nucp1 = nu(c + 1)

        # Compute gamma and the corresponding confidence intervals.
        g = (1 - alpha - nuc) / (nucp1 - nuc)
        ci_lower = np.maximum(proportions - c / n, 0)
        ci_upper = np.minimum(proportions + (c + 2 * g) / n, 1)
        region = np.array([ci_lower, ci_upper]).T
    else:
        raise NotImplementedError('method "%s" is not available' % method)
    return region


def samplesize_confint_proportion(proportion, half_length, alpha=0.05,
                                  method='normal'):
    """
    Find sample size to get desired confidence interval length

    Parameters
    ----------
    proportion : float in (0, 1)
        proportion or quantile
    half_length : float in (0, 1)
        desired half length of the confidence interval
    alpha : float in (0, 1)
        significance level, default 0.05,
        coverage of the two-sided interval is (approximately) ``1 - alpha``
    method : str in ['normal']
        method to use for confidence interval,
        currently only normal approximation

    Returns
    -------
    n : float
        sample size to get the desired half length of the confidence interval

    Notes
    -----
    this is mainly to store the formula.
    possible application: number of replications in bootstrap samples

    """
    q_ = proportion
    if method == 'normal':
        n = q_ * (1 - q_) / (half_length / stats.norm.isf(alpha / 2.))**2
    else:
        raise NotImplementedError('only "normal" is available')

    return n


def proportion_effectsize(prop1, prop2, method='normal'):
    """
    Effect size for a test comparing two proportions

    for use in power function

    Parameters
    ----------
    prop1, prop2 : float or array_like
        The proportion value(s).

    Returns
    -------
    es : float or ndarray
        effect size for (transformed) prop1 - prop2

    Notes
    -----
    only method='normal' is implemented to match pwr.p2.test
    see http://www.statmethods.net/stats/power.html

    Effect size for `normal` is defined as ::

        2 * (arcsin(sqrt(prop1)) - arcsin(sqrt(prop2)))

    I think other conversions to normality can be used, but I need to check.

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> sm.stats.proportion_effectsize(0.5, 0.4)
    0.20135792079033088
    >>> sm.stats.proportion_effectsize([0.3, 0.4, 0.5], 0.4)
    array([-0.21015893,  0.        ,  0.20135792])

    """
    if method != 'normal':
        raise ValueError('only "normal" is implemented')

    es = 2 * (np.arcsin(np.sqrt(prop1)) - np.arcsin(np.sqrt(prop2)))
    return es


def std_prop(prop, nobs):
    """
    Standard error for the estimate of a proportion

    This is just ``np.sqrt(p * (1. - p) / nobs)``

    Parameters
    ----------
    prop : array_like
        proportion
    nobs : int, array_like
        number of observations

    Returns
    -------
    std : array_like
        standard error for a proportion of nobs independent observations
    """
    return np.sqrt(prop * (1. - prop) / nobs)


def _std_diff_prop(p1, p2, ratio=1):
    return np.sqrt(p1 * (1 - p1) + p2 * (1 - p2) / ratio)


def _power_ztost(mean_low, var_low, mean_upp, var_upp, mean_alt, var_alt,
                 alpha=0.05, discrete=True, dist='norm', nobs=None,
                 continuity=0, critval_continuity=0):
    """
    Generic statistical power function for normal based equivalence test

    This includes options to adjust the normal approximation and can use
    the binomial to evaluate the probability of the rejection region

    see power_ztost_prob for a description of the options
    """
    # TODO: refactor structure, separate norm and binom better
    if not isinstance(continuity, tuple):
        continuity = (continuity, continuity)
    crit = stats.norm.isf(alpha)
    k_low = mean_low + np.sqrt(var_low) * crit
    k_upp = mean_upp - np.sqrt(var_upp) * crit
    if discrete or dist == 'binom':
        k_low = np.ceil(k_low * nobs + 0.5 * critval_continuity)
        k_upp = np.trunc(k_upp * nobs - 0.5 * critval_continuity)
        if dist == 'norm':
            #need proportion
            k_low = (k_low) * 1. / nobs #-1 to match PASS
            k_upp = k_upp * 1. / nobs
#    else:
#        if dist == 'binom':
#            #need counts
#            k_low *= nobs
#            k_upp *= nobs
    #print mean_low, np.sqrt(var_low), crit, var_low
    #print mean_upp, np.sqrt(var_upp), crit, var_upp
    if np.any(k_low > k_upp):   #vectorize
        import warnings
        warnings.warn("no overlap, power is zero", HypothesisTestWarning)
    std_alt = np.sqrt(var_alt)
    z_low = (k_low - mean_alt - continuity[0] * 0.5 / nobs) / std_alt
    z_upp = (k_upp - mean_alt + continuity[1] * 0.5 / nobs) / std_alt
    if dist == 'norm':
        power = stats.norm.cdf(z_upp) - stats.norm.cdf(z_low)
    elif dist == 'binom':
        power = (stats.binom.cdf(k_upp, nobs, mean_alt) -
                     stats.binom.cdf(k_low-1, nobs, mean_alt))
    return power, (k_low, k_upp, z_low, z_upp)


def binom_tost(count, nobs, low, upp):
    """
    Exact TOST test for one proportion using binomial distribution

    Parameters
    ----------
    count : {int, array_like}
        the number of successes in nobs trials.
    nobs : int
        the number of trials or observations.
    low, upp : floats
        lower and upper limit of equivalence region

    Returns
    -------
    pvalue : float
        p-value of equivalence test
    pval_low, pval_upp : floats
        p-values of lower and upper one-sided tests

    """
    # binom_test_stat only returns pval
    tt1 = binom_test(count, nobs, alternative='larger', prop=low)
    tt2 = binom_test(count, nobs, alternative='smaller', prop=upp)
    return np.maximum(tt1, tt2), tt1, tt2,


def binom_tost_reject_interval(low, upp, nobs, alpha=0.05):
    """
    Rejection region for binomial TOST

    The interval includes the end points,
    `reject` if and only if `r_low <= x <= r_upp`.

    The interval might be empty with `r_upp < r_low`.

    Parameters
    ----------
    low, upp : floats
        lower and upper limit of equivalence region
    nobs : int
        the number of trials or observations.

    Returns
    -------
    x_low, x_upp : float
        lower and upper bound of rejection region

    """
    x_low = stats.binom.isf(alpha, nobs, low) + 1
    x_upp = stats.binom.ppf(alpha, nobs, upp) - 1
    return x_low, x_upp


def binom_test_reject_interval(value, nobs, alpha=0.05, alternative='two-sided'):
    """
    Rejection region for binomial test for one sample proportion

    The interval includes the end points of the rejection region.

    Parameters
    ----------
    value : float
        proportion under the Null hypothesis
    nobs : int
        the number of trials or observations.

    Returns
    -------
    x_low, x_upp : int
        lower and upper bound of rejection region
    """
    if alternative in ['2s', 'two-sided']:
        alternative = '2s'  # normalize alternative name
        alpha = alpha / 2

    if alternative in ['2s', 'smaller']:
        x_low = stats.binom.ppf(alpha, nobs, value) - 1
    else:
        x_low = 0
    if alternative in ['2s', 'larger']:
        x_upp = stats.binom.isf(alpha, nobs, value) + 1
    else :
        x_upp = nobs

    return int(x_low), int(x_upp)


def binom_test(count, nobs, prop=0.5, alternative='two-sided'):
    """
    Perform a test that the probability of success is p.

    This is an exact, two-sided test of the null hypothesis
    that the probability of success in a Bernoulli experiment
    is `p`.

    Parameters
    ----------
    count : {int, array_like}
        the number of successes in nobs trials.
    nobs : int
        the number of trials or observations.
    prop : float, optional
        The probability of success under the null hypothesis,
        `0 <= prop <= 1`. The default value is `prop = 0.5`
    alternative : str in ['two-sided', 'smaller', 'larger']
        alternative hypothesis, which can be two-sided or either one of the
        one-sided tests.

    Returns
    -------
    p-value : float
        The p-value of the hypothesis test

    Notes
    -----
    This uses scipy.stats.binom_test for the two-sided alternative.
    """

    if np.any(prop > 1.0) or np.any(prop < 0.0):
        raise ValueError("p must be in range [0,1]")
    if alternative in ['2s', 'two-sided']:
        try:
            pval = stats.binomtest(count, n=nobs, p=prop).pvalue
        except AttributeError:
            # Remove after min SciPy >= 1.7
            pval = stats.binom_test(count, n=nobs, p=prop)
    elif alternative in ['l', 'larger']:
        pval = stats.binom.sf(count-1, nobs, prop)
    elif alternative in ['s', 'smaller']:
        pval = stats.binom.cdf(count, nobs, prop)
    else:
        raise ValueError('alternative not recognized\n'
                         'should be two-sided, larger or smaller')
    return pval


def power_binom_tost(low, upp, nobs, p_alt=None, alpha=0.05):
    if p_alt is None:
        p_alt = 0.5 * (low + upp)
    x_low, x_upp = binom_tost_reject_interval(low, upp, nobs, alpha=alpha)
    power = (stats.binom.cdf(x_upp, nobs, p_alt) -
                     stats.binom.cdf(x_low-1, nobs, p_alt))
    return power


def power_ztost_prop(low, upp, nobs, p_alt, alpha=0.05, dist='norm',
                     variance_prop=None, discrete=True, continuity=0,
                     critval_continuity=0):
    """
    Power of proportions equivalence test based on normal distribution

    Parameters
    ----------
    low, upp : floats
        lower and upper limit of equivalence region
    nobs : int
        number of observations
    p_alt : float in (0,1)
        proportion under the alternative
    alpha : float in (0,1)
        significance level of the test
    dist : str in ['norm', 'binom']
        This defines the distribution to evaluate the power of the test. The
        critical values of the TOST test are always based on the normal
        approximation, but the distribution for the power can be either the
        normal (default) or the binomial (exact) distribution.
    variance_prop : None or float in (0,1)
        If this is None, then the variances for the two one sided tests are
        based on the proportions equal to the equivalence limits.
        If variance_prop is given, then it is used to calculate the variance
        for the TOST statistics. If this is based on an sample, then the
        estimated proportion can be used.
    discrete : bool
        If true, then the critical values of the rejection region are converted
        to integers. If dist is "binom", this is automatically assumed.
        If discrete is false, then the TOST critical values are used as
        floating point numbers, and the power is calculated based on the
        rejection region that is not discretized.
    continuity : bool or float
        adjust the rejection region for the normal power probability. This has
        and effect only if ``dist='norm'``
    critval_continuity : bool or float
        If this is non-zero, then the critical values of the tost rejection
        region are adjusted before converting to integers. This affects both
        distributions, ``dist='norm'`` and ``dist='binom'``.

    Returns
    -------
    power : float
        statistical power of the equivalence test.
    (k_low, k_upp, z_low, z_upp) : tuple of floats
        critical limits in intermediate steps
        temporary return, will be changed

    Notes
    -----
    In small samples the power for the ``discrete`` version, has a sawtooth
    pattern as a function of the number of observations. As a consequence,
    small changes in the number of observations or in the normal approximation
    can have a large effect on the power.

    ``continuity`` and ``critval_continuity`` are added to match some results
    of PASS, and are mainly to investigate the sensitivity of the ztost power
    to small changes in the rejection region. From my interpretation of the
    equations in the SAS manual, both are zero in SAS.

    works vectorized

    **verification:**

    The ``dist='binom'`` results match PASS,
    The ``dist='norm'`` results look reasonable, but no benchmark is available.

    References
    ----------
    SAS Manual: Chapter 68: The Power Procedure, Computational Resources
    PASS Chapter 110: Equivalence Tests for One Proportion.

    """
    mean_low = low
    var_low = std_prop(low, nobs)**2
    mean_upp = upp
    var_upp = std_prop(upp, nobs)**2
    mean_alt = p_alt
    var_alt = std_prop(p_alt, nobs)**2
    if variance_prop is not None:
        var_low = var_upp = std_prop(variance_prop, nobs)**2
    power = _power_ztost(mean_low, var_low, mean_upp, var_upp, mean_alt, var_alt,
                 alpha=alpha, discrete=discrete, dist=dist, nobs=nobs,
                 continuity=continuity, critval_continuity=critval_continuity)
    return np.maximum(power[0], 0), power[1:]


def _table_proportion(count, nobs):
    """
    Create a k by 2 contingency table for proportion

    helper function for proportions_chisquare

    Parameters
    ----------
    count : {int, array_like}
        the number of successes in nobs trials.
    nobs : int
        the number of trials or observations.

    Returns
    -------
    table : ndarray
        (k, 2) contingency table

    Notes
    -----
    recent scipy has more elaborate contingency table functions

    """
    count = np.asarray(count)
    dt = np.promote_types(count.dtype, np.float64)
    count = np.asarray(count, dtype=dt)
    table = np.column_stack((count, nobs - count))
    expected = table.sum(0) * table.sum(1)[:, None] * 1. / table.sum()
    n_rows = table.shape[0]
    return table, expected, n_rows


def proportions_ztest(count, nobs, value=None, alternative='two-sided',
                      prop_var=False):
    """
    Test for proportions based on normal (z) test

    Parameters
    ----------
    count : {int, array_like}
        the number of successes in nobs trials. If this is array_like, then
        the assumption is that this represents the number of successes for
        each independent sample
    nobs : {int, array_like}
        the number of trials or observations, with the same length as
        count.
    value : float, array_like or None, optional
        This is the value of the null hypothesis equal to the proportion in the
        case of a one sample test. In the case of a two-sample test, the
        null hypothesis is that prop[0] - prop[1] = value, where prop is the
        proportion in the two samples. If not provided value = 0 and the null
        is prop[0] = prop[1]
    alternative : str in ['two-sided', 'smaller', 'larger']
        The alternative hypothesis can be either two-sided or one of the one-
        sided tests, smaller means that the alternative hypothesis is
        ``prop < value`` and larger means ``prop > value``. In the two sample
        test, smaller means that the alternative hypothesis is ``p1 < p2`` and
        larger means ``p1 > p2`` where ``p1`` is the proportion of the first
        sample and ``p2`` of the second one.
    prop_var : False or float in (0, 1)
        If prop_var is false, then the variance of the proportion estimate is
        calculated based on the sample proportion. Alternatively, a proportion
        can be specified to calculate this variance. Common use case is to
        use the proportion under the Null hypothesis to specify the variance
        of the proportion estimate.

    Returns
    -------
    zstat : float
        test statistic for the z-test
    p-value : float
        p-value for the z-test

    Examples
    --------
    >>> count = 5
    >>> nobs = 83
    >>> value = .05
    >>> stat, pval = proportions_ztest(count, nobs, value)
    >>> print('{0:0.3f}'.format(pval))
    0.695

    >>> import numpy as np
    >>> from statsmodels.stats.proportion import proportions_ztest
    >>> count = np.array([5, 12])
    >>> nobs = np.array([83, 99])
    >>> stat, pval = proportions_ztest(count, nobs)
    >>> print('{0:0.3f}'.format(pval))
    0.159

    Notes
    -----
    This uses a simple normal test for proportions. It should be the same as
    running the mean z-test on the data encoded 1 for event and 0 for no event
    so that the sum corresponds to the count.

    In the one and two sample cases with two-sided alternative, this test
    produces the same p-value as ``proportions_chisquare``, since the
    chisquare is the distribution of the square of a standard normal
    distribution.
    """
    # TODO: verify that this really holds
    # TODO: add continuity correction or other improvements for small samples
    # TODO: change options similar to propotion_ztost ?

    count = np.asarray(count)
    nobs = np.asarray(nobs)

    if nobs.size == 1:
        nobs = nobs * np.ones_like(count)

    prop = count * 1. / nobs
    k_sample = np.size(prop)
    if value is None:
        if k_sample == 1:
            raise ValueError('value must be provided for a 1-sample test')
        value = 0
    if k_sample == 1:
        diff = prop - value
    elif k_sample == 2:
        diff = prop[0] - prop[1] - value
    else:
        msg = 'more than two samples are not implemented yet'
        raise NotImplementedError(msg)

    p_pooled = np.sum(count) * 1. / np.sum(nobs)

    nobs_fact = np.sum(1. / nobs)
    if prop_var:
        p_pooled = prop_var
    var_ = p_pooled * (1 - p_pooled) * nobs_fact
    std_diff = np.sqrt(var_)
    from statsmodels.stats.weightstats import _zstat_generic2
    return _zstat_generic2(diff, std_diff, alternative)


def proportions_ztost(count, nobs, low, upp, prop_var='sample'):
    """
    Equivalence test based on normal distribution

    Parameters
    ----------
    count : {int, array_like}
        the number of successes in nobs trials. If this is array_like, then
        the assumption is that this represents the number of successes for
        each independent sample
    nobs : int
        the number of trials or observations, with the same length as
        count.
    low, upp : float
        equivalence interval low < prop1 - prop2 < upp
    prop_var : str or float in (0, 1)
        prop_var determines which proportion is used for the calculation
        of the standard deviation of the proportion estimate
        The available options for string are 'sample' (default), 'null' and
        'limits'. If prop_var is a float, then it is used directly.

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
    if prop_var == 'limits':
        prop_var_low = low
        prop_var_upp = upp
    elif prop_var == 'sample':
        prop_var_low = prop_var_upp = False  #ztest uses sample
    elif prop_var == 'null':
        prop_var_low = prop_var_upp = 0.5 * (low + upp)
    elif np.isreal(prop_var):
        prop_var_low = prop_var_upp = prop_var

    tt1 = proportions_ztest(count, nobs, alternative='larger',
                            prop_var=prop_var_low, value=low)
    tt2 = proportions_ztest(count, nobs, alternative='smaller',
                            prop_var=prop_var_upp, value=upp)
    return np.maximum(tt1[1], tt2[1]), tt1, tt2,


def proportions_chisquare(count, nobs, value=None):
    """
    Test for proportions based on chisquare test

    Parameters
    ----------
    count : {int, array_like}
        the number of successes in nobs trials. If this is array_like, then
        the assumption is that this represents the number of successes for
        each independent sample
    nobs : int
        the number of trials or observations, with the same length as
        count.
    value : None or float or array_like

    Returns
    -------
    chi2stat : float
        test statistic for the chisquare test
    p-value : float
        p-value for the chisquare test
    (table, expected)
        table is a (k, 2) contingency table, ``expected`` is the corresponding
        table of counts that are expected under independence with given
        margins

    Notes
    -----
    Recent version of scipy.stats have a chisquare test for independence in
    contingency tables.

    This function provides a similar interface to chisquare tests as
    ``prop.test`` in R, however without the option for Yates continuity
    correction.

    count can be the count for the number of events for a single proportion,
    or the counts for several independent proportions. If value is given, then
    all proportions are jointly tested against this value. If value is not
    given and count and nobs are not scalar, then the null hypothesis is
    that all samples have the same proportion.

    """
    nobs = np.atleast_1d(nobs)
    table, expected, n_rows = _table_proportion(count, nobs)
    if value is not None:
        expected = np.column_stack((nobs * value, nobs * (1 - value)))
        ddof = n_rows - 1
    else:
        ddof = n_rows

    #print table, expected
    chi2stat, pval = stats.chisquare(table.ravel(), expected.ravel(),
                                     ddof=ddof)
    return chi2stat, pval, (table, expected)


def proportions_chisquare_allpairs(count, nobs, multitest_method='hs'):
    """
    Chisquare test of proportions for all pairs of k samples

    Performs a chisquare test for proportions for all pairwise comparisons.
    The alternative is two-sided

    Parameters
    ----------
    count : {int, array_like}
        the number of successes in nobs trials.
    nobs : int
        the number of trials or observations.
    multitest_method : str
        This chooses the method for the multiple testing p-value correction,
        that is used as default in the results.
        It can be any method that is available in  ``multipletesting``.
        The default is Holm-Sidak 'hs'.

    Returns
    -------
    result : AllPairsResults instance
        The returned results instance has several statistics, such as p-values,
        attached, and additional methods for using a non-default
        ``multitest_method``.

    Notes
    -----
    Yates continuity correction is not available.
    """
    #all_pairs = lmap(list, lzip(*np.triu_indices(4, 1)))
    all_pairs = lzip(*np.triu_indices(len(count), 1))
    pvals = [proportions_chisquare(count[list(pair)], nobs[list(pair)])[1]
               for pair in all_pairs]
    return AllPairsResults(pvals, all_pairs, multitest_method=multitest_method)


def proportions_chisquare_pairscontrol(count, nobs, value=None,
                               multitest_method='hs', alternative='two-sided'):
    """
    Chisquare test of proportions for pairs of k samples compared to control

    Performs a chisquare test for proportions for pairwise comparisons with a
    control (Dunnet's test). The control is assumed to be the first element
    of ``count`` and ``nobs``. The alternative is two-sided, larger or
    smaller.

    Parameters
    ----------
    count : {int, array_like}
        the number of successes in nobs trials.
    nobs : int
        the number of trials or observations.
    multitest_method : str
        This chooses the method for the multiple testing p-value correction,
        that is used as default in the results.
        It can be any method that is available in  ``multipletesting``.
        The default is Holm-Sidak 'hs'.
    alternative : str in ['two-sided', 'smaller', 'larger']
        alternative hypothesis, which can be two-sided or either one of the
        one-sided tests.

    Returns
    -------
    result : AllPairsResults instance
        The returned results instance has several statistics, such as p-values,
        attached, and additional methods for using a non-default
        ``multitest_method``.


    Notes
    -----
    Yates continuity correction is not available.

    ``value`` and ``alternative`` options are not yet implemented.

    """
    if (value is not None) or (alternative not in ['two-sided', '2s']):
        raise NotImplementedError
    #all_pairs = lmap(list, lzip(*np.triu_indices(4, 1)))
    all_pairs = [(0, k) for k in range(1, len(count))]
    pvals = [proportions_chisquare(count[list(pair)], nobs[list(pair)],
                                   #alternative=alternative)[1]
                                   )[1]
               for pair in all_pairs]
    return AllPairsResults(pvals, all_pairs, multitest_method=multitest_method)


def confint_proportions_2indep(count1, nobs1, count2, nobs2, method=None,
                               compare='diff', alpha=0.05, correction=True):
    """
    Confidence intervals for comparing two independent proportions.

    This assumes that we have two independent binomial samples.

    Parameters
    ----------
    count1, nobs1 : float
        Count and sample size for first sample.
    count2, nobs2 : float
        Count and sample size for the second sample.
    method : str
        Method for computing confidence interval. If method is None, then a
        default method is used. The default might change as more methods are
        added.

        diff:
         - 'wald',
         - 'agresti-caffo'
         - 'newcomb' (default)
         - 'score'

        ratio:
         - 'log'
         - 'log-adjusted' (default)
         - 'score'

        odds-ratio:
         - 'logit'
         - 'logit-adjusted' (default)
         - 'score'

    compare : string in ['diff', 'ratio' 'odds-ratio']
        If compare is diff, then the confidence interval is for diff = p1 - p2.
        If compare is ratio, then the confidence interval is for the risk ratio
        defined by ratio = p1 / p2.
        If compare is odds-ratio, then the confidence interval is for the
        odds-ratio defined by or = p1 / (1 - p1) / (p2 / (1 - p2).
    alpha : float
        Significance level for the confidence interval, default is 0.05.
        The nominal coverage probability is 1 - alpha.

    Returns
    -------
    low, upp

    See Also
    --------
    test_proportions_2indep
    tost_proportions_2indep

    Notes
    -----
    Status: experimental, API and defaults might still change.
        more ``methods`` will be added.

    References
    ----------
    .. [1] Fagerland, Morten W., Stian Lydersen, and Petter Laake. 2015.
       “Recommended Confidence Intervals for Two Independent Binomial
       Proportions.” Statistical Methods in Medical Research 24 (2): 224–54.
       https://doi.org/10.1177/0962280211415469.
    .. [2] Koopman, P. A. R. 1984. “Confidence Intervals for the Ratio of Two
       Binomial Proportions.” Biometrics 40 (2): 513–17.
       https://doi.org/10.2307/2531405.
    .. [3] Miettinen, Olli, and Markku Nurminen. "Comparative analysis of two
       rates." Statistics in medicine 4, no. 2 (1985): 213-226.
    .. [4] Newcombe, Robert G. 1998. “Interval Estimation for the Difference
       between Independent Proportions: Comparison of Eleven Methods.”
       Statistics in Medicine 17 (8): 873–90.
       https://doi.org/10.1002/(SICI)1097-0258(19980430)17:8<873::AID-
       SIM779>3.0.CO;2-I.
    .. [5] Newcombe, Robert G., and Markku M. Nurminen. 2011. “In Defence of
       Score Intervals for Proportions and Their Differences.” Communications
       in Statistics - Theory and Methods 40 (7): 1271–82.
       https://doi.org/10.1080/03610920903576580.
    """
    method_default = {'diff': 'newcomb',
                      'ratio': 'log-adjusted',
                      'odds-ratio': 'logit-adjusted'}
    # normalize compare name
    if compare.lower() == 'or':
        compare = 'odds-ratio'
    if method is None:
        method = method_default[compare]

    method = method.lower()
    if method.startswith('agr'):
        method = 'agresti-caffo'

    p1 = count1 / nobs1
    p2 = count2 / nobs2
    diff = p1 - p2
    addone = 1 if method == 'agresti-caffo' else 0

    if compare == 'diff':
        if method in ['wald', 'agresti-caffo']:
            count1_, nobs1_ = count1 + addone, nobs1 + 2 * addone
            count2_, nobs2_ = count2 + addone, nobs2 + 2 * addone
            p1_ = count1_ / nobs1_
            p2_ = count2_ / nobs2_
            diff_ = p1_ - p2_
            var = p1_ * (1 - p1_) / nobs1_ + p2_ * (1 - p2_) / nobs2_
            z = stats.norm.isf(alpha / 2)
            d_wald = z * np.sqrt(var)
            low = diff_ - d_wald
            upp = diff_ + d_wald

        elif method.startswith('newcomb'):
            low1, upp1 = proportion_confint(count1, nobs1,
                                            method='wilson', alpha=alpha)
            low2, upp2 = proportion_confint(count2, nobs2,
                                            method='wilson', alpha=alpha)
            d_low = np.sqrt((p1 - low1)**2 + (upp2 - p2)**2)
            d_upp = np.sqrt((p2 - low2)**2 + (upp1 - p1)**2)
            low = diff - d_low
            upp = diff + d_upp

        elif method == "score":
            low, upp = _score_confint_inversion(count1, nobs1, count2, nobs2,
                                                compare=compare, alpha=alpha,
                                                correction=correction)

        else:
            raise ValueError('method not recognized')

    elif compare == 'ratio':
        # ratio = p1 / p2
        if method in ['log', 'log-adjusted']:
            addhalf = 0.5 if method == 'log-adjusted' else 0
            count1_, nobs1_ = count1 + addhalf, nobs1 + addhalf
            count2_, nobs2_ = count2 + addhalf, nobs2 + addhalf
            p1_ = count1_ / nobs1_
            p2_ = count2_ / nobs2_
            ratio_ = p1_ / p2_
            var = (1 / count1_) - 1 / nobs1_ + 1 / count2_ - 1 / nobs2_
            z = stats.norm.isf(alpha / 2)
            d_log = z * np.sqrt(var)
            low = np.exp(np.log(ratio_) - d_log)
            upp = np.exp(np.log(ratio_) + d_log)

        elif method == 'score':
            res = _confint_riskratio_koopman(count1, nobs1, count2, nobs2,
                                             alpha=alpha,
                                             correction=correction)
            low, upp = res.confint

        else:
            raise ValueError('method not recognized')

    elif compare == 'odds-ratio':
        # odds_ratio = p1 / (1 - p1) / p2 * (1 - p2)
        if method in ['logit', 'logit-adjusted', 'logit-smoothed']:
            if method in ['logit-smoothed']:
                adjusted = _shrink_prob(count1, nobs1, count2, nobs2,
                                        shrink_factor=2, return_corr=False)[0]
                count1_, nobs1_, count2_, nobs2_ = adjusted

            else:
                addhalf = 0.5 if method == 'logit-adjusted' else 0
                count1_, nobs1_ = count1 + addhalf, nobs1 + 2 * addhalf
                count2_, nobs2_ = count2 + addhalf, nobs2 + 2 * addhalf
            p1_ = count1_ / nobs1_
            p2_ = count2_ / nobs2_
            odds_ratio_ = p1_ / (1 - p1_) / p2_ * (1 - p2_)
            var = (1 / count1_ + 1 / (nobs1_ - count1_) +
                   1 / count2_ + 1 / (nobs2_ - count2_))
            z = stats.norm.isf(alpha / 2)
            d_log = z * np.sqrt(var)
            low = np.exp(np.log(odds_ratio_) - d_log)
            upp = np.exp(np.log(odds_ratio_) + d_log)

        elif method == "score":
            low, upp = _score_confint_inversion(count1, nobs1, count2, nobs2,
                                                compare=compare, alpha=alpha,
                                                correction=correction)

        else:
            raise ValueError('method not recognized')

    else:
        raise ValueError('compare not recognized')

    return low, upp


def _shrink_prob(count1, nobs1, count2, nobs2, shrink_factor=2,
                 return_corr=True):
    """
    Shrink observed counts towards independence

    Helper function for 'logit-smoothed' inference for the odds-ratio of two
    independent proportions.

    Parameters
    ----------
    count1, nobs1 : float or int
        count and sample size for first sample
    count2, nobs2 : float or int
        count and sample size for the second sample
    shrink_factor : float
        This corresponds to the number of observations that are added in total
        proportional to the probabilities under independence.
    return_corr : bool
        If true, then only the correction term is returned
        If false, then the corrected counts, i.e. original counts plus
        correction term, are returned.

    Returns
    -------
    count1_corr, nobs1_corr, count2_corr, nobs2_corr : float
        correction or corrected counts
    prob_indep :
        TODO/Warning : this will change most likely
        probabilities under independence, only returned if return_corr is
        false.

    """
    vectorized = any(np.size(i) > 1 for i in [count1, nobs1, count2, nobs2])
    if vectorized:
        raise ValueError("function is not vectorized")
    nobs_col = np.array([count1 + count2, nobs1 - count1 + nobs2 - count2])
    nobs_row = np.array([nobs1, nobs2])
    nobs = nobs1 + nobs2
    prob_indep = (nobs_col * nobs_row[:, None]) / nobs**2
    corr = shrink_factor * prob_indep
    if return_corr:
        return (corr[0, 0], corr[0].sum(), corr[1, 0], corr[1].sum())
    else:
        return (count1 + corr[0, 0], nobs1 + corr[0].sum(),
                count2 + corr[1, 0], nobs2 + corr[1].sum()), prob_indep


def score_test_proportions_2indep(count1, nobs1, count2, nobs2, value=None,
                                  compare='diff', alternative='two-sided',
                                  correction=True, return_results=True):
    """
    Score test for two independent proportions

    This uses the constrained estimate of the proportions to compute
    the variance under the Null hypothesis.

    Parameters
    ----------
    count1, nobs1 :
        count and sample size for first sample
    count2, nobs2 :
        count and sample size for the second sample
    value : float
        diff, ratio or odds-ratio under the null hypothesis. If value is None,
        then equality of proportions under the Null is assumed,
        i.e. value=0 for 'diff' or value=1 for either rate or odds-ratio.
    compare : string in ['diff', 'ratio' 'odds-ratio']
        If compare is diff, then the confidence interval is for diff = p1 - p2.
        If compare is ratio, then the confidence interval is for the risk ratio
        defined by ratio = p1 / p2.
        If compare is odds-ratio, then the confidence interval is for the
        odds-ratio defined by or = p1 / (1 - p1) / (p2 / (1 - p2)
    return_results : bool
        If true, then a results instance with extra information is returned,
        otherwise a tuple with statistic and pvalue is returned.

    Returns
    -------
    results : results instance or tuple
        If return_results is True, then a results instance with the
        information in attributes is returned.
        If return_results is False, then only ``statistic`` and ``pvalue``
        are returned.

        statistic : float
            test statistic asymptotically normal distributed N(0, 1)
        pvalue : float
            p-value based on normal distribution
        other attributes :
            additional information about the hypothesis test

    Notes
    -----
    Status: experimental, the type or extra information in the return might
    change.

    """

    value_default = 0 if compare == 'diff' else 1
    if value is None:
        # TODO: odds ratio does not work if value=1
        value = value_default

    nobs = nobs1 + nobs2
    count = count1 + count2
    p1 = count1 / nobs1
    p2 = count2 / nobs2
    if value == value_default:
        # use pooled estimator if equality test
        # shortcut, but required for odds ratio
        prop0 = prop1 = count / nobs
    # this uses index 0 from Miettinen Nurminned 1985
    count0, nobs0 = count2, nobs2
    p0 = p2

    if compare == 'diff':
        diff = value  # hypothesis value

        if diff != 0:
            tmp3 = nobs
            tmp2 = (nobs1 + 2 * nobs0) * diff - nobs - count
            tmp1 = (count0 * diff - nobs - 2 * count0) * diff + count
            tmp0 = count0 * diff * (1 - diff)
            q = ((tmp2 / (3 * tmp3))**3 - tmp1 * tmp2 / (6 * tmp3**2) +
                 tmp0 / (2 * tmp3))
            p = np.sign(q) * np.sqrt((tmp2 / (3 * tmp3))**2 -
                                     tmp1 / (3 * tmp3))
            a = (np.pi + np.arccos(q / p**3)) / 3

            prop0 = 2 * p * np.cos(a) - tmp2 / (3 * tmp3)
            prop1 = prop0 + diff

        var = prop1 * (1 - prop1) / nobs1 + prop0 * (1 - prop0) / nobs0
        if correction:
            var *= nobs / (nobs - 1)

        diff_stat = (p1 - p0 - diff)

    elif compare == 'ratio':
        # risk ratio
        ratio = value

        if ratio != 1:
            a = nobs * ratio
            b = -(nobs1 * ratio + count1 + nobs2 + count0 * ratio)
            c = count
            prop0 = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)
            prop1 = prop0 * ratio

        var = (prop1 * (1 - prop1) / nobs1 +
               ratio**2 * prop0 * (1 - prop0) / nobs0)
        if correction:
            var *= nobs / (nobs - 1)

        # NCSS looks incorrect for var, but it is what should be reported
        # diff_stat = (p1 / p0 - ratio)   # NCSS/PASS
        diff_stat = (p1 - ratio * p0)  # Miettinen Nurminen

    elif compare in ['or', 'odds-ratio']:
        # odds ratio
        oratio = value

        if oratio != 1:
            # Note the constraint estimator does not handle odds-ratio = 1
            a = nobs0 * (oratio - 1)
            b = nobs1 * oratio + nobs0 - count * (oratio - 1)
            c = -count
            prop0 = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
            prop1 = prop0 * oratio / (1 + prop0 * (oratio - 1))

        # try to avoid 0 and 1 proportions,
        # those raise Zero Division Runtime Warnings
        eps = 1e-10
        prop0 = np.clip(prop0, eps, 1 - eps)
        prop1 = np.clip(prop1, eps, 1 - eps)

        var = (1 / (prop1 * (1 - prop1) * nobs1) +
               1 / (prop0 * (1 - prop0) * nobs0))
        if correction:
            var *= nobs / (nobs - 1)

        diff_stat = ((p1 - prop1) / (prop1 * (1 - prop1)) -
                     (p0 - prop0) / (prop0 * (1 - prop0)))

    statistic, pvalue = _zstat_generic2(diff_stat, np.sqrt(var),
                                        alternative=alternative)

    if return_results:
        res = HolderTuple(statistic=statistic,
                          pvalue=pvalue,
                          compare=compare,
                          method='score',
                          variance=var,
                          alternative=alternative,
                          prop1_null=prop1,
                          prop2_null=prop0,
                          )
        return res
    else:
        return statistic, pvalue


def test_proportions_2indep(count1, nobs1, count2, nobs2, value=None,
                            method=None, compare='diff',
                            alternative='two-sided', correction=True,
                            return_results=True):
    """
    Hypothesis test for comparing two independent proportions

    This assumes that we have two independent binomial samples.

    The Null and alternative hypothesis are

    for compare = 'diff'

    - H0: prop1 - prop2 - value = 0
    - H1: prop1 - prop2 - value != 0  if alternative = 'two-sided'
    - H1: prop1 - prop2 - value > 0   if alternative = 'larger'
    - H1: prop1 - prop2 - value < 0   if alternative = 'smaller'

    for compare = 'ratio'

    - H0: prop1 / prop2 - value = 0
    - H1: prop1 / prop2 - value != 0  if alternative = 'two-sided'
    - H1: prop1 / prop2 - value > 0   if alternative = 'larger'
    - H1: prop1 / prop2 - value < 0   if alternative = 'smaller'

    for compare = 'odds-ratio'

    - H0: or - value = 0
    - H1: or - value != 0  if alternative = 'two-sided'
    - H1: or - value > 0   if alternative = 'larger'
    - H1: or - value < 0   if alternative = 'smaller'

    where odds-ratio or = prop1 / (1 - prop1) / (prop2 / (1 - prop2))

    Parameters
    ----------
    count1 : int
        Count for first sample.
    nobs1 : int
        Sample size for first sample.
    count2 : int
        Count for the second sample.
    nobs2 : int
        Sample size for the second sample.
    value : float
        Value of the difference, risk ratio or odds ratio of 2 independent
        proportions under the null hypothesis.
        Default is equal proportions, 0 for diff and 1 for risk-ratio and for
        odds-ratio.
    method : string
        Method for computing the hypothesis test. If method is None, then a
        default method is used. The default might change as more methods are
        added.

        diff:

        - 'wald',
        - 'agresti-caffo'
        - 'score' if correction is True, then this uses the degrees of freedom
           correction ``nobs / (nobs - 1)`` as in Miettinen Nurminen 1985

        ratio:

        - 'log': wald test using log transformation
        - 'log-adjusted': wald test using log transformation,
           adds 0.5 to counts
        - 'score': if correction is True, then this uses the degrees of freedom
           correction ``nobs / (nobs - 1)`` as in Miettinen Nurminen 1985

        odds-ratio:

        - 'logit': wald test using logit transformation
        - 'logit-adjusted': wald test using logit transformation,
           adds 0.5 to counts
        - 'logit-smoothed': wald test using logit transformation, biases
           cell counts towards independence by adding two observations in
           total.
        - 'score' if correction is True, then this uses the degrees of freedom
           correction ``nobs / (nobs - 1)`` as in Miettinen Nurminen 1985

    compare : {'diff', 'ratio' 'odds-ratio'}
        If compare is `diff`, then the hypothesis test is for the risk
        difference diff = p1 - p2.
        If compare is `ratio`, then the hypothesis test is for the
        risk ratio defined by ratio = p1 / p2.
        If compare is `odds-ratio`, then the hypothesis test is for the
        odds-ratio defined by or = p1 / (1 - p1) / (p2 / (1 - p2)
    alternative : {'two-sided', 'smaller', 'larger'}
        alternative hypothesis, which can be two-sided or either one of the
        one-sided tests.
    correction : bool
        If correction is True (default), then the Miettinen and Nurminen
        small sample correction to the variance nobs / (nobs - 1) is used.
        Applies only if method='score'.
    return_results : bool
        If true, then a results instance with extra information is returned,
        otherwise a tuple with statistic and pvalue is returned.

    Returns
    -------
    results : results instance or tuple
        If return_results is True, then a results instance with the
        information in attributes is returned.
        If return_results is False, then only ``statistic`` and ``pvalue``
        are returned.

        statistic : float
            test statistic asymptotically normal distributed N(0, 1)
        pvalue : float
            p-value based on normal distribution
        other attributes :
            additional information about the hypothesis test

    See Also
    --------
    tost_proportions_2indep
    confint_proportions_2indep

    Notes
    -----
    Status: experimental, API and defaults might still change.
        More ``methods`` will be added.

    The current default methods are

    - 'diff': 'agresti-caffo',
    - 'ratio': 'log-adjusted',
    - 'odds-ratio': 'logit-adjusted'

    """
    method_default = {'diff': 'agresti-caffo',
                      'ratio': 'log-adjusted',
                      'odds-ratio': 'logit-adjusted'}
    # normalize compare name
    if compare.lower() == 'or':
        compare = 'odds-ratio'
    if method is None:
        method = method_default[compare]

    method = method.lower()
    if method.startswith('agr'):
        method = 'agresti-caffo'

    if value is None:
        # TODO: odds ratio does not work if value=1 for score test
        value = 0 if compare == 'diff' else 1

    count1, nobs1, count2, nobs2 = map(np.asarray,
                                       [count1, nobs1, count2, nobs2])

    p1 = count1 / nobs1
    p2 = count2 / nobs2
    diff = p1 - p2
    ratio = p1 / p2
    odds_ratio = p1 / (1 - p1) / p2 * (1 - p2)
    res = None

    if compare == 'diff':
        if method in ['wald', 'agresti-caffo']:
            addone = 1 if method == 'agresti-caffo' else 0
            count1_, nobs1_ = count1 + addone, nobs1 + 2 * addone
            count2_, nobs2_ = count2 + addone, nobs2 + 2 * addone
            p1_ = count1_ / nobs1_
            p2_ = count2_ / nobs2_
            diff_stat = p1_ - p2_ - value
            var = p1_ * (1 - p1_) / nobs1_ + p2_ * (1 - p2_) / nobs2_
            statistic = diff_stat / np.sqrt(var)
            distr = 'normal'

        elif method.startswith('newcomb'):
            msg = 'newcomb not available for hypothesis test'
            raise NotImplementedError(msg)

        elif method == 'score':
            # Note score part is the same call for all compare
            res = score_test_proportions_2indep(count1, nobs1, count2, nobs2,
                                                value=value, compare=compare,
                                                alternative=alternative,
                                                correction=correction,
                                                return_results=return_results)
            if return_results is False:
                statistic, pvalue = res[:2]
            distr = 'normal'
            # TODO/Note score_test_proportion_2samp returns statistic  and
            #     not diff_stat
            diff_stat = None
        else:
            raise ValueError('method not recognized')

    elif compare == 'ratio':
        if method in ['log', 'log-adjusted']:
            addhalf = 0.5 if method == 'log-adjusted' else 0
            count1_, nobs1_ = count1 + addhalf, nobs1 + addhalf
            count2_, nobs2_ = count2 + addhalf, nobs2 + addhalf
            p1_ = count1_ / nobs1_
            p2_ = count2_ / nobs2_
            ratio_ = p1_ / p2_
            var = (1 / count1_) - 1 / nobs1_ + 1 / count2_ - 1 / nobs2_
            diff_stat = np.log(ratio_) - np.log(value)
            statistic = diff_stat / np.sqrt(var)
            distr = 'normal'

        elif method == 'score':
            res = score_test_proportions_2indep(count1, nobs1, count2, nobs2,
                                                value=value, compare=compare,
                                                alternative=alternative,
                                                correction=correction,
                                                return_results=return_results)
            if return_results is False:
                statistic, pvalue = res[:2]
            distr = 'normal'
            diff_stat = None

        else:
            raise ValueError('method not recognized')

    elif compare == "odds-ratio":

        if method in ['logit', 'logit-adjusted', 'logit-smoothed']:
            if method in ['logit-smoothed']:
                adjusted = _shrink_prob(count1, nobs1, count2, nobs2,
                                        shrink_factor=2, return_corr=False)[0]
                count1_, nobs1_, count2_, nobs2_ = adjusted

            else:
                addhalf = 0.5 if method == 'logit-adjusted' else 0
                count1_, nobs1_ = count1 + addhalf, nobs1 + 2 * addhalf
                count2_, nobs2_ = count2 + addhalf, nobs2 + 2 * addhalf
            p1_ = count1_ / nobs1_
            p2_ = count2_ / nobs2_
            odds_ratio_ = p1_ / (1 - p1_) / p2_ * (1 - p2_)
            var = (1 / count1_ + 1 / (nobs1_ - count1_) +
                   1 / count2_ + 1 / (nobs2_ - count2_))

            diff_stat = np.log(odds_ratio_) - np.log(value)
            statistic = diff_stat / np.sqrt(var)
            distr = 'normal'

        elif method == 'score':
            res = score_test_proportions_2indep(count1, nobs1, count2, nobs2,
                                                value=value, compare=compare,
                                                alternative=alternative,
                                                correction=correction,
                                                return_results=return_results)
            if return_results is False:
                statistic, pvalue = res[:2]
            distr = 'normal'
            diff_stat = None
        else:
            raise ValueError('method "%s" not recognized' % method)

    else:
        raise ValueError('compare "%s" not recognized' % compare)

    if distr == 'normal' and diff_stat is not None:
        statistic, pvalue = _zstat_generic2(diff_stat, np.sqrt(var),
                                            alternative=alternative)

    if return_results:
        if res is None:
            res = HolderTuple(statistic=statistic,
                              pvalue=pvalue,
                              compare=compare,
                              method=method,
                              diff=diff,
                              ratio=ratio,
                              odds_ratio=odds_ratio,
                              variance=var,
                              alternative=alternative,
                              value=value,
                              )
        else:
            # we already have a return result from score test
            # add missing attributes
            res.diff = diff
            res.ratio = ratio
            res.odds_ratio = odds_ratio
            res.value = value
        return res
    else:
        return statistic, pvalue


def tost_proportions_2indep(count1, nobs1, count2, nobs2, low, upp,
                            method=None, compare='diff', correction=True):
    """
    Equivalence test based on two one-sided `test_proportions_2indep`

    This assumes that we have two independent binomial samples.

    The Null and alternative hypothesis for equivalence testing are

    for compare = 'diff'

    - H0: prop1 - prop2 <= low or upp <= prop1 - prop2
    - H1: low < prop1 - prop2 < upp

    for compare = 'ratio'

    - H0: prop1 / prop2 <= low or upp <= prop1 / prop2
    - H1: low < prop1 / prop2 < upp


    for compare = 'odds-ratio'

    - H0: or <= low or upp <= or
    - H1: low < or < upp

    where odds-ratio or = prop1 / (1 - prop1) / (prop2 / (1 - prop2))

    Parameters
    ----------
    count1, nobs1 :
        count and sample size for first sample
    count2, nobs2 :
        count and sample size for the second sample
    low, upp :
        equivalence margin for diff, risk ratio or odds ratio
    method : string
        method for computing the hypothesis test. If method is None, then a
        default method is used. The default might change as more methods are
        added.

        diff:
         - 'wald',
         - 'agresti-caffo'
         - 'score' if correction is True, then this uses the degrees of freedom
           correction ``nobs / (nobs - 1)`` as in Miettinen Nurminen 1985.

        ratio:
         - 'log': wald test using log transformation
         - 'log-adjusted': wald test using log transformation,
            adds 0.5 to counts
         - 'score' if correction is True, then this uses the degrees of freedom
           correction ``nobs / (nobs - 1)`` as in Miettinen Nurminen 1985.

        odds-ratio:
         - 'logit': wald test using logit transformation
         - 'logit-adjusted': : wald test using logit transformation,
            adds 0.5 to counts
         - 'logit-smoothed': : wald test using logit transformation, biases
            cell counts towards independence by adding two observations in
            total.
         - 'score' if correction is True, then this uses the degrees of freedom
            correction ``nobs / (nobs - 1)`` as in Miettinen Nurminen 1985

    compare : string in ['diff', 'ratio' 'odds-ratio']
        If compare is `diff`, then the hypothesis test is for
        diff = p1 - p2.
        If compare is `ratio`, then the hypothesis test is for the
        risk ratio defined by ratio = p1 / p2.
        If compare is `odds-ratio`, then the hypothesis test is for the
        odds-ratio defined by or = p1 / (1 - p1) / (p2 / (1 - p2).
    correction : bool
        If correction is True (default), then the Miettinen and Nurminen
        small sample correction to the variance nobs / (nobs - 1) is used.
        Applies only if method='score'.

    Returns
    -------
    pvalue : float
        p-value is the max of the pvalues of the two one-sided tests
    t1 : test results
        results instance for one-sided hypothesis at the lower margin
    t1 : test results
        results instance for one-sided hypothesis at the upper margin

    See Also
    --------
    test_proportions_2indep
    confint_proportions_2indep

    Notes
    -----
    Status: experimental, API and defaults might still change.

    The TOST equivalence test delegates to `test_proportions_2indep` and has
    the same method and comparison options.

    """

    tt1 = test_proportions_2indep(count1, nobs1, count2, nobs2, value=low,
                                  method=method, compare=compare,
                                  alternative='larger',
                                  correction=correction,
                                  return_results=True)
    tt2 = test_proportions_2indep(count1, nobs1, count2, nobs2, value=upp,
                                  method=method, compare=compare,
                                  alternative='smaller',
                                  correction=correction,
                                  return_results=True)

    # idx_max = 1 if t1.pvalue < t2.pvalue else 0
    idx_max = np.asarray(tt1.pvalue < tt2.pvalue, int)
    statistic = np.choose(idx_max, [tt1.statistic, tt2.statistic])
    pvalue = np.choose(idx_max, [tt1.pvalue, tt2.pvalue])

    res = HolderTuple(statistic=statistic,
                      pvalue=pvalue,
                      compare=compare,
                      method=method,
                      results_larger=tt1,
                      results_smaller=tt2,
                      title="Equivalence test for 2 independent proportions"
                      )

    return res


def _std_2prop_power(diff, p2, ratio=1, alpha=0.05, value=0):
    """
    Compute standard error under null and alternative for 2 proportions

    helper function for power and sample size computation

    """
    if value != 0:
        msg = 'non-zero diff under null, value, is not yet implemented'
        raise NotImplementedError(msg)

    nobs_ratio = ratio
    p1 = p2 + diff
    # The following contains currently redundant variables that will
    # be useful for different options for the null variance
    p_pooled = (p1 + p2 * ratio) / (1 + ratio)
    # probabilities for the variance for the null statistic
    p1_vnull, p2_vnull = p_pooled, p_pooled
    p2_alt = p2
    p1_alt = p2_alt + diff

    std_null = _std_diff_prop(p1_vnull, p2_vnull, ratio=nobs_ratio)
    std_alt = _std_diff_prop(p1_alt, p2_alt, ratio=nobs_ratio)
    return p_pooled, std_null, std_alt


def power_proportions_2indep(diff, prop2, nobs1, ratio=1, alpha=0.05,
                             value=0, alternative='two-sided',
                             return_results=True):
    """
    Power for ztest that two independent proportions are equal

    This assumes that the variance is based on the pooled proportion
    under the null and the non-pooled variance under the alternative

    Parameters
    ----------
    diff : float
        difference between proportion 1 and 2 under the alternative
    prop2 : float
        proportion for the reference case, prop2, proportions for the
        first case will be computed using p2 and diff
        p1 = p2 + diff
    nobs1 : float or int
        number of observations in sample 1
    ratio : float
        sample size ratio, nobs2 = ratio * nobs1
    alpha : float in interval (0,1)
        Significance level, e.g. 0.05, is the probability of a type I
        error, that is wrong rejections if the Null Hypothesis is true.
    value : float
        currently only `value=0`, i.e. equality testing, is supported
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
        If return_results is True, then a results instance with the
        information in attributes is returned.
        If return_results is False, then only the power is returned.

        power : float
            Power of the test, e.g. 0.8, is one minus the probability of a
            type II error. Power is the probability that the test correctly
            rejects the Null Hypothesis if the Alternative Hypothesis is true.

        Other attributes in results instance include :

        p_pooled
            pooled proportion, used for std_null
        std_null
            standard error of difference under the null hypothesis (without
            sqrt(nobs1))
        std_alt
            standard error of difference under the alternative hypothesis
            (without sqrt(nobs1))
    """
    # TODO: avoid possible circular import, check if needed
    from statsmodels.stats.power import normal_power_het

    p_pooled, std_null, std_alt = _std_2prop_power(diff, prop2, ratio=ratio,
                                                   alpha=alpha, value=value)

    pow_ = normal_power_het(diff, nobs1, alpha, std_null=std_null,
                            std_alternative=std_alt,
                            alternative=alternative)

    if return_results:
        res = Holder(power=pow_,
                     p_pooled=p_pooled,
                     std_null=std_null,
                     std_alt=std_alt,
                     nobs1=nobs1,
                     nobs2=ratio * nobs1,
                     nobs_ratio=ratio,
                     alpha=alpha,
                     )
        return res
    else:
        return pow_


def samplesize_proportions_2indep_onetail(diff, prop2, power, ratio=1,
                                          alpha=0.05, value=0,
                                          alternative='two-sided'):
    """
    Required sample size assuming normal distribution based on one tail

    This uses an explicit computation for the sample size that is required
    to achieve a given power corresponding to the appropriate tails of the
    normal distribution. This ignores the far tail in a two-sided test
    which is negligible in the common case when alternative and null are
    far apart.

    Parameters
    ----------
    diff : float
        Difference between proportion 1 and 2 under the alternative
    prop2 : float
        proportion for the reference case, prop2, proportions for the
        first case will be computing using p2 and diff
        p1 = p2 + diff
    power : float
        Power for which sample size is computed.
    ratio : float
        Sample size ratio, nobs2 = ratio * nobs1
    alpha : float in interval (0,1)
        Significance level, e.g. 0.05, is the probability of a type I
        error, that is wrong rejections if the Null Hypothesis is true.
    value : float
        Currently only `value=0`, i.e. equality testing, is supported
    alternative : string, 'two-sided' (default), 'larger', 'smaller'
        Alternative hypothesis whether the power is calculated for a
        two-sided (default) or one sided test. In the case of a one-sided
        alternative, it is assumed that the test is in the appropriate tail.

    Returns
    -------
    nobs1 : float
        Number of observations in sample 1.
    """
    # TODO: avoid possible circular import, check if needed
    from statsmodels.stats.power import normal_sample_size_one_tail

    if alternative in ['two-sided', '2s']:
        alpha = alpha / 2

    _, std_null, std_alt = _std_2prop_power(diff, prop2, ratio=ratio,
                                            alpha=alpha, value=value)

    nobs = normal_sample_size_one_tail(diff, power, alpha, std_null=std_null,
                                       std_alternative=std_alt)
    return nobs


def _score_confint_inversion(count1, nobs1, count2, nobs2, compare='diff',
                             alpha=0.05, correction=True):
    """
    Compute score confidence interval by inverting score test

    Parameters
    ----------
    count1, nobs1 :
        Count and sample size for first sample.
    count2, nobs2 :
        Count and sample size for the second sample.
    compare : string in ['diff', 'ratio' 'odds-ratio']
        If compare is `diff`, then the confidence interval is for
        diff = p1 - p2.
        If compare is `ratio`, then the confidence interval is for the
        risk ratio defined by ratio = p1 / p2.
        If compare is `odds-ratio`, then the confidence interval is for the
        odds-ratio defined by or = p1 / (1 - p1) / (p2 / (1 - p2).
    alpha : float in interval (0,1)
        Significance level, e.g. 0.05, is the probability of a type I
        error, that is wrong rejections if the Null Hypothesis is true.
    correction : bool
        If correction is True (default), then the Miettinen and Nurminen
        small sample correction to the variance nobs / (nobs - 1) is used.
        Applies only if method='score'.

    Returns
    -------
    low : float
        Lower confidence bound.
    upp : float
        Upper confidence bound.
    """

    def func(v):
        r = test_proportions_2indep(count1, nobs1, count2, nobs2,
                                    value=v, compare=compare, method='score',
                                    correction=correction,
                                    alternative="two-sided")
        return r.pvalue - alpha

    rt0 = test_proportions_2indep(count1, nobs1, count2, nobs2,
                                  value=0, compare=compare, method='score',
                                  correction=correction,
                                  alternative="two-sided")

    # use default method to get starting values
    # this will not work if score confint becomes default
    # maybe use "wald" as alias that works for all compare statistics
    use_method = {"diff": "wald", "ratio": "log", "odds-ratio": "logit"}
    rci0 = confint_proportions_2indep(count1, nobs1, count2, nobs2,
                                      method=use_method[compare],
                                      compare=compare, alpha=alpha)

    # Note diff might be negative
    ub = rci0[1] + np.abs(rci0[1]) * 0.5
    lb = rci0[0] - np.abs(rci0[0]) * 0.25
    if compare == 'diff':
        param = rt0.diff
        # 1 might not be the correct upper bound because
        #     rootfinding is for the `diff` and not for a probability.
        ub = min(ub, 0.99999)
    elif compare == 'ratio':
        param = rt0.ratio
        ub *= 2  # add more buffer
    if compare == 'odds-ratio':
        param = rt0.odds_ratio

    # root finding for confint bounds
    upp = optimize.brentq(func, param, ub)
    low = optimize.brentq(func, lb, param)
    return low, upp


def _confint_riskratio_koopman(count1, nobs1, count2, nobs2, alpha=0.05,
                               correction=True):
    """
    Score confidence interval for ratio or proportions, Koopman/Nam

    signature not consistent with other functions

    When correction is True, then the small sample correction nobs / (nobs - 1)
    by Miettinen/Nurminen is used.
    """
    # The names below follow Nam
    x0, x1, n0, n1 = count2, count1, nobs2, nobs1
    x = x0 + x1
    n = n0 + n1
    z = stats.norm.isf(alpha / 2)**2
    if correction:
        # Mietinnen/Nurminen small sample correction
        z *= n / (n - 1)
    # z = stats.chi2.isf(alpha, 1)
    # equ 6 in Nam 1995
    a1 = n0 * (n0 * n * x1 + n1 * (n0 + x1) * z)
    a2 = - n0 * (n0 * n1 * x + 2 * n * x0 * x1 + n1 * (n0 + x0 + 2 * x1) * z)
    a3 = 2 * n0 * n1 * x0 * x + n * x0 * x0 * x1 + n0 * n1 * x * z
    a4 = - n1 * x0 * x0 * x

    p_roots_ = np.sort(np.roots([a1, a2, a3, a4]))
    p_roots = p_roots_[:2][::-1]

    # equ 5
    ci = (1 - (n1 - x1) * (1 - p_roots) / (x0 + n1 - n * p_roots)) / p_roots

    res = Holder()
    res.confint = ci
    res._p_roots = p_roots_  # for unit tests, can be dropped
    return res


def _confint_riskratio_paired_nam(table, alpha=0.05):
    """
    Confidence interval for marginal risk ratio for matched pairs

    need full table

             success fail  marginal
    success    x11    x10  x1.
    fail       x01    x00  x0.
    marginal   x.1    x.0   n

    The confidence interval is for the ratio p1 / p0 where
    p1 = x1. / n and
    p0 - x.1 / n
    Todo: rename p1 to pa and p2 to pb, so we have a, b for treatment and
    0, 1 for success/failure

    current namings follow Nam 2009

    status
    testing:
    compared to example in Nam 2009
    internal polynomial coefficients in calculation correspond at around
        4 decimals
    confidence interval agrees only at 2 decimals

    """
    x11, x10, x01, x00 = np.ravel(table)
    n = np.sum(table)  # nobs
    p10, p01 = x10 / n, x01 / n
    p1 = (x11 + x10) / n
    p0 = (x11 + x01) / n
    q00 = 1 - x00 / n

    z2 = stats.norm.isf(alpha / 2)**2
    # z = stats.chi2.isf(alpha, 1)
    # before equ 3 in Nam 2009

    g1 = (n * p0 + z2 / 2) * p0
    g2 = - (2 * n * p1 * p0 + z2 * q00)
    g3 = (n * p1 + z2 / 2) * p1

    a0 = g1**2 - (z2 * p0 / 2)**2
    a1 = 2 * g1 * g2
    a2 = g2**2 + 2 * g1 * g3 + z2**2 * (p1 * p0 - 2 * p10 * p01) / 2
    a3 = 2 * g2 * g3
    a4 = g3**2 - (z2 * p1 / 2)**2

    p_roots = np.sort(np.roots([a0, a1, a2, a3, a4]))
    # p_roots = np.sort(np.roots([1, a1 / a0, a2 / a0, a3 / a0, a4 / a0]))

    ci = [p_roots.min(), p_roots.max()]
    res = Holder()
    res.confint = ci
    res.p = p1, p0
    res._p_roots = p_roots  # for unit tests, can be dropped
    return res
