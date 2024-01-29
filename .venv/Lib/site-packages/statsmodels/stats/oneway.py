# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 10:33:38 2020

Author: Josef Perktold
License: BSD-3

"""

import numpy as np
from scipy import stats
from scipy.special import ncfdtrinc

# functions that use scipy.special instead of boost based function in stats
from statsmodels.stats.power import ncf_cdf, ncf_ppf

from statsmodels.stats.robust_compare import TrimmedMean, scale_transform
from statsmodels.tools.testing import Holder
from statsmodels.stats.base import HolderTuple


def effectsize_oneway(means, vars_, nobs, use_var="unequal", ddof_between=0):
    """
    Effect size corresponding to Cohen's f = nc / nobs for oneway anova

    This contains adjustment for Welch and Brown-Forsythe Anova so that
    effect size can be used with FTestAnovaPower.

    Parameters
    ----------
    means : array_like
        Mean of samples to be compared
    vars_ : float or array_like
        Residual (within) variance of each sample or pooled
        If ``vars_`` is scalar, then it is interpreted as pooled variance that
        is the same for all samples, ``use_var`` will be ignored.
        Otherwise, the variances are used depending on the ``use_var`` keyword.
    nobs : int or array_like
        Number of observations for the samples.
        If nobs is scalar, then it is assumed that all samples have the same
        number ``nobs`` of observation, i.e. a balanced sample case.
        Otherwise, statistics will be weighted corresponding to nobs.
        Only relative sizes are relevant, any proportional change to nobs does
        not change the effect size.
    use_var : {"unequal", "equal", "bf"}
        If ``use_var`` is "unequal", then the variances can differ across
        samples and the effect size for Welch anova will be computed.
    ddof_between : int
        Degrees of freedom correction for the weighted between sum of squares.
        The denominator is ``nobs_total - ddof_between``
        This can be used to match differences across reference literature.

    Returns
    -------
    f2 : float
        Effect size corresponding to squared Cohen's f, which is also equal
        to the noncentrality divided by total number of observations.

    Notes
    -----
    This currently handles the following cases for oneway anova

    - balanced sample with homoscedastic variances
    - samples with different number of observations and with homoscedastic
      variances
    - samples with different number of observations and with heteroskedastic
      variances. This corresponds to Welch anova

    In the case of "unequal" and "bf" methods for unequal variances, the
    effect sizes do not directly correspond to the test statistic in Anova.
    Both have correction terms dropped or added, so the effect sizes match up
    with using FTestAnovaPower.
    If all variances are equal, then all three methods result in the same
    effect size. If variances are unequal, then the three methods produce
    small differences in effect size.

    Note, the effect size and power computation for BF Anova was not found in
    the literature. The correction terms were added so that FTestAnovaPower
    provides a good approximation to the power.

    Status: experimental
    We might add additional returns, if those are needed to support power
    and sample size applications.

    Examples
    --------
    The following shows how to compute effect size and power for each of the
    three anova methods. The null hypothesis is that the means are equal which
    corresponds to a zero effect size. Under the alternative, means differ
    with two sample means at a distance delta from the mean. We assume the
    variance is the same under the null and alternative hypothesis.

    ``nobs`` for the samples defines the fraction of observations in the
    samples. ``nobs`` in the power method defines the total sample size.

    In simulations, the computed power for standard anova,
    i.e.``use_var="equal"`` overestimates the simulated power by a few percent.
    The equal variance assumption does not hold in this example.

    >>> from statsmodels.stats.oneway import effectsize_oneway
    >>> from statsmodels.stats.power import FTestAnovaPower
    >>>
    >>> nobs = np.array([10, 12, 13, 15])
    >>> delta = 0.5
    >>> means_alt = np.array([-1, 0, 0, 1]) * delta
    >>> vars_ = np.arange(1, len(means_alt) + 1)
    >>>
    >>> f2_alt = effectsize_oneway(means_alt, vars_, nobs, use_var="equal")
    >>> f2_alt
    0.04581300813008131
    >>>
    >>> kwds = {'effect_size': np.sqrt(f2_alt), 'nobs': 100, 'alpha': 0.05,
    ...         'k_groups': 4}
    >>> power = FTestAnovaPower().power(**kwds)
    >>> power
    0.39165892158983273
    >>>
    >>> f2_alt = effectsize_oneway(means_alt, vars_, nobs, use_var="unequal")
    >>> f2_alt
    0.060640138408304504
    >>>
    >>> kwds['effect_size'] = np.sqrt(f2_alt)
    >>> power = FTestAnovaPower().power(**kwds)
    >>> power
    0.5047366512800622
    >>>
    >>> f2_alt = effectsize_oneway(means_alt, vars_, nobs, use_var="bf")
    >>> f2_alt
    0.04391324307956788
    >>>
    >>> kwds['effect_size'] = np.sqrt(f2_alt)
    >>> power = FTestAnovaPower().power(**kwds)
    >>> power
    0.3765792117047725

    """
    # the code here is largely a copy of onway_generic with adjustments

    means = np.asarray(means)
    n_groups = means.shape[0]

    if np.size(nobs) == 1:
        nobs = np.ones(n_groups) * nobs

    nobs_t = nobs.sum()

    if use_var == "equal":
        if np.size(vars_) == 1:
            var_resid = vars_
        else:
            vars_ = np.asarray(vars_)
            var_resid = ((nobs - 1) * vars_).sum() / (nobs_t - n_groups)

        vars_ = var_resid  # scalar, if broadcasting works

    weights = nobs / vars_

    w_total = weights.sum()
    w_rel = weights / w_total
    # meanw_t = (weights * means).sum() / w_total
    meanw_t = w_rel @ means

    f2 = np.dot(weights, (means - meanw_t)**2) / (nobs_t - ddof_between)

    if use_var.lower() == "bf":
        weights = nobs
        w_total = weights.sum()
        w_rel = weights / w_total
        meanw_t = w_rel @ means
        # TODO: reuse general case with weights
        tmp = ((1. - nobs / nobs_t) * vars_).sum()
        statistic = 1. * (nobs * (means - meanw_t)**2).sum()
        statistic /= tmp
        f2 = statistic * (1. - nobs / nobs_t).sum() / nobs_t
        # correction factor for df_num in BFM
        df_num2 = n_groups - 1
        df_num = tmp**2 / ((vars_**2).sum() +
                           (nobs / nobs_t * vars_).sum()**2 -
                           2 * (nobs / nobs_t * vars_**2).sum())
        f2 *= df_num / df_num2

    return f2


def convert_effectsize_fsqu(f2=None, eta2=None):
    """Convert squared effect sizes in f family

    f2 is signal to noise ratio, var_explained / var_residual

    eta2 is proportion of explained variance, var_explained / var_total

    uses the relationship:
    f2 = eta2 / (1 - eta2)

    Parameters
    ----------
    f2 : None or float
       Squared Cohen's F effect size. If f2 is not None, then eta2 will be
       computed.
    eta2 : None or float
       Squared eta effect size. If f2 is None and eta2 is not None, then f2 is
       computed.

    Returns
    -------
    res : Holder instance
        An instance of the Holder class with f2 and eta2 as attributes.

    """
    if f2 is not None:
        eta2 = 1 / (1 + 1 / f2)

    elif eta2 is not None:
        f2 = eta2 / (1 - eta2)

    res = Holder(f2=f2, eta2=eta2)
    return res


def _fstat2effectsize(f_stat, df):
    """Compute anova effect size from F-statistic

    This might be combined with convert_effectsize_fsqu

    Parameters
    ----------
    f_stat : array_like
        Test statistic of an F-test
    df : tuple
        degrees of freedom ``df = (df1, df2)`` where
         - df1 : numerator degrees of freedom, number of constraints
         - df2 : denominator degrees of freedom, df_resid

    Returns
    -------
    res : Holder instance
        This instance contains effect size measures f2, eta2, omega2 and eps2
        as attributes.

    Notes
    -----
    This uses the following definitions:

    - f2 = f_stat * df1 / df2
    - eta2 = f2 / (f2 + 1)
    - omega2 = (f2 - df1 / df2) / (f2 + 2)
    - eps2 = (f2 - df1 / df2) / (f2 + 1)

    This differs from effect size measures in other function which define
    ``f2 = f_stat * df1 / nobs``
    or an equivalent expression for power computation. The noncentrality
    index for the hypothesis test is in those cases given by
    ``nc = f_stat * df1``.

    Currently omega2 and eps2 are computed in two different ways. Those
    values agree for regular cases but can show different behavior in corner
    cases (e.g. zero division).

    """
    df1, df2 = df
    f2 = f_stat * df1 / df2
    eta2 = f2 / (f2 + 1)
    omega2_ = (f_stat - 1) / (f_stat + (df2 + 1) / df1)
    omega2 = (f2 - df1 / df2) / (f2 + 1 + 1 / df2)  # rewrite
    eps2_ = (f_stat - 1) / (f_stat + df2 / df1)
    eps2 = (f2 - df1 / df2) / (f2 + 1)  # rewrite
    return Holder(f2=f2, eta2=eta2, omega2=omega2, eps2=eps2, eps2_=eps2_,
                  omega2_=omega2_)


# conversion functions for Wellek's equivalence effect size
# these are mainly to compare with literature

def wellek_to_f2(eps, n_groups):
    """Convert Wellek's effect size (sqrt) to Cohen's f-squared

    This computes the following effect size :

       f2 = 1 / n_groups * eps**2

    Parameters
    ----------
    eps : float or ndarray
        Wellek's effect size used in anova equivalence test
    n_groups : int
        Number of groups in oneway comparison

    Returns
    -------
    f2 : effect size Cohen's f-squared

    """
    f2 = 1 / n_groups * eps**2
    return f2


def f2_to_wellek(f2, n_groups):
    """Convert Cohen's f-squared to Wellek's effect size (sqrt)

    This computes the following effect size :

       eps = sqrt(n_groups * f2)

    Parameters
    ----------
    f2 : float or ndarray
        Effect size Cohen's f-squared
    n_groups : int
        Number of groups in oneway comparison

    Returns
    -------
    eps : float or ndarray
        Wellek's effect size used in anova equivalence test
    """
    eps = np.sqrt(n_groups * f2)
    return eps


def fstat_to_wellek(f_stat, n_groups, nobs_mean):
    """Convert F statistic to wellek's effect size eps squared

    This computes the following effect size :

       es = f_stat * (n_groups - 1) / nobs_mean

    Parameters
    ----------
    f_stat : float or ndarray
        Test statistic of an F-test.
    n_groups : int
        Number of groups in oneway comparison
    nobs_mean : float or ndarray
        Average number of observations across groups.

    Returns
    -------
    eps : float or ndarray
        Wellek's effect size used in anova equivalence test

    """
    es = f_stat * (n_groups - 1) / nobs_mean
    return es


def confint_noncentrality(f_stat, df, alpha=0.05,
                          alternative="two-sided"):
    """
    Confidence interval for noncentrality parameter in F-test

    This does not yet handle non-negativity constraint on nc.
    Currently only two-sided alternative is supported.

    Parameters
    ----------
    f_stat : float
    df : tuple
        degrees of freedom ``df = (df1, df2)`` where

        - df1 : numerator degrees of freedom, number of constraints
        - df2 : denominator degrees of freedom, df_resid

    alpha : float, default 0.05
    alternative : {"two-sided"}
        Other alternatives have not been implements.

    Returns
    -------
    float
        The end point of the confidence interval.

    Notes
    -----
    The algorithm inverts the cdf of the noncentral F distribution with
    respect to the noncentrality parameters.
    See Steiger 2004 and references cited in it.

    References
    ----------
    .. [1] Steiger, James H. 2004. “Beyond the F Test: Effect Size Confidence
       Intervals and Tests of Close Fit in the Analysis of Variance and
       Contrast Analysis.” Psychological Methods 9 (2): 164–82.
       https://doi.org/10.1037/1082-989X.9.2.164.

    See Also
    --------
    confint_effectsize_oneway
    """

    df1, df2 = df
    if alternative in ["two-sided", "2s", "ts"]:
        alpha1s = alpha / 2
        ci = ncfdtrinc(df1, df2, [1 - alpha1s, alpha1s], f_stat)
    else:
        raise NotImplementedError

    return ci


def confint_effectsize_oneway(f_stat, df, alpha=0.05, nobs=None):
    """
    Confidence interval for effect size in oneway anova for F distribution

    This does not yet handle non-negativity constraint on nc.
    Currently only two-sided alternative is supported.

    Parameters
    ----------
    f_stat : float
    df : tuple
        degrees of freedom ``df = (df1, df2)`` where

        - df1 : numerator degrees of freedom, number of constraints
        - df2 : denominator degrees of freedom, df_resid

    alpha : float, default 0.05
    nobs : int, default None

    Returns
    -------
    Holder
        Class with effect size and confidence attributes

    Notes
    -----
    The confidence interval for the noncentrality parameter is obtained by
    inverting the cdf of the noncentral F distribution. Confidence intervals
    for other effect sizes are computed by endpoint transformation.


    R package ``effectsize`` does not compute the confidence intervals in the
    same way. Their confidence intervals can be replicated with

    >>> ci_nc = confint_noncentrality(f_stat, df1, df2, alpha=0.1)
    >>> ci_es = smo._fstat2effectsize(ci_nc / df1, df1, df2)

    See Also
    --------
    confint_noncentrality
    """

    df1, df2 = df
    if nobs is None:
        nobs = df1 + df2 + 1
    ci_nc = confint_noncentrality(f_stat, df, alpha=alpha)

    ci_f2 = ci_nc / nobs
    ci_res = convert_effectsize_fsqu(f2=ci_f2)
    ci_res.ci_omega2 = (ci_f2 - df1 / df2) / (ci_f2 + 1 + 1 / df2)
    ci_res.ci_nc = ci_nc
    ci_res.ci_f = np.sqrt(ci_res.f2)
    ci_res.ci_eta = np.sqrt(ci_res.eta2)
    ci_res.ci_f_corrected = np.sqrt(ci_res.f2 * (df1 + 1) / df1)

    return ci_res


def anova_generic(means, variances, nobs, use_var="unequal",
                  welch_correction=True, info=None):
    """
    Oneway Anova based on summary statistics

    Parameters
    ----------
    means : array_like
        Mean of samples to be compared
    variances : float or array_like
        Residual (within) variance of each sample or pooled.
        If ``variances`` is scalar, then it is interpreted as pooled variance
        that is the same for all samples, ``use_var`` will be ignored.
        Otherwise, the variances are used depending on the ``use_var`` keyword.
    nobs : int or array_like
        Number of observations for the samples.
        If nobs is scalar, then it is assumed that all samples have the same
        number ``nobs`` of observation, i.e. a balanced sample case.
        Otherwise, statistics will be weighted corresponding to nobs.
        Only relative sizes are relevant, any proportional change to nobs does
        not change the effect size.
    use_var : {"unequal", "equal", "bf"}
        If ``use_var`` is "unequal", then the variances can differ across
        samples and the effect size for Welch anova will be computed.
    welch_correction : bool
        If this is false, then the Welch correction to the test statistic is
        not included. This allows the computation of an effect size measure
        that corresponds more closely to Cohen's f.
    info : not used yet

    Returns
    -------
    res : results instance
        This includes `statistic` and `pvalue`.

    """
    options = {"use_var": use_var,
               "welch_correction": welch_correction
               }
    if means.ndim != 1:
        raise ValueError('data (means, ...) has to be one-dimensional')
    nobs_t = nobs.sum()
    n_groups = len(means)
    # mean_t = (nobs * means).sum() / nobs_t
    if use_var == "unequal":
        weights = nobs / variances
    else:
        weights = nobs

    w_total = weights.sum()
    w_rel = weights / w_total
    # meanw_t = (weights * means).sum() / w_total
    meanw_t = w_rel @ means

    statistic = np.dot(weights, (means - meanw_t)**2) / (n_groups - 1.)
    df_num = n_groups - 1.

    if use_var == "unequal":
        tmp = ((1 - w_rel)**2 / (nobs - 1)).sum() / (n_groups**2 - 1)
        if welch_correction:
            statistic /= 1 + 2 * (n_groups - 2) * tmp
        df_denom = 1. / (3. * tmp)

    elif use_var == "equal":
        # variance of group demeaned total sample, pooled var_resid
        tmp = ((nobs - 1) * variances).sum() / (nobs_t - n_groups)
        statistic /= tmp
        df_denom = nobs_t - n_groups

    elif use_var == "bf":
        tmp = ((1. - nobs / nobs_t) * variances).sum()
        statistic = 1. * (nobs * (means - meanw_t)**2).sum()
        statistic /= tmp

        df_num2 = n_groups - 1
        df_denom = tmp**2 / ((1. - nobs / nobs_t) ** 2 *
                             variances ** 2 / (nobs - 1)).sum()
        df_num = tmp**2 / ((variances ** 2).sum() +
                           (nobs / nobs_t * variances).sum() ** 2 -
                           2 * (nobs / nobs_t * variances ** 2).sum())
        pval2 = stats.f.sf(statistic, df_num2, df_denom)
        options["df2"] = (df_num2, df_denom)
        options["df_num2"] = df_num2
        options["pvalue2"] = pval2

    else:
        raise ValueError('use_var is to be one of "unequal", "equal" or "bf"')

    pval = stats.f.sf(statistic, df_num, df_denom)
    res = HolderTuple(statistic=statistic,
                      pvalue=pval,
                      df=(df_num, df_denom),
                      df_num=df_num,
                      df_denom=df_denom,
                      nobs_t=nobs_t,
                      n_groups=n_groups,
                      means=means,
                      nobs=nobs,
                      vars_=variances,
                      **options
                      )
    return res


def anova_oneway(data, groups=None, use_var="unequal", welch_correction=True,
                 trim_frac=0):
    """Oneway Anova

    This implements standard anova, Welch and Brown-Forsythe, and trimmed
    (Yuen) variants of those.

    Parameters
    ----------
    data : tuple of array_like or DataFrame or Series
        Data for k independent samples, with k >= 2.
        The data can be provided as a tuple or list of arrays or in long
        format with outcome observations in ``data`` and group membership in
        ``groups``.
    groups : ndarray or Series
        If data is in long format, then groups is needed as indicator to which
        group or sample and observations belongs.
    use_var : {"unequal", "equal" or "bf"}
        `use_var` specified how to treat heteroscedasticity, unequal variance,
        across samples. Three approaches are available

        "unequal" : Variances are not assumed to be equal across samples.
            Heteroscedasticity is taken into account with Welch Anova and
            Satterthwaite-Welch degrees of freedom.
            This is the default.
        "equal" : Variances are assumed to be equal across samples.
            This is the standard Anova.
        "bf" : Variances are not assumed to be equal across samples.
            The method is Browne-Forsythe (1971) for testing equality of means
            with the corrected degrees of freedom by Merothra. The original BF
            degrees of freedom are available as additional attributes in the
            results instance, ``df_denom2`` and ``p_value2``.

    welch_correction : bool
        If this is false, then the Welch correction to the test statistic is
        not included. This allows the computation of an effect size measure
        that corresponds more closely to Cohen's f.
    trim_frac : float in [0, 0.5)
        Optional trimming for Anova with trimmed mean and winsorized variances.
        With the default trim_frac equal to zero, the oneway Anova statistics
        are computed without trimming. If `trim_frac` is larger than zero,
        then the largest and smallest observations in each sample are trimmed.
        The number of trimmed observations is the fraction of number of
        observations in the sample truncated to the next lower integer.
        `trim_frac` has to be smaller than 0.5, however, if the fraction is
        so large that there are not enough observations left over, then `nan`
        will be returned.

    Returns
    -------
    res : results instance
        The returned HolderTuple instance has the following main attributes
        and some additional information in other attributes.

        statistic : float
            Test statistic for k-sample mean comparison which is approximately
            F-distributed.
        pvalue : float
            If ``use_var="bf"``, then the p-value is based on corrected
            degrees of freedom following Mehrotra 1997.
        pvalue2 : float
            This is the p-value based on degrees of freedom as in
            Brown-Forsythe 1974 and is only available if ``use_var="bf"``.
        df = (df_denom, df_num) : tuple of floats
            Degreeds of freedom for the F-distribution depend on ``use_var``.
            If ``use_var="bf"``, then `df_denom` is for Mehrotra p-values
            `df_denom2` is available for Brown-Forsythe 1974 p-values.
            `df_num` is the same numerator degrees of freedom for both
            p-values.

    Notes
    -----
    Welch's anova is correctly sized (not liberal or conservative) in smaller
    samples if the distribution of the samples is not very far away from the
    normal distribution. The test can become liberal if the data is strongly
    skewed. Welch's Anova can also be correctly sized for discrete
    distributions with finite support, like Lickert scale data.
    The trimmed version is robust to many non-normal distributions, it stays
    correctly sized in many cases, and is more powerful in some cases with
    skewness or heavy tails.

    Trimming is currently based on the integer part of ``nobs * trim_frac``.
    The default might change to including fractional observations as in the
    original articles by Yuen.


    See Also
    --------
    anova_generic

    References
    ----------
    Brown, Morton B., and Alan B. Forsythe. 1974. “The Small Sample Behavior
    of Some Statistics Which Test the Equality of Several Means.”
    Technometrics 16 (1) (February 1): 129–132. doi:10.2307/1267501.

    Mehrotra, Devan V. 1997. “Improving the Brown-Forsythe Solution to the
    Generalized Behrens-Fisher Problem.” Communications in Statistics -
    Simulation and Computation 26 (3): 1139–1145.
    doi:10.1080/03610919708813431.
    """
    if groups is not None:
        uniques = np.unique(groups)
        data = [data[groups == uni] for uni in uniques]
    else:
        # uniques = None  # not used yet, add to info?
        pass
    args = list(map(np.asarray, data))
    if any([x.ndim != 1 for x in args]):
        raise ValueError('data arrays have to be one-dimensional')

    nobs = np.array([len(x) for x in args], float)
    # n_groups = len(args)  # not used
    # means = np.array([np.mean(x, axis=0) for x in args], float)
    # vars_ = np.array([np.var(x, ddof=1, axis=0) for x in args], float)

    if trim_frac == 0:
        means = np.array([x.mean() for x in args])
        vars_ = np.array([x.var(ddof=1) for x in args])
    else:
        tms = [TrimmedMean(x, trim_frac) for x in args]
        means = np.array([tm.mean_trimmed for tm in tms])
        # R doesn't use uncorrected var_winsorized
        # vars_ = np.array([tm.var_winsorized for tm in tms])
        vars_ = np.array([tm.var_winsorized * (tm.nobs - 1) /
                          (tm.nobs_reduced - 1) for tm in tms])
        # nobs_original = nobs  # store just in case
        nobs = np.array([tm.nobs_reduced for tm in tms])

    res = anova_generic(means, vars_, nobs, use_var=use_var,
                        welch_correction=welch_correction)

    return res


def equivalence_oneway_generic(f_stat, n_groups, nobs, equiv_margin, df,
                               alpha=0.05, margin_type="f2"):
    """Equivalence test for oneway anova (Wellek and extensions)

    This is an helper function when summary statistics are available.
    Use `equivalence_oneway` instead.

    The null hypothesis is that the means differ by more than `equiv_margin`
    in the anova distance measure.
    If the Null is rejected, then the data supports that means are equivalent,
    i.e. within a given distance.

    Parameters
    ----------
    f_stat : float
        F-statistic
    n_groups : int
        Number of groups in oneway comparison.
    nobs : ndarray
        Array of number of observations in groups.
    equiv_margin : float
        Equivalence margin in terms of effect size. Effect size can be chosen
        with `margin_type`. default is squared Cohen's f.
    df : tuple
        degrees of freedom ``df = (df1, df2)`` where

        - df1 : numerator degrees of freedom, number of constraints
        - df2 : denominator degrees of freedom, df_resid

    alpha : float in (0, 1)
        Significance level for the hypothesis test.
    margin_type : "f2" or "wellek"
        Type of effect size used for equivalence margin.

    Returns
    -------
    results : instance of HolderTuple class
        The two main attributes are test statistic `statistic` and p-value
        `pvalue`.

    Notes
    -----
    Equivalence in this function is defined in terms of a squared distance
    measure similar to Mahalanobis distance.
    Alternative definitions for the oneway case are based on maximum difference
    between pairs of means or similar pairwise distances.

    The equivalence margin is used for the noncentrality parameter in the
    noncentral F distribution for the test statistic. In samples with unequal
    variances estimated using Welch or Brown-Forsythe Anova, the f-statistic
    depends on the unequal variances and corrections to the test statistic.
    This means that the equivalence margins are not fully comparable across
    methods for treating unequal variances.

    References
    ----------
    Wellek, Stefan. 2010. Testing Statistical Hypotheses of Equivalence and
    Noninferiority. 2nd ed. Boca Raton: CRC Press.

    Cribbie, Robert A., Chantal A. Arpin-Cribbie, and Jamie A. Gruman. 2009.
    “Tests of Equivalence for One-Way Independent Groups Designs.” The Journal
    of Experimental Education 78 (1): 1–13.
    https://doi.org/10.1080/00220970903224552.

    Jan, Show-Li, and Gwowen Shieh. 2019. “On the Extended Welch Test for
    Assessing Equivalence of Standardized Means.” Statistics in
    Biopharmaceutical Research 0 (0): 1–8.
    https://doi.org/10.1080/19466315.2019.1654915.

    """
    nobs_t = nobs.sum()
    nobs_mean = nobs_t / n_groups

    if margin_type == "wellek":
        nc_null = nobs_mean * equiv_margin**2
        es = f_stat * (n_groups - 1) / nobs_mean
        type_effectsize = "Wellek's psi_squared"
    elif margin_type in ["f2", "fsqu", "fsquared"]:
        nc_null = nobs_t * equiv_margin
        es = f_stat / nobs_t
        type_effectsize = "Cohen's f_squared"
    else:
        raise ValueError('`margin_type` should be "f2" or "wellek"')
    crit_f = ncf_ppf(alpha, df[0], df[1], nc_null)

    if margin_type == "wellek":
        # TODO: do we need a sqrt
        crit_es = crit_f * (n_groups - 1) / nobs_mean
    elif margin_type in ["f2", "fsqu", "fsquared"]:
        crit_es = crit_f / nobs_t

    reject = (es < crit_es)

    pv = ncf_cdf(f_stat, df[0], df[1], nc_null)
    pwr = ncf_cdf(crit_f, df[0], df[1], 1e-13)  # scipy, cannot be 0
    res = HolderTuple(statistic=f_stat,
                      pvalue=pv,
                      effectsize=es,  # match es type to margin_type
                      crit_f=crit_f,
                      crit_es=crit_es,
                      reject=reject,
                      power_zero=pwr,
                      df=df,
                      f_stat=f_stat,
                      type_effectsize=type_effectsize
                      )
    return res


def equivalence_oneway(data, equiv_margin, groups=None, use_var="unequal",
                       welch_correction=True, trim_frac=0, margin_type="f2"):
    """equivalence test for oneway anova (Wellek's Anova)

    The null hypothesis is that the means differ by more than `equiv_margin`
    in the anova distance measure.
    If the Null is rejected, then the data supports that means are equivalent,
    i.e. within a given distance.

    Parameters
    ----------
    data : tuple of array_like or DataFrame or Series
        Data for k independent samples, with k >= 2.
        The data can be provided as a tuple or list of arrays or in long
        format with outcome observations in ``data`` and group membership in
        ``groups``.
    equiv_margin : float
        Equivalence margin in terms of effect size. Effect size can be chosen
        with `margin_type`. default is squared Cohen's f.
    groups : ndarray or Series
        If data is in long format, then groups is needed as indicator to which
        group or sample and observations belongs.
    use_var : {"unequal", "equal" or "bf"}
        `use_var` specified how to treat heteroscedasticity, unequal variance,
        across samples. Three approaches are available

        "unequal" : Variances are not assumed to be equal across samples.
            Heteroscedasticity is taken into account with Welch Anova and
            Satterthwaite-Welch degrees of freedom.
            This is the default.
        "equal" : Variances are assumed to be equal across samples.
            This is the standard Anova.
        "bf: Variances are not assumed to be equal across samples.
            The method is Browne-Forsythe (1971) for testing equality of means
            with the corrected degrees of freedom by Merothra. The original BF
            degrees of freedom are available as additional attributes in the
            results instance, ``df_denom2`` and ``p_value2``.

    welch_correction : bool
        If this is false, then the Welch correction to the test statistic is
        not included. This allows the computation of an effect size measure
        that corresponds more closely to Cohen's f.
    trim_frac : float in [0, 0.5)
        Optional trimming for Anova with trimmed mean and winsorized variances.
        With the default trim_frac equal to zero, the oneway Anova statistics
        are computed without trimming. If `trim_frac` is larger than zero,
        then the largest and smallest observations in each sample are trimmed.
        The number of trimmed observations is the fraction of number of
        observations in the sample truncated to the next lower integer.
        `trim_frac` has to be smaller than 0.5, however, if the fraction is
        so large that there are not enough observations left over, then `nan`
        will be returned.
    margin_type : "f2" or "wellek"
        Type of effect size used for equivalence margin, either squared
        Cohen's f or Wellek's psi. Default is "f2".

    Returns
    -------
    results : instance of HolderTuple class
        The two main attributes are test statistic `statistic` and p-value
        `pvalue`.

    See Also
    --------
    anova_oneway
    equivalence_scale_oneway
    """

    # use anova to compute summary statistics and f-statistic
    res0 = anova_oneway(data, groups=groups, use_var=use_var,
                        welch_correction=welch_correction,
                        trim_frac=trim_frac)
    f_stat = res0.statistic
    res = equivalence_oneway_generic(f_stat, res0.n_groups, res0.nobs_t,
                                     equiv_margin, res0.df, alpha=0.05,
                                     margin_type=margin_type)

    return res


def _power_equivalence_oneway_emp(f_stat, n_groups, nobs, eps, df, alpha=0.05):
    """Empirical power of oneway equivalence test

    This only returns post-hoc, empirical power.

    Warning: eps is currently effect size margin as defined as in Wellek, and
    not the signal to noise ratio (Cohen's f family).

    Parameters
    ----------
    f_stat : float
        F-statistic from oneway anova, used to compute empirical effect size
    n_groups : int
        Number of groups in oneway comparison.
    nobs : ndarray
        Array of number of observations in groups.
    eps : float
        Equivalence margin in terms of effect size given by Wellek's psi.
    df : tuple
        Degrees of freedom for F distribution.
    alpha : float in (0, 1)
        Significance level for the hypothesis test.

    Returns
    -------
    pow : float
        Ex-post, post-hoc or empirical power at f-statistic of the equivalence
        test.
    """

    res = equivalence_oneway_generic(f_stat, n_groups, nobs, eps, df,
                                     alpha=alpha, margin_type="wellek")

    nobs_mean = nobs.sum() / n_groups
    fn = f_stat  # post-hoc power, empirical power at estimate
    esn = fn * (n_groups - 1) / nobs_mean  # Wellek psi
    pow_ = ncf_cdf(res.crit_f, df[0], df[1], nobs_mean * esn)

    return pow_


def power_equivalence_oneway(f2_alt, equiv_margin, nobs_t, n_groups=None,
                             df=None, alpha=0.05, margin_type="f2"):
    """
    Power of  oneway equivalence test

    Parameters
    ----------
    f2_alt : float
        Effect size, squared Cohen's f, under the alternative.
    equiv_margin : float
        Equivalence margin in terms of effect size. Effect size can be chosen
        with `margin_type`. default is squared Cohen's f.
    nobs_t : ndarray
        Total number of observations summed over all groups.
    n_groups : int
        Number of groups in oneway comparison. If margin_type is "wellek",
        then either ``n_groups`` or ``df`` has to be given.
    df : tuple
        Degrees of freedom for F distribution,
        ``df = (n_groups - 1, nobs_t - n_groups)``
    alpha : float in (0, 1)
        Significance level for the hypothesis test.
    margin_type : "f2" or "wellek"
        Type of effect size used for equivalence margin, either squared
        Cohen's f or Wellek's psi. Default is "f2".

    Returns
    -------
    pow_alt : float
        Power of the equivalence test at given equivalence effect size under
        the alternative.
    """

    # one of n_groups or df has to be specified
    if df is None:
        if n_groups is None:
            raise ValueError("either df or n_groups has to be provided")
        df = (n_groups - 1, nobs_t - n_groups)

    # esn = fn * (n_groups - 1) / nobs_mean  # Wellek psi

    # fix for scipy, ncf does not allow nc == 0, fixed in scipy master
    if f2_alt == 0:
        f2_alt = 1e-13
    # effect size, critical value at margin
    # f2_null = equiv_margin
    if margin_type in ["f2", "fsqu", "fsquared"]:
        f2_null = equiv_margin
    elif margin_type == "wellek":
        if n_groups is None:
            raise ValueError("If margin_type is wellek, then n_groups has "
                             "to be provided")
        #  f2_null = (n_groups - 1) * n_groups / nobs_t * equiv_margin**2
        nobs_mean = nobs_t / n_groups
        f2_null = nobs_mean * equiv_margin**2 / nobs_t
        f2_alt = nobs_mean * f2_alt**2 / nobs_t
    else:
        raise ValueError('`margin_type` should be "f2" or "wellek"')

    crit_f_margin = ncf_ppf(alpha, df[0], df[1], nobs_t * f2_null)
    pwr_alt = ncf_cdf(crit_f_margin, df[0], df[1], nobs_t * f2_alt)
    return pwr_alt


def simulate_power_equivalence_oneway(means, nobs, equiv_margin, vars_=None,
                                      k_mc=1000, trim_frac=0,
                                      options_var=None, margin_type="f2"
                                      ):  # , anova_options=None):  #TODO
    """Simulate Power for oneway equivalence test (Wellek's Anova)

    This function is experimental and written to evaluate asymptotic power
    function. This function will change without backwards compatibility
    constraints. The only part that is stable is `pvalue` attribute in results.

    Effect size for equivalence margin

    """
    if options_var is None:
        options_var = ["unequal", "equal", "bf"]
    if vars_ is not None:
        stds = np.sqrt(vars_)
    else:
        stds = np.ones(len(means))

    nobs_mean = nobs.mean()
    n_groups = len(nobs)
    res_mc = []
    f_mc = []
    reject_mc = []
    other_mc = []
    for _ in range(k_mc):
        y0, y1, y2, y3 = [m + std * np.random.randn(n)
                          for (n, m, std) in zip(nobs, means, stds)]

        res_i = []
        f_i = []
        reject_i = []
        other_i = []
        for uv in options_var:
            # for welch in options_welch:
            # res1 = sma.anova_generic(means, vars_, nobs, use_var=uv,
            #                          welch_correction=welch)
            res0 = anova_oneway([y0, y1, y2, y3], use_var=uv,
                                trim_frac=trim_frac)
            f_stat = res0.statistic
            res1 = equivalence_oneway_generic(f_stat, n_groups, nobs.sum(),
                                              equiv_margin, res0.df,
                                              alpha=0.05,
                                              margin_type=margin_type)
            res_i.append(res1.pvalue)
            es_wellek = f_stat * (n_groups - 1) / nobs_mean
            f_i.append(es_wellek)
            reject_i.append(res1.reject)
            other_i.extend([res1.crit_f, res1.crit_es, res1.power_zero])
        res_mc.append(res_i)
        f_mc.append(f_i)
        reject_mc.append(reject_i)
        other_mc.append(other_i)

    f_mc = np.asarray(f_mc)
    other_mc = np.asarray(other_mc)
    res_mc = np.asarray(res_mc)
    reject_mc = np.asarray(reject_mc)
    res = Holder(f_stat=f_mc,
                 other=other_mc,
                 pvalue=res_mc,
                 reject=reject_mc
                 )
    return res


def test_scale_oneway(data, method="bf", center="median", transform="abs",
                      trim_frac_mean=0.1, trim_frac_anova=0.0):
    """Oneway Anova test for equal scale, variance or dispersion

    This hypothesis test performs a oneway anova test on transformed data and
    includes Levene and Brown-Forsythe tests for equal variances as special
    cases.

    Parameters
    ----------
    data : tuple of array_like or DataFrame or Series
        Data for k independent samples, with k >= 2. The data can be provided
        as a tuple or list of arrays or in long format with outcome
        observations in ``data`` and group membership in ``groups``.
    method : {"unequal", "equal" or "bf"}
        How to treat heteroscedasticity across samples. This is used as
        `use_var` option in `anova_oneway` and refers to the variance of the
        transformed data, i.e. assumption is on 4th moment if squares are used
        as transform.
        Three approaches are available:

        "unequal" : Variances are not assumed to be equal across samples.
            Heteroscedasticity is taken into account with Welch Anova and
            Satterthwaite-Welch degrees of freedom.
            This is the default.
        "equal" : Variances are assumed to be equal across samples.
            This is the standard Anova.
        "bf" : Variances are not assumed to be equal across samples.
            The method is Browne-Forsythe (1971) for testing equality of means
            with the corrected degrees of freedom by Merothra. The original BF
            degrees of freedom are available as additional attributes in the
            results instance, ``df_denom2`` and ``p_value2``.

    center : "median", "mean", "trimmed" or float
        Statistic used for centering observations. If a float, then this
        value is used to center. Default is median.
    transform : "abs", "square" or callable
        Transformation for the centered observations. If a callable, then this
        function is called on the centered data.
        Default is absolute value.
    trim_frac_mean=0.1 : float in [0, 0.5)
        Trim fraction for the trimmed mean when `center` is "trimmed"
    trim_frac_anova : float in [0, 0.5)
        Optional trimming for Anova with trimmed mean and Winsorized variances.
        With the default trim_frac equal to zero, the oneway Anova statistics
        are computed without trimming. If `trim_frac` is larger than zero,
        then the largest and smallest observations in each sample are trimmed.
        see ``trim_frac`` option in `anova_oneway`

    Returns
    -------
    res : results instance
        The returned HolderTuple instance has the following main attributes
        and some additional information in other attributes.

        statistic : float
            Test statistic for k-sample mean comparison which is approximately
            F-distributed.
        pvalue : float
            If ``method="bf"``, then the p-value is based on corrected
            degrees of freedom following Mehrotra 1997.
        pvalue2 : float
            This is the p-value based on degrees of freedom as in
            Brown-Forsythe 1974 and is only available if ``method="bf"``.
        df : (df_denom, df_num)
            Tuple containing degrees of freedom for the F-distribution depend
            on ``method``. If ``method="bf"``, then `df_denom` is for Mehrotra
            p-values `df_denom2` is available for Brown-Forsythe 1974 p-values.
            `df_num` is the same numerator degrees of freedom for both
            p-values.

    See Also
    --------
    anova_oneway
    scale_transform
    """

    data = map(np.asarray, data)
    xxd = [scale_transform(x, center=center, transform=transform,
                           trim_frac=trim_frac_mean) for x in data]

    res = anova_oneway(xxd, groups=None, use_var=method,
                       welch_correction=True, trim_frac=trim_frac_anova)
    res.data_transformed = xxd
    return res


def equivalence_scale_oneway(data, equiv_margin, method='bf', center='median',
                             transform='abs', trim_frac_mean=0.,
                             trim_frac_anova=0.):
    """Oneway Anova test for equivalence of scale, variance or dispersion

    This hypothesis test performs a oneway equivalence anova test on
    transformed data.

    Note, the interpretation of the equivalence margin `equiv_margin` will
    depend on the transformation of the data. Transformations like
    absolute deviation are not scaled to correspond to the variance under
    normal distribution.

    Parameters
    ----------
    data : tuple of array_like or DataFrame or Series
        Data for k independent samples, with k >= 2. The data can be provided
        as a tuple or list of arrays or in long format with outcome
        observations in ``data`` and group membership in ``groups``.
    equiv_margin : float
        Equivalence margin in terms of effect size. Effect size can be chosen
        with `margin_type`. default is squared Cohen's f.
    method : {"unequal", "equal" or "bf"}
        How to treat heteroscedasticity across samples. This is used as
        `use_var` option in `anova_oneway` and refers to the variance of the
        transformed data, i.e. assumption is on 4th moment if squares are used
        as transform.
        Three approaches are available:

        "unequal" : Variances are not assumed to be equal across samples.
            Heteroscedasticity is taken into account with Welch Anova and
            Satterthwaite-Welch degrees of freedom.
            This is the default.
        "equal" : Variances are assumed to be equal across samples.
            This is the standard Anova.
        "bf" : Variances are not assumed to be equal across samples.
            The method is Browne-Forsythe (1971) for testing equality of means
            with the corrected degrees of freedom by Merothra. The original BF
            degrees of freedom are available as additional attributes in the
            results instance, ``df_denom2`` and ``p_value2``.
    center : "median", "mean", "trimmed" or float
        Statistic used for centering observations. If a float, then this
        value is used to center. Default is median.
    transform : "abs", "square" or callable
        Transformation for the centered observations. If a callable, then this
        function is called on the centered data.
        Default is absolute value.
    trim_frac_mean : float in [0, 0.5)
        Trim fraction for the trimmed mean when `center` is "trimmed"
    trim_frac_anova : float in [0, 0.5)
        Optional trimming for Anova with trimmed mean and Winsorized variances.
        With the default trim_frac equal to zero, the oneway Anova statistics
        are computed without trimming. If `trim_frac` is larger than zero,
        then the largest and smallest observations in each sample are trimmed.
        see ``trim_frac`` option in `anova_oneway`

    Returns
    -------
    results : instance of HolderTuple class
        The two main attributes are test statistic `statistic` and p-value
        `pvalue`.

    See Also
    --------
    anova_oneway
    scale_transform
    equivalence_oneway
    """
    data = map(np.asarray, data)
    xxd = [scale_transform(x, center=center, transform=transform,
                           trim_frac=trim_frac_mean) for x in data]

    res = equivalence_oneway(xxd, equiv_margin, use_var=method,
                             welch_correction=True, trim_frac=trim_frac_anova)
    res.x_transformed = xxd
    return res
