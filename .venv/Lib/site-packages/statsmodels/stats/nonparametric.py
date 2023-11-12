# -*- coding: utf-8 -*-
"""
Rank based methods for inferential statistics

Created on Sat Aug 15 10:18:53 2020

Author: Josef Perktold
License: BSD-3

"""


import numpy as np
from scipy import stats
from scipy.stats import rankdata

from statsmodels.stats.base import HolderTuple
from statsmodels.stats.weightstats import (
    _tconfint_generic,
    _tstat_generic,
    _zconfint_generic,
    _zstat_generic,
)


def rankdata_2samp(x1, x2):
    """Compute midranks for two samples

    Parameters
    ----------
    x1, x2 : array_like
        Original data for two samples that will be converted to midranks.

    Returns
    -------
    rank1 : ndarray
        Midranks of the first sample in the pooled sample.
    rank2 : ndarray
        Midranks of the second sample in the pooled sample.
    ranki1 : ndarray
        Internal midranks of the first sample.
    ranki2 : ndarray
        Internal midranks of the second sample.

    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)

    nobs1 = len(x1)
    nobs2 = len(x2)
    if nobs1 == 0 or nobs2 == 0:
        raise ValueError("one sample has zero length")

    x_combined = np.concatenate((x1, x2))
    if x_combined.ndim > 1:
        rank = np.apply_along_axis(rankdata, 0, x_combined)
    else:
        rank = rankdata(x_combined)  # no axis in older scipy
    rank1 = rank[:nobs1]
    rank2 = rank[nobs1:]
    if x_combined.ndim > 1:
        ranki1 = np.apply_along_axis(rankdata, 0, x1)
        ranki2 = np.apply_along_axis(rankdata, 0, x2)
    else:
        ranki1 = rankdata(x1)
        ranki2 = rankdata(x2)
    return rank1, rank2, ranki1, ranki2


class RankCompareResult(HolderTuple):
    """Results for rank comparison

    This is a subclass of HolderTuple that includes results from intermediate
    computations, as well as methods for hypothesis tests, confidence intervals
    and summary.
    """

    def conf_int(self, value=None, alpha=0.05, alternative="two-sided"):
        """
        Confidence interval for probability that sample 1 has larger values

        Confidence interval is for the shifted probability

            P(x1 > x2) + 0.5 * P(x1 = x2) - value

        Parameters
        ----------
        value : float
            Value, default 0, shifts the confidence interval,
            e.g. ``value=0.5`` centers the confidence interval at zero.
        alpha : float
            Significance level for the confidence interval, coverage is
            ``1-alpha``
        alternative : str
            The alternative hypothesis, H1, has to be one of the following

               * 'two-sided' : H1: ``prob - value`` not equal to 0.
               * 'larger' :   H1: ``prob - value > 0``
               * 'smaller' :  H1: ``prob - value < 0``

        Returns
        -------
        lower : float or ndarray
            Lower confidence limit. This is -inf for the one-sided alternative
            "smaller".
        upper : float or ndarray
            Upper confidence limit. This is inf for the one-sided alternative
            "larger".

        """

        p0 = value
        if p0 is None:
            p0 = 0
        diff = self.prob1 - p0
        std_diff = np.sqrt(self.var / self.nobs)

        if self.use_t is False:
            return _zconfint_generic(diff, std_diff, alpha, alternative)
        else:
            return _tconfint_generic(diff, std_diff, self.df, alpha,
                                     alternative)

    def test_prob_superior(self, value=0.5, alternative="two-sided"):
        """test for superiority probability

        H0: P(x1 > x2) + 0.5 * P(x1 = x2) = value

        The alternative is that the probability is either not equal, larger
        or smaller than the null-value depending on the chosen alternative.

        Parameters
        ----------
        value : float
            Value of the probability under the Null hypothesis.
        alternative : str
            The alternative hypothesis, H1, has to be one of the following

               * 'two-sided' : H1: ``prob - value`` not equal to 0.
               * 'larger' :   H1: ``prob - value > 0``
               * 'smaller' :  H1: ``prob - value < 0``

        Returns
        -------
        res : HolderTuple
            HolderTuple instance with the following main attributes

            statistic : float
                Test statistic for z- or t-test
            pvalue : float
                Pvalue of the test based on either normal or t distribution.

        """

        p0 = value  # alias
        # diff = self.prob1 - p0  # for reporting, not used in computation
        # TODO: use var_prob
        std_diff = np.sqrt(self.var / self.nobs)

        # corresponds to a one-sample test and either p0 or diff could be used
        if not self.use_t:
            stat, pv = _zstat_generic(self.prob1, p0, std_diff, alternative,
                                      diff=0)
            distr = "normal"
        else:
            stat, pv = _tstat_generic(self.prob1, p0, std_diff, self.df,
                                      alternative, diff=0)
            distr = "t"

        res = HolderTuple(statistic=stat,
                          pvalue=pv,
                          df=self.df,
                          distribution=distr
                          )
        return res

    def tost_prob_superior(self, low, upp):
        '''test of stochastic (non-)equivalence of p = P(x1 > x2)

        Null hypothesis:  p < low or p > upp
        Alternative hypothesis:  low < p < upp

        where p is the probability that a random draw from the population of
        the first sample has a larger value than a random draw from the
        population of the second sample, specifically

            p = P(x1 > x2) + 0.5 * P(x1 = x2)

        If the pvalue is smaller than a threshold, say 0.05, then we reject the
        hypothesis that the probability p that distribution 1 is stochastically
        superior to distribution 2 is outside of the interval given by
        thresholds low and upp.

        Parameters
        ----------
        low, upp : float
            equivalence interval low < mean < upp

        Returns
        -------
        res : HolderTuple
            HolderTuple instance with the following main attributes

            pvalue : float
                Pvalue of the equivalence test given by the larger pvalue of
                the two one-sided tests.
            statistic : float
                Test statistic of the one-sided test that has the larger
                pvalue.
            results_larger : HolderTuple
                Results instanc with test statistic, pvalue and degrees of
                freedom for lower threshold test.
            results_smaller : HolderTuple
                Results instanc with test statistic, pvalue and degrees of
                freedom for upper threshold test.

        '''

        t1 = self.test_prob_superior(low, alternative='larger')
        t2 = self.test_prob_superior(upp, alternative='smaller')

        # idx_max = 1 if t1.pvalue < t2.pvalue else 0
        idx_max = np.asarray(t1.pvalue < t2.pvalue, int)
        title = "Equivalence test for Prob(x1 > x2) + 0.5 Prob(x1 = x2) "
        res = HolderTuple(statistic=np.choose(idx_max,
                                              [t1.statistic, t2.statistic]),
                          # pvalue=[t1.pvalue, t2.pvalue][idx_max], # python
                          # use np.choose for vectorized selection
                          pvalue=np.choose(idx_max, [t1.pvalue, t2.pvalue]),
                          results_larger=t1,
                          results_smaller=t2,
                          title=title
                          )
        return res

    def confint_lintransf(self, const=-1, slope=2, alpha=0.05,
                          alternative="two-sided"):
        """confidence interval of a linear transformation of prob1

        This computes the confidence interval for

            d = const + slope * prob1

        Default values correspond to Somers' d.

        Parameters
        ----------
        const, slope : float
            Constant and slope for linear (affine) transformation.
        alpha : float
            Significance level for the confidence interval, coverage is
            ``1-alpha``
        alternative : str
            The alternative hypothesis, H1, has to be one of the following

               * 'two-sided' : H1: ``prob - value`` not equal to 0.
               * 'larger' :   H1: ``prob - value > 0``
               * 'smaller' :  H1: ``prob - value < 0``

        Returns
        -------
        lower : float or ndarray
            Lower confidence limit. This is -inf for the one-sided alternative
            "smaller".
        upper : float or ndarray
            Upper confidence limit. This is inf for the one-sided alternative
            "larger".

        """

        low_p, upp_p = self.conf_int(alpha=alpha, alternative=alternative)
        low = const + slope * low_p
        upp = const + slope * upp_p
        if slope < 0:
            low, upp = upp, low
        return low, upp

    def effectsize_normal(self, prob=None):
        """
        Cohen's d, standardized mean difference under normality assumption.

        This computes the standardized mean difference, Cohen's d, effect size
        that is equivalent to the rank based probability ``p`` of being
        stochastically larger if we assume that the data is normally
        distributed, given by

            :math: `d = F^{-1}(p) * \\sqrt{2}`

        where :math:`F^{-1}` is the inverse of the cdf of the normal
        distribution.

        Parameters
        ----------
        prob : float in (0, 1)
            Probability to be converted to Cohen's d effect size.
            If prob is None, then the ``prob1`` attribute is used.

        Returns
        -------
        equivalent Cohen's d effect size under normality assumption.

        """
        if prob is None:
            prob = self.prob1
        return stats.norm.ppf(prob) * np.sqrt(2)

    def summary(self, alpha=0.05, xname=None):
        """summary table for probability that random draw x1 is larger than x2

        Parameters
        ----------
        alpha : float
            Significance level for confidence intervals. Coverage is 1 - alpha
        xname : None or list of str
            If None, then each row has a name column with generic names.
            If xname is a list of strings, then it will be included as part
            of those names.

        Returns
        -------
        SimpleTable instance with methods to convert to different output
        formats.
        """

        yname = "None"
        effect = np.atleast_1d(self.prob1)
        if self.pvalue is None:
            statistic, pvalue = self.test_prob_superior()
        else:
            pvalue = self.pvalue
            statistic = self.statistic
        pvalues = np.atleast_1d(pvalue)
        ci = np.atleast_2d(self.conf_int(alpha=alpha))
        if ci.shape[0] > 1:
            ci = ci.T
        use_t = self.use_t
        sd = np.atleast_1d(np.sqrt(self.var_prob))
        statistic = np.atleast_1d(statistic)
        if xname is None:
            xname = ['c%d' % ii for ii in range(len(effect))]

        xname2 = ['prob(x1>x2) %s' % ii for ii in xname]

        title = "Probability sample 1 is stochastically larger"
        from statsmodels.iolib.summary import summary_params

        summ = summary_params((self, effect, sd, statistic,
                               pvalues, ci),
                              yname=yname, xname=xname2, use_t=use_t,
                              title=title, alpha=alpha)
        return summ


def rank_compare_2indep(x1, x2, use_t=True):
    """
    Statistics and tests for the probability that x1 has larger values than x2.

    p is the probability that a random draw from the population of
    the first sample has a larger value than a random draw from the
    population of the second sample, specifically

            p = P(x1 > x2) + 0.5 * P(x1 = x2)

    This is a measure underlying Wilcoxon-Mann-Whitney's U test,
    Fligner-Policello test and Brunner-Munzel test, and
    Inference is based on the asymptotic distribution of the Brunner-Munzel
    test. The half probability for ties corresponds to the use of midranks
    and make it valid for discrete variables.

    The Null hypothesis for stochastic equality is p = 0.5, which corresponds
    to the Brunner-Munzel test.

    Parameters
    ----------
    x1, x2 : array_like
        Array of samples, should be one-dimensional.
    use_t : boolean
        If use_t is true, the t distribution with Welch-Satterthwaite type
        degrees of freedom is used for p-value and confidence interval.
        If use_t is false, then the normal distribution is used.

    Returns
    -------
    res : RankCompareResult
        The results instance contains the results for the Brunner-Munzel test
        and has methods for hypothesis tests, confidence intervals and summary.

        statistic : float
            The Brunner-Munzel W statistic.
        pvalue : float
            p-value assuming an t distribution. One-sided or
            two-sided, depending on the choice of `alternative` and `use_t`.

    See Also
    --------
    RankCompareResult
    scipy.stats.brunnermunzel : Brunner-Munzel test for stochastic equality
    scipy.stats.mannwhitneyu : Mann-Whitney rank test on two samples.

    Notes
    -----
    Wilcoxon-Mann-Whitney assumes equal variance or equal distribution under
    the Null hypothesis. Fligner-Policello test allows for unequal variances
    but assumes continuous distribution, i.e. no ties.
    Brunner-Munzel extend the test to allow for unequal variance and discrete
    or ordered categorical random variables.

    Brunner and Munzel recommended to estimate the p-value by t-distribution
    when the size of data is 50 or less. If the size is lower than 10, it would
    be better to use permuted Brunner Munzel test (see [2]_) for the test
    of stochastic equality.

    This measure has been introduced in the literature under many different
    names relying on a variety of assumptions.
    In psychology, McGraw and Wong (1992) introduced it as Common Language
    effect size for the continuous, normal distribution case,
    Vargha and Delaney (2000) [3]_ extended it to the nonparametric
    continuous distribution case as in Fligner-Policello.

    WMW and related tests can only be interpreted as test of medians or tests
    of central location only under very restrictive additional assumptions
    such as both distribution are identical under the equality null hypothesis
    (assumed by Mann-Whitney) or both distributions are symmetric (shown by
    Fligner-Policello). If the distribution of the two samples can differ in
    an arbitrary way, then the equality Null hypothesis corresponds to p=0.5
    against an alternative p != 0.5.  see for example Conroy (2012) [4]_ and
    Divine et al (2018) [5]_ .

    Note: Brunner-Munzel and related literature define the probability that x1
    is stochastically smaller than x2, while here we use stochastically larger.
    This equivalent to switching x1 and x2 in the two sample case.

    References
    ----------
    .. [1] Brunner, E. and Munzel, U. "The nonparametric Benhrens-Fisher
           problem: Asymptotic theory and a small-sample approximation".
           Biometrical Journal. Vol. 42(2000): 17-25.
    .. [2] Neubert, K. and Brunner, E. "A studentized permutation test for the
           non-parametric Behrens-Fisher problem". Computational Statistics and
           Data Analysis. Vol. 51(2007): 5192-5204.
    .. [3] Vargha, András, and Harold D. Delaney. 2000. “A Critique and
           Improvement of the CL Common Language Effect Size Statistics of
           McGraw and Wong.” Journal of Educational and Behavioral Statistics
           25 (2): 101–32. https://doi.org/10.3102/10769986025002101.
    .. [4] Conroy, Ronán M. 2012. “What Hypotheses Do ‘Nonparametric’ Two-Group
           Tests Actually Test?” The Stata Journal: Promoting Communications on
           Statistics and Stata 12 (2): 182–90.
           https://doi.org/10.1177/1536867X1201200202.
    .. [5] Divine, George W., H. James Norton, Anna E. Barón, and Elizabeth
           Juarez-Colunga. 2018. “The Wilcoxon–Mann–Whitney Procedure Fails as
           a Test of Medians.” The American Statistician 72 (3): 278–86.
           https://doi.org/10.1080/00031305.2017.1305291.

    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)

    nobs1 = len(x1)
    nobs2 = len(x2)
    nobs = nobs1 + nobs2
    if nobs1 == 0 or nobs2 == 0:
        raise ValueError("one sample has zero length")

    rank1, rank2, ranki1, ranki2 = rankdata_2samp(x1, x2)

    meanr1 = np.mean(rank1, axis=0)
    meanr2 = np.mean(rank2, axis=0)
    meanri1 = np.mean(ranki1, axis=0)
    meanri2 = np.mean(ranki2, axis=0)

    S1 = np.sum(np.power(rank1 - ranki1 - meanr1 + meanri1, 2.0), axis=0)
    S1 /= nobs1 - 1
    S2 = np.sum(np.power(rank2 - ranki2 - meanr2 + meanri2, 2.0), axis=0)
    S2 /= nobs2 - 1

    wbfn = nobs1 * nobs2 * (meanr1 - meanr2)
    wbfn /= (nobs1 + nobs2) * np.sqrt(nobs1 * S1 + nobs2 * S2)

    # Here we only use alternative == "two-sided"
    if use_t:
        df_numer = np.power(nobs1 * S1 + nobs2 * S2, 2.0)
        df_denom = np.power(nobs1 * S1, 2.0) / (nobs1 - 1)
        df_denom += np.power(nobs2 * S2, 2.0) / (nobs2 - 1)
        df = df_numer / df_denom
        pvalue = 2 * stats.t.sf(np.abs(wbfn), df)
    else:
        pvalue = 2 * stats.norm.sf(np.abs(wbfn))
        df = None

    # other info
    var1 = S1 / (nobs - nobs1)**2
    var2 = S2 / (nobs - nobs2)**2
    var_prob = (var1 / nobs1 + var2 / nobs2)
    var = nobs * (var1 / nobs1 + var2 / nobs2)
    prob1 = (meanr1 - (nobs1 + 1) / 2) / nobs2
    prob2 = (meanr2 - (nobs2 + 1) / 2) / nobs1

    return RankCompareResult(statistic=wbfn, pvalue=pvalue, s1=S1, s2=S2,
                             var1=var1, var2=var2, var=var,
                             var_prob=var_prob,
                             nobs1=nobs1, nobs2=nobs2, nobs=nobs,
                             mean1=meanr1, mean2=meanr2,
                             prob1=prob1, prob2=prob2,
                             somersd1=prob1 * 2 - 1, somersd2=prob2 * 2 - 1,
                             df=df, use_t=use_t
                             )


def rank_compare_2ordinal(count1, count2, ddof=1, use_t=True):
    """
    Stochastically larger probability for 2 independent ordinal samples.

    This is a special case of `rank_compare_2indep` when the data are given as
    counts of two independent ordinal, i.e. ordered multinomial, samples.

    The statistic of interest is the probability that a random draw from the
    population of the first sample has a larger value than a random draw from
    the population of the second sample, specifically

        p = P(x1 > x2) + 0.5 * P(x1 = x2)

    Parameters
    ----------
    count1 : array_like
        Counts of the first sample, categories are assumed to be ordered.
    count2 : array_like
        Counts of the second sample, number of categories and ordering needs
        to be the same as for sample 1.
    ddof : scalar
        Degrees of freedom correction for variance estimation. The default
        ddof=1 corresponds to `rank_compare_2indep`.
    use_t : bool
        If use_t is true, the t distribution with Welch-Satterthwaite type
        degrees of freedom is used for p-value and confidence interval.
        If use_t is false, then the normal distribution is used.

    Returns
    -------
    res : RankCompareResult
        This includes methods for hypothesis tests and confidence intervals
        for the probability that sample 1 is stochastically larger than
        sample 2.

    See Also
    --------
    rank_compare_2indep
    RankCompareResult

    Notes
    -----
    The implementation is based on the appendix of Munzel and Hauschke (2003)
    with the addition of ``ddof`` so that the results match the general
    function `rank_compare_2indep`.

    """

    count1 = np.asarray(count1)
    count2 = np.asarray(count2)
    nobs1, nobs2 = count1.sum(), count2.sum()
    freq1 = count1 / nobs1
    freq2 = count2 / nobs2
    cdf1 = np.concatenate(([0], freq1)).cumsum(axis=0)
    cdf2 = np.concatenate(([0], freq2)).cumsum(axis=0)

    # mid rank cdf
    cdfm1 = (cdf1[1:] + cdf1[:-1]) / 2
    cdfm2 = (cdf2[1:] + cdf2[:-1]) / 2
    prob1 = (cdfm2 * freq1).sum()
    prob2 = (cdfm1 * freq2).sum()

    var1 = (cdfm2**2 * freq1).sum() - prob1**2
    var2 = (cdfm1**2 * freq2).sum() - prob2**2

    var_prob = (var1 / (nobs1 - ddof) + var2 / (nobs2 - ddof))
    nobs = nobs1 + nobs2
    var = nobs * var_prob
    vn1 = var1 * nobs2 * nobs1 / (nobs1 - ddof)
    vn2 = var2 * nobs1 * nobs2 / (nobs2 - ddof)
    df = (vn1 + vn2)**2 / (vn1**2 / (nobs1 - 1) + vn2**2 / (nobs2 - 1))
    res = RankCompareResult(statistic=None, pvalue=None, s1=None, s2=None,
                            var1=var1, var2=var2, var=var,
                            var_prob=var_prob,
                            nobs1=nobs1, nobs2=nobs2, nobs=nobs,
                            mean1=None, mean2=None,
                            prob1=prob1, prob2=prob2,
                            somersd1=prob1 * 2 - 1, somersd2=prob2 * 2 - 1,
                            df=df, use_t=use_t
                            )

    return res


def prob_larger_continuous(distr1, distr2):
    """
    Probability indicating that distr1 is stochastically larger than distr2.

    This computes

        p = P(x1 > x2)

    for two continuous distributions, where `distr1` and `distr2` are the
    distributions of random variables x1 and x2 respectively.

    Parameters
    ----------
    distr1, distr2 : distributions
        Two instances of scipy.stats.distributions. The required methods are
        cdf of the second distribution and expect of the first distribution.

    Returns
    -------
    p : probability x1 is larger than x2


    Notes
    -----
    This is a one-liner that is added mainly as reference.

    Examples
    --------
    >>> from scipy import stats
    >>> prob_larger_continuous(stats.norm, stats.t(5))
    0.4999999999999999

    # which is the same as
    >>> stats.norm.expect(stats.t(5).cdf)
    0.4999999999999999

    # distribution 1 with smaller mean (loc) than distribution 2
    >>> prob_larger_continuous(stats.norm, stats.norm(loc=1))
    0.23975006109347669

    """

    return distr1.expect(distr2.cdf)


def cohensd2problarger(d):
    """
    Convert Cohen's d effect size to stochastically-larger-probability.

    This assumes observations are normally distributed.

    Computed as

        p = Prob(x1 > x2) = F(d / sqrt(2))

    where `F` is cdf of normal distribution. Cohen's d is defined as

        d = (mean1 - mean2) / std

    where ``std`` is the pooled within standard deviation.

    Parameters
    ----------
    d : float or array_like
        Cohen's d effect size for difference mean1 - mean2.

    Returns
    -------
    prob : float or ndarray
        Prob(x1 > x2)
    """

    return stats.norm.cdf(d / np.sqrt(2))
