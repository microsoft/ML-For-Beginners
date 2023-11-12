# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 14:34:25 2020

Author: Josef Perktold
License: BSD-3

"""

import numpy as np
import pandas as pd
from scipy import stats

from statsmodels.stats.base import HolderTuple


class CombineResults:
    """Results from combined estimate of means or effect sizes

    This currently includes intermediate results that might be removed
    """

    def __init__(self, **kwds):
        self.__dict__.update(kwds)
        self._ini_keys = list(kwds.keys())

        self.df_resid = self.k - 1

        # TODO: move to property ?
        self.sd_eff_w_fe_hksj = np.sqrt(self.var_hksj_fe)
        self.sd_eff_w_re_hksj = np.sqrt(self.var_hksj_re)

        # explained variance measures
        self.h2 = self.q / (self.k - 1)
        self.i2 = 1 - 1 / self.h2

        # memoize ci_samples
        self.cache_ci = {}

    def conf_int_samples(self, alpha=0.05, use_t=None, nobs=None,
                         ci_func=None):
        """confidence intervals for the effect size estimate of samples

        Additional information needs to be provided for confidence intervals
        that are not based on normal distribution using available variance.
        This is likely to change in future.

        Parameters
        ----------
        alpha : float in (0, 1)
            Significance level for confidence interval. Nominal coverage is
            ``1 - alpha``.
        use_t : None or bool
            If use_t is None, then the attribute `use_t` determines whether
            normal or t-distribution is used for confidence intervals.
            Specifying use_t overrides the attribute.
            If use_t is false, then confidence intervals are based on the
            normal distribution. If it is true, then the t-distribution is
            used.
        nobs : None or float
            Number of observations used for degrees of freedom computation.
            Only used if use_t is true.
        ci_func : None or callable
            User provided function to compute confidence intervals.
            This is not used yet and will allow using non-standard confidence
            intervals.

        Returns
        -------
        ci_eff : tuple of ndarrays
            Tuple (ci_low, ci_upp) with confidence interval computed for each
            sample.

        Notes
        -----
        CombineResults currently only has information from the combine_effects
        function, which does not provide details about individual samples.
        """
        # this is a bit messy, we don't have enough information about
        # computing conf_int already in results for other than normal
        # TODO: maybe there is a better
        if (alpha, use_t) in self.cache_ci:
            return self.cache_ci[(alpha, use_t)]

        if use_t is None:
            use_t = self.use_t

        if ci_func is not None:
            kwds = {"use_t": use_t} if use_t is not None else {}
            ci_eff = ci_func(alpha=alpha, **kwds)
            self.ci_sample_distr = "ci_func"
        else:
            if use_t is False:
                crit = stats.norm.isf(alpha / 2)
                self.ci_sample_distr = "normal"
            else:
                if nobs is not None:
                    df_resid = nobs - 1
                    crit = stats.t.isf(alpha / 2, df_resid)
                    self.ci_sample_distr = "t"
                else:
                    msg = ("`use_t=True` requires `nobs` for each sample "
                           "or `ci_func`. Using normal distribution for "
                           "confidence interval of individual samples.")
                    import warnings
                    warnings.warn(msg)
                    crit = stats.norm.isf(alpha / 2)
                    self.ci_sample_distr = "normal"

            # sgn = np.asarray([-1, 1])
            # ci_eff = self.eff + sgn * crit * self.sd_eff
            ci_low = self.eff - crit * self.sd_eff
            ci_upp = self.eff + crit * self.sd_eff
            ci_eff = (ci_low, ci_upp)

        # if (alpha, use_t) not in self.cache_ci:  # not needed
        self.cache_ci[(alpha, use_t)] = ci_eff
        return ci_eff

    def conf_int(self, alpha=0.05, use_t=None):
        """confidence interval for the overall mean estimate

        Parameters
        ----------
        alpha : float in (0, 1)
            Significance level for confidence interval. Nominal coverage is
            ``1 - alpha``.
        use_t : None or bool
            If use_t is None, then the attribute `use_t` determines whether
            normal or t-distribution is used for confidence intervals.
            Specifying use_t overrides the attribute.
            If use_t is false, then confidence intervals are based on the
            normal distribution. If it is true, then the t-distribution is
            used.

        Returns
        -------
        ci_eff_fe : tuple of floats
            Confidence interval for mean effects size based on fixed effects
            model with scale=1.
        ci_eff_re : tuple of floats
            Confidence interval for mean effects size based on random effects
            model with scale=1
        ci_eff_fe_wls : tuple of floats
            Confidence interval for mean effects size based on fixed effects
            model with estimated scale corresponding to WLS, ie. HKSJ.
        ci_eff_re_wls : tuple of floats
            Confidence interval for mean effects size based on random effects
            model with estimated scale corresponding to WLS, ie. HKSJ.
            If random effects method is fully iterated, i.e. Paule-Mandel, then
            the estimated scale is 1.

        """
        if use_t is None:
            use_t = self.use_t

        if use_t is False:
            crit = stats.norm.isf(alpha / 2)
        else:
            crit = stats.t.isf(alpha / 2, self.df_resid)

        sgn = np.asarray([-1, 1])
        m_fe = self.mean_effect_fe
        m_re = self.mean_effect_re
        ci_eff_fe = m_fe + sgn * crit * self.sd_eff_w_fe
        ci_eff_re = m_re + sgn * crit * self.sd_eff_w_re

        ci_eff_fe_wls = m_fe + sgn * crit * np.sqrt(self.var_hksj_fe)
        ci_eff_re_wls = m_re + sgn * crit * np.sqrt(self.var_hksj_re)

        return ci_eff_fe, ci_eff_re, ci_eff_fe_wls, ci_eff_re_wls

    def test_homogeneity(self):
        """Test whether the means of all samples are the same

        currently no options, test uses chisquare distribution
        default might change depending on `use_t`

        Returns
        -------
        res : HolderTuple instance
            The results include the following attributes:

            - statistic : float
                Test statistic, ``q`` in meta-analysis, this is the
                pearson_chi2 statistic for the fixed effects model.
            - pvalue : float
                P-value based on chisquare distribution.
            - df : float
                Degrees of freedom, equal to number of studies or samples
                minus 1.
        """
        pvalue = stats.chi2.sf(self.q, self.k - 1)
        res = HolderTuple(statistic=self.q,
                          pvalue=pvalue,
                          df=self.k - 1,
                          distr="chi2")
        return res

    def summary_array(self, alpha=0.05, use_t=None):
        """Create array with sample statistics and mean estimates

        Parameters
        ----------
        alpha : float in (0, 1)
            Significance level for confidence interval. Nominal coverage is
            ``1 - alpha``.
        use_t : None or bool
            If use_t is None, then the attribute `use_t` determines whether
            normal or t-distribution is used for confidence intervals.
            Specifying use_t overrides the attribute.
            If use_t is false, then confidence intervals are based on the
            normal distribution. If it is true, then the t-distribution is
            used.

        Returns
        -------
        res : ndarray
            Array with columns
            ['eff', "sd_eff", "ci_low", "ci_upp", "w_fe","w_re"].
            Rows include statistics for samples and estimates of overall mean.
        column_names : list of str
            The names for the columns, used when creating summary DataFrame.
        """

        ci_low, ci_upp = self.conf_int_samples(alpha=alpha, use_t=use_t)
        res = np.column_stack([self.eff, self.sd_eff,
                               ci_low, ci_upp,
                               self.weights_rel_fe, self.weights_rel_re])

        ci = self.conf_int(alpha=alpha, use_t=use_t)
        res_fe = [[self.mean_effect_fe, self.sd_eff_w_fe,
                   ci[0][0], ci[0][1], 1, np.nan]]
        res_re = [[self.mean_effect_re, self.sd_eff_w_re,
                   ci[1][0], ci[1][1], np.nan, 1]]
        res_fe_wls = [[self.mean_effect_fe, self.sd_eff_w_fe_hksj,
                       ci[2][0], ci[2][1], 1, np.nan]]
        res_re_wls = [[self.mean_effect_re, self.sd_eff_w_re_hksj,
                       ci[3][0], ci[3][1], np.nan, 1]]

        res = np.concatenate([res, res_fe, res_re, res_fe_wls, res_re_wls],
                             axis=0)
        column_names = ['eff', "sd_eff", "ci_low", "ci_upp", "w_fe", "w_re"]
        return res, column_names

    def summary_frame(self, alpha=0.05, use_t=None):
        """Create DataFrame with sample statistics and mean estimates

        Parameters
        ----------
        alpha : float in (0, 1)
            Significance level for confidence interval. Nominal coverage is
            ``1 - alpha``.
        use_t : None or bool
            If use_t is None, then the attribute `use_t` determines whether
            normal or t-distribution is used for confidence intervals.
            Specifying use_t overrides the attribute.
            If use_t is false, then confidence intervals are based on the
            normal distribution. If it is true, then the t-distribution is
            used.

        Returns
        -------
        res : DataFrame
            pandas DataFrame instance with columns
            ['eff', "sd_eff", "ci_low", "ci_upp", "w_fe","w_re"].
            Rows include statistics for samples and estimates of overall mean.

        """
        if use_t is None:
            use_t = self.use_t
        labels = (list(self.row_names) +
                  ["fixed effect", "random effect",
                   "fixed effect wls", "random effect wls"])
        res, col_names = self.summary_array(alpha=alpha, use_t=use_t)
        results = pd.DataFrame(res, index=labels, columns=col_names)
        return results

    def plot_forest(self, alpha=0.05, use_t=None, use_exp=False,
                    ax=None, **kwds):
        """Forest plot with means and confidence intervals

        Parameters
        ----------
        ax : None or matplotlib axis instance
            If ax is provided, then the plot will be added to it.
        alpha : float in (0, 1)
            Significance level for confidence interval. Nominal coverage is
            ``1 - alpha``.
        use_t : None or bool
            If use_t is None, then the attribute `use_t` determines whether
            normal or t-distribution is used for confidence intervals.
            Specifying use_t overrides the attribute.
            If use_t is false, then confidence intervals are based on the
            normal distribution. If it is true, then the t-distribution is
            used.
        use_exp : bool
            If `use_exp` is True, then the effect size and confidence limits
            will be exponentiated. This transform log-odds-ration into
            odds-ratio, and similarly for risk-ratio.
        ax : AxesSubplot, optional
            If given, this axes is used to plot in instead of a new figure
            being created.
        kwds : optional keyword arguments
            Keywords are forwarded to the dot_plot function that creates the
            plot.

        Returns
        -------
        fig : Matplotlib figure instance

        See Also
        --------
        dot_plot

        """
        from statsmodels.graphics.dotplots import dot_plot
        res_df = self.summary_frame(alpha=alpha, use_t=use_t)
        if use_exp:
            res_df = np.exp(res_df[["eff", "ci_low", "ci_upp"]])
        hw = np.abs(res_df[["ci_low", "ci_upp"]] - res_df[["eff"]].values)
        fig = dot_plot(points=res_df["eff"], intervals=hw,
                       lines=res_df.index, line_order=res_df.index, **kwds)
        return fig


def effectsize_smd(mean1, sd1, nobs1, mean2, sd2, nobs2):
    """effect sizes for mean difference for use in meta-analysis

    mean1, sd1, nobs1 are for treatment
    mean2, sd2, nobs2 are for control

    Effect sizes are computed for the mean difference ``mean1 - mean2``
    standardized by an estimate of the within variance.

    This does not have option yet.
    It uses standardized mean difference with bias correction as effect size.

    This currently does not use np.asarray, all computations are possible in
    pandas.

    Parameters
    ----------
    mean1 : array
        mean of second sample, treatment groups
    sd1 : array
        standard deviation of residuals in treatment groups, within
    nobs1 : array
        number of observations in treatment groups
    mean2, sd2, nobs2 : arrays
        mean, standard deviation and number of observations of control groups

    Returns
    -------
    smd_bc : array
        bias corrected estimate of standardized mean difference
    var_smdbc : array
        estimate of variance of smd_bc

    Notes
    -----
    Status: API will still change. This is currently intended for support of
    meta-analysis.

    References
    ----------
    Borenstein, Michael. 2009. Introduction to Meta-Analysis.
        Chichester: Wiley.

    Chen, Ding-Geng, and Karl E. Peace. 2013. Applied Meta-Analysis with R.
        Chapman & Hall/CRC Biostatistics Series.
        Boca Raton: CRC Press/Taylor & Francis Group.

    """
    # TODO: not used yet, design and options ?
    # k = len(mean1)
    # if row_names is None:
    #    row_names = list(range(k))
    # crit = stats.norm.isf(alpha / 2)
    # var_diff_uneq = sd1**2 / nobs1 + sd2**2 / nobs2
    var_diff = (sd1**2 * (nobs1 - 1) +
                sd2**2 * (nobs2 - 1)) / (nobs1 + nobs2 - 2)
    sd_diff = np.sqrt(var_diff)
    nobs = nobs1 + nobs2
    bias_correction = 1 - 3 / (4 * nobs - 9)
    smd = (mean1 - mean2) / sd_diff
    smd_bc = bias_correction * smd
    var_smdbc = nobs / nobs1 / nobs2 + smd_bc**2 / 2 / (nobs - 3.94)
    return smd_bc, var_smdbc


def effectsize_2proportions(count1, nobs1, count2, nobs2, statistic="diff",
                            zero_correction=None, zero_kwds=None):
    """Effects sizes for two sample binomial proportions

    Parameters
    ----------
    count1, nobs1, count2, nobs2 : array_like
        data for two samples
    statistic : {"diff", "odds-ratio", "risk-ratio", "arcsine"}
        statistic for the comparison of two proportions
        Effect sizes for "odds-ratio" and "risk-ratio" are in logarithm.
    zero_correction : {None, float, "tac", "clip"}
        Some statistics are not finite when zero counts are in the data.
        The options to remove zeros are:

        * float : if zero_correction is a single float, then it will be added
          to all count (cells) if the sample has any zeros.
        * "tac" : treatment arm continuity correction see Ruecker et al 2009,
          section 3.2
        * "clip" : clip proportions without adding a value to all cells
          The clip bounds can be set with zero_kwds["clip_bounds"]

    zero_kwds : dict
        additional options to handle zero counts
        "clip_bounds" tuple, default (1e-6, 1 - 1e-6) if zero_correction="clip"
        other options not yet implemented

    Returns
    -------
    effect size : array
        Effect size for each sample.
    var_es : array
        Estimate of variance of the effect size

    Notes
    -----
    Status: API is experimental, Options for zero handling is incomplete.

    The names for ``statistics`` keyword can be shortened to "rd", "rr", "or"
    and "as".

    The statistics are defined as:

     - risk difference = p1 - p2
     - log risk ratio = log(p1 / p2)
     - log odds_ratio = log(p1 / (1 - p1) * (1 - p2) / p2)
     - arcsine-sqrt = arcsin(sqrt(p1)) - arcsin(sqrt(p2))

    where p1 and p2 are the estimated proportions in sample 1 (treatment) and
    sample 2 (control).

    log-odds-ratio and log-risk-ratio can be transformed back to ``or`` and
    `rr` using `exp` function.

    See Also
    --------
    statsmodels.stats.contingency_tables
    """
    if zero_correction is None:
        cc1 = cc2 = 0
    elif zero_correction == "tac":
        # treatment arm continuity correction Ruecker et al 2009, section 3.2
        nobs_t = nobs1 + nobs2
        cc1 = nobs2 / nobs_t
        cc2 = nobs1 / nobs_t
    elif zero_correction == "clip":
        clip_bounds = zero_kwds.get("clip_bounds", (1e-6, 1 - 1e-6))
        cc1 = cc2 = 0
    elif zero_correction:
        # TODO: check is float_like
        cc1 = cc2 = zero_correction
    else:
        msg = "zero_correction not recognized or supported"
        raise NotImplementedError(msg)

    zero_mask1 = (count1 == 0) | (count1 == nobs1)
    zero_mask2 = (count2 == 0) | (count2 == nobs2)
    zmask = np.logical_or(zero_mask1, zero_mask2)
    n1 = nobs1 + (cc1 + cc2) * zmask
    n2 = nobs2 + (cc1 + cc2) * zmask
    p1 = (count1 + cc1) / (n1)
    p2 = (count2 + cc2) / (n2)

    if zero_correction == "clip":
        p1 = np.clip(p1, *clip_bounds)
        p2 = np.clip(p2, *clip_bounds)

    if statistic in ["diff", "rd"]:
        rd = p1 - p2
        rd_var = p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2
        eff = rd
        var_eff = rd_var
    elif statistic in ["risk-ratio", "rr"]:
        # rr = p1 / p2
        log_rr = np.log(p1) - np.log(p2)
        log_rr_var = (1 - p1) / p1 / n1 + (1 - p2) / p2 / n2
        eff = log_rr
        var_eff = log_rr_var
    elif statistic in ["odds-ratio", "or"]:
        # or_ = p1 / (1 - p1) * (1 - p2) / p2
        log_or = np.log(p1) - np.log(1 - p1) - np.log(p2) + np.log(1 - p2)
        log_or_var = 1 / (p1 * (1 - p1) * n1) + 1 / (p2 * (1 - p2) * n2)
        eff = log_or
        var_eff = log_or_var
    elif statistic in ["arcsine", "arcsin", "as"]:
        as_ = np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2))
        as_var = (1 / n1 + 1 / n2) / 4
        eff = as_
        var_eff = as_var
    else:
        msg = 'statistic not recognized, use one of "rd", "rr", "or", "as"'
        raise NotImplementedError(msg)

    return eff, var_eff


def combine_effects(effect, variance, method_re="iterated", row_names=None,
                    use_t=False, alpha=0.05, **kwds):
    """combining effect sizes for effect sizes using meta-analysis

    This currently does not use np.asarray, all computations are possible in
    pandas.

    Parameters
    ----------
    effect : array
        mean of effect size measure for all samples
    variance : array
        variance of mean or effect size measure for all samples
    method_re : {"iterated", "chi2"}
        method that is use to compute the between random effects variance
        "iterated" or "pm" uses Paule and Mandel method to iteratively
        estimate the random effects variance. Options for the iteration can
        be provided in the ``kwds``
        "chi2" or "dl" uses DerSimonian and Laird one-step estimator.
    row_names : list of strings (optional)
        names for samples or studies, will be included in results summary and
        table.
    alpha : float in (0, 1)
        significance level, default is 0.05, for the confidence intervals

    Returns
    -------
    results : CombineResults
        Contains estimation results and intermediate statistics, and includes
        a method to return a summary table.
        Statistics from intermediate calculations might be removed at a later
        time.

    Notes
    -----
    Status: Basic functionality is verified, mainly compared to R metafor
    package. However, API might still change.

    This computes both fixed effects and random effects estimates. The
    random effects results depend on the method to estimate the RE variance.

    Scale estimate
    In fixed effects models and in random effects models without fully
    iterated random effects variance, the model will in general not account
    for all residual variance. Traditional meta-analysis uses a fixed
    scale equal to 1, that might not produce test statistics and
    confidence intervals with the correct size. Estimating the scale to account
    for residual variance often improves the small sample properties of
    inference and confidence intervals.
    This adjustment to the standard errors is often referred to as HKSJ
    method based attributed to Hartung and Knapp and Sidik and Jonkman.
    However, this is equivalent to estimating the scale in WLS.
    The results instance includes both, fixed scale and estimated scale
    versions of standard errors and confidence intervals.

    References
    ----------
    Borenstein, Michael. 2009. Introduction to Meta-Analysis.
        Chichester: Wiley.

    Chen, Ding-Geng, and Karl E. Peace. 2013. Applied Meta-Analysis with R.
        Chapman & Hall/CRC Biostatistics Series.
        Boca Raton: CRC Press/Taylor & Francis Group.

    """

    k = len(effect)
    if row_names is None:
        row_names = list(range(k))
    crit = stats.norm.isf(alpha / 2)

    # alias for initial version
    eff = effect
    var_eff = variance
    sd_eff = np.sqrt(var_eff)

    # fixed effects computation

    weights_fe = 1 / var_eff  # no bias correction ?
    w_total_fe = weights_fe.sum(0)
    weights_rel_fe = weights_fe / w_total_fe

    eff_w_fe = weights_rel_fe * eff
    mean_effect_fe = eff_w_fe.sum()
    var_eff_w_fe = 1 / w_total_fe
    sd_eff_w_fe = np.sqrt(var_eff_w_fe)

    # random effects computation

    q = (weights_fe * eff**2).sum(0)
    q -= (weights_fe * eff).sum()**2 / w_total_fe
    df = k - 1

    if method_re.lower() in ["iterated", "pm"]:
        tau2, _ = _fit_tau_iterative(eff, var_eff, **kwds)
    elif method_re.lower() in ["chi2", "dl"]:
        c = w_total_fe - (weights_fe**2).sum() / w_total_fe
        tau2 = (q - df) / c
    else:
        raise ValueError('method_re should be "iterated" or "chi2"')

    weights_re = 1 / (var_eff + tau2)  # no  bias_correction ?
    w_total_re = weights_re.sum(0)
    weights_rel_re = weights_re / weights_re.sum(0)

    eff_w_re = weights_rel_re * eff
    mean_effect_re = eff_w_re.sum()
    var_eff_w_re = 1 / w_total_re
    sd_eff_w_re = np.sqrt(var_eff_w_re)
    # ci_low_eff_re = mean_effect_re - crit * sd_eff_w_re
    # ci_upp_eff_re = mean_effect_re + crit * sd_eff_w_re

    scale_hksj_re = (weights_re * (eff - mean_effect_re)**2).sum() / df
    scale_hksj_fe = (weights_fe * (eff - mean_effect_fe)**2).sum() / df
    var_hksj_re = (weights_rel_re * (eff - mean_effect_re)**2).sum() / df
    var_hksj_fe = (weights_rel_fe * (eff - mean_effect_fe)**2).sum() / df

    res = CombineResults(**locals())
    return res


def _fit_tau_iterative(eff, var_eff, tau2_start=0, atol=1e-5, maxiter=50):
    """Paule-Mandel iterative estimate of between random effect variance

    implementation follows DerSimonian and Kacker 2007 Appendix 8
    see also Kacker 2004

    Parameters
    ----------
    eff : ndarray
        effect sizes
    var_eff : ndarray
        variance of effect sizes
    tau2_start : float
        starting value for iteration
    atol : float, default: 1e-5
        convergence tolerance for absolute value of estimating equation
    maxiter : int
        maximum number of iterations

    Returns
    -------
    tau2 : float
        estimate of random effects variance tau squared
    converged : bool
        True if iteration has converged.

    """
    tau2 = tau2_start
    k = eff.shape[0]
    converged = False
    for i in range(maxiter):
        w = 1 / (var_eff + tau2)
        m = w.dot(eff) / w.sum(0)
        resid_sq = (eff - m)**2
        q_w = w.dot(resid_sq)
        # estimating equation
        ee = q_w - (k - 1)
        if ee < 0:
            tau2 = 0
            converged = 0
            break
        if np.allclose(ee, 0, atol=atol):
            converged = True
            break
        # update tau2
        delta = ee / (w**2).dot(resid_sq)
        tau2 += delta

    return tau2, converged


def _fit_tau_mm(eff, var_eff, weights):
    """one-step method of moment estimate of between random effect variance

    implementation follows Kacker 2004 and DerSimonian and Kacker 2007 eq. 6

    Parameters
    ----------
    eff : ndarray
        effect sizes
    var_eff : ndarray
        variance of effect sizes
    weights : ndarray
        weights for estimating overall weighted mean

    Returns
    -------
    tau2 : float
        estimate of random effects variance tau squared

    """
    w = weights

    m = w.dot(eff) / w.sum(0)
    resid_sq = (eff - m)**2
    q_w = w.dot(resid_sq)
    w_t = w.sum()
    expect = w.dot(var_eff) - (w**2).dot(var_eff) / w_t
    denom = w_t - (w**2).sum() / w_t
    # moment estimate from estimating equation
    tau2 = (q_w - expect) / denom

    return tau2


def _fit_tau_iter_mm(eff, var_eff, tau2_start=0, atol=1e-5, maxiter=50):
    """iterated method of moment estimate of between random effect variance

    This repeatedly estimates tau, updating weights in each iteration
    see two-step estimators in DerSimonian and Kacker 2007

    Parameters
    ----------
    eff : ndarray
        effect sizes
    var_eff : ndarray
        variance of effect sizes
    tau2_start : float
        starting value for iteration
    atol : float, default: 1e-5
        convergence tolerance for change in tau2 estimate between iterations
    maxiter : int
        maximum number of iterations

    Returns
    -------
    tau2 : float
        estimate of random effects variance tau squared
    converged : bool
        True if iteration has converged.

    """
    tau2 = tau2_start
    converged = False
    for _ in range(maxiter):
        w = 1 / (var_eff + tau2)

        tau2_new = _fit_tau_mm(eff, var_eff, w)
        tau2_new = max(0, tau2_new)

        delta = tau2_new - tau2
        if np.allclose(delta, 0, atol=atol):
            converged = True
            break

        tau2 = tau2_new

    return tau2, converged
