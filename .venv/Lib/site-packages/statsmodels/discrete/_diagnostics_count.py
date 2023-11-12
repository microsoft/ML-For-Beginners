# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 12:53:45 2017

Author: Josef Perktold
"""

import numpy as np
from scipy import stats

import pandas as pd

from statsmodels.stats.base import HolderTuple
from statsmodels.discrete.discrete_model import Poisson
from statsmodels.regression.linear_model import OLS


def _combine_bins(edge_index, x):
    """group columns into bins using sum

    This is mainly a helper function for combining probabilities into cells.
    It similar to `np.add.reduceat(x, edge_index, axis=-1)` except for the
    treatment of the last index and last cell.

    Parameters
    ----------
    edge_index : array_like
         This defines the (zero-based) indices for the columns that are be
         combined. Each index in `edge_index` except the last is the starting
         index for a bin. The largest index in a bin is the next edge_index-1.
    x : 1d or 2d array
        array for which columns are combined. If x is 1-dimensional that it
        will be treated as a 2-d row vector.

    Returns
    -------
    x_new : ndarray
    k_li : ndarray
        Count of columns combined in bin.


    Examples
    --------
    >>> dia.combine_bins([0,1,5], np.arange(4))
    (array([0, 6]), array([1, 4]))

    this aggregates to two bins with the sum of 1 and 4 elements
    >>> np.arange(4)[0].sum()
    0
    >>> np.arange(4)[1:5].sum()
    6

    If the rightmost index is smaller than len(x)+1, then the remaining
    columns will not be included.

    >>> dia.combine_bins([0,1,3], np.arange(4))
    (array([0, 3]), array([1, 2]))
    """
    x = np.asarray(x)
    if x.ndim == 1:
        is_1d = True
        x = x[None, :]
    else:
        is_1d = False
    xli = []
    kli = []
    for bin_idx in range(len(edge_index) - 1):
        i, j = edge_index[bin_idx : bin_idx + 2]
        xli.append(x[:, i:j].sum(1))
        kli.append(j - i)

    x_new = np.column_stack(xli)
    if is_1d:
        x_new = x_new.squeeze()
    return x_new, np.asarray(kli)


def plot_probs(freq, probs_predicted, label='predicted', upp_xlim=None,
               fig=None):
    """diagnostic plots for comparing two lists of discrete probabilities

    Parameters
    ----------
    freq, probs_predicted : nd_arrays
        two arrays of probabilities, this can be any probabilities for
        the same events, default is designed for comparing predicted
        and observed probabilities
    label : str or tuple
        If string, then it will be used as the label for probs_predicted and
        "freq" is used for the other probabilities.
        If label is a tuple of strings, then the first is they are used as
        label for both probabilities

    upp_xlim : None or int
        If it is not None, then the xlim of the first two plots are set to
        (0, upp_xlim), otherwise the matplotlib default is used
    fig : None or matplotlib figure instance
        If fig is provided, then the axes will be added to it in a (3,1)
        subplots, otherwise a matplotlib figure instance is created

    Returns
    -------
    Figure
        The figure contains 3 subplot with probabilities, cumulative
        probabilities and a PP-plot
    """

    if isinstance(label, list):
        label0, label1 = label
    else:
        label0, label1 = 'freq', label

    if fig is None:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8,12))
    ax1 = fig.add_subplot(311)
    ax1.plot(freq, '-o', label=label0)
    ax1.plot(probs_predicted, '-d', label=label1)
    if upp_xlim is not None:
        ax1.set_xlim(0, upp_xlim)
    ax1.legend()
    ax1.set_title('probabilities')

    ax2 = fig.add_subplot(312)
    ax2.plot(np.cumsum(freq), '-o', label=label0)
    ax2.plot(np.cumsum(probs_predicted), '-d', label=label1)
    if upp_xlim is not None:
        ax2.set_xlim(0, upp_xlim)
    ax2.legend()
    ax2.set_title('cumulative probabilities')

    ax3 = fig.add_subplot(313)
    ax3.plot(np.cumsum(probs_predicted), np.cumsum(freq), 'o')
    ax3.plot(np.arange(len(freq)) / len(freq), np.arange(len(freq)) / len(freq))
    ax3.set_title('PP-plot')
    ax3.set_xlabel(label1)
    ax3.set_ylabel(label0)
    return fig


def test_chisquare_prob(results, probs, bin_edges=None, method=None):
    """
    chisquare test for predicted probabilities using cmt-opg

    Parameters
    ----------
    results : results instance
        Instance of a count regression results
    probs : ndarray
        Array of predicted probabilities with observations
        in rows and event counts in columns
    bin_edges : None or array
        intervals to combine several counts into cells
        see combine_bins

    Returns
    -------
    (api not stable, replace by test-results class)
    statistic : float
        chisquare statistic for tes
    p-value : float
        p-value of test
    df : int
        degrees of freedom for chisquare distribution
    extras : ???
        currently returns a tuple with some intermediate results
        (diff, res_aux)

    Notes
    -----

    Status : experimental, no verified unit tests, needs to be generalized
    currently only OPG version with auxiliary regression is implemented

    Assumes counts are np.arange(probs.shape[1]), i.e. consecutive
    integers starting at zero.

    Auxiliary regression drops the last column of binned probs to avoid
    that probabilities sum to 1.

    References
    ----------
    .. [1] Andrews, Donald W. K. 1988a. “Chi-Square Diagnostic Tests for
           Econometric Models: Theory.” Econometrica 56 (6): 1419–53.
           https://doi.org/10.2307/1913105.

    .. [2] Andrews, Donald W. K. 1988b. “Chi-Square Diagnostic Tests for
           Econometric Models.” Journal of Econometrics 37 (1): 135–56.
           https://doi.org/10.1016/0304-4076(88)90079-6.

    .. [3] Manjón, M., and O. Martínez. 2014. “The Chi-Squared Goodness-of-Fit
           Test for Count-Data Models.” Stata Journal 14 (4): 798–816.
    """
    res = results
    score_obs = results.model.score_obs(results.params)
    d_ind = (res.model.endog[:, None] == np.arange(probs.shape[1])).astype(int)
    if bin_edges is not None:
        d_ind_bins, k_bins = _combine_bins(bin_edges, d_ind)
        probs_bins, k_bins = _combine_bins(bin_edges, probs)
        k_bins = probs_bins.shape[-1]
    else:
        d_ind_bins, k_bins = d_ind, d_ind.shape[1]
        probs_bins = probs
    diff1 = d_ind_bins - probs_bins
    # diff2 = (1 - d_ind.sum(1)) - (1 - probs_bins.sum(1))
    x_aux = np.column_stack((score_obs, diff1[:, :-1]))  # diff2))
    nobs = x_aux.shape[0]
    res_aux = OLS(np.ones(nobs), x_aux).fit()

    chi2_stat = nobs * (1 - res_aux.ssr / res_aux.uncentered_tss)
    df = res_aux.model.rank - score_obs.shape[1]
    if df < k_bins - 1:
        # not a problem in general, but it can be for OPG version
        import warnings
        # TODO: Warning shows up in Monte Carlo loop, skip for now
        warnings.warn('auxiliary model is rank deficient')

    statistic = chi2_stat
    pvalue = stats.chi2.sf(chi2_stat, df)

    res = HolderTuple(
        statistic=statistic,
        pvalue=pvalue,
        df=df,
        diff1=diff1,
        res_aux=res_aux,
        distribution="chi2",
        )
    return res


class DispersionResults(HolderTuple):

    def summary_frame(self):
        frame = pd.DataFrame({
            "statistic": self.statistic,
            "pvalue": self.pvalue,
            "method": self.method,
            "alternative": self.alternative
            })

        return frame


def test_poisson_dispersion(results, method="all", _old=False):
    """Score/LM type tests for Poisson variance assumptions

    Null Hypothesis is

    H0: var(y) = E(y) and assuming E(y) is correctly specified
    H1: var(y) ~= E(y)

    The tests are based on the constrained model, i.e. the Poisson model.
    The tests differ in their assumed alternatives, and in their maintained
    assumptions.

    Parameters
    ----------
    results : Poisson results instance
        This can be a results instance for either a discrete Poisson or a GLM
        with family Poisson.
    method : str
        Not used yet. Currently results for all methods are returned.
    _old : bool
        Temporary keyword for backwards compatibility, will be removed
        in future version of statsmodels.

    Returns
    -------
    res : instance
        The instance of DispersionResults has the hypothesis test results,
        statistic, pvalue, method, alternative, as main attributes and a
        summary_frame method that returns the results as pandas DataFrame.

    """

    if method not in ["all"]:
        raise ValueError(f'unknown method "{method}"')

    if hasattr(results, '_results'):
        results = results._results

    endog = results.model.endog
    nobs = endog.shape[0]  # TODO: use attribute, may need to be added
    fitted = results.predict()
    # fitted = results.fittedvalues  # discrete has linear prediction
    # this assumes Poisson
    resid2 = results.resid_response**2
    var_resid_endog = (resid2 - endog)
    var_resid_fitted = (resid2 - fitted)
    std1 = np.sqrt(2 * (fitted**2).sum())

    var_resid_endog_sum = var_resid_endog.sum()
    dean_a = var_resid_fitted.sum() / std1
    dean_b = var_resid_endog_sum / std1
    dean_c = (var_resid_endog / fitted).sum() / np.sqrt(2 * nobs)

    pval_dean_a = 2 * stats.norm.sf(np.abs(dean_a))
    pval_dean_b = 2 * stats.norm.sf(np.abs(dean_b))
    pval_dean_c = 2 * stats.norm.sf(np.abs(dean_c))

    results_all = [[dean_a, pval_dean_a],
                   [dean_b, pval_dean_b],
                   [dean_c, pval_dean_c]]
    description = [['Dean A', 'mu (1 + a mu)'],
                   ['Dean B', 'mu (1 + a mu)'],
                   ['Dean C', 'mu (1 + a)']]

    # Cameron Trived auxiliary regression page 78 count book 1989
    endog_v = var_resid_endog / fitted
    res_ols_nb2 = OLS(endog_v, fitted).fit(use_t=False)
    stat_ols_nb2 = res_ols_nb2.tvalues[0]
    pval_ols_nb2 = res_ols_nb2.pvalues[0]
    results_all.append([stat_ols_nb2, pval_ols_nb2])
    description.append(['CT nb2', 'mu (1 + a mu)'])

    res_ols_nb1 = OLS(endog_v, fitted).fit(use_t=False)
    stat_ols_nb1 = res_ols_nb1.tvalues[0]
    pval_ols_nb1 = res_ols_nb1.pvalues[0]
    results_all.append([stat_ols_nb1, pval_ols_nb1])
    description.append(['CT nb1', 'mu (1 + a)'])

    endog_v = var_resid_endog / fitted
    res_ols_nb2 = OLS(endog_v, fitted).fit(cov_type='HC3', use_t=False)
    stat_ols_hc1_nb2 = res_ols_nb2.tvalues[0]
    pval_ols_hc1_nb2 = res_ols_nb2.pvalues[0]
    results_all.append([stat_ols_hc1_nb2, pval_ols_hc1_nb2])
    description.append(['CT nb2 HC3', 'mu (1 + a mu)'])

    res_ols_nb1 = OLS(endog_v, np.ones(len(endog_v))).fit(cov_type='HC3',
                                                          use_t=False)
    stat_ols_hc1_nb1 = res_ols_nb1.tvalues[0]
    pval_ols_hc1_nb1 = res_ols_nb1.pvalues[0]
    results_all.append([stat_ols_hc1_nb1, pval_ols_hc1_nb1])
    description.append(['CT nb1 HC3', 'mu (1 + a)'])

    results_all = np.array(results_all)
    if _old:
        # for backwards compatibility in 0.14, remove in later versions
        return results_all, description
    else:
        res = DispersionResults(
            statistic=results_all[:, 0],
            pvalue=results_all[:, 1],
            method=[i[0] for i in description],
            alternative=[i[1] for i in description],
            name="Poisson Dispersion Test"
            )
        return res


def _test_poisson_dispersion_generic(
        results,
        exog_new_test,
        exog_new_control=None,
        include_score=False,
        use_endog=True,
        cov_type='HC3',
        cov_kwds=None,
        use_t=False
        ):
    """A variable addition test for the variance function

    This uses an artificial regression to calculate a variant of an LM or
    generalized score test for the specification of the variance assumption
    in a Poisson model. The performed test is a Wald test on the coefficients
    of the `exog_new_test`.

    Warning: insufficiently tested, especially for options
    """

    if hasattr(results, '_results'):
        results = results._results

    endog = results.model.endog
    nobs = endog.shape[0]   # TODO: use attribute, may need to be added
    # fitted = results.fittedvalues  # generic has linpred as fittedvalues
    fitted = results.predict()
    resid2 = results.resid_response**2
    # the following assumes Poisson
    if use_endog:
        var_resid = (resid2 - endog)
    else:
        var_resid = (resid2 - fitted)

    endog_v = var_resid / fitted

    k_constraints = exog_new_test.shape[1]
    ex_list = [exog_new_test]
    if include_score:
        score_obs = results.model.score_obs(results.params)
        ex_list.append(score_obs)

    if exog_new_control is not None:
        ex_list.append(score_obs)

    if len(ex_list) > 1:
        ex = np.column_stack(ex_list)
        use_wald = True
    else:
        ex = ex_list[0]  # no control variables in exog
        use_wald = False

    res_ols = OLS(endog_v, ex).fit(cov_type=cov_type, cov_kwds=cov_kwds,
                                   use_t=use_t)

    if use_wald:
        # we have controls and need to test coefficients
        k_vars = ex.shape[1]
        constraints = np.eye(k_constraints, k_vars)
        ht = res_ols.wald_test(constraints)
        stat_ols = ht.statistic
        pval_ols = ht.pvalue
    else:
        # we do not have controls and can use overall fit
        nobs = endog_v.shape[0]
        rsquared_noncentered = 1 - res_ols.ssr/res_ols.uncentered_tss
        stat_ols = nobs * rsquared_noncentered
        pval_ols = stats.chi2.sf(stat_ols, k_constraints)

    return stat_ols, pval_ols


def test_poisson_zeroinflation_jh(results_poisson, exog_infl=None):
    """score test for zero inflation or deflation in Poisson

    This implements Jansakul and Hinde 2009 score test
    for excess zeros against a zero modified Poisson
    alternative. They use a linear link function for the
    inflation model to allow for zero deflation.

    Parameters
    ----------
    results_poisson: results instance
        The test is only valid if the results instance is a Poisson
        model.
    exog_infl : ndarray
        Explanatory variables for the zero inflated or zero modified
        alternative. I exog_infl is None, then the inflation
        probability is assumed to be constant.

    Returns
    -------
    score test results based on chisquare distribution

    Notes
    -----
    This is a score test based on the null hypothesis that
    the true model is Poisson. It will also reject for
    other deviations from a Poisson model if those affect
    the zero probabilities, e.g. in the direction of
    excess dispersion as in the Negative Binomial
    or Generalized Poisson model.
    Therefore, rejection in this test does not imply that
    zero-inflated Poisson is the appropriate model.

    Status: experimental, no verified unit tests,

    TODO: If the zero modification probability is assumed
    to be constant under the alternative, then we only have
    a scalar test score and we can use one-sided tests to
    distinguish zero inflation and deflation from the
    two-sided deviations. (The general one-sided case is
    difficult.)
    In this case the test specializes to the test by Broek

    References
    ----------
    .. [1] Jansakul, N., and J. P. Hinde. 2002. “Score Tests for Zero-Inflated
           Poisson Models.” Computational Statistics & Data Analysis 40 (1):
           75–96. https://doi.org/10.1016/S0167-9473(01)00104-9.
    """
    if not isinstance(results_poisson.model, Poisson):
        # GLM Poisson would be also valid, not tried
        import warnings
        warnings.warn('Test is only valid if model is Poisson')

    nobs = results_poisson.model.endog.shape[0]

    if exog_infl is None:
        exog_infl = np.ones((nobs, 1))


    endog = results_poisson.model.endog
    exog = results_poisson.model.exog

    mu = results_poisson.predict()
    prob_zero = np.exp(-mu)

    cov_poi = results_poisson.cov_params()
    cross_derivative = (exog_infl.T * (-mu)).dot(exog).T
    cov_infl = (exog_infl.T * ((1 - prob_zero) / prob_zero)).dot(exog_infl)
    score_obs_infl = exog_infl * (((endog == 0) - prob_zero) / prob_zero)[:,None]
    #score_obs_infl = exog_infl * ((endog == 0) * (1 - prob_zero) / prob_zero - (endog>0))[:,None] #same
    score_infl = score_obs_infl.sum(0)
    cov_score_infl = cov_infl - cross_derivative.T.dot(cov_poi).dot(cross_derivative)
    cov_score_infl_inv = np.linalg.pinv(cov_score_infl)

    statistic = score_infl.dot(cov_score_infl_inv).dot(score_infl)
    df2 = np.linalg.matrix_rank(cov_score_infl)  # more general, maybe not needed
    df = exog_infl.shape[1]
    pvalue = stats.chi2.sf(statistic, df)

    res = HolderTuple(
        statistic=statistic,
        pvalue=pvalue,
        df=df,
        rank_score=df2,
        distribution="chi2",
        )
    return res


def test_poisson_zeroinflation_broek(results_poisson):
    """score test for zero modification in Poisson, special case

    This assumes that the Poisson model has a constant and that
    the zero modification probability is constant.

    This is a special case of test_poisson_zeroinflation derived by
    van den Broek 1995.

    The test reports two sided and one sided alternatives based on
    the normal distribution of the test statistic.

    References
    ----------
    .. [1] Broek, Jan van den. 1995. “A Score Test for Zero Inflation in a
           Poisson Distribution.” Biometrics 51 (2): 738–43.
           https://doi.org/10.2307/2532959.

    """

    mu = results_poisson.predict()
    prob_zero = np.exp(-mu)
    endog = results_poisson.model.endog
    # nobs = len(endog)
    # score =  ((endog == 0) / prob_zero).sum() - nobs
    # var_score = (1 / prob_zero).sum() - nobs - endog.sum()
    score = (((endog == 0) - prob_zero) / prob_zero).sum()
    var_score = ((1 - prob_zero) / prob_zero).sum() - endog.sum()
    statistic = score / np.sqrt(var_score)
    pvalue_two = 2 * stats.norm.sf(np.abs(statistic))
    pvalue_upp = stats.norm.sf(statistic)
    pvalue_low = stats.norm.cdf(statistic)

    res = HolderTuple(
        statistic=statistic,
        pvalue=pvalue_two,
        pvalue_smaller=pvalue_upp,
        pvalue_larger=pvalue_low,
        chi2=statistic**2,
        pvalue_chi2=stats.chi2.sf(statistic**2, 1),
        df_chi2=1,
        distribution="normal",
        )
    return res


def test_poisson_zeros(results):
    """Test for excess zeros in Poisson regression model.

    The test is implemented following Tang and Tang [1]_ equ. (12) which is
    based on the test derived in He et al 2019 [2]_.

    References
    ----------

    .. [1] Tang, Yi, and Wan Tang. 2018. “Testing Modified Zeros for Poisson
           Regression Models:” Statistical Methods in Medical Research,
           September. https://doi.org/10.1177/0962280218796253.

    .. [2] He, Hua, Hui Zhang, Peng Ye, and Wan Tang. 2019. “A Test of Inflated
           Zeros for Poisson Regression Models.” Statistical Methods in
           Medical Research 28 (4): 1157–69.
           https://doi.org/10.1177/0962280217749991.

    """
    x = results.model.exog
    mean = results.predict()
    prob0 = np.exp(-mean)
    counts = (results.model.endog == 0).astype(int)
    diff = counts.sum() - prob0.sum()
    var1 = prob0 @ (1 - prob0)
    pm = prob0 * mean
    c = np.linalg.inv(x.T * mean @ x)
    pmx = pm @ x
    var2 = pmx @ c @ pmx
    var = var1 - var2
    statistic = diff / np.sqrt(var)

    pvalue_two = 2 * stats.norm.sf(np.abs(statistic))
    pvalue_upp = stats.norm.sf(statistic)
    pvalue_low = stats.norm.cdf(statistic)

    res = HolderTuple(
        statistic=statistic,
        pvalue=pvalue_two,
        pvalue_smaller=pvalue_upp,
        pvalue_larger=pvalue_low,
        chi2=statistic**2,
        pvalue_chi2=stats.chi2.sf(statistic**2, 1),
        df_chi2=1,
        distribution="normal",
        )
    return res
