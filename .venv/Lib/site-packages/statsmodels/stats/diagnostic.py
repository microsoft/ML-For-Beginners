# -*- coding: utf-8 -*-
"""
Various Statistical Tests

Author: josef-pktd
License: BSD-3

Notes
-----
Almost fully verified against R or Gretl, not all options are the same.
In many cases of Lagrange multiplier tests both the LM test and the F test is
returned. In some but not all cases, R has the option to choose the test
statistic. Some alternative test statistic results have not been verified.

TODO
* refactor to store intermediate results

missing:

* pvalues for breaks_hansen
* additional options, compare with R, check where ddof is appropriate
* new tests:
  - breaks_ap, more recent breaks tests
  - specification tests against nonparametric alternatives
"""
from statsmodels.compat.pandas import deprecate_kwarg

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from statsmodels.regression.linear_model import OLS, RegressionResultsWrapper
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tools.validation import (array_like, int_like, bool_like,
                                          string_like, dict_like, float_like)
from statsmodels.stats._lilliefors import (kstest_fit, lilliefors,
                                           kstest_normal, kstest_exponential)
from statsmodels.stats._adnorm import normal_ad, anderson_statistic

__all__ = ["kstest_fit", "lilliefors", "kstest_normal", "kstest_exponential",
           "normal_ad", "compare_cox", "compare_j", "acorr_breusch_godfrey",
           "acorr_ljungbox", "acorr_lm", "het_arch", "het_breuschpagan",
           "het_goldfeldquandt", "het_white", "spec_white", "linear_lm",
           "linear_rainbow", "linear_harvey_collier", "anderson_statistic"]


NESTED_ERROR = """\
The exog in results_x and in results_z are nested. {test} requires \
that models are non-nested.
"""


def _check_nested_exog(small, large):
    """
    Check if a larger exog nests a smaller exog

    Parameters
    ----------
    small : ndarray
        exog from smaller model
    large : ndarray
        exog from larger model

    Returns
    -------
    bool
        True if small is nested by large
    """

    if small.shape[1] > large.shape[1]:
        return False
    coef = np.linalg.lstsq(large, small, rcond=None)[0]
    err = small - large @ coef
    return np.linalg.matrix_rank(np.c_[large, err]) == large.shape[1]


def _check_nested_results(results_x, results_z):
    if not isinstance(results_x, RegressionResultsWrapper):
        raise TypeError("results_x must come from a linear regression model")
    if not isinstance(results_z, RegressionResultsWrapper):
        raise TypeError("results_z must come from a linear regression model")
    if not np.allclose(results_x.model.endog, results_z.model.endog):
        raise ValueError("endogenous variables in models are not the same")

    x = results_x.model.exog
    z = results_z.model.exog

    nested = False
    if x.shape[1] <= z.shape[1]:
        nested = nested or _check_nested_exog(x, z)
    else:
        nested = nested or _check_nested_exog(z, x)
    return nested


class ResultsStore:
    def __str__(self):
        return getattr(self, '_str', self.__class__.__name__)


def compare_cox(results_x, results_z, store=False):
    """
    Compute the Cox test for non-nested models

    Parameters
    ----------
    results_x : Result instance
        result instance of first model
    results_z : Result instance
        result instance of second model
    store : bool, default False
        If true, then the intermediate results are returned.

    Returns
    -------
    tstat : float
        t statistic for the test that including the fitted values of the
        first model in the second model has no effect.
    pvalue : float
        two-sided pvalue for the t statistic
    res_store : ResultsStore, optional
        Intermediate results. Returned if store is True.

    Notes
    -----
    Tests of non-nested hypothesis might not provide unambiguous answers.
    The test should be performed in both directions and it is possible
    that both or neither test rejects. see [1]_ for more information.

    Formulas from [1]_, section 8.3.4 translated to code

    Matches results for Example 8.3 in Greene

    References
    ----------
    .. [1] Greene, W. H. Econometric Analysis. New Jersey. Prentice Hall;
       5th edition. (2002).
    """
    if _check_nested_results(results_x, results_z):
        raise ValueError(NESTED_ERROR.format(test="Cox comparison"))
    x = results_x.model.exog
    z = results_z.model.exog
    nobs = results_x.model.endog.shape[0]
    sigma2_x = results_x.ssr / nobs
    sigma2_z = results_z.ssr / nobs
    yhat_x = results_x.fittedvalues
    res_dx = OLS(yhat_x, z).fit()
    err_zx = res_dx.resid
    res_xzx = OLS(err_zx, x).fit()
    err_xzx = res_xzx.resid

    sigma2_zx = sigma2_x + np.dot(err_zx.T, err_zx) / nobs
    c01 = nobs / 2. * (np.log(sigma2_z) - np.log(sigma2_zx))
    v01 = sigma2_x * np.dot(err_xzx.T, err_xzx) / sigma2_zx ** 2
    q = c01 / np.sqrt(v01)
    pval = 2 * stats.norm.sf(np.abs(q))

    if store:
        res = ResultsStore()
        res.res_dx = res_dx
        res.res_xzx = res_xzx
        res.c01 = c01
        res.v01 = v01
        res.q = q
        res.pvalue = pval
        res.dist = stats.norm
        return q, pval, res

    return q, pval


def compare_j(results_x, results_z, store=False):
    """
    Compute the J-test for non-nested models

    Parameters
    ----------
    results_x : RegressionResults
        The result instance of first model.
    results_z : RegressionResults
        The result instance of second model.
    store : bool, default False
        If true, then the intermediate results are returned.

    Returns
    -------
    tstat : float
        t statistic for the test that including the fitted values of the
        first model in the second model has no effect.
    pvalue : float
        two-sided pvalue for the t statistic
    res_store : ResultsStore, optional
        Intermediate results. Returned if store is True.

    Notes
    -----
    From description in Greene, section 8.3.3. Matches results for Example
    8.3, Greene.

    Tests of non-nested hypothesis might not provide unambiguous answers.
    The test should be performed in both directions and it is possible
    that both or neither test rejects. see Greene for more information.

    References
    ----------
    .. [1] Greene, W. H. Econometric Analysis. New Jersey. Prentice Hall;
       5th edition. (2002).
    """
    # TODO: Allow cov to be specified
    if _check_nested_results(results_x, results_z):
        raise ValueError(NESTED_ERROR.format(test="J comparison"))
    y = results_x.model.endog
    z = results_z.model.exog
    yhat_x = results_x.fittedvalues
    res_zx = OLS(y, np.column_stack((yhat_x, z))).fit()
    tstat = res_zx.tvalues[0]
    pval = res_zx.pvalues[0]
    if store:
        res = ResultsStore()
        res.res_zx = res_zx
        res.dist = stats.t(res_zx.df_resid)
        res.teststat = tstat
        res.pvalue = pval
        return tstat, pval, res

    return tstat, pval


def compare_encompassing(results_x, results_z, cov_type="nonrobust",
                         cov_kwargs=None):
    r"""
    Davidson-MacKinnon encompassing test for comparing non-nested models

    Parameters
    ----------
    results_x : Result instance
        result instance of first model
    results_z : Result instance
        result instance of second model
    cov_type : str, default "nonrobust
        Covariance type. The default is "nonrobust` which uses the classic
        OLS covariance estimator. Specify one of "HC0", "HC1", "HC2", "HC3"
        to use White's covariance estimator. All covariance types supported
        by ``OLS.fit`` are accepted.
    cov_kwargs : dict, default None
        Dictionary of covariance options passed to ``OLS.fit``. See OLS.fit
        for more details.

    Returns
    -------
    DataFrame
        A DataFrame with two rows and four columns. The row labeled x
        contains results for the null that the model contained in
        results_x is equivalent to the encompassing model. The results in
        the row labeled z correspond to the test that the model contained
        in results_z are equivalent to the encompassing model. The columns
        are the test statistic, its p-value, and the numerator and
        denominator degrees of freedom. The test statistic has an F
        distribution. The numerator degree of freedom is the number of
        variables in the encompassing model that are not in the x or z model.
        The denominator degree of freedom is the number of observations minus
        the number of variables in the nesting model.

    Notes
    -----
    The null is that the fit produced using x is the same as the fit
    produced using both x and z. When testing whether x is encompassed,
    the model estimated is

    .. math::

        Y = X\beta + Z_1\gamma + \epsilon

    where :math:`Z_1` are the columns of :math:`Z` that are not spanned by
    :math:`X`. The null is :math:`H_0:\gamma=0`. When testing whether z is
    encompassed, the roles of :math:`X` and :math:`Z` are reversed.

    Implementation of  Davidson and MacKinnon (1993)'s encompassing test.
    Performs two Wald tests where models x and z are compared to a model
    that nests the two. The Wald tests are performed by using an OLS
    regression.
    """
    if _check_nested_results(results_x, results_z):
        raise ValueError(NESTED_ERROR.format(test="Testing encompassing"))

    y = results_x.model.endog
    x = results_x.model.exog
    z = results_z.model.exog

    def _test_nested(endog, a, b, cov_est, cov_kwds):
        err = b - a @ np.linalg.lstsq(a, b, rcond=None)[0]
        u, s, v = np.linalg.svd(err)
        eps = np.finfo(np.double).eps
        tol = s.max(axis=-1, keepdims=True) * max(err.shape) * eps
        non_zero = np.abs(s) > tol
        aug = err @ v[:, non_zero]
        aug_reg = np.hstack([a, aug])
        k_a = aug.shape[1]
        k = aug_reg.shape[1]

        res = OLS(endog, aug_reg).fit(cov_type=cov_est, cov_kwds=cov_kwds)
        r_matrix = np.zeros((k_a, k))
        r_matrix[:, -k_a:] = np.eye(k_a)
        test = res.wald_test(r_matrix, use_f=True, scalar=True)
        stat, pvalue = test.statistic, test.pvalue
        df_num, df_denom = int(test.df_num), int(test.df_denom)
        return stat, pvalue, df_num, df_denom

    x_nested = _test_nested(y, x, z, cov_type, cov_kwargs)
    z_nested = _test_nested(y, z, x, cov_type, cov_kwargs)
    return pd.DataFrame([x_nested, z_nested],
                        index=["x", "z"],
                        columns=["stat", "pvalue", "df_num", "df_denom"])


def acorr_ljungbox(x, lags=None, boxpierce=False, model_df=0, period=None,
                   return_df=True, auto_lag=False):
    """
    Ljung-Box test of autocorrelation in residuals.

    Parameters
    ----------
    x : array_like
        The data series. The data is demeaned before the test statistic is
        computed.
    lags : {int, array_like}, default None
        If lags is an integer then this is taken to be the largest lag
        that is included, the test result is reported for all smaller lag
        length. If lags is a list or array, then all lags are included up to
        the largest lag in the list, however only the tests for the lags in
        the list are reported. If lags is None, then the default maxlag is
        min(10, nobs // 5). The default number of lags changes if period
        is set.
    boxpierce : bool, default False
        If true, then additional to the results of the Ljung-Box test also the
        Box-Pierce test results are returned.
    model_df : int, default 0
        Number of degrees of freedom consumed by the model. In an ARMA model,
        this value is usually p+q where p is the AR order and q is the MA
        order. This value is subtracted from the degrees-of-freedom used in
        the test so that the adjusted dof for the statistics are
        lags - model_df. If lags - model_df <= 0, then NaN is returned.
    period : int, default None
        The period of a Seasonal time series.  Used to compute the max lag
        for seasonal data which uses min(2*period, nobs // 5) if set. If None,
        then the default rule is used to set the number of lags. When set, must
        be >= 2.
    auto_lag : bool, default False
        Flag indicating whether to automatically determine the optimal lag
        length based on threshold of maximum correlation value.

    Returns
    -------
    DataFrame
        Frame with columns:

        * lb_stat - The Ljung-Box test statistic.
        * lb_pvalue - The p-value based on chi-square distribution. The
          p-value is computed as 1 - chi2.cdf(lb_stat, dof) where dof is
          lag - model_df. If lag - model_df <= 0, then NaN is returned for
          the pvalue.
        * bp_stat - The Box-Pierce test statistic.
        * bp_pvalue - The p-value based for Box-Pierce test on chi-square
          distribution. The p-value is computed as 1 - chi2.cdf(bp_stat, dof)
          where dof is lag - model_df. If lag - model_df <= 0, then NaN is
          returned for the pvalue.

    See Also
    --------
    statsmodels.regression.linear_model.OLS.fit
        Regression model fitting.
    statsmodels.regression.linear_model.RegressionResults
        Results from linear regression models.
    statsmodels.stats.stattools.q_stat
        Ljung-Box test statistic computed from estimated
        autocorrelations.

    Notes
    -----
    Ljung-Box and Box-Pierce statistic differ in their scaling of the
    autocorrelation function. Ljung-Box test is has better finite-sample
    properties.

    References
    ----------
    .. [*] Green, W. "Econometric Analysis," 5th ed., Pearson, 2003.
    .. [*] J. Carlos Escanciano, Ignacio N. Lobato
          "An automatic Portmanteau test for serial correlation".,
          Volume 151, 2009.

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> data = sm.datasets.sunspots.load_pandas().data
    >>> res = sm.tsa.ARMA(data["SUNACTIVITY"], (1,1)).fit(disp=-1)
    >>> sm.stats.acorr_ljungbox(res.resid, lags=[10], return_df=True)
           lb_stat     lb_pvalue
    10  214.106992  1.827374e-40
    """
    # Avoid cyclic import
    from statsmodels.tsa.stattools import acf
    x = array_like(x, "x")
    period = int_like(period, "period", optional=True)
    model_df = int_like(model_df, "model_df", optional=False)
    if period is not None and period <= 1:
        raise ValueError("period must be >= 2")
    if model_df < 0:
        raise ValueError("model_df must be >= 0")
    nobs = x.shape[0]
    if auto_lag:
        maxlag = nobs - 1

        # Compute sum of squared autocorrelations
        sacf = acf(x, nlags=maxlag, fft=False)

        if not boxpierce:
            q_sacf = (nobs * (nobs + 2) *
                      np.cumsum(sacf[1:maxlag + 1] ** 2
                                / (nobs - np.arange(1, maxlag + 1))))
        else:
            q_sacf = nobs * np.cumsum(sacf[1:maxlag + 1] ** 2)

        # obtain thresholds
        q = 2.4
        threshold = np.sqrt(q * np.log(nobs))
        threshold_metric = np.abs(sacf).max() * np.sqrt(nobs)

        # compute penalized sum of squared autocorrelations
        if (threshold_metric <= threshold):
            q_sacf = q_sacf - (np.arange(1, nobs) * np.log(nobs))
        else:
            q_sacf = q_sacf - (2 * np.arange(1, nobs))

        # note: np.argmax returns first (i.e., smallest) index of largest value
        lags = np.argmax(q_sacf)
        lags = max(1, lags)  # optimal lag has to be at least 1
        lags = int_like(lags, "lags")
        lags = np.arange(1, lags + 1)
    elif period is not None:
        lags = np.arange(1, min(nobs // 5, 2 * period) + 1, dtype=int)
    elif lags is None:
        lags = np.arange(1, min(nobs // 5, 10) + 1, dtype=int)
    elif not isinstance(lags, Iterable):
        lags = int_like(lags, "lags")
        lags = np.arange(1, lags + 1)
    lags = array_like(lags, "lags", dtype="int")
    maxlag = lags.max()

    # normalize by nobs not (nobs-nlags)
    # SS: unbiased=False is default now
    sacf = acf(x, nlags=maxlag, fft=False)
    sacf2 = sacf[1:maxlag + 1] ** 2 / (nobs - np.arange(1, maxlag + 1))
    qljungbox = nobs * (nobs + 2) * np.cumsum(sacf2)[lags - 1]
    adj_lags = lags - model_df
    pval = np.full_like(qljungbox, np.nan)
    loc = adj_lags > 0
    pval[loc] = stats.chi2.sf(qljungbox[loc], adj_lags[loc])

    if not boxpierce:
        return pd.DataFrame({"lb_stat": qljungbox, "lb_pvalue": pval},
                            index=lags)

    qboxpierce = nobs * np.cumsum(sacf[1:maxlag + 1] ** 2)[lags - 1]
    pvalbp = np.full_like(qljungbox, np.nan)
    pvalbp[loc] = stats.chi2.sf(qboxpierce[loc], adj_lags[loc])
    return pd.DataFrame({"lb_stat": qljungbox, "lb_pvalue": pval,
                         "bp_stat": qboxpierce, "bp_pvalue": pvalbp},
                        index=lags)


@deprecate_kwarg("maxlag", "nlags")
def acorr_lm(resid, nlags=None, store=False, *, period=None,
             ddof=0, cov_type="nonrobust", cov_kwargs=None):
    """
    Lagrange Multiplier tests for autocorrelation.

    This is a generic Lagrange Multiplier test for autocorrelation. Returns
    Engle's ARCH test if resid is the squared residual array. Breusch-Godfrey
    is a variation on this test with additional exogenous variables.

    Parameters
    ----------
    resid : array_like
        Time series to test.
    nlags : int, default None
        Highest lag to use.
    store : bool, default False
        If true then the intermediate results are also returned.
    period : int, default none
        The period of a Seasonal time series.  Used to compute the max lag
        for seasonal data which uses min(2*period, nobs // 5) if set. If None,
        then the default rule is used to set the number of lags. When set, must
        be >= 2.
    ddof : int, default 0
        The number of degrees of freedom consumed by the model used to
        produce resid. The default value is 0.
    cov_type : str, default "nonrobust"
        Covariance type. The default is "nonrobust` which uses the classic
        OLS covariance estimator. Specify one of "HC0", "HC1", "HC2", "HC3"
        to use White's covariance estimator. All covariance types supported
        by ``OLS.fit`` are accepted.
    cov_kwargs : dict, default None
        Dictionary of covariance options passed to ``OLS.fit``. See OLS.fit for
        more details.

    Returns
    -------
    lm : float
        Lagrange multiplier test statistic.
    lmpval : float
        The p-value for Lagrange multiplier test.
    fval : float
        The f statistic of the F test, alternative version of the same
        test based on F test for the parameter restriction.
    fpval : float
        The pvalue of the F test.
    res_store : ResultsStore, optional
        Intermediate results. Only returned if store=True.

    See Also
    --------
    het_arch
        Conditional heteroskedasticity testing.
    acorr_breusch_godfrey
        Breusch-Godfrey test for serial correlation.
    acorr_ljung_box
        Ljung-Box test for serial correlation.

    Notes
    -----
    The test statistic is computed as (nobs - ddof) * r2 where r2 is the
    R-squared from a regression on the residual on nlags lags of the
    residual.
    """
    resid = array_like(resid, "resid", ndim=1)
    cov_type = string_like(cov_type, "cov_type")
    cov_kwargs = {} if cov_kwargs is None else cov_kwargs
    cov_kwargs = dict_like(cov_kwargs, "cov_kwargs")
    nobs = resid.shape[0]
    if period is not None and nlags is None:
        maxlag = min(nobs // 5, 2 * period)
    elif nlags is None:
        maxlag = min(10, nobs // 5)
    else:
        maxlag = nlags

    xdall = lagmat(resid[:, None], maxlag, trim="both")
    nobs = xdall.shape[0]
    xdall = np.c_[np.ones((nobs, 1)), xdall]
    xshort = resid[-nobs:]
    res_store = ResultsStore()
    usedlag = maxlag

    resols = OLS(xshort, xdall[:, :usedlag + 1]).fit(cov_type=cov_type,
                                                     cov_kwargs=cov_kwargs)
    fval = float(resols.fvalue)
    fpval = float(resols.f_pvalue)
    if cov_type == "nonrobust":
        lm = (nobs - ddof) * resols.rsquared
        lmpval = stats.chi2.sf(lm, usedlag)
        # Note: deg of freedom for LM test: nvars - constant = lags used
    else:
        r_matrix = np.hstack((np.zeros((usedlag, 1)), np.eye(usedlag)))
        test_stat = resols.wald_test(r_matrix, use_f=False, scalar=True)
        lm = float(test_stat.statistic)
        lmpval = float(test_stat.pvalue)

    if store:
        res_store.resols = resols
        res_store.usedlag = usedlag
        return lm, lmpval, fval, fpval, res_store
    else:
        return lm, lmpval, fval, fpval


@deprecate_kwarg("maxlag", "nlags")
def het_arch(resid, nlags=None, store=False, ddof=0):
    """
    Engle's Test for Autoregressive Conditional Heteroscedasticity (ARCH).

    Parameters
    ----------
    resid : ndarray
        residuals from an estimation, or time series
    nlags : int, default None
        Highest lag to use.
    store : bool, default False
        If true then the intermediate results are also returned
    ddof : int, default 0
        If the residuals are from a regression, or ARMA estimation, then there
        are recommendations to correct the degrees of freedom by the number
        of parameters that have been estimated, for example ddof=p+q for an
        ARMA(p,q).

    Returns
    -------
    lm : float
        Lagrange multiplier test statistic
    lmpval : float
        p-value for Lagrange multiplier test
    fval : float
        fstatistic for F test, alternative version of the same test based on
        F test for the parameter restriction
    fpval : float
        pvalue for F test
    res_store : ResultsStore, optional
        Intermediate results. Returned if store is True.

    Notes
    -----
    verified against R:FinTS::ArchTest
    """
    return acorr_lm(resid ** 2, nlags=nlags, store=store, ddof=ddof)


@deprecate_kwarg("results", "res")
def acorr_breusch_godfrey(res, nlags=None, store=False):
    """
    Breusch-Godfrey Lagrange Multiplier tests for residual autocorrelation.

    Parameters
    ----------
    res : RegressionResults
        Estimation results for which the residuals are tested for serial
        correlation.
    nlags : int, optional
        Number of lags to include in the auxiliary regression. (nlags is
        highest lag).
    store : bool, default False
        If store is true, then an additional class instance that contains
        intermediate results is returned.

    Returns
    -------
    lm : float
        Lagrange multiplier test statistic.
    lmpval : float
        The p-value for Lagrange multiplier test.
    fval : float
        The value of the f statistic for F test, alternative version of the
        same test based on F test for the parameter restriction.
    fpval : float
        The pvalue for F test.
    res_store : ResultsStore
        A class instance that holds intermediate results. Only returned if
        store=True.

    Notes
    -----
    BG adds lags of residual to exog in the design matrix for the auxiliary
    regression with residuals as endog. See [1]_, section 12.7.1.

    References
    ----------
    .. [1] Greene, W. H. Econometric Analysis. New Jersey. Prentice Hall;
      5th edition. (2002).
    """

    x = np.asarray(res.resid).squeeze()
    if x.ndim != 1:
        raise ValueError("Model resid must be a 1d array. Cannot be used on"
                         " multivariate models.")
    exog_old = res.model.exog
    nobs = x.shape[0]
    if nlags is None:
        nlags = min(10, nobs // 5)

    x = np.concatenate((np.zeros(nlags), x))

    xdall = lagmat(x[:, None], nlags, trim="both")
    nobs = xdall.shape[0]
    xdall = np.c_[np.ones((nobs, 1)), xdall]
    xshort = x[-nobs:]
    if exog_old is None:
        exog = xdall
    else:
        exog = np.column_stack((exog_old, xdall))
    k_vars = exog.shape[1]

    resols = OLS(xshort, exog).fit()
    ft = resols.f_test(np.eye(nlags, k_vars, k_vars - nlags))
    fval = ft.fvalue
    fpval = ft.pvalue
    fval = float(np.squeeze(fval))
    fpval = float(np.squeeze(fpval))
    lm = nobs * resols.rsquared
    lmpval = stats.chi2.sf(lm, nlags)
    # Note: degrees of freedom for LM test is nvars minus constant = usedlags

    if store:
        res_store = ResultsStore()
        res_store.resols = resols
        res_store.usedlag = nlags
        return lm, lmpval, fval, fpval, res_store
    else:
        return lm, lmpval, fval, fpval


def _check_het_test(x: np.ndarray, test_name: str) -> None:
    """
    Check validity of the exogenous regressors in a heteroskedasticity test

    Parameters
    ----------
    x : ndarray
        The exogenous regressor array
    test_name : str
        The test name for the exception
    """
    x_max = x.max(axis=0)
    if (
        not np.any(((x_max - x.min(axis=0)) == 0) & (x_max != 0))
        or x.shape[1] < 2
    ):
        raise ValueError(
            f"{test_name} test requires exog to have at least "
            "two columns where one is a constant."
        )


def het_breuschpagan(resid, exog_het, robust=True):
    r"""
    Breusch-Pagan Lagrange Multiplier test for heteroscedasticity

    The tests the hypothesis that the residual variance does not depend on
    the variables in x in the form

    .. :math: \sigma_i = \sigma * f(\alpha_0 + \alpha z_i)

    Homoscedasticity implies that :math:`\alpha=0`.

    Parameters
    ----------
    resid : array_like
        For the Breusch-Pagan test, this should be the residual of a
        regression. If an array is given in exog, then the residuals are
        calculated by the an OLS regression or resid on exog. In this case
        resid should contain the dependent variable. Exog can be the same as x.
    exog_het : array_like
        This contains variables suspected of being related to
        heteroscedasticity in resid.
    robust : bool, default True
        Flag indicating whether to use the Koenker version of the
        test (default) which assumes independent and identically distributed
        error terms, or the original Breusch-Pagan version which assumes
        residuals are normally distributed.

    Returns
    -------
    lm : float
        lagrange multiplier statistic
    lm_pvalue : float
        p-value of lagrange multiplier test
    fvalue : float
        f-statistic of the hypothesis that the error variance does not depend
        on x
    f_pvalue : float
        p-value for the f-statistic

    Notes
    -----
    Assumes x contains constant (for counting dof and calculation of R^2).
    In the general description of LM test, Greene mentions that this test
    exaggerates the significance of results in small or moderately large
    samples. In this case the F-statistic is preferable.

    **Verification**

    Chisquare test statistic is exactly (<1e-13) the same result as bptest
    in R-stats with defaults (studentize=True).

    **Implementation**

    This is calculated using the generic formula for LM test using $R^2$
    (Greene, section 17.6) and not with the explicit formula
    (Greene, section 11.4.3), unless `robust` is set to False.
    The degrees of freedom for the p-value assume x is full rank.

    References
    ----------
    .. [1] Greene, W. H. Econometric Analysis. New Jersey. Prentice Hall;
       5th edition. (2002).
    .. [2]  Breusch, T. S.; Pagan, A. R. (1979). "A Simple Test for
       Heteroskedasticity and Random Coefficient Variation". Econometrica.
       47 (5): 1287–1294.
    .. [3] Koenker, R. (1981). "A note on studentizing a test for
       heteroskedasticity". Journal of Econometrics 17 (1): 107–112.
    """
    x = array_like(exog_het, "exog_het", ndim=2)
    _check_het_test(x, "The Breusch-Pagan")
    y = array_like(resid, "resid", ndim=1) ** 2
    if not robust:
        y = y / np.mean(y)
    nobs, nvars = x.shape
    resols = OLS(y, x).fit()
    fval = resols.fvalue
    fpval = resols.f_pvalue
    lm = nobs * resols.rsquared if robust else resols.ess / 2
    # Note: degrees of freedom for LM test is nvars minus constant
    return lm, stats.chi2.sf(lm, nvars - 1), fval, fpval


def het_white(resid, exog):
    """
    White's Lagrange Multiplier Test for Heteroscedasticity.

    Parameters
    ----------
    resid : array_like
        The residuals. The squared residuals are used as the endogenous
        variable.
    exog : array_like
        The explanatory variables for the variance. Squares and interaction
        terms are automatically included in the auxiliary regression.

    Returns
    -------
    lm : float
        The lagrange multiplier statistic.
    lm_pvalue :float
        The p-value of lagrange multiplier test.
    fvalue : float
        The f-statistic of the hypothesis that the error variance does not
        depend on x. This is an alternative test variant not the original
        LM test.
    f_pvalue : float
        The p-value for the f-statistic.

    Notes
    -----
    Assumes x contains constant (for counting dof).

    question: does f-statistic make sense? constant ?

    References
    ----------
    Greene section 11.4.1 5th edition p. 222. Test statistic reproduces
    Greene 5th, example 11.3.
    """
    x = array_like(exog, "exog", ndim=2)
    y = array_like(resid, "resid", ndim=2, shape=(x.shape[0], 1))
    _check_het_test(x, "White's heteroskedasticity")
    nobs, nvars0 = x.shape
    i0, i1 = np.triu_indices(nvars0)
    exog = x[:, i0] * x[:, i1]
    nobs, nvars = exog.shape
    assert nvars == nvars0 * (nvars0 - 1) / 2. + nvars0
    resols = OLS(y ** 2, exog).fit()
    fval = resols.fvalue
    fpval = resols.f_pvalue
    lm = nobs * resols.rsquared
    # Note: degrees of freedom for LM test is nvars minus constant
    # degrees of freedom take possible reduced rank in exog into account
    # df_model checks the rank to determine df
    # extra calculation that can be removed:
    assert resols.df_model == np.linalg.matrix_rank(exog) - 1
    lmpval = stats.chi2.sf(lm, resols.df_model)
    return lm, lmpval, fval, fpval


def het_goldfeldquandt(y, x, idx=None, split=None, drop=None,
                       alternative="increasing", store=False):
    """
    Goldfeld-Quandt homoskedasticity test.

    This test examines whether the residual variance is the same in 2
    subsamples.

    Parameters
    ----------
    y : array_like
        endogenous variable
    x : array_like
        exogenous variable, regressors
    idx : int, default None
        column index of variable according to which observations are
        sorted for the split
    split : {int, float}, default None
        If an integer, this is the index at which sample is split.
        If a float in 0<split<1 then split is interpreted as fraction
        of the observations in the first sample. If None, uses nobs//2.
    drop : {int, float}, default None
        If this is not None, then observation are dropped from the middle
        part of the sorted series. If 0<split<1 then split is interpreted
        as fraction of the number of observations to be dropped.
        Note: Currently, observations are dropped between split and
        split+drop, where split and drop are the indices (given by rounding
        if specified as fraction). The first sample is [0:split], the
        second sample is [split+drop:]
    alternative : {"increasing", "decreasing", "two-sided"}
        The default is increasing. This specifies the alternative for the
        p-value calculation.
    store : bool, default False
        Flag indicating to return the regression results

    Returns
    -------
    fval : float
        value of the F-statistic
    pval : float
        p-value of the hypothesis that the variance in one subsample is
        larger than in the other subsample
    ordering : str
        The ordering used in the alternative.
    res_store : ResultsStore, optional
        Storage for the intermediate and final results that are calculated

    Notes
    -----
    The Null hypothesis is that the variance in the two sub-samples are the
    same. The alternative hypothesis, can be increasing, i.e. the variance
    in the second sample is larger than in the first, or decreasing or
    two-sided.

    Results are identical R, but the drop option is defined differently.
    (sorting by idx not tested yet)
    """
    x = np.asarray(x)
    y = np.asarray(y)  # **2
    nobs, nvars = x.shape
    if split is None:
        split = nobs // 2
    elif (0 < split) and (split < 1):
        split = int(nobs * split)

    if drop is None:
        start2 = split
    elif (0 < drop) and (drop < 1):
        start2 = split + int(nobs * drop)
    else:
        start2 = split + drop

    if idx is not None:
        xsortind = np.argsort(x[:, idx])
        y = y[xsortind]
        x = x[xsortind, :]

    resols1 = OLS(y[:split], x[:split]).fit()
    resols2 = OLS(y[start2:], x[start2:]).fit()
    fval = resols2.mse_resid / resols1.mse_resid
    # if fval>1:
    if alternative.lower() in ["i", "inc", "increasing"]:
        fpval = stats.f.sf(fval, resols1.df_resid, resols2.df_resid)
        ordering = "increasing"
    elif alternative.lower() in ["d", "dec", "decreasing"]:
        fpval = stats.f.sf(1. / fval, resols2.df_resid, resols1.df_resid)
        ordering = "decreasing"
    elif alternative.lower() in ["2", "2-sided", "two-sided"]:
        fpval_sm = stats.f.cdf(fval, resols2.df_resid, resols1.df_resid)
        fpval_la = stats.f.sf(fval, resols2.df_resid, resols1.df_resid)
        fpval = 2 * min(fpval_sm, fpval_la)
        ordering = "two-sided"
    else:
        raise ValueError("invalid alternative")

    if store:
        res = ResultsStore()
        res.__doc__ = "Test Results for Goldfeld-Quandt test of" \
                      "heterogeneity"
        res.fval = fval
        res.fpval = fpval
        res.df_fval = (resols2.df_resid, resols1.df_resid)
        res.resols1 = resols1
        res.resols2 = resols2
        res.ordering = ordering
        res.split = split
        res._str = """\
The Goldfeld-Quandt test for null hypothesis that the variance in the second
subsample is %s than in the first subsample:
F-statistic =%8.4f and p-value =%8.4f""" % (ordering, fval, fpval)

        return fval, fpval, ordering, res

    return fval, fpval, ordering


@deprecate_kwarg("result", "res")
def linear_reset(res, power=3, test_type="fitted", use_f=False,
                 cov_type="nonrobust", cov_kwargs=None):
    r"""
    Ramsey's RESET test for neglected nonlinearity

    Parameters
    ----------
    res : RegressionResults
        A results instance from a linear regression.
    power : {int, List[int]}, default 3
        The maximum power to include in the model, if an integer. Includes
        powers 2, 3, ..., power. If an list of integers, includes all powers
        in the list.
    test_type : str, default "fitted"
        The type of augmentation to use:

        * "fitted" : (default) Augment regressors with powers of fitted values.
        * "exog" : Augment exog with powers of exog. Excludes binary
          regressors.
        * "princomp": Augment exog with powers of first principal component of
          exog.
    use_f : bool, default False
        Flag indicating whether an F-test should be used (True) or a
        chi-square test (False).
    cov_type : str, default "nonrobust
        Covariance type. The default is "nonrobust` which uses the classic
        OLS covariance estimator. Specify one of "HC0", "HC1", "HC2", "HC3"
        to use White's covariance estimator. All covariance types supported
        by ``OLS.fit`` are accepted.
    cov_kwargs : dict, default None
        Dictionary of covariance options passed to ``OLS.fit``. See OLS.fit
        for more details.

    Returns
    -------
    ContrastResults
        Test results for Ramsey's Reset test. See notes for implementation
        details.

    Notes
    -----
    The RESET test uses an augmented regression of the form

    .. math::

       Y = X\beta + Z\gamma + \epsilon

    where :math:`Z` are a set of regressors that are one of:

    * Powers of :math:`X\hat{\beta}` from the original regression.
    * Powers of :math:`X`, excluding the constant and binary regressors.
    * Powers of the first principal component of :math:`X`. If the
      model includes a constant, this column is dropped before computing
      the principal component. In either case, the principal component
      is extracted from the correlation matrix of remaining columns.

    The test is a Wald test of the null :math:`H_0:\gamma=0`. If use_f
    is True, then the quadratic-form test statistic is divided by the
    number of restrictions and the F distribution is used to compute
    the critical value.
    """
    if not isinstance(res, RegressionResultsWrapper):
        raise TypeError("result must come from a linear regression model")
    if bool(res.model.k_constant) and res.model.exog.shape[1] == 1:
        raise ValueError("exog contains only a constant column. The RESET "
                         "test requires exog to have at least 1 "
                         "non-constant column.")
    test_type = string_like(test_type, "test_type",
                            options=("fitted", "exog", "princomp"))
    cov_kwargs = dict_like(cov_kwargs, "cov_kwargs", optional=True)
    use_f = bool_like(use_f, "use_f")
    if isinstance(power, int):
        if power < 2:
            raise ValueError("power must be >= 2")
        power = np.arange(2, power + 1, dtype=int)
    else:
        try:
            power = np.array(power, dtype=int)
        except Exception:
            raise ValueError("power must be an integer or list of integers")
        if power.ndim != 1 or len(set(power)) != power.shape[0] or \
                (power < 2).any():
            raise ValueError("power must contains distinct integers all >= 2")
    exog = res.model.exog
    if test_type == "fitted":
        aug = res.fittedvalues[:, None]
    elif test_type == "exog":
        # Remove constant and binary
        aug = res.model.exog
        binary = ((exog == exog.max(axis=0)) | (exog == exog.min(axis=0)))
        binary = binary.all(axis=0)
        if binary.all():
            raise ValueError("Model contains only constant or binary data")
        aug = aug[:, ~binary]
    else:
        from statsmodels.multivariate.pca import PCA
        aug = exog
        if res.k_constant:
            retain = np.arange(aug.shape[1]).tolist()
            retain.pop(int(res.model.data.const_idx))
            aug = aug[:, retain]
        pca = PCA(aug, ncomp=1, standardize=bool(res.k_constant),
                  demean=bool(res.k_constant), method="nipals")
        aug = pca.factors[:, :1]
    aug_exog = np.hstack([exog] + [aug ** p for p in power])
    mod_class = res.model.__class__
    mod = mod_class(res.model.data.endog, aug_exog)
    cov_kwargs = {} if cov_kwargs is None else cov_kwargs
    res = mod.fit(cov_type=cov_type, cov_kwargs=cov_kwargs)
    nrestr = aug_exog.shape[1] - exog.shape[1]
    nparams = aug_exog.shape[1]
    r_mat = np.eye(nrestr, nparams, k=nparams-nrestr)
    return res.wald_test(r_mat, use_f=use_f, scalar=True)


def linear_harvey_collier(res, order_by=None, skip=None):
    """
    Harvey Collier test for linearity

    The Null hypothesis is that the regression is correctly modeled as linear.

    Parameters
    ----------
    res : RegressionResults
        A results instance from a linear regression.
    order_by : array_like, default None
        Integer array specifying the order of the residuals. If not provided,
        the order of the residuals is not changed. If provided, must have
        the same number of observations as the endogenous variable.
    skip : int, default None
        The number of observations to use for initial OLS, if None then skip is
        set equal to the number of regressors (columns in exog).

    Returns
    -------
    tvalue : float
        The test statistic, based on ttest_1sample.
    pvalue : float
        The pvalue of the test.

    See Also
    --------
    statsmodels.stats.diadnostic.recursive_olsresiduals
        Recursive OLS residual calculation used in the test.

    Notes
    -----
    This test is a t-test that the mean of the recursive ols residuals is zero.
    Calculating the recursive residuals might take some time for large samples.
    """
    # I think this has different ddof than
    # B.H. Baltagi, Econometrics, 2011, chapter 8
    # but it matches Gretl and R:lmtest, pvalue at decimal=13
    rr = recursive_olsresiduals(res, skip=skip, alpha=0.95, order_by=order_by)

    return stats.ttest_1samp(rr[3][3:], 0)


def linear_rainbow(res, frac=0.5, order_by=None, use_distance=False,
                   center=None):
    """
    Rainbow test for linearity

    The null hypothesis is the fit of the model using full sample is the same
    as using a central subset. The alternative is that the fits are difference.
    The rainbow test has power against many different forms of nonlinearity.

    Parameters
    ----------
    res : RegressionResults
        A results instance from a linear regression.
    frac : float, default 0.5
        The fraction of the data to include in the center model.
    order_by : {ndarray, str, List[str]}, default None
        If an ndarray, the values in the array are used to sort the
        observations. If a string or a list of strings, these are interpreted
        as column name(s) which are then used to lexicographically sort the
        data.
    use_distance : bool, default False
        Flag indicating whether data should be ordered by the Mahalanobis
        distance to the center.
    center : {float, int}, default None
        If a float, the value must be in [0, 1] and the center is center *
        nobs of the ordered data.  If an integer, must be in [0, nobs) and
        is interpreted as the observation of the ordered data to use.

    Returns
    -------
    fstat : float
        The test statistic based on the F test.
    pvalue : float
        The pvalue of the test.

    Notes
    -----
    This test assumes residuals are homoskedastic and may reject a correct
    linear specification if the residuals are heteroskedastic.
    """
    if not isinstance(res, RegressionResultsWrapper):
        raise TypeError("res must be a results instance from a linear model.")
    frac = float_like(frac, "frac")

    use_distance = bool_like(use_distance, "use_distance")
    nobs = res.nobs
    endog = res.model.endog
    exog = res.model.exog
    if order_by is not None and use_distance:
        raise ValueError("order_by and use_distance cannot be simultaneously"
                         "used.")
    if order_by is not None:
        if isinstance(order_by, np.ndarray):
            order_by = array_like(order_by, "order_by", ndim=1, dtype="int")
        else:
            if isinstance(order_by, str):
                order_by = [order_by]
            try:
                cols = res.model.data.orig_exog[order_by].copy()
            except (IndexError, KeyError):
                raise TypeError("order_by must contain valid column names "
                                "from the exog data used to construct res,"
                                "and exog must be a pandas DataFrame.")
            name = "__index__"
            while name in cols:
                name += '_'
            cols[name] = np.arange(cols.shape[0])
            cols = cols.sort_values(order_by)
            order_by = np.asarray(cols[name])
        endog = endog[order_by]
        exog = exog[order_by]
    if use_distance:
        center = int(nobs) // 2 if center is None else center
        if isinstance(center, float):
            if not 0.0 <= center <= 1.0:
                raise ValueError("center must be in (0, 1) when a float.")
            center = int(center * (nobs-1))
        else:
            center = int_like(center, "center")
            if not 0 < center < nobs - 1:
                raise ValueError("center must be in [0, nobs) when an int.")
        center_obs = exog[center:center+1]
        from scipy.spatial.distance import cdist
        try:
            err = exog - center_obs
            vi = np.linalg.inv(err.T @ err / nobs)
        except np.linalg.LinAlgError:
            err = exog - exog.mean(0)
            vi = np.linalg.inv(err.T @ err / nobs)
        dist = cdist(exog, center_obs, metric='mahalanobis', VI=vi)
        idx = np.argsort(dist.ravel())
        endog = endog[idx]
        exog = exog[idx]

    lowidx = np.ceil(0.5 * (1 - frac) * nobs).astype(int)
    uppidx = np.floor(lowidx + frac * nobs).astype(int)
    if uppidx - lowidx < exog.shape[1]:
        raise ValueError("frac is too small to perform test. frac * nobs"
                         "must be greater than the number of exogenous"
                         "variables in the model.")
    mi_sl = slice(lowidx, uppidx)
    res_mi = OLS(endog[mi_sl], exog[mi_sl]).fit()
    nobs_mi = res_mi.model.endog.shape[0]
    ss_mi = res_mi.ssr
    ss = res.ssr
    fstat = (ss - ss_mi) / (nobs - nobs_mi) / ss_mi * res_mi.df_resid
    pval = stats.f.sf(fstat, nobs - nobs_mi, res_mi.df_resid)
    return fstat, pval


def linear_lm(resid, exog, func=None):
    """
    Lagrange multiplier test for linearity against functional alternative

    # TODO: Remove the restriction
    limitations: Assumes currently that the first column is integer.
    Currently it does not check whether the transformed variables contain NaNs,
    for example log of negative number.

    Parameters
    ----------
    resid : ndarray
        residuals of a regression
    exog : ndarray
        exogenous variables for which linearity is tested
    func : callable, default None
        If func is None, then squares are used. func needs to take an array
        of exog and return an array of transformed variables.

    Returns
    -------
    lm : float
       Lagrange multiplier test statistic
    lm_pval : float
       p-value of Lagrange multiplier tes
    ftest : ContrastResult instance
       the results from the F test variant of this test

    Notes
    -----
    Written to match Gretl's linearity test. The test runs an auxiliary
    regression of the residuals on the combined original and transformed
    regressors. The Null hypothesis is that the linear specification is
    correct.
    """
    if func is None:
        def func(x):
            return np.power(x, 2)

    exog_aux = np.column_stack((exog, func(exog[:, 1:])))

    nobs, k_vars = exog.shape
    ls = OLS(resid, exog_aux).fit()
    ftest = ls.f_test(np.eye(k_vars - 1, k_vars * 2 - 1, k_vars))
    lm = nobs * ls.rsquared
    lm_pval = stats.chi2.sf(lm, k_vars - 1)
    return lm, lm_pval, ftest


def spec_white(resid, exog):
    """
    White's Two-Moment Specification Test

    Parameters
    ----------
    resid : array_like
        OLS residuals.
    exog : array_like
        OLS design matrix.

    Returns
    -------
    stat : float
        The test statistic.
    pval : float
        A chi-square p-value for test statistic.
    dof : int
        The degrees of freedom.

    See Also
    --------
    het_white
        White's test for heteroskedasticity.

    Notes
    -----
    Implements the two-moment specification test described by White's
    Theorem 2 (1980, p. 823) which compares the standard OLS covariance
    estimator with White's heteroscedasticity-consistent estimator. The
    test statistic is shown to be chi-square distributed.

    Null hypothesis is homoscedastic and correctly specified.

    Assumes the OLS design matrix contains an intercept term and at least
    one variable. The intercept is removed to calculate the test statistic.

    Interaction terms (squares and crosses of OLS regressors) are added to
    the design matrix to calculate the test statistic.

    Degrees-of-freedom (full rank) = nvar + nvar * (nvar + 1) / 2

    Linearly dependent columns are removed to avoid singular matrix error.

    References
    ----------
    .. [*] White, H. (1980). A heteroskedasticity-consistent covariance matrix
       estimator and a direct test for heteroscedasticity. Econometrica, 48:
       817-838.
    """
    x = array_like(exog, "exog", ndim=2)
    e = array_like(resid, "resid", ndim=1)
    if x.shape[1] < 2 or not np.any(np.ptp(x, 0) == 0.0):
        raise ValueError("White's specification test requires at least two"
                         "columns where one is a constant.")

    # add interaction terms
    i0, i1 = np.triu_indices(x.shape[1])
    exog = np.delete(x[:, i0] * x[:, i1], 0, 1)

    # collinearity check - see _fit_collinear
    atol = 1e-14
    rtol = 1e-13
    tol = atol + rtol * exog.var(0)
    r = np.linalg.qr(exog, mode="r")
    mask = np.abs(r.diagonal()) < np.sqrt(tol)
    exog = exog[:, np.where(~mask)[0]]

    # calculate test statistic
    sqe = e * e
    sqmndevs = sqe - np.mean(sqe)
    d = np.dot(exog.T, sqmndevs)
    devx = exog - np.mean(exog, axis=0)
    devx *= sqmndevs[:, None]
    b = devx.T.dot(devx)
    stat = d.dot(np.linalg.solve(b, d))

    # chi-square test
    dof = devx.shape[1]
    pval = stats.chi2.sf(stat, dof)
    return stat, pval, dof


@deprecate_kwarg("olsresults", "res")
def recursive_olsresiduals(res, skip=None, lamda=0.0, alpha=0.95,
                           order_by=None):
    """
    Calculate recursive ols with residuals and Cusum test statistic

    Parameters
    ----------
    res : RegressionResults
        Results from estimation of a regression model.
    skip : int, default None
        The number of observations to use for initial OLS, if None then skip is
        set equal to the number of regressors (columns in exog).
    lamda : float, default 0.0
        The weight for Ridge correction to initial (X'X)^{-1}.
    alpha : {0.90, 0.95, 0.99}, default 0.95
        Confidence level of test, currently only two values supported,
        used for confidence interval in cusum graph.
    order_by : array_like, default None
        Integer array specifying the order of the residuals. If not provided,
        the order of the residuals is not changed. If provided, must have
        the same number of observations as the endogenous variable.

    Returns
    -------
    rresid : ndarray
        The recursive ols residuals.
    rparams : ndarray
        The recursive ols parameter estimates.
    rypred : ndarray
        The recursive prediction of endogenous variable.
    rresid_standardized : ndarray
        The recursive residuals standardized so that N(0,sigma2) distributed,
        where sigma2 is the error variance.
    rresid_scaled : ndarray
        The recursive residuals normalize so that N(0,1) distributed.
    rcusum : ndarray
        The cumulative residuals for cusum test.
    rcusumci : ndarray
        The confidence interval for cusum test using a size of alpha.

    Notes
    -----
    It produces same recursive residuals as other version. This version updates
    the inverse of the X'X matrix and does not require matrix inversion during
    updating. looks efficient but no timing

    Confidence interval in Greene and Brown, Durbin and Evans is the same as
    in Ploberger after a little bit of algebra.

    References
    ----------
    jplv to check formulas, follows Harvey
    BigJudge 5.5.2b for formula for inverse(X'X) updating
    Greene section 7.5.2

    Brown, R. L., J. Durbin, and J. M. Evans. “Techniques for Testing the
    Constancy of Regression Relationships over Time.”
    Journal of the Royal Statistical Society. Series B (Methodological) 37,
    no. 2 (1975): 149-192.
    """
    if not isinstance(res, RegressionResultsWrapper):
        raise TypeError("res a regression results instance")
    y = res.model.endog
    x = res.model.exog
    order_by = array_like(order_by, "order_by", dtype="int", optional=True,
                          ndim=1, shape=(y.shape[0],))
    # intialize with skip observations
    if order_by is not None:
        x = x[order_by]
        y = y[order_by]

    nobs, nvars = x.shape
    if skip is None:
        skip = nvars
    rparams = np.nan * np.zeros((nobs, nvars))
    rresid = np.nan * np.zeros(nobs)
    rypred = np.nan * np.zeros(nobs)
    rvarraw = np.nan * np.zeros(nobs)

    x0 = x[:skip]
    if np.linalg.matrix_rank(x0) < x0.shape[1]:
        err_msg = """\
"The initial regressor matrix, x[:skip], issingular. You must use a value of
skip large enough to ensure that the first OLS estimator is well-defined.
"""
        raise ValueError(err_msg)
    y0 = y[:skip]
    # add Ridge to start (not in jplv)
    xtxi = np.linalg.inv(np.dot(x0.T, x0) + lamda * np.eye(nvars))
    xty = np.dot(x0.T, y0)  # xi * y   #np.dot(xi, y)
    beta = np.dot(xtxi, xty)
    rparams[skip - 1] = beta
    yipred = np.dot(x[skip - 1], beta)
    rypred[skip - 1] = yipred
    rresid[skip - 1] = y[skip - 1] - yipred
    rvarraw[skip - 1] = 1 + np.dot(x[skip - 1], np.dot(xtxi, x[skip - 1]))
    for i in range(skip, nobs):
        xi = x[i:i + 1, :]
        yi = y[i]

        # get prediction error with previous beta
        yipred = np.dot(xi, beta)
        rypred[i] = np.squeeze(yipred)
        residi = yi - yipred
        rresid[i] = np.squeeze(residi)

        # update beta and inverse(X'X)
        tmp = np.dot(xtxi, xi.T)
        ft = 1 + np.dot(xi, tmp)

        xtxi = xtxi - np.dot(tmp, tmp.T) / ft  # BigJudge equ 5.5.15

        beta = beta + (tmp * residi / ft).ravel()  # BigJudge equ 5.5.14
        rparams[i] = beta
        rvarraw[i] = np.squeeze(ft)

    rresid_scaled = rresid / np.sqrt(rvarraw)  # N(0,sigma2) distributed
    nrr = nobs - skip
    # sigma2 = rresid_scaled[skip-1:].var(ddof=1)  #var or sum of squares ?
    # Greene has var, jplv and Ploberger have sum of squares (Ass.:mean=0)
    # Gretl uses: by reverse engineering matching their numbers
    sigma2 = rresid_scaled[skip:].var(ddof=1)
    rresid_standardized = rresid_scaled / np.sqrt(sigma2)  # N(0,1) distributed
    rcusum = rresid_standardized[skip - 1:].cumsum()
    # confidence interval points in Greene p136 looks strange. Cleared up
    # this assumes sum of independent standard normal, which does not take into
    # account that we make many tests at the same time
    if alpha == 0.90:
        a = 0.850
    elif alpha == 0.95:
        a = 0.948
    elif alpha == 0.99:
        a = 1.143
    else:
        raise ValueError("alpha can only be 0.9, 0.95 or 0.99")

    # following taken from Ploberger,
    # crit = a * np.sqrt(nrr)
    rcusumci = (a * np.sqrt(nrr) + 2 * a * np.arange(0, nobs - skip) / np.sqrt(
        nrr)) * np.array([[-1.], [+1.]])
    return (rresid, rparams, rypred, rresid_standardized, rresid_scaled,
            rcusum, rcusumci)


def breaks_hansen(olsresults):
    """
    Test for model stability, breaks in parameters for ols, Hansen 1992

    Parameters
    ----------
    olsresults : RegressionResults
        Results from estimation of a regression model.

    Returns
    -------
    teststat : float
        Hansen's test statistic.
    crit : ndarray
        The critical values at alpha=0.95 for different nvars.

    Notes
    -----
    looks good in example, maybe not very powerful for small changes in
    parameters

    According to Greene, distribution of test statistics depends on nvar but
    not on nobs.

    Test statistic is verified against R:strucchange

    References
    ----------
    Greene section 7.5.1, notation follows Greene
    """
    x = olsresults.model.exog
    resid = array_like(olsresults.resid, "resid", shape=(x.shape[0], 1))
    nobs, nvars = x.shape
    resid2 = resid ** 2
    ft = np.c_[x * resid[:, None], (resid2 - resid2.mean())]
    score = ft.cumsum(0)
    f = nobs * (ft[:, :, None] * ft[:, None, :]).sum(0)
    s = (score[:, :, None] * score[:, None, :]).sum(0)
    h = np.trace(np.dot(np.linalg.inv(f), s))
    crit95 = np.array([(2, 1.01), (6, 1.9), (15, 3.75), (19, 4.52)],
                      dtype=[("nobs", int), ("crit", float)])
    # TODO: get critical values from Bruce Hansen's 1992 paper
    return h, crit95


def breaks_cusumolsresid(resid, ddof=0):
    """
    Cusum test for parameter stability based on ols residuals.

    Parameters
    ----------
    resid : ndarray
        An array of residuals from an OLS estimation.
    ddof : int
        The number of parameters in the OLS estimation, used as degrees
        of freedom correction for error variance.

    Returns
    -------
    sup_b : float
        The test statistic, maximum of absolute value of scaled cumulative OLS
        residuals.
    pval : float
        Probability of observing the data under the null hypothesis of no
        structural change, based on asymptotic distribution which is a Brownian
        Bridge
    crit: list
        The tabulated critical values, for alpha = 1%, 5% and 10%.

    Notes
    -----
    Tested against R:structchange.

    Not clear: Assumption 2 in Ploberger, Kramer assumes that exog x have
    asymptotically zero mean, x.mean(0) = [1, 0, 0, ..., 0]
    Is this really necessary? I do not see how it can affect the test statistic
    under the null. It does make a difference under the alternative.
    Also, the asymptotic distribution of test statistic depends on this.

    From examples it looks like there is little power for standard cusum if
    exog (other than constant) have mean zero.

    References
    ----------
    Ploberger, Werner, and Walter Kramer. “The Cusum Test with OLS Residuals.”
    Econometrica 60, no. 2 (March 1992): 271-285.
    """
    resid = resid.ravel()
    nobs = len(resid)
    nobssigma2 = (resid ** 2).sum()
    if ddof > 0:
        nobssigma2 = nobssigma2 / (nobs - ddof) * nobs
    # b is asymptotically a Brownian Bridge
    b = resid.cumsum() / np.sqrt(nobssigma2)  # use T*sigma directly
    # asymptotically distributed as standard Brownian Bridge
    sup_b = np.abs(b).max()
    crit = [(1, 1.63), (5, 1.36), (10, 1.22)]
    # Note stats.kstwobign.isf(0.1) is distribution of sup.abs of Brownian
    # Bridge
    # >>> stats.kstwobign.isf([0.01,0.05,0.1])
    # array([ 1.62762361,  1.35809864,  1.22384787])
    pval = stats.kstwobign.sf(sup_b)
    return sup_b, pval, crit

# def breaks_cusum(recolsresid):
#    """renormalized cusum test for parameter stability based on recursive
#    residuals
#
#
#    still incorrect: in PK, the normalization for sigma is by T not T-K
#    also the test statistic is asymptotically a Wiener Process, Brownian
#    motion
#    not Brownian Bridge
#    for testing: result reject should be identical as in standard cusum
#    version
#
#    References
#    ----------
#    Ploberger, Werner, and Walter Kramer. “The Cusum Test with OLS Residuals.”
#    Econometrica 60, no. 2 (March 1992): 271-285.
#
#    """
#    resid = recolsresid.ravel()
#    nobssigma2 = (resid**2).sum()
#    #B is asymptotically a Brownian Bridge
#    B = resid.cumsum()/np.sqrt(nobssigma2) # use T*sigma directly
#    nobs = len(resid)
#    denom = 1. + 2. * np.arange(nobs)/(nobs-1.) #not sure about limits
#    sup_b = np.abs(B/denom).max()
#    #asymptotically distributed as standard Brownian Bridge
#    crit = [(1,1.63), (5, 1.36), (10, 1.22)]
#    #Note stats.kstwobign.isf(0.1) is distribution of sup.abs of Brownian
#    Bridge
#    #>>> stats.kstwobign.isf([0.01,0.05,0.1])
#    #array([ 1.62762361,  1.35809864,  1.22384787])
#    pval = stats.kstwobign.sf(sup_b)
#    return sup_b, pval, crit
