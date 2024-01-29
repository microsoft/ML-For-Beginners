# -*- coding: utf-8 -*-
"""
Vector Autoregression (VAR) processes

References
----------
Lütkepohl (2005) New Introduction to Multiple Time Series Analysis
"""
from __future__ import annotations

from statsmodels.compat.python import lrange

from collections import defaultdict
from io import StringIO

import numpy as np
import pandas as pd
import scipy.stats as stats

import statsmodels.base.wrapper as wrap
from statsmodels.iolib.table import SimpleTable
from statsmodels.tools.decorators import cache_readonly, deprecated_alias
from statsmodels.tools.linalg import logdet_symm
from statsmodels.tools.sm_exceptions import OutputWarning
from statsmodels.tools.validation import array_like
from statsmodels.tsa.base.tsa_model import (
    TimeSeriesModel,
    TimeSeriesResultsWrapper,
)
import statsmodels.tsa.tsatools as tsa
from statsmodels.tsa.tsatools import duplication_matrix, unvec, vec
from statsmodels.tsa.vector_ar import output, plotting, util
from statsmodels.tsa.vector_ar.hypothesis_test_results import (
    CausalityTestResults,
    NormalityTestResults,
    WhitenessTestResults,
)
from statsmodels.tsa.vector_ar.irf import IRAnalysis
from statsmodels.tsa.vector_ar.output import VARSummary

# -------------------------------------------------------------------------------
# VAR process routines


def ma_rep(coefs, maxn=10):
    r"""
    MA(\infty) representation of VAR(p) process

    Parameters
    ----------
    coefs : ndarray (p x k x k)
    maxn : int
        Number of MA matrices to compute

    Notes
    -----
    VAR(p) process as

    .. math:: y_t = A_1 y_{t-1} + \ldots + A_p y_{t-p} + u_t

    can be equivalently represented as

    .. math:: y_t = \mu + \sum_{i=0}^\infty \Phi_i u_{t-i}

    e.g. can recursively compute the \Phi_i matrices with \Phi_0 = I_k

    Returns
    -------
    phis : ndarray (maxn + 1 x k x k)
    """
    p, k, k = coefs.shape
    phis = np.zeros((maxn + 1, k, k))
    phis[0] = np.eye(k)

    # recursively compute Phi matrices
    for i in range(1, maxn + 1):
        for j in range(1, i + 1):
            if j > p:
                break

            phis[i] += np.dot(phis[i - j], coefs[j - 1])

    return phis


def is_stable(coefs, verbose=False):
    """
    Determine stability of VAR(p) system by examining the eigenvalues of the
    VAR(1) representation

    Parameters
    ----------
    coefs : ndarray (p x k x k)

    Returns
    -------
    is_stable : bool
    """
    A_var1 = util.comp_matrix(coefs)
    eigs = np.linalg.eigvals(A_var1)

    if verbose:
        print("Eigenvalues of VAR(1) rep")
        for val in np.abs(eigs):
            print(val)

    return (np.abs(eigs) <= 1).all()


def var_acf(coefs, sig_u, nlags=None):
    """
    Compute autocovariance function ACF_y(h) up to nlags of stable VAR(p)
    process

    Parameters
    ----------
    coefs : ndarray (p x k x k)
        Coefficient matrices A_i
    sig_u : ndarray (k x k)
        Covariance of white noise process u_t
    nlags : int, optional
        Defaults to order p of system

    Notes
    -----
    Ref: Lütkepohl p.28-29

    Returns
    -------
    acf : ndarray, (p, k, k)
    """
    p, k, _ = coefs.shape
    if nlags is None:
        nlags = p

    # p x k x k, ACF for lags 0, ..., p-1
    result = np.zeros((nlags + 1, k, k))
    result[:p] = _var_acf(coefs, sig_u)

    # yule-walker equations
    for h in range(p, nlags + 1):
        # compute ACF for lag=h
        # G(h) = A_1 G(h-1) + ... + A_p G(h-p)

        for j in range(p):
            result[h] += np.dot(coefs[j], result[h - j - 1])

    return result


def _var_acf(coefs, sig_u):
    """
    Compute autocovariance function ACF_y(h) for h=1,...,p

    Notes
    -----
    Lütkepohl (2005) p.29
    """
    p, k, k2 = coefs.shape
    assert k == k2

    A = util.comp_matrix(coefs)
    # construct VAR(1) noise covariance
    SigU = np.zeros((k * p, k * p))
    SigU[:k, :k] = sig_u

    # vec(ACF) = (I_(kp)^2 - kron(A, A))^-1 vec(Sigma_U)
    vecACF = np.linalg.solve(np.eye((k * p) ** 2) - np.kron(A, A), vec(SigU))

    acf = unvec(vecACF)
    acf = [acf[:k, k * i : k * (i + 1)] for i in range(p)]
    acf = np.array(acf)

    return acf


def forecast_cov(ma_coefs, sigma_u, steps):
    r"""
    Compute theoretical forecast error variance matrices

    Parameters
    ----------
    steps : int
        Number of steps ahead

    Notes
    -----
    .. math:: \mathrm{MSE}(h) = \sum_{i=0}^{h-1} \Phi \Sigma_u \Phi^T

    Returns
    -------
    forc_covs : ndarray (steps x neqs x neqs)
    """
    neqs = len(sigma_u)
    forc_covs = np.zeros((steps, neqs, neqs))

    prior = np.zeros((neqs, neqs))
    for h in range(steps):
        # Sigma(h) = Sigma(h-1) + Phi Sig_u Phi'
        phi = ma_coefs[h]
        var = phi @ sigma_u @ phi.T
        forc_covs[h] = prior = prior + var

    return forc_covs


mse = forecast_cov


def forecast(y, coefs, trend_coefs, steps, exog=None):
    """
    Produce linear minimum MSE forecast

    Parameters
    ----------
    y : ndarray (k_ar x neqs)
    coefs : ndarray (k_ar x neqs x neqs)
    trend_coefs : ndarray (1 x neqs) or (neqs)
    steps : int
    exog : ndarray (trend_coefs.shape[1] x neqs)

    Returns
    -------
    forecasts : ndarray (steps x neqs)

    Notes
    -----
    Lütkepohl p. 37
    """
    p = len(coefs)
    k = len(coefs[0])
    if y.shape[0] < p:
        raise ValueError(
            f"y must by have at least order ({p}) observations. "
            f"Got {y.shape[0]}."
        )
    # initial value
    forcs = np.zeros((steps, k))
    if exog is not None and trend_coefs is not None:
        forcs += np.dot(exog, trend_coefs)
    # to make existing code (with trend_coefs=intercept and without exog) work:
    elif exog is None and trend_coefs is not None:
        forcs += trend_coefs

    # h=0 forecast should be latest observation
    # forcs[0] = y[-1]

    # make indices easier to think about
    for h in range(1, steps + 1):
        # y_t(h) = intercept + sum_1^p A_i y_t_(h-i)
        f = forcs[h - 1]
        for i in range(1, p + 1):
            # slightly hackish
            if h - i <= 0:
                # e.g. when h=1, h-1 = 0, which is y[-1]
                prior_y = y[h - i - 1]
            else:
                # e.g. when h=2, h-1=1, which is forcs[0]
                prior_y = forcs[h - i - 1]

            # i=1 is coefs[0]
            f = f + np.dot(coefs[i - 1], prior_y)

        forcs[h - 1] = f

    return forcs


def _forecast_vars(steps, ma_coefs, sig_u):
    """_forecast_vars function used by VECMResults. Note that the definition
    of the local variable covs is the same as in VARProcess and as such it
    differs from the one in VARResults!

    Parameters
    ----------
    steps
    ma_coefs
    sig_u

    Returns
    -------
    """
    covs = mse(ma_coefs, sig_u, steps)
    # Take diagonal for each cov
    neqs = len(sig_u)
    inds = np.arange(neqs)
    return covs[:, inds, inds]


def forecast_interval(
    y, coefs, trend_coefs, sig_u, steps=5, alpha=0.05, exog=1
):
    assert 0 < alpha < 1
    q = util.norm_signif_level(alpha)

    point_forecast = forecast(y, coefs, trend_coefs, steps, exog)
    ma_coefs = ma_rep(coefs, steps)
    sigma = np.sqrt(_forecast_vars(steps, ma_coefs, sig_u))

    forc_lower = point_forecast - q * sigma
    forc_upper = point_forecast + q * sigma

    return point_forecast, forc_lower, forc_upper


def var_loglike(resid, omega, nobs):
    r"""
    Returns the value of the VAR(p) log-likelihood.

    Parameters
    ----------
    resid : ndarray (T x K)
    omega : ndarray
        Sigma hat matrix.  Each element i,j is the average product of the
        OLS residual for variable i and the OLS residual for variable j or
        np.dot(resid.T,resid)/nobs.  There should be no correction for the
        degrees of freedom.
    nobs : int

    Returns
    -------
    llf : float
        The value of the loglikelihood function for a VAR(p) model

    Notes
    -----
    The loglikelihood function for the VAR(p) is

    .. math::

        -\left(\frac{T}{2}\right)
        \left(\ln\left|\Omega\right|-K\ln\left(2\pi\right)-K\right)
    """
    logdet = logdet_symm(np.asarray(omega))
    neqs = len(omega)
    part1 = -(nobs * neqs / 2) * np.log(2 * np.pi)
    part2 = -(nobs / 2) * (logdet + neqs)
    return part1 + part2


def _reordered(self, order):
    # Create new arrays to hold rearranged results from .fit()
    endog = self.endog
    endog_lagged = self.endog_lagged
    params = self.params
    sigma_u = self.sigma_u
    names = self.names
    k_ar = self.k_ar
    endog_new = np.zeros_like(endog)
    endog_lagged_new = np.zeros_like(endog_lagged)
    params_new_inc = np.zeros_like(params)
    params_new = np.zeros_like(params)
    sigma_u_new_inc = np.zeros_like(sigma_u)
    sigma_u_new = np.zeros_like(sigma_u)
    num_end = len(self.params[0])
    names_new = []

    # Rearrange elements and fill in new arrays
    k = self.k_trend
    for i, c in enumerate(order):
        endog_new[:, i] = self.endog[:, c]
        if k > 0:
            params_new_inc[0, i] = params[0, i]
            endog_lagged_new[:, 0] = endog_lagged[:, 0]
        for j in range(k_ar):
            params_new_inc[i + j * num_end + k, :] = self.params[
                c + j * num_end + k, :
            ]
            endog_lagged_new[:, i + j * num_end + k] = endog_lagged[
                :, c + j * num_end + k
            ]
        sigma_u_new_inc[i, :] = sigma_u[c, :]
        names_new.append(names[c])
    for i, c in enumerate(order):
        params_new[:, i] = params_new_inc[:, c]
        sigma_u_new[:, i] = sigma_u_new_inc[:, c]

    return VARResults(
        endog=endog_new,
        endog_lagged=endog_lagged_new,
        params=params_new,
        sigma_u=sigma_u_new,
        lag_order=self.k_ar,
        model=self.model,
        trend="c",
        names=names_new,
        dates=self.dates,
    )


def orth_ma_rep(results, maxn=10, P=None):
    r"""Compute Orthogonalized MA coefficient matrices using P matrix such
    that :math:`\Sigma_u = PP^\prime`. P defaults to the Cholesky
    decomposition of :math:`\Sigma_u`

    Parameters
    ----------
    results : VARResults or VECMResults
    maxn : int
        Number of coefficient matrices to compute
    P : ndarray (neqs x neqs), optional
        Matrix such that Sigma_u = PP', defaults to the Cholesky decomposition.

    Returns
    -------
    coefs : ndarray (maxn x neqs x neqs)
    """
    if P is None:
        P = results._chol_sigma_u

    ma_mats = results.ma_rep(maxn=maxn)
    return np.array([np.dot(coefs, P) for coefs in ma_mats])


def test_normality(results, signif=0.05):
    """
    Test assumption of normal-distributed errors using Jarque-Bera-style
    omnibus Chi^2 test

    Parameters
    ----------
    results : VARResults or statsmodels.tsa.vecm.vecm.VECMResults
    signif : float
        The test's significance level.

    Notes
    -----
    H0 (null) : data are generated by a Gaussian-distributed process

    Returns
    -------
    result : NormalityTestResults

    References
    ----------
    .. [1] Lütkepohl, H. 2005. *New Introduction to Multiple Time Series*
       *Analysis*. Springer.

    .. [2] Kilian, L. & Demiroglu, U. (2000). "Residual-Based Tests for
       Normality in Autoregressions: Asymptotic Theory and Simulation
       Evidence." Journal of Business & Economic Statistics
    """
    resid_c = results.resid - results.resid.mean(0)
    sig = np.dot(resid_c.T, resid_c) / results.nobs
    Pinv = np.linalg.inv(np.linalg.cholesky(sig))

    w = np.dot(Pinv, resid_c.T)
    b1 = (w ** 3).sum(1)[:, None] / results.nobs
    b2 = (w ** 4).sum(1)[:, None] / results.nobs - 3

    lam_skew = results.nobs * np.dot(b1.T, b1) / 6
    lam_kurt = results.nobs * np.dot(b2.T, b2) / 24

    lam_omni = float(np.squeeze(lam_skew + lam_kurt))
    omni_dist = stats.chi2(results.neqs * 2)
    omni_pvalue = float(omni_dist.sf(lam_omni))
    crit_omni = float(omni_dist.ppf(1 - signif))

    return NormalityTestResults(
        lam_omni, crit_omni, omni_pvalue, results.neqs * 2, signif
    )


class LagOrderResults:
    """
    Results class for choosing a model's lag order.

    Parameters
    ----------
    ics : dict
        The keys are the strings ``"aic"``, ``"bic"``, ``"hqic"``, and
        ``"fpe"``. A corresponding value is a list of information criteria for
        various numbers of lags.
    selected_orders : dict
        The keys are the strings ``"aic"``, ``"bic"``, ``"hqic"``, and
        ``"fpe"``. The corresponding value is an integer specifying the number
        of lags chosen according to a given criterion (key).
    vecm : bool, default: `False`
        `True` indicates that the model is a VECM. In case of a VAR model
        this argument must be `False`.

    Notes
    -----
    In case of a VECM the shown lags are lagged differences.
    """

    def __init__(self, ics, selected_orders, vecm=False):
        self.title = ("VECM" if vecm else "VAR") + " Order Selection"
        self.title += " (* highlights the minimums)"
        self.ics = ics
        self.selected_orders = selected_orders
        self.vecm = vecm
        self.aic = selected_orders["aic"]
        self.bic = selected_orders["bic"]
        self.hqic = selected_orders["hqic"]
        self.fpe = selected_orders["fpe"]

    def summary(self):  # basically copied from (now deleted) print_ic_table()
        cols = sorted(self.ics)  # ["aic", "bic", "hqic", "fpe"]
        str_data = np.array(
            [["%#10.4g" % v for v in self.ics[c]] for c in cols], dtype=object
        ).T
        # mark minimum with an asterisk
        for i, col in enumerate(cols):
            idx = int(self.selected_orders[col]), i
            str_data[idx] += "*"
        return SimpleTable(
            str_data,
            [col.upper() for col in cols],
            lrange(len(str_data)),
            title=self.title,
        )

    def __str__(self):
        return (
            f"<{self.__module__}.{self.__class__.__name__} object. Selected "
            f"orders are: AIC -> {str(self.aic)}, BIC -> {str(self.bic)}, "
            f"FPE -> {str(self.fpe)}, HQIC ->  {str(self.hqic)}>"
        )


# -------------------------------------------------------------------------------
# VARProcess class: for known or unknown VAR process


class VAR(TimeSeriesModel):
    r"""
    Fit VAR(p) process and do lag order selection

    .. math:: y_t = A_1 y_{t-1} + \ldots + A_p y_{t-p} + u_t

    Parameters
    ----------
    endog : array_like
        2-d endogenous response variable. The independent variable.
    exog : array_like
        2-d exogenous variable.
    dates : array_like
        must match number of rows of endog

    References
    ----------
    Lütkepohl (2005) New Introduction to Multiple Time Series Analysis
    """

    y = deprecated_alias("y", "endog", remove_version="0.11.0")

    def __init__(
        self, endog, exog=None, dates=None, freq=None, missing="none"
    ):
        super().__init__(endog, exog, dates, freq, missing=missing)
        if self.endog.ndim == 1:
            raise ValueError("Only gave one variable to VAR")
        self.neqs = self.endog.shape[1]
        self.n_totobs = len(endog)

    def predict(self, params, start=None, end=None, lags=1, trend="c"):
        """
        Returns in-sample predictions or forecasts
        """
        params = np.array(params)

        if start is None:
            start = lags

        # Handle start, end
        (
            start,
            end,
            out_of_sample,
            prediction_index,
        ) = self._get_prediction_index(start, end)

        if end < start:
            raise ValueError("end is before start")
        if end == start + out_of_sample:
            return np.array([])

        k_trend = util.get_trendorder(trend)
        k = self.neqs
        k_ar = lags

        predictedvalues = np.zeros((end + 1 - start + out_of_sample, k))
        if k_trend != 0:
            intercept = params[:k_trend]
            predictedvalues += intercept

        y = self.endog
        x = util.get_var_endog(y, lags, trend=trend, has_constant="raise")
        fittedvalues = np.dot(x, params)

        fv_start = start - k_ar
        pv_end = min(len(predictedvalues), len(fittedvalues) - fv_start)
        fv_end = min(len(fittedvalues), end - k_ar + 1)
        predictedvalues[:pv_end] = fittedvalues[fv_start:fv_end]

        if not out_of_sample:
            return predictedvalues

        # fit out of sample
        y = y[-k_ar:]
        coefs = params[k_trend:].reshape((k_ar, k, k)).swapaxes(1, 2)
        predictedvalues[pv_end:] = forecast(y, coefs, intercept, out_of_sample)
        return predictedvalues

    def fit(
        self,
        maxlags: int | None = None,
        method="ols",
        ic=None,
        trend="c",
        verbose=False,
    ):
        # todo: this code is only supporting deterministic terms as exog.
        # This means that all exog-variables have lag 0. If dealing with
        # different exogs is necessary, a `lags_exog`-parameter might make
        # sense (e.g. a sequence of ints specifying lags).
        # Alternatively, leading zeros for exog-variables with smaller number
        # of lags than the maximum number of exog-lags might work.
        """
        Fit the VAR model

        Parameters
        ----------
        maxlags : {int, None}, default None
            Maximum number of lags to check for order selection, defaults to
            12 * (nobs/100.)**(1./4), see select_order function
        method : {'ols'}
            Estimation method to use
        ic : {'aic', 'fpe', 'hqic', 'bic', None}
            Information criterion to use for VAR order selection.
            aic : Akaike
            fpe : Final prediction error
            hqic : Hannan-Quinn
            bic : Bayesian a.k.a. Schwarz
        verbose : bool, default False
            Print order selection output to the screen
        trend : str {"c", "ct", "ctt", "n"}
            "c" - add constant
            "ct" - constant and trend
            "ctt" - constant, linear and quadratic trend
            "n" - co constant, no trend
            Note that these are prepended to the columns of the dataset.

        Returns
        -------
        VARResults
            Estimation results

        Notes
        -----
        See Lütkepohl pp. 146-153 for implementation details.
        """
        lags = maxlags
        if trend not in ["c", "ct", "ctt", "n"]:
            raise ValueError("trend '{}' not supported for VAR".format(trend))

        if ic is not None:
            selections = self.select_order(maxlags=maxlags)
            if not hasattr(selections, ic):
                raise ValueError(
                    "%s not recognized, must be among %s"
                    % (ic, sorted(selections))
                )
            lags = getattr(selections, ic)
            if verbose:
                print(selections)
                print("Using %d based on %s criterion" % (lags, ic))
        else:
            if lags is None:
                lags = 1

        k_trend = util.get_trendorder(trend)
        orig_exog_names = self.exog_names
        self.exog_names = util.make_lag_names(self.endog_names, lags, k_trend)
        self.nobs = self.n_totobs - lags

        # add exog to data.xnames (necessary because the length of xnames also
        # determines the allowed size of VARResults.params)
        if self.exog is not None:
            if orig_exog_names:
                x_names_to_add = orig_exog_names
            else:
                x_names_to_add = [
                    ("exog%d" % i) for i in range(self.exog.shape[1])
                ]
            self.data.xnames = (
                self.data.xnames[:k_trend]
                + x_names_to_add
                + self.data.xnames[k_trend:]
            )
        self.data.cov_names = pd.MultiIndex.from_product(
            (self.data.xnames, self.data.ynames)
        )
        return self._estimate_var(lags, trend=trend)

    def _estimate_var(self, lags, offset=0, trend="c"):
        """
        lags : int
            Lags of the endogenous variable.
        offset : int
            Periods to drop from beginning-- for order selection so it's an
            apples-to-apples comparison
        trend : {str, None}
            As per above
        """
        # have to do this again because select_order does not call fit
        self.k_trend = k_trend = util.get_trendorder(trend)

        if offset < 0:  # pragma: no cover
            raise ValueError("offset must be >= 0")

        nobs = self.n_totobs - lags - offset
        endog = self.endog[offset:]
        exog = None if self.exog is None else self.exog[offset:]
        z = util.get_var_endog(endog, lags, trend=trend, has_constant="raise")
        if exog is not None:
            # TODO: currently only deterministic terms supported (exoglags==0)
            # and since exoglags==0, x will be an array of size 0.
            x = util.get_var_endog(
                exog[-nobs:], 0, trend="n", has_constant="raise"
            )
            x_inst = exog[-nobs:]
            x = np.column_stack((x, x_inst))
            del x_inst  # free memory
            temp_z = z
            z = np.empty((x.shape[0], x.shape[1] + z.shape[1]))
            z[:, : self.k_trend] = temp_z[:, : self.k_trend]
            z[:, self.k_trend : self.k_trend + x.shape[1]] = x
            z[:, self.k_trend + x.shape[1] :] = temp_z[:, self.k_trend :]
            del temp_z, x  # free memory
        # the following modification of z is necessary to get the same results
        # as JMulTi for the constant-term-parameter...
        for i in range(self.k_trend):
            if (np.diff(z[:, i]) == 1).all():  # modify the trend-column
                z[:, i] += lags
            # make the same adjustment for the quadratic term
            if (np.diff(np.sqrt(z[:, i])) == 1).all():
                z[:, i] = (np.sqrt(z[:, i]) + lags) ** 2

        y_sample = endog[lags:]
        # Lütkepohl p75, about 5x faster than stated formula
        params = np.linalg.lstsq(z, y_sample, rcond=1e-15)[0]
        resid = y_sample - np.dot(z, params)

        # Unbiased estimate of covariance matrix $\Sigma_u$ of the white noise
        # process $u$
        # equivalent definition
        # .. math:: \frac{1}{T - Kp - 1} Y^\prime (I_T - Z (Z^\prime Z)^{-1}
        # Z^\prime) Y
        # Ref: Lütkepohl p.75
        # df_resid right now is T - Kp - 1, which is a suggested correction

        avobs = len(y_sample)
        if exog is not None:
            k_trend += exog.shape[1]
        df_resid = avobs - (self.neqs * lags + k_trend)

        sse = np.dot(resid.T, resid)
        if df_resid:
            omega = sse / df_resid
        else:
            omega = np.full_like(sse, np.nan)

        varfit = VARResults(
            endog,
            z,
            params,
            omega,
            lags,
            names=self.endog_names,
            trend=trend,
            dates=self.data.dates,
            model=self,
            exog=self.exog,
        )
        return VARResultsWrapper(varfit)

    def select_order(self, maxlags=None, trend="c"):
        """
        Compute lag order selections based on each of the available information
        criteria

        Parameters
        ----------
        maxlags : int
            if None, defaults to 12 * (nobs/100.)**(1./4)
        trend : str {"n", "c", "ct", "ctt"}
            * "n" - no deterministic terms
            * "c" - constant term
            * "ct" - constant and linear term
            * "ctt" - constant, linear, and quadratic term

        Returns
        -------
        selections : LagOrderResults
        """
        ntrend = len(trend) if trend.startswith("c") else 0
        max_estimable = (self.n_totobs - self.neqs - ntrend) // (1 + self.neqs)
        if maxlags is None:
            maxlags = int(round(12 * (len(self.endog) / 100.0) ** (1 / 4.0)))
            # TODO: This expression shows up in a bunch of places, but
            #  in some it is `int` and in others `np.ceil`.  Also in some
            #  it multiplies by 4 instead of 12.  Let's put these all in
            #  one place and document when to use which variant.

            # Ensure enough obs to estimate model with maxlags
            maxlags = min(maxlags, max_estimable)
        else:
            if maxlags > max_estimable:
                raise ValueError(
                    "maxlags is too large for the number of observations and "
                    "the number of equations. The largest model cannot be "
                    "estimated."
                )

        ics = defaultdict(list)
        p_min = 0 if self.exog is not None or trend != "n" else 1
        for p in range(p_min, maxlags + 1):
            # exclude some periods to same amount of data used for each lag
            # order
            result = self._estimate_var(p, offset=maxlags - p, trend=trend)

            for k, v in result.info_criteria.items():
                ics[k].append(v)

        selected_orders = dict(
            (k, np.array(v).argmin() + p_min) for k, v in ics.items()
        )

        return LagOrderResults(ics, selected_orders, vecm=False)

    @classmethod
    def from_formula(
        cls, formula, data, subset=None, drop_cols=None, *args, **kwargs
    ):
        """
        Not implemented. Formulas are not supported for VAR models.
        """
        raise NotImplementedError("formulas are not supported for VAR models.")


class VARProcess:
    """
    Class represents a known VAR(p) process

    Parameters
    ----------
    coefs : ndarray (p x k x k)
        coefficients for lags of endog, part or params reshaped
    coefs_exog : ndarray
        parameters for trend and user provided exog
    sigma_u : ndarray (k x k)
        residual covariance
    names : sequence (length k)
    _params_info : dict
        internal dict to provide information about the composition of `params`,
        specifically `k_trend` (trend order) and `k_exog_user` (the number of
        exog variables provided by the user).
        If it is None, then coefs_exog are assumed to be for the intercept and
        trend.
    """

    def __init__(
        self, coefs, coefs_exog, sigma_u, names=None, _params_info=None
    ):
        self.k_ar = len(coefs)
        self.neqs = coefs.shape[1]
        self.coefs = coefs
        self.coefs_exog = coefs_exog
        # Note reshaping 1-D coefs_exog to 2_D makes unit tests fail
        self.sigma_u = sigma_u
        self.names = names

        if _params_info is None:
            _params_info = {}
        self.k_exog_user = _params_info.get("k_exog_user", 0)
        if self.coefs_exog is not None:
            k_ex = self.coefs_exog.shape[0] if self.coefs_exog.ndim != 1 else 1
            k_c = k_ex - self.k_exog_user
        else:
            k_c = 0
        self.k_trend = _params_info.get("k_trend", k_c)
        # TODO: we need to distinguish exog including trend and exog_user
        self.k_exog = self.k_trend + self.k_exog_user

        if self.k_trend > 0:
            if coefs_exog.ndim == 2:
                self.intercept = coefs_exog[:, 0]
            else:
                self.intercept = coefs_exog
        else:
            self.intercept = np.zeros(self.neqs)

    def get_eq_index(self, name):
        """Return integer position of requested equation name"""
        return util.get_index(self.names, name)

    def __str__(self):
        output = "VAR(%d) process for %d-dimensional response y_t" % (
            self.k_ar,
            self.neqs,
        )
        output += "\nstable: %s" % self.is_stable()
        output += "\nmean: %s" % self.mean()

        return output

    def is_stable(self, verbose=False):
        """Determine stability based on model coefficients

        Parameters
        ----------
        verbose : bool
            Print eigenvalues of the VAR(1) companion

        Notes
        -----
        Checks if det(I - Az) = 0 for any mod(z) <= 1, so all the eigenvalues of
        the companion matrix must lie outside the unit circle
        """
        return is_stable(self.coefs, verbose=verbose)

    def simulate_var(self, steps=None, offset=None, seed=None, initial_values=None, nsimulations=None):
        """
        simulate the VAR(p) process for the desired number of steps

        Parameters
        ----------
        steps : None or int
            number of observations to simulate, this includes the initial
            observations to start the autoregressive process.
            If offset is not None, then exog of the model are used if they were
            provided in the model
        offset : None or ndarray (steps, neqs)
            If not None, then offset is added as an observation specific
            intercept to the autoregression. If it is None and either trend
            (including intercept) or exog were used in the VAR model, then
            the linear predictor of those components will be used as offset.
            This should have the same number of rows as steps, and the same
            number of columns as endogenous variables (neqs).
        seed : {None, int}
            If seed is not None, then it will be used with for the random
            variables generated by numpy.random.
        initial_values : array_like, optional
            Initial values for use in the simulation. Shape should be
            (nlags, neqs) or (neqs,). Values should be ordered from less to
            most recent. Note that this values will be returned by the
            simulation as the first values of `endog_simulated` and they
            will count for the total number of steps.
        nsimulations : {None, int}
            Number of simulations to perform. If `nsimulations` is None it will
            perform one simulation and return value will have shape (steps, neqs).

        Returns
        -------
        endog_simulated : nd_array
            Endog of the simulated VAR process. Shape will be (nsimulations, steps, neqs)
            or (steps, neqs) if `nsimulations` is None.
        """
        steps_ = None
        if offset is None:
            if self.k_exog_user > 0 or self.k_trend > 1:
                # if more than intercept
                # endog_lagged contains all regressors, trend, exog_user
                # and lagged endog, trimmed initial observations
                offset = self.endog_lagged[:, : self.k_exog].dot(
                    self.coefs_exog.T
                )
                steps_ = self.endog_lagged.shape[0]
            else:
                offset = self.intercept
        else:
            steps_ = offset.shape[0]

        # default, but over written if exog or offset are used
        if steps is None:
            if steps_ is None:
                steps = 1000
            else:
                steps = steps_
        else:
            if steps_ is not None and steps != steps_:
                raise ValueError(
                    "if exog or offset are used, then steps must"
                    "be equal to their length or None"
                )

        y = util.varsim(
            self.coefs,
            offset,
            self.sigma_u,
            steps=steps,
            seed=seed,
            initial_values=initial_values,
            nsimulations=nsimulations
        )
        return y

    def plotsim(self, steps=None, offset=None, seed=None):
        """
        Plot a simulation from the VAR(p) process for the desired number of
        steps
        """
        y = self.simulate_var(steps=steps, offset=offset, seed=seed)
        return plotting.plot_mts(y)

    def intercept_longrun(self):
        r"""
        Long run intercept of stable VAR process

        Lütkepohl eq. 2.1.23

        .. math:: \mu = (I - A_1 - \dots - A_p)^{-1} \alpha

        where \alpha is the intercept (parameter of the constant)
        """
        return np.linalg.solve(self._char_mat, self.intercept)

    def mean(self):
        r"""
        Long run intercept of stable VAR process

        Warning: trend and exog except for intercept are ignored for this.
        This might change in future versions.

        Lütkepohl eq. 2.1.23

        .. math:: \mu = (I - A_1 - \dots - A_p)^{-1} \alpha

        where \alpha is the intercept (parameter of the constant)
        """
        return self.intercept_longrun()

    def ma_rep(self, maxn=10):
        r"""
        Compute MA(:math:`\infty`) coefficient matrices

        Parameters
        ----------
        maxn : int
            Number of coefficient matrices to compute

        Returns
        -------
        coefs : ndarray (maxn x k x k)
        """
        return ma_rep(self.coefs, maxn=maxn)

    def orth_ma_rep(self, maxn=10, P=None):
        r"""
        Compute orthogonalized MA coefficient matrices using P matrix such
        that :math:`\Sigma_u = PP^\prime`. P defaults to the Cholesky
        decomposition of :math:`\Sigma_u`

        Parameters
        ----------
        maxn : int
            Number of coefficient matrices to compute
        P : ndarray (k x k), optional
            Matrix such that Sigma_u = PP', defaults to Cholesky descomp

        Returns
        -------
        coefs : ndarray (maxn x k x k)
        """
        return orth_ma_rep(self, maxn, P)

    def long_run_effects(self):
        r"""Compute long-run effect of unit impulse

        .. math::

            \Psi_\infty = \sum_{i=0}^\infty \Phi_i
        """
        return np.linalg.inv(self._char_mat)

    @cache_readonly
    def _chol_sigma_u(self):
        return np.linalg.cholesky(self.sigma_u)

    @cache_readonly
    def _char_mat(self):
        """Characteristic matrix of the VAR"""
        return np.eye(self.neqs) - self.coefs.sum(0)

    def acf(self, nlags=None):
        """Compute theoretical autocovariance function

        Returns
        -------
        acf : ndarray (p x k x k)
        """
        return var_acf(self.coefs, self.sigma_u, nlags=nlags)

    def acorr(self, nlags=None):
        """
        Autocorrelation function

        Parameters
        ----------
        nlags : int or None
            The number of lags to include in the autocovariance function. The
            default is the number of lags included in the model.

        Returns
        -------
        acorr : ndarray
            Autocorrelation and cross correlations (nlags, neqs, neqs)
        """
        return util.acf_to_acorr(self.acf(nlags=nlags))

    def plot_acorr(self, nlags=10, linewidth=8):
        """Plot theoretical autocorrelation function"""
        fig = plotting.plot_full_acorr(
            self.acorr(nlags=nlags), linewidth=linewidth
        )
        return fig

    def forecast(self, y, steps, exog_future=None):
        """Produce linear minimum MSE forecasts for desired number of steps
        ahead, using prior values y

        Parameters
        ----------
        y : ndarray (p x k)
        steps : int

        Returns
        -------
        forecasts : ndarray (steps x neqs)

        Notes
        -----
        Lütkepohl pp 37-38
        """
        if self.exog is None and exog_future is not None:
            raise ValueError(
                "No exog in model, so no exog_future supported "
                "in forecast method."
            )
        if self.exog is not None and exog_future is None:
            raise ValueError(
                "Please provide an exog_future argument to "
                "the forecast method."
            )

        exog_future = array_like(
            exog_future, "exog_future", optional=True, ndim=2
        )
        if exog_future is not None:
            if exog_future.shape[0] != steps:
                err_msg = f"""\
exog_future only has {exog_future.shape[0]} observations. It must have \
steps ({steps}) observations.
"""
                raise ValueError(err_msg)
        trend_coefs = None if self.coefs_exog.size == 0 else self.coefs_exog.T

        exogs = []
        if self.trend.startswith("c"):  # constant term
            exogs.append(np.ones(steps))
        exog_lin_trend = np.arange(
            self.n_totobs + 1, self.n_totobs + 1 + steps
        )
        if "t" in self.trend:
            exogs.append(exog_lin_trend)
        if "tt" in self.trend:
            exogs.append(exog_lin_trend ** 2)
        if exog_future is not None:
            exogs.append(exog_future)

        if not exogs:
            exog_future = None
        else:
            exog_future = np.column_stack(exogs)
        return forecast(y, self.coefs, trend_coefs, steps, exog_future)

    # TODO: use `mse` module-level function?
    def mse(self, steps):
        r"""
        Compute theoretical forecast error variance matrices

        Parameters
        ----------
        steps : int
            Number of steps ahead

        Notes
        -----
        .. math:: \mathrm{MSE}(h) = \sum_{i=0}^{h-1} \Phi \Sigma_u \Phi^T

        Returns
        -------
        forc_covs : ndarray (steps x neqs x neqs)
        """
        ma_coefs = self.ma_rep(steps)

        k = len(self.sigma_u)
        forc_covs = np.zeros((steps, k, k))

        prior = np.zeros((k, k))
        for h in range(steps):
            # Sigma(h) = Sigma(h-1) + Phi Sig_u Phi'
            phi = ma_coefs[h]
            var = phi @ self.sigma_u @ phi.T
            forc_covs[h] = prior = prior + var

        return forc_covs

    forecast_cov = mse

    def _forecast_vars(self, steps):
        covs = self.forecast_cov(steps)

        # Take diagonal for each cov
        inds = np.arange(self.neqs)
        return covs[:, inds, inds]

    def forecast_interval(self, y, steps, alpha=0.05, exog_future=None):
        """
        Construct forecast interval estimates assuming the y are Gaussian

        Parameters
        ----------
        y : {ndarray, None}
            The initial values to use for the forecasts. If None,
            the last k_ar values of the original endogenous variables are
            used.
        steps : int
            Number of steps ahead to forecast
        alpha : float, optional
            The significance level for the confidence intervals.
        exog_future : ndarray, optional
            Forecast values of the exogenous variables. Should include
            constant, trend, etc. as needed, including extrapolating out
            of sample.
        Returns
        -------
        point : ndarray
            Mean value of forecast
        lower : ndarray
            Lower bound of confidence interval
        upper : ndarray
            Upper bound of confidence interval

        Notes
        -----
        Lütkepohl pp. 39-40
        """
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        q = util.norm_signif_level(alpha)

        point_forecast = self.forecast(y, steps, exog_future=exog_future)
        sigma = np.sqrt(self._forecast_vars(steps))

        forc_lower = point_forecast - q * sigma
        forc_upper = point_forecast + q * sigma

        return point_forecast, forc_lower, forc_upper

    def to_vecm(self):
        """to_vecm"""
        k = self.coefs.shape[1]
        p = self.coefs.shape[0]
        A = self.coefs
        pi = -(np.identity(k) - np.sum(A, 0))
        gamma = np.zeros((p - 1, k, k))
        for i in range(p - 1):
            gamma[i] = -(np.sum(A[i + 1 :], 0))
        gamma = np.concatenate(gamma, 1)
        return {"Gamma": gamma, "Pi": pi}


# ----------------------------------------------------------------------------
# VARResults class


class VARResults(VARProcess):
    """Estimate VAR(p) process with fixed number of lags

    Parameters
    ----------
    endog : ndarray
    endog_lagged : ndarray
    params : ndarray
    sigma_u : ndarray
    lag_order : int
    model : VAR model instance
    trend : str {'n', 'c', 'ct'}
    names : array_like
        List of names of the endogenous variables in order of appearance in
        `endog`.
    dates
    exog : ndarray

    Attributes
    ----------
    params : ndarray (p x K x K)
        Estimated A_i matrices, A_i = coefs[i-1]
    dates
    endog
    endog_lagged
    k_ar : int
        Order of VAR process
    k_trend : int
    model
    names
    neqs : int
        Number of variables (equations)
    nobs : int
    n_totobs : int
    params : ndarray (Kp + 1) x K
        A_i matrices and intercept in stacked form [int A_1 ... A_p]
    names : list
        variables names
    sigma_u : ndarray (K x K)
        Estimate of white noise process variance Var[u_t]
    """

    _model_type = "VAR"

    def __init__(
        self,
        endog,
        endog_lagged,
        params,
        sigma_u,
        lag_order,
        model=None,
        trend="c",
        names=None,
        dates=None,
        exog=None,
    ):

        self.model = model
        self.endog = endog
        self.endog_lagged = endog_lagged
        self.dates = dates

        self.n_totobs, neqs = self.endog.shape
        self.nobs = self.n_totobs - lag_order
        self.trend = trend
        k_trend = util.get_trendorder(trend)
        self.exog_names = util.make_lag_names(
            names, lag_order, k_trend, model.data.orig_exog
        )
        self.params = params
        self.exog = exog

        # Initialize VARProcess parent class
        # construct coefficient matrices
        # Each matrix needs to be transposed
        endog_start = k_trend
        if exog is not None:
            k_exog_user = exog.shape[1]
            endog_start += k_exog_user
        else:
            k_exog_user = 0
        reshaped = self.params[endog_start:]
        reshaped = reshaped.reshape((lag_order, neqs, neqs))
        # Need to transpose each coefficient matrix
        coefs = reshaped.swapaxes(1, 2).copy()

        self.coefs_exog = params[:endog_start].T
        self.k_exog = self.coefs_exog.shape[1]
        self.k_exog_user = k_exog_user

        # maybe change to params class, distinguish exog_all versus exog_user
        # see issue #4535
        _params_info = {
            "k_trend": k_trend,
            "k_exog_user": k_exog_user,
            "k_ar": lag_order,
        }
        super().__init__(
            coefs,
            self.coefs_exog,
            sigma_u,
            names=names,
            _params_info=_params_info,
        )

    def plot(self):
        """Plot input time series"""
        return plotting.plot_mts(
            self.endog, names=self.names, index=self.dates
        )

    @property
    def df_model(self):
        """
        Number of estimated parameters per variable, including the intercept / trends
        """
        return self.neqs * self.k_ar + self.k_exog

    @property
    def df_resid(self):
        """Number of observations minus number of estimated parameters"""
        return self.nobs - self.df_model

    @cache_readonly
    def fittedvalues(self):
        """
        The predicted insample values of the response variables of the model.
        """
        return np.dot(self.endog_lagged, self.params)

    @cache_readonly
    def resid(self):
        """
        Residuals of response variable resulting from estimated coefficients
        """
        return self.endog[self.k_ar :] - self.fittedvalues

    def sample_acov(self, nlags=1):
        """Sample acov"""
        return _compute_acov(self.endog[self.k_ar :], nlags=nlags)

    def sample_acorr(self, nlags=1):
        """Sample acorr"""
        acovs = self.sample_acov(nlags=nlags)
        return _acovs_to_acorrs(acovs)

    def plot_sample_acorr(self, nlags=10, linewidth=8):
        """
        Plot sample autocorrelation function

        Parameters
        ----------
        nlags : int
            The number of lags to use in compute the autocorrelation. Does
            not count the zero lag, which will be returned.
        linewidth : int
            The linewidth for the plots.

        Returns
        -------
        Figure
            The figure that contains the plot axes.
        """
        fig = plotting.plot_full_acorr(
            self.sample_acorr(nlags=nlags), linewidth=linewidth
        )
        return fig

    def resid_acov(self, nlags=1):
        """
        Compute centered sample autocovariance (including lag 0)

        Parameters
        ----------
        nlags : int

        Returns
        -------
        """
        return _compute_acov(self.resid, nlags=nlags)

    def resid_acorr(self, nlags=1):
        """
        Compute sample autocorrelation (including lag 0)

        Parameters
        ----------
        nlags : int

        Returns
        -------
        """
        acovs = self.resid_acov(nlags=nlags)
        return _acovs_to_acorrs(acovs)

    @cache_readonly
    def resid_corr(self):
        """
        Centered residual correlation matrix
        """
        return self.resid_acorr(0)[0]

    @cache_readonly
    def sigma_u_mle(self):
        """(Biased) maximum likelihood estimate of noise process covariance"""
        if not self.df_resid:
            return np.zeros_like(self.sigma_u)
        return self.sigma_u * self.df_resid / self.nobs

    def cov_params(self):
        """Estimated variance-covariance of model coefficients

        Notes
        -----
        Covariance of vec(B), where B is the matrix
        [params_for_deterministic_terms, A_1, ..., A_p] with the shape
        (K x (Kp + number_of_deterministic_terms))
        Adjusted to be an unbiased estimator
        Ref: Lütkepohl p.74-75
        """
        z = self.endog_lagged
        return np.kron(np.linalg.inv(z.T @ z), self.sigma_u)

    def cov_ybar(self):
        r"""Asymptotically consistent estimate of covariance of the sample mean

        .. math::

            \sqrt(T) (\bar{y} - \mu) \rightarrow
                  {\cal N}(0, \Sigma_{\bar{y}}) \\

            \Sigma_{\bar{y}} = B \Sigma_u B^\prime, \text{where }
                  B = (I_K - A_1 - \cdots - A_p)^{-1}

        Notes
        -----
        Lütkepohl Proposition 3.3
        """

        Ainv = np.linalg.inv(np.eye(self.neqs) - self.coefs.sum(0))
        return Ainv @ self.sigma_u @ Ainv.T

    # ------------------------------------------------------------
    # Estimation-related things

    @cache_readonly
    def _zz(self):
        # Z'Z
        return np.dot(self.endog_lagged.T, self.endog_lagged)

    @property
    def _cov_alpha(self):
        """
        Estimated covariance matrix of model coefficients w/o exog
        """
        # drop exog
        kn = self.k_exog * self.neqs
        return self.cov_params()[kn:, kn:]

    @cache_readonly
    def _cov_sigma(self):
        """
        Estimated covariance matrix of vech(sigma_u)
        """
        D_K = tsa.duplication_matrix(self.neqs)
        D_Kinv = np.linalg.pinv(D_K)

        sigxsig = np.kron(self.sigma_u, self.sigma_u)
        return 2 * D_Kinv @ sigxsig @ D_Kinv.T

    @cache_readonly
    def llf(self):
        "Compute VAR(p) loglikelihood"
        return var_loglike(self.resid, self.sigma_u_mle, self.nobs)

    @cache_readonly
    def stderr(self):
        """Standard errors of coefficients, reshaped to match in size"""
        stderr = np.sqrt(np.diag(self.cov_params()))
        return stderr.reshape((self.df_model, self.neqs), order="C")

    bse = stderr  # statsmodels interface?

    @cache_readonly
    def stderr_endog_lagged(self):
        """Stderr_endog_lagged"""
        start = self.k_exog
        return self.stderr[start:]

    @cache_readonly
    def stderr_dt(self):
        """Stderr_dt"""
        end = self.k_exog
        return self.stderr[:end]

    @cache_readonly
    def tvalues(self):
        """
        Compute t-statistics. Use Student-t(T - Kp - 1) = t(df_resid) to
        test significance.
        """
        return self.params / self.stderr

    @cache_readonly
    def tvalues_endog_lagged(self):
        """tvalues_endog_lagged"""
        start = self.k_exog
        return self.tvalues[start:]

    @cache_readonly
    def tvalues_dt(self):
        """tvalues_dt"""
        end = self.k_exog
        return self.tvalues[:end]

    @cache_readonly
    def pvalues(self):
        """
        Two-sided p-values for model coefficients from Student t-distribution
        """
        # return stats.t.sf(np.abs(self.tvalues), self.df_resid)*2
        return 2 * stats.norm.sf(np.abs(self.tvalues))

    @cache_readonly
    def pvalues_endog_lagged(self):
        """pvalues_endog_laggd"""
        start = self.k_exog
        return self.pvalues[start:]

    @cache_readonly
    def pvalues_dt(self):
        """pvalues_dt"""
        end = self.k_exog
        return self.pvalues[:end]

    # todo: ------------------------------------------------------------------

    def plot_forecast(self, steps, alpha=0.05, plot_stderr=True):
        """
        Plot forecast
        """
        mid, lower, upper = self.forecast_interval(
            self.endog[-self.k_ar :], steps, alpha=alpha
        )
        fig = plotting.plot_var_forc(
            self.endog,
            mid,
            lower,
            upper,
            names=self.names,
            plot_stderr=plot_stderr,
        )
        return fig

    # Forecast error covariance functions

    def forecast_cov(self, steps=1, method="mse"):
        r"""Compute forecast covariance matrices for desired number of steps

        Parameters
        ----------
        steps : int

        Notes
        -----
        .. math:: \Sigma_{\hat y}(h) = \Sigma_y(h) + \Omega(h) / T

        Ref: Lütkepohl pp. 96-97

        Returns
        -------
        covs : ndarray (steps x k x k)
        """
        fc_cov = self.mse(steps)
        if method == "mse":
            pass
        elif method == "auto":
            if self.k_exog == 1 and self.k_trend < 2:
                # currently only supported if no exog and trend in ['n', 'c']
                fc_cov += self._omega_forc_cov(steps) / self.nobs
                import warnings

                warnings.warn(
                    "forecast cov takes parameter uncertainty into" "account",
                    OutputWarning,
                    stacklevel = 2,
                )
        else:
            raise ValueError("method has to be either 'mse' or 'auto'")

        return fc_cov

    # Monte Carlo irf standard errors
    def irf_errband_mc(
        self,
        orth=False,
        repl=1000,
        steps=10,
        signif=0.05,
        seed=None,
        burn=100,
        cum=False,
    ):
        """
        Compute Monte Carlo integrated error bands assuming normally
        distributed for impulse response functions

        Parameters
        ----------
        orth : bool, default False
            Compute orthogonalized impulse response error bands
        repl : int
            number of Monte Carlo replications to perform
        steps : int, default 10
            number of impulse response periods
        signif : float (0 < signif <1)
            Significance level for error bars, defaults to 95% CI
        seed : int
            np.random.seed for replications
        burn : int
            number of initial observations to discard for simulation
        cum : bool, default False
            produce cumulative irf error bands

        Notes
        -----
        Lütkepohl (2005) Appendix D

        Returns
        -------
        Tuple of lower and upper arrays of ma_rep monte carlo standard errors
        """
        ma_coll = self.irf_resim(
            orth=orth, repl=repl, steps=steps, seed=seed, burn=burn, cum=cum
        )

        ma_sort = np.sort(ma_coll, axis=0)  # sort to get quantiles
        # python 2: round returns float
        low_idx = int(round(signif / 2 * repl) - 1)
        upp_idx = int(round((1 - signif / 2) * repl) - 1)
        lower = ma_sort[low_idx, :, :, :]
        upper = ma_sort[upp_idx, :, :, :]
        return lower, upper

    def irf_resim(
        self, orth=False, repl=1000, steps=10, seed=None, burn=100, cum=False
    ):
        """
        Simulates impulse response function, returning an array of simulations.
        Used for Sims-Zha error band calculation.

        Parameters
        ----------
        orth : bool, default False
            Compute orthogonalized impulse response error bands
        repl : int
            number of Monte Carlo replications to perform
        steps : int, default 10
            number of impulse response periods
        signif : float (0 < signif <1)
            Significance level for error bars, defaults to 95% CI
        seed : int
            np.random.seed for replications
        burn : int
            number of initial observations to discard for simulation
        cum : bool, default False
            produce cumulative irf error bands

        Notes
        -----
        .. [*] Sims, Christoper A., and Tao Zha. 1999. "Error Bands for Impulse
           Response." Econometrica 67: 1113-1155.

        Returns
        -------
        Array of simulated impulse response functions
        """
        neqs = self.neqs
        k_ar = self.k_ar
        coefs = self.coefs
        sigma_u = self.sigma_u
        intercept = self.intercept
        nobs = self.nobs
        nobs_original = nobs + k_ar

        ma_coll = np.zeros((repl, steps + 1, neqs, neqs))

        def fill_coll(sim):
            ret = VAR(sim, exog=self.exog).fit(maxlags=k_ar, trend=self.trend)
            ret = (
                ret.orth_ma_rep(maxn=steps) if orth else ret.ma_rep(maxn=steps)
            )
            return ret.cumsum(axis=0) if cum else ret

        for i in range(repl):
            # discard first burn to eliminate correct for starting bias
            sim = util.varsim(
                coefs,
                intercept,
                sigma_u,
                seed=seed,
                steps=nobs_original + burn,
            )
            sim = sim[burn:]
            ma_coll[i, :, :, :] = fill_coll(sim)

        return ma_coll

    def _omega_forc_cov(self, steps):
        # Approximate MSE matrix \Omega(h) as defined in Lut p97
        G = self._zz
        Ginv = np.linalg.inv(G)

        # memoize powers of B for speedup
        # TODO: see if can memoize better
        # TODO: much lower-hanging fruit in caching `np.trace` below.
        B = self._bmat_forc_cov()
        _B = {}

        def bpow(i):
            if i not in _B:
                _B[i] = np.linalg.matrix_power(B, i)

            return _B[i]

        phis = self.ma_rep(steps)
        sig_u = self.sigma_u

        omegas = np.zeros((steps, self.neqs, self.neqs))
        for h in range(1, steps + 1):
            if h == 1:
                omegas[h - 1] = self.df_model * self.sigma_u
                continue

            om = omegas[h - 1]
            for i in range(h):
                for j in range(h):
                    Bi = bpow(h - 1 - i)
                    Bj = bpow(h - 1 - j)
                    mult = np.trace(Bi.T @ Ginv @ Bj @ G)
                    om += mult * phis[i] @ sig_u @ phis[j].T
            omegas[h - 1] = om

        return omegas

    def _bmat_forc_cov(self):
        # B as defined on p. 96 of Lut
        upper = np.zeros((self.k_exog, self.df_model))
        upper[:, : self.k_exog] = np.eye(self.k_exog)

        lower_dim = self.neqs * (self.k_ar - 1)
        eye = np.eye(lower_dim)
        lower = np.column_stack(
            (
                np.zeros((lower_dim, self.k_exog)),
                eye,
                np.zeros((lower_dim, self.neqs)),
            )
        )

        return np.vstack((upper, self.params.T, lower))

    def summary(self):
        """Compute console output summary of estimates

        Returns
        -------
        summary : VARSummary
        """
        return VARSummary(self)

    def irf(self, periods=10, var_decomp=None, var_order=None):
        """Analyze impulse responses to shocks in system

        Parameters
        ----------
        periods : int
        var_decomp : ndarray (k x k), lower triangular
            Must satisfy Omega = P P', where P is the passed matrix. Defaults
            to Cholesky decomposition of Omega
        var_order : sequence
            Alternate variable order for Cholesky decomposition

        Returns
        -------
        irf : IRAnalysis
        """
        if var_order is not None:
            raise NotImplementedError(
                "alternate variable order not implemented" " (yet)"
            )

        return IRAnalysis(self, P=var_decomp, periods=periods)

    def fevd(self, periods=10, var_decomp=None):
        """
        Compute forecast error variance decomposition ("fevd")

        Returns
        -------
        fevd : FEVD instance
        """
        return FEVD(self, P=var_decomp, periods=periods)

    def reorder(self, order):
        """Reorder variables for structural specification"""
        if len(order) != len(self.params[0, :]):
            raise ValueError(
                "Reorder specification length should match "
                "number of endogenous variables"
            )
        # This converts order to list of integers if given as strings
        if isinstance(order[0], str):
            order_new = []
            for i, nam in enumerate(order):
                order_new.append(self.names.index(order[i]))
            order = order_new
        return _reordered(self, order)

    # -----------------------------------------------------------
    # VAR Diagnostics: Granger-causality, whiteness of residuals,
    #                  normality, etc

    def test_causality(self, caused, causing=None, kind="f", signif=0.05):
        """
        Test Granger causality

        Parameters
        ----------
        caused : int or str or sequence of int or str
            If int or str, test whether the variable specified via this index
            (int) or name (str) is Granger-caused by the variable(s) specified
            by `causing`.
            If a sequence of int or str, test whether the corresponding
            variables are Granger-caused by the variable(s) specified
            by `causing`.
        causing : int or str or sequence of int or str or None, default: None
            If int or str, test whether the variable specified via this index
            (int) or name (str) is Granger-causing the variable(s) specified by
            `caused`.
            If a sequence of int or str, test whether the corresponding
            variables are Granger-causing the variable(s) specified by
            `caused`.
            If None, `causing` is assumed to be the complement of `caused`.
        kind : {'f', 'wald'}
            Perform F-test or Wald (chi-sq) test
        signif : float, default 5%
            Significance level for computing critical values for test,
            defaulting to standard 0.05 level

        Notes
        -----
        Null hypothesis is that there is no Granger-causality for the indicated
        variables. The degrees of freedom in the F-test are based on the
        number of variables in the VAR system, that is, degrees of freedom
        are equal to the number of equations in the VAR times degree of freedom
        of a single equation.

        Test for Granger-causality as described in chapter 7.6.3 of [1]_.
        Test H0: "`causing` does not Granger-cause the remaining variables of
        the system" against  H1: "`causing` is Granger-causal for the
        remaining variables".

        Returns
        -------
        results : CausalityTestResults

        References
        ----------
        .. [1] Lütkepohl, H. 2005. *New Introduction to Multiple Time Series*
           *Analysis*. Springer.
        """
        if not (0 < signif < 1):
            raise ValueError("signif has to be between 0 and 1")

        allowed_types = (str, int)

        if isinstance(caused, allowed_types):
            caused = [caused]
        if not all(isinstance(c, allowed_types) for c in caused):
            raise TypeError(
                "caused has to be of type string or int (or a "
                "sequence of these types)."
            )
        caused = [self.names[c] if type(c) is int else c for c in caused]
        caused_ind = [util.get_index(self.names, c) for c in caused]

        if causing is not None:

            if isinstance(causing, allowed_types):
                causing = [causing]
            if not all(isinstance(c, allowed_types) for c in causing):
                raise TypeError(
                    "causing has to be of type string or int (or "
                    "a sequence of these types) or None."
                )
            causing = [self.names[c] if type(c) is int else c for c in causing]
            causing_ind = [util.get_index(self.names, c) for c in causing]
        else:
            causing_ind = [i for i in range(self.neqs) if i not in caused_ind]
            causing = [self.names[c] for c in caused_ind]

        k, p = self.neqs, self.k_ar
        if p == 0:
            err = "Cannot test Granger Causality in a model with 0 lags."
            raise RuntimeError(err)

        # number of restrictions
        num_restr = len(causing) * len(caused) * p
        num_det_terms = self.k_exog

        # Make restriction matrix
        C = np.zeros((num_restr, k * num_det_terms + k ** 2 * p), dtype=float)
        cols_det = k * num_det_terms
        row = 0
        for j in range(p):
            for ing_ind in causing_ind:
                for ed_ind in caused_ind:
                    C[row, cols_det + ed_ind + k * ing_ind + k ** 2 * j] = 1
                    row += 1

        # Lütkepohl 3.6.5
        Cb = np.dot(C, vec(self.params.T))
        middle = np.linalg.inv(C @ self.cov_params() @ C.T)

        # wald statistic
        lam_wald = statistic = Cb @ middle @ Cb

        if kind.lower() == "wald":
            df = num_restr
            dist = stats.chi2(df)
        elif kind.lower() == "f":
            statistic = lam_wald / num_restr
            df = (num_restr, k * self.df_resid)
            dist = stats.f(*df)
        else:
            raise ValueError("kind %s not recognized" % kind)

        pvalue = dist.sf(statistic)
        crit_value = dist.ppf(1 - signif)

        return CausalityTestResults(
            causing,
            caused,
            statistic,
            crit_value,
            pvalue,
            df,
            signif,
            test="granger",
            method=kind,
        )

    def test_inst_causality(self, causing, signif=0.05):
        """
        Test for instantaneous causality

        Parameters
        ----------
        causing :
            If int or str, test whether the corresponding variable is causing
            the variable(s) specified in caused.
            If sequence of int or str, test whether the corresponding
            variables are causing the variable(s) specified in caused.
        signif : float between 0 and 1, default 5 %
            Significance level for computing critical values for test,
            defaulting to standard 0.05 level
        verbose : bool
            If True, print a table with the results.

        Returns
        -------
        results : dict
            A dict holding the test's results. The dict's keys are:

            "statistic" : float
              The calculated test statistic.

            "crit_value" : float
              The critical value of the Chi^2-distribution.

            "pvalue" : float
              The p-value corresponding to the test statistic.

            "df" : float
              The degrees of freedom of the Chi^2-distribution.

            "conclusion" : str {"reject", "fail to reject"}
              Whether H0 can be rejected or not.

            "signif" : float
              Significance level

        Notes
        -----
        Test for instantaneous causality as described in chapters 3.6.3 and
        7.6.4 of [1]_.
        Test H0: "No instantaneous causality between caused and causing"
        against H1: "Instantaneous causality between caused and causing
        exists".

        Instantaneous causality is a symmetric relation (i.e. if causing is
        "instantaneously causing" caused, then also caused is "instantaneously
        causing" causing), thus the naming of the parameters (which is chosen
        to be in accordance with test_granger_causality()) may be misleading.

        This method is not returning the same result as JMulTi. This is
        because the test is based on a VAR(k_ar) model in statsmodels
        (in accordance to pp. 104, 320-321 in [1]_) whereas JMulTi seems
        to be using a VAR(k_ar+1) model.

        References
        ----------
        .. [1] Lütkepohl, H. 2005. *New Introduction to Multiple Time Series*
           *Analysis*. Springer.
        """
        if not (0 < signif < 1):
            raise ValueError("signif has to be between 0 and 1")

        allowed_types = (str, int)
        if isinstance(causing, allowed_types):
            causing = [causing]
        if not all(isinstance(c, allowed_types) for c in causing):
            raise TypeError(
                "causing has to be of type string or int (or a "
                + "a sequence of these types)."
            )
        causing = [self.names[c] if type(c) is int else c for c in causing]
        causing_ind = [util.get_index(self.names, c) for c in causing]

        caused_ind = [i for i in range(self.neqs) if i not in causing_ind]
        caused = [self.names[c] for c in caused_ind]

        # Note: JMulTi seems to be using k_ar+1 instead of k_ar
        k, t, p = self.neqs, self.nobs, self.k_ar

        num_restr = len(causing) * len(caused)  # called N in Lütkepohl

        sigma_u = self.sigma_u
        vech_sigma_u = util.vech(sigma_u)
        sig_mask = np.zeros(sigma_u.shape)
        # set =1 twice to ensure, that all the ones needed are below the main
        # diagonal:
        sig_mask[causing_ind, caused_ind] = 1
        sig_mask[caused_ind, causing_ind] = 1
        vech_sig_mask = util.vech(sig_mask)
        inds = np.nonzero(vech_sig_mask)[0]

        # Make restriction matrix
        C = np.zeros((num_restr, len(vech_sigma_u)), dtype=float)
        for row in range(num_restr):
            C[row, inds[row]] = 1
        Cs = np.dot(C, vech_sigma_u)
        d = np.linalg.pinv(duplication_matrix(k))
        Cd = np.dot(C, d)
        middle = np.linalg.inv(Cd @ np.kron(sigma_u, sigma_u) @ Cd.T) / 2

        wald_statistic = t * (Cs.T @ middle @ Cs)
        df = num_restr
        dist = stats.chi2(df)

        pvalue = dist.sf(wald_statistic)
        crit_value = dist.ppf(1 - signif)

        return CausalityTestResults(
            causing,
            caused,
            wald_statistic,
            crit_value,
            pvalue,
            df,
            signif,
            test="inst",
            method="wald",
        )

    def test_whiteness(self, nlags=10, signif=0.05, adjusted=False):
        """
        Residual whiteness tests using Portmanteau test

        Parameters
        ----------
        nlags : int > 0
            The number of lags tested must be larger than the number of lags
            included in the VAR model.
        signif : float, between 0 and 1
            The significance level of the test.
        adjusted : bool, default False
            Flag indicating to apply small-sample adjustments.

        Returns
        -------
        WhitenessTestResults
            The test results.

        Notes
        -----
        Test the whiteness of the residuals using the Portmanteau test as
        described in [1]_, chapter 4.4.3.

        References
        ----------
        .. [1] Lütkepohl, H. 2005. *New Introduction to Multiple Time Series*
           *Analysis*. Springer.
        """
        if nlags - self.k_ar <= 0:
            raise ValueError(
                "The whiteness test can only be used when nlags "
                "is larger than the number of lags included in "
                f"the model ({self.k_ar})."
            )
        statistic = 0
        u = np.asarray(self.resid)
        acov_list = _compute_acov(u, nlags)
        cov0_inv = np.linalg.inv(acov_list[0])
        for t in range(1, nlags + 1):
            ct = acov_list[t]
            to_add = np.trace(ct.T @ cov0_inv @ ct @ cov0_inv)
            if adjusted:
                to_add /= self.nobs - t
            statistic += to_add
        statistic *= self.nobs ** 2 if adjusted else self.nobs
        df = self.neqs ** 2 * (nlags - self.k_ar)
        dist = stats.chi2(df)
        pvalue = dist.sf(statistic)
        crit_value = dist.ppf(1 - signif)

        return WhitenessTestResults(
            statistic, crit_value, pvalue, df, signif, nlags, adjusted
        )

    def plot_acorr(self, nlags=10, resid=True, linewidth=8):
        r"""
        Plot autocorrelation of sample (endog) or residuals

        Sample (Y) or Residual autocorrelations are plotted together with the
        standard :math:`2 / \sqrt{T}` bounds.

        Parameters
        ----------
        nlags : int
            number of lags to display (excluding 0)
        resid : bool
            If True, then the autocorrelation of the residuals is plotted
            If False, then the autocorrelation of endog is plotted.
        linewidth : int
            width of vertical bars

        Returns
        -------
        Figure
            Figure instance containing the plot.
        """
        if resid:
            acorrs = self.resid_acorr(nlags)
        else:
            acorrs = self.sample_acorr(nlags)

        bound = 2 / np.sqrt(self.nobs)

        fig = plotting.plot_full_acorr(
            acorrs[1:],
            xlabel=np.arange(1, nlags + 1),
            err_bound=bound,
            linewidth=linewidth,
        )
        fig.suptitle(r"ACF plots for residuals with $2 / \sqrt{T}$ bounds ")
        return fig

    def test_normality(self, signif=0.05):
        """
        Test assumption of normal-distributed errors using Jarque-Bera-style
        omnibus Chi^2 test.

        Parameters
        ----------
        signif : float
            Test significance level.

        Returns
        -------
        result : NormalityTestResults

        Notes
        -----
        H0 (null) : data are generated by a Gaussian-distributed process
        """
        return test_normality(self, signif=signif)

    @cache_readonly
    def detomega(self):
        r"""
        Return determinant of white noise covariance with degrees of freedom
        correction:

        .. math::

            \hat \Omega = \frac{T}{T - Kp - 1} \hat \Omega_{\mathrm{MLE}}
        """
        return np.linalg.det(self.sigma_u)

    @cache_readonly
    def info_criteria(self):
        "information criteria for lagorder selection"
        nobs = self.nobs
        neqs = self.neqs
        lag_order = self.k_ar
        free_params = lag_order * neqs ** 2 + neqs * self.k_exog
        if self.df_resid:
            ld = logdet_symm(self.sigma_u_mle)
        else:
            ld = -np.inf

        # See Lütkepohl pp. 146-150

        aic = ld + (2.0 / nobs) * free_params
        bic = ld + (np.log(nobs) / nobs) * free_params
        hqic = ld + (2.0 * np.log(np.log(nobs)) / nobs) * free_params
        if self.df_resid:
            fpe = ((nobs + self.df_model) / self.df_resid) ** neqs * np.exp(ld)
        else:
            fpe = np.inf

        return {"aic": aic, "bic": bic, "hqic": hqic, "fpe": fpe}

    @property
    def aic(self):
        """Akaike information criterion"""
        return self.info_criteria["aic"]

    @property
    def fpe(self):
        """Final Prediction Error (FPE)

        Lütkepohl p. 147, see info_criteria
        """
        return self.info_criteria["fpe"]

    @property
    def hqic(self):
        """Hannan-Quinn criterion"""
        return self.info_criteria["hqic"]

    @property
    def bic(self):
        """Bayesian a.k.a. Schwarz info criterion"""
        return self.info_criteria["bic"]

    @cache_readonly
    def roots(self):
        """
        The roots of the VAR process are the solution to
        (I - coefs[0]*z - coefs[1]*z**2 ... - coefs[p-1]*z**k_ar) = 0.
        Note that the inverse roots are returned, and stability requires that
        the roots lie outside the unit circle.
        """
        neqs = self.neqs
        k_ar = self.k_ar
        p = neqs * k_ar
        arr = np.zeros((p, p))
        arr[:neqs, :] = np.column_stack(self.coefs)
        arr[neqs:, :-neqs] = np.eye(p - neqs)
        roots = np.linalg.eig(arr)[0] ** -1
        idx = np.argsort(np.abs(roots))[::-1]  # sort by reverse modulus
        return roots[idx]


class VARResultsWrapper(wrap.ResultsWrapper):
    _attrs = {
        "bse": "columns_eq",
        "cov_params": "cov",
        "params": "columns_eq",
        "pvalues": "columns_eq",
        "tvalues": "columns_eq",
        "sigma_u": "cov_eq",
        "sigma_u_mle": "cov_eq",
        "stderr": "columns_eq",
    }
    _wrap_attrs = wrap.union_dicts(
        TimeSeriesResultsWrapper._wrap_attrs, _attrs
    )
    _methods = {"conf_int": "multivariate_confint"}
    _wrap_methods = wrap.union_dicts(
        TimeSeriesResultsWrapper._wrap_methods, _methods
    )


wrap.populate_wrapper(VARResultsWrapper, VARResults)  # noqa:E305


class FEVD:
    """
    Compute and plot Forecast error variance decomposition and asymptotic
    standard errors
    """

    def __init__(self, model, P=None, periods=None):
        self.periods = periods

        self.model = model
        self.neqs = model.neqs
        self.names = model.model.endog_names

        self.irfobj = model.irf(var_decomp=P, periods=periods)
        self.orth_irfs = self.irfobj.orth_irfs

        # cumulative impulse responses
        irfs = (self.orth_irfs[:periods] ** 2).cumsum(axis=0)

        rng = lrange(self.neqs)
        mse = self.model.mse(periods)[:, rng, rng]

        # lag x equation x component
        fevd = np.empty_like(irfs)

        for i in range(periods):
            fevd[i] = (irfs[i].T / mse[i]).T

        # switch to equation x lag x component
        self.decomp = fevd.swapaxes(0, 1)

    def summary(self):
        buf = StringIO()

        rng = lrange(self.periods)
        for i in range(self.neqs):
            ppm = output.pprint_matrix(self.decomp[i], rng, self.names)

            buf.write("FEVD for %s\n" % self.names[i])
            buf.write(ppm + "\n")

        print(buf.getvalue())

    def cov(self):
        """Compute asymptotic standard errors

        Returns
        -------
        """
        raise NotImplementedError

    def plot(self, periods=None, figsize=(10, 10), **plot_kwds):
        """Plot graphical display of FEVD

        Parameters
        ----------
        periods : int, default None
            Defaults to number originally specified. Can be at most that number
        """
        import matplotlib.pyplot as plt

        k = self.neqs
        periods = periods or self.periods

        fig, axes = plt.subplots(nrows=k, figsize=figsize)

        fig.suptitle("Forecast error variance decomposition (FEVD)")

        colors = [str(c) for c in np.arange(k, dtype=float) / k]
        ticks = np.arange(periods)

        limits = self.decomp.cumsum(2)
        ax = axes[0]
        for i in range(k):
            ax = axes[i]

            this_limits = limits[i].T

            handles = []

            for j in range(k):
                lower = this_limits[j - 1] if j > 0 else 0
                upper = this_limits[j]
                handle = ax.bar(
                    ticks,
                    upper - lower,
                    bottom=lower,
                    color=colors[j],
                    label=self.names[j],
                    **plot_kwds,
                )

                handles.append(handle)

            ax.set_title(self.names[i])

        # just use the last axis to get handles for plotting
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
        plotting.adjust_subplots(right=0.85)
        return fig


# -------------------------------------------------------------------------------


def _compute_acov(x, nlags=1):
    x = x - x.mean(0)

    result = []
    for lag in range(nlags + 1):
        if lag > 0:
            r = np.dot(x[lag:].T, x[:-lag])
        else:
            r = np.dot(x.T, x)

        result.append(r)

    return np.array(result) / len(x)


def _acovs_to_acorrs(acovs):
    sd = np.sqrt(np.diag(acovs[0]))
    return acovs / np.outer(sd, sd)
