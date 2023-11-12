"""
Tests for recursive least squares models

Author: Chad Fulton
License: Simplified-BSD
"""
import os

import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_raises
import pandas as pd
import pytest
from scipy.stats import norm

from statsmodels.datasets import macrodata
from statsmodels.genmod.api import GLM
from statsmodels.regression.linear_model import OLS
from statsmodels.regression.recursive_ls import RecursiveLS
from statsmodels.stats.diagnostic import recursive_olsresiduals
from statsmodels.tools import add_constant
from statsmodels.tools.eval_measures import aic, bic
from statsmodels.tools.sm_exceptions import ValueWarning

current_path = os.path.dirname(os.path.abspath(__file__))

results_R_path = 'results' + os.sep + 'results_rls_R.csv'
results_R = pd.read_csv(current_path + os.sep + results_R_path)

results_stata_path = 'results' + os.sep + 'results_rls_stata.csv'
results_stata = pd.read_csv(current_path + os.sep + results_stata_path)

dta = macrodata.load_pandas().data
dta.index = pd.date_range(start='1959-01-01', end='2009-07-01', freq='QS')

endog = dta['cpi']
exog = add_constant(dta['m1'])


def test_endog():
    # Tests for numpy input
    mod = RecursiveLS(endog.values, exog.values)
    res = mod.fit()

    # Test the RLS estimates against OLS estimates
    mod_ols = OLS(endog, exog)
    res_ols = mod_ols.fit()
    assert_allclose(res.params, res_ols.params)

    # Tests for 1-dim exog
    mod = RecursiveLS(endog, dta['m1'].values)
    res = mod.fit()

    # Test the RLS estimates against OLS estimates
    mod_ols = OLS(endog, dta['m1'])
    res_ols = mod_ols.fit()
    assert_allclose(res.params, res_ols.params)

def test_ols():
    # More comprehensive tests against OLS estimates
    mod = RecursiveLS(endog, dta['m1'])
    res = mod.fit()

    mod_ols = OLS(endog, dta['m1'])
    res_ols = mod_ols.fit()

    # Regression coefficients, standard errors, and estimated scale
    assert_allclose(res.params, res_ols.params)
    assert_allclose(res.bse, res_ols.bse)
    # Note: scale here is computed according to Harvey, 1989, 4.2.5, and is
    # the called the ML estimator and sometimes (e.g. later in section 5)
    # denoted \tilde \sigma_*^2
    assert_allclose(res.filter_results.obs_cov[0, 0], res_ols.scale)

    # OLS residuals are equivalent to smoothed forecast errors
    # (the latter are defined as e_t|T by Harvey, 1989, 5.4.5)
    # (this follows since the smoothed state simply contains the
    # full-information estimates of the regression coefficients)
    actual = (mod.endog[:, 0] -
              np.sum(mod['design', 0, :, :] * res.smoothed_state, axis=0))
    assert_allclose(actual, res_ols.resid)

    # Given the estimate of scale as `sum(v_t^2 / f_t) / (T - d)` (see
    # Harvey, 1989, 4.2.5 on p. 183), then llf_recursive is equivalent to the
    # full OLS loglikelihood (i.e. without the scale concentrated out).
    desired = mod_ols.loglike(res_ols.params, scale=res_ols.scale)
    assert_allclose(res.llf_recursive, desired)
    # Alternatively, we can constrcut the concentrated OLS loglikelihood
    # by computing the scale term with `nobs` in the denominator rather than
    # `nobs - d`.
    scale_alternative = np.sum((
        res.standardized_forecasts_error[0, 1:] *
        res.filter_results.obs_cov[0, 0]**0.5)**2) / mod.nobs
    llf_alternative = np.log(norm.pdf(res.resid_recursive, loc=0,
                                      scale=scale_alternative**0.5)).sum()
    assert_allclose(llf_alternative, res_ols.llf)

    # Prediction
    actual = res.forecast(10, design=np.ones((1, 1, 10)))
    assert_allclose(actual, res_ols.predict(np.ones((10, 1))))

    # Sums of squares, R^2
    assert_allclose(res.ess, res_ols.ess)
    assert_allclose(res.ssr, res_ols.ssr)
    assert_allclose(res.centered_tss, res_ols.centered_tss)
    assert_allclose(res.uncentered_tss, res_ols.uncentered_tss)
    assert_allclose(res.rsquared, res_ols.rsquared)

    # Mean squares
    assert_allclose(res.mse_model, res_ols.mse_model)
    assert_allclose(res.mse_resid, res_ols.mse_resid)
    assert_allclose(res.mse_total, res_ols.mse_total)

    # Hypothesis tests
    actual = res.t_test('m1 = 0')
    desired = res_ols.t_test('m1 = 0')
    assert_allclose(actual.statistic, desired.statistic)
    assert_allclose(actual.pvalue, desired.pvalue, atol=1e-15)

    actual = res.f_test('m1 = 0')
    desired = res_ols.f_test('m1 = 0')
    assert_allclose(actual.statistic, desired.statistic)
    assert_allclose(actual.pvalue, desired.pvalue, atol=1e-15)

    # Information criteria
    # Note: the llf and llf_obs given in the results are based on the Kalman
    # filter and so the ic given in results will not be identical to the
    # OLS versions. Additionally, llf_recursive is comparable to the
    # non-concentrated llf, and not the concentrated llf that is by default
    # used in OLS. Compute new ic based on llf_alternative to compare.
    actual_aic = aic(llf_alternative, res.nobs_effective, res.df_model)
    assert_allclose(actual_aic, res_ols.aic)
    actual_bic = bic(llf_alternative, res.nobs_effective, res.df_model)
    assert_allclose(actual_bic, res_ols.bic)


def test_glm(constraints=None):
    # More comprehensive tests against GLM estimates (this is sort of redundant
    # given `test_ols`, but this is mostly to complement the tests in
    # `test_glm_constrained`)
    endog = dta.infl
    exog = add_constant(dta[['unemp', 'm1']])

    mod = RecursiveLS(endog, exog, constraints=constraints)
    res = mod.fit()

    mod_glm = GLM(endog, exog)
    if constraints is None:
        res_glm = mod_glm.fit()
    else:
        res_glm = mod_glm.fit_constrained(constraints=constraints)

    # Regression coefficients, standard errors, and estimated scale
    assert_allclose(res.params, res_glm.params)
    assert_allclose(res.bse, res_glm.bse, atol=1e-6)
    # Note: scale here is computed according to Harvey, 1989, 4.2.5, and is
    # the called the ML estimator and sometimes (e.g. later in section 5)
    # denoted \tilde \sigma_*^2
    assert_allclose(res.filter_results.obs_cov[0, 0], res_glm.scale)

    # DoF
    # Note: GLM does not include intercept in DoF, so modify by -1
    assert_equal(res.df_model - 1, res_glm.df_model)

    # OLS residuals are equivalent to smoothed forecast errors
    # (the latter are defined as e_t|T by Harvey, 1989, 5.4.5)
    # (this follows since the smoothed state simply contains the
    # full-information estimates of the regression coefficients)
    actual = (mod.endog[:, 0] -
              np.sum(mod['design', 0, :, :] * res.smoothed_state, axis=0))
    assert_allclose(actual, res_glm.resid_response, atol=1e-7)

    # Given the estimate of scale as `sum(v_t^2 / f_t) / (T - d)` (see
    # Harvey, 1989, 4.2.5 on p. 183), then llf_recursive is equivalent to the
    # full OLS loglikelihood (i.e. without the scale concentrated out).
    desired = mod_glm.loglike(res_glm.params, scale=res_glm.scale)
    assert_allclose(res.llf_recursive, desired)
    # Alternatively, we can construct the concentrated OLS loglikelihood
    # by computing the scale term with `nobs` in the denominator rather than
    # `nobs - d`.
    scale_alternative = np.sum((
        res.standardized_forecasts_error[0, 1:] *
        res.filter_results.obs_cov[0, 0]**0.5)**2) / mod.nobs
    llf_alternative = np.log(norm.pdf(res.resid_recursive, loc=0,
                                      scale=scale_alternative**0.5)).sum()
    assert_allclose(llf_alternative, res_glm.llf)

    # Prediction
    # TODO: prediction in this case is not working.
    if constraints is None:
        design = np.ones((1, 3, 10))
        actual = res.forecast(10, design=design)
        assert_allclose(actual, res_glm.predict(np.ones((10, 3))))
    else:
        design = np.ones((2, 3, 10))
        assert_raises(NotImplementedError, res.forecast, 10, design=design)

    # Hypothesis tests
    actual = res.t_test('m1 = 0')
    desired = res_glm.t_test('m1 = 0')
    assert_allclose(actual.statistic, desired.statistic)
    assert_allclose(actual.pvalue, desired.pvalue, atol=1e-15)

    actual = res.f_test('m1 = 0')
    desired = res_glm.f_test('m1 = 0')
    assert_allclose(actual.statistic, desired.statistic)
    assert_allclose(actual.pvalue, desired.pvalue)

    # Information criteria
    # Note: the llf and llf_obs given in the results are based on the Kalman
    # filter and so the ic given in results will not be identical to the
    # OLS versions. Additionally, llf_recursive is comparable to the
    # non-concentrated llf, and not the concentrated llf that is by default
    # used in OLS. Compute new ic based on llf_alternative to compare.
    actual_aic = aic(llf_alternative, res.nobs_effective, res.df_model)
    assert_allclose(actual_aic, res_glm.aic)
    # See gh#1733 for details on why the BIC does not match while AIC does
    # actual_bic = bic(llf_alternative, res.nobs_effective, res.df_model)
    # assert_allclose(actual_bic, res_glm.bic)

def test_glm_constrained():
    test_glm(constraints='m1 + unemp = 1')


def test_filter():
    # Basic test for filtering
    mod = RecursiveLS(endog, exog)
    res = mod.filter()

    # Test the RLS estimates against OLS estimates
    mod_ols = OLS(endog, exog)
    res_ols = mod_ols.fit()
    assert_allclose(res.params, res_ols.params)


def test_estimates():
    mod = RecursiveLS(endog, exog)
    res = mod.fit()

    # Test for start_params
    assert_equal(mod.start_params, 0)


    # Test the RLS coefficient estimates against those from R (quantreg)
    # Due to initialization issues, we get more agreement as we get
    # farther from the initial values.
    assert_allclose(res.recursive_coefficients.filtered[:, 2:10].T,
                    results_R.iloc[:8][['beta1', 'beta2']], rtol=1e-5)
    assert_allclose(res.recursive_coefficients.filtered[:, 9:20].T,
                    results_R.iloc[7:18][['beta1', 'beta2']])
    assert_allclose(res.recursive_coefficients.filtered[:, 19:].T,
                    results_R.iloc[17:][['beta1', 'beta2']])

    # Test the RLS estimates against OLS estimates
    mod_ols = OLS(endog, exog)
    res_ols = mod_ols.fit()
    assert_allclose(res.params, res_ols.params)


@pytest.mark.matplotlib
def test_plots(close_figures):
    exog = add_constant(dta[['m1', 'pop']])
    mod = RecursiveLS(endog, exog)
    res = mod.fit()

    # Basic plot
    try:
        from pandas.plotting import register_matplotlib_converters
        register_matplotlib_converters()
    except ImportError:
        pass
    fig = res.plot_recursive_coefficient()

    # Specific variable
    fig = res.plot_recursive_coefficient(variables=['m1'])

    # All variables
    fig = res.plot_recursive_coefficient(variables=[0, 'm1', 'pop'])

    # Basic plot
    fig = res.plot_cusum()

    # Other alphas
    for alpha in [0.01, 0.10]:
        fig = res.plot_cusum(alpha=alpha)

    # Invalid alpha
    assert_raises(ValueError, res.plot_cusum, alpha=0.123)

    # Basic plot
    fig = res.plot_cusum_squares()

    # Numpy input (no dates)
    mod = RecursiveLS(endog.values, exog.values)
    res = mod.fit()

    # Basic plot
    fig = res.plot_recursive_coefficient()

    # Basic plot
    fig = res.plot_cusum()

    # Basic plot
    fig = res.plot_cusum_squares()


def test_from_formula():
    with pytest.warns(ValueWarning, match="No frequency information"):
        mod = RecursiveLS.from_formula('cpi ~ m1', data=dta)

    res = mod.fit()

    # Test the RLS estimates against OLS estimates
    mod_ols = OLS.from_formula('cpi ~ m1', data=dta)
    res_ols = mod_ols.fit()
    assert_allclose(res.params, res_ols.params)


def test_resid_recursive():
    mod = RecursiveLS(endog, exog)
    res = mod.fit()

    # Test the recursive residuals against those from R (strucchange)
    assert_allclose(res.resid_recursive[2:10].T,
                    results_R.iloc[:8]['rec_resid'])
    assert_allclose(res.resid_recursive[9:20].T,
                    results_R.iloc[7:18]['rec_resid'])
    assert_allclose(res.resid_recursive[19:].T,
                    results_R.iloc[17:]['rec_resid'])

    # Test the RLS estimates against those from Stata (cusum6)
    assert_allclose(res.resid_recursive[3:],
                    results_stata.iloc[3:]['rr'], atol=1e-5, rtol=1e-5)

    # Test the RLS estimates against statsmodels estimates
    mod_ols = OLS(endog, exog)
    res_ols = mod_ols.fit()
    desired_resid_recursive = recursive_olsresiduals(res_ols)[4][2:]
    assert_allclose(res.resid_recursive[2:], desired_resid_recursive)


def test_recursive_olsresiduals_bad_input(reset_randomstate):
    from statsmodels.tsa.arima.model import ARIMA
    e = np.random.standard_normal(250)
    y = e.copy()
    for i in range(1, y.shape[0]):
        y[i] += 0.1 + 0.8 * y[i - 1] + e[i]
    res = ARIMA(y[20:], order=(1,0,0), trend="c").fit()
    with pytest.raises(TypeError, match="res a regression results instance"):
        recursive_olsresiduals(res)


def test_cusum():
    mod = RecursiveLS(endog, exog)
    res = mod.fit()

    # Test the cusum statistics against those from R (strucchange)
    # These values are not even close to ours, to Statas, or to the alternate
    # statsmodels values
    # assert_allclose(res.cusum, results_R['cusum'])

    # Test the cusum statistics against Stata (cusum6)
    # Note: cusum6 excludes the first 3 elements due to OLS initialization
    # whereas we exclude only the first 2. Also there are initialization
    # differences (as seen above in the recursive residuals).
    # Here we explicitly reverse engineer our cusum to match their to show the
    # equivalence
    d = res.nobs_diffuse
    cusum = res.cusum * np.std(res.resid_recursive[d:], ddof=1)
    cusum -= res.resid_recursive[d]
    cusum /= np.std(res.resid_recursive[d+1:], ddof=1)
    cusum = cusum[1:]
    assert_allclose(cusum, results_stata.iloc[3:]['cusum'], atol=1e-6, rtol=1e-5)

    # Test the cusum statistics against statsmodels estimates
    mod_ols = OLS(endog, exog)
    res_ols = mod_ols.fit()
    desired_cusum = recursive_olsresiduals(res_ols)[-2][1:]
    assert_allclose(res.cusum, desired_cusum, rtol=1e-6)

    # Test the cusum bounds against Stata (cusum6)
    # Again note that cusum6 excludes the first 3 elements, so we need to
    # change the ddof and points.
    actual_bounds = res._cusum_significance_bounds(
        alpha=0.05, ddof=1, points=np.arange(d+1, res.nobs))
    desired_bounds = results_stata.iloc[3:][['lw', 'uw']].T
    assert_allclose(actual_bounds, desired_bounds, rtol=1e-6)

    # Test the cusum bounds against statsmodels
    actual_bounds = res._cusum_significance_bounds(
        alpha=0.05, ddof=0, points=np.arange(d, res.nobs))
    desired_bounds = recursive_olsresiduals(res_ols)[-1]
    assert_allclose(actual_bounds, desired_bounds)

    # Test for invalid calls
    assert_raises(ValueError, res._cusum_squares_significance_bounds,
                  alpha=0.123)


def test_stata():
    # Test the cusum and cusumsq statistics against Stata (cusum6)
    mod = RecursiveLS(endog, exog, loglikelihood_burn=3)
    with pytest.warns(UserWarning):
        res = mod.fit()
    d = max(res.nobs_diffuse, res.loglikelihood_burn)

    assert_allclose(res.resid_recursive[3:], results_stata.iloc[3:]['rr'],
                    atol=1e-5, rtol=1e-5)
    assert_allclose(res.cusum, results_stata.iloc[3:]['cusum'], atol=1e-5)
    assert_allclose(res.cusum_squares, results_stata.iloc[3:]['cusum2'],
                    atol=1e-5)

    actual_bounds = res._cusum_significance_bounds(
        alpha=0.05, ddof=0, points=np.arange(d+1, res.nobs+1))
    desired_bounds = results_stata.iloc[3:][['lw', 'uw']].T
    assert_allclose(actual_bounds, desired_bounds, atol=1e-5)

    # Note: Stata uses a set of tabulated critical values whereas we use an
    # approximation formula, so this test is quite imprecise
    actual_bounds = res._cusum_squares_significance_bounds(
        alpha=0.05, points=np.arange(d+1, res.nobs+1))
    desired_bounds = results_stata.iloc[3:][['lww', 'uww']].T
    assert_allclose(actual_bounds, desired_bounds, atol=1e-2)


def test_constraints_stata():
    endog = dta['infl']
    exog = add_constant(dta[['m1', 'unemp']])

    mod = RecursiveLS(endog, exog, constraints='m1 + unemp = 1')
    res = mod.fit()

    # See tests/results/test_rls.do
    desired = [-0.7001083844336, -0.0018477514060, 1.0018477514060]
    assert_allclose(res.params, desired)

    # See tests/results/test_rls.do
    desired = [.4699552366, .0005369357, .0005369357]
    assert_allclose(res.bse[0], desired[0], atol=1e-1)
    assert_allclose(res.bse[1:], desired[1:], atol=1e-4)

    # See tests/results/test_rls.do
    desired = -534.4292052931121
    # Note that to compute what Stata reports as the llf, we need to use a
    # different denominator for estimating the scale, and then compute the
    # llf from the alternative recursive residuals
    scale_alternative = np.sum((
        res.standardized_forecasts_error[0, 1:] *
        res.filter_results.obs_cov[0, 0]**0.5)**2) / mod.nobs
    llf_alternative = np.log(norm.pdf(res.resid_recursive, loc=0,
                                      scale=scale_alternative**0.5)).sum()
    assert_allclose(llf_alternative, desired)


def test_multiple_constraints():
    endog = dta['infl']
    exog = add_constant(dta[['m1', 'unemp', 'cpi']])

    constraints = [
        'm1 + unemp = 1',
        'cpi = 0',
    ]

    mod = RecursiveLS(endog, exog, constraints=constraints)
    res = mod.fit()

    # See tests/results/test_rls.do
    desired = [-0.7001083844336, -0.0018477514060, 1.0018477514060, 0]
    assert_allclose(res.params, desired, atol=1e-10)

    # See tests/results/test_rls.do
    desired = [.4699552366, .0005369357, .0005369357, 0]
    assert_allclose(res.bse[0], desired[0], atol=1e-1)
    assert_allclose(res.bse[1:-1], desired[1:-1], atol=1e-4)

    # See tests/results/test_rls.do
    desired = -534.4292052931121
    # Note that to compute what Stata reports as the llf, we need to use a
    # different denominator for estimating the scale, and then compute the
    # llf from the alternative recursive residuals
    scale_alternative = np.sum((
        res.standardized_forecasts_error[0, 1:] *
        res.filter_results.obs_cov[0, 0]**0.5)**2) / mod.nobs
    llf_alternative = np.log(norm.pdf(res.resid_recursive, loc=0,
                                      scale=scale_alternative**0.5)).sum()
    assert_allclose(llf_alternative, desired)


def test_fix_params():
    mod = RecursiveLS([0, 1, 0, 1], [1, 1, 1, 1])
    with pytest.raises(ValueError, match=('Linear constraints on coefficients'
                                          ' should be given')):
        with mod.fix_params({'const': 0.1}):
            mod.fit()

    with pytest.raises(ValueError, match=('Linear constraints on coefficients'
                                          ' should be given')):
        mod.fit_constrained({'const': 0.1})
