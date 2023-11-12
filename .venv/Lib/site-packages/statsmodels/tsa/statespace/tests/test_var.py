"""
Tests for VAR models (via VARMAX)

These are primarily tests of VAR results from the VARMAX state space approach
compared to the output of estimation via CSS

loglikelihood_burn = k_ar is required, since the loglikelihood reported by the
CSS approach is the conditional loglikelihood.

Author: Chad Fulton
License: Simplified-BSD
"""
import os

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd

from statsmodels.tsa.statespace import varmax
from .results import results_var_R

current_path = os.path.dirname(os.path.abspath(__file__))
results_var_R_output = pd.read_csv(
    os.path.join(current_path, 'results', 'results_var_R_output.csv'))

up2 = os.path.split(os.path.split(current_path)[0])[0]
dta = pd.read_stata(os.path.join(up2, 'tests', 'results', 'lutkepohl2.dta'))
dta.index = pd.PeriodIndex(dta.qtr, freq='Q')
endog = dta[['dln_inv', 'dln_inc', 'dln_consump']].loc['1960Q2':'1978']

# Note that we start the time-trend at 0 here, to match
time_trend0 = np.arange(len(endog))
exog0 = np.c_[np.ones(len(endog)), time_trend0, time_trend0**2]
time_trend0_fcast = np.arange(len(endog), len(endog) + 10)
exog0_fcast = np.c_[np.ones(10), time_trend0_fcast, time_trend0_fcast**2]
# And here we start the time-trend at 1
time_trend1 = np.arange(1, len(endog) + 1)
time_trend1_fcast = np.arange(len(endog) + 1, len(endog) + 1 + 10)
exog1 = np.c_[np.ones(len(endog)), time_trend1, time_trend1**2]
exog1_fcast = np.c_[np.ones(10), time_trend1_fcast, time_trend1_fcast**2]


def check_irf(test, mod, results, params=None):
    # Note that 'vars' uses an estimator of Sigma_u with k_params fewer degrees
    # of freedom to compute impulse responses, so we need to reset the
    # state_cov matrix for this part of the test
    Sigma_u_mle = mod['state_cov']
    nobs_effective = mod.nobs - mod.k_ar
    df_resid = (nobs_effective -
                (mod.k_ar * mod.k_endog + mod.k_trend + mod.k_exog))
    Sigma_u = Sigma_u_mle * nobs_effective / df_resid

    L = np.linalg.cholesky(Sigma_u)
    if params is None:
        params = np.copy(results['params'])
    params[-6:] = L[np.tril_indices_from(L)]
    res = mod.smooth(params)

    for i in range(3):
        impulse_to = endog.columns[i]

        # Non-orthogonalized
        columns = ['%s.irf.%s.%s' % (test, impulse_to, name)
                   for name in endog.columns]
        assert_allclose(res.impulse_responses(10, i),
                        results_var_R_output[columns])

        # Orthogonalized
        columns = ['%s.irf.ortho.%s.%s' % (test, impulse_to, name)
                   for name in endog.columns]
        assert_allclose(res.impulse_responses(10, i, orthogonalized=True),
                        results_var_R_output[columns])

        # Orthogonalized, cumulated
        columns = ['%s.irf.cumu.%s.%s' % (test, impulse_to, name)
                   for name in endog.columns]
        result = res.impulse_responses(10, i,
                                       orthogonalized=True, cumulative=True)
        assert_allclose(result,
                        results_var_R_output[columns])


def test_var_basic():
    test = 'basic'

    # VAR(2), no trend or exog
    results = results_var_R.res_basic
    mod = varmax.VARMAX(endog, order=(2, 0), trend='n', loglikelihood_burn=2)
    res = mod.smooth(results['params'])

    assert_allclose(res.llf, results['llf'])

    # Forecast
    columns = ['%s.fcast.%s.fcst' % (test, name) for name in endog.columns]
    assert_allclose(res.forecast(10), results_var_R_output[columns].iloc[:10])

    # IRF
    check_irf(test, mod, results)

    # FEVD: TODO


def test_var_c():
    test = 'c'

    # VAR(2), constant trend, no exog
    results = results_var_R.res_c
    mod = varmax.VARMAX(endog, order=(2, 0), trend='c', loglikelihood_burn=2)
    res = mod.smooth(results['params'])

    assert_allclose(res.llf, results['llf'])

    # Forecast
    columns = ['%s.fcast.%s.fcst' % (test, name) for name in endog.columns]
    assert_allclose(res.forecast(10), results_var_R_output[columns].iloc[:10])

    # IRF
    check_irf(test, mod, results)

    # FEVD: TODO


def test_var_ct():
    test = 'ct'

    # VAR(2), constant and time trend, no exog
    results = results_var_R.res_ct
    mod = varmax.VARMAX(endog, order=(2, 0), trend='ct', loglikelihood_burn=2)
    res = mod.smooth(results['params'])

    assert_allclose(res.llf, results['llf'])

    # Forecast
    columns = ['%s.fcast.%s.fcst' % (test, name) for name in endog.columns]
    assert_allclose(res.forecast(10), results_var_R_output[columns].iloc[:10])

    # IRF
    check_irf(test, mod, results)

    # FEVD: TODO


def test_var_ct_as_exog0():
    test = 'ct_as_exog0'

    # VAR(2), no built-in trend, constant and time trend as exog
    # Here we start the time-trend at 0
    results = results_var_R.res_ct_as_exog0
    mod = varmax.VARMAX(endog, order=(2, 0), exog=exog0[:, :2], trend='n',
                        loglikelihood_burn=2)
    res = mod.smooth(results['params'])

    assert_allclose(res.llf, results['llf'])

    # Forecast
    columns = ['%s.fcast.%s.fcst' % (test, name) for name in endog.columns]
    assert_allclose(res.forecast(10, exog=exog0_fcast[:, :2]),
                    results_var_R_output[columns].iloc[:10])

    # IRF
    check_irf(test, mod, results)

    # FEVD: TODO


def test_var_ct_as_exog1():
    test = 'ct'

    # VAR(2), no built-in trend, constant and time trend as exog
    # Here we start the time-trend at 1 and so we can compare to the built-in
    # trend results "res_ct"
    results = results_var_R.res_ct
    mod = varmax.VARMAX(endog, order=(2, 0), exog=exog1[:, :2], trend='n',
                        loglikelihood_burn=2)
    # Since the params were given for the built-in trend case, we need to
    # re-order them
    params = results['params']
    params = np.r_[params[6:-6], params[:6], params[-6:]]
    res = mod.smooth(params)

    assert_allclose(res.llf, results['llf'])

    # Forecast
    columns = ['%s.fcast.%s.fcst' % (test, name) for name in endog.columns]
    assert_allclose(res.forecast(10, exog=exog1_fcast[:, :2]),
                    results_var_R_output[columns].iloc[:10])

    # IRF
    check_irf(test, mod, results, params)

    # FEVD: TODO


def test_var_ctt():
    test = 'ctt_as_exog1'

    # VAR(2), constant, trend, and trend**2, no exog
    # Note that this is comparing against trend as exog in the R package,
    # since it does not have a built-in option for trend**2
    results = results_var_R.res_ctt_as_exog1
    mod = varmax.VARMAX(endog, order=(2, 0), trend='ctt',
                        loglikelihood_burn=2)
    params = results['params']
    params = np.r_[params[-(6+9):-6], params[:-(6+9)], params[-6:]]
    res = mod.smooth(params)

    assert_allclose(res.llf, results['llf'])

    # Forecast
    columns = ['%s.fcast.%s.fcst' % (test, name) for name in endog.columns]
    assert_allclose(res.forecast(10), results_var_R_output[columns].iloc[:10])

    # IRF
    check_irf(test, mod, results, params)

    # FEVD: TODO


def test_var_ct_exog():
    test = 'ct_exog'

    # VAR(2), constant and trend, 'inc' variable as exog
    results = results_var_R.res_ct_exog
    exog = dta['inc'].loc['1960Q2':'1978']
    exog_fcast = dta[['inc']].loc['1979Q1':'1981Q2']
    mod = varmax.VARMAX(endog, order=(2, 0), exog=exog, trend='ct',
                        loglikelihood_burn=2)
    res = mod.smooth(results['params'])

    assert_allclose(res.llf, results['llf'])

    # Forecast
    columns = ['%s.fcast.%s.fcst' % (test, name) for name in endog.columns]
    assert_allclose(res.forecast(10, exog=exog_fcast),
                    results_var_R_output[columns].iloc[:10])

    # IRF
    check_irf(test, mod, results)

    # FEVD: TODO


def test_var_c_2exog():
    test = 'c_2exog'

    # VAR(2), constant, 'inc', 'inv' variables as exog
    results = results_var_R.res_c_2exog
    exog = dta[['inc', 'inv']].loc['1960Q2':'1978']
    exog_fcast = dta[['inc', 'inv']].loc['1979Q1':'1981Q2']
    mod = varmax.VARMAX(endog, order=(2, 0), exog=exog, trend='c',
                        loglikelihood_burn=2)
    res = mod.smooth(results['params'])

    assert_allclose(res.llf, results['llf'])

    # Forecast
    columns = ['%s.fcast.%s.fcst' % (test, name) for name in endog.columns]
    assert_allclose(res.forecast(10, exog=exog_fcast),
                    results_var_R_output[columns].iloc[:10])

    # IRF
    check_irf(test, mod, results)

    # FEVD: TODO
