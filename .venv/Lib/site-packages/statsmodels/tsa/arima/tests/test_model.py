"""
Tests for ARIMA model.

Tests are primarily limited to checking that the model is constructed correctly
and that it is calling the appropriate parameter estimators correctly. Tests of
correctness of parameter estimation routines are left to the individual
estimators' test functions.

Author: Chad Fulton
License: BSD-3
"""
from statsmodels.compat.platform import PLATFORM_WIN32

import io

import numpy as np
import pandas as pd
import pytest

from numpy.testing import assert_equal, assert_allclose, assert_raises, assert_

from statsmodels.datasets import macrodata
from statsmodels.tsa.arima.model import ARIMA

from statsmodels.tsa.arima.estimators.yule_walker import yule_walker
from statsmodels.tsa.arima.estimators.burg import burg
from statsmodels.tsa.arima.estimators.hannan_rissanen import hannan_rissanen
from statsmodels.tsa.arima.estimators.innovations import (
    innovations, innovations_mle)
from statsmodels.tsa.arima.estimators.statespace import statespace

dta = macrodata.load_pandas().data
dta.index = pd.date_range(start='1959-01-01', end='2009-07-01', freq='QS')


def test_default_trend():
    # Test that we are setting the trend default correctly
    endog = dta['infl'].iloc[:50]

    # Defaults when only endog is specified
    mod = ARIMA(endog)
    # with no integration, default trend a constant
    assert_equal(mod._spec_arima.trend_order, 0)
    assert_allclose(mod.exog, np.ones((mod.nobs, 1)))

    # Defaults with integrated model
    mod = ARIMA(endog, order=(0, 1, 0))
    # with no integration, default trend is none
    assert_equal(mod._spec_arima.trend_order, None)
    assert_equal(mod.exog, None)


def test_invalid():
    # Tests that invalid options raise errors
    # (note that this is only invalid options specific to `ARIMA`, and not
    # invalid options that would raise errors in SARIMAXSpecification).
    endog = dta['infl'].iloc[:50]
    mod = ARIMA(endog, order=(1, 0, 0))

    # Need valid method
    assert_raises(ValueError, mod.fit, method='not_a_method')

    # Can only use certain methods with fixed parameters
    # (e.g. 'statespace' and 'hannan-rissanen')
    with mod.fix_params({'ar.L1': 0.5}):
        assert_raises(ValueError, mod.fit, method='yule_walker')

    # Cannot override model-level values in fit
    assert_raises(ValueError, mod.fit, method='statespace', method_kwargs={
        'enforce_stationarity': False})

    # start_params only valid for MLE methods
    assert_raises(ValueError, mod.fit, method='yule_walker',
                  start_params=[0.5, 1.])

    # has_exog and gls=False with non-statespace method
    mod2 = ARIMA(endog, order=(1, 0, 0), trend='c')
    assert_raises(ValueError, mod2.fit, method='yule_walker', gls=False)

    # non-stationary parameters
    mod3 = ARIMA(np.arange(100) * 1.0, order=(1, 0, 0), trend='n')
    assert_raises(ValueError, mod3.fit, method='hannan_rissanen')

    # non-invertible parameters
    mod3 = ARIMA(np.arange(20) * 1.0, order=(0, 0, 1), trend='n')
    assert_raises(ValueError, mod3.fit, method='hannan_rissanen')


def test_yule_walker():
    # Test for basic use of Yule-Walker estimation
    endog = dta['infl'].iloc[:50]

    # AR(2), no trend (since trend would imply GLS estimation)
    desired_p, _ = yule_walker(endog, ar_order=2, demean=False)
    mod = ARIMA(endog, order=(2, 0, 0), trend='n')
    res = mod.fit(method='yule_walker')
    assert_allclose(res.params, desired_p.params)


def test_burg():
    # Test for basic use of Yule-Walker estimation
    endog = dta['infl'].iloc[:50]

    # AR(2), no trend (since trend would imply GLS estimation)
    desired_p, _ = burg(endog, ar_order=2, demean=False)
    mod = ARIMA(endog, order=(2, 0, 0), trend='n')
    res = mod.fit(method='burg')
    assert_allclose(res.params, desired_p.params)


def test_hannan_rissanen():
    # Test for basic use of Hannan-Rissanen estimation
    endog = dta['infl'].diff().iloc[1:101]

    # ARMA(1, 1), no trend (since trend would imply GLS estimation)
    desired_p, _ = hannan_rissanen(
        endog, ar_order=1, ma_order=1, demean=False)
    mod = ARIMA(endog, order=(1, 0, 1), trend='n')
    res = mod.fit(method='hannan_rissanen')
    assert_allclose(res.params, desired_p.params)


def test_innovations():
    # Test for basic use of Yule-Walker estimation
    endog = dta['infl'].iloc[:50]

    # MA(2), no trend (since trend would imply GLS estimation)
    desired_p, _ = innovations(endog, ma_order=2, demean=False)
    mod = ARIMA(endog, order=(0, 0, 2), trend='n')
    res = mod.fit(method='innovations')
    assert_allclose(res.params, desired_p[-1].params)


def test_innovations_mle():
    # Test for basic use of Yule-Walker estimation
    endog = dta['infl'].iloc[:100]

    # ARMA(1, 1), no trend (since trend would imply GLS estimation)
    desired_p, _ = innovations_mle(
        endog, order=(1, 0, 1), demean=False)
    mod = ARIMA(endog, order=(1, 0, 1), trend='n')
    res = mod.fit(method='innovations_mle')
    # Note: atol is required only due to precision issues on Windows
    assert_allclose(res.params, desired_p.params, atol=1e-5)

    # SARMA(1, 0)x(1, 0)4, no trend (since trend would imply GLS estimation)
    desired_p, _ = innovations_mle(
        endog, order=(1, 0, 0), seasonal_order=(1, 0, 0, 4), demean=False)
    mod = ARIMA(endog, order=(1, 0, 0), seasonal_order=(1, 0, 0, 4), trend='n')
    res = mod.fit(method='innovations_mle')
    # Note: atol is required only due to precision issues on Windows
    assert_allclose(res.params, desired_p.params, atol=1e-5)


def test_statespace():
    # Test for basic use of Yule-Walker estimation
    endog = dta['infl'].iloc[:100]

    # ARMA(1, 1), no trend
    desired_p, _ = statespace(endog, order=(1, 0, 1),
                              include_constant=False)
    mod = ARIMA(endog, order=(1, 0, 1), trend='n')
    res = mod.fit(method='statespace')
    # Note: tol changes required due to precision issues on Windows
    rtol = 1e-7 if not PLATFORM_WIN32 else 1e-3
    assert_allclose(res.params, desired_p.params, rtol=rtol, atol=1e-4)

    # ARMA(1, 2), with trend
    desired_p, _ = statespace(endog, order=(1, 0, 2),
                              include_constant=True)
    mod = ARIMA(endog, order=(1, 0, 2), trend='c')
    res = mod.fit(method='statespace')
    # Note: atol is required only due to precision issues on Windows
    assert_allclose(res.params, desired_p.params, atol=1e-4)

    # SARMA(1, 0)x(1, 0)4, no trend
    desired_p, _spec = statespace(endog, order=(1, 0, 0),
                                  seasonal_order=(1, 0, 0, 4),
                                  include_constant=False)
    mod = ARIMA(endog, order=(1, 0, 0), seasonal_order=(1, 0, 0, 4), trend='n')
    res = mod.fit(method='statespace')
    # Note: atol is required only due to precision issues on Windows
    assert_allclose(res.params, desired_p.params, atol=1e-4)


def test_low_memory():
    # Basic test that the low_memory option is working
    endog = dta['infl'].iloc[:50]

    mod = ARIMA(endog, order=(1, 0, 0), concentrate_scale=True)
    res1 = mod.fit()
    res2 = mod.fit(low_memory=True)

    # Check that the models produce the same results
    assert_allclose(res2.params, res1.params)
    assert_allclose(res2.llf, res1.llf)

    # Check that the model's basic memory conservation option was not changed
    assert_equal(mod.ssm.memory_conserve, 0)

    # Check that low memory was actually used (just check a couple)
    assert_(res2.llf_obs is None)
    assert_(res2.predicted_state is None)
    assert_(res2.filtered_state is None)
    assert_(res2.smoothed_state is None)


def check_cloned(mod, endog, exog=None):
    mod_c = mod.clone(endog, exog=exog)

    assert_allclose(mod.nobs, mod_c.nobs)
    assert_(mod._index.equals(mod_c._index))
    assert_equal(mod.k_params, mod_c.k_params)
    assert_allclose(mod.start_params, mod_c.start_params)
    p = mod.start_params
    assert_allclose(mod.loglike(p), mod_c.loglike(p))
    assert_allclose(mod.concentrate_scale, mod_c.concentrate_scale)


def test_clone():
    endog = dta['infl'].iloc[:50]
    exog = np.arange(endog.shape[0])

    # Basic model
    check_cloned(ARIMA(endog), endog)
    check_cloned(ARIMA(endog.values), endog.values)
    # With trends
    check_cloned(ARIMA(endog, trend='c'), endog)
    check_cloned(ARIMA(endog, trend='t'), endog)
    check_cloned(ARIMA(endog, trend='ct'), endog)
    # With exog
    check_cloned(ARIMA(endog, exog=exog), endog, exog=exog)
    check_cloned(ARIMA(endog, exog=exog, trend='c'), endog, exog=exog)
    # Concentrated scale
    check_cloned(ARIMA(endog, exog=exog, trend='c', concentrate_scale=True),
                 endog, exog=exog)

    # Higher order (use a different dataset to avoid warnings about
    # non-invertible start params)
    endog = dta['realgdp'].iloc[:100]
    exog = np.arange(endog.shape[0])
    check_cloned(ARIMA(endog, order=(2, 1, 1), seasonal_order=(1, 1, 2, 4),
                       exog=exog, trend=[0, 0, 1], concentrate_scale=True),
                 endog, exog=exog)


def test_constant_integrated_model_error():
    with pytest.raises(ValueError, match="In models with integration"):
        ARIMA(np.ones(100), order=(1, 1, 0), trend='c')

    with pytest.raises(ValueError, match="In models with integration"):
        ARIMA(np.ones(100), order=(1, 0, 0), seasonal_order=(1, 1, 0, 6),
              trend='c')

    with pytest.raises(ValueError, match="In models with integration"):
        ARIMA(np.ones(100), order=(1, 2, 0), trend='t')

    with pytest.raises(ValueError, match="In models with integration"):
        ARIMA(np.ones(100), order=(1, 1, 0), seasonal_order=(1, 1, 0, 6),
              trend='t')


def test_forecast():
    # Numpy
    endog = dta['infl'].iloc[:100].values

    mod = ARIMA(endog[:50], order=(1, 1, 0), trend='t')
    res = mod.filter([0.2, 0.3, 1.0])

    endog2 = endog.copy()
    endog2[50:] = np.nan
    mod2 = mod.clone(endog2)
    res2 = mod2.filter(res.params)

    assert_allclose(res.forecast(50), res2.fittedvalues[-50:])


def test_forecast_with_exog():
    # Numpy
    endog = dta['infl'].iloc[:100].values
    exog = np.arange(len(endog))**2

    mod = ARIMA(endog[:50], order=(1, 1, 0), exog=exog[:50], trend='t')
    res = mod.filter([0.2, 0.05, 0.3, 1.0])

    endog2 = endog.copy()
    endog2[50:] = np.nan
    mod2 = mod.clone(endog2, exog=exog)
    print(mod.param_names)
    print(mod2.param_names)
    res2 = mod2.filter(res.params)

    assert_allclose(res.forecast(50, exog=exog[50:]), res2.fittedvalues[-50:])


def test_append():
    endog = dta['infl'].iloc[:100].values
    mod = ARIMA(endog[:50], trend='c')
    res = mod.fit()
    res_e = res.append(endog[50:])
    mod2 = ARIMA(endog)
    res2 = mod2.filter(res_e.params)

    assert_allclose(res2.llf, res_e.llf)


def test_append_with_exog():
    # Numpy
    endog = dta['infl'].iloc[:100].values
    exog = np.arange(len(endog))
    mod = ARIMA(endog[:50], exog=exog[:50], trend='c')
    res = mod.fit()
    res_e = res.append(endog[50:], exog=exog[50:])
    mod2 = ARIMA(endog, exog=exog, trend='c')
    res2 = mod2.filter(res_e.params)

    assert_allclose(res2.llf, res_e.llf)


def test_append_with_exog_and_trend():
    # Numpy
    endog = dta['infl'].iloc[:100].values
    exog = np.arange(len(endog))**2
    mod = ARIMA(endog[:50], exog=exog[:50], trend='ct')
    res = mod.fit()
    res_e = res.append(endog[50:], exog=exog[50:])
    mod2 = ARIMA(endog, exog=exog, trend='ct')
    res2 = mod2.filter(res_e.params)

    assert_allclose(res2.llf, res_e.llf)


def test_append_with_exog_pandas():
    # Pandas
    endog = dta['infl'].iloc[:100]
    exog = pd.Series(np.arange(len(endog)), index=endog.index)
    mod = ARIMA(endog.iloc[:50], exog=exog.iloc[:50], trend='c')
    res = mod.fit()
    res_e = res.append(endog.iloc[50:], exog=exog.iloc[50:])
    mod2 = ARIMA(endog, exog=exog, trend='c')
    res2 = mod2.filter(res_e.params)

    assert_allclose(res2.llf, res_e.llf)


def test_cov_type_none():
    endog = dta['infl'].iloc[:100].values
    mod = ARIMA(endog[:50], trend='c')
    res = mod.fit(cov_type='none')
    assert_allclose(res.cov_params(), np.nan)


def test_nonstationary_gls_error():
    # GH-6540
    endog = pd.read_csv(
        io.StringIO(
            """\
data\n
9.112\n9.102\n9.103\n9.099\n9.094\n9.090\n9.108\n9.088\n9.091\n9.083\n9.095\n
9.090\n9.098\n9.093\n9.087\n9.088\n9.083\n9.095\n9.077\n9.082\n9.082\n9.081\n
9.081\n9.079\n9.088\n9.096\n9.081\n9.098\n9.081\n9.094\n9.091\n9.095\n9.097\n
9.108\n9.104\n9.098\n9.085\n9.093\n9.094\n9.092\n9.093\n9.106\n9.097\n9.108\n
9.100\n9.106\n9.114\n9.111\n9.097\n9.099\n9.108\n9.108\n9.110\n9.101\n9.111\n
9.114\n9.111\n9.126\n9.124\n9.112\n9.120\n9.142\n9.136\n9.131\n9.106\n9.112\n
9.119\n9.125\n9.123\n9.138\n9.133\n9.133\n9.137\n9.133\n9.138\n9.136\n9.128\n
9.127\n9.143\n9.128\n9.135\n9.133\n9.131\n9.136\n9.120\n9.127\n9.130\n9.116\n
9.132\n9.128\n9.119\n9.119\n9.110\n9.132\n9.130\n9.124\n9.130\n9.135\n9.135\n
9.119\n9.119\n9.136\n9.126\n9.122\n9.119\n9.123\n9.121\n9.130\n9.121\n9.119\n
9.106\n9.118\n9.124\n9.121\n9.127\n9.113\n9.118\n9.103\n9.112\n9.110\n9.111\n
9.108\n9.113\n9.117\n9.111\n9.100\n9.106\n9.109\n9.113\n9.110\n9.101\n9.113\n
9.111\n9.101\n9.097\n9.102\n9.100\n9.110\n9.110\n9.096\n9.095\n9.090\n9.104\n
9.097\n9.099\n9.095\n9.096\n9.085\n9.097\n9.098\n9.090\n9.080\n9.093\n9.085\n
9.075\n9.067\n9.072\n9.062\n9.068\n9.053\n9.051\n9.049\n9.052\n9.059\n9.070\n
9.058\n9.074\n9.063\n9.057\n9.062\n9.058\n9.049\n9.047\n9.062\n9.052\n9.052\n
9.044\n9.060\n9.062\n9.055\n9.058\n9.054\n9.044\n9.047\n9.050\n9.048\n9.041\n
9.055\n9.051\n9.028\n9.030\n9.029\n9.027\n9.016\n9.023\n9.031\n9.042\n9.035\n
"""
        ),
        index_col=None,
    )
    mod = ARIMA(
        endog,
        order=(18, 0, 39),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    with pytest.raises(ValueError, match="Roots of the autoregressive"):
        mod.fit(method="hannan_rissanen", low_memory=True, cov_type="none")


@pytest.mark.parametrize(
    "ar_order, ma_order, fixed_params",
    [
        (1, 1, {}),
        (1, 1, {'ar.L1': 0}),
        (2, 3, {'ar.L2': -1, 'ma.L1': 2}),
        ([0, 1], 0, {'ar.L2': 0}),
        ([1, 5], [0, 0, 1], {'ar.L5': -10, 'ma.L3': 5}),
    ]
)
def test_hannan_rissanen_with_fixed_params(ar_order, ma_order, fixed_params):
    # Test for basic uses of Hannan-Rissanen estimation with fixed parameters
    endog = dta['infl'].diff().iloc[1:101]

    desired_p, _ = hannan_rissanen(
        endog, ar_order=ar_order, ma_order=ma_order,
        demean=False, fixed_params=fixed_params
    )
    # no constant or trend (since constant or trend would imply GLS estimation)
    mod = ARIMA(endog, order=(ar_order, 0, ma_order), trend='n',
                enforce_stationarity=False, enforce_invertibility=False)
    with mod.fix_params(fixed_params):
        res = mod.fit(method='hannan_rissanen')

    assert_allclose(res.params, desired_p.params)
