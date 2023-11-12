"""
Tests for prediction of state space models

Author: Chad Fulton
License: Simplified-BSD
"""

import pytest

import numpy as np
import pandas as pd

from numpy.testing import assert_equal, assert_raises, assert_allclose, assert_

from statsmodels import datasets
from statsmodels.tsa.statespace import sarimax, varmax
from statsmodels.tsa.statespace.tests.test_impulse_responses import TVSS


dta = datasets.macrodata.load_pandas().data
dta.index = pd.period_range(start='1959Q1', end='2009Q3', freq='Q')


def test_predict_dates():
    index = pd.date_range(start='1950-01-01', periods=11, freq='D')
    np.random.seed(324328)
    endog = pd.Series(np.random.normal(size=10), index=index[:-1])

    # Basic test
    mod = sarimax.SARIMAX(endog, order=(1, 0, 0))
    res = mod.filter(mod.start_params)

    # In-sample prediction should have the same index
    pred = res.predict()
    assert_equal(len(pred), mod.nobs)
    assert_equal(pred.index.values, index[:-1].values)
    # Out-of-sample forecasting should extend the index appropriately
    fcast = res.forecast()
    assert_equal(fcast.index[0], index[-1])

    # Simple differencing in the SARIMAX model should eliminate dates of
    # series eliminated due to differencing
    mod = sarimax.SARIMAX(endog, order=(1, 1, 0), simple_differencing=True)
    res = mod.filter(mod.start_params)
    pred = res.predict()
    # In-sample prediction should lose the first index value
    assert_equal(mod.nobs, endog.shape[0] - 1)
    assert_equal(len(pred), mod.nobs)
    assert_equal(pred.index.values, index[1:-1].values)
    # Out-of-sample forecasting should still extend the index appropriately
    fcast = res.forecast()
    assert_equal(fcast.index[0], index[-1])

    # Simple differencing again, this time with a more complex differencing
    # structure
    mod = sarimax.SARIMAX(endog, order=(1, 2, 0), seasonal_order=(0, 1, 0, 4),
                          simple_differencing=True)
    res = mod.filter(mod.start_params)
    pred = res.predict()
    # In-sample prediction should lose the first 6 index values
    assert_equal(mod.nobs, endog.shape[0] - (4 + 2))
    assert_equal(len(pred), mod.nobs)
    assert_equal(pred.index.values, index[4 + 2:-1].values)
    # Out-of-sample forecasting should still extend the index appropriately
    fcast = res.forecast()
    assert_equal(fcast.index[0], index[-1])


def test_memory_no_predicted():
    # Tests for forecasts with memory_no_predicted is set
    endog = [0.5, 1.2, 0.4, 0.6]

    mod = sarimax.SARIMAX(endog, order=(1, 0, 0))
    res1 = mod.filter([0.5, 1.])
    mod.ssm.memory_no_predicted = True
    res2 = mod.filter([0.5, 1.])

    # Make sure we really didn't store all of the values in res2
    assert_equal(res1.predicted_state.shape, (1, 5))
    assert_(res2.predicted_state is None)
    assert_equal(res1.predicted_state_cov.shape, (1, 1, 5))
    assert_(res2.predicted_state_cov is None)

    # Check that we can't do dynamic in-sample prediction
    assert_raises(ValueError, res2.predict, dynamic=True)
    assert_raises(ValueError, res2.get_prediction, dynamic=True)

    # Make sure the point forecasts are the same
    assert_allclose(res1.forecast(10), res2.forecast(10))

    # Make sure the confidence intervals are the same
    fcast1 = res1.get_forecast(10)
    fcast2 = res1.get_forecast(10)

    assert_allclose(fcast1.summary_frame(), fcast2.summary_frame())


@pytest.mark.parametrize('use_exog', [True, False])
@pytest.mark.parametrize('trend', ['n', 'c', 't'])
def test_concatenated_predict_sarimax(use_exog, trend):
    endog = np.arange(100).reshape(100, 1) * 1.0
    exog = np.ones(100) if use_exog else None
    if use_exog:
        exog[10:30] = 2.

    trend_params = [0.1]
    ar_params = [0.5]
    exog_params = [1.2]
    var_params = [1.]

    params = []
    if trend in ['c', 't']:
        params += trend_params
    params += ar_params
    if use_exog:
        params += exog_params
    params += var_params

    y1 = endog.copy()
    y1[-50:] = np.nan
    mod1 = sarimax.SARIMAX(y1, order=(1, 1, 0), trend=trend, exog=exog)
    res1 = mod1.smooth(params)
    p1 = res1.get_prediction()
    pr1 = p1.prediction_results

    x2 = exog[:50] if use_exog else None
    mod2 = sarimax.SARIMAX(endog[:50], order=(1, 1, 0), trend=trend, exog=x2)
    res2 = mod2.smooth(params)
    x2f = exog[50:] if use_exog else None
    p2 = res2.get_prediction(start=0, end=99, exog=x2f)
    pr2 = p2.prediction_results

    attrs = (
        pr1.representation_attributes
        + pr1.filter_attributes
        + pr1.smoother_attributes)
    for key in attrs:
        assert_allclose(getattr(pr2, key), getattr(pr1, key))


@pytest.mark.parametrize('use_exog', [True, False])
@pytest.mark.parametrize('trend', ['n', 'c', 't'])
def test_concatenated_predict_varmax(use_exog, trend):
    endog = np.arange(200).reshape(100, 2) * 1.0
    exog = np.ones(100) if use_exog else None

    trend_params = [0.1, 0.2]
    var_params = [0.5, -0.1, 0.0, 0.2]
    exog_params = [1., 2.]
    cov_params = [1., 0., 1.]

    params = []
    if trend in ['c', 't']:
        params += trend_params
    params += var_params
    if use_exog:
        params += exog_params
    params += cov_params

    y1 = endog.copy()
    y1[-50:] = np.nan
    mod1 = varmax.VARMAX(y1, order=(1, 0), trend=trend, exog=exog)
    res1 = mod1.smooth(params)
    p1 = res1.get_prediction()
    pr1 = p1.prediction_results

    x2 = exog[:50] if use_exog else None
    mod2 = varmax.VARMAX(endog[:50], order=(1, 0), trend=trend, exog=x2)
    res2 = mod2.smooth(params)
    x2f = exog[50:] if use_exog else None
    p2 = res2.get_prediction(start=0, end=99, exog=x2f)
    pr2 = p2.prediction_results

    attrs = (
        pr1.representation_attributes
        + pr1.filter_attributes
        + pr1.smoother_attributes)
    for key in attrs:
        assert_allclose(getattr(pr2, key), getattr(pr1, key))


@pytest.mark.parametrize('use_exog', [True, False])
@pytest.mark.parametrize('trend', ['n', 'c', 't'])
def test_predicted_filtered_smoothed_with_nans(use_exog, trend):
    # In this test, we construct a model with only NaN values for `endog`, so
    # that predicted, filtered, and smoothed forecasts should all be the
    # same
    endog = np.zeros(200).reshape(100, 2) * np.nan
    exog = np.ones(100) if use_exog else None

    trend_params = [0.1, 0.2]
    var_params = [0.5, -0.1, 0.0, 0.2]
    exog_params = [1., 2.]
    cov_params = [1., 0., 1.]

    params = []
    if trend in ['c', 't']:
        params += trend_params
    params += var_params
    if use_exog:
        params += exog_params
    params += cov_params

    x_fit = exog[:50] if use_exog else None
    mod = varmax.VARMAX(endog[:50], order=(1, 0), trend=trend, exog=x_fit)
    res = mod.smooth(params)

    x_fcast = exog[50:61] if use_exog else None
    p_pred = res.get_prediction(
        start=0, end=60, information_set='predicted',
        exog=x_fcast)

    f_pred = res.get_prediction(
        start=0, end=60, information_set='filtered',
        exog=x_fcast)

    s_pred = res.get_prediction(
        start=0, end=60, information_set='smoothed',
        exog=x_fcast)

    # Test forecasts
    assert_allclose(s_pred.predicted_mean, p_pred.predicted_mean)
    assert_allclose(s_pred.var_pred_mean, p_pred.var_pred_mean)
    assert_allclose(f_pred.predicted_mean, p_pred.predicted_mean)
    assert_allclose(f_pred.var_pred_mean, p_pred.var_pred_mean)
    assert_allclose(p_pred.predicted_mean[:50], res.fittedvalues)
    assert_allclose(p_pred.var_pred_mean[:50].T, res.forecasts_error_cov)

    p_signal = res.get_prediction(
        start=0, end=60, information_set='predicted', signal_only=True,
        exog=x_fcast)

    f_signal = res.get_prediction(
        start=0, end=60, information_set='filtered', signal_only=True,
        exog=x_fcast)

    s_signal = res.get_prediction(
        start=0, end=60, information_set='smoothed', signal_only=True,
        exog=x_fcast)

    # Test signal predictions
    assert_allclose(s_signal.predicted_mean, p_signal.predicted_mean)
    assert_allclose(s_signal.var_pred_mean, p_signal.var_pred_mean)
    assert_allclose(f_signal.predicted_mean, p_signal.predicted_mean)
    assert_allclose(f_signal.var_pred_mean, p_signal.var_pred_mean)

    if use_exog is False and trend == 'n':
        assert_allclose(p_signal.predicted_mean[:50], res.fittedvalues)
        assert_allclose(p_signal.var_pred_mean[:50].T, res.forecasts_error_cov)
    else:
        assert_allclose(p_signal.predicted_mean[:50] + mod['obs_intercept'],
                        res.fittedvalues)
        assert_allclose((p_signal.var_pred_mean[:50] + mod['obs_cov']).T,
                        res.forecasts_error_cov)


def test_predicted_filtered_smoothed_with_nans_TVSS(reset_randomstate):
    mod = TVSS(np.zeros((50, 2)) * np.nan)
    mod.ssm.initialize_known([1.2, 0.8], np.eye(2))
    res = mod.smooth([])

    mod_oos = TVSS(np.zeros((11, 2)) * np.nan)
    kwargs = {key: mod_oos[key] for key in [
        'obs_intercept', 'design', 'obs_cov',
        'transition', 'selection', 'state_cov']}

    p_pred = res.get_prediction(
        start=0, end=60, information_set='predicted',
        **kwargs)

    f_pred = res.get_prediction(
        start=0, end=60, information_set='filtered',
        **kwargs)

    s_pred = res.get_prediction(
        start=0, end=60, information_set='smoothed',
        **kwargs)

    # Test forecasts
    assert_allclose(s_pred.predicted_mean, p_pred.predicted_mean)
    assert_allclose(s_pred.var_pred_mean, p_pred.var_pred_mean)
    assert_allclose(f_pred.predicted_mean, p_pred.predicted_mean)
    assert_allclose(f_pred.var_pred_mean, p_pred.var_pred_mean)
    assert_allclose(p_pred.predicted_mean[:50], res.fittedvalues)
    assert_allclose(p_pred.var_pred_mean[:50].T, res.forecasts_error_cov)

    p_signal = res.get_prediction(
        start=0, end=60, information_set='predicted', signal_only=True,
        **kwargs)

    f_signal = res.get_prediction(
        start=0, end=60, information_set='filtered', signal_only=True,
        **kwargs)

    s_signal = res.get_prediction(
        start=0, end=60, information_set='smoothed', signal_only=True,
        **kwargs)

    # Test signal predictions
    assert_allclose(s_signal.predicted_mean, p_signal.predicted_mean)
    assert_allclose(s_signal.var_pred_mean, p_signal.var_pred_mean)
    assert_allclose(f_signal.predicted_mean, p_signal.predicted_mean)
    assert_allclose(f_signal.var_pred_mean, p_signal.var_pred_mean)
    assert_allclose(p_signal.predicted_mean[:50] + mod['obs_intercept'].T,
                    res.fittedvalues)
    assert_allclose((p_signal.var_pred_mean[:50] + mod['obs_cov'].T).T,
                    res.forecasts_error_cov)


@pytest.mark.parametrize('use_exog', [True, False])
@pytest.mark.parametrize('trend', ['n', 'c', 't'])
def test_predicted_filtered_smoothed_varmax(use_exog, trend):
    endog = np.log(dta[['realgdp', 'cpi']])
    if trend in ['n', 'c']:
        endog = endog.diff().iloc[1:] * 100
    if trend == 'n':
        endog -= endog.mean()
    exog = np.ones(100) if use_exog else None
    if use_exog:
        exog[20:40] = 2.

    trend_params = [0.1, 0.2]
    var_params = [0.5, -0.1, 0.0, 0.2]
    exog_params = [1., 2.]
    cov_params = [1., 0., 1.]

    params = []
    if trend in ['c', 't']:
        params += trend_params
    params += var_params
    if use_exog:
        params += exog_params
    params += cov_params

    x_fit = exog[:50] if use_exog else None
    mod = varmax.VARMAX(endog[:50], order=(1, 0), trend=trend, exog=x_fit)

    # Add in an obs_intercept and obs_cov to make the test more comprehensive
    mod['obs_intercept'] = [5, -2.]
    mod['obs_cov'] = np.array([[1.2, 0.3],
                               [0.3, 3.4]])

    res = mod.smooth(params)

    x_fcast = exog[50:61] if use_exog else None
    p_pred = res.get_prediction(
        start=0, end=60, information_set='predicted',
        exog=x_fcast)

    f_pred = res.get_prediction(
        start=0, end=60, information_set='filtered',
        exog=x_fcast)

    s_pred = res.get_prediction(
        start=0, end=60, information_set='smoothed',
        exog=x_fcast)

    # Test forecasts
    fcast = res.get_forecast(11, exog=x_fcast)

    d = mod['obs_intercept'][:, None]
    Z = mod['design']
    H = mod['obs_cov'][:, :, None]
    desired_s_signal = Z @ res.smoothed_state
    desired_f_signal = Z @ res.filtered_state
    desired_p_signal = Z @ res.predicted_state[..., :-1]
    assert_allclose(s_pred.predicted_mean[:50], (d + desired_s_signal).T)
    assert_allclose(s_pred.predicted_mean[50:], fcast.predicted_mean)
    assert_allclose(f_pred.predicted_mean[:50], (d + desired_f_signal).T)
    assert_allclose(f_pred.predicted_mean[50:], fcast.predicted_mean)
    assert_allclose(p_pred.predicted_mean[:50], (d + desired_p_signal).T)
    assert_allclose(p_pred.predicted_mean[50:], fcast.predicted_mean)

    desired_s_signal_cov = (
        Z[None, :, :] @ res.smoothed_state_cov.T @ Z.T[None, :, :])
    desired_f_signal_cov = (
        Z[None, :, :] @ res.filtered_state_cov.T @ Z.T[None, :, :])
    desired_p_signal_cov = (
        Z[None, :, :] @ res.predicted_state_cov[..., :-1].T @ Z.T[None, :, :])
    assert_allclose(s_pred.var_pred_mean[:50], (desired_s_signal_cov.T + H).T)
    assert_allclose(s_pred.var_pred_mean[50:], fcast.var_pred_mean)
    assert_allclose(f_pred.var_pred_mean[:50], (desired_f_signal_cov.T + H).T)
    assert_allclose(f_pred.var_pred_mean[50:], fcast.var_pred_mean)
    assert_allclose(p_pred.var_pred_mean[:50], (desired_p_signal_cov.T + H).T)
    assert_allclose(p_pred.var_pred_mean[50:], fcast.var_pred_mean)

    p_signal = res.get_prediction(
        start=0, end=60, information_set='predicted', signal_only=True,
        exog=x_fcast)

    f_signal = res.get_prediction(
        start=0, end=60, information_set='filtered', signal_only=True,
        exog=x_fcast)

    s_signal = res.get_prediction(
        start=0, end=60, information_set='smoothed', signal_only=True,
        exog=x_fcast)

    # Test signal predictions
    fcast_signal = fcast.predicted_mean - d.T
    fcast_signal_cov = (fcast.var_pred_mean.T - H).T
    assert_allclose(s_signal.predicted_mean[:50], desired_s_signal.T)
    assert_allclose(s_signal.predicted_mean[50:], fcast_signal)
    assert_allclose(f_signal.predicted_mean[:50], desired_f_signal.T)
    assert_allclose(f_signal.predicted_mean[50:], fcast_signal)
    assert_allclose(p_signal.predicted_mean[:50], desired_p_signal.T)
    assert_allclose(p_signal.predicted_mean[50:], fcast_signal)

    assert_allclose(s_signal.var_pred_mean[:50], desired_s_signal_cov)
    assert_allclose(s_signal.var_pred_mean[50:], fcast_signal_cov)
    assert_allclose(f_signal.var_pred_mean[:50], desired_f_signal_cov)
    assert_allclose(f_signal.var_pred_mean[50:], fcast_signal_cov)
    assert_allclose(p_signal.var_pred_mean[:50], desired_p_signal_cov)
    assert_allclose(p_signal.var_pred_mean[50:], fcast_signal_cov)


def test_predicted_filtered_smoothed_TVSS(reset_randomstate):
    mod = TVSS(np.zeros((50, 2)))
    mod.ssm.initialize_known([1.2, 0.8], np.eye(2))
    res = mod.smooth([])

    mod_oos = TVSS(np.zeros((11, 2)) * np.nan)
    kwargs = {key: mod_oos[key] for key in [
        'obs_intercept', 'design', 'obs_cov',
        'transition', 'selection', 'state_cov']}

    p_pred = res.get_prediction(
        start=0, end=60, information_set='predicted',
        **kwargs)

    f_pred = res.get_prediction(
        start=0, end=60, information_set='filtered',
        **kwargs)

    s_pred = res.get_prediction(
        start=0, end=60, information_set='smoothed',
        **kwargs)

    p_signal = res.get_prediction(
        start=0, end=60, information_set='predicted', signal_only=True,
        **kwargs)

    f_signal = res.get_prediction(
        start=0, end=60, information_set='filtered', signal_only=True,
        **kwargs)

    s_signal = res.get_prediction(
        start=0, end=60, information_set='smoothed', signal_only=True,
        **kwargs)

    # Test forecasts and signals
    d = mod['obs_intercept'].transpose(1, 0)[:, :, None]
    Z = mod['design'].transpose(2, 0, 1)
    H = mod['obs_cov'].transpose(2, 0, 1)

    fcast = res.get_forecast(11, **kwargs)
    fcast_signal = fcast.predicted_mean - mod_oos['obs_intercept'].T
    fcast_signal_cov = fcast.var_pred_mean - mod_oos['obs_cov'].T

    desired_s_signal = Z @ res.smoothed_state.T[:, :, None]
    desired_f_signal = Z @ res.filtered_state.T[:, :, None]
    desired_p_signal = Z @ res.predicted_state.T[:-1, :, None]
    assert_allclose(s_pred.predicted_mean[:50], (d + desired_s_signal)[..., 0])
    assert_allclose(s_pred.predicted_mean[50:], fcast.predicted_mean)
    assert_allclose(f_pred.predicted_mean[:50], (d + desired_f_signal)[..., 0])
    assert_allclose(f_pred.predicted_mean[50:], fcast.predicted_mean)
    assert_allclose(p_pred.predicted_mean[:50], (d + desired_p_signal)[..., 0])
    assert_allclose(p_pred.predicted_mean[50:], fcast.predicted_mean)

    assert_allclose(s_signal.predicted_mean[:50], desired_s_signal[..., 0])
    assert_allclose(s_signal.predicted_mean[50:], fcast_signal)
    assert_allclose(f_signal.predicted_mean[:50], desired_f_signal[..., 0])
    assert_allclose(f_signal.predicted_mean[50:], fcast_signal)
    assert_allclose(p_signal.predicted_mean[:50], desired_p_signal[..., 0])
    assert_allclose(p_signal.predicted_mean[50:], fcast_signal)

    for t in range(mod.nobs):
        assert_allclose(s_pred.var_pred_mean[t],
                        Z[t] @ res.smoothed_state_cov[..., t] @ Z[t].T + H[t])
        assert_allclose(f_pred.var_pred_mean[t],
                        Z[t] @ res.filtered_state_cov[..., t] @ Z[t].T + H[t])
        assert_allclose(p_pred.var_pred_mean[t],
                        Z[t] @ res.predicted_state_cov[..., t] @ Z[t].T + H[t])

        assert_allclose(s_signal.var_pred_mean[t],
                        Z[t] @ res.smoothed_state_cov[..., t] @ Z[t].T)
        assert_allclose(f_signal.var_pred_mean[t],
                        Z[t] @ res.filtered_state_cov[..., t] @ Z[t].T)
        assert_allclose(p_signal.var_pred_mean[t],
                        Z[t] @ res.predicted_state_cov[..., t] @ Z[t].T)

    assert_allclose(s_pred.var_pred_mean[50:], fcast.var_pred_mean)
    assert_allclose(f_pred.var_pred_mean[50:], fcast.var_pred_mean)
    assert_allclose(p_pred.var_pred_mean[50:], fcast.var_pred_mean)
    assert_allclose(s_signal.var_pred_mean[50:], fcast_signal_cov)
    assert_allclose(f_signal.var_pred_mean[50:], fcast_signal_cov)
    assert_allclose(p_signal.var_pred_mean[50:], fcast_signal_cov)


@pytest.mark.parametrize('use_exog', [False, True])
@pytest.mark.parametrize('trend', ['n', 'c', 't'])
def test_predicted_filtered_dynamic_varmax(use_exog, trend):
    endog = np.log(dta[['realgdp', 'cpi']])
    if trend in ['n', 'c']:
        endog = endog.diff().iloc[1:] * 100
    if trend == 'n':
        endog -= endog.mean()
    exog = np.ones(100) if use_exog else None
    if use_exog:
        exog[20:40] = 2.

    trend_params = [0.1, 0.2]
    var_params = [0.5, -0.1, 0.0, 0.2]
    exog_params = [1., 2.]
    cov_params = [1., 0., 1.]

    params = []
    if trend in ['c', 't']:
        params += trend_params
    params += var_params
    if use_exog:
        params += exog_params
    params += cov_params

    # Compute basic model with 50 observations
    x_fit1 = exog[:50] if use_exog else None
    x_fcast1 = exog[50:61] if use_exog else None
    mod1 = varmax.VARMAX(endog[:50], order=(1, 0), trend=trend, exog=x_fit1)

    res1 = mod1.filter(params)

    # Compute basic model with only 20 observations
    x_fit2 = exog[:20] if use_exog else None
    x_fcast2 = exog[20:61] if use_exog else None
    mod2 = varmax.VARMAX(endog[:20], order=(1, 0), trend=trend, exog=x_fit2)
    res2 = mod2.filter(params)

    # Test predictions
    p1 = res1.get_prediction(start=0, dynamic=20, end=60, exog=x_fcast1)
    p2 = res2.get_prediction(start=0, end=60, exog=x_fcast2)
    assert_allclose(p1.predicted_mean, p2.predicted_mean)
    assert_allclose(p1.var_pred_mean, p2.var_pred_mean)

    p1 = res1.get_prediction(start=2, dynamic=18, end=60, exog=x_fcast1)
    p2 = res2.get_prediction(start=2, end=60, exog=x_fcast2)
    assert_allclose(p1.predicted_mean, p2.predicted_mean)
    assert_allclose(p1.var_pred_mean, p2.var_pred_mean)

    p1 = res1.get_prediction(start=20, dynamic=True, end=60, exog=x_fcast1)
    p2 = res2.get_prediction(start=20, end=60, exog=x_fcast2)
    assert_allclose(p1.predicted_mean, p2.predicted_mean)
    assert_allclose(p1.var_pred_mean, p2.var_pred_mean)

    # Test predictions, filtered
    p1 = res1.get_prediction(start=0, dynamic=20, end=60, exog=x_fcast1,
                             information_set='filtered')
    p2 = res2.get_prediction(start=0, end=60, exog=x_fcast2,
                             information_set='filtered')
    assert_allclose(p1.predicted_mean, p2.predicted_mean)
    assert_allclose(p1.var_pred_mean, p2.var_pred_mean)

    p1 = res1.get_prediction(start=2, dynamic=18, end=60, exog=x_fcast1,
                             information_set='filtered')
    p2 = res2.get_prediction(start=2, end=60, exog=x_fcast2,
                             information_set='filtered')
    assert_allclose(p1.predicted_mean, p2.predicted_mean)
    assert_allclose(p1.var_pred_mean, p2.var_pred_mean)

    p1 = res1.get_prediction(start=20, dynamic=True, end=60, exog=x_fcast1,
                             information_set='filtered')
    p2 = res2.get_prediction(start=20, end=60, exog=x_fcast2,
                             information_set='filtered')
    assert_allclose(p1.predicted_mean, p2.predicted_mean)
    assert_allclose(p1.var_pred_mean, p2.var_pred_mean)

    # Test signals
    p1 = res1.get_prediction(start=0, dynamic=20, end=60, exog=x_fcast1,
                             signal_only=True)
    p2 = res2.get_prediction(start=0, end=60, exog=x_fcast2, signal_only=True)
    assert_allclose(p1.predicted_mean, p2.predicted_mean)
    assert_allclose(p1.var_pred_mean, p2.var_pred_mean)

    p1 = res1.get_prediction(start=2, dynamic=18, end=60, exog=x_fcast1,
                             signal_only=True)
    p2 = res2.get_prediction(start=2, end=60, exog=x_fcast2, signal_only=True)
    assert_allclose(p1.predicted_mean, p2.predicted_mean)
    assert_allclose(p1.var_pred_mean, p2.var_pred_mean)

    p1 = res1.get_prediction(start=20, dynamic=True, end=60, exog=x_fcast1,
                             signal_only=True)
    p2 = res2.get_prediction(start=20, end=60, exog=x_fcast2, signal_only=True)
    assert_allclose(p1.predicted_mean, p2.predicted_mean)
    assert_allclose(p1.var_pred_mean, p2.var_pred_mean)

    # Test signal, filtered
    p1 = res1.get_prediction(start=0, dynamic=20, end=60, exog=x_fcast1,
                             signal_only=True, information_set='filtered')
    p2 = res2.get_prediction(start=0, end=60, exog=x_fcast2, signal_only=True,
                             information_set='filtered')
    assert_allclose(p1.predicted_mean, p2.predicted_mean)
    assert_allclose(p1.var_pred_mean, p2.var_pred_mean)

    p1 = res1.get_prediction(start=2, dynamic=18, end=60, exog=x_fcast1,
                             signal_only=True, information_set='filtered')
    p2 = res2.get_prediction(start=2, end=60, exog=x_fcast2, signal_only=True,
                             information_set='filtered')
    assert_allclose(p1.predicted_mean, p2.predicted_mean)
    assert_allclose(p1.var_pred_mean, p2.var_pred_mean)

    p1 = res1.get_prediction(start=20, dynamic=True, end=60, exog=x_fcast1,
                             signal_only=True, information_set='filtered')
    p2 = res2.get_prediction(start=20, end=60, exog=x_fcast2, signal_only=True,
                             information_set='filtered')
    assert_allclose(p1.predicted_mean, p2.predicted_mean)
    assert_allclose(p1.var_pred_mean, p2.var_pred_mean)
