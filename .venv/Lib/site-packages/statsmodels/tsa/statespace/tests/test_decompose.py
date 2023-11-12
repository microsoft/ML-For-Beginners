"""
Tests for decomposition of objects in state space models

Author: Chad Fulton
License: Simplified-BSD
"""

import pytest

import numpy as np
import pandas as pd

from numpy.testing import assert_allclose

from statsmodels import datasets
from statsmodels.tsa.statespace import sarimax, varmax, dynamic_factor_mq
from statsmodels.tsa.statespace.tests.test_impulse_responses import TVSS


dta = datasets.macrodata.load_pandas().data
dta.index = pd.period_range(start='1959Q1', end='2009Q3', freq='Q')


@pytest.mark.parametrize('use_exog', [False, True])
@pytest.mark.parametrize('trend', ['n', 'c', 't'])
@pytest.mark.parametrize('concentrate_scale', [False, True])
@pytest.mark.parametrize('measurement_error', [False, True])
def test_smoothed_decomposition_sarimax(use_exog, trend, concentrate_scale,
                                        measurement_error):
    endog = np.array([[0.2, np.nan, 1.2, -0.3, -1.5]]).T
    exog = np.array([2, 5.3, -1, 3.4, 0.]) if use_exog else None

    trend_params = [0.1]
    ar_params = [0.5]
    exog_params = [1.4]
    meas_err_params = [1.2]
    cov_params = [0.8]

    params = []
    if trend in ['c', 't']:
        params += trend_params
    if use_exog:
        params += exog_params
    params += ar_params
    if measurement_error:
        params += meas_err_params
    if not concentrate_scale:
        params += cov_params

    # Fit the models
    mod = sarimax.SARIMAX(endog, order=(1, 0, 0), trend=trend,
                          exog=exog if use_exog else None,
                          concentrate_scale=concentrate_scale,
                          measurement_error=measurement_error)
    prior_mean = np.array([-0.4])
    prior_cov = np.eye(1) * 1.2
    mod.ssm.initialize_known(prior_mean, prior_cov)
    res = mod.smooth(params)

    # Check smoothed state

    # Get the decomposition of the smoothed state
    cd, coi, csi, cp = res.get_smoothed_decomposition(
        decomposition_of='smoothed_state')

    # Sum across contributions (i.e. from observations at each time period and
    # from the initial state)
    css = ((cd + coi).sum(axis=1) + csi.sum(axis=1) + cp.sum(axis=1))
    css = css.unstack(level='state_to').values

    # Summing up all contributions should yield the actual smoothed state,
    # so the smoothed state vector is the desired result of this test
    ss = np.array(res.states.smoothed)

    assert_allclose(css, ss, atol=1e-12)

    # Check smoothed signal

    # Use the summed state contributions and multiply by the design matrix
    # to get the smoothed signal
    csf = ((css.T * mod['design'][:, :, None]).sum(axis=1)
           + mod['obs_intercept']).T

    # Summing up all contributions should yield the smoothed prediction of
    # the observed variables
    s_sig = res.predict(information_set='smoothed', signal_only=True)
    sf = res.predict(information_set='smoothed', signal_only=False)

    assert_allclose(csf[:, 0], sf)

    # Now check the smoothed signal against the sum computed from the
    # decomposed smoothed signal
    cd, coi, csi, cp = res.get_smoothed_decomposition(
        decomposition_of='smoothed_signal')

    # Sum across contributions (i.e. from observations and intercepts at each
    # time period and from the initial state) to get the smoothed signal
    cs_sig = ((cd + coi).sum(axis=1) + csi.sum(axis=1) + cp.sum(axis=1))
    cs_sig = cs_sig.unstack(level='variable_to').values

    assert_allclose(cs_sig[:, 0], s_sig, atol=1e-12)

    # Add in the observation intercept to get the smoothed forecast
    csf = cs_sig + mod['obs_intercept'].T

    assert_allclose(csf[:, 0], sf)


@pytest.mark.parametrize('use_exog', [False, True])
@pytest.mark.parametrize('trend', ['n', 'c', 't'])
def test_smoothed_decomposition_varmax(use_exog, trend):
    endog = np.array([[0.2, 1.0],
                      [-0.3, -0.5],
                      [0.01, 0.4],
                      [-0.4, 0.1],
                      [0.1, 0.05]])
    endog[0, 0] = np.nan
    endog[1, :] = np.nan
    endog[2, 1] = np.nan
    exog = np.array([2, 5.3, -1, 3.4, 0.]) if use_exog else None

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

    # Fit the model
    mod = varmax.VARMAX(endog, order=(1, 0), trend=trend,
                        exog=exog if use_exog else None)
    prior_mean = np.array([-0.4, 0.9])
    prior_cov = np.array([[1.4, 0.3],
                          [0.3, 2.6]])
    mod.ssm.initialize_known(prior_mean, prior_cov)
    res = mod.smooth(params)

    # Check smoothed state

    # Get the decomposition of the smoothed state
    cd, coi, csi, cp = res.get_smoothed_decomposition(
        decomposition_of='smoothed_state')

    # Sum across contributions (i.e. from observations at each time period and
    # from the initial state)
    css = ((cd + coi).sum(axis=1) + csi.sum(axis=1) + cp.sum(axis=1))
    css = css.unstack(level='state_to').values

    # Summing up all contributions should yield the actual smoothed state,
    # so the smoothed state vector is the desired result of this test
    ss = np.array(res.states.smoothed)

    assert_allclose(css, ss, atol=1e-12)

    # Check smoothed signal

    # Use the summed state contributions and multiply by the design matrix
    # to get the smoothed signal
    csf = (css.T * mod['design'][:, :, None]).sum(axis=1).T

    # Summing up all contributions should yield the smoothed prediction of
    # the observed variables
    s_sig = res.predict(information_set='smoothed', signal_only=True)
    sf = res.predict(information_set='smoothed', signal_only=False)

    assert_allclose(csf, sf, atol=1e-12)

    # Now check the smoothed signal against the sum computed from the
    # decomposed smoothed signal
    cd, coi, csi, cp = res.get_smoothed_decomposition(
        decomposition_of='smoothed_signal')

    # Sum across contributions (i.e. from observations and intercepts at each
    # time period and from the initial state) to get the smoothed signal
    cs_sig = ((cd + coi).sum(axis=1) + csi.sum(axis=1) + cp.sum(axis=1))
    cs_sig = cs_sig.unstack(level='variable_to').values

    assert_allclose(cs_sig, s_sig, atol=1e-12)

    # Add in the observation intercept to get the smoothed forecast
    csf = cs_sig + mod['obs_intercept'].T

    assert_allclose(csf, sf)


def test_smoothed_decomposition_dfm_mq():
    # Create the datasets
    index_M = pd.period_range(start='2000', periods=12, freq='M')
    index_Q = pd.period_range(start='2000', periods=4, freq='Q')

    dta_M = pd.DataFrame(np.zeros((12, 2)), index=index_M,
                         columns=['M0', 'M1'])
    dta_Q = pd.DataFrame(np.zeros((4, 2)), index=index_Q, columns=['Q0', 'Q1'])
    # Add some noise so the variables aren't constants
    dta_M.iloc[0] = 1.
    dta_Q.iloc[1] = 1.
    # TODO: remove this once we have the intercept contributions figured out
    dta_M -= dta_M.mean()
    dta_Q -= dta_Q.mean()

    # Create the model instance
    mod = dynamic_factor_mq.DynamicFactorMQ(
        dta_M, endog_quarterly=dta_Q, factors=1, factor_orders=1,
        idiosyncratic_ar1=True)
    params = [
        0.1, -0.4, 0.2, 0.3,   # loadings
        0.95, 1.0,             # factor
        0.5, 0.55, 0.6, 0.65,  # idio ar(1)
        1.1, 1.2, 1.0, 0.9,    # idio variances
    ]
    res = mod.smooth(params)

    # Check smoothed state

    # Get the decomposition of the smoothed state
    cd, coi, csi, cp = res.get_smoothed_decomposition(
        decomposition_of='smoothed_state')

    # Sum across contributions (i.e. from observations at each time period and
    # from the initial state)
    css = ((cd + coi).sum(axis=1) + csi.sum(axis=1) + cp.sum(axis=1))
    css = css.unstack(level='state_to')[mod.state_names].values

    # Summing up all contributions should yield the actual smoothed state,
    # so the smoothed state vector is the desired result of this test
    ss = np.array(res.states.smoothed)

    assert_allclose(css, ss, atol=1e-12)

    # Check smoothed signal

    # Use the summed state contributions and multiply by the design matrix
    # to get the smoothed signal
    csf = (css.T * mod['design'][:, :, None]).sum(axis=1).T
    # Reverse the standardization
    csf = (csf.T * mod._endog_std.values[:, None]
           + mod._endog_mean.values[:, None]).T

    # Summing up all contributions should yield the smoothed prediction of
    # the observed variables
    s_sig = res.predict(information_set='smoothed', signal_only=True)
    sf = res.predict(information_set='smoothed', signal_only=False)

    assert_allclose(csf, sf, atol=1e-12)

    # Now check the smoothed signal against the sum computed from the
    # decomposed smoothed signal
    cd, coi, csi, cp = res.get_smoothed_decomposition(
        decomposition_of='smoothed_signal')

    # Sum across contributions (i.e. from observations and intercepts at each
    # time period and from the initial state) to get the smoothed signal
    cs_sig = ((cd + coi).sum(axis=1) + csi.sum(axis=1) + cp.sum(axis=1))
    cs_sig = cs_sig.unstack(level='variable_to')[mod.endog_names].values

    assert_allclose(cs_sig, s_sig, atol=1e-12)

    # Add in the observation intercept to get the smoothed forecast
    csf = cs_sig + mod['obs_intercept'].T

    assert_allclose(csf, sf)


@pytest.mark.parametrize('univariate', [False, True])
def test_smoothed_decomposition_TVSS(univariate, reset_randomstate):
    endog = np.zeros((10, 3))
    endog[6, 0] = np.nan
    endog[7, :] = np.nan
    endog[8, 1] = np.nan
    mod = TVSS(endog)

    mod['state_intercept'] = np.random.normal(size=(mod.k_states, mod.nobs))

    prior_mean = np.array([1.2, 0.8])
    prior_cov = np.eye(2)
    mod.ssm.initialize_known(prior_mean, prior_cov)
    if univariate:
        mod.ssm.filter_univariate = True
    res = mod.smooth([])

    # Check smoothed state

    # Get the decomposition of the smoothed state
    cd, coi, csi, cp = res.get_smoothed_decomposition(
        decomposition_of='smoothed_state')

    # Sum across contributions (i.e. from observations at each time period and
    # from the initial state)
    css = ((cd + coi).sum(axis=1) + csi.sum(axis=1) + cp.sum(axis=1))
    css = css.unstack(level='state_to')[mod.state_names].values

    # Summing up all contributions should yield the actual smoothed state,
    # so the smoothed state vector is the desired result of this test
    ss = np.array(res.states.smoothed)

    assert_allclose(css, ss, atol=1e-12)

    # Check smoothed signal

    # Use the summed state contributions and multiply by the design matrix
    # to get the smoothed signal
    cs_sig = (css.T * mod['design']).sum(axis=1).T

    # Add in the observation intercept to get the smoothed forecast
    csf = cs_sig + mod['obs_intercept'].T

    # Summing up all contributions should yield the smoothed prediction of
    # the observed variables
    s_sig = res.predict(information_set='smoothed', signal_only=True)
    sf = res.predict(information_set='smoothed', signal_only=False)

    assert_allclose(cs_sig, s_sig, atol=1e-12)
    assert_allclose(csf, sf, atol=1e-12)

    # Now check the smoothed signal against the sum computed from the
    # decomposed smoothed signal
    cd, coi, csi, cp = res.get_smoothed_decomposition(
        decomposition_of='smoothed_signal')

    # Sum across contributions (i.e. from observations and intercepts at each
    # time period and from the initial state) to get the smoothed signal
    cs_sig = ((cd + coi).sum(axis=1) + csi.sum(axis=1) + cp.sum(axis=1))
    cs_sig = cs_sig.unstack(level='variable_to')[mod.endog_names].values

    assert_allclose(cs_sig, s_sig, atol=1e-12)

    # Add in the observation intercept to get the smoothed forecast
    csf = cs_sig + mod['obs_intercept'].T

    assert_allclose(csf, sf)
