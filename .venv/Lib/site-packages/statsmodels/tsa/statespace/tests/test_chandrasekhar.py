"""
Tests for Chandrasekhar recursions

Author: Chad Fulton
License: Simplified-BSD
"""

import numpy as np
import pandas as pd
from .results import results_varmax
from statsmodels.tsa.statespace import sarimax, varmax
from statsmodels.tsa.statespace.kalman_filter import (
    MEMORY_CONSERVE, MEMORY_NO_LIKELIHOOD)
from numpy.testing import assert_allclose

import pytest


def check_output(res_chand, res_orig, memory_conserve=False):
    # Test loglike
    params = res_orig.params
    assert_allclose(res_chand.llf, res_orig.llf)
    assert_allclose(
        res_chand.model.score_obs(params),
        res_orig.model.score_obs(params),
        rtol=5e-5,
        atol=5e-6
    )

    # Test state space representation matrices
    for name in res_chand.model.ssm.shapes:
        if name == 'obs':
            continue
        assert_allclose(getattr(res_chand.filter_results, name),
                        getattr(res_orig.filter_results, name))

    # Test filter / smoother output
    filter_attr = ['predicted_state', 'filtered_state', 'forecasts',
                   'forecasts_error']
    # Can only check kalman gain if we didn't use memory conservation
    if not memory_conserve:
        filter_attr += ['kalman_gain']

    for name in filter_attr:
        actual = getattr(res_chand.filter_results, name)
        desired = getattr(res_orig.filter_results, name)
        assert_allclose(actual, desired, atol=1e-12)

    filter_attr_burn = ['predicted_state_cov', 'filtered_state_cov']

    # Can only check kalman gain if we didn't use memory conservation
    if not memory_conserve:
        filter_attr += ['standardized_forecasts_error', 'tmp1', 'tmp2', 'tmp3',
                        'tmp4']

    for name in filter_attr_burn:
        actual = getattr(res_chand.filter_results, name)
        desired = getattr(res_orig.filter_results, name)
        assert_allclose(actual, desired, atol=1e-12)

    if not memory_conserve:
        smoothed_attr = ['smoothed_state', 'smoothed_state_cov',
                         'smoothed_state_autocov',
                         'smoothed_state_disturbance',
                         'smoothed_state_disturbance_cov',
                         'smoothed_measurement_disturbance',
                         'smoothed_measurement_disturbance_cov',
                         'scaled_smoothed_estimator',
                         'scaled_smoothed_estimator_cov', 'smoothing_error',
                         'smoothed_forecasts', 'smoothed_forecasts_error',
                         'smoothed_forecasts_error_cov']

        for name in smoothed_attr:
            actual = getattr(res_chand.filter_results, name)
            desired = getattr(res_orig.filter_results, name)
            assert_allclose(actual, desired, atol=1e-12)

    # Test prediction output
    nobs = res_chand.model.nobs
    if not memory_conserve:
        pred_chand = res_chand.get_prediction(start=10, end=nobs + 50,
                                              dynamic=40)
        pred_orig = res_chand.get_prediction(start=10, end=nobs + 50,
                                             dynamic=40)
    else:
        # In the memory conservation case, we can't do dynamic prediction
        pred_chand = res_chand.get_prediction(start=10, end=nobs + 50)
        pred_orig = res_chand.get_prediction(start=10, end=nobs + 50)

    assert_allclose(pred_chand.predicted_mean, pred_orig.predicted_mean)
    assert_allclose(pred_chand.se_mean, pred_orig.se_mean)


def check_univariate_chandrasekhar(filter_univariate=False, **kwargs):
    # Test that Chandrasekhar recursions don't change the output
    index = pd.date_range('1960-01-01', '1982-10-01', freq='QS')
    dta = pd.DataFrame(results_varmax.lutkepohl_data,
                       columns=['inv', 'inc', 'consump'], index=index)
    endog = np.log(dta['inv']).diff().loc['1960-04-01':'1978-10-01']

    mod_orig = sarimax.SARIMAX(endog, **kwargs)
    mod_chand = sarimax.SARIMAX(endog, **kwargs)
    mod_chand.ssm.filter_chandrasekhar = True

    params = mod_orig.start_params

    mod_orig.ssm.filter_univariate = filter_univariate
    mod_chand.ssm.filter_univariate = filter_univariate

    res_chand = mod_chand.smooth(params)

    # Non-oncentrated model smoothing
    res_orig = mod_orig.smooth(params)

    check_output(res_chand, res_orig)


def check_multivariate_chandrasekhar(filter_univariate=False,
                                     gen_obs_cov=False, memory_conserve=False,
                                     **kwargs):
    # Test that Chandrasekhar recursions don't change the output
    index = pd.date_range('1960-01-01', '1982-10-01', freq='QS')
    dta = pd.DataFrame(results_varmax.lutkepohl_data,
                       columns=['inv', 'inc', 'consump'], index=index)
    dta['dln_inv'] = np.log(dta['inv']).diff()
    dta['dln_inc'] = np.log(dta['inc']).diff()
    dta['dln_consump'] = np.log(dta['consump']).diff()

    endog = dta.loc['1960-04-01':'1978-10-01', ['dln_inv', 'dln_inc']]

    mod_orig = varmax.VARMAX(endog, **kwargs)
    mod_chand = varmax.VARMAX(endog, **kwargs)
    mod_chand.ssm.filter_chandrasekhar = True

    params = mod_orig.start_params

    mod_orig.ssm.filter_univariate = filter_univariate
    mod_chand.ssm.filter_univariate = filter_univariate

    if gen_obs_cov:
        mod_orig['obs_cov'] = np.array([[1., 0.5],
                                        [0.5, 1.]])
        mod_chand['obs_cov'] = np.array([[1., 0.5],
                                        [0.5, 1.]])

    if memory_conserve:
        mod_orig.ssm.set_conserve_memory(
            MEMORY_CONSERVE & ~ MEMORY_NO_LIKELIHOOD)
        mod_chand.ssm.set_conserve_memory(
            MEMORY_CONSERVE & ~ MEMORY_NO_LIKELIHOOD)

        res_chand = mod_chand.filter(params)
        res_orig = mod_orig.filter(params)
    else:
        res_chand = mod_chand.smooth(params)
        res_orig = mod_orig.smooth(params)

    check_output(res_chand, res_orig, memory_conserve=memory_conserve)


def test_chandrasekhar_conventional():
    check_univariate_chandrasekhar(filter_univariate=False)
    check_univariate_chandrasekhar(filter_univariate=False,
                                   concentrate_scale=True)

    check_multivariate_chandrasekhar(filter_univariate=False)
    check_multivariate_chandrasekhar(filter_univariate=False,
                                     measurement_error=True)
    check_multivariate_chandrasekhar(filter_univariate=False,
                                     error_cov_type='diagonal')
    check_multivariate_chandrasekhar(filter_univariate=False,
                                     gen_obs_cov=True)
    check_multivariate_chandrasekhar(filter_univariate=False,
                                     gen_obs_cov=True, memory_conserve=True)


def test_chandrasekhar_univariate():
    check_univariate_chandrasekhar(filter_univariate=True)
    check_univariate_chandrasekhar(filter_univariate=True,
                                   concentrate_scale=True)

    check_multivariate_chandrasekhar(filter_univariate=True)
    check_multivariate_chandrasekhar(filter_univariate=True,
                                     measurement_error=True)
    check_multivariate_chandrasekhar(filter_univariate=True,
                                     error_cov_type='diagonal')
    check_multivariate_chandrasekhar(filter_univariate=True,
                                     gen_obs_cov=True)
    check_multivariate_chandrasekhar(filter_univariate=True,
                                     gen_obs_cov=True, memory_conserve=True)


def test_invalid():
    # Tests that trying to use the Chandrasekhar recursions in invalid
    # situations raises an error

    # Missing values
    endog = np.zeros(10)
    endog[1] = np.nan
    mod = sarimax.SARIMAX(endog)
    mod.ssm.filter_chandrasekhar = True
    with pytest.raises(RuntimeError, match=('Cannot use Chandrasekhar'
                                            ' recursions with missing data.')):
        mod.filter([0.5, 1.0])

    # Alternative timing
    endog = np.zeros(10)
    mod = sarimax.SARIMAX(endog)
    mod.ssm.filter_chandrasekhar = True
    mod.ssm.timing_init_filtered = True
    with pytest.raises(RuntimeError, match=('Cannot use Chandrasekhar'
                                            ' recursions with filtered'
                                            ' timing.')):
        mod.filter([0.5, 1.0])

    # Time-varying matrices
    endog = np.zeros(10)
    mod = sarimax.SARIMAX(endog)
    mod.ssm.filter_chandrasekhar = True
    mod['obs_cov'] = np.ones((1, 1, 10))
    with pytest.raises(RuntimeError, match=('Cannot use Chandrasekhar'
                                            ' recursions with time-varying'
                                            r' system matrices \(except for'
                                            r' intercept terms\).')):
        mod.filter([0.5, 1.0])
