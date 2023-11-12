"""
Tests for concentrating the scale out of the loglikelihood function

Note: the univariate cases is well tested in test_sarimax.py

Author: Chad Fulton
License: Simplified-BSD
"""

import numpy as np
import pandas as pd
from statsmodels.tools.tools import Bunch
from .results import results_varmax
from statsmodels.tsa.statespace import sarimax, varmax
from numpy.testing import assert_raises, assert_allclose


def get_sarimax_models(endog, filter_univariate=False, **kwargs):
    kwargs.setdefault('tolerance', 0)
    # Construct a concentrated version of the given SARIMAX model, and get
    # the estimate of the scale
    mod_conc = sarimax.SARIMAX(endog, **kwargs)
    mod_conc.ssm.filter_concentrated = True
    mod_conc.ssm.filter_univariate = filter_univariate
    params_conc = mod_conc.start_params
    params_conc[-1] = 1
    res_conc = mod_conc.smooth(params_conc)
    scale = res_conc.scale

    # Construct the non-concentrated version
    mod_orig = sarimax.SARIMAX(endog, **kwargs)
    mod_orig.ssm.filter_univariate = filter_univariate
    params_orig = params_conc.copy()
    k_vars = 1 + kwargs.get('measurement_error', False)
    params_orig[-k_vars:] = scale * params_conc[-k_vars:]
    res_orig = mod_orig.smooth(params_orig)

    return Bunch(**{'mod_conc': mod_conc, 'params_conc': params_conc,
                    'mod_orig': mod_orig, 'params_orig': params_orig,
                    'res_conc': res_conc, 'res_orig': res_orig,
                    'scale': scale})


def test_concentrated_loglike_sarimax():
    # Note: we will not use the "concentrate_scale" option to SARIMAX for this
    # test, which is a lower-level test of the Kalman filter using the SARIMAX
    # model as an example
    nobs = 30
    np.random.seed(28953)
    endog = np.random.normal(size=nobs)
    kwargs = {}

    # Typical model
    out = get_sarimax_models(endog)
    assert_allclose(out.res_conc.llf, out.res_orig.llf)
    assert_allclose(out.res_conc.llf_obs, out.res_orig.llf_obs)
    assert_allclose(out.mod_conc.loglike(out.params_conc),
                    out.mod_orig.loglike(out.params_orig))
    assert_allclose(out.mod_conc.loglikeobs(out.params_conc),
                    out.mod_orig.loglikeobs(out.params_orig))

    # Add missing entries
    endog[2:10] = np.nan
    out = get_sarimax_models(endog)
    assert_allclose(out.res_conc.llf, out.res_orig.llf)
    assert_allclose(out.res_conc.llf_obs, out.res_orig.llf_obs)
    assert_allclose(out.mod_conc.loglike(out.params_conc),
                    out.mod_orig.loglike(out.params_orig))
    assert_allclose(out.mod_conc.loglikeobs(out.params_conc),
                    out.mod_orig.loglikeobs(out.params_orig))

    # Add seasonal differencing
    # Note: now, due to differences in approximate diffuse initialization,
    # we may have differences in the first 2 observations (but notice that
    # this does not affect the computed joint log-likelihood because those
    # observations are not included there)
    kwargs['seasonal_order'] = (1, 1, 1, 2)
    out = get_sarimax_models(endog, **kwargs)
    assert_allclose(out.res_conc.llf, out.res_orig.llf)
    assert_allclose(out.res_conc.llf_obs[2:], out.res_orig.llf_obs[2:])
    assert_allclose(out.mod_conc.loglike(out.params_conc),
                    out.mod_orig.loglike(out.params_orig))
    assert_allclose(out.mod_conc.loglikeobs(out.params_conc)[2:],
                    out.mod_orig.loglikeobs(out.params_orig)[2:])

    # Add loglikelihood burn, trend, and exog
    kwargs['loglikelihood_burn'] = 5
    kwargs['trend'] = 'c'
    kwargs['exog'] = np.arange(nobs)
    out = get_sarimax_models(endog, **kwargs)
    assert_allclose(out.res_conc.llf, out.res_orig.llf)
    assert_allclose(out.res_conc.llf_obs[2:], out.res_orig.llf_obs[2:])
    assert_allclose(out.mod_conc.loglike(out.params_conc),
                    out.mod_orig.loglike(out.params_orig))
    assert_allclose(out.mod_conc.loglikeobs(out.params_conc)[2:],
                    out.mod_orig.loglikeobs(out.params_orig)[2:])


def test_concentrated_predict_sarimax():
    # Note: we will not use the "concentrate_scale" option to SARIMAX for this
    # test, which is a lower-level test of the Kalman filter using the SARIMAX
    # model as an example
    nobs = 30
    np.random.seed(28953)
    endog = np.random.normal(size=nobs)

    # Typical model
    out = get_sarimax_models(endog)
    assert_allclose(out.res_conc.predict(), out.res_orig.predict())
    assert_allclose(out.res_conc.forecast(5), out.res_orig.forecast(5))
    assert_allclose(out.res_conc.predict(start=0, end=45, dynamic=10),
                    out.res_orig.predict(start=0, end=45, dynamic=10))


def test_fixed_scale_sarimax():
    # Test that the fixed_scale context manager works
    nobs = 30
    np.random.seed(28953)
    endog = np.random.normal(size=nobs)
    kwargs = {
        'seasonal_order': (1, 1, 1, 2),
        'trend': 'ct',
        'exog': np.sin(np.arange(nobs))
    }

    # Construct a concentrated version of the given SARIMAX model
    mod_conc = sarimax.SARIMAX(endog, concentrate_scale=True, **kwargs)

    # Construct the non-concentrated version
    mod_orig = sarimax.SARIMAX(endog, **kwargs)

    # Modify the scale parameter
    params = mod_orig.start_params
    params[-1] *= 1.2

    # Test that these are not equal in the default computation
    assert_raises(AssertionError, assert_allclose,
                  mod_conc.loglike(params[:-1]), mod_orig.loglike(params))

    # Now test that the llf is equal when we use the fixed_scale decorator
    with mod_conc.ssm.fixed_scale(params[-1]):
        res1 = mod_conc.smooth(params[:-1])
        llf1 = mod_conc.loglike(params[:-1])
        llf_obs1 = mod_conc.loglikeobs(params[:-1])

    res2 = mod_orig.smooth(params)
    llf2 = mod_orig.loglike(params)
    llf_obs2 = mod_orig.loglikeobs(params)

    assert_allclose(res1.llf, res2.llf)
    assert_allclose(res1.llf_obs[2:], res2.llf_obs[2:])
    assert_allclose(llf1, llf2)
    assert_allclose(llf_obs1, llf_obs2)


def check_concentrated_scale(filter_univariate=False, missing=False, **kwargs):
    # Test that concentrating the scale out of the likelihood function works
    index = pd.date_range('1960-01-01', '1982-10-01', freq='QS')
    dta = pd.DataFrame(results_varmax.lutkepohl_data,
                       columns=['inv', 'inc', 'consump'], index=index)
    dta['dln_inv'] = np.log(dta['inv']).diff()
    dta['dln_inc'] = np.log(dta['inc']).diff()
    dta['dln_consump'] = np.log(dta['consump']).diff()

    endog = dta.loc['1960-04-01':'1978-10-01', ['dln_inv', 'dln_inc']]

    # Optionally add some missing observations
    if missing:
        endog.iloc[0, 0] = np.nan
        endog.iloc[3:5, :] = np.nan
        endog.iloc[8, 1] = np.nan

    # Sometimes we can have slight differences if the Kalman filters
    # converge at different observations, so disable convergence.
    kwargs.update({'tolerance': 0})

    mod_orig = varmax.VARMAX(endog, **kwargs)
    mod_conc = varmax.VARMAX(endog, **kwargs)
    mod_conc.ssm.filter_concentrated = True

    mod_orig.ssm.filter_univariate = filter_univariate
    mod_conc.ssm.filter_univariate = filter_univariate

    # Since VARMAX does not explicitly allow concentrating out the scale, for
    # now we will simulate it by setting the first variance to be 1.
    # Note that start_scale will not be the scale used for the non-concentrated
    # model, because we need to use the MLE scale estimated by the
    # concentrated model.
    conc_params = mod_conc.start_params
    start_scale = conc_params[mod_conc._params_state_cov][0]
    if kwargs.get('error_cov_type', 'unstructured') == 'diagonal':
        conc_params[mod_conc._params_state_cov] /= start_scale
    else:
        conc_params[mod_conc._params_state_cov] /= start_scale**0.5

    # Concentrated model smoothing
    res_conc = mod_conc.smooth(conc_params)
    scale = res_conc.scale

    # Map the concentrated parameters to the non-concentrated model
    orig_params = conc_params.copy()
    if kwargs.get('error_cov_type', 'unstructured') == 'diagonal':
        orig_params[mod_orig._params_state_cov] *= scale
    else:
        orig_params[mod_orig._params_state_cov] *= scale**0.5

    # Measurement error variances also get rescaled
    orig_params[mod_orig._params_obs_cov] *= scale

    # Non-oncentrated model smoothing
    res_orig = mod_orig.smooth(orig_params)

    # Test loglike
    # Need to reduce the tolerance when we have measurement error.
    assert_allclose(res_conc.llf, res_orig.llf)

    # Test state space representation matrices
    for name in mod_conc.ssm.shapes:
        if name == 'obs':
            continue
        assert_allclose(getattr(res_conc.filter_results, name),
                        getattr(res_orig.filter_results, name))

    # Test filter / smoother output
    scale = res_conc.scale
    d = res_conc.loglikelihood_burn

    filter_attr = ['predicted_state', 'filtered_state', 'forecasts',
                   'forecasts_error', 'kalman_gain']

    for name in filter_attr:
        actual = getattr(res_conc.filter_results, name)
        desired = getattr(res_orig.filter_results, name)
        assert_allclose(actual, desired, atol=1e-7)

    # Note: do not want to compare the elements from any diffuse
    # initialization for things like covariances, so only compare for
    # periods past the loglikelihood_burn period
    filter_attr_burn = ['standardized_forecasts_error',
                        'predicted_state_cov', 'filtered_state_cov',
                        'tmp1', 'tmp2', 'tmp3', 'tmp4']

    for name in filter_attr_burn:
        actual = getattr(res_conc.filter_results, name)[..., d:]
        desired = getattr(res_orig.filter_results, name)[..., d:]
        assert_allclose(actual, desired, atol=1e-7)

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
        actual = getattr(res_conc.filter_results, name)
        desired = getattr(res_orig.filter_results, name)
        assert_allclose(actual, desired, atol=1e-7)

    # Test prediction output
    nobs = mod_conc.nobs
    pred_conc = res_conc.get_prediction(start=10, end=nobs + 50, dynamic=40)
    pred_orig = res_conc.get_prediction(start=10, end=nobs + 50, dynamic=40)

    assert_allclose(pred_conc.predicted_mean, pred_orig.predicted_mean)
    assert_allclose(pred_conc.se_mean, pred_orig.se_mean)


def test_concentrated_scale_conventional():
    check_concentrated_scale(filter_univariate=False)
    check_concentrated_scale(filter_univariate=False, measurement_error=True)
    check_concentrated_scale(filter_univariate=False,
                             error_cov_type='diagonal')
    check_concentrated_scale(filter_univariate=False, missing=True)
    check_concentrated_scale(filter_univariate=False, missing=True,
                             loglikelihood_burn=10)


def test_concentrated_scale_univariate():
    check_concentrated_scale(filter_univariate=True)
    check_concentrated_scale(filter_univariate=True, measurement_error=True)
    check_concentrated_scale(filter_univariate=True, error_cov_type='diagonal')
    check_concentrated_scale(filter_univariate=True, missing=True)
    check_concentrated_scale(filter_univariate=True, missing=True,
                             loglikelihood_burn=10)
