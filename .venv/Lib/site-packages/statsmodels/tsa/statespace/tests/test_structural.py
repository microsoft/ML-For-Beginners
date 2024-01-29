"""
Tests for structural time series models

Author: Chad Fulton
License: Simplified-BSD
"""

import warnings

import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_raises
import pandas as pd
import pytest

from statsmodels.datasets import macrodata
from statsmodels.tools.sm_exceptions import SpecificationWarning
from statsmodels.tsa.statespace import structural
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.statespace.tests.results import results_structural

dta = macrodata.load_pandas().data
dta.index = pd.date_range(start='1959-01-01', end='2009-07-01', freq='QS')


def run_ucm(name, use_exact_diffuse=False):
    true = getattr(results_structural, name)

    for model in true['models']:
        kwargs = model.copy()
        kwargs.update(true['kwargs'])
        kwargs['use_exact_diffuse'] = use_exact_diffuse

        # Make a copy of the data
        values = dta.copy()

        freq = kwargs.pop('freq', None)
        if freq is not None:
            values.index = pd.date_range(start='1959-01-01', periods=len(dta),
                                         freq=freq)

        # Test pandas exog
        if 'exog' in kwargs:
            # Default value here is pd.Series object
            exog = np.log(values['realgdp'])

            # Also allow a check with a 1-dim numpy array
            if kwargs['exog'] == 'numpy':
                exog = exog.values.squeeze()

            kwargs['exog'] = exog

        # Create the model
        mod = UnobservedComponents(values['unemp'], **kwargs)

        # Smoke test for starting parameters, untransform, transform
        # Also test that transform and untransform are inverses
        mod.start_params
        roundtrip = mod.transform_params(
            mod.untransform_params(mod.start_params))
        assert_allclose(mod.start_params, roundtrip)

        # Fit the model at the true parameters
        res_true = mod.filter(true['params'])

        # Check that the cycle bounds were computed correctly
        freqstr = freq[0] if freq is not None else values.index.freqstr[0]
        if 'cycle_period_bounds' in kwargs:
            cycle_period_bounds = kwargs['cycle_period_bounds']
        elif freqstr in ('A', 'AS', 'Y', 'YS'):
            cycle_period_bounds = (1.5, 12)
        elif freqstr in ('Q', 'QS'):
            cycle_period_bounds = (1.5*4, 12*4)
        elif freqstr in ('M', 'MS'):
            cycle_period_bounds = (1.5*12, 12*12)
        else:
            # If we have no information on data frequency, require the
            # cycle frequency to be between 0 and pi
            cycle_period_bounds = (2, np.inf)

        # Test that the cycle frequency bound is correct
        assert_equal(mod.cycle_frequency_bound,
                     (2*np.pi / cycle_period_bounds[1],
                      2*np.pi / cycle_period_bounds[0]))

        # Test that the likelihood is correct
        rtol = true.get('rtol', 1e-7)
        atol = true.get('atol', 0)

        if use_exact_diffuse:
            # If we are using exact diffuse initialization, then we need to
            # adjust for the fact that KFAS does not include the constant in
            # the likelihood function for the diffuse periods
            # (see note to test_exact_diffuse_filtering.py for details).
            res_llf = (res_true.llf_obs.sum()
                       + res_true.nobs_diffuse * 0.5 * np.log(2 * np.pi))
        else:
            # If we are using approximate diffuse initialization, then we need
            # to ignore the first period, and this will agree with KFAS (since
            # it does not include the constant in the likelihood function for
            # diffuse periods).
            res_llf = res_true.llf_obs[res_true.loglikelihood_burn:].sum()

        assert_allclose(res_llf, true['llf'], rtol=rtol, atol=atol)

        # Optional smoke test for plot_components
        try:
            import matplotlib.pyplot as plt
            try:
                from pandas.plotting import register_matplotlib_converters
                register_matplotlib_converters()
            except ImportError:
                pass
            fig = plt.figure()
            res_true.plot_components(fig=fig)
        except ImportError:
            pass

        # Now fit the model via MLE
        with warnings.catch_warnings(record=True):
            fit_kwargs = {}
            if 'maxiter' in true:
                fit_kwargs['maxiter'] = true['maxiter']
            res = mod.fit(start_params=true.get('start_params', None),
                          disp=-1, **fit_kwargs)
            # If we found a higher likelihood, no problem; otherwise check
            # that we're very close to that found by R

            # See note above about these computation
            if use_exact_diffuse:
                res_llf = (res.llf_obs.sum()
                           + res.nobs_diffuse * 0.5 * np.log(2 * np.pi))
            else:
                res_llf = res.llf_obs[res_true.loglikelihood_burn:].sum()

            if res_llf <= true['llf']:
                assert_allclose(res_llf, true['llf'], rtol=1e-4)

            # Smoke test for summary
            res.summary()


def test_irregular(close_figures):
    run_ucm('irregular')
    run_ucm('irregular', use_exact_diffuse=True)


def test_fixed_intercept(close_figures):
    # Clear warnings
    structural.__warningregistry__ = {}
    warning = SpecificationWarning
    match = 'Specified model does not contain'
    with pytest.warns(warning, match=match):
        run_ucm('fixed_intercept')
        run_ucm('fixed_intercept', use_exact_diffuse=True)


def test_deterministic_constant(close_figures):
    run_ucm('deterministic_constant')
    run_ucm('deterministic_constant', use_exact_diffuse=True)


def test_random_walk(close_figures):
    run_ucm('random_walk')
    run_ucm('random_walk', use_exact_diffuse=True)


def test_local_level(close_figures):
    run_ucm('local_level')
    run_ucm('local_level', use_exact_diffuse=True)


def test_fixed_slope(close_figures):
    warning = SpecificationWarning
    match = 'irregular component added'
    with pytest.warns(warning, match=match):
        run_ucm('fixed_slope')
        run_ucm('fixed_slope', use_exact_diffuse=True)


def test_fixed_slope_warn(close_figures):
    # Clear warnings
    structural.__warningregistry__ = {}

    warning = SpecificationWarning
    match = 'irregular component added'
    with pytest.warns(warning, match=match):
        run_ucm('fixed_slope')
        run_ucm('fixed_slope', use_exact_diffuse=True)


def test_deterministic_trend(close_figures):
    run_ucm('deterministic_trend')
    run_ucm('deterministic_trend', use_exact_diffuse=True)


def test_random_walk_with_drift(close_figures):
    run_ucm('random_walk_with_drift')
    run_ucm('random_walk_with_drift', use_exact_diffuse=True)


def test_local_linear_deterministic_trend(close_figures):
    run_ucm('local_linear_deterministic_trend')
    run_ucm('local_linear_deterministic_trend', use_exact_diffuse=True)


def test_local_linear_trend(close_figures):
    run_ucm('local_linear_trend')
    run_ucm('local_linear_trend', use_exact_diffuse=True)


def test_smooth_trend(close_figures):
    run_ucm('smooth_trend')
    run_ucm('smooth_trend', use_exact_diffuse=True)


def test_random_trend(close_figures):
    run_ucm('random_trend')
    run_ucm('random_trend', use_exact_diffuse=True)


def test_cycle(close_figures):
    run_ucm('cycle_approx_diffuse')
    run_ucm('cycle', use_exact_diffuse=True)


def test_seasonal(close_figures):
    run_ucm('seasonal_approx_diffuse')
    run_ucm('seasonal', use_exact_diffuse=True)


def test_freq_seasonal(close_figures):
    run_ucm('freq_seasonal_approx_diffuse')
    run_ucm('freq_seasonal', use_exact_diffuse=True)


def test_reg(close_figures):
    run_ucm('reg_approx_diffuse')
    run_ucm('reg', use_exact_diffuse=True)


def test_rtrend_ar1(close_figures):
    run_ucm('rtrend_ar1')
    run_ucm('rtrend_ar1', use_exact_diffuse=True)


@pytest.mark.slow
def test_lltrend_cycle_seasonal_reg_ar1(close_figures):
    run_ucm('lltrend_cycle_seasonal_reg_ar1_approx_diffuse')
    run_ucm('lltrend_cycle_seasonal_reg_ar1', use_exact_diffuse=True)


@pytest.mark.parametrize("use_exact_diffuse", [True, False])
def test_mle_reg(use_exact_diffuse):
    endog = np.arange(100)*1.0
    exog = endog*2
    # Make the fit not-quite-perfect
    endog[::2] += 0.01
    endog[1::2] -= 0.01

    with warnings.catch_warnings(record=True):
        mod1 = UnobservedComponents(endog, irregular=True,
                                    exog=exog, mle_regression=False,
                                    use_exact_diffuse=use_exact_diffuse)
        res1 = mod1.fit(disp=-1)

        mod2 = UnobservedComponents(endog, irregular=True,
                                    exog=exog, mle_regression=True,
                                    use_exact_diffuse=use_exact_diffuse)
        res2 = mod2.fit(disp=-1)

    assert_allclose(res1.regression_coefficients.filtered[0, -1],
                    0.5,
                    atol=1e-5)
    assert_allclose(res2.params[1], 0.5, atol=1e-5)

    # When the regression component is part of the state vector with exact
    # diffuse initialization, we have two diffuse observations
    if use_exact_diffuse:
        print(res1.predicted_diffuse_state_cov)
        assert_equal(res1.nobs_diffuse, 2)
        assert_equal(res2.nobs_diffuse, 0)
    else:
        assert_equal(res1.loglikelihood_burn, 1)
        assert_equal(res2.loglikelihood_burn, 0)


def test_specifications():
    # Clear warnings
    structural.__warningregistry__ = {}

    endog = [1, 2]

    # Test that when nothing specified, a warning is issued and the model that
    # is fit is one with irregular=True and nothing else.
    warning = SpecificationWarning
    match = 'irregular component added'
    with pytest.warns(warning, match=match):
        mod = UnobservedComponents(endog)
        assert_equal(mod.trend_specification, 'irregular')

    # Test an invalid string trend specification
    with pytest.raises(ValueError):
        UnobservedComponents(endog, 'invalid spec')

    # Test that if a trend component is specified without a level component,
    # a warning is issued and a deterministic level component is added
    warning = SpecificationWarning
    match = 'Trend component specified without'
    with pytest.warns(warning, match=match):
        mod = UnobservedComponents(endog, trend=True, irregular=True)
        assert_equal(mod.trend_specification, 'deterministic trend')

    # Test that if a string specification is provided, a warning is issued if
    # the boolean attributes are also specified
    trend_attributes = ['irregular', 'trend', 'stochastic_level',
                        'stochastic_trend']
    for attribute in trend_attributes:
        kwargs = {attribute: True}

        warning = SpecificationWarning
        match = 'may be overridden when the trend'
        with pytest.warns(warning, match=match):
            UnobservedComponents(endog, 'deterministic trend', **kwargs)

    # Test that a seasonal with period less than two is invalid
    with pytest.raises(ValueError):
        UnobservedComponents(endog, seasonal=1)


def test_start_params():
    # Test that the behavior is correct for multiple exogenous and / or
    # autoregressive components

    # Parameters
    nobs = int(1e4)
    beta = np.r_[10, -2]
    phi = np.r_[0.5, 0.1]

    # Generate data
    np.random.seed(1234)
    exog = np.c_[np.ones(nobs), np.arange(nobs)*1.0]
    eps = np.random.normal(size=nobs)
    endog = np.zeros(nobs+2)
    for t in range(1, nobs):
        endog[t+1] = phi[0] * endog[t] + phi[1] * endog[t-1] + eps[t]
    endog = endog[2:]
    endog += np.dot(exog, beta)

    # Now just test that the starting parameters are approximately what they
    # ought to be (could make this arbitrarily precise by increasing nobs,
    # but that would slow down the test for no real gain)
    mod = UnobservedComponents(endog, exog=exog, autoregressive=2)
    assert_allclose(mod.start_params, [1., 0.5, 0.1, 10, -2], atol=1e-1)


def test_forecast():
    endog = np.arange(50) + 10
    exog = np.arange(50)

    mod = UnobservedComponents(endog, exog=exog, level='dconstant', seasonal=4)
    res = mod.smooth([1e-15, 0, 1])

    actual = res.forecast(10, exog=np.arange(50, 60)[:, np.newaxis])
    desired = np.arange(50, 60) + 10
    assert_allclose(actual, desired)


def test_misc_exog():
    # Tests for missing data
    nobs = 20
    k_endog = 1
    np.random.seed(1208)
    endog = np.random.normal(size=(nobs, k_endog))
    endog[:4, 0] = np.nan
    exog1 = np.random.normal(size=(nobs, 1))
    exog2 = np.random.normal(size=(nobs, 2))

    index = pd.date_range('1970-01-01', freq='QS', periods=nobs)
    endog_pd = pd.DataFrame(endog, index=index)
    exog1_pd = pd.Series(exog1.squeeze(), index=index)
    exog2_pd = pd.DataFrame(exog2, index=index)

    models = [
        UnobservedComponents(endog, 'llevel', exog=exog1),
        UnobservedComponents(endog, 'llevel', exog=exog2),
        UnobservedComponents(endog, 'llevel', exog=exog2),
        UnobservedComponents(endog_pd, 'llevel', exog=exog1_pd),
        UnobservedComponents(endog_pd, 'llevel', exog=exog2_pd),
        UnobservedComponents(endog_pd, 'llevel', exog=exog2_pd),
    ]

    for mod in models:
        # Smoke tests
        mod.start_params
        res = mod.fit(disp=False)
        res.summary()
        res.predict()
        res.predict(dynamic=True)
        res.get_prediction()

        oos_exog = np.random.normal(size=(1, mod.k_exog))
        res.forecast(steps=1, exog=oos_exog)
        res.get_forecast(steps=1, exog=oos_exog)

        # Smoke tests for invalid exog
        oos_exog = np.random.normal(size=(2, mod.k_exog))
        with pytest.raises(ValueError):
            res.forecast(steps=1, exog=oos_exog)

        oos_exog = np.random.normal(size=(1, mod.k_exog + 1))
        with pytest.raises(ValueError):
            res.forecast(steps=1, exog=oos_exog)

    # Test invalid model specifications
    with pytest.raises(ValueError):
        UnobservedComponents(endog, 'llevel', exog=np.zeros((10, 4)))


def test_predict_custom_index():
    np.random.seed(328423)
    endog = pd.DataFrame(np.random.normal(size=50))
    mod = structural.UnobservedComponents(endog, 'llevel')
    res = mod.smooth(mod.start_params)
    out = res.predict(start=1, end=1, index=['a'])
    assert_equal(out.index.equals(pd.Index(['a'])), True)


def test_matrices_somewhat_complicated_model():
    values = dta.copy()

    model = UnobservedComponents(values['unemp'],
                                 level='lltrend',
                                 freq_seasonal=[{'period': 4},
                                                {'period': 9, 'harmonics': 3}],
                                 cycle=True,
                                 cycle_period_bounds=[2, 30],
                                 damped_cycle=True,
                                 stochastic_freq_seasonal=[True, False],
                                 stochastic_cycle=True
                                 )
    # Selected parameters
    params = [1,  # irregular_var
              3, 4,  # lltrend parameters:  level_var, trend_var
              5,   # freq_seasonal parameters: freq_seasonal_var_0
              # cycle parameters: cycle_var, cycle_freq, cycle_damp
              6, 2*np.pi/30., .9
              ]
    model.update(params)

    # Check scalar properties
    assert_equal(model.k_states, 2 + 4 + 6 + 2)
    assert_equal(model.k_state_cov, 2 + 1 + 0 + 1)
    assert_equal(model.loglikelihood_burn, 2 + 4 + 6 + 2)
    assert_allclose(model.ssm.k_posdef, 2 + 4 + 0 + 2)
    assert_equal(model.k_params, len(params))

    # Check the statespace model matrices against hand-constructed answers
    # We group the terms by the component
    expected_design = np.r_[[1, 0],
                            [1, 0, 1, 0],
                            [1, 0, 1, 0, 1, 0],
                            [1, 0]].reshape(1, 14)
    assert_allclose(model.ssm.design[:, :, 0], expected_design)

    expected_transition = __direct_sum([
        np.array([[1, 1],
                  [0, 1]]),
        np.array([[0, 1, 0, 0],
                  [-1, 0, 0, 0],
                  [0, 0, -1,  0],
                  [0, 0,  0, -1]]),
        np.array([[np.cos(2*np.pi*1/9.), np.sin(2*np.pi*1/9.), 0, 0, 0, 0],
                  [-np.sin(2*np.pi*1/9.), np.cos(2*np.pi*1/9.), 0, 0, 0, 0],
                  [0, 0,  np.cos(2*np.pi*2/9.), np.sin(2*np.pi*2/9.), 0, 0],
                  [0, 0, -np.sin(2*np.pi*2/9.), np.cos(2*np.pi*2/9.), 0, 0],
                  [0, 0, 0, 0,  np.cos(2*np.pi/3.), np.sin(2*np.pi/3.)],
                  [0, 0, 0, 0, -np.sin(2*np.pi/3.), np.cos(2*np.pi/3.)]]),
        np.array([[.9*np.cos(2*np.pi/30.), .9*np.sin(2*np.pi/30.)],
                 [-.9*np.sin(2*np.pi/30.), .9*np.cos(2*np.pi/30.)]])
    ])
    assert_allclose(
        model.ssm.transition[:, :, 0], expected_transition, atol=1e-7)

    # Since the second seasonal term is not stochastic,
    # the dimensionality of the state disturbance is 14 - 6 = 8
    expected_selection = np.zeros((14, 14 - 6))
    expected_selection[0:2, 0:2] = np.eye(2)
    expected_selection[2:6, 2:6] = np.eye(4)
    expected_selection[-2:, -2:] = np.eye(2)
    assert_allclose(model.ssm.selection[:, :, 0], expected_selection)

    expected_state_cov = __direct_sum([
        np.diag(params[1:3]),
        np.eye(4)*params[3],
        np.eye(2)*params[4]
    ])
    assert_allclose(model.ssm.state_cov[:, :, 0], expected_state_cov)


def __direct_sum(square_matrices):
    """Compute the matrix direct sum of an iterable of square numpy 2-d arrays
    """
    new_shape = np.sum([m.shape for m in square_matrices], axis=0)
    new_array = np.zeros(new_shape)
    offset = 0
    for m in square_matrices:
        rows, cols = m.shape
        assert rows == cols
        new_array[offset:offset + rows, offset:offset + rows] = m
        offset += rows
    return new_array


def test_forecast_exog():
    # Test forecasting with various shapes of `exog`
    nobs = 100
    endog = np.ones(nobs) * 2.0
    exog = np.ones(nobs)

    mod = UnobservedComponents(endog, 'irregular', exog=exog)
    res = mod.smooth([1.0, 2.0])

    # 1-step-ahead, valid
    exog_fcast_scalar = 1.
    exog_fcast_1dim = np.ones(1)
    exog_fcast_2dim = np.ones((1, 1))

    assert_allclose(res.forecast(1, exog=exog_fcast_scalar), 2.)
    assert_allclose(res.forecast(1, exog=exog_fcast_1dim), 2.)
    assert_allclose(res.forecast(1, exog=exog_fcast_2dim), 2.)

    # h-steps-ahead, valid
    h = 10
    exog_fcast_1dim = np.ones(h)
    exog_fcast_2dim = np.ones((h, 1))

    assert_allclose(res.forecast(h, exog=exog_fcast_1dim), 2.)
    assert_allclose(res.forecast(h, exog=exog_fcast_2dim), 2.)

    # h-steps-ahead, invalid
    assert_raises(ValueError, res.forecast, h, exog=1.)
    assert_raises(ValueError, res.forecast, h, exog=[1, 2])
    assert_raises(ValueError, res.forecast, h, exog=np.ones((h, 2)))


def check_equivalent_models(mod, mod2):
    attrs = [
        'level', 'trend', 'seasonal_periods', 'seasonal',
        'freq_seasonal_periods', 'freq_seasonal_harmonics', 'freq_seasonal',
        'cycle', 'ar_order', 'autoregressive', 'irregular', 'stochastic_level',
        'stochastic_trend', 'stochastic_seasonal', 'stochastic_freq_seasonal',
        'stochastic_cycle', 'damped_cycle', 'mle_regression',
        'trend_specification', 'trend_mask', 'regression',
        'cycle_frequency_bound']

    ssm_attrs = [
        'nobs', 'k_endog', 'k_states', 'k_posdef', 'obs_intercept', 'design',
        'obs_cov', 'state_intercept', 'transition', 'selection', 'state_cov']

    for attr in attrs:
        assert_equal(getattr(mod2, attr), getattr(mod, attr))

    for attr in ssm_attrs:
        assert_equal(getattr(mod2.ssm, attr), getattr(mod.ssm, attr))

    assert_equal(mod2._get_init_kwds(), mod._get_init_kwds())


def test_recreate_model():
    nobs = 100
    endog = np.ones(nobs) * 2.0
    exog = np.ones(nobs)

    levels = [
        'irregular', 'ntrend', 'fixed intercept', 'deterministic constant',
        'dconstant', 'local level', 'llevel', 'random walk', 'rwalk',
        'fixed slope', 'deterministic trend', 'dtrend',
        'local linear deterministic trend', 'lldtrend',
        'random walk with drift', 'rwdrift', 'local linear trend',
        'lltrend', 'smooth trend', 'strend', 'random trend', 'rtrend']

    for level in levels:
        # Note: have to add in some stochastic component, otherwise we have
        # problems with entirely deterministic models

        # level + stochastic seasonal
        mod = UnobservedComponents(endog, level=level, seasonal=2,
                                   stochastic_seasonal=True, exog=exog)
        mod2 = UnobservedComponents(endog, exog=exog, **mod._get_init_kwds())
        check_equivalent_models(mod, mod2)

        # level + autoregressive
        mod = UnobservedComponents(endog, level=level, exog=exog,
                                   autoregressive=1)
        mod2 = UnobservedComponents(endog, exog=exog, **mod._get_init_kwds())
        check_equivalent_models(mod, mod2)

        # level + stochastic cycle
        mod = UnobservedComponents(endog, level=level, exog=exog,
                                   cycle=True, stochastic_cycle=True,
                                   damped_cycle=True)
        mod2 = UnobservedComponents(endog, exog=exog, **mod._get_init_kwds())
        check_equivalent_models(mod, mod2)


def test_append_results():
    endog = np.arange(100)
    exog = np.ones_like(endog)
    params = [1., 1., 0.1, 1.]

    mod1 = UnobservedComponents(endog, 'llevel', exog=exog)
    res1 = mod1.smooth(params)

    mod2 = UnobservedComponents(endog[:50], 'llevel', exog=exog[:50])
    res2 = mod2.smooth(params)
    res3 = res2.append(endog[50:], exog=exog[50:])

    assert_equal(res1.specification, res3.specification)

    assert_allclose(res3.cov_params_default, res2.cov_params_default)
    for attr in ['nobs', 'llf', 'llf_obs', 'loglikelihood_burn']:
        assert_equal(getattr(res3, attr), getattr(res1, attr))

    for attr in [
            'filtered_state', 'filtered_state_cov', 'predicted_state',
            'predicted_state_cov', 'forecasts', 'forecasts_error',
            'forecasts_error_cov', 'standardized_forecasts_error',
            'forecasts_error_diffuse_cov', 'predicted_diffuse_state_cov',
            'scaled_smoothed_estimator',
            'scaled_smoothed_estimator_cov', 'smoothing_error',
            'smoothed_state',
            'smoothed_state_cov', 'smoothed_state_autocov',
            'smoothed_measurement_disturbance',
            'smoothed_state_disturbance',
            'smoothed_measurement_disturbance_cov',
            'smoothed_state_disturbance_cov']:
        assert_equal(getattr(res3, attr), getattr(res1, attr))

    assert_allclose(res3.forecast(10, exog=np.ones(10)),
                    res1.forecast(10, exog=np.ones(10)))


def test_extend_results():
    endog = np.arange(100)
    exog = np.ones_like(endog)
    params = [1., 1., 0.1, 1.]

    mod1 = UnobservedComponents(endog, 'llevel', exog=exog)
    res1 = mod1.smooth(params)

    mod2 = UnobservedComponents(endog[:50], 'llevel', exog=exog[:50])
    res2 = mod2.smooth(params)

    res3 = res2.extend(endog[50:], exog=exog[50:])

    assert_allclose(res3.llf_obs, res1.llf_obs[50:])

    for attr in [
            'filtered_state', 'filtered_state_cov', 'predicted_state',
            'predicted_state_cov', 'forecasts', 'forecasts_error',
            'forecasts_error_cov', 'standardized_forecasts_error',
            'forecasts_error_diffuse_cov', 'predicted_diffuse_state_cov',
            'scaled_smoothed_estimator',
            'scaled_smoothed_estimator_cov', 'smoothing_error',
            'smoothed_state',
            'smoothed_state_cov', 'smoothed_state_autocov',
            'smoothed_measurement_disturbance',
            'smoothed_state_disturbance',
            'smoothed_measurement_disturbance_cov',
            'smoothed_state_disturbance_cov']:
        desired = getattr(res1, attr)
        if desired is not None:
            desired = desired[..., 50:]
        assert_equal(getattr(res3, attr), desired)

    assert_allclose(res3.forecast(10, exog=np.ones(10)),
                    res1.forecast(10, exog=np.ones(10)))


def test_apply_results():
    endog = np.arange(100)
    exog = np.ones_like(endog)
    params = [1., 1., 0.1, 1.]

    mod1 = UnobservedComponents(endog[:50], 'llevel', exog=exog[:50])
    res1 = mod1.smooth(params)

    mod2 = UnobservedComponents(endog[50:], 'llevel', exog=exog[50:])
    res2 = mod2.smooth(params)

    res3 = res2.apply(endog[:50], exog=exog[:50])

    assert_equal(res1.specification, res3.specification)

    assert_allclose(res3.cov_params_default, res2.cov_params_default)
    for attr in ['nobs', 'llf', 'llf_obs', 'loglikelihood_burn']:
        assert_equal(getattr(res3, attr), getattr(res1, attr))

    for attr in [
            'filtered_state', 'filtered_state_cov', 'predicted_state',
            'predicted_state_cov', 'forecasts', 'forecasts_error',
            'forecasts_error_cov', 'standardized_forecasts_error',
            'forecasts_error_diffuse_cov', 'predicted_diffuse_state_cov',
            'scaled_smoothed_estimator',
            'scaled_smoothed_estimator_cov', 'smoothing_error',
            'smoothed_state',
            'smoothed_state_cov', 'smoothed_state_autocov',
            'smoothed_measurement_disturbance',
            'smoothed_state_disturbance',
            'smoothed_measurement_disturbance_cov',
            'smoothed_state_disturbance_cov']:
        assert_equal(getattr(res3, attr), getattr(res1, attr))

    assert_allclose(res3.forecast(10, exog=np.ones(10)),
                    res1.forecast(10, exog=np.ones(10)))
