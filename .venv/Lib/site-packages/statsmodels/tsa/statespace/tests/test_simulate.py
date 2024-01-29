"""
Tests for simulation of time series

Author: Chad Fulton
License: Simplified-BSD
"""

from statsmodels.compat.pandas import MONTH_END

import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest
from scipy.signal import lfilter

from statsmodels.tools.sm_exceptions import (
    EstimationWarning,
    SpecificationWarning,
)
from statsmodels.tsa.statespace import (
    dynamic_factor,
    sarimax,
    structural,
    varmax,
)

from .test_impulse_responses import TVSS


def test_arma_lfilter():
    # Tests of an ARMA model simulation against scipy.signal.lfilter
    # Note: the first elements of the generated SARIMAX datasets are based on
    # the initial state, so we do not include them in the comparisons
    np.random.seed(10239)
    nobs = 100
    eps = np.random.normal(size=nobs)

    # AR(1)
    mod = sarimax.SARIMAX([0], order=(1, 0, 0))
    actual = mod.simulate([0.5, 1.], nobs + 1, state_shocks=np.r_[eps, 0],
                          initial_state=np.zeros(mod.k_states))
    desired = lfilter([1], [1, -0.5], eps)
    assert_allclose(actual[1:], desired)

    # MA(1)
    mod = sarimax.SARIMAX([0], order=(0, 0, 1))
    actual = mod.simulate([0.5, 1.], nobs + 1, state_shocks=np.r_[eps, 0],
                          initial_state=np.zeros(mod.k_states))
    desired = lfilter([1, 0.5], [1], eps)
    assert_allclose(actual[1:], desired)

    # ARMA(1, 1)
    mod = sarimax.SARIMAX([0], order=(1, 0, 1))
    actual = mod.simulate([0.5, 0.2, 1.], nobs + 1, state_shocks=np.r_[eps, 0],
                          initial_state=np.zeros(mod.k_states))
    desired = lfilter([1, 0.2], [1, -0.5], eps)
    assert_allclose(actual[1:], desired)


def test_arma_direct():
    # Tests of an ARMA model simulation against direct construction
    # This is useful for e.g. trend components
    # Note: the first elements of the generated SARIMAX datasets are based on
    # the initial state, so we do not include them in the comparisons
    np.random.seed(10239)
    nobs = 100
    eps = np.random.normal(size=nobs)
    exog = np.random.normal(size=nobs)

    # AR(1)
    mod = sarimax.SARIMAX([0], order=(1, 0, 0))
    actual = mod.simulate([0.5, 1.], nobs + 1, state_shocks=np.r_[eps, 0],
                          initial_state=np.zeros(mod.k_states))
    desired = np.zeros(nobs)
    for i in range(nobs):
        if i == 0:
            desired[i] = eps[i]
        else:
            desired[i] = 0.5 * desired[i - 1] + eps[i]
    assert_allclose(actual[1:], desired)

    # MA(1)
    mod = sarimax.SARIMAX([0], order=(0, 0, 1))
    actual = mod.simulate([0.5, 1.], nobs + 1, state_shocks=np.r_[eps, 0],
                          initial_state=np.zeros(mod.k_states))
    desired = np.zeros(nobs)
    for i in range(nobs):
        if i == 0:
            desired[i] = eps[i]
        else:
            desired[i] = 0.5 * eps[i - 1] + eps[i]
    assert_allclose(actual[1:], desired)

    # ARMA(1, 1)
    mod = sarimax.SARIMAX([0], order=(1, 0, 1))
    actual = mod.simulate([0.5, 0.2, 1.], nobs + 1, state_shocks=np.r_[eps, 0],
                          initial_state=np.zeros(mod.k_states))
    desired = np.zeros(nobs)
    for i in range(nobs):
        if i == 0:
            desired[i] = eps[i]
        else:
            desired[i] = 0.5 * desired[i - 1] + 0.2 * eps[i - 1] + eps[i]
    assert_allclose(actual[1:], desired)

    # ARMA(1, 1) + intercept
    mod = sarimax.SARIMAX([0], order=(1, 0, 1), trend='c')
    actual = mod.simulate([1.3, 0.5, 0.2, 1.], nobs + 1,
                          state_shocks=np.r_[eps, 0],
                          initial_state=np.zeros(mod.k_states))
    desired = np.zeros(nobs)
    for i in range(nobs):
        trend = 1.3
        if i == 0:
            desired[i] = trend + eps[i]
        else:
            desired[i] = (trend + 0.5 * desired[i - 1] +
                          0.2 * eps[i - 1] + eps[i])
    assert_allclose(actual[1:], desired)

    # ARMA(1, 1) + intercept + time trend
    # Note: to allow time-varying SARIMAX to simulate 101 observations, need to
    # give it 101 observations up front
    mod = sarimax.SARIMAX(np.zeros(nobs + 1), order=(1, 0, 1), trend='ct')
    actual = mod.simulate([1.3, 0.2, 0.5, 0.2, 1.], nobs + 1,
                          state_shocks=np.r_[eps, 0],
                          initial_state=np.zeros(mod.k_states))
    desired = np.zeros(nobs)
    for i in range(nobs):
        trend = 1.3 + 0.2 * (i + 1)
        if i == 0:
            desired[i] = trend + eps[i]
        else:
            desired[i] = (trend + 0.5 * desired[i - 1] +
                          0.2 * eps[i - 1] + eps[i])
    assert_allclose(actual[1:], desired)

    # ARMA(1, 1) + intercept + time trend + exog
    # Note: to allow time-varying SARIMAX to simulate 101 observations, need to
    # give it 101 observations up front
    # Note: the model is regression with SARIMAX errors, so the exog is
    # introduced into the observation equation rather than the ARMA part
    mod = sarimax.SARIMAX(np.zeros(nobs + 1), exog=np.r_[0, exog],
                          order=(1, 0, 1), trend='ct')
    actual = mod.simulate([1.3, 0.2, -0.5, 0.5, 0.2, 1.], nobs + 1,
                          state_shocks=np.r_[eps, 0],
                          initial_state=np.zeros(mod.k_states))
    desired = np.zeros(nobs)
    for i in range(nobs):
        trend = 1.3 + 0.2 * (i + 1)
        if i == 0:
            desired[i] = trend + eps[i]
        else:
            desired[i] = (trend + 0.5 * desired[i - 1] +
                          0.2 * eps[i - 1] + eps[i])
    desired = desired - 0.5 * exog
    assert_allclose(actual[1:], desired)


def test_structural():
    np.random.seed(38947)
    nobs = 100
    eps = np.random.normal(size=nobs)
    exog = np.random.normal(size=nobs)

    eps1 = np.zeros(nobs)
    eps2 = np.zeros(nobs)
    eps2[49] = 1
    eps3 = np.zeros(nobs)
    eps3[50:] = 1

    # AR(1)
    mod1 = structural.UnobservedComponents([0], autoregressive=1)
    mod2 = sarimax.SARIMAX([0], order=(1, 0, 0))
    actual = mod1.simulate([1, 0.5], nobs, state_shocks=eps,
                           initial_state=np.zeros(mod1.k_states))
    desired = mod2.simulate([0.5, 1], nobs, state_shocks=eps,
                            initial_state=np.zeros(mod2.k_states))
    assert_allclose(actual, desired)

    # ARX(1)
    mod1 = structural.UnobservedComponents(np.zeros(nobs), exog=exog,
                                           autoregressive=1)
    mod2 = sarimax.SARIMAX(np.zeros(nobs), exog=exog, order=(1, 0, 0))
    actual = mod1.simulate([1, 0.5, 0.2], nobs, state_shocks=eps,
                           initial_state=np.zeros(mod2.k_states))
    desired = mod2.simulate([0.2, 0.5, 1], nobs, state_shocks=eps,
                            initial_state=np.zeros(mod2.k_states))
    assert_allclose(actual, desired)

    # Irregular
    mod = structural.UnobservedComponents([0], 'irregular')
    actual = mod.simulate([1.], nobs, measurement_shocks=eps,
                          initial_state=np.zeros(mod.k_states))
    assert_allclose(actual, eps)

    # Fixed intercept
    # (in practice this is a deterministic constant, because an irregular
    #  component must be added)
    warning = SpecificationWarning
    match = 'irregular component added'
    with pytest.warns(warning, match=match):
        mod = structural.UnobservedComponents([0], 'fixed intercept')
    actual = mod.simulate([1.], nobs, measurement_shocks=eps,
                          initial_state=[10])
    assert_allclose(actual, 10 + eps)

    # Deterministic constant
    mod = structural.UnobservedComponents([0], 'deterministic constant')
    actual = mod.simulate([1.], nobs, measurement_shocks=eps,
                          initial_state=[10])
    assert_allclose(actual, 10 + eps)

    # Local level
    mod = structural.UnobservedComponents([0], 'local level')
    actual = mod.simulate([1., 1.], nobs, measurement_shocks=eps,
                          state_shocks=eps2,
                          initial_state=np.zeros(mod.k_states))
    assert_allclose(actual, eps + eps3)

    # Random walk
    mod = structural.UnobservedComponents([0], 'random walk')
    actual = mod.simulate([1.], nobs, measurement_shocks=eps,
                          state_shocks=eps2,
                          initial_state=np.zeros(mod.k_states))
    assert_allclose(actual, eps + eps3)

    # Fixed slope
    # (in practice this is a deterministic trend, because an irregular
    #  component must be added)
    warning = SpecificationWarning
    match = 'irregular component added'
    with pytest.warns(warning, match=match):
        mod = structural.UnobservedComponents([0], 'fixed slope')
    actual = mod.simulate([1., 1.], nobs, measurement_shocks=eps,
                          state_shocks=eps2, initial_state=[0, 1])
    assert_allclose(actual, eps + np.arange(100))

    # Deterministic trend
    mod = structural.UnobservedComponents([0], 'deterministic trend')
    actual = mod.simulate([1.], nobs, measurement_shocks=eps,
                          state_shocks=eps2, initial_state=[0, 1])
    assert_allclose(actual, eps + np.arange(100))

    # Local linear deterministic trend
    mod = structural.UnobservedComponents(
        [0], 'local linear deterministic trend')
    actual = mod.simulate([1., 1.], nobs, measurement_shocks=eps,
                          state_shocks=eps2, initial_state=[0, 1])
    desired = eps + np.r_[np.arange(50), 1 + np.arange(50, 100)]
    assert_allclose(actual, desired)

    # Random walk with drift
    mod = structural.UnobservedComponents([0], 'random walk with drift')
    actual = mod.simulate([1.], nobs, state_shocks=eps2,
                          initial_state=[0, 1])
    desired = np.r_[np.arange(50), 1 + np.arange(50, 100)]
    assert_allclose(actual, desired)

    # Local linear trend
    mod = structural.UnobservedComponents([0], 'local linear trend')
    actual = mod.simulate([1., 1., 1.], nobs, measurement_shocks=eps,
                          state_shocks=np.c_[eps2, eps1], initial_state=[0, 1])
    desired = eps + np.r_[np.arange(50), 1 + np.arange(50, 100)]
    assert_allclose(actual, desired)

    actual = mod.simulate([1., 1., 1.], nobs, measurement_shocks=eps,
                          state_shocks=np.c_[eps1, eps2], initial_state=[0, 1])
    desired = eps + np.r_[np.arange(50), np.arange(50, 150, 2)]
    assert_allclose(actual, desired)

    # Smooth trend
    mod = structural.UnobservedComponents([0], 'smooth trend')
    actual = mod.simulate([1., 1.], nobs, measurement_shocks=eps,
                          state_shocks=eps1, initial_state=[0, 1])
    desired = eps + np.r_[np.arange(100)]
    assert_allclose(actual, desired)

    actual = mod.simulate([1., 1.], nobs, measurement_shocks=eps,
                          state_shocks=eps2, initial_state=[0, 1])
    desired = eps + np.r_[np.arange(50), np.arange(50, 150, 2)]
    assert_allclose(actual, desired)

    # Random trend
    mod = structural.UnobservedComponents([0], 'random trend')
    actual = mod.simulate([1., 1.], nobs,
                          state_shocks=eps1, initial_state=[0, 1])
    desired = np.r_[np.arange(100)]
    assert_allclose(actual, desired)

    actual = mod.simulate([1., 1.], nobs,
                          state_shocks=eps2, initial_state=[0, 1])
    desired = np.r_[np.arange(50), np.arange(50, 150, 2)]
    assert_allclose(actual, desired)

    # Seasonal (deterministic)
    mod = structural.UnobservedComponents([0], 'irregular', seasonal=2,
                                          stochastic_seasonal=False)
    actual = mod.simulate([1.], nobs, measurement_shocks=eps,
                          initial_state=[10])
    desired = eps + np.tile([10, -10], 50)
    assert_allclose(actual, desired)

    # Seasonal (stochastic)
    mod = structural.UnobservedComponents([0], 'irregular', seasonal=2)
    actual = mod.simulate([1., 1.], nobs, measurement_shocks=eps,
                          state_shocks=eps2, initial_state=[10])
    desired = eps + np.r_[np.tile([10, -10], 25), np.tile([11, -11], 25)]
    assert_allclose(actual, desired)

    # Cycle (deterministic)
    mod = structural.UnobservedComponents([0], 'irregular', cycle=True)
    actual = mod.simulate([1., 1.2], nobs, measurement_shocks=eps,
                          initial_state=[1, 0])
    x1 = [np.cos(1.2), np.sin(1.2)]
    x2 = [-np.sin(1.2), np.cos(1.2)]
    T = np.array([x1, x2])
    desired = eps
    states = [1, 0]
    for i in range(nobs):
        desired[i] += states[0]
        states = np.dot(T, states)
    assert_allclose(actual, desired)

    # Cycle (stochastic)
    mod = structural.UnobservedComponents([0], 'irregular', cycle=True,
                                          stochastic_cycle=True)
    actual = mod.simulate([1., 1., 1.2], nobs, measurement_shocks=eps,
                          state_shocks=np.c_[eps2, eps2], initial_state=[1, 0])
    x1 = [np.cos(1.2), np.sin(1.2)]
    x2 = [-np.sin(1.2), np.cos(1.2)]
    T = np.array([x1, x2])
    desired = eps
    states = [1, 0]
    for i in range(nobs):
        desired[i] += states[0]
        states = np.dot(T, states) + eps2[i]
    assert_allclose(actual, desired)


def test_varmax():
    np.random.seed(371934)
    nobs = 100
    eps = np.random.normal(size=nobs)
    exog = np.random.normal(size=(nobs, 1))

    eps1 = np.zeros(nobs)
    eps2 = np.zeros(nobs)
    eps2[49] = 1
    eps3 = np.zeros(nobs)
    eps3[50:] = 1

    # VAR(2) - single series
    mod1 = varmax.VARMAX([[0]], order=(2, 0), trend='n')
    mod2 = sarimax.SARIMAX([0], order=(2, 0, 0))
    actual = mod1.simulate([0.5, 0.2, 1], nobs, state_shocks=eps,
                           initial_state=np.zeros(mod1.k_states))
    desired = mod2.simulate([0.5, 0.2, 1], nobs, state_shocks=eps,
                            initial_state=np.zeros(mod2.k_states))
    assert_allclose(actual, desired)

    # VMA(2) - single series
    mod1 = varmax.VARMAX([[0]], order=(0, 2), trend='n')
    mod2 = sarimax.SARIMAX([0], order=(0, 0, 2))
    actual = mod1.simulate([0.5, 0.2, 1], nobs, state_shocks=eps,
                           initial_state=np.zeros(mod1.k_states))
    desired = mod2.simulate([0.5, 0.2, 1], nobs, state_shocks=eps,
                            initial_state=np.zeros(mod2.k_states))
    assert_allclose(actual, desired)

    # VARMA(2, 2) - single series
    warning = EstimationWarning
    match = r'VARMA\(p,q\) models is not'
    with pytest.warns(warning, match=match):
        mod1 = varmax.VARMAX([[0]], order=(2, 2), trend='n')
    mod2 = sarimax.SARIMAX([0], order=(2, 0, 2))
    actual = mod1.simulate([0.5, 0.2, 0.1, -0.2, 1], nobs, state_shocks=eps,
                           initial_state=np.zeros(mod1.k_states))
    desired = mod2.simulate([0.5, 0.2, 0.1, -0.2, 1], nobs, state_shocks=eps,
                            initial_state=np.zeros(mod2.k_states))
    assert_allclose(actual, desired)

    # VARMA(2, 2) + trend - single series
    warning = EstimationWarning
    match = r'VARMA\(p,q\) models is not'
    with pytest.warns(warning, match=match):
        mod1 = varmax.VARMAX([[0]], order=(2, 2), trend='c')
    mod2 = sarimax.SARIMAX([0], order=(2, 0, 2), trend='c')
    actual = mod1.simulate([10, 0.5, 0.2, 0.1, -0.2, 1], nobs,
                           state_shocks=eps,
                           initial_state=np.zeros(mod1.k_states))
    desired = mod2.simulate([10, 0.5, 0.2, 0.1, -0.2, 1], nobs,
                            state_shocks=eps,
                            initial_state=np.zeros(mod2.k_states))
    assert_allclose(actual, desired)

    # VAR(1)
    transition = np.array([[0.5,  0.1],
                           [-0.1, 0.2]])

    mod = varmax.VARMAX([[0, 0]], order=(1, 0), trend='n')
    actual = mod.simulate(np.r_[transition.ravel(), 1., 0, 1.], nobs,
                          state_shocks=np.c_[eps1, eps1],
                          initial_state=np.zeros(mod.k_states))
    assert_allclose(actual, 0)

    actual = mod.simulate(np.r_[transition.ravel(), 1., 0, 1.], nobs,
                          state_shocks=np.c_[eps1, eps1], initial_state=[1, 1])
    desired = np.zeros((nobs, 2))
    state = np.r_[1, 1]
    for i in range(nobs):
        desired[i] = state
        state = np.dot(transition, state)
    assert_allclose(actual, desired)

    # VAR(1) + measurement error
    mod = varmax.VARMAX([[0, 0]], order=(1, 0), trend='n',
                        measurement_error=True)
    actual = mod.simulate(np.r_[transition.ravel(), 1., 0, 1., 1., 1.], nobs,
                          measurement_shocks=np.c_[eps, eps],
                          state_shocks=np.c_[eps1, eps1],
                          initial_state=np.zeros(mod.k_states))
    assert_allclose(actual, np.c_[eps, eps])

    # VARX(1)
    mod = varmax.VARMAX(np.zeros((nobs, 2)), order=(1, 0), trend='n',
                        exog=exog)
    actual = mod.simulate(np.r_[transition.ravel(), 5, -2, 1., 0, 1.], nobs,
                          state_shocks=np.c_[eps1, eps1], initial_state=[1, 1])
    desired = np.zeros((nobs, 2))
    state = np.r_[1, 1]
    for i in range(nobs):
        desired[i] = state
        if i < nobs - 1:
            state = exog[i + 1] * [5, -2] + np.dot(transition, state)
    assert_allclose(actual, desired)

    # VMA(1)
    # TODO: This is just a smoke test
    mod = varmax.VARMAX(
        np.random.normal(size=(nobs, 2)), order=(0, 1), trend='n')
    mod.simulate(mod.start_params, nobs)

    # VARMA(2, 2) + trend + exog
    # TODO: This is just a smoke test
    warning = EstimationWarning
    match = r"VARMA\(p,q\) models is not"
    with pytest.warns(warning, match=match):
        mod = varmax.VARMAX(
            np.random.normal(size=(nobs, 2)), order=(2, 2), trend='c',
            exog=exog)
    mod.simulate(mod.start_params, nobs)


def test_dynamic_factor():
    np.random.seed(93739)
    nobs = 100
    eps = np.random.normal(size=nobs)
    exog = np.random.normal(size=(nobs, 1))

    eps1 = np.zeros(nobs)
    eps2 = np.zeros(nobs)
    eps2[49] = 1
    eps3 = np.zeros(nobs)
    eps3[50:] = 1

    # DFM: 2 series, AR(2) factor
    mod1 = dynamic_factor.DynamicFactor([[0, 0]], k_factors=1, factor_order=2)
    mod2 = sarimax.SARIMAX([0], order=(2, 0, 0))
    actual = mod1.simulate([-0.9, 0.8, 1., 1., 0.5, 0.2], nobs,
                           measurement_shocks=np.c_[eps1, eps1],
                           state_shocks=eps,
                           initial_state=np.zeros(mod1.k_states))
    desired = mod2.simulate([0.5, 0.2, 1], nobs, state_shocks=eps,
                            initial_state=np.zeros(mod2.k_states))
    assert_allclose(actual[:, 0], -0.9 * desired)
    assert_allclose(actual[:, 1], 0.8 * desired)

    # DFM: 2 series, AR(2) factor, exog
    mod1 = dynamic_factor.DynamicFactor(np.zeros((nobs, 2)), k_factors=1,
                                        factor_order=2, exog=exog)
    mod2 = sarimax.SARIMAX([0], order=(2, 0, 0))
    actual = mod1.simulate([-0.9, 0.8, 5, -2, 1., 1., 0.5, 0.2], nobs,
                           measurement_shocks=np.c_[eps1, eps1],
                           state_shocks=eps,
                           initial_state=np.zeros(mod1.k_states))
    desired = mod2.simulate([0.5, 0.2, 1], nobs, state_shocks=eps,
                            initial_state=np.zeros(mod2.k_states))
    assert_allclose(actual[:, 0], -0.9 * desired + 5 * exog[:, 0])
    assert_allclose(actual[:, 1], 0.8 * desired - 2 * exog[:, 0])

    # DFM, 3 series, VAR(2) factor, exog, error VAR
    # TODO: This is just a smoke test
    mod = dynamic_factor.DynamicFactor(np.random.normal(size=(nobs, 3)),
                                       k_factors=2, factor_order=2, exog=exog,
                                       error_order=2, error_var=True)
    mod.simulate(mod.start_params, nobs)


def test_known_initialization():
    # Need to test that "known" initialization is taken into account in
    # time series simulation
    np.random.seed(38947)
    nobs = 100
    eps = np.random.normal(size=nobs)

    eps1 = np.zeros(nobs)
    eps2 = np.zeros(nobs)
    eps2[49] = 1
    eps3 = np.zeros(nobs)
    eps3[50:] = 1

    # SARIMAX
    # (test that when state shocks are shut down, the initial state
    # geometrically declines according to the AR parameter)
    mod = sarimax.SARIMAX([0], order=(1, 0, 0))
    mod.ssm.initialize_known([100], [[0]])
    actual = mod.simulate([0.5, 1.], nobs, state_shocks=eps1)
    assert_allclose(actual, 100 * 0.5**np.arange(nobs))

    # Unobserved components
    # (test that the initial level shifts the entire path)
    mod = structural.UnobservedComponents([0], 'local level')
    mod.ssm.initialize_known([100], [[0]])
    actual = mod.simulate([1., 1.], nobs, measurement_shocks=eps,
                          state_shocks=eps2)
    assert_allclose(actual, 100 + eps + eps3)

    # VARMAX
    # (here just test that with an independent VAR we have each initial state
    # geometrically declining at the appropriate rate)
    transition = np.diag([0.5, 0.2])
    mod = varmax.VARMAX([[0, 0]], order=(1, 0), trend='n')
    mod.initialize_known([100, 50], np.diag([0, 0]))
    actual = mod.simulate(np.r_[transition.ravel(), 1., 0, 1.], nobs,
                          measurement_shocks=np.c_[eps1, eps1],
                          state_shocks=np.c_[eps1, eps1])

    assert_allclose(actual, np.c_[100 * 0.5**np.arange(nobs),
                                  50 * 0.2**np.arange(nobs)])

    # Dynamic factor
    # (test that the initial state declines geometrically and then loads
    # correctly onto the series)
    mod = dynamic_factor.DynamicFactor([[0, 0]], k_factors=1, factor_order=1)
    mod.initialize_known([100], [[0]])

    actual = mod.simulate([0.8, 0.2, 1.0, 1.0, 0.5], nobs,
                          measurement_shocks=np.c_[eps1, eps1],
                          state_shocks=eps1)
    tmp = 100 * 0.5**np.arange(nobs)
    assert_allclose(actual, np.c_[0.8 * tmp, 0.2 * tmp])


def test_sequential_simulate():
    # Test that we can perform simulation, change the system matrices, and then
    # perform simulation again (i.e. check that everything updates correctly
    # in the simulation smoother).
    n_simulations = 100
    mod = sarimax.SARIMAX([1], order=(0, 0, 0), trend='c')

    actual = mod.simulate([1, 0], n_simulations)
    assert_allclose(actual, np.ones(n_simulations))

    actual = mod.simulate([10, 0], n_simulations)
    assert_allclose(actual, np.ones(n_simulations) * 10)


def test_sarimax_end_time_invariant_noshocks():
    # Test simulating values from the end of a time-invariant SARIMAX model
    # In this test, we suppress randomness by setting the shocks to zeros
    endog = np.arange(1, 11)
    mod = sarimax.SARIMAX(endog)
    res = mod.filter([0.5, 1.])

    nsimulations = 10
    measurement_shocks = np.zeros((nsimulations, mod.k_endog))
    state_shocks = np.zeros((nsimulations, mod.k_states))
    initial_state = res.predicted_state[..., -1]
    assert_allclose(initial_state, 5)

    actual = res.simulate(nsimulations, anchor='end',
                          measurement_shocks=measurement_shocks,
                          state_shocks=state_shocks,
                          initial_state=initial_state)

    # Compute the desired simulated values directly
    desired = 10 * 0.5**np.arange(1, nsimulations + 1)
    assert_allclose(actual, desired)

    # Test using the model versus the results class
    mod_actual = mod.simulate(
        res.params, nsimulations, anchor='end',
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state)

    assert_allclose(mod_actual, desired)

    # Alternatively, since we've shut down the shocks, we can compare against
    # the forecast values
    assert_allclose(actual, res.forecast(nsimulations))


def test_sarimax_simple_differencing_end_time_invariant_noshocks():
    # Test simulating values from the end of a time-invariant SARIMAX model
    # in which simple differencing is used.
    # In this test, we suppress randomness by setting the shocks to zeros
    endog = np.cumsum(np.arange(0, 11))
    mod = sarimax.SARIMAX(endog, order=(1, 1, 0), simple_differencing=True)
    res = mod.filter([0.5, 1.])

    nsimulations = 10
    measurement_shocks = np.zeros((nsimulations, mod.k_endog))
    state_shocks = np.zeros((nsimulations, mod.k_states))
    initial_state = res.predicted_state[..., -1]
    assert_allclose(initial_state, 5)

    actual = res.simulate(nsimulations, anchor='end',
                          measurement_shocks=measurement_shocks,
                          state_shocks=state_shocks,
                          initial_state=initial_state)

    # Compute the desired simulated values directly
    desired = 10 * 0.5**np.arange(1, nsimulations + 1)
    assert_allclose(actual, desired)

    # Test using the model versus the results class
    mod_actual = mod.simulate(
        res.params, nsimulations, anchor='end',
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state)

    assert_allclose(mod_actual, desired)

    # Alternatively, since we've shut down the shocks, we can compare against
    # the forecast values
    assert_allclose(actual, res.forecast(nsimulations))


def test_sarimax_time_invariant_shocks(reset_randomstate):
    # Test simulating values from the end of a time-invariant SARIMAX model,
    # with nonzero shocks
    endog = np.arange(1, 11)
    mod = sarimax.SARIMAX(endog)
    res = mod.filter([0.5, 1.])

    nsimulations = 10
    measurement_shocks = np.random.normal(size=nsimulations)
    state_shocks = np.random.normal(size=nsimulations)
    initial_state = res.predicted_state[:1, -1]

    actual = res.simulate(nsimulations, anchor='end',
                          measurement_shocks=measurement_shocks,
                          state_shocks=state_shocks,
                          initial_state=initial_state)

    desired = (
        lfilter([1], [1, -0.5], np.r_[initial_state, state_shocks])[:-1] +
        measurement_shocks)
    assert_allclose(actual, desired)

    # Test using the model versus the results class
    mod_actual = mod.simulate(
        res.params, nsimulations, anchor='end',
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state)

    assert_allclose(mod_actual, desired)


def test_sarimax_simple_differencing_end_time_invariant_shocks():
    # Test simulating values from the end of a time-invariant SARIMAX model
    # in which simple differencing is used.
    # In this test, we suppress randomness by setting the shocks to zeros
    endog = np.cumsum(np.arange(0, 11))
    mod = sarimax.SARIMAX(endog, order=(1, 1, 0), simple_differencing=True)
    res = mod.filter([0.5, 1.])

    nsimulations = 10
    measurement_shocks = np.random.normal(size=nsimulations)
    state_shocks = np.random.normal(size=nsimulations)
    initial_state = res.predicted_state[:1, -1]

    actual = res.simulate(nsimulations, anchor='end',
                          measurement_shocks=measurement_shocks,
                          state_shocks=state_shocks,
                          initial_state=initial_state)

    desired = (
        lfilter([1], [1, -0.5], np.r_[initial_state, state_shocks])[:-1] +
        measurement_shocks)
    assert_allclose(actual, desired)

    # Test using the model versus the results class
    mod_actual = mod.simulate(
        res.params, nsimulations, anchor='end',
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state)

    assert_allclose(mod_actual, desired)


def test_sarimax_time_varying_trend_noshocks():
    # Test simulating values from the end of a time-varying SARIMAX model
    # In this test, we suppress randomness by setting the shocks to zeros
    endog = np.arange(1, 11)
    mod = sarimax.SARIMAX(endog, trend='t')
    res = mod.filter([1., 0.2, 1.])

    nsimulations = 10
    measurement_shocks = np.zeros((nsimulations, mod.k_endog))
    state_shocks = np.zeros((nsimulations, mod.k_states))
    initial_state = res.predicted_state[..., -1]
    assert_allclose(initial_state, 12)

    actual = res.simulate(nsimulations, anchor='end',
                          measurement_shocks=measurement_shocks,
                          state_shocks=state_shocks,
                          initial_state=initial_state)

    # Compute the desired simulated values directly
    desired = lfilter([1], [1, -0.2], np.r_[12, np.arange(11, 20)])
    assert_allclose(actual, desired)

    # Test using the model versus the results class
    mod_actual = mod.simulate(
        res.params, nsimulations, anchor='end',
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state)

    assert_allclose(mod_actual, desired)

    # Alternatively, since we've shut down the shocks, we can compare against
    # the forecast values
    assert_allclose(actual, res.forecast(nsimulations))


def test_sarimax_simple_differencing_time_varying_trend_noshocks():
    # Test simulating values from the end of a time-varying SARIMAX model
    # in which simple differencing is used.
    # In this test, we suppress randomness by setting the shocks to zeros
    endog = np.cumsum(np.arange(0, 11))
    mod = sarimax.SARIMAX(endog, order=(1, 1, 0), trend='t',
                          simple_differencing=True)
    res = mod.filter([1., 0.2, 1.])

    nsimulations = 10
    measurement_shocks = np.zeros((nsimulations, mod.k_endog))
    state_shocks = np.zeros((nsimulations, mod.k_states))
    initial_state = res.predicted_state[..., -1]
    assert_allclose(initial_state, 12)

    actual = res.simulate(nsimulations, anchor='end',
                          measurement_shocks=measurement_shocks,
                          state_shocks=state_shocks,
                          initial_state=initial_state)

    # Compute the desired simulated values directly
    desired = lfilter([1], [1, -0.2], np.r_[12, np.arange(11, 20)])
    assert_allclose(actual, desired)

    # Test using the model versus the results class
    mod_actual = mod.simulate(
        res.params, nsimulations, anchor='end',
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state)

    assert_allclose(mod_actual, desired)

    # Alternatively, since we've shut down the shocks, we can compare against
    # the forecast values
    assert_allclose(actual, res.forecast(nsimulations))


def test_sarimax_time_varying_trend_shocks(reset_randomstate):
    # Test simulating values from the end of a time-varying SARIMAX model,
    # with nonzero shocks
    endog = np.arange(1, 11)
    mod = sarimax.SARIMAX(endog, trend='t')
    res = mod.filter([1., 0.2, 1.])

    nsimulations = 10
    measurement_shocks = np.random.normal(size=nsimulations)
    state_shocks = np.random.normal(size=nsimulations)
    initial_state = res.predicted_state[:1, -1]

    actual = res.simulate(nsimulations, anchor='end',
                          measurement_shocks=measurement_shocks,
                          state_shocks=state_shocks,
                          initial_state=initial_state)

    x = np.r_[initial_state, state_shocks + np.arange(11, 21)]
    desired = lfilter([1], [1, -0.2], x)[:-1] + measurement_shocks
    assert_allclose(actual, desired)

    # Test using the model versus the results class
    mod_actual = mod.simulate(
        res.params, nsimulations, anchor='end',
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state)

    assert_allclose(mod_actual, desired)


def test_sarimax_simple_differencing_time_varying_trend_shocks(
        reset_randomstate):
    # Test simulating values from the end of a time-varying SARIMAX model
    # in which simple differencing is used.
    # with nonzero shocks
    endog = np.cumsum(np.arange(0, 11))
    mod = sarimax.SARIMAX(endog, order=(1, 1, 0), trend='t',
                          simple_differencing=True)
    res = mod.filter([1., 0.2, 1.])

    nsimulations = 10
    measurement_shocks = np.random.normal(size=nsimulations)
    state_shocks = np.random.normal(size=nsimulations)
    initial_state = res.predicted_state[:1, -1]
    assert_allclose(initial_state, 12)

    actual = res.simulate(nsimulations, anchor='end',
                          measurement_shocks=measurement_shocks,
                          state_shocks=state_shocks,
                          initial_state=initial_state)

    x = np.r_[initial_state, state_shocks + np.arange(11, 21)]
    desired = lfilter([1], [1, -0.2], x)[:-1] + measurement_shocks
    assert_allclose(actual, desired)

    # Test using the model versus the results class
    mod_actual = mod.simulate(
        res.params, nsimulations, anchor='end',
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state)

    assert_allclose(mod_actual, desired)


def test_sarimax_time_varying_exog_noshocks():
    # Test simulating values from the end of a time-varying SARIMAX model
    # In this test, we suppress randomness by setting the shocks to zeros
    # Note that `exog` here has basically the same effect as measurement shocks
    endog = np.arange(1, 11)
    exog = np.arange(1, 21)**2
    mod = sarimax.SARIMAX(endog, exog=exog[:10])
    res = mod.filter([1., 0.2, 1.])

    nsimulations = 10
    measurement_shocks = np.zeros((nsimulations, mod.k_endog))
    state_shocks = np.zeros((nsimulations, mod.k_states))
    initial_state = res.predicted_state[..., -1]

    actual = res.simulate(nsimulations, exog=exog[10:], anchor='end',
                          measurement_shocks=measurement_shocks,
                          state_shocks=state_shocks,
                          initial_state=initial_state)

    # Compute the desired simulated values directly
    desired = (lfilter([1], [1, -0.2], np.r_[initial_state, [0] * 9]) +
               exog[10:])
    assert_allclose(actual, desired)

    # Test using the model versus the results class
    mod_actual = mod.simulate(
        res.params, nsimulations, exog=exog[10:], anchor='end',
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state)

    assert_allclose(mod_actual, desired)

    # Alternatively, since we've shut down the shocks, we can compare against
    # the forecast values
    assert_allclose(actual, res.forecast(nsimulations, exog=exog[10:]))


def test_sarimax_simple_differencing_time_varying_exog_noshocks():
    # Test simulating values from the end of a time-varying SARIMAX model
    # with simple differencing
    # In this test, we suppress randomness by setting the shocks to zeros
    endog = np.cumsum(np.arange(0, 11))
    exog = np.cumsum(np.arange(0, 21)**2)
    mod = sarimax.SARIMAX(endog, order=(1, 1, 0), exog=exog[:11],
                          simple_differencing=True)
    res = mod.filter([1., 0.2, 1.])

    nsimulations = 10
    measurement_shocks = np.zeros((nsimulations, mod.k_endog))
    state_shocks = np.zeros((nsimulations, mod.k_states))
    initial_state = res.predicted_state[..., -1]

    actual = res.simulate(nsimulations, exog=exog[11:], anchor='end',
                          measurement_shocks=measurement_shocks,
                          state_shocks=state_shocks,
                          initial_state=initial_state)

    # Compute the desired simulated values directly
    desired = (lfilter([1], [1, -0.2], np.r_[initial_state, [0] * 9]) +
               np.diff(exog)[10:])
    assert_allclose(actual, desired)

    # Test using the model versus the results class
    mod_actual = mod.simulate(
        res.params, nsimulations, exog=exog[11:], anchor='end',
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state)

    assert_allclose(mod_actual, desired)

    # Alternatively, since we've shut down the shocks, we can compare against
    # the forecast values
    assert_allclose(actual, res.forecast(nsimulations, exog=exog[11:]))


def test_sarimax_time_varying_exog_shocks(reset_randomstate):
    # Test simulating values from the end of a time-varying SARIMAX model,
    # with nonzero shocks
    endog = np.arange(1, 11)
    exog = np.arange(1, 21)**2
    mod = sarimax.SARIMAX(endog, exog=exog[:10])
    res = mod.filter([1., 0.2, 1.])

    nsimulations = 10
    measurement_shocks = np.random.normal(size=nsimulations)
    state_shocks = np.random.normal(size=nsimulations)
    initial_state = res.predicted_state[:1, -1]

    actual = res.simulate(nsimulations, exog=exog[10:], anchor='end',
                          measurement_shocks=measurement_shocks,
                          state_shocks=state_shocks,
                          initial_state=initial_state)

    x = np.r_[initial_state, state_shocks[:-1]]
    desired = lfilter([1], [1, -0.2], x) + exog[10:] + measurement_shocks
    assert_allclose(actual, desired)

    # Test using the model versus the results class
    mod_actual = mod.simulate(
        res.params, nsimulations, exog=exog[10:], anchor='end',
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state)

    assert_allclose(mod_actual, desired)


def test_sarimax_simple_differencing_time_varying_exog_shocks(
        reset_randomstate):
    # Test simulating values from the end of a time-varying SARIMAX model
    # Note that `exog` here has basically the same effect as measurement shocks
    endog = np.cumsum(np.arange(0, 11))
    exog = np.cumsum(np.arange(0, 21)**2)
    mod = sarimax.SARIMAX(endog, order=(1, 1, 0), exog=exog[:11],
                          simple_differencing=True)
    res = mod.filter([1., 0.2, 1.])

    nsimulations = 10
    measurement_shocks = np.random.normal(size=nsimulations)
    state_shocks = np.random.normal(size=nsimulations)
    initial_state = res.predicted_state[:1, -1]

    actual = res.simulate(nsimulations, exog=exog[11:], anchor='end',
                          measurement_shocks=measurement_shocks,
                          state_shocks=state_shocks,
                          initial_state=initial_state)

    # Compute the desired simulated values directly
    x = np.r_[initial_state, state_shocks[:-1]]
    desired = (lfilter([1], [1, -0.2], x) + np.diff(exog)[10:] +
               measurement_shocks)
    assert_allclose(actual, desired)

    # Test using the model versus the results class
    mod_actual = mod.simulate(
        res.params, nsimulations, exog=exog[11:], anchor='end',
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state)

    assert_allclose(mod_actual, desired)


def test_unobserved_components_end_time_invariant_noshocks():
    # Test simulating values from the end of a time-invariant
    # UnobservedComponents model
    # In this test, we suppress randomness by setting the shocks to zeros
    endog = np.arange(1, 11)
    mod = structural.UnobservedComponents(endog, 'llevel')
    res = mod.filter([1., 1.])

    nsimulations = 10
    measurement_shocks = np.zeros((nsimulations, mod.k_endog))
    state_shocks = np.zeros((nsimulations, mod.k_states))
    initial_state = res.predicted_state[..., -1]

    actual = res.simulate(nsimulations, anchor='end',
                          measurement_shocks=measurement_shocks,
                          state_shocks=state_shocks,
                          initial_state=initial_state)

    # The mean of the simulated local level values is just the last value
    desired = initial_state[0]
    assert_allclose(actual, desired)

    # Test using the model versus the results class
    mod_actual = mod.simulate(
        res.params, nsimulations, anchor='end',
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state)

    assert_allclose(mod_actual, desired)

    # Alternatively, since we've shut down the shocks, we can compare against
    # the forecast values
    assert_allclose(actual, res.forecast(nsimulations))


def test_unobserved_components_end_time_invariant_shocks(reset_randomstate):
    # Test simulating values from the end of a time-invariant
    # UnobservedComponents model, with nonzero shocks
    endog = np.arange(1, 11)
    mod = structural.UnobservedComponents(endog, 'llevel')
    res = mod.filter([1., 1.])

    nsimulations = 10
    measurement_shocks = np.random.normal(size=nsimulations)
    state_shocks = np.random.normal(size=nsimulations)
    initial_state = res.predicted_state[:1, -1]

    actual = res.simulate(nsimulations, anchor='end',
                          measurement_shocks=measurement_shocks,
                          state_shocks=state_shocks,
                          initial_state=initial_state)

    desired = (initial_state + np.cumsum(np.r_[0, state_shocks[:-1]]) +
               measurement_shocks)
    assert_allclose(actual, desired)

    # Test using the model versus the results class
    mod_actual = mod.simulate(
        res.params, nsimulations, anchor='end',
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state)

    assert_allclose(mod_actual, desired)


def test_unobserved_components_end_time_varying_exog_noshocks():
    # Test simulating values from the end of a time-varying
    # UnobservedComponents model with exog
    # In this test, we suppress randomness by setting the shocks to zeros
    endog = np.arange(1, 11)
    exog = np.arange(1, 21)**2
    mod = structural.UnobservedComponents(endog, 'llevel', exog=exog[:10])
    res = mod.filter([1., 1., 1.])

    nsimulations = 10
    measurement_shocks = np.zeros((nsimulations, mod.k_endog))
    state_shocks = np.zeros((nsimulations, mod.k_states))
    initial_state = res.predicted_state[..., -1]

    actual = res.simulate(nsimulations, exog=exog[10:], anchor='end',
                          measurement_shocks=measurement_shocks,
                          state_shocks=state_shocks,
                          initial_state=initial_state)

    # The mean of the simulated local level values is just the last value
    desired = initial_state[0] + exog[10:]
    assert_allclose(actual, desired)

    # Test using the model versus the results class
    mod_actual = mod.simulate(
        res.params, nsimulations, exog=exog[10:], anchor='end',
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state)

    assert_allclose(mod_actual, desired)

    # Alternatively, since we've shut down the shocks, we can compare against
    # the forecast values
    assert_allclose(actual, res.forecast(nsimulations, exog=exog[10:]))


def test_unobserved_components_end_time_varying_exog_shocks(reset_randomstate):
    # Test simulating values from the end of a time-varying
    # UnobservedComponents model with exog
    endog = np.arange(1, 11)
    exog = np.arange(1, 21)**2
    mod = structural.UnobservedComponents(endog, 'llevel', exog=exog[:10])
    res = mod.filter([1., 1., 1.])

    nsimulations = 10
    measurement_shocks = np.random.normal(size=nsimulations)
    state_shocks = np.random.normal(size=nsimulations)
    initial_state = res.predicted_state[:1, -1]

    actual = res.simulate(nsimulations, exog=exog[10:], anchor='end',
                          measurement_shocks=measurement_shocks,
                          state_shocks=state_shocks,
                          initial_state=initial_state)

    desired = (initial_state + np.cumsum(np.r_[0, state_shocks[:-1]]) +
               measurement_shocks + exog[10:])
    assert_allclose(actual, desired)

    # Test using the model versus the results class
    mod_actual = mod.simulate(
        res.params, nsimulations, exog=exog[10:], anchor='end',
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state)

    assert_allclose(mod_actual, desired)


def test_varmax_end_time_invariant_noshocks():
    # Test simulating values from the end of a time-invariant VARMAX model
    # In this test, we suppress randomness by setting the shocks to zeros
    endog = np.arange(1, 21).reshape(10, 2)
    mod = varmax.VARMAX(endog, trend='n')
    res = mod.filter([1., 1., 1., 1., 1., 0.5, 1.])

    nsimulations = 10
    measurement_shocks = np.zeros((nsimulations, mod.k_endog))
    state_shocks = np.zeros((nsimulations, mod.k_states))
    initial_state = res.predicted_state[:, -1]

    actual = res.simulate(nsimulations, anchor='end',
                          measurement_shocks=measurement_shocks,
                          state_shocks=state_shocks,
                          initial_state=initial_state)

    desired = (initial_state[:, None] * 2 ** np.arange(10)).T
    assert_allclose(actual, desired)

    # Test using the model versus the results class
    mod_actual = mod.simulate(
        res.params, nsimulations, anchor='end',
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state)

    assert_allclose(mod_actual, desired)

    # Alternatively, since we've shut down the shocks, we can compare against
    # the forecast values
    assert_allclose(actual, res.forecast(nsimulations))


def test_varmax_end_time_invariant_shocks(reset_randomstate):
    # Test simulating values from the end of a time-invariant VARMAX model,
    # with nonzero shocks
    endog = np.arange(1, 21).reshape(10, 2)
    mod = varmax.VARMAX(endog, trend='n')
    res = mod.filter([1., 1., 1., 1., 1., 0.5, 1.])

    nsimulations = 10
    measurement_shocks = np.random.normal(size=(nsimulations, mod.k_endog))
    state_shocks = np.random.normal(size=(nsimulations, mod.k_states))
    initial_state = res.predicted_state[:, -1]

    actual = res.simulate(nsimulations, anchor='end',
                          measurement_shocks=measurement_shocks,
                          state_shocks=state_shocks,
                          initial_state=initial_state)

    desired = np.zeros((nsimulations, mod.k_endog))
    desired[0] = initial_state
    for i in range(1, nsimulations):
        desired[i] = desired[i - 1].sum() + state_shocks[i - 1]
    desired = desired + measurement_shocks
    assert_allclose(actual, desired)

    # Test using the model versus the results class
    mod_actual = mod.simulate(
        res.params, nsimulations, anchor='end',
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state)

    assert_allclose(mod_actual, desired)


def test_varmax_end_time_varying_trend_noshocks():
    # Test simulating values from the end of a time-varying VARMAX model
    # with a trend
    # In this test, we suppress randomness by setting the shocks to zeros
    endog = np.arange(1, 21).reshape(10, 2)
    mod = varmax.VARMAX(endog, trend='ct')
    res = mod.filter([1., 1., 1., 1., 1, 1, 1., 1., 1., 0.5, 1.])

    nsimulations = 10
    measurement_shocks = np.zeros((nsimulations, mod.k_endog))
    state_shocks = np.zeros((nsimulations, mod.k_states))

    # Need to set the final predicted state given the new trend
    with res._set_final_predicted_state(exog=None, out_of_sample=10):
        initial_state = res.predicted_state[:, -1].copy()

    # Simulation
    actual = res.simulate(nsimulations, anchor='end',
                          measurement_shocks=measurement_shocks,
                          state_shocks=state_shocks,
                          initial_state=initial_state)

    desired = np.zeros((nsimulations, mod.k_endog))
    desired[0] = initial_state
    tmp_trend = 1 + np.arange(11, 21)
    for i in range(1, nsimulations):
        desired[i] = desired[i - 1].sum() + tmp_trend[i] + state_shocks[i - 1]
    desired = desired + measurement_shocks
    assert_allclose(actual, desired)

    # Test using the model versus the results class
    mod_actual = mod.simulate(
        res.params, nsimulations, anchor='end',
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state)

    assert_allclose(mod_actual, desired)

    # Alternatively, since we've shut down the shocks, we can compare against
    # the forecast values
    assert_allclose(actual, res.forecast(nsimulations))


def test_varmax_end_time_varying_trend_shocks(reset_randomstate):
    # Test simulating values from the end of a time-varying VARMAX model
    # with a trend
    endog = np.arange(1, 21).reshape(10, 2)
    mod = varmax.VARMAX(endog, trend='ct')
    res = mod.filter([1., 1., 1., 1., 1, 1, 1., 1., 1., 0.5, 1.])

    nsimulations = 10
    measurement_shocks = np.random.normal(size=(nsimulations, mod.k_endog))
    state_shocks = np.random.normal(size=(nsimulations, mod.k_states))

    # Need to set the final predicted state given the new trend
    with res._set_final_predicted_state(exog=None, out_of_sample=10):
        initial_state = res.predicted_state[:, -1].copy()

    # Simulation
    actual = res.simulate(nsimulations, anchor='end',
                          measurement_shocks=measurement_shocks,
                          state_shocks=state_shocks,
                          initial_state=initial_state)

    desired = np.zeros((nsimulations, mod.k_endog))
    desired[0] = initial_state
    tmp_trend = 1 + np.arange(11, 21)
    for i in range(1, nsimulations):
        desired[i] = desired[i - 1].sum() + tmp_trend[i] + state_shocks[i - 1]
    desired = desired + measurement_shocks
    assert_allclose(actual, desired)

    # Test using the model versus the results class
    mod_actual = mod.simulate(
        res.params, nsimulations, anchor='end',
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state)

    assert_allclose(mod_actual, desired)


def test_varmax_end_time_varying_exog_noshocks():
    # Test simulating values from the end of a time-varying VARMAX model
    # with exog
    # In this test, we suppress randomness by setting the shocks to zeros
    endog = np.arange(1, 21).reshape(10, 2)
    exog = np.arange(1, 21)**2
    mod = varmax.VARMAX(endog, trend='n', exog=exog[:10])
    res = mod.filter([1., 1., 1., 1., 1., 1., 1., 0.5, 1.])

    nsimulations = 10
    measurement_shocks = np.zeros((nsimulations, mod.k_endog))
    state_shocks = np.zeros((nsimulations, mod.k_states))

    # Need to set the final predicted state given the new exog
    tmp_exog = mod._validate_out_of_sample_exog(exog[10:], out_of_sample=10)
    with res._set_final_predicted_state(exog=tmp_exog, out_of_sample=10):
        initial_state = res.predicted_state[:, -1].copy()

    # Simulation
    actual = res.simulate(nsimulations, exog=exog[10:], anchor='end',
                          measurement_shocks=measurement_shocks,
                          state_shocks=state_shocks,
                          initial_state=initial_state)

    desired = np.zeros((nsimulations, mod.k_endog))
    desired[0] = initial_state
    for i in range(1, nsimulations):
        desired[i] = desired[i - 1].sum() + exog[10 + i] + state_shocks[i - 1]
    desired = desired + measurement_shocks
    assert_allclose(actual, desired)

    # Test using the model versus the results class
    mod_actual = mod.simulate(
        res.params, nsimulations, exog=exog[10:], anchor='end',
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state)

    assert_allclose(mod_actual, desired)

    # Alternatively, since we've shut down the shocks, we can compare against
    # the forecast values
    assert_allclose(actual, res.forecast(nsimulations, exog=exog[10:]))


def test_varmax_end_time_varying_exog_shocks(reset_randomstate):
    # Test simulating values from the end of a time-varying VARMAX model
    # with exog
    endog = np.arange(1, 23).reshape(11, 2)
    exog = np.arange(1, 21)**2
    mod = varmax.VARMAX(endog[:10], trend='n', exog=exog[:10])
    res = mod.filter([1., 1., 1., 1., 1., 1., 1., 0.5, 1.])

    mod2 = varmax.VARMAX(endog, trend='n', exog=exog[:11])
    res2 = mod2.filter([1., 1., 1., 1., 1., 1., 1., 0.5, 1.])

    nsimulations = 10
    measurement_shocks = np.random.normal(size=(nsimulations, mod.k_endog))
    state_shocks = np.random.normal(size=(nsimulations, mod.k_states))

    # Need to set the final predicted state given the new exog
    tmp_exog = mod._validate_out_of_sample_exog(exog[10:], out_of_sample=10)
    with res._set_final_predicted_state(exog=tmp_exog, out_of_sample=10):
        initial_state = res.predicted_state[:, -1].copy()

    # Simulation
    actual = res.simulate(nsimulations, exog=exog[10:], anchor='end',
                          measurement_shocks=measurement_shocks,
                          state_shocks=state_shocks,
                          initial_state=initial_state)
    actual2 = res2.simulate(nsimulations, exog=exog[11:], anchor=-1,
                            measurement_shocks=measurement_shocks,
                            state_shocks=state_shocks,
                            initial_state=res2.predicted_state[:, -2])

    desired = np.zeros((nsimulations, mod.k_endog))
    desired[0] = initial_state
    for i in range(1, nsimulations):
        desired[i] = desired[i - 1].sum() + exog[10 + i] + state_shocks[i - 1]
    desired = desired + measurement_shocks
    assert_allclose(actual, desired)
    assert_allclose(actual2, desired)

    # Test using the model versus the results class
    mod_actual = mod.simulate(
        res.params, nsimulations, exog=exog[10:], anchor='end',
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state)

    assert_allclose(mod_actual, desired)


def test_dynamic_factor_end_time_invariant_noshocks():
    # Test simulating values from the end of a time-invariant dynamic factor
    # In this test, we suppress randomness by setting the shocks to zeros
    endog = np.arange(1, 21).reshape(10, 2)
    mod = dynamic_factor.DynamicFactor(endog, k_factors=1, factor_order=1)
    mod.ssm.filter_univariate = True
    res = mod.filter([1., 1., 1., 1., 1.])

    nsimulations = 10
    measurement_shocks = np.zeros((nsimulations, mod.k_endog))
    state_shocks = np.zeros((nsimulations, mod.k_states))
    initial_state = res.predicted_state[..., -1]

    # Simulation
    actual = res.simulate(nsimulations, anchor='end',
                          measurement_shocks=measurement_shocks,
                          state_shocks=state_shocks,
                          initial_state=initial_state)

    # Construct the simulation directly
    desired = np.zeros((nsimulations, mod.k_endog))
    desired[0] = initial_state
    for i in range(1, nsimulations):
        desired[i] = desired[i - 1] + state_shocks[i - 1]
    desired = desired + measurement_shocks
    assert_allclose(actual, desired)

    # Test using the model versus the results class
    mod_actual = mod.simulate(
        res.params, nsimulations, anchor='end',
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state)

    assert_allclose(mod_actual, desired)

    # Alternatively, since we've shut down the shocks, we can compare against
    # the forecast values
    assert_allclose(actual, res.forecast(nsimulations))


def test_dynamic_factor_end_time_invariant_shocks(reset_randomstate):
    # Test simulating values from the end of a time-invariant dynamic factor
    endog = np.arange(1, 21).reshape(10, 2)
    mod = dynamic_factor.DynamicFactor(endog, k_factors=1, factor_order=1)
    mod.ssm.filter_univariate = True
    res = mod.filter([1., 1., 1., 1., 1., 1., 1.])

    nsimulations = 10
    measurement_shocks = np.random.normal(size=(nsimulations, mod.k_endog))
    state_shocks = np.random.normal(size=(nsimulations, mod.k_states))
    initial_state = res.predicted_state[..., -1]

    # Simulation
    actual = res.simulate(nsimulations, anchor='end',
                          measurement_shocks=measurement_shocks,
                          state_shocks=state_shocks,
                          initial_state=initial_state)

    # Construct the simulation directly
    desired = np.zeros((nsimulations, mod.k_endog))
    desired[0] = initial_state
    for i in range(1, nsimulations):
        desired[i] = desired[i - 1] + state_shocks[i - 1]
    desired = desired + measurement_shocks
    assert_allclose(actual, desired)

    # Test using the model versus the results class
    mod_actual = mod.simulate(
        res.params, nsimulations, anchor='end',
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state)

    assert_allclose(mod_actual, desired)


def test_dynamic_factor_end_time_varying_exog_noshocks():
    # Test simulating values from the end of a time-varying dynamic factor
    # model with exogenous inputs
    # In this test, we suppress randomness by setting the shocks to zeros
    endog = np.arange(1, 21).reshape(10, 2)
    exog = np.arange(1, 21)**2
    mod = dynamic_factor.DynamicFactor(endog, k_factors=1, factor_order=1,
                                       exog=exog[:10])
    mod.ssm.filter_univariate = True
    res = mod.filter([1., 1., 1., 1., 1., 1., 1.])

    nsimulations = 10
    measurement_shocks = np.zeros((nsimulations, mod.k_endog))
    state_shocks = np.zeros((nsimulations, mod.k_states))
    initial_state = res.predicted_state[..., -1]

    # Simulation
    actual = res.simulate(nsimulations, exog=exog[10:], anchor='end',
                          measurement_shocks=measurement_shocks,
                          state_shocks=state_shocks,
                          initial_state=initial_state)

    # Construct the simulation directly
    desired = np.zeros((nsimulations, mod.k_endog))
    desired[0] = initial_state
    for i in range(1, nsimulations):
        desired[i] = desired[i - 1] + state_shocks[i - 1]
    desired = desired + measurement_shocks + exog[10:, None]
    assert_allclose(actual, desired)

    # Test using the model versus the results class
    mod_actual = mod.simulate(
        res.params, nsimulations, exog=exog[10:], anchor='end',
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state)

    assert_allclose(mod_actual, desired)

    # Alternatively, since we've shut down the shocks, we can compare against
    # the forecast values
    assert_allclose(actual, res.forecast(nsimulations, exog=exog[10:]))


def test_dynamic_factor_end_time_varying_exog_shocks(reset_randomstate):
    # Test simulating values from the end of a time-varying dynamic factor
    # model with exogenous inputs
    endog = np.arange(1, 23).reshape(11, 2)
    exog = np.arange(1, 21)**2
    mod = dynamic_factor.DynamicFactor(
        endog[:10], k_factors=1, factor_order=1, exog=exog[:10])
    mod.ssm.filter_univariate = True
    res = mod.filter([1., 1., 1., 1., 1., 1., 1.])

    mod2 = dynamic_factor.DynamicFactor(
        endog, k_factors=1, factor_order=1, exog=exog[:11])
    mod2.ssm.filter_univariate = True
    res2 = mod2.filter([1., 1., 1., 1., 1., 1., 1.])

    nsimulations = 10
    measurement_shocks = np.random.normal(size=(nsimulations, mod.k_endog))
    state_shocks = np.random.normal(size=(nsimulations, mod.k_states))
    initial_state = res.predicted_state[..., -1]

    # Simulations
    actual = res.simulate(nsimulations, exog=exog[10:], anchor='end',
                          measurement_shocks=measurement_shocks,
                          state_shocks=state_shocks,
                          initial_state=initial_state)
    actual2 = res2.simulate(nsimulations, exog=exog[11:], anchor=-1,
                            measurement_shocks=measurement_shocks,
                            state_shocks=state_shocks,
                            initial_state=initial_state)

    # Construct the simulation directly
    desired = np.zeros((nsimulations, mod.k_endog))
    desired[0] = initial_state
    for i in range(1, nsimulations):
        desired[i] = desired[i - 1] + state_shocks[i - 1]
    desired = desired + measurement_shocks + exog[10:, None]
    assert_allclose(actual, desired)
    assert_allclose(actual2, desired)

    # Test using the model versus the results class
    mod_actual = mod.simulate(
        res.params, nsimulations, exog=exog[10:], anchor='end',
        measurement_shocks=measurement_shocks,
        state_shocks=state_shocks,
        initial_state=initial_state)

    assert_allclose(mod_actual, desired)


def test_pandas_univariate_rangeindex():
    # Simulate will also have RangeIndex
    endog = pd.Series(np.zeros(2))
    mod = sarimax.SARIMAX(endog)
    res = mod.filter([0.5, 1.])

    # Default simulate anchors to the start of the sample
    actual = res.simulate(2, state_shocks=np.zeros(2),
                          initial_state=np.zeros(1))
    desired = pd.Series([0, 0])
    assert_allclose(actual, desired)

    # Alternative anchor changes the index
    actual = res.simulate(2, anchor=2, state_shocks=np.zeros(2),
                          initial_state=np.zeros(1))
    ix = pd.RangeIndex(2, 4)
    desired = pd.Series([0, 0], index=ix)
    assert_allclose(actual, desired)
    assert_(actual.index.equals(desired.index))


def test_pandas_univariate_rangeindex_repetitions():
    # Simulate will also have RangeIndex
    endog = pd.Series(np.zeros(2))
    mod = sarimax.SARIMAX(endog)
    res = mod.filter([0.5, 1.])

    # Default simulate anchors to the start of the sample
    actual = res.simulate(2, state_shocks=np.zeros(2),
                          initial_state=np.zeros(1), repetitions=2)
    columns = pd.MultiIndex.from_product([['y'], [0, 1]])
    desired = pd.DataFrame(np.zeros((2, 2)), columns=columns)
    assert_allclose(actual, desired)
    assert_(actual.columns.equals(desired.columns))

    # Alternative anchor changes the index
    actual = res.simulate(2, anchor=2, state_shocks=np.zeros(2),
                          initial_state=np.zeros(1), repetitions=2)
    ix = pd.RangeIndex(2, 4)
    columns = pd.MultiIndex.from_product([['y'], [0, 1]])
    desired = pd.DataFrame(np.zeros((2, 2)), index=ix, columns=columns)
    assert_allclose(actual, desired)
    assert_(actual.index.equals(desired.index))
    assert_(actual.columns.equals(desired.columns))


def test_pandas_univariate_dateindex():
    # Simulation will maintain have date index
    ix = pd.date_range(start='2000', periods=2, freq=MONTH_END)
    endog = pd.Series(np.zeros(2), index=ix)
    mod = sarimax.SARIMAX(endog)
    res = mod.filter([0.5, 1.])

    # Default simulate anchors to the start of the sample
    actual = res.simulate(2, state_shocks=np.zeros(2),
                          initial_state=np.zeros(1))
    ix = pd.date_range(start='2000-01', periods=2, freq=MONTH_END)
    desired = pd.Series([0, 0], index=ix)
    assert_allclose(actual, desired)
    assert_(actual.index.equals(desired.index))

    # Alternative anchor changes the index
    actual = res.simulate(2, anchor=2, state_shocks=np.zeros(2),
                          initial_state=np.zeros(1))
    ix = pd.date_range(start='2000-03', periods=2, freq=MONTH_END)
    desired = pd.Series([0, 0], index=ix)
    assert_allclose(actual, desired)


def test_pandas_univariate_dateindex_repetitions():
    # Simulation will maintain have date index
    ix = pd.date_range(start='2000', periods=2, freq=MONTH_END)
    endog = pd.Series(np.zeros(2), index=ix)
    mod = sarimax.SARIMAX(endog)
    res = mod.filter([0.5, 1.])

    # Default simulate anchors to the start of the sample
    actual = res.simulate(2, state_shocks=np.zeros(2),
                          initial_state=np.zeros(1), repetitions=2)
    ix = pd.date_range(start='2000-01', periods=2, freq=MONTH_END)
    columns = pd.MultiIndex.from_product([['y'], [0, 1]])
    desired = pd.DataFrame(np.zeros((2, 2)), index=ix, columns=columns)
    assert_allclose(actual, desired)
    assert_(actual.columns.equals(desired.columns))

    # Alternative anchor changes the index
    actual = res.simulate(2, anchor=2, state_shocks=np.zeros(2),
                          initial_state=np.zeros(1), repetitions=2)
    ix = pd.date_range(start='2000-03', periods=2, freq=MONTH_END)
    columns = pd.MultiIndex.from_product([['y'], [0, 1]])
    desired = pd.DataFrame(np.zeros((2, 2)), index=ix, columns=columns)
    assert_allclose(actual, desired)
    assert_(actual.index.equals(desired.index))
    assert_(actual.columns.equals(desired.columns))


def test_pandas_multivariate_rangeindex():
    # Simulate will also have RangeIndex
    endog = pd.DataFrame(np.zeros((2, 2)))
    mod = varmax.VARMAX(endog, trend='n')
    res = mod.filter([0.5, 0., 0., 0.2, 1., 0., 1.])

    # Default simulate anchors to the start of the sample
    actual = res.simulate(2, state_shocks=np.zeros((2, 2)),
                          initial_state=np.zeros(2))
    desired = pd.DataFrame(np.zeros((2, 2)))
    assert_allclose(actual, desired)

    # Alternative anchor changes the index
    actual = res.simulate(2, anchor=2, state_shocks=np.zeros((2, 2)),
                          initial_state=np.zeros(2))
    ix = pd.RangeIndex(2, 4)
    desired = pd.DataFrame(np.zeros((2, 2)), index=ix)
    assert_allclose(actual, desired)
    assert_(actual.index.equals(desired.index))


def test_pandas_multivariate_rangeindex_repetitions():
    # Simulate will also have RangeIndex
    endog = pd.DataFrame(np.zeros((2, 2)), columns=['y1', 'y2'])
    mod = varmax.VARMAX(endog, trend='n')
    res = mod.filter([0.5, 0., 0., 0.2, 1., 0., 1.])

    # Default simulate anchors to the start of the sample
    actual = res.simulate(2, state_shocks=np.zeros((2, 2)),
                          initial_state=np.zeros(2), repetitions=2)
    columns = pd.MultiIndex.from_product([['y1', 'y2'], [0, 1]])
    desired = pd.DataFrame(np.zeros((2, 4)), columns=columns)
    assert_allclose(actual, desired)
    assert_(actual.columns.equals(desired.columns))

    # Alternative anchor changes the index
    actual = res.simulate(2, anchor=2, state_shocks=np.zeros((2, 2)),
                          initial_state=np.zeros(2), repetitions=2)
    ix = pd.RangeIndex(2, 4)
    columns = pd.MultiIndex.from_product([['y1', 'y2'], [0, 1]])
    desired = pd.DataFrame(np.zeros((2, 4)), index=ix, columns=columns)
    assert_allclose(actual, desired)
    assert_(actual.index.equals(desired.index))
    assert_(actual.columns.equals(desired.columns))


def test_pandas_multivariate_dateindex():
    # Simulate will also have RangeIndex
    ix = pd.date_range(start='2000', periods=2, freq=MONTH_END)
    endog = pd.DataFrame(np.zeros((2, 2)), index=ix)
    mod = varmax.VARMAX(endog, trend='n')
    res = mod.filter([0.5, 0., 0., 0.2, 1., 0., 1.])

    # Default simulate anchors to the start of the sample
    actual = res.simulate(2, state_shocks=np.zeros((2, 2)),
                          initial_state=np.zeros(2))
    desired = pd.DataFrame(np.zeros((2, 2)), index=ix)
    assert_allclose(actual, desired)

    # Alternative anchor changes the index
    actual = res.simulate(2, anchor=2, state_shocks=np.zeros((2, 2)),
                          initial_state=np.zeros(2))
    ix = pd.date_range(start='2000-03', periods=2, freq=MONTH_END)
    desired = pd.DataFrame(np.zeros((2, 2)), index=ix)
    assert_allclose(actual, desired)
    assert_(actual.index.equals(desired.index))


def test_pandas_multivariate_dateindex_repetitions():
    # Simulate will also have RangeIndex
    ix = pd.date_range(start='2000', periods=2, freq=MONTH_END)
    endog = pd.DataFrame(np.zeros((2, 2)), columns=['y1', 'y2'], index=ix)
    mod = varmax.VARMAX(endog, trend='n')
    res = mod.filter([0.5, 0., 0., 0.2, 1., 0., 1.])

    # Default simulate anchors to the start of the sample
    actual = res.simulate(2, state_shocks=np.zeros((2, 2)),
                          initial_state=np.zeros(2), repetitions=2)
    columns = pd.MultiIndex.from_product([['y1', 'y2'], [0, 1]])
    desired = pd.DataFrame(np.zeros((2, 4)), columns=columns, index=ix)
    assert_allclose(actual, desired)
    assert_(actual.columns.equals(desired.columns))

    # Alternative anchor changes the index
    actual = res.simulate(2, anchor=2, state_shocks=np.zeros((2, 2)),
                          initial_state=np.zeros(2), repetitions=2)
    ix = pd.date_range(start='2000-03', periods=2, freq=MONTH_END)
    columns = pd.MultiIndex.from_product([['y1', 'y2'], [0, 1]])
    desired = pd.DataFrame(np.zeros((2, 4)), index=ix, columns=columns)
    assert_allclose(actual, desired)
    assert_(actual.index.equals(desired.index))
    assert_(actual.columns.equals(desired.columns))


def test_pandas_anchor():
    # Test that anchor with dates works
    ix = pd.date_range(start='2000', periods=2, freq=MONTH_END)
    endog = pd.Series(np.zeros(2), index=ix)
    mod = sarimax.SARIMAX(endog)
    res = mod.filter([0.5, 1.])

    desired = res.simulate(2, anchor=1, state_shocks=np.zeros(2),
                           initial_state=np.zeros(1))

    # Anchor to date
    actual = res.simulate(2, anchor=ix[1], state_shocks=np.zeros(2),
                          initial_state=np.zeros(1))
    assert_allclose(actual, desired)
    assert_(actual.index.equals(desired.index))

    # Anchor to negative index
    actual = res.simulate(2, anchor=-1, state_shocks=np.zeros(2),
                          initial_state=np.zeros(1))
    assert_allclose(actual, desired)
    assert_(actual.index.equals(desired.index))


@pytest.mark.smoke
def test_time_varying(reset_randomstate):
    mod = TVSS(np.zeros((10, 2)))
    mod.simulate([], 10)


def test_time_varying_obs_cov(reset_randomstate):
    mod = TVSS(np.zeros((10, 2)))
    mod['obs_cov'] = np.zeros((mod.k_endog, mod.k_endog, mod.nobs))
    mod['obs_cov', ..., 9] = np.eye(mod.k_endog)
    mod['state_intercept', :] = 0
    mod['state_cov'] = mod['state_cov', :, :, 0] * 0
    mod['selection'] = mod['selection', :, :, 0]
    assert_equal(mod['state_cov'].shape, (mod.ssm.k_posdef, mod.ssm.k_posdef))
    assert_equal(mod['selection'].shape, (mod.k_states, mod.ssm.k_posdef))

    sim = mod.simulate([], 10, initial_state=np.zeros(mod.k_states))
    assert_allclose(sim[:9], mod['obs_intercept', :, :9].T)


def test_time_varying_state_cov(reset_randomstate):
    mod = TVSS(np.zeros((10, 2)))
    mod['obs_cov'] = mod['obs_cov', :, :, 0] * 0
    mod['selection'] = mod['selection', :, :, 0]
    mod['state_intercept', :] = 0
    mod['state_cov'] = np.zeros((mod.ssm.k_posdef, mod.ssm.k_posdef, mod.nobs))
    mod['state_cov', ..., -1] = np.eye(mod.ssm.k_posdef)
    assert_equal(mod['obs_cov'].shape, (mod.k_endog, mod.k_endog))
    assert_equal(mod['selection'].shape, (mod.k_states, mod.ssm.k_posdef))
    sim = mod.simulate([], 10)
    assert_allclose(sim, mod['obs_intercept'].T)


@pytest.mark.smoke
def test_time_varying_selection(reset_randomstate):
    mod = TVSS(np.zeros((10, 2)))
    mod['obs_cov'] = mod['obs_cov', :, :, 0]
    mod['state_cov'] = mod['state_cov', :, :, 0]
    assert_equal(mod['obs_cov'].shape, (mod.k_endog, mod.k_endog))
    assert_equal(mod['state_cov'].shape, (mod.ssm.k_posdef, mod.ssm.k_posdef))
    mod.simulate([], 10)
