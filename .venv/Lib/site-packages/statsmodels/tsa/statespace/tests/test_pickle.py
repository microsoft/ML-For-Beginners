"""
Tests for python wrapper of state space representation and filtering

Author: Chad Fulton
License: Simplified-BSD

References
----------

Kim, Chang-Jin, and Charles R. Nelson. 1999.
"State-Space Models with Regime Switching:
Classical and Gibbs-Sampling Approaches with Applications".
MIT Press Books. The MIT Press.
"""
import pickle

import numpy as np
import pandas as pd
from numpy.testing import assert_equal, assert_allclose
import pytest

from statsmodels.tsa.statespace import sarimax
from statsmodels.tsa.statespace.kalman_filter import KalmanFilter
from statsmodels.tsa.statespace.representation import Representation
from statsmodels.tsa.statespace.structural import UnobservedComponents
from .results import results_kalman_filter


@pytest.fixture()
def data():
    true = results_kalman_filter.uc_uni
    data_ = pd.DataFrame(
        true['data'],
        index=pd.date_range('1947-01-01', '1995-07-01', freq='QS'),
        columns=['GDP']
    )
    data_['lgdp'] = np.log(data_['GDP'])
    return data_


def test_pickle_fit_sarimax(data):
    # Fit an ARIMA(1,1,0) to log GDP
    mod = sarimax.SARIMAX(data['lgdp'], order=(1, 1, 0))
    pkl_mod = pickle.loads(pickle.dumps(mod))

    res = mod.fit(disp=-1, full_output=True, method='newton')
    pkl_res = pkl_mod.fit(disp=-1, full_output=True, method='newton')

    assert_allclose(res.llf_obs, pkl_res.llf_obs)
    assert_allclose(res.tvalues, pkl_res.tvalues)
    assert_allclose(res.smoothed_state, pkl_res.smoothed_state)
    assert_allclose(res.resid.values, pkl_res.resid.values)
    assert_allclose(res.impulse_responses(10), res.impulse_responses(10))


def test_unobserved_components_pickle():
    # Tests for missing data
    nobs = 20
    k_endog = 1
    np.random.seed(1208)
    endog = np.random.normal(size=(nobs, k_endog))
    endog[:4, 0] = np.nan
    exog2 = np.random.normal(size=(nobs, 2))

    index = pd.date_range('1970-01-01', freq='QS', periods=nobs)
    endog_pd = pd.DataFrame(endog, index=index)
    exog2_pd = pd.DataFrame(exog2, index=index)

    models = [
        UnobservedComponents(endog, 'llevel', exog=exog2),
        UnobservedComponents(endog_pd, 'llevel', exog=exog2_pd),
    ]

    for mod in models:
        # Smoke tests
        pkl_mod = pickle.loads(pickle.dumps(mod))
        assert_equal(mod.start_params, pkl_mod.start_params)
        res = mod.fit(disp=False)
        pkl_res = pkl_mod.fit(disp=False)

        assert_allclose(res.llf_obs, pkl_res.llf_obs)
        assert_allclose(res.tvalues, pkl_res.tvalues)
        assert_allclose(res.smoothed_state, pkl_res.smoothed_state)
        assert_allclose(res.resid, pkl_res.resid)
        assert_allclose(res.impulse_responses(10), res.impulse_responses(10))


def test_kalman_filter_pickle(data):
    # Construct the statespace representation
    true = results_kalman_filter.uc_uni
    k_states = 4
    model = KalmanFilter(k_endog=1, k_states=k_states)
    model.bind(data['lgdp'].values)

    model.design[:, :, 0] = [1, 1, 0, 0]
    model.transition[([0, 0, 1, 1, 2, 3],
                      [0, 3, 1, 2, 1, 3],
                      [0, 0, 0, 0, 0, 0])] = [1, 1, 0, 0, 1, 1]
    model.selection = np.eye(model.k_states)

    # Update matrices with given parameters
    (sigma_v, sigma_e, sigma_w, phi_1, phi_2) = np.array(
        true['parameters']
    )
    model.transition[([1, 1], [1, 2], [0, 0])] = [phi_1, phi_2]
    model.state_cov[
        np.diag_indices(k_states) + (np.zeros(k_states, dtype=int),)] = [
        sigma_v ** 2, sigma_e ** 2, 0, sigma_w ** 2
    ]

    # Initialization
    initial_state = np.zeros((k_states,))
    initial_state_cov = np.eye(k_states) * 100

    # Initialization: modification
    initial_state_cov = np.dot(
        np.dot(model.transition[:, :, 0], initial_state_cov),
        model.transition[:, :, 0].T
    )
    model.initialize_known(initial_state, initial_state_cov)
    pkl_mod = pickle.loads(pickle.dumps(model))

    results = model.filter()
    pkl_results = pkl_mod.filter()

    assert_allclose(results.llf_obs[true['start']:].sum(),
                    pkl_results.llf_obs[true['start']:].sum())
    assert_allclose(results.filtered_state[0][true['start']:],
                    pkl_results.filtered_state[0][true['start']:])
    assert_allclose(results.filtered_state[1][true['start']:],
                    pkl_results.filtered_state[1][true['start']:])
    assert_allclose(results.filtered_state[3][true['start']:],
                    pkl_results.filtered_state[3][true['start']:])


def test_representation_pickle():
    nobs = 10
    k_endog = 2
    arr = np.arange(nobs * k_endog).reshape(k_endog, nobs) * 1.
    endog = np.asfortranarray(arr)
    mod = Representation(endog, k_states=2)
    pkl_mod = pickle.loads(pickle.dumps(mod))

    assert_equal(mod.nobs, pkl_mod.nobs)
    assert_equal(mod.k_endog, pkl_mod.k_endog)

    mod._initialize_representation()
    pkl_mod._initialize_representation()
    assert_equal(mod.design, pkl_mod.design)
    assert_equal(mod.obs_intercept, pkl_mod.obs_intercept)
    assert_equal(mod.initial_variance, pkl_mod.initial_variance)
