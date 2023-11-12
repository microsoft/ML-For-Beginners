"""
Tests for miscellaneous models

Author: Chad Fulton
License: Simplified-BSD
"""

import numpy as np
import pandas as pd
import os
import pytest

from statsmodels.tsa.statespace import mlemodel, sarimax
from statsmodels import datasets

from numpy.testing import assert_equal, assert_allclose, assert_raises

current_path = os.path.dirname(os.path.abspath(__file__))


class Intercepts(mlemodel.MLEModel):
    """
    Test class for observation and state intercepts (which usually do not
    get tested in other models).
    """
    def __init__(self, endog, **kwargs):
        k_states = 3
        k_posdef = 3
        super(Intercepts, self).__init__(
            endog, k_states=k_states, k_posdef=k_posdef, **kwargs)
        self['design'] = np.eye(3)
        self['obs_cov'] = np.eye(3)
        self['transition'] = np.eye(3)
        self['selection'] = np.eye(3)
        self['state_cov'] = np.eye(3)
        self.initialize_approximate_diffuse()

    @property
    def param_names(self):
        return ['d.1', 'd.2', 'd.3', 'c.1', 'c.2', 'c.3']

    @property
    def start_params(self):
        return np.arange(6)

    def update(self, params, **kwargs):
        params = super(Intercepts, self).update(params, **kwargs)

        self['obs_intercept'] = params[:3]
        self['state_intercept'] = params[3:]


class TestIntercepts:
    @classmethod
    def setup_class(cls, which='mixed', **kwargs):
        # Results
        path = current_path + os.sep + 'results/results_intercepts_R.csv'
        cls.desired = pd.read_csv(path)

        # Data
        dta = datasets.macrodata.load_pandas().data
        dta.index = pd.date_range(start='1959-01-01', end='2009-7-01',
                                  freq='QS')
        obs = dta[['realgdp', 'realcons', 'realinv']].copy()
        obs = obs / obs.std()

        if which == 'all':
            obs.iloc[:50, :] = np.nan
            obs.iloc[119:130, :] = np.nan
        elif which == 'partial':
            obs.iloc[0:50, 0] = np.nan
            obs.iloc[119:130, 0] = np.nan
        elif which == 'mixed':
            obs.iloc[0:50, 0] = np.nan
            obs.iloc[19:70, 1] = np.nan
            obs.iloc[39:90, 2] = np.nan
            obs.iloc[119:130, 0] = np.nan
            obs.iloc[119:130, 2] = np.nan

        mod = Intercepts(obs, **kwargs)

        cls.params = np.arange(6) + 1
        cls.model = mod

        cls.results = mod.smooth(cls.params, return_ssm=True)

        # Calculate the determinant of the covariance matrices (for easy
        # comparison to other languages without having to store 2-dim arrays)
        cls.results.det_scaled_smoothed_estimator_cov = (
            np.zeros((1, cls.model.nobs)))
        cls.results.det_predicted_state_cov = np.zeros((1, cls.model.nobs))
        cls.results.det_smoothed_state_cov = np.zeros((1, cls.model.nobs))
        cls.results.det_smoothed_state_disturbance_cov = (
            np.zeros((1, cls.model.nobs)))

        for i in range(cls.model.nobs):
            cls.results.det_scaled_smoothed_estimator_cov[0, i] = (
                np.linalg.det(
                    cls.results.scaled_smoothed_estimator_cov[:, :, i]))
            cls.results.det_predicted_state_cov[0, i] = np.linalg.det(
                cls.results.predicted_state_cov[:, :, i+1])
            cls.results.det_smoothed_state_cov[0, i] = np.linalg.det(
                cls.results.smoothed_state_cov[:, :, i])
            cls.results.det_smoothed_state_disturbance_cov[0, i] = (
                np.linalg.det(
                    cls.results.smoothed_state_disturbance_cov[:, :, i]))

    def test_loglike(self):
        assert_allclose(np.sum(self.results.llf_obs), -7924.03893566)

    def test_scaled_smoothed_estimator(self):
        assert_allclose(
            self.results.scaled_smoothed_estimator.T,
            self.desired[['r1', 'r2', 'r3']]
        )

    def test_scaled_smoothed_estimator_cov(self):
        assert_allclose(
            self.results.det_scaled_smoothed_estimator_cov.T,
            self.desired[['detN']]
        )

    def test_forecasts(self):
        assert_allclose(
            self.results.forecasts.T,
            self.desired[['m1', 'm2', 'm3']]
        )

    def test_forecasts_error(self):
        assert_allclose(
            self.results.forecasts_error.T,
            self.desired[['v1', 'v2', 'v3']]
        )

    def test_forecasts_error_cov(self):
        assert_allclose(
            self.results.forecasts_error_cov.diagonal(),
            self.desired[['F1', 'F2', 'F3']]
        )

    def test_predicted_states(self):
        assert_allclose(
            self.results.predicted_state[:, 1:].T,
            self.desired[['a1', 'a2', 'a3']]
        )

    def test_predicted_states_cov(self):
        assert_allclose(
            self.results.det_predicted_state_cov.T,
            self.desired[['detP']]
        )

    def test_smoothed_states(self):
        assert_allclose(
            self.results.smoothed_state.T,
            self.desired[['alphahat1', 'alphahat2', 'alphahat3']]
        )

    def test_smoothed_states_cov(self):
        assert_allclose(
            self.results.det_smoothed_state_cov.T,
            self.desired[['detV']]
        )

    def test_smoothed_forecasts(self):
        assert_allclose(
            self.results.smoothed_forecasts.T,
            self.desired[['muhat1', 'muhat2', 'muhat3']]
        )

    def test_smoothed_state_disturbance(self):
        assert_allclose(
            self.results.smoothed_state_disturbance.T,
            self.desired[['etahat1', 'etahat2', 'etahat3']]
        )

    def test_smoothed_state_disturbance_cov(self):
        assert_allclose(
            self.results.det_smoothed_state_disturbance_cov.T,
            self.desired[['detVeta']]
        )

    def test_smoothed_measurement_disturbance(self):
        assert_allclose(
            self.results.smoothed_measurement_disturbance.T,
            self.desired[['epshat1', 'epshat2', 'epshat3']], atol=1e-9
        )

    def test_smoothed_measurement_disturbance_cov(self):
        assert_allclose(
            self.results.smoothed_measurement_disturbance_cov.diagonal(),
            self.desired[['Veps1', 'Veps2', 'Veps3']]
        )


class LargeStateCovAR1(mlemodel.MLEModel):
    """
    Test class for k_posdef > k_states (which usually do not get tested in
    other models).

    This is just an AR(1) model with an extra unused state innovation
    """
    def __init__(self, endog, **kwargs):
        k_states = 1
        k_posdef = 2
        super(LargeStateCovAR1, self).__init__(
            endog, k_states=k_states, k_posdef=k_posdef, **kwargs)
        self['design', 0, 0] = 1
        self['selection', 0, 0] = 1
        self['state_cov', 1, 1] = 1
        self.initialize_stationary()

    @property
    def param_names(self):
        return ['phi', 'sigma2']

    @property
    def start_params(self):
        return [0.5, 1]

    def update(self, params, **kwargs):
        params = super(LargeStateCovAR1, self).update(params, **kwargs)

        self['transition', 0, 0] = params[0]
        self['state_cov', 0, 0] = params[1]


def test_large_kposdef():
    assert_raises(ValueError, LargeStateCovAR1, np.arange(10))


class TestLargeStateCovAR1:
    @classmethod
    def setup_class(cls):
        pytest.skip(
            'TODO: This test is skipped since an exception is currently '
            'raised if k_posdef > k_states. However, this test could be '
            'used if models of those types were allowed'
        )

        # Data: just some sample data
        endog = [0.2, -1.5, -.3, -.1, 1.5, 0.2, -0.3, 0.2, 0.5, 0.8]

        # Params
        params = [0.5, 1]

        # Desired model: AR(1)
        mod_desired = sarimax.SARIMAX(endog)
        cls.res_desired = mod_desired.smooth(params)

        # Test class
        mod = LargeStateCovAR1(endog)
        cls.res = mod.smooth(params)

    def test_dimensions(self):
        assert_equal(self.res.filter_results.k_states, 1)
        assert_equal(self.res.filter_results.k_posdef, 2)
        assert_equal(self.res.smoothed_state_disturbance.shape, (2, 10))

        assert_equal(self.res_desired.filter_results.k_states, 1)
        assert_equal(self.res_desired.filter_results.k_posdef, 1)
        assert_equal(self.res_desired.smoothed_state_disturbance.shape,
                     (1, 10))

    def test_loglike(self):
        assert_allclose(self.res.llf_obs, self.res_desired.llf_obs)

    def test_scaled_smoothed_estimator(self):
        assert_allclose(self.res.scaled_smoothed_estimator[0],
                        self.res_desired.scaled_smoothed_estimator[0])

    def test_scaled_smoothed_estimator_cov(self):
        assert_allclose(self.res.scaled_smoothed_estimator_cov[0],
                        self.res_desired.scaled_smoothed_estimator_cov[0])

    def test_forecasts(self):
        assert_allclose(self.res.forecasts, self.res_desired.forecasts)

    def test_forecasts_error(self):
        assert_allclose(self.res.forecasts_error,
                        self.res_desired.forecasts_error)

    def test_forecasts_error_cov(self):
        assert_allclose(self.res.forecasts_error_cov,
                        self.res_desired.forecasts_error_cov)

    def test_predicted_states(self):
        assert_allclose(self.res.predicted_state[0],
                        self.res_desired.predicted_state[0])

    def test_predicted_states_cov(self):
        assert_allclose(self.res.predicted_state_cov[0, 0],
                        self.res_desired.predicted_state_cov[0, 0])

    def test_smoothed_states(self):
        assert_allclose(self.res.smoothed_state[0],
                        self.res_desired.smoothed_state[0])

    def test_smoothed_states_cov(self):
        assert_allclose(self.res.smoothed_state_cov[0, 0],
                        self.res_desired.smoothed_state_cov[0, 0])

    def test_smoothed_state_disturbance(self):
        assert_allclose(self.res.smoothed_state_disturbance[0],
                        self.res_desired.smoothed_state_disturbance[0])
        assert_allclose(self.res.smoothed_state_disturbance[1], 0)

    def test_smoothed_state_disturbance_cov(self):
        assert_allclose(self.res.smoothed_state_disturbance_cov[0, 0],
                        self.res_desired.smoothed_state_disturbance_cov[0, 0])
        assert_allclose(self.res.smoothed_state_disturbance[1, 1], 0)

    def test_smoothed_measurement_disturbance(self):
        assert_allclose(self.res.smoothed_measurement_disturbance,
                        self.res_desired.smoothed_measurement_disturbance)

    def test_smoothed_measurement_disturbance_cov(self):
        assert_allclose(self.res.smoothed_measurement_disturbance_cov,
                        self.res_desired.smoothed_measurement_disturbance_cov)
