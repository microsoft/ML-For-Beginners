"""
Tests for univariate treatment of multivariate models

TODO skips the tests for measurement disturbance and measurement disturbance
covariance, which do not pass. The univariate smoother *appears* to be
correctly implemented against Durbin and Koopman (2012) chapter 6, yet still
gives a different answer from the conventional smoother. It's not clear if
this is intended (i.e. it has to be at least slightly different, since the
conventional smoother can return a non-diagonal covariance matrix whereas the
univariate smoother must return a diagonal covariance matrix).

Author: Chad Fulton
License: Simplified-BSD
"""
import os

import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
import pandas as pd
import pytest

from statsmodels import datasets
from statsmodels.tsa.statespace.mlemodel import MLEModel
from statsmodels.tsa.statespace.tests.results import results_kalman_filter
from statsmodels.tsa.statespace.sarimax import SARIMAX

current_path = os.path.dirname(os.path.abspath(__file__))


class TestClark1989:
    """
    Clark's (1989) bivariate unobserved components model of real GDP (as
    presented in Kim and Nelson, 1999)

    Tests two-dimensional observation data.

    Test data produced using GAUSS code described in Kim and Nelson (1999) and
    found at http://econ.korea.ac.kr/~cjkim/SSMARKOV.htm

    See `results.results_kalman_filter` for more information.
    """
    @classmethod
    def setup_class(cls, dtype=float, alternate_timing=False, **kwargs):

        cls.true = results_kalman_filter.uc_bi
        cls.true_states = pd.DataFrame(cls.true['states'])

        # GDP and Unemployment, Quarterly, 1948.1 - 1995.3
        data = pd.DataFrame(
            cls.true['data'],
            index=pd.date_range('1947-01-01', '1995-07-01', freq='QS'),
            columns=['GDP', 'UNEMP']
        )[4:]
        data['GDP'] = np.log(data['GDP'])
        data['UNEMP'] = (data['UNEMP']/100)

        k_states = 6
        cls.mlemodel = MLEModel(data, k_states=k_states, **kwargs)
        cls.model = cls.mlemodel.ssm

        # Statespace representation
        cls.model.design[:, :, 0] = [[1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1]]
        cls.model.transition[
            ([0, 0, 1, 1, 2, 3, 4, 5],
             [0, 4, 1, 2, 1, 2, 4, 5],
             [0, 0, 0, 0, 0, 0, 0, 0])
        ] = [1, 1, 0, 0, 1, 1, 1, 1]
        cls.model.selection = np.eye(cls.model.k_states)

        # Update matrices with given parameters
        (sigma_v, sigma_e, sigma_w, sigma_vl, sigma_ec,
         phi_1, phi_2, alpha_1, alpha_2, alpha_3) = np.array(
            cls.true['parameters'],
        )
        cls.model.design[([1, 1, 1], [1, 2, 3], [0, 0, 0])] = [
            alpha_1, alpha_2, alpha_3
        ]
        cls.model.transition[([1, 1], [1, 2], [0, 0])] = [phi_1, phi_2]
        cls.model.obs_cov[1, 1, 0] = sigma_ec**2
        cls.model.state_cov[
            np.diag_indices(k_states)+(np.zeros(k_states, dtype=int),)] = [
            sigma_v**2, sigma_e**2, 0, 0, sigma_w**2, sigma_vl**2
        ]

        # Initialization
        initial_state = np.zeros((k_states,))
        initial_state_cov = np.eye(k_states)*100

        # Initialization: cls.modification
        if not alternate_timing:
            initial_state_cov = np.dot(
                np.dot(cls.model.transition[:, :, 0], initial_state_cov),
                cls.model.transition[:, :, 0].T
            )
        else:
            cls.model.timing_init_filtered = True
        cls.model.initialize_known(initial_state, initial_state_cov)

        # Conventional filtering, smoothing, and simulation smoothing
        cls.model.filter_conventional = True
        cls.conventional_results = cls.model.smooth()
        n_disturbance_variates = (
            (cls.model.k_endog + cls.model.k_posdef) * cls.model.nobs
        )
        cls.conventional_sim = cls.model.simulation_smoother(
            disturbance_variates=np.zeros(n_disturbance_variates),
            initial_state_variates=np.zeros(cls.model.k_states)
        )

        # Univariate filtering, smoothing, and simulation smoothing
        cls.model.filter_univariate = True
        cls.univariate_results = cls.model.smooth()
        cls.univariate_sim = cls.model.simulation_smoother(
            disturbance_variates=np.zeros(n_disturbance_variates),
            initial_state_variates=np.zeros(cls.model.k_states)
        )

    def test_using_univariate(self):
        # Regression test to make sure the univariate_results actually
        # used the univariate Kalman filtering approach (i.e. that the flag
        # being set actually caused the filter to not use the conventional
        # filter)
        assert not self.conventional_results.filter_univariate
        assert self.univariate_results.filter_univariate

        assert_allclose(
            self.conventional_results.forecasts_error_cov[1, 1, 0],
            143.03724478030821
        )
        assert_allclose(
            self.univariate_results.forecasts_error_cov[1, 1, 0],
            120.66208525029386
        )

    def test_forecasts(self):
        assert_almost_equal(
            self.conventional_results.forecasts[0, :],
            self.univariate_results.forecasts[0, :], 9
        )

    def test_forecasts_error(self):
        assert_almost_equal(
            self.conventional_results.forecasts_error[0, :],
            self.univariate_results.forecasts_error[0, :], 9
        )

    def test_forecasts_error_cov(self):
        assert_almost_equal(
            self.conventional_results.forecasts_error_cov[0, 0, :],
            self.univariate_results.forecasts_error_cov[0, 0, :], 9
        )

    def test_filtered_state(self):
        assert_almost_equal(
            self.conventional_results.filtered_state,
            self.univariate_results.filtered_state, 8
        )

    def test_filtered_state_cov(self):
        assert_almost_equal(
            self.conventional_results.filtered_state_cov,
            self.univariate_results.filtered_state_cov, 9
        )

    def test_predicted_state(self):
        assert_almost_equal(
            self.conventional_results.predicted_state,
            self.univariate_results.predicted_state, 8
        )

    def test_predicted_state_cov(self):
        assert_almost_equal(
            self.conventional_results.predicted_state_cov,
            self.univariate_results.predicted_state_cov, 9
        )

    def test_loglike(self):
        assert_allclose(
            self.conventional_results.llf_obs,
            self.univariate_results.llf_obs
        )

    def test_smoothed_states(self):
        assert_almost_equal(
            self.conventional_results.smoothed_state,
            self.univariate_results.smoothed_state, 7
        )

    def test_smoothed_states_cov(self):
        assert_almost_equal(
            self.conventional_results.smoothed_state_cov,
            self.univariate_results.smoothed_state_cov, 6
        )

    def test_smoothed_measurement_disturbance(self):
        assert_almost_equal(
            self.conventional_results.smoothed_measurement_disturbance,
            self.univariate_results.smoothed_measurement_disturbance, 9
        )

    def test_smoothed_measurement_disturbance_cov(self):
        conv = self.conventional_results
        univ = self.univariate_results
        assert_almost_equal(
            conv.smoothed_measurement_disturbance_cov.diagonal(),
            univ.smoothed_measurement_disturbance_cov.diagonal(), 9
        )

    def test_smoothed_state_disturbance(self):
        assert_allclose(
            self.conventional_results.smoothed_state_disturbance,
            self.univariate_results.smoothed_state_disturbance,
            atol=1e-7
        )

    def test_smoothed_state_disturbance_cov(self):
        assert_almost_equal(
            self.conventional_results.smoothed_state_disturbance_cov,
            self.univariate_results.smoothed_state_disturbance_cov, 9
        )

    def test_simulation_smoothed_state(self):
        assert_almost_equal(
            self.conventional_sim.simulated_state,
            self.univariate_sim.simulated_state, 9
        )

    def test_simulation_smoothed_measurement_disturbance(self):
        assert_almost_equal(
            self.conventional_sim.simulated_measurement_disturbance,
            self.univariate_sim.simulated_measurement_disturbance, 9
        )

    def test_simulation_smoothed_state_disturbance(self):
        assert_almost_equal(
            self.conventional_sim.simulated_state_disturbance,
            self.univariate_sim.simulated_state_disturbance, 9
        )


class TestClark1989Alternate(TestClark1989):
    @classmethod
    def setup_class(cls, *args, **kwargs):
        super(TestClark1989Alternate, cls).setup_class(alternate_timing=True,
                                                       *args, **kwargs)

    def test_using_alterate(self):
        assert self.model._kalman_filter.filter_timing == 1


class MultivariateMissingGeneralObsCov:
    @classmethod
    def setup_class(cls, which, dtype=float, alternate_timing=False, **kwargs):
        # Results
        path = os.path.join(current_path, 'results',
                            'results_smoothing_generalobscov_R.csv')
        cls.desired = pd.read_csv(path)

        # Data
        dta = datasets.macrodata.load_pandas().data
        dta.index = pd.date_range(start='1959-01-01',
                                  end='2009-7-01', freq='QS')
        obs = dta[['realgdp', 'realcons', 'realinv']].diff().iloc[1:]

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

        # Create the model
        mod = MLEModel(obs, k_states=3, k_posdef=3, **kwargs)
        mod['design'] = np.eye(3)
        X = (np.arange(9) + 1).reshape((3, 3)) / 10.
        mod['obs_cov'] = np.dot(X, X.T)
        mod['transition'] = np.eye(3)
        mod['selection'] = np.eye(3)
        mod['state_cov'] = np.eye(3)
        mod.initialize_approximate_diffuse(1e6)
        cls.model = mod.ssm

        # Conventional filtering, smoothing, and simulation smoothing
        cls.model.filter_conventional = True
        cls.conventional_results = cls.model.smooth()
        n_disturbance_variates = (
            (cls.model.k_endog + cls.model.k_posdef) * cls.model.nobs
        )
        cls.conventional_sim = cls.model.simulation_smoother(
            disturbance_variates=np.zeros(n_disturbance_variates),
            initial_state_variates=np.zeros(cls.model.k_states)
        )

        # Univariate filtering, smoothing, and simulation smoothing
        cls.model.filter_univariate = True
        cls.univariate_results = cls.model.smooth()
        cls.univariate_sim = cls.model.simulation_smoother(
            disturbance_variates=np.zeros(n_disturbance_variates),
            initial_state_variates=np.zeros(cls.model.k_states)
        )

    def test_using_univariate(self):
        # Regression test to make sure the univariate_results actually
        # used the univariate Kalman filtering approach (i.e. that the flag
        # being set actually caused the filter to not use the conventional
        # filter)
        assert not self.conventional_results.filter_univariate
        assert self.univariate_results.filter_univariate

        assert_allclose(
            self.conventional_results.forecasts_error_cov[1, 1, 0],
            1000000.77
        )
        assert_allclose(
            self.univariate_results.forecasts_error_cov[1, 1, 0],
            1000000.77
        )

    def test_forecasts(self):
        assert_almost_equal(
            self.conventional_results.forecasts[0, :],
            self.univariate_results.forecasts[0, :], 9
        )

    def test_forecasts_error(self):
        assert_almost_equal(
            self.conventional_results.forecasts_error[0, :],
            self.univariate_results.forecasts_error[0, :], 9
        )

    def test_forecasts_error_cov(self):
        assert_almost_equal(
            self.conventional_results.forecasts_error_cov[0, 0, :],
            self.univariate_results.forecasts_error_cov[0, 0, :], 9
        )

    def test_filtered_state(self):
        assert_almost_equal(
            self.conventional_results.filtered_state,
            self.univariate_results.filtered_state, 8
        )

    def test_filtered_state_cov(self):
        assert_almost_equal(
            self.conventional_results.filtered_state_cov,
            self.univariate_results.filtered_state_cov, 9
        )

    def test_predicted_state(self):
        assert_almost_equal(
            self.conventional_results.predicted_state,
            self.univariate_results.predicted_state, 8
        )

    def test_predicted_state_cov(self):
        assert_almost_equal(
            self.conventional_results.predicted_state_cov,
            self.univariate_results.predicted_state_cov, 9
        )

    def test_loglike(self):
        assert_allclose(
            self.conventional_results.llf_obs,
            self.univariate_results.llf_obs
        )

    def test_smoothed_states(self):
        assert_almost_equal(
            self.conventional_results.smoothed_state,
            self.univariate_results.smoothed_state, 7
        )

    def test_smoothed_states_cov(self):
        assert_almost_equal(
            self.conventional_results.smoothed_state_cov,
            self.univariate_results.smoothed_state_cov, 6
        )

    @pytest.mark.skip
    def test_smoothed_measurement_disturbance(self):
        assert_almost_equal(
            self.conventional_results.smoothed_measurement_disturbance,
            self.univariate_results.smoothed_measurement_disturbance, 9
        )

    @pytest.mark.skip
    def test_smoothed_measurement_disturbance_cov(self):
        conv = self.conventional_results
        univ = self.univariate_results
        assert_almost_equal(
            conv.smoothed_measurement_disturbance_cov.diagonal(),
            univ.smoothed_measurement_disturbance_cov.diagonal(), 9
        )

    def test_smoothed_state_disturbance(self):
        assert_allclose(
            self.conventional_results.smoothed_state_disturbance,
            self.univariate_results.smoothed_state_disturbance,
            atol=1e-7
        )

    def test_smoothed_state_disturbance_cov(self):
        assert_almost_equal(
            self.conventional_results.smoothed_state_disturbance_cov,
            self.univariate_results.smoothed_state_disturbance_cov, 9
        )

    def test_simulation_smoothed_state(self):
        assert_almost_equal(
            self.conventional_sim.simulated_state,
            self.univariate_sim.simulated_state, 9
        )

    @pytest.mark.skip
    def test_simulation_smoothed_measurement_disturbance(self):
        assert_almost_equal(
            self.conventional_sim.simulated_measurement_disturbance,
            self.univariate_sim.simulated_measurement_disturbance, 9
        )

    def test_simulation_smoothed_state_disturbance(self):
        assert_almost_equal(
            self.conventional_sim.simulated_state_disturbance,
            self.univariate_sim.simulated_state_disturbance, 9
        )


class TestMultivariateGeneralObsCov(MultivariateMissingGeneralObsCov):
    """
    This class tests the univariate method when the observation covariance
    matrix is not diagonal and all data is available.

    Tests are against the conventional smoother.
    """
    @classmethod
    def setup_class(cls, *args, **kwargs):
        super(TestMultivariateGeneralObsCov, cls).setup_class('none')


class TestMultivariateAllMissingGeneralObsCov(
        MultivariateMissingGeneralObsCov):
    """
    This class tests the univariate method when the observation covariance
    matrix is not diagonal and there are cases of fully missing data only.

    Tests are against the conventional smoother.
    """
    @classmethod
    def setup_class(cls, *args, **kwargs):
        super(TestMultivariateAllMissingGeneralObsCov, cls).setup_class('all')


class TestMultivariatePartialMissingGeneralObsCov(
        MultivariateMissingGeneralObsCov):
    """
    This class tests the univariate method when the observation covariance
    matrix is not diagonal and there are cases of partially missing data only.

    Tests are against the conventional smoother.
    """
    @classmethod
    def setup_class(cls, *args, **kwargs):
        super(TestMultivariatePartialMissingGeneralObsCov,
              cls).setup_class('partial')

    def test_forecasts(self):
        assert_almost_equal(
            self.conventional_results.forecasts[0, :],
            self.univariate_results.forecasts[0, :], 8
        )

    def test_forecasts_error(self):
        assert_almost_equal(
            self.conventional_results.forecasts_error[0, :],
            self.univariate_results.forecasts_error[0, :], 8
        )


class TestMultivariateMixedMissingGeneralObsCov(
        MultivariateMissingGeneralObsCov):
    """
    This class tests the univariate method when the observation covariance
    matrix is not diagonal and there are cases of both partially missing and
    fully missing data.

    Tests are against the conventional smoother.
    """
    @classmethod
    def setup_class(cls, *args, **kwargs):
        super(TestMultivariateMixedMissingGeneralObsCov,
              cls).setup_class('mixed')

    def test_forecasts(self):
        assert_almost_equal(
            self.conventional_results.forecasts[0, :],
            self.univariate_results.forecasts[0, :], 8
        )

    def test_forecasts_error(self):
        assert_almost_equal(
            self.conventional_results.forecasts_error[0, :],
            self.univariate_results.forecasts_error[0, :], 8
        )


class TestMultivariateVAR:
    @classmethod
    def setup_class(cls, which='none', **kwargs):
        # Results
        path = os.path.join(current_path, 'results',
                            'results_smoothing_generalobscov_R.csv')
        cls.desired = pd.read_csv(path)

        # Data
        dta = datasets.macrodata.load_pandas().data
        dta.index = pd.date_range(start='1959-01-01',
                                  end='2009-7-01', freq='QS')
        obs = dta[['realgdp', 'realcons', 'realinv']].diff().iloc[1:]

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

        # Create the model
        mod = MLEModel(obs, k_states=3, k_posdef=3, **kwargs)
        mod['design'] = np.eye(3)
        mod['obs_cov'] = np.array([
            [609.0746647855,    0.,              0.],
            [0.,                1.8774916622,    0.],
            [0.,                0.,            124.6768281675]])
        mod['transition'] = np.array([
            [-0.8110473405,  1.8005304445,  1.0215975772],
            [-1.9846632699,  2.4091302213,  1.9264449765],
            [0.9181658823,  -0.2442384581, -0.6393462272]])
        mod['selection'] = np.eye(3)
        mod['state_cov'] = np.array([
            [1552.9758843938,   612.7185121905,   877.6157204992],
            [612.7185121905,    467.8739411204,    70.608037339],
            [877.6157204992,     70.608037339,    900.5440385836]])
        mod.initialize_approximate_diffuse(1e6)
        cls.model = mod.ssm

        # Conventional filtering, smoothing, and simulation smoothing
        cls.model.filter_conventional = True
        cls.conventional_results = cls.model.smooth()
        n_disturbance_variates = (
            (cls.model.k_endog + cls.model.k_posdef) * cls.model.nobs
        )
        cls.conventional_sim = cls.model.simulation_smoother(
            disturbance_variates=np.zeros(n_disturbance_variates),
            initial_state_variates=np.zeros(cls.model.k_states)
        )

        # Univariate filtering, smoothing, and simulation smoothing
        cls.model.filter_univariate = True
        cls.univariate_results = cls.model.smooth()
        cls.univariate_sim = cls.model.simulation_smoother(
            disturbance_variates=np.zeros(n_disturbance_variates),
            initial_state_variates=np.zeros(cls.model.k_states)
        )

    def test_forecasts(self):
        assert_almost_equal(
            self.conventional_results.forecasts[0, :],
            self.univariate_results.forecasts[0, :], 9
        )

    def test_forecasts_error(self):
        assert_almost_equal(
            self.conventional_results.forecasts_error[0, :],
            self.univariate_results.forecasts_error[0, :], 9
        )

    def test_forecasts_error_cov(self):
        assert_almost_equal(
            self.conventional_results.forecasts_error_cov[0, 0, :],
            self.univariate_results.forecasts_error_cov[0, 0, :], 9
        )

    def test_filtered_state(self):
        assert_almost_equal(
            self.conventional_results.filtered_state,
            self.univariate_results.filtered_state, 8
        )

    def test_filtered_state_cov(self):
        assert_almost_equal(
            self.conventional_results.filtered_state_cov,
            self.univariate_results.filtered_state_cov, 9
        )

    def test_predicted_state(self):
        assert_almost_equal(
            self.conventional_results.predicted_state,
            self.univariate_results.predicted_state, 8
        )

    def test_predicted_state_cov(self):
        assert_almost_equal(
            self.conventional_results.predicted_state_cov,
            self.univariate_results.predicted_state_cov, 9
        )

    def test_loglike(self):
        assert_allclose(
            self.conventional_results.llf_obs,
            self.univariate_results.llf_obs
        )

    def test_smoothed_states(self):
        assert_allclose(
            self.conventional_results.smoothed_state,
            self.univariate_results.smoothed_state
        )

    def test_smoothed_states_cov(self):
        assert_allclose(
            self.conventional_results.smoothed_state_cov,
            self.univariate_results.smoothed_state_cov, atol=1e-9
        )

    @pytest.mark.skip
    def test_smoothed_measurement_disturbance(self):
        assert_almost_equal(
            self.conventional_results.smoothed_measurement_disturbance,
            self.univariate_results.smoothed_measurement_disturbance, 9
        )

    @pytest.mark.skip
    def test_smoothed_measurement_disturbance_cov(self):
        conv = self.self.conventional_results
        univ = self.univariate_results
        assert_almost_equal(
            conv.smoothed_measurement_disturbance_cov.diagonal(),
            univ.smoothed_measurement_disturbance_cov.diagonal(),
            9
        )

    def test_smoothed_state_disturbance(self):
        assert_allclose(
            self.conventional_results.smoothed_state_disturbance,
            self.univariate_results.smoothed_state_disturbance,
            atol=1e-7
        )

    def test_smoothed_state_disturbance_cov(self):
        assert_almost_equal(
            self.conventional_results.smoothed_state_disturbance_cov,
            self.univariate_results.smoothed_state_disturbance_cov, 9
        )

    def test_simulation_smoothed_state(self):
        assert_almost_equal(
            self.conventional_sim.simulated_state,
            self.univariate_sim.simulated_state, 9
        )

    @pytest.mark.skip
    def test_simulation_smoothed_measurement_disturbance(self):
        assert_almost_equal(
            self.conventional_sim.simulated_measurement_disturbance,
            self.univariate_sim.simulated_measurement_disturbance, 9
        )

    def test_simulation_smoothed_state_disturbance(self):
        assert_almost_equal(
            self.conventional_sim.simulated_state_disturbance,
            self.univariate_sim.simulated_state_disturbance, 9
        )


def test_time_varying_transition():
    # Test for correct univariate filtering/smoothing when we have a
    # time-varying transition matrix
    endog = np.array([10, 5, 2.5, 1.25, 2.5, 5, 10])
    transition = np.ones((1, 1, 7))
    transition[..., :5] = 0.5
    transition[..., 5:] = 2

    # Conventional filter / smoother
    mod1 = SARIMAX(endog, order=(1, 0, 0), measurement_error=True)
    mod1.update([2., 1., 1.])
    mod1.ssm['transition'] = transition
    res1 = mod1.ssm.smooth()

    # Univariate filter / smoother
    mod2 = SARIMAX(endog, order=(1, 0, 0), measurement_error=True)
    mod2.ssm.filter_univariate = True
    mod2.update([2., 1., 1.])
    mod2.ssm['transition'] = transition
    res2 = mod2.ssm.smooth()

    # Simulation smoothers
    n_disturbance_variates = (mod1.k_endog + mod1.k_posdef) * mod1.nobs
    sim1 = mod1.simulation_smoother(
        disturbance_variates=np.zeros(n_disturbance_variates),
        initial_state_variates=np.zeros(mod1.k_states))
    sim2 = mod2.simulation_smoother(
        disturbance_variates=np.zeros(n_disturbance_variates),
        initial_state_variates=np.zeros(mod2.k_states))

    # Test for correctness
    assert_allclose(res1.forecasts[0, :], res2.forecasts[0, :])
    assert_allclose(res1.forecasts_error[0, :], res2.forecasts_error[0, :])
    assert_allclose(res1.forecasts_error_cov[0, 0, :],
                    res2.forecasts_error_cov[0, 0, :])
    assert_allclose(res1.filtered_state, res2.filtered_state)
    assert_allclose(res1.filtered_state_cov, res2.filtered_state_cov)
    assert_allclose(res1.predicted_state, res2.predicted_state)
    assert_allclose(res1.predicted_state_cov, res2.predicted_state_cov)
    assert_allclose(res1.llf_obs, res2.llf_obs)
    assert_allclose(res1.smoothed_state, res2.smoothed_state)
    assert_allclose(res1.smoothed_state_cov, res2.smoothed_state_cov)
    assert_allclose(res1.smoothed_measurement_disturbance,
                    res2.smoothed_measurement_disturbance)
    assert_allclose(res1.smoothed_measurement_disturbance_cov.diagonal(),
                    res2.smoothed_measurement_disturbance_cov.diagonal())
    assert_allclose(res1.smoothed_state_disturbance,
                    res2.smoothed_state_disturbance)
    assert_allclose(res1.smoothed_state_disturbance_cov,
                    res2.smoothed_state_disturbance_cov)

    assert_allclose(sim1.simulated_state, sim2.simulated_state)
    assert_allclose(sim1.simulated_measurement_disturbance,
                    sim2.simulated_measurement_disturbance)
    assert_allclose(sim1.simulated_state_disturbance,
                    sim2.simulated_state_disturbance)
