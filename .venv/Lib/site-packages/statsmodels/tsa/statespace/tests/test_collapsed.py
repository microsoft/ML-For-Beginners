"""
Tests for collapsed observation vector

These tests cannot be run for the Clark 1989 model since the dimension of
observations (2) is smaller than the number of states (6).

Author: Chad Fulton
License: Simplified-BSD
"""

import numpy as np
import pandas as pd
import pytest
import os

from statsmodels import datasets
from statsmodels.tsa.statespace import dynamic_factor
from statsmodels.tsa.statespace.mlemodel import MLEModel
from statsmodels.tsa.statespace.kalman_filter import (
    FILTER_UNIVARIATE)
from statsmodels.tsa.statespace.kalman_smoother import (
    SMOOTH_CLASSICAL, SMOOTH_ALTERNATIVE,
    SMOOTH_UNIVARIATE)
from statsmodels.tsa.statespace.tests.results import results_kalman_filter
from numpy.testing import assert_equal, assert_allclose

current_path = os.path.dirname(os.path.abspath(__file__))


class Trivariate:
    """
    Tests collapsing three-dimensional observation data to two-dimensional
    """
    @classmethod
    def setup_class(cls, dtype=float, alternate_timing=False, **kwargs):
        cls.results = results_kalman_filter.uc_bi

        # GDP and Unemployment, Quarterly, 1948.1 - 1995.3
        data = pd.DataFrame(
            cls.results['data'],
            index=pd.date_range('1947-01-01', '1995-07-01', freq='QS'),
            columns=['GDP', 'UNEMP']
        )[4:]
        data['GDP'] = np.log(data['GDP'])
        data['UNEMP'] = (data['UNEMP']/100)
        data['X'] = np.exp(data['GDP']) * data['UNEMP']

        k_states = 2
        cls.mlemodel = MLEModel(data, k_states=k_states, **kwargs)
        cls.model = cls.mlemodel.ssm
        if alternate_timing:
            cls.model.timing_init_filtered = True

        # Statespace representation
        cls.model['selection'] = np.eye(cls.model.k_states)

        # Update matrices with test parameters
        cls.model['design'] = np.array([[0.5, 0.2],
                                        [0,   0.8],
                                        [1,  -0.5]])
        cls.model['transition'] = np.array([[0.4, 0.5],
                                            [1,   0]])
        cls.model['obs_cov'] = np.diag([0.2, 1.1, 0.5])
        cls.model['state_cov'] = np.diag([2., 1])

        # Initialization
        cls.model.initialize_approximate_diffuse()

    def test_using_collapsed(self):
        # Test to make sure the results_b actually used a collapsed Kalman
        # filtering approach (i.e. that the flag being set actually caused the
        # filter to not use the conventional filter)

        assert not self.results_a.filter_collapsed
        assert self.results_b.filter_collapsed

        assert self.results_a.collapsed_forecasts is None
        assert self.results_b.collapsed_forecasts is not None

        assert_equal(self.results_a.forecasts.shape[0], 3)
        assert_equal(self.results_b.collapsed_forecasts.shape[0], 2)

    def test_forecasts(self):
        assert_allclose(
            self.results_a.forecasts[0, :],
            self.results_b.forecasts[0, :],
        )

    def test_forecasts_error(self):
        assert_allclose(
            self.results_a.forecasts_error[0, :],
            self.results_b.forecasts_error[0, :]
        )

    def test_forecasts_error_cov(self):
        assert_allclose(
            self.results_a.forecasts_error_cov[0, 0, :],
            self.results_b.forecasts_error_cov[0, 0, :]
        )

    def test_filtered_state(self):
        assert_allclose(
            self.results_a.filtered_state,
            self.results_b.filtered_state
        )

    def test_filtered_state_cov(self):
        assert_allclose(
            self.results_a.filtered_state_cov,
            self.results_b.filtered_state_cov
        )

    def test_predicted_state(self):
        assert_allclose(
            self.results_a.predicted_state,
            self.results_b.predicted_state
        )

    def test_predicted_state_cov(self):
        assert_allclose(
            self.results_a.predicted_state_cov,
            self.results_b.predicted_state_cov
        )

    def test_loglike(self):
        assert_allclose(
            self.results_a.llf_obs,
            self.results_b.llf_obs
        )

    def test_smoothed_states(self):
        assert_allclose(
            self.results_a.smoothed_state,
            self.results_b.smoothed_state
        )

    def test_smoothed_states_cov(self):
        assert_allclose(
            self.results_a.smoothed_state_cov,
            self.results_b.smoothed_state_cov,
            atol=1e-4
        )

    def test_smoothed_states_autocov(self):
        assert_allclose(
            self.results_a.smoothed_state_autocov,
            self.results_b.smoothed_state_autocov
        )

    # Skipped because "measurement" refers to different things; even different
    # dimensions
    @pytest.mark.skip
    def test_smoothed_measurement_disturbance(self):
        assert_allclose(
            self.results_a.smoothed_measurement_disturbance,
            self.results_b.smoothed_measurement_disturbance
        )

    # Skipped because "measurement" refers to different things; even different
    # dimensions
    @pytest.mark.skip
    def test_smoothed_measurement_disturbance_cov(self):
        assert_allclose(
            self.results_a.smoothed_measurement_disturbance_cov,
            self.results_b.smoothed_measurement_disturbance_cov
        )

    def test_smoothed_state_disturbance(self):
        assert_allclose(
            self.results_a.smoothed_state_disturbance,
            self.results_b.smoothed_state_disturbance
        )

    def test_smoothed_state_disturbance_cov(self):
        assert_allclose(
            self.results_a.smoothed_state_disturbance_cov,
            self.results_b.smoothed_state_disturbance_cov
        )

    def test_simulation_smoothed_state(self):
        assert_allclose(
            self.sim_a.simulated_state,
            self.sim_a.simulated_state
        )

    def test_simulation_smoothed_measurement_disturbance(self):
        assert_allclose(
            self.sim_a.simulated_measurement_disturbance,
            self.sim_a.simulated_measurement_disturbance
        )

    def test_simulation_smoothed_state_disturbance(self):
        assert_allclose(
            self.sim_a.simulated_state_disturbance,
            self.sim_a.simulated_state_disturbance
        )


class TestTrivariateConventional(Trivariate):

    @classmethod
    def setup_class(cls, dtype=float, **kwargs):
        super(TestTrivariateConventional, cls).setup_class(dtype, **kwargs)

        nobs = cls.model.nobs
        k_endog = cls.model.k_endog
        k_posdef = cls.model.k_posdef

        # Collapsed filtering, smoothing, and simulation smoothing
        cls.model.filter_conventional = True
        cls.model.filter_collapsed = True
        cls.results_b = cls.model.smooth()
        cls.sim_b = cls.model.simulation_smoother(
            measurement_disturbance_variates=np.zeros(nobs * k_endog),
            state_disturbance_variates=np.zeros(nobs * k_posdef),
            initial_state_variates=np.zeros(cls.model.k_states)
        )

        # Conventional filtering, smoothing, and simulation smoothing
        cls.model.filter_collapsed = False
        cls.results_a = cls.model.smooth()
        cls.sim_a = cls.model.simulation_smoother(
            measurement_disturbance_variates=np.zeros(nobs * k_endog),
            state_disturbance_variates=np.zeros(nobs * k_posdef),
            initial_state_variates=np.zeros(cls.model.k_states)
        )


class TestTrivariateConventionalAlternate(TestTrivariateConventional):
    @classmethod
    def setup_class(cls, *args, **kwargs):
        super(TestTrivariateConventionalAlternate, cls).setup_class(
            alternate_timing=True, *args, **kwargs)

    def test_using_alterate(self):
        assert self.model._kalman_filter.filter_timing == 1


class TestTrivariateConventionalPartialMissing(Trivariate):
    @classmethod
    def setup_class(cls, dtype=float, **kwargs):
        super(TestTrivariateConventionalPartialMissing, cls).setup_class(
            dtype, **kwargs)
        nobs = cls.model.nobs
        k_endog = cls.model.k_endog
        k_posdef = cls.model.k_posdef

        # Set partially missing data
        cls.model.endog[:2, 10:180] = np.nan

        # Collapsed filtering, smoothing, and simulation smoothing
        cls.model.filter_conventional = True
        cls.model.filter_collapsed = True
        cls.results_b = cls.model.smooth()
        cls.sim_b = cls.model.simulation_smoother(
            measurement_disturbance_variates=np.zeros(nobs * k_endog),
            state_disturbance_variates=np.zeros(nobs * k_posdef),
            initial_state_variates=np.zeros(cls.model.k_states)
        )

        # Conventional filtering, smoothing, and simulation smoothing
        cls.model.filter_collapsed = False
        cls.results_a = cls.model.smooth()
        cls.sim_a = cls.model.simulation_smoother(
            measurement_disturbance_variates=np.zeros(nobs * k_endog),
            state_disturbance_variates=np.zeros(nobs * k_posdef),
            initial_state_variates=np.zeros(cls.model.k_states)
        )


class TestTrivariateConventionalPartialMissingAlternate(
        TestTrivariateConventionalPartialMissing):
    @classmethod
    def setup_class(cls, *args, **kwargs):
        super(TestTrivariateConventionalPartialMissingAlternate,
              cls).setup_class(alternate_timing=True, *args, **kwargs)

    def test_using_alterate(self):
        assert self.model._kalman_filter.filter_timing == 1


class TestTrivariateConventionalAllMissing(Trivariate):
    @classmethod
    def setup_class(cls, dtype=float, **kwargs):
        super(TestTrivariateConventionalAllMissing, cls).setup_class(
            dtype, **kwargs)
        nobs = cls.model.nobs
        k_endog = cls.model.k_endog
        k_posdef = cls.model.k_posdef

        # Set partially missing data
        cls.model.endog[:, 10:180] = np.nan

        # Collapsed filtering, smoothing, and simulation smoothing
        cls.model.filter_conventional = True
        cls.model.filter_collapsed = True
        cls.results_b = cls.model.smooth()
        cls.sim_b = cls.model.simulation_smoother(
            measurement_disturbance_variates=np.zeros(nobs * k_endog),
            state_disturbance_variates=np.zeros(nobs * k_posdef),
            initial_state_variates=np.zeros(cls.model.k_states)
        )

        # Conventional filtering, smoothing, and simulation smoothing
        cls.model.filter_collapsed = False
        cls.results_a = cls.model.smooth()
        cls.sim_a = cls.model.simulation_smoother(
            measurement_disturbance_variates=np.zeros(nobs * k_endog),
            state_disturbance_variates=np.zeros(nobs * k_posdef),
            initial_state_variates=np.zeros(cls.model.k_states)
        )


class TestTrivariateConventionalAllMissingAlternate(
        TestTrivariateConventionalAllMissing):
    @classmethod
    def setup_class(cls, *args, **kwargs):
        super(TestTrivariateConventionalAllMissingAlternate, cls).setup_class(
            alternate_timing=True, *args, **kwargs)

    def test_using_alterate(self):
        assert self.model._kalman_filter.filter_timing == 1


class TestTrivariateUnivariate(Trivariate):
    @classmethod
    def setup_class(cls, dtype=float, **kwargs):
        super(TestTrivariateUnivariate, cls).setup_class(dtype, **kwargs)
        nobs = cls.model.nobs
        k_endog = cls.model.k_endog
        k_posdef = cls.model.k_posdef

        # Collapsed filtering, smoothing, and simulation smoothing
        cls.model.filter_univariate = True
        cls.model.filter_collapsed = True
        cls.results_b = cls.model.smooth()
        cls.sim_b = cls.model.simulation_smoother(
            measurement_disturbance_variates=np.zeros(nobs * k_endog),
            state_disturbance_variates=np.zeros(nobs * k_posdef),
            initial_state_variates=np.zeros(cls.model.k_states)
        )

        # Univariate filtering, smoothing, and simulation smoothing
        cls.model.filter_collapsed = False
        cls.results_a = cls.model.smooth()
        cls.sim_a = cls.model.simulation_smoother(
            measurement_disturbance_variates=np.zeros(nobs * k_endog),
            state_disturbance_variates=np.zeros(nobs * k_posdef),
            initial_state_variates=np.zeros(cls.model.k_states)
        )


class TestTrivariateUnivariateAlternate(TestTrivariateUnivariate):
    @classmethod
    def setup_class(cls, *args, **kwargs):
        super(TestTrivariateUnivariateAlternate, cls).setup_class(
            alternate_timing=True, *args, **kwargs)

    def test_using_alterate(self):
        assert self.model._kalman_filter.filter_timing == 1


class TestTrivariateUnivariatePartialMissing(Trivariate):
    @classmethod
    def setup_class(cls, dtype=float, **kwargs):
        super(TestTrivariateUnivariatePartialMissing, cls).setup_class(
            dtype, **kwargs)
        nobs = cls.model.nobs
        k_endog = cls.model.k_endog
        k_posdef = cls.model.k_posdef

        # Set partially missing data
        cls.model.endog[:2, 10:180] = np.nan

        # Collapsed filtering, smoothing, and simulation smoothing
        cls.model.filter_univariate = True
        cls.model.filter_collapsed = True
        cls.results_b = cls.model.smooth()
        cls.sim_b = cls.model.simulation_smoother(
            measurement_disturbance_variates=np.zeros(nobs * k_endog),
            state_disturbance_variates=np.zeros(nobs * k_posdef),
            initial_state_variates=np.zeros(cls.model.k_states)
        )

        # Univariate filtering, smoothing, and simulation smoothing
        cls.model.filter_collapsed = False
        cls.results_a = cls.model.smooth()
        cls.sim_a = cls.model.simulation_smoother(
            measurement_disturbance_variates=np.zeros(nobs * k_endog),
            state_disturbance_variates=np.zeros(nobs * k_posdef),
            initial_state_variates=np.zeros(cls.model.k_states)
        )


class TestTrivariateUnivariatePartialMissingAlternate(
        TestTrivariateUnivariatePartialMissing):
    @classmethod
    def setup_class(cls, *args, **kwargs):
        super(TestTrivariateUnivariatePartialMissingAlternate,
              cls).setup_class(alternate_timing=True, *args, **kwargs)

    def test_using_alterate(self):
        assert self.model._kalman_filter.filter_timing == 1


class TestTrivariateUnivariateAllMissing(Trivariate):
    @classmethod
    def setup_class(cls, dtype=float, **kwargs):
        super(TestTrivariateUnivariateAllMissing, cls).setup_class(
            dtype, **kwargs)
        nobs = cls.model.nobs
        k_endog = cls.model.k_endog
        k_posdef = cls.model.k_posdef

        # Set partially missing data
        cls.model.endog[:, 10:180] = np.nan

        # Univariate filtering, smoothing, and simulation smoothing
        cls.model.filter_univariate = True
        cls.model.filter_collapsed = True
        cls.results_b = cls.model.smooth()
        cls.sim_b = cls.model.simulation_smoother(
            measurement_disturbance_variates=np.zeros(nobs * k_endog),
            state_disturbance_variates=np.zeros(nobs * k_posdef),
            initial_state_variates=np.zeros(cls.model.k_states)
        )

        # Conventional filtering, smoothing, and simulation smoothing
        cls.model.filter_collapsed = False
        cls.results_a = cls.model.smooth()
        cls.sim_a = cls.model.simulation_smoother(
            measurement_disturbance_variates=np.zeros(nobs * k_endog),
            state_disturbance_variates=np.zeros(nobs * k_posdef),
            initial_state_variates=np.zeros(cls.model.k_states)
        )


class TestTrivariateUnivariateAllMissingAlternate(
        TestTrivariateUnivariateAllMissing):
    @classmethod
    def setup_class(cls, *args, **kwargs):
        super(TestTrivariateUnivariateAllMissingAlternate, cls).setup_class(
            alternate_timing=True, *args, **kwargs)

    def test_using_alterate(self):
        assert self.model._kalman_filter.filter_timing == 1


class TestDFM:
    @classmethod
    def create_model(cls, obs, **kwargs):
        # Create the model with typical state space
        mod = MLEModel(obs, k_states=2, k_posdef=2, **kwargs)
        mod['design'] = np.array([[-32.47143586, 17.33779024],
                                  [-7.40264169, 1.69279859],
                                  [-209.04702853, 125.2879374]])
        mod['obs_cov'] = np.diag(
            np.array([0.0622668, 1.95666886, 58.37473642]))
        mod['transition'] = np.array([[0.29935707, 0.33289005],
                                      [-0.7639868, 1.2844237]])
        mod['selection'] = np.eye(2)
        mod['state_cov'] = np.array([[1.2, -0.25],
                                     [-0.25, 1.1]])
        mod.initialize_approximate_diffuse(1e6)
        return mod

    @classmethod
    def collapse(cls, obs, **kwargs):
        mod = cls.create_model(obs, **kwargs)
        mod.smooth([], return_ssm=True)

        _ss = mod.ssm._statespace
        out = np.zeros((mod.nobs, mod.k_states))
        for t in range(mod.nobs):
            _ss.seek(t, mod.ssm.filter_univariate, 1)
            out[t] = np.array(_ss.collapse_obs)
        return out

    @classmethod
    def setup_class(cls, which='mixed', *args, **kwargs):
        # Data
        dta = datasets.macrodata.load_pandas().data
        dta.index = pd.date_range(start='1959-01-01',
                                  end='2009-7-01', freq='QS')
        levels = dta[['realgdp', 'realcons', 'realinv']]
        obs = np.log(levels).diff().iloc[1:] * 400

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

        mod = cls.create_model(obs, **kwargs)
        cls.model = mod.ssm

        nobs = cls.model.nobs
        k_endog = cls.model.k_endog
        k_posdef = cls.model.k_posdef

        np.random.seed(1234)
        mdv = np.random.normal(size=nobs * k_endog)
        sdv = np.random.normal(size=nobs * k_posdef)
        isv = np.random.normal(size=cls.model.k_states)

        # Collapsed filtering, smoothing, and simulation smoothing
        cls.model.filter_collapsed = True
        cls.results_b = cls.model.smooth()
        cls.sim_b = cls.model.simulation_smoother()
        cls.sim_b.simulate(measurement_disturbance_variates=mdv,
                           state_disturbance_variates=sdv,
                           initial_state_variates=isv)

        # Conventional filtering, smoothing, and simulation smoothing
        cls.model.filter_collapsed = False
        cls.results_a = cls.model.smooth()
        cls.sim_a = cls.model.simulation_smoother()
        cls.sim_a.simulate(measurement_disturbance_variates=mdv,
                           state_disturbance_variates=sdv,
                           initial_state_variates=isv)

        # Create the model with augmented state space
        kwargs.pop('filter_collapsed', None)
        mod = MLEModel(obs, k_states=4, k_posdef=2, **kwargs)
        mod['design', :3, :2] = np.array([[-32.47143586, 17.33779024],
                                          [-7.40264169, 1.69279859],
                                          [-209.04702853, 125.2879374]])
        mod['obs_cov'] = np.diag(
            np.array([0.0622668, 1.95666886, 58.37473642]))
        mod['transition', :2, :2] = np.array([[0.29935707, 0.33289005],
                                              [-0.7639868, 1.2844237]])
        mod['transition', 2:, :2] = np.eye(2)
        mod['selection', :2, :2] = np.eye(2)
        mod['state_cov'] = np.array([[1.2, -0.25],
                                     [-0.25, 1.1]])

        mod.initialize_approximate_diffuse(1e6)
        cls.augmented_model = mod.ssm
        cls.augmented_results = mod.ssm.smooth()

    def test_using_collapsed(self):
        # Test to make sure the results_b actually used a collapsed Kalman
        # filtering approach (i.e. that the flag being set actually caused the
        # filter to not use the conventional filter)

        assert not self.results_a.filter_collapsed
        assert self.results_b.filter_collapsed

        assert self.results_a.collapsed_forecasts is None
        assert self.results_b.collapsed_forecasts is not None

        assert_equal(self.results_a.forecasts.shape[0], 3)
        assert_equal(self.results_b.collapsed_forecasts.shape[0], 2)

    def test_forecasts(self):
        assert_allclose(
            self.results_a.forecasts[0, :],
            self.results_b.forecasts[0, :],
        )

    def test_forecasts_error(self):
        assert_allclose(
            self.results_a.forecasts_error[0, :],
            self.results_b.forecasts_error[0, :]
        )

    def test_forecasts_error_cov(self):
        assert_allclose(
            self.results_a.forecasts_error_cov[0, 0, :],
            self.results_b.forecasts_error_cov[0, 0, :]
        )

    def test_filtered_state(self):
        assert_allclose(
            self.results_a.filtered_state,
            self.results_b.filtered_state
        )

    def test_filtered_state_cov(self):
        assert_allclose(
            self.results_a.filtered_state_cov,
            self.results_b.filtered_state_cov
        )

    def test_predicted_state(self):
        assert_allclose(
            self.results_a.predicted_state,
            self.results_b.predicted_state
        )

    def test_predicted_state_cov(self):
        assert_allclose(
            self.results_a.predicted_state_cov,
            self.results_b.predicted_state_cov
        )

    def test_loglike(self):
        assert_allclose(
            self.results_a.llf_obs,
            self.results_b.llf_obs
        )

    def test_smoothed_states(self):
        assert_allclose(
            self.results_a.smoothed_state,
            self.results_b.smoothed_state
        )

    def test_smoothed_states_cov(self):
        assert_allclose(
            self.results_a.smoothed_state_cov,
            self.results_b.smoothed_state_cov,
            atol=1e-4
        )

    def test_smoothed_states_autocov(self):
        # Cov(\alpha_{t+1}, \alpha_t)
        # Test collapsed against non-collapsed
        assert_allclose(
            self.results_a.smoothed_state_autocov,
            self.results_b.smoothed_state_autocov
        )

        # Directly test using the augmented model
        # Initialization makes these two methods slightly different for the
        # first few observations
        assert_allclose(self.results_a.smoothed_state_autocov[:, :, 0:5],
                        self.augmented_results.smoothed_state_cov[:2, 2:, 1:6],
                        atol=1e-4)
        assert_allclose(self.results_a.smoothed_state_autocov[:, :, 5:-1],
                        self.augmented_results.smoothed_state_cov[:2, 2:, 6:],
                        atol=1e-7)

    # Skipped because "measurement" refers to different things; even different
    # dimensions
    @pytest.mark.skip
    def test_smoothed_measurement_disturbance(self):
        assert_allclose(
            self.results_a.smoothed_measurement_disturbance,
            self.results_b.smoothed_measurement_disturbance
        )

    # Skipped because "measurement" refers to different things; even different
    # dimensions
    @pytest.mark.skip
    def test_smoothed_measurement_disturbance_cov(self):
        assert_allclose(
            self.results_a.smoothed_measurement_disturbance_cov,
            self.results_b.smoothed_measurement_disturbance_cov
        )

    def test_smoothed_state_disturbance(self):
        assert_allclose(
            self.results_a.smoothed_state_disturbance,
            self.results_b.smoothed_state_disturbance
        )

    def test_smoothed_state_disturbance_cov(self):
        assert_allclose(
            self.results_a.smoothed_state_disturbance_cov,
            self.results_b.smoothed_state_disturbance_cov
        )

    def test_simulation_smoothed_state(self):
        assert_allclose(
            self.sim_a.simulated_state,
            self.sim_b.simulated_state
        )

    # Skipped because "measurement" refers to different things; even different
    # dimensions
    @pytest.mark.skip
    def test_simulation_smoothed_measurement_disturbance(self):
        assert_allclose(
            self.sim_a.simulated_measurement_disturbance,
            self.sim_b.simulated_measurement_disturbance
        )

    def test_simulation_smoothed_state_disturbance(self):
        assert_allclose(
            self.sim_a.simulated_state_disturbance,
            self.sim_b.simulated_state_disturbance
        )


class TestDFMClassicalSmoothing(TestDFM):
    @classmethod
    def setup_class(cls, *args, **kwargs):
        super(TestDFMClassicalSmoothing, cls).setup_class(
            smooth_method=SMOOTH_CLASSICAL, *args, **kwargs)

    def test_smooth_method(self):
        assert_equal(self.model.smooth_method, SMOOTH_CLASSICAL)
        assert_equal(self.model._kalman_smoother.smooth_method,
                     SMOOTH_CLASSICAL)
        assert_equal(self.model._kalman_smoother._smooth_method,
                     SMOOTH_CLASSICAL)


class TestDFMUnivariateSmoothing(TestDFM):
    @classmethod
    def setup_class(cls, *args, **kwargs):
        super(TestDFMUnivariateSmoothing, cls).setup_class(
            filter_method=FILTER_UNIVARIATE, *args, **kwargs)

    def test_smooth_method(self):
        assert_equal(self.model.smooth_method, 0)
        assert_equal(self.model._kalman_smoother.smooth_method, 0)
        assert_equal(self.model._kalman_smoother._smooth_method,
                     SMOOTH_UNIVARIATE)


class TestDFMAlternativeSmoothing(TestDFM):
    @classmethod
    def setup_class(cls, *args, **kwargs):
        super(TestDFMAlternativeSmoothing, cls).setup_class(
            smooth_method=SMOOTH_ALTERNATIVE, **kwargs)

    def test_smooth_method(self):
        assert_equal(self.model.smooth_method, SMOOTH_ALTERNATIVE)
        assert_equal(self.model._kalman_smoother.smooth_method,
                     SMOOTH_ALTERNATIVE)
        assert_equal(self.model._kalman_smoother._smooth_method,
                     SMOOTH_ALTERNATIVE)


class TestDFMMeasurementDisturbance(TestDFM):
    @classmethod
    def setup_class(cls, *args, **kwargs):
        super(TestDFMMeasurementDisturbance, cls).setup_class(
            smooth_method=SMOOTH_CLASSICAL, which='none', **kwargs)

    def test_smoothed_state_disturbance(self):
        assert_allclose(
            self.results_a.smoothed_state_disturbance,
            self.results_b.smoothed_state_disturbance, atol=1e-7)

    def test_smoothed_measurement_disturbance(self):
        assert_allclose(
            self.collapse(self.results_a.smoothed_measurement_disturbance.T).T,
            self.results_b.smoothed_measurement_disturbance, atol=1e-7)

    def test_simulation_smoothed_measurement_disturbance(self):
        assert_allclose(
            self.collapse(self.sim_a.simulated_measurement_disturbance.T),
            self.sim_b.simulated_measurement_disturbance.T, atol=1e-7)

    def test_simulation_smoothed_state_disturbance(self):
        assert_allclose(
            self.sim_a.simulated_state_disturbance,
            self.sim_b.simulated_state_disturbance, atol=1e-7)


def test_dfm_missing(reset_randomstate):
    # This test is not captured by the TestTrivariate and TestDFM tests above
    # because it has k_states = 1
    endog = np.random.normal(size=(100, 3))
    endog[0, :1] = np.nan

    mod = dynamic_factor.DynamicFactor(endog, k_factors=1, factor_order=1)
    mod.ssm.filter_collapsed = True
    res = mod.smooth(mod.start_params)
    mod.ssm.filter_collapsed = False
    res2 = mod.smooth(mod.start_params)

    assert_allclose(res.llf, res2.llf)
