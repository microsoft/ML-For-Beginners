"""
Tests for _representation and _kalman_filter modules

Author: Chad Fulton
License: Simplified-BSD

References
----------

Kim, Chang-Jin, and Charles R. Nelson. 1999.
"State-Space Models with Regime Switching:
Classical and Gibbs-Sampling Approaches with Applications".
MIT Press Books. The MIT Press.

Hamilton, James D. 1994.
Time Series Analysis.
Princeton, N.J.: Princeton University Press.
"""
import copy
import pickle

import numpy as np
import pandas as pd
import os
import pytest

from scipy.linalg.blas import find_best_blas_type
from scipy.linalg import solve_discrete_lyapunov
from statsmodels.tsa.statespace.kalman_filter import (
    MEMORY_NO_FORECAST, MEMORY_NO_PREDICTED, MEMORY_CONSERVE)
from statsmodels.tsa.statespace.mlemodel import MLEModel
from statsmodels.tsa.statespace import _representation, _kalman_filter
from .results import results_kalman_filter
from numpy.testing import assert_almost_equal, assert_allclose

prefix_statespace_map = {
    's': _representation.sStatespace, 'd': _representation.dStatespace,
    'c': _representation.cStatespace, 'z': _representation.zStatespace
}
prefix_kalman_filter_map = {
    's': _kalman_filter.sKalmanFilter, 'd': _kalman_filter.dKalmanFilter,
    'c': _kalman_filter.cKalmanFilter, 'z': _kalman_filter.zKalmanFilter
}

current_path = os.path.dirname(os.path.abspath(__file__))


class Clark1987:
    """
    Clark's (1987) univariate unobserved components model of real GDP (as
    presented in Kim and Nelson, 1999)

    Test data produced using GAUSS code described in Kim and Nelson (1999) and
    found at http://econ.korea.ac.kr/~cjkim/SSMARKOV.htm

    See `results.results_kalman_filter` for more information.
    """
    @classmethod
    def setup_class(cls, dtype=float, conserve_memory=0, loglikelihood_burn=0):
        cls.true = results_kalman_filter.uc_uni
        cls.true_states = pd.DataFrame(cls.true['states'])

        # GDP, Quarterly, 1947.1 - 1995.3
        data = pd.DataFrame(
            cls.true['data'],
            index=pd.date_range('1947-01-01', '1995-07-01', freq='QS'),
            columns=['GDP']
        )
        data['lgdp'] = np.log(data['GDP'])

        # Parameters
        cls.conserve_memory = conserve_memory
        cls.loglikelihood_burn = loglikelihood_burn

        # Observed data
        cls.obs = np.array(data['lgdp'], ndmin=2, dtype=dtype, order="F")

        # Measurement equation
        cls.k_endog = k_endog = 1  # dimension of observed data
        # design matrix
        cls.design = np.zeros((k_endog, 4, 1), dtype=dtype, order="F")
        cls.design[:, :, 0] = [1, 1, 0, 0]
        # observation intercept
        cls.obs_intercept = np.zeros((k_endog, 1), dtype=dtype, order="F")
        # observation covariance matrix
        cls.obs_cov = np.zeros((k_endog, k_endog, 1), dtype=dtype, order="F")

        # Transition equation
        cls.k_states = k_states = 4  # dimension of state space
        # transition matrix
        cls.transition = np.zeros((k_states, k_states, 1),
                                  dtype=dtype, order="F")
        cls.transition[([0, 0, 1, 1, 2, 3],
                        [0, 3, 1, 2, 1, 3],
                        [0, 0, 0, 0, 0, 0])] = [1, 1, 0, 0, 1, 1]
        # state intercept
        cls.state_intercept = np.zeros((k_states, 1), dtype=dtype, order="F")
        # selection matrix
        cls.selection = np.asfortranarray(np.eye(k_states)[:, :, None],
                                          dtype=dtype)
        # state covariance matrix
        cls.state_cov = np.zeros((k_states, k_states, 1),
                                 dtype=dtype, order="F")

        # Initialization: Diffuse priors
        cls.initial_state = np.zeros((k_states,), dtype=dtype, order="F")
        cls.initial_state_cov = np.asfortranarray(np.eye(k_states)*100,
                                                  dtype=dtype)

        # Update matrices with given parameters
        (sigma_v, sigma_e, sigma_w, phi_1, phi_2) = np.array(
            cls.true['parameters'], dtype=dtype
        )
        cls.transition[([1, 1], [1, 2], [0, 0])] = [phi_1, phi_2]
        cls.state_cov[
            np.diag_indices(k_states)+(np.zeros(k_states, dtype=int),)] = [
            sigma_v**2, sigma_e**2, 0, sigma_w**2
        ]

        # Initialization: modification
        # Due to the difference in the way Kim and Nelson (1999) and Durbin
        # and Koopman (2012) define the order of the Kalman filter routines,
        # we need to modify the initial state covariance matrix to match
        # Kim and Nelson's results, since the *Statespace models follow Durbin
        # and Koopman.
        cls.initial_state_cov = np.asfortranarray(
            np.dot(
                np.dot(cls.transition[:, :, 0], cls.initial_state_cov),
                cls.transition[:, :, 0].T
            )
        )

    @classmethod
    def init_filter(cls):
        # Use the appropriate Statespace model
        prefix = find_best_blas_type((cls.obs,))
        klass = prefix_statespace_map[prefix[0]]

        # Instantiate the statespace model
        model = klass(
            cls.obs, cls.design, cls.obs_intercept, cls.obs_cov,
            cls.transition, cls.state_intercept, cls.selection,
            cls.state_cov
        )
        model.initialize_known(cls.initial_state, cls.initial_state_cov)

        # Initialize the appropriate Kalman filter
        klass = prefix_kalman_filter_map[prefix[0]]
        kfilter = klass(model, conserve_memory=cls.conserve_memory,
                        loglikelihood_burn=cls.loglikelihood_burn)

        return model, kfilter

    @classmethod
    def run_filter(cls):
        # Filter the data
        cls.filter()

        # Get results
        return {
            'loglike': lambda burn: np.sum(cls.filter.loglikelihood[burn:]),
            'state': np.array(cls.filter.filtered_state),
        }

    def test_loglike(self):
        assert_almost_equal(
            self.result['loglike'](self.true['start']), self.true['loglike'], 5
        )

    def test_filtered_state(self):
        assert_almost_equal(
            self.result['state'][0][self.true['start']:],
            self.true_states.iloc[:, 0], 4
        )
        assert_almost_equal(
            self.result['state'][1][self.true['start']:],
            self.true_states.iloc[:, 1], 4
        )
        assert_almost_equal(
            self.result['state'][3][self.true['start']:],
            self.true_states.iloc[:, 2], 4
        )

    def test_pickled_filter(self):
        pickled = pickle.loads(pickle.dumps(self.filter))
        #  Run the filters
        self.filter()
        pickled()

        assert id(filter) != id(pickled)
        assert_allclose(np.array(self.filter.filtered_state),
                        np.array(pickled.filtered_state))
        assert_allclose(np.array(self.filter.loglikelihood),
                        np.array(pickled.loglikelihood))

    def test_copied_filter(self):
        copied = copy.deepcopy(self.filter)
        #  Run the filters
        self.filter()
        copied()

        assert id(filter) != id(copied)
        assert_allclose(np.array(self.filter.filtered_state),
                        np.array(copied.filtered_state))

        assert_allclose(np.array(self.filter.loglikelihood),
                        np.array(copied.loglikelihood))


class TestClark1987Single(Clark1987):
    """
    Basic single precision test for the loglikelihood and filtered states.
    """
    @classmethod
    def setup_class(cls):
        # TODO: Can we be more specific?  How can a contributor help?
        pytest.skip('Not implemented')
        super(TestClark1987Single, cls).setup_class(
            dtype=np.float32, conserve_memory=0
        )
        cls.model, cls.filter = cls.init_filter()
        cls.result = cls.run_filter()

    def test_loglike(self):
        assert_allclose(
            self.result['loglike'](self.true['start']), self.true['loglike'],
            rtol=1e-3
        )

    def test_filtered_state(self):
        assert_allclose(
            self.result['state'][0][self.true['start']:],
            self.true_states.iloc[:, 0],
            atol=1e-2
        )
        assert_allclose(
            self.result['state'][1][self.true['start']:],
            self.true_states.iloc[:, 1],
            atol=1e-2
        )
        assert_allclose(
            self.result['state'][3][self.true['start']:],
            self.true_states.iloc[:, 2],
            atol=1e-2
        )


class TestClark1987Double(Clark1987):
    """
    Basic double precision test for the loglikelihood and filtered states.
    """
    @classmethod
    def setup_class(cls):
        super(TestClark1987Double, cls).setup_class(
            dtype=float, conserve_memory=0
        )
        cls.model, cls.filter = cls.init_filter()
        cls.result = cls.run_filter()


class TestClark1987SingleComplex(Clark1987):
    """
    Basic single precision complex test for the loglikelihood and filtered
    states.
    """
    @classmethod
    def setup_class(cls):
        # TODO: Can we be more specific?  How can a contributor help?
        pytest.skip('Not implemented')
        super(TestClark1987SingleComplex, cls).setup_class(
            dtype=np.complex64, conserve_memory=0
        )
        cls.model, cls.filter = cls.init_filter()
        cls.result = cls.run_filter()

    def test_loglike(self):
        assert_allclose(
            self.result['loglike'](self.true['start']), self.true['loglike'],
            rtol=1e-3
        )

    def test_filtered_state(self):
        assert_allclose(
            self.result['state'][0][self.true['start']:],
            self.true_states.iloc[:, 0],
            atol=1e-2
        )
        assert_allclose(
            self.result['state'][1][self.true['start']:],
            self.true_states.iloc[:, 1],
            atol=1e-2
        )
        assert_allclose(
            self.result['state'][3][self.true['start']:],
            self.true_states.iloc[:, 2],
            atol=1e-2
        )


class TestClark1987DoubleComplex(Clark1987):
    """
    Basic double precision complex test for the loglikelihood and filtered
    states.
    """
    @classmethod
    def setup_class(cls):
        super(TestClark1987DoubleComplex, cls).setup_class(
            dtype=complex, conserve_memory=0
        )
        cls.model, cls.filter = cls.init_filter()
        cls.result = cls.run_filter()


class TestClark1987Conserve(Clark1987):
    """
    Memory conservation test for the loglikelihood and filtered states.
    """
    @classmethod
    def setup_class(cls):
        super(TestClark1987Conserve, cls).setup_class(
            dtype=float,
            conserve_memory=MEMORY_NO_FORECAST | MEMORY_NO_PREDICTED
        )
        cls.model, cls.filter = cls.init_filter()
        cls.result = cls.run_filter()


class Clark1987Forecast(Clark1987):
    """
    Forecasting test for the loglikelihood and filtered states.
    """
    @classmethod
    def setup_class(cls, dtype=float, nforecast=100, conserve_memory=0):
        super(Clark1987Forecast, cls).setup_class(
            dtype, conserve_memory
        )
        cls.nforecast = nforecast

        # Add missing observations to the end (to forecast)
        cls._obs = cls.obs
        cls.obs = np.array(np.r_[cls.obs[0, :], [np.nan]*nforecast],
                           ndmin=2, dtype=dtype, order="F")

    def test_filtered_state(self):
        assert_almost_equal(
            self.result['state'][0][self.true['start']:-self.nforecast],
            self.true_states.iloc[:, 0], 4
        )
        assert_almost_equal(
            self.result['state'][1][self.true['start']:-self.nforecast],
            self.true_states.iloc[:, 1], 4
        )
        assert_almost_equal(
            self.result['state'][3][self.true['start']:-self.nforecast],
            self.true_states.iloc[:, 2], 4
        )


class TestClark1987ForecastDouble(Clark1987Forecast):
    """
    Basic double forecasting test for the loglikelihood and filtered states.
    """
    @classmethod
    def setup_class(cls):
        super(TestClark1987ForecastDouble, cls).setup_class()
        cls.model, cls.filter = cls.init_filter()
        cls.result = cls.run_filter()


class TestClark1987ForecastDoubleComplex(Clark1987Forecast):
    """
    Basic double complex forecasting test for the loglikelihood and filtered
    states.
    """
    @classmethod
    def setup_class(cls):
        super(TestClark1987ForecastDoubleComplex, cls).setup_class(
            dtype=complex
        )
        cls.model, cls.filter = cls.init_filter()
        cls.result = cls.run_filter()


class TestClark1987ForecastConserve(Clark1987Forecast):
    """
    Memory conservation forecasting test for the loglikelihood and filtered
    states.
    """
    @classmethod
    def setup_class(cls):
        super(TestClark1987ForecastConserve, cls).setup_class(
            dtype=float,
            conserve_memory=MEMORY_NO_FORECAST | MEMORY_NO_PREDICTED
        )
        cls.model, cls.filter = cls.init_filter()
        cls.result = cls.run_filter()


class TestClark1987ConserveAll(Clark1987):
    """
    Memory conservation forecasting test for the loglikelihood and filtered
    states.
    """
    @classmethod
    def setup_class(cls):
        super(TestClark1987ConserveAll, cls).setup_class(
            dtype=float, conserve_memory=MEMORY_CONSERVE
        )
        cls.loglikelihood_burn = cls.true['start']
        cls.model, cls.filter = cls.init_filter()
        cls.result = cls.run_filter()

    def test_loglike(self):
        assert_almost_equal(
            self.result['loglike'](0), self.true['loglike'], 5
        )

    def test_filtered_state(self):
        end = self.true_states.shape[0]
        assert_almost_equal(
            self.result['state'][0][-1],
            self.true_states.iloc[end-1, 0], 4
        )
        assert_almost_equal(
            self.result['state'][1][-1],
            self.true_states.iloc[end-1, 1], 4
        )


class Clark1989:
    """
    Clark's (1989) bivariate unobserved components model of real GDP (as
    presented in Kim and Nelson, 1999)

    Tests two-dimensional observation data.

    Test data produced using GAUSS code described in Kim and Nelson (1999) and
    found at http://econ.korea.ac.kr/~cjkim/SSMARKOV.htm

    See `results.results_kalman_filter` for more information.
    """
    @classmethod
    def setup_class(cls, dtype=float, conserve_memory=0, loglikelihood_burn=0):
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

        # Observed data
        cls.obs = np.array(data, ndmin=2, dtype=dtype, order="C").T

        # Parameters
        cls.k_endog = k_endog = 2  # dimension of observed data
        cls.k_states = k_states = 6  # dimension of state space
        cls.conserve_memory = conserve_memory
        cls.loglikelihood_burn = loglikelihood_burn

        # Measurement equation

        # design matrix
        cls.design = np.zeros((k_endog, k_states, 1), dtype=dtype, order="F")
        cls.design[:, :, 0] = [[1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1]]
        # observation intercept
        cls.obs_intercept = np.zeros((k_endog, 1), dtype=dtype, order="F")
        # observation covariance matrix
        cls.obs_cov = np.zeros((k_endog, k_endog, 1), dtype=dtype, order="F")

        # Transition equation

        # transition matrix
        cls.transition = np.zeros((k_states, k_states, 1),
                                  dtype=dtype, order="F")
        cls.transition[([0, 0, 1, 1, 2, 3, 4, 5],
                        [0, 4, 1, 2, 1, 2, 4, 5],
                        [0, 0, 0, 0, 0, 0, 0, 0])] = [1, 1, 0, 0, 1, 1, 1, 1]
        # state intercept
        cls.state_intercept = np.zeros((k_states, 1), dtype=dtype, order="F")
        # selection matrix
        cls.selection = np.asfortranarray(np.eye(k_states)[:, :, None],
                                          dtype=dtype)
        # state covariance matrix
        cls.state_cov = np.zeros((k_states, k_states, 1),
                                 dtype=dtype, order="F")

        # Initialization: Diffuse priors
        cls.initial_state = np.zeros((k_states,), dtype=dtype)
        cls.initial_state_cov = np.asfortranarray(np.eye(k_states)*100,
                                                  dtype=dtype)

        # Update matrices with given parameters
        (sigma_v, sigma_e, sigma_w, sigma_vl, sigma_ec,
         phi_1, phi_2, alpha_1, alpha_2, alpha_3) = np.array(
            cls.true['parameters'], dtype=dtype
        )
        cls.design[([1, 1, 1], [1, 2, 3], [0, 0, 0])] = [
            alpha_1, alpha_2, alpha_3
        ]
        cls.transition[([1, 1], [1, 2], [0, 0])] = [phi_1, phi_2]
        cls.obs_cov[1, 1, 0] = sigma_ec**2
        cls.state_cov[
            np.diag_indices(k_states)+(np.zeros(k_states, dtype=int),)] = [
            sigma_v**2, sigma_e**2, 0, 0, sigma_w**2, sigma_vl**2
        ]

        # Initialization: modification
        # Due to the difference in the way Kim and Nelson (1999) and Drubin
        # and Koopman (2012) define the order of the Kalman filter routines,
        # we need to modify the initial state covariance matrix to match
        # Kim and Nelson's results, since the *Statespace models follow Durbin
        # and Koopman.
        cls.initial_state_cov = np.asfortranarray(
            np.dot(
                np.dot(cls.transition[:, :, 0], cls.initial_state_cov),
                cls.transition[:, :, 0].T
            )
        )

    @classmethod
    def init_filter(cls):
        # Use the appropriate Statespace model
        prefix = find_best_blas_type((cls.obs,))
        klass = prefix_statespace_map[prefix[0]]

        # Instantiate the statespace model
        model = klass(
            cls.obs, cls.design, cls.obs_intercept, cls.obs_cov,
            cls.transition, cls.state_intercept, cls.selection,
            cls.state_cov
        )
        model.initialize_known(cls.initial_state, cls.initial_state_cov)

        # Initialize the appropriate Kalman filter
        klass = prefix_kalman_filter_map[prefix[0]]
        kfilter = klass(model, conserve_memory=cls.conserve_memory,
                        loglikelihood_burn=cls.loglikelihood_burn)

        return model, kfilter

    @classmethod
    def run_filter(cls):
        # Filter the data
        cls.filter()

        # Get results
        return {
            'loglike': lambda burn: np.sum(cls.filter.loglikelihood[burn:]),
            'state': np.array(cls.filter.filtered_state),
        }

    def test_loglike(self):
        assert_almost_equal(
            # self.result['loglike'](self.true['start']),
            self.result['loglike'](0),
            self.true['loglike'], 2
        )

    def test_filtered_state(self):
        assert_almost_equal(
            self.result['state'][0][self.true['start']:],
            self.true_states.iloc[:, 0], 4
        )
        assert_almost_equal(
            self.result['state'][1][self.true['start']:],
            self.true_states.iloc[:, 1], 4
        )
        assert_almost_equal(
            self.result['state'][4][self.true['start']:],
            self.true_states.iloc[:, 2], 4
        )
        assert_almost_equal(
            self.result['state'][5][self.true['start']:],
            self.true_states.iloc[:, 3], 4
        )


class TestClark1989(Clark1989):
    """
    Basic double precision test for the loglikelihood and filtered
    states with two-dimensional observation vector.
    """
    @classmethod
    def setup_class(cls):
        super(TestClark1989, cls).setup_class(dtype=float, conserve_memory=0)
        cls.model, cls.filter = cls.init_filter()
        cls.result = cls.run_filter()


class TestClark1989Conserve(Clark1989):
    """
    Memory conservation test for the loglikelihood and filtered states with
    two-dimensional observation vector.
    """
    @classmethod
    def setup_class(cls):
        super(TestClark1989Conserve, cls).setup_class(
            dtype=float,
            conserve_memory=MEMORY_NO_FORECAST | MEMORY_NO_PREDICTED
        )
        cls.model, cls.filter = cls.init_filter()
        cls.result = cls.run_filter()


class Clark1989Forecast(Clark1989):
    """
    Memory conservation test for the loglikelihood and filtered states with
    two-dimensional observation vector.
    """
    @classmethod
    def setup_class(cls, dtype=float, nforecast=100, conserve_memory=0):
        super(Clark1989Forecast, cls).setup_class(dtype, conserve_memory)
        cls.nforecast = nforecast

        # Add missing observations to the end (to forecast)
        cls._obs = cls.obs
        cls.obs = np.array(
            np.c_[
                cls._obs,
                np.r_[[np.nan, np.nan]*nforecast].reshape(2, nforecast)
            ],
            ndmin=2, dtype=dtype, order="F"
        )

        cls.model, cls.filter = cls.init_filter()
        cls.result = cls.run_filter()

    def test_filtered_state(self):
        assert_almost_equal(
            self.result['state'][0][self.true['start']:-self.nforecast],
            self.true_states.iloc[:, 0], 4
        )
        assert_almost_equal(
            self.result['state'][1][self.true['start']:-self.nforecast],
            self.true_states.iloc[:, 1], 4
        )
        assert_almost_equal(
            self.result['state'][4][self.true['start']:-self.nforecast],
            self.true_states.iloc[:, 2], 4
        )
        assert_almost_equal(
            self.result['state'][5][self.true['start']:-self.nforecast],
            self.true_states.iloc[:, 3], 4
        )


class TestClark1989ForecastDouble(Clark1989Forecast):
    """
    Basic double forecasting test for the loglikelihood and filtered states.
    """
    @classmethod
    def setup_class(cls):
        super(TestClark1989ForecastDouble, cls).setup_class()
        cls.model, cls.filter = cls.init_filter()
        cls.result = cls.run_filter()


class TestClark1989ForecastDoubleComplex(Clark1989Forecast):
    """
    Basic double complex forecasting test for the loglikelihood and filtered
    states.
    """
    @classmethod
    def setup_class(cls):
        super(TestClark1989ForecastDoubleComplex, cls).setup_class(
            dtype=complex
        )
        cls.model, cls.filter = cls.init_filter()
        cls.result = cls.run_filter()


class TestClark1989ForecastConserve(Clark1989Forecast):
    """
    Memory conservation forecasting test for the loglikelihood and filtered
    states.
    """
    @classmethod
    def setup_class(cls):
        super(TestClark1989ForecastConserve, cls).setup_class(
            dtype=float,
            conserve_memory=MEMORY_NO_FORECAST | MEMORY_NO_PREDICTED
        )
        cls.model, cls.filter = cls.init_filter()
        cls.result = cls.run_filter()


class TestClark1989ConserveAll(Clark1989):
    """
    Memory conservation forecasting test for the loglikelihood and filtered
    states.
    """
    @classmethod
    def setup_class(cls):
        super(TestClark1989ConserveAll, cls).setup_class(
            dtype=float, conserve_memory=MEMORY_CONSERVE,
        )
        # cls.loglikelihood_burn = cls.true['start']
        cls.loglikelihood_burn = 0
        cls.model, cls.filter = cls.init_filter()
        cls.result = cls.run_filter()

    def test_loglike(self):
        assert_almost_equal(
            self.result['loglike'](0), self.true['loglike'], 2
        )

    def test_filtered_state(self):
        end = self.true_states.shape[0]
        assert_almost_equal(
            self.result['state'][0][-1],
            self.true_states.iloc[end-1, 0], 4
        )
        assert_almost_equal(
            self.result['state'][1][-1],
            self.true_states.iloc[end-1, 1], 4
        )
        assert_almost_equal(
            self.result['state'][4][-1],
            self.true_states.iloc[end-1, 2], 4
        )
        assert_almost_equal(
            self.result['state'][5][-1],
            self.true_states.iloc[end-1, 3], 4
        )


def check_stationary_initialization_1dim(dtype=float):
    endog = np.zeros(10, dtype=dtype)

    # 1-dimensional example
    mod = MLEModel(endog, k_states=1, k_posdef=1)
    mod.ssm.initialize_stationary()
    intercept = np.array([2.3], dtype=dtype)
    phi = np.diag([0.9]).astype(dtype)
    sigma2 = np.diag([1.3]).astype(dtype)

    mod['state_intercept'] = intercept
    mod['transition'] = phi
    mod['selection'] = np.eye(1).astype(dtype)
    mod['state_cov'] = sigma2

    mod.ssm._initialize_filter()
    mod.ssm._initialize_state()

    _statespace = mod.ssm._statespace
    initial_state = np.array(_statespace.initial_state)
    initial_state_cov = np.array(_statespace.initial_state_cov)
    # precision reductions only required for float complex case

    # mean = intercept + phi * mean
    # intercept = (1 - phi) * mean
    # mean = intercept / (1 - phi)
    assert_allclose(initial_state, intercept / (1 - phi[0, 0]))
    desired = np.linalg.inv(np.eye(1) - phi).dot(intercept)
    assert_allclose(initial_state, desired)
    # var = phi**2 var + sigma2
    # var = sigma2 / (1 - phi**2)
    assert_allclose(initial_state_cov, sigma2 / (1 - phi**2))
    assert_allclose(initial_state_cov, solve_discrete_lyapunov(phi, sigma2))


def check_stationary_initialization_2dim(dtype=float):
    endog = np.zeros(10, dtype=dtype)
    # 2-dimensional example
    mod = MLEModel(endog, k_states=2, k_posdef=2)
    mod.ssm.initialize_stationary()
    intercept = np.array([2.3, -10.2], dtype=dtype)
    phi = np.array([[0.8, 0.1],
                    [-0.2, 0.7]], dtype=dtype)
    sigma2 = np.array([[1.4, -0.2],
                       [-0.2, 4.5]], dtype=dtype)

    mod['state_intercept'] = intercept
    mod['transition'] = phi
    mod['selection'] = np.eye(2).astype(dtype)
    mod['state_cov'] = sigma2

    mod.ssm._initialize_filter()
    mod.ssm._initialize_state()

    _statespace = mod.ssm._statespace
    initial_state = np.array(_statespace.initial_state)
    initial_state_cov = np.array(_statespace.initial_state_cov)

    desired = np.linalg.solve(np.eye(2).astype(dtype) - phi, intercept)
    assert_allclose(initial_state, desired)
    desired = solve_discrete_lyapunov(phi, sigma2)
    # precision reductions only required for single precision float / complex
    assert_allclose(initial_state_cov, desired, atol=1e-5)


def test_stationary_initialization():
    check_stationary_initialization_1dim(np.float32)
    check_stationary_initialization_1dim(np.float64)
    check_stationary_initialization_1dim(np.complex64)
    check_stationary_initialization_1dim(np.complex128)

    check_stationary_initialization_2dim(np.float32)
    check_stationary_initialization_2dim(np.float64)
    check_stationary_initialization_2dim(np.complex64)
    check_stationary_initialization_2dim(np.complex128)
