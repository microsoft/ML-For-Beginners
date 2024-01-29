# -*- coding: utf-8 -*-
"""
Vector Autoregressive Moving Average with eXogenous regressors model

Author: Chad Fulton
License: Simplified-BSD
"""

import contextlib
from warnings import warn

import pandas as pd
import numpy as np

from statsmodels.compat.pandas import Appender
from statsmodels.tools.tools import Bunch
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.vector_ar import var_model
import statsmodels.base.wrapper as wrap
from statsmodels.tools.sm_exceptions import EstimationWarning

from .kalman_filter import INVERT_UNIVARIATE, SOLVE_LU
from .mlemodel import MLEModel, MLEResults, MLEResultsWrapper
from .initialization import Initialization
from .tools import (
    is_invertible, concat, prepare_exog,
    constrain_stationary_multivariate, unconstrain_stationary_multivariate,
    prepare_trend_spec, prepare_trend_data
)


class VARMAX(MLEModel):
    r"""
    Vector Autoregressive Moving Average with eXogenous regressors model

    Parameters
    ----------
    endog : array_like
        The observed time-series process :math:`y`, , shaped nobs x k_endog.
    exog : array_like, optional
        Array of exogenous regressors, shaped nobs x k.
    order : iterable
        The (p,q) order of the model for the number of AR and MA parameters to
        use.
    trend : str{'n','c','t','ct'} or iterable, optional
        Parameter controlling the deterministic trend polynomial :math:`A(t)`.
        Can be specified as a string where 'c' indicates a constant (i.e. a
        degree zero component of the trend polynomial), 't' indicates a
        linear trend with time, and 'ct' is both. Can also be specified as an
        iterable defining the non-zero polynomial exponents to include, in
        increasing order. For example, `[1,1,0,1]` denotes
        :math:`a + bt + ct^3`. Default is a constant trend component.
    error_cov_type : {'diagonal', 'unstructured'}, optional
        The structure of the covariance matrix of the error term, where
        "unstructured" puts no restrictions on the matrix and "diagonal"
        requires it to be a diagonal matrix (uncorrelated errors). Default is
        "unstructured".
    measurement_error : bool, optional
        Whether or not to assume the endogenous observations `endog` were
        measured with error. Default is False.
    enforce_stationarity : bool, optional
        Whether or not to transform the AR parameters to enforce stationarity
        in the autoregressive component of the model. Default is True.
    enforce_invertibility : bool, optional
        Whether or not to transform the MA parameters to enforce invertibility
        in the moving average component of the model. Default is True.
    trend_offset : int, optional
        The offset at which to start time trend values. Default is 1, so that
        if `trend='t'` the trend is equal to 1, 2, ..., nobs. Typically is only
        set when the model created by extending a previous dataset.
    **kwargs
        Keyword arguments may be used to provide default values for state space
        matrices or for Kalman filtering options. See `Representation`, and
        `KalmanFilter` for more details.

    Attributes
    ----------
    order : iterable
        The (p,q) order of the model for the number of AR and MA parameters to
        use.
    trend : str{'n','c','t','ct'} or iterable
        Parameter controlling the deterministic trend polynomial :math:`A(t)`.
        Can be specified as a string where 'c' indicates a constant (i.e. a
        degree zero component of the trend polynomial), 't' indicates a
        linear trend with time, and 'ct' is both. Can also be specified as an
        iterable defining the non-zero polynomial exponents to include, in
        increasing order. For example, `[1,1,0,1]` denotes
        :math:`a + bt + ct^3`.
    error_cov_type : {'diagonal', 'unstructured'}, optional
        The structure of the covariance matrix of the error term, where
        "unstructured" puts no restrictions on the matrix and "diagonal"
        requires it to be a diagonal matrix (uncorrelated errors). Default is
        "unstructured".
    measurement_error : bool, optional
        Whether or not to assume the endogenous observations `endog` were
        measured with error. Default is False.
    enforce_stationarity : bool, optional
        Whether or not to transform the AR parameters to enforce stationarity
        in the autoregressive component of the model. Default is True.
    enforce_invertibility : bool, optional
        Whether or not to transform the MA parameters to enforce invertibility
        in the moving average component of the model. Default is True.

    Notes
    -----
    Generically, the VARMAX model is specified (see for example chapter 18 of
    [1]_):

    .. math::

        y_t = A(t) + A_1 y_{t-1} + \dots + A_p y_{t-p} + B x_t + \epsilon_t +
        M_1 \epsilon_{t-1} + \dots M_q \epsilon_{t-q}

    where :math:`\epsilon_t \sim N(0, \Omega)`, and where :math:`y_t` is a
    `k_endog x 1` vector. Additionally, this model allows considering the case
    where the variables are measured with error.

    Note that in the full VARMA(p,q) case there is a fundamental identification
    problem in that the coefficient matrices :math:`\{A_i, M_j\}` are not
    generally unique, meaning that for a given time series process there may
    be multiple sets of matrices that equivalently represent it. See Chapter 12
    of [1]_ for more information. Although this class can be used to estimate
    VARMA(p,q) models, a warning is issued to remind users that no steps have
    been taken to ensure identification in this case.

    References
    ----------
    .. [1] Lütkepohl, Helmut. 2007.
       New Introduction to Multiple Time Series Analysis.
       Berlin: Springer.
    """

    def __init__(self, endog, exog=None, order=(1, 0), trend='c',
                 error_cov_type='unstructured', measurement_error=False,
                 enforce_stationarity=True, enforce_invertibility=True,
                 trend_offset=1, **kwargs):

        # Model parameters
        self.error_cov_type = error_cov_type
        self.measurement_error = measurement_error
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility

        # Save the given orders
        self.order = order

        # Model orders
        self.k_ar = int(order[0])
        self.k_ma = int(order[1])

        # Check for valid model
        if error_cov_type not in ['diagonal', 'unstructured']:
            raise ValueError('Invalid error covariance matrix type'
                             ' specification.')
        if self.k_ar == 0 and self.k_ma == 0:
            raise ValueError('Invalid VARMAX(p,q) specification; at least one'
                             ' p,q must be greater than zero.')

        # Warn for VARMA model
        if self.k_ar > 0 and self.k_ma > 0:
            warn('Estimation of VARMA(p,q) models is not generically robust,'
                 ' due especially to identification issues.',
                 EstimationWarning)

        # Trend
        self.trend = trend
        self.trend_offset = trend_offset
        self.polynomial_trend, self.k_trend = prepare_trend_spec(self.trend)
        self._trend_is_const = (self.polynomial_trend.size == 1 and
                                self.polynomial_trend[0] == 1)

        # Exogenous data
        (self.k_exog, exog) = prepare_exog(exog)

        # Note: at some point in the future might add state regression, as in
        # SARIMAX.
        self.mle_regression = self.k_exog > 0

        # We need to have an array or pandas at this point
        if not _is_using_pandas(endog, None):
            endog = np.asanyarray(endog)

        # Model order
        # Used internally in various places
        _min_k_ar = max(self.k_ar, 1)
        self._k_order = _min_k_ar + self.k_ma

        # Number of states
        k_endog = endog.shape[1]
        k_posdef = k_endog
        k_states = k_endog * self._k_order

        # By default, initialize as stationary
        kwargs.setdefault('initialization', 'stationary')

        # By default, use LU decomposition
        kwargs.setdefault('inversion_method', INVERT_UNIVARIATE | SOLVE_LU)

        # Initialize the state space model
        super(VARMAX, self).__init__(
            endog, exog=exog, k_states=k_states, k_posdef=k_posdef, **kwargs
        )

        # Set as time-varying model if we have time-trend or exog
        if self.k_exog > 0 or (self.k_trend > 0 and not self._trend_is_const):
            self.ssm._time_invariant = False

        # Initialize the parameters
        self.parameters = {}
        self.parameters['trend'] = self.k_endog * self.k_trend
        self.parameters['ar'] = self.k_endog**2 * self.k_ar
        self.parameters['ma'] = self.k_endog**2 * self.k_ma
        self.parameters['regression'] = self.k_endog * self.k_exog
        if self.error_cov_type == 'diagonal':
            self.parameters['state_cov'] = self.k_endog
        # These parameters fill in a lower-triangular matrix which is then
        # dotted with itself to get a positive definite matrix.
        elif self.error_cov_type == 'unstructured':
            self.parameters['state_cov'] = (
                int(self.k_endog * (self.k_endog + 1) / 2)
            )
        self.parameters['obs_cov'] = self.k_endog * self.measurement_error
        self.k_params = sum(self.parameters.values())

        # Initialize trend data: we create trend data with one more observation
        # than we actually have, to make it easier to insert the appropriate
        # trend component into the final state intercept.
        trend_data = prepare_trend_data(
            self.polynomial_trend, self.k_trend, self.nobs + 1,
            offset=self.trend_offset)
        self._trend_data = trend_data[:-1]
        self._final_trend = trend_data[-1:]

        # Initialize known elements of the state space matrices

        # If we have exog effects, then the state intercept needs to be
        # time-varying
        if (self.k_trend > 0 and not self._trend_is_const) or self.k_exog > 0:
            self.ssm['state_intercept'] = np.zeros((self.k_states, self.nobs))
            # self.ssm['obs_intercept'] = np.zeros((self.k_endog, self.nobs))

        # The design matrix is just an identity for the first k_endog states
        idx = np.diag_indices(self.k_endog)
        self.ssm[('design',) + idx] = 1

        # The transition matrix is described in four blocks, where the upper
        # left block is in companion form with the autoregressive coefficient
        # matrices (so it is shaped k_endog * k_ar x k_endog * k_ar) ...
        if self.k_ar > 0:
            idx = np.diag_indices((self.k_ar - 1) * self.k_endog)
            idx = idx[0] + self.k_endog, idx[1]
            self.ssm[('transition',) + idx] = 1
        # ... and the  lower right block is in companion form with zeros as the
        # coefficient matrices (it is shaped k_endog * k_ma x k_endog * k_ma).
        idx = np.diag_indices((self.k_ma - 1) * self.k_endog)
        idx = (idx[0] + (_min_k_ar + 1) * self.k_endog,
               idx[1] + _min_k_ar * self.k_endog)
        self.ssm[('transition',) + idx] = 1

        # The selection matrix is described in two blocks, where the upper
        # block selects the all k_posdef errors in the first k_endog rows
        # (the upper block is shaped k_endog * k_ar x k) and the lower block
        # also selects all k_posdef errors in the first k_endog rows (the lower
        # block is shaped k_endog * k_ma x k).
        idx = np.diag_indices(self.k_endog)
        self.ssm[('selection',) + idx] = 1
        idx = idx[0] + _min_k_ar * self.k_endog, idx[1]
        if self.k_ma > 0:
            self.ssm[('selection',) + idx] = 1

        # Cache some indices
        if self._trend_is_const and self.k_exog == 0:
            self._idx_state_intercept = np.s_['state_intercept', :k_endog, :]
        elif self.k_trend > 0 or self.k_exog > 0:
            self._idx_state_intercept = np.s_['state_intercept', :k_endog, :-1]
        if self.k_ar > 0:
            self._idx_transition = np.s_['transition', :k_endog, :]
        else:
            self._idx_transition = np.s_['transition', :k_endog, k_endog:]
        if self.error_cov_type == 'diagonal':
            self._idx_state_cov = (
                ('state_cov',) + np.diag_indices(self.k_endog))
        elif self.error_cov_type == 'unstructured':
            self._idx_lower_state_cov = np.tril_indices(self.k_endog)
        if self.measurement_error:
            self._idx_obs_cov = ('obs_cov',) + np.diag_indices(self.k_endog)

        # Cache some slices
        def _slice(key, offset):
            length = self.parameters[key]
            param_slice = np.s_[offset:offset + length]
            offset += length
            return param_slice, offset

        offset = 0
        self._params_trend, offset = _slice('trend', offset)
        self._params_ar, offset = _slice('ar', offset)
        self._params_ma, offset = _slice('ma', offset)
        self._params_regression, offset = _slice('regression', offset)
        self._params_state_cov, offset = _slice('state_cov', offset)
        self._params_obs_cov, offset = _slice('obs_cov', offset)

        # Variable holding optional final `exog`
        # (note: self._final_trend was set earlier)
        self._final_exog = None

        # Update _init_keys attached by super
        self._init_keys += ['order', 'trend', 'error_cov_type',
                            'measurement_error', 'enforce_stationarity',
                            'enforce_invertibility'] + list(kwargs.keys())

    def clone(self, endog, exog=None, **kwargs):
        return self._clone_from_init_kwds(endog, exog=exog, **kwargs)

    @property
    def _res_classes(self):
        return {'fit': (VARMAXResults, VARMAXResultsWrapper)}

    @property
    def start_params(self):
        params = np.zeros(self.k_params, dtype=np.float64)

        # A. Run a multivariate regression to get beta estimates
        endog = pd.DataFrame(self.endog.copy())
        endog = endog.interpolate()
        endog = np.require(endog.bfill(), requirements="W")
        exog = None
        if self.k_trend > 0 and self.k_exog > 0:
            exog = np.c_[self._trend_data, self.exog]
        elif self.k_trend > 0:
            exog = self._trend_data
        elif self.k_exog > 0:
            exog = self.exog

        # Although the Kalman filter can deal with missing values in endog,
        # conditional sum of squares cannot
        if np.any(np.isnan(endog)):
            mask = ~np.any(np.isnan(endog), axis=1)
            endog = endog[mask]
            if exog is not None:
                exog = exog[mask]

        # Regression and trend effects via OLS
        trend_params = np.zeros(0)
        exog_params = np.zeros(0)
        if self.k_trend > 0 or self.k_exog > 0:
            trendexog_params = np.linalg.pinv(exog).dot(endog)
            endog -= np.dot(exog, trendexog_params)
            if self.k_trend > 0:
                trend_params = trendexog_params[:self.k_trend].T
            if self.k_endog > 0:
                exog_params = trendexog_params[self.k_trend:].T

        # B. Run a VAR model on endog to get trend, AR parameters
        ar_params = []
        k_ar = self.k_ar if self.k_ar > 0 else 1
        mod_ar = var_model.VAR(endog)
        res_ar = mod_ar.fit(maxlags=k_ar, ic=None, trend='n')
        if self.k_ar > 0:
            ar_params = np.array(res_ar.params).T.ravel()
        endog = res_ar.resid

        # Test for stationarity
        if self.k_ar > 0 and self.enforce_stationarity:
            coefficient_matrices = (
                ar_params.reshape(
                    self.k_endog * self.k_ar, self.k_endog
                ).T
            ).reshape(self.k_endog, self.k_endog, self.k_ar).T

            stationary = is_invertible([1] + list(-coefficient_matrices))

            if not stationary:
                warn('Non-stationary starting autoregressive parameters'
                     ' found. Using zeros as starting parameters.')
                ar_params *= 0

        # C. Run a VAR model on the residuals to get MA parameters
        ma_params = []
        if self.k_ma > 0:
            mod_ma = var_model.VAR(endog)
            res_ma = mod_ma.fit(maxlags=self.k_ma, ic=None, trend='n')
            ma_params = np.array(res_ma.params.T).ravel()

            # Test for invertibility
            if self.enforce_invertibility:
                coefficient_matrices = (
                    ma_params.reshape(
                        self.k_endog * self.k_ma, self.k_endog
                    ).T
                ).reshape(self.k_endog, self.k_endog, self.k_ma).T

                invertible = is_invertible([1] + list(-coefficient_matrices))

                if not invertible:
                    warn('Non-stationary starting moving-average parameters'
                         ' found. Using zeros as starting parameters.')
                    ma_params *= 0

        # Transform trend / exog params from mean form to intercept form
        if self.k_ar > 0 and (self.k_trend > 0 or self.mle_regression):
            coefficient_matrices = (
                ar_params.reshape(
                    self.k_endog * self.k_ar, self.k_endog
                ).T
            ).reshape(self.k_endog, self.k_endog, self.k_ar).T

            tmp = np.eye(self.k_endog) - np.sum(coefficient_matrices, axis=0)

            if self.k_trend > 0:
                trend_params = np.dot(tmp, trend_params)
            if self.mle_regression > 0:
                exog_params = np.dot(tmp, exog_params)

        # 1. Intercept terms
        if self.k_trend > 0:
            params[self._params_trend] = trend_params.ravel()

        # 2. AR terms
        if self.k_ar > 0:
            params[self._params_ar] = ar_params

        # 3. MA terms
        if self.k_ma > 0:
            params[self._params_ma] = ma_params

        # 4. Regression terms
        if self.mle_regression:
            params[self._params_regression] = exog_params.ravel()

        # 5. State covariance terms
        if self.error_cov_type == 'diagonal':
            params[self._params_state_cov] = res_ar.sigma_u.diagonal()
        elif self.error_cov_type == 'unstructured':
            cov_factor = np.linalg.cholesky(res_ar.sigma_u)
            params[self._params_state_cov] = (
                cov_factor[self._idx_lower_state_cov].ravel())

        # 5. Measurement error variance terms
        if self.measurement_error:
            if self.k_ma > 0:
                params[self._params_obs_cov] = res_ma.sigma_u.diagonal()
            else:
                params[self._params_obs_cov] = res_ar.sigma_u.diagonal()

        return params

    @property
    def param_names(self):
        param_names = []
        endog_names = self.endog_names
        if not isinstance(self.endog_names, list):
            endog_names = [endog_names]

        # 1. Intercept terms
        if self.k_trend > 0:
            for j in range(self.k_endog):
                for i in self.polynomial_trend.nonzero()[0]:
                    if i == 0:
                        param_names += ['intercept.%s' % endog_names[j]]
                    elif i == 1:
                        param_names += ['drift.%s' % endog_names[j]]
                    else:
                        param_names += ['trend.%d.%s' % (i, endog_names[j])]

        # 2. AR terms
        param_names += [
            'L%d.%s.%s' % (i+1, endog_names[k], endog_names[j])
            for j in range(self.k_endog)
            for i in range(self.k_ar)
            for k in range(self.k_endog)
        ]

        # 3. MA terms
        param_names += [
            'L%d.e(%s).%s' % (i+1, endog_names[k], endog_names[j])
            for j in range(self.k_endog)
            for i in range(self.k_ma)
            for k in range(self.k_endog)
        ]

        # 4. Regression terms
        param_names += [
            'beta.%s.%s' % (self.exog_names[j], endog_names[i])
            for i in range(self.k_endog)
            for j in range(self.k_exog)
        ]

        # 5. State covariance terms
        if self.error_cov_type == 'diagonal':
            param_names += [
                'sigma2.%s' % endog_names[i]
                for i in range(self.k_endog)
            ]
        elif self.error_cov_type == 'unstructured':
            param_names += [
                ('sqrt.var.%s' % endog_names[i] if i == j else
                 'sqrt.cov.%s.%s' % (endog_names[j], endog_names[i]))
                for i in range(self.k_endog)
                for j in range(i+1)
            ]

        # 5. Measurement error variance terms
        if self.measurement_error:
            param_names += [
                'measurement_variance.%s' % endog_names[i]
                for i in range(self.k_endog)
            ]

        return param_names

    def transform_params(self, unconstrained):
        """
        Transform unconstrained parameters used by the optimizer to constrained
        parameters used in likelihood evaluation

        Parameters
        ----------
        unconstrained : array_like
            Array of unconstrained parameters used by the optimizer, to be
            transformed.

        Returns
        -------
        constrained : array_like
            Array of constrained parameters which may be used in likelihood
            evaluation.

        Notes
        -----
        Constrains the factor transition to be stationary and variances to be
        positive.
        """
        unconstrained = np.array(unconstrained, ndmin=1)
        constrained = np.zeros(unconstrained.shape, dtype=unconstrained.dtype)

        # 1. Intercept terms: nothing to do
        constrained[self._params_trend] = unconstrained[self._params_trend]

        # 2. AR terms: optionally force to be stationary
        if self.k_ar > 0 and self.enforce_stationarity:
            # Create the state covariance matrix
            if self.error_cov_type == 'diagonal':
                state_cov = np.diag(unconstrained[self._params_state_cov]**2)
            elif self.error_cov_type == 'unstructured':
                state_cov_lower = np.zeros(self.ssm['state_cov'].shape,
                                           dtype=unconstrained.dtype)
                state_cov_lower[self._idx_lower_state_cov] = (
                    unconstrained[self._params_state_cov])
                state_cov = np.dot(state_cov_lower, state_cov_lower.T)

            # Transform the parameters
            coefficients = unconstrained[self._params_ar].reshape(
                self.k_endog, self.k_endog * self.k_ar)
            coefficient_matrices, variance = (
                constrain_stationary_multivariate(coefficients, state_cov))
            constrained[self._params_ar] = coefficient_matrices.ravel()
        else:
            constrained[self._params_ar] = unconstrained[self._params_ar]

        # 3. MA terms: optionally force to be invertible
        if self.k_ma > 0 and self.enforce_invertibility:
            # Transform the parameters, using an identity variance matrix
            state_cov = np.eye(self.k_endog, dtype=unconstrained.dtype)
            coefficients = unconstrained[self._params_ma].reshape(
                self.k_endog, self.k_endog * self.k_ma)
            coefficient_matrices, variance = (
                constrain_stationary_multivariate(coefficients, state_cov))
            constrained[self._params_ma] = coefficient_matrices.ravel()
        else:
            constrained[self._params_ma] = unconstrained[self._params_ma]

        # 4. Regression terms: nothing to do
        constrained[self._params_regression] = (
            unconstrained[self._params_regression])

        # 5. State covariance terms
        # If we have variances, force them to be positive
        if self.error_cov_type == 'diagonal':
            constrained[self._params_state_cov] = (
                unconstrained[self._params_state_cov]**2)
        # Otherwise, nothing needs to be done
        elif self.error_cov_type == 'unstructured':
            constrained[self._params_state_cov] = (
                unconstrained[self._params_state_cov])

        # 5. Measurement error variance terms
        if self.measurement_error:
            # Force these to be positive
            constrained[self._params_obs_cov] = (
                unconstrained[self._params_obs_cov]**2)

        return constrained

    def untransform_params(self, constrained):
        """
        Transform constrained parameters used in likelihood evaluation
        to unconstrained parameters used by the optimizer.

        Parameters
        ----------
        constrained : array_like
            Array of constrained parameters used in likelihood evaluation, to
            be transformed.

        Returns
        -------
        unconstrained : array_like
            Array of unconstrained parameters used by the optimizer.
        """
        constrained = np.array(constrained, ndmin=1)
        unconstrained = np.zeros(constrained.shape, dtype=constrained.dtype)

        # 1. Intercept terms: nothing to do
        unconstrained[self._params_trend] = constrained[self._params_trend]

        # 2. AR terms: optionally were forced to be stationary
        if self.k_ar > 0 and self.enforce_stationarity:
            # Create the state covariance matrix
            if self.error_cov_type == 'diagonal':
                state_cov = np.diag(constrained[self._params_state_cov])
            elif self.error_cov_type == 'unstructured':
                state_cov_lower = np.zeros(self.ssm['state_cov'].shape,
                                           dtype=constrained.dtype)
                state_cov_lower[self._idx_lower_state_cov] = (
                    constrained[self._params_state_cov])
                state_cov = np.dot(state_cov_lower, state_cov_lower.T)

            # Transform the parameters
            coefficients = constrained[self._params_ar].reshape(
                self.k_endog, self.k_endog * self.k_ar)
            unconstrained_matrices, variance = (
                unconstrain_stationary_multivariate(coefficients, state_cov))
            unconstrained[self._params_ar] = unconstrained_matrices.ravel()
        else:
            unconstrained[self._params_ar] = constrained[self._params_ar]

        # 3. MA terms: optionally were forced to be invertible
        if self.k_ma > 0 and self.enforce_invertibility:
            # Transform the parameters, using an identity variance matrix
            state_cov = np.eye(self.k_endog, dtype=constrained.dtype)
            coefficients = constrained[self._params_ma].reshape(
                self.k_endog, self.k_endog * self.k_ma)
            unconstrained_matrices, variance = (
                unconstrain_stationary_multivariate(coefficients, state_cov))
            unconstrained[self._params_ma] = unconstrained_matrices.ravel()
        else:
            unconstrained[self._params_ma] = constrained[self._params_ma]

        # 4. Regression terms: nothing to do
        unconstrained[self._params_regression] = (
            constrained[self._params_regression])

        # 5. State covariance terms
        # If we have variances, then these were forced to be positive
        if self.error_cov_type == 'diagonal':
            unconstrained[self._params_state_cov] = (
                constrained[self._params_state_cov]**0.5)
        # Otherwise, nothing needs to be done
        elif self.error_cov_type == 'unstructured':
            unconstrained[self._params_state_cov] = (
                constrained[self._params_state_cov])

        # 5. Measurement error variance terms
        if self.measurement_error:
            # These were forced to be positive
            unconstrained[self._params_obs_cov] = (
                constrained[self._params_obs_cov]**0.5)

        return unconstrained

    def _validate_can_fix_params(self, param_names):
        super(VARMAX, self)._validate_can_fix_params(param_names)

        ix = np.cumsum(list(self.parameters.values()))[:-1]
        (_, ar_names, ma_names, _, _, _) = [
            arr.tolist() for arr in np.array_split(self.param_names, ix)]

        if self.enforce_stationarity and self.k_ar > 0:
            if self.k_endog > 1 or self.k_ar > 1:
                fix_all = param_names.issuperset(ar_names)
                fix_any = (
                    len(param_names.intersection(ar_names)) > 0)
                if fix_any and not fix_all:
                    raise ValueError(
                        'Cannot fix individual autoregressive parameters'
                        ' when `enforce_stationarity=True`. In this case,'
                        ' must either fix all autoregressive parameters or'
                        ' none.')
        if self.enforce_invertibility and self.k_ma > 0:
            if self.k_endog or self.k_ma > 1:
                fix_all = param_names.issuperset(ma_names)
                fix_any = (
                    len(param_names.intersection(ma_names)) > 0)
                if fix_any and not fix_all:
                    raise ValueError(
                        'Cannot fix individual moving average parameters'
                        ' when `enforce_invertibility=True`. In this case,'
                        ' must either fix all moving average parameters or'
                        ' none.')

    def update(self, params, transformed=True, includes_fixed=False,
               complex_step=False):
        params = self.handle_params(params, transformed=transformed,
                                    includes_fixed=includes_fixed)

        # 1. State intercept
        # - Exog
        if self.mle_regression:
            exog_params = params[self._params_regression].reshape(
                self.k_endog, self.k_exog).T
            intercept = np.dot(self.exog[1:], exog_params)
            self.ssm[self._idx_state_intercept] = intercept.T

            if self._final_exog is not None:
                self.ssm['state_intercept', :self.k_endog, -1] = np.dot(
                    self._final_exog, exog_params)

        # - Trend
        if self.k_trend > 0:
            # If we did not set the intercept above, zero it out so we can
            # just += later
            if not self.mle_regression:
                zero = np.array(0, dtype=params.dtype)
                self.ssm['state_intercept', :] = zero

            trend_params = params[self._params_trend].reshape(
                self.k_endog, self.k_trend).T
            if self._trend_is_const:
                intercept = trend_params
            else:
                intercept = np.dot(self._trend_data[1:], trend_params)
            self.ssm[self._idx_state_intercept] += intercept.T

            if (self._final_trend is not None
                    and self._idx_state_intercept[-1].stop == -1):
                self.ssm['state_intercept', :self.k_endog, -1:] += np.dot(
                    self._final_trend, trend_params).T

        # Need to set the last state intercept to np.nan (with appropriate
        # dtype) if we don't have the final exog
        if self.mle_regression and self._final_exog is None:
            nan = np.array(np.nan, dtype=params.dtype)
            self.ssm['state_intercept', :self.k_endog, -1] = nan

        # 2. Transition
        ar = params[self._params_ar].reshape(
            self.k_endog, self.k_endog * self.k_ar)
        ma = params[self._params_ma].reshape(
            self.k_endog, self.k_endog * self.k_ma)
        self.ssm[self._idx_transition] = np.c_[ar, ma]

        # 3. State covariance
        if self.error_cov_type == 'diagonal':
            self.ssm[self._idx_state_cov] = (
                params[self._params_state_cov]
            )
        elif self.error_cov_type == 'unstructured':
            state_cov_lower = np.zeros(self.ssm['state_cov'].shape,
                                       dtype=params.dtype)
            state_cov_lower[self._idx_lower_state_cov] = (
                params[self._params_state_cov])
            self.ssm['state_cov'] = np.dot(state_cov_lower, state_cov_lower.T)

        # 4. Observation covariance
        if self.measurement_error:
            self.ssm[self._idx_obs_cov] = params[self._params_obs_cov]

    @contextlib.contextmanager
    def _set_final_exog(self, exog):
        """
        Set the final state intercept value using out-of-sample `exog` / trend

        Parameters
        ----------
        exog : ndarray
            Out-of-sample `exog` values, usually produced by
            `_validate_out_of_sample_exog` to ensure the correct shape (this
            method does not do any additional validation of its own).
        out_of_sample : int
            Number of out-of-sample periods.

        Notes
        -----
        We need special handling for simulating or forecasting with `exog` or
        trend, because if we had these then the last predicted_state has been
        set to NaN since we did not have the appropriate `exog` to create it.
        Since we handle trend in the same way as `exog`, we still have this
        issue when only trend is used without `exog`.
        """
        cache_value = self._final_exog
        if self.k_exog > 0:
            if exog is not None:
                exog = np.atleast_1d(exog)
                if exog.ndim == 2:
                    exog = exog[:1]
                try:
                    exog = np.reshape(exog[:1], (self.k_exog,))
                except ValueError:
                    raise ValueError('Provided exogenous values are not of the'
                                     ' appropriate shape. Required %s, got %s.'
                                     % (str((self.k_exog,)),
                                        str(exog.shape)))
            self._final_exog = exog
        try:
            yield
        finally:
            self._final_exog = cache_value

    @Appender(MLEModel.simulate.__doc__)
    def simulate(self, params, nsimulations, measurement_shocks=None,
                 state_shocks=None, initial_state=None, anchor=None,
                 repetitions=None, exog=None, extend_model=None,
                 extend_kwargs=None, transformed=True, includes_fixed=False,
                 **kwargs):
        with self._set_final_exog(exog):
            out = super(VARMAX, self).simulate(
                params, nsimulations, measurement_shocks=measurement_shocks,
                state_shocks=state_shocks, initial_state=initial_state,
                anchor=anchor, repetitions=repetitions, exog=exog,
                extend_model=extend_model, extend_kwargs=extend_kwargs,
                transformed=transformed, includes_fixed=includes_fixed,
                **kwargs)
        return out


class VARMAXResults(MLEResults):
    """
    Class to hold results from fitting an VARMAX model.

    Parameters
    ----------
    model : VARMAX instance
        The fitted model instance

    Attributes
    ----------
    specification : dictionary
        Dictionary including all attributes from the VARMAX model instance.
    coefficient_matrices_var : ndarray
        Array containing autoregressive lag polynomial coefficient matrices,
        ordered from lowest degree to highest.
    coefficient_matrices_vma : ndarray
        Array containing moving average lag polynomial coefficients,
        ordered from lowest degree to highest.

    See Also
    --------
    statsmodels.tsa.statespace.kalman_filter.FilterResults
    statsmodels.tsa.statespace.mlemodel.MLEResults
    """
    def __init__(self, model, params, filter_results, cov_type=None,
                 cov_kwds=None, **kwargs):
        super(VARMAXResults, self).__init__(model, params, filter_results,
                                            cov_type, cov_kwds, **kwargs)

        self.specification = Bunch(**{
            # Set additional model parameters
            'error_cov_type': self.model.error_cov_type,
            'measurement_error': self.model.measurement_error,
            'enforce_stationarity': self.model.enforce_stationarity,
            'enforce_invertibility': self.model.enforce_invertibility,
            'trend_offset': self.model.trend_offset,

            'order': self.model.order,

            # Model order
            'k_ar': self.model.k_ar,
            'k_ma': self.model.k_ma,

            # Trend / Regression
            'trend': self.model.trend,
            'k_trend': self.model.k_trend,
            'k_exog': self.model.k_exog,
        })

        # Polynomials / coefficient matrices
        self.coefficient_matrices_var = None
        self.coefficient_matrices_vma = None
        if self.model.k_ar > 0:
            ar_params = np.array(self.params[self.model._params_ar])
            k_endog = self.model.k_endog
            k_ar = self.model.k_ar
            self.coefficient_matrices_var = (
                ar_params.reshape(k_endog * k_ar, k_endog).T
            ).reshape(k_endog, k_endog, k_ar).T
        if self.model.k_ma > 0:
            ma_params = np.array(self.params[self.model._params_ma])
            k_endog = self.model.k_endog
            k_ma = self.model.k_ma
            self.coefficient_matrices_vma = (
                ma_params.reshape(k_endog * k_ma, k_endog).T
            ).reshape(k_endog, k_endog, k_ma).T

    def extend(self, endog, exog=None, **kwargs):
        # If we have exog, then the last element of predicted_state and
        # predicted_state_cov are nan (since they depend on the exog associated
        # with the first out-of-sample point), so we need to compute them here
        if exog is not None:
            fcast = self.get_prediction(self.nobs, self.nobs, exog=exog[:1])
            fcast_results = fcast.prediction_results
            initial_state = fcast_results.predicted_state[..., 0]
            initial_state_cov = fcast_results.predicted_state_cov[..., 0]
        else:
            initial_state = self.predicted_state[..., -1]
            initial_state_cov = self.predicted_state_cov[..., -1]

        kwargs.setdefault('trend_offset', self.nobs + self.model.trend_offset)
        mod = self.model.clone(endog, exog=exog, **kwargs)

        mod.ssm.initialization = Initialization(
            mod.k_states, 'known', constant=initial_state,
            stationary_cov=initial_state_cov)

        if self.smoother_results is not None:
            res = mod.smooth(self.params)
        else:
            res = mod.filter(self.params)

        return res

    @contextlib.contextmanager
    def _set_final_exog(self, exog):
        """
        Set the final state intercept value using out-of-sample `exog` / trend

        Parameters
        ----------
        exog : ndarray
            Out-of-sample `exog` values, usually produced by
            `_validate_out_of_sample_exog` to ensure the correct shape (this
            method does not do any additional validation of its own).
        out_of_sample : int
            Number of out-of-sample periods.

        Notes
        -----
        This context manager calls the model-level context manager and
        additionally updates the last element of filter_results.state_intercept
        appropriately.
        """
        mod = self.model
        with mod._set_final_exog(exog):
            cache_value = self.filter_results.state_intercept[:, -1]
            mod.update(self.params)
            self.filter_results.state_intercept[:mod.k_endog, -1] = (
                mod['state_intercept', :mod.k_endog, -1])
            try:
                yield
            finally:
                self.filter_results.state_intercept[:, -1] = cache_value

    @contextlib.contextmanager
    def _set_final_predicted_state(self, exog, out_of_sample):
        """
        Set the final predicted state value using out-of-sample `exog` / trend

        Parameters
        ----------
        exog : ndarray
            Out-of-sample `exog` values, usually produced by
            `_validate_out_of_sample_exog` to ensure the correct shape (this
            method does not do any additional validation of its own).
        out_of_sample : int
            Number of out-of-sample periods.

        Notes
        -----
        We need special handling for forecasting with `exog`, because
        if we had these then the last predicted_state has been set to NaN since
        we did not have the appropriate `exog` to create it.
        """
        flag = out_of_sample and self.model.k_exog > 0

        if flag:
            tmp_endog = concat([
                self.model.endog[-1:], np.zeros((1, self.model.k_endog))])
            if self.model.k_exog > 0:
                tmp_exog = concat([self.model.exog[-1:], exog[:1]])
            else:
                tmp_exog = None

            tmp_trend_offset = self.model.trend_offset + self.nobs - 1
            tmp_mod = self.model.clone(tmp_endog, exog=tmp_exog,
                                       trend_offset=tmp_trend_offset)
            constant = self.filter_results.predicted_state[:, -2]
            stationary_cov = self.filter_results.predicted_state_cov[:, :, -2]
            tmp_mod.ssm.initialize_known(constant=constant,
                                         stationary_cov=stationary_cov)
            tmp_res = tmp_mod.filter(self.params, transformed=True,
                                     includes_fixed=True, return_ssm=True)

            # Patch up `predicted_state`
            self.filter_results.predicted_state[:, -1] = (
                tmp_res.predicted_state[:, -2])
        try:
            yield
        finally:
            if flag:
                self.filter_results.predicted_state[:, -1] = np.nan

    @Appender(MLEResults.get_prediction.__doc__)
    def get_prediction(self, start=None, end=None, dynamic=False,
                       information_set='predicted', index=None, exog=None,
                       **kwargs):
        if start is None:
            start = 0

        # Handle end (e.g. date)
        _start, _end, out_of_sample, _ = (
            self.model._get_prediction_index(start, end, index, silent=True))

        # Normalize `exog`
        exog = self.model._validate_out_of_sample_exog(exog, out_of_sample)

        # Handle trend offset for extended model
        extend_kwargs = {}
        if self.model.k_trend > 0:
            extend_kwargs['trend_offset'] = (
                self.model.trend_offset + self.nobs)

        # Get the prediction
        with self._set_final_exog(exog):
            with self._set_final_predicted_state(exog, out_of_sample):
                out = super(VARMAXResults, self).get_prediction(
                    start=start, end=end, dynamic=dynamic,
                    information_set=information_set, index=index, exog=exog,
                    extend_kwargs=extend_kwargs, **kwargs)
        return out

    @Appender(MLEResults.simulate.__doc__)
    def simulate(self, nsimulations, measurement_shocks=None,
                 state_shocks=None, initial_state=None, anchor=None,
                 repetitions=None, exog=None, extend_model=None,
                 extend_kwargs=None, **kwargs):
        if anchor is None or anchor == 'start':
            iloc = 0
        elif anchor == 'end':
            iloc = self.nobs
        else:
            iloc, _, _ = self.model._get_index_loc(anchor)

        if iloc < 0:
            iloc = self.nobs + iloc
        if iloc > self.nobs:
            raise ValueError('Cannot anchor simulation after the estimated'
                             ' sample.')

        out_of_sample = max(iloc + nsimulations - self.nobs, 0)

        # Normalize `exog`
        exog = self.model._validate_out_of_sample_exog(exog, out_of_sample)

        with self._set_final_predicted_state(exog, out_of_sample):
            out = super(VARMAXResults, self).simulate(
                nsimulations, measurement_shocks=measurement_shocks,
                state_shocks=state_shocks, initial_state=initial_state,
                anchor=anchor, repetitions=repetitions, exog=exog,
                extend_model=extend_model, extend_kwargs=extend_kwargs,
                **kwargs)

        return out

    def _news_previous_results(self, previous, start, end, periods,
                               revisions_details_start=False,
                               state_index=None):
        # TODO: tests for:
        # - the model cloning used in `kalman_smoother.news` works when we
        #   have time-varying exog (i.e. or do we need to somehow explicitly
        #   call the _set_final_exog and _set_final_predicted_state methods
        #   on the rev_mod / revision_results)
        # - in the case of revisions to `endog`, should the revised model use
        #   the `previous` exog? or the `revised` exog?
        # We need to figure out the out-of-sample exog, so that we can add back
        # in the last exog, predicted state
        exog = None
        out_of_sample = self.nobs - previous.nobs
        if self.model.k_exog > 0 and out_of_sample > 0:
            exog = self.model.exog[-out_of_sample:]

        # Compute the news
        with contextlib.ExitStack() as stack:
            stack.enter_context(previous.model._set_final_exog(exog))
            stack.enter_context(previous._set_final_predicted_state(
                exog, out_of_sample))

            out = self.smoother_results.news(
                previous.smoother_results, start=start, end=end,
                revisions_details_start=revisions_details_start,
                state_index=state_index)
        return out

    @Appender(MLEResults.summary.__doc__)
    def summary(self, alpha=.05, start=None, separate_params=True):
        from statsmodels.iolib.summary import summary_params

        # Create the model name
        spec = self.specification
        if spec.k_ar > 0 and spec.k_ma > 0:
            model_name = 'VARMA'
            order = '(%s,%s)' % (spec.k_ar, spec.k_ma)
        elif spec.k_ar > 0:
            model_name = 'VAR'
            order = '(%s)' % (spec.k_ar)
        else:
            model_name = 'VMA'
            order = '(%s)' % (spec.k_ma)
        if spec.k_exog > 0:
            model_name += 'X'
        model_name = [model_name + order]

        if spec.k_trend > 0:
            model_name.append('intercept')

        if spec.measurement_error:
            model_name.append('measurement error')

        summary = super(VARMAXResults, self).summary(
            alpha=alpha, start=start, model_name=model_name,
            display_params=not separate_params
        )

        if separate_params:
            indices = np.arange(len(self.params))

            def make_table(self, mask, title, strip_end=True):
                res = (self, self.params[mask], self.bse[mask],
                       self.zvalues[mask], self.pvalues[mask],
                       self.conf_int(alpha)[mask])

                param_names = []
                for name in np.array(self.data.param_names)[mask].tolist():
                    if strip_end:
                        param_name = '.'.join(name.split('.')[:-1])
                    else:
                        param_name = name
                    if name in self.fixed_params:
                        param_name = '%s (fixed)' % param_name
                    param_names.append(param_name)

                return summary_params(res, yname=None, xname=param_names,
                                      alpha=alpha, use_t=False, title=title)

            # Add parameter tables for each endogenous variable
            k_endog = self.model.k_endog
            k_ar = self.model.k_ar
            k_ma = self.model.k_ma
            k_trend = self.model.k_trend
            k_exog = self.model.k_exog
            endog_masks = []
            for i in range(k_endog):
                masks = []
                offset = 0

                # 1. Intercept terms
                if k_trend > 0:
                    masks.append(np.arange(i, i + k_endog * k_trend, k_endog))
                    offset += k_endog * k_trend

                # 2. AR terms
                if k_ar > 0:
                    start = i * k_endog * k_ar
                    end = (i + 1) * k_endog * k_ar
                    masks.append(
                        offset + np.arange(start, end))
                    offset += k_ar * k_endog**2

                # 3. MA terms
                if k_ma > 0:
                    start = i * k_endog * k_ma
                    end = (i + 1) * k_endog * k_ma
                    masks.append(
                        offset + np.arange(start, end))
                    offset += k_ma * k_endog**2

                # 4. Regression terms
                if k_exog > 0:
                    masks.append(
                        offset + np.arange(i * k_exog, (i + 1) * k_exog))
                    offset += k_endog * k_exog

                # 5. Measurement error variance terms
                if self.model.measurement_error:
                    masks.append(
                        np.array(self.model.k_params - i - 1, ndmin=1))

                # Create the table
                mask = np.concatenate(masks)
                endog_masks.append(mask)

                endog_names = self.model.endog_names
                if not isinstance(endog_names, list):
                    endog_names = [endog_names]
                title = "Results for equation %s" % endog_names[i]
                table = make_table(self, mask, title)
                summary.tables.append(table)

            # State covariance terms
            state_cov_mask = (
                np.arange(len(self.params))[self.model._params_state_cov])
            table = make_table(self, state_cov_mask, "Error covariance matrix",
                               strip_end=False)
            summary.tables.append(table)

            # Add a table for all other parameters
            masks = []
            for m in (endog_masks, [state_cov_mask]):
                m = np.array(m).flatten()
                if len(m) > 0:
                    masks.append(m)
            masks = np.concatenate(masks)
            inverse_mask = np.array(list(set(indices).difference(set(masks))))
            if len(inverse_mask) > 0:
                table = make_table(self, inverse_mask, "Other parameters",
                                   strip_end=False)
                summary.tables.append(table)

        return summary


class VARMAXResultsWrapper(MLEResultsWrapper):
    _attrs = {}
    _wrap_attrs = wrap.union_dicts(MLEResultsWrapper._wrap_attrs,
                                   _attrs)
    _methods = {}
    _wrap_methods = wrap.union_dicts(MLEResultsWrapper._wrap_methods,
                                     _methods)
wrap.populate_wrapper(VARMAXResultsWrapper, VARMAXResults)  # noqa:E305
