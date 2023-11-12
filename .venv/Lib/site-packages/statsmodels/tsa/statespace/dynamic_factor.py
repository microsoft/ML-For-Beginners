# -*- coding: utf-8 -*-
"""
Dynamic factor model

Author: Chad Fulton
License: Simplified-BSD
"""

import numpy as np
from .mlemodel import MLEModel, MLEResults, MLEResultsWrapper
from .tools import (
    is_invertible, prepare_exog,
    constrain_stationary_univariate, unconstrain_stationary_univariate,
    constrain_stationary_multivariate, unconstrain_stationary_multivariate
)
from statsmodels.multivariate.pca import PCA
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.tools import Bunch
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tools.decorators import cache_readonly
import statsmodels.base.wrapper as wrap
from statsmodels.compat.pandas import Appender


class DynamicFactor(MLEModel):
    r"""
    Dynamic factor model

    Parameters
    ----------
    endog : array_like
        The observed time-series process :math:`y`
    exog : array_like, optional
        Array of exogenous regressors for the observation equation, shaped
        nobs x k_exog.
    k_factors : int
        The number of unobserved factors.
    factor_order : int
        The order of the vector autoregression followed by the factors.
    error_cov_type : {'scalar', 'diagonal', 'unstructured'}, optional
        The structure of the covariance matrix of the observation error term,
        where "unstructured" puts no restrictions on the matrix, "diagonal"
        requires it to be any diagonal matrix (uncorrelated errors), and
        "scalar" requires it to be a scalar times the identity matrix. Default
        is "diagonal".
    error_order : int, optional
        The order of the vector autoregression followed by the observation
        error component. Default is None, corresponding to white noise errors.
    error_var : bool, optional
        Whether or not to model the errors jointly via a vector autoregression,
        rather than as individual autoregressions. Has no effect unless
        `error_order` is set. Default is False.
    enforce_stationarity : bool, optional
        Whether or not to transform the AR parameters to enforce stationarity
        in the autoregressive component of the model. Default is True.
    **kwargs
        Keyword arguments may be used to provide default values for state space
        matrices or for Kalman filtering options. See `Representation`, and
        `KalmanFilter` for more details.

    Attributes
    ----------
    exog : array_like, optional
        Array of exogenous regressors for the observation equation, shaped
        nobs x k_exog.
    k_factors : int
        The number of unobserved factors.
    factor_order : int
        The order of the vector autoregression followed by the factors.
    error_cov_type : {'diagonal', 'unstructured'}
        The structure of the covariance matrix of the error term, where
        "unstructured" puts no restrictions on the matrix and "diagonal"
        requires it to be a diagonal matrix (uncorrelated errors).
    error_order : int
        The order of the vector autoregression followed by the observation
        error component.
    error_var : bool
        Whether or not to model the errors jointly via a vector autoregression,
        rather than as individual autoregressions. Has no effect unless
        `error_order` is set.
    enforce_stationarity : bool, optional
        Whether or not to transform the AR parameters to enforce stationarity
        in the autoregressive component of the model. Default is True.

    Notes
    -----
    The dynamic factor model considered here is in the so-called static form,
    and is specified:

    .. math::

        y_t & = \Lambda f_t + B x_t + u_t \\
        f_t & = A_1 f_{t-1} + \dots + A_p f_{t-p} + \eta_t \\
        u_t & = C_1 u_{t-1} + \dots + C_q u_{t-q} + \varepsilon_t

    where there are `k_endog` observed series and `k_factors` unobserved
    factors. Thus :math:`y_t` is a `k_endog` x 1 vector and :math:`f_t` is a
    `k_factors` x 1 vector.

    :math:`x_t` are optional exogenous vectors, shaped `k_exog` x 1.

    :math:`\eta_t` and :math:`\varepsilon_t` are white noise error terms. In
    order to identify the factors, :math:`Var(\eta_t) = I`. Denote
    :math:`Var(\varepsilon_t) \equiv \Sigma`.

    Options related to the unobserved factors:

    - `k_factors`: this is the dimension of the vector :math:`f_t`, above.
      To exclude factors completely, set `k_factors = 0`.
    - `factor_order`: this is the number of lags to include in the factor
      evolution equation, and corresponds to :math:`p`, above. To have static
      factors, set `factor_order = 0`.

    Options related to the observation error term :math:`u_t`:

    - `error_order`: the number of lags to include in the error evolution
      equation; corresponds to :math:`q`, above. To have white noise errors,
      set `error_order = 0` (this is the default).
    - `error_cov_type`: this controls the form of the covariance matrix
      :math:`\Sigma`. If it is "dscalar", then :math:`\Sigma = \sigma^2 I`. If
      it is "diagonal", then
      :math:`\Sigma = \text{diag}(\sigma_1^2, \dots, \sigma_n^2)`. If it is
      "unstructured", then :math:`\Sigma` is any valid variance / covariance
      matrix (i.e. symmetric and positive definite).
    - `error_var`: this controls whether or not the errors evolve jointly
      according to a VAR(q), or individually according to separate AR(q)
      processes. In terms of the formulation above, if `error_var = False`,
      then the matrices :math:C_i` are diagonal, otherwise they are general
      VAR matrices.

    References
    ----------
    .. [*] Lütkepohl, Helmut. 2007.
       New Introduction to Multiple Time Series Analysis.
       Berlin: Springer.
    """

    def __init__(self, endog, k_factors, factor_order, exog=None,
                 error_order=0, error_var=False, error_cov_type='diagonal',
                 enforce_stationarity=True, **kwargs):

        # Model properties
        self.enforce_stationarity = enforce_stationarity

        # Factor-related properties
        self.k_factors = k_factors
        self.factor_order = factor_order

        # Error-related properties
        self.error_order = error_order
        self.error_var = error_var and error_order > 0
        self.error_cov_type = error_cov_type

        # Exogenous data
        (self.k_exog, exog) = prepare_exog(exog)

        # Note: at some point in the future might add state regression, as in
        # SARIMAX.
        self.mle_regression = self.k_exog > 0

        # We need to have an array or pandas at this point
        if not _is_using_pandas(endog, None):
            endog = np.asanyarray(endog, order='C')

        # Save some useful model orders, internally used
        k_endog = endog.shape[1] if endog.ndim > 1 else 1
        self._factor_order = max(1, self.factor_order) * self.k_factors
        self._error_order = self.error_order * k_endog

        # Calculate the number of states
        k_states = self._factor_order
        k_posdef = self.k_factors
        if self.error_order > 0:
            k_states += self._error_order
            k_posdef += k_endog

        # We can still estimate the model with no dynamic state (e.g. SUR), we
        # just need to have one state that does nothing.
        self._unused_state = False
        if k_states == 0:
            k_states = 1
            k_posdef = 1
            self._unused_state = True

        # Test for non-multivariate endog
        if k_endog < 2:
            raise ValueError('The dynamic factors model is only valid for'
                             ' multivariate time series.')

        # Test for too many factors
        if self.k_factors >= k_endog:
            raise ValueError('Number of factors must be less than the number'
                             ' of endogenous variables.')

        # Test for invalid error_cov_type
        if self.error_cov_type not in ['scalar', 'diagonal', 'unstructured']:
            raise ValueError('Invalid error covariance matrix type'
                             ' specification.')

        # By default, initialize as stationary
        kwargs.setdefault('initialization', 'stationary')

        # Initialize the state space model
        super(DynamicFactor, self).__init__(
            endog, exog=exog, k_states=k_states, k_posdef=k_posdef, **kwargs
        )

        # Set as time-varying model if we have exog
        if self.k_exog > 0:
            self.ssm._time_invariant = False

        # Initialize the components
        self.parameters = {}
        self._initialize_loadings()
        self._initialize_exog()
        self._initialize_error_cov()
        self._initialize_factor_transition()
        self._initialize_error_transition()
        self.k_params = sum(self.parameters.values())

        # Cache parameter vector slices
        def _slice(key, offset):
            length = self.parameters[key]
            param_slice = np.s_[offset:offset + length]
            offset += length
            return param_slice, offset

        offset = 0
        self._params_loadings, offset = _slice('factor_loadings', offset)
        self._params_exog, offset = _slice('exog', offset)
        self._params_error_cov, offset = _slice('error_cov', offset)
        self._params_factor_transition, offset = (
            _slice('factor_transition', offset))
        self._params_error_transition, offset = (
            _slice('error_transition', offset))

        # Update _init_keys attached by super
        self._init_keys += ['k_factors', 'factor_order', 'error_order',
                            'error_var', 'error_cov_type',
                            'enforce_stationarity'] + list(kwargs.keys())

    def _initialize_loadings(self):
        # Initialize the parameters
        self.parameters['factor_loadings'] = self.k_endog * self.k_factors

        # Setup fixed components of state space matrices
        if self.error_order > 0:
            start = self._factor_order
            end = self._factor_order + self.k_endog
            self.ssm['design', :, start:end] = np.eye(self.k_endog)

        # Setup indices of state space matrices
        self._idx_loadings = np.s_['design', :, :self.k_factors]

    def _initialize_exog(self):
        # Initialize the parameters
        self.parameters['exog'] = self.k_exog * self.k_endog

        # If we have exog effects, then the obs intercept needs to be
        # time-varying
        if self.k_exog > 0:
            self.ssm['obs_intercept'] = np.zeros((self.k_endog, self.nobs))

        # Setup indices of state space matrices
        self._idx_exog = np.s_['obs_intercept', :self.k_endog, :]

    def _initialize_error_cov(self):
        if self.error_cov_type == 'scalar':
            self._initialize_error_cov_diagonal(scalar=True)
        elif self.error_cov_type == 'diagonal':
            self._initialize_error_cov_diagonal(scalar=False)
        elif self.error_cov_type == 'unstructured':
            self._initialize_error_cov_unstructured()

    def _initialize_error_cov_diagonal(self, scalar=False):
        # Initialize the parameters
        self.parameters['error_cov'] = 1 if scalar else self.k_endog

        # Setup fixed components of state space matrices

        # Setup indices of state space matrices
        k_endog = self.k_endog
        k_factors = self.k_factors
        idx = np.diag_indices(k_endog)
        if self.error_order > 0:
            matrix = 'state_cov'
            idx = (idx[0] + k_factors, idx[1] + k_factors)
        else:
            matrix = 'obs_cov'
        self._idx_error_cov = (matrix,) + idx

    def _initialize_error_cov_unstructured(self):
        # Initialize the parameters
        k_endog = self.k_endog
        self.parameters['error_cov'] = int(k_endog * (k_endog + 1) / 2)

        # Setup fixed components of state space matrices

        # Setup indices of state space matrices
        self._idx_lower_error_cov = np.tril_indices(self.k_endog)
        if self.error_order > 0:
            start = self.k_factors
            end = self.k_factors + self.k_endog
            self._idx_error_cov = (
                np.s_['state_cov', start:end, start:end])
        else:
            self._idx_error_cov = np.s_['obs_cov', :, :]

    def _initialize_factor_transition(self):
        order = self.factor_order * self.k_factors
        k_factors = self.k_factors

        # Initialize the parameters
        self.parameters['factor_transition'] = (
            self.factor_order * self.k_factors**2)

        # Setup fixed components of state space matrices
        # VAR(p) for factor transition
        if self.k_factors > 0:
            if self.factor_order > 0:
                self.ssm['transition', k_factors:order, :order - k_factors] = (
                    np.eye(order - k_factors))

            self.ssm['selection', :k_factors, :k_factors] = np.eye(k_factors)
            # Identification requires constraining the state covariance to an
            # identity matrix
            self.ssm['state_cov', :k_factors, :k_factors] = np.eye(k_factors)

        # Setup indices of state space matrices
        self._idx_factor_transition = np.s_['transition', :k_factors, :order]

    def _initialize_error_transition(self):
        # Initialize the appropriate situation
        if self.error_order == 0:
            self._initialize_error_transition_white_noise()
        else:
            # Generic setup fixed components of state space matrices
            # VAR(q) for error transition
            # (in the individual AR case, we still have the VAR(q) companion
            # matrix structure, but force the coefficient matrices to be
            # diagonal)
            k_endog = self.k_endog
            k_factors = self.k_factors
            _factor_order = self._factor_order
            _error_order = self._error_order
            _slice = np.s_['selection',
                           _factor_order:_factor_order + k_endog,
                           k_factors:k_factors + k_endog]
            self.ssm[_slice] = np.eye(k_endog)
            _slice = np.s_[
                'transition',
                _factor_order + k_endog:_factor_order + _error_order,
                _factor_order:_factor_order + _error_order - k_endog]
            self.ssm[_slice] = np.eye(_error_order - k_endog)

            # Now specialized setups
            if self.error_var:
                self._initialize_error_transition_var()
            else:
                self._initialize_error_transition_individual()

    def _initialize_error_transition_white_noise(self):
        # Initialize the parameters
        self.parameters['error_transition'] = 0

        # No fixed components of state space matrices

        # Setup indices of state space matrices (just an empty slice)
        self._idx_error_transition = np.s_['transition', 0:0, 0:0]

    def _initialize_error_transition_var(self):
        k_endog = self.k_endog
        _factor_order = self._factor_order
        _error_order = self._error_order

        # Initialize the parameters
        self.parameters['error_transition'] = _error_order * k_endog

        # Fixed components already setup above

        # Setup indices of state space matrices
        # Here we want to set all of the elements of the coefficient matrices,
        # the same as in a VAR specification
        self._idx_error_transition = np.s_[
            'transition',
            _factor_order:_factor_order + k_endog,
            _factor_order:_factor_order + _error_order]

    def _initialize_error_transition_individual(self):
        k_endog = self.k_endog
        _error_order = self._error_order

        # Initialize the parameters
        self.parameters['error_transition'] = _error_order

        # Fixed components already setup above

        # Setup indices of state space matrices
        # Here we want to set only the diagonal elements of the coefficient
        # matrices, and we want to set them in order by equation, not by
        # matrix (i.e. set the first element of the first matrix's diagonal,
        # then set the first element of the second matrix's diagonal, then...)

        # The basic setup is a tiled list of diagonal indices, one for each
        # coefficient matrix
        idx = np.tile(np.diag_indices(k_endog), self.error_order)
        # Now we need to shift the rows down to the correct location
        row_shift = self._factor_order
        # And we need to shift the columns in an increasing way
        col_inc = self._factor_order + np.repeat(
            [i * k_endog for i in range(self.error_order)], k_endog)
        idx[0] += row_shift
        idx[1] += col_inc

        # Make a copy (without the row shift) so that we can easily get the
        # diagonal parameters back out of a generic coefficients matrix array
        idx_diag = idx.copy()
        idx_diag[0] -= row_shift
        idx_diag[1] -= self._factor_order
        idx_diag = idx_diag[:, np.lexsort((idx_diag[1], idx_diag[0]))]
        self._idx_error_diag = (idx_diag[0], idx_diag[1])

        # Finally, we want to fill the entries in in the correct order, which
        # is to say we want to fill in lexicographically, first by row then by
        # column
        idx = idx[:, np.lexsort((idx[1], idx[0]))]
        self._idx_error_transition = np.s_['transition', idx[0], idx[1]]

    def clone(self, endog, exog=None, **kwargs):
        return self._clone_from_init_kwds(endog, exog=exog, **kwargs)

    @property
    def _res_classes(self):
        return {'fit': (DynamicFactorResults, DynamicFactorResultsWrapper)}

    @property
    def start_params(self):
        params = np.zeros(self.k_params, dtype=np.float64)

        endog = self.endog.copy()
        mask = ~np.any(np.isnan(endog), axis=1)
        endog = endog[mask]
        if self.k_exog > 0:
            exog = self.exog[mask]

        # 1. Factor loadings (estimated via PCA)
        if self.k_factors > 0:
            # Use principal components + OLS as starting values
            res_pca = PCA(endog, ncomp=self.k_factors)
            mod_ols = OLS(endog, res_pca.factors)
            res_ols = mod_ols.fit()

            # Using OLS params for the loadings tends to gives higher starting
            # log-likelihood.
            params[self._params_loadings] = res_ols.params.T.ravel()
            # params[self._params_loadings] = res_pca.loadings.ravel()

            # However, using res_ols.resid tends to causes non-invertible
            # starting VAR coefficients for error VARs
            # endog = res_ols.resid
            endog = endog - np.dot(res_pca.factors, res_pca.loadings.T)

        # 2. Exog (OLS on residuals)
        if self.k_exog > 0:
            mod_ols = OLS(endog, exog=exog)
            res_ols = mod_ols.fit()
            # In the form: beta.x1.y1, beta.x2.y1, beta.x1.y2, ...
            params[self._params_exog] = res_ols.params.T.ravel()
            endog = res_ols.resid

        # 3. Factors (VAR on res_pca.factors)
        stationary = True
        if self.k_factors > 1 and self.factor_order > 0:
            # 3a. VAR transition (OLS on factors estimated via PCA)
            mod_factors = VAR(res_pca.factors)
            res_factors = mod_factors.fit(maxlags=self.factor_order, ic=None,
                                          trend='n')
            # Save the parameters
            params[self._params_factor_transition] = (
                res_factors.params.T.ravel())

            # Test for stationarity
            coefficient_matrices = (
                params[self._params_factor_transition].reshape(
                    self.k_factors * self.factor_order, self.k_factors
                ).T
            ).reshape(self.k_factors, self.k_factors, self.factor_order).T

            stationary = is_invertible([1] + list(-coefficient_matrices))
        elif self.k_factors > 0 and self.factor_order > 0:
            # 3b. AR transition
            Y = res_pca.factors[self.factor_order:]
            X = lagmat(res_pca.factors, self.factor_order, trim='both')
            params_ar = np.linalg.pinv(X).dot(Y)
            stationary = is_invertible(np.r_[1, -params_ar.squeeze()])
            params[self._params_factor_transition] = params_ar[:, 0]

        # Check for stationarity
        if not stationary and self.enforce_stationarity:
            raise ValueError('Non-stationary starting autoregressive'
                             ' parameters found with `enforce_stationarity`'
                             ' set to True.')

        # 4. Errors
        if self.error_order == 0:
            if self.error_cov_type == 'scalar':
                params[self._params_error_cov] = endog.var(axis=0).mean()
            elif self.error_cov_type == 'diagonal':
                params[self._params_error_cov] = endog.var(axis=0)
            elif self.error_cov_type == 'unstructured':
                cov_factor = np.diag(endog.std(axis=0))
                params[self._params_error_cov] = (
                    cov_factor[self._idx_lower_error_cov].ravel())
        elif self.error_var:
            mod_errors = VAR(endog)
            res_errors = mod_errors.fit(maxlags=self.error_order, ic=None,
                                        trend='n')

            # Test for stationarity
            coefficient_matrices = (
                np.array(res_errors.params.T).ravel().reshape(
                    self.k_endog * self.error_order, self.k_endog
                ).T
            ).reshape(self.k_endog, self.k_endog, self.error_order).T

            stationary = is_invertible([1] + list(-coefficient_matrices))
            if not stationary and self.enforce_stationarity:
                raise ValueError('Non-stationary starting error autoregressive'
                                 ' parameters found with'
                                 ' `enforce_stationarity` set to True.')

            # Get the error autoregressive parameters
            params[self._params_error_transition] = (
                    np.array(res_errors.params.T).ravel())

            # Get the error covariance parameters
            if self.error_cov_type == 'scalar':
                params[self._params_error_cov] = (
                    res_errors.sigma_u.diagonal().mean())
            elif self.error_cov_type == 'diagonal':
                params[self._params_error_cov] = res_errors.sigma_u.diagonal()
            elif self.error_cov_type == 'unstructured':
                try:
                    cov_factor = np.linalg.cholesky(res_errors.sigma_u)
                except np.linalg.LinAlgError:
                    cov_factor = np.eye(res_errors.sigma_u.shape[0]) * (
                        res_errors.sigma_u.diagonal().mean()**0.5)
                cov_factor = np.eye(res_errors.sigma_u.shape[0]) * (
                    res_errors.sigma_u.diagonal().mean()**0.5)
                params[self._params_error_cov] = (
                    cov_factor[self._idx_lower_error_cov].ravel())
        else:
            error_ar_params = []
            error_cov_params = []
            for i in range(self.k_endog):
                mod_error = ARIMA(endog[:, i], order=(self.error_order, 0, 0),
                                  trend='n', enforce_stationarity=True)
                res_error = mod_error.fit(method='burg')
                error_ar_params += res_error.params[:self.error_order].tolist()
                error_cov_params += res_error.params[-1:].tolist()

            params[self._params_error_transition] = np.r_[error_ar_params]
            params[self._params_error_cov] = np.r_[error_cov_params]

        return params

    @property
    def param_names(self):
        param_names = []
        endog_names = self.endog_names

        # 1. Factor loadings
        param_names += [
            'loading.f%d.%s' % (j+1, endog_names[i])
            for i in range(self.k_endog)
            for j in range(self.k_factors)
        ]

        # 2. Exog
        # Recall these are in the form: beta.x1.y1, beta.x2.y1, beta.x1.y2, ...
        param_names += [
            'beta.%s.%s' % (self.exog_names[j], endog_names[i])
            for i in range(self.k_endog)
            for j in range(self.k_exog)
        ]

        # 3. Error covariances
        if self.error_cov_type == 'scalar':
            param_names += ['sigma2']
        elif self.error_cov_type == 'diagonal':
            param_names += [
                'sigma2.%s' % endog_names[i]
                for i in range(self.k_endog)
            ]
        elif self.error_cov_type == 'unstructured':
            param_names += [
                'cov.chol[%d,%d]' % (i + 1, j + 1)
                for i in range(self.k_endog)
                for j in range(i+1)
            ]

        # 4. Factor transition VAR
        param_names += [
            'L%d.f%d.f%d' % (i+1, k+1, j+1)
            for j in range(self.k_factors)
            for i in range(self.factor_order)
            for k in range(self.k_factors)
        ]

        # 5. Error transition VAR
        if self.error_var:
            param_names += [
                'L%d.e(%s).e(%s)' % (i+1, endog_names[k], endog_names[j])
                for j in range(self.k_endog)
                for i in range(self.error_order)
                for k in range(self.k_endog)
            ]
        else:
            param_names += [
                'L%d.e(%s).e(%s)' % (i+1, endog_names[j], endog_names[j])
                for j in range(self.k_endog)
                for i in range(self.error_order)
            ]

        return param_names

    @property
    def state_names(self):
        names = []
        endog_names = self.endog_names

        # Factors and lags
        names += [
            (('f%d' % (j + 1)) if i == 0 else ('f%d.L%d' % (j + 1, i)))
            for i in range(max(1, self.factor_order))
            for j in range(self.k_factors)]

        if self.error_order > 0:
            names += [
                (('e(%s)' % endog_names[j]) if i == 0
                 else ('e(%s).L%d' % (endog_names[j], i)))
                for i in range(self.error_order)
                for j in range(self.k_endog)]

        if self._unused_state:
            names += ['dummy']

        return names

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
        dtype = unconstrained.dtype
        constrained = np.zeros(unconstrained.shape, dtype=dtype)

        # 1. Factor loadings
        # The factor loadings do not need to be adjusted
        constrained[self._params_loadings] = (
            unconstrained[self._params_loadings])

        # 2. Exog
        # The regression coefficients do not need to be adjusted
        constrained[self._params_exog] = (
            unconstrained[self._params_exog])

        # 3. Error covariances
        # If we have variances, force them to be positive
        if self.error_cov_type in ['scalar', 'diagonal']:
            constrained[self._params_error_cov] = (
                unconstrained[self._params_error_cov]**2)
        # Otherwise, nothing needs to be done
        elif self.error_cov_type == 'unstructured':
            constrained[self._params_error_cov] = (
                unconstrained[self._params_error_cov])

        # 4. Factor transition VAR
        # VAR transition: optionally force to be stationary
        if self.enforce_stationarity and self.factor_order > 0:
            # Transform the parameters
            unconstrained_matrices = (
                unconstrained[self._params_factor_transition].reshape(
                    self.k_factors, self._factor_order))
            # This is always an identity matrix, but because the transform
            # done prior to update (where the ssm representation matrices
            # change), it may be complex
            cov = self.ssm['state_cov', :self.k_factors, :self.k_factors].real
            coefficient_matrices, variance = (
                constrain_stationary_multivariate(unconstrained_matrices, cov))
            constrained[self._params_factor_transition] = (
                coefficient_matrices.ravel())
        else:
            constrained[self._params_factor_transition] = (
                unconstrained[self._params_factor_transition])

        # 5. Error transition VAR
        # VAR transition: optionally force to be stationary
        if self.enforce_stationarity and self.error_order > 0:

            # Joint VAR specification
            if self.error_var:
                unconstrained_matrices = (
                    unconstrained[self._params_error_transition].reshape(
                        self.k_endog, self._error_order))
                start = self.k_factors
                end = self.k_factors + self.k_endog
                cov = self.ssm['state_cov', start:end, start:end].real
                coefficient_matrices, variance = (
                    constrain_stationary_multivariate(
                        unconstrained_matrices, cov))
                constrained[self._params_error_transition] = (
                    coefficient_matrices.ravel())
            # Separate AR specifications
            else:
                coefficients = (
                    unconstrained[self._params_error_transition].copy())
                for i in range(self.k_endog):
                    start = i * self.error_order
                    end = (i + 1) * self.error_order
                    coefficients[start:end] = constrain_stationary_univariate(
                        coefficients[start:end])
                constrained[self._params_error_transition] = coefficients

        else:
            constrained[self._params_error_transition] = (
                unconstrained[self._params_error_transition])

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
        dtype = constrained.dtype
        unconstrained = np.zeros(constrained.shape, dtype=dtype)

        # 1. Factor loadings
        # The factor loadings do not need to be adjusted
        unconstrained[self._params_loadings] = (
            constrained[self._params_loadings])

        # 2. Exog
        # The regression coefficients do not need to be adjusted
        unconstrained[self._params_exog] = (
            constrained[self._params_exog])

        # 3. Error covariances
        # If we have variances, force them to be positive
        if self.error_cov_type in ['scalar', 'diagonal']:
            unconstrained[self._params_error_cov] = (
                constrained[self._params_error_cov]**0.5)
        # Otherwise, nothing needs to be done
        elif self.error_cov_type == 'unstructured':
            unconstrained[self._params_error_cov] = (
                constrained[self._params_error_cov])

        # 3. Factor transition VAR
        # VAR transition: optionally force to be stationary
        if self.enforce_stationarity and self.factor_order > 0:
            # Transform the parameters
            constrained_matrices = (
                constrained[self._params_factor_transition].reshape(
                    self.k_factors, self._factor_order))
            cov = self.ssm['state_cov', :self.k_factors, :self.k_factors].real
            coefficient_matrices, variance = (
                unconstrain_stationary_multivariate(
                    constrained_matrices, cov))
            unconstrained[self._params_factor_transition] = (
                coefficient_matrices.ravel())
        else:
            unconstrained[self._params_factor_transition] = (
                constrained[self._params_factor_transition])

        # 5. Error transition VAR
        # VAR transition: optionally force to be stationary
        if self.enforce_stationarity and self.error_order > 0:

            # Joint VAR specification
            if self.error_var:
                constrained_matrices = (
                    constrained[self._params_error_transition].reshape(
                        self.k_endog, self._error_order))
                start = self.k_factors
                end = self.k_factors + self.k_endog
                cov = self.ssm['state_cov', start:end, start:end].real
                coefficient_matrices, variance = (
                    unconstrain_stationary_multivariate(
                        constrained_matrices, cov))
                unconstrained[self._params_error_transition] = (
                    coefficient_matrices.ravel())
            # Separate AR specifications
            else:
                coefficients = (
                    constrained[self._params_error_transition].copy())
                for i in range(self.k_endog):
                    start = i * self.error_order
                    end = (i + 1) * self.error_order
                    coefficients[start:end] = (
                        unconstrain_stationary_univariate(
                            coefficients[start:end]))
                unconstrained[self._params_error_transition] = coefficients

        else:
            unconstrained[self._params_error_transition] = (
                constrained[self._params_error_transition])

        return unconstrained

    def _validate_can_fix_params(self, param_names):
        super(DynamicFactor, self)._validate_can_fix_params(param_names)

        ix = np.cumsum(list(self.parameters.values()))[:-1]
        (_, _, _, factor_transition_names, error_transition_names) = [
            arr.tolist() for arr in np.array_split(self.param_names, ix)]

        if self.enforce_stationarity and self.factor_order > 0:
            if self.k_factors > 1 or self.factor_order > 1:
                fix_all = param_names.issuperset(factor_transition_names)
                fix_any = (
                    len(param_names.intersection(factor_transition_names)) > 0)
                if fix_any and not fix_all:
                    raise ValueError(
                        'Cannot fix individual factor transition parameters'
                        ' when `enforce_stationarity=True`. In this case,'
                        ' must either fix all factor transition parameters or'
                        ' none.')
        if self.enforce_stationarity and self.error_order > 0:
            if self.error_var or self.error_order > 1:
                fix_all = param_names.issuperset(error_transition_names)
                fix_any = (
                    len(param_names.intersection(error_transition_names)) > 0)
                if fix_any and not fix_all:
                    raise ValueError(
                        'Cannot fix individual error transition parameters'
                        ' when `enforce_stationarity=True`. In this case,'
                        ' must either fix all error transition parameters or'
                        ' none.')

    def update(self, params, transformed=True, includes_fixed=False,
               complex_step=False):
        """
        Update the parameters of the model

        Updates the representation matrices to fill in the new parameter
        values.

        Parameters
        ----------
        params : array_like
            Array of new parameters.
        transformed : bool, optional
            Whether or not `params` is already transformed. If set to False,
            `transform_params` is called. Default is True..

        Returns
        -------
        params : array_like
            Array of parameters.

        Notes
        -----
        Let `n = k_endog`, `m = k_factors`, and `p = factor_order`. Then the
        `params` vector has length
        :math:`[n \times m] + [n] + [m^2 \times p]`.
        It is expanded in the following way:

        - The first :math:`n \times m` parameters fill out the factor loading
          matrix, starting from the [0,0] entry and then proceeding along rows.
          These parameters are not modified in `transform_params`.
        - The next :math:`n` parameters provide variances for the error_cov
          errors in the observation equation. They fill in the diagonal of the
          observation covariance matrix, and are constrained to be positive by
          `transofrm_params`.
        - The next :math:`m^2 \times p` parameters are used to create the `p`
          coefficient matrices for the vector autoregression describing the
          factor transition. They are transformed in `transform_params` to
          enforce stationarity of the VAR(p). They are placed so as to make
          the transition matrix a companion matrix for the VAR. In particular,
          we assume that the first :math:`m^2` parameters fill the first
          coefficient matrix (starting at [0,0] and filling along rows), the
          second :math:`m^2` parameters fill the second matrix, etc.
        """
        params = self.handle_params(params, transformed=transformed,
                                    includes_fixed=includes_fixed)

        # 1. Factor loadings
        # Update the design / factor loading matrix
        self.ssm[self._idx_loadings] = (
            params[self._params_loadings].reshape(self.k_endog, self.k_factors)
        )

        # 2. Exog
        if self.k_exog > 0:
            exog_params = params[self._params_exog].reshape(
                self.k_endog, self.k_exog).T
            self.ssm[self._idx_exog] = np.dot(self.exog, exog_params).T

        # 3. Error covariances
        if self.error_cov_type in ['scalar', 'diagonal']:
            self.ssm[self._idx_error_cov] = (
                params[self._params_error_cov])
        elif self.error_cov_type == 'unstructured':
            error_cov_lower = np.zeros((self.k_endog, self.k_endog),
                                       dtype=params.dtype)
            error_cov_lower[self._idx_lower_error_cov] = (
                params[self._params_error_cov])
            self.ssm[self._idx_error_cov] = (
                np.dot(error_cov_lower, error_cov_lower.T))

        # 4. Factor transition VAR
        self.ssm[self._idx_factor_transition] = (
            params[self._params_factor_transition].reshape(
                self.k_factors, self.factor_order * self.k_factors))

        # 5. Error transition VAR
        if self.error_var:
            self.ssm[self._idx_error_transition] = (
                params[self._params_error_transition].reshape(
                    self.k_endog, self._error_order))
        else:
            self.ssm[self._idx_error_transition] = (
                params[self._params_error_transition])


class DynamicFactorResults(MLEResults):
    """
    Class to hold results from fitting an DynamicFactor model.

    Parameters
    ----------
    model : DynamicFactor instance
        The fitted model instance

    Attributes
    ----------
    specification : dictionary
        Dictionary including all attributes from the DynamicFactor model
        instance.
    coefficient_matrices_var : ndarray
        Array containing autoregressive lag polynomial coefficient matrices,
        ordered from lowest degree to highest.

    See Also
    --------
    statsmodels.tsa.statespace.kalman_filter.FilterResults
    statsmodels.tsa.statespace.mlemodel.MLEResults
    """
    def __init__(self, model, params, filter_results, cov_type=None,
                 **kwargs):
        super(DynamicFactorResults, self).__init__(model, params,
                                                   filter_results, cov_type,
                                                   **kwargs)

        self.df_resid = np.inf  # attribute required for wald tests

        self.specification = Bunch(**{
            # Model properties
            'k_endog': self.model.k_endog,
            'enforce_stationarity': self.model.enforce_stationarity,

            # Factor-related properties
            'k_factors': self.model.k_factors,
            'factor_order': self.model.factor_order,

            # Error-related properties
            'error_order': self.model.error_order,
            'error_var': self.model.error_var,
            'error_cov_type': self.model.error_cov_type,

            # Other properties
            'k_exog': self.model.k_exog
        })

        # Polynomials / coefficient matrices
        self.coefficient_matrices_var = None
        if self.model.factor_order > 0:
            ar_params = (
                np.array(self.params[self.model._params_factor_transition]))
            k_factors = self.model.k_factors
            factor_order = self.model.factor_order
            self.coefficient_matrices_var = (
                ar_params.reshape(k_factors * factor_order, k_factors).T
            ).reshape(k_factors, k_factors, factor_order).T

        self.coefficient_matrices_error = None
        if self.model.error_order > 0:
            ar_params = (
                np.array(self.params[self.model._params_error_transition]))
            k_endog = self.model.k_endog
            error_order = self.model.error_order
            if self.model.error_var:
                self.coefficient_matrices_error = (
                    ar_params.reshape(k_endog * error_order, k_endog).T
                ).reshape(k_endog, k_endog, error_order).T
            else:
                mat = np.zeros((k_endog, k_endog * error_order))
                mat[self.model._idx_error_diag] = ar_params
                self.coefficient_matrices_error = (
                    mat.T.reshape(error_order, k_endog, k_endog))

    @property
    def factors(self):
        """
        Estimates of unobserved factors

        Returns
        -------
        out : Bunch
            Has the following attributes shown in Notes.

        Notes
        -----
        The output is a bunch of the following format:

        - `filtered`: a time series array with the filtered estimate of
          the component
        - `filtered_cov`: a time series array with the filtered estimate of
          the variance/covariance of the component
        - `smoothed`: a time series array with the smoothed estimate of
          the component
        - `smoothed_cov`: a time series array with the smoothed estimate of
          the variance/covariance of the component
        - `offset`: an integer giving the offset in the state vector where
          this component begins
        """
        # If present, level is always the first component of the state vector
        out = None
        spec = self.specification
        if spec.k_factors > 0:
            offset = 0
            end = spec.k_factors
            res = self.filter_results
            out = Bunch(
                filtered=res.filtered_state[offset:end],
                filtered_cov=res.filtered_state_cov[offset:end, offset:end],
                smoothed=None, smoothed_cov=None,
                offset=offset)
            if self.smoothed_state is not None:
                out.smoothed = self.smoothed_state[offset:end]
            if self.smoothed_state_cov is not None:
                out.smoothed_cov = (
                    self.smoothed_state_cov[offset:end, offset:end])
        return out

    @cache_readonly
    def coefficients_of_determination(self):
        """
        Coefficients of determination (:math:`R^2`) from regressions of
        individual estimated factors on endogenous variables.

        Returns
        -------
        coefficients_of_determination : ndarray
            A `k_endog` x `k_factors` array, where
            `coefficients_of_determination[i, j]` represents the :math:`R^2`
            value from a regression of factor `j` and a constant on endogenous
            variable `i`.

        Notes
        -----
        Although it can be difficult to interpret the estimated factor loadings
        and factors, it is often helpful to use the coefficients of
        determination from univariate regressions to assess the importance of
        each factor in explaining the variation in each endogenous variable.

        In models with many variables and factors, this can sometimes lend
        interpretation to the factors (for example sometimes one factor will
        load primarily on real variables and another on nominal variables).

        See Also
        --------
        plot_coefficients_of_determination
        """
        from statsmodels.tools import add_constant
        spec = self.specification
        coefficients = np.zeros((spec.k_endog, spec.k_factors))
        which = 'filtered' if self.smoothed_state is None else 'smoothed'

        for i in range(spec.k_factors):
            exog = add_constant(self.factors[which][i])
            for j in range(spec.k_endog):
                endog = self.filter_results.endog[j]
                coefficients[j, i] = OLS(endog, exog).fit().rsquared

        return coefficients

    def plot_coefficients_of_determination(self, endog_labels=None,
                                           fig=None, figsize=None):
        """
        Plot the coefficients of determination

        Parameters
        ----------
        endog_labels : bool, optional
            Whether or not to label the endogenous variables along the x-axis
            of the plots. Default is to include labels if there are 5 or fewer
            endogenous variables.
        fig : Figure, optional
            If given, subplots are created in this figure instead of in a new
            figure. Note that the grid will be created in the provided
            figure using `fig.add_subplot()`.
        figsize : tuple, optional
            If a figure is created, this argument allows specifying a size.
            The tuple is (width, height).

        Notes
        -----

        Produces a `k_factors` x 1 plot grid. The `i`th plot shows a bar plot
        of the coefficients of determination associated with factor `i`. The
        endogenous variables are arranged along the x-axis according to their
        position in the `endog` array.

        See Also
        --------
        coefficients_of_determination
        """
        from statsmodels.graphics.utils import _import_mpl, create_mpl_fig
        _import_mpl()
        fig = create_mpl_fig(fig, figsize)

        spec = self.specification

        # Should we label endogenous variables?
        if endog_labels is None:
            endog_labels = spec.k_endog <= 5

        # Plot the coefficients of determination
        coefficients_of_determination = self.coefficients_of_determination
        plot_idx = 1
        locations = np.arange(spec.k_endog)
        for coeffs in coefficients_of_determination.T:
            # Create the new axis
            ax = fig.add_subplot(spec.k_factors, 1, plot_idx)
            ax.set_ylim((0, 1))
            ax.set(title='Factor %i' % plot_idx, ylabel=r'$R^2$')
            bars = ax.bar(locations, coeffs)

            if endog_labels:
                width = bars[0].get_width()
                ax.xaxis.set_ticks(locations + width / 2)
                ax.xaxis.set_ticklabels(self.model.endog_names)
            else:
                ax.set(xlabel='Endogenous variables')
                ax.xaxis.set_ticks([])

            plot_idx += 1

        return fig

    @Appender(MLEResults.summary.__doc__)
    def summary(self, alpha=.05, start=None, separate_params=True):
        from statsmodels.iolib.summary import summary_params
        spec = self.specification

        # Create the model name
        model_name = []
        if spec.k_factors > 0:
            if spec.factor_order > 0:
                model_type = ('DynamicFactor(factors=%d, order=%d)' %
                              (spec.k_factors, spec.factor_order))
            else:
                model_type = 'StaticFactor(factors=%d)' % spec.k_factors

            model_name.append(model_type)
            if spec.k_exog > 0:
                model_name.append('%d regressors' % spec.k_exog)
        else:
            model_name.append('SUR(%d regressors)' % spec.k_exog)

        if spec.error_order > 0:
            error_type = 'VAR' if spec.error_var else 'AR'
            model_name.append('%s(%d) errors' % (error_type, spec.error_order))

        summary = super(DynamicFactorResults, self).summary(
            alpha=alpha, start=start, model_name=model_name,
            display_params=not separate_params
        )

        if separate_params:
            indices = np.arange(len(self.params))

            def make_table(self, mask, title, strip_end=True):
                res = (self, self.params[mask], self.bse[mask],
                       self.zvalues[mask], self.pvalues[mask],
                       self.conf_int(alpha)[mask])

                param_names = [
                    '.'.join(name.split('.')[:-1]) if strip_end else name
                    for name in
                    np.array(self.data.param_names)[mask].tolist()
                ]

                return summary_params(res, yname=None, xname=param_names,
                                      alpha=alpha, use_t=False, title=title)

            k_endog = self.model.k_endog
            k_exog = self.model.k_exog
            k_factors = self.model.k_factors
            factor_order = self.model.factor_order
            _factor_order = self.model._factor_order
            _error_order = self.model._error_order

            # Add parameter tables for each endogenous variable
            loading_indices = indices[self.model._params_loadings]
            loading_masks = []
            exog_indices = indices[self.model._params_exog]
            exog_masks = []
            for i in range(k_endog):
                # 1. Factor loadings
                # Recall these are in the form:
                # 'loading.f1.y1', 'loading.f2.y1', 'loading.f1.y2', ...

                loading_mask = (
                    loading_indices[i * k_factors:(i + 1) * k_factors])
                loading_masks.append(loading_mask)

                # 2. Exog
                # Recall these are in the form:
                # beta.x1.y1, beta.x2.y1, beta.x1.y2, ...
                exog_mask = exog_indices[i * k_exog:(i + 1) * k_exog]
                exog_masks.append(exog_mask)

                # Create the table
                mask = np.concatenate([loading_mask, exog_mask])
                title = "Results for equation %s" % self.model.endog_names[i]
                table = make_table(self, mask, title)
                summary.tables.append(table)

            # Add parameter tables for each factor
            factor_indices = indices[self.model._params_factor_transition]
            factor_masks = []
            if factor_order > 0:
                for i in range(k_factors):
                    start = i * _factor_order
                    factor_mask = factor_indices[start: start + _factor_order]
                    factor_masks.append(factor_mask)

                    # Create the table
                    title = "Results for factor equation f%d" % (i+1)
                    table = make_table(self, factor_mask, title)
                    summary.tables.append(table)

            # Add parameter tables for error transitions
            error_masks = []
            if spec.error_order > 0:
                error_indices = indices[self.model._params_error_transition]
                for i in range(k_endog):
                    if spec.error_var:
                        start = i * _error_order
                        end = (i + 1) * _error_order
                    else:
                        start = i * spec.error_order
                        end = (i + 1) * spec.error_order

                    error_mask = error_indices[start:end]
                    error_masks.append(error_mask)

                    # Create the table
                    title = ("Results for error equation e(%s)" %
                             self.model.endog_names[i])
                    table = make_table(self, error_mask, title)
                    summary.tables.append(table)

            # Error covariance terms
            error_cov_mask = indices[self.model._params_error_cov]
            table = make_table(self, error_cov_mask,
                               "Error covariance matrix", strip_end=False)
            summary.tables.append(table)

            # Add a table for all other parameters
            masks = []
            for m in (loading_masks, exog_masks, factor_masks,
                      error_masks, [error_cov_mask]):
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


class DynamicFactorResultsWrapper(MLEResultsWrapper):
    _attrs = {}
    _wrap_attrs = wrap.union_dicts(MLEResultsWrapper._wrap_attrs,
                                   _attrs)
    _methods = {}
    _wrap_methods = wrap.union_dicts(MLEResultsWrapper._wrap_methods,
                                     _methods)
wrap.populate_wrapper(DynamicFactorResultsWrapper,  # noqa:E305
                      DynamicFactorResults)
