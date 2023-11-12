"""
SARIMAX Model

Author: Chad Fulton
License: Simplified-BSD
"""
from warnings import warn

import numpy as np
import pandas as pd

from statsmodels.compat.pandas import Appender

from statsmodels.tools.tools import Bunch
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.decorators import cache_readonly
import statsmodels.base.wrapper as wrap

from statsmodels.tsa.arima.specification import SARIMAXSpecification
from statsmodels.tsa.arima.params import SARIMAXParams
from statsmodels.tsa.tsatools import lagmat

from .initialization import Initialization
from .mlemodel import MLEModel, MLEResults, MLEResultsWrapper
from .tools import (
    companion_matrix, diff, is_invertible, constrain_stationary_univariate,
    unconstrain_stationary_univariate,
    prepare_exog, prepare_trend_spec, prepare_trend_data)


class SARIMAX(MLEModel):
    r"""
    Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors
    model

    Parameters
    ----------
    endog : array_like
        The observed time-series process :math:`y`
    exog : array_like, optional
        Array of exogenous regressors, shaped nobs x k.
    order : iterable or iterable of iterables, optional
        The (p,d,q) order of the model for the number of AR parameters,
        differences, and MA parameters. `d` must be an integer
        indicating the integration order of the process, while
        `p` and `q` may either be an integers indicating the AR and MA
        orders (so that all lags up to those orders are included) or else
        iterables giving specific AR and / or MA lags to include. Default is
        an AR(1) model: (1,0,0).
    seasonal_order : iterable, optional
        The (P,D,Q,s) order of the seasonal component of the model for the
        AR parameters, differences, MA parameters, and periodicity.
        `D` must be an integer indicating the integration order of the process,
        while `P` and `Q` may either be an integers indicating the AR and MA
        orders (so that all lags up to those orders are included) or else
        iterables giving specific AR and / or MA lags to include. `s` is an
        integer giving the periodicity (number of periods in season), often it
        is 4 for quarterly data or 12 for monthly data. Default is no seasonal
        effect.
    trend : str{'n','c','t','ct'} or iterable, optional
        Parameter controlling the deterministic trend polynomial :math:`A(t)`.
        Can be specified as a string where 'c' indicates a constant (i.e. a
        degree zero component of the trend polynomial), 't' indicates a
        linear trend with time, and 'ct' is both. Can also be specified as an
        iterable defining the non-zero polynomial exponents to include, in
        increasing order. For example, `[1,1,0,1]` denotes
        :math:`a + bt + ct^3`. Default is to not include a trend component.
    measurement_error : bool, optional
        Whether or not to assume the endogenous observations `endog` were
        measured with error. Default is False.
    time_varying_regression : bool, optional
        Used when an explanatory variables, `exog`, are provided
        to select whether or not coefficients on the exogenous regressors are
        allowed to vary over time. Default is False.
    mle_regression : bool, optional
        Whether or not to use estimate the regression coefficients for the
        exogenous variables as part of maximum likelihood estimation or through
        the Kalman filter (i.e. recursive least squares). If
        `time_varying_regression` is True, this must be set to False. Default
        is True.
    simple_differencing : bool, optional
        Whether or not to use partially conditional maximum likelihood
        estimation. If True, differencing is performed prior to estimation,
        which discards the first :math:`s D + d` initial rows but results in a
        smaller state-space formulation. See the Notes section for important
        details about interpreting results when this option is used. If False,
        the full SARIMAX model is put in state-space form so that all
        datapoints can be used in estimation. Default is False.
    enforce_stationarity : bool, optional
        Whether or not to transform the AR parameters to enforce stationarity
        in the autoregressive component of the model. Default is True.
    enforce_invertibility : bool, optional
        Whether or not to transform the MA parameters to enforce invertibility
        in the moving average component of the model. Default is True.
    hamilton_representation : bool, optional
        Whether or not to use the Hamilton representation of an ARMA process
        (if True) or the Harvey representation (if False). Default is False.
    concentrate_scale : bool, optional
        Whether or not to concentrate the scale (variance of the error term)
        out of the likelihood. This reduces the number of parameters estimated
        by maximum likelihood by one, but standard errors will then not
        be available for the scale parameter.
    trend_offset : int, optional
        The offset at which to start time trend values. Default is 1, so that
        if `trend='t'` the trend is equal to 1, 2, ..., nobs. Typically is only
        set when the model created by extending a previous dataset.
    use_exact_diffuse : bool, optional
        Whether or not to use exact diffuse initialization for non-stationary
        states. Default is False (in which case approximate diffuse
        initialization is used).
    **kwargs
        Keyword arguments may be used to provide default values for state space
        matrices or for Kalman filtering options. See `Representation`, and
        `KalmanFilter` for more details.

    Attributes
    ----------
    measurement_error : bool
        Whether or not to assume the endogenous
        observations `endog` were measured with error.
    state_error : bool
        Whether or not the transition equation has an error component.
    mle_regression : bool
        Whether or not the regression coefficients for
        the exogenous variables were estimated via maximum
        likelihood estimation.
    state_regression : bool
        Whether or not the regression coefficients for
        the exogenous variables are included as elements
        of the state space and estimated via the Kalman
        filter.
    time_varying_regression : bool
        Whether or not coefficients on the exogenous
        regressors are allowed to vary over time.
    simple_differencing : bool
        Whether or not to use partially conditional maximum likelihood
        estimation.
    enforce_stationarity : bool
        Whether or not to transform the AR parameters
        to enforce stationarity in the autoregressive
        component of the model.
    enforce_invertibility : bool
        Whether or not to transform the MA parameters
        to enforce invertibility in the moving average
        component of the model.
    hamilton_representation : bool
        Whether or not to use the Hamilton representation of an ARMA process.
    trend : str{'n','c','t','ct'} or iterable
        Parameter controlling the deterministic
        trend polynomial :math:`A(t)`. See the class
        parameter documentation for more information.
    polynomial_ar : ndarray
        Array containing autoregressive lag polynomial lags, ordered from
        lowest degree to highest. The polynomial begins with lag 0.
        Initialized with ones, unless a coefficient is constrained to be
        zero (in which case it is zero).
    polynomial_ma : ndarray
        Array containing moving average lag polynomial lags, ordered from
        lowest degree to highest. Initialized with ones, unless a coefficient
        is constrained to be zero (in which case it is zero).
    polynomial_seasonal_ar : ndarray
        Array containing seasonal moving average lag
        polynomial lags, ordered from lowest degree
        to highest. Initialized with ones, unless a
        coefficient is constrained to be zero (in which
        case it is zero).
    polynomial_seasonal_ma : ndarray
        Array containing seasonal moving average lag
        polynomial lags, ordered from lowest degree
        to highest. Initialized with ones, unless a
        coefficient is constrained to be zero (in which
        case it is zero).
    polynomial_trend : ndarray
        Array containing trend polynomial coefficients,
        ordered from lowest degree to highest. Initialized
        with ones, unless a coefficient is constrained to be
        zero (in which case it is zero).
    k_ar : int
        Highest autoregressive order in the model, zero-indexed.
    k_ar_params : int
        Number of autoregressive parameters to be estimated.
    k_diff : int
        Order of integration.
    k_ma : int
        Highest moving average order in the model, zero-indexed.
    k_ma_params : int
        Number of moving average parameters to be estimated.
    seasonal_periods : int
        Number of periods in a season.
    k_seasonal_ar : int
        Highest seasonal autoregressive order in the model, zero-indexed.
    k_seasonal_ar_params : int
        Number of seasonal autoregressive parameters to be estimated.
    k_seasonal_diff : int
        Order of seasonal integration.
    k_seasonal_ma : int
        Highest seasonal moving average order in the model, zero-indexed.
    k_seasonal_ma_params : int
        Number of seasonal moving average parameters to be estimated.
    k_trend : int
        Order of the trend polynomial plus one (i.e. the constant polynomial
        would have `k_trend=1`).
    k_exog : int
        Number of exogenous regressors.

    Notes
    -----
    The SARIMA model is specified :math:`(p, d, q) \times (P, D, Q)_s`.

    .. math::

        \phi_p (L) \tilde \phi_P (L^s) \Delta^d \Delta_s^D y_t = A(t) +
            \theta_q (L) \tilde \theta_Q (L^s) \zeta_t

    In terms of a univariate structural model, this can be represented as

    .. math::

        y_t & = u_t + \eta_t \\
        \phi_p (L) \tilde \phi_P (L^s) \Delta^d \Delta_s^D u_t & = A(t) +
            \theta_q (L) \tilde \theta_Q (L^s) \zeta_t

    where :math:`\eta_t` is only applicable in the case of measurement error
    (although it is also used in the case of a pure regression model, i.e. if
    p=q=0).

    In terms of this model, regression with SARIMA errors can be represented
    easily as

    .. math::

        y_t & = \beta_t x_t + u_t \\
        \phi_p (L) \tilde \phi_P (L^s) \Delta^d \Delta_s^D u_t & = A(t) +
            \theta_q (L) \tilde \theta_Q (L^s) \zeta_t

    this model is the one used when exogenous regressors are provided.

    Note that the reduced form lag polynomials will be written as:

    .. math::

        \Phi (L) \equiv \phi_p (L) \tilde \phi_P (L^s) \\
        \Theta (L) \equiv \theta_q (L) \tilde \theta_Q (L^s)

    If `mle_regression` is True, regression coefficients are treated as
    additional parameters to be estimated via maximum likelihood. Otherwise
    they are included as part of the state with a diffuse initialization.
    In this case, however, with approximate diffuse initialization, results
    can be sensitive to the initial variance.

    This class allows two different underlying representations of ARMA models
    as state space models: that of Hamilton and that of Harvey. Both are
    equivalent in the sense that they are analytical representations of the
    ARMA model, but the state vectors of each have different meanings. For
    this reason, maximum likelihood does not result in identical parameter
    estimates and even the same set of parameters will result in different
    loglikelihoods.

    The Harvey representation is convenient because it allows integrating
    differencing into the state vector to allow using all observations for
    estimation.

    In this implementation of differenced models, the Hamilton representation
    is not able to accommodate differencing in the state vector, so
    `simple_differencing` (which performs differencing prior to estimation so
    that the first d + sD observations are lost) must be used.

    Many other packages use the Hamilton representation, so that tests against
    Stata and R require using it along with simple differencing (as Stata
    does).

    If `filter_concentrated = True` is used, then the scale of the model is
    concentrated out of the likelihood. A benefit of this is that there the
    dimension of the parameter vector is reduced so that numerical maximization
    of the log-likelihood function may be faster and more stable. If this
    option in a model with measurement error, it is important to note that the
    estimated measurement error parameter will be relative to the scale, and
    is named "snr.measurement_error" instead of "var.measurement_error". To
    compute the variance of the measurement error in this case one would
    multiply `snr.measurement_error` parameter by the scale.

    If `simple_differencing = True` is used, then the `endog` and `exog` data
    are differenced prior to putting the model in state-space form. This has
    the same effect as if the user differenced the data prior to constructing
    the model, which has implications for using the results:

    - Forecasts and predictions will be about the *differenced* data, not about
      the original data. (while if `simple_differencing = False` is used, then
      forecasts and predictions will be about the original data).
    - If the original data has an Int64Index, a new RangeIndex will be created
      for the differenced data that starts from one, and forecasts and
      predictions will use this new index.

    Detailed information about state space models can be found in [1]_. Some
    specific references are:

    - Chapter 3.4 describes ARMA and ARIMA models in state space form (using
      the Harvey representation), and gives references for basic seasonal
      models and models with a multiplicative form (for example the airline
      model). It also shows a state space model for a full ARIMA process (this
      is what is done here if `simple_differencing=False`).
    - Chapter 3.6 describes estimating regression effects via the Kalman filter
      (this is performed if `mle_regression` is False), regression with
      time-varying coefficients, and regression with ARMA errors (recall from
      above that if regression effects are present, the model estimated by this
      class is regression with SARIMA errors).
    - Chapter 8.4 describes the application of an ARMA model to an example
      dataset. A replication of this section is available in an example
      IPython notebook in the documentation.

    References
    ----------
    .. [1] Durbin, James, and Siem Jan Koopman. 2012.
       Time Series Analysis by State Space Methods: Second Edition.
       Oxford University Press.
    """

    def __init__(self, endog, exog=None, order=(1, 0, 0),
                 seasonal_order=(0, 0, 0, 0), trend=None,
                 measurement_error=False, time_varying_regression=False,
                 mle_regression=True, simple_differencing=False,
                 enforce_stationarity=True, enforce_invertibility=True,
                 hamilton_representation=False, concentrate_scale=False,
                 trend_offset=1, use_exact_diffuse=False, dates=None,
                 freq=None, missing='none', validate_specification=True,
                 **kwargs):

        self._spec = SARIMAXSpecification(
            endog, exog=exog, order=order, seasonal_order=seasonal_order,
            trend=trend, enforce_stationarity=None, enforce_invertibility=None,
            concentrate_scale=concentrate_scale, dates=dates, freq=freq,
            missing=missing, validate_specification=validate_specification)
        self._params = SARIMAXParams(self._spec)

        # Save given orders
        order = self._spec.order
        seasonal_order = self._spec.seasonal_order
        self.order = order
        self.seasonal_order = seasonal_order

        # Model parameters
        self.seasonal_periods = seasonal_order[3]
        self.measurement_error = measurement_error
        self.time_varying_regression = time_varying_regression
        self.mle_regression = mle_regression
        self.simple_differencing = simple_differencing
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.hamilton_representation = hamilton_representation
        self.concentrate_scale = concentrate_scale
        self.use_exact_diffuse = use_exact_diffuse

        # Enforce non-MLE coefficients if time varying coefficients is
        # specified
        if self.time_varying_regression and self.mle_regression:
            raise ValueError('Models with time-varying regression coefficients'
                             ' must integrate the coefficients as part of the'
                             ' state vector, so that `mle_regression` must'
                             ' be set to False.')

        # Lag polynomials
        self._params.ar_params = -1
        self.polynomial_ar = self._params.ar_poly.coef
        self._polynomial_ar = self.polynomial_ar.copy()

        self._params.ma_params = 1
        self.polynomial_ma = self._params.ma_poly.coef
        self._polynomial_ma = self.polynomial_ma.copy()

        self._params.seasonal_ar_params = -1
        self.polynomial_seasonal_ar = self._params.seasonal_ar_poly.coef
        self._polynomial_seasonal_ar = self.polynomial_seasonal_ar.copy()

        self._params.seasonal_ma_params = 1
        self.polynomial_seasonal_ma = self._params.seasonal_ma_poly.coef
        self._polynomial_seasonal_ma = self.polynomial_seasonal_ma.copy()

        # Deterministic trend polynomial
        self.trend = trend
        self.trend_offset = trend_offset
        self.polynomial_trend, self.k_trend = prepare_trend_spec(self.trend)
        self._polynomial_trend = self.polynomial_trend.copy()
        self._k_trend = self.k_trend
        # (we internally use _k_trend for mechanics so that the public
        # attribute can be overridden by subclasses)

        # Model orders
        # Note: k_ar, k_ma, k_seasonal_ar, k_seasonal_ma do not include the
        # constant term, so they may be zero.
        # Note: for a typical ARMA(p,q) model, p = k_ar_params = k_ar - 1 and
        # q = k_ma_params = k_ma - 1, although this may not be true for models
        # with arbitrary log polynomials.
        self.k_ar = self._spec.max_ar_order
        self.k_ar_params = self._spec.k_ar_params
        self.k_diff = int(order[1])
        self.k_ma = self._spec.max_ma_order
        self.k_ma_params = self._spec.k_ma_params

        self.k_seasonal_ar = (self._spec.max_seasonal_ar_order *
                              self._spec.seasonal_periods)
        self.k_seasonal_ar_params = self._spec.k_seasonal_ar_params
        self.k_seasonal_diff = int(seasonal_order[1])
        self.k_seasonal_ma = (self._spec.max_seasonal_ma_order *
                              self._spec.seasonal_periods)
        self.k_seasonal_ma_params = self._spec.k_seasonal_ma_params

        # Make internal copies of the differencing orders because if we use
        # simple differencing, then we will need to internally use zeros after
        # the simple differencing has been performed
        self._k_diff = self.k_diff
        self._k_seasonal_diff = self.k_seasonal_diff

        # We can only use the Hamilton representation if differencing is not
        # performed as a part of the state space
        if (self.hamilton_representation and not (self.simple_differencing or
           self._k_diff == self._k_seasonal_diff == 0)):
            raise ValueError('The Hamilton representation is only available'
                             ' for models in which there is no differencing'
                             ' integrated into the state vector. Set'
                             ' `simple_differencing` to True or set'
                             ' `hamilton_representation` to False')

        # Model order
        # (this is used internally in a number of locations)
        self._k_order = max(self.k_ar + self.k_seasonal_ar,
                            self.k_ma + self.k_seasonal_ma + 1)
        if self._k_order == 1 and self.k_ar + self.k_seasonal_ar == 0:
            # Handle time-varying regression
            if self.time_varying_regression:
                self._k_order = 0

        # Exogenous data
        (self._k_exog, exog) = prepare_exog(exog)
        # (we internally use _k_exog for mechanics so that the public attribute
        # can be overridden by subclasses)
        self.k_exog = self._k_exog

        # Redefine mle_regression to be true only if it was previously set to
        # true and there are exogenous regressors
        self.mle_regression = (
            self.mle_regression and exog is not None and self._k_exog > 0
        )
        # State regression is regression with coefficients estimated within
        # the state vector
        self.state_regression = (
            not self.mle_regression and exog is not None and self._k_exog > 0
        )
        # If all we have is a regression (so k_ar = k_ma = 0), then put the
        # error term as measurement error
        if self.state_regression and self._k_order == 0:
            self.measurement_error = True

        # Number of states
        k_states = self._k_order
        if not self.simple_differencing:
            k_states += (self.seasonal_periods * self._k_seasonal_diff +
                         self._k_diff)
        if self.state_regression:
            k_states += self._k_exog

        # Number of positive definite elements of the state covariance matrix
        k_posdef = int(self._k_order > 0)
        # Only have an error component to the states if k_posdef > 0
        self.state_error = k_posdef > 0
        if self.state_regression and self.time_varying_regression:
            k_posdef += self._k_exog

        # Diffuse initialization can be more sensistive to the variance value
        # in the case of state regression, so set a higher than usual default
        # variance
        if self.state_regression:
            kwargs.setdefault('initial_variance', 1e10)

        # Handle non-default loglikelihood burn
        self._loglikelihood_burn = kwargs.get('loglikelihood_burn', None)

        # Number of parameters
        self.k_params = (
            self.k_ar_params + self.k_ma_params +
            self.k_seasonal_ar_params + self.k_seasonal_ma_params +
            self._k_trend +
            self.measurement_error +
            int(not self.concentrate_scale)
        )
        if self.mle_regression:
            self.k_params += self._k_exog

        # We need to have an array or pandas at this point
        self.orig_endog = endog
        self.orig_exog = exog
        if not _is_using_pandas(endog, None):
            endog = np.asanyarray(endog)

        # Update the differencing dimensions if simple differencing is applied
        self.orig_k_diff = self._k_diff
        self.orig_k_seasonal_diff = self._k_seasonal_diff
        if (self.simple_differencing and
           (self._k_diff > 0 or self._k_seasonal_diff > 0)):
            self._k_diff = 0
            self._k_seasonal_diff = 0

        # Internally used in several locations
        self._k_states_diff = (
            self._k_diff + self.seasonal_periods * self._k_seasonal_diff
        )

        # Set some model variables now so they will be available for the
        # initialize() method, below
        self.nobs = len(endog)
        self.k_states = k_states
        self.k_posdef = k_posdef

        # Initialize the statespace
        super(SARIMAX, self).__init__(
            endog, exog=exog, k_states=k_states, k_posdef=k_posdef, **kwargs
        )

        # Set the filter to concentrate out the scale if requested
        if self.concentrate_scale:
            self.ssm.filter_concentrated = True

        # Set as time-varying model if we have time-trend or exog
        if self._k_exog > 0 or len(self.polynomial_trend) > 1:
            self.ssm._time_invariant = False

        # Initialize the fixed components of the statespace model
        self.ssm['design'] = self.initial_design
        self.ssm['state_intercept'] = self.initial_state_intercept
        self.ssm['transition'] = self.initial_transition
        self.ssm['selection'] = self.initial_selection
        if self.concentrate_scale:
            self.ssm['state_cov', 0, 0] = 1.

        # update _init_keys attached by super
        self._init_keys += ['order', 'seasonal_order', 'trend',
                            'measurement_error', 'time_varying_regression',
                            'mle_regression', 'simple_differencing',
                            'enforce_stationarity', 'enforce_invertibility',
                            'hamilton_representation', 'concentrate_scale',
                            'trend_offset'] + list(kwargs.keys())
        # TODO: I think the kwargs or not attached, need to recover from ???

        # Initialize the state
        if self.ssm.initialization is None:
            self.initialize_default()

    def prepare_data(self):
        endog, exog = super(SARIMAX, self).prepare_data()

        # Perform simple differencing if requested
        if (self.simple_differencing and
           (self.orig_k_diff > 0 or self.orig_k_seasonal_diff > 0)):
            # Save the original length
            orig_length = endog.shape[0]
            # Perform simple differencing
            endog = diff(endog.copy(), self.orig_k_diff,
                         self.orig_k_seasonal_diff, self.seasonal_periods)
            if exog is not None:
                exog = diff(exog.copy(), self.orig_k_diff,
                            self.orig_k_seasonal_diff, self.seasonal_periods)

            # Reset the ModelData datasets and cache
            self.data.endog, self.data.exog = (
                self.data._convert_endog_exog(endog, exog))

            # Reset indexes, if provided
            new_length = self.data.endog.shape[0]
            if self.data.row_labels is not None:
                self.data._cache['row_labels'] = (
                    self.data.row_labels[orig_length - new_length:])
            if self._index is not None:
                if self._index_int64:
                    self._index = pd.RangeIndex(start=1, stop=new_length + 1)
                elif self._index_generated:
                    self._index = self._index[:-(orig_length - new_length)]
                else:
                    self._index = self._index[orig_length - new_length:]

        # Reset the nobs
        self.nobs = endog.shape[0]

        # Cache the arrays for calculating the intercept from the trend
        # components
        self._trend_data = prepare_trend_data(
            self.polynomial_trend, self._k_trend, self.nobs, self.trend_offset)

        return endog, exog

    def initialize(self):
        """
        Initialize the SARIMAX model.

        Notes
        -----
        These initialization steps must occur following the parent class
        __init__ function calls.
        """
        super(SARIMAX, self).initialize()

        # Cache the indexes of included polynomial orders (for update below)
        # (but we do not want the index of the constant term, so exclude the
        # first index)
        self._polynomial_ar_idx = np.nonzero(self.polynomial_ar)[0][1:]
        self._polynomial_ma_idx = np.nonzero(self.polynomial_ma)[0][1:]
        self._polynomial_seasonal_ar_idx = np.nonzero(
            self.polynomial_seasonal_ar
        )[0][1:]
        self._polynomial_seasonal_ma_idx = np.nonzero(
            self.polynomial_seasonal_ma
        )[0][1:]

        # Save the indices corresponding to the reduced form lag polynomial
        # parameters in the transition and selection matrices so that they
        # do not have to be recalculated for each update()
        start_row = self._k_states_diff
        end_row = start_row + self.k_ar + self.k_seasonal_ar
        col = self._k_states_diff
        if not self.hamilton_representation:
            self.transition_ar_params_idx = (
                np.s_['transition', start_row:end_row, col]
            )
        else:
            self.transition_ar_params_idx = (
                np.s_['transition', col, start_row:end_row]
            )

        start_row += 1
        end_row = start_row + self.k_ma + self.k_seasonal_ma
        col = 0
        if not self.hamilton_representation:
            self.selection_ma_params_idx = (
                np.s_['selection', start_row:end_row, col]
            )
        else:
            self.design_ma_params_idx = (
                np.s_['design', col, start_row:end_row]
            )

        # Cache indices for exog variances in the state covariance matrix
        if self.state_regression and self.time_varying_regression:
            idx = np.diag_indices(self.k_posdef)
            self._exog_variance_idx = ('state_cov', idx[0][-self._k_exog:],
                                       idx[1][-self._k_exog:])

    def initialize_default(self, approximate_diffuse_variance=None):
        """Initialize default"""
        if approximate_diffuse_variance is None:
            approximate_diffuse_variance = self.ssm.initial_variance
        if self.use_exact_diffuse:
            diffuse_type = 'diffuse'
        else:
            diffuse_type = 'approximate_diffuse'

            # Set the loglikelihood burn parameter, if not given in constructor
            if self._loglikelihood_burn is None:
                k_diffuse_states = self.k_states
                if self.enforce_stationarity:
                    k_diffuse_states -= self._k_order
                self.loglikelihood_burn = k_diffuse_states

        init = Initialization(
            self.k_states,
            approximate_diffuse_variance=approximate_diffuse_variance)

        if self.enforce_stationarity:
            # Differencing operators are at the beginning
            init.set((0, self._k_states_diff), diffuse_type)
            # Stationary component in the middle
            init.set((self._k_states_diff,
                      self._k_states_diff + self._k_order),
                     'stationary')
            # Regression components at the end
            init.set((self._k_states_diff + self._k_order,
                      self._k_states_diff + self._k_order + self._k_exog),
                     diffuse_type)
        # If we're not enforcing a stationarity, then we cannot initialize a
        # stationary component
        else:
            init.set(None, diffuse_type)

        self.ssm.initialization = init

    @property
    def initial_design(self):
        """Initial design matrix"""
        # Basic design matrix
        design = np.r_[
            [1] * self._k_diff,
            ([0] * (self.seasonal_periods - 1) + [1]) * self._k_seasonal_diff,
            [1] * self.state_error, [0] * (self._k_order - 1)
        ]

        if len(design) == 0:
            design = np.r_[0]

        # If we have exogenous regressors included as part of the state vector
        # then the exogenous data is incorporated as a time-varying component
        # of the design matrix
        if self.state_regression:
            if self._k_order > 0:
                design = np.c_[
                    np.reshape(
                        np.repeat(design, self.nobs),
                        (design.shape[0], self.nobs)
                    ).T,
                    self.exog
                ].T[None, :, :]
            else:
                design = self.exog.T[None, :, :]
        return design

    @property
    def initial_state_intercept(self):
        """Initial state intercept vector"""
        # TODO make this self._k_trend > 1 and adjust the update to take
        # into account that if the trend is a constant, it is not time-varying
        if self._k_trend > 0:
            state_intercept = np.zeros((self.k_states, self.nobs))
        else:
            state_intercept = np.zeros((self.k_states,))
        return state_intercept

    @property
    def initial_transition(self):
        """Initial transition matrix"""
        transition = np.zeros((self.k_states, self.k_states))

        # Exogenous regressors component
        if self.state_regression:
            start = -self._k_exog
            # T_\beta
            transition[start:, start:] = np.eye(self._k_exog)

            # Autoregressive component
            start = -(self._k_exog + self._k_order)
            end = -self._k_exog if self._k_exog > 0 else None
        else:
            # Autoregressive component
            start = -self._k_order
            end = None

        # T_c
        if self._k_order > 0:
            transition[start:end, start:end] = companion_matrix(self._k_order)
            if self.hamilton_representation:
                transition[start:end, start:end] = np.transpose(
                    companion_matrix(self._k_order)
                )

        # Seasonal differencing component
        # T^*
        if self._k_seasonal_diff > 0:
            seasonal_companion = companion_matrix(self.seasonal_periods).T
            seasonal_companion[0, -1] = 1
            for d in range(self._k_seasonal_diff):
                start = self._k_diff + d * self.seasonal_periods
                end = self._k_diff + (d + 1) * self.seasonal_periods

                # T_c^*
                transition[start:end, start:end] = seasonal_companion

                # i
                if d < self._k_seasonal_diff - 1:
                    transition[start, end + self.seasonal_periods - 1] = 1

                # \iota
                transition[start, self._k_states_diff] = 1

        # Differencing component
        if self._k_diff > 0:
            idx = np.triu_indices(self._k_diff)
            # T^**
            transition[idx] = 1
            # [0 1]
            if self.seasonal_periods > 0:
                start = self._k_diff
                end = self._k_states_diff
                transition[:self._k_diff, start:end] = (
                    ([0] * (self.seasonal_periods - 1) + [1]) *
                    self._k_seasonal_diff)
            # [1 0]
            column = self._k_states_diff
            transition[:self._k_diff, column] = 1

        return transition

    @property
    def initial_selection(self):
        """Initial selection matrix"""
        if not (self.state_regression and self.time_varying_regression):
            if self.k_posdef > 0:
                selection = np.r_[
                    [0] * (self._k_states_diff),
                    [1] * (self._k_order > 0), [0] * (self._k_order - 1),
                    [0] * ((1 - self.mle_regression) * self._k_exog)
                ][:, None]

                if len(selection) == 0:
                    selection = np.zeros((self.k_states, self.k_posdef))
            else:
                selection = np.zeros((self.k_states, 0))
        else:
            selection = np.zeros((self.k_states, self.k_posdef))
            # Typical state variance
            if self._k_order > 0:
                selection[0, 0] = 1
            # Time-varying regression coefficient variances
            for i in range(self._k_exog, 0, -1):
                selection[-i, -i] = 1
        return selection

    def clone(self, endog, exog=None, **kwargs):
        return self._clone_from_init_kwds(endog, exog=exog, **kwargs)

    @property
    def _res_classes(self):
        return {'fit': (SARIMAXResults, SARIMAXResultsWrapper)}

    @staticmethod
    def _conditional_sum_squares(endog, k_ar, polynomial_ar, k_ma,
                                 polynomial_ma, k_trend=0, trend_data=None,
                                 warning_description=None):
        k = 2 * k_ma
        r = max(k + k_ma, k_ar)

        k_params_ar = 0 if k_ar == 0 else len(polynomial_ar.nonzero()[0]) - 1
        k_params_ma = 0 if k_ma == 0 else len(polynomial_ma.nonzero()[0]) - 1

        residuals = None
        if k_ar + k_ma + k_trend > 0:
            try:
                # If we have MA terms, get residuals from an AR(k) model to use
                # as data for conditional sum of squares estimates of the MA
                # parameters
                if k_ma > 0:
                    Y = endog[k:]
                    X = lagmat(endog, k, trim='both')
                    params_ar = np.linalg.pinv(X).dot(Y)
                    residuals = Y - np.dot(X, params_ar)

                # Run an ARMA(p,q) model using the just computed residuals as
                # data
                Y = endog[r:]

                X = np.empty((Y.shape[0], 0))
                if k_trend > 0:
                    if trend_data is None:
                        raise ValueError('Trend data must be provided if'
                                         ' `k_trend` > 0.')
                    X = np.c_[X, trend_data[:(-r if r > 0 else None), :]]
                if k_ar > 0:
                    cols = polynomial_ar.nonzero()[0][1:] - 1
                    X = np.c_[X, lagmat(endog, k_ar)[r:, cols]]
                if k_ma > 0:
                    cols = polynomial_ma.nonzero()[0][1:] - 1
                    X = np.c_[X, lagmat(residuals, k_ma)[r-k:, cols]]

                # Get the array of [ar_params, ma_params]
                params = np.linalg.pinv(X).dot(Y)
                residuals = Y - np.dot(X, params)
            except ValueError:
                if warning_description is not None:
                    warning_description = ' for %s' % warning_description
                else:
                    warning_description = ''
                warn('Too few observations to estimate starting parameters%s.'
                     ' All parameters except for variances will be set to'
                     ' zeros.' % warning_description)
                # Typically this will be raised if there are not enough
                # observations for the `lagmat` calls.
                params = np.zeros(k_trend + k_ar + k_ma, dtype=endog.dtype)
                if len(endog) == 0:
                    # This case usually happens when there are not even enough
                    # observations for a complete set of differencing
                    # operations (no hope of fitting, just set starting
                    # variance to 1)
                    residuals = np.ones(k_params_ma * 2 + 1, dtype=endog.dtype)
                else:
                    residuals = np.r_[
                        np.zeros(k_params_ma * 2, dtype=endog.dtype),
                        endog - np.mean(endog)]

        # Default output
        params_trend = []
        params_ar = []
        params_ma = []
        params_variance = []

        # Get the params
        offset = 0
        if k_trend > 0:
            params_trend = params[offset:k_trend + offset]
            offset += k_trend
        if k_ar > 0:
            params_ar = params[offset:k_params_ar + offset]
            offset += k_params_ar
        if k_ma > 0:
            params_ma = params[offset:k_params_ma + offset]
            offset += k_params_ma
        if residuals is not None:
            if len(residuals) > max(1, k_params_ma):
                params_variance = (residuals[k_params_ma:] ** 2).mean()
            else:
                params_variance = np.var(endog)

        return (params_trend, params_ar, params_ma,
                params_variance)

    @property
    def start_params(self):
        """
        Starting parameters for maximum likelihood estimation
        """

        # Perform differencing if necessary (i.e. if simple differencing is
        # false so that the state-space model will use the entire dataset)
        trend_data = self._trend_data
        if not self.simple_differencing and (
           self._k_diff > 0 or self._k_seasonal_diff > 0):
            endog = diff(self.endog, self._k_diff,
                         self._k_seasonal_diff, self.seasonal_periods)
            if self.exog is not None:
                exog = diff(self.exog, self._k_diff,
                            self._k_seasonal_diff, self.seasonal_periods)
            else:
                exog = None
            trend_data = trend_data[:endog.shape[0], :]
        else:
            endog = self.endog.copy()
            exog = self.exog.copy() if self.exog is not None else None
        endog = endog.squeeze()

        # Although the Kalman filter can deal with missing values in endog,
        # conditional sum of squares cannot
        if np.any(np.isnan(endog)):
            mask = ~np.isnan(endog).squeeze()
            endog = endog[mask]
            if exog is not None:
                exog = exog[mask]
            if trend_data is not None:
                trend_data = trend_data[mask]

        # Regression effects via OLS
        params_exog = []
        if self._k_exog > 0:
            params_exog = np.linalg.pinv(exog).dot(endog)
            endog = endog - np.dot(exog, params_exog)
        if self.state_regression:
            params_exog = []

        # Non-seasonal ARMA component and trend
        (params_trend, params_ar, params_ma,
         params_variance) = self._conditional_sum_squares(
            endog, self.k_ar, self.polynomial_ar, self.k_ma,
            self.polynomial_ma, self._k_trend, trend_data,
            warning_description='ARMA and trend')

        # If we have estimated non-stationary start parameters but enforce
        # stationarity is on, start with 0 parameters and warn
        invalid_ar = (
            self.k_ar > 0 and
            self.enforce_stationarity and
            not is_invertible(np.r_[1, -params_ar])
        )
        if invalid_ar:
            warn('Non-stationary starting autoregressive parameters'
                 ' found. Using zeros as starting parameters.')
            params_ar *= 0

        # If we have estimated non-invertible start parameters but enforce
        # invertibility is on, raise an error
        invalid_ma = (
            self.k_ma > 0 and
            self.enforce_invertibility and
            not is_invertible(np.r_[1, params_ma])
        )
        if invalid_ma:
            warn('Non-invertible starting MA parameters found.'
                 ' Using zeros as starting parameters.')
            params_ma *= 0

        # Seasonal Parameters
        _, params_seasonal_ar, params_seasonal_ma, params_seasonal_variance = (
            self._conditional_sum_squares(
                endog, self.k_seasonal_ar, self.polynomial_seasonal_ar,
                self.k_seasonal_ma, self.polynomial_seasonal_ma,
                warning_description='seasonal ARMA'))

        # If we have estimated non-stationary start parameters but enforce
        # stationarity is on, warn and set start params to 0
        invalid_seasonal_ar = (
            self.k_seasonal_ar > 0 and
            self.enforce_stationarity and
            not is_invertible(np.r_[1, -params_seasonal_ar])
        )
        if invalid_seasonal_ar:
            warn('Non-stationary starting seasonal autoregressive'
                 ' Using zeros as starting parameters.')
            params_seasonal_ar *= 0

        # If we have estimated non-invertible start parameters but enforce
        # invertibility is on, raise an error
        invalid_seasonal_ma = (
            self.k_seasonal_ma > 0 and
            self.enforce_invertibility and
            not is_invertible(np.r_[1, params_seasonal_ma])
        )
        if invalid_seasonal_ma:
            warn('Non-invertible starting seasonal moving average'
                 ' Using zeros as starting parameters.')
            params_seasonal_ma *= 0

        # Variances
        params_exog_variance = []
        if self.state_regression and self.time_varying_regression:
            # TODO how to set the initial variance parameters?
            params_exog_variance = [1] * self._k_exog
        if (self.state_error and type(params_variance) == list and
                len(params_variance) == 0):
            if not (type(params_seasonal_variance) == list and
                    len(params_seasonal_variance) == 0):
                params_variance = params_seasonal_variance
            elif self._k_exog > 0:
                params_variance = np.inner(endog, endog)
            else:
                params_variance = np.inner(endog, endog) / self.nobs
        params_measurement_variance = 1 if self.measurement_error else []

        # We want to bound the starting variance away from zero
        params_variance = np.atleast_1d(max(np.array(params_variance), 1e-10))

        # Remove state variance as parameter if scale is concentrated out
        if self.concentrate_scale:
            params_variance = []

        # Combine all parameters
        return np.r_[
            params_trend,
            params_exog,
            params_ar,
            params_ma,
            params_seasonal_ar,
            params_seasonal_ma,
            params_exog_variance,
            params_measurement_variance,
            params_variance
        ]

    @property
    def endog_names(self, latex=False):
        """Names of endogenous variables"""
        diff = ''
        if self.k_diff > 0:
            if self.k_diff == 1:
                diff = r'\Delta' if latex else 'D'
            else:
                diff = (r'\Delta^%d' if latex else 'D%d') % self.k_diff

        seasonal_diff = ''
        if self.k_seasonal_diff > 0:
            if self.k_seasonal_diff == 1:
                seasonal_diff = ((r'\Delta_%d' if latex else 'DS%d') %
                                 (self.seasonal_periods))
            else:
                seasonal_diff = ((r'\Delta_%d^%d' if latex else 'D%dS%d') %
                                 (self.k_seasonal_diff, self.seasonal_periods))
        endog_diff = self.simple_differencing
        if endog_diff and self.k_diff > 0 and self.k_seasonal_diff > 0:
            return (('%s%s %s' if latex else '%s.%s.%s') %
                    (diff, seasonal_diff, self.data.ynames))
        elif endog_diff and self.k_diff > 0:
            return (('%s %s' if latex else '%s.%s') %
                    (diff, self.data.ynames))
        elif endog_diff and self.k_seasonal_diff > 0:
            return (('%s %s' if latex else '%s.%s') %
                    (seasonal_diff, self.data.ynames))
        else:
            return self.data.ynames

    params_complete = [
        'trend', 'exog', 'ar', 'ma', 'seasonal_ar', 'seasonal_ma',
        'exog_variance', 'measurement_variance', 'variance'
    ]

    @property
    def param_terms(self):
        """
        List of parameters actually included in the model, in sorted order.

        TODO Make this an dict with slice or indices as the values.
        """
        model_orders = self.model_orders
        # Get basic list from model orders
        params = [
            order for order in self.params_complete
            if model_orders[order] > 0
        ]
        # k_exog may be positive without associated parameters if it is in the
        # state vector
        if 'exog' in params and not self.mle_regression:
            params.remove('exog')

        return params

    @property
    def param_names(self):
        """
        List of human readable parameter names (for parameters actually
        included in the model).
        """
        params_sort_order = self.param_terms
        model_names = self.model_names
        return [
            name for param in params_sort_order for name in model_names[param]
        ]

    @property
    def state_names(self):
        # TODO: we may be able to revisit these states to get somewhat more
        # informative names, but ultimately probably not much better.
        # TODO: alternatively, we may be able to get better for certain models,
        # like pure AR models.
        k_ar_states = self._k_order
        if not self.simple_differencing:
            k_ar_states += (self.seasonal_periods * self._k_seasonal_diff +
                            self._k_diff)
        names = ['state.%d' % i for i in range(k_ar_states)]

        if self._k_exog > 0 and self.state_regression:
            names += ['beta.%s' % self.exog_names[i]
                      for i in range(self._k_exog)]

        return names

    @property
    def model_orders(self):
        """
        The orders of each of the polynomials in the model.
        """
        return {
            'trend': self._k_trend,
            'exog': self._k_exog,
            'ar': self.k_ar,
            'ma': self.k_ma,
            'seasonal_ar': self.k_seasonal_ar,
            'seasonal_ma': self.k_seasonal_ma,
            'reduced_ar': self.k_ar + self.k_seasonal_ar,
            'reduced_ma': self.k_ma + self.k_seasonal_ma,
            'exog_variance': self._k_exog if (
                self.state_regression and self.time_varying_regression) else 0,
            'measurement_variance': int(self.measurement_error),
            'variance': int(self.state_error and not self.concentrate_scale),
        }

    @property
    def model_names(self):
        """
        The plain text names of all possible model parameters.
        """
        return self._get_model_names(latex=False)

    @property
    def model_latex_names(self):
        """
        The latex names of all possible model parameters.
        """
        return self._get_model_names(latex=True)

    def _get_model_names(self, latex=False):
        names = {
            'trend': None,
            'exog': None,
            'ar': None,
            'ma': None,
            'seasonal_ar': None,
            'seasonal_ma': None,
            'reduced_ar': None,
            'reduced_ma': None,
            'exog_variance': None,
            'measurement_variance': None,
            'variance': None,
        }

        # Trend
        if self._k_trend > 0:
            trend_template = 't_%d' if latex else 'trend.%d'
            names['trend'] = []
            for i in self.polynomial_trend.nonzero()[0]:
                if i == 0:
                    names['trend'].append('intercept')
                elif i == 1:
                    names['trend'].append('drift')
                else:
                    names['trend'].append(trend_template % i)

        # Exogenous coefficients
        if self._k_exog > 0:
            names['exog'] = self.exog_names

        # Autoregressive
        if self.k_ar > 0:
            ar_template = '$\\phi_%d$' if latex else 'ar.L%d'
            names['ar'] = []
            for i in self.polynomial_ar.nonzero()[0][1:]:
                names['ar'].append(ar_template % i)

        # Moving Average
        if self.k_ma > 0:
            ma_template = '$\\theta_%d$' if latex else 'ma.L%d'
            names['ma'] = []
            for i in self.polynomial_ma.nonzero()[0][1:]:
                names['ma'].append(ma_template % i)

        # Seasonal Autoregressive
        if self.k_seasonal_ar > 0:
            seasonal_ar_template = (
                '$\\tilde \\phi_%d$' if latex else 'ar.S.L%d'
            )
            names['seasonal_ar'] = []
            for i in self.polynomial_seasonal_ar.nonzero()[0][1:]:
                names['seasonal_ar'].append(seasonal_ar_template % i)

        # Seasonal Moving Average
        if self.k_seasonal_ma > 0:
            seasonal_ma_template = (
                '$\\tilde \\theta_%d$' if latex else 'ma.S.L%d'
            )
            names['seasonal_ma'] = []
            for i in self.polynomial_seasonal_ma.nonzero()[0][1:]:
                names['seasonal_ma'].append(seasonal_ma_template % i)

        # Reduced Form Autoregressive
        if self.k_ar > 0 or self.k_seasonal_ar > 0:
            reduced_polynomial_ar = reduced_polynomial_ar = -np.polymul(
                self.polynomial_ar, self.polynomial_seasonal_ar
            )
            ar_template = '$\\Phi_%d$' if latex else 'ar.R.L%d'
            names['reduced_ar'] = []
            for i in reduced_polynomial_ar.nonzero()[0][1:]:
                names['reduced_ar'].append(ar_template % i)

        # Reduced Form Moving Average
        if self.k_ma > 0 or self.k_seasonal_ma > 0:
            reduced_polynomial_ma = np.polymul(
                self.polynomial_ma, self.polynomial_seasonal_ma
            )
            ma_template = '$\\Theta_%d$' if latex else 'ma.R.L%d'
            names['reduced_ma'] = []
            for i in reduced_polynomial_ma.nonzero()[0][1:]:
                names['reduced_ma'].append(ma_template % i)

        # Exogenous variances
        if self.state_regression and self.time_varying_regression:
            if not self.concentrate_scale:
                exog_var_template = ('$\\sigma_\\text{%s}^2$' if latex
                                     else 'var.%s')
            else:
                exog_var_template = (
                    '$\\sigma_\\text{%s}^2 / \\sigma_\\zeta^2$' if latex
                    else 'snr.%s')
            names['exog_variance'] = [
                exog_var_template % exog_name for exog_name in self.exog_names
            ]

        # Measurement error variance
        if self.measurement_error:
            if not self.concentrate_scale:
                meas_var_tpl = (
                    '$\\sigma_\\eta^2$' if latex else 'var.measurement_error')
            else:
                meas_var_tpl = (
                    '$\\sigma_\\eta^2 / \\sigma_\\zeta^2$' if latex
                    else 'snr.measurement_error')
            names['measurement_variance'] = [meas_var_tpl]

        # State variance
        if self.state_error and not self.concentrate_scale:
            var_tpl = '$\\sigma_\\zeta^2$' if latex else 'sigma2'
            names['variance'] = [var_tpl]

        return names

    def transform_params(self, unconstrained):
        """
        Transform unconstrained parameters used by the optimizer to constrained
        parameters used in likelihood evaluation.

        Used primarily to enforce stationarity of the autoregressive lag
        polynomial, invertibility of the moving average lag polynomial, and
        positive variance parameters.

        Parameters
        ----------
        unconstrained : array_like
            Unconstrained parameters used by the optimizer.

        Returns
        -------
        constrained : array_like
            Constrained parameters used in likelihood evaluation.

        Notes
        -----
        If the lag polynomial has non-consecutive powers (so that the
        coefficient is zero on some element of the polynomial), then the
        constraint function is not onto the entire space of invertible
        polynomials, although it only excludes a very small portion very close
        to the invertibility boundary.
        """
        unconstrained = np.array(unconstrained, ndmin=1)
        constrained = np.zeros(unconstrained.shape, unconstrained.dtype)

        start = end = 0

        # Retain the trend parameters
        if self._k_trend > 0:
            end += self._k_trend
            constrained[start:end] = unconstrained[start:end]
            start += self._k_trend

        # Retain any MLE regression coefficients
        if self.mle_regression:
            end += self._k_exog
            constrained[start:end] = unconstrained[start:end]
            start += self._k_exog

        # Transform the AR parameters (phi) to be stationary
        if self.k_ar_params > 0:
            end += self.k_ar_params
            if self.enforce_stationarity:
                constrained[start:end] = (
                    constrain_stationary_univariate(unconstrained[start:end])
                )
            else:
                constrained[start:end] = unconstrained[start:end]
            start += self.k_ar_params

        # Transform the MA parameters (theta) to be invertible
        if self.k_ma_params > 0:
            end += self.k_ma_params
            if self.enforce_invertibility:
                constrained[start:end] = (
                    -constrain_stationary_univariate(unconstrained[start:end])
                )
            else:
                constrained[start:end] = unconstrained[start:end]
            start += self.k_ma_params

        # Transform the seasonal AR parameters (\tilde phi) to be stationary
        if self.k_seasonal_ar > 0:
            end += self.k_seasonal_ar_params
            if self.enforce_stationarity:
                constrained[start:end] = (
                    constrain_stationary_univariate(unconstrained[start:end])
                )
            else:
                constrained[start:end] = unconstrained[start:end]
            start += self.k_seasonal_ar_params

        # Transform the seasonal MA parameters (\tilde theta) to be invertible
        if self.k_seasonal_ma_params > 0:
            end += self.k_seasonal_ma_params
            if self.enforce_invertibility:
                constrained[start:end] = (
                    -constrain_stationary_univariate(unconstrained[start:end])
                )
            else:
                constrained[start:end] = unconstrained[start:end]
            start += self.k_seasonal_ma_params

        # Transform the standard deviation parameters to be positive
        if self.state_regression and self.time_varying_regression:
            end += self._k_exog
            constrained[start:end] = unconstrained[start:end]**2
            start += self._k_exog
        if self.measurement_error:
            constrained[start] = unconstrained[start]**2
            start += 1
            end += 1
        if self.state_error and not self.concentrate_scale:
            constrained[start] = unconstrained[start]**2
            # start += 1
            # end += 1

        return constrained

    def untransform_params(self, constrained):
        """
        Transform constrained parameters used in likelihood evaluation
        to unconstrained parameters used by the optimizer

        Used primarily to reverse enforcement of stationarity of the
        autoregressive lag polynomial and invertibility of the moving average
        lag polynomial.

        Parameters
        ----------
        constrained : array_like
            Constrained parameters used in likelihood evaluation.

        Returns
        -------
        constrained : array_like
            Unconstrained parameters used by the optimizer.

        Notes
        -----
        If the lag polynomial has non-consecutive powers (so that the
        coefficient is zero on some element of the polynomial), then the
        constraint function is not onto the entire space of invertible
        polynomials, although it only excludes a very small portion very close
        to the invertibility boundary.
        """
        constrained = np.array(constrained, ndmin=1)
        unconstrained = np.zeros(constrained.shape, constrained.dtype)

        start = end = 0

        # Retain the trend parameters
        if self._k_trend > 0:
            end += self._k_trend
            unconstrained[start:end] = constrained[start:end]
            start += self._k_trend

        # Retain any MLE regression coefficients
        if self.mle_regression:
            end += self._k_exog
            unconstrained[start:end] = constrained[start:end]
            start += self._k_exog

        # Transform the AR parameters (phi) to be stationary
        if self.k_ar_params > 0:
            end += self.k_ar_params
            if self.enforce_stationarity:
                unconstrained[start:end] = (
                    unconstrain_stationary_univariate(constrained[start:end])
                )
            else:
                unconstrained[start:end] = constrained[start:end]
            start += self.k_ar_params

        # Transform the MA parameters (theta) to be invertible
        if self.k_ma_params > 0:
            end += self.k_ma_params
            if self.enforce_invertibility:
                unconstrained[start:end] = (
                    unconstrain_stationary_univariate(-constrained[start:end])
                )
            else:
                unconstrained[start:end] = constrained[start:end]
            start += self.k_ma_params

        # Transform the seasonal AR parameters (\tilde phi) to be stationary
        if self.k_seasonal_ar > 0:
            end += self.k_seasonal_ar_params
            if self.enforce_stationarity:
                unconstrained[start:end] = (
                    unconstrain_stationary_univariate(constrained[start:end])
                )
            else:
                unconstrained[start:end] = constrained[start:end]
            start += self.k_seasonal_ar_params

        # Transform the seasonal MA parameters (\tilde theta) to be invertible
        if self.k_seasonal_ma_params > 0:
            end += self.k_seasonal_ma_params
            if self.enforce_invertibility:
                unconstrained[start:end] = (
                    unconstrain_stationary_univariate(-constrained[start:end])
                )
            else:
                unconstrained[start:end] = constrained[start:end]
            start += self.k_seasonal_ma_params

        # Untransform the standard deviation
        if self.state_regression and self.time_varying_regression:
            end += self._k_exog
            unconstrained[start:end] = constrained[start:end]**0.5
            start += self._k_exog
        if self.measurement_error:
            unconstrained[start] = constrained[start]**0.5
            start += 1
            end += 1
        if self.state_error and not self.concentrate_scale:
            unconstrained[start] = constrained[start]**0.5
            # start += 1
            # end += 1

        return unconstrained

    def _validate_can_fix_params(self, param_names):
        super(SARIMAX, self)._validate_can_fix_params(param_names)
        model_names = self.model_names

        items = [
            ('ar', 'autoregressive', self.enforce_stationarity,
                '`enforce_stationarity=True`'),
            ('seasonal_ar', 'seasonal autoregressive',
                self.enforce_stationarity, '`enforce_stationarity=True`'),
            ('ma', 'moving average', self.enforce_invertibility,
                '`enforce_invertibility=True`'),
            ('seasonal_ma', 'seasonal moving average',
                self.enforce_invertibility, '`enforce_invertibility=True`')]

        for name, title, condition, condition_desc in items:
            names = set(model_names[name] or [])
            fix_all = param_names.issuperset(names)
            fix_any = len(param_names.intersection(names)) > 0
            if condition and fix_any and not fix_all:
                raise ValueError('Cannot fix individual %s parameters when'
                                 ' %s. Must either fix all %s parameters or'
                                 ' none.' % (title, condition_desc, title))

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
        """
        params = self.handle_params(params, transformed=transformed,
                                    includes_fixed=includes_fixed)

        params_trend = None
        params_exog = None
        params_ar = None
        params_ma = None
        params_seasonal_ar = None
        params_seasonal_ma = None
        params_exog_variance = None
        params_measurement_variance = None
        params_variance = None

        # Extract the parameters
        start = end = 0
        end += self._k_trend
        params_trend = params[start:end]
        start += self._k_trend
        if self.mle_regression:
            end += self._k_exog
            params_exog = params[start:end]
            start += self._k_exog
        end += self.k_ar_params
        params_ar = params[start:end]
        start += self.k_ar_params
        end += self.k_ma_params
        params_ma = params[start:end]
        start += self.k_ma_params
        end += self.k_seasonal_ar_params
        params_seasonal_ar = params[start:end]
        start += self.k_seasonal_ar_params
        end += self.k_seasonal_ma_params
        params_seasonal_ma = params[start:end]
        start += self.k_seasonal_ma_params
        if self.state_regression and self.time_varying_regression:
            end += self._k_exog
            params_exog_variance = params[start:end]
            start += self._k_exog
        if self.measurement_error:
            params_measurement_variance = params[start]
            start += 1
            end += 1
        if self.state_error and not self.concentrate_scale:
            params_variance = params[start]
        # start += 1
        # end += 1

        # Update lag polynomials
        if self.k_ar > 0:
            if self._polynomial_ar.dtype == params.dtype:
                self._polynomial_ar[self._polynomial_ar_idx] = -params_ar
            else:
                polynomial_ar = self._polynomial_ar.real.astype(params.dtype)
                polynomial_ar[self._polynomial_ar_idx] = -params_ar
                self._polynomial_ar = polynomial_ar

        if self.k_ma > 0:
            if self._polynomial_ma.dtype == params.dtype:
                self._polynomial_ma[self._polynomial_ma_idx] = params_ma
            else:
                polynomial_ma = self._polynomial_ma.real.astype(params.dtype)
                polynomial_ma[self._polynomial_ma_idx] = params_ma
                self._polynomial_ma = polynomial_ma

        if self.k_seasonal_ar > 0:
            idx = self._polynomial_seasonal_ar_idx
            if self._polynomial_seasonal_ar.dtype == params.dtype:
                self._polynomial_seasonal_ar[idx] = -params_seasonal_ar
            else:
                polynomial_seasonal_ar = (
                    self._polynomial_seasonal_ar.real.astype(params.dtype)
                )
                polynomial_seasonal_ar[idx] = -params_seasonal_ar
                self._polynomial_seasonal_ar = polynomial_seasonal_ar

        if self.k_seasonal_ma > 0:
            idx = self._polynomial_seasonal_ma_idx
            if self._polynomial_seasonal_ma.dtype == params.dtype:
                self._polynomial_seasonal_ma[idx] = params_seasonal_ma
            else:
                polynomial_seasonal_ma = (
                    self._polynomial_seasonal_ma.real.astype(params.dtype)
                )
                polynomial_seasonal_ma[idx] = params_seasonal_ma
                self._polynomial_seasonal_ma = polynomial_seasonal_ma

        # Get the reduced form lag polynomial terms by multiplying the regular
        # and seasonal lag polynomials
        # Note: that although the numpy np.polymul examples assume that they
        # are ordered from highest degree to lowest, whereas our are from
        # lowest to highest, it does not matter.
        if self.k_seasonal_ar > 0:
            reduced_polynomial_ar = -np.polymul(
                self._polynomial_ar, self._polynomial_seasonal_ar
            )
        else:
            reduced_polynomial_ar = -self._polynomial_ar
        if self.k_seasonal_ma > 0:
            reduced_polynomial_ma = np.polymul(
                self._polynomial_ma, self._polynomial_seasonal_ma
            )
        else:
            reduced_polynomial_ma = self._polynomial_ma

        # Observation intercept
        # Exogenous data with MLE estimation of parameters enters through a
        # time-varying observation intercept (is equivalent to simply
        # subtracting it out of the endogenous variable first)
        if self.mle_regression:
            self.ssm['obs_intercept'] = np.dot(self.exog, params_exog)[None, :]

        # State intercept (Harvey) or additional observation intercept
        # (Hamilton)
        # SARIMA trend enters through the a time-varying state intercept,
        # associated with the first row of the stationary component of the
        # state vector (i.e. the first element of the state vector following
        # any differencing elements)
        if self._k_trend > 0:
            data = np.dot(self._trend_data, params_trend).astype(params.dtype)
            if not self.hamilton_representation:
                self.ssm['state_intercept', self._k_states_diff, :] = data
            else:
                # The way the trend enters in the Hamilton representation means
                # that the parameter is not an ``intercept'' but instead the
                # mean of the process. The trend values in `data` are meant for
                # an intercept, and so must be transformed to represent the
                # mean instead
                if self.hamilton_representation:
                    data /= np.sum(-reduced_polynomial_ar)

                # If we already set the observation intercept for MLE
                # regression, just add to it
                if self.mle_regression:
                    self.ssm.obs_intercept += data[None, :]
                # Otherwise set it directly
                else:
                    self.ssm['obs_intercept'] = data[None, :]

        # Observation covariance matrix
        if self.measurement_error:
            self.ssm['obs_cov', 0, 0] = params_measurement_variance

        # Transition matrix
        if self.k_ar > 0 or self.k_seasonal_ar > 0:
            self.ssm[self.transition_ar_params_idx] = reduced_polynomial_ar[1:]
        elif not self.ssm.transition.dtype == params.dtype:
            # This is required if the transition matrix is not really in use
            # (e.g. for an MA(q) process) so that it's dtype never changes as
            # the parameters' dtype changes. This changes the dtype manually.
            self.ssm['transition'] = self.ssm['transition'].real.astype(
                params.dtype)

        # Selection matrix (Harvey) or Design matrix (Hamilton)
        if self.k_ma > 0 or self.k_seasonal_ma > 0:
            if not self.hamilton_representation:
                self.ssm[self.selection_ma_params_idx] = (
                    reduced_polynomial_ma[1:]
                )
            else:
                self.ssm[self.design_ma_params_idx] = reduced_polynomial_ma[1:]

        # State covariance matrix
        if self.k_posdef > 0:
            if not self.concentrate_scale:
                self['state_cov', 0, 0] = params_variance
            if self.state_regression and self.time_varying_regression:
                self.ssm[self._exog_variance_idx] = params_exog_variance

        return params

    def _get_extension_time_varying_matrices(
            self, params, exog, out_of_sample, extend_kwargs=None,
            transformed=True, includes_fixed=False, **kwargs):
        """
        Get time-varying state space system matrices for extended model

        Notes
        -----
        We need to override this method for SARIMAX because we need some
        special handling in the `simple_differencing=True` case.
        """

        # Get the appropriate exog for the extended sample
        exog = self._validate_out_of_sample_exog(exog, out_of_sample)

        # Get the tmp endog, exog
        if self.simple_differencing:
            nobs = self.data.orig_endog.shape[0] + out_of_sample
            tmp_endog = np.zeros((nobs, self.k_endog))
            if exog is not None:
                tmp_exog = np.c_[self.data.orig_exog.T, exog.T].T
            else:
                tmp_exog = None
        else:
            tmp_endog = np.zeros((out_of_sample, self.k_endog))
            tmp_exog = exog

        # Create extended model
        if extend_kwargs is None:
            extend_kwargs = {}
        if not self.simple_differencing and self.k_trend > 0:
            extend_kwargs.setdefault(
                'trend_offset', self.trend_offset + self.nobs)
        extend_kwargs.setdefault('validate_specification', False)
        mod_extend = self.clone(
            endog=tmp_endog, exog=tmp_exog, **extend_kwargs)
        mod_extend.update(params, transformed=transformed,
                          includes_fixed=includes_fixed,)

        # Retrieve the extensions to the time-varying system matrices and
        # put them in kwargs
        for name in self.ssm.shapes.keys():
            if name == 'obs' or name in kwargs:
                continue
            original = getattr(self.ssm, name)
            extended = getattr(mod_extend.ssm, name)
            so = original.shape[-1]
            se = extended.shape[-1]
            if ((so > 1 or se > 1) or (
                    so == 1 and self.nobs == 1 and
                    np.any(original[..., 0] != extended[..., 0]))):
                kwargs[name] = extended[..., -out_of_sample:]

        return kwargs


class SARIMAXResults(MLEResults):
    """
    Class to hold results from fitting an SARIMAX model.

    Parameters
    ----------
    model : SARIMAX instance
        The fitted model instance

    Attributes
    ----------
    specification : dictionary
        Dictionary including all attributes from the SARIMAX model instance.
    polynomial_ar : ndarray
        Array containing autoregressive lag polynomial coefficients,
        ordered from lowest degree to highest. Initialized with ones, unless
        a coefficient is constrained to be zero (in which case it is zero).
    polynomial_ma : ndarray
        Array containing moving average lag polynomial coefficients,
        ordered from lowest degree to highest. Initialized with ones, unless
        a coefficient is constrained to be zero (in which case it is zero).
    polynomial_seasonal_ar : ndarray
        Array containing seasonal autoregressive lag polynomial coefficients,
        ordered from lowest degree to highest. Initialized with ones, unless
        a coefficient is constrained to be zero (in which case it is zero).
    polynomial_seasonal_ma : ndarray
        Array containing seasonal moving average lag polynomial coefficients,
        ordered from lowest degree to highest. Initialized with ones, unless
        a coefficient is constrained to be zero (in which case it is zero).
    polynomial_trend : ndarray
        Array containing trend polynomial coefficients, ordered from lowest
        degree to highest. Initialized with ones, unless a coefficient is
        constrained to be zero (in which case it is zero).
    model_orders : list of int
        The orders of each of the polynomials in the model.
    param_terms : list of str
        List of parameters actually included in the model, in sorted order.

    See Also
    --------
    statsmodels.tsa.statespace.kalman_filter.FilterResults
    statsmodels.tsa.statespace.mlemodel.MLEResults
    """
    def __init__(self, model, params, filter_results, cov_type=None,
                 **kwargs):
        super(SARIMAXResults, self).__init__(model, params, filter_results,
                                             cov_type, **kwargs)

        self.df_resid = np.inf  # attribute required for wald tests

        # Save _init_kwds
        self._init_kwds = self.model._get_init_kwds()

        # Save model specification
        self.specification = Bunch(**{
            # Set additional model parameters
            'seasonal_periods': self.model.seasonal_periods,
            'measurement_error': self.model.measurement_error,
            'time_varying_regression': self.model.time_varying_regression,
            'simple_differencing': self.model.simple_differencing,
            'enforce_stationarity': self.model.enforce_stationarity,
            'enforce_invertibility': self.model.enforce_invertibility,
            'hamilton_representation': self.model.hamilton_representation,
            'concentrate_scale': self.model.concentrate_scale,
            'trend_offset': self.model.trend_offset,

            'order': self.model.order,
            'seasonal_order': self.model.seasonal_order,

            # Model order
            'k_diff': self.model.k_diff,
            'k_seasonal_diff': self.model.k_seasonal_diff,
            'k_ar': self.model.k_ar,
            'k_ma': self.model.k_ma,
            'k_seasonal_ar': self.model.k_seasonal_ar,
            'k_seasonal_ma': self.model.k_seasonal_ma,

            # Param Numbers
            'k_ar_params': self.model.k_ar_params,
            'k_ma_params': self.model.k_ma_params,

            # Trend / Regression
            'trend': self.model.trend,
            'k_trend': self.model.k_trend,
            'k_exog': self.model.k_exog,

            'mle_regression': self.model.mle_regression,
            'state_regression': self.model.state_regression,
        })

        # Polynomials
        self.polynomial_trend = self.model._polynomial_trend
        self.polynomial_ar = self.model._polynomial_ar
        self.polynomial_ma = self.model._polynomial_ma
        self.polynomial_seasonal_ar = self.model._polynomial_seasonal_ar
        self.polynomial_seasonal_ma = self.model._polynomial_seasonal_ma
        self.polynomial_reduced_ar = np.polymul(
            self.polynomial_ar, self.polynomial_seasonal_ar
        )
        self.polynomial_reduced_ma = np.polymul(
            self.polynomial_ma, self.polynomial_seasonal_ma
        )

        # Distinguish parameters
        self.model_orders = self.model.model_orders
        self.param_terms = self.model.param_terms
        start = end = 0
        for name in self.param_terms:
            if name == 'ar':
                k = self.model.k_ar_params
            elif name == 'ma':
                k = self.model.k_ma_params
            elif name == 'seasonal_ar':
                k = self.model.k_seasonal_ar_params
            elif name == 'seasonal_ma':
                k = self.model.k_seasonal_ma_params
            else:
                k = self.model_orders[name]
            end += k
            setattr(self, '_params_%s' % name, self.params[start:end])
            start += k
        # GH7527, all terms must be defined
        all_terms = ['ar', 'ma', 'seasonal_ar', 'seasonal_ma', 'variance']
        for name in set(all_terms).difference(self.param_terms):
            setattr(self, '_params_%s' % name, np.empty(0))

        # Handle removing data
        self._data_attr_model.extend(['orig_endog', 'orig_exog'])

    def extend(self, endog, exog=None, **kwargs):
        kwargs.setdefault('trend_offset', self.nobs + 1)
        return super(SARIMAXResults, self).extend(endog, exog=exog, **kwargs)

    @cache_readonly
    def arroots(self):
        """
        (array) Roots of the reduced form autoregressive lag polynomial
        """
        return np.roots(self.polynomial_reduced_ar)**-1

    @cache_readonly
    def maroots(self):
        """
        (array) Roots of the reduced form moving average lag polynomial
        """
        return np.roots(self.polynomial_reduced_ma)**-1

    @cache_readonly
    def arfreq(self):
        """
        (array) Frequency of the roots of the reduced form autoregressive
        lag polynomial
        """
        z = self.arroots
        if not z.size:
            return
        return np.arctan2(z.imag, z.real) / (2 * np.pi)

    @cache_readonly
    def mafreq(self):
        """
        (array) Frequency of the roots of the reduced form moving average
        lag polynomial
        """
        z = self.maroots
        if not z.size:
            return
        return np.arctan2(z.imag, z.real) / (2 * np.pi)

    @cache_readonly
    def arparams(self):
        """
        (array) Autoregressive parameters actually estimated in the model.
        Does not include seasonal autoregressive parameters (see
        `seasonalarparams`) or parameters whose values are constrained to be
        zero.
        """
        return self._params_ar

    @cache_readonly
    def seasonalarparams(self):
        """
        (array) Seasonal autoregressive parameters actually estimated in the
        model. Does not include nonseasonal autoregressive parameters (see
        `arparams`) or parameters whose values are constrained to be zero.
        """
        return self._params_seasonal_ar

    @cache_readonly
    def maparams(self):
        """
        (array) Moving average parameters actually estimated in the model.
        Does not include seasonal moving average parameters (see
        `seasonalmaparams`) or parameters whose values are constrained to be
        zero.
        """
        return self._params_ma

    @cache_readonly
    def seasonalmaparams(self):
        """
        (array) Seasonal moving average parameters actually estimated in the
        model. Does not include nonseasonal moving average parameters (see
        `maparams`) or parameters whose values are constrained to be zero.
        """
        return self._params_seasonal_ma

    @Appender(MLEResults.summary.__doc__)
    def summary(self, alpha=.05, start=None):
        # Create the model name

        # See if we have an ARIMA component
        order = ''
        if self.model.k_ar + self.model.k_diff + self.model.k_ma > 0:
            if self.model.k_ar == self.model.k_ar_params:
                order_ar = self.model.k_ar
            else:
                order_ar = list(self.model._spec.ar_lags)
            if self.model.k_ma == self.model.k_ma_params:
                order_ma = self.model.k_ma
            else:
                order_ma = list(self.model._spec.ma_lags)
            # If there is simple differencing, then that is reflected in the
            # dependent variable name
            k_diff = 0 if self.model.simple_differencing else self.model.k_diff
            order = '(%s, %d, %s)' % (order_ar, k_diff, order_ma)
        # See if we have an SARIMA component
        seasonal_order = ''
        has_seasonal = (
            self.model.k_seasonal_ar +
            self.model.k_seasonal_diff +
            self.model.k_seasonal_ma
        ) > 0
        if has_seasonal:
            tmp = int(self.model.k_seasonal_ar / self.model.seasonal_periods)
            if tmp == self.model.k_seasonal_ar_params:
                order_seasonal_ar = (
                    int(self.model.k_seasonal_ar / self.model.seasonal_periods)
                )
            else:
                order_seasonal_ar = list(self.model._spec.seasonal_ar_lags)
            tmp = int(self.model.k_seasonal_ma / self.model.seasonal_periods)
            if tmp == self.model.k_ma_params:
                order_seasonal_ma = tmp
            else:
                order_seasonal_ma = list(self.model._spec.seasonal_ma_lags)
            # If there is simple differencing, then that is reflected in the
            # dependent variable name
            k_seasonal_diff = self.model.k_seasonal_diff
            if self.model.simple_differencing:
                k_seasonal_diff = 0
            seasonal_order = ('(%s, %d, %s, %d)' %
                              (str(order_seasonal_ar), k_seasonal_diff,
                               str(order_seasonal_ma),
                               self.model.seasonal_periods))
            if not order == '':
                order += 'x'
        model_name = (
            '%s%s%s' % (self.model.__class__.__name__, order, seasonal_order)
            )
        return super(SARIMAXResults, self).summary(
            alpha=alpha, start=start, title='SARIMAX Results',
            model_name=model_name
        )


class SARIMAXResultsWrapper(MLEResultsWrapper):
    _attrs = {}
    _wrap_attrs = wrap.union_dicts(MLEResultsWrapper._wrap_attrs,
                                   _attrs)
    _methods = {}
    _wrap_methods = wrap.union_dicts(MLEResultsWrapper._wrap_methods,
                                     _methods)
wrap.populate_wrapper(SARIMAXResultsWrapper, SARIMAXResults)  # noqa:E305
