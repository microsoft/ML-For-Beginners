"""
Linear exponential smoothing models

Author: Chad Fulton
License: BSD-3
"""
import numpy as np
import pandas as pd
from statsmodels.base.data import PandasData

from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools.validation import (array_like, bool_like, float_like,
                                          string_like, int_like)

from statsmodels.tsa.exponential_smoothing import initialization as es_init
from statsmodels.tsa.statespace import initialization as ss_init
from statsmodels.tsa.statespace.kalman_filter import (
    MEMORY_CONSERVE, MEMORY_NO_FORECAST)

from statsmodels.compat.pandas import Appender
import statsmodels.base.wrapper as wrap

from statsmodels.iolib.summary import forg
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.tableformatting import fmt_params

from .mlemodel import MLEModel, MLEResults, MLEResultsWrapper


class ExponentialSmoothing(MLEModel):
    """
    Linear exponential smoothing models

    Parameters
    ----------
    endog : array_like
        The observed time-series process :math:`y`
    trend : bool, optional
        Whether or not to include a trend component. Default is False.
    damped_trend : bool, optional
        Whether or not an included trend component is damped. Default is False.
    seasonal : int, optional
        The number of periods in a complete seasonal cycle for seasonal
        (Holt-Winters) models. For example, 4 for quarterly data with an
        annual cycle or 7 for daily data with a weekly cycle. Default is
        no seasonal effects.
    initialization_method : str, optional
        Method for initialize the recursions. One of:

        * 'estimated'
        * 'concentrated'
        * 'heuristic'
        * 'known'

        If 'known' initialization is used, then `initial_level` must be
        passed, as well as `initial_slope` and `initial_seasonal` if
        applicable. Default is 'estimated'.
    initial_level : float, optional
        The initial level component. Only used if initialization is 'known'.
    initial_trend : float, optional
        The initial trend component. Only used if initialization is 'known'.
    initial_seasonal : array_like, optional
        The initial seasonal component. An array of length `seasonal`
        or length `seasonal - 1` (in which case the last initial value
        is computed to make the average effect zero). Only used if
        initialization is 'known'.
    bounds : iterable[tuple], optional
        An iterable containing bounds for the parameters. Must contain four
        elements, where each element is a tuple of the form (lower, upper).
        Default is (0.0001, 0.9999) for the level, trend, and seasonal
        smoothing parameters and (0.8, 0.98) for the trend damping parameter.
    concentrate_scale : bool, optional
        Whether or not to concentrate the scale (variance of the error term)
        out of the likelihood.

    Notes
    -----

    **Overview**

    The parameters and states of this model are estimated by setting up the
    exponential smoothing equations as a special case of a linear Gaussian
    state space model and applying the Kalman filter. As such, it has slightly
    worse performance than the dedicated exponential smoothing model,
    :class:`statsmodels.tsa.holtwinters.ExponentialSmoothing`, and it does not
    support multiplicative (nonlinear) exponential smoothing models.

    However, as a subclass of the state space models, this model class shares
    a consistent set of functionality with those models, which can make it
    easier to work with. In addition, it supports computing confidence
    intervals for forecasts and it supports concentrating the initial
    state out of the likelihood function.

    **Model timing**

    Typical exponential smoothing results correspond to the "filtered" output
    from state space models, because they incorporate both the transition to
    the new time point (adding the trend to the level and advancing the season)
    and updating to incorporate information from the observed datapoint. By
    contrast, the "predicted" output from state space models only incorporates
    the transition.

    One consequence is that the "initial state" corresponds to the "filtered"
    state at time t=0, but this is different from the usual state space
    initialization used in Statsmodels, which initializes the model with the
    "predicted" state at time t=1. This is important to keep in mind if
    setting the initial state directly (via `initialization_method='known'`).

    **Seasonality**

    In seasonal models, it is important to note that seasonals are included in
    the state vector of this model in the order:
    `[seasonal, seasonal.L1, seasonal.L2, seasonal.L3, ...]`. At time t, the
    `'seasonal'` state holds the seasonal factor operative at time t, while
    the `'seasonal.L'` state holds the seasonal factor that would have been
    operative at time t-1.

    Suppose that the seasonal order is `n_seasons = 4`. Then, because the
    initial state corresponds to time t=0 and the time t=1 is in the same
    season as time t=-3, the initial seasonal factor for time t=1 comes from
    the lag "L3" initial seasonal factor (i.e. at time t=1 this will be both
    the "L4" seasonal factor as well as the "L0", or current, seasonal factor).

    When the initial state is estimated (`initialization_method='estimated'`),
    there are only `n_seasons - 1` parameters, because the seasonal factors are
    normalized to sum to one. The three parameters that are estimated
    correspond to the lags "L0", "L1", and "L2" seasonal factors as of time
    t=0 (alternatively, the lags "L1", "L2", and "L3" as of time t=1).

    When the initial state is given (`initialization_method='known'`), the
    initial seasonal factors for time t=0 must be given by the argument
    `initial_seasonal`. This can either be a length `n_seasons - 1` array --
    in which case it should contain the lags "L0" - "L2" (in that order)
    seasonal factors as of time t=0 -- or a length `n_seasons` array, in which
    case it should contain the "L0" - "L3" (in that order) seasonal factors
    as of time t=0.

    Note that in the state vector and parameters, the "L0" seasonal is
    called "seasonal" or "initial_seasonal", while the i>0 lag is
    called "seasonal.L{i}".

    References
    ----------
    [1] Hyndman, Rob, Anne B. Koehler, J. Keith Ord, and Ralph D. Snyder.
        Forecasting with exponential smoothing: the state space approach.
        Springer Science & Business Media, 2008.
    """
    def __init__(self, endog, trend=False, damped_trend=False, seasonal=None,
                 initialization_method='estimated', initial_level=None,
                 initial_trend=None, initial_seasonal=None, bounds=None,
                 concentrate_scale=True, dates=None, freq=None):
        # Model definition
        self.trend = bool_like(trend, 'trend')
        self.damped_trend = bool_like(damped_trend, 'damped_trend')
        self.seasonal_periods = int_like(seasonal, 'seasonal', optional=True)
        self.seasonal = self.seasonal_periods is not None
        self.initialization_method = string_like(
            initialization_method, 'initialization_method').lower()
        self.concentrate_scale = bool_like(concentrate_scale,
                                           'concentrate_scale')

        # TODO: add validation for bounds (e.g. have all bounds, upper > lower)
        # TODO: add `bounds_method` argument to choose between "usual" and
        # "admissible" as in Hyndman et al. (2008)
        self.bounds = bounds
        if self.bounds is None:
            self.bounds = [(1e-4, 1-1e-4)] * 3 + [(0.8, 0.98)]

        # Validation
        if self.seasonal_periods == 1:
            raise ValueError('Cannot have a seasonal period of 1.')

        if self.seasonal and self.seasonal_periods is None:
            raise NotImplementedError('Unable to detect season automatically;'
                                      ' please specify `seasonal_periods`.')

        if self.initialization_method not in ['concentrated', 'estimated',
                                              'simple', 'heuristic', 'known']:
            raise ValueError('Invalid initialization method "%s".'
                             % initialization_method)

        if self.initialization_method == 'known':
            if initial_level is None:
                raise ValueError('`initial_level` argument must be provided'
                                 ' when initialization method is set to'
                                 ' "known".')
            if initial_trend is None and self.trend:
                raise ValueError('`initial_trend` argument must be provided'
                                 ' for models with a trend component when'
                                 ' initialization method is set to "known".')
            if initial_seasonal is None and self.seasonal:
                raise ValueError('`initial_seasonal` argument must be provided'
                                 ' for models with a seasonal component when'
                                 ' initialization method is set to "known".')

        # Initialize the state space model
        if not self.seasonal or self.seasonal_periods is None:
            self._seasonal_periods = 0
        else:
            self._seasonal_periods = self.seasonal_periods

        k_states = 2 + int(self.trend) + self._seasonal_periods
        k_posdef = 1

        init = ss_init.Initialization(k_states, 'known',
                                      constant=[0] * k_states)
        super(ExponentialSmoothing, self).__init__(
            endog, k_states=k_states, k_posdef=k_posdef,
            initialization=init, dates=dates, freq=freq)

        # Concentrate the scale out of the likelihood function
        if self.concentrate_scale:
            self.ssm.filter_concentrated = True

        # Setup fixed elements of the system matrices
        # Observation error
        self.ssm['design', 0, 0] = 1.
        self.ssm['selection', 0, 0] = 1.
        self.ssm['state_cov', 0, 0] = 1.

        # Level
        self.ssm['design', 0, 1] = 1.
        self.ssm['transition', 1, 1] = 1.

        # Trend
        if self.trend:
            self.ssm['transition', 1:3, 2] = 1.

        # Seasonal
        if self.seasonal:
            k = 2 + int(self.trend)
            self.ssm['design', 0, k] = 1.
            self.ssm['transition', k, -1] = 1.
            self.ssm['transition', k + 1:k_states, k:k_states - 1] = (
                np.eye(self.seasonal_periods - 1))

        # Initialization of the states
        if self.initialization_method != 'known':
            msg = ('Cannot give `%%s` argument when initialization is "%s"'
                   % initialization_method)
            if initial_level is not None:
                raise ValueError(msg % 'initial_level')
            if initial_trend is not None:
                raise ValueError(msg % 'initial_trend')
            if initial_seasonal is not None:
                raise ValueError(msg % 'initial_seasonal')

        if self.initialization_method == 'simple':
            initial_level, initial_trend, initial_seasonal = (
                es_init._initialization_simple(
                    self.endog[:, 0], trend='add' if self.trend else None,
                    seasonal='add' if self.seasonal else None,
                    seasonal_periods=self.seasonal_periods))
        elif self.initialization_method == 'heuristic':
            initial_level, initial_trend, initial_seasonal = (
                es_init._initialization_heuristic(
                    self.endog[:, 0], trend='add' if self.trend else None,
                    seasonal='add' if self.seasonal else None,
                    seasonal_periods=self.seasonal_periods))
        elif self.initialization_method == 'known':
            initial_level = float_like(initial_level, 'initial_level')
            if self.trend:
                initial_trend = float_like(initial_trend, 'initial_trend')
            if self.seasonal:
                initial_seasonal = array_like(initial_seasonal,
                                              'initial_seasonal')

                if len(initial_seasonal) == self.seasonal_periods - 1:
                    initial_seasonal = np.r_[initial_seasonal,
                                             0 - np.sum(initial_seasonal)]

                if len(initial_seasonal) != self.seasonal_periods:
                    raise ValueError(
                        'Invalid length of initial seasonal values. Must be'
                        ' one of s or s-1, where s is the number of seasonal'
                        ' periods.')

        # Note that the simple and heuristic methods of computing initial
        # seasonal factors return estimated seasonal factors associated with
        # the first t = 1, 2, ..., `n_seasons` observations. To use these as
        # the initial state, we lag them by `n_seasons`. This yields, for
        # example for `n_seasons = 4`, the seasons lagged L3, L2, L1, L0.
        # As described above, the state vector in this model should have
        # seasonal factors ordered L0, L1, L2, L3, and as a result we need to
        # reverse the order of the computed initial seasonal factors from
        # these methods.
        methods = ['simple', 'heuristic']
        if (self.initialization_method in methods
                and initial_seasonal is not None):
            initial_seasonal = initial_seasonal[::-1]

        self._initial_level = initial_level
        self._initial_trend = initial_trend
        self._initial_seasonal = initial_seasonal
        self._initial_state = None

        # Initialize now if possible (if we have a damped trend, then
        # initialization will depend on the phi parameter, and so has to be
        # done at each `update`)
        methods = ['simple', 'heuristic', 'known']
        if not self.damped_trend and self.initialization_method in methods:
            self._initialize_constant_statespace(initial_level, initial_trend,
                                                 initial_seasonal)

        # Save keys for kwarg initialization
        self._init_keys += ['trend', 'damped_trend', 'seasonal',
                            'initialization_method', 'initial_level',
                            'initial_trend', 'initial_seasonal', 'bounds',
                            'concentrate_scale', 'dates', 'freq']

    def _get_init_kwds(self):
        kwds = super()._get_init_kwds()
        kwds['seasonal'] = self.seasonal_periods
        return kwds

    @property
    def _res_classes(self):
        return {'fit': (ExponentialSmoothingResults,
                        ExponentialSmoothingResultsWrapper)}

    def clone(self, endog, exog=None, **kwargs):
        if exog is not None:
            raise NotImplementedError(
                'ExponentialSmoothing does not support `exog`.')
        return self._clone_from_init_kwds(endog, **kwargs)

    @property
    def state_names(self):
        state_names = ['error', 'level']
        if self.trend:
            state_names += ['trend']
        if self.seasonal:
            state_names += (
                ['seasonal'] + ['seasonal.L%d' % i
                                for i in range(1, self.seasonal_periods)])

        return state_names

    @property
    def param_names(self):
        param_names = ['smoothing_level']
        if self.trend:
            param_names += ['smoothing_trend']
        if self.seasonal:
            param_names += ['smoothing_seasonal']
        if self.damped_trend:
            param_names += ['damping_trend']
        if not self.concentrate_scale:
            param_names += ['sigma2']

        # Initialization
        if self.initialization_method == 'estimated':
            param_names += ['initial_level']
            if self.trend:
                param_names += ['initial_trend']
            if self.seasonal:
                param_names += (
                    ['initial_seasonal']
                    + ['initial_seasonal.L%d' % i
                       for i in range(1, self.seasonal_periods - 1)])

        return param_names

    @property
    def start_params(self):
        # Make sure starting parameters aren't beyond or right on the bounds
        bounds = [(x[0] + 1e-3, x[1] - 1e-3) for x in self.bounds]

        # See Hyndman p.24
        start_params = [np.clip(0.1, *bounds[0])]
        if self.trend:
            start_params += [np.clip(0.01, *bounds[1])]
        if self.seasonal:
            start_params += [np.clip(0.01, *bounds[2])]
        if self.damped_trend:
            start_params += [np.clip(0.98, *bounds[3])]
        if not self.concentrate_scale:
            start_params += [np.var(self.endog)]

        # Initialization
        if self.initialization_method == 'estimated':
            initial_level, initial_trend, initial_seasonal = (
                es_init._initialization_simple(
                    self.endog[:, 0],
                    trend='add' if self.trend else None,
                    seasonal='add' if self.seasonal else None,
                    seasonal_periods=self.seasonal_periods))
            start_params += [initial_level]
            if self.trend:
                start_params += [initial_trend]
            if self.seasonal:
                start_params += initial_seasonal.tolist()[::-1][:-1]

        return np.array(start_params)

    @property
    def k_params(self):
        k_params = (
            1 + int(self.trend) + int(self.seasonal) +
            int(not self.concentrate_scale) + int(self.damped_trend))
        if self.initialization_method == 'estimated':
            k_params += (
                1 + int(self.trend) +
                int(self.seasonal) * (self._seasonal_periods - 1))
        return k_params

    def transform_params(self, unconstrained):
        unconstrained = np.array(unconstrained, ndmin=1)
        constrained = np.zeros_like(unconstrained)

        # Alpha in (0, 1)
        low, high = self.bounds[0]
        constrained[0] = (
            1 / (1 + np.exp(-unconstrained[0])) * (high - low) + low)
        i = 1

        # Beta in (0, alpha)
        if self.trend:
            low, high = self.bounds[1]
            high = min(high, constrained[0])
            constrained[i] = (
                1 / (1 + np.exp(-unconstrained[i])) * (high - low) + low)
            i += 1

        # Gamma in (0, 1 - alpha)
        if self.seasonal:
            low, high = self.bounds[2]
            high = min(high, 1 - constrained[0])
            constrained[i] = (
                1 / (1 + np.exp(-unconstrained[i])) * (high - low) + low)
            i += 1

        # Phi in bounds (e.g. default is [0.8, 0.98])
        if self.damped_trend:
            low, high = self.bounds[3]
            constrained[i] = (
                1 / (1 + np.exp(-unconstrained[i])) * (high - low) + low)
            i += 1

        # sigma^2 positive
        if not self.concentrate_scale:
            constrained[i] = unconstrained[i]**2
            i += 1

        # Initial parameters are as-is
        if self.initialization_method == 'estimated':
            constrained[i:] = unconstrained[i:]

        return constrained

    def untransform_params(self, constrained):
        constrained = np.array(constrained, ndmin=1)
        unconstrained = np.zeros_like(constrained)

        # Alpha in (0, 1)
        low, high = self.bounds[0]
        tmp = (constrained[0] - low) / (high - low)
        unconstrained[0] = np.log(tmp / (1 - tmp))
        i = 1

        # Beta in (0, alpha)
        if self.trend:
            low, high = self.bounds[1]
            high = min(high, constrained[0])
            tmp = (constrained[i] - low) / (high - low)
            unconstrained[i] = np.log(tmp / (1 - tmp))
            i += 1

        # Gamma in (0, 1 - alpha)
        if self.seasonal:
            low, high = self.bounds[2]
            high = min(high, 1 - constrained[0])
            tmp = (constrained[i] - low) / (high - low)
            unconstrained[i] = np.log(tmp / (1 - tmp))
            i += 1

        # Phi in bounds (e.g. default is [0.8, 0.98])
        if self.damped_trend:
            low, high = self.bounds[3]
            tmp = (constrained[i] - low) / (high - low)
            unconstrained[i] = np.log(tmp / (1 - tmp))
            i += 1

        # sigma^2 positive
        if not self.concentrate_scale:
            unconstrained[i] = constrained[i]**0.5
            i += 1

        # Initial parameters are as-is
        if self.initialization_method == 'estimated':
            unconstrained[i:] = constrained[i:]

        return unconstrained

    def _initialize_constant_statespace(self, initial_level,
                                        initial_trend=None,
                                        initial_seasonal=None):
        # Note: this should be run after `update` has already put any new
        # parameters into the transition matrix, since it uses the transition
        # matrix explicitly.

        # Due to timing differences, the state space representation integrates
        # the trend into the level in the "predicted_state" (only the
        # "filtered_state" corresponds to the timing of the exponential
        # smoothing models)

        # Initial values are interpreted as "filtered" values
        constant = np.array([0., initial_level])
        if self.trend and initial_trend is not None:
            constant = np.r_[constant, initial_trend]
        if self.seasonal and initial_seasonal is not None:
            constant = np.r_[constant, initial_seasonal]
        self._initial_state = constant[1:]

        # Apply the prediction step to get to what we need for our Kalman
        # filter implementation
        constant = np.dot(self.ssm['transition'], constant)

        self.initialization.constant = constant

    def _initialize_stationary_cov_statespace(self):
        R = self.ssm['selection']
        Q = self.ssm['state_cov']
        self.initialization.stationary_cov = R.dot(Q).dot(R.T)

    def update(self, params, transformed=True, includes_fixed=False,
               complex_step=False):
        params = self.handle_params(params, transformed=transformed,
                                    includes_fixed=includes_fixed)

        # State space system matrices
        self.ssm['selection', 0, 0] = 1 - params[0]
        self.ssm['selection', 1, 0] = params[0]
        i = 1
        if self.trend:
            self.ssm['selection', 2, 0] = params[i]
            i += 1
        if self.seasonal:
            self.ssm['selection', 0, 0] -= params[i]
            self.ssm['selection', i + 1, 0] = params[i]
            i += 1
        if self.damped_trend:
            self.ssm['transition', 1:3, 2] = params[i]
            i += 1
        if not self.concentrate_scale:
            self.ssm['state_cov', 0, 0] = params[i]
            i += 1

        # State initialization
        if self.initialization_method == 'estimated':
            initial_level = params[i]
            i += 1
            initial_trend = None
            initial_seasonal = None

            if self.trend:
                initial_trend = params[i]
                i += 1
            if self.seasonal:
                initial_seasonal = params[i: i + self.seasonal_periods - 1]
                initial_seasonal = np.r_[initial_seasonal,
                                         0 - np.sum(initial_seasonal)]
            self._initialize_constant_statespace(initial_level, initial_trend,
                                                 initial_seasonal)

        methods = ['simple', 'heuristic', 'known']
        if self.damped_trend and self.initialization_method in methods:
            self._initialize_constant_statespace(
                self._initial_level, self._initial_trend,
                self._initial_seasonal)

        self._initialize_stationary_cov_statespace()

    def _compute_concentrated_states(self, params, *args, **kwargs):
        # Apply the usual filter, but keep forecasts
        kwargs['conserve_memory'] = MEMORY_CONSERVE & ~MEMORY_NO_FORECAST
        super().loglike(params, *args, **kwargs)

        # Compute the initial state vector
        y_tilde = np.array(self.ssm._kalman_filter.forecast_error[0],
                           copy=True)

        # Need to modify our state space system matrices slightly to get them
        # back into the form of the innovations framework of
        # De Livera et al. (2011)
        T = self['transition', 1:, 1:]
        R = self['selection', 1:]
        Z = self['design', :, 1:].copy()
        i = 1
        if self.trend:
            Z[0, i] = 1.
            i += 1
        if self.seasonal:
            Z[0, i] = 0.
            Z[0, -1] = 1.

        # Now compute the regression components as described in
        # De Livera et al. (2011), equation (10).
        D = T - R.dot(Z)
        w = np.zeros((self.nobs, self.k_states - 1), dtype=D.dtype)
        w[0] = Z
        for i in range(self.nobs - 1):
            w[i + 1] = w[i].dot(D)
        mod_ols = GLM(y_tilde, w)

        # If we have seasonal parameters, constrain them to sum to zero
        # (otherwise the initial level gets confounded with the sum of the
        # seasonals).
        if self.seasonal:
            R = np.zeros_like(Z)
            R[0, -self.seasonal_periods:] = 1.
            q = np.zeros((1, 1))
            res_ols = mod_ols.fit_constrained((R, q))
        else:
            res_ols = mod_ols.fit()

        # Separate into individual components
        initial_level = res_ols.params[0]
        initial_trend = res_ols.params[1] if self.trend else None
        initial_seasonal = (
            res_ols.params[-self.seasonal_periods:] if self.seasonal else None)

        return initial_level, initial_trend, initial_seasonal

    @Appender(MLEModel.loglike.__doc__)
    def loglike(self, params, *args, **kwargs):
        if self.initialization_method == 'concentrated':
            self._initialize_constant_statespace(
                *self._compute_concentrated_states(params, *args, **kwargs))
            llf = self.ssm.loglike()
            self.ssm.initialization.constant = np.zeros(self.k_states)
        else:
            llf = super().loglike(params, *args, **kwargs)
        return llf

    @Appender(MLEModel.filter.__doc__)
    def filter(self, params, cov_type=None, cov_kwds=None,
               return_ssm=False, results_class=None,
               results_wrapper_class=None, *args, **kwargs):
        if self.initialization_method == 'concentrated':
            self._initialize_constant_statespace(
                *self._compute_concentrated_states(params, *args, **kwargs))

        results = super().filter(
            params, cov_type=cov_type, cov_kwds=cov_kwds,
            return_ssm=return_ssm, results_class=results_class,
            results_wrapper_class=results_wrapper_class, *args, **kwargs)

        if self.initialization_method == 'concentrated':
            self.ssm.initialization.constant = np.zeros(self.k_states)
        return results

    @Appender(MLEModel.smooth.__doc__)
    def smooth(self, params, cov_type=None, cov_kwds=None,
               return_ssm=False, results_class=None,
               results_wrapper_class=None, *args, **kwargs):
        if self.initialization_method == 'concentrated':
            self._initialize_constant_statespace(
                *self._compute_concentrated_states(params, *args, **kwargs))

        results = super().smooth(
            params, cov_type=cov_type, cov_kwds=cov_kwds,
            return_ssm=return_ssm, results_class=results_class,
            results_wrapper_class=results_wrapper_class, *args, **kwargs)

        if self.initialization_method == 'concentrated':
            self.ssm.initialization.constant = np.zeros(self.k_states)
        return results


class ExponentialSmoothingResults(MLEResults):
    """
    Results from fitting a linear exponential smoothing model
    """
    def __init__(self, model, params, filter_results, cov_type=None,
                 **kwargs):
        super().__init__(model, params, filter_results, cov_type, **kwargs)

        # Save the states
        self.initial_state = model._initial_state
        if isinstance(self.data, PandasData):
            index = self.data.row_labels
            self.initial_state = pd.DataFrame(
                [model._initial_state], columns=model.state_names[1:])
            if model._index_dates and model._index_freq is not None:
                self.initial_state.index = index.shift(-1)[:1]

    @Appender(MLEResults.summary.__doc__)
    def summary(self, alpha=.05, start=None):
        specification = ['A']
        if self.model.trend and self.model.damped_trend:
            specification.append('Ad')
        elif self.model.trend:
            specification.append('A')
        else:
            specification.append('N')
        if self.model.seasonal:
            specification.append('A')
        else:
            specification.append('N')

        model_name = 'ETS(' + ', '.join(specification) + ')'

        summary = super(ExponentialSmoothingResults, self).summary(
            alpha=alpha, start=start, title='Exponential Smoothing Results',
            model_name=model_name)

        if self.model.initialization_method != 'estimated':
            params = np.array(self.initial_state)
            if params.ndim > 1:
                params = params[0]
            names = self.model.state_names[1:]
            param_header = ['initialization method: %s'
                            % self.model.initialization_method]
            params_stubs = names
            params_data = [[forg(params[i], prec=4)]
                           for i in range(len(params))]

            initial_state_table = SimpleTable(params_data,
                                              param_header,
                                              params_stubs,
                                              txt_fmt=fmt_params)
            summary.tables.insert(-1, initial_state_table)

        return summary


class ExponentialSmoothingResultsWrapper(MLEResultsWrapper):
    _attrs = {}
    _wrap_attrs = wrap.union_dicts(MLEResultsWrapper._wrap_attrs,
                                   _attrs)
    _methods = {}
    _wrap_methods = wrap.union_dicts(MLEResultsWrapper._wrap_methods,
                                     _methods)
wrap.populate_wrapper(ExponentialSmoothingResultsWrapper,  # noqa:E305
                      ExponentialSmoothingResults)
