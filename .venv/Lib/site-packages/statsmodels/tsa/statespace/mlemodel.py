# -*- coding: utf-8 -*-
"""
State Space Model

Author: Chad Fulton
License: Simplified-BSD
"""
from statsmodels.compat.pandas import is_int_index

import contextlib
import warnings

import datetime as dt
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy.stats import norm

from statsmodels.tools.tools import pinv_extended, Bunch
from statsmodels.tools.sm_exceptions import PrecisionWarning, ValueWarning
from statsmodels.tools.numdiff import (_get_epsilon, approx_hess_cs,
                                       approx_fprime_cs, approx_fprime)
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.eval_measures import aic, aicc, bic, hqic

import statsmodels.base.wrapper as wrap

import statsmodels.tsa.base.prediction as pred

from statsmodels.base.data import PandasData
import statsmodels.tsa.base.tsa_model as tsbase

from .news import NewsResults
from .simulation_smoother import SimulationSmoother
from .kalman_smoother import SmootherResults
from .kalman_filter import INVERT_UNIVARIATE, SOLVE_LU, MEMORY_CONSERVE
from .initialization import Initialization
from .tools import prepare_exog, concat, _safe_cond, get_impact_dates


def _handle_args(names, defaults, *args, **kwargs):
    output_args = []
    # We need to handle positional arguments in two ways, in case this was
    # called by a Scipy optimization routine
    if len(args) > 0:
        # the fit() method will pass a dictionary
        if isinstance(args[0], dict):
            flags = args[0]
        # otherwise, a user may have just used positional arguments...
        else:
            flags = dict(zip(names, args))
        for i in range(len(names)):
            output_args.append(flags.get(names[i], defaults[i]))

        for name, value in flags.items():
            if name in kwargs:
                raise TypeError("loglike() got multiple values for keyword"
                                " argument '%s'" % name)
    else:
        for i in range(len(names)):
            output_args.append(kwargs.pop(names[i], defaults[i]))

    return tuple(output_args) + (kwargs,)


def _check_index(desired_index, dta, title='data'):
    given_index = None
    if isinstance(dta, (pd.Series, pd.DataFrame)):
        given_index = dta.index
    if given_index is not None and not desired_index.equals(given_index):
        desired_freq = getattr(desired_index, 'freq', None)
        given_freq = getattr(given_index, 'freq', None)
        if ((desired_freq is not None or given_freq is not None) and
                desired_freq != given_freq):
            raise ValueError('Given %s does not have an index'
                             ' that extends the index of the'
                             ' model. Expected index frequency is'
                             ' "%s", but got "%s".'
                             % (title, desired_freq, given_freq))
        else:
            raise ValueError('Given %s does not have an index'
                             ' that extends the index of the'
                             ' model.' % title)


class MLEModel(tsbase.TimeSeriesModel):
    r"""
    State space model for maximum likelihood estimation

    Parameters
    ----------
    endog : array_like
        The observed time-series process :math:`y`
    k_states : int
        The dimension of the unobserved state process.
    exog : array_like, optional
        Array of exogenous regressors, shaped nobs x k. Default is no
        exogenous regressors.
    dates : array_like of datetime, optional
        An array-like object of datetime objects. If a Pandas object is given
        for endog, it is assumed to have a DateIndex.
    freq : str, optional
        The frequency of the time-series. A Pandas offset or 'B', 'D', 'W',
        'M', 'A', or 'Q'. This is optional if dates are given.
    **kwargs
        Keyword arguments may be used to provide default values for state space
        matrices or for Kalman filtering options. See `Representation`, and
        `KalmanFilter` for more details.

    Attributes
    ----------
    ssm : statsmodels.tsa.statespace.kalman_filter.KalmanFilter
        Underlying state space representation.

    See Also
    --------
    statsmodels.tsa.statespace.mlemodel.MLEResults
    statsmodels.tsa.statespace.kalman_filter.KalmanFilter
    statsmodels.tsa.statespace.representation.Representation

    Notes
    -----
    This class wraps the state space model with Kalman filtering to add in
    functionality for maximum likelihood estimation. In particular, it adds
    the concept of updating the state space representation based on a defined
    set of parameters, through the `update` method or `updater` attribute (see
    below for more details on which to use when), and it adds a `fit` method
    which uses a numerical optimizer to select the parameters that maximize
    the likelihood of the model.

    The `start_params` `update` method must be overridden in the
    child class (and the `transform` and `untransform` methods, if needed).
    """

    def __init__(self, endog, k_states, exog=None, dates=None, freq=None,
                 **kwargs):
        # Initialize the model base
        super(MLEModel, self).__init__(endog=endog, exog=exog,
                                       dates=dates, freq=freq,
                                       missing='none')

        # Store kwargs to recreate model
        self._init_kwargs = kwargs

        # Prepared the endog array: C-ordered, shape=(nobs x k_endog)
        self.endog, self.exog = self.prepare_data()

        # Dimensions
        self.nobs = self.endog.shape[0]
        self.k_states = k_states

        # Initialize the state-space representation
        self.initialize_statespace(**kwargs)

        # Setup holder for fixed parameters
        self._has_fixed_params = False
        self._fixed_params = None
        self._params_index = None
        self._fixed_params_index = None
        self._free_params_index = None

    def prepare_data(self):
        """
        Prepare data for use in the state space representation
        """
        endog = np.array(self.data.orig_endog, order='C')
        exog = self.data.orig_exog
        if exog is not None:
            exog = np.array(exog)

        # Base class may allow 1-dim data, whereas we need 2-dim
        if endog.ndim == 1:
            endog.shape = (endog.shape[0], 1)  # this will be C-contiguous

        return endog, exog

    def initialize_statespace(self, **kwargs):
        """
        Initialize the state space representation

        Parameters
        ----------
        **kwargs
            Additional keyword arguments to pass to the state space class
            constructor.
        """
        # (Now self.endog is C-ordered and in long format (nobs x k_endog). To
        # get F-ordered and in wide format just need to transpose)
        endog = self.endog.T

        # Instantiate the state space object
        self.ssm = SimulationSmoother(endog.shape[0], self.k_states,
                                      nobs=endog.shape[1], **kwargs)
        # Bind the data to the model
        self.ssm.bind(endog)

        # Other dimensions, now that `ssm` is available
        self.k_endog = self.ssm.k_endog

    def _get_index_with_final_state(self):
        # The index we inherit from `TimeSeriesModel` will only cover the
        # data sample itself, but we will also need an index value for the
        # final state which is the next time step to the last datapoint.
        # This method figures out an appropriate value for the three types of
        # supported indexes: date-based, Int64Index, or RangeIndex
        if self._index_dates:
            if isinstance(self._index, pd.DatetimeIndex):
                index = pd.date_range(
                    start=self._index[0], periods=len(self._index) + 1,
                    freq=self._index.freq)
            elif isinstance(self._index, pd.PeriodIndex):
                index = pd.period_range(
                    start=self._index[0], periods=len(self._index) + 1,
                    freq=self._index.freq)
            else:
                raise NotImplementedError
        elif isinstance(self._index, pd.RangeIndex):
            # COMPAT: pd.RangeIndex does not have start, stop, step prior to
            #         pandas 0.25
            try:
                start = self._index.start
                stop = self._index.stop
                step = self._index.step
            except AttributeError:
                start = self._index._start
                stop = self._index._stop
                step = self._index._step
            index = pd.RangeIndex(start, stop + step, step)
        elif is_int_index(self._index):
            # The only valid Int64Index is a full, incrementing index, so this
            # is general
            value = self._index[-1] + 1
            index = pd.Index(self._index.tolist() + [value])
        else:
            raise NotImplementedError
        return index

    def __setitem__(self, key, value):
        return self.ssm.__setitem__(key, value)

    def __getitem__(self, key):
        return self.ssm.__getitem__(key)

    def _get_init_kwds(self):
        # Get keywords based on model attributes
        kwds = super(MLEModel, self)._get_init_kwds()

        for key, value in kwds.items():
            if value is None and hasattr(self.ssm, key):
                kwds[key] = getattr(self.ssm, key)

        return kwds

    def clone(self, endog, exog=None, **kwargs):
        """
        Clone state space model with new data and optionally new specification

        Parameters
        ----------
        endog : array_like
            The observed time-series process :math:`y`
        k_states : int
            The dimension of the unobserved state process.
        exog : array_like, optional
            Array of exogenous regressors, shaped nobs x k. Default is no
            exogenous regressors.
        kwargs
            Keyword arguments to pass to the new model class to change the
            model specification.

        Returns
        -------
        model : MLEModel subclass

        Notes
        -----
        This method must be implemented
        """
        raise NotImplementedError('This method is not implemented in the base'
                                  ' class and must be set up by each specific'
                                  ' model.')

    def _clone_from_init_kwds(self, endog, **kwargs):
        # Cannot make this the default, because there is extra work required
        # for subclasses to make _get_init_kwds useful.
        use_kwargs = self._get_init_kwds()
        use_kwargs.update(kwargs)

        # Check for `exog`
        if getattr(self, 'k_exog', 0) > 0 and kwargs.get('exog', None) is None:
            raise ValueError('Cloning a model with an exogenous component'
                             ' requires specifying a new exogenous array using'
                             ' the `exog` argument.')

        mod = self.__class__(endog, **use_kwargs)
        return mod

    def set_filter_method(self, filter_method=None, **kwargs):
        """
        Set the filtering method

        The filtering method controls aspects of which Kalman filtering
        approach will be used.

        Parameters
        ----------
        filter_method : int, optional
            Bitmask value to set the filter method to. See notes for details.
        **kwargs
            Keyword arguments may be used to influence the filter method by
            setting individual boolean flags. See notes for details.

        Notes
        -----
        This method is rarely used. See the corresponding function in the
        `KalmanFilter` class for details.
        """
        self.ssm.set_filter_method(filter_method, **kwargs)

    def set_inversion_method(self, inversion_method=None, **kwargs):
        """
        Set the inversion method

        The Kalman filter may contain one matrix inversion: that of the
        forecast error covariance matrix. The inversion method controls how and
        if that inverse is performed.

        Parameters
        ----------
        inversion_method : int, optional
            Bitmask value to set the inversion method to. See notes for
            details.
        **kwargs
            Keyword arguments may be used to influence the inversion method by
            setting individual boolean flags. See notes for details.

        Notes
        -----
        This method is rarely used. See the corresponding function in the
        `KalmanFilter` class for details.
        """
        self.ssm.set_inversion_method(inversion_method, **kwargs)

    def set_stability_method(self, stability_method=None, **kwargs):
        """
        Set the numerical stability method

        The Kalman filter is a recursive algorithm that may in some cases
        suffer issues with numerical stability. The stability method controls
        what, if any, measures are taken to promote stability.

        Parameters
        ----------
        stability_method : int, optional
            Bitmask value to set the stability method to. See notes for
            details.
        **kwargs
            Keyword arguments may be used to influence the stability method by
            setting individual boolean flags. See notes for details.

        Notes
        -----
        This method is rarely used. See the corresponding function in the
        `KalmanFilter` class for details.
        """
        self.ssm.set_stability_method(stability_method, **kwargs)

    def set_conserve_memory(self, conserve_memory=None, **kwargs):
        """
        Set the memory conservation method

        By default, the Kalman filter computes a number of intermediate
        matrices at each iteration. The memory conservation options control
        which of those matrices are stored.

        Parameters
        ----------
        conserve_memory : int, optional
            Bitmask value to set the memory conservation method to. See notes
            for details.
        **kwargs
            Keyword arguments may be used to influence the memory conservation
            method by setting individual boolean flags.

        Notes
        -----
        This method is rarely used. See the corresponding function in the
        `KalmanFilter` class for details.
        """
        self.ssm.set_conserve_memory(conserve_memory, **kwargs)

    def set_smoother_output(self, smoother_output=None, **kwargs):
        """
        Set the smoother output

        The smoother can produce several types of results. The smoother output
        variable controls which are calculated and returned.

        Parameters
        ----------
        smoother_output : int, optional
            Bitmask value to set the smoother output to. See notes for details.
        **kwargs
            Keyword arguments may be used to influence the smoother output by
            setting individual boolean flags.

        Notes
        -----
        This method is rarely used. See the corresponding function in the
        `KalmanSmoother` class for details.
        """
        self.ssm.set_smoother_output(smoother_output, **kwargs)

    def initialize_known(self, initial_state, initial_state_cov):
        """Initialize known"""
        self.ssm.initialize_known(initial_state, initial_state_cov)

    def initialize_approximate_diffuse(self, variance=None):
        """Initialize approximate diffuse"""
        self.ssm.initialize_approximate_diffuse(variance)

    def initialize_stationary(self):
        """Initialize stationary"""
        self.ssm.initialize_stationary()

    @property
    def initialization(self):
        return self.ssm.initialization

    @initialization.setter
    def initialization(self, value):
        self.ssm.initialization = value

    @property
    def initial_variance(self):
        return self.ssm.initial_variance

    @initial_variance.setter
    def initial_variance(self, value):
        self.ssm.initial_variance = value

    @property
    def loglikelihood_burn(self):
        return self.ssm.loglikelihood_burn

    @loglikelihood_burn.setter
    def loglikelihood_burn(self, value):
        self.ssm.loglikelihood_burn = value

    @property
    def tolerance(self):
        return self.ssm.tolerance

    @tolerance.setter
    def tolerance(self, value):
        self.ssm.tolerance = value

    def _validate_can_fix_params(self, param_names):
        for param_name in param_names:
            if param_name not in self.param_names:
                raise ValueError('Invalid parameter name passed: "%s".'
                                 % param_name)

    @contextlib.contextmanager
    def fix_params(self, params):
        """
        Fix parameters to specific values (context manager)

        Parameters
        ----------
        params : dict
            Dictionary describing the fixed parameter values, of the form
            `param_name: fixed_value`. See the `param_names` property for valid
            parameter names.

        Examples
        --------
        >>> mod = sm.tsa.SARIMAX(endog, order=(1, 0, 1))
        >>> with mod.fix_params({'ar.L1': 0.5}):
                res = mod.fit()
        """
        k_params = len(self.param_names)
        # Initialization (this is done here rather than in the constructor
        # because param_names may not be available at that point)
        if self._fixed_params is None:
            self._fixed_params = {}
            self._params_index = dict(
                zip(self.param_names, np.arange(k_params)))

        # Cache the current fixed parameters
        cache_fixed_params = self._fixed_params.copy()
        cache_has_fixed_params = self._has_fixed_params
        cache_fixed_params_index = self._fixed_params_index
        cache_free_params_index = self._free_params_index

        # Validate parameter names and values
        all_fixed_param_names = (
            set(params.keys()) | set(self._fixed_params.keys())
        )
        self._validate_can_fix_params(all_fixed_param_names)

        # Set the new fixed parameters, keeping the order as given by
        # param_names
        self._fixed_params.update(params)
        self._fixed_params = dict([
            (name, self._fixed_params[name]) for name in self.param_names
            if name in self._fixed_params])

        # Update associated values
        self._has_fixed_params = True
        self._fixed_params_index = [self._params_index[key]
                                    for key in self._fixed_params.keys()]
        self._free_params_index = list(
            set(np.arange(k_params)).difference(self._fixed_params_index))

        try:
            yield
        finally:
            # Reset the fixed parameters
            self._has_fixed_params = cache_has_fixed_params
            self._fixed_params = cache_fixed_params
            self._fixed_params_index = cache_fixed_params_index
            self._free_params_index = cache_free_params_index

    def fit(self, start_params=None, transformed=True, includes_fixed=False,
            cov_type=None, cov_kwds=None, method='lbfgs', maxiter=50,
            full_output=1, disp=5, callback=None, return_params=False,
            optim_score=None, optim_complex_step=None, optim_hessian=None,
            flags=None, low_memory=False, **kwargs):
        """
        Fits the model by maximum likelihood via Kalman filter.

        Parameters
        ----------
        start_params : array_like, optional
            Initial guess of the solution for the loglikelihood maximization.
            If None, the default is given by Model.start_params.
        transformed : bool, optional
            Whether or not `start_params` is already transformed. Default is
            True.
        includes_fixed : bool, optional
            If parameters were previously fixed with the `fix_params` method,
            this argument describes whether or not `start_params` also includes
            the fixed parameters, in addition to the free parameters. Default
            is False.
        cov_type : str, optional
            The `cov_type` keyword governs the method for calculating the
            covariance matrix of parameter estimates. Can be one of:

            - 'opg' for the outer product of gradient estimator
            - 'oim' for the observed information matrix estimator, calculated
              using the method of Harvey (1989)
            - 'approx' for the observed information matrix estimator,
              calculated using a numerical approximation of the Hessian matrix.
            - 'robust' for an approximate (quasi-maximum likelihood) covariance
              matrix that may be valid even in the presence of some
              misspecifications. Intermediate calculations use the 'oim'
              method.
            - 'robust_approx' is the same as 'robust' except that the
              intermediate calculations use the 'approx' method.
            - 'none' for no covariance matrix calculation.

            Default is 'opg' unless memory conservation is used to avoid
            computing the loglikelihood values for each observation, in which
            case the default is 'approx'.
        cov_kwds : dict or None, optional
            A dictionary of arguments affecting covariance matrix computation.

            **opg, oim, approx, robust, robust_approx**

            - 'approx_complex_step' : bool, optional - If True, numerical
              approximations are computed using complex-step methods. If False,
              numerical approximations are computed using finite difference
              methods. Default is True.
            - 'approx_centered' : bool, optional - If True, numerical
              approximations computed using finite difference methods use a
              centered approximation. Default is False.
        method : str, optional
            The `method` determines which solver from `scipy.optimize`
            is used, and it can be chosen from among the following strings:

            - 'newton' for Newton-Raphson
            - 'nm' for Nelder-Mead
            - 'bfgs' for Broyden-Fletcher-Goldfarb-Shanno (BFGS)
            - 'lbfgs' for limited-memory BFGS with optional box constraints
            - 'powell' for modified Powell's method
            - 'cg' for conjugate gradient
            - 'ncg' for Newton-conjugate gradient
            - 'basinhopping' for global basin-hopping solver

            The explicit arguments in `fit` are passed to the solver,
            with the exception of the basin-hopping solver. Each
            solver has several optional arguments that are not the same across
            solvers. See the notes section below (or scipy.optimize) for the
            available arguments and for the list of explicit arguments that the
            basin-hopping solver supports.
        maxiter : int, optional
            The maximum number of iterations to perform.
        full_output : bool, optional
            Set to True to have all available output in the Results object's
            mle_retvals attribute. The output is dependent on the solver.
            See LikelihoodModelResults notes section for more information.
        disp : bool, optional
            Set to True to print convergence messages.
        callback : callable callback(xk), optional
            Called after each iteration, as callback(xk), where xk is the
            current parameter vector.
        return_params : bool, optional
            Whether or not to return only the array of maximizing parameters.
            Default is False.
        optim_score : {'harvey', 'approx'} or None, optional
            The method by which the score vector is calculated. 'harvey' uses
            the method from Harvey (1989), 'approx' uses either finite
            difference or complex step differentiation depending upon the
            value of `optim_complex_step`, and None uses the built-in gradient
            approximation of the optimizer. Default is None. This keyword is
            only relevant if the optimization method uses the score.
        optim_complex_step : bool, optional
            Whether or not to use complex step differentiation when
            approximating the score; if False, finite difference approximation
            is used. Default is True. This keyword is only relevant if
            `optim_score` is set to 'harvey' or 'approx'.
        optim_hessian : {'opg','oim','approx'}, optional
            The method by which the Hessian is numerically approximated. 'opg'
            uses outer product of gradients, 'oim' uses the information
            matrix formula from Harvey (1989), and 'approx' uses numerical
            approximation. This keyword is only relevant if the
            optimization method uses the Hessian matrix.
        low_memory : bool, optional
            If set to True, techniques are applied to substantially reduce
            memory usage. If used, some features of the results object will
            not be available (including smoothed results and in-sample
            prediction), although out-of-sample forecasting is possible.
            Default is False.
        **kwargs
            Additional keyword arguments to pass to the optimizer.

        Returns
        -------
        results
            Results object holding results from fitting a state space model.

        See Also
        --------
        statsmodels.base.model.LikelihoodModel.fit
        statsmodels.tsa.statespace.mlemodel.MLEResults
        statsmodels.tsa.statespace.structural.UnobservedComponentsResults
        """
        if start_params is None:
            start_params = self.start_params
            transformed = True
            includes_fixed = True

        # Update the score method
        if optim_score is None and method == 'lbfgs':
            kwargs.setdefault('approx_grad', True)
            kwargs.setdefault('epsilon', 1e-5)
        elif optim_score is None:
            optim_score = 'approx'

        # Check for complex step differentiation
        if optim_complex_step is None:
            optim_complex_step = not self.ssm._complex_endog
        elif optim_complex_step and self.ssm._complex_endog:
            raise ValueError('Cannot use complex step derivatives when data'
                             ' or parameters are complex.')

        # Standardize starting parameters
        start_params = self.handle_params(start_params, transformed=True,
                                          includes_fixed=includes_fixed)

        # Unconstrain the starting parameters
        if transformed:
            start_params = self.untransform_params(start_params)

        # Remove any fixed parameters
        if self._has_fixed_params:
            start_params = start_params[self._free_params_index]

        # If all parameters are fixed, we are done
        if self._has_fixed_params and len(start_params) == 0:
            mlefit = Bunch(params=[], mle_retvals=None,
                           mle_settings=None)
        else:
            # Remove disallowed kwargs
            disallow = (
                "concentrate_scale",
                "enforce_stationarity",
                "enforce_invertibility"
            )
            kwargs = {k: v for k, v in kwargs.items() if k not in disallow}
            # Maximum likelihood estimation
            if flags is None:
                flags = {}
            flags.update({
                'transformed': False,
                'includes_fixed': False,
                'score_method': optim_score,
                'approx_complex_step': optim_complex_step
            })
            if optim_hessian is not None:
                flags['hessian_method'] = optim_hessian
            fargs = (flags,)
            mlefit = super(MLEModel, self).fit(start_params, method=method,
                                               fargs=fargs,
                                               maxiter=maxiter,
                                               full_output=full_output,
                                               disp=disp, callback=callback,
                                               skip_hessian=True, **kwargs)

        # Just return the fitted parameters if requested
        if return_params:
            return self.handle_params(mlefit.params, transformed=False,
                                      includes_fixed=False)
        # Otherwise construct the results class if desired
        else:
            # Handle memory conservation option
            if low_memory:
                conserve_memory = self.ssm.conserve_memory
                self.ssm.set_conserve_memory(MEMORY_CONSERVE)

            # Perform filtering / smoothing
            if (self.ssm.memory_no_predicted or self.ssm.memory_no_gain
                    or self.ssm.memory_no_smoothing):
                func = self.filter
            else:
                func = self.smooth
            res = func(mlefit.params, transformed=False, includes_fixed=False,
                       cov_type=cov_type, cov_kwds=cov_kwds)

            res.mlefit = mlefit
            res.mle_retvals = mlefit.mle_retvals
            res.mle_settings = mlefit.mle_settings

            # Reset memory conservation
            if low_memory:
                self.ssm.set_conserve_memory(conserve_memory)

            return res

    def fit_constrained(self, constraints, start_params=None, **fit_kwds):
        """
        Fit the model with some parameters subject to equality constraints.

        Parameters
        ----------
        constraints : dict
            Dictionary of constraints, of the form `param_name: fixed_value`.
            See the `param_names` property for valid parameter names.
        start_params : array_like, optional
            Initial guess of the solution for the loglikelihood maximization.
            If None, the default is given by Model.start_params.
        **fit_kwds : keyword arguments
            fit_kwds are used in the optimization of the remaining parameters.

        Returns
        -------
        results : Results instance

        Examples
        --------
        >>> mod = sm.tsa.SARIMAX(endog, order=(1, 0, 1))
        >>> res = mod.fit_constrained({'ar.L1': 0.5})
        """
        with self.fix_params(constraints):
            res = self.fit(start_params, **fit_kwds)
        return res

    @property
    def _res_classes(self):
        return {'fit': (MLEResults, MLEResultsWrapper)}

    def _wrap_results(self, params, result, return_raw, cov_type=None,
                      cov_kwds=None, results_class=None, wrapper_class=None):
        if not return_raw:
            # Wrap in a results object
            result_kwargs = {}
            if cov_type is not None:
                result_kwargs['cov_type'] = cov_type
            if cov_kwds is not None:
                result_kwargs['cov_kwds'] = cov_kwds

            if results_class is None:
                results_class = self._res_classes['fit'][0]
            if wrapper_class is None:
                wrapper_class = self._res_classes['fit'][1]

            res = results_class(self, params, result, **result_kwargs)
            result = wrapper_class(res)
        return result

    def filter(self, params, transformed=True, includes_fixed=False,
               complex_step=False, cov_type=None, cov_kwds=None,
               return_ssm=False, results_class=None,
               results_wrapper_class=None, low_memory=False, **kwargs):
        """
        Kalman filtering

        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the loglikelihood
            function.
        transformed : bool, optional
            Whether or not `params` is already transformed. Default is True.
        return_ssm : bool,optional
            Whether or not to return only the state space output or a full
            results object. Default is to return a full results object.
        cov_type : str, optional
            See `MLEResults.fit` for a description of covariance matrix types
            for results object.
        cov_kwds : dict or None, optional
            See `MLEResults.get_robustcov_results` for a description required
            keywords for alternative covariance estimators
        low_memory : bool, optional
            If set to True, techniques are applied to substantially reduce
            memory usage. If used, some features of the results object will
            not be available (including in-sample prediction), although
            out-of-sample forecasting is possible. Default is False.
        **kwargs
            Additional keyword arguments to pass to the Kalman filter. See
            `KalmanFilter.filter` for more details.
        """
        params = self.handle_params(params, transformed=transformed,
                                    includes_fixed=includes_fixed)
        self.update(params, transformed=True, includes_fixed=True,
                    complex_step=complex_step)

        # Save the parameter names
        self.data.param_names = self.param_names

        if complex_step:
            kwargs['inversion_method'] = INVERT_UNIVARIATE | SOLVE_LU

        # Handle memory conservation
        if low_memory:
            kwargs['conserve_memory'] = MEMORY_CONSERVE

        # Get the state space output
        result = self.ssm.filter(complex_step=complex_step, **kwargs)

        # Wrap in a results object
        return self._wrap_results(params, result, return_ssm, cov_type,
                                  cov_kwds, results_class,
                                  results_wrapper_class)

    def smooth(self, params, transformed=True, includes_fixed=False,
               complex_step=False, cov_type=None, cov_kwds=None,
               return_ssm=False, results_class=None,
               results_wrapper_class=None, **kwargs):
        """
        Kalman smoothing

        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the loglikelihood
            function.
        transformed : bool, optional
            Whether or not `params` is already transformed. Default is True.
        return_ssm : bool,optional
            Whether or not to return only the state space output or a full
            results object. Default is to return a full results object.
        cov_type : str, optional
            See `MLEResults.fit` for a description of covariance matrix types
            for results object.
        cov_kwds : dict or None, optional
            See `MLEResults.get_robustcov_results` for a description required
            keywords for alternative covariance estimators
        **kwargs
            Additional keyword arguments to pass to the Kalman filter. See
            `KalmanFilter.filter` for more details.
        """
        params = self.handle_params(params, transformed=transformed,
                                    includes_fixed=includes_fixed)
        self.update(params, transformed=True, includes_fixed=True,
                    complex_step=complex_step)

        # Save the parameter names
        self.data.param_names = self.param_names

        if complex_step:
            kwargs['inversion_method'] = INVERT_UNIVARIATE | SOLVE_LU

        # Get the state space output
        result = self.ssm.smooth(complex_step=complex_step, **kwargs)

        # Wrap in a results object
        return self._wrap_results(params, result, return_ssm, cov_type,
                                  cov_kwds, results_class,
                                  results_wrapper_class)

    _loglike_param_names = ['transformed', 'includes_fixed', 'complex_step']
    _loglike_param_defaults = [True, False, False]

    def loglike(self, params, *args, **kwargs):
        """
        Loglikelihood evaluation

        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the loglikelihood
            function.
        transformed : bool, optional
            Whether or not `params` is already transformed. Default is True.
        **kwargs
            Additional keyword arguments to pass to the Kalman filter. See
            `KalmanFilter.filter` for more details.

        See Also
        --------
        update : modifies the internal state of the state space model to
                 reflect new params

        Notes
        -----
        [1]_ recommend maximizing the average likelihood to avoid scale issues;
        this is done automatically by the base Model fit method.

        References
        ----------
        .. [1] Koopman, Siem Jan, Neil Shephard, and Jurgen A. Doornik. 1999.
           Statistical Algorithms for Models in State Space Using SsfPack 2.2.
           Econometrics Journal 2 (1): 107-60. doi:10.1111/1368-423X.00023.
        """
        transformed, includes_fixed, complex_step, kwargs = _handle_args(
            MLEModel._loglike_param_names, MLEModel._loglike_param_defaults,
            *args, **kwargs)

        params = self.handle_params(params, transformed=transformed,
                                    includes_fixed=includes_fixed)
        self.update(params, transformed=True, includes_fixed=True,
                    complex_step=complex_step)

        if complex_step:
            kwargs['inversion_method'] = INVERT_UNIVARIATE | SOLVE_LU

        loglike = self.ssm.loglike(complex_step=complex_step, **kwargs)

        # Koopman, Shephard, and Doornik recommend maximizing the average
        # likelihood to avoid scale issues, but the averaging is done
        # automatically in the base model `fit` method
        return loglike

    def loglikeobs(self, params, transformed=True, includes_fixed=False,
                   complex_step=False, **kwargs):
        """
        Loglikelihood evaluation

        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the loglikelihood
            function.
        transformed : bool, optional
            Whether or not `params` is already transformed. Default is True.
        **kwargs
            Additional keyword arguments to pass to the Kalman filter. See
            `KalmanFilter.filter` for more details.

        See Also
        --------
        update : modifies the internal state of the Model to reflect new params

        Notes
        -----
        [1]_ recommend maximizing the average likelihood to avoid scale issues;
        this is done automatically by the base Model fit method.

        References
        ----------
        .. [1] Koopman, Siem Jan, Neil Shephard, and Jurgen A. Doornik. 1999.
           Statistical Algorithms for Models in State Space Using SsfPack 2.2.
           Econometrics Journal 2 (1): 107-60. doi:10.1111/1368-423X.00023.
        """
        params = self.handle_params(params, transformed=transformed,
                                    includes_fixed=includes_fixed)

        # If we're using complex-step differentiation, then we cannot use
        # Cholesky factorization
        if complex_step:
            kwargs['inversion_method'] = INVERT_UNIVARIATE | SOLVE_LU

        self.update(params, transformed=True, includes_fixed=True,
                    complex_step=complex_step)

        return self.ssm.loglikeobs(complex_step=complex_step, **kwargs)

    def simulation_smoother(self, simulation_output=None, **kwargs):
        r"""
        Retrieve a simulation smoother for the state space model.

        Parameters
        ----------
        simulation_output : int, optional
            Determines which simulation smoother output is calculated.
            Default is all (including state and disturbances).
        **kwargs
            Additional keyword arguments, used to set the simulation output.
            See `set_simulation_output` for more details.

        Returns
        -------
        SimulationSmoothResults
        """
        return self.ssm.simulation_smoother(
            simulation_output=simulation_output, **kwargs)

    def _forecasts_error_partial_derivatives(self, params, transformed=True,
                                             includes_fixed=False,
                                             approx_complex_step=None,
                                             approx_centered=False,
                                             res=None, **kwargs):
        params = np.array(params, ndmin=1)

        # We cannot use complex-step differentiation with non-transformed
        # parameters
        if approx_complex_step is None:
            approx_complex_step = transformed
        if not transformed and approx_complex_step:
            raise ValueError("Cannot use complex-step approximations to"
                             " calculate the observed_information_matrix"
                             " with untransformed parameters.")

        # If we're using complex-step differentiation, then we cannot use
        # Cholesky factorization
        if approx_complex_step:
            kwargs['inversion_method'] = INVERT_UNIVARIATE | SOLVE_LU

        # Get values at the params themselves
        if res is None:
            self.update(params, transformed=transformed,
                        includes_fixed=includes_fixed,
                        complex_step=approx_complex_step)
            res = self.ssm.filter(complex_step=approx_complex_step, **kwargs)

        # Setup
        n = len(params)

        # Compute partial derivatives w.r.t. forecast error and forecast
        # error covariance
        partials_forecasts_error = (
            np.zeros((self.k_endog, self.nobs, n))
        )
        partials_forecasts_error_cov = (
            np.zeros((self.k_endog, self.k_endog, self.nobs, n))
        )
        if approx_complex_step:
            epsilon = _get_epsilon(params, 2, None, n)
            increments = np.identity(n) * 1j * epsilon

            for i, ih in enumerate(increments):
                self.update(params + ih, transformed=transformed,
                            includes_fixed=includes_fixed,
                            complex_step=True)
                _res = self.ssm.filter(complex_step=True, **kwargs)

                partials_forecasts_error[:, :, i] = (
                    _res.forecasts_error.imag / epsilon[i]
                )

                partials_forecasts_error_cov[:, :, :, i] = (
                    _res.forecasts_error_cov.imag / epsilon[i]
                )
        elif not approx_centered:
            epsilon = _get_epsilon(params, 2, None, n)
            ei = np.zeros((n,), float)
            for i in range(n):
                ei[i] = epsilon[i]
                self.update(params + ei, transformed=transformed,
                            includes_fixed=includes_fixed, complex_step=False)
                _res = self.ssm.filter(complex_step=False, **kwargs)

                partials_forecasts_error[:, :, i] = (
                    _res.forecasts_error - res.forecasts_error) / epsilon[i]

                partials_forecasts_error_cov[:, :, :, i] = (
                    _res.forecasts_error_cov -
                    res.forecasts_error_cov) / epsilon[i]
                ei[i] = 0.0
        else:
            epsilon = _get_epsilon(params, 3, None, n) / 2.
            ei = np.zeros((n,), float)
            for i in range(n):
                ei[i] = epsilon[i]

                self.update(params + ei, transformed=transformed,
                            includes_fixed=includes_fixed, complex_step=False)
                _res1 = self.ssm.filter(complex_step=False, **kwargs)

                self.update(params - ei, transformed=transformed,
                            includes_fixed=includes_fixed, complex_step=False)
                _res2 = self.ssm.filter(complex_step=False, **kwargs)

                partials_forecasts_error[:, :, i] = (
                    (_res1.forecasts_error - _res2.forecasts_error) /
                    (2 * epsilon[i]))

                partials_forecasts_error_cov[:, :, :, i] = (
                    (_res1.forecasts_error_cov - _res2.forecasts_error_cov) /
                    (2 * epsilon[i]))

                ei[i] = 0.0

        return partials_forecasts_error, partials_forecasts_error_cov

    def observed_information_matrix(self, params, transformed=True,
                                    includes_fixed=False,
                                    approx_complex_step=None,
                                    approx_centered=False, **kwargs):
        """
        Observed information matrix

        Parameters
        ----------
        params : array_like, optional
            Array of parameters at which to evaluate the loglikelihood
            function.
        **kwargs
            Additional keyword arguments to pass to the Kalman filter. See
            `KalmanFilter.filter` for more details.

        Notes
        -----
        This method is from Harvey (1989), which shows that the information
        matrix only depends on terms from the gradient. This implementation is
        partially analytic and partially numeric approximation, therefore,
        because it uses the analytic formula for the information matrix, with
        numerically computed elements of the gradient.

        References
        ----------
        Harvey, Andrew C. 1990.
        Forecasting, Structural Time Series Models and the Kalman Filter.
        Cambridge University Press.
        """
        params = np.array(params, ndmin=1)

        # Setup
        n = len(params)

        # We cannot use complex-step differentiation with non-transformed
        # parameters
        if approx_complex_step is None:
            approx_complex_step = transformed
        if not transformed and approx_complex_step:
            raise ValueError("Cannot use complex-step approximations to"
                             " calculate the observed_information_matrix"
                             " with untransformed parameters.")

        # Get values at the params themselves
        params = self.handle_params(params, transformed=transformed,
                                    includes_fixed=includes_fixed)
        self.update(params, transformed=True, includes_fixed=True,
                    complex_step=approx_complex_step)
        # If we're using complex-step differentiation, then we cannot use
        # Cholesky factorization
        if approx_complex_step:
            kwargs['inversion_method'] = INVERT_UNIVARIATE | SOLVE_LU
        res = self.ssm.filter(complex_step=approx_complex_step, **kwargs)
        dtype = self.ssm.dtype

        # Save this for inversion later
        inv_forecasts_error_cov = res.forecasts_error_cov.copy()

        partials_forecasts_error, partials_forecasts_error_cov = (
            self._forecasts_error_partial_derivatives(
                params, transformed=transformed, includes_fixed=includes_fixed,
                approx_complex_step=approx_complex_step,
                approx_centered=approx_centered, res=res, **kwargs))

        # Compute the information matrix
        tmp = np.zeros((self.k_endog, self.k_endog, self.nobs, n), dtype=dtype)

        information_matrix = np.zeros((n, n), dtype=dtype)
        d = np.maximum(self.ssm.loglikelihood_burn, res.nobs_diffuse)
        for t in range(d, self.nobs):
            inv_forecasts_error_cov[:, :, t] = (
                np.linalg.inv(res.forecasts_error_cov[:, :, t])
            )
            for i in range(n):
                tmp[:, :, t, i] = np.dot(
                    inv_forecasts_error_cov[:, :, t],
                    partials_forecasts_error_cov[:, :, t, i]
                )
            for i in range(n):
                for j in range(n):
                    information_matrix[i, j] += (
                        0.5 * np.trace(np.dot(tmp[:, :, t, i],
                                              tmp[:, :, t, j]))
                    )
                    information_matrix[i, j] += np.inner(
                        partials_forecasts_error[:, t, i],
                        np.dot(inv_forecasts_error_cov[:, :, t],
                               partials_forecasts_error[:, t, j])
                    )
        return information_matrix / (self.nobs - self.ssm.loglikelihood_burn)

    def opg_information_matrix(self, params, transformed=True,
                               includes_fixed=False, approx_complex_step=None,
                               **kwargs):
        """
        Outer product of gradients information matrix

        Parameters
        ----------
        params : array_like, optional
            Array of parameters at which to evaluate the loglikelihood
            function.
        **kwargs
            Additional arguments to the `loglikeobs` method.

        References
        ----------
        Berndt, Ernst R., Bronwyn Hall, Robert Hall, and Jerry Hausman. 1974.
        Estimation and Inference in Nonlinear Structural Models.
        NBER Chapters. National Bureau of Economic Research, Inc.
        """
        # We cannot use complex-step differentiation with non-transformed
        # parameters
        if approx_complex_step is None:
            approx_complex_step = transformed
        if not transformed and approx_complex_step:
            raise ValueError("Cannot use complex-step approximations to"
                             " calculate the observed_information_matrix"
                             " with untransformed parameters.")

        score_obs = self.score_obs(params, transformed=transformed,
                                   includes_fixed=includes_fixed,
                                   approx_complex_step=approx_complex_step,
                                   **kwargs).transpose()
        return (
            np.inner(score_obs, score_obs) /
            (self.nobs - self.ssm.loglikelihood_burn)
        )

    def _score_complex_step(self, params, **kwargs):
        # the default epsilon can be too small
        # inversion_method = INVERT_UNIVARIATE | SOLVE_LU
        epsilon = _get_epsilon(params, 2., None, len(params))
        kwargs['transformed'] = True
        kwargs['complex_step'] = True
        return approx_fprime_cs(params, self.loglike, epsilon=epsilon,
                                kwargs=kwargs)

    def _score_finite_difference(self, params, approx_centered=False,
                                 **kwargs):
        kwargs['transformed'] = True
        return approx_fprime(params, self.loglike, kwargs=kwargs,
                             centered=approx_centered)

    def _score_harvey(self, params, approx_complex_step=True, **kwargs):
        score_obs = self._score_obs_harvey(
            params, approx_complex_step=approx_complex_step, **kwargs)
        return np.sum(score_obs, axis=0)

    def _score_obs_harvey(self, params, approx_complex_step=True,
                          approx_centered=False, includes_fixed=False,
                          **kwargs):
        """
        Score

        Parameters
        ----------
        params : array_like, optional
            Array of parameters at which to evaluate the loglikelihood
            function.
        **kwargs
            Additional keyword arguments to pass to the Kalman filter. See
            `KalmanFilter.filter` for more details.

        Notes
        -----
        This method is from Harvey (1989), section 3.4.5

        References
        ----------
        Harvey, Andrew C. 1990.
        Forecasting, Structural Time Series Models and the Kalman Filter.
        Cambridge University Press.
        """
        params = np.array(params, ndmin=1)
        n = len(params)

        # Get values at the params themselves
        self.update(params, transformed=True, includes_fixed=includes_fixed,
                    complex_step=approx_complex_step)
        if approx_complex_step:
            kwargs['inversion_method'] = INVERT_UNIVARIATE | SOLVE_LU
        if 'transformed' in kwargs:
            del kwargs['transformed']
        res = self.ssm.filter(complex_step=approx_complex_step, **kwargs)

        # Get forecasts error partials
        partials_forecasts_error, partials_forecasts_error_cov = (
            self._forecasts_error_partial_derivatives(
                params, transformed=True, includes_fixed=includes_fixed,
                approx_complex_step=approx_complex_step,
                approx_centered=approx_centered, res=res, **kwargs))

        # Compute partial derivatives w.r.t. likelihood function
        partials = np.zeros((self.nobs, n))
        k_endog = self.k_endog
        for t in range(self.nobs):
            inv_forecasts_error_cov = np.linalg.inv(
                    res.forecasts_error_cov[:, :, t])

            for i in range(n):
                partials[t, i] += np.trace(np.dot(
                    np.dot(inv_forecasts_error_cov,
                           partials_forecasts_error_cov[:, :, t, i]),
                    (np.eye(k_endog) -
                     np.dot(inv_forecasts_error_cov,
                            np.outer(res.forecasts_error[:, t],
                                     res.forecasts_error[:, t])))))
                # 2 * dv / di * F^{-1} v_t
                # where x = F^{-1} v_t or F x = v
                partials[t, i] += 2 * np.dot(
                    partials_forecasts_error[:, t, i],
                    np.dot(inv_forecasts_error_cov, res.forecasts_error[:, t]))

        return -partials / 2.

    _score_param_names = ['transformed', 'includes_fixed', 'score_method',
                          'approx_complex_step', 'approx_centered']
    _score_param_defaults = [True, False, 'approx', None, False]

    def score(self, params, *args, **kwargs):
        """
        Compute the score function at params.

        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the score.
        *args
            Additional positional arguments to the `loglike` method.
        **kwargs
            Additional keyword arguments to the `loglike` method.

        Returns
        -------
        score : ndarray
            Score, evaluated at `params`.

        Notes
        -----
        This is a numerical approximation, calculated using first-order complex
        step differentiation on the `loglike` method.

        Both args and kwargs are necessary because the optimizer from
        `fit` must call this function and only supports passing arguments via
        args (for example `scipy.optimize.fmin_l_bfgs`).
        """
        (transformed, includes_fixed, method, approx_complex_step,
         approx_centered, kwargs) = (
            _handle_args(MLEModel._score_param_names,
                         MLEModel._score_param_defaults, *args, **kwargs))
        # For fit() calls, the method is called 'score_method' (to distinguish
        # it from the method used for fit) but generally in kwargs the method
        # will just be called 'method'
        if 'method' in kwargs:
            method = kwargs.pop('method')

        if approx_complex_step is None:
            approx_complex_step = not self.ssm._complex_endog
        if approx_complex_step and self.ssm._complex_endog:
            raise ValueError('Cannot use complex step derivatives when data'
                             ' or parameters are complex.')

        out = self.handle_params(
            params, transformed=transformed, includes_fixed=includes_fixed,
            return_jacobian=not transformed)
        if transformed:
            params = out
        else:
            params, transform_score = out

        if method == 'harvey':
            kwargs['includes_fixed'] = True
            score = self._score_harvey(
                params, approx_complex_step=approx_complex_step, **kwargs)
        elif method == 'approx' and approx_complex_step:
            kwargs['includes_fixed'] = True
            score = self._score_complex_step(params, **kwargs)
        elif method == 'approx':
            kwargs['includes_fixed'] = True
            score = self._score_finite_difference(
                params, approx_centered=approx_centered, **kwargs)
        else:
            raise NotImplementedError('Invalid score method.')

        if not transformed:
            score = np.dot(transform_score, score)

        if self._has_fixed_params and not includes_fixed:
            score = score[self._free_params_index]

        return score

    def score_obs(self, params, method='approx', transformed=True,
                  includes_fixed=False, approx_complex_step=None,
                  approx_centered=False, **kwargs):
        """
        Compute the score per observation, evaluated at params

        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the score.
        **kwargs
            Additional arguments to the `loglike` method.

        Returns
        -------
        score : ndarray
            Score per observation, evaluated at `params`.

        Notes
        -----
        This is a numerical approximation, calculated using first-order complex
        step differentiation on the `loglikeobs` method.
        """
        if not transformed and approx_complex_step:
            raise ValueError("Cannot use complex-step approximations to"
                             " calculate the score at each observation"
                             " with untransformed parameters.")

        if approx_complex_step is None:
            approx_complex_step = not self.ssm._complex_endog
        if approx_complex_step and self.ssm._complex_endog:
            raise ValueError('Cannot use complex step derivatives when data'
                             ' or parameters are complex.')

        params = self.handle_params(params, transformed=True,
                                    includes_fixed=includes_fixed)
        kwargs['transformed'] = transformed
        kwargs['includes_fixed'] = True

        if method == 'harvey':
            score = self._score_obs_harvey(
                params, approx_complex_step=approx_complex_step, **kwargs)
        elif method == 'approx' and approx_complex_step:
            # the default epsilon can be too small
            epsilon = _get_epsilon(params, 2., None, len(params))
            kwargs['complex_step'] = True
            score = approx_fprime_cs(params, self.loglikeobs, epsilon=epsilon,
                                     kwargs=kwargs)
        elif method == 'approx':
            score = approx_fprime(params, self.loglikeobs, kwargs=kwargs,
                                  centered=approx_centered)
        else:
            raise NotImplementedError('Invalid scoreobs method.')

        return score

    _hessian_param_names = ['transformed', 'hessian_method',
                            'approx_complex_step', 'approx_centered']
    _hessian_param_defaults = [True, 'approx', None, False]

    def hessian(self, params, *args, **kwargs):
        r"""
        Hessian matrix of the likelihood function, evaluated at the given
        parameters

        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the hessian.
        *args
            Additional positional arguments to the `loglike` method.
        **kwargs
            Additional keyword arguments to the `loglike` method.

        Returns
        -------
        hessian : ndarray
            Hessian matrix evaluated at `params`

        Notes
        -----
        This is a numerical approximation.

        Both args and kwargs are necessary because the optimizer from
        `fit` must call this function and only supports passing arguments via
        args (for example `scipy.optimize.fmin_l_bfgs`).
        """
        transformed, method, approx_complex_step, approx_centered, kwargs = (
            _handle_args(MLEModel._hessian_param_names,
                         MLEModel._hessian_param_defaults,
                         *args, **kwargs))
        # For fit() calls, the method is called 'hessian_method' (to
        # distinguish it from the method used for fit) but generally in kwargs
        # the method will just be called 'method'
        if 'method' in kwargs:
            method = kwargs.pop('method')

        if not transformed and approx_complex_step:
            raise ValueError("Cannot use complex-step approximations to"
                             " calculate the hessian with untransformed"
                             " parameters.")

        if approx_complex_step is None:
            approx_complex_step = not self.ssm._complex_endog
        if approx_complex_step and self.ssm._complex_endog:
            raise ValueError('Cannot use complex step derivatives when data'
                             ' or parameters are complex.')

        if method == 'oim':
            hessian = self._hessian_oim(
                params, transformed=transformed,
                approx_complex_step=approx_complex_step,
                approx_centered=approx_centered, **kwargs)
        elif method == 'opg':
            hessian = self._hessian_opg(
                params, transformed=transformed,
                approx_complex_step=approx_complex_step,
                approx_centered=approx_centered, **kwargs)
        elif method == 'approx' and approx_complex_step:
            hessian = self._hessian_complex_step(
                params, transformed=transformed, **kwargs)
        elif method == 'approx':
            hessian = self._hessian_finite_difference(
                params, transformed=transformed,
                approx_centered=approx_centered, **kwargs)
        else:
            raise NotImplementedError('Invalid Hessian calculation method.')
        return hessian

    def _hessian_oim(self, params, **kwargs):
        """
        Hessian matrix computed using the Harvey (1989) information matrix
        """
        return -self.observed_information_matrix(params, **kwargs)

    def _hessian_opg(self, params, **kwargs):
        """
        Hessian matrix computed using the outer product of gradients
        information matrix
        """
        return -self.opg_information_matrix(params, **kwargs)

    def _hessian_finite_difference(self, params, approx_centered=False,
                                   **kwargs):
        params = np.array(params, ndmin=1)

        warnings.warn('Calculation of the Hessian using finite differences'
                      ' is usually subject to substantial approximation'
                      ' errors.', PrecisionWarning)

        if not approx_centered:
            epsilon = _get_epsilon(params, 3, None, len(params))
        else:
            epsilon = _get_epsilon(params, 4, None, len(params)) / 2
        hessian = approx_fprime(params, self._score_finite_difference,
                                epsilon=epsilon, kwargs=kwargs,
                                centered=approx_centered)

        return hessian / (self.nobs - self.ssm.loglikelihood_burn)

    def _hessian_complex_step(self, params, **kwargs):
        """
        Hessian matrix computed by second-order complex-step differentiation
        on the `loglike` function.
        """
        # the default epsilon can be too small
        epsilon = _get_epsilon(params, 3., None, len(params))
        kwargs['transformed'] = True
        kwargs['complex_step'] = True
        hessian = approx_hess_cs(
            params, self.loglike, epsilon=epsilon, kwargs=kwargs)

        return hessian / (self.nobs - self.ssm.loglikelihood_burn)

    @property
    def start_params(self):
        """
        (array) Starting parameters for maximum likelihood estimation.
        """
        if hasattr(self, '_start_params'):
            return self._start_params
        else:
            raise NotImplementedError

    @property
    def param_names(self):
        """
        (list of str) List of human readable parameter names (for parameters
        actually included in the model).
        """
        if hasattr(self, '_param_names'):
            return self._param_names
        else:
            try:
                names = ['param.%d' % i for i in range(len(self.start_params))]
            except NotImplementedError:
                names = []
            return names

    @property
    def state_names(self):
        """
        (list of str) List of human readable names for unobserved states.
        """
        if hasattr(self, '_state_names'):
            return self._state_names
        else:
            names = ['state.%d' % i for i in range(self.k_states)]
        return names

    def transform_jacobian(self, unconstrained, approx_centered=False):
        """
        Jacobian matrix for the parameter transformation function

        Parameters
        ----------
        unconstrained : array_like
            Array of unconstrained parameters used by the optimizer.

        Returns
        -------
        jacobian : ndarray
            Jacobian matrix of the transformation, evaluated at `unconstrained`

        See Also
        --------
        transform_params

        Notes
        -----
        This is a numerical approximation using finite differences. Note that
        in general complex step methods cannot be used because it is not
        guaranteed that the `transform_params` method is a real function (e.g.
        if Cholesky decomposition is used).
        """
        return approx_fprime(unconstrained, self.transform_params,
                             centered=approx_centered)

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
        This is a noop in the base class, subclasses should override where
        appropriate.
        """
        return np.array(unconstrained, ndmin=1)

    def untransform_params(self, constrained):
        """
        Transform constrained parameters used in likelihood evaluation
        to unconstrained parameters used by the optimizer

        Parameters
        ----------
        constrained : array_like
            Array of constrained parameters used in likelihood evaluation, to
            be transformed.

        Returns
        -------
        unconstrained : array_like
            Array of unconstrained parameters used by the optimizer.

        Notes
        -----
        This is a noop in the base class, subclasses should override where
        appropriate.
        """
        return np.array(constrained, ndmin=1)

    def handle_params(self, params, transformed=True, includes_fixed=False,
                      return_jacobian=False):
        """
        Ensure model parameters satisfy shape and other requirements
        """
        params = np.array(params, ndmin=1)

        # Never want integer dtype, so convert to floats
        if np.issubdtype(params.dtype, np.integer):
            params = params.astype(np.float64)

        if not includes_fixed and self._has_fixed_params:
            k_params = len(self.param_names)
            new_params = np.zeros(k_params, dtype=params.dtype) * np.nan
            new_params[self._free_params_index] = params
            params = new_params

        if not transformed:
            # It may be the case that the transformation relies on having
            # "some" (non-NaN) values for the fixed parameters, even if we will
            # not actually be transforming the fixed parameters (as they will)
            # be set below regardless
            if not includes_fixed and self._has_fixed_params:
                params[self._fixed_params_index] = (
                    list(self._fixed_params.values()))

            if return_jacobian:
                transform_score = self.transform_jacobian(params)
            params = self.transform_params(params)

        if not includes_fixed and self._has_fixed_params:
            params[self._fixed_params_index] = (
                list(self._fixed_params.values()))

        return (params, transform_score) if return_jacobian else params

    def update(self, params, transformed=True, includes_fixed=False,
               complex_step=False):
        """
        Update the parameters of the model

        Parameters
        ----------
        params : array_like
            Array of new parameters.
        transformed : bool, optional
            Whether or not `params` is already transformed. If set to False,
            `transform_params` is called. Default is True.

        Returns
        -------
        params : array_like
            Array of parameters.

        Notes
        -----
        Since Model is a base class, this method should be overridden by
        subclasses to perform actual updating steps.
        """
        return self.handle_params(params=params, transformed=transformed,
                                  includes_fixed=includes_fixed)

    def _validate_out_of_sample_exog(self, exog, out_of_sample):
        """
        Validate given `exog` as satisfactory for out-of-sample operations

        Parameters
        ----------
        exog : array_like or None
            New observations of exogenous regressors, if applicable.
        out_of_sample : int
            Number of new observations required.

        Returns
        -------
        exog : array or None
            A numpy array of shape (out_of_sample, k_exog) if the model
            contains an `exog` component, or None if it does not.
        """
        k_exog = getattr(self, 'k_exog', 0)
        if out_of_sample and k_exog > 0:
            if exog is None:
                raise ValueError('Out-of-sample operations in a model'
                                 ' with a regression component require'
                                 ' additional exogenous values via the'
                                 ' `exog` argument.')
            exog = np.array(exog)
            required_exog_shape = (out_of_sample, self.k_exog)
            try:
                exog = exog.reshape(required_exog_shape)
            except ValueError:
                raise ValueError('Provided exogenous values are not of the'
                                 ' appropriate shape. Required %s, got %s.'
                                 % (str(required_exog_shape),
                                    str(exog.shape)))
        elif k_exog > 0 and exog is not None:
            exog = None
            warnings.warn('Exogenous array provided, but additional data'
                          ' is not required. `exog` argument ignored.',
                          ValueWarning)

        return exog

    def _get_extension_time_varying_matrices(
            self, params, exog, out_of_sample, extend_kwargs=None,
            transformed=True, includes_fixed=False, **kwargs):
        """
        Get updated time-varying state space system matrices

        Parameters
        ----------
        params : array_like
            Array of parameters used to construct the time-varying system
            matrices.
        exog : array_like or None
            New observations of exogenous regressors, if applicable.
        out_of_sample : int
            Number of new observations required.
        extend_kwargs : dict, optional
            Dictionary of keyword arguments to pass to the state space model
            constructor. For example, for an SARIMAX state space model, this
            could be used to pass the `concentrate_scale=True` keyword
            argument. Any arguments that are not explicitly set in this
            dictionary will be copied from the current model instance.
        transformed : bool, optional
            Whether or not `start_params` is already transformed. Default is
            True.
        includes_fixed : bool, optional
            If parameters were previously fixed with the `fix_params` method,
            this argument describes whether or not `start_params` also includes
            the fixed parameters, in addition to the free parameters. Default
            is False.
        """
        # Get the appropriate exog for the extended sample
        exog = self._validate_out_of_sample_exog(exog, out_of_sample)

        # Create extended model
        if extend_kwargs is None:
            extend_kwargs = {}

        # Handle trend offset for extended model
        if getattr(self, 'k_trend', 0) > 0 and hasattr(self, 'trend_offset'):
            extend_kwargs.setdefault(
                'trend_offset', self.trend_offset + self.nobs)

        mod_extend = self.clone(
            endog=np.zeros((out_of_sample, self.k_endog)), exog=exog,
            **extend_kwargs)
        mod_extend.update(params, transformed=transformed,
                          includes_fixed=includes_fixed)

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

    def simulate(self, params, nsimulations, measurement_shocks=None,
                 state_shocks=None, initial_state=None, anchor=None,
                 repetitions=None, exog=None, extend_model=None,
                 extend_kwargs=None, transformed=True, includes_fixed=False,
                 pretransformed_measurement_shocks=True,
                 pretransformed_state_shocks=True,
                 pretransformed_initial_state=True, random_state=None,
                 **kwargs):
        r"""
        Simulate a new time series following the state space model

        Parameters
        ----------
        params : array_like
            Array of parameters to use in constructing the state space
            representation to use when simulating.
        nsimulations : int
            The number of observations to simulate. If the model is
            time-invariant this can be any number. If the model is
            time-varying, then this number must be less than or equal to the
            number of observations.
        measurement_shocks : array_like, optional
            If specified, these are the shocks to the measurement equation,
            :math:`\varepsilon_t`. If unspecified, these are automatically
            generated using a pseudo-random number generator. If specified,
            must be shaped `nsimulations` x `k_endog`, where `k_endog` is the
            same as in the state space model.
        state_shocks : array_like, optional
            If specified, these are the shocks to the state equation,
            :math:`\eta_t`. If unspecified, these are automatically
            generated using a pseudo-random number generator. If specified,
            must be shaped `nsimulations` x `k_posdef` where `k_posdef` is the
            same as in the state space model.
        initial_state : array_like, optional
            If specified, this is the initial state vector to use in
            simulation, which should be shaped (`k_states` x 1), where
            `k_states` is the same as in the state space model. If unspecified,
            but the model has been initialized, then that initialization is
            used. This must be specified if `anchor` is anything other than
            "start" or 0 (or else you can use the `simulate` method on a
            results object rather than on the model object).
        anchor : int, str, or datetime, optional
            First period for simulation. The simulation will be conditional on
            all existing datapoints prior to the `anchor`.  Type depends on the
            index of the given `endog` in the model. Two special cases are the
            strings 'start' and 'end'. `start` refers to beginning the
            simulation at the first period of the sample, and `end` refers to
            beginning the simulation at the first period after the sample.
            Integer values can run from 0 to `nobs`, or can be negative to
            apply negative indexing. Finally, if a date/time index was provided
            to the model, then this argument can be a date string to parse or a
            datetime type. Default is 'start'.
        repetitions : int, optional
            Number of simulated paths to generate. Default is 1 simulated path.
        exog : array_like, optional
            New observations of exogenous regressors, if applicable.
        transformed : bool, optional
            Whether or not `params` is already transformed. Default is
            True.
        includes_fixed : bool, optional
            If parameters were previously fixed with the `fix_params` method,
            this argument describes whether or not `params` also includes
            the fixed parameters, in addition to the free parameters. Default
            is False.
        pretransformed_measurement_shocks : bool, optional
            If `measurement_shocks` is provided, this flag indicates whether it
            should be directly used as the shocks. If False, then it is assumed
            to contain draws from the standard Normal distribution that must be
            transformed using the `obs_cov` covariance matrix. Default is True.
        pretransformed_state_shocks : bool, optional
            If `state_shocks` is provided, this flag indicates whether it
            should be directly used as the shocks. If False, then it is assumed
            to contain draws from the standard Normal distribution that must be
            transformed using the `state_cov` covariance matrix. Default is
            True.
        pretransformed_initial_state : bool, optional
            If `initial_state` is provided, this flag indicates whether it
            should be directly used as the initial_state. If False, then it is
            assumed to contain draws from the standard Normal distribution that
            must be transformed using the `initial_state_cov` covariance
            matrix. Default is True.
        random_state : {None, int, Generator, RandomState}, optional
            If `seed` is None (or `np.random`), the
            class:``~numpy.random.RandomState`` singleton is used.
            If `seed` is an int, a new class:``~numpy.random.RandomState``
            instance is used, seeded with `seed`.
            If `seed` is already a class:``~numpy.random.Generator`` or
            class:``~numpy.random.RandomState`` instance then that instance is
            used.

        Returns
        -------
        simulated_obs : ndarray
            An array of simulated observations. If `repetitions=None`, then it
            will be shaped (nsimulations x k_endog) or (nsimulations,) if
            `k_endog=1`. Otherwise it will be shaped
            (nsimulations x k_endog x repetitions). If the model was given
            Pandas input then the output will be a Pandas object. If
            `k_endog > 1` and `repetitions` is not None, then the output will
            be a Pandas DataFrame that has a MultiIndex for the columns, with
            the first level containing the names of the `endog` variables and
            the second level containing the repetition number.

        See Also
        --------
        impulse_responses
            Impulse response functions
        """
        # Make sure the model class has the current parameters
        self.update(params, transformed=transformed,
                    includes_fixed=includes_fixed)

        # Get the starting location
        if anchor is None or anchor == 'start':
            iloc = 0
        elif anchor == 'end':
            iloc = self.nobs
        else:
            iloc, _, _ = self._get_index_loc(anchor)
            if isinstance(iloc, slice):
                iloc = iloc.start

        if iloc < 0:
            iloc = self.nobs + iloc
        if iloc > self.nobs:
            raise ValueError('Cannot anchor simulation outside of the sample.')

        if iloc > 0 and initial_state is None:
            raise ValueError('If `anchor` is after the start of the sample,'
                             ' must provide a value for `initial_state`.')

        # Get updated time-varying system matrices in **kwargs, if necessary
        out_of_sample = max(iloc + nsimulations - self.nobs, 0)
        if extend_model is None:
            extend_model = self.exog is not None or not self.ssm.time_invariant
        if out_of_sample and extend_model:
            kwargs = self._get_extension_time_varying_matrices(
                params, exog, out_of_sample, extend_kwargs,
                transformed=transformed, includes_fixed=includes_fixed,
                **kwargs)

        # Standardize the dimensions of the initial state
        if initial_state is not None:
            initial_state = np.array(initial_state)
            if initial_state.ndim < 2:
                initial_state = np.atleast_2d(initial_state).T

        # Construct a model that represents the simulation period
        end = min(self.nobs, iloc + nsimulations)
        nextend = iloc + nsimulations - end
        sim_model = self.ssm.extend(np.zeros((nextend, self.k_endog)),
                                    start=iloc, end=end, **kwargs)

        # Simulate the data
        _repetitions = 1 if repetitions is None else repetitions
        sim = np.zeros((nsimulations, self.k_endog, _repetitions))
        simulator = None

        for i in range(_repetitions):
            initial_state_variates = None
            if initial_state is not None:
                if initial_state.shape[1] == 1:
                    initial_state_variates = initial_state[:, 0]
                else:
                    initial_state_variates = initial_state[:, i]

            # TODO: allow specifying measurement / state shocks for each
            # repetition?

            out, _, simulator = sim_model.simulate(
                nsimulations, measurement_shocks, state_shocks,
                initial_state_variates,
                pretransformed_measurement_shocks=(
                    pretransformed_measurement_shocks),
                pretransformed_state_shocks=pretransformed_state_shocks,
                pretransformed_initial_state=pretransformed_initial_state,
                simulator=simulator, return_simulator=True,
                random_state=random_state)

            sim[:, :, i] = out

        # Wrap data / squeeze where appropriate
        use_pandas = isinstance(self.data, PandasData)
        index = None
        if use_pandas:
            _, _, _, index = self._get_prediction_index(
                iloc, iloc + nsimulations - 1)
        # If `repetitions` isn't set, we squeeze the last dimension(s)
        if repetitions is None:
            if self.k_endog == 1:
                sim = sim[:, 0, 0]
                if use_pandas:
                    sim = pd.Series(sim, index=index, name=self.endog_names)
            else:
                sim = sim[:, :, 0]
                if use_pandas:
                    sim = pd.DataFrame(sim, index=index,
                                       columns=self.endog_names)
        elif use_pandas:
            shape = sim.shape
            endog_names = self.endog_names
            if not isinstance(endog_names, list):
                endog_names = [endog_names]
            columns = pd.MultiIndex.from_product([endog_names,
                                                  np.arange(shape[2])])
            sim = pd.DataFrame(sim.reshape(shape[0], shape[1] * shape[2]),
                               index=index, columns=columns)

        return sim

    def impulse_responses(self, params, steps=1, impulse=0,
                          orthogonalized=False, cumulative=False, anchor=None,
                          exog=None, extend_model=None, extend_kwargs=None,
                          transformed=True, includes_fixed=False, **kwargs):
        """
        Impulse response function

        Parameters
        ----------
        params : array_like
            Array of model parameters.
        steps : int, optional
            The number of steps for which impulse responses are calculated.
            Default is 1. Note that for time-invariant models, the initial
            impulse is not counted as a step, so if `steps=1`, the output will
            have 2 entries.
        impulse : int, str or array_like
            If an integer, the state innovation to pulse; must be between 0
            and `k_posdef-1`. If a str, it indicates which column of df
            the unit (1) impulse is given.
            Alternatively, a custom impulse vector may be provided; must be
            shaped `k_posdef x 1`.
        orthogonalized : bool, optional
            Whether or not to perform impulse using orthogonalized innovations.
            Note that this will also affect custum `impulse` vectors. Default
            is False.
        cumulative : bool, optional
            Whether or not to return cumulative impulse responses. Default is
            False.
        anchor : int, str, or datetime, optional
            Time point within the sample for the state innovation impulse. Type
            depends on the index of the given `endog` in the model. Two special
            cases are the strings 'start' and 'end', which refer to setting the
            impulse at the first and last points of the sample, respectively.
            Integer values can run from 0 to `nobs - 1`, or can be negative to
            apply negative indexing. Finally, if a date/time index was provided
            to the model, then this argument can be a date string to parse or a
            datetime type. Default is 'start'.
        exog : array_like, optional
            New observations of exogenous regressors for our-of-sample periods,
            if applicable.
        transformed : bool, optional
            Whether or not `params` is already transformed. Default is
            True.
        includes_fixed : bool, optional
            If parameters were previously fixed with the `fix_params` method,
            this argument describes whether or not `params` also includes
            the fixed parameters, in addition to the free parameters. Default
            is False.
        **kwargs
            If the model has time-varying design or transition matrices and the
            combination of `anchor` and `steps` implies creating impulse
            responses for the out-of-sample period, then these matrices must
            have updated values provided for the out-of-sample steps. For
            example, if `design` is a time-varying component, `nobs` is 10,
            `anchor=1`, and `steps` is 15, a (`k_endog` x `k_states` x 7)
            matrix must be provided with the new design matrix values.

        Returns
        -------
        impulse_responses : ndarray
            Responses for each endogenous variable due to the impulse
            given by the `impulse` argument. For a time-invariant model, the
            impulse responses are given for `steps + 1` elements (this gives
            the "initial impulse" followed by `steps` responses for the
            important cases of VAR and SARIMAX models), while for time-varying
            models the impulse responses are only given for `steps` elements
            (to avoid having to unexpectedly provide updated time-varying
            matrices).

        See Also
        --------
        simulate
            Simulate a time series according to the given state space model,
            optionally with specified series for the innovations.

        Notes
        -----
        Intercepts in the measurement and state equation are ignored when
        calculating impulse responses.

        TODO: add an option to allow changing the ordering for the
              orthogonalized option. Will require permuting matrices when
              constructing the extended model.
        """
        # Make sure the model class has the current parameters
        self.update(params, transformed=transformed,
                    includes_fixed=includes_fixed)

        # For time-invariant models, add an additional `step`. This is the
        # default for time-invariant models based on the expected behavior for
        # ARIMA and VAR models: we want to record the initial impulse and also
        # `steps` values of the responses afterwards.
        # Note: we don't modify `steps` itself, because
        # `KalmanFilter.impulse_responses` also adds an additional step in this
        # case (this is so that there isn't different behavior when calling
        # this method versus that method). We just need to also keep track of
        # this here because we need to generate the correct extended model.
        additional_steps = 0
        if (self.ssm._design.shape[2] == 1 and
                self.ssm._transition.shape[2] == 1 and
                self.ssm._selection.shape[2] == 1):
            additional_steps = 1

        # Get the starting location
        if anchor is None or anchor == 'start':
            iloc = 0
        elif anchor == 'end':
            iloc = self.nobs - 1
        else:
            iloc, _, _ = self._get_index_loc(anchor)
            if isinstance(iloc, slice):
                iloc = iloc.start

        if iloc < 0:
            iloc = self.nobs + iloc
        if iloc >= self.nobs:
            raise ValueError('Cannot anchor impulse responses outside of the'
                             ' sample.')

        time_invariant = (
            self.ssm._design.shape[2] == self.ssm._obs_cov.shape[2] ==
            self.ssm._transition.shape[2] == self.ssm._selection.shape[2] ==
            self.ssm._state_cov.shape[2] == 1)

        # Get updated time-varying system matrices in **kwargs, if necessary
        # (Note: KalmanFilter adds 1 to steps to account for the first impulse)
        out_of_sample = max(
            iloc + (steps + additional_steps + 1) - self.nobs, 0)
        if extend_model is None:
            extend_model = self.exog is not None and not time_invariant
        if out_of_sample and extend_model:
            kwargs = self._get_extension_time_varying_matrices(
                params, exog, out_of_sample, extend_kwargs,
                transformed=transformed, includes_fixed=includes_fixed,
                **kwargs)

        # Special handling for matrix terms that are time-varying but
        # irrelevant for impulse response functions. Must be set since
        # ssm.extend() requires that we pass new matrices for these, but they
        # are ignored for IRF purposes.
        end = min(self.nobs, iloc + steps + additional_steps)
        nextend = iloc + (steps + additional_steps + 1) - end
        if ('obs_intercept' not in kwargs and
                self.ssm._obs_intercept.shape[1] > 1):
            kwargs['obs_intercept'] = np.zeros((self.k_endog, nextend))
        if ('state_intercept' not in kwargs and
                self.ssm._state_intercept.shape[1] > 1):
            kwargs['state_intercept'] = np.zeros((self.k_states, nextend))
        if 'obs_cov' not in kwargs and self.ssm._obs_cov.shape[2] > 1:
            kwargs['obs_cov'] = np.zeros((self.k_endog, self.k_endog, nextend))
        # Special handling for matrix terms that are time-varying but
        # only the value at the anchor matters for IRF purposes.
        if 'state_cov' not in kwargs and self.ssm._state_cov.shape[2] > 1:
            tmp = np.zeros((self.ssm.k_posdef, self.ssm.k_posdef, nextend))
            tmp[:] = self['state_cov', :, :, iloc:iloc + 1]
            kwargs['state_cov'] = tmp
        if 'selection' not in kwargs and self.ssm._selection.shape[2] > 1:
            tmp = np.zeros((self.k_states, self.ssm.k_posdef, nextend))
            tmp[:] = self['selection', :, :, iloc:iloc + 1]
            kwargs['selection'] = tmp

        # Construct a model that represents the simulation period
        sim_model = self.ssm.extend(np.empty((nextend, self.k_endog)),
                                    start=iloc, end=end, **kwargs)

        # Compute the impulse responses

        # Convert endog name to index
        use_pandas = isinstance(self.data, PandasData)
        if type(impulse) == str:
            if not use_pandas:
                raise ValueError('Endog must be pd.DataFrame.')
            impulse = self.endog_names.index(impulse)

        irfs = sim_model.impulse_responses(
            steps, impulse, orthogonalized, cumulative)

        # IRF is (nobs x k_endog); do not want to squeeze in case of steps = 1
        if irfs.shape[1] == 1:
            irfs = irfs[:, 0]

        # Wrap data / squeeze where appropriate
        if use_pandas:
            if self.k_endog == 1:
                irfs = pd.Series(irfs, name=self.endog_names)
            else:
                irfs = pd.DataFrame(irfs, columns=self.endog_names)
        return irfs

    @classmethod
    def from_formula(cls, formula, data, subset=None):
        """
        Not implemented for state space models
        """
        raise NotImplementedError


class MLEResults(tsbase.TimeSeriesModelResults):
    r"""
    Class to hold results from fitting a state space model.

    Parameters
    ----------
    model : MLEModel instance
        The fitted model instance
    params : ndarray
        Fitted parameters
    filter_results : KalmanFilter instance
        The underlying state space model and Kalman filter output

    Attributes
    ----------
    model : Model instance
        A reference to the model that was fit.
    filter_results : KalmanFilter instance
        The underlying state space model and Kalman filter output
    nobs : float
        The number of observations used to fit the model.
    params : ndarray
        The parameters of the model.
    scale : float
        This is currently set to 1.0 unless the model uses concentrated
        filtering.

    See Also
    --------
    MLEModel
    statsmodels.tsa.statespace.kalman_filter.FilterResults
    statsmodels.tsa.statespace.representation.FrozenRepresentation
    """
    def __init__(self, model, params, results, cov_type=None, cov_kwds=None,
                 **kwargs):
        self.data = model.data
        scale = results.scale

        tsbase.TimeSeriesModelResults.__init__(self, model, params,
                                               normalized_cov_params=None,
                                               scale=scale)

        # Save the fixed parameters
        self._has_fixed_params = self.model._has_fixed_params
        self._fixed_params_index = self.model._fixed_params_index
        self._free_params_index = self.model._free_params_index
        # TODO: seems like maybe self.fixed_params should be the dictionary
        # itself, not just the keys?
        if self._has_fixed_params:
            self._fixed_params = self.model._fixed_params.copy()
            self.fixed_params = list(self._fixed_params.keys())
        else:
            self._fixed_params = None
            self.fixed_params = []
        self.param_names = [
            '%s (fixed)' % name if name in self.fixed_params else name
            for name in (self.data.param_names or [])]

        # Save the state space representation output
        self.filter_results = results
        if isinstance(results, SmootherResults):
            self.smoother_results = results
        else:
            self.smoother_results = None

        # Dimensions
        self.nobs = self.filter_results.nobs
        self.nobs_diffuse = self.filter_results.nobs_diffuse
        if self.nobs_diffuse > 0 and self.loglikelihood_burn > 0:
            warnings.warn('Care should be used when applying a loglikelihood'
                          ' burn to a model with exact diffuse initialization.'
                          ' Some results objects, e.g. degrees of freedom,'
                          ' expect only one of the two to be set.')
        # This only excludes explicitly burned (usually approximate diffuse)
        # periods but does not exclude exact diffuse periods. This is
        # because the loglikelihood remains valid for the initial periods in
        # the exact diffuse case (see DK, 2012, section 7.2) and so also do
        # e.g. information criteria (see DK, 2012, section 7.4) and the score
        # vector (see DK, 2012, section 7.3.3, equation 7.15).
        # However, other objects should be excluded in the diffuse periods
        # (e.g. the diffuse forecast errors, so in some cases a different
        # nobs_effective will have to be computed and used)
        self.nobs_effective = self.nobs - self.loglikelihood_burn

        P = self.filter_results.initial_diffuse_state_cov
        self.k_diffuse_states = 0 if P is None else np.sum(np.diagonal(P) == 1)

        # Degrees of freedom (see DK 2012, section 7.4)
        k_free_params = self.params.size - len(self.fixed_params)
        self.df_model = (k_free_params + self.k_diffuse_states
                         + self.filter_results.filter_concentrated)
        self.df_resid = self.nobs_effective - self.df_model

        # Setup covariance matrix notes dictionary
        if not hasattr(self, 'cov_kwds'):
            self.cov_kwds = {}
        if cov_type is None:
            cov_type = 'approx' if results.memory_no_likelihood else 'opg'
        self.cov_type = cov_type

        # Setup the cache
        self._cache = {}

        # Handle covariance matrix calculation
        if cov_kwds is None:
            cov_kwds = {}
        self._cov_approx_complex_step = (
            cov_kwds.pop('approx_complex_step', True))
        self._cov_approx_centered = cov_kwds.pop('approx_centered', False)
        try:
            self._rank = None
            self._get_robustcov_results(cov_type=cov_type, use_self=True,
                                        **cov_kwds)
        except np.linalg.LinAlgError:
            self._rank = 0
            k_params = len(self.params)
            self.cov_params_default = np.zeros((k_params, k_params)) * np.nan
            self.cov_kwds['cov_type'] = (
                'Covariance matrix could not be calculated: singular.'
                ' information matrix.')
        self.model.update(self.params, transformed=True, includes_fixed=True)

        # References of filter and smoother output
        extra_arrays = [
            'filtered_state', 'filtered_state_cov', 'predicted_state',
            'predicted_state_cov', 'forecasts', 'forecasts_error',
            'forecasts_error_cov', 'standardized_forecasts_error',
            'forecasts_error_diffuse_cov', 'predicted_diffuse_state_cov',
            'scaled_smoothed_estimator',
            'scaled_smoothed_estimator_cov', 'smoothing_error',
            'smoothed_state',
            'smoothed_state_cov', 'smoothed_state_autocov',
            'smoothed_measurement_disturbance',
            'smoothed_state_disturbance',
            'smoothed_measurement_disturbance_cov',
            'smoothed_state_disturbance_cov']
        for name in extra_arrays:
            setattr(self, name, getattr(self.filter_results, name, None))

        # Remove too-short results when memory conservation was used
        if self.filter_results.memory_no_forecast_mean:
            self.forecasts = None
            self.forecasts_error = None
        if self.filter_results.memory_no_forecast_cov:
            self.forecasts_error_cov = None
        if self.filter_results.memory_no_predicted_mean:
            self.predicted_state = None
        if self.filter_results.memory_no_predicted_cov:
            self.predicted_state_cov = None
        if self.filter_results.memory_no_filtered_mean:
            self.filtered_state = None
        if self.filter_results.memory_no_filtered_cov:
            self.filtered_state_cov = None
        if self.filter_results.memory_no_gain:
            pass
        if self.filter_results.memory_no_smoothing:
            pass
        if self.filter_results.memory_no_std_forecast:
            self.standardized_forecasts_error = None

        # Save more convenient access to states
        # (will create a private attribute _states here and provide actual
        # access via a getter, so that we can e.g. issue a warning in the case
        # that a useless Pandas index was given in the model specification)
        self._states = SimpleNamespace()

        use_pandas = isinstance(self.data, PandasData)
        index = self.model._index
        columns = self.model.state_names

        # Predicted states
        # Note: a complication here is that we also include the initial values
        # here, so that we need an extended index in the Pandas case
        if (self.predicted_state is None or
                self.filter_results.memory_no_predicted_mean):
            self._states.predicted = None
        elif use_pandas:
            extended_index = self.model._get_index_with_final_state()
            self._states.predicted = pd.DataFrame(
                self.predicted_state.T, index=extended_index, columns=columns)
        else:
            self._states.predicted = self.predicted_state.T
        if (self.predicted_state_cov is None or
                self.filter_results.memory_no_predicted_cov):
            self._states.predicted_cov = None
        elif use_pandas:
            extended_index = self.model._get_index_with_final_state()
            tmp = np.transpose(self.predicted_state_cov, (2, 0, 1))
            self._states.predicted_cov = pd.DataFrame(
                np.reshape(tmp, (tmp.shape[0] * tmp.shape[1], tmp.shape[2])),
                index=pd.MultiIndex.from_product(
                    [extended_index, columns]).swaplevel(),
                columns=columns)
        else:
            self._states.predicted_cov = np.transpose(
                self.predicted_state_cov, (2, 0, 1))

        # Filtered states
        if (self.filtered_state is None or
                self.filter_results.memory_no_filtered_mean):
            self._states.filtered = None
        elif use_pandas:
            self._states.filtered = pd.DataFrame(
                self.filtered_state.T, index=index, columns=columns)
        else:
            self._states.filtered = self.filtered_state.T
        if (self.filtered_state_cov is None or
                self.filter_results.memory_no_filtered_cov):
            self._states.filtered_cov = None
        elif use_pandas:
            tmp = np.transpose(self.filtered_state_cov, (2, 0, 1))
            self._states.filtered_cov = pd.DataFrame(
                np.reshape(tmp, (tmp.shape[0] * tmp.shape[1], tmp.shape[2])),
                index=pd.MultiIndex.from_product([index, columns]).swaplevel(),
                columns=columns)
        else:
            self._states.filtered_cov = np.transpose(
                self.filtered_state_cov, (2, 0, 1))

        # Smoothed states
        if self.smoothed_state is None:
            self._states.smoothed = None
        elif use_pandas:
            self._states.smoothed = pd.DataFrame(
                self.smoothed_state.T, index=index, columns=columns)
        else:
            self._states.smoothed = self.smoothed_state.T
        if self.smoothed_state_cov is None:
            self._states.smoothed_cov = None
        elif use_pandas:
            tmp = np.transpose(self.smoothed_state_cov, (2, 0, 1))
            self._states.smoothed_cov = pd.DataFrame(
                np.reshape(tmp, (tmp.shape[0] * tmp.shape[1], tmp.shape[2])),
                index=pd.MultiIndex.from_product([index, columns]).swaplevel(),
                columns=columns)
        else:
            self._states.smoothed_cov = np.transpose(
                self.smoothed_state_cov, (2, 0, 1))

        # Handle removing data
        self._data_attr_model = getattr(self, '_data_attr_model', [])
        self._data_attr_model.extend(['ssm'])
        self._data_attr.extend(extra_arrays)
        self._data_attr.extend(['filter_results', 'smoother_results'])

    def _get_robustcov_results(self, cov_type='opg', **kwargs):
        """
        Create new results instance with specified covariance estimator as
        default

        Note: creating new results instance currently not supported.

        Parameters
        ----------
        cov_type : str
            the type of covariance matrix estimator to use. See Notes below
        kwargs : depends on cov_type
            Required or optional arguments for covariance calculation.
            See Notes below.

        Returns
        -------
        results : results instance
            This method creates a new results instance with the requested
            covariance as the default covariance of the parameters.
            Inferential statistics like p-values and hypothesis tests will be
            based on this covariance matrix.

        Notes
        -----
        The following covariance types and required or optional arguments are
        currently available:

        - 'opg' for the outer product of gradient estimator
        - 'oim' for the observed information matrix estimator, calculated
          using the method of Harvey (1989)
        - 'approx' for the observed information matrix estimator,
          calculated using a numerical approximation of the Hessian matrix.
          Uses complex step approximation by default, or uses finite
          differences if `approx_complex_step=False` in the `cov_kwds`
          dictionary.
        - 'robust' for an approximate (quasi-maximum likelihood) covariance
          matrix that may be valid even in the presence of some
          misspecifications. Intermediate calculations use the 'oim'
          method.
        - 'robust_approx' is the same as 'robust' except that the
          intermediate calculations use the 'approx' method.
        - 'none' for no covariance matrix calculation.
        """
        from statsmodels.base.covtype import descriptions

        use_self = kwargs.pop('use_self', False)
        if use_self:
            res = self
        else:
            raise NotImplementedError
            res = self.__class__(
                self.model, self.params,
                normalized_cov_params=self.normalized_cov_params,
                scale=self.scale)

        # Set the new covariance type
        res.cov_type = cov_type
        res.cov_kwds = {}

        # Calculate the new covariance matrix
        approx_complex_step = self._cov_approx_complex_step
        if approx_complex_step:
            approx_type_str = 'complex-step'
        elif self._cov_approx_centered:
            approx_type_str = 'centered finite differences'
        else:
            approx_type_str = 'finite differences'

        k_params = len(self.params)
        if k_params == 0:
            res.cov_params_default = np.zeros((0, 0))
            res._rank = 0
            res.cov_kwds['description'] = 'No parameters estimated.'
        elif cov_type == 'custom':
            res.cov_type = kwargs['custom_cov_type']
            res.cov_params_default = kwargs['custom_cov_params']
            res.cov_kwds['description'] = kwargs['custom_description']
            if len(self.fixed_params) > 0:
                mask = np.ix_(self._free_params_index, self._free_params_index)
            else:
                mask = np.s_[...]
            res._rank = np.linalg.matrix_rank(res.cov_params_default[mask])
        elif cov_type == 'none':
            res.cov_params_default = np.zeros((k_params, k_params)) * np.nan
            res._rank = np.nan
            res.cov_kwds['description'] = descriptions['none']
        elif self.cov_type == 'approx':
            res.cov_params_default = res.cov_params_approx
            res.cov_kwds['description'] = descriptions['approx'].format(
                                                approx_type=approx_type_str)
        elif self.cov_type == 'oim':
            res.cov_params_default = res.cov_params_oim
            res.cov_kwds['description'] = descriptions['OIM'].format(
                                                approx_type=approx_type_str)
        elif self.cov_type == 'opg':
            res.cov_params_default = res.cov_params_opg
            res.cov_kwds['description'] = descriptions['OPG'].format(
                                                approx_type=approx_type_str)
        elif self.cov_type == 'robust' or self.cov_type == 'robust_oim':
            res.cov_params_default = res.cov_params_robust_oim
            res.cov_kwds['description'] = descriptions['robust-OIM'].format(
                                                approx_type=approx_type_str)
        elif self.cov_type == 'robust_approx':
            res.cov_params_default = res.cov_params_robust_approx
            res.cov_kwds['description'] = descriptions['robust-approx'].format(
                                                approx_type=approx_type_str)
        else:
            raise NotImplementedError('Invalid covariance matrix type.')

        return res

    @cache_readonly
    def aic(self):
        """
        (float) Akaike Information Criterion
        """
        return aic(self.llf, self.nobs_effective, self.df_model)

    @cache_readonly
    def aicc(self):
        """
        (float) Akaike Information Criterion with small sample correction
        """
        return aicc(self.llf, self.nobs_effective, self.df_model)

    @cache_readonly
    def bic(self):
        """
        (float) Bayes Information Criterion
        """
        return bic(self.llf, self.nobs_effective, self.df_model)

    def _cov_params_approx(self, approx_complex_step=True,
                           approx_centered=False):
        evaluated_hessian = self.nobs_effective * self.model.hessian(
            params=self.params, transformed=True, includes_fixed=True,
            method='approx', approx_complex_step=approx_complex_step,
            approx_centered=approx_centered)
        # TODO: Case with "not approx_complex_step" is not hit in
        # tests as of 2017-05-19

        if len(self.fixed_params) > 0:
            mask = np.ix_(self._free_params_index, self._free_params_index)
            (tmp, singular_values) = pinv_extended(evaluated_hessian[mask])
            neg_cov = np.zeros_like(evaluated_hessian) * np.nan
            neg_cov[mask] = tmp
        else:
            (neg_cov, singular_values) = pinv_extended(evaluated_hessian)

        self.model.update(self.params, transformed=True, includes_fixed=True)
        if self._rank is None:
            self._rank = np.linalg.matrix_rank(np.diag(singular_values))
        return -neg_cov

    @cache_readonly
    def cov_params_approx(self):
        """
        (array) The variance / covariance matrix. Computed using the numerical
        Hessian approximated by complex step or finite differences methods.
        """
        return self._cov_params_approx(self._cov_approx_complex_step,
                                       self._cov_approx_centered)

    def _cov_params_oim(self, approx_complex_step=True, approx_centered=False):
        evaluated_hessian = self.nobs_effective * self.model.hessian(
            self.params, hessian_method='oim', transformed=True,
            includes_fixed=True, approx_complex_step=approx_complex_step,
            approx_centered=approx_centered)

        if len(self.fixed_params) > 0:
            mask = np.ix_(self._free_params_index, self._free_params_index)
            (tmp, singular_values) = pinv_extended(evaluated_hessian[mask])
            neg_cov = np.zeros_like(evaluated_hessian) * np.nan
            neg_cov[mask] = tmp
        else:
            (neg_cov, singular_values) = pinv_extended(evaluated_hessian)

        self.model.update(self.params, transformed=True, includes_fixed=True)
        if self._rank is None:
            self._rank = np.linalg.matrix_rank(np.diag(singular_values))
        return -neg_cov

    @cache_readonly
    def cov_params_oim(self):
        """
        (array) The variance / covariance matrix. Computed using the method
        from Harvey (1989).
        """
        return self._cov_params_oim(self._cov_approx_complex_step,
                                    self._cov_approx_centered)

    def _cov_params_opg(self, approx_complex_step=True, approx_centered=False):
        evaluated_hessian = self.nobs_effective * self.model._hessian_opg(
            self.params, transformed=True, includes_fixed=True,
            approx_complex_step=approx_complex_step,
            approx_centered=approx_centered)

        no_free_params = (self._free_params_index is not None and
                          len(self._free_params_index) == 0)

        if no_free_params:
            neg_cov = np.zeros_like(evaluated_hessian) * np.nan
            singular_values = np.empty(0)
        elif len(self.fixed_params) > 0:
            mask = np.ix_(self._free_params_index, self._free_params_index)
            (tmp, singular_values) = pinv_extended(evaluated_hessian[mask])
            neg_cov = np.zeros_like(evaluated_hessian) * np.nan
            neg_cov[mask] = tmp
        else:
            (neg_cov, singular_values) = pinv_extended(evaluated_hessian)

        self.model.update(self.params, transformed=True, includes_fixed=True)
        if self._rank is None:
            if no_free_params:
                self._rank = 0
            else:
                self._rank = np.linalg.matrix_rank(np.diag(singular_values))
        return -neg_cov

    @cache_readonly
    def cov_params_opg(self):
        """
        (array) The variance / covariance matrix. Computed using the outer
        product of gradients method.
        """
        return self._cov_params_opg(self._cov_approx_complex_step,
                                    self._cov_approx_centered)

    @cache_readonly
    def cov_params_robust(self):
        """
        (array) The QMLE variance / covariance matrix. Alias for
        `cov_params_robust_oim`
        """
        return self.cov_params_robust_oim

    def _cov_params_robust_oim(self, approx_complex_step=True,
                               approx_centered=False):
        cov_opg = self._cov_params_opg(approx_complex_step=approx_complex_step,
                                       approx_centered=approx_centered)

        evaluated_hessian = self.nobs_effective * self.model.hessian(
            self.params, hessian_method='oim', transformed=True,
            includes_fixed=True, approx_complex_step=approx_complex_step,
            approx_centered=approx_centered)

        if len(self.fixed_params) > 0:
            mask = np.ix_(self._free_params_index, self._free_params_index)
            cov_params = np.zeros_like(evaluated_hessian) * np.nan

            cov_opg = cov_opg[mask]
            evaluated_hessian = evaluated_hessian[mask]

            tmp, singular_values = pinv_extended(
                np.dot(np.dot(evaluated_hessian, cov_opg), evaluated_hessian))

            cov_params[mask] = tmp
        else:
            (cov_params, singular_values) = pinv_extended(
                np.dot(np.dot(evaluated_hessian, cov_opg), evaluated_hessian))

        self.model.update(self.params, transformed=True, includes_fixed=True)
        if self._rank is None:
            self._rank = np.linalg.matrix_rank(np.diag(singular_values))
        return cov_params

    @cache_readonly
    def cov_params_robust_oim(self):
        """
        (array) The QMLE variance / covariance matrix. Computed using the
        method from Harvey (1989) as the evaluated hessian.
        """
        return self._cov_params_robust_oim(self._cov_approx_complex_step,
                                           self._cov_approx_centered)

    def _cov_params_robust_approx(self, approx_complex_step=True,
                                  approx_centered=False):
        cov_opg = self._cov_params_opg(approx_complex_step=approx_complex_step,
                                       approx_centered=approx_centered)

        evaluated_hessian = self.nobs_effective * self.model.hessian(
            self.params, transformed=True, includes_fixed=True,
            method='approx', approx_complex_step=approx_complex_step)
        # TODO: Case with "not approx_complex_step" is not
        # hit in tests as of 2017-05-19

        if len(self.fixed_params) > 0:
            mask = np.ix_(self._free_params_index, self._free_params_index)
            cov_params = np.zeros_like(evaluated_hessian) * np.nan

            cov_opg = cov_opg[mask]
            evaluated_hessian = evaluated_hessian[mask]

            tmp, singular_values = pinv_extended(
                np.dot(np.dot(evaluated_hessian, cov_opg), evaluated_hessian))

            cov_params[mask] = tmp
        else:
            (cov_params, singular_values) = pinv_extended(
                np.dot(np.dot(evaluated_hessian, cov_opg), evaluated_hessian))

        self.model.update(self.params, transformed=True, includes_fixed=True)
        if self._rank is None:
            self._rank = np.linalg.matrix_rank(np.diag(singular_values))
        return cov_params

    @cache_readonly
    def cov_params_robust_approx(self):
        """
        (array) The QMLE variance / covariance matrix. Computed using the
        numerical Hessian as the evaluated hessian.
        """
        return self._cov_params_robust_approx(self._cov_approx_complex_step,
                                              self._cov_approx_centered)

    def info_criteria(self, criteria, method='standard'):
        r"""
        Information criteria

        Parameters
        ----------
        criteria : {'aic', 'bic', 'hqic'}
            The information criteria to compute.
        method : {'standard', 'lutkepohl'}
            The method for information criteria computation. Default is
            'standard' method; 'lutkepohl' computes the information criteria
            as in Lütkepohl (2007). See Notes for formulas.

        Notes
        -----
        The `'standard'` formulas are:

        .. math::

            AIC & = -2 \log L(Y_n | \hat \psi) + 2 k \\
            BIC & = -2 \log L(Y_n | \hat \psi) + k \log n \\
            HQIC & = -2 \log L(Y_n | \hat \psi) + 2 k \log \log n \\

        where :math:`\hat \psi` are the maximum likelihood estimates of the
        parameters, :math:`n` is the number of observations, and `k` is the
        number of estimated parameters.

        Note that the `'standard'` formulas are returned from the `aic`, `bic`,
        and `hqic` results attributes.

        The `'lutkepohl'` formulas are (Lütkepohl, 2010):

        .. math::

            AIC_L & = \log | Q | + \frac{2 k}{n} \\
            BIC_L & = \log | Q | + \frac{k \log n}{n} \\
            HQIC_L & = \log | Q | + \frac{2 k \log \log n}{n} \\

        where :math:`Q` is the state covariance matrix. Note that the Lütkepohl
        definitions do not apply to all state space models, and should be used
        with care outside of SARIMAX and VARMAX models.

        References
        ----------
        .. [*] Lütkepohl, Helmut. 2007. *New Introduction to Multiple Time*
           *Series Analysis.* Berlin: Springer.
        """
        criteria = criteria.lower()
        method = method.lower()

        if method == 'standard':
            out = getattr(self, criteria)
        elif method == 'lutkepohl':
            if self.filter_results.state_cov.shape[-1] > 1:
                raise ValueError('Cannot compute Lütkepohl statistics for'
                                 ' models with time-varying state covariance'
                                 ' matrix.')

            cov = self.filter_results.state_cov[:, :, 0]
            if criteria == 'aic':
                out = np.squeeze(np.linalg.slogdet(cov)[1] +
                                 2 * self.df_model / self.nobs_effective)
            elif criteria == 'bic':
                out = np.squeeze(np.linalg.slogdet(cov)[1] +
                                 self.df_model * np.log(self.nobs_effective) /
                                 self.nobs_effective)
            elif criteria == 'hqic':
                out = np.squeeze(np.linalg.slogdet(cov)[1] +
                                 2 * self.df_model *
                                 np.log(np.log(self.nobs_effective)) /
                                 self.nobs_effective)
            else:
                raise ValueError('Invalid information criteria')

        else:
            raise ValueError('Invalid information criteria computation method')

        return out

    @cache_readonly
    def fittedvalues(self):
        """
        (array) The predicted values of the model. An (nobs x k_endog) array.
        """
        # This is a (k_endog x nobs array; do not want to squeeze in case of
        # the corner case where nobs = 1 (mostly a concern in the predict or
        # forecast functions, but here also to maintain consistency)
        fittedvalues = self.forecasts
        if fittedvalues is None:
            pass
        elif fittedvalues.shape[0] == 1:
            fittedvalues = fittedvalues[0, :]
        else:
            fittedvalues = fittedvalues.T
        return fittedvalues

    @cache_readonly
    def hqic(self):
        """
        (float) Hannan-Quinn Information Criterion
        """
        # return (-2 * self.llf +
        #         2 * np.log(np.log(self.nobs_effective)) * self.df_model)
        return hqic(self.llf, self.nobs_effective, self.df_model)

    @cache_readonly
    def llf_obs(self):
        """
        (float) The value of the log-likelihood function evaluated at `params`.
        """
        return self.filter_results.llf_obs

    @cache_readonly
    def llf(self):
        """
        (float) The value of the log-likelihood function evaluated at `params`.
        """
        return self.filter_results.llf

    @cache_readonly
    def loglikelihood_burn(self):
        """
        (float) The number of observations during which the likelihood is not
        evaluated.
        """
        return self.filter_results.loglikelihood_burn

    @cache_readonly
    def mae(self):
        """
        (float) Mean absolute error
        """
        return np.mean(np.abs(self.resid))

    @cache_readonly
    def mse(self):
        """
        (float) Mean squared error
        """
        return self.sse / self.nobs

    @cache_readonly
    def pvalues(self):
        """
        (array) The p-values associated with the z-statistics of the
        coefficients. Note that the coefficients are assumed to have a Normal
        distribution.
        """
        pvalues = np.zeros_like(self.zvalues) * np.nan
        mask = np.ones_like(pvalues, dtype=bool)
        mask[self._free_params_index] = True
        mask &= ~np.isnan(self.zvalues)
        pvalues[mask] = norm.sf(np.abs(self.zvalues[mask])) * 2
        return pvalues

    @cache_readonly
    def resid(self):
        """
        (array) The model residuals. An (nobs x k_endog) array.
        """
        # This is a (k_endog x nobs array; do not want to squeeze in case of
        # the corner case where nobs = 1 (mostly a concern in the predict or
        # forecast functions, but here also to maintain consistency)
        resid = self.forecasts_error
        if resid is None:
            pass
        elif resid.shape[0] == 1:
            resid = resid[0, :]
        else:
            resid = resid.T
        return resid

    @property
    def states(self):
        if self.model._index_generated and not self.model._index_none:
            warnings.warn('No supported index is available. The `states`'
                          ' DataFrame uses a generated integer index',
                          ValueWarning)
        return self._states

    @cache_readonly
    def sse(self):
        """
        (float) Sum of squared errors
        """
        return np.sum(self.resid**2)

    @cache_readonly
    def zvalues(self):
        """
        (array) The z-statistics for the coefficients.
        """
        return self.params / self.bse

    def test_normality(self, method):
        """
        Test for normality of standardized residuals.

        Null hypothesis is normality.

        Parameters
        ----------
        method : {'jarquebera', None}
            The statistical test for normality. Must be 'jarquebera' for
            Jarque-Bera normality test. If None, an attempt is made to select
            an appropriate test.

        See Also
        --------
        statsmodels.stats.stattools.jarque_bera
            The Jarque-Bera test of normality.

        Notes
        -----
        Let `d` = max(loglikelihood_burn, nobs_diffuse); this test is
        calculated ignoring the first `d` residuals.

        In the case of missing data, the maintained hypothesis is that the
        data are missing completely at random. This test is then run on the
        standardized residuals excluding those corresponding to missing
        observations.
        """
        if method is None:
            method = 'jarquebera'

        if self.standardized_forecasts_error is None:
            raise ValueError('Cannot compute test statistic when standardized'
                             ' forecast errors have not been computed.')

        if method == 'jarquebera':
            from statsmodels.stats.stattools import jarque_bera
            d = np.maximum(self.loglikelihood_burn, self.nobs_diffuse)
            output = []
            for i in range(self.model.k_endog):
                resid = self.filter_results.standardized_forecasts_error[i, d:]
                mask = ~np.isnan(resid)
                output.append(jarque_bera(resid[mask]))
        else:
            raise NotImplementedError('Invalid normality test method.')

        return np.array(output)

    def test_heteroskedasticity(self, method, alternative='two-sided',
                                use_f=True):
        r"""
        Test for heteroskedasticity of standardized residuals

        Tests whether the sum-of-squares in the first third of the sample is
        significantly different than the sum-of-squares in the last third
        of the sample. Analogous to a Goldfeld-Quandt test. The null hypothesis
        is of no heteroskedasticity.

        Parameters
        ----------
        method : {'breakvar', None}
            The statistical test for heteroskedasticity. Must be 'breakvar'
            for test of a break in the variance. If None, an attempt is
            made to select an appropriate test.
        alternative : str, 'increasing', 'decreasing' or 'two-sided'
            This specifies the alternative for the p-value calculation. Default
            is two-sided.
        use_f : bool, optional
            Whether or not to compare against the asymptotic distribution
            (chi-squared) or the approximate small-sample distribution (F).
            Default is True (i.e. default is to compare against an F
            distribution).

        Returns
        -------
        output : ndarray
            An array with `(test_statistic, pvalue)` for each endogenous
            variable. The array is then sized `(k_endog, 2)`. If the method is
            called as `het = res.test_heteroskedasticity()`, then `het[0]` is
            an array of size 2 corresponding to the first endogenous variable,
            where `het[0][0]` is the test statistic, and `het[0][1]` is the
            p-value.

        See Also
        --------
        statsmodels.tsa.stattools.breakvar_heteroskedasticity_test

        Notes
        -----
        The null hypothesis is of no heteroskedasticity.

        For :math:`h = [T/3]`, the test statistic is:

        .. math::

            H(h) = \sum_{t=T-h+1}^T  \tilde v_t^2
            \Bigg / \sum_{t=d+1}^{d+1+h} \tilde v_t^2

        where :math:`d` = max(loglikelihood_burn, nobs_diffuse)` (usually
        corresponding to diffuse initialization under either the approximate
        or exact approach).

        This statistic can be tested against an :math:`F(h,h)` distribution.
        Alternatively, :math:`h H(h)` is asymptotically distributed according
        to :math:`\chi_h^2`; this second test can be applied by passing
        `use_f=True` as an argument.

        See section 5.4 of [1]_ for the above formula and discussion, as well
        as additional details.

        TODO

        - Allow specification of :math:`h`

        References
        ----------
        .. [1] Harvey, Andrew C. 1990. *Forecasting, Structural Time Series*
               *Models and the Kalman Filter.* Cambridge University Press.
        """
        if method is None:
            method = 'breakvar'

        if self.standardized_forecasts_error is None:
            raise ValueError('Cannot compute test statistic when standardized'
                             ' forecast errors have not been computed.')

        if method == 'breakvar':
            from statsmodels.tsa.stattools import (
                breakvar_heteroskedasticity_test
                )
            # Store some values
            resid = self.filter_results.standardized_forecasts_error
            d = np.maximum(self.loglikelihood_burn, self.nobs_diffuse)
            # This differs from self.nobs_effective because here we want to
            # exclude exact diffuse periods, whereas self.nobs_effective only
            # excludes explicitly burned (usually approximate diffuse) periods.
            nobs_effective = self.nobs - d
            h = int(np.round(nobs_effective / 3))

            test_statistics = []
            p_values = []
            for i in range(self.model.k_endog):
                test_statistic, p_value = breakvar_heteroskedasticity_test(
                    resid[i, d:],
                    subset_length=h,
                    alternative=alternative,
                    use_f=use_f
                    )
                test_statistics.append(test_statistic)
                p_values.append(p_value)

            output = np.c_[test_statistics, p_values]
        else:
            raise NotImplementedError('Invalid heteroskedasticity test'
                                      ' method.')

        return output

    def test_serial_correlation(self, method, df_adjust=False, lags=None):
        """
        Ljung-Box test for no serial correlation of standardized residuals

        Null hypothesis is no serial correlation.

        Parameters
        ----------
        method : {'ljungbox','boxpierece', None}
            The statistical test for serial correlation. If None, an attempt is
            made to select an appropriate test.
        lags : None, int or array_like
            If lags is an integer then this is taken to be the largest lag
            that is included, the test result is reported for all smaller lag
            length.
            If lags is a list or array, then all lags are included up to the
            largest lag in the list, however only the tests for the lags in the
            list are reported.
            If lags is None, then the default maxlag is min(10, nobs // 5) for
            non-seasonal models and min(2*m, nobs // 5) for seasonal time
            series where m is the seasonal period.
        df_adjust : bool, optional
            If True, the degrees of freedom consumed by the model is subtracted
            from the degrees-of-freedom used in the test so that the adjusted
            dof for the statistics are lags - model_df. In an ARMA model, this
            value is usually p+q where p is the AR order and q is the MA order.
            When using df_adjust, it is not possible to use tests based on
            fewer than model_df lags.
        Returns
        -------
        output : ndarray
            An array with `(test_statistic, pvalue)` for each endogenous
            variable and each lag. The array is then sized
            `(k_endog, 2, lags)`. If the method is called as
            `ljungbox = res.test_serial_correlation()`, then `ljungbox[i]`
            holds the results of the Ljung-Box test (as would be returned by
            `statsmodels.stats.diagnostic.acorr_ljungbox`) for the `i` th
            endogenous variable.

        See Also
        --------
        statsmodels.stats.diagnostic.acorr_ljungbox
            Ljung-Box test for serial correlation.

        Notes
        -----
        Let `d` = max(loglikelihood_burn, nobs_diffuse); this test is
        calculated ignoring the first `d` residuals.

        Output is nan for any endogenous variable which has missing values.
        """
        if method is None:
            method = 'ljungbox'

        if self.standardized_forecasts_error is None:
            raise ValueError('Cannot compute test statistic when standardized'
                             ' forecast errors have not been computed.')

        if method == 'ljungbox' or method == 'boxpierce':
            from statsmodels.stats.diagnostic import acorr_ljungbox
            d = np.maximum(self.loglikelihood_burn, self.nobs_diffuse)
            # This differs from self.nobs_effective because here we want to
            # exclude exact diffuse periods, whereas self.nobs_effective only
            # excludes explicitly burned (usually approximate diffuse) periods.
            nobs_effective = self.nobs - d
            output = []

            # Default lags for acorr_ljungbox is 40, but may not always have
            # that many observations
            if lags is None:
                seasonal_periods = getattr(self.model, "seasonal_periods", 0)
                if seasonal_periods:
                    lags = min(2 * seasonal_periods, nobs_effective // 5)
                else:
                    lags = min(10, nobs_effective // 5)

            model_df = 0
            if df_adjust:
                model_df = max(0, self.df_model - self.k_diffuse_states - 1)

            cols = [2, 3] if method == 'boxpierce' else [0, 1]
            for i in range(self.model.k_endog):
                results = acorr_ljungbox(
                    self.filter_results.standardized_forecasts_error[i][d:],
                    lags=lags, boxpierce=(method == 'boxpierce'),
                    model_df=model_df)
                output.append(np.asarray(results)[:, cols].T)

            output = np.c_[output]
        else:
            raise NotImplementedError('Invalid serial correlation test'
                                      ' method.')
        return output

    def get_prediction(self, start=None, end=None, dynamic=False,
                       information_set='predicted', signal_only=False,
                       index=None, exog=None, extend_model=None,
                       extend_kwargs=None, **kwargs):
        r"""
        In-sample prediction and out-of-sample forecasting

        Parameters
        ----------
        start : int, str, or datetime, optional
            Zero-indexed observation number at which to start forecasting,
            i.e., the first forecast is start. Can also be a date string to
            parse or a datetime type. Default is the the zeroth observation.
        end : int, str, or datetime, optional
            Zero-indexed observation number at which to end forecasting, i.e.,
            the last forecast is end. Can also be a date string to
            parse or a datetime type. However, if the dates index does not
            have a fixed frequency, end must be an integer index if you
            want out of sample prediction. Default is the last observation in
            the sample.
        dynamic : bool, int, str, or datetime, optional
            Integer offset relative to `start` at which to begin dynamic
            prediction. Can also be an absolute date string to parse or a
            datetime type (these are not interpreted as offsets).
            Prior to this observation, true endogenous values will be used for
            prediction; starting with this observation and continuing through
            the end of prediction, forecasted endogenous values will be used
            instead.
        information_set : str, optional
            The information set to condition each prediction on. Default is
            "predicted", which computes predictions of period t values
            conditional on observed data through period t-1; these are
            one-step-ahead predictions, and correspond with the typical
            `fittedvalues` results attribute. Alternatives are "filtered",
            which computes predictions of period t values conditional on
            observed data through period t, and "smoothed", which computes
            predictions of period t values conditional on the entire dataset
            (including also future observations t+1, t+2, ...).
        signal_only : bool, optional
            Whether to compute predictions of only the "signal" component of
            the observation equation. Default is False. For example, the
            observation equation of a time-invariant model is
            :math:`y_t = d + Z \alpha_t + \varepsilon_t`, and the "signal"
            component is then :math:`Z \alpha_t`. If this argument is set to
            True, then predictions of the "signal" :math:`Z \alpha_t` will be
            returned. Otherwise, the default is for predictions of :math:`y_t`
            to be returned.
        **kwargs
            Additional arguments may required for forecasting beyond the end
            of the sample. See `FilterResults.predict` for more details.

        Returns
        -------
        predictions : PredictionResults
            PredictionResults instance containing in-sample predictions /
            out-of-sample forecasts and results including confidence intervals.

        See Also
        --------
        forecast
            Out-of-sample forecasts.
        predict
            In-sample predictions and out-of-sample forecasts.
        get_forecast
            Out-of-sample forecasts and results including confidence intervals.
        """
        if start is None:
            start = 0

        # Handle start, end, dynamic
        start, end, out_of_sample, prediction_index = (
            self.model._get_prediction_index(start, end, index))

        # Handle `dynamic`
        if isinstance(dynamic, (str, dt.datetime, pd.Timestamp)):
            dynamic, _, _ = self.model._get_index_loc(dynamic)
            # Convert to offset relative to start
            dynamic = dynamic - start

        # If we have out-of-sample forecasting and `exog` or in general any
        # kind of time-varying state space model, then we need to create an
        # extended model to get updated state space system matrices
        if extend_model is None:
            extend_model = (self.model.exog is not None or
                            not self.filter_results.time_invariant)
        if out_of_sample and extend_model:
            kwargs = self.model._get_extension_time_varying_matrices(
                self.params, exog, out_of_sample, extend_kwargs,
                transformed=True, includes_fixed=True, **kwargs)

        # Make sure the model class has the current parameters
        self.model.update(self.params, transformed=True, includes_fixed=True)

        # Perform the prediction
        # This is a (k_endog x npredictions) array; do not want to squeeze in
        # case of npredictions = 1
        prediction_results = self.filter_results.predict(
            start, end + out_of_sample + 1, dynamic, **kwargs)

        # Return a new mlemodel.PredictionResults object
        return PredictionResultsWrapper(PredictionResults(
            self, prediction_results, information_set=information_set,
            signal_only=signal_only, row_labels=prediction_index))

    def get_forecast(self, steps=1, signal_only=False, **kwargs):
        r"""
        Out-of-sample forecasts and prediction intervals

        Parameters
        ----------
        steps : int, str, or datetime, optional
            If an integer, the number of steps to forecast from the end of the
            sample. Can also be a date string to parse or a datetime type.
            However, if the dates index does not have a fixed frequency, steps
            must be an integer. Default is 1.
        signal_only : bool, optional
            Whether to compute forecasts of only the "signal" component of
            the observation equation. Default is False. For example, the
            observation equation of a time-invariant model is
            :math:`y_t = d + Z \alpha_t + \varepsilon_t`, and the "signal"
            component is then :math:`Z \alpha_t`. If this argument is set to
            True, then forecasts of the "signal" :math:`Z \alpha_t` will be
            returned. Otherwise, the default is for forecasts of :math:`y_t`
            to be returned.
        **kwargs
            Additional arguments may required for forecasting beyond the end
            of the sample. See `FilterResults.predict` for more details.

        Returns
        -------
        forecasts : PredictionResults
            PredictionResults instance containing out-of-sample forecasts and
            results including confidence intervals.

        See also
        --------
        forecast
            Out-of-sample forecasts.
        predict
            In-sample predictions and out-of-sample forecasts.
        get_prediction
            In-sample predictions / out-of-sample forecasts and results
            including confidence intervals.
        """
        if isinstance(steps, int):
            end = self.nobs + steps - 1
        else:
            end = steps
        return self.get_prediction(start=self.nobs, end=end,
                                   signal_only=signal_only, **kwargs)

    def predict(self, start=None, end=None, dynamic=False,
                information_set='predicted', signal_only=False, **kwargs):
        r"""
        In-sample prediction and out-of-sample forecasting

        Parameters
        ----------
        start : {int, str,datetime}, optional
            Zero-indexed observation number at which to start forecasting,
            i.e., the first forecast is start. Can also be a date string to
            parse or a datetime type. Default is the zeroth observation.
        end : {int, str,datetime}, optional
            Zero-indexed observation number at which to end forecasting, i.e.,
            the last forecast is end. Can also be a date string to
            parse or a datetime type. However, if the dates index does not
            have a fixed frequency, end must be an integer index if you
            want out of sample prediction. Default is the last observation in
            the sample.
        dynamic : {bool, int, str,datetime}, optional
            Integer offset relative to `start` at which to begin dynamic
            prediction. Can also be an absolute date string to parse or a
            datetime type (these are not interpreted as offsets).
            Prior to this observation, true endogenous values will be used for
            prediction; starting with this observation and continuing through
            the end of prediction, forecasted endogenous values will be used
            instead.
        information_set : str, optional
            The information set to condition each prediction on. Default is
            "predicted", which computes predictions of period t values
            conditional on observed data through period t-1; these are
            one-step-ahead predictions, and correspond with the typical
            `fittedvalues` results attribute. Alternatives are "filtered",
            which computes predictions of period t values conditional on
            observed data through period t, and "smoothed", which computes
            predictions of period t values conditional on the entire dataset
            (including also future observations t+1, t+2, ...).
        signal_only : bool, optional
            Whether to compute predictions of only the "signal" component of
            the observation equation. Default is False. For example, the
            observation equation of a time-invariant model is
            :math:`y_t = d + Z \alpha_t + \varepsilon_t`, and the "signal"
            component is then :math:`Z \alpha_t`. If this argument is set to
            True, then predictions of the "signal" :math:`Z \alpha_t` will be
            returned. Otherwise, the default is for predictions of :math:`y_t`
            to be returned.
        **kwargs
            Additional arguments may be required for forecasting beyond the end
            of the sample. See ``FilterResults.predict`` for more details.

        Returns
        -------
        predictions : array_like
            In-sample predictions / Out-of-sample forecasts. (Numpy array or
            Pandas Series or DataFrame, depending on input and dimensions).
            Dimensions are `(npredict x k_endog)`.

        See Also
        --------
        forecast
            Out-of-sample forecasts.
        get_forecast
            Out-of-sample forecasts and results including confidence intervals.
        get_prediction
            In-sample predictions / out-of-sample forecasts and results
            including confidence intervals.
        """
        # Perform the prediction
        prediction_results = self.get_prediction(
            start, end, dynamic, information_set=information_set,
            signal_only=signal_only, **kwargs)
        return prediction_results.predicted_mean

    def forecast(self, steps=1, signal_only=False, **kwargs):
        r"""
        Out-of-sample forecasts

        Parameters
        ----------
        steps : int, str, or datetime, optional
            If an integer, the number of steps to forecast from the end of the
            sample. Can also be a date string to parse or a datetime type.
            However, if the dates index does not have a fixed frequency, steps
            must be an integer. Default is 1.
        signal_only : bool, optional
            Whether to compute forecasts of only the "signal" component of
            the observation equation. Default is False. For example, the
            observation equation of a time-invariant model is
            :math:`y_t = d + Z \alpha_t + \varepsilon_t`, and the "signal"
            component is then :math:`Z \alpha_t`. If this argument is set to
            True, then forecasts of the "signal" :math:`Z \alpha_t` will be
            returned. Otherwise, the default is for forecasts of :math:`y_t`
            to be returned.
        **kwargs
            Additional arguments may required for forecasting beyond the end
            of the sample. See `FilterResults.predict` for more details.

        Returns
        -------
        forecast : array_like
            Out-of-sample forecasts (Numpy array or Pandas Series or DataFrame,
            depending on input and dimensions).
            Dimensions are `(steps x k_endog)`.

        See Also
        --------
        predict
            In-sample predictions and out-of-sample forecasts.
        get_forecast
            Out-of-sample forecasts and results including confidence intervals.
        get_prediction
            In-sample predictions / out-of-sample forecasts and results
            including confidence intervals.
        """
        if isinstance(steps, int):
            end = self.nobs + steps - 1
        else:
            end = steps
        return self.predict(start=self.nobs, end=end, signal_only=signal_only,
                            **kwargs)

    def simulate(self, nsimulations, measurement_shocks=None,
                 state_shocks=None, initial_state=None, anchor=None,
                 repetitions=None, exog=None, extend_model=None,
                 extend_kwargs=None,
                 pretransformed_measurement_shocks=True,
                 pretransformed_state_shocks=True,
                 pretransformed_initial_state=True,
                 random_state=None, **kwargs):
        r"""
        Simulate a new time series following the state space model

        Parameters
        ----------
        nsimulations : int
            The number of observations to simulate. If the model is
            time-invariant this can be any number. If the model is
            time-varying, then this number must be less than or equal to the
            number
        measurement_shocks : array_like, optional
            If specified, these are the shocks to the measurement equation,
            :math:`\varepsilon_t`. If unspecified, these are automatically
            generated using a pseudo-random number generator. If specified,
            must be shaped `nsimulations` x `k_endog`, where `k_endog` is the
            same as in the state space model.
        state_shocks : array_like, optional
            If specified, these are the shocks to the state equation,
            :math:`\eta_t`. If unspecified, these are automatically
            generated using a pseudo-random number generator. If specified,
            must be shaped `nsimulations` x `k_posdef` where `k_posdef` is the
            same as in the state space model.
        initial_state : array_like, optional
            If specified, this is the initial state vector to use in
            simulation, which should be shaped (`k_states` x 1), where
            `k_states` is the same as in the state space model. If unspecified,
            but the model has been initialized, then that initialization is
            used. This must be specified if `anchor` is anything other than
            "start" or 0.
        anchor : int, str, or datetime, optional
            Starting point from which to begin the simulations; type depends on
            the index of the given `endog` model. Two special cases are the
            strings 'start' and 'end', which refer to starting at the beginning
            and end of the sample, respectively. If a date/time index was
            provided to the model, then this argument can be a date string to
            parse or a datetime type. Otherwise, an integer index should be
            given. Default is 'start'.
        repetitions : int, optional
            Number of simulated paths to generate. Default is 1 simulated path.
        exog : array_like, optional
            New observations of exogenous regressors, if applicable.
        pretransformed_measurement_shocks : bool, optional
            If `measurement_shocks` is provided, this flag indicates whether it
            should be directly used as the shocks. If False, then it is assumed
            to contain draws from the standard Normal distribution that must be
            transformed using the `obs_cov` covariance matrix. Default is True.
        pretransformed_state_shocks : bool, optional
            If `state_shocks` is provided, this flag indicates whether it
            should be directly used as the shocks. If False, then it is assumed
            to contain draws from the standard Normal distribution that must be
            transformed using the `state_cov` covariance matrix. Default is
            True.
        pretransformed_initial_state : bool, optional
            If `initial_state` is provided, this flag indicates whether it
            should be directly used as the initial_state. If False, then it is
            assumed to contain draws from the standard Normal distribution that
            must be transformed using the `initial_state_cov` covariance
            matrix. Default is True.
        random_state : {None, int, Generator, RandomState}, optional
            If `seed` is None (or `np.random`), the
            class:``~numpy.random.RandomState`` singleton is used.
            If `seed` is an int, a new class:``~numpy.random.RandomState``
            instance is used, seeded with `seed`.
            If `seed` is already a class:``~numpy.random.Generator`` or
            class:``~numpy.random.RandomState`` instance then that instance is
            used.

        Returns
        -------
        simulated_obs : ndarray
            An array of simulated observations. If `repetitions=None`, then it
            will be shaped (nsimulations x k_endog) or (nsimulations,) if
            `k_endog=1`. Otherwise it will be shaped
            (nsimulations x k_endog x repetitions). If the model was given
            Pandas input then the output will be a Pandas object. If
            `k_endog > 1` and `repetitions` is not None, then the output will
            be a Pandas DataFrame that has a MultiIndex for the columns, with
            the first level containing the names of the `endog` variables and
            the second level containing the repetition number.

        See Also
        --------
        impulse_responses
            Impulse response functions
        """
        # Get the starting location
        if anchor is None or anchor == 'start':
            iloc = 0
        elif anchor == 'end':
            iloc = self.nobs
        else:
            iloc, _, _ = self.model._get_index_loc(anchor)
            if isinstance(iloc, slice):
                iloc = iloc.start

        if iloc < 0:
            iloc = self.nobs + iloc
        if iloc > self.nobs:
            raise ValueError('Cannot anchor simulation outside of the sample.')

        # Setup the initial state
        if initial_state is None:
            initial_state_moments = (
                self.predicted_state[:, iloc],
                self.predicted_state_cov[:, :, iloc])

            _repetitions = 1 if repetitions is None else repetitions

            initial_state = np.random.multivariate_normal(
                *initial_state_moments, size=_repetitions).T

        scale = self.scale if self.filter_results.filter_concentrated else None
        with self.model.ssm.fixed_scale(scale):
            sim = self.model.simulate(
                self.params, nsimulations,
                measurement_shocks=measurement_shocks,
                state_shocks=state_shocks, initial_state=initial_state,
                anchor=anchor, repetitions=repetitions, exog=exog,
                transformed=True, includes_fixed=True,
                extend_model=extend_model, extend_kwargs=extend_kwargs,
                pretransformed_measurement_shocks=(
                    pretransformed_measurement_shocks),
                pretransformed_state_shocks=pretransformed_state_shocks,
                pretransformed_initial_state=pretransformed_initial_state,
                random_state=random_state, **kwargs)

        return sim

    def impulse_responses(self, steps=1, impulse=0, orthogonalized=False,
                          cumulative=False, **kwargs):
        """
        Impulse response function

        Parameters
        ----------
        steps : int, optional
            The number of steps for which impulse responses are calculated.
            Default is 1. Note that for time-invariant models, the initial
            impulse is not counted as a step, so if `steps=1`, the output will
            have 2 entries.
        impulse : int, str or array_like
            If an integer, the state innovation to pulse; must be between 0
            and `k_posdef-1`. If a str, it indicates which column of df
            the unit (1) impulse is given.
            Alternatively, a custom impulse vector may be provided; must be
            shaped `k_posdef x 1`.
        orthogonalized : bool, optional
            Whether or not to perform impulse using orthogonalized innovations.
            Note that this will also affect custum `impulse` vectors. Default
            is False.
        cumulative : bool, optional
            Whether or not to return cumulative impulse responses. Default is
            False.
        anchor : int, str, or datetime, optional
            Time point within the sample for the state innovation impulse. Type
            depends on the index of the given `endog` in the model. Two special
            cases are the strings 'start' and 'end', which refer to setting the
            impulse at the first and last points of the sample, respectively.
            Integer values can run from 0 to `nobs - 1`, or can be negative to
            apply negative indexing. Finally, if a date/time index was provided
            to the model, then this argument can be a date string to parse or a
            datetime type. Default is 'start'.
        exog : array_like, optional
            New observations of exogenous regressors, if applicable.
        **kwargs
            If the model has time-varying design or transition matrices and the
            combination of `anchor` and `steps` implies creating impulse
            responses for the out-of-sample period, then these matrices must
            have updated values provided for the out-of-sample steps. For
            example, if `design` is a time-varying component, `nobs` is 10,
            `anchor=1`, and `steps` is 15, a (`k_endog` x `k_states` x 7)
            matrix must be provided with the new design matrix values.

        Returns
        -------
        impulse_responses : ndarray
            Responses for each endogenous variable due to the impulse
            given by the `impulse` argument. For a time-invariant model, the
            impulse responses are given for `steps + 1` elements (this gives
            the "initial impulse" followed by `steps` responses for the
            important cases of VAR and SARIMAX models), while for time-varying
            models the impulse responses are only given for `steps` elements
            (to avoid having to unexpectedly provide updated time-varying
            matrices).

        See Also
        --------
        simulate
            Simulate a time series according to the given state space model,
            optionally with specified series for the innovations.

        Notes
        -----
        Intercepts in the measurement and state equation are ignored when
        calculating impulse responses.
        """
        scale = self.scale if self.filter_results.filter_concentrated else None
        with self.model.ssm.fixed_scale(scale):
            irfs = self.model.impulse_responses(self.params, steps, impulse,
                                                orthogonalized, cumulative,
                                                **kwargs)
            # These are wrapped automatically, so just return the array
            if isinstance(irfs, (pd.Series, pd.DataFrame)):
                irfs = irfs.values
        return irfs

    def _apply(self, mod, refit=False, fit_kwargs=None):
        if fit_kwargs is None:
            fit_kwargs = {}

        if refit:
            fit_kwargs.setdefault('start_params', self.params)
            if self._has_fixed_params:
                fit_kwargs.setdefault('includes_fixed', True)
                res = mod.fit_constrained(self._fixed_params, **fit_kwargs)
            else:
                res = mod.fit(**fit_kwargs)
        else:
            if 'cov_type' in fit_kwargs:
                raise ValueError('Cannot specify covariance type in'
                                 ' `fit_kwargs` unless refitting'
                                 ' parameters (not available in extend).')
            if 'cov_kwds' in fit_kwargs:
                raise ValueError('Cannot specify covariance keyword arguments'
                                 ' in `fit_kwargs` unless refitting'
                                 ' parameters (not available in extend).')

            if self.cov_type == 'none':
                fit_kwargs['cov_type'] = 'none'
            else:
                fit_kwargs['cov_type'] = 'custom'
                fit_kwargs['cov_kwds'] = {
                    'custom_cov_type': self.cov_type,
                    'custom_cov_params': self.cov_params_default,
                    'custom_description': ('Parameters and standard errors'
                                           ' were estimated using a different'
                                           ' dataset and were then applied to'
                                           ' this dataset. %s'
                                           % self.cov_kwds['description'])}

            if self.smoother_results is not None:
                func = mod.smooth
            else:
                func = mod.filter

            if self._has_fixed_params:
                with mod.fix_params(self._fixed_params):
                    fit_kwargs.setdefault('includes_fixed', True)
                    res = func(self.params, **fit_kwargs)
            else:
                res = func(self.params, **fit_kwargs)

        return res

    def _get_previous_updated(self, comparison, exog=None,
                              comparison_type=None, **kwargs):
        # If we were given data, create a new results object
        comparison_dataset = not isinstance(
            comparison, (MLEResults, MLEResultsWrapper))
        if comparison_dataset:
            # If `exog` is longer than `comparison`, then we extend it to match
            nobs_endog = len(comparison)
            nobs_exog = len(exog) if exog is not None else nobs_endog

            if nobs_exog > nobs_endog:
                _, _, _, ix = self.model._get_prediction_index(
                    start=0, end=nobs_exog - 1)
                # TODO: check that the index of `comparison` matches the model
                comparison = np.asarray(comparison)
                if comparison.ndim < 2:
                    comparison = np.atleast_2d(comparison).T
                if (comparison.ndim != 2 or
                        comparison.shape[1] != self.model.k_endog):
                    raise ValueError('Invalid shape for `comparison`. Must'
                                     f' contain {self.model.k_endog} columns.')
                extra = np.zeros((nobs_exog - nobs_endog,
                                  self.model.k_endog)) * np.nan
                comparison = pd.DataFrame(
                    np.concatenate([comparison, extra], axis=0), index=ix,
                    columns=self.model.endog_names)

            # Get the results object
            comparison = self.apply(comparison, exog=exog,
                                    copy_initialization=True, **kwargs)

        # Now, figure out the `updated` versus `previous` results objects
        nmissing = self.filter_results.missing.sum()
        nmissing_comparison = comparison.filter_results.missing.sum()
        if (comparison_type == 'updated' or (comparison_type is None and (
                comparison.nobs > self.nobs or
                (comparison.nobs == self.nobs and
                 nmissing > nmissing_comparison)))):
            updated = comparison
            previous = self
        elif (comparison_type == 'previous' or (comparison_type is None and (
                comparison.nobs < self.nobs or
                (comparison.nobs == self.nobs and
                 nmissing < nmissing_comparison)))):
            updated = self
            previous = comparison
        else:
            raise ValueError('Could not automatically determine the type'
                             ' of comparison requested to compute the'
                             ' News, so it must be specified as "updated"'
                             ' or "previous", using the `comparison_type`'
                             ' keyword argument')

        # Check that the index of `updated` is a superset of the
        # index of `previous`
        # Note: the try/except block is for Pandas < 0.25, in which
        # `PeriodIndex.difference` raises a ValueError if the argument is not
        # also a `PeriodIndex`.
        diff = previous.model._index.difference(updated.model._index)
        if len(diff) > 0:
            raise ValueError('The index associated with the updated results is'
                             ' not a superset of the index associated with the'
                             ' previous results, and so these datasets do not'
                             ' appear to be related. Can only compute the'
                             ' news by comparing this results set to previous'
                             ' results objects.')

        return previous, updated, comparison_dataset

    def _news_previous_results(self, previous, start, end, periods,
                               state_index=None):
        # Compute the news
        out = self.smoother_results.news(previous.smoother_results,
                                         start=start, end=end,
                                         state_index=state_index)
        return out

    def _news_updated_results(self, updated, start, end, periods,
                              state_index=None):
        return updated._news_previous_results(self, start, end, periods,
                                              state_index=state_index)

    def _news_previous_data(self, endog, start, end, periods, exog,
                            state_index=None):
        previous = self.apply(endog, exog=exog, copy_initialization=True)
        return self._news_previous_results(previous, start, end, periods,
                                           state_index=state_index)

    def _news_updated_data(self, endog, start, end, periods, exog,
                           state_index=None):
        updated = self.apply(endog, exog=exog, copy_initialization=True)
        return self._news_updated_results(updated, start, end, periods,
                                          state_index=state_index)

    def news(self, comparison, impact_date=None, impacted_variable=None,
             start=None, end=None, periods=None, exog=None,
             comparison_type=None, state_index=None, return_raw=False,
             tolerance=1e-10, **kwargs):
        """
        Compute impacts from updated data (news and revisions)

        Parameters
        ----------
        comparison : array_like or MLEResults
            An updated dataset with updated and/or revised data from which the
            news can be computed, or an updated or previous results object
            to use in computing the news.
        impact_date : int, str, or datetime, optional
            A single specific period of impacts from news and revisions to
            compute. Can also be a date string to parse or a datetime type.
            This argument cannot be used in combination with `start`, `end`, or
            `periods`. Default is the first out-of-sample observation.
        impacted_variable : str, list, array, or slice, optional
            Observation variable label or slice of labels specifying that only
            specific impacted variables should be shown in the News output. The
            impacted variable(s) describe the variables that were *affected* by
            the news. If you do not know the labels for the variables, check
            the `endog_names` attribute of the model instance.
        start : int, str, or datetime, optional
            The first period of impacts from news and revisions to compute.
            Can also be a date string to parse or a datetime type. Default is
            the first out-of-sample observation.
        end : int, str, or datetime, optional
            The last period of impacts from news and revisions to compute.
            Can also be a date string to parse or a datetime type. Default is
            the first out-of-sample observation.
        periods : int, optional
            The number of periods of impacts from news and revisions to
            compute.
        exog : array_like, optional
            Array of exogenous regressors for the out-of-sample period, if
            applicable.
        comparison_type : {None, 'previous', 'updated'}
            This denotes whether the `comparison` argument represents a
            *previous* results object or dataset or an *updated* results object
            or dataset. If not specified, then an attempt is made to determine
            the comparison type.
        state_index : array_like, optional
            An optional index specifying a subset of states to use when
            constructing the impacts of revisions and news. For example, if
            `state_index=[0, 1]` is passed, then only the impacts to the
            observed variables arising from the impacts to the first two
            states will be returned. Default is to use all states.
        return_raw : bool, optional
            Whether or not to return only the specific output or a full
            results object. Default is to return a full results object.
        tolerance : float, optional
            The numerical threshold for determining zero impact. Default is
            that any impact less than 1e-10 is assumed to be zero.

        Returns
        -------
        NewsResults
            Impacts of data revisions and news on estimates

        References
        ----------
        .. [1] Bańbura, Marta, and Michele Modugno.
               "Maximum likelihood estimation of factor models on datasets with
               arbitrary pattern of missing data."
               Journal of Applied Econometrics 29, no. 1 (2014): 133-160.
        .. [2] Bańbura, Marta, Domenico Giannone, and Lucrezia Reichlin.
               "Nowcasting."
               The Oxford Handbook of Economic Forecasting. July 8, 2011.
        .. [3] Bańbura, Marta, Domenico Giannone, Michele Modugno, and Lucrezia
               Reichlin.
               "Now-casting and the real-time data flow."
               In Handbook of economic forecasting, vol. 2, pp. 195-237.
               Elsevier, 2013.
        """
        # Validate input
        if self.smoother_results is None:
            raise ValueError('Cannot compute news without Kalman smoother'
                             ' results.')

        if state_index is not None:
            state_index = np.sort(np.array(state_index, dtype=int))
            if state_index[0] < 0:
                raise ValueError('Cannot include negative indexes in'
                                 ' `state_index`.')
            if state_index[-1] >= self.model.k_states:
                raise ValueError(f'Given state index {state_index[-1]} is too'
                                 ' large for the number of states in the model'
                                 f' ({self.model.k_states}).')

        # Get the previous and updated results objects from `self` and
        # `comparison`:
        previous, updated, comparison_dataset = self._get_previous_updated(
            comparison, exog=exog, comparison_type=comparison_type, **kwargs)

        # Handle start, end, periods
        start, end, prediction_index = get_impact_dates(
            previous_model=previous.model, updated_model=updated.model,
            impact_date=impact_date, start=start, end=end, periods=periods)

        # News results will always use Pandas, so if the model's data was not
        # from Pandas, we'll create an index, as if the model's data had been
        # given a default Pandas index.
        if prediction_index is None:
            prediction_index = pd.RangeIndex(start=start, stop=end + 1)

        # For time-varying models try to create an appended `updated` model
        # with NaN values. Do not extend the model if this was already done
        # above (i.e. the case that `comparison` was a new dataset), because
        # in that case `exog` and `kwargs` should have
        # been set with the input `comparison` dataset in mind, and so would be
        # useless here. Ultimately, we've already extended `updated` as far
        # as we can. So raise an  exception in that case with a useful message.
        # However, we still want to try to accommodate extending the model here
        # if it is possible.
        # Note that we do not need to extend time-invariant models, because
        # `KalmanSmoother.news` can itself handle any impact dates for
        # time-invariant models.
        time_varying = not (previous.filter_results.time_invariant or
                            updated.filter_results.time_invariant)
        if time_varying and end >= updated.nobs:
            # If we the given `comparison` was a dataset and either `exog` or
            # `kwargs` was set, then we assume that we cannot create an updated
            # time-varying model (because then we can't tell if `kwargs` and
            # `exog` arguments are meant to apply to the `comparison` dataset
            # or to this extension)
            if comparison_dataset and (exog is not None or len(kwargs) > 0):
                if comparison is updated:
                    raise ValueError('If providing an updated dataset as the'
                                     ' `comparison` with a time-varying model,'
                                     ' then the `end` period cannot be beyond'
                                     ' the end of that updated dataset.')
                else:
                    raise ValueError('If providing an previous dataset as the'
                                     ' `comparison` with a time-varying model,'
                                     ' then the `end` period cannot be beyond'
                                     ' the end of the (updated) results'
                                     ' object.')

            # Try to extend `updated`
            updated_orig = updated
            # TODO: `append` should fix this k_endog=1 issue for us
            # TODO: is the + 1 necessary?
            if self.model.k_endog > 1:
                extra = np.zeros((end - updated.nobs + 1,
                                  self.model.k_endog)) * np.nan
            else:
                extra = np.zeros((end - updated.nobs + 1,)) * np.nan
            updated = updated_orig.append(extra, exog=exog, **kwargs)

        # Compute the news
        news_results = (
            updated._news_previous_results(previous, start, end + 1, periods,
                                           state_index=state_index))

        if not return_raw:
            news_results = NewsResults(
                news_results, self, updated, previous, impacted_variable,
                tolerance, row_labels=prediction_index)
        return news_results

    def get_smoothed_decomposition(self, decomposition_of='smoothed_state',
                                   state_index=None):
        r"""
        Decompose smoothed output into contributions from observations

        Parameters
        ----------
        decomposition_of : {"smoothed_state", "smoothed_signal"}
            The object to perform a decomposition of. If it is set to
            "smoothed_state", then the elements of the smoothed state vector
            are decomposed into the contributions of each observation. If it
            is set to "smoothed_signal", then the predictions of the
            observation vector based on the smoothed state vector are
            decomposed. Default is "smoothed_state".
        state_index : array_like, optional
            An optional index specifying a subset of states to use when
            constructing the decomposition of the "smoothed_signal". For
            example, if `state_index=[0, 1]` is passed, then only the
            contributions of observed variables to the smoothed signal arising
            from the first two states will be returned. Note that if not all
            states are used, the contributions will not sum to the smoothed
            signal. Default is to use all states.

        Returns
        -------
        data_contributions : pd.DataFrame
            Contributions of observations to the decomposed object. If the
            smoothed state is being decomposed, then `data_contributions` is
            shaped `(k_states x nobs, k_endog x nobs)` with a `pd.MultiIndex`
            index corresponding to `state_to x date_to` and `pd.MultiIndex`
            columns corresponding to `variable_from x date_from`. If the
            smoothed signal is being decomposed, then `data_contributions` is
            shaped `(k_endog x nobs, k_endog x nobs)` with `pd.MultiIndex`-es
            corresponding to `variable_to x date_to` and
            `variable_from x date_from`.
        obs_intercept_contributions : pd.DataFrame
            Contributions of the observation intercept to the decomposed
            object. If the smoothed state is being decomposed, then
            `obs_intercept_contributions` is
            shaped `(k_states x nobs, k_endog x nobs)` with a `pd.MultiIndex`
            index corresponding to `state_to x date_to` and `pd.MultiIndex`
            columns corresponding to `obs_intercept_from x date_from`. If the
            smoothed signal is being decomposed, then
            `obs_intercept_contributions` is shaped
            `(k_endog x nobs, k_endog x nobs)` with `pd.MultiIndex`-es
            corresponding to `variable_to x date_to` and
            `obs_intercept_from x date_from`.
        state_intercept_contributions : pd.DataFrame
            Contributions of the state intercept to the decomposed
            object. If the smoothed state is being decomposed, then
            `state_intercept_contributions` is
            shaped `(k_states x nobs, k_states x nobs)` with a `pd.MultiIndex`
            index corresponding to `state_to x date_to` and `pd.MultiIndex`
            columns corresponding to `state_intercept_from x date_from`. If the
            smoothed signal is being decomposed, then
            `state_intercept_contributions` is shaped
            `(k_endog x nobs, k_states x nobs)` with `pd.MultiIndex`-es
            corresponding to `variable_to x date_to` and
            `state_intercept_from x date_from`.
        prior_contributions : pd.DataFrame
            Contributions of the prior to the decomposed object. If the
            smoothed state is being decomposed, then `prior_contributions` is
            shaped `(nobs x k_states, k_states)`, with a `pd.MultiIndex`
            index corresponding to `state_to x date_to` and columns
            corresponding to elements of the prior mean (aka "initial state").
            If the smoothed signal is being decomposed, then
            `prior_contributions` is shaped `(nobs x k_endog, k_states)`,
            with a `pd.MultiIndex` index corresponding to
            `variable_to x date_to` and columns corresponding to elements of
            the prior mean.

        Notes
        -----
        Denote the smoothed state at time :math:`t` by :math:`\alpha_t`. Then
        the smoothed signal is :math:`Z_t \alpha_t`, where :math:`Z_t` is the
        design matrix operative at time :math:`t`.
        """
        (data_contributions, obs_intercept_contributions,
         state_intercept_contributions, prior_contributions) = (
            self.smoother_results.get_smoothed_decomposition(
                decomposition_of=decomposition_of, state_index=state_index))

        # Construct indexes
        endog_names = self.model.endog_names
        if self.model.k_endog == 1:
            endog_names = [endog_names]

        if decomposition_of == 'smoothed_state':
            contributions_to = pd.MultiIndex.from_product(
                [self.model.state_names, self.model._index],
                names=['state_to', 'date_to'])
        else:
            contributions_to = pd.MultiIndex.from_product(
                [endog_names, self.model._index],
                names=['variable_to', 'date_to'])
        contributions_from = pd.MultiIndex.from_product(
            [endog_names, self.model._index],
            names=['variable_from', 'date_from'])
        obs_intercept_contributions_from = pd.MultiIndex.from_product(
            [endog_names, self.model._index],
            names=['obs_intercept_from', 'date_from'])
        state_intercept_contributions_from = pd.MultiIndex.from_product(
            [self.model.state_names, self.model._index],
            names=['state_intercept_from', 'date_from'])
        prior_contributions_from = pd.Index(self.model.state_names,
                                            name='initial_state_from')

        # Construct DataFrames
        shape = data_contributions.shape
        data_contributions = pd.DataFrame(
            data_contributions.reshape(
                shape[0] * shape[1], shape[2] * shape[3], order='F'),
            index=contributions_to, columns=contributions_from)

        shape = obs_intercept_contributions.shape
        obs_intercept_contributions = pd.DataFrame(
            obs_intercept_contributions.reshape(
                shape[0] * shape[1], shape[2] * shape[3], order='F'),
            index=contributions_to, columns=obs_intercept_contributions_from)

        shape = state_intercept_contributions.shape
        state_intercept_contributions = pd.DataFrame(
            state_intercept_contributions.reshape(
                shape[0] * shape[1], shape[2] * shape[3], order='F'),
            index=contributions_to, columns=state_intercept_contributions_from)

        shape = prior_contributions.shape
        prior_contributions = pd.DataFrame(
            prior_contributions.reshape(shape[0] * shape[1], shape[2],
                                        order='F'),
            index=contributions_to, columns=prior_contributions_from)

        return (data_contributions, obs_intercept_contributions,
                state_intercept_contributions, prior_contributions)

    def append(self, endog, exog=None, refit=False, fit_kwargs=None,
               copy_initialization=False, **kwargs):
        """
        Recreate the results object with new data appended to the original data

        Creates a new result object applied to a dataset that is created by
        appending new data to the end of the model's original data. The new
        results can then be used for analysis or forecasting.

        Parameters
        ----------
        endog : array_like
            New observations from the modeled time-series process.
        exog : array_like, optional
            New observations of exogenous regressors, if applicable.
        refit : bool, optional
            Whether to re-fit the parameters, based on the combined dataset.
            Default is False (so parameters from the current results object
            are used to create the new results object).
        copy_initialization : bool, optional
            Whether or not to copy the initialization from the current results
            set to the new model. Default is False
        fit_kwargs : dict, optional
            Keyword arguments to pass to `fit` (if `refit=True`) or `filter` /
            `smooth`.
        copy_initialization : bool, optional
        **kwargs
            Keyword arguments may be used to modify model specification
            arguments when created the new model object.

        Returns
        -------
        results
            Updated Results object, that includes results from both the
            original dataset and the new dataset.

        Notes
        -----
        The `endog` and `exog` arguments to this method must be formatted in
        the same way (e.g. Pandas Series versus Numpy array) as were the
        `endog` and `exog` arrays passed to the original model.

        The `endog` argument to this method should consist of new observations
        that occurred directly after the last element of `endog`. For any other
        kind of dataset, see the `apply` method.

        This method will apply filtering to all of the original data as well
        as to the new data. To apply filtering only to the new data (which
        can be much faster if the original dataset is large), see the `extend`
        method.

        See Also
        --------
        statsmodels.tsa.statespace.mlemodel.MLEResults.extend
        statsmodels.tsa.statespace.mlemodel.MLEResults.apply

        Examples
        --------
        >>> index = pd.period_range(start='2000', periods=2, freq='A')
        >>> original_observations = pd.Series([1.2, 1.5], index=index)
        >>> mod = sm.tsa.SARIMAX(original_observations)
        >>> res = mod.fit()
        >>> print(res.params)
        ar.L1     0.9756
        sigma2    0.0889
        dtype: float64
        >>> print(res.fittedvalues)
        2000    0.0000
        2001    1.1707
        Freq: A-DEC, dtype: float64
        >>> print(res.forecast(1))
        2002    1.4634
        Freq: A-DEC, dtype: float64

        >>> new_index = pd.period_range(start='2002', periods=1, freq='A')
        >>> new_observations = pd.Series([0.9], index=new_index)
        >>> updated_res = res.append(new_observations)
        >>> print(updated_res.params)
        ar.L1     0.9756
        sigma2    0.0889
        dtype: float64
        >>> print(updated_res.fittedvalues)
        2000    0.0000
        2001    1.1707
        2002    1.4634
        Freq: A-DEC, dtype: float64
        >>> print(updated_res.forecast(1))
        2003    0.878
        Freq: A-DEC, dtype: float64
        """
        start = self.nobs
        end = self.nobs + len(endog) - 1
        _, _, _, append_ix = self.model._get_prediction_index(start, end)

        # Check the index of the new data
        if isinstance(self.model.data, PandasData):
            _check_index(append_ix, endog, '`endog`')

        # Concatenate the new data to original data
        new_endog = concat([self.model.data.orig_endog, endog], axis=0,
                           allow_mix=True)

        # Handle `exog`
        if exog is not None:
            _, exog = prepare_exog(exog)
            _check_index(append_ix, exog, '`exog`')

            new_exog = concat([self.model.data.orig_exog, exog], axis=0,
                              allow_mix=True)
        else:
            new_exog = None

        # Create a continuous index for the combined data
        if isinstance(self.model.data, PandasData):
            start = 0
            end = len(new_endog) - 1
            _, _, _, new_index = self.model._get_prediction_index(start, end)

            # Standardize `endog` to have the right index and columns
            columns = self.model.endog_names
            if not isinstance(columns, list):
                columns = [columns]
            new_endog = pd.DataFrame(new_endog, index=new_index,
                                     columns=columns)

            # Standardize `exog` to have the right index
            if new_exog is not None:
                new_exog = pd.DataFrame(new_exog, index=new_index,
                                        columns=self.model.exog_names)

        if copy_initialization:
            init = Initialization.from_results(self.filter_results)
            kwargs.setdefault('initialization', init)

        mod = self.model.clone(new_endog, exog=new_exog, **kwargs)
        res = self._apply(mod, refit=refit, fit_kwargs=fit_kwargs)

        return res

    def extend(self, endog, exog=None, fit_kwargs=None, **kwargs):
        """
        Recreate the results object for new data that extends the original data

        Creates a new result object applied to a new dataset that is assumed to
        follow directly from the end of the model's original data. The new
        results can then be used for analysis or forecasting.

        Parameters
        ----------
        endog : array_like
            New observations from the modeled time-series process.
        exog : array_like, optional
            New observations of exogenous regressors, if applicable.
        fit_kwargs : dict, optional
            Keyword arguments to pass to `filter` or `smooth`.
        **kwargs
            Keyword arguments may be used to modify model specification
            arguments when created the new model object.

        Returns
        -------
        results
            Updated Results object, that includes results only for the new
            dataset.

        See Also
        --------
        statsmodels.tsa.statespace.mlemodel.MLEResults.append
        statsmodels.tsa.statespace.mlemodel.MLEResults.apply

        Notes
        -----
        The `endog` argument to this method should consist of new observations
        that occurred directly after the last element of the model's original
        `endog` array. For any other kind of dataset, see the `apply` method.

        This method will apply filtering only to the new data provided by the
        `endog` argument, which can be much faster than re-filtering the entire
        dataset. However, the returned results object will only have results
        for the new data. To retrieve results for both the new data and the
        original data, see the `append` method.

        Examples
        --------
        >>> index = pd.period_range(start='2000', periods=2, freq='A')
        >>> original_observations = pd.Series([1.2, 1.5], index=index)
        >>> mod = sm.tsa.SARIMAX(original_observations)
        >>> res = mod.fit()
        >>> print(res.params)
        ar.L1     0.9756
        sigma2    0.0889
        dtype: float64
        >>> print(res.fittedvalues)
        2000    0.0000
        2001    1.1707
        Freq: A-DEC, dtype: float64
        >>> print(res.forecast(1))
        2002    1.4634
        Freq: A-DEC, dtype: float64

        >>> new_index = pd.period_range(start='2002', periods=1, freq='A')
        >>> new_observations = pd.Series([0.9], index=new_index)
        >>> updated_res = res.extend(new_observations)
        >>> print(updated_res.params)
        ar.L1     0.9756
        sigma2    0.0889
        dtype: float64
        >>> print(updated_res.fittedvalues)
        2002    1.4634
        Freq: A-DEC, dtype: float64
        >>> print(updated_res.forecast(1))
        2003    0.878
        Freq: A-DEC, dtype: float64
        """
        start = self.nobs
        end = self.nobs + len(endog) - 1
        _, _, _, extend_ix = self.model._get_prediction_index(start, end)

        if isinstance(self.model.data, PandasData):
            _check_index(extend_ix, endog, '`endog`')

            # Standardize `endog` to have the right index and columns
            columns = self.model.endog_names
            if not isinstance(columns, list):
                columns = [columns]
            endog = pd.DataFrame(endog, index=extend_ix, columns=columns)
        # Extend the current fit result to additional data
        mod = self.model.clone(endog, exog=exog, **kwargs)
        mod.ssm.initialization = Initialization(
            mod.k_states, 'known', constant=self.predicted_state[..., -1],
            stationary_cov=self.predicted_state_cov[..., -1])
        res = self._apply(mod, refit=False, fit_kwargs=fit_kwargs)

        return res

    def apply(self, endog, exog=None, refit=False, fit_kwargs=None,
              copy_initialization=False, **kwargs):
        """
        Apply the fitted parameters to new data unrelated to the original data

        Creates a new result object using the current fitted parameters,
        applied to a completely new dataset that is assumed to be unrelated to
        the model's original data. The new results can then be used for
        analysis or forecasting.

        Parameters
        ----------
        endog : array_like
            New observations from the modeled time-series process.
        exog : array_like, optional
            New observations of exogenous regressors, if applicable.
        refit : bool, optional
            Whether to re-fit the parameters, using the new dataset.
            Default is False (so parameters from the current results object
            are used to create the new results object).
        copy_initialization : bool, optional
            Whether or not to copy the initialization from the current results
            set to the new model. Default is False
        fit_kwargs : dict, optional
            Keyword arguments to pass to `fit` (if `refit=True`) or `filter` /
            `smooth`.
        **kwargs
            Keyword arguments may be used to modify model specification
            arguments when created the new model object.

        Returns
        -------
        results
            Updated Results object, that includes results only for the new
            dataset.

        See Also
        --------
        statsmodels.tsa.statespace.mlemodel.MLEResults.append
        statsmodels.tsa.statespace.mlemodel.MLEResults.apply

        Notes
        -----
        The `endog` argument to this method should consist of new observations
        that are not necessarily related to the original model's `endog`
        dataset. For observations that continue that original dataset by follow
        directly after its last element, see the `append` and `extend` methods.

        Examples
        --------
        >>> index = pd.period_range(start='2000', periods=2, freq='A')
        >>> original_observations = pd.Series([1.2, 1.5], index=index)
        >>> mod = sm.tsa.SARIMAX(original_observations)
        >>> res = mod.fit()
        >>> print(res.params)
        ar.L1     0.9756
        sigma2    0.0889
        dtype: float64
        >>> print(res.fittedvalues)
        2000    0.0000
        2001    1.1707
        Freq: A-DEC, dtype: float64
        >>> print(res.forecast(1))
        2002    1.4634
        Freq: A-DEC, dtype: float64

        >>> new_index = pd.period_range(start='1980', periods=3, freq='A')
        >>> new_observations = pd.Series([1.4, 0.3, 1.2], index=new_index)
        >>> new_res = res.apply(new_observations)
        >>> print(new_res.params)
        ar.L1     0.9756
        sigma2    0.0889
        dtype: float64
        >>> print(new_res.fittedvalues)
        1980    1.1707
        1981    1.3659
        1982    0.2927
        Freq: A-DEC, dtype: float64
        Freq: A-DEC, dtype: float64
        >>> print(new_res.forecast(1))
        1983    1.1707
        Freq: A-DEC, dtype: float64
        """
        mod = self.model.clone(endog, exog=exog, **kwargs)

        if copy_initialization:
            init = Initialization.from_results(self.filter_results)
            mod.ssm.initialization = init

        res = self._apply(mod, refit=refit, fit_kwargs=fit_kwargs)

        return res

    def plot_diagnostics(self, variable=0, lags=10, fig=None, figsize=None,
                         truncate_endog_names=24, auto_ylims=False,
                         bartlett_confint=False, acf_kwargs=None):
        """
        Diagnostic plots for standardized residuals of one endogenous variable

        Parameters
        ----------
        variable : int, optional
            Index of the endogenous variable for which the diagnostic plots
            should be created. Default is 0.
        lags : int, optional
            Number of lags to include in the correlogram. Default is 10.
        fig : Figure, optional
            If given, subplots are created in this figure instead of in a new
            figure. Note that the 2x2 grid will be created in the provided
            figure using `fig.add_subplot()`.
        figsize : tuple, optional
            If a figure is created, this argument allows specifying a size.
            The tuple is (width, height).
        auto_ylims : bool, optional
            If True, adjusts automatically the y-axis limits to ACF values.
        bartlett_confint : bool, default True
            Confidence intervals for ACF values are generally placed at 2
            standard errors around r_k. The formula used for standard error
            depends upon the situation. If the autocorrelations are being used
            to test for randomness of residuals as part of the ARIMA routine,
            the standard errors are determined assuming the residuals are white
            noise. The approximate formula for any lag is that standard error
            of each r_k = 1/sqrt(N). See section 9.4 of [1] for more details on
            the 1/sqrt(N) result. For more elementary discussion, see section
            5.3.2 in [2].
            For the ACF of raw data, the standard error at a lag k is
            found as if the right model was an MA(k-1). This allows the
            possible interpretation that if all autocorrelations past a
            certain lag are within the limits, the model might be an MA of
            order defined by the last significant autocorrelation. In this
            case, a moving average model is assumed for the data and the
            standard errors for the confidence intervals should be
            generated using Bartlett's formula. For more details on
            Bartlett formula result, see section 7.2 in [1].+
        acf_kwargs : dict, optional
            Optional dictionary of keyword arguments that are directly passed
            on to the correlogram Matplotlib plot produced by plot_acf().

        Returns
        -------
        Figure
            Figure instance with diagnostic plots

        See Also
        --------
        statsmodels.graphics.gofplots.qqplot
        statsmodels.graphics.tsaplots.plot_acf

        Notes
        -----
        Produces a 2x2 plot grid with the following plots (ordered clockwise
        from top left):

        1. Standardized residuals over time
        2. Histogram plus estimated density of standardized residuals, along
           with a Normal(0,1) density plotted for reference.
        3. Normal Q-Q plot, with Normal reference line.
        4. Correlogram

        References
        ----------
        [1] Brockwell and Davis, 1987. Time Series Theory and Methods
        [2] Brockwell and Davis, 2010. Introduction to Time Series and
        Forecasting, 2nd edition.
        """
        from statsmodels.graphics.utils import _import_mpl, create_mpl_fig
        _import_mpl()
        fig = create_mpl_fig(fig, figsize)
        # Eliminate residuals associated with burned or diffuse likelihoods
        d = np.maximum(self.loglikelihood_burn, self.nobs_diffuse)

        # If given a variable name, find the index
        if isinstance(variable, str):
            variable = self.model.endog_names.index(variable)

        # Get residuals
        if hasattr(self.data, 'dates') and self.data.dates is not None:
            ix = self.data.dates[d:]
        else:
            ix = np.arange(self.nobs - d)
        resid = pd.Series(
            self.filter_results.standardized_forecasts_error[variable, d:],
            index=ix)

        if resid.shape[0] < max(d, lags):
            raise ValueError(
                "Length of endogenous variable must be larger the the number "
                "of lags used in the model and the number of observations "
                "burned in the log-likelihood calculation."
            )

        # Top-left: residuals vs time
        ax = fig.add_subplot(221)
        resid.dropna().plot(ax=ax)
        ax.hlines(0, ix[0], ix[-1], alpha=0.5)
        ax.set_xlim(ix[0], ix[-1])
        name = self.model.endog_names[variable]
        if len(name) > truncate_endog_names:
            name = name[:truncate_endog_names - 3] + '...'
        ax.set_title(f'Standardized residual for "{name}"')

        # Top-right: histogram, Gaussian kernel density, Normal density
        # Can only do histogram and Gaussian kernel density on the non-null
        # elements
        resid_nonmissing = resid.dropna()
        ax = fig.add_subplot(222)

        ax.hist(resid_nonmissing, density=True, label='Hist',
                edgecolor='#FFFFFF')

        from scipy.stats import gaussian_kde, norm
        kde = gaussian_kde(resid_nonmissing)
        xlim = (-1.96*2, 1.96*2)
        x = np.linspace(xlim[0], xlim[1])
        ax.plot(x, kde(x), label='KDE')
        ax.plot(x, norm.pdf(x), label='N(0,1)')
        ax.set_xlim(xlim)
        ax.legend()
        ax.set_title('Histogram plus estimated density')

        # Bottom-left: QQ plot
        ax = fig.add_subplot(223)
        from statsmodels.graphics.gofplots import qqplot
        qqplot(resid_nonmissing, line='s', ax=ax)
        ax.set_title('Normal Q-Q')

        # Bottom-right: Correlogram
        ax = fig.add_subplot(224)
        from statsmodels.graphics.tsaplots import plot_acf

        if acf_kwargs is None:
            acf_kwargs = {}
        plot_acf(resid, ax=ax, lags=lags, auto_ylims=auto_ylims,
                 bartlett_confint=bartlett_confint, **acf_kwargs)
        ax.set_title('Correlogram')

        return fig

    def summary(self, alpha=.05, start=None, title=None, model_name=None,
                display_params=True, display_diagnostics=True,
                truncate_endog_names=None, display_max_endog=None,
                extra_top_left=None, extra_top_right=None):
        """
        Summarize the Model

        Parameters
        ----------
        alpha : float, optional
            Significance level for the confidence intervals. Default is 0.05.
        start : int, optional
            Integer of the start observation. Default is 0.
        model_name : str
            The name of the model used. Default is to use model class name.

        Returns
        -------
        summary : Summary instance
            This holds the summary table and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary
        """
        from statsmodels.iolib.summary import Summary
        from statsmodels.iolib.table import SimpleTable
        from statsmodels.iolib.tableformatting import fmt_params

        # Model specification results
        model = self.model
        if title is None:
            title = 'Statespace Model Results'

        if start is None:
            start = 0
        if self.model._index_dates:
            ix = self.model._index
            d = ix[start]
            sample = ['%02d-%02d-%02d' % (d.month, d.day, d.year)]
            d = ix[-1]
            sample += ['- ' + '%02d-%02d-%02d' % (d.month, d.day, d.year)]
        else:
            sample = [str(start), ' - ' + str(self.nobs)]

        # Standardize the model name as a list of str
        if model_name is None:
            model_name = model.__class__.__name__

        # Truncate endog names
        if truncate_endog_names is None:
            truncate_endog_names = False if self.model.k_endog == 1 else 24
        endog_names = self.model.endog_names
        if not isinstance(endog_names, list):
            endog_names = [endog_names]
        endog_names = [str(name) for name in endog_names]
        if truncate_endog_names is not False:
            n = truncate_endog_names
            endog_names = [name if len(name) <= n else name[:n] + '...'
                           for name in endog_names]

        # Shorten the endog name list if applicable
        if display_max_endog is None:
            display_max_endog = np.inf
        yname = None
        if self.model.k_endog > display_max_endog:
            k = self.model.k_endog - 1
            yname = '"' + endog_names[0] + f'", and {k} more'

        # Create the tables
        if not isinstance(model_name, list):
            model_name = [model_name]

        top_left = [('Dep. Variable:', None)]
        top_left.append(('Model:', [model_name[0]]))
        for i in range(1, len(model_name)):
            top_left.append(('', ['+ ' + model_name[i]]))
        top_left += [
            ('Date:', None),
            ('Time:', None),
            ('Sample:', [sample[0]]),
            ('', [sample[1]])
        ]

        top_right = [
            ('No. Observations:', [self.nobs]),
            ('Log Likelihood', ["%#5.3f" % self.llf]),
        ]
        if hasattr(self, 'rsquared'):
            top_right.append(('R-squared:', ["%#8.3f" % self.rsquared]))
        top_right += [
            ('AIC', ["%#5.3f" % self.aic]),
            ('BIC', ["%#5.3f" % self.bic]),
            ('HQIC', ["%#5.3f" % self.hqic])]
        if (self.filter_results is not None and
                self.filter_results.filter_concentrated):
            top_right.append(('Scale', ["%#5.3f" % self.scale]))

        if hasattr(self, 'cov_type'):
            cov_type = self.cov_type
            if cov_type == 'none':
                cov_type = 'Not computed'
            top_left.append(('Covariance Type:', [cov_type]))

        if extra_top_left is not None:
            top_left += extra_top_left
        if extra_top_right is not None:
            top_right += extra_top_right

        summary = Summary()
        summary.add_table_2cols(self, gleft=top_left, gright=top_right,
                                title=title, yname=yname)
        table_ix = 1
        if len(self.params) > 0 and display_params:
            summary.add_table_params(self, alpha=alpha,
                                     xname=self.param_names, use_t=False)
            table_ix += 1

        # Diagnostic tests results
        if display_diagnostics:
            try:
                het = self.test_heteroskedasticity(method='breakvar')
            except Exception:  # FIXME: catch something specific
                het = np.zeros((self.model.k_endog, 2)) * np.nan
            try:
                lb = self.test_serial_correlation(method='ljungbox', lags=[1])
            except Exception:  # FIXME: catch something specific
                lb = np.zeros((self.model.k_endog, 2, 1)) * np.nan
            try:
                jb = self.test_normality(method='jarquebera')
            except Exception:  # FIXME: catch something specific
                jb = np.zeros((self.model.k_endog, 4)) * np.nan

            if self.model.k_endog <= display_max_endog:
                format_str = lambda array: [  # noqa:E731
                    ', '.join(['{0:.2f}'.format(i) for i in array])
                ]
                diagn_left = [
                    ('Ljung-Box (L1) (Q):', format_str(lb[:, 0, -1])),
                    ('Prob(Q):', format_str(lb[:, 1, -1])),
                    ('Heteroskedasticity (H):', format_str(het[:, 0])),
                    ('Prob(H) (two-sided):', format_str(het[:, 1]))]

                diagn_right = [('Jarque-Bera (JB):', format_str(jb[:, 0])),
                               ('Prob(JB):', format_str(jb[:, 1])),
                               ('Skew:', format_str(jb[:, 2])),
                               ('Kurtosis:', format_str(jb[:, 3]))
                               ]

                summary.add_table_2cols(self, gleft=diagn_left,
                                        gright=diagn_right, title="")
            else:
                columns = ['LjungBox\n(L1) (Q)', 'Prob(Q)',
                           'Het.(H)', 'Prob(H)',
                           'Jarque\nBera(JB)', 'Prob(JB)', 'Skew', 'Kurtosis']
                data = pd.DataFrame(
                    np.c_[lb[:, :2, -1], het[:, :2], jb[:, :4]],
                    index=endog_names, columns=columns).applymap(
                        lambda num: '' if pd.isnull(num) else '%.2f' % num)
                data.index.name = 'Residual of\nDep. variable'
                data = data.reset_index()

                params_data = data.values
                params_header = data.columns.tolist()
                params_stubs = None

                title = 'Residual diagnostics:'
                table = SimpleTable(
                    params_data, params_header, params_stubs,
                    txt_fmt=fmt_params, title=title)
                summary.tables.insert(table_ix, table)

        # Add warnings/notes, added to text format only
        etext = []
        if hasattr(self, 'cov_type') and 'description' in self.cov_kwds:
            etext.append(self.cov_kwds['description'])
        if self._rank < (len(self.params) - len(self.fixed_params)):
            cov_params = self.cov_params()
            if len(self.fixed_params) > 0:
                mask = np.ix_(self._free_params_index, self._free_params_index)
                cov_params = cov_params[mask]
            etext.append("Covariance matrix is singular or near-singular,"
                         " with condition number %6.3g. Standard errors may be"
                         " unstable." % _safe_cond(cov_params))

        if etext:
            etext = ["[{0}] {1}".format(i + 1, text)
                     for i, text in enumerate(etext)]
            etext.insert(0, "Warnings:")
            summary.add_extra_txt(etext)

        return summary


class MLEResultsWrapper(wrap.ResultsWrapper):
    _attrs = {
        'zvalues': 'columns',
        'cov_params_approx': 'cov',
        'cov_params_default': 'cov',
        'cov_params_oim': 'cov',
        'cov_params_opg': 'cov',
        'cov_params_robust': 'cov',
        'cov_params_robust_approx': 'cov',
        'cov_params_robust_oim': 'cov',
    }
    _wrap_attrs = wrap.union_dicts(tsbase.TimeSeriesResultsWrapper._wrap_attrs,
                                   _attrs)
    _methods = {
        'forecast': 'dates',
        'impulse_responses': 'ynames'
    }
    _wrap_methods = wrap.union_dicts(
        tsbase.TimeSeriesResultsWrapper._wrap_methods, _methods)
wrap.populate_wrapper(MLEResultsWrapper, MLEResults)  # noqa:E305


class PredictionResults(pred.PredictionResults):
    """
    Prediction result from MLE models

    Parameters
    ----------
    model : MLEModel
        The models used to make the prediction
    prediction_results : kalman_filter.PredictionResults instance
        Results object from prediction after fitting or filtering a state space
        model.
    row_labels : iterable
        Row labels for the predicted data.
    information_set : str
        Name of information set
    signal_only : bool
        Whether the prediction is for the signal only

    Attributes
    ----------
    model : MLEModel
        The models used to make the prediction
    prediction_results : kalman_filter.PredictionResults instance
        Results object from prediction after fitting or filtering a state space
        model.
    information_set : str
        Name of information set
    signal_only : bool
        Whether the prediction is for the signal only
    """
    def __init__(self, model, prediction_results, row_labels=None,
                 information_set='predicted', signal_only=False):
        if model.model.k_endog == 1:
            endog = pd.Series(prediction_results.endog[0],
                              name=model.model.endog_names)
        else:
            endog = pd.DataFrame(prediction_results.endog.T,
                                 columns=model.model.endog_names)
        self.model = Bunch(data=model.data.__class__(
            endog=endog, predict_dates=row_labels))
        self.prediction_results = prediction_results

        self.information_set = information_set
        self.signal_only = signal_only

        # Get required values
        k_endog, nobs = prediction_results.endog.shape
        res = self.prediction_results.results
        if information_set == 'predicted' and not res.memory_no_forecast_mean:
            if not signal_only:
                predicted_mean = self.prediction_results.forecasts
            else:
                predicted_mean = self.prediction_results.predicted_signal
        elif information_set == 'filtered' and not res.memory_no_filtered_mean:
            if not signal_only:
                predicted_mean = self.prediction_results.filtered_forecasts
            else:
                predicted_mean = self.prediction_results.filtered_signal
        elif information_set == 'smoothed':
            if not signal_only:
                predicted_mean = self.prediction_results.smoothed_forecasts
            else:
                predicted_mean = self.prediction_results.smoothed_signal
        else:
            predicted_mean = np.zeros((k_endog, nobs)) * np.nan

        if predicted_mean.shape[0] == 1:
            predicted_mean = predicted_mean[0, :]
        else:
            predicted_mean = predicted_mean.transpose()

        if information_set == 'predicted' and not res.memory_no_forecast_cov:
            if not signal_only:
                var_pred_mean = self.prediction_results.forecasts_error_cov
            else:
                var_pred_mean = self.prediction_results.predicted_signal_cov
        elif information_set == 'filtered' and not res.memory_no_filtered_mean:
            if not signal_only:
                var_pred_mean = (
                    self.prediction_results.filtered_forecasts_error_cov)
            else:
                var_pred_mean = self.prediction_results.filtered_signal_cov
        elif information_set == 'smoothed':
            if not signal_only:
                var_pred_mean = (
                    self.prediction_results.smoothed_forecasts_error_cov)
            else:
                var_pred_mean = self.prediction_results.smoothed_signal_cov
        else:
            var_pred_mean = np.zeros((k_endog, k_endog, nobs)) * np.nan

        if var_pred_mean.shape[0] == 1:
            var_pred_mean = var_pred_mean[0, 0, :]
        else:
            var_pred_mean = var_pred_mean.transpose()

        # Initialize
        super(PredictionResults, self).__init__(predicted_mean, var_pred_mean,
                                                dist='norm',
                                                row_labels=row_labels)

    @property
    def se_mean(self):
        # Replace negative values with np.nan to avoid a RuntimeWarning
        var_pred_mean = self.var_pred_mean.copy()
        var_pred_mean[var_pred_mean < 0] = np.nan
        if var_pred_mean.ndim == 1:
            se_mean = np.sqrt(var_pred_mean)
        else:
            se_mean = np.sqrt(var_pred_mean.T.diagonal())
        return se_mean

    def conf_int(self, method='endpoint', alpha=0.05, **kwds):
        # TODO: this performs metadata wrapping, and that should be handled
        #       by attach_* methods. However, they do not currently support
        #       this use case.
        _use_pandas = self._use_pandas
        self._use_pandas = False
        conf_int = super(PredictionResults, self).conf_int(alpha, **kwds)
        self._use_pandas = _use_pandas

        # Create a dataframe
        if self._row_labels is not None:
            conf_int = pd.DataFrame(conf_int, index=self.row_labels)

            # Attach the endog names
            ynames = self.model.data.ynames
            if not type(ynames) == list:
                ynames = [ynames]
            names = (['lower {0}'.format(name) for name in ynames] +
                     ['upper {0}'.format(name) for name in ynames])
            conf_int.columns = names

        return conf_int

    def summary_frame(self, endog=0, alpha=0.05):
        # TODO: finish and cleanup
        # import pandas as pd
        # ci_obs = self.conf_int(alpha=alpha, obs=True) # need to split
        ci_mean = np.asarray(self.conf_int(alpha=alpha))
        _use_pandas = self._use_pandas
        self._use_pandas = False
        to_include = {}
        if self.predicted_mean.ndim == 1:
            yname = self.model.data.ynames
            to_include['mean'] = self.predicted_mean
            to_include['mean_se'] = self.se_mean
            k_endog = 1
        else:
            yname = self.model.data.ynames[endog]
            to_include['mean'] = self.predicted_mean[:, endog]
            to_include['mean_se'] = self.se_mean[:, endog]
            k_endog = self.predicted_mean.shape[1]
        self._use_pandas = _use_pandas
        to_include['mean_ci_lower'] = ci_mean[:, endog]
        to_include['mean_ci_upper'] = ci_mean[:, k_endog + endog]

        # pandas dict does not handle 2d_array
        # data = np.column_stack(list(to_include.values()))
        # names = ....
        res = pd.DataFrame(to_include, index=self._row_labels,
                           columns=list(to_include.keys()))
        res.columns.name = yname
        return res


class PredictionResultsWrapper(wrap.ResultsWrapper):
    _attrs = {
        'predicted_mean': 'dates',
        'se_mean': 'dates',
        't_values': 'dates',
    }
    _wrap_attrs = wrap.union_dicts(_attrs)

    _methods = {}
    _wrap_methods = wrap.union_dicts(_methods)
wrap.populate_wrapper(PredictionResultsWrapper, PredictionResults)  # noqa:E305
