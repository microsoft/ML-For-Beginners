"""
State Space Representation and Kalman Filter

Author: Chad Fulton
License: Simplified-BSD
"""

import contextlib
from warnings import warn

import numpy as np
from .representation import OptionWrapper, Representation, FrozenRepresentation
from .tools import reorder_missing_matrix, reorder_missing_vector
from . import tools
from statsmodels.tools.sm_exceptions import ValueWarning

# Define constants
FILTER_CONVENTIONAL = 0x01     # Durbin and Koopman (2012), Chapter 4
FILTER_EXACT_INITIAL = 0x02    # ibid., Chapter 5.6
FILTER_AUGMENTED = 0x04        # ibid., Chapter 5.7
FILTER_SQUARE_ROOT = 0x08      # ibid., Chapter 6.3
FILTER_UNIVARIATE = 0x10       # ibid., Chapter 6.4
FILTER_COLLAPSED = 0x20        # ibid., Chapter 6.5
FILTER_EXTENDED = 0x40         # ibid., Chapter 10.2
FILTER_UNSCENTED = 0x80        # ibid., Chapter 10.3
FILTER_CONCENTRATED = 0x100    # Harvey (1989), Chapter 3.4
FILTER_CHANDRASEKHAR = 0x200   # Herbst (2015)

INVERT_UNIVARIATE = 0x01
SOLVE_LU = 0x02
INVERT_LU = 0x04
SOLVE_CHOLESKY = 0x08
INVERT_CHOLESKY = 0x10

STABILITY_FORCE_SYMMETRY = 0x01

MEMORY_STORE_ALL = 0
MEMORY_NO_FORECAST_MEAN = 0x01
MEMORY_NO_FORECAST_COV = 0x02
MEMORY_NO_FORECAST = MEMORY_NO_FORECAST_MEAN | MEMORY_NO_FORECAST_COV
MEMORY_NO_PREDICTED_MEAN = 0x04
MEMORY_NO_PREDICTED_COV = 0x08
MEMORY_NO_PREDICTED = MEMORY_NO_PREDICTED_MEAN | MEMORY_NO_PREDICTED_COV
MEMORY_NO_FILTERED_MEAN = 0x10
MEMORY_NO_FILTERED_COV = 0x20
MEMORY_NO_FILTERED = MEMORY_NO_FILTERED_MEAN | MEMORY_NO_FILTERED_COV
MEMORY_NO_LIKELIHOOD = 0x40
MEMORY_NO_GAIN = 0x80
MEMORY_NO_SMOOTHING = 0x100
MEMORY_NO_STD_FORECAST = 0x200
MEMORY_CONSERVE = (
    MEMORY_NO_FORECAST_COV | MEMORY_NO_PREDICTED | MEMORY_NO_FILTERED |
    MEMORY_NO_LIKELIHOOD | MEMORY_NO_GAIN | MEMORY_NO_SMOOTHING
)

TIMING_INIT_PREDICTED = 0
TIMING_INIT_FILTERED = 1


class KalmanFilter(Representation):
    r"""
    State space representation of a time series process, with Kalman filter

    Parameters
    ----------
    k_endog : {array_like, int}
        The observed time-series process :math:`y` if array like or the
        number of variables in the process if an integer.
    k_states : int
        The dimension of the unobserved state process.
    k_posdef : int, optional
        The dimension of a guaranteed positive definite covariance matrix
        describing the shocks in the transition equation. Must be less than
        or equal to `k_states`. Default is `k_states`.
    loglikelihood_burn : int, optional
        The number of initial periods during which the loglikelihood is not
        recorded. Default is 0.
    tolerance : float, optional
        The tolerance at which the Kalman filter determines convergence to
        steady-state. Default is 1e-19.
    results_class : class, optional
        Default results class to use to save filtering output. Default is
        `FilterResults`. If specified, class must extend from `FilterResults`.
    **kwargs
        Keyword arguments may be used to provide values for the filter,
        inversion, and stability methods. See `set_filter_method`,
        `set_inversion_method`, and `set_stability_method`.
        Keyword arguments may be used to provide default values for state space
        matrices. See `Representation` for more details.

    See Also
    --------
    FilterResults
    statsmodels.tsa.statespace.representation.Representation

    Notes
    -----
    There are several types of options available for controlling the Kalman
    filter operation. All options are internally held as bitmasks, but can be
    manipulated by setting class attributes, which act like boolean flags. For
    more information, see the `set_*` class method documentation. The options
    are:

    filter_method
        The filtering method controls aspects of which
        Kalman filtering approach will be used.
    inversion_method
        The Kalman filter may contain one matrix inversion: that of the
        forecast error covariance matrix. The inversion method controls how and
        if that inverse is performed.
    stability_method
        The Kalman filter is a recursive algorithm that may in some cases
        suffer issues with numerical stability. The stability method controls
        what, if any, measures are taken to promote stability.
    conserve_memory
        By default, the Kalman filter computes a number of intermediate
        matrices at each iteration. The memory conservation options control
        which of those matrices are stored.
    filter_timing
        By default, the Kalman filter follows Durbin and Koopman, 2012, in
        initializing the filter with predicted values. Kim and Nelson, 1999,
        instead initialize the filter with filtered values, which is
        essentially just a different timing convention.

    The `filter_method` and `inversion_method` options intentionally allow
    the possibility that multiple methods will be indicated. In the case that
    multiple methods are selected, the underlying Kalman filter will attempt to
    select the optional method given the input data.

    For example, it may be that INVERT_UNIVARIATE and SOLVE_CHOLESKY are
    indicated (this is in fact the default case). In this case, if the
    endogenous vector is 1-dimensional (`k_endog` = 1), then INVERT_UNIVARIATE
    is used and inversion reduces to simple division, and if it has a larger
    dimension, the Cholesky decomposition along with linear solving (rather
    than explicit matrix inversion) is used. If only SOLVE_CHOLESKY had been
    set, then the Cholesky decomposition method would *always* be used, even in
    the case of 1-dimensional data.
    """

    filter_methods = [
        'filter_conventional', 'filter_exact_initial', 'filter_augmented',
        'filter_square_root', 'filter_univariate', 'filter_collapsed',
        'filter_extended', 'filter_unscented', 'filter_concentrated',
        'filter_chandrasekhar'
    ]

    filter_conventional = OptionWrapper('filter_method', FILTER_CONVENTIONAL)
    """
    (bool) Flag for conventional Kalman filtering.
    """
    filter_exact_initial = OptionWrapper('filter_method', FILTER_EXACT_INITIAL)
    """
    (bool) Flag for exact initial Kalman filtering. Not implemented.
    """
    filter_augmented = OptionWrapper('filter_method', FILTER_AUGMENTED)
    """
    (bool) Flag for augmented Kalman filtering. Not implemented.
    """
    filter_square_root = OptionWrapper('filter_method', FILTER_SQUARE_ROOT)
    """
    (bool) Flag for square-root Kalman filtering. Not implemented.
    """
    filter_univariate = OptionWrapper('filter_method', FILTER_UNIVARIATE)
    """
    (bool) Flag for univariate filtering of multivariate observation vector.
    """
    filter_collapsed = OptionWrapper('filter_method', FILTER_COLLAPSED)
    """
    (bool) Flag for Kalman filtering with collapsed observation vector.
    """
    filter_extended = OptionWrapper('filter_method', FILTER_EXTENDED)
    """
    (bool) Flag for extended Kalman filtering. Not implemented.
    """
    filter_unscented = OptionWrapper('filter_method', FILTER_UNSCENTED)
    """
    (bool) Flag for unscented Kalman filtering. Not implemented.
    """
    filter_concentrated = OptionWrapper('filter_method', FILTER_CONCENTRATED)
    """
    (bool) Flag for Kalman filtering with concentrated log-likelihood.
    """
    filter_chandrasekhar = OptionWrapper('filter_method', FILTER_CHANDRASEKHAR)
    """
    (bool) Flag for filtering with Chandrasekhar recursions.
    """

    inversion_methods = [
        'invert_univariate', 'solve_lu', 'invert_lu', 'solve_cholesky',
        'invert_cholesky'
    ]

    invert_univariate = OptionWrapper('inversion_method', INVERT_UNIVARIATE)
    """
    (bool) Flag for univariate inversion method (recommended).
    """
    solve_lu = OptionWrapper('inversion_method', SOLVE_LU)
    """
    (bool) Flag for LU and linear solver inversion method.
    """
    invert_lu = OptionWrapper('inversion_method', INVERT_LU)
    """
    (bool) Flag for LU inversion method.
    """
    solve_cholesky = OptionWrapper('inversion_method', SOLVE_CHOLESKY)
    """
    (bool) Flag for Cholesky and linear solver inversion method (recommended).
    """
    invert_cholesky = OptionWrapper('inversion_method', INVERT_CHOLESKY)
    """
    (bool) Flag for Cholesky inversion method.
    """

    stability_methods = ['stability_force_symmetry']

    stability_force_symmetry = (
        OptionWrapper('stability_method', STABILITY_FORCE_SYMMETRY)
    )
    """
    (bool) Flag for enforcing covariance matrix symmetry
    """

    memory_options = [
        'memory_store_all', 'memory_no_forecast_mean',
        'memory_no_forecast_cov', 'memory_no_forecast',
        'memory_no_predicted_mean', 'memory_no_predicted_cov',
        'memory_no_predicted', 'memory_no_filtered_mean',
        'memory_no_filtered_cov', 'memory_no_filtered',
        'memory_no_likelihood', 'memory_no_gain',
        'memory_no_smoothing', 'memory_no_std_forecast', 'memory_conserve'
    ]

    memory_store_all = OptionWrapper('conserve_memory', MEMORY_STORE_ALL)
    """
    (bool) Flag for storing all intermediate results in memory (default).
    """
    memory_no_forecast_mean = OptionWrapper(
        'conserve_memory', MEMORY_NO_FORECAST_MEAN)
    """
    (bool) Flag to prevent storing forecasts and forecast errors.
    """
    memory_no_forecast_cov = OptionWrapper(
        'conserve_memory', MEMORY_NO_FORECAST_COV)
    """
    (bool) Flag to prevent storing forecast error covariance matrices.
    """
    @property
    def memory_no_forecast(self):
        """
        (bool) Flag to prevent storing all forecast-related output.
        """
        return self.memory_no_forecast_mean or self.memory_no_forecast_cov

    @memory_no_forecast.setter
    def memory_no_forecast(self, value):
        if bool(value):
            self.memory_no_forecast_mean = True
            self.memory_no_forecast_cov = True
        else:
            self.memory_no_forecast_mean = False
            self.memory_no_forecast_cov = False

    memory_no_predicted_mean = OptionWrapper(
        'conserve_memory', MEMORY_NO_PREDICTED_MEAN)
    """
    (bool) Flag to prevent storing predicted states.
    """
    memory_no_predicted_cov = OptionWrapper(
        'conserve_memory', MEMORY_NO_PREDICTED_COV)
    """
    (bool) Flag to prevent storing predicted state covariance matrices.
    """
    @property
    def memory_no_predicted(self):
        """
        (bool) Flag to prevent storing predicted state and covariance matrices.
        """
        return self.memory_no_predicted_mean or self.memory_no_predicted_cov

    @memory_no_predicted.setter
    def memory_no_predicted(self, value):
        if bool(value):
            self.memory_no_predicted_mean = True
            self.memory_no_predicted_cov = True
        else:
            self.memory_no_predicted_mean = False
            self.memory_no_predicted_cov = False

    memory_no_filtered_mean = OptionWrapper(
        'conserve_memory', MEMORY_NO_FILTERED_MEAN)
    """
    (bool) Flag to prevent storing filtered states.
    """
    memory_no_filtered_cov = OptionWrapper(
        'conserve_memory', MEMORY_NO_FILTERED_COV)
    """
    (bool) Flag to prevent storing filtered state covariance matrices.
    """
    @property
    def memory_no_filtered(self):
        """
        (bool) Flag to prevent storing filtered state and covariance matrices.
        """
        return self.memory_no_filtered_mean or self.memory_no_filtered_cov

    @memory_no_filtered.setter
    def memory_no_filtered(self, value):
        if bool(value):
            self.memory_no_filtered_mean = True
            self.memory_no_filtered_cov = True
        else:
            self.memory_no_filtered_mean = False
            self.memory_no_filtered_cov = False

    memory_no_likelihood = (
        OptionWrapper('conserve_memory', MEMORY_NO_LIKELIHOOD)
    )
    """
    (bool) Flag to prevent storing likelihood values for each observation.
    """
    memory_no_gain = OptionWrapper('conserve_memory', MEMORY_NO_GAIN)
    """
    (bool) Flag to prevent storing the Kalman gain matrices.
    """
    memory_no_smoothing = OptionWrapper('conserve_memory', MEMORY_NO_SMOOTHING)
    """
    (bool) Flag to prevent storing likelihood values for each observation.
    """
    memory_no_std_forecast = (
        OptionWrapper('conserve_memory', MEMORY_NO_STD_FORECAST))
    """
    (bool) Flag to prevent storing standardized forecast errors.
    """
    memory_conserve = OptionWrapper('conserve_memory', MEMORY_CONSERVE)
    """
    (bool) Flag to conserve the maximum amount of memory.
    """

    timing_options = [
        'timing_init_predicted', 'timing_init_filtered'
    ]
    timing_init_predicted = OptionWrapper('filter_timing',
                                          TIMING_INIT_PREDICTED)
    """
    (bool) Flag for the default timing convention (Durbin and Koopman, 2012).
    """
    timing_init_filtered = OptionWrapper('filter_timing', TIMING_INIT_FILTERED)
    """
    (bool) Flag for the alternate timing convention (Kim and Nelson, 2012).
    """

    # Default filter options
    filter_method = FILTER_CONVENTIONAL
    """
    (int) Filtering method bitmask.
    """
    inversion_method = INVERT_UNIVARIATE | SOLVE_CHOLESKY
    """
    (int) Inversion method bitmask.
    """
    stability_method = STABILITY_FORCE_SYMMETRY
    """
    (int) Stability method bitmask.
    """
    conserve_memory = MEMORY_STORE_ALL
    """
    (int) Memory conservation bitmask.
    """
    filter_timing = TIMING_INIT_PREDICTED
    """
    (int) Filter timing.
    """

    def __init__(self, k_endog, k_states, k_posdef=None,
                 loglikelihood_burn=0, tolerance=1e-19, results_class=None,
                 kalman_filter_classes=None, **kwargs):
        # Extract keyword arguments to-be-used later
        keys = ['filter_method'] + KalmanFilter.filter_methods
        filter_method_kwargs = {key: kwargs.pop(key) for key in keys
                                if key in kwargs}
        keys = ['inversion_method'] + KalmanFilter.inversion_methods
        inversion_method_kwargs = {key: kwargs.pop(key) for key in keys
                                   if key in kwargs}
        keys = ['stability_method'] + KalmanFilter.stability_methods
        stability_method_kwargs = {key: kwargs.pop(key) for key in keys
                                   if key in kwargs}
        keys = ['conserve_memory'] + KalmanFilter.memory_options
        conserve_memory_kwargs = {key: kwargs.pop(key) for key in keys
                                  if key in kwargs}
        keys = ['alternate_timing'] + KalmanFilter.timing_options
        filter_timing_kwargs = {key: kwargs.pop(key) for key in keys
                                if key in kwargs}

        # Initialize the base class
        super(KalmanFilter, self).__init__(
            k_endog, k_states, k_posdef, **kwargs
        )

        # Setup the underlying Kalman filter storage
        self._kalman_filters = {}

        # Filter options
        self.loglikelihood_burn = loglikelihood_burn
        self.results_class = (
            results_class if results_class is not None else FilterResults
        )
        # Options
        self.prefix_kalman_filter_map = (
            kalman_filter_classes
            if kalman_filter_classes is not None
            else tools.prefix_kalman_filter_map.copy())

        self.set_filter_method(**filter_method_kwargs)
        self.set_inversion_method(**inversion_method_kwargs)
        self.set_stability_method(**stability_method_kwargs)
        self.set_conserve_memory(**conserve_memory_kwargs)
        self.set_filter_timing(**filter_timing_kwargs)

        self.tolerance = tolerance

        # Internal flags
        # The _scale internal flag is used because we may want to
        # use a fixed scale, in which case we want the flag to the Cython
        # Kalman filter to indicate that the scale should not be concentrated
        # out, so that self.filter_concentrated = False, but we still want to
        # alert the results object that we are viewing the model as one in
        # which the scale had been concentrated out for e.g. degree of freedom
        # computations.
        # This value should always be None, except within the fixed_scale
        # context, and should not be modified by users or anywhere else.
        self._scale = None

    def _clone_kwargs(self, endog, **kwargs):
        # See Representation._clone_kwargs for docstring
        kwargs = super(KalmanFilter, self)._clone_kwargs(endog, **kwargs)

        # Get defaults for options
        kwargs.setdefault('filter_method', self.filter_method)
        kwargs.setdefault('inversion_method', self.inversion_method)
        kwargs.setdefault('stability_method', self.stability_method)
        kwargs.setdefault('conserve_memory', self.conserve_memory)
        kwargs.setdefault('alternate_timing', bool(self.filter_timing))
        kwargs.setdefault('tolerance', self.tolerance)
        kwargs.setdefault('loglikelihood_burn', self.loglikelihood_burn)

        return kwargs

    @property
    def _kalman_filter(self):
        prefix = self.prefix
        if prefix in self._kalman_filters:
            return self._kalman_filters[prefix]
        return None

    def _initialize_filter(self, filter_method=None, inversion_method=None,
                           stability_method=None, conserve_memory=None,
                           tolerance=None, filter_timing=None,
                           loglikelihood_burn=None):
        if filter_method is None:
            filter_method = self.filter_method
        if inversion_method is None:
            inversion_method = self.inversion_method
        if stability_method is None:
            stability_method = self.stability_method
        if conserve_memory is None:
            conserve_memory = self.conserve_memory
        if loglikelihood_burn is None:
            loglikelihood_burn = self.loglikelihood_burn
        if filter_timing is None:
            filter_timing = self.filter_timing
        if tolerance is None:
            tolerance = self.tolerance

        # Make sure we have endog
        if self.endog is None:
            raise RuntimeError('Must bind a dataset to the model before'
                               ' filtering or smoothing.')

        # Initialize the representation matrices
        prefix, dtype, create_statespace = self._initialize_representation()

        # Determine if we need to (re-)create the filter
        # (definitely need to recreate if we recreated the _statespace object)
        create_filter = create_statespace or prefix not in self._kalman_filters
        if not create_filter:
            kalman_filter = self._kalman_filters[prefix]

            create_filter = (
                not kalman_filter.conserve_memory == conserve_memory or
                not kalman_filter.loglikelihood_burn == loglikelihood_burn
            )

        # If the dtype-specific _kalman_filter does not exist (or if we need
        # to re-create it), create it
        if create_filter:
            if prefix in self._kalman_filters:
                # Delete the old filter
                del self._kalman_filters[prefix]
            # Setup the filter
            cls = self.prefix_kalman_filter_map[prefix]
            self._kalman_filters[prefix] = cls(
                self._statespaces[prefix], filter_method, inversion_method,
                stability_method, conserve_memory, filter_timing, tolerance,
                loglikelihood_burn
            )
        # Otherwise, update the filter parameters
        else:
            kalman_filter = self._kalman_filters[prefix]
            kalman_filter.set_filter_method(filter_method, False)
            kalman_filter.inversion_method = inversion_method
            kalman_filter.stability_method = stability_method
            kalman_filter.filter_timing = filter_timing
            kalman_filter.tolerance = tolerance
            # conserve_memory and loglikelihood_burn changes always lead to
            # re-created filters

        return prefix, dtype, create_filter, create_statespace

    def set_filter_method(self, filter_method=None, **kwargs):
        r"""
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
        The filtering method is defined by a collection of boolean flags, and
        is internally stored as a bitmask. The methods available are:

        FILTER_CONVENTIONAL
            Conventional Kalman filter.
        FILTER_UNIVARIATE
            Univariate approach to Kalman filtering. Overrides conventional
            method if both are specified.
        FILTER_COLLAPSED
            Collapsed approach to Kalman filtering. Will be used *in addition*
            to conventional or univariate filtering.
        FILTER_CONCENTRATED
            Use the concentrated log-likelihood function. Will be used
            *in addition* to the other options.

        Note that only the first method is available if using a Scipy version
        older than 0.16.

        If the bitmask is set directly via the `filter_method` argument, then
        the full method must be provided.

        If keyword arguments are used to set individual boolean flags, then
        the lowercase of the method must be used as an argument name, and the
        value is the desired value of the boolean flag (True or False).

        Note that the filter method may also be specified by directly modifying
        the class attributes which are defined similarly to the keyword
        arguments.

        The default filtering method is FILTER_CONVENTIONAL.

        Examples
        --------
        >>> mod = sm.tsa.statespace.SARIMAX(range(10))
        >>> mod.ssm.filter_method
        1
        >>> mod.ssm.filter_conventional
        True
        >>> mod.ssm.filter_univariate = True
        >>> mod.ssm.filter_method
        17
        >>> mod.ssm.set_filter_method(filter_univariate=False,
        ...                           filter_collapsed=True)
        >>> mod.ssm.filter_method
        33
        >>> mod.ssm.set_filter_method(filter_method=1)
        >>> mod.ssm.filter_conventional
        True
        >>> mod.ssm.filter_univariate
        False
        >>> mod.ssm.filter_collapsed
        False
        >>> mod.ssm.filter_univariate = True
        >>> mod.ssm.filter_method
        17
        """
        if filter_method is not None:
            self.filter_method = filter_method
        for name in KalmanFilter.filter_methods:
            if name in kwargs:
                setattr(self, name, kwargs[name])

    def set_inversion_method(self, inversion_method=None, **kwargs):
        r"""
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
        The inversion method is defined by a collection of boolean flags, and
        is internally stored as a bitmask. The methods available are:

        INVERT_UNIVARIATE
            If the endogenous time series is univariate, then inversion can be
            performed by simple division. If this flag is set and the time
            series is univariate, then division will always be used even if
            other flags are also set.
        SOLVE_LU
            Use an LU decomposition along with a linear solver (rather than
            ever actually inverting the matrix).
        INVERT_LU
            Use an LU decomposition along with typical matrix inversion.
        SOLVE_CHOLESKY
            Use a Cholesky decomposition along with a linear solver.
        INVERT_CHOLESKY
            Use an Cholesky decomposition along with typical matrix inversion.

        If the bitmask is set directly via the `inversion_method` argument,
        then the full method must be provided.

        If keyword arguments are used to set individual boolean flags, then
        the lowercase of the method must be used as an argument name, and the
        value is the desired value of the boolean flag (True or False).

        Note that the inversion method may also be specified by directly
        modifying the class attributes which are defined similarly to the
        keyword arguments.

        The default inversion method is `INVERT_UNIVARIATE | SOLVE_CHOLESKY`

        Several things to keep in mind are:

        - If the filtering method is specified to be univariate, then simple
          division is always used regardless of the dimension of the endogenous
          time series.
        - Cholesky decomposition is about twice as fast as LU decomposition,
          but it requires that the matrix be positive definite. While this
          should generally be true, it may not be in every case.
        - Using a linear solver rather than true matrix inversion is generally
          faster and is numerically more stable.

        Examples
        --------
        >>> mod = sm.tsa.statespace.SARIMAX(range(10))
        >>> mod.ssm.inversion_method
        1
        >>> mod.ssm.solve_cholesky
        True
        >>> mod.ssm.invert_univariate
        True
        >>> mod.ssm.invert_lu
        False
        >>> mod.ssm.invert_univariate = False
        >>> mod.ssm.inversion_method
        8
        >>> mod.ssm.set_inversion_method(solve_cholesky=False,
        ...                              invert_cholesky=True)
        >>> mod.ssm.inversion_method
        16
        """
        if inversion_method is not None:
            self.inversion_method = inversion_method
        for name in KalmanFilter.inversion_methods:
            if name in kwargs:
                setattr(self, name, kwargs[name])

    def set_stability_method(self, stability_method=None, **kwargs):
        r"""
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
        The stability method is defined by a collection of boolean flags, and
        is internally stored as a bitmask. The methods available are:

        STABILITY_FORCE_SYMMETRY = 0x01
            If this flag is set, symmetry of the predicted state covariance
            matrix is enforced at each iteration of the filter, where each
            element is set to the average of the corresponding elements in the
            upper and lower triangle.

        If the bitmask is set directly via the `stability_method` argument,
        then the full method must be provided.

        If keyword arguments are used to set individual boolean flags, then
        the lowercase of the method must be used as an argument name, and the
        value is the desired value of the boolean flag (True or False).

        Note that the stability method may also be specified by directly
        modifying the class attributes which are defined similarly to the
        keyword arguments.

        The default stability method is `STABILITY_FORCE_SYMMETRY`

        Examples
        --------
        >>> mod = sm.tsa.statespace.SARIMAX(range(10))
        >>> mod.ssm.stability_method
        1
        >>> mod.ssm.stability_force_symmetry
        True
        >>> mod.ssm.stability_force_symmetry = False
        >>> mod.ssm.stability_method
        0
        """
        if stability_method is not None:
            self.stability_method = stability_method
        for name in KalmanFilter.stability_methods:
            if name in kwargs:
                setattr(self, name, kwargs[name])

    def set_conserve_memory(self, conserve_memory=None, **kwargs):
        r"""
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
            method by setting individual boolean flags. See notes for details.

        Notes
        -----
        The memory conservation method is defined by a collection of boolean
        flags, and is internally stored as a bitmask. The methods available
        are:

        MEMORY_STORE_ALL
            Store all intermediate matrices. This is the default value.
        MEMORY_NO_FORECAST_MEAN
            Do not store the forecast or forecast errors. If this option is
            used, the `predict` method from the results class is unavailable.
        MEMORY_NO_FORECAST_COV
            Do not store the forecast error covariance matrices.
        MEMORY_NO_FORECAST
            Do not store the forecast, forecast error, or forecast error
            covariance matrices. If this option is used, the `predict` method
            from the results class is unavailable.
        MEMORY_NO_PREDICTED_MEAN
            Do not store the predicted state.
        MEMORY_NO_PREDICTED_COV
            Do not store the predicted state covariance
            matrices.
        MEMORY_NO_PREDICTED
            Do not store the predicted state or predicted state covariance
            matrices.
        MEMORY_NO_FILTERED_MEAN
            Do not store the filtered state.
        MEMORY_NO_FILTERED_COV
            Do not store the filtered state covariance
            matrices.
        MEMORY_NO_FILTERED
            Do not store the filtered state or filtered state covariance
            matrices.
        MEMORY_NO_LIKELIHOOD
            Do not store the vector of loglikelihood values for each
            observation. Only the sum of the loglikelihood values is stored.
        MEMORY_NO_GAIN
            Do not store the Kalman gain matrices.
        MEMORY_NO_SMOOTHING
            Do not store temporary variables related to Kalman smoothing. If
            this option is used, smoothing is unavailable.
        MEMORY_NO_STD_FORECAST
            Do not store standardized forecast errors.
        MEMORY_CONSERVE
            Do not store any intermediate matrices.

        If the bitmask is set directly via the `conserve_memory` argument,
        then the full method must be provided.

        If keyword arguments are used to set individual boolean flags, then
        the lowercase of the method must be used as an argument name, and the
        value is the desired value of the boolean flag (True or False).

        Note that the memory conservation method may also be specified by
        directly modifying the class attributes which are defined similarly to
        the keyword arguments.

        The default memory conservation method is `MEMORY_STORE_ALL`, so that
        all intermediate matrices are stored.

        Examples
        --------
        >>> mod = sm.tsa.statespace.SARIMAX(range(10))
        >>> mod.ssm..conserve_memory
        0
        >>> mod.ssm.memory_no_predicted
        False
        >>> mod.ssm.memory_no_predicted = True
        >>> mod.ssm.conserve_memory
        2
        >>> mod.ssm.set_conserve_memory(memory_no_filtered=True,
        ...                             memory_no_forecast=True)
        >>> mod.ssm.conserve_memory
        7
        """
        if conserve_memory is not None:
            self.conserve_memory = conserve_memory
        for name in KalmanFilter.memory_options:
            if name in kwargs:
                setattr(self, name, kwargs[name])

    def set_filter_timing(self, alternate_timing=None, **kwargs):
        r"""
        Set the filter timing convention

        By default, the Kalman filter follows Durbin and Koopman, 2012, in
        initializing the filter with predicted values. Kim and Nelson, 1999,
        instead initialize the filter with filtered values, which is
        essentially just a different timing convention.

        Parameters
        ----------
        alternate_timing : int, optional
            Whether or not to use the alternate timing convention. Default is
            unspecified.
        **kwargs
            Keyword arguments may be used to influence the memory conservation
            method by setting individual boolean flags. See notes for details.
        """
        if alternate_timing is not None:
            self.filter_timing = int(alternate_timing)
        if 'timing_init_predicted' in kwargs:
            self.filter_timing = int(not kwargs['timing_init_predicted'])
        if 'timing_init_filtered' in kwargs:
            self.filter_timing = int(kwargs['timing_init_filtered'])

    @contextlib.contextmanager
    def fixed_scale(self, scale):
        """
        fixed_scale(scale)

        Context manager for fixing the scale when FILTER_CONCENTRATED is set

        Parameters
        ----------
        scale : numeric
            Scale of the model.

        Notes
        -----
        This a no-op if scale is None.

        This context manager is most useful in models which are explicitly
        concentrating out the scale, so that the set of parameters they are
        estimating does not include the scale.
        """
        # If a scale was provided, use it and do not concentrate it out of the
        # loglikelihood
        if scale is not None and scale != 1:
            if not self.filter_concentrated:
                raise ValueError('Cannot provide scale if filter method does'
                                 ' not include FILTER_CONCENTRATED.')
            self.filter_concentrated = False
            self._scale = scale
            obs_cov = self['obs_cov']
            state_cov = self['state_cov']
            self['obs_cov'] = scale * obs_cov
            self['state_cov'] = scale * state_cov
        try:
            yield
        finally:
            # If a scale was provided, reset the model
            if scale is not None and scale != 1:
                self['state_cov'] = state_cov
                self['obs_cov'] = obs_cov
                self.filter_concentrated = True
                self._scale = None

    def _filter(self, filter_method=None, inversion_method=None,
                stability_method=None, conserve_memory=None,
                filter_timing=None, tolerance=None, loglikelihood_burn=None,
                complex_step=False):
        # Initialize the filter
        prefix, dtype, create_filter, create_statespace = (
            self._initialize_filter(
                filter_method, inversion_method, stability_method,
                conserve_memory, filter_timing, tolerance, loglikelihood_burn
            )
        )
        kfilter = self._kalman_filters[prefix]

        # Initialize the state
        self._initialize_state(prefix=prefix, complex_step=complex_step)

        # Run the filter
        kfilter()

        return kfilter

    def filter(self, filter_method=None, inversion_method=None,
               stability_method=None, conserve_memory=None, filter_timing=None,
               tolerance=None, loglikelihood_burn=None, complex_step=False):
        r"""
        Apply the Kalman filter to the statespace model.

        Parameters
        ----------
        filter_method : int, optional
            Determines which Kalman filter to use. Default is conventional.
        inversion_method : int, optional
            Determines which inversion technique to use. Default is by Cholesky
            decomposition.
        stability_method : int, optional
            Determines which numerical stability techniques to use. Default is
            to enforce symmetry of the predicted state covariance matrix.
        conserve_memory : int, optional
            Determines what output from the filter to store. Default is to
            store everything.
        filter_timing : int, optional
            Determines the timing convention of the filter. Default is that
            from Durbin and Koopman (2012), in which the filter is initialized
            with predicted values.
        tolerance : float, optional
            The tolerance at which the Kalman filter determines convergence to
            steady-state. Default is 1e-19.
        loglikelihood_burn : int, optional
            The number of initial periods during which the loglikelihood is not
            recorded. Default is 0.

        Notes
        -----
        This function by default does not compute variables required for
        smoothing.
        """
        # Handle memory conservation
        if conserve_memory is None:
            conserve_memory = self.conserve_memory | MEMORY_NO_SMOOTHING
        conserve_memory_cache = self.conserve_memory
        self.set_conserve_memory(conserve_memory)

        # Run the filter
        kfilter = self._filter(
            filter_method, inversion_method, stability_method, conserve_memory,
            filter_timing, tolerance, loglikelihood_burn, complex_step)

        # Create the results object
        results = self.results_class(self)
        results.update_representation(self)
        results.update_filter(kfilter)

        # Resent memory conservation
        self.set_conserve_memory(conserve_memory_cache)

        return results

    def loglike(self, **kwargs):
        r"""
        Calculate the loglikelihood associated with the statespace model.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments to pass to the Kalman filter. See
            `KalmanFilter.filter` for more details.

        Returns
        -------
        loglike : float
            The joint loglikelihood.
        """
        kwargs.setdefault('conserve_memory',
                          MEMORY_CONSERVE ^ MEMORY_NO_LIKELIHOOD)
        kfilter = self._filter(**kwargs)
        loglikelihood_burn = kwargs.get('loglikelihood_burn',
                                        self.loglikelihood_burn)
        if not (kwargs['conserve_memory'] & MEMORY_NO_LIKELIHOOD):
            loglike = np.sum(kfilter.loglikelihood[loglikelihood_burn:])
        else:
            loglike = np.sum(kfilter.loglikelihood)

        # Need to modify the computed log-likelihood to incorporate the
        # MLE scale.
        if self.filter_method & FILTER_CONCENTRATED:
            d = max(loglikelihood_burn, kfilter.nobs_diffuse)
            nobs_k_endog = np.sum(
                self.k_endog -
                np.array(self._statespace.nmissing)[d:])

            # In the univariate case, we need to subtract observations
            # associated with a singular forecast error covariance matrix
            nobs_k_endog -= kfilter.nobs_kendog_univariate_singular

            if not (kwargs['conserve_memory'] & MEMORY_NO_LIKELIHOOD):
                scale = np.sum(kfilter.scale[d:]) / nobs_k_endog
            else:
                scale = kfilter.scale[0] / nobs_k_endog

            loglike += -0.5 * nobs_k_endog

            # Now need to modify this for diffuse initialization, since for
            # diffuse periods we only need to add in the scale value part if
            # the diffuse forecast error covariance matrix element was singular
            if kfilter.nobs_diffuse > 0:
                nobs_k_endog -= kfilter.nobs_kendog_diffuse_nonsingular

            loglike += -0.5 * nobs_k_endog * np.log(scale)
        return loglike

    def loglikeobs(self, **kwargs):
        r"""
        Calculate the loglikelihood for each observation associated with the
        statespace model.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments to pass to the Kalman filter. See
            `KalmanFilter.filter` for more details.

        Notes
        -----
        If `loglikelihood_burn` is positive, then the entries in the returned
        loglikelihood vector are set to be zero for those initial time periods.

        Returns
        -------
        loglike : array of float
            Array of loglikelihood values for each observation.
        """
        if self.memory_no_likelihood:
            raise RuntimeError('Cannot compute loglikelihood if'
                               ' MEMORY_NO_LIKELIHOOD option is selected.')
        if not self.filter_method & FILTER_CONCENTRATED:
            kwargs.setdefault('conserve_memory',
                              MEMORY_CONSERVE ^ MEMORY_NO_LIKELIHOOD)
        else:
            kwargs.setdefault(
                'conserve_memory',
                MEMORY_CONSERVE ^ (MEMORY_NO_FORECAST | MEMORY_NO_LIKELIHOOD))
        kfilter = self._filter(**kwargs)
        llf_obs = np.array(kfilter.loglikelihood, copy=True)
        loglikelihood_burn = kwargs.get('loglikelihood_burn',
                                        self.loglikelihood_burn)

        # If the scale was concentrated out of the log-likelihood function,
        # then the llf_obs above is:
        # -0.5 * k_endog * log 2 * pi - 0.5 * log |F_t|
        # and we need to add in the effect of the scale:
        # -0.5 * k_endog * log scale - 0.5 v' F_t^{-1} v / scale
        # and note that v' F_t^{-1} is in the _kalman_filter.scale array
        # Also note that we need to adjust the nobs and k_endog in both the
        # denominator of the scale computation and in the llf_obs adjustment
        # to take into account missing values.
        if self.filter_method & FILTER_CONCENTRATED:
            d = max(loglikelihood_burn, kfilter.nobs_diffuse)
            nmissing = np.array(self._statespace.nmissing)
            nobs_k_endog = np.sum(self.k_endog - nmissing[d:])

            # In the univariate case, we need to subtract observations
            # associated with a singular forecast error covariance matrix
            nobs_k_endog -= kfilter.nobs_kendog_univariate_singular

            scale = np.sum(kfilter.scale[d:]) / nobs_k_endog

            # Need to modify this for diffuse initialization, since for
            # diffuse periods we only need to add in the scale value if the
            # diffuse forecast error covariance matrix element was singular
            nsingular = 0
            if kfilter.nobs_diffuse > 0:
                d = kfilter.nobs_diffuse
                Finf = kfilter.forecast_error_diffuse_cov
                singular = np.diagonal(Finf).real <= kfilter.tolerance_diffuse
                nsingular = np.sum(~singular, axis=1)

            scale_obs = np.array(kfilter.scale, copy=True)
            llf_obs += -0.5 * (
                (self.k_endog - nmissing - nsingular) * np.log(scale) +
                scale_obs / scale)

        # Set any burned observations to have zero likelihood
        llf_obs[:loglikelihood_burn] = 0

        return llf_obs

    def simulate(self, nsimulations, measurement_shocks=None,
                 state_shocks=None, initial_state=None,
                 pretransformed_measurement_shocks=True,
                 pretransformed_state_shocks=True,
                 pretransformed_initial_state=True,
                 simulator=None, return_simulator=False,
                 random_state=None):
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
            If specified, this is the state vector at time zero, which should
            be shaped (`k_states` x 1), where `k_states` is the same as in the
            state space model. If unspecified, but the model has been
            initialized, then that initialization is used. If unspecified and
            the model has not been initialized, then a vector of zeros is used.
            Note that this is not included in the returned `simulated_states`
            array.
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
        return_simulator : bool, optional
            Whether or not to return the simulator object. Typically used to
            improve performance when performing repeated sampling. Default is
            False.
        random_state : {None, int, Generator, RandomState}, optionall
            If `seed` is None (or `np.random`), the `numpy.random.RandomState`
            singleton is used.
            If `seed` is an int, a new ``RandomState`` instance is used,
            seeded with `seed`.
            If `seed` is already a ``Generator`` or ``RandomState`` instance
            then that instance is used.

        Returns
        -------
        simulated_obs : ndarray
            An (nsimulations x k_endog) array of simulated observations.
        simulated_states : ndarray
            An (nsimulations x k_states) array of simulated states.
        simulator : SimulationSmoothResults
            If `return_simulator=True`, then an instance of a simulator is
            returned, which can be reused for additional simulations of the
            same size.
        """
        time_invariant = self.time_invariant
        # Check for valid number of simulations
        if not time_invariant and nsimulations > self.nobs:
            raise ValueError('In a time-varying model, cannot create more'
                             ' simulations than there are observations.')

        return self._simulate(
            nsimulations,
            measurement_disturbance_variates=measurement_shocks,
            state_disturbance_variates=state_shocks,
            initial_state_variates=initial_state,
            pretransformed_measurement_disturbance_variates=(
                pretransformed_measurement_shocks),
            pretransformed_state_disturbance_variates=(
                pretransformed_state_shocks),
            pretransformed_initial_state_variates=(
                pretransformed_initial_state),
            simulator=simulator, return_simulator=return_simulator,
            random_state=random_state)

    def _simulate(self, nsimulations, simulator=None, random_state=None,
                  **kwargs):
        raise NotImplementedError('Simulation only available through'
                                  ' the simulation smoother.')

    def impulse_responses(self, steps=10, impulse=0, orthogonalized=False,
                          cumulative=False, direct=False):
        r"""
        Impulse response function

        Parameters
        ----------
        steps : int, optional
            The number of steps for which impulse responses are calculated.
            Default is 10. Note that the initial impulse is not counted as a
            step, so if `steps=1`, the output will have 2 entries.
        impulse : int or array_like
            If an integer, the state innovation to pulse; must be between 0
            and `k_posdef-1` where `k_posdef` is the same as in the state
            space model. Alternatively, a custom impulse vector may be
            provided; must be a column vector with shape `(k_posdef, 1)`.
        orthogonalized : bool, optional
            Whether or not to perform impulse using orthogonalized innovations.
            Note that this will also affect custum `impulse` vectors. Default
            is False.
        cumulative : bool, optional
            Whether or not to return cumulative impulse responses. Default is
            False.

        Returns
        -------
        impulse_responses : ndarray
            Responses for each endogenous variable due to the impulse
            given by the `impulse` argument. A (steps + 1 x k_endog) array.

        Notes
        -----
        Intercepts in the measurement and state equation are ignored when
        calculating impulse responses.

        TODO: add note about how for time-varying systems this is - perhaps
        counter-intuitively - returning the impulse response within the given
        model (i.e. starting at period 0 defined by the model) and it is *not*
        doing impulse responses after the end of the model. To compute impulse
        responses from arbitrary time points, it is necessary to clone a new
        model with the appropriate system matrices.
        """
        # We need to add an additional step, since the first simulated value
        # will always be zeros (note that we take this value out at the end).
        steps += 1

        # For time-invariant models, add an additional `step`. This is the
        # default for time-invariant models based on the expected behavior for
        # ARIMA and VAR models: we want to record the initial impulse and also
        # `steps` values of the responses afterwards.
        if (self._design.shape[2] == 1 and self._transition.shape[2] == 1 and
                self._selection.shape[2] == 1):
            steps += 1

        # Check for what kind of impulse we want
        if type(impulse) == int:
            if impulse >= self.k_posdef or impulse < 0:
                raise ValueError('Invalid value for `impulse`. Must be the'
                                 ' index of one of the state innovations.')

            # Create the (non-orthogonalized) impulse vector
            idx = impulse
            impulse = np.zeros(self.k_posdef)
            impulse[idx] = 1
        else:
            impulse = np.array(impulse)
            if impulse.ndim > 1:
                impulse = np.squeeze(impulse)
            if not impulse.shape == (self.k_posdef,):
                raise ValueError('Invalid impulse vector. Must be shaped'
                                 ' (%d,)' % self.k_posdef)

        # Orthogonalize the impulses, if requested, using Cholesky on the
        # first state covariance matrix
        if orthogonalized:
            state_chol = np.linalg.cholesky(self.state_cov[:, :, 0])
            impulse = np.dot(state_chol, impulse)

        # If we have time-varying design, transition, or selection matrices,
        # then we can't produce more IRFs than we have time points
        time_invariant_irf = (
            self._design.shape[2] == self._transition.shape[2] ==
            self._selection.shape[2] == 1)

        # Note: to generate impulse responses following the end of a
        # time-varying model, one should `clone` the state space model with the
        # new time-varying model, and then compute the IRFs using the cloned
        # model
        if not time_invariant_irf and steps > self.nobs:
            raise ValueError('In a time-varying model, cannot create more'
                             ' impulse responses than there are'
                             ' observations')

        # Impulse responses only depend on the design, transition, and
        # selection matrices. We set the others to zeros because they must be
        # set in the call to `clone`.
        # Note: we don't even need selection after the first point, because
        # the state shocks will be zeros in every period except the first.
        sim_model = self.clone(
            endog=np.zeros((steps, self.k_endog), dtype=self.dtype),
            obs_intercept=np.zeros(self.k_endog),
            design=self['design', :, :, :steps],
            obs_cov=np.zeros((self.k_endog, self.k_endog)),
            state_intercept=np.zeros(self.k_states),
            transition=self['transition', :, :, :steps],
            selection=self['selection', :, :, :steps],
            state_cov=np.zeros((self.k_posdef, self.k_posdef)))

        # Get the impulse response function via simulation of the state
        # space model, but with other shocks set to zero
        measurement_shocks = np.zeros((steps, self.k_endog))
        state_shocks = np.zeros((steps, self.k_posdef))
        state_shocks[0] = impulse
        initial_state = np.zeros((self.k_states,))
        irf, _ = sim_model.simulate(
            steps, measurement_shocks=measurement_shocks,
            state_shocks=state_shocks, initial_state=initial_state)

        # Get the cumulative response if requested
        if cumulative:
            irf = np.cumsum(irf, axis=0)

        # Here we ignore the first value, because it is always zeros (we added
        # an additional `step` at the top to account for this).
        return irf[1:]


class FilterResults(FrozenRepresentation):
    """
    Results from applying the Kalman filter to a state space model.

    Parameters
    ----------
    model : Representation
        A Statespace representation

    Attributes
    ----------
    nobs : int
        Number of observations.
    nobs_diffuse : int
        Number of observations under the diffuse Kalman filter.
    k_endog : int
        The dimension of the observation series.
    k_states : int
        The dimension of the unobserved state process.
    k_posdef : int
        The dimension of a guaranteed positive definite
        covariance matrix describing the shocks in the
        measurement equation.
    dtype : dtype
        Datatype of representation matrices
    prefix : str
        BLAS prefix of representation matrices
    shapes : dictionary of name,tuple
        A dictionary recording the shapes of each of the
        representation matrices as tuples.
    endog : ndarray
        The observation vector.
    design : ndarray
        The design matrix, :math:`Z`.
    obs_intercept : ndarray
        The intercept for the observation equation, :math:`d`.
    obs_cov : ndarray
        The covariance matrix for the observation equation :math:`H`.
    transition : ndarray
        The transition matrix, :math:`T`.
    state_intercept : ndarray
        The intercept for the transition equation, :math:`c`.
    selection : ndarray
        The selection matrix, :math:`R`.
    state_cov : ndarray
        The covariance matrix for the state equation :math:`Q`.
    missing : array of bool
        An array of the same size as `endog`, filled
        with boolean values that are True if the
        corresponding entry in `endog` is NaN and False
        otherwise.
    nmissing : array of int
        An array of size `nobs`, where the ith entry
        is the number (between 0 and `k_endog`) of NaNs in
        the ith row of the `endog` array.
    time_invariant : bool
        Whether or not the representation matrices are time-invariant
    initialization : str
        Kalman filter initialization method.
    initial_state : array_like
        The state vector used to initialize the Kalamn filter.
    initial_state_cov : array_like
        The state covariance matrix used to initialize the Kalamn filter.
    initial_diffuse_state_cov : array_like
        Diffuse state covariance matrix used to initialize the Kalamn filter.
    filter_method : int
        Bitmask representing the Kalman filtering method
    inversion_method : int
        Bitmask representing the method used to
        invert the forecast error covariance matrix.
    stability_method : int
        Bitmask representing the methods used to promote
        numerical stability in the Kalman filter
        recursions.
    conserve_memory : int
        Bitmask representing the selected memory conservation method.
    filter_timing : int
        Whether or not to use the alternate timing convention.
    tolerance : float
        The tolerance at which the Kalman filter
        determines convergence to steady-state.
    loglikelihood_burn : int
        The number of initial periods during which
        the loglikelihood is not recorded.
    converged : bool
        Whether or not the Kalman filter converged.
    period_converged : int
        The time period in which the Kalman filter converged.
    filtered_state : ndarray
        The filtered state vector at each time period.
    filtered_state_cov : ndarray
        The filtered state covariance matrix at each time period.
    predicted_state : ndarray
        The predicted state vector at each time period.
    predicted_state_cov : ndarray
        The predicted state covariance matrix at each time period.
    forecast_error_diffuse_cov : ndarray
        Diffuse forecast error covariance matrix at each time period.
    predicted_diffuse_state_cov : ndarray
        The predicted diffuse state covariance matrix at each time period.
    kalman_gain : ndarray
        The Kalman gain at each time period.
    forecasts : ndarray
        The one-step-ahead forecasts of observations at each time period.
    forecasts_error : ndarray
        The forecast errors at each time period.
    forecasts_error_cov : ndarray
        The forecast error covariance matrices at each time period.
    llf_obs : ndarray
        The loglikelihood values at each time period.
    """
    _filter_attributes = [
        'filter_method', 'inversion_method', 'stability_method',
        'conserve_memory', 'filter_timing', 'tolerance', 'loglikelihood_burn',
        'converged', 'period_converged', 'filtered_state',
        'filtered_state_cov', 'predicted_state', 'predicted_state_cov',
        'forecasts_error_diffuse_cov', 'predicted_diffuse_state_cov',
        'tmp1', 'tmp2', 'tmp3', 'tmp4', 'forecasts',
        'forecasts_error', 'forecasts_error_cov', 'llf', 'llf_obs',
        'collapsed_forecasts', 'collapsed_forecasts_error',
        'collapsed_forecasts_error_cov', 'scale'
    ]

    _filter_options = (
        KalmanFilter.filter_methods + KalmanFilter.stability_methods +
        KalmanFilter.inversion_methods + KalmanFilter.memory_options
    )

    _attributes = FrozenRepresentation._model_attributes + _filter_attributes

    def __init__(self, model):
        super(FilterResults, self).__init__(model)

        # Setup caches for uninitialized objects
        self._kalman_gain = None
        self._standardized_forecasts_error = None

    def update_representation(self, model, only_options=False):
        """
        Update the results to match a given model

        Parameters
        ----------
        model : Representation
            The model object from which to take the updated values.
        only_options : bool, optional
            If set to true, only the filter options are updated, and the state
            space representation is not updated. Default is False.

        Notes
        -----
        This method is rarely required except for internal usage.
        """
        if not only_options:
            super(FilterResults, self).update_representation(model)

        # Save the options as boolean variables
        for name in self._filter_options:
            setattr(self, name, getattr(model, name, None))

    def update_filter(self, kalman_filter):
        """
        Update the filter results

        Parameters
        ----------
        kalman_filter : statespace.kalman_filter.KalmanFilter
            The model object from which to take the updated values.

        Notes
        -----
        This method is rarely required except for internal usage.
        """
        # State initialization
        self.initial_state = np.array(
            kalman_filter.model.initial_state, copy=True
        )
        self.initial_state_cov = np.array(
            kalman_filter.model.initial_state_cov, copy=True
        )

        # Save Kalman filter parameters
        self.filter_method = kalman_filter.filter_method
        self.inversion_method = kalman_filter.inversion_method
        self.stability_method = kalman_filter.stability_method
        self.conserve_memory = kalman_filter.conserve_memory
        self.filter_timing = kalman_filter.filter_timing
        self.tolerance = kalman_filter.tolerance
        self.loglikelihood_burn = kalman_filter.loglikelihood_burn

        # Save Kalman filter output
        self.converged = bool(kalman_filter.converged)
        self.period_converged = kalman_filter.period_converged
        self.univariate_filter = np.array(kalman_filter.univariate_filter,
                                          copy=True)

        self.filtered_state = np.array(kalman_filter.filtered_state, copy=True)
        self.filtered_state_cov = np.array(
            kalman_filter.filtered_state_cov, copy=True
        )
        self.predicted_state = np.array(
            kalman_filter.predicted_state, copy=True
        )
        self.predicted_state_cov = np.array(
            kalman_filter.predicted_state_cov, copy=True
        )

        # Reset caches
        has_missing = np.sum(self.nmissing) > 0
        if not (self.memory_no_std_forecast or self.invert_lu or
                self.solve_lu or self.filter_collapsed):
            if has_missing:
                self._standardized_forecasts_error = np.array(
                    reorder_missing_vector(
                        kalman_filter.standardized_forecast_error,
                        self.missing, prefix=self.prefix))
            else:
                self._standardized_forecasts_error = np.array(
                    kalman_filter.standardized_forecast_error, copy=True)
        else:
            self._standardized_forecasts_error = None

        # In the partially missing data case, all entries will
        # be in the upper left submatrix rather than the correct placement
        # Re-ordering does not make sense in the collapsed case.
        if has_missing and (not self.memory_no_gain and
                            not self.filter_collapsed):
            self._kalman_gain = np.array(reorder_missing_matrix(
                kalman_filter.kalman_gain, self.missing, reorder_cols=True,
                prefix=self.prefix))
            self.tmp1 = np.array(reorder_missing_matrix(
                kalman_filter.tmp1, self.missing, reorder_cols=True,
                prefix=self.prefix))
            self.tmp2 = np.array(reorder_missing_vector(
                kalman_filter.tmp2, self.missing, prefix=self.prefix))
            self.tmp3 = np.array(reorder_missing_matrix(
                kalman_filter.tmp3, self.missing, reorder_rows=True,
                prefix=self.prefix))
            self.tmp4 = np.array(reorder_missing_matrix(
                kalman_filter.tmp4, self.missing, reorder_cols=True,
                reorder_rows=True, prefix=self.prefix))
        else:
            if not self.memory_no_gain:
                self._kalman_gain = np.array(
                    kalman_filter.kalman_gain, copy=True)
            self.tmp1 = np.array(kalman_filter.tmp1, copy=True)
            self.tmp2 = np.array(kalman_filter.tmp2, copy=True)
            self.tmp3 = np.array(kalman_filter.tmp3, copy=True)
            self.tmp4 = np.array(kalman_filter.tmp4, copy=True)
            self.M = np.array(kalman_filter.M, copy=True)
            self.M_diffuse = np.array(kalman_filter.M_inf, copy=True)

        # Note: use forecasts rather than forecast, so as not to interfer
        # with the `forecast` methods in subclasses
        self.forecasts = np.array(kalman_filter.forecast, copy=True)
        self.forecasts_error = np.array(
            kalman_filter.forecast_error, copy=True
        )
        self.forecasts_error_cov = np.array(
            kalman_filter.forecast_error_cov, copy=True
        )
        # Note: below we will set self.llf, and in the memory_no_likelihood
        # case we will replace self.llf_obs = None at that time.
        self.llf_obs = np.array(kalman_filter.loglikelihood, copy=True)

        # Diffuse objects
        self.nobs_diffuse = kalman_filter.nobs_diffuse
        self.initial_diffuse_state_cov = None
        self.forecasts_error_diffuse_cov = None
        self.predicted_diffuse_state_cov = None
        if self.nobs_diffuse > 0:
            self.initial_diffuse_state_cov = np.array(
                kalman_filter.model.initial_diffuse_state_cov, copy=True)
            self.predicted_diffuse_state_cov = np.array(
                    kalman_filter.predicted_diffuse_state_cov, copy=True)
            if has_missing and not self.filter_collapsed:
                self.forecasts_error_diffuse_cov = np.array(
                    reorder_missing_matrix(
                        kalman_filter.forecast_error_diffuse_cov,
                        self.missing, reorder_cols=True, reorder_rows=True,
                        prefix=self.prefix))
            else:
                self.forecasts_error_diffuse_cov = np.array(
                    kalman_filter.forecast_error_diffuse_cov, copy=True)

        # If there was missing data, save the original values from the Kalman
        # filter output, since below will set the values corresponding to
        # the missing observations to nans.
        self.missing_forecasts = None
        self.missing_forecasts_error = None
        self.missing_forecasts_error_cov = None
        if np.sum(self.nmissing) > 0:
            # Copy the provided arrays (which are as the Kalman filter dataset)
            # into new variables
            self.missing_forecasts = np.copy(self.forecasts)
            self.missing_forecasts_error = np.copy(self.forecasts_error)
            self.missing_forecasts_error_cov = (
                np.copy(self.forecasts_error_cov)
            )

        # Save the collapsed values
        self.collapsed_forecasts = None
        self.collapsed_forecasts_error = None
        self.collapsed_forecasts_error_cov = None
        if self.filter_collapsed:
            # Copy the provided arrays (which are from the collapsed dataset)
            # into new variables
            self.collapsed_forecasts = self.forecasts[:self.k_states, :]
            self.collapsed_forecasts_error = (
                self.forecasts_error[:self.k_states, :]
            )
            self.collapsed_forecasts_error_cov = (
                self.forecasts_error_cov[:self.k_states, :self.k_states, :]
            )
            # Recreate the original arrays (which should be from the original
            # dataset) in the appropriate dimension
            dtype = self.collapsed_forecasts.dtype
            self.forecasts = np.zeros((self.k_endog, self.nobs), dtype=dtype)
            self.forecasts_error = np.zeros((self.k_endog, self.nobs),
                                            dtype=dtype)
            self.forecasts_error_cov = (
                np.zeros((self.k_endog, self.k_endog, self.nobs), dtype=dtype)
            )

        # Fill in missing values in the forecast, forecast error, and
        # forecast error covariance matrix (this is required due to how the
        # Kalman filter implements observations that are either partly or
        # completely missing)
        # Construct the predictions, forecasts
        can_compute_mean = not (self.memory_no_forecast_mean or
                                self.memory_no_predicted_mean)
        can_compute_cov = not (self.memory_no_forecast_cov or
                               self.memory_no_predicted_cov)
        if can_compute_mean or can_compute_cov:
            for t in range(self.nobs):
                design_t = 0 if self.design.shape[2] == 1 else t
                obs_cov_t = 0 if self.obs_cov.shape[2] == 1 else t
                obs_intercept_t = 0 if self.obs_intercept.shape[1] == 1 else t

                # For completely missing observations, the Kalman filter will
                # produce forecasts, but forecast errors and the forecast
                # error covariance matrix will be zeros - make them nan to
                # improve clarity of results.
                if self.nmissing[t] > 0:
                    mask = ~self.missing[:, t].astype(bool)
                    # We can recover forecasts
                    # For partially missing observations, the Kalman filter
                    # will produce all elements (forecasts, forecast errors,
                    # forecast error covariance matrices) as usual, but their
                    # dimension will only be equal to the number of non-missing
                    # elements, and their location in memory will be in the
                    # first blocks (e.g. for the forecasts_error, the first
                    # k_endog - nmissing[t] columns will be filled in),
                    # regardless of which endogenous variables they refer to
                    # (i.e. the non- missing endogenous variables for that
                    # observation). Furthermore, the forecast error covariance
                    # matrix is only valid for those elements. What is done is
                    # to set all elements to nan for these observations so that
                    # they are flagged as missing. The variables
                    # missing_forecasts, etc. then provide the forecasts, etc.
                    # provided by the Kalman filter, from which the data can be
                    # retrieved if desired.
                    if can_compute_mean:
                        self.forecasts[:, t] = np.dot(
                            self.design[:, :, design_t],
                            self.predicted_state[:, t]
                        ) + self.obs_intercept[:, obs_intercept_t]
                        self.forecasts_error[:, t] = np.nan
                        self.forecasts_error[mask, t] = (
                            self.endog[mask, t] - self.forecasts[mask, t])
                    # TODO: We should only fill in the non-masked elements of
                    # this array. Also, this will give the multivariate version
                    # even if univariate filtering was selected. Instead, we
                    # should use the reordering methods and then replace the
                    # masked values with NaNs
                    if can_compute_cov:
                        self.forecasts_error_cov[:, :, t] = np.dot(
                            np.dot(self.design[:, :, design_t],
                                   self.predicted_state_cov[:, :, t]),
                            self.design[:, :, design_t].T
                        ) + self.obs_cov[:, :, obs_cov_t]
                # In the collapsed case, everything just needs to be rebuilt
                # for the original observed data, since the Kalman filter
                # produced these values for the collapsed data.
                elif self.filter_collapsed:
                    if can_compute_mean:
                        self.forecasts[:, t] = np.dot(
                            self.design[:, :, design_t],
                            self.predicted_state[:, t]
                        ) + self.obs_intercept[:, obs_intercept_t]

                        self.forecasts_error[:, t] = (
                            self.endog[:, t] - self.forecasts[:, t]
                        )

                    if can_compute_cov:
                        self.forecasts_error_cov[:, :, t] = np.dot(
                            np.dot(self.design[:, :, design_t],
                                   self.predicted_state_cov[:, :, t]),
                            self.design[:, :, design_t].T
                        ) + self.obs_cov[:, :, obs_cov_t]

        # Note: if we concentrated out the scale, need to adjust the
        # loglikelihood values and all of the covariance matrices and the
        # values that depend on the covariance matrices
        # Note: concentrated computation is not permitted with collapsed
        # version, so we do not need to modify collapsed arrays.
        self.scale = 1.
        if self.filter_concentrated and self.model._scale is None:
            d = max(self.loglikelihood_burn, self.nobs_diffuse)
            # Compute the scale
            nmissing = np.array(kalman_filter.model.nmissing)
            nobs_k_endog = np.sum(self.k_endog - nmissing[d:])

            # In the univariate case, we need to subtract observations
            # associated with a singular forecast error covariance matrix
            nobs_k_endog -= kalman_filter.nobs_kendog_univariate_singular

            scale_obs = np.array(kalman_filter.scale, copy=True)
            if not self.memory_no_likelihood:
                self.scale = np.sum(scale_obs[d:]) / nobs_k_endog
            else:
                self.scale = scale_obs[0] / nobs_k_endog

            # Need to modify this for diffuse initialization, since for
            # diffuse periods we only need to add in the scale value if the
            # diffuse forecast error covariance matrix element was singular
            nsingular = 0
            if kalman_filter.nobs_diffuse > 0:
                Finf = kalman_filter.forecast_error_diffuse_cov
                singular = (np.diagonal(Finf).real <=
                            kalman_filter.tolerance_diffuse)
                nsingular = np.sum(~singular, axis=1)

            # Adjust the loglikelihood obs (see `KalmanFilter.loglikeobs` for
            # defaults on the adjustment)
            if not self.memory_no_likelihood:
                self.llf_obs += -0.5 * (
                    (self.k_endog - nmissing - nsingular) * np.log(self.scale)
                    + scale_obs / self.scale)
            else:
                self.llf_obs[0] += -0.5 * np.squeeze(
                    np.sum(
                        (self.k_endog - nmissing - nsingular)
                        * np.log(self.scale)
                    )
                    + scale_obs / self.scale
                )

            # Scale the filter output
            self.obs_cov = self.obs_cov * self.scale
            self.state_cov = self.state_cov * self.scale

            self.initial_state_cov = self.initial_state_cov * self.scale
            self.predicted_state_cov = self.predicted_state_cov * self.scale
            self.filtered_state_cov = self.filtered_state_cov * self.scale
            self.forecasts_error_cov = self.forecasts_error_cov * self.scale
            if self.missing_forecasts_error_cov is not None:
                self.missing_forecasts_error_cov = (
                    self.missing_forecasts_error_cov * self.scale)

            # Note: do not have to adjust the Kalman gain or tmp4
            self.tmp1 = self.tmp1 * self.scale
            self.tmp2 = self.tmp2 / self.scale
            self.tmp3 = self.tmp3 / self.scale
            if not (self.memory_no_std_forecast or
                    self.invert_lu or
                    self.solve_lu or
                    self.filter_collapsed):
                self._standardized_forecasts_error = (
                    self._standardized_forecasts_error / self.scale**0.5)
        # The self.model._scale value is only not None within a fixed_scale
        # context, in which case it is set and indicates that we should
        # generally view this results object as using a concentrated scale
        # (e.g. for d.o.f. computations), but because the fixed scale was
        # actually applied to the model prior to filtering, we do not need to
        # make any adjustments to the filter output, etc.
        elif self.model._scale is not None:
            self.filter_concentrated = True
            self.scale = self.model._scale

        # Now, save self.llf, and handle the memory_no_likelihood case
        if not self.memory_no_likelihood:
            self.llf = np.sum(self.llf_obs[self.loglikelihood_burn:])
        else:
            self.llf = self.llf_obs[0]
            self.llf_obs = None

    @property
    def kalman_gain(self):
        """
        Kalman gain matrices
        """
        if self._kalman_gain is None:
            # k x n
            self._kalman_gain = np.zeros(
                (self.k_states, self.k_endog, self.nobs), dtype=self.dtype)
            for t in range(self.nobs):
                # In the case of entirely missing observations, let the Kalman
                # gain be zeros.
                if self.nmissing[t] == self.k_endog:
                    continue

                design_t = 0 if self.design.shape[2] == 1 else t
                transition_t = 0 if self.transition.shape[2] == 1 else t
                if self.nmissing[t] == 0:
                    self._kalman_gain[:, :, t] = np.dot(
                        np.dot(
                            self.transition[:, :, transition_t],
                            self.predicted_state_cov[:, :, t]
                        ),
                        np.dot(
                            np.transpose(self.design[:, :, design_t]),
                            np.linalg.inv(self.forecasts_error_cov[:, :, t])
                        )
                    )
                else:
                    mask = ~self.missing[:, t].astype(bool)
                    F = self.forecasts_error_cov[np.ix_(mask, mask, [t])]
                    self._kalman_gain[:, mask, t] = np.dot(
                        np.dot(
                            self.transition[:, :, transition_t],
                            self.predicted_state_cov[:, :, t]
                        ),
                        np.dot(
                            np.transpose(self.design[mask, :, design_t]),
                            np.linalg.inv(F[:, :, 0])
                        )
                    )
        return self._kalman_gain

    @property
    def standardized_forecasts_error(self):
        r"""
        Standardized forecast errors

        Notes
        -----
        The forecast errors produced by the Kalman filter are

        .. math::

            v_t \sim N(0, F_t)

        Hypothesis tests are usually applied to the standardized residuals

        .. math::

            v_t^s = B_t v_t \sim N(0, I)

        where :math:`B_t = L_t^{-1}` and :math:`F_t = L_t L_t'`; then
        :math:`F_t^{-1} = (L_t')^{-1} L_t^{-1} = B_t' B_t`; :math:`B_t`
        and :math:`L_t` are lower triangular. Finally,
        :math:`B_t v_t \sim N(0, B_t F_t B_t')` and
        :math:`B_t F_t B_t' = L_t^{-1} L_t L_t' (L_t')^{-1} = I`.

        Thus we can rewrite :math:`v_t^s = L_t^{-1} v_t` or
        :math:`L_t v_t^s = v_t`; the latter equation is the form required to
        use a linear solver to recover :math:`v_t^s`. Since :math:`L_t` is
        lower triangular, we can use a triangular solver (?TRTRS).
        """
        if (self._standardized_forecasts_error is None
                and not self.memory_no_forecast):
            if self.k_endog == 1:
                self._standardized_forecasts_error = (
                    self.forecasts_error /
                    self.forecasts_error_cov[0, 0, :]**0.5)
            else:
                from scipy import linalg
                self._standardized_forecasts_error = np.zeros(
                    self.forecasts_error.shape, dtype=self.dtype)
                for t in range(self.forecasts_error_cov.shape[2]):
                    if self.nmissing[t] > 0:
                        self._standardized_forecasts_error[:, t] = np.nan
                    if self.nmissing[t] < self.k_endog:
                        mask = ~self.missing[:, t].astype(bool)
                        F = self.forecasts_error_cov[np.ix_(mask, mask, [t])]
                        try:
                            upper, _ = linalg.cho_factor(F[:, :, 0])
                            self._standardized_forecasts_error[mask, t] = (
                                linalg.solve_triangular(
                                    upper, self.forecasts_error[mask, t],
                                    trans=1))
                        except linalg.LinAlgError:
                            self._standardized_forecasts_error[mask, t] = (
                                np.nan)

        return self._standardized_forecasts_error

    def predict(self, start=None, end=None, dynamic=None, **kwargs):
        r"""
        In-sample and out-of-sample prediction for state space models generally

        Parameters
        ----------
        start : int, optional
            Zero-indexed observation number at which to start prediction, i.e.,
            the first prediction will be at start.
        end : int, optional
            Zero-indexed observation number at which to end prediction, i.e.,
            the last prediction will be at end.
        dynamic : int, optional
            Offset relative to `start` at which to begin dynamic prediction.
            Prior to this observation, true endogenous values will be used for
            prediction; starting with this observation and continuing through
            the end of prediction, predicted endogenous values will be used
            instead.
        **kwargs
            If the prediction range is outside of the sample range, any
            of the state space representation matrices that are time-varying
            must have updated values provided for the out-of-sample range.
            For example, of `obs_intercept` is a time-varying component and
            the prediction range extends 10 periods beyond the end of the
            sample, a (`k_endog` x 10) matrix must be provided with the new
            intercept values.

        Returns
        -------
        results : kalman_filter.PredictionResults
            A PredictionResults object.

        Notes
        -----
        All prediction is performed by applying the deterministic part of the
        measurement equation using the predicted state variables.

        Out-of-sample prediction first applies the Kalman filter to missing
        data for the number of periods desired to obtain the predicted states.
        """
        # Get the start and the end of the entire prediction range
        if start is None:
            start = 0
        elif start < 0:
            raise ValueError('Cannot predict values previous to the sample.')
        if end is None:
            end = self.nobs

        # Prediction and forecasting is performed by iterating the Kalman
        # Kalman filter through the entire range [0, end]
        # Then, everything is returned corresponding to the range [start, end].
        # In order to perform the calculations, the range is separately split
        # up into the following categories:
        # - static:   (in-sample) the Kalman filter is run as usual
        # - dynamic:  (in-sample) the Kalman filter is run, but on missing data
        # - forecast: (out-of-sample) the Kalman filter is run, but on missing
        #             data

        # Short-circuit if end is before start
        if end <= start:
            raise ValueError('End of prediction must be after start.')

        # Get the number of forecasts to make after the end of the sample
        nforecast = max(0, end - self.nobs)

        # Get the number of dynamic prediction periods

        # If `dynamic=True`, then assume that we want to begin dynamic
        # prediction at the start of the sample prediction.
        if dynamic is True:
            dynamic = 0
        # If `dynamic=False`, then assume we want no dynamic prediction
        if dynamic is False:
            dynamic = None

        # Check validity of dynamic and warn or error if issues
        dynamic, ndynamic = _check_dynamic(dynamic, start, end, self.nobs)

        # Get the number of in-sample static predictions
        if dynamic is None:
            nstatic = min(end, self.nobs) - min(start, self.nobs)
        else:
            # (use max(., 0), since dynamic can be prior to start)
            nstatic = max(dynamic - start, 0)

        # Cannot do in-sample prediction if we do not have appropriate
        # arrays (we can do out-of-sample forecasting, however)
        if nstatic > 0 and self.memory_no_forecast_mean:
            raise ValueError('In-sample prediction is not available if memory'
                             ' conservation has been used to avoid storing'
                             ' forecast means.')
        # Cannot do dynamic in-sample prediction if we do not have appropriate
        # arrays (we can do out-of-sample forecasting, however)
        if ndynamic > 0 and self.memory_no_predicted:
            raise ValueError('In-sample dynamic prediction is not available if'
                             ' memory conservation has been used to avoid'
                             ' storing forecasted or predicted state means'
                             ' or covariances.')

        # Construct the predicted state and covariance matrix for each time
        # period depending on whether that time period corresponds to
        # one-step-ahead prediction, dynamic prediction, or out-of-sample
        # forecasting.

        # If we only have simple prediction, then we can use the already saved
        # Kalman filter output
        if ndynamic == 0 and nforecast == 0:
            results = self
            oos_results = None
        # If we have dynamic prediction or forecasting, then we need to
        # re-apply the Kalman filter
        else:
            # Figure out the period for which we need to run the Kalman filter
            if dynamic is not None:
                kf_start = min(dynamic, self.nobs)
            else:
                kf_start = self.nobs
            kf_end = end

            # Make start, end consistent with the results that we're generating
            # start = max(start - kf_start, 0)
            # end = kf_end - kf_start

            # We must at least store forecasts and predictions
            kwargs['conserve_memory'] = (
                self.conserve_memory & ~MEMORY_NO_FORECAST &
                ~MEMORY_NO_PREDICTED)

            # Can't use Chandrasekhar recursions for prediction
            kwargs['filter_method'] = (
                self.model.filter_method & ~FILTER_CHANDRASEKHAR)

            # TODO: there is a corner case here when the filter has not
            #       exited the diffuse filter, in which case this known
            #       initialization is not correct.
            # Even if we have not stored all predicted values (means and covs),
            # we can still do pure out-of-sample forecasting because we will
            # always have stored the last predicted values. In this case, we
            # will initialize the forecasting filter with these values
            if self.memory_no_predicted:
                constant = self.predicted_state[..., -1]
                stationary_cov = self.predicted_state_cov[..., -1]
            # Otherwise initialize with the predicted state / cov from the
            # existing results, at index kf_start (note that the time
            # dimension of predicted_state and predicted_state_cov is
            # self.nobs + 1; so e.g. in the case of pure forecasting we should
            # be using the very last predicted state and predicted state cov
            # elements, and kf_start will equal self.nobs which is correct)
            else:
                constant = self.predicted_state[..., kf_start]
                stationary_cov = self.predicted_state_cov[..., kf_start]

            kwargs.update({'initialization': 'known',
                           'constant': constant,
                           'stationary_cov': stationary_cov})

            # Construct the new endogenous array.
            endog = np.zeros((nforecast, self.k_endog)) * np.nan
            model = self.model.extend(
                endog, start=kf_start, end=kf_end - nforecast, **kwargs)
            # Have to retroactively modify the model's endog
            if ndynamic > 0:
                model.endog[:, -(ndynamic + nforecast):] = np.nan

            with model.fixed_scale(self.scale):
                oos_results = model.filter()

            results = self

        return PredictionResults(results, start, end, nstatic, ndynamic,
                                 nforecast, oos_results=oos_results)


class PredictionResults(FilterResults):
    r"""
    Results of in-sample and out-of-sample prediction for state space models
    generally

    Parameters
    ----------
    results : FilterResults
        Output from filtering, corresponding to the prediction desired
    start : int
        Zero-indexed observation number at which to start forecasting,
        i.e., the first forecast will be at start.
    end : int
        Zero-indexed observation number at which to end forecasting, i.e.,
        the last forecast will be at end.
    nstatic : int
        Number of in-sample static predictions (these are always the first
        elements of the prediction output).
    ndynamic : int
        Number of in-sample dynamic predictions (these always follow the static
        predictions directly, and are directly followed by the forecasts).
    nforecast : int
        Number of in-sample forecasts (these always follow the dynamic
        predictions directly).

    Attributes
    ----------
    npredictions : int
        Number of observations in the predicted series; this is not necessarily
        the same as the number of observations in the original model from which
        prediction was performed.
    start : int
        Zero-indexed observation number at which to start prediction,
        i.e., the first predict will be at `start`; this is relative to the
        original model from which prediction was performed.
    end : int
        Zero-indexed observation number at which to end prediction,
        i.e., the last predict will be at `end`; this is relative to the
        original model from which prediction was performed.
    nstatic : int
        Number of in-sample static predictions.
    ndynamic : int
        Number of in-sample dynamic predictions.
    nforecast : int
        Number of in-sample forecasts.
    endog : ndarray
        The observation vector.
    design : ndarray
        The design matrix, :math:`Z`.
    obs_intercept : ndarray
        The intercept for the observation equation, :math:`d`.
    obs_cov : ndarray
        The covariance matrix for the observation equation :math:`H`.
    transition : ndarray
        The transition matrix, :math:`T`.
    state_intercept : ndarray
        The intercept for the transition equation, :math:`c`.
    selection : ndarray
        The selection matrix, :math:`R`.
    state_cov : ndarray
        The covariance matrix for the state equation :math:`Q`.
    filtered_state : ndarray
        The filtered state vector at each time period.
    filtered_state_cov : ndarray
        The filtered state covariance matrix at each time period.
    predicted_state : ndarray
        The predicted state vector at each time period.
    predicted_state_cov : ndarray
        The predicted state covariance matrix at each time period.
    forecasts : ndarray
        The one-step-ahead forecasts of observations at each time period.
    forecasts_error : ndarray
        The forecast errors at each time period.
    forecasts_error_cov : ndarray
        The forecast error covariance matrices at each time period.

    Notes
    -----
    The provided ranges must be conformable, meaning that it must be that
    `end - start == nstatic + ndynamic + nforecast`.

    This class is essentially a view to the FilterResults object, but
    returning the appropriate ranges for everything.
    """
    representation_attributes = [
        'endog', 'design', 'obs_intercept',
        'obs_cov', 'transition', 'state_intercept', 'selection',
        'state_cov'
    ]
    filter_attributes = [
        'filtered_state', 'filtered_state_cov',
        'predicted_state', 'predicted_state_cov',
        'forecasts', 'forecasts_error', 'forecasts_error_cov'
    ]
    smoother_attributes = [
        'smoothed_state', 'smoothed_state_cov',
    ]

    def __init__(self, results, start, end, nstatic, ndynamic, nforecast,
                 oos_results=None):
        # Save the filter results object
        self.results = results
        self.oos_results = oos_results

        # Save prediction ranges
        self.npredictions = start - end
        self.start = start
        self.end = end
        self.nstatic = nstatic
        self.ndynamic = ndynamic
        self.nforecast = nforecast

        self._predicted_signal = None
        self._predicted_signal_cov = None
        self._filtered_signal = None
        self._filtered_signal_cov = None
        self._smoothed_signal = None
        self._smoothed_signal_cov = None
        self._filtered_forecasts = None
        self._filtered_forecasts_error_cov = None
        self._smoothed_forecasts = None
        self._smoothed_forecasts_error_cov = None

    def clear(self):
        attributes = (['endog'] + self.representation_attributes
                      + self.filter_attributes)
        for attr in attributes:
            _attr = '_' + attr
            if hasattr(self, _attr):
                delattr(self, _attr)

    def __getattr__(self, attr):
        """
        Provide access to the representation and filtered output in the
        appropriate range (`start` - `end`).
        """
        # Prevent infinite recursive lookups
        if attr[0] == '_':
            raise AttributeError("'%s' object has no attribute '%s'" %
                                 (self.__class__.__name__, attr))

        _attr = '_' + attr

        # Cache the attribute
        if not hasattr(self, _attr):
            if attr == 'endog' or attr in self.filter_attributes:
                # Get a copy
                value = getattr(self.results, attr).copy()
                if self.ndynamic > 0:
                    end = self.end - self.ndynamic - self.nforecast
                    value = value[..., :end]
                if self.oos_results is not None:
                    oos_value = getattr(self.oos_results, attr).copy()

                    # Note that the last element of the results predicted state
                    # and state cov will overlap with the first element of the
                    # oos predicted state and state cov, so eliminate the
                    # last element of the results versions
                    # But if we have dynamic prediction, then we have already
                    # eliminated the last element of the predicted state, so
                    # we do not need to do it here.
                    if self.ndynamic == 0 and attr[:9] == 'predicted':
                        value = value[..., :-1]

                    value = np.concatenate([value, oos_value], axis=-1)

                # Subset to the correct time frame
                value = value[..., self.start:self.end]
            elif attr in self.smoother_attributes:
                if self.ndynamic > 0:
                    raise NotImplementedError(
                        'Cannot retrieve smoothed attributes when using'
                        ' dynamic prediction, since the information set used'
                        ' to compute the smoothed results differs from the'
                        ' information set implied by the dynamic prediction.')
                # Get a copy
                value = getattr(self.results, attr).copy()

                # The oos_results object is only dynamic or out-of-sample,
                # so filtered == smoothed
                if self.oos_results is not None:
                    filtered_attr = 'filtered' + attr[8:]
                    oos_value = getattr(self.oos_results, filtered_attr).copy()
                    value = np.concatenate([value, oos_value], axis=-1)

                # Subset to the correct time frame
                value = value[..., self.start:self.end]
            elif attr in self.representation_attributes:
                value = getattr(self.results, attr).copy()
                # If a time-invariant matrix, return it. Otherwise, subset to
                # the correct period.
                if value.shape[-1] == 1:
                    value = value[..., 0]
                else:
                    if self.ndynamic > 0:
                        end = self.end - self.ndynamic - self.nforecast
                        value = value[..., :end]

                    if self.oos_results is not None:
                        oos_value = getattr(self.oos_results, attr).copy()
                        value = np.concatenate([value, oos_value], axis=-1)
                    value = value[..., self.start:self.end]
            else:
                raise AttributeError("'%s' object has no attribute '%s'" %
                                     (self.__class__.__name__, attr))

            setattr(self, _attr, value)

        return getattr(self, _attr)

    def _compute_forecasts(self, states, states_cov, signal_only=False):
        d = self.obs_intercept
        Z = self.design
        H = self.obs_cov

        if d.ndim == 1:
            d = d[:, None]

        if Z.ndim == 2:
            if not signal_only:
                forecasts = d + Z @ states
                forecasts_error_cov = (
                    Z[None, ...] @ states_cov.T @ Z.T[None, ...] + H.T).T
            else:
                forecasts = Z @ states
                forecasts_error_cov = (
                    Z[None, ...] @ states_cov.T @ Z.T[None, ...]).T
        else:
            if not signal_only:
                forecasts = d + (Z * states[None, :, :]).sum(axis=1)
                tmp = Z[:, None, ...] * states_cov[None, ...]
                tmp = (tmp[:, :, :, None, :]
                       * Z.transpose(1, 0, 2)[None, :, None, ...])
                forecasts_error_cov = (tmp.sum(axis=1).sum(axis=1).T + H.T).T
            else:
                forecasts = (Z * states[None, :, :]).sum(axis=1)
                tmp = Z[:, None, ...] * states_cov[None, ...]
                tmp = (tmp[:, :, :, None, :]
                       * Z.transpose(1, 0, 2)[None, :, None, ...])
                forecasts_error_cov = tmp.sum(axis=1).sum(axis=1)

        return forecasts, forecasts_error_cov

    @property
    def predicted_signal(self):
        if self._predicted_signal is None:
            self._predicted_signal, self._predicted_signal_cov = (
                self._compute_forecasts(self.predicted_state,
                                        self.predicted_state_cov,
                                        signal_only=True))
        return self._predicted_signal

    @property
    def predicted_signal_cov(self):
        if self._predicted_signal_cov is None:
            self._predicted_signal, self._predicted_signal_cov = (
                self._compute_forecasts(self.predicted_state,
                                        self.predicted_state_cov,
                                        signal_only=True))
        return self._predicted_signal_cov

    @property
    def filtered_signal(self):
        if self._filtered_signal is None:
            self._filtered_signal, self._filtered_signal_cov = (
                self._compute_forecasts(self.filtered_state,
                                        self.filtered_state_cov,
                                        signal_only=True))
        return self._filtered_signal

    @property
    def filtered_signal_cov(self):
        if self._filtered_signal_cov is None:
            self._filtered_signal, self._filtered_signal_cov = (
                self._compute_forecasts(self.filtered_state,
                                        self.filtered_state_cov,
                                        signal_only=True))
        return self._filtered_signal_cov

    @property
    def smoothed_signal(self):
        if self._smoothed_signal is None:
            self._smoothed_signal, self._smoothed_signal_cov = (
                self._compute_forecasts(self.smoothed_state,
                                        self.smoothed_state_cov,
                                        signal_only=True))
        return self._smoothed_signal

    @property
    def smoothed_signal_cov(self):
        if self._smoothed_signal_cov is None:
            self._smoothed_signal, self._smoothed_signal_cov = (
                self._compute_forecasts(self.smoothed_state,
                                        self.smoothed_state_cov,
                                        signal_only=True))
        return self._smoothed_signal_cov

    @property
    def filtered_forecasts(self):
        if self._filtered_forecasts is None:
            self._filtered_forecasts, self._filtered_forecasts_cov = (
                self._compute_forecasts(self.filtered_state,
                                        self.filtered_state_cov))
        return self._filtered_forecasts

    @property
    def filtered_forecasts_error_cov(self):
        if self._filtered_forecasts_cov is None:
            self._filtered_forecasts, self._filtered_forecasts_cov = (
                self._compute_forecasts(self.filtered_state,
                                        self.filtered_state_cov))
        return self._filtered_forecasts_cov

    @property
    def smoothed_forecasts(self):
        if self._smoothed_forecasts is None:
            self._smoothed_forecasts, self._smoothed_forecasts_cov = (
                self._compute_forecasts(self.smoothed_state,
                                        self.smoothed_state_cov))
        return self._smoothed_forecasts

    @property
    def smoothed_forecasts_error_cov(self):
        if self._smoothed_forecasts_cov is None:
            self._smoothed_forecasts, self._smoothed_forecasts_cov = (
                self._compute_forecasts(self.smoothed_state,
                                        self.smoothed_state_cov))
        return self._smoothed_forecasts_cov


def _check_dynamic(dynamic, start, end, nobs):
    """
    Verify dynamic and warn or error if issues

    Parameters
    ----------
    dynamic : {int, None}
        The offset relative to start of the dynamic forecasts. None if no
        dynamic forecasts are required.
    start : int
        The location of the first forecast.
    end : int
        The location of the final forecast (inclusive).
    nobs : int
        The number of observations in the time series.

    Returns
    -------
    dynamic : {int, None}
        The start location of the first dynamic forecast. None if there
        are no in-sample dynamic forecasts.
    ndynamic : int
        The number of dynamic forecasts
    """
    if dynamic is None:
        return dynamic, 0

    # Replace the relative dynamic offset with an absolute offset
    dynamic = start + dynamic

    # Validate the `dynamic` parameter
    if dynamic < 0:
        raise ValueError('Dynamic prediction cannot begin prior to the'
                         ' first observation in the sample.')
    elif dynamic > end:
        warn('Dynamic prediction specified to begin after the end of'
             ' prediction, and so has no effect.', ValueWarning)
        return None, 0
    elif dynamic > nobs:
        warn('Dynamic prediction specified to begin during'
             ' out-of-sample forecasting period, and so has no'
             ' effect.', ValueWarning)
        return None, 0

    # Get the total size of the desired dynamic forecasting component
    # Note: the first `dynamic` periods of prediction are actually
    # *not* dynamic, because dynamic prediction begins at observation
    # `dynamic`.
    ndynamic = max(0, min(end, nobs) - dynamic)
    return dynamic, ndynamic
