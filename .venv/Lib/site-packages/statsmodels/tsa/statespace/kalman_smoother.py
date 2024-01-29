"""
State Space Representation and Kalman Filter, Smoother

Author: Chad Fulton
License: Simplified-BSD
"""

import numpy as np
from types import SimpleNamespace

from statsmodels.tsa.statespace.representation import OptionWrapper
from statsmodels.tsa.statespace.kalman_filter import (KalmanFilter,
                                                      FilterResults)
from statsmodels.tsa.statespace.tools import (
    reorder_missing_matrix, reorder_missing_vector, copy_index_matrix)
from statsmodels.tsa.statespace import tools, initialization

SMOOTHER_STATE = 0x01              # Durbin and Koopman (2012), Chapter 4.4.2
SMOOTHER_STATE_COV = 0x02          # ibid., Chapter 4.4.3
SMOOTHER_DISTURBANCE = 0x04        # ibid., Chapter 4.5
SMOOTHER_DISTURBANCE_COV = 0x08    # ibid., Chapter 4.5
SMOOTHER_STATE_AUTOCOV = 0x10      # ibid., Chapter 4.7
SMOOTHER_ALL = (
    SMOOTHER_STATE | SMOOTHER_STATE_COV | SMOOTHER_DISTURBANCE |
    SMOOTHER_DISTURBANCE_COV | SMOOTHER_STATE_AUTOCOV
)

SMOOTH_CONVENTIONAL = 0x01
SMOOTH_CLASSICAL = 0x02
SMOOTH_ALTERNATIVE = 0x04
SMOOTH_UNIVARIATE = 0x08


class KalmanSmoother(KalmanFilter):
    r"""
    State space representation of a time series process, with Kalman filter
    and smoother.

    Parameters
    ----------
    k_endog : {array_like, int}
        The observed time-series process :math:`y` if array like or the
        number of variables in the process if an integer.
    k_states : int
        The dimension of the unobserved state process.
    k_posdef : int, optional
        The dimension of a guaranteed positive definite covariance matrix
        describing the shocks in the measurement equation. Must be less than
        or equal to `k_states`. Default is `k_states`.
    results_class : class, optional
        Default results class to use to save filtering output. Default is
        `SmootherResults`. If specified, class must extend from
        `SmootherResults`.
    **kwargs
        Keyword arguments may be used to provide default values for state space
        matrices, for Kalman filtering options, or for Kalman smoothing
        options. See `Representation` for more details.
    """

    smoother_outputs = [
        'smoother_state', 'smoother_state_cov', 'smoother_state_autocov',
        'smoother_disturbance', 'smoother_disturbance_cov', 'smoother_all',
    ]

    smoother_state = OptionWrapper('smoother_output', SMOOTHER_STATE)
    smoother_state_cov = OptionWrapper('smoother_output', SMOOTHER_STATE_COV)
    smoother_disturbance = (
        OptionWrapper('smoother_output', SMOOTHER_DISTURBANCE)
    )
    smoother_disturbance_cov = (
        OptionWrapper('smoother_output', SMOOTHER_DISTURBANCE_COV)
    )
    smoother_state_autocov = (
        OptionWrapper('smoother_output', SMOOTHER_STATE_AUTOCOV)
    )
    smoother_all = OptionWrapper('smoother_output', SMOOTHER_ALL)

    smooth_methods = [
        'smooth_conventional', 'smooth_alternative', 'smooth_classical'
    ]

    smooth_conventional = OptionWrapper('smooth_method', SMOOTH_CONVENTIONAL)
    """
    (bool) Flag for conventional (Durbin and Koopman, 2012) Kalman smoothing.
    """
    smooth_alternative = OptionWrapper('smooth_method', SMOOTH_ALTERNATIVE)
    """
    (bool) Flag for alternative (modified Bryson-Frazier) smoothing.
    """
    smooth_classical = OptionWrapper('smooth_method', SMOOTH_CLASSICAL)
    """
    (bool) Flag for classical (see e.g. Anderson and Moore, 1979) smoothing.
    """
    smooth_univariate = OptionWrapper('smooth_method', SMOOTH_UNIVARIATE)
    """
    (bool) Flag for univariate smoothing (uses modified Bryson-Frazier timing).
    """

    # Default smoother options
    smoother_output = SMOOTHER_ALL
    smooth_method = 0

    def __init__(self, k_endog, k_states, k_posdef=None, results_class=None,
                 kalman_smoother_classes=None, **kwargs):
        # Set the default results class
        if results_class is None:
            results_class = SmootherResults

        # Extract keyword arguments to-be-used later
        keys = ['smoother_output'] + KalmanSmoother.smoother_outputs
        smoother_output_kwargs = {key: kwargs.pop(key) for key in keys
                                  if key in kwargs}
        keys = ['smooth_method'] + KalmanSmoother.smooth_methods
        smooth_method_kwargs = {key: kwargs.pop(key) for key in keys
                                if key in kwargs}

        # Initialize the base class
        super(KalmanSmoother, self).__init__(
            k_endog, k_states, k_posdef, results_class=results_class, **kwargs
        )

        # Options
        self.prefix_kalman_smoother_map = (
            kalman_smoother_classes
            if kalman_smoother_classes is not None
            else tools.prefix_kalman_smoother_map.copy())

        # Setup the underlying Kalman smoother storage
        self._kalman_smoothers = {}

        # Set the smoother options
        self.set_smoother_output(**smoother_output_kwargs)
        self.set_smooth_method(**smooth_method_kwargs)

    def _clone_kwargs(self, endog, **kwargs):
        # See Representation._clone_kwargs for docstring
        kwargs = super(KalmanSmoother, self)._clone_kwargs(endog, **kwargs)

        # Get defaults for options
        kwargs.setdefault('smoother_output', self.smoother_output)
        kwargs.setdefault('smooth_method', self.smooth_method)

        return kwargs

    @property
    def _kalman_smoother(self):
        prefix = self.prefix
        if prefix in self._kalman_smoothers:
            return self._kalman_smoothers[prefix]
        return None

    def _initialize_smoother(self, smoother_output=None, smooth_method=None,
                             prefix=None, **kwargs):
        if smoother_output is None:
            smoother_output = self.smoother_output
        if smooth_method is None:
            smooth_method = self.smooth_method

        # Make sure we have the required Kalman filter
        prefix, dtype, create_filter, create_statespace = (
            self._initialize_filter(prefix, **kwargs)
        )

        # Determine if we need to (re-)create the smoother
        # (definitely need to recreate if we recreated the filter)
        create_smoother = (create_filter or
                           prefix not in self._kalman_smoothers)
        if not create_smoother:
            kalman_smoother = self._kalman_smoothers[prefix]

            create_smoother = (kalman_smoother.kfilter is not
                               self._kalman_filters[prefix])

        # If the dtype-specific _kalman_smoother does not exist (or if we
        # need to re-create it), create it
        if create_smoother:
            # Setup the smoother
            cls = self.prefix_kalman_smoother_map[prefix]
            self._kalman_smoothers[prefix] = cls(
                self._statespaces[prefix], self._kalman_filters[prefix],
                smoother_output, smooth_method
            )
        # Otherwise, update the smoother parameters
        else:
            self._kalman_smoothers[prefix].set_smoother_output(
                smoother_output, False)
            self._kalman_smoothers[prefix].set_smooth_method(smooth_method)

        return prefix, dtype, create_smoother, create_filter, create_statespace

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
            setting individual boolean flags. See notes for details.

        Notes
        -----
        The smoother output is defined by a collection of boolean flags, and
        is internally stored as a bitmask. The methods available are:

        SMOOTHER_STATE = 0x01
            Calculate and return the smoothed states.
        SMOOTHER_STATE_COV = 0x02
            Calculate and return the smoothed state covariance matrices.
        SMOOTHER_STATE_AUTOCOV = 0x10
            Calculate and return the smoothed state lag-one autocovariance
            matrices.
        SMOOTHER_DISTURBANCE = 0x04
            Calculate and return the smoothed state and observation
            disturbances.
        SMOOTHER_DISTURBANCE_COV = 0x08
            Calculate and return the covariance matrices for the smoothed state
            and observation disturbances.
        SMOOTHER_ALL
            Calculate and return all results.

        If the bitmask is set directly via the `smoother_output` argument, then
        the full method must be provided.

        If keyword arguments are used to set individual boolean flags, then
        the lowercase of the method must be used as an argument name, and the
        value is the desired value of the boolean flag (True or False).

        Note that the smoother output may also be specified by directly
        modifying the class attributes which are defined similarly to the
        keyword arguments.

        The default smoother output is SMOOTHER_ALL.

        If performance is a concern, only those results which are needed should
        be specified as any results that are not specified will not be
        calculated. For example, if the smoother output is set to only include
        SMOOTHER_STATE, the smoother operates much more quickly than if all
        output is required.

        Examples
        --------
        >>> import statsmodels.tsa.statespace.kalman_smoother as ks
        >>> mod = ks.KalmanSmoother(1,1)
        >>> mod.smoother_output
        15
        >>> mod.set_smoother_output(smoother_output=0)
        >>> mod.smoother_state = True
        >>> mod.smoother_output
        1
        >>> mod.smoother_state
        True
        """
        if smoother_output is not None:
            self.smoother_output = smoother_output
        for name in KalmanSmoother.smoother_outputs:
            if name in kwargs:
                setattr(self, name, kwargs[name])

    def set_smooth_method(self, smooth_method=None, **kwargs):
        r"""
        Set the smoothing method

        The smoothing method can be used to override the Kalman smoother
        approach used. By default, the Kalman smoother used depends on the
        Kalman filter method.

        Parameters
        ----------
        smooth_method : int, optional
            Bitmask value to set the filter method to. See notes for details.
        **kwargs
            Keyword arguments may be used to influence the filter method by
            setting individual boolean flags. See notes for details.

        Notes
        -----
        The smoothing method is defined by a collection of boolean flags, and
        is internally stored as a bitmask. The methods available are:

        SMOOTH_CONVENTIONAL = 0x01
            Default Kalman smoother, as presented in Durbin and Koopman, 2012
            chapter 4.
        SMOOTH_CLASSICAL = 0x02
            Classical Kalman smoother, as presented in Anderson and Moore, 1979
            or Durbin and Koopman, 2012 chapter 4.6.1.
        SMOOTH_ALTERNATIVE = 0x04
            Modified Bryson-Frazier Kalman smoother method; this is identical
            to the conventional method of Durbin and Koopman, 2012, except that
            an additional intermediate step is included.
        SMOOTH_UNIVARIATE = 0x08
            Univariate Kalman smoother, as presented in Durbin and Koopman,
            2012 chapter 6, except with modified Bryson-Frazier timing.

        Practically speaking, these methods should all produce the same output
        but different computational implications, numerical stability
        implications, or internal timing assumptions.

        Note that only the first method is available if using a Scipy version
        older than 0.16.

        If the bitmask is set directly via the `smooth_method` argument, then
        the full method must be provided.

        If keyword arguments are used to set individual boolean flags, then
        the lowercase of the method must be used as an argument name, and the
        value is the desired value of the boolean flag (True or False).

        Note that the filter method may also be specified by directly modifying
        the class attributes which are defined similarly to the keyword
        arguments.

        The default filtering method is SMOOTH_CONVENTIONAL.

        Examples
        --------
        >>> mod = sm.tsa.statespace.SARIMAX(range(10))
        >>> mod.smooth_method
        1
        >>> mod.filter_conventional
        True
        >>> mod.filter_univariate = True
        >>> mod.smooth_method
        17
        >>> mod.set_smooth_method(filter_univariate=False,
                                  filter_collapsed=True)
        >>> mod.smooth_method
        33
        >>> mod.set_smooth_method(smooth_method=1)
        >>> mod.filter_conventional
        True
        >>> mod.filter_univariate
        False
        >>> mod.filter_collapsed
        False
        >>> mod.filter_univariate = True
        >>> mod.smooth_method
        17
        """
        if smooth_method is not None:
            self.smooth_method = smooth_method
        for name in KalmanSmoother.smooth_methods:
            if name in kwargs:
                setattr(self, name, kwargs[name])

    def _smooth(self, smoother_output=None, smooth_method=None, prefix=None,
                complex_step=False, results=None, **kwargs):
        # Initialize the smoother
        prefix, dtype, create_smoother, create_filter, create_statespace = (
            self._initialize_smoother(
                smoother_output, smooth_method, prefix=prefix, **kwargs
            ))

        # Check that the filter and statespace weren't just recreated
        if create_filter or create_statespace:
            raise ValueError('Passed settings forced re-creation of the'
                             ' Kalman filter. Please run `_filter` before'
                             ' running `_smooth`.')

        # Get the appropriate smoother
        smoother = self._kalman_smoothers[prefix]

        # Run the smoother
        smoother()

        return smoother

    def smooth(self, smoother_output=None, smooth_method=None, results=None,
               run_filter=True, prefix=None, complex_step=False,
               update_representation=True, update_filter=True,
               update_smoother=True, **kwargs):
        """
        Apply the Kalman smoother to the statespace model.

        Parameters
        ----------
        smoother_output : int, optional
            Determines which Kalman smoother output calculate. Default is all
            (including state, disturbances, and all covariances).
        results : class or object, optional
            If a class, then that class is instantiated and returned with the
            result of both filtering and smoothing.
            If an object, then that object is updated with the smoothing data.
            If None, then a SmootherResults object is returned with both
            filtering and smoothing results.
        run_filter : bool, optional
            Whether or not to run the Kalman filter prior to smoothing. Default
            is True.
        prefix : str
            The prefix of the datatype. Usually only used internally.

        Returns
        -------
        SmootherResults object
        """

        # Run the filter
        kfilter = self._filter(**kwargs)

        # Create the results object
        results = self.results_class(self)
        if update_representation:
            results.update_representation(self)
        if update_filter:
            results.update_filter(kfilter)
        else:
            # (even if we don't update all filter results, still need to
            # update this)
            results.nobs_diffuse = kfilter.nobs_diffuse

        # Run the smoother
        if smoother_output is None:
            smoother_output = self.smoother_output
        smoother = self._smooth(smoother_output, results=results, **kwargs)

        # Update the results
        if update_smoother:
            results.update_smoother(smoother)

        return results


class SmootherResults(FilterResults):
    r"""
    Results from applying the Kalman smoother and/or filter to a state space
    model.

    Parameters
    ----------
    model : Representation
        A Statespace representation

    Attributes
    ----------
    nobs : int
        Number of observations.
    k_endog : int
        The dimension of the observation series.
    k_states : int
        The dimension of the unobserved state process.
    k_posdef : int
        The dimension of a guaranteed positive definite covariance matrix
        describing the shocks in the measurement equation.
    dtype : dtype
        Datatype of representation matrices
    prefix : str
        BLAS prefix of representation matrices
    shapes : dictionary of name:tuple
        A dictionary recording the shapes of each of the representation
        matrices as tuples.
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
        An array of the same size as `endog`, filled with boolean values that
        are True if the corresponding entry in `endog` is NaN and False
        otherwise.
    nmissing : array of int
        An array of size `nobs`, where the ith entry is the number (between 0
        and k_endog) of NaNs in the ith row of the `endog` array.
    time_invariant : bool
        Whether or not the representation matrices are time-invariant
    initialization : str
        Kalman filter initialization method.
    initial_state : array_like
        The state vector used to initialize the Kalamn filter.
    initial_state_cov : array_like
        The state covariance matrix used to initialize the Kalamn filter.
    filter_method : int
        Bitmask representing the Kalman filtering method
    inversion_method : int
        Bitmask representing the method used to invert the forecast error
        covariance matrix.
    stability_method : int
        Bitmask representing the methods used to promote numerical stability in
        the Kalman filter recursions.
    conserve_memory : int
        Bitmask representing the selected memory conservation method.
    tolerance : float
        The tolerance at which the Kalman filter determines convergence to
        steady-state.
    loglikelihood_burn : int
        The number of initial periods during which the loglikelihood is not
        recorded.
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
    kalman_gain : ndarray
        The Kalman gain at each time period.
    forecasts : ndarray
        The one-step-ahead forecasts of observations at each time period.
    forecasts_error : ndarray
        The forecast errors at each time period.
    forecasts_error_cov : ndarray
        The forecast error covariance matrices at each time period.
    loglikelihood : ndarray
        The loglikelihood values at each time period.
    collapsed_forecasts : ndarray
        If filtering using collapsed observations, stores the one-step-ahead
        forecasts of collapsed observations at each time period.
    collapsed_forecasts_error : ndarray
        If filtering using collapsed observations, stores the one-step-ahead
        forecast errors of collapsed observations at each time period.
    collapsed_forecasts_error_cov : ndarray
        If filtering using collapsed observations, stores the one-step-ahead
        forecast error covariance matrices of collapsed observations at each
        time period.
    standardized_forecast_error : ndarray
        The standardized forecast errors
    smoother_output : int
        Bitmask representing the generated Kalman smoothing output
    scaled_smoothed_estimator : ndarray
        The scaled smoothed estimator at each time period.
    scaled_smoothed_estimator_cov : ndarray
        The scaled smoothed estimator covariance matrices at each time period.
    smoothing_error : ndarray
        The smoothing error covariance matrices at each time period.
    smoothed_state : ndarray
        The smoothed state at each time period.
    smoothed_state_cov : ndarray
        The smoothed state covariance matrices at each time period.
    smoothed_state_autocov : ndarray
        The smoothed state lago-one autocovariance matrices at each time
        period: :math:`Cov(\alpha_{t+1}, \alpha_t)`.
    smoothed_measurement_disturbance : ndarray
        The smoothed measurement at each time period.
    smoothed_state_disturbance : ndarray
        The smoothed state at each time period.
    smoothed_measurement_disturbance_cov : ndarray
        The smoothed measurement disturbance covariance matrices at each time
        period.
    smoothed_state_disturbance_cov : ndarray
        The smoothed state disturbance covariance matrices at each time period.
    """

    _smoother_attributes = [
        'smoother_output', 'scaled_smoothed_estimator',
        'scaled_smoothed_estimator_cov', 'smoothing_error',
        'smoothed_state', 'smoothed_state_cov', 'smoothed_state_autocov',
        'smoothed_measurement_disturbance', 'smoothed_state_disturbance',
        'smoothed_measurement_disturbance_cov',
        'smoothed_state_disturbance_cov', 'innovations_transition'
    ]

    _smoother_options = KalmanSmoother.smoother_outputs

    _attributes = FilterResults._model_attributes + _smoother_attributes

    def update_representation(self, model, only_options=False):
        """
        Update the results to match a given model

        Parameters
        ----------
        model : Representation
            The model object from which to take the updated values.
        only_options : bool, optional
            If set to true, only the smoother and filter options are updated,
            and the state space representation is not updated. Default is
            False.

        Notes
        -----
        This method is rarely required except for internal usage.
        """
        super(SmootherResults, self).update_representation(model, only_options)

        # Save the options as boolean variables
        for name in self._smoother_options:
            setattr(self, name, getattr(model, name, None))

        # Initialize holders for smoothed forecasts
        self._smoothed_forecasts = None
        self._smoothed_forecasts_error = None
        self._smoothed_forecasts_error_cov = None

    def update_smoother(self, smoother):
        """
        Update the smoother results

        Parameters
        ----------
        smoother : KalmanSmoother
            The model object from which to take the updated values.

        Notes
        -----
        This method is rarely required except for internal usage.
        """
        # Copy the appropriate output
        attributes = []

        # Since update_representation will already have been called, we can
        # use the boolean options smoother_* and know they match the smoother
        # itself
        if self.smoother_state or self.smoother_disturbance:
            attributes.append('scaled_smoothed_estimator')
        if self.smoother_state_cov or self.smoother_disturbance_cov:
            attributes.append('scaled_smoothed_estimator_cov')
        if self.smoother_state:
            attributes.append('smoothed_state')
        if self.smoother_state_cov:
            attributes.append('smoothed_state_cov')
        if self.smoother_state_autocov:
            attributes.append('smoothed_state_autocov')
        if self.smoother_disturbance:
            attributes += [
                'smoothing_error',
                'smoothed_measurement_disturbance',
                'smoothed_state_disturbance'
            ]
        if self.smoother_disturbance_cov:
            attributes += [
                'smoothed_measurement_disturbance_cov',
                'smoothed_state_disturbance_cov'
            ]

        has_missing = np.sum(self.nmissing) > 0
        for name in self._smoother_attributes:
            if name == 'smoother_output':
                pass
            elif name in attributes:
                if name in ['smoothing_error',
                            'smoothed_measurement_disturbance']:
                    vector = getattr(smoother, name, None)
                    if vector is not None and has_missing:
                        vector = np.array(reorder_missing_vector(
                            vector, self.missing, prefix=self.prefix))
                    else:
                        vector = np.array(vector, copy=True)
                    setattr(self, name, vector)
                elif name == 'smoothed_measurement_disturbance_cov':
                    matrix = getattr(smoother, name, None)
                    if matrix is not None and has_missing:
                        matrix = reorder_missing_matrix(
                            matrix, self.missing, reorder_rows=True,
                            reorder_cols=True, prefix=self.prefix)
                        # In the missing data case, we want to set the missing
                        # components equal to their unconditional distribution
                        copy_index_matrix(
                            self.obs_cov, matrix, self.missing,
                            index_rows=True, index_cols=True, inplace=True,
                            prefix=self.prefix)
                    else:
                        matrix = np.array(matrix, copy=True)
                    setattr(self, name, matrix)
                else:
                    setattr(self, name,
                            np.array(getattr(smoother, name, None), copy=True))
            else:
                setattr(self, name, None)

        self.innovations_transition = (
            np.array(smoother.innovations_transition, copy=True))

        # Diffuse objects
        self.scaled_smoothed_diffuse_estimator = None
        self.scaled_smoothed_diffuse1_estimator_cov = None
        self.scaled_smoothed_diffuse2_estimator_cov = None
        if self.nobs_diffuse > 0:
            self.scaled_smoothed_diffuse_estimator = np.array(
                smoother.scaled_smoothed_diffuse_estimator, copy=True)
            self.scaled_smoothed_diffuse1_estimator_cov = np.array(
                smoother.scaled_smoothed_diffuse1_estimator_cov, copy=True)
            self.scaled_smoothed_diffuse2_estimator_cov = np.array(
                smoother.scaled_smoothed_diffuse2_estimator_cov, copy=True)

        # Adjustments

        # For r_t (and similarly for N_t), what was calculated was
        # r_T, ..., r_{-1}. We only want r_0, ..., r_T
        # so exclude the appropriate element so that the time index is
        # consistent with the other returned output
        # r_t stored such that scaled_smoothed_estimator[0] == r_{-1}
        start = 1
        end = None
        if 'scaled_smoothed_estimator' in attributes:
            self.scaled_smoothed_estimator_presample = (
                self.scaled_smoothed_estimator[:, 0])
            self.scaled_smoothed_estimator = (
                self.scaled_smoothed_estimator[:, start:end]
            )
        if 'scaled_smoothed_estimator_cov' in attributes:
            self.scaled_smoothed_estimator_cov_presample = (
                self.scaled_smoothed_estimator_cov[:, :, 0])
            self.scaled_smoothed_estimator_cov = (
                self.scaled_smoothed_estimator_cov[:, :, start:end]
            )

        # Clear the smoothed forecasts
        self._smoothed_forecasts = None
        self._smoothed_forecasts_error = None
        self._smoothed_forecasts_error_cov = None

        # Note: if we concentrated out the scale, need to adjust the
        # loglikelihood values and all of the covariance matrices and the
        # values that depend on the covariance matrices
        if self.filter_concentrated and self.model._scale is None:
            self.smoothed_state_cov *= self.scale
            self.smoothed_state_autocov *= self.scale
            self.smoothed_state_disturbance_cov *= self.scale
            self.smoothed_measurement_disturbance_cov *= self.scale
            self.scaled_smoothed_estimator_presample /= self.scale
            self.scaled_smoothed_estimator /= self.scale
            self.scaled_smoothed_estimator_cov_presample /= self.scale
            self.scaled_smoothed_estimator_cov /= self.scale
            self.smoothing_error /= self.scale

        # Cache
        self.__smoothed_state_autocovariance = {}

    def _smoothed_state_autocovariance(self, shift, start, end,
                                       extend_kwargs=None):
        """
        Compute "forward" autocovariances, Cov(t, t+j)

        Parameters
        ----------
        shift : int
            The number of period to shift forwards when computing the
            autocovariance. This has the opposite sign as `lag` from the
            `smoothed_state_autocovariance` method.
        start : int, optional
            The start of the interval (inclusive) of autocovariances to compute
            and return.
        end : int, optional
            The end of the interval (exclusive) autocovariances to compute and
            return. Note that since it is an exclusive endpoint, the returned
            autocovariances do not include the value at this index.
        extend_kwargs : dict, optional
            Keyword arguments containing updated state space system matrices
            for handling out-of-sample autocovariance computations in
            time-varying state space models.

        """
        if extend_kwargs is None:
            extend_kwargs = {}

        # Size of returned array in the time dimension
        n = end - start

        # Get number of post-sample periods we need to create an extended
        # model to compute
        if shift == 0:
            max_insample = self.nobs - shift
        else:
            max_insample = self.nobs - shift + 1
        n_postsample = max(0, end - max_insample)

        # Get full in-sample arrays
        if shift != 0:
            L = self.innovations_transition
            P = self.predicted_state_cov
            N = self.scaled_smoothed_estimator_cov
        else:
            acov = self.smoothed_state_cov

        # If applicable, append out-of-sample arrays
        if n_postsample > 0:
            # Note: we need 1 less than the number of post
            endog = np.zeros((n_postsample, self.k_endog)) * np.nan
            mod = self.model.extend(endog, start=self.nobs, **extend_kwargs)
            mod.initialize_known(self.predicted_state[..., self.nobs],
                                 self.predicted_state_cov[..., self.nobs])
            res = mod.smooth()

            if shift != 0:
                start_insample = max(0, start)
                L = np.concatenate((L[..., start_insample:],
                                    res.innovations_transition), axis=2)
                P = np.concatenate((P[..., start_insample:],
                                    res.predicted_state_cov[..., 1:]),
                                   axis=2)
                N = np.concatenate((N[..., start_insample:],
                                    res.scaled_smoothed_estimator_cov),
                                   axis=2)
                end -= start_insample
                start -= start_insample
            else:
                acov = np.concatenate((acov, res.predicted_state_cov), axis=2)

        if shift != 0:
            # Subset to appropriate start, end
            start_insample = max(0, start)
            LT = L[..., start_insample:end + shift - 1].T
            P = P[..., start_insample:end + shift].T
            N = N[..., start_insample:end + shift - 1].T

            # Intermediate computations
            tmpLT = np.eye(self.k_states)[None, :, :]
            length = P.shape[0] - shift  # this is the required length of LT
            for i in range(1, shift + 1):
                tmpLT = LT[shift - i:length + shift - i] @ tmpLT
            eye = np.eye(self.k_states)[None, ...]

            # Compute the autocovariance
            acov = np.zeros((n, self.k_states, self.k_states))
            acov[:start_insample - start] = np.nan
            acov[start_insample - start:] = (
                P[:-shift] @ tmpLT @ (eye - N[shift - 1:] @ P[shift:]))
        else:
            acov = acov.T[start:end]

        return acov

    def smoothed_state_autocovariance(self, lag=1, t=None, start=None,
                                      end=None, extend_kwargs=None):
        r"""
        Compute state vector autocovariances, conditional on the full dataset

        Computes:

        .. math::

            Cov(\alpha_t - \hat \alpha_t, \alpha_{t - j} - \hat \alpha_{t - j})

        where the `lag` argument gives the value for :math:`j`. Thus when
        the `lag` argument is positive, the autocovariance is between the
        current and previous periods, while if `lag` is negative the
        autocovariance is between the current and future periods.

        Parameters
        ----------
        lag : int, optional
            The number of period to shift when computing the autocovariance.
            Default is 1.
        t : int, optional
            A specific period for which to compute and return the
            autocovariance. Cannot be used in combination with `start` or
            `end`. See the Returns section for details on how this
            parameter affects what is what is returned.
        start : int, optional
            The start of the interval (inclusive) of autocovariances to compute
            and return. Cannot be used in combination with the `t` argument.
            See the Returns section for details on how this parameter affects
            what is what is returned. Default is 0.
        end : int, optional
            The end of the interval (exclusive) autocovariances to compute and
            return. Note that since it is an exclusive endpoint, the returned
            autocovariances do not include the value at this index. Cannot be
            used in combination with the `t` argument. See the Returns section
            for details on how this parameter affects what is what is returned
            and what the default value is.
        extend_kwargs : dict, optional
            Keyword arguments containing updated state space system matrices
            for handling out-of-sample autocovariance computations in
            time-varying state space models.

        Returns
        -------
        acov : ndarray
            Array of autocovariance matrices. If the argument `t` is not
            provided, then it is shaped `(k_states, k_states, n)`, while if `t`
            given then the third axis is dropped and the array is shaped
            `(k_states, k_states)`.

            The output under the default case differs somewhat based on the
            state space model and the sign of the lag. To see how these cases
            differ, denote the output at each time point as Cov(t, t-j). Then:

            - If `lag > 0` (and the model is either time-varying or
              time-invariant), then the returned array is shaped `(*, *, nobs)`
              and each entry [:, :, t] contains Cov(t, t-j). However, the model
              does not have enough information to compute autocovariances in
              the pre-sample period, so that we cannot compute Cov(1, 1-lag),
              Cov(2, 2-lag), ..., Cov(lag, 0). Thus the first `lag` entries
              have all values set to NaN.

            - If the model is time-invariant and `lag < -1` or if `lag` is
              0 or -1, and the model is either time-invariant or time-varying,
              then the returned array is shaped `(*, *, nobs)` and each
              entry [:, :, t] contains Cov(t, t+j). Moreover, all entries are
              available (i.e. there are no NaNs).

            - If the model is time-varying and `lag < -1` and `extend_kwargs`
              is not provided, then the returned array is shaped
              `(*, *, nobs - lag + 1)`.

            - However, if the model is time-varying and `lag < -1`, then
              `extend_kwargs` can be provided with `lag - 1` additional
              matrices so that the returned array is shaped `(*, *, nobs)` as
              usual.

            More generally, the dimension of the last axis will be
            `start - end`.

        Notes
        -----
        This method computes:

        .. math::

            Cov(\alpha_t - \hat \alpha_t, \alpha_{t - j} - \hat \alpha_{t - j})

        where the `lag` argument determines the autocovariance order :math:`j`,
        and `lag` is an integer (positive, zero, or negative). This method
        cannot compute values associated with time points prior to the sample,
        and so it returns a matrix of NaN values for these time points.
        For example, if `start=0` and `lag=2`, then assuming the output is
        assigned to the variable `acov`, we will have `acov[..., 0]` and
        `acov[..., 1]` as matrices filled with NaN values.

        Based only on the "current" results object (i.e. the Kalman smoother
        applied to the sample), there is not enough information to compute
        Cov(t, t+j) for the last `lag - 1` observations of the sample. However,
        the values can be computed for these time points using the transition
        equation of the state space representation, and so for time-invariant
        state space models we do compute these values. For time-varying models,
        this can also be done, but updated state space matrices for the
        out-of-sample time points must be provided via the `extend_kwargs`
        argument.

        See [1]_, Chapter 4.7, for all details about how these autocovariances
        are computed.

        The `t` and `start`/`end` parameters compute and return only the
        requested autocovariances. As a result, using these parameters is
        recommended to reduce the computational burden, particularly if the
        number of observations and/or the dimension of the state vector is
        large.

        References
        ----------
        .. [1] Durbin, James, and Siem Jan Koopman. 2012.
               Time Series Analysis by State Space Methods: Second Edition.
               Oxford University Press.
        """
        # We can cache the results for time-invariant models
        cache_key = None
        if extend_kwargs is None or len(extend_kwargs) == 0:
            cache_key = (lag, t, start, end)

        # Short-circuit for a cache-hit
        if (cache_key is not None and
                cache_key in self.__smoothed_state_autocovariance):
            return self.__smoothed_state_autocovariance[cache_key]

        # Switch to only positive values for `lag`
        forward_autocovariances = False
        if lag < 0:
            lag = -lag
            forward_autocovariances = True

        # Handle `t`
        if t is not None and (start is not None or end is not None):
            raise ValueError('Cannot specify both `t` and `start` or `end`.')
        if t is not None:
            start = t
            end = t + 1

        # Defaults
        if start is None:
            start = 0
        if end is None:
            if forward_autocovariances and lag > 1 and extend_kwargs is None:
                end = self.nobs - lag + 1
            else:
                end = self.nobs
        if extend_kwargs is None:
            extend_kwargs = {}

        # Sanity checks
        if start < 0 or end < 0:
            raise ValueError('Negative `t`, `start`, or `end` is not allowed.')
        if end < start:
            raise ValueError('`end` must be after `start`')
        if lag == 0 and self.smoothed_state_cov is None:
            raise RuntimeError('Cannot return smoothed state covariances'
                               ' if those values have not been computed by'
                               ' Kalman smoothing.')

        # We already have in-sample (+1 out-of-sample) smoothed covariances
        if lag == 0 and end <= self.nobs + 1:
            acov = self.smoothed_state_cov
            if end == self.nobs + 1:
                acov = np.concatenate(
                    (acov[..., start:], self.predicted_state_cov[..., -1:]),
                    axis=2).T
            else:
                acov = acov.T[start:end]
        # In-sample, we can compute up to Cov(T, T+1) or Cov(T+1, T) and down
        # to Cov(1, 2) or Cov(2, 1). So:
        # - For lag=1 we set Cov(1, 0) = np.nan and then can compute up to T-1
        #   in-sample values Cov(2, 1), ..., Cov(T, T-1) and the first
        #   out-of-sample value Cov(T+1, T)
        elif (lag == 1 and self.smoothed_state_autocov is not None and
                not forward_autocovariances and end <= self.nobs + 1):
            # nans = np.zeros((self.k_states, self.k_states, lag)) * np.nan
            # acov = np.concatenate((nans, self.smoothed_state_autocov),
            #                       axis=2).transpose(2, 0, 1)[start:end]
            if start == 0:
                nans = np.zeros((self.k_states, self.k_states, lag)) * np.nan
                acov = np.concatenate(
                    (nans, self.smoothed_state_autocov[..., :end - 1]),
                    axis=2)
            else:
                acov = self.smoothed_state_autocov[..., start - 1:end - 1]
            acov = acov.transpose(2, 0, 1)
        # - For lag=-1 we can compute T in-sample values, Cov(1, 2), ...,
        #   Cov(T, T+1) but we cannot compute the first out-of-sample value
        #   Cov(T+1, T+2).
        elif (lag == 1 and self.smoothed_state_autocov is not None and
                forward_autocovariances and end < self.nobs + 1):
            acov = self.smoothed_state_autocov.T[start:end]
        # Otherwise, we need to compute additional values at the end of the
        # sample
        else:
            if forward_autocovariances:
                # Cov(t, t + lag), t = start, ..., end
                acov = self._smoothed_state_autocovariance(
                    lag, start, end, extend_kwargs=extend_kwargs)
            else:
                # Cov(t, t + lag)' = Cov(t + lag, t),
                # with t = start - lag, ..., end - lag
                out = self._smoothed_state_autocovariance(
                    lag, start - lag, end - lag, extend_kwargs=extend_kwargs)
                acov = out.transpose(0, 2, 1)

        # Squeeze the last axis or else reshape to have the same axis
        # definitions as e.g. smoothed_state_cov
        if t is not None:
            acov = acov[0]
        else:
            acov = acov.transpose(1, 2, 0)

        # Fill in the cache, if applicable
        if cache_key is not None:
            self.__smoothed_state_autocovariance[cache_key] = acov

        return acov

    def news(self, previous, t=None, start=None, end=None,
             revisions_details_start=True, design=None, state_index=None):
        r"""
        Compute the news and impacts associated with a data release

        Parameters
        ----------
        previous : SmootherResults
            Prior results object relative to which to compute the news. This
            results object must have identical state space representation for
            the prior sample period so that the only difference is that this
            results object has updates to the observed data.
        t : int, optional
            A specific period for which to compute the news. Cannot be used in
            combination with `start` or `end`.
        start : int, optional
            The start of the interval (inclusive) of news to compute. Cannot be
            used in combination with the `t` argument. Default is the last
            period of the sample (`nobs - 1`).
        end : int, optional
            The end of the interval (exclusive) of news to compute. Note that
            since it is an exclusive endpoint, the returned news do not include
            the value at this index. Cannot be used in combination with the `t`
            argument.
        revisions_details_start : bool or int, optional
            The period at which to beging computing the detailed impacts of
            data revisions. Any revisions prior to this period will have their
            impacts grouped together. If a negative integer, interpreted as
            an offset from the end of the dataset. If set to True, detailed
            impacts are computed for all revisions, while if set to False, all
            revisions are grouped together. Default is False. Note that for
            large models, setting this to be near the beginning of the sample
            can cause this function to be slow.
        design : array, optional
            Design matrix for the period `t` in time-varying models. If this
            model has a time-varying design matrix, and the argument `t` is out
            of this model's sample, then a new design matrix for period `t`
            must be provided. Unused otherwise.
        state_index : array_like, optional
            An optional index specifying a subset of states to use when
            constructing the impacts of revisions and news. For example, if
            `state_index=[0, 1]` is passed, then only the impacts to the
            observed variables arising from the impacts to the first two
            states will be returned.

        Returns
        -------
        news_results : SimpleNamespace
            News and impacts associated with a data release. Includes the
            following attributes:

            - `update_impacts`: update to forecasts of impacted variables from
              the news. It is equivalent to E[y^i | post] - E[y^i | revision],
              where y^i are the variables of interest. In [1]_, this is
              described as "revision" in equation (17).
            - `revision_detailed_impacts`: update to forecasts of variables
              impacted variables from data revisions. It is
              E[y^i | revision] - E[y^i | previous], and does not have a
              specific notation in [1]_, since there for simplicity they assume
              that there are no revisions.
            - `news`: the unexpected component of the updated data. Denoted
              I = y^u - E[y^u | previous], where y^u are the data points that
              were newly incorporated in a data release (but not including
              revisions to data points that already existed in the previous
              release). In [1]_, this is described as "news" in equation (17).
            - `revisions`: y^r(updated) - y^r(previous) for periods in
              which detailed impacts were computed
            - `revisions_all` : y^r(updated) - y^r(previous) for all revisions
            - `gain`: the gain matrix associated with the "Kalman-like" update
              from the news, E[y I'] E[I I']^{-1}. In [1]_, this can be found
              in the equation For E[y_{k,t_k} \mid I_{v+1}] in the middle of
              page 17.
            - `revision_weights` weights on observations for the smoothed
              signal
            - `update_forecasts`: forecasts of the updated periods used to
              construct the news, E[y^u | previous].
            - `update_realized`: realizations of the updated periods used to
              construct the news, y^u.
            - `revised`: revised observations of the periods that were revised
              and for which detailed impacts were computed
            - `revised`: revised observations of the periods that were revised
            - `revised_prev`: previous observations of the periods that were
              revised and for which detailed impacts were computed
            - `revised_prev_all`: previous observations of the periods that
              were revised and for which detailed impacts were computed
            - `prev_impacted_forecasts`: previous forecast of the periods of
              interest, E[y^i | previous].
            - `post_impacted_forecasts`: forecast of the periods of interest
              after taking into account both revisions and updates,
              E[y^i | post].
            - `revision_results`: results object that updates the `previous`
              results to take into account data revisions.
            - `revision_results`: results object associated with the revisions
            - `revision_impacts`: total impacts from all revisions (both
              grouped and detailed)
            - `revisions_ix`: list of `(t, i)` positions of revisions in endog
            - `revisions_details`: list of `(t, i)` positions of revisions to
              endog for which details of impacts were computed
            - `revisions_grouped`: list of `(t, i)` positions of revisions to
              endog for which impacts were grouped
            - `revisions_details_start`: period in which revision details start
              to be computed
            - `updates_ix`: list of `(t, i)` positions of updates to endog
            - `state_index`: index of state variables used to compute impacts

        Notes
        -----
        This method computes the effect of new data (e.g. from a new data
        release) on smoothed forecasts produced by a state space model, as
        described in [1]_. It also computes the effect of revised data on
        smoothed forecasts.

        References
        ----------
        .. [1] Bańbura, Marta and Modugno, Michele. 2010.
               "Maximum likelihood estimation of factor models on data sets
               with arbitrary pattern of missing data."
               No 1189, Working Paper Series, European Central Bank.
               https://EconPapers.repec.org/RePEc:ecb:ecbwps:20101189.
        .. [2] Bańbura, Marta, and Michele Modugno.
               "Maximum likelihood estimation of factor models on datasets with
               arbitrary pattern of missing data."
               Journal of Applied Econometrics 29, no. 1 (2014): 133-160.

        """
        # Handle `t`
        if t is not None and (start is not None or end is not None):
            raise ValueError('Cannot specify both `t` and `start` or `end`.')
        if t is not None:
            start = t
            end = t + 1

        # Defaults
        if start is None:
            start = self.nobs - 1
        if end is None:
            end = self.nobs

        # Sanity checks
        if start < 0 or end < 0:
            raise ValueError('Negative `t`, `start`, or `end` is not allowed.')
        if end <= start:
            raise ValueError('`end` must be after `start`')

        if self.smoothed_state_cov is None:
            raise ValueError('Cannot compute news without having applied the'
                             ' Kalman smoother first.')

        error_ss = ('This results object has %s and so it does not appear to'
                    ' by an extension of `previous`. Can only compute the'
                    ' news by comparing this results set to previous results'
                    ' objects.')
        if self.nobs < previous.nobs:
            raise ValueError(error_ss % 'fewer observations than'
                             ' `previous`')

        if not (self.k_endog == previous.k_endog and
                self.k_states == previous.k_states and
                self.k_posdef == previous.k_posdef):
            raise ValueError(error_ss % 'different state space dimensions than'
                             ' `previous`')

        for key in self.model.shapes.keys():
            if key == 'obs':
                continue
            tv = getattr(self, key).shape[-1] > 1
            tv_prev = getattr(previous, key).shape[-1] > 1
            if tv and not tv_prev:
                raise ValueError(error_ss % f'time-varying {key} while'
                                 ' `previous` does not')
            if not tv and tv_prev:
                raise ValueError(error_ss % f'time-invariant {key} while'
                                 ' `previous` does not')

        # Standardize
        if state_index is not None:
            state_index = np.atleast_1d(
                np.sort(np.array(state_index, dtype=int)))

        # We cannot forecast out-of-sample periods in a time-varying model
        if end > self.nobs and not self.model.time_invariant:
            raise RuntimeError('Cannot compute the impacts of news on periods'
                               ' outside of the sample in time-varying'
                               ' models.')

        # For time-varying case, figure out extension kwargs
        extend_kwargs = {}
        for key in self.model.shapes.keys():
            if key == 'obs':
                continue
            mat = getattr(self, key)
            prev_mat = getattr(previous, key)
            if mat.shape[-1] > prev_mat.shape[-1]:
                extend_kwargs[key] = mat[..., prev_mat.shape[-1]:]

        # Figure out which indices have changed
        revisions_ix, updates_ix = previous.model.diff_endog(self.endog.T)

        # Compute prev / post impact forecasts
        prev_impacted_forecasts = previous.predict(
            start=start, end=end, **extend_kwargs).smoothed_forecasts
        post_impacted_forecasts = self.predict(
            start=start, end=end).smoothed_forecasts

        # Separate revisions into those with detailed impacts and those where
        # impacts are grouped together
        if revisions_details_start is True:
            revisions_details_start = 0
        elif revisions_details_start is False:
            revisions_details_start = previous.nobs
        elif revisions_details_start < 0:
            revisions_details_start = previous.nobs + revisions_details_start

        revisions_grouped = []
        revisions_details = []
        if revisions_details_start > 0:
            for s, i in revisions_ix:
                if s < revisions_details_start:
                    revisions_grouped.append((s, i))
                else:
                    revisions_details.append((s, i))
        else:
            revisions_details = revisions_ix

        # Practically, don't compute impacts of revisions prior to first
        # point that was actually revised
        if len(revisions_ix) > 0:
            revisions_details_start = max(revisions_ix[0][0],
                                          revisions_details_start)

        # Setup default (empty) output for revisions
        revised_endog = None
        revised_all = None
        revised_prev_all = None
        revisions_all = None

        revised = None
        revised_prev = None
        revisions = None
        revision_weights = None
        revision_detailed_impacts = None
        revision_results = None
        revision_impacts = None

        # Get revisions datapoints for all revisions (regardless of whether
        # or not we are computing detailed impacts)
        if len(revisions_ix) > 0:
            # Indexes
            revised_j, revised_p = zip(*revisions_ix)
            compute_j = np.arange(revised_j[0], revised_j[-1] + 1)

            # Data from updated model
            revised_endog = self.endog[:, :previous.nobs].copy()
            # ("revisions" are points where data was previously published and
            # then changed, so we need to ignore "updates", which are points
            # that were not previously published)
            revised_endog[previous.missing.astype(bool)] = np.nan
            # subset to revision periods
            revised_all = revised_endog.T[compute_j]

            # Data from original model
            revised_prev_all = previous.endog.T[compute_j]

            # revision = updated - original
            revisions_all = (revised_all - revised_prev_all)

            # Construct a model from which we can create weights for impacts
            # through `end`
            # Construct endog for the new model
            tmp_endog = revised_endog.T.copy()
            tmp_nobs = max(end, previous.nobs)
            oos_nobs = tmp_nobs - previous.nobs
            if oos_nobs > 0:
                tmp_endog = np.concatenate([
                    tmp_endog, np.zeros((oos_nobs, self.k_endog)) * np.nan
                ], axis=0)

            # Copy time-varying matrices (required by clone)
            clone_kwargs = {}
            for key in self.model.shapes.keys():
                if key == 'obs':
                    continue
                mat = getattr(self, key)
                if mat.shape[-1] > 1:
                    clone_kwargs[key] = mat[..., :tmp_nobs]

            rev_mod = previous.model.clone(tmp_endog, **clone_kwargs)
            init = initialization.Initialization.from_results(self)
            rev_mod.initialize(init)
            revision_results = rev_mod.smooth()

            # Get detailed revision weights, impacts, and forecasts
            if len(revisions_details) > 0:
                # Indexes for the subset of revisions for which we are
                # computing detailed impacts
                compute_j = np.arange(revisions_details_start,
                                      revised_j[-1] + 1)
                # Offset describing revisions for which we are not computing
                # detailed impacts
                offset = revisions_details_start - revised_j[0]
                revised = revised_all[offset:]
                revised_prev = revised_prev_all[offset:]
                revisions = revisions_all[offset:]

                # Compute the weights of the smoothed state vector
                compute_t = np.arange(start, end)

                smoothed_state_weights, _, _ = (
                    tools._compute_smoothed_state_weights(
                        rev_mod, compute_t=compute_t, compute_j=compute_j,
                        compute_prior_weights=False, scale=previous.scale))

                # Convert the weights in terms of smoothed forecasts
                # t, j, m, p, i
                ZT = rev_mod.design.T
                if ZT.shape[0] > 1:
                    ZT = ZT[compute_t]

                # Subset the states used for the impacts if applicable
                if state_index is not None:
                    ZT = ZT[:, state_index, :]
                    smoothed_state_weights = (
                        smoothed_state_weights[:, :, state_index])

                # Multiplication gives: t, j, m, p * t, j, m, p, k
                # Sum along axis=2 gives: t, j, p, k
                # Transpose to: t, j, k, p (i.e. like t, j, m, p but with k
                # instead of m)
                revision_weights = np.nansum(
                    smoothed_state_weights[..., None]
                    * ZT[:, None, :, None, :], axis=2).transpose(0, 1, 3, 2)

                # Multiplication gives: t, j, k, p * t, j, k, p
                # Sum along axes 1, 3 gives: t, k
                # This is also a valid way to compute impacts, but it employs
                # unnecessary multiplications with zeros; it is better to use
                # the below method that flattens the revision indices before
                # computing the impacts
                # revision_detailed_impacts = np.nansum(
                #     revision_weights * revisions[None, :, None, :],
                #     axis=(1, 3))

                # Flatten the weights and revisions along the revised j, k
                # dimensions so that we only retain the actual revision
                # elements
                revised_j, revised_p = zip(*[
                    s for s in revisions_ix
                    if s[0] >= revisions_details_start])
                ix_j = revised_j - revised_j[0]
                # Shape is: t, k, j * p
                # Note: have to transpose first so that the two advanced
                # indexes are next to each other, so that "the dimensions from
                # the advanced indexing operations are inserted into the result
                # array at the same spot as they were in the initial array"
                # (see https://numpy.org/doc/stable/user/basics.indexing.html,
                # "Combining advanced and basic indexing")
                revision_weights = (
                    revision_weights.transpose(0, 2, 1, 3)[:, :,
                                                           ix_j, revised_p])
                # Shape is j * k
                revisions = revisions[ix_j, revised_p]
                # Shape is t, k
                revision_detailed_impacts = revision_weights @ revisions

                # Similarly, flatten the revised and revised_prev series
                revised = revised[ix_j, revised_p]
                revised_prev = revised_prev[ix_j, revised_p]

                # Squeeze if `t` argument used
                if t is not None:
                    revision_weights = revision_weights[0]
                    revision_detailed_impacts = revision_detailed_impacts[0]

            # Get total revision impacts
            revised_impact_forecasts = (
                revision_results.smoothed_forecasts[..., start:end])
            if end > revision_results.nobs:
                predict_start = max(start, revision_results.nobs)
                p = revision_results.predict(
                    start=predict_start, end=end, **extend_kwargs)
                revised_impact_forecasts = np.concatenate(
                    (revised_impact_forecasts, p.forecasts), axis=1)

            revision_impacts = (revised_impact_forecasts -
                                prev_impacted_forecasts).T
            if t is not None:
                revision_impacts = revision_impacts[0]

        # Need to also flatten the revisions items that contain all revisions
        if len(revisions_ix) > 0:
            revised_j, revised_p = zip(*revisions_ix)
            ix_j = revised_j - revised_j[0]

            revisions_all = revisions_all[ix_j, revised_p]
            revised_all = revised_all[ix_j, revised_p]
            revised_prev_all = revised_prev_all[ix_j, revised_p]

        # Now handle updates
        if len(updates_ix) > 0:
            # Figure out which time points we need forecast errors for
            update_t, update_k = zip(*updates_ix)
            update_start_t = np.min(update_t)
            update_end_t = np.max(update_t)

            if revision_results is None:
                forecasts = previous.predict(
                    start=update_start_t, end=update_end_t + 1,
                    **extend_kwargs).smoothed_forecasts.T
            else:
                forecasts = revision_results.predict(
                    start=update_start_t,
                    end=update_end_t + 1).smoothed_forecasts.T
            realized = self.endog.T[update_start_t:update_end_t + 1]
            forecasts_error = realized - forecasts

            # Now subset forecast errors to only the (time, endog) elements
            # that are updates
            ix_t = update_t - update_start_t
            update_realized = realized[ix_t, update_k]
            update_forecasts = forecasts[ix_t, update_k]
            update_forecasts_error = forecasts_error[ix_t, update_k]

            # Get the gains associated with each of the periods
            if self.design.shape[2] == 1:
                design = self.design[..., 0][None, ...]
            elif end <= self.nobs:
                design = self.design[..., start:end].transpose(2, 0, 1)
            else:
                # Note: this case is no longer possible, since above we raise
                # ValueError for time-varying case with end > self.nobs
                if design is None:
                    raise ValueError('Model has time-varying design matrix, so'
                                     ' an updated time-varying matrix for'
                                     ' period `t` is required.')
                elif design.ndim == 2:
                    design = design[None, ...]
                else:
                    design = design.transpose(2, 0, 1)

            state_gain = previous.smoothed_state_gain(
                updates_ix, start=start, end=end, extend_kwargs=extend_kwargs)

            # Subset the states used for the impacts if applicable
            if state_index is not None:
                design = design[:, :, state_index]
                state_gain = state_gain[:, state_index]

            # Compute the gain in terms of observed variables
            obs_gain = design @ state_gain

            # Get the news
            update_impacts = obs_gain @ update_forecasts_error

            # Squeeze if `t` argument used
            if t is not None:
                obs_gain = obs_gain[0]
                update_impacts = update_impacts[0]
        else:
            update_impacts = None
            update_forecasts = None
            update_realized = None
            update_forecasts_error = None
            obs_gain = None

        # Results
        out = SimpleNamespace(
            # update to forecast of impacted variables from news
            # = E[y^i | post] - E[y^i | revision] = weight @ news
            update_impacts=update_impacts,
            # update to forecast of variables of interest from revisions
            # = E[y^i | revision] - E[y^i | previous]
            revision_detailed_impacts=revision_detailed_impacts,
            # news = A = y^u - E[y^u | previous]
            news=update_forecasts_error,
            # revivions y^r(updated) - y^r(previous) for periods in which
            # detailed impacts were computed
            revisions=revisions,
            # revivions y^r(updated) - y^r(previous)
            revisions_all=revisions_all,
            # gain matrix = E[y A'] E[A A']^{-1}
            gain=obs_gain,
            # weights on observations for the smoothed signal
            revision_weights=revision_weights,
            # forecasts of the updated periods used to construct the news
            # = E[y^u | revised]
            update_forecasts=update_forecasts,
            # realizations of the updated periods used to construct the news
            # = y^u
            update_realized=update_realized,
            # revised observations of the periods that were revised and for
            # which detailed impacts were computed
            # = y^r_{revised}
            revised=revised,
            # revised observations of the periods that were revised
            # = y^r_{revised}
            revised_all=revised_all,
            # previous observations of the periods that were revised and for
            # which detailed impacts were computed
            # = y^r_{previous}
            revised_prev=revised_prev,
            # previous observations of the periods that were revised
            # = y^r_{previous}
            revised_prev_all=revised_prev_all,
            # previous forecast of the periods of interest, E[y^i | previous]
            prev_impacted_forecasts=prev_impacted_forecasts,
            # post. forecast of the periods of interest, E[y^i | post]
            post_impacted_forecasts=post_impacted_forecasts,
            # results object associated with the revision
            revision_results=revision_results,
            # total impacts from all revisions (both grouped and detailed)
            revision_impacts=revision_impacts,
            # list of (x, y) positions of revisions to endog
            revisions_ix=revisions_ix,
            # list of (x, y) positions of revisions to endog for which details
            # of impacts were computed
            revisions_details=revisions_details,
            # list of (x, y) positions of revisions to endog for which impacts
            # were grouped
            revisions_grouped=revisions_grouped,
            # period in which revision details start to be computed
            revisions_details_start=revisions_details_start,
            # list of (x, y) positions of updates to endog
            updates_ix=updates_ix,
            # index of state variables used to compute impacts
            state_index=state_index)

        return out

    def smoothed_state_gain(self, updates_ix, t=None, start=None,
                            end=None, extend_kwargs=None):
        r"""
        Cov(\tilde \alpha_{t}, I) Var(I, I)^{-1}

        where I is a vector of forecast errors associated with
        `update_indices`.

        Parameters
        ----------
        updates_ix : list
            List of indices `(t, i)`, where `t` denotes a zero-indexed time
            location and `i` denotes a zero-indexed endog variable.
        """
        # Handle `t`
        if t is not None and (start is not None or end is not None):
            raise ValueError('Cannot specify both `t` and `start` or `end`.')
        if t is not None:
            start = t
            end = t + 1

        # Defaults
        if start is None:
            start = self.nobs - 1
        if end is None:
            end = self.nobs
        if extend_kwargs is None:
            extend_kwargs = {}

        # Sanity checks
        if start < 0 or end < 0:
            raise ValueError('Negative `t`, `start`, or `end` is not allowed.')
        if end <= start:
            raise ValueError('`end` must be after `start`')

        # Dimensions
        n_periods = end - start
        n_updates = len(updates_ix)

        # Helper to get possibly matrix that is possibly time-varying
        def get_mat(which, t):
            mat = getattr(self, which)
            if mat.shape[-1] > 1:
                if t < self.nobs:
                    out = mat[..., t]
                else:
                    if (which not in extend_kwargs or
                            extend_kwargs[which].shape[-1] <= t - self.nobs):
                        raise ValueError(f'Model has time-varying {which}'
                                         ' matrix, so an updated time-varying'
                                         ' matrix for the extension period is'
                                         ' required.')
                    out = extend_kwargs[which][..., t - self.nobs]
            else:
                out = mat[..., 0]
            return out

        # Helper to get Cov(\tilde \alpha_{t}, I)
        def get_cov_state_revision(t):
            tmp1 = np.zeros((self.k_states, n_updates))
            for i in range(n_updates):
                t_i, k_i = updates_ix[i]
                acov = self.smoothed_state_autocovariance(
                    lag=t - t_i, t=t, extend_kwargs=extend_kwargs)
                Z_i = get_mat('design', t_i)
                tmp1[:, i:i + 1] = acov @ Z_i[k_i:k_i + 1].T
            return tmp1

        # Compute Cov(\tilde \alpha_{t}, I)
        tmp1 = np.zeros((n_periods, self.k_states, n_updates))
        for s in range(start, end):
            tmp1[s - start] = get_cov_state_revision(s)

        # Compute Var(I)
        tmp2 = np.zeros((n_updates, n_updates))
        for i in range(n_updates):
            t_i, k_i = updates_ix[i]
            for j in range(i + 1):
                t_j, k_j = updates_ix[j]

                Z_i = get_mat('design', t_i)
                Z_j = get_mat('design', t_j)

                acov = self.smoothed_state_autocovariance(
                    lag=t_i - t_j, t=t_i, extend_kwargs=extend_kwargs)
                tmp2[i, j] = tmp2[j, i] = np.squeeze(
                    Z_i[k_i:k_i + 1] @ acov @ Z_j[k_j:k_j + 1].T
                )

                if t_i == t_j:
                    H = get_mat('obs_cov', t_i)

                    if i == j:
                        tmp2[i, j] += H[k_i, k_j]
                    else:
                        tmp2[i, j] += H[k_i, k_j]
                        tmp2[j, i] += H[k_i, k_j]

        # Gain
        gain = tmp1 @ np.linalg.inv(tmp2)

        if t is not None:
            gain = gain[0]

        return gain

    def _get_smoothed_forecasts(self):
        if self._smoothed_forecasts is None:
            # Initialize empty arrays
            self._smoothed_forecasts = np.zeros(self.forecasts.shape,
                                                dtype=self.dtype)
            self._smoothed_forecasts_error = (
                np.zeros(self.forecasts_error.shape, dtype=self.dtype)
            )
            self._smoothed_forecasts_error_cov = (
                np.zeros(self.forecasts_error_cov.shape, dtype=self.dtype)
            )

            for t in range(self.nobs):
                design_t = 0 if self.design.shape[2] == 1 else t
                obs_cov_t = 0 if self.obs_cov.shape[2] == 1 else t
                obs_intercept_t = 0 if self.obs_intercept.shape[1] == 1 else t

                mask = ~self.missing[:, t].astype(bool)
                # We can recover forecasts
                self._smoothed_forecasts[:, t] = np.dot(
                    self.design[:, :, design_t], self.smoothed_state[:, t]
                ) + self.obs_intercept[:, obs_intercept_t]
                if self.nmissing[t] > 0:
                    self._smoothed_forecasts_error[:, t] = np.nan
                self._smoothed_forecasts_error[mask, t] = (
                    self.endog[mask, t] - self._smoothed_forecasts[mask, t]
                )
                self._smoothed_forecasts_error_cov[:, :, t] = np.dot(
                    np.dot(self.design[:, :, design_t],
                           self.smoothed_state_cov[:, :, t]),
                    self.design[:, :, design_t].T
                ) + self.obs_cov[:, :, obs_cov_t]

        return (
            self._smoothed_forecasts,
            self._smoothed_forecasts_error,
            self._smoothed_forecasts_error_cov
        )

    @property
    def smoothed_forecasts(self):
        return self._get_smoothed_forecasts()[0]

    @property
    def smoothed_forecasts_error(self):
        return self._get_smoothed_forecasts()[1]

    @property
    def smoothed_forecasts_error_cov(self):
        return self._get_smoothed_forecasts()[2]

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
        data_contributions : array
            Contributions of observations to the decomposed object. If the
            smoothed state is being decomposed, then `data_contributions` are
            shaped `(nobs, k_states, nobs, k_endog)`, where the
            `(t, m, j, p)`-th element is the contribution of the `p`-th
            observation at time `j` to the `m`-th state at time `t`. If the
            smoothed signal is being decomposed, then `data_contributions` are
            shaped `(nobs, k_endog, nobs, k_endog)`, where the
            `(t, k, j, p)`-th element is the contribution of the `p`-th
            observation at time `j` to the smoothed prediction of the `k`-th
            observation at time `t`.
        obs_intercept_contributions : array
            Contributions of the observation intercept to the decomposed
            object. If the smoothed state is being decomposed, then
            `obs_intercept_contributions` are shaped
            `(nobs, k_states, nobs, k_endog)`, where the `(t, m, j, p)`-th
            element is the contribution of the `p`-th observation intercept at
            time `j` to the `m`-th state at time `t`. If the smoothed signal
            is being decomposed, then `obs_intercept_contributions` are shaped
            `(nobs, k_endog, nobs, k_endog)`, where the `(t, k, j, p)`-th
            element is the contribution of the `p`-th observation at time `j`
            to the smoothed prediction of the `k`-th observation at time `t`.
        state_intercept_contributions : array
            Contributions of the state intercept to the decomposed object. If
            the smoothed state is being decomposed, then
            `state_intercept_contributions` are shaped
            `(nobs, k_states, nobs, k_states)`, where the `(t, m, j, l)`-th
            element is the contribution of the `l`-th state intercept at
            time `j` to the `m`-th state at time `t`. If the smoothed signal
            is being decomposed, then `state_intercept_contributions` are
            shaped `(nobs, k_endog, nobs, k_endog)`, where the
            `(t, k, j, l)`-th element is the contribution of the `p`-th
            observation at time `j` to the smoothed prediction of the `k`-th
            observation at time `t`.
        prior_contributions : array
            Contributions of the prior to the decomposed object. If the
            smoothed state is being decomposed, then `prior_contributions` are
            shaped `(nobs, k_states, k_states)`, where the `(t, m, l)`-th
            element is the contribution of the `l`-th element of the prior
            mean to the `m`-th state at time `t`. If the smoothed signal is
            being decomposed, then `prior_contributions` are shaped
            `(nobs, k_endog, k_states)`, where the `(t, k, l)`-th
            element is the contribution of the `l`-th element of the prior mean
            to the smoothed prediction of the `k`-th observation at time `t`.

        Notes
        -----
        Denote the smoothed state at time :math:`t` by :math:`\alpha_t`. Then
        the smoothed signal is :math:`Z_t \alpha_t`, where :math:`Z_t` is the
        design matrix operative at time :math:`t`.
        """
        if decomposition_of not in ['smoothed_state', 'smoothed_signal']:
            raise ValueError('Invalid value for `decomposition_of`. Must be'
                             ' one of "smoothed_state" or "smoothed_signal".')

        weights, state_intercept_weights, prior_weights = (
            tools._compute_smoothed_state_weights(
                self.model, compute_prior_weights=True, scale=self.scale))

        # Get state space objects
        ZT = self.model.design.T           # t, m, p
        dT = self.model.obs_intercept.T    # t, p
        cT = self.model.state_intercept.T  # t, m

        # Subset the states used for the impacts if applicable
        if decomposition_of == 'smoothed_signal' and state_index is not None:
            ZT = ZT[:, state_index, :]
            weights = weights[:, :, state_index]
            prior_weights = prior_weights[:, state_index, :]

        # Convert the weights in terms of smoothed signal
        # t, j, m, p, i
        if decomposition_of == 'smoothed_signal':
            # Multiplication gives: t, j, m, p * t, j, m, p, k
            # Sum along axis=2 gives: t, j, p, k
            # Transpose to: t, j, k, p (i.e. like t, j, m, p but with k instead
            # of m)
            weights = np.nansum(weights[..., None] * ZT[:, None, :, None, :],
                                axis=2).transpose(0, 1, 3, 2)

            # Multiplication gives: t, j, m, l * t, j, m, l, k
            # Sum along axis=2 gives: t, j, l, k
            # Transpose to: t, j, k, l (i.e. like t, j, m, p but with k instead
            # of m and l instead of p)
            state_intercept_weights = np.nansum(
                state_intercept_weights[..., None] * ZT[:, None, :, None, :],
                axis=2).transpose(0, 1, 3, 2)

            # Multiplication gives: t, m, l * t, m, l, k = t, m, l, k
            # Sum along axis=1 gives: t, l, k
            # Transpose to: t, k, l (i.e. like t, m, l but with k instead of m)
            prior_weights = np.nansum(
                prior_weights[..., None] * ZT[:, :, None, :],
                axis=1).transpose(0, 2, 1)

        # Contributions of observations: multiply weights by observations
        # Multiplication gives t, j, {m,k}, p
        data_contributions = weights * self.model.endog.T[None, :, None, :]
        # Transpose to: t, {m,k}, j, p
        data_contributions = data_contributions.transpose(0, 2, 1, 3)

        # Contributions of obs intercept: multiply data weights by obs
        # intercept
        # Multiplication gives t, j, {m,k}, p
        obs_intercept_contributions = -weights * dT[None, :, None, :]
        # Transpose to: t, {m,k}, j, p
        obs_intercept_contributions = (
            obs_intercept_contributions.transpose(0, 2, 1, 3))

        # Contributions of state intercept: multiply state intercept weights
        # by state intercept
        # Multiplication gives t, j, {m,k}, l
        state_intercept_contributions = (
            state_intercept_weights * cT[None, :, None, :])
        # Transpose to: t, {m,k}, j, l
        state_intercept_contributions = (
            state_intercept_contributions.transpose(0, 2, 1, 3))

        # Contributions of prior: multiply weights by prior
        # Multiplication gives t, {m, k}, l
        prior_contributions = prior_weights * self.initial_state[None, None, :]

        return (data_contributions, obs_intercept_contributions,
                state_intercept_contributions, prior_contributions)
