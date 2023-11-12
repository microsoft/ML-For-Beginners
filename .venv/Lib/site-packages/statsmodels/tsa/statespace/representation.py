"""
State Space Representation

Author: Chad Fulton
License: Simplified-BSD
"""

import warnings
import numpy as np
from .tools import (
    find_best_blas_type, validate_matrix_shape, validate_vector_shape
)
from .initialization import Initialization
from . import tools


class OptionWrapper:
    def __init__(self, mask_attribute, mask_value):
        # Name of the class-level bitmask attribute
        self.mask_attribute = mask_attribute
        # Value of this option
        self.mask_value = mask_value

    def __get__(self, obj, objtype):
        # Return True / False based on whether the bit is set in the bitmask
        return bool(getattr(obj, self.mask_attribute, 0) & self.mask_value)

    def __set__(self, obj, value):
        mask_attribute_value = getattr(obj, self.mask_attribute, 0)
        if bool(value):
            value = mask_attribute_value | self.mask_value
        else:
            value = mask_attribute_value & ~self.mask_value
        setattr(obj, self.mask_attribute, value)


class MatrixWrapper:
    def __init__(self, name, attribute):
        self.name = name
        self.attribute = attribute
        self._attribute = '_' + attribute

    def __get__(self, obj, objtype):
        matrix = getattr(obj, self._attribute, None)
        # # Remove last dimension if the array is not actually time-varying
        # if matrix is not None and matrix.shape[-1] == 1:
        #     return np.squeeze(matrix, -1)
        return matrix

    def __set__(self, obj, value):
        value = np.asarray(value, order="F")
        shape = obj.shapes[self.attribute]

        if len(shape) == 3:
            value = self._set_matrix(obj, value, shape)
        else:
            value = self._set_vector(obj, value, shape)

        setattr(obj, self._attribute, value)
        obj.shapes[self.attribute] = value.shape

    def _set_matrix(self, obj, value, shape):
        # Expand 1-dimensional array if possible
        if (value.ndim == 1 and shape[0] == 1 and
                value.shape[0] == shape[1]):
            value = value[None, :]

        # Enforce that the matrix is appropriate size
        validate_matrix_shape(
            self.name, value.shape, shape[0], shape[1], obj.nobs
        )

        # Expand time-invariant matrix
        if value.ndim == 2:
            value = np.array(value[:, :, None], order="F")

        return value

    def _set_vector(self, obj, value, shape):
        # Enforce that the vector has appropriate length
        validate_vector_shape(
            self.name, value.shape, shape[0], obj.nobs
        )

        # Expand the time-invariant vector
        if value.ndim == 1:
            value = np.array(value[:, None], order="F")

        return value


class Representation:
    r"""
    State space representation of a time series process

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
    initial_variance : float, optional
        Initial variance used when approximate diffuse initialization is
        specified. Default is 1e6.
    initialization : Initialization object or str, optional
        Initialization method for the initial state. If a string, must be one
        of {'diffuse', 'approximate_diffuse', 'stationary', 'known'}.
    initial_state : array_like, optional
        If `initialization='known'` is used, the mean of the initial state's
        distribution.
    initial_state_cov : array_like, optional
        If `initialization='known'` is used, the covariance matrix of the
        initial state's distribution.
    nobs : int, optional
        If an endogenous vector is not given (i.e. `k_endog` is an integer),
        the number of observations can optionally be specified. If not
        specified, they will be set to zero until data is bound to the model.
    dtype : np.dtype, optional
        If an endogenous vector is not given (i.e. `k_endog` is an integer),
        the default datatype of the state space matrices can optionally be
        specified. Default is `np.float64`.
    design : array_like, optional
        The design matrix, :math:`Z`. Default is set to zeros.
    obs_intercept : array_like, optional
        The intercept for the observation equation, :math:`d`. Default is set
        to zeros.
    obs_cov : array_like, optional
        The covariance matrix for the observation equation :math:`H`. Default
        is set to zeros.
    transition : array_like, optional
        The transition matrix, :math:`T`. Default is set to zeros.
    state_intercept : array_like, optional
        The intercept for the transition equation, :math:`c`. Default is set to
        zeros.
    selection : array_like, optional
        The selection matrix, :math:`R`. Default is set to zeros.
    state_cov : array_like, optional
        The covariance matrix for the state equation :math:`Q`. Default is set
        to zeros.
    **kwargs
        Additional keyword arguments. Not used directly. It is present to
        improve compatibility with subclasses, so that they can use `**kwargs`
        to specify any default state space matrices (e.g. `design`) without
        having to clean out any other keyword arguments they might have been
        passed.

    Attributes
    ----------
    nobs : int
        The number of observations.
    k_endog : int
        The dimension of the observation series.
    k_states : int
        The dimension of the unobserved state process.
    k_posdef : int
        The dimension of a guaranteed positive
        definite covariance matrix describing
        the shocks in the measurement equation.
    shapes : dictionary of name:tuple
        A dictionary recording the initial shapes
        of each of the representation matrices as
        tuples.
    initialization : str
        Kalman filter initialization method. Default is unset.
    initial_variance : float
        Initial variance for approximate diffuse
        initialization. Default is 1e6.

    Notes
    -----
    A general state space model is of the form

    .. math::

        y_t & = Z_t \alpha_t + d_t + \varepsilon_t \\
        \alpha_t & = T_t \alpha_{t-1} + c_t + R_t \eta_t \\

    where :math:`y_t` refers to the observation vector at time :math:`t`,
    :math:`\alpha_t` refers to the (unobserved) state vector at time
    :math:`t`, and where the irregular components are defined as

    .. math::

        \varepsilon_t \sim N(0, H_t) \\
        \eta_t \sim N(0, Q_t) \\

    The remaining variables (:math:`Z_t, d_t, H_t, T_t, c_t, R_t, Q_t`) in the
    equations are matrices describing the process. Their variable names and
    dimensions are as follows

    Z : `design`          :math:`(k\_endog \times k\_states \times nobs)`

    d : `obs_intercept`   :math:`(k\_endog \times nobs)`

    H : `obs_cov`         :math:`(k\_endog \times k\_endog \times nobs)`

    T : `transition`      :math:`(k\_states \times k\_states \times nobs)`

    c : `state_intercept` :math:`(k\_states \times nobs)`

    R : `selection`       :math:`(k\_states \times k\_posdef \times nobs)`

    Q : `state_cov`       :math:`(k\_posdef \times k\_posdef \times nobs)`

    In the case that one of the matrices is time-invariant (so that, for
    example, :math:`Z_t = Z_{t+1} ~ \forall ~ t`), its last dimension may
    be of size :math:`1` rather than size `nobs`.

    References
    ----------
    .. [*] Durbin, James, and Siem Jan Koopman. 2012.
       Time Series Analysis by State Space Methods: Second Edition.
       Oxford University Press.
    """

    endog = None
    r"""
    (array) The observation vector, alias for `obs`.
    """
    design = MatrixWrapper('design', 'design')
    r"""
    (array) Design matrix: :math:`Z~(k\_endog \times k\_states \times nobs)`
    """
    obs_intercept = MatrixWrapper('observation intercept', 'obs_intercept')
    r"""
    (array) Observation intercept: :math:`d~(k\_endog \times nobs)`
    """
    obs_cov = MatrixWrapper('observation covariance matrix', 'obs_cov')
    r"""
    (array) Observation covariance matrix:
    :math:`H~(k\_endog \times k\_endog \times nobs)`
    """
    transition = MatrixWrapper('transition', 'transition')
    r"""
    (array) Transition matrix:
    :math:`T~(k\_states \times k\_states \times nobs)`
    """
    state_intercept = MatrixWrapper('state intercept', 'state_intercept')
    r"""
    (array) State intercept: :math:`c~(k\_states \times nobs)`
    """
    selection = MatrixWrapper('selection', 'selection')
    r"""
    (array) Selection matrix:
    :math:`R~(k\_states \times k\_posdef \times nobs)`
    """
    state_cov = MatrixWrapper('state covariance matrix', 'state_cov')
    r"""
    (array) State covariance matrix:
    :math:`Q~(k\_posdef \times k\_posdef \times nobs)`
    """

    def __init__(self, k_endog, k_states, k_posdef=None,
                 initial_variance=1e6, nobs=0, dtype=np.float64,
                 design=None, obs_intercept=None, obs_cov=None,
                 transition=None, state_intercept=None, selection=None,
                 state_cov=None, statespace_classes=None, **kwargs):
        self.shapes = {}

        # Check if k_endog is actually the endog array
        endog = None
        if isinstance(k_endog, np.ndarray):
            endog = k_endog
            # If so, assume that it is either column-ordered and in wide format
            # or row-ordered and in long format
            if (endog.flags['C_CONTIGUOUS'] and
                    (endog.shape[0] > 1 or nobs == 1)):
                endog = endog.T
            k_endog = endog.shape[0]

        # Endogenous array, dimensions, dtype
        self.k_endog = k_endog
        if k_endog < 1:
            raise ValueError('Number of endogenous variables in statespace'
                             ' model must be a positive number.')
        self.nobs = nobs

        # Get dimensions from transition equation
        if k_states < 1:
            raise ValueError('Number of states in statespace model must be a'
                             ' positive number.')
        self.k_states = k_states
        self.k_posdef = k_posdef if k_posdef is not None else k_states

        # Make sure k_posdef <= k_states
        # TODO: we could technically allow k_posdef > k_states, but the Cython
        # code needs to be more thoroughly checked to avoid seg faults.
        if self.k_posdef > self.k_states:
            raise ValueError('Dimension of state innovation `k_posdef` cannot'
                             ' be larger than the dimension of the state.')

        # Bind endog, if it was given
        if endog is not None:
            self.bind(endog)

        # Record the shapes of all of our matrices
        # Note: these are time-invariant shapes; in practice the last dimension
        # may also be `self.nobs` for any or all of these.
        self.shapes = {
            'obs': (self.k_endog, self.nobs),
            'design': (self.k_endog, self.k_states, 1),
            'obs_intercept': (self.k_endog, 1),
            'obs_cov': (self.k_endog, self.k_endog, 1),
            'transition': (self.k_states, self.k_states, 1),
            'state_intercept': (self.k_states, 1),
            'selection': (self.k_states, self.k_posdef, 1),
            'state_cov': (self.k_posdef, self.k_posdef, 1),
        }

        # Representation matrices
        # These matrices are only used in the Python object as containers,
        # which will be copied to the appropriate _statespace object if a
        # filter is called.
        scope = locals()
        for name, shape in self.shapes.items():
            if name == 'obs':
                continue
            # Create the initial storage array for each matrix
            setattr(self, '_' + name, np.zeros(shape, dtype=dtype, order="F"))

            # If we were given an initial value for the matrix, set it
            # (notice it is being set via the descriptor)
            if scope[name] is not None:
                setattr(self, name, scope[name])

        # Options
        self.initial_variance = initial_variance
        self.prefix_statespace_map = (statespace_classes
                                      if statespace_classes is not None
                                      else tools.prefix_statespace_map.copy())

        # State-space initialization data
        self.initialization = kwargs.pop('initialization', None)
        basic_inits = ['diffuse', 'approximate_diffuse', 'stationary']

        if self.initialization in basic_inits:
            self.initialize(self.initialization)
        elif self.initialization == 'known':
            if 'constant' in kwargs:
                constant = kwargs.pop('constant')
            elif 'initial_state' in kwargs:
                # TODO deprecation warning
                constant = kwargs.pop('initial_state')
            else:
                raise ValueError('Initial state must be provided when "known"'
                                 ' is the specified initialization method.')
            if 'stationary_cov' in kwargs:
                stationary_cov = kwargs.pop('stationary_cov')
            elif 'initial_state_cov' in kwargs:
                # TODO deprecation warning
                stationary_cov = kwargs.pop('initial_state_cov')
            else:
                raise ValueError('Initial state covariance matrix must be'
                                 ' provided when "known" is the specified'
                                 ' initialization method.')
            self.initialize('known', constant=constant,
                            stationary_cov=stationary_cov)
        elif (not isinstance(self.initialization, Initialization) and
                self.initialization is not None):
            raise ValueError("Invalid state space initialization method.")

        # Check for unused kwargs
        if len(kwargs):
            # raise TypeError(f'{__class__} constructor got unexpected keyword'
            #                 f' argument(s): {kwargs}.')
            msg = (f'Unknown keyword arguments: {kwargs.keys()}.'
                   'Passing unknown keyword arguments will raise a TypeError'
                   ' beginning in version 0.15.')
            warnings.warn(msg, FutureWarning)

        # Matrix representations storage
        self._representations = {}

        # Setup the underlying statespace object storage
        self._statespaces = {}

        # Caches
        self._time_invariant = None

    def __getitem__(self, key):
        _type = type(key)
        # If only a string is given then we must be getting an entire matrix
        if _type is str:
            if key not in self.shapes:
                raise IndexError('"%s" is an invalid state space matrix name'
                                 % key)
            matrix = getattr(self, '_' + key)

            # See note on time-varying arrays, below
            if matrix.shape[-1] == 1:
                return matrix[(slice(None),)*(matrix.ndim-1) + (0,)]
            else:
                return matrix
        # Otherwise if we have a tuple, we want a slice of a matrix
        elif _type is tuple:
            name, slice_ = key[0], key[1:]
            if name not in self.shapes:
                raise IndexError('"%s" is an invalid state space matrix name'
                                 % name)

            matrix = getattr(self, '_' + name)

            # Since the model can support time-varying arrays, but often we
            # will instead have time-invariant arrays, we want to allow setting
            # a matrix slice like mod['transition',0,:] even though technically
            # it should be mod['transition',0,:,0]. Thus if the array in
            # question is time-invariant but the last slice was excluded,
            # add it in as a zero.
            if matrix.shape[-1] == 1 and len(slice_) <= matrix.ndim-1:
                slice_ = slice_ + (0,)

            return matrix[slice_]
        # Otherwise, we have only a single slice index, but it is not a string
        else:
            raise IndexError('First index must the name of a valid state space'
                             ' matrix.')

    def __setitem__(self, key, value):
        _type = type(key)
        # If only a string is given then we must be setting an entire matrix
        if _type is str:
            if key not in self.shapes:
                raise IndexError('"%s" is an invalid state space matrix name'
                                 % key)
            setattr(self, key, value)
        # If it's a tuple (with a string as the first element) then we must be
        # setting a slice of a matrix
        elif _type is tuple:
            name, slice_ = key[0], key[1:]
            if name not in self.shapes:
                raise IndexError('"%s" is an invalid state space matrix name'
                                 % key[0])

            # Change the dtype of the corresponding matrix
            dtype = np.array(value).dtype
            matrix = getattr(self, '_' + name)
            valid_types = ['f', 'd', 'F', 'D']
            if not matrix.dtype == dtype and dtype.char in valid_types:
                matrix = getattr(self, '_' + name).real.astype(dtype)

            # Since the model can support time-varying arrays, but often we
            # will instead have time-invariant arrays, we want to allow setting
            # a matrix slice like mod['transition',0,:] even though technically
            # it should be mod['transition',0,:,0]. Thus if the array in
            # question is time-invariant but the last slice was excluded,
            # add it in as a zero.
            if matrix.shape[-1] == 1 and len(slice_) == matrix.ndim-1:
                slice_ = slice_ + (0,)

            # Set the new value
            matrix[slice_] = value
            setattr(self, name, matrix)
        # Otherwise we got a single non-string key, (e.g. mod[:]), which is
        # invalid
        else:
            raise IndexError('First index must the name of a valid state space'
                             ' matrix.')

    def _clone_kwargs(self, endog, **kwargs):
        """
        Construct keyword arguments for cloning a state space model

        Parameters
        ----------
        endog : array_like
            An observed time-series process :math:`y`.
        **kwargs
            Keyword arguments to pass to the new state space representation
            model constructor. Those that are not specified are copied from
            the specification of the current state space model.
        """

        # We always need the base dimensions, but they cannot change from
        # the base model when cloning (the idea is: if these need to change,
        # need to make a new instance manually, since it's not really cloning).
        kwargs['nobs'] = len(endog)
        kwargs['k_endog'] = self.k_endog
        for key in ['k_states', 'k_posdef']:
            val = getattr(self, key)
            if key not in kwargs or kwargs[key] is None:
                kwargs[key] = val
            if kwargs[key] != val:
                raise ValueError('Cannot change the dimension of %s when'
                                 ' cloning.' % key)

        # Get defaults for time-invariant system matrices, if not otherwise
        # provided
        # Time-varying matrices must be replaced.
        for name in self.shapes.keys():
            if name == 'obs':
                continue

            if name not in kwargs:
                mat = getattr(self, name)
                if mat.shape[-1] != 1:
                    raise ValueError('The `%s` matrix is time-varying. Cloning'
                                     ' this model requires specifying an'
                                     ' updated matrix.' % name)
                kwargs[name] = mat

        # Default is to use the same initialization
        kwargs.setdefault('initialization', self.initialization)

        return kwargs

    def clone(self, endog, **kwargs):
        """
        Clone a state space representation while overriding some elements

        Parameters
        ----------
        endog : array_like
            An observed time-series process :math:`y`.
        **kwargs
            Keyword arguments to pass to the new state space representation
            model constructor. Those that are not specified are copied from
            the specification of the current state space model.

        Returns
        -------
        Representation

        Notes
        -----
        If some system matrices are time-varying, then new time-varying
        matrices *must* be provided.
        """
        kwargs = self._clone_kwargs(endog, **kwargs)
        mod = self.__class__(**kwargs)
        mod.bind(endog)
        return mod

    def extend(self, endog, start=None, end=None, **kwargs):
        """
        Extend the current state space model, or a specific (time) subset

        Parameters
        ----------
        endog : array_like
            An observed time-series process :math:`y`.
        start : int, optional
            The first period of a time-varying state space model to include in
            the new model. Has no effect if the state space model is
            time-invariant. Default is the initial period.
        end : int, optional
            The last period of a time-varying state space model to include in
            the new model. Has no effect if the state space model is
            time-invariant. Default is the final period.
        **kwargs
            Keyword arguments to pass to the new state space representation
            model constructor. Those that are not specified are copied from
            the specification of the current state space model.

        Returns
        -------
        Representation

        Notes
        -----
        This method does not allow replacing a time-varying system matrix with
        a time-invariant one (or vice-versa). If that is required, use `clone`.
        """
        endog = np.atleast_1d(endog)
        if endog.ndim == 1:
            endog = endog[:, np.newaxis]
        nobs = len(endog)

        if start is None:
            start = 0
        if end is None:
            end = self.nobs

        if start < 0:
            start = self.nobs + start
        if end < 0:
            end = self.nobs + end
        if start > self.nobs:
            raise ValueError('The `start` argument of the extension within the'
                             ' base model cannot be after the end of the'
                             ' base model.')
        if end > self.nobs:
            raise ValueError('The `end` argument of the extension within the'
                             ' base model cannot be after the end of the'
                             ' base model.')
        if start > end:
            raise ValueError('The `start` argument of the extension within the'
                             ' base model cannot be after the `end` argument.')

        # Note: if start == end or if end < self.nobs, then we're just cloning
        # (no extension)
        endog = tools.concat([self.endog[:, start:end].T, endog])

        # Extend any time-varying arrays
        error_ti = ('Model has time-invariant %s matrix, so cannot provide'
                    ' an extended matrix.')
        error_tv = ('Model has time-varying %s matrix, so an updated'
                    ' time-varying matrix for the extension period'
                    ' is required.')
        for name, shape in self.shapes.items():
            if name == 'obs':
                continue

            mat = getattr(self, name)

            # If we were *not* given an extended value for this matrix...
            if name not in kwargs:
                # If this is a time-varying matrix in the existing model
                if mat.shape[-1] > 1:
                    # If we have an extension period, then raise an error
                    # because we should have been given an extended value
                    if end + nobs > self.nobs:
                        raise ValueError(error_tv % name)
                    # If we do not have an extension period, then set the new
                    # time-varying matrix to be the portion of the existing
                    # time-varying matrix that corresponds to the period of
                    # interest
                    else:
                        kwargs[name] = mat[..., start:end + nobs]
            elif nobs == 0:
                raise ValueError('Extension is being performed within-sample'
                                 ' so cannot provide an extended matrix')
            # If we were given an extended value for this matrix
            else:
                # TODO: Need to add a check for ndim, and if the matrix has
                # one fewer dimensions than the existing matrix, add a new axis

                # If this is a time-invariant matrix in the existing model,
                # raise an error
                if mat.shape[-1] == 1 and self.nobs > 1:
                    raise ValueError(error_ti % name)

                # Otherwise, validate the shape of the given extended value
                # Note: we do not validate the number of observations here
                # (so we pass in updated_mat.shape[-1] as the nobs argument
                # in the validate_* calls); instead, we check below that we
                # at least `nobs` values were passed in and then only take the
                # first of them as required. This can be useful when e.g. the
                # end user knows the extension values up to some maximum
                # endpoint, but does not know what the calling methods may
                # specifically require.
                updated_mat = np.asarray(kwargs[name])
                if len(shape) == 2:
                    validate_vector_shape(name, updated_mat.shape, shape[0],
                                          updated_mat.shape[-1])
                else:
                    validate_matrix_shape(name, updated_mat.shape, shape[0],
                                          shape[1], updated_mat.shape[-1])

                if updated_mat.shape[-1] < nobs:
                    raise ValueError(error_tv % name)
                else:
                    updated_mat = updated_mat[..., :nobs]

                # Concatenate to get the new time-varying matrix
                kwargs[name] = np.c_[mat[..., start:end], updated_mat]

        return self.clone(endog, **kwargs)

    def diff_endog(self, new_endog, tolerance=1e-10):
        # TODO: move this function to tools?
        endog = self.endog.T
        if len(new_endog) < len(endog):
            raise ValueError('Given data (length %d) is too short to diff'
                             ' against model data (length %d).'
                             % (len(new_endog), len(endog)))
        if len(new_endog) > len(endog):
            nobs_append = len(new_endog) - len(endog)
            endog = np.c_[endog.T, new_endog[-nobs_append:].T * np.nan].T

        new_nan = np.isnan(new_endog)
        existing_nan = np.isnan(endog)
        diff = np.abs(new_endog - endog)
        diff[new_nan ^ existing_nan] = np.inf
        diff[new_nan & existing_nan] = 0.

        is_revision = (diff > tolerance)
        is_new = existing_nan & ~new_nan
        is_revision[is_new] = False

        revision_ix = list(zip(*np.where(is_revision)))
        new_ix = list(zip(*np.where(is_new)))

        return revision_ix, new_ix

    @property
    def prefix(self):
        """
        (str) BLAS prefix of currently active representation matrices
        """
        arrays = (
            self._design, self._obs_intercept, self._obs_cov,
            self._transition, self._state_intercept, self._selection,
            self._state_cov
        )
        if self.endog is not None:
            arrays = (self.endog,) + arrays
        return find_best_blas_type(arrays)[0]

    @property
    def dtype(self):
        """
        (dtype) Datatype of currently active representation matrices
        """
        return tools.prefix_dtype_map[self.prefix]

    @property
    def time_invariant(self):
        """
        (bool) Whether or not currently active representation matrices are
        time-invariant
        """
        if self._time_invariant is None:
            return (
                self._design.shape[2] == self._obs_intercept.shape[1] ==
                self._obs_cov.shape[2] == self._transition.shape[2] ==
                self._state_intercept.shape[1] == self._selection.shape[2] ==
                self._state_cov.shape[2]
            )
        else:
            return self._time_invariant

    @property
    def _statespace(self):
        prefix = self.prefix
        if prefix in self._statespaces:
            return self._statespaces[prefix]
        return None

    @property
    def obs(self):
        r"""
        (array) Observation vector: :math:`y~(k\_endog \times nobs)`
        """
        return self.endog

    def bind(self, endog):
        """
        Bind data to the statespace representation

        Parameters
        ----------
        endog : ndarray
            Endogenous data to bind to the model. Must be column-ordered
            ndarray with shape (`k_endog`, `nobs`) or row-ordered ndarray with
            shape (`nobs`, `k_endog`).

        Notes
        -----
        The strict requirements arise because the underlying statespace and
        Kalman filtering classes require Fortran-ordered arrays in the wide
        format (shaped (`k_endog`, `nobs`)), and this structure is setup to
        prevent copying arrays in memory.

        By default, numpy arrays are row (C)-ordered and most time series are
        represented in the long format (with time on the 0-th axis). In this
        case, no copying or re-ordering needs to be performed, instead the
        array can simply be transposed to get it in the right order and shape.

        Although this class (Representation) has stringent `bind` requirements,
        it is assumed that it will rarely be used directly.
        """
        if not isinstance(endog, np.ndarray):
            raise ValueError("Invalid endogenous array; must be an ndarray.")

        # Make sure we have a 2-dimensional array
        # Note: reshaping a 1-dim array into a 2-dim array by changing the
        #       shape tuple always results in a row (C)-ordered array, so it
        #       must be shaped (nobs, k_endog)
        if endog.ndim == 1:
            # In the case of nobs x 0 arrays
            if self.k_endog == 1:
                endog.shape = (endog.shape[0], 1)
            # In the case of k_endog x 0 arrays
            else:
                endog.shape = (1, endog.shape[0])
        if not endog.ndim == 2:
            raise ValueError('Invalid endogenous array provided; must be'
                             ' 2-dimensional.')

        # Check for valid column-ordered arrays
        if endog.flags['F_CONTIGUOUS'] and endog.shape[0] == self.k_endog:
            pass
        # Check for valid row-ordered arrays, and transpose them to be the
        # correct column-ordered array
        elif endog.flags['C_CONTIGUOUS'] and endog.shape[1] == self.k_endog:
            endog = endog.T
        # Invalid column-ordered arrays
        elif endog.flags['F_CONTIGUOUS']:
            raise ValueError('Invalid endogenous array; column-ordered'
                             ' arrays must have first axis shape of'
                             ' `k_endog`.')
        # Invalid row-ordered arrays
        elif endog.flags['C_CONTIGUOUS']:
            raise ValueError('Invalid endogenous array; row-ordered'
                             ' arrays must have last axis shape of'
                             ' `k_endog`.')
        # Non-contiguous arrays
        else:
            raise ValueError('Invalid endogenous array; must be ordered in'
                             ' contiguous memory.')

        # We may still have a non-fortran contiguous array, so double-check
        if not endog.flags['F_CONTIGUOUS']:
            endog = np.asfortranarray(endog)

        # Set a flag for complex data
        self._complex_endog = np.iscomplexobj(endog)

        # Set the data
        self.endog = endog
        self.nobs = self.endog.shape[1]

        # Reset shapes
        if hasattr(self, 'shapes'):
            self.shapes['obs'] = self.endog.shape

    def initialize(self, initialization, approximate_diffuse_variance=None,
                   constant=None, stationary_cov=None, a=None, Pstar=None,
                   Pinf=None, A=None, R0=None, Q0=None):
        """Create an Initialization object if necessary"""
        if initialization == 'known':
            initialization = Initialization(self.k_states, 'known',
                                            constant=constant,
                                            stationary_cov=stationary_cov)
        elif initialization == 'components':
            initialization = Initialization.from_components(
                a=a, Pstar=Pstar, Pinf=Pinf, A=A, R0=R0, Q0=Q0)
        elif initialization == 'approximate_diffuse':
            if approximate_diffuse_variance is None:
                approximate_diffuse_variance = self.initial_variance
            initialization = Initialization(
                self.k_states, 'approximate_diffuse',
                approximate_diffuse_variance=approximate_diffuse_variance)
        elif initialization == 'stationary':
            initialization = Initialization(self.k_states, 'stationary')
        elif initialization == 'diffuse':
            initialization = Initialization(self.k_states, 'diffuse')

        # We must have an initialization object at this point
        if not isinstance(initialization, Initialization):
            raise ValueError("Invalid state space initialization method.")

        self.initialization = initialization

    def initialize_known(self, constant, stationary_cov):
        """
        Initialize the statespace model with known distribution for initial
        state.

        These values are assumed to be known with certainty or else
        filled with parameters during, for example, maximum likelihood
        estimation.

        Parameters
        ----------
        constant : array_like
            Known mean of the initial state vector.
        stationary_cov : array_like
            Known covariance matrix of the initial state vector.
        """
        constant = np.asarray(constant, order="F")
        stationary_cov = np.asarray(stationary_cov, order="F")

        if not constant.shape == (self.k_states,):
            raise ValueError('Invalid dimensions for constant state vector.'
                             ' Requires shape (%d,), got %s' %
                             (self.k_states, str(constant.shape)))
        if not stationary_cov.shape == (self.k_states, self.k_states):
            raise ValueError('Invalid dimensions for stationary covariance'
                             ' matrix. Requires shape (%d,%d), got %s' %
                             (self.k_states, self.k_states,
                              str(stationary_cov.shape)))

        self.initialize('known', constant=constant,
                        stationary_cov=stationary_cov)

    def initialize_approximate_diffuse(self, variance=None):
        """
        Initialize the statespace model with approximate diffuse values.

        Rather than following the exact diffuse treatment (which is developed
        for the case that the variance becomes infinitely large), this assigns
        an arbitrary large number for the variance.

        Parameters
        ----------
        variance : float, optional
            The variance for approximating diffuse initial conditions. Default
            is 1e6.
        """
        if variance is None:
            variance = self.initial_variance

        self.initialize('approximate_diffuse',
                        approximate_diffuse_variance=variance)

    def initialize_components(self, a=None, Pstar=None, Pinf=None, A=None,
                              R0=None, Q0=None):
        """
        Initialize the statespace model with component matrices

        Parameters
        ----------
        a : array_like, optional
            Vector of constant values describing the mean of the stationary
            component of the initial state.
        Pstar : array_like, optional
            Stationary component of the initial state covariance matrix. If
            given, should be a matrix shaped `k_states x k_states`. The
            submatrix associated with the diffuse states should contain zeros.
            Note that by definition, `Pstar = R0 @ Q0 @ R0.T`, so either
            `R0,Q0` or `Pstar` may be given, but not both.
        Pinf : array_like, optional
            Diffuse component of the initial state covariance matrix. If given,
            should be a matrix shaped `k_states x k_states` with ones in the
            diagonal positions corresponding to states with diffuse
            initialization and zeros otherwise. Note that by definition,
            `Pinf = A @ A.T`, so either `A` or `Pinf` may be given, but not
            both.
        A : array_like, optional
            Diffuse selection matrix, used in the definition of the diffuse
            initial state covariance matrix. If given, should be a
            `k_states x k_diffuse_states` matrix that contains the subset of
            the columns of the identity matrix that correspond to states with
            diffuse initialization. Note that by definition, `Pinf = A @ A.T`,
            so either `A` or `Pinf` may be given, but not both.
        R0 : array_like, optional
            Stationary selection matrix, used in the definition of the
            stationary initial state covariance matrix. If given, should be a
            `k_states x k_nondiffuse_states` matrix that contains the subset of
            the columns of the identity matrix that correspond to states with a
            non-diffuse initialization. Note that by definition,
            `Pstar = R0 @ Q0 @ R0.T`, so either `R0,Q0` or `Pstar` may be
            given, but not both.
        Q0 : array_like, optional
            Covariance matrix associated with stationary initial states. If
            given, should be a matrix shaped
            `k_nondiffuse_states x k_nondiffuse_states`.
            Note that by definition, `Pstar = R0 @ Q0 @ R0.T`, so either
            `R0,Q0` or `Pstar` may be given, but not both.

        Notes
        -----
        The matrices `a, Pstar, Pinf, A, R0, Q0` and the process for
        initializing the state space model is as given in Chapter 5 of [1]_.
        For the definitions of these matrices, see equation (5.2) and the
        subsequent discussion there.

        References
        ----------
        .. [1] Durbin, James, and Siem Jan Koopman. 2012.
           Time Series Analysis by State Space Methods: Second Edition.
           Oxford University Press.
        """
        self.initialize('components', a=a, Pstar=Pstar, Pinf=Pinf, A=A, R0=R0,
                        Q0=Q0)

    def initialize_stationary(self):
        """
        Initialize the statespace model as stationary.
        """
        self.initialize('stationary')

    def initialize_diffuse(self):
        """
        Initialize the statespace model as diffuse.
        """
        self.initialize('diffuse')

    def _initialize_representation(self, prefix=None):
        if prefix is None:
            prefix = self.prefix
        dtype = tools.prefix_dtype_map[prefix]

        # If the dtype-specific representation matrices do not exist, create
        # them
        if prefix not in self._representations:
            # Copy the statespace representation matrices
            self._representations[prefix] = {}
            for matrix in self.shapes.keys():
                if matrix == 'obs':
                    self._representations[prefix][matrix] = (
                        self.obs.astype(dtype)
                    )
                else:
                    # Note: this always makes a copy
                    self._representations[prefix][matrix] = (
                        getattr(self, '_' + matrix).astype(dtype)
                    )
        # If they do exist, update them
        else:
            for matrix in self.shapes.keys():
                existing = self._representations[prefix][matrix]
                if matrix == 'obs':
                    # existing[:] = self.obs.astype(dtype)
                    pass
                else:
                    new = getattr(self, '_' + matrix).astype(dtype)
                    if existing.shape == new.shape:
                        existing[:] = new[:]
                    else:
                        self._representations[prefix][matrix] = new

        # Determine if we need to (re-)create the _statespace models
        # (if time-varying matrices changed)
        if prefix in self._statespaces:
            ss = self._statespaces[prefix]
            create = (
                not ss.obs.shape[1] == self.endog.shape[1] or
                not ss.design.shape[2] == self.design.shape[2] or
                not ss.obs_intercept.shape[1] == self.obs_intercept.shape[1] or
                not ss.obs_cov.shape[2] == self.obs_cov.shape[2] or
                not ss.transition.shape[2] == self.transition.shape[2] or
                not (ss.state_intercept.shape[1] ==
                     self.state_intercept.shape[1]) or
                not ss.selection.shape[2] == self.selection.shape[2] or
                not ss.state_cov.shape[2] == self.state_cov.shape[2]
            )
        else:
            create = True

        # (re-)create if necessary
        if create:
            if prefix in self._statespaces:
                del self._statespaces[prefix]

            # Setup the base statespace object
            cls = self.prefix_statespace_map[prefix]
            self._statespaces[prefix] = cls(
                self._representations[prefix]['obs'],
                self._representations[prefix]['design'],
                self._representations[prefix]['obs_intercept'],
                self._representations[prefix]['obs_cov'],
                self._representations[prefix]['transition'],
                self._representations[prefix]['state_intercept'],
                self._representations[prefix]['selection'],
                self._representations[prefix]['state_cov']
            )

        return prefix, dtype, create

    def _initialize_state(self, prefix=None, complex_step=False):
        # TODO once the transition to using the Initialization objects is
        # complete, this should be moved entirely to the _{{prefix}}Statespace
        # object.
        if prefix is None:
            prefix = self.prefix

        # (Re-)initialize the statespace model
        if isinstance(self.initialization, Initialization):
            if not self.initialization.initialized:
                raise RuntimeError('Initialization is incomplete.')
            self._statespaces[prefix].initialize(self.initialization,
                                                 complex_step=complex_step)
        else:
            raise RuntimeError('Statespace model not initialized.')


class FrozenRepresentation:
    """
    Frozen Statespace Model

    Takes a snapshot of a Statespace model.

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
        The dimension of a guaranteed positive definite
        covariance matrix describing the shocks in the
        measurement equation.
    dtype : dtype
        Datatype of representation matrices
    prefix : str
        BLAS prefix of representation matrices
    shapes : dictionary of name:tuple
        A dictionary recording the shapes of each of
        the representation matrices as tuples.
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
    initialization : Initialization object
        Kalman filter initialization method.
    initial_state : array_like
        The state vector used to initialize the Kalamn filter.
    initial_state_cov : array_like
        The state covariance matrix used to initialize the Kalamn filter.
    """
    _model_attributes = [
        'model', 'prefix', 'dtype', 'nobs', 'k_endog', 'k_states',
        'k_posdef', 'time_invariant', 'endog', 'design', 'obs_intercept',
        'obs_cov', 'transition', 'state_intercept', 'selection',
        'state_cov', 'missing', 'nmissing', 'shapes', 'initialization',
        'initial_state', 'initial_state_cov', 'initial_variance'
    ]
    _attributes = _model_attributes

    def __init__(self, model):
        # Initialize all attributes to None
        for name in self._attributes:
            setattr(self, name, None)

        # Update the representation attributes
        self.update_representation(model)

    def update_representation(self, model):
        """Update model Representation"""
        # Model
        self.model = model

        # Data type
        self.prefix = model.prefix
        self.dtype = model.dtype

        # Copy the model dimensions
        self.nobs = model.nobs
        self.k_endog = model.k_endog
        self.k_states = model.k_states
        self.k_posdef = model.k_posdef
        self.time_invariant = model.time_invariant

        # Save the state space representation at the time
        self.endog = model.endog
        self.design = model._design.copy()
        self.obs_intercept = model._obs_intercept.copy()
        self.obs_cov = model._obs_cov.copy()
        self.transition = model._transition.copy()
        self.state_intercept = model._state_intercept.copy()
        self.selection = model._selection.copy()
        self.state_cov = model._state_cov.copy()

        self.missing = np.array(model._statespaces[self.prefix].missing,
                                copy=True)
        self.nmissing = np.array(model._statespaces[self.prefix].nmissing,
                                 copy=True)

        # Save the final shapes of the matrices
        self.shapes = dict(model.shapes)
        for name in self.shapes.keys():
            if name == 'obs':
                continue
            self.shapes[name] = getattr(self, name).shape
        self.shapes['obs'] = self.endog.shape

        # Save the state space initialization
        self.initialization = model.initialization

        if model.initialization is not None:
            model._initialize_state()
            self.initial_state = np.array(
                model._statespaces[self.prefix].initial_state, copy=True)
            self.initial_state_cov = np.array(
                model._statespaces[self.prefix].initial_state_cov, copy=True)
            self.initial_diffuse_state_cov = np.array(
                model._statespaces[self.prefix].initial_diffuse_state_cov,
                copy=True)
