"""
State Space Representation - Initialization

Author: Chad Fulton
License: Simplified-BSD
"""
import warnings

import numpy as np

from . import tools


class Initialization:
    r"""
    State space initialization

    Parameters
    ----------
    k_states : int
    exact_diffuse_initialization : bool, optional
        Whether or not to use exact diffuse initialization; only has an effect
        if some states are initialized as diffuse. Default is True.
    approximate_diffuse_variance : float, optional
        If using approximate diffuse initialization, the initial variance used.
        Default is 1e6.

    Notes
    -----
    As developed in Durbin and Koopman (2012), the state space recursions
    must be initialized for the first time period. The general form of this
    initialization is:

    .. math::

        \alpha_1 & = a + A \delta + R_0 \eta_0 \\
        \delta & \sim N(0, \kappa I), \kappa \to \infty \\
        \eta_0 & \sim N(0, Q_0)

    Thus the state vector can be initialized with a known constant part
    (elements of :math:`a`), with part modeled as a diffuse initial
    distribution (as a part of :math:`\delta`), and with a part modeled as a
    known (proper) initial distribution (as a part of :math:`\eta_0`).

    There are two important restrictions:

    1. An element of the state vector initialized as diffuse cannot be also
       modeled with a stationary component. In the `validate` method,
       violations of this cause an exception to be raised.
    2. If an element of the state vector is initialized with both a known
       constant part and with a diffuse initial distribution, the effect of
       the diffuse initialization will essentially ignore the given known
       constant value. In the `validate` method, violations of this cause a
       warning to be given, since it is not technically invalid but may
       indicate user error.

    The :math:`\eta_0` compoenent is also referred to as the stationary part
    because it is often set to the unconditional distribution of a stationary
    process.

    Initialization is specified for blocks (consecutive only, for now) of the
    state vector, with the entire state vector and individual elements as
    special cases. Denote the block in question as :math:`\alpha_1^{(i)}`. It
    can be initialized in the following ways:

    - 'known'
    - 'diffuse' or 'exact_diffuse' or 'approximate_diffuse'
    - 'stationary'
    - 'mixed'

    In the first three cases, the block's initialization is specified as an
    instance of the `Initialization` class, with the `initialization_type`
    attribute set to one of those three string values. In the 'mixed' cases,
    the initialization is also an instance of the `Initialization` class, but
    it will itself contain sub-blocks. Details of each type follow.

    Regardless of the type, for each block, the following must be defined:
    the `constant` array, an array `diffuse` with indices corresponding to
    diffuse elements, an array `stationary` with indices corresponding to
    stationary elements, and `stationary_cov`, a matrix with order equal to the
    number of stationary elements in the block.

    **Known**

    If a block is initialized as known, then a known (possibly degenerate)
    distribution is used; in particular, the block of states is understood to
    be distributed
    :math:`\alpha_1^{(i)} \sim N(a^{(i)}, Q_0^{(i)})`. Here, is is possible to
    set :math:`a^{(i)} = 0`, and it is also possible that
    :math:`Q_0^{(i)}` is only positive-semidefinite; i.e.
    :math:`\alpha_1^{(i)}` may be degenerate. One particular example is
    that if the entire block's initial values are known, then
    :math:`R_0^{(i)} = 0`, and so `Var(\alpha_1^{(i)}) = 0`.

    Here, `constant` must be provided (although it can be zeros), and
    `stationary_cov` is optional (by default it is a matrix of zeros).

    **Diffuse**

    If a block is initialized as diffuse, then set
    :math:`\alpha_1^{(i)} \sim N(a^{(i)}, \kappa^{(i)} I)`. If the block is
    initialized using the exact diffuse initialization procedure, then it is
    understood that :math:`\kappa^{(i)} \to \infty`.

    If the block is initialized using the approximate diffuse initialization
    procedure, then `\kappa^{(i)}` is set to some large value rather than
    driven to infinity.

    In the approximate diffuse initialization case, it is possible, although
    unlikely, that a known constant value may have some effect on
    initialization if :math:`\kappa^{(i)}` is not set large enough.

    Here, `constant` may be provided, and `approximate_diffuse_variance` may be
    provided.

    **Stationary**

    If a block is initialized as stationary, then the block of states is
    understood to have the distribution
    :math:`\alpha_1^{(i)} \sim N(a^{(i)}, Q_0^{(i)})`. :math:`a^{(i)}` is
    the unconditional mean of the block, computed as
    :math:`(I - T^{(i)})^{-1} c_t`. :math:`Q_0^{(i)}` is the unconditional
    variance of the block, computed as the solution to the discrete Lyapunov
    equation:

    .. math::

        T^{(i)} Q_0^{(i)} T^{(i)} + (R Q R')^{(i)} = Q_0^{(i)}

    :math:`T^{(i)}` and :math:`(R Q R')^{(i)}` are the submatrices of
    the corresponding state space system matrices corresponding to the given
    block of states.

    Here, no values can be provided.

    **Mixed**

    In this case, the block can be further broken down into sub-blocks.
    Usually, only the top-level `Initialization` instance will be of 'mixed'
    type, and in many cases, even the top-level instance will be purely
    'known', 'diffuse', or 'stationary'.

    For a block of type mixed, suppose that it has `J` sub-blocks,
    :math:`\alpha_1^{(i,j)}`. Then
    :math:`\alpha_1^{(i)} = a^{(i)} + A^{(i)} \delta + R_0^{(i)} \eta_0^{(i)}`.

    Examples
    --------

    Basic examples have one specification for all of the states:

    >>> Initialization(k_states=2, 'known', constant=[0, 1])
    >>> Initialization(k_states=2, 'known', stationary_cov=np.eye(2))
    >>> Initialization(k_states=2, 'known', constant=[0, 1],
                       stationary_cov=np.eye(2))
    >>> Initialization(k_states=2, 'diffuse')
    >>> Initialization(k_states=2, 'approximate_diffuse',
                       approximate_diffuse_variance=1e6)
    >>> Initialization(k_states=2, 'stationary')

    More complex examples initialize different blocks of states separately

    >>> init = Initialization(k_states=3)
    >>> init.set((0, 1), 'known', constant=[0, 1])
    >>> init.set((0, 1), 'known', stationary_cov=np.eye(2))
    >>> init.set((0, 1), 'known', constant=[0, 1],
                 stationary_cov=np.eye(2))
    >>> init.set((0, 1), 'diffuse')
    >>> init.set((0, 1), 'approximate_diffuse',
                 approximate_diffuse_variance=1e6)
    >>> init.set((0, 1), 'stationary')

    A still more complex example initializes a block using a previously
    created `Initialization` object:

    >>> init1 = Initialization(k_states=2, 'known', constant=[0, 1])
    >>> init2 = Initialization(k_states=3)
    >>> init2.set((1, 2), init1)
    """

    def __init__(self, k_states, initialization_type=None,
                 initialization_classes=None, approximate_diffuse_variance=1e6,
                 constant=None, stationary_cov=None):
        # Parameters
        self.k_states = k_states

        # Attributes handling blocks of states with different initializations
        self._states = tuple(np.arange(k_states))
        self._initialization = np.array([None] * k_states)
        self.blocks = {}

        # Attributes handling initialization of the entire set of states
        # `constant` is a vector of constant values (i.e. it is the vector
        # a from DK)
        self.initialization_type = None
        self.constant = np.zeros(self.k_states)
        self.stationary_cov = np.zeros((self.k_states, self.k_states))
        self.approximate_diffuse_variance = approximate_diffuse_variance

        # Cython interface attributes
        self.prefix_initialization_map = (
            initialization_classes if initialization_classes is not None
            else tools.prefix_initialization_map.copy())
        self._representations = {}
        self._initializations = {}

        # If given a global initialization, use it now
        if initialization_type is not None:
            self.set(None, initialization_type, constant=constant,
                     stationary_cov=stationary_cov)

    @classmethod
    def from_components(cls, k_states, a=None, Pstar=None, Pinf=None, A=None,
                        R0=None, Q0=None):
        r"""
        Construct initialization object from component matrices

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

        Returns
        -------
        initialization
            Initialization object.

        Notes
        -----
        The matrices `a, Pstar, Pinf, A, R0, Q0` and the process for
        initializing the state space model is as given in Chapter 5 of [1]_.
        For the definitions of these matrices, see equation (5.2) and the
        subsequent discussion there.

        References
        ----------
        .. [*] Durbin, James, and Siem Jan Koopman. 2012.
           Time Series Analysis by State Space Methods: Second Edition.
           Oxford University Press.
        """
        k_states = k_states

        # Standardize the input
        a = tools._atleast_1d(a)
        Pstar, Pinf, A, R0, Q0 = tools._atleast_2d(Pstar, Pinf, A, R0, Q0)

        # Validate the diffuse component
        if Pstar is not None and (R0 is not None or Q0 is not None):
            raise ValueError('Cannot specify the initial state covariance both'
                             ' as `Pstar` and as the components R0 and Q0'
                             '  (because `Pstar` is defined such that'
                             " `Pstar=R0 Q0 R0'`).")
        if Pinf is not None and A is not None:
            raise ValueError('Cannot specify both the diffuse covariance'
                             ' matrix `Pinf` and the selection matrix for'
                             ' diffuse elements, A, (because Pinf is defined'
                             " such that `Pinf=A A'`).")
        elif A is not None:
            Pinf = np.dot(A, A.T)

        # Validate the non-diffuse component
        if a is None:
            a = np.zeros(k_states)
        if len(a) != k_states:
            raise ValueError('Must provide constant initialization vector for'
                             ' the entire state vector.')
        if R0 is not None or Q0 is not None:
            if R0 is None or Q0 is None:
                raise ValueError('If specifying either of R0 or Q0 then you'
                                 ' must specify both R0 and Q0.')
            Pstar = R0.dot(Q0).dot(R0.T)

        # Handle the diffuse component
        diffuse_ix = []
        if Pinf is not None:
            diffuse_ix = np.where(np.diagonal(Pinf))[0].tolist()

            if Pstar is not None:
                for i in diffuse_ix:
                    if not (np.all(Pstar[i] == 0) and
                            np.all(Pstar[:, i] == 0)):
                        raise ValueError(f'The state at position {i} was'
                                         ' specified as diffuse in Pinf, but'
                                         ' also contains a non-diffuse'
                                         ' diagonal or off-diagonal in Pstar.')
        k_diffuse_states = len(diffuse_ix)

        nondiffuse_ix = [i for i in np.arange(k_states) if i not in diffuse_ix]
        k_nondiffuse_states = k_states - k_diffuse_states

        # If there are non-diffuse states, require Pstar
        if Pstar is None and k_nondiffuse_states > 0:
            raise ValueError('Must provide initial covariance matrix for'
                             ' non-diffuse states.')

        # Construct the initialization
        init = cls(k_states)
        if nondiffuse_ix:
            nondiffuse_groups = np.split(
                nondiffuse_ix, np.where(np.diff(nondiffuse_ix) != 1)[0] + 1)
        else:
            nondiffuse_groups = []
        for group in nondiffuse_groups:
            s = slice(group[0], group[-1] + 1)
            init.set(s, 'known', constant=a[s], stationary_cov=Pstar[s, s])
        for i in diffuse_ix:
            init.set(i, 'diffuse')

        return init

    @classmethod
    def from_results(cls, filter_results):
        a = filter_results.initial_state
        Pstar = filter_results.initial_state_cov
        Pinf = filter_results.initial_diffuse_state_cov

        return cls.from_components(filter_results.model.k_states,
                                   a=a, Pstar=Pstar, Pinf=Pinf)

    def __setitem__(self, index, initialization_type):
        self.set(index, initialization_type)

    def _initialize_initialization(self, prefix):
        dtype = tools.prefix_dtype_map[prefix]

        # If the dtype-specific representation matrices do not exist, create
        # them
        if prefix not in self._representations:
            # Copy the statespace representation matrices
            self._representations[prefix] = {
                'constant': self.constant.astype(dtype),
                'stationary_cov': np.asfortranarray(
                    self.stationary_cov.astype(dtype)),
            }
        # If they do exist, update them
        else:
            self._representations[prefix]['constant'][:] = (
                self.constant.astype(dtype)[:])
            self._representations[prefix]['stationary_cov'][:] = (
                self.stationary_cov.astype(dtype)[:])

        # Create if necessary
        if prefix not in self._initializations:
            # Setup the base statespace object
            cls = self.prefix_initialization_map[prefix]
            self._initializations[prefix] = cls(
                self.k_states, self._representations[prefix]['constant'],
                self._representations[prefix]['stationary_cov'],
                self.approximate_diffuse_variance)
        # Otherwise update
        else:
            self._initializations[prefix].approximate_diffuse_variance = (
                self.approximate_diffuse_variance)

        return prefix, dtype

    def set(self, index, initialization_type, constant=None,
            stationary_cov=None, approximate_diffuse_variance=None):
        r"""
        Set initialization for states, either globally or for a block

        Parameters
        ----------
        index : tuple or int or None
            Arguments used to create a `slice` of states. Can be a tuple with
            `(start, stop)` (note that for `slice`, stop is not inclusive), or
            an integer (to select a specific state), or None (to select all the
            states).
        initialization_type : str
            The type of initialization used for the states selected by `index`.
            Must be one of 'known', 'diffuse', 'approximate_diffuse', or
            'stationary'.
        constant : array_like, optional
            A vector of constant values, denoted :math:`a`. Most often used
            with 'known' initialization, but may also be used with
            'approximate_diffuse' (although it will then likely have little
            effect).
        stationary_cov : array_like, optional
            The covariance matrix of the stationary part, denoted :math:`Q_0`.
            Only used with 'known' initialization.
        approximate_diffuse_variance : float, optional
            The approximate diffuse variance, denoted :math:`\kappa`. Only
            applicable with 'approximate_diffuse' initialization. Default is
            1e6.
        """
        # Construct the index, using a slice object as an intermediate step
        # to enforce regularity
        if not isinstance(index, slice):
            if isinstance(index, (int, np.integer)):
                index = int(index)
                if index < 0 or index >= self.k_states:
                    raise ValueError('Invalid index.')
                index = (index, index + 1)
            elif index is None:
                index = (index,)
            elif not isinstance(index, tuple):
                raise ValueError('Invalid index.')
            if len(index) > 2:
                raise ValueError('Cannot include a slice step in `index`.')
            index = slice(*index)
        index = self._states[index]

        # Compatibility with zero-length slices (can make it easier to set up
        # initialization without lots of if statements)
        if len(index) == 0:
            return

        # Make sure that we are not setting a block when global initialization
        # was previously set
        if self.initialization_type is not None and not index == self._states:
            raise ValueError('Cannot set initialization for the block of'
                             '  states %s because initialization was'
                             ' previously performed globally. You must either'
                             ' re-initialize globally or'
                             ' else unset the global initialization before'
                             ' initializing specific blocks of states.'
                             % str(index))
        # Make sure that we are not setting a block that *overlaps* with
        # another block (although we are free to *replace* an entire block)
        uninitialized = np.equal(self._initialization[index, ], None)
        if index not in self.blocks and not np.all(uninitialized):
            raise ValueError('Cannot set initialization for the state(s) %s'
                             ' because they are a subset of a previously'
                             ' initialized block. You must either'
                             ' re-initialize the entire block as a whole or'
                             ' else unset the entire block before'
                             ' re-initializing the subset.'
                             % str(np.array(index)[~uninitialized]))

        # If setting for all states, set this object's initialization
        # attributes
        k_states = len(index)
        if k_states == self.k_states:
            self.initialization_type = initialization_type

            # General validation
            if (approximate_diffuse_variance is not None and
                    not initialization_type == 'approximate_diffuse'):
                raise ValueError('`approximate_diffuse_variance` can only be'
                                 ' provided when using approximate diffuse'
                                 ' initialization.')
            if (stationary_cov is not None and
                    not initialization_type == 'known'):
                raise ValueError('`stationary_cov` can only be provided when'
                                 ' using known initialization.')

            # Specific initialization handling
            if initialization_type == 'known':
                # Make sure we were given some known initialization
                if constant is None and stationary_cov is None:
                    raise ValueError('Must specify either the constant vector'
                                     ' or the stationary covariance matrix'
                                     ' (or both) if using known'
                                     ' initialization.')
                # Defaults
                if stationary_cov is None:
                    stationary_cov = np.zeros((k_states, k_states))
                else:
                    stationary_cov = np.array(stationary_cov)

                # Validate
                if not stationary_cov.shape == (k_states, k_states):
                    raise ValueError('Invalid stationary covariance matrix;'
                                     ' given shape %s but require shape %s.'
                                     % (str(stationary_cov.shape),
                                        str((k_states, k_states))))

                # Set values
                self.stationary_cov = stationary_cov
            elif initialization_type == 'diffuse':
                if constant is not None:
                    warnings.warn('Constant values provided, but they are'
                                  ' ignored in exact diffuse initialization.')
            elif initialization_type == 'approximate_diffuse':
                if approximate_diffuse_variance is not None:
                    self.approximate_diffuse_variance = (
                        approximate_diffuse_variance)
            elif initialization_type == 'stationary':
                if constant is not None:
                    raise ValueError('Constant values cannot be provided for'
                                     ' stationary initialization.')
            else:
                raise ValueError('Invalid initialization type.')

            # Handle constant
            if constant is None:
                constant = np.zeros(k_states)
            else:
                constant = np.array(constant)
            if not constant.shape == (k_states,):
                raise ValueError('Invalid constant vector; given shape %s'
                                 ' but require shape %s.'
                                 % (str(constant.shape), str((k_states,))))
            self.constant = constant
        # Otherwise, if setting a sub-block, construct the new initialization
        # object
        else:
            if isinstance(initialization_type, Initialization):
                init = initialization_type
            else:
                if approximate_diffuse_variance is None:
                    approximate_diffuse_variance = (
                        self.approximate_diffuse_variance)
                init = Initialization(
                    k_states, initialization_type, constant=constant,
                    stationary_cov=stationary_cov,
                    approximate_diffuse_variance=approximate_diffuse_variance)

            self.blocks[index] = init
            for i in index:
                self._initialization[i] = index

    def unset(self, index):
        """
        Unset initialization for states, either globally or for a block

        Parameters
        ----------
        index : tuple or int or None
            Arguments used to create a `slice` of states. Can be a tuple with
            `(start, stop)` (note that for `slice`, stop is not inclusive), or
            an integer (to select a specific state), or None (to select all the
            states).

        Notes
        -----
        Note that this specifically unsets initializations previously created
        using `set` with this same index. Thus you cannot use `index=None` to
        unset all initializations, but only to unset a previously set global
        initialization. To unset all initializations (including both global and
        block level), use the `clear` method.
        """
        if isinstance(index, (int, np.integer)):
            index = int(index)
            if index < 0 or index > self.k_states:
                raise ValueError('Invalid index.')
            index = (index, index + 1)
        elif index is None:
            index = (index,)
        elif not isinstance(index, tuple):
            raise ValueError('Invalid index.')
        if len(index) > 2:
            raise ValueError('Cannot include a slice step in `index`.')
        index = self._states[slice(*index)]

        # Compatibility with zero-length slices (can make it easier to set up
        # initialization without lots of if statements)
        if len(index) == 0:
            return

        # Unset the values
        k_states = len(index)
        if k_states == self.k_states and self.initialization_type is not None:
            self.initialization_type = None
            self.constant[:] = 0
            self.stationary_cov[:] = 0
        elif index in self.blocks:
            for i in index:
                self._initialization[i] = None
            del self.blocks[index]
        else:
            raise ValueError('The given index does not correspond to a'
                             ' previously initialized block.')

    def clear(self):
        """
        Clear all previously set initializations, either global or block level
        """
        # Clear initializations
        for i in self._states:
            self._initialization[i] = None

        # Delete block initializations
        keys = list(self.blocks.keys())
        for key in keys:
            del self.blocks[key]

        # Clear global attributes
        self.initialization_type = None
        self.constant[:] = 0
        self.stationary_cov[:] = 0

    @property
    def initialized(self):
        return not (self.initialization_type is None and
                    np.any(np.equal(self._initialization, None)))

    def __call__(self, index=None, model=None, initial_state_mean=None,
                 initial_diffuse_state_cov=None,
                 initial_stationary_state_cov=None, complex_step=False):
        r"""
        Construct initialization representation

        Parameters
        ----------
        model : Representation, optional
            A state space model representation object, optional if 'stationary'
            initialization is used and ignored otherwise. See notes for
            details in the stationary initialization case.
        model_index : ndarray, optional
            The base index of the block in the model.
        initial_state_mean : ndarray, optional
            An array (or more usually view) in which to place the initial state
            mean.
        initial_diffuse_state_cov : ndarray, optional
            An array (or more usually view) in which to place the diffuse
            component of initial state covariance matrix.
        initial_stationary_state_cov : ndarray, optional
            An array (or more usually view) in which to place the stationary
            component of initial state covariance matrix.


        Returns
        -------
        initial_state_mean : ndarray
            Initial state mean, :math:`a_1^{(0)} = a`
        initial_diffuse_state_cov : ndarray
            Diffuse component of initial state covariance matrix,
            :math:`P_\infty = A A'`
        initial_stationary_state_cov : ndarray
            Stationary component of initial state covariance matrix,
            :math:`P_* = R_0 Q_0 R_0'`

        Notes
        -----
        If stationary initialization is used either globally or for any block
        of states, then either `model` or all of `state_intercept`,
        `transition`, `selection`, and `state_cov` must be provided.
        """
        # Check that all states are initialized somehow
        if (self.initialization_type is None and
                np.any(np.equal(self._initialization, None))):
            raise ValueError('Cannot construct initialization representation'
                             ' because not all states have been initialized.')

        # Setup indexes
        if index is None:
            index = self._states
            ix1 = np.s_[:]
            ix2 = np.s_[:, :]
        else:
            ix1 = np.s_[index[0]:index[-1] + 1]
            ix2 = np.ix_(index, index)

        # Retrieve state_intercept, etc. if `model` was given
        if model is not None:
            state_intercept = model['state_intercept', ix1, 0]
            transition = model[('transition',) + ix2 + (0,)]
            selection = model['selection', ix1, :, 0]
            state_cov = model['state_cov', :, :, 0]
            selected_state_cov = np.dot(selection, state_cov).dot(selection.T)

        # Create output arrays if not given
        if initial_state_mean is None:
            initial_state_mean = np.zeros(self.k_states)
        cov_shape = (self.k_states, self.k_states)
        if initial_diffuse_state_cov is None:
            initial_diffuse_state_cov = np.zeros(cov_shape)
        if initial_stationary_state_cov is None:
            initial_stationary_state_cov = np.zeros(cov_shape)

        # If using global initialization, compute the actual elements and
        # return them
        if self.initialization_type is not None:
            eye = np.eye(self.k_states)
            zeros = np.zeros((self.k_states, self.k_states))

            # General validation
            if self.initialization_type == 'stationary' and model is None:
                raise ValueError('Stationary initialization requires passing'
                                 ' either the `model` argument or all of the'
                                 ' individual transition equation arguments.')
            if self.initialization_type == 'stationary':
                # TODO performance
                eigvals = np.linalg.eigvals(transition)
                threshold = 1. - 1e-10
                if not np.max(np.abs(eigvals)) < threshold:
                    raise ValueError('Transition equation is not stationary,'
                                     ' and so stationary initialization cannot'
                                     ' be used.')

            # Set the initial state mean
            if self.initialization_type == 'stationary':
                # TODO performance
                initial_state_mean[ix1] = np.linalg.solve(eye - transition,
                                                          state_intercept)
            else:
                initial_state_mean[ix1] = self.constant

            # Set the diffuse component
            if self.initialization_type == 'diffuse':
                initial_diffuse_state_cov[ix2] = np.eye(self.k_states)
            else:
                initial_diffuse_state_cov[ix2] = zeros

            # Set the stationary component
            if self.initialization_type == 'known':
                initial_stationary_state_cov[ix2] = self.stationary_cov
            elif self.initialization_type == 'diffuse':
                initial_stationary_state_cov[ix2] = zeros
            elif self.initialization_type == 'approximate_diffuse':
                initial_stationary_state_cov[ix2] = (
                    eye * self.approximate_diffuse_variance)
            elif self.initialization_type == 'stationary':
                # TODO performance
                initial_stationary_state_cov[ix2] = (
                    tools.solve_discrete_lyapunov(transition,
                                                  selected_state_cov,
                                                  complex_step=complex_step))
        else:
            # Otherwise, if using blocks, recursively initialize
            # them (values will be set in-place)
            for block_index, init in self.blocks.items():
                init(index=tuple(np.array(index)[block_index, ]),
                     model=model, initial_state_mean=initial_state_mean,
                     initial_diffuse_state_cov=initial_diffuse_state_cov,
                     initial_stationary_state_cov=initial_stationary_state_cov)

        return (initial_state_mean, initial_diffuse_state_cov,
                initial_stationary_state_cov)
