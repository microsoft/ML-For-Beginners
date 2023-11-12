"""
State Space Representation, Kalman Filter, Smoother, and Simulation Smoother

Author: Chad Fulton
License: Simplified-BSD
"""

import numbers
import warnings
import numpy as np
from .kalman_smoother import KalmanSmoother
from .cfa_simulation_smoother import CFASimulationSmoother
from . import tools

SIMULATION_STATE = 0x01
SIMULATION_DISTURBANCE = 0x04
SIMULATION_ALL = (
    SIMULATION_STATE | SIMULATION_DISTURBANCE
)


# Based on scipy.states._qmc.check_random_state
def check_random_state(seed=None):
    """Turn `seed` into a `numpy.random.Generator` instance.
    Parameters
    ----------
    seed : {None, int, Generator, RandomState}, optional
        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``numpy.random.RandomState`` instance
        is used, seeded with `seed`.
        If `seed` is already a ``numpy.random.Generator`` or
        ``numpy.random.RandomState`` instance then that instance is used.
    Returns
    -------
    seed : {`numpy.random.Generator`, `numpy.random.RandomState`}
        Random number generator.
    """
    if seed is None or isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.default_rng(seed)
    elif isinstance(seed, (np.random.RandomState, np.random.Generator)):
        return seed
    else:
        raise ValueError(f'{seed!r} cannot be used to seed a'
                         ' numpy.random.Generator instance')


class SimulationSmoother(KalmanSmoother):
    r"""
    State space representation of a time series process, with Kalman filter
    and smoother, and with simulation smoother.

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
    simulation_smooth_results_class : class, optional
        Default results class to use to save output of simulation smoothing.
        Default is `SimulationSmoothResults`. If specified, class must extend
        from `SimulationSmoothResults`.
    simulation_smoother_classes : dict, optional
        Dictionary with BLAS prefixes as keys and classes as values.
    **kwargs
        Keyword arguments may be used to provide default values for state space
        matrices, for Kalman filtering options, for Kalman smoothing
        options, or for Simulation smoothing options.
        See `Representation`, `KalmanFilter`, and `KalmanSmoother` for more
        details.
    """

    simulation_outputs = [
        'simulate_state', 'simulate_disturbance', 'simulate_all'
    ]

    def __init__(self, k_endog, k_states, k_posdef=None,
                 simulation_smooth_results_class=None,
                 simulation_smoother_classes=None, **kwargs):
        super(SimulationSmoother, self).__init__(
            k_endog, k_states, k_posdef, **kwargs
        )

        if simulation_smooth_results_class is None:
            simulation_smooth_results_class = SimulationSmoothResults
        self.simulation_smooth_results_class = simulation_smooth_results_class

        self.prefix_simulation_smoother_map = (
            simulation_smoother_classes
            if simulation_smoother_classes is not None
            else tools.prefix_simulation_smoother_map.copy())

        # Holder for an model-level simulation smoother objects, to use in
        # simulating new time series.
        self._simulators = {}

    def get_simulation_output(self, simulation_output=None,
                              simulate_state=None, simulate_disturbance=None,
                              simulate_all=None, **kwargs):
        r"""
        Get simulation output bitmask

        Helper method to get final simulation output bitmask from a set of
        optional arguments including the bitmask itself and possibly boolean
        flags.

        Parameters
        ----------
        simulation_output : int, optional
            Simulation output bitmask. If this is specified, it is simply
            returned and the other arguments are ignored.
        simulate_state : bool, optional
            Whether or not to include the state in the simulation output.
        simulate_disturbance : bool, optional
            Whether or not to include the state and observation disturbances
            in the simulation output.
        simulate_all : bool, optional
            Whether or not to include all simulation output.
        \*\*kwargs
            Additional keyword arguments. Present so that calls to this method
            can use \*\*kwargs without clearing out additional arguments.
        """
        # If we do not explicitly have simulation_output, try to get it from
        # kwargs
        if simulation_output is None:
            simulation_output = 0

            if simulate_state:
                simulation_output |= SIMULATION_STATE
            if simulate_disturbance:
                simulation_output |= SIMULATION_DISTURBANCE
            if simulate_all:
                simulation_output |= SIMULATION_ALL

            # Handle case of no information in kwargs
            if simulation_output == 0:

                # If some arguments were passed, but we still do not have any
                # simulation output, raise an exception
                argument_set = not all([
                    simulate_state is None, simulate_disturbance is None,
                    simulate_all is None
                ])
                if argument_set:
                    raise ValueError("Invalid simulation output options:"
                                     " given options would result in no"
                                     " output.")

                # Otherwise set simulation output to be the same as smoother
                # output
                simulation_output = self.smoother_output

        return simulation_output

    def _simulate(self, nsimulations, simulator=None, random_state=None,
                  return_simulator=False, **kwargs):
        # Create the simulator, if necessary
        if simulator is None:
            simulator = self.simulator(nsimulations, random_state=random_state)

        # Perform simulation smoothing
        simulator.simulate(**kwargs)

        # Retrieve and return the objects of interest
        simulated_obs = np.array(simulator.generated_obs, copy=True)
        simulated_state = np.array(simulator.generated_state, copy=True)

        out = (simulated_obs.T[:nsimulations],
               simulated_state.T[:nsimulations])
        if return_simulator:
            out = out + (simulator,)
        return out

    def simulator(self, nsimulations, random_state=None):
        return self.simulation_smoother(simulation_output=0, method='kfs',
                                        nobs=nsimulations,
                                        random_state=random_state)

    def simulation_smoother(self, simulation_output=None, method='kfs',
                            results_class=None, prefix=None, nobs=-1,
                            random_state=None, **kwargs):
        r"""
        Retrieve a simulation smoother for the statespace model.

        Parameters
        ----------
        simulation_output : int, optional
            Determines which simulation smoother output is calculated.
            Default is all (including state and disturbances).
        method : {'kfs', 'cfa'}, optional
            Method for simulation smoothing. If `method='kfs'`, then the
            simulation smoother is based on Kalman filtering and smoothing
            recursions. If `method='cfa'`, then the simulation smoother is
            based on the Cholesky Factor Algorithm (CFA) approach. The CFA
            approach is not applicable to all state space models, but can be
            faster for the cases in which it is supported.
        results_class : class, optional
            Default results class to use to save output of simulation
            smoothing. Default is `SimulationSmoothResults`. If specified,
            class must extend from `SimulationSmoothResults`.
        prefix : str
            The prefix of the datatype. Usually only used internally.
        nobs : int
            The number of observations to simulate. If set to anything other
            than -1, only simulation will be performed (i.e. simulation
            smoothing will not be performed), so that only the `generated_obs`
            and `generated_state` attributes will be available.
        random_state : {None, int, Generator, RandomState}, optional
            If `seed` is None (or `np.random`), the `numpy.random.RandomState`
            singleton is used.
            If `seed` is an int, a new ``numpy.random.RandomState`` instance
            is used, seeded with `seed`.
            If `seed` is already a ``numpy.random.Generator`` or
            ``numpy.random.RandomState`` instance then that instance is used.
        **kwargs
            Additional keyword arguments, used to set the simulation output.
            See `set_simulation_output` for more details.

        Returns
        -------
        SimulationSmoothResults
        """
        method = method.lower()

        # Short-circuit for CFA
        if method == 'cfa':
            if simulation_output not in [None, 1, -1]:
                raise ValueError('Can only retrieve simulations of the state'
                                 ' vector using the CFA simulation smoother.')
            return CFASimulationSmoother(self)
        elif method != 'kfs':
            raise ValueError('Invalid simulation smoother method "%s". Valid'
                             ' methods are "kfs" or "cfa".' % method)

        # Set the class to be the default results class, if None provided
        if results_class is None:
            results_class = self.simulation_smooth_results_class

        # Instantiate a new results object
        if not issubclass(results_class, SimulationSmoothResults):
            raise ValueError('Invalid results class provided.')

        # Make sure we have the required Statespace representation
        prefix, dtype, create_smoother, create_filter, create_statespace = (
            self._initialize_smoother())

        # Simulation smoother parameters
        simulation_output = self.get_simulation_output(simulation_output,
                                                       **kwargs)

        # Kalman smoother parameters
        smoother_output = kwargs.get('smoother_output', simulation_output)

        # Kalman filter parameters
        filter_method = kwargs.get('filter_method', self.filter_method)
        inversion_method = kwargs.get('inversion_method',
                                      self.inversion_method)
        stability_method = kwargs.get('stability_method',
                                      self.stability_method)
        conserve_memory = kwargs.get('conserve_memory',
                                     self.conserve_memory)
        filter_timing = kwargs.get('filter_timing',
                                   self.filter_timing)
        loglikelihood_burn = kwargs.get('loglikelihood_burn',
                                        self.loglikelihood_burn)
        tolerance = kwargs.get('tolerance', self.tolerance)

        # Create a new simulation smoother object
        cls = self.prefix_simulation_smoother_map[prefix]
        simulation_smoother = cls(
            self._statespaces[prefix],
            filter_method, inversion_method, stability_method, conserve_memory,
            filter_timing, tolerance, loglikelihood_burn, smoother_output,
            simulation_output, nobs
        )

        # Create results object
        results = results_class(self, simulation_smoother,
                                random_state=random_state)

        return results


class SimulationSmoothResults:
    r"""
    Results from applying the Kalman smoother and/or filter to a state space
    model.

    Parameters
    ----------
    model : Representation
        A Statespace representation
    simulation_smoother : {{prefix}}SimulationSmoother object
        The Cython simulation smoother object with which to simulation smooth.
    random_state : {None, int, Generator, RandomState}, optional
        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``numpy.random.RandomState`` instance
        is used, seeded with `seed`.
        If `seed` is already a ``numpy.random.Generator`` or
        ``numpy.random.RandomState`` instance then that instance is used.

    Attributes
    ----------
    model : Representation
        A Statespace representation
    dtype : dtype
        Datatype of representation matrices
    prefix : str
        BLAS prefix of representation matrices
    simulation_output : int
        Bitmask controlling simulation output.
    simulate_state : bool
        Flag for if the state is included in simulation output.
    simulate_disturbance : bool
        Flag for if the state and observation disturbances are included in
        simulation output.
    simulate_all : bool
        Flag for if simulation output should include everything.
    generated_measurement_disturbance : ndarray
        Measurement disturbance variates used to genereate the observation
        vector.
    generated_state_disturbance : ndarray
        State disturbance variates used to genereate the state and
        observation vectors.
    generated_obs : ndarray
        Generated observation vector produced as a byproduct of simulation
        smoothing.
    generated_state : ndarray
        Generated state vector produced as a byproduct of simulation smoothing.
    simulated_state : ndarray
        Simulated state.
    simulated_measurement_disturbance : ndarray
        Simulated measurement disturbance.
    simulated_state_disturbance : ndarray
        Simulated state disturbance.
    """

    def __init__(self, model, simulation_smoother, random_state=None):
        self.model = model
        self.prefix = model.prefix
        self.dtype = model.dtype
        self._simulation_smoother = simulation_smoother
        self.random_state = check_random_state(random_state)

        # Output
        self._generated_measurement_disturbance = None
        self._generated_state_disturbance = None
        self._generated_obs = None
        self._generated_state = None
        self._simulated_state = None
        self._simulated_measurement_disturbance = None
        self._simulated_state_disturbance = None

    @property
    def simulation_output(self):
        return self._simulation_smoother.simulation_output

    @simulation_output.setter
    def simulation_output(self, value):
        self._simulation_smoother.simulation_output = value

    @property
    def simulate_state(self):
        return bool(self.simulation_output & SIMULATION_STATE)

    @simulate_state.setter
    def simulate_state(self, value):
        if bool(value):
            self.simulation_output = self.simulation_output | SIMULATION_STATE
        else:
            self.simulation_output = self.simulation_output & ~SIMULATION_STATE

    @property
    def simulate_disturbance(self):
        return bool(self.simulation_output & SIMULATION_DISTURBANCE)

    @simulate_disturbance.setter
    def simulate_disturbance(self, value):
        if bool(value):
            self.simulation_output = (
                self.simulation_output | SIMULATION_DISTURBANCE)
        else:
            self.simulation_output = (
                self.simulation_output & ~SIMULATION_DISTURBANCE)

    @property
    def simulate_all(self):
        return bool(self.simulation_output & SIMULATION_ALL)

    @simulate_all.setter
    def simulate_all(self, value):
        if bool(value):
            self.simulation_output = self.simulation_output | SIMULATION_ALL
        else:
            self.simulation_output = self.simulation_output & ~SIMULATION_ALL

    @property
    def generated_measurement_disturbance(self):
        r"""
        Randomly drawn measurement disturbance variates

        Used to construct `generated_obs`.

        Notes
        -----

        .. math::

           \varepsilon_t^+ ~ N(0, H_t)

        If `disturbance_variates` were provided to the `simulate()` method,
        then this returns those variates (which were N(0,1)) transformed to the
        distribution above.
        """
        if self._generated_measurement_disturbance is None:
            self._generated_measurement_disturbance = np.array(
                self._simulation_smoother.measurement_disturbance_variates,
                copy=True).reshape(self.model.nobs, self.model.k_endog)
        return self._generated_measurement_disturbance

    @property
    def generated_state_disturbance(self):
        r"""
        Randomly drawn state disturbance variates, used to construct
        `generated_state` and `generated_obs`.

        Notes
        -----

        .. math::

            \eta_t^+ ~ N(0, Q_t)

        If `disturbance_variates` were provided to the `simulate()` method,
        then this returns those variates (which were N(0,1)) transformed to the
        distribution above.
        """
        if self._generated_state_disturbance is None:
            self._generated_state_disturbance = np.array(
                self._simulation_smoother.state_disturbance_variates,
                copy=True).reshape(self.model.nobs, self.model.k_posdef)
        return self._generated_state_disturbance

    @property
    def generated_obs(self):
        r"""
        Generated vector of observations by iterating on the observation and
        transition equations, given a random initial state draw and random
        disturbance draws.

        Notes
        -----

        .. math::

            y_t^+ = d_t + Z_t \alpha_t^+ + \varepsilon_t^+
        """
        if self._generated_obs is None:
            self._generated_obs = np.array(
                self._simulation_smoother.generated_obs, copy=True
            )
        return self._generated_obs

    @property
    def generated_state(self):
        r"""
        Generated vector of states by iterating on the transition equation,
        given a random initial state draw and random disturbance draws.

        Notes
        -----

        .. math::

            \alpha_{t+1}^+ = c_t + T_t \alpha_t^+ + \eta_t^+
        """
        if self._generated_state is None:
            self._generated_state = np.array(
                self._simulation_smoother.generated_state, copy=True
            )
        return self._generated_state

    @property
    def simulated_state(self):
        r"""
        Random draw of the state vector from its conditional distribution.

        Notes
        -----

        .. math::

            \alpha ~ p(\alpha \mid Y_n)
        """
        if self._simulated_state is None:
            self._simulated_state = np.array(
                self._simulation_smoother.simulated_state, copy=True
            )
        return self._simulated_state

    @property
    def simulated_measurement_disturbance(self):
        r"""
        Random draw of the measurement disturbance vector from its conditional
        distribution.

        Notes
        -----

        .. math::

            \varepsilon ~ N(\hat \varepsilon, Var(\hat \varepsilon \mid Y_n))
        """
        if self._simulated_measurement_disturbance is None:
            self._simulated_measurement_disturbance = np.array(
                self._simulation_smoother.simulated_measurement_disturbance,
                copy=True
            )
        return self._simulated_measurement_disturbance

    @property
    def simulated_state_disturbance(self):
        r"""
        Random draw of the state disturbanc e vector from its conditional
        distribution.

        Notes
        -----

        .. math::

            \eta ~ N(\hat \eta, Var(\hat \eta \mid Y_n))
        """
        if self._simulated_state_disturbance is None:
            self._simulated_state_disturbance = np.array(
                self._simulation_smoother.simulated_state_disturbance,
                copy=True
            )
        return self._simulated_state_disturbance

    def simulate(self, simulation_output=-1,
                 disturbance_variates=None,
                 measurement_disturbance_variates=None,
                 state_disturbance_variates=None,
                 initial_state_variates=None,
                 pretransformed=None,
                 pretransformed_measurement_disturbance_variates=None,
                 pretransformed_state_disturbance_variates=None,
                 pretransformed_initial_state_variates=False,
                 random_state=None):
        r"""
        Perform simulation smoothing

        Does not return anything, but populates the object's `simulated_*`
        attributes, as specified by simulation output.

        Parameters
        ----------
        simulation_output : int, optional
            Bitmask controlling simulation output. Default is to use the
            simulation output defined in object initialization.
        measurement_disturbance_variates : array_like, optional
            If specified, these are the shocks to the measurement equation,
            :math:`\varepsilon_t`. If unspecified, these are automatically
            generated using a pseudo-random number generator. If specified,
            must be shaped `nsimulations` x `k_endog`, where `k_endog` is the
            same as in the state space model.
        state_disturbance_variates : array_like, optional
            If specified, these are the shocks to the state equation,
            :math:`\eta_t`. If unspecified, these are automatically
            generated using a pseudo-random number generator. If specified,
            must be shaped `nsimulations` x `k_posdef` where `k_posdef` is the
            same as in the state space model.
        initial_state_variates : array_like, optional
            If specified, this is the state vector at time zero, which should
            be shaped (`k_states` x 1), where `k_states` is the same as in the
            state space model. If unspecified, but the model has been
            initialized, then that initialization is used.
        initial_state_variates : array_likes, optional
            Random values to use as initial state variates. Usually only
            specified if results are to be replicated (e.g. to enforce a seed)
            or for testing. If not specified, random variates are drawn.
        pretransformed_measurement_disturbance_variates : bool, optional
            If `measurement_disturbance_variates` is provided, this flag
            indicates whether it should be directly used as the shocks. If
            False, then it is assumed to contain draws from the standard Normal
            distribution that must be transformed using the `obs_cov`
            covariance matrix. Default is False.
        pretransformed_state_disturbance_variates : bool, optional
            If `state_disturbance_variates` is provided, this flag indicates
            whether it should be directly used as the shocks. If False, then it
            is assumed to contain draws from the standard Normal distribution
            that must be transformed using the `state_cov` covariance matrix.
            Default is False.
        pretransformed_initial_state_variates : bool, optional
            If `initial_state_variates` is provided, this flag indicates
            whether it should be directly used as the initial_state. If False,
            then it is assumed to contain draws from the standard Normal
            distribution that must be transformed using the `initial_state_cov`
            covariance matrix. Default is False.
        random_state : {None, int, Generator, RandomState}, optional
            If `seed` is None (or `np.random`), the `numpy.random.RandomState`
            singleton is used.
            If `seed` is an int, a new ``numpy.random.RandomState`` instance
            is used, seeded with `seed`.
            If `seed` is already a ``numpy.random.Generator`` or
            ``numpy.random.RandomState`` instance then that instance is used.
        disturbance_variates : bool, optional
            Deprecated, please use pretransformed_measurement_shocks and
            pretransformed_state_shocks instead.

            .. deprecated:: 0.14.0

               Use ``measurement_disturbance_variates`` and
               ``state_disturbance_variates`` as replacements.

        pretransformed : bool, optional
            Deprecated, please use pretransformed_measurement_shocks and
            pretransformed_state_shocks instead.

            .. deprecated:: 0.14.0

               Use ``pretransformed_measurement_disturbance_variates`` and
               ``pretransformed_state_disturbance_variates`` as replacements.
        """
        # Handle deprecated argumennts
        if disturbance_variates is not None:
            msg = ('`disturbance_variates` keyword is deprecated, use'
                   ' `measurement_disturbance_variates` and'
                   ' `state_disturbance_variates` instead.')
            warnings.warn(msg, FutureWarning)
            if (measurement_disturbance_variates is not None
                    or state_disturbance_variates is not None):
                raise ValueError('Cannot use `disturbance_variates` in'
                                 ' combination with '
                                 ' `measurement_disturbance_variates` or'
                                 ' `state_disturbance_variates`.')
            if disturbance_variates is not None:
                disturbance_variates = disturbance_variates.ravel()
                n_mds = self.model.nobs * self.model.k_endog
                measurement_disturbance_variates = disturbance_variates[:n_mds]
                state_disturbance_variates = disturbance_variates[n_mds:]
        if pretransformed is not None:
            msg = ('`pretransformed` keyword is deprecated, use'
                   ' `pretransformed_measurement_disturbance_variates` and'
                   ' `pretransformed_state_disturbance_variates` instead.')
            warnings.warn(msg, FutureWarning)
            if (pretransformed_measurement_disturbance_variates is not None
                    or pretransformed_state_disturbance_variates is not None):
                raise ValueError(
                    'Cannot use `pretransformed` in combination with '
                    ' `pretransformed_measurement_disturbance_variates` or'
                    ' `pretransformed_state_disturbance_variates`.')
            if pretransformed is not None:
                pretransformed_measurement_disturbance_variates = (
                    pretransformed)
                pretransformed_state_disturbance_variates = pretransformed

        if pretransformed_measurement_disturbance_variates is None:
            pretransformed_measurement_disturbance_variates = False
        if pretransformed_state_disturbance_variates is None:
            pretransformed_state_disturbance_variates = False

        # Clear any previous output
        self._generated_measurement_disturbance = None
        self._generated_state_disturbance = None
        self._generated_state = None
        self._generated_obs = None
        self._generated_state = None
        self._simulated_state = None
        self._simulated_measurement_disturbance = None
        self._simulated_state_disturbance = None

        # Handle the random state
        if random_state is None:
            random_state = self.random_state
        else:
            random_state = check_random_state(random_state)

        # Re-initialize the _statespace representation
        prefix, dtype, create_smoother, create_filter, create_statespace = (
            self.model._initialize_smoother())
        if create_statespace:
            raise ValueError('The simulation smoother currently cannot replace'
                             ' the underlying _{{prefix}}Representation model'
                             ' object if it changes (which happens e.g. if the'
                             ' dimensions of some system matrices change.')

        # Initialize the state
        self.model._initialize_state(prefix=prefix)

        # Draw the (independent) random variates for disturbances in the
        # simulation
        if measurement_disturbance_variates is not None:
            self._simulation_smoother.set_measurement_disturbance_variates(
                np.array(measurement_disturbance_variates,
                         dtype=self.dtype).ravel(),
                pretransformed=pretransformed_measurement_disturbance_variates
            )
        else:
            self._simulation_smoother.draw_measurement_disturbance_variates(
                random_state)

        # Draw the (independent) random variates for disturbances in the
        # simulation
        if state_disturbance_variates is not None:
            self._simulation_smoother.set_state_disturbance_variates(
                np.array(state_disturbance_variates, dtype=self.dtype).ravel(),
                pretransformed=pretransformed_state_disturbance_variates
            )
        else:
            self._simulation_smoother.draw_state_disturbance_variates(
                random_state)

        # Draw the (independent) random variates for the initial states in the
        # simulation
        if initial_state_variates is not None:
            if pretransformed_initial_state_variates:
                self._simulation_smoother.set_initial_state(
                    np.array(initial_state_variates, dtype=self.dtype)
                )
            else:
                self._simulation_smoother.set_initial_state_variates(
                    np.array(initial_state_variates, dtype=self.dtype),
                    pretransformed=False
                )
            # Note: there is a third option, which is to set the initial state
            # variates with pretransformed = True. However, this option simply
            # eliminates the multiplication by the Cholesky factor of the
            # initial state cov, but still adds the initial state mean. It's
            # not clear when this would be useful...
        else:
            self._simulation_smoother.draw_initial_state_variates(
                random_state)

        # Perform simulation smoothing
        self._simulation_smoother.simulate(simulation_output)
