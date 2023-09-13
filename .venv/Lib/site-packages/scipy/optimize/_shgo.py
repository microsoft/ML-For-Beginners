"""shgo: The simplicial homology global optimisation algorithm."""
from collections import namedtuple
import time
import logging
import warnings
import sys

import numpy as np

from scipy import spatial
from scipy.optimize import OptimizeResult, minimize, Bounds
from scipy.optimize._optimize import MemoizeJac
from scipy.optimize._constraints import new_bounds_to_old
from scipy.optimize._minimize import standardize_constraints
from scipy._lib._util import _FunctionWrapper

from scipy.optimize._shgo_lib._complex import Complex

__all__ = ['shgo']


def shgo(
    func, bounds, args=(), constraints=None, n=100, iters=1, callback=None,
    minimizer_kwargs=None, options=None, sampling_method='simplicial', *,
    workers=1
):
    """
    Finds the global minimum of a function using SHG optimization.

    SHGO stands for "simplicial homology global optimization".

    Parameters
    ----------
    func : callable
        The objective function to be minimized.  Must be in the form
        ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array
        and ``args`` is a tuple of any additional fixed parameters needed to
        completely specify the function.
    bounds : sequence or `Bounds`
        Bounds for variables. There are two ways to specify the bounds:

        1. Instance of `Bounds` class.
        2. Sequence of ``(min, max)`` pairs for each element in `x`.

    args : tuple, optional
        Any additional fixed parameters needed to completely specify the
        objective function.
    constraints : {Constraint, dict} or List of {Constraint, dict}, optional
        Constraints definition. Only for COBYLA, SLSQP and trust-constr.
        See the tutorial [5]_ for further details on specifying constraints.

        .. note::

           Only COBYLA, SLSQP, and trust-constr local minimize methods
           currently support constraint arguments. If the ``constraints``
           sequence used in the local optimization problem is not defined in
           ``minimizer_kwargs`` and a constrained method is used then the
           global ``constraints`` will be used.
           (Defining a ``constraints`` sequence in ``minimizer_kwargs``
           means that ``constraints`` will not be added so if equality
           constraints and so forth need to be added then the inequality
           functions in ``constraints`` need to be added to
           ``minimizer_kwargs`` too).
           COBYLA only supports inequality constraints.

        .. versionchanged:: 1.11.0

           ``constraints`` accepts `NonlinearConstraint`, `LinearConstraint`.

    n : int, optional
        Number of sampling points used in the construction of the simplicial
        complex. For the default ``simplicial`` sampling method 2**dim + 1
        sampling points are generated instead of the default `n=100`. For all
        other specified values `n` sampling points are generated. For
        ``sobol``, ``halton`` and other arbitrary `sampling_methods` `n=100` or
        another speciefied number of sampling points are generated.
    iters : int, optional
        Number of iterations used in the construction of the simplicial
        complex. Default is 1.
    callback : callable, optional
        Called after each iteration, as ``callback(xk)``, where ``xk`` is the
        current parameter vector.
    minimizer_kwargs : dict, optional
        Extra keyword arguments to be passed to the minimizer
        ``scipy.optimize.minimize`` Some important options could be:

            * method : str
                The minimization method. If not given, chosen to be one of
                BFGS, L-BFGS-B, SLSQP, depending on whether or not the
                problem has constraints or bounds.
            * args : tuple
                Extra arguments passed to the objective function (``func``) and
                its derivatives (Jacobian, Hessian).
            * options : dict, optional
                Note that by default the tolerance is specified as
                ``{ftol: 1e-12}``

    options : dict, optional
        A dictionary of solver options. Many of the options specified for the
        global routine are also passed to the scipy.optimize.minimize routine.
        The options that are also passed to the local routine are marked with
        "(L)".

        Stopping criteria, the algorithm will terminate if any of the specified
        criteria are met. However, the default algorithm does not require any
        to be specified:

        * maxfev : int (L)
            Maximum number of function evaluations in the feasible domain.
            (Note only methods that support this option will terminate
            the routine at precisely exact specified value. Otherwise the
            criterion will only terminate during a global iteration)
        * f_min
            Specify the minimum objective function value, if it is known.
        * f_tol : float
            Precision goal for the value of f in the stopping
            criterion. Note that the global routine will also
            terminate if a sampling point in the global routine is
            within this tolerance.
        * maxiter : int
            Maximum number of iterations to perform.
        * maxev : int
            Maximum number of sampling evaluations to perform (includes
            searching in infeasible points).
        * maxtime : float
            Maximum processing runtime allowed
        * minhgrd : int
            Minimum homology group rank differential. The homology group of the
            objective function is calculated (approximately) during every
            iteration. The rank of this group has a one-to-one correspondence
            with the number of locally convex subdomains in the objective
            function (after adequate sampling points each of these subdomains
            contain a unique global minimum). If the difference in the hgr is 0
            between iterations for ``maxhgrd`` specified iterations the
            algorithm will terminate.

        Objective function knowledge:

        * symmetry : list or bool
            Specify if the objective function contains symmetric variables.
            The search space (and therefore performance) is decreased by up to
            O(n!) times in the fully symmetric case. If `True` is specified
            then all variables will be set symmetric to the first variable.
            Default
            is set to False.

            E.g.  f(x) = (x_1 + x_2 + x_3) + (x_4)**2 + (x_5)**2 + (x_6)**2

            In this equation x_2 and x_3 are symmetric to x_1, while x_5 and
            x_6 are symmetric to x_4, this can be specified to the solver as:

            symmetry = [0,  # Variable 1
                        0,  # symmetric to variable 1
                        0,  # symmetric to variable 1
                        3,  # Variable 4
                        3,  # symmetric to variable 4
                        3,  # symmetric to variable 4
                        ]

        * jac : bool or callable, optional
            Jacobian (gradient) of objective function. Only for CG, BFGS,
            Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg. If ``jac`` is a
            boolean and is True, ``fun`` is assumed to return the gradient
            along with the objective function. If False, the gradient will be
            estimated numerically. ``jac`` can also be a callable returning the
            gradient of the objective. In this case, it must accept the same
            arguments as ``fun``. (Passed to `scipy.optimize.minmize`
            automatically)

        * hess, hessp : callable, optional
            Hessian (matrix of second-order derivatives) of objective function
            or Hessian of objective function times an arbitrary vector p.
            Only for Newton-CG, dogleg, trust-ncg. Only one of ``hessp`` or
            ``hess`` needs to be given. If ``hess`` is provided, then
            ``hessp`` will be ignored. If neither ``hess`` nor ``hessp`` is
            provided, then the Hessian product will be approximated using
            finite differences on ``jac``. ``hessp`` must compute the Hessian
            times an arbitrary vector. (Passed to `scipy.optimize.minmize`
            automatically)

        Algorithm settings:

        * minimize_every_iter : bool
            If True then promising global sampling points will be passed to a
            local minimization routine every iteration. If True then only the
            final minimizer pool will be run. Defaults to True.
        * local_iter : int
            Only evaluate a few of the best minimizer pool candidates every
            iteration. If False all potential points are passed to the local
            minimization routine.
        * infty_constraints : bool
            If True then any sampling points generated which are outside will
            the feasible domain will be saved and given an objective function
            value of ``inf``. If False then these points will be discarded.
            Using this functionality could lead to higher performance with
            respect to function evaluations before the global minimum is found,
            specifying False will use less memory at the cost of a slight
            decrease in performance. Defaults to True.

        Feedback:

        * disp : bool (L)
            Set to True to print convergence messages.

    sampling_method : str or function, optional
        Current built in sampling method options are ``halton``, ``sobol`` and
        ``simplicial``. The default ``simplicial`` provides
        the theoretical guarantee of convergence to the global minimum in
        finite time. ``halton`` and ``sobol`` method are faster in terms of
        sampling point generation at the cost of the loss of
        guaranteed convergence. It is more appropriate for most "easier"
        problems where the convergence is relatively fast.
        User defined sampling functions must accept two arguments of ``n``
        sampling points of dimension ``dim`` per call and output an array of
        sampling points with shape `n x dim`.

    workers : int or map-like callable, optional
        Sample and run the local serial minimizations in parallel.
        Supply -1 to use all available CPU cores, or an int to use
        that many Processes (uses `multiprocessing.Pool <multiprocessing>`).

        Alternatively supply a map-like callable, such as
        `multiprocessing.Pool.map` for parallel evaluation.
        This evaluation is carried out as ``workers(func, iterable)``.
        Requires that `func` be pickleable.

        .. versionadded:: 1.11.0

    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a `OptimizeResult` object.
        Important attributes are:
        ``x`` the solution array corresponding to the global minimum,
        ``fun`` the function output at the global solution,
        ``xl`` an ordered list of local minima solutions,
        ``funl`` the function output at the corresponding local solutions,
        ``success`` a Boolean flag indicating if the optimizer exited
        successfully,
        ``message`` which describes the cause of the termination,
        ``nfev`` the total number of objective function evaluations including
        the sampling calls,
        ``nlfev`` the total number of objective function evaluations
        culminating from all local search optimizations,
        ``nit`` number of iterations performed by the global routine.

    Notes
    -----
    Global optimization using simplicial homology global optimization [1]_.
    Appropriate for solving general purpose NLP and blackbox optimization
    problems to global optimality (low-dimensional problems).

    In general, the optimization problems are of the form::

        minimize f(x) subject to

        g_i(x) >= 0,  i = 1,...,m
        h_j(x)  = 0,  j = 1,...,p

    where x is a vector of one or more variables. ``f(x)`` is the objective
    function ``R^n -> R``, ``g_i(x)`` are the inequality constraints, and
    ``h_j(x)`` are the equality constraints.

    Optionally, the lower and upper bounds for each element in x can also be
    specified using the `bounds` argument.

    While most of the theoretical advantages of SHGO are only proven for when
    ``f(x)`` is a Lipschitz smooth function, the algorithm is also proven to
    converge to the global optimum for the more general case where ``f(x)`` is
    non-continuous, non-convex and non-smooth, if the default sampling method
    is used [1]_.

    The local search method may be specified using the ``minimizer_kwargs``
    parameter which is passed on to ``scipy.optimize.minimize``. By default,
    the ``SLSQP`` method is used. In general, it is recommended to use the
    ``SLSQP`` or ``COBYLA`` local minimization if inequality constraints
    are defined for the problem since the other methods do not use constraints.

    The ``halton`` and ``sobol`` method points are generated using
    `scipy.stats.qmc`. Any other QMC method could be used.

    References
    ----------
    .. [1] Endres, SC, Sandrock, C, Focke, WW (2018) "A simplicial homology
           algorithm for lipschitz optimisation", Journal of Global
           Optimization.
    .. [2] Joe, SW and Kuo, FY (2008) "Constructing Sobol' sequences with
           better  two-dimensional projections", SIAM J. Sci. Comput. 30,
           2635-2654.
    .. [3] Hock, W and Schittkowski, K (1981) "Test examples for nonlinear
           programming codes", Lecture Notes in Economics and Mathematical
           Systems, 187. Springer-Verlag, New York.
           http://www.ai7.uni-bayreuth.de/test_problem_coll.pdf
    .. [4] Wales, DJ (2015) "Perspective: Insight into reaction coordinates and
           dynamics from the potential energy landscape",
           Journal of Chemical Physics, 142(13), 2015.
    .. [5] https://docs.scipy.org/doc/scipy/tutorial/optimize.html#constrained-minimization-of-multivariate-scalar-functions-minimize

    Examples
    --------
    First consider the problem of minimizing the Rosenbrock function, `rosen`:

    >>> from scipy.optimize import rosen, shgo
    >>> bounds = [(0,2), (0, 2), (0, 2), (0, 2), (0, 2)]
    >>> result = shgo(rosen, bounds)
    >>> result.x, result.fun
    (array([1., 1., 1., 1., 1.]), 2.920392374190081e-18)

    Note that bounds determine the dimensionality of the objective
    function and is therefore a required input, however you can specify
    empty bounds using ``None`` or objects like ``np.inf`` which will be
    converted to large float numbers.

    >>> bounds = [(None, None), ]*4
    >>> result = shgo(rosen, bounds)
    >>> result.x
    array([0.99999851, 0.99999704, 0.99999411, 0.9999882 ])

    Next, we consider the Eggholder function, a problem with several local
    minima and one global minimum. We will demonstrate the use of arguments and
    the capabilities of `shgo`.
    (https://en.wikipedia.org/wiki/Test_functions_for_optimization)

    >>> import numpy as np
    >>> def eggholder(x):
    ...     return (-(x[1] + 47.0)
    ...             * np.sin(np.sqrt(abs(x[0]/2.0 + (x[1] + 47.0))))
    ...             - x[0] * np.sin(np.sqrt(abs(x[0] - (x[1] + 47.0))))
    ...             )
    ...
    >>> bounds = [(-512, 512), (-512, 512)]

    `shgo` has built-in low discrepancy sampling sequences. First, we will
    input 64 initial sampling points of the *Sobol'* sequence:

    >>> result = shgo(eggholder, bounds, n=64, sampling_method='sobol')
    >>> result.x, result.fun
    (array([512.        , 404.23180824]), -959.6406627208397)

    `shgo` also has a return for any other local minima that was found, these
    can be called using:

    >>> result.xl
    array([[ 512.        ,  404.23180824],
           [ 283.0759062 , -487.12565635],
           [-294.66820039, -462.01964031],
           [-105.87688911,  423.15323845],
           [-242.97926   ,  274.38030925],
           [-506.25823477,    6.3131022 ],
           [-408.71980731, -156.10116949],
           [ 150.23207937,  301.31376595],
           [  91.00920901, -391.283763  ],
           [ 202.89662724, -269.38043241],
           [ 361.66623976, -106.96493868],
           [-219.40612786, -244.06020508]])

    >>> result.funl
    array([-959.64066272, -718.16745962, -704.80659592, -565.99778097,
           -559.78685655, -557.36868733, -507.87385942, -493.9605115 ,
           -426.48799655, -421.15571437, -419.31194957, -410.98477763])

    These results are useful in applications where there are many global minima
    and the values of other global minima are desired or where the local minima
    can provide insight into the system (for example morphologies
    in physical chemistry [4]_).

    If we want to find a larger number of local minima, we can increase the
    number of sampling points or the number of iterations. We'll increase the
    number of sampling points to 64 and the number of iterations from the
    default of 1 to 3. Using ``simplicial`` this would have given us
    64 x 3 = 192 initial sampling points.

    >>> result_2 = shgo(eggholder,
    ...                 bounds, n=64, iters=3, sampling_method='sobol')
    >>> len(result.xl), len(result_2.xl)
    (12, 23)

    Note the difference between, e.g., ``n=192, iters=1`` and ``n=64,
    iters=3``.
    In the first case the promising points contained in the minimiser pool
    are processed only once. In the latter case it is processed every 64
    sampling points for a total of 3 times.

    To demonstrate solving problems with non-linear constraints consider the
    following example from Hock and Schittkowski problem 73 (cattle-feed)
    [3]_::

        minimize: f = 24.55 * x_1 + 26.75 * x_2 + 39 * x_3 + 40.50 * x_4

        subject to: 2.3 * x_1 + 5.6 * x_2 + 11.1 * x_3 + 1.3 * x_4 - 5    >= 0,

                    12 * x_1 + 11.9 * x_2 + 41.8 * x_3 + 52.1 * x_4 - 21
                        -1.645 * sqrt(0.28 * x_1**2 + 0.19 * x_2**2 +
                                      20.5 * x_3**2 + 0.62 * x_4**2)      >= 0,

                    x_1 + x_2 + x_3 + x_4 - 1                             == 0,

                    1 >= x_i >= 0 for all i

    The approximate answer given in [3]_ is::

        f([0.6355216, -0.12e-11, 0.3127019, 0.05177655]) = 29.894378

    >>> def f(x):  # (cattle-feed)
    ...     return 24.55*x[0] + 26.75*x[1] + 39*x[2] + 40.50*x[3]
    ...
    >>> def g1(x):
    ...     return 2.3*x[0] + 5.6*x[1] + 11.1*x[2] + 1.3*x[3] - 5  # >=0
    ...
    >>> def g2(x):
    ...     return (12*x[0] + 11.9*x[1] +41.8*x[2] + 52.1*x[3] - 21
    ...             - 1.645 * np.sqrt(0.28*x[0]**2 + 0.19*x[1]**2
    ...                             + 20.5*x[2]**2 + 0.62*x[3]**2)
    ...             ) # >=0
    ...
    >>> def h1(x):
    ...     return x[0] + x[1] + x[2] + x[3] - 1  # == 0
    ...
    >>> cons = ({'type': 'ineq', 'fun': g1},
    ...         {'type': 'ineq', 'fun': g2},
    ...         {'type': 'eq', 'fun': h1})
    >>> bounds = [(0, 1.0),]*4
    >>> res = shgo(f, bounds, n=150, constraints=cons)
    >>> res
     message: Optimization terminated successfully.
     success: True
         fun: 29.894378159142136
        funl: [ 2.989e+01]
           x: [ 6.355e-01  1.137e-13  3.127e-01  5.178e-02]
          xl: [[ 6.355e-01  1.137e-13  3.127e-01  5.178e-02]]
         nit: 1
        nfev: 142
       nlfev: 35
       nljev: 5
       nlhev: 0

    >>> g1(res.x), g2(res.x), h1(res.x)
    (-5.062616992290714e-14, -2.9594104944408173e-12, 0.0)

    """
    # if necessary, convert bounds class to old bounds
    if isinstance(bounds, Bounds):
        bounds = new_bounds_to_old(bounds.lb, bounds.ub, len(bounds.lb))

    # Initiate SHGO class
    # use in context manager to make sure that any parallelization
    # resources are freed.
    with SHGO(func, bounds, args=args, constraints=constraints, n=n,
               iters=iters, callback=callback,
               minimizer_kwargs=minimizer_kwargs,
               options=options, sampling_method=sampling_method,
               workers=workers) as shc:
        # Run the algorithm, process results and test success
        shc.iterate_all()

    if not shc.break_routine:
        if shc.disp:
            logging.info("Successfully completed construction of complex.")

    # Test post iterations success
    if len(shc.LMC.xl_maps) == 0:
        # If sampling failed to find pool, return lowest sampled point
        # with a warning
        shc.find_lowest_vertex()
        shc.break_routine = True
        shc.fail_routine(mes="Failed to find a feasible minimizer point. "
                             "Lowest sampling point = {}".format(shc.f_lowest))
        shc.res.fun = shc.f_lowest
        shc.res.x = shc.x_lowest
        shc.res.nfev = shc.fn
        shc.res.tnev = shc.n_sampled
    else:
        # Test that the optimal solutions do not violate any constraints
        pass  # TODO

    # Confirm the routine ran successfully
    if not shc.break_routine:
        shc.res.message = 'Optimization terminated successfully.'
        shc.res.success = True

    # Return the final results
    return shc.res


class SHGO:
    def __init__(self, func, bounds, args=(), constraints=None, n=None,
                 iters=None, callback=None, minimizer_kwargs=None,
                 options=None, sampling_method='simplicial', workers=1):
        from scipy.stats import qmc
        # Input checks
        methods = ['halton', 'sobol', 'simplicial']
        if isinstance(sampling_method, str) and sampling_method not in methods:
            raise ValueError(("Unknown sampling_method specified."
                              " Valid methods: {}").format(', '.join(methods)))

        # Split obj func if given with Jac
        try:
            if ((minimizer_kwargs['jac'] is True) and
                    (not callable(minimizer_kwargs['jac']))):
                self.func = MemoizeJac(func)
                jac = self.func.derivative
                minimizer_kwargs['jac'] = jac
                func = self.func  # .fun
            else:
                self.func = func  # Normal definition of objective function
        except (TypeError, KeyError):
            self.func = func  # Normal definition of objective function

        # Initiate class
        self.func = _FunctionWrapper(func, args)
        self.bounds = bounds
        self.args = args
        self.callback = callback

        # Bounds
        abound = np.array(bounds, float)
        self.dim = np.shape(abound)[0]  # Dimensionality of problem

        # Set none finite values to large floats
        infind = ~np.isfinite(abound)
        abound[infind[:, 0], 0] = -1e50
        abound[infind[:, 1], 1] = 1e50

        # Check if bounds are correctly specified
        bnderr = abound[:, 0] > abound[:, 1]
        if bnderr.any():
            raise ValueError('Error: lb > ub in bounds {}.'
                             .format(', '.join(str(b) for b in bnderr)))

        self.bounds = abound

        # Constraints
        # Process constraint dict sequence:
        self.constraints = constraints
        if constraints is not None:
            self.min_cons = constraints
            self.g_cons = []
            self.g_args = []

            # shgo internals deals with old-style constraints
            # self.constraints is used to create Complex, so need
            # to be stored internally in old-style.
            # `minimize` takes care of normalising these constraints
            # for slsqp/cobyla/trust-constr.
            self.constraints = standardize_constraints(
                constraints,
                np.empty(self.dim, float),
                'old'
            )
            for cons in self.constraints:
                if cons['type'] in ('ineq'):
                    self.g_cons.append(cons['fun'])
                    try:
                        self.g_args.append(cons['args'])
                    except KeyError:
                        self.g_args.append(())
            self.g_cons = tuple(self.g_cons)
            self.g_args = tuple(self.g_args)
        else:
            self.g_cons = None
            self.g_args = None

        # Define local minimization keyword arguments
        # Start with defaults
        self.minimizer_kwargs = {'method': 'SLSQP',
                                 'bounds': self.bounds,
                                 'options': {},
                                 'callback': self.callback
                                 }
        if minimizer_kwargs is not None:
            # Overwrite with supplied values
            self.minimizer_kwargs.update(minimizer_kwargs)

        else:
            self.minimizer_kwargs['options'] = {'ftol': 1e-12}

        if (
            self.minimizer_kwargs['method'].lower() in ('slsqp', 'cobyla', 'trust-constr') and
            (
                minimizer_kwargs is not None and
                'constraints' not in minimizer_kwargs and
                constraints is not None
            ) or
            (self.g_cons is not None)
        ):
            self.minimizer_kwargs['constraints'] = self.min_cons

        # Process options dict
        if options is not None:
            self.init_options(options)
        else:  # Default settings:
            self.f_min_true = None
            self.minimize_every_iter = True

            # Algorithm limits
            self.maxiter = None
            self.maxfev = None
            self.maxev = None
            self.maxtime = None
            self.f_min_true = None
            self.minhgrd = None

            # Objective function knowledge
            self.symmetry = None

            # Algorithm functionality
            self.infty_cons_sampl = True
            self.local_iter = False

            # Feedback
            self.disp = False

        # Remove unknown arguments in self.minimizer_kwargs
        # Start with arguments all the solvers have in common
        self.min_solver_args = ['fun', 'x0', 'args',
                                'callback', 'options', 'method']
        # then add the ones unique to specific solvers
        solver_args = {
            '_custom': ['jac', 'hess', 'hessp', 'bounds', 'constraints'],
            'nelder-mead': [],
            'powell': [],
            'cg': ['jac'],
            'bfgs': ['jac'],
            'newton-cg': ['jac', 'hess', 'hessp'],
            'l-bfgs-b': ['jac', 'bounds'],
            'tnc': ['jac', 'bounds'],
            'cobyla': ['constraints', 'catol'],
            'slsqp': ['jac', 'bounds', 'constraints'],
            'dogleg': ['jac', 'hess'],
            'trust-ncg': ['jac', 'hess', 'hessp'],
            'trust-krylov': ['jac', 'hess', 'hessp'],
            'trust-exact': ['jac', 'hess'],
            'trust-constr': ['jac', 'hess', 'hessp', 'constraints'],
        }
        method = self.minimizer_kwargs['method']
        self.min_solver_args += solver_args[method.lower()]

        # Only retain the known arguments
        def _restrict_to_keys(dictionary, goodkeys):
            """Remove keys from dictionary if not in goodkeys - inplace"""
            existingkeys = set(dictionary)
            for key in existingkeys - set(goodkeys):
                dictionary.pop(key, None)

        _restrict_to_keys(self.minimizer_kwargs, self.min_solver_args)
        _restrict_to_keys(self.minimizer_kwargs['options'],
                          self.min_solver_args + ['ftol'])

        # Algorithm controls
        # Global controls
        self.stop_global = False  # Used in the stopping_criteria method
        self.break_routine = False  # Break the algorithm globally
        self.iters = iters  # Iterations to be ran
        self.iters_done = 0  # Iterations completed
        self.n = n  # Sampling points per iteration
        self.nc = 0  # n  # Sampling points to sample in current iteration
        self.n_prc = 0  # Processed points (used to track Delaunay iters)
        self.n_sampled = 0  # To track no. of sampling points already generated
        self.fn = 0  # Number of feasible sampling points evaluations performed
        self.hgr = 0  # Homology group rank
        # Initially attempt to build the triangulation incrementally:
        self.qhull_incremental = True

        # Default settings if no sampling criteria.
        if (self.n is None) and (self.iters is None) \
                and (sampling_method == 'simplicial'):
            self.n = 2 ** self.dim + 1
            self.nc = 0  # self.n
        if self.iters is None:
            self.iters = 1
        if (self.n is None) and not (sampling_method == 'simplicial'):
            self.n = self.n = 100
            self.nc = 0  # self.n
        if (self.n == 100) and (sampling_method == 'simplicial'):
            self.n = 2 ** self.dim + 1

        if not ((self.maxiter is None) and (self.maxfev is None) and (
                self.maxev is None)
                and (self.minhgrd is None) and (self.f_min_true is None)):
            self.iters = None

        # Set complex construction mode based on a provided stopping criteria:
        # Initialise sampling Complex and function cache
        # Note that sfield_args=() since args are already wrapped in self.func
        # using the_FunctionWrapper class.
        self.HC = Complex(dim=self.dim, domain=self.bounds,
                          sfield=self.func, sfield_args=(),
                          symmetry=self.symmetry,
                          constraints=self.constraints,
                          workers=workers)

        # Choose complex constructor
        if sampling_method == 'simplicial':
            self.iterate_complex = self.iterate_hypercube
            self.sampling_method = sampling_method

        elif sampling_method in ['halton', 'sobol'] or \
                not isinstance(sampling_method, str):
            self.iterate_complex = self.iterate_delaunay
            # Sampling method used
            if sampling_method in ['halton', 'sobol']:
                if sampling_method == 'sobol':
                    self.n = int(2 ** np.ceil(np.log2(self.n)))
                    # self.n #TODO: Should always be self.n, this is
                    # unacceptable for shgo, check that nfev behaves as
                    # expected.
                    self.nc = 0
                    self.sampling_method = 'sobol'
                    self.qmc_engine = qmc.Sobol(d=self.dim, scramble=False,
                                                seed=0)
                else:
                    self.sampling_method = 'halton'
                    self.qmc_engine = qmc.Halton(d=self.dim, scramble=True,
                                                 seed=0)

                def sampling_method(n, d):
                    return self.qmc_engine.random(n)

            else:
                # A user defined sampling method:
                self.sampling_method = 'custom'

            self.sampling = self.sampling_custom
            self.sampling_function = sampling_method  # F(n, d)

        # Local controls
        self.stop_l_iter = False  # Local minimisation iterations
        self.stop_complex_iter = False  # Sampling iterations

        # Initiate storage objects used in algorithm classes
        self.minimizer_pool = []

        # Cache of local minimizers mapped
        self.LMC = LMapCache()

        # Initialize return object
        self.res = OptimizeResult()  # scipy.optimize.OptimizeResult object
        self.res.nfev = 0  # Includes each sampling point as func evaluation
        self.res.nlfev = 0  # Local function evals for all minimisers
        self.res.nljev = 0  # Local Jacobian evals for all minimisers
        self.res.nlhev = 0  # Local Hessian evals for all minimisers

    # Initiation aids
    def init_options(self, options):
        """
        Initiates the options.

        Can also be useful to change parameters after class initiation.

        Parameters
        ----------
        options : dict

        Returns
        -------
        None

        """
        # Update 'options' dict passed to optimize.minimize
        # Do this first so we don't mutate `options` below.
        self.minimizer_kwargs['options'].update(options)

        # Ensure that 'jac', 'hess', and 'hessp' are passed directly to
        # `minimize` as keywords, not as part of its 'options' dictionary.
        for opt in ['jac', 'hess', 'hessp']:
            if opt in self.minimizer_kwargs['options']:
                self.minimizer_kwargs[opt] = (
                    self.minimizer_kwargs['options'].pop(opt))

        # Default settings:
        self.minimize_every_iter = options.get('minimize_every_iter', True)

        # Algorithm limits
        # Maximum number of iterations to perform.
        self.maxiter = options.get('maxiter', None)
        # Maximum number of function evaluations in the feasible domain
        self.maxfev = options.get('maxfev', None)
        # Maximum number of sampling evaluations (includes searching in
        # infeasible points
        self.maxev = options.get('maxev', None)
        # Maximum processing runtime allowed
        self.init = time.time()
        self.maxtime = options.get('maxtime', None)
        if 'f_min' in options:
            # Specify the minimum objective function value, if it is known.
            self.f_min_true = options['f_min']
            self.f_tol = options.get('f_tol', 1e-4)
        else:
            self.f_min_true = None

        self.minhgrd = options.get('minhgrd', None)

        # Objective function knowledge
        self.symmetry = options.get('symmetry', False)
        if self.symmetry:
            self.symmetry = [0, ]*len(self.bounds)
        else:
            self.symmetry = None
        # Algorithm functionality
        # Only evaluate a few of the best candiates
        self.local_iter = options.get('local_iter', False)
        self.infty_cons_sampl = options.get('infty_constraints', True)

        # Feedback
        self.disp = options.get('disp', False)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self.HC.V._mapwrapper.__exit__(*args)

    # Iteration properties
    # Main construction loop:
    def iterate_all(self):
        """
        Construct for `iters` iterations.

        If uniform sampling is used, every iteration adds 'n' sampling points.

        Iterations if a stopping criteria (e.g., sampling points or
        processing time) has been met.

        """
        if self.disp:
            logging.info('Splitting first generation')

        while not self.stop_global:
            if self.break_routine:
                break
            # Iterate complex, process minimisers
            self.iterate()
            self.stopping_criteria()

        # Build minimiser pool
        # Final iteration only needed if pools weren't minimised every
        # iteration
        if not self.minimize_every_iter:
            if not self.break_routine:
                self.find_minima()

        self.res.nit = self.iters_done  # + 1
        self.fn = self.HC.V.nfev

    def find_minima(self):
        """
        Construct the minimizer pool, map the minimizers to local minima
        and sort the results into a global return object.
        """
        if self.disp:
            logging.info('Searching for minimizer pool...')

        self.minimizers()

        if len(self.X_min) != 0:
            # Minimize the pool of minimizers with local minimization methods
            # Note that if Options['local_iter'] is an `int` instead of default
            # value False then only that number of candidates will be minimized
            self.minimise_pool(self.local_iter)
            # Sort results and build the global return object
            self.sort_result()

            # Lowest values used to report in case of failures
            self.f_lowest = self.res.fun
            self.x_lowest = self.res.x
        else:
            self.find_lowest_vertex()

        if self.disp:
            logging.info(f"Minimiser pool = SHGO.X_min = {self.X_min}")

    def find_lowest_vertex(self):
        # Find the lowest objective function value on one of
        # the vertices of the simplicial complex
        self.f_lowest = np.inf
        for x in self.HC.V.cache:
            if self.HC.V[x].f < self.f_lowest:
                if self.disp:
                    logging.info(f'self.HC.V[x].f = {self.HC.V[x].f}')
                self.f_lowest = self.HC.V[x].f
                self.x_lowest = self.HC.V[x].x_a
        for lmc in self.LMC.cache:
            if self.LMC[lmc].f_min < self.f_lowest:
                self.f_lowest = self.LMC[lmc].f_min
                self.x_lowest = self.LMC[lmc].x_l

        if self.f_lowest == np.inf:  # no feasible point
            self.f_lowest = None
            self.x_lowest = None

    # Stopping criteria functions:
    def finite_iterations(self):
        mi = min(x for x in [self.iters, self.maxiter] if x is not None)
        if self.disp:
            logging.info(f'Iterations done = {self.iters_done} / {mi}')
        if self.iters is not None:
            if self.iters_done >= (self.iters):
                self.stop_global = True

        if self.maxiter is not None:  # Stop for infeasible sampling
            if self.iters_done >= (self.maxiter):
                self.stop_global = True
        return self.stop_global

    def finite_fev(self):
        # Finite function evals in the feasible domain
        if self.disp:
            logging.info(f'Function evaluations done = {self.fn} / {self.maxfev}')
        if self.fn >= self.maxfev:
            self.stop_global = True
        return self.stop_global

    def finite_ev(self):
        # Finite evaluations including infeasible sampling points
        if self.disp:
            logging.info(f'Sampling evaluations done = {self.n_sampled} '
                         f'/ {self.maxev}')
        if self.n_sampled >= self.maxev:
            self.stop_global = True

    def finite_time(self):
        if self.disp:
            logging.info(f'Time elapsed = {time.time() - self.init} '
                         f'/ {self.maxtime}')
        if (time.time() - self.init) >= self.maxtime:
            self.stop_global = True

    def finite_precision(self):
        """
        Stop the algorithm if the final function value is known

        Specify in options (with ``self.f_min_true = options['f_min']``)
        and the tolerance with ``f_tol = options['f_tol']``
        """
        # If no minimizer has been found use the lowest sampling value
        self.find_lowest_vertex()
        if self.disp:
            logging.info(f'Lowest function evaluation = {self.f_lowest}')
            logging.info(f'Specified minimum = {self.f_min_true}')
        # If no feasible point was return from test
        if self.f_lowest is None:
            return self.stop_global

        # Function to stop algorithm at specified percentage error:
        if self.f_min_true == 0.0:
            if self.f_lowest <= self.f_tol:
                self.stop_global = True
        else:
            pe = (self.f_lowest - self.f_min_true) / abs(self.f_min_true)
            if self.f_lowest <= self.f_min_true:
                self.stop_global = True
                # 2if (pe - self.f_tol) <= abs(1.0 / abs(self.f_min_true)):
                if abs(pe) >= 2 * self.f_tol:
                    warnings.warn("A much lower value than expected f* =" +
                                  f" {self.f_min_true} than" +
                                  " the was found f_lowest =" +
                                  f"{self.f_lowest} ")
            if pe <= self.f_tol:
                self.stop_global = True

        return self.stop_global

    def finite_homology_growth(self):
        """
        Stop the algorithm if homology group rank did not grow in iteration.
        """
        if self.LMC.size == 0:
            return  # pass on no reason to stop yet.
        self.hgrd = self.LMC.size - self.hgr

        self.hgr = self.LMC.size
        if self.hgrd <= self.minhgrd:
            self.stop_global = True
        if self.disp:
            logging.info(f'Current homology growth = {self.hgrd} '
                         f' (minimum growth = {self.minhgrd})')
        return self.stop_global

    def stopping_criteria(self):
        """
        Various stopping criteria ran every iteration

        Returns
        -------
        stop : bool
        """
        if self.maxiter is not None:
            self.finite_iterations()
        if self.iters is not None:
            self.finite_iterations()
        if self.maxfev is not None:
            self.finite_fev()
        if self.maxev is not None:
            self.finite_ev()
        if self.maxtime is not None:
            self.finite_time()
        if self.f_min_true is not None:
            self.finite_precision()
        if self.minhgrd is not None:
            self.finite_homology_growth()
        return self.stop_global

    def iterate(self):
        self.iterate_complex()

        # Build minimizer pool
        if self.minimize_every_iter:
            if not self.break_routine:
                self.find_minima()  # Process minimizer pool

        # Algorithm updates
        self.iters_done += 1

    def iterate_hypercube(self):
        """
        Iterate a subdivision of the complex

        Note: called with ``self.iterate_complex()`` after class initiation
        """
        # Iterate the complex
        if self.disp:
            logging.info('Constructing and refining simplicial complex graph '
                         'structure')
        if self.n is None:
            self.HC.refine_all()
            self.n_sampled = self.HC.V.size()  # nevs counted
        else:
            self.HC.refine(self.n)
            self.n_sampled += self.n

        if self.disp:
            logging.info('Triangulation completed, evaluating all contraints '
                         'and objective function values.')

        # Readd minimisers to complex
        if len(self.LMC.xl_maps) > 0:
            for xl in self.LMC.cache:
                v = self.HC.V[xl]
                v_near = v.star()
                for v in v.nn:
                    v_near = v_near.union(v.nn)
                # Reconnect vertices to complex
                # if self.HC.connect_vertex_non_symm(tuple(self.LMC[xl].x_l),
                #                                   near=v_near):
                #    continue
                # else:
                    # If failure to find in v_near, then search all vertices
                    # (very expensive operation:
                #    self.HC.connect_vertex_non_symm(tuple(self.LMC[xl].x_l)
                #                                    )

        # Evaluate all constraints and functions
        self.HC.V.process_pools()
        if self.disp:
            logging.info('Evaluations completed.')

        # feasible sampling points counted by the triangulation.py routines
        self.fn = self.HC.V.nfev
        return

    def iterate_delaunay(self):
        """
        Build a complex of Delaunay triangulated points

        Note: called with ``self.iterate_complex()`` after class initiation
        """
        self.nc += self.n
        self.sampled_surface(infty_cons_sampl=self.infty_cons_sampl)

        # Add sampled points to a triangulation, construct self.Tri
        if self.disp:
            logging.info(f'self.n = {self.n}')
            logging.info(f'self.nc = {self.nc}')
            logging.info('Constructing and refining simplicial complex graph '
                         'structure from sampling points.')

        if self.dim < 2:
            self.Ind_sorted = np.argsort(self.C, axis=0)
            self.Ind_sorted = self.Ind_sorted.flatten()
            tris = []
            for ind, ind_s in enumerate(self.Ind_sorted):
                if ind > 0:
                    tris.append(self.Ind_sorted[ind - 1:ind + 1])

            tris = np.array(tris)
            # Store 1D triangulation:
            self.Tri = namedtuple('Tri', ['points', 'simplices'])(self.C, tris)
            self.points = {}
        else:
            if self.C.shape[0] > self.dim + 1:  # Ensure a simplex can be built
                self.delaunay_triangulation(n_prc=self.n_prc)
            self.n_prc = self.C.shape[0]

        if self.disp:
            logging.info('Triangulation completed, evaluating all '
                         'constraints and objective function values.')

        if hasattr(self, 'Tri'):
            self.HC.vf_to_vv(self.Tri.points, self.Tri.simplices)

        # Process all pools
        # Evaluate all constraints and functions
        if self.disp:
            logging.info('Triangulation completed, evaluating all contraints '
                         'and objective function values.')

        # Evaluate all constraints and functions
        self.HC.V.process_pools()
        if self.disp:
            logging.info('Evaluations completed.')

        # feasible sampling points counted by the triangulation.py routines
        self.fn = self.HC.V.nfev
        self.n_sampled = self.nc  # nevs counted in triangulation
        return

    # Hypercube minimizers
    def minimizers(self):
        """
        Returns the indexes of all minimizers
        """
        self.minimizer_pool = []
        # Note: Can implement parallelization here
        for x in self.HC.V.cache:
            in_LMC = False
            if len(self.LMC.xl_maps) > 0:
                for xlmi in self.LMC.xl_maps:
                    if np.all(np.array(x) == np.array(xlmi)):
                        in_LMC = True
            if in_LMC:
                continue

            if self.HC.V[x].minimiser():
                if self.disp:
                    logging.info('=' * 60)
                    logging.info(f'v.x = {self.HC.V[x].x_a} is minimizer')
                    logging.info(f'v.f = {self.HC.V[x].f} is minimizer')
                    logging.info('=' * 30)

                if self.HC.V[x] not in self.minimizer_pool:
                    self.minimizer_pool.append(self.HC.V[x])

                if self.disp:
                    logging.info('Neighbors:')
                    logging.info('=' * 30)
                    for vn in self.HC.V[x].nn:
                        logging.info(f'x = {vn.x} || f = {vn.f}')

                    logging.info('=' * 60)
        self.minimizer_pool_F = []
        self.X_min = []
        # normalized tuple in the Vertex cache
        self.X_min_cache = {}  # Cache used in hypercube sampling

        for v in self.minimizer_pool:
            self.X_min.append(v.x_a)
            self.minimizer_pool_F.append(v.f)
            self.X_min_cache[tuple(v.x_a)] = v.x

        self.minimizer_pool_F = np.array(self.minimizer_pool_F)
        self.X_min = np.array(self.X_min)

        # TODO: Only do this if global mode
        self.sort_min_pool()

        return self.X_min

    # Local minimisation
    # Minimiser pool processing
    def minimise_pool(self, force_iter=False):
        """
        This processing method can optionally minimise only the best candidate
        solutions in the minimiser pool

        Parameters
        ----------
        force_iter : int
                     Number of starting minimizers to process (can be sepcified
                     globally or locally)

        """
        # Find first local minimum
        # NOTE: Since we always minimize this value regardless it is a waste to
        # build the topograph first before minimizing
        lres_f_min = self.minimize(self.X_min[0], ind=self.minimizer_pool[0])

        # Trim minimized point from current minimizer set
        self.trim_min_pool(0)

        while not self.stop_l_iter:
            # Global stopping criteria:
            self.stopping_criteria()

            # Note first iteration is outside loop:
            if force_iter:
                force_iter -= 1
                if force_iter == 0:
                    self.stop_l_iter = True
                    break

            if np.shape(self.X_min)[0] == 0:
                self.stop_l_iter = True
                break

            # Construct topograph from current minimizer set
            # (NOTE: This is a very small topograph using only the minizer pool
            #        , it might be worth using some graph theory tools instead.
            self.g_topograph(lres_f_min.x, self.X_min)

            # Find local minimum at the miniser with the greatest Euclidean
            # distance from the current solution
            ind_xmin_l = self.Z[:, -1]
            lres_f_min = self.minimize(self.Ss[-1, :], self.minimizer_pool[-1])

            # Trim minimised point from current minimizer set
            self.trim_min_pool(ind_xmin_l)

        # Reset controls
        self.stop_l_iter = False
        return

    def sort_min_pool(self):
        # Sort to find minimum func value in min_pool
        self.ind_f_min = np.argsort(self.minimizer_pool_F)
        self.minimizer_pool = np.array(self.minimizer_pool)[self.ind_f_min]
        self.minimizer_pool_F = np.array(self.minimizer_pool_F)[
            self.ind_f_min]
        return

    def trim_min_pool(self, trim_ind):
        self.X_min = np.delete(self.X_min, trim_ind, axis=0)
        self.minimizer_pool_F = np.delete(self.minimizer_pool_F, trim_ind)
        self.minimizer_pool = np.delete(self.minimizer_pool, trim_ind)
        return

    def g_topograph(self, x_min, X_min):
        """
        Returns the topographical vector stemming from the specified value
        ``x_min`` for the current feasible set ``X_min`` with True boolean
        values indicating positive entries and False values indicating
        negative entries.

        """
        x_min = np.array([x_min])
        self.Y = spatial.distance.cdist(x_min, X_min, 'euclidean')
        # Find sorted indexes of spatial distances:
        self.Z = np.argsort(self.Y, axis=-1)

        self.Ss = X_min[self.Z][0]
        self.minimizer_pool = self.minimizer_pool[self.Z]
        self.minimizer_pool = self.minimizer_pool[0]
        return self.Ss

    # Local bound functions
    def construct_lcb_simplicial(self, v_min):
        """
        Construct locally (approximately) convex bounds

        Parameters
        ----------
        v_min : Vertex object
                The minimizer vertex

        Returns
        -------
        cbounds : list of lists
            List of size dimension with length-2 list of bounds for each
            dimension.

        """
        cbounds = [[x_b_i[0], x_b_i[1]] for x_b_i in self.bounds]
        # Loop over all bounds
        for vn in v_min.nn:
            for i, x_i in enumerate(vn.x_a):
                # Lower bound
                if (x_i < v_min.x_a[i]) and (x_i > cbounds[i][0]):
                    cbounds[i][0] = x_i

                # Upper bound
                if (x_i > v_min.x_a[i]) and (x_i < cbounds[i][1]):
                    cbounds[i][1] = x_i

        if self.disp:
            logging.info(f'cbounds found for v_min.x_a = {v_min.x_a}')
            logging.info(f'cbounds = {cbounds}')

        return cbounds

    def construct_lcb_delaunay(self, v_min, ind=None):
        """
        Construct locally (approximately) convex bounds

        Parameters
        ----------
        v_min : Vertex object
                The minimizer vertex

        Returns
        -------
        cbounds : list of lists
            List of size dimension with length-2 list of bounds for each
            dimension.
        """
        cbounds = [[x_b_i[0], x_b_i[1]] for x_b_i in self.bounds]

        return cbounds

    # Minimize a starting point locally
    def minimize(self, x_min, ind=None):
        """
        This function is used to calculate the local minima using the specified
        sampling point as a starting value.

        Parameters
        ----------
        x_min : vector of floats
            Current starting point to minimize.

        Returns
        -------
        lres : OptimizeResult
            The local optimization result represented as a `OptimizeResult`
            object.
        """
        # Use minima maps if vertex was already run
        if self.disp:
            logging.info(f'Vertex minimiser maps = {self.LMC.v_maps}')

        if self.LMC[x_min].lres is not None:
            logging.info(f'Found self.LMC[x_min].lres = '
                         f'{self.LMC[x_min].lres}')
            return self.LMC[x_min].lres

        if self.callback is not None:
            logging.info('Callback for '
                  'minimizer starting at {}:'.format(x_min))

        if self.disp:
            logging.info('Starting '
                  'minimization at {}...'.format(x_min))

        if self.sampling_method == 'simplicial':
            x_min_t = tuple(x_min)
            # Find the normalized tuple in the Vertex cache:
            x_min_t_norm = self.X_min_cache[tuple(x_min_t)]
            x_min_t_norm = tuple(x_min_t_norm)
            g_bounds = self.construct_lcb_simplicial(self.HC.V[x_min_t_norm])
            if 'bounds' in self.min_solver_args:
                self.minimizer_kwargs['bounds'] = g_bounds
                logging.info(self.minimizer_kwargs['bounds'])

        else:
            g_bounds = self.construct_lcb_delaunay(x_min, ind=ind)
            if 'bounds' in self.min_solver_args:
                self.minimizer_kwargs['bounds'] = g_bounds
                logging.info(self.minimizer_kwargs['bounds'])

        if self.disp and 'bounds' in self.minimizer_kwargs:
            logging.info('bounds in kwarg:')
            logging.info(self.minimizer_kwargs['bounds'])

        # Local minimization using scipy.optimize.minimize:
        lres = minimize(self.func, x_min, **self.minimizer_kwargs)

        if self.disp:
            logging.info(f'lres = {lres}')

        # Local function evals for all minimizers
        self.res.nlfev += lres.nfev
        if 'njev' in lres:
            self.res.nljev += lres.njev
        if 'nhev' in lres:
            self.res.nlhev += lres.nhev

        try:  # Needed because of the brain dead 1x1 NumPy arrays
            lres.fun = lres.fun[0]
        except (IndexError, TypeError):
            lres.fun

        # Append minima maps
        self.LMC[x_min]
        self.LMC.add_res(x_min, lres, bounds=g_bounds)

        return lres

    # Post local minimization processing
    def sort_result(self):
        """
        Sort results and build the global return object
        """
        # Sort results in local minima cache
        results = self.LMC.sort_cache_result()
        self.res.xl = results['xl']
        self.res.funl = results['funl']
        self.res.x = results['x']
        self.res.fun = results['fun']

        # Add local func evals to sampling func evals
        # Count the number of feasible vertices and add to local func evals:
        self.res.nfev = self.fn + self.res.nlfev
        return self.res

    # Algorithm controls
    def fail_routine(self, mes=("Failed to converge")):
        self.break_routine = True
        self.res.success = False
        self.X_min = [None]
        self.res.message = mes

    def sampled_surface(self, infty_cons_sampl=False):
        """
        Sample the function surface.

        There are 2 modes, if ``infty_cons_sampl`` is True then the sampled
        points that are generated outside the feasible domain will be
        assigned an ``inf`` value in accordance with SHGO rules.
        This guarantees convergence and usually requires less objective
        function evaluations at the computational costs of more Delaunay
        triangulation points.

        If ``infty_cons_sampl`` is False, then the infeasible points are
        discarded and only a subspace of the sampled points are used. This
        comes at the cost of the loss of guaranteed convergence and usually
        requires more objective function evaluations.
        """
        # Generate sampling points
        if self.disp:
            logging.info('Generating sampling points')
        self.sampling(self.nc, self.dim)
        if len(self.LMC.xl_maps) > 0:
            self.C = np.vstack((self.C, np.array(self.LMC.xl_maps)))
        if not infty_cons_sampl:
            # Find subspace of feasible points
            if self.g_cons is not None:
                self.sampling_subspace()

        # Sort remaining samples
        self.sorted_samples()

        # Find objective function references
        self.n_sampled = self.nc

    def sampling_custom(self, n, dim):
        """
        Generates uniform sampling points in a hypercube and scales the points
        to the bound limits.
        """
        # Generate sampling points.
        # Generate uniform sample points in [0, 1]^m \subset R^m
        if self.n_sampled == 0:
            self.C = self.sampling_function(n, dim)
        else:
            self.C = self.sampling_function(n, dim)
        # Distribute over bounds
        for i in range(len(self.bounds)):
            self.C[:, i] = (self.C[:, i] *
                            (self.bounds[i][1] - self.bounds[i][0])
                            + self.bounds[i][0])
        return self.C

    def sampling_subspace(self):
        """Find subspace of feasible points from g_func definition"""
        # Subspace of feasible points.
        for ind, g in enumerate(self.g_cons):
            # C.shape = (Z, dim) where Z is the number of sampling points to
            # evaluate and dim is the dimensionality of the problem.
            # the constraint function may not be vectorised so have to step
            # through each sampling point sequentially.
            feasible = np.array(
                [np.all(g(x_C, *self.g_args[ind]) >= 0.0) for x_C in self.C],
                dtype=bool
            )
            self.C = self.C[feasible]

            if self.C.size == 0:
                self.res.message = ('No sampling point found within the '
                                    + 'feasible set. Increasing sampling '
                                    + 'size.')
                # sampling correctly for both 1-D and >1-D cases
                if self.disp:
                    logging.info(self.res.message)

    def sorted_samples(self):  # Validated
        """Find indexes of the sorted sampling points"""
        self.Ind_sorted = np.argsort(self.C, axis=0)
        self.Xs = self.C[self.Ind_sorted]
        return self.Ind_sorted, self.Xs

    def delaunay_triangulation(self, n_prc=0):
        if hasattr(self, 'Tri') and self.qhull_incremental:
            # TODO: Uncertain if n_prc needs to add len(self.LMC.xl_maps)
            # in self.sampled_surface
            self.Tri.add_points(self.C[n_prc:, :])
        else:
            try:
                self.Tri = spatial.Delaunay(self.C,
                                            incremental=self.qhull_incremental,
                                            )
            except spatial.QhullError:
                if str(sys.exc_info()[1])[:6] == 'QH6239':
                    logging.warning('QH6239 Qhull precision error detected, '
                                    'this usually occurs when no bounds are '
                                    'specified, Qhull can only run with '
                                    'handling cocircular/cospherical points'
                                    ' and in this case incremental mode is '
                                    'switched off. The performance of shgo '
                                    'will be reduced in this mode.')
                    self.qhull_incremental = False
                    self.Tri = spatial.Delaunay(self.C,
                                                incremental=
                                                self.qhull_incremental)
                else:
                    raise

        return self.Tri


class LMap:
    def __init__(self, v):
        self.v = v
        self.x_l = None
        self.lres = None
        self.f_min = None
        self.lbounds = []


class LMapCache:
    def __init__(self):
        self.cache = {}

        # Lists for search queries
        self.v_maps = []
        self.xl_maps = []
        self.xl_maps_set = set()
        self.f_maps = []
        self.lbound_maps = []
        self.size = 0

    def __getitem__(self, v):
        try:
            v = np.ndarray.tolist(v)
        except TypeError:
            pass
        v = tuple(v)
        try:
            return self.cache[v]
        except KeyError:
            xval = LMap(v)
            self.cache[v] = xval

            return self.cache[v]

    def add_res(self, v, lres, bounds=None):
        v = np.ndarray.tolist(v)
        v = tuple(v)
        self.cache[v].x_l = lres.x
        self.cache[v].lres = lres
        self.cache[v].f_min = lres.fun
        self.cache[v].lbounds = bounds

        # Update cache size
        self.size += 1

        # Cache lists for search queries
        self.v_maps.append(v)
        self.xl_maps.append(lres.x)
        self.xl_maps_set.add(tuple(lres.x))
        self.f_maps.append(lres.fun)
        self.lbound_maps.append(bounds)

    def sort_cache_result(self):
        """
        Sort results and build the global return object
        """
        results = {}
        # Sort results and save
        self.xl_maps = np.array(self.xl_maps)
        self.f_maps = np.array(self.f_maps)

        # Sorted indexes in Func_min
        ind_sorted = np.argsort(self.f_maps)

        # Save ordered list of minima
        results['xl'] = self.xl_maps[ind_sorted]  # Ordered x vals
        self.f_maps = np.array(self.f_maps)
        results['funl'] = self.f_maps[ind_sorted]
        results['funl'] = results['funl'].T

        # Find global of all minimizers
        results['x'] = self.xl_maps[ind_sorted[0]]  # Save global minima
        results['fun'] = self.f_maps[ind_sorted[0]]  # Save global fun value

        self.xl_maps = np.ndarray.tolist(self.xl_maps)
        self.f_maps = np.ndarray.tolist(self.f_maps)
        return results
