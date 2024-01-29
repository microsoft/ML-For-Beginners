import warnings
import numpy as np
from scipy.sparse import csc_array, vstack, issparse
from scipy._lib._util import VisibleDeprecationWarning
from ._highs._highs_wrapper import _highs_wrapper  # type: ignore[import]
from ._constraints import LinearConstraint, Bounds
from ._optimize import OptimizeResult
from ._linprog_highs import _highs_to_scipy_status_message


def _constraints_to_components(constraints):
    """
    Convert sequence of constraints to a single set of components A, b_l, b_u.

    `constraints` could be

    1. A LinearConstraint
    2. A tuple representing a LinearConstraint
    3. An invalid object
    4. A sequence of composed entirely of objects of type 1/2
    5. A sequence containing at least one object of type 3

    We want to accept 1, 2, and 4 and reject 3 and 5.
    """
    message = ("`constraints` (or each element within `constraints`) must be "
               "convertible into an instance of "
               "`scipy.optimize.LinearConstraint`.")
    As = []
    b_ls = []
    b_us = []

    # Accept case 1 by standardizing as case 4
    if isinstance(constraints, LinearConstraint):
        constraints = [constraints]
    else:
        # Reject case 3
        try:
            iter(constraints)
        except TypeError as exc:
            raise ValueError(message) from exc

        # Accept case 2 by standardizing as case 4
        if len(constraints) == 3:
            # argument could be a single tuple representing a LinearConstraint
            try:
                constraints = [LinearConstraint(*constraints)]
            except (TypeError, ValueError, VisibleDeprecationWarning):
                # argument was not a tuple representing a LinearConstraint
                pass

    # Address cases 4/5
    for constraint in constraints:
        # if it's not a LinearConstraint or something that represents a
        # LinearConstraint at this point, it's invalid
        if not isinstance(constraint, LinearConstraint):
            try:
                constraint = LinearConstraint(*constraint)
            except TypeError as exc:
                raise ValueError(message) from exc
        As.append(csc_array(constraint.A))
        b_ls.append(np.atleast_1d(constraint.lb).astype(np.float64))
        b_us.append(np.atleast_1d(constraint.ub).astype(np.float64))

    if len(As) > 1:
        A = vstack(As, format="csc")
        b_l = np.concatenate(b_ls)
        b_u = np.concatenate(b_us)
    else:  # avoid unnecessary copying
        A = As[0]
        b_l = b_ls[0]
        b_u = b_us[0]

    return A, b_l, b_u


def _milp_iv(c, integrality, bounds, constraints, options):
    # objective IV
    if issparse(c):
        raise ValueError("`c` must be a dense array.")
    c = np.atleast_1d(c).astype(np.float64)
    if c.ndim != 1 or c.size == 0 or not np.all(np.isfinite(c)):
        message = ("`c` must be a one-dimensional array of finite numbers "
                   "with at least one element.")
        raise ValueError(message)

    # integrality IV
    if issparse(integrality):
        raise ValueError("`integrality` must be a dense array.")
    message = ("`integrality` must contain integers 0-3 and be broadcastable "
               "to `c.shape`.")
    if integrality is None:
        integrality = 0
    try:
        integrality = np.broadcast_to(integrality, c.shape).astype(np.uint8)
    except ValueError:
        raise ValueError(message)
    if integrality.min() < 0 or integrality.max() > 3:
        raise ValueError(message)

    # bounds IV
    if bounds is None:
        bounds = Bounds(0, np.inf)
    elif not isinstance(bounds, Bounds):
        message = ("`bounds` must be convertible into an instance of "
                   "`scipy.optimize.Bounds`.")
        try:
            bounds = Bounds(*bounds)
        except TypeError as exc:
            raise ValueError(message) from exc

    try:
        lb = np.broadcast_to(bounds.lb, c.shape).astype(np.float64)
        ub = np.broadcast_to(bounds.ub, c.shape).astype(np.float64)
    except (ValueError, TypeError) as exc:
        message = ("`bounds.lb` and `bounds.ub` must contain reals and "
                   "be broadcastable to `c.shape`.")
        raise ValueError(message) from exc

    # constraints IV
    if not constraints:
        constraints = [LinearConstraint(np.empty((0, c.size)),
                                        np.empty((0,)), np.empty((0,)))]
    try:
        A, b_l, b_u = _constraints_to_components(constraints)
    except ValueError as exc:
        message = ("`constraints` (or each element within `constraints`) must "
                   "be convertible into an instance of "
                   "`scipy.optimize.LinearConstraint`.")
        raise ValueError(message) from exc

    if A.shape != (b_l.size, c.size):
        message = "The shape of `A` must be (len(b_l), len(c))."
        raise ValueError(message)
    indptr, indices, data = A.indptr, A.indices, A.data.astype(np.float64)

    # options IV
    options = options or {}
    supported_options = {'disp', 'presolve', 'time_limit', 'node_limit',
                         'mip_rel_gap'}
    unsupported_options = set(options).difference(supported_options)
    if unsupported_options:
        message = (f"Unrecognized options detected: {unsupported_options}. "
                   "These will be passed to HiGHS verbatim.")
        warnings.warn(message, RuntimeWarning, stacklevel=3)
    options_iv = {'log_to_console': options.pop("disp", False),
                  'mip_max_nodes': options.pop("node_limit", None)}
    options_iv.update(options)

    return c, integrality, lb, ub, indptr, indices, data, b_l, b_u, options_iv


def milp(c, *, integrality=None, bounds=None, constraints=None, options=None):
    r"""
    Mixed-integer linear programming

    Solves problems of the following form:

    .. math::

        \min_x \ & c^T x \\
        \mbox{such that} \ & b_l \leq A x \leq b_u,\\
        & l \leq x \leq u, \\
        & x_i \in \mathbb{Z}, i \in X_i

    where :math:`x` is a vector of decision variables;
    :math:`c`, :math:`b_l`, :math:`b_u`, :math:`l`, and :math:`u` are vectors;
    :math:`A` is a matrix, and :math:`X_i` is the set of indices of
    decision variables that must be integral. (In this context, a
    variable that can assume only integer values is said to be "integral";
    it has an "integrality" constraint.)

    Alternatively, that's:

    minimize::

        c @ x

    such that::

        b_l <= A @ x <= b_u
        l <= x <= u
        Specified elements of x must be integers

    By default, ``l = 0`` and ``u = np.inf`` unless specified with
    ``bounds``.

    Parameters
    ----------
    c : 1D dense array_like
        The coefficients of the linear objective function to be minimized.
        `c` is converted to a double precision array before the problem is
        solved.
    integrality : 1D dense array_like, optional
        Indicates the type of integrality constraint on each decision variable.

        ``0`` : Continuous variable; no integrality constraint.

        ``1`` : Integer variable; decision variable must be an integer
        within `bounds`.

        ``2`` : Semi-continuous variable; decision variable must be within
        `bounds` or take value ``0``.

        ``3`` : Semi-integer variable; decision variable must be an integer
        within `bounds` or take value ``0``.

        By default, all variables are continuous. `integrality` is converted
        to an array of integers before the problem is solved.

    bounds : scipy.optimize.Bounds, optional
        Bounds on the decision variables. Lower and upper bounds are converted
        to double precision arrays before the problem is solved. The
        ``keep_feasible`` parameter of the `Bounds` object is ignored. If
        not specified, all decision variables are constrained to be
        non-negative.
    constraints : sequence of scipy.optimize.LinearConstraint, optional
        Linear constraints of the optimization problem. Arguments may be
        one of the following:

        1. A single `LinearConstraint` object
        2. A single tuple that can be converted to a `LinearConstraint` object
           as ``LinearConstraint(*constraints)``
        3. A sequence composed entirely of objects of type 1. and 2.

        Before the problem is solved, all values are converted to double
        precision, and the matrices of constraint coefficients are converted to
        instances of `scipy.sparse.csc_array`. The ``keep_feasible`` parameter
        of `LinearConstraint` objects is ignored.
    options : dict, optional
        A dictionary of solver options. The following keys are recognized.

        disp : bool (default: ``False``)
            Set to ``True`` if indicators of optimization status are to be
            printed to the console during optimization.
        node_limit : int, optional
            The maximum number of nodes (linear program relaxations) to solve
            before stopping. Default is no maximum number of nodes.
        presolve : bool (default: ``True``)
            Presolve attempts to identify trivial infeasibilities,
            identify trivial unboundedness, and simplify the problem before
            sending it to the main solver.
        time_limit : float, optional
            The maximum number of seconds allotted to solve the problem.
            Default is no time limit.
        mip_rel_gap : float, optional
            Termination criterion for MIP solver: solver will terminate when
            the gap between the primal objective value and the dual objective
            bound, scaled by the primal objective value, is <= mip_rel_gap.

    Returns
    -------
    res : OptimizeResult
        An instance of :class:`scipy.optimize.OptimizeResult`. The object
        is guaranteed to have the following attributes.

        status : int
            An integer representing the exit status of the algorithm.

            ``0`` : Optimal solution found.

            ``1`` : Iteration or time limit reached.

            ``2`` : Problem is infeasible.

            ``3`` : Problem is unbounded.

            ``4`` : Other; see message for details.

        success : bool
            ``True`` when an optimal solution is found and ``False`` otherwise.

        message : str
            A string descriptor of the exit status of the algorithm.

        The following attributes will also be present, but the values may be
        ``None``, depending on the solution status.

        x : ndarray
            The values of the decision variables that minimize the
            objective function while satisfying the constraints.
        fun : float
            The optimal value of the objective function ``c @ x``.
        mip_node_count : int
            The number of subproblems or "nodes" solved by the MILP solver.
        mip_dual_bound : float
            The MILP solver's final estimate of the lower bound on the optimal
            solution.
        mip_gap : float
            The difference between the primal objective value and the dual
            objective bound, scaled by the primal objective value.

    Notes
    -----
    `milp` is a wrapper of the HiGHS linear optimization software [1]_. The
    algorithm is deterministic, and it typically finds the global optimum of
    moderately challenging mixed-integer linear programs (when it exists).

    References
    ----------
    .. [1] Huangfu, Q., Galabova, I., Feldmeier, M., and Hall, J. A. J.
           "HiGHS - high performance software for linear optimization."
           https://highs.dev/
    .. [2] Huangfu, Q. and Hall, J. A. J. "Parallelizing the dual revised
           simplex method." Mathematical Programming Computation, 10 (1),
           119-142, 2018. DOI: 10.1007/s12532-017-0130-5

    Examples
    --------
    Consider the problem at
    https://en.wikipedia.org/wiki/Integer_programming#Example, which is
    expressed as a maximization problem of two variables. Since `milp` requires
    that the problem be expressed as a minimization problem, the objective
    function coefficients on the decision variables are:

    >>> import numpy as np
    >>> c = -np.array([0, 1])

    Note the negative sign: we maximize the original objective function
    by minimizing the negative of the objective function.

    We collect the coefficients of the constraints into arrays like:

    >>> A = np.array([[-1, 1], [3, 2], [2, 3]])
    >>> b_u = np.array([1, 12, 12])
    >>> b_l = np.full_like(b_u, -np.inf)

    Because there is no lower limit on these constraints, we have defined a
    variable ``b_l`` full of values representing negative infinity. This may
    be unfamiliar to users of `scipy.optimize.linprog`, which only accepts
    "less than" (or "upper bound") inequality constraints of the form
    ``A_ub @ x <= b_u``. By accepting both ``b_l`` and ``b_u`` of constraints
    ``b_l <= A_ub @ x <= b_u``, `milp` makes it easy to specify "greater than"
    inequality constraints, "less than" inequality constraints, and equality
    constraints concisely.

    These arrays are collected into a single `LinearConstraint` object like:

    >>> from scipy.optimize import LinearConstraint
    >>> constraints = LinearConstraint(A, b_l, b_u)

    The non-negativity bounds on the decision variables are enforced by
    default, so we do not need to provide an argument for `bounds`.

    Finally, the problem states that both decision variables must be integers:

    >>> integrality = np.ones_like(c)

    We solve the problem like:

    >>> from scipy.optimize import milp
    >>> res = milp(c=c, constraints=constraints, integrality=integrality)
    >>> res.x
    [1.0, 2.0]

    Note that had we solved the relaxed problem (without integrality
    constraints):

    >>> res = milp(c=c, constraints=constraints)  # OR:
    >>> # from scipy.optimize import linprog; res = linprog(c, A, b_u)
    >>> res.x
    [1.8, 2.8]

    we would not have obtained the correct solution by rounding to the nearest
    integers.

    Other examples are given :ref:`in the tutorial <tutorial-optimize_milp>`.

    """
    args_iv = _milp_iv(c, integrality, bounds, constraints, options)
    c, integrality, lb, ub, indptr, indices, data, b_l, b_u, options = args_iv

    highs_res = _highs_wrapper(c, indptr, indices, data, b_l, b_u,
                               lb, ub, integrality, options)

    res = {}

    # Convert to scipy-style status and message
    highs_status = highs_res.get('status', None)
    highs_message = highs_res.get('message', None)
    status, message = _highs_to_scipy_status_message(highs_status,
                                                     highs_message)
    res['status'] = status
    res['message'] = message
    res['success'] = (status == 0)
    x = highs_res.get('x', None)
    res['x'] = np.array(x) if x is not None else None
    res['fun'] = highs_res.get('fun', None)
    res['mip_node_count'] = highs_res.get('mip_node_count', None)
    res['mip_dual_bound'] = highs_res.get('mip_dual_bound', None)
    res['mip_gap'] = highs_res.get('mip_gap', None)

    return OptimizeResult(res)
