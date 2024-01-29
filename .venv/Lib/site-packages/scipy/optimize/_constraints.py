"""Constraints definition for minimize."""
import numpy as np
from ._hessian_update_strategy import BFGS
from ._differentiable_functions import (
    VectorFunction, LinearVectorFunction, IdentityVectorFunction)
from ._optimize import OptimizeWarning
from warnings import warn, catch_warnings, simplefilter, filterwarnings
from scipy.sparse import issparse


def _arr_to_scalar(x):
    # If x is a numpy array, return x.item().  This will
    # fail if the array has more than one element.
    return x.item() if isinstance(x, np.ndarray) else x


class NonlinearConstraint:
    """Nonlinear constraint on the variables.

    The constraint has the general inequality form::

        lb <= fun(x) <= ub

    Here the vector of independent variables x is passed as ndarray of shape
    (n,) and ``fun`` returns a vector with m components.

    It is possible to use equal bounds to represent an equality constraint or
    infinite bounds to represent a one-sided constraint.

    Parameters
    ----------
    fun : callable
        The function defining the constraint.
        The signature is ``fun(x) -> array_like, shape (m,)``.
    lb, ub : array_like
        Lower and upper bounds on the constraint. Each array must have the
        shape (m,) or be a scalar, in the latter case a bound will be the same
        for all components of the constraint. Use ``np.inf`` with an
        appropriate sign to specify a one-sided constraint.
        Set components of `lb` and `ub` equal to represent an equality
        constraint. Note that you can mix constraints of different types:
        interval, one-sided or equality, by setting different components of
        `lb` and `ub` as  necessary.
    jac : {callable,  '2-point', '3-point', 'cs'}, optional
        Method of computing the Jacobian matrix (an m-by-n matrix,
        where element (i, j) is the partial derivative of f[i] with
        respect to x[j]).  The keywords {'2-point', '3-point',
        'cs'} select a finite difference scheme for the numerical estimation.
        A callable must have the following signature:
        ``jac(x) -> {ndarray, sparse matrix}, shape (m, n)``.
        Default is '2-point'.
    hess : {callable, '2-point', '3-point', 'cs', HessianUpdateStrategy, None}, optional
        Method for computing the Hessian matrix. The keywords
        {'2-point', '3-point', 'cs'} select a finite difference scheme for
        numerical  estimation.  Alternatively, objects implementing
        `HessianUpdateStrategy` interface can be used to approximate the
        Hessian. Currently available implementations are:

            - `BFGS` (default option)
            - `SR1`

        A callable must return the Hessian matrix of ``dot(fun, v)`` and
        must have the following signature:
        ``hess(x, v) -> {LinearOperator, sparse matrix, array_like}, shape (n, n)``.
        Here ``v`` is ndarray with shape (m,) containing Lagrange multipliers.
    keep_feasible : array_like of bool, optional
        Whether to keep the constraint components feasible throughout
        iterations. A single value set this property for all components.
        Default is False. Has no effect for equality constraints.
    finite_diff_rel_step: None or array_like, optional
        Relative step size for the finite difference approximation. Default is
        None, which will select a reasonable value automatically depending
        on a finite difference scheme.
    finite_diff_jac_sparsity: {None, array_like, sparse matrix}, optional
        Defines the sparsity structure of the Jacobian matrix for finite
        difference estimation, its shape must be (m, n). If the Jacobian has
        only few non-zero elements in *each* row, providing the sparsity
        structure will greatly speed up the computations. A zero entry means
        that a corresponding element in the Jacobian is identically zero.
        If provided, forces the use of 'lsmr' trust-region solver.
        If None (default) then dense differencing will be used.

    Notes
    -----
    Finite difference schemes {'2-point', '3-point', 'cs'} may be used for
    approximating either the Jacobian or the Hessian. We, however, do not allow
    its use for approximating both simultaneously. Hence whenever the Jacobian
    is estimated via finite-differences, we require the Hessian to be estimated
    using one of the quasi-Newton strategies.

    The scheme 'cs' is potentially the most accurate, but requires the function
    to correctly handles complex inputs and be analytically continuable to the
    complex plane. The scheme '3-point' is more accurate than '2-point' but
    requires twice as many operations.

    Examples
    --------
    Constrain ``x[0] < sin(x[1]) + 1.9``

    >>> from scipy.optimize import NonlinearConstraint
    >>> import numpy as np
    >>> con = lambda x: x[0] - np.sin(x[1])
    >>> nlc = NonlinearConstraint(con, -np.inf, 1.9)

    """
    def __init__(self, fun, lb, ub, jac='2-point', hess=BFGS(),
                 keep_feasible=False, finite_diff_rel_step=None,
                 finite_diff_jac_sparsity=None):
        self.fun = fun
        self.lb = lb
        self.ub = ub
        self.finite_diff_rel_step = finite_diff_rel_step
        self.finite_diff_jac_sparsity = finite_diff_jac_sparsity
        self.jac = jac
        self.hess = hess
        self.keep_feasible = keep_feasible


class LinearConstraint:
    """Linear constraint on the variables.

    The constraint has the general inequality form::

        lb <= A.dot(x) <= ub

    Here the vector of independent variables x is passed as ndarray of shape
    (n,) and the matrix A has shape (m, n).

    It is possible to use equal bounds to represent an equality constraint or
    infinite bounds to represent a one-sided constraint.

    Parameters
    ----------
    A : {array_like, sparse matrix}, shape (m, n)
        Matrix defining the constraint.
    lb, ub : dense array_like, optional
        Lower and upper limits on the constraint. Each array must have the
        shape (m,) or be a scalar, in the latter case a bound will be the same
        for all components of the constraint. Use ``np.inf`` with an
        appropriate sign to specify a one-sided constraint.
        Set components of `lb` and `ub` equal to represent an equality
        constraint. Note that you can mix constraints of different types:
        interval, one-sided or equality, by setting different components of
        `lb` and `ub` as  necessary. Defaults to ``lb = -np.inf``
        and ``ub = np.inf`` (no limits).
    keep_feasible : dense array_like of bool, optional
        Whether to keep the constraint components feasible throughout
        iterations. A single value set this property for all components.
        Default is False. Has no effect for equality constraints.
    """
    def _input_validation(self):
        if self.A.ndim != 2:
            message = "`A` must have exactly two dimensions."
            raise ValueError(message)

        try:
            shape = self.A.shape[0:1]
            self.lb = np.broadcast_to(self.lb, shape)
            self.ub = np.broadcast_to(self.ub, shape)
            self.keep_feasible = np.broadcast_to(self.keep_feasible, shape)
        except ValueError:
            message = ("`lb`, `ub`, and `keep_feasible` must be broadcastable "
                       "to shape `A.shape[0:1]`")
            raise ValueError(message)

    def __init__(self, A, lb=-np.inf, ub=np.inf, keep_feasible=False):
        if not issparse(A):
            # In some cases, if the constraint is not valid, this emits a
            # VisibleDeprecationWarning about ragged nested sequences
            # before eventually causing an error. `scipy.optimize.milp` would
            # prefer that this just error out immediately so it can handle it
            # rather than concerning the user.
            with catch_warnings():
                simplefilter("error")
                self.A = np.atleast_2d(A).astype(np.float64)
        else:
            self.A = A
        if issparse(lb) or issparse(ub):
            raise ValueError("Constraint limits must be dense arrays.")
        self.lb = np.atleast_1d(lb).astype(np.float64)
        self.ub = np.atleast_1d(ub).astype(np.float64)

        if issparse(keep_feasible):
            raise ValueError("`keep_feasible` must be a dense array.")
        self.keep_feasible = np.atleast_1d(keep_feasible).astype(bool)
        self._input_validation()

    def residual(self, x):
        """
        Calculate the residual between the constraint function and the limits

        For a linear constraint of the form::

            lb <= A@x <= ub

        the lower and upper residuals between ``A@x`` and the limits are values
        ``sl`` and ``sb`` such that::

            lb + sl == A@x == ub - sb

        When all elements of ``sl`` and ``sb`` are positive, all elements of
        the constraint are satisfied; a negative element in ``sl`` or ``sb``
        indicates that the corresponding element of the constraint is not
        satisfied.

        Parameters
        ----------
        x: array_like
            Vector of independent variables

        Returns
        -------
        sl, sb : array-like
            The lower and upper residuals
        """
        return self.A@x - self.lb, self.ub - self.A@x


class Bounds:
    """Bounds constraint on the variables.

    The constraint has the general inequality form::

        lb <= x <= ub

    It is possible to use equal bounds to represent an equality constraint or
    infinite bounds to represent a one-sided constraint.

    Parameters
    ----------
    lb, ub : dense array_like, optional
        Lower and upper bounds on independent variables. `lb`, `ub`, and
        `keep_feasible` must be the same shape or broadcastable.
        Set components of `lb` and `ub` equal
        to fix a variable. Use ``np.inf`` with an appropriate sign to disable
        bounds on all or some variables. Note that you can mix constraints of
        different types: interval, one-sided or equality, by setting different
        components of `lb` and `ub` as necessary. Defaults to ``lb = -np.inf``
        and ``ub = np.inf`` (no bounds).
    keep_feasible : dense array_like of bool, optional
        Whether to keep the constraint components feasible throughout
        iterations. Must be broadcastable with `lb` and `ub`.
        Default is False. Has no effect for equality constraints.
    """
    def _input_validation(self):
        try:
            res = np.broadcast_arrays(self.lb, self.ub, self.keep_feasible)
            self.lb, self.ub, self.keep_feasible = res
        except ValueError:
            message = "`lb`, `ub`, and `keep_feasible` must be broadcastable."
            raise ValueError(message)

    def __init__(self, lb=-np.inf, ub=np.inf, keep_feasible=False):
        if issparse(lb) or issparse(ub):
            raise ValueError("Lower and upper bounds must be dense arrays.")
        self.lb = np.atleast_1d(lb)
        self.ub = np.atleast_1d(ub)

        if issparse(keep_feasible):
            raise ValueError("`keep_feasible` must be a dense array.")
        self.keep_feasible = np.atleast_1d(keep_feasible).astype(bool)
        self._input_validation()

    def __repr__(self):
        start = f"{type(self).__name__}({self.lb!r}, {self.ub!r}"
        if np.any(self.keep_feasible):
            end = f", keep_feasible={self.keep_feasible!r})"
        else:
            end = ")"
        return start + end

    def residual(self, x):
        """Calculate the residual (slack) between the input and the bounds

        For a bound constraint of the form::

            lb <= x <= ub

        the lower and upper residuals between `x` and the bounds are values
        ``sl`` and ``sb`` such that::

            lb + sl == x == ub - sb

        When all elements of ``sl`` and ``sb`` are positive, all elements of
        ``x`` lie within the bounds; a negative element in ``sl`` or ``sb``
        indicates that the corresponding element of ``x`` is out of bounds.

        Parameters
        ----------
        x: array_like
            Vector of independent variables

        Returns
        -------
        sl, sb : array-like
            The lower and upper residuals
        """
        return x - self.lb, self.ub - x


class PreparedConstraint:
    """Constraint prepared from a user defined constraint.

    On creation it will check whether a constraint definition is valid and
    the initial point is feasible. If created successfully, it will contain
    the attributes listed below.

    Parameters
    ----------
    constraint : {NonlinearConstraint, LinearConstraint`, Bounds}
        Constraint to check and prepare.
    x0 : array_like
        Initial vector of independent variables.
    sparse_jacobian : bool or None, optional
        If bool, then the Jacobian of the constraint will be converted
        to the corresponded format if necessary. If None (default), such
        conversion is not made.
    finite_diff_bounds : 2-tuple, optional
        Lower and upper bounds on the independent variables for the finite
        difference approximation, if applicable. Defaults to no bounds.

    Attributes
    ----------
    fun : {VectorFunction, LinearVectorFunction, IdentityVectorFunction}
        Function defining the constraint wrapped by one of the convenience
        classes.
    bounds : 2-tuple
        Contains lower and upper bounds for the constraints --- lb and ub.
        These are converted to ndarray and have a size equal to the number of
        the constraints.
    keep_feasible : ndarray
         Array indicating which components must be kept feasible with a size
         equal to the number of the constraints.
    """
    def __init__(self, constraint, x0, sparse_jacobian=None,
                 finite_diff_bounds=(-np.inf, np.inf)):
        if isinstance(constraint, NonlinearConstraint):
            fun = VectorFunction(constraint.fun, x0,
                                 constraint.jac, constraint.hess,
                                 constraint.finite_diff_rel_step,
                                 constraint.finite_diff_jac_sparsity,
                                 finite_diff_bounds, sparse_jacobian)
        elif isinstance(constraint, LinearConstraint):
            fun = LinearVectorFunction(constraint.A, x0, sparse_jacobian)
        elif isinstance(constraint, Bounds):
            fun = IdentityVectorFunction(x0, sparse_jacobian)
        else:
            raise ValueError("`constraint` of an unknown type is passed.")

        m = fun.m

        lb = np.asarray(constraint.lb, dtype=float)
        ub = np.asarray(constraint.ub, dtype=float)
        keep_feasible = np.asarray(constraint.keep_feasible, dtype=bool)

        lb = np.broadcast_to(lb, m)
        ub = np.broadcast_to(ub, m)
        keep_feasible = np.broadcast_to(keep_feasible, m)

        if keep_feasible.shape != (m,):
            raise ValueError("`keep_feasible` has a wrong shape.")

        mask = keep_feasible & (lb != ub)
        f0 = fun.f
        if np.any(f0[mask] < lb[mask]) or np.any(f0[mask] > ub[mask]):
            raise ValueError("`x0` is infeasible with respect to some "
                             "inequality constraint with `keep_feasible` "
                             "set to True.")

        self.fun = fun
        self.bounds = (lb, ub)
        self.keep_feasible = keep_feasible

    def violation(self, x):
        """How much the constraint is exceeded by.

        Parameters
        ----------
        x : array-like
            Vector of independent variables

        Returns
        -------
        excess : array-like
            How much the constraint is exceeded by, for each of the
            constraints specified by `PreparedConstraint.fun`.
        """
        with catch_warnings():
            # Ignore the following warning, it's not important when
            # figuring out total violation
            # UserWarning: delta_grad == 0.0. Check if the approximated
            # function is linear
            filterwarnings("ignore", "delta_grad", UserWarning)
            ev = self.fun.fun(np.asarray(x))

        excess_lb = np.maximum(self.bounds[0] - ev, 0)
        excess_ub = np.maximum(ev - self.bounds[1], 0)

        return excess_lb + excess_ub


def new_bounds_to_old(lb, ub, n):
    """Convert the new bounds representation to the old one.

    The new representation is a tuple (lb, ub) and the old one is a list
    containing n tuples, ith containing lower and upper bound on a ith
    variable.
    If any of the entries in lb/ub are -np.inf/np.inf they are replaced by
    None.
    """
    lb = np.broadcast_to(lb, n)
    ub = np.broadcast_to(ub, n)

    lb = [float(x) if x > -np.inf else None for x in lb]
    ub = [float(x) if x < np.inf else None for x in ub]

    return list(zip(lb, ub))


def old_bound_to_new(bounds):
    """Convert the old bounds representation to the new one.

    The new representation is a tuple (lb, ub) and the old one is a list
    containing n tuples, ith containing lower and upper bound on a ith
    variable.
    If any of the entries in lb/ub are None they are replaced by
    -np.inf/np.inf.
    """
    lb, ub = zip(*bounds)

    # Convert occurrences of None to -inf or inf, and replace occurrences of
    # any numpy array x with x.item(). Then wrap the results in numpy arrays.
    lb = np.array([float(_arr_to_scalar(x)) if x is not None else -np.inf
                   for x in lb])
    ub = np.array([float(_arr_to_scalar(x)) if x is not None else np.inf
                   for x in ub])

    return lb, ub


def strict_bounds(lb, ub, keep_feasible, n_vars):
    """Remove bounds which are not asked to be kept feasible."""
    strict_lb = np.resize(lb, n_vars).astype(float)
    strict_ub = np.resize(ub, n_vars).astype(float)
    keep_feasible = np.resize(keep_feasible, n_vars)
    strict_lb[~keep_feasible] = -np.inf
    strict_ub[~keep_feasible] = np.inf
    return strict_lb, strict_ub


def new_constraint_to_old(con, x0):
    """
    Converts new-style constraint objects to old-style constraint dictionaries.
    """
    if isinstance(con, NonlinearConstraint):
        if (con.finite_diff_jac_sparsity is not None or
                con.finite_diff_rel_step is not None or
                not isinstance(con.hess, BFGS) or  # misses user specified BFGS
                con.keep_feasible):
            warn("Constraint options `finite_diff_jac_sparsity`, "
                 "`finite_diff_rel_step`, `keep_feasible`, and `hess`"
                 "are ignored by this method.",
                 OptimizeWarning, stacklevel=3)

        fun = con.fun
        if callable(con.jac):
            jac = con.jac
        else:
            jac = None

    else:  # LinearConstraint
        if np.any(con.keep_feasible):
            warn("Constraint option `keep_feasible` is ignored by this method.",
                 OptimizeWarning, stacklevel=3)

        A = con.A
        if issparse(A):
            A = A.toarray()
        def fun(x):
            return np.dot(A, x)
        def jac(x):
            return A

    # FIXME: when bugs in VectorFunction/LinearVectorFunction are worked out,
    # use pcon.fun.fun and pcon.fun.jac. Until then, get fun/jac above.
    pcon = PreparedConstraint(con, x0)
    lb, ub = pcon.bounds

    i_eq = lb == ub
    i_bound_below = np.logical_xor(lb != -np.inf, i_eq)
    i_bound_above = np.logical_xor(ub != np.inf, i_eq)
    i_unbounded = np.logical_and(lb == -np.inf, ub == np.inf)

    if np.any(i_unbounded):
        warn("At least one constraint is unbounded above and below. Such "
             "constraints are ignored.",
             OptimizeWarning, stacklevel=3)

    ceq = []
    if np.any(i_eq):
        def f_eq(x):
            y = np.array(fun(x)).flatten()
            return y[i_eq] - lb[i_eq]
        ceq = [{"type": "eq", "fun": f_eq}]

        if jac is not None:
            def j_eq(x):
                dy = jac(x)
                if issparse(dy):
                    dy = dy.toarray()
                dy = np.atleast_2d(dy)
                return dy[i_eq, :]
            ceq[0]["jac"] = j_eq

    cineq = []
    n_bound_below = np.sum(i_bound_below)
    n_bound_above = np.sum(i_bound_above)
    if n_bound_below + n_bound_above:
        def f_ineq(x):
            y = np.zeros(n_bound_below + n_bound_above)
            y_all = np.array(fun(x)).flatten()
            y[:n_bound_below] = y_all[i_bound_below] - lb[i_bound_below]
            y[n_bound_below:] = -(y_all[i_bound_above] - ub[i_bound_above])
            return y
        cineq = [{"type": "ineq", "fun": f_ineq}]

        if jac is not None:
            def j_ineq(x):
                dy = np.zeros((n_bound_below + n_bound_above, len(x0)))
                dy_all = jac(x)
                if issparse(dy_all):
                    dy_all = dy_all.toarray()
                dy_all = np.atleast_2d(dy_all)
                dy[:n_bound_below, :] = dy_all[i_bound_below]
                dy[n_bound_below:, :] = -dy_all[i_bound_above]
                return dy
            cineq[0]["jac"] = j_ineq

    old_constraints = ceq + cineq

    if len(old_constraints) > 1:
        warn("Equality and inequality constraints are specified in the same "
             "element of the constraint list. For efficient use with this "
             "method, equality and inequality constraints should be specified "
             "in separate elements of the constraint list. ",
             OptimizeWarning, stacklevel=3)
    return old_constraints


def old_constraint_to_new(ic, con):
    """
    Converts old-style constraint dictionaries to new-style constraint objects.
    """
    # check type
    try:
        ctype = con['type'].lower()
    except KeyError as e:
        raise KeyError('Constraint %d has no type defined.' % ic) from e
    except TypeError as e:
        raise TypeError(
            'Constraints must be a sequence of dictionaries.'
        ) from e
    except AttributeError as e:
        raise TypeError("Constraint's type must be a string.") from e
    else:
        if ctype not in ['eq', 'ineq']:
            raise ValueError("Unknown constraint type '%s'." % con['type'])
    if 'fun' not in con:
        raise ValueError('Constraint %d has no function defined.' % ic)

    lb = 0
    if ctype == 'eq':
        ub = 0
    else:
        ub = np.inf

    jac = '2-point'
    if 'args' in con:
        args = con['args']
        def fun(x):
            return con["fun"](x, *args)
        if 'jac' in con:
            def jac(x):
                return con["jac"](x, *args)
    else:
        fun = con['fun']
        if 'jac' in con:
            jac = con['jac']

    return NonlinearConstraint(fun, lb, ub, jac)
