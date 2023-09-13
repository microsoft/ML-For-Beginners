from __future__ import annotations
from typing import (  # noqa: UP035
    Any, Callable, Iterable, TYPE_CHECKING
)

import numpy as np
from scipy.optimize import OptimizeResult
from ._constraints import old_bound_to_new, Bounds
from ._direct import direct as _direct  # type: ignore

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = ['direct']

ERROR_MESSAGES = (
    "Number of function evaluations done is larger than maxfun={}",
    "Number of iterations is larger than maxiter={}",
    "u[i] < l[i] for some i",
    "maxfun is too large",
    "Initialization failed",
    "There was an error in the creation of the sample points",
    "An error occurred while the function was sampled",
    "Maximum number of levels has been reached.",
    "Forced stop",
    "Invalid arguments",
    "Out of memory",
)

SUCCESS_MESSAGES = (
    ("The best function value found is within a relative error={} "
     "of the (known) global optimum f_min"),
    ("The volume of the hyperrectangle containing the lowest function value "
     "found is below vol_tol={}"),
    ("The side length measure of the hyperrectangle containing the lowest "
     "function value found is below len_tol={}"),
)


def direct(
    func: Callable[[npt.ArrayLike, tuple[Any]], float],
    bounds: Iterable | Bounds,
    *,
    args: tuple = (),
    eps: float = 1e-4,
    maxfun: int | None = None,
    maxiter: int = 1000,
    locally_biased: bool = True,
    f_min: float = -np.inf,
    f_min_rtol: float = 1e-4,
    vol_tol: float = 1e-16,
    len_tol: float = 1e-6,
    callback: Callable[[npt.ArrayLike], None] | None = None
) -> OptimizeResult:
    """
    Finds the global minimum of a function using the
    DIRECT algorithm.

    Parameters
    ----------
    func : callable
        The objective function to be minimized.
        ``func(x, *args) -> float``
        where ``x`` is an 1-D array with shape (n,) and ``args`` is a tuple of
        the fixed parameters needed to completely specify the function.
    bounds : sequence or `Bounds`
        Bounds for variables. There are two ways to specify the bounds:

        1. Instance of `Bounds` class.
        2. ``(min, max)`` pairs for each element in ``x``.

    args : tuple, optional
        Any additional fixed parameters needed to
        completely specify the objective function.
    eps : float, optional
        Minimal required difference of the objective function values
        between the current best hyperrectangle and the next potentially
        optimal hyperrectangle to be divided. In consequence, `eps` serves as a
        tradeoff between local and global search: the smaller, the more local
        the search becomes. Default is 1e-4.
    maxfun : int or None, optional
        Approximate upper bound on objective function evaluations.
        If `None`, will be automatically set to ``1000 * N`` where ``N``
        represents the number of dimensions. Will be capped if necessary to
        limit DIRECT's RAM usage to app. 1GiB. This will only occur for very
        high dimensional problems and excessive `max_fun`. Default is `None`.
    maxiter : int, optional
        Maximum number of iterations. Default is 1000.
    locally_biased : bool, optional
        If `True` (default), use the locally biased variant of the
        algorithm known as DIRECT_L. If `False`, use the original unbiased
        DIRECT algorithm. For hard problems with many local minima,
        `False` is recommended.
    f_min : float, optional
        Function value of the global optimum. Set this value only if the
        global optimum is known. Default is ``-np.inf``, so that this
        termination criterion is deactivated.
    f_min_rtol : float, optional
        Terminate the optimization once the relative error between the
        current best minimum `f` and the supplied global minimum `f_min`
        is smaller than `f_min_rtol`. This parameter is only used if
        `f_min` is also set. Must lie between 0 and 1. Default is 1e-4.
    vol_tol : float, optional
        Terminate the optimization once the volume of the hyperrectangle
        containing the lowest function value is smaller than `vol_tol`
        of the complete search space. Must lie between 0 and 1.
        Default is 1e-16.
    len_tol : float, optional
        If `locally_biased=True`, terminate the optimization once half of
        the normalized maximal side length of the hyperrectangle containing
        the lowest function value is smaller than `len_tol`.
        If `locally_biased=False`, terminate the optimization once half of
        the normalized diagonal of the hyperrectangle containing the lowest
        function value is smaller than `len_tol`. Must lie between 0 and 1.
        Default is 1e-6.
    callback : callable, optional
        A callback function with signature ``callback(xk)`` where ``xk``
        represents the best function value found so far.

    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a ``OptimizeResult`` object.
        Important attributes are: ``x`` the solution array, ``success`` a
        Boolean flag indicating if the optimizer exited successfully and
        ``message`` which describes the cause of the termination. See
        `OptimizeResult` for a description of other attributes.

    Notes
    -----
    DIviding RECTangles (DIRECT) is a deterministic global
    optimization algorithm capable of minimizing a black box function with
    its variables subject to lower and upper bound constraints by sampling
    potential solutions in the search space [1]_. The algorithm starts by
    normalising the search space to an n-dimensional unit hypercube.
    It samples the function at the center of this hypercube and at 2n
    (n is the number of variables) more points, 2 in each coordinate
    direction. Using these function values, DIRECT then divides the
    domain into hyperrectangles, each having exactly one of the sampling
    points as its center. In each iteration, DIRECT chooses, using the `eps`
    parameter which defaults to 1e-4, some of the existing hyperrectangles
    to be further divided. This division process continues until either the
    maximum number of iterations or maximum function evaluations allowed
    are exceeded, or the hyperrectangle containing the minimal value found
    so far becomes small enough. If `f_min` is specified, the optimization
    will stop once this function value is reached within a relative tolerance.
    The locally biased variant of DIRECT (originally called DIRECT_L) [2]_ is
    used by default. It makes the search more locally biased and more
    efficient for cases with only a few local minima.

    A note about termination criteria: `vol_tol` refers to the volume of the
    hyperrectangle containing the lowest function value found so far. This
    volume decreases exponentially with increasing dimensionality of the
    problem. Therefore `vol_tol` should be decreased to avoid premature
    termination of the algorithm for higher dimensions. This does not hold
    for `len_tol`: it refers either to half of the maximal side length
    (for ``locally_biased=True``) or half of the diagonal of the
    hyperrectangle (for ``locally_biased=False``).

    This code is based on the DIRECT 2.0.4 Fortran code by Gablonsky et al. at
    https://ctk.math.ncsu.edu/SOFTWARE/DIRECTv204.tar.gz .
    This original version was initially converted via f2c and then cleaned up
    and reorganized by Steven G. Johnson, August 2007, for the NLopt project.
    The `direct` function wraps the C implementation.

    .. versionadded:: 1.9.0

    References
    ----------
    .. [1] Jones, D.R., Perttunen, C.D. & Stuckman, B.E. Lipschitzian
        optimization without the Lipschitz constant. J Optim Theory Appl
        79, 157-181 (1993).
    .. [2] Gablonsky, J., Kelley, C. A Locally-Biased form of the DIRECT
        Algorithm. Journal of Global Optimization 21, 27-37 (2001).

    Examples
    --------
    The following example is a 2-D problem with four local minima: minimizing
    the Styblinski-Tang function
    (https://en.wikipedia.org/wiki/Test_functions_for_optimization).

    >>> from scipy.optimize import direct, Bounds
    >>> def styblinski_tang(pos):
    ...     x, y = pos
    ...     return 0.5 * (x**4 - 16*x**2 + 5*x + y**4 - 16*y**2 + 5*y)
    >>> bounds = Bounds([-4., -4.], [4., 4.])
    >>> result = direct(styblinski_tang, bounds)
    >>> result.x, result.fun, result.nfev
    array([-2.90321597, -2.90321597]), -78.3323279095383, 2011

    The correct global minimum was found but with a huge number of function
    evaluations (2011). Loosening the termination tolerances `vol_tol` and
    `len_tol` can be used to stop DIRECT earlier.

    >>> result = direct(styblinski_tang, bounds, len_tol=1e-3)
    >>> result.x, result.fun, result.nfev
    array([-2.9044353, -2.9044353]), -78.33230330754142, 207

    """
    # convert bounds to new Bounds class if necessary
    if not isinstance(bounds, Bounds):
        if isinstance(bounds, list) or isinstance(bounds, tuple):
            lb, ub = old_bound_to_new(bounds)
            bounds = Bounds(lb, ub)
        else:
            message = ("bounds must be a sequence or "
                       "instance of Bounds class")
            raise ValueError(message)

    lb = np.ascontiguousarray(bounds.lb, dtype=np.float64)
    ub = np.ascontiguousarray(bounds.ub, dtype=np.float64)

    # validate bounds
    # check that lower bounds are smaller than upper bounds
    if not np.all(lb < ub):
        raise ValueError('Bounds are not consistent min < max')
    # check for infs
    if (np.any(np.isinf(lb)) or np.any(np.isinf(ub))):
        raise ValueError("Bounds must not be inf.")

    # validate tolerances
    if (vol_tol < 0 or vol_tol > 1):
        raise ValueError("vol_tol must be between 0 and 1.")
    if (len_tol < 0 or len_tol > 1):
        raise ValueError("len_tol must be between 0 and 1.")
    if (f_min_rtol < 0 or f_min_rtol > 1):
        raise ValueError("f_min_rtol must be between 0 and 1.")

    # validate maxfun and maxiter
    if maxfun is None:
        maxfun = 1000 * lb.shape[0]
    if not isinstance(maxfun, int):
        raise ValueError("maxfun must be of type int.")
    if maxfun < 0:
        raise ValueError("maxfun must be > 0.")
    if not isinstance(maxiter, int):
        raise ValueError("maxiter must be of type int.")
    if maxiter < 0:
        raise ValueError("maxiter must be > 0.")

    # validate boolean parameters
    if not isinstance(locally_biased, bool):
        raise ValueError("locally_biased must be True or False.")

    def _func_wrap(x, args=None):
        x = np.asarray(x)
        if args is None:
            f = func(x)
        else:
            f = func(x, *args)
        # always return a float
        return np.asarray(f).item()

    # TODO: fix disp argument
    x, fun, ret_code, nfev, nit = _direct(
        _func_wrap,
        np.asarray(lb), np.asarray(ub),
        args,
        False, eps, maxfun, maxiter,
        locally_biased,
        f_min, f_min_rtol,
        vol_tol, len_tol, callback
    )

    format_val = (maxfun, maxiter, f_min_rtol, vol_tol, len_tol)
    if ret_code > 2:
        message = SUCCESS_MESSAGES[ret_code - 3].format(
                    format_val[ret_code - 1])
    elif 0 < ret_code <= 2:
        message = ERROR_MESSAGES[ret_code - 1].format(format_val[ret_code - 1])
    elif 0 > ret_code > -100:
        message = ERROR_MESSAGES[abs(ret_code) + 1]
    else:
        message = ERROR_MESSAGES[ret_code + 99]

    return OptimizeResult(x=np.asarray(x), fun=fun, status=ret_code,
                          success=ret_code > 2, message=message,
                          nfev=nfev, nit=nit)
