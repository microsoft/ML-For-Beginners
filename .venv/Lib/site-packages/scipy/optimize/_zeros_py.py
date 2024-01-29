import warnings
from collections import namedtuple
import operator
from . import _zeros
from ._optimize import OptimizeResult, _call_callback_maybe_halt
import numpy as np


_iter = 100
_xtol = 2e-12
_rtol = 4 * np.finfo(float).eps

__all__ = ['newton', 'bisect', 'ridder', 'brentq', 'brenth', 'toms748',
           'RootResults']

# Must agree with CONVERGED, SIGNERR, CONVERR, ...  in zeros.h
_ECONVERGED = 0
_ESIGNERR = -1  # used in _chandrupatla
_EERRORINCREASE = -1  # used in _differentiate
_ELIMITS = -1  # used in _bracket_root
_ECONVERR = -2
_EVALUEERR = -3
_ECALLBACK = -4
_EINPROGRESS = 1
_ESTOPONESIDE = 2  # used in _bracket_root

CONVERGED = 'converged'
SIGNERR = 'sign error'
CONVERR = 'convergence error'
VALUEERR = 'value error'
INPROGRESS = 'No error'


flag_map = {_ECONVERGED: CONVERGED, _ESIGNERR: SIGNERR, _ECONVERR: CONVERR,
            _EVALUEERR: VALUEERR, _EINPROGRESS: INPROGRESS}


class RootResults(OptimizeResult):
    """Represents the root finding result.

    Attributes
    ----------
    root : float
        Estimated root location.
    iterations : int
        Number of iterations needed to find the root.
    function_calls : int
        Number of times the function was called.
    converged : bool
        True if the routine converged.
    flag : str
        Description of the cause of termination.
    method : str
        Root finding method used.

    """

    def __init__(self, root, iterations, function_calls, flag, method):
        self.root = root
        self.iterations = iterations
        self.function_calls = function_calls
        self.converged = flag == _ECONVERGED
        if flag in flag_map:
            self.flag = flag_map[flag]
        else:
            self.flag = flag
        self.method = method


def results_c(full_output, r, method):
    if full_output:
        x, funcalls, iterations, flag = r
        results = RootResults(root=x,
                              iterations=iterations,
                              function_calls=funcalls,
                              flag=flag, method=method)
        return x, results
    else:
        return r


def _results_select(full_output, r, method):
    """Select from a tuple of (root, funccalls, iterations, flag)"""
    x, funcalls, iterations, flag = r
    if full_output:
        results = RootResults(root=x,
                              iterations=iterations,
                              function_calls=funcalls,
                              flag=flag, method=method)
        return x, results
    return x


def _wrap_nan_raise(f):

    def f_raise(x, *args):
        fx = f(x, *args)
        f_raise._function_calls += 1
        if np.isnan(fx):
            msg = (f'The function value at x={x} is NaN; '
                   'solver cannot continue.')
            err = ValueError(msg)
            err._x = x
            err._function_calls = f_raise._function_calls
            raise err
        return fx

    f_raise._function_calls = 0
    return f_raise


def newton(func, x0, fprime=None, args=(), tol=1.48e-8, maxiter=50,
           fprime2=None, x1=None, rtol=0.0,
           full_output=False, disp=True):
    """
    Find a root of a real or complex function using the Newton-Raphson
    (or secant or Halley's) method.

    Find a root of the scalar-valued function `func` given a nearby scalar
    starting point `x0`.
    The Newton-Raphson method is used if the derivative `fprime` of `func`
    is provided, otherwise the secant method is used. If the second order
    derivative `fprime2` of `func` is also provided, then Halley's method is
    used.

    If `x0` is a sequence with more than one item, `newton` returns an array:
    the roots of the function from each (scalar) starting point in `x0`.
    In this case, `func` must be vectorized to return a sequence or array of
    the same shape as its first argument. If `fprime` (`fprime2`) is given,
    then its return must also have the same shape: each element is the first
    (second) derivative of `func` with respect to its only variable evaluated
    at each element of its first argument.

    `newton` is for finding roots of a scalar-valued functions of a single
    variable. For problems involving several variables, see `root`.

    Parameters
    ----------
    func : callable
        The function whose root is wanted. It must be a function of a
        single variable of the form ``f(x,a,b,c...)``, where ``a,b,c...``
        are extra arguments that can be passed in the `args` parameter.
    x0 : float, sequence, or ndarray
        An initial estimate of the root that should be somewhere near the
        actual root. If not scalar, then `func` must be vectorized and return
        a sequence or array of the same shape as its first argument.
    fprime : callable, optional
        The derivative of the function when available and convenient. If it
        is None (default), then the secant method is used.
    args : tuple, optional
        Extra arguments to be used in the function call.
    tol : float, optional
        The allowable error of the root's value. If `func` is complex-valued,
        a larger `tol` is recommended as both the real and imaginary parts
        of `x` contribute to ``|x - x0|``.
    maxiter : int, optional
        Maximum number of iterations.
    fprime2 : callable, optional
        The second order derivative of the function when available and
        convenient. If it is None (default), then the normal Newton-Raphson
        or the secant method is used. If it is not None, then Halley's method
        is used.
    x1 : float, optional
        Another estimate of the root that should be somewhere near the
        actual root. Used if `fprime` is not provided.
    rtol : float, optional
        Tolerance (relative) for termination.
    full_output : bool, optional
        If `full_output` is False (default), the root is returned.
        If True and `x0` is scalar, the return value is ``(x, r)``, where ``x``
        is the root and ``r`` is a `RootResults` object.
        If True and `x0` is non-scalar, the return value is ``(x, converged,
        zero_der)`` (see Returns section for details).
    disp : bool, optional
        If True, raise a RuntimeError if the algorithm didn't converge, with
        the error message containing the number of iterations and current
        function value. Otherwise, the convergence status is recorded in a
        `RootResults` return object.
        Ignored if `x0` is not scalar.
        *Note: this has little to do with displaying, however,
        the `disp` keyword cannot be renamed for backwards compatibility.*

    Returns
    -------
    root : float, sequence, or ndarray
        Estimated location where function is zero.
    r : `RootResults`, optional
        Present if ``full_output=True`` and `x0` is scalar.
        Object containing information about the convergence. In particular,
        ``r.converged`` is True if the routine converged.
    converged : ndarray of bool, optional
        Present if ``full_output=True`` and `x0` is non-scalar.
        For vector functions, indicates which elements converged successfully.
    zero_der : ndarray of bool, optional
        Present if ``full_output=True`` and `x0` is non-scalar.
        For vector functions, indicates which elements had a zero derivative.

    See Also
    --------
    root_scalar : interface to root solvers for scalar functions
    root : interface to root solvers for multi-input, multi-output functions

    Notes
    -----
    The convergence rate of the Newton-Raphson method is quadratic,
    the Halley method is cubic, and the secant method is
    sub-quadratic. This means that if the function is well-behaved
    the actual error in the estimated root after the nth iteration
    is approximately the square (cube for Halley) of the error
    after the (n-1)th step. However, the stopping criterion used
    here is the step size and there is no guarantee that a root
    has been found. Consequently, the result should be verified.
    Safer algorithms are brentq, brenth, ridder, and bisect,
    but they all require that the root first be bracketed in an
    interval where the function changes sign. The brentq algorithm
    is recommended for general use in one dimensional problems
    when such an interval has been found.

    When `newton` is used with arrays, it is best suited for the following
    types of problems:

    * The initial guesses, `x0`, are all relatively the same distance from
      the roots.
    * Some or all of the extra arguments, `args`, are also arrays so that a
      class of similar problems can be solved together.
    * The size of the initial guesses, `x0`, is larger than O(100) elements.
      Otherwise, a naive loop may perform as well or better than a vector.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy import optimize

    >>> def f(x):
    ...     return (x**3 - 1)  # only one real root at x = 1

    ``fprime`` is not provided, use the secant method:

    >>> root = optimize.newton(f, 1.5)
    >>> root
    1.0000000000000016
    >>> root = optimize.newton(f, 1.5, fprime2=lambda x: 6 * x)
    >>> root
    1.0000000000000016

    Only ``fprime`` is provided, use the Newton-Raphson method:

    >>> root = optimize.newton(f, 1.5, fprime=lambda x: 3 * x**2)
    >>> root
    1.0

    Both ``fprime2`` and ``fprime`` are provided, use Halley's method:

    >>> root = optimize.newton(f, 1.5, fprime=lambda x: 3 * x**2,
    ...                        fprime2=lambda x: 6 * x)
    >>> root
    1.0

    When we want to find roots for a set of related starting values and/or
    function parameters, we can provide both of those as an array of inputs:

    >>> f = lambda x, a: x**3 - a
    >>> fder = lambda x, a: 3 * x**2
    >>> rng = np.random.default_rng()
    >>> x = rng.standard_normal(100)
    >>> a = np.arange(-50, 50)
    >>> vec_res = optimize.newton(f, x, fprime=fder, args=(a, ), maxiter=200)

    The above is the equivalent of solving for each value in ``(x, a)``
    separately in a for-loop, just faster:

    >>> loop_res = [optimize.newton(f, x0, fprime=fder, args=(a0,),
    ...                             maxiter=200)
    ...             for x0, a0 in zip(x, a)]
    >>> np.allclose(vec_res, loop_res)
    True

    Plot the results found for all values of ``a``:

    >>> analytical_result = np.sign(a) * np.abs(a)**(1/3)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(a, analytical_result, 'o')
    >>> ax.plot(a, vec_res, '.')
    >>> ax.set_xlabel('$a$')
    >>> ax.set_ylabel('$x$ where $f(x, a)=0$')
    >>> plt.show()

    """
    if tol <= 0:
        raise ValueError("tol too small (%g <= 0)" % tol)
    maxiter = operator.index(maxiter)
    if maxiter < 1:
        raise ValueError("maxiter must be greater than 0")
    if np.size(x0) > 1:
        return _array_newton(func, x0, fprime, args, tol, maxiter, fprime2,
                             full_output)

    # Convert to float (don't use float(x0); this works also for complex x0)
    # Use np.asarray because we want x0 to be a numpy object, not a Python
    # object. e.g. np.complex(1+1j) > 0 is possible, but (1 + 1j) > 0 raises
    # a TypeError
    x0 = np.asarray(x0)[()] * 1.0
    p0 = x0
    funcalls = 0
    if fprime is not None:
        # Newton-Raphson method
        method = "newton"
        for itr in range(maxiter):
            # first evaluate fval
            fval = func(p0, *args)
            funcalls += 1
            # If fval is 0, a root has been found, then terminate
            if fval == 0:
                return _results_select(
                    full_output, (p0, funcalls, itr, _ECONVERGED), method)
            fder = fprime(p0, *args)
            funcalls += 1
            if fder == 0:
                msg = "Derivative was zero."
                if disp:
                    msg += (
                        " Failed to converge after %d iterations, value is %s."
                        % (itr + 1, p0))
                    raise RuntimeError(msg)
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
                return _results_select(
                    full_output, (p0, funcalls, itr + 1, _ECONVERR), method)
            newton_step = fval / fder
            if fprime2:
                fder2 = fprime2(p0, *args)
                funcalls += 1
                method = "halley"
                # Halley's method:
                #   newton_step /= (1.0 - 0.5 * newton_step * fder2 / fder)
                # Only do it if denominator stays close enough to 1
                # Rationale: If 1-adj < 0, then Halley sends x in the
                # opposite direction to Newton. Doesn't happen if x is close
                # enough to root.
                adj = newton_step * fder2 / fder / 2
                if np.abs(adj) < 1:
                    newton_step /= 1.0 - adj
            p = p0 - newton_step
            if np.isclose(p, p0, rtol=rtol, atol=tol):
                return _results_select(
                    full_output, (p, funcalls, itr + 1, _ECONVERGED), method)
            p0 = p
    else:
        # Secant method
        method = "secant"
        if x1 is not None:
            if x1 == x0:
                raise ValueError("x1 and x0 must be different")
            p1 = x1
        else:
            eps = 1e-4
            p1 = x0 * (1 + eps)
            p1 += (eps if p1 >= 0 else -eps)
        q0 = func(p0, *args)
        funcalls += 1
        q1 = func(p1, *args)
        funcalls += 1
        if abs(q1) < abs(q0):
            p0, p1, q0, q1 = p1, p0, q1, q0
        for itr in range(maxiter):
            if q1 == q0:
                if p1 != p0:
                    msg = "Tolerance of %s reached." % (p1 - p0)
                    if disp:
                        msg += (
                            " Failed to converge after %d iterations, value is %s."
                            % (itr + 1, p1))
                        raise RuntimeError(msg)
                    warnings.warn(msg, RuntimeWarning, stacklevel=2)
                p = (p1 + p0) / 2.0
                return _results_select(
                    full_output, (p, funcalls, itr + 1, _ECONVERR), method)
            else:
                if abs(q1) > abs(q0):
                    p = (-q0 / q1 * p1 + p0) / (1 - q0 / q1)
                else:
                    p = (-q1 / q0 * p0 + p1) / (1 - q1 / q0)
            if np.isclose(p, p1, rtol=rtol, atol=tol):
                return _results_select(
                    full_output, (p, funcalls, itr + 1, _ECONVERGED), method)
            p0, q0 = p1, q1
            p1 = p
            q1 = func(p1, *args)
            funcalls += 1

    if disp:
        msg = ("Failed to converge after %d iterations, value is %s."
               % (itr + 1, p))
        raise RuntimeError(msg)

    return _results_select(full_output, (p, funcalls, itr + 1, _ECONVERR), method)


def _array_newton(func, x0, fprime, args, tol, maxiter, fprime2, full_output):
    """
    A vectorized version of Newton, Halley, and secant methods for arrays.

    Do not use this method directly. This method is called from `newton`
    when ``np.size(x0) > 1`` is ``True``. For docstring, see `newton`.
    """
    # Explicitly copy `x0` as `p` will be modified inplace, but the
    # user's array should not be altered.
    p = np.array(x0, copy=True)

    failures = np.ones_like(p, dtype=bool)
    nz_der = np.ones_like(failures)
    if fprime is not None:
        # Newton-Raphson method
        for iteration in range(maxiter):
            # first evaluate fval
            fval = np.asarray(func(p, *args))
            # If all fval are 0, all roots have been found, then terminate
            if not fval.any():
                failures = fval.astype(bool)
                break
            fder = np.asarray(fprime(p, *args))
            nz_der = (fder != 0)
            # stop iterating if all derivatives are zero
            if not nz_der.any():
                break
            # Newton step
            dp = fval[nz_der] / fder[nz_der]
            if fprime2 is not None:
                fder2 = np.asarray(fprime2(p, *args))
                dp = dp / (1.0 - 0.5 * dp * fder2[nz_der] / fder[nz_der])
            # only update nonzero derivatives
            p = np.asarray(p, dtype=np.result_type(p, dp, np.float64))
            p[nz_der] -= dp
            failures[nz_der] = np.abs(dp) >= tol  # items not yet converged
            # stop iterating if there aren't any failures, not incl zero der
            if not failures[nz_der].any():
                break
    else:
        # Secant method
        dx = np.finfo(float).eps**0.33
        p1 = p * (1 + dx) + np.where(p >= 0, dx, -dx)
        q0 = np.asarray(func(p, *args))
        q1 = np.asarray(func(p1, *args))
        active = np.ones_like(p, dtype=bool)
        for iteration in range(maxiter):
            nz_der = (q1 != q0)
            # stop iterating if all derivatives are zero
            if not nz_der.any():
                p = (p1 + p) / 2.0
                break
            # Secant Step
            dp = (q1 * (p1 - p))[nz_der] / (q1 - q0)[nz_der]
            # only update nonzero derivatives
            p = np.asarray(p, dtype=np.result_type(p, p1, dp, np.float64))
            p[nz_der] = p1[nz_der] - dp
            active_zero_der = ~nz_der & active
            p[active_zero_der] = (p1 + p)[active_zero_der] / 2.0
            active &= nz_der  # don't assign zero derivatives again
            failures[nz_der] = np.abs(dp) >= tol  # not yet converged
            # stop iterating if there aren't any failures, not incl zero der
            if not failures[nz_der].any():
                break
            p1, p = p, p1
            q0 = q1
            q1 = np.asarray(func(p1, *args))

    zero_der = ~nz_der & failures  # don't include converged with zero-ders
    if zero_der.any():
        # Secant warnings
        if fprime is None:
            nonzero_dp = (p1 != p)
            # non-zero dp, but infinite newton step
            zero_der_nz_dp = (zero_der & nonzero_dp)
            if zero_der_nz_dp.any():
                rms = np.sqrt(
                    sum((p1[zero_der_nz_dp] - p[zero_der_nz_dp]) ** 2)
                )
                warnings.warn(f'RMS of {rms:g} reached', RuntimeWarning, stacklevel=3)
        # Newton or Halley warnings
        else:
            all_or_some = 'all' if zero_der.all() else 'some'
            msg = f'{all_or_some:s} derivatives were zero'
            warnings.warn(msg, RuntimeWarning, stacklevel=3)
    elif failures.any():
        all_or_some = 'all' if failures.all() else 'some'
        msg = f'{all_or_some:s} failed to converge after {maxiter:d} iterations'
        if failures.all():
            raise RuntimeError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=3)

    if full_output:
        result = namedtuple('result', ('root', 'converged', 'zero_der'))
        p = result(p, ~failures, zero_der)

    return p


def bisect(f, a, b, args=(),
           xtol=_xtol, rtol=_rtol, maxiter=_iter,
           full_output=False, disp=True):
    """
    Find root of a function within an interval using bisection.

    Basic bisection routine to find a root of the function `f` between the
    arguments `a` and `b`. `f(a)` and `f(b)` cannot have the same signs.
    Slow but sure.

    Parameters
    ----------
    f : function
        Python function returning a number.  `f` must be continuous, and
        f(a) and f(b) must have opposite signs.
    a : scalar
        One end of the bracketing interval [a,b].
    b : scalar
        The other end of the bracketing interval [a,b].
    xtol : number, optional
        The computed root ``x0`` will satisfy ``np.allclose(x, x0,
        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
        parameter must be positive.
    rtol : number, optional
        The computed root ``x0`` will satisfy ``np.allclose(x, x0,
        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
        parameter cannot be smaller than its default value of
        ``4*np.finfo(float).eps``.
    maxiter : int, optional
        If convergence is not achieved in `maxiter` iterations, an error is
        raised. Must be >= 0.
    args : tuple, optional
        Containing extra arguments for the function `f`.
        `f` is called by ``apply(f, (x)+args)``.
    full_output : bool, optional
        If `full_output` is False, the root is returned. If `full_output` is
        True, the return value is ``(x, r)``, where x is the root, and r is
        a `RootResults` object.
    disp : bool, optional
        If True, raise RuntimeError if the algorithm didn't converge.
        Otherwise, the convergence status is recorded in a `RootResults`
        return object.

    Returns
    -------
    root : float
        Root of `f` between `a` and `b`.
    r : `RootResults` (present if ``full_output = True``)
        Object containing information about the convergence. In particular,
        ``r.converged`` is True if the routine converged.

    Examples
    --------

    >>> def f(x):
    ...     return (x**2 - 1)

    >>> from scipy import optimize

    >>> root = optimize.bisect(f, 0, 2)
    >>> root
    1.0

    >>> root = optimize.bisect(f, -2, 0)
    >>> root
    -1.0

    See Also
    --------
    brentq, brenth, bisect, newton
    fixed_point : scalar fixed-point finder
    fsolve : n-dimensional root-finding

    """
    if not isinstance(args, tuple):
        args = (args,)
    maxiter = operator.index(maxiter)
    if xtol <= 0:
        raise ValueError("xtol too small (%g <= 0)" % xtol)
    if rtol < _rtol:
        raise ValueError(f"rtol too small ({rtol:g} < {_rtol:g})")
    f = _wrap_nan_raise(f)
    r = _zeros._bisect(f, a, b, xtol, rtol, maxiter, args, full_output, disp)
    return results_c(full_output, r, "bisect")


def ridder(f, a, b, args=(),
           xtol=_xtol, rtol=_rtol, maxiter=_iter,
           full_output=False, disp=True):
    """
    Find a root of a function in an interval using Ridder's method.

    Parameters
    ----------
    f : function
        Python function returning a number. f must be continuous, and f(a) and
        f(b) must have opposite signs.
    a : scalar
        One end of the bracketing interval [a,b].
    b : scalar
        The other end of the bracketing interval [a,b].
    xtol : number, optional
        The computed root ``x0`` will satisfy ``np.allclose(x, x0,
        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
        parameter must be positive.
    rtol : number, optional
        The computed root ``x0`` will satisfy ``np.allclose(x, x0,
        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
        parameter cannot be smaller than its default value of
        ``4*np.finfo(float).eps``.
    maxiter : int, optional
        If convergence is not achieved in `maxiter` iterations, an error is
        raised. Must be >= 0.
    args : tuple, optional
        Containing extra arguments for the function `f`.
        `f` is called by ``apply(f, (x)+args)``.
    full_output : bool, optional
        If `full_output` is False, the root is returned. If `full_output` is
        True, the return value is ``(x, r)``, where `x` is the root, and `r` is
        a `RootResults` object.
    disp : bool, optional
        If True, raise RuntimeError if the algorithm didn't converge.
        Otherwise, the convergence status is recorded in any `RootResults`
        return object.

    Returns
    -------
    root : float
        Root of `f` between `a` and `b`.
    r : `RootResults` (present if ``full_output = True``)
        Object containing information about the convergence.
        In particular, ``r.converged`` is True if the routine converged.

    See Also
    --------
    brentq, brenth, bisect, newton : 1-D root-finding
    fixed_point : scalar fixed-point finder

    Notes
    -----
    Uses [Ridders1979]_ method to find a root of the function `f` between the
    arguments `a` and `b`. Ridders' method is faster than bisection, but not
    generally as fast as the Brent routines. [Ridders1979]_ provides the
    classic description and source of the algorithm. A description can also be
    found in any recent edition of Numerical Recipes.

    The routine used here diverges slightly from standard presentations in
    order to be a bit more careful of tolerance.

    References
    ----------
    .. [Ridders1979]
       Ridders, C. F. J. "A New Algorithm for Computing a
       Single Root of a Real Continuous Function."
       IEEE Trans. Circuits Systems 26, 979-980, 1979.

    Examples
    --------

    >>> def f(x):
    ...     return (x**2 - 1)

    >>> from scipy import optimize

    >>> root = optimize.ridder(f, 0, 2)
    >>> root
    1.0

    >>> root = optimize.ridder(f, -2, 0)
    >>> root
    -1.0
    """
    if not isinstance(args, tuple):
        args = (args,)
    maxiter = operator.index(maxiter)
    if xtol <= 0:
        raise ValueError("xtol too small (%g <= 0)" % xtol)
    if rtol < _rtol:
        raise ValueError(f"rtol too small ({rtol:g} < {_rtol:g})")
    f = _wrap_nan_raise(f)
    r = _zeros._ridder(f, a, b, xtol, rtol, maxiter, args, full_output, disp)
    return results_c(full_output, r, "ridder")


def brentq(f, a, b, args=(),
           xtol=_xtol, rtol=_rtol, maxiter=_iter,
           full_output=False, disp=True):
    """
    Find a root of a function in a bracketing interval using Brent's method.

    Uses the classic Brent's method to find a root of the function `f` on
    the sign changing interval [a , b]. Generally considered the best of the
    rootfinding routines here. It is a safe version of the secant method that
    uses inverse quadratic extrapolation. Brent's method combines root
    bracketing, interval bisection, and inverse quadratic interpolation. It is
    sometimes known as the van Wijngaarden-Dekker-Brent method. Brent (1973)
    claims convergence is guaranteed for functions computable within [a,b].

    [Brent1973]_ provides the classic description of the algorithm. Another
    description can be found in a recent edition of Numerical Recipes, including
    [PressEtal1992]_. A third description is at
    http://mathworld.wolfram.com/BrentsMethod.html. It should be easy to
    understand the algorithm just by reading our code. Our code diverges a bit
    from standard presentations: we choose a different formula for the
    extrapolation step.

    Parameters
    ----------
    f : function
        Python function returning a number. The function :math:`f`
        must be continuous, and :math:`f(a)` and :math:`f(b)` must
        have opposite signs.
    a : scalar
        One end of the bracketing interval :math:`[a, b]`.
    b : scalar
        The other end of the bracketing interval :math:`[a, b]`.
    xtol : number, optional
        The computed root ``x0`` will satisfy ``np.allclose(x, x0,
        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
        parameter must be positive. For nice functions, Brent's
        method will often satisfy the above condition with ``xtol/2``
        and ``rtol/2``. [Brent1973]_
    rtol : number, optional
        The computed root ``x0`` will satisfy ``np.allclose(x, x0,
        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
        parameter cannot be smaller than its default value of
        ``4*np.finfo(float).eps``. For nice functions, Brent's
        method will often satisfy the above condition with ``xtol/2``
        and ``rtol/2``. [Brent1973]_
    maxiter : int, optional
        If convergence is not achieved in `maxiter` iterations, an error is
        raised. Must be >= 0.
    args : tuple, optional
        Containing extra arguments for the function `f`.
        `f` is called by ``apply(f, (x)+args)``.
    full_output : bool, optional
        If `full_output` is False, the root is returned. If `full_output` is
        True, the return value is ``(x, r)``, where `x` is the root, and `r` is
        a `RootResults` object.
    disp : bool, optional
        If True, raise RuntimeError if the algorithm didn't converge.
        Otherwise, the convergence status is recorded in any `RootResults`
        return object.

    Returns
    -------
    root : float
        Root of `f` between `a` and `b`.
    r : `RootResults` (present if ``full_output = True``)
        Object containing information about the convergence. In particular,
        ``r.converged`` is True if the routine converged.

    Notes
    -----
    `f` must be continuous.  f(a) and f(b) must have opposite signs.

    Related functions fall into several classes:

    multivariate local optimizers
      `fmin`, `fmin_powell`, `fmin_cg`, `fmin_bfgs`, `fmin_ncg`
    nonlinear least squares minimizer
      `leastsq`
    constrained multivariate optimizers
      `fmin_l_bfgs_b`, `fmin_tnc`, `fmin_cobyla`
    global optimizers
      `basinhopping`, `brute`, `differential_evolution`
    local scalar minimizers
      `fminbound`, `brent`, `golden`, `bracket`
    N-D root-finding
      `fsolve`
    1-D root-finding
      `brenth`, `ridder`, `bisect`, `newton`
    scalar fixed-point finder
      `fixed_point`

    References
    ----------
    .. [Brent1973]
       Brent, R. P.,
       *Algorithms for Minimization Without Derivatives*.
       Englewood Cliffs, NJ: Prentice-Hall, 1973. Ch. 3-4.

    .. [PressEtal1992]
       Press, W. H.; Flannery, B. P.; Teukolsky, S. A.; and Vetterling, W. T.
       *Numerical Recipes in FORTRAN: The Art of Scientific Computing*, 2nd ed.
       Cambridge, England: Cambridge University Press, pp. 352-355, 1992.
       Section 9.3:  "Van Wijngaarden-Dekker-Brent Method."

    Examples
    --------
    >>> def f(x):
    ...     return (x**2 - 1)

    >>> from scipy import optimize

    >>> root = optimize.brentq(f, -2, 0)
    >>> root
    -1.0

    >>> root = optimize.brentq(f, 0, 2)
    >>> root
    1.0
    """
    if not isinstance(args, tuple):
        args = (args,)
    maxiter = operator.index(maxiter)
    if xtol <= 0:
        raise ValueError("xtol too small (%g <= 0)" % xtol)
    if rtol < _rtol:
        raise ValueError(f"rtol too small ({rtol:g} < {_rtol:g})")
    f = _wrap_nan_raise(f)
    r = _zeros._brentq(f, a, b, xtol, rtol, maxiter, args, full_output, disp)
    return results_c(full_output, r, "brentq")


def brenth(f, a, b, args=(),
           xtol=_xtol, rtol=_rtol, maxiter=_iter,
           full_output=False, disp=True):
    """Find a root of a function in a bracketing interval using Brent's
    method with hyperbolic extrapolation.

    A variation on the classic Brent routine to find a root of the function f
    between the arguments a and b that uses hyperbolic extrapolation instead of
    inverse quadratic extrapolation. Bus & Dekker (1975) guarantee convergence
    for this method, claiming that the upper bound of function evaluations here
    is 4 or 5 times that of bisection.
    f(a) and f(b) cannot have the same signs. Generally, on a par with the
    brent routine, but not as heavily tested. It is a safe version of the
    secant method that uses hyperbolic extrapolation.
    The version here is by Chuck Harris, and implements Algorithm M of
    [BusAndDekker1975]_, where further details (convergence properties,
    additional remarks and such) can be found

    Parameters
    ----------
    f : function
        Python function returning a number. f must be continuous, and f(a) and
        f(b) must have opposite signs.
    a : scalar
        One end of the bracketing interval [a,b].
    b : scalar
        The other end of the bracketing interval [a,b].
    xtol : number, optional
        The computed root ``x0`` will satisfy ``np.allclose(x, x0,
        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
        parameter must be positive. As with `brentq`, for nice
        functions the method will often satisfy the above condition
        with ``xtol/2`` and ``rtol/2``.
    rtol : number, optional
        The computed root ``x0`` will satisfy ``np.allclose(x, x0,
        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
        parameter cannot be smaller than its default value of
        ``4*np.finfo(float).eps``. As with `brentq`, for nice functions
        the method will often satisfy the above condition with
        ``xtol/2`` and ``rtol/2``.
    maxiter : int, optional
        If convergence is not achieved in `maxiter` iterations, an error is
        raised. Must be >= 0.
    args : tuple, optional
        Containing extra arguments for the function `f`.
        `f` is called by ``apply(f, (x)+args)``.
    full_output : bool, optional
        If `full_output` is False, the root is returned. If `full_output` is
        True, the return value is ``(x, r)``, where `x` is the root, and `r` is
        a `RootResults` object.
    disp : bool, optional
        If True, raise RuntimeError if the algorithm didn't converge.
        Otherwise, the convergence status is recorded in any `RootResults`
        return object.

    Returns
    -------
    root : float
        Root of `f` between `a` and `b`.
    r : `RootResults` (present if ``full_output = True``)
        Object containing information about the convergence. In particular,
        ``r.converged`` is True if the routine converged.

    See Also
    --------
    fmin, fmin_powell, fmin_cg, fmin_bfgs, fmin_ncg : multivariate local optimizers
    leastsq : nonlinear least squares minimizer
    fmin_l_bfgs_b, fmin_tnc, fmin_cobyla : constrained multivariate optimizers
    basinhopping, differential_evolution, brute : global optimizers
    fminbound, brent, golden, bracket : local scalar minimizers
    fsolve : N-D root-finding
    brentq, brenth, ridder, bisect, newton : 1-D root-finding
    fixed_point : scalar fixed-point finder

    References
    ----------
    .. [BusAndDekker1975]
       Bus, J. C. P., Dekker, T. J.,
       "Two Efficient Algorithms with Guaranteed Convergence for Finding a Zero
       of a Function", ACM Transactions on Mathematical Software, Vol. 1, Issue
       4, Dec. 1975, pp. 330-345. Section 3: "Algorithm M".
       :doi:`10.1145/355656.355659`

    Examples
    --------
    >>> def f(x):
    ...     return (x**2 - 1)

    >>> from scipy import optimize

    >>> root = optimize.brenth(f, -2, 0)
    >>> root
    -1.0

    >>> root = optimize.brenth(f, 0, 2)
    >>> root
    1.0

    """
    if not isinstance(args, tuple):
        args = (args,)
    maxiter = operator.index(maxiter)
    if xtol <= 0:
        raise ValueError("xtol too small (%g <= 0)" % xtol)
    if rtol < _rtol:
        raise ValueError(f"rtol too small ({rtol:g} < {_rtol:g})")
    f = _wrap_nan_raise(f)
    r = _zeros._brenth(f, a, b, xtol, rtol, maxiter, args, full_output, disp)
    return results_c(full_output, r, "brenth")


################################
# TOMS "Algorithm 748: Enclosing Zeros of Continuous Functions", by
#  Alefeld, G. E. and Potra, F. A. and Shi, Yixun,
#  See [1]


def _notclose(fs, rtol=_rtol, atol=_xtol):
    # Ensure not None, not 0, all finite, and not very close to each other
    notclosefvals = (
            all(fs) and all(np.isfinite(fs)) and
            not any(any(np.isclose(_f, fs[i + 1:], rtol=rtol, atol=atol))
                    for i, _f in enumerate(fs[:-1])))
    return notclosefvals


def _secant(xvals, fvals):
    """Perform a secant step, taking a little care"""
    # Secant has many "mathematically" equivalent formulations
    # x2 = x0 - (x1 - x0)/(f1 - f0) * f0
    #    = x1 - (x1 - x0)/(f1 - f0) * f1
    #    = (-x1 * f0 + x0 * f1) / (f1 - f0)
    #    = (-f0 / f1 * x1 + x0) / (1 - f0 / f1)
    #    = (-f1 / f0 * x0 + x1) / (1 - f1 / f0)
    x0, x1 = xvals[:2]
    f0, f1 = fvals[:2]
    if f0 == f1:
        return np.nan
    if np.abs(f1) > np.abs(f0):
        x2 = (-f0 / f1 * x1 + x0) / (1 - f0 / f1)
    else:
        x2 = (-f1 / f0 * x0 + x1) / (1 - f1 / f0)
    return x2


def _update_bracket(ab, fab, c, fc):
    """Update a bracket given (c, fc), return the discarded endpoints."""
    fa, fb = fab
    idx = (0 if np.sign(fa) * np.sign(fc) > 0 else 1)
    rx, rfx = ab[idx], fab[idx]
    fab[idx] = fc
    ab[idx] = c
    return rx, rfx


def _compute_divided_differences(xvals, fvals, N=None, full=True,
                                 forward=True):
    """Return a matrix of divided differences for the xvals, fvals pairs

    DD[i, j] = f[x_{i-j}, ..., x_i] for 0 <= j <= i

    If full is False, just return the main diagonal(or last row):
      f[a], f[a, b] and f[a, b, c].
    If forward is False, return f[c], f[b, c], f[a, b, c]."""
    if full:
        if forward:
            xvals = np.asarray(xvals)
        else:
            xvals = np.array(xvals)[::-1]
        M = len(xvals)
        N = M if N is None else min(N, M)
        DD = np.zeros([M, N])
        DD[:, 0] = fvals[:]
        for i in range(1, N):
            DD[i:, i] = (np.diff(DD[i - 1:, i - 1]) /
                         (xvals[i:] - xvals[:M - i]))
        return DD

    xvals = np.asarray(xvals)
    dd = np.array(fvals)
    row = np.array(fvals)
    idx2Use = (0 if forward else -1)
    dd[0] = fvals[idx2Use]
    for i in range(1, len(xvals)):
        denom = xvals[i:i + len(row) - 1] - xvals[:len(row) - 1]
        row = np.diff(row)[:] / denom
        dd[i] = row[idx2Use]
    return dd


def _interpolated_poly(xvals, fvals, x):
    """Compute p(x) for the polynomial passing through the specified locations.

    Use Neville's algorithm to compute p(x) where p is the minimal degree
    polynomial passing through the points xvals, fvals"""
    xvals = np.asarray(xvals)
    N = len(xvals)
    Q = np.zeros([N, N])
    D = np.zeros([N, N])
    Q[:, 0] = fvals[:]
    D[:, 0] = fvals[:]
    for k in range(1, N):
        alpha = D[k:, k - 1] - Q[k - 1:N - 1, k - 1]
        diffik = xvals[0:N - k] - xvals[k:N]
        Q[k:, k] = (xvals[k:] - x) / diffik * alpha
        D[k:, k] = (xvals[:N - k] - x) / diffik * alpha
    # Expect Q[-1, 1:] to be small relative to Q[-1, 0] as x approaches a root
    return np.sum(Q[-1, 1:]) + Q[-1, 0]


def _inverse_poly_zero(a, b, c, d, fa, fb, fc, fd):
    """Inverse cubic interpolation f-values -> x-values

    Given four points (fa, a), (fb, b), (fc, c), (fd, d) with
    fa, fb, fc, fd all distinct, find poly IP(y) through the 4 points
    and compute x=IP(0).
    """
    return _interpolated_poly([fa, fb, fc, fd], [a, b, c, d], 0)


def _newton_quadratic(ab, fab, d, fd, k):
    """Apply Newton-Raphson like steps, using divided differences to approximate f'

    ab is a real interval [a, b] containing a root,
    fab holds the real values of f(a), f(b)
    d is a real number outside [ab, b]
    k is the number of steps to apply
    """
    a, b = ab
    fa, fb = fab
    _, B, A = _compute_divided_differences([a, b, d], [fa, fb, fd],
                                           forward=True, full=False)

    # _P  is the quadratic polynomial through the 3 points
    def _P(x):
        # Horner evaluation of fa + B * (x - a) + A * (x - a) * (x - b)
        return (A * (x - b) + B) * (x - a) + fa

    if A == 0:
        r = a - fa / B
    else:
        r = (a if np.sign(A) * np.sign(fa) > 0 else b)
        # Apply k Newton-Raphson steps to _P(x), starting from x=r
        for i in range(k):
            r1 = r - _P(r) / (B + A * (2 * r - a - b))
            if not (ab[0] < r1 < ab[1]):
                if (ab[0] < r < ab[1]):
                    return r
                r = sum(ab) / 2.0
                break
            r = r1

    return r


class TOMS748Solver:
    """Solve f(x, *args) == 0 using Algorithm748 of Alefeld, Potro & Shi.
    """
    _MU = 0.5
    _K_MIN = 1
    _K_MAX = 100  # A very high value for real usage. Expect 1, 2, maybe 3.

    def __init__(self):
        self.f = None
        self.args = None
        self.function_calls = 0
        self.iterations = 0
        self.k = 2
        # ab=[a,b] is a global interval containing a root
        self.ab = [np.nan, np.nan]
        # fab is function values at a, b
        self.fab = [np.nan, np.nan]
        self.d = None
        self.fd = None
        self.e = None
        self.fe = None
        self.disp = False
        self.xtol = _xtol
        self.rtol = _rtol
        self.maxiter = _iter

    def configure(self, xtol, rtol, maxiter, disp, k):
        self.disp = disp
        self.xtol = xtol
        self.rtol = rtol
        self.maxiter = maxiter
        # Silently replace a low value of k with 1
        self.k = max(k, self._K_MIN)
        # Noisily replace a high value of k with self._K_MAX
        if self.k > self._K_MAX:
            msg = "toms748: Overriding k: ->%d" % self._K_MAX
            warnings.warn(msg, RuntimeWarning, stacklevel=3)
            self.k = self._K_MAX

    def _callf(self, x, error=True):
        """Call the user-supplied function, update book-keeping"""
        fx = self.f(x, *self.args)
        self.function_calls += 1
        if not np.isfinite(fx) and error:
            raise ValueError(f"Invalid function value: f({x:f}) -> {fx} ")
        return fx

    def get_result(self, x, flag=_ECONVERGED):
        r"""Package the result and statistics into a tuple."""
        return (x, self.function_calls, self.iterations, flag)

    def _update_bracket(self, c, fc):
        return _update_bracket(self.ab, self.fab, c, fc)

    def start(self, f, a, b, args=()):
        r"""Prepare for the iterations."""
        self.function_calls = 0
        self.iterations = 0

        self.f = f
        self.args = args
        self.ab[:] = [a, b]
        if not np.isfinite(a) or np.imag(a) != 0:
            raise ValueError("Invalid x value: %s " % (a))
        if not np.isfinite(b) or np.imag(b) != 0:
            raise ValueError("Invalid x value: %s " % (b))

        fa = self._callf(a)
        if not np.isfinite(fa) or np.imag(fa) != 0:
            raise ValueError(f"Invalid function value: f({a:f}) -> {fa} ")
        if fa == 0:
            return _ECONVERGED, a
        fb = self._callf(b)
        if not np.isfinite(fb) or np.imag(fb) != 0:
            raise ValueError(f"Invalid function value: f({b:f}) -> {fb} ")
        if fb == 0:
            return _ECONVERGED, b

        if np.sign(fb) * np.sign(fa) > 0:
            raise ValueError("f(a) and f(b) must have different signs, but "
                             f"f({a:e})={fa:e}, f({b:e})={fb:e} ")
        self.fab[:] = [fa, fb]

        return _EINPROGRESS, sum(self.ab) / 2.0

    def get_status(self):
        """Determine the current status."""
        a, b = self.ab[:2]
        if np.isclose(a, b, rtol=self.rtol, atol=self.xtol):
            return _ECONVERGED, sum(self.ab) / 2.0
        if self.iterations >= self.maxiter:
            return _ECONVERR, sum(self.ab) / 2.0
        return _EINPROGRESS, sum(self.ab) / 2.0

    def iterate(self):
        """Perform one step in the algorithm.

        Implements Algorithm 4.1(k=1) or 4.2(k=2) in [APS1995]
        """
        self.iterations += 1
        eps = np.finfo(float).eps
        d, fd, e, fe = self.d, self.fd, self.e, self.fe
        ab_width = self.ab[1] - self.ab[0]  # Need the start width below
        c = None

        for nsteps in range(2, self.k+2):
            # If the f-values are sufficiently separated, perform an inverse
            # polynomial interpolation step. Otherwise, nsteps repeats of
            # an approximate Newton-Raphson step.
            if _notclose(self.fab + [fd, fe], rtol=0, atol=32*eps):
                c0 = _inverse_poly_zero(self.ab[0], self.ab[1], d, e,
                                        self.fab[0], self.fab[1], fd, fe)
                if self.ab[0] < c0 < self.ab[1]:
                    c = c0
            if c is None:
                c = _newton_quadratic(self.ab, self.fab, d, fd, nsteps)

            fc = self._callf(c)
            if fc == 0:
                return _ECONVERGED, c

            # re-bracket
            e, fe = d, fd
            d, fd = self._update_bracket(c, fc)

        # u is the endpoint with the smallest f-value
        uix = (0 if np.abs(self.fab[0]) < np.abs(self.fab[1]) else 1)
        u, fu = self.ab[uix], self.fab[uix]

        _, A = _compute_divided_differences(self.ab, self.fab,
                                            forward=(uix == 0), full=False)
        c = u - 2 * fu / A
        if np.abs(c - u) > 0.5 * (self.ab[1] - self.ab[0]):
            c = sum(self.ab) / 2.0
        else:
            if np.isclose(c, u, rtol=eps, atol=0):
                # c didn't change (much).
                # Either because the f-values at the endpoints have vastly
                # differing magnitudes, or because the root is very close to
                # that endpoint
                frs = np.frexp(self.fab)[1]
                if frs[uix] < frs[1 - uix] - 50:  # Differ by more than 2**50
                    c = (31 * self.ab[uix] + self.ab[1 - uix]) / 32
                else:
                    # Make a bigger adjustment, about the
                    # size of the requested tolerance.
                    mm = (1 if uix == 0 else -1)
                    adj = mm * np.abs(c) * self.rtol + mm * self.xtol
                    c = u + adj
                if not self.ab[0] < c < self.ab[1]:
                    c = sum(self.ab) / 2.0

        fc = self._callf(c)
        if fc == 0:
            return _ECONVERGED, c

        e, fe = d, fd
        d, fd = self._update_bracket(c, fc)

        # If the width of the new interval did not decrease enough, bisect
        if self.ab[1] - self.ab[0] > self._MU * ab_width:
            e, fe = d, fd
            z = sum(self.ab) / 2.0
            fz = self._callf(z)
            if fz == 0:
                return _ECONVERGED, z
            d, fd = self._update_bracket(z, fz)

        # Record d and e for next iteration
        self.d, self.fd = d, fd
        self.e, self.fe = e, fe

        status, xn = self.get_status()
        return status, xn

    def solve(self, f, a, b, args=(),
              xtol=_xtol, rtol=_rtol, k=2, maxiter=_iter, disp=True):
        r"""Solve f(x) = 0 given an interval containing a root."""
        self.configure(xtol=xtol, rtol=rtol, maxiter=maxiter, disp=disp, k=k)
        status, xn = self.start(f, a, b, args)
        if status == _ECONVERGED:
            return self.get_result(xn)

        # The first step only has two x-values.
        c = _secant(self.ab, self.fab)
        if not self.ab[0] < c < self.ab[1]:
            c = sum(self.ab) / 2.0
        fc = self._callf(c)
        if fc == 0:
            return self.get_result(c)

        self.d, self.fd = self._update_bracket(c, fc)
        self.e, self.fe = None, None
        self.iterations += 1

        while True:
            status, xn = self.iterate()
            if status == _ECONVERGED:
                return self.get_result(xn)
            if status == _ECONVERR:
                fmt = "Failed to converge after %d iterations, bracket is %s"
                if disp:
                    msg = fmt % (self.iterations + 1, self.ab)
                    raise RuntimeError(msg)
                return self.get_result(xn, _ECONVERR)


def toms748(f, a, b, args=(), k=1,
            xtol=_xtol, rtol=_rtol, maxiter=_iter,
            full_output=False, disp=True):
    """
    Find a root using TOMS Algorithm 748 method.

    Implements the Algorithm 748 method of Alefeld, Potro and Shi to find a
    root of the function `f` on the interval `[a , b]`, where `f(a)` and
    `f(b)` must have opposite signs.

    It uses a mixture of inverse cubic interpolation and
    "Newton-quadratic" steps. [APS1995].

    Parameters
    ----------
    f : function
        Python function returning a scalar. The function :math:`f`
        must be continuous, and :math:`f(a)` and :math:`f(b)`
        have opposite signs.
    a : scalar,
        lower boundary of the search interval
    b : scalar,
        upper boundary of the search interval
    args : tuple, optional
        containing extra arguments for the function `f`.
        `f` is called by ``f(x, *args)``.
    k : int, optional
        The number of Newton quadratic steps to perform each
        iteration. ``k>=1``.
    xtol : scalar, optional
        The computed root ``x0`` will satisfy ``np.allclose(x, x0,
        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
        parameter must be positive.
    rtol : scalar, optional
        The computed root ``x0`` will satisfy ``np.allclose(x, x0,
        atol=xtol, rtol=rtol)``, where ``x`` is the exact root.
    maxiter : int, optional
        If convergence is not achieved in `maxiter` iterations, an error is
        raised. Must be >= 0.
    full_output : bool, optional
        If `full_output` is False, the root is returned. If `full_output` is
        True, the return value is ``(x, r)``, where `x` is the root, and `r` is
        a `RootResults` object.
    disp : bool, optional
        If True, raise RuntimeError if the algorithm didn't converge.
        Otherwise, the convergence status is recorded in the `RootResults`
        return object.

    Returns
    -------
    root : float
        Approximate root of `f`
    r : `RootResults` (present if ``full_output = True``)
        Object containing information about the convergence. In particular,
        ``r.converged`` is True if the routine converged.

    See Also
    --------
    brentq, brenth, ridder, bisect, newton
    fsolve : find roots in N dimensions.

    Notes
    -----
    `f` must be continuous.
    Algorithm 748 with ``k=2`` is asymptotically the most efficient
    algorithm known for finding roots of a four times continuously
    differentiable function.
    In contrast with Brent's algorithm, which may only decrease the length of
    the enclosing bracket on the last step, Algorithm 748 decreases it each
    iteration with the same asymptotic efficiency as it finds the root.

    For easy statement of efficiency indices, assume that `f` has 4
    continuouous deriviatives.
    For ``k=1``, the convergence order is at least 2.7, and with about
    asymptotically 2 function evaluations per iteration, the efficiency
    index is approximately 1.65.
    For ``k=2``, the order is about 4.6 with asymptotically 3 function
    evaluations per iteration, and the efficiency index 1.66.
    For higher values of `k`, the efficiency index approaches
    the kth root of ``(3k-2)``, hence ``k=1`` or ``k=2`` are
    usually appropriate.

    References
    ----------
    .. [APS1995]
       Alefeld, G. E. and Potra, F. A. and Shi, Yixun,
       *Algorithm 748: Enclosing Zeros of Continuous Functions*,
       ACM Trans. Math. Softw. Volume 221(1995)
       doi = {10.1145/210089.210111}

    Examples
    --------
    >>> def f(x):
    ...     return (x**3 - 1)  # only one real root at x = 1

    >>> from scipy import optimize
    >>> root, results = optimize.toms748(f, 0, 2, full_output=True)
    >>> root
    1.0
    >>> results
          converged: True
               flag: converged
     function_calls: 11
         iterations: 5
               root: 1.0
             method: toms748
    """
    if xtol <= 0:
        raise ValueError("xtol too small (%g <= 0)" % xtol)
    if rtol < _rtol / 4:
        raise ValueError(f"rtol too small ({rtol:g} < {_rtol/4:g})")
    maxiter = operator.index(maxiter)
    if maxiter < 1:
        raise ValueError("maxiter must be greater than 0")
    if not np.isfinite(a):
        raise ValueError("a is not finite %s" % a)
    if not np.isfinite(b):
        raise ValueError("b is not finite %s" % b)
    if a >= b:
        raise ValueError(f"a and b are not an interval [{a}, {b}]")
    if not k >= 1:
        raise ValueError("k too small (%s < 1)" % k)

    if not isinstance(args, tuple):
        args = (args,)
    f = _wrap_nan_raise(f)
    solver = TOMS748Solver()
    result = solver.solve(f, a, b, args=args, k=k, xtol=xtol, rtol=rtol,
                          maxiter=maxiter, disp=disp)
    x, function_calls, iterations, flag = result
    return _results_select(full_output, (x, function_calls, iterations, flag),
                           "toms748")


def _bracket_root_iv(func, a, b, min, max, factor, args, maxiter):

    if not callable(func):
        raise ValueError('`func` must be callable.')

    if not np.iterable(args):
        args = (args,)

    a = np.asarray(a)[()]
    if not np.issubdtype(a.dtype, np.number) or np.iscomplex(a).any():
        raise ValueError('`a` must be numeric and real.')

    b = a + 1 if b is None else b
    min = -np.inf if min is None else min
    max = np.inf if max is None else max
    factor = 2. if factor is None else factor
    a, b, min, max, factor = np.broadcast_arrays(a, b, min, max, factor)

    if not np.issubdtype(b.dtype, np.number) or np.iscomplex(b).any():
        raise ValueError('`b` must be numeric and real.')

    if not np.issubdtype(min.dtype, np.number) or np.iscomplex(min).any():
        raise ValueError('`min` must be numeric and real.')

    if not np.issubdtype(max.dtype, np.number) or np.iscomplex(max).any():
        raise ValueError('`max` must be numeric and real.')

    if not np.issubdtype(factor.dtype, np.number) or np.iscomplex(factor).any():
        raise ValueError('`factor` must be numeric and real.')
    if not np.all(factor > 1):
        raise ValueError('All elements of `factor` must be greater than 1.')

    maxiter = np.asarray(maxiter)
    message = '`maxiter` must be a non-negative integer.'
    if (not np.issubdtype(maxiter.dtype, np.number) or maxiter.shape != tuple()
            or np.iscomplex(maxiter)):
        raise ValueError(message)
    maxiter_int = int(maxiter[()])
    if not maxiter == maxiter_int or maxiter < 0:
        raise ValueError(message)

    if not np.all((min <= a) & (a < b) & (b <= max)):
        raise ValueError('`min <= a < b <= max` must be True (elementwise).')

    return func, a, b, min, max, factor, args, maxiter


def _bracket_root(func, a, b=None, *, min=None, max=None, factor=None,
                  args=(), maxiter=1000):
    """Bracket the root of a monotonic scalar function of one variable

    This function works elementwise when `a`, `b`, `min`, `max`, `factor`, and
    the elements of `args` are broadcastable arrays.

    Parameters
    ----------
    func : callable
        The function for which the root is to be bracketed.
        The signature must be::

            func(x: ndarray, *args) -> ndarray

        where each element of ``x`` is a finite real and ``args`` is a tuple,
        which may contain an arbitrary number of arrays that are broadcastable
        with `x`. ``func`` must be an elementwise function: each element
        ``func(x)[i]`` must equal ``func(x[i])`` for all indices ``i``.
    a, b : float array_like
        Starting guess of bracket, which need not contain a root. If `b` is
        not provided, ``b = a + 1``. Must be broadcastable with one another.
    min, max : float array_like, optional
        Minimum and maximum allowable endpoints of the bracket, inclusive. Must
        be broadcastable with `a` and `b`.
    factor : float array_like, default: 2
        The factor used to grow the bracket. See notes for details.
    args : tuple, optional
        Additional positional arguments to be passed to `func`.  Must be arrays
        broadcastable with `a`, `b`, `min`, and `max`. If the callable to be
        bracketed requires arguments that are not broadcastable with these
        arrays, wrap that callable with `func` such that `func` accepts
        only `x` and broadcastable arrays.
    maxiter : int, optional
        The maximum number of iterations of the algorithm to perform.

    Returns
    -------
    res : OptimizeResult
        An instance of `scipy.optimize.OptimizeResult` with the following
        attributes. The descriptions are written as though the values will be
        scalars; however, if `func` returns an array, the outputs will be
        arrays of the same shape.

        xl, xr : float
            The lower and upper ends of the bracket, if the algorithm
            terminated successfully.
        fl, fr : float
            The function value at the lower and upper ends of the bracket.
        nfev : int
            The number of function evaluations required to find the bracket.
            This is distinct from the number of times `func` is *called*
            because the function may evaluated at multiple points in a single
            call.
        nit : int
            The number of iterations of the algorithm that were performed.
        status : int
            An integer representing the exit status of the algorithm.

            - ``0`` : The algorithm produced a valid bracket.
            - ``-1`` : The bracket expanded to the allowable limits without finding a bracket.
            - ``-2`` : The maximum number of iterations was reached.
            - ``-3`` : A non-finite value was encountered.
            - ``-4`` : Iteration was terminated by `callback`.
            - ``1`` : The algorithm is proceeding normally (in `callback` only).
            - ``2`` : A bracket was found in the opposite search direction (in `callback` only).

        success : bool
            ``True`` when the algorithm terminated successfully (status ``0``).

    Notes
    -----
    This function generalizes an algorithm found in pieces throughout
    `scipy.stats`. The strategy is to iteratively grow the bracket `(l, r)`
     until ``func(l) < 0 < func(r)``. The bracket grows to the left as follows.

    - If `min` is not provided, the distance between `b` and `l` is iteratively
      increased by `factor`.
    - If `min` is provided, the distance between `min` and `l` is iteratively
      decreased by `factor`. Note that this also *increases* the bracket size.

    Growth of the bracket to the right is analogous.

    Growth of the bracket in one direction stops when the endpoint is no longer
    finite, the function value at the endpoint is no longer finite, or the
    endpoint reaches its limiting value (`min` or `max`). Iteration terminates
    when the bracket stops growing in both directions, the bracket surrounds
    the root, or a root is found (accidentally).

    If two brackets are found - that is, a bracket is found on both sides in
    the same iteration, the smaller of the two is returned.
    If roots of the function are found, both `l` and `r` are set to the
    leftmost root.

    """  # noqa: E501
    # Todo:
    # - find bracket with sign change in specified direction
    # - Add tolerance
    # - allow factor < 1?

    callback = None  # works; I just don't want to test it
    temp = _bracket_root_iv(func, a, b, min, max, factor, args, maxiter)
    func, a, b, min, max, factor, args, maxiter = temp

    xs = (a, b)
    temp = _scalar_optimization_initialize(func, xs, args)
    xs, fs, args, shape, dtype = temp  # line split for PEP8

    # The approach is to treat the left and right searches as though they were
    # (almost) totally independent one-sided bracket searches. (The interaction
    # is considered when checking for termination and preparing the result
    # object.)
    # `x` is the "moving" end of the bracket
    x = np.concatenate(xs)
    f = np.concatenate(fs)
    n = len(x) // 2

    # `x_last` is the previous location of the moving end of the bracket. If
    # the signs of `f` and `f_last` are different, `x` and `x_last` form a
    # bracket.
    x_last = np.concatenate((x[n:], x[:n]))
    f_last = np.concatenate((f[n:], f[:n]))
    # `x0` is the "fixed" end of the bracket.
    x0 = x_last
    # We don't need to retain the corresponding function value, since the
    # fixed end of the bracket is only needed to compute the new value of the
    # moving end; it is never returned.

    min = np.broadcast_to(min, shape).astype(dtype, copy=False).ravel()
    max = np.broadcast_to(max, shape).astype(dtype, copy=False).ravel()
    limit = np.concatenate((min, max))

    factor = np.broadcast_to(factor, shape).astype(dtype, copy=False).ravel()
    factor = np.concatenate((factor, factor))

    active = np.arange(2*n)
    args = [np.concatenate((arg, arg)) for arg in args]

    # This is needed due to inner workings of `_scalar_optimization_loop`.
    # We're abusing it a tiny bit.
    shape = shape + (2,)

    # `d` is for "distance".
    # For searches without a limit, the distance between the fixed end of the
    # bracket `x0` and the moving end `x` will grow by `factor` each iteration.
    # For searches with a limit, the distance between the `limit` and moving
    # end of the bracket `x` will shrink by `factor` each iteration.
    i = np.isinf(limit)
    ni = ~i
    d = np.zeros_like(x)
    d[i] = x[i] - x0[i]
    d[ni] = limit[ni] - x[ni]

    status = np.full_like(x, _EINPROGRESS, dtype=int)  # in progress
    nit, nfev = 0, 1  # one function evaluation per side performed above

    work = OptimizeResult(x=x, x0=x0, f=f, limit=limit, factor=factor,
                          active=active, d=d, x_last=x_last, f_last=f_last,
                          nit=nit, nfev=nfev, status=status, args=args,
                          xl=None, xr=None, fl=None, fr=None, n=n)
    res_work_pairs = [('status', 'status'), ('xl', 'xl'), ('xr', 'xr'),
                      ('nit', 'nit'), ('nfev', 'nfev'), ('fl', 'fl'),
                      ('fr', 'fr'), ('x', 'x'), ('f', 'f'),
                      ('x_last', 'x_last'), ('f_last', 'f_last')]

    def pre_func_eval(work):
        # Initialize moving end of bracket
        x = np.zeros_like(work.x)

        # Unlimited brackets grow by `factor` by increasing distance from fixed
        # end to moving end.
        i = np.isinf(work.limit)  # indices of unlimited brackets
        work.d[i] *= work.factor[i]
        x[i] = work.x0[i] + work.d[i]

        # Limited brackets grow by decreasing the distance from the limit to
        # the moving end.
        ni = ~i  # indices of limited brackets
        work.d[ni] /= work.factor[ni]
        x[ni] = work.limit[ni] - work.d[ni]

        return x

    def post_func_eval(x, f, work):
        # Keep track of the previous location of the moving end so that we can
        # return a narrower bracket. (The alternative is to remember the
        # original fixed end, but then the bracket would be wider than needed.)
        work.x_last = work.x
        work.f_last = work.f
        work.x = x
        work.f = f

    def check_termination(work):
        stop = np.zeros_like(work.x, dtype=bool)

        # Condition 1: a valid bracket (or the root itself) has been found
        sf = np.sign(work.f)
        sf_last = np.sign(work.f_last)
        i = (sf_last == -sf) | (sf_last == 0) | (sf == 0)
        work.status[i] = _ECONVERGED
        stop[i] = True

        # Condition 2: the other side's search found a valid bracket.
        # (If we just found a bracket with the rightward search, we can stop
        #  the leftward search, and vice-versa.)
        # To do this, we need to set the status of the other side's search;
        # this is tricky because `work.status` contains only the *active*
        # elements, so we don't immediately know the index of the element we
        # need to set - or even if it's still there. (That search may have
        # terminated already, e.g. by reaching its `limit`.)
        # To facilitate this, `work.active` contains a unit integer index of
        # each search. Index `k` (`k < n)` and `k + n` correspond with a
        # leftward and rightward search, respectively. Elements are removed
        # from `work.active` just as they are removed from `work.status`, so
        # we use `work.active` to help find the right location in
        # `work.status`.
        # Get the integer indices of the elements that can also stop
        also_stop = (work.active[i] + work.n) % (2*work.n)
        # Check whether they are still active.
        # To start, we need to find out where in `work.active` they would
        # appear if they are indeed there.
        j = np.searchsorted(work.active, also_stop)
        # If the location exceeds the length of the `work.active`, they are
        # not there.
        j = j[j < len(work.active)]
        # Check whether they are still there.
        j = j[also_stop == work.active[j]]
        # Now convert these to boolean indices to use with `work.status`.
        i = np.zeros_like(stop)
        i[j] = True  # boolean indices of elements that can also stop
        i = i & ~stop
        work.status[i] = _ESTOPONESIDE
        stop[i] = True

        # Condition 3: moving end of bracket reaches limit
        i = (work.x == work.limit) & ~stop
        work.status[i] = _ELIMITS
        stop[i] = True

        # Condition 4: non-finite value encountered
        i = ~(np.isfinite(work.x) & np.isfinite(work.f)) & ~stop
        work.status[i] = _EVALUEERR
        stop[i] = True

        return stop

    def post_termination_check(work):
        pass

    def customize_result(res, shape):
        n = len(res['x']) // 2

        # Because we treat the two one-sided searches as though they were
        # independent, what we keep track of in `work` and what we want to
        # return in `res` look quite different. Combine the results from the
        # two one-sided searches before reporting the results to the user.
        # - "a" refers to the leftward search (the moving end started at `a`)
        # - "b" refers to the rightward search (the moving end started at `b`)
        # - "l" refers to the left end of the bracket (closer to -oo)
        # - "r" refers to the right end of the bracket (closer to +oo)
        xal = res['x'][:n]
        xar = res['x_last'][:n]
        xbl = res['x_last'][n:]
        xbr = res['x'][n:]

        fal = res['f'][:n]
        far = res['f_last'][:n]
        fbl = res['f_last'][n:]
        fbr = res['f'][n:]

        # Initialize the brackets and corresponding function values to return
        # to the user. Brackets may not be valid (e.g. there is no root,
        # there weren't enough iterations, NaN encountered), but we still need
        # to return something. One option would be all NaNs, but what I've
        # chosen here is the left- and right-most points at which the function
        # has been evaluated. This gives the user some information about what
        # interval of the real line has been searched and shows that there is
        # no sign change between the two ends.
        xl = xal.copy()
        fl = fal.copy()
        xr = xbr.copy()
        fr = fbr.copy()

        # `status` indicates whether the bracket is valid or not. If so,
        # we want to adjust the bracket we return to be the narrowest possible
        # given the points at which we evaluated the function.
        # For example if bracket "a" is valid and smaller than bracket "b" OR
        # if bracket "a" is valid and bracket "b" is not valid, we want to
        # return bracket "a" (and vice versa).
        sa = res['status'][:n]
        sb = res['status'][n:]

        da = xar - xal
        db = xbr - xbl

        i1 = ((da <= db) & (sa == 0)) | ((sa == 0) & (sb != 0))
        i2 = ((db <= da) & (sb == 0)) | ((sb == 0) & (sa != 0))

        xr[i1] = xar[i1]
        fr[i1] = far[i1]
        xl[i2] = xbl[i2]
        fl[i2] = fbl[i2]

        # Finish assembling the result object
        res['xl'] = xl
        res['xr'] = xr
        res['fl'] = fl
        res['fr'] = fr

        res['nit'] = np.maximum(res['nit'][:n], res['nit'][n:])
        res['nfev'] = res['nfev'][:n] + res['nfev'][n:]
        # If the status on one side is zero, the status is zero. In any case,
        # report the status from one side only.
        res['status'] = np.choose(sa == 0, (sb, sa))
        res['success'] = (res['status'] == 0)

        del res['x']
        del res['f']
        del res['x_last']
        del res['f_last']

        return shape[:-1]

    return _scalar_optimization_loop(work, callback, shape,
                                     maxiter, func, args, dtype,
                                     pre_func_eval, post_func_eval,
                                     check_termination, post_termination_check,
                                     customize_result, res_work_pairs)


def _chandrupatla(func, a, b, *, args=(), xatol=_xtol, xrtol=_rtol,
                  fatol=None, frtol=0, maxiter=_iter, callback=None):
    """Find the root of an elementwise function using Chandrupatla's algorithm.

    For each element of the output of `func`, `chandrupatla` seeks the scalar
    root that makes the element 0. This function allows for `a`, `b`, and the
    output of `func` to be of any broadcastable shapes.

    Parameters
    ----------
    func : callable
        The function whose root is desired. The signature must be::

            func(x: ndarray, *args) -> ndarray

         where each element of ``x`` is a finite real and ``args`` is a tuple,
         which may contain an arbitrary number of components of any type(s).
         ``func`` must be an elementwise function: each element ``func(x)[i]``
         must equal ``func(x[i])`` for all indices ``i``. `_chandrupatla`
         seeks an array ``x`` such that ``func(x)`` is an array of zeros.
    a, b : array_like
        The lower and upper bounds of the root of the function. Must be
        broadcastable with one another.
    args : tuple, optional
        Additional positional arguments to be passed to `func`.
    xatol, xrtol, fatol, frtol : float, optional
        Absolute and relative tolerances on the root and function value.
        See Notes for details.
    maxiter : int, optional
        The maximum number of iterations of the algorithm to perform.
    callback : callable, optional
        An optional user-supplied function to be called before the first
        iteration and after each iteration.
        Called as ``callback(res)``, where ``res`` is an ``OptimizeResult``
        similar to that returned by `_chandrupatla` (but containing the current
        iterate's values of all variables). If `callback` raises a
        ``StopIteration``, the algorithm will terminate immediately and
        `_chandrupatla` will return a result.

    Returns
    -------
    res : OptimizeResult
        An instance of `scipy.optimize.OptimizeResult` with the following
        attributes. The descriptions are written as though the values will be
        scalars; however, if `func` returns an array, the outputs will be
        arrays of the same shape.

        x : float
            The root of the function, if the algorithm terminated successfully.
        nfev : int
            The number of times the function was called to find the root.
        nit : int
            The number of iterations of Chandrupatla's algorithm performed.
        status : int
            An integer representing the exit status of the algorithm.
            ``0`` : The algorithm converged to the specified tolerances.
            ``-1`` : The algorithm encountered an invalid bracket.
            ``-2`` : The maximum number of iterations was reached.
            ``-3`` : A non-finite value was encountered.
            ``-4`` : Iteration was terminated by `callback`.
            ``1`` : The algorithm is proceeding normally (in `callback` only).
        success : bool
            ``True`` when the algorithm terminated successfully (status ``0``).
        fun : float
            The value of `func` evaluated at `x`.
        xl, xr : float
            The lower and upper ends of the bracket.
        fl, fr : float
            The function value at the lower and upper ends of the bracket.

    Notes
    -----
    Implemented based on Chandrupatla's original paper [1]_.

    If ``xl`` and ``xr`` are the left and right ends of the bracket,
    ``xmin = xl if abs(func(xl)) <= abs(func(xr)) else xr``,
    and ``fmin0 = min(func(a), func(b))``, then the algorithm is considered to
    have converged when ``abs(xr - xl) < xatol + abs(xmin) * xrtol`` or
    ``fun(xmin) <= fatol + abs(fmin0) * frtol``. This is equivalent to the
    termination condition described in [1]_ with ``xrtol = 4e-10``,
    ``xatol = 1e-5``, and ``fatol = frtol = 0``. The default values are
    ``xatol = 2e-12``, ``xrtol = 4 * np.finfo(float).eps``, ``frtol = 0``,
    and ``fatol`` is the smallest normal number of the ``dtype`` returned
    by ``func``.

    References
    ----------

    .. [1] Chandrupatla, Tirupathi R.
        "A new hybrid quadratic/bisection algorithm for finding the zero of a
        nonlinear function without using derivatives".
        Advances in Engineering Software, 28(3), 145-149.
        https://doi.org/10.1016/s0965-9978(96)00051-8

    See Also
    --------
    brentq, brenth, ridder, bisect, newton

    Examples
    --------
    >>> from scipy import optimize
    >>> def f(x, c):
    ...     return x**3 - 2*x - c
    >>> c = 5
    >>> res = optimize._zeros_py._chandrupatla(f, 0, 3, args=(c,))
    >>> res.x
    2.0945514818937463

    >>> c = [3, 4, 5]
    >>> res = optimize._zeros_py._chandrupatla(f, 0, 3, args=(c,))
    >>> res.x
    array([1.8932892 , 2.        , 2.09455148])

    """
    res = _chandrupatla_iv(func, args, xatol, xrtol,
                           fatol, frtol, maxiter, callback)
    func, args, xatol, xrtol, fatol, frtol, maxiter, callback = res

    # Initialization
    xs, fs, args, shape, dtype = _scalar_optimization_initialize(func, (a, b),
                                                                 args)
    x1, x2 = xs
    f1, f2 = fs
    status = np.full_like(x1, _EINPROGRESS, dtype=int)  # in progress
    nit, nfev = 0, 2  # two function evaluations performed above
    xatol = _xtol if xatol is None else xatol
    xrtol = _rtol if xrtol is None else xrtol
    fatol = np.finfo(dtype).tiny if fatol is None else fatol
    frtol = frtol * np.minimum(np.abs(f1), np.abs(f2))
    work = OptimizeResult(x1=x1, f1=f1, x2=x2, f2=f2, x3=None, f3=None, t=0.5,
                          xatol=xatol, xrtol=xrtol, fatol=fatol, frtol=frtol,
                          nit=nit, nfev=nfev, status=status)
    res_work_pairs = [('status', 'status'), ('x', 'xmin'), ('fun', 'fmin'),
                      ('nit', 'nit'), ('nfev', 'nfev'), ('xl', 'x1'),
                      ('fl', 'f1'), ('xr', 'x2'), ('fr', 'f2')]

    def pre_func_eval(work):
        # [1] Figure 1 (first box)
        x = work.x1 + work.t * (work.x2 - work.x1)
        return x

    def post_func_eval(x, f, work):
        # [1] Figure 1 (first diamond and boxes)
        # Note: y/n are reversed in figure; compare to BASIC in appendix
        work.x3, work.f3 = work.x2.copy(), work.f2.copy()
        j = np.sign(f) == np.sign(work.f1)
        nj = ~j
        work.x3[j], work.f3[j] = work.x1[j], work.f1[j]
        work.x2[nj], work.f2[nj] = work.x1[nj], work.f1[nj]
        work.x1, work.f1 = x, f

    def check_termination(work):
        # [1] Figure 1 (second diamond)
        # Check for all terminal conditions and record statuses.

        # See [1] Section 4 (first two sentences)
        i = np.abs(work.f1) < np.abs(work.f2)
        work.xmin = np.choose(i, (work.x2, work.x1))
        work.fmin = np.choose(i, (work.f2, work.f1))
        stop = np.zeros_like(work.x1, dtype=bool)  # termination condition met

        # This is the convergence criterion used in bisect. Chandrupatla's
        # criterion is equivalent to this except with a factor of 4 on `xrtol`.
        work.dx = abs(work.x2 - work.x1)
        work.tol = abs(work.xmin) * work.xrtol + work.xatol
        i = work.dx < work.tol
        # Modify in place to incorporate tolerance on function value. Note that
        # `frtol` has been redefined as `frtol = frtol * np.minimum(f1, f2)`,
        # where `f1` and `f2` are the function evaluated at the original ends of
        # the bracket.
        i |= np.abs(work.fmin) <= work.fatol + work.frtol
        work.status[i] = _ECONVERGED
        stop[i] = True

        i = (np.sign(work.f1) == np.sign(work.f2)) & ~stop
        work.xmin[i], work.fmin[i], work.status[i] = np.nan, np.nan, _ESIGNERR
        stop[i] = True

        i = ~((np.isfinite(work.x1) & np.isfinite(work.x2)
               & np.isfinite(work.f1) & np.isfinite(work.f2)) | stop)
        work.xmin[i], work.fmin[i], work.status[i] = np.nan, np.nan, _EVALUEERR
        stop[i] = True

        return stop

    def post_termination_check(work):
        # [1] Figure 1 (third diamond and boxes / Equation 1)
        xi1 = (work.x1 - work.x2) / (work.x3 - work.x2)
        phi1 = (work.f1 - work.f2) / (work.f3 - work.f2)
        alpha = (work.x3 - work.x1) / (work.x2 - work.x1)
        j = ((1 - np.sqrt(1 - xi1)) < phi1) & (phi1 < np.sqrt(xi1))

        f1j, f2j, f3j, alphaj = work.f1[j], work.f2[j], work.f3[j], alpha[j]
        t = np.full_like(alpha, 0.5)
        t[j] = (f1j / (f1j - f2j) * f3j / (f3j - f2j)
                - alphaj * f1j / (f3j - f1j) * f2j / (f2j - f3j))

        # [1] Figure 1 (last box; see also BASIC in appendix with comment
        # "Adjust T Away from the Interval Boundary")
        tl = 0.5 * work.tol / work.dx
        work.t = np.clip(t, tl, 1 - tl)

    def customize_result(res, shape):
        xl, xr, fl, fr = res['xl'], res['xr'], res['fl'], res['fr']
        i = res['xl'] < res['xr']
        res['xl'] = np.choose(i, (xr, xl))
        res['xr'] = np.choose(i, (xl, xr))
        res['fl'] = np.choose(i, (fr, fl))
        res['fr'] = np.choose(i, (fl, fr))
        return shape

    return _scalar_optimization_loop(work, callback, shape,
                                     maxiter, func, args, dtype,
                                     pre_func_eval, post_func_eval,
                                     check_termination, post_termination_check,
                                     customize_result, res_work_pairs)


def _scalar_optimization_loop(work, callback, shape, maxiter,
                              func, args, dtype, pre_func_eval, post_func_eval,
                              check_termination, post_termination_check,
                              customize_result, res_work_pairs):
    """Main loop of a vectorized scalar optimization algorithm

    Parameters
    ----------
    work : OptimizeResult
        All variables that need to be retained between iterations. Must
        contain attributes `nit`, `nfev`, and `success`
    callback : callable
        User-specified callback function
    shape : tuple of ints
        The shape of all output arrays
    maxiter :
        Maximum number of iterations of the algorithm
    func : callable
        The user-specified callable that is being optimized or solved
    args : tuple
        Additional positional arguments to be passed to `func`.
    dtype : NumPy dtype
        The common dtype of all abscissae and function values
    pre_func_eval : callable
        A function that accepts `work` and returns `x`, the active elements
        of `x` at which `func` will be evaluated. May modify attributes
        of `work` with any algorithmic steps that need to happen
         at the beginning of an iteration, before `func` is evaluated,
    post_func_eval : callable
        A function that accepts `x`, `func(x)`, and `work`. May modify
        attributes of `work` with any algorithmic steps that need to happen
         in the middle of an iteration, after `func` is evaluated but before
         the termination check.
    check_termination : callable
        A function that accepts `work` and returns `stop`, a boolean array
        indicating which of the active elements have met a termination
        condition.
    post_termination_check : callable
        A function that accepts `work`. May modify `work` with any algorithmic
        steps that need to happen after the termination check and before the
        end of the iteration.
    customize_result : callable
        A function that accepts `res` and `shape` and returns `shape`. May
        modify `res` (in-place) according to preferences (e.g. rearrange
        elements between attributes) and modify `shape` if needed.
    res_work_pairs : list of (str, str)
        Identifies correspondence between attributes of `res` and attributes
        of `work`; i.e., attributes of active elements of `work` will be
        copied to the appropriate indices of `res` when appropriate. The order
        determines the order in which OptimizeResult attributes will be
        pretty-printed.

    Returns
    -------
    res : OptimizeResult
        The final result object

    Notes
    -----
    Besides providing structure, this framework provides several important
    services for a vectorized optimization algorithm.

    - It handles common tasks involving iteration count, function evaluation
      count, a user-specified callback, and associated termination conditions.
    - It compresses the attributes of `work` to eliminate unnecessary
      computation on elements that have already converged.

    """
    cb_terminate = False

    # Initialize the result object and active element index array
    n_elements = int(np.prod(shape))
    active = np.arange(n_elements)  # in-progress element indices
    res_dict = {i: np.zeros(n_elements, dtype=dtype) for i, j in res_work_pairs}
    res_dict['success'] = np.zeros(n_elements, dtype=bool)
    res_dict['status'] = np.full(n_elements, _EINPROGRESS)
    res_dict['nit'] = np.zeros(n_elements, dtype=int)
    res_dict['nfev'] = np.zeros(n_elements, dtype=int)
    res = OptimizeResult(res_dict)
    work.args = args

    active = _scalar_optimization_check_termination(
        work, res, res_work_pairs, active, check_termination)

    if callback is not None:
        temp = _scalar_optimization_prepare_result(
            work, res, res_work_pairs, active, shape, customize_result)
        if _call_callback_maybe_halt(callback, temp):
            cb_terminate = True

    while work.nit < maxiter and active.size and not cb_terminate and n_elements:
        x = pre_func_eval(work)

        if work.args and work.args[0].ndim != x.ndim:
            # `x` always starts as 1D. If the SciPy function that uses
            # _scalar_optimization_loop added dimensions to `x`, we need to
            # add them to the elements of `args`.
            dims = np.arange(x.ndim, dtype=np.int64)
            work.args = [np.expand_dims(arg, tuple(dims[arg.ndim:]))
                         for arg in work.args]

        f = func(x, *work.args)
        f = np.asarray(f, dtype=dtype)
        work.nfev += 1 if x.ndim == 1 else x.shape[-1]

        post_func_eval(x, f, work)

        work.nit += 1
        active = _scalar_optimization_check_termination(
            work, res, res_work_pairs, active, check_termination)

        if callback is not None:
            temp = _scalar_optimization_prepare_result(
                work, res, res_work_pairs, active, shape, customize_result)
            if _call_callback_maybe_halt(callback, temp):
                cb_terminate = True
                break
        if active.size == 0:
            break

        post_termination_check(work)

    work.status[:] = _ECALLBACK if cb_terminate else _ECONVERR
    return _scalar_optimization_prepare_result(
        work, res, res_work_pairs, active, shape, customize_result)


def _chandrupatla_iv(func, args, xatol, xrtol,
                     fatol, frtol, maxiter, callback):
    # Input validation for `_chandrupatla`

    if not callable(func):
        raise ValueError('`func` must be callable.')

    if not np.iterable(args):
        args = (args,)

    tols = np.asarray([xatol if xatol is not None else 1,
                       xrtol if xrtol is not None else 1,
                       fatol if fatol is not None else 1,
                       frtol if frtol is not None else 1])
    if (not np.issubdtype(tols.dtype, np.number) or np.any(tols < 0)
            or np.any(np.isnan(tols)) or tols.shape != (4,)):
        raise ValueError('Tolerances must be non-negative scalars.')

    maxiter_int = int(maxiter)
    if maxiter != maxiter_int or maxiter < 0:
        raise ValueError('`maxiter` must be a non-negative integer.')

    if callback is not None and not callable(callback):
        raise ValueError('`callback` must be callable.')

    return func, args, xatol, xrtol, fatol, frtol, maxiter, callback


def _scalar_optimization_initialize(func, xs, args, complex_ok=False):
    """Initialize abscissa, function, and args arrays for elementwise function

    Parameters
    ----------
    func : callable
        An elementwise function with signature

            func(x: ndarray, *args) -> ndarray

        where each element of ``x`` is a finite real and ``args`` is a tuple,
        which may contain an arbitrary number of arrays that are broadcastable
        with ``x``.
    xs : tuple of arrays
        Finite real abscissa arrays. Must be broadcastable.
    args : tuple, optional
        Additional positional arguments to be passed to `func`.

    Returns
    -------
    xs, fs, args : tuple of arrays
        Broadcasted, writeable, 1D abscissa and function value arrays (or
        NumPy floats, if appropriate). The dtypes of the `xs` and `fs` are
        `xfat`; the dtype of the `args` are unchanged.
    shape : tuple of ints
        Original shape of broadcasted arrays.
    xfat : NumPy dtype
        Result dtype of abscissae, function values, and args determined using
        `np.result_type`, except integer types are promoted to `np.float64`.

    Raises
    ------
    ValueError
        If the result dtype is not that of a real scalar

    Notes
    -----
    Useful for initializing the input of SciPy functions that accept
    an elementwise callable, abscissae, and arguments; e.g.
    `scipy.optimize._chandrupatla`.
    """
    nx = len(xs)

    # Try to preserve `dtype`, but we need to ensure that the arguments are at
    # least floats before passing them into the function; integers can overflow
    # and cause failure.
    # There might be benefit to combining the `xs` into a single array and
    # calling `func` once on the combined array. For now, keep them separate.
    xas = np.broadcast_arrays(*xs, *args)  # broadcast and rename
    xat = np.result_type(*[xa.dtype for xa in xas])
    xat = np.float64 if np.issubdtype(xat, np.integer) else xat
    xs, args = xas[:nx], xas[nx:]
    xs = [x.astype(xat, copy=False)[()] for x in xs]
    fs = [np.asarray(func(x, *args)) for x in xs]
    shape = xs[0].shape

    message = ("The shape of the array returned by `func` must be the same as "
               "the broadcasted shape of `x` and all other `args`.")
    shapes_equal = [f.shape == shape for f in fs]
    if not np.all(shapes_equal):
        raise ValueError(message)

    # These algorithms tend to mix the dtypes of the abscissae and function
    # values, so figure out what the result will be and convert them all to
    # that type from the outset.
    xfat = np.result_type(*([f.dtype for f in fs] + [xat]))
    if not complex_ok and not np.issubdtype(xfat, np.floating):
        raise ValueError("Abscissae and function output must be real numbers.")
    xs = [x.astype(xfat, copy=True)[()] for x in xs]
    fs = [f.astype(xfat, copy=True)[()] for f in fs]

    # To ensure that we can do indexing, we'll work with at least 1d arrays,
    # but remember the appropriate shape of the output.
    xs = [x.ravel() for x in xs]
    fs = [f.ravel() for f in fs]
    args = [arg.flatten() for arg in args]
    return xs, fs, args, shape, xfat


def _scalar_optimization_check_termination(work, res, res_work_pairs, active,
                                           check_termination):
    # Checks termination conditions, updates elements of `res` with
    # corresponding elements of `work`, and compresses `work`.

    stop = check_termination(work)

    if np.any(stop):
        # update the active elements of the result object with the active
        # elements for which a termination condition has been met
        _scalar_optimization_update_active(work, res, res_work_pairs, active,
                                           stop)

        # compress the arrays to avoid unnecessary computation
        proceed = ~stop
        active = active[proceed]
        for key, val in work.items():
            work[key] = val[proceed] if isinstance(val, np.ndarray) else val
        work.args = [arg[proceed] for arg in work.args]

    return active


def _scalar_optimization_update_active(work, res, res_work_pairs, active,
                                       mask=None):
    # Update `active` indices of the arrays in result object `res` with the
    # contents of the scalars and arrays in `update_dict`. When provided,
    # `mask` is a boolean array applied both to the arrays in `update_dict`
    # that are to be used and to the arrays in `res` that are to be updated.
    update_dict = {key1: work[key2] for key1, key2 in res_work_pairs}
    update_dict['success'] = work.status == 0

    if mask is not None:
        active_mask = active[mask]
        for key, val in update_dict.items():
            res[key][active_mask] = val[mask] if np.size(val) > 1 else val
    else:
        for key, val in update_dict.items():
            res[key][active] = val


def _scalar_optimization_prepare_result(work, res, res_work_pairs, active,
                                        shape, customize_result):
    # Prepare the result object `res` by creating a copy, copying the latest
    # data from work, running the provided result customization function,
    # and reshaping the data to the original shapes.
    res = res.copy()
    _scalar_optimization_update_active(work, res, res_work_pairs, active)

    shape = customize_result(res, shape)

    for key, val in res.items():
        res[key] = np.reshape(val, shape)[()]
    res['_order_keys'] = ['success'] + [i for i, j in res_work_pairs]
    return OptimizeResult(**res)


def _differentiate_iv(func, x, args, atol, rtol, maxiter, order,
                      initial_step, step_factor, step_direction, callback):
    # Input validation for `_differentiate`

    if not callable(func):
        raise ValueError('`func` must be callable.')

    # x has more complex IV that is taken care of during initialization
    x = np.asarray(x)
    dtype = x.dtype if np.issubdtype(x.dtype, np.inexact) else np.float64

    if not np.iterable(args):
        args = (args,)

    if atol is None:
        atol = np.finfo(dtype).tiny

    if rtol is None:
        rtol = np.sqrt(np.finfo(dtype).eps)

    message = 'Tolerances and step parameters must be non-negative scalars.'
    tols = np.asarray([atol, rtol, initial_step, step_factor])
    if (not np.issubdtype(tols.dtype, np.number)
            or np.any(tols < 0)
            or tols.shape != (4,)):
        raise ValueError(message)
    initial_step, step_factor = tols[2:].astype(dtype)

    maxiter_int = int(maxiter)
    if maxiter != maxiter_int or maxiter <= 0:
        raise ValueError('`maxiter` must be a positive integer.')

    order_int = int(order)
    if order_int != order or order <= 0:
        raise ValueError('`order` must be a positive integer.')

    step_direction = np.sign(step_direction).astype(dtype)
    x, step_direction = np.broadcast_arrays(x, step_direction)
    x, step_direction = x[()], step_direction[()]

    if callback is not None and not callable(callback):
        raise ValueError('`callback` must be callable.')

    return (func, x, args, atol, rtol, maxiter_int, order_int, initial_step,
            step_factor, step_direction, callback)


def _differentiate(func, x, *, args=(), atol=None, rtol=None, maxiter=10,
                   order=8, initial_step=0.5, step_factor=2.0,
                   step_direction=0, callback=None):
    """Evaluate the derivative of an elementwise scalar function numerically.

    Parameters
    ----------
    func : callable
        The function whose derivative is desired. The signature must be::

            func(x: ndarray, *args) -> ndarray

         where each element of ``x`` is a finite real and ``args`` is a tuple,
         which may contain an arbitrary number of arrays that are broadcastable
         with `x`. ``func`` must be an elementwise function: each element
         ``func(x)[i]`` must equal ``func(x[i])`` for all indices ``i``.
    x : array_like
        Abscissae at which to evaluate the derivative.
    args : tuple, optional
        Additional positional arguments to be passed to `func`. Must be arrays
        broadcastable with `x`. If the callable to be differentiated requires
        arguments that are not broadcastable with `x`, wrap that callable with
        `func`. See Examples.
    atol, rtol : float, optional
        Absolute and relative tolerances for the stopping condition: iteration
        will stop when ``res.error < atol + rtol * abs(res.df)``. The default
        `atol` is the smallest normal number of the appropriate dtype, and
        the default `rtol` is the square root of the precision of the
        appropriate dtype.
    order : int, default: 8
        The (positive integer) order of the finite difference formula to be
        used. Odd integers will be rounded up to the next even integer.
    initial_step : float, default: 0.5
        The (absolute) initial step size for the finite difference derivative
        approximation.
    step_factor : float, default: 2.0
        The factor by which the step size is *reduced* in each iteration; i.e.
        the step size in iteration 1 is ``initial_step/step_factor``. If
        ``step_factor < 1``, subsequent steps will be greater than the initial
        step; this may be useful if steps smaller than some threshold are
        undesirable (e.g. due to subtractive cancellation error).
    maxiter : int, default: 10
        The maximum number of iterations of the algorithm to perform. See
        notes.
    step_direction : array_like
        An array representing the direction of the finite difference steps (for
        use when `x` lies near to the boundary of the domain of the function.)
        Must be broadcastable with `x` and all `args`.
        Where 0 (default), central differences are used; where negative (e.g.
        -1), steps are non-positive; and where positive (e.g. 1), all steps are
        non-negative.
    callback : callable, optional
        An optional user-supplied function to be called before the first
        iteration and after each iteration.
        Called as ``callback(res)``, where ``res`` is an ``OptimizeResult``
        similar to that returned by `_differentiate` (but containing the
        current iterate's values of all variables). If `callback` raises a
        ``StopIteration``, the algorithm will terminate immediately and
        `_differentiate` will return a result.

    Returns
    -------
    res : OptimizeResult
        An instance of `scipy.optimize.OptimizeResult` with the following
        attributes. (The descriptions are written as though the values will be
        scalars; however, if `func` returns an array, the outputs will be
        arrays of the same shape.)

        success : bool
            ``True`` when the algorithm terminated successfully (status ``0``).
        status : int
            An integer representing the exit status of the algorithm.
            ``0`` : The algorithm converged to the specified tolerances.
            ``-1`` : The error estimate increased, so iteration was terminated.
            ``-2`` : The maximum number of iterations was reached.
            ``-3`` : A non-finite value was encountered.
            ``-4`` : Iteration was terminated by `callback`.
            ``1`` : The algorithm is proceeding normally (in `callback` only).
        df : float
            The derivative of `func` at `x`, if the algorithm terminated
            successfully.
        error : float
            An estimate of the error: the magnitude of the difference between
            the current estimate of the derivative and the estimate in the
            previous iteration.
        nit : int
            The number of iterations performed.
        nfev : int
            The number of points at which `func` was evaluated.
        x : float
            The value at which the derivative of `func` was evaluated
            (after broadcasting with `args` and `step_direction`).

    Notes
    -----
    The implementation was inspired by jacobi [1]_, numdifftools [2]_, and
    DERIVEST [3]_, but the implementation follows the theory of Taylor series
    more straightforwardly (and arguably naively so).
    In the first iteration, the derivative is estimated using a finite
    difference formula of order `order` with maximum step size `initial_step`.
    Each subsequent iteration, the maximum step size is reduced by
    `step_factor`, and the derivative is estimated again until a termination
    condition is reached. The error estimate is the magnitude of the difference
    between the current derivative approximation and that of the previous
    iteration.

    The stencils of the finite difference formulae are designed such that
    abscissae are "nested": after `func` is evaluated at ``order + 1``
    points in the first iteration, `func` is evaluated at only two new points
    in each subsequent iteration; ``order - 1`` previously evaluated function
    values required by the finite difference formula are reused, and two
    function values (evaluations at the points furthest from `x`) are unused.

    Step sizes are absolute. When the step size is small relative to the
    magnitude of `x`, precision is lost; for example, if `x` is ``1e20``, the
    default initial step size of ``0.5`` cannot be resolved. Accordingly,
    consider using larger initial step sizes for large magnitudes of `x`.

    The default tolerances are challenging to satisfy at points where the
    true derivative is exactly zero. If the derivative may be exactly zero,
    consider specifying an absolute tolerance (e.g. ``atol=1e-16``) to
    improve convergence.

    References
    ----------
    [1]_ Hans Dembinski (@HDembinski). jacobi.
         https://github.com/HDembinski/jacobi
    [2]_ Per A. Brodtkorb and John D'Errico. numdifftools.
         https://numdifftools.readthedocs.io/en/latest/
    [3]_ John D'Errico. DERIVEST: Adaptive Robust Numerical Differentiation.
         https://www.mathworks.com/matlabcentral/fileexchange/13490-adaptive-robust-numerical-differentiation
    [4]_ Numerical Differentition. Wikipedia.
         https://en.wikipedia.org/wiki/Numerical_differentiation

    Examples
    --------
    Evaluate the derivative of ``np.exp`` at several points ``x``.

    >>> import numpy as np
    >>> from scipy.optimize._zeros_py import _differentiate
    >>> f = np.exp
    >>> df = np.exp  # true derivative
    >>> x = np.linspace(1, 2, 5)
    >>> res = _differentiate(f, x)
    >>> res.df  # approximation of the derivative
    array([2.71828183, 3.49034296, 4.48168907, 5.75460268, 7.3890561 ])
    >>> res.error  # estimate of the error
    array(
        [7.12940817e-12, 9.16688947e-12, 1.17594823e-11, 1.50972568e-11, 1.93942640e-11]
    )
    >>> abs(res.df - df(x))  # true error
    array(
        [3.06421555e-14, 3.01980663e-14, 5.06261699e-14, 6.30606678e-14, 8.34887715e-14]
    )

    Show the convergence of the approximation as the step size is reduced.
    Each iteration, the step size is reduced by `step_factor`, so for
    sufficiently small initial step, each iteration reduces the error by a
    factor of ``1/step_factor**order`` until finite precision arithmetic
    inhibits further improvement.

    >>> iter = list(range(1, 12))  # maximum iterations
    >>> hfac = 2  # step size reduction per iteration
    >>> hdir = [-1, 0, 1]  # compare left-, central-, and right- steps
    >>> order = 4  # order of differentiation formula
    >>> x = 1
    >>> ref = df(x)
    >>> errors = []  # true error
    >>> for i in iter:
    ...     res = _differentiate(f, x, maxiter=i, step_factor=hfac,
    ...                          step_direction=hdir, order=order,
    ...                          atol=0, rtol=0)  # prevent early termination
    ...     errors.append(abs(res.df - ref))
    >>> errors = np.array(errors)
    >>> plt.semilogy(iter, errors[:, 0], label='left differences')
    >>> plt.semilogy(iter, errors[:, 1], label='central differences')
    >>> plt.semilogy(iter, errors[:, 2], label='right differences')
    >>> plt.xlabel('iteration')
    >>> plt.ylabel('error')
    >>> plt.legend()
    >>> plt.show()
    >>> (errors[1, 1] / errors[0, 1], 1 / hfac**order)
    (0.06215223140159822, 0.0625)

    The implementation is vectorized over `x`, `step_direction`, and `args`.
    The function is evaluated once before the first iteration to perform input
    validation and standardization, and once per iteration thereafter.

    >>> def f(x, p):
    ...     print('here')
    ...     f.nit += 1
    ...     return x**p
    >>> f.nit = 0
    >>> def df(x, p):
    ...     return p*x**(p-1)
    >>> x = np.arange(1, 5)
    >>> p = np.arange(1, 6).reshape((-1, 1))
    >>> hdir = np.arange(-1, 2).reshape((-1, 1, 1))
    >>> res = _differentiate(f, x, args=(p,), step_direction=hdir, maxiter=1)
    >>> np.allclose(res.df, df(x, p))
    True
    >>> res.df.shape
    (3, 5, 4)
    >>> f.nit
    2

    """
    # TODO (followup):
    #  - investigate behavior at saddle points
    #  - array initial_step / step_factor?
    #  - multivariate functions?
    #  - vector-valued functions?

    res = _differentiate_iv(func, x, args, atol, rtol, maxiter, order,
                            initial_step, step_factor, step_direction, callback)
    func, x, args, atol, rtol, maxiter, order, h0, fac, hdir, callback = res

    # Initialization
    # Since f(x) (no step) is not needed for central differences, it may be
    # possible to eliminate this function evaluation. However, it's useful for
    # input validation and standardization, and everything else is designed to
    # reduce function calls, so let's keep it simple.
    xs, fs, args, shape, dtype = _scalar_optimization_initialize(func, (x,), args)
    x, f = xs[0], fs[0]
    df = np.full_like(f, np.nan)
    # Ideally we'd broadcast the shape of `hdir` in `_scalar_opt_init`, but
    # it's simpler to do it here than to generalize `_scalar_opt_init` further.
    # `hdir` and `x` are already broadcasted in `_differentiate_iv`, so we know
    # that `hdir` can be broadcasted to the final shape.
    hdir = np.broadcast_to(hdir, shape).flatten()

    status = np.full_like(x, _EINPROGRESS, dtype=int)  # in progress
    nit, nfev = 0, 1  # one function evaluations performed above
    # Boolean indices of left, central, right, and (all) one-sided steps
    il = hdir < 0
    ic = hdir == 0
    ir = hdir > 0
    io = il | ir

    # Most of these attributes are reasonably obvious, but:
    # - `fs` holds all the function values of all active `x`. The zeroth
    #   axis corresponds with active points `x`, the first axis corresponds
    #   with the different steps (in the order described in
    #   `_differentiate_weights`).
    # - `terms` (which could probably use a better name) is half the `order`,
    #   which is always even.
    work = OptimizeResult(x=x, df=df, fs=f[:, np.newaxis], error=np.nan, h=h0,
                          df_last=np.nan, error_last=np.nan, h0=h0, fac=fac,
                          atol=atol, rtol=rtol, nit=nit, nfev=nfev,
                          status=status, dtype=dtype, terms=(order+1)//2,
                          hdir=hdir, il=il, ic=ic, ir=ir, io=io)
    # This is the correspondence between terms in the `work` object and the
    # final result. In this case, the mapping is trivial. Note that `success`
    # is prepended automatically.
    res_work_pairs = [('status', 'status'), ('df', 'df'), ('error', 'error'),
                      ('nit', 'nit'), ('nfev', 'nfev'), ('x', 'x')]

    def pre_func_eval(work):
        """Determine the abscissae at which the function needs to be evaluated.

        See `_differentiate_weights` for a description of the stencil (pattern
        of the abscissae).

        In the first iteration, there is only one stored function value in
        `work.fs`, `f(x)`, so we need to evaluate at `order` new points. In
        subsequent iterations, we evaluate at two new points. Note that
        `work.x` is always flattened into a 1D array after broadcasting with
        all `args`, so we add a new axis at the end and evaluate all point
        in one call to the function.

        For improvement:
        - Consider measuring the step size actually taken, since `(x + h) - x`
          is not identically equal to `h` with floating point arithmetic.
        - Adjust the step size automatically if `x` is too big to resolve the
          step.
        - We could probably save some work if there are no central difference
          steps or no one-sided steps.
        """
        n = work.terms  # half the order
        h = work.h  # step size
        c = work.fac  # step reduction factor
        d = c**0.5  # square root of step reduction factor (one-sided stencil)
        # Note - no need to be careful about dtypes until we allocate `x_eval`

        if work.nit == 0:
            hc = h / c**np.arange(n)
            hc = np.concatenate((-hc[::-1], hc))
        else:
            hc = np.asarray([-h, h]) / c**(n-1)

        if work.nit == 0:
            hr = h / d**np.arange(2*n)
        else:
            hr = np.asarray([h, h/d]) / c**(n-1)

        n_new = 2*n if work.nit == 0 else 2  # number of new abscissae
        x_eval = np.zeros((len(work.hdir), n_new), dtype=work.dtype)
        il, ic, ir = work.il, work.ic, work.ir
        x_eval[ir] = work.x[ir, np.newaxis] + hr
        x_eval[ic] = work.x[ic, np.newaxis] + hc
        x_eval[il] = work.x[il, np.newaxis] - hr
        return x_eval

    def post_func_eval(x, f, work):
        """ Estimate the derivative and error from the function evaluations

        As in `pre_func_eval`: in the first iteration, there is only one stored
        function value in `work.fs`, `f(x)`, so we need to add the `order` new
        points. In subsequent iterations, we add two new points. The tricky
        part is getting the order to match that of the weights, which is
        described in `_differentiate_weights`.

        For improvement:
        - Change the order of the weights (and steps in `pre_func_eval`) to
          simplify `work_fc` concatenation and eliminate `fc` concatenation.
        - It would be simple to do one-step Richardson extrapolation with `df`
          and `df_last` to increase the order of the estimate and/or improve
          the error estimate.
        - Process the function evaluations in a more numerically favorable
          way. For instance, combining the pairs of central difference evals
          into a second-order approximation and using Richardson extrapolation
          to produce a higher order approximation seemed to retain accuracy up
          to very high order.
        - Alternatively, we could use `polyfit` like Jacobi. An advantage of
          fitting polynomial to more points than necessary is improved noise
          tolerance.
        """
        n = work.terms
        n_new = n if work.nit == 0 else 1
        il, ic, io = work.il, work.ic, work.io

        # Central difference
        # `work_fc` is *all* the points at which the function has been evaluated
        # `fc` is the points we're using *this iteration* to produce the estimate
        work_fc = (f[ic, :n_new], work.fs[ic, :], f[ic, -n_new:])
        work_fc = np.concatenate(work_fc, axis=-1)
        if work.nit == 0:
            fc = work_fc
        else:
            fc = (work_fc[:, :n], work_fc[:, n:n+1], work_fc[:, -n:])
            fc = np.concatenate(fc, axis=-1)

        # One-sided difference
        work_fo = np.concatenate((work.fs[io, :], f[io, :]), axis=-1)
        if work.nit == 0:
            fo = work_fo
        else:
            fo = np.concatenate((work_fo[:, 0:1], work_fo[:, -2*n:]), axis=-1)

        work.fs = np.zeros((len(ic), work.fs.shape[-1] + 2*n_new))
        work.fs[ic] = work_fc
        work.fs[io] = work_fo

        wc, wo = _differentiate_weights(work, n)
        work.df_last = work.df.copy()
        work.df[ic] = fc @ wc / work.h
        work.df[io] = fo @ wo / work.h
        work.df[il] *= -1

        work.h /= work.fac
        work.error_last = work.error
        # Simple error estimate - the difference in derivative estimates between
        # this iteration and the last. This is typically conservative because if
        # convergence has begin, the true error is much closer to the difference
        # between the current estimate and the *next* error estimate. However,
        # we could use Richarson extrapolation to produce an error estimate that
        # is one order higher, and take the difference between that and
        # `work.df` (which would just be constant factor that depends on `fac`.)
        work.error = abs(work.df - work.df_last)

    def check_termination(work):
        """Terminate due to convergence, non-finite values, or error increase"""
        stop = np.zeros_like(work.df).astype(bool)

        i = work.error < work.atol + work.rtol*abs(work.df)
        work.status[i] = _ECONVERGED
        stop[i] = True

        if work.nit > 0:
            i = ~((np.isfinite(work.x) & np.isfinite(work.df)) | stop)
            work.df[i], work.status[i] = np.nan, _EVALUEERR
            stop[i] = True

        # With infinite precision, there is a step size below which
        # all smaller step sizes will reduce the error. But in floating point
        # arithmetic, catastrophic cancellation will begin to cause the error
        # to increase again. This heuristic tries to avoid step sizes that are
        # too small. There may be more theoretically sound approaches for
        # detecting a step size that minimizes the total error, but this
        # heuristic seems simple and effective.
        i = (work.error > work.error_last*10) & ~stop
        work.status[i] = _EERRORINCREASE
        stop[i] = True

        return stop

    def post_termination_check(work):
        return

    def customize_result(res, shape):
        return shape

    return _scalar_optimization_loop(work, callback, shape,
                                     maxiter, func, args, dtype,
                                     pre_func_eval, post_func_eval,
                                     check_termination, post_termination_check,
                                     customize_result, res_work_pairs)


def _differentiate_weights(work, n):
    # This produces the weights of the finite difference formula for a given
    # stencil. In experiments, use of a second-order central difference formula
    # with Richardson extrapolation was more accurate numerically, but it was
    # more complicated, and it would have become even more complicated when
    # adding support for one-sided differences. However, now that all the
    # function evaluation values are stored, they can be processed in whatever
    # way is desired to produce the derivative estimate. We leave alternative
    # approaches to future work. To be more self-contained, here is the theory
    # for deriving the weights below.
    #
    # Recall that the Taylor expansion of a univariate, scalar-values function
    # about a point `x` may be expressed as:
    #      f(x + h)  =     f(x) + f'(x)*h + f''(x)/2!*h**2  + O(h**3)
    # Suppose we evaluate f(x), f(x+h), and f(x-h).  We have:
    #      f(x)      =     f(x)
    #      f(x + h)  =     f(x) + f'(x)*h + f''(x)/2!*h**2  + O(h**3)
    #      f(x - h)  =     f(x) - f'(x)*h + f''(x)/2!*h**2  + O(h**3)
    # We can solve for weights `wi` such that:
    #   w1*f(x)      = w1*(f(x))
    # + w2*f(x + h)  = w2*(f(x) + f'(x)*h + f''(x)/2!*h**2) + O(h**3)
    # + w3*f(x - h)  = w3*(f(x) - f'(x)*h + f''(x)/2!*h**2) + O(h**3)
    #                =     0    + f'(x)*h + 0               + O(h**3)
    # Then
    #     f'(x) ~ (w1*f(x) + w2*f(x+h) + w3*f(x-h))/h
    # is a finite difference derivative approximation with error O(h**2),
    # and so it is said to be a "second-order" approximation. Under certain
    # conditions (e.g. well-behaved function, `h` sufficiently small), the
    # error in the approximation will decrease with h**2; that is, if `h` is
    # reduced by a factor of 2, the error is reduced by a factor of 4.
    #
    # By default, we use eighth-order formulae. Our central-difference formula
    # uses abscissae:
    #   x-h/c**3, x-h/c**2, x-h/c, x-h, x, x+h, x+h/c, x+h/c**2, x+h/c**3
    # where `c` is the step factor. (Typically, the step factor is greater than
    # one, so the outermost points - as written above - are actually closest to
    # `x`.) This "stencil" is chosen so that each iteration, the step can be
    # reduced by the factor `c`, and most of the function evaluations can be
    # reused with the new step size. For example, in the next iteration, we
    # will have:
    #   x-h/c**4, x-h/c**3, x-h/c**2, x-h/c, x, x+h/c, x+h/c**2, x+h/c**3, x+h/c**4
    # We do not reuse `x-h` and `x+h` for the new derivative estimate.
    # While this would increase the order of the formula and thus the
    # theoretical convergence rate, it is also less stable numerically.
    # (As noted above, there are other ways of processing the values that are
    # more stable. Thus, even now we store `f(x-h)` and `f(x+h)` in `work.fs`
    # to simplify future development of this sort of improvement.)
    #
    # The (right) one-sided formula is produced similarly using abscissae
    #   x, x+h, x+h/d, x+h/d**2, ..., x+h/d**6, x+h/d**7, x+h/d**7
    # where `d` is the square root of `c`. (The left one-sided formula simply
    # uses -h.) When the step size is reduced by factor `c = d**2`, we have
    # abscissae:
    #   x, x+h/d**2, x+h/d**3..., x+h/d**8, x+h/d**9, x+h/d**9
    # `d` is chosen as the square root of `c` so that the rate of the step-size
    # reduction is the same per iteration as in the central difference case.
    # Note that because the central difference formulas are inherently of even
    # order, for simplicity, we use only even-order formulas for one-sided
    # differences, too.

    # It's possible for the user to specify `fac` in, say, double precision but
    # `x` and `args` in single precision. `fac` gets converted to single
    # precision, but we should always use double precision for the intermediate
    # calculations here to avoid additional error in the weights.
    fac = work.fac.astype(np.float64)

    # Note that if the user switches back to floating point precision with
    # `x` and `args`, then `fac` will not necessarily equal the (lower
    # precision) cached `_differentiate_weights.fac`, and the weights will
    # need to be recalculated. This could be fixed, but it's late, and of
    # low consequence.
    if fac != _differentiate_weights.fac:
        _differentiate_weights.central = []
        _differentiate_weights.right = []
        _differentiate_weights.fac = fac

    if len(_differentiate_weights.central) != 2*n + 1:
        # Central difference weights. Consider refactoring this; it could
        # probably be more compact.
        i = np.arange(-n, n + 1)
        p = np.abs(i) - 1.  # center point has power `p` -1, but sign `s` is 0
        s = np.sign(i)

        h = s / fac ** p
        A = np.vander(h, increasing=True).T
        b = np.zeros(2*n + 1)
        b[1] = 1
        weights = np.linalg.solve(A, b)

        # Enforce identities to improve accuracy
        weights[n] = 0
        for i in range(n):
            weights[-i-1] = -weights[i]

        # Cache the weights. We only need to calculate them once unless
        # the step factor changes.
        _differentiate_weights.central = weights

        # One-sided difference weights. The left one-sided weights (with
        # negative steps) are simply the negative of the right one-sided
        # weights, so no need to compute them separately.
        i = np.arange(2*n + 1)
        p = i - 1.
        s = np.sign(i)

        h = s / np.sqrt(fac) ** p
        A = np.vander(h, increasing=True).T
        b = np.zeros(2 * n + 1)
        b[1] = 1
        weights = np.linalg.solve(A, b)

        _differentiate_weights.right = weights

    return (_differentiate_weights.central.astype(work.dtype, copy=False),
            _differentiate_weights.right.astype(work.dtype, copy=False))
_differentiate_weights.central = []
_differentiate_weights.right = []
_differentiate_weights.fac = None
