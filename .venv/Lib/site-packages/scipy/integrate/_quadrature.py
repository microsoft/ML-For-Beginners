from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Any, cast
import numpy as np
import math
import warnings
from collections import namedtuple

from scipy.special import roots_legendre
from scipy.special import gammaln, logsumexp
from scipy._lib._util import _rng_spawn


__all__ = ['fixed_quad', 'quadrature', 'romberg', 'romb',
           'trapezoid', 'trapz', 'simps', 'simpson',
           'cumulative_trapezoid', 'cumtrapz', 'newton_cotes',
           'qmc_quad', 'AccuracyWarning']


def trapezoid(y, x=None, dx=1.0, axis=-1):
    r"""
    Integrate along the given axis using the composite trapezoidal rule.

    If `x` is provided, the integration happens in sequence along its
    elements - they are not sorted.

    Integrate `y` (`x`) along each 1d slice on the given axis, compute
    :math:`\int y(x) dx`.
    When `x` is specified, this integrates along the parametric curve,
    computing :math:`\int_t y(t) dt =
    \int_t y(t) \left.\frac{dx}{dt}\right|_{x=x(t)} dt`.

    Parameters
    ----------
    y : array_like
        Input array to integrate.
    x : array_like, optional
        The sample points corresponding to the `y` values. If `x` is None,
        the sample points are assumed to be evenly spaced `dx` apart. The
        default is None.
    dx : scalar, optional
        The spacing between sample points when `x` is None. The default is 1.
    axis : int, optional
        The axis along which to integrate.

    Returns
    -------
    trapezoid : float or ndarray
        Definite integral of `y` = n-dimensional array as approximated along
        a single axis by the trapezoidal rule. If `y` is a 1-dimensional array,
        then the result is a float. If `n` is greater than 1, then the result
        is an `n`-1 dimensional array.

    See Also
    --------
    cumulative_trapezoid, simpson, romb

    Notes
    -----
    Image [2]_ illustrates trapezoidal rule -- y-axis locations of points
    will be taken from `y` array, by default x-axis distances between
    points will be 1.0, alternatively they can be provided with `x` array
    or with `dx` scalar.  Return value will be equal to combined area under
    the red lines.

    References
    ----------
    .. [1] Wikipedia page: https://en.wikipedia.org/wiki/Trapezoidal_rule

    .. [2] Illustration image:
           https://en.wikipedia.org/wiki/File:Composite_trapezoidal_rule_illustration.png

    Examples
    --------
    Use the trapezoidal rule on evenly spaced points:

    >>> import numpy as np
    >>> from scipy import integrate
    >>> integrate.trapezoid([1, 2, 3])
    4.0

    The spacing between sample points can be selected by either the
    ``x`` or ``dx`` arguments:

    >>> integrate.trapezoid([1, 2, 3], x=[4, 6, 8])
    8.0
    >>> integrate.trapezoid([1, 2, 3], dx=2)
    8.0

    Using a decreasing ``x`` corresponds to integrating in reverse:

    >>> integrate.trapezoid([1, 2, 3], x=[8, 6, 4])
    -8.0

    More generally ``x`` is used to integrate along a parametric curve. We can
    estimate the integral :math:`\int_0^1 x^2 = 1/3` using:

    >>> x = np.linspace(0, 1, num=50)
    >>> y = x**2
    >>> integrate.trapezoid(y, x)
    0.33340274885464394

    Or estimate the area of a circle, noting we repeat the sample which closes
    the curve:

    >>> theta = np.linspace(0, 2 * np.pi, num=1000, endpoint=True)
    >>> integrate.trapezoid(np.cos(theta), x=np.sin(theta))
    3.141571941375841

    ``trapezoid`` can be applied along a specified axis to do multiple
    computations in one call:

    >>> a = np.arange(6).reshape(2, 3)
    >>> a
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> integrate.trapezoid(a, axis=0)
    array([1.5, 2.5, 3.5])
    >>> integrate.trapezoid(a, axis=1)
    array([2.,  8.])
    """
    # Future-proofing, in case NumPy moves from trapz to trapezoid for the same
    # reasons as SciPy
    if hasattr(np, 'trapezoid'):
        return np.trapezoid(y, x=x, dx=dx, axis=axis)
    else:
        return np.trapz(y, x=x, dx=dx, axis=axis)


# Note: alias kept for backwards compatibility. Rename was done
# because trapz is a slur in colloquial English (see gh-12924).
def trapz(y, x=None, dx=1.0, axis=-1):
    """An alias of `trapezoid`.

    `trapz` is kept for backwards compatibility. For new code, prefer
    `trapezoid` instead.
    """
    return trapezoid(y, x=x, dx=dx, axis=axis)


class AccuracyWarning(Warning):
    pass


if TYPE_CHECKING:
    # workaround for mypy function attributes see:
    # https://github.com/python/mypy/issues/2087#issuecomment-462726600
    from typing import Protocol

    class CacheAttributes(Protocol):
        cache: dict[int, tuple[Any, Any]]
else:
    CacheAttributes = Callable


def cache_decorator(func: Callable) -> CacheAttributes:
    return cast(CacheAttributes, func)


@cache_decorator
def _cached_roots_legendre(n):
    """
    Cache roots_legendre results to speed up calls of the fixed_quad
    function.
    """
    if n in _cached_roots_legendre.cache:
        return _cached_roots_legendre.cache[n]

    _cached_roots_legendre.cache[n] = roots_legendre(n)
    return _cached_roots_legendre.cache[n]


_cached_roots_legendre.cache = dict()


def fixed_quad(func, a, b, args=(), n=5):
    """
    Compute a definite integral using fixed-order Gaussian quadrature.

    Integrate `func` from `a` to `b` using Gaussian quadrature of
    order `n`.

    Parameters
    ----------
    func : callable
        A Python function or method to integrate (must accept vector inputs).
        If integrating a vector-valued function, the returned array must have
        shape ``(..., len(x))``.
    a : float
        Lower limit of integration.
    b : float
        Upper limit of integration.
    args : tuple, optional
        Extra arguments to pass to function, if any.
    n : int, optional
        Order of quadrature integration. Default is 5.

    Returns
    -------
    val : float
        Gaussian quadrature approximation to the integral
    none : None
        Statically returned value of None

    See Also
    --------
    quad : adaptive quadrature using QUADPACK
    dblquad : double integrals
    tplquad : triple integrals
    romberg : adaptive Romberg quadrature
    quadrature : adaptive Gaussian quadrature
    romb : integrators for sampled data
    simpson : integrators for sampled data
    cumulative_trapezoid : cumulative integration for sampled data
    ode : ODE integrator
    odeint : ODE integrator

    Examples
    --------
    >>> from scipy import integrate
    >>> import numpy as np
    >>> f = lambda x: x**8
    >>> integrate.fixed_quad(f, 0.0, 1.0, n=4)
    (0.1110884353741496, None)
    >>> integrate.fixed_quad(f, 0.0, 1.0, n=5)
    (0.11111111111111102, None)
    >>> print(1/9.0)  # analytical result
    0.1111111111111111

    >>> integrate.fixed_quad(np.cos, 0.0, np.pi/2, n=4)
    (0.9999999771971152, None)
    >>> integrate.fixed_quad(np.cos, 0.0, np.pi/2, n=5)
    (1.000000000039565, None)
    >>> np.sin(np.pi/2)-np.sin(0)  # analytical result
    1.0

    """
    x, w = _cached_roots_legendre(n)
    x = np.real(x)
    if np.isinf(a) or np.isinf(b):
        raise ValueError("Gaussian quadrature is only available for "
                         "finite limits.")
    y = (b-a)*(x+1)/2.0 + a
    return (b-a)/2.0 * np.sum(w*func(y, *args), axis=-1), None


def vectorize1(func, args=(), vec_func=False):
    """Vectorize the call to a function.

    This is an internal utility function used by `romberg` and
    `quadrature` to create a vectorized version of a function.

    If `vec_func` is True, the function `func` is assumed to take vector
    arguments.

    Parameters
    ----------
    func : callable
        User defined function.
    args : tuple, optional
        Extra arguments for the function.
    vec_func : bool, optional
        True if the function func takes vector arguments.

    Returns
    -------
    vfunc : callable
        A function that will take a vector argument and return the
        result.

    """
    if vec_func:
        def vfunc(x):
            return func(x, *args)
    else:
        def vfunc(x):
            if np.isscalar(x):
                return func(x, *args)
            x = np.asarray(x)
            # call with first point to get output type
            y0 = func(x[0], *args)
            n = len(x)
            dtype = getattr(y0, 'dtype', type(y0))
            output = np.empty((n,), dtype=dtype)
            output[0] = y0
            for i in range(1, n):
                output[i] = func(x[i], *args)
            return output
    return vfunc


def quadrature(func, a, b, args=(), tol=1.49e-8, rtol=1.49e-8, maxiter=50,
               vec_func=True, miniter=1):
    """
    Compute a definite integral using fixed-tolerance Gaussian quadrature.

    Integrate `func` from `a` to `b` using Gaussian quadrature
    with absolute tolerance `tol`.

    Parameters
    ----------
    func : function
        A Python function or method to integrate.
    a : float
        Lower limit of integration.
    b : float
        Upper limit of integration.
    args : tuple, optional
        Extra arguments to pass to function.
    tol, rtol : float, optional
        Iteration stops when error between last two iterates is less than
        `tol` OR the relative change is less than `rtol`.
    maxiter : int, optional
        Maximum order of Gaussian quadrature.
    vec_func : bool, optional
        True or False if func handles arrays as arguments (is
        a "vector" function). Default is True.
    miniter : int, optional
        Minimum order of Gaussian quadrature.

    Returns
    -------
    val : float
        Gaussian quadrature approximation (within tolerance) to integral.
    err : float
        Difference between last two estimates of the integral.

    See Also
    --------
    romberg : adaptive Romberg quadrature
    fixed_quad : fixed-order Gaussian quadrature
    quad : adaptive quadrature using QUADPACK
    dblquad : double integrals
    tplquad : triple integrals
    romb : integrator for sampled data
    simpson : integrator for sampled data
    cumulative_trapezoid : cumulative integration for sampled data
    ode : ODE integrator
    odeint : ODE integrator

    Examples
    --------
    >>> from scipy import integrate
    >>> import numpy as np
    >>> f = lambda x: x**8
    >>> integrate.quadrature(f, 0.0, 1.0)
    (0.11111111111111106, 4.163336342344337e-17)
    >>> print(1/9.0)  # analytical result
    0.1111111111111111

    >>> integrate.quadrature(np.cos, 0.0, np.pi/2)
    (0.9999999999999536, 3.9611425250996035e-11)
    >>> np.sin(np.pi/2)-np.sin(0)  # analytical result
    1.0

    """
    if not isinstance(args, tuple):
        args = (args,)
    vfunc = vectorize1(func, args, vec_func=vec_func)
    val = np.inf
    err = np.inf
    maxiter = max(miniter+1, maxiter)
    for n in range(miniter, maxiter+1):
        newval = fixed_quad(vfunc, a, b, (), n)[0]
        err = abs(newval-val)
        val = newval

        if err < tol or err < rtol*abs(val):
            break
    else:
        warnings.warn(
            "maxiter (%d) exceeded. Latest difference = %e" % (maxiter, err),
            AccuracyWarning)
    return val, err


def tupleset(t, i, value):
    l = list(t)
    l[i] = value
    return tuple(l)


# Note: alias kept for backwards compatibility. Rename was done
# because cumtrapz is a slur in colloquial English (see gh-12924).
def cumtrapz(y, x=None, dx=1.0, axis=-1, initial=None):
    """An alias of `cumulative_trapezoid`.

    `cumtrapz` is kept for backwards compatibility. For new code, prefer
    `cumulative_trapezoid` instead.
    """
    return cumulative_trapezoid(y, x=x, dx=dx, axis=axis, initial=initial)


def cumulative_trapezoid(y, x=None, dx=1.0, axis=-1, initial=None):
    """
    Cumulatively integrate y(x) using the composite trapezoidal rule.

    Parameters
    ----------
    y : array_like
        Values to integrate.
    x : array_like, optional
        The coordinate to integrate along. If None (default), use spacing `dx`
        between consecutive elements in `y`.
    dx : float, optional
        Spacing between elements of `y`. Only used if `x` is None.
    axis : int, optional
        Specifies the axis to cumulate. Default is -1 (last axis).
    initial : scalar, optional
        If given, insert this value at the beginning of the returned result.
        Typically this value should be 0. Default is None, which means no
        value at ``x[0]`` is returned and `res` has one element less than `y`
        along the axis of integration.

    Returns
    -------
    res : ndarray
        The result of cumulative integration of `y` along `axis`.
        If `initial` is None, the shape is such that the axis of integration
        has one less value than `y`. If `initial` is given, the shape is equal
        to that of `y`.

    See Also
    --------
    numpy.cumsum, numpy.cumprod
    quad : adaptive quadrature using QUADPACK
    romberg : adaptive Romberg quadrature
    quadrature : adaptive Gaussian quadrature
    fixed_quad : fixed-order Gaussian quadrature
    dblquad : double integrals
    tplquad : triple integrals
    romb : integrators for sampled data
    ode : ODE integrators
    odeint : ODE integrators

    Examples
    --------
    >>> from scipy import integrate
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    >>> x = np.linspace(-2, 2, num=20)
    >>> y = x
    >>> y_int = integrate.cumulative_trapezoid(y, x, initial=0)
    >>> plt.plot(x, y_int, 'ro', x, y[0] + 0.5 * x**2, 'b-')
    >>> plt.show()

    """
    y = np.asarray(y)
    if x is None:
        d = dx
    else:
        x = np.asarray(x)
        if x.ndim == 1:
            d = np.diff(x)
            # reshape to correct shape
            shape = [1] * y.ndim
            shape[axis] = -1
            d = d.reshape(shape)
        elif len(x.shape) != len(y.shape):
            raise ValueError("If given, shape of x must be 1-D or the "
                             "same as y.")
        else:
            d = np.diff(x, axis=axis)

        if d.shape[axis] != y.shape[axis] - 1:
            raise ValueError("If given, length of x along axis must be the "
                             "same as y.")

    nd = len(y.shape)
    slice1 = tupleset((slice(None),)*nd, axis, slice(1, None))
    slice2 = tupleset((slice(None),)*nd, axis, slice(None, -1))
    res = np.cumsum(d * (y[slice1] + y[slice2]) / 2.0, axis=axis)

    if initial is not None:
        if not np.isscalar(initial):
            raise ValueError("`initial` parameter should be a scalar.")

        shape = list(res.shape)
        shape[axis] = 1
        res = np.concatenate([np.full(shape, initial, dtype=res.dtype), res],
                             axis=axis)

    return res


def _basic_simpson(y, start, stop, x, dx, axis):
    nd = len(y.shape)
    if start is None:
        start = 0
    step = 2
    slice_all = (slice(None),)*nd
    slice0 = tupleset(slice_all, axis, slice(start, stop, step))
    slice1 = tupleset(slice_all, axis, slice(start+1, stop+1, step))
    slice2 = tupleset(slice_all, axis, slice(start+2, stop+2, step))

    if x is None:  # Even-spaced Simpson's rule.
        result = np.sum(y[slice0] + 4.0*y[slice1] + y[slice2], axis=axis)
        result *= dx / 3.0
    else:
        # Account for possibly different spacings.
        #    Simpson's rule changes a bit.
        h = np.diff(x, axis=axis)
        sl0 = tupleset(slice_all, axis, slice(start, stop, step))
        sl1 = tupleset(slice_all, axis, slice(start+1, stop+1, step))
        h0 = np.float64(h[sl0])
        h1 = np.float64(h[sl1])
        hsum = h0 + h1
        hprod = h0 * h1
        h0divh1 = np.true_divide(h0, h1, out=np.zeros_like(h0), where=h1 != 0)
        tmp = hsum/6.0 * (y[slice0] *
                          (2.0 - np.true_divide(1.0, h0divh1,
                                                out=np.zeros_like(h0divh1),
                                                where=h0divh1 != 0)) +
                          y[slice1] * (hsum *
                                       np.true_divide(hsum, hprod,
                                                      out=np.zeros_like(hsum),
                                                      where=hprod != 0)) +
                          y[slice2] * (2.0 - h0divh1))
        result = np.sum(tmp, axis=axis)
    return result


# Note: alias kept for backwards compatibility. simps was renamed to simpson
# because the former is a slur in colloquial English (see gh-12924).
def simps(y, x=None, dx=1.0, axis=-1, even=None):
    """An alias of `simpson`.

    `simps` is kept for backwards compatibility. For new code, prefer
    `simpson` instead.
    """
    return simpson(y, x=x, dx=dx, axis=axis, even=even)


def simpson(y, x=None, dx=1.0, axis=-1, even=None):
    """
    Integrate y(x) using samples along the given axis and the composite
    Simpson's rule. If x is None, spacing of dx is assumed.

    If there are an even number of samples, N, then there are an odd
    number of intervals (N-1), but Simpson's rule requires an even number
    of intervals. The parameter 'even' controls how this is handled.

    Parameters
    ----------
    y : array_like
        Array to be integrated.
    x : array_like, optional
        If given, the points at which `y` is sampled.
    dx : float, optional
        Spacing of integration points along axis of `x`. Only used when
        `x` is None. Default is 1.
    axis : int, optional
        Axis along which to integrate. Default is the last axis.
    even : {None, 'simpson', 'avg', 'first', 'last'}, optional
        'avg' : Average two results:
            1) use the first N-2 intervals with
               a trapezoidal rule on the last interval and
            2) use the last
               N-2 intervals with a trapezoidal rule on the first interval.

        'first' : Use Simpson's rule for the first N-2 intervals with
                a trapezoidal rule on the last interval.

        'last' : Use Simpson's rule for the last N-2 intervals with a
               trapezoidal rule on the first interval.

        None : equivalent to 'simpson' (default)

        'simpson' : Use Simpson's rule for the first N-2 intervals with the
                  addition of a 3-point parabolic segment for the last
                  interval using equations outlined by Cartwright [1]_.
                  If the axis to be integrated over only has two points then
                  the integration falls back to a trapezoidal integration.

                  .. versionadded:: 1.11.0

        .. versionchanged:: 1.11.0
            The newly added 'simpson' option is now the default as it is more
            accurate in most situations.

        .. deprecated:: 1.11.0
            Parameter `even` is deprecated and will be removed in SciPy
            1.13.0. After this time the behaviour for an even number of
            points will follow that of `even='simpson'`.

    Returns
    -------
    float
        The estimated integral computed with the composite Simpson's rule.

    See Also
    --------
    quad : adaptive quadrature using QUADPACK
    romberg : adaptive Romberg quadrature
    quadrature : adaptive Gaussian quadrature
    fixed_quad : fixed-order Gaussian quadrature
    dblquad : double integrals
    tplquad : triple integrals
    romb : integrators for sampled data
    cumulative_trapezoid : cumulative integration for sampled data
    ode : ODE integrators
    odeint : ODE integrators

    Notes
    -----
    For an odd number of samples that are equally spaced the result is
    exact if the function is a polynomial of order 3 or less. If
    the samples are not equally spaced, then the result is exact only
    if the function is a polynomial of order 2 or less.

    References
    ----------
    .. [1] Cartwright, Kenneth V. Simpson's Rule Cumulative Integration with
           MS Excel and Irregularly-spaced Data. Journal of Mathematical
           Sciences and Mathematics Education. 12 (2): 1-9

    Examples
    --------
    >>> from scipy import integrate
    >>> import numpy as np
    >>> x = np.arange(0, 10)
    >>> y = np.arange(0, 10)

    >>> integrate.simpson(y, x)
    40.5

    >>> y = np.power(x, 3)
    >>> integrate.simpson(y, x)
    1640.5
    >>> integrate.quad(lambda x: x**3, 0, 9)[0]
    1640.25

    >>> integrate.simpson(y, x, even='first')
    1644.5

    """
    y = np.asarray(y)
    nd = len(y.shape)
    N = y.shape[axis]
    last_dx = dx
    first_dx = dx
    returnshape = 0
    if x is not None:
        x = np.asarray(x)
        if len(x.shape) == 1:
            shapex = [1] * nd
            shapex[axis] = x.shape[0]
            saveshape = x.shape
            returnshape = 1
            x = x.reshape(tuple(shapex))
        elif len(x.shape) != len(y.shape):
            raise ValueError("If given, shape of x must be 1-D or the "
                             "same as y.")
        if x.shape[axis] != N:
            raise ValueError("If given, length of x along axis must be the "
                             "same as y.")

    # even keyword parameter is deprecated
    if even is not None:
        warnings.warn(
            "The 'even' keyword is deprecated as of SciPy 1.11.0 and will be "
            "removed in SciPy 1.13.0",
            DeprecationWarning, stacklevel=2
        )

    if N % 2 == 0:
        val = 0.0
        result = 0.0
        slice_all = (slice(None),) * nd

        # default is 'simpson'
        even = even if even is not None else "simpson"

        if even not in ['avg', 'last', 'first', 'simpson']:
            raise ValueError(
                "Parameter 'even' must be 'simpson', "
                "'avg', 'last', or 'first'."
            )

        if N == 2:
            # need at least 3 points in integration axis to form parabolic
            # segment. If there are two points then any of 'avg', 'first',
            # 'last' should give the same result.
            slice1 = tupleset(slice_all, axis, -1)
            slice2 = tupleset(slice_all, axis, -2)
            if x is not None:
                last_dx = x[slice1] - x[slice2]
            val += 0.5 * last_dx * (y[slice1] + y[slice2])

            # calculation is finished. Set `even` to None to skip other
            # scenarios
            even = None

        if even == 'simpson':
            # use Simpson's rule on first intervals
            result = _basic_simpson(y, 0, N-3, x, dx, axis)

            slice1 = tupleset(slice_all, axis, -1)
            slice2 = tupleset(slice_all, axis, -2)
            slice3 = tupleset(slice_all, axis, -3)

            h = np.asfarray([dx, dx])
            if x is not None:
                # grab the last two spacings from the appropriate axis
                hm2 = tupleset(slice_all, axis, slice(-2, -1, 1))
                hm1 = tupleset(slice_all, axis, slice(-1, None, 1))

                diffs = np.float64(np.diff(x, axis=axis))
                h = [np.squeeze(diffs[hm2], axis=axis),
                     np.squeeze(diffs[hm1], axis=axis)]

            # This is the correction for the last interval according to
            # Cartwright.
            # However, I used the equations given at
            # https://en.wikipedia.org/wiki/Simpson%27s_rule#Composite_Simpson's_rule_for_irregularly_spaced_data
            # A footnote on Wikipedia says:
            # Cartwright 2017, Equation 8. The equation in Cartwright is
            # calculating the first interval whereas the equations in the
            # Wikipedia article are adjusting for the last integral. If the
            # proper algebraic substitutions are made, the equation results in
            # the values shown.
            num = 2 * h[1] ** 2 + 3 * h[0] * h[1]
            den = 6 * (h[1] + h[0])
            alpha = np.true_divide(
                num,
                den,
                out=np.zeros_like(den),
                where=den != 0
            )

            num = h[1] ** 2 + 3.0 * h[0] * h[1]
            den = 6 * h[0]
            beta = np.true_divide(
                num,
                den,
                out=np.zeros_like(den),
                where=den != 0
            )

            num = 1 * h[1] ** 3
            den = 6 * h[0] * (h[0] + h[1])
            eta = np.true_divide(
                num,
                den,
                out=np.zeros_like(den),
                where=den != 0
            )

            result += alpha*y[slice1] + beta*y[slice2] - eta*y[slice3]

        # The following code (down to result=result+val) can be removed
        # once the 'even' keyword is removed.

        # Compute using Simpson's rule on first intervals
        if even in ['avg', 'first']:
            slice1 = tupleset(slice_all, axis, -1)
            slice2 = tupleset(slice_all, axis, -2)
            if x is not None:
                last_dx = x[slice1] - x[slice2]
            val += 0.5*last_dx*(y[slice1]+y[slice2])
            result = _basic_simpson(y, 0, N-3, x, dx, axis)
        # Compute using Simpson's rule on last set of intervals
        if even in ['avg', 'last']:
            slice1 = tupleset(slice_all, axis, 0)
            slice2 = tupleset(slice_all, axis, 1)
            if x is not None:
                first_dx = x[tuple(slice2)] - x[tuple(slice1)]
            val += 0.5*first_dx*(y[slice2]+y[slice1])
            result += _basic_simpson(y, 1, N-2, x, dx, axis)
        if even == 'avg':
            val /= 2.0
            result /= 2.0
        result = result + val
    else:
        result = _basic_simpson(y, 0, N-2, x, dx, axis)
    if returnshape:
        x = x.reshape(saveshape)
    return result


def romb(y, dx=1.0, axis=-1, show=False):
    """
    Romberg integration using samples of a function.

    Parameters
    ----------
    y : array_like
        A vector of ``2**k + 1`` equally-spaced samples of a function.
    dx : float, optional
        The sample spacing. Default is 1.
    axis : int, optional
        The axis along which to integrate. Default is -1 (last axis).
    show : bool, optional
        When `y` is a single 1-D array, then if this argument is True
        print the table showing Richardson extrapolation from the
        samples. Default is False.

    Returns
    -------
    romb : ndarray
        The integrated result for `axis`.

    See Also
    --------
    quad : adaptive quadrature using QUADPACK
    romberg : adaptive Romberg quadrature
    quadrature : adaptive Gaussian quadrature
    fixed_quad : fixed-order Gaussian quadrature
    dblquad : double integrals
    tplquad : triple integrals
    simpson : integrators for sampled data
    cumulative_trapezoid : cumulative integration for sampled data
    ode : ODE integrators
    odeint : ODE integrators

    Examples
    --------
    >>> from scipy import integrate
    >>> import numpy as np
    >>> x = np.arange(10, 14.25, 0.25)
    >>> y = np.arange(3, 12)

    >>> integrate.romb(y)
    56.0

    >>> y = np.sin(np.power(x, 2.5))
    >>> integrate.romb(y)
    -0.742561336672229

    >>> integrate.romb(y, show=True)
    Richardson Extrapolation Table for Romberg Integration
    ======================================================
    -0.81576
     4.63862  6.45674
    -1.10581 -3.02062 -3.65245
    -2.57379 -3.06311 -3.06595 -3.05664
    -1.34093 -0.92997 -0.78776 -0.75160 -0.74256
    ======================================================
    -0.742561336672229  # may vary

    """
    y = np.asarray(y)
    nd = len(y.shape)
    Nsamps = y.shape[axis]
    Ninterv = Nsamps-1
    n = 1
    k = 0
    while n < Ninterv:
        n <<= 1
        k += 1
    if n != Ninterv:
        raise ValueError("Number of samples must be one plus a "
                         "non-negative power of 2.")

    R = {}
    slice_all = (slice(None),) * nd
    slice0 = tupleset(slice_all, axis, 0)
    slicem1 = tupleset(slice_all, axis, -1)
    h = Ninterv * np.asarray(dx, dtype=float)
    R[(0, 0)] = (y[slice0] + y[slicem1])/2.0*h
    slice_R = slice_all
    start = stop = step = Ninterv
    for i in range(1, k+1):
        start >>= 1
        slice_R = tupleset(slice_R, axis, slice(start, stop, step))
        step >>= 1
        R[(i, 0)] = 0.5*(R[(i-1, 0)] + h*y[slice_R].sum(axis=axis))
        for j in range(1, i+1):
            prev = R[(i, j-1)]
            R[(i, j)] = prev + (prev-R[(i-1, j-1)]) / ((1 << (2*j))-1)
        h /= 2.0

    if show:
        if not np.isscalar(R[(0, 0)]):
            print("*** Printing table only supported for integrals" +
                  " of a single data set.")
        else:
            try:
                precis = show[0]
            except (TypeError, IndexError):
                precis = 5
            try:
                width = show[1]
            except (TypeError, IndexError):
                width = 8
            formstr = "%%%d.%df" % (width, precis)

            title = "Richardson Extrapolation Table for Romberg Integration"
            print(title, "=" * len(title), sep="\n", end="\n")
            for i in range(k+1):
                for j in range(i+1):
                    print(formstr % R[(i, j)], end=" ")
                print()
            print("=" * len(title))

    return R[(k, k)]

# Romberg quadratures for numeric integration.
#
# Written by Scott M. Ransom <ransom@cfa.harvard.edu>
# last revision: 14 Nov 98
#
# Cosmetic changes by Konrad Hinsen <hinsen@cnrs-orleans.fr>
# last revision: 1999-7-21
#
# Adapted to SciPy by Travis Oliphant <oliphant.travis@ieee.org>
# last revision: Dec 2001


def _difftrap(function, interval, numtraps):
    """
    Perform part of the trapezoidal rule to integrate a function.
    Assume that we had called difftrap with all lower powers-of-2
    starting with 1. Calling difftrap only returns the summation
    of the new ordinates. It does _not_ multiply by the width
    of the trapezoids. This must be performed by the caller.
        'function' is the function to evaluate (must accept vector arguments).
        'interval' is a sequence with lower and upper limits
                   of integration.
        'numtraps' is the number of trapezoids to use (must be a
                   power-of-2).
    """
    if numtraps <= 0:
        raise ValueError("numtraps must be > 0 in difftrap().")
    elif numtraps == 1:
        return 0.5*(function(interval[0])+function(interval[1]))
    else:
        numtosum = numtraps/2
        h = float(interval[1]-interval[0])/numtosum
        lox = interval[0] + 0.5 * h
        points = lox + h * np.arange(numtosum)
        s = np.sum(function(points), axis=0)
        return s


def _romberg_diff(b, c, k):
    """
    Compute the differences for the Romberg quadrature corrections.
    See Forman Acton's "Real Computing Made Real," p 143.
    """
    tmp = 4.0**k
    return (tmp * c - b)/(tmp - 1.0)


def _printresmat(function, interval, resmat):
    # Print the Romberg result matrix.
    i = j = 0
    print('Romberg integration of', repr(function), end=' ')
    print('from', interval)
    print('')
    print('%6s %9s %9s' % ('Steps', 'StepSize', 'Results'))
    for i in range(len(resmat)):
        print('%6d %9f' % (2**i, (interval[1]-interval[0])/(2.**i)), end=' ')
        for j in range(i+1):
            print('%9f' % (resmat[i][j]), end=' ')
        print('')
    print('')
    print('The final result is', resmat[i][j], end=' ')
    print('after', 2**(len(resmat)-1)+1, 'function evaluations.')


def romberg(function, a, b, args=(), tol=1.48e-8, rtol=1.48e-8, show=False,
            divmax=10, vec_func=False):
    """
    Romberg integration of a callable function or method.

    Returns the integral of `function` (a function of one variable)
    over the interval (`a`, `b`).

    If `show` is 1, the triangular array of the intermediate results
    will be printed. If `vec_func` is True (default is False), then
    `function` is assumed to support vector arguments.

    Parameters
    ----------
    function : callable
        Function to be integrated.
    a : float
        Lower limit of integration.
    b : float
        Upper limit of integration.

    Returns
    -------
    results : float
        Result of the integration.

    Other Parameters
    ----------------
    args : tuple, optional
        Extra arguments to pass to function. Each element of `args` will
        be passed as a single argument to `func`. Default is to pass no
        extra arguments.
    tol, rtol : float, optional
        The desired absolute and relative tolerances. Defaults are 1.48e-8.
    show : bool, optional
        Whether to print the results. Default is False.
    divmax : int, optional
        Maximum order of extrapolation. Default is 10.
    vec_func : bool, optional
        Whether `func` handles arrays as arguments (i.e., whether it is a
        "vector" function). Default is False.

    See Also
    --------
    fixed_quad : Fixed-order Gaussian quadrature.
    quad : Adaptive quadrature using QUADPACK.
    dblquad : Double integrals.
    tplquad : Triple integrals.
    romb : Integrators for sampled data.
    simpson : Integrators for sampled data.
    cumulative_trapezoid : Cumulative integration for sampled data.
    ode : ODE integrator.
    odeint : ODE integrator.

    References
    ----------
    .. [1] 'Romberg's method' https://en.wikipedia.org/wiki/Romberg%27s_method

    Examples
    --------
    Integrate a gaussian from 0 to 1 and compare to the error function.

    >>> from scipy import integrate
    >>> from scipy.special import erf
    >>> import numpy as np
    >>> gaussian = lambda x: 1/np.sqrt(np.pi) * np.exp(-x**2)
    >>> result = integrate.romberg(gaussian, 0, 1, show=True)
    Romberg integration of <function vfunc at ...> from [0, 1]

    ::

       Steps  StepSize  Results
           1  1.000000  0.385872
           2  0.500000  0.412631  0.421551
           4  0.250000  0.419184  0.421368  0.421356
           8  0.125000  0.420810  0.421352  0.421350  0.421350
          16  0.062500  0.421215  0.421350  0.421350  0.421350  0.421350
          32  0.031250  0.421317  0.421350  0.421350  0.421350  0.421350  0.421350

    The final result is 0.421350396475 after 33 function evaluations.

    >>> print("%g %g" % (2*result, erf(1)))
    0.842701 0.842701

    """
    if np.isinf(a) or np.isinf(b):
        raise ValueError("Romberg integration only available "
                         "for finite limits.")
    vfunc = vectorize1(function, args, vec_func=vec_func)
    n = 1
    interval = [a, b]
    intrange = b - a
    ordsum = _difftrap(vfunc, interval, n)
    result = intrange * ordsum
    resmat = [[result]]
    err = np.inf
    last_row = resmat[0]
    for i in range(1, divmax+1):
        n *= 2
        ordsum += _difftrap(vfunc, interval, n)
        row = [intrange * ordsum / n]
        for k in range(i):
            row.append(_romberg_diff(last_row[k], row[k], k+1))
        result = row[i]
        lastresult = last_row[i-1]
        if show:
            resmat.append(row)
        err = abs(result - lastresult)
        if err < tol or err < rtol * abs(result):
            break
        last_row = row
    else:
        warnings.warn(
            "divmax (%d) exceeded. Latest difference = %e" % (divmax, err),
            AccuracyWarning)

    if show:
        _printresmat(vfunc, interval, resmat)
    return result


# Coefficients for Newton-Cotes quadrature
#
# These are the points being used
#  to construct the local interpolating polynomial
#  a are the weights for Newton-Cotes integration
#  B is the error coefficient.
#  error in these coefficients grows as N gets larger.
#  or as samples are closer and closer together

# You can use maxima to find these rational coefficients
#  for equally spaced data using the commands
#  a(i,N) := integrate(product(r-j,j,0,i-1) * product(r-j,j,i+1,N),r,0,N) / ((N-i)! * i!) * (-1)^(N-i);
#  Be(N) := N^(N+2)/(N+2)! * (N/(N+3) - sum((i/N)^(N+2)*a(i,N),i,0,N));
#  Bo(N) := N^(N+1)/(N+1)! * (N/(N+2) - sum((i/N)^(N+1)*a(i,N),i,0,N));
#  B(N) := (if (mod(N,2)=0) then Be(N) else Bo(N));
#
# pre-computed for equally-spaced weights
#
# num_a, den_a, int_a, num_B, den_B = _builtincoeffs[N]
#
#  a = num_a*array(int_a)/den_a
#  B = num_B*1.0 / den_B
#
#  integrate(f(x),x,x_0,x_N) = dx*sum(a*f(x_i)) + B*(dx)^(2k+3) f^(2k+2)(x*)
#    where k = N // 2
#
_builtincoeffs = {
    1: (1,2,[1,1],-1,12),
    2: (1,3,[1,4,1],-1,90),
    3: (3,8,[1,3,3,1],-3,80),
    4: (2,45,[7,32,12,32,7],-8,945),
    5: (5,288,[19,75,50,50,75,19],-275,12096),
    6: (1,140,[41,216,27,272,27,216,41],-9,1400),
    7: (7,17280,[751,3577,1323,2989,2989,1323,3577,751],-8183,518400),
    8: (4,14175,[989,5888,-928,10496,-4540,10496,-928,5888,989],
        -2368,467775),
    9: (9,89600,[2857,15741,1080,19344,5778,5778,19344,1080,
                 15741,2857], -4671, 394240),
    10: (5,299376,[16067,106300,-48525,272400,-260550,427368,
                   -260550,272400,-48525,106300,16067],
         -673175, 163459296),
    11: (11,87091200,[2171465,13486539,-3237113, 25226685,-9595542,
                      15493566,15493566,-9595542,25226685,-3237113,
                      13486539,2171465], -2224234463, 237758976000),
    12: (1, 5255250, [1364651,9903168,-7587864,35725120,-51491295,
                      87516288,-87797136,87516288,-51491295,35725120,
                      -7587864,9903168,1364651], -3012, 875875),
    13: (13, 402361344000,[8181904909, 56280729661, -31268252574,
                           156074417954,-151659573325,206683437987,
                           -43111992612,-43111992612,206683437987,
                           -151659573325,156074417954,-31268252574,
                           56280729661,8181904909], -2639651053,
         344881152000),
    14: (7, 2501928000, [90241897,710986864,-770720657,3501442784,
                         -6625093363,12630121616,-16802270373,19534438464,
                         -16802270373,12630121616,-6625093363,3501442784,
                         -770720657,710986864,90241897], -3740727473,
         1275983280000)
    }


def newton_cotes(rn, equal=0):
    r"""
    Return weights and error coefficient for Newton-Cotes integration.

    Suppose we have (N+1) samples of f at the positions
    x_0, x_1, ..., x_N. Then an N-point Newton-Cotes formula for the
    integral between x_0 and x_N is:

    :math:`\int_{x_0}^{x_N} f(x)dx = \Delta x \sum_{i=0}^{N} a_i f(x_i)
    + B_N (\Delta x)^{N+2} f^{N+1} (\xi)`

    where :math:`\xi \in [x_0,x_N]`
    and :math:`\Delta x = \frac{x_N-x_0}{N}` is the average samples spacing.

    If the samples are equally-spaced and N is even, then the error
    term is :math:`B_N (\Delta x)^{N+3} f^{N+2}(\xi)`.

    Parameters
    ----------
    rn : int
        The integer order for equally-spaced data or the relative positions of
        the samples with the first sample at 0 and the last at N, where N+1 is
        the length of `rn`. N is the order of the Newton-Cotes integration.
    equal : int, optional
        Set to 1 to enforce equally spaced data.

    Returns
    -------
    an : ndarray
        1-D array of weights to apply to the function at the provided sample
        positions.
    B : float
        Error coefficient.

    Notes
    -----
    Normally, the Newton-Cotes rules are used on smaller integration
    regions and a composite rule is used to return the total integral.

    Examples
    --------
    Compute the integral of sin(x) in [0, :math:`\pi`]:

    >>> from scipy.integrate import newton_cotes
    >>> import numpy as np
    >>> def f(x):
    ...     return np.sin(x)
    >>> a = 0
    >>> b = np.pi
    >>> exact = 2
    >>> for N in [2, 4, 6, 8, 10]:
    ...     x = np.linspace(a, b, N + 1)
    ...     an, B = newton_cotes(N, 1)
    ...     dx = (b - a) / N
    ...     quad = dx * np.sum(an * f(x))
    ...     error = abs(quad - exact)
    ...     print('{:2d}  {:10.9f}  {:.5e}'.format(N, quad, error))
    ...
     2   2.094395102   9.43951e-02
     4   1.998570732   1.42927e-03
     6   2.000017814   1.78136e-05
     8   1.999999835   1.64725e-07
    10   2.000000001   1.14677e-09

    """
    try:
        N = len(rn)-1
        if equal:
            rn = np.arange(N+1)
        elif np.all(np.diff(rn) == 1):
            equal = 1
    except Exception:
        N = rn
        rn = np.arange(N+1)
        equal = 1

    if equal and N in _builtincoeffs:
        na, da, vi, nb, db = _builtincoeffs[N]
        an = na * np.array(vi, dtype=float) / da
        return an, float(nb)/db

    if (rn[0] != 0) or (rn[-1] != N):
        raise ValueError("The sample positions must start at 0"
                         " and end at N")
    yi = rn / float(N)
    ti = 2 * yi - 1
    nvec = np.arange(N+1)
    C = ti ** nvec[:, np.newaxis]
    Cinv = np.linalg.inv(C)
    # improve precision of result
    for i in range(2):
        Cinv = 2*Cinv - Cinv.dot(C).dot(Cinv)
    vec = 2.0 / (nvec[::2]+1)
    ai = Cinv[:, ::2].dot(vec) * (N / 2.)

    if (N % 2 == 0) and equal:
        BN = N/(N+3.)
        power = N+2
    else:
        BN = N/(N+2.)
        power = N+1

    BN = BN - np.dot(yi**power, ai)
    p1 = power+1
    fac = power*math.log(N) - gammaln(p1)
    fac = math.exp(fac)
    return ai, BN*fac


def _qmc_quad_iv(func, a, b, n_points, n_estimates, qrng, log):

    # lazy import to avoid issues with partially-initialized submodule
    if not hasattr(qmc_quad, 'qmc'):
        from scipy import stats
        qmc_quad.stats = stats
    else:
        stats = qmc_quad.stats

    if not callable(func):
        message = "`func` must be callable."
        raise TypeError(message)

    # a, b will be modified, so copy. Oh well if it's copied twice.
    a = np.atleast_1d(a).copy()
    b = np.atleast_1d(b).copy()
    a, b = np.broadcast_arrays(a, b)
    dim = a.shape[0]

    try:
        func((a + b) / 2)
    except Exception as e:
        message = ("`func` must evaluate the integrand at points within "
                   "the integration range; e.g. `func( (a + b) / 2)` "
                   "must return the integrand at the centroid of the "
                   "integration volume.")
        raise ValueError(message) from e

    try:
        func(np.array([a, b]).T)
        vfunc = func
    except Exception as e:
        message = ("Exception encountered when attempting vectorized call to "
                   f"`func`: {e}. For better performance, `func` should "
                   "accept two-dimensional array `x` with shape `(len(a), "
                   "n_points)` and return an array of the integrand value at "
                   "each of the `n_points.")
        warnings.warn(message, stacklevel=3)

        def vfunc(x):
            return np.apply_along_axis(func, axis=-1, arr=x)

    n_points_int = np.int64(n_points)
    if n_points != n_points_int:
        message = "`n_points` must be an integer."
        raise TypeError(message)

    n_estimates_int = np.int64(n_estimates)
    if n_estimates != n_estimates_int:
        message = "`n_estimates` must be an integer."
        raise TypeError(message)

    if qrng is None:
        qrng = stats.qmc.Halton(dim)
    elif not isinstance(qrng, stats.qmc.QMCEngine):
        message = "`qrng` must be an instance of scipy.stats.qmc.QMCEngine."
        raise TypeError(message)

    if qrng.d != a.shape[0]:
        message = ("`qrng` must be initialized with dimensionality equal to "
                   "the number of variables in `a`, i.e., "
                   "`qrng.random().shape[-1]` must equal `a.shape[0]`.")
        raise ValueError(message)

    rng_seed = getattr(qrng, 'rng_seed', None)
    rng = stats._qmc.check_random_state(rng_seed)

    if log not in {True, False}:
        message = "`log` must be boolean (`True` or `False`)."
        raise TypeError(message)

    return (vfunc, a, b, n_points_int, n_estimates_int, qrng, rng, log, stats)


QMCQuadResult = namedtuple('QMCQuadResult', ['integral', 'standard_error'])


def qmc_quad(func, a, b, *, n_estimates=8, n_points=1024, qrng=None,
             log=False):
    """
    Compute an integral in N-dimensions using Quasi-Monte Carlo quadrature.

    Parameters
    ----------
    func : callable
        The integrand. Must accept a single argument ``x``, an array which
        specifies the point(s) at which to evaluate the scalar-valued
        integrand, and return the value(s) of the integrand.
        For efficiency, the function should be vectorized to accept an array of
        shape ``(d, n_points)``, where ``d`` is the number of variables (i.e.
        the dimensionality of the function domain) and `n_points` is the number
        of quadrature points, and return an array of shape ``(n_points,)``,
        the integrand at each quadrature point.
    a, b : array-like
        One-dimensional arrays specifying the lower and upper integration
        limits, respectively, of each of the ``d`` variables.
    n_estimates, n_points : int, optional
        `n_estimates` (default: 8) statistically independent QMC samples, each
        of `n_points` (default: 1024) points, will be generated by `qrng`.
        The total number of points at which the integrand `func` will be
        evaluated is ``n_points * n_estimates``. See Notes for details.
    qrng : `~scipy.stats.qmc.QMCEngine`, optional
        An instance of the QMCEngine from which to sample QMC points.
        The QMCEngine must be initialized to a number of dimensions ``d``
        corresponding with the number of variables ``x1, ..., xd`` passed to
        `func`.
        The provided QMCEngine is used to produce the first integral estimate.
        If `n_estimates` is greater than one, additional QMCEngines are
        spawned from the first (with scrambling enabled, if it is an option.)
        If a QMCEngine is not provided, the default `scipy.stats.qmc.Halton`
        will be initialized with the number of dimensions determine from
        the length of `a`.
    log : boolean, default: False
        When set to True, `func` returns the log of the integrand, and
        the result object contains the log of the integral.

    Returns
    -------
    result : object
        A result object with attributes:

        integral : float
            The estimate of the integral.
        standard_error :
            The error estimate. See Notes for interpretation.

    Notes
    -----
    Values of the integrand at each of the `n_points` points of a QMC sample
    are used to produce an estimate of the integral. This estimate is drawn
    from a population of possible estimates of the integral, the value of
    which we obtain depends on the particular points at which the integral
    was evaluated. We perform this process `n_estimates` times, each time
    evaluating the integrand at different scrambled QMC points, effectively
    drawing i.i.d. random samples from the population of integral estimates.
    The sample mean :math:`m` of these integral estimates is an
    unbiased estimator of the true value of the integral, and the standard
    error of the mean :math:`s` of these estimates may be used to generate
    confidence intervals using the t distribution with ``n_estimates - 1``
    degrees of freedom. Perhaps counter-intuitively, increasing `n_points`
    while keeping the total number of function evaluation points
    ``n_points * n_estimates`` fixed tends to reduce the actual error, whereas
    increasing `n_estimates` tends to decrease the error estimate.

    Examples
    --------
    QMC quadrature is particularly useful for computing integrals in higher
    dimensions. An example integrand is the probability density function
    of a multivariate normal distribution.

    >>> import numpy as np
    >>> from scipy import stats
    >>> dim = 8
    >>> mean = np.zeros(dim)
    >>> cov = np.eye(dim)
    >>> def func(x):
    ...     # `multivariate_normal` expects the _last_ axis to correspond with
    ...     # the dimensionality of the space, so `x` must be transposed
    ...     return stats.multivariate_normal.pdf(x.T, mean, cov)

    To compute the integral over the unit hypercube:

    >>> from scipy.integrate import qmc_quad
    >>> a = np.zeros(dim)
    >>> b = np.ones(dim)
    >>> rng = np.random.default_rng()
    >>> qrng = stats.qmc.Halton(d=dim, seed=rng)
    >>> n_estimates = 8
    >>> res = qmc_quad(func, a, b, n_estimates=n_estimates, qrng=qrng)
    >>> res.integral, res.standard_error
    (0.00018429555666024108, 1.0389431116001344e-07)

    A two-sided, 99% confidence interval for the integral may be estimated
    as:

    >>> t = stats.t(df=n_estimates-1, loc=res.integral,
    ...             scale=res.standard_error)
    >>> t.interval(0.99)
    (0.0001839319802536469, 0.00018465913306683527)

    Indeed, the value reported by `scipy.stats.multivariate_normal` is
    within this range.

    >>> stats.multivariate_normal.cdf(b, mean, cov, lower_limit=a)
    0.00018430867675187443

    """
    args = _qmc_quad_iv(func, a, b, n_points, n_estimates, qrng, log)
    func, a, b, n_points, n_estimates, qrng, rng, log, stats = args

    def sum_product(integrands, dA, log=False):
        if log:
            return logsumexp(integrands) + np.log(dA)
        else:
            return np.sum(integrands * dA)

    def mean(estimates, log=False):
        if log:
            return logsumexp(estimates) - np.log(n_estimates)
        else:
            return np.mean(estimates)

    def std(estimates, m=None, ddof=0, log=False):
        m = m or mean(estimates, log)
        if log:
            estimates, m = np.broadcast_arrays(estimates, m)
            temp = np.vstack((estimates, m + np.pi * 1j))
            diff = logsumexp(temp, axis=0)
            return np.real(0.5 * (logsumexp(2 * diff)
                                  - np.log(n_estimates - ddof)))
        else:
            return np.std(estimates, ddof=ddof)

    def sem(estimates, m=None, s=None, log=False):
        m = m or mean(estimates, log)
        s = s or std(estimates, m, ddof=1, log=log)
        if log:
            return s - 0.5*np.log(n_estimates)
        else:
            return s / np.sqrt(n_estimates)

    # The sign of the integral depends on the order of the limits. Fix this by
    # ensuring that lower bounds are indeed lower and setting sign of resulting
    # integral manually
    if np.any(a == b):
        message = ("A lower limit was equal to an upper limit, so the value "
                   "of the integral is zero by definition.")
        warnings.warn(message, stacklevel=2)
        return QMCQuadResult(-np.inf if log else 0, 0)

    i_swap = b < a
    sign = (-1)**(i_swap.sum(axis=-1))  # odd # of swaps -> negative
    a[i_swap], b[i_swap] = b[i_swap], a[i_swap]

    A = np.prod(b - a)
    dA = A / n_points

    estimates = np.zeros(n_estimates)
    rngs = _rng_spawn(qrng.rng, n_estimates)
    for i in range(n_estimates):
        # Generate integral estimate
        sample = qrng.random(n_points)
        # The rationale for transposing is that this allows users to easily
        # unpack `x` into separate variables, if desired. This is consistent
        # with the `xx` array passed into the `scipy.integrate.nquad` `func`.
        x = stats.qmc.scale(sample, a, b).T  # (n_dim, n_points)
        integrands = func(x)
        estimates[i] = sum_product(integrands, dA, log)

        # Get a new, independently-scrambled QRNG for next time
        qrng = type(qrng)(seed=rngs[i], **qrng._init_quad)

    integral = mean(estimates, log)
    standard_error = sem(estimates, m=integral, log=log)
    integral = integral + np.pi*1j if (log and sign < 0) else integral*sign
    return QMCQuadResult(integral, standard_error)
