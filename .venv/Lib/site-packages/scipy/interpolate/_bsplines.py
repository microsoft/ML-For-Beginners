import operator
from math import prod

import numpy as np
from scipy._lib._util import normalize_axis_index
from scipy.linalg import (get_lapack_funcs, LinAlgError,
                          cholesky_banded, cho_solve_banded,
                          solve, solve_banded)
from scipy.optimize import minimize_scalar
from . import _bspl
from . import _fitpack_impl
from scipy.sparse import csr_array
from scipy.special import poch
from itertools import combinations

__all__ = ["BSpline", "make_interp_spline", "make_lsq_spline",
           "make_smoothing_spline"]


def _get_dtype(dtype):
    """Return np.complex128 for complex dtypes, np.float64 otherwise."""
    if np.issubdtype(dtype, np.complexfloating):
        return np.complex128
    else:
        return np.float64


def _as_float_array(x, check_finite=False):
    """Convert the input into a C contiguous float array.

    NB: Upcasts half- and single-precision floats to double precision.
    """
    x = np.ascontiguousarray(x)
    dtyp = _get_dtype(x.dtype)
    x = x.astype(dtyp, copy=False)
    if check_finite and not np.isfinite(x).all():
        raise ValueError("Array must not contain infs or nans.")
    return x


def _dual_poly(j, k, t, y):
    """
    Dual polynomial of the B-spline B_{j,k,t} -
    polynomial which is associated with B_{j,k,t}:
    $p_{j,k}(y) = (y - t_{j+1})(y - t_{j+2})...(y - t_{j+k})$
    """
    if k == 0:
        return 1
    return np.prod([(y - t[j + i]) for i in range(1, k + 1)])


def _diff_dual_poly(j, k, y, d, t):
    """
    d-th derivative of the dual polynomial $p_{j,k}(y)$
    """
    if d == 0:
        return _dual_poly(j, k, t, y)
    if d == k:
        return poch(1, k)
    comb = list(combinations(range(j + 1, j + k + 1), d))
    res = 0
    for i in range(len(comb) * len(comb[0])):
        res += np.prod([(y - t[j + p]) for p in range(1, k + 1)
                        if (j + p) not in comb[i//d]])
    return res


class BSpline:
    r"""Univariate spline in the B-spline basis.

    .. math::

        S(x) = \sum_{j=0}^{n-1} c_j  B_{j, k; t}(x)

    where :math:`B_{j, k; t}` are B-spline basis functions of degree `k`
    and knots `t`.

    Parameters
    ----------
    t : ndarray, shape (n+k+1,)
        knots
    c : ndarray, shape (>=n, ...)
        spline coefficients
    k : int
        B-spline degree
    extrapolate : bool or 'periodic', optional
        whether to extrapolate beyond the base interval, ``t[k] .. t[n]``,
        or to return nans.
        If True, extrapolates the first and last polynomial pieces of b-spline
        functions active on the base interval.
        If 'periodic', periodic extrapolation is used.
        Default is True.
    axis : int, optional
        Interpolation axis. Default is zero.

    Attributes
    ----------
    t : ndarray
        knot vector
    c : ndarray
        spline coefficients
    k : int
        spline degree
    extrapolate : bool
        If True, extrapolates the first and last polynomial pieces of b-spline
        functions active on the base interval.
    axis : int
        Interpolation axis.
    tck : tuple
        A read-only equivalent of ``(self.t, self.c, self.k)``

    Methods
    -------
    __call__
    basis_element
    derivative
    antiderivative
    integrate
    construct_fast
    design_matrix
    from_power_basis

    Notes
    -----
    B-spline basis elements are defined via

    .. math::

        B_{i, 0}(x) = 1, \textrm{if $t_i \le x < t_{i+1}$, otherwise $0$,}

        B_{i, k}(x) = \frac{x - t_i}{t_{i+k} - t_i} B_{i, k-1}(x)
                 + \frac{t_{i+k+1} - x}{t_{i+k+1} - t_{i+1}} B_{i+1, k-1}(x)

    **Implementation details**

    - At least ``k+1`` coefficients are required for a spline of degree `k`,
      so that ``n >= k+1``. Additional coefficients, ``c[j]`` with
      ``j > n``, are ignored.

    - B-spline basis elements of degree `k` form a partition of unity on the
      *base interval*, ``t[k] <= x <= t[n]``.


    Examples
    --------

    Translating the recursive definition of B-splines into Python code, we have:

    >>> def B(x, k, i, t):
    ...    if k == 0:
    ...       return 1.0 if t[i] <= x < t[i+1] else 0.0
    ...    if t[i+k] == t[i]:
    ...       c1 = 0.0
    ...    else:
    ...       c1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t)
    ...    if t[i+k+1] == t[i+1]:
    ...       c2 = 0.0
    ...    else:
    ...       c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t)
    ...    return c1 + c2

    >>> def bspline(x, t, c, k):
    ...    n = len(t) - k - 1
    ...    assert (n >= k+1) and (len(c) >= n)
    ...    return sum(c[i] * B(x, k, i, t) for i in range(n))

    Note that this is an inefficient (if straightforward) way to
    evaluate B-splines --- this spline class does it in an equivalent,
    but much more efficient way.

    Here we construct a quadratic spline function on the base interval
    ``2 <= x <= 4`` and compare with the naive way of evaluating the spline:

    >>> from scipy.interpolate import BSpline
    >>> k = 2
    >>> t = [0, 1, 2, 3, 4, 5, 6]
    >>> c = [-1, 2, 0, -1]
    >>> spl = BSpline(t, c, k)
    >>> spl(2.5)
    array(1.375)
    >>> bspline(2.5, t, c, k)
    1.375

    Note that outside of the base interval results differ. This is because
    `BSpline` extrapolates the first and last polynomial pieces of B-spline
    functions active on the base interval.

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> fig, ax = plt.subplots()
    >>> xx = np.linspace(1.5, 4.5, 50)
    >>> ax.plot(xx, [bspline(x, t, c ,k) for x in xx], 'r-', lw=3, label='naive')
    >>> ax.plot(xx, spl(xx), 'b-', lw=4, alpha=0.7, label='BSpline')
    >>> ax.grid(True)
    >>> ax.legend(loc='best')
    >>> plt.show()


    References
    ----------
    .. [1] Tom Lyche and Knut Morken, Spline methods,
        http://www.uio.no/studier/emner/matnat/ifi/INF-MAT5340/v05/undervisningsmateriale/
    .. [2] Carl de Boor, A practical guide to splines, Springer, 2001.

    """

    def __init__(self, t, c, k, extrapolate=True, axis=0):
        super().__init__()

        self.k = operator.index(k)
        self.c = np.asarray(c)
        self.t = np.ascontiguousarray(t, dtype=np.float64)

        if extrapolate == 'periodic':
            self.extrapolate = extrapolate
        else:
            self.extrapolate = bool(extrapolate)

        n = self.t.shape[0] - self.k - 1

        axis = normalize_axis_index(axis, self.c.ndim)

        # Note that the normalized axis is stored in the object.
        self.axis = axis
        if axis != 0:
            # roll the interpolation axis to be the first one in self.c
            # More specifically, the target shape for self.c is (n, ...),
            # and axis !=0 means that we have c.shape (..., n, ...)
            #                                               ^
            #                                              axis
            self.c = np.moveaxis(self.c, axis, 0)

        if k < 0:
            raise ValueError("Spline order cannot be negative.")
        if self.t.ndim != 1:
            raise ValueError("Knot vector must be one-dimensional.")
        if n < self.k + 1:
            raise ValueError("Need at least %d knots for degree %d" %
                             (2*k + 2, k))
        if (np.diff(self.t) < 0).any():
            raise ValueError("Knots must be in a non-decreasing order.")
        if len(np.unique(self.t[k:n+1])) < 2:
            raise ValueError("Need at least two internal knots.")
        if not np.isfinite(self.t).all():
            raise ValueError("Knots should not have nans or infs.")
        if self.c.ndim < 1:
            raise ValueError("Coefficients must be at least 1-dimensional.")
        if self.c.shape[0] < n:
            raise ValueError("Knots, coefficients and degree are inconsistent.")

        dt = _get_dtype(self.c.dtype)
        self.c = np.ascontiguousarray(self.c, dtype=dt)

    @classmethod
    def construct_fast(cls, t, c, k, extrapolate=True, axis=0):
        """Construct a spline without making checks.

        Accepts same parameters as the regular constructor. Input arrays
        `t` and `c` must of correct shape and dtype.
        """
        self = object.__new__(cls)
        self.t, self.c, self.k = t, c, k
        self.extrapolate = extrapolate
        self.axis = axis
        return self

    @property
    def tck(self):
        """Equivalent to ``(self.t, self.c, self.k)`` (read-only).
        """
        return self.t, self.c, self.k

    @classmethod
    def basis_element(cls, t, extrapolate=True):
        """Return a B-spline basis element ``B(x | t[0], ..., t[k+1])``.

        Parameters
        ----------
        t : ndarray, shape (k+2,)
            internal knots
        extrapolate : bool or 'periodic', optional
            whether to extrapolate beyond the base interval, ``t[0] .. t[k+1]``,
            or to return nans.
            If 'periodic', periodic extrapolation is used.
            Default is True.

        Returns
        -------
        basis_element : callable
            A callable representing a B-spline basis element for the knot
            vector `t`.

        Notes
        -----
        The degree of the B-spline, `k`, is inferred from the length of `t` as
        ``len(t)-2``. The knot vector is constructed by appending and prepending
        ``k+1`` elements to internal knots `t`.

        Examples
        --------

        Construct a cubic B-spline:

        >>> import numpy as np
        >>> from scipy.interpolate import BSpline
        >>> b = BSpline.basis_element([0, 1, 2, 3, 4])
        >>> k = b.k
        >>> b.t[k:-k]
        array([ 0.,  1.,  2.,  3.,  4.])
        >>> k
        3

        Construct a quadratic B-spline on ``[0, 1, 1, 2]``, and compare
        to its explicit form:

        >>> t = [0, 1, 1, 2]
        >>> b = BSpline.basis_element(t)
        >>> def f(x):
        ...     return np.where(x < 1, x*x, (2. - x)**2)

        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> x = np.linspace(0, 2, 51)
        >>> ax.plot(x, b(x), 'g', lw=3)
        >>> ax.plot(x, f(x), 'r', lw=8, alpha=0.4)
        >>> ax.grid(True)
        >>> plt.show()

        """
        k = len(t) - 2
        t = _as_float_array(t)
        t = np.r_[(t[0]-1,) * k, t, (t[-1]+1,) * k]
        c = np.zeros_like(t)
        c[k] = 1.
        return cls.construct_fast(t, c, k, extrapolate)

    @classmethod
    def design_matrix(cls, x, t, k, extrapolate=False):
        """
        Returns a design matrix as a CSR format sparse array.

        Parameters
        ----------
        x : array_like, shape (n,)
            Points to evaluate the spline at.
        t : array_like, shape (nt,)
            Sorted 1D array of knots.
        k : int
            B-spline degree.
        extrapolate : bool or 'periodic', optional
            Whether to extrapolate based on the first and last intervals
            or raise an error. If 'periodic', periodic extrapolation is used.
            Default is False.

            .. versionadded:: 1.10.0

        Returns
        -------
        design_matrix : `csr_array` object
            Sparse matrix in CSR format where each row contains all the basis
            elements of the input row (first row = basis elements of x[0],
            ..., last row = basis elements x[-1]).

        Examples
        --------
        Construct a design matrix for a B-spline

        >>> from scipy.interpolate import make_interp_spline, BSpline
        >>> import numpy as np
        >>> x = np.linspace(0, np.pi * 2, 4)
        >>> y = np.sin(x)
        >>> k = 3
        >>> bspl = make_interp_spline(x, y, k=k)
        >>> design_matrix = bspl.design_matrix(x, bspl.t, k)
        >>> design_matrix.toarray()
        [[1.        , 0.        , 0.        , 0.        ],
        [0.2962963 , 0.44444444, 0.22222222, 0.03703704],
        [0.03703704, 0.22222222, 0.44444444, 0.2962963 ],
        [0.        , 0.        , 0.        , 1.        ]]

        Construct a design matrix for some vector of knots

        >>> k = 2
        >>> t = [-1, 0, 1, 2, 3, 4, 5, 6]
        >>> x = [1, 2, 3, 4]
        >>> design_matrix = BSpline.design_matrix(x, t, k).toarray()
        >>> design_matrix
        [[0.5, 0.5, 0. , 0. , 0. ],
        [0. , 0.5, 0.5, 0. , 0. ],
        [0. , 0. , 0.5, 0.5, 0. ],
        [0. , 0. , 0. , 0.5, 0.5]]

        This result is equivalent to the one created in the sparse format

        >>> c = np.eye(len(t) - k - 1)
        >>> design_matrix_gh = BSpline(t, c, k)(x)
        >>> np.allclose(design_matrix, design_matrix_gh, atol=1e-14)
        True

        Notes
        -----
        .. versionadded:: 1.8.0

        In each row of the design matrix all the basis elements are evaluated
        at the certain point (first row - x[0], ..., last row - x[-1]).

        `nt` is a length of the vector of knots: as far as there are
        `nt - k - 1` basis elements, `nt` should be not less than `2 * k + 2`
        to have at least `k + 1` basis element.

        Out of bounds `x` raises a ValueError.
        """
        x = _as_float_array(x, True)
        t = _as_float_array(t, True)

        if extrapolate != 'periodic':
            extrapolate = bool(extrapolate)

        if k < 0:
            raise ValueError("Spline order cannot be negative.")
        if t.ndim != 1 or np.any(t[1:] < t[:-1]):
            raise ValueError(f"Expect t to be a 1-D sorted array_like, but "
                             f"got t={t}.")
        # There are `nt - k - 1` basis elements in a BSpline built on the
        # vector of knots with length `nt`, so to have at least `k + 1` basis
        # elements we need to have at least `2 * k + 2` elements in the vector
        # of knots.
        if len(t) < 2 * k + 2:
            raise ValueError(f"Length t is not enough for k={k}.")

        if extrapolate == 'periodic':
            # With periodic extrapolation we map x to the segment
            # [t[k], t[n]].
            n = t.size - k - 1
            x = t[k] + (x - t[k]) % (t[n] - t[k])
            extrapolate = False
        elif not extrapolate and (
            (min(x) < t[k]) or (max(x) > t[t.shape[0] - k - 1])
        ):
            # Checks from `find_interval` function
            raise ValueError(f'Out of bounds w/ x = {x}.')

        # Compute number of non-zeros of final CSR array in order to determine
        # the dtype of indices and indptr of the CSR array.
        n = x.shape[0]
        nnz = n * (k + 1)
        if nnz < np.iinfo(np.int32).max:
            int_dtype = np.int32
        else:
            int_dtype = np.int64
        # Preallocate indptr and indices
        indices = np.empty(n * (k + 1), dtype=int_dtype)
        indptr = np.arange(0, (n + 1) * (k + 1), k + 1, dtype=int_dtype)

        # indptr is not passed to Cython as it is already fully computed
        data, indices = _bspl._make_design_matrix(
            x, t, k, extrapolate, indices
        )
        return csr_array(
            (data, indices, indptr),
            shape=(x.shape[0], t.shape[0] - k - 1)
        )

    def __call__(self, x, nu=0, extrapolate=None):
        """
        Evaluate a spline function.

        Parameters
        ----------
        x : array_like
            points to evaluate the spline at.
        nu : int, optional
            derivative to evaluate (default is 0).
        extrapolate : bool or 'periodic', optional
            whether to extrapolate based on the first and last intervals
            or return nans. If 'periodic', periodic extrapolation is used.
            Default is `self.extrapolate`.

        Returns
        -------
        y : array_like
            Shape is determined by replacing the interpolation axis
            in the coefficient array with the shape of `x`.

        """
        if extrapolate is None:
            extrapolate = self.extrapolate
        x = np.asarray(x)
        x_shape, x_ndim = x.shape, x.ndim
        x = np.ascontiguousarray(x.ravel(), dtype=np.float64)

        # With periodic extrapolation we map x to the segment
        # [self.t[k], self.t[n]].
        if extrapolate == 'periodic':
            n = self.t.size - self.k - 1
            x = self.t[self.k] + (x - self.t[self.k]) % (self.t[n] -
                                                         self.t[self.k])
            extrapolate = False

        out = np.empty((len(x), prod(self.c.shape[1:])), dtype=self.c.dtype)
        self._ensure_c_contiguous()
        self._evaluate(x, nu, extrapolate, out)
        out = out.reshape(x_shape + self.c.shape[1:])
        if self.axis != 0:
            # transpose to move the calculated values to the interpolation axis
            l = list(range(out.ndim))
            l = l[x_ndim:x_ndim+self.axis] + l[:x_ndim] + l[x_ndim+self.axis:]
            out = out.transpose(l)
        return out

    def _evaluate(self, xp, nu, extrapolate, out):
        _bspl.evaluate_spline(self.t, self.c.reshape(self.c.shape[0], -1),
                              self.k, xp, nu, extrapolate, out)

    def _ensure_c_contiguous(self):
        """
        c and t may be modified by the user. The Cython code expects
        that they are C contiguous.

        """
        if not self.t.flags.c_contiguous:
            self.t = self.t.copy()
        if not self.c.flags.c_contiguous:
            self.c = self.c.copy()

    def derivative(self, nu=1):
        """Return a B-spline representing the derivative.

        Parameters
        ----------
        nu : int, optional
            Derivative order.
            Default is 1.

        Returns
        -------
        b : BSpline object
            A new instance representing the derivative.

        See Also
        --------
        splder, splantider

        """
        c = self.c
        # pad the c array if needed
        ct = len(self.t) - len(c)
        if ct > 0:
            c = np.r_[c, np.zeros((ct,) + c.shape[1:])]
        tck = _fitpack_impl.splder((self.t, c, self.k), nu)
        return self.construct_fast(*tck, extrapolate=self.extrapolate,
                                   axis=self.axis)

    def antiderivative(self, nu=1):
        """Return a B-spline representing the antiderivative.

        Parameters
        ----------
        nu : int, optional
            Antiderivative order. Default is 1.

        Returns
        -------
        b : BSpline object
            A new instance representing the antiderivative.

        Notes
        -----
        If antiderivative is computed and ``self.extrapolate='periodic'``,
        it will be set to False for the returned instance. This is done because
        the antiderivative is no longer periodic and its correct evaluation
        outside of the initially given x interval is difficult.

        See Also
        --------
        splder, splantider

        """
        c = self.c
        # pad the c array if needed
        ct = len(self.t) - len(c)
        if ct > 0:
            c = np.r_[c, np.zeros((ct,) + c.shape[1:])]
        tck = _fitpack_impl.splantider((self.t, c, self.k), nu)

        if self.extrapolate == 'periodic':
            extrapolate = False
        else:
            extrapolate = self.extrapolate

        return self.construct_fast(*tck, extrapolate=extrapolate,
                                   axis=self.axis)

    def integrate(self, a, b, extrapolate=None):
        """Compute a definite integral of the spline.

        Parameters
        ----------
        a : float
            Lower limit of integration.
        b : float
            Upper limit of integration.
        extrapolate : bool or 'periodic', optional
            whether to extrapolate beyond the base interval,
            ``t[k] .. t[-k-1]``, or take the spline to be zero outside of the
            base interval. If 'periodic', periodic extrapolation is used.
            If None (default), use `self.extrapolate`.

        Returns
        -------
        I : array_like
            Definite integral of the spline over the interval ``[a, b]``.

        Examples
        --------
        Construct the linear spline ``x if x < 1 else 2 - x`` on the base
        interval :math:`[0, 2]`, and integrate it

        >>> from scipy.interpolate import BSpline
        >>> b = BSpline.basis_element([0, 1, 2])
        >>> b.integrate(0, 1)
        array(0.5)

        If the integration limits are outside of the base interval, the result
        is controlled by the `extrapolate` parameter

        >>> b.integrate(-1, 1)
        array(0.0)
        >>> b.integrate(-1, 1, extrapolate=False)
        array(0.5)

        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> ax.grid(True)
        >>> ax.axvline(0, c='r', lw=5, alpha=0.5)  # base interval
        >>> ax.axvline(2, c='r', lw=5, alpha=0.5)
        >>> xx = [-1, 1, 2]
        >>> ax.plot(xx, b(xx))
        >>> plt.show()

        """
        if extrapolate is None:
            extrapolate = self.extrapolate

        # Prepare self.t and self.c.
        self._ensure_c_contiguous()

        # Swap integration bounds if needed.
        sign = 1
        if b < a:
            a, b = b, a
            sign = -1
        n = self.t.size - self.k - 1

        if extrapolate != "periodic" and not extrapolate:
            # Shrink the integration interval, if needed.
            a = max(a, self.t[self.k])
            b = min(b, self.t[n])

            if self.c.ndim == 1:
                # Fast path: use FITPACK's routine
                # (cf _fitpack_impl.splint).
                integral = _fitpack_impl.splint(a, b, self.tck)
                return integral * sign

        out = np.empty((2, prod(self.c.shape[1:])), dtype=self.c.dtype)

        # Compute the antiderivative.
        c = self.c
        ct = len(self.t) - len(c)
        if ct > 0:
            c = np.r_[c, np.zeros((ct,) + c.shape[1:])]
        ta, ca, ka = _fitpack_impl.splantider((self.t, c, self.k), 1)

        if extrapolate == 'periodic':
            # Split the integral into the part over period (can be several
            # of them) and the remaining part.

            ts, te = self.t[self.k], self.t[n]
            period = te - ts
            interval = b - a
            n_periods, left = divmod(interval, period)

            if n_periods > 0:
                # Evaluate the difference of antiderivatives.
                x = np.asarray([ts, te], dtype=np.float64)
                _bspl.evaluate_spline(ta, ca.reshape(ca.shape[0], -1),
                                      ka, x, 0, False, out)
                integral = out[1] - out[0]
                integral *= n_periods
            else:
                integral = np.zeros((1, prod(self.c.shape[1:])),
                                    dtype=self.c.dtype)

            # Map a to [ts, te], b is always a + left.
            a = ts + (a - ts) % period
            b = a + left

            # If b <= te then we need to integrate over [a, b], otherwise
            # over [a, te] and from xs to what is remained.
            if b <= te:
                x = np.asarray([a, b], dtype=np.float64)
                _bspl.evaluate_spline(ta, ca.reshape(ca.shape[0], -1),
                                      ka, x, 0, False, out)
                integral += out[1] - out[0]
            else:
                x = np.asarray([a, te], dtype=np.float64)
                _bspl.evaluate_spline(ta, ca.reshape(ca.shape[0], -1),
                                      ka, x, 0, False, out)
                integral += out[1] - out[0]

                x = np.asarray([ts, ts + b - te], dtype=np.float64)
                _bspl.evaluate_spline(ta, ca.reshape(ca.shape[0], -1),
                                      ka, x, 0, False, out)
                integral += out[1] - out[0]
        else:
            # Evaluate the difference of antiderivatives.
            x = np.asarray([a, b], dtype=np.float64)
            _bspl.evaluate_spline(ta, ca.reshape(ca.shape[0], -1),
                                  ka, x, 0, extrapolate, out)
            integral = out[1] - out[0]

        integral *= sign
        return integral.reshape(ca.shape[1:])

    @classmethod
    def from_power_basis(cls, pp, bc_type='not-a-knot'):
        r"""
        Construct a polynomial in the B-spline basis
        from a piecewise polynomial in the power basis.

        For now, accepts ``CubicSpline`` instances only.

        Parameters
        ----------
        pp : CubicSpline
            A piecewise polynomial in the power basis, as created
            by ``CubicSpline``
        bc_type : string, optional
            Boundary condition type as in ``CubicSpline``: one of the
            ``not-a-knot``, ``natural``, ``clamped``, or ``periodic``.
            Necessary for construction an instance of ``BSpline`` class.
            Default is ``not-a-knot``.

        Returns
        -------
        b : BSpline object
            A new instance representing the initial polynomial
            in the B-spline basis.

        Notes
        -----
        .. versionadded:: 1.8.0

        Accepts only ``CubicSpline`` instances for now.

        The algorithm follows from differentiation
        the Marsden's identity [1]: each of coefficients of spline
        interpolation function in the B-spline basis is computed as follows:

        .. math::

            c_j = \sum_{m=0}^{k} \frac{(k-m)!}{k!}
                       c_{m,i} (-1)^{k-m} D^m p_{j,k}(x_i)

        :math:`c_{m, i}` - a coefficient of CubicSpline,
        :math:`D^m p_{j, k}(x_i)` - an m-th defivative of a dual polynomial
        in :math:`x_i`.

        ``k`` always equals 3 for now.

        First ``n - 2`` coefficients are computed in :math:`x_i = x_j`, e.g.

        .. math::

            c_1 = \sum_{m=0}^{k} \frac{(k-1)!}{k!} c_{m,1} D^m p_{j,3}(x_1)

        Last ``nod + 2`` coefficients are computed in ``x[-2]``,
        ``nod`` - number of derivatives at the ends.

        For example, consider :math:`x = [0, 1, 2, 3, 4]`,
        :math:`y = [1, 1, 1, 1, 1]` and bc_type = ``natural``

        The coefficients of CubicSpline in the power basis:

        :math:`[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]`

        The knot vector: :math:`t = [0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4]`

        In this case

        .. math::

            c_j = \frac{0!}{k!} c_{3, i} k! = c_{3, i} = 1,~j = 0, ..., 6

        References
        ----------
        .. [1] Tom Lyche and Knut Morken, Spline Methods, 2005, Section 3.1.2

        """
        from ._cubic import CubicSpline
        if not isinstance(pp, CubicSpline):
            raise NotImplementedError("Only CubicSpline objects are accepted"
                                      "for now. Got %s instead." % type(pp))
        x = pp.x
        coef = pp.c
        k = pp.c.shape[0] - 1
        n = x.shape[0]

        if bc_type == 'not-a-knot':
            t = _not_a_knot(x, k)
        elif bc_type == 'natural' or bc_type == 'clamped':
            t = _augknt(x, k)
        elif bc_type == 'periodic':
            t = _periodic_knots(x, k)
        else:
            raise TypeError('Unknown boundary condition: %s' % bc_type)

        nod = t.shape[0] - (n + k + 1)  # number of derivatives at the ends
        c = np.zeros(n + nod, dtype=pp.c.dtype)
        for m in range(k + 1):
            for i in range(n - 2):
                c[i] += poch(k + 1, -m) * coef[m, i]\
                        * np.power(-1, k - m)\
                        * _diff_dual_poly(i, k, x[i], m, t)
            for j in range(n - 2, n + nod):
                c[j] += poch(k + 1, -m) * coef[m, n - 2]\
                        * np.power(-1, k - m)\
                        * _diff_dual_poly(j, k, x[n - 2], m, t)
        return cls.construct_fast(t, c, k, pp.extrapolate, pp.axis)


#################################
#  Interpolating spline helpers #
#################################

def _not_a_knot(x, k):
    """Given data x, construct the knot vector w/ not-a-knot BC.
    cf de Boor, XIII(12)."""
    x = np.asarray(x)
    if k % 2 != 1:
        raise ValueError("Odd degree for now only. Got %s." % k)

    m = (k - 1) // 2
    t = x[m+1:-m-1]
    t = np.r_[(x[0],)*(k+1), t, (x[-1],)*(k+1)]
    return t


def _augknt(x, k):
    """Construct a knot vector appropriate for the order-k interpolation."""
    return np.r_[(x[0],)*k, x, (x[-1],)*k]


def _convert_string_aliases(deriv, target_shape):
    if isinstance(deriv, str):
        if deriv == "clamped":
            deriv = [(1, np.zeros(target_shape))]
        elif deriv == "natural":
            deriv = [(2, np.zeros(target_shape))]
        else:
            raise ValueError("Unknown boundary condition : %s" % deriv)
    return deriv


def _process_deriv_spec(deriv):
    if deriv is not None:
        try:
            ords, vals = zip(*deriv)
        except TypeError as e:
            msg = ("Derivatives, `bc_type`, should be specified as a pair of "
                   "iterables of pairs of (order, value).")
            raise ValueError(msg) from e
    else:
        ords, vals = [], []
    return np.atleast_1d(ords, vals)


def _woodbury_algorithm(A, ur, ll, b, k):
    '''
    Solve a cyclic banded linear system with upper right
    and lower blocks of size ``(k-1) / 2`` using
    the Woodbury formula

    Parameters
    ----------
    A : 2-D array, shape(k, n)
        Matrix of diagonals of original matrix (see
        ``solve_banded`` documentation).
    ur : 2-D array, shape(bs, bs)
        Upper right block matrix.
    ll : 2-D array, shape(bs, bs)
        Lower left block matrix.
    b : 1-D array, shape(n,)
        Vector of constant terms of the system of linear equations.
    k : int
        B-spline degree.

    Returns
    -------
    c : 1-D array, shape(n,)
        Solution of the original system of linear equations.

    Notes
    -----
    This algorithm works only for systems with banded matrix A plus
    a correction term U @ V.T, where the matrix U @ V.T gives upper right
    and lower left block of A
    The system is solved with the following steps:
        1.  New systems of linear equations are constructed:
            A @ z_i = u_i,
            u_i - column vector of U,
            i = 1, ..., k - 1
        2.  Matrix Z is formed from vectors z_i:
            Z = [ z_1 | z_2 | ... | z_{k - 1} ]
        3.  Matrix H = (1 + V.T @ Z)^{-1}
        4.  The system A' @ y = b is solved
        5.  x = y - Z @ (H @ V.T @ y)
    Also, ``n`` should be greater than ``k``, otherwise corner block
    elements will intersect with diagonals.

    Examples
    --------
    Consider the case of n = 8, k = 5 (size of blocks - 2 x 2).
    The matrix of a system:       U:          V:
      x  x  x  *  *  a  b         a b 0 0     0 0 1 0
      x  x  x  x  *  *  c         0 c 0 0     0 0 0 1
      x  x  x  x  x  *  *         0 0 0 0     0 0 0 0
      *  x  x  x  x  x  *         0 0 0 0     0 0 0 0
      *  *  x  x  x  x  x         0 0 0 0     0 0 0 0
      d  *  *  x  x  x  x         0 0 d 0     1 0 0 0
      e  f  *  *  x  x  x         0 0 e f     0 1 0 0

    References
    ----------
    .. [1] William H. Press, Saul A. Teukolsky, William T. Vetterling
           and Brian P. Flannery, Numerical Recipes, 2007, Section 2.7.3

    '''
    k_mod = k - k % 2
    bs = int((k - 1) / 2) + (k + 1) % 2

    n = A.shape[1] + 1
    U = np.zeros((n - 1, k_mod))
    VT = np.zeros((k_mod, n - 1))  # V transpose

    # upper right block
    U[:bs, :bs] = ur
    VT[np.arange(bs), np.arange(bs) - bs] = 1

    # lower left block
    U[-bs:, -bs:] = ll
    VT[np.arange(bs) - bs, np.arange(bs)] = 1

    Z = solve_banded((bs, bs), A, U)

    H = solve(np.identity(k_mod) + VT @ Z, np.identity(k_mod))

    y = solve_banded((bs, bs), A, b)
    c = y - Z @ (H @ (VT @ y))

    return c


def _periodic_knots(x, k):
    '''
    returns vector of nodes on circle
    '''
    xc = np.copy(x)
    n = len(xc)
    if k % 2 == 0:
        dx = np.diff(xc)
        xc[1: -1] -= dx[:-1] / 2
    dx = np.diff(xc)
    t = np.zeros(n + 2 * k)
    t[k: -k] = xc
    for i in range(0, k):
        # filling first `k` elements in descending order
        t[k - i - 1] = t[k - i] - dx[-(i % (n - 1)) - 1]
        # filling last `k` elements in ascending order
        t[-k + i] = t[-k + i - 1] + dx[i % (n - 1)]
    return t


def _make_interp_per_full_matr(x, y, t, k):
    '''
    Returns a solution of a system for B-spline interpolation with periodic
    boundary conditions. First ``k - 1`` rows of matrix are conditions of
    periodicity (continuity of ``k - 1`` derivatives at the boundary points).
    Last ``n`` rows are interpolation conditions.
    RHS is ``k - 1`` zeros and ``n`` ordinates in this case.

    Parameters
    ----------
    x : 1-D array, shape (n,)
        Values of x - coordinate of a given set of points.
    y : 1-D array, shape (n,)
        Values of y - coordinate of a given set of points.
    t : 1-D array, shape(n+2*k,)
        Vector of knots.
    k : int
        The maximum degree of spline

    Returns
    -------
    c : 1-D array, shape (n+k-1,)
        B-spline coefficients

    Notes
    -----
    ``t`` is supposed to be taken on circle.

    '''

    x, y, t = map(np.asarray, (x, y, t))

    n = x.size
    # LHS: the collocation matrix + derivatives at edges
    matr = np.zeros((n + k - 1, n + k - 1))

    # derivatives at x[0] and x[-1]:
    for i in range(k - 1):
        bb = _bspl.evaluate_all_bspl(t, k, x[0], k, nu=i + 1)
        matr[i, : k + 1] += bb
        bb = _bspl.evaluate_all_bspl(t, k, x[-1], n + k - 1, nu=i + 1)[:-1]
        matr[i, -k:] -= bb

    # collocation matrix
    for i in range(n):
        xval = x[i]
        # find interval
        if xval == t[k]:
            left = k
        else:
            left = np.searchsorted(t, xval) - 1

        # fill a row
        bb = _bspl.evaluate_all_bspl(t, k, xval, left)
        matr[i + k - 1, left-k:left+1] = bb

    # RHS
    b = np.r_[[0] * (k - 1), y]

    c = solve(matr, b)
    return c


def _make_periodic_spline(x, y, t, k, axis):
    '''
    Compute the (coefficients of) interpolating B-spline with periodic
    boundary conditions.

    Parameters
    ----------
    x : array_like, shape (n,)
        Abscissas.
    y : array_like, shape (n,)
        Ordinates.
    k : int
        B-spline degree.
    t : array_like, shape (n + 2 * k,).
        Knots taken on a circle, ``k`` on the left and ``k`` on the right
        of the vector ``x``.

    Returns
    -------
    b : a BSpline object of the degree ``k`` and with knots ``t``.

    Notes
    -----
    The original system is formed by ``n + k - 1`` equations where the first
    ``k - 1`` of them stand for the ``k - 1`` derivatives continuity on the
    edges while the other equations correspond to an interpolating case
    (matching all the input points). Due to a special form of knot vector, it
    can be proved that in the original system the first and last ``k``
    coefficients of a spline function are the same, respectively. It follows
    from the fact that all ``k - 1`` derivatives are equal term by term at ends
    and that the matrix of the original system of linear equations is
    non-degenerate. So, we can reduce the number of equations to ``n - 1``
    (first ``k - 1`` equations could be reduced). Another trick of this
    implementation is cyclic shift of values of B-splines due to equality of
    ``k`` unknown coefficients. With this we can receive matrix of the system
    with upper right and lower left blocks, and ``k`` diagonals.  It allows
    to use Woodbury formula to optimize the computations.

    '''
    n = y.shape[0]

    extradim = prod(y.shape[1:])
    y_new = y.reshape(n, extradim)
    c = np.zeros((n + k - 1, extradim))

    # n <= k case is solved with full matrix
    if n <= k:
        for i in range(extradim):
            c[:, i] = _make_interp_per_full_matr(x, y_new[:, i], t, k)
        c = np.ascontiguousarray(c.reshape((n + k - 1,) + y.shape[1:]))
        return BSpline.construct_fast(t, c, k, extrapolate='periodic', axis=axis)

    nt = len(t) - k - 1

    # size of block elements
    kul = int(k / 2)

    # kl = ku = k
    ab = np.zeros((3 * k + 1, nt), dtype=np.float64, order='F')

    # upper right and lower left blocks
    ur = np.zeros((kul, kul))
    ll = np.zeros_like(ur)

    # `offset` is made to shift all the non-zero elements to the end of the
    # matrix
    _bspl._colloc(x, t, k, ab, offset=k)

    # remove zeros before the matrix
    ab = ab[-k - (k + 1) % 2:, :]

    # The least elements in rows (except repetitions) are diagonals
    # of block matrices. Upper right matrix is an upper triangular
    # matrix while lower left is a lower triangular one.
    for i in range(kul):
        ur += np.diag(ab[-i - 1, i: kul], k=i)
        ll += np.diag(ab[i, -kul - (k % 2): n - 1 + 2 * kul - i], k=-i)

    # remove elements that occur in the last point
    # (first and last points are equivalent)
    A = ab[:, kul: -k + kul]

    for i in range(extradim):
        cc = _woodbury_algorithm(A, ur, ll, y_new[:, i][:-1], k)
        c[:, i] = np.concatenate((cc[-kul:], cc, cc[:kul + k % 2]))
    c = np.ascontiguousarray(c.reshape((n + k - 1,) + y.shape[1:]))
    return BSpline.construct_fast(t, c, k, extrapolate='periodic', axis=axis)


def make_interp_spline(x, y, k=3, t=None, bc_type=None, axis=0,
                       check_finite=True):
    """Compute the (coefficients of) interpolating B-spline.

    Parameters
    ----------
    x : array_like, shape (n,)
        Abscissas.
    y : array_like, shape (n, ...)
        Ordinates.
    k : int, optional
        B-spline degree. Default is cubic, ``k = 3``.
    t : array_like, shape (nt + k + 1,), optional.
        Knots.
        The number of knots needs to agree with the number of data points and
        the number of derivatives at the edges. Specifically, ``nt - n`` must
        equal ``len(deriv_l) + len(deriv_r)``.
    bc_type : 2-tuple or None
        Boundary conditions.
        Default is None, which means choosing the boundary conditions
        automatically. Otherwise, it must be a length-two tuple where the first
        element (``deriv_l``) sets the boundary conditions at ``x[0]`` and
        the second element (``deriv_r``) sets the boundary conditions at
        ``x[-1]``. Each of these must be an iterable of pairs
        ``(order, value)`` which gives the values of derivatives of specified
        orders at the given edge of the interpolation interval.
        Alternatively, the following string aliases are recognized:

        * ``"clamped"``: The first derivatives at the ends are zero. This is
           equivalent to ``bc_type=([(1, 0.0)], [(1, 0.0)])``.
        * ``"natural"``: The second derivatives at ends are zero. This is
          equivalent to ``bc_type=([(2, 0.0)], [(2, 0.0)])``.
        * ``"not-a-knot"`` (default): The first and second segments are the
          same polynomial. This is equivalent to having ``bc_type=None``.
        * ``"periodic"``: The values and the first ``k-1`` derivatives at the
          ends are equivalent.

    axis : int, optional
        Interpolation axis. Default is 0.
    check_finite : bool, optional
        Whether to check that the input arrays contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
        Default is True.

    Returns
    -------
    b : a BSpline object of the degree ``k`` and with knots ``t``.

    See Also
    --------
    BSpline : base class representing the B-spline objects
    CubicSpline : a cubic spline in the polynomial basis
    make_lsq_spline : a similar factory function for spline fitting
    UnivariateSpline : a wrapper over FITPACK spline fitting routines
    splrep : a wrapper over FITPACK spline fitting routines

    Examples
    --------

    Use cubic interpolation on Chebyshev nodes:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> def cheb_nodes(N):
    ...     jj = 2.*np.arange(N) + 1
    ...     x = np.cos(np.pi * jj / 2 / N)[::-1]
    ...     return x

    >>> x = cheb_nodes(20)
    >>> y = np.sqrt(1 - x**2)

    >>> from scipy.interpolate import BSpline, make_interp_spline
    >>> b = make_interp_spline(x, y)
    >>> np.allclose(b(x), y)
    True

    Note that the default is a cubic spline with a not-a-knot boundary condition

    >>> b.k
    3

    Here we use a 'natural' spline, with zero 2nd derivatives at edges:

    >>> l, r = [(2, 0.0)], [(2, 0.0)]
    >>> b_n = make_interp_spline(x, y, bc_type=(l, r))  # or, bc_type="natural"
    >>> np.allclose(b_n(x), y)
    True
    >>> x0, x1 = x[0], x[-1]
    >>> np.allclose([b_n(x0, 2), b_n(x1, 2)], [0, 0])
    True

    Interpolation of parametric curves is also supported. As an example, we
    compute a discretization of a snail curve in polar coordinates

    >>> phi = np.linspace(0, 2.*np.pi, 40)
    >>> r = 0.3 + np.cos(phi)
    >>> x, y = r*np.cos(phi), r*np.sin(phi)  # convert to Cartesian coordinates

    Build an interpolating curve, parameterizing it by the angle

    >>> spl = make_interp_spline(phi, np.c_[x, y])

    Evaluate the interpolant on a finer grid (note that we transpose the result
    to unpack it into a pair of x- and y-arrays)

    >>> phi_new = np.linspace(0, 2.*np.pi, 100)
    >>> x_new, y_new = spl(phi_new).T

    Plot the result

    >>> plt.plot(x, y, 'o')
    >>> plt.plot(x_new, y_new, '-')
    >>> plt.show()

    Build a B-spline curve with 2 dimensional y

    >>> x = np.linspace(0, 2*np.pi, 10)
    >>> y = np.array([np.sin(x), np.cos(x)])

    Periodic condition is satisfied because y coordinates of points on the ends
    are equivalent

    >>> ax = plt.axes(projection='3d')
    >>> xx = np.linspace(0, 2*np.pi, 100)
    >>> bspl = make_interp_spline(x, y, k=5, bc_type='periodic', axis=1)
    >>> ax.plot3D(xx, *bspl(xx))
    >>> ax.scatter3D(x, *y, color='red')
    >>> plt.show()

    """
    # convert string aliases for the boundary conditions
    if bc_type is None or bc_type == 'not-a-knot' or bc_type == 'periodic':
        deriv_l, deriv_r = None, None
    elif isinstance(bc_type, str):
        deriv_l, deriv_r = bc_type, bc_type
    else:
        try:
            deriv_l, deriv_r = bc_type
        except TypeError as e:
            raise ValueError("Unknown boundary condition: %s" % bc_type) from e

    y = np.asarray(y)

    axis = normalize_axis_index(axis, y.ndim)

    x = _as_float_array(x, check_finite)
    y = _as_float_array(y, check_finite)

    y = np.moveaxis(y, axis, 0)    # now internally interp axis is zero

    # sanity check the input
    if bc_type == 'periodic' and not np.allclose(y[0], y[-1], atol=1e-15):
        raise ValueError("First and last points does not match while "
                         "periodic case expected")
    if x.size != y.shape[0]:
        raise ValueError(f'Shapes of x {x.shape} and y {y.shape} are incompatible')
    if np.any(x[1:] == x[:-1]):
        raise ValueError("Expect x to not have duplicates")
    if x.ndim != 1 or np.any(x[1:] < x[:-1]):
        raise ValueError("Expect x to be a 1D strictly increasing sequence.")

    # special-case k=0 right away
    if k == 0:
        if any(_ is not None for _ in (t, deriv_l, deriv_r)):
            raise ValueError("Too much info for k=0: t and bc_type can only "
                             "be None.")
        t = np.r_[x, x[-1]]
        c = np.asarray(y)
        c = np.ascontiguousarray(c, dtype=_get_dtype(c.dtype))
        return BSpline.construct_fast(t, c, k, axis=axis)

    # special-case k=1 (e.g., Lyche and Morken, Eq.(2.16))
    if k == 1 and t is None:
        if not (deriv_l is None and deriv_r is None):
            raise ValueError("Too much info for k=1: bc_type can only be None.")
        t = np.r_[x[0], x, x[-1]]
        c = np.asarray(y)
        c = np.ascontiguousarray(c, dtype=_get_dtype(c.dtype))
        return BSpline.construct_fast(t, c, k, axis=axis)

    k = operator.index(k)

    if bc_type == 'periodic' and t is not None:
        raise NotImplementedError("For periodic case t is constructed "
                                  "automatically and can not be passed "
                                  "manually")

    # come up with a sensible knot vector, if needed
    if t is None:
        if deriv_l is None and deriv_r is None:
            if bc_type == 'periodic':
                t = _periodic_knots(x, k)
            elif k == 2:
                # OK, it's a bit ad hoc: Greville sites + omit
                # 2nd and 2nd-to-last points, a la not-a-knot
                t = (x[1:] + x[:-1]) / 2.
                t = np.r_[(x[0],)*(k+1),
                          t[1:-1],
                          (x[-1],)*(k+1)]
            else:
                t = _not_a_knot(x, k)
        else:
            t = _augknt(x, k)

    t = _as_float_array(t, check_finite)

    if k < 0:
        raise ValueError("Expect non-negative k.")
    if t.ndim != 1 or np.any(t[1:] < t[:-1]):
        raise ValueError("Expect t to be a 1-D sorted array_like.")
    if t.size < x.size + k + 1:
        raise ValueError('Got %d knots, need at least %d.' %
                         (t.size, x.size + k + 1))
    if (x[0] < t[k]) or (x[-1] > t[-k]):
        raise ValueError('Out of bounds w/ x = %s.' % x)

    if bc_type == 'periodic':
        return _make_periodic_spline(x, y, t, k, axis)

    # Here : deriv_l, r = [(nu, value), ...]
    deriv_l = _convert_string_aliases(deriv_l, y.shape[1:])
    deriv_l_ords, deriv_l_vals = _process_deriv_spec(deriv_l)
    nleft = deriv_l_ords.shape[0]

    deriv_r = _convert_string_aliases(deriv_r, y.shape[1:])
    deriv_r_ords, deriv_r_vals = _process_deriv_spec(deriv_r)
    nright = deriv_r_ords.shape[0]

    # have `n` conditions for `nt` coefficients; need nt-n derivatives
    n = x.size
    nt = t.size - k - 1

    if nt - n != nleft + nright:
        raise ValueError("The number of derivatives at boundaries does not "
                         f"match: expected {nt-n}, got {nleft}+{nright}")

    # bail out if the `y` array is zero-sized
    if y.size == 0:
        c = np.zeros((nt,) + y.shape[1:], dtype=float)
        return BSpline.construct_fast(t, c, k, axis=axis)

    # set up the LHS: the collocation matrix + derivatives at boundaries
    kl = ku = k
    ab = np.zeros((2*kl + ku + 1, nt), dtype=np.float64, order='F')
    _bspl._colloc(x, t, k, ab, offset=nleft)
    if nleft > 0:
        _bspl._handle_lhs_derivatives(t, k, x[0], ab, kl, ku,
                                      deriv_l_ords.astype(np.dtype("long")))
    if nright > 0:
        _bspl._handle_lhs_derivatives(t, k, x[-1], ab, kl, ku,
                                      deriv_r_ords.astype(np.dtype("long")),
                                      offset=nt-nright)

    # set up the RHS: values to interpolate (+ derivative values, if any)
    extradim = prod(y.shape[1:])
    rhs = np.empty((nt, extradim), dtype=y.dtype)
    if nleft > 0:
        rhs[:nleft] = deriv_l_vals.reshape(-1, extradim)
    rhs[nleft:nt - nright] = y.reshape(-1, extradim)
    if nright > 0:
        rhs[nt - nright:] = deriv_r_vals.reshape(-1, extradim)

    # solve Ab @ x = rhs; this is the relevant part of linalg.solve_banded
    if check_finite:
        ab, rhs = map(np.asarray_chkfinite, (ab, rhs))
    gbsv, = get_lapack_funcs(('gbsv',), (ab, rhs))
    lu, piv, c, info = gbsv(kl, ku, ab, rhs,
                            overwrite_ab=True, overwrite_b=True)

    if info > 0:
        raise LinAlgError("Collocation matrix is singular.")
    elif info < 0:
        raise ValueError('illegal value in %d-th argument of internal gbsv' % -info)

    c = np.ascontiguousarray(c.reshape((nt,) + y.shape[1:]))
    return BSpline.construct_fast(t, c, k, axis=axis)


def make_lsq_spline(x, y, t, k=3, w=None, axis=0, check_finite=True):
    r"""Compute the (coefficients of) an LSQ (Least SQuared) based
    fitting B-spline.

    The result is a linear combination

    .. math::

            S(x) = \sum_j c_j B_j(x; t)

    of the B-spline basis elements, :math:`B_j(x; t)`, which minimizes

    .. math::

        \sum_{j} \left( w_j \times (S(x_j) - y_j) \right)^2

    Parameters
    ----------
    x : array_like, shape (m,)
        Abscissas.
    y : array_like, shape (m, ...)
        Ordinates.
    t : array_like, shape (n + k + 1,).
        Knots.
        Knots and data points must satisfy Schoenberg-Whitney conditions.
    k : int, optional
        B-spline degree. Default is cubic, ``k = 3``.
    w : array_like, shape (m,), optional
        Weights for spline fitting. Must be positive. If ``None``,
        then weights are all equal.
        Default is ``None``.
    axis : int, optional
        Interpolation axis. Default is zero.
    check_finite : bool, optional
        Whether to check that the input arrays contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
        Default is True.

    Returns
    -------
    b : a BSpline object of the degree ``k`` with knots ``t``.

    See Also
    --------
    BSpline : base class representing the B-spline objects
    make_interp_spline : a similar factory function for interpolating splines
    LSQUnivariateSpline : a FITPACK-based spline fitting routine
    splrep : a FITPACK-based fitting routine

    Notes
    -----
    The number of data points must be larger than the spline degree ``k``.

    Knots ``t`` must satisfy the Schoenberg-Whitney conditions,
    i.e., there must be a subset of data points ``x[j]`` such that
    ``t[j] < x[j] < t[j+k+1]``, for ``j=0, 1,...,n-k-2``.

    Examples
    --------
    Generate some noisy data:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()
    >>> x = np.linspace(-3, 3, 50)
    >>> y = np.exp(-x**2) + 0.1 * rng.standard_normal(50)

    Now fit a smoothing cubic spline with a pre-defined internal knots.
    Here we make the knot vector (k+1)-regular by adding boundary knots:

    >>> from scipy.interpolate import make_lsq_spline, BSpline
    >>> t = [-1, 0, 1]
    >>> k = 3
    >>> t = np.r_[(x[0],)*(k+1),
    ...           t,
    ...           (x[-1],)*(k+1)]
    >>> spl = make_lsq_spline(x, y, t, k)

    For comparison, we also construct an interpolating spline for the same
    set of data:

    >>> from scipy.interpolate import make_interp_spline
    >>> spl_i = make_interp_spline(x, y)

    Plot both:

    >>> xs = np.linspace(-3, 3, 100)
    >>> plt.plot(x, y, 'ro', ms=5)
    >>> plt.plot(xs, spl(xs), 'g-', lw=3, label='LSQ spline')
    >>> plt.plot(xs, spl_i(xs), 'b-', lw=3, alpha=0.7, label='interp spline')
    >>> plt.legend(loc='best')
    >>> plt.show()

    **NaN handling**: If the input arrays contain ``nan`` values, the result is
    not useful since the underlying spline fitting routines cannot deal with
    ``nan``. A workaround is to use zero weights for not-a-number data points:

    >>> y[8] = np.nan
    >>> w = np.isnan(y)
    >>> y[w] = 0.
    >>> tck = make_lsq_spline(x, y, t, w=~w)

    Notice the need to replace a ``nan`` by a numerical value (precise value
    does not matter as long as the corresponding weight is zero.)

    """
    x = _as_float_array(x, check_finite)
    y = _as_float_array(y, check_finite)
    t = _as_float_array(t, check_finite)
    if w is not None:
        w = _as_float_array(w, check_finite)
    else:
        w = np.ones_like(x)
    k = operator.index(k)

    axis = normalize_axis_index(axis, y.ndim)

    y = np.moveaxis(y, axis, 0)    # now internally interp axis is zero

    if x.ndim != 1 or np.any(x[1:] - x[:-1] <= 0):
        raise ValueError("Expect x to be a 1-D sorted array_like.")
    if x.shape[0] < k+1:
        raise ValueError("Need more x points.")
    if k < 0:
        raise ValueError("Expect non-negative k.")
    if t.ndim != 1 or np.any(t[1:] - t[:-1] < 0):
        raise ValueError("Expect t to be a 1-D sorted array_like.")
    if x.size != y.shape[0]:
        raise ValueError(f'Shapes of x {x.shape} and y {y.shape} are incompatible')
    if k > 0 and np.any((x < t[k]) | (x > t[-k])):
        raise ValueError('Out of bounds w/ x = %s.' % x)
    if x.size != w.size:
        raise ValueError(f'Shapes of x {x.shape} and w {w.shape} are incompatible')

    # number of coefficients
    n = t.size - k - 1

    # construct A.T @ A and rhs with A the collocation matrix, and
    # rhs = A.T @ y for solving the LSQ problem  ``A.T @ A @ c = A.T @ y``
    lower = True
    extradim = prod(y.shape[1:])
    ab = np.zeros((k+1, n), dtype=np.float64, order='F')
    rhs = np.zeros((n, extradim), dtype=y.dtype, order='F')
    _bspl._norm_eq_lsq(x, t, k,
                       y.reshape(-1, extradim),
                       w,
                       ab, rhs)
    rhs = rhs.reshape((n,) + y.shape[1:])

    # have observation matrix & rhs, can solve the LSQ problem
    cho_decomp = cholesky_banded(ab, overwrite_ab=True, lower=lower,
                                 check_finite=check_finite)
    c = cho_solve_banded((cho_decomp, lower), rhs, overwrite_b=True,
                         check_finite=check_finite)

    c = np.ascontiguousarray(c)
    return BSpline.construct_fast(t, c, k, axis=axis)


#############################
#  Smoothing spline helpers #
#############################

def _compute_optimal_gcv_parameter(X, wE, y, w):
    """
    Returns an optimal regularization parameter from the GCV criteria [1].

    Parameters
    ----------
    X : array, shape (5, n)
        5 bands of the design matrix ``X`` stored in LAPACK banded storage.
    wE : array, shape (5, n)
        5 bands of the penalty matrix :math:`W^{-1} E` stored in LAPACK banded
        storage.
    y : array, shape (n,)
        Ordinates.
    w : array, shape (n,)
        Vector of weights.

    Returns
    -------
    lam : float
        An optimal from the GCV criteria point of view regularization
        parameter.

    Notes
    -----
    No checks are performed.

    References
    ----------
    .. [1] G. Wahba, "Estimating the smoothing parameter" in Spline models
        for observational data, Philadelphia, Pennsylvania: Society for
        Industrial and Applied Mathematics, 1990, pp. 45-65.
        :doi:`10.1137/1.9781611970128`

    """

    def compute_banded_symmetric_XT_W_Y(X, w, Y):
        """
        Assuming that the product :math:`X^T W Y` is symmetric and both ``X``
        and ``Y`` are 5-banded, compute the unique bands of the product.

        Parameters
        ----------
        X : array, shape (5, n)
            5 bands of the matrix ``X`` stored in LAPACK banded storage.
        w : array, shape (n,)
            Array of weights
        Y : array, shape (5, n)
            5 bands of the matrix ``Y`` stored in LAPACK banded storage.

        Returns
        -------
        res : array, shape (4, n)
            The result of the product :math:`X^T Y` stored in the banded way.

        Notes
        -----
        As far as the matrices ``X`` and ``Y`` are 5-banded, their product
        :math:`X^T W Y` is 7-banded. It is also symmetric, so we can store only
        unique diagonals.

        """
        # compute W Y
        W_Y = np.copy(Y)

        W_Y[2] *= w
        for i in range(2):
            W_Y[i, 2 - i:] *= w[:-2 + i]
            W_Y[3 + i, :-1 - i] *= w[1 + i:]

        n = X.shape[1]
        res = np.zeros((4, n))
        for i in range(n):
            for j in range(min(n-i, 4)):
                res[-j-1, i + j] = sum(X[j:, i] * W_Y[:5-j, i + j])
        return res

    def compute_b_inv(A):
        """
        Inverse 3 central bands of matrix :math:`A=U^T D^{-1} U` assuming that
        ``U`` is a unit upper triangular banded matrix using an algorithm
        proposed in [1].

        Parameters
        ----------
        A : array, shape (4, n)
            Matrix to inverse, stored in LAPACK banded storage.

        Returns
        -------
        B : array, shape (4, n)
            3 unique bands of the symmetric matrix that is an inverse to ``A``.
            The first row is filled with zeros.

        Notes
        -----
        The algorithm is based on the cholesky decomposition and, therefore,
        in case matrix ``A`` is close to not positive defined, the function
        raises LinalgError.

        Both matrices ``A`` and ``B`` are stored in LAPACK banded storage.

        References
        ----------
        .. [1] M. F. Hutchinson and F. R. de Hoog, "Smoothing noisy data with
            spline functions," Numerische Mathematik, vol. 47, no. 1,
            pp. 99-106, 1985.
            :doi:`10.1007/BF01389878`

        """

        def find_b_inv_elem(i, j, U, D, B):
            rng = min(3, n - i - 1)
            rng_sum = 0.
            if j == 0:
                # use 2-nd formula from [1]
                for k in range(1, rng + 1):
                    rng_sum -= U[-k - 1, i + k] * B[-k - 1, i + k]
                rng_sum += D[i]
                B[-1, i] = rng_sum
            else:
                # use 1-st formula from [1]
                for k in range(1, rng + 1):
                    diag = abs(k - j)
                    ind = i + min(k, j)
                    rng_sum -= U[-k - 1, i + k] * B[-diag - 1, ind + diag]
                B[-j - 1, i + j] = rng_sum

        U = cholesky_banded(A)
        for i in range(2, 5):
            U[-i, i-1:] /= U[-1, :-i+1]
        D = 1. / (U[-1])**2
        U[-1] /= U[-1]

        n = U.shape[1]

        B = np.zeros(shape=(4, n))
        for i in range(n - 1, -1, -1):
            for j in range(min(3, n - i - 1), -1, -1):
                find_b_inv_elem(i, j, U, D, B)
        # the first row contains garbage and should be removed
        B[0] = [0.] * n
        return B

    def _gcv(lam, X, XtWX, wE, XtE):
        r"""
        Computes the generalized cross-validation criteria [1].

        Parameters
        ----------
        lam : float, (:math:`\lambda \geq 0`)
            Regularization parameter.
        X : array, shape (5, n)
            Matrix is stored in LAPACK banded storage.
        XtWX : array, shape (4, n)
            Product :math:`X^T W X` stored in LAPACK banded storage.
        wE : array, shape (5, n)
            Matrix :math:`W^{-1} E` stored in LAPACK banded storage.
        XtE : array, shape (4, n)
            Product :math:`X^T E` stored in LAPACK banded storage.

        Returns
        -------
        res : float
            Value of the GCV criteria with the regularization parameter
            :math:`\lambda`.

        Notes
        -----
        Criteria is computed from the formula (1.3.2) [3]:

        .. math:

        GCV(\lambda) = \dfrac{1}{n} \sum\limits_{k = 1}^{n} \dfrac{ \left(
        y_k - f_{\lambda}(x_k) \right)^2}{\left( 1 - \Tr{A}/n\right)^2}$.
        The criteria is discussed in section 1.3 [3].

        The numerator is computed using (2.2.4) [3] and the denominator is
        computed using an algorithm from [2] (see in the ``compute_b_inv``
        function).

        References
        ----------
        .. [1] G. Wahba, "Estimating the smoothing parameter" in Spline models
            for observational data, Philadelphia, Pennsylvania: Society for
            Industrial and Applied Mathematics, 1990, pp. 45-65.
            :doi:`10.1137/1.9781611970128`
        .. [2] M. F. Hutchinson and F. R. de Hoog, "Smoothing noisy data with
            spline functions," Numerische Mathematik, vol. 47, no. 1,
            pp. 99-106, 1985.
            :doi:`10.1007/BF01389878`
        .. [3] E. Zemlyanoy, "Generalized cross-validation smoothing splines",
            BSc thesis, 2022. Might be available (in Russian)
            `here <https://www.hse.ru/ba/am/students/diplomas/620910604>`_

        """
        # Compute the numerator from (2.2.4) [3]
        n = X.shape[1]
        c = solve_banded((2, 2), X + lam * wE, y)
        res = np.zeros(n)
        # compute ``W^{-1} E c`` with respect to banded-storage of ``E``
        tmp = wE * c
        for i in range(n):
            for j in range(max(0, i - n + 3), min(5, i + 3)):
                res[i] += tmp[j, i + 2 - j]
        numer = np.linalg.norm(lam * res)**2 / n

        # compute the denominator
        lhs = XtWX + lam * XtE
        try:
            b_banded = compute_b_inv(lhs)
            # compute the trace of the product b_banded @ XtX
            tr = b_banded * XtWX
            tr[:-1] *= 2
            # find the denominator
            denom = (1 - sum(sum(tr)) / n)**2
        except LinAlgError:
            # cholesky decomposition cannot be performed
            raise ValueError('Seems like the problem is ill-posed')

        res = numer / denom

        return res

    n = X.shape[1]

    XtWX = compute_banded_symmetric_XT_W_Y(X, w, X)
    XtE = compute_banded_symmetric_XT_W_Y(X, w, wE)

    def fun(lam):
        return _gcv(lam, X, XtWX, wE, XtE)

    gcv_est = minimize_scalar(fun, bounds=(0, n), method='Bounded')
    if gcv_est.success:
        return gcv_est.x
    raise ValueError(f"Unable to find minimum of the GCV "
                     f"function: {gcv_est.message}")


def _coeff_of_divided_diff(x):
    """
    Returns the coefficients of the divided difference.

    Parameters
    ----------
    x : array, shape (n,)
        Array which is used for the computation of divided difference.

    Returns
    -------
    res : array_like, shape (n,)
        Coefficients of the divided difference.

    Notes
    -----
    Vector ``x`` should have unique elements, otherwise an error division by
    zero might be raised.

    No checks are performed.

    """
    n = x.shape[0]
    res = np.zeros(n)
    for i in range(n):
        pp = 1.
        for k in range(n):
            if k != i:
                pp *= (x[i] - x[k])
        res[i] = 1. / pp
    return res


def make_smoothing_spline(x, y, w=None, lam=None):
    r"""
    Compute the (coefficients of) smoothing cubic spline function using
    ``lam`` to control the tradeoff between the amount of smoothness of the
    curve and its proximity to the data. In case ``lam`` is None, using the
    GCV criteria [1] to find it.

    A smoothing spline is found as a solution to the regularized weighted
    linear regression problem:

    .. math::

        \sum\limits_{i=1}^n w_i\lvert y_i - f(x_i) \rvert^2 +
        \lambda\int\limits_{x_1}^{x_n} (f^{(2)}(u))^2 d u

    where :math:`f` is a spline function, :math:`w` is a vector of weights and
    :math:`\lambda` is a regularization parameter.

    If ``lam`` is None, we use the GCV criteria to find an optimal
    regularization parameter, otherwise we solve the regularized weighted
    linear regression problem with given parameter. The parameter controls
    the tradeoff in the following way: the larger the parameter becomes, the
    smoother the function gets.

    Parameters
    ----------
    x : array_like, shape (n,)
        Abscissas. `n` must be at least 5.
    y : array_like, shape (n,)
        Ordinates. `n` must be at least 5.
    w : array_like, shape (n,), optional
        Vector of weights. Default is ``np.ones_like(x)``.
    lam : float, (:math:`\lambda \geq 0`), optional
        Regularization parameter. If ``lam`` is None, then it is found from
        the GCV criteria. Default is None.

    Returns
    -------
    func : a BSpline object.
        A callable representing a spline in the B-spline basis
        as a solution of the problem of smoothing splines using
        the GCV criteria [1] in case ``lam`` is None, otherwise using the
        given parameter ``lam``.

    Notes
    -----
    This algorithm is a clean room reimplementation of the algorithm
    introduced by Woltring in FORTRAN [2]. The original version cannot be used
    in SciPy source code because of the license issues. The details of the
    reimplementation are discussed here (available only in Russian) [4].

    If the vector of weights ``w`` is None, we assume that all the points are
    equal in terms of weights, and vector of weights is vector of ones.

    Note that in weighted residual sum of squares, weights are not squared:
    :math:`\sum\limits_{i=1}^n w_i\lvert y_i - f(x_i) \rvert^2` while in
    ``splrep`` the sum is built from the squared weights.

    In cases when the initial problem is ill-posed (for example, the product
    :math:`X^T W X` where :math:`X` is a design matrix is not a positive
    defined matrix) a ValueError is raised.

    References
    ----------
    .. [1] G. Wahba, "Estimating the smoothing parameter" in Spline models for
        observational data, Philadelphia, Pennsylvania: Society for Industrial
        and Applied Mathematics, 1990, pp. 45-65.
        :doi:`10.1137/1.9781611970128`
    .. [2] H. J. Woltring, A Fortran package for generalized, cross-validatory
        spline smoothing and differentiation, Advances in Engineering
        Software, vol. 8, no. 2, pp. 104-113, 1986.
        :doi:`10.1016/0141-1195(86)90098-7`
    .. [3] T. Hastie, J. Friedman, and R. Tisbshirani, "Smoothing Splines" in
        The elements of Statistical Learning: Data Mining, Inference, and
        prediction, New York: Springer, 2017, pp. 241-249.
        :doi:`10.1007/978-0-387-84858-7`
    .. [4] E. Zemlyanoy, "Generalized cross-validation smoothing splines",
        BSc thesis, 2022.
        `<https://www.hse.ru/ba/am/students/diplomas/620910604>`_ (in
        Russian)

    Examples
    --------
    Generate some noisy data

    >>> import numpy as np
    >>> np.random.seed(1234)
    >>> n = 200
    >>> def func(x):
    ...    return x**3 + x**2 * np.sin(4 * x)
    >>> x = np.sort(np.random.random_sample(n) * 4 - 2)
    >>> y = func(x) + np.random.normal(scale=1.5, size=n)

    Make a smoothing spline function

    >>> from scipy.interpolate import make_smoothing_spline
    >>> spl = make_smoothing_spline(x, y)

    Plot both

    >>> import matplotlib.pyplot as plt
    >>> grid = np.linspace(x[0], x[-1], 400)
    >>> plt.plot(grid, spl(grid), label='Spline')
    >>> plt.plot(grid, func(grid), label='Original function')
    >>> plt.scatter(x, y, marker='.')
    >>> plt.legend(loc='best')
    >>> plt.show()

    """

    x = np.ascontiguousarray(x, dtype=float)
    y = np.ascontiguousarray(y, dtype=float)

    if any(x[1:] - x[:-1] <= 0):
        raise ValueError('``x`` should be an ascending array')

    if x.ndim != 1 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError('``x`` and ``y`` should be one dimensional and the'
                         ' same size')

    if w is None:
        w = np.ones(len(x))
    else:
        w = np.ascontiguousarray(w)
        if any(w <= 0):
            raise ValueError('Invalid vector of weights')

    t = np.r_[[x[0]] * 3, x, [x[-1]] * 3]
    n = x.shape[0]

    if n <= 4:
        raise ValueError('``x`` and ``y`` length must be at least 5')

    # It is known that the solution to the stated minimization problem exists
    # and is a natural cubic spline with vector of knots equal to the unique
    # elements of ``x`` [3], so we will solve the problem in the basis of
    # natural splines.

    # create design matrix in the B-spline basis
    X_bspl = BSpline.design_matrix(x, t, 3)
    # move from B-spline basis to the basis of natural splines using equations
    # (2.1.7) [4]
    # central elements
    X = np.zeros((5, n))
    for i in range(1, 4):
        X[i, 2: -2] = X_bspl[i: i - 4, 3: -3][np.diag_indices(n - 4)]

    # first elements
    X[1, 1] = X_bspl[0, 0]
    X[2, :2] = ((x[2] + x[1] - 2 * x[0]) * X_bspl[0, 0],
                X_bspl[1, 1] + X_bspl[1, 2])
    X[3, :2] = ((x[2] - x[0]) * X_bspl[1, 1], X_bspl[2, 2])

    # last elements
    X[1, -2:] = (X_bspl[-3, -3], (x[-1] - x[-3]) * X_bspl[-2, -2])
    X[2, -2:] = (X_bspl[-2, -3] + X_bspl[-2, -2],
                 (2 * x[-1] - x[-2] - x[-3]) * X_bspl[-1, -1])
    X[3, -2] = X_bspl[-1, -1]

    # create penalty matrix and divide it by vector of weights: W^{-1} E
    wE = np.zeros((5, n))
    wE[2:, 0] = _coeff_of_divided_diff(x[:3]) / w[:3]
    wE[1:, 1] = _coeff_of_divided_diff(x[:4]) / w[:4]
    for j in range(2, n - 2):
        wE[:, j] = (x[j+2] - x[j-2]) * _coeff_of_divided_diff(x[j-2:j+3])\
                   / w[j-2: j+3]

    wE[:-1, -2] = -_coeff_of_divided_diff(x[-4:]) / w[-4:]
    wE[:-2, -1] = _coeff_of_divided_diff(x[-3:]) / w[-3:]
    wE *= 6

    if lam is None:
        lam = _compute_optimal_gcv_parameter(X, wE, y, w)
    elif lam < 0.:
        raise ValueError('Regularization parameter should be non-negative')

    # solve the initial problem in the basis of natural splines
    c = solve_banded((2, 2), X + lam * wE, y)
    # move back to B-spline basis using equations (2.2.10) [4]
    c_ = np.r_[c[0] * (t[5] + t[4] - 2 * t[3]) + c[1],
               c[0] * (t[5] - t[3]) + c[1],
               c[1: -1],
               c[-1] * (t[-4] - t[-6]) + c[-2],
               c[-1] * (2 * t[-4] - t[-5] - t[-6]) + c[-2]]

    return BSpline.construct_fast(t, c_, 3)
