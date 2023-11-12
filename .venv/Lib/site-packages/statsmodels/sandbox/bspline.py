'''
Bspines and smoothing splines.

General references:

    Craven, P. and Wahba, G. (1978) "Smoothing noisy data with spline functions.
    Estimating the correct degree of smoothing by
    the method of generalized cross-validation."
    Numerische Mathematik, 31(4), 377-403.

    Hastie, Tibshirani and Friedman (2001). "The Elements of Statistical
    Learning." Springer-Verlag. 536 pages.

    Hutchison, M. and Hoog, F. "Smoothing noisy data with spline functions."
    Numerische Mathematik, 47(1), 99-106.
'''

import numpy as np
import numpy.linalg as L

from scipy.linalg import solveh_banded
from scipy.optimize import golden
from models import _hbspline     #removed because this was segfaulting

# Issue warning regarding heavy development status of this module
import warnings
_msg = """
The bspline code is technology preview and requires significant work
on the public API and documentation. The API will likely change in the future
"""
warnings.warn(_msg, FutureWarning)


def _band2array(a, lower=0, symmetric=False, hermitian=False):
    """
    Take an upper or lower triangular banded matrix and return a
    numpy array.

    INPUTS:
       a         -- a matrix in upper or lower triangular banded matrix
       lower     -- is the matrix upper or lower triangular?
       symmetric -- if True, return the original result plus its transpose
       hermitian -- if True (and symmetric False), return the original
                    result plus its conjugate transposed
    """

    n = a.shape[1]
    r = a.shape[0]
    _a = 0

    if not lower:
        for j in range(r):
            _b = np.diag(a[r-1-j],k=j)[j:(n+j),j:(n+j)]
            _a += _b
            if symmetric and j > 0:
                _a += _b.T
            elif hermitian and j > 0:
                _a += _b.conjugate().T
    else:
        for j in range(r):
            _b = np.diag(a[j],k=j)[0:n,0:n]
            _a += _b
            if symmetric and j > 0:
                _a += _b.T
            elif hermitian and j > 0:
                _a += _b.conjugate().T
        _a = _a.T

    return _a


def _upper2lower(ub):
    """
    Convert upper triangular banded matrix to lower banded form.

    INPUTS:
       ub  -- an upper triangular banded matrix

    OUTPUTS: lb
       lb  -- a lower triangular banded matrix with same entries
              as ub
    """

    lb = np.zeros(ub.shape, ub.dtype)
    nrow, ncol = ub.shape
    for i in range(ub.shape[0]):
        lb[i,0:(ncol-i)] = ub[nrow-1-i,i:ncol]
        lb[i,(ncol-i):] = ub[nrow-1-i,0:i]
    return lb

def _lower2upper(lb):
    """
    Convert lower triangular banded matrix to upper banded form.

    INPUTS:
       lb  -- a lower triangular banded matrix

    OUTPUTS: ub
       ub  -- an upper triangular banded matrix with same entries
              as lb
    """

    ub = np.zeros(lb.shape, lb.dtype)
    nrow, ncol = lb.shape
    for i in range(lb.shape[0]):
        ub[nrow-1-i,i:ncol] = lb[i,0:(ncol-i)]
        ub[nrow-1-i,0:i] = lb[i,(ncol-i):]
    return ub

def _triangle2unit(tb, lower=0):
    """
    Take a banded triangular matrix and return its diagonal and the
    unit matrix: the banded triangular matrix with 1's on the diagonal,
    i.e. each row is divided by the corresponding entry on the diagonal.

    INPUTS:
       tb    -- a lower triangular banded matrix
       lower -- if True, then tb is assumed to be lower triangular banded,
                in which case return value is also lower triangular banded.

    OUTPUTS: d, b
       d     -- diagonal entries of tb
       b     -- unit matrix: if lower is False, b is upper triangular
                banded and its rows of have been divided by d,
                else lower is True, b is lower triangular banded
                and its columns have been divieed by d.
    """

    if lower:
        d = tb[0].copy()
    else:
        d = tb[-1].copy()

    if lower:
        return d, (tb / d)
    else:
        lnum = _upper2lower(tb)
        return d, _lower2upper(lnum / d)


def _trace_symbanded(a, b, lower=0):
    """
    Compute the trace(ab) for two upper or banded real symmetric matrices
    stored either in either upper or lower form.

    INPUTS:
       a, b    -- two banded real symmetric matrices (either lower or upper)
       lower   -- if True, a and b are assumed to be the lower half


    OUTPUTS: trace
       trace   -- trace(ab)
    """

    if lower:
        t = _zero_triband(a * b, lower=1)
        return t[0].sum() + 2 * t[1:].sum()
    else:
        t = _zero_triband(a * b, lower=0)
        return t[-1].sum() + 2 * t[:-1].sum()


def _zero_triband(a, lower=0):
    """
    Explicitly zero out unused elements of a real symmetric banded matrix.

    INPUTS:
       a   -- a real symmetric banded matrix (either upper or lower hald)
       lower   -- if True, a is assumed to be the lower half
    """

    nrow, ncol = a.shape
    if lower:
        for i in range(nrow):
            a[i, (ncol-i):] = 0.
    else:
        for i in range(nrow):
            a[i, 0:i] = 0.
    return a


class BSpline:

    '''

    Bsplines of a given order and specified knots.

    Implementation is based on description in Chapter 5 of

    Hastie, Tibshirani and Friedman (2001). "The Elements of Statistical
    Learning." Springer-Verlag. 536 pages.


    INPUTS:
       knots  -- a sorted array of knots with knots[0] the lower boundary,
                 knots[1] the upper boundary and knots[1:-1] the internal
                 knots.
       order  -- order of the Bspline, default is 4 which yields cubic
                 splines
       M      -- number of additional boundary knots, if None it defaults
                 to order
       coef   -- an optional array of real-valued coefficients for the Bspline
                 of shape (knots.shape + 2 * (M - 1) - order,).
       x      -- an optional set of x values at which to evaluate the
                 Bspline to avoid extra evaluation in the __call__ method

    '''
    # FIXME: update parameter names, replace single character names
    # FIXME: `order` should be actual spline order (implemented as order+1)
    ## FIXME: update the use of spline order in extension code (evaluate is recursively called)
    # FIXME: eliminate duplicate M and m attributes (m is order, M is related to tau size)

    def __init__(self, knots, order=4, M=None, coef=None, x=None):

        knots = np.squeeze(np.unique(np.asarray(knots)))

        if knots.ndim != 1:
            raise ValueError('expecting 1d array for knots')

        self.m = order
        if M is None:
            M = self.m
        self.M = M

        self.tau = np.hstack([[knots[0]]*(self.M-1), knots, [knots[-1]]*(self.M-1)])

        self.K = knots.shape[0] - 2
        if coef is None:
            self.coef = np.zeros((self.K + 2 * self.M - self.m), np.float64)
        else:
            self.coef = np.squeeze(coef)
            if self.coef.shape != (self.K + 2 * self.M - self.m):
                raise ValueError('coefficients of Bspline have incorrect shape')
        if x is not None:
            self.x = x

    def _setx(self, x):
        self._x = x
        self._basisx = self.basis(self._x)

    def _getx(self):
        return self._x

    x = property(_getx, _setx)

    def __call__(self, *args):
        """
        Evaluate the BSpline at a given point, yielding
        a matrix B and return

        B * self.coef


        INPUTS:
           args -- optional arguments. If None, it returns self._basisx,
                   the BSpline evaluated at the x values passed in __init__.
                   Otherwise, return the BSpline evaluated at the
                   first argument args[0].

        OUTPUTS: y
           y    -- value of Bspline at specified x values

        BUGS:
           If self has no attribute x, an exception will be raised
           because self has no attribute _basisx.
        """

        if not args:
            b = self._basisx.T
        else:
            x = args[0]
            b = np.asarray(self.basis(x)).T
        return np.squeeze(np.dot(b, self.coef))

    def basis_element(self, x, i, d=0):
        """
        Evaluate a particular basis element of the BSpline,
        or its derivative.

        INPUTS:
           x  -- x values at which to evaluate the basis element
           i  -- which element of the BSpline to return
           d  -- the order of derivative

        OUTPUTS: y
           y  -- value of d-th derivative of the i-th basis element
                 of the BSpline at specified x values
        """

        x = np.asarray(x, np.float64)
        _shape = x.shape
        if _shape == ():
            x.shape = (1,)
        x.shape = (np.product(_shape,axis=0),)
        if i < self.tau.shape[0] - 1:
            # TODO: OWNDATA flags...
            v = _hbspline.evaluate(x, self.tau, self.m, d, i, i+1)
        else:
            return np.zeros(x.shape, np.float64)

        if (i == self.tau.shape[0] - self.m):
            v = np.where(np.equal(x, self.tau[-1]), 1, v)
        v.shape = _shape
        return v

    def basis(self, x, d=0, lower=None, upper=None):
        """
        Evaluate the basis of the BSpline or its derivative.
        If lower or upper is specified, then only
        the [lower:upper] elements of the basis are returned.

        INPUTS:
           x     -- x values at which to evaluate the basis element
           i     -- which element of the BSpline to return
           d     -- the order of derivative
           lower -- optional lower limit of the set of basis
                    elements
           upper -- optional upper limit of the set of basis
                    elements

        OUTPUTS: y
           y  -- value of d-th derivative of the basis elements
                 of the BSpline at specified x values
        """
        x = np.asarray(x)
        _shape = x.shape
        if _shape == ():
            x.shape = (1,)
        x.shape = (np.product(_shape,axis=0),)

        if upper is None:
            upper = self.tau.shape[0] - self.m
        if lower is None:
            lower = 0
        upper = min(upper, self.tau.shape[0] - self.m)
        lower = max(0, lower)

        d = np.asarray(d)
        if d.shape == ():
            v = _hbspline.evaluate(x, self.tau, self.m, int(d), lower, upper)
        else:
            if d.shape[0] != 2:
                raise ValueError("if d is not an integer, expecting a jx2 \
                   array with first row indicating order \
                   of derivative, second row coefficient in front.")
            v = 0
            for i in range(d.shape[1]):
                v += d[1,i] * _hbspline.evaluate(x, self.tau, self.m, d[0,i], lower, upper)

        v.shape = (upper-lower,) + _shape
        if upper == self.tau.shape[0] - self.m:
            v[-1] = np.where(np.equal(x, self.tau[-1]), 1, v[-1])
        return v

    def gram(self, d=0):
        """
        Compute Gram inner product matrix, storing it in lower
        triangular banded form.

        The (i,j) entry is

        G_ij = integral b_i^(d) b_j^(d)

        where b_i are the basis elements of the BSpline and (d) is the
        d-th derivative.

        If d is a matrix then, it is assumed to specify a differential
        operator as follows: the first row represents the order of derivative
        with the second row the coefficient corresponding to that order.

        For instance:

        [[2, 3],
         [3, 1]]

        represents 3 * f^(2) + 1 * f^(3).

        INPUTS:
           d    -- which derivative to apply to each basis element,
                   if d is a matrix, it is assumed to specify
                   a differential operator as above

        OUTPUTS: gram
           gram -- the matrix of inner products of (derivatives)
                   of the BSpline elements
        """

        d = np.squeeze(d)
        if np.asarray(d).shape == ():
            self.g = _hbspline.gram(self.tau, self.m, int(d), int(d))
        else:
            d = np.asarray(d)
            if d.shape[0] != 2:
                raise ValueError("if d is not an integer, expecting a jx2 \
                   array with first row indicating order \
                   of derivative, second row coefficient in front.")
            if d.shape == (2,):
                d.shape = (2,1)
            self.g = 0
            for i in range(d.shape[1]):
                for j in range(d.shape[1]):
                    self.g += d[1,i]* d[1,j] * _hbspline.gram(self.tau, self.m, int(d[0,i]), int(d[0,j]))
        self.g = self.g.T
        self.d = d
        return np.nan_to_num(self.g)

class SmoothingSpline(BSpline):

    penmax = 30.
    method = "target_df"
    target_df = 5
    default_pen = 1.0e-03
    optimize = True

    '''
    A smoothing spline, which can be used to smooth scatterplots, i.e.
    a list of (x,y) tuples.

    See fit method for more information.

    '''

    def fit(self, y, x=None, weights=None, pen=0.):
        """
        Fit the smoothing spline to a set of (x,y) pairs.

        INPUTS:
           y       -- response variable
           x       -- if None, uses self.x
           weights -- optional array of weights
           pen     -- constant in front of Gram matrix

        OUTPUTS: None
           The smoothing spline is determined by self.coef,
           subsequent calls of __call__ will be the smoothing spline.

        ALGORITHM:
           Formally, this solves a minimization:

           fhat = ARGMIN_f SUM_i=1^n (y_i-f(x_i))^2 + pen * int f^(2)^2

           int is integral. pen is lambda (from Hastie)

           See Chapter 5 of

           Hastie, Tibshirani and Friedman (2001). "The Elements of Statistical
           Learning." Springer-Verlag. 536 pages.

           for more details.

        TODO:
           Should add arbitrary derivative penalty instead of just
           second derivative.
        """

        banded = True

        if x is None:
            x = self._x
            bt = self._basisx.copy()
        else:
            bt = self.basis(x)

        if pen == 0.: # cannot use cholesky for singular matrices
            banded = False

        if x.shape != y.shape:
            raise ValueError('x and y shape do not agree, by default x are \
               the Bspline\'s internal knots')

        if pen >= self.penmax:
            pen = self.penmax


        if weights is not None:
            self.weights = weights
        else:
            self.weights = 1.

        _w = np.sqrt(self.weights)
        bt *= _w

        # throw out rows with zeros (this happens at boundary points!)

        mask = np.flatnonzero(1 - np.all(np.equal(bt, 0), axis=0))

        bt = bt[:,mask]
        y = y[mask]

        self.df_total = y.shape[0]

        bty = np.squeeze(np.dot(bt, _w * y))
        self.N = y.shape[0]

        if not banded:
            self.btb = np.dot(bt, bt.T)
            _g = _band2array(self.g, lower=1, symmetric=True)
            self.coef, _, self.rank = L.lstsq(self.btb + pen*_g, bty)[0:3]
            self.rank = min(self.rank, self.btb.shape[0])
            del _g
        else:
            self.btb = np.zeros(self.g.shape, np.float64)
            nband, nbasis = self.g.shape
            for i in range(nbasis):
                for k in range(min(nband, nbasis-i)):
                    self.btb[k,i] = (bt[i] * bt[i+k]).sum()

            bty.shape = (1,bty.shape[0])
            self.pen = pen
            self.chol, self.coef = solveh_banded(self.btb +
                                                 pen*self.g,
                                                 bty, lower=1)

        self.coef = np.squeeze(self.coef)
        self.resid = y * self.weights - np.dot(self.coef, bt)
        self.pen = pen

        del bty
        del mask
        del bt

    def smooth(self, y, x=None, weights=None):

        if self.method == "target_df":
            if hasattr(self, 'pen'):
                self.fit(y, x=x, weights=weights, pen=self.pen)
            else:
                self.fit_target_df(y, x=x, weights=weights, df=self.target_df)
        elif self.method == "optimize_gcv":
            self.fit_optimize_gcv(y, x=x, weights=weights)


    def gcv(self):
        """
        Generalized cross-validation score of current fit.

        Craven, P. and Wahba, G.  "Smoothing noisy data with spline functions.
        Estimating the correct degree of smoothing by
        the method of generalized cross-validation."
        Numerische Mathematik, 31(4), 377-403.
        """

        norm_resid = (self.resid**2).sum()
        return norm_resid / (self.df_total - self.trace())

    def df_resid(self):
        """
        Residual degrees of freedom in the fit.

        self.N - self.trace()

        where self.N is the number of observations of last fit.
        """

        return self.N - self.trace()

    def df_fit(self):
        """
        How many degrees of freedom used in the fit?

        self.trace()
        """
        return self.trace()

    def trace(self):
        """
        Trace of the smoothing matrix S(pen)

        TODO: addin a reference to Wahba, and whoever else I used.
        """

        if self.pen > 0:
            _invband = _hbspline.invband(self.chol.copy())
            tr = _trace_symbanded(_invband, self.btb, lower=1)
            return tr
        else:
            return self.rank

    def fit_target_df(self, y, x=None, df=None, weights=None, tol=1.0e-03,
                      apen=0, bpen=1.0e-03):
        """
        Fit smoothing spline with approximately df degrees of freedom
        used in the fit, i.e. so that self.trace() is approximately df.

        Uses binary search strategy.

        In general, df must be greater than the dimension of the null space
        of the Gram inner product. For cubic smoothing splines, this means
        that df > 2.

        INPUTS:
           y       -- response variable
           x       -- if None, uses self.x
           df      -- target degrees of freedom
           weights -- optional array of weights
           tol     -- (relative) tolerance for convergence
           apen    -- lower bound of penalty for binary search
           bpen    -- upper bound of penalty for binary search

        OUTPUTS: None
           The smoothing spline is determined by self.coef,
           subsequent calls of __call__ will be the smoothing spline.
        """

        df = df or self.target_df

        olddf = y.shape[0] - self.m

        if hasattr(self, "pen"):
            self.fit(y, x=x, weights=weights, pen=self.pen)
            curdf = self.trace()
            if np.fabs(curdf - df) / df < tol:
                return
            if curdf > df:
                apen, bpen = self.pen, 2 * self.pen
            else:
                apen, bpen = 0., self.pen

        while True:

            curpen = 0.5 * (apen + bpen)
            self.fit(y, x=x, weights=weights, pen=curpen)
            curdf = self.trace()
            if curdf > df:
                apen, bpen = curpen, 2 * curpen
            else:
                apen, bpen = apen, curpen
            if apen >= self.penmax:
                raise ValueError("penalty too large, try setting penmax \
                   higher or decreasing df")
            if np.fabs(curdf - df) / df < tol:
                break

    def fit_optimize_gcv(self, y, x=None, weights=None, tol=1.0e-03,
                         brack=(-100,20)):
        """
        Fit smoothing spline trying to optimize GCV.

        Try to find a bracketing interval for scipy.optimize.golden
        based on bracket.

        It is probably best to use target_df instead, as it is
        sometimes difficult to find a bracketing interval.

        INPUTS:
           y       -- response variable
           x       -- if None, uses self.x
           df      -- target degrees of freedom
           weights -- optional array of weights
           tol     -- (relative) tolerance for convergence
           brack   -- an initial guess at the bracketing interval

        OUTPUTS: None
           The smoothing spline is determined by self.coef,
           subsequent calls of __call__ will be the smoothing spline.
        """

        def _gcv(pen, y, x):
            self.fit(y, x=x, pen=np.exp(pen))
            a = self.gcv()
            return a

        a = golden(_gcv, args=(y,x), brack=brack, tol=tol)
