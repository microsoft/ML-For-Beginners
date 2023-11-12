# -*- coding: utf-8 -*-
"""
Spline and other smoother classes for Generalized Additive Models

Author: Luca Puggini
Author: Josef Perktold

Created on Fri Jun  5 16:32:00 2015
"""

# import useful only for development
from abc import ABCMeta, abstractmethod
from statsmodels.compat.python import with_metaclass

import numpy as np
import pandas as pd
from patsy import dmatrix
from patsy.mgcv_cubic_splines import _get_all_sorted_knots

from statsmodels.tools.linalg import transf_constraints


# Obtain b splines from patsy

def _equally_spaced_knots(x, df):
    n_knots = df - 2
    x_min = x.min()
    x_max = x.max()
    knots = np.linspace(x_min, x_max, n_knots)
    return knots


def _R_compat_quantile(x, probs):
    # return np.percentile(x, 100 * np.asarray(probs))
    probs = np.asarray(probs)
    quantiles = np.asarray([np.percentile(x, 100 * prob)
                            for prob in probs.ravel(order="C")])
    return quantiles.reshape(probs.shape, order="C")


# FIXME: is this copy/pasted?  If so, why do we need it?  If not, get
#  rid of the try/except for scipy import
# from patsy splines.py
def _eval_bspline_basis(x, knots, degree, deriv='all', include_intercept=True):
    try:
        from scipy.interpolate import splev
    except ImportError:
        raise ImportError("spline functionality requires scipy")
    # 'knots' are assumed to be already pre-processed. E.g. usually you
    # want to include duplicate copies of boundary knots; you should do
    # that *before* calling this constructor.
    knots = np.atleast_1d(np.asarray(knots, dtype=float))
    assert knots.ndim == 1
    knots.sort()
    degree = int(degree)
    x = np.atleast_1d(x)
    if x.ndim == 2 and x.shape[1] == 1:
        x = x[:, 0]
    assert x.ndim == 1
    # XX FIXME: when points fall outside of the boundaries, splev and R seem
    # to handle them differently. I do not know why yet. So until we understand
    # this and decide what to do with it, I'm going to play it safe and
    # disallow such points.
    if np.min(x) < np.min(knots) or np.max(x) > np.max(knots):
        raise NotImplementedError("some data points fall outside the "
                                  "outermost knots, and I'm not sure how "
                                  "to handle them. (Patches accepted!)")
    # Thanks to Charles Harris for explaining splev. It's not well
    # documented, but basically it computes an arbitrary b-spline basis
    # given knots and degree on some specificed points (or derivatives
    # thereof, but we do not use that functionality), and then returns some
    # linear combination of these basis functions. To get out the basis
    # functions themselves, we use linear combinations like [1, 0, 0], [0,
    # 1, 0], [0, 0, 1].
    # NB: This probably makes it rather inefficient (though I have not checked
    # to be sure -- maybe the fortran code actually skips computing the basis
    # function for coefficients that are zero).
    # Note: the order of a spline is the same as its degree + 1.
    # Note: there are (len(knots) - order) basis functions.

    k_const = 1 - int(include_intercept)
    n_bases = len(knots) - (degree + 1) - k_const
    if deriv in ['all', 0]:
        basis = np.empty((x.shape[0], n_bases), dtype=float)
        ret = basis
    if deriv in ['all', 1]:
        der1_basis = np.empty((x.shape[0], n_bases), dtype=float)
        ret = der1_basis
    if deriv in ['all', 2]:
        der2_basis = np.empty((x.shape[0], n_bases), dtype=float)
        ret = der2_basis

    for i in range(n_bases):
        coefs = np.zeros((n_bases + k_const,))
        # we are skipping the first column of the basis to drop constant
        coefs[i + k_const] = 1
        ii = i
        if deriv in ['all', 0]:
            basis[:, ii] = splev(x, (knots, coefs, degree))
        if deriv in ['all', 1]:
            der1_basis[:, ii] = splev(x, (knots, coefs, degree), der=1)
        if deriv in ['all', 2]:
            der2_basis[:, ii] = splev(x, (knots, coefs, degree), der=2)

    if deriv == 'all':
        return basis, der1_basis, der2_basis
    else:
        return ret


def compute_all_knots(x, df, degree):
    order = degree + 1
    n_inner_knots = df - order
    lower_bound = np.min(x)
    upper_bound = np.max(x)
    knot_quantiles = np.linspace(0, 1, n_inner_knots + 2)[1:-1]
    inner_knots = _R_compat_quantile(x, knot_quantiles)
    all_knots = np.concatenate(([lower_bound, upper_bound] * order,
                                inner_knots))
    return all_knots, lower_bound, upper_bound, inner_knots


def make_bsplines_basis(x, df, degree):
    ''' make a spline basis for x '''

    all_knots, _, _, _ = compute_all_knots(x, df, degree)
    basis, der_basis, der2_basis = _eval_bspline_basis(x, all_knots, degree)
    return basis, der_basis, der2_basis


def get_knots_bsplines(x=None, df=None, knots=None, degree=3,
                       spacing='quantile', lower_bound=None,
                       upper_bound=None, all_knots=None):
    """knots for use in B-splines

    There are two main options for the knot placement

    - quantile spacing with multiplicity of boundary knots
    - equal spacing extended to boundary or exterior knots

    The first corresponds to splines as used by patsy. the second is the
    knot spacing for P-Splines.
    """
    # based on patsy memorize_finish
    if all_knots is not None:
        return all_knots

    x_min = x.min()
    x_max = x.max()

    if degree < 0:
        raise ValueError("degree must be greater than 0 (not %r)"
                         % (degree,))
    if int(degree) != degree:
        raise ValueError("degree must be an integer (not %r)"
                         % (degree,))

    # These are guaranteed to all be 1d vectors by the code above
    # x = np.concatenate(tmp["xs"])
    if df is None and knots is None:
        raise ValueError("must specify either df or knots")
    order = degree + 1
    if df is not None:
        n_inner_knots = df - order
        if n_inner_knots < 0:
            raise ValueError("df=%r is too small for degree=%r; must be >= %s"
                             % (df, degree,
                                # We know that n_inner_knots is negative;
                                # if df were that much larger, it would
                                # have been zero, and things would work.
                                df - n_inner_knots))
        if knots is not None:
            if len(knots) != n_inner_knots:
                raise ValueError("df=%s with degree=%r implies %s knots, "
                                 "but %s knots were provided"
                                 % (df, degree,
                                    n_inner_knots, len(knots)))
        elif spacing == 'quantile':
            # Need to compute inner knots
            knot_quantiles = np.linspace(0, 1, n_inner_knots + 2)[1:-1]
            inner_knots = _R_compat_quantile(x, knot_quantiles)
        elif spacing == 'equal':
            # Need to compute inner knots
            grid = np.linspace(0, 1, n_inner_knots + 2)[1:-1]
            inner_knots = x_min + grid * (x_max - x_min)
            diff_knots = inner_knots[1] - inner_knots[0]
        else:
            raise ValueError("incorrect option for spacing")
    if knots is not None:
        inner_knots = knots
    if lower_bound is None:
        lower_bound = np.min(x)
    if upper_bound is None:
        upper_bound = np.max(x)

    if lower_bound > upper_bound:
        raise ValueError("lower_bound > upper_bound (%r > %r)"
                         % (lower_bound, upper_bound))
    inner_knots = np.asarray(inner_knots)
    if inner_knots.ndim > 1:
        raise ValueError("knots must be 1 dimensional")
    if np.any(inner_knots < lower_bound):
        raise ValueError("some knot values (%s) fall below lower bound "
                         "(%r)"
                         % (inner_knots[inner_knots < lower_bound],
                            lower_bound))
    if np.any(inner_knots > upper_bound):
        raise ValueError("some knot values (%s) fall above upper bound "
                         "(%r)"
                         % (inner_knots[inner_knots > upper_bound],
                            upper_bound))

    if spacing == "equal":
        diffs = np.arange(1, order + 1) * diff_knots
        lower_knots = inner_knots[0] - diffs[::-1]
        upper_knots = inner_knots[-1] + diffs
        all_knots = np.concatenate((lower_knots, inner_knots, upper_knots))
    else:
        all_knots = np.concatenate(([lower_bound, upper_bound] * order,
                                    inner_knots))
    all_knots.sort()

    return all_knots


def _get_integration_points(knots, k_points=3):
    """add points to each subinterval defined by knots

    inserts k_points between each two consecutive knots
    """
    k_points = k_points + 1
    knots = np.unique(knots)
    dxi = np.arange(k_points) / k_points
    dxk = np.diff(knots)
    dx = dxk[:, None] * dxi
    x = np.concatenate(((knots[:-1, None] + dx).ravel(), [knots[-1]]))
    return x


def get_covder2(smoother, k_points=4, integration_points=None,
                skip_ctransf=False, deriv=2):
    """
    Approximate integral of cross product of second derivative of smoother

    This uses scipy.integrate simps to compute an approximation to the
    integral of the smoother derivative cross-product at knots plus k_points
    in between knots.
    """
    from scipy.integrate import simps
    knots = smoother.knots
    x = _get_integration_points(knots, k_points=3)
    if integration_points is None:
        d2 = smoother.transform(x, deriv=deriv, skip_ctransf=skip_ctransf)
    else:
        x = integration_points
    covd2 = simps(d2[:, :, None] * d2[:, None, :], x, axis=0)
    return covd2


# TODO: this function should be deleted
def make_poly_basis(x, degree, intercept=True):
    '''
    given a vector x returns poly=(1, x, x^2, ..., x^degree)
    and its first and second derivative
    '''

    if intercept:
        start = 0
    else:
        start = 1

    nobs = len(x)
    basis = np.zeros(shape=(nobs, degree + 1 - start))
    der_basis = np.zeros(shape=(nobs, degree + 1 - start))
    der2_basis = np.zeros(shape=(nobs, degree + 1 - start))

    for i in range(start, degree + 1):
        basis[:, i - start] = x ** i
        der_basis[:, i - start] = i * x ** (i - 1)
        der2_basis[:, i - start] = i * (i - 1) * x ** (i - 2)

    return basis, der_basis, der2_basis


# TODO: try to include other kinds of splines from patsy
# x = np.linspace(0, 1, 30)
# df = 10
# degree = 3
# from patsy.mgcv_cubic_splines import cc, cr, te
# all_knots, lower, upper, inner  = compute_all_knots(x, df, degree)
# result = cc(x, df=df, knots=all_knots, lower_bound=lower, upper_bound=upper,
#             constraints=None)
#
# import matplotlib.pyplot as plt
#
# result = np.array(result)
# print(result.shape)
# plt.plot(result.T)
# plt.show()

class UnivariateGamSmoother(with_metaclass(ABCMeta)):
    """Base Class for single smooth component
    """
    def __init__(self, x, constraints=None, variable_name='x'):
        self.x = x
        self.constraints = constraints
        self.variable_name = variable_name
        self.nobs, self.k_variables = len(x), 1

        base4 = self._smooth_basis_for_single_variable()
        if constraints == 'center':
            constraints = base4[0].mean(0)[None, :]

        if constraints is not None and not isinstance(constraints, str):
            ctransf = transf_constraints(constraints)
            self.ctransf = ctransf
        else:
            # subclasses might set ctransf directly
            # only used if constraints is None
            if not hasattr(self, 'ctransf'):
                self.ctransf = None

        self.basis, self.der_basis, self.der2_basis, self.cov_der2 = base4
        if self.ctransf is not None:
            ctransf = self.ctransf
            # transform attributes that are not None
            if base4[0] is not None:
                self.basis = base4[0].dot(ctransf)
            if base4[1] is not None:
                self.der_basis = base4[1].dot(ctransf)
            if base4[2] is not None:
                self.der2_basis = base4[2].dot(ctransf)
            if base4[3] is not None:
                self.cov_der2 = ctransf.T.dot(base4[3]).dot(ctransf)

        self.dim_basis = self.basis.shape[1]
        self.col_names = [self.variable_name + "_s" + str(i)
                          for i in range(self.dim_basis)]

    @abstractmethod
    def _smooth_basis_for_single_variable(self):
        return


class UnivariateGenericSmoother(UnivariateGamSmoother):
    """Generic single smooth component
    """
    def __init__(self, x, basis, der_basis, der2_basis, cov_der2,
                 variable_name='x'):
        self.basis = basis
        self.der_basis = der_basis
        self.der2_basis = der2_basis
        self.cov_der2 = cov_der2

        super(UnivariateGenericSmoother, self).__init__(
            x, variable_name=variable_name)

    def _smooth_basis_for_single_variable(self):
        return self.basis, self.der_basis, self.der2_basis, self.cov_der2


class UnivariatePolynomialSmoother(UnivariateGamSmoother):
    """polynomial single smooth component
    """
    def __init__(self, x, degree, variable_name='x'):
        self.degree = degree
        super(UnivariatePolynomialSmoother, self).__init__(
            x, variable_name=variable_name)

    def _smooth_basis_for_single_variable(self):
        # TODO: unclear description
        """
        given a vector x returns poly=(1, x, x^2, ..., x^degree)
        and its first and second derivative
        """

        basis = np.zeros(shape=(self.nobs, self.degree))
        der_basis = np.zeros(shape=(self.nobs, self.degree))
        der2_basis = np.zeros(shape=(self.nobs, self.degree))
        for i in range(self.degree):
            dg = i + 1
            basis[:, i] = self.x ** dg
            der_basis[:, i] = dg * self.x ** (dg - 1)
            if dg > 1:
                der2_basis[:, i] = dg * (dg - 1) * self.x ** (dg - 2)
            else:
                der2_basis[:, i] = 0

        cov_der2 = np.dot(der2_basis.T, der2_basis)

        return basis, der_basis, der2_basis, cov_der2


class UnivariateBSplines(UnivariateGamSmoother):
    """B-Spline single smooth component

    This creates and holds the B-Spline basis function for one
    component.

    Parameters
    ----------
    x : ndarray, 1-D
        underlying explanatory variable for smooth terms.
    df : int
        number of basis functions or degrees of freedom
    degree : int
        degree of the spline
    include_intercept : bool
        If False, then the basis functions are transformed so that they
        do not include a constant. This avoids perfect collinearity if
        a constant or several components are included in the model.
    constraints : {None, str, array}
        Constraints are used to transform the basis functions to satisfy
        those constraints.
        `constraints = 'center'` applies a linear transform to remove the
        constant and center the basis functions.
    variable_name : {None, str}
        The name for the underlying explanatory variable, x, used in for
        creating the column and parameter names for the basis functions.
    covder2_kwds : {None, dict}
        options for computing the penalty matrix from the second derivative
        of the spline.
    knot_kwds : {None, list[dict]}
        option for the knot selection.
        By default knots are selected in the same way as in patsy, however the
        number of knots is independent of keeping or removing the constant.
        Interior knot selection is based on quantiles of the data and is the
        same in patsy and mgcv. Boundary points are at the limits of the data
        range.
        The available options use with `get_knots_bsplines` are

        - knots : None or array
          interior knots
        - spacing : 'quantile' or 'equal'
        - lower_bound : None or float
          location of lower boundary knots, all boundary knots are at the same
          point
        - upper_bound : None or float
          location of upper boundary knots, all boundary knots are at the same
          point
        - all_knots : None or array
          If all knots are provided, then those will be taken as given and
          all other options will be ignored.
    """
    def __init__(self, x, df, degree=3, include_intercept=False,
                 constraints=None, variable_name='x',
                 covder2_kwds=None, **knot_kwds):
        self.degree = degree
        self.df = df
        self.include_intercept = include_intercept
        self.knots = get_knots_bsplines(x, degree=degree, df=df, **knot_kwds)
        self.covder2_kwds = (covder2_kwds if covder2_kwds is not None
                             else {})
        super(UnivariateBSplines, self).__init__(
            x, constraints=constraints, variable_name=variable_name)

    def _smooth_basis_for_single_variable(self):
        basis, der_basis, der2_basis = _eval_bspline_basis(
            self.x, self.knots, self.degree,
            include_intercept=self.include_intercept)
        # cov_der2 = np.dot(der2_basis.T, der2_basis)

        cov_der2 = get_covder2(self, skip_ctransf=True,
                               **self.covder2_kwds)

        return basis, der_basis, der2_basis, cov_der2

    def transform(self, x_new, deriv=0, skip_ctransf=False):
        """create the spline basis for new observations

        The main use of this stateful transformation is for prediction
        using the same specification of the spline basis.

        Parameters
        ----------
        x_new : ndarray
            observations of the underlying explanatory variable
        deriv : int
            which derivative of the spline basis to compute
            This is an options for internal computation.
        skip_ctransf : bool
            whether to skip the constraint transform
            This is an options for internal computation.

        Returns
        -------
        basis : ndarray
            design matrix for the spline basis for given ``x_new``
        """

        if x_new is None:
            x_new = self.x
        exog = _eval_bspline_basis(x_new, self.knots, self.degree,
                                   deriv=deriv,
                                   include_intercept=self.include_intercept)

        # ctransf does not exist yet when cov_der2 is computed
        ctransf = getattr(self, 'ctransf', None)
        if ctransf is not None and not skip_ctransf:
            exog = exog.dot(self.ctransf)
        return exog


class UnivariateCubicSplines(UnivariateGamSmoother):
    """Cubic Spline single smooth component

    Cubic splines as described in the wood's book in chapter 3
    """

    def __init__(self, x, df, constraints=None, transform='domain',
                 variable_name='x'):

        self.degree = 3
        self.df = df
        self.transform_data_method = transform

        self.x = x = self.transform_data(x, initialize=True)
        self.knots = _equally_spaced_knots(x, df)
        super(UnivariateCubicSplines, self).__init__(
            x, constraints=constraints, variable_name=variable_name)

    def transform_data(self, x, initialize=False):
        tm = self.transform_data_method
        if tm is None:
            return x

        if initialize is True:
            if tm == 'domain':
                self.domain_low = x.min(0)
                self.domain_upp = x.max(0)
            elif isinstance(tm, tuple):
                self.domain_low = tm[0]
                self.domain_upp = tm[1]
                self.transform_data_method = 'domain'
            else:
                raise ValueError("transform should be None, 'domain' "
                                 "or a tuple")
            self.domain_diff = self.domain_upp - self.domain_low

        if self.transform_data_method == 'domain':
            x = (x - self.domain_low) / self.domain_diff
            return x
        else:
            raise ValueError("incorrect transform_data_method")

    def _smooth_basis_for_single_variable(self):

        basis = self._splines_x()[:, :-1]
        # demean except for constant, does not affect derivatives
        if not self.constraints == 'none':
            self.transf_mean = basis[:, 1:].mean(0)
            basis[:, 1:] -= self.transf_mean
        else:
            self.transf_mean = np.zeros(basis.shape[1])
        s = self._splines_s()[:-1, :-1]
        if not self.constraints == 'none':
            ctransf = np.diag(1/np.max(np.abs(basis), axis=0))
        else:
            ctransf = np.eye(basis.shape[1])
        # use np.eye to avoid rescaling
        # ctransf = np.eye(basis.shape[1])

        if self.constraints == 'no-const':
            ctransf = ctransf[1:]

        self.ctransf = ctransf

        return basis, None, None, s

    def _rk(self, x, z):
        p1 = ((z - 1 / 2) ** 2 - 1 / 12) * ((x - 1 / 2) ** 2 - 1 / 12) / 4
        p2 = ((np.abs(z - x) - 1 / 2) ** 4 -
              1 / 2 * (np.abs(z - x) - 1 / 2) ** 2 +
              7 / 240) / 24.
        return p1 - p2

    def _splines_x(self, x=None):
        if x is None:
            x = self.x
        n_columns = len(self.knots) + 2
        nobs = x.shape[0]
        basis = np.ones(shape=(nobs, n_columns))
        basis[:, 1] = x
        # for loop equivalent to outer(x, xk, fun=rk)
        for i, xi in enumerate(x):
            for j, xkj in enumerate(self.knots):
                s_ij = self._rk(xi, xkj)
                basis[i, j + 2] = s_ij
        return basis

    def _splines_s(self):
        q = len(self.knots) + 2
        s = np.zeros(shape=(q, q))
        for i, x1 in enumerate(self.knots):
            for j, x2 in enumerate(self.knots):
                s[i + 2, j + 2] = self._rk(x1, x2)
        return s

    def transform(self, x_new):
        x_new = self.transform_data(x_new, initialize=False)
        exog = self._splines_x(x_new)
        exog[:, 1:] -= self.transf_mean
        if self.ctransf is not None:
            exog = exog.dot(self.ctransf)
        return exog


class UnivariateCubicCyclicSplines(UnivariateGamSmoother):
    """cyclic cubic regression spline single smooth component

    This creates and holds the Cyclic CubicSpline basis function for one
    component.

    Parameters
    ----------
    x : ndarray, 1-D
        underlying explanatory variable for smooth terms.
    df : int
        number of basis functions or degrees of freedom
    degree : int
        degree of the spline
    include_intercept : bool
        If False, then the basis functions are transformed so that they
        do not include a constant. This avoids perfect collinearity if
        a constant or several components are included in the model.
    constraints : {None, str, array}
        Constraints are used to transform the basis functions to satisfy
        those constraints.
        `constraints = 'center'` applies a linear transform to remove the
        constant and center the basis functions.
    variable_name : None or str
        The name for the underlying explanatory variable, x, used in for
        creating the column and parameter names for the basis functions.
    """
    def __init__(self, x, df, constraints=None, variable_name='x'):
        self.degree = 3
        self.df = df
        self.x = x
        self.knots = _equally_spaced_knots(x, df)
        super(UnivariateCubicCyclicSplines, self).__init__(
            x, constraints=constraints, variable_name=variable_name)

    def _smooth_basis_for_single_variable(self):
        basis = dmatrix("cc(x, df=" + str(self.df) + ") - 1", {"x": self.x})
        self.design_info = basis.design_info
        n_inner_knots = self.df - 2 + 1  # +n_constraints
        # TODO: from CubicRegressionSplines class
        all_knots = _get_all_sorted_knots(self.x, n_inner_knots=n_inner_knots,
                                          inner_knots=None,
                                          lower_bound=None, upper_bound=None)

        b, d = self._get_b_and_d(all_knots)
        s = self._get_s(b, d)

        return basis, None, None, s

    def _get_b_and_d(self, knots):
        """Returns mapping of cyclic cubic spline values to 2nd derivatives.

        .. note:: See 'Generalized Additive Models', Simon N. Wood, 2006,
           pp 146-147

        Parameters
        ----------
        knots : ndarray
            The 1-d array knots used for cubic spline parametrization,
            must be sorted in ascending order.

        Returns
        -------
        b : ndarray
            Array for mapping cyclic cubic spline values at knots to
            second derivatives.
        d : ndarray
            Array for mapping cyclic cubic spline values at knots to
            second derivatives.

        Notes
        -----
        The penalty matrix is equal to ``s = d.T.dot(b^-1).dot(d)``
        """
        h = knots[1:] - knots[:-1]
        n = knots.size - 1

        # b and d are defined such that the penalty matrix is equivalent to:
        # s = d.T.dot(b^-1).dot(d)
        # reference in particular to pag 146 of Wood's book
        b = np.zeros((n, n))  # the b matrix on page 146 of Wood's book
        d = np.zeros((n, n))  # the d matrix on page 146 of Wood's book

        b[0, 0] = (h[n - 1] + h[0]) / 3.
        b[0, n - 1] = h[n - 1] / 6.
        b[n - 1, 0] = h[n - 1] / 6.

        d[0, 0] = -1. / h[0] - 1. / h[n - 1]
        d[0, n - 1] = 1. / h[n - 1]
        d[n - 1, 0] = 1. / h[n - 1]

        for i in range(1, n):
            b[i, i] = (h[i - 1] + h[i]) / 3.
            b[i, i - 1] = h[i - 1] / 6.
            b[i - 1, i] = h[i - 1] / 6.

            d[i, i] = -1. / h[i - 1] - 1. / h[i]
            d[i, i - 1] = 1. / h[i - 1]
            d[i - 1, i] = 1. / h[i - 1]

        return b, d

    def _get_s(self, b, d):
        return d.T.dot(np.linalg.inv(b)).dot(d)

    def transform(self, x_new):
        exog = dmatrix(self.design_info, {"x": x_new})
        if self.ctransf is not None:
            exog = exog.dot(self.ctransf)
        return exog


class AdditiveGamSmoother(with_metaclass(ABCMeta)):
    """Base class for additive smooth components
    """
    def __init__(self, x, variable_names=None, include_intercept=False,
                 **kwargs):

        # get pandas names before using asarray
        if isinstance(x, pd.DataFrame):
            data_names = x.columns.tolist()
        elif isinstance(x, pd.Series):
            data_names = [x.name]
        else:
            data_names = None

        x = np.asarray(x)

        if x.ndim == 1:
            self.x = x.copy()
            self.x.shape = (len(x), 1)
        else:
            self.x = x

        self.nobs, self.k_variables = self.x.shape
        if isinstance(include_intercept, bool):
            self.include_intercept = [include_intercept] * self.k_variables
        else:
            self.include_intercept = include_intercept

        if variable_names is None:
            if data_names is not None:
                self.variable_names = data_names
            else:
                self.variable_names = ['x' + str(i)
                                       for i in range(self.k_variables)]
        else:
            self.variable_names = variable_names

        self.smoothers = self._make_smoothers_list()
        self.basis = np.hstack(list(smoother.basis
                               for smoother in self.smoothers))
        self.dim_basis = self.basis.shape[1]
        self.penalty_matrices = [smoother.cov_der2
                                 for smoother in self.smoothers]
        self.col_names = []
        for smoother in self.smoothers:
            self.col_names.extend(smoother.col_names)

        self.mask = []
        last_column = 0
        for smoother in self.smoothers:
            mask = np.array([False] * self.dim_basis)
            mask[last_column:smoother.dim_basis + last_column] = True
            last_column = last_column + smoother.dim_basis
            self.mask.append(mask)

    @abstractmethod
    def _make_smoothers_list(self):
        pass

    def transform(self, x_new):
        """create the spline basis for new observations

        The main use of this stateful transformation is for prediction
        using the same specification of the spline basis.

        Parameters
        ----------
        x_new: ndarray
            observations of the underlying explanatory variable

        Returns
        -------
        basis : ndarray
            design matrix for the spline basis for given ``x_new``.
        """
        if x_new.ndim == 1 and self.k_variables == 1:
            x_new = x_new.reshape(-1, 1)
        exog = np.hstack(list(self.smoothers[i].transform(x_new[:, i])
                         for i in range(self.k_variables)))
        return exog


class GenericSmoothers(AdditiveGamSmoother):
    """generic class for additive smooth components for GAM
    """
    def __init__(self, x, smoothers):
        self.smoothers = smoothers
        super(GenericSmoothers, self).__init__(x, variable_names=None)

    def _make_smoothers_list(self):
        return self.smoothers


class PolynomialSmoother(AdditiveGamSmoother):
    """additive polynomial components for GAM
    """
    def __init__(self, x, degrees, variable_names=None):
        self.degrees = degrees
        super(PolynomialSmoother, self).__init__(x,
                                                 variable_names=variable_names)

    def _make_smoothers_list(self):
        smoothers = []
        for v in range(self.k_variables):
            uv_smoother = UnivariatePolynomialSmoother(
                self.x[:, v],
                degree=self.degrees[v],
                variable_name=self.variable_names[v])
            smoothers.append(uv_smoother)
        return smoothers


class BSplines(AdditiveGamSmoother):
    """additive smooth components using B-Splines

    This creates and holds the B-Spline basis function for several
    components.

    Parameters
    ----------
    x : array_like, 1-D or 2-D
        underlying explanatory variable for smooth terms.
        If 2-dimensional, then observations should be in rows and
        explanatory variables in columns.
    df :  {int, array_like[int]}
        number of basis functions or degrees of freedom; should be equal
        in length to the number of columns of `x`; may be an integer if
        `x` has one column or is 1-D.
    degree : {int, array_like[int]}
        degree(s) of the spline; the same length and type rules apply as
        to `df`
    include_intercept : bool
        If False, then the basis functions are transformed so that they
        do not include a constant. This avoids perfect collinearity if
        a constant or several components are included in the model.
    constraints : {None, str, array}
        Constraints are used to transform the basis functions to satisfy
        those constraints.
        `constraints = 'center'` applies a linear transform to remove the
        constant and center the basis functions.
    variable_names : {list[str], None}
        The names for the underlying explanatory variables, x used in for
        creating the column and parameter names for the basis functions.
        If ``x`` is a pandas object, then the names will be taken from it.
    knot_kwds : None or list of dict
        option for the knot selection.
        By default knots are selected in the same way as in patsy, however the
        number of knots is independent of keeping or removing the constant.
        Interior knot selection is based on quantiles of the data and is the
        same in patsy and mgcv. Boundary points are at the limits of the data
        range.
        The available options use with `get_knots_bsplines` are

        - knots : None or array
          interior knots
        - spacing : 'quantile' or 'equal'
        - lower_bound : None or float
          location of lower boundary knots, all boundary knots are at the same
          point
        - upper_bound : None or float
          location of upper boundary knots, all boundary knots are at the same
          point
        - all_knots : None or array
          If all knots are provided, then those will be taken as given and
          all other options will be ignored.


    Attributes
    ----------
    smoothers : list of univariate smooth component instances
    basis : design matrix, array of spline bases columns for all components
    penalty_matrices : list of penalty matrices, one for each smooth term
    dim_basis : number of columns in the basis
    k_variables : number of smooth components
    col_names : created names for the basis columns

    There are additional attributes about the specification of the splines
    and some attributes mainly for internal use.

    Notes
    -----
    A constant in the spline basis function can be removed in two different
    ways.
    The first is by dropping one basis column and normalizing the
    remaining columns. This is obtained by the default
    ``include_intercept=False, constraints=None``
    The second option is by using the centering transform which is a linear
    transformation of all basis functions. As a consequence of the
    transformation, the B-spline basis functions do not have locally bounded
    support anymore. This is obtained ``constraints='center'``. In this case
    ``include_intercept`` will be automatically set to True to avoid
    dropping an additional column.
    """
    def __init__(self, x, df, degree, include_intercept=False,
                 constraints=None, variable_names=None, knot_kwds=None):
        if isinstance(degree, int):
            self.degrees = np.array([degree], dtype=int)
        else:
            self.degrees = degree
        if isinstance(df, int):
            self.dfs = np.array([df], dtype=int)
        else:
            self.dfs = df
        self.knot_kwds = knot_kwds
        # TODO: move attaching constraints to super call
        self.constraints = constraints
        if constraints == 'center':
            include_intercept = True

        super(BSplines, self).__init__(x, include_intercept=include_intercept,
                                       variable_names=variable_names)

    def _make_smoothers_list(self):
        smoothers = []
        for v in range(self.k_variables):
            kwds = self.knot_kwds[v] if self.knot_kwds else {}
            uv_smoother = UnivariateBSplines(
                self.x[:, v],
                df=self.dfs[v], degree=self.degrees[v],
                include_intercept=self.include_intercept[v],
                constraints=self.constraints,
                variable_name=self.variable_names[v], **kwds)
            smoothers.append(uv_smoother)

        return smoothers


class CubicSplines(AdditiveGamSmoother):
    """additive smooth components using cubic splines as in Wood 2006.

    Note, these splines do NOT use the same spline basis as
    ``Cubic Regression Splines``.
    """
    def __init__(self, x, df, constraints='center', transform='domain',
                 variable_names=None):
        self.dfs = df
        self.constraints = constraints
        self.transform = transform
        super(CubicSplines, self).__init__(x, constraints=constraints,
                                           variable_names=variable_names)

    def _make_smoothers_list(self):
        smoothers = []
        for v in range(self.k_variables):
            uv_smoother = UnivariateCubicSplines(
                            self.x[:, v], df=self.dfs[v],
                            constraints=self.constraints,
                            transform=self.transform,
                            variable_name=self.variable_names[v])
            smoothers.append(uv_smoother)

        return smoothers


class CyclicCubicSplines(AdditiveGamSmoother):
    """additive smooth components using cyclic cubic regression splines

    This spline basis is the same as in patsy.

    Parameters
    ----------
    x : array_like, 1-D or 2-D
        underlying explanatory variable for smooth terms.
        If 2-dimensional, then observations should be in rows and
        explanatory variables in columns.
    df :  int
        numer of basis functions or degrees of freedom
    constraints : {None, str, array}
        Constraints are used to transform the basis functions to satisfy
        those constraints.
    variable_names : {list[str], None}
        The names for the underlying explanatory variables, x used in for
        creating the column and parameter names for the basis functions.
        If ``x`` is a pandas object, then the names will be taken from it.
    """
    def __init__(self, x, df, constraints=None, variable_names=None):
        self.dfs = df
        # TODO: move attaching constraints to super call
        self.constraints = constraints
        super(CyclicCubicSplines, self).__init__(x,
                                                 variable_names=variable_names)

    def _make_smoothers_list(self):
        smoothers = []
        for v in range(self.k_variables):
            uv_smoother = UnivariateCubicCyclicSplines(
                self.x[:, v],
                df=self.dfs[v], constraints=self.constraints,
                variable_name=self.variable_names[v])
            smoothers.append(uv_smoother)

        return smoothers

# class CubicRegressionSplines(BaseCubicSplines):
#     # TODO: this class is still not tested
#
#     def __init__(self, x, df=10):
#         import warnings
#         warnings.warn("This class is still not tested and it is probably"
#                       " not working properly. "
#                       "I suggest to use another smoother", Warning)
#
#         super(CubicRegressionSplines, self).__init__(x, df)
#
#         self.basis = dmatrix("cc(x, df=" + str(df) + ") - 1", {"x": x})
#         n_inner_knots = df - 2 + 1 # +n_constraints
#         # TODO: ACcording to CubicRegressionSplines class this should be
#         #  n_inner_knots = df - 2
#         all_knots = _get_all_sorted_knots(x, n_inner_knots=n_inner_knots,
#                                           inner_knots=None,
#                                           lower_bound=None, upper_bound=None)
#
#         b, d = self._get_b_and_d(all_knots)
#         self.s = self._get_s(b, d)
#
#         self.dim_basis = self.basis.shape[1]
#
#     def _get_b_and_d(self, knots):
#
#         h = knots[1:] - knots[:-1]
#         n = knots.size - 1
#
#         # b and d are defined such that the penalty matrix is equivalent to:
#         # s = d.T.dot(b^-1).dot(d)
#         # reference in particular to pag 146 of Wood's book
#         b = np.zeros((n, n)) # the b matrix on page 146 of Wood's book
#         d = np.zeros((n, n)) # the d matrix on page 146 of Wood's book
#
#         for i in range(n-2):
#             d[i, i] = 1/h[i]
#             d[i, i+1] = -1/h[i] - 1/h[i+1]
#             d[i, i+2] = 1/h[i+1]
#
#             b[i, i] = (h[i] + h[i+1])/3
#
#         for i in range(n-3):
#             b[i, i+1] = h[i+1]/6
#             b[i+1, i] = h[i+1]/6
#
#         return b, d
#
#     def _get_s(self, b, d):
#
#         return d.T.dot(np.linalg.pinv(b)).dot(d)
