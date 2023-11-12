# This file is part of Patsy
# Copyright (C) 2014 GDF Suez, http://www.gdfsuez.com/
# See file LICENSE.txt for license information.

# R package 'mgcv' compatible cubic spline basis functions

# These are made available in the patsy.* namespace
__all__ = ["cr", "cc", "te"]

import numpy as np

from patsy.util import (have_pandas, atleast_2d_column_default,
                        no_pickling, assert_no_pickling, safe_string_eq)
from patsy.state import stateful_transform

if have_pandas:
    import pandas


def _get_natural_f(knots):
    """Returns mapping of natural cubic spline values to 2nd derivatives.

    .. note:: See 'Generalized Additive Models', Simon N. Wood, 2006, pp 145-146

    :param knots: The 1-d array knots used for cubic spline parametrization,
     must be sorted in ascending order.
    :return: A 2-d array mapping natural cubic spline values at
     knots to second derivatives.

    :raise ImportError: if scipy is not found, required for
     ``linalg.solve_banded()``
    """
    try:
        from scipy import linalg
    except ImportError: # pragma: no cover
        raise ImportError("Cubic spline functionality requires scipy.")

    h = knots[1:] - knots[:-1]
    diag = (h[:-1] + h[1:]) / 3.
    ul_diag = h[1:-1] / 6.
    banded_b = np.array([np.r_[0., ul_diag], diag, np.r_[ul_diag, 0.]])
    d = np.zeros((knots.size - 2, knots.size))
    for i in range(knots.size - 2):
        d[i, i] = 1. / h[i]
        d[i, i + 2] = 1. / h[i + 1]
        d[i, i + 1] = - d[i, i] - d[i, i + 2]

    fm = linalg.solve_banded((1, 1), banded_b, d)

    return np.vstack([np.zeros(knots.size), fm, np.zeros(knots.size)])


# Cyclic Cubic Regression Splines


def _map_cyclic(x, lbound, ubound):
    """Maps values into the interval [lbound, ubound] in a cyclic fashion.

    :param x: The 1-d array values to be mapped.
    :param lbound: The lower bound of the interval.
    :param ubound: The upper bound of the interval.
    :return: A new 1-d array containing mapped x values.

    :raise ValueError: if lbound >= ubound.
    """
    if lbound >= ubound:
        raise ValueError("Invalid argument: lbound (%r) should be "
                         "less than ubound (%r)."
                         % (lbound, ubound))

    x = np.copy(x)
    x[x > ubound] = lbound + (x[x > ubound] - ubound) % (ubound - lbound)
    x[x < lbound] = ubound - (lbound - x[x < lbound]) % (ubound - lbound)

    return x


def test__map_cyclic():
    x = np.array([1.5, 2.6, 0.1, 4.4, 10.7])
    x_orig = np.copy(x)
    expected_mapped_x = np.array([3.0, 2.6, 3.1, 2.9, 3.2])
    mapped_x = _map_cyclic(x, 2.1, 3.6)
    assert np.allclose(x, x_orig)
    assert np.allclose(mapped_x, expected_mapped_x)


def test__map_cyclic_errors():
    import pytest
    x = np.linspace(0.2, 5.7, 10)
    pytest.raises(ValueError, _map_cyclic, x, 4.5, 3.6)
    pytest.raises(ValueError, _map_cyclic, x, 4.5, 4.5)


def _get_cyclic_f(knots):
    """Returns mapping of cyclic cubic spline values to 2nd derivatives.

    .. note:: See 'Generalized Additive Models', Simon N. Wood, 2006, pp 146-147

    :param knots: The 1-d array knots used for cubic spline parametrization,
     must be sorted in ascending order.
    :return: A 2-d array mapping cyclic cubic spline values at
     knots to second derivatives.
    """
    h = knots[1:] - knots[:-1]
    n = knots.size - 1
    b = np.zeros((n, n))
    d = np.zeros((n, n))

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

    return np.linalg.solve(b, d)


# Tensor Product


def _row_tensor_product(dms):
    """Computes row-wise tensor product of given arguments.

    .. note:: Custom algorithm to precisely match what is done in 'mgcv',
    in particular look out for order of result columns!
    For reference implementation see 'mgcv' source code,
    file 'mat.c', mgcv_tensor_mm(), l.62

    :param dms: A sequence of 2-d arrays (marginal design matrices).
    :return: The 2-d array row-wise tensor product of given arguments.

    :raise ValueError: if argument sequence is empty, does not contain only
     2-d arrays or if the arrays number of rows does not match.
    """
    if len(dms) == 0:
        raise ValueError("Tensor product arrays sequence should not be empty.")
    for dm in dms:
        if dm.ndim != 2:
            raise ValueError("Tensor product arguments should be 2-d arrays.")

    tp_nrows = dms[0].shape[0]
    tp_ncols = 1
    for dm in dms:
        if dm.shape[0] != tp_nrows:
            raise ValueError("Tensor product arguments should have "
                             "same number of rows.")
        tp_ncols *= dm.shape[1]
    tp = np.zeros((tp_nrows, tp_ncols))
    tp[:, -dms[-1].shape[1]:] = dms[-1]
    filled_tp_ncols = dms[-1].shape[1]
    for dm in dms[-2::-1]:
        p = - filled_tp_ncols * dm.shape[1]
        for j in range(dm.shape[1]):
            xj = dm[:, j]
            for t in range(-filled_tp_ncols, 0):
                tp[:, p] = tp[:, t] * xj
                p += 1
        filled_tp_ncols *= dm.shape[1]

    return tp


def test__row_tensor_product_errors():
    import pytest
    pytest.raises(ValueError, _row_tensor_product, [])
    pytest.raises(ValueError, _row_tensor_product, [np.arange(1, 5)])
    pytest.raises(ValueError, _row_tensor_product,
                  [np.arange(1, 5), np.arange(1, 5)])
    pytest.raises(ValueError, _row_tensor_product,
                  [np.arange(1, 13).reshape((3, 4)),
                   np.arange(1, 13).reshape((4, 3))])


def test__row_tensor_product():
    # Testing cases where main input array should not be modified
    dm1 = np.arange(1, 17).reshape((4, 4))
    assert np.array_equal(_row_tensor_product([dm1]), dm1)
    ones = np.ones(4).reshape((4, 1))
    tp1 = _row_tensor_product([ones, dm1])
    assert np.array_equal(tp1, dm1)
    tp2 = _row_tensor_product([dm1, ones])
    assert np.array_equal(tp2, dm1)

    # Testing cases where main input array should be scaled
    twos = 2 * ones
    tp3 = _row_tensor_product([twos, dm1])
    assert np.array_equal(tp3, 2 * dm1)
    tp4 = _row_tensor_product([dm1, twos])
    assert np.array_equal(tp4, 2 * dm1)

    # Testing main cases
    dm2 = np.array([[1, 2], [1, 2]])
    dm3 = np.arange(1, 7).reshape((2, 3))
    expected_tp5 = np.array([[1,  2,  3,  2,  4,  6],
                             [4,  5,  6,  8, 10, 12]])
    tp5 = _row_tensor_product([dm2, dm3])
    assert np.array_equal(tp5, expected_tp5)
    expected_tp6 = np.array([[1,  2,  2,  4,  3,  6],
                             [4,  8,  5, 10,  6, 12]])
    tp6 = _row_tensor_product([dm3, dm2])
    assert np.array_equal(tp6, expected_tp6)


# Common code


def _find_knots_lower_bounds(x, knots):
    """Finds knots lower bounds for given values.

    Returns an array of indices ``I`` such that
    ``0 <= I[i] <= knots.size - 2`` for all ``i``
    and
    ``knots[I[i]] < x[i] <= knots[I[i] + 1]`` if
    ``np.min(knots) < x[i] <= np.max(knots)``,
    ``I[i] = 0`` if ``x[i] <= np.min(knots)``
    ``I[i] = knots.size - 2`` if ``np.max(knots) < x[i]``

    :param x: The 1-d array values whose knots lower bounds are to be found.
    :param knots: The 1-d array knots used for cubic spline parametrization,
     must be sorted in ascending order.
    :return: An array of knots lower bounds indices.
    """
    lb = np.searchsorted(knots, x) - 1

    # I[i] = 0 for x[i] <= np.min(knots)
    lb[lb == -1] = 0

    # I[i] = knots.size - 2 for x[i] > np.max(knots)
    lb[lb == knots.size - 1] = knots.size - 2

    return lb


def _compute_base_functions(x, knots):
    """Computes base functions used for building cubic splines basis.

    .. note:: See 'Generalized Additive Models', Simon N. Wood, 2006, p. 146
      and for the special treatment of ``x`` values outside ``knots`` range
      see 'mgcv' source code, file 'mgcv.c', function 'crspl()', l.249

    :param x: The 1-d array values for which base functions should be computed.
    :param knots: The 1-d array knots used for cubic spline parametrization,
     must be sorted in ascending order.
    :return: 4 arrays corresponding to the 4 base functions ajm, ajp, cjm, cjp
     + the 1-d array of knots lower bounds indices corresponding to
     the given ``x`` values.
    """
    j = _find_knots_lower_bounds(x, knots)

    h = knots[1:] - knots[:-1]
    hj = h[j]
    xj1_x = knots[j + 1] - x
    x_xj = x - knots[j]

    ajm = xj1_x / hj
    ajp = x_xj / hj

    cjm_3 = xj1_x * xj1_x * xj1_x / (6. * hj)
    cjm_3[x > np.max(knots)] = 0.
    cjm_1 = hj * xj1_x / 6.
    cjm = cjm_3 - cjm_1

    cjp_3 = x_xj * x_xj * x_xj / (6. * hj)
    cjp_3[x < np.min(knots)] = 0.
    cjp_1 = hj * x_xj / 6.
    cjp = cjp_3 - cjp_1

    return ajm, ajp, cjm, cjp, j


def _absorb_constraints(design_matrix, constraints):
    """Absorb model parameters constraints into the design matrix.

    :param design_matrix: The (2-d array) initial design matrix.
    :param constraints: The 2-d array defining initial model parameters
     (``betas``) constraints (``np.dot(constraints, betas) = 0``).
    :return: The new design matrix with absorbed parameters constraints.

    :raise ImportError: if scipy is not found, used for ``scipy.linalg.qr()``
      which is cleaner than numpy's version requiring a call like
      ``qr(..., mode='complete')`` to get a full QR decomposition.
    """
    try:
        from scipy import linalg
    except ImportError: # pragma: no cover
        raise ImportError("Cubic spline functionality requires scipy.")

    m = constraints.shape[0]
    q, r = linalg.qr(np.transpose(constraints))

    return np.dot(design_matrix, q[:, m:])


def _get_free_crs_dmatrix(x, knots, cyclic=False):
    """Builds an unconstrained cubic regression spline design matrix.

    Returns design matrix with dimensions ``len(x) x n``
    for a cubic regression spline smoother
    where
     - ``n = len(knots)`` for natural CRS
     - ``n = len(knots) - 1`` for cyclic CRS

    .. note:: See 'Generalized Additive Models', Simon N. Wood, 2006, p. 145

    :param x: The 1-d array values.
    :param knots: The 1-d array knots used for cubic spline parametrization,
     must be sorted in ascending order.
    :param cyclic: Indicates whether used cubic regression splines should
     be cyclic or not. Default is ``False``.
    :return: The (2-d array) design matrix.
    """
    n = knots.size
    if cyclic:
        x = _map_cyclic(x, min(knots), max(knots))
        n -= 1

    ajm, ajp, cjm, cjp, j = _compute_base_functions(x, knots)

    j1 = j + 1
    if cyclic:
        j1[j1 == n] = 0

    i = np.identity(n)

    if cyclic:
        f = _get_cyclic_f(knots)
    else:
        f = _get_natural_f(knots)

    dmt = ajm * i[j, :].T + ajp * i[j1, :].T + \
        cjm * f[j, :].T + cjp * f[j1, :].T

    return dmt.T


def _get_crs_dmatrix(x, knots, constraints=None, cyclic=False):
    """Builds a cubic regression spline design matrix.

    Returns design matrix with dimensions len(x) x n
    where:
     - ``n = len(knots) - nrows(constraints)`` for natural CRS
     - ``n = len(knots) - nrows(constraints) - 1`` for cyclic CRS
    for a cubic regression spline smoother

    :param x: The 1-d array values.
    :param knots: The 1-d array knots used for cubic spline parametrization,
     must be sorted in ascending order.
    :param constraints: The 2-d array defining model parameters (``betas``)
     constraints (``np.dot(constraints, betas) = 0``).
    :param cyclic: Indicates whether used cubic regression splines should
     be cyclic or not. Default is ``False``.
    :return: The (2-d array) design matrix.
    """
    dm = _get_free_crs_dmatrix(x, knots, cyclic)
    if constraints is not None:
        dm = _absorb_constraints(dm, constraints)

    return dm


def _get_te_dmatrix(design_matrices, constraints=None):
    """Builds tensor product design matrix, given the marginal design matrices.

    :param design_matrices: A sequence of 2-d arrays (marginal design matrices).
    :param constraints: The 2-d array defining model parameters (``betas``)
     constraints (``np.dot(constraints, betas) = 0``).
    :return: The (2-d array) design matrix.
    """
    dm = _row_tensor_product(design_matrices)
    if constraints is not None:
        dm = _absorb_constraints(dm, constraints)

    return dm


# Stateful Transforms


def _get_all_sorted_knots(x, n_inner_knots=None, inner_knots=None,
                              lower_bound=None, upper_bound=None):
    """Gets all knots locations with lower and upper exterior knots included.

    If needed, inner knots are computed as equally spaced quantiles of the
    input data falling between given lower and upper bounds.

    :param x: The 1-d array data values.
    :param n_inner_knots: Number of inner knots to compute.
    :param inner_knots: Provided inner knots if any.
    :param lower_bound: The lower exterior knot location. If unspecified, the
     minimum of ``x`` values is used.
    :param upper_bound: The upper exterior knot location. If unspecified, the
     maximum of ``x`` values is used.
    :return: The array of ``n_inner_knots + 2`` distinct knots.

    :raise ValueError: for various invalid parameters sets or if unable to
     compute ``n_inner_knots + 2`` distinct knots.
    """
    if lower_bound is None and x.size == 0:
        raise ValueError("Cannot set lower exterior knot location: empty "
                         "input data and lower_bound not specified.")
    elif lower_bound is None and x.size != 0:
        lower_bound = np.min(x)

    if upper_bound is None and x.size == 0:
        raise ValueError("Cannot set upper exterior knot location: empty "
                         "input data and upper_bound not specified.")
    elif upper_bound is None and x.size != 0:
        upper_bound = np.max(x)

    if upper_bound < lower_bound:
        raise ValueError("lower_bound > upper_bound (%r > %r)"
                         % (lower_bound, upper_bound))

    if inner_knots is None and n_inner_knots is not None:
        if n_inner_knots < 0:
            raise ValueError("Invalid requested number of inner knots: %r"
                             % (n_inner_knots,))

        x = x[(lower_bound <= x) & (x <= upper_bound)]
        x = np.unique(x)

        if x.size != 0:
            inner_knots_q = np.linspace(0, 100, n_inner_knots + 2)[1:-1]
            # .tolist() is necessary to work around a bug in numpy 1.8
            inner_knots = np.asarray(np.percentile(x, inner_knots_q.tolist()))
        elif n_inner_knots == 0:
            inner_knots = np.array([])
        else:
            raise ValueError("No data values between lower_bound(=%r) and "
                             "upper_bound(=%r): cannot compute requested "
                             "%r inner knot(s)."
                             % (lower_bound, upper_bound, n_inner_knots))
    elif inner_knots is not None:
        inner_knots = np.unique(inner_knots)
        if n_inner_knots is not None and n_inner_knots != inner_knots.size:
            raise ValueError("Needed number of inner knots=%r does not match "
                             "provided number of inner knots=%r."
                             % (n_inner_knots, inner_knots.size))
        n_inner_knots = inner_knots.size
        if np.any(inner_knots < lower_bound):
            raise ValueError("Some knot values (%s) fall below lower bound "
                             "(%r)."
                             % (inner_knots[inner_knots < lower_bound],
                                lower_bound))
        if np.any(inner_knots > upper_bound):
            raise ValueError("Some knot values (%s) fall above upper bound "
                             "(%r)."
                             % (inner_knots[inner_knots > upper_bound],
                                upper_bound))
    else:
        raise ValueError("Must specify either 'n_inner_knots' or 'inner_knots'.")

    all_knots = np.concatenate(([lower_bound, upper_bound], inner_knots))
    all_knots = np.unique(all_knots)
    if all_knots.size != n_inner_knots + 2:
        raise ValueError("Unable to compute n_inner_knots(=%r) + 2 distinct "
                         "knots: %r data value(s) found between "
                         "lower_bound(=%r) and upper_bound(=%r)."
                         % (n_inner_knots, x.size, lower_bound, upper_bound))

    return all_knots


def test__get_all_sorted_knots():
    import pytest
    pytest.raises(ValueError, _get_all_sorted_knots,
                  np.array([]), -1)
    pytest.raises(ValueError, _get_all_sorted_knots,
                  np.array([]), 0)
    pytest.raises(ValueError, _get_all_sorted_knots,
                  np.array([]), 0, lower_bound=1)
    pytest.raises(ValueError, _get_all_sorted_knots,
                  np.array([]), 0, upper_bound=5)
    pytest.raises(ValueError, _get_all_sorted_knots,
                  np.array([]), 0, lower_bound=3, upper_bound=1)
    assert np.array_equal(
        _get_all_sorted_knots(np.array([]), 0, lower_bound=1, upper_bound=5),
        [1, 5])
    pytest.raises(ValueError, _get_all_sorted_knots,
                  np.array([]), 0, lower_bound=1, upper_bound=1)
    x = np.arange(6) * 2
    pytest.raises(ValueError, _get_all_sorted_knots,
                  x, -2)
    assert np.array_equal(
        _get_all_sorted_knots(x, 0),
        [0, 10])
    assert np.array_equal(
        _get_all_sorted_knots(x, 0, lower_bound=3, upper_bound=8),
        [3, 8])
    assert np.array_equal(
        _get_all_sorted_knots(x, 2, lower_bound=1, upper_bound=9),
        [1, 4, 6, 9])
    pytest.raises(ValueError, _get_all_sorted_knots,
                  x, 2, lower_bound=1, upper_bound=3)
    pytest.raises(ValueError, _get_all_sorted_knots,
                  x, 1, lower_bound=1.3, upper_bound=1.4)
    assert np.array_equal(
        _get_all_sorted_knots(x, 1, lower_bound=1, upper_bound=3),
        [1, 2, 3])
    pytest.raises(ValueError, _get_all_sorted_knots,
                  x, 1, lower_bound=2, upper_bound=3)
    pytest.raises(ValueError, _get_all_sorted_knots,
                  x, 1, inner_knots=[2, 3])
    pytest.raises(ValueError, _get_all_sorted_knots,
                  x, lower_bound=2, upper_bound=3)
    assert np.array_equal(
        _get_all_sorted_knots(x, inner_knots=[3, 7]),
        [0, 3, 7, 10])
    assert np.array_equal(
        _get_all_sorted_knots(x, inner_knots=[3, 7], lower_bound=2),
        [2, 3, 7, 10])
    pytest.raises(ValueError, _get_all_sorted_knots,
                  x, inner_knots=[3, 7], lower_bound=4)
    pytest.raises(ValueError, _get_all_sorted_knots,
                  x, inner_knots=[3, 7], upper_bound=6)


def _get_centering_constraint_from_dmatrix(design_matrix):
    """ Computes the centering constraint from the given design matrix.

    We want to ensure that if ``b`` is the array of parameters, our
    model is centered, ie ``np.mean(np.dot(design_matrix, b))`` is zero.
    We can rewrite this as ``np.dot(c, b)`` being zero with ``c`` a 1-row
    constraint matrix containing the mean of each column of ``design_matrix``.

    :param design_matrix: The 2-d array design matrix.
    :return: A 2-d array (1 x ncols(design_matrix)) defining the
     centering constraint.
    """
    return design_matrix.mean(axis=0).reshape((1, design_matrix.shape[1]))


class CubicRegressionSpline(object):
    """Base class for cubic regression spline stateful transforms

    This class contains all the functionality for the following stateful
    transforms:
     - ``cr(x, df=None, knots=None, lower_bound=None, upper_bound=None, constraints=None)``
       for natural cubic regression spline
     - ``cc(x, df=None, knots=None, lower_bound=None, upper_bound=None, constraints=None)``
       for cyclic cubic regression spline
    """
    common_doc = """
    :arg df: The number of degrees of freedom to use for this spline. The
      return value will have this many columns. You must specify at least one
      of ``df`` and ``knots``.
    :arg knots: The interior knots to use for the spline. If unspecified, then
      equally spaced quantiles of the input data are used. You must specify at
      least one of ``df`` and ``knots``.
    :arg lower_bound: The lower exterior knot location.
    :arg upper_bound: The upper exterior knot location.
    :arg constraints: Either a 2-d array defining general linear constraints
     (that is ``np.dot(constraints, betas)`` is zero, where ``betas`` denotes
     the array of *initial* parameters, corresponding to the *initial*
     unconstrained design matrix), or the string
     ``'center'`` indicating that we should apply a centering constraint
     (this constraint will be computed from the input data, remembered and
     re-used for prediction from the fitted model).
     The constraints are absorbed in the resulting design matrix which means
     that the model is actually rewritten in terms of
     *unconstrained* parameters. For more details see :ref:`spline-regression`.

    This is a stateful transforms (for details see
    :ref:`stateful-transforms`). If ``knots``, ``lower_bound``, or
    ``upper_bound`` are not specified, they will be calculated from the data
    and then the chosen values will be remembered and re-used for prediction
    from the fitted model.

    Using this function requires scipy be installed.

    .. versionadded:: 0.3.0
    """

    def __init__(self, name, cyclic):
        self._name = name
        self._cyclic = cyclic
        self._tmp = {}
        self._all_knots = None
        self._constraints = None

    def memorize_chunk(self, x, df=None, knots=None,
                       lower_bound=None, upper_bound=None,
                       constraints=None):
        args = {"df": df,
                "knots": knots,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "constraints": constraints,
                }
        self._tmp["args"] = args

        x = np.atleast_1d(x)
        if x.ndim == 2 and x.shape[1] == 1:
            x = x[:, 0]
        if x.ndim > 1:
            raise ValueError("Input to %r must be 1-d, "
                             "or a 2-d column vector."
                             % (self._name,))

        self._tmp.setdefault("xs", []).append(x)

    def memorize_finish(self):
        args = self._tmp["args"]
        xs = self._tmp["xs"]
        # Guards against invalid subsequent memorize_chunk() calls.
        del self._tmp

        x = np.concatenate(xs)
        if args["df"] is None and args["knots"] is None:
            raise ValueError("Must specify either 'df' or 'knots'.")

        constraints = args["constraints"]
        n_constraints = 0
        if constraints is not None:
            if safe_string_eq(constraints, "center"):
                # Here we collect only number of constraints,
                # actual centering constraint will be computed after all_knots
                n_constraints = 1
            else:
                constraints = np.atleast_2d(constraints)
                if constraints.ndim != 2:
                    raise ValueError("Constraints must be 2-d array or "
                                     "1-d vector.")
                n_constraints = constraints.shape[0]

        n_inner_knots = None
        if args["df"] is not None:
            min_df = 1
            if not self._cyclic and n_constraints == 0:
                min_df = 2
            if args["df"] < min_df:
                raise ValueError("'df'=%r must be greater than or equal to %r."
                                 % (args["df"], min_df))
            n_inner_knots = args["df"] - 2 + n_constraints
            if self._cyclic:
                n_inner_knots += 1
        self._all_knots = _get_all_sorted_knots(x,
                                                n_inner_knots=n_inner_knots,
                                                inner_knots=args["knots"],
                                                lower_bound=args["lower_bound"],
                                                upper_bound=args["upper_bound"])
        if constraints is not None:
            if safe_string_eq(constraints, "center"):
                # Now we can compute centering constraints
                constraints = _get_centering_constraint_from_dmatrix(
                    _get_free_crs_dmatrix(x, self._all_knots, cyclic=self._cyclic)
                )

            df_before_constraints = self._all_knots.size
            if self._cyclic:
                df_before_constraints -= 1
            if constraints.shape[1] != df_before_constraints:
                raise ValueError("Constraints array should have %r columns but"
                                 " %r found."
                                 % (df_before_constraints, constraints.shape[1]))
            self._constraints = constraints

    def transform(self, x, df=None, knots=None,
                  lower_bound=None, upper_bound=None,
                  constraints=None):
        x_orig = x
        x = np.atleast_1d(x)
        if x.ndim == 2 and x.shape[1] == 1:
            x = x[:, 0]
        if x.ndim > 1:
            raise ValueError("Input to %r must be 1-d, "
                             "or a 2-d column vector."
                             % (self._name,))
        dm = _get_crs_dmatrix(x, self._all_knots,
                              self._constraints, cyclic=self._cyclic)
        if have_pandas:
            if isinstance(x_orig, (pandas.Series, pandas.DataFrame)):
                dm = pandas.DataFrame(dm)
                dm.index = x_orig.index
        return dm

    __getstate__ = no_pickling


class CR(CubicRegressionSpline):
    """cr(x, df=None, knots=None, lower_bound=None, upper_bound=None, constraints=None)

    Generates a natural cubic spline basis for ``x``
    (with the option of absorbing centering or more general parameters
    constraints), allowing non-linear fits. The usual usage is something like::

      y ~ 1 + cr(x, df=5, constraints='center')

    to fit ``y`` as a smooth function of ``x``, with 5 degrees of freedom
    given to the smooth, and centering constraint absorbed in
    the resulting design matrix. Note that in this example, due to the centering
    constraint, 6 knots will get computed from the input data ``x``
    to achieve 5 degrees of freedom.


    .. note:: This function reproduce the cubic regression splines 'cr' and 'cs'
      as implemented in the R package 'mgcv' (GAM modelling).

    """

    # Under python -OO, __doc__ will be defined but set to None
    if __doc__:
        __doc__ += CubicRegressionSpline.common_doc

    def __init__(self):
        CubicRegressionSpline.__init__(self, name='cr', cyclic=False)

cr = stateful_transform(CR)


class CC(CubicRegressionSpline):
    """cc(x, df=None, knots=None, lower_bound=None, upper_bound=None, constraints=None)

    Generates a cyclic cubic spline basis for ``x``
    (with the option of absorbing centering or more general parameters
    constraints), allowing non-linear fits. The usual usage is something like::

      y ~ 1 + cc(x, df=7, constraints='center')

    to fit ``y`` as a smooth function of ``x``, with 7 degrees of freedom
    given to the smooth, and centering constraint absorbed in
    the resulting design matrix. Note that in this example, due to the centering
    and cyclic constraints, 9 knots will get computed from the input data ``x``
    to achieve 7 degrees of freedom.

    .. note:: This function reproduce the cubic regression splines 'cc'
      as implemented in the R package 'mgcv' (GAM modelling).

    """

    # Under python -OO, __doc__ will be defined but set to None
    if __doc__:
        __doc__ += CubicRegressionSpline.common_doc

    def __init__(self):
        CubicRegressionSpline.__init__(self, name='cc', cyclic=True)

cc = stateful_transform(CC)


def test_crs_errors():
    import pytest
    # Invalid 'x' shape
    pytest.raises(ValueError, cr, np.arange(16).reshape((4, 4)), df=4)
    pytest.raises(ValueError, CR().transform,
                  np.arange(16).reshape((4, 4)), df=4)
    # Should provide at least 'df' or 'knots'
    pytest.raises(ValueError, cr, np.arange(50))
    # Invalid constraints shape
    pytest.raises(ValueError, cr, np.arange(50), df=4,
                  constraints=np.arange(27).reshape((3, 3, 3)))
    # Invalid nb of columns in constraints
    # (should have df + 1 = 5, but 6 provided)
    pytest.raises(ValueError, cr, np.arange(50), df=4,
                  constraints=np.arange(6))
    # Too small 'df' for natural cubic spline
    pytest.raises(ValueError, cr, np.arange(50), df=1)
    # Too small 'df' for cyclic cubic spline
    pytest.raises(ValueError, cc, np.arange(50), df=0)


def test_crs_compat():
    from patsy.test_state import check_stateful
    from patsy.test_splines_crs_data import (R_crs_test_x,
                                             R_crs_test_data,
                                             R_crs_num_tests)
    lines = R_crs_test_data.split("\n")
    tests_ran = 0
    start_idx = lines.index("--BEGIN TEST CASE--")
    while True:
        if not lines[start_idx] == "--BEGIN TEST CASE--":
            break
        start_idx += 1
        stop_idx = lines.index("--END TEST CASE--", start_idx)
        block = lines[start_idx:stop_idx]
        test_data = {}
        for line in block:
            key, value = line.split("=", 1)
            test_data[key] = value
        # Translate the R output into Python calling conventions
        adjust_df = 0
        if test_data["spline_type"] == "cr" or test_data["spline_type"] == "cs":
            spline_type = CR
        elif test_data["spline_type"] == "cc":
            spline_type = CC
            adjust_df += 1
        else:
            raise ValueError("Unrecognized spline type %r"
                             % (test_data["spline_type"],))
        kwargs = {}
        if test_data["absorb_cons"] == "TRUE":
            kwargs["constraints"] = "center"
            adjust_df += 1
        if test_data["knots"] != "None":
            all_knots = np.asarray(eval(test_data["knots"]))
            all_knots.sort()
            kwargs["knots"] = all_knots[1:-1]
            kwargs["lower_bound"] = all_knots[0]
            kwargs["upper_bound"] = all_knots[-1]
        else:
            kwargs["df"] = eval(test_data["nb_knots"]) - adjust_df
        output = np.asarray(eval(test_data["output"]))
        # Do the actual test
        check_stateful(spline_type, False, R_crs_test_x, output, **kwargs)
        tests_ran += 1
        # Set up for the next one
        start_idx = stop_idx + 1
    assert tests_ran == R_crs_num_tests

test_crs_compat.slow = True

def test_crs_with_specific_constraint():
    from patsy.highlevel import incr_dbuilder, build_design_matrices, dmatrix
    x = (-1.5)**np.arange(20)
    # Hard coded R values for smooth: s(x, bs="cr", k=5)
    # R> knots <- smooth$xp
    knots_R = np.array([-2216.837820053100585937,
                        -50.456909179687500000,
                        -0.250000000000000000,
                        33.637939453125000000,
                        1477.891880035400390625])
    # R> centering.constraint <- t(qr.X(attr(smooth, "qrc")))
    centering_constraint_R = np.array([[0.064910676323168478574,
                                        1.4519875239407085132,
                                        -2.1947446912471946234,
                                        1.6129783104357671153,
                                        0.064868180547550072235]])
    # values for which we want a prediction
    new_x = np.array([-3000., -200., 300., 2000.])
    result1 = dmatrix("cr(new_x, knots=knots_R[1:-1], "
                      "lower_bound=knots_R[0], upper_bound=knots_R[-1], "
                      "constraints=centering_constraint_R)")

    data_chunked = [{"x": x[:10]}, {"x": x[10:]}]
    new_data = {"x": new_x}
    builder = incr_dbuilder("cr(x, df=4, constraints='center')",
                            lambda: iter(data_chunked))
    result2 = build_design_matrices([builder], new_data)[0]

    assert np.allclose(result1, result2, rtol=1e-12, atol=0.)


class TE(object):
    """te(s1, .., sn, constraints=None)

    Generates smooth of several covariates as a tensor product of the bases
    of marginal univariate smooths ``s1, .., sn``. The marginal smooths are
    required to transform input univariate data into some kind of smooth
    functions basis producing a 2-d array output with the ``(i, j)`` element
    corresponding to the value of the ``j`` th basis function at the ``i`` th
    data point.
    The resulting basis dimension is the product of the basis dimensions of
    the marginal smooths. The usual usage is something like::

      y ~ 1 + te(cr(x1, df=5), cc(x2, df=6), constraints='center')

    to fit ``y`` as a smooth function of both ``x1`` and ``x2``, with a natural
    cubic spline for ``x1`` marginal smooth and a cyclic cubic spline for
    ``x2`` (and centering constraint absorbed in the resulting design matrix).

    :arg constraints: Either a 2-d array defining general linear constraints
     (that is ``np.dot(constraints, betas)`` is zero, where ``betas`` denotes
     the array of *initial* parameters, corresponding to the *initial*
     unconstrained design matrix), or the string
     ``'center'`` indicating that we should apply a centering constraint
     (this constraint will be computed from the input data, remembered and
     re-used for prediction from the fitted model).
     The constraints are absorbed in the resulting design matrix which means
     that the model is actually rewritten in terms of
     *unconstrained* parameters. For more details see :ref:`spline-regression`.

    Using this function requires scipy be installed.

    .. note:: This function reproduce the tensor product smooth 'te' as
      implemented in the R package 'mgcv' (GAM modelling).
      See also 'Generalized Additive Models', Simon N. Wood, 2006, pp 158-163

    .. versionadded:: 0.3.0
    """
    def __init__(self):
        self._tmp = {}
        self._constraints = None

    def memorize_chunk(self, *args, **kwargs):
        constraints = self._tmp.setdefault("constraints",
                                           kwargs.get("constraints"))
        if safe_string_eq(constraints, "center"):
            args_2d = []
            for arg in args:
                arg = atleast_2d_column_default(arg)
                if arg.ndim != 2:
                    raise ValueError("Each tensor product argument must be "
                                     "a 2-d array or 1-d vector.")
                args_2d.append(arg)

            tp = _row_tensor_product(args_2d)
            self._tmp.setdefault("count", 0)
            self._tmp["count"] += tp.shape[0]

            chunk_sum = np.atleast_2d(tp.sum(axis=0))
            self._tmp.setdefault("sum", np.zeros(chunk_sum.shape))
            self._tmp["sum"] += chunk_sum

    def memorize_finish(self):
        tmp = self._tmp
        constraints = self._tmp["constraints"]
        # Guards against invalid subsequent memorize_chunk() calls.
        del self._tmp

        if constraints is not None:
            if safe_string_eq(constraints, "center"):
                constraints = np.atleast_2d(tmp["sum"] / tmp["count"])
            else:
                constraints = np.atleast_2d(constraints)
                if constraints.ndim != 2:
                    raise ValueError("Constraints must be 2-d array or "
                                     "1-d vector.")

        self._constraints = constraints

    def transform(self, *args, **kwargs):
        args_2d = []
        for arg in args:
            arg = atleast_2d_column_default(arg)
            if arg.ndim != 2:
                raise ValueError("Each tensor product argument must be "
                                 "a 2-d array or 1-d vector.")
            args_2d.append(arg)

        return _get_te_dmatrix(args_2d, self._constraints)

    __getstate__ = no_pickling

te = stateful_transform(TE)


def test_te_errors():
    import pytest
    x = np.arange(27)
    # Invalid input shape
    pytest.raises(ValueError, te, x.reshape((3, 3, 3)))
    pytest.raises(ValueError, te, x.reshape((3, 3, 3)), constraints='center')
    # Invalid constraints shape
    pytest.raises(ValueError, te, x,
                  constraints=np.arange(8).reshape((2, 2, 2)))


def test_te_1smooth():
    from patsy.splines import bs
    # Tensor product of 1 smooth covariate should be the same
    # as the smooth alone
    x = (-1.5)**np.arange(20)
    assert np.allclose(cr(x, df=6), te(cr(x, df=6)))
    assert np.allclose(cc(x, df=5), te(cc(x, df=5)))
    assert np.allclose(bs(x, df=4), te(bs(x, df=4)))
    # Adding centering constraint to tensor product
    assert np.allclose(cr(x, df=3, constraints='center'),
                       te(cr(x, df=4), constraints='center'))
    # Adding specific constraint
    center_constraint = np.arange(1, 5)
    assert np.allclose(cr(x, df=3, constraints=center_constraint),
                       te(cr(x, df=4), constraints=center_constraint))


def test_te_2smooths():
    from patsy.highlevel import incr_dbuilder, build_design_matrices
    x1 = (-1.5)**np.arange(20)
    x2 = (1.6)**np.arange(20)
    # Hard coded R results for smooth: te(x1, x2, bs=c("cs", "cc"), k=c(5,7))
    # Without centering constraint:
    dmatrix_R_nocons = \
        np.array([[-4.4303024184609255207e-06,  7.9884438387230142235e-06,
                   9.7987758194797719025e-06,   -7.2894213245475212959e-08,
                   1.5907686862964493897e-09,   -3.2565884983072595159e-11,
                   0.0170749607855874667439,    -3.0788499835965849050e-02,
                   -3.7765754357352458725e-02,  2.8094376299826799787e-04,
                   -6.1310290747349201414e-06,  1.2551314933193442915e-07,
                   -0.26012671685838206770,     4.6904420337437874311e-01,
                   0.5753384627946153129230,    -4.2800085814700449330e-03,
                   9.3402525733484874533e-05,   -1.9121170389937518131e-06,
                   -0.0904312240489447832781,   1.6305991924427923334e-01,
                   2.0001237112941641638e-01,   -1.4879148887003382663e-03,
                   3.2470731316462736135e-05,   -6.6473404365914134499e-07,
                   2.0447857920168824846e-05,   -3.6870296695050991799e-05,
                   -4.5225801045409022233e-05,  3.3643990293641665710e-07,
                   -7.3421200200015877329e-09,  1.5030635073660743297e-10],
                  [-9.4006130602653794302e-04,  7.8681398069163730347e-04,
                   2.4573006857381437217e-04,   -1.4524712230452725106e-04,
                   7.8216741353106329551e-05,   -3.1304283003914264551e-04,
                   3.6231183382798337611064,    -3.0324832476174168328e+00,
                   -9.4707559178211142559e-01,  5.5980126937492580286e-01,
                   -3.0145747744342332730e-01,  1.2065077148806895302e+00,
                   -35.17561267504181188315,    2.9441339255948005160e+01,
                   9.1948319320782125885216,    -5.4349184288245195873e+00,
                   2.9267472035096449012e+00,   -1.1713569391233907169e+01,
                   34.0275626863976370373166,   -2.8480442582712722555e+01,
                   -8.8947340548151565542e+00,  5.2575353623762932642e+00,
                   -2.8312249982592527786e+00,  1.1331265795534763541e+01,
                   7.9462158845078978420e-01,   -6.6508361863670617531e-01,
                   -2.0771242914526857892e-01,  1.2277550230353953542e-01,
                   -6.6115593588420035198e-02,  2.6461103043402139923e-01]])
    # With centering constraint:
    dmatrix_R_cons = \
        np.array([[0.00329998606323867252343,   1.6537431155796576600e-04,
                   -1.2392262709790753433e-04,  6.5405304166706783407e-05,
                   -6.6764045799537624095e-05,  -0.1386431081763726258504,
                   0.124297283800864313830,     -3.5487293655619825405e-02,
                   -3.0527115315785902268e-03,  5.2009247643311604277e-04,
                   -0.00384203992301702674378,  -0.058901915802819435064,
                   0.266422358491648914036,     0.5739281693874087597607,
                   -1.3171008503525844392e-03,  8.2573456631878912413e-04,
                   6.6730833453016958831e-03,   -0.1467677784718444955470,
                   0.220757650934837484913,     0.1983127687880171796664,
                   -1.6269930328365173316e-03,  -1.7785892412241208812e-03,
                   -3.2702835436351201243e-03,  -4.3252183044300757109e-02,
                   4.3403766976235179376e-02,   3.5973406402893762387e-05,
                   -5.4035858568225075046e-04,  2.9565209382794241247e-04,
                   -2.2769990750264097637e-04],
                  [0.41547954838956052681098,   1.9843570584107707994e-02,
                   -1.5746590234791378593e-02,  8.3171184312221431434e-03,
                   -8.7233014052017516377e-03,  -15.9926770785086258541696,
                   16.503663226274017716833,    -6.6005803955894726265e-01,
                   1.3986092022708346283e-01,   -2.3516913533670955050e-01,
                   0.72251037497207359905360,   -9.827337059999853963177,
                   3.917078117294827688255,     9.0171773596973618936090,
                   -5.0616811270787671617e+00,  3.0189990249009683865e+00,
                   -1.0872720629943064097e+01,  26.9308504460453121964747,
                   -21.212262927009287949431,   -9.1088328555582247503253,
                   5.2400156972500298025e+00,   -3.0593641098325474736e+00,
                   1.0919392118399086300e+01,   -4.6564290223265718538e+00,
                   4.8071307441606982991e+00,   -1.9748377005689798924e-01,
                   5.4664183716965096538e-02,   -2.8871392916916285148e-02,
                   2.3592766838010845176e-01]])
    new_x1 = np.array([11.390625, 656.84083557128906250])
    new_x2 = np.array([16.777216000000006346, 1844.6744073709567147])
    new_data = {"x1": new_x1, "x2": new_x2}
    data_chunked = [{"x1": x1[:10], "x2": x2[:10]},
                    {"x1": x1[10:], "x2": x2[10:]}]

    builder = incr_dbuilder("te(cr(x1, df=5), cc(x2, df=6)) - 1",
                            lambda: iter(data_chunked))
    dmatrix_nocons = build_design_matrices([builder], new_data)[0]
    assert np.allclose(dmatrix_nocons, dmatrix_R_nocons, rtol=1e-12, atol=0.)

    builder = incr_dbuilder("te(cr(x1, df=5), cc(x2, df=6), "
                            "constraints='center') - 1",
                            lambda: iter(data_chunked))
    dmatrix_cons = build_design_matrices([builder], new_data)[0]
    assert np.allclose(dmatrix_cons, dmatrix_R_cons, rtol=1e-12, atol=0.)


def test_te_3smooths():
    from patsy.highlevel import incr_dbuilder, build_design_matrices
    x1 = (-1.5)**np.arange(20)
    x2 = (1.6)**np.arange(20)
    x3 = (-1.2)**np.arange(20)
    # Hard coded R results for smooth:  te(x1, x2, x3, bs=c("cr", "cs", "cc"), k=c(3,3,4))
    design_matrix_R = \
        np.array([[7.2077663709837084334e-05,   2.0648333344343273131e-03,
                   -4.7934014082310591768e-04,  2.3923430783992746568e-04,
                   6.8534265421922660466e-03,   -1.5909867344112936776e-03,
                   -6.8057712777151204314e-09,  -1.9496724335203412851e-07,
                   4.5260614658693259131e-08,   0.0101479754187435277507,
                   0.290712501531622591333,     -0.067487370093906928759,
                   0.03368233306025386619709,   0.9649092451763204847381,
                   -0.2239985793289433757547,   -9.5819975394704535133e-07,
                   -2.7449874082511405643e-05,  6.3723431275833230217e-06,
                   -1.5205851762850489204e-04,  -0.00435607204539782688624,
                   0.00101123909269346416370,   -5.0470024059694933508e-04,
                   -1.4458319360584082416e-02,  3.3564223914790921634e-03,
                   1.4357783514933466209e-08,   4.1131230514870551983e-07,
                   -9.5483976834512651038e-08]])
    new_data = {"x1": -38.443359375000000000,
                "x2": 68.719476736000032702,
                "x3": -5.1597803519999985156}
    data_chunked = [{"x1": x1[:10], "x2": x2[:10], "x3": x3[:10]},
                    {"x1": x1[10:], "x2": x2[10:], "x3": x3[10:]}]
    builder = incr_dbuilder("te(cr(x1, df=3), cr(x2, df=3), cc(x3, df=3)) - 1",
                            lambda: iter(data_chunked))
    design_matrix = build_design_matrices([builder], new_data)[0]
    assert np.allclose(design_matrix, design_matrix_R, rtol=1e-12, atol=0.)
