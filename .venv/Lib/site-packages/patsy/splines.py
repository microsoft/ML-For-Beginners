# This file is part of Patsy
# Copyright (C) 2012-2013 Nathaniel Smith <njs@pobox.com>
# See file LICENSE.txt for license information.

# R-compatible spline basis functions

# These are made available in the patsy.* namespace
__all__ = ["bs"]

import numpy as np

from patsy.util import have_pandas, no_pickling, assert_no_pickling
from patsy.state import stateful_transform

if have_pandas:
    import pandas

def _eval_bspline_basis(x, knots, degree):
    try:
        from scipy.interpolate import splev
    except ImportError: # pragma: no cover
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
    # to handle them differently. I don't know why yet. So until we understand
    # this and decide what to do with it, I'm going to play it safe and
    # disallow such points.
    if np.min(x) < np.min(knots) or np.max(x) > np.max(knots):
        raise NotImplementedError("some data points fall outside the "
                                  "outermost knots, and I'm not sure how "
                                  "to handle them. (Patches accepted!)")
    # Thanks to Charles Harris for explaining splev. It's not well
    # documented, but basically it computes an arbitrary b-spline basis
    # given knots and degree on some specified points (or derivatives
    # thereof, but we don't use that functionality), and then returns some
    # linear combination of these basis functions. To get out the basis
    # functions themselves, we use linear combinations like [1, 0, 0], [0,
    # 1, 0], [0, 0, 1].
    # NB: This probably makes it rather inefficient (though I haven't checked
    # to be sure -- maybe the fortran code actually skips computing the basis
    # function for coefficients that are zero).
    # Note: the order of a spline is the same as its degree + 1.
    # Note: there are (len(knots) - order) basis functions.
    n_bases = len(knots) - (degree + 1)
    basis = np.empty((x.shape[0], n_bases), dtype=float)
    for i in range(n_bases):
        coefs = np.zeros((n_bases,))
        coefs[i] = 1
        basis[:, i] = splev(x, (knots, coefs, degree))
    return basis

def _R_compat_quantile(x, probs):
    #return np.percentile(x, 100 * np.asarray(probs))
    probs = np.asarray(probs)
    quantiles = np.asarray([np.percentile(x, 100 * prob)
                            for prob in probs.ravel(order="C")])
    return quantiles.reshape(probs.shape, order="C")

def test__R_compat_quantile():
    def t(x, prob, expected):
        assert np.allclose(_R_compat_quantile(x, prob), expected)
    t([10, 20], 0.5, 15)
    t([10, 20], 0.3, 13)
    t([10, 20], [0.3, 0.7], [13, 17])
    t(list(range(10)), [0.3, 0.7], [2.7, 6.3])

class BS(object):
    """bs(x, df=None, knots=None, degree=3, include_intercept=False, lower_bound=None, upper_bound=None)

    Generates a B-spline basis for ``x``, allowing non-linear fits. The usual
    usage is something like::

      y ~ 1 + bs(x, 4)

    to fit ``y`` as a smooth function of ``x``, with 4 degrees of freedom
    given to the smooth.

    :arg df: The number of degrees of freedom to use for this spline. The
      return value will have this many columns. You must specify at least one
      of ``df`` and ``knots``.
    :arg knots: The interior knots to use for the spline. If unspecified, then
      equally spaced quantiles of the input data are used. You must specify at
      least one of ``df`` and ``knots``.
    :arg degree: The degree of the spline to use.
    :arg include_intercept: If ``True``, then the resulting
      spline basis will span the intercept term (i.e., the constant
      function). If ``False`` (the default) then this will not be the case,
      which is useful for avoiding overspecification in models that include
      multiple spline terms and/or an intercept term.
    :arg lower_bound: The lower exterior knot location.
    :arg upper_bound: The upper exterior knot location.

    A spline with ``degree=0`` is piecewise constant with breakpoints at each
    knot, and the default knot positions are quantiles of the input. So if you
    find yourself in the situation of wanting to quantize a continuous
    variable into ``num_bins`` equal-sized bins with a constant effect across
    each bin, you can use ``bs(x, num_bins - 1, degree=0)``. (The ``- 1`` is
    because one degree of freedom will be taken by the intercept;
    alternatively, you could leave the intercept term out of your model and
    use ``bs(x, num_bins, degree=0, include_intercept=True)``.

    A spline with ``degree=1`` is piecewise linear with breakpoints at each
    knot.

    The default is ``degree=3``, which gives a cubic b-spline.

    This is a stateful transform (for details see
    :ref:`stateful-transforms`). If ``knots``, ``lower_bound``, or
    ``upper_bound`` are not specified, they will be calculated from the data
    and then the chosen values will be remembered and re-used for prediction
    from the fitted model.

    Using this function requires scipy be installed.

    .. note:: This function is very similar to the R function of the same
      name. In cases where both return output at all (e.g., R's ``bs`` will
      raise an error if ``degree=0``, while patsy's will not), they should
      produce identical output given identical input and parameter settings.

    .. warning:: I'm not sure on what the proper handling of points outside
      the lower/upper bounds is, so for now attempting to evaluate a spline
      basis at such points produces an error. Patches gratefully accepted.

    .. versionadded:: 0.2.0
    """
    def __init__(self):
        self._tmp = {}
        self._degree = None
        self._all_knots = None

    def memorize_chunk(self, x, df=None, knots=None, degree=3,
                       include_intercept=False,
                       lower_bound=None, upper_bound=None):
        args = {"df": df,
                "knots": knots,
                "degree": degree,
                "include_intercept": include_intercept,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                }
        self._tmp["args"] = args
        # XX: check whether we need x values before saving them
        x = np.atleast_1d(x)
        if x.ndim == 2 and x.shape[1] == 1:
            x = x[:, 0]
        if x.ndim > 1:
            raise ValueError("input to 'bs' must be 1-d, "
                             "or a 2-d column vector")
        # There's no better way to compute exact quantiles than memorizing
        # all data.
        self._tmp.setdefault("xs", []).append(x)

    def memorize_finish(self):
        tmp = self._tmp
        args = tmp["args"]
        del self._tmp

        if args["degree"] < 0:
            raise ValueError("degree must be greater than 0 (not %r)"
                             % (args["degree"],))
        if int(args["degree"]) != args["degree"]:
            raise ValueError("degree must be an integer (not %r)"
                             % (self._degree,))

        # These are guaranteed to all be 1d vectors by the code above
        x = np.concatenate(tmp["xs"])
        if args["df"] is None and args["knots"] is None:
            raise ValueError("must specify either df or knots")
        order = args["degree"] + 1
        if args["df"] is not None:
            n_inner_knots = args["df"] - order
            if not args["include_intercept"]:
                n_inner_knots += 1
            if n_inner_knots < 0:
                raise ValueError("df=%r is too small for degree=%r and "
                                 "include_intercept=%r; must be >= %s"
                                 % (args["df"], args["degree"],
                                    args["include_intercept"],
                                    # We know that n_inner_knots is negative;
                                    # if df were that much larger, it would
                                    # have been zero, and things would work.
                                    args["df"] - n_inner_knots))
            if args["knots"] is not None:
                if len(args["knots"]) != n_inner_knots:
                    raise ValueError("df=%s with degree=%r implies %s knots, "
                                     "but %s knots were provided"
                                     % (args["df"], args["degree"],
                                        n_inner_knots, len(args["knots"])))
            else:
                # Need to compute inner knots
                knot_quantiles = np.linspace(0, 1, n_inner_knots + 2)[1:-1]
                inner_knots = _R_compat_quantile(x, knot_quantiles)
        if args["knots"] is not None:
            inner_knots = args["knots"]
        if args["lower_bound"] is not None:
            lower_bound = args["lower_bound"]
        else:
            lower_bound = np.min(x)
        if args["upper_bound"] is not None:
            upper_bound = args["upper_bound"]
        else:
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
        all_knots = np.concatenate(([lower_bound, upper_bound] * order,
                                    inner_knots))
        all_knots.sort()

        self._degree = args["degree"]
        self._all_knots = all_knots

    def transform(self, x, df=None, knots=None, degree=3,
                  include_intercept=False,
                  lower_bound=None, upper_bound=None):
        basis = _eval_bspline_basis(x, self._all_knots, self._degree)
        if not include_intercept:
            basis = basis[:, 1:]
        if have_pandas:
            if isinstance(x, (pandas.Series, pandas.DataFrame)):
                basis = pandas.DataFrame(basis)
                basis.index = x.index
        return basis

    __getstate__ = no_pickling

bs = stateful_transform(BS)

def test_bs_compat():
    from patsy.test_state import check_stateful
    from patsy.test_splines_bs_data import (R_bs_test_x,
                                            R_bs_test_data,
                                            R_bs_num_tests)
    lines = R_bs_test_data.split("\n")
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
        kwargs = {
            "degree": int(test_data["degree"]),
            # integer, or None
            "df": eval(test_data["df"]),
            # np.array() call, or None
            "knots": eval(test_data["knots"]),
            }
        if test_data["Boundary.knots"] != "None":
            lower, upper = eval(test_data["Boundary.knots"])
            kwargs["lower_bound"] = lower
            kwargs["upper_bound"] = upper
        kwargs["include_intercept"] = (test_data["intercept"] == "TRUE")
        # Special case: in R, setting intercept=TRUE increases the effective
        # dof by 1. Adjust our arguments to match.
        # if kwargs["df"] is not None and kwargs["include_intercept"]:
        #     kwargs["df"] += 1
        output = np.asarray(eval(test_data["output"]))
        if kwargs["df"] is not None:
            assert output.shape[1] == kwargs["df"]
        # Do the actual test
        check_stateful(BS, False, R_bs_test_x, output, **kwargs)
        tests_ran += 1
        # Set up for the next one
        start_idx = stop_idx + 1
    assert tests_ran == R_bs_num_tests

test_bs_compat.slow = 1

# This isn't checked by the above, because R doesn't have zero degree
# b-splines.
def test_bs_0degree():
    x = np.logspace(-1, 1, 10)
    result = bs(x, knots=[1, 4], degree=0, include_intercept=True)
    assert result.shape[1] == 3
    expected_0 = np.zeros(10)
    expected_0[x < 1] = 1
    assert np.array_equal(result[:, 0], expected_0)
    expected_1 = np.zeros(10)
    expected_1[(x >= 1) & (x < 4)] = 1
    assert np.array_equal(result[:, 1], expected_1)
    expected_2 = np.zeros(10)
    expected_2[x >= 4] = 1
    assert np.array_equal(result[:, 2], expected_2)
    # Check handling of points that exactly fall on knots. They arbitrarily
    # get included into the larger region, not the smaller. This is consistent
    # with Python's half-open interval convention -- each basis function is
    # constant on [knot[i], knot[i + 1]).
    assert np.array_equal(bs([0, 1, 2], degree=0, knots=[1],
                             include_intercept=True),
                          [[1, 0],
                           [0, 1],
                           [0, 1]])

    result_int = bs(x, knots=[1, 4], degree=0, include_intercept=True)
    result_no_int = bs(x, knots=[1, 4], degree=0, include_intercept=False)
    assert np.array_equal(result_int[:, 1:], result_no_int)

def test_bs_errors():
    import pytest
    x = np.linspace(-10, 10, 20)
    # error checks:
    # out of bounds
    pytest.raises(NotImplementedError, bs, x, 3, lower_bound=0)
    pytest.raises(NotImplementedError, bs, x, 3, upper_bound=0)
    # must specify df or knots
    pytest.raises(ValueError, bs, x)
    # df/knots match/mismatch (with and without intercept)
    #   match:
    bs(x, df=10, include_intercept=False, knots=[0] * 7)
    bs(x, df=10, include_intercept=True, knots=[0] * 6)
    bs(x, df=10, include_intercept=False, knots=[0] * 9, degree=1)
    bs(x, df=10, include_intercept=True, knots=[0] * 8, degree=1)
    #   too many knots:
    pytest.raises(ValueError,
                  bs, x, df=10, include_intercept=False, knots=[0] * 8)
    pytest.raises(ValueError,
                  bs, x, df=10, include_intercept=True, knots=[0] * 7)
    pytest.raises(ValueError,
                  bs, x, df=10, include_intercept=False, knots=[0] * 10,
                  degree=1)
    pytest.raises(ValueError,
                  bs, x, df=10, include_intercept=True, knots=[0] * 9,
                  degree=1)
    #   too few knots:
    pytest.raises(ValueError,
                  bs, x, df=10, include_intercept=False, knots=[0] * 6)
    pytest.raises(ValueError,
                  bs, x, df=10, include_intercept=True, knots=[0] * 5)
    pytest.raises(ValueError,
                  bs, x, df=10, include_intercept=False, knots=[0] * 8,
                  degree=1)
    pytest.raises(ValueError,
                  bs, x, df=10, include_intercept=True, knots=[0] * 7,
                  degree=1)
    # df too small
    pytest.raises(ValueError,
                  bs, x, df=1, degree=3)
    pytest.raises(ValueError,
                  bs, x, df=3, degree=5)
    # bad degree
    pytest.raises(ValueError,
                  bs, x, df=10, degree=-1)
    pytest.raises(ValueError,
                  bs, x, df=10, degree=1.5)
    # upper_bound < lower_bound
    pytest.raises(ValueError,
                  bs, x, 3, lower_bound=1, upper_bound=-1)
    # multidimensional input
    pytest.raises(ValueError,
                  bs, np.column_stack((x, x)), 3)
    # unsorted knots are okay, and get sorted
    assert np.array_equal(bs(x, knots=[1, 4]), bs(x, knots=[4, 1]))
    # 2d knots
    pytest.raises(ValueError,
                  bs, x, knots=[[0], [20]])
    # knots > upper_bound
    pytest.raises(ValueError,
                  bs, x, knots=[0, 20])
    pytest.raises(ValueError,
                  bs, x, knots=[0, 4], upper_bound=3)
    # knots < lower_bound
    pytest.raises(ValueError,
                  bs, x, knots=[-20, 0])
    pytest.raises(ValueError,
                  bs, x, knots=[-4, 0], lower_bound=-3)



# differences between bs and ns (since the R code is a pile of copy-paste):
# - degree is always 3
# - different number of interior knots given df (b/c fewer dof used at edges I
#   guess)
# - boundary knots always repeated exactly 4 times (same as bs with degree=3)
# - complications at the end to handle boundary conditions
# the 'rcs' function uses slightly different conventions -- in particular it
# picks boundary knots that are not quite at the edges of the data, which
# makes sense for a natural spline.
