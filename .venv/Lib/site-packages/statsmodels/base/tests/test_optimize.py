from statsmodels.compat.scipy import SP_LT_15, SP_LT_17
import pytest
from numpy.testing import assert_
from numpy.testing import assert_almost_equal

from statsmodels.base.optimizer import (
    _fit_newton,
    _fit_nm,
    _fit_bfgs,
    _fit_cg,
    _fit_ncg,
    _fit_powell,
    _fit_lbfgs,
    _fit_basinhopping,
    _fit_minimize,
)

fit_funcs = {
    "newton": _fit_newton,
    "nm": _fit_nm,  # Nelder-Mead
    "bfgs": _fit_bfgs,
    "cg": _fit_cg,
    "ncg": _fit_ncg,
    "powell": _fit_powell,
    "lbfgs": _fit_lbfgs,
    "basinhopping": _fit_basinhopping,
    "minimize": _fit_minimize,
}


def dummy_func(x):
    return x ** 2


def dummy_score(x):
    return 2.0 * x


def dummy_hess(x):
    return [[2.0]]


def dummy_bounds_constraint_func(x):
    return (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2


def dummy_bounds():
    return ((0, None), (0, None))


def dummy_bounds_tight():
    return ((2, None), (3.5, None))


def dummy_constraints():
    cons = (
        {"type": "ineq", "fun": lambda x: x[0] - 2 * x[1] + 2},
        {"type": "ineq", "fun": lambda x: -x[0] - 2 * x[1] + 6},
        {"type": "ineq", "fun": lambda x: -x[0] + 2 * x[1] + 2},
    )
    return cons


@pytest.mark.smoke
def test_full_output_false(reset_randomstate):
    # newton needs f, score, start, fargs, kwargs
    # bfgs needs f, score start, fargs, kwargs
    # nm needs ""
    # cg ""
    # ncg ""
    # powell ""
    for method in fit_funcs:
        func = fit_funcs[method]
        if method == "newton":
            xopt, retvals = func(
                dummy_func,
                dummy_score,
                [1.0],
                (),
                {},
                hess=dummy_hess,
                full_output=False,
                disp=0,
            )

        else:
            xopt, retvals = func(
                dummy_func,
                dummy_score,
                [1.0],
                (),
                {},
                full_output=False,
                disp=0
            )
        assert_(retvals is None)
        if method == "powell" and SP_LT_15:
            # Fixed in SP 1.5
            assert_(xopt.shape == () and xopt.size == 1)
        else:
            assert_(len(xopt) == 1)


def test_full_output(reset_randomstate):
    for method in fit_funcs:
        func = fit_funcs[method]
        if method == "newton":
            xopt, retvals = func(
                dummy_func,
                dummy_score,
                [1.0],
                (),
                {},
                hess=dummy_hess,
                full_output=True,
                disp=0,
            )

        else:
            xopt, retvals = func(
                dummy_func,
                dummy_score,
                [1.0],
                (),
                {},
                full_output=True,
                disp=0
            )

        assert_(retvals is not None)
        assert_("converged" in retvals)

        if method == "powell" and SP_LT_15:
            # Fixed in SP 1.5
            assert_(xopt.shape == () and xopt.size == 1)
        else:
            assert_(len(xopt) == 1)


def test_minimize_scipy_slsqp():
    func = fit_funcs["minimize"]
    xopt, _ = func(
        dummy_bounds_constraint_func,
        None,
        (2.0, 0.0),
        (),
        {
            "min_method": "SLSQP",
            "bounds": dummy_bounds(),
            "constraints": dummy_constraints(),
        },
        hess=None,
        full_output=False,
        disp=0,
    )
    assert_almost_equal(xopt, [1.4, 1.7], 4)


@pytest.mark.skipif(SP_LT_15, reason="Powell bounds support added in SP 1.5")
def test_minimize_scipy_powell():
    func = fit_funcs["minimize"]
    xopt, _ = func(
        dummy_bounds_constraint_func,
        None,
        (3, 4.5),
        (),
        {
            "min_method": "Powell",
            "bounds": dummy_bounds_tight(),
        },
        hess=None,
        full_output=False,
        disp=0,
    )
    assert_almost_equal(xopt, [2, 3.5], 4)


@pytest.mark.skipif(SP_LT_17, reason="NM bounds support added in SP 1.7")
def test_minimize_scipy_nm():
    func = fit_funcs["minimize"]
    xopt, _ = func(
        dummy_bounds_constraint_func,
        None,
        (3, 4.5),
        (),
        {
            "min_method": "Nelder-Mead",
            "bounds": dummy_bounds_tight(),
        },
        hess=None,
        full_output=False,
        disp=0,
    )
    assert_almost_equal(xopt, [2, 3.5], 4)
