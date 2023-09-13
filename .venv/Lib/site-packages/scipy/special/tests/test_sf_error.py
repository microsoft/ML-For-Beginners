import sys
import warnings

from numpy.testing import assert_, assert_equal, IS_PYPY
import pytest
from pytest import raises as assert_raises

import scipy.special as sc
from scipy.special._ufuncs import _sf_error_test_function

_sf_error_code_map = {
    # skip 'ok'
    'singular': 1,
    'underflow': 2,
    'overflow': 3,
    'slow': 4,
    'loss': 5,
    'no_result': 6,
    'domain': 7,
    'arg': 8,
    'other': 9
}

_sf_error_actions = [
    'ignore',
    'warn',
    'raise'
]


def _check_action(fun, args, action):
    if action == 'warn':
        with pytest.warns(sc.SpecialFunctionWarning):
            fun(*args)
    elif action == 'raise':
        with assert_raises(sc.SpecialFunctionError):
            fun(*args)
    else:
        # action == 'ignore', make sure there are no warnings/exceptions
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fun(*args)


def test_geterr():
    err = sc.geterr()
    for key, value in err.items():
        assert_(key in _sf_error_code_map)
        assert_(value in _sf_error_actions)


def test_seterr():
    entry_err = sc.geterr()
    try:
        for category, error_code in _sf_error_code_map.items():
            for action in _sf_error_actions:
                geterr_olderr = sc.geterr()
                seterr_olderr = sc.seterr(**{category: action})
                assert_(geterr_olderr == seterr_olderr)
                newerr = sc.geterr()
                assert_(newerr[category] == action)
                geterr_olderr.pop(category)
                newerr.pop(category)
                assert_(geterr_olderr == newerr)
                _check_action(_sf_error_test_function, (error_code,), action)
    finally:
        sc.seterr(**entry_err)


@pytest.mark.skipif(IS_PYPY, reason="Test not meaningful on PyPy")
def test_sf_error_special_refcount():
    # Regression test for gh-16233.
    # Check that the reference count of scipy.special is not increased
    # when a SpecialFunctionError is raised.
    refcount_before = sys.getrefcount(sc)
    with sc.errstate(all='raise'):
        with pytest.raises(sc.SpecialFunctionError, match='domain error'):
            sc.ndtri(2.0)
    refcount_after = sys.getrefcount(sc)
    assert refcount_after == refcount_before


def test_errstate_pyx_basic():
    olderr = sc.geterr()
    with sc.errstate(singular='raise'):
        with assert_raises(sc.SpecialFunctionError):
            sc.loggamma(0)
    assert_equal(olderr, sc.geterr())


def test_errstate_c_basic():
    olderr = sc.geterr()
    with sc.errstate(domain='raise'):
        with assert_raises(sc.SpecialFunctionError):
            sc.spence(-1)
    assert_equal(olderr, sc.geterr())


def test_errstate_cpp_basic():
    olderr = sc.geterr()
    with sc.errstate(underflow='raise'):
        with assert_raises(sc.SpecialFunctionError):
            sc.wrightomega(-1000)
    assert_equal(olderr, sc.geterr())


def test_errstate():
    for category, error_code in _sf_error_code_map.items():
        for action in _sf_error_actions:
            olderr = sc.geterr()
            with sc.errstate(**{category: action}):
                _check_action(_sf_error_test_function, (error_code,), action)
            assert_equal(olderr, sc.geterr())


def test_errstate_all_but_one():
    olderr = sc.geterr()
    with sc.errstate(all='raise', singular='ignore'):
        sc.gammaln(0)
        with assert_raises(sc.SpecialFunctionError):
            sc.spence(-1.0)
    assert_equal(olderr, sc.geterr())
