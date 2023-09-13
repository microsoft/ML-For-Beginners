# gh-14777 regression tests
# Test stdtr and stdtrit with infinite df and large values of df

import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy.special import stdtr, stdtrit, ndtr, ndtri


def test_stdtr_vs_R_large_df():
    df = [1e10, 1e12, 1e120, np.inf]
    t = 1.
    res = stdtr(df, t)
    # R Code:
    #   options(digits=20)
    #   pt(1., c(1e10, 1e12, 1e120, Inf))
    res_R = [0.84134474605644460343,
             0.84134474606842180044,
             0.84134474606854281475,
             0.84134474606854292578]
    assert_allclose(res, res_R, rtol=2e-15)
    # last value should also agree with ndtr
    assert_equal(res[3], ndtr(1.))


def test_stdtrit_vs_R_large_df():
    df = [1e10, 1e12, 1e120, np.inf]
    p = 0.1
    res = stdtrit(df, p)
    # R Code:
    #   options(digits=20)
    #   qt(0.1, c(1e10, 1e12, 1e120, Inf))
    res_R = [-1.2815515656292593150,
             -1.2815515655454472466,
             -1.2815515655446008125,
             -1.2815515655446008125]
    assert_allclose(res, res_R, rtol=1e-15)
    # last value should also agree with ndtri
    assert_equal(res[3], ndtri(0.1))


def test_stdtr_stdtri_invalid():
    # a mix of large and inf df with t/p equal to nan
    df = [1e10, 1e12, 1e120, np.inf]
    x = np.nan
    res1 = stdtr(df, x)
    res2 = stdtrit(df, x)
    res_ex = 4*[np.nan]
    assert_equal(res1, res_ex)
    assert_equal(res2, res_ex)
