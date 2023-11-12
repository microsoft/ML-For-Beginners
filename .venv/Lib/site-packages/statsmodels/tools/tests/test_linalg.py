from statsmodels.tools import linalg
import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import toeplitz


def test_stationary_solve_1d():
    b = np.random.uniform(size=10)
    r = np.random.uniform(size=9)
    t = np.concatenate((np.r_[1], r))
    tmat = toeplitz(t)
    soln = np.linalg.solve(tmat, b)
    soln1 = linalg.stationary_solve(r, b)
    assert_allclose(soln, soln1, rtol=1e-5, atol=1e-5)


def test_stationary_solve_2d():
    b = np.random.uniform(size=(10, 2))
    r = np.random.uniform(size=9)
    t = np.concatenate((np.r_[1], r))
    tmat = toeplitz(t)
    soln = np.linalg.solve(tmat, b)
    soln1 = linalg.stationary_solve(r, b)
    assert_allclose(soln, soln1, rtol=1e-5, atol=1e-5)
