import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy.special import powm1


# Expected values were computed with mpmath, e.g.
#
#   >>> import mpmath
#   >>> mpmath.np.dps = 200
#   >>> print(float(mpmath.powm1(2.0, 1e-7))
#   6.931472045825965e-08
#
powm1_test_cases = [
    (1.25, 0.75, 0.18217701125396976, 1e-15),
    (2.0, 1e-7, 6.931472045825965e-08, 1e-15),
    (25.0, 5e-11, 1.6094379125636148e-10, 1e-15),
    (0.99996, 0.75, -3.0000150002530058e-05, 1e-15),
    (0.9999999999990905, 20, -1.81898940353014e-11, 1e-15),
    (-1.25, 751.0, -6.017550852453444e+72, 2e-15)
]


@pytest.mark.parametrize('x, y, expected, rtol', powm1_test_cases)
def test_powm1(x, y, expected, rtol):
    p = powm1(x, y)
    assert_allclose(p, expected, rtol=rtol)


@pytest.mark.parametrize('x, y, expected',
                         [(0.0, 0.0, 0.0),
                          (0.0, -1.5, np.inf),
                          (0.0, 1.75, -1.0),
                          (-1.5, 2.0, 1.25),
                          (-1.5, 3.0, -4.375),
                          (np.nan, 0.0, 0.0),
                          (1.0, np.nan, 0.0),
                          (1.0, np.inf, 0.0),
                          (1.0, -np.inf, 0.0),
                          (np.inf, 7.5, np.inf),
                          (np.inf, -7.5, -1.0),
                          (3.25, np.inf, np.inf),
                          (np.inf, np.inf, np.inf),
                          (np.inf, -np.inf, -1.0),
                          (np.inf, 0.0, 0.0),
                          (-np.inf, 0.0, 0.0),
                          (-np.inf, 2.0, np.inf),
                          (-np.inf, 3.0, -np.inf),
                          (-1.0, float(2**53 - 1), -2.0)])
def test_powm1_exact_cases(x, y, expected):
    # Test cases where we have an exact expected value.
    p = powm1(x, y)
    assert p == expected


@pytest.mark.parametrize('x, y',
                         [(-1.25, 751.03),
                          (-1.25, np.inf),
                          (np.nan, np.nan),
                          (-np.inf, -np.inf),
                          (-np.inf, 2.5)])
def test_powm1_return_nan(x, y):
    # Test cases where the expected return value is nan.
    p = powm1(x, y)
    assert np.isnan(p)
