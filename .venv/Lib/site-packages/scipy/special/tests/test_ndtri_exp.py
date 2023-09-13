import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from scipy.special import log_ndtr, ndtri_exp
from scipy.special._testutils import assert_func_equal


def log_ndtr_ndtri_exp(y):
    return log_ndtr(ndtri_exp(y))


@pytest.fixture(scope="class")
def uniform_random_points():
    random_state = np.random.RandomState(1234)
    points = random_state.random_sample(1000)
    return points


class TestNdtriExp:
    """Tests that ndtri_exp is sufficiently close to an inverse of log_ndtr.

    We have separate tests for the five intervals (-inf, -10),
    [-10, -2), [-2, -0.14542), [-0.14542, -1e-6), and [-1e-6, 0).
    ndtri_exp(y) is computed in three different ways depending on if y
    is in (-inf, -2), [-2, log(1 - exp(-2))], or [log(1 - exp(-2), 0).
    Each of these intervals is given its own test with two additional tests
    for handling very small values and values very close to zero.
    """

    @pytest.mark.parametrize(
        "test_input", [-1e1, -1e2, -1e10, -1e20, -np.finfo(float).max]
    )
    def test_very_small_arg(self, test_input, uniform_random_points):
        scale = test_input
        points = scale * (0.5 * uniform_random_points + 0.5)
        assert_func_equal(
            log_ndtr_ndtri_exp,
            lambda y: y, points,
            rtol=1e-14,
            nan_ok=True
        )

    @pytest.mark.parametrize(
        "interval,expected_rtol",
        [
            ((-10, -2), 1e-14),
            ((-2, -0.14542), 1e-12),
            ((-0.14542, -1e-6), 1e-10),
            ((-1e-6, 0), 1e-6),
        ],
    )
    def test_in_interval(self, interval, expected_rtol, uniform_random_points):
        left, right = interval
        points = (right - left) * uniform_random_points + left
        assert_func_equal(
            log_ndtr_ndtri_exp,
            lambda y: y, points,
            rtol=expected_rtol,
            nan_ok=True
        )

    def test_extreme(self):
        # bigneg is not quite the largest negative double precision value.
        # Here's why:
        # The round-trip calculation
        #    y = ndtri_exp(bigneg)
        #    bigneg2 = log_ndtr(y)
        # where bigneg is a very large negative value, would--with infinite
        # precision--result in bigneg2 == bigneg.  When bigneg is large enough,
        # y is effectively equal to -sqrt(2)*sqrt(-bigneg), and log_ndtr(y) is
        # effectively -(y/sqrt(2))**2.  If we use bigneg = np.finfo(float).min,
        # then by construction, the theoretical value is the most negative
        # finite value that can be represented with 64 bit float point.  This
        # means tiny changes in how the computation proceeds can result in the
        # return value being -inf.  (E.g. changing the constant representation
        # of 1/sqrt(2) from 0.7071067811865475--which is the value returned by
        # 1/np.sqrt(2)--to 0.7071067811865476--which is the most accurate 64
        # bit floating point representation of 1/sqrt(2)--results in the
        # round-trip that starts with np.finfo(float).min returning -inf.  So
        # we'll move the bigneg value a few ULPs towards 0 to avoid this
        # sensitivity.
        # Use the reduce method to apply nextafter four times.
        bigneg = np.nextafter.reduce([np.finfo(float).min, 0, 0, 0, 0])
        # tinyneg is approx. -2.225e-308.
        tinyneg = -np.finfo(float).tiny
        x = np.array([tinyneg, bigneg])
        result = log_ndtr_ndtri_exp(x)
        assert_allclose(result, x, rtol=1e-12)

    def test_asymptotes(self):
        assert_equal(ndtri_exp([-np.inf, 0.0]), [-np.inf, np.inf])

    def test_outside_domain(self):
        assert np.isnan(ndtri_exp(1.0))
