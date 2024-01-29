import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import scipy.special as sc


class TestHyperu:

    def test_negative_x(self):
        a, b, x = np.meshgrid(
            [-1, -0.5, 0, 0.5, 1],
            [-1, -0.5, 0, 0.5, 1],
            np.linspace(-100, -1, 10),
        )
        assert np.all(np.isnan(sc.hyperu(a, b, x)))

    def test_special_cases(self):
        assert sc.hyperu(0, 1, 1) == 1.0

    @pytest.mark.parametrize('a', [0.5, 1, np.nan])
    @pytest.mark.parametrize('b', [1, 2, np.nan])
    @pytest.mark.parametrize('x', [0.25, 3, np.nan])
    def test_nan_inputs(self, a, b, x):
        assert np.isnan(sc.hyperu(a, b, x)) == np.any(np.isnan([a, b, x]))


class TestHyp1f1:

    @pytest.mark.parametrize('a, b, x', [
        (np.nan, 1, 1),
        (1, np.nan, 1),
        (1, 1, np.nan)
    ])
    def test_nan_inputs(self, a, b, x):
        assert np.isnan(sc.hyp1f1(a, b, x))

    def test_poles(self):
        assert_equal(sc.hyp1f1(1, [0, -1, -2, -3, -4], 0.5), np.inf)

    @pytest.mark.parametrize('a, b, x, result', [
        (-1, 1, 0.5, 0.5),
        (1, 1, 0.5, 1.6487212707001281468),
        (2, 1, 0.5, 2.4730819060501922203),
        (1, 2, 0.5, 1.2974425414002562937),
        (-10, 1, 0.5, -0.38937441413785204475)
    ])
    def test_special_cases(self, a, b, x, result):
        # Hit all the special case branches at the beginning of the
        # function. Desired answers computed using Mpmath.
        assert_allclose(sc.hyp1f1(a, b, x), result, atol=0, rtol=1e-15)

    @pytest.mark.parametrize('a, b, x, result', [
        (1, 1, 0.44, 1.5527072185113360455),
        (-1, 1, 0.44, 0.55999999999999999778),
        (100, 100, 0.89, 2.4351296512898745592),
        (-100, 100, 0.89, 0.40739062490768104667),
        (1.5, 100, 59.99, 3.8073513625965598107),
        (-1.5, 100, 59.99, 0.25099240047125826943)
    ])
    def test_geometric_convergence(self, a, b, x, result):
        # Test the region where we are relying on the ratio of
        #
        # (|a| + 1) * |x| / |b|
        #
        # being small. Desired answers computed using Mpmath
        assert_allclose(sc.hyp1f1(a, b, x), result, atol=0, rtol=1e-15)

    @pytest.mark.parametrize('a, b, x, result', [
        (-1, 1, 1.5, -0.5),
        (-10, 1, 1.5, 0.41801777430943080357),
        (-25, 1, 1.5, 0.25114491646037839809),
        (-50, 1, 1.5, -0.25683643975194756115),
        (-80, 1, 1.5, -0.24554329325751503601),
        (-150, 1, 1.5, -0.173364795515420454496),
    ])
    def test_a_negative_integer(self, a, b, x, result):
        # Desired answers computed using Mpmath.
        assert_allclose(sc.hyp1f1(a, b, x), result, atol=0, rtol=2e-14)

    @pytest.mark.parametrize('a, b, x, expected', [
        (0.01, 150, -4, 0.99973683897677527773),        # gh-3492
        (1, 5, 0.01, 1.0020033381011970966),            # gh-3593
        (50, 100, 0.01, 1.0050126452421463411),         # gh-3593
        (1, 0.3, -1e3, -7.011932249442947651455e-04),   # gh-14149
        (1, 0.3, -1e4, -7.001190321418937164734e-05),   # gh-14149
        (9, 8.5, -350, -5.224090831922378361082e-20),   # gh-17120
        (9, 8.5, -355, -4.595407159813368193322e-20),   # gh-17120
        (75, -123.5, 15, 3.425753920814889017493e+06),
    ])
    def test_assorted_cases(self, a, b, x, expected):
        # Expected values were computed with mpmath.hyp1f1(a, b, x).
        assert_allclose(sc.hyp1f1(a, b, x), expected, atol=0, rtol=1e-14)

    def test_a_neg_int_and_b_equal_x(self):
        # This is a case where the Boost wrapper will call hypergeometric_pFq
        # instead of hypergeometric_1F1.  When we use a version of Boost in
        # which https://github.com/boostorg/math/issues/833 is fixed, this
        # test case can probably be moved into test_assorted_cases.
        # The expected value was computed with mpmath.hyp1f1(a, b, x).
        a = -10.0
        b = 2.5
        x = 2.5
        expected = 0.0365323664364104338721
        computed = sc.hyp1f1(a, b, x)
        assert_allclose(computed, expected, atol=0, rtol=1e-13)

    @pytest.mark.parametrize('a, b, x, desired', [
        (-1, -2, 2, 2),
        (-1, -4, 10, 3.5),
        (-2, -2, 1, 2.5)
    ])
    def test_gh_11099(self, a, b, x, desired):
        # All desired results computed using Mpmath
        assert sc.hyp1f1(a, b, x) == desired

    @pytest.mark.parametrize('a', [-3, -2])
    def test_x_zero_a_and_b_neg_ints_and_a_ge_b(self, a):
        assert sc.hyp1f1(a, -3, 0) == 1

    # The "legacy edge cases" mentioned in the comments in the following
    # tests refers to the behavior of hyp1f1(a, b, x) when b is a nonpositive
    # integer.  In some subcases, the behavior of SciPy does not match that
    # of Boost (1.81+), mpmath and Mathematica (via Wolfram Alpha online).
    # If the handling of these edges cases is changed to agree with those
    # libraries, these test will have to be updated.

    @pytest.mark.parametrize('b', [0, -1, -5])
    def test_legacy_case1(self, b):
        # Test results of hyp1f1(0, n, x) for n <= 0.
        # This is a legacy edge case.
        # Boost (versions greater than 1.80), Mathematica (via Wolfram Alpha
        # online) and mpmath all return 1 in this case, but SciPy's hyp1f1
        # returns inf.
        assert_equal(sc.hyp1f1(0, b, [-1.5, 0, 1.5]), [np.inf, np.inf, np.inf])

    def test_legacy_case2(self):
        # This is a legacy edge case.
        # In software such as boost (1.81+), mpmath and Mathematica,
        # the value is 1.
        assert sc.hyp1f1(-4, -3, 0) == np.inf
