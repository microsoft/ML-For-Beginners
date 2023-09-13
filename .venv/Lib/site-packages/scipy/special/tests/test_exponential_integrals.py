import pytest

import numpy as np
from numpy.testing import assert_allclose
import scipy.special as sc


class TestExp1:

    def test_branch_cut(self):
        assert np.isnan(sc.exp1(-1))
        assert sc.exp1(complex(-1, 0)).imag == (
            -sc.exp1(complex(-1, -0.0)).imag
        )

        assert_allclose(
            sc.exp1(complex(-1, 0)),
            sc.exp1(-1 + 1e-20j),
            atol=0,
            rtol=1e-15
        )
        assert_allclose(
            sc.exp1(complex(-1, -0.0)),
            sc.exp1(-1 - 1e-20j),
            atol=0,
            rtol=1e-15
        )

    def test_834(self):
        # Regression test for #834
        a = sc.exp1(-complex(19.9999990))
        b = sc.exp1(-complex(19.9999991))
        assert_allclose(a.imag, b.imag, atol=0, rtol=1e-15)


class TestScaledExp1:

    @pytest.mark.parametrize('x, expected', [(0, 0), (np.inf, 1)])
    def test_limits(self, x, expected):
        y = sc._ufuncs._scaled_exp1(x)
        assert y == expected

    # The expected values were computed with mpmath, e.g.:
    #
    #   from mpmath import mp
    #   mp.dps = 80
    #   x = 1e-25
    #   print(float(x*mp.exp(x)*np.expint(1, x)))
    #
    # prints 5.698741165994961e-24
    #
    # The method used to compute _scaled_exp1 changes at x=1
    # and x=1250, so values at those inputs, and values just
    # above and below them, are included in the test data.
    @pytest.mark.parametrize('x, expected',
                             [(1e-25, 5.698741165994961e-24),
                              (0.1, 0.20146425447084518),
                              (0.9995, 0.5962509885831002),
                              (1.0, 0.5963473623231941),
                              (1.0005, 0.5964436833238044),
                              (2.5, 0.7588145912149602),
                              (10.0, 0.9156333393978808),
                              (100.0, 0.9901942286733019),
                              (500.0, 0.9980079523802055),
                              (1000.0, 0.9990019940238807),
                              (1249.5, 0.9992009578306811),
                              (1250.0, 0.9992012769377913),
                              (1250.25, 0.9992014363957858),
                              (2000.0, 0.9995004992514963),
                              (1e4, 0.9999000199940024),
                              (1e10, 0.9999999999),
                              (1e15, 0.999999999999999),
                              ])
    def test_scaled_exp1(self, x, expected):
        y = sc._ufuncs._scaled_exp1(x)
        assert_allclose(y, expected, rtol=2e-15)


class TestExpi:

    @pytest.mark.parametrize('result', [
        sc.expi(complex(-1, 0)),
        sc.expi(complex(-1, -0.0)),
        sc.expi(-1)
    ])
    def test_branch_cut(self, result):
        desired = -0.21938393439552027368  # Computed using Mpmath
        assert_allclose(result, desired, atol=0, rtol=1e-14)

    def test_near_branch_cut(self):
        lim_from_above = sc.expi(-1 + 1e-20j)
        lim_from_below = sc.expi(-1 - 1e-20j)
        assert_allclose(
            lim_from_above.real,
            lim_from_below.real,
            atol=0,
            rtol=1e-15
        )
        assert_allclose(
            lim_from_above.imag,
            -lim_from_below.imag,
            atol=0,
            rtol=1e-15
        )

    def test_continuity_on_positive_real_axis(self):
        assert_allclose(
            sc.expi(complex(1, 0)),
            sc.expi(complex(1, -0.0)),
            atol=0,
            rtol=1e-15
        )


class TestExpn:

    def test_out_of_domain(self):
        assert all(np.isnan([sc.expn(-1, 1.0), sc.expn(1, -1.0)]))
