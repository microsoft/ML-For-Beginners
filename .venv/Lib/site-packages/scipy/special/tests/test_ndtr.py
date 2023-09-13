import numpy as np
from numpy.testing import assert_equal, assert_allclose
import scipy.special as sc


def test_ndtr():
    assert_equal(sc.ndtr(0), 0.5)
    assert_allclose(sc.ndtr(1), 0.8413447460685429)


class TestNdtri:

    def test_zero(self):
        assert sc.ndtri(0.5) == 0.0

    def test_asymptotes(self):
        assert_equal(sc.ndtri([0.0, 1.0]), [-np.inf, np.inf])

    def test_outside_of_domain(self):
        assert all(np.isnan(sc.ndtri([-1.5, 1.5])))


class TestLogNdtr:

    # The expected values in these tests were computed with mpmath:
    #
    #   def log_ndtr_mp(x):
    #       return mpmath.log(mpmath.ncdf(x))
    #

    def test_log_ndtr_moderate_le8(self):
        x = np.array([-0.75, -0.25, 0, 0.5, 1.5, 2.5, 3, 4, 5, 7, 8])
        expected = np.array([-1.4844482299196562,
                             -0.9130617648111351,
                             -0.6931471805599453,
                             -0.3689464152886564,
                             -0.06914345561223398,
                             -0.006229025485860002,
                             -0.0013508099647481938,
                             -3.167174337748927e-05,
                             -2.866516129637636e-07,
                             -1.279812543886654e-12,
                             -6.220960574271786e-16])
        y = sc.log_ndtr(x)
        assert_allclose(y, expected, rtol=1e-14)

    def test_log_ndtr_values_8_16(self):
        x = np.array([8.001, 8.06, 8.15, 8.5, 10, 12, 14, 16])
        expected = [-6.170639424817055e-16,
                    -3.814722443652823e-16,
                    -1.819621363526629e-16,
                    -9.479534822203318e-18,
                    -7.619853024160525e-24,
                    -1.776482112077679e-33,
                    -7.7935368191928e-45,
                    -6.388754400538087e-58]
        y = sc.log_ndtr(x)
        assert_allclose(y, expected, rtol=5e-14)

    def test_log_ndtr_values_16_31(self):
        x = np.array([16.15, 20.3, 21.4, 26.2, 30.9])
        expected = [-5.678084565148492e-59,
                    -6.429244467698346e-92,
                    -6.680402412553295e-102,
                    -1.328698078458869e-151,
                    -5.972288641838264e-210]
        y = sc.log_ndtr(x)
        assert_allclose(y, expected, rtol=2e-13)

    def test_log_ndtr_values_gt31(self):
        x = np.array([31.6, 32.8, 34.9, 37.1])
        expected = [-1.846036234858162e-219,
                    -2.9440539964066835e-236,
                    -3.71721649450857e-267,
                    -1.4047119663106221e-301]
        y = sc.log_ndtr(x)
        assert_allclose(y, expected, rtol=3e-13)
