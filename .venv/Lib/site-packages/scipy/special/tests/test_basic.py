# this program corresponds to special.py

### Means test is not done yet
# E   Means test is giving error (E)
# F   Means test is failing (F)
# EF  Means test is giving error and Failing
#!   Means test is segfaulting
# 8   Means test runs forever

###  test_besselpoly
###  test_mathieu_a
###  test_mathieu_even_coef
###  test_mathieu_odd_coef
###  test_modfresnelp
###  test_modfresnelm
#    test_pbdv_seq
###  test_pbvv_seq
###  test_sph_harm

import functools
import itertools
import operator
import platform
import sys

import numpy as np
from numpy import (array, isnan, r_, arange, finfo, pi, sin, cos, tan, exp,
        log, zeros, sqrt, asarray, inf, nan_to_num, real, arctan, double,
        array_equal)

import pytest
from pytest import raises as assert_raises
from numpy.testing import (assert_equal, assert_almost_equal,
        assert_array_equal, assert_array_almost_equal, assert_approx_equal,
        assert_, assert_allclose, assert_array_almost_equal_nulp,
        suppress_warnings)

from scipy import special
import scipy.special._ufuncs as cephes
from scipy.special import ellipe, ellipk, ellipkm1
from scipy.special import elliprc, elliprd, elliprf, elliprg, elliprj
from scipy.special import mathieu_odd_coef, mathieu_even_coef, stirling2
from scipy._lib.deprecation import _NoValue
from scipy._lib._util import np_long, np_ulong

from scipy.special._basic import _FACTORIALK_LIMITS_64BITS, \
    _FACTORIALK_LIMITS_32BITS
from scipy.special._testutils import with_special_errors, \
     assert_func_equal, FuncData

import math


class TestCephes:
    def test_airy(self):
        cephes.airy(0)

    def test_airye(self):
        cephes.airye(0)

    def test_binom(self):
        n = np.array([0.264, 4, 5.2, 17])
        k = np.array([2, 0.4, 7, 3.3])
        nk = np.array(np.broadcast_arrays(n[:,None], k[None,:])
                      ).reshape(2, -1).T
        rknown = np.array([[-0.097152, 0.9263051596159367, 0.01858423645695389,
            -0.007581020651518199],[6, 2.0214389119675666, 0, 2.9827344527963846],
            [10.92, 2.22993515861399, -0.00585728, 10.468891352063146],
            [136, 3.5252179590758828, 19448, 1024.5526916174495]])
        assert_func_equal(cephes.binom, rknown.ravel(), nk, rtol=1e-13)

        # Test branches in implementation
        np.random.seed(1234)
        n = np.r_[np.arange(-7, 30), 1000*np.random.rand(30) - 500]
        k = np.arange(0, 102)
        nk = np.array(np.broadcast_arrays(n[:,None], k[None,:])
                      ).reshape(2, -1).T

        assert_func_equal(cephes.binom,
                          cephes.binom(nk[:,0], nk[:,1] * (1 + 1e-15)),
                          nk,
                          atol=1e-10, rtol=1e-10)

    def test_binom_2(self):
        # Test branches in implementation
        np.random.seed(1234)
        n = np.r_[np.logspace(1, 300, 20)]
        k = np.arange(0, 102)
        nk = np.array(np.broadcast_arrays(n[:,None], k[None,:])
                      ).reshape(2, -1).T

        assert_func_equal(cephes.binom,
                          cephes.binom(nk[:,0], nk[:,1] * (1 + 1e-15)),
                          nk,
                          atol=1e-10, rtol=1e-10)

    def test_binom_exact(self):
        @np.vectorize
        def binom_int(n, k):
            n = int(n)
            k = int(k)
            num = 1
            den = 1
            for i in range(1, k+1):
                num *= i + n - k
                den *= i
            return float(num/den)

        np.random.seed(1234)
        n = np.arange(1, 15)
        k = np.arange(0, 15)
        nk = np.array(np.broadcast_arrays(n[:,None], k[None,:])
                      ).reshape(2, -1).T
        nk = nk[nk[:,0] >= nk[:,1]]
        assert_func_equal(cephes.binom,
                          binom_int(nk[:,0], nk[:,1]),
                          nk,
                          atol=0, rtol=0)

    def test_binom_nooverflow_8346(self):
        # Test (binom(n, k) doesn't overflow prematurely */
        dataset = [
            (1000, 500, 2.70288240945436551e+299),
            (1002, 501, 1.08007396880791225e+300),
            (1004, 502, 4.31599279169058121e+300),
            (1006, 503, 1.72468101616263781e+301),
            (1008, 504, 6.89188009236419153e+301),
            (1010, 505, 2.75402257948335448e+302),
            (1012, 506, 1.10052048531923757e+303),
            (1014, 507, 4.39774063758732849e+303),
            (1016, 508, 1.75736486108312519e+304),
            (1018, 509, 7.02255427788423734e+304),
            (1020, 510, 2.80626776829962255e+305),
            (1022, 511, 1.12140876377061240e+306),
            (1024, 512, 4.48125455209897109e+306),
            (1026, 513, 1.79075474304149900e+307),
            (1028, 514, 7.15605105487789676e+307)
        ]
        dataset = np.asarray(dataset)
        FuncData(cephes.binom, dataset, (0, 1), 2, rtol=1e-12).check()

    def test_bdtr(self):
        assert_equal(cephes.bdtr(1,1,0.5),1.0)

    def test_bdtri(self):
        assert_equal(cephes.bdtri(1,3,0.5),0.5)

    def test_bdtrc(self):
        assert_equal(cephes.bdtrc(1,3,0.5),0.5)

    def test_bdtrin(self):
        assert_equal(cephes.bdtrin(1,0,1),5.0)

    def test_bdtrik(self):
        cephes.bdtrik(1,3,0.5)

    def test_bei(self):
        assert_equal(cephes.bei(0),0.0)

    def test_beip(self):
        assert_equal(cephes.beip(0),0.0)

    def test_ber(self):
        assert_equal(cephes.ber(0),1.0)

    def test_berp(self):
        assert_equal(cephes.berp(0),0.0)

    def test_besselpoly(self):
        assert_equal(cephes.besselpoly(0,0,0),1.0)

    def test_btdtr(self):
        with pytest.deprecated_call(match='deprecated in SciPy 1.12.0'):
            y = special.btdtr(1, 1, 1)
        assert_equal(y, 1.0)

    def test_btdtri(self):
        with pytest.deprecated_call(match='deprecated in SciPy 1.12.0'):
            y = special.btdtri(1, 1, 1)
        assert_equal(y, 1.0)

    def test_btdtria(self):
        assert_equal(cephes.btdtria(1,1,1),5.0)

    def test_btdtrib(self):
        assert_equal(cephes.btdtrib(1,1,1),5.0)

    def test_cbrt(self):
        assert_approx_equal(cephes.cbrt(1),1.0)

    def test_chdtr(self):
        assert_equal(cephes.chdtr(1,0),0.0)

    def test_chdtrc(self):
        assert_equal(cephes.chdtrc(1,0),1.0)

    def test_chdtri(self):
        assert_equal(cephes.chdtri(1,1),0.0)

    def test_chdtriv(self):
        assert_equal(cephes.chdtriv(0,0),5.0)

    def test_chndtr(self):
        assert_equal(cephes.chndtr(0,1,0),0.0)

        # Each row holds (x, nu, lam, expected_value)
        # These values were computed using Wolfram Alpha with
        #     CDF[NoncentralChiSquareDistribution[nu, lam], x]
        values = np.array([
            [25.00, 20.0, 400, 4.1210655112396197139e-57],
            [25.00, 8.00, 250, 2.3988026526832425878e-29],
            [0.001, 8.00, 40., 5.3761806201366039084e-24],
            [0.010, 8.00, 40., 5.45396231055999457039e-20],
            [20.00, 2.00, 107, 1.39390743555819597802e-9],
            [22.50, 2.00, 107, 7.11803307138105870671e-9],
            [25.00, 2.00, 107, 3.11041244829864897313e-8],
            [3.000, 2.00, 1.0, 0.62064365321954362734],
            [350.0, 300., 10., 0.93880128006276407710],
            [100.0, 13.5, 10., 0.99999999650104210949],
            [700.0, 20.0, 400, 0.99999999925680650105],
            [150.0, 13.5, 10., 0.99999999999999983046],
            [160.0, 13.5, 10., 0.99999999999999999518],  # 1.0
        ])
        cdf = cephes.chndtr(values[:, 0], values[:, 1], values[:, 2])
        assert_allclose(cdf, values[:, 3], rtol=1e-12)

        assert_almost_equal(cephes.chndtr(np.inf, np.inf, 0), 2.0)
        assert_almost_equal(cephes.chndtr(2, 1, np.inf), 0.0)
        assert_(np.isnan(cephes.chndtr(np.nan, 1, 2)))
        assert_(np.isnan(cephes.chndtr(5, np.nan, 2)))
        assert_(np.isnan(cephes.chndtr(5, 1, np.nan)))

    def test_chndtridf(self):
        assert_equal(cephes.chndtridf(0,0,1),5.0)

    def test_chndtrinc(self):
        assert_equal(cephes.chndtrinc(0,1,0),5.0)

    def test_chndtrix(self):
        assert_equal(cephes.chndtrix(0,1,0),0.0)

    def test_cosdg(self):
        assert_equal(cephes.cosdg(0),1.0)

    def test_cosm1(self):
        assert_equal(cephes.cosm1(0),0.0)

    def test_cotdg(self):
        assert_almost_equal(cephes.cotdg(45),1.0)

    def test_dawsn(self):
        assert_equal(cephes.dawsn(0),0.0)
        assert_allclose(cephes.dawsn(1.23), 0.50053727749081767)

    def test_diric(self):
        # Test behavior near multiples of 2pi.  Regression test for issue
        # described in gh-4001.
        n_odd = [1, 5, 25]
        x = np.array(2*np.pi + 5e-5).astype(np.float32)
        assert_almost_equal(special.diric(x, n_odd), 1.0, decimal=7)
        x = np.array(2*np.pi + 1e-9).astype(np.float64)
        assert_almost_equal(special.diric(x, n_odd), 1.0, decimal=15)
        x = np.array(2*np.pi + 1e-15).astype(np.float64)
        assert_almost_equal(special.diric(x, n_odd), 1.0, decimal=15)
        if hasattr(np, 'float128'):
            # No float128 available in 32-bit numpy
            x = np.array(2*np.pi + 1e-12).astype(np.float128)
            assert_almost_equal(special.diric(x, n_odd), 1.0, decimal=19)

        n_even = [2, 4, 24]
        x = np.array(2*np.pi + 1e-9).astype(np.float64)
        assert_almost_equal(special.diric(x, n_even), -1.0, decimal=15)

        # Test at some values not near a multiple of pi
        x = np.arange(0.2*np.pi, 1.0*np.pi, 0.2*np.pi)
        octave_result = [0.872677996249965, 0.539344662916632,
                         0.127322003750035, -0.206011329583298]
        assert_almost_equal(special.diric(x, 3), octave_result, decimal=15)

    def test_diric_broadcasting(self):
        x = np.arange(5)
        n = np.array([1, 3, 7])
        assert_(special.diric(x[:, np.newaxis], n).shape == (x.size, n.size))

    def test_ellipe(self):
        assert_equal(cephes.ellipe(1),1.0)

    def test_ellipeinc(self):
        assert_equal(cephes.ellipeinc(0,1),0.0)

    def test_ellipj(self):
        cephes.ellipj(0,1)

    def test_ellipk(self):
        assert_allclose(ellipk(0), pi/2)

    def test_ellipkinc(self):
        assert_equal(cephes.ellipkinc(0,0),0.0)

    def test_erf(self):
        assert_equal(cephes.erf(0), 0.0)

    def test_erf_symmetry(self):
        x = 5.905732037710919
        assert_equal(cephes.erf(x) + cephes.erf(-x), 0.0)

    def test_erfc(self):
        assert_equal(cephes.erfc(0), 1.0)

    def test_exp10(self):
        assert_approx_equal(cephes.exp10(2),100.0)

    def test_exp2(self):
        assert_equal(cephes.exp2(2),4.0)

    def test_expm1(self):
        assert_equal(cephes.expm1(0),0.0)
        assert_equal(cephes.expm1(np.inf), np.inf)
        assert_equal(cephes.expm1(-np.inf), -1)
        assert_equal(cephes.expm1(np.nan), np.nan)

    def test_expm1_complex(self):
        expm1 = cephes.expm1
        assert_equal(expm1(0 + 0j), 0 + 0j)
        assert_equal(expm1(complex(np.inf, 0)), complex(np.inf, 0))
        assert_equal(expm1(complex(np.inf, 1)), complex(np.inf, np.inf))
        assert_equal(expm1(complex(np.inf, 2)), complex(-np.inf, np.inf))
        assert_equal(expm1(complex(np.inf, 4)), complex(-np.inf, -np.inf))
        assert_equal(expm1(complex(np.inf, 5)), complex(np.inf, -np.inf))
        assert_equal(expm1(complex(1, np.inf)), complex(np.nan, np.nan))
        assert_equal(expm1(complex(0, np.inf)), complex(np.nan, np.nan))
        assert_equal(expm1(complex(np.inf, np.inf)), complex(np.inf, np.nan))
        assert_equal(expm1(complex(-np.inf, np.inf)), complex(-1, 0))
        assert_equal(expm1(complex(-np.inf, np.nan)), complex(-1, 0))
        assert_equal(expm1(complex(np.inf, np.nan)), complex(np.inf, np.nan))
        assert_equal(expm1(complex(0, np.nan)), complex(np.nan, np.nan))
        assert_equal(expm1(complex(1, np.nan)), complex(np.nan, np.nan))
        assert_equal(expm1(complex(np.nan, 1)), complex(np.nan, np.nan))
        assert_equal(expm1(complex(np.nan, np.nan)), complex(np.nan, np.nan))

    @pytest.mark.xfail(reason='The real part of expm1(z) bad at these points')
    def test_expm1_complex_hard(self):
        # The real part of this function is difficult to evaluate when
        # z.real = -log(cos(z.imag)).
        y = np.array([0.1, 0.2, 0.3, 5, 11, 20])
        x = -np.log(np.cos(y))
        z = x + 1j*y

        # evaluate using mpmath.expm1 with dps=1000
        expected = np.array([-5.5507901846769623e-17+0.10033467208545054j,
                              2.4289354732893695e-18+0.20271003550867248j,
                              4.5235500262585768e-17+0.30933624960962319j,
                              7.8234305217489006e-17-3.3805150062465863j,
                             -1.3685191953697676e-16-225.95084645419513j,
                              8.7175620481291045e-17+2.2371609442247422j])
        found = cephes.expm1(z)
        # this passes.
        assert_array_almost_equal_nulp(found.imag, expected.imag, 3)
        # this fails.
        assert_array_almost_equal_nulp(found.real, expected.real, 20)

    def test_fdtr(self):
        assert_equal(cephes.fdtr(1, 1, 0), 0.0)
        # Computed using Wolfram Alpha: CDF[FRatioDistribution[1e-6, 5], 10]
        assert_allclose(cephes.fdtr(1e-6, 5, 10), 0.9999940790193488,
                        rtol=1e-12)

    def test_fdtrc(self):
        assert_equal(cephes.fdtrc(1, 1, 0), 1.0)
        # Computed using Wolfram Alpha:
        #   1 - CDF[FRatioDistribution[2, 1/10], 1e10]
        assert_allclose(cephes.fdtrc(2, 0.1, 1e10), 0.27223784621293512,
                        rtol=1e-12)

    def test_fdtri(self):
        assert_allclose(cephes.fdtri(1, 1, [0.499, 0.501]),
                        array([0.9937365, 1.00630298]), rtol=1e-6)
        # From Wolfram Alpha:
        #   CDF[FRatioDistribution[1/10, 1], 3] = 0.8756751669632105666874...
        p = 0.8756751669632105666874
        assert_allclose(cephes.fdtri(0.1, 1, p), 3, rtol=1e-12)

    @pytest.mark.xfail(reason='Returns nan on i686.')
    def test_fdtri_mysterious_failure(self):
        assert_allclose(cephes.fdtri(1, 1, 0.5), 1)

    def test_fdtridfd(self):
        assert_equal(cephes.fdtridfd(1,0,0),5.0)

    def test_fresnel(self):
        assert_equal(cephes.fresnel(0),(0.0,0.0))

    def test_gamma(self):
        assert_equal(cephes.gamma(5),24.0)

    def test_gammainccinv(self):
        assert_equal(cephes.gammainccinv(5,1),0.0)

    def test_gammaln(self):
        cephes.gammaln(10)

    def test_gammasgn(self):
        vals = np.array([-4, -3.5, -2.3, 1, 4.2], np.float64)
        assert_array_equal(cephes.gammasgn(vals), np.sign(cephes.rgamma(vals)))

    def test_gdtr(self):
        assert_equal(cephes.gdtr(1,1,0),0.0)

    def test_gdtr_inf(self):
        assert_equal(cephes.gdtr(1,1,np.inf),1.0)

    def test_gdtrc(self):
        assert_equal(cephes.gdtrc(1,1,0),1.0)

    def test_gdtria(self):
        assert_equal(cephes.gdtria(0,1,1),0.0)

    def test_gdtrib(self):
        cephes.gdtrib(1,0,1)
        # assert_equal(cephes.gdtrib(1,0,1),5.0)

    def test_gdtrix(self):
        cephes.gdtrix(1,1,.1)

    def test_hankel1(self):
        cephes.hankel1(1,1)

    def test_hankel1e(self):
        cephes.hankel1e(1,1)

    def test_hankel2(self):
        cephes.hankel2(1,1)

    def test_hankel2e(self):
        cephes.hankel2e(1,1)

    def test_hyp1f1(self):
        assert_approx_equal(cephes.hyp1f1(1,1,1), exp(1.0))
        assert_approx_equal(cephes.hyp1f1(3,4,-6), 0.026056422099537251095)
        cephes.hyp1f1(1,1,1)

    def test_hyp2f1(self):
        assert_equal(cephes.hyp2f1(1,1,1,0),1.0)

    def test_i0(self):
        assert_equal(cephes.i0(0),1.0)

    def test_i0e(self):
        assert_equal(cephes.i0e(0),1.0)

    def test_i1(self):
        assert_equal(cephes.i1(0),0.0)

    def test_i1e(self):
        assert_equal(cephes.i1e(0),0.0)

    def test_it2i0k0(self):
        cephes.it2i0k0(1)

    def test_it2j0y0(self):
        cephes.it2j0y0(1)

    def test_it2struve0(self):
        cephes.it2struve0(1)

    def test_itairy(self):
        cephes.itairy(1)

    def test_iti0k0(self):
        assert_equal(cephes.iti0k0(0),(0.0,0.0))

    def test_itj0y0(self):
        assert_equal(cephes.itj0y0(0),(0.0,0.0))

    def test_itmodstruve0(self):
        assert_equal(cephes.itmodstruve0(0),0.0)

    def test_itstruve0(self):
        assert_equal(cephes.itstruve0(0),0.0)

    def test_iv(self):
        assert_equal(cephes.iv(1,0),0.0)

    def test_ive(self):
        assert_equal(cephes.ive(1,0),0.0)

    def test_j0(self):
        assert_equal(cephes.j0(0),1.0)

    def test_j1(self):
        assert_equal(cephes.j1(0),0.0)

    def test_jn(self):
        assert_equal(cephes.jn(0,0),1.0)

    def test_jv(self):
        assert_equal(cephes.jv(0,0),1.0)

    def test_jve(self):
        assert_equal(cephes.jve(0,0),1.0)

    def test_k0(self):
        cephes.k0(2)

    def test_k0e(self):
        cephes.k0e(2)

    def test_k1(self):
        cephes.k1(2)

    def test_k1e(self):
        cephes.k1e(2)

    def test_kei(self):
        cephes.kei(2)

    def test_keip(self):
        assert_equal(cephes.keip(0),0.0)

    def test_ker(self):
        cephes.ker(2)

    def test_kerp(self):
        cephes.kerp(2)

    def test_kelvin(self):
        cephes.kelvin(2)

    def test_kn(self):
        cephes.kn(1,1)

    def test_kolmogi(self):
        assert_equal(cephes.kolmogi(1),0.0)
        assert_(np.isnan(cephes.kolmogi(np.nan)))

    def test_kolmogorov(self):
        assert_equal(cephes.kolmogorov(0), 1.0)

    def test_kolmogp(self):
        assert_equal(cephes._kolmogp(0), -0.0)

    def test_kolmogc(self):
        assert_equal(cephes._kolmogc(0), 0.0)

    def test_kolmogci(self):
        assert_equal(cephes._kolmogci(0), 0.0)
        assert_(np.isnan(cephes._kolmogci(np.nan)))

    def test_kv(self):
        cephes.kv(1,1)

    def test_kve(self):
        cephes.kve(1,1)

    def test_log1p(self):
        log1p = cephes.log1p
        assert_equal(log1p(0), 0.0)
        assert_equal(log1p(-1), -np.inf)
        assert_equal(log1p(-2), np.nan)
        assert_equal(log1p(np.inf), np.inf)

    def test_log1p_complex(self):
        log1p = cephes.log1p
        c = complex
        assert_equal(log1p(0 + 0j), 0 + 0j)
        assert_equal(log1p(c(-1, 0)), c(-np.inf, 0))
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "invalid value encountered in multiply")
            assert_allclose(log1p(c(1, np.inf)), c(np.inf, np.pi/2))
            assert_equal(log1p(c(1, np.nan)), c(np.nan, np.nan))
            assert_allclose(log1p(c(-np.inf, 1)), c(np.inf, np.pi))
            assert_equal(log1p(c(np.inf, 1)), c(np.inf, 0))
            assert_allclose(log1p(c(-np.inf, np.inf)), c(np.inf, 3*np.pi/4))
            assert_allclose(log1p(c(np.inf, np.inf)), c(np.inf, np.pi/4))
            assert_equal(log1p(c(np.inf, np.nan)), c(np.inf, np.nan))
            assert_equal(log1p(c(-np.inf, np.nan)), c(np.inf, np.nan))
            assert_equal(log1p(c(np.nan, np.inf)), c(np.inf, np.nan))
            assert_equal(log1p(c(np.nan, 1)), c(np.nan, np.nan))
            assert_equal(log1p(c(np.nan, np.nan)), c(np.nan, np.nan))

    def test_lpmv(self):
        assert_equal(cephes.lpmv(0,0,1),1.0)

    def test_mathieu_a(self):
        assert_equal(cephes.mathieu_a(1,0),1.0)

    def test_mathieu_b(self):
        assert_equal(cephes.mathieu_b(1,0),1.0)

    def test_mathieu_cem(self):
        assert_equal(cephes.mathieu_cem(1,0,0),(1.0,0.0))

        # Test AMS 20.2.27
        @np.vectorize
        def ce_smallq(m, q, z):
            z *= np.pi/180
            if m == 0:
                # + O(q^2)
                return 2**(-0.5) * (1 - .5*q*cos(2*z))
            elif m == 1:
                # + O(q^2)
                return cos(z) - q/8 * cos(3*z)
            elif m == 2:
                # + O(q^2)
                return cos(2*z) - q*(cos(4*z)/12 - 1/4)
            else:
                # + O(q^2)
                return cos(m*z) - q*(cos((m+2)*z)/(4*(m+1)) - cos((m-2)*z)/(4*(m-1)))
        m = np.arange(0, 100)
        q = np.r_[0, np.logspace(-30, -9, 10)]
        assert_allclose(cephes.mathieu_cem(m[:,None], q[None,:], 0.123)[0],
                        ce_smallq(m[:,None], q[None,:], 0.123),
                        rtol=1e-14, atol=0)

    def test_mathieu_sem(self):
        assert_equal(cephes.mathieu_sem(1,0,0),(0.0,1.0))

        # Test AMS 20.2.27
        @np.vectorize
        def se_smallq(m, q, z):
            z *= np.pi/180
            if m == 1:
                # + O(q^2)
                return sin(z) - q/8 * sin(3*z)
            elif m == 2:
                # + O(q^2)
                return sin(2*z) - q*sin(4*z)/12
            else:
                # + O(q^2)
                return sin(m*z) - q*(sin((m+2)*z)/(4*(m+1)) - sin((m-2)*z)/(4*(m-1)))
        m = np.arange(1, 100)
        q = np.r_[0, np.logspace(-30, -9, 10)]
        assert_allclose(cephes.mathieu_sem(m[:,None], q[None,:], 0.123)[0],
                        se_smallq(m[:,None], q[None,:], 0.123),
                        rtol=1e-14, atol=0)

    def test_mathieu_modcem1(self):
        assert_equal(cephes.mathieu_modcem1(1,0,0),(0.0,0.0))

    def test_mathieu_modcem2(self):
        cephes.mathieu_modcem2(1,1,1)

        # Test reflection relation AMS 20.6.19
        m = np.arange(0, 4)[:,None,None]
        q = np.r_[np.logspace(-2, 2, 10)][None,:,None]
        z = np.linspace(0, 1, 7)[None,None,:]

        y1 = cephes.mathieu_modcem2(m, q, -z)[0]

        fr = -cephes.mathieu_modcem2(m, q, 0)[0] / cephes.mathieu_modcem1(m, q, 0)[0]
        y2 = (-cephes.mathieu_modcem2(m, q, z)[0] 
              - 2*fr*cephes.mathieu_modcem1(m, q, z)[0])

        assert_allclose(y1, y2, rtol=1e-10)

    def test_mathieu_modsem1(self):
        assert_equal(cephes.mathieu_modsem1(1,0,0),(0.0,0.0))

    def test_mathieu_modsem2(self):
        cephes.mathieu_modsem2(1,1,1)

        # Test reflection relation AMS 20.6.20
        m = np.arange(1, 4)[:,None,None]
        q = np.r_[np.logspace(-2, 2, 10)][None,:,None]
        z = np.linspace(0, 1, 7)[None,None,:]

        y1 = cephes.mathieu_modsem2(m, q, -z)[0]
        fr = cephes.mathieu_modsem2(m, q, 0)[1] / cephes.mathieu_modsem1(m, q, 0)[1]
        y2 = (cephes.mathieu_modsem2(m, q, z)[0]
              - 2*fr*cephes.mathieu_modsem1(m, q, z)[0])
        assert_allclose(y1, y2, rtol=1e-10)

    def test_mathieu_overflow(self):
        # Check that these return NaNs instead of causing a SEGV
        assert_equal(cephes.mathieu_cem(10000, 0, 1.3), (np.nan, np.nan))
        assert_equal(cephes.mathieu_sem(10000, 0, 1.3), (np.nan, np.nan))
        assert_equal(cephes.mathieu_cem(10000, 1.5, 1.3), (np.nan, np.nan))
        assert_equal(cephes.mathieu_sem(10000, 1.5, 1.3), (np.nan, np.nan))
        assert_equal(cephes.mathieu_modcem1(10000, 1.5, 1.3), (np.nan, np.nan))
        assert_equal(cephes.mathieu_modsem1(10000, 1.5, 1.3), (np.nan, np.nan))
        assert_equal(cephes.mathieu_modcem2(10000, 1.5, 1.3), (np.nan, np.nan))
        assert_equal(cephes.mathieu_modsem2(10000, 1.5, 1.3), (np.nan, np.nan))

    def test_mathieu_ticket_1847(self):
        # Regression test --- this call had some out-of-bounds access
        # and could return nan occasionally
        for k in range(60):
            v = cephes.mathieu_modsem2(2, 100, -1)
            # Values from ACM TOMS 804 (derivate by numerical differentiation)
            assert_allclose(v[0], 0.1431742913063671074347, rtol=1e-10)
            assert_allclose(v[1], 0.9017807375832909144719, rtol=1e-4)

    def test_modfresnelm(self):
        cephes.modfresnelm(0)

    def test_modfresnelp(self):
        cephes.modfresnelp(0)

    def test_modstruve(self):
        assert_equal(cephes.modstruve(1,0),0.0)

    def test_nbdtr(self):
        assert_equal(cephes.nbdtr(1,1,1),1.0)

    def test_nbdtrc(self):
        assert_equal(cephes.nbdtrc(1,1,1),0.0)

    def test_nbdtri(self):
        assert_equal(cephes.nbdtri(1,1,1),1.0)

    def test_nbdtrik(self):
        cephes.nbdtrik(1,.4,.5)

    def test_nbdtrin(self):
        assert_equal(cephes.nbdtrin(1,0,0),5.0)

    def test_ncfdtr(self):
        assert_equal(cephes.ncfdtr(1,1,1,0),0.0)

    def test_ncfdtri(self):
        assert_equal(cephes.ncfdtri(1, 1, 1, 0), 0.0)
        f = [0.5, 1, 1.5]
        p = cephes.ncfdtr(2, 3, 1.5, f)
        assert_allclose(cephes.ncfdtri(2, 3, 1.5, p), f)

    def test_ncfdtridfd(self):
        dfd = [1, 2, 3]
        p = cephes.ncfdtr(2, dfd, 0.25, 15)
        assert_allclose(cephes.ncfdtridfd(2, p, 0.25, 15), dfd)

    def test_ncfdtridfn(self):
        dfn = [0.1, 1, 2, 3, 1e4]
        p = cephes.ncfdtr(dfn, 2, 0.25, 15)
        assert_allclose(cephes.ncfdtridfn(p, 2, 0.25, 15), dfn, rtol=1e-5)

    def test_ncfdtrinc(self):
        nc = [0.5, 1.5, 2.0]
        p = cephes.ncfdtr(2, 3, nc, 15)
        assert_allclose(cephes.ncfdtrinc(2, 3, p, 15), nc)

    def test_nctdtr(self):
        assert_equal(cephes.nctdtr(1,0,0),0.5)
        assert_equal(cephes.nctdtr(9, 65536, 45), 0.0)

        assert_approx_equal(cephes.nctdtr(np.inf, 1., 1.), 0.5, 5)
        assert_(np.isnan(cephes.nctdtr(2., np.inf, 10.)))
        assert_approx_equal(cephes.nctdtr(2., 1., np.inf), 1.)

        assert_(np.isnan(cephes.nctdtr(np.nan, 1., 1.)))
        assert_(np.isnan(cephes.nctdtr(2., np.nan, 1.)))
        assert_(np.isnan(cephes.nctdtr(2., 1., np.nan)))

    def test_nctdtridf(self):
        cephes.nctdtridf(1,0.5,0)

    def test_nctdtrinc(self):
        cephes.nctdtrinc(1,0,0)

    def test_nctdtrit(self):
        cephes.nctdtrit(.1,0.2,.5)

    def test_nrdtrimn(self):
        assert_approx_equal(cephes.nrdtrimn(0.5,1,1),1.0)

    def test_nrdtrisd(self):
        assert_allclose(cephes.nrdtrisd(0.5,0.5,0.5), 0.0,
                         atol=0, rtol=0)

    def test_obl_ang1(self):
        cephes.obl_ang1(1,1,1,0)

    def test_obl_ang1_cv(self):
        result = cephes.obl_ang1_cv(1,1,1,1,0)
        assert_almost_equal(result[0],1.0)
        assert_almost_equal(result[1],0.0)

    def test_obl_cv(self):
        assert_equal(cephes.obl_cv(1,1,0),2.0)

    def test_obl_rad1(self):
        cephes.obl_rad1(1,1,1,0)

    def test_obl_rad1_cv(self):
        cephes.obl_rad1_cv(1,1,1,1,0)

    def test_obl_rad2(self):
        cephes.obl_rad2(1,1,1,0)

    def test_obl_rad2_cv(self):
        cephes.obl_rad2_cv(1,1,1,1,0)

    def test_pbdv(self):
        assert_equal(cephes.pbdv(1,0),(0.0,1.0))

    def test_pbvv(self):
        cephes.pbvv(1,0)

    def test_pbwa(self):
        cephes.pbwa(1,0)

    def test_pdtr(self):
        val = cephes.pdtr(0, 1)
        assert_almost_equal(val, np.exp(-1))
        # Edge case: m = 0.
        val = cephes.pdtr([0, 1, 2], 0)
        assert_array_equal(val, [1, 1, 1])

    def test_pdtrc(self):
        val = cephes.pdtrc(0, 1)
        assert_almost_equal(val, 1 - np.exp(-1))
        # Edge case: m = 0.
        val = cephes.pdtrc([0, 1, 2], 0.0)
        assert_array_equal(val, [0, 0, 0])

    def test_pdtri(self):
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "floating point number truncated to an integer")
            cephes.pdtri(0.5,0.5)

    def test_pdtrik(self):
        k = cephes.pdtrik(0.5, 1)
        assert_almost_equal(cephes.gammaincc(k + 1, 1), 0.5)
        # Edge case: m = 0 or very small.
        k = cephes.pdtrik([[0], [0.25], [0.95]], [0, 1e-20, 1e-6])
        assert_array_equal(k, np.zeros((3, 3)))

    def test_pro_ang1(self):
        cephes.pro_ang1(1,1,1,0)

    def test_pro_ang1_cv(self):
        assert_array_almost_equal(cephes.pro_ang1_cv(1,1,1,1,0),
                                  array((1.0,0.0)))

    def test_pro_cv(self):
        assert_equal(cephes.pro_cv(1,1,0),2.0)

    def test_pro_rad1(self):
        cephes.pro_rad1(1,1,1,0.1)

    def test_pro_rad1_cv(self):
        cephes.pro_rad1_cv(1,1,1,1,0)

    def test_pro_rad2(self):
        cephes.pro_rad2(1,1,1,0)

    def test_pro_rad2_cv(self):
        cephes.pro_rad2_cv(1,1,1,1,0)

    def test_psi(self):
        cephes.psi(1)

    def test_radian(self):
        assert_equal(cephes.radian(0,0,0),0)

    def test_rgamma(self):
        assert_equal(cephes.rgamma(1),1.0)

    def test_round(self):
        assert_equal(cephes.round(3.4),3.0)
        assert_equal(cephes.round(-3.4),-3.0)
        assert_equal(cephes.round(3.6),4.0)
        assert_equal(cephes.round(-3.6),-4.0)
        assert_equal(cephes.round(3.5),4.0)
        assert_equal(cephes.round(-3.5),-4.0)

    def test_shichi(self):
        cephes.shichi(1)

    def test_sici(self):
        cephes.sici(1)

        s, c = cephes.sici(np.inf)
        assert_almost_equal(s, np.pi * 0.5)
        assert_almost_equal(c, 0)

        s, c = cephes.sici(-np.inf)
        assert_almost_equal(s, -np.pi * 0.5)
        assert_(np.isnan(c), "cosine integral(-inf) is not nan")

    def test_sindg(self):
        assert_equal(cephes.sindg(90),1.0)

    def test_smirnov(self):
        assert_equal(cephes.smirnov(1,.1),0.9)
        assert_(np.isnan(cephes.smirnov(1,np.nan)))

    def test_smirnovp(self):
        assert_equal(cephes._smirnovp(1, .1), -1)
        assert_equal(cephes._smirnovp(2, 0.75), -2*(0.25)**(2-1))
        assert_equal(cephes._smirnovp(3, 0.75), -3*(0.25)**(3-1))
        assert_(np.isnan(cephes._smirnovp(1, np.nan)))

    def test_smirnovc(self):
        assert_equal(cephes._smirnovc(1,.1),0.1)
        assert_(np.isnan(cephes._smirnovc(1,np.nan)))
        x10 = np.linspace(0, 1, 11, endpoint=True)
        assert_almost_equal(cephes._smirnovc(3, x10), 1-cephes.smirnov(3, x10))
        x4 = np.linspace(0, 1, 5, endpoint=True)
        assert_almost_equal(cephes._smirnovc(4, x4), 1-cephes.smirnov(4, x4))

    def test_smirnovi(self):
        assert_almost_equal(cephes.smirnov(1,cephes.smirnovi(1,0.4)),0.4)
        assert_almost_equal(cephes.smirnov(1,cephes.smirnovi(1,0.6)),0.6)
        assert_(np.isnan(cephes.smirnovi(1,np.nan)))

    def test_smirnovci(self):
        assert_almost_equal(cephes._smirnovc(1,cephes._smirnovci(1,0.4)),0.4)
        assert_almost_equal(cephes._smirnovc(1,cephes._smirnovci(1,0.6)),0.6)
        assert_(np.isnan(cephes._smirnovci(1,np.nan)))

    def test_spence(self):
        assert_equal(cephes.spence(1),0.0)

    def test_stdtr(self):
        assert_equal(cephes.stdtr(1,0),0.5)
        assert_almost_equal(cephes.stdtr(1,1), 0.75)
        assert_almost_equal(cephes.stdtr(1,2), 0.852416382349)

    def test_stdtridf(self):
        cephes.stdtridf(0.7,1)

    def test_stdtrit(self):
        cephes.stdtrit(1,0.7)

    def test_struve(self):
        assert_equal(cephes.struve(0,0),0.0)

    def test_tandg(self):
        assert_equal(cephes.tandg(45),1.0)

    def test_tklmbda(self):
        assert_almost_equal(cephes.tklmbda(1,1),1.0)

    def test_y0(self):
        cephes.y0(1)

    def test_y1(self):
        cephes.y1(1)

    def test_yn(self):
        cephes.yn(1,1)

    def test_yv(self):
        cephes.yv(1,1)

    def test_yve(self):
        cephes.yve(1,1)

    def test_wofz(self):
        z = [complex(624.2,-0.26123), complex(-0.4,3.), complex(0.6,2.),
             complex(-1.,1.), complex(-1.,-9.), complex(-1.,9.),
             complex(-0.0000000234545,1.1234), complex(-3.,5.1),
             complex(-53,30.1), complex(0.0,0.12345),
             complex(11,1), complex(-22,-2), complex(9,-28),
             complex(21,-33), complex(1e5,1e5), complex(1e14,1e14)
             ]
        w = [
            complex(-3.78270245518980507452677445620103199303131110e-7,
                    0.000903861276433172057331093754199933411710053155),
            complex(0.1764906227004816847297495349730234591778719532788,
                    -0.02146550539468457616788719893991501311573031095617),
            complex(0.2410250715772692146133539023007113781272362309451,
                    0.06087579663428089745895459735240964093522265589350),
            complex(0.30474420525691259245713884106959496013413834051768,
                    -0.20821893820283162728743734725471561394145872072738),
            complex(7.317131068972378096865595229600561710140617977e34,
                    8.321873499714402777186848353320412813066170427e34),
            complex(0.0615698507236323685519612934241429530190806818395,
                    -0.00676005783716575013073036218018565206070072304635),
            complex(0.3960793007699874918961319170187598400134746631,
                    -5.593152259116644920546186222529802777409274656e-9),
            complex(0.08217199226739447943295069917990417630675021771804,
                    -0.04701291087643609891018366143118110965272615832184),
            complex(0.00457246000350281640952328010227885008541748668738,
                    -0.00804900791411691821818731763401840373998654987934),
            complex(0.8746342859608052666092782112565360755791467973338452,
                    0.),
            complex(0.00468190164965444174367477874864366058339647648741,
                    0.0510735563901306197993676329845149741675029197050),
            complex(-0.0023193175200187620902125853834909543869428763219,
                    -0.025460054739731556004902057663500272721780776336),
            complex(9.11463368405637174660562096516414499772662584e304,
                    3.97101807145263333769664875189354358563218932e305),
            complex(-4.4927207857715598976165541011143706155432296e281,
                    -2.8019591213423077494444700357168707775769028e281),
            complex(2.820947917809305132678577516325951485807107151e-6,
                    2.820947917668257736791638444590253942253354058e-6),
            complex(2.82094791773878143474039725787438662716372268e-15,
                    2.82094791773878143474039725773333923127678361e-15)
        ]
        assert_func_equal(cephes.wofz, w, z, rtol=1e-13)


class TestAiry:
    def test_airy(self):
        # This tests the airy function to ensure 8 place accuracy in computation

        x = special.airy(.99)
        assert_array_almost_equal(
            x,
            array([0.13689066,-0.16050153,1.19815925,0.92046818]),
            8,
        )
        x = special.airy(.41)
        assert_array_almost_equal(
            x,
            array([0.25238916,-.23480512,0.80686202,0.51053919]),
            8,
        )
        x = special.airy(-.36)
        assert_array_almost_equal(
            x,
            array([0.44508477,-0.23186773,0.44939534,0.48105354]),
            8,
        )

    def test_airye(self):
        a = special.airye(0.01)
        b = special.airy(0.01)
        b1 = [None]*4
        for n in range(2):
            b1[n] = b[n]*exp(2.0/3.0*0.01*sqrt(0.01))
        for n in range(2,4):
            b1[n] = b[n]*exp(-abs(real(2.0/3.0*0.01*sqrt(0.01))))
        assert_array_almost_equal(a,b1,6)

    def test_bi_zeros(self):
        bi = special.bi_zeros(2)
        bia = (array([-1.17371322, -3.2710930]),
               array([-2.29443968, -4.07315509]),
               array([-0.45494438, 0.39652284]),
               array([0.60195789, -0.76031014]))
        assert_array_almost_equal(bi,bia,4)

        bi = special.bi_zeros(5)
        assert_array_almost_equal(bi[0],array([-1.173713222709127,
                                               -3.271093302836352,
                                               -4.830737841662016,
                                               -6.169852128310251,
                                               -7.376762079367764]),11)

        assert_array_almost_equal(bi[1],array([-2.294439682614122,
                                               -4.073155089071828,
                                               -5.512395729663599,
                                               -6.781294445990305,
                                               -7.940178689168587]),10)

        assert_array_almost_equal(bi[2],array([-0.454944383639657,
                                               0.396522836094465,
                                               -0.367969161486959,
                                               0.349499116831805,
                                               -0.336026240133662]),11)

        assert_array_almost_equal(bi[3],array([0.601957887976239,
                                               -0.760310141492801,
                                               0.836991012619261,
                                               -0.88947990142654,
                                               0.929983638568022]),10)

    def test_ai_zeros(self):
        ai = special.ai_zeros(1)
        assert_array_almost_equal(ai,(array([-2.33810741]),
                                     array([-1.01879297]),
                                     array([0.5357]),
                                     array([0.7012])),4)

    def test_ai_zeros_big(self):
        z, zp, ai_zpx, aip_zx = special.ai_zeros(50000)
        ai_z, aip_z, _, _ = special.airy(z)
        ai_zp, aip_zp, _, _ = special.airy(zp)

        ai_envelope = 1/abs(z)**(1./4)
        aip_envelope = abs(zp)**(1./4)

        # Check values
        assert_allclose(ai_zpx, ai_zp, rtol=1e-10)
        assert_allclose(aip_zx, aip_z, rtol=1e-10)

        # Check they are zeros
        assert_allclose(ai_z/ai_envelope, 0, atol=1e-10, rtol=0)
        assert_allclose(aip_zp/aip_envelope, 0, atol=1e-10, rtol=0)

        # Check first zeros, DLMF 9.9.1
        assert_allclose(z[:6],
            [-2.3381074105, -4.0879494441, -5.5205598281,
             -6.7867080901, -7.9441335871, -9.0226508533], rtol=1e-10)
        assert_allclose(zp[:6],
            [-1.0187929716, -3.2481975822, -4.8200992112,
             -6.1633073556, -7.3721772550, -8.4884867340], rtol=1e-10)

    def test_bi_zeros_big(self):
        z, zp, bi_zpx, bip_zx = special.bi_zeros(50000)
        _, _, bi_z, bip_z = special.airy(z)
        _, _, bi_zp, bip_zp = special.airy(zp)

        bi_envelope = 1/abs(z)**(1./4)
        bip_envelope = abs(zp)**(1./4)

        # Check values
        assert_allclose(bi_zpx, bi_zp, rtol=1e-10)
        assert_allclose(bip_zx, bip_z, rtol=1e-10)

        # Check they are zeros
        assert_allclose(bi_z/bi_envelope, 0, atol=1e-10, rtol=0)
        assert_allclose(bip_zp/bip_envelope, 0, atol=1e-10, rtol=0)

        # Check first zeros, DLMF 9.9.2
        assert_allclose(z[:6],
            [-1.1737132227, -3.2710933028, -4.8307378417,
             -6.1698521283, -7.3767620794, -8.4919488465], rtol=1e-10)
        assert_allclose(zp[:6],
            [-2.2944396826, -4.0731550891, -5.5123957297,
             -6.7812944460, -7.9401786892, -9.0195833588], rtol=1e-10)


class TestAssocLaguerre:
    def test_assoc_laguerre(self):
        a1 = special.genlaguerre(11,1)
        a2 = special.assoc_laguerre(.2,11,1)
        assert_array_almost_equal(a2,a1(.2),8)
        a2 = special.assoc_laguerre(1,11,1)
        assert_array_almost_equal(a2,a1(1),8)


class TestBesselpoly:
    def test_besselpoly(self):
        pass


class TestKelvin:
    def test_bei(self):
        mbei = special.bei(2)
        assert_almost_equal(mbei, 0.9722916273066613,5)  # this may not be exact

    def test_beip(self):
        mbeip = special.beip(2)
        assert_almost_equal(mbeip,0.91701361338403631,5)  # this may not be exact

    def test_ber(self):
        mber = special.ber(2)
        assert_almost_equal(mber,0.75173418271380821,5)  # this may not be exact

    def test_berp(self):
        mberp = special.berp(2)
        assert_almost_equal(mberp,-0.49306712470943909,5)  # this may not be exact

    def test_bei_zeros(self):
        # Abramowitz & Stegun, Table 9.12
        bi = special.bei_zeros(5)
        assert_array_almost_equal(bi,array([5.02622,
                                            9.45541,
                                            13.89349,
                                            18.33398,
                                            22.77544]),4)

    def test_beip_zeros(self):
        bip = special.beip_zeros(5)
        assert_array_almost_equal(bip,array([3.772673304934953,
                                               8.280987849760042,
                                               12.742147523633703,
                                               17.193431752512542,
                                               21.641143941167325]),8)

    def test_ber_zeros(self):
        ber = special.ber_zeros(5)
        assert_array_almost_equal(ber,array([2.84892,
                                             7.23883,
                                             11.67396,
                                             16.11356,
                                             20.55463]),4)

    def test_berp_zeros(self):
        brp = special.berp_zeros(5)
        assert_array_almost_equal(brp,array([6.03871,
                                             10.51364,
                                             14.96844,
                                             19.41758,
                                             23.86430]),4)

    def test_kelvin(self):
        mkelv = special.kelvin(2)
        assert_array_almost_equal(mkelv,(special.ber(2) + special.bei(2)*1j,
                                         special.ker(2) + special.kei(2)*1j,
                                         special.berp(2) + special.beip(2)*1j,
                                         special.kerp(2) + special.keip(2)*1j),8)

    def test_kei(self):
        mkei = special.kei(2)
        assert_almost_equal(mkei,-0.20240006776470432,5)

    def test_keip(self):
        mkeip = special.keip(2)
        assert_almost_equal(mkeip,0.21980790991960536,5)

    def test_ker(self):
        mker = special.ker(2)
        assert_almost_equal(mker,-0.041664513991509472,5)

    def test_kerp(self):
        mkerp = special.kerp(2)
        assert_almost_equal(mkerp,-0.10660096588105264,5)

    def test_kei_zeros(self):
        kei = special.kei_zeros(5)
        assert_array_almost_equal(kei,array([3.91467,
                                              8.34422,
                                              12.78256,
                                              17.22314,
                                              21.66464]),4)

    def test_keip_zeros(self):
        keip = special.keip_zeros(5)
        assert_array_almost_equal(keip,array([4.93181,
                                                9.40405,
                                                13.85827,
                                                18.30717,
                                                22.75379]),4)

    # numbers come from 9.9 of A&S pg. 381
    def test_kelvin_zeros(self):
        tmp = special.kelvin_zeros(5)
        berz,beiz,kerz,keiz,berpz,beipz,kerpz,keipz = tmp
        assert_array_almost_equal(berz,array([2.84892,
                                               7.23883,
                                               11.67396,
                                               16.11356,
                                               20.55463]),4)
        assert_array_almost_equal(beiz,array([5.02622,
                                               9.45541,
                                               13.89349,
                                               18.33398,
                                               22.77544]),4)
        assert_array_almost_equal(kerz,array([1.71854,
                                               6.12728,
                                               10.56294,
                                               15.00269,
                                               19.44382]),4)
        assert_array_almost_equal(keiz,array([3.91467,
                                               8.34422,
                                               12.78256,
                                               17.22314,
                                               21.66464]),4)
        assert_array_almost_equal(berpz,array([6.03871,
                                                10.51364,
                                                14.96844,
                                                19.41758,
                                                23.86430]),4)
        assert_array_almost_equal(beipz,array([3.77267,
                 # table from 1927 had 3.77320
                 #  but this is more accurate
                                                8.28099,
                                                12.74215,
                                                17.19343,
                                                21.64114]),4)
        assert_array_almost_equal(kerpz,array([2.66584,
                                                7.17212,
                                                11.63218,
                                                16.08312,
                                                20.53068]),4)
        assert_array_almost_equal(keipz,array([4.93181,
                                                9.40405,
                                                13.85827,
                                                18.30717,
                                                22.75379]),4)

    def test_ker_zeros(self):
        ker = special.ker_zeros(5)
        assert_array_almost_equal(ker,array([1.71854,
                                               6.12728,
                                               10.56294,
                                               15.00269,
                                               19.44381]),4)

    def test_kerp_zeros(self):
        kerp = special.kerp_zeros(5)
        assert_array_almost_equal(kerp,array([2.66584,
                                                7.17212,
                                                11.63218,
                                                16.08312,
                                                20.53068]),4)


class TestBernoulli:
    def test_bernoulli(self):
        brn = special.bernoulli(5)
        assert_array_almost_equal(brn,array([1.0000,
                                             -0.5000,
                                             0.1667,
                                             0.0000,
                                             -0.0333,
                                             0.0000]),4)


class TestBeta:
    """
    Test beta and betaln.
    """

    def test_beta(self):
        assert_equal(special.beta(1, 1), 1.0)
        assert_allclose(special.beta(-100.3, 1e-200), special.gamma(1e-200))
        assert_allclose(special.beta(0.0342, 171), 24.070498359873497,
                        rtol=1e-13, atol=0)

        bet = special.beta(2, 4)
        betg = (special.gamma(2)*special.gamma(4))/special.gamma(6)
        assert_allclose(bet, betg, rtol=1e-13)

    def test_beta_inf(self):
        assert_(np.isinf(special.beta(-1, 2)))

    def test_betaln(self):
        assert_equal(special.betaln(1, 1), 0.0)
        assert_allclose(special.betaln(-100.3, 1e-200),
                        special.gammaln(1e-200))
        assert_allclose(special.betaln(0.0342, 170), 3.1811881124242447,
                        rtol=1e-14, atol=0)

        betln = special.betaln(2, 4)
        bet = log(abs(special.beta(2, 4)))
        assert_allclose(betln, bet, rtol=1e-13)


class TestBetaInc:
    """
    Tests for betainc, betaincinv, betaincc, betainccinv.
    """

    def test_a1_b1(self):
        # betainc(1, 1, x) is x.
        x = np.array([0, 0.25, 1])
        assert_equal(special.betainc(1, 1, x), x)
        assert_equal(special.betaincinv(1, 1, x), x)
        assert_equal(special.betaincc(1, 1, x), 1 - x)
        assert_equal(special.betainccinv(1, 1, x), 1 - x)

    # Nontrivial expected values computed with mpmath:
    #    from mpmath import mp
    #    mp.dps = 100
    #    p = mp.betainc(a, b, 0, x, regularized=True)
    #
    # or, e.g.,
    #
    #    p = 0.25
    #    a, b = 0.0342, 171
    #    x = mp.findroot(
    #        lambda t: mp.betainc(a, b, 0, t, regularized=True) - p,
    #        (8e-21, 9e-21),
    #        solver='anderson',
    #    )
    #
    @pytest.mark.parametrize(
        'a, b, x, p',
        [(2, 4, 0.3138101704556974, 0.5),
         (0.0342, 171.0, 1e-10, 0.552699169018070910641),
         # gh-3761:
         (0.0342, 171, 8.42313169354797e-21, 0.25),
         # gh-4244:
         (0.0002742794749792665, 289206.03125, 1.639984034231756e-56,
          0.9688708782196045),
         # gh-12796:
         (4, 99997, 0.0001947841578892121, 0.999995)])
    def test_betainc_betaincinv(self, a, b, x, p):
        p1 = special.betainc(a, b, x)
        assert_allclose(p1, p, rtol=1e-15)
        x1 = special.betaincinv(a, b, p)
        assert_allclose(x1, x, rtol=5e-13)

    # Expected values computed with mpmath:
    #     from mpmath import mp
    #     mp.dps = 100
    #     p = mp.betainc(a, b, x, 1, regularized=True)
    @pytest.mark.parametrize('a, b, x, p',
                             [(2.5, 3.0, 0.25, 0.833251953125),
                              (7.5, 13.25, 0.375, 0.43298734645560368593),
                              (0.125, 7.5, 0.425, 0.0006688257851314237),
                              (0.125, 18.0, 1e-6, 0.72982359145096327654),
                              (0.125, 18.0, 0.996, 7.2745875538380150586e-46),
                              (0.125, 24.0, 0.75, 3.70853404816862016966e-17),
                              (16.0, 0.75, 0.99999999975,
                               5.4408759277418629909e-07),
                              # gh-4677 (numbers from stackoverflow question):
                              (0.4211959643503401, 16939.046996018118,
                               0.000815296167195521, 1e-7)])
    def test_betaincc_betainccinv(self, a, b, x, p):
        p1 = special.betaincc(a, b, x)
        assert_allclose(p1, p, rtol=5e-15)
        x1 = special.betainccinv(a, b, p)
        assert_allclose(x1, x, rtol=8e-15)

    @pytest.mark.parametrize(
        'a, b, y, ref',
        [(14.208308325339239, 14.208308325339239, 7.703145458496392e-307,
          8.566004561846704e-23),
         (14.0, 14.5, 1e-280, 2.9343915006642424e-21),
         (3.5, 15.0, 4e-95, 1.3290751429289227e-28),
         (10.0, 1.25, 2e-234, 3.982659092143654e-24),
         (4.0, 99997.0, 5e-88, 3.309800566862242e-27)]
    )
    def test_betaincinv_tiny_y(self, a, b, y, ref):
        # Test with extremely small y values.  This test includes
        # a regression test for an issue in the boost code;
        # see https://github.com/boostorg/math/issues/961
        #
        # The reference values were computed with mpmath. For example,
        #
        #   from mpmath import mp
        #   mp.dps = 1000
        #   a = 14.208308325339239
        #   p = 7.703145458496392e-307
        #   x = mp.findroot(lambda t: mp.betainc(a, a, 0, t,
        #                                        regularized=True) - p,
        #                   x0=8.566e-23)
        #   print(float(x))
        #
        x = special.betaincinv(a, b, y)
        assert_allclose(x, ref, rtol=1e-14)

    @pytest.mark.parametrize('func', [special.betainc, special.betaincinv,
                                      special.betaincc, special.betainccinv])
    @pytest.mark.parametrize('args', [(-1.0, 2, 0.5), (0, 2, 0.5),
                                      (1.5, -2.0, 0.5), (1.5, 0, 0.5),
                                      (1.5, 2.0, -0.3), (1.5, 2.0, 1.1)])
    def test_betainc_domain_errors(self, func, args):
        with special.errstate(domain='raise'):
            with pytest.raises(special.SpecialFunctionError, match='domain'):
                special.betainc(*args)


class TestCombinatorics:
    def test_comb(self):
        assert_array_almost_equal(special.comb([10, 10], [3, 4]), [120., 210.])
        assert_almost_equal(special.comb(10, 3), 120.)
        assert_equal(special.comb(10, 3, exact=True), 120)
        assert_equal(special.comb(10, 3, exact=True, repetition=True), 220)

        assert_allclose([special.comb(20, k, exact=True) for k in range(21)],
                        special.comb(20, list(range(21))), atol=1e-15)

        ii = np.iinfo(int).max + 1
        assert_equal(special.comb(ii, ii-1, exact=True), ii)

        expected = 100891344545564193334812497256
        assert special.comb(100, 50, exact=True) == expected

    @pytest.mark.parametrize("repetition", [True, False])
    @pytest.mark.parametrize("legacy", [True, False, _NoValue])
    @pytest.mark.parametrize("k", [3.5, 3])
    @pytest.mark.parametrize("N", [4.5, 4])
    def test_comb_legacy(self, N, k, legacy, repetition):
        # test is only relevant for exact=True
        if legacy is not _NoValue:
            with pytest.warns(
                DeprecationWarning,
                match=r"Using 'legacy' keyword is deprecated"
            ):
                result = special.comb(N, k, exact=True, legacy=legacy,
                                      repetition=repetition)
        else:
            result = special.comb(N, k, exact=True, legacy=legacy,
                                  repetition=repetition)
        if legacy:
            # for exact=True and legacy=True, cast input arguments, else don't
            if repetition:
                # the casting in legacy mode happens AFTER transforming N & k,
                # so rounding can change (e.g. both floats, but sum to int);
                # hence we need to emulate the repetition-transformation here
                N, k = int(N + k - 1), int(k)
                repetition = False
            else:
                N, k = int(N), int(k)
        # expected result is the same as with exact=False
        with suppress_warnings() as sup:
            if legacy is not _NoValue:
                sup.filter(DeprecationWarning)
            expected = special.comb(N, k, legacy=legacy, repetition=repetition)
        assert_equal(result, expected)

    def test_comb_with_np_int64(self):
        n = 70
        k = 30
        np_n = np.int64(n)
        np_k = np.int64(k)
        res_np = special.comb(np_n, np_k, exact=True)
        res_py = special.comb(n, k, exact=True)
        assert res_np == res_py

    def test_comb_zeros(self):
        assert_equal(special.comb(2, 3, exact=True), 0)
        assert_equal(special.comb(-1, 3, exact=True), 0)
        assert_equal(special.comb(2, -1, exact=True), 0)
        assert_equal(special.comb(2, -1, exact=False), 0)
        assert_array_almost_equal(special.comb([2, -1, 2, 10], [3, 3, -1, 3]),
                [0., 0., 0., 120.])

    def test_perm(self):
        assert_array_almost_equal(special.perm([10, 10], [3, 4]), [720., 5040.])
        assert_almost_equal(special.perm(10, 3), 720.)
        assert_equal(special.perm(10, 3, exact=True), 720)

    def test_perm_zeros(self):
        assert_equal(special.perm(2, 3, exact=True), 0)
        assert_equal(special.perm(-1, 3, exact=True), 0)
        assert_equal(special.perm(2, -1, exact=True), 0)
        assert_equal(special.perm(2, -1, exact=False), 0)
        assert_array_almost_equal(special.perm([2, -1, 2, 10], [3, 3, -1, 3]),
                [0., 0., 0., 720.])

    def test_positional_deprecation(self):
        with pytest.deprecated_call(match="use keyword arguments"):
            # from test_comb
            special.comb([10, 10], [3, 4], False, False)


class TestTrigonometric:
    def test_cbrt(self):
        cb = special.cbrt(27)
        cbrl = 27**(1.0/3.0)
        assert_approx_equal(cb,cbrl)

    def test_cbrtmore(self):
        cb1 = special.cbrt(27.9)
        cbrl1 = 27.9**(1.0/3.0)
        assert_almost_equal(cb1,cbrl1,8)

    def test_cosdg(self):
        cdg = special.cosdg(90)
        cdgrl = cos(pi/2.0)
        assert_almost_equal(cdg,cdgrl,8)

    def test_cosdgmore(self):
        cdgm = special.cosdg(30)
        cdgmrl = cos(pi/6.0)
        assert_almost_equal(cdgm,cdgmrl,8)

    def test_cosm1(self):
        cs = (special.cosm1(0),special.cosm1(.3),special.cosm1(pi/10))
        csrl = (cos(0)-1,cos(.3)-1,cos(pi/10)-1)
        assert_array_almost_equal(cs,csrl,8)

    def test_cotdg(self):
        ct = special.cotdg(30)
        ctrl = tan(pi/6.0)**(-1)
        assert_almost_equal(ct,ctrl,8)

    def test_cotdgmore(self):
        ct1 = special.cotdg(45)
        ctrl1 = tan(pi/4.0)**(-1)
        assert_almost_equal(ct1,ctrl1,8)

    def test_specialpoints(self):
        assert_almost_equal(special.cotdg(45), 1.0, 14)
        assert_almost_equal(special.cotdg(-45), -1.0, 14)
        assert_almost_equal(special.cotdg(90), 0.0, 14)
        assert_almost_equal(special.cotdg(-90), 0.0, 14)
        assert_almost_equal(special.cotdg(135), -1.0, 14)
        assert_almost_equal(special.cotdg(-135), 1.0, 14)
        assert_almost_equal(special.cotdg(225), 1.0, 14)
        assert_almost_equal(special.cotdg(-225), -1.0, 14)
        assert_almost_equal(special.cotdg(270), 0.0, 14)
        assert_almost_equal(special.cotdg(-270), 0.0, 14)
        assert_almost_equal(special.cotdg(315), -1.0, 14)
        assert_almost_equal(special.cotdg(-315), 1.0, 14)
        assert_almost_equal(special.cotdg(765), 1.0, 14)

    def test_sinc(self):
        # the sinc implementation and more extensive sinc tests are in numpy
        assert_array_equal(special.sinc([0]), 1)
        assert_equal(special.sinc(0.0), 1.0)

    def test_sindg(self):
        sn = special.sindg(90)
        assert_equal(sn,1.0)

    def test_sindgmore(self):
        snm = special.sindg(30)
        snmrl = sin(pi/6.0)
        assert_almost_equal(snm,snmrl,8)
        snm1 = special.sindg(45)
        snmrl1 = sin(pi/4.0)
        assert_almost_equal(snm1,snmrl1,8)


class TestTandg:

    def test_tandg(self):
        tn = special.tandg(30)
        tnrl = tan(pi/6.0)
        assert_almost_equal(tn,tnrl,8)

    def test_tandgmore(self):
        tnm = special.tandg(45)
        tnmrl = tan(pi/4.0)
        assert_almost_equal(tnm,tnmrl,8)
        tnm1 = special.tandg(60)
        tnmrl1 = tan(pi/3.0)
        assert_almost_equal(tnm1,tnmrl1,8)

    def test_specialpoints(self):
        assert_almost_equal(special.tandg(0), 0.0, 14)
        assert_almost_equal(special.tandg(45), 1.0, 14)
        assert_almost_equal(special.tandg(-45), -1.0, 14)
        assert_almost_equal(special.tandg(135), -1.0, 14)
        assert_almost_equal(special.tandg(-135), 1.0, 14)
        assert_almost_equal(special.tandg(180), 0.0, 14)
        assert_almost_equal(special.tandg(-180), 0.0, 14)
        assert_almost_equal(special.tandg(225), 1.0, 14)
        assert_almost_equal(special.tandg(-225), -1.0, 14)
        assert_almost_equal(special.tandg(315), -1.0, 14)
        assert_almost_equal(special.tandg(-315), 1.0, 14)


class TestEllip:
    def test_ellipj_nan(self):
        """Regression test for #912."""
        special.ellipj(0.5, np.nan)

    def test_ellipj(self):
        el = special.ellipj(0.2,0)
        rel = [sin(0.2),cos(0.2),1.0,0.20]
        assert_array_almost_equal(el,rel,13)

    def test_ellipk(self):
        elk = special.ellipk(.2)
        assert_almost_equal(elk,1.659623598610528,11)

        assert_equal(special.ellipkm1(0.0), np.inf)
        assert_equal(special.ellipkm1(1.0), pi/2)
        assert_equal(special.ellipkm1(np.inf), 0.0)
        assert_equal(special.ellipkm1(np.nan), np.nan)
        assert_equal(special.ellipkm1(-1), np.nan)
        assert_allclose(special.ellipk(-10), 0.7908718902387385)

    def test_ellipkinc(self):
        elkinc = special.ellipkinc(pi/2,.2)
        elk = special.ellipk(0.2)
        assert_almost_equal(elkinc,elk,15)
        alpha = 20*pi/180
        phi = 45*pi/180
        m = sin(alpha)**2
        elkinc = special.ellipkinc(phi,m)
        assert_almost_equal(elkinc,0.79398143,8)
        # From pg. 614 of A & S

        assert_equal(special.ellipkinc(pi/2, 0.0), pi/2)
        assert_equal(special.ellipkinc(pi/2, 1.0), np.inf)
        assert_equal(special.ellipkinc(pi/2, -np.inf), 0.0)
        assert_equal(special.ellipkinc(pi/2, np.nan), np.nan)
        assert_equal(special.ellipkinc(pi/2, 2), np.nan)
        assert_equal(special.ellipkinc(0, 0.5), 0.0)
        assert_equal(special.ellipkinc(np.inf, 0.5), np.inf)
        assert_equal(special.ellipkinc(-np.inf, 0.5), -np.inf)
        assert_equal(special.ellipkinc(np.inf, np.inf), np.nan)
        assert_equal(special.ellipkinc(np.inf, -np.inf), np.nan)
        assert_equal(special.ellipkinc(-np.inf, -np.inf), np.nan)
        assert_equal(special.ellipkinc(-np.inf, np.inf), np.nan)
        assert_equal(special.ellipkinc(np.nan, 0.5), np.nan)
        assert_equal(special.ellipkinc(np.nan, np.nan), np.nan)

        assert_allclose(special.ellipkinc(0.38974112035318718, 1), 0.4, rtol=1e-14)
        assert_allclose(special.ellipkinc(1.5707, -10), 0.79084284661724946)

    def test_ellipkinc_2(self):
        # Regression test for gh-3550
        # ellipkinc(phi, mbad) was NaN and mvals[2:6] were twice the correct value
        mbad = 0.68359375000000011
        phi = 0.9272952180016123
        m = np.nextafter(mbad, 0)
        mvals = []
        for j in range(10):
            mvals.append(m)
            m = np.nextafter(m, 1)
        f = special.ellipkinc(phi, mvals)
        assert_array_almost_equal_nulp(f, np.full_like(f, 1.0259330100195334), 1)
        # this bug also appears at phi + n * pi for at least small n
        f1 = special.ellipkinc(phi + pi, mvals)
        assert_array_almost_equal_nulp(f1, np.full_like(f1, 5.1296650500976675), 2)

    def test_ellipkinc_singular(self):
        # ellipkinc(phi, 1) has closed form and is finite only for phi in (-pi/2, pi/2)
        xlog = np.logspace(-300, -17, 25)
        xlin = np.linspace(1e-17, 0.1, 25)
        xlin2 = np.linspace(0.1, pi/2, 25, endpoint=False)

        assert_allclose(special.ellipkinc(xlog, 1), np.arcsinh(np.tan(xlog)),
                        rtol=1e14)
        assert_allclose(special.ellipkinc(xlin, 1), np.arcsinh(np.tan(xlin)),
                        rtol=1e14)
        assert_allclose(special.ellipkinc(xlin2, 1), np.arcsinh(np.tan(xlin2)),
                        rtol=1e14)
        assert_equal(special.ellipkinc(np.pi/2, 1), np.inf)
        assert_allclose(special.ellipkinc(-xlog, 1), np.arcsinh(np.tan(-xlog)),
                        rtol=1e14)
        assert_allclose(special.ellipkinc(-xlin, 1), np.arcsinh(np.tan(-xlin)),
                        rtol=1e14)
        assert_allclose(special.ellipkinc(-xlin2, 1), np.arcsinh(np.tan(-xlin2)),
                        rtol=1e14)
        assert_equal(special.ellipkinc(-np.pi/2, 1), np.inf)

    def test_ellipe(self):
        ele = special.ellipe(.2)
        assert_almost_equal(ele,1.4890350580958529,8)

        assert_equal(special.ellipe(0.0), pi/2)
        assert_equal(special.ellipe(1.0), 1.0)
        assert_equal(special.ellipe(-np.inf), np.inf)
        assert_equal(special.ellipe(np.nan), np.nan)
        assert_equal(special.ellipe(2), np.nan)
        assert_allclose(special.ellipe(-10), 3.6391380384177689)

    def test_ellipeinc(self):
        eleinc = special.ellipeinc(pi/2,.2)
        ele = special.ellipe(0.2)
        assert_almost_equal(eleinc,ele,14)
        # pg 617 of A & S
        alpha, phi = 52*pi/180,35*pi/180
        m = sin(alpha)**2
        eleinc = special.ellipeinc(phi,m)
        assert_almost_equal(eleinc, 0.58823065, 8)

        assert_equal(special.ellipeinc(pi/2, 0.0), pi/2)
        assert_equal(special.ellipeinc(pi/2, 1.0), 1.0)
        assert_equal(special.ellipeinc(pi/2, -np.inf), np.inf)
        assert_equal(special.ellipeinc(pi/2, np.nan), np.nan)
        assert_equal(special.ellipeinc(pi/2, 2), np.nan)
        assert_equal(special.ellipeinc(0, 0.5), 0.0)
        assert_equal(special.ellipeinc(np.inf, 0.5), np.inf)
        assert_equal(special.ellipeinc(-np.inf, 0.5), -np.inf)
        assert_equal(special.ellipeinc(np.inf, -np.inf), np.inf)
        assert_equal(special.ellipeinc(-np.inf, -np.inf), -np.inf)
        assert_equal(special.ellipeinc(np.inf, np.inf), np.nan)
        assert_equal(special.ellipeinc(-np.inf, np.inf), np.nan)
        assert_equal(special.ellipeinc(np.nan, 0.5), np.nan)
        assert_equal(special.ellipeinc(np.nan, np.nan), np.nan)
        assert_allclose(special.ellipeinc(1.5707, -10), 3.6388185585822876)

    def test_ellipeinc_2(self):
        # Regression test for gh-3550
        # ellipeinc(phi, mbad) was NaN and mvals[2:6] were twice the correct value
        mbad = 0.68359375000000011
        phi = 0.9272952180016123
        m = np.nextafter(mbad, 0)
        mvals = []
        for j in range(10):
            mvals.append(m)
            m = np.nextafter(m, 1)
        f = special.ellipeinc(phi, mvals)
        assert_array_almost_equal_nulp(f, np.full_like(f, 0.84442884574781019), 2)
        # this bug also appears at phi + n * pi for at least small n
        f1 = special.ellipeinc(phi + pi, mvals)
        assert_array_almost_equal_nulp(f1, np.full_like(f1, 3.3471442287390509), 4)


class TestEllipCarlson:
    """Test for Carlson elliptic integrals ellipr[cdfgj].
    The special values used in these tests can be found in Sec. 3 of Carlson
    (1994), https://arxiv.org/abs/math/9409227
    """
    def test_elliprc(self):
        assert_allclose(elliprc(1, 1), 1)
        assert elliprc(1, inf) == 0.0
        assert isnan(elliprc(1, 0))
        assert elliprc(1, complex(1, inf)) == 0.0
        args = array([[0.0, 0.25],
                      [2.25, 2.0],
                      [0.0, 1.0j],
                      [-1.0j, 1.0j],
                      [0.25, -2.0],
                      [1.0j, -1.0]])
        expected_results = array([np.pi,
                                  np.log(2.0),
                                  1.1107207345396 * (1.0-1.0j),
                                  1.2260849569072-0.34471136988768j,
                                  np.log(2.0) / 3.0,
                                  0.77778596920447+0.19832484993429j])
        for i, arr in enumerate(args):
            assert_allclose(elliprc(*arr), expected_results[i])

    def test_elliprd(self):
        assert_allclose(elliprd(1, 1, 1), 1)
        assert_allclose(elliprd(0, 2, 1) / 3.0, 0.59907011736779610371)
        assert elliprd(1, 1, inf) == 0.0
        assert np.isinf(elliprd(1, 1, 0))
        assert np.isinf(elliprd(1, 1, complex(0, 0)))
        assert np.isinf(elliprd(0, 1, complex(0, 0)))
        assert isnan(elliprd(1, 1, -np.finfo(np.float64).tiny / 2.0))
        assert isnan(elliprd(1, 1, complex(-1, 0)))
        args = array([[0.0, 2.0, 1.0],
                      [2.0, 3.0, 4.0],
                      [1.0j, -1.0j, 2.0],
                      [0.0, 1.0j, -1.0j],
                      [0.0, -1.0+1.0j, 1.0j],
                      [-2.0-1.0j, -1.0j, -1.0+1.0j]])
        expected_results = array([1.7972103521034,
                                  0.16510527294261,
                                  0.65933854154220,
                                  1.2708196271910+2.7811120159521j,
                                  -1.8577235439239-0.96193450888839j,
                                  1.8249027393704-1.2218475784827j])
        for i, arr in enumerate(args):
            assert_allclose(elliprd(*arr), expected_results[i])

    def test_elliprf(self):
        assert_allclose(elliprf(1, 1, 1), 1)
        assert_allclose(elliprf(0, 1, 2), 1.31102877714605990523)
        assert elliprf(1, inf, 1) == 0.0
        assert np.isinf(elliprf(0, 1, 0))
        assert isnan(elliprf(1, 1, -1))
        assert elliprf(complex(inf), 0, 1) == 0.0
        assert isnan(elliprf(1, 1, complex(-inf, 1)))
        args = array([[1.0, 2.0, 0.0],
                      [1.0j, -1.0j, 0.0],
                      [0.5, 1.0, 0.0],
                      [-1.0+1.0j, 1.0j, 0.0],
                      [2.0, 3.0, 4.0],
                      [1.0j, -1.0j, 2.0],
                      [-1.0+1.0j, 1.0j, 1.0-1.0j]])
        expected_results = array([1.3110287771461,
                                  1.8540746773014,
                                  1.8540746773014,
                                  0.79612586584234-1.2138566698365j,
                                  0.58408284167715,
                                  1.0441445654064,
                                  0.93912050218619-0.53296252018635j])
        for i, arr in enumerate(args):
            assert_allclose(elliprf(*arr), expected_results[i])

    def test_elliprg(self):
        assert_allclose(elliprg(1, 1, 1), 1)
        assert_allclose(elliprg(0, 0, 1), 0.5)
        assert_allclose(elliprg(0, 0, 0), 0)
        assert np.isinf(elliprg(1, inf, 1))
        assert np.isinf(elliprg(complex(inf), 1, 1))
        args = array([[0.0, 16.0, 16.0],
                      [2.0, 3.0, 4.0],
                      [0.0, 1.0j, -1.0j],
                      [-1.0+1.0j, 1.0j, 0.0],
                      [-1.0j, -1.0+1.0j, 1.0j],
                      [0.0, 0.0796, 4.0]])
        expected_results = array([np.pi,
                                  1.7255030280692,
                                  0.42360654239699,
                                  0.44660591677018+0.70768352357515j,
                                  0.36023392184473+0.40348623401722j,
                                  1.0284758090288])
        for i, arr in enumerate(args):
            assert_allclose(elliprg(*arr), expected_results[i])

    def test_elliprj(self):
        assert_allclose(elliprj(1, 1, 1, 1), 1)
        assert elliprj(1, 1, inf, 1) == 0.0
        assert isnan(elliprj(1, 0, 0, 0))
        assert isnan(elliprj(-1, 1, 1, 1))
        assert elliprj(1, 1, 1, inf) == 0.0
        args = array([[0.0, 1.0, 2.0, 3.0],
                      [2.0, 3.0, 4.0, 5.0],
                      [2.0, 3.0, 4.0, -1.0+1.0j],
                      [1.0j, -1.0j, 0.0, 2.0],
                      [-1.0+1.0j, -1.0-1.0j, 1.0, 2.0],
                      [1.0j, -1.0j, 0.0, 1.0-1.0j],
                      [-1.0+1.0j, -1.0-1.0j, 1.0, -3.0+1.0j],
                      [2.0, 3.0, 4.0, -0.5],    # Cauchy principal value
                      [2.0, 3.0, 4.0, -5.0]])   # Cauchy principal value
        expected_results = array([0.77688623778582,
                                  0.14297579667157,
                                  0.13613945827771-0.38207561624427j,
                                  1.6490011662711,
                                  0.94148358841220,
                                  1.8260115229009+1.2290661908643j,
                                  -0.61127970812028-1.0684038390007j,
                                  0.24723819703052,    # Cauchy principal value
                                  -0.12711230042964])  # Caucny principal value
        for i, arr in enumerate(args):
            assert_allclose(elliprj(*arr), expected_results[i])

    @pytest.mark.xfail(reason="Insufficient accuracy on 32-bit")
    def test_elliprj_hard(self):
        assert_allclose(elliprj(6.483625725195452e-08,
                                1.1649136528196886e-27,
                                3.6767340167168e+13,
                                0.493704617023468),
                        8.63426920644241857617477551054e-6,
                        rtol=5e-15, atol=1e-20)
        assert_allclose(elliprj(14.375105857849121,
                                9.993988969725365e-11,
                                1.72844262269944e-26,
                                5.898871222598245e-06),
                        829774.1424801627252574054378691828,
                        rtol=5e-15, atol=1e-20)


class TestEllipLegendreCarlsonIdentities:
    """Test identities expressing the Legendre elliptic integrals in terms
    of Carlson's symmetric integrals.  These identities can be found
    in the DLMF https://dlmf.nist.gov/19.25#i .
    """

    def setup_class(self):
        self.m_n1_1 = np.arange(-1., 1., 0.01)
        # For double, this is -(2**1024)
        self.max_neg = finfo(double).min
        # Lots of very negative numbers
        self.very_neg_m = -1. * 2.**arange(-1 +
                                           np.log2(-self.max_neg), 0.,
                                           -1.)
        self.ms_up_to_1 = np.concatenate(([self.max_neg],
                                          self.very_neg_m,
                                          self.m_n1_1))

    def test_k(self):
        """Test identity:
        K(m) = R_F(0, 1-m, 1)
        """
        m = self.ms_up_to_1
        assert_allclose(ellipk(m), elliprf(0., 1.-m, 1.))

    def test_km1(self):
        """Test identity:
        K(m) = R_F(0, 1-m, 1)
        But with the ellipkm1 function
        """
        # For double, this is 2**-1022
        tiny = finfo(double).tiny
        # All these small powers of 2, up to 2**-1
        m1 = tiny * 2.**arange(0., -np.log2(tiny))
        assert_allclose(ellipkm1(m1), elliprf(0., m1, 1.))

    def test_e(self):
        """Test identity:
        E(m) = 2*R_G(0, 1-k^2, 1)
        """
        m = self.ms_up_to_1
        assert_allclose(ellipe(m), 2.*elliprg(0., 1.-m, 1.))


class TestErf:

    def test_erf(self):
        er = special.erf(.25)
        assert_almost_equal(er,0.2763263902,8)

    def test_erf_zeros(self):
        erz = special.erf_zeros(5)
        erzr = array([1.45061616+1.88094300j,
                     2.24465928+2.61657514j,
                     2.83974105+3.17562810j,
                     3.33546074+3.64617438j,
                     3.76900557+4.06069723j])
        assert_array_almost_equal(erz,erzr,4)

    def _check_variant_func(self, func, other_func, rtol, atol=0):
        np.random.seed(1234)
        n = 10000
        x = np.random.pareto(0.02, n) * (2*np.random.randint(0, 2, n) - 1)
        y = np.random.pareto(0.02, n) * (2*np.random.randint(0, 2, n) - 1)
        z = x + 1j*y

        with np.errstate(all='ignore'):
            w = other_func(z)
            w_real = other_func(x).real

            mask = np.isfinite(w)
            w = w[mask]
            z = z[mask]

            mask = np.isfinite(w_real)
            w_real = w_real[mask]
            x = x[mask]

            # test both real and complex variants
            assert_func_equal(func, w, z, rtol=rtol, atol=atol)
            assert_func_equal(func, w_real, x, rtol=rtol, atol=atol)

    def test_erfc_consistent(self):
        self._check_variant_func(
            cephes.erfc,
            lambda z: 1 - cephes.erf(z),
            rtol=1e-12,
            atol=1e-14  # <- the test function loses precision
            )

    def test_erfcx_consistent(self):
        self._check_variant_func(
            cephes.erfcx,
            lambda z: np.exp(z*z) * cephes.erfc(z),
            rtol=1e-12
            )

    def test_erfi_consistent(self):
        self._check_variant_func(
            cephes.erfi,
            lambda z: -1j * cephes.erf(1j*z),
            rtol=1e-12
            )

    def test_dawsn_consistent(self):
        self._check_variant_func(
            cephes.dawsn,
            lambda z: sqrt(pi)/2 * np.exp(-z*z) * cephes.erfi(z),
            rtol=1e-12
            )

    def test_erf_nan_inf(self):
        vals = [np.nan, -np.inf, np.inf]
        expected = [np.nan, -1, 1]
        assert_allclose(special.erf(vals), expected, rtol=1e-15)

    def test_erfc_nan_inf(self):
        vals = [np.nan, -np.inf, np.inf]
        expected = [np.nan, 2, 0]
        assert_allclose(special.erfc(vals), expected, rtol=1e-15)

    def test_erfcx_nan_inf(self):
        vals = [np.nan, -np.inf, np.inf]
        expected = [np.nan, np.inf, 0]
        assert_allclose(special.erfcx(vals), expected, rtol=1e-15)

    def test_erfi_nan_inf(self):
        vals = [np.nan, -np.inf, np.inf]
        expected = [np.nan, -np.inf, np.inf]
        assert_allclose(special.erfi(vals), expected, rtol=1e-15)

    def test_dawsn_nan_inf(self):
        vals = [np.nan, -np.inf, np.inf]
        expected = [np.nan, -0.0, 0.0]
        assert_allclose(special.dawsn(vals), expected, rtol=1e-15)

    def test_wofz_nan_inf(self):
        vals = [np.nan, -np.inf, np.inf]
        expected = [np.nan + np.nan * 1.j, 0.-0.j, 0.+0.j]
        assert_allclose(special.wofz(vals), expected, rtol=1e-15)


class TestEuler:
    def test_euler(self):
        eu0 = special.euler(0)
        eu1 = special.euler(1)
        eu2 = special.euler(2)   # just checking segfaults
        assert_allclose(eu0, [1], rtol=1e-15)
        assert_allclose(eu1, [1, 0], rtol=1e-15)
        assert_allclose(eu2, [1, 0, -1], rtol=1e-15)
        eu24 = special.euler(24)
        mathworld = [1,1,5,61,1385,50521,2702765,199360981,
                     19391512145,2404879675441,
                     370371188237525,69348874393137901,
                     15514534163557086905]
        correct = zeros((25,),'d')
        for k in range(0,13):
            if (k % 2):
                correct[2*k] = -float(mathworld[k])
            else:
                correct[2*k] = float(mathworld[k])
        with np.errstate(all='ignore'):
            err = nan_to_num((eu24-correct)/correct)
            errmax = max(err)
        assert_almost_equal(errmax, 0.0, 14)


class TestExp:
    def test_exp2(self):
        ex = special.exp2(2)
        exrl = 2**2
        assert_equal(ex,exrl)

    def test_exp2more(self):
        exm = special.exp2(2.5)
        exmrl = 2**(2.5)
        assert_almost_equal(exm,exmrl,8)

    def test_exp10(self):
        ex = special.exp10(2)
        exrl = 10**2
        assert_approx_equal(ex,exrl)

    def test_exp10more(self):
        exm = special.exp10(2.5)
        exmrl = 10**(2.5)
        assert_almost_equal(exm,exmrl,8)

    def test_expm1(self):
        ex = (special.expm1(2),special.expm1(3),special.expm1(4))
        exrl = (exp(2)-1,exp(3)-1,exp(4)-1)
        assert_array_almost_equal(ex,exrl,8)

    def test_expm1more(self):
        ex1 = (special.expm1(2),special.expm1(2.1),special.expm1(2.2))
        exrl1 = (exp(2)-1,exp(2.1)-1,exp(2.2)-1)
        assert_array_almost_equal(ex1,exrl1,8)


class TestFactorialFunctions:
    @pytest.mark.parametrize("exact", [True, False])
    def test_factorialx_scalar_return_type(self, exact):
        assert np.isscalar(special.factorial(1, exact=exact))
        assert np.isscalar(special.factorial2(1, exact=exact))
        assert np.isscalar(special.factorialk(1, 3, exact=True))

    @pytest.mark.parametrize("n", [-1, -2, -3])
    @pytest.mark.parametrize("exact", [True, False])
    def test_factorialx_negative(self, exact, n):
        assert_equal(special.factorial(n, exact=exact), 0)
        assert_equal(special.factorial2(n, exact=exact), 0)
        assert_equal(special.factorialk(n, 3, exact=True), 0)

    @pytest.mark.parametrize("exact", [True, False])
    def test_factorialx_negative_array(self, exact):
        assert_func = assert_array_equal if exact else assert_allclose
        # Consistent output for n < 0
        assert_func(special.factorial([-5, -4, 0, 1], exact=exact),
                    [0, 0, 1, 1])
        assert_func(special.factorial2([-5, -4, 0, 1], exact=exact),
                    [0, 0, 1, 1])
        assert_func(special.factorialk([-5, -4, 0, 1], 3, exact=True),
                    [0, 0, 1, 1])

    @pytest.mark.parametrize("exact", [True, False])
    @pytest.mark.parametrize("content", [np.nan, None, np.datetime64('nat')],
                             ids=["NaN", "None", "NaT"])
    def test_factorialx_nan(self, content, exact):
        # scalar
        assert special.factorial(content, exact=exact) is np.nan
        assert special.factorial2(content, exact=exact) is np.nan
        assert special.factorialk(content, 3, exact=True) is np.nan
        # array-like (initializes np.array with default dtype)
        if content is not np.nan:
            # None causes object dtype, which is not supported; as is datetime
            with pytest.raises(ValueError, match="Unsupported datatype.*"):
                special.factorial([content], exact=exact)
        elif exact:
            # cannot use `is np.nan` see https://stackoverflow.com/a/52124109
            with pytest.warns(DeprecationWarning, match="Non-integer array.*"):
                assert np.isnan(special.factorial([content], exact=exact)[0])
        else:
            assert np.isnan(special.factorial([content], exact=exact)[0])
        # factorial{2,k} don't support array case due to dtype constraints
        with pytest.raises(ValueError, match="factorial2 does not support.*"):
            special.factorial2([content], exact=exact)
        with pytest.raises(ValueError, match="factorialk does not support.*"):
            special.factorialk([content], 3, exact=True)
        # array-case also tested in test_factorial{,2,k}_corner_cases

    @pytest.mark.parametrize("levels", range(1, 5))
    @pytest.mark.parametrize("exact", [True, False])
    def test_factorialx_array_shape(self, levels, exact):
        def _nest_me(x, k=1):
            """
            Double x and nest it k times

            For example:
            >>> _nest_me([3, 4], 2)
            [[[3, 4], [3, 4]], [[3, 4], [3, 4]]]
            """
            if k == 0:
                return x
            else:
                return _nest_me([x, x], k-1)

        def _check(res, nucleus):
            exp = np.array(_nest_me(nucleus, k=levels), dtype=object)
            # test that ndarray shape is maintained
            # need to cast to float due to numpy/numpy#21220
            assert_allclose(res.astype(np.float64), exp.astype(np.float64))

        n = np.array(_nest_me([5, 25], k=levels))
        exp_nucleus = {1: [120, math.factorial(25)],
                       # correctness of factorial2() is tested elsewhere
                       2: [15, special.factorial2(25, exact=True)],
                       3: [10, special.factorialk(25, 3)]}

        _check(special.factorial(n, exact=exact), exp_nucleus[1])
        _check(special.factorial2(n, exact=exact), exp_nucleus[2])
        _check(special.factorialk(n, 3, exact=True), exp_nucleus[3])

    @pytest.mark.parametrize("exact", [True, False])
    @pytest.mark.parametrize("dim", range(0, 5))
    def test_factorialx_array_dimension(self, dim, exact):
        n = np.array(5, ndmin=dim)
        exp = {1: 120, 2: 15, 3: 10}
        assert_allclose(special.factorial(n, exact=exact),
                        np.array(exp[1], ndmin=dim))
        assert_allclose(special.factorial2(n, exact=exact),
                        np.array(exp[2], ndmin=dim))
        assert_allclose(special.factorialk(n, 3, exact=True),
                        np.array(exp[3], ndmin=dim))

    @pytest.mark.parametrize("exact", [True, False])
    @pytest.mark.parametrize("level", range(1, 5))
    def test_factorialx_array_like(self, level, exact):
        def _nest_me(x, k=1):
            if k == 0:
                return x
            else:
                return _nest_me([x], k-1)

        n = _nest_me([5], k=level-1)  # nested list
        exp_nucleus = {1: 120, 2: 15, 3: 10}
        assert_func = assert_array_equal if exact else assert_allclose
        assert_func(special.factorial(n, exact=exact),
                    np.array(exp_nucleus[1], ndmin=level))
        assert_func(special.factorial2(n, exact=exact),
                    np.array(exp_nucleus[2], ndmin=level))
        assert_func(special.factorialk(n, 3, exact=True),
                    np.array(exp_nucleus[3], ndmin=level))

    # note that n=170 is the last integer such that factorial(n) fits float64
    @pytest.mark.parametrize('n', range(30, 180, 10))
    def test_factorial_accuracy(self, n):
        # Compare exact=True vs False, i.e. that the accuracy of the
        # approximation is better than the specified tolerance.

        rtol = 6e-14 if sys.platform == 'win32' else 1e-15
        # need to cast exact result to float due to numpy/numpy#21220
        assert_allclose(float(special.factorial(n, exact=True)),
                        special.factorial(n, exact=False), rtol=rtol)
        assert_allclose(special.factorial([n], exact=True).astype(float),
                        special.factorial([n], exact=False), rtol=rtol)

    @pytest.mark.parametrize('n',
                             list(range(0, 22)) + list(range(30, 180, 10)))
    def test_factorial_int_reference(self, n):
        # Compare all with math.factorial
        correct = math.factorial(n)
        assert_array_equal(correct, special.factorial(n, True))
        assert_array_equal(correct, special.factorial([n], True)[0])

        rtol = 6e-14 if sys.platform == 'win32' else 1e-15
        assert_allclose(float(correct), special.factorial(n, False),
                        rtol=rtol)
        assert_allclose(float(correct), special.factorial([n], False)[0],
                        rtol=rtol)

    @pytest.mark.parametrize("exact", [True, False])
    def test_factorial_float_reference(self, exact):
        def _check(n, expected):
            # support for exact=True with scalar floats grandfathered in
            assert_allclose(special.factorial(n, exact=exact), expected)
            # non-integer types in arrays only allowed with exact=False
            assert_allclose(special.factorial([n])[0], expected)

        # Reference values from mpmath for gamma(n+1)
        _check(0.01, 0.994325851191506032181932988)
        _check(1.11, 1.051609009483625091514147465)
        _check(5.55, 314.9503192327208241614959052)
        _check(11.1, 50983227.84411615655137170553)
        _check(33.3, 2.493363339642036352229215273e+37)
        _check(55.5, 9.479934358436729043289162027e+73)
        _check(77.7, 3.060540559059579022358692625e+114)
        _check(99.9, 5.885840419492871504575693337e+157)
        # close to maximum for float64
        _check(170.6243, 1.79698185749571048960082e+308)

    @pytest.mark.parametrize("dtype", [np.int64, np.float64,
                                       np.complex128, object])
    @pytest.mark.parametrize("exact", [True, False])
    @pytest.mark.parametrize("dim", range(0, 5))
    # test empty & non-empty arrays, with nans and mixed
    @pytest.mark.parametrize("content",
                             [[], [1], [1.1], [np.nan], [np.nan, 1]],
                             ids=["[]", "[1]", "[1.1]", "[NaN]", "[NaN, 1]"])
    def test_factorial_array_corner_cases(self, content, dim, exact, dtype):
        if dtype == np.int64 and any(np.isnan(x) for x in content):
            pytest.skip("impossible combination")
        # np.array(x, ndim=0) will not be 0-dim. unless x is too
        content = content if (dim > 0 or len(content) != 1) else content[0]
        n = np.array(content, ndmin=dim, dtype=dtype)
        result = None
        if not content:
            result = special.factorial(n, exact=exact)
        elif not (np.issubdtype(n.dtype, np.integer)
                  or np.issubdtype(n.dtype, np.floating)):
            with pytest.raises(ValueError, match="Unsupported datatype*"):
                special.factorial(n, exact=exact)
        elif (exact and not np.issubdtype(n.dtype, np.integer) and n.size and
              np.allclose(n[~np.isnan(n)], n[~np.isnan(n)].astype(np.int64))):
            # using integers but in array with wrong dtype (e.g. due to NaNs)
            with pytest.warns(DeprecationWarning, match="Non-integer array.*"):
                result = special.factorial(n, exact=exact)
                # expected dtype is integer, unless there are NaNs
                if np.any(np.isnan(n)):
                    dtype = np.dtype(np.float64)
                else:
                    dtype = np.dtype(int)
        elif exact and not np.issubdtype(n.dtype, np.integer):
            with pytest.raises(ValueError, match="factorial with exact=.*"):
                special.factorial(n, exact=exact)
        else:
            # no error
            result = special.factorial(n, exact=exact)

        # assert_equal does not distinguish scalars and 0-dim arrays of the same value,
        # see https://github.com/numpy/numpy/issues/24050
        def assert_really_equal(x, y):
            assert type(x) == type(y), f"types not equal: {type(x)}, {type(y)}"
            assert_equal(x, y)

        if result is not None:
            # expected result is empty if and only if n is empty,
            # and has the same dtype & dimension as n
            with suppress_warnings() as sup:
                sup.filter(DeprecationWarning)
                # keep 0-dim.; otherwise n.ravel().ndim==1, even if n.ndim==0
                n_flat = n.ravel() if n.ndim else n
                r = special.factorial(n_flat, exact=exact) if n.size else []
            expected = np.array(r, ndmin=dim, dtype=dtype)
            assert_really_equal(result, expected)

    @pytest.mark.parametrize("exact", [True, False])
    @pytest.mark.parametrize("n", [1, 1.1, 2 + 2j, np.nan, None],
                             ids=["1", "1.1", "2+2j", "NaN", "None"])
    def test_factorial_scalar_corner_cases(self, n, exact):
        if (n is None or n is np.nan or np.issubdtype(type(n), np.integer)
                or np.issubdtype(type(n), np.floating)):
            # no error
            result = special.factorial(n, exact=exact)
            exp = np.nan if n is np.nan or n is None else special.factorial(n)
            assert_equal(result, exp)
        else:
            with pytest.raises(ValueError, match="Unsupported datatype*"):
                special.factorial(n, exact=exact)

    # use odd increment to make sure both odd & even numbers are tested!
    @pytest.mark.parametrize('n', range(30, 180, 11))
    def test_factorial2_accuracy(self, n):
        # Compare exact=True vs False, i.e. that the accuracy of the
        # approximation is better than the specified tolerance.

        rtol = 2e-14 if sys.platform == 'win32' else 1e-15
        # need to cast exact result to float due to numpy/numpy#21220
        assert_allclose(float(special.factorial2(n, exact=True)),
                        special.factorial2(n, exact=False), rtol=rtol)
        assert_allclose(special.factorial2([n], exact=True).astype(float),
                        special.factorial2([n], exact=False), rtol=rtol)

    @pytest.mark.parametrize('n',
                             list(range(0, 22)) + list(range(30, 180, 11)))
    def test_factorial2_int_reference(self, n):
        # Compare all with correct value

        # Cannot use np.product due to overflow
        correct = functools.reduce(operator.mul, list(range(n, 0, -2)), 1)

        assert_array_equal(correct, special.factorial2(n, True))
        assert_array_equal(correct, special.factorial2([n], True)[0])

        assert_allclose(float(correct), special.factorial2(n, False))
        assert_allclose(float(correct), special.factorial2([n], False)[0])

    @pytest.mark.parametrize("dtype", [np.int64, np.float64,
                                       np.complex128, object])
    @pytest.mark.parametrize("exact", [True, False])
    @pytest.mark.parametrize("dim", range(0, 5))
    # test empty & non-empty arrays, with nans and mixed
    @pytest.mark.parametrize("content", [[], [1], [np.nan], [np.nan, 1]],
                             ids=["[]", "[1]", "[NaN]", "[NaN, 1]"])
    def test_factorial2_array_corner_cases(self, content, dim, exact, dtype):
        if dtype == np.int64 and any(np.isnan(x) for x in content):
            pytest.skip("impossible combination")
        # np.array(x, ndim=0) will not be 0-dim. unless x is too
        content = content if (dim > 0 or len(content) != 1) else content[0]
        n = np.array(content, ndmin=dim, dtype=dtype)
        if np.issubdtype(n.dtype, np.integer) or (not content):
            # no error
            result = special.factorial2(n, exact=exact)
            # expected result is identical to n for exact=True resp. empty
            # arrays (assert_allclose chokes on object), otherwise up to tol
            func = assert_equal if exact or (not content) else assert_allclose
            func(result, n)
        else:
            with pytest.raises(ValueError, match="factorial2 does not*"):
                special.factorial2(n, 3)

    @pytest.mark.parametrize("exact", [True, False])
    @pytest.mark.parametrize("n", [1, 1.1, 2 + 2j, np.nan, None],
                             ids=["1", "1.1", "2+2j", "NaN", "None"])
    def test_factorial2_scalar_corner_cases(self, n, exact):
        if n is None or n is np.nan or np.issubdtype(type(n), np.integer):
            # no error
            result = special.factorial2(n, exact=exact)
            exp = np.nan if n is np.nan or n is None else special.factorial(n)
            assert_equal(result, exp)
        else:
            with pytest.raises(ValueError, match="factorial2 does not*"):
                special.factorial2(n, exact=exact)

    @pytest.mark.parametrize('k', list(range(1, 5)) + [10, 20])
    @pytest.mark.parametrize('n',
                             list(range(0, 22)) + list(range(22, 100, 11)))
    def test_factorialk_int_reference(self, n, k):
        # Compare all with correct value

        # Would be nice to use np.product here, but that's
        # broken on windows, see numpy/numpy#21219
        correct = functools.reduce(operator.mul, list(range(n, 0, -k)), 1)

        assert_array_equal(correct, special.factorialk(n, k, True))
        assert_array_equal(correct, special.factorialk([n], k, True)[0])

        # exact=False not yet supported
        # assert_allclose(float(correct), special.factorialk(n, k, False))
        # assert_allclose(float(correct), special.factorialk([n], k, False)[0])

    @pytest.mark.parametrize("dtype", [np.int64, np.float64,
                                       np.complex128, object])
    @pytest.mark.parametrize("dim", range(0, 5))
    # test empty & non-empty arrays, with nans and mixed
    @pytest.mark.parametrize("content", [[], [1], [np.nan], [np.nan, 1]],
                             ids=["[]", "[1]", "[NaN]", "[NaN, 1]"])
    def test_factorialk_array_corner_cases(self, content, dim, dtype):
        if dtype == np.int64 and any(np.isnan(x) for x in content):
            pytest.skip("impossible combination")
        # np.array(x, ndim=0) will not be 0-dim. unless x is too
        content = content if (dim > 0 or len(content) != 1) else content[0]
        n = np.array(content, ndmin=dim, dtype=dtype)
        if np.issubdtype(n.dtype, np.integer) or (not content):
            # no error; expected result is identical to n
            assert_equal(special.factorialk(n, 3), n)
        else:
            with pytest.raises(ValueError, match="factorialk does not*"):
                special.factorialk(n, 3)

    @pytest.mark.parametrize("exact", [True, False])
    @pytest.mark.parametrize("k", range(1, 5))
    @pytest.mark.parametrize("n", [1, 1.1, 2 + 2j, np.nan, None],
                             ids=["1", "1.1", "2+2j", "NaN", "None"])
    def test_factorialk_scalar_corner_cases(self, n, k, exact):
        if not exact:
            with pytest.raises(NotImplementedError):
                special.factorialk(n, k=k, exact=exact)
        elif n is None or n is np.nan or np.issubdtype(type(n), np.integer):
            # no error
            result = special.factorial2(n, exact=exact)
            nan_cond = n is np.nan or n is None
            expected = np.nan if nan_cond else special.factorialk(n, k=k)
            assert_equal(result, expected)
        else:
            with pytest.raises(ValueError, match="factorialk does not*"):
                special.factorialk(n, k=k, exact=exact)

    @pytest.mark.parametrize("k", [0, 1.1, np.nan, "1"])
    def test_factorialk_raises_k(self, k):
        with pytest.raises(ValueError, match="k must be a positive integer*"):
            special.factorialk(1, k)

    @pytest.mark.parametrize("k", range(1, 12))
    def test_factorialk_dtype(self, k):
        if k in _FACTORIALK_LIMITS_64BITS.keys():
            n = np.array([_FACTORIALK_LIMITS_32BITS[k]])
            assert_equal(special.factorialk(n, k).dtype, np_long)
            assert_equal(special.factorialk(n + 1, k).dtype, np.int64)
            # assert maximality of limits for given dtype
            assert special.factorialk(n + 1, k) > np.iinfo(np.int32).max

            n = np.array([_FACTORIALK_LIMITS_64BITS[k]])
            assert_equal(special.factorialk(n, k).dtype, np.int64)
            assert_equal(special.factorialk(n + 1, k).dtype, object)
            assert special.factorialk(n + 1, k) > np.iinfo(np.int64).max
        else:
            # for k >= 10, we always return object
            assert_equal(special.factorialk(np.array([1]), k).dtype, object)

    def test_factorial_mixed_nan_inputs(self):
        x = np.array([np.nan, 1, 2, 3, np.nan])
        expected = np.array([np.nan, 1, 2, 6, np.nan])
        assert_equal(special.factorial(x, exact=False), expected)
        with pytest.warns(DeprecationWarning, match=r"Non-integer array.*"):
            assert_equal(special.factorial(x, exact=True), expected)


class TestFresnel:
    @pytest.mark.parametrize("z, s, c", [
        # some positive value
        (.5, 0.064732432859999287, 0.49234422587144644),
        (.5 + .0j, 0.064732432859999287, 0.49234422587144644),
        # negative half annulus
        # https://github.com/scipy/scipy/issues/12309
        # Reference values can be reproduced with
        # https://www.wolframalpha.com/input/?i=FresnelS%5B-2.0+%2B+0.1i%5D
        # https://www.wolframalpha.com/input/?i=FresnelC%5B-2.0+%2B+0.1i%5D
        (
            -2.0 + 0.1j,
            -0.3109538687728942-0.0005870728836383176j,
            -0.4879956866358554+0.10670801832903172j
        ),
        (
            -0.1 - 1.5j,
            -0.03918309471866977+0.7197508454568574j,
            0.09605692502968956-0.43625191013617465j
        ),
        # a different algorithm kicks in for "large" values, i.e., |z| >= 4.5,
        # make sure to test both float and complex values; a different
        # algorithm is used
        (6.0, 0.44696076, 0.49953147),
        (6.0 + 0.0j, 0.44696076, 0.49953147),
        (6.0j, -0.44696076j, 0.49953147j),
        (-6.0 + 0.0j, -0.44696076, -0.49953147),
        (-6.0j, 0.44696076j, -0.49953147j),
        # inf
        (np.inf, 0.5, 0.5),
        (-np.inf, -0.5, -0.5),
    ])
    def test_fresnel_values(self, z, s, c):
        frs = array(special.fresnel(z))
        assert_array_almost_equal(frs, array([s, c]), 8)

    # values from pg 329  Table 7.11 of A & S
    #  slightly corrected in 4th decimal place
    def test_fresnel_zeros(self):
        szo, czo = special.fresnel_zeros(5)
        assert_array_almost_equal(szo,
                                  array([2.0093+0.2885j,
                                          2.8335+0.2443j,
                                          3.4675+0.2185j,
                                          4.0026+0.2009j,
                                          4.4742+0.1877j]),3)
        assert_array_almost_equal(czo,
                                  array([1.7437+0.3057j,
                                          2.6515+0.2529j,
                                          3.3204+0.2240j,
                                          3.8757+0.2047j,
                                          4.3611+0.1907j]),3)
        vals1 = special.fresnel(szo)[0]
        vals2 = special.fresnel(czo)[1]
        assert_array_almost_equal(vals1,0,14)
        assert_array_almost_equal(vals2,0,14)

    def test_fresnelc_zeros(self):
        szo, czo = special.fresnel_zeros(6)
        frc = special.fresnelc_zeros(6)
        assert_array_almost_equal(frc,czo,12)

    def test_fresnels_zeros(self):
        szo, czo = special.fresnel_zeros(5)
        frs = special.fresnels_zeros(5)
        assert_array_almost_equal(frs,szo,12)


class TestGamma:
    def test_gamma(self):
        gam = special.gamma(5)
        assert_equal(gam,24.0)

    def test_gammaln(self):
        gamln = special.gammaln(3)
        lngam = log(special.gamma(3))
        assert_almost_equal(gamln,lngam,8)

    def test_gammainccinv(self):
        gccinv = special.gammainccinv(.5,.5)
        gcinv = special.gammaincinv(.5,.5)
        assert_almost_equal(gccinv,gcinv,8)

    @with_special_errors
    def test_gammaincinv(self):
        y = special.gammaincinv(.4,.4)
        x = special.gammainc(.4,y)
        assert_almost_equal(x,0.4,1)
        y = special.gammainc(10, 0.05)
        x = special.gammaincinv(10, 2.5715803516000736e-20)
        assert_almost_equal(0.05, x, decimal=10)
        assert_almost_equal(y, 2.5715803516000736e-20, decimal=10)
        x = special.gammaincinv(50, 8.20754777388471303050299243573393e-18)
        assert_almost_equal(11.0, x, decimal=10)

    @with_special_errors
    def test_975(self):
        # Regression test for ticket #975 -- switch point in algorithm
        # check that things work OK at the point, immediately next floats
        # around it, and a bit further away
        pts = [0.25,
               np.nextafter(0.25, 0), 0.25 - 1e-12,
               np.nextafter(0.25, 1), 0.25 + 1e-12]
        for xp in pts:
            y = special.gammaincinv(.4, xp)
            x = special.gammainc(0.4, y)
            assert_allclose(x, xp, rtol=1e-12)

    def test_rgamma(self):
        rgam = special.rgamma(8)
        rlgam = 1/special.gamma(8)
        assert_almost_equal(rgam,rlgam,8)

    def test_infinity(self):
        assert_(np.isinf(special.gamma(-1)))
        assert_equal(special.rgamma(-1), 0)


class TestHankel:

    def test_negv1(self):
        assert_almost_equal(special.hankel1(-3,2), -special.hankel1(3,2), 14)

    def test_hankel1(self):
        hank1 = special.hankel1(1,.1)
        hankrl = (special.jv(1,.1) + special.yv(1,.1)*1j)
        assert_almost_equal(hank1,hankrl,8)

    def test_negv1e(self):
        assert_almost_equal(special.hankel1e(-3,2), -special.hankel1e(3,2), 14)

    def test_hankel1e(self):
        hank1e = special.hankel1e(1,.1)
        hankrle = special.hankel1(1,.1)*exp(-.1j)
        assert_almost_equal(hank1e,hankrle,8)

    def test_negv2(self):
        assert_almost_equal(special.hankel2(-3,2), -special.hankel2(3,2), 14)

    def test_hankel2(self):
        hank2 = special.hankel2(1,.1)
        hankrl2 = (special.jv(1,.1) - special.yv(1,.1)*1j)
        assert_almost_equal(hank2,hankrl2,8)

    def test_neg2e(self):
        assert_almost_equal(special.hankel2e(-3,2), -special.hankel2e(3,2), 14)

    def test_hankl2e(self):
        hank2e = special.hankel2e(1,.1)
        hankrl2e = special.hankel2e(1,.1)
        assert_almost_equal(hank2e,hankrl2e,8)


class TestHyper:
    def test_h1vp(self):
        h1 = special.h1vp(1,.1)
        h1real = (special.jvp(1,.1) + special.yvp(1,.1)*1j)
        assert_almost_equal(h1,h1real,8)

    def test_h2vp(self):
        h2 = special.h2vp(1,.1)
        h2real = (special.jvp(1,.1) - special.yvp(1,.1)*1j)
        assert_almost_equal(h2,h2real,8)

    def test_hyp0f1(self):
        # scalar input
        assert_allclose(special.hyp0f1(2.5, 0.5), 1.21482702689997, rtol=1e-12)
        assert_allclose(special.hyp0f1(2.5, 0), 1.0, rtol=1e-15)

        # float input, expected values match mpmath
        x = special.hyp0f1(3.0, [-1.5, -1, 0, 1, 1.5])
        expected = np.array([0.58493659229143, 0.70566805723127, 1.0,
                             1.37789689539747, 1.60373685288480])
        assert_allclose(x, expected, rtol=1e-12)

        # complex input
        x = special.hyp0f1(3.0, np.array([-1.5, -1, 0, 1, 1.5]) + 0.j)
        assert_allclose(x, expected.astype(complex), rtol=1e-12)

        # test broadcasting
        x1 = [0.5, 1.5, 2.5]
        x2 = [0, 1, 0.5]
        x = special.hyp0f1(x1, x2)
        expected = [1.0, 1.8134302039235093, 1.21482702689997]
        assert_allclose(x, expected, rtol=1e-12)
        x = special.hyp0f1(np.vstack([x1] * 2), x2)
        assert_allclose(x, np.vstack([expected] * 2), rtol=1e-12)
        assert_raises(ValueError, special.hyp0f1,
                      np.vstack([x1] * 3), [0, 1])

    def test_hyp0f1_gh5764(self):
        # Just checks the point that failed; there's a more systematic
        # test in test_mpmath
        res = special.hyp0f1(0.8, 0.5 + 0.5*1J)
        # The expected value was generated using mpmath
        assert_almost_equal(res, 1.6139719776441115 + 1J*0.80893054061790665)

    def test_hyp1f1(self):
        hyp1 = special.hyp1f1(.1,.1,.3)
        assert_almost_equal(hyp1, 1.3498588075760032,7)

        # test contributed by Moritz Deger (2008-05-29)
        # https://github.com/scipy/scipy/issues/1186 (Trac #659)

        # reference data obtained from mathematica [ a, b, x, m(a,b,x)]:
        # produced with test_hyp1f1.nb
        ref_data = array([
            [-8.38132975e+00, -1.28436461e+01, -2.91081397e+01, 1.04178330e+04],
            [2.91076882e+00, -6.35234333e+00, -1.27083993e+01, 6.68132725e+00],
            [-1.42938258e+01, 1.80869131e-01, 1.90038728e+01, 1.01385897e+05],
            [5.84069088e+00, 1.33187908e+01, 2.91290106e+01, 1.59469411e+08],
            [-2.70433202e+01, -1.16274873e+01, -2.89582384e+01, 1.39900152e+24],
            [4.26344966e+00, -2.32701773e+01, 1.91635759e+01, 6.13816915e+21],
            [1.20514340e+01, -3.40260240e+00, 7.26832235e+00, 1.17696112e+13],
            [2.77372955e+01, -1.99424687e+00, 3.61332246e+00, 3.07419615e+13],
            [1.50310939e+01, -2.91198675e+01, -1.53581080e+01, -3.79166033e+02],
            [1.43995827e+01, 9.84311196e+00, 1.93204553e+01, 2.55836264e+10],
            [-4.08759686e+00, 1.34437025e+01, -1.42072843e+01, 1.70778449e+01],
            [8.05595738e+00, -1.31019838e+01, 1.52180721e+01, 3.06233294e+21],
            [1.81815804e+01, -1.42908793e+01, 9.57868793e+00, -2.84771348e+20],
            [-2.49671396e+01, 1.25082843e+01, -1.71562286e+01, 2.36290426e+07],
            [2.67277673e+01, 1.70315414e+01, 6.12701450e+00, 7.77917232e+03],
            [2.49565476e+01, 2.91694684e+01, 6.29622660e+00, 2.35300027e+02],
            [6.11924542e+00, -1.59943768e+00, 9.57009289e+00, 1.32906326e+11],
            [-1.47863653e+01, 2.41691301e+01, -1.89981821e+01, 2.73064953e+03],
            [2.24070483e+01, -2.93647433e+00, 8.19281432e+00, -6.42000372e+17],
            [8.04042600e-01, 1.82710085e+01, -1.97814534e+01, 5.48372441e-01],
            [1.39590390e+01, 1.97318686e+01, 2.37606635e+00, 5.51923681e+00],
            [-4.66640483e+00, -2.00237930e+01, 7.40365095e+00, 4.50310752e+00],
            [2.76821999e+01, -6.36563968e+00, 1.11533984e+01, -9.28725179e+23],
            [-2.56764457e+01, 1.24544906e+00, 1.06407572e+01, 1.25922076e+01],
            [3.20447808e+00, 1.30874383e+01, 2.26098014e+01, 2.03202059e+04],
            [-1.24809647e+01, 4.15137113e+00, -2.92265700e+01, 2.39621411e+08],
            [2.14778108e+01, -2.35162960e+00, -1.13758664e+01, 4.46882152e-01],
            [-9.85469168e+00, -3.28157680e+00, 1.67447548e+01, -1.07342390e+07],
            [1.08122310e+01, -2.47353236e+01, -1.15622349e+01, -2.91733796e+03],
            [-2.67933347e+01, -3.39100709e+00, 2.56006986e+01, -5.29275382e+09],
            [-8.60066776e+00, -8.02200924e+00, 1.07231926e+01, 1.33548320e+06],
            [-1.01724238e-01, -1.18479709e+01, -2.55407104e+01, 1.55436570e+00],
            [-3.93356771e+00, 2.11106818e+01, -2.57598485e+01, 2.13467840e+01],
            [3.74750503e+00, 1.55687633e+01, -2.92841720e+01, 1.43873509e-02],
            [6.99726781e+00, 2.69855571e+01, -1.63707771e+01, 3.08098673e-02],
            [-2.31996011e+01, 3.47631054e+00, 9.75119815e-01, 1.79971073e-02],
            [2.38951044e+01, -2.91460190e+01, -2.50774708e+00, 9.56934814e+00],
            [1.52730825e+01, 5.77062507e+00, 1.21922003e+01, 1.32345307e+09],
            [1.74673917e+01, 1.89723426e+01, 4.94903250e+00, 9.90859484e+01],
            [1.88971241e+01, 2.86255413e+01, 5.52360109e-01, 1.44165360e+00],
            [1.02002319e+01, -1.66855152e+01, -2.55426235e+01, 6.56481554e+02],
            [-1.79474153e+01, 1.22210200e+01, -1.84058212e+01, 8.24041812e+05],
            [-1.36147103e+01, 1.32365492e+00, -7.22375200e+00, 9.92446491e+05],
            [7.57407832e+00, 2.59738234e+01, -1.34139168e+01, 3.64037761e-02],
            [2.21110169e+00, 1.28012666e+01, 1.62529102e+01, 1.33433085e+02],
            [-2.64297569e+01, -1.63176658e+01, -1.11642006e+01, -2.44797251e+13],
            [-2.46622944e+01, -3.02147372e+00, 8.29159315e+00, -3.21799070e+05],
            [-1.37215095e+01, -1.96680183e+01, 2.91940118e+01, 3.21457520e+12],
            [-5.45566105e+00, 2.81292086e+01, 1.72548215e-01, 9.66973000e-01],
            [-1.55751298e+00, -8.65703373e+00, 2.68622026e+01, -3.17190834e+16],
            [2.45393609e+01, -2.70571903e+01, 1.96815505e+01, 1.80708004e+37],
            [5.77482829e+00, 1.53203143e+01, 2.50534322e+01, 1.14304242e+06],
            [-1.02626819e+01, 2.36887658e+01, -2.32152102e+01, 7.28965646e+02],
            [-1.30833446e+00, -1.28310210e+01, 1.87275544e+01, -9.33487904e+12],
            [5.83024676e+00, -1.49279672e+01, 2.44957538e+01, -7.61083070e+27],
            [-2.03130747e+01, 2.59641715e+01, -2.06174328e+01, 4.54744859e+04],
            [1.97684551e+01, -2.21410519e+01, -2.26728740e+01, 3.53113026e+06],
            [2.73673444e+01, 2.64491725e+01, 1.57599882e+01, 1.07385118e+07],
            [5.73287971e+00, 1.21111904e+01, 1.33080171e+01, 2.63220467e+03],
            [-2.82751072e+01, 2.08605881e+01, 9.09838900e+00, -6.60957033e-07],
            [1.87270691e+01, -1.74437016e+01, 1.52413599e+01, 6.59572851e+27],
            [6.60681457e+00, -2.69449855e+00, 9.78972047e+00, -2.38587870e+12],
            [1.20895561e+01, -2.51355765e+01, 2.30096101e+01, 7.58739886e+32],
            [-2.44682278e+01, 2.10673441e+01, -1.36705538e+01, 4.54213550e+04],
            [-4.50665152e+00, 3.72292059e+00, -4.83403707e+00, 2.68938214e+01],
            [-7.46540049e+00, -1.08422222e+01, -1.72203805e+01, -2.09402162e+02],
            [-2.00307551e+01, -7.50604431e+00, -2.78640020e+01, 4.15985444e+19],
            [1.99890876e+01, 2.20677419e+01, -2.51301778e+01, 1.23840297e-09],
            [2.03183823e+01, -7.66942559e+00, 2.10340070e+01, 1.46285095e+31],
            [-2.90315825e+00, -2.55785967e+01, -9.58779316e+00, 2.65714264e-01],
            [2.73960829e+01, -1.80097203e+01, -2.03070131e+00, 2.52908999e+02],
            [-2.11708058e+01, -2.70304032e+01, 2.48257944e+01, 3.09027527e+08],
            [2.21959758e+01, 4.00258675e+00, -1.62853977e+01, -9.16280090e-09],
            [1.61661840e+01, -2.26845150e+01, 2.17226940e+01, -8.24774394e+33],
            [-3.35030306e+00, 1.32670581e+00, 9.39711214e+00, -1.47303163e+01],
            [7.23720726e+00, -2.29763909e+01, 2.34709682e+01, -9.20711735e+29],
            [2.71013568e+01, 1.61951087e+01, -7.11388906e-01, 2.98750911e-01],
            [8.40057933e+00, -7.49665220e+00, 2.95587388e+01, 6.59465635e+29],
            [-1.51603423e+01, 1.94032322e+01, -7.60044357e+00, 1.05186941e+02],
            [-8.83788031e+00, -2.72018313e+01, 1.88269907e+00, 1.81687019e+00],
            [-1.87283712e+01, 5.87479570e+00, -1.91210203e+01, 2.52235612e+08],
            [-5.61338513e-01, 2.69490237e+01, 1.16660111e-01, 9.97567783e-01],
            [-5.44354025e+00, -1.26721408e+01, -4.66831036e+00, 1.06660735e-01],
            [-2.18846497e+00, 2.33299566e+01, 9.62564397e+00, 3.03842061e-01],
            [6.65661299e+00, -2.39048713e+01, 1.04191807e+01, 4.73700451e+13],
            [-2.57298921e+01, -2.60811296e+01, 2.74398110e+01, -5.32566307e+11],
            [-1.11431826e+01, -1.59420160e+01, -1.84880553e+01, -1.01514747e+02],
            [6.50301931e+00, 2.59859051e+01, -2.33270137e+01, 1.22760500e-02],
            [-1.94987891e+01, -2.62123262e+01, 3.90323225e+00, 1.71658894e+01],
            [7.26164601e+00, -1.41469402e+01, 2.81499763e+01, -2.50068329e+31],
            [-1.52424040e+01, 2.99719005e+01, -2.85753678e+01, 1.31906693e+04],
            [5.24149291e+00, -1.72807223e+01, 2.22129493e+01, 2.50748475e+25],
            [3.63207230e-01, -9.54120862e-02, -2.83874044e+01, 9.43854939e-01],
            [-2.11326457e+00, -1.25707023e+01, 1.17172130e+00, 1.20812698e+00],
            [2.48513582e+00, 1.03652647e+01, -1.84625148e+01, 6.47910997e-02],
            [2.65395942e+01, 2.74794672e+01, 1.29413428e+01, 2.89306132e+05],
            [-9.49445460e+00, 1.59930921e+01, -1.49596331e+01, 3.27574841e+02],
            [-5.89173945e+00, 9.96742426e+00, 2.60318889e+01, -3.15842908e-01],
            [-1.15387239e+01, -2.21433107e+01, -2.17686413e+01, 1.56724718e-01],
            [-5.30592244e+00, -2.42752190e+01, 1.29734035e+00, 1.31985534e+00]
        ])

        for a,b,c,expected in ref_data:
            result = special.hyp1f1(a,b,c)
            assert_(abs(expected - result)/expected < 1e-4)

    def test_hyp1f1_gh2957(self):
        hyp1 = special.hyp1f1(0.5, 1.5, -709.7827128933)
        hyp2 = special.hyp1f1(0.5, 1.5, -709.7827128934)
        assert_almost_equal(hyp1, hyp2, 12)

    def test_hyp1f1_gh2282(self):
        hyp = special.hyp1f1(0.5, 1.5, -1000)
        assert_almost_equal(hyp, 0.028024956081989643, 12)

    def test_hyp2f1(self):
        # a collection of special cases taken from AMS 55
        values = [
            [0.5, 1, 1.5, 0.2**2, 0.5/0.2*log((1+0.2)/(1-0.2))],
            [0.5, 1, 1.5, -0.2**2, 1./0.2*arctan(0.2)],
            [1, 1, 2, 0.2, -1/0.2*log(1-0.2)],
            [3, 3.5, 1.5, 0.2**2, 0.5/0.2/(-5)*((1+0.2)**(-5)-(1-0.2)**(-5))],
            [-3, 3, 0.5, sin(0.2)**2, cos(2*3*0.2)],
            [3, 4, 8, 1,
             special.gamma(8) * special.gamma(8-4-3)
             / special.gamma(8-3) / special.gamma(8-4)],
            [3, 2, 3-2+1, -1,
             1./2**3*sqrt(pi) * special.gamma(1+3-2)
             / special.gamma(1+0.5*3-2) / special.gamma(0.5+0.5*3)],
            [5, 2, 5-2+1, -1,
             1./2**5*sqrt(pi) * special.gamma(1+5-2)
             / special.gamma(1+0.5*5-2) / special.gamma(0.5+0.5*5)],
            [4, 0.5+4, 1.5-2*4, -1./3,
             (8./9)**(-2*4)*special.gamma(4./3) * special.gamma(1.5-2*4)
             / special.gamma(3./2) / special.gamma(4./3-2*4)],
            # and some others
            # ticket #424
            [1.5, -0.5, 1.0, -10.0, 4.1300097765277476484],
            # negative integer a or b, with c-a-b integer and x > 0.9
            [-2,3,1,0.95,0.715],
            [2,-3,1,0.95,-0.007],
            [-6,3,1,0.95,0.0000810625],
            [2,-5,1,0.95,-0.000029375],
            # huge negative integers
            (10, -900, 10.5, 0.99, 1.91853705796607664803709475658e-24),
            (10, -900, -10.5, 0.99, 3.54279200040355710199058559155e-18),
        ]
        for i, (a, b, c, x, v) in enumerate(values):
            cv = special.hyp2f1(a, b, c, x)
            assert_almost_equal(cv, v, 8, err_msg='test #%d' % i)

    def test_hyperu(self):
        val1 = special.hyperu(1,0.1,100)
        assert_almost_equal(val1,0.0098153,7)
        a,b = [0.3,0.6,1.2,-2.7],[1.5,3.2,-0.4,-3.2]
        a,b = asarray(a), asarray(b)
        z = 0.5
        hypu = special.hyperu(a,b,z)
        hprl = (pi/sin(pi*b))*(special.hyp1f1(a,b,z) /
                               (special.gamma(1+a-b)*special.gamma(b)) -
                               z**(1-b)*special.hyp1f1(1+a-b,2-b,z)
                               / (special.gamma(a)*special.gamma(2-b)))
        assert_array_almost_equal(hypu,hprl,12)

    def test_hyperu_gh2287(self):
        assert_almost_equal(special.hyperu(1, 1.5, 20.2),
                            0.048360918656699191, 12)


class TestBessel:
    def test_itj0y0(self):
        it0 = array(special.itj0y0(.2))
        assert_array_almost_equal(
            it0,
            array([0.19933433254006822, -0.34570883800412566]),
            8,
        )

    def test_it2j0y0(self):
        it2 = array(special.it2j0y0(.2))
        assert_array_almost_equal(
            it2,
            array([0.0049937546274601858, -0.43423067011231614]),
            8,
        )

    def test_negv_iv(self):
        assert_equal(special.iv(3,2), special.iv(-3,2))

    def test_j0(self):
        oz = special.j0(.1)
        ozr = special.jn(0,.1)
        assert_almost_equal(oz,ozr,8)

    def test_j1(self):
        o1 = special.j1(.1)
        o1r = special.jn(1,.1)
        assert_almost_equal(o1,o1r,8)

    def test_jn(self):
        jnnr = special.jn(1,.2)
        assert_almost_equal(jnnr,0.099500832639235995,8)

    def test_negv_jv(self):
        assert_almost_equal(special.jv(-3,2), -special.jv(3,2), 14)

    def test_jv(self):
        values = [[0, 0.1, 0.99750156206604002],
                  [2./3, 1e-8, 0.3239028506761532e-5],
                  [2./3, 1e-10, 0.1503423854873779e-6],
                  [3.1, 1e-10, 0.1711956265409013e-32],
                  [2./3, 4.0, -0.2325440850267039],
                  ]
        for i, (v, x, y) in enumerate(values):
            yc = special.jv(v, x)
            assert_almost_equal(yc, y, 8, err_msg='test #%d' % i)

    def test_negv_jve(self):
        assert_almost_equal(special.jve(-3,2), -special.jve(3,2), 14)

    def test_jve(self):
        jvexp = special.jve(1,.2)
        assert_almost_equal(jvexp,0.099500832639235995,8)
        jvexp1 = special.jve(1,.2+1j)
        z = .2+1j
        jvexpr = special.jv(1,z)*exp(-abs(z.imag))
        assert_almost_equal(jvexp1,jvexpr,8)

    def test_jn_zeros(self):
        jn0 = special.jn_zeros(0,5)
        jn1 = special.jn_zeros(1,5)
        assert_array_almost_equal(jn0,array([2.4048255577,
                                              5.5200781103,
                                              8.6537279129,
                                              11.7915344391,
                                              14.9309177086]),4)
        assert_array_almost_equal(jn1,array([3.83171,
                                              7.01559,
                                              10.17347,
                                              13.32369,
                                              16.47063]),4)

        jn102 = special.jn_zeros(102,5)
        assert_allclose(jn102, array([110.89174935992040343,
                                       117.83464175788308398,
                                       123.70194191713507279,
                                       129.02417238949092824,
                                       134.00114761868422559]), rtol=1e-13)

        jn301 = special.jn_zeros(301,5)
        assert_allclose(jn301, array([313.59097866698830153,
                                       323.21549776096288280,
                                       331.22338738656748796,
                                       338.39676338872084500,
                                       345.03284233056064157]), rtol=1e-13)

    def test_jn_zeros_slow(self):
        jn0 = special.jn_zeros(0, 300)
        assert_allclose(jn0[260-1], 816.02884495068867280, rtol=1e-13)
        assert_allclose(jn0[280-1], 878.86068707124422606, rtol=1e-13)
        assert_allclose(jn0[300-1], 941.69253065317954064, rtol=1e-13)

        jn10 = special.jn_zeros(10, 300)
        assert_allclose(jn10[260-1], 831.67668514305631151, rtol=1e-13)
        assert_allclose(jn10[280-1], 894.51275095371316931, rtol=1e-13)
        assert_allclose(jn10[300-1], 957.34826370866539775, rtol=1e-13)

        jn3010 = special.jn_zeros(3010,5)
        assert_allclose(jn3010, array([3036.86590780927,
                                        3057.06598526482,
                                        3073.66360690272,
                                        3088.37736494778,
                                        3101.86438139042]), rtol=1e-8)

    def test_jnjnp_zeros(self):
        jn = special.jn

        def jnp(n, x):
            return (jn(n-1,x) - jn(n+1,x))/2
        for nt in range(1, 30):
            z, n, m, t = special.jnjnp_zeros(nt)
            for zz, nn, tt in zip(z, n, t):
                if tt == 0:
                    assert_allclose(jn(nn, zz), 0, atol=1e-6)
                elif tt == 1:
                    assert_allclose(jnp(nn, zz), 0, atol=1e-6)
                else:
                    raise AssertionError("Invalid t return for nt=%d" % nt)

    def test_jnp_zeros(self):
        jnp = special.jnp_zeros(1,5)
        assert_array_almost_equal(jnp, array([1.84118,
                                                5.33144,
                                                8.53632,
                                                11.70600,
                                                14.86359]),4)
        jnp = special.jnp_zeros(443,5)
        assert_allclose(special.jvp(443, jnp), 0, atol=1e-15)

    def test_jnyn_zeros(self):
        jnz = special.jnyn_zeros(1,5)
        assert_array_almost_equal(jnz,(array([3.83171,
                                                7.01559,
                                                10.17347,
                                                13.32369,
                                                16.47063]),
                                       array([1.84118,
                                                5.33144,
                                                8.53632,
                                                11.70600,
                                                14.86359]),
                                       array([2.19714,
                                                5.42968,
                                                8.59601,
                                                11.74915,
                                                14.89744]),
                                       array([3.68302,
                                                6.94150,
                                                10.12340,
                                                13.28576,
                                                16.44006])),5)

    def test_jvp(self):
        jvprim = special.jvp(2,2)
        jv0 = (special.jv(1,2)-special.jv(3,2))/2
        assert_almost_equal(jvprim,jv0,10)

    def test_k0(self):
        ozk = special.k0(.1)
        ozkr = special.kv(0,.1)
        assert_almost_equal(ozk,ozkr,8)

    def test_k0e(self):
        ozke = special.k0e(.1)
        ozker = special.kve(0,.1)
        assert_almost_equal(ozke,ozker,8)

    def test_k1(self):
        o1k = special.k1(.1)
        o1kr = special.kv(1,.1)
        assert_almost_equal(o1k,o1kr,8)

    def test_k1e(self):
        o1ke = special.k1e(.1)
        o1ker = special.kve(1,.1)
        assert_almost_equal(o1ke,o1ker,8)

    def test_jacobi(self):
        a = 5*np.random.random() - 1
        b = 5*np.random.random() - 1
        P0 = special.jacobi(0,a,b)
        P1 = special.jacobi(1,a,b)
        P2 = special.jacobi(2,a,b)
        P3 = special.jacobi(3,a,b)

        assert_array_almost_equal(P0.c,[1],13)
        assert_array_almost_equal(P1.c,array([a+b+2,a-b])/2.0,13)
        cp = [(a+b+3)*(a+b+4), 4*(a+b+3)*(a+2), 4*(a+1)*(a+2)]
        p2c = [cp[0],cp[1]-2*cp[0],cp[2]-cp[1]+cp[0]]
        assert_array_almost_equal(P2.c,array(p2c)/8.0,13)
        cp = [(a+b+4)*(a+b+5)*(a+b+6),6*(a+b+4)*(a+b+5)*(a+3),
              12*(a+b+4)*(a+2)*(a+3),8*(a+1)*(a+2)*(a+3)]
        p3c = [cp[0],cp[1]-3*cp[0],cp[2]-2*cp[1]+3*cp[0],cp[3]-cp[2]+cp[1]-cp[0]]
        assert_array_almost_equal(P3.c,array(p3c)/48.0,13)

    def test_kn(self):
        kn1 = special.kn(0,.2)
        assert_almost_equal(kn1,1.7527038555281462,8)

    def test_negv_kv(self):
        assert_equal(special.kv(3.0, 2.2), special.kv(-3.0, 2.2))

    def test_kv0(self):
        kv0 = special.kv(0,.2)
        assert_almost_equal(kv0, 1.7527038555281462, 10)

    def test_kv1(self):
        kv1 = special.kv(1,0.2)
        assert_almost_equal(kv1, 4.775972543220472, 10)

    def test_kv2(self):
        kv2 = special.kv(2,0.2)
        assert_almost_equal(kv2, 49.51242928773287, 10)

    def test_kn_largeorder(self):
        assert_allclose(special.kn(32, 1), 1.7516596664574289e+43)

    def test_kv_largearg(self):
        assert_equal(special.kv(0, 1e19), 0)

    def test_negv_kve(self):
        assert_equal(special.kve(3.0, 2.2), special.kve(-3.0, 2.2))

    def test_kve(self):
        kve1 = special.kve(0,.2)
        kv1 = special.kv(0,.2)*exp(.2)
        assert_almost_equal(kve1,kv1,8)
        z = .2+1j
        kve2 = special.kve(0,z)
        kv2 = special.kv(0,z)*exp(z)
        assert_almost_equal(kve2,kv2,8)

    def test_kvp_v0n1(self):
        z = 2.2
        assert_almost_equal(-special.kv(1,z), special.kvp(0,z, n=1), 10)

    def test_kvp_n1(self):
        v = 3.
        z = 2.2
        xc = -special.kv(v+1,z) + v/z*special.kv(v,z)
        x = special.kvp(v,z, n=1)
        assert_almost_equal(xc, x, 10)   # this function (kvp) is broken

    def test_kvp_n2(self):
        v = 3.
        z = 2.2
        xc = (z**2+v**2-v)/z**2 * special.kv(v,z) + special.kv(v+1,z)/z
        x = special.kvp(v, z, n=2)
        assert_almost_equal(xc, x, 10)

    def test_y0(self):
        oz = special.y0(.1)
        ozr = special.yn(0,.1)
        assert_almost_equal(oz,ozr,8)

    def test_y1(self):
        o1 = special.y1(.1)
        o1r = special.yn(1,.1)
        assert_almost_equal(o1,o1r,8)

    def test_y0_zeros(self):
        yo,ypo = special.y0_zeros(2)
        zo,zpo = special.y0_zeros(2,complex=1)
        all = r_[yo,zo]
        allval = r_[ypo,zpo]
        assert_array_almost_equal(abs(special.yv(0.0,all)),0.0,11)
        assert_array_almost_equal(abs(special.yv(1,all)-allval),0.0,11)

    def test_y1_zeros(self):
        y1 = special.y1_zeros(1)
        assert_array_almost_equal(y1,(array([2.19714]),array([0.52079])),5)

    def test_y1p_zeros(self):
        y1p = special.y1p_zeros(1,complex=1)
        assert_array_almost_equal(
            y1p,
            (array([0.5768+0.904j]), array([-0.7635+0.5892j])),
            3,
        )

    def test_yn_zeros(self):
        an = special.yn_zeros(4,2)
        assert_array_almost_equal(an,array([5.64515, 9.36162]),5)
        an = special.yn_zeros(443,5)
        assert_allclose(an, [450.13573091578090314,
                             463.05692376675001542,
                             472.80651546418663566,
                             481.27353184725625838,
                             488.98055964441374646],
                        rtol=1e-15,)

    def test_ynp_zeros(self):
        ao = special.ynp_zeros(0,2)
        assert_array_almost_equal(ao,array([2.19714133, 5.42968104]),6)
        ao = special.ynp_zeros(43,5)
        assert_allclose(special.yvp(43, ao), 0, atol=1e-15)
        ao = special.ynp_zeros(443,5)
        assert_allclose(special.yvp(443, ao), 0, atol=1e-9)

    def test_ynp_zeros_large_order(self):
        ao = special.ynp_zeros(443,5)
        assert_allclose(special.yvp(443, ao), 0, atol=1e-14)

    def test_yn(self):
        yn2n = special.yn(1,.2)
        assert_almost_equal(yn2n,-3.3238249881118471,8)

    def test_negv_yv(self):
        assert_almost_equal(special.yv(-3,2), -special.yv(3,2), 14)

    def test_yv(self):
        yv2 = special.yv(1,.2)
        assert_almost_equal(yv2,-3.3238249881118471,8)

    def test_negv_yve(self):
        assert_almost_equal(special.yve(-3,2), -special.yve(3,2), 14)

    def test_yve(self):
        yve2 = special.yve(1,.2)
        assert_almost_equal(yve2,-3.3238249881118471,8)
        yve2r = special.yv(1,.2+1j)*exp(-1)
        yve22 = special.yve(1,.2+1j)
        assert_almost_equal(yve22,yve2r,8)

    def test_yvp(self):
        yvpr = (special.yv(1,.2) - special.yv(3,.2))/2.0
        yvp1 = special.yvp(2,.2)
        assert_array_almost_equal(yvp1,yvpr,10)

    def _cephes_vs_amos_points(self):
        """Yield points at which to compare Cephes implementation to AMOS"""
        # check several points, including large-amplitude ones
        v = [-120, -100.3, -20., -10., -1., -.5, 0., 1., 12.49, 120., 301]
        z = [-1300, -11, -10, -1, 1., 10., 200.5, 401., 600.5, 700.6, 1300,
             10003]
        yield from itertools.product(v, z)

        # check half-integers; these are problematic points at least
        # for cephes/iv
        yield from itertools.product(0.5 + arange(-60, 60), [3.5])

    def check_cephes_vs_amos(self, f1, f2, rtol=1e-11, atol=0, skip=None):
        for v, z in self._cephes_vs_amos_points():
            if skip is not None and skip(v, z):
                continue
            c1, c2, c3 = f1(v, z), f1(v,z+0j), f2(int(v), z)
            if np.isinf(c1):
                assert_(np.abs(c2) >= 1e300, (v, z))
            elif np.isnan(c1):
                assert_(c2.imag != 0, (v, z))
            else:
                assert_allclose(c1, c2, err_msg=(v, z), rtol=rtol, atol=atol)
                if v == int(v):
                    assert_allclose(c3, c2, err_msg=(v, z),
                                     rtol=rtol, atol=atol)

    @pytest.mark.xfail(platform.machine() == 'ppc64le',
                       reason="fails on ppc64le")
    def test_jv_cephes_vs_amos(self):
        self.check_cephes_vs_amos(special.jv, special.jn, rtol=1e-10, atol=1e-305)

    @pytest.mark.xfail(platform.machine() == 'ppc64le',
                       reason="fails on ppc64le")
    def test_yv_cephes_vs_amos(self):
        self.check_cephes_vs_amos(special.yv, special.yn, rtol=1e-11, atol=1e-305)

    def test_yv_cephes_vs_amos_only_small_orders(self):
        def skipper(v, z):
            return abs(v) > 50
        self.check_cephes_vs_amos(special.yv, special.yn, rtol=1e-11, atol=1e-305,
                                  skip=skipper)

    def test_iv_cephes_vs_amos(self):
        with np.errstate(all='ignore'):
            self.check_cephes_vs_amos(special.iv, special.iv, rtol=5e-9, atol=1e-305)

    @pytest.mark.slow
    def test_iv_cephes_vs_amos_mass_test(self):
        N = 1000000
        np.random.seed(1)
        v = np.random.pareto(0.5, N) * (-1)**np.random.randint(2, size=N)
        x = np.random.pareto(0.2, N) * (-1)**np.random.randint(2, size=N)

        imsk = (np.random.randint(8, size=N) == 0)
        v[imsk] = v[imsk].astype(int)

        with np.errstate(all='ignore'):
            c1 = special.iv(v, x)
            c2 = special.iv(v, x+0j)

            # deal with differences in the inf and zero cutoffs
            c1[abs(c1) > 1e300] = np.inf
            c2[abs(c2) > 1e300] = np.inf
            c1[abs(c1) < 1e-300] = 0
            c2[abs(c2) < 1e-300] = 0

            dc = abs(c1/c2 - 1)
            dc[np.isnan(dc)] = 0

        k = np.argmax(dc)

        # Most error apparently comes from AMOS and not our implementation;
        # there are some problems near integer orders there
        assert_(
            dc[k] < 2e-7,
            (v[k], x[k], special.iv(v[k], x[k]), special.iv(v[k], x[k]+0j))
        )

    def test_kv_cephes_vs_amos(self):
        self.check_cephes_vs_amos(special.kv, special.kn, rtol=1e-9, atol=1e-305)
        self.check_cephes_vs_amos(special.kv, special.kv, rtol=1e-9, atol=1e-305)

    def test_ticket_623(self):
        assert_allclose(special.jv(3, 4), 0.43017147387562193)
        assert_allclose(special.jv(301, 1300), 0.0183487151115275)
        assert_allclose(special.jv(301, 1296.0682), -0.0224174325312048)

    def test_ticket_853(self):
        """Negative-order Bessels"""
        # cephes
        assert_allclose(special.jv(-1, 1), -0.4400505857449335)
        assert_allclose(special.jv(-2, 1), 0.1149034849319005)
        assert_allclose(special.yv(-1, 1), 0.7812128213002887)
        assert_allclose(special.yv(-2, 1), -1.650682606816255)
        assert_allclose(special.iv(-1, 1), 0.5651591039924851)
        assert_allclose(special.iv(-2, 1), 0.1357476697670383)
        assert_allclose(special.kv(-1, 1), 0.6019072301972347)
        assert_allclose(special.kv(-2, 1), 1.624838898635178)
        assert_allclose(special.jv(-0.5, 1), 0.43109886801837607952)
        assert_allclose(special.yv(-0.5, 1), 0.6713967071418031)
        assert_allclose(special.iv(-0.5, 1), 1.231200214592967)
        assert_allclose(special.kv(-0.5, 1), 0.4610685044478945)
        # amos
        assert_allclose(special.jv(-1, 1+0j), -0.4400505857449335)
        assert_allclose(special.jv(-2, 1+0j), 0.1149034849319005)
        assert_allclose(special.yv(-1, 1+0j), 0.7812128213002887)
        assert_allclose(special.yv(-2, 1+0j), -1.650682606816255)

        assert_allclose(special.iv(-1, 1+0j), 0.5651591039924851)
        assert_allclose(special.iv(-2, 1+0j), 0.1357476697670383)
        assert_allclose(special.kv(-1, 1+0j), 0.6019072301972347)
        assert_allclose(special.kv(-2, 1+0j), 1.624838898635178)

        assert_allclose(special.jv(-0.5, 1+0j), 0.43109886801837607952)
        assert_allclose(special.jv(-0.5, 1+1j), 0.2628946385649065-0.827050182040562j)
        assert_allclose(special.yv(-0.5, 1+0j), 0.6713967071418031)
        assert_allclose(special.yv(-0.5, 1+1j), 0.967901282890131+0.0602046062142816j)

        assert_allclose(special.iv(-0.5, 1+0j), 1.231200214592967)
        assert_allclose(special.iv(-0.5, 1+1j), 0.77070737376928+0.39891821043561j)
        assert_allclose(special.kv(-0.5, 1+0j), 0.4610685044478945)
        assert_allclose(special.kv(-0.5, 1+1j), 0.06868578341999-0.38157825981268j)

        assert_allclose(special.jve(-0.5,1+0.3j), special.jv(-0.5, 1+0.3j)*exp(-0.3))
        assert_allclose(special.yve(-0.5,1+0.3j), special.yv(-0.5, 1+0.3j)*exp(-0.3))
        assert_allclose(special.ive(-0.5,0.3+1j), special.iv(-0.5, 0.3+1j)*exp(-0.3))
        assert_allclose(special.kve(-0.5,0.3+1j), special.kv(-0.5, 0.3+1j)*exp(0.3+1j))

        assert_allclose(
            special.hankel1(-0.5, 1+1j),
            special.jv(-0.5, 1+1j) + 1j*special.yv(-0.5,1+1j)
        )
        assert_allclose(
            special.hankel2(-0.5, 1+1j),
            special.jv(-0.5, 1+1j) - 1j*special.yv(-0.5,1+1j)
        )

    def test_ticket_854(self):
        """Real-valued Bessel domains"""
        assert_(isnan(special.jv(0.5, -1)))
        assert_(isnan(special.iv(0.5, -1)))
        assert_(isnan(special.yv(0.5, -1)))
        assert_(isnan(special.yv(1, -1)))
        assert_(isnan(special.kv(0.5, -1)))
        assert_(isnan(special.kv(1, -1)))
        assert_(isnan(special.jve(0.5, -1)))
        assert_(isnan(special.ive(0.5, -1)))
        assert_(isnan(special.yve(0.5, -1)))
        assert_(isnan(special.yve(1, -1)))
        assert_(isnan(special.kve(0.5, -1)))
        assert_(isnan(special.kve(1, -1)))
        assert_(isnan(special.airye(-1)[0:2]).all(), special.airye(-1))
        assert_(not isnan(special.airye(-1)[2:4]).any(), special.airye(-1))

    def test_gh_7909(self):
        assert_(special.kv(1.5, 0) == np.inf)
        assert_(special.kve(1.5, 0) == np.inf)

    def test_ticket_503(self):
        """Real-valued Bessel I overflow"""
        assert_allclose(special.iv(1, 700), 1.528500390233901e302)
        assert_allclose(special.iv(1000, 1120), 1.301564549405821e301)

    def test_iv_hyperg_poles(self):
        assert_allclose(special.iv(-0.5, 1), 1.231200214592967)

    def iv_series(self, v, z, n=200):
        k = arange(0, n).astype(double)
        r = (v+2*k)*log(.5*z) - special.gammaln(k+1) - special.gammaln(v+k+1)
        r[isnan(r)] = inf
        r = exp(r)
        err = abs(r).max() * finfo(double).eps * n + abs(r[-1])*10
        return r.sum(), err

    def test_i0_series(self):
        for z in [1., 10., 200.5]:
            value, err = self.iv_series(0, z)
            assert_allclose(special.i0(z), value, atol=err, err_msg=z)

    def test_i1_series(self):
        for z in [1., 10., 200.5]:
            value, err = self.iv_series(1, z)
            assert_allclose(special.i1(z), value, atol=err, err_msg=z)

    def test_iv_series(self):
        for v in [-20., -10., -1., 0., 1., 12.49, 120.]:
            for z in [1., 10., 200.5, -1+2j]:
                value, err = self.iv_series(v, z)
                assert_allclose(special.iv(v, z), value, atol=err, err_msg=(v, z))

    def test_i0(self):
        values = [[0.0, 1.0],
                  [1e-10, 1.0],
                  [0.1, 0.9071009258],
                  [0.5, 0.6450352706],
                  [1.0, 0.4657596077],
                  [2.5, 0.2700464416],
                  [5.0, 0.1835408126],
                  [20.0, 0.0897803119],
                  ]
        for i, (x, v) in enumerate(values):
            cv = special.i0(x) * exp(-x)
            assert_almost_equal(cv, v, 8, err_msg='test #%d' % i)

    def test_i0e(self):
        oize = special.i0e(.1)
        oizer = special.ive(0,.1)
        assert_almost_equal(oize,oizer,8)

    def test_i1(self):
        values = [[0.0, 0.0],
                  [1e-10, 0.4999999999500000e-10],
                  [0.1, 0.0452984468],
                  [0.5, 0.1564208032],
                  [1.0, 0.2079104154],
                  [5.0, 0.1639722669],
                  [20.0, 0.0875062222],
                  ]
        for i, (x, v) in enumerate(values):
            cv = special.i1(x) * exp(-x)
            assert_almost_equal(cv, v, 8, err_msg='test #%d' % i)

    def test_i1e(self):
        oi1e = special.i1e(.1)
        oi1er = special.ive(1,.1)
        assert_almost_equal(oi1e,oi1er,8)

    def test_iti0k0(self):
        iti0 = array(special.iti0k0(5))
        assert_array_almost_equal(
            iti0,
            array([31.848667776169801, 1.5673873907283657]),
            5,
        )

    def test_it2i0k0(self):
        it2k = special.it2i0k0(.1)
        assert_array_almost_equal(
            it2k,
            array([0.0012503906973464409, 3.3309450354686687]),
            6,
        )

    def test_iv(self):
        iv1 = special.iv(0,.1)*exp(-.1)
        assert_almost_equal(iv1,0.90710092578230106,10)

    def test_negv_ive(self):
        assert_equal(special.ive(3,2), special.ive(-3,2))

    def test_ive(self):
        ive1 = special.ive(0,.1)
        iv1 = special.iv(0,.1)*exp(-.1)
        assert_almost_equal(ive1,iv1,10)

    def test_ivp0(self):
        assert_almost_equal(special.iv(1,2), special.ivp(0,2), 10)

    def test_ivp(self):
        y = (special.iv(0,2) + special.iv(2,2))/2
        x = special.ivp(1,2)
        assert_almost_equal(x,y,10)


class TestLaguerre:
    def test_laguerre(self):
        lag0 = special.laguerre(0)
        lag1 = special.laguerre(1)
        lag2 = special.laguerre(2)
        lag3 = special.laguerre(3)
        lag4 = special.laguerre(4)
        lag5 = special.laguerre(5)
        assert_array_almost_equal(lag0.c,[1],13)
        assert_array_almost_equal(lag1.c,[-1,1],13)
        assert_array_almost_equal(lag2.c,array([1,-4,2])/2.0,13)
        assert_array_almost_equal(lag3.c,array([-1,9,-18,6])/6.0,13)
        assert_array_almost_equal(lag4.c,array([1,-16,72,-96,24])/24.0,13)
        assert_array_almost_equal(lag5.c,array([-1,25,-200,600,-600,120])/120.0,13)

    def test_genlaguerre(self):
        k = 5*np.random.random() - 0.9
        lag0 = special.genlaguerre(0,k)
        lag1 = special.genlaguerre(1,k)
        lag2 = special.genlaguerre(2,k)
        lag3 = special.genlaguerre(3,k)
        assert_equal(lag0.c, [1])
        assert_equal(lag1.c, [-1, k + 1])
        assert_almost_equal(
            lag2.c,
            array([1,-2*(k+2),(k+1.)*(k+2.)])/2.0
        )
        assert_almost_equal(
            lag3.c,
            array([-1,3*(k+3),-3*(k+2)*(k+3),(k+1)*(k+2)*(k+3)])/6.0
        )


# Base polynomials come from Abrahmowitz and Stegan
class TestLegendre:
    def test_legendre(self):
        leg0 = special.legendre(0)
        leg1 = special.legendre(1)
        leg2 = special.legendre(2)
        leg3 = special.legendre(3)
        leg4 = special.legendre(4)
        leg5 = special.legendre(5)
        assert_equal(leg0.c, [1])
        assert_equal(leg1.c, [1,0])
        assert_almost_equal(leg2.c, array([3,0,-1])/2.0, decimal=13)
        assert_almost_equal(leg3.c, array([5,0,-3,0])/2.0)
        assert_almost_equal(leg4.c, array([35,0,-30,0,3])/8.0)
        assert_almost_equal(leg5.c, array([63,0,-70,0,15,0])/8.0)


class TestLambda:
    def test_lmbda(self):
        lam = special.lmbda(1,.1)
        lamr = (
            array([special.jn(0,.1), 2*special.jn(1,.1)/.1]),
            array([special.jvp(0,.1), -2*special.jv(1,.1)/.01 + 2*special.jvp(1,.1)/.1])
        )
        assert_array_almost_equal(lam,lamr,8)


class TestLog1p:
    def test_log1p(self):
        l1p = (special.log1p(10), special.log1p(11), special.log1p(12))
        l1prl = (log(11), log(12), log(13))
        assert_array_almost_equal(l1p,l1prl,8)

    def test_log1pmore(self):
        l1pm = (special.log1p(1), special.log1p(1.1), special.log1p(1.2))
        l1pmrl = (log(2),log(2.1),log(2.2))
        assert_array_almost_equal(l1pm,l1pmrl,8)


class TestLegendreFunctions:
    def test_clpmn(self):
        z = 0.5+0.3j
        clp = special.clpmn(2, 2, z, 3)
        assert_array_almost_equal(clp,
                   (array([[1.0000, z, 0.5*(3*z*z-1)],
                           [0.0000, sqrt(z*z-1), 3*z*sqrt(z*z-1)],
                           [0.0000, 0.0000, 3*(z*z-1)]]),
                    array([[0.0000, 1.0000, 3*z],
                           [0.0000, z/sqrt(z*z-1), 3*(2*z*z-1)/sqrt(z*z-1)],
                           [0.0000, 0.0000, 6*z]])),
                    7)

    def test_clpmn_close_to_real_2(self):
        eps = 1e-10
        m = 1
        n = 3
        x = 0.5
        clp_plus = special.clpmn(m, n, x+1j*eps, 2)[0][m, n]
        clp_minus = special.clpmn(m, n, x-1j*eps, 2)[0][m, n]
        assert_array_almost_equal(array([clp_plus, clp_minus]),
                                  array([special.lpmv(m, n, x),
                                         special.lpmv(m, n, x)]),
                                  7)

    def test_clpmn_close_to_real_3(self):
        eps = 1e-10
        m = 1
        n = 3
        x = 0.5
        clp_plus = special.clpmn(m, n, x+1j*eps, 3)[0][m, n]
        clp_minus = special.clpmn(m, n, x-1j*eps, 3)[0][m, n]
        assert_array_almost_equal(array([clp_plus, clp_minus]),
                                  array([special.lpmv(m, n, x)*np.exp(-0.5j*m*np.pi),
                                         special.lpmv(m, n, x)*np.exp(0.5j*m*np.pi)]),
                                  7)

    def test_clpmn_across_unit_circle(self):
        eps = 1e-7
        m = 1
        n = 1
        x = 1j
        for type in [2, 3]:
            assert_almost_equal(special.clpmn(m, n, x+1j*eps, type)[0][m, n],
                            special.clpmn(m, n, x-1j*eps, type)[0][m, n], 6)

    def test_inf(self):
        for z in (1, -1):
            for n in range(4):
                for m in range(1, n):
                    lp = special.clpmn(m, n, z)
                    assert_(np.isinf(lp[1][1,1:]).all())
                    lp = special.lpmn(m, n, z)
                    assert_(np.isinf(lp[1][1,1:]).all())

    def test_deriv_clpmn(self):
        # data inside and outside of the unit circle
        zvals = [0.5+0.5j, -0.5+0.5j, -0.5-0.5j, 0.5-0.5j,
                 1+1j, -1+1j, -1-1j, 1-1j]
        m = 2
        n = 3
        for type in [2, 3]:
            for z in zvals:
                for h in [1e-3, 1e-3j]:
                    approx_derivative = (special.clpmn(m, n, z+0.5*h, type)[0]
                                         - special.clpmn(m, n, z-0.5*h, type)[0])/h
                    assert_allclose(special.clpmn(m, n, z, type)[1],
                                    approx_derivative,
                                    rtol=1e-4)

    def test_lpmn(self):
        lp = special.lpmn(0,2,.5)
        assert_array_almost_equal(lp,(array([[1.00000,
                                                      0.50000,
                                                      -0.12500]]),
                                      array([[0.00000,
                                                      1.00000,
                                                      1.50000]])),4)

    def test_lpn(self):
        lpnf = special.lpn(2,.5)
        assert_array_almost_equal(lpnf,(array([1.00000,
                                                        0.50000,
                                                        -0.12500]),
                                      array([0.00000,
                                                      1.00000,
                                                      1.50000])),4)

    def test_lpmv(self):
        lp = special.lpmv(0,2,.5)
        assert_almost_equal(lp,-0.125,7)
        lp = special.lpmv(0,40,.001)
        assert_almost_equal(lp,0.1252678976534484,7)

        # XXX: this is outside the domain of the current implementation,
        #      so ensure it returns a NaN rather than a wrong answer.
        with np.errstate(all='ignore'):
            lp = special.lpmv(-1,-1,.001)
        assert_(lp != 0 or np.isnan(lp))

    def test_lqmn(self):
        lqmnf = special.lqmn(0,2,.5)
        lqf = special.lqn(2,.5)
        assert_array_almost_equal(lqmnf[0][0],lqf[0],4)
        assert_array_almost_equal(lqmnf[1][0],lqf[1],4)

    def test_lqmn_gt1(self):
        """algorithm for real arguments changes at 1.0001
           test against analytical result for m=2, n=1
        """
        x0 = 1.0001
        delta = 0.00002
        for x in (x0-delta, x0+delta):
            lq = special.lqmn(2, 1, x)[0][-1, -1]
            expected = 2/(x*x-1)
            assert_almost_equal(lq, expected)

    def test_lqmn_shape(self):
        a, b = special.lqmn(4, 4, 1.1)
        assert_equal(a.shape, (5, 5))
        assert_equal(b.shape, (5, 5))

        a, b = special.lqmn(4, 0, 1.1)
        assert_equal(a.shape, (5, 1))
        assert_equal(b.shape, (5, 1))

    def test_lqn(self):
        lqf = special.lqn(2,.5)
        assert_array_almost_equal(lqf,(array([0.5493, -0.7253, -0.8187]),
                                       array([1.3333, 1.216, -0.8427])),4)


class TestMathieu:

    def test_mathieu_a(self):
        pass

    def test_mathieu_even_coef(self):
        special.mathieu_even_coef(2,5)
        # Q not defined broken and cannot figure out proper reporting order

    def test_mathieu_odd_coef(self):
        # same problem as above
        pass


class TestFresnelIntegral:

    def test_modfresnelp(self):
        pass

    def test_modfresnelm(self):
        pass


class TestOblCvSeq:
    def test_obl_cv_seq(self):
        obl = special.obl_cv_seq(0,3,1)
        assert_array_almost_equal(obl,array([-0.348602,
                                              1.393206,
                                              5.486800,
                                              11.492120]),5)


class TestParabolicCylinder:
    def test_pbdn_seq(self):
        pb = special.pbdn_seq(1,.1)
        assert_array_almost_equal(pb,(array([0.9975,
                                              0.0998]),
                                      array([-0.0499,
                                             0.9925])),4)

    def test_pbdv(self):
        special.pbdv(1,.2)
        1/2*(.2)*special.pbdv(1,.2)[0] - special.pbdv(0,.2)[0]

    def test_pbdv_seq(self):
        pbn = special.pbdn_seq(1,.1)
        pbv = special.pbdv_seq(1,.1)
        assert_array_almost_equal(pbv,(real(pbn[0]),real(pbn[1])),4)

    def test_pbdv_points(self):
        # simple case
        eta = np.linspace(-10, 10, 5)
        z = 2**(eta/2)*np.sqrt(np.pi)/special.gamma(.5-.5*eta)
        assert_allclose(special.pbdv(eta, 0.)[0], z, rtol=1e-14, atol=1e-14)

        # some points
        assert_allclose(special.pbdv(10.34, 20.44)[0], 1.3731383034455e-32, rtol=1e-12)
        assert_allclose(special.pbdv(-9.53, 3.44)[0], 3.166735001119246e-8, rtol=1e-12)

    def test_pbdv_gradient(self):
        x = np.linspace(-4, 4, 8)[:,None]
        eta = np.linspace(-10, 10, 5)[None,:]

        p = special.pbdv(eta, x)
        eps = 1e-7 + 1e-7*abs(x)
        dp = (special.pbdv(eta, x + eps)[0] - special.pbdv(eta, x - eps)[0]) / eps / 2.
        assert_allclose(p[1], dp, rtol=1e-6, atol=1e-6)

    def test_pbvv_gradient(self):
        x = np.linspace(-4, 4, 8)[:,None]
        eta = np.linspace(-10, 10, 5)[None,:]

        p = special.pbvv(eta, x)
        eps = 1e-7 + 1e-7*abs(x)
        dp = (special.pbvv(eta, x + eps)[0] - special.pbvv(eta, x - eps)[0]) / eps / 2.
        assert_allclose(p[1], dp, rtol=1e-6, atol=1e-6)


class TestPolygamma:
    # from Table 6.2 (pg. 271) of A&S
    def test_polygamma(self):
        poly2 = special.polygamma(2,1)
        poly3 = special.polygamma(3,1)
        assert_almost_equal(poly2,-2.4041138063,10)
        assert_almost_equal(poly3,6.4939394023,10)

        # Test polygamma(0, x) == psi(x)
        x = [2, 3, 1.1e14]
        assert_almost_equal(special.polygamma(0, x), special.psi(x))

        # Test broadcasting
        n = [0, 1, 2]
        x = [0.5, 1.5, 2.5]
        expected = [-1.9635100260214238, 0.93480220054467933,
                    -0.23620405164172739]
        assert_almost_equal(special.polygamma(n, x), expected)
        expected = np.vstack([expected]*2)
        assert_almost_equal(special.polygamma(n, np.vstack([x]*2)),
                            expected)
        assert_almost_equal(special.polygamma(np.vstack([n]*2), x),
                            expected)


class TestProCvSeq:
    def test_pro_cv_seq(self):
        prol = special.pro_cv_seq(0,3,1)
        assert_array_almost_equal(prol,array([0.319000,
                                               2.593084,
                                               6.533471,
                                               12.514462]),5)


class TestPsi:
    def test_psi(self):
        ps = special.psi(1)
        assert_almost_equal(ps,-0.57721566490153287,8)


class TestRadian:
    def test_radian(self):
        rad = special.radian(90,0,0)
        assert_almost_equal(rad,pi/2.0,5)

    def test_radianmore(self):
        rad1 = special.radian(90,1,60)
        assert_almost_equal(rad1,pi/2+0.0005816135199345904,5)


class TestRiccati:
    def test_riccati_jn(self):
        N, x = 2, 0.2
        S = np.empty((N, N))
        for n in range(N):
            j = special.spherical_jn(n, x)
            jp = special.spherical_jn(n, x, derivative=True)
            S[0,n] = x*j
            S[1,n] = x*jp + j
        assert_array_almost_equal(S, special.riccati_jn(n, x), 8)

    def test_riccati_yn(self):
        N, x = 2, 0.2
        C = np.empty((N, N))
        for n in range(N):
            y = special.spherical_yn(n, x)
            yp = special.spherical_yn(n, x, derivative=True)
            C[0,n] = x*y
            C[1,n] = x*yp + y
        assert_array_almost_equal(C, special.riccati_yn(n, x), 8)


class TestRound:
    def test_round(self):
        rnd = list(map(int, (special.round(10.1),
                             special.round(10.4),
                             special.round(10.5),
                             special.round(10.6))))

        # Note: According to the documentation, scipy.special.round is
        # supposed to round to the nearest even number if the fractional
        # part is exactly 0.5. On some platforms, this does not appear
        # to work and thus this test may fail. However, this unit test is
        # correctly written.
        rndrl = (10,10,10,11)
        assert_array_equal(rnd,rndrl)


def test_sph_harm():
    # Tests derived from tables in
    # https://en.wikipedia.org/wiki/Table_of_spherical_harmonics
    sh = special.sph_harm
    pi = np.pi
    exp = np.exp
    sqrt = np.sqrt
    sin = np.sin
    cos = np.cos
    assert_array_almost_equal(sh(0,0,0,0),
           0.5/sqrt(pi))
    assert_array_almost_equal(sh(-2,2,0.,pi/4),
           0.25*sqrt(15./(2.*pi)) *
           (sin(pi/4))**2.)
    assert_array_almost_equal(sh(-2,2,0.,pi/2),
           0.25*sqrt(15./(2.*pi)))
    assert_array_almost_equal(sh(2,2,pi,pi/2),
           0.25*sqrt(15/(2.*pi)) *
           exp(0+2.*pi*1j)*sin(pi/2.)**2.)
    assert_array_almost_equal(sh(2,4,pi/4.,pi/3.),
           (3./8.)*sqrt(5./(2.*pi)) *
           exp(0+2.*pi/4.*1j) *
           sin(pi/3.)**2. *
           (7.*cos(pi/3.)**2.-1))
    assert_array_almost_equal(sh(4,4,pi/8.,pi/6.),
           (3./16.)*sqrt(35./(2.*pi)) *
           exp(0+4.*pi/8.*1j)*sin(pi/6.)**4.)


def test_sph_harm_ufunc_loop_selection():
    # see https://github.com/scipy/scipy/issues/4895
    dt = np.dtype(np.complex128)
    assert_equal(special.sph_harm(0, 0, 0, 0).dtype, dt)
    assert_equal(special.sph_harm([0], 0, 0, 0).dtype, dt)
    assert_equal(special.sph_harm(0, [0], 0, 0).dtype, dt)
    assert_equal(special.sph_harm(0, 0, [0], 0).dtype, dt)
    assert_equal(special.sph_harm(0, 0, 0, [0]).dtype, dt)
    assert_equal(special.sph_harm([0], [0], [0], [0]).dtype, dt)


class TestStruve:
    def _series(self, v, z, n=100):
        """Compute Struve function & error estimate from its power series."""
        k = arange(0, n)
        r = (-1)**k * (.5*z)**(2*k+v+1)/special.gamma(k+1.5)/special.gamma(k+v+1.5)
        err = abs(r).max() * finfo(double).eps * n
        return r.sum(), err

    def test_vs_series(self):
        """Check Struve function versus its power series"""
        for v in [-20, -10, -7.99, -3.4, -1, 0, 1, 3.4, 12.49, 16]:
            for z in [1, 10, 19, 21, 30]:
                value, err = self._series(v, z)
                assert_allclose(special.struve(v, z), value, rtol=0, atol=err), (v, z)

    def test_some_values(self):
        assert_allclose(special.struve(-7.99, 21), 0.0467547614113, rtol=1e-7)
        assert_allclose(special.struve(-8.01, 21), 0.0398716951023, rtol=1e-8)
        assert_allclose(special.struve(-3.0, 200), 0.0142134427432, rtol=1e-12)
        assert_allclose(special.struve(-8.0, -41), 0.0192469727846, rtol=1e-11)
        assert_equal(special.struve(-12, -41), -special.struve(-12, 41))
        assert_equal(special.struve(+12, -41), -special.struve(+12, 41))
        assert_equal(special.struve(-11, -41), +special.struve(-11, 41))
        assert_equal(special.struve(+11, -41), +special.struve(+11, 41))

        assert_(isnan(special.struve(-7.1, -1)))
        assert_(isnan(special.struve(-10.1, -1)))

    def test_regression_679(self):
        """Regression test for #679"""
        assert_allclose(special.struve(-1.0, 20 - 1e-8),
                        special.struve(-1.0, 20 + 1e-8))
        assert_allclose(special.struve(-2.0, 20 - 1e-8),
                        special.struve(-2.0, 20 + 1e-8))
        assert_allclose(special.struve(-4.3, 20 - 1e-8),
                        special.struve(-4.3, 20 + 1e-8))


def test_chi2_smalldf():
    assert_almost_equal(special.chdtr(0.6,3), 0.957890536704110)


def test_ch2_inf():
    assert_equal(special.chdtr(0.7,np.inf), 1.0)


def test_chi2c_smalldf():
    assert_almost_equal(special.chdtrc(0.6,3), 1-0.957890536704110)


def test_chi2_inv_smalldf():
    assert_almost_equal(special.chdtri(0.6,1-0.957890536704110), 3)


def test_agm_simple():
    rtol = 1e-13

    # Gauss's constant
    assert_allclose(1/special.agm(1, np.sqrt(2)), 0.834626841674073186,
                    rtol=rtol)

    # These values were computed using Wolfram Alpha, with the
    # function ArithmeticGeometricMean[a, b].
    agm13 = 1.863616783244897
    agm15 = 2.604008190530940
    agm35 = 3.936235503649555
    assert_allclose(special.agm([[1], [3]], [1, 3, 5]),
                    [[1, agm13, agm15],
                     [agm13, 3, agm35]], rtol=rtol)

    # Computed by the iteration formula using mpmath,
    # with mpmath.mp.prec = 1000:
    agm12 = 1.4567910310469068
    assert_allclose(special.agm(1, 2), agm12, rtol=rtol)
    assert_allclose(special.agm(2, 1), agm12, rtol=rtol)
    assert_allclose(special.agm(-1, -2), -agm12, rtol=rtol)
    assert_allclose(special.agm(24, 6), 13.458171481725614, rtol=rtol)
    assert_allclose(special.agm(13, 123456789.5), 11111458.498599306,
                    rtol=rtol)
    assert_allclose(special.agm(1e30, 1), 2.229223055945383e+28, rtol=rtol)
    assert_allclose(special.agm(1e-22, 1), 0.030182566420169886, rtol=rtol)
    assert_allclose(special.agm(1e150, 1e180), 2.229223055945383e+178,
                    rtol=rtol)
    assert_allclose(special.agm(1e180, 1e-150), 2.0634722510162677e+177,
                    rtol=rtol)
    assert_allclose(special.agm(1e-150, 1e-170), 3.3112619670463756e-152,
                    rtol=rtol)
    fi = np.finfo(1.0)
    assert_allclose(special.agm(fi.tiny, fi.max), 1.9892072050015473e+305,
                    rtol=rtol)
    assert_allclose(special.agm(0.75*fi.max, fi.max), 1.564904312298045e+308,
                    rtol=rtol)
    assert_allclose(special.agm(fi.tiny, 3*fi.tiny), 4.1466849866735005e-308,
                    rtol=rtol)

    # zero, nan and inf cases.
    assert_equal(special.agm(0, 0), 0)
    assert_equal(special.agm(99, 0), 0)

    assert_equal(special.agm(-1, 10), np.nan)
    assert_equal(special.agm(0, np.inf), np.nan)
    assert_equal(special.agm(np.inf, 0), np.nan)
    assert_equal(special.agm(0, -np.inf), np.nan)
    assert_equal(special.agm(-np.inf, 0), np.nan)
    assert_equal(special.agm(np.inf, -np.inf), np.nan)
    assert_equal(special.agm(-np.inf, np.inf), np.nan)
    assert_equal(special.agm(1, np.nan), np.nan)
    assert_equal(special.agm(np.nan, -1), np.nan)

    assert_equal(special.agm(1, np.inf), np.inf)
    assert_equal(special.agm(np.inf, 1), np.inf)
    assert_equal(special.agm(-1, -np.inf), -np.inf)
    assert_equal(special.agm(-np.inf, -1), -np.inf)


def test_legacy():
    # Legacy behavior: truncating arguments to integers
    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning, "floating point number truncated to an integer")
        assert_equal(special.expn(1, 0.3), special.expn(1.8, 0.3))
        assert_equal(special.nbdtrc(1, 2, 0.3), special.nbdtrc(1.8, 2.8, 0.3))
        assert_equal(special.nbdtr(1, 2, 0.3), special.nbdtr(1.8, 2.8, 0.3))
        assert_equal(special.nbdtri(1, 2, 0.3), special.nbdtri(1.8, 2.8, 0.3))
        assert_equal(special.pdtri(1, 0.3), special.pdtri(1.8, 0.3))
        assert_equal(special.kn(1, 0.3), special.kn(1.8, 0.3))
        assert_equal(special.yn(1, 0.3), special.yn(1.8, 0.3))
        assert_equal(special.smirnov(1, 0.3), special.smirnov(1.8, 0.3))
        assert_equal(special.smirnovi(1, 0.3), special.smirnovi(1.8, 0.3))


@with_special_errors
def test_error_raising():
    assert_raises(special.SpecialFunctionError, special.iv, 1, 1e99j)


def test_xlogy():
    def xfunc(x, y):
        with np.errstate(invalid='ignore'):
            if x == 0 and not np.isnan(y):
                return x
            else:
                return x*np.log(y)

    z1 = np.asarray([(0,0), (0, np.nan), (0, np.inf), (1.0, 2.0)], dtype=float)
    z2 = np.r_[z1, [(0, 1j), (1, 1j)]]

    w1 = np.vectorize(xfunc)(z1[:,0], z1[:,1])
    assert_func_equal(special.xlogy, w1, z1, rtol=1e-13, atol=1e-13)
    w2 = np.vectorize(xfunc)(z2[:,0], z2[:,1])
    assert_func_equal(special.xlogy, w2, z2, rtol=1e-13, atol=1e-13)


def test_xlog1py():
    def xfunc(x, y):
        with np.errstate(invalid='ignore'):
            if x == 0 and not np.isnan(y):
                return x
            else:
                return x * np.log1p(y)

    z1 = np.asarray([(0,0), (0, np.nan), (0, np.inf), (1.0, 2.0),
                     (1, 1e-30)], dtype=float)
    w1 = np.vectorize(xfunc)(z1[:,0], z1[:,1])
    assert_func_equal(special.xlog1py, w1, z1, rtol=1e-13, atol=1e-13)


def test_entr():
    def xfunc(x):
        if x < 0:
            return -np.inf
        else:
            return -special.xlogy(x, x)
    values = (0, 0.5, 1.0, np.inf)
    signs = [-1, 1]
    arr = []
    for sgn, v in itertools.product(signs, values):
        arr.append(sgn * v)
    z = np.array(arr, dtype=float)
    w = np.vectorize(xfunc, otypes=[np.float64])(z)
    assert_func_equal(special.entr, w, z, rtol=1e-13, atol=1e-13)


def test_kl_div():
    def xfunc(x, y):
        if x < 0 or y < 0 or (y == 0 and x != 0):
            # extension of natural domain to preserve convexity
            return np.inf
        elif np.isposinf(x) or np.isposinf(y):
            # limits within the natural domain
            return np.inf
        elif x == 0:
            return y
        else:
            return special.xlogy(x, x/y) - x + y
    values = (0, 0.5, 1.0)
    signs = [-1, 1]
    arr = []
    for sgna, va, sgnb, vb in itertools.product(signs, values, signs, values):
        arr.append((sgna*va, sgnb*vb))
    z = np.array(arr, dtype=float)
    w = np.vectorize(xfunc, otypes=[np.float64])(z[:,0], z[:,1])
    assert_func_equal(special.kl_div, w, z, rtol=1e-13, atol=1e-13)


def test_rel_entr():
    def xfunc(x, y):
        if x > 0 and y > 0:
            return special.xlogy(x, x/y)
        elif x == 0 and y >= 0:
            return 0
        else:
            return np.inf
    values = (0, 0.5, 1.0)
    signs = [-1, 1]
    arr = []
    for sgna, va, sgnb, vb in itertools.product(signs, values, signs, values):
        arr.append((sgna*va, sgnb*vb))
    z = np.array(arr, dtype=float)
    w = np.vectorize(xfunc, otypes=[np.float64])(z[:,0], z[:,1])
    assert_func_equal(special.rel_entr, w, z, rtol=1e-13, atol=1e-13)


def test_huber():
    assert_equal(special.huber(-1, 1.5), np.inf)
    assert_allclose(special.huber(2, 1.5), 0.5 * np.square(1.5))
    assert_allclose(special.huber(2, 2.5), 2 * (2.5 - 0.5 * 2))

    def xfunc(delta, r):
        if delta < 0:
            return np.inf
        elif np.abs(r) < delta:
            return 0.5 * np.square(r)
        else:
            return delta * (np.abs(r) - 0.5 * delta)

    z = np.random.randn(10, 2)
    w = np.vectorize(xfunc, otypes=[np.float64])(z[:,0], z[:,1])
    assert_func_equal(special.huber, w, z, rtol=1e-13, atol=1e-13)


def test_pseudo_huber():
    def xfunc(delta, r):
        if delta < 0:
            return np.inf
        elif (not delta) or (not r):
            return 0
        else:
            return delta**2 * (np.sqrt(1 + (r/delta)**2) - 1)

    z = np.array(np.random.randn(10, 2).tolist() + [[0, 0.5], [0.5, 0]])
    w = np.vectorize(xfunc, otypes=[np.float64])(z[:,0], z[:,1])
    assert_func_equal(special.pseudo_huber, w, z, rtol=1e-13, atol=1e-13)


def test_pseudo_huber_small_r():
    delta = 1.0
    r = 1e-18
    y = special.pseudo_huber(delta, r)
    # expected computed with mpmath:
    #     import mpmath
    #     mpmath.mp.dps = 200
    #     r = mpmath.mpf(1e-18)
    #     expected = float(mpmath.sqrt(1 + r**2) - 1)
    expected = 5.0000000000000005e-37
    assert_allclose(y, expected, rtol=1e-13)


def test_runtime_warning():
    with pytest.warns(RuntimeWarning,
                      match=r'Too many predicted coefficients'):
        mathieu_odd_coef(1000, 1000)
    with pytest.warns(RuntimeWarning,
                      match=r'Too many predicted coefficients'):
        mathieu_even_coef(1000, 1000)


class TestStirling2:
    table = [
        [1],
        [0, 1],
        [0, 1, 1],
        [0, 1, 3, 1],
        [0, 1, 7, 6, 1],
        [0, 1, 15, 25, 10, 1],
        [0, 1, 31, 90, 65, 15, 1],
        [0, 1, 63, 301, 350, 140, 21, 1],
        [0, 1, 127, 966, 1701, 1050, 266, 28, 1],
        [0, 1, 255, 3025, 7770, 6951, 2646, 462, 36, 1],
        [0, 1, 511, 9330, 34105, 42525, 22827, 5880, 750, 45, 1],
    ]

    @pytest.mark.parametrize("is_exact, comp, kwargs", [
        (True, assert_equal, {}),
        (False, assert_allclose, {'rtol': 1e-12})
    ])
    def test_table_cases(self, is_exact, comp, kwargs):
        for n in range(1, len(self.table)):
            k_values = list(range(n+1))
            row = self.table[n]
            comp(row, stirling2([n], k_values, exact=is_exact), **kwargs)

    @pytest.mark.parametrize("is_exact, comp, kwargs", [
        (True, assert_equal, {}),
        (False, assert_allclose, {'rtol': 1e-12})
    ])
    def test_valid_single_integer(self, is_exact, comp, kwargs):
        comp(stirling2(0, 0, exact=is_exact), self.table[0][0], **kwargs)
        comp(stirling2(4, 2, exact=is_exact), self.table[4][2], **kwargs)
        # a single 2-tuple of integers as arguments must return an int and not
        # an array whereas arrays of single values should return array
        comp(stirling2(5, 3, exact=is_exact), 25, **kwargs)
        comp(stirling2([5], [3], exact=is_exact), [25], **kwargs)

    @pytest.mark.parametrize("is_exact, comp, kwargs", [
        (True, assert_equal, {}),
        (False, assert_allclose, {'rtol': 1e-12})
    ])
    def test_negative_integer(self, is_exact, comp, kwargs):
        # negative integers for n or k arguments return 0
        comp(stirling2(-1, -1, exact=is_exact), 0, **kwargs)
        comp(stirling2(-1, 2, exact=is_exact), 0, **kwargs)
        comp(stirling2(2, -1, exact=is_exact), 0, **kwargs)

    @pytest.mark.parametrize("is_exact, comp, kwargs", [
        (True, assert_equal, {}),
        (False, assert_allclose, {'rtol': 1e-12})
    ])
    def test_array_inputs(self, is_exact, comp, kwargs):
        ans = [self.table[10][3], self.table[10][4]]
        comp(stirling2(asarray([10, 10]),
                               asarray([3, 4]),
                               exact=is_exact),
                     ans)
        comp(stirling2([10, 10],
                               asarray([3, 4]),
                               exact=is_exact),
                     ans)
        comp(stirling2(asarray([10, 10]),
                               [3, 4],
                               exact=is_exact),
                     ans)

    @pytest.mark.parametrize("is_exact, comp, kwargs", [
        (True, assert_equal, {}),
        (False, assert_allclose, {'rtol': 1e-13})
    ])
    def test_mixed_values(self, is_exact, comp, kwargs):
        # negative values-of either n or k-should return 0 for the entry
        ans = [0, 1, 3, 25, 1050, 5880, 9330]
        n = [-1, 0, 3, 5, 8, 10, 10]
        k = [-2, 0, 2, 3, 5, 7, 3]
        comp(stirling2(n, k, exact=is_exact), ans, **kwargs)

    def test_correct_parity(self):
        """Test parity follows well known identity.

        en.wikipedia.org/wiki/Stirling_numbers_of_the_second_kind#Parity
        """
        n, K = 100, np.arange(101)
        assert_equal(
            stirling2(n, K, exact=True) % 2,
            [math.comb(n - (k // 2) - 1, n - k) % 2 for k in K],
        )

    def test_big_numbers(self):
        # via mpmath (bigger than 32bit)
        ans = asarray([48063331393110, 48004081105038305])
        n = [25, 30]
        k = [17, 4]
        assert array_equal(stirling2(n, k, exact=True), ans)
        # bigger than 64 bit
        ans = asarray([2801934359500572414253157841233849412,
                       14245032222277144547280648984426251])
        n = [42, 43]
        k = [17, 23]
        assert array_equal(stirling2(n, k, exact=True), ans)

    @pytest.mark.parametrize("N", [4.5, 3., 4+1j, "12", np.nan])
    @pytest.mark.parametrize("K", [3.5, 3, "2", None])
    @pytest.mark.parametrize("is_exact", [True, False])
    def test_unsupported_input_types(self, N, K, is_exact):
        # object, float, string, complex are not supported and raise TypeError
        with pytest.raises(TypeError):
            stirling2(N, K, exact=is_exact)

    @pytest.mark.parametrize("is_exact", [True, False])
    def test_numpy_array_int_object_dtype(self, is_exact):
        # python integers with arbitrary precision are *not* allowed as
        # object type in numpy arrays are inconsistent from api perspective
        ans = asarray(self.table[4][1:])
        n = asarray([4, 4, 4, 4], dtype=object)
        k = asarray([1, 2, 3, 4], dtype=object)
        with pytest.raises(TypeError):
            array_equal(stirling2(n, k, exact=is_exact), ans)

    @pytest.mark.parametrize("is_exact, comp, kwargs", [
        (True, assert_equal, {}),
        (False, assert_allclose, {'rtol': 1e-13})
    ])
    def test_numpy_array_unsigned_int_dtype(self, is_exact, comp, kwargs):
        # numpy unsigned integers are allowed as dtype in numpy arrays
        ans = asarray(self.table[4][1:])
        n = asarray([4, 4, 4, 4], dtype=np_ulong)
        k = asarray([1, 2, 3, 4], dtype=np_ulong)
        comp(stirling2(n, k, exact=False), ans, **kwargs)

    @pytest.mark.parametrize("is_exact, comp, kwargs", [
        (True, assert_equal, {}),
        (False, assert_allclose, {'rtol': 1e-13})
    ])
    def test_broadcasting_arrays_correctly(self, is_exact, comp, kwargs):
        # broadcasting is handled by stirling2
        # test leading 1s are replicated
        ans = asarray([[1, 15, 25, 10], [1, 7, 6, 1]])  # shape (2,4)
        n = asarray([[5, 5, 5, 5], [4, 4, 4, 4]])  # shape (2,4)
        k = asarray([1, 2, 3, 4])  # shape (4,)
        comp(stirling2(n, k, exact=is_exact), ans, **kwargs)
        # test that dims both mismatch broadcast correctly (5,1) & (6,)
        n = asarray([[4], [4], [4], [4], [4]])
        k = asarray([0, 1, 2, 3, 4, 5])
        ans = asarray([[0, 1, 7, 6, 1, 0] for _ in range(5)])
        comp(stirling2(n, k, exact=False), ans, **kwargs)

    def test_temme_rel_max_error(self):
        # python integers with arbitrary precision are *not* allowed as
        # object type in numpy arrays are inconsistent from api perspective
        x = list(range(51, 101, 5))
        for n in x:
            k_entries = list(range(1, n+1))
            denom = stirling2([n], k_entries, exact=True)
            num = denom - stirling2([n], k_entries, exact=False)
            assert np.max(np.abs(num / denom)) < 2e-5
