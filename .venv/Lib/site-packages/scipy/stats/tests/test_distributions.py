"""
Test functions for stats module
"""
import warnings
import re
import sys
import pickle
from pathlib import Path
import os
import json
import platform

from numpy.testing import (assert_equal, assert_array_equal,
                           assert_almost_equal, assert_array_almost_equal,
                           assert_allclose, assert_, assert_warns,
                           assert_array_less, suppress_warnings, IS_PYPY)
import pytest
from pytest import raises as assert_raises

import numpy
import numpy as np
from numpy import typecodes, array
from numpy.lib.recfunctions import rec_append_fields
from scipy import special
from scipy._lib._util import check_random_state
from scipy.integrate import (IntegrationWarning, quad, trapezoid,
                             cumulative_trapezoid)
import scipy.stats as stats
from scipy.stats._distn_infrastructure import argsreduce
import scipy.stats.distributions

from scipy.special import xlogy, polygamma, entr
from scipy.stats._distr_params import distcont, invdistcont
from .test_discrete_basic import distdiscrete, invdistdiscrete
from scipy.stats._continuous_distns import FitDataError, _argus_phi
from scipy.optimize import root, fmin
from itertools import product

# python -OO strips docstrings
DOCSTRINGS_STRIPPED = sys.flags.optimize > 1

# Failing on macOS 11, Intel CPUs. See gh-14901
MACOS_INTEL = (sys.platform == 'darwin') and (platform.machine() == 'x86_64')


# distributions to skip while testing the fix for the support method
# introduced in gh-13294. These distributions are skipped as they
# always return a non-nan support for every parametrization.
skip_test_support_gh13294_regression = ['tukeylambda', 'pearson3']


def _assert_hasattr(a, b, msg=None):
    if msg is None:
        msg = f'{a} does not have attribute {b}'
    assert_(hasattr(a, b), msg=msg)


def test_api_regression():
    # https://github.com/scipy/scipy/issues/3802
    _assert_hasattr(scipy.stats.distributions, 'f_gen')


def test_distributions_submodule():
    actual = set(scipy.stats.distributions.__all__)
    continuous = [dist[0] for dist in distcont]    # continuous dist names
    discrete = [dist[0] for dist in distdiscrete]  # discrete dist names
    other = ['rv_discrete', 'rv_continuous', 'rv_histogram',
             'entropy', 'trapz']
    expected = continuous + discrete + other

    # need to remove, e.g.,
    # <scipy.stats._continuous_distns.trapezoid_gen at 0x1df83bbc688>
    expected = set(filter(lambda s: not str(s).startswith('<'), expected))

    assert actual == expected


class TestVonMises:
    @pytest.mark.parametrize('k', [0.1, 1, 101])
    @pytest.mark.parametrize('x', [0, 1, np.pi, 10, 100])
    def test_vonmises_periodic(self, k, x):
        def check_vonmises_pdf_periodic(k, L, s, x):
            vm = stats.vonmises(k, loc=L, scale=s)
            assert_almost_equal(vm.pdf(x), vm.pdf(x % (2 * np.pi * s)))

        def check_vonmises_cdf_periodic(k, L, s, x):
            vm = stats.vonmises(k, loc=L, scale=s)
            assert_almost_equal(vm.cdf(x) % 1,
                                vm.cdf(x % (2 * np.pi * s)) % 1)

        check_vonmises_pdf_periodic(k, 0, 1, x)
        check_vonmises_pdf_periodic(k, 1, 1, x)
        check_vonmises_pdf_periodic(k, 0, 10, x)

        check_vonmises_cdf_periodic(k, 0, 1, x)
        check_vonmises_cdf_periodic(k, 1, 1, x)
        check_vonmises_cdf_periodic(k, 0, 10, x)

    def test_vonmises_line_support(self):
        assert_equal(stats.vonmises_line.a, -np.pi)
        assert_equal(stats.vonmises_line.b, np.pi)

    def test_vonmises_numerical(self):
        vm = stats.vonmises(800)
        assert_almost_equal(vm.cdf(0), 0.5)

    # Expected values of the vonmises PDF were computed using
    # mpmath with 50 digits of precision:
    #
    # def vmpdf_mp(x, kappa):
    #     x = mpmath.mpf(x)
    #     kappa = mpmath.mpf(kappa)
    #     num = mpmath.exp(kappa*mpmath.cos(x))
    #     den = 2 * mpmath.pi * mpmath.besseli(0, kappa)
    #     return num/den

    @pytest.mark.parametrize('x, kappa, expected_pdf',
                             [(0.1, 0.01, 0.16074242744907072),
                              (0.1, 25.0, 1.7515464099118245),
                              (0.1, 800, 0.2073272544458798),
                              (2.0, 0.01, 0.15849003875385817),
                              (2.0, 25.0, 8.356882934278192e-16),
                              (2.0, 800, 0.0)])
    def test_vonmises_pdf(self, x, kappa, expected_pdf):
        pdf = stats.vonmises.pdf(x, kappa)
        assert_allclose(pdf, expected_pdf, rtol=1e-15)

    # Expected values of the vonmises entropy were computed using
    # mpmath with 50 digits of precision:
    #
    # def vonmises_entropy(kappa):
    #     kappa = mpmath.mpf(kappa)
    #     return (-kappa * mpmath.besseli(1, kappa) /
    #             mpmath.besseli(0, kappa) + mpmath.log(2 * mpmath.pi *
    #             mpmath.besseli(0, kappa)))
    # >>> float(vonmises_entropy(kappa))

    @pytest.mark.parametrize('kappa, expected_entropy',
                             [(1, 1.6274014590199897),
                              (5, 0.6756431570114528),
                              (100, -0.8811275441649473),
                              (1000, -2.03468891852547),
                              (2000, -2.3813876496587847)])
    def test_vonmises_entropy(self, kappa, expected_entropy):
        entropy = stats.vonmises.entropy(kappa)
        assert_allclose(entropy, expected_entropy, rtol=1e-13)

    def test_vonmises_rvs_gh4598(self):
        # check that random variates wrap around as discussed in gh-4598
        seed = abs(hash('von_mises_rvs'))
        rng1 = np.random.default_rng(seed)
        rng2 = np.random.default_rng(seed)
        rng3 = np.random.default_rng(seed)
        rvs1 = stats.vonmises(1, loc=0, scale=1).rvs(random_state=rng1)
        rvs2 = stats.vonmises(1, loc=2*np.pi, scale=1).rvs(random_state=rng2)
        rvs3 = stats.vonmises(1, loc=0,
                              scale=(2*np.pi/abs(rvs1)+1)).rvs(random_state=rng3)
        assert_allclose(rvs1, rvs2, atol=1e-15)
        assert_allclose(rvs1, rvs3, atol=1e-15)

    # Expected values of the vonmises LOGPDF were computed
    # using wolfram alpha:
    # kappa * cos(x) - log(2*pi*I0(kappa))
    @pytest.mark.parametrize('x, kappa, expected_logpdf',
                             [(0.1, 0.01, -1.8279520246003170),
                              (0.1, 25.0, 0.5604990605420549),
                              (0.1, 800, -1.5734567947337514),
                              (2.0, 0.01, -1.8420635346185686),
                              (2.0, 25.0, -34.7182759850871489),
                              (2.0, 800, -1130.4942582548682739)])
    def test_vonmises_logpdf(self, x, kappa, expected_logpdf):
        logpdf = stats.vonmises.logpdf(x, kappa)
        assert_allclose(logpdf, expected_logpdf, rtol=1e-15)

    def test_vonmises_expect(self):
        """
        Test that the vonmises expectation values are
        computed correctly.  This test checks that the
        numeric integration estimates the correct normalization
        (1) and mean angle (loc).  These expectations are
        independent of the chosen 2pi interval.
        """
        rng = np.random.default_rng(6762668991392531563)

        loc, kappa, lb = rng.random(3) * 10
        res = stats.vonmises(loc=loc, kappa=kappa).expect(lambda x: 1)
        assert_allclose(res, 1)
        assert np.issubdtype(res.dtype, np.floating)

        bounds = lb, lb + 2 * np.pi
        res = stats.vonmises(loc=loc, kappa=kappa).expect(lambda x: 1, *bounds)
        assert_allclose(res, 1)
        assert np.issubdtype(res.dtype, np.floating)

        bounds = lb, lb + 2 * np.pi
        res = stats.vonmises(loc=loc, kappa=kappa).expect(lambda x: np.exp(1j*x),
                                                          *bounds, complex_func=1)
        assert_allclose(np.angle(res), loc % (2*np.pi))
        assert np.issubdtype(res.dtype, np.complexfloating)

    @pytest.mark.xslow
    @pytest.mark.parametrize("rvs_loc", [0, 2])
    @pytest.mark.parametrize("rvs_shape", [1, 100, 1e8])
    @pytest.mark.parametrize('fix_loc', [True, False])
    @pytest.mark.parametrize('fix_shape', [True, False])
    def test_fit_MLE_comp_optimizer(self, rvs_loc, rvs_shape,
                                    fix_loc, fix_shape):
        if fix_shape and fix_loc:
            pytest.skip("Nothing to fit.")

        rng = np.random.default_rng(6762668991392531563)
        data = stats.vonmises.rvs(rvs_shape, size=1000, loc=rvs_loc,
                                  random_state=rng)

        kwds = {'fscale': 1}
        if fix_loc:
            kwds['floc'] = rvs_loc
        if fix_shape:
            kwds['f0'] = rvs_shape

        _assert_less_or_close_loglike(stats.vonmises, data,
                                      stats.vonmises.nnlf, **kwds)

    @pytest.mark.parametrize('loc', [-0.5 * np.pi, 0, np.pi])
    @pytest.mark.parametrize('kappa_tol', [(1e-1, 5e-2), (1e2, 1e-2),
                                           (1e5, 1e-2)])
    def test_vonmises_fit_all(self, kappa_tol, loc):
        rng = np.random.default_rng(6762668991392531563)
        kappa, tol = kappa_tol
        data = stats.vonmises(loc=loc, kappa=kappa).rvs(100000,
                                                        random_state=rng)
        kappa_fit, loc_fit, scale_fit = stats.vonmises.fit(data)
        assert scale_fit == 1
        loc_vec = np.array([np.cos(loc), np.sin(loc)])
        loc_fit_vec = np.array([np.cos(loc_fit), np.sin(loc_fit)])
        angle = np.arccos(loc_vec.dot(loc_fit_vec))
        assert_allclose(angle, 0, atol=tol, rtol=0)
        assert_allclose(kappa, kappa_fit, rtol=tol)

    def test_vonmises_fit_shape(self):
        rng = np.random.default_rng(6762668991392531563)
        loc = 0.25*np.pi
        kappa = 10
        data = stats.vonmises(loc=loc, kappa=kappa).rvs(100000, random_state=rng)
        kappa_fit, loc_fit, scale_fit = stats.vonmises.fit(data, floc=loc)
        assert loc_fit == loc
        assert scale_fit == 1
        assert_allclose(kappa, kappa_fit, rtol=1e-2)

    @pytest.mark.xslow
    @pytest.mark.parametrize('loc', [-0.5 * np.pi, -0.9 * np.pi])
    def test_vonmises_fit_bad_floc(self, loc):
        data = [-0.92923506, -0.32498224, 0.13054989, -0.97252014, 2.79658071,
                -0.89110948, 1.22520295, 1.44398065, 2.49163859, 1.50315096,
                3.05437696, -2.73126329, -3.06272048, 1.64647173, 1.94509247,
                -1.14328023, 0.8499056, 2.36714682, -1.6823179, -0.88359996]
        data = np.asarray(data)
        kappa_fit, loc_fit, scale_fit = stats.vonmises.fit(data, floc=loc)
        assert kappa_fit == np.finfo(float).tiny
        _assert_less_or_close_loglike(stats.vonmises, data,
                                      stats.vonmises.nnlf, fscale=1, floc=loc)

    @pytest.mark.parametrize('sign', [-1, 1])
    def test_vonmises_fit_unwrapped_data(self, sign):
        rng = np.random.default_rng(6762668991392531563)
        data = stats.vonmises(loc=sign*0.5*np.pi, kappa=10).rvs(100000,
                                                                random_state=rng)
        shifted_data = data + 4*np.pi
        kappa_fit, loc_fit, scale_fit = stats.vonmises.fit(data)
        kappa_fit_shifted, loc_fit_shifted, _ = stats.vonmises.fit(shifted_data)
        assert_allclose(loc_fit, loc_fit_shifted)
        assert_allclose(kappa_fit, kappa_fit_shifted)
        assert scale_fit == 1
        assert -np.pi < loc_fit < np.pi


def _assert_less_or_close_loglike(dist, data, func=None, **kwds):
    """
    This utility function checks that the negative log-likelihood function
    (or `func`) of the result computed using dist.fit() is less than or equal
    to the result computed using the generic fit method.  Because of
    normal numerical imprecision, the "equality" check is made using
    `np.allclose` with a relative tolerance of 1e-15.
    """
    if func is None:
        func = dist.nnlf

    mle_analytical = dist.fit(data, **kwds)
    numerical_opt = super(type(dist), dist).fit(data, **kwds)
    ll_mle_analytical = func(mle_analytical, data)
    ll_numerical_opt = func(numerical_opt, data)
    assert (ll_mle_analytical <= ll_numerical_opt or
            np.allclose(ll_mle_analytical, ll_numerical_opt, rtol=1e-15))

    # Ideally we'd check that shapes are correctly fixed, too, but that is
    # complicated by the many ways of fixing them (e.g. f0, fix_a, fa).
    if 'floc' in kwds:
        assert mle_analytical[-2] == kwds['floc']
    if 'fscale' in kwds:
        assert mle_analytical[-1] == kwds['fscale']


def assert_fit_warnings(dist):
    param = ['floc', 'fscale']
    if dist.shapes:
        nshapes = len(dist.shapes.split(","))
        param += ['f0', 'f1', 'f2'][:nshapes]
    all_fixed = dict(zip(param, np.arange(len(param))))
    data = [1, 2, 3]
    with pytest.raises(RuntimeError,
                       match="All parameters fixed. There is nothing "
                       "to optimize."):
        dist.fit(data, **all_fixed)
    with pytest.raises(ValueError,
                       match="The data contains non-finite values"):
        dist.fit([np.nan])
    with pytest.raises(ValueError,
                       match="The data contains non-finite values"):
        dist.fit([np.inf])
    with pytest.raises(TypeError, match="Unknown keyword arguments:"):
        dist.fit(data, extra_keyword=2)
    with pytest.raises(TypeError, match="Too many positional arguments."):
        dist.fit(data, *[1]*(len(param) - 1))


@pytest.mark.parametrize('dist',
                         ['alpha', 'betaprime',
                          'fatiguelife', 'invgamma', 'invgauss', 'invweibull',
                          'johnsonsb', 'levy', 'levy_l', 'lognorm', 'gibrat',
                          'powerlognorm', 'rayleigh', 'wald'])
def test_support(dist):
    """gh-6235"""
    dct = dict(distcont)
    args = dct[dist]

    dist = getattr(stats, dist)

    assert_almost_equal(dist.pdf(dist.a, *args), 0)
    assert_equal(dist.logpdf(dist.a, *args), -np.inf)
    assert_almost_equal(dist.pdf(dist.b, *args), 0)
    assert_equal(dist.logpdf(dist.b, *args), -np.inf)


class TestRandInt:
    def setup_method(self):
        np.random.seed(1234)

    def test_rvs(self):
        vals = stats.randint.rvs(5, 30, size=100)
        assert_(numpy.all(vals < 30) & numpy.all(vals >= 5))
        assert_(len(vals) == 100)
        vals = stats.randint.rvs(5, 30, size=(2, 50))
        assert_(numpy.shape(vals) == (2, 50))
        assert_(vals.dtype.char in typecodes['AllInteger'])
        val = stats.randint.rvs(15, 46)
        assert_((val >= 15) & (val < 46))
        assert_(isinstance(val, numpy.ScalarType), msg=repr(type(val)))
        val = stats.randint(15, 46).rvs(3)
        assert_(val.dtype.char in typecodes['AllInteger'])

    def test_pdf(self):
        k = numpy.r_[0:36]
        out = numpy.where((k >= 5) & (k < 30), 1.0/(30-5), 0)
        vals = stats.randint.pmf(k, 5, 30)
        assert_array_almost_equal(vals, out)

    def test_cdf(self):
        x = np.linspace(0, 36, 100)
        k = numpy.floor(x)
        out = numpy.select([k >= 30, k >= 5], [1.0, (k-5.0+1)/(30-5.0)], 0)
        vals = stats.randint.cdf(x, 5, 30)
        assert_array_almost_equal(vals, out, decimal=12)


class TestBinom:
    def setup_method(self):
        np.random.seed(1234)

    def test_rvs(self):
        vals = stats.binom.rvs(10, 0.75, size=(2, 50))
        assert_(numpy.all(vals >= 0) & numpy.all(vals <= 10))
        assert_(numpy.shape(vals) == (2, 50))
        assert_(vals.dtype.char in typecodes['AllInteger'])
        val = stats.binom.rvs(10, 0.75)
        assert_(isinstance(val, int))
        val = stats.binom(10, 0.75).rvs(3)
        assert_(isinstance(val, numpy.ndarray))
        assert_(val.dtype.char in typecodes['AllInteger'])

    def test_pmf(self):
        # regression test for Ticket #1842
        vals1 = stats.binom.pmf(100, 100, 1)
        vals2 = stats.binom.pmf(0, 100, 0)
        assert_allclose(vals1, 1.0, rtol=1e-15, atol=0)
        assert_allclose(vals2, 1.0, rtol=1e-15, atol=0)

    def test_entropy(self):
        # Basic entropy tests.
        b = stats.binom(2, 0.5)
        expected_p = np.array([0.25, 0.5, 0.25])
        expected_h = -sum(xlogy(expected_p, expected_p))
        h = b.entropy()
        assert_allclose(h, expected_h)

        b = stats.binom(2, 0.0)
        h = b.entropy()
        assert_equal(h, 0.0)

        b = stats.binom(2, 1.0)
        h = b.entropy()
        assert_equal(h, 0.0)

    def test_warns_p0(self):
        # no spurious warnigns are generated for p=0; gh-3817
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            assert_equal(stats.binom(n=2, p=0).mean(), 0)
            assert_equal(stats.binom(n=2, p=0).std(), 0)

    def test_ppf_p1(self):
        # Check that gh-17388 is resolved: PPF == n when p = 1
        n = 4
        assert stats.binom.ppf(q=0.3, n=n, p=1.0) == n

    def test_pmf_poisson(self):
        # Check that gh-17146 is resolved: binom -> poisson
        n = 1541096362225563.0
        p = 1.0477878413173978e-18
        x = np.arange(3)
        res = stats.binom.pmf(x, n=n, p=p)
        ref = stats.poisson.pmf(x, n * p)
        assert_allclose(res, ref, atol=1e-16)

    def test_pmf_cdf(self):
        # Check that gh-17809 is resolved: binom.pmf(0) ~ binom.cdf(0)
        n = 25.0 * 10 ** 21
        p = 1.0 * 10 ** -21
        r = 0
        res = stats.binom.pmf(r, n, p)
        ref = stats.binom.cdf(r, n, p)
        assert_allclose(res, ref, atol=1e-16)

    def test_pmf_gh15101(self):
        # Check that gh-15101 is resolved (no divide warnings when p~1, n~oo)
        res = stats.binom.pmf(3, 2000, 0.999)
        assert_allclose(res, 0, atol=1e-16)


class TestArcsine:

    def test_endpoints(self):
        # Regression test for gh-13697.  The following calculation
        # should not generate a warning.
        p = stats.arcsine.pdf([0, 1])
        assert_equal(p, [np.inf, np.inf])


class TestBernoulli:
    def setup_method(self):
        np.random.seed(1234)

    def test_rvs(self):
        vals = stats.bernoulli.rvs(0.75, size=(2, 50))
        assert_(numpy.all(vals >= 0) & numpy.all(vals <= 1))
        assert_(numpy.shape(vals) == (2, 50))
        assert_(vals.dtype.char in typecodes['AllInteger'])
        val = stats.bernoulli.rvs(0.75)
        assert_(isinstance(val, int))
        val = stats.bernoulli(0.75).rvs(3)
        assert_(isinstance(val, numpy.ndarray))
        assert_(val.dtype.char in typecodes['AllInteger'])

    def test_entropy(self):
        # Simple tests of entropy.
        b = stats.bernoulli(0.25)
        expected_h = -0.25*np.log(0.25) - 0.75*np.log(0.75)
        h = b.entropy()
        assert_allclose(h, expected_h)

        b = stats.bernoulli(0.0)
        h = b.entropy()
        assert_equal(h, 0.0)

        b = stats.bernoulli(1.0)
        h = b.entropy()
        assert_equal(h, 0.0)


class TestBradford:
    # gh-6216
    def test_cdf_ppf(self):
        c = 0.1
        x = np.logspace(-20, -4)
        q = stats.bradford.cdf(x, c)
        xx = stats.bradford.ppf(q, c)
        assert_allclose(x, xx)


class TestChi:

    # "Exact" value of chi.sf(10, 4), as computed by Wolfram Alpha with
    #     1 - CDF[ChiDistribution[4], 10]
    CHI_SF_10_4 = 9.83662422461598e-21
    # "Exact" value of chi.mean(df=1000) as computed by Wolfram Alpha with
    #       Mean[ChiDistribution[1000]]
    CHI_MEAN_1000 = 31.614871896980

    def test_sf(self):
        s = stats.chi.sf(10, 4)
        assert_allclose(s, self.CHI_SF_10_4, rtol=1e-15)

    def test_isf(self):
        x = stats.chi.isf(self.CHI_SF_10_4, 4)
        assert_allclose(x, 10, rtol=1e-15)

    def test_mean(self):
        x = stats.chi.mean(df=1000)
        assert_allclose(x, self.CHI_MEAN_1000, rtol=1e-12)

    # Entropy references values were computed with the following mpmath code
    # from mpmath import mp
    # mp.dps = 50
    # def chi_entropy_mpmath(df):
    #     df = mp.mpf(df)
    #     half_df = 0.5 * df
    #     entropy = mp.log(mp.gamma(half_df)) + 0.5 * \
    #               (df - mp.log(2) - (df - mp.one) * mp.digamma(half_df))
    #     return float(entropy)

    @pytest.mark.parametrize('df, ref',
                             [(1e-4, -9989.7316027504),
                              (1, 0.7257913526447274),
                              (1e3, 1.0721981095025448),
                              (1e10, 1.0723649429080335),
                              (1e100, 1.0723649429247002)])
    def test_entropy(self, df, ref):
        assert_allclose(stats.chi(df).entropy(), ref, rtol=1e-15)


class TestNBinom:
    def setup_method(self):
        np.random.seed(1234)

    def test_rvs(self):
        vals = stats.nbinom.rvs(10, 0.75, size=(2, 50))
        assert_(numpy.all(vals >= 0))
        assert_(numpy.shape(vals) == (2, 50))
        assert_(vals.dtype.char in typecodes['AllInteger'])
        val = stats.nbinom.rvs(10, 0.75)
        assert_(isinstance(val, int))
        val = stats.nbinom(10, 0.75).rvs(3)
        assert_(isinstance(val, numpy.ndarray))
        assert_(val.dtype.char in typecodes['AllInteger'])

    def test_pmf(self):
        # regression test for ticket 1779
        assert_allclose(np.exp(stats.nbinom.logpmf(700, 721, 0.52)),
                        stats.nbinom.pmf(700, 721, 0.52))
        # logpmf(0,1,1) shouldn't return nan (regression test for gh-4029)
        val = scipy.stats.nbinom.logpmf(0, 1, 1)
        assert_equal(val, 0)

    def test_logcdf_gh16159(self):
        # check that gh16159 is resolved.
        vals = stats.nbinom.logcdf([0, 5, 0, 5], n=4.8, p=0.45)
        ref = np.log(stats.nbinom.cdf([0, 5, 0, 5], n=4.8, p=0.45))
        assert_allclose(vals, ref)


class TestGenInvGauss:
    def setup_method(self):
        np.random.seed(1234)

    @pytest.mark.slow
    def test_rvs_with_mode_shift(self):
        # ratio_unif w/ mode shift
        gig = stats.geninvgauss(2.3, 1.5)
        _, p = stats.kstest(gig.rvs(size=1500, random_state=1234), gig.cdf)
        assert_equal(p > 0.05, True)

    @pytest.mark.slow
    def test_rvs_without_mode_shift(self):
        # ratio_unif w/o mode shift
        gig = stats.geninvgauss(0.9, 0.75)
        _, p = stats.kstest(gig.rvs(size=1500, random_state=1234), gig.cdf)
        assert_equal(p > 0.05, True)

    @pytest.mark.slow
    def test_rvs_new_method(self):
        # new algorithm of Hoermann / Leydold
        gig = stats.geninvgauss(0.1, 0.2)
        _, p = stats.kstest(gig.rvs(size=1500, random_state=1234), gig.cdf)
        assert_equal(p > 0.05, True)

    @pytest.mark.slow
    def test_rvs_p_zero(self):
        def my_ks_check(p, b):
            gig = stats.geninvgauss(p, b)
            rvs = gig.rvs(size=1500, random_state=1234)
            return stats.kstest(rvs, gig.cdf)[1] > 0.05
        # boundary cases when p = 0
        assert_equal(my_ks_check(0, 0.2), True)  # new algo
        assert_equal(my_ks_check(0, 0.9), True)  # ratio_unif w/o shift
        assert_equal(my_ks_check(0, 1.5), True)  # ratio_unif with shift

    def test_rvs_negative_p(self):
        # if p negative, return inverse
        assert_equal(
                stats.geninvgauss(-1.5, 2).rvs(size=10, random_state=1234),
                1 / stats.geninvgauss(1.5, 2).rvs(size=10, random_state=1234))

    def test_invgauss(self):
        # test that invgauss is special case
        ig = stats.geninvgauss.rvs(size=1500, p=-0.5, b=1, random_state=1234)
        assert_equal(stats.kstest(ig, 'invgauss', args=[1])[1] > 0.15, True)
        # test pdf and cdf
        mu, x = 100, np.linspace(0.01, 1, 10)
        pdf_ig = stats.geninvgauss.pdf(x, p=-0.5, b=1 / mu, scale=mu)
        assert_allclose(pdf_ig, stats.invgauss(mu).pdf(x))
        cdf_ig = stats.geninvgauss.cdf(x, p=-0.5, b=1 / mu, scale=mu)
        assert_allclose(cdf_ig, stats.invgauss(mu).cdf(x))

    def test_pdf_R(self):
        # test against R package GIGrvg
        # x <- seq(0.01, 5, length.out = 10)
        # GIGrvg::dgig(x, 0.5, 1, 1)
        vals_R = np.array([2.081176820e-21, 4.488660034e-01, 3.747774338e-01,
                           2.693297528e-01, 1.905637275e-01, 1.351476913e-01,
                           9.636538981e-02, 6.909040154e-02, 4.978006801e-02,
                           3.602084467e-02])
        x = np.linspace(0.01, 5, 10)
        assert_allclose(vals_R, stats.geninvgauss.pdf(x, 0.5, 1))

    def test_pdf_zero(self):
        # pdf at 0 is 0, needs special treatment to avoid 1/x in pdf
        assert_equal(stats.geninvgauss.pdf(0, 0.5, 0.5), 0)
        # if x is large and p is moderate, make sure that pdf does not
        # overflow because of x**(p-1); exp(-b*x) forces pdf to zero
        assert_equal(stats.geninvgauss.pdf(2e6, 50, 2), 0)


class TestGenHyperbolic:
    def setup_method(self):
        np.random.seed(1234)

    def test_pdf_r(self):
        # test against R package GeneralizedHyperbolic
        # x <- seq(-10, 10, length.out = 10)
        # GeneralizedHyperbolic::dghyp(
        #    x = x, lambda = 2, alpha = 2, beta = 1, delta = 1.5, mu = 0.5
        # )
        vals_R = np.array([
            2.94895678275316e-13, 1.75746848647696e-10, 9.48149804073045e-08,
            4.17862521692026e-05, 0.0103947630463822, 0.240864958986839,
            0.162833527161649, 0.0374609592899472, 0.00634894847327781,
            0.000941920705790324
            ])

        lmbda, alpha, beta = 2, 2, 1
        mu, delta = 0.5, 1.5
        args = (lmbda, alpha*delta, beta*delta)

        gh = stats.genhyperbolic(*args, loc=mu, scale=delta)
        x = np.linspace(-10, 10, 10)

        assert_allclose(gh.pdf(x), vals_R, atol=0, rtol=1e-13)

    def test_cdf_r(self):
        # test against R package GeneralizedHyperbolic
        # q <- seq(-10, 10, length.out = 10)
        # GeneralizedHyperbolic::pghyp(
        #   q = q, lambda = 2, alpha = 2, beta = 1, delta = 1.5, mu = 0.5
        # )
        vals_R = np.array([
            1.01881590921421e-13, 6.13697274983578e-11, 3.37504977637992e-08,
            1.55258698166181e-05, 0.00447005453832497, 0.228935323956347,
            0.755759458895243, 0.953061062884484, 0.992598013917513,
            0.998942646586662
            ])

        lmbda, alpha, beta = 2, 2, 1
        mu, delta = 0.5, 1.5
        args = (lmbda, alpha*delta, beta*delta)

        gh = stats.genhyperbolic(*args, loc=mu, scale=delta)
        x = np.linspace(-10, 10, 10)

        assert_allclose(gh.cdf(x), vals_R, atol=0, rtol=1e-6)

    # The reference values were computed by implementing the PDF with mpmath
    # and integrating it with mp.quad.  The values were computed with
    # mp.dps=250, and then again with mp.dps=400 to ensure the full 64 bit
    # precision was computed.
    @pytest.mark.parametrize(
        'x, p, a, b, loc, scale, ref',
        [(-15, 2, 3, 1.5, 0.5, 1.5, 4.770036428808252e-20),
         (-15, 10, 1.5, 0.25, 1, 5, 0.03282964575089294),
         (-15, 10, 1.5, 1.375, 0, 1, 3.3711159600215594e-23),
         (-15, 0.125, 1.5, 1.49995, 0, 1, 4.729401428898605e-23),
         (-1, 0.125, 1.5, 1.49995, 0, 1, 0.0003565725914786859),
         (5, -0.125, 1.5, 1.49995, 0, 1, 0.2600651974023352),
         (5, -0.125, 1000, 999, 0, 1, 5.923270556517253e-28),
         (20, -0.125, 1000, 999, 0, 1, 0.23452293711665634),
         (40, -0.125, 1000, 999, 0, 1, 0.9999648749561968),
         (60, -0.125, 1000, 999, 0, 1, 0.9999999999975475)]
    )
    def test_cdf_mpmath(self, x, p, a, b, loc, scale, ref):
        cdf = stats.genhyperbolic.cdf(x, p, a, b, loc=loc, scale=scale)
        assert_allclose(cdf, ref, rtol=5e-12)

    # The reference values were computed by implementing the PDF with mpmath
    # and integrating it with mp.quad.  The values were computed with
    # mp.dps=250, and then again with mp.dps=400 to ensure the full 64 bit
    # precision was computed.
    @pytest.mark.parametrize(
        'x, p, a, b, loc, scale, ref',
        [(0, 1e-6, 12, -1, 0, 1, 0.38520358671350524),
         (-1, 3, 2.5, 2.375, 1, 3, 0.9999901774267577),
         (-20, 3, 2.5, 2.375, 1, 3, 1.0),
         (25, 2, 3, 1.5, 0.5, 1.5, 8.593419916523976e-10),
         (300, 10, 1.5, 0.25, 1, 5, 6.137415609872158e-24),
         (60, -0.125, 1000, 999, 0, 1, 2.4524915075944173e-12),
         (75, -0.125, 1000, 999, 0, 1, 2.9435194886214633e-18)]
    )
    def test_sf_mpmath(self, x, p, a, b, loc, scale, ref):
        sf = stats.genhyperbolic.sf(x, p, a, b, loc=loc, scale=scale)
        assert_allclose(sf, ref, rtol=5e-12)

    def test_moments_r(self):
        # test against R package GeneralizedHyperbolic
        # sapply(1:4,
        #    function(x) GeneralizedHyperbolic::ghypMom(
        #        order = x, lambda = 2, alpha = 2,
        #        beta = 1, delta = 1.5, mu = 0.5,
        #        momType = 'raw')
        # )

        vals_R = [2.36848366948115, 8.4739346779246,
                  37.8870502710066, 205.76608511485]

        lmbda, alpha, beta = 2, 2, 1
        mu, delta = 0.5, 1.5
        args = (lmbda, alpha*delta, beta*delta)

        vals_us = [
            stats.genhyperbolic(*args, loc=mu, scale=delta).moment(i)
            for i in range(1, 5)
            ]

        assert_allclose(vals_us, vals_R, atol=0, rtol=1e-13)

    def test_rvs(self):
        # Kolmogorov-Smirnov test to ensure alignment
        # of analytical and empirical cdfs

        lmbda, alpha, beta = 2, 2, 1
        mu, delta = 0.5, 1.5
        args = (lmbda, alpha*delta, beta*delta)

        gh = stats.genhyperbolic(*args, loc=mu, scale=delta)
        _, p = stats.kstest(gh.rvs(size=1500, random_state=1234), gh.cdf)

        assert_equal(p > 0.05, True)

    def test_pdf_t(self):
        # Test Against T-Student with 1 - 30 df
        df = np.linspace(1, 30, 10)

        # in principle alpha should be zero in practice for big lmbdas
        # alpha cannot be too small else pdf does not integrate
        alpha, beta = np.float_power(df, 2)*np.finfo(np.float32).eps, 0
        mu, delta = 0, np.sqrt(df)
        args = (-df/2, alpha, beta)

        gh = stats.genhyperbolic(*args, loc=mu, scale=delta)
        x = np.linspace(gh.ppf(0.01), gh.ppf(0.99), 50)[:, np.newaxis]

        assert_allclose(
            gh.pdf(x), stats.t.pdf(x, df),
            atol=0, rtol=1e-6
            )

    def test_pdf_cauchy(self):
        # Test Against Cauchy distribution

        # in principle alpha should be zero in practice for big lmbdas
        # alpha cannot be too small else pdf does not integrate
        lmbda, alpha, beta = -0.5, np.finfo(np.float32).eps, 0
        mu, delta = 0, 1
        args = (lmbda, alpha, beta)

        gh = stats.genhyperbolic(*args, loc=mu, scale=delta)
        x = np.linspace(gh.ppf(0.01), gh.ppf(0.99), 50)[:, np.newaxis]

        assert_allclose(
            gh.pdf(x), stats.cauchy.pdf(x),
            atol=0, rtol=1e-6
            )

    def test_pdf_laplace(self):
        # Test Against Laplace with location param [-10, 10]
        loc = np.linspace(-10, 10, 10)

        # in principle delta should be zero in practice for big loc delta
        # cannot be too small else pdf does not integrate
        delta = np.finfo(np.float32).eps

        lmbda, alpha, beta = 1, 1, 0
        args = (lmbda, alpha*delta, beta*delta)

        # ppf does not integrate for scale < 5e-4
        # therefore using simple linspace to define the support
        gh = stats.genhyperbolic(*args, loc=loc, scale=delta)
        x = np.linspace(-20, 20, 50)[:, np.newaxis]

        assert_allclose(
            gh.pdf(x), stats.laplace.pdf(x, loc=loc, scale=1),
            atol=0, rtol=1e-11
            )

    def test_pdf_norminvgauss(self):
        # Test Against NIG with varying alpha/beta/delta/mu

        alpha, beta, delta, mu = (
                np.linspace(1, 20, 10),
                np.linspace(0, 19, 10)*np.float_power(-1, range(10)),
                np.linspace(1, 1, 10),
                np.linspace(-100, 100, 10)
                )

        lmbda = - 0.5
        args = (lmbda, alpha * delta, beta * delta)

        gh = stats.genhyperbolic(*args, loc=mu, scale=delta)
        x = np.linspace(gh.ppf(0.01), gh.ppf(0.99), 50)[:, np.newaxis]

        assert_allclose(
            gh.pdf(x), stats.norminvgauss.pdf(
                x, a=alpha, b=beta, loc=mu, scale=delta),
            atol=0, rtol=1e-13
            )


class TestNormInvGauss:
    def setup_method(self):
        np.random.seed(1234)

    def test_cdf_R(self):
        # test pdf and cdf vals against R
        # require("GeneralizedHyperbolic")
        # x_test <- c(-7, -5, 0, 8, 15)
        # r_cdf <- GeneralizedHyperbolic::pnig(x_test, mu = 0, a = 1, b = 0.5)
        # r_pdf <- GeneralizedHyperbolic::dnig(x_test, mu = 0, a = 1, b = 0.5)
        r_cdf = np.array([8.034920282e-07, 2.512671945e-05, 3.186661051e-01,
                          9.988650664e-01, 9.999848769e-01])
        x_test = np.array([-7, -5, 0, 8, 15])
        vals_cdf = stats.norminvgauss.cdf(x_test, a=1, b=0.5)
        assert_allclose(vals_cdf, r_cdf, atol=1e-9)

    def test_pdf_R(self):
        # values from R as defined in test_cdf_R
        r_pdf = np.array([1.359600783e-06, 4.413878805e-05, 4.555014266e-01,
                          7.450485342e-04, 8.917889931e-06])
        x_test = np.array([-7, -5, 0, 8, 15])
        vals_pdf = stats.norminvgauss.pdf(x_test, a=1, b=0.5)
        assert_allclose(vals_pdf, r_pdf, atol=1e-9)

    @pytest.mark.parametrize('x, a, b, sf, rtol',
                             [(-1, 1, 0, 0.8759652211005315, 1e-13),
                              (25, 1, 0, 1.1318690184042579e-13, 1e-4),
                              (1, 5, -1.5, 0.002066711134653577, 1e-12),
                              (10, 5, -1.5, 2.308435233930669e-29, 1e-9)])
    def test_sf_isf_mpmath(self, x, a, b, sf, rtol):
        # Reference data generated with `reference_distributions.NormInvGauss`,
        # e.g. `NormInvGauss(alpha=1, beta=0).sf(-1)` with mp.dps = 50
        s = stats.norminvgauss.sf(x, a, b)
        assert_allclose(s, sf, rtol=rtol)
        i = stats.norminvgauss.isf(sf, a, b)
        assert_allclose(i, x, rtol=rtol)

    def test_sf_isf_mpmath_vectorized(self):
        x = [-1, 25]
        a = [1, 1]
        b = 0
        sf = [0.8759652211005315, 1.1318690184042579e-13]  # see previous test
        s = stats.norminvgauss.sf(x, a, b)
        assert_allclose(s, sf, rtol=1e-13, atol=1e-16)
        i = stats.norminvgauss.isf(sf, a, b)
        # Not perfect, but better than it was. See gh-13338.
        assert_allclose(i, x, rtol=1e-6)

    def test_gh8718(self):
        # Add test that gh-13338 resolved gh-8718
        dst = stats.norminvgauss(1, 0)
        x = np.arange(0, 20, 2)
        sf = dst.sf(x)
        isf = dst.isf(sf)
        assert_allclose(isf, x)

    def test_stats(self):
        a, b = 1, 0.5
        gamma = np.sqrt(a**2 - b**2)
        v_stats = (b / gamma, a**2 / gamma**3, 3.0 * b / (a * np.sqrt(gamma)),
                   3.0 * (1 + 4 * b**2 / a**2) / gamma)
        assert_equal(v_stats, stats.norminvgauss.stats(a, b, moments='mvsk'))

    def test_ppf(self):
        a, b = 1, 0.5
        x_test = np.array([0.001, 0.5, 0.999])
        vals = stats.norminvgauss.ppf(x_test, a, b)
        assert_allclose(x_test, stats.norminvgauss.cdf(vals, a, b))


class TestGeom:
    def setup_method(self):
        np.random.seed(1234)

    def test_rvs(self):
        vals = stats.geom.rvs(0.75, size=(2, 50))
        assert_(numpy.all(vals >= 0))
        assert_(numpy.shape(vals) == (2, 50))
        assert_(vals.dtype.char in typecodes['AllInteger'])
        val = stats.geom.rvs(0.75)
        assert_(isinstance(val, int))
        val = stats.geom(0.75).rvs(3)
        assert_(isinstance(val, numpy.ndarray))
        assert_(val.dtype.char in typecodes['AllInteger'])

    def test_rvs_9313(self):
        # previously, RVS were converted to `np.int32` on some platforms,
        # causing overflow for moderately large integer output (gh-9313).
        # Check that this is resolved to the extent possible w/ `np.int64`.
        rng = np.random.default_rng(649496242618848)
        rvs = stats.geom.rvs(np.exp(-35), size=5, random_state=rng)
        assert rvs.dtype == np.int64
        assert np.all(rvs > np.iinfo(np.int32).max)

    def test_pmf(self):
        vals = stats.geom.pmf([1, 2, 3], 0.5)
        assert_array_almost_equal(vals, [0.5, 0.25, 0.125])

    def test_logpmf(self):
        # regression test for ticket 1793
        vals1 = np.log(stats.geom.pmf([1, 2, 3], 0.5))
        vals2 = stats.geom.logpmf([1, 2, 3], 0.5)
        assert_allclose(vals1, vals2, rtol=1e-15, atol=0)

        # regression test for gh-4028
        val = stats.geom.logpmf(1, 1)
        assert_equal(val, 0.0)

    def test_cdf_sf(self):
        vals = stats.geom.cdf([1, 2, 3], 0.5)
        vals_sf = stats.geom.sf([1, 2, 3], 0.5)
        expected = array([0.5, 0.75, 0.875])
        assert_array_almost_equal(vals, expected)
        assert_array_almost_equal(vals_sf, 1-expected)

    def test_logcdf_logsf(self):
        vals = stats.geom.logcdf([1, 2, 3], 0.5)
        vals_sf = stats.geom.logsf([1, 2, 3], 0.5)
        expected = array([0.5, 0.75, 0.875])
        assert_array_almost_equal(vals, np.log(expected))
        assert_array_almost_equal(vals_sf, np.log1p(-expected))

    def test_ppf(self):
        vals = stats.geom.ppf([0.5, 0.75, 0.875], 0.5)
        expected = array([1.0, 2.0, 3.0])
        assert_array_almost_equal(vals, expected)

    def test_ppf_underflow(self):
        # this should not underflow
        assert_allclose(stats.geom.ppf(1e-20, 1e-20), 1.0, atol=1e-14)

    def test_entropy_gh18226(self):
        # gh-18226 reported that `geom.entropy` produced a warning and
        # inaccurate output for small p. Check that this is resolved.
        h = stats.geom(0.0146).entropy()
        assert_allclose(h, 5.219397961962308, rtol=1e-15)


class TestPlanck:
    def setup_method(self):
        np.random.seed(1234)

    def test_sf(self):
        vals = stats.planck.sf([1, 2, 3], 5.)
        expected = array([4.5399929762484854e-05,
                          3.0590232050182579e-07,
                          2.0611536224385579e-09])
        assert_array_almost_equal(vals, expected)

    def test_logsf(self):
        vals = stats.planck.logsf([1000., 2000., 3000.], 1000.)
        expected = array([-1001000., -2001000., -3001000.])
        assert_array_almost_equal(vals, expected)


class TestGennorm:
    def test_laplace(self):
        # test against Laplace (special case for beta=1)
        points = [1, 2, 3]
        pdf1 = stats.gennorm.pdf(points, 1)
        pdf2 = stats.laplace.pdf(points)
        assert_almost_equal(pdf1, pdf2)

    def test_norm(self):
        # test against normal (special case for beta=2)
        points = [1, 2, 3]
        pdf1 = stats.gennorm.pdf(points, 2)
        pdf2 = stats.norm.pdf(points, scale=2**-.5)
        assert_almost_equal(pdf1, pdf2)

    def test_rvs(self):
        np.random.seed(0)
        # 0 < beta < 1
        dist = stats.gennorm(0.5)
        rvs = dist.rvs(size=1000)
        assert stats.kstest(rvs, dist.cdf).pvalue > 0.1
        # beta = 1
        dist = stats.gennorm(1)
        rvs = dist.rvs(size=1000)
        rvs_laplace = stats.laplace.rvs(size=1000)
        assert stats.ks_2samp(rvs, rvs_laplace).pvalue > 0.1
        # beta = 2
        dist = stats.gennorm(2)
        rvs = dist.rvs(size=1000)
        rvs_norm = stats.norm.rvs(scale=1/2**0.5, size=1000)
        assert stats.ks_2samp(rvs, rvs_norm).pvalue > 0.1

    def test_rvs_broadcasting(self):
        np.random.seed(0)
        dist = stats.gennorm([[0.5, 1.], [2., 5.]])
        rvs = dist.rvs(size=[1000, 2, 2])
        assert stats.kstest(rvs[:, 0, 0], stats.gennorm(0.5).cdf)[1] > 0.1
        assert stats.kstest(rvs[:, 0, 1], stats.gennorm(1.0).cdf)[1] > 0.1
        assert stats.kstest(rvs[:, 1, 0], stats.gennorm(2.0).cdf)[1] > 0.1
        assert stats.kstest(rvs[:, 1, 1], stats.gennorm(5.0).cdf)[1] > 0.1


class TestGibrat:

    # sfx is sf(x).  The values were computed with mpmath:
    #
    #   from mpmath import mp
    #   mp.dps = 100
    #   def gibrat_sf(x):
    #       return 1 - mp.ncdf(mp.log(x))
    #
    # E.g.
    #
    #   >>> float(gibrat_sf(1.5))
    #   0.3425678305148459
    #
    @pytest.mark.parametrize('x, sfx', [(1.5, 0.3425678305148459),
                                        (5000, 8.173334352522493e-18)])
    def test_sf_isf(self, x, sfx):
        assert_allclose(stats.gibrat.sf(x), sfx, rtol=2e-14)
        assert_allclose(stats.gibrat.isf(sfx), x, rtol=2e-14)


class TestGompertz:

    def test_gompertz_accuracy(self):
        # Regression test for gh-4031
        p = stats.gompertz.ppf(stats.gompertz.cdf(1e-100, 1), 1)
        assert_allclose(p, 1e-100)

    # sfx is sf(x).  The values were computed with mpmath:
    #
    #   from mpmath import mp
    #   mp.dps = 100
    #   def gompertz_sf(x, c):
    #       reurn mp.exp(-c*mp.expm1(x))
    #
    # E.g.
    #
    #   >>> float(gompertz_sf(1, 2.5))
    #   0.013626967146253437
    #
    @pytest.mark.parametrize('x, c, sfx', [(1, 2.5, 0.013626967146253437),
                                           (3, 2.5, 1.8973243273704087e-21),
                                           (0.05, 5, 0.7738668242570479),
                                           (2.25, 5, 3.707795833465481e-19)])
    def test_sf_isf(self, x, c, sfx):
        assert_allclose(stats.gompertz.sf(x, c), sfx, rtol=1e-14)
        assert_allclose(stats.gompertz.isf(sfx, c), x, rtol=1e-14)

    # reference values were computed with mpmath
    # from mpmath import mp
    # mp.dps = 100
    # def gompertz_entropy(c):
    #     c = mp.mpf(c)
    #     return float(mp.one - mp.log(c) - mp.exp(c)*mp.e1(c))

    @pytest.mark.parametrize('c, ref', [(1e-4, 1.5762523017634573),
                                        (1, 0.4036526376768059),
                                        (1000, -5.908754280976161),
                                        (1e10, -22.025850930040455)])
    def test_entropy(self, c, ref):
        assert_allclose(stats.gompertz.entropy(c), ref, rtol=1e-14)


class TestFoldNorm:

    # reference values were computed with mpmath with 50 digits of precision
    # from mpmath import mp
    # mp.dps = 50
    # mp.mpf(0.5) * (mp.erf((x - c)/mp.sqrt(2)) + mp.erf((x + c)/mp.sqrt(2)))

    @pytest.mark.parametrize('x, c, ref', [(1e-4, 1e-8, 7.978845594730578e-05),
                                           (1e-4, 1e-4, 7.97884555483635e-05)])
    def test_cdf(self, x, c, ref):
        assert_allclose(stats.foldnorm.cdf(x, c), ref, rtol=1e-15)


class TestHalfNorm:

    # sfx is sf(x).  The values were computed with mpmath:
    #
    #   from mpmath import mp
    #   mp.dps = 100
    #   def halfnorm_sf(x):
    #       reurn 2*(1 - mp.ncdf(x))
    #
    # E.g.
    #
    #   >>> float(halfnorm_sf(1))
    #   0.3173105078629141
    #
    @pytest.mark.parametrize('x, sfx', [(1, 0.3173105078629141),
                                        (10, 1.523970604832105e-23)])
    def test_sf_isf(self, x, sfx):
        assert_allclose(stats.halfnorm.sf(x), sfx, rtol=1e-14)
        assert_allclose(stats.halfnorm.isf(sfx), x, rtol=1e-14)

    #   reference values were computed via mpmath
    #   from mpmath import mp
    #   mp.dps = 100
    #   def halfnorm_cdf_mpmath(x):
    #       x = mp.mpf(x)
    #       return float(mp.erf(x/mp.sqrt(2.)))

    @pytest.mark.parametrize('x, ref', [(1e-40, 7.978845608028653e-41),
                                        (1e-18, 7.978845608028654e-19),
                                        (8, 0.9999999999999988)])
    def test_cdf(self, x, ref):
        assert_allclose(stats.halfnorm.cdf(x), ref, rtol=1e-15)


class TestHalfLogistic:
    # survival function reference values were computed with mpmath
    # from mpmath import mp
    # mp.dps = 50
    # def sf_mpmath(x):
    #     x = mp.mpf(x)
    #     return float(mp.mpf(2.)/(mp.exp(x) + mp.one))

    @pytest.mark.parametrize('x, ref', [(100, 7.440151952041672e-44),
                                        (200, 2.767793053473475e-87)])
    def test_sf(self, x, ref):
        assert_allclose(stats.halflogistic.sf(x), ref, rtol=1e-15)

    # inverse survival function reference values were computed with mpmath
    # from mpmath import mp
    # mp.dps = 200
    # def isf_mpmath(x):
    #     halfx = mp.mpf(x)/2
    #     return float(-mp.log(halfx/(mp.one - halfx)))

    @pytest.mark.parametrize('q, ref', [(7.440151952041672e-44, 100),
                                        (2.767793053473475e-87, 200),
                                        (1-1e-9, 1.999999943436137e-09),
                                        (1-1e-15, 1.9984014443252818e-15)])
    def test_isf(self, q, ref):
        assert_allclose(stats.halflogistic.isf(q), ref, rtol=1e-15)


class TestHalfgennorm:
    def test_expon(self):
        # test against exponential (special case for beta=1)
        points = [1, 2, 3]
        pdf1 = stats.halfgennorm.pdf(points, 1)
        pdf2 = stats.expon.pdf(points)
        assert_almost_equal(pdf1, pdf2)

    def test_halfnorm(self):
        # test against half normal (special case for beta=2)
        points = [1, 2, 3]
        pdf1 = stats.halfgennorm.pdf(points, 2)
        pdf2 = stats.halfnorm.pdf(points, scale=2**-.5)
        assert_almost_equal(pdf1, pdf2)

    def test_gennorm(self):
        # test against generalized normal
        points = [1, 2, 3]
        pdf1 = stats.halfgennorm.pdf(points, .497324)
        pdf2 = stats.gennorm.pdf(points, .497324)
        assert_almost_equal(pdf1, 2*pdf2)


class TestLaplaceasymmetric:
    def test_laplace(self):
        # test against Laplace (special case for kappa=1)
        points = np.array([1, 2, 3])
        pdf1 = stats.laplace_asymmetric.pdf(points, 1)
        pdf2 = stats.laplace.pdf(points)
        assert_allclose(pdf1, pdf2)

    def test_asymmetric_laplace_pdf(self):
        # test assymetric Laplace
        points = np.array([1, 2, 3])
        kappa = 2
        kapinv = 1/kappa
        pdf1 = stats.laplace_asymmetric.pdf(points, kappa)
        pdf2 = stats.laplace_asymmetric.pdf(points*(kappa**2), kapinv)
        assert_allclose(pdf1, pdf2)

    def test_asymmetric_laplace_log_10_16(self):
        # test assymetric Laplace
        points = np.array([-np.log(16), np.log(10)])
        kappa = 2
        pdf1 = stats.laplace_asymmetric.pdf(points, kappa)
        cdf1 = stats.laplace_asymmetric.cdf(points, kappa)
        sf1 = stats.laplace_asymmetric.sf(points, kappa)
        pdf2 = np.array([1/10, 1/250])
        cdf2 = np.array([1/5, 1 - 1/500])
        sf2 = np.array([4/5, 1/500])
        ppf1 = stats.laplace_asymmetric.ppf(cdf2, kappa)
        ppf2 = points
        isf1 = stats.laplace_asymmetric.isf(sf2, kappa)
        isf2 = points
        assert_allclose(np.concatenate((pdf1, cdf1, sf1, ppf1, isf1)),
                        np.concatenate((pdf2, cdf2, sf2, ppf2, isf2)))


class TestTruncnorm:
    def setup_method(self):
        np.random.seed(1234)

    @pytest.mark.parametrize("a, b, ref",
                             [(0, 100, 0.7257913526447274),
                             (0.6, 0.7, -2.3027610681852573),
                             (1e-06, 2e-06, -13.815510557964274)])
    def test_entropy(self, a, b, ref):
        # All reference values were calculated with mpmath:
        # import numpy as np
        # from mpmath import mp
        # mp.dps = 50
        # def entropy_trun(a, b):
        #     a, b = mp.mpf(a), mp.mpf(b)
        #     Z = mp.ncdf(b) - mp.ncdf(a)
        #
        #     def pdf(x):
        #         return mp.npdf(x) / Z
        #
        #     res = -mp.quad(lambda t: pdf(t) * mp.log(pdf(t)), [a, b])
        #     return np.float64(res)
        assert_allclose(stats.truncnorm.entropy(a, b), ref, rtol=1e-10)

    @pytest.mark.parametrize("a, b, ref",
                             [(1e-11, 10000000000.0, 0.725791352640738),
                             (1e-100, 1e+100, 0.7257913526447274),
                             (-1e-100, 1e+100, 0.7257913526447274),
                             (-1e+100, 1e+100, 1.4189385332046727)])
    def test_extreme_entropy(self, a, b, ref):
        # The reference values were calculated with mpmath
        # import numpy as np
        # from mpmath import mp
        # mp.dps = 50
        # def trunc_norm_entropy(a, b):
        #     a, b = mp.mpf(a), mp.mpf(b)
        #     Z = mp.ncdf(b) - mp.ncdf(a)
        #     A = mp.log(mp.sqrt(2 * mp.pi * mp.e) * Z)
        #     B = (a * mp.npdf(a) - b * mp.npdf(b)) / (2 * Z)
        #     return np.float64(A + B)
        assert_allclose(stats.truncnorm.entropy(a, b), ref, rtol=1e-14)

    def test_ppf_ticket1131(self):
        vals = stats.truncnorm.ppf([-0.5, 0, 1e-4, 0.5, 1-1e-4, 1, 2], -1., 1.,
                                   loc=[3]*7, scale=2)
        expected = np.array([np.nan, 1, 1.00056419, 3, 4.99943581, 5, np.nan])
        assert_array_almost_equal(vals, expected)

    def test_isf_ticket1131(self):
        vals = stats.truncnorm.isf([-0.5, 0, 1e-4, 0.5, 1-1e-4, 1, 2], -1., 1.,
                                   loc=[3]*7, scale=2)
        expected = np.array([np.nan, 5, 4.99943581, 3, 1.00056419, 1, np.nan])
        assert_array_almost_equal(vals, expected)

    def test_gh_2477_small_values(self):
        # Check a case that worked in the original issue.
        low, high = -11, -10
        x = stats.truncnorm.rvs(low, high, 0, 1, size=10)
        assert_(low < x.min() < x.max() < high)
        # Check a case that failed in the original issue.
        low, high = 10, 11
        x = stats.truncnorm.rvs(low, high, 0, 1, size=10)
        assert_(low < x.min() < x.max() < high)

    def test_gh_2477_large_values(self):
        # Check a case that used to fail because of extreme tailness.
        low, high = 100, 101
        x = stats.truncnorm.rvs(low, high, 0, 1, size=10)
        assert_(low <= x.min() <= x.max() <= high), str([low, high, x])

        # Check some additional extreme tails
        low, high = 1000, 1001
        x = stats.truncnorm.rvs(low, high, 0, 1, size=10)
        assert_(low < x.min() < x.max() < high)

        low, high = 10000, 10001
        x = stats.truncnorm.rvs(low, high, 0, 1, size=10)
        assert_(low < x.min() < x.max() < high)

        low, high = -10001, -10000
        x = stats.truncnorm.rvs(low, high, 0, 1, size=10)
        assert_(low < x.min() < x.max() < high)

    def test_gh_9403_nontail_values(self):
        for low, high in [[3, 4], [-4, -3]]:
            xvals = np.array([-np.inf, low, high, np.inf])
            xmid = (high+low)/2.0
            cdfs = stats.truncnorm.cdf(xvals, low, high)
            sfs = stats.truncnorm.sf(xvals, low, high)
            pdfs = stats.truncnorm.pdf(xvals, low, high)
            expected_cdfs = np.array([0, 0, 1, 1])
            expected_sfs = np.array([1.0, 1.0, 0.0, 0.0])
            expected_pdfs = np.array([0, 3.3619772, 0.1015229, 0])
            if low < 0:
                expected_pdfs = np.array([0, 0.1015229, 3.3619772, 0])
            assert_almost_equal(cdfs, expected_cdfs)
            assert_almost_equal(sfs, expected_sfs)
            assert_almost_equal(pdfs, expected_pdfs)
            assert_almost_equal(np.log(expected_pdfs[1]/expected_pdfs[2]),
                                low + 0.5)
            pvals = np.array([0, 0.5, 1.0])
            ppfs = stats.truncnorm.ppf(pvals, low, high)
            expected_ppfs = np.array([low, np.sign(low)*3.1984741, high])
            assert_almost_equal(ppfs, expected_ppfs)

            if low < 0:
                assert_almost_equal(stats.truncnorm.sf(xmid, low, high),
                                    0.8475544278436675)
                assert_almost_equal(stats.truncnorm.cdf(xmid, low, high),
                                    0.1524455721563326)
            else:
                assert_almost_equal(stats.truncnorm.cdf(xmid, low, high),
                                    0.8475544278436675)
                assert_almost_equal(stats.truncnorm.sf(xmid, low, high),
                                    0.1524455721563326)
            pdf = stats.truncnorm.pdf(xmid, low, high)
            assert_almost_equal(np.log(pdf/expected_pdfs[2]), (xmid+0.25)/2)

    def test_gh_9403_medium_tail_values(self):
        for low, high in [[39, 40], [-40, -39]]:
            xvals = np.array([-np.inf, low, high, np.inf])
            xmid = (high+low)/2.0
            cdfs = stats.truncnorm.cdf(xvals, low, high)
            sfs = stats.truncnorm.sf(xvals, low, high)
            pdfs = stats.truncnorm.pdf(xvals, low, high)
            expected_cdfs = np.array([0, 0, 1, 1])
            expected_sfs = np.array([1.0, 1.0, 0.0, 0.0])
            expected_pdfs = np.array([0, 3.90256074e+01, 2.73349092e-16, 0])
            if low < 0:
                expected_pdfs = np.array([0, 2.73349092e-16,
                                          3.90256074e+01, 0])
            assert_almost_equal(cdfs, expected_cdfs)
            assert_almost_equal(sfs, expected_sfs)
            assert_almost_equal(pdfs, expected_pdfs)
            assert_almost_equal(np.log(expected_pdfs[1]/expected_pdfs[2]),
                                low + 0.5)
            pvals = np.array([0, 0.5, 1.0])
            ppfs = stats.truncnorm.ppf(pvals, low, high)
            expected_ppfs = np.array([low, np.sign(low)*39.01775731, high])
            assert_almost_equal(ppfs, expected_ppfs)
            cdfs = stats.truncnorm.cdf(ppfs, low, high)
            assert_almost_equal(cdfs, pvals)

            if low < 0:
                assert_almost_equal(stats.truncnorm.sf(xmid, low, high),
                                    0.9999999970389126)
                assert_almost_equal(stats.truncnorm.cdf(xmid, low, high),
                                    2.961048103554866e-09)
            else:
                assert_almost_equal(stats.truncnorm.cdf(xmid, low, high),
                                    0.9999999970389126)
                assert_almost_equal(stats.truncnorm.sf(xmid, low, high),
                                    2.961048103554866e-09)
            pdf = stats.truncnorm.pdf(xmid, low, high)
            assert_almost_equal(np.log(pdf/expected_pdfs[2]), (xmid+0.25)/2)

            xvals = np.linspace(low, high, 11)
            xvals2 = -xvals[::-1]
            assert_almost_equal(stats.truncnorm.cdf(xvals, low, high),
                                stats.truncnorm.sf(xvals2, -high, -low)[::-1])
            assert_almost_equal(stats.truncnorm.sf(xvals, low, high),
                                stats.truncnorm.cdf(xvals2, -high, -low)[::-1])
            assert_almost_equal(stats.truncnorm.pdf(xvals, low, high),
                                stats.truncnorm.pdf(xvals2, -high, -low)[::-1])

    def test_cdf_tail_15110_14753(self):
        # Check accuracy issues reported in gh-14753 and gh-155110
        # Ground truth values calculated using Wolfram Alpha, e.g.
        # (CDF[NormalDistribution[0,1],83/10]-CDF[NormalDistribution[0,1],8])/
        #     (1 - CDF[NormalDistribution[0,1],8])
        assert_allclose(stats.truncnorm(13., 15.).cdf(14.),
                        0.9999987259565643)
        assert_allclose(stats.truncnorm(8, np.inf).cdf(8.3),
                        0.9163220907327540)

    # Test data for the truncnorm stats() method.
    # The data in each row is:
    #   a, b, mean, variance, skewness, excess kurtosis. Generated using
    # https://gist.github.com/WarrenWeckesser/636b537ee889679227d53543d333a720
    _truncnorm_stats_data = [
        [-30, 30,
         0.0, 1.0, 0.0, 0.0],
        [-10, 10,
         0.0, 1.0, 0.0, -1.4927521335810455e-19],
        [-3, 3,
         0.0, 0.9733369246625415, 0.0, -0.17111443639774404],
        [-2, 2,
         0.0, 0.7737413035499232, 0.0, -0.6344632828703505],
        [0, np.inf,
         0.7978845608028654,
         0.3633802276324187,
         0.995271746431156,
         0.8691773036059741],
        [-np.inf, 0,
         -0.7978845608028654,
         0.3633802276324187,
         -0.995271746431156,
         0.8691773036059741],
        [-1, 3,
         0.282786110727154,
         0.6161417353578293,
         0.5393018494027877,
         -0.20582065135274694],
        [-3, 1,
         -0.282786110727154,
         0.6161417353578293,
         -0.5393018494027877,
         -0.20582065135274694],
        [-10, -9,
         -9.108456288012409,
         0.011448805821636248,
         -1.8985607290949496,
         5.0733461105025075],
    ]
    _truncnorm_stats_data = np.array(_truncnorm_stats_data)

    @pytest.mark.parametrize("case", _truncnorm_stats_data)
    def test_moments(self, case):
        a, b, m0, v0, s0, k0 = case
        m, v, s, k = stats.truncnorm.stats(a, b, moments='mvsk')
        assert_allclose([m, v, s, k], [m0, v0, s0, k0], atol=1e-17)

    def test_9902_moments(self):
        m, v = stats.truncnorm.stats(0, np.inf, moments='mv')
        assert_almost_equal(m, 0.79788456)
        assert_almost_equal(v, 0.36338023)

    def test_gh_1489_trac_962_rvs(self):
        # Check the original example.
        low, high = 10, 15
        x = stats.truncnorm.rvs(low, high, 0, 1, size=10)
        assert_(low < x.min() < x.max() < high)

    def test_gh_11299_rvs(self):
        # Arose from investigating gh-11299
        # Test multiple shape parameters simultaneously.
        low = [-10, 10, -np.inf, -5, -np.inf, -np.inf, -45, -45, 40, -10, 40]
        high = [-5, 11, 5, np.inf, 40, -40, 40, -40, 45, np.inf, np.inf]
        x = stats.truncnorm.rvs(low, high, size=(5, len(low)))
        assert np.shape(x) == (5, len(low))
        assert_(np.all(low <= x.min(axis=0)))
        assert_(np.all(x.max(axis=0) <= high))

    def test_rvs_Generator(self):
        # check that rvs can use a Generator
        if hasattr(np.random, "default_rng"):
            stats.truncnorm.rvs(-10, -5, size=5,
                                random_state=np.random.default_rng())

    def test_logcdf_gh17064(self):
        # regression test for gh-17064 - avoid roundoff error for logcdfs ~0
        a = np.array([-np.inf, -np.inf, -8, -np.inf, 10])
        b = np.array([np.inf, np.inf, 8, 10, np.inf])
        x = np.array([10, 7.5, 7.5, 9, 20])
        expected = [-7.619853024160525e-24, -3.190891672910947e-14,
                    -3.128682067168231e-14, -1.1285122074235991e-19,
                    -3.61374964828753e-66]
        assert_allclose(stats.truncnorm(a, b).logcdf(x), expected)
        assert_allclose(stats.truncnorm(-b, -a).logsf(-x), expected)

    def test_moments_gh18634(self):
        # gh-18634 reported that moments 5 and higher didn't work; check that
        # this is resolved
        res = stats.truncnorm(-2, 3).moment(5)
        # From Mathematica:
        # Moment[TruncatedDistribution[{-2, 3}, NormalDistribution[]], 5]
        ref = 1.645309620208361
        assert_allclose(res, ref)


class TestGenLogistic:

    # Expected values computed with mpmath with 50 digits of precision.
    @pytest.mark.parametrize('x, expected', [(-1000, -1499.5945348918917),
                                             (-125, -187.09453489189184),
                                             (0, -1.3274028432916989),
                                             (100, -99.59453489189184),
                                             (1000, -999.5945348918918)])
    def test_logpdf(self, x, expected):
        c = 1.5
        logp = stats.genlogistic.logpdf(x, c)
        assert_allclose(logp, expected, rtol=1e-13)

    # Expected values computed with mpmath with 50 digits of precision
    # from mpmath import mp
    # mp.dps = 50
    # def entropy_mp(c):
    #     c = mp.mpf(c)
    #     return float(-mp.log(c)+mp.one+mp.digamma(c + mp.one) + mp.euler)

    @pytest.mark.parametrize('c, ref', [(1e-100, 231.25850929940458),
                                        (1e-4, 10.21050485336338),
                                        (1e8, 1.577215669901533),
                                        (1e100, 1.5772156649015328)])
    def test_entropy(self, c, ref):
        assert_allclose(stats.genlogistic.entropy(c), ref, rtol=5e-15)

    # Expected values computed with mpmath with 50 digits of precision
    # from mpmath import mp
    # mp.dps = 1000
    #
    # def genlogistic_cdf_mp(x, c):
    #     x = mp.mpf(x)
    #     c = mp.mpf(c)
    #     return (mp.one + mp.exp(-x)) ** (-c)
    #
    # def genlogistic_sf_mp(x, c):
    #     return mp.one - genlogistic_cdf_mp(x, c)
    #
    # x, c, ref = 100, 0.02, -7.440151952041672e-466
    # print(float(mp.log(genlogistic_cdf_mp(x, c))))
    # ppf/isf reference values generated by passing in `ref` (`q` is produced)

    @pytest.mark.parametrize('x, c, ref', [(200, 10, 1.3838965267367375e-86),
                                           (500, 20, 1.424915281348257e-216)])
    def test_sf(self, x, c, ref):
        assert_allclose(stats.genlogistic.sf(x, c), ref, rtol=1e-14)

    @pytest.mark.parametrize('q, c, ref', [(0.01, 200, 9.898441467379765),
                                           (0.001, 2, 7.600152115573173)])
    def test_isf(self, q, c, ref):
        assert_allclose(stats.genlogistic.isf(q, c), ref, rtol=5e-16)

    @pytest.mark.parametrize('q, c, ref', [(0.5, 200, 5.6630969187064615),
                                           (0.99, 20, 7.595630231412436)])
    def test_ppf(self, q, c, ref):
        assert_allclose(stats.genlogistic.ppf(q, c), ref, rtol=5e-16)

    @pytest.mark.parametrize('x, c, ref', [(100, 0.02, -7.440151952041672e-46),
                                           (50, 20, -3.857499695927835e-21)])
    def test_logcdf(self, x, c, ref):
        assert_allclose(stats.genlogistic.logcdf(x, c), ref, rtol=1e-15)


class TestHypergeom:
    def setup_method(self):
        np.random.seed(1234)

    def test_rvs(self):
        vals = stats.hypergeom.rvs(20, 10, 3, size=(2, 50))
        assert_(numpy.all(vals >= 0) &
                numpy.all(vals <= 3))
        assert_(numpy.shape(vals) == (2, 50))
        assert_(vals.dtype.char in typecodes['AllInteger'])
        val = stats.hypergeom.rvs(20, 3, 10)
        assert_(isinstance(val, int))
        val = stats.hypergeom(20, 3, 10).rvs(3)
        assert_(isinstance(val, numpy.ndarray))
        assert_(val.dtype.char in typecodes['AllInteger'])

    def test_precision(self):
        # comparison number from mpmath
        M = 2500
        n = 50
        N = 500
        tot = M
        good = n
        hgpmf = stats.hypergeom.pmf(2, tot, good, N)
        assert_almost_equal(hgpmf, 0.0010114963068932233, 11)

    def test_args(self):
        # test correct output for corner cases of arguments
        # see gh-2325
        assert_almost_equal(stats.hypergeom.pmf(0, 2, 1, 0), 1.0, 11)
        assert_almost_equal(stats.hypergeom.pmf(1, 2, 1, 0), 0.0, 11)

        assert_almost_equal(stats.hypergeom.pmf(0, 2, 0, 2), 1.0, 11)
        assert_almost_equal(stats.hypergeom.pmf(1, 2, 1, 0), 0.0, 11)

    def test_cdf_above_one(self):
        # for some values of parameters, hypergeom cdf was >1, see gh-2238
        assert_(0 <= stats.hypergeom.cdf(30, 13397950, 4363, 12390) <= 1.0)

    def test_precision2(self):
        # Test hypergeom precision for large numbers.  See #1218.
        # Results compared with those from R.
        oranges = 9.9e4
        pears = 1.1e5
        fruits_eaten = np.array([3, 3.8, 3.9, 4, 4.1, 4.2, 5]) * 1e4
        quantile = 2e4
        res = [stats.hypergeom.sf(quantile, oranges + pears, oranges, eaten)
               for eaten in fruits_eaten]
        expected = np.array([0, 1.904153e-114, 2.752693e-66, 4.931217e-32,
                             8.265601e-11, 0.1237904, 1])
        assert_allclose(res, expected, atol=0, rtol=5e-7)

        # Test with array_like first argument
        quantiles = [1.9e4, 2e4, 2.1e4, 2.15e4]
        res2 = stats.hypergeom.sf(quantiles, oranges + pears, oranges, 4.2e4)
        expected2 = [1, 0.1237904, 6.511452e-34, 3.277667e-69]
        assert_allclose(res2, expected2, atol=0, rtol=5e-7)

    def test_entropy(self):
        # Simple tests of entropy.
        hg = stats.hypergeom(4, 1, 1)
        h = hg.entropy()
        expected_p = np.array([0.75, 0.25])
        expected_h = -np.sum(xlogy(expected_p, expected_p))
        assert_allclose(h, expected_h)

        hg = stats.hypergeom(1, 1, 1)
        h = hg.entropy()
        assert_equal(h, 0.0)

    def test_logsf(self):
        # Test logsf for very large numbers. See issue #4982
        # Results compare with those from R (v3.2.0):
        # phyper(k, n, M-n, N, lower.tail=FALSE, log.p=TRUE)
        # -2239.771

        k = 1e4
        M = 1e7
        n = 1e6
        N = 5e4

        result = stats.hypergeom.logsf(k, M, n, N)
        expected = -2239.771   # From R
        assert_almost_equal(result, expected, decimal=3)

        k = 1
        M = 1600
        n = 600
        N = 300

        result = stats.hypergeom.logsf(k, M, n, N)
        expected = -2.566567e-68   # From R
        assert_almost_equal(result, expected, decimal=15)

    def test_logcdf(self):
        # Test logcdf for very large numbers. See issue #8692
        # Results compare with those from R (v3.3.2):
        # phyper(k, n, M-n, N, lower.tail=TRUE, log.p=TRUE)
        # -5273.335

        k = 1
        M = 1e7
        n = 1e6
        N = 5e4

        result = stats.hypergeom.logcdf(k, M, n, N)
        expected = -5273.335   # From R
        assert_almost_equal(result, expected, decimal=3)

        # Same example as in issue #8692
        k = 40
        M = 1600
        n = 50
        N = 300

        result = stats.hypergeom.logcdf(k, M, n, N)
        expected = -7.565148879229e-23    # From R
        assert_almost_equal(result, expected, decimal=15)

        k = 125
        M = 1600
        n = 250
        N = 500

        result = stats.hypergeom.logcdf(k, M, n, N)
        expected = -4.242688e-12    # From R
        assert_almost_equal(result, expected, decimal=15)

        # test broadcasting robustness based on reviewer
        # concerns in PR 9603; using an array version of
        # the example from issue #8692
        k = np.array([40, 40, 40])
        M = 1600
        n = 50
        N = 300

        result = stats.hypergeom.logcdf(k, M, n, N)
        expected = np.full(3, -7.565148879229e-23)  # filled from R result
        assert_almost_equal(result, expected, decimal=15)


class TestLoggamma:

    # Expected cdf values were computed with mpmath. For given x and c,
    #     x = mpmath.mpf(x)
    #     c = mpmath.mpf(c)
    #     cdf = mpmath.gammainc(c, 0, mpmath.exp(x),
    #                           regularized=True)
    @pytest.mark.parametrize('x, c, cdf',
                             [(1, 2, 0.7546378854206702),
                              (-1, 14, 6.768116452566383e-18),
                              (-745.1, 0.001, 0.4749605142005238),
                              (-800, 0.001, 0.44958802911019136),
                              (-725, 0.1, 3.4301205868273265e-32),
                              (-740, 0.75, 1.0074360436599631e-241)])
    def test_cdf_ppf(self, x, c, cdf):
        p = stats.loggamma.cdf(x, c)
        assert_allclose(p, cdf, rtol=1e-13)
        y = stats.loggamma.ppf(cdf, c)
        assert_allclose(y, x, rtol=1e-13)

    # Expected sf values were computed with mpmath. For given x and c,
    #     x = mpmath.mpf(x)
    #     c = mpmath.mpf(c)
    #     sf = mpmath.gammainc(c, mpmath.exp(x), mpmath.inf,
    #                          regularized=True)
    @pytest.mark.parametrize('x, c, sf',
                             [(4, 1.5, 1.6341528919488565e-23),
                              (6, 100, 8.23836829202024e-74),
                              (-800, 0.001, 0.5504119708898086),
                              (-743, 0.0025, 0.8437131370024089)])
    def test_sf_isf(self, x, c, sf):
        s = stats.loggamma.sf(x, c)
        assert_allclose(s, sf, rtol=1e-13)
        y = stats.loggamma.isf(sf, c)
        assert_allclose(y, x, rtol=1e-13)

    def test_logpdf(self):
        # Test logpdf with x=-500, c=2.  ln(gamma(2)) = 0, and
        # exp(-500) ~= 7e-218, which is far smaller than the ULP
        # of c*x=-1000, so logpdf(-500, 2) = c*x - exp(x) - ln(gamma(2))
        # should give -1000.0.
        lp = stats.loggamma.logpdf(-500, 2)
        assert_allclose(lp, -1000.0, rtol=1e-14)

    def test_stats(self):
        # The following precomputed values are from the table in section 2.2
        # of "A Statistical Study of Log-Gamma Distribution", by Ping Shing
        # Chan (thesis, McMaster University, 1993).
        table = np.array([
                # c,    mean,   var,    skew,    exc. kurt.
                0.5, -1.9635, 4.9348, -1.5351, 4.0000,
                1.0, -0.5772, 1.6449, -1.1395, 2.4000,
                12.0, 2.4427, 0.0869, -0.2946, 0.1735,
            ]).reshape(-1, 5)
        for c, mean, var, skew, kurt in table:
            computed = stats.loggamma.stats(c, moments='msvk')
            assert_array_almost_equal(computed, [mean, var, skew, kurt],
                                      decimal=4)

    @pytest.mark.parametrize('c', [0.1, 0.001])
    def test_rvs(self, c):
        # Regression test for gh-11094.
        x = stats.loggamma.rvs(c, size=100000)
        # Before gh-11094 was fixed, the case with c=0.001 would
        # generate many -inf values.
        assert np.isfinite(x).all()
        # Crude statistical test.  About half the values should be
        # less than the median and half greater than the median.
        med = stats.loggamma.median(c)
        btest = stats.binomtest(np.count_nonzero(x < med), len(x))
        ci = btest.proportion_ci(confidence_level=0.999)
        assert ci.low < 0.5 < ci.high


class TestJohnsonsu:
    # reference values were computed via mpmath
    # from mpmath import mp
    # mp.dps = 50
    # def johnsonsu_sf(x, a, b):
    #     x = mp.mpf(x)
    #     a = mp.mpf(a)
    #     b = mp.mpf(b)
    #     return float(mp.ncdf(-(a + b * mp.log(x + mp.sqrt(x*x + 1)))))
    # Order is x, a, b, sf, isf tol
    # (Can't expect full precision when the ISF input is very nearly 1)
    cases = [(-500, 1, 1, 0.9999999982660072, 1e-8),
             (2000, 1, 1, 7.426351000595343e-21, 5e-14),
             (100000, 1, 1, 4.046923979269977e-40, 5e-14)]

    @pytest.mark.parametrize("case", cases)
    def test_sf_isf(self, case):
        x, a, b, sf, tol = case
        assert_allclose(stats.johnsonsu.sf(x, a, b), sf, rtol=5e-14)
        assert_allclose(stats.johnsonsu.isf(sf, a, b), x, rtol=tol)


class TestJohnsonb:
    # reference values were computed via mpmath
    # from mpmath import mp
    # mp.dps = 50
    # def johnsonb_sf(x, a, b):
    #     x = mp.mpf(x)
    #     a = mp.mpf(a)
    #     b = mp.mpf(b)
    #     return float(mp.ncdf(-(a + b * mp.log(x/(mp.one - x)))))
    # Order is x, a, b, sf, isf atol
    # (Can't expect full precision when the ISF input is very nearly 1)
    cases = [(1e-4, 1, 1, 0.9999999999999999, 1e-7),
             (0.9999, 1, 1, 8.921114313932308e-25, 5e-14),
             (0.999999, 1, 1, 5.815197487181902e-50, 5e-14)]

    @pytest.mark.parametrize("case", cases)
    def test_sf_isf(self, case):
        x, a, b, sf, tol = case
        assert_allclose(stats.johnsonsb.sf(x, a, b), sf, rtol=5e-14)
        assert_allclose(stats.johnsonsb.isf(sf, a, b), x, atol=tol)


class TestLogistic:
    # gh-6226
    def test_cdf_ppf(self):
        x = np.linspace(-20, 20)
        y = stats.logistic.cdf(x)
        xx = stats.logistic.ppf(y)
        assert_allclose(x, xx)

    def test_sf_isf(self):
        x = np.linspace(-20, 20)
        y = stats.logistic.sf(x)
        xx = stats.logistic.isf(y)
        assert_allclose(x, xx)

    def test_extreme_values(self):
        # p is chosen so that 1 - (1 - p) == p in double precision
        p = 9.992007221626409e-16
        desired = 34.53957599234088
        assert_allclose(stats.logistic.ppf(1 - p), desired)
        assert_allclose(stats.logistic.isf(p), desired)

    def test_logpdf_basic(self):
        logp = stats.logistic.logpdf([-15, 0, 10])
        # Expected values computed with mpmath with 50 digits of precision.
        expected = [-15.000000611804547,
                    -1.3862943611198906,
                    -10.000090797798434]
        assert_allclose(logp, expected, rtol=1e-13)

    def test_logpdf_extreme_values(self):
        logp = stats.logistic.logpdf([800, -800])
        # For such large arguments, logpdf(x) = -abs(x) when computed
        # with 64 bit floating point.
        assert_equal(logp, [-800, -800])

    @pytest.mark.parametrize("loc_rvs,scale_rvs", [(0.4484955, 0.10216821),
                                                   (0.62918191, 0.74367064)])
    def test_fit(self, loc_rvs, scale_rvs):
        data = stats.logistic.rvs(size=100, loc=loc_rvs, scale=scale_rvs)

        # test that result of fit method is the same as optimization
        def func(input, data):
            a, b = input
            n = len(data)
            x1 = np.sum(np.exp((data - a) / b) /
                        (1 + np.exp((data - a) / b))) - n / 2
            x2 = np.sum(((data - a) / b) *
                        ((np.exp((data - a) / b) - 1) /
                         (np.exp((data - a) / b) + 1))) - n
            return x1, x2

        expected_solution = root(func, stats.logistic._fitstart(data), args=(
            data,)).x
        fit_method = stats.logistic.fit(data)

        # other than computational variances, the fit method and the solution
        # to this system of equations are equal
        assert_allclose(fit_method, expected_solution, atol=1e-30)

    def test_fit_comp_optimizer(self):
        data = stats.logistic.rvs(size=100, loc=0.5, scale=2)
        _assert_less_or_close_loglike(stats.logistic, data)
        _assert_less_or_close_loglike(stats.logistic, data, floc=1)
        _assert_less_or_close_loglike(stats.logistic, data, fscale=1)

    @pytest.mark.parametrize('testlogcdf', [True, False])
    def test_logcdfsf_tails(self, testlogcdf):
        # Test either logcdf or logsf.  By symmetry, we can use the same
        # expected values for both by switching the sign of x for logsf.
        x = np.array([-10000, -800, 17, 50, 500])
        if testlogcdf:
            y = stats.logistic.logcdf(x)
        else:
            y = stats.logistic.logsf(-x)
        # The expected values were computed with mpmath.
        expected = [-10000.0, -800.0, -4.139937633089748e-08,
                    -1.9287498479639178e-22, -7.124576406741286e-218]
        assert_allclose(y, expected, rtol=2e-15)

    def test_fit_gh_18176(self):
        # logistic.fit returned `scale < 0` for this data. Check that this has
        # been fixed.
        data = np.array([-459, 37, 43, 45, 45, 48, 54, 55, 58]
                        + [59] * 3 + [61] * 9)
        # If scale were negative, NLLF would be infinite, so this would fail
        _assert_less_or_close_loglike(stats.logistic, data)


class TestLogser:
    def setup_method(self):
        np.random.seed(1234)

    def test_rvs(self):
        vals = stats.logser.rvs(0.75, size=(2, 50))
        assert_(numpy.all(vals >= 1))
        assert_(numpy.shape(vals) == (2, 50))
        assert_(vals.dtype.char in typecodes['AllInteger'])
        val = stats.logser.rvs(0.75)
        assert_(isinstance(val, int))
        val = stats.logser(0.75).rvs(3)
        assert_(isinstance(val, numpy.ndarray))
        assert_(val.dtype.char in typecodes['AllInteger'])

    def test_pmf_small_p(self):
        m = stats.logser.pmf(4, 1e-20)
        # The expected value was computed using mpmath:
        #   >>> import mpmath
        #   >>> mpmath.mp.dps = 64
        #   >>> k = 4
        #   >>> p = mpmath.mpf('1e-20')
        #   >>> float(-(p**k)/k/mpmath.log(1-p))
        #   2.5e-61
        # It is also clear from noticing that for very small p,
        # log(1-p) is approximately -p, and the formula becomes
        #    p**(k-1) / k
        assert_allclose(m, 2.5e-61)

    def test_mean_small_p(self):
        m = stats.logser.mean(1e-8)
        # The expected mean was computed using mpmath:
        #   >>> import mpmath
        #   >>> mpmath.dps = 60
        #   >>> p = mpmath.mpf('1e-8')
        #   >>> float(-p / ((1 - p)*mpmath.log(1 - p)))
        #   1.000000005
        assert_allclose(m, 1.000000005)


class TestGumbel_r_l:
    @pytest.fixture(scope='function')
    def rng(self):
        return np.random.default_rng(1234)

    @pytest.mark.parametrize("dist", [stats.gumbel_r, stats.gumbel_l])
    @pytest.mark.parametrize("loc_rvs", [-1, 0, 1])
    @pytest.mark.parametrize("scale_rvs", [.1, 1, 5])
    @pytest.mark.parametrize('fix_loc, fix_scale',
                             ([True, False], [False, True]))
    def test_fit_comp_optimizer(self, dist, loc_rvs, scale_rvs,
                                fix_loc, fix_scale, rng):
        data = dist.rvs(size=100, loc=loc_rvs, scale=scale_rvs,
                        random_state=rng)

        kwds = dict()
        # the fixed location and scales are arbitrarily modified to not be
        # close to the true value.
        if fix_loc:
            kwds['floc'] = loc_rvs * 2
        if fix_scale:
            kwds['fscale'] = scale_rvs * 2

        # test that the gumbel_* fit method is better than super method
        _assert_less_or_close_loglike(dist, data, **kwds)

    @pytest.mark.parametrize("dist, sgn", [(stats.gumbel_r, 1),
                                           (stats.gumbel_l, -1)])
    def test_fit(self, dist, sgn):
        z = sgn*np.array([3, 3, 3, 3, 3, 3, 3, 3.00000001])
        loc, scale = dist.fit(z)
        # The expected values were computed with mpmath with 60 digits
        # of precision.
        assert_allclose(loc, sgn*3.0000000001667906)
        assert_allclose(scale, 1.2495222465145514e-09, rtol=1e-6)


class TestPareto:
    def test_stats(self):
        # Check the stats() method with some simple values. Also check
        # that the calculations do not trigger RuntimeWarnings.
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)

            m, v, s, k = stats.pareto.stats(0.5, moments='mvsk')
            assert_equal(m, np.inf)
            assert_equal(v, np.inf)
            assert_equal(s, np.nan)
            assert_equal(k, np.nan)

            m, v, s, k = stats.pareto.stats(1.0, moments='mvsk')
            assert_equal(m, np.inf)
            assert_equal(v, np.inf)
            assert_equal(s, np.nan)
            assert_equal(k, np.nan)

            m, v, s, k = stats.pareto.stats(1.5, moments='mvsk')
            assert_equal(m, 3.0)
            assert_equal(v, np.inf)
            assert_equal(s, np.nan)
            assert_equal(k, np.nan)

            m, v, s, k = stats.pareto.stats(2.0, moments='mvsk')
            assert_equal(m, 2.0)
            assert_equal(v, np.inf)
            assert_equal(s, np.nan)
            assert_equal(k, np.nan)

            m, v, s, k = stats.pareto.stats(2.5, moments='mvsk')
            assert_allclose(m, 2.5 / 1.5)
            assert_allclose(v, 2.5 / (1.5*1.5*0.5))
            assert_equal(s, np.nan)
            assert_equal(k, np.nan)

            m, v, s, k = stats.pareto.stats(3.0, moments='mvsk')
            assert_allclose(m, 1.5)
            assert_allclose(v, 0.75)
            assert_equal(s, np.nan)
            assert_equal(k, np.nan)

            m, v, s, k = stats.pareto.stats(3.5, moments='mvsk')
            assert_allclose(m, 3.5 / 2.5)
            assert_allclose(v, 3.5 / (2.5*2.5*1.5))
            assert_allclose(s, (2*4.5/0.5)*np.sqrt(1.5/3.5))
            assert_equal(k, np.nan)

            m, v, s, k = stats.pareto.stats(4.0, moments='mvsk')
            assert_allclose(m, 4.0 / 3.0)
            assert_allclose(v, 4.0 / 18.0)
            assert_allclose(s, 2*(1+4.0)/(4.0-3) * np.sqrt((4.0-2)/4.0))
            assert_equal(k, np.nan)

            m, v, s, k = stats.pareto.stats(4.5, moments='mvsk')
            assert_allclose(m, 4.5 / 3.5)
            assert_allclose(v, 4.5 / (3.5*3.5*2.5))
            assert_allclose(s, (2*5.5/1.5) * np.sqrt(2.5/4.5))
            assert_allclose(k, 6*(4.5**3 + 4.5**2 - 6*4.5 - 2)/(4.5*1.5*0.5))

    def test_sf(self):
        x = 1e9
        b = 2
        scale = 1.5
        p = stats.pareto.sf(x, b, loc=0, scale=scale)
        expected = (scale/x)**b   # 2.25e-18
        assert_allclose(p, expected)

    @pytest.fixture(scope='function')
    def rng(self):
        return np.random.default_rng(1234)

    @pytest.mark.filterwarnings("ignore:invalid value encountered in "
                                "double_scalars")
    @pytest.mark.parametrize("rvs_shape", [1, 2])
    @pytest.mark.parametrize("rvs_loc", [0, 2])
    @pytest.mark.parametrize("rvs_scale", [1, 5])
    def test_fit(self, rvs_shape, rvs_loc, rvs_scale, rng):
        data = stats.pareto.rvs(size=100, b=rvs_shape, scale=rvs_scale,
                                loc=rvs_loc, random_state=rng)

        # shape can still be fixed with multiple names
        shape_mle_analytical1 = stats.pareto.fit(data, floc=0, f0=1.04)[0]
        shape_mle_analytical2 = stats.pareto.fit(data, floc=0, fix_b=1.04)[0]
        shape_mle_analytical3 = stats.pareto.fit(data, floc=0, fb=1.04)[0]
        assert (shape_mle_analytical1 == shape_mle_analytical2 ==
                shape_mle_analytical3 == 1.04)

        # data can be shifted with changes to `loc`
        data = stats.pareto.rvs(size=100, b=rvs_shape, scale=rvs_scale,
                                loc=(rvs_loc + 2), random_state=rng)
        shape_mle_a, loc_mle_a, scale_mle_a = stats.pareto.fit(data, floc=2)
        assert_equal(scale_mle_a + 2, data.min())

        data_shift = data - 2
        ndata = data_shift.shape[0]
        assert_equal(shape_mle_a,
                     ndata / np.sum(np.log(data_shift/data_shift.min())))
        assert_equal(loc_mle_a, 2)

    @pytest.mark.parametrize("rvs_shape", [.1, 2])
    @pytest.mark.parametrize("rvs_loc", [0, 2])
    @pytest.mark.parametrize("rvs_scale", [1, 5])
    @pytest.mark.parametrize('fix_shape, fix_loc, fix_scale',
                             [p for p in product([True, False], repeat=3)
                              if False in p])
    @np.errstate(invalid="ignore")
    def test_fit_MLE_comp_optimizer(self, rvs_shape, rvs_loc, rvs_scale,
                                    fix_shape, fix_loc, fix_scale, rng):
        data = stats.pareto.rvs(size=100, b=rvs_shape, scale=rvs_scale,
                                loc=rvs_loc, random_state=rng)

        kwds = {}
        if fix_shape:
            kwds['f0'] = rvs_shape
        if fix_loc:
            kwds['floc'] = rvs_loc
        if fix_scale:
            kwds['fscale'] = rvs_scale

        _assert_less_or_close_loglike(stats.pareto, data, **kwds)

    @np.errstate(invalid="ignore")
    def test_fit_known_bad_seed(self):
        # Tests a known seed and set of parameters that would produce a result
        # would violate the support of Pareto if the fit method did not check
        # the constraint `fscale + floc < min(data)`.
        shape, location, scale = 1, 0, 1
        data = stats.pareto.rvs(shape, location, scale, size=100,
                                random_state=np.random.default_rng(2535619))
        _assert_less_or_close_loglike(stats.pareto, data)

    def test_fit_warnings(self):
        assert_fit_warnings(stats.pareto)
        # `floc` that causes invalid negative data
        assert_raises(FitDataError, stats.pareto.fit, [1, 2, 3], floc=2)
        # `floc` and `fscale` combination causes invalid data
        assert_raises(FitDataError, stats.pareto.fit, [5, 2, 3], floc=1,
                      fscale=3)

    def test_negative_data(self, rng):
        data = stats.pareto.rvs(loc=-130, b=1, size=100, random_state=rng)
        assert_array_less(data, 0)
        # The purpose of this test is to make sure that no runtime warnings are
        # raised for all negative data, not the output of the fit method. Other
        # methods test the output but have to silence warnings from the super
        # method.
        _ = stats.pareto.fit(data)


class TestGenpareto:
    def test_ab(self):
        # c >= 0: a, b = [0, inf]
        for c in [1., 0.]:
            c = np.asarray(c)
            a, b = stats.genpareto._get_support(c)
            assert_equal(a, 0.)
            assert_(np.isposinf(b))

        # c < 0: a=0, b=1/|c|
        c = np.asarray(-2.)
        a, b = stats.genpareto._get_support(c)
        assert_allclose([a, b], [0., 0.5])

    def test_c0(self):
        # with c=0, genpareto reduces to the exponential distribution
        # rv = stats.genpareto(c=0.)
        rv = stats.genpareto(c=0.)
        x = np.linspace(0, 10., 30)
        assert_allclose(rv.pdf(x), stats.expon.pdf(x))
        assert_allclose(rv.cdf(x), stats.expon.cdf(x))
        assert_allclose(rv.sf(x), stats.expon.sf(x))

        q = np.linspace(0., 1., 10)
        assert_allclose(rv.ppf(q), stats.expon.ppf(q))

    def test_cm1(self):
        # with c=-1, genpareto reduces to the uniform distr on [0, 1]
        rv = stats.genpareto(c=-1.)
        x = np.linspace(0, 10., 30)
        assert_allclose(rv.pdf(x), stats.uniform.pdf(x))
        assert_allclose(rv.cdf(x), stats.uniform.cdf(x))
        assert_allclose(rv.sf(x), stats.uniform.sf(x))

        q = np.linspace(0., 1., 10)
        assert_allclose(rv.ppf(q), stats.uniform.ppf(q))

        # logpdf(1., c=-1) should be zero
        assert_allclose(rv.logpdf(1), 0)

    def test_x_inf(self):
        # make sure x=inf is handled gracefully
        rv = stats.genpareto(c=0.1)
        assert_allclose([rv.pdf(np.inf), rv.cdf(np.inf)], [0., 1.])
        assert_(np.isneginf(rv.logpdf(np.inf)))

        rv = stats.genpareto(c=0.)
        assert_allclose([rv.pdf(np.inf), rv.cdf(np.inf)], [0., 1.])
        assert_(np.isneginf(rv.logpdf(np.inf)))

        rv = stats.genpareto(c=-1.)
        assert_allclose([rv.pdf(np.inf), rv.cdf(np.inf)], [0., 1.])
        assert_(np.isneginf(rv.logpdf(np.inf)))

    def test_c_continuity(self):
        # pdf is continuous at c=0, -1
        x = np.linspace(0, 10, 30)
        for c in [0, -1]:
            pdf0 = stats.genpareto.pdf(x, c)
            for dc in [1e-14, -1e-14]:
                pdfc = stats.genpareto.pdf(x, c + dc)
                assert_allclose(pdf0, pdfc, atol=1e-12)

            cdf0 = stats.genpareto.cdf(x, c)
            for dc in [1e-14, 1e-14]:
                cdfc = stats.genpareto.cdf(x, c + dc)
                assert_allclose(cdf0, cdfc, atol=1e-12)

    def test_c_continuity_ppf(self):
        q = np.r_[np.logspace(1e-12, 0.01, base=0.1),
                  np.linspace(0.01, 1, 30, endpoint=False),
                  1. - np.logspace(1e-12, 0.01, base=0.1)]
        for c in [0., -1.]:
            ppf0 = stats.genpareto.ppf(q, c)
            for dc in [1e-14, -1e-14]:
                ppfc = stats.genpareto.ppf(q, c + dc)
                assert_allclose(ppf0, ppfc, atol=1e-12)

    def test_c_continuity_isf(self):
        q = np.r_[np.logspace(1e-12, 0.01, base=0.1),
                  np.linspace(0.01, 1, 30, endpoint=False),
                  1. - np.logspace(1e-12, 0.01, base=0.1)]
        for c in [0., -1.]:
            isf0 = stats.genpareto.isf(q, c)
            for dc in [1e-14, -1e-14]:
                isfc = stats.genpareto.isf(q, c + dc)
                assert_allclose(isf0, isfc, atol=1e-12)

    def test_cdf_ppf_roundtrip(self):
        # this should pass with machine precision. hat tip @pbrod
        q = np.r_[np.logspace(1e-12, 0.01, base=0.1),
                  np.linspace(0.01, 1, 30, endpoint=False),
                  1. - np.logspace(1e-12, 0.01, base=0.1)]
        for c in [1e-8, -1e-18, 1e-15, -1e-15]:
            assert_allclose(stats.genpareto.cdf(stats.genpareto.ppf(q, c), c),
                            q, atol=1e-15)

    def test_logsf(self):
        logp = stats.genpareto.logsf(1e10, .01, 0, 1)
        assert_allclose(logp, -1842.0680753952365)

    # Values in 'expected_stats' are
    # [mean, variance, skewness, excess kurtosis].
    @pytest.mark.parametrize(
        'c, expected_stats',
        [(0, [1, 1, 2, 6]),
         (1/4, [4/3, 32/9, 10/np.sqrt(2), np.nan]),
         (1/9, [9/8, (81/64)*(9/7), (10/9)*np.sqrt(7), 754/45]),
         (-1, [1/2, 1/12, 0, -6/5])])
    def test_stats(self, c, expected_stats):
        result = stats.genpareto.stats(c, moments='mvsk')
        assert_allclose(result, expected_stats, rtol=1e-13, atol=1e-15)

    def test_var(self):
        # Regression test for gh-11168.
        v = stats.genpareto.var(1e-8)
        assert_allclose(v, 1.000000040000001, rtol=1e-13)


class TestPearson3:
    def setup_method(self):
        np.random.seed(1234)

    def test_rvs(self):
        vals = stats.pearson3.rvs(0.1, size=(2, 50))
        assert_(numpy.shape(vals) == (2, 50))
        assert_(vals.dtype.char in typecodes['AllFloat'])
        val = stats.pearson3.rvs(0.5)
        assert_(isinstance(val, float))
        val = stats.pearson3(0.5).rvs(3)
        assert_(isinstance(val, numpy.ndarray))
        assert_(val.dtype.char in typecodes['AllFloat'])
        assert_(len(val) == 3)

    def test_pdf(self):
        vals = stats.pearson3.pdf(2, [0.0, 0.1, 0.2])
        assert_allclose(vals, np.array([0.05399097, 0.05555481, 0.05670246]),
                        atol=1e-6)
        vals = stats.pearson3.pdf(-3, 0.1)
        assert_allclose(vals, np.array([0.00313791]), atol=1e-6)
        vals = stats.pearson3.pdf([-3, -2, -1, 0, 1], 0.1)
        assert_allclose(vals, np.array([0.00313791, 0.05192304, 0.25028092,
                                        0.39885918, 0.23413173]), atol=1e-6)

    def test_cdf(self):
        vals = stats.pearson3.cdf(2, [0.0, 0.1, 0.2])
        assert_allclose(vals, np.array([0.97724987, 0.97462004, 0.97213626]),
                        atol=1e-6)
        vals = stats.pearson3.cdf(-3, 0.1)
        assert_allclose(vals, [0.00082256], atol=1e-6)
        vals = stats.pearson3.cdf([-3, -2, -1, 0, 1], 0.1)
        assert_allclose(vals, [8.22563821e-04, 1.99860448e-02, 1.58550710e-01,
                               5.06649130e-01, 8.41442111e-01], atol=1e-6)

    def test_negative_cdf_bug_11186(self):
        # incorrect CDFs for negative skews in gh-11186; fixed in gh-12640
        # Also check vectorization w/ negative, zero, and positive skews
        skews = [-3, -1, 0, 0.5]
        x_eval = 0.5
        neg_inf = -30  # avoid RuntimeWarning caused by np.log(0)
        cdfs = stats.pearson3.cdf(x_eval, skews)
        int_pdfs = [quad(stats.pearson3(skew).pdf, neg_inf, x_eval)[0]
                    for skew in skews]
        assert_allclose(cdfs, int_pdfs)

    def test_return_array_bug_11746(self):
        # pearson3.moment was returning size 0 or 1 array instead of float
        # The first moment is equal to the loc, which defaults to zero
        moment = stats.pearson3.moment(1, 2)
        assert_equal(moment, 0)
        assert isinstance(moment, np.number)

        moment = stats.pearson3.moment(1, 0.000001)
        assert_equal(moment, 0)
        assert isinstance(moment, np.number)

    def test_ppf_bug_17050(self):
        # incorrect PPF for negative skews were reported in gh-17050
        # Check that this is fixed (even in the array case)
        skews = [-3, -1, 0, 0.5]
        x_eval = 0.5
        res = stats.pearson3.ppf(stats.pearson3.cdf(x_eval, skews), skews)
        assert_allclose(res, x_eval)

        # Negation of the skew flips the distribution about the origin, so
        # the following should hold
        skew = np.array([[-0.5], [1.5]])
        x = np.linspace(-2, 2)
        assert_allclose(stats.pearson3.pdf(x, skew),
                        stats.pearson3.pdf(-x, -skew))
        assert_allclose(stats.pearson3.cdf(x, skew),
                        stats.pearson3.sf(-x, -skew))
        assert_allclose(stats.pearson3.ppf(x, skew),
                        -stats.pearson3.isf(x, -skew))


class TestKappa4:
    def test_cdf_genpareto(self):
        # h = 1 and k != 0 is generalized Pareto
        x = [0.0, 0.1, 0.2, 0.5]
        h = 1.0
        for k in [-1.9, -1.0, -0.5, -0.2, -0.1, 0.1, 0.2, 0.5, 1.0,
                  1.9]:
            vals = stats.kappa4.cdf(x, h, k)
            # shape parameter is opposite what is expected
            vals_comp = stats.genpareto.cdf(x, -k)
            assert_allclose(vals, vals_comp)

    def test_cdf_genextreme(self):
        # h = 0 and k != 0 is generalized extreme value
        x = np.linspace(-5, 5, 10)
        h = 0.0
        k = np.linspace(-3, 3, 10)
        vals = stats.kappa4.cdf(x, h, k)
        vals_comp = stats.genextreme.cdf(x, k)
        assert_allclose(vals, vals_comp)

    def test_cdf_expon(self):
        # h = 1 and k = 0 is exponential
        x = np.linspace(0, 10, 10)
        h = 1.0
        k = 0.0
        vals = stats.kappa4.cdf(x, h, k)
        vals_comp = stats.expon.cdf(x)
        assert_allclose(vals, vals_comp)

    def test_cdf_gumbel_r(self):
        # h = 0 and k = 0 is gumbel_r
        x = np.linspace(-5, 5, 10)
        h = 0.0
        k = 0.0
        vals = stats.kappa4.cdf(x, h, k)
        vals_comp = stats.gumbel_r.cdf(x)
        assert_allclose(vals, vals_comp)

    def test_cdf_logistic(self):
        # h = -1 and k = 0 is logistic
        x = np.linspace(-5, 5, 10)
        h = -1.0
        k = 0.0
        vals = stats.kappa4.cdf(x, h, k)
        vals_comp = stats.logistic.cdf(x)
        assert_allclose(vals, vals_comp)

    def test_cdf_uniform(self):
        # h = 1 and k = 1 is uniform
        x = np.linspace(-5, 5, 10)
        h = 1.0
        k = 1.0
        vals = stats.kappa4.cdf(x, h, k)
        vals_comp = stats.uniform.cdf(x)
        assert_allclose(vals, vals_comp)

    def test_integers_ctor(self):
        # regression test for gh-7416: _argcheck fails for integer h and k
        # in numpy 1.12
        stats.kappa4(1, 2)


class TestPoisson:
    def setup_method(self):
        np.random.seed(1234)

    def test_pmf_basic(self):
        # Basic case
        ln2 = np.log(2)
        vals = stats.poisson.pmf([0, 1, 2], ln2)
        expected = [0.5, ln2/2, ln2**2/4]
        assert_allclose(vals, expected)

    def test_mu0(self):
        # Edge case: mu=0
        vals = stats.poisson.pmf([0, 1, 2], 0)
        expected = [1, 0, 0]
        assert_array_equal(vals, expected)

        interval = stats.poisson.interval(0.95, 0)
        assert_equal(interval, (0, 0))

    def test_rvs(self):
        vals = stats.poisson.rvs(0.5, size=(2, 50))
        assert_(numpy.all(vals >= 0))
        assert_(numpy.shape(vals) == (2, 50))
        assert_(vals.dtype.char in typecodes['AllInteger'])
        val = stats.poisson.rvs(0.5)
        assert_(isinstance(val, int))
        val = stats.poisson(0.5).rvs(3)
        assert_(isinstance(val, numpy.ndarray))
        assert_(val.dtype.char in typecodes['AllInteger'])

    def test_stats(self):
        mu = 16.0
        result = stats.poisson.stats(mu, moments='mvsk')
        assert_allclose(result, [mu, mu, np.sqrt(1.0/mu), 1.0/mu])

        mu = np.array([0.0, 1.0, 2.0])
        result = stats.poisson.stats(mu, moments='mvsk')
        expected = (mu, mu, [np.inf, 1, 1/np.sqrt(2)], [np.inf, 1, 0.5])
        assert_allclose(result, expected)


class TestKSTwo:
    def setup_method(self):
        np.random.seed(1234)

    def test_cdf(self):
        for n in [1, 2, 3, 10, 100, 1000]:
            # Test x-values:
            #  0, 1/2n, where the cdf should be 0
            #  1/n, where the cdf should be n!/n^n
            #  0.5, where the cdf should match ksone.cdf
            # 1-1/n, where cdf = 1-2/n^n
            # 1, where cdf == 1
            # (E.g. Exact values given by Eqn 1 in Simard / L'Ecuyer)
            x = np.array([0, 0.5/n, 1/n, 0.5, 1-1.0/n, 1])
            v1 = (1.0/n)**n
            lg = scipy.special.gammaln(n+1)
            elg = (np.exp(lg) if v1 != 0 else 0)
            expected = np.array([0, 0, v1 * elg,
                                 1 - 2*stats.ksone.sf(0.5, n),
                                 max(1 - 2*v1, 0.0),
                                 1.0])
            vals_cdf = stats.kstwo.cdf(x, n)
            assert_allclose(vals_cdf, expected)

    def test_sf(self):
        x = np.linspace(0, 1, 11)
        for n in [1, 2, 3, 10, 100, 1000]:
            # Same x values as in test_cdf, and use sf = 1 - cdf
            x = np.array([0, 0.5/n, 1/n, 0.5, 1-1.0/n, 1])
            v1 = (1.0/n)**n
            lg = scipy.special.gammaln(n+1)
            elg = (np.exp(lg) if v1 != 0 else 0)
            expected = np.array([1.0, 1.0,
                                 1 - v1 * elg,
                                 2*stats.ksone.sf(0.5, n),
                                 min(2*v1, 1.0), 0])
            vals_sf = stats.kstwo.sf(x, n)
            assert_allclose(vals_sf, expected)

    def test_cdf_sqrtn(self):
        # For fixed a, cdf(a/sqrt(n), n) -> kstwobign(a) as n->infinity
        # cdf(a/sqrt(n), n) is an increasing function of n (and a)
        # Check that the function is indeed increasing (allowing for some
        # small floating point and algorithm differences.)
        x = np.linspace(0, 2, 11)[1:]
        ns = [50, 100, 200, 400, 1000, 2000]
        for _x in x:
            xn = _x / np.sqrt(ns)
            probs = stats.kstwo.cdf(xn, ns)
            diffs = np.diff(probs)
            assert_array_less(diffs, 1e-8)

    def test_cdf_sf(self):
        x = np.linspace(0, 1, 11)
        for n in [1, 2, 3, 10, 100, 1000]:
            vals_cdf = stats.kstwo.cdf(x, n)
            vals_sf = stats.kstwo.sf(x, n)
            assert_array_almost_equal(vals_cdf, 1 - vals_sf)

    def test_cdf_sf_sqrtn(self):
        x = np.linspace(0, 1, 11)
        for n in [1, 2, 3, 10, 100, 1000]:
            xn = x / np.sqrt(n)
            vals_cdf = stats.kstwo.cdf(xn, n)
            vals_sf = stats.kstwo.sf(xn, n)
            assert_array_almost_equal(vals_cdf, 1 - vals_sf)

    def test_ppf_of_cdf(self):
        x = np.linspace(0, 1, 11)
        for n in [1, 2, 3, 10, 100, 1000]:
            xn = x[x > 0.5/n]
            vals_cdf = stats.kstwo.cdf(xn, n)
            # CDFs close to 1 are better dealt with using the SF
            cond = (0 < vals_cdf) & (vals_cdf < 0.99)
            vals = stats.kstwo.ppf(vals_cdf, n)
            assert_allclose(vals[cond], xn[cond], rtol=1e-4)

    def test_isf_of_sf(self):
        x = np.linspace(0, 1, 11)
        for n in [1, 2, 3, 10, 100, 1000]:
            xn = x[x > 0.5/n]
            vals_isf = stats.kstwo.isf(xn, n)
            cond = (0 < vals_isf) & (vals_isf < 1.0)
            vals = stats.kstwo.sf(vals_isf, n)
            assert_allclose(vals[cond], xn[cond], rtol=1e-4)

    def test_ppf_of_cdf_sqrtn(self):
        x = np.linspace(0, 1, 11)
        for n in [1, 2, 3, 10, 100, 1000]:
            xn = (x / np.sqrt(n))[x > 0.5/n]
            vals_cdf = stats.kstwo.cdf(xn, n)
            cond = (0 < vals_cdf) & (vals_cdf < 1.0)
            vals = stats.kstwo.ppf(vals_cdf, n)
            assert_allclose(vals[cond], xn[cond])

    def test_isf_of_sf_sqrtn(self):
        x = np.linspace(0, 1, 11)
        for n in [1, 2, 3, 10, 100, 1000]:
            xn = (x / np.sqrt(n))[x > 0.5/n]
            vals_sf = stats.kstwo.sf(xn, n)
            # SFs close to 1 are better dealt with using the CDF
            cond = (0 < vals_sf) & (vals_sf < 0.95)
            vals = stats.kstwo.isf(vals_sf, n)
            assert_allclose(vals[cond], xn[cond])

    def test_ppf(self):
        probs = np.linspace(0, 1, 11)[1:]
        for n in [1, 2, 3, 10, 100, 1000]:
            xn = stats.kstwo.ppf(probs, n)
            vals_cdf = stats.kstwo.cdf(xn, n)
            assert_allclose(vals_cdf, probs)

    def test_simard_lecuyer_table1(self):
        # Compute the cdf for values near the mean of the distribution.
        # The mean u ~ log(2)*sqrt(pi/(2n))
        # Compute for x in [u/4, u/3, u/2, u, 2u, 3u]
        # This is the computation of Table 1 of Simard, R., L'Ecuyer, P. (2011)
        #  "Computing the Two-Sided Kolmogorov-Smirnov Distribution".
        # Except that the values below are not from the published table, but
        # were generated using an independent SageMath implementation of
        # Durbin's algorithm (with the exponentiation and scaling of
        # Marsaglia/Tsang/Wang's version) using 500 bit arithmetic.
        # Some of the values in the published table have relative
        # errors greater than 1e-4.
        ns = [10, 50, 100, 200, 500, 1000]
        ratios = np.array([1.0/4, 1.0/3, 1.0/2, 1, 2, 3])
        expected = np.array([
            [1.92155292e-08, 5.72933228e-05, 2.15233226e-02, 6.31566589e-01,
             9.97685592e-01, 9.99999942e-01],
            [2.28096224e-09, 1.99142563e-05, 1.42617934e-02, 5.95345542e-01,
             9.96177701e-01, 9.99998662e-01],
            [1.00201886e-09, 1.32673079e-05, 1.24608594e-02, 5.86163220e-01,
             9.95866877e-01, 9.99998240e-01],
            [4.93313022e-10, 9.52658029e-06, 1.12123138e-02, 5.79486872e-01,
             9.95661824e-01, 9.99997964e-01],
            [2.37049293e-10, 6.85002458e-06, 1.01309221e-02, 5.73427224e-01,
             9.95491207e-01, 9.99997750e-01],
            [1.56990874e-10, 5.71738276e-06, 9.59725430e-03, 5.70322692e-01,
             9.95409545e-01, 9.99997657e-01]
        ])
        for idx, n in enumerate(ns):
            x = ratios * np.log(2) * np.sqrt(np.pi/2/n)
            vals_cdf = stats.kstwo.cdf(x, n)
            assert_allclose(vals_cdf, expected[idx], rtol=1e-5)


class TestZipf:
    def setup_method(self):
        np.random.seed(1234)

    def test_rvs(self):
        vals = stats.zipf.rvs(1.5, size=(2, 50))
        assert_(numpy.all(vals >= 1))
        assert_(numpy.shape(vals) == (2, 50))
        assert_(vals.dtype.char in typecodes['AllInteger'])
        val = stats.zipf.rvs(1.5)
        assert_(isinstance(val, int))
        val = stats.zipf(1.5).rvs(3)
        assert_(isinstance(val, numpy.ndarray))
        assert_(val.dtype.char in typecodes['AllInteger'])

    def test_moments(self):
        # n-th moment is finite iff a > n + 1
        m, v = stats.zipf.stats(a=2.8)
        assert_(np.isfinite(m))
        assert_equal(v, np.inf)

        s, k = stats.zipf.stats(a=4.8, moments='sk')
        assert_(not np.isfinite([s, k]).all())


class TestDLaplace:
    def setup_method(self):
        np.random.seed(1234)

    def test_rvs(self):
        vals = stats.dlaplace.rvs(1.5, size=(2, 50))
        assert_(numpy.shape(vals) == (2, 50))
        assert_(vals.dtype.char in typecodes['AllInteger'])
        val = stats.dlaplace.rvs(1.5)
        assert_(isinstance(val, int))
        val = stats.dlaplace(1.5).rvs(3)
        assert_(isinstance(val, numpy.ndarray))
        assert_(val.dtype.char in typecodes['AllInteger'])
        assert_(stats.dlaplace.rvs(0.8) is not None)

    def test_stats(self):
        # compare the explicit formulas w/ direct summation using pmf
        a = 1.
        dl = stats.dlaplace(a)
        m, v, s, k = dl.stats('mvsk')

        N = 37
        xx = np.arange(-N, N+1)
        pp = dl.pmf(xx)
        m2, m4 = np.sum(pp*xx**2), np.sum(pp*xx**4)
        assert_equal((m, s), (0, 0))
        assert_allclose((v, k), (m2, m4/m2**2 - 3.), atol=1e-14, rtol=1e-8)

    def test_stats2(self):
        a = np.log(2.)
        dl = stats.dlaplace(a)
        m, v, s, k = dl.stats('mvsk')
        assert_equal((m, s), (0., 0.))
        assert_allclose((v, k), (4., 3.25))


class TestInvgauss:
    def setup_method(self):
        np.random.seed(1234)

    @pytest.mark.parametrize("rvs_mu,rvs_loc,rvs_scale",
                             [(2, 0, 1), (4.635, 4.362, 6.303)])
    def test_fit(self, rvs_mu, rvs_loc, rvs_scale):
        data = stats.invgauss.rvs(size=100, mu=rvs_mu,
                                  loc=rvs_loc, scale=rvs_scale)
        # Analytical MLEs are calculated with formula when `floc` is fixed
        mu, loc, scale = stats.invgauss.fit(data, floc=rvs_loc)

        data = data - rvs_loc
        mu_temp = np.mean(data)
        scale_mle = len(data) / (np.sum(data**(-1) - mu_temp**(-1)))
        mu_mle = mu_temp/scale_mle

        # `mu` and `scale` match analytical formula
        assert_allclose(mu_mle, mu, atol=1e-15, rtol=1e-15)
        assert_allclose(scale_mle, scale, atol=1e-15, rtol=1e-15)
        assert_equal(loc, rvs_loc)
        data = stats.invgauss.rvs(size=100, mu=rvs_mu,
                                  loc=rvs_loc, scale=rvs_scale)
        # fixed parameters are returned
        mu, loc, scale = stats.invgauss.fit(data, floc=rvs_loc - 1,
                                            fscale=rvs_scale + 1)
        assert_equal(rvs_scale + 1, scale)
        assert_equal(rvs_loc - 1, loc)

        # shape can still be fixed with multiple names
        shape_mle1 = stats.invgauss.fit(data, fmu=1.04)[0]
        shape_mle2 = stats.invgauss.fit(data, fix_mu=1.04)[0]
        shape_mle3 = stats.invgauss.fit(data, f0=1.04)[0]
        assert shape_mle1 == shape_mle2 == shape_mle3 == 1.04

    @pytest.mark.parametrize("rvs_mu,rvs_loc,rvs_scale",
                             [(2, 0, 1), (6.311, 3.225, 4.520)])
    def test_fit_MLE_comp_optimizer(self, rvs_mu, rvs_loc, rvs_scale):
        data = stats.invgauss.rvs(size=100, mu=rvs_mu,
                                  loc=rvs_loc, scale=rvs_scale)

        super_fit = super(type(stats.invgauss), stats.invgauss).fit
        # fitting without `floc` uses superclass fit method
        super_fitted = super_fit(data)
        invgauss_fit = stats.invgauss.fit(data)
        assert_equal(super_fitted, invgauss_fit)

        # fitting with `fmu` is uses superclass fit method
        super_fitted = super_fit(data, floc=0, fmu=2)
        invgauss_fit = stats.invgauss.fit(data, floc=0, fmu=2)
        assert_equal(super_fitted, invgauss_fit)

        # fixed `floc` uses analytical formula and provides better fit than
        # super method
        _assert_less_or_close_loglike(stats.invgauss, data, floc=rvs_loc)

        # fixed `floc` not resulting in invalid data < 0 uses analytical
        # formulas and provides a better fit than the super method
        assert np.all((data - (rvs_loc - 1)) > 0)
        _assert_less_or_close_loglike(stats.invgauss, data, floc=rvs_loc - 1)

        # fixed `floc` to an arbitrary number, 0, still provides a better fit
        # than the super method
        _assert_less_or_close_loglike(stats.invgauss, data, floc=0)

        # fixed `fscale` to an arbitrary number still provides a better fit
        # than the super method
        _assert_less_or_close_loglike(stats.invgauss, data, floc=rvs_loc,
                                      fscale=np.random.rand(1)[0])

    def test_fit_raise_errors(self):
        assert_fit_warnings(stats.invgauss)
        # FitDataError is raised when negative invalid data
        with pytest.raises(FitDataError):
            stats.invgauss.fit([1, 2, 3], floc=2)

    def test_cdf_sf(self):
        # Regression tests for gh-13614.
        # Ground truth from R's statmod library (pinvgauss), e.g.
        # library(statmod)
        # options(digits=15)
        # mu = c(4.17022005e-04, 7.20324493e-03, 1.14374817e-06,
        #        3.02332573e-03, 1.46755891e-03)
        # print(pinvgauss(5, mu, 1))

        # make sure a finite value is returned when mu is very small. see
        # GH-13614
        mu = [4.17022005e-04, 7.20324493e-03, 1.14374817e-06,
              3.02332573e-03, 1.46755891e-03]
        expected = [1, 1, 1, 1, 1]
        actual = stats.invgauss.cdf(0.4, mu=mu)
        assert_equal(expected, actual)

        # test if the function can distinguish small left/right tail
        # probabilities from zero.
        cdf_actual = stats.invgauss.cdf(0.001, mu=1.05)
        assert_allclose(cdf_actual, 4.65246506892667e-219)
        sf_actual = stats.invgauss.sf(110, mu=1.05)
        assert_allclose(sf_actual, 4.12851625944048e-25)

        # test if x does not cause numerical issues when mu is very small
        # and x is close to mu in value.

        # slightly smaller than mu
        actual = stats.invgauss.cdf(0.00009, 0.0001)
        assert_allclose(actual, 2.9458022894924e-26)

        # slightly bigger than mu
        actual = stats.invgauss.cdf(0.000102, 0.0001)
        assert_allclose(actual, 0.976445540507925)

    def test_logcdf_logsf(self):
        # Regression tests for improvements made in gh-13616.
        # Ground truth from R's statmod library (pinvgauss), e.g.
        # library(statmod)
        # options(digits=15)
        # print(pinvgauss(0.001, 1.05, 1, log.p=TRUE, lower.tail=FALSE))

        # test if logcdf and logsf can compute values too small to
        # be represented on the unlogged scale. See: gh-13616
        logcdf = stats.invgauss.logcdf(0.0001, mu=1.05)
        assert_allclose(logcdf, -5003.87872590367)
        logcdf = stats.invgauss.logcdf(110, 1.05)
        assert_allclose(logcdf, -4.12851625944087e-25)
        logsf = stats.invgauss.logsf(0.001, mu=1.05)
        assert_allclose(logsf, -4.65246506892676e-219)
        logsf = stats.invgauss.logsf(110, 1.05)
        assert_allclose(logsf, -56.1467092416426)

    # from mpmath import mp
    # mp.dps = 100
    # mu = mp.mpf(1e-2)
    # ref = (1/2 * mp.log(2 * mp.pi * mp.e * mu**3)
    #        - 3/2* mp.exp(2/mu) * mp.e1(2/mu))
    @pytest.mark.parametrize("mu, ref", [(2e-8, -25.172361826883957),
                                         (1e-3, -8.943444010642972),
                                         (1e-2, -5.4962796152622335),
                                         (1e8, 3.3244822568873476),
                                         (1e100, 3.32448280139689)])
    def test_entropy(self, mu, ref):
        assert_allclose(stats.invgauss.entropy(mu), ref, rtol=5e-14)


class TestLaplace:
    @pytest.mark.parametrize("rvs_loc", [-5, 0, 1, 2])
    @pytest.mark.parametrize("rvs_scale", [1, 2, 3, 10])
    def test_fit(self, rvs_loc, rvs_scale):
        # tests that various inputs follow expected behavior
        # for a variety of `loc` and `scale`.
        data = stats.laplace.rvs(size=100, loc=rvs_loc, scale=rvs_scale)

        # MLE estimates are given by
        loc_mle = np.median(data)
        scale_mle = np.sum(np.abs(data - loc_mle)) / len(data)

        # standard outputs should match analytical MLE formulas
        loc, scale = stats.laplace.fit(data)
        assert_allclose(loc, loc_mle, atol=1e-15, rtol=1e-15)
        assert_allclose(scale, scale_mle, atol=1e-15, rtol=1e-15)

        # fixed parameter should use analytical formula for other
        loc, scale = stats.laplace.fit(data, floc=loc_mle)
        assert_allclose(scale, scale_mle, atol=1e-15, rtol=1e-15)
        loc, scale = stats.laplace.fit(data, fscale=scale_mle)
        assert_allclose(loc, loc_mle)

        # test with non-mle fixed parameter
        # create scale with non-median loc
        loc = rvs_loc * 2
        scale_mle = np.sum(np.abs(data - loc)) / len(data)

        # fixed loc to non median, scale should match
        # scale calculation with modified loc
        loc, scale = stats.laplace.fit(data, floc=loc)
        assert_equal(scale_mle, scale)

        # fixed scale created with non median loc,
        # loc output should still be the data median.
        loc, scale = stats.laplace.fit(data, fscale=scale_mle)
        assert_equal(loc_mle, loc)

        # error raised when both `floc` and `fscale` are fixed
        assert_raises(RuntimeError, stats.laplace.fit, data, floc=loc_mle,
                      fscale=scale_mle)

        # error is raised with non-finite values
        assert_raises(ValueError, stats.laplace.fit, [np.nan])
        assert_raises(ValueError, stats.laplace.fit, [np.inf])

    @pytest.mark.parametrize("rvs_loc,rvs_scale", [(-5, 10),
                                                   (10, 5),
                                                   (0.5, 0.2)])
    def test_fit_MLE_comp_optimizer(self, rvs_loc, rvs_scale):
        data = stats.laplace.rvs(size=1000, loc=rvs_loc, scale=rvs_scale)

        # the log-likelihood function for laplace is given by
        def ll(loc, scale, data):
            return -1 * (- (len(data)) * np.log(2*scale) -
                         (1/scale)*np.sum(np.abs(data - loc)))

        # test that the objective function result of the analytical MLEs is
        # less than or equal to that of the numerically optimized estimate
        loc, scale = stats.laplace.fit(data)
        loc_opt, scale_opt = super(type(stats.laplace),
                                   stats.laplace).fit(data)
        ll_mle = ll(loc, scale, data)
        ll_opt = ll(loc_opt, scale_opt, data)
        assert ll_mle < ll_opt or np.allclose(ll_mle, ll_opt,
                                              atol=1e-15, rtol=1e-15)

    def test_fit_simple_non_random_data(self):
        data = np.array([1.0, 1.0, 3.0, 5.0, 8.0, 14.0])
        # with `floc` fixed to 6, scale should be 4.
        loc, scale = stats.laplace.fit(data, floc=6)
        assert_allclose(scale, 4, atol=1e-15, rtol=1e-15)
        # with `fscale` fixed to 6, loc should be 4.
        loc, scale = stats.laplace.fit(data, fscale=6)
        assert_allclose(loc, 4, atol=1e-15, rtol=1e-15)

    def test_sf_cdf_extremes(self):
        # These calculations should not generate warnings.
        x = 1000
        p0 = stats.laplace.cdf(-x)
        # The exact value is smaller than can be represented with
        # 64 bit floating point, so the exected result is 0.
        assert p0 == 0.0
        # The closest 64 bit floating point representation of the
        # exact value is 1.0.
        p1 = stats.laplace.cdf(x)
        assert p1 == 1.0

        p0 = stats.laplace.sf(x)
        # The exact value is smaller than can be represented with
        # 64 bit floating point, so the exected result is 0.
        assert p0 == 0.0
        # The closest 64 bit floating point representation of the
        # exact value is 1.0.
        p1 = stats.laplace.sf(-x)
        assert p1 == 1.0

    def test_sf(self):
        x = 200
        p = stats.laplace.sf(x)
        assert_allclose(p, np.exp(-x)/2, rtol=1e-13)

    def test_isf(self):
        p = 1e-25
        x = stats.laplace.isf(p)
        assert_allclose(x, -np.log(2*p), rtol=1e-13)


class TestPowerlaw:

    # In the following data, `sf` was computed with mpmath.
    @pytest.mark.parametrize('x, a, sf',
                             [(0.25, 2.0, 0.9375),
                              (0.99609375, 1/256, 1.528855235208108e-05)])
    def test_sf(self, x, a, sf):
        assert_allclose(stats.powerlaw.sf(x, a), sf, rtol=1e-15)

    @pytest.fixture(scope='function')
    def rng(self):
        return np.random.default_rng(1234)

    @pytest.mark.parametrize("rvs_shape", [.1, .5, .75, 1, 2])
    @pytest.mark.parametrize("rvs_loc", [-1, 0, 1])
    @pytest.mark.parametrize("rvs_scale", [.1, 1, 5])
    @pytest.mark.parametrize('fix_shape, fix_loc, fix_scale',
                             [p for p in product([True, False], repeat=3)
                              if False in p])
    def test_fit_MLE_comp_optimizer(self, rvs_shape, rvs_loc, rvs_scale,
                                    fix_shape, fix_loc, fix_scale, rng):
        data = stats.powerlaw.rvs(size=250, a=rvs_shape, loc=rvs_loc,
                                  scale=rvs_scale, random_state=rng)

        kwds = dict()
        if fix_shape:
            kwds['f0'] = rvs_shape
        if fix_loc:
            kwds['floc'] = np.nextafter(data.min(), -np.inf)
        if fix_scale:
            kwds['fscale'] = rvs_scale
        _assert_less_or_close_loglike(stats.powerlaw, data, **kwds)

    def test_problem_case(self):
        # An observed problem with the test method indicated that some fixed
        # scale values could cause bad results, this is now corrected.
        a = 2.50002862645130604506
        location = 0.0
        scale = 35.249023299873095

        data = stats.powerlaw.rvs(a=a, loc=location, scale=scale, size=100,
                                  random_state=np.random.default_rng(5))

        kwds = {'fscale': data.ptp() * 2}

        _assert_less_or_close_loglike(stats.powerlaw, data, **kwds)

    def test_fit_warnings(self):
        assert_fit_warnings(stats.powerlaw)
        # test for error when `fscale + floc <= np.max(data)` is not satisfied
        msg = r" Maximum likelihood estimation with 'powerlaw' requires"
        with assert_raises(FitDataError, match=msg):
            stats.powerlaw.fit([1, 2, 4], floc=0, fscale=3)

        # test for error when `data - floc >= 0`  is not satisfied
        msg = r" Maximum likelihood estimation with 'powerlaw' requires"
        with assert_raises(FitDataError, match=msg):
            stats.powerlaw.fit([1, 2, 4], floc=2)

        # test for fixed location not less than `min(data)`.
        msg = r" Maximum likelihood estimation with 'powerlaw' requires"
        with assert_raises(FitDataError, match=msg):
            stats.powerlaw.fit([1, 2, 4], floc=1)

        # test for when fixed scale is less than or equal to range of data
        msg = r"Negative or zero `fscale` is outside"
        with assert_raises(ValueError, match=msg):
            stats.powerlaw.fit([1, 2, 4], fscale=-3)

        # test for when fixed scale is less than or equal to range of data
        msg = r"`fscale` must be greater than the range of data."
        with assert_raises(ValueError, match=msg):
            stats.powerlaw.fit([1, 2, 4], fscale=3)

    def test_minimum_data_zero_gh17801(self):
        # gh-17801 reported an overflow error when the minimum value of the
        # data is zero. Check that this problem is resolved.
        data = [0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 6]
        dist = stats.powerlaw
        with np.errstate(over='ignore'):
            _assert_less_or_close_loglike(dist, data)


class TestPowerLogNorm:

    # reference values were computed via mpmath
    # from mpmath import mp
    # mp.dps = 80
    # def powerlognorm_sf_mp(x, c, s):
    #     x = mp.mpf(x)
    #     c = mp.mpf(c)
    #     s = mp.mpf(s)
    #     return mp.ncdf(-mp.log(x) / s)**c
    #
    # def powerlognormal_cdf_mp(x, c, s):
    #     return mp.one - powerlognorm_sf_mp(x, c, s)
    #
    # x, c, s = 100, 20, 1
    # print(float(powerlognorm_sf_mp(x, c, s)))

    @pytest.mark.parametrize("x, c, s, ref",
                             [(100, 20, 1, 1.9057100820561928e-114),
                              (1e-3, 20, 1, 0.9999999999507617),
                              (1e-3, 0.02, 1, 0.9999999999999508),
                              (1e22, 0.02, 1, 6.50744044621611e-12)])
    def test_sf(self, x, c, s, ref):
        assert_allclose(stats.powerlognorm.sf(x, c, s), ref, rtol=1e-13)

    # reference values were computed via mpmath using the survival
    # function above (passing in `ref` and getting `q`).
    @pytest.mark.parametrize("q, c, s, ref",
                             [(0.9999999587870905, 0.02, 1, 0.01),
                              (6.690376686108851e-233, 20, 1, 1000)])
    def test_isf(self, q, c, s, ref):
        assert_allclose(stats.powerlognorm.isf(q, c, s), ref, rtol=5e-11)

    @pytest.mark.parametrize("x, c, s, ref",
                             [(1e25, 0.02, 1, 0.9999999999999963),
                              (1e-6, 0.02, 1, 2.054921078040843e-45),
                              (1e-6, 200, 1, 2.0549210780408428e-41),
                              (0.3, 200, 1, 0.9999999999713368)])
    def test_cdf(self, x, c, s, ref):
        assert_allclose(stats.powerlognorm.cdf(x, c, s), ref, rtol=3e-14)

    # reference values were computed via mpmath
    # from mpmath import mp
    # mp.dps = 50
    # def powerlognorm_pdf_mpmath(x, c, s):
    #     x = mp.mpf(x)
    #     c = mp.mpf(c)
    #     s = mp.mpf(s)
    #     res = (c/(x * s) * mp.npdf(mp.log(x)/s) *
    #            mp.ncdf(-mp.log(x)/s)**(c - mp.one))
    #     return float(res)

    @pytest.mark.parametrize("x, c, s, ref",
                             [(1e22, 0.02, 1, 6.5954987852335016e-34),
                              (1e20, 1e-3, 1, 1.588073750563988e-22),
                              (1e40, 1e-3, 1, 1.3179391812506349e-43)])
    def test_pdf(self, x, c, s, ref):
        assert_allclose(stats.powerlognorm.pdf(x, c, s), ref, rtol=3e-12)


class TestPowerNorm:

    # survival function references were computed with mpmath via
    # from mpmath import mp
    # x = mp.mpf(x)
    # c = mp.mpf(x)
    # float(mp.ncdf(-x)**c)

    @pytest.mark.parametrize("x, c, ref",
                             [(9, 1, 1.1285884059538405e-19),
                              (20, 2, 7.582445786569958e-178),
                              (100, 0.02, 3.330957891903866e-44),
                              (200, 0.01, 1.3004759092324774e-87)])
    def test_sf(self, x, c, ref):
        assert_allclose(stats.powernorm.sf(x, c), ref, rtol=1e-13)

    # inverse survival function references were computed with mpmath via
    # from mpmath import mp
    # def isf_mp(q, c):
    #     q = mp.mpf(q)
    #     c = mp.mpf(c)
    #     arg = q**(mp.one / c)
    #     return float(-mp.sqrt(2) * mp.erfinv(mp.mpf(2.) * arg - mp.one))

    @pytest.mark.parametrize("q, c, ref",
                             [(1e-5, 20, -0.15690800666514138),
                              (0.99999, 100, -5.19933666203545),
                              (0.9999, 0.02, -2.576676052143387),
                              (5e-2, 0.02, 17.089518110222244),
                              (1e-18, 2, 5.9978070150076865),
                              (1e-50, 5, 6.361340902404057)])
    def test_isf(self, q, c, ref):
        assert_allclose(stats.powernorm.isf(q, c), ref, rtol=5e-12)

    # CDF reference values were computed with mpmath via
    # from mpmath import mp
    # def cdf_mp(x, c):
    #     x = mp.mpf(x)
    #     c = mp.mpf(c)
    #     return float(mp.one - mp.ncdf(-x)**c)

    @pytest.mark.parametrize("x, c, ref",
                             [(-12, 9, 1.598833900869911e-32),
                              (2, 9, 0.9999999999999983),
                              (-20, 9, 2.4782617067456103e-88),
                              (-5, 0.02, 5.733032242841443e-09),
                              (-20, 0.02, 5.507248237212467e-91)])
    def test_cdf(self, x, c, ref):
        assert_allclose(stats.powernorm.cdf(x, c), ref, rtol=5e-14)


class TestInvGamma:
    def test_invgamma_inf_gh_1866(self):
        # invgamma's moments are only finite for a>n
        # specific numbers checked w/ boost 1.54
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            mvsk = stats.invgamma.stats(a=19.31, moments='mvsk')
            expected = [0.05461496450, 0.0001723162534, 1.020362676,
                        2.055616582]
            assert_allclose(mvsk, expected)

            a = [1.1, 3.1, 5.6]
            mvsk = stats.invgamma.stats(a=a, moments='mvsk')
            expected = ([10., 0.476190476, 0.2173913043],       # mmm
                        [np.inf, 0.2061430632, 0.01312749422],  # vvv
                        [np.nan, 41.95235392, 2.919025532],     # sss
                        [np.nan, np.nan, 24.51923076])          # kkk
            for x, y in zip(mvsk, expected):
                assert_almost_equal(x, y)

    def test_cdf_ppf(self):
        # gh-6245
        x = np.logspace(-2.6, 0)
        y = stats.invgamma.cdf(x, 1)
        xx = stats.invgamma.ppf(y, 1)
        assert_allclose(x, xx)

    def test_sf_isf(self):
        # gh-6245
        if sys.maxsize > 2**32:
            x = np.logspace(2, 100)
        else:
            # Invgamme roundtrip on 32-bit systems has relative accuracy
            # ~1e-15 until x=1e+15, and becomes inf above x=1e+18
            x = np.logspace(2, 18)

        y = stats.invgamma.sf(x, 1)
        xx = stats.invgamma.isf(y, 1)
        assert_allclose(x, xx, rtol=1.0)

    @pytest.mark.parametrize("a, ref",
                             [(100000000.0, -26.21208257605721),
                              (1e+100, -343.9688254159022)])
    def test_large_entropy(self, a, ref):
        # The reference values were calculated with mpmath:
        # from mpmath import mp
        # mp.dps = 500

        # def invgamma_entropy(a):
        #     a = mp.mpf(a)
        #     h = a + mp.loggamma(a) - (mp.one + a) * mp.digamma(a)
        #     return float(h)
        assert_allclose(stats.invgamma.entropy(a), ref, rtol=1e-15)


class TestF:
    def test_endpoints(self):
        # Compute the pdf at the left endpoint dst.a.
        data = [[stats.f, (2, 1), 1.0]]
        for _f, _args, _correct in data:
            ans = _f.pdf(_f.a, *_args)

        ans = [_f.pdf(_f.a, *_args) for _f, _args, _ in data]
        correct = [_correct_ for _f, _args, _correct_ in data]
        assert_array_almost_equal(ans, correct)

    def test_f_moments(self):
        # n-th moment of F distributions is only finite for n < dfd / 2
        m, v, s, k = stats.f.stats(11, 6.5, moments='mvsk')
        assert_(np.isfinite(m))
        assert_(np.isfinite(v))
        assert_(np.isfinite(s))
        assert_(not np.isfinite(k))

    def test_moments_warnings(self):
        # no warnings should be generated for dfd = 2, 4, 6, 8 (div by zero)
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            stats.f.stats(dfn=[11]*4, dfd=[2, 4, 6, 8], moments='mvsk')

    def test_stats_broadcast(self):
        dfn = np.array([[3], [11]])
        dfd = np.array([11, 12])
        m, v, s, k = stats.f.stats(dfn=dfn, dfd=dfd, moments='mvsk')
        m2 = [dfd / (dfd - 2)]*2
        assert_allclose(m, m2)
        v2 = 2 * dfd**2 * (dfn + dfd - 2) / dfn / (dfd - 2)**2 / (dfd - 4)
        assert_allclose(v, v2)
        s2 = ((2*dfn + dfd - 2) * np.sqrt(8*(dfd - 4)) /
              ((dfd - 6) * np.sqrt(dfn*(dfn + dfd - 2))))
        assert_allclose(s, s2)
        k2num = 12 * (dfn * (5*dfd - 22) * (dfn + dfd - 2) +
                      (dfd - 4) * (dfd - 2)**2)
        k2den = dfn * (dfd - 6) * (dfd - 8) * (dfn + dfd - 2)
        k2 = k2num / k2den
        assert_allclose(k, k2)


class TestStudentT:
    def test_rvgeneric_std(self):
        # Regression test for #1191
        assert_array_almost_equal(stats.t.std([5, 6]), [1.29099445, 1.22474487])

    def test_moments_t(self):
        # regression test for #8786
        assert_equal(stats.t.stats(df=1, moments='mvsk'),
                    (np.inf, np.nan, np.nan, np.nan))
        assert_equal(stats.t.stats(df=1.01, moments='mvsk'),
                    (0.0, np.inf, np.nan, np.nan))
        assert_equal(stats.t.stats(df=2, moments='mvsk'),
                    (0.0, np.inf, np.nan, np.nan))
        assert_equal(stats.t.stats(df=2.01, moments='mvsk'),
                    (0.0, 2.01/(2.01-2.0), np.nan, np.inf))
        assert_equal(stats.t.stats(df=3, moments='sk'), (np.nan, np.inf))
        assert_equal(stats.t.stats(df=3.01, moments='sk'), (0.0, np.inf))
        assert_equal(stats.t.stats(df=4, moments='sk'), (0.0, np.inf))
        assert_equal(stats.t.stats(df=4.01, moments='sk'), (0.0, 6.0/(4.01 - 4.0)))

    def test_t_entropy(self):
        df = [1, 2, 25, 100]
        # Expected values were computed with mpmath.
        expected = [2.5310242469692907, 1.9602792291600821,
                    1.459327578078393, 1.4289633653182439]
        assert_allclose(stats.t.entropy(df), expected, rtol=1e-13)

    @pytest.mark.parametrize("v, ref",
                            [(100, 1.4289633653182439),
                            (1e+100, 1.4189385332046727)])
    def test_t_extreme_entropy(self, v, ref):
        # Reference values were calculated with mpmath:
        # from mpmath import mp
        # mp.dps = 500
        #
        # def t_entropy(v):
        #   v = mp.mpf(v)
        #   C = (v + mp.one) / 2
        #   A = C * (mp.digamma(C) - mp.digamma(v / 2))
        #   B = 0.5 * mp.log(v) + mp.log(mp.beta(v / 2, mp.one / 2))
        #   h = A + B
        #   return float(h)
        assert_allclose(stats.t.entropy(v), ref, rtol=1e-14)

    @pytest.mark.parametrize("methname", ["pdf", "logpdf", "cdf",
                                        "ppf", "sf", "isf"])
    @pytest.mark.parametrize("df_infmask", [[0, 0], [1, 1], [0, 1],
                                            [[0, 1, 0], [1, 1, 1]],
                                            [[1, 0], [0, 1]],
                                            [[0], [1]]])
    def test_t_inf_df(self, methname, df_infmask):
        np.random.seed(0)
        df_infmask = np.asarray(df_infmask, dtype=bool)
        df = np.random.uniform(0, 10, size=df_infmask.shape)
        x = np.random.randn(*df_infmask.shape)
        df[df_infmask] = np.inf
        t_dist = stats.t(df=df, loc=3, scale=1)
        t_dist_ref = stats.t(df=df[~df_infmask], loc=3, scale=1)
        norm_dist = stats.norm(loc=3, scale=1)
        t_meth = getattr(t_dist, methname)
        t_meth_ref = getattr(t_dist_ref, methname)
        norm_meth = getattr(norm_dist, methname)
        res = t_meth(x)
        assert_equal(res[df_infmask], norm_meth(x[df_infmask]))
        assert_equal(res[~df_infmask], t_meth_ref(x[~df_infmask]))

    @pytest.mark.parametrize("df_infmask", [[0, 0], [1, 1], [0, 1],
                                            [[0, 1, 0], [1, 1, 1]],
                                            [[1, 0], [0, 1]],
                                            [[0], [1]]])
    def test_t_inf_df_stats_entropy(self, df_infmask):
        np.random.seed(0)
        df_infmask = np.asarray(df_infmask, dtype=bool)
        df = np.random.uniform(0, 10, size=df_infmask.shape)
        df[df_infmask] = np.inf
        res = stats.t.stats(df=df, loc=3, scale=1, moments='mvsk')
        res_ex_inf = stats.norm.stats(loc=3, scale=1, moments='mvsk')
        res_ex_noinf = stats.t.stats(df=df[~df_infmask], loc=3, scale=1,
                                    moments='mvsk')
        for i in range(4):
            assert_equal(res[i][df_infmask], res_ex_inf[i])
            assert_equal(res[i][~df_infmask], res_ex_noinf[i])

        res = stats.t.entropy(df=df, loc=3, scale=1)
        res_ex_inf = stats.norm.entropy(loc=3, scale=1)
        res_ex_noinf = stats.t.entropy(df=df[~df_infmask], loc=3, scale=1)
        assert_equal(res[df_infmask], res_ex_inf)
        assert_equal(res[~df_infmask], res_ex_noinf)


    def test_logpdf_pdf(self):
        # reference values were computed via the reference distribution, e.g.
        # mp.dps = 500; StudentT(df=df).logpdf(x), StudentT(df=df).pdf(x)
        x = [1, 1e3, 10, 1]
        df = [1e100, 1e50, 1e20, 1]
        logpdf_ref = [-1.4189385332046727, -500000.9189385332,
                      -50.918938533204674, -1.8378770664093456]
        pdf_ref = [0.24197072451914334, 0,
                   7.69459862670642e-23, 0.15915494309189535]
        assert_allclose(stats.t.logpdf(x, df), logpdf_ref, rtol=1e-15)
        assert_allclose(stats.t.pdf(x, df), pdf_ref, rtol=1e-14)


class TestRvDiscrete:
    def setup_method(self):
        np.random.seed(1234)

    def test_rvs(self):
        states = [-1, 0, 1, 2, 3, 4]
        probability = [0.0, 0.3, 0.4, 0.0, 0.3, 0.0]
        samples = 1000
        r = stats.rv_discrete(name='sample', values=(states, probability))
        x = r.rvs(size=samples)
        assert_(isinstance(x, numpy.ndarray))

        for s, p in zip(states, probability):
            assert_(abs(sum(x == s)/float(samples) - p) < 0.05)

        x = r.rvs()
        assert np.issubdtype(type(x), np.integer)

    def test_entropy(self):
        # Basic tests of entropy.
        pvals = np.array([0.25, 0.45, 0.3])
        p = stats.rv_discrete(values=([0, 1, 2], pvals))
        expected_h = -sum(xlogy(pvals, pvals))
        h = p.entropy()
        assert_allclose(h, expected_h)

        p = stats.rv_discrete(values=([0, 1, 2], [1.0, 0, 0]))
        h = p.entropy()
        assert_equal(h, 0.0)

    def test_pmf(self):
        xk = [1, 2, 4]
        pk = [0.5, 0.3, 0.2]
        rv = stats.rv_discrete(values=(xk, pk))

        x = [[1., 4.],
             [3., 2]]
        assert_allclose(rv.pmf(x),
                        [[0.5, 0.2],
                         [0., 0.3]], atol=1e-14)

    def test_cdf(self):
        xk = [1, 2, 4]
        pk = [0.5, 0.3, 0.2]
        rv = stats.rv_discrete(values=(xk, pk))

        x_values = [-2, 1., 1.1, 1.5, 2.0, 3.0, 4, 5]
        expected = [0, 0.5, 0.5, 0.5, 0.8, 0.8, 1, 1]
        assert_allclose(rv.cdf(x_values), expected, atol=1e-14)

        # also check scalar arguments
        assert_allclose([rv.cdf(xx) for xx in x_values],
                        expected, atol=1e-14)

    def test_ppf(self):
        xk = [1, 2, 4]
        pk = [0.5, 0.3, 0.2]
        rv = stats.rv_discrete(values=(xk, pk))

        q_values = [0.1, 0.5, 0.6, 0.8, 0.9, 1.]
        expected = [1, 1, 2, 2, 4, 4]
        assert_allclose(rv.ppf(q_values), expected, atol=1e-14)

        # also check scalar arguments
        assert_allclose([rv.ppf(q) for q in q_values],
                        expected, atol=1e-14)

    def test_cdf_ppf_next(self):
        # copied and special cased from test_discrete_basic
        vals = ([1, 2, 4, 7, 8], [0.1, 0.2, 0.3, 0.3, 0.1])
        rv = stats.rv_discrete(values=vals)

        assert_array_equal(rv.ppf(rv.cdf(rv.xk[:-1]) + 1e-8),
                           rv.xk[1:])

    def test_multidimension(self):
        xk = np.arange(12).reshape((3, 4))
        pk = np.array([[0.1, 0.1, 0.15, 0.05],
                       [0.1, 0.1, 0.05, 0.05],
                       [0.1, 0.1, 0.05, 0.05]])
        rv = stats.rv_discrete(values=(xk, pk))

        assert_allclose(rv.expect(), np.sum(rv.xk * rv.pk), atol=1e-14)

    def test_bad_input(self):
        xk = [1, 2, 3]
        pk = [0.5, 0.5]
        assert_raises(ValueError, stats.rv_discrete, **dict(values=(xk, pk)))

        pk = [1, 2, 3]
        assert_raises(ValueError, stats.rv_discrete, **dict(values=(xk, pk)))

        xk = [1, 2, 3]
        pk = [0.5, 1.2, -0.7]
        assert_raises(ValueError, stats.rv_discrete, **dict(values=(xk, pk)))

        xk = [1, 2, 3, 4, 5]
        pk = [0.3, 0.3, 0.3, 0.3, -0.2]
        assert_raises(ValueError, stats.rv_discrete, **dict(values=(xk, pk)))

    def test_shape_rv_sample(self):
        # tests added for gh-9565

        # mismatch of 2d inputs
        xk, pk = np.arange(4).reshape((2, 2)), np.full((2, 3), 1/6)
        assert_raises(ValueError, stats.rv_discrete, **dict(values=(xk, pk)))

        # same number of elements, but shapes not compatible
        xk, pk = np.arange(6).reshape((3, 2)), np.full((2, 3), 1/6)
        assert_raises(ValueError, stats.rv_discrete, **dict(values=(xk, pk)))

        # same shapes => no error
        xk, pk = np.arange(6).reshape((3, 2)), np.full((3, 2), 1/6)
        assert_equal(stats.rv_discrete(values=(xk, pk)).pmf(0), 1/6)

    def test_expect1(self):
        xk = [1, 2, 4, 6, 7, 11]
        pk = [0.1, 0.2, 0.2, 0.2, 0.2, 0.1]
        rv = stats.rv_discrete(values=(xk, pk))

        assert_allclose(rv.expect(), np.sum(rv.xk * rv.pk), atol=1e-14)

    def test_expect2(self):
        # rv_sample should override _expect. Bug report from
        # https://stackoverflow.com/questions/63199792
        y = [200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0,
             1100.0, 1200.0, 1300.0, 1400.0, 1500.0, 1600.0, 1700.0, 1800.0,
             1900.0, 2000.0, 2100.0, 2200.0, 2300.0, 2400.0, 2500.0, 2600.0,
             2700.0, 2800.0, 2900.0, 3000.0, 3100.0, 3200.0, 3300.0, 3400.0,
             3500.0, 3600.0, 3700.0, 3800.0, 3900.0, 4000.0, 4100.0, 4200.0,
             4300.0, 4400.0, 4500.0, 4600.0, 4700.0, 4800.0]

        py = [0.0004, 0.0, 0.0033, 0.006500000000000001, 0.0, 0.0,
              0.004399999999999999, 0.6862, 0.0, 0.0, 0.0,
              0.00019999999999997797, 0.0006000000000000449,
              0.024499999999999966, 0.006400000000000072,
              0.0043999999999999595, 0.019499999999999962,
              0.03770000000000007, 0.01759999999999995, 0.015199999999999991,
              0.018100000000000005, 0.04500000000000004, 0.0025999999999999357,
              0.0, 0.0041000000000001036, 0.005999999999999894,
              0.0042000000000000925, 0.0050000000000000044,
              0.0041999999999999815, 0.0004999999999999449,
              0.009199999999999986, 0.008200000000000096,
              0.0, 0.0, 0.0046999999999999265, 0.0019000000000000128,
              0.0006000000000000449, 0.02510000000000001, 0.0,
              0.007199999999999984, 0.0, 0.012699999999999934, 0.0, 0.0,
              0.008199999999999985, 0.005600000000000049, 0.0]

        rv = stats.rv_discrete(values=(y, py))

        # check the mean
        assert_allclose(rv.expect(), rv.mean(), atol=1e-14)
        assert_allclose(rv.expect(),
                        sum(v * w for v, w in zip(y, py)), atol=1e-14)

        # also check the second moment
        assert_allclose(rv.expect(lambda x: x**2),
                        sum(v**2 * w for v, w in zip(y, py)), atol=1e-14)


class TestSkewCauchy:
    def test_cauchy(self):
        x = np.linspace(-5, 5, 100)
        assert_array_almost_equal(stats.skewcauchy.pdf(x, a=0),
                                  stats.cauchy.pdf(x))
        assert_array_almost_equal(stats.skewcauchy.cdf(x, a=0),
                                  stats.cauchy.cdf(x))
        assert_array_almost_equal(stats.skewcauchy.ppf(x, a=0),
                                  stats.cauchy.ppf(x))

    def test_skewcauchy_R(self):
        # options(digits=16)
        # library(sgt)
        # # lmbda, x contain the values generated for a, x below
        # lmbda <- c(0.0976270078546495, 0.430378732744839, 0.2055267521432877,
        #            0.0897663659937937, -0.15269040132219, 0.2917882261333122,
        #            -0.12482557747462, 0.7835460015641595, 0.9273255210020589,
        #            -0.2331169623484446)
        # x <- c(2.917250380826646, 0.2889491975290444, 0.6804456109393229,
        #        4.25596638292661, -4.289639418021131, -4.1287070029845925,
        #        -4.797816025596743, 3.32619845547938, 2.7815675094985046,
        #        3.700121482468191)
        # pdf = dsgt(x, mu=0, lambda=lambda, sigma=1, q=1/2, mean.cent=FALSE,
        #            var.adj = sqrt(2))
        # cdf = psgt(x, mu=0, lambda=lambda, sigma=1, q=1/2, mean.cent=FALSE,
        #            var.adj = sqrt(2))
        # qsgt(cdf, mu=0, lambda=lambda, sigma=1, q=1/2, mean.cent=FALSE,
        #      var.adj = sqrt(2))

        np.random.seed(0)
        a = np.random.rand(10) * 2 - 1
        x = np.random.rand(10) * 10 - 5
        pdf = [0.039473975217333909, 0.305829714049903223, 0.24140158118994162,
               0.019585772402693054, 0.021436553695989482, 0.00909817103867518,
               0.01658423410016873, 0.071083288030394126, 0.103250045941454524,
               0.013110230778426242]
        cdf = [0.87426677718213752, 0.37556468910780882, 0.59442096496538066,
               0.91304659850890202, 0.09631964100300605, 0.03829624330921733,
               0.08245240578402535, 0.72057062945510386, 0.62826415852515449,
               0.95011308463898292]
        assert_allclose(stats.skewcauchy.pdf(x, a), pdf)
        assert_allclose(stats.skewcauchy.cdf(x, a), cdf)
        assert_allclose(stats.skewcauchy.ppf(cdf, a), x)


# Test data for TestSkewNorm.test_noncentral_moments()
# The expected noncentral moments were computed by Wolfram Alpha.
# In Wolfram Alpha, enter
#    SkewNormalDistribution[0, 1, a] moment
# with `a` replaced by the desired shape parameter.  In the results, there
# should be a table of the first four moments. Click on "More" to get more
# moments.  The expected moments start with the first moment (order = 1).
_skewnorm_noncentral_moments = [
    (2, [2*np.sqrt(2/(5*np.pi)),
         1,
         22/5*np.sqrt(2/(5*np.pi)),
         3,
         446/25*np.sqrt(2/(5*np.pi)),
         15,
         2682/25*np.sqrt(2/(5*np.pi)),
         105,
         107322/125*np.sqrt(2/(5*np.pi))]),
    (0.1, [np.sqrt(2/(101*np.pi)),
           1,
           302/101*np.sqrt(2/(101*np.pi)),
           3,
           (152008*np.sqrt(2/(101*np.pi)))/10201,
           15,
           (107116848*np.sqrt(2/(101*np.pi)))/1030301,
           105,
           (97050413184*np.sqrt(2/(101*np.pi)))/104060401]),
    (-3, [-3/np.sqrt(5*np.pi),
          1,
          -63/(10*np.sqrt(5*np.pi)),
          3,
          -2529/(100*np.sqrt(5*np.pi)),
          15,
          -30357/(200*np.sqrt(5*np.pi)),
          105,
          -2428623/(2000*np.sqrt(5*np.pi)),
          945,
          -242862867/(20000*np.sqrt(5*np.pi)),
          10395,
          -29143550277/(200000*np.sqrt(5*np.pi)),
          135135]),
]


class TestSkewNorm:
    def setup_method(self):
        self.rng = check_random_state(1234)

    def test_normal(self):
        # When the skewness is 0 the distribution is normal
        x = np.linspace(-5, 5, 100)
        assert_array_almost_equal(stats.skewnorm.pdf(x, a=0),
                                  stats.norm.pdf(x))

    def test_rvs(self):
        shape = (3, 4, 5)
        x = stats.skewnorm.rvs(a=0.75, size=shape, random_state=self.rng)
        assert_equal(shape, x.shape)

        x = stats.skewnorm.rvs(a=-3, size=shape, random_state=self.rng)
        assert_equal(shape, x.shape)

    def test_moments(self):
        X = stats.skewnorm.rvs(a=4, size=int(1e6), loc=5, scale=2,
                               random_state=self.rng)
        expected = [np.mean(X), np.var(X), stats.skew(X), stats.kurtosis(X)]
        computed = stats.skewnorm.stats(a=4, loc=5, scale=2, moments='mvsk')
        assert_array_almost_equal(computed, expected, decimal=2)

        X = stats.skewnorm.rvs(a=-4, size=int(1e6), loc=5, scale=2,
                               random_state=self.rng)
        expected = [np.mean(X), np.var(X), stats.skew(X), stats.kurtosis(X)]
        computed = stats.skewnorm.stats(a=-4, loc=5, scale=2, moments='mvsk')
        assert_array_almost_equal(computed, expected, decimal=2)

    def test_cdf_large_x(self):
        # Regression test for gh-7746.
        # The x values are large enough that the closest 64 bit floating
        # point representation of the exact CDF is 1.0.
        p = stats.skewnorm.cdf([10, 20, 30], -1)
        assert_allclose(p, np.ones(3), rtol=1e-14)
        p = stats.skewnorm.cdf(25, 2.5)
        assert_allclose(p, 1.0, rtol=1e-14)

    def test_cdf_sf_small_values(self):
        # Triples are [x, a, cdf(x, a)].  These values were computed
        # using CDF[SkewNormDistribution[0, 1, a], x] in Wolfram Alpha.
        cdfvals = [
            [-8, 1, 3.870035046664392611e-31],
            [-4, 2, 8.1298399188811398e-21],
            [-2, 5, 1.55326826787106273e-26],
            [-9, -1, 2.257176811907681295e-19],
            [-10, -4, 1.523970604832105213e-23],
        ]
        for x, a, cdfval in cdfvals:
            p = stats.skewnorm.cdf(x, a)
            assert_allclose(p, cdfval, rtol=1e-8)
            # For the skew normal distribution, sf(-x, -a) = cdf(x, a).
            p = stats.skewnorm.sf(-x, -a)
            assert_allclose(p, cdfval, rtol=1e-8)

    @pytest.mark.parametrize('a, moments', _skewnorm_noncentral_moments)
    def test_noncentral_moments(self, a, moments):
        for order, expected in enumerate(moments, start=1):
            mom = stats.skewnorm.moment(order, a)
            assert_allclose(mom, expected, rtol=1e-14)

    def test_fit(self):
        rng = np.random.default_rng(4609813989115202851)

        a, loc, scale = -2, 3.5, 0.5  # arbitrary, valid parameters
        dist = stats.skewnorm(a, loc, scale)
        rvs = dist.rvs(size=100, random_state=rng)

        # test that MLE still honors guesses and fixed parameters
        a2, loc2, scale2 = stats.skewnorm.fit(rvs, -1.5, floc=3)
        a3, loc3, scale3 = stats.skewnorm.fit(rvs, -1.6, floc=3)
        assert loc2 == loc3 == 3  # fixed parameter is respected
        assert a2 != a3  # different guess -> (slightly) different outcome
        # quality of fit is tested elsewhere

        # test that MoM honors fixed parameters, accepts (but ignores) guesses
        a4, loc4, scale4 = stats.skewnorm.fit(rvs, 3, fscale=3, method='mm')
        assert scale4 == 3
        # because scale was fixed, only the mean and skewness will be matched
        dist4 = stats.skewnorm(a4, loc4, scale4)
        res = dist4.stats(moments='ms')
        ref = np.mean(rvs), stats.skew(rvs)
        assert_allclose(res, ref)

        # Test behavior when skew of data is beyond maximum of skewnorm
        rvs = stats.pareto.rvs(1, size=100, random_state=rng)

        # MLE still works
        res = stats.skewnorm.fit(rvs)
        assert np.all(np.isfinite(res))

        # MoM fits variance and skewness
        a5, loc5, scale5 = stats.skewnorm.fit(rvs, method='mm')
        assert np.isinf(a5)
        # distribution infrastruction doesn't allow infinite shape parameters
        # into _stats; it just bypasses it and produces NaNs. Calculate
        # moments manually.
        m, v = np.mean(rvs), np.var(rvs)
        assert_allclose(m, loc5 + scale5 * np.sqrt(2/np.pi))
        assert_allclose(v, scale5**2 * (1 - 2 / np.pi))


class TestExpon:
    def test_zero(self):
        assert_equal(stats.expon.pdf(0), 1)

    def test_tail(self):  # Regression test for ticket 807
        assert_equal(stats.expon.cdf(1e-18), 1e-18)
        assert_equal(stats.expon.isf(stats.expon.sf(40)), 40)

    def test_nan_raises_error(self):
        # see gh-issue 10300
        x = np.array([1.6483, 2.7169, 2.4667, 1.1791, 3.5433, np.nan])
        assert_raises(ValueError, stats.expon.fit, x)

    def test_inf_raises_error(self):
        # see gh-issue 10300
        x = np.array([1.6483, 2.7169, 2.4667, 1.1791, 3.5433, np.inf])
        assert_raises(ValueError, stats.expon.fit, x)


class TestNorm:
    def test_nan_raises_error(self):
        # see gh-issue 10300
        x = np.array([1.6483, 2.7169, 2.4667, 1.1791, 3.5433, np.nan])
        assert_raises(ValueError, stats.norm.fit, x)

    def test_inf_raises_error(self):
        # see gh-issue 10300
        x = np.array([1.6483, 2.7169, 2.4667, 1.1791, 3.5433, np.inf])
        assert_raises(ValueError, stats.norm.fit, x)

    def test_bad_keyword_arg(self):
        x = [1, 2, 3]
        assert_raises(TypeError, stats.norm.fit, x, plate="shrimp")

    @pytest.mark.parametrize('loc', [0, 1])
    def test_delta_cdf(self, loc):
        # The expected value is computed with mpmath:
        # >>> import mpmath
        # >>> mpmath.mp.dps = 60
        # >>> float(mpmath.ncdf(12) - mpmath.ncdf(11))
        # 1.910641809677555e-28
        expected = 1.910641809677555e-28
        delta = stats.norm._delta_cdf(11+loc, 12+loc, loc=loc)
        assert_allclose(delta, expected, rtol=1e-13)
        delta = stats.norm._delta_cdf(-(12+loc), -(11+loc), loc=-loc)
        assert_allclose(delta, expected, rtol=1e-13)


class TestUniform:
    """gh-10300"""
    def test_nan_raises_error(self):
        # see gh-issue 10300
        x = np.array([1.6483, 2.7169, 2.4667, 1.1791, 3.5433, np.nan])
        assert_raises(ValueError, stats.uniform.fit, x)

    def test_inf_raises_error(self):
        # see gh-issue 10300
        x = np.array([1.6483, 2.7169, 2.4667, 1.1791, 3.5433, np.inf])
        assert_raises(ValueError, stats.uniform.fit, x)


class TestExponNorm:
    def test_moments(self):
        # Some moment test cases based on non-loc/scaled formula
        def get_moms(lam, sig, mu):
            # See wikipedia for these formulae
            #  where it is listed as an exponentially modified gaussian
            opK2 = 1.0 + 1 / (lam*sig)**2
            exp_skew = 2 / (lam * sig)**3 * opK2**(-1.5)
            exp_kurt = 6.0 * (1 + (lam * sig)**2)**(-2)
            return [mu + 1/lam, sig*sig + 1.0/(lam*lam), exp_skew, exp_kurt]

        mu, sig, lam = 0, 1, 1
        K = 1.0 / (lam * sig)
        sts = stats.exponnorm.stats(K, loc=mu, scale=sig, moments='mvsk')
        assert_almost_equal(sts, get_moms(lam, sig, mu))
        mu, sig, lam = -3, 2, 0.1
        K = 1.0 / (lam * sig)
        sts = stats.exponnorm.stats(K, loc=mu, scale=sig, moments='mvsk')
        assert_almost_equal(sts, get_moms(lam, sig, mu))
        mu, sig, lam = 0, 3, 1
        K = 1.0 / (lam * sig)
        sts = stats.exponnorm.stats(K, loc=mu, scale=sig, moments='mvsk')
        assert_almost_equal(sts, get_moms(lam, sig, mu))
        mu, sig, lam = -5, 11, 3.5
        K = 1.0 / (lam * sig)
        sts = stats.exponnorm.stats(K, loc=mu, scale=sig, moments='mvsk')
        assert_almost_equal(sts, get_moms(lam, sig, mu))

    def test_nan_raises_error(self):
        # see gh-issue 10300
        x = np.array([1.6483, 2.7169, 2.4667, 1.1791, 3.5433, np.nan])
        assert_raises(ValueError, stats.exponnorm.fit, x, floc=0, fscale=1)

    def test_inf_raises_error(self):
        # see gh-issue 10300
        x = np.array([1.6483, 2.7169, 2.4667, 1.1791, 3.5433, np.inf])
        assert_raises(ValueError, stats.exponnorm.fit, x, floc=0, fscale=1)

    def test_extremes_x(self):
        # Test for extreme values against overflows
        assert_almost_equal(stats.exponnorm.pdf(-900, 1), 0.0)
        assert_almost_equal(stats.exponnorm.pdf(+900, 1), 0.0)
        assert_almost_equal(stats.exponnorm.pdf(-900, 0.01), 0.0)
        assert_almost_equal(stats.exponnorm.pdf(+900, 0.01), 0.0)

    # Expected values for the PDF were computed with mpmath, with
    # the following function, and with mpmath.mp.dps = 50.
    #
    #   def exponnorm_stdpdf(x, K):
    #       x = mpmath.mpf(x)
    #       K = mpmath.mpf(K)
    #       t1 = mpmath.exp(1/(2*K**2) - x/K)
    #       erfcarg = -(x - 1/K)/mpmath.sqrt(2)
    #       t2 = mpmath.erfc(erfcarg)
    #       return t1 * t2 / (2*K)
    #
    @pytest.mark.parametrize('x, K, expected',
                             [(20, 0.01, 6.90010764753618e-88),
                              (1, 0.01, 0.24438994313247364),
                              (-1, 0.01, 0.23955149623472075),
                              (-20, 0.01, 4.6004708690125477e-88),
                              (10, 1, 7.48518298877006e-05),
                              (10, 10000, 9.990005048283775e-05)])
    def test_std_pdf(self, x, K, expected):
        assert_allclose(stats.exponnorm.pdf(x, K), expected, rtol=5e-12)

    # Expected values for the CDF were computed with mpmath using
    # the following function and with mpmath.mp.dps = 60:
    #
    #   def mp_exponnorm_cdf(x, K, loc=0, scale=1):
    #       x = mpmath.mpf(x)
    #       K = mpmath.mpf(K)
    #       loc = mpmath.mpf(loc)
    #       scale = mpmath.mpf(scale)
    #       z = (x - loc)/scale
    #       return (mpmath.ncdf(z)
    #               - mpmath.exp((1/(2*K) - z)/K)*mpmath.ncdf(z - 1/K))
    #
    @pytest.mark.parametrize('x, K, scale, expected',
                             [[0, 0.01, 1, 0.4960109760186432],
                              [-5, 0.005, 1, 2.7939945412195734e-07],
                              [-1e4, 0.01, 100, 0.0],
                              [-1e4, 0.01, 1000, 6.920401854427357e-24],
                              [5, 0.001, 1, 0.9999997118542392]])
    def test_cdf_small_K(self, x, K, scale, expected):
        p = stats.exponnorm.cdf(x, K, scale=scale)
        if expected == 0.0:
            assert p == 0.0
        else:
            assert_allclose(p, expected, rtol=1e-13)

    # Expected values for the SF were computed with mpmath using
    # the following function and with mpmath.mp.dps = 60:
    #
    #   def mp_exponnorm_sf(x, K, loc=0, scale=1):
    #       x = mpmath.mpf(x)
    #       K = mpmath.mpf(K)
    #       loc = mpmath.mpf(loc)
    #       scale = mpmath.mpf(scale)
    #       z = (x - loc)/scale
    #       return (mpmath.ncdf(-z)
    #               + mpmath.exp((1/(2*K) - z)/K)*mpmath.ncdf(z - 1/K))
    #
    @pytest.mark.parametrize('x, K, scale, expected',
                             [[10, 0.01, 1, 8.474702916146657e-24],
                              [2, 0.005, 1, 0.02302280664231312],
                              [5, 0.005, 0.5, 8.024820681931086e-24],
                              [10, 0.005, 0.5, 3.0603340062892486e-89],
                              [20, 0.005, 0.5, 0.0],
                              [-3, 0.001, 1, 0.9986545205566117]])
    def test_sf_small_K(self, x, K, scale, expected):
        p = stats.exponnorm.sf(x, K, scale=scale)
        if expected == 0.0:
            assert p == 0.0
        else:
            assert_allclose(p, expected, rtol=5e-13)


class TestGenExpon:
    def test_pdf_unity_area(self):
        from scipy.integrate import simps
        # PDF should integrate to one
        p = stats.genexpon.pdf(numpy.arange(0, 10, 0.01), 0.5, 0.5, 2.0)
        assert_almost_equal(simps(p, dx=0.01), 1, 1)

    def test_cdf_bounds(self):
        # CDF should always be positive
        cdf = stats.genexpon.cdf(numpy.arange(0, 10, 0.01), 0.5, 0.5, 2.0)
        assert_(numpy.all((0 <= cdf) & (cdf <= 1)))

    # The values of p in the following data were computed with mpmath.
    # E.g. the script
    #     from mpmath import mp
    #     mp.dps = 80
    #     x = mp.mpf('15.0')
    #     a = mp.mpf('1.0')
    #     b = mp.mpf('2.0')
    #     c = mp.mpf('1.5')
    #     print(float(mp.exp((-a-b)*x + (b/c)*-mp.expm1(-c*x))))
    # prints
    #     1.0859444834514553e-19
    @pytest.mark.parametrize('x, p, a, b, c',
                             [(15, 1.0859444834514553e-19, 1, 2, 1.5),
                              (0.25, 0.7609068232534623, 0.5, 2, 3),
                              (0.25, 0.09026661397565876, 9.5, 2, 0.5),
                              (0.01, 0.9753038265071597, 2.5, 0.25, 0.5),
                              (3.25, 0.0001962824553094492, 2.5, 0.25, 0.5),
                              (0.125, 0.9508674287164001, 0.25, 5, 0.5)])
    def test_sf_isf(self, x, p, a, b, c):
        sf = stats.genexpon.sf(x, a, b, c)
        assert_allclose(sf, p, rtol=1e-14)
        isf = stats.genexpon.isf(p, a, b, c)
        assert_allclose(isf, x, rtol=1e-14)

    # The values of p in the following data were computed with mpmath.
    @pytest.mark.parametrize('x, p, a, b, c',
                             [(0.25, 0.2390931767465377, 0.5, 2, 3),
                              (0.25, 0.9097333860243412, 9.5, 2, 0.5),
                              (0.01, 0.0246961734928403, 2.5, 0.25, 0.5),
                              (3.25, 0.9998037175446906, 2.5, 0.25, 0.5),
                              (0.125, 0.04913257128359998, 0.25, 5, 0.5)])
    def test_cdf_ppf(self, x, p, a, b, c):
        cdf = stats.genexpon.cdf(x, a, b, c)
        assert_allclose(cdf, p, rtol=1e-14)
        ppf = stats.genexpon.ppf(p, a, b, c)
        assert_allclose(ppf, x, rtol=1e-14)

class TestTruncexpon:

    def test_sf_isf(self):
        # reference values were computed via the reference distribution, e.g.
        # mp.dps = 50; TruncExpon(b=b).sf(x)
        b = [20, 100]
        x = [19.999999, 99.999999]
        ref = [2.0611546593828472e-15, 3.7200778266671455e-50]
        assert_allclose(stats.truncexpon.sf(x, b), ref, rtol=1e-10)
        assert_allclose(stats.truncexpon.isf(ref, b), x, rtol=1e-12)


class TestExponpow:
    def test_tail(self):
        assert_almost_equal(stats.exponpow.cdf(1e-10, 2.), 1e-20)
        assert_almost_equal(stats.exponpow.isf(stats.exponpow.sf(5, .8), .8),
                            5)


class TestSkellam:
    def test_pmf(self):
        # comparison to R
        k = numpy.arange(-10, 15)
        mu1, mu2 = 10, 5
        skpmfR = numpy.array(
                   [4.2254582961926893e-005, 1.1404838449648488e-004,
                    2.8979625801752660e-004, 6.9177078182101231e-004,
                    1.5480716105844708e-003, 3.2412274963433889e-003,
                    6.3373707175123292e-003, 1.1552351566696643e-002,
                    1.9606152375042644e-002, 3.0947164083410337e-002,
                    4.5401737566767360e-002, 6.1894328166820688e-002,
                    7.8424609500170578e-002, 9.2418812533573133e-002,
                    1.0139793148019728e-001, 1.0371927988298846e-001,
                    9.9076583077406091e-002, 8.8546660073089561e-002,
                    7.4187842052486810e-002, 5.8392772862200251e-002,
                    4.3268692953013159e-002, 3.0248159818374226e-002,
                    1.9991434305603021e-002, 1.2516877303301180e-002,
                    7.4389876226229707e-003])

        assert_almost_equal(stats.skellam.pmf(k, mu1, mu2), skpmfR, decimal=15)

    def test_cdf(self):
        # comparison to R, only 5 decimals
        k = numpy.arange(-10, 15)
        mu1, mu2 = 10, 5
        skcdfR = numpy.array(
                   [6.4061475386192104e-005, 1.7810985988267694e-004,
                    4.6790611790020336e-004, 1.1596768997212152e-003,
                    2.7077485103056847e-003, 5.9489760066490718e-003,
                    1.2286346724161398e-002, 2.3838698290858034e-002,
                    4.3444850665900668e-002, 7.4392014749310995e-002,
                    1.1979375231607835e-001, 1.8168808048289900e-001,
                    2.6011268998306952e-001, 3.5253150251664261e-001,
                    4.5392943399683988e-001, 5.5764871387982828e-001,
                    6.5672529695723436e-001, 7.4527195703032389e-001,
                    8.1945979908281064e-001, 8.7785257194501087e-001,
                    9.2112126489802404e-001, 9.5136942471639818e-001,
                    9.7136085902200120e-001, 9.8387773632530240e-001,
                    9.9131672394792536e-001])

        assert_almost_equal(stats.skellam.cdf(k, mu1, mu2), skcdfR, decimal=5)

    def test_extreme_mu2(self):
        # check that crash reported by gh-17916 large mu2 is resolved
        x, mu1, mu2 = 0, 1, 4820232647677555.0
        assert_allclose(stats.skellam.pmf(x, mu1, mu2), 0, atol=1e-16)
        assert_allclose(stats.skellam.cdf(x, mu1, mu2), 1, atol=1e-16)


class TestLognorm:
    def test_pdf(self):
        # Regression test for Ticket #1471: avoid nan with 0/0 situation
        # Also make sure there are no warnings at x=0, cf gh-5202
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            pdf = stats.lognorm.pdf([0, 0.5, 1], 1)
            assert_array_almost_equal(pdf, [0.0, 0.62749608, 0.39894228])

    def test_logcdf(self):
        # Regression test for gh-5940: sf et al would underflow too early
        x2, mu, sigma = 201.68, 195, 0.149
        assert_allclose(stats.lognorm.sf(x2-mu, s=sigma),
                        stats.norm.sf(np.log(x2-mu)/sigma))
        assert_allclose(stats.lognorm.logsf(x2-mu, s=sigma),
                        stats.norm.logsf(np.log(x2-mu)/sigma))

    @pytest.fixture(scope='function')
    def rng(self):
        return np.random.default_rng(1234)

    @pytest.mark.parametrize("rvs_shape", [.1, 2])
    @pytest.mark.parametrize("rvs_loc", [-2, 0, 2])
    @pytest.mark.parametrize("rvs_scale", [.2, 1, 5])
    @pytest.mark.parametrize('fix_shape, fix_loc, fix_scale',
                             [e for e in product((False, True), repeat=3)
                              if False in e])
    @np.errstate(invalid="ignore")
    def test_fit_MLE_comp_optimizer(self, rvs_shape, rvs_loc, rvs_scale,
                                    fix_shape, fix_loc, fix_scale, rng):
        data = stats.lognorm.rvs(size=100, s=rvs_shape, scale=rvs_scale,
                                 loc=rvs_loc, random_state=rng)

        kwds = {}
        if fix_shape:
            kwds['f0'] = rvs_shape
        if fix_loc:
            kwds['floc'] = rvs_loc
        if fix_scale:
            kwds['fscale'] = rvs_scale

        _assert_less_or_close_loglike(stats.lognorm, data, **kwds)


class TestBeta:
    def test_logpdf(self):
        # Regression test for Ticket #1326: avoid nan with 0*log(0) situation
        logpdf = stats.beta.logpdf(0, 1, 0.5)
        assert_almost_equal(logpdf, -0.69314718056)
        logpdf = stats.beta.logpdf(0, 0.5, 1)
        assert_almost_equal(logpdf, np.inf)

    def test_logpdf_ticket_1866(self):
        alpha, beta = 267, 1472
        x = np.array([0.2, 0.5, 0.6])
        b = stats.beta(alpha, beta)
        assert_allclose(b.logpdf(x).sum(), -1201.699061824062)
        assert_allclose(b.pdf(x), np.exp(b.logpdf(x)))

    def test_fit_bad_keyword_args(self):
        x = [0.1, 0.5, 0.6]
        assert_raises(TypeError, stats.beta.fit, x, floc=0, fscale=1,
                      plate="shrimp")

    def test_fit_duplicated_fixed_parameter(self):
        # At most one of 'f0', 'fa' or 'fix_a' can be given to the fit method.
        # More than one raises a ValueError.
        x = [0.1, 0.5, 0.6]
        assert_raises(ValueError, stats.beta.fit, x, fa=0.5, fix_a=0.5)

    @pytest.mark.skipif(MACOS_INTEL, reason="Overflow, see gh-14901")
    def test_issue_12635(self):
        # Confirm that Boost's beta distribution resolves gh-12635.
        # Check against R:
        # options(digits=16)
        # p = 0.9999999999997369
        # a = 75.0
        # b = 66334470.0
        # print(qbeta(p, a, b))
        p, a, b = 0.9999999999997369, 75.0, 66334470.0
        assert_allclose(stats.beta.ppf(p, a, b), 2.343620802982393e-06)

    @pytest.mark.skipif(MACOS_INTEL, reason="Overflow, see gh-14901")
    def test_issue_12794(self):
        # Confirm that Boost's beta distribution resolves gh-12794.
        # Check against R.
        # options(digits=16)
        # p = 1e-11
        # count_list = c(10,100,1000)
        # print(qbeta(1-p, count_list + 1, 100000 - count_list))
        inv_R = np.array([0.0004944464889611935,
                          0.0018360586912635726,
                          0.0122663919942518351])
        count_list = np.array([10, 100, 1000])
        p = 1e-11
        inv = stats.beta.isf(p, count_list + 1, 100000 - count_list)
        assert_allclose(inv, inv_R)
        res = stats.beta.sf(inv, count_list + 1, 100000 - count_list)
        assert_allclose(res, p)

    @pytest.mark.skipif(MACOS_INTEL, reason="Overflow, see gh-14901")
    def test_issue_12796(self):
        # Confirm that Boost's beta distribution succeeds in the case
        # of gh-12796
        alpha_2 = 5e-6
        count_ = np.arange(1, 20)
        nobs = 100000
        q, a, b = 1 - alpha_2, count_ + 1, nobs - count_
        inv = stats.beta.ppf(q, a, b)
        res = stats.beta.cdf(inv, a, b)
        assert_allclose(res, 1 - alpha_2)

    def test_endpoints(self):
        # Confirm that boost's beta distribution returns inf at x=1
        # when b<1
        a, b = 1, 0.5
        assert_equal(stats.beta.pdf(1, a, b), np.inf)

        # Confirm that boost's beta distribution returns inf at x=0
        # when a<1
        a, b = 0.2, 3
        assert_equal(stats.beta.pdf(0, a, b), np.inf)

        # Confirm that boost's beta distribution returns 5 at x=0
        # when a=1, b=5
        a, b = 1, 5
        assert_equal(stats.beta.pdf(0, a, b), 5)
        assert_equal(stats.beta.pdf(1e-310, a, b), 5)

        # Confirm that boost's beta distribution returns 5 at x=1
        # when a=5, b=1
        a, b = 5, 1
        assert_equal(stats.beta.pdf(1, a, b), 5)
        assert_equal(stats.beta.pdf(1-1e-310, a, b), 5)

    @pytest.mark.xfail(IS_PYPY, reason="Does not convert boost warning")
    def test_boost_eval_issue_14606(self):
        q, a, b = 0.995, 1.0e11, 1.0e13
        with pytest.warns(RuntimeWarning):
            stats.beta.ppf(q, a, b)

    @pytest.mark.parametrize('method', [stats.beta.ppf, stats.beta.isf])
    @pytest.mark.parametrize('a, b', [(1e-310, 12.5), (12.5, 1e-310)])
    def test_beta_ppf_with_subnormal_a_b(self, method, a, b):
        # Regression test for gh-17444: beta.ppf(p, a, b) and beta.isf(p, a, b)
        # would result in a segmentation fault if either a or b was subnormal.
        p = 0.9
        # Depending on the version of Boost that we have vendored and
        # our setting of the Boost double promotion policy, the call
        # `stats.beta.ppf(p, a, b)` might raise an OverflowError or
        # return a value.  We'll accept either behavior (and not care about
        # the value), because our goal here is to verify that the call does
        # not trigger a segmentation fault.
        try:
            method(p, a, b)
        except OverflowError:
            # The OverflowError exception occurs with Boost 1.80 or earlier
            # when Boost's double promotion policy is false; see
            #   https://github.com/boostorg/math/issues/882
            # and
            #   https://github.com/boostorg/math/pull/883
            # Once we have vendored the fixed version of Boost, we can drop
            # this try-except wrapper and just call the function.
            pass

    # entropy accuracy was confirmed using the following mpmath function
    # from mpmath import mp
    # mp.dps = 50
    # def beta_entropy_mpmath(a, b):
    #     a = mp.mpf(a)
    #     b = mp.mpf(b)
    #     entropy = mp.log(mp.beta(a, b)) - (a - 1) * mp.digamma(a) -\
    #              (b - 1) * mp.digamma(b) + (a + b -2) * mp.digamma(a + b)
    #     return float(entropy)

    @pytest.mark.parametrize('a, b, ref',
                             [(0.5, 0.5, -0.24156447527049044),
                              (0.001, 1, -992.0922447210179),
                              (1, 10000, -8.210440371976183),
                              (100000, 100000, -5.377247470132859)])
    def test_entropy(self, a, b, ref):
        assert_allclose(stats.beta(a, b).entropy(), ref)


class TestBetaPrime:
    # the test values are used in test_cdf_gh_17631 / test_ppf_gh_17631
    # They are computed with mpmath. Example:
    # from mpmath import mp
    # mp.dps = 50
    # a, b = mp.mpf(0.05), mp.mpf(0.1)
    # x = mp.mpf(1e22)
    # float(mp.betainc(a, b, 0.0, x/(1+x), regularized=True))
    # note: we use the values computed by the cdf to test whether
    # ppf(cdf(x)) == x (up to a small tolerance)
    # since the ppf can be very sensitive to small variations of the input,
    # it can be required to generate the test case for the ppf separately,
    # see self.test_ppf
    cdf_vals = [
        (1e22, 100.0, 0.05, 0.8973027435427167),
        (1e10, 100.0, 0.05, 0.5911548582766262),
        (1e8, 0.05, 0.1, 0.9467768090820048),
        (1e8, 100.0, 0.05, 0.4852944858726726),
        (1e-10, 0.05, 0.1, 0.21238845427095),
        (1e-10, 1.5, 1.5, 1.697652726007973e-15),
        (1e-10, 0.05, 100.0, 0.40884514172337383),
        (1e-22, 0.05, 0.1, 0.053349567649287326),
        (1e-22, 1.5, 1.5, 1.6976527263135503e-33),
        (1e-22, 0.05, 100.0, 0.10269725645728331),
        (1e-100, 0.05, 0.1, 6.7163126421919795e-06),
        (1e-100, 1.5, 1.5, 1.6976527263135503e-150),
        (1e-100, 0.05, 100.0, 1.2928818587561651e-05),
    ]

    def test_logpdf(self):
        alpha, beta = 267, 1472
        x = np.array([0.2, 0.5, 0.6])
        b = stats.betaprime(alpha, beta)
        assert_(np.isfinite(b.logpdf(x)).all())
        assert_allclose(b.pdf(x), np.exp(b.logpdf(x)))

    def test_cdf(self):
        # regression test for gh-4030: Implementation of
        # scipy.stats.betaprime.cdf()
        x = stats.betaprime.cdf(0, 0.2, 0.3)
        assert_equal(x, 0.0)

        alpha, beta = 267, 1472
        x = np.array([0.2, 0.5, 0.6])
        cdfs = stats.betaprime.cdf(x, alpha, beta)
        assert_(np.isfinite(cdfs).all())

        # check the new cdf implementation vs generic one:
        gen_cdf = stats.rv_continuous._cdf_single
        cdfs_g = [gen_cdf(stats.betaprime, val, alpha, beta) for val in x]
        assert_allclose(cdfs, cdfs_g, atol=0, rtol=2e-12)

    # The expected values for test_ppf() were computed with mpmath, e.g.
    #
    #   from mpmath import mp
    #   mp.dps = 125
    #   p = 0.01
    #   a, b = 1.25, 2.5
    #   x = mp.findroot(lambda t: mp.betainc(a, b, x1=0, x2=t/(1+t),
    #                                        regularized=True) - p,
    #                   x0=(0.01, 0.011), method='secant')
    #   print(float(x))
    #
    # prints
    #
    #   0.01080162700956614
    #
    @pytest.mark.parametrize(
        'p, a, b, expected',
        [(0.010, 1.25, 2.5, 0.01080162700956614),
         (1e-12, 1.25, 2.5, 1.0610141996279122e-10),
         (1e-18, 1.25, 2.5, 1.6815941817974941e-15),
         (1e-17, 0.25, 7.0, 1.0179194531881782e-69),
         (0.375, 0.25, 7.0, 0.002036820346115211),
         (0.9978811466052919, 0.05, 0.1, 1.0000000000001218e22),]
    )
    def test_ppf(self, p, a, b, expected):
        x = stats.betaprime.ppf(p, a, b)
        assert_allclose(x, expected, rtol=1e-14)

    @pytest.mark.parametrize('x, a, b, p', cdf_vals)
    def test_ppf_gh_17631(self, x, a, b, p):
        assert_allclose(stats.betaprime.ppf(p, a, b), x, rtol=1e-14)

    @pytest.mark.parametrize(
        'x, a, b, expected',
        cdf_vals + [
            (1e10, 1.5, 1.5, 0.9999999999999983),
            (1e10, 0.05, 0.1, 0.9664184367890859),
            (1e22, 0.05, 0.1, 0.9978811466052919),
        ])
    def test_cdf_gh_17631(self, x, a, b, expected):
        assert_allclose(stats.betaprime.cdf(x, a, b), expected, rtol=1e-14)

    @pytest.mark.parametrize(
        'x, a, b, expected',
        [(1e50, 0.05, 0.1, 0.9999966641709545),
         (1e50, 100.0, 0.05, 0.995925162631006)])
    def test_cdf_extreme_tails(self, x, a, b, expected):
        # for even more extreme values, we only get a few correct digits
        # results are still < 1
        y = stats.betaprime.cdf(x, a, b)
        assert y < 1.0
        assert_allclose(y, expected, rtol=2e-5)

    def test_sf(self):
        # reference values were computed via the reference distribution,
        # e.g.
        # mp.dps = 50
        # a, b = 5, 3
        # x = 1e10
        # BetaPrime(a=a, b=b).sf(x); returns 3.4999999979e-29
        a = [5, 4, 2, 0.05, 0.05, 0.05, 0.05, 100.0, 100.0, 0.05, 0.05,
             0.05, 1.5, 1.5]
        b = [3, 2, 1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 100.0, 100.0,
             100.0, 1.5, 1.5]
        x = [1e10, 1e20, 1e30, 1e22, 1e-10, 1e-22, 1e-100, 1e22, 1e10,
             1e-10, 1e-22, 1e-100, 1e10, 1e-10]
        ref = [3.4999999979e-29, 9.999999999994357e-40, 1.9999999999999998e-30,
               0.0021188533947081017, 0.78761154572905, 0.9466504323507127,
               0.9999932836873578, 0.10269725645728331, 0.40884514172337383,
               0.5911548582766262, 0.8973027435427167, 0.9999870711814124,
               1.6976527260079727e-15, 0.9999999999999983]
        sf_values = stats.betaprime.sf(x, a, b)
        assert_allclose(sf_values, ref, rtol=1e-12)

    def test_fit_stats_gh18274(self):
        # gh-18274 reported spurious warning emitted when fitting `betaprime`
        # to data. Some of these were emitted by stats, too. Check that the
        # warnings are no longer emitted.
        stats.betaprime.fit([0.1, 0.25, 0.3, 1.2, 1.6], floc=0, fscale=1)
        stats.betaprime(a=1, b=1).stats('mvsk')

    def test_moment_gh18634(self):
        # Testing for gh-18634 revealed that `betaprime` raised a
        # NotImplementedError for higher moments. Check that this is
        # resolved. Parameters are arbitrary but lie on either side of the
        # moment order (5) to test both branches of `_lazywhere`. Reference
        # values produced with Mathematica, e.g.
        # `Moment[BetaPrimeDistribution[2,7],5]`
        ref = [np.inf, 0.867096912929055]
        res = stats.betaprime(2, [4.2, 7.1]).moment(5)
        assert_allclose(res, ref)


class TestGamma:
    def test_pdf(self):
        # a few test cases to compare with R
        pdf = stats.gamma.pdf(90, 394, scale=1./5)
        assert_almost_equal(pdf, 0.002312341)

        pdf = stats.gamma.pdf(3, 10, scale=1./5)
        assert_almost_equal(pdf, 0.1620358)

    def test_logpdf(self):
        # Regression test for Ticket #1326: cornercase avoid nan with 0*log(0)
        # situation
        logpdf = stats.gamma.logpdf(0, 1)
        assert_almost_equal(logpdf, 0)

    def test_fit_bad_keyword_args(self):
        x = [0.1, 0.5, 0.6]
        assert_raises(TypeError, stats.gamma.fit, x, floc=0, plate="shrimp")

    def test_isf(self):
        # Test cases for when the probability is very small. See gh-13664.
        # The expected values can be checked with mpmath.  With mpmath,
        # the survival function sf(x, k) can be computed as
        #
        #     mpmath.gammainc(k, x, mpmath.inf, regularized=True)
        #
        # Here we have:
        #
        # >>> mpmath.mp.dps = 60
        # >>> float(mpmath.gammainc(1, 39.14394658089878, mpmath.inf,
        # ...                       regularized=True))
        # 9.99999999999999e-18
        # >>> float(mpmath.gammainc(100, 330.6557590436547, mpmath.inf,
        #                           regularized=True))
        # 1.000000000000028e-50
        #
        assert np.isclose(stats.gamma.isf(1e-17, 1),
                          39.14394658089878, atol=1e-14)
        assert np.isclose(stats.gamma.isf(1e-50, 100),
                          330.6557590436547, atol=1e-13)

    @pytest.mark.parametrize('scale', [1.0, 5.0])
    def test_delta_cdf(self, scale):
        # Expected value computed with mpmath:
        #
        # >>> import mpmath
        # >>> mpmath.mp.dps = 150
        # >>> cdf1 = mpmath.gammainc(3, 0, 245, regularized=True)
        # >>> cdf2 = mpmath.gammainc(3, 0, 250, regularized=True)
        # >>> float(cdf2 - cdf1)
        # 1.1902609356171962e-102
        #
        delta = stats.gamma._delta_cdf(scale*245, scale*250, 3, scale=scale)
        assert_allclose(delta, 1.1902609356171962e-102, rtol=1e-13)

    @pytest.mark.parametrize('a, ref, rtol',
                             [(1e-4, -9990.366610819761, 1e-15),
                              (2, 1.5772156649015328, 1e-15),
                              (100, 3.7181819485047463, 1e-13),
                              (1e4, 6.024075385026086, 1e-15),
                              (1e18, 22.142204370151084, 1e-15),
                              (1e100, 116.54819318290696, 1e-15)])
    def test_entropy(self, a, ref, rtol):
        # expected value computed with mpmath:
        # from mpmath import mp
        # mp.dps = 500
        # def gamma_entropy_reference(x):
        #     x = mp.mpf(x)
        #     return float(mp.digamma(x) * (mp.one - x) + x + mp.loggamma(x))

        assert_allclose(stats.gamma.entropy(a), ref, rtol=rtol)


class TestDgamma:
    def test_pdf(self):
        rng = np.random.default_rng(3791303244302340058)
        size = 10  # number of points to check
        x = rng.normal(scale=10, size=size)
        a = rng.uniform(high=10, size=size)
        res = stats.dgamma.pdf(x, a)
        ref = stats.gamma.pdf(np.abs(x), a) / 2
        assert_allclose(res, ref)

        dist = stats.dgamma(a)
        assert_equal(dist.pdf(x), res)

    # mpmath was used to compute the expected values.
    # For x < 0, cdf(x, a) is mp.gammainc(a, -x, mp.inf, regularized=True)/2
    # For x > 0, cdf(x, a) is (1 + mp.gammainc(a, 0, x, regularized=True))/2
    # E.g.
    #    from mpmath import mp
    #    mp.dps = 50
    #    print(float(mp.gammainc(1, 20, mp.inf, regularized=True)/2))
    # prints
    #    1.030576811219279e-09
    @pytest.mark.parametrize('x, a, expected',
                             [(-20, 1, 1.030576811219279e-09),
                              (-40, 1, 2.1241771276457944e-18),
                              (-50, 5, 2.7248509914602648e-17),
                              (-25, 0.125, 5.333071920958156e-14),
                              (5, 1, 0.9966310265004573)])
    def test_cdf_ppf_sf_isf_tail(self, x, a, expected):
        cdf = stats.dgamma.cdf(x, a)
        assert_allclose(cdf, expected, rtol=5e-15)
        ppf = stats.dgamma.ppf(expected, a)
        assert_allclose(ppf, x, rtol=5e-15)
        sf = stats.dgamma.sf(-x, a)
        assert_allclose(sf, expected, rtol=5e-15)
        isf = stats.dgamma.isf(expected, a)
        assert_allclose(isf, -x, rtol=5e-15)

    @pytest.mark.parametrize("a, ref",
                             [(1.5, 2.0541199559354117),
                             (1.3, 1.9357296377121247),
                             (1.1, 1.7856502333412134)])
    def test_entropy(self, a, ref):
        # The reference values were calculated with mpmath:
        # def entropy_dgamma(a):
        #    def pdf(x):
        #        A = mp.one / (mp.mpf(2.) * mp.gamma(a))
        #        B = mp.fabs(x) ** (a - mp.one)
        #        C = mp.exp(-mp.fabs(x))
        #        h = A * B * C
        #        return h
        #
        #    return -mp.quad(lambda t: pdf(t) * mp.log(pdf(t)),
        #                    [-mp.inf, mp.inf])
        assert_allclose(stats.dgamma.entropy(a), ref, rtol=1e-14)

    @pytest.mark.parametrize("a, ref",
                             [(1e-100, -1e+100),
                             (1e-10, -9999999975.858217),
                             (1e-5, -99987.37111657023),
                             (1e4, 6.717222565586032),
                             (1000000000000000.0, 19.38147391121996),
                             (1e+100, 117.2413403634669)])
    def test_entropy_entreme_values(self, a, ref):
        # The reference values were calculated with mpmath:
        # from mpmath import mp
        # mp.dps = 500
        # def second_dgamma(a):
        #     a = mp.mpf(a)
        #     x_1 = a + mp.log(2) + mp.loggamma(a)
        #     x_2 = (mp.one - a) * mp.digamma(a)
        #     h = x_1 + x_2
        #     return h
        assert_allclose(stats.dgamma.entropy(a), ref, rtol=1e-10)

    def test_entropy_array_input(self):
        x = np.array([1, 5, 1e20, 1e-5])
        y = stats.dgamma.entropy(x)
        for i in range(len(y)):
            assert y[i] == stats.dgamma.entropy(x[i])


class TestChi2:
    # regression tests after precision improvements, ticket:1041, not verified
    def test_precision(self):
        assert_almost_equal(stats.chi2.pdf(1000, 1000), 8.919133934753128e-003,
                            decimal=14)
        assert_almost_equal(stats.chi2.pdf(100, 100), 0.028162503162596778,
                            decimal=14)

    def test_ppf(self):
        # Expected values computed with mpmath.
        df = 4.8
        x = stats.chi2.ppf(2e-47, df)
        assert_allclose(x, 1.098472479575179840604902808e-19, rtol=1e-10)
        x = stats.chi2.ppf(0.5, df)
        assert_allclose(x, 4.15231407598589358660093156, rtol=1e-10)

        df = 13
        x = stats.chi2.ppf(2e-77, df)
        assert_allclose(x, 1.0106330688195199050507943e-11, rtol=1e-10)
        x = stats.chi2.ppf(0.1, df)
        assert_allclose(x, 7.041504580095461859307179763, rtol=1e-10)

    # Entropy references values were computed with the following mpmath code
    # from mpmath import mp
    # mp.dps = 50
    # def chisq_entropy_mpmath(df):
    #     df = mp.mpf(df)
    #     half_df = 0.5 * df
    #     entropy = (half_df + mp.log(2) + mp.log(mp.gamma(half_df)) +
    #                (mp.one - half_df) * mp.digamma(half_df))
    #     return float(entropy)

    @pytest.mark.parametrize('df, ref',
                             [(1e-4, -19988.980448690163),
                              (1, 0.7837571104739337),
                              (100, 4.061397128938114),
                              (251, 4.525577254045129),
                              (1e15, 19.034900320939986)])
    def test_entropy(self, df, ref):
        assert_allclose(stats.chi2(df).entropy(), ref, rtol=1e-13)


class TestGumbelL:
    # gh-6228
    def test_cdf_ppf(self):
        x = np.linspace(-100, -4)
        y = stats.gumbel_l.cdf(x)
        xx = stats.gumbel_l.ppf(y)
        assert_allclose(x, xx)

    def test_logcdf_logsf(self):
        x = np.linspace(-100, -4)
        y = stats.gumbel_l.logcdf(x)
        z = stats.gumbel_l.logsf(x)
        u = np.exp(y)
        v = -special.expm1(z)
        assert_allclose(u, v)

    def test_sf_isf(self):
        x = np.linspace(-20, 5)
        y = stats.gumbel_l.sf(x)
        xx = stats.gumbel_l.isf(y)
        assert_allclose(x, xx)

    @pytest.mark.parametrize('loc', [-1, 1])
    def test_fit_fixed_param(self, loc):
        # ensure fixed location is correctly reflected from `gumbel_r.fit`
        # See comments at end of gh-12737.
        data = stats.gumbel_l.rvs(size=100, loc=loc)
        fitted_loc, _ = stats.gumbel_l.fit(data, floc=loc)
        assert_equal(fitted_loc, loc)


class TestGumbelR:

    def test_sf(self):
        # Expected value computed with mpmath:
        #   >>> import mpmath
        #   >>> mpmath.mp.dps = 40
        #   >>> float(mpmath.mp.one - mpmath.exp(-mpmath.exp(-50)))
        #   1.9287498479639178e-22
        assert_allclose(stats.gumbel_r.sf(50), 1.9287498479639178e-22,
                        rtol=1e-14)

    def test_isf(self):
        # Expected value computed with mpmath:
        #   >>> import mpmath
        #   >>> mpmath.mp.dps = 40
        #   >>> float(-mpmath.log(-mpmath.log(mpmath.mp.one - 1e-17)))
        #   39.14394658089878
        assert_allclose(stats.gumbel_r.isf(1e-17), 39.14394658089878,
                        rtol=1e-14)


class TestLevyStable:
    @pytest.fixture
    def nolan_pdf_sample_data(self):
        """Sample data points for pdf computed with Nolan's stablec

        See - http://fs2.american.edu/jpnolan/www/stable/stable.html

        There's a known limitation of Nolan's executable for alpha < 0.2.

        The data table loaded below is generated from Nolan's stablec
        with the following parameter space:

            alpha = 0.1, 0.2, ..., 2.0
            beta = -1.0, -0.9, ..., 1.0
            p = 0.01, 0.05, 0.1, 0.25, 0.35, 0.5,
        and the equivalent for the right tail

        Typically inputs for stablec:

            stablec.exe <<
            1 # pdf
            1 # Nolan S equivalent to S0 in scipy
            .25,2,.25 # alpha
            -1,-1,0 # beta
            -10,10,1 # x
            1,0 # gamma, delta
            2 # output file
        """
        data = np.load(
            Path(__file__).parent /
            'data/levy_stable/stable-Z1-pdf-sample-data.npy'
        )
        data = np.core.records.fromarrays(data.T, names='x,p,alpha,beta,pct')
        return data

    @pytest.fixture
    def nolan_cdf_sample_data(self):
        """Sample data points for cdf computed with Nolan's stablec

        See - http://fs2.american.edu/jpnolan/www/stable/stable.html

        There's a known limitation of Nolan's executable for alpha < 0.2.

        The data table loaded below is generated from Nolan's stablec
        with the following parameter space:

            alpha = 0.1, 0.2, ..., 2.0
            beta = -1.0, -0.9, ..., 1.0
            p = 0.01, 0.05, 0.1, 0.25, 0.35, 0.5,

        and the equivalent for the right tail

        Ideally, Nolan's output for CDF values should match the percentile
        from where they have been sampled from. Even more so as we extract
        percentile x positions from stablec too. However, we note at places
        Nolan's stablec will produce absolute errors in order of 1e-5. We
        compare against his calculations here. In future, once we less
        reliant on Nolan's paper we might switch to comparing directly at
        percentiles (those x values being produced from some alternative
        means).

        Typically inputs for stablec:

            stablec.exe <<
            2 # cdf
            1 # Nolan S equivalent to S0 in scipy
            .25,2,.25 # alpha
            -1,-1,0 # beta
            -10,10,1 # x
            1,0 # gamma, delta
            2 # output file
        """
        data = np.load(
            Path(__file__).parent /
            'data/levy_stable/stable-Z1-cdf-sample-data.npy'
        )
        data = np.core.records.fromarrays(data.T, names='x,p,alpha,beta,pct')
        return data

    @pytest.fixture
    def nolan_loc_scale_sample_data(self):
        """Sample data where loc, scale are different from 0, 1

        Data extracted in similar way to pdf/cdf above using
        Nolan's stablec but set to an arbitrary location scale of
        (2, 3) for various important parameters alpha, beta and for
        parameterisations S0 and S1.
        """
        data = np.load(
            Path(__file__).parent /
            'data/levy_stable/stable-loc-scale-sample-data.npy'
        )
        return data

    @pytest.mark.parametrize(
        "sample_size", [
            pytest.param(50), pytest.param(1500, marks=pytest.mark.slow)
        ]
    )
    @pytest.mark.parametrize("parameterization", ["S0", "S1"])
    @pytest.mark.parametrize(
        "alpha,beta", [(1.0, 0), (1.0, -0.5), (1.5, 0), (1.9, 0.5)]
    )
    @pytest.mark.parametrize("gamma,delta", [(1, 0), (3, 2)])
    def test_rvs(
            self,
            parameterization,
            alpha,
            beta,
            gamma,
            delta,
            sample_size,
    ):
        stats.levy_stable.parameterization = parameterization
        ls = stats.levy_stable(
            alpha=alpha, beta=beta, scale=gamma, loc=delta
        )
        _, p = stats.kstest(
            ls.rvs(size=sample_size, random_state=1234), ls.cdf
        )
        assert p > 0.05

    @pytest.mark.slow
    @pytest.mark.parametrize('beta', [0.5, 1])
    def test_rvs_alpha1(self, beta):
        """Additional test cases for rvs for alpha equal to 1."""
        np.random.seed(987654321)
        alpha = 1.0
        loc = 0.5
        scale = 1.5
        x = stats.levy_stable.rvs(alpha, beta, loc=loc, scale=scale,
                                  size=5000)
        stat, p = stats.kstest(x, 'levy_stable',
                               args=(alpha, beta, loc, scale))
        assert p > 0.01

    def test_fit(self):
        # construct data to have percentiles that match
        # example in McCulloch 1986.
        x = [
            -.05413, -.05413, 0., 0., 0., 0., .00533, .00533, .00533, .00533,
            .00533, .03354, .03354, .03354, .03354, .03354, .05309, .05309,
            .05309, .05309, .05309
        ]
        alpha1, beta1, loc1, scale1 = stats.levy_stable._fitstart(x)
        assert_allclose(alpha1, 1.48, rtol=0, atol=0.01)
        assert_almost_equal(beta1, -.22, 2)
        assert_almost_equal(scale1, 0.01717, 4)
        assert_almost_equal(
            loc1, 0.00233, 2
        )  # to 2 dps due to rounding error in McCulloch86

        # cover alpha=2 scenario
        x2 = x + [.05309, .05309, .05309, .05309, .05309]
        alpha2, beta2, loc2, scale2 = stats.levy_stable._fitstart(x2)
        assert_equal(alpha2, 2)
        assert_equal(beta2, -1)
        assert_almost_equal(scale2, .02503, 4)
        assert_almost_equal(loc2, .03354, 4)

    @pytest.mark.xfail(reason="Unknown problem with fitstart.")
    @pytest.mark.parametrize(
        "alpha,beta,delta,gamma",
        [
            (1.5, 0.4, 2, 3),
            (1.0, 0.4, 2, 3),
        ]
    )
    @pytest.mark.parametrize(
        "parametrization", ["S0", "S1"]
    )
    def test_fit_rvs(self, alpha, beta, delta, gamma, parametrization):
        """Test that fit agrees with rvs for each parametrization."""
        stats.levy_stable.parametrization = parametrization
        data = stats.levy_stable.rvs(
            alpha, beta, loc=delta, scale=gamma, size=10000, random_state=1234
        )
        fit = stats.levy_stable._fitstart(data)
        alpha_obs, beta_obs, delta_obs, gamma_obs = fit
        assert_allclose(
            [alpha, beta, delta, gamma],
            [alpha_obs, beta_obs, delta_obs, gamma_obs],
            rtol=0.01,
        )

    def test_fit_beta_flip(self):
        # Confirm that sign of beta affects loc, not alpha or scale.
        x = np.array([1, 1, 3, 3, 10, 10, 10, 30, 30, 100, 100])
        alpha1, beta1, loc1, scale1 = stats.levy_stable._fitstart(x)
        alpha2, beta2, loc2, scale2 = stats.levy_stable._fitstart(-x)
        assert_equal(beta1, 1)
        assert loc1 != 0
        assert_almost_equal(alpha2, alpha1)
        assert_almost_equal(beta2, -beta1)
        assert_almost_equal(loc2, -loc1)
        assert_almost_equal(scale2, scale1)

    def test_fit_delta_shift(self):
        # Confirm that loc slides up and down if data shifts.
        SHIFT = 1
        x = np.array([1, 1, 3, 3, 10, 10, 10, 30, 30, 100, 100])
        alpha1, beta1, loc1, scale1 = stats.levy_stable._fitstart(-x)
        alpha2, beta2, loc2, scale2 = stats.levy_stable._fitstart(-x + SHIFT)
        assert_almost_equal(alpha2, alpha1)
        assert_almost_equal(beta2, beta1)
        assert_almost_equal(loc2, loc1 + SHIFT)
        assert_almost_equal(scale2, scale1)

    def test_fit_loc_extrap(self):
        # Confirm that loc goes out of sample for alpha close to 1.
        x = [1, 1, 3, 3, 10, 10, 10, 30, 30, 140, 140]
        alpha1, beta1, loc1, scale1 = stats.levy_stable._fitstart(x)
        assert alpha1 < 1, f"Expected alpha < 1, got {alpha1}"
        assert loc1 < min(x), f"Expected loc < {min(x)}, got {loc1}"

        x2 = [1, 1, 3, 3, 10, 10, 10, 30, 30, 130, 130]
        alpha2, beta2, loc2, scale2 = stats.levy_stable._fitstart(x2)
        assert alpha2 > 1, f"Expected alpha > 1, got {alpha2}"
        assert loc2 > max(x2), f"Expected loc > {max(x2)}, got {loc2}"

    @pytest.mark.parametrize(
        "pct_range,alpha_range,beta_range", [
            pytest.param(
                [.01, .5, .99],
                [.1, 1, 2],
                [-1, 0, .8],
            ),
            pytest.param(
                [.01, .05, .5, .95, .99],
                [.1, .5, 1, 1.5, 2],
                [-.9, -.5, 0, .3, .6, 1],
                marks=pytest.mark.slow
            ),
            pytest.param(
                [.01, .05, .1, .25, .35, .5, .65, .75, .9, .95, .99],
                np.linspace(0.1, 2, 20),
                np.linspace(-1, 1, 21),
                marks=pytest.mark.xslow,
            ),
        ]
    )
    def test_pdf_nolan_samples(
            self, nolan_pdf_sample_data, pct_range, alpha_range, beta_range
    ):
        """Test pdf values against Nolan's stablec.exe output"""
        data = nolan_pdf_sample_data

        # some tests break on linux 32 bit
        uname = platform.uname()
        is_linux_32 = uname.system == 'Linux' and uname.machine == 'i686'
        platform_desc = "/".join(
            [uname.system, uname.machine, uname.processor])

        # fmt: off
        # There are a number of cases which fail on some but not all platforms.
        # These are excluded by the filters below. TODO: Rewrite tests so that
        # the now filtered out test cases are still run but marked in pytest as
        # expected to fail.
        tests = [
            [
                'dni', 1e-7, lambda r: (
                    np.isin(r['pct'], pct_range) &
                    np.isin(r['alpha'], alpha_range) &
                    np.isin(r['beta'], beta_range) &
                    ~(
                        (
                            (r['beta'] == 0) &
                            (r['pct'] == 0.5)
                        ) |
                        (
                            (r['beta'] >= 0.9) &
                            (r['alpha'] >= 1.6) &
                            (r['pct'] == 0.5)
                        ) |
                        (
                            (r['alpha'] <= 0.4) &
                            np.isin(r['pct'], [.01, .99])
                        ) |
                        (
                            (r['alpha'] <= 0.3) &
                            np.isin(r['pct'], [.05, .95])
                        ) |
                        (
                            (r['alpha'] <= 0.2) &
                            np.isin(r['pct'], [.1, .9])
                        ) |
                        (
                            (r['alpha'] == 0.1) &
                            np.isin(r['pct'], [.25, .75]) &
                            np.isin(np.abs(r['beta']), [.5, .6, .7])
                        ) |
                        (
                            (r['alpha'] == 0.1) &
                            np.isin(r['pct'], [.5]) &
                            np.isin(np.abs(r['beta']), [.1])
                        ) |
                        (
                            (r['alpha'] == 0.1) &
                            np.isin(r['pct'], [.35, .65]) &
                            np.isin(np.abs(r['beta']), [-.4, -.3, .3, .4, .5])
                        ) |
                        (
                            (r['alpha'] == 0.2) &
                            (r['beta'] == 0.5) &
                            (r['pct'] == 0.25)
                        ) |
                        (
                            (r['alpha'] == 0.2) &
                            (r['beta'] == -0.3) &
                            (r['pct'] == 0.65)
                        ) |
                        (
                            (r['alpha'] == 0.2) &
                            (r['beta'] == 0.3) &
                            (r['pct'] == 0.35)
                        ) |
                        (
                            (r['alpha'] == 1.) &
                            np.isin(r['pct'], [.5]) &
                            np.isin(np.abs(r['beta']), [.1, .2, .3, .4])
                        ) |
                        (
                            (r['alpha'] == 1.) &
                            np.isin(r['pct'], [.35, .65]) &
                            np.isin(np.abs(r['beta']), [.8, .9, 1.])
                        ) |
                        (
                            (r['alpha'] == 1.) &
                            np.isin(r['pct'], [.01, .99]) &
                            np.isin(np.abs(r['beta']), [-.1, .1])
                        ) |
                        # various points ok but too sparse to list
                        (r['alpha'] >= 1.1)
                    )
                )
            ],
            # piecewise generally good accuracy
            [
                'piecewise', 1e-11, lambda r: (
                    np.isin(r['pct'], pct_range) &
                    np.isin(r['alpha'], alpha_range) &
                    np.isin(r['beta'], beta_range) &
                    (r['alpha'] > 0.2) &
                    (r['alpha'] != 1.)
                )
            ],
            # for alpha = 1. for linux 32 bit optimize.bisect
            # has some issues for .01 and .99 percentile
            [
                'piecewise', 1e-11, lambda r: (
                    (r['alpha'] == 1.) &
                    (not is_linux_32) &
                    np.isin(r['pct'], pct_range) &
                    (1. in alpha_range) &
                    np.isin(r['beta'], beta_range)
                )
            ],
            # for small alpha very slightly reduced accuracy
            [
                'piecewise', 2.5e-10, lambda r: (
                    np.isin(r['pct'], pct_range) &
                    np.isin(r['alpha'], alpha_range) &
                    np.isin(r['beta'], beta_range) &
                    (r['alpha'] <= 0.2)
                )
            ],
            # fft accuracy reduces as alpha decreases
            [
                'fft-simpson', 1e-5, lambda r: (
                    (r['alpha'] >= 1.9) &
                    np.isin(r['pct'], pct_range) &
                    np.isin(r['alpha'], alpha_range) &
                    np.isin(r['beta'], beta_range)
                ),
            ],
            [
                'fft-simpson', 1e-6, lambda r: (
                    np.isin(r['pct'], pct_range) &
                    np.isin(r['alpha'], alpha_range) &
                    np.isin(r['beta'], beta_range) &
                    (r['alpha'] > 1) &
                    (r['alpha'] < 1.9)
                )
            ],
            # fft relative errors for alpha < 1, will raise if enabled
            # ['fft-simpson', 1e-4, lambda r: r['alpha'] == 0.9],
            # ['fft-simpson', 1e-3, lambda r: r['alpha'] == 0.8],
            # ['fft-simpson', 1e-2, lambda r: r['alpha'] == 0.7],
            # ['fft-simpson', 1e-1, lambda r: r['alpha'] == 0.6],
        ]
        # fmt: on
        for ix, (default_method, rtol,
                 filter_func) in enumerate(tests):
            stats.levy_stable.pdf_default_method = default_method
            subdata = data[filter_func(data)
                           ] if filter_func is not None else data
            with suppress_warnings() as sup:
                # occurs in FFT methods only
                sup.record(
                    RuntimeWarning,
                    "Density calculations experimental for FFT method.*"
                )
                p = stats.levy_stable.pdf(
                    subdata['x'],
                    subdata['alpha'],
                    subdata['beta'],
                    scale=1,
                    loc=0
                )
                with np.errstate(over="ignore"):
                    subdata2 = rec_append_fields(
                        subdata,
                        ['calc', 'abserr', 'relerr'],
                        [
                            p,
                            np.abs(p - subdata['p']),
                            np.abs(p - subdata['p']) / np.abs(subdata['p'])
                        ]
                    )
                failures = subdata2[
                  (subdata2['relerr'] >= rtol) |
                  np.isnan(p)
                ]
                assert_allclose(
                    p,
                    subdata['p'],
                    rtol,
                    err_msg="pdf test %s failed with method '%s'"
                            " [platform: %s]\n%s\n%s" %
                    (ix, default_method, platform_desc, failures.dtype.names,
                        failures),
                    verbose=False
                )

    @pytest.mark.parametrize(
        "pct_range,alpha_range,beta_range", [
            pytest.param(
                [.01, .5, .99],
                [.1, 1, 2],
                [-1, 0, .8],
            ),
            pytest.param(
                [.01, .05, .5, .95, .99],
                [.1, .5, 1, 1.5, 2],
                [-.9, -.5, 0, .3, .6, 1],
                marks=pytest.mark.slow
            ),
            pytest.param(
                [.01, .05, .1, .25, .35, .5, .65, .75, .9, .95, .99],
                np.linspace(0.1, 2, 20),
                np.linspace(-1, 1, 21),
                marks=pytest.mark.xslow,
            ),
        ]
    )
    def test_cdf_nolan_samples(
            self, nolan_cdf_sample_data, pct_range, alpha_range, beta_range
    ):
        """ Test cdf values against Nolan's stablec.exe output."""
        data = nolan_cdf_sample_data
        tests = [
            # piecewise generally good accuracy
            [
                'piecewise', 2e-12, lambda r: (
                    np.isin(r['pct'], pct_range) &
                    np.isin(r['alpha'], alpha_range) &
                    np.isin(r['beta'], beta_range) &
                    ~(
                        (
                            (r['alpha'] == 1.) &
                            np.isin(r['beta'], [-0.3, -0.2, -0.1]) &
                            (r['pct'] == 0.01)
                        ) |
                        (
                            (r['alpha'] == 1.) &
                            np.isin(r['beta'], [0.1, 0.2, 0.3]) &
                            (r['pct'] == 0.99)
                        )
                    )
                )
            ],
            # for some points with alpha=1, Nolan's STABLE clearly
            # loses accuracy
            [
                'piecewise', 5e-2, lambda r: (
                    np.isin(r['pct'], pct_range) &
                    np.isin(r['alpha'], alpha_range) &
                    np.isin(r['beta'], beta_range) &
                    (
                        (r['alpha'] == 1.) &
                        np.isin(r['beta'], [-0.3, -0.2, -0.1]) &
                        (r['pct'] == 0.01)
                    ) |
                    (
                        (r['alpha'] == 1.) &
                        np.isin(r['beta'], [0.1, 0.2, 0.3]) &
                        (r['pct'] == 0.99)
                    )
                )
            ],
            # fft accuracy poor, very poor alpha < 1
            [
                'fft-simpson', 1e-5, lambda r: (
                    np.isin(r['pct'], pct_range) &
                    np.isin(r['alpha'], alpha_range) &
                    np.isin(r['beta'], beta_range) &
                    (r['alpha'] > 1.7)
                )
            ],
            [
                'fft-simpson', 1e-4, lambda r: (
                    np.isin(r['pct'], pct_range) &
                    np.isin(r['alpha'], alpha_range) &
                    np.isin(r['beta'], beta_range) &
                    (r['alpha'] > 1.5) &
                    (r['alpha'] <= 1.7)
                )
            ],
            [
                'fft-simpson', 1e-3, lambda r: (
                    np.isin(r['pct'], pct_range) &
                    np.isin(r['alpha'], alpha_range) &
                    np.isin(r['beta'], beta_range) &
                    (r['alpha'] > 1.3) &
                    (r['alpha'] <= 1.5)
                )
            ],
            [
                'fft-simpson', 1e-2, lambda r: (
                    np.isin(r['pct'], pct_range) &
                    np.isin(r['alpha'], alpha_range) &
                    np.isin(r['beta'], beta_range) &
                    (r['alpha'] > 1.0) &
                    (r['alpha'] <= 1.3)
                )
            ],
        ]
        for ix, (default_method, rtol,
                 filter_func) in enumerate(tests):
            stats.levy_stable.cdf_default_method = default_method
            subdata = data[filter_func(data)
                           ] if filter_func is not None else data
            with suppress_warnings() as sup:
                sup.record(
                    RuntimeWarning,
                    'Cumulative density calculations experimental for FFT'
                    + ' method. Use piecewise method instead.*'
                )
                p = stats.levy_stable.cdf(
                    subdata['x'],
                    subdata['alpha'],
                    subdata['beta'],
                    scale=1,
                    loc=0
                )
                with np.errstate(over="ignore"):
                    subdata2 = rec_append_fields(
                        subdata,
                        ['calc', 'abserr', 'relerr'],
                        [
                            p,
                            np.abs(p - subdata['p']),
                            np.abs(p - subdata['p']) / np.abs(subdata['p'])
                        ]
                    )
                failures = subdata2[
                  (subdata2['relerr'] >= rtol) |
                  np.isnan(p)
                ]
                assert_allclose(
                    p,
                    subdata['p'],
                    rtol,
                    err_msg="cdf test %s failed with method '%s'\n%s\n%s" %
                    (ix, default_method, failures.dtype.names, failures),
                    verbose=False
                )

    @pytest.mark.parametrize("param", [0, 1])
    @pytest.mark.parametrize("case", ["pdf", "cdf"])
    def test_location_scale(
            self, nolan_loc_scale_sample_data, param, case
    ):
        """Tests for pdf and cdf where loc, scale are different from 0, 1
        """

        uname = platform.uname()
        is_linux_32 = uname.system == 'Linux' and "32bit" in platform.architecture()[0]
        # Test seems to be unstable (see gh-17839 for a bug report on Debian
        # i386), so skip it.
        if is_linux_32 and case == 'pdf':
            pytest.skip("Test unstable on some platforms; see gh-17839, 17859")

        data = nolan_loc_scale_sample_data
        # We only test against piecewise as location/scale transforms
        # are same for other methods.
        stats.levy_stable.cdf_default_method = "piecewise"
        stats.levy_stable.pdf_default_method = "piecewise"

        subdata = data[data["param"] == param]
        stats.levy_stable.parameterization = f"S{param}"

        assert case in ["pdf", "cdf"]
        function = (
            stats.levy_stable.pdf if case == "pdf" else stats.levy_stable.cdf
        )

        v1 = function(
            subdata['x'], subdata['alpha'], subdata['beta'], scale=2, loc=3
        )
        assert_allclose(v1, subdata[case], 1e-5)

    @pytest.mark.parametrize(
        "method,decimal_places",
        [
            ['dni', 4],
            ['piecewise', 4],
        ]
    )
    def test_pdf_alpha_equals_one_beta_non_zero(self, method, decimal_places):
        """ sample points extracted from Tables and Graphs of Stable
        Probability Density Functions - Donald R Holt - 1973 - p 187.
        """
        xs = np.array(
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
        )
        density = np.array(
            [
                .3183, .3096, .2925, .2622, .1591, .1587, .1599, .1635, .0637,
                .0729, .0812, .0955, .0318, .0390, .0458, .0586, .0187, .0236,
                .0285, .0384
            ]
        )
        betas = np.array(
            [
                0, .25, .5, 1, 0, .25, .5, 1, 0, .25, .5, 1, 0, .25, .5, 1, 0,
                .25, .5, 1
            ]
        )
        with np.errstate(all='ignore'), suppress_warnings() as sup:
            sup.filter(
                category=RuntimeWarning,
                message="Density calculation unstable.*"
            )
            stats.levy_stable.pdf_default_method = method
            # stats.levy_stable.fft_grid_spacing = 0.0001
            pdf = stats.levy_stable.pdf(xs, 1, betas, scale=1, loc=0)
            assert_almost_equal(
                pdf, density, decimal_places, method
            )

    @pytest.mark.parametrize(
        "params,expected",
        [
            [(1.48, -.22, 0, 1), (0, np.inf, np.nan, np.nan)],
            [(2, .9, 10, 1.5), (10, 4.5, 0, 0)]
        ]
    )
    def test_stats(self, params, expected):
        observed = stats.levy_stable.stats(
            params[0], params[1], loc=params[2], scale=params[3],
            moments='mvsk'
        )
        assert_almost_equal(observed, expected)

    @pytest.mark.parametrize('alpha', [0.25, 0.5, 0.75])
    @pytest.mark.parametrize(
        'function,beta,points,expected',
        [
            (
                stats.levy_stable.cdf,
                1.0,
                np.linspace(-25, 0, 10),
                0.0,
            ),
            (
                stats.levy_stable.pdf,
                1.0,
                np.linspace(-25, 0, 10),
                0.0,
            ),
            (
                stats.levy_stable.cdf,
                -1.0,
                np.linspace(0, 25, 10),
                1.0,
            ),
            (
                stats.levy_stable.pdf,
                -1.0,
                np.linspace(0, 25, 10),
                0.0,
            )
        ]
    )
    def test_distribution_outside_support(
            self, alpha, function, beta, points, expected
    ):
        """Ensure the pdf/cdf routines do not return nan outside support.

        This distribution's support becomes truncated in a few special cases:
            support is [mu, infty) if alpha < 1 and beta = 1
            support is (-infty, mu] if alpha < 1 and beta = -1
        Otherwise, the support is all reals. Here, mu is zero by default.
        """
        assert 0 < alpha < 1
        assert_almost_equal(
            function(points, alpha=alpha, beta=beta),
            np.full(len(points), expected)
        )


class TestArrayArgument:  # test for ticket:992
    def setup_method(self):
        np.random.seed(1234)

    def test_noexception(self):
        rvs = stats.norm.rvs(loc=(np.arange(5)), scale=np.ones(5),
                             size=(10, 5))
        assert_equal(rvs.shape, (10, 5))


class TestDocstring:
    def test_docstrings(self):
        # See ticket #761
        if stats.rayleigh.__doc__ is not None:
            assert_("rayleigh" in stats.rayleigh.__doc__.lower())
        if stats.bernoulli.__doc__ is not None:
            assert_("bernoulli" in stats.bernoulli.__doc__.lower())

    def test_no_name_arg(self):
        # If name is not given, construction shouldn't fail.  See #1508.
        stats.rv_continuous()
        stats.rv_discrete()


def test_args_reduce():
    a = array([1, 3, 2, 1, 2, 3, 3])
    b, c = argsreduce(a > 1, a, 2)

    assert_array_equal(b, [3, 2, 2, 3, 3])
    assert_array_equal(c, [2])

    b, c = argsreduce(2 > 1, a, 2)
    assert_array_equal(b, a)
    assert_array_equal(c, [2] * np.size(a))

    b, c = argsreduce(a > 0, a, 2)
    assert_array_equal(b, a)
    assert_array_equal(c, [2] * np.size(a))


class TestFitMethod:
    skip = ['ncf', 'ksone', 'kstwo']

    def setup_method(self):
        np.random.seed(1234)

    # skip these b/c deprecated, or only loc and scale arguments
    fitSkipNonFinite = ['expon', 'norm', 'uniform']

    @pytest.mark.parametrize('dist,args', distcont)
    def test_fit_w_non_finite_data_values(self, dist, args):
        """gh-10300"""
        if dist in self.fitSkipNonFinite:
            pytest.skip("%s fit known to fail or deprecated" % dist)
        x = np.array([1.6483, 2.7169, 2.4667, 1.1791, 3.5433, np.nan])
        y = np.array([1.6483, 2.7169, 2.4667, 1.1791, 3.5433, np.inf])
        distfunc = getattr(stats, dist)
        assert_raises(ValueError, distfunc.fit, x, fscale=1)
        assert_raises(ValueError, distfunc.fit, y, fscale=1)

    def test_fix_fit_2args_lognorm(self):
        # Regression test for #1551.
        np.random.seed(12345)
        with np.errstate(all='ignore'):
            x = stats.lognorm.rvs(0.25, 0., 20.0, size=20)
            expected_shape = np.sqrt(((np.log(x) - np.log(20))**2).mean())
            assert_allclose(np.array(stats.lognorm.fit(x, floc=0, fscale=20)),
                            [expected_shape, 0, 20], atol=1e-8)

    def test_fix_fit_norm(self):
        x = np.arange(1, 6)

        loc, scale = stats.norm.fit(x)
        assert_almost_equal(loc, 3)
        assert_almost_equal(scale, np.sqrt(2))

        loc, scale = stats.norm.fit(x, floc=2)
        assert_equal(loc, 2)
        assert_equal(scale, np.sqrt(3))

        loc, scale = stats.norm.fit(x, fscale=2)
        assert_almost_equal(loc, 3)
        assert_equal(scale, 2)

    def test_fix_fit_gamma(self):
        x = np.arange(1, 6)
        meanlog = np.log(x).mean()

        # A basic test of gamma.fit with floc=0.
        floc = 0
        a, loc, scale = stats.gamma.fit(x, floc=floc)
        s = np.log(x.mean()) - meanlog
        assert_almost_equal(np.log(a) - special.digamma(a), s, decimal=5)
        assert_equal(loc, floc)
        assert_almost_equal(scale, x.mean()/a, decimal=8)

        # Regression tests for gh-2514.
        # The problem was that if `floc=0` was given, any other fixed
        # parameters were ignored.
        f0 = 1
        floc = 0
        a, loc, scale = stats.gamma.fit(x, f0=f0, floc=floc)
        assert_equal(a, f0)
        assert_equal(loc, floc)
        assert_almost_equal(scale, x.mean()/a, decimal=8)

        f0 = 2
        floc = 0
        a, loc, scale = stats.gamma.fit(x, f0=f0, floc=floc)
        assert_equal(a, f0)
        assert_equal(loc, floc)
        assert_almost_equal(scale, x.mean()/a, decimal=8)

        # loc and scale fixed.
        floc = 0
        fscale = 2
        a, loc, scale = stats.gamma.fit(x, floc=floc, fscale=fscale)
        assert_equal(loc, floc)
        assert_equal(scale, fscale)
        c = meanlog - np.log(fscale)
        assert_almost_equal(special.digamma(a), c)

    def test_fix_fit_beta(self):
        # Test beta.fit when both floc and fscale are given.

        def mlefunc(a, b, x):
            # Zeros of this function are critical points of
            # the maximum likelihood function.
            n = len(x)
            s1 = np.log(x).sum()
            s2 = np.log(1-x).sum()
            psiab = special.psi(a + b)
            func = [s1 - n * (-psiab + special.psi(a)),
                    s2 - n * (-psiab + special.psi(b))]
            return func

        # Basic test with floc and fscale given.
        x = np.array([0.125, 0.25, 0.5])
        a, b, loc, scale = stats.beta.fit(x, floc=0, fscale=1)
        assert_equal(loc, 0)
        assert_equal(scale, 1)
        assert_allclose(mlefunc(a, b, x), [0, 0], atol=1e-6)

        # Basic test with f0, floc and fscale given.
        # This is also a regression test for gh-2514.
        x = np.array([0.125, 0.25, 0.5])
        a, b, loc, scale = stats.beta.fit(x, f0=2, floc=0, fscale=1)
        assert_equal(a, 2)
        assert_equal(loc, 0)
        assert_equal(scale, 1)
        da, db = mlefunc(a, b, x)
        assert_allclose(db, 0, atol=1e-5)

        # Same floc and fscale values as above, but reverse the data
        # and fix b (f1).
        x2 = 1 - x
        a2, b2, loc2, scale2 = stats.beta.fit(x2, f1=2, floc=0, fscale=1)
        assert_equal(b2, 2)
        assert_equal(loc2, 0)
        assert_equal(scale2, 1)
        da, db = mlefunc(a2, b2, x2)
        assert_allclose(da, 0, atol=1e-5)
        # a2 of this test should equal b from above.
        assert_almost_equal(a2, b)

        # Check for detection of data out of bounds when floc and fscale
        # are given.
        assert_raises(ValueError, stats.beta.fit, x, floc=0.5, fscale=1)
        y = np.array([0, .5, 1])
        assert_raises(ValueError, stats.beta.fit, y, floc=0, fscale=1)
        assert_raises(ValueError, stats.beta.fit, y, floc=0, fscale=1, f0=2)
        assert_raises(ValueError, stats.beta.fit, y, floc=0, fscale=1, f1=2)

        # Check that attempting to fix all the parameters raises a ValueError.
        assert_raises(ValueError, stats.beta.fit, y, f0=0, f1=1,
                      floc=2, fscale=3)

    def test_expon_fit(self):
        x = np.array([2, 2, 4, 4, 4, 4, 4, 8])

        loc, scale = stats.expon.fit(x)
        assert_equal(loc, 2)    # x.min()
        assert_equal(scale, 2)  # x.mean() - x.min()

        loc, scale = stats.expon.fit(x, fscale=3)
        assert_equal(loc, 2)    # x.min()
        assert_equal(scale, 3)  # fscale

        loc, scale = stats.expon.fit(x, floc=0)
        assert_equal(loc, 0)    # floc
        assert_equal(scale, 4)  # x.mean() - loc

    def test_lognorm_fit(self):
        x = np.array([1.5, 3, 10, 15, 23, 59])
        lnxm1 = np.log(x - 1)

        shape, loc, scale = stats.lognorm.fit(x, floc=1)
        assert_allclose(shape, lnxm1.std(), rtol=1e-12)
        assert_equal(loc, 1)
        assert_allclose(scale, np.exp(lnxm1.mean()), rtol=1e-12)

        shape, loc, scale = stats.lognorm.fit(x, floc=1, fscale=6)
        assert_allclose(shape, np.sqrt(((lnxm1 - np.log(6))**2).mean()),
                        rtol=1e-12)
        assert_equal(loc, 1)
        assert_equal(scale, 6)

        shape, loc, scale = stats.lognorm.fit(x, floc=1, fix_s=0.75)
        assert_equal(shape, 0.75)
        assert_equal(loc, 1)
        assert_allclose(scale, np.exp(lnxm1.mean()), rtol=1e-12)

    def test_uniform_fit(self):
        x = np.array([1.0, 1.1, 1.2, 9.0])

        loc, scale = stats.uniform.fit(x)
        assert_equal(loc, x.min())
        assert_equal(scale, x.ptp())

        loc, scale = stats.uniform.fit(x, floc=0)
        assert_equal(loc, 0)
        assert_equal(scale, x.max())

        loc, scale = stats.uniform.fit(x, fscale=10)
        assert_equal(loc, 0)
        assert_equal(scale, 10)

        assert_raises(ValueError, stats.uniform.fit, x, floc=2.0)
        assert_raises(ValueError, stats.uniform.fit, x, fscale=5.0)

    @pytest.mark.slow
    @pytest.mark.parametrize("method", ["MLE", "MM"])
    def test_fshapes(self, method):
        # take a beta distribution, with shapes='a, b', and make sure that
        # fa is equivalent to f0, and fb is equivalent to f1
        a, b = 3., 4.
        x = stats.beta.rvs(a, b, size=100, random_state=1234)
        res_1 = stats.beta.fit(x, f0=3., method=method)
        res_2 = stats.beta.fit(x, fa=3., method=method)
        assert_allclose(res_1, res_2, atol=1e-12, rtol=1e-12)

        res_2 = stats.beta.fit(x, fix_a=3., method=method)
        assert_allclose(res_1, res_2, atol=1e-12, rtol=1e-12)

        res_3 = stats.beta.fit(x, f1=4., method=method)
        res_4 = stats.beta.fit(x, fb=4., method=method)
        assert_allclose(res_3, res_4, atol=1e-12, rtol=1e-12)

        res_4 = stats.beta.fit(x, fix_b=4., method=method)
        assert_allclose(res_3, res_4, atol=1e-12, rtol=1e-12)

        # cannot specify both positional and named args at the same time
        assert_raises(ValueError, stats.beta.fit, x, fa=1, f0=2, method=method)

        # check that attempting to fix all parameters raises a ValueError
        assert_raises(ValueError, stats.beta.fit, x, fa=0, f1=1,
                      floc=2, fscale=3, method=method)

        # check that specifying floc, fscale and fshapes works for
        # beta and gamma which override the generic fit method
        res_5 = stats.beta.fit(x, fa=3., floc=0, fscale=1, method=method)
        aa, bb, ll, ss = res_5
        assert_equal([aa, ll, ss], [3., 0, 1])

        # gamma distribution
        a = 3.
        data = stats.gamma.rvs(a, size=100)
        aa, ll, ss = stats.gamma.fit(data, fa=a, method=method)
        assert_equal(aa, a)

    @pytest.mark.parametrize("method", ["MLE", "MM"])
    def test_extra_params(self, method):
        # unknown parameters should raise rather than be silently ignored
        dist = stats.exponnorm
        data = dist.rvs(K=2, size=100)
        dct = dict(enikibeniki=-101)
        assert_raises(TypeError, dist.fit, data, **dct, method=method)


class TestFrozen:
    def setup_method(self):
        np.random.seed(1234)

    # Test that a frozen distribution gives the same results as the original
    # object.
    #
    # Only tested for the normal distribution (with loc and scale specified)
    # and for the gamma distribution (with a shape parameter specified).
    def test_norm(self):
        dist = stats.norm
        frozen = stats.norm(loc=10.0, scale=3.0)

        result_f = frozen.pdf(20.0)
        result = dist.pdf(20.0, loc=10.0, scale=3.0)
        assert_equal(result_f, result)

        result_f = frozen.cdf(20.0)
        result = dist.cdf(20.0, loc=10.0, scale=3.0)
        assert_equal(result_f, result)

        result_f = frozen.ppf(0.25)
        result = dist.ppf(0.25, loc=10.0, scale=3.0)
        assert_equal(result_f, result)

        result_f = frozen.isf(0.25)
        result = dist.isf(0.25, loc=10.0, scale=3.0)
        assert_equal(result_f, result)

        result_f = frozen.sf(10.0)
        result = dist.sf(10.0, loc=10.0, scale=3.0)
        assert_equal(result_f, result)

        result_f = frozen.median()
        result = dist.median(loc=10.0, scale=3.0)
        assert_equal(result_f, result)

        result_f = frozen.mean()
        result = dist.mean(loc=10.0, scale=3.0)
        assert_equal(result_f, result)

        result_f = frozen.var()
        result = dist.var(loc=10.0, scale=3.0)
        assert_equal(result_f, result)

        result_f = frozen.std()
        result = dist.std(loc=10.0, scale=3.0)
        assert_equal(result_f, result)

        result_f = frozen.entropy()
        result = dist.entropy(loc=10.0, scale=3.0)
        assert_equal(result_f, result)

        result_f = frozen.moment(2)
        result = dist.moment(2, loc=10.0, scale=3.0)
        assert_equal(result_f, result)

        assert_equal(frozen.a, dist.a)
        assert_equal(frozen.b, dist.b)

    def test_gamma(self):
        a = 2.0
        dist = stats.gamma
        frozen = stats.gamma(a)

        result_f = frozen.pdf(20.0)
        result = dist.pdf(20.0, a)
        assert_equal(result_f, result)

        result_f = frozen.cdf(20.0)
        result = dist.cdf(20.0, a)
        assert_equal(result_f, result)

        result_f = frozen.ppf(0.25)
        result = dist.ppf(0.25, a)
        assert_equal(result_f, result)

        result_f = frozen.isf(0.25)
        result = dist.isf(0.25, a)
        assert_equal(result_f, result)

        result_f = frozen.sf(10.0)
        result = dist.sf(10.0, a)
        assert_equal(result_f, result)

        result_f = frozen.median()
        result = dist.median(a)
        assert_equal(result_f, result)

        result_f = frozen.mean()
        result = dist.mean(a)
        assert_equal(result_f, result)

        result_f = frozen.var()
        result = dist.var(a)
        assert_equal(result_f, result)

        result_f = frozen.std()
        result = dist.std(a)
        assert_equal(result_f, result)

        result_f = frozen.entropy()
        result = dist.entropy(a)
        assert_equal(result_f, result)

        result_f = frozen.moment(2)
        result = dist.moment(2, a)
        assert_equal(result_f, result)

        assert_equal(frozen.a, frozen.dist.a)
        assert_equal(frozen.b, frozen.dist.b)

    def test_regression_ticket_1293(self):
        # Create a frozen distribution.
        frozen = stats.lognorm(1)
        # Call one of its methods that does not take any keyword arguments.
        m1 = frozen.moment(2)
        # Now call a method that takes a keyword argument.
        frozen.stats(moments='mvsk')
        # Call moment(2) again.
        # After calling stats(), the following was raising an exception.
        # So this test passes if the following does not raise an exception.
        m2 = frozen.moment(2)
        # The following should also be true, of course.  But it is not
        # the focus of this test.
        assert_equal(m1, m2)

    def test_ab(self):
        # test that the support of a frozen distribution
        # (i) remains frozen even if it changes for the original one
        # (ii) is actually correct if the shape parameters are such that
        #      the values of [a, b] are not the default [0, inf]
        # take a genpareto as an example where the support
        # depends on the value of the shape parameter:
        # for c > 0: a, b = 0, inf
        # for c < 0: a, b = 0, -1/c

        c = -0.1
        rv = stats.genpareto(c=c)
        a, b = rv.dist._get_support(c)
        assert_equal([a, b], [0., 10.])

        c = 0.1
        stats.genpareto.pdf(0, c=c)
        assert_equal(rv.dist._get_support(c), [0, np.inf])

        c = -0.1
        rv = stats.genpareto(c=c)
        a, b = rv.dist._get_support(c)
        assert_equal([a, b], [0., 10.])

        c = 0.1
        stats.genpareto.pdf(0, c)  # this should NOT change genpareto.b
        assert_equal((rv.dist.a, rv.dist.b), stats.genpareto._get_support(c))

        rv1 = stats.genpareto(c=0.1)
        assert_(rv1.dist is not rv.dist)

        # c >= 0: a, b = [0, inf]
        for c in [1., 0.]:
            c = np.asarray(c)
            rv = stats.genpareto(c=c)
            a, b = rv.a, rv.b
            assert_equal(a, 0.)
            assert_(np.isposinf(b))

            # c < 0: a=0, b=1/|c|
            c = np.asarray(-2.)
            a, b = stats.genpareto._get_support(c)
            assert_allclose([a, b], [0., 0.5])

    def test_rv_frozen_in_namespace(self):
        # Regression test for gh-3522
        assert_(hasattr(stats.distributions, 'rv_frozen'))

    def test_random_state(self):
        # only check that the random_state attribute exists,
        frozen = stats.norm()
        assert_(hasattr(frozen, 'random_state'))

        # ... that it can be set,
        frozen.random_state = 42
        assert_equal(frozen.random_state.get_state(),
                     np.random.RandomState(42).get_state())

        # ... and that .rvs method accepts it as an argument
        rndm = np.random.RandomState(1234)
        frozen.rvs(size=8, random_state=rndm)

    def test_pickling(self):
        # test that a frozen instance pickles and unpickles
        # (this method is a clone of common_tests.check_pickling)
        beta = stats.beta(2.3098496451481823, 0.62687954300963677)
        poiss = stats.poisson(3.)
        sample = stats.rv_discrete(values=([0, 1, 2, 3],
                                           [0.1, 0.2, 0.3, 0.4]))

        for distfn in [beta, poiss, sample]:
            distfn.random_state = 1234
            distfn.rvs(size=8)
            s = pickle.dumps(distfn)
            r0 = distfn.rvs(size=8)

            unpickled = pickle.loads(s)
            r1 = unpickled.rvs(size=8)
            assert_equal(r0, r1)

            # also smoke test some methods
            medians = [distfn.ppf(0.5), unpickled.ppf(0.5)]
            assert_equal(medians[0], medians[1])
            assert_equal(distfn.cdf(medians[0]),
                         unpickled.cdf(medians[1]))

    def test_expect(self):
        # smoke test the expect method of the frozen distribution
        # only take a gamma w/loc and scale and poisson with loc specified
        def func(x):
            return x

        gm = stats.gamma(a=2, loc=3, scale=4)
        with np.errstate(invalid="ignore", divide="ignore"):
            gm_val = gm.expect(func, lb=1, ub=2, conditional=True)
            gamma_val = stats.gamma.expect(func, args=(2,), loc=3, scale=4,
                                           lb=1, ub=2, conditional=True)
        assert_allclose(gm_val, gamma_val)

        p = stats.poisson(3, loc=4)
        p_val = p.expect(func)
        poisson_val = stats.poisson.expect(func, args=(3,), loc=4)
        assert_allclose(p_val, poisson_val)


class TestExpect:
    # Test for expect method.
    #
    # Uses normal distribution and beta distribution for finite bounds, and
    # hypergeom for discrete distribution with finite support
    def test_norm(self):
        v = stats.norm.expect(lambda x: (x-5)*(x-5), loc=5, scale=2)
        assert_almost_equal(v, 4, decimal=14)

        m = stats.norm.expect(lambda x: (x), loc=5, scale=2)
        assert_almost_equal(m, 5, decimal=14)

        lb = stats.norm.ppf(0.05, loc=5, scale=2)
        ub = stats.norm.ppf(0.95, loc=5, scale=2)
        prob90 = stats.norm.expect(lambda x: 1, loc=5, scale=2, lb=lb, ub=ub)
        assert_almost_equal(prob90, 0.9, decimal=14)

        prob90c = stats.norm.expect(lambda x: 1, loc=5, scale=2, lb=lb, ub=ub,
                                    conditional=True)
        assert_almost_equal(prob90c, 1., decimal=14)

    def test_beta(self):
        # case with finite support interval
        v = stats.beta.expect(lambda x: (x-19/3.)*(x-19/3.), args=(10, 5),
                              loc=5, scale=2)
        assert_almost_equal(v, 1./18., decimal=13)

        m = stats.beta.expect(lambda x: x, args=(10, 5), loc=5., scale=2.)
        assert_almost_equal(m, 19/3., decimal=13)

        ub = stats.beta.ppf(0.95, 10, 10, loc=5, scale=2)
        lb = stats.beta.ppf(0.05, 10, 10, loc=5, scale=2)
        prob90 = stats.beta.expect(lambda x: 1., args=(10, 10), loc=5.,
                                   scale=2., lb=lb, ub=ub, conditional=False)
        assert_almost_equal(prob90, 0.9, decimal=13)

        prob90c = stats.beta.expect(lambda x: 1, args=(10, 10), loc=5,
                                    scale=2, lb=lb, ub=ub, conditional=True)
        assert_almost_equal(prob90c, 1., decimal=13)

    def test_hypergeom(self):
        # test case with finite bounds

        # without specifying bounds
        m_true, v_true = stats.hypergeom.stats(20, 10, 8, loc=5.)
        m = stats.hypergeom.expect(lambda x: x, args=(20, 10, 8), loc=5.)
        assert_almost_equal(m, m_true, decimal=13)

        v = stats.hypergeom.expect(lambda x: (x-9.)**2, args=(20, 10, 8),
                                   loc=5.)
        assert_almost_equal(v, v_true, decimal=14)

        # with bounds, bounds equal to shifted support
        v_bounds = stats.hypergeom.expect(lambda x: (x-9.)**2,
                                          args=(20, 10, 8),
                                          loc=5., lb=5, ub=13)
        assert_almost_equal(v_bounds, v_true, decimal=14)

        # drop boundary points
        prob_true = 1-stats.hypergeom.pmf([5, 13], 20, 10, 8, loc=5).sum()
        prob_bounds = stats.hypergeom.expect(lambda x: 1, args=(20, 10, 8),
                                             loc=5., lb=6, ub=12)
        assert_almost_equal(prob_bounds, prob_true, decimal=13)

        # conditional
        prob_bc = stats.hypergeom.expect(lambda x: 1, args=(20, 10, 8), loc=5.,
                                         lb=6, ub=12, conditional=True)
        assert_almost_equal(prob_bc, 1, decimal=14)

        # check simple integral
        prob_b = stats.hypergeom.expect(lambda x: 1, args=(20, 10, 8),
                                        lb=0, ub=8)
        assert_almost_equal(prob_b, 1, decimal=13)

    def test_poisson(self):
        # poisson, use lower bound only
        prob_bounds = stats.poisson.expect(lambda x: 1, args=(2,), lb=3,
                                           conditional=False)
        prob_b_true = 1-stats.poisson.cdf(2, 2)
        assert_almost_equal(prob_bounds, prob_b_true, decimal=14)

        prob_lb = stats.poisson.expect(lambda x: 1, args=(2,), lb=2,
                                       conditional=True)
        assert_almost_equal(prob_lb, 1, decimal=14)

    def test_genhalflogistic(self):
        # genhalflogistic, changes upper bound of support in _argcheck
        # regression test for gh-2622
        halflog = stats.genhalflogistic
        # check consistency when calling expect twice with the same input
        res1 = halflog.expect(args=(1.5,))
        halflog.expect(args=(0.5,))
        res2 = halflog.expect(args=(1.5,))
        assert_almost_equal(res1, res2, decimal=14)

    def test_rice_overflow(self):
        # rice.pdf(999, 0.74) was inf since special.i0 silentyly overflows
        # check that using i0e fixes it
        assert_(np.isfinite(stats.rice.pdf(999, 0.74)))

        assert_(np.isfinite(stats.rice.expect(lambda x: 1, args=(0.74,))))
        assert_(np.isfinite(stats.rice.expect(lambda x: 2, args=(0.74,))))
        assert_(np.isfinite(stats.rice.expect(lambda x: 3, args=(0.74,))))

    def test_logser(self):
        # test a discrete distribution with infinite support and loc
        p, loc = 0.3, 3
        res_0 = stats.logser.expect(lambda k: k, args=(p,))
        # check against the correct answer (sum of a geom series)
        assert_allclose(res_0,
                        p / (p - 1.) / np.log(1. - p), atol=1e-15)

        # now check it with `loc`
        res_l = stats.logser.expect(lambda k: k, args=(p,), loc=loc)
        assert_allclose(res_l, res_0 + loc, atol=1e-15)

    def test_skellam(self):
        # Use a discrete distribution w/ bi-infinite support. Compute two first
        # moments and compare to known values (cf skellam.stats)
        p1, p2 = 18, 22
        m1 = stats.skellam.expect(lambda x: x, args=(p1, p2))
        m2 = stats.skellam.expect(lambda x: x**2, args=(p1, p2))
        assert_allclose(m1, p1 - p2, atol=1e-12)
        assert_allclose(m2 - m1**2, p1 + p2, atol=1e-12)

    def test_randint(self):
        # Use a discrete distribution w/ parameter-dependent support, which
        # is larger than the default chunksize
        lo, hi = 0, 113
        res = stats.randint.expect(lambda x: x, (lo, hi))
        assert_allclose(res,
                        sum(_ for _ in range(lo, hi)) / (hi - lo), atol=1e-15)

    def test_zipf(self):
        # Test that there is no infinite loop even if the sum diverges
        assert_warns(RuntimeWarning, stats.zipf.expect,
                     lambda x: x**2, (2,))

    def test_discrete_kwds(self):
        # check that discrete expect accepts keywords to control the summation
        n0 = stats.poisson.expect(lambda x: 1, args=(2,))
        n1 = stats.poisson.expect(lambda x: 1, args=(2,),
                                  maxcount=1001, chunksize=32, tolerance=1e-8)
        assert_almost_equal(n0, n1, decimal=14)

    def test_moment(self):
        # test the .moment() method: compute a higher moment and compare to
        # a known value
        def poiss_moment5(mu):
            return mu**5 + 10*mu**4 + 25*mu**3 + 15*mu**2 + mu

        for mu in [5, 7]:
            m5 = stats.poisson.moment(5, mu)
            assert_allclose(m5, poiss_moment5(mu), rtol=1e-10)

    def test_challenging_cases_gh8928(self):
        # Several cases where `expect` failed to produce a correct result were
        # reported in gh-8928. Check that these cases have been resolved.
        assert_allclose(stats.norm.expect(loc=36, scale=1.0), 36)
        assert_allclose(stats.norm.expect(loc=40, scale=1.0), 40)
        assert_allclose(stats.norm.expect(loc=10, scale=0.1), 10)
        assert_allclose(stats.gamma.expect(args=(148,)), 148)
        assert_allclose(stats.logistic.expect(loc=85), 85)

    def test_lb_ub_gh15855(self):
        # Make sure changes to `expect` made in gh15855 treat lb/ub correctly
        dist = stats.uniform
        ref = dist.mean(loc=10, scale=5)  # 12.5
        # moment over whole distribution
        assert_allclose(dist.expect(loc=10, scale=5), ref)
        # moment over whole distribution, lb and ub outside of support
        assert_allclose(dist.expect(loc=10, scale=5, lb=9, ub=16), ref)
        # moment over 60% of distribution, [lb, ub] centered within support
        assert_allclose(dist.expect(loc=10, scale=5, lb=11, ub=14), ref*0.6)
        # moment over truncated distribution, essentially
        assert_allclose(dist.expect(loc=10, scale=5, lb=11, ub=14,
                                    conditional=True), ref)
        # moment over 40% of distribution, [lb, ub] not centered within support
        assert_allclose(dist.expect(loc=10, scale=5, lb=11, ub=13), 12*0.4)
        # moment with lb > ub
        assert_allclose(dist.expect(loc=10, scale=5, lb=13, ub=11), -12*0.4)
        # moment with lb > ub, conditional
        assert_allclose(dist.expect(loc=10, scale=5, lb=13, ub=11,
                                    conditional=True), 12)


class TestNct:
    def test_nc_parameter(self):
        # Parameter values c<=0 were not enabled (gh-2402).
        # For negative values c and for c=0 results of rv.cdf(0) below were nan
        rv = stats.nct(5, 0)
        assert_equal(rv.cdf(0), 0.5)
        rv = stats.nct(5, -1)
        assert_almost_equal(rv.cdf(0), 0.841344746069, decimal=10)

    def test_broadcasting(self):
        res = stats.nct.pdf(5, np.arange(4, 7)[:, None],
                            np.linspace(0.1, 1, 4))
        expected = array([[0.00321886, 0.00557466, 0.00918418, 0.01442997],
                          [0.00217142, 0.00395366, 0.00683888, 0.01126276],
                          [0.00153078, 0.00291093, 0.00525206, 0.00900815]])
        assert_allclose(res, expected, rtol=1e-5)

    def test_variance_gh_issue_2401(self):
        # Computation of the variance of a non-central t-distribution resulted
        # in a TypeError: ufunc 'isinf' not supported for the input types,
        # and the inputs could not be safely coerced to any supported types
        # according to the casting rule 'safe'
        rv = stats.nct(4, 0)
        assert_equal(rv.var(), 2.0)

    def test_nct_inf_moments(self):
        # n-th moment of nct only exists for df > n
        m, v, s, k = stats.nct.stats(df=0.9, nc=0.3, moments='mvsk')
        assert_equal([m, v, s, k], [np.nan, np.nan, np.nan, np.nan])

        m, v, s, k = stats.nct.stats(df=1.9, nc=0.3, moments='mvsk')
        assert_(np.isfinite(m))
        assert_equal([v, s, k], [np.nan, np.nan, np.nan])

        m, v, s, k = stats.nct.stats(df=3.1, nc=0.3, moments='mvsk')
        assert_(np.isfinite([m, v, s]).all())
        assert_equal(k, np.nan)

    def test_nct_stats_large_df_values(self):
        # previously gamma function was used which lost precision at df=345
        # cf. https://github.com/scipy/scipy/issues/12919 for details
        nct_mean_df_1000 = stats.nct.mean(1000, 2)
        nct_stats_df_1000 = stats.nct.stats(1000, 2)
        # These expected values were computed with mpmath. They were also
        # verified with the Wolfram Alpha expressions:
        #     Mean[NoncentralStudentTDistribution[1000, 2]]
        #     Var[NoncentralStudentTDistribution[1000, 2]]
        expected_stats_df_1000 = [2.0015015641422464, 1.0040115288163005]
        assert_allclose(nct_mean_df_1000, expected_stats_df_1000[0],
                        rtol=1e-10)
        assert_allclose(nct_stats_df_1000, expected_stats_df_1000,
                        rtol=1e-10)
        # and a bigger df value
        nct_mean = stats.nct.mean(100000, 2)
        nct_stats = stats.nct.stats(100000, 2)
        # These expected values were computed with mpmath.
        expected_stats = [2.0000150001562518, 1.0000400011500288]
        assert_allclose(nct_mean, expected_stats[0], rtol=1e-10)
        assert_allclose(nct_stats, expected_stats, rtol=1e-9)

    def test_cdf_large_nc(self):
        # gh-17916 reported a crash with large `nc` values
        assert_allclose(stats.nct.cdf(2, 2, float(2**16)), 0)


class TestRecipInvGauss:

    def test_pdf_endpoint(self):
        p = stats.recipinvgauss.pdf(0, 0.6)
        assert p == 0.0

    def test_logpdf_endpoint(self):
        logp = stats.recipinvgauss.logpdf(0, 0.6)
        assert logp == -np.inf

    def test_cdf_small_x(self):
        # The expected value was computer with mpmath:
        #
        # import mpmath
        #
        # mpmath.mp.dps = 100
        #
        # def recipinvgauss_cdf_mp(x, mu):
        #     x = mpmath.mpf(x)
        #     mu = mpmath.mpf(mu)
        #     trm1 = 1/mu - x
        #     trm2 = 1/mu + x
        #     isqx = 1/mpmath.sqrt(x)
        #     return (mpmath.ncdf(-isqx*trm1)
        #             - mpmath.exp(2/mu)*mpmath.ncdf(-isqx*trm2))
        #
        p = stats.recipinvgauss.cdf(0.05, 0.5)
        expected = 6.590396159501331e-20
        assert_allclose(p, expected, rtol=1e-14)

    def test_sf_large_x(self):
        # The expected value was computed with mpmath; see test_cdf_small.
        p = stats.recipinvgauss.sf(80, 0.5)
        expected = 2.699819200556787e-18
        assert_allclose(p, expected, 5e-15)


class TestRice:
    def test_rice_zero_b(self):
        # rice distribution should work with b=0, cf gh-2164
        x = [0.2, 1., 5.]
        assert_(np.isfinite(stats.rice.pdf(x, b=0.)).all())
        assert_(np.isfinite(stats.rice.logpdf(x, b=0.)).all())
        assert_(np.isfinite(stats.rice.cdf(x, b=0.)).all())
        assert_(np.isfinite(stats.rice.logcdf(x, b=0.)).all())

        q = [0.1, 0.1, 0.5, 0.9]
        assert_(np.isfinite(stats.rice.ppf(q, b=0.)).all())

        mvsk = stats.rice.stats(0, moments='mvsk')
        assert_(np.isfinite(mvsk).all())

        # furthermore, pdf is continuous as b\to 0
        # rice.pdf(x, b\to 0) = x exp(-x^2/2) + O(b^2)
        # see e.g. Abramovich & Stegun 9.6.7 & 9.6.10
        b = 1e-8
        assert_allclose(stats.rice.pdf(x, 0), stats.rice.pdf(x, b),
                        atol=b, rtol=0)

    def test_rice_rvs(self):
        rvs = stats.rice.rvs
        assert_equal(rvs(b=3.).size, 1)
        assert_equal(rvs(b=3., size=(3, 5)).shape, (3, 5))

    def test_rice_gh9836(self):
        # test that gh-9836 is resolved; previously jumped to 1 at the end

        cdf = stats.rice.cdf(np.arange(10, 160, 10), np.arange(10, 160, 10))
        # Generated in R
        # library(VGAM)
        # options(digits=16)
        # x = seq(10, 150, 10)
        # print(price(x, sigma=1, vee=x))
        cdf_exp = [0.4800278103504522, 0.4900233218590353, 0.4933500379379548,
                   0.4950128317658719, 0.4960103776798502, 0.4966753655438764,
                   0.4971503395812474, 0.4975065620443196, 0.4977836197921638,
                   0.4980052636649550, 0.4981866072661382, 0.4983377260666599,
                   0.4984655952615694, 0.4985751970541413, 0.4986701850071265]
        assert_allclose(cdf, cdf_exp)

        probabilities = np.arange(0.1, 1, 0.1)
        ppf = stats.rice.ppf(probabilities, 500/4, scale=4)
        # Generated in R
        # library(VGAM)
        # options(digits=16)
        # p = seq(0.1, .9, by = .1)
        # print(qrice(p, vee = 500, sigma = 4))
        ppf_exp = [494.8898762347361, 496.6495690858350, 497.9184315188069,
                   499.0026277378915, 500.0159999146250, 501.0293721352668,
                   502.1135684981884, 503.3824312270405, 505.1421247157822]
        assert_allclose(ppf, ppf_exp)

        ppf = scipy.stats.rice.ppf(0.5, np.arange(10, 150, 10))
        # Generated in R
        # library(VGAM)
        # options(digits=16)
        # b <- seq(10, 140, 10)
        # print(qrice(0.5, vee = b, sigma = 1))
        ppf_exp = [10.04995862522287, 20.02499480078302, 30.01666512465732,
                   40.01249934924363, 50.00999966676032, 60.00833314046875,
                   70.00714273568241, 80.00624991862573, 90.00555549840364,
                   100.00499995833597, 110.00454542324384, 120.00416664255323,
                   130.00384613488120, 140.00357141338748]
        assert_allclose(ppf, ppf_exp)


class TestErlang:
    def setup_method(self):
        np.random.seed(1234)

    def test_erlang_runtimewarning(self):
        # erlang should generate a RuntimeWarning if a non-integer
        # shape parameter is used.
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)

            # The non-integer shape parameter 1.3 should trigger a
            # RuntimeWarning
            assert_raises(RuntimeWarning,
                          stats.erlang.rvs, 1.3, loc=0, scale=1, size=4)

            # Calling the fit method with `f0` set to an integer should
            # *not* trigger a RuntimeWarning.  It should return the same
            # values as gamma.fit(...).
            data = [0.5, 1.0, 2.0, 4.0]
            result_erlang = stats.erlang.fit(data, f0=1)
            result_gamma = stats.gamma.fit(data, f0=1)
            assert_allclose(result_erlang, result_gamma, rtol=1e-3)

    def test_gh_pr_10949_argcheck(self):
        assert_equal(stats.erlang.pdf(0.5, a=[1, -1]),
                     stats.gamma.pdf(0.5, a=[1, -1]))


class TestRayleigh:
    def setup_method(self):
        np.random.seed(987654321)

    # gh-6227
    def test_logpdf(self):
        y = stats.rayleigh.logpdf(50)
        assert_allclose(y, -1246.0879769945718)

    def test_logsf(self):
        y = stats.rayleigh.logsf(50)
        assert_allclose(y, -1250)

    @pytest.mark.parametrize("rvs_loc,rvs_scale", [(0.85373171, 0.86932204),
                                                   (0.20558821, 0.61621008)])
    def test_fit(self, rvs_loc, rvs_scale):
        data = stats.rayleigh.rvs(size=250, loc=rvs_loc, scale=rvs_scale)

        def scale_mle(data, floc):
            return (np.sum((data - floc) ** 2) / (2 * len(data))) ** .5

        # when `floc` is provided, `scale` is found with an analytical formula
        scale_expect = scale_mle(data, rvs_loc)
        loc, scale = stats.rayleigh.fit(data, floc=rvs_loc)
        assert_equal(loc, rvs_loc)
        assert_equal(scale, scale_expect)

        # when `fscale` is fixed, superclass fit is used to determine `loc`.
        loc, scale = stats.rayleigh.fit(data, fscale=.6)
        assert_equal(scale, .6)

        # with both parameters free, one dimensional optimization is done
        # over a new function that takes into account the dependent relation
        # of `scale` to `loc`.
        loc, scale = stats.rayleigh.fit(data)
        # test that `scale` is defined by its relation to `loc`
        assert_equal(scale, scale_mle(data, loc))

    @pytest.mark.parametrize("rvs_loc,rvs_scale", [[0.74, 0.01],
                                                   [0.08464463, 0.12069025]])
    def test_fit_comparison_super_method(self, rvs_loc, rvs_scale):
        # test that the objective function result of the analytical MLEs is
        # less than or equal to that of the numerically optimized estimate
        data = stats.rayleigh.rvs(size=250, loc=rvs_loc, scale=rvs_scale)
        _assert_less_or_close_loglike(stats.rayleigh, data)

    def test_fit_warnings(self):
        assert_fit_warnings(stats.rayleigh)

    def test_fit_gh17088(self):
        # `rayleigh.fit` could return a location that was inconsistent with
        # the data. See gh-17088.
        rng = np.random.default_rng(456)
        loc, scale, size = 50, 600, 500
        rvs = stats.rayleigh.rvs(loc, scale, size=size, random_state=rng)
        loc_fit, _ = stats.rayleigh.fit(rvs)
        assert loc_fit < np.min(rvs)
        loc_fit, scale_fit = stats.rayleigh.fit(rvs, fscale=scale)
        assert loc_fit < np.min(rvs)
        assert scale_fit == scale


class TestExponWeib:

    def test_pdf_logpdf(self):
        # Regression test for gh-3508.
        x = 0.1
        a = 1.0
        c = 100.0
        p = stats.exponweib.pdf(x, a, c)
        logp = stats.exponweib.logpdf(x, a, c)
        # Expected values were computed with mpmath.
        assert_allclose([p, logp],
                        [1.0000000000000054e-97, -223.35075402042244])

    def test_a_is_1(self):
        # For issue gh-3508.
        # Check that when a=1, the pdf and logpdf methods of exponweib are the
        # same as those of weibull_min.
        x = np.logspace(-4, -1, 4)
        a = 1
        c = 100

        p = stats.exponweib.pdf(x, a, c)
        expected = stats.weibull_min.pdf(x, c)
        assert_allclose(p, expected)

        logp = stats.exponweib.logpdf(x, a, c)
        expected = stats.weibull_min.logpdf(x, c)
        assert_allclose(logp, expected)

    def test_a_is_1_c_is_1(self):
        # When a = 1 and c = 1, the distribution is exponential.
        x = np.logspace(-8, 1, 10)
        a = 1
        c = 1

        p = stats.exponweib.pdf(x, a, c)
        expected = stats.expon.pdf(x)
        assert_allclose(p, expected)

        logp = stats.exponweib.logpdf(x, a, c)
        expected = stats.expon.logpdf(x)
        assert_allclose(logp, expected)

    # Reference values were computed with mpmath, e.g:
    #
    #     from mpmath import mp
    #
    #     def mp_sf(x, a, c):
    #         x = mp.mpf(x)
    #         a = mp.mpf(a)
    #         c = mp.mpf(c)
    #         return -mp.powm1(-mp.expm1(-x**c)), a)
    #
    #     mp.dps = 100
    #     print(float(mp_sf(1, 2.5, 0.75)))
    #
    # prints
    #
    #     0.6823127476985246
    #
    @pytest.mark.parametrize(
        'x, a, c, ref',
        [(1, 2.5, 0.75, 0.6823127476985246),
         (50, 2.5, 0.75, 1.7056666054719663e-08),
         (125, 2.5, 0.75, 1.4534393150714602e-16),
         (250, 2.5, 0.75, 1.2391389689773512e-27),
         (250, 0.03125, 0.75, 1.548923711221689e-29),
         (3, 0.03125, 3.0,  5.873527551689983e-14),
         (2e80, 10.0, 0.02, 2.9449084156902135e-17)]
    )
    def test_sf(self, x, a, c, ref):
        sf = stats.exponweib.sf(x, a, c)
        assert_allclose(sf, ref, rtol=1e-14)

    # Reference values were computed with mpmath, e.g.
    #
    #     from mpmath import mp
    #
    #     def mp_isf(p, a, c):
    #         p = mp.mpf(p)
    #         a = mp.mpf(a)
    #         c = mp.mpf(c)
    #         return (-mp.log(-mp.expm1(mp.log1p(-p)/a)))**(1/c)
    #
    #     mp.dps = 100
    #     print(float(mp_isf(0.25, 2.5, 0.75)))
    #
    # prints
    #
    #     2.8946008178158924
    #
    @pytest.mark.parametrize(
        'p, a, c, ref',
        [(0.25, 2.5, 0.75, 2.8946008178158924),
         (3e-16, 2.5, 0.75, 121.77966713102938),
         (1e-12, 1, 2, 5.256521769756932),
         (2e-13, 0.03125, 3, 2.953915059484589),
         (5e-14, 10.0, 0.02, 7.57094886384687e+75)]
    )
    def test_isf(self, p, a, c, ref):
        isf = stats.exponweib.isf(p, a, c)
        assert_allclose(isf, ref, rtol=5e-14)


class TestFatigueLife:

    def test_sf_tail(self):
        # Expected value computed with mpmath:
        #     import mpmath
        #     mpmath.mp.dps = 80
        #     x = mpmath.mpf(800.0)
        #     c = mpmath.mpf(2.5)
        #     s = float(1 - mpmath.ncdf(1/c * (mpmath.sqrt(x)
        #                                      - 1/mpmath.sqrt(x))))
        #     print(s)
        # Output:
        #     6.593376447038406e-30
        s = stats.fatiguelife.sf(800.0, 2.5)
        assert_allclose(s, 6.593376447038406e-30, rtol=1e-13)

    def test_isf_tail(self):
        # See test_sf_tail for the mpmath code.
        p = 6.593376447038406e-30
        q = stats.fatiguelife.isf(p, 2.5)
        assert_allclose(q, 800.0, rtol=1e-13)


class TestWeibull:

    def test_logpdf(self):
        # gh-6217
        y = stats.weibull_min.logpdf(0, 1)
        assert_equal(y, 0)

    def test_with_maxima_distrib(self):
        # Tests for weibull_min and weibull_max.
        # The expected values were computed using the symbolic algebra
        # program 'maxima' with the package 'distrib', which has
        # 'pdf_weibull' and 'cdf_weibull'.  The mapping between the
        # scipy and maxima functions is as follows:
        # -----------------------------------------------------------------
        # scipy                              maxima
        # ---------------------------------  ------------------------------
        # weibull_min.pdf(x, a, scale=b)     pdf_weibull(x, a, b)
        # weibull_min.logpdf(x, a, scale=b)  log(pdf_weibull(x, a, b))
        # weibull_min.cdf(x, a, scale=b)     cdf_weibull(x, a, b)
        # weibull_min.logcdf(x, a, scale=b)  log(cdf_weibull(x, a, b))
        # weibull_min.sf(x, a, scale=b)      1 - cdf_weibull(x, a, b)
        # weibull_min.logsf(x, a, scale=b)   log(1 - cdf_weibull(x, a, b))
        #
        # weibull_max.pdf(x, a, scale=b)     pdf_weibull(-x, a, b)
        # weibull_max.logpdf(x, a, scale=b)  log(pdf_weibull(-x, a, b))
        # weibull_max.cdf(x, a, scale=b)     1 - cdf_weibull(-x, a, b)
        # weibull_max.logcdf(x, a, scale=b)  log(1 - cdf_weibull(-x, a, b))
        # weibull_max.sf(x, a, scale=b)      cdf_weibull(-x, a, b)
        # weibull_max.logsf(x, a, scale=b)   log(cdf_weibull(-x, a, b))
        # -----------------------------------------------------------------
        x = 1.5
        a = 2.0
        b = 3.0

        # weibull_min

        p = stats.weibull_min.pdf(x, a, scale=b)
        assert_allclose(p, np.exp(-0.25)/3)

        lp = stats.weibull_min.logpdf(x, a, scale=b)
        assert_allclose(lp, -0.25 - np.log(3))

        c = stats.weibull_min.cdf(x, a, scale=b)
        assert_allclose(c, -special.expm1(-0.25))

        lc = stats.weibull_min.logcdf(x, a, scale=b)
        assert_allclose(lc, np.log(-special.expm1(-0.25)))

        s = stats.weibull_min.sf(x, a, scale=b)
        assert_allclose(s, np.exp(-0.25))

        ls = stats.weibull_min.logsf(x, a, scale=b)
        assert_allclose(ls, -0.25)

        # Also test using a large value x, for which computing the survival
        # function using the CDF would result in 0.
        s = stats.weibull_min.sf(30, 2, scale=3)
        assert_allclose(s, np.exp(-100))

        ls = stats.weibull_min.logsf(30, 2, scale=3)
        assert_allclose(ls, -100)

        # weibull_max
        x = -1.5

        p = stats.weibull_max.pdf(x, a, scale=b)
        assert_allclose(p, np.exp(-0.25)/3)

        lp = stats.weibull_max.logpdf(x, a, scale=b)
        assert_allclose(lp, -0.25 - np.log(3))

        c = stats.weibull_max.cdf(x, a, scale=b)
        assert_allclose(c, np.exp(-0.25))

        lc = stats.weibull_max.logcdf(x, a, scale=b)
        assert_allclose(lc, -0.25)

        s = stats.weibull_max.sf(x, a, scale=b)
        assert_allclose(s, -special.expm1(-0.25))

        ls = stats.weibull_max.logsf(x, a, scale=b)
        assert_allclose(ls, np.log(-special.expm1(-0.25)))

        # Also test using a value of x close to 0, for which computing the
        # survival function using the CDF would result in 0.
        s = stats.weibull_max.sf(-1e-9, 2, scale=3)
        assert_allclose(s, -special.expm1(-1/9000000000000000000))

        ls = stats.weibull_max.logsf(-1e-9, 2, scale=3)
        assert_allclose(ls, np.log(-special.expm1(-1/9000000000000000000)))

    @pytest.mark.parametrize('scale', [1.0, 0.1])
    def test_delta_cdf(self, scale):
        # Expected value computed with mpmath:
        #
        # def weibull_min_sf(x, k, scale):
        #     x = mpmath.mpf(x)
        #     k = mpmath.mpf(k)
        #     scale =mpmath.mpf(scale)
        #     return mpmath.exp(-(x/scale)**k)
        #
        # >>> import mpmath
        # >>> mpmath.mp.dps = 60
        # >>> sf1 = weibull_min_sf(7.5, 3, 1)
        # >>> sf2 = weibull_min_sf(8.0, 3, 1)
        # >>> float(sf1 - sf2)
        # 6.053624060118734e-184
        #
        delta = stats.weibull_min._delta_cdf(scale*7.5, scale*8, 3,
                                             scale=scale)
        assert_allclose(delta, 6.053624060118734e-184)

    def test_fit_min(self):
        rng = np.random.default_rng(5985959307161735394)

        c, loc, scale = 2, 3.5, 0.5  # arbitrary, valid parameters
        dist = stats.weibull_min(c, loc, scale)
        rvs = dist.rvs(size=100, random_state=rng)

        # test that MLE still honors guesses and fixed parameters
        c2, loc2, scale2 = stats.weibull_min.fit(rvs, 1.5, floc=3)
        c3, loc3, scale3 = stats.weibull_min.fit(rvs, 1.6, floc=3)
        assert loc2 == loc3 == 3  # fixed parameter is respected
        assert c2 != c3  # different guess -> (slightly) different outcome
        # quality of fit is tested elsewhere

        # test that MoM honors fixed parameters, accepts (but ignores) guesses
        c4, loc4, scale4 = stats.weibull_min.fit(rvs, 3, fscale=3, method='mm')
        assert scale4 == 3
        # because scale was fixed, only the mean and skewness will be matched
        dist4 = stats.weibull_min(c4, loc4, scale4)
        res = dist4.stats(moments='ms')
        ref = np.mean(rvs), stats.skew(rvs)
        assert_allclose(res, ref)

    # reference values were computed via mpmath
    # from mpmath import mp
    # def weibull_sf_mpmath(x, c):
    #     x = mp.mpf(x)
    #     c = mp.mpf(c)
    #     return float(mp.exp(-x**c))

    @pytest.mark.parametrize('x, c, ref', [(50, 1, 1.9287498479639178e-22),
                                           (1000, 0.8,
                                            8.131269637872743e-110)])
    def test_sf_isf(self, x, c, ref):
        assert_allclose(stats.weibull_min.sf(x, c), ref, rtol=5e-14)
        assert_allclose(stats.weibull_min.isf(ref, c), x, rtol=5e-14)


class TestDweibull:
    def test_entropy(self):
        # Test that dweibull entropy follows that of weibull_min.
        # (Generic tests check that the dweibull entropy is consistent
        #  with its PDF. As for accuracy, dweibull entropy should be just
        #  as accurate as weibull_min entropy. Checks of accuracy against
        #  a reference need only be applied to the fundamental distribution -
        #  weibull_min.)
        rng = np.random.default_rng(8486259129157041777)
        c = 10**rng.normal(scale=100, size=10)
        res = stats.dweibull.entropy(c)
        ref = stats.weibull_min.entropy(c) - np.log(0.5)
        assert_allclose(res, ref, rtol=1e-15)

    def test_sf(self):
        # test that for positive values the dweibull survival function is half
        # the weibull_min survival function
        rng = np.random.default_rng(8486259129157041777)
        c = 10**rng.normal(scale=1, size=10)
        x = 10 * rng.uniform()
        res = stats.dweibull.sf(x, c)
        ref = 0.5 * stats.weibull_min.sf(x, c)
        assert_allclose(res, ref, rtol=1e-15)


class TestTruncWeibull:

    def test_pdf_bounds(self):
        # test bounds
        y = stats.truncweibull_min.pdf([0.1, 2.0], 2.0, 0.11, 1.99)
        assert_equal(y, [0.0, 0.0])

    def test_logpdf(self):
        y = stats.truncweibull_min.logpdf(2.0, 1.0, 2.0, np.inf)
        assert_equal(y, 0.0)

        # hand calculation
        y = stats.truncweibull_min.logpdf(2.0, 1.0, 2.0, 4.0)
        assert_allclose(y, 0.14541345786885884)

    def test_ppf_bounds(self):
        # test bounds
        y = stats.truncweibull_min.ppf([0.0, 1.0], 2.0, 0.1, 2.0)
        assert_equal(y, [0.1, 2.0])

    def test_cdf_to_ppf(self):
        q = [0., 0.1, .25, 0.50, 0.75, 0.90, 1.]
        x = stats.truncweibull_min.ppf(q, 2., 0., 3.)
        q_out = stats.truncweibull_min.cdf(x, 2., 0., 3.)
        assert_allclose(q, q_out)

    def test_sf_to_isf(self):
        q = [0., 0.1, .25, 0.50, 0.75, 0.90, 1.]
        x = stats.truncweibull_min.isf(q, 2., 0., 3.)
        q_out = stats.truncweibull_min.sf(x, 2., 0., 3.)
        assert_allclose(q, q_out)

    def test_munp(self):
        c = 2.
        a = 1.
        b = 3.

        def xnpdf(x, n):
            return x**n*stats.truncweibull_min.pdf(x, c, a, b)

        m0 = stats.truncweibull_min.moment(0, c, a, b)
        assert_equal(m0, 1.)

        m1 = stats.truncweibull_min.moment(1, c, a, b)
        m1_expected, _ = quad(lambda x: xnpdf(x, 1), a, b)
        assert_allclose(m1, m1_expected)

        m2 = stats.truncweibull_min.moment(2, c, a, b)
        m2_expected, _ = quad(lambda x: xnpdf(x, 2), a, b)
        assert_allclose(m2, m2_expected)

        m3 = stats.truncweibull_min.moment(3, c, a, b)
        m3_expected, _ = quad(lambda x: xnpdf(x, 3), a, b)
        assert_allclose(m3, m3_expected)

        m4 = stats.truncweibull_min.moment(4, c, a, b)
        m4_expected, _ = quad(lambda x: xnpdf(x, 4), a, b)
        assert_allclose(m4, m4_expected)

    def test_reference_values(self):
        a = 1.
        b = 3.
        c = 2.
        x_med = np.sqrt(1 - np.log(0.5 + np.exp(-(8. + np.log(2.)))))

        cdf = stats.truncweibull_min.cdf(x_med, c, a, b)
        assert_allclose(cdf, 0.5)

        lc = stats.truncweibull_min.logcdf(x_med, c, a, b)
        assert_allclose(lc, -np.log(2.))

        ppf = stats.truncweibull_min.ppf(0.5, c, a, b)
        assert_allclose(ppf, x_med)

        sf = stats.truncweibull_min.sf(x_med, c, a, b)
        assert_allclose(sf, 0.5)

        ls = stats.truncweibull_min.logsf(x_med, c, a, b)
        assert_allclose(ls, -np.log(2.))

        isf = stats.truncweibull_min.isf(0.5, c, a, b)
        assert_allclose(isf, x_med)

    def test_compare_weibull_min(self):
        # Verify that the truncweibull_min distribution gives the same results
        # as the original weibull_min
        x = 1.5
        c = 2.0
        a = 0.0
        b = np.inf
        scale = 3.0

        p = stats.weibull_min.pdf(x, c, scale=scale)
        p_trunc = stats.truncweibull_min.pdf(x, c, a, b, scale=scale)
        assert_allclose(p, p_trunc)

        lp = stats.weibull_min.logpdf(x, c, scale=scale)
        lp_trunc = stats.truncweibull_min.logpdf(x, c, a, b, scale=scale)
        assert_allclose(lp, lp_trunc)

        cdf = stats.weibull_min.cdf(x, c, scale=scale)
        cdf_trunc = stats.truncweibull_min.cdf(x, c, a, b, scale=scale)
        assert_allclose(cdf, cdf_trunc)

        lc = stats.weibull_min.logcdf(x, c, scale=scale)
        lc_trunc = stats.truncweibull_min.logcdf(x, c, a, b, scale=scale)
        assert_allclose(lc, lc_trunc)

        s = stats.weibull_min.sf(x, c, scale=scale)
        s_trunc = stats.truncweibull_min.sf(x, c, a, b, scale=scale)
        assert_allclose(s, s_trunc)

        ls = stats.weibull_min.logsf(x, c, scale=scale)
        ls_trunc = stats.truncweibull_min.logsf(x, c, a, b, scale=scale)
        assert_allclose(ls, ls_trunc)

        # # Also test using a large value x, for which computing the survival
        # # function using the CDF would result in 0.
        s = stats.truncweibull_min.sf(30, 2, a, b, scale=3)
        assert_allclose(s, np.exp(-100))

        ls = stats.truncweibull_min.logsf(30, 2, a, b, scale=3)
        assert_allclose(ls, -100)

    def test_compare_weibull_min2(self):
        # Verify that the truncweibull_min distribution PDF and CDF results
        # are the same as those calculated from truncating weibull_min
        c, a, b = 2.5, 0.25, 1.25
        x = np.linspace(a, b, 100)

        pdf1 = stats.truncweibull_min.pdf(x, c, a, b)
        cdf1 = stats.truncweibull_min.cdf(x, c, a, b)

        norm = stats.weibull_min.cdf(b, c) - stats.weibull_min.cdf(a, c)
        pdf2 = stats.weibull_min.pdf(x, c) / norm
        cdf2 = (stats.weibull_min.cdf(x, c) - stats.weibull_min.cdf(a, c))/norm

        np.testing.assert_allclose(pdf1, pdf2)
        np.testing.assert_allclose(cdf1, cdf2)


class TestRdist:
    def test_rdist_cdf_gh1285(self):
        # check workaround in rdist._cdf for issue gh-1285.
        distfn = stats.rdist
        values = [0.001, 0.5, 0.999]
        assert_almost_equal(distfn.cdf(distfn.ppf(values, 541.0), 541.0),
                            values, decimal=5)

    def test_rdist_beta(self):
        # rdist is a special case of stats.beta
        x = np.linspace(-0.99, 0.99, 10)
        c = 2.7
        assert_almost_equal(0.5*stats.beta(c/2, c/2).pdf((x + 1)/2),
                            stats.rdist(c).pdf(x))


class TestTrapezoid:
    def test_reduces_to_triang(self):
        modes = [0, 0.3, 0.5, 1]
        for mode in modes:
            x = [0, mode, 1]
            assert_almost_equal(stats.trapezoid.pdf(x, mode, mode),
                                stats.triang.pdf(x, mode))
            assert_almost_equal(stats.trapezoid.cdf(x, mode, mode),
                                stats.triang.cdf(x, mode))

    def test_reduces_to_uniform(self):
        x = np.linspace(0, 1, 10)
        assert_almost_equal(stats.trapezoid.pdf(x, 0, 1), stats.uniform.pdf(x))
        assert_almost_equal(stats.trapezoid.cdf(x, 0, 1), stats.uniform.cdf(x))

    def test_cases(self):
        # edge cases
        assert_almost_equal(stats.trapezoid.pdf(0, 0, 0), 2)
        assert_almost_equal(stats.trapezoid.pdf(1, 1, 1), 2)
        assert_almost_equal(stats.trapezoid.pdf(0.5, 0, 0.8),
                            1.11111111111111111)
        assert_almost_equal(stats.trapezoid.pdf(0.5, 0.2, 1.0),
                            1.11111111111111111)

        # straightforward case
        assert_almost_equal(stats.trapezoid.pdf(0.1, 0.2, 0.8), 0.625)
        assert_almost_equal(stats.trapezoid.pdf(0.5, 0.2, 0.8), 1.25)
        assert_almost_equal(stats.trapezoid.pdf(0.9, 0.2, 0.8), 0.625)

        assert_almost_equal(stats.trapezoid.cdf(0.1, 0.2, 0.8), 0.03125)
        assert_almost_equal(stats.trapezoid.cdf(0.2, 0.2, 0.8), 0.125)
        assert_almost_equal(stats.trapezoid.cdf(0.5, 0.2, 0.8), 0.5)
        assert_almost_equal(stats.trapezoid.cdf(0.9, 0.2, 0.8), 0.96875)
        assert_almost_equal(stats.trapezoid.cdf(1.0, 0.2, 0.8), 1.0)

    def test_moments_and_entropy(self):
        # issue #11795: improve precision of trapezoid stats
        # Apply formulas from Wikipedia for the following parameters:
        a, b, c, d = -3, -1, 2, 3  # => 1/3, 5/6, -3, 6
        p1, p2, loc, scale = (b-a) / (d-a), (c-a) / (d-a), a, d-a
        h = 2 / (d+c-b-a)

        def moment(n):
            return (h * ((d**(n+2) - c**(n+2)) / (d-c)
                         - (b**(n+2) - a**(n+2)) / (b-a)) /
                    (n+1) / (n+2))

        mean = moment(1)
        var = moment(2) - mean**2
        entropy = 0.5 * (d-c+b-a) / (d+c-b-a) + np.log(0.5 * (d+c-b-a))
        assert_almost_equal(stats.trapezoid.mean(p1, p2, loc, scale),
                            mean, decimal=13)
        assert_almost_equal(stats.trapezoid.var(p1, p2, loc, scale),
                            var, decimal=13)
        assert_almost_equal(stats.trapezoid.entropy(p1, p2, loc, scale),
                            entropy, decimal=13)

        # Check boundary cases where scipy d=0 or d=1.
        assert_almost_equal(stats.trapezoid.mean(0, 0, -3, 6), -1, decimal=13)
        assert_almost_equal(stats.trapezoid.mean(0, 1, -3, 6), 0, decimal=13)
        assert_almost_equal(stats.trapezoid.var(0, 1, -3, 6), 3, decimal=13)

    def test_trapezoid_vect(self):
        # test that array-valued shapes and arguments are handled
        c = np.array([0.1, 0.2, 0.3])
        d = np.array([0.5, 0.6])[:, None]
        x = np.array([0.15, 0.25, 0.9])
        v = stats.trapezoid.pdf(x, c, d)

        cc, dd, xx = np.broadcast_arrays(c, d, x)

        res = np.empty(xx.size, dtype=xx.dtype)
        ind = np.arange(xx.size)
        for i, x1, c1, d1 in zip(ind, xx.ravel(), cc.ravel(), dd.ravel()):
            res[i] = stats.trapezoid.pdf(x1, c1, d1)

        assert_allclose(v, res.reshape(v.shape), atol=1e-15)

        # Check that the stats() method supports vector arguments.
        v = np.asarray(stats.trapezoid.stats(c, d, moments="mvsk"))
        cc, dd = np.broadcast_arrays(c, d)
        res = np.empty((cc.size, 4))  # 4 stats returned per value
        ind = np.arange(cc.size)
        for i, c1, d1 in zip(ind, cc.ravel(), dd.ravel()):
            res[i] = stats.trapezoid.stats(c1, d1, moments="mvsk")

        assert_allclose(v, res.T.reshape(v.shape), atol=1e-15)

    def test_trapz(self):
        # Basic test for alias
        x = np.linspace(0, 1, 10)
        assert_almost_equal(stats.trapz.pdf(x, 0, 1), stats.uniform.pdf(x))


class TestTriang:
    def test_edge_cases(self):
        with np.errstate(all='raise'):
            assert_equal(stats.triang.pdf(0, 0), 2.)
            assert_equal(stats.triang.pdf(0.5, 0), 1.)
            assert_equal(stats.triang.pdf(1, 0), 0.)

            assert_equal(stats.triang.pdf(0, 1), 0)
            assert_equal(stats.triang.pdf(0.5, 1), 1.)
            assert_equal(stats.triang.pdf(1, 1), 2)

            assert_equal(stats.triang.cdf(0., 0.), 0.)
            assert_equal(stats.triang.cdf(0.5, 0.), 0.75)
            assert_equal(stats.triang.cdf(1.0, 0.), 1.0)

            assert_equal(stats.triang.cdf(0., 1.), 0.)
            assert_equal(stats.triang.cdf(0.5, 1.), 0.25)
            assert_equal(stats.triang.cdf(1., 1.), 1)


class TestMaxwell:

    # reference values were computed with wolfram alpha
    # erfc(x/sqrt(2)) + sqrt(2/pi) * x * e^(-x^2/2)

    @pytest.mark.parametrize("x, ref",
                             [(20, 2.2138865931011177e-86),
                              (0.01, 0.999999734046458435)])
    def test_sf(self, x, ref):
        assert_allclose(stats.maxwell.sf(x), ref, rtol=1e-14)

    # reference values were computed with wolfram alpha
    # sqrt(2) * sqrt(Q^(-1)(3/2, q))

    @pytest.mark.parametrize("q, ref",
                             [(0.001, 4.033142223656157022),
                              (0.9999847412109375, 0.0385743284050381),
                              (2**-55, 8.95564974719481)])
    def test_isf(self, q, ref):
        assert_allclose(stats.maxwell.isf(q), ref, rtol=1e-15)


class TestMielke:
    def test_moments(self):
        k, s = 4.642, 0.597
        # n-th moment exists only if n < s
        assert_equal(stats.mielke(k, s).moment(1), np.inf)
        assert_equal(stats.mielke(k, 1.0).moment(1), np.inf)
        assert_(np.isfinite(stats.mielke(k, 1.01).moment(1)))

    def test_burr_equivalence(self):
        x = np.linspace(0.01, 100, 50)
        k, s = 2.45, 5.32
        assert_allclose(stats.burr.pdf(x, s, k/s), stats.mielke.pdf(x, k, s))


class TestBurr:
    def test_endpoints_7491(self):
        # gh-7491
        # Compute the pdf at the left endpoint dst.a.
        data = [
            [stats.fisk, (1,), 1],
            [stats.burr, (0.5, 2), 1],
            [stats.burr, (1, 1), 1],
            [stats.burr, (2, 0.5), 1],
            [stats.burr12, (1, 0.5), 0.5],
            [stats.burr12, (1, 1), 1.0],
            [stats.burr12, (1, 2), 2.0]]

        ans = [_f.pdf(_f.a, *_args) for _f, _args, _ in data]
        correct = [_correct_ for _f, _args, _correct_ in data]
        assert_array_almost_equal(ans, correct)

        ans = [_f.logpdf(_f.a, *_args) for _f, _args, _ in data]
        correct = [np.log(_correct_) for _f, _args, _correct_ in data]
        assert_array_almost_equal(ans, correct)

    def test_burr_stats_9544(self):
        # gh-9544.  Test from gh-9978
        c, d = 5.0, 3
        mean, variance = stats.burr(c, d).stats()
        # mean = sc.beta(3 + 1/5, 1. - 1/5) * 3  = 1.4110263...
        # var =  sc.beta(3 + 2 / 5, 1. - 2 / 5) * 3 -
        #        (sc.beta(3 + 1 / 5, 1. - 1 / 5) * 3) ** 2
        mean_hc, variance_hc = 1.4110263183925857, 0.22879948026191643
        assert_allclose(mean, mean_hc)
        assert_allclose(variance, variance_hc)

    def test_burr_nan_mean_var_9544(self):
        # gh-9544.  Test from gh-9978
        c, d = 0.5, 3
        mean, variance = stats.burr(c, d).stats()
        assert_(np.isnan(mean))
        assert_(np.isnan(variance))
        c, d = 1.5, 3
        mean, variance = stats.burr(c, d).stats()
        assert_(np.isfinite(mean))
        assert_(np.isnan(variance))

        c, d = 0.5, 3
        e1, e2, e3, e4 = stats.burr._munp(np.array([1, 2, 3, 4]), c, d)
        assert_(np.isnan(e1))
        assert_(np.isnan(e2))
        assert_(np.isnan(e3))
        assert_(np.isnan(e4))
        c, d = 1.5, 3
        e1, e2, e3, e4 = stats.burr._munp([1, 2, 3, 4], c, d)
        assert_(np.isfinite(e1))
        assert_(np.isnan(e2))
        assert_(np.isnan(e3))
        assert_(np.isnan(e4))
        c, d = 2.5, 3
        e1, e2, e3, e4 = stats.burr._munp([1, 2, 3, 4], c, d)
        assert_(np.isfinite(e1))
        assert_(np.isfinite(e2))
        assert_(np.isnan(e3))
        assert_(np.isnan(e4))
        c, d = 3.5, 3
        e1, e2, e3, e4 = stats.burr._munp([1, 2, 3, 4], c, d)
        assert_(np.isfinite(e1))
        assert_(np.isfinite(e2))
        assert_(np.isfinite(e3))
        assert_(np.isnan(e4))
        c, d = 4.5, 3
        e1, e2, e3, e4 = stats.burr._munp([1, 2, 3, 4], c, d)
        assert_(np.isfinite(e1))
        assert_(np.isfinite(e2))
        assert_(np.isfinite(e3))
        assert_(np.isfinite(e4))


class TestBurr12:

    @pytest.mark.parametrize('scale, expected',
                             [(1.0, 2.3283064359965952e-170),
                              (3.5, 5.987114417447875e-153)])
    def test_delta_cdf(self, scale, expected):
        # Expected value computed with mpmath:
        #
        # def burr12sf(x, c, d, scale):
        #     x = mpmath.mpf(x)
        #     c = mpmath.mpf(c)
        #     d = mpmath.mpf(d)
        #     scale = mpmath.mpf(scale)
        #     return (mpmath.mp.one + (x/scale)**c)**(-d)
        #
        # >>> import mpmath
        # >>> mpmath.mp.dps = 60
        # >>> float(burr12sf(2e5, 4, 8, 1) - burr12sf(4e5, 4, 8, 1))
        # 2.3283064359965952e-170
        # >>> float(burr12sf(2e5, 4, 8, 3.5) - burr12sf(4e5, 4, 8, 3.5))
        # 5.987114417447875e-153
        #
        delta = stats.burr12._delta_cdf(2e5, 4e5, 4, 8, scale=scale)
        assert_allclose(delta, expected, rtol=1e-13)


class TestStudentizedRange:
    # For alpha = .05, .01, and .001, and for each value of
    # v = [1, 3, 10, 20, 120, inf], a Q was picked from each table for
    # k = [2, 8, 14, 20].

    # these arrays are written with `k` as column, and `v` as rows.
    # Q values are taken from table 3:
    # https://www.jstor.org/stable/2237810
    q05 = [17.97, 45.40, 54.33, 59.56,
           4.501, 8.853, 10.35, 11.24,
           3.151, 5.305, 6.028, 6.467,
           2.950, 4.768, 5.357, 5.714,
           2.800, 4.363, 4.842, 5.126,
           2.772, 4.286, 4.743, 5.012]
    q01 = [90.03, 227.2, 271.8, 298.0,
           8.261, 15.64, 18.22, 19.77,
           4.482, 6.875, 7.712, 8.226,
           4.024, 5.839, 6.450, 6.823,
           3.702, 5.118, 5.562, 5.827,
           3.643, 4.987, 5.400, 5.645]
    q001 = [900.3, 2272, 2718, 2980,
            18.28, 34.12, 39.69, 43.05,
            6.487, 9.352, 10.39, 11.03,
            5.444, 7.313, 7.966, 8.370,
            4.772, 6.039, 6.448, 6.695,
            4.654, 5.823, 6.191, 6.411]
    qs = np.concatenate((q05, q01, q001))
    ps = [.95, .99, .999]
    vs = [1, 3, 10, 20, 120, np.inf]
    ks = [2, 8, 14, 20]

    data = list(zip(product(ps, vs, ks), qs))

    # A small selection of large-v cases generated with R's `ptukey`
    # Each case is in the format (q, k, v, r_result)
    r_data = [
        (0.1, 3, 9001, 0.002752818526842),
        (1, 10, 1000, 0.000526142388912),
        (1, 3, np.inf, 0.240712641229283),
        (4, 3, np.inf, 0.987012338626815),
        (1, 10, np.inf, 0.000519869467083),
    ]

    def test_cdf_against_tables(self):
        for pvk, q in self.data:
            p_expected, v, k = pvk
            res_p = stats.studentized_range.cdf(q, k, v)
            assert_allclose(res_p, p_expected, rtol=1e-4)

    @pytest.mark.slow
    def test_ppf_against_tables(self):
        for pvk, q_expected in self.data:
            p, v, k = pvk
            res_q = stats.studentized_range.ppf(p, k, v)
            assert_allclose(res_q, q_expected, rtol=5e-4)

    path_prefix = os.path.dirname(__file__)
    relative_path = "data/studentized_range_mpmath_ref.json"
    with open(os.path.join(path_prefix, relative_path)) as file:
        pregenerated_data = json.load(file)

    @pytest.mark.parametrize("case_result", pregenerated_data["cdf_data"])
    def test_cdf_against_mp(self, case_result):
        src_case = case_result["src_case"]
        mp_result = case_result["mp_result"]
        qkv = src_case["q"], src_case["k"], src_case["v"]
        res = stats.studentized_range.cdf(*qkv)

        assert_allclose(res, mp_result,
                        atol=src_case["expected_atol"],
                        rtol=src_case["expected_rtol"])

    @pytest.mark.parametrize("case_result", pregenerated_data["pdf_data"])
    def test_pdf_against_mp(self, case_result):
        src_case = case_result["src_case"]
        mp_result = case_result["mp_result"]
        qkv = src_case["q"], src_case["k"], src_case["v"]
        res = stats.studentized_range.pdf(*qkv)

        assert_allclose(res, mp_result,
                        atol=src_case["expected_atol"],
                        rtol=src_case["expected_rtol"])

    @pytest.mark.slow
    @pytest.mark.xfail_on_32bit("intermittent RuntimeWarning: invalid value.")
    @pytest.mark.parametrize("case_result", pregenerated_data["moment_data"])
    def test_moment_against_mp(self, case_result):
        src_case = case_result["src_case"]
        mp_result = case_result["mp_result"]
        mkv = src_case["m"], src_case["k"], src_case["v"]

        # Silence invalid value encountered warnings. Actual problems will be
        # caught by the result comparison.
        with np.errstate(invalid='ignore'):
            res = stats.studentized_range.moment(*mkv)

        assert_allclose(res, mp_result,
                        atol=src_case["expected_atol"],
                        rtol=src_case["expected_rtol"])

    def test_pdf_integration(self):
        k, v = 3, 10
        # Test whether PDF integration is 1 like it should be.
        res = quad(stats.studentized_range.pdf, 0, np.inf, args=(k, v))
        assert_allclose(res[0], 1)

    @pytest.mark.xslow
    def test_pdf_against_cdf(self):
        k, v = 3, 10

        # Test whether the integrated PDF matches the CDF using cumulative
        # integration. Use a small step size to reduce error due to the
        # summation. This is slow, but tests the results well.
        x = np.arange(0, 10, step=0.01)

        y_cdf = stats.studentized_range.cdf(x, k, v)[1:]
        y_pdf_raw = stats.studentized_range.pdf(x, k, v)
        y_pdf_cumulative = cumulative_trapezoid(y_pdf_raw, x)

        # Because of error caused by the summation, use a relatively large rtol
        assert_allclose(y_pdf_cumulative, y_cdf, rtol=1e-4)

    @pytest.mark.parametrize("r_case_result", r_data)
    def test_cdf_against_r(self, r_case_result):
        # Test large `v` values using R
        q, k, v, r_res = r_case_result
        with np.errstate(invalid='ignore'):
            res = stats.studentized_range.cdf(q, k, v)
        assert_allclose(res, r_res)

    @pytest.mark.slow
    @pytest.mark.xfail_on_32bit("intermittent RuntimeWarning: invalid value.")
    def test_moment_vectorization(self):
        # Test moment broadcasting. Calls `_munp` directly because
        # `rv_continuous.moment` is broken at time of writing. See gh-12192

        # Silence invalid value encountered warnings. Actual problems will be
        # caught by the result comparison.
        with np.errstate(invalid='ignore'):
            m = stats.studentized_range._munp([1, 2], [4, 5], [10, 11])

        assert_allclose(m.shape, (2,))

        with pytest.raises(ValueError, match="...could not be broadcast..."):
            stats.studentized_range._munp(1, [4, 5], [10, 11, 12])

    @pytest.mark.xslow
    def test_fitstart_valid(self):
        with suppress_warnings() as sup, np.errstate(invalid="ignore"):
            # the integration warning message may differ
            sup.filter(IntegrationWarning)
            k, df, _, _ = stats.studentized_range._fitstart([1, 2, 3])
        assert_(stats.studentized_range._argcheck(k, df))

    def test_infinite_df(self):
        # Check that the CDF and PDF infinite and normal integrators
        # roughly match for a high df case
        res = stats.studentized_range.pdf(3, 10, np.inf)
        res_finite = stats.studentized_range.pdf(3, 10, 99999)
        assert_allclose(res, res_finite, atol=1e-4, rtol=1e-4)

        res = stats.studentized_range.cdf(3, 10, np.inf)
        res_finite = stats.studentized_range.cdf(3, 10, 99999)
        assert_allclose(res, res_finite, atol=1e-4, rtol=1e-4)

    def test_df_cutoff(self):
        # Test that the CDF and PDF properly switch integrators at df=100,000.
        # The infinite integrator should be different enough that it fails
        # an allclose assertion. Also sanity check that using the same
        # integrator does pass the allclose with a 1-df difference, which
        # should be tiny.

        res = stats.studentized_range.pdf(3, 10, 100000)
        res_finite = stats.studentized_range.pdf(3, 10, 99999)
        res_sanity = stats.studentized_range.pdf(3, 10, 99998)
        assert_raises(AssertionError, assert_allclose, res, res_finite,
                      atol=1e-6, rtol=1e-6)
        assert_allclose(res_finite, res_sanity, atol=1e-6, rtol=1e-6)

        res = stats.studentized_range.cdf(3, 10, 100000)
        res_finite = stats.studentized_range.cdf(3, 10, 99999)
        res_sanity = stats.studentized_range.cdf(3, 10, 99998)
        assert_raises(AssertionError, assert_allclose, res, res_finite,
                      atol=1e-6, rtol=1e-6)
        assert_allclose(res_finite, res_sanity, atol=1e-6, rtol=1e-6)

    def test_clipping(self):
        # The result of this computation was -9.9253938401489e-14 on some
        # systems. The correct result is very nearly zero, but should not be
        # negative.
        q, k, v = 34.6413996195345746, 3, 339
        p = stats.studentized_range.sf(q, k, v)
        assert_allclose(p, 0, atol=1e-10)
        assert p >= 0


def test_540_567():
    # test for nan returned in tickets 540, 567
    assert_almost_equal(stats.norm.cdf(-1.7624320982), 0.03899815971089126,
                        decimal=10, err_msg='test_540_567')
    assert_almost_equal(stats.norm.cdf(-1.7624320983), 0.038998159702449846,
                        decimal=10, err_msg='test_540_567')
    assert_almost_equal(stats.norm.cdf(1.38629436112, loc=0.950273420309,
                                       scale=0.204423758009),
                        0.98353464004309321,
                        decimal=10, err_msg='test_540_567')


def test_regression_ticket_1326():
    # adjust to avoid nan with 0*log(0)
    assert_almost_equal(stats.chi2.pdf(0.0, 2), 0.5, 14)


def test_regression_tukey_lambda():
    # Make sure that Tukey-Lambda distribution correctly handles
    # non-positive lambdas.
    x = np.linspace(-5.0, 5.0, 101)

    with np.errstate(divide='ignore'):
        for lam in [0.0, -1.0, -2.0, np.array([[-1.0], [0.0], [-2.0]])]:
            p = stats.tukeylambda.pdf(x, lam)
            assert_((p != 0.0).all())
            assert_(~np.isnan(p).all())

        lam = np.array([[-1.0], [0.0], [2.0]])
        p = stats.tukeylambda.pdf(x, lam)

    assert_(~np.isnan(p).all())
    assert_((p[0] != 0.0).all())
    assert_((p[1] != 0.0).all())
    assert_((p[2] != 0.0).any())
    assert_((p[2] == 0.0).any())


@pytest.mark.skipif(DOCSTRINGS_STRIPPED, reason="docstrings stripped")
def test_regression_ticket_1421():
    assert_('pdf(x, mu, loc=0, scale=1)' not in stats.poisson.__doc__)
    assert_('pmf(x,' in stats.poisson.__doc__)


def test_nan_arguments_gh_issue_1362():
    with np.errstate(invalid='ignore'):
        assert_(np.isnan(stats.t.logcdf(1, np.nan)))
        assert_(np.isnan(stats.t.cdf(1, np.nan)))
        assert_(np.isnan(stats.t.logsf(1, np.nan)))
        assert_(np.isnan(stats.t.sf(1, np.nan)))
        assert_(np.isnan(stats.t.pdf(1, np.nan)))
        assert_(np.isnan(stats.t.logpdf(1, np.nan)))
        assert_(np.isnan(stats.t.ppf(1, np.nan)))
        assert_(np.isnan(stats.t.isf(1, np.nan)))

        assert_(np.isnan(stats.bernoulli.logcdf(np.nan, 0.5)))
        assert_(np.isnan(stats.bernoulli.cdf(np.nan, 0.5)))
        assert_(np.isnan(stats.bernoulli.logsf(np.nan, 0.5)))
        assert_(np.isnan(stats.bernoulli.sf(np.nan, 0.5)))
        assert_(np.isnan(stats.bernoulli.pmf(np.nan, 0.5)))
        assert_(np.isnan(stats.bernoulli.logpmf(np.nan, 0.5)))
        assert_(np.isnan(stats.bernoulli.ppf(np.nan, 0.5)))
        assert_(np.isnan(stats.bernoulli.isf(np.nan, 0.5)))


def test_frozen_fit_ticket_1536():
    np.random.seed(5678)
    true = np.array([0.25, 0., 0.5])
    x = stats.lognorm.rvs(true[0], true[1], true[2], size=100)

    with np.errstate(divide='ignore'):
        params = np.array(stats.lognorm.fit(x, floc=0.))

    assert_almost_equal(params, true, decimal=2)

    params = np.array(stats.lognorm.fit(x, fscale=0.5, loc=0))
    assert_almost_equal(params, true, decimal=2)

    params = np.array(stats.lognorm.fit(x, f0=0.25, loc=0))
    assert_almost_equal(params, true, decimal=2)

    params = np.array(stats.lognorm.fit(x, f0=0.25, floc=0))
    assert_almost_equal(params, true, decimal=2)

    np.random.seed(5678)
    loc = 1
    floc = 0.9
    x = stats.norm.rvs(loc, 2., size=100)
    params = np.array(stats.norm.fit(x, floc=floc))
    expected = np.array([floc, np.sqrt(((x-floc)**2).mean())])
    assert_almost_equal(params, expected, decimal=4)


def test_regression_ticket_1530():
    # Check the starting value works for Cauchy distribution fit.
    np.random.seed(654321)
    rvs = stats.cauchy.rvs(size=100)
    params = stats.cauchy.fit(rvs)
    expected = (0.045, 1.142)
    assert_almost_equal(params, expected, decimal=1)


def test_gh_pr_4806():
    # Check starting values for Cauchy distribution fit.
    np.random.seed(1234)
    x = np.random.randn(42)
    for offset in 10000.0, 1222333444.0:
        loc, scale = stats.cauchy.fit(x + offset)
        assert_allclose(loc, offset, atol=1.0)
        assert_allclose(scale, 0.6, atol=1.0)


def test_tukeylambda_stats_ticket_1545():
    # Some test for the variance and kurtosis of the Tukey Lambda distr.
    # See test_tukeylamdba_stats.py for more tests.

    mv = stats.tukeylambda.stats(0, moments='mvsk')
    # Known exact values:
    expected = [0, np.pi**2/3, 0, 1.2]
    assert_almost_equal(mv, expected, decimal=10)

    mv = stats.tukeylambda.stats(3.13, moments='mvsk')
    # 'expected' computed with mpmath.
    expected = [0, 0.0269220858861465102, 0, -0.898062386219224104]
    assert_almost_equal(mv, expected, decimal=10)

    mv = stats.tukeylambda.stats(0.14, moments='mvsk')
    # 'expected' computed with mpmath.
    expected = [0, 2.11029702221450250, 0, -0.02708377353223019456]
    assert_almost_equal(mv, expected, decimal=10)


def test_poisson_logpmf_ticket_1436():
    assert_(np.isfinite(stats.poisson.logpmf(1500, 200)))


def test_powerlaw_stats():
    """Test the powerlaw stats function.

    This unit test is also a regression test for ticket 1548.

    The exact values are:
    mean:
        mu = a / (a + 1)
    variance:
        sigma**2 = a / ((a + 2) * (a + 1) ** 2)
    skewness:
        One formula (see https://en.wikipedia.org/wiki/Skewness) is
            gamma_1 = (E[X**3] - 3*mu*E[X**2] + 2*mu**3) / sigma**3
        A short calculation shows that E[X**k] is a / (a + k), so gamma_1
        can be implemented as
            n = a/(a+3) - 3*(a/(a+1))*a/(a+2) + 2*(a/(a+1))**3
            d = sqrt(a/((a+2)*(a+1)**2)) ** 3
            gamma_1 = n/d
        Either by simplifying, or by a direct calculation of mu_3 / sigma**3,
        one gets the more concise formula:
            gamma_1 = -2.0 * ((a - 1) / (a + 3)) * sqrt((a + 2) / a)
    kurtosis: (See https://en.wikipedia.org/wiki/Kurtosis)
        The excess kurtosis is
            gamma_2 = mu_4 / sigma**4 - 3
        A bit of calculus and algebra (sympy helps) shows that
            mu_4 = 3*a*(3*a**2 - a + 2) / ((a+1)**4 * (a+2) * (a+3) * (a+4))
        so
            gamma_2 = 3*(3*a**2 - a + 2) * (a+2) / (a*(a+3)*(a+4)) - 3
        which can be rearranged to
            gamma_2 = 6 * (a**3 - a**2 - 6*a + 2) / (a*(a+3)*(a+4))
    """
    cases = [(1.0, (0.5, 1./12, 0.0, -1.2)),
             (2.0, (2./3, 2./36, -0.56568542494924734, -0.6))]
    for a, exact_mvsk in cases:
        mvsk = stats.powerlaw.stats(a, moments="mvsk")
        assert_array_almost_equal(mvsk, exact_mvsk)


def test_powerlaw_edge():
    # Regression test for gh-3986.
    p = stats.powerlaw.logpdf(0, 1)
    assert_equal(p, 0.0)


def test_exponpow_edge():
    # Regression test for gh-3982.
    p = stats.exponpow.logpdf(0, 1)
    assert_equal(p, 0.0)

    # Check pdf and logpdf at x = 0 for other values of b.
    p = stats.exponpow.pdf(0, [0.25, 1.0, 1.5])
    assert_equal(p, [np.inf, 1.0, 0.0])
    p = stats.exponpow.logpdf(0, [0.25, 1.0, 1.5])
    assert_equal(p, [np.inf, 0.0, -np.inf])


def test_gengamma_edge():
    # Regression test for gh-3985.
    p = stats.gengamma.pdf(0, 1, 1)
    assert_equal(p, 1.0)


@pytest.mark.parametrize("a, c, ref, tol",
                         [(1500000.0, 1, 8.529426144018633, 1e-15),
                          (1e+30, 1, 35.95771492811536, 1e-15),
                          (1e+100, 1, 116.54819318290696, 1e-15),
                          (3e3, 1, 5.422011196659015, 1e-13),
                          (3e6, -1e100, -236.29663213396054, 1e-15),
                          (3e60, 1e-100, 1.3925371786831085e+102, 1e-15)])
def test_gengamma_extreme_entropy(a, c, ref, tol):
    # The reference values were calculated with mpmath:
    # from mpmath import mp
    # mp.dps = 500
    #
    # def gen_entropy(a, c):
    #     a, c = mp.mpf(a), mp.mpf(c)
    #     val = mp.digamma(a)
    #     h = (a * (mp.one - val) + val/c + mp.loggamma(a) - mp.log(abs(c)))
    #     return float(h)
    assert_allclose(stats.gengamma.entropy(a, c), ref, rtol=tol)


def test_gengamma_endpoint_with_neg_c():
    p = stats.gengamma.pdf(0, 1, -1)
    assert p == 0.0
    logp = stats.gengamma.logpdf(0, 1, -1)
    assert logp == -np.inf


def test_gengamma_munp():
    # Regression tests for gh-4724.
    p = stats.gengamma._munp(-2, 200, 1.)
    assert_almost_equal(p, 1./199/198)

    p = stats.gengamma._munp(-2, 10, 1.)
    assert_almost_equal(p, 1./9/8)


def test_ksone_fit_freeze():
    # Regression test for ticket #1638.
    d = np.array(
        [-0.18879233, 0.15734249, 0.18695107, 0.27908787, -0.248649,
         -0.2171497, 0.12233512, 0.15126419, 0.03119282, 0.4365294,
         0.08930393, -0.23509903, 0.28231224, -0.09974875, -0.25196048,
         0.11102028, 0.1427649, 0.10176452, 0.18754054, 0.25826724,
         0.05988819, 0.0531668, 0.21906056, 0.32106729, 0.2117662,
         0.10886442, 0.09375789, 0.24583286, -0.22968366, -0.07842391,
         -0.31195432, -0.21271196, 0.1114243, -0.13293002, 0.01331725,
         -0.04330977, -0.09485776, -0.28434547, 0.22245721, -0.18518199,
         -0.10943985, -0.35243174, 0.06897665, -0.03553363, -0.0701746,
         -0.06037974, 0.37670779, -0.21684405])

    with np.errstate(invalid='ignore'):
        with suppress_warnings() as sup:
            sup.filter(IntegrationWarning,
                       "The maximum number of subdivisions .50. has been "
                       "achieved.")
            sup.filter(RuntimeWarning,
                       "floating point number truncated to an integer")
            stats.ksone.fit(d)


def test_norm_logcdf():
    # Test precision of the logcdf of the normal distribution.
    # This precision was enhanced in ticket 1614.
    x = -np.asarray(list(range(0, 120, 4)))
    # Values from R
    expected = [-0.69314718, -10.36010149, -35.01343716, -75.41067300,
                -131.69539607, -203.91715537, -292.09872100, -396.25241451,
                -516.38564863, -652.50322759, -804.60844201, -972.70364403,
                -1156.79057310, -1356.87055173, -1572.94460885, -1805.01356068,
                -2053.07806561, -2317.13866238, -2597.19579746, -2893.24984493,
                -3205.30112136, -3533.34989701, -3877.39640444, -4237.44084522,
                -4613.48339520, -5005.52420869, -5413.56342187, -5837.60115548,
                -6277.63751711, -6733.67260303]

    assert_allclose(stats.norm().logcdf(x), expected, atol=1e-8)

    # also test the complex-valued code path
    assert_allclose(stats.norm().logcdf(x + 1e-14j).real, expected, atol=1e-8)

    # test the accuracy: d(logcdf)/dx = pdf / cdf \equiv exp(logpdf - logcdf)
    deriv = (stats.norm.logcdf(x + 1e-10j)/1e-10).imag
    deriv_expected = np.exp(stats.norm.logpdf(x) - stats.norm.logcdf(x))
    assert_allclose(deriv, deriv_expected, atol=1e-10)


def test_levy_cdf_ppf():
    # Test levy.cdf, including small arguments.
    x = np.array([1000, 1.0, 0.5, 0.1, 0.01, 0.001])

    # Expected values were calculated separately with mpmath.
    # E.g.
    # >>> mpmath.mp.dps = 100
    # >>> x = mpmath.mp.mpf('0.01')
    # >>> cdf = mpmath.erfc(mpmath.sqrt(1/(2*x)))
    expected = np.array([0.9747728793699604,
                         0.3173105078629141,
                         0.1572992070502851,
                         0.0015654022580025495,
                         1.523970604832105e-23,
                         1.795832784800726e-219])

    y = stats.levy.cdf(x)
    assert_allclose(y, expected, rtol=1e-10)

    # ppf(expected) should get us back to x.
    xx = stats.levy.ppf(expected)
    assert_allclose(xx, x, rtol=1e-13)


def test_levy_sf():
    # Large values, far into the tail of the distribution.
    x = np.array([1e15, 1e25, 1e35, 1e50])
    # Expected values were calculated with mpmath.
    expected = np.array([2.5231325220201597e-08,
                         2.52313252202016e-13,
                         2.52313252202016e-18,
                         7.978845608028653e-26])
    y = stats.levy.sf(x)
    assert_allclose(y, expected, rtol=1e-14)


# The expected values for levy.isf(p) were calculated with mpmath.
# For loc=0 and scale=1, the inverse SF can be computed with
#
#     import mpmath
#
#     def levy_invsf(p):
#         return 1/(2*mpmath.erfinv(p)**2)
#
# For example, with mpmath.mp.dps set to 60, float(levy_invsf(1e-20))
# returns 6.366197723675814e+39.
#
@pytest.mark.parametrize('p, expected_isf',
                         [(1e-20, 6.366197723675814e+39),
                          (1e-8, 6366197723675813.0),
                          (0.375, 4.185810119346273),
                          (0.875, 0.42489442055310134),
                          (0.999, 0.09235685880262713),
                          (0.9999999962747097, 0.028766845244146945)])
def test_levy_isf(p, expected_isf):
    x = stats.levy.isf(p)
    assert_allclose(x, expected_isf, atol=5e-15)


def test_levy_l_sf():
    # Test levy_l.sf for small arguments.
    x = np.array([-0.016, -0.01, -0.005, -0.0015])
    # Expected values were calculated with mpmath.
    expected = np.array([2.6644463892359302e-15,
                         1.523970604832107e-23,
                         2.0884875837625492e-45,
                         5.302850374626878e-147])
    y = stats.levy_l.sf(x)
    assert_allclose(y, expected, rtol=1e-13)


def test_levy_l_isf():
    # Test roundtrip sf(isf(p)), including a small input value.
    p = np.array([3.0e-15, 0.25, 0.99])
    x = stats.levy_l.isf(p)
    q = stats.levy_l.sf(x)
    assert_allclose(q, p, rtol=5e-14)


def test_hypergeom_interval_1802():
    # these two had endless loops
    assert_equal(stats.hypergeom.interval(.95, 187601, 43192, 757),
                 (152.0, 197.0))
    assert_equal(stats.hypergeom.interval(.945, 187601, 43192, 757),
                 (152.0, 197.0))
    # this was working also before
    assert_equal(stats.hypergeom.interval(.94, 187601, 43192, 757),
                 (153.0, 196.0))

    # degenerate case .a == .b
    assert_equal(stats.hypergeom.ppf(0.02, 100, 100, 8), 8)
    assert_equal(stats.hypergeom.ppf(1, 100, 100, 8), 8)


def test_distribution_too_many_args():
    np.random.seed(1234)

    # Check that a TypeError is raised when too many args are given to a method
    # Regression test for ticket 1815.
    x = np.linspace(0.1, 0.7, num=5)
    assert_raises(TypeError, stats.gamma.pdf, x, 2, 3, loc=1.0)
    assert_raises(TypeError, stats.gamma.pdf, x, 2, 3, 4, loc=1.0)
    assert_raises(TypeError, stats.gamma.pdf, x, 2, 3, 4, 5)
    assert_raises(TypeError, stats.gamma.pdf, x, 2, 3, loc=1.0, scale=0.5)
    assert_raises(TypeError, stats.gamma.rvs, 2., 3, loc=1.0, scale=0.5)
    assert_raises(TypeError, stats.gamma.cdf, x, 2., 3, loc=1.0, scale=0.5)
    assert_raises(TypeError, stats.gamma.ppf, x, 2., 3, loc=1.0, scale=0.5)
    assert_raises(TypeError, stats.gamma.stats, 2., 3, loc=1.0, scale=0.5)
    assert_raises(TypeError, stats.gamma.entropy, 2., 3, loc=1.0, scale=0.5)
    assert_raises(TypeError, stats.gamma.fit, x, 2., 3, loc=1.0, scale=0.5)

    # These should not give errors
    stats.gamma.pdf(x, 2, 3)  # loc=3
    stats.gamma.pdf(x, 2, 3, 4)  # loc=3, scale=4
    stats.gamma.stats(2., 3)
    stats.gamma.stats(2., 3, 4)
    stats.gamma.stats(2., 3, 4, 'mv')
    stats.gamma.rvs(2., 3, 4, 5)
    stats.gamma.fit(stats.gamma.rvs(2., size=7), 2.)

    # Also for a discrete distribution
    stats.geom.pmf(x, 2, loc=3)  # no error, loc=3
    assert_raises(TypeError, stats.geom.pmf, x, 2, 3, 4)
    assert_raises(TypeError, stats.geom.pmf, x, 2, 3, loc=4)

    # And for distributions with 0, 2 and 3 args respectively
    assert_raises(TypeError, stats.expon.pdf, x, 3, loc=1.0)
    assert_raises(TypeError, stats.exponweib.pdf, x, 3, 4, 5, loc=1.0)
    assert_raises(TypeError, stats.exponweib.pdf, x, 3, 4, 5, 0.1, 0.1)
    assert_raises(TypeError, stats.ncf.pdf, x, 3, 4, 5, 6, loc=1.0)
    assert_raises(TypeError, stats.ncf.pdf, x, 3, 4, 5, 6, 1.0, scale=0.5)
    stats.ncf.pdf(x, 3, 4, 5, 6, 1.0)  # 3 args, plus loc/scale


def test_ncx2_tails_ticket_955():
    # Trac #955 -- check that the cdf computed by special functions
    # matches the integrated pdf
    a = stats.ncx2.cdf(np.arange(20, 25, 0.2), 2, 1.07458615e+02)
    b = stats.ncx2._cdfvec(np.arange(20, 25, 0.2), 2, 1.07458615e+02)
    assert_allclose(a, b, rtol=1e-3, atol=0)


def test_ncx2_tails_pdf():
    # ncx2.pdf does not return nans in extreme tails(example from gh-1577)
    # NB: this is to check that nan_to_num is not needed in ncx2.pdf
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        assert_equal(stats.ncx2.pdf(1, np.arange(340, 350), 2), 0)
        logval = stats.ncx2.logpdf(1, np.arange(340, 350), 2)

    assert_(np.isneginf(logval).all())

    # Verify logpdf has extended precision when pdf underflows to 0
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        assert_equal(stats.ncx2.pdf(10000, 3, 12), 0)
        assert_allclose(stats.ncx2.logpdf(10000, 3, 12), -4662.444377524883)


@pytest.mark.parametrize('method, expected', [
    ('cdf', np.array([2.497951336e-09, 3.437288941e-10])),
    ('pdf', np.array([1.238579980e-07, 1.710041145e-08])),
    ('logpdf', np.array([-15.90413011, -17.88416331])),
    ('ppf', np.array([4.865182052, 7.017182271]))
])
def test_ncx2_zero_nc(method, expected):
    # gh-5441
    # ncx2 with nc=0 is identical to chi2
    # Comparison to R (v3.5.1)
    # > options(digits=10)
    # > pchisq(0.1, df=10, ncp=c(0,4))
    # > dchisq(0.1, df=10, ncp=c(0,4))
    # > dchisq(0.1, df=10, ncp=c(0,4), log=TRUE)
    # > qchisq(0.1, df=10, ncp=c(0,4))

    result = getattr(stats.ncx2, method)(0.1, nc=[0, 4], df=10)
    assert_allclose(result, expected, atol=1e-15)


def test_ncx2_zero_nc_rvs():
    # gh-5441
    # ncx2 with nc=0 is identical to chi2
    result = stats.ncx2.rvs(df=10, nc=0, random_state=1)
    expected = stats.chi2.rvs(df=10, random_state=1)
    assert_allclose(result, expected, atol=1e-15)


def test_ncx2_gh12731():
    # test that gh-12731 is resolved; previously these were all 0.5
    nc = 10**np.arange(5, 10)
    assert_equal(stats.ncx2.cdf(1e4, df=1, nc=nc), 0)


def test_ncx2_gh8665():
    # test that gh-8665 is resolved; previously this tended to nonzero value
    x = np.array([4.99515382e+00, 1.07617327e+01, 2.31854502e+01,
                  4.99515382e+01, 1.07617327e+02, 2.31854502e+02,
                  4.99515382e+02, 1.07617327e+03, 2.31854502e+03,
                  4.99515382e+03, 1.07617327e+04, 2.31854502e+04,
                  4.99515382e+04])
    nu, lam = 20, 499.51538166556196

    sf = stats.ncx2.sf(x, df=nu, nc=lam)
    # computed in R. Couldn't find a survival function implementation
    # options(digits=16)
    # x <- c(4.99515382e+00, 1.07617327e+01, 2.31854502e+01, 4.99515382e+01,
    #        1.07617327e+02, 2.31854502e+02, 4.99515382e+02, 1.07617327e+03,
    #        2.31854502e+03, 4.99515382e+03, 1.07617327e+04, 2.31854502e+04,
    #        4.99515382e+04)
    # nu <- 20
    # lam <- 499.51538166556196
    # 1 - pchisq(x, df = nu, ncp = lam)
    sf_expected = [1.0000000000000000, 1.0000000000000000, 1.0000000000000000,
                   1.0000000000000000, 1.0000000000000000, 0.9999999999999888,
                   0.6646525582135460, 0.0000000000000000, 0.0000000000000000,
                   0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
                   0.0000000000000000]
    assert_allclose(sf, sf_expected, atol=1e-12)


def test_ncx2_gh11777():
    # regression test for gh-11777:
    # At high values of degrees of freedom df, ensure the pdf of ncx2 does
    # not get clipped to zero when the non-centrality parameter is
    # sufficiently less than df
    df = 6700
    nc = 5300
    x = np.linspace(stats.ncx2.ppf(0.001, df, nc),
                    stats.ncx2.ppf(0.999, df, nc), num=10000)
    ncx2_pdf = stats.ncx2.pdf(x, df, nc)
    gauss_approx = stats.norm.pdf(x, df + nc, np.sqrt(2 * df + 4 * nc))
    # use huge tolerance as we're only looking for obvious discrepancy
    assert_allclose(ncx2_pdf, gauss_approx, atol=1e-4)


# Expected values for foldnorm.sf were computed with mpmath:
#
#    from mpmath import mp
#    mp.dps = 60
#    def foldcauchy_sf(x, c):
#        x = mp.mpf(x)
#        c = mp.mpf(c)
#        return mp.one - (mp.atan(x - c) + mp.atan(x + c))/mp.pi
#
# E.g.
#
#    >>> float(foldcauchy_sf(2, 1))
#    0.35241638234956674
#
@pytest.mark.parametrize('x, c, expected',
                         [(2, 1, 0.35241638234956674),
                          (2, 2, 0.5779791303773694),
                          (1e13, 1, 6.366197723675813e-14),
                          (2e16, 1, 3.183098861837907e-17),
                          (1e13, 2e11, 6.368745221764519e-14),
                          (0.125, 200, 0.999998010612169)])
def test_foldcauchy_sf(x, c, expected):
    sf = stats.foldcauchy.sf(x, c)
    assert_allclose(sf, expected, 2e-15)


# The same mpmath code shown in the comments above test_foldcauchy_sf()
# is used to create these expected values.
@pytest.mark.parametrize('x, expected',
                         [(2, 0.2951672353008665),
                          (1e13, 6.366197723675813e-14),
                          (2e16, 3.183098861837907e-17),
                          (5e80, 1.2732395447351629e-81)])
def test_halfcauchy_sf(x, expected):
    sf = stats.halfcauchy.sf(x)
    assert_allclose(sf, expected, 2e-15)


# Expected value computed with mpmath:
#     expected = mp.cot(mp.pi*p/2)
@pytest.mark.parametrize('p, expected',
                         [(0.9999995, 7.853981633329977e-07),
                          (0.975, 0.039290107007669675),
                          (0.5, 1.0),
                          (0.01, 63.65674116287158),
                          (1e-14, 63661977236758.13),
                          (5e-80, 1.2732395447351627e+79)])
def test_halfcauchy_isf(p, expected):
    x = stats.halfcauchy.isf(p)
    assert_allclose(x, expected)


def test_foldnorm_zero():
    # Parameter value c=0 was not enabled, see gh-2399.
    rv = stats.foldnorm(0, scale=1)
    assert_equal(rv.cdf(0), 0)  # rv.cdf(0) previously resulted in: nan


# Expected values for foldnorm.sf were computed with mpmath:
#
#    from mpmath import mp
#    mp.dps = 60
#    def foldnorm_sf(x, c):
#        x = mp.mpf(x)
#        c = mp.mpf(c)
#        return mp.ncdf(-x+c) + mp.ncdf(-x-c)
#
# E.g.
#
#    >>> float(foldnorm_sf(2, 1))
#    0.16000515196308715
#
@pytest.mark.parametrize('x, c, expected',
                         [(2, 1, 0.16000515196308715),
                          (20, 1, 8.527223952630977e-81),
                          (10, 15, 0.9999997133484281),
                          (25, 15, 7.619853024160525e-24)])
def test_foldnorm_sf(x, c, expected):
    sf = stats.foldnorm.sf(x, c)
    assert_allclose(sf, expected, 1e-14)


def test_stats_shapes_argcheck():
    # stats method was failing for vector shapes if some of the values
    # were outside of the allowed range, see gh-2678
    mv3 = stats.invgamma.stats([0.0, 0.5, 1.0], 1, 0.5)  # 0 is not a legal `a`
    mv2 = stats.invgamma.stats([0.5, 1.0], 1, 0.5)
    mv2_augmented = tuple(np.r_[np.nan, _] for _ in mv2)
    assert_equal(mv2_augmented, mv3)

    # -1 is not a legal shape parameter
    mv3 = stats.lognorm.stats([2, 2.4, -1])
    mv2 = stats.lognorm.stats([2, 2.4])
    mv2_augmented = tuple(np.r_[_, np.nan] for _ in mv2)
    assert_equal(mv2_augmented, mv3)

    # FIXME: this is only a quick-and-dirty test of a quick-and-dirty bugfix.
    # stats method with multiple shape parameters is not properly vectorized
    # anyway, so some distributions may or may not fail.


# Test subclassing distributions w/ explicit shapes

class _distr_gen(stats.rv_continuous):
    def _pdf(self, x, a):
        return 42


class _distr2_gen(stats.rv_continuous):
    def _cdf(self, x, a):
        return 42 * a + x


class _distr3_gen(stats.rv_continuous):
    def _pdf(self, x, a, b):
        return a + b

    def _cdf(self, x, a):
        # Different # of shape params from _pdf, to be able to check that
        # inspection catches the inconsistency.
        return 42 * a + x


class _distr6_gen(stats.rv_continuous):
    # Two shape parameters (both _pdf and _cdf defined, consistent shapes.)
    def _pdf(self, x, a, b):
        return a*x + b

    def _cdf(self, x, a, b):
        return 42 * a + x


class TestSubclassingExplicitShapes:
    # Construct a distribution w/ explicit shapes parameter and test it.

    def test_correct_shapes(self):
        dummy_distr = _distr_gen(name='dummy', shapes='a')
        assert_equal(dummy_distr.pdf(1, a=1), 42)

    def test_wrong_shapes_1(self):
        dummy_distr = _distr_gen(name='dummy', shapes='A')
        assert_raises(TypeError, dummy_distr.pdf, 1, **dict(a=1))

    def test_wrong_shapes_2(self):
        dummy_distr = _distr_gen(name='dummy', shapes='a, b, c')
        dct = dict(a=1, b=2, c=3)
        assert_raises(TypeError, dummy_distr.pdf, 1, **dct)

    def test_shapes_string(self):
        # shapes must be a string
        dct = dict(name='dummy', shapes=42)
        assert_raises(TypeError, _distr_gen, **dct)

    def test_shapes_identifiers_1(self):
        # shapes must be a comma-separated list of valid python identifiers
        dct = dict(name='dummy', shapes='(!)')
        assert_raises(SyntaxError, _distr_gen, **dct)

    def test_shapes_identifiers_2(self):
        dct = dict(name='dummy', shapes='4chan')
        assert_raises(SyntaxError, _distr_gen, **dct)

    def test_shapes_identifiers_3(self):
        dct = dict(name='dummy', shapes='m(fti)')
        assert_raises(SyntaxError, _distr_gen, **dct)

    def test_shapes_identifiers_nodefaults(self):
        dct = dict(name='dummy', shapes='a=2')
        assert_raises(SyntaxError, _distr_gen, **dct)

    def test_shapes_args(self):
        dct = dict(name='dummy', shapes='*args')
        assert_raises(SyntaxError, _distr_gen, **dct)

    def test_shapes_kwargs(self):
        dct = dict(name='dummy', shapes='**kwargs')
        assert_raises(SyntaxError, _distr_gen, **dct)

    def test_shapes_keywords(self):
        # python keywords cannot be used for shape parameters
        dct = dict(name='dummy', shapes='a, b, c, lambda')
        assert_raises(SyntaxError, _distr_gen, **dct)

    def test_shapes_signature(self):
        # test explicit shapes which agree w/ the signature of _pdf
        class _dist_gen(stats.rv_continuous):
            def _pdf(self, x, a):
                return stats.norm._pdf(x) * a

        dist = _dist_gen(shapes='a')
        assert_equal(dist.pdf(0.5, a=2), stats.norm.pdf(0.5)*2)

    def test_shapes_signature_inconsistent(self):
        # test explicit shapes which do not agree w/ the signature of _pdf
        class _dist_gen(stats.rv_continuous):
            def _pdf(self, x, a):
                return stats.norm._pdf(x) * a

        dist = _dist_gen(shapes='a, b')
        assert_raises(TypeError, dist.pdf, 0.5, **dict(a=1, b=2))

    def test_star_args(self):
        # test _pdf with only starargs
        # NB: **kwargs of pdf will never reach _pdf
        class _dist_gen(stats.rv_continuous):
            def _pdf(self, x, *args):
                extra_kwarg = args[0]
                return stats.norm._pdf(x) * extra_kwarg

        dist = _dist_gen(shapes='extra_kwarg')
        assert_equal(dist.pdf(0.5, extra_kwarg=33), stats.norm.pdf(0.5)*33)
        assert_equal(dist.pdf(0.5, 33), stats.norm.pdf(0.5)*33)
        assert_raises(TypeError, dist.pdf, 0.5, **dict(xxx=33))

    def test_star_args_2(self):
        # test _pdf with named & starargs
        # NB: **kwargs of pdf will never reach _pdf
        class _dist_gen(stats.rv_continuous):
            def _pdf(self, x, offset, *args):
                extra_kwarg = args[0]
                return stats.norm._pdf(x) * extra_kwarg + offset

        dist = _dist_gen(shapes='offset, extra_kwarg')
        assert_equal(dist.pdf(0.5, offset=111, extra_kwarg=33),
                     stats.norm.pdf(0.5)*33 + 111)
        assert_equal(dist.pdf(0.5, 111, 33),
                     stats.norm.pdf(0.5)*33 + 111)

    def test_extra_kwarg(self):
        # **kwargs to _pdf are ignored.
        # this is a limitation of the framework (_pdf(x, *goodargs))
        class _distr_gen(stats.rv_continuous):
            def _pdf(self, x, *args, **kwargs):
                # _pdf should handle *args, **kwargs itself.  Here "handling"
                # is ignoring *args and looking for ``extra_kwarg`` and using
                # that.
                extra_kwarg = kwargs.pop('extra_kwarg', 1)
                return stats.norm._pdf(x) * extra_kwarg

        dist = _distr_gen(shapes='extra_kwarg')
        assert_equal(dist.pdf(1, extra_kwarg=3), stats.norm.pdf(1))

    def test_shapes_empty_string(self):
        # shapes='' is equivalent to shapes=None
        class _dist_gen(stats.rv_continuous):
            def _pdf(self, x):
                return stats.norm.pdf(x)

        dist = _dist_gen(shapes='')
        assert_equal(dist.pdf(0.5), stats.norm.pdf(0.5))


class TestSubclassingNoShapes:
    # Construct a distribution w/o explicit shapes parameter and test it.

    def test_only__pdf(self):
        dummy_distr = _distr_gen(name='dummy')
        assert_equal(dummy_distr.pdf(1, a=1), 42)

    def test_only__cdf(self):
        # _pdf is determined from _cdf by taking numerical derivative
        dummy_distr = _distr2_gen(name='dummy')
        assert_almost_equal(dummy_distr.pdf(1, a=1), 1)

    @pytest.mark.skipif(DOCSTRINGS_STRIPPED, reason="docstring stripped")
    def test_signature_inspection(self):
        # check that _pdf signature inspection works correctly, and is used in
        # the class docstring
        dummy_distr = _distr_gen(name='dummy')
        assert_equal(dummy_distr.numargs, 1)
        assert_equal(dummy_distr.shapes, 'a')
        res = re.findall(r'logpdf\(x, a, loc=0, scale=1\)',
                         dummy_distr.__doc__)
        assert_(len(res) == 1)

    @pytest.mark.skipif(DOCSTRINGS_STRIPPED, reason="docstring stripped")
    def test_signature_inspection_2args(self):
        # same for 2 shape params and both _pdf and _cdf defined
        dummy_distr = _distr6_gen(name='dummy')
        assert_equal(dummy_distr.numargs, 2)
        assert_equal(dummy_distr.shapes, 'a, b')
        res = re.findall(r'logpdf\(x, a, b, loc=0, scale=1\)',
                         dummy_distr.__doc__)
        assert_(len(res) == 1)

    def test_signature_inspection_2args_incorrect_shapes(self):
        # both _pdf and _cdf defined, but shapes are inconsistent: raises
        assert_raises(TypeError, _distr3_gen, name='dummy')

    def test_defaults_raise(self):
        # default arguments should raise
        class _dist_gen(stats.rv_continuous):
            def _pdf(self, x, a=42):
                return 42
        assert_raises(TypeError, _dist_gen, **dict(name='dummy'))

    def test_starargs_raise(self):
        # without explicit shapes, *args are not allowed
        class _dist_gen(stats.rv_continuous):
            def _pdf(self, x, a, *args):
                return 42
        assert_raises(TypeError, _dist_gen, **dict(name='dummy'))

    def test_kwargs_raise(self):
        # without explicit shapes, **kwargs are not allowed
        class _dist_gen(stats.rv_continuous):
            def _pdf(self, x, a, **kwargs):
                return 42
        assert_raises(TypeError, _dist_gen, **dict(name='dummy'))


@pytest.mark.skipif(DOCSTRINGS_STRIPPED, reason="docstring stripped")
def test_docstrings():
    badones = [r',\s*,', r'\(\s*,', r'^\s*:']
    for distname in stats.__all__:
        dist = getattr(stats, distname)
        if isinstance(dist, (stats.rv_discrete, stats.rv_continuous)):
            for regex in badones:
                assert_(re.search(regex, dist.__doc__) is None)


def test_infinite_input():
    assert_almost_equal(stats.skellam.sf(np.inf, 10, 11), 0)
    assert_almost_equal(stats.ncx2._cdf(np.inf, 8, 0.1), 1)


def test_lomax_accuracy():
    # regression test for gh-4033
    p = stats.lomax.ppf(stats.lomax.cdf(1e-100, 1), 1)
    assert_allclose(p, 1e-100)


def test_truncexpon_accuracy():
    # regression test for gh-4035
    p = stats.truncexpon.ppf(stats.truncexpon.cdf(1e-100, 1), 1)
    assert_allclose(p, 1e-100)


def test_rayleigh_accuracy():
    # regression test for gh-4034
    p = stats.rayleigh.isf(stats.rayleigh.sf(9, 1), 1)
    assert_almost_equal(p, 9.0, decimal=15)


def test_genextreme_give_no_warnings():
    """regression test for gh-6219"""

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        stats.genextreme.cdf(.5, 0)
        stats.genextreme.pdf(.5, 0)
        stats.genextreme.ppf(.5, 0)
        stats.genextreme.logpdf(-np.inf, 0.0)
        number_of_warnings_thrown = len(w)
        assert_equal(number_of_warnings_thrown, 0)


def test_genextreme_entropy():
    # regression test for gh-5181
    euler_gamma = 0.5772156649015329

    h = stats.genextreme.entropy(-1.0)
    assert_allclose(h, 2*euler_gamma + 1, rtol=1e-14)

    h = stats.genextreme.entropy(0)
    assert_allclose(h, euler_gamma + 1, rtol=1e-14)

    h = stats.genextreme.entropy(1.0)
    assert_equal(h, 1)

    h = stats.genextreme.entropy(-2.0, scale=10)
    assert_allclose(h, euler_gamma*3 + np.log(10) + 1, rtol=1e-14)

    h = stats.genextreme.entropy(10)
    assert_allclose(h, -9*euler_gamma + 1, rtol=1e-14)

    h = stats.genextreme.entropy(-10)
    assert_allclose(h, 11*euler_gamma + 1, rtol=1e-14)


def test_genextreme_sf_isf():
    # Expected values were computed using mpmath:
    #
    #    import mpmath
    #
    #    def mp_genextreme_sf(x, xi, mu=0, sigma=1):
    #        # Formula from wikipedia, which has a sign convention for xi that
    #        # is the opposite of scipy's shape parameter.
    #        if xi != 0:
    #            t = mpmath.power(1 + ((x - mu)/sigma)*xi, -1/xi)
    #        else:
    #            t = mpmath.exp(-(x - mu)/sigma)
    #        return 1 - mpmath.exp(-t)
    #
    # >>> mpmath.mp.dps = 1000
    # >>> s = mp_genextreme_sf(mpmath.mp.mpf("1e8"), mpmath.mp.mpf("0.125"))
    # >>> float(s)
    # 1.6777205262585625e-57
    # >>> s = mp_genextreme_sf(mpmath.mp.mpf("7.98"), mpmath.mp.mpf("-0.125"))
    # >>> float(s)
    # 1.52587890625e-21
    # >>> s = mp_genextreme_sf(mpmath.mp.mpf("7.98"), mpmath.mp.mpf("0"))
    # >>> float(s)
    # 0.00034218086528426593

    x = 1e8
    s = stats.genextreme.sf(x, -0.125)
    assert_allclose(s, 1.6777205262585625e-57)
    x2 = stats.genextreme.isf(s, -0.125)
    assert_allclose(x2, x)

    x = 7.98
    s = stats.genextreme.sf(x, 0.125)
    assert_allclose(s, 1.52587890625e-21)
    x2 = stats.genextreme.isf(s, 0.125)
    assert_allclose(x2, x)

    x = 7.98
    s = stats.genextreme.sf(x, 0)
    assert_allclose(s, 0.00034218086528426593)
    x2 = stats.genextreme.isf(s, 0)
    assert_allclose(x2, x)


def test_burr12_ppf_small_arg():
    prob = 1e-16
    quantile = stats.burr12.ppf(prob, 2, 3)
    # The expected quantile was computed using mpmath:
    #   >>> import mpmath
    #   >>> mpmath.mp.dps = 100
    #   >>> prob = mpmath.mpf('1e-16')
    #   >>> c = mpmath.mpf(2)
    #   >>> d = mpmath.mpf(3)
    #   >>> float(((1-prob)**(-1/d) - 1)**(1/c))
    #   5.7735026918962575e-09
    assert_allclose(quantile, 5.7735026918962575e-09)


def test_crystalball_function():
    """
    All values are calculated using the independent implementation of the
    ROOT framework (see https://root.cern.ch/).
    Corresponding ROOT code is given in the comments.
    """
    X = np.linspace(-5.0, 5.0, 21)[:-1]

    # for(float x = -5.0; x < 5.0; x+=0.5)
    #   std::cout << ROOT::Math::crystalball_pdf(x, 1.0, 2.0, 1.0) << ", ";
    calculated = stats.crystalball.pdf(X, beta=1.0, m=2.0)
    expected = np.array([0.0202867, 0.0241428, 0.0292128, 0.0360652, 0.045645,
                         0.059618, 0.0811467, 0.116851, 0.18258, 0.265652,
                         0.301023, 0.265652, 0.18258, 0.097728, 0.0407391,
                         0.013226, 0.00334407, 0.000658486, 0.000100982,
                         1.20606e-05])
    assert_allclose(expected, calculated, rtol=0.001)

    # for(float x = -5.0; x < 5.0; x+=0.5)
    #   std::cout << ROOT::Math::crystalball_pdf(x, 2.0, 3.0, 1.0) << ", ";
    calculated = stats.crystalball.pdf(X, beta=2.0, m=3.0)
    expected = np.array([0.0019648, 0.00279754, 0.00417592, 0.00663121,
                         0.0114587, 0.0223803, 0.0530497, 0.12726, 0.237752,
                         0.345928, 0.391987, 0.345928, 0.237752, 0.12726,
                         0.0530497, 0.0172227, 0.00435458, 0.000857469,
                         0.000131497, 1.57051e-05])
    assert_allclose(expected, calculated, rtol=0.001)

    # for(float x = -5.0; x < 5.0; x+=0.5) {
    #   std::cout << ROOT::Math::crystalball_pdf(x, 2.0, 3.0, 2.0, 0.5);
    #   std::cout << ", ";
    # }
    calculated = stats.crystalball.pdf(X, beta=2.0, m=3.0, loc=0.5, scale=2.0)
    expected = np.array([0.00785921, 0.0111902, 0.0167037, 0.0265249,
                         0.0423866, 0.0636298, 0.0897324, 0.118876, 0.147944,
                         0.172964, 0.189964, 0.195994, 0.189964, 0.172964,
                         0.147944, 0.118876, 0.0897324, 0.0636298, 0.0423866,
                         0.0265249])
    assert_allclose(expected, calculated, rtol=0.001)

    # for(float x = -5.0; x < 5.0; x+=0.5)
    #   std::cout << ROOT::Math::crystalball_cdf(x, 1.0, 2.0, 1.0) << ", ";
    calculated = stats.crystalball.cdf(X, beta=1.0, m=2.0)
    expected = np.array([0.12172, 0.132785, 0.146064, 0.162293, 0.18258,
                         0.208663, 0.24344, 0.292128, 0.36516, 0.478254,
                         0.622723, 0.767192, 0.880286, 0.94959, 0.982834,
                         0.995314, 0.998981, 0.999824, 0.999976, 0.999997])
    assert_allclose(expected, calculated, rtol=0.001)

    # for(float x = -5.0; x < 5.0; x+=0.5)
    #   std::cout << ROOT::Math::crystalball_cdf(x, 2.0, 3.0, 1.0) << ", ";
    calculated = stats.crystalball.cdf(X, beta=2.0, m=3.0)
    expected = np.array([0.00442081, 0.00559509, 0.00730787, 0.00994682,
                         0.0143234, 0.0223803, 0.0397873, 0.0830763, 0.173323,
                         0.320592, 0.508717, 0.696841, 0.844111, 0.934357,
                         0.977646, 0.993899, 0.998674, 0.999771, 0.999969,
                         0.999997])
    assert_allclose(expected, calculated, rtol=0.001)

    # for(float x = -5.0; x < 5.0; x+=0.5) {
    #   std::cout << ROOT::Math::crystalball_cdf(x, 2.0, 3.0, 2.0, 0.5);
    #   std::cout << ", ";
    # }
    calculated = stats.crystalball.cdf(X, beta=2.0, m=3.0, loc=0.5, scale=2.0)
    expected = np.array([0.0176832, 0.0223803, 0.0292315, 0.0397873, 0.0567945,
                         0.0830763, 0.121242, 0.173323, 0.24011, 0.320592,
                         0.411731, 0.508717, 0.605702, 0.696841, 0.777324,
                         0.844111, 0.896192, 0.934357, 0.960639, 0.977646])
    assert_allclose(expected, calculated, rtol=0.001)


def test_crystalball_function_moments():
    """
    All values are calculated using the pdf formula and the integrate function
    of Mathematica
    """
    # The Last two (alpha, n) pairs test the special case n == alpha**2
    beta = np.array([2.0, 1.0, 3.0, 2.0, 3.0])
    m = np.array([3.0, 3.0, 2.0, 4.0, 9.0])

    # The distribution should be correctly normalised
    expected_0th_moment = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    calculated_0th_moment = stats.crystalball._munp(0, beta, m)
    assert_allclose(expected_0th_moment, calculated_0th_moment, rtol=0.001)

    # calculated using wolframalpha.com
    # e.g. for beta = 2 and m = 3 we calculate the norm like this:
    #   integrate exp(-x^2/2) from -2 to infinity +
    #   integrate (3/2)^3*exp(-2^2/2)*(3/2-2-x)^(-3) from -infinity to -2
    norm = np.array([2.5511, 3.01873, 2.51065, 2.53983, 2.507410455])

    a = np.array([-0.21992, -3.03265, np.inf, -0.135335, -0.003174])
    expected_1th_moment = a / norm
    calculated_1th_moment = stats.crystalball._munp(1, beta, m)
    assert_allclose(expected_1th_moment, calculated_1th_moment, rtol=0.001)

    a = np.array([np.inf, np.inf, np.inf, 3.2616, 2.519908])
    expected_2th_moment = a / norm
    calculated_2th_moment = stats.crystalball._munp(2, beta, m)
    assert_allclose(expected_2th_moment, calculated_2th_moment, rtol=0.001)

    a = np.array([np.inf, np.inf, np.inf, np.inf, -0.0577668])
    expected_3th_moment = a / norm
    calculated_3th_moment = stats.crystalball._munp(3, beta, m)
    assert_allclose(expected_3th_moment, calculated_3th_moment, rtol=0.001)

    a = np.array([np.inf, np.inf, np.inf, np.inf, 7.78468])
    expected_4th_moment = a / norm
    calculated_4th_moment = stats.crystalball._munp(4, beta, m)
    assert_allclose(expected_4th_moment, calculated_4th_moment, rtol=0.001)

    a = np.array([np.inf, np.inf, np.inf, np.inf, -1.31086])
    expected_5th_moment = a / norm
    calculated_5th_moment = stats.crystalball._munp(5, beta, m)
    assert_allclose(expected_5th_moment, calculated_5th_moment, rtol=0.001)


def test_crystalball_entropy():
    # regression test for gh-13602
    cb = stats.crystalball(2, 3)
    res1 = cb.entropy()
    # -20000 and 30 are negative and positive infinity, respectively
    lo, hi, N = -20000, 30, 200000
    x = np.linspace(lo, hi, N)
    res2 = trapezoid(entr(cb.pdf(x)), x)
    assert_allclose(res1, res2, rtol=1e-7)


def test_invweibull_fit():
    """
    Test fitting invweibull to data.

    Here is a the same calculation in R:

    > library(evd)
    > library(fitdistrplus)
    > x = c(1, 1.25, 2, 2.5, 2.8,  3, 3.8, 4, 5, 8, 10, 12, 64, 99)
    > result = fitdist(x, 'frechet', control=list(reltol=1e-13),
    +                  fix.arg=list(loc=0), start=list(shape=2, scale=3))
    > result
    Fitting of the distribution ' frechet ' by maximum likelihood
    Parameters:
          estimate Std. Error
    shape 1.048482  0.2261815
    scale 3.099456  0.8292887
    Fixed parameters:
        value
    loc     0

    """

    def optimizer(func, x0, args=(), disp=0):
        return fmin(func, x0, args=args, disp=disp, xtol=1e-12, ftol=1e-12)

    x = np.array([1, 1.25, 2, 2.5, 2.8, 3, 3.8, 4, 5, 8, 10, 12, 64, 99])
    c, loc, scale = stats.invweibull.fit(x, floc=0, optimizer=optimizer)
    assert_allclose(c, 1.048482, rtol=5e-6)
    assert loc == 0
    assert_allclose(scale, 3.099456, rtol=5e-6)


# Expected values were computed with mpmath.
@pytest.mark.parametrize('x, c, expected',
                         [(3, 1.5, 0.175064510070713299327),
                          (2000, 1.5, 1.11802773877318715787e-5),
                          (2000, 9.25, 2.92060308832269637092e-31),
                          (1e15, 1.5, 3.16227766016837933199884e-23)])
def test_invweibull_sf(x, c, expected):
    computed = stats.invweibull.sf(x, c)
    assert_allclose(computed, expected, rtol=1e-15)


# Expected values were computed with mpmath.
@pytest.mark.parametrize('p, c, expected',
                         [(0.5, 2.5, 1.15789669836468183976),
                          (3e-18, 5, 3195.77171838060906447)])
def test_invweibull_isf(p, c, expected):
    computed = stats.invweibull.isf(p, c)
    assert_allclose(computed, expected, rtol=1e-15)


@pytest.mark.parametrize(
    'df1,df2,x',
    [(2, 2, [-0.5, 0.2, 1.0, 2.3]),
     (4, 11, [-0.5, 0.2, 1.0, 2.3]),
     (7, 17, [1, 2, 3, 4, 5])]
)
def test_ncf_edge_case(df1, df2, x):
    # Test for edge case described in gh-11660.
    # Non-central Fisher distribution when nc = 0
    # should be the same as Fisher distribution.
    nc = 0
    expected_cdf = stats.f.cdf(x, df1, df2)
    calculated_cdf = stats.ncf.cdf(x, df1, df2, nc)
    assert_allclose(expected_cdf, calculated_cdf, rtol=1e-14)

    # when ncf_gen._skip_pdf will be used instead of generic pdf,
    # this additional test will be useful.
    expected_pdf = stats.f.pdf(x, df1, df2)
    calculated_pdf = stats.ncf.pdf(x, df1, df2, nc)
    assert_allclose(expected_pdf, calculated_pdf, rtol=1e-6)


def test_ncf_variance():
    # Regression test for gh-10658 (incorrect variance formula for ncf).
    # The correct value of ncf.var(2, 6, 4), 42.75, can be verified with, for
    # example, Wolfram Alpha with the expression
    #     Variance[NoncentralFRatioDistribution[2, 6, 4]]
    # or with the implementation of the noncentral F distribution in the C++
    # library Boost.
    v = stats.ncf.var(2, 6, 4)
    assert_allclose(v, 42.75, rtol=1e-14)


def test_ncf_cdf_spotcheck():
    # Regression test for gh-15582 testing against values from R/MATLAB
    # Generate check_val from R or MATLAB as follows:
    #          R: pf(20, df1 = 6, df2 = 33, ncp = 30.4) = 0.998921
    #     MATLAB: ncfcdf(20, 6, 33, 30.4) = 0.998921
    scipy_val = stats.ncf.cdf(20, 6, 33, 30.4)
    check_val = 0.998921
    assert_allclose(check_val, np.round(scipy_val, decimals=6))


@pytest.mark.skipif(sys.maxsize <= 2**32,
                    reason="On some 32-bit the warning is not raised")
def test_ncf_ppf_issue_17026():
    # Regression test for gh-17026
    x = np.linspace(0, 1, 600)
    x[0] = 1e-16
    par = (0.1, 2, 5, 0, 1)
    with pytest.warns(RuntimeWarning):
        q = stats.ncf.ppf(x, *par)
        q0 = [stats.ncf.ppf(xi, *par) for xi in x]
    assert_allclose(q, q0)


class TestHistogram:
    def setup_method(self):
        np.random.seed(1234)

        # We have 8 bins
        # [1,2), [2,3), [3,4), [4,5), [5,6), [6,7), [7,8), [8,9)
        # But actually np.histogram will put the last 9 also in the [8,9) bin!
        # Therefore there is a slight difference below for the last bin, from
        # what you might have expected.
        histogram = np.histogram([1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,
                                  6, 6, 6, 6, 7, 7, 7, 8, 8, 9], bins=8)
        self.template = stats.rv_histogram(histogram)

        data = stats.norm.rvs(loc=1.0, scale=2.5, size=10000, random_state=123)
        norm_histogram = np.histogram(data, bins=50)
        self.norm_template = stats.rv_histogram(norm_histogram)

    def test_pdf(self):
        values = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,
                           5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5])
        pdf_values = np.asarray([0.0/25.0, 0.0/25.0, 1.0/25.0, 1.0/25.0,
                                 2.0/25.0, 2.0/25.0, 3.0/25.0, 3.0/25.0,
                                 4.0/25.0, 4.0/25.0, 5.0/25.0, 5.0/25.0,
                                 4.0/25.0, 4.0/25.0, 3.0/25.0, 3.0/25.0,
                                 3.0/25.0, 3.0/25.0, 0.0/25.0, 0.0/25.0])
        assert_allclose(self.template.pdf(values), pdf_values)

        # Test explicitly the corner cases:
        # As stated above the pdf in the bin [8,9) is greater than
        # one would naively expect because np.histogram putted the 9
        # into the [8,9) bin.
        assert_almost_equal(self.template.pdf(8.0), 3.0/25.0)
        assert_almost_equal(self.template.pdf(8.5), 3.0/25.0)
        # 9 is outside our defined bins [8,9) hence the pdf is already 0
        # for a continuous distribution this is fine, because a single value
        # does not have a finite probability!
        assert_almost_equal(self.template.pdf(9.0), 0.0/25.0)
        assert_almost_equal(self.template.pdf(10.0), 0.0/25.0)

        x = np.linspace(-2, 2, 10)
        assert_allclose(self.norm_template.pdf(x),
                        stats.norm.pdf(x, loc=1.0, scale=2.5), rtol=0.1)

    def test_cdf_ppf(self):
        values = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,
                           5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5])
        cdf_values = np.asarray([0.0/25.0, 0.0/25.0, 0.0/25.0, 0.5/25.0,
                                 1.0/25.0, 2.0/25.0, 3.0/25.0, 4.5/25.0,
                                 6.0/25.0, 8.0/25.0, 10.0/25.0, 12.5/25.0,
                                 15.0/25.0, 17.0/25.0, 19.0/25.0, 20.5/25.0,
                                 22.0/25.0, 23.5/25.0, 25.0/25.0, 25.0/25.0])
        assert_allclose(self.template.cdf(values), cdf_values)
        # First three and last two values in cdf_value are not unique
        assert_allclose(self.template.ppf(cdf_values[2:-1]), values[2:-1])

        # Test of cdf and ppf are inverse functions
        x = np.linspace(1.0, 9.0, 100)
        assert_allclose(self.template.ppf(self.template.cdf(x)), x)
        x = np.linspace(0.0, 1.0, 100)
        assert_allclose(self.template.cdf(self.template.ppf(x)), x)

        x = np.linspace(-2, 2, 10)
        assert_allclose(self.norm_template.cdf(x),
                        stats.norm.cdf(x, loc=1.0, scale=2.5), rtol=0.1)

    def test_rvs(self):
        N = 10000
        sample = self.template.rvs(size=N, random_state=123)
        assert_equal(np.sum(sample < 1.0), 0.0)
        assert_allclose(np.sum(sample <= 2.0), 1.0/25.0 * N, rtol=0.2)
        assert_allclose(np.sum(sample <= 2.5), 2.0/25.0 * N, rtol=0.2)
        assert_allclose(np.sum(sample <= 3.0), 3.0/25.0 * N, rtol=0.1)
        assert_allclose(np.sum(sample <= 3.5), 4.5/25.0 * N, rtol=0.1)
        assert_allclose(np.sum(sample <= 4.0), 6.0/25.0 * N, rtol=0.1)
        assert_allclose(np.sum(sample <= 4.5), 8.0/25.0 * N, rtol=0.1)
        assert_allclose(np.sum(sample <= 5.0), 10.0/25.0 * N, rtol=0.05)
        assert_allclose(np.sum(sample <= 5.5), 12.5/25.0 * N, rtol=0.05)
        assert_allclose(np.sum(sample <= 6.0), 15.0/25.0 * N, rtol=0.05)
        assert_allclose(np.sum(sample <= 6.5), 17.0/25.0 * N, rtol=0.05)
        assert_allclose(np.sum(sample <= 7.0), 19.0/25.0 * N, rtol=0.05)
        assert_allclose(np.sum(sample <= 7.5), 20.5/25.0 * N, rtol=0.05)
        assert_allclose(np.sum(sample <= 8.0), 22.0/25.0 * N, rtol=0.05)
        assert_allclose(np.sum(sample <= 8.5), 23.5/25.0 * N, rtol=0.05)
        assert_allclose(np.sum(sample <= 9.0), 25.0/25.0 * N, rtol=0.05)
        assert_allclose(np.sum(sample <= 9.0), 25.0/25.0 * N, rtol=0.05)
        assert_equal(np.sum(sample > 9.0), 0.0)

    def test_munp(self):
        for n in range(4):
            assert_allclose(self.norm_template._munp(n),
                            stats.norm(1.0, 2.5).moment(n), rtol=0.05)

    def test_entropy(self):
        assert_allclose(self.norm_template.entropy(),
                        stats.norm.entropy(loc=1.0, scale=2.5), rtol=0.05)


def test_histogram_non_uniform():
    # Tests rv_histogram works even for non-uniform bin widths
    counts, bins = ([1, 1], [0, 1, 1001])

    dist = stats.rv_histogram((counts, bins), density=False)
    np.testing.assert_allclose(dist.pdf([0.5, 200]), [0.5, 0.0005])
    assert dist.median() == 1

    dist = stats.rv_histogram((counts, bins), density=True)
    np.testing.assert_allclose(dist.pdf([0.5, 200]), 1/1001)
    assert dist.median() == 1001/2

    # Omitting density produces a warning for non-uniform bins...
    message = "Bin widths are not constant. Assuming..."
    with assert_warns(RuntimeWarning, match=message):
        dist = stats.rv_histogram((counts, bins))
        assert dist.median() == 1001/2  # default is like `density=True`

    # ... but not for uniform bins
    dist = stats.rv_histogram((counts, [0, 1, 2]))
    assert dist.median() == 1


class TestLogUniform:
    def test_alias(self):
        # This test makes sure that "reciprocal" and "loguniform" are
        # aliases of the same distribution and that both are log-uniform
        rng = np.random.default_rng(98643218961)
        rv = stats.loguniform(10 ** -3, 10 ** 0)
        rvs = rv.rvs(size=10000, random_state=rng)

        rng = np.random.default_rng(98643218961)
        rv2 = stats.reciprocal(10 ** -3, 10 ** 0)
        rvs2 = rv2.rvs(size=10000, random_state=rng)

        assert_allclose(rvs2, rvs)

        vals, _ = np.histogram(np.log10(rvs), bins=10)
        assert 900 <= vals.min() <= vals.max() <= 1100
        assert np.abs(np.median(vals) - 1000) <= 10

    @pytest.mark.parametrize("method", ['mle', 'mm'])
    def test_fit_override(self, method):
        # loguniform is overparameterized, so check that fit override enforces
        # scale=1 unless fscale is provided by the user
        rng = np.random.default_rng(98643218961)
        rvs = stats.loguniform.rvs(0.1, 1, size=1000, random_state=rng)

        a, b, loc, scale = stats.loguniform.fit(rvs, method=method)
        assert scale == 1

        a, b, loc, scale = stats.loguniform.fit(rvs, fscale=2, method=method)
        assert scale == 2

    def test_overflow(self):
        # original formulation had overflow issues; check that this is resolved
        # Extensive accuracy tests elsewhere, no need to test all methods
        rng = np.random.default_rng(7136519550773909093)
        a, b = 1e-200, 1e200
        dist = stats.loguniform(a, b)

        # test roundtrip error
        cdf = rng.uniform(0, 1, size=1000)
        assert_allclose(dist.cdf(dist.ppf(cdf)), cdf)
        rvs = dist.rvs(size=1000)
        assert_allclose(dist.ppf(dist.cdf(rvs)), rvs)

        # test a property of the pdf (and that there is no overflow)
        x = 10.**np.arange(-200, 200)
        pdf = dist.pdf(x)  # no overflow
        assert_allclose(pdf[:-1]/pdf[1:], 10)

        # check munp against wikipedia reference
        mean = (b - a)/(np.log(b) - np.log(a))
        assert_allclose(dist.mean(), mean)


class TestArgus:
    def test_argus_rvs_large_chi(self):
        # test that the algorithm can handle large values of chi
        x = stats.argus.rvs(50, size=500, random_state=325)
        assert_almost_equal(stats.argus(50).mean(), x.mean(), decimal=4)

    @pytest.mark.parametrize('chi, random_state', [
            [0.1, 325],   # chi <= 0.5: rejection method case 1
            [1.3, 155],   # 0.5 < chi <= 1.8: rejection method case 2
            [3.5, 135]    # chi > 1.8: transform conditional Gamma distribution
        ])
    def test_rvs(self, chi, random_state):
        x = stats.argus.rvs(chi, size=500, random_state=random_state)
        _, p = stats.kstest(x, "argus", (chi, ))
        assert_(p > 0.05)

    @pytest.mark.parametrize('chi', [1e-9, 1e-6])
    def test_rvs_small_chi(self, chi):
        # test for gh-11699 => rejection method case 1 can even handle chi=0
        # the CDF of the distribution for chi=0 is 1 - (1 - x**2)**(3/2)
        # test rvs against distribution of limit chi=0
        r = stats.argus.rvs(chi, size=500, random_state=890981)
        _, p = stats.kstest(r, lambda x: 1 - (1 - x**2)**(3/2))
        assert_(p > 0.05)

    # Expected values were computed with mpmath.
    @pytest.mark.parametrize('chi, expected_mean',
                             [(1, 0.6187026683551835),
                              (10, 0.984805536783744),
                              (40, 0.9990617659702923),
                              (60, 0.9995831885165300),
                              (99, 0.9998469348663028)])
    def test_mean(self, chi, expected_mean):
        m = stats.argus.mean(chi, scale=1)
        assert_allclose(m, expected_mean, rtol=1e-13)

    # Expected values were computed with mpmath.
    @pytest.mark.parametrize('chi, expected_var, rtol',
                             [(1, 0.05215651254197807, 1e-13),
                              (10, 0.00015805472008165595, 1e-11),
                              (40, 5.877763210262901e-07, 1e-8),
                              (60, 1.1590179389611416e-07, 1e-8),
                              (99, 1.5623277006064666e-08, 1e-8)])
    def test_var(self, chi, expected_var, rtol):
        v = stats.argus.var(chi, scale=1)
        assert_allclose(v, expected_var, rtol=rtol)

    # Expected values were computed with mpmath (code: see gh-13370).
    @pytest.mark.parametrize('chi, expected, rtol',
                             [(0.9, 0.07646314974436118, 1e-14),
                              (0.5, 0.015429797891863365, 1e-14),
                              (0.1, 0.0001325825293278049, 1e-14),
                              (0.01, 1.3297677078224565e-07, 1e-15),
                              (1e-3, 1.3298072023958999e-10, 1e-14),
                              (1e-4, 1.3298075973486862e-13, 1e-14),
                              (1e-6, 1.32980760133771e-19, 1e-14),
                              (1e-9, 1.329807601338109e-28, 1e-15)])
    def test_argus_phi_small_chi(self, chi, expected, rtol):
        assert_allclose(_argus_phi(chi), expected, rtol=rtol)

    # Expected values were computed with mpmath (code: see gh-13370).
    @pytest.mark.parametrize(
        'chi, expected',
        [(0.5, (0.28414073302940573, 1.2742227939992954, 1.2381254688255896)),
         (0.2, (0.296172952995264, 1.2951290588110516, 1.1865767100877576)),
         (0.1, (0.29791447523536274, 1.29806307956989, 1.1793168289857412)),
         (0.01, (0.2984904104866452, 1.2990283628160553, 1.1769268414080531)),
         (1e-3, (0.298496172925224, 1.2990380082487925, 1.176902956021053)),
         (1e-4, (0.29849623054991836, 1.2990381047023793, 1.1769027171686324)),
         (1e-6, (0.2984962311319278, 1.2990381056765605, 1.1769027147562232)),
         (1e-9, (0.298496231131986, 1.299038105676658, 1.1769027147559818))])
    def test_pdf_small_chi(self, chi, expected):
        x = np.array([0.1, 0.5, 0.9])
        assert_allclose(stats.argus.pdf(x, chi), expected, rtol=1e-13)

    # Expected values were computed with mpmath (code: see gh-13370).
    @pytest.mark.parametrize(
        'chi, expected',
        [(0.5, (0.9857660526895221, 0.6616565930168475, 0.08796070398429937)),
         (0.2, (0.9851555052359501, 0.6514666238985464, 0.08362690023746594)),
         (0.1, (0.9850670974995661, 0.6500061310508574, 0.08302050640683846)),
         (0.01, (0.9850378582451867, 0.6495239242251358, 0.08282109244852445)),
         (1e-3, (0.9850375656906663, 0.6495191015522573, 0.08281910005231098)),
         (1e-4, (0.9850375627651049, 0.6495190533254682, 0.08281908012852317)),
         (1e-6, (0.9850375627355568, 0.6495190528383777, 0.08281907992729293)),
         (1e-9, (0.9850375627355538, 0.649519052838329, 0.0828190799272728))])
    def test_sf_small_chi(self, chi, expected):
        x = np.array([0.1, 0.5, 0.9])
        assert_allclose(stats.argus.sf(x, chi), expected, rtol=1e-14)

    # Expected values were computed with mpmath (code: see gh-13370).
    @pytest.mark.parametrize(
        'chi, expected',
        [(0.5, (0.0142339473104779, 0.3383434069831524, 0.9120392960157007)),
         (0.2, (0.014844494764049919, 0.34853337610145363, 0.916373099762534)),
         (0.1, (0.014932902500433911, 0.34999386894914264, 0.9169794935931616)),
         (0.01, (0.014962141754813293, 0.35047607577486417, 0.9171789075514756)),
         (1e-3, (0.01496243430933372, 0.35048089844774266, 0.917180899947689)),
         (1e-4, (0.014962437234895118, 0.3504809466745317, 0.9171809198714769)),
         (1e-6, (0.01496243726444329, 0.3504809471616223, 0.9171809200727071)),
         (1e-9, (0.014962437264446245, 0.350480947161671, 0.9171809200727272))])
    def test_cdf_small_chi(self, chi, expected):
        x = np.array([0.1, 0.5, 0.9])
        assert_allclose(stats.argus.cdf(x, chi), expected, rtol=1e-12)

    # Expected values were computed with mpmath (code: see gh-13370).
    @pytest.mark.parametrize(
        'chi, expected, rtol',
        [(0.5, (0.5964284712757741, 0.052890651988588604), 1e-12),
         (0.101, (0.5893490968089076, 0.053017469847275685), 1e-11),
         (0.1, (0.5893431757009437, 0.05301755449499372), 1e-13),
         (0.01, (0.5890515677940915, 0.05302167905837031), 1e-13),
         (1e-3, (0.5890486520005177, 0.053021719862088104), 1e-13),
         (1e-4, (0.5890486228426105, 0.0530217202700811), 1e-13),
         (1e-6, (0.5890486225481156, 0.05302172027420182), 1e-13),
         (1e-9, (0.5890486225480862, 0.05302172027420224), 1e-13)])
    def test_stats_small_chi(self, chi, expected, rtol):
        val = stats.argus.stats(chi, moments='mv')
        assert_allclose(val, expected, rtol=rtol)


class TestNakagami:

    def test_logpdf(self):
        # Test nakagami logpdf for an input where the PDF is smaller
        # than can be represented with 64 bit floating point.
        # The expected value of logpdf was computed with mpmath:
        #
        #   def logpdf(x, nu):
        #       x = mpmath.mpf(x)
        #       nu = mpmath.mpf(nu)
        #       return (mpmath.log(2) + nu*mpmath.log(nu) -
        #               mpmath.loggamma(nu) + (2*nu - 1)*mpmath.log(x) -
        #               nu*x**2)
        #
        nu = 2.5
        x = 25
        logp = stats.nakagami.logpdf(x, nu)
        assert_allclose(logp, -1546.9253055607549)

    def test_sf_isf(self):
        # Test nakagami sf and isf when the survival function
        # value is very small.
        # The expected value of the survival function was computed
        # with mpmath:
        #
        #   def sf(x, nu):
        #       x = mpmath.mpf(x)
        #       nu = mpmath.mpf(nu)
        #       return mpmath.gammainc(nu, nu*x*x, regularized=True)
        #
        nu = 2.5
        x0 = 5.0
        sf = stats.nakagami.sf(x0, nu)
        assert_allclose(sf, 2.736273158588307e-25, rtol=1e-13)
        # Check round trip back to x0.
        x1 = stats.nakagami.isf(sf, nu)
        assert_allclose(x1, x0, rtol=1e-13)

    @pytest.mark.parametrize("m, ref",
        [(5, -0.097341814372152),
        (0.5, 0.7257913526447274),
        (10, -0.43426184310934907)])
    def test_entropy(self, m, ref):
        # from sympy import *
        # from mpmath import mp
        # import numpy as np
        # v, x = symbols('v, x', real=True, positive=True)
        # pdf = 2 * v ** v / gamma(v) * x ** (2 * v - 1) * exp(-v * x ** 2)
        # h = simplify(simplify(integrate(-pdf * log(pdf), (x, 0, oo))))
        # entropy = lambdify(v, h, 'mpmath')
        # mp.dps = 200
        # nu = 5
        # ref = np.float64(entropy(mp.mpf(nu)))
        # print(ref)
        assert_allclose(stats.nakagami.entropy(m), ref, rtol=1.1e-14)

    @pytest.mark.parametrize("m, ref",
        [(1e-100, -5.0e+99), (1e-10, -4999999965.442979),
         (9.999e6, -7.333206478668433), (1.001e7, -7.3337562313259825),
         (1e10, -10.787134112333835), (1e100, -114.40346329705756)])
    def test_extreme_nu(self, m, ref):
        assert_allclose(stats.nakagami.entropy(m), ref)

    def test_entropy_overflow(self):
        assert np.isfinite(stats.nakagami._entropy(1e100))
        assert np.isfinite(stats.nakagami._entropy(1e-100))

    @pytest.mark.xfail(reason="Fit of nakagami not reliable, see gh-10908.")
    @pytest.mark.parametrize('nu', [1.6, 2.5, 3.9])
    @pytest.mark.parametrize('loc', [25.0, 10, 35])
    @pytest.mark.parametrize('scale', [13, 5, 20])
    def test_fit(self, nu, loc, scale):
        # Regression test for gh-13396 (21/27 cases failed previously)
        # The first tuple of the parameters' values is discussed in gh-10908
        N = 100
        samples = stats.nakagami.rvs(size=N, nu=nu, loc=loc,
                                     scale=scale, random_state=1337)
        nu_est, loc_est, scale_est = stats.nakagami.fit(samples)
        assert_allclose(nu_est, nu, rtol=0.2)
        assert_allclose(loc_est, loc, rtol=0.2)
        assert_allclose(scale_est, scale, rtol=0.2)

        def dlogl_dnu(nu, loc, scale):
            return ((-2*nu + 1) * np.sum(1/(samples - loc))
                    + 2*nu/scale**2 * np.sum(samples - loc))

        def dlogl_dloc(nu, loc, scale):
            return (N * (1 + np.log(nu) - polygamma(0, nu)) +
                    2 * np.sum(np.log((samples - loc) / scale))
                    - np.sum(((samples - loc) / scale)**2))

        def dlogl_dscale(nu, loc, scale):
            return (- 2 * N * nu / scale
                    + 2 * nu / scale ** 3 * np.sum((samples - loc) ** 2))

        assert_allclose(dlogl_dnu(nu_est, loc_est, scale_est), 0, atol=1e-3)
        assert_allclose(dlogl_dloc(nu_est, loc_est, scale_est), 0, atol=1e-3)
        assert_allclose(dlogl_dscale(nu_est, loc_est, scale_est), 0, atol=1e-3)

    @pytest.mark.parametrize('loc', [25.0, 10, 35])
    @pytest.mark.parametrize('scale', [13, 5, 20])
    def test_fit_nu(self, loc, scale):
        # For nu = 0.5, we have analytical values for
        # the MLE of the loc and the scale
        nu = 0.5
        n = 100
        samples = stats.nakagami.rvs(size=n, nu=nu, loc=loc,
                                     scale=scale, random_state=1337)
        nu_est, loc_est, scale_est = stats.nakagami.fit(samples, f0=nu)

        # Analytical values
        loc_theo = np.min(samples)
        scale_theo = np.sqrt(np.mean((samples - loc_est) ** 2))

        assert_allclose(nu_est, nu, rtol=1e-7)
        assert_allclose(loc_est, loc_theo, rtol=1e-7)
        assert_allclose(scale_est, scale_theo, rtol=1e-7)


class TestWrapCauchy:

    def test_cdf_shape_broadcasting(self):
        # Regression test for gh-13791.
        # Check that wrapcauchy.cdf broadcasts the shape parameter
        # correctly.
        c = np.array([[0.03, 0.25], [0.5, 0.75]])
        x = np.array([[1.0], [4.0]])
        p = stats.wrapcauchy.cdf(x, c)
        assert p.shape == (2, 2)
        scalar_values = [stats.wrapcauchy.cdf(x1, c1)
                         for (x1, c1) in np.nditer((x, c))]
        assert_allclose(p.ravel(), scalar_values, rtol=1e-13)

    def test_cdf_center(self):
        p = stats.wrapcauchy.cdf(np.pi, 0.03)
        assert_allclose(p, 0.5, rtol=1e-14)

    def test_cdf(self):
        x1 = 1.0  # less than pi
        x2 = 4.0  # greater than pi
        c = 0.75
        p = stats.wrapcauchy.cdf([x1, x2], c)
        cr = (1 + c)/(1 - c)
        assert_allclose(p[0], np.arctan(cr*np.tan(x1/2))/np.pi)
        assert_allclose(p[1], 1 - np.arctan(cr*np.tan(np.pi - x2/2))/np.pi)


def test_rvs_no_size_error():
    # _rvs methods must have parameter `size`; see gh-11394
    class rvs_no_size_gen(stats.rv_continuous):
        def _rvs(self):
            return 1

    rvs_no_size = rvs_no_size_gen(name='rvs_no_size')

    with assert_raises(TypeError, match=r"_rvs\(\) got (an|\d) unexpected"):
        rvs_no_size.rvs()


@pytest.mark.parametrize('distname, args', invdistdiscrete + invdistcont)
def test_support_gh13294_regression(distname, args):
    if distname in skip_test_support_gh13294_regression:
        pytest.skip(f"skipping test for the support method for "
                    f"distribution {distname}.")
    dist = getattr(stats, distname)
    # test support method with invalid arguents
    if isinstance(dist, stats.rv_continuous):
        # test with valid scale
        if len(args) != 0:
            a0, b0 = dist.support(*args)
            assert_equal(a0, np.nan)
            assert_equal(b0, np.nan)
        # test with invalid scale
        # For some distributions, that take no parameters,
        # the case of only invalid scale occurs and hence,
        # it is implicitly tested in this test case.
        loc1, scale1 = 0, -1
        a1, b1 = dist.support(*args, loc1, scale1)
        assert_equal(a1, np.nan)
        assert_equal(b1, np.nan)
    else:
        a, b = dist.support(*args)
        assert_equal(a, np.nan)
        assert_equal(b, np.nan)


def test_support_broadcasting_gh13294_regression():
    a0, b0 = stats.norm.support([0, 0, 0, 1], [1, 1, 1, -1])
    ex_a0 = np.array([-np.inf, -np.inf, -np.inf, np.nan])
    ex_b0 = np.array([np.inf, np.inf, np.inf, np.nan])
    assert_equal(a0, ex_a0)
    assert_equal(b0, ex_b0)
    assert a0.shape == ex_a0.shape
    assert b0.shape == ex_b0.shape

    a1, b1 = stats.norm.support([], [])
    ex_a1, ex_b1 = np.array([]), np.array([])
    assert_equal(a1, ex_a1)
    assert_equal(b1, ex_b1)
    assert a1.shape == ex_a1.shape
    assert b1.shape == ex_b1.shape

    a2, b2 = stats.norm.support([0, 0, 0, 1], [-1])
    ex_a2 = np.array(4*[np.nan])
    ex_b2 = np.array(4*[np.nan])
    assert_equal(a2, ex_a2)
    assert_equal(b2, ex_b2)
    assert a2.shape == ex_a2.shape
    assert b2.shape == ex_b2.shape


def test_stats_broadcasting_gh14953_regression():
    # test case in gh14953
    loc = [0., 0.]
    scale = [[1.], [2.], [3.]]
    assert_equal(stats.norm.var(loc, scale), [[1., 1.], [4., 4.], [9., 9.]])
    # test some edge cases
    loc = np.empty((0, ))
    scale = np.empty((1, 0))
    assert stats.norm.var(loc, scale).shape == (1, 0)


# Check a few values of the cosine distribution's cdf, sf, ppf and
# isf methods.  Expected values were computed with mpmath.

@pytest.mark.parametrize('x, expected',
                         [(-3.14159, 4.956444476505336e-19),
                          (3.14, 0.9999999998928399)])
def test_cosine_cdf_sf(x, expected):
    assert_allclose(stats.cosine.cdf(x), expected)
    assert_allclose(stats.cosine.sf(-x), expected)


@pytest.mark.parametrize('p, expected',
                         [(1e-6, -3.1080612413765905),
                          (1e-17, -3.141585429601399),
                          (0.975, 2.1447547020964923)])
def test_cosine_ppf_isf(p, expected):
    assert_allclose(stats.cosine.ppf(p), expected)
    assert_allclose(stats.cosine.isf(p), -expected)


def test_cosine_logpdf_endpoints():
    logp = stats.cosine.logpdf([-np.pi, np.pi])
    # reference value calculated using mpmath assuming `np.cos(-1)` is four
    # floating point numbers too high. See gh-18382.
    assert_array_less(logp, -37.18838327496655)


def test_distr_params_lists():
    # distribution objects are extra distributions added in
    # test_discrete_basic. All other distributions are strings (names)
    # and so we only choose those to compare whether both lists match.
    discrete_distnames = {name for name, _ in distdiscrete
                          if isinstance(name, str)}
    invdiscrete_distnames = {name for name, _ in invdistdiscrete}
    assert discrete_distnames == invdiscrete_distnames

    cont_distnames = {name for name, _ in distcont}
    invcont_distnames = {name for name, _ in invdistcont}
    assert cont_distnames == invcont_distnames


def test_moment_order_4():
    # gh-13655 reported that if a distribution has a `_stats` method that
    # accepts the `moments` parameter, then if the distribution's `moment`
    # method is called with `order=4`, the faster/more accurate`_stats` gets
    # called, but the results aren't used, and the generic `_munp` method is
    # called to calculate the moment anyway. This tests that the issue has
    # been fixed.
    # stats.skewnorm._stats accepts the `moments` keyword
    stats.skewnorm._stats(a=0, moments='k')  # no failure = has `moments`
    # When `moment` is called, `_stats` is used, so the moment is very accurate
    # (exactly equal to Pearson's kurtosis of the normal distribution, 3)
    assert stats.skewnorm.moment(order=4, a=0) == 3.0
    # At the time of gh-13655, skewnorm._munp() used the generic method
    # to compute its result, which was inefficient and not very accurate.
    # At that time, the following assertion would fail.  skewnorm._munp()
    # has since been made more accurate and efficient, so now this test
    # is expected to pass.
    assert stats.skewnorm._munp(4, 0) == 3.0


class TestRelativisticBW:
    @pytest.fixture
    def ROOT_pdf_sample_data(self):
        """Sample data points for pdf computed with CERN's ROOT

        See - https://root.cern/

        Uses ROOT.TMath.BreitWignerRelativistic, available in ROOT
        versions 6.27+

        pdf calculated for Z0 Boson, W Boson, and Higgs Boson for
        x in `np.linspace(0, 200, 401)`.
        """
        data = np.load(
            Path(__file__).parent /
            'data/rel_breitwigner_pdf_sample_data_ROOT.npy'
        )
        data = np.core.records.fromarrays(data.T, names='x,pdf,rho,gamma')
        return data

    @pytest.mark.parametrize(
        "rho,gamma,rtol", [
            (36.545206797050334, 2.4952, 5e-14),  # Z0 Boson
            (38.55107913669065, 2.085, 1e-14),  # W Boson
            (96292.3076923077, 0.0013, 5e-13),  # Higgs Boson
        ]
    )
    def test_pdf_against_ROOT(self, ROOT_pdf_sample_data, rho, gamma, rtol):
        data = ROOT_pdf_sample_data[
            (ROOT_pdf_sample_data['rho'] == rho)
            & (ROOT_pdf_sample_data['gamma'] == gamma)
        ]
        x, pdf = data['x'], data['pdf']
        assert_allclose(
            pdf, stats.rel_breitwigner.pdf(x, rho, scale=gamma), rtol=rtol
        )

    @pytest.mark.parametrize("rho, Gamma, rtol", [
              (36.545206797050334, 2.4952, 5e-13),  # Z0 Boson
              (38.55107913669065, 2.085, 5e-13),  # W Boson
              (96292.3076923077, 0.0013, 5e-10),  # Higgs Boson
          ]
      )
    def test_pdf_against_simple_implementation(self, rho, Gamma, rtol):
        # reference implementation straight from formulas on Wikipedia [1]
        def pdf(E, M, Gamma):
            gamma = np.sqrt(M**2 * (M**2 + Gamma**2))
            k = (2 * np.sqrt(2) * M * Gamma * gamma
                 / (np.pi * np.sqrt(M**2 + gamma)))
            return k / ((E**2 - M**2)**2 + M**2*Gamma**2)
        # get reasonable values at which to evaluate the CDF
        p = np.linspace(0.05, 0.95, 10)
        x = stats.rel_breitwigner.ppf(p, rho, scale=Gamma)
        res = stats.rel_breitwigner.pdf(x, rho, scale=Gamma)
        ref = pdf(x, rho*Gamma, Gamma)
        assert_allclose(res, ref, rtol=rtol)

    @pytest.mark.parametrize(
        "rho,gamma", [
            pytest.param(
                36.545206797050334, 2.4952, marks=pytest.mark.slow
            ),  # Z0 Boson
            pytest.param(
                38.55107913669065, 2.085, marks=pytest.mark.xslow
            ),  # W Boson
            pytest.param(
                96292.3076923077, 0.0013, marks=pytest.mark.xslow
            ),  # Higgs Boson
        ]
    )
    def test_fit_floc(self, rho, gamma):
        """Tests fit for cases where floc is set.

        `rel_breitwigner` has special handling for these cases.
        """
        seed = 6936804688480013683
        rng = np.random.default_rng(seed)
        data = stats.rel_breitwigner.rvs(
            rho, scale=gamma, size=1000, random_state=rng
        )
        fit = stats.rel_breitwigner.fit(data, floc=0)
        assert_allclose((fit[0], fit[2]), (rho, gamma), rtol=2e-1)
        assert fit[1] == 0
        # Check again with fscale set.
        fit = stats.rel_breitwigner.fit(data, floc=0, fscale=gamma)
        assert_allclose(fit[0], rho, rtol=1e-2)
        assert (fit[1], fit[2]) == (0, gamma)


class TestJohnsonSU:
    @pytest.mark.parametrize("case", [  # a, b, loc, scale, m1, m2, g1, g2
            (-0.01, 1.1, 0.02, 0.0001, 0.02000137427557091,
             2.1112742956578063e-08, 0.05989781342460999, 20.36324408592951-3),
            (2.554395574161155, 2.2482281679651965, 0, 1, -1.54215386737391,
             0.7629882028469993, -1.256656139406788, 6.303058419339775-3)])
    def test_moment_gh18071(self, case):
        # gh-18071 reported an IntegrationWarning emitted by johnsonsu.stats
        # Check that the warning is no longer emitted and that the values
        # are accurate compared against results from Mathematica.
        # Reference values from Mathematica, e.g.
        # Mean[JohnsonDistribution["SU",-0.01, 1.1, 0.02, 0.0001]]
        res = stats.johnsonsu.stats(*case[:4], moments='mvsk')
        assert_allclose(res, case[4:], rtol=1e-14)


class TestTruncPareto:
    def test_pdf(self):
        # PDF is that of the truncated pareto distribution
        b, c = 1.8, 5.3
        x = np.linspace(1.8, 5.3)
        res = stats.truncpareto(b, c).pdf(x)
        ref = stats.pareto(b).pdf(x) / stats.pareto(b).cdf(c)
        assert_allclose(res, ref)

    @pytest.mark.parametrize('fix_loc', [True, False])
    @pytest.mark.parametrize('fix_scale', [True, False])
    @pytest.mark.parametrize('fix_b', [True, False])
    @pytest.mark.parametrize('fix_c', [True, False])
    def test_fit(self, fix_loc, fix_scale, fix_b, fix_c):

        rng = np.random.default_rng(6747363148258237171)
        b, c, loc, scale = 1.8, 5.3, 1, 2.5
        dist = stats.truncpareto(b, c, loc=loc, scale=scale)
        data = dist.rvs(size=500, random_state=rng)

        kwds = {}
        if fix_loc:
            kwds['floc'] = loc
        if fix_scale:
            kwds['fscale'] = scale
        if fix_b:
            kwds['f0'] = b
        if fix_c:
            kwds['f1'] = c

        if fix_loc and fix_scale and fix_b and fix_c:
            message = "All parameters fixed. There is nothing to optimize."
            with pytest.raises(RuntimeError, match=message):
                stats.truncpareto.fit(data, **kwds)
        else:
            _assert_less_or_close_loglike(stats.truncpareto, data, **kwds)
