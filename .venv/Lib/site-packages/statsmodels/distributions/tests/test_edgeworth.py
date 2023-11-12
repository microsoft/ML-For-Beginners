
import warnings

import numpy as np
from numpy.testing import (assert_equal, assert_raises,
                           assert_allclose)
import numpy.testing as npt

from scipy.special import gamma, factorial, factorial2
import scipy.stats as stats

from statsmodels.distributions.edgeworth import (_faa_di_bruno_partitions,
        cumulant_from_moments, ExpandedNormal)

class TestFaaDiBruno:
    def test_neg_arg(self):
        assert_raises(ValueError, _faa_di_bruno_partitions, -1)
        assert_raises(ValueError, _faa_di_bruno_partitions, 0)

    def test_small_vals(self):
        for n in range(1, 5):
            for ks in _faa_di_bruno_partitions(n):
                lhs = sum(m * k for (m, k) in ks)
                assert_equal(lhs, n)


def _norm_moment(n):
    # moments of N(0, 1)
    return (1 - n % 2) * factorial2(n - 1)

def _norm_cumulant(n):
    # cumulants of N(0, 1)
    try:
        return {1: 0, 2: 1}[n]
    except KeyError:
        return 0

def _chi2_moment(n, df):
    # (raw) moments of \chi^2(df)
    return (2**n) * gamma(n + df/2.) / gamma(df/2.)

def _chi2_cumulant(n, df):
    assert n > 0
    return 2**(n-1) * factorial(n - 1) * df


class TestCumulants:
    def test_badvalues(self):
        assert_raises(ValueError, cumulant_from_moments, [1, 2, 3], 0)
        assert_raises(ValueError, cumulant_from_moments, [1, 2, 3], 4)

    def test_norm(self):
        N = 4
        momt = [_norm_moment(j+1) for j in range(N)]
        for n in range(1, N+1):
            kappa = cumulant_from_moments(momt, n)
            assert_allclose(kappa, _norm_cumulant(n),
                    atol=1e-12)

    def test_chi2(self):
        N = 4
        df = 8
        momt = [_chi2_moment(j+1, df) for j in range(N)]
        for n in range(1, N+1):
            kappa = cumulant_from_moments(momt, n)
            assert_allclose(kappa, _chi2_cumulant(n, df))


class TestExpandedNormal:
    def test_too_few_cumulants(self):
        assert_raises(ValueError, ExpandedNormal, [1])

    def test_coefficients(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            # 3rd order in n**(1/2)
            ne3 = ExpandedNormal([0., 1., 1.])
            assert_allclose(ne3._coef, [1., 0., 0., 1./6])

            # 4th order in n**(1/2)
            ne4 = ExpandedNormal([0., 1., 1., 1.])
            assert_allclose(ne4._coef, [1., 0., 0., 1./6, 1./24, 0., 1./72])

            # 5th order
            ne5 = ExpandedNormal([0., 1., 1., 1., 1.])
            assert_allclose(ne5._coef, [1., 0., 0., 1./6, 1./24, 1./120,
                    1./72, 1./144, 0., 1./1296])

            # adding trailing zeroes increases the order
            ne33 = ExpandedNormal([0., 1., 1., 0.])
            assert_allclose(ne33._coef, [1., 0., 0., 1./6, 0., 0., 1./72])

    def test_normal(self):
        # with two cumulants, it's just a gaussian
        ne2 = ExpandedNormal([3, 4])
        x = np.linspace(-2., 2., 100)
        assert_allclose(ne2.pdf(x), stats.norm.pdf(x, loc=3, scale=2))

    def test_chi2_moments(self):
        # construct the expansion for \chi^2
        N, df = 6, 15
        cum = [_chi2_cumulant(n+1, df) for n in range(N)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ne = ExpandedNormal(cum, name='edgw_chi2')

        # compare the moments
        assert_allclose([_chi2_moment(n, df) for n in range(N)],
                        [ne.moment(n) for n in range(N)])

        # compare the pdf [fragile!]
        # this one is actually not a very good test: there is, strictly
        # speaking, no guarantee that the pdfs match point-by-point
        # m, s = df, np.sqrt(df)
        # x = np.linspace(m - s, m + s, 10)
        # assert_allclose(ne.pdf(x), stats.chi2.pdf(x, df),
        #        atol=1e-4, rtol=1e-5)

        # pdf-cdf roundtrip
        check_pdf(ne, arg=(), msg='')

        # cdf-ppf roundtrip
        check_cdf_ppf(ne, arg=(), msg='')

        # cdf + sf == 1
        check_cdf_sf(ne, arg=(), msg='')

        # generate rvs & run a KS test
        np.random.seed(765456)
        rvs = ne.rvs(size=500)
        check_distribution_rvs(ne, args=(), alpha=0.01, rvs=rvs)

    def test_pdf_no_roots(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            ne = ExpandedNormal([0, 1])
            ne = ExpandedNormal([0, 1, 0.1, 0.1])

    def test_pdf_has_roots(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            assert_raises(RuntimeWarning, ExpandedNormal, [0, 1, 101])


## stolen verbatim from scipy/stats/tests/test_continuous_extra.py
DECIMAL = 8

def check_pdf(distfn, arg, msg):
    # compares pdf at median with numerical derivative of cdf
    median = distfn.ppf(0.5, *arg)
    eps = 1e-6
    pdfv = distfn.pdf(median, *arg)
    if (pdfv < 1e-4) or (pdfv > 1e4):
        # avoid checking a case where pdf is close to zero
        # or huge (singularity)
        median = median + 0.1
        pdfv = distfn.pdf(median, *arg)
    cdfdiff = (distfn.cdf(median + eps, *arg) -
               distfn.cdf(median - eps, *arg))/eps/2.0
    # replace with better diff and better test (more points),
    # actually, this works pretty well
    npt.assert_almost_equal(pdfv, cdfdiff,
                decimal=DECIMAL, err_msg=msg + ' - cdf-pdf relationship')


def check_cdf_ppf(distfn, arg, msg):
    values = [0.001, 0.5, 0.999]
    npt.assert_almost_equal(distfn.cdf(distfn.ppf(values, *arg), *arg),
            values, decimal=DECIMAL, err_msg=msg + ' - cdf-ppf roundtrip')


def check_cdf_sf(distfn, arg, msg):
    values = [0.001, 0.5, 0.999]
    npt.assert_almost_equal(distfn.cdf(values, *arg),
            1. - distfn.sf(values, *arg),
            decimal=DECIMAL, err_msg=msg +' - sf+cdf == 1')


def check_distribution_rvs(distfn, args, alpha, rvs):
    ## signature changed to avoid calling a distribution by name
    # test from scipy.stats.tests
    # this version reuses existing random variables
    D,pval = stats.kstest(rvs, distfn.cdf, args=args, N=1000)
    if (pval < alpha):
        D,pval = stats.kstest(distfn.rvs, distfn.cdf, args=args, N=1000)
        npt.assert_(pval > alpha, "D = " + str(D) + "; pval = " + str(pval) +
               "; alpha = " + str(alpha) + "\nargs = " + str(args))
