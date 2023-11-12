

import numpy as np
from numpy.testing import assert_allclose, assert_equal

from scipy import stats
from scipy.stats import poisson, nbinom

from statsmodels.tools.tools import Bunch

from statsmodels.distributions.discrete import (
    genpoisson_p,
    truncatedpoisson,
    truncatednegbin,
    zipoisson,
    zinegbin,
    zigenpoisson,
    DiscretizedCount,
    DiscretizedModel
    )


class TestGenpoisson_p:
    # Test Generalized Poisson Destribution

    def test_pmf_p1(self):
        poisson_pmf = poisson.pmf(1, 1)
        genpoisson_pmf = genpoisson_p.pmf(1, 1, 0, 1)
        assert_allclose(poisson_pmf, genpoisson_pmf, rtol=1e-15)

    def test_pmf_p2(self):
        poisson_pmf = poisson.pmf(2, 2)
        genpoisson_pmf = genpoisson_p.pmf(2, 2, 0, 2)
        assert_allclose(poisson_pmf, genpoisson_pmf, rtol=1e-15)

    def test_pmf_p5(self):
        poisson_pmf = poisson.pmf(10, 2)
        genpoisson_pmf_5 = genpoisson_p.pmf(10, 2, 1e-25, 5)
        assert_allclose(poisson_pmf, genpoisson_pmf_5, rtol=1e-12)

    def test_logpmf_p1(self):
        poisson_pmf = poisson.logpmf(5, 2)
        genpoisson_pmf = genpoisson_p.logpmf(5, 2, 0, 1)
        assert_allclose(poisson_pmf, genpoisson_pmf, rtol=1e-15)

    def test_logpmf_p2(self):
        poisson_pmf = poisson.logpmf(6, 1)
        genpoisson_pmf = genpoisson_p.logpmf(6, 1, 0, 2)
        assert_allclose(poisson_pmf, genpoisson_pmf, rtol=1e-15)


class TestTruncatedPoisson:
    """
    Test Truncated Poisson distribution
    """
    def test_pmf_zero(self):
        poisson_pmf = poisson.pmf(2, 2) / poisson.sf(0, 2)
        tpoisson_pmf = truncatedpoisson.pmf(2, 2, 0)
        assert_allclose(poisson_pmf, tpoisson_pmf, rtol=1e-7)

    def test_logpmf_zero(self):
        poisson_logpmf = poisson.logpmf(2, 2) - np.log(poisson.sf(0, 2))
        tpoisson_logpmf = truncatedpoisson.logpmf(2, 2, 0)
        assert_allclose(poisson_logpmf, tpoisson_logpmf, rtol=1e-7)

    def test_pmf(self):
        poisson_pmf = poisson.pmf(4, 6) / (1 - poisson.cdf(2, 6))
        tpoisson_pmf = truncatedpoisson.pmf(4, 6, 2)
        assert_allclose(poisson_pmf, tpoisson_pmf, rtol=1e-7)

    def test_logpmf(self):
        poisson_logpmf = poisson.logpmf(4, 6) - np.log(poisson.sf(2, 6))
        tpoisson_logpmf = truncatedpoisson.logpmf(4, 6, 2)
        assert_allclose(poisson_logpmf, tpoisson_logpmf, rtol=1e-7)


class TestZIPoisson:

    def test_pmf_zero(self):
        poisson_pmf = poisson.pmf(3, 2)
        zipoisson_pmf = zipoisson.pmf(3, 2, 0)
        assert_allclose(poisson_pmf, zipoisson_pmf, rtol=1e-12)

    def test_logpmf_zero(self):
        poisson_logpmf = poisson.logpmf(5, 1)
        zipoisson_logpmf = zipoisson.logpmf(5, 1, 0)
        assert_allclose(poisson_logpmf, zipoisson_logpmf, rtol=1e-12)

    def test_pmf(self):
        poisson_pmf = poisson.pmf(2, 2)
        zipoisson_pmf = zipoisson.pmf(2, 2, 0.1)
        assert_allclose(poisson_pmf, zipoisson_pmf, rtol=5e-2, atol=5e-2)

    def test_logpmf(self):
        poisson_logpmf = poisson.logpmf(7, 3)
        zipoisson_logpmf = zipoisson.logpmf(7, 3, 0.1)
        assert_allclose(poisson_logpmf, zipoisson_logpmf, rtol=5e-2, atol=5e-2)

    def test_cdf_zero(self):
        poisson_cdf = poisson.cdf(3, 2)
        zipoisson_cdf = zipoisson.cdf(3, 2, 0)
        assert_allclose(poisson_cdf, zipoisson_cdf, rtol=1e-12)

    def test_ppf_zero(self):
        poisson_ppf = poisson.ppf(5, 1)
        zipoisson_ppf = zipoisson.ppf(5, 1, 0)
        assert_allclose(poisson_ppf, zipoisson_ppf, rtol=1e-12)

    def test_mean_var(self):
        poisson_mean, poisson_var = poisson.mean(12), poisson.var(12)
        zipoisson_mean = zipoisson.mean(12, 0)
        zipoisson_var = zipoisson.var(12, 0)
        assert_allclose(poisson_mean, zipoisson_mean, rtol=1e-10)
        assert_allclose(poisson_var, zipoisson_var, rtol=1e-10)

        m = np.array([1, 5, 10])
        poisson_mean, poisson_var = poisson.mean(m), poisson.var(m)
        zipoisson_mean = zipoisson.mean(m, 0)
        zipoisson_var = zipoisson.var(m, 0.0)
        assert_allclose(poisson_mean, zipoisson_mean, rtol=1e-10)
        assert_allclose(poisson_var, zipoisson_var, rtol=1e-10)

    def test_moments(self):
        poisson_m1, poisson_m2 = poisson.moment(1, 12), poisson.moment(2, 12)
        zip_m0 = zipoisson.moment(0, 12, 0)
        zip_m1 = zipoisson.moment(1, 12, 0)
        zip_m2 = zipoisson.moment(2, 12, 0)
        assert_allclose(1, zip_m0, rtol=1e-10)
        assert_allclose(poisson_m1, zip_m1, rtol=1e-10)
        assert_allclose(poisson_m2, zip_m2, rtol=1e-10)


class TestZIGeneralizedPoisson:
    def test_pmf_zero(self):
        gp_pmf = genpoisson_p.pmf(3, 2, 1, 1)
        zigp_pmf = zigenpoisson.pmf(3, 2, 1, 1, 0)
        assert_allclose(gp_pmf, zigp_pmf, rtol=1e-12)

    def test_logpmf_zero(self):
        gp_logpmf = genpoisson_p.logpmf(7, 3, 1, 1)
        zigp_logpmf = zigenpoisson.logpmf(7, 3, 1, 1, 0)
        assert_allclose(gp_logpmf, zigp_logpmf, rtol=1e-12)

    def test_pmf(self):
        gp_pmf = genpoisson_p.pmf(3, 2, 2, 2)
        zigp_pmf = zigenpoisson.pmf(3, 2, 2, 2, 0.1)
        assert_allclose(gp_pmf, zigp_pmf, rtol=5e-2, atol=5e-2)

    def test_logpmf(self):
        gp_logpmf = genpoisson_p.logpmf(2, 3, 0, 2)
        zigp_logpmf = zigenpoisson.logpmf(2, 3, 0, 2, 0.1)
        assert_allclose(gp_logpmf, zigp_logpmf, rtol=5e-2, atol=5e-2)

    def test_mean_var(self):

        # compare with Poisson special case
        m = np.array([1, 5, 10])
        poisson_mean, poisson_var = poisson.mean(m), poisson.var(m)
        zigenpoisson_mean = zigenpoisson.mean(m, 0, 1, 0)
        zigenpoisson_var = zigenpoisson.var(m, 0.0, 1, 0)
        assert_allclose(poisson_mean, zigenpoisson_mean, rtol=1e-10)
        assert_allclose(poisson_var, zigenpoisson_var, rtol=1e-10)


class TestZiNBP:

    def test_pmf_p2(self):
        n, p = zinegbin.convert_params(30, 0.1, 2)
        nb_pmf = nbinom.pmf(100, n, p)
        tnb_pmf = zinegbin.pmf(100, 30, 0.1, 2, 0.01)
        assert_allclose(nb_pmf, tnb_pmf, rtol=1e-5, atol=1e-5)

    def test_logpmf_p2(self):
        n, p = zinegbin.convert_params(10, 1, 2)
        nb_logpmf = nbinom.logpmf(200, n, p)
        tnb_logpmf = zinegbin.logpmf(200, 10, 1, 2, 0.01)
        assert_allclose(nb_logpmf, tnb_logpmf, rtol=1e-2, atol=1e-2)

    def test_cdf_p2(self):
        n, p = zinegbin.convert_params(30, 0.1, 2)
        nbinom_cdf = nbinom.cdf(10, n, p)
        zinbinom_cdf = zinegbin.cdf(10, 30, 0.1, 2, 0)
        assert_allclose(nbinom_cdf, zinbinom_cdf, rtol=1e-12, atol=1e-12)

    def test_ppf_p2(self):
        n, p = zinegbin.convert_params(100, 1, 2)
        nbinom_ppf = nbinom.ppf(0.27, n, p)
        zinbinom_ppf = zinegbin.ppf(0.27, 100, 1, 2, 0)
        assert_allclose(nbinom_ppf, zinbinom_ppf, rtol=1e-12, atol=1e-12)

    def test_mran_var_p2(self):
        n, p = zinegbin.convert_params(7, 1, 2)
        nbinom_mean, nbinom_var = nbinom.mean(n, p), nbinom.var(n, p)
        zinb_mean = zinegbin.mean(7, 1, 2, 0)
        zinb_var = zinegbin.var(7, 1, 2, 0)
        assert_allclose(nbinom_mean, zinb_mean, rtol=1e-10)
        assert_allclose(nbinom_var, zinb_var, rtol=1e-10)

    def test_moments_p2(self):
        n, p = zinegbin.convert_params(7, 1, 2)
        nb_m1, nb_m2 = nbinom.moment(1, n, p), nbinom.moment(2, n, p)
        zinb_m0 = zinegbin.moment(0, 7, 1, 2, 0)
        zinb_m1 = zinegbin.moment(1, 7, 1, 2, 0)
        zinb_m2 = zinegbin.moment(2, 7, 1, 2, 0)
        assert_allclose(1, zinb_m0, rtol=1e-10)
        assert_allclose(nb_m1, zinb_m1, rtol=1e-10)
        assert_allclose(nb_m2, zinb_m2, rtol=1e-10)

    def test_pmf(self):
        n, p = zinegbin.convert_params(1, 0.9, 1)
        nb_logpmf = nbinom.pmf(2, n, p)
        tnb_pmf = zinegbin.pmf(2, 1, 0.9, 2, 0.5)
        assert_allclose(nb_logpmf, tnb_pmf * 2, rtol=1e-7)

    def test_logpmf(self):
        n, p = zinegbin.convert_params(5, 1, 1)
        nb_logpmf = nbinom.logpmf(2, n, p)
        tnb_logpmf = zinegbin.logpmf(2, 5, 1, 1, 0.005)
        assert_allclose(nb_logpmf, tnb_logpmf, rtol=1e-2, atol=1e-2)

    def test_cdf(self):
        n, p = zinegbin.convert_params(1, 0.9, 1)
        nbinom_cdf = nbinom.cdf(2, n, p)
        zinbinom_cdf = zinegbin.cdf(2, 1, 0.9, 2, 0)
        assert_allclose(nbinom_cdf, zinbinom_cdf, rtol=1e-12, atol=1e-12)

    def test_ppf(self):
        n, p = zinegbin.convert_params(5, 1, 1)
        nbinom_ppf = nbinom.ppf(0.71, n, p)
        zinbinom_ppf = zinegbin.ppf(0.71, 5, 1, 1, 0)
        assert_allclose(nbinom_ppf, zinbinom_ppf, rtol=1e-12, atol=1e-12)

    def test_convert(self):
        n, p = zinegbin.convert_params(25, 0.85, 2)
        n_true, p_true = 1.1764705882352942, 0.04494382022471911
        assert_allclose(n, n_true, rtol=1e-12, atol=1e-12)
        assert_allclose(p, p_true, rtol=1e-12, atol=1e-12)

        n, p = zinegbin.convert_params(7, 0.17, 1)
        n_true, p_true = 41.17647058823529, 0.8547008547008547
        assert_allclose(n, n_true, rtol=1e-12, atol=1e-12)
        assert_allclose(p, p_true, rtol=1e-12, atol=1e-12)

    def test_mean_var(self):
        for m in [9, np.array([1, 5, 10])]:
            n, p = zinegbin.convert_params(m, 1, 1)
            nbinom_mean, nbinom_var = nbinom.mean(n, p), nbinom.var(n, p)
            zinb_mean = zinegbin.mean(m, 1, 1, 0)
            zinb_var = zinegbin.var(m, 1, 1, 0)
            assert_allclose(nbinom_mean, zinb_mean, rtol=1e-10)
            assert_allclose(nbinom_var, zinb_var, rtol=1e-10)

    def test_moments(self):
        n, p = zinegbin.convert_params(9, 1, 1)
        nb_m1, nb_m2 = nbinom.moment(1, n, p), nbinom.moment(2, n, p)
        zinb_m0 = zinegbin.moment(0, 9, 1, 1, 0)
        zinb_m1 = zinegbin.moment(1, 9, 1, 1, 0)
        zinb_m2 = zinegbin.moment(2, 9, 1, 1, 0)
        assert_allclose(1, zinb_m0, rtol=1e-10)
        assert_allclose(nb_m1, zinb_m1, rtol=1e-10)
        assert_allclose(nb_m2, zinb_m2, rtol=1e-10)


class CheckDiscretized():

    def convert_params(self, params):
        args = params.tolist()
        args.insert(-1, 0)
        return args

    def test_basic(self):
        d_offset = self.d_offset
        ddistr = self.ddistr
        paramg = self.paramg
        paramd = self.paramd
        shapes = self.shapes
        start_params = self.start_params

        np.random.seed(987146)

        dp = DiscretizedCount(ddistr, d_offset)
        assert dp.shapes == shapes
        xi = np.arange(5)
        p = dp._pmf(xi, *paramd)

        cdf1 = ddistr.cdf(xi, *paramg)
        p1 = np.diff(cdf1)
        assert_allclose(p[: len(p1)], p1, rtol=1e-13)
        cdf = dp._cdf(xi, *paramd)
        assert_allclose(cdf[: len(cdf1) - 1], cdf1[1:], rtol=1e-13)

        # check that scipy dispatch methods work
        p2 = dp.pmf(xi, *paramd)
        assert_allclose(p2, p, rtol=1e-13)
        cdf2 = dp.cdf(xi, *paramd)
        assert_allclose(cdf2, cdf, rtol=1e-13)
        sf = dp.sf(xi, *paramd)
        assert_allclose(sf, 1 - cdf, rtol=1e-13)

        nobs = 2000

        xx = dp.rvs(*paramd, size=nobs)  # , random_state=987146)
        # check that we go a non-trivial rvs
        assert len(xx) == nobs
        assert xx.var() > 0.001
        mod = DiscretizedModel(xx, distr=dp)
        res = mod.fit(start_params=start_params)
        p = mod.predict(res.params, which="probs")
        args = self.convert_params(res.params)
        p1 = -np.diff(ddistr.sf(np.arange(21), *args))
        assert_allclose(p, p1, rtol=1e-13)

        # using cdf limits precision to computation around 1
        p1 = np.diff(ddistr.cdf(np.arange(21), *args))
        assert_allclose(p, p1, rtol=1e-13, atol=1e-15)
        freq = np.bincount(xx.astype(int))
        # truncate at last observed
        k = len(freq)
        if k > 10:
            # reduce low count bins for heavy tailed distributions
            k = 10
            freq[k - 1] += freq[k:].sum()
            freq = freq[:k]
        p = mod.predict(res.params, which="probs", k_max=k)
        p[k - 1] += 1 - p[:k].sum()
        tchi2 = stats.chisquare(freq, p[:k] * nobs)
        assert tchi2.pvalue > 0.01

        # estimated distribution methods rvs, ppf
        # frozen distribution with estimated parameters
        # Todo results method
        dfr = mod.get_distr(res.params)
        nobs_rvs = 500
        rvs = dfr.rvs(size=nobs_rvs)
        freq = np.bincount(rvs)
        p = mod.predict(res.params, which="probs", k_max=nobs_rvs)
        k = len(freq)
        p[k - 1] += 1 - p[:k].sum()
        tchi2 = stats.chisquare(freq, p[:k] * nobs_rvs)
        assert tchi2.pvalue > 0.01

        # round trip cdf-ppf
        q = dfr.ppf(dfr.cdf(np.arange(-1, 5) + 1e-6))
        q1 = np.array([-1.,  1.,  2.,  3.,  4.,  5.])
        assert_equal(q, q1)
        p = np.maximum(dfr.cdf(np.arange(-1, 5)) - 1e-6, 0)
        q = dfr.ppf(p)
        q1 = np.arange(-1, 5)
        assert_equal(q, q1)
        q = dfr.ppf(dfr.cdf(np.arange(5)))
        q1 = np.arange(0, 5)
        assert_equal(q, q1)
        q = dfr.isf(1 - dfr.cdf(np.arange(-1, 5) + 1e-6))
        q1 = np.array([-1.,  1.,  2.,  3.,  4.,  5.])
        assert_equal(q, q1)


class TestDiscretizedGamma(CheckDiscretized):

    @classmethod
    def setup_class(cls):
        cls.d_offset = 0
        cls.ddistr = stats.gamma
        cls.paramg = (5, 0, 0.5)  # include constant so we can use args
        cls.paramd = (5, 0.5)
        cls.shapes = "a, s"

        cls.start_params = (1, 0.5)


class TestDiscretizedExponential(CheckDiscretized):

    @classmethod
    def setup_class(cls):
        cls.d_offset = 0
        cls.ddistr = stats.expon
        cls.paramg = (0, 5)  # include constant so we can use args
        cls.paramd = (5,)
        cls.shapes = "s"

        cls.start_params = (0.5)


class TestDiscretizedLomax(CheckDiscretized):

    @classmethod
    def setup_class(cls):
        cls.d_offset = 0
        cls.ddistr = stats.lomax  # instead of pareto to avoid p(y=0) = 0
        cls.paramg = (2, 0, 1.5)  # include constant so we can use args
        cls.paramd = (2, 1.5,)
        cls.shapes = "c, s"

        cls.start_params = (0.5, 0.5)


class TestDiscretizedBurr12(CheckDiscretized):

    @classmethod
    def setup_class(cls):
        cls.d_offset = 0
        cls.ddistr = stats.burr12  # should be lomax as special case of burr12
        cls.paramg = (2, 1, 0, 1.5)
        cls.paramd = (2, 1, 1.5)
        cls.shapes = "c, d, s"

        cls.start_params = (0.5, 1, 0.5)


class TestDiscretizedGammaEx():
    # strike outbreaks example from Ch... 2012

    def test_all(self):
        # expand frequencies to observations, (no freq_weights yet)
        freq = [46, 76, 24, 9, 1]
        y = np.repeat(np.arange(5), freq)
        # results from article table 7
        res1 = Bunch(
            params=[3.52636, 0.425617],
            llf=-187.469,
            chi2=1.701208,  # chisquare test
            df_model=0,
            p=0.4272,  # p-value for chi2
            aic=378.938,
            probs=[46.48, 73.72, 27.88, 6.5, 1.42])

        dp = DiscretizedCount(stats.gamma)
        mod = DiscretizedModel(y, distr=dp)
        res = mod.fit(start_params=[1, 1])
        nobs = len(y)

        assert_allclose(res.params, res1.params, rtol=1e-5)
        assert_allclose(res.llf, res1.llf, atol=6e-3)
        assert_allclose(res.aic, res1.aic, atol=6e-3)
        assert_equal(res.df_model, res1.df_model)

        probs = mod.predict(res.params, which="probs")
        probs_trunc = probs[:len(res1.probs)]
        probs_trunc[-1] += 1 - probs_trunc.sum()
        assert_allclose(probs_trunc * nobs, res1.probs, atol=6e-2)

        assert_allclose(np.sum(freq), (probs_trunc * nobs).sum(), rtol=1e-10)
        res_chi2 = stats.chisquare(freq, probs_trunc * nobs,
                                   ddof=len(res.params))
        # regression test, numbers from running test
        # close but not identical to article
        assert_allclose(res_chi2.statistic, 1.70409356, rtol=1e-7)
        assert_allclose(res_chi2.pvalue, 0.42654100, rtol=1e-7)

        # smoke test for summary
        res.summary()

        np.random.seed(987146)
        res_boots = res.bootstrap()
        # only loose check, small default n_rep=100, agreement at around 3%
        assert_allclose(res.params, res_boots[0], rtol=0.05)
        assert_allclose(res.bse, res_boots[1], rtol=0.05)


class TestGeometric():

    def test_all(self):
        p_geom = 0.6
        scale_dexpon = -1 / np.log(1-p_geom)
        dgeo = stats.geom(p_geom, loc=-1)
        dpg = DiscretizedCount(stats.expon)(scale_dexpon)

        xi = np.arange(6)
        pmf1 = dgeo.pmf(xi)
        pmf = dpg.pmf(xi)
        assert_allclose(pmf, pmf1, rtol=1e-10)
        cdf1 = dgeo.cdf(xi)
        cdf = dpg.cdf(xi)
        assert_allclose(cdf, cdf1, rtol=1e-10)
        sf1 = dgeo.sf(xi)
        sf = dpg.sf(xi)
        assert_allclose(sf, sf1, rtol=1e-10)

        ppf1 = dgeo.ppf(cdf1)
        ppf = dpg.ppf(cdf1)
        assert_equal(ppf, ppf1)
        ppf1 = dgeo.ppf(cdf1 - 1e-8)
        ppf = dpg.ppf(cdf1 - 1e-8)
        assert_equal(ppf, ppf1)
        ppf1 = dgeo.ppf(cdf1 + 1e-8)
        ppf = dpg.ppf(cdf1 + 1e-8)
        assert_equal(ppf, ppf1)
        ppf1 = dgeo.ppf(0)  # incorrect in scipy < 1.5.0
        ppf = dpg.ppf(0)
        assert_equal(ppf, -1)

        # isf
        isf1 = dgeo.isf(sf1)
        isf = dpg.isf(sf1)
        assert_equal(isf, isf1)
        isf1 = dgeo.isf(sf1 - 1e-8)
        isf = dpg.isf(sf1 - 1e-8)
        assert_equal(isf, isf1)
        isf1 = dgeo.isf(sf1 + 1e-8)
        isf = dpg.isf(sf1 + 1e-8)
        assert_equal(isf, isf1)
        isf1 = dgeo.isf(0)
        isf = dpg.isf(0)
        assert_equal(isf, isf1)  # inf
        isf1 = dgeo.isf(1)  # currently incorrect in scipy
        isf = dpg.isf(1)
        assert_equal(isf, -1)


class TestTruncatedNBP:
    """
    Test Truncated Poisson distribution
    """
    def test_pmf_zero(self):
        n, p = truncatednegbin.convert_params(5, 0.1, 2)
        nb_pmf = nbinom.pmf(1, n, p) / nbinom.sf(0, n, p)
        tnb_pmf = truncatednegbin.pmf(1, 5, 0.1, 2, 0)
        assert_allclose(nb_pmf, tnb_pmf, rtol=1e-5)

    def test_logpmf_zero(self):
        n, p = truncatednegbin.convert_params(5, 1, 2)
        nb_logpmf = nbinom.logpmf(1, n, p) - np.log(nbinom.sf(0, n, p))
        tnb_logpmf = truncatednegbin.logpmf(1, 5, 1, 2, 0)
        assert_allclose(nb_logpmf, tnb_logpmf, rtol=1e-2, atol=1e-2)

    def test_pmf(self):
        n, p = truncatednegbin.convert_params(2, 0.5, 2)
        nb_logpmf = nbinom.pmf(6, n, p) / nbinom.sf(5, n, p)
        tnb_pmf = truncatednegbin.pmf(6, 2, 0.5, 2, 5)
        assert_allclose(nb_logpmf, tnb_pmf, rtol=1e-7)

        tnb_pmf = truncatednegbin.pmf(5, 2, 0.5, 2, 5)
        assert_equal(tnb_pmf, 0)

    def test_logpmf(self):
        n, p = truncatednegbin.convert_params(5, 0.1, 2)
        nb_logpmf = nbinom.logpmf(6, n, p) - np.log(nbinom.sf(5, n, p))
        tnb_logpmf = truncatednegbin.logpmf(6, 5, 0.1, 2, 5)

        assert_allclose(nb_logpmf, tnb_logpmf, rtol=1e-7)

        tnb_logpmf = truncatednegbin.logpmf(5, 5, 0.1, 2, 5)
        assert np.isneginf(tnb_logpmf)
