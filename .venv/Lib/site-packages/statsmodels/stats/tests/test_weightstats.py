'''tests for weightstats, compares with replication

no failures but needs cleanup
update 2012-09-09:
   added test after fixing bug in covariance
   TODOs:
     - I do not remember what all the commented out code is doing
     - should be refactored to use generator or inherited tests
     - still gaps in test coverage
       - value/diff in ttest_ind is tested in test_tost.py
     - what about pandas data structures?

Author: Josef Perktold
License: BSD (3-clause)

'''

import numpy as np
from scipy import stats
import pandas as pd
from numpy.testing import assert_, assert_almost_equal, assert_allclose

from statsmodels.stats.weightstats import (DescrStatsW, CompareMeans,
                                           ttest_ind, ztest, zconfint)
from statsmodels.tools.testing import Holder


# Mixin for tests against other packages.
class CheckExternalMixin:

    @classmethod
    def get_descriptives(cls, ddof=0):
        cls.descriptive = DescrStatsW(cls.data, cls.weights, ddof)

    # TODO: not a test, belongs elsewhere?
    @classmethod
    def save_data(cls, fname="data.csv"):
        # Utility to get data into another package.
        df = pd.DataFrame(index=np.arange(len(cls.weights)))
        df["weights"] = cls.weights
        if cls.data.ndim == 1:
            df["data1"] = cls.data
        else:
            for k in range(cls.data.shape[1]):
                df["data%d" % (k + 1)] = cls.data[:, k]
        df.to_csv(fname)

    def test_mean(self):
        mn = self.descriptive.mean
        assert_allclose(mn, self.mean, rtol=1e-4)

    def test_sum(self):
        sm = self.descriptive.sum
        assert_allclose(sm, self.sum, rtol=1e-4)

    def test_var(self):
        # Use vardef=wgt option in SAS to match
        var = self.descriptive.var
        assert_allclose(var, self.var, rtol=1e-4)

    def test_std(self):
        # Use vardef=wgt option in SAS to match
        std = self.descriptive.std
        assert_allclose(std, self.std, rtol=1e-4)

    def test_sem(self):
        # Use default vardef in SAS to match; only makes sense if
        # weights sum to n.
        if not hasattr(self, "sem"):
            return
        sem = self.descriptive.std_mean
        assert_allclose(sem, self.sem, rtol=1e-4)

    def test_quantiles(self):
        quant = np.asarray(self.quantiles, dtype=np.float64)
        for return_pandas in False, True:
            qtl = self.descriptive.quantile(self.quantile_probs,
                                            return_pandas=return_pandas)
            qtl = np.asarray(qtl, dtype=np.float64)
            assert_allclose(qtl, quant, rtol=1e-4)


class TestSim1(CheckExternalMixin):
    # 1d data

    # Taken from SAS
    mean = 0.401499
    sum = 12.9553441
    var = 1.08022
    std = 1.03933
    quantiles = np.r_[-1.81098, -0.84052, 0.32859, 0.77808, 2.93431]

    @classmethod
    def setup_class(cls):
        np.random.seed(9876789)
        cls.data = np.random.normal(size=20)
        cls.weights = np.random.uniform(0, 3, size=20)
        cls.quantile_probs = np.r_[0, 0.1, 0.5, 0.75, 1]
        cls.get_descriptives()


class TestSim1t(CheckExternalMixin):
    # 1d data with ties

    # Taken from SAS
    mean = 5.05103296
    sum = 156.573464
    var = 9.9711934
    std = 3.15771965
    quantiles = np.r_[0, 1, 5, 8, 9]

    @classmethod
    def setup_class(cls):
        np.random.seed(9876789)
        cls.data = np.random.randint(0, 10, size=20)
        cls.data[15:20] = cls.data[0:5]
        cls.data[18:20] = cls.data[15:17]
        cls.weights = np.random.uniform(0, 3, size=20)
        cls.quantile_probs = np.r_[0, 0.1, 0.5, 0.75, 1]
        cls.get_descriptives()


class TestSim1n(CheckExternalMixin):
    # 1d data with weights summing to n so we can check the standard
    # error of the mean

    # Taken from SAS
    mean = -0.3131058
    sum = -6.2621168
    var = 0.49722696
    std = 0.70514322
    sem = 0.15767482
    quantiles = np.r_[-1.61593, -1.45576, -0.24356, 0.16770, 1.18791]

    @classmethod
    def setup_class(cls):
        np.random.seed(4342)
        cls.data = np.random.normal(size=20)
        cls.weights = np.random.uniform(0, 3, size=20)
        cls.weights *= 20 / cls.weights.sum()
        cls.quantile_probs = np.r_[0, 0.1, 0.5, 0.75, 1]
        cls.get_descriptives(1)


class TestSim2(CheckExternalMixin):
    # 2d data

    # Taken from SAS
    mean = [-0.2170406, -0.2387543]
    sum = [-6.8383999, -7.5225444]
    var = [1.77426344, 0.61933542]
    std = [1.3320148, 0.78697867]
    quantiles = np.column_stack(
        (np.r_[-2.55277, -1.40479, -0.61040, 0.52740, 2.66246],
         np.r_[-1.49263, -1.15403, -0.16231, 0.16464, 1.83062]))

    @classmethod
    def setup_class(cls):
        np.random.seed(2249)
        cls.data = np.random.normal(size=(20, 2))
        cls.weights = np.random.uniform(0, 3, size=20)
        cls.quantile_probs = np.r_[0, 0.1, 0.5, 0.75, 1]
        cls.get_descriptives()


class TestWeightstats:

    @classmethod
    def setup_class(cls):
        np.random.seed(9876789)
        n1, n2 = 20, 20
        m1, m2 = 1, 1.2
        x1 = m1 + np.random.randn(n1)
        x2 = m2 + np.random.randn(n2)
        x1_2d = m1 + np.random.randn(n1, 3)
        x2_2d = m2 + np.random.randn(n2, 3)
        w1 = np.random.randint(1,4, n1)
        w2 = np.random.randint(1,4, n2)
        cls.x1, cls.x2 = x1, x2
        cls.w1, cls.w2 = w1, w2
        cls.x1_2d, cls.x2_2d = x1_2d, x2_2d

    def test_weightstats_1(self):
        x1, x2 = self.x1, self.x2
        w1, w2 = self.w1, self.w2
        w1_ = 2. * np.ones(len(x1))
        w2_ = 2. * np.ones(len(x2))

        d1 = DescrStatsW(x1)
        #        print ttest_ind(x1, x2)
        #        print ttest_ind(x1, x2, usevar='unequal')
        #        #print ttest_ind(x1, x2, usevar='unequal')
        #        print stats.ttest_ind(x1, x2)
        #        print ttest_ind(x1, x2, usevar='unequal', alternative='larger')
        #        print ttest_ind(x1, x2, usevar='unequal', alternative='smaller')
        #        print ttest_ind(x1, x2, usevar='unequal', weights=(w1_, w2_))
        #        print stats.ttest_ind(np.r_[x1, x1], np.r_[x2,x2])
        assert_almost_equal(ttest_ind(x1, x2, weights=(w1_, w2_))[:2],
                            stats.ttest_ind(np.r_[x1, x1], np.r_[x2, x2]))

    def test_weightstats_2(self):
        x1, x2 = self.x1, self.x2
        w1, w2 = self.w1, self.w2

        d1 = DescrStatsW(x1)
        d1w = DescrStatsW(x1, weights=w1)
        d2w = DescrStatsW(x2, weights=w2)
        x1r = d1w.asrepeats()
        x2r = d2w.asrepeats()
        #        print 'random weights'
        #        print ttest_ind(x1, x2, weights=(w1, w2))
        #        print stats.ttest_ind(x1r, x2r)
        assert_almost_equal(ttest_ind(x1, x2, weights=(w1, w2))[:2],
                            stats.ttest_ind(x1r, x2r), 14)
        # not the same as new version with random weights/replication
        #        assert x1r.shape[0] == d1w.sum_weights
        #        assert x2r.shape[0] == d2w.sum_weights

        assert_almost_equal(x2r.mean(0), d2w.mean, 14)
        assert_almost_equal(x2r.var(), d2w.var, 14)
        assert_almost_equal(x2r.std(), d2w.std, 14)
        # note: the following is for 1d
        assert_almost_equal(np.cov(x2r, bias=1), d2w.cov, 14)
        # assert_almost_equal(np.corrcoef(np.x2r), d2w.corrcoef, 19)
        # TODO: exception in corrcoef (scalar case)

        # one-sample tests
        #        print d1.ttest_mean(3)
        #        print stats.ttest_1samp(x1, 3)
        #        print d1w.ttest_mean(3)
        #        print stats.ttest_1samp(x1r, 3)
        assert_almost_equal(d1.ttest_mean(3)[:2], stats.ttest_1samp(x1, 3), 11)
        assert_almost_equal(d1w.ttest_mean(3)[:2],
                            stats.ttest_1samp(x1r, 3), 11)

    def test_weightstats_3(self):
        x1_2d, x2_2d = self.x1_2d, self.x2_2d
        w1, w2 = self.w1, self.w2

        d1w_2d = DescrStatsW(x1_2d, weights=w1)
        d2w_2d = DescrStatsW(x2_2d, weights=w2)
        x1r_2d = d1w_2d.asrepeats()
        x2r_2d = d2w_2d.asrepeats()

        assert_almost_equal(x2r_2d.mean(0), d2w_2d.mean, 14)
        assert_almost_equal(x2r_2d.var(0), d2w_2d.var, 14)
        assert_almost_equal(x2r_2d.std(0), d2w_2d.std, 14)
        assert_almost_equal(np.cov(x2r_2d.T, bias=1), d2w_2d.cov, 14)
        assert_almost_equal(np.corrcoef(x2r_2d.T), d2w_2d.corrcoef, 14)

        #        print d1w_2d.ttest_mean(3)
        #        #scipy.stats.ttest is also vectorized
        #        print stats.ttest_1samp(x1r_2d, 3)
        t, p, d = d1w_2d.ttest_mean(3)
        assert_almost_equal([t, p], stats.ttest_1samp(x1r_2d, 3), 11)
        # print [stats.ttest_1samp(xi, 3) for xi in x1r_2d.T]
        cm = CompareMeans(d1w_2d, d2w_2d)
        ressm = cm.ttest_ind()
        resss = stats.ttest_ind(x1r_2d, x2r_2d)
        assert_almost_equal(ressm[:2], resss, 14)

#        does not work for 2d, levene does not use weights
#        cm = CompareMeans(d1w_2d, d2w_2d)
#        ressm = cm.test_equal_var()
#        resss = stats.levene(x1r_2d, x2r_2d)
#        assert_almost_equal(ressm[:2], resss, 14)

    def test_weightstats_ddof_tests(self):
        # explicit test that ttest and confint are independent of ddof
        # one sample case
        x1_2d = self.x1_2d
        w1 = self.w1

        d1w_d0 = DescrStatsW(x1_2d, weights=w1, ddof=0)
        d1w_d1 = DescrStatsW(x1_2d, weights=w1, ddof=1)
        d1w_d2 = DescrStatsW(x1_2d, weights=w1, ddof=2)

        # check confint independent of user ddof
        res0 = d1w_d0.ttest_mean()
        res1 = d1w_d1.ttest_mean()
        res2 = d1w_d2.ttest_mean()
        # concatenate into one array with np.r_
        assert_almost_equal(np.r_[res1], np.r_[res0], 14)
        assert_almost_equal(np.r_[res2], np.r_[res0], 14)

        res0 = d1w_d0.ttest_mean(0.5)
        res1 = d1w_d1.ttest_mean(0.5)
        res2 = d1w_d2.ttest_mean(0.5)
        assert_almost_equal(np.r_[res1], np.r_[res0], 14)
        assert_almost_equal(np.r_[res2], np.r_[res0], 14)

        # check confint independent of user ddof
        res0 = d1w_d0.tconfint_mean()
        res1 = d1w_d1.tconfint_mean()
        res2 = d1w_d2.tconfint_mean()
        assert_almost_equal(res1, res0, 14)
        assert_almost_equal(res2, res0, 14)

    def test_comparemeans_convenient_interface(self):
        x1_2d, x2_2d = self.x1_2d, self.x2_2d
        d1 = DescrStatsW(x1_2d)
        d2 = DescrStatsW(x2_2d)
        cm1 = CompareMeans(d1, d2)

        # smoke test for summary
        from statsmodels.iolib.table import SimpleTable
        for use_t in [True, False]:
            for usevar in ['pooled', 'unequal']:
                smry = cm1.summary(use_t=use_t, usevar=usevar)
                assert_(isinstance(smry, SimpleTable))

        # test for from_data method
        cm2 = CompareMeans.from_data(x1_2d, x2_2d)
        assert_(str(cm1.summary()) == str(cm2.summary()))

    def test_comparemeans_convenient_interface_1d(self):
        # same as above for 2d, just use 1d data instead
        x1_2d, x2_2d = self.x1, self.x2
        d1 = DescrStatsW(x1_2d)
        d2 = DescrStatsW(x2_2d)
        cm1 = CompareMeans(d1, d2)

        # smoke test for summary
        from statsmodels.iolib.table import SimpleTable
        for use_t in [True, False]:
            for usevar in ['pooled', 'unequal']:
                smry = cm1.summary(use_t=use_t, usevar=usevar)
                assert_(isinstance(smry, SimpleTable))

        # test for from_data method
        cm2 = CompareMeans.from_data(x1_2d, x2_2d)
        assert_(str(cm1.summary()) == str(cm2.summary()))


class CheckWeightstats1dMixin:

    def test_basic(self):
        x1r = self.x1r
        d1w = self.d1w

        assert_almost_equal(x1r.mean(0), d1w.mean, 14)
        assert_almost_equal(x1r.var(0, ddof=d1w.ddof), d1w.var, 14)
        assert_almost_equal(x1r.std(0, ddof=d1w.ddof), d1w.std, 14)
        var1 = d1w.var_ddof(ddof=1)
        assert_almost_equal(x1r.var(0, ddof=1), var1, 14)
        std1 = d1w.std_ddof(ddof=1)
        assert_almost_equal(x1r.std(0, ddof=1), std1, 14)

        assert_almost_equal(np.cov(x1r.T, bias=1-d1w.ddof), d1w.cov, 14)

        # assert_almost_equal(np.corrcoef(x1r.T), d1w.corrcoef, 14)

    def test_ttest(self):
        x1r = self.x1r
        d1w = self.d1w
        assert_almost_equal(d1w.ttest_mean(3)[:2],
                            stats.ttest_1samp(x1r, 3), 11)

#    def
#        assert_almost_equal(ttest_ind(x1, x2, weights=(w1, w2))[:2],
#                            stats.ttest_ind(x1r, x2r), 14)

    def test_ttest_2sample(self):
        x1, x2 = self.x1, self.x2
        x1r, x2r = self.x1r, self.x2r
        w1, w2 = self.w1, self.w2

        # Note: stats.ttest_ind handles 2d/nd arguments
        res_sp = stats.ttest_ind(x1r, x2r)
        assert_almost_equal(ttest_ind(x1, x2, weights=(w1, w2))[:2],
                            res_sp, 14)

        # check correct ttest independent of user ddof
        cm = CompareMeans(DescrStatsW(x1, weights=w1, ddof=0),
                          DescrStatsW(x2, weights=w2, ddof=1))
        assert_almost_equal(cm.ttest_ind()[:2], res_sp, 14)

        cm = CompareMeans(DescrStatsW(x1, weights=w1, ddof=1),
                          DescrStatsW(x2, weights=w2, ddof=2))
        assert_almost_equal(cm.ttest_ind()[:2], res_sp, 14)

        cm0 = CompareMeans(DescrStatsW(x1, weights=w1, ddof=0),
                           DescrStatsW(x2, weights=w2, ddof=0))
        cm1 = CompareMeans(DescrStatsW(x1, weights=w1, ddof=0),
                           DescrStatsW(x2, weights=w2, ddof=1))
        cm2 = CompareMeans(DescrStatsW(x1, weights=w1, ddof=1),
                           DescrStatsW(x2, weights=w2, ddof=2))

        res0 = cm0.ttest_ind(usevar='unequal')
        res1 = cm1.ttest_ind(usevar='unequal')
        res2 = cm2.ttest_ind(usevar='unequal')
        assert_almost_equal(res1, res0, 14)
        assert_almost_equal(res2, res0, 14)

        # check confint independent of user ddof
        res0 = cm0.tconfint_diff(usevar='pooled')
        res1 = cm1.tconfint_diff(usevar='pooled')
        res2 = cm2.tconfint_diff(usevar='pooled')
        assert_almost_equal(res1, res0, 14)
        assert_almost_equal(res2, res0, 14)

        res0 = cm0.tconfint_diff(usevar='unequal')
        res1 = cm1.tconfint_diff(usevar='unequal')
        res2 = cm2.tconfint_diff(usevar='unequal')
        assert_almost_equal(res1, res0, 14)
        assert_almost_equal(res2, res0, 14)

    def test_confint_mean(self):
        # compare confint_mean with ttest
        d1w = self.d1w
        alpha = 0.05
        low, upp = d1w.tconfint_mean()
        t, p, d = d1w.ttest_mean(low)
        assert_almost_equal(p, alpha * np.ones(p.shape), 8)
        t, p, d = d1w.ttest_mean(upp)
        assert_almost_equal(p, alpha * np.ones(p.shape), 8)
        t, p, d = d1w.ttest_mean(np.vstack((low, upp)))
        assert_almost_equal(p, alpha * np.ones(p.shape), 8)


class CheckWeightstats2dMixin(CheckWeightstats1dMixin):

    def test_corr(self):
        x1r = self.x1r
        d1w = self.d1w

        assert_almost_equal(np.corrcoef(x1r.T), d1w.corrcoef, 14)


class TestWeightstats1d_ddof(CheckWeightstats1dMixin):

    @classmethod
    def setup_class(cls):
        np.random.seed(9876789)
        n1, n2 = 20, 20
        m1, m2 = 1, 1.2
        x1 = m1 + np.random.randn(n1, 1)
        x2 = m2 + np.random.randn(n2, 1)
        w1 = np.random.randint(1, 4, n1)
        w2 = np.random.randint(1, 4, n2)

        cls.x1, cls.x2 = x1, x2
        cls.w1, cls.w2 = w1, w2
        cls.d1w = DescrStatsW(x1, weights=w1, ddof=1)
        cls.d2w = DescrStatsW(x2, weights=w2, ddof=1)
        cls.x1r = cls.d1w.asrepeats()
        cls.x2r = cls.d2w.asrepeats()


class TestWeightstats2d(CheckWeightstats2dMixin):

    @classmethod
    def setup_class(cls):
        np.random.seed(9876789)
        n1, n2 = 20, 20
        m1, m2 = 1, 1.2
        x1 = m1 + np.random.randn(n1, 3)
        x2 = m2 + np.random.randn(n2, 3)
        w1 = np.random.randint(1, 4, n1)
        w2 = np.random.randint(1, 4, n2)
        cls.x1, cls.x2 = x1, x2
        cls.w1, cls.w2 = w1, w2

        cls.d1w = DescrStatsW(x1, weights=w1)
        cls.d2w = DescrStatsW(x2, weights=w2)
        cls.x1r = cls.d1w.asrepeats()
        cls.x2r = cls.d2w.asrepeats()


class TestWeightstats2d_ddof(CheckWeightstats2dMixin):

    @classmethod
    def setup_class(cls):
        np.random.seed(9876789)
        n1, n2 = 20, 20
        m1, m2 = 1, 1.2
        x1 = m1 + np.random.randn(n1, 3)
        x2 = m2 + np.random.randn(n2, 3)
        w1 = np.random.randint(1, 4, n1)
        w2 = np.random.randint(1, 4, n2)

        cls.x1, cls.x2 = x1, x2
        cls.w1, cls.w2 = w1, w2
        cls.d1w = DescrStatsW(x1, weights=w1, ddof=1)
        cls.d2w = DescrStatsW(x2, weights=w2, ddof=1)
        cls.x1r = cls.d1w.asrepeats()
        cls.x2r = cls.d2w.asrepeats()


class TestWeightstats2d_nobs(CheckWeightstats2dMixin):

    @classmethod
    def setup_class(cls):
        np.random.seed(9876789)
        n1, n2 = 20, 30
        m1, m2 = 1, 1.2
        x1 = m1 + np.random.randn(n1, 3)
        x2 = m2 + np.random.randn(n2, 3)
        w1 = np.random.randint(1, 4, n1)
        w2 = np.random.randint(1, 4, n2)

        cls.x1, cls.x2 = x1, x2
        cls.w1, cls.w2 = w1, w2
        cls.d1w = DescrStatsW(x1, weights=w1, ddof=0)
        cls.d2w = DescrStatsW(x2, weights=w2, ddof=1)
        cls.x1r = cls.d1w.asrepeats()
        cls.x2r = cls.d2w.asrepeats()


def test_ttest_ind_with_uneq_var():

    # from scipy
    # check vs. R
    a = (1, 2, 3)
    b = (1.1, 2.9, 4.2)
    pr = 0.53619490753126731
    tr = -0.68649512735572582
    t, p, df = ttest_ind(a, b, usevar='unequal')
    assert_almost_equal([t, p], [tr, pr], 13)

    a = (1, 2, 3, 4)
    pr = 0.84354139131608286
    tr = -0.2108663315950719
    t, p, df = ttest_ind(a, b, usevar='unequal')
    assert_almost_equal([t, p], [tr, pr], 13)


def test_ztest_ztost():
    # compare weightstats with separately tested proportion ztest ztost
    import statsmodels.stats.proportion as smprop

    x1 = [0, 1]
    w1 = [5, 15]

    res2 = smprop.proportions_ztest(15, 20., value=0.5)
    d1 = DescrStatsW(x1, w1)
    res1 = d1.ztest_mean(0.5)
    assert_allclose(res1, res2, rtol=0.03, atol=0.003)

    d2 = DescrStatsW(x1, np.array(w1)*21./20)
    res1 = d2.ztest_mean(0.5)
    assert_almost_equal(res1, res2, decimal=12)

    res1 = d2.ztost_mean(0.4, 0.6)
    res2 = smprop.proportions_ztost(15, 20., 0.4, 0.6)
    assert_almost_equal(res1[0], res2[0], decimal=12)

    x2 = [0, 1]
    w2 = [10, 10]
    # d2 = DescrStatsW(x1, np.array(w1)*21./20)
    d2 = DescrStatsW(x2, w2)
    res1 = ztest(d1.asrepeats(), d2.asrepeats())
    res2 = smprop.proportions_chisquare(np.asarray([15, 10]),
                                        np.asarray([20., 20]))
    # TODO: check this is this difference expected?, see test_proportion
    assert_allclose(res1[1], res2[1], rtol=0.03)

    res1a = CompareMeans(d1, d2).ztest_ind()
    assert_allclose(res1a[1], res2[1], rtol=0.03)
    assert_almost_equal(res1a, res1, decimal=12)


# test for ztest and z confidence interval against R BSDA z.test
# Note: I needed to calculate the pooled standard deviation for R
#       std = np.std(np.concatenate((x-x.mean(),y-y.mean())), ddof=2)

# > zt = z.test(x, sigma.x=0.57676142668828667, y, sigma.y=0.57676142668828667)
# > cat_items(zt, "ztest.")
ztest_ = Holder()
ztest_.statistic = 6.55109865675183
ztest_.p_value = 5.711530850508982e-11
ztest_.conf_int = np.array([1.230415246535603, 2.280948389828034])
ztest_.estimate = np.array([7.01818181818182, 5.2625])
ztest_.null_value = 0
ztest_.alternative = 'two.sided'
ztest_.method = 'Two-sample z-Test'
ztest_.data_name = 'x and y'
# > zt = z.test(x, sigma.x=0.57676142668828667, y,
#               sigma.y=0.57676142668828667, alternative="less")
# > cat_items(zt, "ztest_smaller.")
ztest_smaller = Holder()
ztest_smaller.statistic = 6.55109865675183
ztest_smaller.p_value = 0.999999999971442
ztest_smaller.conf_int = np.array([np.nan, 2.196499421109045])
ztest_smaller.estimate = np.array([7.01818181818182, 5.2625])
ztest_smaller.null_value = 0
ztest_smaller.alternative = 'less'
ztest_smaller.method = 'Two-sample z-Test'
ztest_smaller.data_name = 'x and y'
# > zt = z.test(x, sigma.x=0.57676142668828667, y,
#               sigma.y=0.57676142668828667, alternative="greater")
# > cat_items(zt, "ztest_larger.")
ztest_larger = Holder()
ztest_larger.statistic = 6.55109865675183
ztest_larger.p_value = 2.855760072861813e-11
ztest_larger.conf_int = np.array([1.314864215254592, np.nan])
ztest_larger.estimate = np.array([7.01818181818182, 5.2625])
ztest_larger.null_value = 0
ztest_larger.alternative = 'greater'
ztest_larger.method = 'Two-sample z-Test'
ztest_larger.data_name = 'x and y'


# > zt = z.test(x, sigma.x=0.57676142668828667, y,
#               sigma.y=0.57676142668828667, mu=1, alternative="two.sided")
# > cat_items(zt, "ztest_mu.")
ztest_mu = Holder()
ztest_mu.statistic = 2.81972854805176
ztest_mu.p_value = 0.00480642898427981
ztest_mu.conf_int = np.array([1.230415246535603, 2.280948389828034])
ztest_mu.estimate = np.array([7.01818181818182, 5.2625])
ztest_mu.null_value = 1
ztest_mu.alternative = 'two.sided'
ztest_mu.method = 'Two-sample z-Test'
ztest_mu.data_name = 'x and y'

# > zt = z.test(x, sigma.x=0.57676142668828667, y,
#               sigma.y=0.57676142668828667, mu=1, alternative="greater")
# > cat_items(zt, "ztest_larger_mu.")
ztest_larger_mu = Holder()
ztest_larger_mu.statistic = 2.81972854805176
ztest_larger_mu.p_value = 0.002403214492139871
ztest_larger_mu.conf_int = np.array([1.314864215254592, np.nan])
ztest_larger_mu.estimate = np.array([7.01818181818182, 5.2625])
ztest_larger_mu.null_value = 1
ztest_larger_mu.alternative = 'greater'
ztest_larger_mu.method = 'Two-sample z-Test'
ztest_larger_mu.data_name = 'x and y'

# > zt = z.test(x, sigma.x=0.57676142668828667, y,
#               sigma.y=0.57676142668828667, mu=2, alternative="less")
# > cat_items(zt, "ztest_smaller_mu.")
ztest_smaller_mu = Holder()
ztest_smaller_mu.statistic = -0.911641560648313
ztest_smaller_mu.p_value = 0.1809787183191324
ztest_smaller_mu.conf_int = np.array([np.nan, 2.196499421109045])
ztest_smaller_mu.estimate = np.array([7.01818181818182, 5.2625])
ztest_smaller_mu.null_value = 2
ztest_smaller_mu.alternative = 'less'
ztest_smaller_mu.method = 'Two-sample z-Test'
ztest_smaller_mu.data_name = 'x and y'

# > zt = z.test(x, sigma.x=0.46436662631627995, mu=6.4,
#               alternative="two.sided")
# > cat_items(zt, "ztest_mu_1s.")
ztest_mu_1s = Holder()
ztest_mu_1s.statistic = 4.415212090914452
ztest_mu_1s.p_value = 1.009110038015147e-05
ztest_mu_1s.conf_int = np.array([6.74376372125119, 7.29259991511245])
ztest_mu_1s.estimate = 7.01818181818182
ztest_mu_1s.null_value = 6.4
ztest_mu_1s.alternative = 'two.sided'
ztest_mu_1s.method = 'One-sample z-Test'
ztest_mu_1s.data_name = 'x'

# > zt = z.test(x, sigma.x=0.46436662631627995, mu=7.4, alternative="less")
# > cat_items(zt, "ztest_smaller_mu_1s.")
ztest_smaller_mu_1s = Holder()
ztest_smaller_mu_1s.statistic = -2.727042762035397
ztest_smaller_mu_1s.p_value = 0.00319523783881176
ztest_smaller_mu_1s.conf_int = np.array([np.nan, 7.248480744895716])
ztest_smaller_mu_1s.estimate = 7.01818181818182
ztest_smaller_mu_1s.null_value = 7.4
ztest_smaller_mu_1s.alternative = 'less'
ztest_smaller_mu_1s.method = 'One-sample z-Test'
ztest_smaller_mu_1s.data_name = 'x'

# > zt = z.test(x, sigma.x=0.46436662631627995, mu=6.4, alternative="greater")
# > cat_items(zt, "ztest_greater_mu_1s.")
ztest_larger_mu_1s = Holder()
ztest_larger_mu_1s.statistic = 4.415212090914452
ztest_larger_mu_1s.p_value = 5.045550190097003e-06
ztest_larger_mu_1s.conf_int = np.array([6.78788289146792, np.nan])
ztest_larger_mu_1s.estimate = 7.01818181818182
ztest_larger_mu_1s.null_value = 6.4
ztest_larger_mu_1s.alternative = 'greater'
ztest_larger_mu_1s.method = 'One-sample z-Test'
ztest_larger_mu_1s.data_name = 'x'

# > zt = z.test(x, sigma.x=0.46436662631627995, y, sigma.y=0.7069805008424409)
# > cat_items(zt, "ztest_unequal.")
ztest_unequal = Holder()
ztest_unequal.statistic = 6.12808151466544
ztest_unequal.p_value = 8.89450168270109e-10
ztest_unequal.conf_int = np.array([1.19415646579981, 2.31720717056382])
ztest_unequal.estimate = np.array([7.01818181818182, 5.2625])
ztest_unequal.null_value = 0
ztest_unequal.alternative = 'two.sided'
ztest_unequal.usevar = 'unequal'
ztest_unequal.method = 'Two-sample z-Test'
ztest_unequal.data_name = 'x and y'

# > zt = z.test(x, sigma.x=0.46436662631627995, y, sigma.y=0.7069805008424409, alternative="less")
# > cat_items(zt, "ztest_smaller_unequal.")
ztest_smaller_unequal = Holder()
ztest_smaller_unequal.statistic = 6.12808151466544
ztest_smaller_unequal.p_value = 0.999999999555275
ztest_smaller_unequal.conf_int = np.array([np.nan, 2.22692874913371])
ztest_smaller_unequal.estimate = np.array([7.01818181818182, 5.2625])
ztest_smaller_unequal.null_value = 0
ztest_smaller_unequal.alternative = 'less'
ztest_smaller_unequal.usevar = 'unequal'
ztest_smaller_unequal.method = 'Two-sample z-Test'
ztest_smaller_unequal.data_name = 'x and y'

# > zt = z.test(x, sigma.x=0.46436662631627995, y, sigma.y=0.7069805008424409, alternative="greater")
# > cat_items(zt, "ztest_larger_unequal.")
ztest_larger_unequal = Holder()
ztest_larger_unequal.statistic = 6.12808151466544
ztest_larger_unequal.p_value = 4.44725034576265e-10
ztest_larger_unequal.conf_int = np.array([1.28443488722992, np.nan])
ztest_larger_unequal.estimate = np.array([7.01818181818182, 5.2625])
ztest_larger_unequal.null_value = 0
ztest_larger_unequal.alternative = 'greater'
ztest_larger_unequal.usevar = 'unequal'
ztest_larger_unequal.method = 'Two-sample z-Test'
ztest_larger_unequal.data_name = 'x and y'


alternatives = {'less': 'smaller',
                'greater': 'larger',
                'two.sided': 'two-sided'}


class TestZTest:
    # all examples use the same data
    # no weights used in tests

    @classmethod
    def setup_class(cls):
        cls.x1 = np.array([7.8, 6.6, 6.5, 7.4, 7.3, 7., 6.4,
                          7.1, 6.7, 7.6, 6.8])
        cls.x2 = np.array([4.5, 5.4, 6.1, 6.1, 5.4, 5., 4.1, 5.5])
        cls.d1 = DescrStatsW(cls.x1)
        cls.d2 = DescrStatsW(cls.x2)
        cls.cm = CompareMeans(cls.d1, cls.d2)

    def test(self):
        x1, x2 = self.x1, self.x2
        cm = self.cm

        # tc : test cases
        for tc in [ztest_, ztest_smaller, ztest_larger,
                   ztest_mu, ztest_smaller_mu, ztest_larger_mu]:

            zstat, pval = ztest(x1, x2, value=tc.null_value,
                                alternative=alternatives[tc.alternative])
            assert_allclose(zstat, tc.statistic, rtol=1e-10)
            assert_allclose(pval, tc.p_value, rtol=1e-10, atol=1e-16)

            zstat, pval = cm.ztest_ind(value=tc.null_value,
                                alternative=alternatives[tc.alternative])
            assert_allclose(zstat, tc.statistic, rtol=1e-10)
            assert_allclose(pval, tc.p_value, rtol=1e-10, atol=1e-16)

            # overwrite nan in R's confint
            tc_conf_int = tc.conf_int.copy()
            if np.isnan(tc_conf_int[0]):
                tc_conf_int[0] = - np.inf
            if np.isnan(tc_conf_int[1]):
                tc_conf_int[1] = np.inf

            # Note: value is shifting our confidence interval in zconfint
            ci = zconfint(x1, x2, value=0,
                          alternative=alternatives[tc.alternative])
            assert_allclose(ci, tc_conf_int, rtol=1e-10)

            ci = cm.zconfint_diff(alternative=alternatives[tc.alternative])
            assert_allclose(ci, tc_conf_int, rtol=1e-10)

            ci = zconfint(x1, x2, value=tc.null_value,
                          alternative=alternatives[tc.alternative])
            assert_allclose(ci, tc_conf_int - tc.null_value, rtol=1e-10)

        # unequal variances
        for tc in [ztest_unequal, ztest_smaller_unequal, ztest_larger_unequal]:
            zstat, pval = ztest(x1, x2, value=tc.null_value,
                                alternative=alternatives[tc.alternative],
                                usevar="unequal")
            assert_allclose(zstat, tc.statistic, rtol=1e-10)
            assert_allclose(pval, tc.p_value, rtol=1e-10, atol=1e-16)

        # 1 sample test copy-paste
        d1 = self.d1
        for tc in [ztest_mu_1s, ztest_smaller_mu_1s, ztest_larger_mu_1s]:
            zstat, pval = ztest(x1, value=tc.null_value,
                                alternative=alternatives[tc.alternative])
            assert_allclose(zstat, tc.statistic, rtol=1e-10)
            assert_allclose(pval, tc.p_value, rtol=1e-10, atol=1e-16)

            zstat, pval = d1.ztest_mean(value=tc.null_value,
                                 alternative=alternatives[tc.alternative])
            assert_allclose(zstat, tc.statistic, rtol=1e-10)
            assert_allclose(pval, tc.p_value, rtol=1e-10, atol=1e-16)

            # overwrite nan in R's confint
            tc_conf_int = tc.conf_int.copy()
            if np.isnan(tc_conf_int[0]):
                tc_conf_int[0] = - np.inf
            if np.isnan(tc_conf_int[1]):
                tc_conf_int[1] = np.inf

            # Note: value is shifting our confidence interval in zconfint
            ci = zconfint(x1, value=0,
                          alternative=alternatives[tc.alternative])
            assert_allclose(ci, tc_conf_int, rtol=1e-10)

            ci = d1.zconfint_mean(alternative=alternatives[tc.alternative])
            assert_allclose(ci, tc_conf_int, rtol=1e-10)


def test_weightstats_len_1():
    x1 = [1]
    w1 = [1]
    d1 = DescrStatsW(x1, w1)
    assert (d1.quantile([0.0, 0.5, 1.0]) == 1).all()


def test_weightstats_2d_w1():
    x1 = [[1], [2]]
    w1 = [[1], [2]]
    d1 = DescrStatsW(x1, w1)
    print(len(np.array(w1).shape))
    assert (d1.quantile([0.5, 1.0]) == 2).all().all()


def test_weightstats_2d_w2():
    x1 = [[1]]
    w1 = [[1]]
    d1 = DescrStatsW(x1, w1)
    assert (d1.quantile([0, 0.5, 1.0]) == 1).all().all()
