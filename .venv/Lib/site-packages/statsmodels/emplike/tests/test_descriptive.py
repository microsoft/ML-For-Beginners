import numpy as np
from numpy.testing import assert_almost_equal

from statsmodels.datasets import star98
from statsmodels.emplike.descriptive import DescStat

from .results.el_results import DescStatRes


class GenRes:
    """
    Reads in the data and creates class instance to be tested
    """
    @classmethod
    def setup_class(cls):
        data = star98.load()
        data.exog = np.asarray(data.exog)
        desc_stat_data = data.exog[:50, 5]
        mv_desc_stat_data = data.exog[:50, 5:7]  # mv = multivariate
        cls.res1 = DescStat(desc_stat_data)
        cls.res2 = DescStatRes()
        cls.mvres1 = DescStat(mv_desc_stat_data)


class TestDescriptiveStatistics(GenRes):
    @classmethod
    def setup_class(cls):
        super(TestDescriptiveStatistics, cls).setup_class()

    def test_test_mean(self):
        assert_almost_equal(self.res1.test_mean(14),
                            self.res2.test_mean_14, 4)

    def test_test_mean_weights(self):
        assert_almost_equal(self.res1.test_mean(14, return_weights=1)[2],
                            self.res2.test_mean_weights, 4)

    def test_ci_mean(self):
        assert_almost_equal(self.res1.ci_mean(), self.res2.ci_mean, 4)

    def test_test_var(self):
        assert_almost_equal(self.res1.test_var(3),
                            self.res2.test_var_3, 4)

    def test_test_var_weights(self):
        assert_almost_equal(self.res1.test_var(3, return_weights=1)[2],
                            self.res2.test_var_weights, 4)

    def test_ci_var(self):
        assert_almost_equal(self.res1.ci_var(), self.res2.ci_var, 4)

    def test_mv_test_mean(self):
        assert_almost_equal(self.mvres1.mv_test_mean(np.array([14, 56])),
                            self.res2.mv_test_mean, 4)

    def test_mv_test_mean_weights(self):
        assert_almost_equal(self.mvres1.mv_test_mean(np.array([14, 56]),
                                                     return_weights=1)[2],
                            self.res2.mv_test_mean_wts, 4)

    def test_test_skew(self):
        assert_almost_equal(self.res1.test_skew(0),
                            self.res2.test_skew, 4)

    def test_ci_skew(self):
        # This will be tested in a round about way since MATLAB fails when
        # computing CI with multiple nuisance parameters.  The process is:
        #
        # (1) Get CI for skewness from ci.skew()
        # (2) In MATLAB test the hypotheis that skew=results of test_skew.
        # (3) If p-value approx .05, test confirmed
        skew_ci = self.res1.ci_skew()
        lower_lim = skew_ci[0]
        upper_lim = skew_ci[1]
        ul_pval = self.res1.test_skew(lower_lim)[1]
        ll_pval = self.res1.test_skew(upper_lim)[1]
        assert_almost_equal(ul_pval, .050000, 4)
        assert_almost_equal(ll_pval, .050000, 4)

    def test_ci_skew_weights(self):
        assert_almost_equal(self.res1.test_skew(0, return_weights=1)[2],
                            self.res2.test_skew_wts, 4)

    def test_test_kurt(self):
        assert_almost_equal(self.res1.test_kurt(0),
                            self.res2.test_kurt_0, 4)

    def test_ci_kurt(self):
        # Same strategy for skewness CI
        kurt_ci = self.res1.ci_kurt(upper_bound=.5, lower_bound=-1.5)
        lower_lim = kurt_ci[0]
        upper_lim = kurt_ci[1]
        ul_pval = self.res1.test_kurt(upper_lim)[1]
        ll_pval = self.res1.test_kurt(lower_lim)[1]
        assert_almost_equal(ul_pval, .050000, 4)
        assert_almost_equal(ll_pval, .050000, 4)

    def test_joint_skew_kurt(self):
        assert_almost_equal(self.res1.test_joint_skew_kurt(0, 0),
                            self.res2.test_joint_skew_kurt, 4)

    def test_test_corr(self):
        assert_almost_equal(self.mvres1.test_corr(.5),
                            self.res2.test_corr, 4)

    def test_ci_corr(self):
        corr_ci = self.mvres1.ci_corr()
        lower_lim = corr_ci[0]
        upper_lim = corr_ci[1]
        ul_pval = self.mvres1.test_corr(upper_lim)[1]
        ll_pval = self.mvres1.test_corr(lower_lim)[1]
        assert_almost_equal(ul_pval, .050000, 4)
        assert_almost_equal(ll_pval, .050000, 4)

    def test_test_corr_weights(self):
        assert_almost_equal(self.mvres1.test_corr(.5, return_weights=1)[2],
                            self.res2.test_corr_weights, 4)
