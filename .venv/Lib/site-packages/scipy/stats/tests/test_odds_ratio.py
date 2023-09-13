import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from .._discrete_distns import nchypergeom_fisher, hypergeom
from scipy.stats._odds_ratio import odds_ratio
from .data.fisher_exact_results_from_r import data


class TestOddsRatio:

    @pytest.mark.parametrize('parameters, rresult', data)
    def test_results_from_r(self, parameters, rresult):
        alternative = parameters.alternative.replace('.', '-')
        result = odds_ratio(parameters.table)
        # The results computed by R are not very accurate.
        if result.statistic < 400:
            or_rtol = 5e-4
            ci_rtol = 2e-2
        else:
            or_rtol = 5e-2
            ci_rtol = 1e-1
        assert_allclose(result.statistic,
                        rresult.conditional_odds_ratio, rtol=or_rtol)
        ci = result.confidence_interval(parameters.confidence_level,
                                        alternative)
        assert_allclose((ci.low, ci.high), rresult.conditional_odds_ratio_ci,
                        rtol=ci_rtol)

        # Also do a self-check for the conditional odds ratio.
        # With the computed conditional odds ratio as the noncentrality
        # parameter of the noncentral hypergeometric distribution with
        # parameters table.sum(), table[0].sum(), and table[:,0].sum() as
        # total, ngood and nsample, respectively, the mean of the distribution
        # should equal table[0, 0].
        cor = result.statistic
        table = np.array(parameters.table)
        total = table.sum()
        ngood = table[0].sum()
        nsample = table[:, 0].sum()
        # nchypergeom_fisher does not allow the edge cases where the
        # noncentrality parameter is 0 or inf, so handle those values
        # separately here.
        if cor == 0:
            nchg_mean = hypergeom.support(total, ngood, nsample)[0]
        elif cor == np.inf:
            nchg_mean = hypergeom.support(total, ngood, nsample)[1]
        else:
            nchg_mean = nchypergeom_fisher.mean(total, ngood, nsample, cor)
        assert_allclose(nchg_mean, table[0, 0], rtol=1e-13)

        # Check that the confidence interval is correct.
        alpha = 1 - parameters.confidence_level
        if alternative == 'two-sided':
            if ci.low > 0:
                sf = nchypergeom_fisher.sf(table[0, 0] - 1,
                                           total, ngood, nsample, ci.low)
                assert_allclose(sf, alpha/2, rtol=1e-11)
            if np.isfinite(ci.high):
                cdf = nchypergeom_fisher.cdf(table[0, 0],
                                             total, ngood, nsample, ci.high)
                assert_allclose(cdf, alpha/2, rtol=1e-11)
        elif alternative == 'less':
            if np.isfinite(ci.high):
                cdf = nchypergeom_fisher.cdf(table[0, 0],
                                             total, ngood, nsample, ci.high)
                assert_allclose(cdf, alpha, rtol=1e-11)
        else:
            # alternative == 'greater'
            if ci.low > 0:
                sf = nchypergeom_fisher.sf(table[0, 0] - 1,
                                           total, ngood, nsample, ci.low)
                assert_allclose(sf, alpha, rtol=1e-11)

    @pytest.mark.parametrize('table', [
        [[0, 0], [5, 10]],
        [[5, 10], [0, 0]],
        [[0, 5], [0, 10]],
        [[5, 0], [10, 0]],
    ])
    def test_row_or_col_zero(self, table):
        result = odds_ratio(table)
        assert_equal(result.statistic, np.nan)
        ci = result.confidence_interval()
        assert_equal((ci.low, ci.high), (0, np.inf))

    @pytest.mark.parametrize("case",
                             [[0.95, 'two-sided', 0.4879913, 2.635883],
                              [0.90, 'two-sided', 0.5588516, 2.301663]])
    def test_sample_odds_ratio_ci(self, case):
        # Compare the sample odds ratio confidence interval to the R function
        # oddsratio.wald from the epitools package, e.g.
        # > library(epitools)
        # > table = matrix(c(10, 20, 41, 93), nrow=2, ncol=2, byrow=TRUE)
        # > result = oddsratio.wald(table)
        # > result$measure
        #           odds ratio with 95% C.I.
        # Predictor  estimate     lower    upper
        #   Exposed1 1.000000        NA       NA
        #   Exposed2 1.134146 0.4879913 2.635883

        confidence_level, alternative, ref_low, ref_high = case
        table = [[10, 20], [41, 93]]
        result = odds_ratio(table, kind='sample')
        assert_allclose(result.statistic, 1.134146, rtol=1e-6)
        ci = result.confidence_interval(confidence_level, alternative)
        assert_allclose([ci.low, ci.high], [ref_low, ref_high], rtol=1e-6)

    @pytest.mark.parametrize('alternative', ['less', 'greater', 'two-sided'])
    def test_sample_odds_ratio_one_sided_ci(self, alternative):
        # can't find a good reference for one-sided CI, so bump up the sample
        # size and compare against the conditional odds ratio CI
        table = [[1000, 2000], [4100, 9300]]
        res = odds_ratio(table, kind='sample')
        ref = odds_ratio(table, kind='conditional')
        assert_allclose(res.statistic, ref.statistic, atol=1e-5)
        assert_allclose(res.confidence_interval(alternative=alternative),
                        ref.confidence_interval(alternative=alternative),
                        atol=2e-3)

    @pytest.mark.parametrize('kind', ['sample', 'conditional'])
    @pytest.mark.parametrize('bad_table', [123, "foo", [10, 11, 12]])
    def test_invalid_table_shape(self, kind, bad_table):
        with pytest.raises(ValueError, match="Invalid shape"):
            odds_ratio(bad_table, kind=kind)

    def test_invalid_table_type(self):
        with pytest.raises(ValueError, match='must be an array of integers'):
            odds_ratio([[1.0, 3.4], [5.0, 9.9]])

    def test_negative_table_values(self):
        with pytest.raises(ValueError, match='must be nonnegative'):
            odds_ratio([[1, 2], [3, -4]])

    def test_invalid_kind(self):
        with pytest.raises(ValueError, match='`kind` must be'):
            odds_ratio([[10, 20], [30, 14]], kind='magnetoreluctance')

    def test_invalid_alternative(self):
        result = odds_ratio([[5, 10], [2, 32]])
        with pytest.raises(ValueError, match='`alternative` must be'):
            result.confidence_interval(alternative='depleneration')

    @pytest.mark.parametrize('level', [-0.5, 1.5])
    def test_invalid_confidence_level(self, level):
        result = odds_ratio([[5, 10], [2, 32]])
        with pytest.raises(ValueError, match='must be between 0 and 1'):
            result.confidence_interval(confidence_level=level)
