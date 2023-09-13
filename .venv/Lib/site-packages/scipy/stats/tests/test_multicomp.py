import copy

import numpy as np
import pytest
from numpy.testing import assert_allclose

from scipy import stats
from scipy.stats._multicomp import _pvalue_dunnett, DunnettResult


class TestDunnett:
    # For the following tests, p-values were computed using Matlab, e.g.
    #     sample = [18.  15.  18.  16.  17.  15.  14.  14.  14.  15.  15....
    #               14.  15.  14.  22.  18.  21.  21.  10.  10.  11.  9....
    #               25.  26.  17.5 16.  15.5 14.5 22.  22.  24.  22.5 29....
    #               24.5 20.  18.  18.5 17.5 26.5 13.  16.5 13.  13.  13....
    #               28.  27.  34.  31.  29.  27.  24.  23.  38.  36.  25....
    #               38. 26.  22.  36.  27.  27.  32.  28.  31....
    #               24.  27.  33.  32.  28.  19. 37.  31.  36.  36....
    #               34.  38.  32.  38.  32....
    #               26.  24.  26.  25.  29. 29.5 16.5 36.  44....
    #               25.  27.  19....
    #               25.  20....
    #               28.];
    #     j = [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ...
    #          0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ...
    #          0 0 0 0...
    #          1 1 1 1 1 1 1 1 1 1 1 1 1 1 1...
    #          2 2 2 2 2 2 2 2 2...
    #          3 3 3...
    #          4 4...
    #          5];
    #     [~, ~, stats] = anova1(sample, j, "off");
    #     [results, ~, ~, gnames] = multcompare(stats, ...
    #     "CriticalValueType", "dunnett", ...
    #     "Approximate", false);
    #     tbl = array2table(results, "VariableNames", ...
    #     ["Group", "Control Group", "Lower Limit", ...
    #     "Difference", "Upper Limit", "P-value"]);
    #     tbl.("Group") = gnames(tbl.("Group"));
    #     tbl.("Control Group") = gnames(tbl.("Control Group"))

    # Matlab doesn't report the statistic, so the statistics were
    # computed using R multcomp `glht`, e.g.:
    #     library(multcomp)
    #     options(digits=16)
    #     control < - c(18.0, 15.0, 18.0, 16.0, 17.0, 15.0, 14.0, 14.0, 14.0,
    #                   15.0, 15.0, 14.0, 15.0, 14.0, 22.0, 18.0, 21.0, 21.0,
    #                   10.0, 10.0, 11.0, 9.0, 25.0, 26.0, 17.5, 16.0, 15.5,
    #                   14.5, 22.0, 22.0, 24.0, 22.5, 29.0, 24.5, 20.0, 18.0,
    #                   18.5, 17.5, 26.5, 13.0, 16.5, 13.0, 13.0, 13.0, 28.0,
    #                   27.0, 34.0, 31.0, 29.0, 27.0, 24.0, 23.0, 38.0, 36.0,
    #                   25.0, 38.0, 26.0, 22.0, 36.0, 27.0, 27.0, 32.0, 28.0,
    #                   31.0)
    #     t < - c(24.0, 27.0, 33.0, 32.0, 28.0, 19.0, 37.0, 31.0, 36.0, 36.0,
    #             34.0, 38.0, 32.0, 38.0, 32.0)
    #     w < - c(26.0, 24.0, 26.0, 25.0, 29.0, 29.5, 16.5, 36.0, 44.0)
    #     x < - c(25.0, 27.0, 19.0)
    #     y < - c(25.0, 20.0)
    #     z < - c(28.0)
    #
    #     groups = factor(rep(c("control", "t", "w", "x", "y", "z"),
    #                         times=c(length(control), length(t), length(w),
    #                                 length(x), length(y), length(z))))
    #     df < - data.frame(response=c(control, t, w, x, y, z),
    #                       group=groups)
    #     model < - aov(response
    #     ~group, data = df)
    #     test < - glht(model=model,
    #                   linfct=mcp(group="Dunnett"),
    #                   alternative="g")
    #     summary(test)
    #     confint(test)
    # p-values agreed with those produced by Matlab to at least atol=1e-3

    # From Matlab's documentation on multcompare
    samples_1 = [
        [
            24.0, 27.0, 33.0, 32.0, 28.0, 19.0, 37.0, 31.0, 36.0, 36.0,
            34.0, 38.0, 32.0, 38.0, 32.0
        ],
        [26.0, 24.0, 26.0, 25.0, 29.0, 29.5, 16.5, 36.0, 44.0],
        [25.0, 27.0, 19.0],
        [25.0, 20.0],
        [28.0]
    ]
    control_1 = [
        18.0, 15.0, 18.0, 16.0, 17.0, 15.0, 14.0, 14.0, 14.0, 15.0, 15.0,
        14.0, 15.0, 14.0, 22.0, 18.0, 21.0, 21.0, 10.0, 10.0, 11.0, 9.0,
        25.0, 26.0, 17.5, 16.0, 15.5, 14.5, 22.0, 22.0, 24.0, 22.5, 29.0,
        24.5, 20.0, 18.0, 18.5, 17.5, 26.5, 13.0, 16.5, 13.0, 13.0, 13.0,
        28.0, 27.0, 34.0, 31.0, 29.0, 27.0, 24.0, 23.0, 38.0, 36.0, 25.0,
        38.0, 26.0, 22.0, 36.0, 27.0, 27.0, 32.0, 28.0, 31.0
    ]
    pvalue_1 = [4.727e-06, 0.022346, 0.97912, 0.99953, 0.86579]  # Matlab
    # Statistic, alternative p-values, and CIs computed with R multcomp `glht`
    p_1_twosided = [1e-4, 0.02237, 0.97913, 0.99953, 0.86583]
    p_1_greater = [1e-4, 0.011217, 0.768500, 0.896991, 0.577211]
    p_1_less = [1, 1, 0.99660, 0.98398, .99953]
    statistic_1 = [5.27356, 2.91270, 0.60831, 0.27002, 0.96637]
    ci_1_twosided = [[5.3633917835622, 0.7296142201217, -8.3879817106607,
                      -11.9090753452911, -11.7655021543469],
                     [15.9709832164378, 13.8936496687672, 13.4556900439941,
                      14.6434503452911, 25.4998771543469]]
    ci_1_greater = [5.9036402398526, 1.4000632918725, -7.2754756323636,
                    -10.5567456382391, -9.8675629499576]
    ci_1_less = [15.4306165948619, 13.2230539537359, 12.3429406339544,
                 13.2908248513211, 23.6015228251660]
    pvalues_1 = dict(twosided=p_1_twosided, less=p_1_less, greater=p_1_greater)
    cis_1 = dict(twosided=ci_1_twosided, less=ci_1_less, greater=ci_1_greater)
    case_1 = dict(samples=samples_1, control=control_1, statistic=statistic_1,
                  pvalues=pvalues_1, cis=cis_1)

    # From Dunnett1955 comparing with R's DescTools: DunnettTest
    samples_2 = [[9.76, 8.80, 7.68, 9.36], [12.80, 9.68, 12.16, 9.20, 10.55]]
    control_2 = [7.40, 8.50, 7.20, 8.24, 9.84, 8.32]
    pvalue_2 = [0.6201, 0.0058]
    # Statistic, alternative p-values, and CIs computed with R multcomp `glht`
    p_2_twosided = [0.6201020, 0.0058254]
    p_2_greater = [0.3249776, 0.0029139]
    p_2_less = [0.91676, 0.99984]
    statistic_2 = [0.85703, 3.69375]
    ci_2_twosided = [[-1.2564116462124, 0.8396273539789],
                     [2.5564116462124, 4.4163726460211]]
    ci_2_greater = [-0.9588591188156, 1.1187563667543]
    ci_2_less = [2.2588591188156, 4.1372436332457]
    pvalues_2 = dict(twosided=p_2_twosided, less=p_2_less, greater=p_2_greater)
    cis_2 = dict(twosided=ci_2_twosided, less=ci_2_less, greater=ci_2_greater)
    case_2 = dict(samples=samples_2, control=control_2, statistic=statistic_2,
                  pvalues=pvalues_2, cis=cis_2)

    samples_3 = [[55, 64, 64], [55, 49, 52], [50, 44, 41]]
    control_3 = [55, 47, 48]
    pvalue_3 = [0.0364, 0.8966, 0.4091]
    # Statistic, alternative p-values, and CIs computed with R multcomp `glht`
    p_3_twosided = [0.036407, 0.896539, 0.409295]
    p_3_greater = [0.018277, 0.521109, 0.981892]
    p_3_less = [0.99944, 0.90054, 0.20974]
    statistic_3 = [3.09073, 0.56195, -1.40488]
    ci_3_twosided = [[0.7529028025053, -8.2470971974947, -15.2470971974947],
                     [21.2470971974947, 12.2470971974947, 5.2470971974947]]
    ci_3_greater = [2.4023682323149, -6.5976317676851, -13.5976317676851]
    ci_3_less = [19.5984402363662, 10.5984402363662, 3.5984402363662]
    pvalues_3 = dict(twosided=p_3_twosided, less=p_3_less, greater=p_3_greater)
    cis_3 = dict(twosided=ci_3_twosided, less=ci_3_less, greater=ci_3_greater)
    case_3 = dict(samples=samples_3, control=control_3, statistic=statistic_3,
                  pvalues=pvalues_3, cis=cis_3)

    # From Thomson and Short,
    # Mucociliary function in health, chronic obstructive airway disease,
    # and asbestosis, Journal of Applied Physiology, 1969. Table 1
    # Comparing with R's DescTools: DunnettTest
    samples_4 = [[3.8, 2.7, 4.0, 2.4], [2.8, 3.4, 3.7, 2.2, 2.0]]
    control_4 = [2.9, 3.0, 2.5, 2.6, 3.2]
    pvalue_4 = [0.5832, 0.9982]
    # Statistic, alternative p-values, and CIs computed with R multcomp `glht`
    p_4_twosided = [0.58317, 0.99819]
    p_4_greater = [0.30225, 0.69115]
    p_4_less = [0.91929, 0.65212]
    statistic_4 = [0.90875, -0.05007]
    ci_4_twosided = [[-0.6898153448579, -1.0333456251632],
                     [1.4598153448579, 0.9933456251632]]
    ci_4_greater = [-0.5186459268412, -0.8719655502147 ]
    ci_4_less = [1.2886459268412, 0.8319655502147]
    pvalues_4 = dict(twosided=p_4_twosided, less=p_4_less, greater=p_4_greater)
    cis_4 = dict(twosided=ci_4_twosided, less=ci_4_less, greater=ci_4_greater)
    case_4 = dict(samples=samples_4, control=control_4, statistic=statistic_4,
                  pvalues=pvalues_4, cis=cis_4)

    @pytest.mark.parametrize(
        'rho, n_groups, df, statistic, pvalue, alternative',
        [
            # From Dunnett1955
            # Tables 1a and 1b pages 1117-1118
            (0.5, 1, 10, 1.81, 0.05, "greater"),  # different than two-sided
            (0.5, 3, 10, 2.34, 0.05, "greater"),
            (0.5, 2, 30, 1.99, 0.05, "greater"),
            (0.5, 5, 30, 2.33, 0.05, "greater"),
            (0.5, 4, 12, 3.32, 0.01, "greater"),
            (0.5, 7, 12, 3.56, 0.01, "greater"),
            (0.5, 2, 60, 2.64, 0.01, "greater"),
            (0.5, 4, 60, 2.87, 0.01, "greater"),
            (0.5, 4, 60, [2.87, 2.21], [0.01, 0.05], "greater"),
            # Tables 2a and 2b pages 1119-1120
            (0.5, 1, 10, 2.23, 0.05, "two-sided"),  # two-sided
            (0.5, 3, 10, 2.81, 0.05, "two-sided"),
            (0.5, 2, 30, 2.32, 0.05, "two-sided"),
            (0.5, 3, 20, 2.57, 0.05, "two-sided"),
            (0.5, 4, 12, 3.76, 0.01, "two-sided"),
            (0.5, 7, 12, 4.08, 0.01, "two-sided"),
            (0.5, 2, 60, 2.90, 0.01, "two-sided"),
            (0.5, 4, 60, 3.14, 0.01, "two-sided"),
            (0.5, 4, 60, [3.14, 2.55], [0.01, 0.05], "two-sided"),
        ],
    )
    def test_critical_values(
        self, rho, n_groups, df, statistic, pvalue, alternative
    ):
        rng = np.random.default_rng(165250594791731684851746311027739134893)
        rho = np.full((n_groups, n_groups), rho)
        np.fill_diagonal(rho, 1)

        statistic = np.array(statistic)
        res = _pvalue_dunnett(
            rho=rho, df=df, statistic=statistic,
            alternative=alternative,
            rng=rng
        )
        assert_allclose(res, pvalue, atol=5e-3)

    @pytest.mark.parametrize(
        'samples, control, pvalue, statistic',
        [
            (samples_1, control_1, pvalue_1, statistic_1),
            (samples_2, control_2, pvalue_2, statistic_2),
            (samples_3, control_3, pvalue_3, statistic_3),
            (samples_4, control_4, pvalue_4, statistic_4),
        ]
    )
    def test_basic(self, samples, control, pvalue, statistic):
        rng = np.random.default_rng(11681140010308601919115036826969764808)

        res = stats.dunnett(*samples, control=control, random_state=rng)

        assert isinstance(res, DunnettResult)
        assert_allclose(res.statistic, statistic, rtol=5e-5)
        assert_allclose(res.pvalue, pvalue, rtol=1e-2, atol=1e-4)

    @pytest.mark.parametrize(
        'alternative',
        ['two-sided', 'less', 'greater']
    )
    def test_ttest_ind(self, alternative):
        # check that `dunnett` agrees with `ttest_ind`
        # when there are only two groups
        rng = np.random.default_rng(114184017807316971636137493526995620351)

        for _ in range(10):
            sample = rng.integers(-100, 100, size=(10,))
            control = rng.integers(-100, 100, size=(10,))

            res = stats.dunnett(
                sample, control=control,
                alternative=alternative, random_state=rng
            )
            ref = stats.ttest_ind(
                sample, control,
                alternative=alternative, random_state=rng
            )

            assert_allclose(res.statistic, ref.statistic, rtol=1e-3, atol=1e-5)
            assert_allclose(res.pvalue, ref.pvalue, rtol=1e-3, atol=1e-5)

    @pytest.mark.parametrize(
        'alternative, pvalue',
        [
            ('less', [0, 1]),
            ('greater', [1, 0]),
            ('two-sided', [0, 0]),
        ]
    )
    def test_alternatives(self, alternative, pvalue):
        rng = np.random.default_rng(114184017807316971636137493526995620351)

        # width of 20 and min diff between samples/control is 60
        # and maximal diff would be 100
        sample_less = rng.integers(0, 20, size=(10,))
        control = rng.integers(80, 100, size=(10,))
        sample_greater = rng.integers(160, 180, size=(10,))

        res = stats.dunnett(
            sample_less, sample_greater, control=control,
            alternative=alternative, random_state=rng
        )
        assert_allclose(res.pvalue, pvalue, atol=1e-7)

        ci = res.confidence_interval()
        # two-sided is comparable for high/low
        if alternative == 'less':
            assert np.isneginf(ci.low).all()
            assert -100 < ci.high[0] < -60
            assert 60 < ci.high[1] < 100
        elif alternative == 'greater':
            assert -100 < ci.low[0] < -60
            assert 60 < ci.low[1] < 100
            assert np.isposinf(ci.high).all()
        elif alternative == 'two-sided':
            assert -100 < ci.low[0] < -60
            assert 60 < ci.low[1] < 100
            assert -100 < ci.high[0] < -60
            assert 60 < ci.high[1] < 100

    @pytest.mark.parametrize("case", [case_1, case_2, case_3, case_4])
    @pytest.mark.parametrize("alternative", ['less', 'greater', 'two-sided'])
    def test_against_R_multicomp_glht(self, case, alternative):
        rng = np.random.default_rng(189117774084579816190295271136455278291)
        samples = case['samples']
        control = case['control']
        alternatives = {'less': 'less', 'greater': 'greater',
                        'two-sided': 'twosided'}
        p_ref = case['pvalues'][alternative.replace('-', '')]

        res = stats.dunnett(*samples, control=control, alternative=alternative,
                            random_state=rng)
        # atol can't be tighter because R reports some pvalues as "< 1e-4"
        assert_allclose(res.pvalue, p_ref, rtol=5e-3, atol=1e-4)

        ci_ref = case['cis'][alternatives[alternative]]
        if alternative == "greater":
            ci_ref = [ci_ref, np.inf]
        elif alternative == "less":
            ci_ref = [-np.inf, ci_ref]
        assert res._ci is None
        assert res._ci_cl is None
        ci = res.confidence_interval(confidence_level=0.95)
        assert_allclose(ci.low, ci_ref[0], rtol=5e-3, atol=1e-5)
        assert_allclose(ci.high, ci_ref[1], rtol=5e-3, atol=1e-5)

        # re-run to use the cached value "is" to check id as same object
        assert res._ci is ci
        assert res._ci_cl == 0.95
        ci_ = res.confidence_interval(confidence_level=0.95)
        assert ci_ is ci

    @pytest.mark.parametrize('alternative', ["two-sided", "less", "greater"])
    def test_str(self, alternative):
        rng = np.random.default_rng(189117774084579816190295271136455278291)

        res = stats.dunnett(
            *self.samples_3, control=self.control_3, alternative=alternative,
            random_state=rng
        )

        # check some str output
        res_str = str(res)
        assert '(Sample 2 - Control)' in res_str
        assert '95.0%' in res_str

        if alternative == 'less':
            assert '-inf' in res_str
            assert '19.' in res_str
        elif alternative == 'greater':
            assert 'inf' in res_str
            assert '-13.' in res_str
        else:
            assert 'inf' not in res_str
            assert '21.' in res_str

    def test_warnings(self):
        rng = np.random.default_rng(189117774084579816190295271136455278291)

        res = stats.dunnett(
            *self.samples_3, control=self.control_3, random_state=rng
        )
        msg = r"Computation of the confidence interval did not converge"
        with pytest.warns(UserWarning, match=msg):
            res._allowance(tol=1e-5)

    def test_raises(self):
        samples, control = self.samples_3, self.control_3

        # alternative
        with pytest.raises(ValueError, match="alternative must be"):
            stats.dunnett(*samples, control=control, alternative='bob')

        # 2D for a sample
        samples_ = copy.deepcopy(samples)
        samples_[0] = [samples_[0]]
        with pytest.raises(ValueError, match="must be 1D arrays"):
            stats.dunnett(*samples_, control=control)

        # 2D for control
        control_ = copy.deepcopy(control)
        control_ = [control_]
        with pytest.raises(ValueError, match="must be 1D arrays"):
            stats.dunnett(*samples, control=control_)

        # No obs in a sample
        samples_ = copy.deepcopy(samples)
        samples_[1] = []
        with pytest.raises(ValueError, match="at least 1 observation"):
            stats.dunnett(*samples_, control=control)

        # No obs in control
        control_ = []
        with pytest.raises(ValueError, match="at least 1 observation"):
            stats.dunnett(*samples, control=control_)

        res = stats.dunnett(*samples, control=control)
        with pytest.raises(ValueError, match="Confidence level must"):
            res.confidence_interval(confidence_level=3)

    @pytest.mark.filterwarnings("ignore:Computation of the confidence")
    @pytest.mark.parametrize('n_samples', [1, 2, 3])
    def test_shapes(self, n_samples):
        rng = np.random.default_rng(689448934110805334)
        samples = rng.normal(size=(n_samples, 10))
        control = rng.normal(size=10)
        res = stats.dunnett(*samples, control=control, random_state=rng)
        assert res.statistic.shape == (n_samples,)
        assert res.pvalue.shape == (n_samples,)
        ci = res.confidence_interval()
        assert ci.low.shape == (n_samples,)
        assert ci.high.shape == (n_samples,)
