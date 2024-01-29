import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from scipy import stats
from scipy.stats import _survival


def _kaplan_meier_reference(times, censored):
    # This is a very straightforward implementation of the Kaplan-Meier
    # estimator that does almost everything differently from the implementation
    # in stats.ecdf.

    # Begin by sorting the raw data. Note that the order of death and loss
    # at a given time matters: death happens first. See [2] page 461:
    # "These conventions may be paraphrased by saying that deaths recorded as
    # of an age t are treated as if they occurred slightly before t, and losses
    # recorded as of an age t are treated as occurring slightly after t."
    # We implement this by sorting the data first by time, then by `censored`,
    # (which is 0 when there is a death and 1 when there is only a loss).
    dtype = [('time', float), ('censored', int)]
    data = np.array([(t, d) for t, d in zip(times, censored)], dtype=dtype)
    data = np.sort(data, order=('time', 'censored'))
    times = data['time']
    died = np.logical_not(data['censored'])

    m = times.size
    n = np.arange(m, 0, -1)  # number at risk
    sf = np.cumprod((n - died) / n)

    # Find the indices of the *last* occurrence of unique times. The
    # corresponding entries of `times` and `sf` are what we want.
    _, indices = np.unique(times[::-1], return_index=True)
    ref_times = times[-indices - 1]
    ref_sf = sf[-indices - 1]
    return ref_times, ref_sf


class TestSurvival:

    @staticmethod
    def get_random_sample(rng, n_unique):
        # generate random sample
        unique_times = rng.random(n_unique)
        # convert to `np.int32` to resolve `np.repeat` failure in 32-bit CI
        repeats = rng.integers(1, 4, n_unique).astype(np.int32)
        times = rng.permuted(np.repeat(unique_times, repeats))
        censored = rng.random(size=times.size) > rng.random()
        sample = stats.CensoredData.right_censored(times, censored)
        return sample, times, censored

    def test_input_validation(self):
        message = '`sample` must be a one-dimensional sequence.'
        with pytest.raises(ValueError, match=message):
            stats.ecdf([[1]])
        with pytest.raises(ValueError, match=message):
            stats.ecdf(1)

        message = '`sample` must not contain nan'
        with pytest.raises(ValueError, match=message):
            stats.ecdf([np.nan])

        message = 'Currently, only uncensored and right-censored data...'
        with pytest.raises(NotImplementedError, match=message):
            stats.ecdf(stats.CensoredData.left_censored([1], censored=[True]))

        message = 'method` must be one of...'
        res = stats.ecdf([1, 2, 3])
        with pytest.raises(ValueError, match=message):
            res.cdf.confidence_interval(method='ekki-ekki')
        with pytest.raises(ValueError, match=message):
            res.sf.confidence_interval(method='shrubbery')

        message = 'confidence_level` must be a scalar between 0 and 1'
        with pytest.raises(ValueError, match=message):
            res.cdf.confidence_interval(-1)
        with pytest.raises(ValueError, match=message):
            res.sf.confidence_interval([0.5, 0.6])

        message = 'The confidence interval is undefined at some observations.'
        with pytest.warns(RuntimeWarning, match=message):
            ci = res.cdf.confidence_interval()

        message = 'Confidence interval bounds do not implement...'
        with pytest.raises(NotImplementedError, match=message):
            ci.low.confidence_interval()
        with pytest.raises(NotImplementedError, match=message):
            ci.high.confidence_interval()

    def test_edge_cases(self):
        res = stats.ecdf([])
        assert_equal(res.cdf.quantiles, [])
        assert_equal(res.cdf.probabilities, [])

        res = stats.ecdf([1])
        assert_equal(res.cdf.quantiles, [1])
        assert_equal(res.cdf.probabilities, [1])

    def test_unique(self):
        # Example with unique observations; `stats.ecdf` ref. [1] page 80
        sample = [6.23, 5.58, 7.06, 6.42, 5.20]
        res = stats.ecdf(sample)
        ref_x = np.sort(np.unique(sample))
        ref_cdf = np.arange(1, 6) / 5
        ref_sf = 1 - ref_cdf
        assert_equal(res.cdf.quantiles, ref_x)
        assert_equal(res.cdf.probabilities, ref_cdf)
        assert_equal(res.sf.quantiles, ref_x)
        assert_equal(res.sf.probabilities, ref_sf)

    def test_nonunique(self):
        # Example with non-unique observations; `stats.ecdf` ref. [1] page 82
        sample = [0, 2, 1, 2, 3, 4]
        res = stats.ecdf(sample)
        ref_x = np.sort(np.unique(sample))
        ref_cdf = np.array([1/6, 2/6, 4/6, 5/6, 1])
        ref_sf = 1 - ref_cdf
        assert_equal(res.cdf.quantiles, ref_x)
        assert_equal(res.cdf.probabilities, ref_cdf)
        assert_equal(res.sf.quantiles, ref_x)
        assert_equal(res.sf.probabilities, ref_sf)

    def test_evaluate_methods(self):
        # Test CDF and SF `evaluate` methods
        rng = np.random.default_rng(1162729143302572461)
        sample, _, _ = self.get_random_sample(rng, 15)
        res = stats.ecdf(sample)
        x = res.cdf.quantiles
        xr = x + np.diff(x, append=x[-1]+1)/2  # right shifted points

        assert_equal(res.cdf.evaluate(x), res.cdf.probabilities)
        assert_equal(res.cdf.evaluate(xr), res.cdf.probabilities)
        assert_equal(res.cdf.evaluate(x[0]-1), 0)  # CDF starts at 0
        assert_equal(res.cdf.evaluate([-np.inf, np.inf]), [0, 1])

        assert_equal(res.sf.evaluate(x), res.sf.probabilities)
        assert_equal(res.sf.evaluate(xr), res.sf.probabilities)
        assert_equal(res.sf.evaluate(x[0]-1), 1)  # SF starts at 1
        assert_equal(res.sf.evaluate([-np.inf, np.inf]), [1, 0])

    # ref. [1] page 91
    t1 = [37, 43, 47, 56, 60, 62, 71, 77, 80, 81]  # times
    d1 = [0, 0, 1, 1, 0, 0, 0, 1, 1, 1]  # 1 means deaths (not censored)
    r1 = [1, 1, 0.875, 0.75, 0.75, 0.75, 0.75, 0.5, 0.25, 0]  # reference SF

    # https://sphweb.bumc.bu.edu/otlt/mph-modules/bs/bs704_survival/BS704_Survival5.html
    t2 = [8, 12, 26, 14, 21, 27, 8, 32, 20, 40]
    d2 = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    r2 = [0.9, 0.788, 0.675, 0.675, 0.54, 0.405, 0.27, 0.27, 0.27]
    t3 = [33, 28, 41, 48, 48, 25, 37, 48, 25, 43]
    d3 = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    r3 = [1, 0.875, 0.75, 0.75, 0.6, 0.6, 0.6]

    # https://sphweb.bumc.bu.edu/otlt/mph-modules/bs/bs704_survival/bs704_survival4.html
    t4 = [24, 3, 11, 19, 24, 13, 14, 2, 18, 17,
          24, 21, 12, 1, 10, 23, 6, 5, 9, 17]
    d4 = [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1]
    r4 = [0.95, 0.95, 0.897, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844,
          0.844, 0.76, 0.676, 0.676, 0.676, 0.676, 0.507, 0.507]

    # https://www.real-statistics.com/survival-analysis/kaplan-meier-procedure/confidence-interval-for-the-survival-function/
    t5 = [3, 5, 8, 10, 5, 5, 8, 12, 15, 14, 2, 11, 10, 9, 12, 5, 8, 11]
    d5 = [1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1]
    r5 = [0.944, 0.889, 0.722, 0.542, 0.542, 0.542, 0.361, 0.181, 0.181, 0.181]

    @pytest.mark.parametrize("case", [(t1, d1, r1), (t2, d2, r2), (t3, d3, r3),
                                      (t4, d4, r4), (t5, d5, r5)])
    def test_right_censored_against_examples(self, case):
        # test `ecdf` against other implementations on example problems
        times, died, ref = case
        sample = stats.CensoredData.right_censored(times, np.logical_not(died))
        res = stats.ecdf(sample)
        assert_allclose(res.sf.probabilities, ref, atol=1e-3)
        assert_equal(res.sf.quantiles, np.sort(np.unique(times)))

        # test reference implementation against other implementations
        res = _kaplan_meier_reference(times, np.logical_not(died))
        assert_equal(res[0], np.sort(np.unique(times)))
        assert_allclose(res[1], ref, atol=1e-3)

    @pytest.mark.parametrize('seed', [182746786639392128, 737379171436494115,
                                      576033618403180168, 308115465002673650])
    def test_right_censored_against_reference_implementation(self, seed):
        # test `ecdf` against reference implementation on random problems
        rng = np.random.default_rng(seed)
        n_unique = rng.integers(10, 100)
        sample, times, censored = self.get_random_sample(rng, n_unique)
        res = stats.ecdf(sample)
        ref = _kaplan_meier_reference(times, censored)
        assert_allclose(res.sf.quantiles, ref[0])
        assert_allclose(res.sf.probabilities, ref[1])

        # If all observations are uncensored, the KM estimate should match
        # the usual estimate for uncensored data
        sample = stats.CensoredData(uncensored=times)
        res = _survival._ecdf_right_censored(sample)  # force Kaplan-Meier
        ref = stats.ecdf(times)
        assert_equal(res[0], ref.sf.quantiles)
        assert_allclose(res[1], ref.cdf.probabilities, rtol=1e-14)
        assert_allclose(res[2], ref.sf.probabilities, rtol=1e-14)

    def test_right_censored_ci(self):
        # test "greenwood" confidence interval against example 4 (URL above).
        times, died = self.t4, self.d4
        sample = stats.CensoredData.right_censored(times, np.logical_not(died))
        res = stats.ecdf(sample)
        ref_allowance = [0.096, 0.096, 0.135, 0.162, 0.162, 0.162, 0.162,
                         0.162, 0.162, 0.162, 0.214, 0.246, 0.246, 0.246,
                         0.246, 0.341, 0.341]

        sf_ci = res.sf.confidence_interval()
        cdf_ci = res.cdf.confidence_interval()
        allowance = res.sf.probabilities - sf_ci.low.probabilities

        assert_allclose(allowance, ref_allowance, atol=1e-3)
        assert_allclose(sf_ci.low.probabilities,
                        np.clip(res.sf.probabilities - allowance, 0, 1))
        assert_allclose(sf_ci.high.probabilities,
                        np.clip(res.sf.probabilities + allowance, 0, 1))
        assert_allclose(cdf_ci.low.probabilities,
                        np.clip(res.cdf.probabilities - allowance, 0, 1))
        assert_allclose(cdf_ci.high.probabilities,
                        np.clip(res.cdf.probabilities + allowance, 0, 1))

        # test "log-log" confidence interval against Mathematica
        # e = {24, 3, 11, 19, 24, 13, 14, 2, 18, 17, 24, 21, 12, 1, 10, 23, 6, 5,
        #      9, 17}
        # ci = {1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0}
        # R = EventData[e, ci]
        # S = SurvivalModelFit[R]
        # S["PointwiseIntervals", ConfidenceLevel->0.95,
        #   ConfidenceTransform->"LogLog"]

        ref_low = [0.694743, 0.694743, 0.647529, 0.591142, 0.591142, 0.591142,
                   0.591142, 0.591142, 0.591142, 0.591142, 0.464605, 0.370359,
                   0.370359, 0.370359, 0.370359, 0.160489, 0.160489]
        ref_high = [0.992802, 0.992802, 0.973299, 0.947073, 0.947073, 0.947073,
                    0.947073, 0.947073, 0.947073, 0.947073, 0.906422, 0.856521,
                    0.856521, 0.856521, 0.856521, 0.776724, 0.776724]
        sf_ci = res.sf.confidence_interval(method='log-log')
        assert_allclose(sf_ci.low.probabilities, ref_low, atol=1e-6)
        assert_allclose(sf_ci.high.probabilities, ref_high, atol=1e-6)

    def test_right_censored_ci_example_5(self):
        # test "exponential greenwood" confidence interval against example 5
        times, died = self.t5, self.d5
        sample = stats.CensoredData.right_censored(times, np.logical_not(died))
        res = stats.ecdf(sample)
        lower = np.array([0.66639, 0.624174, 0.456179, 0.287822, 0.287822,
                          0.287822, 0.128489, 0.030957, 0.030957, 0.030957])
        upper = np.array([0.991983, 0.970995, 0.87378, 0.739467, 0.739467,
                          0.739467, 0.603133, 0.430365, 0.430365, 0.430365])

        sf_ci = res.sf.confidence_interval(method='log-log')
        cdf_ci = res.cdf.confidence_interval(method='log-log')

        assert_allclose(sf_ci.low.probabilities, lower, atol=1e-5)
        assert_allclose(sf_ci.high.probabilities, upper, atol=1e-5)
        assert_allclose(cdf_ci.low.probabilities, 1-upper, atol=1e-5)
        assert_allclose(cdf_ci.high.probabilities, 1-lower, atol=1e-5)

        # Test against R's `survival` library `survfit` function, 90%CI
        # library(survival)
        # options(digits=16)
        # time = c(3, 5, 8, 10, 5, 5, 8, 12, 15, 14, 2, 11, 10, 9, 12, 5, 8, 11)
        # status = c(1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1)
        # res = survfit(Surv(time, status)
        # ~1, conf.type = "log-log", conf.int = 0.90)
        # res$time; res$lower; res$upper
        low = [0.74366748406861172, 0.68582332289196246, 0.50596835651480121,
               0.32913131413336727, 0.32913131413336727, 0.32913131413336727,
               0.15986912028781664, 0.04499539918147757, 0.04499539918147757,
               0.04499539918147757]
        high = [0.9890291867238429, 0.9638835422144144, 0.8560366823086629,
                0.7130167643978450, 0.7130167643978450, 0.7130167643978450,
                0.5678602982997164, 0.3887616766886558, 0.3887616766886558,
                0.3887616766886558]
        sf_ci = res.sf.confidence_interval(method='log-log',
                                           confidence_level=0.9)
        assert_allclose(sf_ci.low.probabilities, low)
        assert_allclose(sf_ci.high.probabilities, high)

        # And with conf.type = "plain"
        low = [0.8556383113628162, 0.7670478794850761, 0.5485720663578469,
               0.3441515412527123, 0.3441515412527123, 0.3441515412527123,
               0.1449184105424544, 0., 0., 0.]
        high = [1., 1., 0.8958723780865975, 0.7391817920806210,
                0.7391817920806210, 0.7391817920806210, 0.5773038116797676,
                0.3642270254596720, 0.3642270254596720, 0.3642270254596720]
        sf_ci = res.sf.confidence_interval(confidence_level=0.9)
        assert_allclose(sf_ci.low.probabilities, low)
        assert_allclose(sf_ci.high.probabilities, high)

    def test_right_censored_ci_nans(self):
        # test `ecdf` confidence interval on a problem that results in NaNs
        times, died = self.t1, self.d1
        sample = stats.CensoredData.right_censored(times, np.logical_not(died))
        res = stats.ecdf(sample)

        # Reference values generated with Matlab
        # format long
        # t = [37 43 47 56 60 62 71 77 80 81];
        # d = [0 0 1 1 0 0 0 1 1 1];
        # censored = ~d1;
        # [f, x, flo, fup] = ecdf(t, 'Censoring', censored, 'Alpha', 0.05);
        x = [37, 47, 56, 77, 80, 81]
        flo = [np.nan, 0, 0, 0.052701464070711, 0.337611126231790, np.nan]
        fup = [np.nan, 0.35417230377, 0.5500569798, 0.9472985359, 1.0, np.nan]
        i = np.searchsorted(res.cdf.quantiles, x)

        message = "The confidence interval is undefined at some observations"
        with pytest.warns(RuntimeWarning, match=message):
            ci = res.cdf.confidence_interval()

        # Matlab gives NaN as the first element of the CIs. Mathematica agrees,
        # but R's survfit does not. It makes some sense, but it's not what the
        # formula gives, so skip that element.
        assert_allclose(ci.low.probabilities[i][1:], flo[1:])
        assert_allclose(ci.high.probabilities[i][1:], fup[1:])

        # [f, x, flo, fup] = ecdf(t, 'Censoring', censored, 'Function',
        #                        'survivor', 'Alpha', 0.05);
        flo = [np.nan, 0.64582769623, 0.449943020228, 0.05270146407, 0, np.nan]
        fup = [np.nan, 1.0, 1.0, 0.947298535929289, 0.662388873768210, np.nan]
        i = np.searchsorted(res.cdf.quantiles, x)

        with pytest.warns(RuntimeWarning, match=message):
            ci = res.sf.confidence_interval()

        assert_allclose(ci.low.probabilities[i][1:], flo[1:])
        assert_allclose(ci.high.probabilities[i][1:], fup[1:])

        # With the same data, R's `survival` library `survfit` function
        # doesn't produce the leading NaN
        # library(survival)
        # options(digits=16)
        # time = c(37, 43, 47, 56, 60, 62, 71, 77, 80, 81)
        # status = c(0, 0, 1, 1, 0, 0, 0, 1, 1, 1)
        # res = survfit(Surv(time, status)
        # ~1, conf.type = "plain", conf.int = 0.95)
        # res$time
        # res$lower
        # res$upper
        low = [1., 1., 0.64582769623233816, 0.44994302022779326,
               0.44994302022779326, 0.44994302022779326, 0.44994302022779326,
               0.05270146407071086, 0., np.nan]
        high = [1., 1., 1., 1., 1., 1., 1., 0.9472985359292891,
                0.6623888737682101, np.nan]
        assert_allclose(ci.low.probabilities, low)
        assert_allclose(ci.high.probabilities, high)

        # It does with conf.type="log-log", as do we
        with pytest.warns(RuntimeWarning, match=message):
            ci = res.sf.confidence_interval(method='log-log')
        low = [np.nan, np.nan, 0.38700001403202522, 0.31480711370551911,
               0.31480711370551911, 0.31480711370551911, 0.31480711370551911,
               0.08048821148507734, 0.01049958986680601, np.nan]
        high = [np.nan, np.nan, 0.9813929658789660, 0.9308983170906275,
                0.9308983170906275, 0.9308983170906275, 0.9308983170906275,
                0.8263946341076415, 0.6558775085110887, np.nan]
        assert_allclose(ci.low.probabilities, low)
        assert_allclose(ci.high.probabilities, high)

    def test_right_censored_against_uncensored(self):
        rng = np.random.default_rng(7463952748044886637)
        sample = rng.integers(10, 100, size=1000)
        censored = np.zeros_like(sample)
        censored[np.argmax(sample)] = True
        res = stats.ecdf(sample)
        ref = stats.ecdf(stats.CensoredData.right_censored(sample, censored))
        assert_equal(res.sf.quantiles, ref.sf.quantiles)
        assert_equal(res.sf._n, ref.sf._n)
        assert_equal(res.sf._d[:-1], ref.sf._d[:-1])  # difference @ [-1]
        assert_allclose(res.sf._sf[:-1], ref.sf._sf[:-1], rtol=1e-14)

    def test_plot_iv(self):
        rng = np.random.default_rng(1769658657308472721)
        n_unique = rng.integers(10, 100)
        sample, _, _ = self.get_random_sample(rng, n_unique)
        res = stats.ecdf(sample)

        try:
            import matplotlib.pyplot as plt  # noqa: F401
            res.sf.plot()  # no other errors occur
        except (ModuleNotFoundError, ImportError):
            message = r"matplotlib must be installed to use method `plot`."
            with pytest.raises(ModuleNotFoundError, match=message):
                res.sf.plot()


class TestLogRank:

    @pytest.mark.parametrize(
        "x, y, statistic, pvalue",
        # Results validate with R
        # library(survival)
        # options(digits=16)
        #
        # futime_1 <- c(8, 12, 26, 14, 21, 27, 8, 32, 20, 40)
        # fustat_1 <- c(1, 1, 1, 1, 1, 1, 0, 0, 0, 0)
        # rx_1 <- c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        #
        # futime_2 <- c(33, 28, 41, 48, 48, 25, 37, 48, 25, 43)
        # fustat_2 <- c(1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
        # rx_2 <- c(1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
        #
        # futime <- c(futime_1, futime_2)
        # fustat <- c(fustat_1, fustat_2)
        # rx <- c(rx_1, rx_2)
        #
        # survdiff(formula = Surv(futime, fustat) ~ rx)
        #
        # Also check against another library which handle alternatives
        # library(nph)
        # logrank.test(futime, fustat, rx, alternative = "two.sided")
        # res["test"]
        [(
              # https://sphweb.bumc.bu.edu/otlt/mph-modules/bs/bs704_survival/BS704_Survival5.html
              # uncensored, censored
              [[8, 12, 26, 14, 21, 27], [8, 32, 20, 40]],
              [[33, 28, 41], [48, 48, 25, 37, 48, 25, 43]],
              # chi2, ["two-sided", "less", "greater"]
              6.91598157449,
              [0.008542873404, 0.9957285632979385, 0.004271436702061537]
         ),
         (
              # https://sphweb.bumc.bu.edu/otlt/mph-modules/bs/bs704_survival/BS704_Survival5.html
              [[19, 6, 5, 4], [20, 19, 17, 14]],
              [[16, 21, 7], [21, 15, 18, 18, 5]],
              0.835004855038,
              [0.3608293039, 0.8195853480676912, 0.1804146519323088]
         ),
         (
              # Bland, Altman, "The logrank test", BMJ, 2004
              # https://www.bmj.com/content/328/7447/1073.short
              [[6, 13, 21, 30, 37, 38, 49, 50, 63, 79, 86, 98, 202, 219],
               [31, 47, 80, 82, 82, 149]],
              [[10, 10, 12, 13, 14, 15, 16, 17, 18, 20, 24, 24, 25, 28, 30,
                33, 35, 37, 40, 40, 46, 48, 76, 81, 82, 91, 112, 181],
               [34, 40, 70]],
              7.49659416854,
              [0.006181578637, 0.003090789318730882, 0.9969092106812691]
         )]
    )
    def test_log_rank(self, x, y, statistic, pvalue):
        x = stats.CensoredData(uncensored=x[0], right=x[1])
        y = stats.CensoredData(uncensored=y[0], right=y[1])

        for i, alternative in enumerate(["two-sided", "less", "greater"]):
            res = stats.logrank(x=x, y=y, alternative=alternative)

            # we return z and use the normal distribution while other framework
            # return z**2. The p-value are directly comparable, but we have to
            # square the statistic
            assert_allclose(res.statistic**2, statistic, atol=1e-10)
            assert_allclose(res.pvalue, pvalue[i], atol=1e-10)

    def test_raises(self):
        sample = stats.CensoredData([1, 2])

        msg = r"`y` must be"
        with pytest.raises(ValueError, match=msg):
            stats.logrank(x=sample, y=[[1, 2]])

        msg = r"`x` must be"
        with pytest.raises(ValueError, match=msg):
            stats.logrank(x=[[1, 2]], y=sample)
