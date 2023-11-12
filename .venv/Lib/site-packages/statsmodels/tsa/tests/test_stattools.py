from statsmodels.compat.numpy import lstsq
from statsmodels.compat.pandas import assert_index_equal
from statsmodels.compat.platform import PLATFORM_WIN
from statsmodels.compat.python import lrange

import os
import warnings

import numpy as np
from numpy.testing import (
    assert_,
    assert_allclose,
    assert_almost_equal,
    assert_equal,
    assert_raises,
)
import pandas as pd
from pandas import DataFrame, Series, date_range
import pytest
from scipy.interpolate import interp1d

from statsmodels.datasets import macrodata, modechoice, nile, randhie, sunspots
from statsmodels.tools.sm_exceptions import (
    CollinearityWarning,
    InfeasibleTestError,
    InterpolationWarning,
    MissingDataError,
    ValueWarning,
)
# Remove imports when range unit root test gets an R implementation
from statsmodels.tools.validation import array_like, bool_like
from statsmodels.tsa.arima_process import arma_acovf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import (
    acf,
    acovf,
    adfuller,
    arma_order_select_ic,
    breakvar_heteroskedasticity_test,
    ccovf,
    coint,
    grangercausalitytests,
    innovations_algo,
    innovations_filter,
    kpss,
    levinson_durbin,
    levinson_durbin_pacf,
    pacf,
    pacf_burg,
    pacf_ols,
    pacf_yw,
    range_unit_root_test,
    zivot_andrews,
)

DECIMAL_8 = 8
DECIMAL_6 = 6
DECIMAL_5 = 5
DECIMAL_4 = 4
DECIMAL_3 = 3
DECIMAL_2 = 2
DECIMAL_1 = 1

CURR_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="module")
def acovf_data():
    rnd = np.random.RandomState(12345)
    return rnd.randn(250)


@pytest.fixture(scope="module")
def gc_data():
    mdata = macrodata.load_pandas().data
    mdata = mdata[["realgdp", "realcons"]].values
    data = mdata.astype(float)
    return np.diff(np.log(data), axis=0)


class CheckADF:
    """
    Test Augmented Dickey-Fuller

    Test values taken from Stata.
    """

    levels = ["1%", "5%", "10%"]
    data = macrodata.load_pandas()
    x = data.data["realgdp"].values
    y = data.data["infl"].values

    def test_teststat(self):
        assert_almost_equal(self.res1[0], self.teststat, DECIMAL_5)

    def test_pvalue(self):
        assert_almost_equal(self.res1[1], self.pvalue, DECIMAL_5)

    def test_critvalues(self):
        critvalues = [self.res1[4][lev] for lev in self.levels]
        assert_almost_equal(critvalues, self.critvalues, DECIMAL_2)


class TestADFConstant(CheckADF):
    """
    Dickey-Fuller test for unit root
    """

    @classmethod
    def setup_class(cls):
        cls.res1 = adfuller(cls.x, regression="c", autolag=None, maxlag=4)
        cls.teststat = 0.97505319
        cls.pvalue = 0.99399563
        cls.critvalues = [-3.476, -2.883, -2.573]


class TestADFConstantTrend(CheckADF):
    """"""

    @classmethod
    def setup_class(cls):
        cls.res1 = adfuller(cls.x, regression="ct", autolag=None, maxlag=4)
        cls.teststat = -1.8566374
        cls.pvalue = 0.67682968
        cls.critvalues = [-4.007, -3.437, -3.137]


# FIXME: do not leave commented-out
# class TestADFConstantTrendSquared(CheckADF):
#    """
#    """
#    pass
# TODO: get test values from R?


class TestADFNoConstant(CheckADF):
    """"""

    @classmethod
    def setup_class(cls):
        cls.res1 = adfuller(cls.x, regression="n", autolag=None, maxlag=4)
        cls.teststat = 3.5227498

        cls.pvalue = 0.99999
        # Stata does not return a p-value for noconstant.
        # Tau^max in MacKinnon (1994) is missing, so it is
        # assumed that its right-tail is well-behaved

        cls.critvalues = [-2.587, -1.950, -1.617]


# No Unit Root


class TestADFConstant2(CheckADF):
    @classmethod
    def setup_class(cls):
        cls.res1 = adfuller(cls.y, regression="c", autolag=None, maxlag=1)
        cls.teststat = -4.3346988
        cls.pvalue = 0.00038661
        cls.critvalues = [-3.476, -2.883, -2.573]


class TestADFConstantTrend2(CheckADF):
    @classmethod
    def setup_class(cls):
        cls.res1 = adfuller(cls.y, regression="ct", autolag=None, maxlag=1)
        cls.teststat = -4.425093
        cls.pvalue = 0.00199633
        cls.critvalues = [-4.006, -3.437, -3.137]


class TestADFNoConstant2(CheckADF):
    @classmethod
    def setup_class(cls):
        cls.res1 = adfuller(cls.y, regression="n", autolag=None, maxlag=1)
        cls.teststat = -2.4511596
        cls.pvalue = 0.013747
        # Stata does not return a p-value for noconstant
        # this value is just taken from our results
        cls.critvalues = [-2.587, -1.950, -1.617]
        _, _1, _2, cls.store = adfuller(
            cls.y, regression="n", autolag=None, maxlag=1, store=True
        )

    def test_store_str(self):
        assert_equal(
            self.store.__str__(), "Augmented Dickey-Fuller Test Results"
        )


@pytest.mark.parametrize("x", [np.full(8, 5.0)])
def test_adfuller_resid_variance_zero(x):
    with pytest.raises(ValueError):
        adfuller(x)


class CheckCorrGram:
    """
    Set up for ACF, PACF tests.
    """

    data = macrodata.load_pandas()
    x = data.data["realgdp"]
    filename = os.path.join(CURR_DIR, "results", "results_corrgram.csv")
    results = pd.read_csv(filename, delimiter=",")


class TestACF(CheckCorrGram):
    """
    Test Autocorrelation Function
    """

    @classmethod
    def setup_class(cls):
        cls.acf = cls.results["acvar"]
        # cls.acf = np.concatenate(([1.], cls.acf))
        cls.qstat = cls.results["Q1"]
        cls.res1 = acf(cls.x, nlags=40, qstat=True, alpha=0.05, fft=False)
        cls.confint_res = cls.results[["acvar_lb", "acvar_ub"]].values

    def test_acf(self):
        assert_almost_equal(self.res1[0][1:41], self.acf, DECIMAL_8)

    def test_confint(self):
        centered = self.res1[1] - self.res1[1].mean(1)[:, None]
        assert_almost_equal(centered[1:41], self.confint_res, DECIMAL_8)

    def test_qstat(self):
        assert_almost_equal(self.res1[2][:40], self.qstat, DECIMAL_3)
        # 3 decimal places because of stata rounding

    # FIXME: enable/xfail/skip or delete
    # def pvalue(self):
    #    pass
    # NOTE: should not need testing if Q stat is correct


class TestACF_FFT(CheckCorrGram):
    # Test Autocorrelation Function using FFT
    @classmethod
    def setup_class(cls):
        cls.acf = cls.results["acvarfft"]
        cls.qstat = cls.results["Q1"]
        cls.res1 = acf(cls.x, nlags=40, qstat=True, fft=True)

    def test_acf(self):
        assert_almost_equal(self.res1[0][1:], self.acf, DECIMAL_8)

    def test_qstat(self):
        # todo why is res1/qstat 1 short
        assert_almost_equal(self.res1[1], self.qstat, DECIMAL_3)


class TestACFMissing(CheckCorrGram):
    # Test Autocorrelation Function using Missing
    @classmethod
    def setup_class(cls):
        cls.x = np.concatenate((np.array([np.nan]), cls.x))
        cls.acf = cls.results["acvar"]  # drop and conservative
        cls.qstat = cls.results["Q1"]
        cls.res_drop = acf(
            cls.x, nlags=40, qstat=True, alpha=0.05, missing="drop", fft=False
        )
        cls.res_conservative = acf(
            cls.x,
            nlags=40,
            qstat=True,
            alpha=0.05,
            fft=False,
            missing="conservative",
        )
        cls.acf_none = np.empty(40) * np.nan  # lags 1 to 40 inclusive
        cls.qstat_none = np.empty(40) * np.nan
        cls.res_none = acf(
            cls.x, nlags=40, qstat=True, alpha=0.05, missing="none", fft=False
        )

    def test_raise(self):
        with pytest.raises(MissingDataError):
            acf(
                self.x,
                nlags=40,
                qstat=True,
                fft=False,
                alpha=0.05,
                missing="raise",
            )

    def test_acf_none(self):
        assert_almost_equal(self.res_none[0][1:41], self.acf_none, DECIMAL_8)

    def test_acf_drop(self):
        assert_almost_equal(self.res_drop[0][1:41], self.acf, DECIMAL_8)

    def test_acf_conservative(self):
        assert_almost_equal(
            self.res_conservative[0][1:41], self.acf, DECIMAL_8
        )

    def test_qstat_none(self):
        # todo why is res1/qstat 1 short
        assert_almost_equal(self.res_none[2], self.qstat_none, DECIMAL_3)


# FIXME: enable/xfail/skip or delete
# how to do this test? the correct q_stat depends on whether nobs=len(x) is
# used when x contains NaNs or whether nobs<len(x) when x contains NaNs
#    def test_qstat_drop(self):
#        assert_almost_equal(self.res_drop[2][:40], self.qstat, DECIMAL_3)


class TestPACF(CheckCorrGram):
    @classmethod
    def setup_class(cls):
        cls.pacfols = cls.results["PACOLS"]
        cls.pacfyw = cls.results["PACYW"]

    def test_ols(self):
        pacfols, confint = pacf(self.x, nlags=40, alpha=0.05, method="ols")
        assert_almost_equal(pacfols[1:], self.pacfols, DECIMAL_6)
        centered = confint - confint.mean(1)[:, None]
        # from edited Stata ado file
        res = [[-0.1375625, 0.1375625]] * 40
        assert_almost_equal(centered[1:41], res, DECIMAL_6)
        # check lag 0
        assert_equal(centered[0], [0.0, 0.0])
        assert_equal(confint[0], [1, 1])
        assert_equal(pacfols[0], 1)

    def test_ols_inefficient(self):
        lag_len = 5
        pacfols = pacf_ols(self.x, nlags=lag_len, efficient=False)
        x = self.x.copy()
        x -= x.mean()
        n = x.shape[0]
        lags = np.zeros((n - 5, 5))
        lead = x[5:]
        direct = np.empty(lag_len + 1)
        direct[0] = 1.0
        for i in range(lag_len):
            lags[:, i] = x[5 - (i + 1) : -(i + 1)]
            direct[i + 1] = lstsq(lags[:, : (i + 1)], lead, rcond=None)[0][-1]
        assert_allclose(pacfols, direct, atol=1e-8)

    def test_yw(self):
        pacfyw = pacf_yw(self.x, nlags=40, method="mle")
        assert_almost_equal(pacfyw[1:], self.pacfyw, DECIMAL_8)

    def test_yw_singular(self):
        with pytest.warns(ValueWarning):
            pacf(np.ones(30), nlags=6)

    def test_ld(self):
        pacfyw = pacf_yw(self.x, nlags=40, method="mle")
        pacfld = pacf(self.x, nlags=40, method="ldb")
        assert_almost_equal(pacfyw, pacfld, DECIMAL_8)

        pacfyw = pacf(self.x, nlags=40, method="yw")
        pacfld = pacf(self.x, nlags=40, method="lda")
        assert_almost_equal(pacfyw, pacfld, DECIMAL_8)

    def test_burg(self):
        pacfburg_, _ = pacf_burg(self.x, nlags=40)
        pacfburg = pacf(self.x, nlags=40, method="burg")
        assert_almost_equal(pacfburg_, pacfburg, DECIMAL_8)

class TestBreakvarHeteroskedasticityTest:
    from scipy.stats import chi2, f

    def test_1d_input(self):

        input_residuals = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        expected_statistic = (4.0 ** 2 + 5.0 ** 2) / (0.0 ** 2 + 1.0 ** 2)
        # ~ F(2, 2), two-sided test
        expected_pvalue = 2 * min(
            self.f.cdf(expected_statistic, 2, 2),
            self.f.sf(expected_statistic, 2, 2),
        )
        actual_statistic, actual_pvalue = breakvar_heteroskedasticity_test(
            input_residuals
        )

        assert actual_statistic == expected_statistic
        assert actual_pvalue == expected_pvalue

    def test_2d_input_with_missing_values(self):

        input_residuals = np.array(
            [
                [0.0, 0.0, np.nan],
                [1.0, np.nan, 1.0],
                [2.0, 2.0, np.nan],
                [3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0],
                [5.0, 5.0, 5.0],
                [6.0, 6.0, 6.0],
                [7.0, 7.0, 7.0],
                [8.0, 8.0, 8.0],
            ]
        )
        expected_statistic = np.array(
            [
                (8.0 ** 2 + 7.0 ** 2 + 6.0 ** 2)
                / (0.0 ** 2 + 1.0 ** 2 + 2.0 ** 2),
                (8.0 ** 2 + 7.0 ** 2 + 6.0 ** 2) / (0.0 ** 2 + 2.0 ** 2),
                np.nan,
            ]
        )
        expected_pvalue = np.array(
            [
                2
                * min(
                    self.f.cdf(expected_statistic[0], 3, 3),
                    self.f.sf(expected_statistic[0], 3, 3),
                ),
                2
                * min(
                    self.f.cdf(expected_statistic[1], 3, 2),
                    self.f.sf(expected_statistic[1], 3, 2),
                ),
                np.nan,
            ]
        )
        actual_statistic, actual_pvalue = breakvar_heteroskedasticity_test(
            input_residuals
        )

        assert_equal(actual_statistic, expected_statistic)
        assert_equal(actual_pvalue, expected_pvalue)

    @pytest.mark.parametrize(
        "subset_length,expected_statistic,expected_pvalue",
        [
            (2, 41, 2 * min(f.cdf(41, 2, 2), f.sf(41, 2, 2))),
            (0.5, 10, 2 * min(f.cdf(10, 3, 3), f.sf(10, 3, 3))),
        ],
    )
    def test_subset_length(
        self, subset_length, expected_statistic, expected_pvalue
    ):

        input_residuals = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        actual_statistic, actual_pvalue = breakvar_heteroskedasticity_test(
            input_residuals,
            subset_length=subset_length,
        )

        assert actual_statistic == expected_statistic
        assert actual_pvalue == expected_pvalue

    @pytest.mark.parametrize(
        "alternative,expected_statistic,expected_pvalue",
        [
            ("two-sided", 41, 2 * min(f.cdf(41, 2, 2), f.sf(41, 2, 2))),
            ("decreasing", 1 / 41, f.sf(1 / 41, 2, 2)),
            ("increasing", 41, f.sf(41, 2, 2)),
        ],
    )
    def test_alternative(
        self, alternative, expected_statistic, expected_pvalue
    ):

        input_residuals = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        actual_statistic, actual_pvalue = breakvar_heteroskedasticity_test(
            input_residuals,
            alternative=alternative,
        )
        assert actual_statistic == expected_statistic
        assert actual_pvalue == expected_pvalue

    def test_use_chi2(self):

        input_residuals = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        expected_statistic = (4.0 ** 2 + 5.0 ** 2) / (0.0 ** 2 + 1.0 ** 2)
        expected_pvalue = 2 * min(
            self.chi2.cdf(2 * expected_statistic, 2),
            self.chi2.sf(2 * expected_statistic, 2),
        )
        actual_statistic, actual_pvalue = breakvar_heteroskedasticity_test(
            input_residuals,
            use_f=False,
        )
        assert actual_statistic == expected_statistic
        assert actual_pvalue == expected_pvalue


class CheckCoint:
    """
    Test Cointegration Test Results for 2-variable system

    Test values taken from Stata
    """

    levels = ["1%", "5%", "10%"]
    data = macrodata.load_pandas()
    y1 = data.data["realcons"].values
    y2 = data.data["realgdp"].values

    def test_tstat(self):
        assert_almost_equal(self.coint_t, self.teststat, DECIMAL_4)


# this does not produce the old results anymore
class TestCoint_t(CheckCoint):
    """
    Get AR(1) parameter on residuals
    """

    @classmethod
    def setup_class(cls):
        # cls.coint_t = coint(cls.y1, cls.y2, trend="c")[0]
        cls.coint_t = coint(cls.y1, cls.y2, trend="c", maxlag=0, autolag=None)[
            0
        ]
        cls.teststat = -1.8208817
        cls.teststat = -1.830170986148


def test_coint():
    nobs = 200
    scale_e = 1
    const = [1, 0, 0.5, 0]
    np.random.seed(123)
    unit = np.random.randn(nobs).cumsum()
    y = scale_e * np.random.randn(nobs, 4)
    y[:, :2] += unit[:, None]
    y += const
    y = np.round(y, 4)

    # FIXME: enable/xfail/skip or delete
    for trend in []:  # ['c', 'ct', 'ctt', 'n']:
        print("\n", trend)
        print(coint(y[:, 0], y[:, 1], trend=trend, maxlag=4, autolag=None))
        print(coint(y[:, 0], y[:, 1:3], trend=trend, maxlag=4, autolag=None))
        print(coint(y[:, 0], y[:, 2:], trend=trend, maxlag=4, autolag=None))
        print(coint(y[:, 0], y[:, 1:], trend=trend, maxlag=4, autolag=None))

    # results from Stata egranger
    res_egranger = {}
    # trend = 'ct'
    res = res_egranger["ct"] = {}
    res[0] = [
        -5.615251442239,
        -4.406102369132,
        -3.82866685109,
        -3.532082997903,
    ]
    res[1] = [
        -5.63591313706,
        -4.758609717199,
        -4.179130554708,
        -3.880909696863,
    ]
    res[2] = [
        -2.892029275027,
        -4.758609717199,
        -4.179130554708,
        -3.880909696863,
    ]
    res[3] = [-5.626932544079, -5.08363327039, -4.502469783057, -4.2031051091]

    # trend = 'c'
    res = res_egranger["c"] = {}
    # first critical value res[0][1] has a discrepancy starting at 4th decimal
    res[0] = [
        -5.760696844656,
        -3.952043522638,
        -3.367006313729,
        -3.065831247948,
    ]
    # manually adjusted to have higher precision as in other cases
    res[0][1] = -3.952321293401682
    res[1] = [
        -5.781087068772,
        -4.367111915942,
        -3.783961136005,
        -3.483501524709,
    ]
    res[2] = [
        -2.477444137366,
        -4.367111915942,
        -3.783961136005,
        -3.483501524709,
    ]
    res[3] = [
        -5.778205811661,
        -4.735249216434,
        -4.152738973763,
        -3.852480848968,
    ]

    # trend = 'ctt'
    res = res_egranger["ctt"] = {}
    res[0] = [
        -5.644431269946,
        -4.796038299708,
        -4.221469431008,
        -3.926472577178,
    ]
    res[1] = [-5.665691609506, -5.111158174219, -4.53317278104, -4.23601008516]
    res[2] = [-3.161462374828, -5.111158174219, -4.53317278104, -4.23601008516]
    res[3] = [
        -5.657904558563,
        -5.406880189412,
        -4.826111619543,
        -4.527090164875,
    ]

    # The following for 'n' are only regression test numbers
    # trend = 'n' not allowed in egranger
    # trend = 'n'
    res = res_egranger["n"] = {}
    nan = np.nan  # shortcut for table
    res[0] = [-3.7146175989071137, nan, nan, nan]
    res[1] = [-3.8199323012888384, nan, nan, nan]
    res[2] = [-1.6865000791270679, nan, nan, nan]
    res[3] = [-3.7991270451873675, nan, nan, nan]

    for trend in ["c", "ct", "ctt", "n"]:
        res1 = {}
        res1[0] = coint(y[:, 0], y[:, 1], trend=trend, maxlag=4, autolag=None)
        res1[1] = coint(
            y[:, 0], y[:, 1:3], trend=trend, maxlag=4, autolag=None
        )
        res1[2] = coint(y[:, 0], y[:, 2:], trend=trend, maxlag=4, autolag=None)
        res1[3] = coint(y[:, 0], y[:, 1:], trend=trend, maxlag=4, autolag=None)

        for i in range(4):
            res = res_egranger[trend]

            assert_allclose(res1[i][0], res[i][0], rtol=1e-11)
            r2 = res[i][1:]
            r1 = res1[i][2]
            assert_allclose(r1, r2, rtol=0, atol=6e-7)

    # use default autolag #4490
    res1_0 = coint(y[:, 0], y[:, 1], trend="ct", maxlag=4)
    assert_allclose(res1_0[2], res_egranger["ct"][0][1:], rtol=0, atol=6e-7)
    # the following is just a regression test
    assert_allclose(
        res1_0[:2],
        [-13.992946638547112, 2.270898990540678e-27],
        rtol=1e-10,
        atol=1e-27,
    )


def test_coint_identical_series():
    nobs = 200
    scale_e = 1
    np.random.seed(123)
    y = scale_e * np.random.randn(nobs)
    warnings.simplefilter("always", CollinearityWarning)
    with pytest.warns(CollinearityWarning):
        c = coint(y, y, trend="c", maxlag=0, autolag=None)
    assert_equal(c[1], 0.0)
    assert_(np.isneginf(c[0]))


def test_coint_perfect_collinearity():
    # test uses nearly perfect collinearity
    nobs = 200
    scale_e = 1
    np.random.seed(123)
    x = scale_e * np.random.randn(nobs, 2)
    y = 1 + x.sum(axis=1) + 1e-7 * np.random.randn(nobs)
    warnings.simplefilter("always", CollinearityWarning)
    with warnings.catch_warnings(record=True) as w:
        c = coint(y, x, trend="c", maxlag=0, autolag=None)
    assert_equal(c[1], 0.0)
    assert_(np.isneginf(c[0]))


class TestGrangerCausality:
    def test_grangercausality(self):
        # some example data
        mdata = macrodata.load_pandas().data
        mdata = mdata[["realgdp", "realcons"]].values
        data = mdata.astype(float)
        data = np.diff(np.log(data), axis=0)

        # R: lmtest:grangertest
        r_result = [0.243097, 0.7844328, 195, 2]  # f_test
        with pytest.warns(FutureWarning, match="verbose is"):
            gr = grangercausalitytests(data[:, 1::-1], 2, verbose=False)
        assert_almost_equal(r_result, gr[2][0]["ssr_ftest"], decimal=7)
        assert_almost_equal(
            gr[2][0]["params_ftest"], gr[2][0]["ssr_ftest"], decimal=7
        )

    def test_grangercausality_single(self):
        mdata = macrodata.load_pandas().data
        mdata = mdata[["realgdp", "realcons"]].values
        data = mdata.astype(float)
        data = np.diff(np.log(data), axis=0)
        with pytest.warns(FutureWarning, match="verbose is"):
            gr = grangercausalitytests(data[:, 1::-1], 2, verbose=False)
        with pytest.warns(FutureWarning, match="verbose is"):
            gr2 = grangercausalitytests(data[:, 1::-1], [2], verbose=False)
        assert 1 in gr
        assert 1 not in gr2
        assert_almost_equal(
            gr[2][0]["ssr_ftest"], gr2[2][0]["ssr_ftest"], decimal=7
        )
        assert_almost_equal(
            gr[2][0]["params_ftest"], gr2[2][0]["ssr_ftest"], decimal=7
        )

    def test_granger_fails_on_nobs_check(self, reset_randomstate):
        # Test that if maxlag is too large, Granger Test raises a clear error.
        x = np.random.rand(10, 2)
        with pytest.warns(FutureWarning, match="verbose is"):
            grangercausalitytests(x, 2, verbose=False)  # This should pass.
        with pytest.raises(ValueError):
            with pytest.warns(FutureWarning, match="verbose is"):
                grangercausalitytests(x, 3, verbose=False)

    def test_granger_fails_on_finite_check(self, reset_randomstate):
        x = np.random.rand(1000, 2)
        x[500, 0] = np.nan
        x[750, 1] = np.inf
        with pytest.raises(ValueError, match="x contains NaN"):
            grangercausalitytests(x, 2)

    def test_granger_fails_on_zero_lag(self, reset_randomstate):
        x = np.random.rand(1000, 2)
        with pytest.raises(
            ValueError,
            match="maxlag must be a non-empty list containing only positive integers",
        ):
            grangercausalitytests(x, [0, 1, 2])


class TestKPSS:
    """
    R-code
    ------
    library(tseries)
    kpss.stat(x, "Level")
    kpss.stat(x, "Trend")

    In this context, x is the vector containing the
    macrodata['realgdp'] series.
    """

    @classmethod
    def setup(cls):
        cls.data = macrodata.load_pandas()
        cls.x = cls.data.data["realgdp"].values

    def test_fail_nonvector_input(self, reset_randomstate):
        # should be fine
        with pytest.warns(InterpolationWarning):
            kpss(self.x, nlags="legacy")

        x = np.random.rand(20, 2)
        assert_raises(ValueError, kpss, x)

    def test_fail_unclear_hypothesis(self):
        # these should be fine,
        with pytest.warns(InterpolationWarning):
            kpss(self.x, "c", nlags="legacy")
        with pytest.warns(InterpolationWarning):
            kpss(self.x, "C", nlags="legacy")
        with pytest.warns(InterpolationWarning):
            kpss(self.x, "ct", nlags="legacy")
        with pytest.warns(InterpolationWarning):
            kpss(self.x, "CT", nlags="legacy")

        assert_raises(
            ValueError, kpss, self.x, "unclear hypothesis", nlags="legacy"
        )

    def test_teststat(self):
        with pytest.warns(InterpolationWarning):
            kpss_stat, _, _, _ = kpss(self.x, "c", 3)
        assert_almost_equal(kpss_stat, 5.0169, DECIMAL_3)

        with pytest.warns(InterpolationWarning):
            kpss_stat, _, _, _ = kpss(self.x, "ct", 3)
        assert_almost_equal(kpss_stat, 1.1828, DECIMAL_3)

    def test_pval(self):
        with pytest.warns(InterpolationWarning):
            _, pval, _, _ = kpss(self.x, "c", 3)
        assert_equal(pval, 0.01)

        with pytest.warns(InterpolationWarning):
            _, pval, _, _ = kpss(self.x, "ct", 3)
        assert_equal(pval, 0.01)

    def test_store(self):
        with pytest.warns(InterpolationWarning):
            _, _, _, store = kpss(self.x, "c", 3, True)

        # assert attributes, and make sure they're correct
        assert_equal(store.nobs, len(self.x))
        assert_equal(store.lags, 3)

    # test autolag function _kpss_autolag against SAS 9.3
    def test_lags(self):
        # real GDP from macrodata data set
        with pytest.warns(InterpolationWarning):
            res = kpss(self.x, "c", nlags="auto")
        assert_equal(res[2], 9)
        # real interest rates from macrodata data set
        res = kpss(sunspots.load().data["SUNACTIVITY"], "c", nlags="auto")
        assert_equal(res[2], 7)
        # volumes from nile data set
        with pytest.warns(InterpolationWarning):
            res = kpss(nile.load().data["volume"], "c", nlags="auto")
        assert_equal(res[2], 5)
        # log-coinsurance from randhie data set
        with pytest.warns(InterpolationWarning):
            res = kpss(randhie.load().data["lncoins"], "ct", nlags="auto")
        assert_equal(res[2], 75)
        # in-vehicle time from modechoice data set
        with pytest.warns(InterpolationWarning):
            res = kpss(modechoice.load().data["invt"], "ct", nlags="auto")
        assert_equal(res[2], 18)

    def test_kpss_fails_on_nobs_check(self):
        # Test that if lags exceeds number of observations KPSS raises a
        # clear error
        # GH5925
        nobs = len(self.x)
        msg = r"lags \({}\) must be < number of observations \({}\)".format(
            nobs, nobs
        )
        with pytest.raises(ValueError, match=msg):
            kpss(self.x, "c", nlags=nobs)

    def test_kpss_autolags_does_not_assign_lags_equal_to_nobs(self):
        # Test that if *autolags* exceeds number of observations, we set
        # suitable lags
        # GH5925
        base = np.array([0, 0, 0, 0, 0, 1, 1.0])
        data_which_breaks_autolag = np.r_[np.tile(base, 297 // 7), [0, 0, 0]]
        kpss(data_which_breaks_autolag, nlags="auto")

    def test_legacy_lags(self):
        # Test legacy lags are the same
        with pytest.warns(InterpolationWarning):
            res = kpss(self.x, "c", nlags="legacy")
        assert_equal(res[2], 15)

    def test_unknown_lags(self):
        # Test legacy lags are the same
        with pytest.raises(ValueError):
            kpss(self.x, "c", nlags="unknown")

    def test_none(self):
        with pytest.warns(FutureWarning):
            kpss(self.x, nlags=None)


class TestRUR:
    """
    Simple implementation
    ------
    Since an R implementation of the test cannot be found, the method is tested against
    a simple implementation using a for loop.
    In this context, x is the vector containing the
    macrodata['realgdp'] series.
    """

    @classmethod
    def setup(cls):
        cls.data = macrodata.load_pandas()
        cls.x = cls.data.data["realgdp"].values

    # To be removed when range unit test gets an R implementation
    def simple_rur(self, x, store=False):
        x = array_like(x, "x")
        store = bool_like(store, "store")

        nobs = x.shape[0]

        # if m is not one, n != m * n
        if nobs != x.size:
            raise ValueError("x of shape {0} not understood".format(x.shape))

        # Table from [1] has been replicated using 200,000 samples
        # Critical values for new n_obs values have been identified
        pvals = [0.01, 0.025, 0.05, 0.10, 0.90, 0.95]
        n = np.array(
            [25, 50, 100, 150, 200, 250, 500, 1000, 2000, 3000, 4000, 5000]
        )
        crit = np.array(
            [
                [0.6626, 0.8126, 0.9192, 1.0712, 2.4863, 2.7312],
                [0.7977, 0.9274, 1.0478, 1.1964, 2.6821, 2.9613],
                [0.907, 1.0243, 1.1412, 1.2888, 2.8317, 3.1393],
                [0.9543, 1.0768, 1.1869, 1.3294, 2.8915, 3.2049],
                [0.9833, 1.0984, 1.2101, 1.3494, 2.9308, 3.2482],
                [0.9982, 1.1137, 1.2242, 1.3632, 2.9571, 3.2482],
                [1.0494, 1.1643, 1.2712, 1.4076, 3.0207, 3.3584],
                [1.0846, 1.1959, 1.2988, 1.4344, 3.0653, 3.4073],
                [1.1121, 1.2200, 1.3230, 1.4556, 3.0948, 3.4439],
                [1.1204, 1.2295, 1.3318, 1.4656, 3.1054, 3.4632],
                [1.1309, 1.2347, 1.3318, 1.4693, 3.1165, 3.4717],
                [1.1377, 1.2402, 1.3408, 1.4729, 3.1252, 3.4807],
            ]
        )

        # Interpolation for nobs
        inter_crit = np.zeros((1, crit.shape[1]))
        for i in range(crit.shape[1]):
            f = interp1d(n, crit[:, i])
            inter_crit[0, i] = f(nobs)

        # Calculate RUR stat
        count = 0

        max_p = x[0]
        min_p = x[0]

        for v in x[1:]:
            if v > max_p:
                max_p = v
                count = count + 1
            if v < min_p:
                min_p = v
                count = count + 1

        rur_stat = count / np.sqrt(len(x))

        k = len(pvals) - 1
        for i in range(len(pvals) - 1, -1, -1):
            if rur_stat < inter_crit[0, i]:
                k = i
            else:
                break

        p_value = pvals[k]

        warn_msg = """\
        The test statistic is outside of the range of p-values available in the
        look-up table. The actual p-value is {direction} than the p-value returned.
        """
        direction = ""
        if p_value == pvals[-1]:
            direction = "smaller"
        elif p_value == pvals[0]:
            direction = "larger"

        if direction:
            warnings.warn(
                warn_msg.format(direction=direction), InterpolationWarning
            )

        crit_dict = {
            "10%": inter_crit[0, 3],
            "5%": inter_crit[0, 2],
            "2.5%": inter_crit[0, 1],
            "1%": inter_crit[0, 0],
        }

        if store:
            from statsmodels.stats.diagnostic import ResultsStore

            rstore = ResultsStore()
            rstore.nobs = nobs

            rstore.H0 = "The series is not stationary"
            rstore.HA = "The series is stationary"

            return rur_stat, p_value, crit_dict, rstore
        else:
            return rur_stat, p_value, crit_dict

    def test_fail_nonvector_input(self, reset_randomstate):
        with pytest.warns(InterpolationWarning):
            range_unit_root_test(self.x)

        x = np.random.rand(20, 2)
        assert_raises(ValueError, range_unit_root_test, x)

    def test_teststat(self):
        with pytest.warns(InterpolationWarning):
            rur_stat, _, _ = range_unit_root_test(self.x)
            simple_rur_stat, _, _ = self.simple_rur(self.x)
        assert_almost_equal(rur_stat, simple_rur_stat, DECIMAL_3)

    def test_pval(self):
        with pytest.warns(InterpolationWarning):
            _, pval, _ = range_unit_root_test(self.x)
            _, simple_pval, _ = self.simple_rur(self.x)
        assert_equal(pval, simple_pval)

    def test_store(self):
        with pytest.warns(InterpolationWarning):
            _, _, _, store = range_unit_root_test(self.x, True)

        # assert attributes, and make sure they're correct
        assert_equal(store.nobs, len(self.x))


def test_pandasacovf():
    s = Series(lrange(1, 11))
    assert_almost_equal(acovf(s, fft=False), acovf(s.values, fft=False))


def test_acovf2d(reset_randomstate):
    dta = sunspots.load_pandas().data
    dta.index = date_range(start="1700", end="2009", freq="A")[:309]
    del dta["YEAR"]
    res = acovf(dta, fft=False)
    assert_equal(res, acovf(dta.values, fft=False))
    x = np.random.random((10, 2))
    with pytest.raises(ValueError):
        acovf(x, fft=False)


@pytest.mark.parametrize("demean", [True, False])
@pytest.mark.parametrize("adjusted", [True, False])
def test_acovf_fft_vs_convolution(demean, adjusted, reset_randomstate):
    q = np.random.normal(size=100)

    F1 = acovf(q, demean=demean, adjusted=adjusted, fft=True)
    F2 = acovf(q, demean=demean, adjusted=adjusted, fft=False)
    assert_almost_equal(F1, F2, decimal=7)


@pytest.mark.parametrize("demean", [True, False])
@pytest.mark.parametrize("adjusted", [True, False])
def test_ccovf_fft_vs_convolution(demean, adjusted, reset_randomstate):
    x = np.random.normal(size=128)
    y = np.random.normal(size=128)

    F1 = ccovf(x, y, demean=demean, adjusted=adjusted, fft=False)
    F2 = ccovf(x, y, demean=demean, adjusted=adjusted, fft=True)
    assert_almost_equal(F1, F2, decimal=7)


@pytest.mark.parametrize("demean", [True, False])
@pytest.mark.parametrize("adjusted", [True, False])
@pytest.mark.parametrize("fft", [True, False])
def test_compare_acovf_vs_ccovf(demean, adjusted, fft, reset_randomstate):
    x = np.random.normal(size=128)

    F1 = acovf(x, demean=demean, adjusted=adjusted, fft=fft)
    F2 = ccovf(x, x, demean=demean, adjusted=adjusted, fft=fft)
    assert_almost_equal(F1, F2, decimal=7)


@pytest.mark.smoke
@pytest.mark.slow
def test_arma_order_select_ic():
    # smoke test, assumes info-criteria are right
    from statsmodels.tsa.arima_process import arma_generate_sample

    arparams = np.array([0.75, -0.25])
    maparams = np.array([0.65, 0.35])
    arparams = np.r_[1, -arparams]
    maparam = np.r_[1, maparams]  # FIXME: Never used
    nobs = 250
    np.random.seed(2014)
    y = arma_generate_sample(arparams, maparams, nobs)
    res = arma_order_select_ic(y, ic=["aic", "bic"], trend="n")
    # regression tests in case we change algorithm to minic in sas
    aic_x = np.array(
        [
            [764.36517643, 552.7342255, 484.29687843],
            [562.10924262, 485.5197969, 480.32858497],
            [507.04581344, 482.91065829, 481.91926034],
            [484.03995962, 482.14868032, 483.86378955],
            [481.8849479, 483.8377379, 485.83756612],
        ]
    )
    bic_x = np.array(
        [
            [767.88663735, 559.77714733, 494.86126118],
            [569.15216446, 496.08417966, 494.41442864],
            [517.61019619, 496.99650196, 499.52656493],
            [498.12580329, 499.75598491, 504.99255506],
            [499.49225249, 504.96650341, 510.48779255],
        ]
    )
    aic = DataFrame(aic_x, index=lrange(5), columns=lrange(3))
    bic = DataFrame(bic_x, index=lrange(5), columns=lrange(3))
    assert_almost_equal(res.aic.values, aic.values, 5)
    assert_almost_equal(res.bic.values, bic.values, 5)
    assert_equal(res.aic_min_order, (1, 2))
    assert_equal(res.bic_min_order, (1, 2))
    assert_(res.aic.index.equals(aic.index))
    assert_(res.aic.columns.equals(aic.columns))
    assert_(res.bic.index.equals(bic.index))
    assert_(res.bic.columns.equals(bic.columns))

    index = pd.date_range("2000-1-1", freq="M", periods=len(y))
    y_series = pd.Series(y, index=index)
    res_pd = arma_order_select_ic(
        y_series, max_ar=2, max_ma=1, ic=["aic", "bic"], trend="n"
    )
    assert_almost_equal(res_pd.aic.values, aic.values[:3, :2], 5)
    assert_almost_equal(res_pd.bic.values, bic.values[:3, :2], 5)
    assert_equal(res_pd.aic_min_order, (2, 1))
    assert_equal(res_pd.bic_min_order, (1, 1))

    res = arma_order_select_ic(y, ic="aic", trend="n")
    assert_almost_equal(res.aic.values, aic.values, 5)
    assert_(res.aic.index.equals(aic.index))
    assert_(res.aic.columns.equals(aic.columns))
    assert_equal(res.aic_min_order, (1, 2))


def test_arma_order_select_ic_failure():
    # this should trigger an SVD convergence failure, smoke test that it
    # returns, likely platform dependent failure...
    # looks like AR roots may be cancelling out for 4, 1?
    y = np.array(
        [
            0.86074377817203640006,
            0.85316549067906921611,
            0.87104653774363305363,
            0.60692382068987393851,
            0.69225941967301307667,
            0.73336177248909339976,
            0.03661329261479619179,
            0.15693067239962379955,
            0.12777403512447857437,
            -0.27531446294481976,
            -0.24198139631653581283,
            -0.23903317951236391359,
            -0.26000241325906497947,
            -0.21282920015519238288,
            -0.15943768324388354896,
            0.25169301564268781179,
            0.1762305709151877342,
            0.12678133368791388857,
            0.89755829086753169399,
            0.82667068795350151511,
        ]
    )
    import warnings

    with warnings.catch_warnings():
        # catch a hessian inversion and convergence failure warning
        warnings.simplefilter("ignore")
        res = arma_order_select_ic(y)


def test_acf_fft_dataframe():
    # regression test #322

    result = acf(
        sunspots.load_pandas().data[["SUNACTIVITY"]], fft=True, nlags=20
    )
    assert_equal(result.ndim, 1)


def test_levinson_durbin_acov():
    rho = 0.9
    m = 20
    acov = rho ** np.arange(200)
    sigma2_eps, ar, pacf, _, _ = levinson_durbin(acov, m, isacov=True)
    assert_allclose(sigma2_eps, 1 - rho ** 2)
    assert_allclose(ar, np.array([rho] + [0] * (m - 1)), atol=1e-8)
    assert_allclose(pacf, np.array([1, rho] + [0] * (m - 1)), atol=1e-8)


@pytest.mark.parametrize("missing", ["conservative", "drop", "raise", "none"])
@pytest.mark.parametrize("fft", [False, True])
@pytest.mark.parametrize("demean", [True, False])
@pytest.mark.parametrize("adjusted", [True, False])
def test_acovf_nlags(acovf_data, adjusted, demean, fft, missing):
    full = acovf(
        acovf_data, adjusted=adjusted, demean=demean, fft=fft, missing=missing
    )
    limited = acovf(
        acovf_data,
        adjusted=adjusted,
        demean=demean,
        fft=fft,
        missing=missing,
        nlag=10,
    )
    assert_allclose(full[:11], limited)


@pytest.mark.parametrize("missing", ["conservative", "drop"])
@pytest.mark.parametrize("fft", [False, True])
@pytest.mark.parametrize("demean", [True, False])
@pytest.mark.parametrize("adjusted", [True, False])
def test_acovf_nlags_missing(acovf_data, adjusted, demean, fft, missing):
    acovf_data = acovf_data.copy()
    acovf_data[1:3] = np.nan
    full = acovf(
        acovf_data, adjusted=adjusted, demean=demean, fft=fft, missing=missing
    )
    limited = acovf(
        acovf_data,
        adjusted=adjusted,
        demean=demean,
        fft=fft,
        missing=missing,
        nlag=10,
    )
    assert_allclose(full[:11], limited)


def test_acovf_error(acovf_data):
    with pytest.raises(ValueError):
        acovf(acovf_data, nlag=250, fft=False)


def test_pacf2acf_ar():
    pacf = np.zeros(10)
    pacf[0] = 1
    pacf[1] = 0.9
    ar, acf = levinson_durbin_pacf(pacf)
    assert_allclose(acf, 0.9 ** np.arange(10.0))
    assert_allclose(ar, pacf[1:], atol=1e-8)

    ar, acf = levinson_durbin_pacf(pacf, nlags=5)
    assert_allclose(acf, 0.9 ** np.arange(6.0))
    assert_allclose(ar, pacf[1:6], atol=1e-8)


def test_pacf2acf_levinson_durbin():
    pacf = -(0.9 ** np.arange(11.0))
    pacf[0] = 1
    ar, acf = levinson_durbin_pacf(pacf)
    _, ar_ld, pacf_ld, _, _ = levinson_durbin(acf, 10, isacov=True)
    assert_allclose(ar, ar_ld, atol=1e-8)
    assert_allclose(pacf, pacf_ld, atol=1e-8)

    # From R, FitAR, PacfToAR
    ar_from_r = [
        -4.1609,
        -9.2549,
        -14.4826,
        -17.6505,
        -17.5012,
        -14.2969,
        -9.5020,
        -4.9184,
        -1.7911,
        -0.3486,
    ]
    assert_allclose(ar, ar_from_r, atol=1e-4)


def test_pacf2acf_errors():
    pacf = -(0.9 ** np.arange(11.0))
    pacf[0] = 1
    with pytest.raises(ValueError):
        levinson_durbin_pacf(pacf, nlags=20)
    with pytest.raises(ValueError):
        levinson_durbin_pacf(pacf[1:])
    with pytest.raises(ValueError):
        levinson_durbin_pacf(np.zeros(10))
    with pytest.raises(ValueError):
        levinson_durbin_pacf(np.zeros((10, 2)))


def test_pacf_burg():
    rnd = np.random.RandomState(12345)
    e = rnd.randn(10001)
    y = e[1:] + 0.5 * e[:-1]
    pacf, sigma2 = pacf_burg(y, 10)
    yw_pacf = pacf_yw(y, 10)
    assert_allclose(pacf, yw_pacf, atol=5e-4)
    # Internal consistency check between pacf and sigma2
    ye = y - y.mean()
    s2y = ye.dot(ye) / 10000
    pacf[0] = 0
    sigma2_direct = s2y * np.cumprod(1 - pacf ** 2)
    assert_allclose(sigma2, sigma2_direct, atol=1e-3)


def test_pacf_burg_error():
    with pytest.raises(ValueError):
        pacf_burg(np.empty((20, 2)), 10)
    with pytest.raises(ValueError):
        pacf_burg(np.empty(100), 101)


def test_innovations_algo_brockwell_davis():
    ma = -0.9
    acovf = np.array([1 + ma ** 2, ma])
    theta, sigma2 = innovations_algo(acovf, nobs=4)
    exp_theta = np.array([[0], [-0.4972], [-0.6606], [-0.7404]])
    assert_allclose(theta, exp_theta, rtol=1e-4)
    assert_allclose(sigma2, [1.81, 1.3625, 1.2155, 1.1436], rtol=1e-4)

    theta, sigma2 = innovations_algo(acovf, nobs=500)
    assert_allclose(theta[-1, 0], ma)
    assert_allclose(sigma2[-1], 1.0)


def test_innovations_algo_rtol():
    ma = np.array([-0.9, 0.5])
    acovf = np.array([1 + (ma ** 2).sum(), ma[0] + ma[1] * ma[0], ma[1]])
    theta, sigma2 = innovations_algo(acovf, nobs=500)
    theta_2, sigma2_2 = innovations_algo(acovf, nobs=500, rtol=1e-8)
    assert_allclose(theta, theta_2)
    assert_allclose(sigma2, sigma2_2)


def test_innovations_errors():
    ma = -0.9
    acovf = np.array([1 + ma ** 2, ma])
    with pytest.raises(TypeError):
        innovations_algo(acovf, nobs=2.2)
    with pytest.raises(ValueError):
        innovations_algo(acovf, nobs=-1)
    with pytest.raises(ValueError):
        innovations_algo(np.empty((2, 2)))
    with pytest.raises(TypeError):
        innovations_algo(acovf, rtol="none")


def test_innovations_filter_brockwell_davis(reset_randomstate):
    ma = -0.9
    acovf = np.array([1 + ma ** 2, ma])
    theta, _ = innovations_algo(acovf, nobs=4)
    e = np.random.randn(5)
    endog = e[1:] + ma * e[:-1]
    resid = innovations_filter(endog, theta)
    expected = [endog[0]]
    for i in range(1, 4):
        expected.append(endog[i] - theta[i, 0] * expected[-1])
    expected = np.array(expected)
    assert_allclose(resid, expected)


def test_innovations_filter_pandas(reset_randomstate):
    ma = np.array([-0.9, 0.5])
    acovf = np.array([1 + (ma ** 2).sum(), ma[0] + ma[1] * ma[0], ma[1]])
    theta, _ = innovations_algo(acovf, nobs=10)
    endog = np.random.randn(10)
    endog_pd = pd.Series(endog, index=pd.date_range("2000-01-01", periods=10))
    resid = innovations_filter(endog, theta)
    resid_pd = innovations_filter(endog_pd, theta)
    assert_allclose(resid, resid_pd.values)
    assert_index_equal(endog_pd.index, resid_pd.index)


def test_innovations_filter_errors():
    ma = -0.9
    acovf = np.array([1 + ma ** 2, ma])
    theta, _ = innovations_algo(acovf, nobs=4)
    with pytest.raises(ValueError):
        innovations_filter(np.empty((2, 2)), theta)
    with pytest.raises(ValueError):
        innovations_filter(np.empty(4), theta[:-1])
    with pytest.raises(ValueError):
        innovations_filter(pd.DataFrame(np.empty((1, 4))), theta)


def test_innovations_algo_filter_kalman_filter(reset_randomstate):
    # Test the innovations algorithm and filter against the Kalman filter
    # for exact likelihood evaluation of an ARMA process
    ar_params = np.array([0.5])
    ma_params = np.array([0.2])
    # TODO could generalize to sigma2 != 1, if desired, after #5324 is merged
    # and there is a sigma2 argument to arma_acovf
    # (but maybe this is not really necessary for the point of this test)
    sigma2 = 1

    endog = np.random.normal(size=10)

    # Innovations algorithm approach
    acovf = arma_acovf(
        np.r_[1, -ar_params], np.r_[1, ma_params], nobs=len(endog)
    )

    theta, v = innovations_algo(acovf)
    u = innovations_filter(endog, theta)
    llf_obs = -0.5 * u ** 2 / (sigma2 * v) - 0.5 * np.log(2 * np.pi * v)

    # Kalman filter apparoach
    mod = SARIMAX(endog, order=(len(ar_params), 0, len(ma_params)))
    res = mod.filter(np.r_[ar_params, ma_params, sigma2])

    # Test that the two approaches are identical
    atol = 1e-6 if PLATFORM_WIN else 0.0
    assert_allclose(u, res.forecasts_error[0], rtol=1e-6, atol=atol)
    assert_allclose(
        theta[1:, 0], res.filter_results.kalman_gain[0, 0, :-1], atol=atol
    )
    assert_allclose(llf_obs, res.llf_obs, atol=atol)


def test_adfuller_short_series(reset_randomstate):
    y = np.random.standard_normal(7)
    res = adfuller(y, store=True)
    assert res[-1].maxlag == 1
    y = np.random.standard_normal(2)
    with pytest.raises(ValueError, match="sample size is too short"):
        adfuller(y)
    y = np.random.standard_normal(3)
    with pytest.raises(ValueError, match="sample size is too short"):
        adfuller(y, regression="ct")


def test_adfuller_maxlag_too_large(reset_randomstate):
    y = np.random.standard_normal(100)
    with pytest.raises(ValueError, match="maxlag must be less than"):
        adfuller(y, maxlag=51)


class SetupZivotAndrews:
    # test directory
    cur_dir = CURR_DIR
    run_dir = os.path.join(cur_dir, "results")
    # use same file for testing failure modes
    fail_file = os.path.join(run_dir, "rgnp.csv")
    fail_mdl = np.asarray(pd.read_csv(fail_file))


class TestZivotAndrews(SetupZivotAndrews):

    # failure mode tests
    def test_fail_regression_type(self):
        with pytest.raises(ValueError):
            zivot_andrews(self.fail_mdl, regression="x")

    def test_fail_trim_value(self):
        with pytest.raises(ValueError):
            zivot_andrews(self.fail_mdl, trim=0.5)

    def test_fail_array_shape(self):
        with pytest.raises(ValueError):
            zivot_andrews(np.random.rand(50, 2))

    def test_fail_autolag_type(self):
        with pytest.raises(ValueError):
            zivot_andrews(self.fail_mdl, autolag="None")

    @pytest.mark.parametrize("autolag", ["AIC", "aic", "Aic"])
    def test_autolag_case_sensitivity(self, autolag):
        res = zivot_andrews(self.fail_mdl, autolag=autolag)
        assert res[3] == 1

    # following tests compare results to R package urca.ur.za (1.13-0)
    def test_rgnp_case(self):
        res = zivot_andrews(
            self.fail_mdl, maxlag=8, regression="c", autolag=None
        )
        assert_allclose(
            [res[0], res[1], res[4]], [-5.57615, 0.00312, 20], rtol=1e-3
        )

    def test_gnpdef_case(self):
        mdlfile = os.path.join(self.run_dir, "gnpdef.csv")
        mdl = np.asarray(pd.read_csv(mdlfile))
        res = zivot_andrews(mdl, maxlag=8, regression="c", autolag="t-stat")
        assert_allclose(
            [res[0], res[1], res[3], res[4]],
            [-4.12155, 0.28024, 5, 40],
            rtol=1e-3,
        )

    def test_stkprc_case(self):
        mdlfile = os.path.join(self.run_dir, "stkprc.csv")
        mdl = np.asarray(pd.read_csv(mdlfile))
        res = zivot_andrews(mdl, maxlag=8, regression="ct", autolag="t-stat")
        assert_allclose(
            [res[0], res[1], res[3], res[4]],
            [-5.60689, 0.00894, 1, 65],
            rtol=1e-3,
        )

    def test_rgnpq_case(self):
        mdlfile = os.path.join(self.run_dir, "rgnpq.csv")
        mdl = np.asarray(pd.read_csv(mdlfile))
        res = zivot_andrews(mdl, maxlag=12, regression="t", autolag="t-stat")
        assert_allclose(
            [res[0], res[1], res[3], res[4]],
            [-3.02761, 0.63993, 12, 102],
            rtol=1e-3,
        )

    def test_rand10000_case(self):
        mdlfile = os.path.join(self.run_dir, "rand10000.csv")
        mdl = np.asarray(pd.read_csv(mdlfile))
        res = zivot_andrews(mdl, regression="c", autolag="t-stat")
        assert_allclose(
            [res[0], res[1], res[3], res[4]],
            [-3.48223, 0.69111, 25, 7071],
            rtol=1e-3,
        )


def test_acf_conservate_nanops(reset_randomstate):
    # GH 6729
    e = np.random.standard_normal(100)
    for i in range(1, e.shape[0]):
        e[i] += 0.9 * e[i - 1]
    e[::7] = np.nan
    result = acf(e, missing="conservative", nlags=10, fft=False)
    resid = e - np.nanmean(e)
    expected = np.ones(11)
    nobs = e.shape[0]
    gamma0 = np.nansum(resid * resid)
    for i in range(1, 10 + 1):
        expected[i] = np.nansum(resid[i:] * resid[: nobs - i]) / gamma0
    assert_allclose(result, expected, rtol=1e-4, atol=1e-4)


def test_pacf_nlags_error(reset_randomstate):
    e = np.random.standard_normal(100)
    with pytest.raises(ValueError, match="Can only compute partial"):
        pacf(e, 50)


def test_coint_auto_tstat():
    rs = np.random.RandomState(3733696641)
    x = np.cumsum(rs.standard_normal(100))
    y = np.cumsum(rs.standard_normal(100))
    res = coint(
        x,
        y,
        trend="c",
        method="aeg",
        maxlag=0,
        autolag="t-stat",
        return_results=False,
    )
    assert np.abs(res[0]) < 1.65


rs = np.random.RandomState(1)
a = rs.random_sample(120)
b = np.zeros_like(a)
df1 = pd.DataFrame({"b": b, "a": a})
df2 = pd.DataFrame({"a": a, "b": b})

b = np.ones_like(a)
df3 = pd.DataFrame({"b": b, "a": a})
df4 = pd.DataFrame({"a": a, "b": b})

gc_data_sets = [df1, df2, df3, df4]


@pytest.mark.parametrize("dataset", gc_data_sets)
def test_granger_causality_exceptions(dataset):
    with pytest.raises(InfeasibleTestError):
        with pytest.warns(FutureWarning, match="verbose"):
            grangercausalitytests(dataset, 4, verbose=False)


def test_granger_causality_exception_maxlag(gc_data):
    with pytest.raises(ValueError, match="maxlag must be"):
        grangercausalitytests(gc_data, maxlag=-1)
    with pytest.raises(NotImplementedError):
        grangercausalitytests(gc_data, 3, addconst=False)


def test_granger_causality_verbose(gc_data):
    with pytest.warns(FutureWarning, match="verbose"):
        grangercausalitytests(gc_data, 3, verbose=True)
