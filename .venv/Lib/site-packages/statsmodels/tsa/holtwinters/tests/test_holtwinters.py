"""
Author: Terence L van Zyl
Modified: Kevin Sheppard
"""
from statsmodels.compat.pandas import MONTH_END
from statsmodels.compat.pytest import pytest_warns

import os
import re
import warnings

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pandas as pd
import pytest
import scipy.stats

from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
from statsmodels.tsa.holtwinters import (
    PY_SMOOTHERS,
    SMOOTHERS,
    ExponentialSmoothing,
    Holt,
    SimpleExpSmoothing,
)
from statsmodels.tsa.holtwinters._exponential_smoothers import (
    HoltWintersArgs,
    _test_to_restricted,
)
from statsmodels.tsa.holtwinters._smoothers import (
    HoltWintersArgs as PyHoltWintersArgs,
    to_restricted,
    to_unrestricted,
)

base, _ = os.path.split(os.path.abspath(__file__))
housing_data = pd.read_csv(
    os.path.join(base, "results", "housing-data.csv"),
    index_col="DATE",
    parse_dates=True,
)
housing_data = housing_data.asfreq("MS")

SEASONALS = ("add", "mul", None)
TRENDS = ("add", "mul", None)

# aust = pd.read_json(aust_json, typ='Series').sort_index()
data = [
    41.727457999999999,
    24.04185,
    32.328102999999999,
    37.328707999999999,
    46.213152999999998,
    29.346326000000001,
    36.482909999999997,
    42.977719,
    48.901524999999999,
    31.180221,
    37.717880999999998,
    40.420211000000002,
    51.206862999999998,
    31.887228,
    40.978262999999998,
    43.772491000000002,
    55.558566999999996,
    33.850915000000001,
    42.076383,
    45.642291999999998,
    59.766779999999997,
    35.191876999999998,
    44.319737000000003,
    47.913736,
]
index = [
    "2005-03-01 00:00:00",
    "2005-06-01 00:00:00",
    "2005-09-01 00:00:00",
    "2005-12-01 00:00:00",
    "2006-03-01 00:00:00",
    "2006-06-01 00:00:00",
    "2006-09-01 00:00:00",
    "2006-12-01 00:00:00",
    "2007-03-01 00:00:00",
    "2007-06-01 00:00:00",
    "2007-09-01 00:00:00",
    "2007-12-01 00:00:00",
    "2008-03-01 00:00:00",
    "2008-06-01 00:00:00",
    "2008-09-01 00:00:00",
    "2008-12-01 00:00:00",
    "2009-03-01 00:00:00",
    "2009-06-01 00:00:00",
    "2009-09-01 00:00:00",
    "2009-12-01 00:00:00",
    "2010-03-01 00:00:00",
    "2010-06-01 00:00:00",
    "2010-09-01 00:00:00",
    "2010-12-01 00:00:00",
]
idx = pd.to_datetime(index)
aust = pd.Series(data, index=pd.DatetimeIndex(idx, freq=pd.infer_freq(idx)))


@pytest.fixture(scope="module")
def ses():
    rs = np.random.RandomState(0)
    e = rs.standard_normal(1200)
    y = e.copy()
    for i in range(1, 1200):
        y[i] = y[i - 1] + e[i] - 0.2 * e[i - 1]
    y = y[200:]
    index = pd.date_range("2000-1-1", periods=y.shape[0], freq=MONTH_END)
    return pd.Series(y, index=index, name="y")


def _simple_dbl_exp_smoother(x, alpha, beta, l0, b0, nforecast=0):
    """
    Simple, slow, direct implementation of double exp smoothing for testing
    """
    n = x.shape[0]
    lvals = np.zeros(n)
    b = np.zeros(n)
    xhat = np.zeros(n)
    f = np.zeros(nforecast)
    lvals[0] = l0
    b[0] = b0
    # Special case the 0 observations since index -1 is not available
    xhat[0] = l0 + b0
    lvals[0] = alpha * x[0] + (1 - alpha) * (l0 + b0)
    b[0] = beta * (lvals[0] - l0) + (1 - beta) * b0
    for t in range(1, n):
        # Obs in index t is the time t forecast for t + 1
        lvals[t] = alpha * x[t] + (1 - alpha) * (lvals[t - 1] + b[t - 1])
        b[t] = beta * (lvals[t] - lvals[t - 1]) + (1 - beta) * b[t - 1]

    xhat[1:] = lvals[0:-1] + b[0:-1]
    f[:] = lvals[-1] + np.arange(1, nforecast + 1) * b[-1]
    err = x - xhat
    return lvals, b, f, err, xhat


class TestHoltWinters:
    @classmethod
    def setup_class(cls):
        # Changed for backwards compatibility with pandas

        # oildata_oil = pd.read_json(oildata_oil_json,
        #                            typ='Series').sort_index()
        data = [
            446.65652290000003,
            454.47330649999998,
            455.66297400000002,
            423.63223879999998,
            456.27132790000002,
            440.58805009999998,
            425.33252010000001,
            485.14944789999998,
            506.04816210000001,
            526.79198329999997,
            514.26888899999994,
            494.21101929999998,
        ]
        index = [
            "1996-12-31 00:00:00",
            "1997-12-31 00:00:00",
            "1998-12-31 00:00:00",
            "1999-12-31 00:00:00",
            "2000-12-31 00:00:00",
            "2001-12-31 00:00:00",
            "2002-12-31 00:00:00",
            "2003-12-31 00:00:00",
            "2004-12-31 00:00:00",
            "2005-12-31 00:00:00",
            "2006-12-31 00:00:00",
            "2007-12-31 00:00:00",
        ]
        oildata_oil = pd.Series(data, index)
        oildata_oil.index = pd.DatetimeIndex(
            oildata_oil.index, freq=pd.infer_freq(oildata_oil.index)
        )
        cls.oildata_oil = oildata_oil

        # air_ausair = pd.read_json(air_ausair_json,
        #                           typ='Series').sort_index()
        data = [
            17.5534,
            21.860099999999999,
            23.886600000000001,
            26.929300000000001,
            26.888500000000001,
            28.831399999999999,
            30.075099999999999,
            30.953499999999998,
            30.185700000000001,
            31.579699999999999,
            32.577568999999997,
            33.477398000000001,
            39.021580999999998,
            41.386431999999999,
            41.596552000000003,
        ]
        index = [
            "1990-12-31 00:00:00",
            "1991-12-31 00:00:00",
            "1992-12-31 00:00:00",
            "1993-12-31 00:00:00",
            "1994-12-31 00:00:00",
            "1995-12-31 00:00:00",
            "1996-12-31 00:00:00",
            "1997-12-31 00:00:00",
            "1998-12-31 00:00:00",
            "1999-12-31 00:00:00",
            "2000-12-31 00:00:00",
            "2001-12-31 00:00:00",
            "2002-12-31 00:00:00",
            "2003-12-31 00:00:00",
            "2004-12-31 00:00:00",
        ]
        air_ausair = pd.Series(data, index)
        air_ausair.index = pd.DatetimeIndex(
            air_ausair.index, freq=pd.infer_freq(air_ausair.index)
        )
        cls.air_ausair = air_ausair

        # livestock2_livestock = pd.read_json(livestock2_livestock_json,
        #                                     typ='Series').sort_index()
        data = [
            263.91774700000002,
            268.30722200000002,
            260.662556,
            266.63941899999998,
            277.51577800000001,
            283.834045,
            290.30902800000001,
            292.474198,
            300.83069399999999,
            309.28665699999999,
            318.33108099999998,
            329.37239,
            338.88399800000002,
            339.24412599999999,
            328.60063200000002,
            314.25538499999999,
            314.45969500000001,
            321.41377899999998,
            329.78929199999999,
            346.38516499999997,
            352.29788200000002,
            348.37051500000001,
            417.56292200000001,
            417.12356999999997,
            417.749459,
            412.233904,
            411.94681700000001,
            394.69707499999998,
            401.49927000000002,
            408.27046799999999,
            414.24279999999999,
        ]
        index = [
            "1970-12-31 00:00:00",
            "1971-12-31 00:00:00",
            "1972-12-31 00:00:00",
            "1973-12-31 00:00:00",
            "1974-12-31 00:00:00",
            "1975-12-31 00:00:00",
            "1976-12-31 00:00:00",
            "1977-12-31 00:00:00",
            "1978-12-31 00:00:00",
            "1979-12-31 00:00:00",
            "1980-12-31 00:00:00",
            "1981-12-31 00:00:00",
            "1982-12-31 00:00:00",
            "1983-12-31 00:00:00",
            "1984-12-31 00:00:00",
            "1985-12-31 00:00:00",
            "1986-12-31 00:00:00",
            "1987-12-31 00:00:00",
            "1988-12-31 00:00:00",
            "1989-12-31 00:00:00",
            "1990-12-31 00:00:00",
            "1991-12-31 00:00:00",
            "1992-12-31 00:00:00",
            "1993-12-31 00:00:00",
            "1994-12-31 00:00:00",
            "1995-12-31 00:00:00",
            "1996-12-31 00:00:00",
            "1997-12-31 00:00:00",
            "1998-12-31 00:00:00",
            "1999-12-31 00:00:00",
            "2000-12-31 00:00:00",
        ]
        livestock2_livestock = pd.Series(data, index)
        livestock2_livestock.index = pd.DatetimeIndex(
            livestock2_livestock.index,
            freq=pd.infer_freq(livestock2_livestock.index),
        )
        cls.livestock2_livestock = livestock2_livestock

        cls.aust = aust
        cls.start_params = [
            1.5520372162082909e-09,
            2.066338221674873e-18,
            1.727109018250519e-09,
            50.568333479425036,
            0.9129273810171223,
            0.83535867,
            0.50297119,
            0.62439273,
            0.67723128,
        ]

    def test_predict(self):
        fit1 = ExponentialSmoothing(
            self.aust,
            seasonal_periods=4,
            trend="add",
            seasonal="mul",
            initialization_method="estimated",
        ).fit(start_params=self.start_params)
        fit2 = ExponentialSmoothing(
            self.aust,
            seasonal_periods=4,
            trend="add",
            seasonal="mul",
            initialization_method="estimated",
        ).fit(start_params=self.start_params)
        # fit3 = ExponentialSmoothing(self.aust, seasonal_periods=4,
        # trend='add', seasonal='mul').fit(remove_bias=True,
        # use_basinhopping=True)
        assert_almost_equal(
            fit1.predict("2011-03-01 00:00:00", "2011-12-01 00:00:00"),
            [61.3083, 37.3730, 46.9652, 51.5578],
            3,
        )
        assert_almost_equal(
            fit2.predict(end="2011-12-01 00:00:00"),
            [61.3083, 37.3730, 46.9652, 51.5578],
            3,
        )

    # assert_almost_equal(fit3.predict('2010-10-01 00:00:00',
    #                                  '2010-10-01 00:00:00'),
    #                                  [49.087], 3)

    def test_ndarray(self):
        fit1 = ExponentialSmoothing(
            self.aust.values,
            seasonal_periods=4,
            trend="add",
            seasonal="mul",
            initialization_method="estimated",
        ).fit(start_params=self.start_params)
        assert_almost_equal(
            fit1.forecast(4), [61.3083, 37.3730, 46.9652, 51.5578], 3
        )

    # FIXME: this is passing 2019-05-22 on some platforms; what has changed?
    @pytest.mark.xfail(reason="Optimizer does not converge", strict=False)
    def test_forecast(self):
        fit1 = ExponentialSmoothing(
            self.aust,
            seasonal_periods=4,
            trend="add",
            seasonal="add",
        ).fit(method="bh", use_brute=True)
        assert_almost_equal(
            fit1.forecast(steps=4), [60.9542, 36.8505, 46.1628, 50.1272], 3
        )

    def test_simple_exp_smoothing(self):
        fit1 = SimpleExpSmoothing(
            self.oildata_oil, initialization_method="legacy-heuristic"
        ).fit(0.2, optimized=False)
        fit2 = SimpleExpSmoothing(
            self.oildata_oil, initialization_method="legacy-heuristic"
        ).fit(0.6, optimized=False)
        fit3 = SimpleExpSmoothing(
            self.oildata_oil, initialization_method="estimated"
        ).fit()
        assert_almost_equal(fit1.forecast(1), [484.802468], 4)
        assert_almost_equal(
            fit1.level,
            [
                446.65652290,
                448.21987962,
                449.7084985,
                444.49324656,
                446.84886283,
                445.59670028,
                441.54386424,
                450.26498098,
                461.4216172,
                474.49569042,
                482.45033014,
                484.80246797,
            ],
            4,
        )
        assert_almost_equal(fit2.forecast(1), [501.837461], 4)
        assert_almost_equal(fit3.forecast(1), [496.493543], 4)
        assert_almost_equal(fit3.params["smoothing_level"], 0.891998, 4)
        assert_almost_equal(fit3.params["initial_level"], 447.478440, 3)

    def test_holt(self):
        fit1 = Holt(
            self.air_ausair, initialization_method="legacy-heuristic"
        ).fit(smoothing_level=0.8, smoothing_trend=0.2, optimized=False)
        fit2 = Holt(
            self.air_ausair,
            exponential=True,
            initialization_method="legacy-heuristic",
        ).fit(smoothing_level=0.8, smoothing_trend=0.2, optimized=False)
        fit3 = Holt(
            self.air_ausair,
            damped_trend=True,
            initialization_method="estimated",
        ).fit(smoothing_level=0.8, smoothing_trend=0.2)
        assert_almost_equal(
            fit1.forecast(5), [43.76, 45.59, 47.43, 49.27, 51.10], 2
        )
        assert_almost_equal(
            fit1.trend,
            [
                3.617628,
                3.59006512,
                3.33438212,
                3.23657639,
                2.69263502,
                2.46388914,
                2.2229097,
                1.95959226,
                1.47054601,
                1.3604894,
                1.28045881,
                1.20355193,
                1.88267152,
                2.09564416,
                1.83655482,
            ],
            4,
        )
        assert_almost_equal(
            fit1.fittedfcast,
            [
                21.8601,
                22.032368,
                25.48461872,
                27.54058587,
                30.28813356,
                30.26106173,
                31.58122149,
                32.599234,
                33.24223906,
                32.26755382,
                33.07776017,
                33.95806605,
                34.77708354,
                40.05535303,
                43.21586036,
                43.75696849,
            ],
            4,
        )
        assert_almost_equal(
            fit2.forecast(5), [44.60, 47.24, 50.04, 53.01, 56.15], 2
        )
        assert_almost_equal(
            fit3.forecast(5), [42.85, 43.81, 44.66, 45.41, 46.06], 2
        )

    @pytest.mark.smoke
    def test_holt_damp_fit(self):
        # Smoke test for parameter estimation
        fit1 = SimpleExpSmoothing(
            self.livestock2_livestock, initialization_method="estimated"
        ).fit()
        mod4 = Holt(
            self.livestock2_livestock,
            damped_trend=True,
            initialization_method="estimated",
        )
        fit4 = mod4.fit(damping_trend=0.98, method="least_squares")
        mod5 = Holt(
            self.livestock2_livestock,
            exponential=True,
            damped_trend=True,
            initialization_method="estimated",
        )
        fit5 = mod5.fit()
        # We accept the below values as we getting a better SSE than text book
        assert_almost_equal(fit1.params["smoothing_level"], 1.00, 2)
        assert_almost_equal(fit1.params["smoothing_trend"], np.NaN, 2)
        assert_almost_equal(fit1.params["damping_trend"], np.NaN, 2)
        assert_almost_equal(fit1.params["initial_level"], 263.96, 1)
        assert_almost_equal(fit1.params["initial_trend"], np.NaN, 2)
        assert_almost_equal(fit1.sse, 6761.35, 2)  # 6080.26
        assert isinstance(fit1.summary().as_text(), str)

        assert_almost_equal(fit4.params["smoothing_level"], 0.98, 2)
        assert_almost_equal(fit4.params["smoothing_trend"], 0.00, 2)
        assert_almost_equal(fit4.params["damping_trend"], 0.98, 2)
        assert_almost_equal(fit4.params["initial_level"], 257.36, 2)
        assert_almost_equal(fit4.params["initial_trend"], 6.64, 2)
        assert_almost_equal(fit4.sse, 6036.56, 2)  # 6080.26
        assert isinstance(fit4.summary().as_text(), str)

        assert_almost_equal(fit5.params["smoothing_level"], 0.97, 2)
        assert_almost_equal(fit5.params["smoothing_trend"], 0.00, 2)
        assert_almost_equal(fit5.params["damping_trend"], 0.98, 2)
        assert_almost_equal(fit5.params["initial_level"], 258.95, 1)
        assert_almost_equal(fit5.params["initial_trend"], 1.04, 2)
        assert_almost_equal(fit5.sse, 6082.00, 0)  # 6100.11
        assert isinstance(fit5.summary().as_text(), str)

    def test_holt_damp_r(self):
        # Test the damping parameters against the R forecast packages `ets`
        # library(ets)
        # livestock2_livestock <- c(...)
        # res <- ets(livestock2_livestock, model='AAN', damped_trend=TRUE,
        #            phi=0.98)
        mod = Holt(
            self.livestock2_livestock,
            damped_trend=True,
            initialization_method="estimated",
        )
        params = {
            "smoothing_level": 0.97402626,
            "smoothing_trend": 0.00010006,
            "damping_trend": 0.98,
            "initial_level": 252.59039965,
            "initial_trend": 6.90265918,
        }
        with mod.fix_params(params):
            fit = mod.fit(optimized=False)

        # Check that we captured the parameters correctly
        for key in params.keys():
            assert_allclose(fit.params[key], params[key])

        with mod.fix_params(params):
            opt_fit = mod.fit(optimized=True)
        assert_allclose(fit.sse, opt_fit.sse)
        assert_allclose(
            opt_fit.params["initial_trend"], params["initial_trend"]
        )
        alt_params = {k: v for k, v in params.items() if "level" not in k}
        with mod.fix_params(alt_params):
            alt_fit = mod.fit(optimized=True)
        assert not np.allclose(alt_fit.trend.iloc[0], opt_fit.trend.iloc[0])
        # Summary output
        # print(res$mse)
        assert_allclose(fit.sse / mod.nobs, 195.4397924865488, atol=1e-3)
        # print(res$aicc)
        # TODO: this fails - different AICC definition?
        # assert_allclose(fit.aicc, 282.386426659408, atol=1e-3)
        # print(res$bic)
        # TODO: this fails - different BIC definition?
        # assert_allclose(fit.bic, 287.1563626818338)

        # print(res$states[,'l'])
        # note: this array includes the initial level
        desired = [
            252.5903996514365,
            263.7992355246843,
            268.3623324350207,
            261.0312983437606,
            266.6590942700923,
            277.3958197247272,
            283.8256217863908,
            290.2962560621914,
            292.5701438129583,
            300.7655919939834,
            309.2118057241649,
            318.2377698496536,
            329.2238709362550,
            338.7709778307978,
            339.3669793596703,
            329.0127022356033,
            314.7684267018998,
            314.5948077575944,
            321.3612035017972,
            329.6924360833211,
            346.0712138652086,
            352.2534120008911,
            348.5862874190927,
            415.8839400693967,
            417.2018843196238,
            417.8435306633725,
            412.4857261252961,
            412.0647865321129,
            395.2500605270393,
            401.4367438266322,
            408.1907701386275,
            414.1814574903921,
        ]
        assert_allclose(np.r_[fit.params["initial_level"], fit.level], desired)

        # print(res$states[,'b'])
        # note: this array includes the initial slope
        desired = [
            6.902659175332394,
            6.765062519124909,
            6.629548973536494,
            6.495537532917715,
            6.365550989616566,
            6.238702070454378,
            6.113960476763530,
            5.991730467006233,
            5.871526257315264,
            5.754346516684953,
            5.639547926790058,
            5.527116419415724,
            5.417146212898857,
            5.309238662451385,
            5.202580636191761,
            5.096941655567694,
            4.993026494493987,
            4.892645486210410,
            4.794995106664251,
            4.699468310763351,
            4.606688340205792,
            4.514725879754355,
            4.423600168391240,
            4.341595902295941,
            4.254462303550087,
            4.169010676686062,
            4.084660399498803,
            4.002512751871354,
            3.920332298146730,
            3.842166514133902,
            3.765630194200260,
            3.690553892582855,
        ]
        # TODO: not sure why the precision is so low here...
        assert_allclose(
            np.r_[fit.params["initial_trend"], fit.trend], desired, atol=1e-3
        )

        # print(res$fitted)
        desired = [
            259.3550056432622,
            270.4289967934267,
            274.8592904290865,
            267.3969251260200,
            272.8973342399166,
            283.5097477537724,
            289.8173030536191,
            296.1681519198575,
            298.3242395451272,
            306.4048515803347,
            314.7385626924191,
            323.6543439406810,
            334.5326742248959,
            343.9740317200002,
            344.4655083831382,
            334.0077050580596,
            319.6615926665040,
            319.3896003340806,
            326.0602987063282,
            334.2979150278692,
            350.5857684386102,
            356.6778433630504,
            352.9214155841161,
            420.1387040536467,
            421.3712573771029,
            421.9291611265248,
            416.4886933168049,
            415.9872490289468,
            399.0919861792231,
            405.2020670104834,
            411.8810877289437,
        ]
        assert_allclose(fit.fittedvalues, desired, atol=1e-3)

        # print(forecast(res)$mean)
        desired = [
            417.7982003051233,
            421.3426082635598,
            424.8161280628277,
            428.2201774661102,
            431.5561458813270,
            434.8253949282395,
            438.0292589942138,
            441.1690457788685,
            444.2460368278302,
            447.2614880558126,
        ]
        assert_allclose(fit.forecast(10), desired, atol=1e-4)

    def test_hw_seasonal(self):
        mod = ExponentialSmoothing(
            self.aust,
            seasonal_periods=4,
            trend="additive",
            seasonal="additive",
            initialization_method="estimated",
            use_boxcox=True,
        )
        fit1 = mod.fit()
        assert_almost_equal(
            fit1.forecast(8),
            [59.96, 38.63, 47.48, 51.89, 62.81, 41.0, 50.06, 54.57],
            2,
        )

    def test_hw_seasonal_add_mul(self):
        mod2 = ExponentialSmoothing(
            self.aust,
            seasonal_periods=4,
            trend="add",
            seasonal="mul",
            initialization_method="estimated",
            use_boxcox=True,
        )
        fit2 = mod2.fit()
        assert_almost_equal(
            fit2.forecast(8),
            [61.69, 37.37, 47.22, 52.03, 65.08, 39.34, 49.72, 54.79],
            2,
        )
        ExponentialSmoothing(
            self.aust,
            seasonal_periods=4,
            trend="mul",
            seasonal="add",
            initialization_method="estimated",
            use_boxcox=0.0,
        ).fit()
        ExponentialSmoothing(
            self.aust,
            seasonal_periods=4,
            trend="multiplicative",
            seasonal="multiplicative",
            initialization_method="estimated",
            use_boxcox=0.0,
        ).fit()
        # Skip since estimator is unstable
        # assert_almost_equal(fit5.forecast(1), [60.60], 2)
        # assert_almost_equal(fit6.forecast(1), [61.47], 2)

    def test_hw_seasonal_buggy(self):
        fit3 = ExponentialSmoothing(
            self.aust,
            seasonal_periods=4,
            seasonal="add",
            initialization_method="estimated",
            use_boxcox=True,
        ).fit()
        assert_almost_equal(
            fit3.forecast(8),
            [
                59.487190,
                35.758854,
                44.600641,
                47.751384,
                59.487190,
                35.758854,
                44.600641,
                47.751384,
            ],
            2,
        )
        fit4 = ExponentialSmoothing(
            self.aust,
            seasonal_periods=4,
            seasonal="mul",
            initialization_method="estimated",
            use_boxcox=True,
        ).fit()
        assert_almost_equal(
            fit4.forecast(8),
            [
                59.26155037,
                35.27811302,
                44.00438543,
                47.97732693,
                59.26155037,
                35.27811302,
                44.00438543,
                47.97732693,
            ],
            2,
        )


@pytest.mark.parametrize(
    "trend_seasonal", (("mul", None), (None, "mul"), ("mul", "mul"))
)
def test_negative_multipliative(trend_seasonal):
    trend, seasonal = trend_seasonal
    y = -np.ones(100)
    with pytest.raises(ValueError):
        ExponentialSmoothing(
            y, trend=trend, seasonal=seasonal, seasonal_periods=10
        )


@pytest.mark.parametrize("seasonal", SEASONALS)
def test_dampen_no_trend(seasonal):
    with pytest.raises(TypeError):
        ExponentialSmoothing(
            housing_data,
            trend=False,
            seasonal=seasonal,
            damped_trend=True,
            seasonal_periods=10,
        )


@pytest.mark.parametrize("seasonal", ("add", "mul"))
def test_invalid_seasonal(seasonal):
    y = pd.Series(
        -np.ones(100), index=pd.date_range("2000-1-1", periods=100, freq="MS")
    )
    with pytest.raises(ValueError):
        ExponentialSmoothing(y, seasonal=seasonal, seasonal_periods=1)


def test_2d_data():
    with pytest.raises(ValueError):
        ExponentialSmoothing(
            pd.concat([housing_data, housing_data], axis=1)
        ).fit()


def test_infer_freq():
    hd2 = housing_data.copy()
    hd2.index = list(hd2.index)
    with warnings.catch_warnings(record=True) as w:
        mod = ExponentialSmoothing(
            hd2, trend="add", seasonal="add", initialization_method="estimated"
        )
        assert len(w) == 1
        assert "ValueWarning" in str(w[0])
    assert mod.seasonal_periods == 12


@pytest.mark.parametrize("trend", TRENDS)
@pytest.mark.parametrize("seasonal", SEASONALS)
def test_start_params(trend, seasonal):
    mod = ExponentialSmoothing(
        housing_data,
        trend=trend,
        seasonal=seasonal,
        initialization_method="estimated",
    )
    res = mod.fit()
    res2 = mod.fit(
        method="basinhopping",
        minimize_kwargs={"minimizer_kwargs": {"method": "SLSQP"}},
    )
    assert isinstance(res.summary().as_text(), str)
    assert res2.sse < 1.01 * res.sse
    assert isinstance(res2.params, dict)


def test_no_params_to_optimize():
    mod = ExponentialSmoothing(
        housing_data,
        initial_level=housing_data.iloc[0],
        initialization_method="known",
    )
    mod.fit(smoothing_level=0.5)


def test_invalid_start_param_length():
    mod = ExponentialSmoothing(housing_data, initialization_method="estimated")
    with pytest.raises(ValueError):
        mod.fit(start_params=np.array([0.5]))


def test_basin_hopping(reset_randomstate):
    mod = ExponentialSmoothing(
        housing_data, trend="add", initialization_method="estimated"
    )
    res = mod.fit()
    res2 = mod.fit(method="basinhopping")
    assert isinstance(res.summary().as_text(), str)
    assert isinstance(res2.summary().as_text(), str)
    # Basin hopping occasionally produces a slightly larger objective
    tol = 1e-5
    assert res2.sse <= res.sse + tol
    res3 = mod.fit(method="basinhopping")
    assert_almost_equal(res2.sse, res3.sse, decimal=2)


def test_debiased():
    mod = ExponentialSmoothing(
        housing_data, trend="add", initialization_method="estimated"
    )
    res = mod.fit()
    res2 = mod.fit(remove_bias=True)
    assert np.any(res.fittedvalues != res2.fittedvalues)
    err2 = housing_data.iloc[:, 0] - res2.fittedvalues
    assert_almost_equal(err2.mean(), 0.0)
    assert isinstance(res.summary().as_text(), str)
    assert isinstance(res2.summary().as_text(), str)


@pytest.mark.smoke
@pytest.mark.parametrize("trend", TRENDS)
@pytest.mark.parametrize("seasonal", SEASONALS)
def test_float_boxcox(trend, seasonal):
    res = ExponentialSmoothing(
        housing_data,
        trend=trend,
        seasonal=seasonal,
        initialization_method="estimated",
        use_boxcox=0.5,
    ).fit()
    assert_allclose(res.params["use_boxcox"], 0.5)
    res = ExponentialSmoothing(
        housing_data,
        trend=trend,
        seasonal=seasonal,
        use_boxcox=0.5,
    ).fit()
    assert_allclose(res.params["use_boxcox"], 0.5)


@pytest.mark.parametrize("trend", TRENDS)
@pytest.mark.parametrize("seasonal", SEASONALS)
def test_equivalence_cython_python(trend, seasonal):
    mod = ExponentialSmoothing(
        housing_data,
        trend=trend,
        seasonal=seasonal,
        initialization_method="estimated",
    )

    # Overflow in mul-mul case fixed
    res = mod.fit()
    assert isinstance(res.summary().as_text(), str)

    params = res.params
    nobs = housing_data.shape[0]
    y = np.squeeze(np.asarray(housing_data))
    m = 12 if seasonal else 0
    p = np.zeros(6 + m)
    alpha = params["smoothing_level"]
    beta = params["smoothing_trend"]
    gamma = params["smoothing_seasonal"]
    phi = params["damping_trend"]
    phi = 1.0 if np.isnan(phi) else phi
    l0 = params["initial_level"]
    b0 = params["initial_trend"]
    p[:6] = alpha, beta, gamma, l0, b0, phi
    if seasonal:
        p[6:] = params["initial_seasons"]
    xi = np.ones_like(p).astype(int)

    p_copy = p.copy()

    bounds = np.array([[0.0, 1.0]] * 3)
    py_func = PY_SMOOTHERS[(seasonal, trend)]
    cy_func = SMOOTHERS[(seasonal, trend)]
    py_hw_args = PyHoltWintersArgs(xi, p_copy, bounds, y, m, nobs, False)
    cy_hw_args = HoltWintersArgs(xi, p_copy, bounds, y, m, nobs, False)

    sse_cy = cy_func(p, cy_hw_args)
    sse_py = py_func(p, py_hw_args)
    assert_allclose(sse_py, sse_cy)
    sse_py = py_func(p, cy_hw_args)
    assert_allclose(sse_py, sse_cy)


def test_direct_holt_add():
    mod = SimpleExpSmoothing(housing_data, initialization_method="estimated")
    res = mod.fit()
    assert isinstance(res.summary().as_text(), str)
    x = np.squeeze(np.asarray(mod.endog))
    alpha = res.params["smoothing_level"]
    l, b, f, _, xhat = _simple_dbl_exp_smoother(
        x, alpha, beta=0.0, l0=res.params["initial_level"], b0=0.0, nforecast=5
    )

    assert_allclose(l, res.level)
    assert_allclose(f, res.level.iloc[-1] * np.ones(5))
    assert_allclose(f, res.forecast(5))

    mod = ExponentialSmoothing(
        housing_data, trend="add", initialization_method="estimated"
    )
    res = mod.fit()
    x = np.squeeze(np.asarray(mod.endog))
    alpha = res.params["smoothing_level"]
    beta = res.params["smoothing_trend"]
    l, b, f, _, xhat = _simple_dbl_exp_smoother(
        x,
        alpha,
        beta=beta,
        l0=res.params["initial_level"],
        b0=res.params["initial_trend"],
        nforecast=5,
    )

    assert_allclose(xhat, res.fittedvalues)
    assert_allclose(l + b, res.level + res.trend)
    assert_allclose(l, res.level)
    assert_allclose(b, res.trend)
    assert_allclose(
        f, res.level.iloc[-1] + res.trend.iloc[-1] * np.array([1, 2, 3, 4, 5])
    )
    assert_allclose(f, res.forecast(5))
    assert isinstance(res.summary().as_text(), str)


def test_integer_array(reset_randomstate):
    rs = np.random.RandomState(12345)
    e = 10 * rs.standard_normal((1000, 2))
    y_star = np.cumsum(e[:, 0])
    y = y_star + e[:, 1]
    y = y.astype(int)
    res = ExponentialSmoothing(
        y, trend="add", initialization_method="estimated"
    ).fit()
    assert res.params["smoothing_level"] != 0.0


def test_damping_trend_zero():
    endog = np.arange(10)
    mod = ExponentialSmoothing(
        endog,
        trend="add",
        damped_trend=True,
        initialization_method="estimated",
    )
    res1 = mod.fit(smoothing_level=1, smoothing_trend=0.0, damping_trend=1e-20)
    pred1 = res1.predict(start=0)
    assert_allclose(pred1, np.r_[0.0, np.arange(9)], atol=1e-10)

    res2 = mod.fit(smoothing_level=1, smoothing_trend=0.0, damping_trend=0)
    pred2 = res2.predict(start=0)
    assert_allclose(pred2, np.r_[0.0, np.arange(9)], atol=1e-10)

    assert_allclose(pred1, pred2, atol=1e-10)


def test_different_inputs():
    array_input_add = [10, 20, 30, 40, 50]
    series_index_add = pd.date_range(
        start="2000-1-1", periods=len(array_input_add)
    )
    series_input_add = pd.Series(array_input_add, series_index_add)

    array_input_mul = [2, 4, 8, 16, 32]
    series_index_mul = pd.date_range(
        start="2000-1-1", periods=len(array_input_mul)
    )
    series_input_mul = pd.Series(array_input_mul, series_index_mul)

    fit1 = ExponentialSmoothing(array_input_add, trend="add").fit()
    fit2 = ExponentialSmoothing(series_input_add, trend="add").fit()
    fit3 = ExponentialSmoothing(array_input_mul, trend="mul").fit()
    fit4 = ExponentialSmoothing(series_input_mul, trend="mul").fit()

    assert_almost_equal(fit1.predict(), [60], 1)
    assert_almost_equal(fit1.predict(start=5, end=7), [60, 70, 80], 1)
    assert_almost_equal(fit2.predict(), [60], 1)
    assert_almost_equal(
        fit2.predict(start="2000-1-6", end="2000-1-8"), [60, 70, 80], 1
    )
    assert_almost_equal(fit3.predict(), [64], 1)
    assert_almost_equal(fit3.predict(start=5, end=7), [64, 128, 256], 1)
    assert_almost_equal(fit4.predict(), [64], 1)
    assert_almost_equal(
        fit4.predict(start="2000-1-6", end="2000-1-8"), [64, 128, 256], 1
    )


@pytest.fixture
def austourists():
    # austourists dataset from fpp2 package
    # https://cran.r-project.org/web/packages/fpp2/index.html
    data = [
        30.05251,
        19.14850,
        25.31769,
        27.59144,
        32.07646,
        23.48796,
        28.47594,
        35.12375,
        36.83848,
        25.00702,
        30.72223,
        28.69376,
        36.64099,
        23.82461,
        29.31168,
        31.77031,
        35.17788,
        19.77524,
        29.60175,
        34.53884,
        41.27360,
        26.65586,
        28.27986,
        35.19115,
        42.20566,
        24.64917,
        32.66734,
        37.25735,
        45.24246,
        29.35048,
        36.34421,
        41.78208,
        49.27660,
        31.27540,
        37.85063,
        38.83704,
        51.23690,
        31.83855,
        41.32342,
        42.79900,
        55.70836,
        33.40714,
        42.31664,
        45.15712,
        59.57608,
        34.83733,
        44.84168,
        46.97125,
        60.01903,
        38.37118,
        46.97586,
        50.73380,
        61.64687,
        39.29957,
        52.67121,
        54.33232,
        66.83436,
        40.87119,
        51.82854,
        57.49191,
        65.25147,
        43.06121,
        54.76076,
        59.83447,
        73.25703,
        47.69662,
        61.09777,
        66.05576,
    ]
    index = pd.date_range("1999-03-01", "2015-12-01", freq="3MS")
    return pd.Series(data, index)


@pytest.fixture
def simulate_expected_results_r():
    """
    obtained from ets.simulate in the R package forecast, data is from fpp2
    package.

    library(magrittr)
    library(fpp2)
    library(forecast)
    concat <- function(...) {
      return(paste(..., sep=""))
    }
    error <- c("A", "M")
    trend <- c("A", "M", "N")
    seasonal <- c("A", "M", "N")
    models <- outer(error, trend, FUN = "concat") %>%
      outer(seasonal, FUN = "concat") %>% as.vector
    # innov from np.random.seed(0); np.random.randn(4)
    innov <- c(1.76405235, 0.40015721, 0.97873798, 2.2408932)
    params <- expand.grid(models, c(TRUE, FALSE))
    results <- apply(params, 1, FUN = function(p) {
      tryCatch(
        simulate(ets(austourists, model = p[1], damped = as.logical(p[2])),
                 innov = innov),
        error = function(e) c(NA, NA, NA, NA))
    }) %>% t
    rownames(results) <- apply(params, 1, FUN = function(x) paste(x[1], x[2]))
    """
    damped = {
        "AAA": [77.84173, 52.69818, 65.83254, 71.85204],
        "MAA": [207.81653, 136.97700, 253.56234, 588.95800],
        "MAM": [215.83822, 127.17132, 269.09483, 704.32105],
        "MMM": [216.52591, 132.47637, 283.04889, 759.08043],
        "AAN": [62.51423, 61.87381, 63.14735, 65.11360],
        "MAN": [168.25189, 90.46201, 133.54769, 232.81738],
        "MMN": [167.97747, 90.59675, 134.20300, 235.64502],
    }
    undamped = {
        "AAA": [77.10860, 51.51669, 64.46857, 70.36349],
        "MAA": [209.23158, 149.62943, 270.65579, 637.03828],
        "ANA": [77.09320, 51.52384, 64.36231, 69.84786],
        "MNA": [207.86986, 169.42706, 313.97960, 793.97948],
        "MAM": [214.45750, 106.19605, 211.61304, 492.12223],
        "MMM": [221.01861, 158.55914, 403.22625, 1389.33384],
        "MNM": [215.00997, 140.93035, 309.92465, 875.07985],
        "AAN": [63.66619, 63.09571, 64.45832, 66.51967],
        "MAN": [172.37584, 91.51932, 134.11221, 230.98970],
        "MMN": [169.88595, 97.33527, 142.97017, 252.51834],
        "ANN": [60.53589, 59.51851, 60.17570, 61.63011],
        "MNN": [163.01575, 112.58317, 172.21992, 338.93918],
    }
    return {True: damped, False: undamped}


@pytest.fixture
def simulate_fit_state_r():
    """
    The final state from the R model fits to get an exact comparison
    Obtained with this R script:

    library(magrittr)
    library(fpp2)
    library(forecast)

    concat <- function(...) {
      return(paste(..., sep=""))
    }

    as_dict_string <- function(named) {
      string <- '{'
      for (name in names(named)) {
        string <- concat(string, "\"", name, "\": ", named[name], ", ")
      }
      string <- concat(string, '}')
      return(string)
    }

    get_var <- function(named, name) {
      if (name %in% names(named))
        val <- c(named[name])
      else
        val <- c(NaN)
      names(val) <- c(name)
      return(val)
    }

    error <- c("A", "M")
    trend <- c("A", "M", "N")
    seasonal <- c("A", "M", "N")
    models <- outer(error, trend, FUN = "concat") %>%
      outer(seasonal, FUN = "concat") %>% as.vector

    # innov from np.random.seed(0); np.random.randn(4)
    innov <- c(1.76405235, 0.40015721, 0.97873798, 2.2408932)
    n <- length(austourists) + 1

    # print fit parameters and final states
    for (damped in c(TRUE, FALSE)) {
      print(paste("damped =", damped))
      for (model in models) {
        state <- tryCatch((function(){
          fit <- ets(austourists, model = model, damped = damped)
          pars <- c()
          # alpha, beta, gamma, phi
          for (name in c("alpha", "beta", "gamma", "phi")) {
            pars <- c(pars, get_var(fit$par, name))
          }
          # l, b, s1, s2, s3, s4
          states <- c()
          for (name in c("l", "b", "s1", "s2", "s3", "s4"))
            states <- c(states, get_var(fit$states[n,], name))
          c(pars, states)
        })(),
        error = function(e) rep(NA, 10))
        cat(concat("\"", model, "\": ", as_dict_string(state), ",\n"))
      }
    }
    """
    damped = {
        "AAA": {
            "alpha": 0.35445427317618,
            "beta": 0.0320074905894167,
            "gamma": 0.399933869627979,
            "phi": 0.979999965983533,
            "l": 62.003405788717,
            "b": 0.706524957599738,
            "s1": 3.58786406600866,
            "s2": -0.0747450283892903,
            "s3": -11.7569356589817,
            "s4": 13.3818805055271,
        },
        "MAA": {
            "alpha": 0.31114284033284,
            "beta": 0.0472138763848083,
            "gamma": 0.309502324693322,
            "phi": 0.870889202791893,
            "l": 59.2902342851514,
            "b": 0.62538315801909,
            "s1": 5.66660224738038,
            "s2": 2.16097311633352,
            "s3": -9.20020909069337,
            "s4": 15.3505801601698,
        },
        "MAM": {
            "alpha": 0.483975835390643,
            "beta": 0.00351728130401287,
            "gamma": 0.00011309784353818,
            "phi": 0.979999998322032,
            "l": 63.0042707536293,
            "b": 0.275035160634846,
            "s1": 1.03531670491486,
            "s2": 0.960515682506077,
            "s3": 0.770086097577864,
            "s4": 1.23412213281709,
        },
        "MMM": {
            "alpha": 0.523526123191035,
            "beta": 0.000100021136675999,
            "gamma": 0.000100013723372502,
            "phi": 0.971025672907157,
            "l": 63.2030316675533,
            "b": 1.00458391644788,
            "s1": 1.03476354353096,
            "s2": 0.959953222294316,
            "s3": 0.771346403552048,
            "s4": 1.23394845160922,
        },
        "AAN": {
            "alpha": 0.014932817259302,
            "beta": 0.0149327068053362,
            "gamma": np.nan,
            "phi": 0.979919958387887,
            "l": 60.0651024395378,
            "b": 0.699112782133822,
            "s1": np.nan,
            "s2": np.nan,
            "s3": np.nan,
            "s4": np.nan,
        },
        "MAN": {
            "alpha": 0.0144217343786778,
            "beta": 0.0144216994589862,
            "gamma": np.nan,
            "phi": 0.979999719878659,
            "l": 60.1870032363649,
            "b": 0.698421913047609,
            "s1": np.nan,
            "s2": np.nan,
            "s3": np.nan,
            "s4": np.nan,
        },
        "MMN": {
            "alpha": 0.015489181776072,
            "beta": 0.0154891632646377,
            "gamma": np.nan,
            "phi": 0.975139118496093,
            "l": 60.1855946424729,
            "b": 1.00999589024928,
            "s1": np.nan,
            "s2": np.nan,
            "s3": np.nan,
            "s4": np.nan,
        },
    }
    undamped = {
        "AAA": {
            "alpha": 0.20281951627363,
            "beta": 0.000169786227368617,
            "gamma": 0.464523797585052,
            "phi": np.nan,
            "l": 62.5598121416791,
            "b": 0.578091734736357,
            "s1": 2.61176734723357,
            "s2": -1.24386240029203,
            "s3": -12.9575427049515,
            "s4": 12.2066400808086,
        },
        "MAA": {
            "alpha": 0.416371920801538,
            "beta": 0.000100008012920072,
            "gamma": 0.352943901103959,
            "phi": np.nan,
            "l": 62.0497742976079,
            "b": 0.450130087198346,
            "s1": 3.50368220490457,
            "s2": -0.0544297321113539,
            "s3": -11.6971093199679,
            "s4": 13.1974985095916,
        },
        "ANA": {
            "alpha": 0.54216694759434,
            "beta": np.nan,
            "gamma": 0.392030170511872,
            "phi": np.nan,
            "l": 57.606831186929,
            "b": np.nan,
            "s1": 8.29613785790501,
            "s2": 4.6033791939889,
            "s3": -7.43956343440823,
            "s4": 17.722316385643,
        },
        "MNA": {
            "alpha": 0.532842556756286,
            "beta": np.nan,
            "gamma": 0.346387433608713,
            "phi": np.nan,
            "l": 58.0372808528325,
            "b": np.nan,
            "s1": 7.70802088750111,
            "s2": 4.14885814748503,
            "s3": -7.72115936226225,
            "s4": 17.1674660340923,
        },
        "MAM": {
            "alpha": 0.315621390571192,
            "beta": 0.000100011993615961,
            "gamma": 0.000100051297784532,
            "phi": np.nan,
            "l": 62.4082004238551,
            "b": 0.513327867101983,
            "s1": 1.03713425342421,
            "s2": 0.959607104686072,
            "s3": 0.770172817592091,
            "s4": 1.23309264451638,
        },
        "MMM": {
            "alpha": 0.546068965886,
            "beta": 0.0737816453485457,
            "gamma": 0.000100031693302807,
            "phi": np.nan,
            "l": 63.8203866275649,
            "b": 1.01833305374778,
            "s1": 1.03725227137871,
            "s2": 0.961177239042923,
            "s3": 0.771173487523454,
            "s4": 1.23036313932852,
        },
        "MNM": {
            "alpha": 0.608993139624813,
            "beta": np.nan,
            "gamma": 0.000167258612971303,
            "phi": np.nan,
            "l": 63.1472153330648,
            "b": np.nan,
            "s1": 1.0384840572776,
            "s2": 0.961456755855531,
            "s3": 0.768427399477366,
            "s4": 1.23185085956321,
        },
        "AAN": {
            "alpha": 0.0097430554119077,
            "beta": 0.00974302759255084,
            "gamma": np.nan,
            "phi": np.nan,
            "l": 61.1430969243248,
            "b": 0.759041621012503,
            "s1": np.nan,
            "s2": np.nan,
            "s3": np.nan,
            "s4": np.nan,
        },
        "MAN": {
            "alpha": 0.0101749952821338,
            "beta": 0.0101749138539332,
            "gamma": np.nan,
            "phi": np.nan,
            "l": 61.6020426238699,
            "b": 0.761407500773051,
            "s1": np.nan,
            "s2": np.nan,
            "s3": np.nan,
            "s4": np.nan,
        },
        "MMN": {
            "alpha": 0.0664382968951546,
            "beta": 0.000100001678373356,
            "gamma": np.nan,
            "phi": np.nan,
            "l": 60.7206911970871,
            "b": 1.01221899136391,
            "s1": np.nan,
            "s2": np.nan,
            "s3": np.nan,
            "s4": np.nan,
        },
        "ANN": {
            "alpha": 0.196432515825523,
            "beta": np.nan,
            "gamma": np.nan,
            "phi": np.nan,
            "l": 58.7718395431632,
            "b": np.nan,
            "s1": np.nan,
            "s2": np.nan,
            "s3": np.nan,
            "s4": np.nan,
        },
        "MNN": {
            "alpha": 0.205985314333856,
            "beta": np.nan,
            "gamma": np.nan,
            "phi": np.nan,
            "l": 58.9770839944419,
            "b": np.nan,
            "s1": np.nan,
            "s2": np.nan,
            "s3": np.nan,
            "s4": np.nan,
        },
    }
    return {True: damped, False: undamped}


@pytest.mark.parametrize("trend", TRENDS)
@pytest.mark.parametrize("seasonal", SEASONALS)
@pytest.mark.parametrize("damped", (True, False))
@pytest.mark.parametrize("error", ("add", "mul"))
def test_simulate_expected_r(
    trend,
    seasonal,
    damped,
    error,
    austourists,
    simulate_expected_results_r,
    simulate_fit_state_r,
):
    """
    Test for :meth:``statsmodels.tsa.holtwinters.HoltWintersResults``.

    The tests are using the implementation in the R package ``forecast`` as
    reference, and example data is taken from ``fpp2`` (package and book).
    """

    short_name = {"add": "A", "mul": "M", None: "N"}
    model_name = short_name[error] + short_name[trend] + short_name[seasonal]
    if model_name in simulate_expected_results_r[damped]:
        expected = np.asarray(simulate_expected_results_r[damped][model_name])
        state = simulate_fit_state_r[damped][model_name]
    else:
        return

    # create HoltWintersResults object with same parameters as in R
    fit = ExponentialSmoothing(
        austourists,
        seasonal_periods=4,
        trend=trend,
        seasonal=seasonal,
        damped_trend=damped,
    ).fit(
        smoothing_level=state["alpha"],
        smoothing_trend=state["beta"],
        smoothing_seasonal=state["gamma"],
        damping_trend=state["phi"],
        optimized=False,
    )

    # set the same final state as in R
    fit._level[-1] = state["l"]
    fit._trend[-1] = state["b"]
    fit._season[-1] = state["s1"]
    fit._season[-2] = state["s2"]
    fit._season[-3] = state["s3"]
    fit._season[-4] = state["s4"]

    # for MMM with damped trend the fit fails
    if np.any(np.isnan(fit.fittedvalues)):
        return

    innov = np.asarray([[1.76405235, 0.40015721, 0.97873798, 2.2408932]]).T
    sim = fit.simulate(4, repetitions=1, error=error, random_errors=innov)

    assert_almost_equal(expected, sim.values, 5)


def test_simulate_keywords(austourists):
    """
    check whether all keywords are accepted and work without throwing errors.
    """
    fit = ExponentialSmoothing(
        austourists,
        seasonal_periods=4,
        trend="add",
        seasonal="add",
        damped_trend=True,
        initialization_method="estimated",
    ).fit()

    # test anchor
    assert_almost_equal(
        fit.simulate(4, anchor=0, random_state=0).values,
        fit.simulate(4, anchor="start", random_state=0).values,
    )
    assert_almost_equal(
        fit.simulate(4, anchor=-1, random_state=0).values,
        fit.simulate(4, anchor="2015-12-01", random_state=0).values,
    )
    assert_almost_equal(
        fit.simulate(4, anchor="end", random_state=0).values,
        fit.simulate(4, anchor="2016-03-01", random_state=0).values,
    )

    # test different random error options
    fit.simulate(4, repetitions=10, random_errors=scipy.stats.norm)
    fit.simulate(4, repetitions=10, random_errors=scipy.stats.norm())

    fit.simulate(4, repetitions=10, random_errors=np.random.randn(4, 10))
    fit.simulate(4, repetitions=10, random_errors="bootstrap")

    # test seeding
    res = fit.simulate(4, repetitions=10, random_state=10).values
    res2 = fit.simulate(
        4, repetitions=10, random_state=np.random.RandomState(10)
    ).values
    assert np.all(res == res2)


def test_simulate_boxcox(austourists):
    """
    check if simulation results with boxcox fits are reasonable
    """
    fit = ExponentialSmoothing(
        austourists,
        seasonal_periods=4,
        trend="add",
        seasonal="mul",
        damped_trend=False,
        initialization_method="estimated",
        use_boxcox=True,
    ).fit()
    expected = fit.forecast(4).values

    res = fit.simulate(4, repetitions=10, random_state=0).values
    mean = np.mean(res, axis=1)

    assert np.all(np.abs(mean - expected) < 5)


@pytest.mark.parametrize("ix", [10, 100, 1000, 2000])
def test_forecast_index(ix):
    # GH 6549
    ts_1 = pd.Series(
        [85601, 89662, 85122, 84400, 78250, 84434, 71072, 70357, 72635, 73210],
        index=range(ix, ix + 10),
    )
    with pytest.warns(ConvergenceWarning):
        model = ExponentialSmoothing(
            ts_1, trend="add", damped_trend=False
        ).fit()
    index = model.forecast(steps=10).index
    assert index[0] == ix + 10
    assert index[-1] == ix + 19


def test_error_dampen():
    with pytest.raises(ValueError, match="Can only dampen the"):
        ExponentialSmoothing(np.ones(100), damped_trend=True)


def test_error_boxcox():
    y = np.random.standard_normal(100)
    with pytest.raises(TypeError, match="use_boxcox must be True"):
        ExponentialSmoothing(y, use_boxcox="a", initialization_method="known")

    mod = ExponentialSmoothing(y**2, use_boxcox=True)
    assert isinstance(mod, ExponentialSmoothing)

    mod = ExponentialSmoothing(
        y**2, use_boxcox=True, initialization_method="legacy-heuristic"
    )
    with pytest.raises(ValueError, match="use_boxcox was set"):
        mod.fit(use_boxcox=False)


def test_error_initialization(ses):
    with pytest.raises(
        ValueError, match="initialization is 'known' but initial_level"
    ):
        ExponentialSmoothing(ses, initialization_method="known")
    with pytest.raises(ValueError, match="initial_trend set but model"):
        ExponentialSmoothing(
            ses,
            initialization_method="known",
            initial_level=1.0,
            initial_trend=1.0,
        )
    with pytest.raises(ValueError, match="initial_seasonal set but model"):
        ExponentialSmoothing(
            ses,
            initialization_method="known",
            initial_level=1.0,
            initial_seasonal=[0.2, 0.3, 0.4, 0.5],
        )
    with pytest.raises(ValueError):
        ExponentialSmoothing(ses, initial_level=1.0)
    with pytest.raises(ValueError):
        ExponentialSmoothing(ses, initial_trend=1.0)
    with pytest.raises(ValueError):
        ExponentialSmoothing(ses, initial_seasonal=[1.0, 0.2, 0.05, 4])
    with pytest.raises(ValueError):
        ExponentialSmoothing(
            ses,
            trend="add",
            initialization_method="known",
            initial_level=1.0,
        )
    with pytest.raises(ValueError):
        ExponentialSmoothing(
            ses,
            trend="add",
            seasonal="add",
            initialization_method="known",
            initial_level=1.0,
            initial_trend=2.0,
        )
    mod = ExponentialSmoothing(
        ses, initialization_method="known", initial_level=1.0
    )
    with pytest.raises(ValueError):
        mod.fit(initial_level=2.0)
    with pytest.raises(ValueError):
        mod.fit(use_basinhopping=True, method="least_squares")


@pytest.mark.parametrize(
    "method",
    [
        "least_squares",
        "basinhopping",
        "L-BFGS-B",
        "TNC",
        "SLSQP",
        "Powell",
        "trust-constr",
    ],
)
def test_alternative_minimizers(method, ses):
    sv = np.array([0.77, 11.00])
    minimize_kwargs = {}
    mod = ExponentialSmoothing(ses, initialization_method="estimated")
    res = mod.fit(
        method=method, start_params=sv, minimize_kwargs=minimize_kwargs
    )
    assert_allclose(res.params["smoothing_level"], 0.77232545, rtol=1e-3)
    assert_allclose(res.params["initial_level"], 11.00359693, rtol=1e-3)
    assert isinstance(res.summary().as_text(), str)


def test_minimizer_kwargs_error(ses):
    mod = ExponentialSmoothing(ses, initialization_method="estimated")
    kwargs = {"args": "anything"}
    with pytest.raises(ValueError):
        mod.fit(minimize_kwargs=kwargs)
    with pytest.raises(ValueError):
        mod.fit(method="least_squares", minimize_kwargs=kwargs)
    kwargs = {"minimizer_kwargs": {"args": "anything"}}
    with pytest.raises(ValueError):
        mod.fit(method="basinhopping", minimize_kwargs=kwargs)
    kwargs = {"minimizer_kwargs": {"method": "SLSQP"}}
    res = mod.fit(method="basinhopping", minimize_kwargs=kwargs)
    assert isinstance(res.params, dict)
    assert isinstance(res.summary().as_text(), str)


@pytest.mark.parametrize(
    "params", [[0.8, 0.3, 0.9], [0.3, 0.8, 0.2], [0.5, 0.6, 0.6]]
)
def test_to_restricted_equiv(params):
    params = np.array(params)
    sel = np.array([True] * 3)
    bounds = np.array([[0.0, 1.0]] * 3)
    assert_allclose(
        to_restricted(params, sel, bounds),
        _test_to_restricted(params, sel.astype(int), bounds),
    )


@pytest.mark.parametrize(
    "params", [[0.8, 0.3, 0.1], [0.3, 0.2, 0.6], [0.5, 0.5, 0.5]]
)
def test_restricted_round_tip(params):
    params = np.array(params)
    sel = np.array([True] * 3)
    bounds = np.array([[0.0, 1.0]] * 3)
    assert_allclose(
        params,
        to_unrestricted(to_restricted(params, sel, bounds), sel, bounds),
    )


def test_bad_bounds(ses):
    bounds = {"bad_key": (0.0, 1.0)}
    with pytest.raises(KeyError):
        ExponentialSmoothing(ses, bounds=bounds)
    bounds = {"smoothing_level": [0.0, 1.0]}
    with pytest.raises(TypeError):
        ExponentialSmoothing(ses, bounds=bounds)
    bounds = {"smoothing_level": (0.0, 1.0, 2.0)}
    with pytest.raises(ValueError):
        ExponentialSmoothing(ses, bounds=bounds)
    bounds = {"smoothing_level": (1.0, 0.0)}
    with pytest.raises(ValueError):
        ExponentialSmoothing(ses, bounds=bounds)
    bounds = {"smoothing_level": (-1.0, 2.0)}
    with pytest.raises(ValueError):
        ExponentialSmoothing(ses, bounds=bounds)


def test_valid_bounds(ses):
    bounds = {"smoothing_level": (0.1, 1.0)}
    res = ExponentialSmoothing(
        ses, bounds=bounds, initialization_method="estimated"
    ).fit(method="least_squares")
    res2 = ExponentialSmoothing(ses, initialization_method="estimated").fit(
        method="least_squares"
    )
    assert_allclose(
        res.params["smoothing_level"],
        res2.params["smoothing_level"],
        rtol=1e-4,
    )


def test_fixed_basic(ses):
    mod = ExponentialSmoothing(ses, initialization_method="estimated")
    with mod.fix_params({"smoothing_level": 0.3}):
        res = mod.fit()
    assert res.params["smoothing_level"] == 0.3
    assert isinstance(res.summary().as_text(), str)

    mod = ExponentialSmoothing(
        ses, trend="add", damped_trend=True, initialization_method="estimated"
    )
    with mod.fix_params({"damping_trend": 0.98}):
        res = mod.fit()
    assert res.params["damping_trend"] == 0.98
    assert isinstance(res.summary().as_text(), str)

    mod = ExponentialSmoothing(
        ses, trend="add", seasonal="add", initialization_method="estimated"
    )
    with mod.fix_params({"smoothing_seasonal": 0.1, "smoothing_level": 0.2}):
        res = mod.fit()
    assert res.params["smoothing_seasonal"] == 0.1
    assert res.params["smoothing_level"] == 0.2
    assert isinstance(res.summary().as_text(), str)


def test_fixed_errors(ses):
    mod = ExponentialSmoothing(ses, initialization_method="estimated")
    with pytest.raises(KeyError):
        with mod.fix_params({"smoothing_trend": 0.3}):
            pass
    with pytest.raises(ValueError):
        with mod.fix_params({"smoothing_level": -0.3}):
            pass
    mod = ExponentialSmoothing(
        ses, trend="add", initialization_method="estimated"
    )
    with pytest.raises(ValueError):
        with mod.fix_params({"smoothing_level": 0.3, "smoothing_trend": 0.4}):
            pass
    mod = ExponentialSmoothing(
        ses, trend="add", seasonal="add", initialization_method="estimated"
    )
    with pytest.raises(ValueError):
        with mod.fix_params(
            {"smoothing_level": 0.3, "smoothing_seasonal": 0.8}
        ):
            pass

    bounds = {"smoothing_level": (0.4, 0.8), "smoothing_seasonal": (0.7, 0.9)}
    mod = ExponentialSmoothing(
        ses,
        trend="add",
        seasonal="add",
        bounds=bounds,
        initialization_method="estimated",
    )
    with pytest.raises(ValueError, match="After adjusting for user-provided"):
        with mod.fix_params(
            {"smoothing_trend": 0.3, "smoothing_seasonal": 0.6}
        ):
            mod.fit()


@pytest.mark.parametrize("trend", ["add", None])
@pytest.mark.parametrize("seasonal", ["add", None])
def test_brute(ses, trend, seasonal):
    mod = ExponentialSmoothing(
        ses, trend=trend, seasonal=seasonal, initialization_method="estimated"
    )
    res = mod.fit(use_brute=True)
    assert res.mle_retvals.success

    with mod.fix_params({"smoothing_level": 0.1}):
        res = mod.fit(use_brute=True)
    assert res.mle_retvals.success
    assert isinstance(res.summary().as_text(), str)


def test_fix_set_parameters(ses):
    with pytest.raises(ValueError):
        ExponentialSmoothing(
            ses, initial_level=1.0, initialization_method="heuristic"
        )
    with pytest.raises(ValueError):
        ExponentialSmoothing(
            ses,
            trend="add",
            initial_trend=1.0,
            initialization_method="legacy-heuristic",
        )
    with pytest.raises(ValueError):
        ExponentialSmoothing(
            ses,
            seasonal="add",
            initial_seasonal=np.ones(12),
            initialization_method="estimated",
        )


def test_fix_unfixable(ses):
    mod = ExponentialSmoothing(ses, initialization_method="estimated")
    with pytest.raises(ValueError, match="Cannot fix a parameter"):
        with mod.fix_params({"smoothing_level": 0.25}):
            mod.fit(smoothing_level=0.2)


def test_infeasible_bounds(ses):
    bounds = {"smoothing_level": (0.1, 0.2), "smoothing_trend": (0.3, 0.4)}
    with pytest.raises(ValueError, match="The bounds for smoothing_trend"):
        ExponentialSmoothing(
            ses, trend="add", bounds=bounds, initialization_method="estimated"
        ).fit()
    bounds = {"smoothing_level": (0.3, 0.5), "smoothing_seasonal": (0.7, 0.8)}
    with pytest.raises(ValueError, match="The bounds for smoothing_seasonal"):
        ExponentialSmoothing(
            ses,
            seasonal="add",
            bounds=bounds,
            initialization_method="estimated",
        ).fit()


@pytest.mark.parametrize(
    "method", ["estimated", "heuristic", "legacy-heuristic"]
)
@pytest.mark.parametrize("trend", [None, "add"])
@pytest.mark.parametrize("seasonal", [None, "add"])
def test_initialization_methods(ses, method, trend, seasonal):
    mod = ExponentialSmoothing(
        ses, trend=trend, seasonal=seasonal, initialization_method=method
    )
    res = mod.fit()
    assert res.mle_retvals.success
    assert isinstance(res.summary().as_text(), str)


def test_attributes(ses):
    res = ExponentialSmoothing(ses, initialization_method="estimated").fit()
    assert res.k > 0
    assert res.resid.shape[0] == ses.shape[0]
    assert_allclose(res.fcastvalues, res.fittedfcast[-1:])


def test_summary_boxcox(ses):
    mod = ExponentialSmoothing(
        ses**2, use_boxcox=True, initialization_method="heuristic"
    )
    with pytest.raises(ValueError, match="use_boxcox was set at model"):
        mod.fit(use_boxcox=True)
    res = mod.fit()
    summ = str(res.summary())
    assert re.findall(r"Box-Cox:[\s]*True", summ)
    assert isinstance(res.summary().as_text(), str)


def test_simulate(ses):
    mod = ExponentialSmoothing(
        np.asarray(ses), initialization_method="heuristic"
    )
    res = mod.fit()
    assert isinstance(res.summary().as_text(), str)
    with pytest.raises(ValueError, match="error must be"):
        res.simulate(10, error="unknown")
    with pytest.raises(ValueError, match="If random"):
        res.simulate(10, error="additive", random_errors=np.empty((20, 20)))
    res.simulate(10, error="additive", anchor=100)
    with pytest.raises(ValueError, match="Cannot anchor"):
        res.simulate(10, error="additive", anchor=2000)
    with pytest.raises(ValueError, match="Argument random_state"):
        res.simulate(
            10, error="additive", anchor=100, random_state="bad_value"
        )
    with pytest.raises(ValueError, match="Argument random_errors"):
        res.simulate(10, error="additive", random_errors="bad_values")


@pytest.mark.parametrize(
    "index_typ", ["date_range", "period", "range", "irregular"]
)
def test_forecast_index_types(ses, index_typ):
    nobs = ses.shape[0]
    kwargs = {}
    warning = None
    fcast_index = None
    if index_typ == "period":
        index = pd.period_range("2000-1-1", periods=nobs + 36, freq="M")
    elif index_typ == "date_range":
        index = pd.date_range("2000-1-1", periods=nobs + 36, freq=MONTH_END)
    elif index_typ == "range":
        index = pd.RangeIndex(nobs + 36)
        kwargs["seasonal_periods"] = 12
    elif index_typ == "irregular":
        rs = np.random.RandomState(0)
        index = pd.Index(np.cumsum(rs.randint(0, 4, size=nobs + 36)))
        warning = ValueWarning
        kwargs["seasonal_periods"] = 12
        fcast_index = pd.RangeIndex(start=1000, stop=1036, step=1)
    if fcast_index is None:
        fcast_index = index[-36:]
    ses = ses.copy()
    ses.index = index[:-36]

    with pytest_warns(warning):
        res = ExponentialSmoothing(
            ses,
            trend="add",
            seasonal="add",
            initialization_method="heuristic",
            **kwargs
        ).fit()
    with pytest_warns(warning):
        fcast = res.forecast(36)
    assert isinstance(fcast, pd.Series)
    pd.testing.assert_index_equal(fcast.index, fcast_index)


def test_boxcox_components(ses):
    mod = ExponentialSmoothing(
        ses + 1 - ses.min(), initialization_method="estimated", use_boxcox=True
    )
    res = mod.fit()
    assert isinstance(res.summary().as_text(), str)
    with pytest.raises(AssertionError):
        # Must be different since level is not transformed
        assert_allclose(res.level, res.fittedvalues)
    assert not hasattr(res, "_untransformed_level")
    assert not hasattr(res, "_untransformed_trend")
    assert not hasattr(res, "_untransformed_seasonal")


@pytest.mark.parametrize("repetitions", [1, 10])
@pytest.mark.parametrize("random_errors", [None, "bootstrap"])
def test_forecast_1_simulation(austourists, random_errors, repetitions):
    # GH 7053
    fit = ExponentialSmoothing(
        austourists,
        seasonal_periods=4,
        trend="add",
        seasonal="add",
        damped_trend=True,
        initialization_method="estimated",
    ).fit()

    sim = fit.simulate(
        1, anchor=0, random_errors=random_errors, repetitions=repetitions
    )
    expected_shape = (1,) if repetitions == 1 else (1, repetitions)
    assert sim.shape == expected_shape
    sim = fit.simulate(
        10, anchor=0, random_errors=random_errors, repetitions=repetitions
    )
    expected_shape = (10,) if repetitions == 1 else (10, repetitions)
    assert sim.shape == expected_shape


@pytest.mark.parametrize("trend", [None, "add"])
@pytest.mark.parametrize("seasonal", [None, "add"])
@pytest.mark.parametrize("nobs", [9, 10])
def test_estimated_initialization_short_data(ses, trend, seasonal, nobs):
    # GH 7319
    res = ExponentialSmoothing(
        ses[:nobs],
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=4,
        initialization_method="estimated",
    ).fit()
    assert res.mle_retvals.success


def test_invalid_index(reset_randomstate):
    y = np.random.standard_normal(12 * 200)
    df_y = pd.DataFrame(data=y)
    # Can't have a freq here
    df_y.index.freq = "d"

    model = ExponentialSmoothing(
        df_y,
        seasonal_periods=12,
        trend="add",
        seasonal="add",
        initialization_method="heuristic",
    )
    fitted = model.fit(optimized=True, use_brute=True)

    fcast = fitted.forecast(steps=157200)
    assert fcast.shape[0] == 157200

    index = pd.date_range("2020-01-01", periods=2 * y.shape[0])
    index = np.random.choice(index, size=df_y.shape[0], replace=False)
    index = sorted(index)
    df_y.index = index
    assert isinstance(df_y.index, pd.DatetimeIndex)
    assert df_y.index.freq is None
    assert df_y.index.inferred_freq is None
    with pytest.warns(ValueWarning, match="A date index has been provided"):
        model = ExponentialSmoothing(
            df_y,
            seasonal_periods=12,
            trend="add",
            seasonal="add",
            initialization_method="heuristic",
        )
    fitted = model.fit(optimized=True, use_brute=True)
    with pytest.warns(ValueWarning, match="No supported"):
        fitted.forecast(steps=157200)


def test_initial_level():
    # GH 8634
    series = [0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0]
    es = ExponentialSmoothing(
        series, initialization_method="known", initial_level=20.0
    )
    es_fit = es.fit()
    es_fit.params
    assert_allclose(es_fit.params["initial_level"], 20.0)


def test_all_initial_values():
    fit1 = ExponentialSmoothing(
        aust,
        seasonal_periods=4,
        trend="add",
        seasonal="mul",
        initialization_method="estimated",
    ).fit()
    lvl = np.round(fit1.params["initial_level"])
    trend = np.round(fit1.params["initial_trend"], 1)
    seas = np.round(fit1.params["initial_seasons"], 1)
    fit2 = ExponentialSmoothing(
        aust,
        seasonal_periods=4,
        trend="add",
        seasonal="mul",
        initialization_method="known",
        initial_level=lvl,
        initial_trend=trend,
        initial_seasonal=seas,
    ).fit()
    assert_allclose(fit2.params["initial_level"], lvl)
    assert_allclose(fit2.params["initial_trend"], trend)
    assert_allclose(fit2.params["initial_seasons"], seas)
