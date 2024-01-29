"""tests for some time series analysis functions

"""
from statsmodels.compat.pandas import (
    PD_LT_2_2_0,
    QUARTER_END,
    YEAR_END,
    assert_frame_equal,
    assert_series_equal,
)

import numpy as np
from numpy.testing import (
    assert_array_almost_equal,
    assert_array_equal,
    assert_equal,
    assert_raises,
)
import pandas as pd
from pandas.tseries.frequencies import to_offset
import pytest

from statsmodels import regression
from statsmodels.datasets import macrodata
from statsmodels.tsa import stattools
from statsmodels.tsa.tests.results import savedrvs
from statsmodels.tsa.tests.results.datamlw_tls import (
    mlacf,
    mlccf,
    mlpacf,
    mlywar,
)
import statsmodels.tsa.tsatools as tools
from statsmodels.tsa.tsatools import vec, vech

xo = savedrvs.rvsdata.xar2
x100 = xo[-100:] / 1000.0
x1000 = xo / 1000.0


def test_acf():
    acf_x = stattools.acf(x100, adjusted=False, fft=False, nlags=20)
    assert_array_almost_equal(mlacf.acf100.ravel(), acf_x, 8)  # why only dec=8
    acf_x = stattools.acf(x1000, adjusted=False, fft=False, nlags=20)
    assert_array_almost_equal(
        mlacf.acf1000.ravel(), acf_x, 8
    )  # why only dec=9


def test_ccf():
    ccf_x = stattools.ccf(x100[4:], x100[:-4], adjusted=False)[:21]
    assert_array_almost_equal(mlccf.ccf100.ravel()[:21][::-1], ccf_x, 8)
    ccf_x = stattools.ccf(x1000[4:], x1000[:-4], adjusted=False)[:21]
    assert_array_almost_equal(mlccf.ccf1000.ravel()[:21][::-1], ccf_x, 8)


def test_pacf_yw():
    pacfyw = stattools.pacf_yw(x100, 20, method="mle")
    assert_array_almost_equal(mlpacf.pacf100.ravel(), pacfyw, 1)
    pacfyw = stattools.pacf_yw(x1000, 20, method="mle")
    assert_array_almost_equal(mlpacf.pacf1000.ravel(), pacfyw, 2)
    # assert False


def test_pacf_ols():
    pacfols = stattools.pacf_ols(x100, 20)
    assert_array_almost_equal(mlpacf.pacf100.ravel(), pacfols, 8)
    pacfols = stattools.pacf_ols(x1000, 20)
    assert_array_almost_equal(mlpacf.pacf1000.ravel(), pacfols, 8)
    # assert False


def test_ywcoef():
    assert_array_almost_equal(
        mlywar.arcoef100[1:],
        -regression.yule_walker(x100, 10, method="mle")[0],
        8,
    )
    assert_array_almost_equal(
        mlywar.arcoef1000[1:],
        -regression.yule_walker(x1000, 20, method="mle")[0],
        8,
    )


@pytest.mark.smoke
def test_yule_walker_inter():
    # see 1869
    x = np.array([1, -1, 2, 2, 0, -2, 1, 0, -3, 0, 0])
    # it works
    regression.yule_walker(x, 3)


def test_duplication_matrix(reset_randomstate):
    for k in range(2, 10):
        m = tools.unvech(np.random.randn(k * (k + 1) // 2))
        Dk = tools.duplication_matrix(k)
        assert np.array_equal(vec(m), np.dot(Dk, vech(m)))


def test_elimination_matrix(reset_randomstate):
    for k in range(2, 10):
        m = np.random.randn(k, k)
        Lk = tools.elimination_matrix(k)
        assert np.array_equal(vech(m), np.dot(Lk, vec(m)))


def test_commutation_matrix(reset_randomstate):
    m = np.random.randn(4, 3)
    K = tools.commutation_matrix(4, 3)
    assert np.array_equal(vec(m.T), np.dot(K, vec(m)))


def test_vec():
    arr = np.array([[1, 2], [3, 4]])
    assert np.array_equal(vec(arr), [1, 3, 2, 4])


def test_vech():
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert np.array_equal(vech(arr), [1, 4, 7, 5, 8, 9])


def test_ar_transparams():
    arr = np.array(
        [-1000.0, -100.0, -10.0, 1.0, 0.0, 1.0, 10.0, 100.0, 1000.0]
    )
    assert not np.isnan(tools._ar_transparams(arr)).any()


class TestLagmat:
    @classmethod
    def setup_class(cls):
        data = macrodata.load_pandas()
        cls.macro_df = data.data[["year", "quarter", "realgdp", "cpi"]]
        cols = list(cls.macro_df.columns)
        cls.realgdp_loc = cols.index("realgdp")
        cls.cpi_loc = cols.index("cpi")
        np.random.seed(12345)
        cls.random_data = np.random.randn(100)

        index = [
            str(int(yr)) + "-Q" + str(int(qu))
            for yr, qu in zip(cls.macro_df.year, cls.macro_df.quarter)
        ]
        cls.macro_df.index = index
        cls.series = cls.macro_df.cpi

    def test_add_lag_insert(self):
        data = self.macro_df.values
        nddata = data.astype(float)
        lagmat = stattools.lagmat(nddata[:, 2], 3, trim="Both")
        results = np.column_stack((nddata[3:, :3], lagmat, nddata[3:, -1]))
        lag_data = tools.add_lag(data, self.realgdp_loc, 3)
        assert_equal(lag_data, results)

    def test_add_lag_noinsert(self):
        data = self.macro_df.values
        nddata = data.astype(float)
        lagmat = stattools.lagmat(nddata[:, 2], 3, trim="Both")
        results = np.column_stack((nddata[3:, :], lagmat))
        lag_data = tools.add_lag(data, self.realgdp_loc, 3, insert=False)
        assert_equal(lag_data, results)

    def test_add_lag_noinsert_atend(self):
        data = self.macro_df.values
        nddata = data.astype(float)
        lagmat = stattools.lagmat(nddata[:, -1], 3, trim="Both")
        results = np.column_stack((nddata[3:, :], lagmat))
        lag_data = tools.add_lag(data, self.cpi_loc, 3, insert=False)
        assert_equal(lag_data, results)
        # should be the same as insert
        lag_data2 = tools.add_lag(data, self.cpi_loc, 3, insert=True)
        assert_equal(lag_data2, results)

    def test_add_lag_ndarray(self):
        data = self.macro_df.values
        nddata = data.astype(float)
        lagmat = stattools.lagmat(nddata[:, 2], 3, trim="Both")
        results = np.column_stack((nddata[3:, :3], lagmat, nddata[3:, -1]))
        lag_data = tools.add_lag(nddata, 2, 3)
        assert_equal(lag_data, results)

    def test_add_lag_noinsert_ndarray(self):
        data = self.macro_df.values
        nddata = data.astype(float)
        lagmat = stattools.lagmat(nddata[:, 2], 3, trim="Both")
        results = np.column_stack((nddata[3:, :], lagmat))
        lag_data = tools.add_lag(nddata, 2, 3, insert=False)
        assert_equal(lag_data, results)

    def test_add_lag_noinsertatend_ndarray(self):
        data = self.macro_df.values
        nddata = data.astype(float)
        lagmat = stattools.lagmat(nddata[:, -1], 3, trim="Both")
        results = np.column_stack((nddata[3:, :], lagmat))
        lag_data = tools.add_lag(nddata, 3, 3, insert=False)
        assert_equal(lag_data, results)
        # should be the same as insert also check negative col number
        lag_data2 = tools.add_lag(nddata, -1, 3, insert=True)
        assert_equal(lag_data2, results)

    def test_sep_return(self):
        data = self.random_data
        n = data.shape[0]
        lagmat, leads = stattools.lagmat(data, 3, trim="none", original="sep")
        expected = np.zeros((n + 3, 4))
        for i in range(4):
            expected[i : i + n, i] = data
        expected_leads = expected[:, :1]
        expected_lags = expected[:, 1:]
        assert_equal(expected_lags, lagmat)
        assert_equal(expected_leads, leads)

    def test_add_lag1d(self):
        data = self.random_data
        lagmat = stattools.lagmat(data, 3, trim="Both")
        results = np.column_stack((data[3:], lagmat))
        lag_data = tools.add_lag(data, lags=3, insert=True)
        assert_equal(results, lag_data)

        # add index
        data = data[:, None]
        lagmat = stattools.lagmat(data, 3, trim="Both")  # test for lagmat too
        results = np.column_stack((data[3:], lagmat))
        lag_data = tools.add_lag(data, lags=3, insert=True)
        assert_equal(results, lag_data)

    def test_add_lag1d_drop(self):
        data = self.random_data
        lagmat = stattools.lagmat(data, 3, trim="Both")
        lag_data = tools.add_lag(data, lags=3, drop=True, insert=True)
        assert_equal(lagmat, lag_data)

        # no insert, should be the same
        lag_data = tools.add_lag(data, lags=3, drop=True, insert=False)
        assert_equal(lagmat, lag_data)

    def test_add_lag1d_struct(self):
        data = np.zeros(100, dtype=[("variable", float)])
        nddata = self.random_data
        data["variable"] = nddata

        lagmat = stattools.lagmat(nddata, 3, trim="Both", original="in")
        lag_data = tools.add_lag(data, 0, lags=3, insert=True)
        assert_equal(lagmat, lag_data)

        lag_data = tools.add_lag(data, 0, lags=3, insert=False)
        assert_equal(lagmat, lag_data)

        lag_data = tools.add_lag(data, lags=3, insert=True)
        assert_equal(lagmat, lag_data)

    def test_add_lag_1d_drop_struct(self):
        data = np.zeros(100, dtype=[("variable", float)])
        nddata = self.random_data
        data["variable"] = nddata

        lagmat = stattools.lagmat(nddata, 3, trim="Both")
        lag_data = tools.add_lag(data, lags=3, drop=True)
        assert_equal(lagmat, lag_data)

    def test_add_lag_drop_insert(self):
        data = self.macro_df.values
        nddata = data.astype(float)
        lagmat = stattools.lagmat(nddata[:, 2], 3, trim="Both")
        results = np.column_stack((nddata[3:, :2], lagmat, nddata[3:, -1]))
        lag_data = tools.add_lag(data, self.realgdp_loc, 3, drop=True)
        assert_equal(lag_data, results)

    def test_add_lag_drop_noinsert(self):
        data = self.macro_df.values
        nddata = data.astype(float)
        lagmat = stattools.lagmat(nddata[:, 2], 3, trim="Both")
        results = np.column_stack((nddata[3:, np.array([0, 1, 3])], lagmat))
        lag_data = tools.add_lag(
            data, self.realgdp_loc, 3, insert=False, drop=True
        )
        assert_equal(lag_data, results)

    def test_dataframe_without_pandas(self):
        data = self.macro_df
        both = stattools.lagmat(data, 3, trim="both", original="in")
        both_np = stattools.lagmat(data.values, 3, trim="both", original="in")
        assert_equal(both, both_np)

        lags = stattools.lagmat(data, 3, trim="none", original="ex")
        lags_np = stattools.lagmat(data.values, 3, trim="none", original="ex")
        assert_equal(lags, lags_np)

        lags, lead = stattools.lagmat(data, 3, trim="forward", original="sep")
        lags_np, lead_np = stattools.lagmat(
            data.values, 3, trim="forward", original="sep"
        )
        assert_equal(lags, lags_np)
        assert_equal(lead, lead_np)

    def test_dataframe_both(self):
        data = self.macro_df
        columns = list(data.columns)
        n = data.shape[0]
        values = np.zeros((n + 3, 16))
        values[:n, :4] = data.values
        for lag in range(1, 4):
            new_cols = [col + ".L." + str(lag) for col in data]
            columns.extend(new_cols)
            values[lag : n + lag, 4 * lag : 4 * (lag + 1)] = data.values
        index = data.index
        values = values[:n]
        expected = pd.DataFrame(values, columns=columns, index=index)
        expected = expected.iloc[3:]

        both = stattools.lagmat(
            self.macro_df, 3, trim="both", original="in", use_pandas=True
        )
        assert_frame_equal(both, expected)
        lags = stattools.lagmat(
            self.macro_df, 3, trim="both", original="ex", use_pandas=True
        )
        assert_frame_equal(lags, expected.iloc[:, 4:])
        lags, lead = stattools.lagmat(
            self.macro_df, 3, trim="both", original="sep", use_pandas=True
        )
        assert_frame_equal(lags, expected.iloc[:, 4:])
        assert_frame_equal(lead, expected.iloc[:, :4])

    def test_too_few_observations(self):
        assert_raises(
            ValueError, stattools.lagmat, self.macro_df, 300, use_pandas=True
        )
        assert_raises(ValueError, stattools.lagmat, self.macro_df.values, 300)

    def test_unknown_trim(self):
        assert_raises(
            ValueError,
            stattools.lagmat,
            self.macro_df,
            3,
            trim="unknown",
            use_pandas=True,
        )
        assert_raises(
            ValueError,
            stattools.lagmat,
            self.macro_df.values,
            3,
            trim="unknown",
        )

    def test_dataframe_forward(self):
        data = self.macro_df
        columns = list(data.columns)
        n = data.shape[0]
        values = np.zeros((n + 3, 16))
        values[:n, :4] = data.values
        for lag in range(1, 4):
            new_cols = [col + ".L." + str(lag) for col in data]
            columns.extend(new_cols)
            values[lag : n + lag, 4 * lag : 4 * (lag + 1)] = data.values
        index = data.index
        values = values[:n]
        expected = pd.DataFrame(values, columns=columns, index=index)
        both = stattools.lagmat(
            self.macro_df, 3, trim="forward", original="in", use_pandas=True
        )
        assert_frame_equal(both, expected)
        lags = stattools.lagmat(
            self.macro_df, 3, trim="forward", original="ex", use_pandas=True
        )
        assert_frame_equal(lags, expected.iloc[:, 4:])
        lags, lead = stattools.lagmat(
            self.macro_df, 3, trim="forward", original="sep", use_pandas=True
        )
        assert_frame_equal(lags, expected.iloc[:, 4:])
        assert_frame_equal(lead, expected.iloc[:, :4])

    def test_pandas_errors(self):
        assert_raises(
            ValueError,
            stattools.lagmat,
            self.macro_df,
            3,
            trim="none",
            use_pandas=True,
        )
        assert_raises(
            ValueError,
            stattools.lagmat,
            self.macro_df,
            3,
            trim="backward",
            use_pandas=True,
        )
        assert_raises(
            ValueError,
            stattools.lagmat,
            self.series,
            3,
            trim="none",
            use_pandas=True,
        )
        assert_raises(
            ValueError,
            stattools.lagmat,
            self.series,
            3,
            trim="backward",
            use_pandas=True,
        )

    def test_series_forward(self):
        expected = pd.DataFrame(
            index=self.series.index,
            columns=["cpi", "cpi.L.1", "cpi.L.2", "cpi.L.3"],
        )
        expected["cpi"] = self.series
        for lag in range(1, 4):
            expected["cpi.L." + str(int(lag))] = self.series.shift(lag)
        expected = expected.fillna(0.0)

        both = stattools.lagmat(
            self.series, 3, trim="forward", original="in", use_pandas=True
        )
        assert_frame_equal(both, expected)
        lags = stattools.lagmat(
            self.series, 3, trim="forward", original="ex", use_pandas=True
        )
        assert_frame_equal(lags, expected.iloc[:, 1:])
        lags, lead = stattools.lagmat(
            self.series, 3, trim="forward", original="sep", use_pandas=True
        )
        assert_frame_equal(lead, expected.iloc[:, :1])
        assert_frame_equal(lags, expected.iloc[:, 1:])

    def test_series_both(self):
        expected = pd.DataFrame(
            index=self.series.index,
            columns=["cpi", "cpi.L.1", "cpi.L.2", "cpi.L.3"],
        )
        expected["cpi"] = self.series
        for lag in range(1, 4):
            expected["cpi.L." + str(int(lag))] = self.series.shift(lag)
        expected = expected.iloc[3:]

        both = stattools.lagmat(
            self.series, 3, trim="both", original="in", use_pandas=True
        )
        assert_frame_equal(both, expected)
        lags = stattools.lagmat(
            self.series, 3, trim="both", original="ex", use_pandas=True
        )
        assert_frame_equal(lags, expected.iloc[:, 1:])
        lags, lead = stattools.lagmat(
            self.series, 3, trim="both", original="sep", use_pandas=True
        )
        assert_frame_equal(lead, expected.iloc[:, :1])
        assert_frame_equal(lags, expected.iloc[:, 1:])

    def test_range_index_columns(self):
        # GH 8377
        df = pd.DataFrame(np.arange(200).reshape((-1, 2)))
        df.columns = pd.RangeIndex(2)
        result = stattools.lagmat(df, maxlag=2, use_pandas=True)
        assert result.shape == (100, 4)
        assert list(result.columns) == ["0.L.1", "1.L.1", "0.L.2", "1.L.2"]

    def test_duplicate_column_names(self):
        # GH 8377
        df = pd.DataFrame(np.arange(200).reshape((-1, 2)))
        df.columns = [0, "0"]
        with pytest.raises(ValueError, match="Columns names must be"):
            stattools.lagmat(df, maxlag=2, use_pandas=True)


ANNUAL = "A" if PD_LT_2_2_0 else YEAR_END
freqs = [
    YEAR_END,
    f"{ANNUAL}-MAR",
    QUARTER_END,
    "QS",
    "QS-APR",
    "W",
    "W-MON",
    "B",
    "D",
    "h",
]
expected = [1, 1, 4, 4, 4, 52, 52, 5, 7, 24]
freq_expected = [(f, e) for f, e in zip(freqs, expected)]


@pytest.mark.parametrize("freq_expected", freq_expected)
def test_freq_to_period(freq_expected):
    freq, expected = freq_expected
    assert_equal(tools.freq_to_period(freq), expected)
    assert_equal(tools.freq_to_period(to_offset(freq)), expected)


class TestDetrend:
    @classmethod
    def setup_class(cls):
        cls.data_1d = np.arange(5.0)
        cls.data_2d = np.arange(10.0).reshape(5, 2)

    def test_detrend_1d(self):
        data = self.data_1d
        assert_array_almost_equal(
            tools.detrend(data, order=1), np.zeros_like(data)
        )
        assert_array_almost_equal(
            tools.detrend(data, order=0), [-2, -1, 0, 1, 2]
        )

    def test_detrend_2d(self):
        data = self.data_2d
        assert_array_almost_equal(
            tools.detrend(data, order=1, axis=0), np.zeros_like(data)
        )
        assert_array_almost_equal(
            tools.detrend(data, order=0, axis=0),
            [[-4, -4], [-2, -2], [0, 0], [2, 2], [4, 4]],
        )
        assert_array_almost_equal(
            tools.detrend(data, order=0, axis=1),
            [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]],
        )

    def test_detrend_series(self):
        data = pd.Series(self.data_1d, name="one")
        detrended = tools.detrend(data, order=1)
        assert_array_almost_equal(detrended.values, np.zeros_like(data))
        assert_series_equal(detrended, pd.Series(detrended.values, name="one"))
        detrended = tools.detrend(data, order=0)
        assert_array_almost_equal(
            detrended.values, pd.Series([-2, -1, 0, 1, 2])
        )
        assert_series_equal(detrended, pd.Series(detrended.values, name="one"))

    def test_detrend_dataframe(self):
        columns = ["one", "two"]
        index = [c for c in "abcde"]
        data = pd.DataFrame(self.data_2d, columns=columns, index=index)

        detrended = tools.detrend(data, order=1, axis=0)
        assert_array_almost_equal(detrended.values, np.zeros_like(data))
        assert_frame_equal(
            detrended,
            pd.DataFrame(detrended.values, columns=columns, index=index),
        )

        detrended = tools.detrend(data, order=0, axis=0)
        assert_array_almost_equal(
            detrended.values, [[-4, -4], [-2, -2], [0, 0], [2, 2], [4, 4]]
        )
        assert_frame_equal(
            detrended,
            pd.DataFrame(detrended.values, columns=columns, index=index),
        )

        detrended = tools.detrend(data, order=0, axis=1)
        assert_array_almost_equal(
            detrended.values,
            [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]],
        )
        assert_frame_equal(
            detrended,
            pd.DataFrame(detrended.values, columns=columns, index=index),
        )

    def test_detrend_dim_too_large(self):
        assert_raises(NotImplementedError, tools.detrend, np.ones((3, 3, 3)))


class TestAddTrend:
    @classmethod
    def setup_class(cls):
        cls.n = 200
        cls.arr_1d = np.arange(float(cls.n))
        cls.arr_2d = np.tile(np.arange(float(cls.n))[:, None], 2)
        cls.c = np.ones(cls.n)
        cls.t = np.arange(1.0, cls.n + 1)

    def test_series(self):
        s = pd.Series(self.arr_1d)
        appended = tools.add_trend(s)
        expected = pd.DataFrame(s)
        expected["const"] = self.c
        assert_frame_equal(expected, appended)

        prepended = tools.add_trend(s, prepend=True)
        expected = pd.DataFrame(s)
        expected.insert(0, "const", self.c)
        assert_frame_equal(expected, prepended)

        s = pd.Series(self.arr_1d)
        appended = tools.add_trend(s, trend="ct")
        expected = pd.DataFrame(s)
        expected["const"] = self.c
        expected["trend"] = self.t
        assert_frame_equal(expected, appended)

    def test_dataframe(self):
        df = pd.DataFrame(self.arr_2d)
        appended = tools.add_trend(df)
        expected = df.copy()
        expected["const"] = self.c
        assert_frame_equal(expected, appended)

        prepended = tools.add_trend(df, prepend=True)
        expected = df.copy()
        expected.insert(0, "const", self.c)
        assert_frame_equal(expected, prepended)

        df = pd.DataFrame(self.arr_2d)
        appended = tools.add_trend(df, trend="t")
        expected = df.copy()
        expected["trend"] = self.t
        assert_frame_equal(expected, appended)

        df = pd.DataFrame(self.arr_2d)
        appended = tools.add_trend(df, trend="ctt")
        expected = df.copy()
        expected["const"] = self.c
        expected["trend"] = self.t
        expected["trend_squared"] = self.t ** 2
        assert_frame_equal(expected, appended)

    def test_duplicate_const(self):
        assert_raises(
            ValueError,
            tools.add_trend,
            x=self.c,
            trend="c",
            has_constant="raise",
        )
        assert_raises(
            ValueError,
            tools.add_trend,
            x=self.c,
            trend="ct",
            has_constant="raise",
        )
        df = pd.DataFrame(self.c)
        assert_raises(
            ValueError, tools.add_trend, x=df, trend="c", has_constant="raise"
        )
        assert_raises(
            ValueError, tools.add_trend, x=df, trend="ct", has_constant="raise"
        )

        skipped = tools.add_trend(self.c, trend="c")
        assert_equal(skipped, self.c[:, None])

        skipped_const = tools.add_trend(
            self.c, trend="ct", has_constant="skip"
        )
        expected = np.vstack((self.c, self.t)).T
        assert_equal(skipped_const, expected)

        added = tools.add_trend(self.c, trend="c", has_constant="add")
        expected = np.vstack((self.c, self.c)).T
        assert_equal(added, expected)

        added = tools.add_trend(self.c, trend="ct", has_constant="add")
        expected = np.vstack((self.c, self.c, self.t)).T
        assert_equal(added, expected)

    def test_dataframe_duplicate(self):
        df = pd.DataFrame(self.arr_2d, columns=["const", "trend"])
        tools.add_trend(df, trend="ct")
        tools.add_trend(df, trend="ct", prepend=True)

    def test_array(self):
        base = np.vstack((self.arr_1d, self.c, self.t, self.t ** 2)).T
        assert_equal(tools.add_trend(self.arr_1d), base[:, :2])
        assert_equal(tools.add_trend(self.arr_1d, trend="t"), base[:, [0, 2]])
        assert_equal(tools.add_trend(self.arr_1d, trend="ct"), base[:, :3])
        assert_equal(tools.add_trend(self.arr_1d, trend="ctt"), base)

        base = np.hstack(
            (
                self.c[:, None],
                self.t[:, None],
                self.t[:, None] ** 2,
                self.arr_2d,
            )
        )
        assert_equal(
            tools.add_trend(self.arr_2d, prepend=True), base[:, [0, 3, 4]]
        )
        assert_equal(
            tools.add_trend(self.arr_2d, trend="t", prepend=True),
            base[:, [1, 3, 4]],
        )
        assert_equal(
            tools.add_trend(self.arr_2d, trend="ct", prepend=True),
            base[:, [0, 1, 3, 4]],
        )
        assert_equal(
            tools.add_trend(self.arr_2d, trend="ctt", prepend=True), base
        )

    def test_unknown_trend(self):
        assert_raises(
            ValueError, tools.add_trend, x=self.arr_1d, trend="unknown"
        )

    def test_trend_n(self):
        assert_equal(tools.add_trend(self.arr_1d, "n"), self.arr_1d)
        assert tools.add_trend(self.arr_1d, "n") is not self.arr_1d
        assert_equal(tools.add_trend(self.arr_2d, "n"), self.arr_2d)
        assert tools.add_trend(self.arr_2d, "n") is not self.arr_2d


class TestLagmat2DS:
    @classmethod
    def setup_class(cls):
        data = macrodata.load_pandas()
        cls.macro_df = data.data[["year", "quarter", "realgdp", "cpi"]]
        np.random.seed(12345)
        cls.random_data = np.random.randn(100)
        index = [
            str(int(yr)) + "-Q" + str(int(qu))
            for yr, qu in zip(cls.macro_df.year, cls.macro_df.quarter)
        ]
        cls.macro_df.index = index
        cls.series = cls.macro_df.cpi

    @staticmethod
    def _prepare_expected(data, lags, trim="front"):
        t, k = data.shape
        expected = np.zeros((t + lags, (lags + 1) * k))
        for col in range(k):
            for i in range(lags + 1):
                if i < lags:
                    expected[i : -lags + i, (lags + 1) * col + i] = data[
                        :, col
                    ]
                else:
                    expected[i:, (lags + 1) * col + i] = data[:, col]
        if trim == "front":
            expected = expected[:-lags]
        return expected

    def test_lagmat2ds_numpy(self):
        data = self.macro_df
        npdata = data.values
        lagmat = stattools.lagmat2ds(npdata, 2)
        expected = self._prepare_expected(npdata, 2)
        assert_array_equal(lagmat, expected)

        lagmat = stattools.lagmat2ds(npdata[:, :2], 3)
        expected = self._prepare_expected(npdata[:, :2], 3)
        assert_array_equal(lagmat, expected)

        npdata = self.series.values
        lagmat = stattools.lagmat2ds(npdata, 5)
        expected = self._prepare_expected(npdata[:, None], 5)
        assert_array_equal(lagmat, expected)

    def test_lagmat2ds_pandas(self):
        data = self.macro_df
        lagmat = stattools.lagmat2ds(data, 2)
        expected = self._prepare_expected(data.values, 2)
        assert_array_equal(lagmat, expected)

        lagmat = stattools.lagmat2ds(data.iloc[:, :2], 3, trim="both")
        expected = self._prepare_expected(data.values[:, :2], 3)
        expected = expected[3:]
        assert_array_equal(lagmat, expected)

        data = self.series
        lagmat = stattools.lagmat2ds(data, 5)
        expected = self._prepare_expected(data.values[:, None], 5)
        assert_array_equal(lagmat, expected)

    def test_lagmat2ds_use_pandas(self):
        data = self.macro_df
        lagmat = stattools.lagmat2ds(data, 2, use_pandas=True)
        expected = self._prepare_expected(data.values, 2)
        cols = []
        for c in data:
            for lags in range(3):
                if lags == 0:
                    cols.append(c)
                else:
                    cols.append(c + ".L." + str(lags))
        expected = pd.DataFrame(expected, index=data.index, columns=cols)
        assert_frame_equal(lagmat, expected)

        lagmat = stattools.lagmat2ds(
            data.iloc[:, :2], 3, use_pandas=True, trim="both"
        )
        expected = self._prepare_expected(data.values[:, :2], 3)
        cols = []
        for c in data.iloc[:, :2]:
            for lags in range(4):
                if lags == 0:
                    cols.append(c)
                else:
                    cols.append(c + ".L." + str(lags))
        expected = pd.DataFrame(expected, index=data.index, columns=cols)
        expected = expected.iloc[3:]
        assert_frame_equal(lagmat, expected)

        data = self.series
        lagmat = stattools.lagmat2ds(data, 5, use_pandas=True)
        expected = self._prepare_expected(data.values[:, None], 5)

        cols = []
        c = data.name
        for lags in range(6):
            if lags == 0:
                cols.append(c)
            else:
                cols.append(c + ".L." + str(lags))

        expected = pd.DataFrame(expected, index=data.index, columns=cols)
        assert_frame_equal(lagmat, expected)

    def test_3d_error(self):
        data = np.array(2)
        with pytest.raises(ValueError):
            stattools.lagmat2ds(data, 5)

        data = np.zeros((100, 2, 2))
        with pytest.raises(ValueError):
            stattools.lagmat2ds(data, 5)


def test_grangercausality():
    data = np.random.rand(100, 2)
    with pytest.warns(FutureWarning, match="verbose"):
        out = stattools.grangercausalitytests(data, maxlag=2, verbose=False)
    result, models = out[1]
    res2down, res2djoint, rconstr = models
    assert res2djoint.centered_tss is not res2djoint.uncentered_tss
