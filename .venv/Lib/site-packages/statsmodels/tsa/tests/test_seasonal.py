import numpy as np
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_equal,
    assert_raises,
)
import pandas as pd
import pytest

from statsmodels.tsa.seasonal import seasonal_decompose

# Verification values for tests
SEASONAL = [62.46, 86.17, -88.38, -60.25, 62.46, 86.17, -88.38,
            -60.25, 62.46, 86.17, -88.38, -60.25, 62.46, 86.17,
            -88.38, -60.25, 62.46, 86.17, -88.38, -60.25,
            62.46, 86.17, -88.38, -60.25, 62.46, 86.17, -88.38,
            -60.25, 62.46, 86.17, -88.38, -60.25]
TREND = [np.nan, np.nan, 159.12, 204.00, 221.25, 245.12, 319.75,
         451.50, 561.12, 619.25, 615.62, 548.00, 462.12, 381.12,
         316.62, 264.00, 228.38, 210.75, 188.38, 199.00, 207.12,
         191.00, 166.88, 72.00, -9.25, -33.12, -36.75, 36.25,
         103.00, 131.62, np.nan, np.nan]
RANDOM = [np.nan, np.nan, 78.254, 70.254, -36.710, -94.299, -6.371,
          -62.246, 105.415, 103.576, 2.754, 1.254, 15.415, -10.299,
          -33.246, -27.746, 46.165, -57.924, 28.004, -36.746,
          -37.585, 151.826, -75.496, 86.254, -10.210, -194.049,
          48.129, 11.004, -40.460, 143.201, np.nan, np.nan]

MULT_SEASONAL = [1.0815, 1.5538, 0.6716, 0.6931, 1.0815, 1.5538, 0.6716,
                 0.6931, 1.0815, 1.5538, 0.6716, 0.6931, 1.0815, 1.5538,
                 0.6716, 0.6931, 1.0815, 1.5538, 0.6716, 0.6931, 1.0815,
                 1.5538, 0.6716, 0.6931, 1.0815, 1.5538, 0.6716, 0.6931,
                 1.0815, 1.5538, 0.6716, 0.6931]
MULT_TREND = [np.nan, np.nan, 171.62, 204.00, 221.25, 245.12, 319.75,
              451.50, 561.12, 619.25, 615.62, 548.00, 462.12, 381.12,
              316.62, 264.00, 228.38, 210.75, 188.38, 199.00, 207.12,
              191.00, 166.88, 107.25, 80.50, 79.12, 78.75, 116.50,
              140.00, 157.38, np.nan, np.nan]
MULT_RANDOM = [np.nan, np.nan, 1.29263, 1.51360, 1.03223, 0.62226,
               1.04771, 1.05139, 1.20124, 0.84080, 1.28182, 1.28752,
               1.08043, 0.77172, 0.91697, 0.96191, 1.36441, 0.72986,
               1.01171, 0.73956, 1.03566, 1.44556, 0.02677, 1.31843,
               0.49390, 1.14688, 1.45582, 0.16101, 0.82555, 1.47633,
               np.nan, np.nan]


class TestDecompose:
    @classmethod
    def setup_class(cls):
        # even
        data = [-50, 175, 149, 214, 247, 237, 225, 329, 729, 809,
                530, 489, 540, 457, 195, 176, 337, 239, 128, 102,
                232, 429, 3, 98, 43, -141, -77, -13, 125, 361, -45, 184]
        cls.data = pd.DataFrame(data, pd.date_range(start='1/1/1951',
                                                    periods=len(data),
                                                    freq='Q'))

    def test_ndarray(self):
        res_add = seasonal_decompose(self.data.values, period=4)
        assert_almost_equal(res_add.seasonal, SEASONAL, 2)
        assert_almost_equal(res_add.trend, TREND, 2)
        assert_almost_equal(res_add.resid, RANDOM, 3)

        res_mult = seasonal_decompose(np.abs(self.data.values), 'm', period=4)

        assert_almost_equal(res_mult.seasonal, MULT_SEASONAL, 4)
        assert_almost_equal(res_mult.trend, MULT_TREND, 2)
        assert_almost_equal(res_mult.resid, MULT_RANDOM, 4)

        # test odd
        res_add = seasonal_decompose(self.data.values[:-1], period=4)
        seasonal = [68.18, 69.02, -82.66, -54.54, 68.18, 69.02, -82.66,
                    -54.54, 68.18, 69.02, -82.66, -54.54, 68.18, 69.02,
                    -82.66, -54.54, 68.18, 69.02, -82.66, -54.54, 68.18,
                    69.02, -82.66, -54.54, 68.18, 69.02, -82.66, -54.54,
                    68.18, 69.02, -82.66]
        trend = [np.nan, np.nan, 159.12, 204.00, 221.25, 245.12, 319.75,
                 451.50, 561.12, 619.25, 615.62, 548.00, 462.12, 381.12,
                 316.62, 264.00, 228.38, 210.75, 188.38, 199.00, 207.12,
                 191.00, 166.88, 72.00, -9.25, -33.12, -36.75, 36.25,
                 103.00, np.nan, np.nan]
        random = [np.nan, np.nan, 72.538, 64.538, -42.426, -77.150,
                  -12.087, -67.962, 99.699, 120.725, -2.962, -4.462,
                  9.699, 6.850, -38.962, -33.462, 40.449, -40.775, 22.288,
                  -42.462, -43.301, 168.975, -81.212, 80.538, -15.926,
                  -176.900, 42.413, 5.288, -46.176, np.nan, np.nan]
        assert_almost_equal(res_add.seasonal, seasonal, 2)
        assert_almost_equal(res_add.trend, trend, 2)
        assert_almost_equal(res_add.resid, random, 3)

    def test_pandas(self):
        res_add = seasonal_decompose(self.data, period=4)
        freq_override_data = self.data.copy()
        freq_override_data.index = pd.date_range(
            start='1/1/1951', periods=len(freq_override_data), freq='A')
        res_add_override = seasonal_decompose(freq_override_data, period=4)

        assert_almost_equal(res_add.seasonal.values.squeeze(), SEASONAL, 2)
        assert_almost_equal(res_add.trend.values.squeeze(), TREND, 2)
        assert_almost_equal(res_add.resid.values.squeeze(), RANDOM, 3)
        assert_almost_equal(res_add_override.seasonal.values.squeeze(),
                            SEASONAL, 2)
        assert_almost_equal(res_add_override.trend.values.squeeze(),
                            TREND, 2)
        assert_almost_equal(res_add_override.resid.values.squeeze(),
                            RANDOM, 3)
        assert_equal(res_add.seasonal.index.values.squeeze(),
                     self.data.index.values)

        res_mult = seasonal_decompose(np.abs(self.data), 'm', period=4)
        res_mult_override = seasonal_decompose(np.abs(freq_override_data), 'm',
                                               period=4)
        assert_almost_equal(res_mult.seasonal.values.squeeze(), MULT_SEASONAL,
                            4)
        assert_almost_equal(res_mult.trend.values.squeeze(), MULT_TREND, 2)
        assert_almost_equal(res_mult.resid.values.squeeze(), MULT_RANDOM, 4)
        assert_almost_equal(res_mult_override.seasonal.values.squeeze(),
                            MULT_SEASONAL, 4)
        assert_almost_equal(res_mult_override.trend.values.squeeze(),
                            MULT_TREND, 2)
        assert_almost_equal(res_mult_override.resid.values.squeeze(),
                            MULT_RANDOM, 4)
        assert_equal(res_mult.seasonal.index.values.squeeze(),
                     self.data.index.values)

    def test_pandas_nofreq(self, reset_randomstate):
        # issue #3503
        nobs = 100
        dta = pd.Series([x % 3 for x in range(nobs)] + np.random.randn(nobs))
        res_np = seasonal_decompose(dta.values, period=3)
        res = seasonal_decompose(dta, period=3)

        atol = 1e-8
        rtol = 1e-10
        assert_allclose(res.seasonal.values.squeeze(), res_np.seasonal,
                        atol=atol, rtol=rtol)
        assert_allclose(res.trend.values.squeeze(), res_np.trend,
                        atol=atol, rtol=rtol)
        assert_allclose(res.resid.values.squeeze(), res_np.resid,
                        atol=atol, rtol=rtol)

    def test_filt(self):
        filt = np.array([1 / 8., 1 / 4., 1. / 4, 1 / 4., 1 / 8.])
        res_add = seasonal_decompose(self.data.values, filt=filt, period=4)
        assert_almost_equal(res_add.seasonal, SEASONAL, 2)
        assert_almost_equal(res_add.trend, TREND, 2)
        assert_almost_equal(res_add.resid, RANDOM, 3)

    def test_one_sided_moving_average_in_stl_decompose(self):
        res_add = seasonal_decompose(self.data.values, period=4,
                                     two_sided=False)

        seasonal = np.array([76.76, 90.03, -114.4, -52.4, 76.76, 90.03, -114.4,
                             -52.4, 76.76, 90.03, -114.4, -52.4, 76.76, 90.03,
                             -114.4, -52.4, 76.76, 90.03, -114.4, -52.4, 76.76,
                             90.03, -114.4, -52.4, 76.76, 90.03, -114.4, -52.4,
                             76.76, 90.03, -114.4, -52.4])

        trend = np.array([np.nan, np.nan, np.nan, np.nan, 159.12, 204., 221.25,
                          245.12, 319.75, 451.5, 561.12, 619.25, 615.62, 548.,
                          462.12, 381.12, 316.62, 264., 228.38, 210.75, 188.38,
                          199., 207.12, 191., 166.88, 72., -9.25, -33.12,
                          -36.75, 36.25, 103., 131.62])

        resid = np.array([np.nan, np.nan, np.nan, np.nan, 11.112, -57.031,
                          118.147, 136.272, 332.487, 267.469, 83.272, -77.853,
                          -152.388, -181.031, -152.728, -152.728, -56.388,
                          -115.031, 14.022, -56.353, -33.138, 139.969, -89.728,
                          -40.603, -200.638, -303.031, 46.647, 72.522, 84.987,
                          234.719, -33.603, 104.772])

        assert_almost_equal(res_add.seasonal, seasonal, 2)
        assert_almost_equal(res_add.trend, trend, 2)
        assert_almost_equal(res_add.resid, resid, 3)

        res_mult = seasonal_decompose(np.abs(self.data.values), 'm', period=4,
                                      two_sided=False)

        seasonal = np.array([1.1985, 1.5449, 0.5811, 0.6755, 1.1985, 1.5449,
                             0.5811, 0.6755, 1.1985, 1.5449, 0.5811, 0.6755,
                             1.1985, 1.5449, 0.5811, 0.6755, 1.1985, 1.5449,
                             0.5811, 0.6755, 1.1985, 1.5449, 0.5811, 0.6755,
                             1.1985, 1.5449, 0.5811, 0.6755, 1.1985, 1.5449,
                             0.5811, 0.6755])

        trend = np.array([np.nan, np.nan, np.nan, np.nan, 171.625, 204.,
                          221.25, 245.125, 319.75, 451.5, 561.125, 619.25,
                          615.625, 548., 462.125, 381.125, 316.625, 264.,
                          228.375, 210.75, 188.375, 199., 207.125, 191.,
                          166.875, 107.25, 80.5, 79.125, 78.75, 116.5,
                          140., 157.375])

        resid = np.array([np.nan, np.nan, np.nan, np.nan, 1.2008, 0.752, 1.75,
                          1.987, 1.9023, 1.1598, 1.6253, 1.169, 0.7319, 0.5398,
                          0.7261, 0.6837, 0.888, 0.586, 0.9645, 0.7165, 1.0276,
                          1.3954, 0.0249, 0.7596, 0.215, 0.851, 1.646, 0.2432,
                          1.3244, 2.0058, 0.5531, 1.7309])

        assert_almost_equal(res_mult.seasonal, seasonal, 4)
        assert_almost_equal(res_mult.trend, trend, 2)
        assert_almost_equal(res_mult.resid, resid, 4)

        # test odd
        res_add = seasonal_decompose(self.data.values[:-1], period=4,
                                     two_sided=False)
        seasonal = np.array([81.21, 94.48, -109.95, -65.74, 81.21, 94.48,
                             -109.95, -65.74, 81.21, 94.48, -109.95, -65.74,
                             81.21, 94.48, -109.95, -65.74, 81.21, 94.48,
                             -109.95, -65.74, 81.21, 94.48, -109.95, -65.74,
                             81.21, 94.48, -109.95, -65.74, 81.21, 94.48,
                             -109.95])

        trend = [np.nan, np.nan, np.nan, np.nan, 159.12, 204., 221.25,
                 245.12, 319.75, 451.5, 561.12, 619.25, 615.62, 548.,
                 462.12, 381.12, 316.62, 264., 228.38, 210.75, 188.38,
                 199., 207.12, 191., 166.88, 72., -9.25, -33.12,
                 -36.75, 36.25, 103.]

        random = [np.nan, np.nan, np.nan, np.nan, 6.663, -61.48,
                  113.699, 149.618, 328.038, 263.02, 78.824, -64.507,
                  -156.837, -185.48, -157.176, -139.382, -60.837, -119.48,
                  9.574, -43.007, -37.587, 135.52, -94.176, -27.257,
                  -205.087, -307.48, 42.199, 85.868, 80.538, 230.27, -38.051]

        assert_almost_equal(res_add.seasonal, seasonal, 2)
        assert_almost_equal(res_add.trend, trend, 2)
        assert_almost_equal(res_add.resid, random, 3)

    def test_2d(self):
        x = np.tile(np.arange(6), (2, 1)).T
        trend = seasonal_decompose(x, period=2).trend
        expected = np.tile(np.arange(6, dtype=float), (2, 1)).T
        expected[0] = expected[-1] = np.nan
        assert_equal(trend, expected)

    def test_interpolate_trend(self):
        x = np.arange(12)
        freq = 4
        trend = seasonal_decompose(x, period=freq).trend
        assert_equal(trend[0], np.nan)

        trend = seasonal_decompose(x, period=freq, extrapolate_trend=5).trend
        assert_almost_equal(trend, x)

        trend = seasonal_decompose(x, period=freq,
                                   extrapolate_trend='freq').trend
        assert_almost_equal(trend, x)

        trend = seasonal_decompose(x[:, None], period=freq,
                                   extrapolate_trend=5).trend
        assert_almost_equal(trend, x)

        # 2d case
        x = np.tile(np.arange(12), (2, 1)).T
        trend = seasonal_decompose(x, period=freq, extrapolate_trend=1).trend
        assert_almost_equal(trend, x)

        trend = seasonal_decompose(x, period=freq,
                                   extrapolate_trend='freq').trend
        assert_almost_equal(trend, x)

    def test_raises(self):
        assert_raises(ValueError, seasonal_decompose, self.data.values)
        assert_raises(ValueError, seasonal_decompose, self.data, 'm',
                      period=4)
        x = self.data.astype(float).copy()
        x.iloc[2] = np.nan
        assert_raises(ValueError, seasonal_decompose, x)


def test_seasonal_decompose_too_short(reset_randomstate):
    dates = pd.date_range('2000-01-31', periods=4, freq='Q')
    y = np.sin(np.arange(4) / 4 * 2 * np.pi)
    y += np.random.standard_normal(y.size)
    y = pd.Series(y, name='y', index=dates)
    with pytest.raises(ValueError):
        seasonal_decompose(y)

    dates = pd.date_range('2000-01-31', periods=12, freq='M')
    y = np.sin(np.arange(12) / 12 * 2 * np.pi)
    y += np.random.standard_normal(y.size)
    y = pd.Series(y, name='y', index=dates)
    with pytest.raises(ValueError):
        seasonal_decompose(y)
    with pytest.raises(ValueError):
        seasonal_decompose(y.values, period=12)


@pytest.mark.smoke
def test_seasonal_decompose_smoke():
    x = np.array([-50, 175, 149, 214, 247, 237, 225, 329, 729, 809,
                  530, 489, 540, 457, 195, 176, 337, 239, 128, 102,
                  232, 429, 3, 98, 43, -141, -77, -13, 125, 361, -45, 184])
    seasonal_decompose(x, period=4)

    data = pd.DataFrame(x, pd.date_range(start='1/1/1951',
                                         periods=len(x),
                                         freq='Q'))

    seasonal_decompose(data)


def test_seasonal_decompose_multiple():
    x = np.array([-50, 175, 149, 214, 247, 237, 225, 329, 729, 809,
                  530, 489, 540, 457, 195, 176, 337, 239, 128, 102,
                  232, 429, 3, 98, 43, -141, -77, -13, 125, 361, -45, 184])
    x = np.c_[x, x]
    res = seasonal_decompose(x, period=4)
    assert_allclose(res.trend[:, 0], res.trend[:, 1])
    assert_allclose(res.seasonal[:, 0], res.seasonal[:, 1])
    assert_allclose(res.resid[:, 0], res.resid[:, 1])


@pytest.mark.matplotlib
@pytest.mark.parametrize('model', ['additive', 'multiplicative'])
@pytest.mark.parametrize('freq', [4, 12])
@pytest.mark.parametrize('two_sided', [True, False])
@pytest.mark.parametrize('extrapolate_trend', [True, False])
def test_seasonal_decompose_plot(model, freq, two_sided, extrapolate_trend):
    x = np.array([-50, 175, 149, 214, 247, 237, 225, 329, 729, 809,
                  530, 489, 540, 457, 195, 176, 337, 239, 128, 102,
                  232, 429, 3, 98, 43, -141, -77, -13, 125, 361, -45, 184])
    x -= x.min() + 1
    x2 = np.r_[x[12:], x[:12]]
    x = np.c_[x, x2]
    res = seasonal_decompose(x, period=freq, two_sided=two_sided,
                             extrapolate_trend=extrapolate_trend)
    res.plot()
