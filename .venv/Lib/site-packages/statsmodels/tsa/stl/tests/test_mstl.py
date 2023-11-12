from pathlib import Path

from numpy.testing import assert_allclose, assert_equal
import pandas as pd
import pytest

from statsmodels.tsa.seasonal import MSTL


@pytest.fixture(scope="function")
def mstl_results():
    cur_dir = Path(__file__).parent.resolve()
    file_path = cur_dir / "results/mstl_test_results.csv"
    return pd.read_csv(file_path)


@pytest.fixture(scope="function")
def data_pd():
    cur_dir = Path(__file__).parent.resolve()
    file_path = cur_dir / "results/mstl_elec_vic.csv"
    return pd.read_csv(file_path, index_col=["ds"], parse_dates=["ds"])


@pytest.fixture(scope="function")
def data(data_pd):
    return data_pd["y"].values


def test_return_pandas_series_when_input_pandas_and_len_periods_one(data_pd):
    mod = MSTL(endog=data_pd, periods=5)
    res = mod.fit()
    assert isinstance(res.trend, pd.Series)
    assert isinstance(res.seasonal, pd.Series)
    assert isinstance(res.resid, pd.Series)
    assert isinstance(res.weights, pd.Series)


def test_seasonal_is_datafame_when_input_pandas_and_multiple_periods(data_pd):
    mod = MSTL(endog=data_pd, periods=(3, 5))
    res = mod.fit()
    assert isinstance(res.seasonal, pd.DataFrame)


@pytest.mark.parametrize(
    "data, periods, windows, expected",
    [
        (data, 3, None, 1),
        (data, (3, 6), None, 2),
        (data, (3, 6, 1e6), None, 2),
    ],
    indirect=["data"],
)
def test_number_of_seasonal_components(data, periods, windows, expected):
    mod = MSTL(endog=data, periods=periods, windows=windows)
    res = mod.fit()
    n_seasonal_components = (
        res.seasonal.shape[1] if res.seasonal.ndim > 1 else res.seasonal.ndim
    )
    assert n_seasonal_components == expected


@pytest.mark.parametrize(
    "periods, windows",
    [((3, 5), 1), (7, (3, 5))],
)
def test_raise_value_error_when_periods_and_windows_diff_lengths(
    periods, windows
):
    with pytest.raises(
        ValueError, match="Periods and windows must have same length"
    ):
        MSTL(endog=[1, 2, 3, 4, 5], periods=periods, windows=windows)


@pytest.mark.parametrize(
    "data, lmbda",
    [(data, 0.1), (data, 1), (data, -3.0), (data, "auto")],
    indirect=["data"],
)
def test_fit_with_box_cox(data, lmbda):
    periods = (5, 6, 7)
    mod = MSTL(endog=data, periods=periods, lmbda=lmbda)
    mod.fit()


def test_auto_fit_with_box_cox(data):
    periods = (5, 6, 7)
    mod = MSTL(endog=data, periods=periods, lmbda="auto")
    mod.fit()
    assert hasattr(mod, "est_lmbda")
    assert isinstance(mod.est_lmbda, float)


def test_stl_kwargs_smoke(data):
    stl_kwargs = {
        "period": 12,
        "seasonal": 15,
        "trend": 17,
        "low_pass": 15,
        "seasonal_deg": 0,
        "trend_deg": 1,
        "low_pass_deg": 1,
        "seasonal_jump": 2,
        "trend_jump": 2,
        "low_pass_jump": 3,
        "robust": False,
        "inner_iter": 3,
        "outer_iter": 3,
    }
    periods = (5, 6, 7)
    mod = MSTL(
        endog=data, periods=periods, lmbda="auto", stl_kwargs=stl_kwargs
    )
    mod.fit()


@pytest.mark.matplotlib
def test_plot(data, data_pd, close_figures):
    mod = MSTL(endog=data, periods=5)
    res = mod.fit()
    res.plot()

    mod = MSTL(endog=data_pd, periods=5)
    res = mod.fit()
    res.plot()


def test_output_similar_to_R_implementation(data_pd, mstl_results):
    mod = MSTL(
        endog=data_pd,
        periods=(24, 24 * 7),
        stl_kwargs={
            "seasonal_deg": 0,
            "seasonal_jump": 1,
            "trend_jump": 1,
            "trend_deg": 1,
            "low_pass_jump": 1,
            "low_pass_deg": 1,
            "inner_iter": 2,
            "outer_iter": 0,
        },
    )
    res = mod.fit()

    expected_observed = mstl_results["Data"]
    expected_trend = mstl_results["Trend"]
    expected_seasonal = mstl_results[["Seasonal24", "Seasonal168"]]
    expected_resid = mstl_results["Remainder"]

    assert_allclose(res.observed, expected_observed)
    assert_allclose(res.trend, expected_trend)
    assert_allclose(res.seasonal, expected_seasonal)
    assert_allclose(res.resid, expected_resid)


@pytest.mark.parametrize(
    "data, periods_ordered, windows_ordered, periods_not_ordered, "
    "windows_not_ordered",
    [
        (data, (12, 24, 24 * 7), (11, 15, 19), (12, 24 * 7, 24), (11, 19, 15)),
        (
            data,
            (12, 24, 24 * 7 * 1e6),
            (11, 15, 19),
            (12, 24 * 7 * 1e6, 24),
            (11, 19, 15),
        ),
        (data, (12, 24, 24 * 7), None, (12, 24 * 7, 24), None),
    ],
    indirect=["data"],
)
def test_output_invariant_to_period_order(
    data,
    periods_ordered,
    windows_ordered,
    periods_not_ordered,
    windows_not_ordered,
):
    mod1 = MSTL(endog=data, periods=periods_ordered, windows=windows_ordered)
    res1 = mod1.fit()
    mod2 = MSTL(
        endog=data, periods=periods_not_ordered, windows=windows_not_ordered
    )
    res2 = mod2.fit()

    assert_equal(res1.observed, res2.observed)
    assert_equal(res1.trend, res2.trend)
    assert_equal(res1.seasonal, res2.seasonal)
    assert_equal(res1.resid, res2.resid)
