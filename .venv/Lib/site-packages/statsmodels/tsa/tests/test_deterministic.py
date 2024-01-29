from statsmodels.compat.pandas import (
    MONTH_END,
    PD_LT_1_0_0,
    QUARTER_END,
    YEAR_END,
    is_int_index,
)
from statsmodels.compat.pytest import pytest_warns

from typing import Hashable, Tuple

import numpy as np
import pandas as pd
import pytest

from statsmodels.tsa.deterministic import (
    CalendarFourier,
    CalendarSeasonality,
    CalendarTimeTrend,
    DeterministicProcess,
    DeterministicTerm,
    Fourier,
    Seasonality,
    TimeTrend,
)


@pytest.fixture(scope="module")
def time_index(request):
    return pd.date_range("2000-01-01", periods=833, freq="B")


@pytest.fixture(
    scope="module", params=["range", "period", "datetime", "fib", "int64"]
)
def index(request):
    param = request.param
    if param in ("period", "datetime"):
        idx = pd.date_range("2000-01-01", periods=137, freq=MONTH_END)
        if param == "period":
            idx = idx.to_period("M")
    elif param == "range":
        idx = pd.RangeIndex(0, 123)
    elif param == "int64":
        idx = pd.Index(np.arange(123))
    elif param == "fib":
        fib = [0, 1]
        for _ in range(113):
            fib.append(fib[-2] + fib[-1])
        idx = pd.Index(fib)
    else:
        raise NotImplementedError()
    return idx


@pytest.fixture(scope="module", params=[None, False, "list"])
def forecast_index(request):
    idx = pd.date_range("2000-01-01", periods=400, freq="B")
    if request.param is None:
        return None
    elif request.param == "list":
        return list(idx)
    return idx


@pytest.mark.smoke
def test_time_trend_smoke(index, forecast_index):
    tt = TimeTrend(True, 2)
    tt.in_sample(index)
    steps = 83 if forecast_index is None else len(forecast_index)
    warn = None
    if (
        is_int_index(index)
        and np.any(np.diff(index) != 1)
        or (
            type(index) is pd.Index
            and max(index) > 2**63
            and forecast_index is None
        )
    ):
        warn = UserWarning
    with pytest_warns(warn):
        tt.out_of_sample(steps, index, forecast_index)
    str(tt)
    hash(tt)
    assert isinstance(tt.order, int)
    assert isinstance(tt._constant, bool)
    assert TimeTrend.from_string("ctt") == tt
    assert TimeTrend.from_string("ct") != tt
    assert TimeTrend.from_string("t") != tt
    assert TimeTrend.from_string("n") != tt
    assert Seasonality(12) != tt
    tt0 = TimeTrend(False, 0)
    tt0.in_sample(index)
    str(tt0)


@pytest.mark.smoke
def test_seasonality_smoke(index, forecast_index):
    s = Seasonality(12)
    s.in_sample(index)
    steps = 83 if forecast_index is None else len(forecast_index)
    warn = None
    if (
        is_int_index(index)
        and np.any(np.diff(index) != 1)
        or (
            type(index) is pd.Index
            and max(index) > 2**63
            and forecast_index is None
        )
    ):
        warn = UserWarning
    with pytest_warns(warn):
        s.out_of_sample(steps, index, forecast_index)
    assert isinstance(s.period, int)
    str(s)
    hash(s)
    if isinstance(index, (pd.DatetimeIndex, pd.PeriodIndex)) and index.freq:
        s = Seasonality.from_index(index)
        s.in_sample(index)
        s.out_of_sample(steps, index, forecast_index)
        Seasonality.from_index(list(index))


@pytest.mark.smoke
def test_fourier_smoke(index, forecast_index):
    f = Fourier(12, 2)
    f.in_sample(index)
    steps = 83 if forecast_index is None else len(forecast_index)
    warn = None
    if (
        is_int_index(index)
        and np.any(np.diff(index) != 1)
        or (
            type(index) is pd.Index
            and max(index) > 2**63
            and forecast_index is None
        )
    ):
        warn = UserWarning
    with pytest_warns(warn):
        f.out_of_sample(steps, index, forecast_index)
    assert isinstance(f.period, float)
    assert isinstance(f.order, int)
    str(f)
    hash(f)
    with pytest.raises(ValueError, match=r"2 \* order must be <= period"):
        Fourier(12, 7)


@pytest.mark.smoke
def test_calendar_time_trend_smoke(time_index, forecast_index):
    ct = CalendarTimeTrend(YEAR_END, order=2)
    ct.in_sample(time_index)
    steps = 83 if forecast_index is None else len(forecast_index)
    ct.out_of_sample(steps, time_index, forecast_index)
    str(ct)
    hash(ct)
    assert isinstance(ct.order, int)
    assert isinstance(ct.constant, bool)
    assert isinstance(ct.freq, str)
    assert ct.base_period is None


@pytest.mark.smoke
def test_calendar_fourier_smoke(time_index, forecast_index):
    cf = CalendarFourier(YEAR_END, 2)
    cf.in_sample(time_index)
    steps = 83 if forecast_index is None else len(forecast_index)
    cf.out_of_sample(steps, time_index, forecast_index)
    assert isinstance(cf.order, int)
    assert isinstance(cf.freq, str)
    str(cf)
    repr(cf)
    hash(cf)


params = CalendarSeasonality._supported
cs_params = [(k, k2) for k, v in params.items() for k2 in v.keys()]


@pytest.mark.parametrize("freq_period", cs_params)
def test_calendar_seasonality(time_index, forecast_index, freq_period):
    freq, period = freq_period
    cs = CalendarSeasonality(period, freq)
    cs.in_sample(time_index)
    steps = 83 if forecast_index is None else len(forecast_index)
    cs.out_of_sample(steps, time_index, forecast_index)
    assert isinstance(cs.period, str)
    assert isinstance(cs.freq, str)
    str(cs)
    repr(cs)
    hash(cs)
    cs2 = CalendarSeasonality(period, freq)
    assert cs == cs2


def test_forbidden_index():
    index = pd.RangeIndex(0, 10)
    ct = CalendarTimeTrend(YEAR_END, order=2)
    with pytest.raises(TypeError, match="CalendarTimeTrend terms can only"):
        ct.in_sample(index)


def test_calendar_time_trend_base(time_index):
    ct = CalendarTimeTrend(MONTH_END, True, order=3, base_period="1960-1-1")
    ct2 = CalendarTimeTrend(MONTH_END, True, order=3)
    assert ct != ct2
    str(ct)
    str(ct2)
    assert ct.base_period is not None
    assert ct2.base_period is None


def test_invalid_freq_period(time_index):
    with pytest.raises(ValueError, match="The combination of freq="):
        CalendarSeasonality("h", YEAR_END)
    cs = CalendarSeasonality("B", "W")
    with pytest.raises(ValueError, match="freq is B but index contains"):
        cs.in_sample(pd.date_range("2000-1-1", periods=10, freq="D"))


def test_check_index_type():
    ct = CalendarTimeTrend(YEAR_END, True, order=3)
    idx = pd.RangeIndex(0, 20)
    with pytest.raises(TypeError, match="CalendarTimeTrend terms can only"):
        ct._check_index_type(idx, pd.DatetimeIndex)
    with pytest.raises(TypeError, match="CalendarTimeTrend terms can only"):
        ct._check_index_type(idx, (pd.DatetimeIndex,))
    with pytest.raises(TypeError, match="CalendarTimeTrend terms can only"):
        ct._check_index_type(idx, (pd.DatetimeIndex, pd.PeriodIndex))
    idx = pd.Index([0, 1, 1, 2, 3, 5, 8, 13])
    with pytest.raises(TypeError, match="CalendarTimeTrend terms can only"):
        types = (pd.DatetimeIndex, pd.PeriodIndex, pd.RangeIndex)
        ct._check_index_type(idx, types)


def test_unknown_freq():
    with pytest.raises(ValueError, match="freq is not understood by pandas"):
        CalendarTimeTrend("unknown", True, order=3)


def test_invalid_formcast_index(index):
    tt = TimeTrend(order=4)
    with pytest.raises(ValueError, match="The number of values in forecast_"):
        tt.out_of_sample(10, index, pd.RangeIndex(11))


def test_seasonal_from_index_err():
    index = pd.Index([0, 1, 1, 2, 3, 5, 8, 12])
    with pytest.raises(TypeError):
        Seasonality.from_index(index)
    index = pd.date_range("2000-1-1", periods=10)[[0, 1, 2, 3, 5, 8]]
    with pytest.raises(ValueError):
        Seasonality.from_index(index)


def test_time_trend(index):
    tt = TimeTrend(constant=True)
    const = tt.in_sample(index)
    assert const.shape == (index.shape[0], 1)
    assert np.all(const == 1)
    pd.testing.assert_index_equal(const.index, index)
    warn = None
    if (is_int_index(index) and np.any(np.diff(index) != 1)) or (
        type(index) is pd.Index and max(index) > 2**63
    ):
        warn = UserWarning
    with pytest_warns(warn):
        const_fcast = tt.out_of_sample(23, index)
    assert np.all(const_fcast == 1)

    tt = TimeTrend(constant=False)
    empty = tt.in_sample(index)
    assert empty.shape == (index.shape[0], 0)

    tt = TimeTrend(constant=False, order=2)
    t2 = tt.in_sample(index)
    assert t2.shape == (index.shape[0], 2)
    assert list(t2.columns) == ["trend", "trend_squared"]

    tt = TimeTrend(constant=True, order=2)
    final = tt.in_sample(index)
    expected = pd.concat([const, t2], axis=1)
    pd.testing.assert_frame_equal(final, expected)

    tt = TimeTrend(constant=True, order=2)
    short = tt.in_sample(index[:-50])
    with pytest_warns(warn):
        remainder = tt.out_of_sample(50, index[:-50])
    direct = tt.out_of_sample(
        steps=50, index=index[:-50], forecast_index=index[-50:]
    )
    combined = pd.concat([short, remainder], axis=0)
    if isinstance(index, (pd.DatetimeIndex, pd.RangeIndex)):
        pd.testing.assert_frame_equal(combined, final)
    combined = pd.concat([short, direct], axis=0)
    pd.testing.assert_frame_equal(combined, final, check_index_type=False)


def test_seasonality(index):
    s = Seasonality(period=12)
    exog = s.in_sample(index)
    assert s.is_dummy
    assert exog.shape == (index.shape[0], 12)
    pd.testing.assert_index_equal(exog.index, index)
    assert np.all(exog.sum(1) == 1.0)
    assert list(exog.columns) == [f"s({i},12)" for i in range(1, 13)]
    expected = np.zeros((index.shape[0], 12))
    for i in range(12):
        expected[i::12, i] = 1.0
    np.testing.assert_equal(expected, np.asarray(exog))

    warn = None
    if (is_int_index(index) and np.any(np.diff(index) != 1)) or (
        type(index) is pd.Index and max(index) > 2**63
    ):
        warn = UserWarning
    with pytest_warns(warn):
        fcast = s.out_of_sample(steps=12, index=index)
    assert fcast.iloc[0, len(index) % 12] == 1.0
    assert np.all(fcast.sum(1) == 1)

    s = Seasonality(period=7, initial_period=3)
    exog = s.in_sample(index)
    assert exog.iloc[0, 2] == 1.0
    assert exog.iloc[0].sum() == 1.0
    assert s.initial_period == 3
    with pytest.raises(ValueError, match="initial_period must be in"):
        Seasonality(period=12, initial_period=-3)
    with pytest.raises(ValueError, match="period must be >= 2"):
        Seasonality(period=1)


def test_seasonality_time_index(time_index):
    tt = Seasonality.from_index(time_index)
    assert tt.period == 5

    fcast = tt.out_of_sample(steps=12, index=time_index)
    new_idx = DeterministicTerm._extend_index(time_index, 12)
    pd.testing.assert_index_equal(fcast.index, new_idx)


def test_fourier(index):
    f = Fourier(period=12, order=3)
    terms = f.in_sample(index)
    assert f.order == 3
    assert terms.shape == (index.shape[0], 2 * f.order)
    loc = np.arange(index.shape[0]) / 12
    for i, col in enumerate(terms):
        j = i // 2 + 1
        fn = np.cos if (i % 2) else np.sin
        expected = fn(2 * np.pi * j * loc)
        np.testing.assert_allclose(terms[col], expected, atol=1e-8)
    cols = []
    for i in range(2 * f.order):
        fn = "cos" if (i % 2) else "sin"
        cols.append(f"{fn}({(i // 2) + 1},12)")
    assert list(terms.columns) == cols


@pytest.mark.skipif(PD_LT_1_0_0, reason="bug in old pandas")
def test_index_like():
    idx = np.empty((100, 2))
    with pytest.raises(TypeError, match="index must be a pandas"):
        DeterministicTerm._index_like(idx)


def test_calendar_fourier(reset_randomstate):
    inc = np.abs(np.random.standard_normal(1000))
    inc = np.cumsum(inc)
    inc = 10 * inc / inc[-1]
    offset = (24 * 3600 * inc).astype(np.int64)
    base = pd.Timestamp("2000-1-1")
    index = [base + pd.Timedelta(val, unit="s") for val in offset]
    index = pd.Index(index)

    cf = CalendarFourier("D", 2)
    assert cf.order == 2
    terms = cf.in_sample(index)
    cols = []
    for i in range(2 * cf.order):
        fn = "cos" if (i % 2) else "sin"
        cols.append(f"{fn}({(i // 2) + 1},freq=D)")
    assert list(terms.columns) == cols

    inc = offset / (24 * 3600)
    loc = 2 * np.pi * (inc - np.floor(inc))
    expected = []
    for i in range(4):
        scale = i // 2 + 1
        fn = np.cos if (i % 2) else np.sin
        expected.append(fn(scale * loc))
    expected = np.column_stack(expected)
    np.testing.assert_allclose(expected, terms.values)


def test_calendar_time_trend(reset_randomstate):
    inc = np.abs(np.random.standard_normal(1000))
    inc = np.cumsum(inc)
    inc = 10 * inc / inc[-1]
    offset = (24 * 3600 * inc).astype(np.int64)
    base = pd.Timestamp("2000-1-1")
    index = [base + pd.Timedelta(val, "s") for val in offset]
    index = pd.Index(index)

    ctt = CalendarTimeTrend("D", True, order=3, base_period=base)
    assert ctt.order == 3
    terms = ctt.in_sample(index)
    cols = ["const", "trend", "trend_squared", "trend_cubed"]
    assert list(terms.columns) == cols

    inc = 1 + offset / (24 * 3600)
    expected = []
    for i in range(4):
        expected.append(inc**i)
    expected = np.column_stack(expected)
    np.testing.assert_allclose(expected, terms.values)

    ctt = CalendarTimeTrend("D", True, order=2, base_period=base)
    ctt2 = CalendarTimeTrend.from_string("D", trend="ctt", base_period=base)
    pd.testing.assert_frame_equal(ctt.in_sample(index), ctt2.in_sample(index))

    ct = CalendarTimeTrend("D", True, order=1, base_period=base)
    ct2 = CalendarTimeTrend.from_string("D", trend="ct", base_period=base)
    pd.testing.assert_frame_equal(ct.in_sample(index), ct2.in_sample(index))

    ctttt = CalendarTimeTrend("D", True, order=4, base_period=base)
    assert ctttt.order == 4
    terms = ctttt.in_sample(index)
    cols = ["const", "trend", "trend_squared", "trend_cubed", "trend**4"]
    assert list(terms.columns) == cols


def test_calendar_seasonal_period_w():
    period = "W"
    index = pd.date_range("2000-01-03", freq="h", periods=600)
    cs = CalendarSeasonality("h", period=period)
    terms = cs.in_sample(index)
    assert np.all(terms.sum(1) == 1.0)
    for i in range(index.shape[0]):
        assert terms.iloc[i, i % 168] == 1.0

    index = pd.date_range("2000-01-03", freq="B", periods=600)
    cs = CalendarSeasonality("B", period=period)
    terms = cs.in_sample(index)
    assert np.all(terms.sum(1) == 1.0)
    for i in range(index.shape[0]):
        assert terms.iloc[i, i % 5] == 1.0

    index = pd.date_range("2000-01-03", freq="D", periods=600)
    cs = CalendarSeasonality("D", period=period)
    terms = cs.in_sample(index)
    assert np.all(terms.sum(1) == 1.0)
    for i in range(index.shape[0]):
        assert terms.iloc[i, i % 7] == 1.0


def test_calendar_seasonal_period_d():
    period = "D"
    index = pd.date_range("2000-01-03", freq="h", periods=600)
    cs = CalendarSeasonality("h", period=period)
    terms = cs.in_sample(index)
    assert np.all(terms.sum(1) == 1.0)
    for i in range(index.shape[0]):
        assert terms.iloc[i, i % 24] == 1.0


def test_calendar_seasonal_period_q():
    period = "Q"
    index = pd.date_range("2000-01-01", freq=MONTH_END, periods=600)
    cs = CalendarSeasonality(MONTH_END, period=period)
    terms = cs.in_sample(index)
    assert np.all(terms.sum(1) == 1.0)
    for i in range(index.shape[0]):
        assert terms.iloc[i, i % 3] == 1.0


def test_calendar_seasonal_period_a():
    period = "Y"
    index = pd.date_range("2000-01-01", freq=MONTH_END, periods=600)
    cs = CalendarSeasonality(MONTH_END, period=period)
    terms = cs.in_sample(index)
    assert np.all(terms.sum(1) == 1.0)
    for i in range(index.shape[0]):
        assert terms.iloc[i, i % 12] == 1.0

    cs = CalendarSeasonality(QUARTER_END, period=period)
    terms = cs.in_sample(index)
    assert np.all(terms.sum(1) == 1.0)
    for i in range(index.shape[0]):
        assert terms.iloc[i, (i % 12) // 3] == 1.0


@pytest.mark.parametrize("constant", [True, False])
@pytest.mark.parametrize("order", [0, 1])
@pytest.mark.parametrize("seasonal", [True, False])
@pytest.mark.parametrize("fourier", [0, 1])
@pytest.mark.parametrize("period", [None, 10])
@pytest.mark.parametrize("drop", [True, False])
def test_deterministic_process(
    time_index, constant, order, seasonal, fourier, period, drop
):
    if seasonal and fourier:
        return
    dp = DeterministicProcess(
        time_index,
        constant=constant,
        order=order,
        seasonal=seasonal,
        fourier=fourier,
        period=period,
        drop=drop,
    )
    terms = dp.in_sample()
    pd.testing.assert_index_equal(terms.index, time_index)
    terms = dp.out_of_sample(23)
    assert isinstance(terms, pd.DataFrame)


def test_deterministic_process_errors(time_index):
    with pytest.raises(ValueError, match="seasonal and fourier"):
        DeterministicProcess(time_index, seasonal=True, fourier=2, period=5)
    with pytest.raises(TypeError, match="All additional terms"):
        DeterministicProcess(time_index, seasonal=True, additional_terms=[1])


def test_range_error():
    idx = pd.Index([0, 1, 1, 2, 3, 5, 8, 13])
    dp = DeterministicProcess(
        idx, constant=True, order=2, seasonal=True, period=2
    )
    with pytest.raises(TypeError, match="The index in the deterministic"):
        dp.range(0, 12)


def test_range_index_basic():
    idx = pd.date_range("2000-1-1", freq=MONTH_END, periods=120)
    dp = DeterministicProcess(idx, constant=True, order=1, seasonal=True)
    dp.range("2001-1-1", "2008-1-1")
    dp.range("2001-1-1", "2015-1-1")
    dp.range("2013-1-1", "2008-1-1")
    dp.range(0, 100)
    dp.range(100, 150)
    dp.range(130, 150)
    with pytest.raises(ValueError):
        dp.range("1990-1-1", "2010-1-1")

    idx = pd.period_range("2000-1-1", freq="M", periods=120)
    dp = DeterministicProcess(idx, constant=True, order=1, seasonal=True)
    dp.range("2001-1-1", "2008-1-1")
    dp.range("2001-1-1", "2015-1-1")
    dp.range("2013-1-1", "2008-1-1")
    with pytest.raises(ValueError, match="start must be non-negative"):
        dp.range(-7, 200)

    dp.range(0, 100)
    dp.range(100, 150)
    dp.range(130, 150)

    idx = pd.RangeIndex(0, 120)
    dp = DeterministicProcess(
        idx, constant=True, order=1, seasonal=True, period=12
    )
    dp.range(0, 100)
    dp.range(100, 150)
    dp.range(120, 150)
    dp.range(130, 150)
    with pytest.raises(ValueError):
        dp.range(-10, 0)


def test_range_casting():
    idx = np.arange(120).astype(np.int64)
    dp = DeterministicProcess(
        idx, constant=True, order=1, seasonal=True, period=12
    )
    idx = pd.RangeIndex(0, 120)
    dp2 = DeterministicProcess(
        idx, constant=True, order=1, seasonal=True, period=12
    )
    pd.testing.assert_frame_equal(dp.in_sample(), dp2.in_sample())
    pd.testing.assert_frame_equal(dp.range(100, 150), dp2.range(100, 150))


def test_non_unit_range():
    idx = pd.RangeIndex(0, 700, 7)
    dp = DeterministicProcess(idx, constant=True)
    with pytest.raises(ValueError, match="The step of the index is not 1"):
        dp.range(11, 900)


def test_additional_terms(time_index):
    add_terms = [TimeTrend(True, order=1)]
    dp = DeterministicProcess(time_index, additional_terms=add_terms)
    dp2 = DeterministicProcess(time_index, constant=True, order=1)
    pd.testing.assert_frame_equal(dp.in_sample(), dp2.in_sample())
    with pytest.raises(
        ValueError, match="One or more terms in additional_terms"
    ):
        DeterministicProcess(
            time_index, additional_terms=add_terms + add_terms
        )
    with pytest.raises(
        ValueError, match="One or more terms in additional_terms"
    ):
        DeterministicProcess(
            time_index, constant=True, order=1, additional_terms=add_terms
        )


def test_drop_two_consants(time_index):
    tt = TimeTrend(constant=True, order=1)
    dp = DeterministicProcess(
        time_index, constant=True, additional_terms=[tt], drop=True
    )
    assert dp.in_sample().shape[1] == 2
    dp2 = DeterministicProcess(time_index, additional_terms=[tt], drop=True)
    pd.testing.assert_frame_equal(dp.in_sample(), dp2.in_sample())


@pytest.mark.parametrize(
    "index",
    [
        pd.RangeIndex(0, 200),
        pd.Index(np.arange(200)),
        pd.date_range("2000-1-1", freq="MS", periods=200),
        pd.period_range("2000-1-1", freq="M", periods=200),
    ],
)
def test_determintic_term_equiv(index):
    base = DeterministicProcess(pd.RangeIndex(0, 200), constant=True, order=2)
    dp = DeterministicProcess(index, constant=True, order=2)
    np.testing.assert_array_equal(base.in_sample(), dp.in_sample())
    np.testing.assert_array_equal(base.out_of_sample(37), dp.out_of_sample(37))
    np.testing.assert_array_equal(base.range(200, 237), dp.range(200, 237))
    np.testing.assert_array_equal(base.range(50, 150), dp.range(50, 150))
    np.testing.assert_array_equal(base.range(50, 250), dp.range(50, 250))


class DummyTerm(DeterministicTerm):
    @property
    def _eq_attr(self) -> Tuple[Hashable, ...]:
        return ("Dummy",)

    def __str__(self) -> str:
        return "Dummy"

    columns = [
        "const1",
        "const2",
        "trend1",
        "trend2",
        "normal1",
        "normal2",
        "dummy1_1",
        "dummy1_2",
        "always_drop1",
        "always_drop2",
        "dummy2_1",
        "dummy2_2",
    ]

    def in_sample(self, index: pd.Index) -> pd.DataFrame:
        nobs = index.shape[0]
        terms = np.empty((index.shape[0], 12))
        for i in range(0, 12, 2):
            if i == 0:
                value = 1
            elif i == 2:
                value = np.arange(nobs)
            elif i == 4:
                value = np.random.standard_normal(nobs)
            elif i == 6:
                value = np.zeros(nobs)
                value[::2] = 1
            elif i == 8:
                value = 0
            else:  # elif i == 8:
                value = np.zeros(nobs)
                value[1::2] = 1
            terms[:, i] = terms[:, i + 1] = value
        return pd.DataFrame(terms, columns=self.columns, index=index)

    def out_of_sample(
        self,
        steps: int,
        index: pd.Index,
        forecast_index: pd.Index = None,
    ) -> pd.DataFrame:
        fcast_index = self._extend_index(index, steps, forecast_index)
        terms = np.random.standard_normal((steps, 12))

        return pd.DataFrame(terms, columns=self.columns, index=fcast_index)


def test_drop():
    index = pd.RangeIndex(0, 200)
    dummy = DummyTerm()
    str(dummy)
    assert dummy != TimeTrend()
    dp = DeterministicProcess(index, additional_terms=[dummy], drop=True)
    in_samp = dp.in_sample()
    assert in_samp.shape == (200, 4)
    oos = dp.out_of_sample(37)
    assert oos.shape == (37, 4)
    assert list(oos.columns) == list(in_samp.columns)
    valid = ("const", "trend", "dummy", "normal")
    for valid_col in valid:
        assert sum([1 for col in oos if valid_col in col]) == 1
