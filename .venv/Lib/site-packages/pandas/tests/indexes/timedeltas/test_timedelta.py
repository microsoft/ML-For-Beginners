from datetime import timedelta

import numpy as np
import pytest

from pandas import (
    Index,
    NaT,
    Series,
    Timedelta,
    timedelta_range,
)
import pandas._testing as tm
from pandas.core.arrays import TimedeltaArray


class TestTimedeltaIndex:
    @pytest.fixture
    def index(self):
        return tm.makeTimedeltaIndex(10)

    def test_misc_coverage(self):
        rng = timedelta_range("1 day", periods=5)
        result = rng.groupby(rng.days)
        assert isinstance(next(iter(result.values()))[0], Timedelta)

    def test_map(self):
        # test_map_dictlike generally tests

        rng = timedelta_range("1 day", periods=10)

        f = lambda x: x.days
        result = rng.map(f)
        exp = Index([f(x) for x in rng], dtype=np.int64)
        tm.assert_index_equal(result, exp)

    def test_pass_TimedeltaIndex_to_index(self):
        rng = timedelta_range("1 days", "10 days")
        idx = Index(rng, dtype=object)

        expected = Index(rng.to_pytimedelta(), dtype=object)

        tm.assert_numpy_array_equal(idx.values, expected.values)

    def test_fields(self):
        rng = timedelta_range("1 days, 10:11:12.100123456", periods=2, freq="s")
        tm.assert_index_equal(rng.days, Index([1, 1], dtype=np.int64))
        tm.assert_index_equal(
            rng.seconds,
            Index([10 * 3600 + 11 * 60 + 12, 10 * 3600 + 11 * 60 + 13], dtype=np.int32),
        )
        tm.assert_index_equal(
            rng.microseconds,
            Index([100 * 1000 + 123, 100 * 1000 + 123], dtype=np.int32),
        )
        tm.assert_index_equal(rng.nanoseconds, Index([456, 456], dtype=np.int32))

        msg = "'TimedeltaIndex' object has no attribute '{}'"
        with pytest.raises(AttributeError, match=msg.format("hours")):
            rng.hours
        with pytest.raises(AttributeError, match=msg.format("minutes")):
            rng.minutes
        with pytest.raises(AttributeError, match=msg.format("milliseconds")):
            rng.milliseconds

        # with nat
        s = Series(rng)
        s[1] = np.nan

        tm.assert_series_equal(s.dt.days, Series([1, np.nan], index=[0, 1]))
        tm.assert_series_equal(
            s.dt.seconds, Series([10 * 3600 + 11 * 60 + 12, np.nan], index=[0, 1])
        )

        # preserve name (GH15589)
        rng.name = "name"
        assert rng.days.name == "name"

    def test_freq_conversion_always_floating(self):
        # pre-2.0 td64 astype converted to float64. now for supported units
        #  (s, ms, us, ns) this converts to the requested dtype.
        # This matches TDA and Series
        tdi = timedelta_range("1 Day", periods=30)

        res = tdi.astype("m8[s]")
        exp_values = np.asarray(tdi).astype("m8[s]")
        exp_tda = TimedeltaArray._simple_new(
            exp_values, dtype=exp_values.dtype, freq=tdi.freq
        )
        expected = Index(exp_tda)
        assert expected.dtype == "m8[s]"
        tm.assert_index_equal(res, expected)

        # check this matches Series and TimedeltaArray
        res = tdi._data.astype("m8[s]")
        tm.assert_equal(res, expected._values)

        res = tdi.to_series().astype("m8[s]")
        tm.assert_equal(res._values, expected._values._with_freq(None))

    def test_freq_conversion(self, index_or_series):
        # doc example

        scalar = Timedelta(days=31)
        td = index_or_series(
            [scalar, scalar, scalar + timedelta(minutes=5, seconds=3), NaT],
            dtype="m8[ns]",
        )

        result = td / np.timedelta64(1, "D")
        expected = index_or_series(
            [31, 31, (31 * 86400 + 5 * 60 + 3) / 86400.0, np.nan]
        )
        tm.assert_equal(result, expected)

        # We don't support "D" reso, so we use the pre-2.0 behavior
        #  casting to float64
        msg = (
            r"Cannot convert from timedelta64\[ns\] to timedelta64\[D\]. "
            "Supported resolutions are 's', 'ms', 'us', 'ns'"
        )
        with pytest.raises(ValueError, match=msg):
            td.astype("timedelta64[D]")

        result = td / np.timedelta64(1, "s")
        expected = index_or_series(
            [31 * 86400, 31 * 86400, 31 * 86400 + 5 * 60 + 3, np.nan]
        )
        tm.assert_equal(result, expected)

        exp_values = np.asarray(td).astype("m8[s]")
        exp_tda = TimedeltaArray._simple_new(exp_values, dtype=exp_values.dtype)
        expected = index_or_series(exp_tda)
        assert expected.dtype == "m8[s]"
        result = td.astype("timedelta64[s]")
        tm.assert_equal(result, expected)

    def test_arithmetic_zero_freq(self):
        # GH#51575 don't get a .freq with freq.n = 0
        tdi = timedelta_range(0, periods=100, freq="ns")
        result = tdi / 2
        assert result.freq is None
        expected = tdi[:50].repeat(2)
        tm.assert_index_equal(result, expected)

        result2 = tdi // 2
        assert result2.freq is None
        expected2 = expected
        tm.assert_index_equal(result2, expected2)

        result3 = tdi * 0
        assert result3.freq is None
        expected3 = tdi[:1].repeat(100)
        tm.assert_index_equal(result3, expected3)
