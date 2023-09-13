from datetime import datetime

from dateutil.tz import tzlocal
import pytest

from pandas.compat import IS64

from pandas import (
    DatetimeIndex,
    Index,
    bdate_range,
    date_range,
)
import pandas._testing as tm

START, END = datetime(2009, 1, 1), datetime(2010, 1, 1)


class TestDatetimeIndexOps:
    @pytest.mark.parametrize(
        "freq,expected",
        [
            ("A", "day"),
            ("Q", "day"),
            ("M", "day"),
            ("D", "day"),
            ("H", "hour"),
            ("T", "minute"),
            ("S", "second"),
            ("L", "millisecond"),
            ("U", "microsecond"),
        ],
    )
    def test_resolution(self, request, tz_naive_fixture, freq, expected):
        tz = tz_naive_fixture
        if freq == "A" and not IS64 and isinstance(tz, tzlocal):
            request.node.add_marker(
                pytest.mark.xfail(reason="OverflowError inside tzlocal past 2038")
            )

        idx = date_range(start="2013-04-01", periods=30, freq=freq, tz=tz)
        assert idx.resolution == expected

    def test_infer_freq(self, freq_sample):
        # GH 11018
        idx = date_range("2011-01-01 09:00:00", freq=freq_sample, periods=10)
        result = DatetimeIndex(idx.asi8, freq="infer")
        tm.assert_index_equal(idx, result)
        assert result.freq == freq_sample


@pytest.mark.parametrize("freq", ["B", "C"])
class TestBusinessDatetimeIndex:
    @pytest.fixture
    def rng(self, freq):
        return bdate_range(START, END, freq=freq)

    def test_comparison(self, rng):
        d = rng[10]

        comp = rng > d
        assert comp[11]
        assert not comp[9]

    def test_copy(self, rng):
        cp = rng.copy()
        repr(cp)
        tm.assert_index_equal(cp, rng)

    def test_identical(self, rng):
        t1 = rng.copy()
        t2 = rng.copy()
        assert t1.identical(t2)

        # name
        t1 = t1.rename("foo")
        assert t1.equals(t2)
        assert not t1.identical(t2)
        t2 = t2.rename("foo")
        assert t1.identical(t2)

        # freq
        t2v = Index(t2.values)
        assert t1.equals(t2v)
        assert not t1.identical(t2v)
