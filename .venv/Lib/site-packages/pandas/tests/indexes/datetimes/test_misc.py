import calendar
from datetime import datetime
import locale
import unicodedata

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DatetimeIndex,
    Index,
    Timedelta,
    Timestamp,
    date_range,
    offsets,
)
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray

from pandas.tseries.frequencies import to_offset


class TestDatetime64:
    def test_no_millisecond_field(self):
        msg = "type object 'DatetimeIndex' has no attribute 'millisecond'"
        with pytest.raises(AttributeError, match=msg):
            DatetimeIndex.millisecond

        msg = "'DatetimeIndex' object has no attribute 'millisecond'"
        with pytest.raises(AttributeError, match=msg):
            DatetimeIndex([]).millisecond

    def test_datetimeindex_accessors(self):
        dti_naive = date_range(freq="D", start=datetime(1998, 1, 1), periods=365)
        # GH#13303
        dti_tz = date_range(
            freq="D", start=datetime(1998, 1, 1), periods=365, tz="US/Eastern"
        )
        for dti in [dti_naive, dti_tz]:
            assert dti.year[0] == 1998
            assert dti.month[0] == 1
            assert dti.day[0] == 1
            assert dti.hour[0] == 0
            assert dti.minute[0] == 0
            assert dti.second[0] == 0
            assert dti.microsecond[0] == 0
            assert dti.dayofweek[0] == 3

            assert dti.dayofyear[0] == 1
            assert dti.dayofyear[120] == 121

            assert dti.isocalendar().week.iloc[0] == 1
            assert dti.isocalendar().week.iloc[120] == 18

            assert dti.quarter[0] == 1
            assert dti.quarter[120] == 2

            assert dti.days_in_month[0] == 31
            assert dti.days_in_month[90] == 30

            assert dti.is_month_start[0]
            assert not dti.is_month_start[1]
            assert dti.is_month_start[31]
            assert dti.is_quarter_start[0]
            assert dti.is_quarter_start[90]
            assert dti.is_year_start[0]
            assert not dti.is_year_start[364]
            assert not dti.is_month_end[0]
            assert dti.is_month_end[30]
            assert not dti.is_month_end[31]
            assert dti.is_month_end[364]
            assert not dti.is_quarter_end[0]
            assert not dti.is_quarter_end[30]
            assert dti.is_quarter_end[89]
            assert dti.is_quarter_end[364]
            assert not dti.is_year_end[0]
            assert dti.is_year_end[364]

            assert len(dti.year) == 365
            assert len(dti.month) == 365
            assert len(dti.day) == 365
            assert len(dti.hour) == 365
            assert len(dti.minute) == 365
            assert len(dti.second) == 365
            assert len(dti.microsecond) == 365
            assert len(dti.dayofweek) == 365
            assert len(dti.dayofyear) == 365
            assert len(dti.isocalendar()) == 365
            assert len(dti.quarter) == 365
            assert len(dti.is_month_start) == 365
            assert len(dti.is_month_end) == 365
            assert len(dti.is_quarter_start) == 365
            assert len(dti.is_quarter_end) == 365
            assert len(dti.is_year_start) == 365
            assert len(dti.is_year_end) == 365

            dti.name = "name"

            # non boolean accessors -> return Index
            for accessor in DatetimeArray._field_ops:
                res = getattr(dti, accessor)
                assert len(res) == 365
                assert isinstance(res, Index)
                assert res.name == "name"

            # boolean accessors -> return array
            for accessor in DatetimeArray._bool_ops:
                res = getattr(dti, accessor)
                assert len(res) == 365
                assert isinstance(res, np.ndarray)

            # test boolean indexing
            res = dti[dti.is_quarter_start]
            exp = dti[[0, 90, 181, 273]]
            tm.assert_index_equal(res, exp)
            res = dti[dti.is_leap_year]
            exp = DatetimeIndex([], freq="D", tz=dti.tz, name="name")
            tm.assert_index_equal(res, exp)

    def test_datetimeindex_accessors2(self):
        dti = date_range(freq="BQ-FEB", start=datetime(1998, 1, 1), periods=4)

        assert sum(dti.is_quarter_start) == 0
        assert sum(dti.is_quarter_end) == 4
        assert sum(dti.is_year_start) == 0
        assert sum(dti.is_year_end) == 1

    def test_datetimeindex_accessors3(self):
        # Ensure is_start/end accessors throw ValueError for CustomBusinessDay,
        bday_egypt = offsets.CustomBusinessDay(weekmask="Sun Mon Tue Wed Thu")
        dti = date_range(datetime(2013, 4, 30), periods=5, freq=bday_egypt)
        msg = "Custom business days is not supported by is_month_start"
        with pytest.raises(ValueError, match=msg):
            dti.is_month_start

    def test_datetimeindex_accessors4(self):
        dti = DatetimeIndex(["2000-01-01", "2000-01-02", "2000-01-03"])

        assert dti.is_month_start[0] == 1

    def test_datetimeindex_accessors5(self):
        freq_m = to_offset("M")
        bm = to_offset("BM")
        qfeb = to_offset("Q-FEB")
        qsfeb = to_offset("QS-FEB")
        bq = to_offset("BQ")
        bqs_apr = to_offset("BQS-APR")
        as_nov = to_offset("AS-NOV")

        tests = [
            (freq_m.is_month_start(Timestamp("2013-06-01")), 1),
            (bm.is_month_start(Timestamp("2013-06-01")), 0),
            (freq_m.is_month_start(Timestamp("2013-06-03")), 0),
            (bm.is_month_start(Timestamp("2013-06-03")), 1),
            (qfeb.is_month_end(Timestamp("2013-02-28")), 1),
            (qfeb.is_quarter_end(Timestamp("2013-02-28")), 1),
            (qfeb.is_year_end(Timestamp("2013-02-28")), 1),
            (qfeb.is_month_start(Timestamp("2013-03-01")), 1),
            (qfeb.is_quarter_start(Timestamp("2013-03-01")), 1),
            (qfeb.is_year_start(Timestamp("2013-03-01")), 1),
            (qsfeb.is_month_end(Timestamp("2013-03-31")), 1),
            (qsfeb.is_quarter_end(Timestamp("2013-03-31")), 0),
            (qsfeb.is_year_end(Timestamp("2013-03-31")), 0),
            (qsfeb.is_month_start(Timestamp("2013-02-01")), 1),
            (qsfeb.is_quarter_start(Timestamp("2013-02-01")), 1),
            (qsfeb.is_year_start(Timestamp("2013-02-01")), 1),
            (bq.is_month_end(Timestamp("2013-06-30")), 0),
            (bq.is_quarter_end(Timestamp("2013-06-30")), 0),
            (bq.is_year_end(Timestamp("2013-06-30")), 0),
            (bq.is_month_end(Timestamp("2013-06-28")), 1),
            (bq.is_quarter_end(Timestamp("2013-06-28")), 1),
            (bq.is_year_end(Timestamp("2013-06-28")), 0),
            (bqs_apr.is_month_end(Timestamp("2013-06-30")), 0),
            (bqs_apr.is_quarter_end(Timestamp("2013-06-30")), 0),
            (bqs_apr.is_year_end(Timestamp("2013-06-30")), 0),
            (bqs_apr.is_month_end(Timestamp("2013-06-28")), 1),
            (bqs_apr.is_quarter_end(Timestamp("2013-06-28")), 1),
            (bqs_apr.is_year_end(Timestamp("2013-03-29")), 1),
            (as_nov.is_year_start(Timestamp("2013-11-01")), 1),
            (as_nov.is_year_end(Timestamp("2013-10-31")), 1),
            (Timestamp("2012-02-01").days_in_month, 29),
            (Timestamp("2013-02-01").days_in_month, 28),
        ]

        for ts, value in tests:
            assert ts == value

    def test_datetimeindex_accessors6(self):
        # GH 6538: Check that DatetimeIndex and its TimeStamp elements
        # return the same weekofyear accessor close to new year w/ tz
        dates = ["2013/12/29", "2013/12/30", "2013/12/31"]
        dates = DatetimeIndex(dates, tz="Europe/Brussels")
        expected = [52, 1, 1]
        assert dates.isocalendar().week.tolist() == expected
        assert [d.weekofyear for d in dates] == expected

    # GH 12806
    # error: Unsupported operand types for + ("List[None]" and "List[str]")
    @pytest.mark.parametrize(
        "time_locale", [None] + tm.get_locales()  # type: ignore[operator]
    )
    def test_datetime_name_accessors(self, time_locale):
        # Test Monday -> Sunday and January -> December, in that sequence
        if time_locale is None:
            # If the time_locale is None, day-name and month_name should
            # return the english attributes
            expected_days = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
            expected_months = [
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ]
        else:
            with tm.set_locale(time_locale, locale.LC_TIME):
                expected_days = calendar.day_name[:]
                expected_months = calendar.month_name[1:]

        # GH#11128
        dti = date_range(freq="D", start=datetime(1998, 1, 1), periods=365)
        english_days = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        for day, name, eng_name in zip(range(4, 11), expected_days, english_days):
            name = name.capitalize()
            assert dti.day_name(locale=time_locale)[day] == name
            assert dti.day_name(locale=None)[day] == eng_name
            ts = Timestamp(datetime(2016, 4, day))
            assert ts.day_name(locale=time_locale) == name
        dti = dti.append(DatetimeIndex([pd.NaT]))
        assert np.isnan(dti.day_name(locale=time_locale)[-1])
        ts = Timestamp(pd.NaT)
        assert np.isnan(ts.day_name(locale=time_locale))

        # GH#12805
        dti = date_range(freq="M", start="2012", end="2013")
        result = dti.month_name(locale=time_locale)
        expected = Index([month.capitalize() for month in expected_months])

        # work around different normalization schemes
        # https://github.com/pandas-dev/pandas/issues/22342
        result = result.str.normalize("NFD")
        expected = expected.str.normalize("NFD")

        tm.assert_index_equal(result, expected)

        for date, expected in zip(dti, expected_months):
            result = date.month_name(locale=time_locale)
            expected = expected.capitalize()

            result = unicodedata.normalize("NFD", result)
            expected = unicodedata.normalize("NFD", result)

            assert result == expected
        dti = dti.append(DatetimeIndex([pd.NaT]))
        assert np.isnan(dti.month_name(locale=time_locale)[-1])

    def test_nanosecond_field(self):
        dti = DatetimeIndex(np.arange(10))
        expected = Index(np.arange(10, dtype=np.int32))

        tm.assert_index_equal(dti.nanosecond, expected)


def test_iter_readonly():
    # GH#28055 ints_to_pydatetime with readonly array
    arr = np.array([np.datetime64("2012-02-15T12:00:00.000000000")])
    arr.setflags(write=False)
    dti = pd.to_datetime(arr)
    list(dti)


def test_add_timedelta_preserves_freq():
    # GH#37295 should hold for any DTI with freq=None or Tick freq
    tz = "Canada/Eastern"
    dti = date_range(
        start=Timestamp("2019-03-26 00:00:00-0400", tz=tz),
        end=Timestamp("2020-10-17 00:00:00-0400", tz=tz),
        freq="D",
    )
    result = dti + Timedelta(days=1)
    assert result.freq == dti.freq
