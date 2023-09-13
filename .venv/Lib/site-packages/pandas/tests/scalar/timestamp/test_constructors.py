import calendar
from datetime import (
    date,
    datetime,
    timedelta,
    timezone,
)
import zoneinfo

import dateutil.tz
from dateutil.tz import tzutc
import numpy as np
import pytest
import pytz

from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.compat import PY310
from pandas.errors import OutOfBoundsDatetime

from pandas import (
    NA,
    NaT,
    Period,
    Timedelta,
    Timestamp,
)


class TestTimestampConstructors:
    def test_construct_from_time_unit(self):
        # GH#54097 only passing a time component, no date
        ts = Timestamp("01:01:01.111")
        assert ts.unit == "ms"

    def test_weekday_but_no_day_raises(self):
        # GH#52659
        msg = "Parsing datetimes with weekday but no day information is not supported"
        with pytest.raises(ValueError, match=msg):
            Timestamp("2023 Sept Thu")

    def test_construct_from_string_invalid_raises(self):
        # dateutil (weirdly) parses "200622-12-31" as
        #  datetime(2022, 6, 20, 12, 0, tzinfo=tzoffset(None, -111600)
        #  which besides being mis-parsed, is a tzoffset that will cause
        #  str(ts) to raise ValueError.  Ensure we raise in the constructor
        #  instead.
        # see test_to_datetime_malformed_raise for analogous to_datetime test
        with pytest.raises(ValueError, match="gives an invalid tzoffset"):
            Timestamp("200622-12-31")

    def test_constructor_str_infer_reso(self):
        # non-iso8601 path

        # _parse_delimited_date path
        ts = Timestamp("01/30/2023")
        assert ts.unit == "s"

        # _parse_dateabbr_string path
        ts = Timestamp("2015Q1")
        assert ts.unit == "s"

        # dateutil_parse path
        ts = Timestamp("2016-01-01 1:30:01 PM")
        assert ts.unit == "s"

        ts = Timestamp("2016 June 3 15:25:01.345")
        assert ts.unit == "ms"

        ts = Timestamp("300-01-01")
        assert ts.unit == "s"

        ts = Timestamp("300 June 1:30:01.300")
        assert ts.unit == "ms"

    def test_constructor_from_iso8601_str_with_offset_reso(self):
        # GH#49737
        ts = Timestamp("2016-01-01 04:05:06-01:00")
        assert ts.unit == "s"

        ts = Timestamp("2016-01-01 04:05:06.000-01:00")
        assert ts.unit == "ms"

        ts = Timestamp("2016-01-01 04:05:06.000000-01:00")
        assert ts.unit == "us"

        ts = Timestamp("2016-01-01 04:05:06.000000001-01:00")
        assert ts.unit == "ns"

    def test_constructor_from_date_second_reso(self):
        # GH#49034 constructing from a pydate object gets lowest supported
        #  reso, i.e. seconds
        obj = date(2012, 9, 1)
        ts = Timestamp(obj)
        assert ts.unit == "s"

    @pytest.mark.parametrize("typ", [int, float])
    def test_construct_from_int_float_with_unit_out_of_bound_raises(self, typ):
        # GH#50870  make sure we get a OutOfBoundsDatetime instead of OverflowError
        val = typ(150000000000000)

        msg = f"cannot convert input {val} with the unit 'D'"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp(val, unit="D")

    @pytest.mark.parametrize("typ", [int, float])
    def test_constructor_int_float_with_YM_unit(self, typ):
        # GH#47266 avoid the conversions in cast_from_unit
        val = typ(150)

        ts = Timestamp(val, unit="Y")
        expected = Timestamp("2120-01-01")
        assert ts == expected

        ts = Timestamp(val, unit="M")
        expected = Timestamp("1982-07-01")
        assert ts == expected

    def test_constructor_float_not_round_with_YM_unit_deprecated(self):
        # GH#47267 avoid the conversions in cast_from-unit

        msg = "Conversion of non-round float with unit=[MY] is ambiguous"
        with pytest.raises(ValueError, match=msg):
            Timestamp(150.5, unit="Y")

        with pytest.raises(ValueError, match=msg):
            Timestamp(150.5, unit="M")

    def test_constructor_datetime64_with_tz(self):
        # GH#42288, GH#24559
        dt = np.datetime64("1970-01-01 05:00:00")
        tzstr = "UTC+05:00"

        # pre-2.0 this interpreted dt as a UTC time. in 2.0 this is treated
        #  as a wall-time, consistent with DatetimeIndex behavior
        ts = Timestamp(dt, tz=tzstr)

        alt = Timestamp(dt).tz_localize(tzstr)
        assert ts == alt
        assert ts.hour == 5

    def test_constructor(self):
        base_str = "2014-07-01 09:00"
        base_dt = datetime(2014, 7, 1, 9)
        base_expected = 1_404_205_200_000_000_000

        # confirm base representation is correct
        assert calendar.timegm(base_dt.timetuple()) * 1_000_000_000 == base_expected

        tests = [
            (base_str, base_dt, base_expected),
            (
                "2014-07-01 10:00",
                datetime(2014, 7, 1, 10),
                base_expected + 3600 * 1_000_000_000,
            ),
            (
                "2014-07-01 09:00:00.000008000",
                datetime(2014, 7, 1, 9, 0, 0, 8),
                base_expected + 8000,
            ),
            (
                "2014-07-01 09:00:00.000000005",
                Timestamp("2014-07-01 09:00:00.000000005"),
                base_expected + 5,
            ),
        ]

        timezones = [
            (None, 0),
            ("UTC", 0),
            (pytz.utc, 0),
            ("Asia/Tokyo", 9),
            ("US/Eastern", -4),
            ("dateutil/US/Pacific", -7),
            (pytz.FixedOffset(-180), -3),
            (dateutil.tz.tzoffset(None, 18000), 5),
        ]

        for date_str, date_obj, expected in tests:
            for result in [Timestamp(date_str), Timestamp(date_obj)]:
                result = result.as_unit("ns")  # test originally written before non-nano
                # only with timestring
                assert result.as_unit("ns")._value == expected

                # re-creation shouldn't affect to internal value
                result = Timestamp(result)
                assert result.as_unit("ns")._value == expected

            # with timezone
            for tz, offset in timezones:
                for result in [Timestamp(date_str, tz=tz), Timestamp(date_obj, tz=tz)]:
                    result = result.as_unit(
                        "ns"
                    )  # test originally written before non-nano
                    expected_tz = expected - offset * 3600 * 1_000_000_000
                    assert result.as_unit("ns")._value == expected_tz

                    # should preserve tz
                    result = Timestamp(result)
                    assert result.as_unit("ns")._value == expected_tz

                    # should convert to UTC
                    if tz is not None:
                        result = Timestamp(result).tz_convert("UTC")
                    else:
                        result = Timestamp(result, tz="UTC")
                    expected_utc = expected - offset * 3600 * 1_000_000_000
                    assert result.as_unit("ns")._value == expected_utc

    def test_constructor_with_stringoffset(self):
        # GH 7833
        base_str = "2014-07-01 11:00:00+02:00"
        base_dt = datetime(2014, 7, 1, 9)
        base_expected = 1_404_205_200_000_000_000

        # confirm base representation is correct
        assert calendar.timegm(base_dt.timetuple()) * 1_000_000_000 == base_expected

        tests = [
            (base_str, base_expected),
            ("2014-07-01 12:00:00+02:00", base_expected + 3600 * 1_000_000_000),
            ("2014-07-01 11:00:00.000008000+02:00", base_expected + 8000),
            ("2014-07-01 11:00:00.000000005+02:00", base_expected + 5),
        ]

        timezones = [
            (None, 0),
            ("UTC", 0),
            (pytz.utc, 0),
            ("Asia/Tokyo", 9),
            ("US/Eastern", -4),
            ("dateutil/US/Pacific", -7),
            (pytz.FixedOffset(-180), -3),
            (dateutil.tz.tzoffset(None, 18000), 5),
        ]

        for date_str, expected in tests:
            for result in [Timestamp(date_str)]:
                # only with timestring
                assert result.as_unit("ns")._value == expected

                # re-creation shouldn't affect to internal value
                result = Timestamp(result)
                assert result.as_unit("ns")._value == expected

            # with timezone
            for tz, offset in timezones:
                result = Timestamp(date_str, tz=tz)
                expected_tz = expected
                assert result.as_unit("ns")._value == expected_tz

                # should preserve tz
                result = Timestamp(result)
                assert result.as_unit("ns")._value == expected_tz

                # should convert to UTC
                result = Timestamp(result).tz_convert("UTC")
                expected_utc = expected
                assert result.as_unit("ns")._value == expected_utc

        # This should be 2013-11-01 05:00 in UTC
        # converted to Chicago tz
        result = Timestamp("2013-11-01 00:00:00-0500", tz="America/Chicago")
        assert result._value == Timestamp("2013-11-01 05:00")._value
        expected = "Timestamp('2013-11-01 00:00:00-0500', tz='America/Chicago')"
        assert repr(result) == expected
        assert result == eval(repr(result))

        # This should be 2013-11-01 05:00 in UTC
        # converted to Tokyo tz (+09:00)
        result = Timestamp("2013-11-01 00:00:00-0500", tz="Asia/Tokyo")
        assert result._value == Timestamp("2013-11-01 05:00")._value
        expected = "Timestamp('2013-11-01 14:00:00+0900', tz='Asia/Tokyo')"
        assert repr(result) == expected
        assert result == eval(repr(result))

        # GH11708
        # This should be 2015-11-18 10:00 in UTC
        # converted to Asia/Katmandu
        result = Timestamp("2015-11-18 15:45:00+05:45", tz="Asia/Katmandu")
        assert result._value == Timestamp("2015-11-18 10:00")._value
        expected = "Timestamp('2015-11-18 15:45:00+0545', tz='Asia/Katmandu')"
        assert repr(result) == expected
        assert result == eval(repr(result))

        # This should be 2015-11-18 10:00 in UTC
        # converted to Asia/Kolkata
        result = Timestamp("2015-11-18 15:30:00+05:30", tz="Asia/Kolkata")
        assert result._value == Timestamp("2015-11-18 10:00")._value
        expected = "Timestamp('2015-11-18 15:30:00+0530', tz='Asia/Kolkata')"
        assert repr(result) == expected
        assert result == eval(repr(result))

    def test_constructor_invalid(self):
        msg = "Cannot convert input"
        with pytest.raises(TypeError, match=msg):
            Timestamp(slice(2))
        msg = "Cannot convert Period"
        with pytest.raises(ValueError, match=msg):
            Timestamp(Period("1000-01-01"))

    def test_constructor_invalid_tz(self):
        # GH#17690
        msg = (
            "Argument 'tzinfo' has incorrect type "
            r"\(expected datetime.tzinfo, got str\)"
        )
        with pytest.raises(TypeError, match=msg):
            Timestamp("2017-10-22", tzinfo="US/Eastern")

        msg = "at most one of"
        with pytest.raises(ValueError, match=msg):
            Timestamp("2017-10-22", tzinfo=pytz.utc, tz="UTC")

        msg = "Cannot pass a date attribute keyword argument when passing a date string"
        with pytest.raises(ValueError, match=msg):
            # GH#5168
            # case where user tries to pass tz as an arg, not kwarg, gets
            # interpreted as `year`
            Timestamp("2012-01-01", "US/Pacific")

    def test_constructor_strptime(self):
        # GH25016
        # Test support for Timestamp.strptime
        fmt = "%Y%m%d-%H%M%S-%f%z"
        ts = "20190129-235348-000001+0000"
        msg = r"Timestamp.strptime\(\) is not implemented"
        with pytest.raises(NotImplementedError, match=msg):
            Timestamp.strptime(ts, fmt)

    def test_constructor_tz_or_tzinfo(self):
        # GH#17943, GH#17690, GH#5168
        stamps = [
            Timestamp(year=2017, month=10, day=22, tz="UTC"),
            Timestamp(year=2017, month=10, day=22, tzinfo=pytz.utc),
            Timestamp(year=2017, month=10, day=22, tz=pytz.utc),
            Timestamp(datetime(2017, 10, 22), tzinfo=pytz.utc),
            Timestamp(datetime(2017, 10, 22), tz="UTC"),
            Timestamp(datetime(2017, 10, 22), tz=pytz.utc),
        ]
        assert all(ts == stamps[0] for ts in stamps)

    def test_constructor_positional_with_tzinfo(self):
        # GH#31929
        ts = Timestamp(2020, 12, 31, tzinfo=timezone.utc)
        expected = Timestamp("2020-12-31", tzinfo=timezone.utc)
        assert ts == expected

    @pytest.mark.parametrize("kwd", ["nanosecond", "microsecond", "second", "minute"])
    def test_constructor_positional_keyword_mixed_with_tzinfo(self, kwd, request):
        # TODO: if we passed microsecond with a keyword we would mess up
        #  xref GH#45307
        if kwd != "nanosecond":
            # nanosecond is keyword-only as of 2.0, others are not
            mark = pytest.mark.xfail(reason="GH#45307")
            request.node.add_marker(mark)

        kwargs = {kwd: 4}
        ts = Timestamp(2020, 12, 31, tzinfo=timezone.utc, **kwargs)

        td_kwargs = {kwd + "s": 4}
        td = Timedelta(**td_kwargs)
        expected = Timestamp("2020-12-31", tz=timezone.utc) + td
        assert ts == expected

    def test_constructor_positional(self):
        # see gh-10758
        msg = (
            "'NoneType' object cannot be interpreted as an integer"
            if PY310
            else "an integer is required"
        )
        with pytest.raises(TypeError, match=msg):
            Timestamp(2000, 1)

        msg = "month must be in 1..12"
        with pytest.raises(ValueError, match=msg):
            Timestamp(2000, 0, 1)
        with pytest.raises(ValueError, match=msg):
            Timestamp(2000, 13, 1)

        msg = "day is out of range for month"
        with pytest.raises(ValueError, match=msg):
            Timestamp(2000, 1, 0)
        with pytest.raises(ValueError, match=msg):
            Timestamp(2000, 1, 32)

        # see gh-11630
        assert repr(Timestamp(2015, 11, 12)) == repr(Timestamp("20151112"))
        assert repr(Timestamp(2015, 11, 12, 1, 2, 3, 999999)) == repr(
            Timestamp("2015-11-12 01:02:03.999999")
        )

    def test_constructor_keyword(self):
        # GH 10758
        msg = "function missing required argument 'day'|Required argument 'day'"
        with pytest.raises(TypeError, match=msg):
            Timestamp(year=2000, month=1)

        msg = "month must be in 1..12"
        with pytest.raises(ValueError, match=msg):
            Timestamp(year=2000, month=0, day=1)
        with pytest.raises(ValueError, match=msg):
            Timestamp(year=2000, month=13, day=1)

        msg = "day is out of range for month"
        with pytest.raises(ValueError, match=msg):
            Timestamp(year=2000, month=1, day=0)
        with pytest.raises(ValueError, match=msg):
            Timestamp(year=2000, month=1, day=32)

        assert repr(Timestamp(year=2015, month=11, day=12)) == repr(
            Timestamp("20151112")
        )

        assert repr(
            Timestamp(
                year=2015,
                month=11,
                day=12,
                hour=1,
                minute=2,
                second=3,
                microsecond=999999,
            )
        ) == repr(Timestamp("2015-11-12 01:02:03.999999"))

    def test_constructor_fromordinal(self):
        base = datetime(2000, 1, 1)

        ts = Timestamp.fromordinal(base.toordinal())
        assert base == ts
        assert base.toordinal() == ts.toordinal()

        ts = Timestamp.fromordinal(base.toordinal(), tz="US/Eastern")
        assert Timestamp("2000-01-01", tz="US/Eastern") == ts
        assert base.toordinal() == ts.toordinal()

        # GH#3042
        dt = datetime(2011, 4, 16, 0, 0)
        ts = Timestamp.fromordinal(dt.toordinal())
        assert ts.to_pydatetime() == dt

        # with a tzinfo
        stamp = Timestamp("2011-4-16", tz="US/Eastern")
        dt_tz = stamp.to_pydatetime()
        ts = Timestamp.fromordinal(dt_tz.toordinal(), tz="US/Eastern")
        assert ts.to_pydatetime() == dt_tz

    @pytest.mark.parametrize(
        "result",
        [
            Timestamp(datetime(2000, 1, 2, 3, 4, 5, 6), nanosecond=1),
            Timestamp(
                year=2000,
                month=1,
                day=2,
                hour=3,
                minute=4,
                second=5,
                microsecond=6,
                nanosecond=1,
            ),
            Timestamp(
                year=2000,
                month=1,
                day=2,
                hour=3,
                minute=4,
                second=5,
                microsecond=6,
                nanosecond=1,
                tz="UTC",
            ),
            Timestamp(2000, 1, 2, 3, 4, 5, 6, None, nanosecond=1),
            Timestamp(2000, 1, 2, 3, 4, 5, 6, tz=pytz.UTC, nanosecond=1),
        ],
    )
    def test_constructor_nanosecond(self, result):
        # GH 18898
        # As of 2.0 (GH 49416), nanosecond should not be accepted positionally
        expected = Timestamp(datetime(2000, 1, 2, 3, 4, 5, 6), tz=result.tz)
        expected = expected + Timedelta(nanoseconds=1)
        assert result == expected

    @pytest.mark.parametrize("z", ["Z0", "Z00"])
    def test_constructor_invalid_Z0_isostring(self, z):
        # GH 8910
        msg = f"Unknown datetime string format, unable to parse: 2014-11-02 01:00{z}"
        with pytest.raises(ValueError, match=msg):
            Timestamp(f"2014-11-02 01:00{z}")

    @pytest.mark.parametrize(
        "arg",
        [
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "second",
            "microsecond",
            "nanosecond",
        ],
    )
    def test_invalid_date_kwarg_with_string_input(self, arg):
        kwarg = {arg: 1}
        msg = "Cannot pass a date attribute keyword argument"
        with pytest.raises(ValueError, match=msg):
            Timestamp("2010-10-10 12:59:59.999999999", **kwarg)

    def test_out_of_bounds_integer_value(self):
        # GH#26651 check that we raise OutOfBoundsDatetime, not OverflowError
        msg = str(Timestamp.max._value * 2)
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp(Timestamp.max._value * 2)
        msg = str(Timestamp.min._value * 2)
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp(Timestamp.min._value * 2)

    def test_out_of_bounds_value(self):
        one_us = np.timedelta64(1).astype("timedelta64[us]")

        # By definition we can't go out of bounds in [ns], so we
        # convert the datetime64s to [us] so we can go out of bounds
        min_ts_us = np.datetime64(Timestamp.min).astype("M8[us]") + one_us
        max_ts_us = np.datetime64(Timestamp.max).astype("M8[us]")

        # No error for the min/max datetimes
        Timestamp(min_ts_us)
        Timestamp(max_ts_us)

        # We used to raise on these before supporting non-nano
        us_val = NpyDatetimeUnit.NPY_FR_us.value
        assert Timestamp(min_ts_us - one_us)._creso == us_val
        assert Timestamp(max_ts_us + one_us)._creso == us_val

        # https://github.com/numpy/numpy/issues/22346 for why
        #  we can't use the same construction as above with minute resolution

        # too_low, too_high are the _just_ outside the range of M8[s]
        too_low = np.datetime64("-292277022657-01-27T08:29", "m")
        too_high = np.datetime64("292277026596-12-04T15:31", "m")

        msg = "Out of bounds"
        # One us less than the minimum is an error
        with pytest.raises(ValueError, match=msg):
            Timestamp(too_low)

        # One us more than the maximum is an error
        with pytest.raises(ValueError, match=msg):
            Timestamp(too_high)

    def test_out_of_bounds_string(self):
        msg = "Cannot cast .* to unit='ns' without overflow"
        with pytest.raises(ValueError, match=msg):
            Timestamp("1676-01-01").as_unit("ns")
        with pytest.raises(ValueError, match=msg):
            Timestamp("2263-01-01").as_unit("ns")

        ts = Timestamp("2263-01-01")
        assert ts.unit == "s"

        ts = Timestamp("1676-01-01")
        assert ts.unit == "s"

    def test_barely_out_of_bounds(self):
        # GH#19529
        # GH#19382 close enough to bounds that dropping nanos would result
        # in an in-bounds datetime
        msg = "Out of bounds nanosecond timestamp: 2262-04-11 23:47:16"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp("2262-04-11 23:47:16.854775808")

    def test_bounds_with_different_units(self):
        out_of_bounds_dates = ("1677-09-21", "2262-04-12")

        time_units = ("D", "h", "m", "s", "ms", "us")

        for date_string in out_of_bounds_dates:
            for unit in time_units:
                dt64 = np.datetime64(date_string, unit)
                ts = Timestamp(dt64)
                if unit in ["s", "ms", "us"]:
                    # We can preserve the input unit
                    assert ts._value == dt64.view("i8")
                else:
                    # we chose the closest unit that we _do_ support
                    assert ts._creso == NpyDatetimeUnit.NPY_FR_s.value

        # With more extreme cases, we can't even fit inside second resolution
        info = np.iinfo(np.int64)
        msg = "Out of bounds nanosecond timestamp:"
        for value in [info.min + 1, info.max]:
            for unit in ["D", "h", "m"]:
                dt64 = np.datetime64(value, unit)
                with pytest.raises(OutOfBoundsDatetime, match=msg):
                    Timestamp(dt64)

        in_bounds_dates = ("1677-09-23", "2262-04-11")

        for date_string in in_bounds_dates:
            for unit in time_units:
                dt64 = np.datetime64(date_string, unit)
                Timestamp(dt64)

    @pytest.mark.parametrize("arg", ["001-01-01", "0001-01-01"])
    def test_out_of_bounds_string_consistency(self, arg):
        # GH 15829
        msg = "Cannot cast 0001-01-01 00:00:00 to unit='ns' without overflow"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp(arg).as_unit("ns")

        ts = Timestamp(arg)
        assert ts.unit == "s"
        assert ts.year == ts.month == ts.day == 1

    def test_min_valid(self):
        # Ensure that Timestamp.min is a valid Timestamp
        Timestamp(Timestamp.min)

    def test_max_valid(self):
        # Ensure that Timestamp.max is a valid Timestamp
        Timestamp(Timestamp.max)

    def test_now(self):
        # GH#9000
        ts_from_string = Timestamp("now")
        ts_from_method = Timestamp.now()
        ts_datetime = datetime.now()

        ts_from_string_tz = Timestamp("now", tz="US/Eastern")
        ts_from_method_tz = Timestamp.now(tz="US/Eastern")

        # Check that the delta between the times is less than 1s (arbitrarily
        # small)
        delta = Timedelta(seconds=1)
        assert abs(ts_from_method - ts_from_string) < delta
        assert abs(ts_datetime - ts_from_method) < delta
        assert abs(ts_from_method_tz - ts_from_string_tz) < delta
        assert (
            abs(
                ts_from_string_tz.tz_localize(None)
                - ts_from_method_tz.tz_localize(None)
            )
            < delta
        )

    def test_today(self):
        ts_from_string = Timestamp("today")
        ts_from_method = Timestamp.today()
        ts_datetime = datetime.today()

        ts_from_string_tz = Timestamp("today", tz="US/Eastern")
        ts_from_method_tz = Timestamp.today(tz="US/Eastern")

        # Check that the delta between the times is less than 1s (arbitrarily
        # small)
        delta = Timedelta(seconds=1)
        assert abs(ts_from_method - ts_from_string) < delta
        assert abs(ts_datetime - ts_from_method) < delta
        assert abs(ts_from_method_tz - ts_from_string_tz) < delta
        assert (
            abs(
                ts_from_string_tz.tz_localize(None)
                - ts_from_method_tz.tz_localize(None)
            )
            < delta
        )

    @pytest.mark.parametrize("tz", [None, pytz.timezone("US/Pacific")])
    def test_disallow_setting_tz(self, tz):
        # GH 3746
        ts = Timestamp("2010")
        msg = "Cannot directly set timezone"
        with pytest.raises(AttributeError, match=msg):
            ts.tz = tz

    @pytest.mark.parametrize("offset", ["+0300", "+0200"])
    def test_construct_timestamp_near_dst(self, offset):
        # GH 20854
        expected = Timestamp(f"2016-10-30 03:00:00{offset}", tz="Europe/Helsinki")
        result = Timestamp(expected).tz_convert("Europe/Helsinki")
        assert result == expected

    @pytest.mark.parametrize(
        "arg", ["2013/01/01 00:00:00+09:00", "2013-01-01 00:00:00+09:00"]
    )
    def test_construct_with_different_string_format(self, arg):
        # GH 12064
        result = Timestamp(arg)
        expected = Timestamp(datetime(2013, 1, 1), tz=pytz.FixedOffset(540))
        assert result == expected

    @pytest.mark.parametrize("box", [datetime, Timestamp])
    def test_raise_tz_and_tzinfo_in_datetime_input(self, box):
        # GH 23579
        kwargs = {"year": 2018, "month": 1, "day": 1, "tzinfo": pytz.utc}
        msg = "Cannot pass a datetime or Timestamp"
        with pytest.raises(ValueError, match=msg):
            Timestamp(box(**kwargs), tz="US/Pacific")
        msg = "Cannot pass a datetime or Timestamp"
        with pytest.raises(ValueError, match=msg):
            Timestamp(box(**kwargs), tzinfo=pytz.timezone("US/Pacific"))

    def test_dont_convert_dateutil_utc_to_pytz_utc(self):
        result = Timestamp(datetime(2018, 1, 1), tz=tzutc())
        expected = Timestamp(datetime(2018, 1, 1)).tz_localize(tzutc())
        assert result == expected

    def test_constructor_subclassed_datetime(self):
        # GH 25851
        # ensure that subclassed datetime works for
        # Timestamp creation
        class SubDatetime(datetime):
            pass

        data = SubDatetime(2000, 1, 1)
        result = Timestamp(data)
        expected = Timestamp(2000, 1, 1)
        assert result == expected

    def test_constructor_fromisocalendar(self):
        # GH 30395
        expected_timestamp = Timestamp("2000-01-03 00:00:00")
        expected_stdlib = datetime.fromisocalendar(2000, 1, 1)
        result = Timestamp.fromisocalendar(2000, 1, 1)
        assert result == expected_timestamp
        assert result == expected_stdlib
        assert isinstance(result, Timestamp)


def test_constructor_ambiguous_dst():
    # GH 24329
    # Make sure that calling Timestamp constructor
    # on Timestamp created from ambiguous time
    # doesn't change Timestamp.value
    ts = Timestamp(1382835600000000000, tz="dateutil/Europe/London")
    expected = ts._value
    result = Timestamp(ts)._value
    assert result == expected


@pytest.mark.parametrize("epoch", [1552211999999999872, 1552211999999999999])
def test_constructor_before_dst_switch(epoch):
    # GH 31043
    # Make sure that calling Timestamp constructor
    # on time just before DST switch doesn't lead to
    # nonexistent time or value change
    ts = Timestamp(epoch, tz="dateutil/America/Los_Angeles")
    result = ts.tz.dst(ts)
    expected = timedelta(seconds=0)
    assert Timestamp(ts)._value == epoch
    assert result == expected


def test_timestamp_constructor_identity():
    # Test for #30543
    expected = Timestamp("2017-01-01T12")
    result = Timestamp(expected)
    assert result is expected


@pytest.mark.parametrize("kwargs", [{}, {"year": 2020}, {"year": 2020, "month": 1}])
def test_constructor_missing_keyword(kwargs):
    # GH 31200

    # The exact error message of datetime() depends on its version
    msg1 = r"function missing required argument '(year|month|day)' \(pos [123]\)"
    msg2 = r"Required argument '(year|month|day)' \(pos [123]\) not found"
    msg = "|".join([msg1, msg2])

    with pytest.raises(TypeError, match=msg):
        Timestamp(**kwargs)


@pytest.mark.parametrize("nano", [-1, 1000])
def test_timestamp_nano_range(nano):
    # GH 48255
    with pytest.raises(ValueError, match="nanosecond must be in 0..999"):
        Timestamp(year=2022, month=1, day=1, nanosecond=nano)


def test_non_nano_value():
    # https://github.com/pandas-dev/pandas/issues/49076
    result = Timestamp("1800-01-01", unit="s").value
    # `.value` shows nanoseconds, even though unit is 's'
    assert result == -5364662400000000000

    # out-of-nanoseconds-bounds `.value` raises informative message
    msg = (
        r"Cannot convert Timestamp to nanoseconds without overflow. "
        r"Use `.asm8.view\('i8'\)` to cast represent Timestamp in its "
        r"own unit \(here, s\).$"
    )
    ts = Timestamp("0300-01-01")
    with pytest.raises(OverflowError, match=msg):
        ts.value
    # check that the suggested workaround actually works
    result = ts.asm8.view("i8")
    assert result == -52700112000


def test_timestamp_constructor_invalid_fold_raise():
    # Test forGH #25057
    # Valid fold values are only [None, 0, 1]
    msg = "Valid values for the fold argument are None, 0, or 1."
    with pytest.raises(ValueError, match=msg):
        Timestamp(123, fold=2)


def test_timestamp_constructor_pytz_fold_raise():
    # Test for GH#25057
    # pytz doesn't support fold. Check that we raise
    # if fold is passed with pytz
    msg = "pytz timezones do not support fold. Please use dateutil timezones."
    tz = pytz.timezone("Europe/London")
    with pytest.raises(ValueError, match=msg):
        Timestamp(datetime(2019, 10, 27, 0, 30, 0, 0), tz=tz, fold=0)


@pytest.mark.parametrize("fold", [0, 1])
@pytest.mark.parametrize(
    "ts_input",
    [
        1572136200000000000,
        1572136200000000000.0,
        np.datetime64(1572136200000000000, "ns"),
        "2019-10-27 01:30:00+01:00",
        datetime(2019, 10, 27, 0, 30, 0, 0, tzinfo=timezone.utc),
    ],
)
def test_timestamp_constructor_fold_conflict(ts_input, fold):
    # Test for GH#25057
    # Check that we raise on fold conflict
    msg = (
        "Cannot pass fold with possibly unambiguous input: int, float, "
        "numpy.datetime64, str, or timezone-aware datetime-like. "
        "Pass naive datetime-like or build Timestamp from components."
    )
    with pytest.raises(ValueError, match=msg):
        Timestamp(ts_input=ts_input, fold=fold)


@pytest.mark.parametrize("tz", ["dateutil/Europe/London", None])
@pytest.mark.parametrize("fold", [0, 1])
def test_timestamp_constructor_retain_fold(tz, fold):
    # Test for GH#25057
    # Check that we retain fold
    ts = Timestamp(year=2019, month=10, day=27, hour=1, minute=30, tz=tz, fold=fold)
    result = ts.fold
    expected = fold
    assert result == expected


try:
    _tzs = [
        "dateutil/Europe/London",
        zoneinfo.ZoneInfo("Europe/London"),
    ]
except zoneinfo.ZoneInfoNotFoundError:
    _tzs = ["dateutil/Europe/London"]


@pytest.mark.parametrize("tz", _tzs)
@pytest.mark.parametrize(
    "ts_input,fold_out",
    [
        (1572136200000000000, 0),
        (1572139800000000000, 1),
        ("2019-10-27 01:30:00+01:00", 0),
        ("2019-10-27 01:30:00+00:00", 1),
        (datetime(2019, 10, 27, 1, 30, 0, 0, fold=0), 0),
        (datetime(2019, 10, 27, 1, 30, 0, 0, fold=1), 1),
    ],
)
def test_timestamp_constructor_infer_fold_from_value(tz, ts_input, fold_out):
    # Test for GH#25057
    # Check that we infer fold correctly based on timestamps since utc
    # or strings
    ts = Timestamp(ts_input, tz=tz)
    result = ts.fold
    expected = fold_out
    assert result == expected


@pytest.mark.parametrize("tz", ["dateutil/Europe/London"])
@pytest.mark.parametrize(
    "ts_input,fold,value_out",
    [
        (datetime(2019, 10, 27, 1, 30, 0, 0), 0, 1572136200000000),
        (datetime(2019, 10, 27, 1, 30, 0, 0), 1, 1572139800000000),
    ],
)
def test_timestamp_constructor_adjust_value_for_fold(tz, ts_input, fold, value_out):
    # Test for GH#25057
    # Check that we adjust value for fold correctly
    # based on timestamps since utc
    ts = Timestamp(ts_input, tz=tz, fold=fold)
    result = ts._value
    expected = value_out
    assert result == expected


@pytest.mark.parametrize("na_value", [None, np.nan, np.datetime64("NaT"), NaT, NA])
def test_timestamp_constructor_na_value(na_value):
    # GH45481
    result = Timestamp(na_value)
    expected = NaT
    assert result is expected
