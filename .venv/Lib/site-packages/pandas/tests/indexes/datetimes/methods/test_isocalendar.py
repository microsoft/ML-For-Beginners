from pandas import (
    DataFrame,
    DatetimeIndex,
)
import pandas._testing as tm


def test_isocalendar_returns_correct_values_close_to_new_year_with_tz():
    # GH#6538: Check that DatetimeIndex and its TimeStamp elements
    # return the same weekofyear accessor close to new year w/ tz
    dates = ["2013/12/29", "2013/12/30", "2013/12/31"]
    dates = DatetimeIndex(dates, tz="Europe/Brussels")
    result = dates.isocalendar()
    expected_data_frame = DataFrame(
        [[2013, 52, 7], [2014, 1, 1], [2014, 1, 2]],
        columns=["year", "week", "day"],
        index=dates,
        dtype="UInt32",
    )
    tm.assert_frame_equal(result, expected_data_frame)
