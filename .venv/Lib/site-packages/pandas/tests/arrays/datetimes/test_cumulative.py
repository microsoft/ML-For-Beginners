import pytest

import pandas._testing as tm
from pandas.core.arrays import DatetimeArray


class TestAccumulator:
    def test_accumulators_freq(self):
        # GH#50297
        arr = DatetimeArray._from_sequence_not_strict(
            [
                "2000-01-01",
                "2000-01-02",
                "2000-01-03",
            ],
            freq="D",
        )
        result = arr._accumulate("cummin")
        expected = DatetimeArray._from_sequence_not_strict(
            ["2000-01-01"] * 3, freq=None
        )
        tm.assert_datetime_array_equal(result, expected)

        result = arr._accumulate("cummax")
        expected = DatetimeArray._from_sequence_not_strict(
            [
                "2000-01-01",
                "2000-01-02",
                "2000-01-03",
            ],
            freq=None,
        )
        tm.assert_datetime_array_equal(result, expected)

    @pytest.mark.parametrize("func", ["cumsum", "cumprod"])
    def test_accumulators_disallowed(self, func):
        # GH#50297
        arr = DatetimeArray._from_sequence_not_strict(
            [
                "2000-01-01",
                "2000-01-02",
            ],
            freq="D",
        )
        with pytest.raises(TypeError, match=f"Accumulation {func}"):
            arr._accumulate(func)
