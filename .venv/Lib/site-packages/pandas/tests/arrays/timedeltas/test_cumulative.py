import pytest

import pandas._testing as tm
from pandas.core.arrays import TimedeltaArray


class TestAccumulator:
    def test_accumulators_disallowed(self):
        # GH#50297
        arr = TimedeltaArray._from_sequence_not_strict(["1D", "2D"])
        with pytest.raises(TypeError, match="cumprod not supported"):
            arr._accumulate("cumprod")

    def test_cumsum(self):
        # GH#50297
        arr = TimedeltaArray._from_sequence_not_strict(["1D", "2D"])
        result = arr._accumulate("cumsum")
        expected = TimedeltaArray._from_sequence_not_strict(["1D", "3D"])
        tm.assert_timedelta_array_equal(result, expected)
