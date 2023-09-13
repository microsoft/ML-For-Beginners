import numpy as np
import pytest

from pandas import (
    Period,
    PeriodIndex,
    period_range,
)
import pandas._testing as tm


class TestPeriodRepresentation:
    """
    Wish to match NumPy units
    """

    @pytest.mark.parametrize(
        "freq, base_date",
        [
            ("W-THU", "1970-01-01"),
            ("D", "1970-01-01"),
            ("B", "1970-01-01"),
            ("H", "1970-01-01"),
            ("T", "1970-01-01"),
            ("S", "1970-01-01"),
            ("L", "1970-01-01"),
            ("U", "1970-01-01"),
            ("N", "1970-01-01"),
            ("M", "1970-01"),
            ("A", 1970),
        ],
    )
    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    @pytest.mark.filterwarnings(
        "ignore:Period with BDay freq is deprecated:FutureWarning"
    )
    def test_freq(self, freq, base_date):
        rng = period_range(start=base_date, periods=10, freq=freq)
        exp = np.arange(10, dtype=np.int64)

        tm.assert_numpy_array_equal(rng.asi8, exp)


class TestPeriodIndexConversion:
    def test_tolist(self):
        index = period_range(freq="A", start="1/1/2001", end="12/1/2009")
        rs = index.tolist()
        for x in rs:
            assert isinstance(x, Period)

        recon = PeriodIndex(rs)
        tm.assert_index_equal(index, recon)
