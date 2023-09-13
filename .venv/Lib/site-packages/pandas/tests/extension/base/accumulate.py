import pytest

import pandas as pd
import pandas._testing as tm


class BaseAccumulateTests:
    """
    Accumulation specific tests. Generally these only
    make sense for numeric/boolean operations.
    """

    def _supports_accumulation(self, ser: pd.Series, op_name: str) -> bool:
        # Do we expect this accumulation to be supported for this dtype?
        # We default to assuming "no"; subclass authors should override here.
        return False

    def check_accumulate(self, ser: pd.Series, op_name: str, skipna: bool):
        alt = ser.astype("float64")
        result = getattr(ser, op_name)(skipna=skipna)

        if result.dtype == pd.Float32Dtype() and op_name == "cumprod" and skipna:
            # TODO: avoid special-casing here
            pytest.skip(
                f"Float32 precision lead to large differences with op {op_name} "
                f"and skipna={skipna}"
            )

        expected = getattr(alt, op_name)(skipna=skipna)
        tm.assert_series_equal(result, expected, check_dtype=False)

    @pytest.mark.parametrize("skipna", [True, False])
    def test_accumulate_series(self, data, all_numeric_accumulations, skipna):
        op_name = all_numeric_accumulations
        ser = pd.Series(data)

        if self._supports_accumulation(ser, op_name):
            self.check_accumulate(ser, op_name, skipna)
        else:
            with pytest.raises(NotImplementedError):
                getattr(ser, op_name)(skipna=skipna)
