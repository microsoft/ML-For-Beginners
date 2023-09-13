import numpy as np
import pytest

from pandas._libs import iNaT

from pandas.core.dtypes.dtypes import DatetimeTZDtype

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
from pandas.core.arrays.datetimes import _sequence_to_dt64ns


class TestDatetimeArrayConstructor:
    def test_from_sequence_invalid_type(self):
        mi = pd.MultiIndex.from_product([np.arange(5), np.arange(5)])
        with pytest.raises(TypeError, match="Cannot create a DatetimeArray"):
            DatetimeArray._from_sequence(mi)

    def test_only_1dim_accepted(self):
        arr = np.array([0, 1, 2, 3], dtype="M8[h]").astype("M8[ns]")

        with pytest.raises(ValueError, match="Only 1-dimensional"):
            # 3-dim, we allow 2D to sneak in for ops purposes GH#29853
            DatetimeArray(arr.reshape(2, 2, 1))

        with pytest.raises(ValueError, match="Only 1-dimensional"):
            # 0-dim
            DatetimeArray(arr[[0]].squeeze())

    def test_freq_validation(self):
        # GH#24623 check that invalid instances cannot be created with the
        #  public constructor
        arr = np.arange(5, dtype=np.int64) * 3600 * 10**9

        msg = (
            "Inferred frequency H from passed values does not "
            "conform to passed frequency W-SUN"
        )
        with pytest.raises(ValueError, match=msg):
            DatetimeArray(arr, freq="W")

    @pytest.mark.parametrize(
        "meth",
        [
            DatetimeArray._from_sequence,
            _sequence_to_dt64ns,
            pd.to_datetime,
            pd.DatetimeIndex,
        ],
    )
    def test_mixing_naive_tzaware_raises(self, meth):
        # GH#24569
        arr = np.array([pd.Timestamp("2000"), pd.Timestamp("2000", tz="CET")])

        msg = (
            "Cannot mix tz-aware with tz-naive values|"
            "Tz-aware datetime.datetime cannot be converted "
            "to datetime64 unless utc=True"
        )

        for obj in [arr, arr[::-1]]:
            # check that we raise regardless of whether naive is found
            #  before aware or vice-versa
            with pytest.raises(ValueError, match=msg):
                meth(obj)

    def test_from_pandas_array(self):
        arr = pd.array(np.arange(5, dtype=np.int64)) * 3600 * 10**9

        result = DatetimeArray._from_sequence(arr)._with_freq("infer")

        expected = pd.date_range("1970-01-01", periods=5, freq="H")._data
        tm.assert_datetime_array_equal(result, expected)

    def test_mismatched_timezone_raises(self):
        arr = DatetimeArray(
            np.array(["2000-01-01T06:00:00"], dtype="M8[ns]"),
            dtype=DatetimeTZDtype(tz="US/Central"),
        )
        dtype = DatetimeTZDtype(tz="US/Eastern")
        msg = r"dtype=datetime64\[ns.*\] does not match data dtype datetime64\[ns.*\]"
        with pytest.raises(TypeError, match=msg):
            DatetimeArray(arr, dtype=dtype)

        # also with mismatched tzawareness
        with pytest.raises(TypeError, match=msg):
            DatetimeArray(arr, dtype=np.dtype("M8[ns]"))
        with pytest.raises(TypeError, match=msg):
            DatetimeArray(arr.tz_localize(None), dtype=arr.dtype)

    def test_non_array_raises(self):
        with pytest.raises(ValueError, match="list"):
            DatetimeArray([1, 2, 3])

    def test_bool_dtype_raises(self):
        arr = np.array([1, 2, 3], dtype="bool")

        msg = "Unexpected value for 'dtype': 'bool'. Must be"
        with pytest.raises(ValueError, match=msg):
            DatetimeArray(arr)

        msg = r"dtype bool cannot be converted to datetime64\[ns\]"
        with pytest.raises(TypeError, match=msg):
            DatetimeArray._from_sequence(arr)

        with pytest.raises(TypeError, match=msg):
            _sequence_to_dt64ns(arr)

        with pytest.raises(TypeError, match=msg):
            pd.DatetimeIndex(arr)

        with pytest.raises(TypeError, match=msg):
            pd.to_datetime(arr)

    def test_incorrect_dtype_raises(self):
        with pytest.raises(ValueError, match="Unexpected value for 'dtype'."):
            DatetimeArray(np.array([1, 2, 3], dtype="i8"), dtype="category")

    def test_freq_infer_raises(self):
        with pytest.raises(ValueError, match="Frequency inference"):
            DatetimeArray(np.array([1, 2, 3], dtype="i8"), freq="infer")

    def test_copy(self):
        data = np.array([1, 2, 3], dtype="M8[ns]")
        arr = DatetimeArray(data, copy=False)
        assert arr._ndarray is data

        arr = DatetimeArray(data, copy=True)
        assert arr._ndarray is not data

    @pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
    def test_numpy_datetime_unit(self, unit):
        data = np.array([1, 2, 3], dtype=f"M8[{unit}]")
        arr = DatetimeArray(data)
        assert arr.unit == unit
        assert arr[0].unit == unit


class TestSequenceToDT64NS:
    def test_tz_dtype_mismatch_raises(self):
        arr = DatetimeArray._from_sequence(
            ["2000"], dtype=DatetimeTZDtype(tz="US/Central")
        )
        with pytest.raises(TypeError, match="data is already tz-aware"):
            DatetimeArray._from_sequence_not_strict(
                arr, dtype=DatetimeTZDtype(tz="UTC")
            )

    def test_tz_dtype_matches(self):
        dtype = DatetimeTZDtype(tz="US/Central")
        arr = DatetimeArray._from_sequence(["2000"], dtype=dtype)
        result = DatetimeArray._from_sequence_not_strict(arr, dtype=dtype)
        tm.assert_equal(arr, result)

    @pytest.mark.parametrize("order", ["F", "C"])
    def test_2d(self, order):
        dti = pd.date_range("2016-01-01", periods=6, tz="US/Pacific")
        arr = np.array(dti, dtype=object).reshape(3, 2)
        if order == "F":
            arr = arr.T

        res = _sequence_to_dt64ns(arr)
        expected = _sequence_to_dt64ns(arr.ravel())

        tm.assert_numpy_array_equal(res[0].ravel(), expected[0])
        assert res[1] == expected[1]
        assert res[2] == expected[2]

        res = DatetimeArray._from_sequence(arr)
        expected = DatetimeArray._from_sequence(arr.ravel()).reshape(arr.shape)
        tm.assert_datetime_array_equal(res, expected)


# ----------------------------------------------------------------------------
# Arrow interaction


EXTREME_VALUES = [0, 123456789, None, iNaT, 2**63 - 1, -(2**63) + 1]
FINE_TO_COARSE_SAFE = [123_000_000_000, None, -123_000_000_000]
COARSE_TO_FINE_SAFE = [123, None, -123]


@pytest.mark.parametrize(
    ("pa_unit", "pd_unit", "pa_tz", "pd_tz", "data"),
    [
        ("s", "s", "UTC", "UTC", EXTREME_VALUES),
        ("ms", "ms", "UTC", "Europe/Berlin", EXTREME_VALUES),
        ("us", "us", "US/Eastern", "UTC", EXTREME_VALUES),
        ("ns", "ns", "US/Central", "Asia/Kolkata", EXTREME_VALUES),
        ("ns", "s", "UTC", "UTC", FINE_TO_COARSE_SAFE),
        ("us", "ms", "UTC", "Europe/Berlin", FINE_TO_COARSE_SAFE),
        ("ms", "us", "US/Eastern", "UTC", COARSE_TO_FINE_SAFE),
        ("s", "ns", "US/Central", "Asia/Kolkata", COARSE_TO_FINE_SAFE),
    ],
)
def test_from_arrowtest_from_arrow_with_different_units_and_timezones_with_(
    pa_unit, pd_unit, pa_tz, pd_tz, data
):
    pa = pytest.importorskip("pyarrow")

    pa_type = pa.timestamp(pa_unit, tz=pa_tz)
    arr = pa.array(data, type=pa_type)
    dtype = DatetimeTZDtype(unit=pd_unit, tz=pd_tz)

    result = dtype.__from_arrow__(arr)
    expected = DatetimeArray(
        np.array(data, dtype=f"datetime64[{pa_unit}]").astype(f"datetime64[{pd_unit}]"),
        dtype=dtype,
    )
    tm.assert_extension_array_equal(result, expected)

    result = dtype.__from_arrow__(pa.chunked_array([arr]))
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize(
    ("unit", "tz"),
    [
        ("s", "UTC"),
        ("ms", "Europe/Berlin"),
        ("us", "US/Eastern"),
        ("ns", "Asia/Kolkata"),
        ("ns", "UTC"),
    ],
)
def test_from_arrow_from_empty(unit, tz):
    pa = pytest.importorskip("pyarrow")

    data = []
    arr = pa.array(data)
    dtype = DatetimeTZDtype(unit=unit, tz=tz)

    result = dtype.__from_arrow__(arr)
    expected = DatetimeArray(np.array(data, dtype=f"datetime64[{unit}]"))
    expected = expected.tz_localize(tz=tz)
    tm.assert_extension_array_equal(result, expected)

    result = dtype.__from_arrow__(pa.chunked_array([arr]))
    tm.assert_extension_array_equal(result, expected)


def test_from_arrow_from_integers():
    pa = pytest.importorskip("pyarrow")

    data = [0, 123456789, None, 2**63 - 1, iNaT, -123456789]
    arr = pa.array(data)
    dtype = DatetimeTZDtype(unit="ns", tz="UTC")

    result = dtype.__from_arrow__(arr)
    expected = DatetimeArray(np.array(data, dtype="datetime64[ns]"))
    expected = expected.tz_localize("UTC")
    tm.assert_extension_array_equal(result, expected)

    result = dtype.__from_arrow__(pa.chunked_array([arr]))
    tm.assert_extension_array_equal(result, expected)
