import pytest

from pandas import (
    Index,
    NaT,
    Series,
)
import pandas._testing as tm


def test_astype_str_from_bytes():
    # https://github.com/pandas-dev/pandas/issues/38607
    # GH#49658 pre-2.0 Index called .values.astype(str) here, which effectively
    #  did a .decode() on the bytes object.  In 2.0 we go through
    #  ensure_string_array which does f"{val}"
    idx = Index(["あ", b"a"], dtype="object")
    result = idx.astype(str)
    expected = Index(["あ", "a"], dtype="object")
    tm.assert_index_equal(result, expected)

    # while we're here, check that Series.astype behaves the same
    result = Series(idx).astype(str)
    expected = Series(expected, dtype=object)
    tm.assert_series_equal(result, expected)


def test_astype_invalid_nas_to_tdt64_raises():
    # GH#45722 don't cast np.datetime64 NaTs to timedelta64 NaT
    idx = Index([NaT.asm8] * 2, dtype=object)

    msg = r"Invalid type for timedelta scalar: <class 'numpy.datetime64'>"
    with pytest.raises(TypeError, match=msg):
        idx.astype("m8[ns]")
