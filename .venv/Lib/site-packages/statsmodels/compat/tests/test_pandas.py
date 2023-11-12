from statsmodels.compat.pandas import PD_LT_1_4, is_float_index, is_int_index

import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize("int_type", ["u", "i"])
@pytest.mark.parametrize("int_size", [1, 2, 4, 8])
def test_is_int_index(int_type, int_size):
    index = pd.Index(np.arange(100), dtype=f"{int_type}{int_size}")
    assert is_int_index(index)
    assert not is_float_index(index)


@pytest.mark.parametrize("float_size", [4, 8])
def test_is_float_index(float_size):
    index = pd.Index(np.arange(100.0), dtype=f"f{float_size}")
    assert is_float_index(index)
    assert not is_int_index(index)


@pytest.mark.skipif(not PD_LT_1_4, reason="Requires U/Int64Index")
def test_legacy_int_index():
    from pandas import Int64Index, UInt64Index

    index = Int64Index(np.arange(100))
    assert is_int_index(index)
    assert not is_float_index(index)

    index = UInt64Index(np.arange(100))
    assert is_int_index(index)
    assert not is_float_index(index)


@pytest.mark.skipif(not PD_LT_1_4, reason="Requires Float64Index")
def test_legacy_float_index():
    from pandas import Float64Index

    index = Float64Index(np.arange(100))
    assert not is_int_index(index)
    assert is_float_index(index)
