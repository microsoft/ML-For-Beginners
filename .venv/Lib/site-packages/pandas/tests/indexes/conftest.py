import numpy as np
import pytest

from pandas import (
    Series,
    array,
)
import pandas._testing as tm


@pytest.fixture(params=[None, False])
def sort(request):
    """
    Valid values for the 'sort' parameter used in the Index
    setops methods (intersection, union, etc.)

    Caution:
        Don't confuse this one with the "sort" fixture used
        for DataFrame.append or concat. That one has
        parameters [True, False].

        We can't combine them as sort=True is not permitted
        in the Index setops methods.
    """
    return request.param


@pytest.fixture(params=["D", "3D", "-3D", "H", "2H", "-2H", "T", "2T", "S", "-3S"])
def freq_sample(request):
    """
    Valid values for 'freq' parameter used to create date_range and
    timedelta_range..
    """
    return request.param


@pytest.fixture(params=[list, tuple, np.array, array, Series])
def listlike_box(request):
    """
    Types that may be passed as the indexer to searchsorted.
    """
    return request.param


@pytest.fixture(
    params=tm.ALL_REAL_NUMPY_DTYPES
    + [
        "object",
        "category",
        "datetime64[ns]",
        "timedelta64[ns]",
    ]
)
def any_dtype_for_small_pos_integer_indexes(request):
    """
    Dtypes that can be given to an Index with small positive integers.

    This means that for any dtype `x` in the params list, `Index([1, 2, 3], dtype=x)` is
    valid and gives the correct Index (sub-)class.
    """
    return request.param
