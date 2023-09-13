import datetime

import pytest

from pandas._libs.tslibs import Timestamp
from pandas._libs.tslibs.offsets import MonthOffset

from pandas.tseries import offsets


@pytest.fixture(
    params=[
        getattr(offsets, o) for o in offsets.__all__ if o not in ("Tick", "BaseOffset")
    ]
)
def offset_types(request):
    """
    Fixture for all the datetime offsets available for a time series.
    """
    return request.param


@pytest.fixture(
    params=[
        getattr(offsets, o)
        for o in offsets.__all__
        if issubclass(getattr(offsets, o), MonthOffset) and o != "MonthOffset"
    ]
)
def month_classes(request):
    """
    Fixture for month based datetime offsets available for a time series.
    """
    return request.param


@pytest.fixture
def dt():
    """
    Fixture for common Timestamp.
    """
    return Timestamp(datetime.datetime(2008, 1, 2))
