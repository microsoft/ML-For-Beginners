import pytest

from pandas import Categorical


@pytest.fixture(params=[True, False])
def allow_fill(request):
    """Boolean 'allow_fill' parameter for Categorical.take"""
    return request.param


@pytest.fixture
def factor():
    """Fixture returning  a Categorical object"""
    return Categorical(["a", "b", "b", "a", "a", "c", "c", "c"], ordered=True)
