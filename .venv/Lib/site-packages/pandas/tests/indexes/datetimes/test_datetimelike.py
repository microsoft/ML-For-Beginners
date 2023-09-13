""" generic tests from the Datetimelike class """
from pandas import date_range


class TestDatetimeIndex:
    def test_format(self):
        # GH35439
        idx = date_range("20130101", periods=5)
        expected = [f"{x:%Y-%m-%d}" for x in idx]
        assert idx.format() == expected
