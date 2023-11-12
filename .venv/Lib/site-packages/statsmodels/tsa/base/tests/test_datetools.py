from datetime import datetime

import numpy.testing as npt

from statsmodels.tsa.base.datetools import date_parser, dates_from_range


def test_regex_matching_month():
    t1 = "1999m4"
    t2 = "1999:m4"
    t3 = "1999:mIV"
    t4 = "1999mIV"
    result = datetime(1999, 4, 30)
    npt.assert_equal(date_parser(t1), result)
    npt.assert_equal(date_parser(t2), result)
    npt.assert_equal(date_parser(t3), result)
    npt.assert_equal(date_parser(t4), result)


def test_regex_matching_quarter():
    t1 = "1999q4"
    t2 = "1999:q4"
    t3 = "1999:qIV"
    t4 = "1999qIV"
    result = datetime(1999, 12, 31)
    npt.assert_equal(date_parser(t1), result)
    npt.assert_equal(date_parser(t2), result)
    npt.assert_equal(date_parser(t3), result)
    npt.assert_equal(date_parser(t4), result)


def test_dates_from_range():
    results = [datetime(1959, 3, 31, 0, 0),
               datetime(1959, 6, 30, 0, 0),
               datetime(1959, 9, 30, 0, 0),
               datetime(1959, 12, 31, 0, 0),
               datetime(1960, 3, 31, 0, 0),
               datetime(1960, 6, 30, 0, 0),
               datetime(1960, 9, 30, 0, 0),
               datetime(1960, 12, 31, 0, 0),
               datetime(1961, 3, 31, 0, 0),
               datetime(1961, 6, 30, 0, 0),
               datetime(1961, 9, 30, 0, 0),
               datetime(1961, 12, 31, 0, 0),
               datetime(1962, 3, 31, 0, 0),
               datetime(1962, 6, 30, 0, 0)]
    dt_range = dates_from_range('1959q1', '1962q2')
    npt.assert_(results == dt_range)

    # test with starting period not the first with length
    results = results[2:]
    dt_range = dates_from_range('1959q3', length=len(results))
    npt.assert_(results == dt_range)

    # check month
    results = [datetime(1959, 3, 31, 0, 0),
               datetime(1959, 4, 30, 0, 0),
               datetime(1959, 5, 31, 0, 0),
               datetime(1959, 6, 30, 0, 0),
               datetime(1959, 7, 31, 0, 0),
               datetime(1959, 8, 31, 0, 0),
               datetime(1959, 9, 30, 0, 0),
               datetime(1959, 10, 31, 0, 0),
               datetime(1959, 11, 30, 0, 0),
               datetime(1959, 12, 31, 0, 0),
               datetime(1960, 1, 31, 0, 0),
               datetime(1960, 2, 28, 0, 0),
               datetime(1960, 3, 31, 0, 0),
               datetime(1960, 4, 30, 0, 0),
               datetime(1960, 5, 31, 0, 0),
               datetime(1960, 6, 30, 0, 0),
               datetime(1960, 7, 31, 0, 0),
               datetime(1960, 8, 31, 0, 0),
               datetime(1960, 9, 30, 0, 0),
               datetime(1960, 10, 31, 0, 0),
               datetime(1960, 12, 31, 0, 0),
               datetime(1961, 1, 31, 0, 0),
               datetime(1961, 2, 28, 0, 0),
               datetime(1961, 3, 31, 0, 0),
               datetime(1961, 4, 30, 0, 0),
               datetime(1961, 5, 31, 0, 0),
               datetime(1961, 6, 30, 0, 0),
               datetime(1961, 7, 31, 0, 0),
               datetime(1961, 8, 31, 0, 0),
               datetime(1961, 9, 30, 0, 0),
               datetime(1961, 10, 31, 0, 0)]

    dt_range = dates_from_range("1959m3", length=len(results))
