from statsmodels.compat.pandas import PD_LT_2_2_0

from datetime import datetime

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tools.testing import assert_equal
from statsmodels.tsa.base.tsa_model import TimeSeriesModel

YE_APR = "A-APR" if PD_LT_2_2_0 else "YE-APR"

def test_pandas_nodates_index():

    data = [988, 819, 964]
    dates = ['a', 'b', 'c']
    s = pd.Series(data, index=dates)

    # TODO: Remove this, this is now valid
    # npt.assert_raises(ValueError, TimeSeriesModel, s)

    # Test with a non-date index that does not raise an exception because it
    # can be coerced into a nanosecond DatetimeIndex
    data = [988, 819, 964]
    # index=pd.date_range('1970-01-01', periods=3, freq='QS')
    index = pd.to_datetime([100, 101, 102])
    s = pd.Series(data, index=index)

    actual_str = (index[0].strftime('%Y-%m-%d %H:%M:%S.%f') +
                  str(index[0].value))
    assert_equal(actual_str, '1970-01-01 00:00:00.000000100')

    with pytest.warns(ValueWarning, match="No frequency information"):
        mod = TimeSeriesModel(s)

    start, end, out_of_sample, _ = mod._get_prediction_index(0, 4)
    assert_equal(len(mod.data.predict_dates), 5)


def test_predict_freq():
    # test that predicted dates have same frequency
    x = np.arange(1,36.)

    # there's a bug in pandas up to 0.10.2 for YearBegin
    #dates = date_range("1972-4-1", "2007-4-1", freq="AS-APR")

    dates = pd.date_range("1972-4-30", "2006-4-30", freq=YE_APR)
    series = pd.Series(x, index=dates)
    model = TimeSeriesModel(series)
    #npt.assert_(model.data.freq == "AS-APR")
    # two possabilities due to future changes in pandas 2.2+
    assert model._index.freqstr in ("Y-APR", "A-APR", "YE-APR")

    start, end, out_of_sample, _ = (
        model._get_prediction_index("2006-4-30", "2016-4-30"))

    predict_dates = model.data.predict_dates

    #expected_dates = date_range("2006-12-31", "2016-12-31",
    #                            freq="AS-APR")
    expected_dates = pd.date_range("2006-4-30", "2016-4-30", freq=YE_APR)
    assert_equal(predict_dates, expected_dates)
    #ptesting.assert_series_equal(predict_dates, expected_dates)


def test_keyerror_start_date():
    x = np.arange(1,36.)

    # dates = date_range("1972-4-1", "2007-4-1", freq="AS-APR")
    dates = pd.date_range("1972-4-30", "2006-4-30", freq=YE_APR)
    series = pd.Series(x, index=dates)
    model = TimeSeriesModel(series)

    npt.assert_raises(KeyError, model._get_prediction_index, "1970-4-30", None)


def test_period_index():
    # test 1285

    dates = pd.period_range(start="1/1/1990", periods=20, freq="M")
    x = np.arange(1, 21.)

    model = TimeSeriesModel(pd.Series(x, index=dates))
    assert_equal(model._index.freqstr, "M")
    model = TimeSeriesModel(pd.Series(x, index=dates))
    npt.assert_(model.data.freq == "M")


def test_pandas_dates():

    data = [988, 819, 964]
    dates = ['2016-01-01 12:00:00', '2016-02-01 12:00:00', '2016-03-01 12:00:00']

    datetime_dates = pd.to_datetime(dates)

    result = pd.Series(data=data, index=datetime_dates, name='price')
    df = pd.DataFrame(data={'price': data}, index=pd.DatetimeIndex(dates, freq='MS'))

    model = TimeSeriesModel(df['price'])

    assert_equal(model.data.dates, result.index)


def test_get_predict_start_end():
    index = pd.date_range(start='1970-01-01', end='1990-01-01', freq='YS')
    endog = pd.Series(np.zeros(10), index[:10])
    model = TimeSeriesModel(endog)

    predict_starts = [1, '1971-01-01', datetime(1971, 1, 1), index[1]]
    predict_ends = [20, '1990-01-01', datetime(1990, 1, 1), index[-1]]

    desired = (1, 9, 11)
    for start in predict_starts:
        for end in predict_ends:
            assert_equal(model._get_prediction_index(start, end)[:3], desired)
