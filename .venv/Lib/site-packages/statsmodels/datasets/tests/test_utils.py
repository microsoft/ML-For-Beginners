import os
from ssl import SSLError
from socket import timeout
from urllib.error import HTTPError, URLError

import numpy as np
from numpy.testing import assert_, assert_array_equal
import pytest

from statsmodels.datasets import get_rdataset, webuse, check_internet, utils

cur_dir = os.path.dirname(os.path.abspath(__file__))

IGNORED_EXCEPTIONS = (HTTPError, URLError, SSLError,  UnicodeEncodeError,
                      timeout)


@pytest.mark.smoke
def test_get_rdataset():
    test_url = "https://raw.githubusercontent.com/vincentarelbundock/" \
               "Rdatasets/master/csv/datasets/cars.csv"
    internet_available = check_internet(test_url)
    if not internet_available:  # pragma: no cover
        pytest.skip('Unable to retrieve file - skipping test')
    try:
        duncan = get_rdataset("Duncan", "carData", cache=cur_dir)
    except IGNORED_EXCEPTIONS:
        pytest.skip('Failed with HTTPError or URLError, these are random')
    assert_(isinstance(duncan, utils.Dataset))
    duncan = get_rdataset("Duncan", "carData", cache=cur_dir)
    assert_(duncan.from_cache)


@pytest.mark.smoke
def test_get_rdataset_write_read_cache():
    # test writing and reading cache
    try:
        guerry = get_rdataset("Guerry", "HistData", cache=cur_dir)
    except IGNORED_EXCEPTIONS:
        pytest.skip('Failed with HTTPError or URLError, these are random')

    assert_(guerry.from_cache is False)
    guerry2 = get_rdataset("Guerry", "HistData", cache=cur_dir)
    assert_(guerry2.from_cache is True)
    fn = "raw.githubusercontent.com,vincentarelbundock,Rdatasets,master,csv," \
         "HistData,Guerry-v2.csv.zip"
    os.remove(os.path.join(cur_dir, fn))
    fn = "raw.githubusercontent.com,vincentarelbundock,Rdatasets,master,doc," \
         "HistData,rst,Guerry-v2.rst.zip"
    os.remove(os.path.join(cur_dir, fn))


def test_webuse():
    # test copied and adjusted from iolib/tests/test_foreign
    from statsmodels.iolib.tests.results.macrodata import macrodata_result
    res2 = np.array([list(row) for row in macrodata_result])
    base_gh = "https://github.com/statsmodels/statsmodels/raw/main/" \
              "statsmodels/datasets/macrodata/"
    internet_available = check_internet(base_gh)
    if not internet_available:  # pragma: no cover
        pytest.skip('Unable to retrieve file - skipping test')
    try:
        res1 = webuse('macrodata', baseurl=base_gh, as_df=False)
    except IGNORED_EXCEPTIONS:
        pytest.skip('Failed with HTTPError or URLError, these are random')
    assert_array_equal(res1, res2)


def test_webuse_pandas():
    # test copied and adjusted from iolib/tests/test_foreign
    from statsmodels.compat.pandas import assert_frame_equal
    from statsmodels.datasets import macrodata
    dta = macrodata.load_pandas().data
    base_gh = "https://github.com/statsmodels/statsmodels/raw/main/" \
              "statsmodels/datasets/macrodata/"
    internet_available = check_internet(base_gh)
    if not internet_available:  # pragma: no cover
        pytest.skip('Unable to retrieve file - skipping test')
    try:
        res1 = webuse('macrodata', baseurl=base_gh)
    except IGNORED_EXCEPTIONS:
        pytest.skip('Failed with HTTP Error, these are random')
    res1 = res1.astype(float)
    assert_frame_equal(res1, dta.astype(float))
