# Authors: Gael Varoquaux <gael.varoquaux@normalesup.org>
#          Justin Vincent
#          Lars Buitinck
# License: BSD 3 clause

import numpy as np
import pytest

from sklearn.utils._testing import assert_array_equal
from sklearn.utils.fixes import _object_dtype_isnan, delayed


@pytest.mark.parametrize("dtype, val", ([object, 1], [object, "a"], [float, 1]))
def test_object_dtype_isnan(dtype, val):
    X = np.array([[val, np.nan], [np.nan, val]], dtype=dtype)

    expected_mask = np.array([[False, True], [True, False]])

    mask = _object_dtype_isnan(X)

    assert_array_equal(mask, expected_mask)


def test_delayed_deprecation():
    """Check that we issue the FutureWarning regarding the deprecation of delayed."""

    def func(x):
        return x

    warn_msg = "The function `delayed` has been moved from `sklearn.utils.fixes`"
    with pytest.warns(FutureWarning, match=warn_msg):
        delayed(func)
