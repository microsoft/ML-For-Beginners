import numpy as np
import pytest

from sklearn.utils._testing import assert_allclose
from sklearn.utils.arrayfuncs import _all_with_any_reduction_axis_1, min_pos


def test_min_pos():
    # Check that min_pos returns a positive value and that it's consistent
    # between float and double
    X = np.random.RandomState(0).randn(100)

    min_double = min_pos(X)
    min_float = min_pos(X.astype(np.float32))

    assert_allclose(min_double, min_float)
    assert min_double >= 0


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_min_pos_no_positive(dtype):
    # Check that the return value of min_pos is the maximum representable
    # value of the input dtype when all input elements are <= 0 (#19328)
    X = np.full(100, -1.0).astype(dtype, copy=False)

    assert min_pos(X) == np.finfo(dtype).max


@pytest.mark.parametrize("dtype", [np.int16, np.int32, np.float32, np.float64])
@pytest.mark.parametrize("value", [0, 1.5, -1])
def test_all_with_any_reduction_axis_1(dtype, value):
    # Check that return value is False when there is no row equal to `value`
    X = np.arange(12, dtype=dtype).reshape(3, 4)
    assert not _all_with_any_reduction_axis_1(X, value=value)

    # Make a row equal to `value`
    X[1, :] = value
    assert _all_with_any_reduction_axis_1(X, value=value)
