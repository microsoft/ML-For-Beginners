# Authors: Gael Varoquaux <gael.varoquaux@normalesup.org>
#          Justin Vincent
#          Lars Buitinck
# License: BSD 3 clause

import numpy as np
import pytest

from sklearn.utils._testing import assert_array_equal
from sklearn.utils.fixes import (
    _object_dtype_isnan,
    _smallest_admissible_index_dtype,
    delayed,
)


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


@pytest.mark.parametrize(
    "params, expected_dtype",
    [
        ({}, np.int32),  # default behaviour
        ({"maxval": np.iinfo(np.int32).max}, np.int32),
        ({"maxval": np.iinfo(np.int32).max + 1}, np.int64),
    ],
)
def test_smallest_admissible_index_dtype_max_val(params, expected_dtype):
    """Check the behaviour of `smallest_admissible_index_dtype` depending only on the
    `max_val` parameter.
    """
    assert _smallest_admissible_index_dtype(**params) == expected_dtype


@pytest.mark.parametrize(
    "params, expected_dtype",
    [
        # Arrays dtype is int64 and thus should not be downcasted to int32 without
        # checking the content of providing maxval.
        ({"arrays": np.array([1, 2], dtype=np.int64)}, np.int64),
        # One of the array is int64 and should not be downcasted to int32
        # for the same reasons.
        (
            {
                "arrays": (
                    np.array([1, 2], dtype=np.int32),
                    np.array([1, 2], dtype=np.int64),
                )
            },
            np.int64,
        ),
        # Both arrays are already int32: we can just keep this dtype.
        (
            {
                "arrays": (
                    np.array([1, 2], dtype=np.int32),
                    np.array([1, 2], dtype=np.int32),
                )
            },
            np.int32,
        ),
        # Arrays should be upcasted to at least int32 precision.
        ({"arrays": np.array([1, 2], dtype=np.int8)}, np.int32),
        # Check that `maxval` takes precedence over the arrays and thus upcast to
        # int64.
        (
            {
                "arrays": np.array([1, 2], dtype=np.int32),
                "maxval": np.iinfo(np.int32).max + 1,
            },
            np.int64,
        ),
    ],
)
def test_smallest_admissible_index_dtype_without_checking_contents(
    params, expected_dtype
):
    """Check the behaviour of `smallest_admissible_index_dtype` using the passed
    arrays but without checking the contents of the arrays.
    """
    assert _smallest_admissible_index_dtype(**params) == expected_dtype


@pytest.mark.parametrize(
    "params, expected_dtype",
    [
        # empty arrays should always be converted to int32 indices
        (
            {
                "arrays": (np.array([], dtype=np.int64), np.array([], dtype=np.int64)),
                "check_contents": True,
            },
            np.int32,
        ),
        # arrays respecting np.iinfo(np.int32).min < x < np.iinfo(np.int32).max should
        # be converted to int32,
        (
            {"arrays": np.array([1], dtype=np.int64), "check_contents": True},
            np.int32,
        ),
        # otherwise, it should be converted to int64. We need to create a uint32
        # arrays to accommodate a value > np.iinfo(np.int32).max
        (
            {
                "arrays": np.array([np.iinfo(np.int32).max + 1], dtype=np.uint32),
                "check_contents": True,
            },
            np.int64,
        ),
        # maxval should take precedence over the arrays contents and thus upcast to
        # int64.
        (
            {
                "arrays": np.array([1], dtype=np.int32),
                "check_contents": True,
                "maxval": np.iinfo(np.int32).max + 1,
            },
            np.int64,
        ),
        # when maxval is small, but check_contents is True and the contents
        # require np.int64, we still require np.int64 indexing in the end.
        (
            {
                "arrays": np.array([np.iinfo(np.int32).max + 1], dtype=np.uint32),
                "check_contents": True,
                "maxval": 1,
            },
            np.int64,
        ),
    ],
)
def test_smallest_admissible_index_dtype_by_checking_contents(params, expected_dtype):
    """Check the behaviour of `smallest_admissible_index_dtype` using the dtype of the
    arrays but as well the contents.
    """
    assert _smallest_admissible_index_dtype(**params) == expected_dtype


@pytest.mark.parametrize(
    "params, err_type, err_msg",
    [
        (
            {"maxval": np.iinfo(np.int64).max + 1},
            ValueError,
            "is to large to be represented as np.int64",
        ),
        (
            {"arrays": np.array([1, 2], dtype=np.float64)},
            ValueError,
            "Array dtype float64 is not supported",
        ),
        ({"arrays": [1, 2]}, TypeError, "Arrays should be of type np.ndarray"),
    ],
)
def test_smallest_admissible_index_dtype_error(params, err_type, err_msg):
    """Check that we raise the proper error message."""
    with pytest.raises(err_type, match=err_msg):
        _smallest_admissible_index_dtype(**params)
