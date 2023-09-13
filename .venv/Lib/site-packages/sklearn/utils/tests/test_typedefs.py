import numpy as np
import pytest

from sklearn.utils._typedefs import testing_make_array_from_typed_val


@pytest.mark.parametrize(
    "type_t, value, expected_dtype",
    [
        ("uint8_t", 1, np.uint8),
        ("intp_t", 1, np.intp),
        ("float64_t", 1.0, np.float64),
        ("float32_t", 1.0, np.float32),
        ("int32_t", 1, np.int32),
        ("int64_t", 1, np.int64),
    ],
)
def test_types(type_t, value, expected_dtype):
    """Check that the types defined in _typedefs correspond to the expected
    numpy dtypes.
    """
    assert testing_make_array_from_typed_val[type_t](value).dtype == expected_dtype
