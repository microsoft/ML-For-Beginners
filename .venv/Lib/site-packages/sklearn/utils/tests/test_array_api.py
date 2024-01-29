from functools import partial

import numpy
import pytest
from numpy.testing import assert_allclose

from sklearn._config import config_context
from sklearn.base import BaseEstimator
from sklearn.utils._array_api import (
    _ArrayAPIWrapper,
    _asarray_with_order,
    _atol_for_type,
    _convert_to_numpy,
    _estimator_with_converted_arrays,
    _nanmax,
    _nanmin,
    _NumPyAPIWrapper,
    _weighted_sum,
    get_namespace,
    supported_float_dtypes,
    yield_namespace_device_dtype_combinations,
)
from sklearn.utils._testing import (
    _array_api_for_tests,
    skip_if_array_api_compat_not_configured,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:The numpy.array_api submodule:UserWarning"
)


@pytest.mark.parametrize("X", [numpy.asarray([1, 2, 3]), [1, 2, 3]])
def test_get_namespace_ndarray_default(X):
    """Check that get_namespace returns NumPy wrapper"""
    xp_out, is_array_api_compliant = get_namespace(X)
    assert isinstance(xp_out, _NumPyAPIWrapper)
    assert not is_array_api_compliant


def test_get_namespace_ndarray_creation_device():
    """Check expected behavior with device and creation functions."""
    X = numpy.asarray([1, 2, 3])
    xp_out, _ = get_namespace(X)

    full_array = xp_out.full(10, fill_value=2.0, device="cpu")
    assert_allclose(full_array, [2.0] * 10)

    with pytest.raises(ValueError, match="Unsupported device"):
        xp_out.zeros(10, device="cuda")


@skip_if_array_api_compat_not_configured
def test_get_namespace_ndarray_with_dispatch():
    """Test get_namespace on NumPy ndarrays."""
    array_api_compat = pytest.importorskip("array_api_compat")

    X_np = numpy.asarray([[1, 2, 3]])

    with config_context(array_api_dispatch=True):
        xp_out, is_array_api_compliant = get_namespace(X_np)
        assert is_array_api_compliant
        assert xp_out is array_api_compat.numpy


@skip_if_array_api_compat_not_configured
def test_get_namespace_array_api():
    """Test get_namespace for ArrayAPI arrays."""
    xp = pytest.importorskip("numpy.array_api")

    X_np = numpy.asarray([[1, 2, 3]])
    X_xp = xp.asarray(X_np)
    with config_context(array_api_dispatch=True):
        xp_out, is_array_api_compliant = get_namespace(X_xp)
        assert is_array_api_compliant
        assert isinstance(xp_out, _ArrayAPIWrapper)

        with pytest.raises(TypeError):
            xp_out, is_array_api_compliant = get_namespace(X_xp, X_np)


class _AdjustableNameAPITestWrapper(_ArrayAPIWrapper):
    """API wrapper that has an adjustable name. Used for testing."""

    def __init__(self, array_namespace, name):
        super().__init__(array_namespace=array_namespace)
        self.__name__ = name


def test_array_api_wrapper_astype():
    """Test _ArrayAPIWrapper for ArrayAPIs that is not NumPy."""
    numpy_array_api = pytest.importorskip("numpy.array_api")
    xp_ = _AdjustableNameAPITestWrapper(numpy_array_api, "wrapped_numpy.array_api")
    xp = _ArrayAPIWrapper(xp_)

    X = xp.asarray(([[1, 2, 3], [3, 4, 5]]), dtype=xp.float64)
    X_converted = xp.astype(X, xp.float32)
    assert X_converted.dtype == xp.float32

    X_converted = xp.asarray(X, dtype=xp.float32)
    assert X_converted.dtype == xp.float32


@pytest.mark.parametrize("array_api", ["numpy", "numpy.array_api"])
def test_asarray_with_order(array_api):
    """Test _asarray_with_order passes along order for NumPy arrays."""
    xp = pytest.importorskip(array_api)

    X = xp.asarray([1.2, 3.4, 5.1])
    X_new = _asarray_with_order(X, order="F", xp=xp)

    X_new_np = numpy.asarray(X_new)
    assert X_new_np.flags["F_CONTIGUOUS"]


def test_asarray_with_order_ignored():
    """Test _asarray_with_order ignores order for Generic ArrayAPI."""
    xp = pytest.importorskip("numpy.array_api")
    xp_ = _AdjustableNameAPITestWrapper(xp, "wrapped.array_api")

    X = numpy.asarray([[1.2, 3.4, 5.1], [3.4, 5.5, 1.2]], order="C")
    X = xp_.asarray(X)

    X_new = _asarray_with_order(X, order="F", xp=xp_)

    X_new_np = numpy.asarray(X_new)
    assert X_new_np.flags["C_CONTIGUOUS"]
    assert not X_new_np.flags["F_CONTIGUOUS"]


@pytest.mark.parametrize(
    "array_namespace, device, dtype_name", yield_namespace_device_dtype_combinations()
)
@pytest.mark.parametrize(
    "sample_weight, normalize, expected",
    [
        (None, False, 10.0),
        (None, True, 2.5),
        ([0.4, 0.4, 0.5, 0.7], False, 5.5),
        ([0.4, 0.4, 0.5, 0.7], True, 2.75),
        ([1, 2, 3, 4], False, 30.0),
        ([1, 2, 3, 4], True, 3.0),
    ],
)
def test_weighted_sum(
    array_namespace, device, dtype_name, sample_weight, normalize, expected
):
    xp = _array_api_for_tests(array_namespace, device)
    sample_score = numpy.asarray([1, 2, 3, 4], dtype=dtype_name)
    sample_score = xp.asarray(sample_score, device=device)
    if sample_weight is not None:
        sample_weight = numpy.asarray(sample_weight, dtype=dtype_name)
        sample_weight = xp.asarray(sample_weight, device=device)

    with config_context(array_api_dispatch=True):
        result = _weighted_sum(sample_score, sample_weight, normalize)

    assert isinstance(result, float)
    assert_allclose(result, expected, atol=_atol_for_type(dtype_name))


@skip_if_array_api_compat_not_configured
@pytest.mark.parametrize(
    "library", ["numpy", "numpy.array_api", "cupy", "cupy.array_api", "torch"]
)
@pytest.mark.parametrize(
    "X,reduction,expected",
    [
        ([1, 2, numpy.nan], _nanmin, 1),
        ([1, -2, -numpy.nan], _nanmin, -2),
        ([numpy.inf, numpy.inf], _nanmin, numpy.inf),
        (
            [[1, 2, 3], [numpy.nan, numpy.nan, numpy.nan], [4, 5, 6.0]],
            partial(_nanmin, axis=0),
            [1.0, 2.0, 3.0],
        ),
        (
            [[1, 2, 3], [numpy.nan, numpy.nan, numpy.nan], [4, 5, 6.0]],
            partial(_nanmin, axis=1),
            [1.0, numpy.nan, 4.0],
        ),
        ([1, 2, numpy.nan], _nanmax, 2),
        ([1, 2, numpy.nan], _nanmax, 2),
        ([-numpy.inf, -numpy.inf], _nanmax, -numpy.inf),
        (
            [[1, 2, 3], [numpy.nan, numpy.nan, numpy.nan], [4, 5, 6.0]],
            partial(_nanmax, axis=0),
            [4.0, 5.0, 6.0],
        ),
        (
            [[1, 2, 3], [numpy.nan, numpy.nan, numpy.nan], [4, 5, 6.0]],
            partial(_nanmax, axis=1),
            [3.0, numpy.nan, 6.0],
        ),
    ],
)
def test_nan_reductions(library, X, reduction, expected):
    """Check NaN reductions like _nanmin and _nanmax"""
    xp = pytest.importorskip(library)

    if isinstance(expected, list):
        expected = xp.asarray(expected)

    with config_context(array_api_dispatch=True):
        result = reduction(xp.asarray(X))

    assert_allclose(result, expected)


@skip_if_array_api_compat_not_configured
@pytest.mark.parametrize("library", ["cupy", "torch", "cupy.array_api"])
def test_convert_to_numpy_gpu(library):  # pragma: nocover
    """Check convert_to_numpy for GPU backed libraries."""
    xp = pytest.importorskip(library)

    if library == "torch":
        if not xp.backends.cuda.is_built():
            pytest.skip("test requires cuda")
        X_gpu = xp.asarray([1.0, 2.0, 3.0], device="cuda")
    else:
        X_gpu = xp.asarray([1.0, 2.0, 3.0])

    X_cpu = _convert_to_numpy(X_gpu, xp=xp)
    expected_output = numpy.asarray([1.0, 2.0, 3.0])
    assert_allclose(X_cpu, expected_output)


def test_convert_to_numpy_cpu():
    """Check convert_to_numpy for PyTorch CPU arrays."""
    torch = pytest.importorskip("torch")
    X_torch = torch.asarray([1.0, 2.0, 3.0], device="cpu")

    X_cpu = _convert_to_numpy(X_torch, xp=torch)
    expected_output = numpy.asarray([1.0, 2.0, 3.0])
    assert_allclose(X_cpu, expected_output)


class SimpleEstimator(BaseEstimator):
    def fit(self, X, y=None):
        self.X_ = X
        self.n_features_ = X.shape[0]
        return self


@skip_if_array_api_compat_not_configured
@pytest.mark.parametrize(
    "array_namespace, converter",
    [
        ("torch", lambda array: array.cpu().numpy()),
        ("numpy.array_api", lambda array: numpy.asarray(array)),
        ("cupy.array_api", lambda array: array._array.get()),
    ],
)
def test_convert_estimator_to_ndarray(array_namespace, converter):
    """Convert estimator attributes to ndarray."""
    xp = pytest.importorskip(array_namespace)

    X = xp.asarray([[1.3, 4.5]])
    est = SimpleEstimator().fit(X)

    new_est = _estimator_with_converted_arrays(est, converter)
    assert isinstance(new_est.X_, numpy.ndarray)


@skip_if_array_api_compat_not_configured
def test_convert_estimator_to_array_api():
    """Convert estimator attributes to ArrayAPI arrays."""
    xp = pytest.importorskip("numpy.array_api")

    X_np = numpy.asarray([[1.3, 4.5]])
    est = SimpleEstimator().fit(X_np)

    new_est = _estimator_with_converted_arrays(est, lambda array: xp.asarray(array))
    assert hasattr(new_est.X_, "__array_namespace__")


def test_reshape_behavior():
    """Check reshape behavior with copy and is strict with non-tuple shape."""
    xp = _NumPyAPIWrapper()
    X = xp.asarray([[1, 2, 3], [3, 4, 5]])

    X_no_copy = xp.reshape(X, (-1,), copy=False)
    assert X_no_copy.base is X

    X_copy = xp.reshape(X, (6, 1), copy=True)
    assert X_copy.base is not X.base

    with pytest.raises(TypeError, match="shape must be a tuple"):
        xp.reshape(X, -1)


@pytest.mark.parametrize("wrapper", [_ArrayAPIWrapper, _NumPyAPIWrapper])
def test_get_namespace_array_api_isdtype(wrapper):
    """Test isdtype implementation from _ArrayAPIWrapper and _NumPyAPIWrapper."""

    if wrapper == _ArrayAPIWrapper:
        xp_ = pytest.importorskip("numpy.array_api")
        xp = _ArrayAPIWrapper(xp_)
    else:
        xp = _NumPyAPIWrapper()

    assert xp.isdtype(xp.float32, xp.float32)
    assert xp.isdtype(xp.float32, "real floating")
    assert xp.isdtype(xp.float64, "real floating")
    assert not xp.isdtype(xp.int32, "real floating")

    for dtype in supported_float_dtypes(xp):
        assert xp.isdtype(dtype, "real floating")

    assert xp.isdtype(xp.bool, "bool")
    assert not xp.isdtype(xp.float32, "bool")

    assert xp.isdtype(xp.int16, "signed integer")
    assert not xp.isdtype(xp.uint32, "signed integer")

    assert xp.isdtype(xp.uint16, "unsigned integer")
    assert not xp.isdtype(xp.int64, "unsigned integer")

    assert xp.isdtype(xp.int64, "numeric")
    assert xp.isdtype(xp.float32, "numeric")
    assert xp.isdtype(xp.uint32, "numeric")

    assert not xp.isdtype(xp.float32, "complex floating")

    if wrapper == _NumPyAPIWrapper:
        assert not xp.isdtype(xp.int8, "complex floating")
        assert xp.isdtype(xp.complex64, "complex floating")
        assert xp.isdtype(xp.complex128, "complex floating")

    with pytest.raises(ValueError, match="Unrecognized data type"):
        assert xp.isdtype(xp.int16, "unknown")
