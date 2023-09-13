import numpy
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from sklearn._config import config_context
from sklearn.base import BaseEstimator
from sklearn.utils._array_api import (
    _ArrayAPIWrapper,
    _asarray_with_order,
    _convert_to_numpy,
    _estimator_with_converted_arrays,
    _NumPyAPIWrapper,
    get_namespace,
)
from sklearn.utils._testing import skip_if_array_api_compat_not_configured

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


def test_array_api_wrapper_take_for_numpy_api():
    """Test that fast path is called for numpy.array_api."""
    numpy_array_api = pytest.importorskip("numpy.array_api")
    # USe the same name as numpy.array_api
    xp_ = _AdjustableNameAPITestWrapper(numpy_array_api, "numpy.array_api")
    xp = _ArrayAPIWrapper(xp_)

    X = xp.asarray(([[1, 2, 3], [3, 4, 5]]), dtype=xp.float64)
    X_take = xp.take(X, xp.asarray([1]), axis=0)
    assert hasattr(X_take, "__array_namespace__")
    assert_array_equal(X_take, numpy.take(X, [1], axis=0))


def test_array_api_wrapper_take():
    """Test _ArrayAPIWrapper API for take."""
    numpy_array_api = pytest.importorskip("numpy.array_api")
    xp_ = _AdjustableNameAPITestWrapper(numpy_array_api, "wrapped_numpy.array_api")
    xp = _ArrayAPIWrapper(xp_)

    # Check take compared to NumPy's with axis=0
    X_1d = xp.asarray([1, 2, 3], dtype=xp.float64)
    X_take = xp.take(X_1d, xp.asarray([1]), axis=0)
    assert hasattr(X_take, "__array_namespace__")
    assert_array_equal(X_take, numpy.take(X_1d, [1], axis=0))

    X = xp.asarray(([[1, 2, 3], [3, 4, 5]]), dtype=xp.float64)
    X_take = xp.take(X, xp.asarray([0]), axis=0)
    assert hasattr(X_take, "__array_namespace__")
    assert_array_equal(X_take, numpy.take(X, [0], axis=0))

    # Check take compared to NumPy's with axis=1
    X_take = xp.take(X, xp.asarray([0, 2]), axis=1)
    assert hasattr(X_take, "__array_namespace__")
    assert_array_equal(X_take, numpy.take(X, [0, 2], axis=1))

    with pytest.raises(ValueError, match=r"Only axis in \(0, 1\) is supported"):
        xp.take(X, xp.asarray([0]), axis=2)

    with pytest.raises(ValueError, match=r"Only X.ndim in \(1, 2\) is supported"):
        xp.take(xp.asarray([[[0]]]), xp.asarray([0]), axis=0)


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


@skip_if_array_api_compat_not_configured
@pytest.mark.parametrize("library", ["cupy", "torch", "cupy.array_api"])
def test_convert_to_numpy_gpu(library):  # pragma: nocover
    """Check convert_to_numpy for GPU backed libraries."""
    xp = pytest.importorskip(library)

    if library == "torch":
        if not xp.has_cuda:
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
