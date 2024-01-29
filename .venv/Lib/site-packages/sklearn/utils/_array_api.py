"""Tools to support array_api."""
import itertools
import math
from functools import wraps

import numpy
import scipy.special as special

from .._config import get_config
from .fixes import parse_version


def yield_namespace_device_dtype_combinations():
    """Yield supported namespace, device, dtype tuples for testing.

    Use this to test that an estimator works with all combinations.

    Returns
    -------
    array_namespace : str
        The name of the Array API namespace.

    device : str
        The name of the device on which to allocate the arrays. Can be None to
        indicate that the default value should be used.

    dtype_name : str
        The name of the data type to use for arrays. Can be None to indicate
        that the default value should be used.
    """
    for array_namespace in [
        # The following is used to test the array_api_compat wrapper when
        # array_api_dispatch is enabled: in particular, the arrays used in the
        # tests are regular numpy arrays without any "device" attribute.
        "numpy",
        # Stricter NumPy-based Array API implementation. The
        # numpy.array_api.Array instances always a dummy "device" attribute.
        "numpy.array_api",
        "cupy",
        "cupy.array_api",
        "torch",
    ]:
        if array_namespace == "torch":
            for device, dtype in itertools.product(
                ("cpu", "cuda"), ("float64", "float32")
            ):
                yield array_namespace, device, dtype
            yield array_namespace, "mps", "float32"
        else:
            yield array_namespace, None, None


def _check_array_api_dispatch(array_api_dispatch):
    """Check that array_api_compat is installed and NumPy version is compatible.

    array_api_compat follows NEP29, which has a higher minimum NumPy version than
    scikit-learn.
    """
    if array_api_dispatch:
        try:
            import array_api_compat  # noqa
        except ImportError:
            raise ImportError(
                "array_api_compat is required to dispatch arrays using the API"
                " specification"
            )

        numpy_version = parse_version(numpy.__version__)
        min_numpy_version = "1.21"
        if numpy_version < parse_version(min_numpy_version):
            raise ImportError(
                f"NumPy must be {min_numpy_version} or newer to dispatch array using"
                " the API specification"
            )


def device(x):
    """Hardware device the array data resides on.

    Parameters
    ----------
    x : array
        Array instance from NumPy or an array API compatible library.

    Returns
    -------
    out : device
        `device` object (see the "Device Support" section of the array API spec).
    """
    if isinstance(x, (numpy.ndarray, numpy.generic)):
        return "cpu"
    return x.device


def size(x):
    """Return the total number of elements of x.

    Parameters
    ----------
    x : array
        Array instance from NumPy or an array API compatible library.

    Returns
    -------
    out : int
        Total number of elements.
    """
    return math.prod(x.shape)


def _is_numpy_namespace(xp):
    """Return True if xp is backed by NumPy."""
    return xp.__name__ in {"numpy", "array_api_compat.numpy", "numpy.array_api"}


def _union1d(a, b, xp):
    if _is_numpy_namespace(xp):
        return xp.asarray(numpy.union1d(a, b))
    assert a.ndim == b.ndim == 1
    return xp.unique_values(xp.concat([xp.unique_values(a), xp.unique_values(b)]))


def isdtype(dtype, kind, *, xp):
    """Returns a boolean indicating whether a provided dtype is of type "kind".

    Included in the v2022.12 of the Array API spec.
    https://data-apis.org/array-api/latest/API_specification/generated/array_api.isdtype.html
    """
    if isinstance(kind, tuple):
        return any(_isdtype_single(dtype, k, xp=xp) for k in kind)
    else:
        return _isdtype_single(dtype, kind, xp=xp)


def _isdtype_single(dtype, kind, *, xp):
    if isinstance(kind, str):
        if kind == "bool":
            return dtype == xp.bool
        elif kind == "signed integer":
            return dtype in {xp.int8, xp.int16, xp.int32, xp.int64}
        elif kind == "unsigned integer":
            return dtype in {xp.uint8, xp.uint16, xp.uint32, xp.uint64}
        elif kind == "integral":
            return any(
                _isdtype_single(dtype, k, xp=xp)
                for k in ("signed integer", "unsigned integer")
            )
        elif kind == "real floating":
            return dtype in supported_float_dtypes(xp)
        elif kind == "complex floating":
            # Some name spaces do not have complex, such as cupy.array_api
            # and numpy.array_api
            complex_dtypes = set()
            if hasattr(xp, "complex64"):
                complex_dtypes.add(xp.complex64)
            if hasattr(xp, "complex128"):
                complex_dtypes.add(xp.complex128)
            return dtype in complex_dtypes
        elif kind == "numeric":
            return any(
                _isdtype_single(dtype, k, xp=xp)
                for k in ("integral", "real floating", "complex floating")
            )
        else:
            raise ValueError(f"Unrecognized data type kind: {kind!r}")
    else:
        return dtype == kind


def supported_float_dtypes(xp):
    """Supported floating point types for the namespace

    Note: float16 is not officially part of the Array API spec at the
    time of writing but scikit-learn estimators and functions can choose
    to accept it when xp.float16 is defined.

    https://data-apis.org/array-api/latest/API_specification/data_types.html
    """
    if hasattr(xp, "float16"):
        return (xp.float64, xp.float32, xp.float16)
    else:
        return (xp.float64, xp.float32)


class _ArrayAPIWrapper:
    """sklearn specific Array API compatibility wrapper

    This wrapper makes it possible for scikit-learn maintainers to
    deal with discrepancies between different implementations of the
    Python Array API standard and its evolution over time.

    The Python Array API standard specification:
    https://data-apis.org/array-api/latest/

    Documentation of the NumPy implementation:
    https://numpy.org/neps/nep-0047-array-api-standard.html
    """

    def __init__(self, array_namespace):
        self._namespace = array_namespace

    def __getattr__(self, name):
        return getattr(self._namespace, name)

    def __eq__(self, other):
        return self._namespace == other._namespace

    def isdtype(self, dtype, kind):
        return isdtype(dtype, kind, xp=self._namespace)


def _check_device_cpu(device):  # noqa
    if device not in {"cpu", None}:
        raise ValueError(f"Unsupported device for NumPy: {device!r}")


def _accept_device_cpu(func):
    @wraps(func)
    def wrapped_func(*args, **kwargs):
        _check_device_cpu(kwargs.pop("device", None))
        return func(*args, **kwargs)

    return wrapped_func


class _NumPyAPIWrapper:
    """Array API compat wrapper for any numpy version

    NumPy < 1.22 does not expose the numpy.array_api namespace. This
    wrapper makes it possible to write code that uses the standard
    Array API while working with any version of NumPy supported by
    scikit-learn.

    See the `get_namespace()` public function for more details.
    """

    # Creation functions in spec:
    # https://data-apis.org/array-api/latest/API_specification/creation_functions.html
    _CREATION_FUNCS = {
        "arange",
        "empty",
        "empty_like",
        "eye",
        "full",
        "full_like",
        "linspace",
        "ones",
        "ones_like",
        "zeros",
        "zeros_like",
    }
    # Data types in spec
    # https://data-apis.org/array-api/latest/API_specification/data_types.html
    _DTYPES = {
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        # XXX: float16 is not part of the Array API spec but exposed by
        # some namespaces.
        "float16",
        "float32",
        "float64",
        "complex64",
        "complex128",
    }

    def __getattr__(self, name):
        attr = getattr(numpy, name)

        # Support device kwargs and make sure they are on the CPU
        if name in self._CREATION_FUNCS:
            return _accept_device_cpu(attr)

        # Convert to dtype objects
        if name in self._DTYPES:
            return numpy.dtype(attr)
        return attr

    @property
    def bool(self):
        return numpy.bool_

    def astype(self, x, dtype, *, copy=True, casting="unsafe"):
        # astype is not defined in the top level NumPy namespace
        return x.astype(dtype, copy=copy, casting=casting)

    def asarray(self, x, *, dtype=None, device=None, copy=None):  # noqa
        _check_device_cpu(device)
        # Support copy in NumPy namespace
        if copy is True:
            return numpy.array(x, copy=True, dtype=dtype)
        else:
            return numpy.asarray(x, dtype=dtype)

    def unique_inverse(self, x):
        return numpy.unique(x, return_inverse=True)

    def unique_counts(self, x):
        return numpy.unique(x, return_counts=True)

    def unique_values(self, x):
        return numpy.unique(x)

    def concat(self, arrays, *, axis=None):
        return numpy.concatenate(arrays, axis=axis)

    def reshape(self, x, shape, *, copy=None):
        """Gives a new shape to an array without changing its data.

        The Array API specification requires shape to be a tuple.
        https://data-apis.org/array-api/latest/API_specification/generated/array_api.reshape.html
        """
        if not isinstance(shape, tuple):
            raise TypeError(
                f"shape must be a tuple, got {shape!r} of type {type(shape)}"
            )

        if copy is True:
            x = x.copy()
        return numpy.reshape(x, shape)

    def isdtype(self, dtype, kind):
        return isdtype(dtype, kind, xp=self)


_NUMPY_API_WRAPPER_INSTANCE = _NumPyAPIWrapper()


def get_namespace(*arrays):
    """Get namespace of arrays.

    Introspect `arrays` arguments and return their common Array API
    compatible namespace object, if any. NumPy 1.22 and later can
    construct such containers using the `numpy.array_api` namespace
    for instance.

    See: https://numpy.org/neps/nep-0047-array-api-standard.html

    If `arrays` are regular numpy arrays, an instance of the
    `_NumPyAPIWrapper` compatibility wrapper is returned instead.

    Namespace support is not enabled by default. To enabled it
    call:

      sklearn.set_config(array_api_dispatch=True)

    or:

      with sklearn.config_context(array_api_dispatch=True):
          # your code here

    Otherwise an instance of the `_NumPyAPIWrapper`
    compatibility wrapper is always returned irrespective of
    the fact that arrays implement the `__array_namespace__`
    protocol or not.

    Parameters
    ----------
    *arrays : array objects
        Array objects.

    Returns
    -------
    namespace : module
        Namespace shared by array objects. If any of the `arrays` are not arrays,
        the namespace defaults to NumPy.

    is_array_api_compliant : bool
        True if the arrays are containers that implement the Array API spec.
        Always False when array_api_dispatch=False.
    """
    array_api_dispatch = get_config()["array_api_dispatch"]
    if not array_api_dispatch:
        return _NUMPY_API_WRAPPER_INSTANCE, False

    _check_array_api_dispatch(array_api_dispatch)

    # array-api-compat is a required dependency of scikit-learn only when
    # configuring `array_api_dispatch=True`. Its import should therefore be
    # protected by _check_array_api_dispatch to display an informative error
    # message in case it is missing.
    import array_api_compat

    namespace, is_array_api_compliant = array_api_compat.get_namespace(*arrays), True

    # These namespaces need additional wrapping to smooth out small differences
    # between implementations
    if namespace.__name__ in {"numpy.array_api", "cupy.array_api"}:
        namespace = _ArrayAPIWrapper(namespace)

    return namespace, is_array_api_compliant


def _expit(X):
    xp, _ = get_namespace(X)
    if _is_numpy_namespace(xp):
        return xp.asarray(special.expit(numpy.asarray(X)))

    return 1.0 / (1.0 + xp.exp(-X))


def _add_to_diagonal(array, value, xp):
    # Workaround for the lack of support for xp.reshape(a, shape, copy=False) in
    # numpy.array_api: https://github.com/numpy/numpy/issues/23410
    value = xp.asarray(value, dtype=array.dtype)
    if _is_numpy_namespace(xp):
        array_np = numpy.asarray(array)
        array_np.flat[:: array.shape[0] + 1] += value
        return xp.asarray(array_np)
    elif value.ndim == 1:
        for i in range(array.shape[0]):
            array[i, i] += value[i]
    else:
        # scalar value
        for i in range(array.shape[0]):
            array[i, i] += value


def _weighted_sum(sample_score, sample_weight, normalize=False, xp=None):
    # XXX: this function accepts Array API input but returns a Python scalar
    # float. The call to float() is convenient because it removes the need to
    # move back results from device to host memory (e.g. calling `.cpu()` on a
    # torch tensor). However, this might interact in unexpected ways (break?)
    # with lazy Array API implementations. See:
    # https://github.com/data-apis/array-api/issues/642
    if xp is None:
        xp, _ = get_namespace(sample_score)
    if normalize and _is_numpy_namespace(xp):
        sample_score_np = numpy.asarray(sample_score)
        if sample_weight is not None:
            sample_weight_np = numpy.asarray(sample_weight)
        else:
            sample_weight_np = None
        return float(numpy.average(sample_score_np, weights=sample_weight_np))

    if not xp.isdtype(sample_score.dtype, "real floating"):
        # We move to cpu device ahead of time since certain devices may not support
        # float64, but we want the same precision for all devices and namespaces.
        sample_score = xp.astype(xp.asarray(sample_score, device="cpu"), xp.float64)

    if sample_weight is not None:
        sample_weight = xp.asarray(
            sample_weight, dtype=sample_score.dtype, device=device(sample_score)
        )
        if not xp.isdtype(sample_weight.dtype, "real floating"):
            sample_weight = xp.astype(sample_weight, xp.float64)

    if normalize:
        if sample_weight is not None:
            scale = xp.sum(sample_weight)
        else:
            scale = sample_score.shape[0]
        if scale != 0:
            sample_score = sample_score / scale

    if sample_weight is not None:
        return float(sample_score @ sample_weight)
    else:
        return float(xp.sum(sample_score))


def _nanmin(X, axis=None):
    # TODO: refactor once nan-aware reductions are standardized:
    # https://github.com/data-apis/array-api/issues/621
    xp, _ = get_namespace(X)
    if _is_numpy_namespace(xp):
        return xp.asarray(numpy.nanmin(X, axis=axis))

    else:
        mask = xp.isnan(X)
        X = xp.min(xp.where(mask, xp.asarray(+xp.inf, device=device(X)), X), axis=axis)
        # Replace Infs from all NaN slices with NaN again
        mask = xp.all(mask, axis=axis)
        if xp.any(mask):
            X = xp.where(mask, xp.asarray(xp.nan), X)
        return X


def _nanmax(X, axis=None):
    # TODO: refactor once nan-aware reductions are standardized:
    # https://github.com/data-apis/array-api/issues/621
    xp, _ = get_namespace(X)
    if _is_numpy_namespace(xp):
        return xp.asarray(numpy.nanmax(X, axis=axis))

    else:
        mask = xp.isnan(X)
        X = xp.max(xp.where(mask, xp.asarray(-xp.inf, device=device(X)), X), axis=axis)
        # Replace Infs from all NaN slices with NaN again
        mask = xp.all(mask, axis=axis)
        if xp.any(mask):
            X = xp.where(mask, xp.asarray(xp.nan), X)
        return X


def _asarray_with_order(array, dtype=None, order=None, copy=None, *, xp=None):
    """Helper to support the order kwarg only for NumPy-backed arrays

    Memory layout parameter `order` is not exposed in the Array API standard,
    however some input validation code in scikit-learn needs to work both
    for classes and functions that will leverage Array API only operations
    and for code that inherently relies on NumPy backed data containers with
    specific memory layout constraints (e.g. our own Cython code). The
    purpose of this helper is to make it possible to share code for data
    container validation without memory copies for both downstream use cases:
    the `order` parameter is only enforced if the input array implementation
    is NumPy based, otherwise `order` is just silently ignored.
    """
    if xp is None:
        xp, _ = get_namespace(array)
    if _is_numpy_namespace(xp):
        # Use NumPy API to support order
        if copy is True:
            array = numpy.array(array, order=order, dtype=dtype)
        else:
            array = numpy.asarray(array, order=order, dtype=dtype)

        # At this point array is a NumPy ndarray. We convert it to an array
        # container that is consistent with the input's namespace.
        return xp.asarray(array)
    else:
        return xp.asarray(array, dtype=dtype, copy=copy)


def _convert_to_numpy(array, xp):
    """Convert X into a NumPy ndarray on the CPU."""
    xp_name = xp.__name__

    if xp_name in {"array_api_compat.torch", "torch"}:
        return array.cpu().numpy()
    elif xp_name == "cupy.array_api":
        return array._array.get()
    elif xp_name in {"array_api_compat.cupy", "cupy"}:  # pragma: nocover
        return array.get()

    return numpy.asarray(array)


def _estimator_with_converted_arrays(estimator, converter):
    """Create new estimator which converting all attributes that are arrays.

    The converter is called on all NumPy arrays and arrays that support the
    `DLPack interface <https://dmlc.github.io/dlpack/latest/>`__.

    Parameters
    ----------
    estimator : Estimator
        Estimator to convert

    converter : callable
        Callable that takes an array attribute and returns the converted array.

    Returns
    -------
    new_estimator : Estimator
        Convert estimator
    """
    from sklearn.base import clone

    new_estimator = clone(estimator)
    for key, attribute in vars(estimator).items():
        if hasattr(attribute, "__dlpack__") or isinstance(attribute, numpy.ndarray):
            attribute = converter(attribute)
        setattr(new_estimator, key, attribute)
    return new_estimator


def _atol_for_type(dtype):
    """Return the absolute tolerance for a given dtype."""
    return numpy.finfo(dtype).eps * 100
