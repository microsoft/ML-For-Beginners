"""Compatibility fixes for older version of python, numpy and scipy

If you add content to this file, please give the version of the package
at which the fix is no longer needed.
"""
# Authors: Emmanuelle Gouillart <emmanuelle.gouillart@normalesup.org>
#          Gael Varoquaux <gael.varoquaux@normalesup.org>
#          Fabian Pedregosa <fpedregosa@acm.org>
#          Lars Buitinck
#
# License: BSD 3 clause


import numpy as np
import scipy
import scipy.sparse.linalg
import scipy.stats
import threadpoolctl

import sklearn

from ..externals._packaging.version import parse as parse_version
from .deprecation import deprecated

np_version = parse_version(np.__version__)
np_base_version = parse_version(np_version.base_version)
sp_version = parse_version(scipy.__version__)
sp_base_version = parse_version(sp_version.base_version)

# TODO: We can consider removing the containers and importing
# directly from SciPy when sparse matrices will be deprecated.
CSR_CONTAINERS = [scipy.sparse.csr_matrix]
CSC_CONTAINERS = [scipy.sparse.csc_matrix]
COO_CONTAINERS = [scipy.sparse.coo_matrix]
LIL_CONTAINERS = [scipy.sparse.lil_matrix]
DOK_CONTAINERS = [scipy.sparse.dok_matrix]
BSR_CONTAINERS = [scipy.sparse.bsr_matrix]
DIA_CONTAINERS = [scipy.sparse.dia_matrix]

if parse_version(scipy.__version__) >= parse_version("1.8"):
    # Sparse Arrays have been added in SciPy 1.8
    # TODO: When SciPy 1.8 is the minimum supported version,
    # those list can be created directly without this condition.
    # See: https://github.com/scikit-learn/scikit-learn/issues/27090
    CSR_CONTAINERS.append(scipy.sparse.csr_array)
    CSC_CONTAINERS.append(scipy.sparse.csc_array)
    COO_CONTAINERS.append(scipy.sparse.coo_array)
    LIL_CONTAINERS.append(scipy.sparse.lil_array)
    DOK_CONTAINERS.append(scipy.sparse.dok_array)
    BSR_CONTAINERS.append(scipy.sparse.bsr_array)
    DIA_CONTAINERS.append(scipy.sparse.dia_array)

try:
    from scipy.optimize._linesearch import line_search_wolfe1, line_search_wolfe2
except ImportError:  # SciPy < 1.8
    from scipy.optimize.linesearch import line_search_wolfe2, line_search_wolfe1  # type: ignore  # noqa


def _object_dtype_isnan(X):
    return X != X


# Rename the `method` kwarg to `interpolation` for NumPy < 1.22, because
# `interpolation` kwarg was deprecated in favor of `method` in NumPy >= 1.22.
def _percentile(a, q, *, method="linear", **kwargs):
    return np.percentile(a, q, interpolation=method, **kwargs)


if np_version < parse_version("1.22"):
    percentile = _percentile
else:  # >= 1.22
    from numpy import percentile  # type: ignore  # noqa


# compatibility fix for threadpoolctl >= 3.0.0
# since version 3 it's possible to setup a global threadpool controller to avoid
# looping through all loaded shared libraries each time.
# the global controller is created during the first call to threadpoolctl.
def _get_threadpool_controller():
    if not hasattr(threadpoolctl, "ThreadpoolController"):
        return None

    if not hasattr(sklearn, "_sklearn_threadpool_controller"):
        sklearn._sklearn_threadpool_controller = threadpoolctl.ThreadpoolController()

    return sklearn._sklearn_threadpool_controller


def threadpool_limits(limits=None, user_api=None):
    controller = _get_threadpool_controller()
    if controller is not None:
        return controller.limit(limits=limits, user_api=user_api)
    else:
        return threadpoolctl.threadpool_limits(limits=limits, user_api=user_api)


threadpool_limits.__doc__ = threadpoolctl.threadpool_limits.__doc__


def threadpool_info():
    controller = _get_threadpool_controller()
    if controller is not None:
        return controller.info()
    else:
        return threadpoolctl.threadpool_info()


threadpool_info.__doc__ = threadpoolctl.threadpool_info.__doc__


@deprecated(
    "The function `delayed` has been moved from `sklearn.utils.fixes` to "
    "`sklearn.utils.parallel`. This import path will be removed in 1.5."
)
def delayed(function):
    from sklearn.utils.parallel import delayed

    return delayed(function)


# TODO: Remove when SciPy 1.11 is the minimum supported version
def _mode(a, axis=0):
    if sp_version >= parse_version("1.9.0"):
        mode = scipy.stats.mode(a, axis=axis, keepdims=True)
        if sp_version >= parse_version("1.10.999"):
            # scipy.stats.mode has changed returned array shape with axis=None
            # and keepdims=True, see https://github.com/scipy/scipy/pull/17561
            if axis is None:
                mode = np.ravel(mode)
        return mode
    return scipy.stats.mode(a, axis=axis)


# TODO: Remove when Scipy 1.12 is the minimum supported version
if sp_base_version >= parse_version("1.12.0"):
    _sparse_linalg_cg = scipy.sparse.linalg.cg
else:

    def _sparse_linalg_cg(A, b, **kwargs):
        if "rtol" in kwargs:
            kwargs["tol"] = kwargs.pop("rtol")
        if "atol" not in kwargs:
            kwargs["atol"] = "legacy"
        return scipy.sparse.linalg.cg(A, b, **kwargs)


# TODO: Fuse the modern implementations of _sparse_min_max and _sparse_nan_min_max
# into the public min_max_axis function when Scipy 1.11 is the minimum supported
# version and delete the backport in the else branch below.
if sp_base_version >= parse_version("1.11.0"):

    def _sparse_min_max(X, axis):
        the_min = X.min(axis=axis)
        the_max = X.max(axis=axis)

        if axis is not None:
            the_min = the_min.toarray().ravel()
            the_max = the_max.toarray().ravel()

        return the_min, the_max

    def _sparse_nan_min_max(X, axis):
        the_min = X.nanmin(axis=axis)
        the_max = X.nanmax(axis=axis)

        if axis is not None:
            the_min = the_min.toarray().ravel()
            the_max = the_max.toarray().ravel()

        return the_min, the_max

else:
    # This code is mostly taken from scipy 0.14 and extended to handle nans, see
    # https://github.com/scikit-learn/scikit-learn/pull/11196
    def _minor_reduce(X, ufunc):
        major_index = np.flatnonzero(np.diff(X.indptr))

        # reduceat tries casts X.indptr to intp, which errors
        # if it is int64 on a 32 bit system.
        # Reinitializing prevents this where possible, see #13737
        X = type(X)((X.data, X.indices, X.indptr), shape=X.shape)
        value = ufunc.reduceat(X.data, X.indptr[major_index])
        return major_index, value

    def _min_or_max_axis(X, axis, min_or_max):
        N = X.shape[axis]
        if N == 0:
            raise ValueError("zero-size array to reduction operation")
        M = X.shape[1 - axis]
        mat = X.tocsc() if axis == 0 else X.tocsr()
        mat.sum_duplicates()
        major_index, value = _minor_reduce(mat, min_or_max)
        not_full = np.diff(mat.indptr)[major_index] < N
        value[not_full] = min_or_max(value[not_full], 0)
        mask = value != 0
        major_index = np.compress(mask, major_index)
        value = np.compress(mask, value)

        if axis == 0:
            res = scipy.sparse.coo_matrix(
                (value, (np.zeros(len(value)), major_index)),
                dtype=X.dtype,
                shape=(1, M),
            )
        else:
            res = scipy.sparse.coo_matrix(
                (value, (major_index, np.zeros(len(value)))),
                dtype=X.dtype,
                shape=(M, 1),
            )
        return res.A.ravel()

    def _sparse_min_or_max(X, axis, min_or_max):
        if axis is None:
            if 0 in X.shape:
                raise ValueError("zero-size array to reduction operation")
            zero = X.dtype.type(0)
            if X.nnz == 0:
                return zero
            m = min_or_max.reduce(X.data.ravel())
            if X.nnz != np.prod(X.shape):
                m = min_or_max(zero, m)
            return m
        if axis < 0:
            axis += 2
        if (axis == 0) or (axis == 1):
            return _min_or_max_axis(X, axis, min_or_max)
        else:
            raise ValueError("invalid axis, use 0 for rows, or 1 for columns")

    def _sparse_min_max(X, axis):
        return (
            _sparse_min_or_max(X, axis, np.minimum),
            _sparse_min_or_max(X, axis, np.maximum),
        )

    def _sparse_nan_min_max(X, axis):
        return (
            _sparse_min_or_max(X, axis, np.fmin),
            _sparse_min_or_max(X, axis, np.fmax),
        )


# For +1.25 NumPy versions exceptions and warnings are being moved
# to a dedicated submodule.
if np_version >= parse_version("1.25.0"):
    from numpy.exceptions import ComplexWarning, VisibleDeprecationWarning
else:
    from numpy import ComplexWarning, VisibleDeprecationWarning  # type: ignore  # noqa


# TODO: Remove when Scipy 1.6 is the minimum supported version
try:
    from scipy.integrate import trapezoid  # type: ignore  # noqa
except ImportError:
    from scipy.integrate import trapz as trapezoid  # type: ignore  # noqa


# TODO: Remove when Pandas > 2.2 is the minimum supported version
def pd_fillna(pd, frame):
    pd_version = parse_version(pd.__version__).base_version
    if parse_version(pd_version) < parse_version("2.2"):
        frame = frame.fillna(value=np.nan)
    else:
        with pd.option_context("future.no_silent_downcasting", True):
            frame = frame.fillna(value=np.nan).infer_objects(copy=False)
    return frame


# TODO: remove when SciPy 1.12 is the minimum supported version
def _preserve_dia_indices_dtype(
    sparse_container, original_container_format, requested_sparse_format
):
    """Preserve indices dtype for SciPy < 1.12 when converting from DIA to CSR/CSC.

    For SciPy < 1.12, DIA arrays indices are upcasted to `np.int64` that is
    inconsistent with DIA matrices. We downcast the indices dtype to `np.int32` to
    be consistent with DIA matrices.

    The converted indices arrays are affected back inplace to the sparse container.

    Parameters
    ----------
    sparse_container : sparse container
        Sparse container to be checked.
    requested_sparse_format : str or bool
        The type of format of `sparse_container`.

    Notes
    -----
    See https://github.com/scipy/scipy/issues/19245 for more details.
    """
    if original_container_format == "dia_array" and requested_sparse_format in (
        "csr",
        "coo",
    ):
        if requested_sparse_format == "csr":
            index_dtype = _smallest_admissible_index_dtype(
                arrays=(sparse_container.indptr, sparse_container.indices),
                maxval=max(sparse_container.nnz, sparse_container.shape[1]),
                check_contents=True,
            )
            sparse_container.indices = sparse_container.indices.astype(
                index_dtype, copy=False
            )
            sparse_container.indptr = sparse_container.indptr.astype(
                index_dtype, copy=False
            )
        else:  # requested_sparse_format == "coo"
            index_dtype = _smallest_admissible_index_dtype(
                maxval=max(sparse_container.shape)
            )
            sparse_container.row = sparse_container.row.astype(index_dtype, copy=False)
            sparse_container.col = sparse_container.col.astype(index_dtype, copy=False)


# TODO: remove when SciPy 1.12 is the minimum supported version
def _smallest_admissible_index_dtype(arrays=(), maxval=None, check_contents=False):
    """Based on input (integer) arrays `a`, determine a suitable index data
    type that can hold the data in the arrays.

    This function returns `np.int64` if it either required by `maxval` or based on the
    largest precision of the dtype of the arrays passed as argument, or by the their
    contents (when `check_contents is True`). If none of the condition requires
    `np.int64` then this function returns `np.int32`.

    Parameters
    ----------
    arrays : ndarray or tuple of ndarrays, default=()
        Input arrays whose types/contents to check.

    maxval : float, default=None
        Maximum value needed.

    check_contents : bool, default=False
        Whether to check the values in the arrays and not just their types.
        By default, check only the types.

    Returns
    -------
    dtype : {np.int32, np.int64}
        Suitable index data type (int32 or int64).
    """

    int32min = np.int32(np.iinfo(np.int32).min)
    int32max = np.int32(np.iinfo(np.int32).max)

    if maxval is not None:
        if maxval > np.iinfo(np.int64).max:
            raise ValueError(
                f"maxval={maxval} is to large to be represented as np.int64."
            )
        if maxval > int32max:
            return np.int64

    if isinstance(arrays, np.ndarray):
        arrays = (arrays,)

    for arr in arrays:
        if not isinstance(arr, np.ndarray):
            raise TypeError(
                f"Arrays should be of type np.ndarray, got {type(arr)} instead."
            )
        if not np.issubdtype(arr.dtype, np.integer):
            raise ValueError(
                f"Array dtype {arr.dtype} is not supported for index dtype. We expect "
                "integral values."
            )
        if not np.can_cast(arr.dtype, np.int32):
            if not check_contents:
                # when `check_contents` is False, we stay on the safe side and return
                # np.int64.
                return np.int64
            if arr.size == 0:
                # a bigger type not needed yet, let's look at the next array
                continue
            else:
                maxval = arr.max()
                minval = arr.min()
                if minval < int32min or maxval > int32max:
                    # a big index type is actually needed
                    return np.int64

    return np.int32


# TODO: Remove when Scipy 1.12 is the minimum supported version
if sp_version < parse_version("1.12"):
    from ..externals._scipy.sparse.csgraph import laplacian  # type: ignore  # noqa
else:
    from scipy.sparse.csgraph import laplacian  # type: ignore  # noqa  # pragma: no cover
