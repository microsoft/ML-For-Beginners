""" Utility functions for sparse matrix module
"""

import sys
import operator
import numpy as np
from math import prod
import scipy.sparse as sp
from scipy._lib._util import np_long, np_ulong


__all__ = ['upcast', 'getdtype', 'getdata', 'isscalarlike', 'isintlike',
           'isshape', 'issequence', 'isdense', 'ismatrix', 'get_sum_dtype']

supported_dtypes = [np.bool_, np.byte, np.ubyte, np.short, np.ushort, np.intc,
                    np.uintc, np_long, np_ulong, np.longlong, np.ulonglong,
                    np.float32, np.float64, np.longdouble, 
                    np.complex64, np.complex128, np.clongdouble]

_upcast_memo = {}


def upcast(*args):
    """Returns the nearest supported sparse dtype for the
    combination of one or more types.

    upcast(t0, t1, ..., tn) -> T  where T is a supported dtype

    Examples
    --------
    >>> from scipy.sparse._sputils import upcast
    >>> upcast('int32')
    <type 'numpy.int32'>
    >>> upcast('bool')
    <type 'numpy.bool_'>
    >>> upcast('int32','float32')
    <type 'numpy.float64'>
    >>> upcast('bool',complex,float)
    <type 'numpy.complex128'>

    """

    t = _upcast_memo.get(hash(args))
    if t is not None:
        return t

    upcast = np.result_type(*args)

    for t in supported_dtypes:
        if np.can_cast(upcast, t):
            _upcast_memo[hash(args)] = t
            return t

    raise TypeError(f'no supported conversion for types: {args!r}')


def upcast_char(*args):
    """Same as `upcast` but taking dtype.char as input (faster)."""
    t = _upcast_memo.get(args)
    if t is not None:
        return t
    t = upcast(*map(np.dtype, args))
    _upcast_memo[args] = t
    return t


def upcast_scalar(dtype, scalar):
    """Determine data type for binary operation between an array of
    type `dtype` and a scalar.
    """
    return (np.array([0], dtype=dtype) * scalar).dtype


def downcast_intp_index(arr):
    """
    Down-cast index array to np.intp dtype if it is of a larger dtype.

    Raise an error if the array contains a value that is too large for
    intp.
    """
    if arr.dtype.itemsize > np.dtype(np.intp).itemsize:
        if arr.size == 0:
            return arr.astype(np.intp)
        maxval = arr.max()
        minval = arr.min()
        if maxval > np.iinfo(np.intp).max or minval < np.iinfo(np.intp).min:
            raise ValueError("Cannot deal with arrays with indices larger "
                             "than the machine maximum address size "
                             "(e.g. 64-bit indices on 32-bit machine).")
        return arr.astype(np.intp)
    return arr


def to_native(A):
    """
    Ensure that the data type of the NumPy array `A` has native byte order.

    `A` must be a NumPy array.  If the data type of `A` does not have native
    byte order, a copy of `A` with a native byte order is returned. Otherwise
    `A` is returned.
    """
    dt = A.dtype
    if dt.isnative:
        # Don't call `asarray()` if A is already native, to avoid unnecessarily
        # creating a view of the input array.
        return A
    return np.asarray(A, dtype=dt.newbyteorder('native'))


def getdtype(dtype, a=None, default=None):
    """Function used to simplify argument processing. If 'dtype' is not
    specified (is None), returns a.dtype; otherwise returns a np.dtype
    object created from the specified dtype argument. If 'dtype' and 'a'
    are both None, construct a data type out of the 'default' parameter.
    Furthermore, 'dtype' must be in 'allowed' set.
    """
    # TODO is this really what we want?
    if dtype is None:
        try:
            newdtype = a.dtype
        except AttributeError as e:
            if default is not None:
                newdtype = np.dtype(default)
            else:
                raise TypeError("could not interpret data type") from e
    else:
        newdtype = np.dtype(dtype)
        if newdtype == np.object_:
            raise ValueError(
                "object dtype is not supported by sparse matrices"
            )

    return newdtype


def getdata(obj, dtype=None, copy=False) -> np.ndarray:
    """
    This is a wrapper of `np.array(obj, dtype=dtype, copy=copy)`
    that will generate a warning if the result is an object array.
    """
    data = np.array(obj, dtype=dtype, copy=copy)
    # Defer to getdtype for checking that the dtype is OK.
    # This is called for the validation only; we don't need the return value.
    getdtype(data.dtype)
    return data


def get_index_dtype(arrays=(), maxval=None, check_contents=False):
    """
    Based on input (integer) arrays `a`, determine a suitable index data
    type that can hold the data in the arrays.

    Parameters
    ----------
    arrays : tuple of array_like
        Input arrays whose types/contents to check
    maxval : float, optional
        Maximum value needed
    check_contents : bool, optional
        Whether to check the values in the arrays and not just their types.
        Default: False (check only the types)

    Returns
    -------
    dtype : dtype
        Suitable index data type (int32 or int64)

    """

    int32min = np.int32(np.iinfo(np.int32).min)
    int32max = np.int32(np.iinfo(np.int32).max)

    # not using intc directly due to misinteractions with pythran
    dtype = np.int32 if np.intc().itemsize == 4 else np.int64
    if maxval is not None:
        maxval = np.int64(maxval)
        if maxval > int32max:
            dtype = np.int64

    if isinstance(arrays, np.ndarray):
        arrays = (arrays,)

    for arr in arrays:
        arr = np.asarray(arr)
        if not np.can_cast(arr.dtype, np.int32):
            if check_contents:
                if arr.size == 0:
                    # a bigger type not needed
                    continue
                elif np.issubdtype(arr.dtype, np.integer):
                    maxval = arr.max()
                    minval = arr.min()
                    if minval >= int32min and maxval <= int32max:
                        # a bigger type not needed
                        continue

            dtype = np.int64
            break

    return dtype


def get_sum_dtype(dtype: np.dtype) -> np.dtype:
    """Mimic numpy's casting for np.sum"""
    if dtype.kind == 'u' and np.can_cast(dtype, np.uint):
        return np.uint
    if np.can_cast(dtype, np.int_):
        return np.int_
    return dtype


def isscalarlike(x) -> bool:
    """Is x either a scalar, an array scalar, or a 0-dim array?"""
    return np.isscalar(x) or (isdense(x) and x.ndim == 0)


def isintlike(x) -> bool:
    """Is x appropriate as an index into a sparse matrix? Returns True
    if it can be cast safely to a machine int.
    """
    # Fast-path check to eliminate non-scalar values. operator.index would
    # catch this case too, but the exception catching is slow.
    if np.ndim(x) != 0:
        return False
    try:
        operator.index(x)
    except (TypeError, ValueError):
        try:
            loose_int = bool(int(x) == x)
        except (TypeError, ValueError):
            return False
        if loose_int:
            msg = "Inexact indices into sparse matrices are not allowed"
            raise ValueError(msg)
        return loose_int
    return True


def isshape(x, nonneg=False, allow_ndim=False) -> bool:
    """Is x a valid tuple of dimensions?

    If nonneg, also checks that the dimensions are non-negative.
    If allow_ndim, shapes of any dimensionality are allowed.
    """
    ndim = len(x)
    if not allow_ndim and ndim != 2:
        return False
    for d in x:
        if not isintlike(d):
            return False
        if nonneg and d < 0:
            return False
    return True


def issequence(t) -> bool:
    return ((isinstance(t, (list, tuple)) and
            (len(t) == 0 or np.isscalar(t[0]))) or
            (isinstance(t, np.ndarray) and (t.ndim == 1)))


def ismatrix(t) -> bool:
    return ((isinstance(t, (list, tuple)) and
             len(t) > 0 and issequence(t[0])) or
            (isinstance(t, np.ndarray) and t.ndim == 2))


def isdense(x) -> bool:
    return isinstance(x, np.ndarray)


def validateaxis(axis) -> None:
    if axis is None:
        return
    axis_type = type(axis)

    # In NumPy, you can pass in tuples for 'axis', but they are
    # not very useful for sparse matrices given their limited
    # dimensions, so let's make it explicit that they are not
    # allowed to be passed in
    if axis_type == tuple:
        raise TypeError("Tuples are not accepted for the 'axis' parameter. "
                        "Please pass in one of the following: "
                        "{-2, -1, 0, 1, None}.")

    # If not a tuple, check that the provided axis is actually
    # an integer and raise a TypeError similar to NumPy's
    if not np.issubdtype(np.dtype(axis_type), np.integer):
        raise TypeError(f"axis must be an integer, not {axis_type.__name__}")

    if not (-2 <= axis <= 1):
        raise ValueError("axis out of range")


def check_shape(args, current_shape=None):
    """Imitate numpy.matrix handling of shape arguments"""
    if len(args) == 0:
        raise TypeError("function missing 1 required positional argument: "
                        "'shape'")
    if len(args) == 1:
        try:
            shape_iter = iter(args[0])
        except TypeError:
            new_shape = (operator.index(args[0]), )
        else:
            new_shape = tuple(operator.index(arg) for arg in shape_iter)
    else:
        new_shape = tuple(operator.index(arg) for arg in args)

    if current_shape is None:
        if len(new_shape) != 2:
            raise ValueError('shape must be a 2-tuple of positive integers')
        elif any(d < 0 for d in new_shape):
            raise ValueError("'shape' elements cannot be negative")
    else:
        # Check the current size only if needed
        current_size = prod(current_shape)

        # Check for negatives
        negative_indexes = [i for i, x in enumerate(new_shape) if x < 0]
        if not negative_indexes:
            new_size = prod(new_shape)
            if new_size != current_size:
                raise ValueError('cannot reshape array of size {} into shape {}'
                                 .format(current_size, new_shape))
        elif len(negative_indexes) == 1:
            skip = negative_indexes[0]
            specified = prod(new_shape[:skip] + new_shape[skip+1:])
            unspecified, remainder = divmod(current_size, specified)
            if remainder != 0:
                err_shape = tuple('newshape' if x < 0 else x for x in new_shape)
                raise ValueError('cannot reshape array of size {} into shape {}'
                                 ''.format(current_size, err_shape))
            new_shape = new_shape[:skip] + (unspecified,) + new_shape[skip+1:]
        else:
            raise ValueError('can only specify one unknown dimension')

    if len(new_shape) != 2:
        raise ValueError('matrix shape must be two-dimensional')

    return new_shape


def check_reshape_kwargs(kwargs):
    """Unpack keyword arguments for reshape function.

    This is useful because keyword arguments after star arguments are not
    allowed in Python 2, but star keyword arguments are. This function unpacks
    'order' and 'copy' from the star keyword arguments (with defaults) and
    throws an error for any remaining.
    """

    order = kwargs.pop('order', 'C')
    copy = kwargs.pop('copy', False)
    if kwargs:  # Some unused kwargs remain
        raise TypeError('reshape() got unexpected keywords arguments: {}'
                        .format(', '.join(kwargs.keys())))
    return order, copy


def is_pydata_spmatrix(m) -> bool:
    """
    Check whether object is pydata/sparse matrix, avoiding importing the module.
    """
    base_cls = getattr(sys.modules.get('sparse'), 'SparseArray', None)
    return base_cls is not None and isinstance(m, base_cls)


###############################################################################
# Wrappers for NumPy types that are deprecated

# Numpy versions of these functions raise deprecation warnings, the
# ones below do not.

def matrix(*args, **kwargs):
    return np.array(*args, **kwargs).view(np.matrix)


def asmatrix(data, dtype=None):
    if isinstance(data, np.matrix) and (dtype is None or data.dtype == dtype):
        return data
    return np.asarray(data, dtype=dtype).view(np.matrix)

###############################################################################


def _todata(s) -> np.ndarray:
    """Access nonzero values, possibly after summing duplicates.

    Parameters
    ----------
    s : sparse array
        Input sparse array.

    Returns
    -------
    data: ndarray
      Nonzero values of the array, with shape (s.nnz,)

    """
    if isinstance(s, sp._data._data_matrix):
        return s._deduped_data()

    if isinstance(s, sp.dok_array):
        return np.fromiter(s.values(), dtype=s.dtype, count=s.nnz)

    if isinstance(s, sp.lil_array):
        data = np.empty(s.nnz, dtype=s.dtype)
        sp._csparsetools.lil_flatten_to_array(s.data, data)
        return data

    return s.tocoo()._deduped_data()
