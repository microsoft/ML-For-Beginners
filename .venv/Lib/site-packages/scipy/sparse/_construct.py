"""Functions to construct sparse matrices and arrays
"""

__docformat__ = "restructuredtext en"

__all__ = ['spdiags', 'eye', 'identity', 'kron', 'kronsum',
           'hstack', 'vstack', 'bmat', 'rand', 'random', 'diags', 'block_diag',
           'diags_array', 'block_array', 'eye_array', 'random_array']

import numbers
import math
import numpy as np

from scipy._lib._util import check_random_state, rng_integers
from ._sputils import upcast, get_index_dtype, isscalarlike

from ._sparsetools import csr_hstack
from ._bsr import bsr_matrix, bsr_array
from ._coo import coo_matrix, coo_array
from ._csc import csc_matrix, csc_array
from ._csr import csr_matrix, csr_array
from ._dia import dia_matrix, dia_array

from ._base import issparse, sparray


def spdiags(data, diags, m=None, n=None, format=None):
    """
    Return a sparse matrix from diagonals.

    Parameters
    ----------
    data : array_like
        Matrix diagonals stored row-wise
    diags : sequence of int or an int
        Diagonals to set:

        * k = 0  the main diagonal
        * k > 0  the kth upper diagonal
        * k < 0  the kth lower diagonal
    m, n : int, tuple, optional
        Shape of the result. If `n` is None and `m` is a given tuple,
        the shape is this tuple. If omitted, the matrix is square and
        its shape is len(data[0]).
    format : str, optional
        Format of the result. By default (format=None) an appropriate sparse
        matrix format is returned. This choice is subject to change.

    .. warning::

        This function returns a sparse matrix -- not a sparse array.
        You are encouraged to use ``diags_array`` to take advantage
        of the sparse array functionality.

    See Also
    --------
    diags_array : more convenient form of this function
    diags : matrix version of diags_array
    dia_matrix : the sparse DIAgonal format.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import spdiags
    >>> data = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
    >>> diags = np.array([0, -1, 2])
    >>> spdiags(data, diags, 4, 4).toarray()
    array([[1, 0, 3, 0],
           [1, 2, 0, 4],
           [0, 2, 3, 0],
           [0, 0, 3, 4]])

    """
    if m is None and n is None:
        m = n = len(data[0])
    elif n is None:
        m, n = m
    return dia_matrix((data, diags), shape=(m, n)).asformat(format)


def diags_array(diagonals, /, *, offsets=0, shape=None, format=None, dtype=None):
    """
    Construct a sparse array from diagonals.

    Parameters
    ----------
    diagonals : sequence of array_like
        Sequence of arrays containing the array diagonals,
        corresponding to `offsets`.
    offsets : sequence of int or an int, optional
        Diagonals to set:
          - k = 0  the main diagonal (default)
          - k > 0  the kth upper diagonal
          - k < 0  the kth lower diagonal
    shape : tuple of int, optional
        Shape of the result. If omitted, a square array large enough
        to contain the diagonals is returned.
    format : {"dia", "csr", "csc", "lil", ...}, optional
        Matrix format of the result. By default (format=None) an
        appropriate sparse array format is returned. This choice is
        subject to change.
    dtype : dtype, optional
        Data type of the array.

    Notes
    -----
    The result from `diags_array` is the sparse equivalent of::

        np.diag(diagonals[0], offsets[0])
        + ...
        + np.diag(diagonals[k], offsets[k])

    Repeated diagonal offsets are disallowed.

    .. versionadded:: 1.11

    Examples
    --------
    >>> from scipy.sparse import diags_array
    >>> diagonals = [[1, 2, 3, 4], [1, 2, 3], [1, 2]]
    >>> diags_array(diagonals, offsets=[0, -1, 2]).toarray()
    array([[1, 0, 1, 0],
           [1, 2, 0, 2],
           [0, 2, 3, 0],
           [0, 0, 3, 4]])

    Broadcasting of scalars is supported (but shape needs to be
    specified):

    >>> diags_array([1, -2, 1], offsets=[-1, 0, 1], shape=(4, 4)).toarray()
    array([[-2.,  1.,  0.,  0.],
           [ 1., -2.,  1.,  0.],
           [ 0.,  1., -2.,  1.],
           [ 0.,  0.,  1., -2.]])


    If only one diagonal is wanted (as in `numpy.diag`), the following
    works as well:

    >>> diags_array([1, 2, 3], offsets=1).toarray()
    array([[ 0.,  1.,  0.,  0.],
           [ 0.,  0.,  2.,  0.],
           [ 0.,  0.,  0.,  3.],
           [ 0.,  0.,  0.,  0.]])
    """
    # if offsets is not a sequence, assume that there's only one diagonal
    if isscalarlike(offsets):
        # now check that there's actually only one diagonal
        if len(diagonals) == 0 or isscalarlike(diagonals[0]):
            diagonals = [np.atleast_1d(diagonals)]
        else:
            raise ValueError("Different number of diagonals and offsets.")
    else:
        diagonals = list(map(np.atleast_1d, diagonals))

    offsets = np.atleast_1d(offsets)

    # Basic check
    if len(diagonals) != len(offsets):
        raise ValueError("Different number of diagonals and offsets.")

    # Determine shape, if omitted
    if shape is None:
        m = len(diagonals[0]) + abs(int(offsets[0]))
        shape = (m, m)

    # Determine data type, if omitted
    if dtype is None:
        dtype = np.common_type(*diagonals)

    # Construct data array
    m, n = shape

    M = max([min(m + offset, n - offset) + max(0, offset)
             for offset in offsets])
    M = max(0, M)
    data_arr = np.zeros((len(offsets), M), dtype=dtype)

    K = min(m, n)

    for j, diagonal in enumerate(diagonals):
        offset = offsets[j]
        k = max(0, offset)
        length = min(m + offset, n - offset, K)
        if length < 0:
            raise ValueError("Offset %d (index %d) out of bounds" % (offset, j))
        try:
            data_arr[j, k:k+length] = diagonal[...,:length]
        except ValueError as e:
            if len(diagonal) != length and len(diagonal) != 1:
                raise ValueError(
                    "Diagonal length (index %d: %d at offset %d) does not "
                    "agree with array size (%d, %d)." % (
                    j, len(diagonal), offset, m, n)) from e
            raise

    return dia_array((data_arr, offsets), shape=(m, n)).asformat(format)


def diags(diagonals, offsets=0, shape=None, format=None, dtype=None):
    """
    Construct a sparse matrix from diagonals.

    .. warning::

        This function returns a sparse matrix -- not a sparse array.
        You are encouraged to use ``diags_array`` to take advantage
        of the sparse array functionality.

    Parameters
    ----------
    diagonals : sequence of array_like
        Sequence of arrays containing the matrix diagonals,
        corresponding to `offsets`.
    offsets : sequence of int or an int, optional
        Diagonals to set:
          - k = 0  the main diagonal (default)
          - k > 0  the kth upper diagonal
          - k < 0  the kth lower diagonal
    shape : tuple of int, optional
        Shape of the result. If omitted, a square matrix large enough
        to contain the diagonals is returned.
    format : {"dia", "csr", "csc", "lil", ...}, optional
        Matrix format of the result. By default (format=None) an
        appropriate sparse matrix format is returned. This choice is
        subject to change.
    dtype : dtype, optional
        Data type of the matrix.

    See Also
    --------
    spdiags : construct matrix from diagonals
    diags_array : construct sparse array instead of sparse matrix

    Notes
    -----
    This function differs from `spdiags` in the way it handles
    off-diagonals.

    The result from `diags` is the sparse equivalent of::

        np.diag(diagonals[0], offsets[0])
        + ...
        + np.diag(diagonals[k], offsets[k])

    Repeated diagonal offsets are disallowed.

    .. versionadded:: 0.11

    Examples
    --------
    >>> from scipy.sparse import diags
    >>> diagonals = [[1, 2, 3, 4], [1, 2, 3], [1, 2]]
    >>> diags(diagonals, [0, -1, 2]).toarray()
    array([[1, 0, 1, 0],
           [1, 2, 0, 2],
           [0, 2, 3, 0],
           [0, 0, 3, 4]])

    Broadcasting of scalars is supported (but shape needs to be
    specified):

    >>> diags([1, -2, 1], [-1, 0, 1], shape=(4, 4)).toarray()
    array([[-2.,  1.,  0.,  0.],
           [ 1., -2.,  1.,  0.],
           [ 0.,  1., -2.,  1.],
           [ 0.,  0.,  1., -2.]])


    If only one diagonal is wanted (as in `numpy.diag`), the following
    works as well:

    >>> diags([1, 2, 3], 1).toarray()
    array([[ 0.,  1.,  0.,  0.],
           [ 0.,  0.,  2.,  0.],
           [ 0.,  0.,  0.,  3.],
           [ 0.,  0.,  0.,  0.]])
    """
    A = diags_array(diagonals, offsets=offsets, shape=shape, dtype=dtype)
    return dia_matrix(A).asformat(format)


def identity(n, dtype='d', format=None):
    """Identity matrix in sparse format

    Returns an identity matrix with shape (n,n) using a given
    sparse format and dtype. This differs from `eye_array` in
    that it has a square shape with ones only on the main diagonal.
    It is thus the multiplicative identity. `eye_array` allows
    rectangular shapes and the diagonal can be offset from the main one.

    .. warning::

        This function returns a sparse matrix -- not a sparse array.
        You are encouraged to use ``eye_array`` to take advantage
        of the sparse array functionality.

    Parameters
    ----------
    n : int
        Shape of the identity matrix.
    dtype : dtype, optional
        Data type of the matrix
    format : str, optional
        Sparse format of the result, e.g., format="csr", etc.

    Examples
    --------
    >>> import scipy as sp
    >>> sp.sparse.identity(3).toarray()
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    >>> sp.sparse.identity(3, dtype='int8', format='dia')
    <3x3 sparse matrix of type '<class 'numpy.int8'>'
            with 3 stored elements (1 diagonals) in DIAgonal format>
    >>> sp.sparse.eye_array(3, dtype='int8', format='dia')
    <3x3 sparse array of type '<class 'numpy.int8'>'
            with 3 stored elements (1 diagonals) in DIAgonal format>

    """
    return eye(n, n, dtype=dtype, format=format)


def eye_array(m, n=None, *, k=0, dtype=float, format=None):
    """Identity matrix in sparse array format

    Return a sparse array with ones on diagonal.
    Specifically a sparse array (m x n) where the kth diagonal
    is all ones and everything else is zeros.

    Parameters
    ----------
    m : int or tuple of ints
        Number of rows requested.
    n : int, optional
        Number of columns. Default: `m`.
    k : int, optional
        Diagonal to place ones on. Default: 0 (main diagonal).
    dtype : dtype, optional
        Data type of the array
    format : str, optional (default: "dia")
        Sparse format of the result, e.g., format="csr", etc.

    Examples
    --------
    >>> import numpy as np
    >>> import scipy as sp
    >>> sp.sparse.eye_array(3).toarray()
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    >>> sp.sparse.eye_array(3, dtype=np.int8)
    <3x3 sparse array of type '<class 'numpy.int8'>'
            with 3 stored elements (1 diagonals) in DIAgonal format>

    """
    # TODO: delete next 15 lines [combine with _eye()] once spmatrix removed
    return _eye(m, n, k, dtype, format)


def _eye(m, n, k, dtype, format, as_sparray=True):
    if as_sparray:
        csr_sparse = csr_array
        csc_sparse = csc_array
        coo_sparse = coo_array
        diags_sparse = diags_array
    else:
        csr_sparse = csr_matrix
        csc_sparse = csc_matrix
        coo_sparse = coo_matrix
        diags_sparse = diags

    if n is None:
        n = m
    m, n = int(m), int(n)

    if m == n and k == 0:
        # fast branch for special formats
        if format in ['csr', 'csc']:
            idx_dtype = get_index_dtype(maxval=n)
            indptr = np.arange(n+1, dtype=idx_dtype)
            indices = np.arange(n, dtype=idx_dtype)
            data = np.ones(n, dtype=dtype)
            cls = {'csr': csr_sparse, 'csc': csc_sparse}[format]
            return cls((data, indices, indptr), (n, n))

        elif format == 'coo':
            idx_dtype = get_index_dtype(maxval=n)
            row = np.arange(n, dtype=idx_dtype)
            col = np.arange(n, dtype=idx_dtype)
            data = np.ones(n, dtype=dtype)
            return coo_sparse((data, (row, col)), (n, n))

    data = np.ones((1, max(0, min(m + k, n))), dtype=dtype)
    return diags_sparse(data, offsets=[k], shape=(m, n), dtype=dtype).asformat(format)


def eye(m, n=None, k=0, dtype=float, format=None):
    """Sparse matrix with ones on diagonal

    Returns a sparse matrix (m x n) where the kth diagonal
    is all ones and everything else is zeros.

    Parameters
    ----------
    m : int
        Number of rows in the matrix.
    n : int, optional
        Number of columns. Default: `m`.
    k : int, optional
        Diagonal to place ones on. Default: 0 (main diagonal).
    dtype : dtype, optional
        Data type of the matrix.
    format : str, optional
        Sparse format of the result, e.g., format="csr", etc.

    .. warning::

        This function returns a sparse matrix -- not a sparse array.
        You are encouraged to use ``eye_array`` to take advantage
        of the sparse array functionality.

    Examples
    --------
    >>> import numpy as np
    >>> import scipy as sp
    >>> sp.sparse.eye(3).toarray()
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    >>> sp.sparse.eye(3, dtype=np.int8)
    <3x3 sparse matrix of type '<class 'numpy.int8'>'
        with 3 stored elements (1 diagonals) in DIAgonal format>

    """
    return _eye(m, n, k, dtype, format, False)


def kron(A, B, format=None):
    """kronecker product of sparse matrices A and B

    Parameters
    ----------
    A : sparse or dense matrix
        first matrix of the product
    B : sparse or dense matrix
        second matrix of the product
    format : str, optional (default: 'bsr' or 'coo')
        format of the result (e.g. "csr")
        If None, choose 'bsr' for relatively dense array and 'coo' for others

    Returns
    -------
    kronecker product in a sparse format.
    Returns a sparse matrix unless either A or B is a
    sparse array in which case returns a sparse array.

    Examples
    --------
    >>> import numpy as np
    >>> import scipy as sp
    >>> A = sp.sparse.csr_array(np.array([[0, 2], [5, 0]]))
    >>> B = sp.sparse.csr_array(np.array([[1, 2], [3, 4]]))
    >>> sp.sparse.kron(A, B).toarray()
    array([[ 0,  0,  2,  4],
           [ 0,  0,  6,  8],
           [ 5, 10,  0,  0],
           [15, 20,  0,  0]])

    >>> sp.sparse.kron(A, [[1, 2], [3, 4]]).toarray()
    array([[ 0,  0,  2,  4],
           [ 0,  0,  6,  8],
           [ 5, 10,  0,  0],
           [15, 20,  0,  0]])

    """
    # TODO: delete next 10 lines and replace _sparse with _array when spmatrix removed
    if isinstance(A, sparray) or isinstance(B, sparray):
        # convert to local variables
        bsr_sparse = bsr_array
        csr_sparse = csr_array
        coo_sparse = coo_array
    else:  # use spmatrix
        bsr_sparse = bsr_matrix
        csr_sparse = csr_matrix
        coo_sparse = coo_matrix

    B = coo_sparse(B)

    # B is fairly dense, use BSR
    if (format is None or format == "bsr") and 2*B.nnz >= B.shape[0] * B.shape[1]:
        A = csr_sparse(A,copy=True)
        output_shape = (A.shape[0]*B.shape[0], A.shape[1]*B.shape[1])

        if A.nnz == 0 or B.nnz == 0:
            # kronecker product is the zero matrix
            return coo_sparse(output_shape).asformat(format)

        B = B.toarray()
        data = A.data.repeat(B.size).reshape(-1,B.shape[0],B.shape[1])
        data = data * B

        return bsr_sparse((data,A.indices,A.indptr), shape=output_shape)
    else:
        # use COO
        A = coo_sparse(A)
        output_shape = (A.shape[0]*B.shape[0], A.shape[1]*B.shape[1])

        if A.nnz == 0 or B.nnz == 0:
            # kronecker product is the zero matrix
            return coo_sparse(output_shape).asformat(format)

        # expand entries of a into blocks
        row = A.row.repeat(B.nnz)
        col = A.col.repeat(B.nnz)
        data = A.data.repeat(B.nnz)

        if max(A.shape[0]*B.shape[0], A.shape[1]*B.shape[1]) > np.iinfo('int32').max:
            row = row.astype(np.int64)
            col = col.astype(np.int64)

        row *= B.shape[0]
        col *= B.shape[1]

        # increment block indices
        row,col = row.reshape(-1,B.nnz),col.reshape(-1,B.nnz)
        row += B.row
        col += B.col
        row,col = row.reshape(-1),col.reshape(-1)

        # compute block entries
        data = data.reshape(-1,B.nnz) * B.data
        data = data.reshape(-1)

        return coo_sparse((data,(row,col)), shape=output_shape).asformat(format)


def kronsum(A, B, format=None):
    """kronecker sum of square sparse matrices A and B

    Kronecker sum of two sparse matrices is a sum of two Kronecker
    products kron(I_n,A) + kron(B,I_m) where A has shape (m,m)
    and B has shape (n,n) and I_m and I_n are identity matrices
    of shape (m,m) and (n,n), respectively.

    Parameters
    ----------
    A
        square matrix
    B
        square matrix
    format : str
        format of the result (e.g. "csr")

    Returns
    -------
    kronecker sum in a sparse matrix format

    """
    # TODO: delete next 8 lines and replace _sparse with _array when spmatrix removed
    if isinstance(A, sparray) or isinstance(B, sparray):
        # convert to local variables
        coo_sparse = coo_array
        identity_sparse = eye_array
    else:
        coo_sparse = coo_matrix
        identity_sparse = identity

    A = coo_sparse(A)
    B = coo_sparse(B)

    if A.shape[0] != A.shape[1]:
        raise ValueError('A is not square')

    if B.shape[0] != B.shape[1]:
        raise ValueError('B is not square')

    dtype = upcast(A.dtype, B.dtype)

    I_n = identity_sparse(A.shape[0], dtype=dtype)
    I_m = identity_sparse(B.shape[0], dtype=dtype)
    L = kron(I_m, A, format='coo')
    R = kron(B, I_n, format='coo')

    return (L + R).asformat(format)


def _compressed_sparse_stack(blocks, axis, return_spmatrix):
    """
    Stacking fast path for CSR/CSC matrices or arrays
    (i) vstack for CSR, (ii) hstack for CSC.
    """
    other_axis = 1 if axis == 0 else 0
    data = np.concatenate([b.data for b in blocks])
    constant_dim = blocks[0].shape[other_axis]
    idx_dtype = get_index_dtype(arrays=[b.indptr for b in blocks],
                                maxval=max(data.size, constant_dim))
    indices = np.empty(data.size, dtype=idx_dtype)
    indptr = np.empty(sum(b.shape[axis] for b in blocks) + 1, dtype=idx_dtype)
    last_indptr = idx_dtype(0)
    sum_dim = 0
    sum_indices = 0
    for b in blocks:
        if b.shape[other_axis] != constant_dim:
            raise ValueError(f'incompatible dimensions for axis {other_axis}')
        indices[sum_indices:sum_indices+b.indices.size] = b.indices
        sum_indices += b.indices.size
        idxs = slice(sum_dim, sum_dim + b.shape[axis])
        indptr[idxs] = b.indptr[:-1]
        indptr[idxs] += last_indptr
        sum_dim += b.shape[axis]
        last_indptr += b.indptr[-1]
    indptr[-1] = last_indptr
    # TODO remove this if-structure when sparse matrices removed
    if return_spmatrix:
        if axis == 0:
            return csr_matrix((data, indices, indptr),
                              shape=(sum_dim, constant_dim))
        else:
            return csc_matrix((data, indices, indptr),
                              shape=(constant_dim, sum_dim))

    if axis == 0:
        return csr_array((data, indices, indptr),
                          shape=(sum_dim, constant_dim))
    else:
        return csc_array((data, indices, indptr),
                          shape=(constant_dim, sum_dim))


def _stack_along_minor_axis(blocks, axis):
    """
    Stacking fast path for CSR/CSC matrices along the minor axis
    (i) hstack for CSR, (ii) vstack for CSC.
    """
    n_blocks = len(blocks)
    if n_blocks == 0:
        raise ValueError('Missing block matrices')

    if n_blocks == 1:
        return blocks[0]

    # check for incompatible dimensions
    other_axis = 1 if axis == 0 else 0
    other_axis_dims = {b.shape[other_axis] for b in blocks}
    if len(other_axis_dims) > 1:
        raise ValueError(f'Mismatching dimensions along axis {other_axis}: '
                         f'{other_axis_dims}')
    constant_dim, = other_axis_dims

    # Do the stacking
    indptr_list = [b.indptr for b in blocks]
    data_cat = np.concatenate([b.data for b in blocks])

    # Need to check if any indices/indptr, would be too large post-
    # concatenation for np.int32:
    # - The max value of indices is the output array's stacking-axis length - 1
    # - The max value in indptr is the number of non-zero entries. This is
    #   exceedingly unlikely to require int64, but is checked out of an
    #   abundance of caution.
    sum_dim = sum(b.shape[axis] for b in blocks)
    nnz = sum(len(b.indices) for b in blocks)
    idx_dtype = get_index_dtype(maxval=max(sum_dim - 1, nnz))
    stack_dim_cat = np.array([b.shape[axis] for b in blocks], dtype=idx_dtype)
    if data_cat.size > 0:
        indptr_cat = np.concatenate(indptr_list).astype(idx_dtype)
        indices_cat = (np.concatenate([b.indices for b in blocks])
                       .astype(idx_dtype))
        indptr = np.empty(constant_dim + 1, dtype=idx_dtype)
        indices = np.empty_like(indices_cat)
        data = np.empty_like(data_cat)
        csr_hstack(n_blocks, constant_dim, stack_dim_cat,
                   indptr_cat, indices_cat, data_cat,
                   indptr, indices, data)
    else:
        indptr = np.zeros(constant_dim + 1, dtype=idx_dtype)
        indices = np.empty(0, dtype=idx_dtype)
        data = np.empty(0, dtype=data_cat.dtype)

    if axis == 0:
        return blocks[0]._csc_container((data, indices, indptr),
                          shape=(sum_dim, constant_dim))
    else:
        return blocks[0]._csr_container((data, indices, indptr),
                          shape=(constant_dim, sum_dim))


def hstack(blocks, format=None, dtype=None):
    """
    Stack sparse matrices horizontally (column wise)

    Parameters
    ----------
    blocks
        sequence of sparse matrices with compatible shapes
    format : str
        sparse format of the result (e.g., "csr")
        by default an appropriate sparse matrix format is returned.
        This choice is subject to change.
    dtype : dtype, optional
        The data-type of the output matrix. If not given, the dtype is
        determined from that of `blocks`.

    Returns
    -------
    new_array : sparse matrix or array
        If any block in blocks is a sparse array, return a sparse array.
        Otherwise return a sparse matrix.

        If you want a sparse array built from blocks that are not sparse
        arrays, use `block(hstack(blocks))` or convert one block
        e.g. `blocks[0] = csr_array(blocks[0])`.

    See Also
    --------
    vstack : stack sparse matrices vertically (row wise)

    Examples
    --------
    >>> from scipy.sparse import coo_matrix, hstack
    >>> A = coo_matrix([[1, 2], [3, 4]])
    >>> B = coo_matrix([[5], [6]])
    >>> hstack([A,B]).toarray()
    array([[1, 2, 5],
           [3, 4, 6]])

    """
    blocks = np.asarray(blocks, dtype='object')
    if any(isinstance(b, sparray) for b in blocks.flat):
        return _block([blocks], format, dtype)
    else:
        return _block([blocks], format, dtype, return_spmatrix=True)


def vstack(blocks, format=None, dtype=None):
    """
    Stack sparse arrays vertically (row wise)

    Parameters
    ----------
    blocks
        sequence of sparse arrays with compatible shapes
    format : str, optional
        sparse format of the result (e.g., "csr")
        by default an appropriate sparse array format is returned.
        This choice is subject to change.
    dtype : dtype, optional
        The data-type of the output array. If not given, the dtype is
        determined from that of `blocks`.

    Returns
    -------
    new_array : sparse matrix or array
        If any block in blocks is a sparse array, return a sparse array.
        Otherwise return a sparse matrix.

        If you want a sparse array built from blocks that are not sparse
        arrays, use `block(vstack(blocks))` or convert one block
        e.g. `blocks[0] = csr_array(blocks[0])`.

    See Also
    --------
    hstack : stack sparse matrices horizontally (column wise)

    Examples
    --------
    >>> from scipy.sparse import coo_array, vstack
    >>> A = coo_array([[1, 2], [3, 4]])
    >>> B = coo_array([[5, 6]])
    >>> vstack([A, B]).toarray()
    array([[1, 2],
           [3, 4],
           [5, 6]])

    """
    blocks = np.asarray(blocks, dtype='object')
    if any(isinstance(b, sparray) for b in blocks.flat):
        return _block([[b] for b in blocks], format, dtype)
    else:
        return _block([[b] for b in blocks], format, dtype, return_spmatrix=True)


def bmat(blocks, format=None, dtype=None):
    """
    Build a sparse array or matrix from sparse sub-blocks

    Note: `block_array` is preferred over `bmat`. They are the same function
    except that `bmat` can return a deprecated sparse matrix.
    `bmat` returns a coo_matrix if none of the inputs are a sparse array.

    .. warning::

        This function returns a sparse matrix -- not a sparse array.
        You are encouraged to use ``block_array`` to take advantage
        of the sparse array functionality.

    Parameters
    ----------
    blocks : array_like
        Grid of sparse matrices with compatible shapes.
        An entry of None implies an all-zero matrix.
    format : {'bsr', 'coo', 'csc', 'csr', 'dia', 'dok', 'lil'}, optional
        The sparse format of the result (e.g. "csr"). By default an
        appropriate sparse matrix format is returned.
        This choice is subject to change.
    dtype : dtype, optional
        The data-type of the output matrix. If not given, the dtype is
        determined from that of `blocks`.

    Returns
    -------
    bmat : sparse matrix or array
        If any block in blocks is a sparse array, return a sparse array.
        Otherwise return a sparse matrix.

        If you want a sparse array built from blocks that are not sparse
        arrays, use `block_array()`.

    See Also
    --------
    block_array

    Examples
    --------
    >>> from scipy.sparse import coo_array, bmat
    >>> A = coo_array([[1, 2], [3, 4]])
    >>> B = coo_array([[5], [6]])
    >>> C = coo_array([[7]])
    >>> bmat([[A, B], [None, C]]).toarray()
    array([[1, 2, 5],
           [3, 4, 6],
           [0, 0, 7]])

    >>> bmat([[A, None], [None, C]]).toarray()
    array([[1, 2, 0],
           [3, 4, 0],
           [0, 0, 7]])

    """
    blocks = np.asarray(blocks, dtype='object')
    if any(isinstance(b, sparray) for b in blocks.flat):
        return _block(blocks, format, dtype)
    else:
        return _block(blocks, format, dtype, return_spmatrix=True)


def block_array(blocks, *, format=None, dtype=None):
    """
    Build a sparse array from sparse sub-blocks

    Parameters
    ----------
    blocks : array_like
        Grid of sparse arrays with compatible shapes.
        An entry of None implies an all-zero array.
    format : {'bsr', 'coo', 'csc', 'csr', 'dia', 'dok', 'lil'}, optional
        The sparse format of the result (e.g. "csr"). By default an
        appropriate sparse array format is returned.
        This choice is subject to change.
    dtype : dtype, optional
        The data-type of the output array. If not given, the dtype is
        determined from that of `blocks`.

    Returns
    -------
    block : sparse array

    See Also
    --------
    block_diag : specify blocks along the main diagonals
    diags : specify (possibly offset) diagonals

    Examples
    --------
    >>> from scipy.sparse import coo_array, block_array
    >>> A = coo_array([[1, 2], [3, 4]])
    >>> B = coo_array([[5], [6]])
    >>> C = coo_array([[7]])
    >>> block_array([[A, B], [None, C]]).toarray()
    array([[1, 2, 5],
           [3, 4, 6],
           [0, 0, 7]])

    >>> block_array([[A, None], [None, C]]).toarray()
    array([[1, 2, 0],
           [3, 4, 0],
           [0, 0, 7]])

    """
    return _block(blocks, format, dtype)


def _block(blocks, format, dtype, return_spmatrix=False):
    blocks = np.asarray(blocks, dtype='object')

    if blocks.ndim != 2:
        raise ValueError('blocks must be 2-D')

    M,N = blocks.shape

    # check for fast path cases
    if (format in (None, 'csr') and
        all(issparse(b) and b.format == 'csr' for b in blocks.flat)
    ):
        if N > 1:
            # stack along columns (axis 1): must have shape (M, 1)
            blocks = [[_stack_along_minor_axis(blocks[b, :], 1)] for b in range(M)]
            blocks = np.asarray(blocks, dtype='object')

        # stack along rows (axis 0):
        A = _compressed_sparse_stack(blocks[:, 0], 0, return_spmatrix)
        if dtype is not None:
            A = A.astype(dtype)
        return A
    elif (format in (None, 'csc') and
          all(issparse(b) and b.format == 'csc' for b in blocks.flat)
    ):
        if M > 1:
            # stack along rows (axis 0): must have shape (1, N)
            blocks = [[_stack_along_minor_axis(blocks[:, b], 0) for b in range(N)]]
            blocks = np.asarray(blocks, dtype='object')

        # stack along columns (axis 1):
        A = _compressed_sparse_stack(blocks[0, :], 1, return_spmatrix)
        if dtype is not None:
            A = A.astype(dtype)
        return A

    block_mask = np.zeros(blocks.shape, dtype=bool)
    brow_lengths = np.zeros(M, dtype=np.int64)
    bcol_lengths = np.zeros(N, dtype=np.int64)

    # convert everything to COO format
    for i in range(M):
        for j in range(N):
            if blocks[i,j] is not None:
                A = coo_array(blocks[i,j])
                blocks[i,j] = A
                block_mask[i,j] = True

                if brow_lengths[i] == 0:
                    brow_lengths[i] = A.shape[0]
                elif brow_lengths[i] != A.shape[0]:
                    msg = (f'blocks[{i},:] has incompatible row dimensions. '
                           f'Got blocks[{i},{j}].shape[0] == {A.shape[0]}, '
                           f'expected {brow_lengths[i]}.')
                    raise ValueError(msg)

                if bcol_lengths[j] == 0:
                    bcol_lengths[j] = A.shape[1]
                elif bcol_lengths[j] != A.shape[1]:
                    msg = (f'blocks[:,{j}] has incompatible column '
                           f'dimensions. '
                           f'Got blocks[{i},{j}].shape[1] == {A.shape[1]}, '
                           f'expected {bcol_lengths[j]}.')
                    raise ValueError(msg)

    nnz = sum(block.nnz for block in blocks[block_mask])
    if dtype is None:
        all_dtypes = [blk.dtype for blk in blocks[block_mask]]
        dtype = upcast(*all_dtypes) if all_dtypes else None

    row_offsets = np.append(0, np.cumsum(brow_lengths))
    col_offsets = np.append(0, np.cumsum(bcol_lengths))

    shape = (row_offsets[-1], col_offsets[-1])

    data = np.empty(nnz, dtype=dtype)
    idx_dtype = get_index_dtype(maxval=max(shape))
    row = np.empty(nnz, dtype=idx_dtype)
    col = np.empty(nnz, dtype=idx_dtype)

    nnz = 0
    ii, jj = np.nonzero(block_mask)
    for i, j in zip(ii, jj):
        B = blocks[i, j]
        idx = slice(nnz, nnz + B.nnz)
        data[idx] = B.data
        np.add(B.row, row_offsets[i], out=row[idx], dtype=idx_dtype)
        np.add(B.col, col_offsets[j], out=col[idx], dtype=idx_dtype)
        nnz += B.nnz

    if return_spmatrix:
        return coo_matrix((data, (row, col)), shape=shape).asformat(format)
    return coo_array((data, (row, col)), shape=shape).asformat(format)


def block_diag(mats, format=None, dtype=None):
    """
    Build a block diagonal sparse matrix or array from provided matrices.

    Parameters
    ----------
    mats : sequence of matrices or arrays
        Input matrices or arrays.
    format : str, optional
        The sparse format of the result (e.g., "csr"). If not given, the result
        is returned in "coo" format.
    dtype : dtype specifier, optional
        The data-type of the output. If not given, the dtype is
        determined from that of `blocks`.

    Returns
    -------
    res : sparse matrix or array
        If at least one input is a sparse array, the output is a sparse array.
        Otherwise the output is a sparse matrix.

    Notes
    -----

    .. versionadded:: 0.11.0

    See Also
    --------
    block_array
    diags_array

    Examples
    --------
    >>> from scipy.sparse import coo_array, block_diag
    >>> A = coo_array([[1, 2], [3, 4]])
    >>> B = coo_array([[5], [6]])
    >>> C = coo_array([[7]])
    >>> block_diag((A, B, C)).toarray()
    array([[1, 2, 0, 0],
           [3, 4, 0, 0],
           [0, 0, 5, 0],
           [0, 0, 6, 0],
           [0, 0, 0, 7]])

    """
    if any(isinstance(a, sparray) for a in mats):
        container = coo_array
    else:
        container = coo_matrix

    row = []
    col = []
    data = []
    r_idx = 0
    c_idx = 0
    for a in mats:
        if isinstance(a, (list, numbers.Number)):
            a = coo_array(a)
        nrows, ncols = a.shape
        if issparse(a):
            a = a.tocoo()
            row.append(a.row + r_idx)
            col.append(a.col + c_idx)
            data.append(a.data)
        else:
            a_row, a_col = np.divmod(np.arange(nrows*ncols), ncols)
            row.append(a_row + r_idx)
            col.append(a_col + c_idx)
            data.append(a.ravel())
        r_idx += nrows
        c_idx += ncols
    row = np.concatenate(row)
    col = np.concatenate(col)
    data = np.concatenate(data)
    return container((data, (row, col)),
                      shape=(r_idx, c_idx),
                      dtype=dtype).asformat(format)


def random_array(shape, *, density=0.01, format='coo', dtype=None,
                 random_state=None, data_sampler=None):
    """Return a sparse array of uniformly random numbers in [0, 1)

    Returns a sparse array with the given shape and density
    where values are generated uniformly randomly in the range [0, 1).

    .. warning::

        Since numpy 1.17, passing a ``np.random.Generator`` (e.g.
        ``np.random.default_rng``) for ``random_state`` will lead to much
        faster execution times.

        A much slower implementation is used by default for backwards
        compatibility.

    Parameters
    ----------
    shape : int or tuple of ints
        shape of the array
    density : real, optional (default: 0.01)
        density of the generated matrix: density equal to one means a full
        matrix, density of 0 means a matrix with no non-zero items.
    format : str, optional (default: 'coo')
        sparse matrix format.
    dtype : dtype, optional (default: np.float64)
        type of the returned matrix values.
    random_state : {None, int, `Generator`, `RandomState`}, optional
        A random number generator to determine nonzero structure. We recommend using
        a `numpy.random.Generator` manually provided for every call as it is much
        faster than RandomState.

        - If `None` (or `np.random`), the `numpy.random.RandomState`
          singleton is used.
        - If an int, a new ``Generator`` instance is used,
          seeded with the int.
        - If a ``Generator`` or ``RandomState`` instance then
          that instance is used.

        This random state will be used for sampling `indices` (the sparsity
        structure), and by default for the data values too (see `data_sampler`).

    data_sampler : callable, optional (default depends on dtype)
        Sampler of random data values with keyword arg `size`.
        This function should take a single keyword argument `size` specifying
        the length of its returned ndarray. It is used to generate the nonzero
        values in the matrix after the locations of those values are chosen.
        By default, uniform [0, 1) random values are used unless `dtype` is
        an integer (default uniform integers from that dtype) or
        complex (default uniform over the unit square in the complex plane).
        For these, the `random_state` rng is used e.g. `rng.uniform(size=size)`.

    Returns
    -------
    res : sparse array

    Examples
    --------

    Passing a ``np.random.Generator`` instance for better performance:

    >>> import numpy as np
    >>> import scipy as sp
    >>> rng = np.random.default_rng()

    Default sampling uniformly from [0, 1):

    >>> S = sp.sparse.random_array((3, 4), density=0.25, random_state=rng)

    Providing a sampler for the values:

    >>> rvs = sp.stats.poisson(25, loc=10).rvs
    >>> S = sp.sparse.random_array((3, 4), density=0.25,
    ...                            random_state=rng, data_sampler=rvs)
    >>> S.toarray()
    array([[ 36.,   0.,  33.,   0.],   # random
           [  0.,   0.,   0.,   0.],
           [  0.,   0.,  36.,   0.]])

    Building a custom distribution.
    This example builds a squared normal from np.random:

    >>> def np_normal_squared(size=None, random_state=rng):
    ...     return random_state.standard_normal(size) ** 2
    >>> S = sp.sparse.random_array((3, 4), density=0.25, random_state=rng,
    ...                      data_sampler=np_normal_squared)

    Or we can build it from sp.stats style rvs functions:

    >>> def sp_stats_normal_squared(size=None, random_state=rng):
    ...     std_normal = sp.stats.distributions.norm_gen().rvs
    ...     return std_normal(size=size, random_state=random_state) ** 2
    >>> S = sp.sparse.random_array((3, 4), density=0.25, random_state=rng,
    ...                      data_sampler=sp_stats_normal_squared)

    Or we can subclass sp.stats rv_continous or rv_discrete:

    >>> class NormalSquared(sp.stats.rv_continuous):
    ...     def _rvs(self,  size=None, random_state=rng):
    ...         return random_state.standard_normal(size) ** 2
    >>> X = NormalSquared()
    >>> Y = X().rvs
    >>> S = sp.sparse.random_array((3, 4), density=0.25,
    ...                            random_state=rng, data_sampler=Y)
    """
    # Use the more efficient RNG by default.
    if random_state is None:
        random_state = np.random.default_rng()
    data, ind = _random(shape, density, format, dtype, random_state, data_sampler)
    return coo_array((data, ind), shape=shape).asformat(format)


def _random(shape, density=0.01, format=None, dtype=None,
            random_state=None, data_sampler=None):
    if density < 0 or density > 1:
        raise ValueError("density expected to be 0 <= density <= 1")

    tot_prod = math.prod(shape)  # use `math` for when prod is >= 2**64

    # Number of non zero values
    size = int(round(density * tot_prod))

    rng = check_random_state(random_state)

    if data_sampler is None:
        if np.issubdtype(dtype, np.integer):
            def data_sampler(size):
                return rng_integers(rng,
                                    np.iinfo(dtype).min,
                                    np.iinfo(dtype).max,
                                    size,
                                    dtype=dtype)
        elif np.issubdtype(dtype, np.complexfloating):
            def data_sampler(size):
                return (rng.uniform(size=size) +
                        rng.uniform(size=size) * 1j)
        else:
            data_sampler = rng.uniform

    # rng.choice uses int64 if first arg is an int
    if tot_prod < np.iinfo(np.int64).max:
        raveled_ind = rng.choice(tot_prod, size=size, replace=False)
        ind = np.unravel_index(raveled_ind, shape=shape)
    else:
        # for ravel indices bigger than dtype max, use sets to remove duplicates
        ndim = len(shape)
        seen = set()
        while len(seen) < size:
            dsize = size - len(seen)
            seen.update(map(tuple, rng_integers(rng, shape, size=(dsize, ndim))))
        ind = tuple(np.array(list(seen)).T)

    # size kwarg allows eg data_sampler=partial(np.random.poisson, lam=5)
    vals = data_sampler(size=size).astype(dtype, copy=False)
    return vals, ind


def random(m, n, density=0.01, format='coo', dtype=None,
           random_state=None, data_rvs=None):
    """Generate a sparse matrix of the given shape and density with randomly
    distributed values.

    .. warning::

        Since numpy 1.17, passing a ``np.random.Generator`` (e.g.
        ``np.random.default_rng``) for ``random_state`` will lead to much
        faster execution times.

        A much slower implementation is used by default for backwards
        compatibility.

    .. warning::

        This function returns a sparse matrix -- not a sparse array.
        You are encouraged to use ``random_array`` to take advantage of the
        sparse array functionality.

    Parameters
    ----------
    m, n : int
        shape of the matrix
    density : real, optional
        density of the generated matrix: density equal to one means a full
        matrix, density of 0 means a matrix with no non-zero items.
    format : str, optional
        sparse matrix format.
    dtype : dtype, optional
        type of the returned matrix values.
    random_state : {None, int, `numpy.random.Generator`,
                    `numpy.random.RandomState`}, optional

        - If `seed` is None (or `np.random`), the `numpy.random.RandomState`
          singleton is used.
        - If `seed` is an int, a new ``RandomState`` instance is used,
          seeded with `seed`.
        - If `seed` is already a ``Generator`` or ``RandomState`` instance then
          that instance is used.

        This random state will be used for sampling the sparsity structure, but
        not necessarily for sampling the values of the structurally nonzero
        entries of the matrix.
    data_rvs : callable, optional
        Samples a requested number of random values.
        This function should take a single argument specifying the length
        of the ndarray that it will return. The structurally nonzero entries
        of the sparse random matrix will be taken from the array sampled
        by this function. By default, uniform [0, 1) random values will be
        sampled using the same random state as is used for sampling
        the sparsity structure.

    Returns
    -------
    res : sparse matrix

    See Also
    --------
    random_array : constructs sparse arrays instead of sparse matrices

    Examples
    --------

    Passing a ``np.random.Generator`` instance for better performance:

    >>> import scipy as sp
    >>> import numpy as np
    >>> rng = np.random.default_rng()
    >>> S = sp.sparse.random(3, 4, density=0.25, random_state=rng)

    Providing a sampler for the values:

    >>> rvs = sp.stats.poisson(25, loc=10).rvs
    >>> S = sp.sparse.random(3, 4, density=0.25, random_state=rng, data_rvs=rvs)
    >>> S.toarray()
    array([[ 36.,   0.,  33.,   0.],   # random
           [  0.,   0.,   0.,   0.],
           [  0.,   0.,  36.,   0.]])

    Building a custom distribution.
    This example builds a squared normal from np.random:

    >>> def np_normal_squared(size=None, random_state=rng):
    ...     return random_state.standard_normal(size) ** 2
    >>> S = sp.sparse.random(3, 4, density=0.25, random_state=rng,
    ...                      data_rvs=np_normal_squared)

    Or we can build it from sp.stats style rvs functions:

    >>> def sp_stats_normal_squared(size=None, random_state=rng):
    ...     std_normal = sp.stats.distributions.norm_gen().rvs
    ...     return std_normal(size=size, random_state=random_state) ** 2
    >>> S = sp.sparse.random(3, 4, density=0.25, random_state=rng,
    ...                      data_rvs=sp_stats_normal_squared)

    Or we can subclass sp.stats rv_continous or rv_discrete:

    >>> class NormalSquared(sp.stats.rv_continuous):
    ...     def _rvs(self,  size=None, random_state=rng):
    ...         return random_state.standard_normal(size) ** 2
    >>> X = NormalSquared()
    >>> Y = X()  # get a frozen version of the distribution
    >>> S = sp.sparse.random(3, 4, density=0.25, random_state=rng, data_rvs=Y.rvs)
    """
    if n is None:
        n = m
    m, n = int(m), int(n)
    # make keyword syntax work for data_rvs e.g. data_rvs(size=7)
    if data_rvs is not None:
        def data_rvs_kw(size):
            return data_rvs(size)
    else:
        data_rvs_kw = None
    vals, ind = _random((m, n), density, format, dtype, random_state, data_rvs_kw)
    return coo_matrix((vals, ind), shape=(m, n)).asformat(format)


def rand(m, n, density=0.01, format="coo", dtype=None, random_state=None):
    """Generate a sparse matrix of the given shape and density with uniformly
    distributed values.

    .. warning::

        This function returns a sparse matrix -- not a sparse array.
        You are encouraged to use ``random_array`` to take advantage
        of the sparse array functionality.

    Parameters
    ----------
    m, n : int
        shape of the matrix
    density : real, optional
        density of the generated matrix: density equal to one means a full
        matrix, density of 0 means a matrix with no non-zero items.
    format : str, optional
        sparse matrix format.
    dtype : dtype, optional
        type of the returned matrix values.
    random_state : {None, int, `numpy.random.Generator`,
                    `numpy.random.RandomState`}, optional

        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.

    Returns
    -------
    res : sparse matrix

    Notes
    -----
    Only float types are supported for now.

    See Also
    --------
    random : Similar function allowing a custom random data sampler
    random_array : Similar to random() but returns a sparse array

    Examples
    --------
    >>> from scipy.sparse import rand
    >>> matrix = rand(3, 4, density=0.25, format="csr", random_state=42)
    >>> matrix
    <3x4 sparse matrix of type '<class 'numpy.float64'>'
       with 3 stored elements in Compressed Sparse Row format>
    >>> matrix.toarray()
    array([[0.05641158, 0.        , 0.        , 0.65088847],  # random
           [0.        , 0.        , 0.        , 0.14286682],
           [0.        , 0.        , 0.        , 0.        ]])

    """
    return random(m, n, density, format, dtype, random_state)
