"""Compressed Block Sparse Row format"""

__docformat__ = "restructuredtext en"

__all__ = ['bsr_array', 'bsr_matrix', 'isspmatrix_bsr']

from warnings import warn

import numpy as np

from ._matrix import spmatrix
from ._data import _data_matrix, _minmax_mixin
from ._compressed import _cs_matrix
from ._base import issparse, _formats, _spbase, sparray
from ._sputils import (isshape, getdtype, getdata, to_native, upcast,
                       check_shape)
from . import _sparsetools
from ._sparsetools import (bsr_matvec, bsr_matvecs, csr_matmat_maxnnz,
                           bsr_matmat, bsr_transpose, bsr_sort_indices,
                           bsr_tocsr)


class _bsr_base(_cs_matrix, _minmax_mixin):
    _format = 'bsr'

    def __init__(self, arg1, shape=None, dtype=None, copy=False, blocksize=None):
        _data_matrix.__init__(self)

        if issparse(arg1):
            if arg1.format == self.format and copy:
                arg1 = arg1.copy()
            else:
                arg1 = arg1.tobsr(blocksize=blocksize)
            self._set_self(arg1)

        elif isinstance(arg1,tuple):
            if isshape(arg1):
                # it's a tuple of matrix dimensions (M,N)
                self._shape = check_shape(arg1)
                M,N = self.shape
                # process blocksize
                if blocksize is None:
                    blocksize = (1,1)
                else:
                    if not isshape(blocksize):
                        raise ValueError('invalid blocksize=%s' % blocksize)
                    blocksize = tuple(blocksize)
                self.data = np.zeros((0,) + blocksize, getdtype(dtype, default=float))

                R,C = blocksize
                if (M % R) != 0 or (N % C) != 0:
                    raise ValueError('shape must be multiple of blocksize')

                # Select index dtype large enough to pass array and
                # scalar parameters to sparsetools
                idx_dtype = self._get_index_dtype(maxval=max(M//R, N//C, R, C))
                self.indices = np.zeros(0, dtype=idx_dtype)
                self.indptr = np.zeros(M//R + 1, dtype=idx_dtype)

            elif len(arg1) == 2:
                # (data,(row,col)) format
                self._set_self(
                    self._coo_container(arg1, dtype=dtype, shape=shape).tobsr(
                        blocksize=blocksize
                    )
                )

            elif len(arg1) == 3:
                # (data,indices,indptr) format
                (data, indices, indptr) = arg1

                # Select index dtype large enough to pass array and
                # scalar parameters to sparsetools
                maxval = 1
                if shape is not None:
                    maxval = max(shape)
                if blocksize is not None:
                    maxval = max(maxval, max(blocksize))
                idx_dtype = self._get_index_dtype((indices, indptr), maxval=maxval,
                                                  check_contents=True)
                self.indices = np.array(indices, copy=copy, dtype=idx_dtype)
                self.indptr = np.array(indptr, copy=copy, dtype=idx_dtype)
                self.data = getdata(data, copy=copy, dtype=dtype)
                if self.data.ndim != 3:
                    raise ValueError(
                        f'BSR data must be 3-dimensional, got shape={self.data.shape}'
                    )
                if blocksize is not None:
                    if not isshape(blocksize):
                        raise ValueError(f'invalid blocksize={blocksize}')
                    if tuple(blocksize) != self.data.shape[1:]:
                        raise ValueError('mismatching blocksize={} vs {}'.format(
                            blocksize, self.data.shape[1:]))
            else:
                raise ValueError('unrecognized bsr_array constructor usage')
        else:
            # must be dense
            try:
                arg1 = np.asarray(arg1)
            except Exception as e:
                raise ValueError("unrecognized form for"
                        " %s_matrix constructor" % self.format) from e
            arg1 = self._coo_container(
                arg1, dtype=dtype
            ).tobsr(blocksize=blocksize)
            self._set_self(arg1)

        if shape is not None:
            self._shape = check_shape(shape)
        else:
            if self.shape is None:
                # shape not already set, try to infer dimensions
                try:
                    M = len(self.indptr) - 1
                    N = self.indices.max() + 1
                except Exception as e:
                    raise ValueError('unable to infer matrix dimensions') from e
                else:
                    R,C = self.blocksize
                    self._shape = check_shape((M*R,N*C))

        if self.shape is None:
            if shape is None:
                # TODO infer shape here
                raise ValueError('need to infer shape')
            else:
                self._shape = check_shape(shape)

        if dtype is not None:
            self.data = self.data.astype(dtype, copy=False)

        self.check_format(full_check=False)

    def check_format(self, full_check=True):
        """Check whether the array/matrix respects the BSR format.

        Parameters
        ----------
        full_check : bool, optional
            If `True`, run rigorous check, scanning arrays for valid values.
            Note that activating those check might copy arrays for casting,
            modifying indices and index pointers' inplace.
            If `False`, run basic checks on attributes. O(1) operations.
            Default is `True`.
        """
        M,N = self.shape
        R,C = self.blocksize

        # index arrays should have integer data types
        if self.indptr.dtype.kind != 'i':
            warn(f"indptr array has non-integer dtype ({self.indptr.dtype.name})",
                 stacklevel=2)
        if self.indices.dtype.kind != 'i':
            warn(f"indices array has non-integer dtype ({self.indices.dtype.name})",
                 stacklevel=2)

        # check array shapes
        if self.indices.ndim != 1 or self.indptr.ndim != 1:
            raise ValueError("indices, and indptr should be 1-D")
        if self.data.ndim != 3:
            raise ValueError("data should be 3-D")

        # check index pointer
        if (len(self.indptr) != M//R + 1):
            raise ValueError("index pointer size (%d) should be (%d)" %
                                (len(self.indptr), M//R + 1))
        if (self.indptr[0] != 0):
            raise ValueError("index pointer should start with 0")

        # check index and data arrays
        if (len(self.indices) != len(self.data)):
            raise ValueError("indices and data should have the same size")
        if (self.indptr[-1] > len(self.indices)):
            raise ValueError("Last value of index pointer should be less than "
                                "the size of index and data arrays")

        self.prune()

        if full_check:
            # check format validity (more expensive)
            if self.nnz > 0:
                if self.indices.max() >= N//C:
                    raise ValueError("column index values must be < %d (now max %d)"
                                     % (N//C, self.indices.max()))
                if self.indices.min() < 0:
                    raise ValueError("column index values must be >= 0")
                if np.diff(self.indptr).min() < 0:
                    raise ValueError("index pointer values must form a "
                                        "non-decreasing sequence")

            idx_dtype = self._get_index_dtype((self.indices, self.indptr))
            self.indptr = np.asarray(self.indptr, dtype=idx_dtype)
            self.indices = np.asarray(self.indices, dtype=idx_dtype)
            self.data = to_native(self.data)
        # if not self.has_sorted_indices():
        #    warn('Indices were not in sorted order. Sorting indices.')
        #    self.sort_indices(check_first=False)

    @property
    def blocksize(self) -> tuple:
        """Block size of the matrix."""
        return self.data.shape[1:]

    def _getnnz(self, axis=None):
        if axis is not None:
            raise NotImplementedError("_getnnz over an axis is not implemented "
                                      "for BSR format")
        R,C = self.blocksize
        return int(self.indptr[-1] * R * C)

    _getnnz.__doc__ = _spbase._getnnz.__doc__

    def __repr__(self):
        _, fmt = _formats[self.format]
        sparse_cls = 'array' if isinstance(self, sparray) else 'matrix'
        shape_str = 'x'.join(str(x) for x in self.shape)
        blksz = 'x'.join(str(x) for x in self.blocksize)
        return (
            f"<{shape_str} sparse {sparse_cls} of type '{self.dtype.type}'\n"
            f"\twith {self.nnz} stored elements (blocksize = {blksz}) in {fmt} format>"
        )

    def diagonal(self, k=0):
        rows, cols = self.shape
        if k <= -rows or k >= cols:
            return np.empty(0, dtype=self.data.dtype)
        R, C = self.blocksize
        y = np.zeros(min(rows + min(k, 0), cols - max(k, 0)),
                     dtype=upcast(self.dtype))
        _sparsetools.bsr_diagonal(k, rows // R, cols // C, R, C,
                                  self.indptr, self.indices,
                                  np.ravel(self.data), y)
        return y

    diagonal.__doc__ = _spbase.diagonal.__doc__

    ##########################
    # NotImplemented methods #
    ##########################

    def __getitem__(self,key):
        raise NotImplementedError

    def __setitem__(self,key,val):
        raise NotImplementedError

    ######################
    # Arithmetic methods #
    ######################

    def _add_dense(self, other):
        return self.tocoo(copy=False)._add_dense(other)

    def _mul_vector(self, other):
        M,N = self.shape
        R,C = self.blocksize

        result = np.zeros(self.shape[0], dtype=upcast(self.dtype, other.dtype))

        bsr_matvec(M//R, N//C, R, C,
            self.indptr, self.indices, self.data.ravel(),
            other, result)

        return result

    def _mul_multivector(self,other):
        R,C = self.blocksize
        M,N = self.shape
        n_vecs = other.shape[1]  # number of column vectors

        result = np.zeros((M,n_vecs), dtype=upcast(self.dtype,other.dtype))

        bsr_matvecs(M//R, N//C, n_vecs, R, C,
                self.indptr, self.indices, self.data.ravel(),
                other.ravel(), result.ravel())

        return result

    def _mul_sparse_matrix(self, other):
        M, K1 = self.shape
        K2, N = other.shape

        R,n = self.blocksize

        # convert to this format
        if other.format == "bsr":
            C = other.blocksize[1]
        else:
            C = 1

        if other.format == "csr" and n == 1:
            other = other.tobsr(blocksize=(n,C), copy=False)  # lightweight conversion
        else:
            other = other.tobsr(blocksize=(n,C))

        idx_dtype = self._get_index_dtype((self.indptr, self.indices,
                                           other.indptr, other.indices))

        bnnz = csr_matmat_maxnnz(M//R, N//C,
                                 self.indptr.astype(idx_dtype),
                                 self.indices.astype(idx_dtype),
                                 other.indptr.astype(idx_dtype),
                                 other.indices.astype(idx_dtype))

        idx_dtype = self._get_index_dtype((self.indptr, self.indices,
                                           other.indptr, other.indices),
                                          maxval=bnnz)
        indptr = np.empty(self.indptr.shape, dtype=idx_dtype)
        indices = np.empty(bnnz, dtype=idx_dtype)
        data = np.empty(R*C*bnnz, dtype=upcast(self.dtype,other.dtype))

        bsr_matmat(bnnz, M//R, N//C, R, C, n,
                   self.indptr.astype(idx_dtype),
                   self.indices.astype(idx_dtype),
                   np.ravel(self.data),
                   other.indptr.astype(idx_dtype),
                   other.indices.astype(idx_dtype),
                   np.ravel(other.data),
                   indptr,
                   indices,
                   data)

        data = data.reshape(-1,R,C)

        # TODO eliminate zeros

        return self._bsr_container(
            (data, indices, indptr), shape=(M, N), blocksize=(R, C)
        )

    ######################
    # Conversion methods #
    ######################

    def tobsr(self, blocksize=None, copy=False):
        """Convert this array/matrix into Block Sparse Row Format.

        With copy=False, the data/indices may be shared between this
        array/matrix and the resultant bsr_array/bsr_matrix.

        If blocksize=(R, C) is provided, it will be used for determining
        block size of the bsr_array/bsr_matrix.
        """
        if blocksize not in [None, self.blocksize]:
            return self.tocsr().tobsr(blocksize=blocksize)
        if copy:
            return self.copy()
        else:
            return self

    def tocsr(self, copy=False):
        M, N = self.shape
        R, C = self.blocksize
        nnz = self.nnz
        idx_dtype = self._get_index_dtype((self.indptr, self.indices),
                                          maxval=max(nnz, N))
        indptr = np.empty(M + 1, dtype=idx_dtype)
        indices = np.empty(nnz, dtype=idx_dtype)
        data = np.empty(nnz, dtype=upcast(self.dtype))

        bsr_tocsr(M // R,  # n_brow
                  N // C,  # n_bcol
                  R, C,
                  self.indptr.astype(idx_dtype, copy=False),
                  self.indices.astype(idx_dtype, copy=False),
                  self.data,
                  indptr,
                  indices,
                  data)
        return self._csr_container((data, indices, indptr), shape=self.shape)

    tocsr.__doc__ = _spbase.tocsr.__doc__

    def tocsc(self, copy=False):
        return self.tocsr(copy=False).tocsc(copy=copy)

    tocsc.__doc__ = _spbase.tocsc.__doc__

    def tocoo(self, copy=True):
        """Convert this array/matrix to COOrdinate format.

        When copy=False the data array will be shared between
        this array/matrix and the resultant coo_array/coo_matrix.
        """

        M,N = self.shape
        R,C = self.blocksize

        indptr_diff = np.diff(self.indptr)
        if indptr_diff.dtype.itemsize > np.dtype(np.intp).itemsize:
            # Check for potential overflow
            indptr_diff_limited = indptr_diff.astype(np.intp)
            if np.any(indptr_diff_limited != indptr_diff):
                raise ValueError("Matrix too big to convert")
            indptr_diff = indptr_diff_limited

        idx_dtype = self._get_index_dtype(maxval=max(M, N))
        row = (R * np.arange(M//R, dtype=idx_dtype)).repeat(indptr_diff)
        row = row.repeat(R*C).reshape(-1,R,C)
        row += np.tile(np.arange(R, dtype=idx_dtype).reshape(-1,1), (1,C))
        row = row.reshape(-1)

        col = ((C * self.indices).astype(idx_dtype, copy=False)
               .repeat(R*C).reshape(-1,R,C))
        col += np.tile(np.arange(C, dtype=idx_dtype), (R,1))
        col = col.reshape(-1)

        data = self.data.reshape(-1)

        if copy:
            data = data.copy()

        return self._coo_container(
            (data, (row, col)), shape=self.shape
        )

    def toarray(self, order=None, out=None):
        return self.tocoo(copy=False).toarray(order=order, out=out)

    toarray.__doc__ = _spbase.toarray.__doc__

    def transpose(self, axes=None, copy=False):
        if axes is not None and axes != (1, 0):
            raise ValueError("Sparse matrices do not support "
                              "an 'axes' parameter because swapping "
                              "dimensions is the only logical permutation.")

        R, C = self.blocksize
        M, N = self.shape
        NBLK = self.nnz//(R*C)

        if self.nnz == 0:
            return self._bsr_container((N, M), blocksize=(C, R),
                                       dtype=self.dtype, copy=copy)

        indptr = np.empty(N//C + 1, dtype=self.indptr.dtype)
        indices = np.empty(NBLK, dtype=self.indices.dtype)
        data = np.empty((NBLK, C, R), dtype=self.data.dtype)

        bsr_transpose(M//R, N//C, R, C,
                      self.indptr, self.indices, self.data.ravel(),
                      indptr, indices, data.ravel())

        return self._bsr_container((data, indices, indptr),
                                   shape=(N, M), copy=copy)

    transpose.__doc__ = _spbase.transpose.__doc__

    ##############################################################
    # methods that examine or modify the internal data structure #
    ##############################################################

    def eliminate_zeros(self):
        """Remove zero elements in-place."""

        if not self.nnz:
            return  # nothing to do

        R,C = self.blocksize
        M,N = self.shape

        mask = (self.data != 0).reshape(-1,R*C).sum(axis=1)  # nonzero blocks

        nonzero_blocks = mask.nonzero()[0]

        self.data[:len(nonzero_blocks)] = self.data[nonzero_blocks]

        # modifies self.indptr and self.indices *in place*
        _sparsetools.csr_eliminate_zeros(M//R, N//C, self.indptr,
                                         self.indices, mask)
        self.prune()

    def sum_duplicates(self):
        """Eliminate duplicate array/matrix entries by adding them together

        The is an *in place* operation
        """
        if self.has_canonical_format:
            return
        self.sort_indices()
        R, C = self.blocksize
        M, N = self.shape

        # port of _sparsetools.csr_sum_duplicates
        n_row = M // R
        nnz = 0
        row_end = 0
        for i in range(n_row):
            jj = row_end
            row_end = self.indptr[i+1]
            while jj < row_end:
                j = self.indices[jj]
                x = self.data[jj]
                jj += 1
                while jj < row_end and self.indices[jj] == j:
                    x += self.data[jj]
                    jj += 1
                self.indices[nnz] = j
                self.data[nnz] = x
                nnz += 1
            self.indptr[i+1] = nnz

        self.prune()  # nnz may have changed
        self.has_canonical_format = True

    def sort_indices(self):
        """Sort the indices of this array/matrix *in place*
        """
        if self.has_sorted_indices:
            return

        R,C = self.blocksize
        M,N = self.shape

        bsr_sort_indices(M//R, N//C, R, C, self.indptr, self.indices, self.data.ravel())

        self.has_sorted_indices = True

    def prune(self):
        """Remove empty space after all non-zero elements.
        """

        R,C = self.blocksize
        M,N = self.shape

        if len(self.indptr) != M//R + 1:
            raise ValueError("index pointer has invalid length")

        bnnz = self.indptr[-1]

        if len(self.indices) < bnnz:
            raise ValueError("indices array has too few elements")
        if len(self.data) < bnnz:
            raise ValueError("data array has too few elements")

        self.data = self.data[:bnnz]
        self.indices = self.indices[:bnnz]

    # utility functions
    def _binopt(self, other, op, in_shape=None, out_shape=None):
        """Apply the binary operation fn to two sparse matrices."""

        # Ideally we'd take the GCDs of the blocksize dimensions
        # and explode self and other to match.
        other = self.__class__(other, blocksize=self.blocksize)

        # e.g. bsr_plus_bsr, etc.
        fn = getattr(_sparsetools, self.format + op + self.format)

        R,C = self.blocksize

        max_bnnz = len(self.data) + len(other.data)
        idx_dtype = self._get_index_dtype((self.indptr, self.indices,
                                           other.indptr, other.indices),
                                          maxval=max_bnnz)
        indptr = np.empty(self.indptr.shape, dtype=idx_dtype)
        indices = np.empty(max_bnnz, dtype=idx_dtype)

        bool_ops = ['_ne_', '_lt_', '_gt_', '_le_', '_ge_']
        if op in bool_ops:
            data = np.empty(R*C*max_bnnz, dtype=np.bool_)
        else:
            data = np.empty(R*C*max_bnnz, dtype=upcast(self.dtype,other.dtype))

        fn(self.shape[0]//R, self.shape[1]//C, R, C,
           self.indptr.astype(idx_dtype),
           self.indices.astype(idx_dtype),
           self.data,
           other.indptr.astype(idx_dtype),
           other.indices.astype(idx_dtype),
           np.ravel(other.data),
           indptr,
           indices,
           data)

        actual_bnnz = indptr[-1]
        indices = indices[:actual_bnnz]
        data = data[:R*C*actual_bnnz]

        if actual_bnnz < max_bnnz/2:
            indices = indices.copy()
            data = data.copy()

        data = data.reshape(-1,R,C)

        return self.__class__((data, indices, indptr), shape=self.shape)

    # needed by _data_matrix
    def _with_data(self,data,copy=True):
        """Returns a matrix with the same sparsity structure as self,
        but with different data.  By default the structure arrays
        (i.e. .indptr and .indices) are copied.
        """
        if copy:
            return self.__class__((data,self.indices.copy(),self.indptr.copy()),
                                   shape=self.shape,dtype=data.dtype)
        else:
            return self.__class__((data,self.indices,self.indptr),
                                   shape=self.shape,dtype=data.dtype)

#    # these functions are used by the parent class
#    # to remove redundancy between bsc_matrix and bsr_matrix
#    def _swap(self,x):
#        """swap the members of x if this is a column-oriented matrix
#        """
#        return (x[0],x[1])


def isspmatrix_bsr(x):
    """Is `x` of a bsr_matrix type?

    Parameters
    ----------
    x
        object to check for being a bsr matrix

    Returns
    -------
    bool
        True if `x` is a bsr matrix, False otherwise

    Examples
    --------
    >>> from scipy.sparse import bsr_array, bsr_matrix, csr_matrix, isspmatrix_bsr
    >>> isspmatrix_bsr(bsr_matrix([[5]]))
    True
    >>> isspmatrix_bsr(bsr_array([[5]]))
    False
    >>> isspmatrix_bsr(csr_matrix([[5]]))
    False
    """
    return isinstance(x, bsr_matrix)


# This namespace class separates array from matrix with isinstance
class bsr_array(_bsr_base, sparray):
    """
    Block Sparse Row format sparse array.

    This can be instantiated in several ways:
        bsr_array(D, [blocksize=(R,C)])
            where D is a 2-D ndarray.

        bsr_array(S, [blocksize=(R,C)])
            with another sparse array or matrix S (equivalent to S.tobsr())

        bsr_array((M, N), [blocksize=(R,C), dtype])
            to construct an empty sparse array with shape (M, N)
            dtype is optional, defaulting to dtype='d'.

        bsr_array((data, ij), [blocksize=(R,C), shape=(M, N)])
            where ``data`` and ``ij`` satisfy ``a[ij[0, k], ij[1, k]] = data[k]``

        bsr_array((data, indices, indptr), [shape=(M, N)])
            is the standard BSR representation where the block column
            indices for row i are stored in ``indices[indptr[i]:indptr[i+1]]``
            and their corresponding block values are stored in
            ``data[ indptr[i]: indptr[i+1] ]``. If the shape parameter is not
            supplied, the array dimensions are inferred from the index arrays.

    Attributes
    ----------
    dtype : dtype
        Data type of the array
    shape : 2-tuple
        Shape of the array
    ndim : int
        Number of dimensions (this is always 2)
    nnz
    size
    data
        BSR format data array of the array
    indices
        BSR format index array of the array
    indptr
        BSR format index pointer array of the array
    blocksize
        Block size
    has_sorted_indices : bool
        Whether indices are sorted
    has_canonical_format : bool
    T

    Notes
    -----
    Sparse arrays can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.

    **Summary of BSR format**

    The Block Sparse Row (BSR) format is very similar to the Compressed
    Sparse Row (CSR) format. BSR is appropriate for sparse matrices with dense
    sub matrices like the last example below. Such sparse block matrices often
    arise in vector-valued finite element discretizations. In such cases, BSR is
    considerably more efficient than CSR and CSC for many sparse arithmetic
    operations.

    **Blocksize**

    The blocksize (R,C) must evenly divide the shape of the sparse array (M,N).
    That is, R and C must satisfy the relationship ``M % R = 0`` and
    ``N % C = 0``.

    If no blocksize is specified, a simple heuristic is applied to determine
    an appropriate blocksize.

    **Canonical Format**

    In canonical format, there are no duplicate blocks and indices are sorted
    per row.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import bsr_array
    >>> bsr_array((3, 4), dtype=np.int8).toarray()
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=int8)

    >>> row = np.array([0, 0, 1, 2, 2, 2])
    >>> col = np.array([0, 2, 2, 0, 1, 2])
    >>> data = np.array([1, 2, 3 ,4, 5, 6])
    >>> bsr_array((data, (row, col)), shape=(3, 3)).toarray()
    array([[1, 0, 2],
           [0, 0, 3],
           [4, 5, 6]])

    >>> indptr = np.array([0, 2, 3, 6])
    >>> indices = np.array([0, 2, 2, 0, 1, 2])
    >>> data = np.array([1, 2, 3, 4, 5, 6]).repeat(4).reshape(6, 2, 2)
    >>> bsr_array((data,indices,indptr), shape=(6, 6)).toarray()
    array([[1, 1, 0, 0, 2, 2],
           [1, 1, 0, 0, 2, 2],
           [0, 0, 0, 0, 3, 3],
           [0, 0, 0, 0, 3, 3],
           [4, 4, 5, 5, 6, 6],
           [4, 4, 5, 5, 6, 6]])

    """


class bsr_matrix(spmatrix, _bsr_base):
    """
    Block Sparse Row format sparse matrix.

    This can be instantiated in several ways:
        bsr_matrix(D, [blocksize=(R,C)])
            where D is a 2-D ndarray.

        bsr_matrix(S, [blocksize=(R,C)])
            with another sparse array or matrix S (equivalent to S.tobsr())

        bsr_matrix((M, N), [blocksize=(R,C), dtype])
            to construct an empty sparse matrix with shape (M, N)
            dtype is optional, defaulting to dtype='d'.

        bsr_matrix((data, ij), [blocksize=(R,C), shape=(M, N)])
            where ``data`` and ``ij`` satisfy ``a[ij[0, k], ij[1, k]] = data[k]``

        bsr_matrix((data, indices, indptr), [shape=(M, N)])
            is the standard BSR representation where the block column
            indices for row i are stored in ``indices[indptr[i]:indptr[i+1]]``
            and their corresponding block values are stored in
            ``data[ indptr[i]: indptr[i+1] ]``. If the shape parameter is not
            supplied, the matrix dimensions are inferred from the index arrays.

    Attributes
    ----------
    dtype : dtype
        Data type of the matrix
    shape : 2-tuple
        Shape of the matrix
    ndim : int
        Number of dimensions (this is always 2)
    nnz
    size
    data
        BSR format data array of the matrix
    indices
        BSR format index array of the matrix
    indptr
        BSR format index pointer array of the matrix
    blocksize
        Block size
    has_sorted_indices : bool
        Whether indices are sorted
    has_canonical_format : bool
    T

    Notes
    -----
    Sparse matrices can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.

    **Summary of BSR format**

    The Block Sparse Row (BSR) format is very similar to the Compressed
    Sparse Row (CSR) format. BSR is appropriate for sparse matrices with dense
    sub matrices like the last example below. Such sparse block matrices often
    arise in vector-valued finite element discretizations. In such cases, BSR is
    considerably more efficient than CSR and CSC for many sparse arithmetic
    operations.

    **Blocksize**

    The blocksize (R,C) must evenly divide the shape of the sparse matrix (M,N).
    That is, R and C must satisfy the relationship ``M % R = 0`` and
    ``N % C = 0``.

    If no blocksize is specified, a simple heuristic is applied to determine
    an appropriate blocksize.

    **Canonical Format**

    In canonical format, there are no duplicate blocks and indices are sorted
    per row.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import bsr_matrix
    >>> bsr_matrix((3, 4), dtype=np.int8).toarray()
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=int8)

    >>> row = np.array([0, 0, 1, 2, 2, 2])
    >>> col = np.array([0, 2, 2, 0, 1, 2])
    >>> data = np.array([1, 2, 3 ,4, 5, 6])
    >>> bsr_matrix((data, (row, col)), shape=(3, 3)).toarray()
    array([[1, 0, 2],
           [0, 0, 3],
           [4, 5, 6]])

    >>> indptr = np.array([0, 2, 3, 6])
    >>> indices = np.array([0, 2, 2, 0, 1, 2])
    >>> data = np.array([1, 2, 3, 4, 5, 6]).repeat(4).reshape(6, 2, 2)
    >>> bsr_matrix((data,indices,indptr), shape=(6, 6)).toarray()
    array([[1, 1, 0, 0, 2, 2],
           [1, 1, 0, 0, 2, 2],
           [0, 0, 0, 0, 3, 3],
           [0, 0, 0, 0, 3, 3],
           [4, 4, 5, 5, 6, 6],
           [4, 4, 5, 5, 6, 6]])

    """

