"""Compressed Sparse Column matrix format"""
__docformat__ = "restructuredtext en"

__all__ = ['csc_array', 'csc_matrix', 'isspmatrix_csc']


import numpy as np

from ._matrix import spmatrix, _array_doc_to_matrix
from ._base import _spbase, sparray
from ._sparsetools import csc_tocsr, expandptr
from ._sputils import upcast

from ._compressed import _cs_matrix


class _csc_base(_cs_matrix):
    """
    Compressed Sparse Column matrix

    This can be instantiated in several ways:

        csc_array(D)
            with a dense matrix or rank-2 ndarray D

        csc_array(S)
            with another sparse matrix S (equivalent to S.tocsc())

        csc_array((M, N), [dtype])
            to construct an empty matrix with shape (M, N)
            dtype is optional, defaulting to dtype='d'.

        csc_array((data, (row_ind, col_ind)), [shape=(M, N)])
            where ``data``, ``row_ind`` and ``col_ind`` satisfy the
            relationship ``a[row_ind[k], col_ind[k]] = data[k]``.

        csc_array((data, indices, indptr), [shape=(M, N)])
            is the standard CSC representation where the row indices for
            column i are stored in ``indices[indptr[i]:indptr[i+1]]``
            and their corresponding values are stored in
            ``data[indptr[i]:indptr[i+1]]``.  If the shape parameter is
            not supplied, the matrix dimensions are inferred from
            the index arrays.

    Attributes
    ----------
    dtype : dtype
        Data type of the matrix
    shape : 2-tuple
        Shape of the matrix
    ndim : int
        Number of dimensions (this is always 2)
    nnz
        Number of stored values, including explicit zeros
    data
        Data array of the matrix
    indices
        CSC format index array
    indptr
        CSC format index pointer array
    has_sorted_indices
        Whether indices are sorted

    Notes
    -----

    Sparse matrices can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.

    Advantages of the CSC format
        - efficient arithmetic operations CSC + CSC, CSC * CSC, etc.
        - efficient column slicing
        - fast matrix vector products (CSR, BSR may be faster)

    Disadvantages of the CSC format
      - slow row slicing operations (consider CSR)
      - changes to the sparsity structure are expensive (consider LIL or DOK)

    Canonical format
      - Within each column, indices are sorted by row.
      - There are no duplicate entries.

    Examples
    --------

    >>> import numpy as np
    >>> from scipy.sparse import csc_array
    >>> csc_array((3, 4), dtype=np.int8).toarray()
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=int8)

    >>> row = np.array([0, 2, 2, 0, 1, 2])
    >>> col = np.array([0, 0, 1, 2, 2, 2])
    >>> data = np.array([1, 2, 3, 4, 5, 6])
    >>> csc_array((data, (row, col)), shape=(3, 3)).toarray()
    array([[1, 0, 4],
           [0, 0, 5],
           [2, 3, 6]])

    >>> indptr = np.array([0, 2, 3, 6])
    >>> indices = np.array([0, 2, 2, 0, 1, 2])
    >>> data = np.array([1, 2, 3, 4, 5, 6])
    >>> csc_array((data, indices, indptr), shape=(3, 3)).toarray()
    array([[1, 0, 4],
           [0, 0, 5],
           [2, 3, 6]])

    """
    _format = 'csc'

    def transpose(self, axes=None, copy=False):
        if axes is not None:
            raise ValueError("Sparse matrices do not support "
                              "an 'axes' parameter because swapping "
                              "dimensions is the only logical permutation.")

        M, N = self.shape

        return self._csr_container((self.data, self.indices,
                                    self.indptr), (N, M), copy=copy)

    transpose.__doc__ = _spbase.transpose.__doc__

    def __iter__(self):
        yield from self.tocsr()

    def tocsc(self, copy=False):
        if copy:
            return self.copy()
        else:
            return self

    tocsc.__doc__ = _spbase.tocsc.__doc__

    def tocsr(self, copy=False):
        M,N = self.shape
        idx_dtype = self._get_index_dtype((self.indptr, self.indices),
                                    maxval=max(self.nnz, N))
        indptr = np.empty(M + 1, dtype=idx_dtype)
        indices = np.empty(self.nnz, dtype=idx_dtype)
        data = np.empty(self.nnz, dtype=upcast(self.dtype))

        csc_tocsr(M, N,
                  self.indptr.astype(idx_dtype),
                  self.indices.astype(idx_dtype),
                  self.data,
                  indptr,
                  indices,
                  data)

        A = self._csr_container(
            (data, indices, indptr),
            shape=self.shape, copy=False
        )
        A.has_sorted_indices = True
        return A

    tocsr.__doc__ = _spbase.tocsr.__doc__

    def nonzero(self):
        # CSC can't use _cs_matrix's .nonzero method because it
        # returns the indices sorted for self transposed.

        # Get row and col indices, from _cs_matrix.tocoo
        major_dim, minor_dim = self._swap(self.shape)
        minor_indices = self.indices
        major_indices = np.empty(len(minor_indices), dtype=self.indices.dtype)
        expandptr(major_dim, self.indptr, major_indices)
        row, col = self._swap((major_indices, minor_indices))

        # Remove explicit zeros
        nz_mask = self.data != 0
        row = row[nz_mask]
        col = col[nz_mask]

        # Sort them to be in C-style order
        ind = np.argsort(row, kind='mergesort')
        row = row[ind]
        col = col[ind]

        return row, col

    nonzero.__doc__ = _cs_matrix.nonzero.__doc__

    def _getrow(self, i):
        """Returns a copy of row i of the matrix, as a (1 x n)
        CSR matrix (row vector).
        """
        M, N = self.shape
        i = int(i)
        if i < 0:
            i += M
        if i < 0 or i >= M:
            raise IndexError('index (%d) out of range' % i)
        return self._get_submatrix(minor=i).tocsr()

    def _getcol(self, i):
        """Returns a copy of column i of the matrix, as a (m x 1)
        CSC matrix (column vector).
        """
        M, N = self.shape
        i = int(i)
        if i < 0:
            i += N
        if i < 0 or i >= N:
            raise IndexError('index (%d) out of range' % i)
        return self._get_submatrix(major=i, copy=True)

    def _get_intXarray(self, row, col):
        return self._major_index_fancy(col)._get_submatrix(minor=row)

    def _get_intXslice(self, row, col):
        if col.step in (1, None):
            return self._get_submatrix(major=col, minor=row, copy=True)
        return self._major_slice(col)._get_submatrix(minor=row)

    def _get_sliceXint(self, row, col):
        if row.step in (1, None):
            return self._get_submatrix(major=col, minor=row, copy=True)
        return self._get_submatrix(major=col)._minor_slice(row)

    def _get_sliceXarray(self, row, col):
        return self._major_index_fancy(col)._minor_slice(row)

    def _get_arrayXint(self, row, col):
        return self._get_submatrix(major=col)._minor_index_fancy(row)

    def _get_arrayXslice(self, row, col):
        return self._major_slice(col)._minor_index_fancy(row)

    # these functions are used by the parent class (_cs_matrix)
    # to remove redudancy between csc_array and csr_matrix
    def _swap(self, x):
        """swap the members of x if this is a column-oriented matrix
        """
        return x[1], x[0]


def isspmatrix_csc(x):
    """Is `x` of csc_matrix type?

    Parameters
    ----------
    x
        object to check for being a csc matrix

    Returns
    -------
    bool
        True if `x` is a csc matrix, False otherwise

    Examples
    --------
    >>> from scipy.sparse import csc_array, csc_matrix, coo_matrix, isspmatrix_csc
    >>> isspmatrix_csc(csc_matrix([[5]]))
    True
    >>> isspmatrix_csc(csc_array([[5]]))
    False
    >>> isspmatrix_csc(coo_matrix([[5]]))
    False
    """
    return isinstance(x, csc_matrix)


# This namespace class separates array from matrix with isinstance
class csc_array(_csc_base, sparray):
    pass

csc_array.__doc__ = _csc_base.__doc__

class csc_matrix(spmatrix, _csc_base):
    pass

csc_matrix.__doc__ = _array_doc_to_matrix(_csc_base.__doc__)
