from ._sputils import isintlike, isscalarlike


class spmatrix:
    """This class provides a base class for all sparse matrix classes.

    It cannot be instantiated.  Most of the work is provided by subclasses.
    """
    _is_array = False

    @property
    def _bsr_container(self):
        from ._bsr import bsr_matrix
        return bsr_matrix

    @property
    def _coo_container(self):
        from ._coo import coo_matrix
        return coo_matrix

    @property
    def _csc_container(self):
        from ._csc import csc_matrix
        return csc_matrix

    @property
    def _csr_container(self):
        from ._csr import csr_matrix
        return csr_matrix

    @property
    def _dia_container(self):
        from ._dia import dia_matrix
        return dia_matrix

    @property
    def _dok_container(self):
        from ._dok import dok_matrix
        return dok_matrix

    @property
    def _lil_container(self):
        from ._lil import lil_matrix
        return lil_matrix

    # Restore matrix multiplication
    def __mul__(self, other):
        return self._mul_dispatch(other)

    def __rmul__(self, other):
        return self._rmul_dispatch(other)

    # Restore matrix power
    def __pow__(self, other):
        M, N = self.shape
        if M != N:
            raise TypeError('sparse matrix is not square')

        if isintlike(other):
            other = int(other)
            if other < 0:
                raise ValueError('exponent must be >= 0')

            if other == 0:
                from ._construct import eye
                return eye(M, dtype=self.dtype)

            if other == 1:
                return self.copy()

            tmp = self.__pow__(other // 2)
            if other % 2:
                return self @ tmp @ tmp
            else:
                return tmp @ tmp

        if isscalarlike(other):
            raise ValueError('exponent must be an integer')
        return NotImplemented

    ## Backward compatibility

    def set_shape(self, shape):
        """Set the shape of the matrix in-place"""
        # Make sure copy is False since this is in place
        # Make sure format is unchanged because we are doing a __dict__ swap
        new_self = self.reshape(shape, copy=False).asformat(self.format)
        self.__dict__ = new_self.__dict__

    def get_shape(self):
        """Get the shape of the matrix"""
        return self._shape

    shape = property(fget=get_shape, fset=set_shape,
                     doc="Shape of the matrix")

    def asfptype(self):
        """Upcast array to a floating point format (if necessary)"""
        return self._asfptype()

    def getmaxprint(self):
        """Maximum number of elements to display when printed."""
        return self._getmaxprint()

    def getformat(self):
        """Matrix storage format"""
        return self.format

    def getnnz(self, axis=None):
        """Number of stored values, including explicit zeros.

        Parameters
        ----------
        axis : None, 0, or 1
            Select between the number of values across the whole array, in
            each column, or in each row.
        """
        return self._getnnz(axis=axis)

    def getH(self):
        """Return the Hermitian transpose of this array.

        See Also
        --------
        numpy.matrix.getH : NumPy's implementation of `getH` for matrices
        """
        return self.conjugate().transpose()

    def getcol(self, j):
        """Returns a copy of column j of the array, as an (m x 1) sparse
        array (column vector).
        """
        return self._getcol(j)

    def getrow(self, i):
        """Returns a copy of row i of the array, as a (1 x n) sparse
        array (row vector).
        """
        return self._getrow(i)


def _array_doc_to_matrix(docstr):
    # For opimized builds with stripped docstrings
    if docstr is None:
        return None
    return (
        docstr.replace('sparse arrays', 'sparse matrices')
              .replace('sparse array', 'sparse matrix')
    )
