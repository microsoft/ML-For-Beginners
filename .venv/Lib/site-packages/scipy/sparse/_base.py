"""Base class for sparse matrices"""
from warnings import warn

import numpy as np

from ._sputils import (asmatrix, check_reshape_kwargs, check_shape,
                       get_sum_dtype, isdense, isscalarlike,
                       matrix, validateaxis,)

from ._matrix import spmatrix

__all__ = ['isspmatrix', 'issparse', 'sparray',
           'SparseWarning', 'SparseEfficiencyWarning']


class SparseWarning(Warning):
    pass


class SparseFormatWarning(SparseWarning):
    pass


class SparseEfficiencyWarning(SparseWarning):
    pass


# The formats that we might potentially understand.
_formats = {'csc': [0, "Compressed Sparse Column"],
            'csr': [1, "Compressed Sparse Row"],
            'dok': [2, "Dictionary Of Keys"],
            'lil': [3, "List of Lists"],
            'dod': [4, "Dictionary of Dictionaries"],
            'sss': [5, "Symmetric Sparse Skyline"],
            'coo': [6, "COOrdinate"],
            'lba': [7, "Linpack BAnded"],
            'egd': [8, "Ellpack-itpack Generalized Diagonal"],
            'dia': [9, "DIAgonal"],
            'bsr': [10, "Block Sparse Row"],
            'msr': [11, "Modified compressed Sparse Row"],
            'bsc': [12, "Block Sparse Column"],
            'msc': [13, "Modified compressed Sparse Column"],
            'ssk': [14, "Symmetric SKyline"],
            'nsk': [15, "Nonsymmetric SKyline"],
            'jad': [16, "JAgged Diagonal"],
            'uss': [17, "Unsymmetric Sparse Skyline"],
            'vbr': [18, "Variable Block Row"],
            'und': [19, "Undefined"]
            }


# These univariate ufuncs preserve zeros.
_ufuncs_with_fixed_point_at_zero = frozenset([
        np.sin, np.tan, np.arcsin, np.arctan, np.sinh, np.tanh, np.arcsinh,
        np.arctanh, np.rint, np.sign, np.expm1, np.log1p, np.deg2rad,
        np.rad2deg, np.floor, np.ceil, np.trunc, np.sqrt])


MAXPRINT = 50


class _spbase:
    """ This class provides a base class for all sparse arrays.  It
    cannot be instantiated.  Most of the work is provided by subclasses.
    """

    __array_priority__ = 10.1
    _format = 'und'  # undefined
    ndim = 2

    @property
    def _bsr_container(self):
        from ._bsr import bsr_array
        return bsr_array

    @property
    def _coo_container(self):
        from ._coo import coo_array
        return coo_array

    @property
    def _csc_container(self):
        from ._csc import csc_array
        return csc_array

    @property
    def _csr_container(self):
        from ._csr import csr_array
        return csr_array

    @property
    def _dia_container(self):
        from ._dia import dia_array
        return dia_array

    @property
    def _dok_container(self):
        from ._dok import dok_array
        return dok_array

    @property
    def _lil_container(self):
        from ._lil import lil_array
        return lil_array

    _is_array = True

    def __init__(self, maxprint=MAXPRINT):
        self._shape = None
        if self.__class__.__name__ == '_spbase':
            raise ValueError("This class is not intended"
                             " to be instantiated directly.")
        self.maxprint = maxprint

    # Use this in 1.13.0 and later:
    #
    # @property
    # def shape(self):
    #   return self._shape

    def reshape(self, *args, **kwargs):
        """reshape(self, shape, order='C', copy=False)

        Gives a new shape to a sparse array without changing its data.

        Parameters
        ----------
        shape : length-2 tuple of ints
            The new shape should be compatible with the original shape.
        order : {'C', 'F'}, optional
            Read the elements using this index order. 'C' means to read and
            write the elements using C-like index order; e.g., read entire first
            row, then second row, etc. 'F' means to read and write the elements
            using Fortran-like index order; e.g., read entire first column, then
            second column, etc.
        copy : bool, optional
            Indicates whether or not attributes of self should be copied
            whenever possible. The degree to which attributes are copied varies
            depending on the type of sparse array being used.

        Returns
        -------
        reshaped : sparse array
            A sparse array with the given `shape`, not necessarily of the same
            format as the current object.

        See Also
        --------
        numpy.reshape : NumPy's implementation of 'reshape' for ndarrays
        """
        # If the shape already matches, don't bother doing an actual reshape
        # Otherwise, the default is to convert to COO and use its reshape
        shape = check_shape(args, self.shape)
        order, copy = check_reshape_kwargs(kwargs)
        if shape == self.shape:
            if copy:
                return self.copy()
            else:
                return self

        return self.tocoo(copy=copy).reshape(shape, order=order, copy=False)

    def resize(self, shape):
        """Resize the array in-place to dimensions given by ``shape``

        Any elements that lie within the new shape will remain at the same
        indices, while non-zero elements lying outside the new shape are
        removed.

        Parameters
        ----------
        shape : (int, int)
            number of rows and columns in the new array

        Notes
        -----
        The semantics are not identical to `numpy.ndarray.resize` or
        `numpy.resize`. Here, the same data will be maintained at each index
        before and after reshape, if that index is within the new bounds. In
        numpy, resizing maintains contiguity of the array, moving elements
        around in the logical array but not within a flattened representation.

        We give no guarantees about whether the underlying data attributes
        (arrays, etc.) will be modified in place or replaced with new objects.
        """
        # As an inplace operation, this requires implementation in each format.
        raise NotImplementedError(
            f'{type(self).__name__}.resize is not implemented')

    def astype(self, dtype, casting='unsafe', copy=True):
        """Cast the array elements to a specified type.

        Parameters
        ----------
        dtype : string or numpy dtype
            Typecode or data-type to which to cast the data.
        casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
            Controls what kind of data casting may occur.
            Defaults to 'unsafe' for backwards compatibility.
            'no' means the data types should not be cast at all.
            'equiv' means only byte-order changes are allowed.
            'safe' means only casts which can preserve values are allowed.
            'same_kind' means only safe casts or casts within a kind,
            like float64 to float32, are allowed.
            'unsafe' means any data conversions may be done.
        copy : bool, optional
            If `copy` is `False`, the result might share some memory with this
            array. If `copy` is `True`, it is guaranteed that the result and
            this array do not share any memory.
        """

        dtype = np.dtype(dtype)
        if self.dtype != dtype:
            return self.tocsr().astype(
                dtype, casting=casting, copy=copy).asformat(self.format)
        elif copy:
            return self.copy()
        else:
            return self

    @classmethod
    def _ascontainer(cls, X, **kwargs):
        if cls._is_array:
            return np.asarray(X, **kwargs)
        else:
            return asmatrix(X, **kwargs)

    @classmethod
    def _container(cls, X, **kwargs):
        if cls._is_array:
            return np.array(X, **kwargs)
        else:
            return matrix(X, **kwargs)

    def _asfptype(self):
        """Upcast array to a floating point format (if necessary)"""

        fp_types = ['f', 'd', 'F', 'D']

        if self.dtype.char in fp_types:
            return self
        else:
            for fp_type in fp_types:
                if self.dtype <= np.dtype(fp_type):
                    return self.astype(fp_type)

            raise TypeError('cannot upcast [%s] to a floating '
                            'point format' % self.dtype.name)

    def __iter__(self):
        for r in range(self.shape[0]):
            yield self[r, :]

    def _getmaxprint(self):
        """Maximum number of elements to display when printed."""
        return self.maxprint

    def count_nonzero(self):
        """Number of non-zero entries, equivalent to

        np.count_nonzero(a.toarray())

        Unlike the nnz property, which return the number of stored
        entries (the length of the data attribute), this method counts the
        actual number of non-zero entries in data.
        """
        raise NotImplementedError("count_nonzero not implemented for %s." %
                                  self.__class__.__name__)

    def _getnnz(self, axis=None):
        """Number of stored values, including explicit zeros.

        Parameters
        ----------
        axis : None, 0, or 1
            Select between the number of values across the whole array, in
            each column, or in each row.

        See also
        --------
        count_nonzero : Number of non-zero entries
        """
        raise NotImplementedError("getnnz not implemented for %s." %
                                  self.__class__.__name__)

    @property
    def nnz(self):
        """Number of stored values, including explicit zeros.

        See also
        --------
        count_nonzero : Number of non-zero entries
        """
        return self._getnnz()

    @property
    def format(self):
        return self._format

    def __repr__(self):
        _, format_name = _formats[self.format]
        sparse_cls = 'array' if self._is_array else 'matrix'
        return f"<%dx%d sparse {sparse_cls} of type '%s'\n" \
               "\twith %d stored elements in %s format>" % \
               (self.shape + (self.dtype.type, self.nnz, format_name))

    def __str__(self):
        maxprint = self._getmaxprint()

        A = self.tocoo()

        # helper function, outputs "(i,j)  v"
        def tostr(row, col, data):
            triples = zip(list(zip(row, col)), data)
            return '\n'.join([('  %s\t%s' % t) for t in triples])

        if self.nnz > maxprint:
            half = maxprint // 2
            out = tostr(A.row[:half], A.col[:half], A.data[:half])
            out += "\n  :\t:\n"
            half = maxprint - maxprint//2
            out += tostr(A.row[-half:], A.col[-half:], A.data[-half:])
        else:
            out = tostr(A.row, A.col, A.data)

        return out

    def __bool__(self):  # Simple -- other ideas?
        if self.shape == (1, 1):
            return self.nnz != 0
        else:
            raise ValueError("The truth value of an array with more than one "
                             "element is ambiguous. Use a.any() or a.all().")
    __nonzero__ = __bool__

    # What should len(sparse) return? For consistency with dense matrices,
    # perhaps it should be the number of rows?  But for some uses the number of
    # non-zeros is more important.  For now, raise an exception!
    def __len__(self):
        raise TypeError("sparse array length is ambiguous; use getnnz()"
                        " or shape[0]")

    def asformat(self, format, copy=False):
        """Return this array in the passed format.

        Parameters
        ----------
        format : {str, None}
            The desired sparse format ("csr", "csc", "lil", "dok", "array", ...)
            or None for no conversion.
        copy : bool, optional
            If True, the result is guaranteed to not share data with self.

        Returns
        -------
        A : This array in the passed format.
        """
        if format is None or format == self.format:
            if copy:
                return self.copy()
            else:
                return self
        else:
            try:
                convert_method = getattr(self, 'to' + format)
            except AttributeError as e:
                raise ValueError(f'Format {format} is unknown.') from e

            # Forward the copy kwarg, if it's accepted.
            try:
                return convert_method(copy=copy)
            except TypeError:
                return convert_method()

    ###################################################################
    #  NOTE: All arithmetic operations use csr_matrix by default.
    # Therefore a new sparse array format just needs to define a
    # .tocsr() method to provide arithmetic support. Any of these
    # methods can be overridden for efficiency.
    ####################################################################

    def multiply(self, other):
        """Point-wise multiplication by another array
        """
        return self.tocsr().multiply(other)

    def maximum(self, other):
        """Element-wise maximum between this and another array."""
        return self.tocsr().maximum(other)

    def minimum(self, other):
        """Element-wise minimum between this and another array."""
        return self.tocsr().minimum(other)

    def dot(self, other):
        """Ordinary dot product

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.sparse import csr_array
        >>> A = csr_array([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
        >>> v = np.array([1, 0, -1])
        >>> A.dot(v)
        array([ 1, -3, -1], dtype=int64)

        """
        if np.isscalar(other):
            return self * other
        else:
            return self @ other

    def power(self, n, dtype=None):
        """Element-wise power."""
        return self.tocsr().power(n, dtype=dtype)

    def __eq__(self, other):
        return self.tocsr().__eq__(other)

    def __ne__(self, other):
        return self.tocsr().__ne__(other)

    def __lt__(self, other):
        return self.tocsr().__lt__(other)

    def __gt__(self, other):
        return self.tocsr().__gt__(other)

    def __le__(self, other):
        return self.tocsr().__le__(other)

    def __ge__(self, other):
        return self.tocsr().__ge__(other)

    def __abs__(self):
        return abs(self.tocsr())

    def __round__(self, ndigits=0):
        return round(self.tocsr(), ndigits=ndigits)

    def _add_sparse(self, other):
        return self.tocsr()._add_sparse(other)

    def _add_dense(self, other):
        return self.tocoo()._add_dense(other)

    def _sub_sparse(self, other):
        return self.tocsr()._sub_sparse(other)

    def _sub_dense(self, other):
        return self.todense() - other

    def _rsub_dense(self, other):
        # note: this can't be replaced by other + (-self) for unsigned types
        return other - self.todense()

    def __add__(self, other):  # self + other
        if isscalarlike(other):
            if other == 0:
                return self.copy()
            # Now we would add this scalar to every element.
            raise NotImplementedError('adding a nonzero scalar to a '
                                      'sparse array is not supported')
        elif issparse(other):
            if other.shape != self.shape:
                raise ValueError("inconsistent shapes")
            return self._add_sparse(other)
        elif isdense(other):
            other = np.broadcast_to(other, self.shape)
            return self._add_dense(other)
        else:
            return NotImplemented

    def __radd__(self,other):  # other + self
        return self.__add__(other)

    def __sub__(self, other):  # self - other
        if isscalarlike(other):
            if other == 0:
                return self.copy()
            raise NotImplementedError('subtracting a nonzero scalar from a '
                                      'sparse array is not supported')
        elif issparse(other):
            if other.shape != self.shape:
                raise ValueError("inconsistent shapes")
            return self._sub_sparse(other)
        elif isdense(other):
            other = np.broadcast_to(other, self.shape)
            return self._sub_dense(other)
        else:
            return NotImplemented

    def __rsub__(self,other):  # other - self
        if isscalarlike(other):
            if other == 0:
                return -self.copy()
            raise NotImplementedError('subtracting a sparse array from a '
                                      'nonzero scalar is not supported')
        elif isdense(other):
            other = np.broadcast_to(other, self.shape)
            return self._rsub_dense(other)
        else:
            return NotImplemented

    def _mul_dispatch(self, other):
        """`np.matrix`-compatible mul, i.e. `dot` or `NotImplemented`

        interpret other and call one of the following
        self._mul_scalar()
        self._mul_vector()
        self._mul_multivector()
        self._mul_sparse_matrix()
        """
        # This method has to be different from `__matmul__` because it is also
        # called by sparse matrix classes.

        M, N = self.shape

        if other.__class__ is np.ndarray:
            # Fast path for the most common case
            if other.shape == (N,):
                return self._mul_vector(other)
            elif other.shape == (N, 1):
                return self._mul_vector(other.ravel()).reshape(M, 1)
            elif other.ndim == 2 and other.shape[0] == N:
                return self._mul_multivector(other)

        if isscalarlike(other):
            # scalar value
            return self._mul_scalar(other)

        if issparse(other):
            if self.shape[1] != other.shape[0]:
                raise ValueError('dimension mismatch')
            return self._mul_sparse_matrix(other)

        # If it's a list or whatever, treat it like an array
        other_a = np.asanyarray(other)

        if other_a.ndim == 0 and other_a.dtype == np.object_:
            # Not interpretable as an array; return NotImplemented so that
            # other's __rmul__ can kick in if that's implemented.
            return NotImplemented

        try:
            other.shape
        except AttributeError:
            other = other_a

        if other.ndim == 1 or other.ndim == 2 and other.shape[1] == 1:
            # dense row or column vector
            if other.shape != (N,) and other.shape != (N, 1):
                raise ValueError('dimension mismatch')

            result = self._mul_vector(np.ravel(other))

            if isinstance(other, np.matrix):
                result = self._ascontainer(result)

            if other.ndim == 2 and other.shape[1] == 1:
                # If 'other' was an (nx1) column vector, reshape the result
                result = result.reshape(-1, 1)

            return result

        elif other.ndim == 2:
            ##
            # dense 2D array or matrix ("multivector")

            if other.shape[0] != self.shape[1]:
                raise ValueError('dimension mismatch')

            result = self._mul_multivector(np.asarray(other))

            if isinstance(other, np.matrix):
                result = self._ascontainer(result)

            return result

        else:
            raise ValueError('could not interpret dimensions')

    def __mul__(self, *args, **kwargs):
        return self.multiply(*args, **kwargs)

    # by default, use CSR for __mul__ handlers
    def _mul_scalar(self, other):
        return self.tocsr()._mul_scalar(other)

    def _mul_vector(self, other):
        return self.tocsr()._mul_vector(other)

    def _mul_multivector(self, other):
        return self.tocsr()._mul_multivector(other)

    def _mul_sparse_matrix(self, other):
        return self.tocsr()._mul_sparse_matrix(other)

    def _rmul_dispatch(self, other):
        if isscalarlike(other):
            return self._mul_scalar(other)
        else:
            # Don't use asarray unless we have to
            try:
                tr = other.transpose()
            except AttributeError:
                tr = np.asarray(other).transpose()
            ret = self.transpose()._mul_dispatch(tr)
            if ret is NotImplemented:
                return NotImplemented
            return ret.transpose()

    def __rmul__(self, *args, **kwargs):  # other * self
        return self.multiply(*args, **kwargs)

    #######################
    # matmul (@) operator #
    #######################

    def __matmul__(self, other):
        if isscalarlike(other):
            raise ValueError("Scalar operands are not allowed, "
                             "use '*' instead")
        return self._mul_dispatch(other)

    def __rmatmul__(self, other):
        if isscalarlike(other):
            raise ValueError("Scalar operands are not allowed, "
                             "use '*' instead")
        return self._rmul_dispatch(other)

    ####################
    # Other Arithmetic #
    ####################

    def _divide(self, other, true_divide=False, rdivide=False):
        if isscalarlike(other):
            if rdivide:
                if true_divide:
                    return np.true_divide(other, self.todense())
                else:
                    return np.divide(other, self.todense())

            if true_divide and np.can_cast(self.dtype, np.float_):
                return self.astype(np.float_)._mul_scalar(1./other)
            else:
                r = self._mul_scalar(1./other)

                scalar_dtype = np.asarray(other).dtype
                if (np.issubdtype(self.dtype, np.integer) and
                        np.issubdtype(scalar_dtype, np.integer)):
                    return r.astype(self.dtype)
                else:
                    return r

        elif isdense(other):
            if not rdivide:
                if true_divide:
                    recip = np.true_divide(1., other)
                else:
                    recip = np.divide(1., other)
                return self.multiply(recip)
            else:
                if true_divide:
                    return np.true_divide(other, self.todense())
                else:
                    return np.divide(other, self.todense())
        elif issparse(other):
            if rdivide:
                return other._divide(self, true_divide, rdivide=False)

            self_csr = self.tocsr()
            if true_divide and np.can_cast(self.dtype, np.float_):
                return self_csr.astype(np.float_)._divide_sparse(other)
            else:
                return self_csr._divide_sparse(other)
        else:
            return NotImplemented

    def __truediv__(self, other):
        return self._divide(other, true_divide=True)

    def __div__(self, other):
        # Always do true division
        return self._divide(other, true_divide=True)

    def __rtruediv__(self, other):
        # Implementing this as the inverse would be too magical -- bail out
        return NotImplemented

    def __rdiv__(self, other):
        # Implementing this as the inverse would be too magical -- bail out
        return NotImplemented

    def __neg__(self):
        return -self.tocsr()

    def __iadd__(self, other):
        return NotImplemented

    def __isub__(self, other):
        return NotImplemented

    def __imul__(self, other):
        return NotImplemented

    def __idiv__(self, other):
        return self.__itruediv__(other)

    def __itruediv__(self, other):
        return NotImplemented

    def __pow__(self, *args, **kwargs):
        return self.power(*args, **kwargs)

    @property
    def A(self) -> np.ndarray:
        if self._is_array:
            warn(np.VisibleDeprecationWarning(
                "`.A` is deprecated and will be removed in v1.13.0. "
                "Use `.toarray()` instead."
            ))
        return self.toarray()

    @property
    def T(self):
        return self.transpose()

    @property
    def H(self):
        if self._is_array:
            warn(np.VisibleDeprecationWarning(
                "`.H` is deprecated and will be removed in v1.13.0. "
                "Please use `.T.conjugate()` instead."
            ))
        return self.T.conjugate()

    @property
    def real(self):
        return self._real()

    @property
    def imag(self):
        return self._imag()

    @property
    def size(self):
        return self._getnnz()

    def transpose(self, axes=None, copy=False):
        """
        Reverses the dimensions of the sparse array.

        Parameters
        ----------
        axes : None, optional
            This argument is in the signature *solely* for NumPy
            compatibility reasons. Do not pass in anything except
            for the default value.
        copy : bool, optional
            Indicates whether or not attributes of `self` should be
            copied whenever possible. The degree to which attributes
            are copied varies depending on the type of sparse array
            being used.

        Returns
        -------
        p : `self` with the dimensions reversed.

        See Also
        --------
        numpy.transpose : NumPy's implementation of 'transpose' for ndarrays
        """
        return self.tocsr(copy=copy).transpose(axes=axes, copy=False)

    def conjugate(self, copy=True):
        """Element-wise complex conjugation.

        If the array is of non-complex data type and `copy` is False,
        this method does nothing and the data is not copied.

        Parameters
        ----------
        copy : bool, optional
            If True, the result is guaranteed to not share data with self.

        Returns
        -------
        A : The element-wise complex conjugate.

        """
        if np.issubdtype(self.dtype, np.complexfloating):
            return self.tocsr(copy=copy).conjugate(copy=False)
        elif copy:
            return self.copy()
        else:
            return self

    def conj(self, copy=True):
        return self.conjugate(copy=copy)

    conj.__doc__ = conjugate.__doc__

    def _real(self):
        return self.tocsr()._real()

    def _imag(self):
        return self.tocsr()._imag()

    def nonzero(self):
        """nonzero indices

        Returns a tuple of arrays (row,col) containing the indices
        of the non-zero elements of the array.

        Examples
        --------
        >>> from scipy.sparse import csr_array
        >>> A = csr_array([[1,2,0],[0,0,3],[4,0,5]])
        >>> A.nonzero()
        (array([0, 0, 1, 2, 2]), array([0, 1, 2, 0, 2]))

        """

        # convert to COOrdinate format
        A = self.tocoo()
        nz_mask = A.data != 0
        return (A.row[nz_mask], A.col[nz_mask])

    def _getcol(self, j):
        """Returns a copy of column j of the array, as an (m x 1) sparse
        array (column vector).
        """
        # Subclasses should override this method for efficiency.
        # Post-multiply by a (n x 1) column vector 'a' containing all zeros
        # except for a_j = 1
        n = self.shape[1]
        if j < 0:
            j += n
        if j < 0 or j >= n:
            raise IndexError("index out of bounds")
        col_selector = self._csc_container(([1], [[j], [0]]),
                                           shape=(n, 1), dtype=self.dtype)
        return self @ col_selector

    def _getrow(self, i):
        """Returns a copy of row i of the array, as a (1 x n) sparse
        array (row vector).
        """
        # Subclasses should override this method for efficiency.
        # Pre-multiply by a (1 x m) row vector 'a' containing all zeros
        # except for a_i = 1
        m = self.shape[0]
        if i < 0:
            i += m
        if i < 0 or i >= m:
            raise IndexError("index out of bounds")
        row_selector = self._csr_container(([1], [[0], [i]]),
                                           shape=(1, m), dtype=self.dtype)
        return row_selector @ self

    # The following dunder methods cannot be implemented.
    #
    # def __array__(self):
    #     # Sparse matrices rely on NumPy wrapping them in object arrays under
    #     # the hood to make unary ufuncs work on them. So we cannot raise
    #     # TypeError here - which would be handy to not give users object
    #     # arrays they probably don't want (they're looking for `.toarray()`).
    #     #
    #     # Conversion with `toarray()` would also break things because of the
    #     # behavior discussed above, plus we want to avoid densification by
    #     # accident because that can too easily blow up memory.
    #
    # def __array_ufunc__(self):
    #     # We cannot implement __array_ufunc__ due to mismatching semantics.
    #     # See gh-7707 and gh-7349 for details.
    #
    # def __array_function__(self):
    #     # We cannot implement __array_function__ due to mismatching semantics.
    #     # See gh-10362 for details.

    def todense(self, order=None, out=None):
        """
        Return a dense matrix representation of this sparse array.

        Parameters
        ----------
        order : {'C', 'F'}, optional
            Whether to store multi-dimensional data in C (row-major)
            or Fortran (column-major) order in memory. The default
            is 'None', which provides no ordering guarantees.
            Cannot be specified in conjunction with the `out`
            argument.

        out : ndarray, 2-D, optional
            If specified, uses this array (or `numpy.matrix`) as the
            output buffer instead of allocating a new array to
            return. The provided array must have the same shape and
            dtype as the sparse array on which you are calling the
            method.

        Returns
        -------
        arr : numpy.matrix, 2-D
            A NumPy matrix object with the same shape and containing
            the same data represented by the sparse array, with the
            requested memory order. If `out` was passed and was an
            array (rather than a `numpy.matrix`), it will be filled
            with the appropriate values and returned wrapped in a
            `numpy.matrix` object that shares the same memory.
        """
        return self._ascontainer(self.toarray(order=order, out=out))

    def toarray(self, order=None, out=None):
        """
        Return a dense ndarray representation of this sparse array.

        Parameters
        ----------
        order : {'C', 'F'}, optional
            Whether to store multidimensional data in C (row-major)
            or Fortran (column-major) order in memory. The default
            is 'None', which provides no ordering guarantees.
            Cannot be specified in conjunction with the `out`
            argument.

        out : ndarray, 2-D, optional
            If specified, uses this array as the output buffer
            instead of allocating a new array to return. The provided
            array must have the same shape and dtype as the sparse
            array on which you are calling the method. For most
            sparse types, `out` is required to be memory contiguous
            (either C or Fortran ordered).

        Returns
        -------
        arr : ndarray, 2-D
            An array with the same shape and containing the same
            data represented by the sparse array, with the requested
            memory order. If `out` was passed, the same object is
            returned after being modified in-place to contain the
            appropriate values.
        """
        return self.tocoo(copy=False).toarray(order=order, out=out)

    # Any sparse array format deriving from _spbase must define one of
    # tocsr or tocoo. The other conversion methods may be implemented for
    # efficiency, but are not required.
    def tocsr(self, copy=False):
        """Convert this array to Compressed Sparse Row format.

        With copy=False, the data/indices may be shared between this array and
        the resultant csr_array.
        """
        return self.tocoo(copy=copy).tocsr(copy=False)

    def todok(self, copy=False):
        """Convert this array to Dictionary Of Keys format.

        With copy=False, the data/indices may be shared between this array and
        the resultant dok_array.
        """
        return self.tocoo(copy=copy).todok(copy=False)

    def tocoo(self, copy=False):
        """Convert this array to COOrdinate format.

        With copy=False, the data/indices may be shared between this array and
        the resultant coo_array.
        """
        return self.tocsr(copy=False).tocoo(copy=copy)

    def tolil(self, copy=False):
        """Convert this array to List of Lists format.

        With copy=False, the data/indices may be shared between this array and
        the resultant lil_array.
        """
        return self.tocsr(copy=False).tolil(copy=copy)

    def todia(self, copy=False):
        """Convert this array to sparse DIAgonal format.

        With copy=False, the data/indices may be shared between this array and
        the resultant dia_array.
        """
        return self.tocoo(copy=copy).todia(copy=False)

    def tobsr(self, blocksize=None, copy=False):
        """Convert this array to Block Sparse Row format.

        With copy=False, the data/indices may be shared between this array and
        the resultant bsr_array.

        When blocksize=(R, C) is provided, it will be used for construction of
        the bsr_array.
        """
        return self.tocsr(copy=False).tobsr(blocksize=blocksize, copy=copy)

    def tocsc(self, copy=False):
        """Convert this array to Compressed Sparse Column format.

        With copy=False, the data/indices may be shared between this array and
        the resultant csc_array.
        """
        return self.tocsr(copy=copy).tocsc(copy=False)

    def copy(self):
        """Returns a copy of this array.

        No data/indices will be shared between the returned value and current
        array.
        """
        return self.__class__(self, copy=True)

    def sum(self, axis=None, dtype=None, out=None):
        """
        Sum the array elements over a given axis.

        Parameters
        ----------
        axis : {-2, -1, 0, 1, None} optional
            Axis along which the sum is computed. The default is to
            compute the sum of all the array elements, returning a scalar
            (i.e., `axis` = `None`).
        dtype : dtype, optional
            The type of the returned array and of the accumulator in which
            the elements are summed.  The dtype of `a` is used by default
            unless `a` has an integer dtype of less precision than the default
            platform integer.  In that case, if `a` is signed then the platform
            integer is used while if `a` is unsigned then an unsigned integer
            of the same precision as the platform integer is used.

            .. versionadded:: 0.18.0

        out : np.matrix, optional
            Alternative output matrix in which to place the result. It must
            have the same shape as the expected output, but the type of the
            output values will be cast if necessary.

            .. versionadded:: 0.18.0

        Returns
        -------
        sum_along_axis : np.matrix
            A matrix with the same shape as `self`, with the specified
            axis removed.

        See Also
        --------
        numpy.matrix.sum : NumPy's implementation of 'sum' for matrices

        """
        validateaxis(axis)

        # We use multiplication by a matrix of ones to achieve this.
        # For some sparse array formats more efficient methods are
        # possible -- these should override this function.
        m, n = self.shape

        # Mimic numpy's casting.
        res_dtype = get_sum_dtype(self.dtype)

        if axis is None:
            # sum over rows and columns
            return (
                self @ self._ascontainer(np.ones((n, 1), dtype=res_dtype))
            ).sum(dtype=dtype, out=out)

        if axis < 0:
            axis += 2

        # axis = 0 or 1 now
        if axis == 0:
            # sum over columns
            ret = self._ascontainer(
                np.ones((1, m), dtype=res_dtype)
            ) @ self
        else:
            # sum over rows
            ret = self @ self._ascontainer(
                np.ones((n, 1), dtype=res_dtype)
            )

        if out is not None and out.shape != ret.shape:
            raise ValueError("dimensions do not match")

        return ret.sum(axis=axis, dtype=dtype, out=out)

    def mean(self, axis=None, dtype=None, out=None):
        """
        Compute the arithmetic mean along the specified axis.

        Returns the average of the array elements. The average is taken
        over all elements in the array by default, otherwise over the
        specified axis. `float64` intermediate and return values are used
        for integer inputs.

        Parameters
        ----------
        axis : {-2, -1, 0, 1, None} optional
            Axis along which the mean is computed. The default is to compute
            the mean of all elements in the array (i.e., `axis` = `None`).
        dtype : data-type, optional
            Type to use in computing the mean. For integer inputs, the default
            is `float64`; for floating point inputs, it is the same as the
            input dtype.

            .. versionadded:: 0.18.0

        out : np.matrix, optional
            Alternative output matrix in which to place the result. It must
            have the same shape as the expected output, but the type of the
            output values will be cast if necessary.

            .. versionadded:: 0.18.0

        Returns
        -------
        m : np.matrix

        See Also
        --------
        numpy.matrix.mean : NumPy's implementation of 'mean' for matrices

        """
        def _is_integral(dtype):
            return (np.issubdtype(dtype, np.integer) or
                    np.issubdtype(dtype, np.bool_))

        validateaxis(axis)

        res_dtype = self.dtype.type
        integral = _is_integral(self.dtype)

        # output dtype
        if dtype is None:
            if integral:
                res_dtype = np.float64
        else:
            res_dtype = np.dtype(dtype).type

        # intermediate dtype for summation
        inter_dtype = np.float64 if integral else res_dtype
        inter_self = self.astype(inter_dtype)

        if axis is None:
            return (inter_self / np.array(
                self.shape[0] * self.shape[1]))\
                .sum(dtype=res_dtype, out=out)

        if axis < 0:
            axis += 2

        # axis = 0 or 1 now
        if axis == 0:
            return (inter_self * (1.0 / self.shape[0])).sum(
                axis=0, dtype=res_dtype, out=out)
        else:
            return (inter_self * (1.0 / self.shape[1])).sum(
                axis=1, dtype=res_dtype, out=out)

    def diagonal(self, k=0):
        """Returns the kth diagonal of the array.

        Parameters
        ----------
        k : int, optional
            Which diagonal to get, corresponding to elements a[i, i+k].
            Default: 0 (the main diagonal).

            .. versionadded:: 1.0

        See also
        --------
        numpy.diagonal : Equivalent numpy function.

        Examples
        --------
        >>> from scipy.sparse import csr_array
        >>> A = csr_array([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
        >>> A.diagonal()
        array([1, 0, 5])
        >>> A.diagonal(k=1)
        array([2, 3])
        """
        return self.tocsr().diagonal(k=k)

    def trace(self, offset=0):
        """Returns the sum along diagonals of the sparse array.

        Parameters
        ----------
        offset : int, optional
            Which diagonal to get, corresponding to elements a[i, i+offset].
            Default: 0 (the main diagonal).

        """
        return self.diagonal(k=offset).sum()

    def setdiag(self, values, k=0):
        """
        Set diagonal or off-diagonal elements of the array.

        Parameters
        ----------
        values : array_like
            New values of the diagonal elements.

            Values may have any length. If the diagonal is longer than values,
            then the remaining diagonal entries will not be set. If values are
            longer than the diagonal, then the remaining values are ignored.

            If a scalar value is given, all of the diagonal is set to it.

        k : int, optional
            Which off-diagonal to set, corresponding to elements a[i,i+k].
            Default: 0 (the main diagonal).

        """
        M, N = self.shape
        if (k > 0 and k >= N) or (k < 0 and -k >= M):
            raise ValueError("k exceeds array dimensions")
        self._setdiag(np.asarray(values), k)

    def _setdiag(self, values, k):
        """This part of the implementation gets overridden by the
        different formats.
        """
        M, N = self.shape
        if k < 0:
            if values.ndim == 0:
                # broadcast
                max_index = min(M+k, N)
                for i in range(max_index):
                    self[i - k, i] = values
            else:
                max_index = min(M+k, N, len(values))
                if max_index <= 0:
                    return
                for i, v in enumerate(values[:max_index]):
                    self[i - k, i] = v
        else:
            if values.ndim == 0:
                # broadcast
                max_index = min(M, N-k)
                for i in range(max_index):
                    self[i, i + k] = values
            else:
                max_index = min(M, N-k, len(values))
                if max_index <= 0:
                    return
                for i, v in enumerate(values[:max_index]):
                    self[i, i + k] = v

    def _process_toarray_args(self, order, out):
        if out is not None:
            if order is not None:
                raise ValueError('order cannot be specified if out '
                                 'is not None')
            if out.shape != self.shape or out.dtype != self.dtype:
                raise ValueError('out array must be same dtype and shape as '
                                 'sparse array')
            out[...] = 0.
            return out
        else:
            return np.zeros(self.shape, dtype=self.dtype, order=order)

    def _get_index_dtype(self, arrays=(), maxval=None, check_contents=False):
        """
        Determine index dtype for array.

        This wraps _sputils.get_index_dtype, providing compatibility for both
        array and matrix API sparse matrices. Matrix API sparse matrices would
        attempt to downcast the indices - which can be computationally
        expensive and undesirable for users. The array API changes this
        behaviour.

        See discussion: https://github.com/scipy/scipy/issues/16774

        The get_index_dtype import is due to implementation details of the test
        suite. It allows the decorator ``with_64bit_maxval_limit`` to mock a
        lower int32 max value for checks on the matrix API's downcasting
        behaviour.
        """
        from ._sputils import get_index_dtype

        # Don't check contents for array API
        return get_index_dtype(arrays, maxval, (check_contents and not self._is_array))


    ## All methods below are deprecated and should be removed in
    ## scipy 1.13.0
    ##
    ## Also uncomment the definition of shape above.

    def get_shape(self):
        """Get shape of a sparse array.

        .. deprecated:: 1.11.0
           This method will be removed in SciPy 1.13.0.
           Use `X.shape` instead.
        """
        msg = (
            "`get_shape` is deprecated and will be removed in v1.13.0; "
            "use `X.shape` instead."
        )
        warn(msg, DeprecationWarning, stacklevel=2)

        return self._shape

    def set_shape(self, shape):
        """See `reshape`.

        .. deprecated:: 1.11.0
           This method will be removed in SciPy 1.13.0.
           Use `X.reshape` instead.
        """
        msg = (
            "Shape assignment is deprecated and will be removed in v1.13.0; "
            "use `reshape` instead."
        )
        warn(msg, DeprecationWarning, stacklevel=2)

        # Make sure copy is False since this is in place
        # Make sure format is unchanged because we are doing a __dict__ swap
        new_self = self.reshape(shape, copy=False).asformat(self.format)
        self.__dict__ = new_self.__dict__

    shape = property(
        fget=lambda self: self._shape,
        fset=set_shape,
        doc="""The shape of the array.

Note that, starting in SciPy 1.13.0, this property will no longer be
settable. To change the array shape, use `X.reshape` instead.
"""
    )  # noqa: F811

    def asfptype(self):
        """Upcast array to a floating point format (if necessary)

        .. deprecated:: 1.11.0
           This method is for internal use only, and will be removed from the
           public API in SciPy 1.13.0.
        """
        msg = (
            "`asfptype` is an internal function, and is deprecated "
            "as part of the public API. It will be removed in v1.13.0."
        )
        warn(msg, DeprecationWarning, stacklevel=2)
        return self._asfptype()

    def getmaxprint(self):
        """Maximum number of elements to display when printed.

        .. deprecated:: 1.11.0
           This method is for internal use only, and will be removed from the
           public API in SciPy 1.13.0.
        """
        msg = (
            "`getmaxprint` is an internal function, and is deprecated "
            "as part of the public API. It will be removed in v1.13.0."
        )
        warn(msg, DeprecationWarning, stacklevel=2)
        return self._getmaxprint()

    def getformat(self):
        """Matrix storage format.

        .. deprecated:: 1.11.0
           This method will be removed in SciPy 1.13.0.
           Use `X.format` instead.
        """
        msg = (
            "`getformat` is deprecated and will be removed in v1.13.0; "
            "use `X.format` instead."
        )
        warn(msg, DeprecationWarning, stacklevel=2)
        return self.format

    def getnnz(self, axis=None):
        """Number of stored values, including explicit zeros.

        .. deprecated:: 1.11.0
           This method will be removed in SciPy 1.13.0. Use `X.nnz`
           instead.  The `axis` argument will no longer be supported;
           please let us know if you still need this functionality.

        Parameters
        ----------
        axis : None, 0, or 1
            Select between the number of values across the whole array, in
            each column, or in each row.

        See also
        --------
        count_nonzero : Number of non-zero entries
        """
        msg = (
            "`getnnz` is deprecated and will be removed in v1.13.0; "
            "use `X.nnz` instead."
        )
        warn(msg, DeprecationWarning, stacklevel=2)
        return self._getnnz(axis=axis)

    def getH(self):
        """Return the Hermitian transpose of this array.

        .. deprecated:: 1.11.0
           This method will be removed in SciPy 1.13.0.
           Use `X.conj().T` instead.
        """
        msg = (
            "`getH` is deprecated and will be removed in v1.13.0; "
            "use `X.conj().T` instead."
        )
        warn(msg, DeprecationWarning, stacklevel=2)
        return self.conjugate().transpose()

    def getcol(self, j):
        """Returns a copy of column j of the array, as an (m x 1) sparse
        array (column vector).

        .. deprecated:: 1.11.0
           This method will be removed in SciPy 1.13.0.
           Use array indexing instead.
        """
        msg = (
            "`getcol` is deprecated and will be removed in v1.13.0; "
            f"use `X[:, [{j}]]` instead."
        )
        warn(msg, DeprecationWarning, stacklevel=2)
        return self._getcol(j)

    def getrow(self, i):
        """Returns a copy of row i of the array, as a (1 x n) sparse
        array (row vector).

        .. deprecated:: 1.11.0
           This method will be removed in SciPy 1.13.0.
           Use array indexing instead.
        """
        msg = (
            "`getrow` is deprecated and will be removed in v1.13.0; "
            f"use `X[[{i}]]` instead."
        )
        warn(msg, DeprecationWarning, stacklevel=2)
        return self._getrow(i)

    ## End 1.13.0 deprecated methods


class sparray:
    """A namespace class to separate sparray from spmatrix"""
    pass

sparray.__doc__ = _spbase.__doc__


def issparse(x):
    """Is `x` of a sparse array type?

    Parameters
    ----------
    x
        object to check for being a sparse array

    Returns
    -------
    bool
        True if `x` is a sparse array or a sparse matrix, False otherwise

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csr_array, csr_matrix, issparse
    >>> issparse(csr_matrix([[5]]))
    True
    >>> issparse(csr_array([[5]]))
    True
    >>> issparse(np.array([[5]]))
    False
    >>> issparse(5)
    False
    """
    return isinstance(x, _spbase)


def isspmatrix(x):
    """Is `x` of a sparse matrix type?

    Parameters
    ----------
    x
        object to check for being a sparse matrix

    Returns
    -------
    bool
        True if `x` is a sparse matrix, False otherwise

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csr_array, csr_matrix, isspmatrix
    >>> isspmatrix(csr_matrix([[5]]))
    True
    >>> isspmatrix(csr_array([[5]]))
    False
    >>> isspmatrix(np.array([[5]]))
    False
    >>> isspmatrix(5)
    False
    """
    return isinstance(x, spmatrix)
