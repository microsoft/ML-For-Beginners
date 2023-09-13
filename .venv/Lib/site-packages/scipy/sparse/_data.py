"""Base class for sparse matrice with a .data attribute

    subclasses must provide a _with_data() method that
    creates a new matrix with the same sparsity pattern
    as self but with a different data array

"""

import numpy as np

from ._base import _spbase, _ufuncs_with_fixed_point_at_zero
from ._sputils import isscalarlike, validateaxis

__all__ = []


# TODO implement all relevant operations
# use .data.__methods__() instead of /=, *=, etc.
class _data_matrix(_spbase):
    def __init__(self):
        _spbase.__init__(self)

    def _get_dtype(self):
        return self.data.dtype

    def _set_dtype(self, newtype):
        self.data.dtype = newtype
    dtype = property(fget=_get_dtype, fset=_set_dtype)

    def _deduped_data(self):
        if hasattr(self, 'sum_duplicates'):
            self.sum_duplicates()
        return self.data

    def __abs__(self):
        return self._with_data(abs(self._deduped_data()))

    def __round__(self, ndigits=0):
        return self._with_data(np.around(self._deduped_data(), decimals=ndigits))

    def _real(self):
        return self._with_data(self.data.real)

    def _imag(self):
        return self._with_data(self.data.imag)

    def __neg__(self):
        if self.dtype.kind == 'b':
            raise NotImplementedError('negating a boolean sparse array is not '
                                      'supported')
        return self._with_data(-self.data)

    def __imul__(self, other):  # self *= other
        if isscalarlike(other):
            self.data *= other
            return self
        else:
            return NotImplemented

    def __itruediv__(self, other):  # self /= other
        if isscalarlike(other):
            recip = 1.0 / other
            self.data *= recip
            return self
        else:
            return NotImplemented

    def astype(self, dtype, casting='unsafe', copy=True):
        dtype = np.dtype(dtype)
        if self.dtype != dtype:
            matrix = self._with_data(
                self.data.astype(dtype, casting=casting, copy=True),
                copy=True
            )
            return matrix._with_data(matrix._deduped_data(), copy=False)
        elif copy:
            return self.copy()
        else:
            return self

    astype.__doc__ = _spbase.astype.__doc__

    def conjugate(self, copy=True):
        if np.issubdtype(self.dtype, np.complexfloating):
            return self._with_data(self.data.conjugate(), copy=copy)
        elif copy:
            return self.copy()
        else:
            return self

    conjugate.__doc__ = _spbase.conjugate.__doc__

    def copy(self):
        return self._with_data(self.data.copy(), copy=True)

    copy.__doc__ = _spbase.copy.__doc__

    def count_nonzero(self):
        return np.count_nonzero(self._deduped_data())

    count_nonzero.__doc__ = _spbase.count_nonzero.__doc__

    def power(self, n, dtype=None):
        """
        This function performs element-wise power.

        Parameters
        ----------
        n : n is a scalar

        dtype : If dtype is not specified, the current dtype will be preserved.
        """
        if not isscalarlike(n):
            raise NotImplementedError("input is not scalar")

        data = self._deduped_data()
        if dtype is not None:
            data = data.astype(dtype)
        return self._with_data(data ** n)

    ###########################
    # Multiplication handlers #
    ###########################

    def _mul_scalar(self, other):
        return self._with_data(self.data * other)


# Add the numpy unary ufuncs for which func(0) = 0 to _data_matrix.
for npfunc in _ufuncs_with_fixed_point_at_zero:
    name = npfunc.__name__

    def _create_method(op):
        def method(self):
            result = op(self._deduped_data())
            return self._with_data(result, copy=True)

        method.__doc__ = ("Element-wise {}.\n\n"
                          "See `numpy.{}` for more information.".format(name, name))
        method.__name__ = name

        return method

    setattr(_data_matrix, name, _create_method(npfunc))


def _find_missing_index(ind, n):
    for k, a in enumerate(ind):
        if k != a:
            return k

    k += 1
    if k < n:
        return k
    else:
        return -1


class _minmax_mixin:
    """Mixin for min and max methods.

    These are not implemented for dia_matrix, hence the separate class.
    """

    def _min_or_max_axis(self, axis, min_or_max):
        N = self.shape[axis]
        if N == 0:
            raise ValueError("zero-size array to reduction operation")
        M = self.shape[1 - axis]
        idx_dtype = self._get_index_dtype(maxval=M)

        mat = self.tocsc() if axis == 0 else self.tocsr()
        mat.sum_duplicates()

        major_index, value = mat._minor_reduce(min_or_max)
        not_full = np.diff(mat.indptr)[major_index] < N
        value[not_full] = min_or_max(value[not_full], 0)

        mask = value != 0
        major_index = np.compress(mask, major_index)
        value = np.compress(mask, value)

        if axis == 0:
            return self._coo_container(
                (value, (np.zeros(len(value), dtype=idx_dtype), major_index)),
                dtype=self.dtype, shape=(1, M)
            )
        else:
            return self._coo_container(
                (value, (major_index, np.zeros(len(value), dtype=idx_dtype))),
                dtype=self.dtype, shape=(M, 1)
            )

    def _min_or_max(self, axis, out, min_or_max):
        if out is not None:
            raise ValueError("Sparse matrices do not support "
                              "an 'out' parameter.")

        validateaxis(axis)

        if axis is None:
            if 0 in self.shape:
                raise ValueError("zero-size array to reduction operation")

            zero = self.dtype.type(0)
            if self.nnz == 0:
                return zero
            m = min_or_max.reduce(self._deduped_data().ravel())
            if self.nnz != np.prod(self.shape):
                m = min_or_max(zero, m)
            return m

        if axis < 0:
            axis += 2

        if (axis == 0) or (axis == 1):
            return self._min_or_max_axis(axis, min_or_max)
        else:
            raise ValueError("axis out of range")

    def _arg_min_or_max_axis(self, axis, argmin_or_argmax, compare):
        if self.shape[axis] == 0:
            raise ValueError("Can't apply the operation along a zero-sized "
                             "dimension.")

        if axis < 0:
            axis += 2

        zero = self.dtype.type(0)

        mat = self.tocsc() if axis == 0 else self.tocsr()
        mat.sum_duplicates()

        ret_size, line_size = mat._swap(mat.shape)
        ret = np.zeros(ret_size, dtype=int)

        nz_lines, = np.nonzero(np.diff(mat.indptr))
        for i in nz_lines:
            p, q = mat.indptr[i:i + 2]
            data = mat.data[p:q]
            indices = mat.indices[p:q]
            extreme_index = argmin_or_argmax(data)
            extreme_value = data[extreme_index]
            if compare(extreme_value, zero) or q - p == line_size:
                ret[i] = indices[extreme_index]
            else:
                zero_ind = _find_missing_index(indices, line_size)
                if extreme_value == zero:
                    ret[i] = min(extreme_index, zero_ind)
                else:
                    ret[i] = zero_ind

        if axis == 1:
            ret = ret.reshape(-1, 1)

        return self._ascontainer(ret)

    def _arg_min_or_max(self, axis, out, argmin_or_argmax, compare):
        if out is not None:
            raise ValueError("Sparse types do not support an 'out' parameter.")

        validateaxis(axis)

        if axis is not None:
            return self._arg_min_or_max_axis(axis, argmin_or_argmax, compare)

        if 0 in self.shape:
            raise ValueError("Can't apply the operation to an empty matrix.")

        if self.nnz == 0:
            return 0

        zero = self.dtype.type(0)
        mat = self.tocoo()
        # Convert to canonical form: no duplicates, sorted indices.
        mat.sum_duplicates()
        extreme_index = argmin_or_argmax(mat.data)
        extreme_value = mat.data[extreme_index]
        num_row, num_col = mat.shape

        # If the min value is less than zero, or max is greater than zero,
        # then we don't need to worry about implicit zeros.
        if compare(extreme_value, zero):
            # cast to Python int to avoid overflow and RuntimeError
            return (int(mat.row[extreme_index]) * num_col +
                    int(mat.col[extreme_index]))

        # Cheap test for the rare case where we have no implicit zeros.
        size = num_row * num_col
        if size == mat.nnz:
            return (int(mat.row[extreme_index]) * num_col +
                    int(mat.col[extreme_index]))

        # At this stage, any implicit zero could be the min or max value.
        # After sum_duplicates(), the `row` and `col` arrays are guaranteed to
        # be sorted in C-order, which means the linearized indices are sorted.
        linear_indices = mat.row * num_col + mat.col
        first_implicit_zero_index = _find_missing_index(linear_indices, size)
        if extreme_value == zero:
            return min(first_implicit_zero_index, extreme_index)
        return first_implicit_zero_index

    def max(self, axis=None, out=None):
        """
        Return the maximum of the matrix or maximum along an axis.
        This takes all elements into account, not just the non-zero ones.

        Parameters
        ----------
        axis : {-2, -1, 0, 1, None} optional
            Axis along which the sum is computed. The default is to
            compute the maximum over all the matrix elements, returning
            a scalar (i.e., `axis` = `None`).

        out : None, optional
            This argument is in the signature *solely* for NumPy
            compatibility reasons. Do not pass in anything except
            for the default value, as this argument is not used.

        Returns
        -------
        amax : coo_matrix or scalar
            Maximum of `a`. If `axis` is None, the result is a scalar value.
            If `axis` is given, the result is a sparse.coo_matrix of dimension
            ``a.ndim - 1``.

        See Also
        --------
        min : The minimum value of a sparse matrix along a given axis.
        numpy.matrix.max : NumPy's implementation of 'max' for matrices

        """
        return self._min_or_max(axis, out, np.maximum)

    def min(self, axis=None, out=None):
        """
        Return the minimum of the matrix or maximum along an axis.
        This takes all elements into account, not just the non-zero ones.

        Parameters
        ----------
        axis : {-2, -1, 0, 1, None} optional
            Axis along which the sum is computed. The default is to
            compute the minimum over all the matrix elements, returning
            a scalar (i.e., `axis` = `None`).

        out : None, optional
            This argument is in the signature *solely* for NumPy
            compatibility reasons. Do not pass in anything except for
            the default value, as this argument is not used.

        Returns
        -------
        amin : coo_matrix or scalar
            Minimum of `a`. If `axis` is None, the result is a scalar value.
            If `axis` is given, the result is a sparse.coo_matrix of dimension
            ``a.ndim - 1``.

        See Also
        --------
        max : The maximum value of a sparse matrix along a given axis.
        numpy.matrix.min : NumPy's implementation of 'min' for matrices

        """
        return self._min_or_max(axis, out, np.minimum)

    def nanmax(self, axis=None, out=None):
        """
        Return the maximum of the matrix or maximum along an axis, ignoring any
        NaNs. This takes all elements into account, not just the non-zero
        ones.

        .. versionadded:: 1.11.0

        Parameters
        ----------
        axis : {-2, -1, 0, 1, None} optional
            Axis along which the maximum is computed. The default is to
            compute the maximum over all the matrix elements, returning
            a scalar (i.e., `axis` = `None`).

        out : None, optional
            This argument is in the signature *solely* for NumPy
            compatibility reasons. Do not pass in anything except
            for the default value, as this argument is not used.

        Returns
        -------
        amax : coo_matrix or scalar
            Maximum of `a`. If `axis` is None, the result is a scalar value.
            If `axis` is given, the result is a sparse.coo_matrix of dimension
            ``a.ndim - 1``.

        See Also
        --------
        nanmin : The minimum value of a sparse matrix along a given axis,
                 ignoring NaNs.
        max : The maximum value of a sparse matrix along a given axis,
              propagating NaNs.
        numpy.nanmax : NumPy's implementation of 'nanmax'.

        """
        return self._min_or_max(axis, out, np.fmax)

    def nanmin(self, axis=None, out=None):
        """
        Return the minimum of the matrix or minimum along an axis, ignoring any
        NaNs. This takes all elements into account, not just the non-zero
        ones.

        .. versionadded:: 1.11.0

        Parameters
        ----------
        axis : {-2, -1, 0, 1, None} optional
            Axis along which the minimum is computed. The default is to
            compute the minimum over all the matrix elements, returning
            a scalar (i.e., `axis` = `None`).

        out : None, optional
            This argument is in the signature *solely* for NumPy
            compatibility reasons. Do not pass in anything except for
            the default value, as this argument is not used.

        Returns
        -------
        amin : coo_matrix or scalar
            Minimum of `a`. If `axis` is None, the result is a scalar value.
            If `axis` is given, the result is a sparse.coo_matrix of dimension
            ``a.ndim - 1``.

        See Also
        --------
        nanmax : The maximum value of a sparse matrix along a given axis,
                 ignoring NaNs.
        min : The minimum value of a sparse matrix along a given axis,
              propagating NaNs.
        numpy.nanmin : NumPy's implementation of 'nanmin'.

        """
        return self._min_or_max(axis, out, np.fmin)

    def argmax(self, axis=None, out=None):
        """Return indices of maximum elements along an axis.

        Implicit zero elements are also taken into account. If there are
        several maximum values, the index of the first occurrence is returned.

        Parameters
        ----------
        axis : {-2, -1, 0, 1, None}, optional
            Axis along which the argmax is computed. If None (default), index
            of the maximum element in the flatten data is returned.
        out : None, optional
            This argument is in the signature *solely* for NumPy
            compatibility reasons. Do not pass in anything except for
            the default value, as this argument is not used.

        Returns
        -------
        ind : numpy.matrix or int
            Indices of maximum elements. If matrix, its size along `axis` is 1.
        """
        return self._arg_min_or_max(axis, out, np.argmax, np.greater)

    def argmin(self, axis=None, out=None):
        """Return indices of minimum elements along an axis.

        Implicit zero elements are also taken into account. If there are
        several minimum values, the index of the first occurrence is returned.

        Parameters
        ----------
        axis : {-2, -1, 0, 1, None}, optional
            Axis along which the argmin is computed. If None (default), index
            of the minimum element in the flatten data is returned.
        out : None, optional
            This argument is in the signature *solely* for NumPy
            compatibility reasons. Do not pass in anything except for
            the default value, as this argument is not used.

        Returns
        -------
         ind : numpy.matrix or int
            Indices of minimum elements. If matrix, its size along `axis` is 1.
        """
        return self._arg_min_or_max(axis, out, np.argmin, np.less)
