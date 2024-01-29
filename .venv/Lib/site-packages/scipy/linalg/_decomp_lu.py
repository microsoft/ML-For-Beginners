"""LU decomposition functions."""

from warnings import warn

from numpy import asarray, asarray_chkfinite
import numpy as np
from itertools import product

# Local imports
from ._misc import _datacopied, LinAlgWarning
from .lapack import get_lapack_funcs
from ._decomp_lu_cython import lu_dispatcher

# deprecated imports to be removed in SciPy 1.13.0
from scipy.linalg._flinalg_py import get_flinalg_funcs  # noqa: F401


lapack_cast_dict = {x: ''.join([y for y in 'fdFD' if np.can_cast(x, y)])
                    for x in np.typecodes['All']}

__all__ = ['lu', 'lu_solve', 'lu_factor']


def lu_factor(a, overwrite_a=False, check_finite=True):
    """
    Compute pivoted LU decomposition of a matrix.

    The decomposition is::

        A = P L U

    where P is a permutation matrix, L lower triangular with unit
    diagonal elements, and U upper triangular.

    Parameters
    ----------
    a : (M, N) array_like
        Matrix to decompose
    overwrite_a : bool, optional
        Whether to overwrite data in A (may increase performance)
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    lu : (M, N) ndarray
        Matrix containing U in its upper triangle, and L in its lower triangle.
        The unit diagonal elements of L are not stored.
    piv : (K,) ndarray
        Pivot indices representing the permutation matrix P:
        row i of matrix was interchanged with row piv[i].
        Of shape ``(K,)``, with ``K = min(M, N)``.

    See Also
    --------
    lu : gives lu factorization in more user-friendly format
    lu_solve : solve an equation system using the LU factorization of a matrix

    Notes
    -----
    This is a wrapper to the ``*GETRF`` routines from LAPACK. Unlike
    :func:`lu`, it outputs the L and U factors into a single array
    and returns pivot indices instead of a permutation matrix.

    While the underlying ``*GETRF`` routines return 1-based pivot indices, the
    ``piv`` array returned by ``lu_factor`` contains 0-based indices.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import lu_factor
    >>> A = np.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]])
    >>> lu, piv = lu_factor(A)
    >>> piv
    array([2, 2, 3, 3], dtype=int32)

    Convert LAPACK's ``piv`` array to NumPy index and test the permutation

    >>> def pivot_to_permutation(piv):
    ...     perm = np.arange(len(piv))
    ...     for i in range(len(piv)):
    ...         perm[i], perm[piv[i]] = perm[piv[i]], perm[i]
    ...     return perm
    ...
    >>> p_inv = pivot_to_permutation(piv)
    >>> p_inv
    array([2, 0, 3, 1])
    >>> L, U = np.tril(lu, k=-1) + np.eye(4), np.triu(lu)
    >>> np.allclose(A[p_inv] - L @ U, np.zeros((4, 4)))
    True

    The P matrix in P L U is defined by the inverse permutation and
    can be recovered using argsort:

    >>> p = np.argsort(p_inv)
    >>> p
    array([1, 3, 0, 2])
    >>> np.allclose(A - L[p] @ U, np.zeros((4, 4)))
    True

    or alternatively:

    >>> P = np.eye(4)[p]
    >>> np.allclose(A - P @ L @ U, np.zeros((4, 4)))
    True
    """
    if check_finite:
        a1 = asarray_chkfinite(a)
    else:
        a1 = asarray(a)
    overwrite_a = overwrite_a or (_datacopied(a1, a))
    getrf, = get_lapack_funcs(('getrf',), (a1,))
    lu, piv, info = getrf(a1, overwrite_a=overwrite_a)
    if info < 0:
        raise ValueError('illegal value in %dth argument of '
                         'internal getrf (lu_factor)' % -info)
    if info > 0:
        warn("Diagonal number %d is exactly zero. Singular matrix." % info,
             LinAlgWarning, stacklevel=2)
    return lu, piv


def lu_solve(lu_and_piv, b, trans=0, overwrite_b=False, check_finite=True):
    """Solve an equation system, a x = b, given the LU factorization of a

    Parameters
    ----------
    (lu, piv)
        Factorization of the coefficient matrix a, as given by lu_factor.
        In particular piv are 0-indexed pivot indices.
    b : array
        Right-hand side
    trans : {0, 1, 2}, optional
        Type of system to solve:

        =====  =========
        trans  system
        =====  =========
        0      a x   = b
        1      a^T x = b
        2      a^H x = b
        =====  =========
    overwrite_b : bool, optional
        Whether to overwrite data in b (may increase performance)
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    x : array
        Solution to the system

    See Also
    --------
    lu_factor : LU factorize a matrix

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import lu_factor, lu_solve
    >>> A = np.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]])
    >>> b = np.array([1, 1, 1, 1])
    >>> lu, piv = lu_factor(A)
    >>> x = lu_solve((lu, piv), b)
    >>> np.allclose(A @ x - b, np.zeros((4,)))
    True

    """
    (lu, piv) = lu_and_piv
    if check_finite:
        b1 = asarray_chkfinite(b)
    else:
        b1 = asarray(b)
    overwrite_b = overwrite_b or _datacopied(b1, b)
    if lu.shape[0] != b1.shape[0]:
        raise ValueError(f"Shapes of lu {lu.shape} and b {b1.shape} are incompatible")

    getrs, = get_lapack_funcs(('getrs',), (lu, b1))
    x, info = getrs(lu, piv, b1, trans=trans, overwrite_b=overwrite_b)
    if info == 0:
        return x
    raise ValueError('illegal value in %dth argument of internal gesv|posv'
                     % -info)


def lu(a, permute_l=False, overwrite_a=False, check_finite=True,
       p_indices=False):
    """
    Compute LU decomposition of a matrix with partial pivoting.

    The decomposition satisfies::

        A = P @ L @ U

    where ``P`` is a permutation matrix, ``L`` lower triangular with unit
    diagonal elements, and ``U`` upper triangular. If `permute_l` is set to
    ``True`` then ``L`` is returned already permuted and hence satisfying
    ``A = L @ U``.

    Parameters
    ----------
    a : (M, N) array_like
        Array to decompose
    permute_l : bool, optional
        Perform the multiplication P*L (Default: do not permute)
    overwrite_a : bool, optional
        Whether to overwrite data in a (may improve performance)
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
    p_indices : bool, optional
        If ``True`` the permutation information is returned as row indices.
        The default is ``False`` for backwards-compatibility reasons.

    Returns
    -------
    **(If `permute_l` is ``False``)**

    p : (..., M, M) ndarray
        Permutation arrays or vectors depending on `p_indices`
    l : (..., M, K) ndarray
        Lower triangular or trapezoidal array with unit diagonal.
        ``K = min(M, N)``
    u : (..., K, N) ndarray
        Upper triangular or trapezoidal array

    **(If `permute_l` is ``True``)**

    pl : (..., M, K) ndarray
        Permuted L matrix.
        ``K = min(M, N)``
    u : (..., K, N) ndarray
        Upper triangular or trapezoidal array

    Notes
    -----
    Permutation matrices are costly since they are nothing but row reorder of
    ``L`` and hence indices are strongly recommended to be used instead if the
    permutation is required. The relation in the 2D case then becomes simply
    ``A = L[P, :] @ U``. In higher dimensions, it is better to use `permute_l`
    to avoid complicated indexing tricks.

    In 2D case, if one has the indices however, for some reason, the
    permutation matrix is still needed then it can be constructed by
    ``np.eye(M)[P, :]``.

    Examples
    --------

    >>> import numpy as np
    >>> from scipy.linalg import lu
    >>> A = np.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]])
    >>> p, l, u = lu(A)
    >>> np.allclose(A, p @ l @ u)
    True
    >>> p  # Permutation matrix
    array([[0., 1., 0., 0.],  # Row index 1
           [0., 0., 0., 1.],  # Row index 3
           [1., 0., 0., 0.],  # Row index 0
           [0., 0., 1., 0.]]) # Row index 2
    >>> p, _, _ = lu(A, p_indices=True)
    >>> p
    array([1, 3, 0, 2])  # as given by row indices above
    >>> np.allclose(A, l[p, :] @ u)
    True

    We can also use nd-arrays, for example, a demonstration with 4D array:

    >>> rng = np.random.default_rng()
    >>> A = rng.uniform(low=-4, high=4, size=[3, 2, 4, 8])
    >>> p, l, u = lu(A)
    >>> p.shape, l.shape, u.shape
    ((3, 2, 4, 4), (3, 2, 4, 4), (3, 2, 4, 8))
    >>> np.allclose(A, p @ l @ u)
    True
    >>> PL, U = lu(A, permute_l=True)
    >>> np.allclose(A, PL @ U)
    True

    """
    a1 = np.asarray_chkfinite(a) if check_finite else np.asarray(a)
    if a1.ndim < 2:
        raise ValueError('The input array must be at least two-dimensional.')

    # Also check if dtype is LAPACK compatible
    if a1.dtype.char not in 'fdFD':
        dtype_char = lapack_cast_dict[a1.dtype.char]
        if not dtype_char:  # No casting possible
            raise TypeError(f'The dtype {a1.dtype} cannot be cast '
                            'to float(32, 64) or complex(64, 128).')

        a1 = a1.astype(dtype_char[0])  # makes a copy, free to scratch
        overwrite_a = True

    *nd, m, n = a1.shape
    k = min(m, n)
    real_dchar = 'f' if a1.dtype.char in 'fF' else 'd'

    # Empty input
    if min(*a1.shape) == 0:
        if permute_l:
            PL = np.empty(shape=[*nd, m, k], dtype=a1.dtype)
            U = np.empty(shape=[*nd, k, n], dtype=a1.dtype)
            return PL, U
        else:
            P = (np.empty([*nd, 0], dtype=np.int32) if p_indices else
                 np.empty([*nd, 0, 0], dtype=real_dchar))
            L = np.empty(shape=[*nd, m, k], dtype=a1.dtype)
            U = np.empty(shape=[*nd, k, n], dtype=a1.dtype)
            return P, L, U

    # Scalar case
    if a1.shape[-2:] == (1, 1):
        if permute_l:
            return np.ones_like(a1), (a1 if overwrite_a else a1.copy())
        else:
            P = (np.zeros(shape=[*nd, m], dtype=int) if p_indices
                 else np.ones_like(a1))
            return P, np.ones_like(a1), (a1 if overwrite_a else a1.copy())

    # Then check overwrite permission
    if not _datacopied(a1, a):  # "a"  still alive through "a1"
        if not overwrite_a:
            # Data belongs to "a" so make a copy
            a1 = a1.copy(order='C')
        #  else: Do nothing we'll use "a" if possible
    # else:  a1 has its own data thus free to scratch

    # Then layout checks, might happen that overwrite is allowed but original
    # array was read-only or non-contiguous.

    if not (a1.flags['C_CONTIGUOUS'] and a1.flags['WRITEABLE']):
        a1 = a1.copy(order='C')

    if not nd:  # 2D array

        p = np.empty(m, dtype=np.int32)
        u = np.zeros([k, k], dtype=a1.dtype)
        lu_dispatcher(a1, u, p, permute_l)
        P, L, U = (p, a1, u) if m > n else (p, u, a1)

    else:  # Stacked array

        # Prepare the contiguous data holders
        P = np.empty([*nd, m], dtype=np.int32)  # perm vecs

        if m > n:  # Tall arrays, U will be created
            U = np.zeros([*nd, k, k], dtype=a1.dtype)
            for ind in product(*[range(x) for x in a1.shape[:-2]]):
                lu_dispatcher(a1[ind], U[ind], P[ind], permute_l)
            L = a1

        else:  # Fat arrays, L will be created
            L = np.zeros([*nd, k, k], dtype=a1.dtype)
            for ind in product(*[range(x) for x in a1.shape[:-2]]):
                lu_dispatcher(a1[ind], L[ind], P[ind], permute_l)
            U = a1

    # Convert permutation vecs to permutation arrays
    # permute_l=False needed to enter here to avoid wasted efforts
    if (not p_indices) and (not permute_l):
        if nd:
            Pa = np.zeros([*nd, m, m], dtype=real_dchar)
            # An unreadable index hack - One-hot encoding for perm matrices
            nd_ix = np.ix_(*([np.arange(x) for x in nd]+[np.arange(m)]))
            Pa[(*nd_ix, P)] = 1
            P = Pa
        else:  # 2D case
            Pa = np.zeros([m, m], dtype=real_dchar)
            Pa[np.arange(m), P] = 1
            P = Pa

    return (L, U) if permute_l else (P, L, U)
