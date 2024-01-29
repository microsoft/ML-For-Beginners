import warnings

import numpy as np
from numpy import asarray_chkfinite
from ._misc import LinAlgError, _datacopied, LinAlgWarning
from .lapack import get_lapack_funcs


__all__ = ['qz', 'ordqz']

_double_precision = ['i', 'l', 'd']


def _select_function(sort):
    if callable(sort):
        # assume the user knows what they're doing
        sfunction = sort
    elif sort == 'lhp':
        sfunction = _lhp
    elif sort == 'rhp':
        sfunction = _rhp
    elif sort == 'iuc':
        sfunction = _iuc
    elif sort == 'ouc':
        sfunction = _ouc
    else:
        raise ValueError("sort parameter must be None, a callable, or "
                         "one of ('lhp','rhp','iuc','ouc')")

    return sfunction


def _lhp(x, y):
    out = np.empty_like(x, dtype=bool)
    nonzero = (y != 0)
    # handles (x, y) = (0, 0) too
    out[~nonzero] = False
    out[nonzero] = (np.real(x[nonzero]/y[nonzero]) < 0.0)
    return out


def _rhp(x, y):
    out = np.empty_like(x, dtype=bool)
    nonzero = (y != 0)
    # handles (x, y) = (0, 0) too
    out[~nonzero] = False
    out[nonzero] = (np.real(x[nonzero]/y[nonzero]) > 0.0)
    return out


def _iuc(x, y):
    out = np.empty_like(x, dtype=bool)
    nonzero = (y != 0)
    # handles (x, y) = (0, 0) too
    out[~nonzero] = False
    out[nonzero] = (abs(x[nonzero]/y[nonzero]) < 1.0)
    return out


def _ouc(x, y):
    out = np.empty_like(x, dtype=bool)
    xzero = (x == 0)
    yzero = (y == 0)
    out[xzero & yzero] = False
    out[~xzero & yzero] = True
    out[~yzero] = (abs(x[~yzero]/y[~yzero]) > 1.0)
    return out


def _qz(A, B, output='real', lwork=None, sort=None, overwrite_a=False,
        overwrite_b=False, check_finite=True):
    if sort is not None:
        # Disabled due to segfaults on win32, see ticket 1717.
        raise ValueError("The 'sort' input of qz() has to be None and will be "
                         "removed in a future release. Use ordqz instead.")

    if output not in ['real', 'complex', 'r', 'c']:
        raise ValueError("argument must be 'real', or 'complex'")

    if check_finite:
        a1 = asarray_chkfinite(A)
        b1 = asarray_chkfinite(B)
    else:
        a1 = np.asarray(A)
        b1 = np.asarray(B)

    a_m, a_n = a1.shape
    b_m, b_n = b1.shape
    if not (a_m == a_n == b_m == b_n):
        raise ValueError("Array dimensions must be square and agree")

    typa = a1.dtype.char
    if output in ['complex', 'c'] and typa not in ['F', 'D']:
        if typa in _double_precision:
            a1 = a1.astype('D')
            typa = 'D'
        else:
            a1 = a1.astype('F')
            typa = 'F'
    typb = b1.dtype.char
    if output in ['complex', 'c'] and typb not in ['F', 'D']:
        if typb in _double_precision:
            b1 = b1.astype('D')
            typb = 'D'
        else:
            b1 = b1.astype('F')
            typb = 'F'

    overwrite_a = overwrite_a or (_datacopied(a1, A))
    overwrite_b = overwrite_b or (_datacopied(b1, B))

    gges, = get_lapack_funcs(('gges',), (a1, b1))

    if lwork is None or lwork == -1:
        # get optimal work array size
        result = gges(lambda x: None, a1, b1, lwork=-1)
        lwork = result[-2][0].real.astype(int)

    def sfunction(x):
        return None
    result = gges(sfunction, a1, b1, lwork=lwork, overwrite_a=overwrite_a,
                  overwrite_b=overwrite_b, sort_t=0)

    info = result[-1]
    if info < 0:
        raise ValueError(f"Illegal value in argument {-info} of gges")
    elif info > 0 and info <= a_n:
        warnings.warn("The QZ iteration failed. (a,b) are not in Schur "
                      "form, but ALPHAR(j), ALPHAI(j), and BETA(j) should be "
                      f"correct for J={info-1},...,N", LinAlgWarning,
                      stacklevel=3)
    elif info == a_n+1:
        raise LinAlgError("Something other than QZ iteration failed")
    elif info == a_n+2:
        raise LinAlgError("After reordering, roundoff changed values of some "
                          "complex eigenvalues so that leading eigenvalues "
                          "in the Generalized Schur form no longer satisfy "
                          "sort=True. This could also be due to scaling.")
    elif info == a_n+3:
        raise LinAlgError("Reordering failed in <s,d,c,z>tgsen")

    return result, gges.typecode


def qz(A, B, output='real', lwork=None, sort=None, overwrite_a=False,
       overwrite_b=False, check_finite=True):
    """
    QZ decomposition for generalized eigenvalues of a pair of matrices.

    The QZ, or generalized Schur, decomposition for a pair of n-by-n
    matrices (A,B) is::

        (A,B) = (Q @ AA @ Z*, Q @ BB @ Z*)

    where AA, BB is in generalized Schur form if BB is upper-triangular
    with non-negative diagonal and AA is upper-triangular, or for real QZ
    decomposition (``output='real'``) block upper triangular with 1x1
    and 2x2 blocks. In this case, the 1x1 blocks correspond to real
    generalized eigenvalues and 2x2 blocks are 'standardized' by making
    the corresponding elements of BB have the form::

        [ a 0 ]
        [ 0 b ]

    and the pair of corresponding 2x2 blocks in AA and BB will have a complex
    conjugate pair of generalized eigenvalues. If (``output='complex'``) or
    A and B are complex matrices, Z' denotes the conjugate-transpose of Z.
    Q and Z are unitary matrices.

    Parameters
    ----------
    A : (N, N) array_like
        2-D array to decompose
    B : (N, N) array_like
        2-D array to decompose
    output : {'real', 'complex'}, optional
        Construct the real or complex QZ decomposition for real matrices.
        Default is 'real'.
    lwork : int, optional
        Work array size. If None or -1, it is automatically computed.
    sort : {None, callable, 'lhp', 'rhp', 'iuc', 'ouc'}, optional
        NOTE: THIS INPUT IS DISABLED FOR NOW. Use ordqz instead.

        Specifies whether the upper eigenvalues should be sorted. A callable
        may be passed that, given a eigenvalue, returns a boolean denoting
        whether the eigenvalue should be sorted to the top-left (True). For
        real matrix pairs, the sort function takes three real arguments
        (alphar, alphai, beta). The eigenvalue
        ``x = (alphar + alphai*1j)/beta``. For complex matrix pairs or
        output='complex', the sort function takes two complex arguments
        (alpha, beta). The eigenvalue ``x = (alpha/beta)``.  Alternatively,
        string parameters may be used:

            - 'lhp'   Left-hand plane (x.real < 0.0)
            - 'rhp'   Right-hand plane (x.real > 0.0)
            - 'iuc'   Inside the unit circle (x*x.conjugate() < 1.0)
            - 'ouc'   Outside the unit circle (x*x.conjugate() > 1.0)

        Defaults to None (no sorting).
    overwrite_a : bool, optional
        Whether to overwrite data in a (may improve performance)
    overwrite_b : bool, optional
        Whether to overwrite data in b (may improve performance)
    check_finite : bool, optional
        If true checks the elements of `A` and `B` are finite numbers. If
        false does no checking and passes matrix through to
        underlying algorithm.

    Returns
    -------
    AA : (N, N) ndarray
        Generalized Schur form of A.
    BB : (N, N) ndarray
        Generalized Schur form of B.
    Q : (N, N) ndarray
        The left Schur vectors.
    Z : (N, N) ndarray
        The right Schur vectors.

    See Also
    --------
    ordqz

    Notes
    -----
    Q is transposed versus the equivalent function in Matlab.

    .. versionadded:: 0.11.0

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import qz

    >>> A = np.array([[1, 2, -1], [5, 5, 5], [2, 4, -8]])
    >>> B = np.array([[1, 1, -3], [3, 1, -1], [5, 6, -2]])

    Compute the decomposition.  The QZ decomposition is not unique, so
    depending on the underlying library that is used, there may be
    differences in the signs of coefficients in the following output.

    >>> AA, BB, Q, Z = qz(A, B)
    >>> AA
    array([[-1.36949157, -4.05459025,  7.44389431],
           [ 0.        ,  7.65653432,  5.13476017],
           [ 0.        , -0.65978437,  2.4186015 ]])  # may vary
    >>> BB
    array([[ 1.71890633, -1.64723705, -0.72696385],
           [ 0.        ,  8.6965692 , -0.        ],
           [ 0.        ,  0.        ,  2.27446233]])  # may vary
    >>> Q
    array([[-0.37048362,  0.1903278 ,  0.90912992],
           [-0.90073232,  0.16534124, -0.40167593],
           [ 0.22676676,  0.96769706, -0.11017818]])  # may vary
    >>> Z
    array([[-0.67660785,  0.63528924, -0.37230283],
           [ 0.70243299,  0.70853819, -0.06753907],
           [ 0.22088393, -0.30721526, -0.92565062]])  # may vary

    Verify the QZ decomposition.  With real output, we only need the
    transpose of ``Z`` in the following expressions.

    >>> Q @ AA @ Z.T  # Should be A
    array([[ 1.,  2., -1.],
           [ 5.,  5.,  5.],
           [ 2.,  4., -8.]])
    >>> Q @ BB @ Z.T  # Should be B
    array([[ 1.,  1., -3.],
           [ 3.,  1., -1.],
           [ 5.,  6., -2.]])

    Repeat the decomposition, but with ``output='complex'``.

    >>> AA, BB, Q, Z = qz(A, B, output='complex')

    For conciseness in the output, we use ``np.set_printoptions()`` to set
    the output precision of NumPy arrays to 3 and display tiny values as 0.

    >>> np.set_printoptions(precision=3, suppress=True)
    >>> AA
    array([[-1.369+0.j   ,  2.248+4.237j,  4.861-5.022j],
           [ 0.   +0.j   ,  7.037+2.922j,  0.794+4.932j],
           [ 0.   +0.j   ,  0.   +0.j   ,  2.655-1.103j]])  # may vary
    >>> BB
    array([[ 1.719+0.j   , -1.115+1.j   , -0.763-0.646j],
           [ 0.   +0.j   ,  7.24 +0.j   , -3.144+3.322j],
           [ 0.   +0.j   ,  0.   +0.j   ,  2.732+0.j   ]])  # may vary
    >>> Q
    array([[ 0.326+0.175j, -0.273-0.029j, -0.886-0.052j],
           [ 0.794+0.426j, -0.093+0.134j,  0.402-0.02j ],
           [-0.2  -0.107j, -0.816+0.482j,  0.151-0.167j]])  # may vary
    >>> Z
    array([[ 0.596+0.32j , -0.31 +0.414j,  0.393-0.347j],
           [-0.619-0.332j, -0.479+0.314j,  0.154-0.393j],
           [-0.195-0.104j,  0.576+0.27j ,  0.715+0.187j]])  # may vary

    With complex arrays, we must use ``Z.conj().T`` in the following
    expressions to verify the decomposition.

    >>> Q @ AA @ Z.conj().T  # Should be A
    array([[ 1.-0.j,  2.-0.j, -1.-0.j],
           [ 5.+0.j,  5.+0.j,  5.-0.j],
           [ 2.+0.j,  4.+0.j, -8.+0.j]])
    >>> Q @ BB @ Z.conj().T  # Should be B
    array([[ 1.+0.j,  1.+0.j, -3.+0.j],
           [ 3.-0.j,  1.-0.j, -1.+0.j],
           [ 5.+0.j,  6.+0.j, -2.+0.j]])

    """
    # output for real
    # AA, BB, sdim, alphar, alphai, beta, vsl, vsr, work, info
    # output for complex
    # AA, BB, sdim, alpha, beta, vsl, vsr, work, info
    result, _ = _qz(A, B, output=output, lwork=lwork, sort=sort,
                    overwrite_a=overwrite_a, overwrite_b=overwrite_b,
                    check_finite=check_finite)
    return result[0], result[1], result[-4], result[-3]


def ordqz(A, B, sort='lhp', output='real', overwrite_a=False,
          overwrite_b=False, check_finite=True):
    """QZ decomposition for a pair of matrices with reordering.

    Parameters
    ----------
    A : (N, N) array_like
        2-D array to decompose
    B : (N, N) array_like
        2-D array to decompose
    sort : {callable, 'lhp', 'rhp', 'iuc', 'ouc'}, optional
        Specifies whether the upper eigenvalues should be sorted. A
        callable may be passed that, given an ordered pair ``(alpha,
        beta)`` representing the eigenvalue ``x = (alpha/beta)``,
        returns a boolean denoting whether the eigenvalue should be
        sorted to the top-left (True). For the real matrix pairs
        ``beta`` is real while ``alpha`` can be complex, and for
        complex matrix pairs both ``alpha`` and ``beta`` can be
        complex. The callable must be able to accept a NumPy
        array. Alternatively, string parameters may be used:

            - 'lhp'   Left-hand plane (x.real < 0.0)
            - 'rhp'   Right-hand plane (x.real > 0.0)
            - 'iuc'   Inside the unit circle (x*x.conjugate() < 1.0)
            - 'ouc'   Outside the unit circle (x*x.conjugate() > 1.0)

        With the predefined sorting functions, an infinite eigenvalue
        (i.e., ``alpha != 0`` and ``beta = 0``) is considered to lie in
        neither the left-hand nor the right-hand plane, but it is
        considered to lie outside the unit circle. For the eigenvalue
        ``(alpha, beta) = (0, 0)``, the predefined sorting functions
        all return `False`.
    output : str {'real','complex'}, optional
        Construct the real or complex QZ decomposition for real matrices.
        Default is 'real'.
    overwrite_a : bool, optional
        If True, the contents of A are overwritten.
    overwrite_b : bool, optional
        If True, the contents of B are overwritten.
    check_finite : bool, optional
        If true checks the elements of `A` and `B` are finite numbers. If
        false does no checking and passes matrix through to
        underlying algorithm.

    Returns
    -------
    AA : (N, N) ndarray
        Generalized Schur form of A.
    BB : (N, N) ndarray
        Generalized Schur form of B.
    alpha : (N,) ndarray
        alpha = alphar + alphai * 1j. See notes.
    beta : (N,) ndarray
        See notes.
    Q : (N, N) ndarray
        The left Schur vectors.
    Z : (N, N) ndarray
        The right Schur vectors.

    See Also
    --------
    qz

    Notes
    -----
    On exit, ``(ALPHAR(j) + ALPHAI(j)*i)/BETA(j), j=1,...,N``, will be the
    generalized eigenvalues.  ``ALPHAR(j) + ALPHAI(j)*i`` and
    ``BETA(j),j=1,...,N`` are the diagonals of the complex Schur form (S,T)
    that would result if the 2-by-2 diagonal blocks of the real generalized
    Schur form of (A,B) were further reduced to triangular form using complex
    unitary transformations. If ALPHAI(j) is zero, then the jth eigenvalue is
    real; if positive, then the ``j``\\ th and ``(j+1)``\\ st eigenvalues are a
    complex conjugate pair, with ``ALPHAI(j+1)`` negative.

    .. versionadded:: 0.17.0

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import ordqz
    >>> A = np.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]])
    >>> B = np.array([[0, 6, 0, 0], [5, 0, 2, 1], [5, 2, 6, 6], [4, 7, 7, 7]])
    >>> AA, BB, alpha, beta, Q, Z = ordqz(A, B, sort='lhp')

    Since we have sorted for left half plane eigenvalues, negatives come first

    >>> (alpha/beta).real < 0
    array([ True,  True, False, False], dtype=bool)

    """
    (AA, BB, _, *ab, Q, Z, _, _), typ = _qz(A, B, output=output, sort=None,
                                            overwrite_a=overwrite_a,
                                            overwrite_b=overwrite_b,
                                            check_finite=check_finite)

    if typ == 's':
        alpha, beta = ab[0] + ab[1]*np.complex64(1j), ab[2]
    elif typ == 'd':
        alpha, beta = ab[0] + ab[1]*1.j, ab[2]
    else:
        alpha, beta = ab

    sfunction = _select_function(sort)
    select = sfunction(alpha, beta)

    tgsen = get_lapack_funcs('tgsen', (AA, BB))
    # the real case needs 4n + 16 lwork
    lwork = 4*AA.shape[0] + 16 if typ in 'sd' else 1
    AAA, BBB, *ab, QQ, ZZ, _, _, _, _, info = tgsen(select, AA, BB, Q, Z,
                                                    ijob=0,
                                                    lwork=lwork, liwork=1)

    # Once more for tgsen output
    if typ == 's':
        alpha, beta = ab[0] + ab[1]*np.complex64(1j), ab[2]
    elif typ == 'd':
        alpha, beta = ab[0] + ab[1]*1.j, ab[2]
    else:
        alpha, beta = ab

    if info < 0:
        raise ValueError(f"Illegal value in argument {-info} of tgsen")
    elif info == 1:
        raise ValueError("Reordering of (A, B) failed because the transformed"
                         " matrix pair (A, B) would be too far from "
                         "generalized Schur form; the problem is very "
                         "ill-conditioned. (A, B) may have been partially "
                         "reordered.")

    return AAA, BBB, alpha, beta, QQ, ZZ
