"""
The :mod:`sklearn.utils.extmath` module includes utilities to perform
optimal mathematical operations in scikit-learn that are not available in SciPy.
"""
# Authors: Gael Varoquaux
#          Alexandre Gramfort
#          Alexandre T. Passos
#          Olivier Grisel
#          Lars Buitinck
#          Stefan van der Walt
#          Kyle Kastner
#          Giorgio Patrini
# License: BSD 3 clause

import warnings
from functools import partial
from numbers import Integral

import numpy as np
from scipy import linalg, sparse

from ..utils import deprecated
from ..utils._param_validation import Interval, StrOptions, validate_params
from . import check_random_state
from ._array_api import _is_numpy_namespace, device, get_namespace
from .sparsefuncs_fast import csr_row_norms
from .validation import check_array


def squared_norm(x):
    """Squared Euclidean or Frobenius norm of x.

    Faster than norm(x) ** 2.

    Parameters
    ----------
    x : array-like
        The input array which could be either be a vector or a 2 dimensional array.

    Returns
    -------
    float
        The Euclidean norm when x is a vector, the Frobenius norm when x
        is a matrix (2-d array).
    """
    x = np.ravel(x, order="K")
    if np.issubdtype(x.dtype, np.integer):
        warnings.warn(
            (
                "Array type is integer, np.dot may overflow. "
                "Data should be float type to avoid this issue"
            ),
            UserWarning,
        )
    return np.dot(x, x)


def row_norms(X, squared=False):
    """Row-wise (squared) Euclidean norm of X.

    Equivalent to np.sqrt((X * X).sum(axis=1)), but also supports sparse
    matrices and does not create an X.shape-sized temporary.

    Performs no input validation.

    Parameters
    ----------
    X : array-like
        The input array.
    squared : bool, default=False
        If True, return squared norms.

    Returns
    -------
    array-like
        The row-wise (squared) Euclidean norm of X.
    """
    if sparse.issparse(X):
        X = X.tocsr()
        norms = csr_row_norms(X)
        if not squared:
            norms = np.sqrt(norms)
    else:
        xp, _ = get_namespace(X)
        if _is_numpy_namespace(xp):
            X = np.asarray(X)
            norms = np.einsum("ij,ij->i", X, X)
            norms = xp.asarray(norms)
        else:
            norms = xp.sum(xp.multiply(X, X), axis=1)
        if not squared:
            norms = xp.sqrt(norms)
    return norms


def fast_logdet(A):
    """Compute logarithm of determinant of a square matrix.

    The (natural) logarithm of the determinant of a square matrix
    is returned if det(A) is non-negative and well defined.
    If the determinant is zero or negative returns -Inf.

    Equivalent to : np.log(np.det(A)) but more robust.

    Parameters
    ----------
    A : array_like of shape (n, n)
        The square matrix.

    Returns
    -------
    logdet : float
        When det(A) is strictly positive, log(det(A)) is returned.
        When det(A) is non-positive or not defined, then -inf is returned.

    See Also
    --------
    numpy.linalg.slogdet : Compute the sign and (natural) logarithm of the determinant
        of an array.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.utils.extmath import fast_logdet
    >>> a = np.array([[5, 1], [2, 8]])
    >>> fast_logdet(a)
    3.6375861597263857
    """
    xp, _ = get_namespace(A)
    sign, ld = xp.linalg.slogdet(A)
    if not sign > 0:
        return -xp.inf
    return ld


def density(w):
    """Compute density of a sparse vector.

    Parameters
    ----------
    w : array-like
        The sparse vector.

    Returns
    -------
    float
        The density of w, between 0 and 1.
    """
    if hasattr(w, "toarray"):
        d = float(w.nnz) / (w.shape[0] * w.shape[1])
    else:
        d = 0 if w is None else float((w != 0).sum()) / w.size
    return d


def safe_sparse_dot(a, b, *, dense_output=False):
    """Dot product that handle the sparse matrix case correctly.

    Parameters
    ----------
    a : {ndarray, sparse matrix}
    b : {ndarray, sparse matrix}
    dense_output : bool, default=False
        When False, ``a`` and ``b`` both being sparse will yield sparse output.
        When True, output will always be a dense array.

    Returns
    -------
    dot_product : {ndarray, sparse matrix}
        Sparse if ``a`` and ``b`` are sparse and ``dense_output=False``.
    """
    if a.ndim > 2 or b.ndim > 2:
        if sparse.issparse(a):
            # sparse is always 2D. Implies b is 3D+
            # [i, j] @ [k, ..., l, m, n] -> [i, k, ..., l, n]
            b_ = np.rollaxis(b, -2)
            b_2d = b_.reshape((b.shape[-2], -1))
            ret = a @ b_2d
            ret = ret.reshape(a.shape[0], *b_.shape[1:])
        elif sparse.issparse(b):
            # sparse is always 2D. Implies a is 3D+
            # [k, ..., l, m] @ [i, j] -> [k, ..., l, j]
            a_2d = a.reshape(-1, a.shape[-1])
            ret = a_2d @ b
            ret = ret.reshape(*a.shape[:-1], b.shape[1])
        else:
            ret = np.dot(a, b)
    else:
        ret = a @ b

    if (
        sparse.issparse(a)
        and sparse.issparse(b)
        and dense_output
        and hasattr(ret, "toarray")
    ):
        return ret.toarray()
    return ret


def randomized_range_finder(
    A, *, size, n_iter, power_iteration_normalizer="auto", random_state=None
):
    """Compute an orthonormal matrix whose range approximates the range of A.

    Parameters
    ----------
    A : 2D array
        The input data matrix.

    size : int
        Size of the return array.

    n_iter : int
        Number of power iterations used to stabilize the result.

    power_iteration_normalizer : {'auto', 'QR', 'LU', 'none'}, default='auto'
        Whether the power iterations are normalized with step-by-step
        QR factorization (the slowest but most accurate), 'none'
        (the fastest but numerically unstable when `n_iter` is large, e.g.
        typically 5 or larger), or 'LU' factorization (numerically stable
        but can lose slightly in accuracy). The 'auto' mode applies no
        normalization if `n_iter` <= 2 and switches to LU otherwise.

        .. versionadded:: 0.18

    random_state : int, RandomState instance or None, default=None
        The seed of the pseudo random number generator to use when shuffling
        the data, i.e. getting the random vectors to initialize the algorithm.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    Q : ndarray
        A (size x size) projection matrix, the range of which
        approximates well the range of the input matrix A.

    Notes
    -----

    Follows Algorithm 4.3 of
    :arxiv:`"Finding structure with randomness:
    Stochastic algorithms for constructing approximate matrix decompositions"
    <0909.4061>`
    Halko, et al. (2009)

    An implementation of a randomized algorithm for principal component
    analysis
    A. Szlam et al. 2014
    """
    xp, is_array_api_compliant = get_namespace(A)
    random_state = check_random_state(random_state)

    # Generating normal random vectors with shape: (A.shape[1], size)
    # XXX: generate random number directly from xp if it's possible
    # one day.
    Q = xp.asarray(random_state.normal(size=(A.shape[1], size)))
    if hasattr(A, "dtype") and xp.isdtype(A.dtype, kind="real floating"):
        # Use float32 computation and components if A has a float32 dtype.
        Q = xp.astype(Q, A.dtype, copy=False)

    # Move Q to device if needed only after converting to float32 if needed to
    # avoid allocating unnecessary memory on the device.

    # Note: we cannot combine the astype and to_device operations in one go
    # using xp.asarray(..., dtype=dtype, device=device) because downcasting
    # from float64 to float32 in asarray might not always be accepted as only
    # casts following type promotion rules are guarateed to work.
    # https://github.com/data-apis/array-api/issues/647
    if is_array_api_compliant:
        Q = xp.asarray(Q, device=device(A))

    # Deal with "auto" mode
    if power_iteration_normalizer == "auto":
        if n_iter <= 2:
            power_iteration_normalizer = "none"
        elif is_array_api_compliant:
            # XXX: https://github.com/data-apis/array-api/issues/627
            warnings.warn(
                "Array API does not support LU factorization, falling back to QR"
                " instead. Set `power_iteration_normalizer='QR'` explicitly to silence"
                " this warning."
            )
            power_iteration_normalizer = "QR"
        else:
            power_iteration_normalizer = "LU"
    elif power_iteration_normalizer == "LU" and is_array_api_compliant:
        raise ValueError(
            "Array API does not support LU factorization. Set "
            "`power_iteration_normalizer='QR'` instead."
        )

    if is_array_api_compliant:
        qr_normalizer = partial(xp.linalg.qr, mode="reduced")
    else:
        # Use scipy.linalg instead of numpy.linalg when not explicitly
        # using the Array API.
        qr_normalizer = partial(linalg.qr, mode="economic")

    if power_iteration_normalizer == "QR":
        normalizer = qr_normalizer
    elif power_iteration_normalizer == "LU":
        normalizer = partial(linalg.lu, permute_l=True)
    else:
        normalizer = lambda x: (x, None)

    # Perform power iterations with Q to further 'imprint' the top
    # singular vectors of A in Q
    for _ in range(n_iter):
        Q, _ = normalizer(A @ Q)
        Q, _ = normalizer(A.T @ Q)

    # Sample the range of A using by linear projection of Q
    # Extract an orthonormal basis
    Q, _ = qr_normalizer(A @ Q)

    return Q


@validate_params(
    {
        "M": [np.ndarray, "sparse matrix"],
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "n_oversamples": [Interval(Integral, 0, None, closed="left")],
        "n_iter": [Interval(Integral, 0, None, closed="left"), StrOptions({"auto"})],
        "power_iteration_normalizer": [StrOptions({"auto", "QR", "LU", "none"})],
        "transpose": ["boolean", StrOptions({"auto"})],
        "flip_sign": ["boolean"],
        "random_state": ["random_state"],
        "svd_lapack_driver": [StrOptions({"gesdd", "gesvd"})],
    },
    prefer_skip_nested_validation=True,
)
def randomized_svd(
    M,
    n_components,
    *,
    n_oversamples=10,
    n_iter="auto",
    power_iteration_normalizer="auto",
    transpose="auto",
    flip_sign=True,
    random_state=None,
    svd_lapack_driver="gesdd",
):
    """Compute a truncated randomized SVD.

    This method solves the fixed-rank approximation problem described in [1]_
    (problem (1.5), p5).

    Parameters
    ----------
    M : {ndarray, sparse matrix}
        Matrix to decompose.

    n_components : int
        Number of singular values and vectors to extract.

    n_oversamples : int, default=10
        Additional number of random vectors to sample the range of `M` so as
        to ensure proper conditioning. The total number of random vectors
        used to find the range of `M` is `n_components + n_oversamples`. Smaller
        number can improve speed but can negatively impact the quality of
        approximation of singular vectors and singular values. Users might wish
        to increase this parameter up to `2*k - n_components` where k is the
        effective rank, for large matrices, noisy problems, matrices with
        slowly decaying spectrums, or to increase precision accuracy. See [1]_
        (pages 5, 23 and 26).

    n_iter : int or 'auto', default='auto'
        Number of power iterations. It can be used to deal with very noisy
        problems. When 'auto', it is set to 4, unless `n_components` is small
        (< .1 * min(X.shape)) in which case `n_iter` is set to 7.
        This improves precision with few components. Note that in general
        users should rather increase `n_oversamples` before increasing `n_iter`
        as the principle of the randomized method is to avoid usage of these
        more costly power iterations steps. When `n_components` is equal
        or greater to the effective matrix rank and the spectrum does not
        present a slow decay, `n_iter=0` or `1` should even work fine in theory
        (see [1]_ page 9).

        .. versionchanged:: 0.18

    power_iteration_normalizer : {'auto', 'QR', 'LU', 'none'}, default='auto'
        Whether the power iterations are normalized with step-by-step
        QR factorization (the slowest but most accurate), 'none'
        (the fastest but numerically unstable when `n_iter` is large, e.g.
        typically 5 or larger), or 'LU' factorization (numerically stable
        but can lose slightly in accuracy). The 'auto' mode applies no
        normalization if `n_iter` <= 2 and switches to LU otherwise.

        .. versionadded:: 0.18

    transpose : bool or 'auto', default='auto'
        Whether the algorithm should be applied to M.T instead of M. The
        result should approximately be the same. The 'auto' mode will
        trigger the transposition if M.shape[1] > M.shape[0] since this
        implementation of randomized SVD tend to be a little faster in that
        case.

        .. versionchanged:: 0.18

    flip_sign : bool, default=True
        The output of a singular value decomposition is only unique up to a
        permutation of the signs of the singular vectors. If `flip_sign` is
        set to `True`, the sign ambiguity is resolved by making the largest
        loadings for each component in the left singular vectors positive.

    random_state : int, RandomState instance or None, default='warn'
        The seed of the pseudo random number generator to use when
        shuffling the data, i.e. getting the random vectors to initialize
        the algorithm. Pass an int for reproducible results across multiple
        function calls. See :term:`Glossary <random_state>`.

        .. versionchanged:: 1.2
            The default value changed from 0 to None.

    svd_lapack_driver : {"gesdd", "gesvd"}, default="gesdd"
        Whether to use the more efficient divide-and-conquer approach
        (`"gesdd"`) or more general rectangular approach (`"gesvd"`) to compute
        the SVD of the matrix B, which is the projection of M into a low
        dimensional subspace, as described in [1]_.

        .. versionadded:: 1.2

    Returns
    -------
    u : ndarray of shape (n_samples, n_components)
        Unitary matrix having left singular vectors with signs flipped as columns.
    s : ndarray of shape (n_components,)
        The singular values, sorted in non-increasing order.
    vh : ndarray of shape (n_components, n_features)
        Unitary matrix having right singular vectors with signs flipped as rows.

    Notes
    -----
    This algorithm finds a (usually very good) approximate truncated
    singular value decomposition using randomization to speed up the
    computations. It is particularly fast on large matrices on which
    you wish to extract only a small number of components. In order to
    obtain further speed up, `n_iter` can be set <=2 (at the cost of
    loss of precision). To increase the precision it is recommended to
    increase `n_oversamples`, up to `2*k-n_components` where k is the
    effective rank. Usually, `n_components` is chosen to be greater than k
    so increasing `n_oversamples` up to `n_components` should be enough.

    References
    ----------
    .. [1] :arxiv:`"Finding structure with randomness:
      Stochastic algorithms for constructing approximate matrix decompositions"
      <0909.4061>`
      Halko, et al. (2009)

    .. [2] A randomized algorithm for the decomposition of matrices
      Per-Gunnar Martinsson, Vladimir Rokhlin and Mark Tygert

    .. [3] An implementation of a randomized algorithm for principal component
      analysis A. Szlam et al. 2014

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.utils.extmath import randomized_svd
    >>> a = np.array([[1, 2, 3, 5],
    ...               [3, 4, 5, 6],
    ...               [7, 8, 9, 10]])
    >>> U, s, Vh = randomized_svd(a, n_components=2, random_state=0)
    >>> U.shape, s.shape, Vh.shape
    ((3, 2), (2,), (2, 4))
    """
    if sparse.issparse(M) and M.format in ("lil", "dok"):
        warnings.warn(
            "Calculating SVD of a {} is expensive. "
            "csr_matrix is more efficient.".format(type(M).__name__),
            sparse.SparseEfficiencyWarning,
        )

    random_state = check_random_state(random_state)
    n_random = n_components + n_oversamples
    n_samples, n_features = M.shape

    if n_iter == "auto":
        # Checks if the number of iterations is explicitly specified
        # Adjust n_iter. 7 was found a good compromise for PCA. See #5299
        n_iter = 7 if n_components < 0.1 * min(M.shape) else 4

    if transpose == "auto":
        transpose = n_samples < n_features
    if transpose:
        # this implementation is a bit faster with smaller shape[1]
        M = M.T

    Q = randomized_range_finder(
        M,
        size=n_random,
        n_iter=n_iter,
        power_iteration_normalizer=power_iteration_normalizer,
        random_state=random_state,
    )

    # project M to the (k + p) dimensional space using the basis vectors
    B = Q.T @ M

    # compute the SVD on the thin matrix: (k + p) wide
    xp, is_array_api_compliant = get_namespace(B)
    if is_array_api_compliant:
        Uhat, s, Vt = xp.linalg.svd(B, full_matrices=False)
    else:
        # When when array_api_dispatch is disabled, rely on scipy.linalg
        # instead of numpy.linalg to avoid introducing a behavior change w.r.t.
        # previous versions of scikit-learn.
        Uhat, s, Vt = linalg.svd(
            B, full_matrices=False, lapack_driver=svd_lapack_driver
        )
    del B
    U = Q @ Uhat

    if flip_sign:
        if not transpose:
            U, Vt = svd_flip(U, Vt)
        else:
            # In case of transpose u_based_decision=false
            # to actually flip based on u and not v.
            U, Vt = svd_flip(U, Vt, u_based_decision=False)

    if transpose:
        # transpose back the results according to the input convention
        return Vt[:n_components, :].T, s[:n_components], U[:, :n_components].T
    else:
        return U[:, :n_components], s[:n_components], Vt[:n_components, :]


def _randomized_eigsh(
    M,
    n_components,
    *,
    n_oversamples=10,
    n_iter="auto",
    power_iteration_normalizer="auto",
    selection="module",
    random_state=None,
):
    """Computes a truncated eigendecomposition using randomized methods

    This method solves the fixed-rank approximation problem described in the
    Halko et al paper.

    The choice of which components to select can be tuned with the `selection`
    parameter.

    .. versionadded:: 0.24

    Parameters
    ----------
    M : ndarray or sparse matrix
        Matrix to decompose, it should be real symmetric square or complex
        hermitian

    n_components : int
        Number of eigenvalues and vectors to extract.

    n_oversamples : int, default=10
        Additional number of random vectors to sample the range of M so as
        to ensure proper conditioning. The total number of random vectors
        used to find the range of M is n_components + n_oversamples. Smaller
        number can improve speed but can negatively impact the quality of
        approximation of eigenvectors and eigenvalues. Users might wish
        to increase this parameter up to `2*k - n_components` where k is the
        effective rank, for large matrices, noisy problems, matrices with
        slowly decaying spectrums, or to increase precision accuracy. See Halko
        et al (pages 5, 23 and 26).

    n_iter : int or 'auto', default='auto'
        Number of power iterations. It can be used to deal with very noisy
        problems. When 'auto', it is set to 4, unless `n_components` is small
        (< .1 * min(X.shape)) in which case `n_iter` is set to 7.
        This improves precision with few components. Note that in general
        users should rather increase `n_oversamples` before increasing `n_iter`
        as the principle of the randomized method is to avoid usage of these
        more costly power iterations steps. When `n_components` is equal
        or greater to the effective matrix rank and the spectrum does not
        present a slow decay, `n_iter=0` or `1` should even work fine in theory
        (see Halko et al paper, page 9).

    power_iteration_normalizer : {'auto', 'QR', 'LU', 'none'}, default='auto'
        Whether the power iterations are normalized with step-by-step
        QR factorization (the slowest but most accurate), 'none'
        (the fastest but numerically unstable when `n_iter` is large, e.g.
        typically 5 or larger), or 'LU' factorization (numerically stable
        but can lose slightly in accuracy). The 'auto' mode applies no
        normalization if `n_iter` <= 2 and switches to LU otherwise.

    selection : {'value', 'module'}, default='module'
        Strategy used to select the n components. When `selection` is `'value'`
        (not yet implemented, will become the default when implemented), the
        components corresponding to the n largest eigenvalues are returned.
        When `selection` is `'module'`, the components corresponding to the n
        eigenvalues with largest modules are returned.

    random_state : int, RandomState instance, default=None
        The seed of the pseudo random number generator to use when shuffling
        the data, i.e. getting the random vectors to initialize the algorithm.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Notes
    -----
    This algorithm finds a (usually very good) approximate truncated
    eigendecomposition using randomized methods to speed up the computations.

    This method is particularly fast on large matrices on which
    you wish to extract only a small number of components. In order to
    obtain further speed up, `n_iter` can be set <=2 (at the cost of
    loss of precision). To increase the precision it is recommended to
    increase `n_oversamples`, up to `2*k-n_components` where k is the
    effective rank. Usually, `n_components` is chosen to be greater than k
    so increasing `n_oversamples` up to `n_components` should be enough.

    Strategy 'value': not implemented yet.
    Algorithms 5.3, 5.4 and 5.5 in the Halko et al paper should provide good
    candidates for a future implementation.

    Strategy 'module':
    The principle is that for diagonalizable matrices, the singular values and
    eigenvalues are related: if t is an eigenvalue of A, then :math:`|t|` is a
    singular value of A. This method relies on a randomized SVD to find the n
    singular components corresponding to the n singular values with largest
    modules, and then uses the signs of the singular vectors to find the true
    sign of t: if the sign of left and right singular vectors are different
    then the corresponding eigenvalue is negative.

    Returns
    -------
    eigvals : 1D array of shape (n_components,) containing the `n_components`
        eigenvalues selected (see ``selection`` parameter).
    eigvecs : 2D array of shape (M.shape[0], n_components) containing the
        `n_components` eigenvectors corresponding to the `eigvals`, in the
        corresponding order. Note that this follows the `scipy.linalg.eigh`
        convention.

    See Also
    --------
    :func:`randomized_svd`

    References
    ----------
    * :arxiv:`"Finding structure with randomness:
      Stochastic algorithms for constructing approximate matrix decompositions"
      (Algorithm 4.3 for strategy 'module') <0909.4061>`
      Halko, et al. (2009)
    """
    if selection == "value":  # pragma: no cover
        # to do : an algorithm can be found in the Halko et al reference
        raise NotImplementedError()

    elif selection == "module":
        # Note: no need for deterministic U and Vt (flip_sign=True),
        # as we only use the dot product UVt afterwards
        U, S, Vt = randomized_svd(
            M,
            n_components=n_components,
            n_oversamples=n_oversamples,
            n_iter=n_iter,
            power_iteration_normalizer=power_iteration_normalizer,
            flip_sign=False,
            random_state=random_state,
        )

        eigvecs = U[:, :n_components]
        eigvals = S[:n_components]

        # Conversion of Singular values into Eigenvalues:
        # For any eigenvalue t, the corresponding singular value is |t|.
        # So if there is a negative eigenvalue t, the corresponding singular
        # value will be -t, and the left (U) and right (V) singular vectors
        # will have opposite signs.
        # Fastest way: see <https://stackoverflow.com/a/61974002/7262247>
        diag_VtU = np.einsum("ji,ij->j", Vt[:n_components, :], U[:, :n_components])
        signs = np.sign(diag_VtU)
        eigvals = eigvals * signs

    else:  # pragma: no cover
        raise ValueError("Invalid `selection`: %r" % selection)

    return eigvals, eigvecs


def weighted_mode(a, w, *, axis=0):
    """Return an array of the weighted modal (most common) value in the passed array.

    If there is more than one such value, only the first is returned.
    The bin-count for the modal bins is also returned.

    This is an extension of the algorithm in scipy.stats.mode.

    Parameters
    ----------
    a : array-like of shape (n_samples,)
        Array of which values to find mode(s).
    w : array-like of shape (n_samples,)
        Array of weights for each value.
    axis : int, default=0
        Axis along which to operate. Default is 0, i.e. the first axis.

    Returns
    -------
    vals : ndarray
        Array of modal values.
    score : ndarray
        Array of weighted counts for each mode.

    See Also
    --------
    scipy.stats.mode: Calculates the Modal (most common) value of array elements
        along specified axis.

    Examples
    --------
    >>> from sklearn.utils.extmath import weighted_mode
    >>> x = [4, 1, 4, 2, 4, 2]
    >>> weights = [1, 1, 1, 1, 1, 1]
    >>> weighted_mode(x, weights)
    (array([4.]), array([3.]))

    The value 4 appears three times: with uniform weights, the result is
    simply the mode of the distribution.

    >>> weights = [1, 3, 0.5, 1.5, 1, 2]  # deweight the 4's
    >>> weighted_mode(x, weights)
    (array([2.]), array([3.5]))

    The value 2 has the highest score: it appears twice with weights of
    1.5 and 2: the sum of these is 3.5.
    """
    if axis is None:
        a = np.ravel(a)
        w = np.ravel(w)
        axis = 0
    else:
        a = np.asarray(a)
        w = np.asarray(w)

    if a.shape != w.shape:
        w = np.full(a.shape, w, dtype=w.dtype)

    scores = np.unique(np.ravel(a))  # get ALL unique values
    testshape = list(a.shape)
    testshape[axis] = 1
    oldmostfreq = np.zeros(testshape)
    oldcounts = np.zeros(testshape)
    for score in scores:
        template = np.zeros(a.shape)
        ind = a == score
        template[ind] = w[ind]
        counts = np.expand_dims(np.sum(template, axis), axis)
        mostfrequent = np.where(counts > oldcounts, score, oldmostfreq)
        oldcounts = np.maximum(counts, oldcounts)
        oldmostfreq = mostfrequent
    return mostfrequent, oldcounts


def cartesian(arrays, out=None):
    """Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray of shape (M, len(arrays)), default=None
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray of shape (M, len(arrays))
        Array containing the cartesian products formed of input arrays.
        If not provided, the `dtype` of the output array is set to the most
        permissive `dtype` of the input arrays, according to NumPy type
        promotion.

        .. versionadded:: 1.2
           Add support for arrays of different types.

    Notes
    -----
    This function may not be used on more than 32 arrays
    because the underlying numpy functions do not support it.

    Examples
    --------
    >>> from sklearn.utils.extmath import cartesian
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])
    """
    arrays = [np.asarray(x) for x in arrays]
    shape = (len(x) for x in arrays)

    ix = np.indices(shape)
    ix = ix.reshape(len(arrays), -1).T

    if out is None:
        dtype = np.result_type(*arrays)  # find the most permissive dtype
        out = np.empty_like(ix, dtype=dtype)

    for n, arr in enumerate(arrays):
        out[:, n] = arrays[n][ix[:, n]]

    return out


def svd_flip(u, v, u_based_decision=True):
    """Sign correction to ensure deterministic output from SVD.

    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.

    If u_based_decision is False, then the same sign correction is applied to
    so that the rows in v that are largest in absolute value are always
    positive.

    Parameters
    ----------
    u : ndarray
        Parameters u and v are the output of `linalg.svd` or
        :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
        dimensions so one can compute `np.dot(u * s, v)`.

    v : ndarray
        Parameters u and v are the output of `linalg.svd` or
        :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
        dimensions so one can compute `np.dot(u * s, v)`. The input v should
        really be called vt to be consistent with scipy's output.

    u_based_decision : bool, default=True
        If True, use the columns of u as the basis for sign flipping.
        Otherwise, use the rows of v. The choice of which variable to base the
        decision on is generally algorithm dependent.

    Returns
    -------
    u_adjusted : ndarray
        Array u with adjusted columns and the same dimensions as u.

    v_adjusted : ndarray
        Array v with adjusted rows and the same dimensions as v.
    """
    xp, _ = get_namespace(u, v)
    device = getattr(u, "device", None)

    if u_based_decision:
        # columns of u, rows of v, or equivalently rows of u.T and v
        max_abs_u_cols = xp.argmax(xp.abs(u.T), axis=1)
        shift = xp.arange(u.T.shape[0], device=device)
        indices = max_abs_u_cols + shift * u.T.shape[1]
        signs = xp.sign(xp.take(xp.reshape(u.T, (-1,)), indices, axis=0))
        u *= signs[np.newaxis, :]
        v *= signs[:, np.newaxis]
    else:
        # rows of v, columns of u
        max_abs_v_rows = xp.argmax(xp.abs(v), axis=1)
        shift = xp.arange(v.shape[0], device=device)
        indices = max_abs_v_rows + shift * v.shape[1]
        signs = xp.sign(xp.take(xp.reshape(v, (-1,)), indices))
        u *= signs[np.newaxis, :]
        v *= signs[:, np.newaxis]
    return u, v


# TODO(1.6): remove
@deprecated(  # type: ignore
    "The function `log_logistic` is deprecated and will be removed in 1.6. "
    "Use `-np.logaddexp(0, -x)` instead."
)
def log_logistic(X, out=None):
    """Compute the log of the logistic function, ``log(1 / (1 + e ** -x))``.

    This implementation is numerically stable and uses `-np.logaddexp(0, -x)`.

    For the ordinary logistic function, use ``scipy.special.expit``.

    Parameters
    ----------
    X : array-like of shape (M, N) or (M,)
        Argument to the logistic function.

    out : array-like of shape (M, N) or (M,), default=None
        Preallocated output array.

    Returns
    -------
    out : ndarray of shape (M, N) or (M,)
        Log of the logistic function evaluated at every point in x.

    Notes
    -----
    See the blog post describing this implementation:
    http://fa.bianp.net/blog/2013/numerical-optimizers-for-logistic-regression/
    """
    X = check_array(X, dtype=np.float64, ensure_2d=False)

    if out is None:
        out = np.empty_like(X)

    np.logaddexp(0, -X, out=out)
    out *= -1
    return out


def softmax(X, copy=True):
    """
    Calculate the softmax function.

    The softmax function is calculated by
    np.exp(X) / np.sum(np.exp(X), axis=1)

    This will cause overflow when large values are exponentiated.
    Hence the largest value in each row is subtracted from each data
    point to prevent this.

    Parameters
    ----------
    X : array-like of float of shape (M, N)
        Argument to the logistic function.

    copy : bool, default=True
        Copy X or not.

    Returns
    -------
    out : ndarray of shape (M, N)
        Softmax function evaluated at every point in x.
    """
    xp, is_array_api_compliant = get_namespace(X)
    if copy:
        X = xp.asarray(X, copy=True)
    max_prob = xp.reshape(xp.max(X, axis=1), (-1, 1))
    X -= max_prob

    if _is_numpy_namespace(xp):
        # optimization for NumPy arrays
        np.exp(X, out=np.asarray(X))
    else:
        # array_api does not have `out=`
        X = xp.exp(X)

    sum_prob = xp.reshape(xp.sum(X, axis=1), (-1, 1))
    X /= sum_prob
    return X


def make_nonnegative(X, min_value=0):
    """Ensure `X.min()` >= `min_value`.

    Parameters
    ----------
    X : array-like
        The matrix to make non-negative.
    min_value : float, default=0
        The threshold value.

    Returns
    -------
    array-like
        The thresholded array.

    Raises
    ------
    ValueError
        When X is sparse.
    """
    min_ = X.min()
    if min_ < min_value:
        if sparse.issparse(X):
            raise ValueError(
                "Cannot make the data matrix"
                " nonnegative because it is sparse."
                " Adding a value to every entry would"
                " make it no longer sparse."
            )
        X = X + (min_value - min_)
    return X


# Use at least float64 for the accumulating functions to avoid precision issue
# see https://github.com/numpy/numpy/issues/9393. The float64 is also retained
# as it is in case the float overflows
def _safe_accumulator_op(op, x, *args, **kwargs):
    """
    This function provides numpy accumulator functions with a float64 dtype
    when used on a floating point input. This prevents accumulator overflow on
    smaller floating point dtypes.

    Parameters
    ----------
    op : function
        A numpy accumulator function such as np.mean or np.sum.
    x : ndarray
        A numpy array to apply the accumulator function.
    *args : positional arguments
        Positional arguments passed to the accumulator function after the
        input x.
    **kwargs : keyword arguments
        Keyword arguments passed to the accumulator function.

    Returns
    -------
    result
        The output of the accumulator function passed to this function.
    """
    if np.issubdtype(x.dtype, np.floating) and x.dtype.itemsize < 8:
        result = op(x, *args, **kwargs, dtype=np.float64)
    else:
        result = op(x, *args, **kwargs)
    return result


def _incremental_mean_and_var(
    X, last_mean, last_variance, last_sample_count, sample_weight=None
):
    """Calculate mean update and a Youngs and Cramer variance update.

    If sample_weight is given, the weighted mean and variance is computed.

    Update a given mean and (possibly) variance according to new data given
    in X. last_mean is always required to compute the new mean.
    If last_variance is None, no variance is computed and None return for
    updated_variance.

    From the paper "Algorithms for computing the sample variance: analysis and
    recommendations", by Chan, Golub, and LeVeque.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data to use for variance update.

    last_mean : array-like of shape (n_features,)

    last_variance : array-like of shape (n_features,)

    last_sample_count : array-like of shape (n_features,)
        The number of samples encountered until now if sample_weight is None.
        If sample_weight is not None, this is the sum of sample_weight
        encountered.

    sample_weight : array-like of shape (n_samples,) or None
        Sample weights. If None, compute the unweighted mean/variance.

    Returns
    -------
    updated_mean : ndarray of shape (n_features,)

    updated_variance : ndarray of shape (n_features,)
        None if last_variance was None.

    updated_sample_count : ndarray of shape (n_features,)

    Notes
    -----
    NaNs are ignored during the algorithm.

    References
    ----------
    T. Chan, G. Golub, R. LeVeque. Algorithms for computing the sample
        variance: recommendations, The American Statistician, Vol. 37, No. 3,
        pp. 242-247

    Also, see the sparse implementation of this in
    `utils.sparsefuncs.incr_mean_variance_axis` and
    `utils.sparsefuncs_fast.incr_mean_variance_axis0`
    """
    # old = stats until now
    # new = the current increment
    # updated = the aggregated stats
    last_sum = last_mean * last_sample_count
    X_nan_mask = np.isnan(X)
    if np.any(X_nan_mask):
        sum_op = np.nansum
    else:
        sum_op = np.sum
    if sample_weight is not None:
        # equivalent to np.nansum(X * sample_weight, axis=0)
        # safer because np.float64(X*W) != np.float64(X)*np.float64(W)
        new_sum = _safe_accumulator_op(
            np.matmul, sample_weight, np.where(X_nan_mask, 0, X)
        )
        new_sample_count = _safe_accumulator_op(
            np.sum, sample_weight[:, None] * (~X_nan_mask), axis=0
        )
    else:
        new_sum = _safe_accumulator_op(sum_op, X, axis=0)
        n_samples = X.shape[0]
        new_sample_count = n_samples - np.sum(X_nan_mask, axis=0)

    updated_sample_count = last_sample_count + new_sample_count

    updated_mean = (last_sum + new_sum) / updated_sample_count

    if last_variance is None:
        updated_variance = None
    else:
        T = new_sum / new_sample_count
        temp = X - T
        if sample_weight is not None:
            # equivalent to np.nansum((X-T)**2 * sample_weight, axis=0)
            # safer because np.float64(X*W) != np.float64(X)*np.float64(W)
            correction = _safe_accumulator_op(
                np.matmul, sample_weight, np.where(X_nan_mask, 0, temp)
            )
            temp **= 2
            new_unnormalized_variance = _safe_accumulator_op(
                np.matmul, sample_weight, np.where(X_nan_mask, 0, temp)
            )
        else:
            correction = _safe_accumulator_op(sum_op, temp, axis=0)
            temp **= 2
            new_unnormalized_variance = _safe_accumulator_op(sum_op, temp, axis=0)

        # correction term of the corrected 2 pass algorithm.
        # See "Algorithms for computing the sample variance: analysis
        # and recommendations", by Chan, Golub, and LeVeque.
        new_unnormalized_variance -= correction**2 / new_sample_count

        last_unnormalized_variance = last_variance * last_sample_count

        with np.errstate(divide="ignore", invalid="ignore"):
            last_over_new_count = last_sample_count / new_sample_count
            updated_unnormalized_variance = (
                last_unnormalized_variance
                + new_unnormalized_variance
                + last_over_new_count
                / updated_sample_count
                * (last_sum / last_over_new_count - new_sum) ** 2
            )

        zeros = last_sample_count == 0
        updated_unnormalized_variance[zeros] = new_unnormalized_variance[zeros]
        updated_variance = updated_unnormalized_variance / updated_sample_count

    return updated_mean, updated_variance, updated_sample_count


def _deterministic_vector_sign_flip(u):
    """Modify the sign of vectors for reproducibility.

    Flips the sign of elements of all the vectors (rows of u) such that
    the absolute maximum element of each vector is positive.

    Parameters
    ----------
    u : ndarray
        Array with vectors as its rows.

    Returns
    -------
    u_flipped : ndarray with same shape as u
        Array with the sign flipped vectors as its rows.
    """
    max_abs_rows = np.argmax(np.abs(u), axis=1)
    signs = np.sign(u[range(u.shape[0]), max_abs_rows])
    u *= signs[:, np.newaxis]
    return u


def stable_cumsum(arr, axis=None, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum.

    Warns if the final cumulative sum does not match the sum (up to the chosen
    tolerance).

    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat.
    axis : int, default=None
        Axis along which the cumulative sum is computed.
        The default (None) is to compute the cumsum over the flattened array.
    rtol : float, default=1e-05
        Relative tolerance, see ``np.allclose``.
    atol : float, default=1e-08
        Absolute tolerance, see ``np.allclose``.

    Returns
    -------
    out : ndarray
        Array with the cumulative sums along the chosen axis.
    """
    out = np.cumsum(arr, axis=axis, dtype=np.float64)
    expected = np.sum(arr, axis=axis, dtype=np.float64)
    if not np.allclose(
        out.take(-1, axis=axis), expected, rtol=rtol, atol=atol, equal_nan=True
    ):
        warnings.warn(
            (
                "cumsum was found to be unstable: "
                "its last element does not correspond to sum"
            ),
            RuntimeWarning,
        )
    return out


def _nanaverage(a, weights=None):
    """Compute the weighted average, ignoring NaNs.

    Parameters
    ----------
    a : ndarray
        Array containing data to be averaged.
    weights : array-like, default=None
        An array of weights associated with the values in a. Each value in a
        contributes to the average according to its associated weight. The
        weights array can either be 1-D of the same shape as a. If `weights=None`,
        then all data in a are assumed to have a weight equal to one.

    Returns
    -------
    weighted_average : float
        The weighted average.

    Notes
    -----
    This wrapper to combine :func:`numpy.average` and :func:`numpy.nanmean`, so
    that :func:`np.nan` values are ignored from the average and weights can
    be passed. Note that when possible, we delegate to the prime methods.
    """

    if len(a) == 0:
        return np.nan

    mask = np.isnan(a)
    if mask.all():
        return np.nan

    if weights is None:
        return np.nanmean(a)

    weights = np.array(weights, copy=False)
    a, weights = a[~mask], weights[~mask]
    try:
        return np.average(a, weights=weights)
    except ZeroDivisionError:
        # this is when all weights are zero, then ignore them
        return np.average(a)
