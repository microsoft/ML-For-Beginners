import os
import numpy as np

from .arpack import _arpack  # type: ignore[attr-defined]
from . import eigsh

from scipy._lib._util import check_random_state
from scipy.sparse.linalg._interface import LinearOperator, aslinearoperator
from scipy.sparse.linalg._eigen.lobpcg import lobpcg  # type: ignore[no-redef]
if os.environ.get("SCIPY_USE_PROPACK"):
    from scipy.sparse.linalg._svdp import _svdp
    HAS_PROPACK = True
else:
    HAS_PROPACK = False
from scipy.linalg import svd

arpack_int = _arpack.timing.nbx.dtype
__all__ = ['svds']


def _herm(x):
    return x.T.conj()


def _iv(A, k, ncv, tol, which, v0, maxiter,
        return_singular, solver, random_state):

    # input validation/standardization for `solver`
    # out of order because it's needed for other parameters
    solver = str(solver).lower()
    solvers = {"arpack", "lobpcg", "propack"}
    if solver not in solvers:
        raise ValueError(f"solver must be one of {solvers}.")

    # input validation/standardization for `A`
    A = aslinearoperator(A)  # this takes care of some input validation
    if not (np.issubdtype(A.dtype, np.complexfloating)
            or np.issubdtype(A.dtype, np.floating)):
        message = "`A` must be of floating or complex floating data type."
        raise ValueError(message)
    if np.prod(A.shape) == 0:
        message = "`A` must not be empty."
        raise ValueError(message)

    # input validation/standardization for `k`
    kmax = min(A.shape) if solver == 'propack' else min(A.shape) - 1
    if int(k) != k or not (0 < k <= kmax):
        message = "`k` must be an integer satisfying `0 < k < min(A.shape)`."
        raise ValueError(message)
    k = int(k)

    # input validation/standardization for `ncv`
    if solver == "arpack" and ncv is not None:
        if int(ncv) != ncv or not (k < ncv < min(A.shape)):
            message = ("`ncv` must be an integer satisfying "
                       "`k < ncv < min(A.shape)`.")
            raise ValueError(message)
        ncv = int(ncv)

    # input validation/standardization for `tol`
    if tol < 0 or not np.isfinite(tol):
        message = "`tol` must be a non-negative floating point value."
        raise ValueError(message)
    tol = float(tol)

    # input validation/standardization for `which`
    which = str(which).upper()
    whichs = {'LM', 'SM'}
    if which not in whichs:
        raise ValueError(f"`which` must be in {whichs}.")

    # input validation/standardization for `v0`
    if v0 is not None:
        v0 = np.atleast_1d(v0)
        if not (np.issubdtype(v0.dtype, np.complexfloating)
                or np.issubdtype(v0.dtype, np.floating)):
            message = ("`v0` must be of floating or complex floating "
                       "data type.")
            raise ValueError(message)

        shape = (A.shape[0],) if solver == 'propack' else (min(A.shape),)
        if v0.shape != shape:
            message = f"`v0` must have shape {shape}."
            raise ValueError(message)

    # input validation/standardization for `maxiter`
    if maxiter is not None and (int(maxiter) != maxiter or maxiter <= 0):
        message = "`maxiter` must be a positive integer."
        raise ValueError(message)
    maxiter = int(maxiter) if maxiter is not None else maxiter

    # input validation/standardization for `return_singular_vectors`
    # not going to be flexible with this; too complicated for little gain
    rs_options = {True, False, "vh", "u"}
    if return_singular not in rs_options:
        raise ValueError(f"`return_singular_vectors` must be in {rs_options}.")

    random_state = check_random_state(random_state)

    return (A, k, ncv, tol, which, v0, maxiter,
            return_singular, solver, random_state)


def svds(A, k=6, ncv=None, tol=0, which='LM', v0=None,
         maxiter=None, return_singular_vectors=True,
         solver='arpack', random_state=None, options=None):
    """
    Partial singular value decomposition of a sparse matrix.

    Compute the largest or smallest `k` singular values and corresponding
    singular vectors of a sparse matrix `A`. The order in which the singular
    values are returned is not guaranteed.

    In the descriptions below, let ``M, N = A.shape``.

    Parameters
    ----------
    A : ndarray, sparse matrix, or LinearOperator
        Matrix to decompose of a floating point numeric dtype.
    k : int, default: 6
        Number of singular values and singular vectors to compute.
        Must satisfy ``1 <= k <= kmax``, where ``kmax=min(M, N)`` for
        ``solver='propack'`` and ``kmax=min(M, N) - 1`` otherwise.
    ncv : int, optional
        When ``solver='arpack'``, this is the number of Lanczos vectors
        generated. See :ref:`'arpack' <sparse.linalg.svds-arpack>` for details.
        When ``solver='lobpcg'`` or ``solver='propack'``, this parameter is
        ignored.
    tol : float, optional
        Tolerance for singular values. Zero (default) means machine precision.
    which : {'LM', 'SM'}
        Which `k` singular values to find: either the largest magnitude ('LM')
        or smallest magnitude ('SM') singular values.
    v0 : ndarray, optional
        The starting vector for iteration; see method-specific
        documentation (:ref:`'arpack' <sparse.linalg.svds-arpack>`,
        :ref:`'lobpcg' <sparse.linalg.svds-lobpcg>`), or
        :ref:`'propack' <sparse.linalg.svds-propack>` for details.
    maxiter : int, optional
        Maximum number of iterations; see method-specific
        documentation (:ref:`'arpack' <sparse.linalg.svds-arpack>`,
        :ref:`'lobpcg' <sparse.linalg.svds-lobpcg>`), or
        :ref:`'propack' <sparse.linalg.svds-propack>` for details.
    return_singular_vectors : {True, False, "u", "vh"}
        Singular values are always computed and returned; this parameter
        controls the computation and return of singular vectors.

        - ``True``: return singular vectors.
        - ``False``: do not return singular vectors.
        - ``"u"``: if ``M <= N``, compute only the left singular vectors and
          return ``None`` for the right singular vectors. Otherwise, compute
          all singular vectors.
        - ``"vh"``: if ``M > N``, compute only the right singular vectors and
          return ``None`` for the left singular vectors. Otherwise, compute
          all singular vectors.

        If ``solver='propack'``, the option is respected regardless of the
        matrix shape.

    solver :  {'arpack', 'propack', 'lobpcg'}, optional
            The solver used.
            :ref:`'arpack' <sparse.linalg.svds-arpack>`,
            :ref:`'lobpcg' <sparse.linalg.svds-lobpcg>`, and
            :ref:`'propack' <sparse.linalg.svds-propack>` are supported.
            Default: `'arpack'`.
    random_state : {None, int, `numpy.random.Generator`,
                    `numpy.random.RandomState`}, optional

        Pseudorandom number generator state used to generate resamples.

        If `random_state` is ``None`` (or `np.random`), the
        `numpy.random.RandomState` singleton is used.
        If `random_state` is an int, a new ``RandomState`` instance is used,
        seeded with `random_state`.
        If `random_state` is already a ``Generator`` or ``RandomState``
        instance then that instance is used.
    options : dict, optional
        A dictionary of solver-specific options. No solver-specific options
        are currently supported; this parameter is reserved for future use.

    Returns
    -------
    u : ndarray, shape=(M, k)
        Unitary matrix having left singular vectors as columns.
    s : ndarray, shape=(k,)
        The singular values.
    vh : ndarray, shape=(k, N)
        Unitary matrix having right singular vectors as rows.

    Notes
    -----
    This is a naive implementation using ARPACK or LOBPCG as an eigensolver
    on the matrix ``A.conj().T @ A`` or ``A @ A.conj().T``, depending on
    which one is smaller size, followed by the Rayleigh-Ritz method
    as postprocessing; see
    Using the normal matrix, in Rayleigh-Ritz method, (2022, Nov. 19),
    Wikipedia, https://w.wiki/4zms.

    Alternatively, the PROPACK solver can be called.

    Choices of the input matrix `A` numeric dtype may be limited.
    Only ``solver="lobpcg"`` supports all floating point dtypes
    real: 'np.single', 'np.double', 'np.longdouble' and
    complex: 'np.csingle', 'np.cdouble', 'np.clongdouble'.
    The ``solver="arpack"`` supports only
    'np.single', 'np.double', and 'np.cdouble'.

    Examples
    --------
    Construct a matrix `A` from singular values and vectors.

    >>> import numpy as np
    >>> from scipy import sparse, linalg, stats
    >>> from scipy.sparse.linalg import svds, aslinearoperator, LinearOperator

    Construct a dense matrix `A` from singular values and vectors.

    >>> rng = np.random.default_rng(258265244568965474821194062361901728911)
    >>> orthogonal = stats.ortho_group.rvs(10, random_state=rng)
    >>> s = [1e-3, 1, 2, 3, 4]  # non-zero singular values
    >>> u = orthogonal[:, :5]         # left singular vectors
    >>> vT = orthogonal[:, 5:].T      # right singular vectors
    >>> A = u @ np.diag(s) @ vT

    With only four singular values/vectors, the SVD approximates the original
    matrix.

    >>> u4, s4, vT4 = svds(A, k=4)
    >>> A4 = u4 @ np.diag(s4) @ vT4
    >>> np.allclose(A4, A, atol=1e-3)
    True

    With all five non-zero singular values/vectors, we can reproduce
    the original matrix more accurately.

    >>> u5, s5, vT5 = svds(A, k=5)
    >>> A5 = u5 @ np.diag(s5) @ vT5
    >>> np.allclose(A5, A)
    True

    The singular values match the expected singular values.

    >>> np.allclose(s5, s)
    True

    Since the singular values are not close to each other in this example,
    every singular vector matches as expected up to a difference in sign.

    >>> (np.allclose(np.abs(u5), np.abs(u)) and
    ...  np.allclose(np.abs(vT5), np.abs(vT)))
    True

    The singular vectors are also orthogonal.

    >>> (np.allclose(u5.T @ u5, np.eye(5)) and
    ...  np.allclose(vT5 @ vT5.T, np.eye(5)))
    True

    If there are (nearly) multiple singular values, the corresponding
    individual singular vectors may be unstable, but the whole invariant
    subspace containing all such singular vectors is computed accurately
    as can be measured by angles between subspaces via 'subspace_angles'.

    >>> rng = np.random.default_rng(178686584221410808734965903901790843963)
    >>> s = [1, 1 + 1e-6]  # non-zero singular values
    >>> u, _ = np.linalg.qr(rng.standard_normal((99, 2)))
    >>> v, _ = np.linalg.qr(rng.standard_normal((99, 2)))
    >>> vT = v.T
    >>> A = u @ np.diag(s) @ vT
    >>> A = A.astype(np.float32)
    >>> u2, s2, vT2 = svds(A, k=2, random_state=rng)
    >>> np.allclose(s2, s)
    True

    The angles between the individual exact and computed singular vectors
    may not be so small. To check use:

    >>> (linalg.subspace_angles(u2[:, :1], u[:, :1]) +
    ...  linalg.subspace_angles(u2[:, 1:], u[:, 1:]))
    array([0.06562513])  # may vary
    >>> (linalg.subspace_angles(vT2[:1, :].T, vT[:1, :].T) +
    ...  linalg.subspace_angles(vT2[1:, :].T, vT[1:, :].T))
    array([0.06562507])  # may vary

    As opposed to the angles between the 2-dimensional invariant subspaces
    that these vectors span, which are small for rights singular vectors

    >>> linalg.subspace_angles(u2, u).sum() < 1e-6
    True

    as well as for left singular vectors.

    >>> linalg.subspace_angles(vT2.T, vT.T).sum() < 1e-6
    True

    The next example follows that of 'sklearn.decomposition.TruncatedSVD'.

    >>> rng = np.random.RandomState(0)
    >>> X_dense = rng.random(size=(100, 100))
    >>> X_dense[:, 2 * np.arange(50)] = 0
    >>> X = sparse.csr_matrix(X_dense)
    >>> _, singular_values, _ = svds(X, k=5, random_state=rng)
    >>> print(singular_values)
    [ 4.3293...  4.4491...  4.5420...  4.5987... 35.2410...]

    The function can be called without the transpose of the input matrix
    ever explicitly constructed.

    >>> rng = np.random.default_rng(102524723947864966825913730119128190974)
    >>> G = sparse.rand(8, 9, density=0.5, random_state=rng)
    >>> Glo = aslinearoperator(G)
    >>> _, singular_values_svds, _ = svds(Glo, k=5, random_state=rng)
    >>> _, singular_values_svd, _ = linalg.svd(G.toarray())
    >>> np.allclose(singular_values_svds, singular_values_svd[-4::-1])
    True

    The most memory efficient scenario is where neither
    the original matrix, nor its transpose, is explicitly constructed.
    Our example computes the smallest singular values and vectors
    of 'LinearOperator' constructed from the numpy function 'np.diff' used
    column-wise to be consistent with 'LinearOperator' operating on columns.

    >>> diff0 = lambda a: np.diff(a, axis=0)

    Let us create the matrix from 'diff0' to be used for validation only.

    >>> n = 5  # The dimension of the space.
    >>> M_from_diff0 = diff0(np.eye(n))
    >>> print(M_from_diff0.astype(int))
    [[-1  1  0  0  0]
     [ 0 -1  1  0  0]
     [ 0  0 -1  1  0]
     [ 0  0  0 -1  1]]

    The matrix 'M_from_diff0' is bi-diagonal and could be alternatively
    created directly by

    >>> M = - np.eye(n - 1, n, dtype=int)
    >>> np.fill_diagonal(M[:,1:], 1)
    >>> np.allclose(M, M_from_diff0)
    True

    Its transpose

    >>> print(M.T)
    [[-1  0  0  0]
     [ 1 -1  0  0]
     [ 0  1 -1  0]
     [ 0  0  1 -1]
     [ 0  0  0  1]]

    can be viewed as the incidence matrix; see
    Incidence matrix, (2022, Nov. 19), Wikipedia, https://w.wiki/5YXU,
    of a linear graph with 5 vertices and 4 edges. The 5x5 normal matrix
    ``M.T @ M`` thus is

    >>> print(M.T @ M)
    [[ 1 -1  0  0  0]
     [-1  2 -1  0  0]
     [ 0 -1  2 -1  0]
     [ 0  0 -1  2 -1]
     [ 0  0  0 -1  1]]

    the graph Laplacian, while the actually used in 'svds' smaller size
    4x4 normal matrix ``M @ M.T``

    >>> print(M @ M.T)
    [[ 2 -1  0  0]
     [-1  2 -1  0]
     [ 0 -1  2 -1]
     [ 0  0 -1  2]]

    is the so-called edge-based Laplacian; see
    Symmetric Laplacian via the incidence matrix, in Laplacian matrix,
    (2022, Nov. 19), Wikipedia, https://w.wiki/5YXW.

    The 'LinearOperator' setup needs the options 'rmatvec' and 'rmatmat'
    of multiplication by the matrix transpose ``M.T``, but we want to be
    matrix-free to save memory, so knowing how ``M.T`` looks like, we
    manually construct the following function to be
    used in ``rmatmat=diff0t``.

    >>> def diff0t(a):
    ...     if a.ndim == 1:
    ...         a = a[:,np.newaxis]  # Turn 1D into 2D array
    ...     d = np.zeros((a.shape[0] + 1, a.shape[1]), dtype=a.dtype)
    ...     d[0, :] = - a[0, :]
    ...     d[1:-1, :] = a[0:-1, :] - a[1:, :]
    ...     d[-1, :] = a[-1, :]
    ...     return d

    We check that our function 'diff0t' for the matrix transpose is valid.

    >>> np.allclose(M.T, diff0t(np.eye(n-1)))
    True

    Now we setup our matrix-free 'LinearOperator' called 'diff0_func_aslo'
    and for validation the matrix-based 'diff0_matrix_aslo'.

    >>> def diff0_func_aslo_def(n):
    ...     return LinearOperator(matvec=diff0,
    ...                           matmat=diff0,
    ...                           rmatvec=diff0t,
    ...                           rmatmat=diff0t,
    ...                           shape=(n - 1, n))
    >>> diff0_func_aslo = diff0_func_aslo_def(n)
    >>> diff0_matrix_aslo = aslinearoperator(M_from_diff0)

    And validate both the matrix and its transpose in 'LinearOperator'.

    >>> np.allclose(diff0_func_aslo(np.eye(n)),
    ...             diff0_matrix_aslo(np.eye(n)))
    True
    >>> np.allclose(diff0_func_aslo.T(np.eye(n-1)),
    ...             diff0_matrix_aslo.T(np.eye(n-1)))
    True

    Having the 'LinearOperator' setup validated, we run the solver.

    >>> n = 100
    >>> diff0_func_aslo = diff0_func_aslo_def(n)
    >>> u, s, vT = svds(diff0_func_aslo, k=3, which='SM')

    The singular values squared and the singular vectors are known
    explicitly; see
    Pure Dirichlet boundary conditions, in
    Eigenvalues and eigenvectors of the second derivative,
    (2022, Nov. 19), Wikipedia, https://w.wiki/5YX6,
    since 'diff' corresponds to first
    derivative, and its smaller size n-1 x n-1 normal matrix
    ``M @ M.T`` represent the discrete second derivative with the Dirichlet
    boundary conditions. We use these analytic expressions for validation.

    >>> se = 2. * np.sin(np.pi * np.arange(1, 4) / (2. * n))
    >>> ue = np.sqrt(2 / n) * np.sin(np.pi * np.outer(np.arange(1, n),
    ...                              np.arange(1, 4)) / n)
    >>> np.allclose(s, se, atol=1e-3)
    True
    >>> print(np.allclose(np.abs(u), np.abs(ue), atol=1e-6))
    True

    """
    args = _iv(A, k, ncv, tol, which, v0, maxiter, return_singular_vectors,
               solver, random_state)
    (A, k, ncv, tol, which, v0, maxiter,
     return_singular_vectors, solver, random_state) = args

    largest = (which == 'LM')
    n, m = A.shape

    if n >= m:
        X_dot = A.matvec
        X_matmat = A.matmat
        XH_dot = A.rmatvec
        XH_mat = A.rmatmat
        transpose = False
    else:
        X_dot = A.rmatvec
        X_matmat = A.rmatmat
        XH_dot = A.matvec
        XH_mat = A.matmat
        transpose = True

        dtype = getattr(A, 'dtype', None)
        if dtype is None:
            dtype = A.dot(np.zeros([m, 1])).dtype

    def matvec_XH_X(x):
        return XH_dot(X_dot(x))

    def matmat_XH_X(x):
        return XH_mat(X_matmat(x))

    XH_X = LinearOperator(matvec=matvec_XH_X, dtype=A.dtype,
                          matmat=matmat_XH_X,
                          shape=(min(A.shape), min(A.shape)))

    # Get a low rank approximation of the implicitly defined gramian matrix.
    # This is not a stable way to approach the problem.
    if solver == 'lobpcg':

        if k == 1 and v0 is not None:
            X = np.reshape(v0, (-1, 1))
        else:
            X = random_state.standard_normal(size=(min(A.shape), k))

        _, eigvec = lobpcg(XH_X, X, tol=tol ** 2, maxiter=maxiter,
                           largest=largest)

    elif solver == 'propack':
        if not HAS_PROPACK:
            raise ValueError("`solver='propack'` is opt-in due "
                             "to potential issues on Windows, "
                             "it can be enabled by setting the "
                             "`SCIPY_USE_PROPACK` environment "
                             "variable before importing scipy")
        jobu = return_singular_vectors in {True, 'u'}
        jobv = return_singular_vectors in {True, 'vh'}
        irl_mode = (which == 'SM')
        res = _svdp(A, k=k, tol=tol**2, which=which, maxiter=None,
                    compute_u=jobu, compute_v=jobv, irl_mode=irl_mode,
                    kmax=maxiter, v0=v0, random_state=random_state)

        u, s, vh, _ = res  # but we'll ignore bnd, the last output

        # PROPACK order appears to be largest first. `svds` output order is not
        # guaranteed, according to documentation, but for ARPACK and LOBPCG
        # they actually are ordered smallest to largest, so reverse for
        # consistency.
        s = s[::-1]
        u = u[:, ::-1]
        vh = vh[::-1]

        u = u if jobu else None
        vh = vh if jobv else None

        if return_singular_vectors:
            return u, s, vh
        else:
            return s

    elif solver == 'arpack' or solver is None:
        if v0 is None:
            v0 = random_state.standard_normal(size=(min(A.shape),))
        _, eigvec = eigsh(XH_X, k=k, tol=tol ** 2, maxiter=maxiter,
                          ncv=ncv, which=which, v0=v0)
        # arpack do not guarantee exactly orthonormal eigenvectors
        # for clustered eigenvalues, especially in complex arithmetic
        eigvec, _ = np.linalg.qr(eigvec)

    # the eigenvectors eigvec must be orthonomal here; see gh-16712
    Av = X_matmat(eigvec)
    if not return_singular_vectors:
        s = svd(Av, compute_uv=False, overwrite_a=True)
        return s[::-1]

    # compute the left singular vectors of X and update the right ones
    # accordingly
    u, s, vh = svd(Av, full_matrices=False, overwrite_a=True)
    u = u[:, ::-1]
    s = s[::-1]
    vh = vh[::-1]

    jobu = return_singular_vectors in {True, 'u'}
    jobv = return_singular_vectors in {True, 'vh'}

    if transpose:
        u_tmp = eigvec @ _herm(vh) if jobu else None
        vh = _herm(u) if jobv else None
        u = u_tmp
    else:
        if not jobu:
            u = None
        vh = vh @ _herm(eigvec) if jobv else None

    return u, s, vh
