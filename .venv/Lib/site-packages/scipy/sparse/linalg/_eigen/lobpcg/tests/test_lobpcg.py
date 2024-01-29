""" Test functions for the sparse.linalg._eigen.lobpcg module
"""

import itertools
import platform
import sys
import pytest
import numpy as np
from numpy import ones, r_, diag
from numpy.testing import (assert_almost_equal, assert_equal,
                           assert_allclose, assert_array_less)

from scipy import sparse
from scipy.linalg import eig, eigh, toeplitz, orth
from scipy.sparse import spdiags, diags, eye, csr_matrix
from scipy.sparse.linalg import eigs, LinearOperator
from scipy.sparse.linalg._eigen.lobpcg import lobpcg
from scipy.sparse.linalg._eigen.lobpcg.lobpcg import _b_orthonormalize
from scipy._lib._util import np_long, np_ulong

_IS_32BIT = (sys.maxsize < 2**32)

INT_DTYPES = {np.intc, np_long, np.longlong, np.uintc, np_ulong, np.ulonglong}
# np.half is unsupported on many test systems so excluded
REAL_DTYPES = {np.float32, np.float64, np.longdouble}
COMPLEX_DTYPES = {np.complex64, np.complex128, np.clongdouble}
# use sorted list to ensure fixed order of tests
VDTYPES = sorted(REAL_DTYPES ^ COMPLEX_DTYPES, key=str)
MDTYPES = sorted(INT_DTYPES ^ REAL_DTYPES ^ COMPLEX_DTYPES, key=str)


def sign_align(A, B):
    """Align signs of columns of A match those of B: column-wise remove
    sign of A by multiplying with its sign then multiply in sign of B.
    """
    return np.array([col_A * np.sign(col_A[0]) * np.sign(col_B[0])
                     for col_A, col_B in zip(A.T, B.T)]).T

def ElasticRod(n):
    """Build the matrices for the generalized eigenvalue problem of the
    fixed-free elastic rod vibration model.
    """
    L = 1.0
    le = L/n
    rho = 7.85e3
    S = 1.e-4
    E = 2.1e11
    mass = rho*S*le/6.
    k = E*S/le
    A = k*(diag(r_[2.*ones(n-1), 1])-diag(ones(n-1), 1)-diag(ones(n-1), -1))
    B = mass*(diag(r_[4.*ones(n-1), 2])+diag(ones(n-1), 1)+diag(ones(n-1), -1))
    return A, B


def MikotaPair(n):
    """Build a pair of full diagonal matrices for the generalized eigenvalue
    problem. The Mikota pair acts as a nice test since the eigenvalues are the
    squares of the integers n, n=1,2,...
    """
    x = np.arange(1, n+1)
    B = diag(1./x)
    y = np.arange(n-1, 0, -1)
    z = np.arange(2*n-1, 0, -2)
    A = diag(z)-diag(y, -1)-diag(y, 1)
    return A, B


def compare_solutions(A, B, m):
    """Check eig vs. lobpcg consistency.
    """
    n = A.shape[0]
    rnd = np.random.RandomState(0)
    V = rnd.random((n, m))
    X = orth(V)
    eigvals, _ = lobpcg(A, X, B=B, tol=1e-2, maxiter=50, largest=False)
    eigvals.sort()
    w, _ = eig(A, b=B)
    w.sort()
    assert_almost_equal(w[:int(m/2)], eigvals[:int(m/2)], decimal=2)


def test_Small():
    A, B = ElasticRod(10)
    with pytest.warns(UserWarning, match="The problem size"):
        compare_solutions(A, B, 10)
    A, B = MikotaPair(10)
    with pytest.warns(UserWarning, match="The problem size"):
        compare_solutions(A, B, 10)


def test_ElasticRod():
    A, B = ElasticRod(20)
    msg = "Exited at iteration.*|Exited postprocessing with accuracies.*"
    with pytest.warns(UserWarning, match=msg):
        compare_solutions(A, B, 2)


def test_MikotaPair():
    A, B = MikotaPair(20)
    compare_solutions(A, B, 2)


@pytest.mark.parametrize("n", [50])
@pytest.mark.parametrize("m", [1, 2, 10])
@pytest.mark.parametrize("Vdtype", sorted(REAL_DTYPES, key=str))
@pytest.mark.parametrize("Bdtype", sorted(REAL_DTYPES, key=str))
@pytest.mark.parametrize("BVdtype", sorted(REAL_DTYPES, key=str))
def test_b_orthonormalize(n, m, Vdtype, Bdtype, BVdtype):
    """Test B-orthonormalization by Cholesky with callable 'B'.
    The function '_b_orthonormalize' is key in LOBPCG but may
    lead to numerical instabilities. The input vectors are often
    badly scaled, so the function needs scale-invariant Cholesky;
    see https://netlib.org/lapack/lawnspdf/lawn14.pdf.
    """
    rnd = np.random.RandomState(0)
    X = rnd.standard_normal((n, m)).astype(Vdtype)
    Xcopy = np.copy(X)
    vals = np.arange(1, n+1, dtype=float)
    B = diags([vals], [0], (n, n)).astype(Bdtype)
    BX = B @ X
    BX = BX.astype(BVdtype)
    dtype = min(X.dtype, B.dtype, BX.dtype)
    # np.longdouble tol cannot be achieved on most systems
    atol = m * n * max(np.finfo(dtype).eps, np.finfo(np.float64).eps)

    Xo, BXo, _ = _b_orthonormalize(lambda v: B @ v, X, BX)
    # Check in-place.
    assert_equal(X, Xo)
    assert_equal(id(X), id(Xo))
    assert_equal(BX, BXo)
    assert_equal(id(BX), id(BXo))
    # Check BXo.
    assert_allclose(B @ Xo, BXo, atol=atol, rtol=atol)
    # Check B-orthonormality
    assert_allclose(Xo.T.conj() @ B @ Xo, np.identity(m),
                    atol=atol, rtol=atol)
    # Repeat without BX in outputs
    X = np.copy(Xcopy)
    Xo1, BXo1, _ = _b_orthonormalize(lambda v: B @ v, X)
    assert_allclose(Xo, Xo1, atol=atol, rtol=atol)
    assert_allclose(BXo, BXo1, atol=atol, rtol=atol)
    # Check in-place.
    assert_equal(X, Xo1)
    assert_equal(id(X), id(Xo1))
    # Check BXo1.
    assert_allclose(B @ Xo1, BXo1, atol=atol, rtol=atol)

    # Introduce column-scaling in X.
    scaling = 1.0 / np.geomspace(10, 1e10, num=m)
    X = Xcopy * scaling
    X = X.astype(Vdtype)
    BX = B @ X
    BX = BX.astype(BVdtype)
    # Check scaling-invariance of Cholesky-based orthonormalization
    Xo1, BXo1, _ = _b_orthonormalize(lambda v: B @ v, X, BX)
    # The output should be the same, up the signs of the columns.
    Xo1 =  sign_align(Xo1, Xo)
    assert_allclose(Xo, Xo1, atol=atol, rtol=atol)
    BXo1 =  sign_align(BXo1, BXo)
    assert_allclose(BXo, BXo1, atol=atol, rtol=atol)


@pytest.mark.filterwarnings("ignore:Exited at iteration 0")
@pytest.mark.filterwarnings("ignore:Exited postprocessing")
def test_nonhermitian_warning(capsys):
    """Check the warning of a Ritz matrix being not Hermitian
    by feeding a non-Hermitian input matrix.
    Also check stdout since verbosityLevel=1 and lack of stderr.
    """
    n = 10
    X = np.arange(n * 2).reshape(n, 2).astype(np.float32)
    A = np.arange(n * n).reshape(n, n).astype(np.float32)
    with pytest.warns(UserWarning, match="Matrix gramA"):
        _, _ = lobpcg(A, X, verbosityLevel=1, maxiter=0)
    out, err = capsys.readouterr()  # Capture output
    assert out.startswith("Solving standard eigenvalue")  # Test stdout
    assert err == ''  # Test empty stderr
    # Make the matrix symmetric and the UserWarning disappears.
    A += A.T
    _, _ = lobpcg(A, X, verbosityLevel=1, maxiter=0)
    out, err = capsys.readouterr()  # Capture output
    assert out.startswith("Solving standard eigenvalue")  # Test stdout
    assert err == ''  # Test empty stderr


def test_regression():
    """Check the eigenvalue of the identity matrix is one.
    """
    # https://mail.python.org/pipermail/scipy-user/2010-October/026944.html
    n = 10
    X = np.ones((n, 1))
    A = np.identity(n)
    w, _ = lobpcg(A, X)
    assert_allclose(w, [1])


@pytest.mark.filterwarnings("ignore:The problem size")
@pytest.mark.parametrize('n, m, m_excluded', [(30, 4, 3), (4, 2, 0)])
def test_diagonal(n, m, m_excluded):
    """Test ``m - m_excluded`` eigenvalues and eigenvectors of
    diagonal matrices of the size ``n`` varying matrix formats:
    dense array, spare matrix, and ``LinearOperator`` for both
    matrixes in the generalized eigenvalue problem ``Av = cBv``
    and for the preconditioner.
    """
    rnd = np.random.RandomState(0)

    # Define the generalized eigenvalue problem Av = cBv
    # where (c, v) is a generalized eigenpair,
    # A is the diagonal matrix whose entries are 1,...n,
    # B is the identity matrix.
    vals = np.arange(1, n+1, dtype=float)
    A_s = diags([vals], [0], (n, n))
    A_a = A_s.toarray()

    def A_f(x):
        return A_s @ x

    A_lo = LinearOperator(matvec=A_f,
                          matmat=A_f,
                          shape=(n, n), dtype=float)

    B_a = eye(n)
    B_s = csr_matrix(B_a)

    def B_f(x):
        return B_a @ x

    B_lo = LinearOperator(matvec=B_f,
                          matmat=B_f,
                          shape=(n, n), dtype=float)

    # Let the preconditioner M be the inverse of A.
    M_s = diags([1./vals], [0], (n, n))
    M_a = M_s.toarray()

    def M_f(x):
        return M_s @ x

    M_lo = LinearOperator(matvec=M_f,
                          matmat=M_f,
                          shape=(n, n), dtype=float)

    # Pick random initial vectors.
    X = rnd.normal(size=(n, m))

    # Require that the returned eigenvectors be in the orthogonal complement
    # of the first few standard basis vectors.
    if m_excluded > 0:
        Y = np.eye(n, m_excluded)
    else:
        Y = None

    for A in [A_a, A_s, A_lo]:
        for B in [B_a, B_s, B_lo]:
            for M in [M_a, M_s, M_lo]:
                eigvals, vecs = lobpcg(A, X, B, M=M, Y=Y,
                                       maxiter=40, largest=False)

                assert_allclose(eigvals, np.arange(1+m_excluded,
                                                   1+m_excluded+m))
                _check_eigen(A, eigvals, vecs, rtol=1e-3, atol=1e-3)


def _check_eigen(M, w, V, rtol=1e-8, atol=1e-14):
    """Check if the eigenvalue residual is small.
    """
    mult_wV = np.multiply(w, V)
    dot_MV = M.dot(V)
    assert_allclose(mult_wV, dot_MV, rtol=rtol, atol=atol)


def _check_fiedler(n, p):
    """Check the Fiedler vector computation.
    """
    # This is not necessarily the recommended way to find the Fiedler vector.
    col = np.zeros(n)
    col[1] = 1
    A = toeplitz(col)
    D = np.diag(A.sum(axis=1))
    L = D - A
    # Compute the full eigendecomposition using tricks, e.g.
    # http://www.cs.yale.edu/homes/spielman/561/2009/lect02-09.pdf
    tmp = np.pi * np.arange(n) / n
    analytic_w = 2 * (1 - np.cos(tmp))
    analytic_V = np.cos(np.outer(np.arange(n) + 1/2, tmp))
    _check_eigen(L, analytic_w, analytic_V)
    # Compute the full eigendecomposition using eigh.
    eigh_w, eigh_V = eigh(L)
    _check_eigen(L, eigh_w, eigh_V)
    # Check that the first eigenvalue is near zero and that the rest agree.
    assert_array_less(np.abs([eigh_w[0], analytic_w[0]]), 1e-14)
    assert_allclose(eigh_w[1:], analytic_w[1:])

    # Check small lobpcg eigenvalues.
    X = analytic_V[:, :p]
    lobpcg_w, lobpcg_V = lobpcg(L, X, largest=False)
    assert_equal(lobpcg_w.shape, (p,))
    assert_equal(lobpcg_V.shape, (n, p))
    _check_eigen(L, lobpcg_w, lobpcg_V)
    assert_array_less(np.abs(np.min(lobpcg_w)), 1e-14)
    assert_allclose(np.sort(lobpcg_w)[1:], analytic_w[1:p])

    # Check large lobpcg eigenvalues.
    X = analytic_V[:, -p:]
    lobpcg_w, lobpcg_V = lobpcg(L, X, largest=True)
    assert_equal(lobpcg_w.shape, (p,))
    assert_equal(lobpcg_V.shape, (n, p))
    _check_eigen(L, lobpcg_w, lobpcg_V)
    assert_allclose(np.sort(lobpcg_w), analytic_w[-p:])

    # Look for the Fiedler vector using good but not exactly correct guesses.
    fiedler_guess = np.concatenate((np.ones(n//2), -np.ones(n-n//2)))
    X = np.vstack((np.ones(n), fiedler_guess)).T
    lobpcg_w, _ = lobpcg(L, X, largest=False)
    # Mathematically, the smaller eigenvalue should be zero
    # and the larger should be the algebraic connectivity.
    lobpcg_w = np.sort(lobpcg_w)
    assert_allclose(lobpcg_w, analytic_w[:2], atol=1e-14)


def test_fiedler_small_8():
    """Check the dense workaround path for small matrices.
    """
    # This triggers the dense path because 8 < 2*5.
    with pytest.warns(UserWarning, match="The problem size"):
        _check_fiedler(8, 2)


def test_fiedler_large_12():
    """Check the dense workaround path avoided for non-small matrices.
    """
    # This does not trigger the dense path, because 2*5 <= 12.
    _check_fiedler(12, 2)


@pytest.mark.filterwarnings("ignore:Failed at iteration")
@pytest.mark.filterwarnings("ignore:Exited at iteration")
@pytest.mark.filterwarnings("ignore:Exited postprocessing")
def test_failure_to_run_iterations():
    """Check that the code exits gracefully without breaking. Issue #10974.
    The code may or not issue a warning, filtered out. Issue #15935, #17954.
    """
    rnd = np.random.RandomState(0)
    X = rnd.standard_normal((100, 10))
    A = X @ X.T
    Q = rnd.standard_normal((X.shape[0], 4))
    eigenvalues, _ = lobpcg(A, Q, maxiter=40, tol=1e-12)
    assert np.max(eigenvalues) > 0


def test_failure_to_run_iterations_nonsymmetric():
    """Check that the code exists gracefully without breaking
    if the matrix in not symmetric.
    """
    A = np.zeros((10, 10))
    A[0, 1] = 1
    Q = np.ones((10, 1))
    msg = "Exited at iteration 2|Exited postprocessing with accuracies.*"
    with pytest.warns(UserWarning, match=msg):
        eigenvalues, _ = lobpcg(A, Q, maxiter=20)
    assert np.max(eigenvalues) > 0


@pytest.mark.filterwarnings("ignore:The problem size")
def test_hermitian():
    """Check complex-value Hermitian cases.
    """
    rnd = np.random.RandomState(0)

    sizes = [3, 12]
    ks = [1, 2]
    gens = [True, False]

    for s, k, gen, dh, dx, db in (
        itertools.product(sizes, ks, gens, gens, gens, gens)
    ):
        H = rnd.random((s, s)) + 1.j * rnd.random((s, s))
        H = 10 * np.eye(s) + H + H.T.conj()
        H = H.astype(np.complex128) if dh else H.astype(np.complex64)

        X = rnd.standard_normal((s, k))
        X = X + 1.j * rnd.standard_normal((s, k))
        X = X.astype(np.complex128) if dx else X.astype(np.complex64)

        if not gen:
            B = np.eye(s)
            w, v = lobpcg(H, X, maxiter=99, verbosityLevel=0)
            # Also test mixing complex H with real B.
            wb, _ = lobpcg(H, X, B, maxiter=99, verbosityLevel=0)
            assert_allclose(w, wb, rtol=1e-6)
            w0, _ = eigh(H)
        else:
            B = rnd.random((s, s)) + 1.j * rnd.random((s, s))
            B = 10 * np.eye(s) + B.dot(B.T.conj())
            B = B.astype(np.complex128) if db else B.astype(np.complex64)
            w, v = lobpcg(H, X, B, maxiter=99, verbosityLevel=0)
            w0, _ = eigh(H, B)

        for wx, vx in zip(w, v.T):
            # Check eigenvector
            assert_allclose(np.linalg.norm(H.dot(vx) - B.dot(vx) * wx)
                            / np.linalg.norm(H.dot(vx)),
                            0, atol=5e-2, rtol=0)

            # Compare eigenvalues
            j = np.argmin(abs(w0 - wx))
            assert_allclose(wx, w0[j], rtol=1e-4)


# The n=5 case tests the alternative small matrix code path that uses eigh().
@pytest.mark.filterwarnings("ignore:The problem size")
@pytest.mark.parametrize('n, atol', [(20, 1e-3), (5, 1e-8)])
def test_eigs_consistency(n, atol):
    """Check eigs vs. lobpcg consistency.
    """
    vals = np.arange(1, n+1, dtype=np.float64)
    A = spdiags(vals, 0, n, n)
    rnd = np.random.RandomState(0)
    X = rnd.standard_normal((n, 2))
    lvals, lvecs = lobpcg(A, X, largest=True, maxiter=100)
    vals, _ = eigs(A, k=2)

    _check_eigen(A, lvals, lvecs, atol=atol, rtol=0)
    assert_allclose(np.sort(vals), np.sort(lvals), atol=1e-14)


def test_verbosity():
    """Check that nonzero verbosity level code runs.
    """
    rnd = np.random.RandomState(0)
    X = rnd.standard_normal((10, 10))
    A = X @ X.T
    Q = rnd.standard_normal((X.shape[0], 1))
    msg = "Exited at iteration.*|Exited postprocessing with accuracies.*"
    with pytest.warns(UserWarning, match=msg):
        _, _ = lobpcg(A, Q, maxiter=3, verbosityLevel=9)


@pytest.mark.xfail(_IS_32BIT and sys.platform == 'win32',
                   reason="tolerance violation on windows")
@pytest.mark.xfail(platform.machine() == 'ppc64le',
                   reason="fails on ppc64le")
@pytest.mark.filterwarnings("ignore:Exited postprocessing")
def test_tolerance_float32():
    """Check lobpcg for attainable tolerance in float32.
    """
    rnd = np.random.RandomState(0)
    n = 50
    m = 3
    vals = -np.arange(1, n + 1)
    A = diags([vals], [0], (n, n))
    A = A.astype(np.float32)
    X = rnd.standard_normal((n, m))
    X = X.astype(np.float32)
    eigvals, _ = lobpcg(A, X, tol=1.25e-5, maxiter=50, verbosityLevel=0)
    assert_allclose(eigvals, -np.arange(1, 1 + m), atol=2e-5, rtol=1e-5)


@pytest.mark.parametrize("vdtype", VDTYPES)
@pytest.mark.parametrize("mdtype", MDTYPES)
@pytest.mark.parametrize("arr_type", [np.array,
                                      sparse.csr_matrix,
                                      sparse.coo_matrix])
def test_dtypes(vdtype, mdtype, arr_type):
    """Test lobpcg in various dtypes.
    """
    rnd = np.random.RandomState(0)
    n = 12
    m = 2
    A = arr_type(np.diag(np.arange(1, n + 1)).astype(mdtype))
    X = rnd.random((n, m))
    X = X.astype(vdtype)
    eigvals, eigvecs = lobpcg(A, X, tol=1e-2, largest=False)
    assert_allclose(eigvals, np.arange(1, 1 + m), atol=1e-1)
    # eigenvectors must be nearly real in any case
    assert_allclose(np.sum(np.abs(eigvecs - eigvecs.conj())), 0, atol=1e-2)


@pytest.mark.filterwarnings("ignore:Exited at iteration")
@pytest.mark.filterwarnings("ignore:Exited postprocessing")
def test_inplace_warning():
    """Check lobpcg gives a warning in '_b_orthonormalize'
    that in-place orthogonalization is impossible due to dtype mismatch.
    """
    rnd = np.random.RandomState(0)
    n = 6
    m = 1
    vals = -np.arange(1, n + 1)
    A = diags([vals], [0], (n, n))
    A = A.astype(np.cdouble)
    X = rnd.standard_normal((n, m))
    with pytest.warns(UserWarning, match="Inplace update"):
        eigvals, _ = lobpcg(A, X, maxiter=2, verbosityLevel=1)


def test_maxit():
    """Check lobpcg if maxit=maxiter runs maxiter iterations and
    if maxit=None runs 20 iterations (the default)
    by checking the size of the iteration history output, which should
    be the number of iterations plus 3 (initial, final, and postprocessing)
    typically when maxiter is small and the choice of the best is passive.
    """
    rnd = np.random.RandomState(0)
    n = 50
    m = 4
    vals = -np.arange(1, n + 1)
    A = diags([vals], [0], (n, n))
    A = A.astype(np.float32)
    X = rnd.standard_normal((n, m))
    X = X.astype(np.float64)
    msg = "Exited at iteration.*|Exited postprocessing with accuracies.*"
    for maxiter in range(1, 4):
        with pytest.warns(UserWarning, match=msg):
            _, _, l_h, r_h = lobpcg(A, X, tol=1e-8, maxiter=maxiter,
                                    retLambdaHistory=True,
                                    retResidualNormsHistory=True)
        assert_allclose(np.shape(l_h)[0], maxiter+3)
        assert_allclose(np.shape(r_h)[0], maxiter+3)
    with pytest.warns(UserWarning, match=msg):
        l, _, l_h, r_h = lobpcg(A, X, tol=1e-8,
                                retLambdaHistory=True,
                                retResidualNormsHistory=True)
    assert_allclose(np.shape(l_h)[0], 20+3)
    assert_allclose(np.shape(r_h)[0], 20+3)
    # Check that eigenvalue output is the last one in history
    assert_allclose(l, l_h[-1])
    # Make sure that both history outputs are lists
    assert isinstance(l_h, list)
    assert isinstance(r_h, list)
    # Make sure that both history lists are arrays-like
    assert_allclose(np.shape(l_h), np.shape(np.asarray(l_h)))
    assert_allclose(np.shape(r_h), np.shape(np.asarray(r_h)))


@pytest.mark.slow
@pytest.mark.parametrize("n", [15])
@pytest.mark.parametrize("m", [1, 2])
@pytest.mark.filterwarnings("ignore:Exited at iteration")
@pytest.mark.filterwarnings("ignore:Exited postprocessing")
def test_diagonal_data_types(n, m):
    """Check lobpcg for diagonal matrices for all matrix types.
    Constraints are imposed, so a dense eigensolver eig cannot run.
    """
    rnd = np.random.RandomState(0)
    # Define the generalized eigenvalue problem Av = cBv
    # where (c, v) is a generalized eigenpair,
    # and where we choose A  and B to be diagonal.
    vals = np.arange(1, n + 1)

    # list_sparse_format = ['bsr', 'coo', 'csc', 'csr', 'dia', 'dok', 'lil']
    list_sparse_format = ['coo']
    sparse_formats = len(list_sparse_format)
    for s_f_i, s_f in enumerate(list_sparse_format):

        As64 = diags([vals * vals], [0], (n, n), format=s_f)
        As32 = As64.astype(np.float32)
        Af64 = As64.toarray()
        Af32 = Af64.astype(np.float32)

        def As32f(x):
            return As32 @ x
        As32LO = LinearOperator(matvec=As32f,
                                matmat=As32f,
                                shape=(n, n),
                                dtype=As32.dtype)

        listA = [Af64, As64, Af32, As32, As32f, As32LO, lambda v: As32 @ v]

        Bs64 = diags([vals], [0], (n, n), format=s_f)
        Bf64 = Bs64.toarray()
        Bs32 = Bs64.astype(np.float32)

        def Bs32f(x):
            return Bs32 @ x
        Bs32LO = LinearOperator(matvec=Bs32f,
                                matmat=Bs32f,
                                shape=(n, n),
                                dtype=Bs32.dtype)
        listB = [Bf64, Bs64, Bs32, Bs32f, Bs32LO, lambda v: Bs32 @ v]

        # Define the preconditioner function as LinearOperator.
        Ms64 = diags([1./vals], [0], (n, n), format=s_f)

        def Ms64precond(x):
            return Ms64 @ x
        Ms64precondLO = LinearOperator(matvec=Ms64precond,
                                       matmat=Ms64precond,
                                       shape=(n, n),
                                       dtype=Ms64.dtype)
        Mf64 = Ms64.toarray()

        def Mf64precond(x):
            return Mf64 @ x
        Mf64precondLO = LinearOperator(matvec=Mf64precond,
                                       matmat=Mf64precond,
                                       shape=(n, n),
                                       dtype=Mf64.dtype)
        Ms32 = Ms64.astype(np.float32)

        def Ms32precond(x):
            return Ms32 @ x
        Ms32precondLO = LinearOperator(matvec=Ms32precond,
                                       matmat=Ms32precond,
                                       shape=(n, n),
                                       dtype=Ms32.dtype)
        Mf32 = Ms32.toarray()

        def Mf32precond(x):
            return Mf32 @ x
        Mf32precondLO = LinearOperator(matvec=Mf32precond,
                                       matmat=Mf32precond,
                                       shape=(n, n),
                                       dtype=Mf32.dtype)
        listM = [None, Ms64, Ms64precondLO, Mf64precondLO, Ms64precond,
                 Ms32, Ms32precondLO, Mf32precondLO, Ms32precond]

        # Setup matrix of the initial approximation to the eigenvectors
        # (cannot be sparse array).
        Xf64 = rnd.random((n, m))
        Xf32 = Xf64.astype(np.float32)
        listX = [Xf64, Xf32]

        # Require that the returned eigenvectors be in the orthogonal complement
        # of the first few standard basis vectors (cannot be sparse array).
        m_excluded = 3
        Yf64 = np.eye(n, m_excluded, dtype=float)
        Yf32 = np.eye(n, m_excluded, dtype=np.float32)
        listY = [Yf64, Yf32]

        tests = list(itertools.product(listA, listB, listM, listX, listY))
        # This is one of the slower tests because there are >1,000 configs
        # to test here, instead of checking product of all input, output types
        # test each configuration for the first sparse format, and then
        # for one additional sparse format. this takes 2/7=30% as long as
        # testing all configurations for all sparse formats.
        if s_f_i > 0:
            tests = tests[s_f_i - 1::sparse_formats-1]

        for A, B, M, X, Y in tests:
            eigvals, _ = lobpcg(A, X, B=B, M=M, Y=Y, tol=1e-4,
                                maxiter=100, largest=False)
            assert_allclose(eigvals,
                            np.arange(1 + m_excluded, 1 + m_excluded + m),
                            atol=1e-5)
