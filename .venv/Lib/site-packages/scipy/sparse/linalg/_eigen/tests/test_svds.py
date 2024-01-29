import os
import re
import copy
import numpy as np

from numpy.testing import assert_allclose, assert_equal, assert_array_equal
import pytest

from scipy.linalg import svd, null_space
from scipy.sparse import csc_matrix, issparse, spdiags, random
from scipy.sparse.linalg import LinearOperator, aslinearoperator
if os.environ.get("SCIPY_USE_PROPACK"):
    has_propack = True
else:
    has_propack = False
from scipy.sparse.linalg import svds
from scipy.sparse.linalg._eigen.arpack import ArpackNoConvergence


# --- Helper Functions / Classes ---


def sorted_svd(m, k, which='LM'):
    # Compute svd of a dense matrix m, and return singular vectors/values
    # sorted.
    if issparse(m):
        m = m.toarray()
    u, s, vh = svd(m)
    if which == 'LM':
        ii = np.argsort(s)[-k:]
    elif which == 'SM':
        ii = np.argsort(s)[:k]
    else:
        raise ValueError(f"unknown which={which!r}")

    return u[:, ii], s[ii], vh[ii]


def _check_svds(A, k, u, s, vh, which="LM", check_usvh_A=False,
                check_svd=True, atol=1e-10, rtol=1e-7):
    n, m = A.shape

    # Check shapes.
    assert_equal(u.shape, (n, k))
    assert_equal(s.shape, (k,))
    assert_equal(vh.shape, (k, m))

    # Check that the original matrix can be reconstituted.
    A_rebuilt = (u*s).dot(vh)
    assert_equal(A_rebuilt.shape, A.shape)
    if check_usvh_A:
        assert_allclose(A_rebuilt, A, atol=atol, rtol=rtol)

    # Check that u is a semi-orthogonal matrix.
    uh_u = np.dot(u.T.conj(), u)
    assert_equal(uh_u.shape, (k, k))
    assert_allclose(uh_u, np.identity(k), atol=atol, rtol=rtol)

    # Check that vh is a semi-orthogonal matrix.
    vh_v = np.dot(vh, vh.T.conj())
    assert_equal(vh_v.shape, (k, k))
    assert_allclose(vh_v, np.identity(k), atol=atol, rtol=rtol)

    # Check that scipy.sparse.linalg.svds ~ scipy.linalg.svd
    if check_svd:
        u2, s2, vh2 = sorted_svd(A, k, which)
        assert_allclose(np.abs(u), np.abs(u2), atol=atol, rtol=rtol)
        assert_allclose(s, s2, atol=atol, rtol=rtol)
        assert_allclose(np.abs(vh), np.abs(vh2), atol=atol, rtol=rtol)


def _check_svds_n(A, k, u, s, vh, which="LM", check_res=True,
                  check_svd=True, atol=1e-10, rtol=1e-7):
    n, m = A.shape

    # Check shapes.
    assert_equal(u.shape, (n, k))
    assert_equal(s.shape, (k,))
    assert_equal(vh.shape, (k, m))

    # Check that u is a semi-orthogonal matrix.
    uh_u = np.dot(u.T.conj(), u)
    assert_equal(uh_u.shape, (k, k))
    error = np.sum(np.abs(uh_u - np.identity(k))) / (k * k)
    assert_allclose(error, 0.0, atol=atol, rtol=rtol)

    # Check that vh is a semi-orthogonal matrix.
    vh_v = np.dot(vh, vh.T.conj())
    assert_equal(vh_v.shape, (k, k))
    error = np.sum(np.abs(vh_v - np.identity(k))) / (k * k)
    assert_allclose(error, 0.0, atol=atol, rtol=rtol)

    # Check residuals
    if check_res:
        ru = A.T.conj() @ u - vh.T.conj() * s
        rus = np.sum(np.abs(ru)) / (n * k)
        rvh = A @ vh.T.conj() - u * s
        rvhs = np.sum(np.abs(rvh)) / (m * k)
        assert_allclose(rus, 0.0, atol=atol, rtol=rtol)
        assert_allclose(rvhs, 0.0, atol=atol, rtol=rtol)

    # Check that scipy.sparse.linalg.svds ~ scipy.linalg.svd
    if check_svd:
        u2, s2, vh2 = sorted_svd(A, k, which)
        assert_allclose(s, s2, atol=atol, rtol=rtol)
        A_rebuilt_svd = (u2*s2).dot(vh2)
        A_rebuilt = (u*s).dot(vh)
        assert_equal(A_rebuilt.shape, A.shape)
        error = np.sum(np.abs(A_rebuilt_svd - A_rebuilt)) / (k * k)
        assert_allclose(error, 0.0, atol=atol, rtol=rtol)


class CheckingLinearOperator(LinearOperator):
    def __init__(self, A):
        self.A = A
        self.dtype = A.dtype
        self.shape = A.shape

    def _matvec(self, x):
        assert_equal(max(x.shape), np.size(x))
        return self.A.dot(x)

    def _rmatvec(self, x):
        assert_equal(max(x.shape), np.size(x))
        return self.A.T.conjugate().dot(x)


# --- Test Input Validation ---
# Tests input validation on parameters `k` and `which`.
# Needs better input validation checks for all other parameters.

class SVDSCommonTests:

    solver = None

    # some of these IV tests could run only once, say with solver=None

    _A_empty_msg = "`A` must not be empty."
    _A_dtype_msg = "`A` must be of floating or complex floating data type"
    _A_type_msg = "type not understood"
    _A_ndim_msg = "array must have ndim <= 2"
    _A_validation_inputs = [
        (np.asarray([[]]), ValueError, _A_empty_msg),
        (np.asarray([[1, 2], [3, 4]]), ValueError, _A_dtype_msg),
        ("hi", TypeError, _A_type_msg),
        (np.asarray([[[1., 2.], [3., 4.]]]), ValueError, _A_ndim_msg)]

    @pytest.mark.parametrize("args", _A_validation_inputs)
    def test_svds_input_validation_A(self, args):
        A, error_type, message = args
        with pytest.raises(error_type, match=message):
            svds(A, k=1, solver=self.solver)

    @pytest.mark.parametrize("k", [-1, 0, 3, 4, 5, 1.5, "1"])
    def test_svds_input_validation_k_1(self, k):
        rng = np.random.default_rng(0)
        A = rng.random((4, 3))

        # propack can do complete SVD
        if self.solver == 'propack' and k == 3:
            if not has_propack:
                pytest.skip("PROPACK not enabled")
            res = svds(A, k=k, solver=self.solver)
            _check_svds(A, k, *res, check_usvh_A=True, check_svd=True)
            return

        message = ("`k` must be an integer satisfying")
        with pytest.raises(ValueError, match=message):
            svds(A, k=k, solver=self.solver)

    def test_svds_input_validation_k_2(self):
        # I think the stack trace is reasonable when `k` can't be converted
        # to an int.
        message = "int() argument must be a"
        with pytest.raises(TypeError, match=re.escape(message)):
            svds(np.eye(10), k=[], solver=self.solver)

        message = "invalid literal for int()"
        with pytest.raises(ValueError, match=message):
            svds(np.eye(10), k="hi", solver=self.solver)

    @pytest.mark.parametrize("tol", (-1, np.inf, np.nan))
    def test_svds_input_validation_tol_1(self, tol):
        message = "`tol` must be a non-negative floating point value."
        with pytest.raises(ValueError, match=message):
            svds(np.eye(10), tol=tol, solver=self.solver)

    @pytest.mark.parametrize("tol", ([], 'hi'))
    def test_svds_input_validation_tol_2(self, tol):
        # I think the stack trace is reasonable here
        message = "'<' not supported between instances"
        with pytest.raises(TypeError, match=message):
            svds(np.eye(10), tol=tol, solver=self.solver)

    @pytest.mark.parametrize("which", ('LA', 'SA', 'ekki', 0))
    def test_svds_input_validation_which(self, which):
        # Regression test for a github issue.
        # https://github.com/scipy/scipy/issues/4590
        # Function was not checking for eigenvalue type and unintended
        # values could be returned.
        with pytest.raises(ValueError, match="`which` must be in"):
            svds(np.eye(10), which=which, solver=self.solver)

    @pytest.mark.parametrize("transpose", (True, False))
    @pytest.mark.parametrize("n", range(4, 9))
    def test_svds_input_validation_v0_1(self, transpose, n):
        rng = np.random.default_rng(0)
        A = rng.random((5, 7))
        v0 = rng.random(n)
        if transpose:
            A = A.T
        k = 2
        message = "`v0` must have shape"

        required_length = (A.shape[0] if self.solver == 'propack'
                           else min(A.shape))
        if n != required_length:
            with pytest.raises(ValueError, match=message):
                svds(A, k=k, v0=v0, solver=self.solver)

    def test_svds_input_validation_v0_2(self):
        A = np.ones((10, 10))
        v0 = np.ones((1, 10))
        message = "`v0` must have shape"
        with pytest.raises(ValueError, match=message):
            svds(A, k=1, v0=v0, solver=self.solver)

    @pytest.mark.parametrize("v0", ("hi", 1, np.ones(10, dtype=int)))
    def test_svds_input_validation_v0_3(self, v0):
        A = np.ones((10, 10))
        message = "`v0` must be of floating or complex floating data type."
        with pytest.raises(ValueError, match=message):
            svds(A, k=1, v0=v0, solver=self.solver)

    @pytest.mark.parametrize("maxiter", (-1, 0, 5.5))
    def test_svds_input_validation_maxiter_1(self, maxiter):
        message = ("`maxiter` must be a positive integer.")
        with pytest.raises(ValueError, match=message):
            svds(np.eye(10), maxiter=maxiter, solver=self.solver)

    def test_svds_input_validation_maxiter_2(self):
        # I think the stack trace is reasonable when `k` can't be converted
        # to an int.
        message = "int() argument must be a"
        with pytest.raises(TypeError, match=re.escape(message)):
            svds(np.eye(10), maxiter=[], solver=self.solver)

        message = "invalid literal for int()"
        with pytest.raises(ValueError, match=message):
            svds(np.eye(10), maxiter="hi", solver=self.solver)

    @pytest.mark.parametrize("rsv", ('ekki', 10))
    def test_svds_input_validation_return_singular_vectors(self, rsv):
        message = "`return_singular_vectors` must be in"
        with pytest.raises(ValueError, match=message):
            svds(np.eye(10), return_singular_vectors=rsv, solver=self.solver)

    # --- Test Parameters ---

    @pytest.mark.parametrize("k", [3, 5])
    @pytest.mark.parametrize("which", ["LM", "SM"])
    def test_svds_parameter_k_which(self, k, which):
        if self.solver == 'propack':
            if not has_propack:
                pytest.skip("PROPACK not available")
        # check that the `k` parameter sets the number of eigenvalues/
        # eigenvectors returned.
        # Also check that the `which` parameter sets whether the largest or
        # smallest eigenvalues are returned
        rng = np.random.default_rng(0)
        A = rng.random((10, 10))
        if self.solver == 'lobpcg':
            with pytest.warns(UserWarning, match="The problem size"):
                res = svds(A, k=k, which=which, solver=self.solver,
                           random_state=0)
        else:
            res = svds(A, k=k, which=which, solver=self.solver,
                       random_state=0)
        _check_svds(A, k, *res, which=which, atol=8e-10)

    # loop instead of parametrize for simplicity
    def test_svds_parameter_tol(self):
        if self.solver == 'propack':
            if not has_propack:
                pytest.skip("PROPACK not available")
        return  # TODO: needs work, disabling for now
        # check the effect of the `tol` parameter on solver accuracy by solving
        # the same problem with varying `tol` and comparing the eigenvalues
        # against ground truth computed
        n = 100  # matrix size
        k = 3    # number of eigenvalues to check

        # generate a random, sparse-ish matrix
        # effect isn't apparent for matrices that are too small
        rng = np.random.default_rng(0)
        A = rng.random((n, n))
        A[A > .1] = 0
        A = A @ A.T

        _, s, _ = svd(A)  # calculate ground truth

        # calculate the error as a function of `tol`
        A = csc_matrix(A)

        def err(tol):
            if self.solver == 'lobpcg' and tol == 1e-4:
                with pytest.warns(UserWarning, match="Exited at iteration"):
                    _, s2, _ = svds(A, k=k, v0=np.ones(n),
                                    solver=self.solver, tol=tol)
            else:
                _, s2, _ = svds(A, k=k, v0=np.ones(n),
                                solver=self.solver, tol=tol)
            return np.linalg.norm((s2 - s[k-1::-1])/s[k-1::-1])

        tols = [1e-4, 1e-2, 1e0]  # tolerance levels to check
        # for 'arpack' and 'propack', accuracies make discrete steps
        accuracies = {'propack': [1e-12, 1e-6, 1e-4],
                      'arpack': [2e-15, 1e-10, 1e-10],
                      'lobpcg': [1e-11, 1e-3, 10]}

        for tol, accuracy in zip(tols, accuracies[self.solver]):
            error = err(tol)
            assert error < accuracy
            assert error > accuracy/10

    def test_svd_v0(self):
        if self.solver == 'propack':
            if not has_propack:
                pytest.skip("PROPACK not available")
        # check that the `v0` parameter affects the solution
        n = 100
        k = 1
        # If k != 1, LOBPCG needs more initial vectors, which are generated
        # with random_state, so it does not pass w/ k >= 2.
        # For some other values of `n`, the AssertionErrors are not raised
        # with different v0s, which is reasonable.

        rng = np.random.default_rng(0)
        A = rng.random((n, n))

        # with the same v0, solutions are the same, and they are accurate
        # v0 takes precedence over random_state
        v0a = rng.random(n)
        res1a = svds(A, k, v0=v0a, solver=self.solver, random_state=0)
        res2a = svds(A, k, v0=v0a, solver=self.solver, random_state=1)
        for idx in range(3):
            assert_allclose(res1a[idx], res2a[idx], rtol=1e-15, atol=2e-16)
        _check_svds(A, k, *res1a)

        # with the same v0, solutions are the same, and they are accurate
        v0b = rng.random(n)
        res1b = svds(A, k, v0=v0b, solver=self.solver, random_state=2)
        res2b = svds(A, k, v0=v0b, solver=self.solver, random_state=3)
        for idx in range(3):
            assert_allclose(res1b[idx], res2b[idx], rtol=1e-15, atol=2e-16)
        _check_svds(A, k, *res1b)

        # with different v0, solutions can be numerically different
        message = "Arrays are not equal"
        with pytest.raises(AssertionError, match=message):
            assert_equal(res1a, res1b)

    def test_svd_random_state(self):
        if self.solver == 'propack':
            if not has_propack:
                pytest.skip("PROPACK not available")
        # check that the `random_state` parameter affects the solution
        # Admittedly, `n` and `k` are chosen so that all solver pass all
        # these checks. That's a tall order, since LOBPCG doesn't want to
        # achieve the desired accuracy and ARPACK often returns the same
        # singular values/vectors for different v0.
        n = 100
        k = 1

        rng = np.random.default_rng(0)
        A = rng.random((n, n))

        # with the same random_state, solutions are the same and accurate
        res1a = svds(A, k, solver=self.solver, random_state=0)
        res2a = svds(A, k, solver=self.solver, random_state=0)
        for idx in range(3):
            assert_allclose(res1a[idx], res2a[idx], rtol=1e-15, atol=2e-16)
        _check_svds(A, k, *res1a)

        # with the same random_state, solutions are the same and accurate
        res1b = svds(A, k, solver=self.solver, random_state=1)
        res2b = svds(A, k, solver=self.solver, random_state=1)
        for idx in range(3):
            assert_allclose(res1b[idx], res2b[idx], rtol=1e-15, atol=2e-16)
        _check_svds(A, k, *res1b)

        # with different random_state, solutions can be numerically different
        message = "Arrays are not equal"
        with pytest.raises(AssertionError, match=message):
            assert_equal(res1a, res1b)

    @pytest.mark.parametrize("random_state", (0, 1,
                                              np.random.RandomState(0),
                                              np.random.default_rng(0)))
    def test_svd_random_state_2(self, random_state):
        if self.solver == 'propack':
            if not has_propack:
                pytest.skip("PROPACK not available")

        n = 100
        k = 1

        rng = np.random.default_rng(0)
        A = rng.random((n, n))

        random_state_2 = copy.deepcopy(random_state)

        # with the same random_state, solutions are the same and accurate
        res1a = svds(A, k, solver=self.solver, random_state=random_state)
        res2a = svds(A, k, solver=self.solver, random_state=random_state_2)
        for idx in range(3):
            assert_allclose(res1a[idx], res2a[idx], rtol=1e-15, atol=2e-16)
        _check_svds(A, k, *res1a)

    @pytest.mark.parametrize("random_state", (None,
                                              np.random.RandomState(0),
                                              np.random.default_rng(0)))
    def test_svd_random_state_3(self, random_state):
        if self.solver == 'propack':
            if not has_propack:
                pytest.skip("PROPACK not available")

        n = 100
        k = 5

        rng = np.random.default_rng(0)
        A = rng.random((n, n))

        # random_state in different state produces accurate - but not
        # not necessarily identical - results
        res1a = svds(A, k, solver=self.solver, random_state=random_state)
        res2a = svds(A, k, solver=self.solver, random_state=random_state)
        _check_svds(A, k, *res1a, atol=2e-10, rtol=1e-6)
        _check_svds(A, k, *res2a, atol=2e-10, rtol=1e-6)

        message = "Arrays are not equal"
        with pytest.raises(AssertionError, match=message):
            assert_equal(res1a, res2a)

    @pytest.mark.filterwarnings("ignore:Exited postprocessing")
    def test_svd_maxiter(self):
        # check that maxiter works as expected: should not return accurate
        # solution after 1 iteration, but should with default `maxiter`
        if self.solver == 'propack':
            if not has_propack:
                pytest.skip("PROPACK not available")
        A = np.diag(np.arange(9)).astype(np.float64)
        k = 1
        u, s, vh = sorted_svd(A, k)

        if self.solver == 'arpack':
            message = "ARPACK error -1: No convergence"
            with pytest.raises(ArpackNoConvergence, match=message):
                svds(A, k, ncv=3, maxiter=1, solver=self.solver)
        elif self.solver == 'lobpcg':
            with pytest.warns(UserWarning, match="Exited at iteration"):
                svds(A, k, maxiter=1, solver=self.solver)
        elif self.solver == 'propack':
            message = "k=1 singular triplets did not converge within"
            with pytest.raises(np.linalg.LinAlgError, match=message):
                svds(A, k, maxiter=1, solver=self.solver)

        ud, sd, vhd = svds(A, k, solver=self.solver)  # default maxiter
        _check_svds(A, k, ud, sd, vhd, atol=1e-8)
        assert_allclose(np.abs(ud), np.abs(u), atol=1e-8)
        assert_allclose(np.abs(vhd), np.abs(vh), atol=1e-8)
        assert_allclose(np.abs(sd), np.abs(s), atol=1e-9)

    @pytest.mark.parametrize("rsv", (True, False, 'u', 'vh'))
    @pytest.mark.parametrize("shape", ((5, 7), (6, 6), (7, 5)))
    def test_svd_return_singular_vectors(self, rsv, shape):
        # check that the return_singular_vectors parameter works as expected
        if self.solver == 'propack':
            if not has_propack:
                pytest.skip("PROPACK not available")

        rng = np.random.default_rng(0)
        A = rng.random(shape)
        k = 2
        M, N = shape
        u, s, vh = sorted_svd(A, k)

        respect_u = True if self.solver == 'propack' else M <= N
        respect_vh = True if self.solver == 'propack' else M > N

        if self.solver == 'lobpcg':
            with pytest.warns(UserWarning, match="The problem size"):
                if rsv is False:
                    s2 = svds(A, k, return_singular_vectors=rsv,
                              solver=self.solver, random_state=rng)
                    assert_allclose(s2, s)
                elif rsv == 'u' and respect_u:
                    u2, s2, vh2 = svds(A, k, return_singular_vectors=rsv,
                                       solver=self.solver, random_state=rng)
                    assert_allclose(np.abs(u2), np.abs(u))
                    assert_allclose(s2, s)
                    assert vh2 is None
                elif rsv == 'vh' and respect_vh:
                    u2, s2, vh2 = svds(A, k, return_singular_vectors=rsv,
                                       solver=self.solver, random_state=rng)
                    assert u2 is None
                    assert_allclose(s2, s)
                    assert_allclose(np.abs(vh2), np.abs(vh))
                else:
                    u2, s2, vh2 = svds(A, k, return_singular_vectors=rsv,
                                       solver=self.solver, random_state=rng)
                    if u2 is not None:
                        assert_allclose(np.abs(u2), np.abs(u))
                    assert_allclose(s2, s)
                    if vh2 is not None:
                        assert_allclose(np.abs(vh2), np.abs(vh))
        else:
            if rsv is False:
                s2 = svds(A, k, return_singular_vectors=rsv,
                          solver=self.solver, random_state=rng)
                assert_allclose(s2, s)
            elif rsv == 'u' and respect_u:
                u2, s2, vh2 = svds(A, k, return_singular_vectors=rsv,
                                   solver=self.solver, random_state=rng)
                assert_allclose(np.abs(u2), np.abs(u))
                assert_allclose(s2, s)
                assert vh2 is None
            elif rsv == 'vh' and respect_vh:
                u2, s2, vh2 = svds(A, k, return_singular_vectors=rsv,
                                   solver=self.solver, random_state=rng)
                assert u2 is None
                assert_allclose(s2, s)
                assert_allclose(np.abs(vh2), np.abs(vh))
            else:
                u2, s2, vh2 = svds(A, k, return_singular_vectors=rsv,
                                   solver=self.solver, random_state=rng)
                if u2 is not None:
                    assert_allclose(np.abs(u2), np.abs(u))
                assert_allclose(s2, s)
                if vh2 is not None:
                    assert_allclose(np.abs(vh2), np.abs(vh))

    # --- Test Basic Functionality ---
    # Tests the accuracy of each solver for real and complex matrices provided
    # as list, dense array, sparse matrix, and LinearOperator.

    A1 = [[1, 2, 3], [3, 4, 3], [1 + 1j, 0, 2], [0, 0, 1]]
    A2 = [[1, 2, 3, 8 + 5j], [3 - 2j, 4, 3, 5], [1, 0, 2, 3], [0, 0, 1, 0]]

    @pytest.mark.filterwarnings("ignore:k >= N - 1",
                                reason="needed to demonstrate #16725")
    @pytest.mark.parametrize('A', (A1, A2))
    @pytest.mark.parametrize('k', range(1, 5))
    # PROPACK fails a lot if @pytest.mark.parametrize('which', ("SM", "LM"))
    @pytest.mark.parametrize('real', (True, False))
    @pytest.mark.parametrize('transpose', (False, True))
    # In gh-14299, it was suggested the `svds` should _not_ work with lists
    @pytest.mark.parametrize('lo_type', (np.asarray, csc_matrix,
                                         aslinearoperator))
    def test_svd_simple(self, A, k, real, transpose, lo_type):

        if self.solver == 'propack':
            if not has_propack:
                pytest.skip("PROPACK not available")

        A = np.asarray(A)
        A = np.real(A) if real else A
        A = A.T if transpose else A
        A2 = lo_type(A)

        # could check for the appropriate errors, but that is tested above
        if k > min(A.shape):
            pytest.skip("`k` cannot be greater than `min(A.shape)`")
        if self.solver != 'propack' and k >= min(A.shape):
            pytest.skip("Only PROPACK supports complete SVD")
        if self.solver == 'arpack' and not real and k == min(A.shape) - 1:
            pytest.skip("#16725")

        if self.solver == 'propack' and (np.intp(0).itemsize < 8 and not real):
            pytest.skip('PROPACK complex-valued SVD methods not available '
                        'for 32-bit builds')

        if self.solver == 'lobpcg':
            with pytest.warns(UserWarning, match="The problem size"):
                u, s, vh = svds(A2, k, solver=self.solver)
        else:
            u, s, vh = svds(A2, k, solver=self.solver)
        _check_svds(A, k, u, s, vh, atol=3e-10)

    def test_svd_linop(self):
        solver = self.solver
        if self.solver == 'propack':
            if not has_propack:
                pytest.skip("PROPACK not available")

        nmks = [(6, 7, 3),
                (9, 5, 4),
                (10, 8, 5)]

        def reorder(args):
            U, s, VH = args
            j = np.argsort(s)
            return U[:, j], s[j], VH[j, :]

        for n, m, k in nmks:
            # Test svds on a LinearOperator.
            A = np.random.RandomState(52).randn(n, m)
            L = CheckingLinearOperator(A)

            if solver == 'propack':
                v0 = np.ones(n)
            else:
                v0 = np.ones(min(A.shape))
            if solver == 'lobpcg':
                with pytest.warns(UserWarning, match="The problem size"):
                    U1, s1, VH1 = reorder(svds(A, k, v0=v0, solver=solver))
                    U2, s2, VH2 = reorder(svds(L, k, v0=v0, solver=solver))
            else:
                U1, s1, VH1 = reorder(svds(A, k, v0=v0, solver=solver))
                U2, s2, VH2 = reorder(svds(L, k, v0=v0, solver=solver))

            assert_allclose(np.abs(U1), np.abs(U2))
            assert_allclose(s1, s2)
            assert_allclose(np.abs(VH1), np.abs(VH2))
            assert_allclose(np.dot(U1, np.dot(np.diag(s1), VH1)),
                            np.dot(U2, np.dot(np.diag(s2), VH2)))

            # Try again with which="SM".
            A = np.random.RandomState(1909).randn(n, m)
            L = CheckingLinearOperator(A)

            # TODO: arpack crashes when v0=v0, which="SM"
            kwargs = {'v0': v0} if solver not in {None, 'arpack'} else {}
            if self.solver == 'lobpcg':
                with pytest.warns(UserWarning, match="The problem size"):
                    U1, s1, VH1 = reorder(svds(A, k, which="SM", solver=solver,
                                               **kwargs))
                    U2, s2, VH2 = reorder(svds(L, k, which="SM", solver=solver,
                                               **kwargs))
            else:
                U1, s1, VH1 = reorder(svds(A, k, which="SM", solver=solver,
                                           **kwargs))
                U2, s2, VH2 = reorder(svds(L, k, which="SM", solver=solver,
                                           **kwargs))

            assert_allclose(np.abs(U1), np.abs(U2))
            assert_allclose(s1 + 1, s2 + 1)
            assert_allclose(np.abs(VH1), np.abs(VH2))
            assert_allclose(np.dot(U1, np.dot(np.diag(s1), VH1)),
                            np.dot(U2, np.dot(np.diag(s2), VH2)))

            if k < min(n, m) - 1:
                # Complex input and explicit which="LM".
                for (dt, eps) in [(complex, 1e-7), (np.complex64, 1e-3)]:
                    if self.solver == 'propack' and np.intp(0).itemsize < 8:
                        pytest.skip('PROPACK complex-valued SVD methods '
                                    'not available for 32-bit builds')
                    rng = np.random.RandomState(1648)
                    A = (rng.randn(n, m) + 1j * rng.randn(n, m)).astype(dt)
                    L = CheckingLinearOperator(A)

                    if self.solver == 'lobpcg':
                        with pytest.warns(UserWarning,
                                          match="The problem size"):
                            U1, s1, VH1 = reorder(svds(A, k, which="LM",
                                                       solver=solver))
                            U2, s2, VH2 = reorder(svds(L, k, which="LM",
                                                       solver=solver))
                    else:
                        U1, s1, VH1 = reorder(svds(A, k, which="LM",
                                                   solver=solver))
                        U2, s2, VH2 = reorder(svds(L, k, which="LM",
                                                   solver=solver))

                    assert_allclose(np.abs(U1), np.abs(U2), rtol=eps)
                    assert_allclose(s1, s2, rtol=eps)
                    assert_allclose(np.abs(VH1), np.abs(VH2), rtol=eps)
                    assert_allclose(np.dot(U1, np.dot(np.diag(s1), VH1)),
                                    np.dot(U2, np.dot(np.diag(s2), VH2)),
                                    rtol=eps)

    SHAPES = ((100, 100), (100, 101), (101, 100))

    @pytest.mark.filterwarnings("ignore:Exited at iteration")
    @pytest.mark.filterwarnings("ignore:Exited postprocessing")
    @pytest.mark.parametrize("shape", SHAPES)
    # ARPACK supports only dtype float, complex, or np.float32
    @pytest.mark.parametrize("dtype", (float, complex, np.float32))
    def test_small_sigma_sparse(self, shape, dtype):
        # https://github.com/scipy/scipy/pull/11829
        solver = self.solver
        # 2do: PROPACK fails orthogonality of singular vectors
        # if dtype == complex and self.solver == 'propack':
        #    pytest.skip("PROPACK unsupported for complex dtype")
        if solver == 'propack':
            pytest.skip("PROPACK failures unrelated to PR")
        rng = np.random.default_rng(0)
        k = 5
        (m, n) = shape
        S = random(m, n, density=0.1, random_state=rng)
        if dtype == complex:
            S = + 1j * random(m, n, density=0.1, random_state=rng)
        e = np.ones(m)
        e[0:5] *= 1e1 ** np.arange(-5, 0, 1)
        S = spdiags(e, 0, m, m) @ S
        S = S.astype(dtype)
        u, s, vh = svds(S, k, which='SM', solver=solver, maxiter=1000)
        c_svd = False  # partial SVD can be different from full SVD
        _check_svds_n(S, k, u, s, vh, which="SM", check_svd=c_svd, atol=1e-1)

    # --- Test Edge Cases ---
    # Checks a few edge cases.

    @pytest.mark.parametrize("shape", ((6, 5), (5, 5), (5, 6)))
    @pytest.mark.parametrize("dtype", (float, complex))
    def test_svd_LM_ones_matrix(self, shape, dtype):
        # Check that svds can deal with matrix_rank less than k in LM mode.
        k = 3
        n, m = shape
        A = np.ones((n, m), dtype=dtype)

        if self.solver == 'lobpcg':
            with pytest.warns(UserWarning, match="The problem size"):
                U, s, VH = svds(A, k, solver=self.solver)
        else:
            U, s, VH = svds(A, k, solver=self.solver)

        _check_svds(A, k, U, s, VH, check_usvh_A=True, check_svd=False)

        # Check that the largest singular value is near sqrt(n*m)
        # and the other singular values have been forced to zero.
        assert_allclose(np.max(s), np.sqrt(n*m))
        s = np.array(sorted(s)[:-1]) + 1
        z = np.ones_like(s)
        assert_allclose(s, z)

    @pytest.mark.filterwarnings("ignore:k >= N - 1",
                                reason="needed to demonstrate #16725")
    @pytest.mark.parametrize("shape", ((3, 4), (4, 4), (4, 3), (4, 2)))
    @pytest.mark.parametrize("dtype", (float, complex))
    def test_zero_matrix(self, shape, dtype):
        # Check that svds can deal with matrices containing only zeros;
        # see https://github.com/scipy/scipy/issues/3452/
        # shape = (4, 2) is included because it is the particular case
        # reported in the issue
        k = 1
        n, m = shape
        A = np.zeros((n, m), dtype=dtype)

        if (self.solver == 'arpack' and dtype is complex
                and k == min(A.shape) - 1):
            pytest.skip("#16725")

        if self.solver == 'propack':
            pytest.skip("PROPACK failures unrelated to PR #16712")

        if self.solver == 'lobpcg':
            with pytest.warns(UserWarning, match="The problem size"):
                U, s, VH = svds(A, k, solver=self.solver)
        else:
            U, s, VH = svds(A, k, solver=self.solver)

        # Check some generic properties of svd.
        _check_svds(A, k, U, s, VH, check_usvh_A=True, check_svd=False)

        # Check that the singular values are zero.
        assert_array_equal(s, 0)

    @pytest.mark.parametrize("shape", ((20, 20), (20, 21), (21, 20)))
    # ARPACK supports only dtype float, complex, or np.float32
    @pytest.mark.parametrize("dtype", (float, complex, np.float32))
    def test_small_sigma(self, shape, dtype):
        if not has_propack:
            pytest.skip("PROPACK not enabled")
        # https://github.com/scipy/scipy/pull/11829
        if dtype == complex and self.solver == 'propack':
            pytest.skip("PROPACK unsupported for complex dtype")
        rng = np.random.default_rng(179847540)
        A = rng.random(shape).astype(dtype)
        u, _, vh = svd(A, full_matrices=False)
        if dtype == np.float32:
            e = 10.0
        else:
            e = 100.0
        t = e**(-np.arange(len(vh))).astype(dtype)
        A = (u*t).dot(vh)
        k = 4
        u, s, vh = svds(A, k, solver=self.solver, maxiter=100)
        t = np.sum(s > 0)
        assert_equal(t, k)
        # LOBPCG needs larger atol and rtol to pass
        _check_svds_n(A, k, u, s, vh, atol=1e-3, rtol=1e0, check_svd=False)

    # ARPACK supports only dtype float, complex, or np.float32
    @pytest.mark.filterwarnings("ignore:The problem size")
    @pytest.mark.parametrize("dtype", (float, complex, np.float32))
    def test_small_sigma2(self, dtype):
        if self.solver == 'propack':
            if not has_propack:
                pytest.skip("PROPACK not enabled")
            elif dtype == np.float32:
                pytest.skip("Test failures in CI, see gh-17004")
            elif dtype == complex:
                # https://github.com/scipy/scipy/issues/11406
                pytest.skip("PROPACK unsupported for complex dtype")

        rng = np.random.default_rng(179847540)
        # create a 10x10 singular matrix with a 4-dim null space
        dim = 4
        size = 10
        x = rng.random((size, size-dim))
        y = x[:, :dim] * rng.random(dim)
        mat = np.hstack((x, y))
        mat = mat.astype(dtype)

        nz = null_space(mat)
        assert_equal(nz.shape[1], dim)

        # Tolerances atol and rtol adjusted to pass np.float32
        # Use non-sparse svd
        u, s, vh = svd(mat)
        # Singular values are 0:
        assert_allclose(s[-dim:], 0, atol=1e-6, rtol=1e0)
        # Smallest right singular vectors in null space:
        assert_allclose(mat @ vh[-dim:, :].T, 0, atol=1e-6, rtol=1e0)

        # Smallest singular values should be 0
        sp_mat = csc_matrix(mat)
        su, ss, svh = svds(sp_mat, k=dim, which='SM', solver=self.solver)
        # Smallest dim singular values are 0:
        assert_allclose(ss, 0, atol=1e-5, rtol=1e0)
        # Smallest singular vectors via svds in null space:
        n, m = mat.shape
        if n < m:  # else the assert fails with some libraries unclear why
            assert_allclose(sp_mat.transpose() @ su, 0, atol=1e-5, rtol=1e0)
        assert_allclose(sp_mat @ svh.T, 0, atol=1e-5, rtol=1e0)

# --- Perform tests with each solver ---


class Test_SVDS_once:
    @pytest.mark.parametrize("solver", ['ekki', object])
    def test_svds_input_validation_solver(self, solver):
        message = "solver must be one of"
        with pytest.raises(ValueError, match=message):
            svds(np.ones((3, 4)), k=2, solver=solver)


class Test_SVDS_ARPACK(SVDSCommonTests):

    def setup_method(self):
        self.solver = 'arpack'

    @pytest.mark.parametrize("ncv", list(range(-1, 8)) + [4.5, "5"])
    def test_svds_input_validation_ncv_1(self, ncv):
        rng = np.random.default_rng(0)
        A = rng.random((6, 7))
        k = 3
        if ncv in {4, 5}:
            u, s, vh = svds(A, k=k, ncv=ncv, solver=self.solver)
        # partial decomposition, so don't check that u@diag(s)@vh=A;
        # do check that scipy.sparse.linalg.svds ~ scipy.linalg.svd
            _check_svds(A, k, u, s, vh)
        else:
            message = ("`ncv` must be an integer satisfying")
            with pytest.raises(ValueError, match=message):
                svds(A, k=k, ncv=ncv, solver=self.solver)

    def test_svds_input_validation_ncv_2(self):
        # I think the stack trace is reasonable when `ncv` can't be converted
        # to an int.
        message = "int() argument must be a"
        with pytest.raises(TypeError, match=re.escape(message)):
            svds(np.eye(10), ncv=[], solver=self.solver)

        message = "invalid literal for int()"
        with pytest.raises(ValueError, match=message):
            svds(np.eye(10), ncv="hi", solver=self.solver)

    # I can't see a robust relationship between `ncv` and relevant outputs
    # (e.g. accuracy, time), so no test of the parameter.


class Test_SVDS_LOBPCG(SVDSCommonTests):

    def setup_method(self):
        self.solver = 'lobpcg'

    def test_svd_random_state_3(self):
        pytest.xfail("LOBPCG is having trouble with accuracy.")


class Test_SVDS_PROPACK(SVDSCommonTests):

    def setup_method(self):
        self.solver = 'propack'

    def test_svd_LM_ones_matrix(self):
        message = ("PROPACK does not return orthonormal singular vectors "
                   "associated with zero singular values.")
        # There are some other issues with this matrix of all ones, e.g.
        # `which='sm'` and `k=1` returns the largest singular value
        pytest.xfail(message)

    def test_svd_LM_zeros_matrix(self):
        message = ("PROPACK does not return orthonormal singular vectors "
                   "associated with zero singular values.")
        pytest.xfail(message)
