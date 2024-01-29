""" Test functions for the sparse.linalg._isolve module
"""

import itertools
import platform
import sys
import pytest

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from numpy import zeros, arange, array, ones, eye, iscomplexobj
from numpy.linalg import norm

from scipy.sparse import spdiags, csr_matrix, kronsum

from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse.linalg._isolve import (bicg, bicgstab, cg, cgs,
                                         gcrotmk, gmres, lgmres,
                                         minres, qmr, tfqmr)

# TODO check that method preserve shape and type
# TODO test both preconditioner methods


# list of all solvers under test
_SOLVERS = [bicg, bicgstab, cg, cgs, gcrotmk, gmres, lgmres,
            minres, qmr, tfqmr]

pytestmark = [
    # remove this once atol defaults to 0.0 for all methods
    pytest.mark.filterwarnings("ignore:.*called without specifying.*"),
]


# create parametrized fixture for easy reuse in tests
@pytest.fixture(params=_SOLVERS, scope="session")
def solver(request):
    """
    Fixture for all solvers in scipy.sparse.linalg._isolve
    """
    return request.param


class Case:
    def __init__(self, name, A, b=None, skip=None, nonconvergence=None):
        self.name = name
        self.A = A
        if b is None:
            self.b = arange(A.shape[0], dtype=float)
        else:
            self.b = b
        if skip is None:
            self.skip = []
        else:
            self.skip = skip
        if nonconvergence is None:
            self.nonconvergence = []
        else:
            self.nonconvergence = nonconvergence


class SingleTest:
    def __init__(self, A, b, solver, casename, convergence=True):
        self.A = A
        self.b = b
        self.solver = solver
        self.name = casename + '-' + solver.__name__
        self.convergence = convergence

    def __repr__(self):
        return f"<{self.name}>"


class IterativeParams:
    def __init__(self):
        sym_solvers = [minres, cg]
        posdef_solvers = [cg]
        real_solvers = [minres]

        # list of Cases
        self.cases = []

        # Symmetric and Positive Definite
        N = 40
        data = ones((3, N))
        data[0, :] = 2
        data[1, :] = -1
        data[2, :] = -1
        Poisson1D = spdiags(data, [0, -1, 1], N, N, format='csr')
        self.cases.append(Case("poisson1d", Poisson1D))
        # note: minres fails for single precision
        self.cases.append(Case("poisson1d-F", Poisson1D.astype('f'),
                               skip=[minres]))

        # Symmetric and Negative Definite
        self.cases.append(Case("neg-poisson1d", -Poisson1D,
                               skip=posdef_solvers))
        # note: minres fails for single precision
        self.cases.append(Case("neg-poisson1d-F", (-Poisson1D).astype('f'),
                               skip=posdef_solvers + [minres]))

        # 2-dimensional Poisson equations
        Poisson2D = kronsum(Poisson1D, Poisson1D)
        # note: minres fails for 2-d poisson problem,
        # it will be fixed in the future PR
        self.cases.append(Case("poisson2d", Poisson2D, skip=[minres]))
        # note: minres fails for single precision
        self.cases.append(Case("poisson2d-F", Poisson2D.astype('f'),
                               skip=[minres]))

        # Symmetric and Indefinite
        data = array([[6, -5, 2, 7, -1, 10, 4, -3, -8, 9]], dtype='d')
        RandDiag = spdiags(data, [0], 10, 10, format='csr')
        self.cases.append(Case("rand-diag", RandDiag, skip=posdef_solvers))
        self.cases.append(Case("rand-diag-F", RandDiag.astype('f'),
                               skip=posdef_solvers))

        # Random real-valued
        np.random.seed(1234)
        data = np.random.rand(4, 4)
        self.cases.append(Case("rand", data,
                               skip=posdef_solvers + sym_solvers))
        self.cases.append(Case("rand-F", data.astype('f'),
                               skip=posdef_solvers + sym_solvers))

        # Random symmetric real-valued
        np.random.seed(1234)
        data = np.random.rand(4, 4)
        data = data + data.T
        self.cases.append(Case("rand-sym", data, skip=posdef_solvers))
        self.cases.append(Case("rand-sym-F", data.astype('f'),
                               skip=posdef_solvers))

        # Random pos-def symmetric real
        np.random.seed(1234)
        data = np.random.rand(9, 9)
        data = np.dot(data.conj(), data.T)
        self.cases.append(Case("rand-sym-pd", data))
        # note: minres fails for single precision
        self.cases.append(Case("rand-sym-pd-F", data.astype('f'),
                               skip=[minres]))

        # Random complex-valued
        np.random.seed(1234)
        data = np.random.rand(4, 4) + 1j * np.random.rand(4, 4)
        skip_cmplx = posdef_solvers + sym_solvers + real_solvers
        self.cases.append(Case("rand-cmplx", data, skip=skip_cmplx))
        self.cases.append(Case("rand-cmplx-F", data.astype('F'),
                               skip=skip_cmplx))

        # Random hermitian complex-valued
        np.random.seed(1234)
        data = np.random.rand(4, 4) + 1j * np.random.rand(4, 4)
        data = data + data.T.conj()
        self.cases.append(Case("rand-cmplx-herm", data,
                               skip=posdef_solvers + real_solvers))
        self.cases.append(Case("rand-cmplx-herm-F", data.astype('F'),
                               skip=posdef_solvers + real_solvers))

        # Random pos-def hermitian complex-valued
        np.random.seed(1234)
        data = np.random.rand(9, 9) + 1j * np.random.rand(9, 9)
        data = np.dot(data.conj(), data.T)
        self.cases.append(Case("rand-cmplx-sym-pd", data, skip=real_solvers))
        self.cases.append(Case("rand-cmplx-sym-pd-F", data.astype('F'),
                               skip=real_solvers))

        # Non-symmetric and Positive Definite
        #
        # cgs, qmr, bicg and tfqmr fail to converge on this one
        #   -- algorithmic limitation apparently
        data = ones((2, 10))
        data[0, :] = 2
        data[1, :] = -1
        A = spdiags(data, [0, -1], 10, 10, format='csr')
        self.cases.append(Case("nonsymposdef", A,
                               skip=sym_solvers + [cgs, qmr, bicg, tfqmr]))
        self.cases.append(Case("nonsymposdef-F", A.astype('F'),
                               skip=sym_solvers + [cgs, qmr, bicg, tfqmr]))

        # Symmetric, non-pd, hitting cgs/bicg/bicgstab/qmr/tfqmr breakdown
        A = np.array([[0, 0, 0, 0, 0, 1, -1, -0, -0, -0, -0],
                      [0, 0, 0, 0, 0, 2, -0, -1, -0, -0, -0],
                      [0, 0, 0, 0, 0, 2, -0, -0, -1, -0, -0],
                      [0, 0, 0, 0, 0, 2, -0, -0, -0, -1, -0],
                      [0, 0, 0, 0, 0, 1, -0, -0, -0, -0, -1],
                      [1, 2, 2, 2, 1, 0, -0, -0, -0, -0, -0],
                      [-1, 0, 0, 0, 0, 0, -1, -0, -0, -0, -0],
                      [0, -1, 0, 0, 0, 0, -0, -1, -0, -0, -0],
                      [0, 0, -1, 0, 0, 0, -0, -0, -1, -0, -0],
                      [0, 0, 0, -1, 0, 0, -0, -0, -0, -1, -0],
                      [0, 0, 0, 0, -1, 0, -0, -0, -0, -0, -1]], dtype=float)
        b = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=float)
        assert (A == A.T).all()
        self.cases.append(Case("sym-nonpd", A, b,
                               skip=posdef_solvers,
                               nonconvergence=[cgs, bicg, bicgstab, qmr, tfqmr]
                               )
                          )

    def generate_tests(self):
        # generate test cases with skips applied
        tests = []
        for case in self.cases:
            for solver in _SOLVERS:
                if (solver in case.skip):
                    continue
                if solver in case.nonconvergence:
                    tests += [SingleTest(case.A, case.b, solver, case.name,
                                         convergence=False)]
                else:
                    tests += [SingleTest(case.A, case.b, solver, case.name)]
        return tests


cases = IterativeParams().generate_tests()


@pytest.fixture(params=cases, ids=[x.name for x in cases], scope="module")
def case(request):
    """
    Fixture for all cases in IterativeParams
    """
    return request.param


def test_maxiter(case):
    if not case.convergence:
        pytest.skip("Solver - Breakdown case, see gh-8829")
    A = case.A
    rtol = 1e-12

    b = case.b
    x0 = 0 * b

    residuals = []

    def callback(x):
        residuals.append(norm(b - case.A * x))

    x, info = case.solver(A, b, x0=x0, rtol=rtol, maxiter=1, callback=callback)

    assert len(residuals) == 1
    assert info == 1


def test_convergence(case):
    A = case.A

    if A.dtype.char in "dD":
        rtol = 1e-8
    else:
        rtol = 1e-2

    b = case.b
    x0 = 0 * b

    x, info = case.solver(A, b, x0=x0, rtol=rtol)

    assert_array_equal(x0, 0 * b)  # ensure that x0 is not overwritten
    if case.convergence:
        assert info == 0
        assert norm(A @ x - b) <= norm(b) * rtol
    else:
        assert info != 0
        assert norm(A @ x - b) <= norm(b)


def test_precond_dummy(case):
    if not case.convergence:
        pytest.skip("Solver - Breakdown case, see gh-8829")

    rtol = 1e-8

    def identity(b, which=None):
        """trivial preconditioner"""
        return b

    A = case.A

    M, N = A.shape
    # Ensure the diagonal elements of A are non-zero before calculating
    # 1.0/A.diagonal()
    diagOfA = A.diagonal()
    if np.count_nonzero(diagOfA) == len(diagOfA):
        spdiags([1.0 / diagOfA], [0], M, N)

    b = case.b
    x0 = 0 * b

    precond = LinearOperator(A.shape, identity, rmatvec=identity)

    if case.solver is qmr:
        x, info = case.solver(A, b, M1=precond, M2=precond, x0=x0, rtol=rtol)
    else:
        x, info = case.solver(A, b, M=precond, x0=x0, rtol=rtol)
    assert info == 0
    assert norm(A @ x - b) <= norm(b) * rtol

    A = aslinearoperator(A)
    A.psolve = identity
    A.rpsolve = identity

    x, info = case.solver(A, b, x0=x0, rtol=rtol)
    assert info == 0
    assert norm(A @ x - b) <= norm(b) * rtol


# Specific test for poisson1d and poisson2d cases
@pytest.mark.parametrize('case', [x for x in IterativeParams().cases
                                  if x.name in ('poisson1d', 'poisson2d')],
                         ids=['poisson1d', 'poisson2d'])
def test_precond_inverse(case):
    for solver in _SOLVERS:
        if solver in case.skip or solver is qmr:
            continue

        rtol = 1e-8

        def inverse(b, which=None):
            """inverse preconditioner"""
            A = case.A
            if not isinstance(A, np.ndarray):
                A = A.toarray()
            return np.linalg.solve(A, b)

        def rinverse(b, which=None):
            """inverse preconditioner"""
            A = case.A
            if not isinstance(A, np.ndarray):
                A = A.toarray()
            return np.linalg.solve(A.T, b)

        matvec_count = [0]

        def matvec(b):
            matvec_count[0] += 1
            return case.A @ b

        def rmatvec(b):
            matvec_count[0] += 1
            return case.A.T @ b

        b = case.b
        x0 = 0 * b

        A = LinearOperator(case.A.shape, matvec, rmatvec=rmatvec)
        precond = LinearOperator(case.A.shape, inverse, rmatvec=rinverse)

        # Solve with preconditioner
        matvec_count = [0]
        x, info = solver(A, b, M=precond, x0=x0, rtol=rtol)

        assert info == 0
        assert norm(case.A @ x - b) <= norm(b) * rtol

        # Solution should be nearly instant
        assert matvec_count[0] <= 3


def test_atol(solver):
    # TODO: minres / tfqmr. It didn't historically use absolute tolerances, so
    # fixing it is less urgent.
    if solver in (minres, tfqmr):
        pytest.skip("TODO: Add atol to minres/tfqmr")

    # Historically this is tested as below, all pass but for some reason
    # gcrotmk is over-sensitive to difference between random.seed/rng.random
    # Hence tol lower bound is changed from -10 to -9
    # np.random.seed(1234)
    # A = np.random.rand(10, 10)
    # A = A @ A.T + 10 * np.eye(10)
    # b = 1e3*np.random.rand(10)

    rng = np.random.default_rng(168441431005389)
    A = rng.uniform(size=[10, 10])
    A = A @ A.T + 10*np.eye(10)
    b = 1e3 * rng.uniform(size=10)

    b_norm = np.linalg.norm(b)

    tols = np.r_[0, np.logspace(-9, 2, 7), np.inf]

    # Check effect of badly scaled preconditioners
    M0 = rng.standard_normal(size=(10, 10))
    M0 = M0 @ M0.T
    Ms = [None, 1e-6 * M0, 1e6 * M0]

    for M, rtol, atol in itertools.product(Ms, tols, tols):
        if rtol == 0 and atol == 0:
            continue

        if solver is qmr:
            if M is not None:
                M = aslinearoperator(M)
                M2 = aslinearoperator(np.eye(10))
            else:
                M2 = None
            x, info = solver(A, b, M1=M, M2=M2, rtol=rtol, atol=atol)
        else:
            x, info = solver(A, b, M=M, rtol=rtol, atol=atol)

        assert info == 0
        residual = A @ x - b
        err = np.linalg.norm(residual)
        atol2 = rtol * b_norm
        # Added 1.00025 fudge factor because of `err` exceeding `atol` just
        # very slightly on s390x (see gh-17839)
        assert err <= 1.00025 * max(atol, atol2)


def test_zero_rhs(solver):
    rng = np.random.default_rng(1684414984100503)
    A = rng.random(size=[10, 10])
    A = A @ A.T + 10 * np.eye(10)

    b = np.zeros(10)
    tols = np.r_[np.logspace(-10, 2, 7)]

    for tol in tols:
        x, info = solver(A, b, rtol=tol)
        assert info == 0
        assert_allclose(x, 0., atol=1e-15)

        x, info = solver(A, b, rtol=tol, x0=ones(10))
        assert info == 0
        assert_allclose(x, 0., atol=tol)

        if solver is not minres:
            x, info = solver(A, b, rtol=tol, atol=0, x0=ones(10))
            if info == 0:
                assert_allclose(x, 0)

            x, info = solver(A, b, rtol=tol, atol=tol)
            assert info == 0
            assert_allclose(x, 0, atol=1e-300)

            x, info = solver(A, b, rtol=tol, atol=0)
            assert info == 0
            assert_allclose(x, 0, atol=1e-300)


@pytest.mark.xfail(reason="see gh-18697")
def test_maxiter_worsening(solver):
    if solver not in (gmres, lgmres, qmr):
        # these were skipped from the very beginning, see gh-9201; gh-14160
        pytest.skip("Solver breakdown case")
    # Check error does not grow (boundlessly) with increasing maxiter.
    # This can occur due to the solvers hitting close to breakdown,
    # which they should detect and halt as necessary.
    # cf. gh-9100
    if (solver is gmres and platform.machine() == 'aarch64'
            and sys.version_info[1] == 9):
        pytest.xfail(reason="gh-13019")
    if (solver is lgmres and
            platform.machine() not in ['x86_64' 'x86', 'aarch64', 'arm64']):
        # see gh-17839
        pytest.xfail(reason="fails on at least ppc64le, ppc64 and riscv64")

    # Singular matrix, rhs numerically not in range
    A = np.array([[-0.1112795288033378, 0, 0, 0.16127952880333685],
                  [0, -0.13627952880333782 + 6.283185307179586j, 0, 0],
                  [0, 0, -0.13627952880333782 - 6.283185307179586j, 0],
                  [0.1112795288033368, 0j, 0j, -0.16127952880333785]])
    v = np.ones(4)
    best_error = np.inf

    # Unable to match the Fortran code tolerance levels with this example
    # Original tolerance values

    # slack_tol = 7 if platform.machine() == 'aarch64' else 5
    slack_tol = 9

    for maxiter in range(1, 20):
        x, info = solver(A, v, maxiter=maxiter, rtol=1e-8, atol=0)

        if info == 0:
            assert norm(A @ x - v) <= 1e-8 * norm(v)

        error = np.linalg.norm(A @ x - v)
        best_error = min(best_error, error)

        # Check with slack
        assert error <= slack_tol * best_error


def test_x0_working(solver):
    # Easy problem
    rng = np.random.default_rng(1685363802304750)
    n = 10
    A = rng.random(size=[n, n])
    A = A @ A.T
    b = rng.random(n)
    x0 = rng.random(n)

    if solver is minres:
        kw = dict(rtol=1e-6)
    else:
        kw = dict(atol=0, rtol=1e-6)

    x, info = solver(A, b, **kw)
    assert info == 0
    assert norm(A @ x - b) <= 1e-6 * norm(b)

    x, info = solver(A, b, x0=x0, **kw)
    assert info == 0
    assert norm(A @ x - b) <= 2e-6*norm(b)


def test_x0_equals_Mb(case):
    if case.solver is tfqmr:
        pytest.skip("Solver does not support x0='Mb'")
    A = case.A
    b = case.b
    x0 = 'Mb'
    rtol = 1e-8
    x, info = case.solver(A, b, x0=x0, rtol=rtol)

    assert_array_equal(x0, 'Mb')  # ensure that x0 is not overwritten
    assert info == 0
    assert norm(A @ x - b) <= rtol * norm(b)


# Specific tfqmr test
@pytest.mark.parametrize('case', IterativeParams().cases)
def test_show(case, capsys):
    def cb(x):
        pass

    x, info = tfqmr(case.A, case.b, callback=cb, show=True)
    out, err = capsys.readouterr()

    if case.name == "sym-nonpd":
        # no logs for some reason
        exp = ""
    elif case.name in ("nonsymposdef", "nonsymposdef-F"):
        # Asymmetric and Positive Definite
        exp = "TFQMR: Linear solve not converged due to reach MAXIT iterations"
    else:  # all other cases
        exp = "TFQMR: Linear solve converged due to reach TOL iterations"

    assert out.startswith(exp)
    assert err == ""


def test_positional_deprecation(solver):
    # from test_x0_working
    rng = np.random.default_rng(1685363802304750)
    n = 10
    A = rng.random(size=[n, n])
    A = A @ A.T
    b = rng.random(n)
    x0 = rng.random(n)
    with pytest.deprecated_call(
        # due to the use of the _deprecate_positional_args decorator, it's not possible
        # to separate the two warnings (1 for positional use, 1 for `tol` deprecation).
        match="use keyword arguments.*|argument `tol` is deprecated.*"
    ):
        solver(A, b, x0, 1e-5)


class TestQMR:
    @pytest.mark.filterwarnings('ignore::scipy.sparse.SparseEfficiencyWarning')
    def test_leftright_precond(self):
        """Check that QMR works with left and right preconditioners"""

        from scipy.sparse.linalg._dsolve import splu
        from scipy.sparse.linalg._interface import LinearOperator

        n = 100

        dat = ones(n)
        A = spdiags([-2 * dat, 4 * dat, -dat], [-1, 0, 1], n, n)
        b = arange(n, dtype='d')

        L = spdiags([-dat / 2, dat], [-1, 0], n, n)
        U = spdiags([4 * dat, -dat], [0, 1], n, n)
        L_solver = splu(L)
        U_solver = splu(U)

        def L_solve(b):
            return L_solver.solve(b)

        def U_solve(b):
            return U_solver.solve(b)

        def LT_solve(b):
            return L_solver.solve(b, 'T')

        def UT_solve(b):
            return U_solver.solve(b, 'T')

        M1 = LinearOperator((n, n), matvec=L_solve, rmatvec=LT_solve)
        M2 = LinearOperator((n, n), matvec=U_solve, rmatvec=UT_solve)

        rtol = 1e-8
        x, info = qmr(A, b, rtol=rtol, maxiter=15, M1=M1, M2=M2)

        assert info == 0
        assert norm(A @ x - b) <= rtol * norm(b)


class TestGMRES:
    def test_basic(self):
        A = np.vander(np.arange(10) + 1)[:, ::-1]
        b = np.zeros(10)
        b[0] = 1

        x_gm, err = gmres(A, b, restart=5, maxiter=1)

        assert_allclose(x_gm[0], 0.359, rtol=1e-2)

    def test_callback(self):

        def store_residual(r, rvec):
            rvec[rvec.nonzero()[0].max() + 1] = r

        # Define, A,b
        A = csr_matrix(array([[-2, 1, 0, 0, 0, 0],
                              [1, -2, 1, 0, 0, 0],
                              [0, 1, -2, 1, 0, 0],
                              [0, 0, 1, -2, 1, 0],
                              [0, 0, 0, 1, -2, 1],
                              [0, 0, 0, 0, 1, -2]]))
        b = ones((A.shape[0],))
        maxiter = 1
        rvec = zeros(maxiter + 1)
        rvec[0] = 1.0

        def callback(r):
            return store_residual(r, rvec)

        x, flag = gmres(A, b, x0=zeros(A.shape[0]), rtol=1e-16,
                        maxiter=maxiter, callback=callback)

        # Expected output from SciPy 1.0.0
        assert_allclose(rvec, array([1.0, 0.81649658092772603]), rtol=1e-10)

        # Test preconditioned callback
        M = 1e-3 * np.eye(A.shape[0])
        rvec = zeros(maxiter + 1)
        rvec[0] = 1.0
        x, flag = gmres(A, b, M=M, rtol=1e-16, maxiter=maxiter,
                        callback=callback)

        # Expected output from SciPy 1.0.0
        # (callback has preconditioned residual!)
        assert_allclose(rvec, array([1.0, 1e-3 * 0.81649658092772603]),
                        rtol=1e-10)

    def test_abi(self):
        # Check we don't segfault on gmres with complex argument
        A = eye(2)
        b = ones(2)
        r_x, r_info = gmres(A, b)
        r_x = r_x.astype(complex)
        x, info = gmres(A.astype(complex), b.astype(complex))

        assert iscomplexobj(x)
        assert_allclose(r_x, x)
        assert r_info == info

    def test_atol_legacy(self):

        A = eye(2)
        b = ones(2)
        x, info = gmres(A, b, rtol=1e-5)
        assert np.linalg.norm(A @ x - b) <= 1e-5 * np.linalg.norm(b)
        assert_allclose(x, b, atol=0, rtol=1e-8)

        rndm = np.random.RandomState(12345)
        A = rndm.rand(30, 30)
        b = 1e-6 * ones(30)
        x, info = gmres(A, b, rtol=1e-7, restart=20)
        assert np.linalg.norm(A @ x - b) > 1e-7

        A = eye(2)
        b = 1e-10 * ones(2)
        x, info = gmres(A, b, rtol=1e-8, atol=0)
        assert np.linalg.norm(A @ x - b) <= 1e-8 * np.linalg.norm(b)

    def test_defective_precond_breakdown(self):
        # Breakdown due to defective preconditioner
        M = np.eye(3)
        M[2, 2] = 0

        b = np.array([0, 1, 1])
        x = np.array([1, 0, 0])
        A = np.diag([2, 3, 4])

        x, info = gmres(A, b, x0=x, M=M, rtol=1e-15, atol=0)

        # Should not return nans, nor terminate with false success
        assert not np.isnan(x).any()
        if info == 0:
            assert np.linalg.norm(A @ x - b) <= 1e-15 * np.linalg.norm(b)

        # The solution should be OK outside null space of M
        assert_allclose(M @ (A @ x), M @ b)

    def test_defective_matrix_breakdown(self):
        # Breakdown due to defective matrix
        A = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        b = np.array([1, 0, 1])
        rtol = 1e-8
        x, info = gmres(A, b, rtol=rtol, atol=0)

        # Should not return nans, nor terminate with false success
        assert not np.isnan(x).any()
        if info == 0:
            assert np.linalg.norm(A @ x - b) <= rtol * np.linalg.norm(b)

        # The solution should be OK outside null space of A
        assert_allclose(A @ (A @ x), A @ b)

    def test_callback_type(self):
        # The legacy callback type changes meaning of 'maxiter'
        np.random.seed(1)
        A = np.random.rand(20, 20)
        b = np.random.rand(20)

        cb_count = [0]

        def pr_norm_cb(r):
            cb_count[0] += 1
            assert isinstance(r, float)

        def x_cb(x):
            cb_count[0] += 1
            assert isinstance(x, np.ndarray)

        # 2 iterations is not enough to solve the problem
        cb_count = [0]
        x, info = gmres(A, b, rtol=1e-6, atol=0, callback=pr_norm_cb,
                        maxiter=2, restart=50)
        assert info == 2
        assert cb_count[0] == 2

        # With `callback_type` specified, no warning should be raised
        cb_count = [0]
        x, info = gmres(A, b, rtol=1e-6, atol=0, callback=pr_norm_cb,
                        maxiter=2, restart=50, callback_type='legacy')
        assert info == 2
        assert cb_count[0] == 2

        # 2 restart cycles is enough to solve the problem
        cb_count = [0]
        x, info = gmres(A, b, rtol=1e-6, atol=0, callback=pr_norm_cb,
                        maxiter=2, restart=50, callback_type='pr_norm')
        assert info == 0
        assert cb_count[0] > 2

        # 2 restart cycles is enough to solve the problem
        cb_count = [0]
        x, info = gmres(A, b, rtol=1e-6, atol=0, callback=x_cb, maxiter=2,
                        restart=50, callback_type='x')
        assert info == 0
        assert cb_count[0] == 1

    def test_callback_x_monotonic(self):
        # Check that callback_type='x' gives monotonic norm decrease
        np.random.seed(1)
        A = np.random.rand(20, 20) + np.eye(20)
        b = np.random.rand(20)

        prev_r = [np.inf]
        count = [0]

        def x_cb(x):
            r = np.linalg.norm(A @ x - b)
            assert r <= prev_r[0]
            prev_r[0] = r
            count[0] += 1

        x, info = gmres(A, b, rtol=1e-6, atol=0, callback=x_cb, maxiter=20,
                        restart=10, callback_type='x')
        assert info == 20
        assert count[0] == 20

    def test_restrt_dep(self):
        with pytest.warns(
            DeprecationWarning,
            match="'gmres' keyword argument 'restrt'"
        ):
            gmres(np.array([1]), np.array([1]), restrt=10)
