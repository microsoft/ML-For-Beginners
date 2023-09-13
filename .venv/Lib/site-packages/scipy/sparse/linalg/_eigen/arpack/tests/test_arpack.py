__usage__ = """
To run tests locally:
  python tests/test_arpack.py [-l<int>] [-v<int>]

"""

import threading
import itertools

import numpy as np

from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from pytest import raises as assert_raises
import pytest

from numpy import dot, conj, random
from scipy.linalg import eig, eigh
from scipy.sparse import csc_matrix, csr_matrix, diags, rand
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse.linalg._eigen.arpack import (eigs, eigsh, arpack,
                                              ArpackNoConvergence)


from scipy._lib._gcutils import assert_deallocated, IS_PYPY


# precision for tests
_ndigits = {'f': 3, 'd': 11, 'F': 3, 'D': 11}


def _get_test_tolerance(type_char, mattype=None, D_type=None, which=None):
    """
    Return tolerance values suitable for a given test:

    Parameters
    ----------
    type_char : {'f', 'd', 'F', 'D'}
        Data type in ARPACK eigenvalue problem
    mattype : {csr_matrix, aslinearoperator, asarray}, optional
        Linear operator type

    Returns
    -------
    tol
        Tolerance to pass to the ARPACK routine
    rtol
        Relative tolerance for outputs
    atol
        Absolute tolerance for outputs

    """

    rtol = {'f': 3000 * np.finfo(np.float32).eps,
            'F': 3000 * np.finfo(np.float32).eps,
            'd': 2000 * np.finfo(np.float64).eps,
            'D': 2000 * np.finfo(np.float64).eps}[type_char]
    atol = rtol
    tol = 0

    if mattype is aslinearoperator and type_char in ('f', 'F'):
        # iterative methods in single precision: worse errors
        # also: bump ARPACK tolerance so that the iterative method converges
        tol = 30 * np.finfo(np.float32).eps
        rtol *= 5

    if mattype is csr_matrix and type_char in ('f', 'F'):
        # sparse in single precision: worse errors
        rtol *= 5

    if (
        which in ('LM', 'SM', 'LA')
        and D_type.name == "gen-hermitian-Mc"
    ):
        if type_char == 'F':
            # missing case 1, 2, and more, from PR 14798
            rtol *= 5

        if type_char == 'D':
            # missing more cases, from PR 14798
            rtol *= 7

    return tol, rtol, atol


def generate_matrix(N, complex_=False, hermitian=False,
                    pos_definite=False, sparse=False):
    M = np.random.random((N, N))
    if complex_:
        M = M + 1j * np.random.random((N, N))

    if hermitian:
        if pos_definite:
            if sparse:
                i = np.arange(N)
                j = np.random.randint(N, size=N-2)
                i, j = np.meshgrid(i, j)
                M[i, j] = 0
            M = np.dot(M.conj(), M.T)
        else:
            M = np.dot(M.conj(), M.T)
            if sparse:
                i = np.random.randint(N, size=N * N // 4)
                j = np.random.randint(N, size=N * N // 4)
                ind = np.nonzero(i == j)
                j[ind] = (j[ind] + 1) % N
                M[i, j] = 0
                M[j, i] = 0
    else:
        if sparse:
            i = np.random.randint(N, size=N * N // 2)
            j = np.random.randint(N, size=N * N // 2)
            M[i, j] = 0
    return M


def generate_matrix_symmetric(N, pos_definite=False, sparse=False):
    M = np.random.random((N, N))

    M = 0.5 * (M + M.T)  # Make M symmetric

    if pos_definite:
        Id = N * np.eye(N)
        if sparse:
            M = csr_matrix(M)
        M += Id
    else:
        if sparse:
            M = csr_matrix(M)

    return M


def assert_allclose_cc(actual, desired, **kw):
    """Almost equal or complex conjugates almost equal"""
    try:
        assert_allclose(actual, desired, **kw)
    except AssertionError:
        assert_allclose(actual, conj(desired), **kw)


def argsort_which(eigenvalues, typ, k, which,
                  sigma=None, OPpart=None, mode=None):
    """Return sorted indices of eigenvalues using the "which" keyword
    from eigs and eigsh"""
    if sigma is None:
        reval = np.round(eigenvalues, decimals=_ndigits[typ])
    else:
        if mode is None or mode == 'normal':
            if OPpart is None:
                reval = 1. / (eigenvalues - sigma)
            elif OPpart == 'r':
                reval = 0.5 * (1. / (eigenvalues - sigma)
                               + 1. / (eigenvalues - np.conj(sigma)))
            elif OPpart == 'i':
                reval = -0.5j * (1. / (eigenvalues - sigma)
                                 - 1. / (eigenvalues - np.conj(sigma)))
        elif mode == 'cayley':
            reval = (eigenvalues + sigma) / (eigenvalues - sigma)
        elif mode == 'buckling':
            reval = eigenvalues / (eigenvalues - sigma)
        else:
            raise ValueError("mode='%s' not recognized" % mode)

        reval = np.round(reval, decimals=_ndigits[typ])

    if which in ['LM', 'SM']:
        ind = np.argsort(abs(reval))
    elif which in ['LR', 'SR', 'LA', 'SA', 'BE']:
        ind = np.argsort(np.real(reval))
    elif which in ['LI', 'SI']:
        # for LI,SI ARPACK returns largest,smallest abs(imaginary) why?
        if typ.islower():
            ind = np.argsort(abs(np.imag(reval)))
        else:
            ind = np.argsort(np.imag(reval))
    else:
        raise ValueError("which='%s' is unrecognized" % which)

    if which in ['LM', 'LA', 'LR', 'LI']:
        return ind[-k:]
    elif which in ['SM', 'SA', 'SR', 'SI']:
        return ind[:k]
    elif which == 'BE':
        return np.concatenate((ind[:k//2], ind[k//2-k:]))


def eval_evec(symmetric, d, typ, k, which, v0=None, sigma=None,
              mattype=np.asarray, OPpart=None, mode='normal'):
    general = ('bmat' in d)

    if symmetric:
        eigs_func = eigsh
    else:
        eigs_func = eigs

    if general:
        err = ("error for {}:general, typ={}, which={}, sigma={}, "
               "mattype={}, OPpart={}, mode={}".format(eigs_func.__name__,
                                                   typ, which, sigma,
                                                   mattype.__name__,
                                                   OPpart, mode))
    else:
        err = ("error for {}:standard, typ={}, which={}, sigma={}, "
               "mattype={}, OPpart={}, mode={}".format(eigs_func.__name__,
                                                   typ, which, sigma,
                                                   mattype.__name__,
                                                   OPpart, mode))

    a = d['mat'].astype(typ)
    ac = mattype(a)

    if general:
        b = d['bmat'].astype(typ)
        bc = mattype(b)

    # get exact eigenvalues
    exact_eval = d['eval'].astype(typ.upper())
    ind = argsort_which(exact_eval, typ, k, which,
                        sigma, OPpart, mode)
    exact_eval = exact_eval[ind]

    # compute arpack eigenvalues
    kwargs = dict(which=which, v0=v0, sigma=sigma)
    if eigs_func is eigsh:
        kwargs['mode'] = mode
    else:
        kwargs['OPpart'] = OPpart

    # compute suitable tolerances
    kwargs['tol'], rtol, atol = _get_test_tolerance(typ, mattype, d, which)
    # on rare occasions, ARPACK routines return results that are proper
    # eigenvalues and -vectors, but not necessarily the ones requested in
    # the parameter which. This is inherent to the Krylov methods, and
    # should not be treated as a failure. If such a rare situation
    # occurs, the calculation is tried again (but at most a few times).
    ntries = 0
    while ntries < 5:
        # solve
        if general:
            try:
                eigenvalues, evec = eigs_func(ac, k, bc, **kwargs)
            except ArpackNoConvergence:
                kwargs['maxiter'] = 20*a.shape[0]
                eigenvalues, evec = eigs_func(ac, k, bc, **kwargs)
        else:
            try:
                eigenvalues, evec = eigs_func(ac, k, **kwargs)
            except ArpackNoConvergence:
                kwargs['maxiter'] = 20*a.shape[0]
                eigenvalues, evec = eigs_func(ac, k, **kwargs)

        ind = argsort_which(eigenvalues, typ, k, which,
                            sigma, OPpart, mode)
        eigenvalues = eigenvalues[ind]
        evec = evec[:, ind]

        try:
            # check eigenvalues
            assert_allclose_cc(eigenvalues, exact_eval, rtol=rtol, atol=atol,
                               err_msg=err)
            check_evecs = True
        except AssertionError:
            check_evecs = False
            ntries += 1

        if check_evecs:
            # check eigenvectors
            LHS = np.dot(a, evec)
            if general:
                RHS = eigenvalues * np.dot(b, evec)
            else:
                RHS = eigenvalues * evec

            assert_allclose(LHS, RHS, rtol=rtol, atol=atol, err_msg=err)
            break

    # check eigenvalues
    assert_allclose_cc(eigenvalues, exact_eval, rtol=rtol, atol=atol, err_msg=err)


class DictWithRepr(dict):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return "<%s>" % self.name


class SymmetricParams:
    def __init__(self):
        self.eigs = eigsh
        self.which = ['LM', 'SM', 'LA', 'SA', 'BE']
        self.mattypes = [csr_matrix, aslinearoperator, np.asarray]
        self.sigmas_modes = {None: ['normal'],
                             0.5: ['normal', 'buckling', 'cayley']}

        # generate matrices
        # these should all be float32 so that the eigenvalues
        # are the same in float32 and float64
        N = 6
        np.random.seed(2300)
        Ar = generate_matrix(N, hermitian=True,
                             pos_definite=True).astype('f').astype('d')
        M = generate_matrix(N, hermitian=True,
                            pos_definite=True).astype('f').astype('d')
        Ac = generate_matrix(N, hermitian=True, pos_definite=True,
                             complex_=True).astype('F').astype('D')
        Mc = generate_matrix(N, hermitian=True, pos_definite=True,
                             complex_=True).astype('F').astype('D')
        v0 = np.random.random(N)

        # standard symmetric problem
        SS = DictWithRepr("std-symmetric")
        SS['mat'] = Ar
        SS['v0'] = v0
        SS['eval'] = eigh(SS['mat'], eigvals_only=True)

        # general symmetric problem
        GS = DictWithRepr("gen-symmetric")
        GS['mat'] = Ar
        GS['bmat'] = M
        GS['v0'] = v0
        GS['eval'] = eigh(GS['mat'], GS['bmat'], eigvals_only=True)

        # standard hermitian problem
        SH = DictWithRepr("std-hermitian")
        SH['mat'] = Ac
        SH['v0'] = v0
        SH['eval'] = eigh(SH['mat'], eigvals_only=True)

        # general hermitian problem
        GH = DictWithRepr("gen-hermitian")
        GH['mat'] = Ac
        GH['bmat'] = M
        GH['v0'] = v0
        GH['eval'] = eigh(GH['mat'], GH['bmat'], eigvals_only=True)

        # general hermitian problem with hermitian M
        GHc = DictWithRepr("gen-hermitian-Mc")
        GHc['mat'] = Ac
        GHc['bmat'] = Mc
        GHc['v0'] = v0
        GHc['eval'] = eigh(GHc['mat'], GHc['bmat'], eigvals_only=True)

        self.real_test_cases = [SS, GS]
        self.complex_test_cases = [SH, GH, GHc]


class NonSymmetricParams:
    def __init__(self):
        self.eigs = eigs
        self.which = ['LM', 'LR', 'LI']  # , 'SM', 'LR', 'SR', 'LI', 'SI']
        self.mattypes = [csr_matrix, aslinearoperator, np.asarray]
        self.sigmas_OPparts = {None: [None],
                               0.1: ['r'],
                               0.1 + 0.1j: ['r', 'i']}

        # generate matrices
        # these should all be float32 so that the eigenvalues
        # are the same in float32 and float64
        N = 6
        np.random.seed(2300)
        Ar = generate_matrix(N).astype('f').astype('d')
        M = generate_matrix(N, hermitian=True,
                            pos_definite=True).astype('f').astype('d')
        Ac = generate_matrix(N, complex_=True).astype('F').astype('D')
        v0 = np.random.random(N)

        # standard real nonsymmetric problem
        SNR = DictWithRepr("std-real-nonsym")
        SNR['mat'] = Ar
        SNR['v0'] = v0
        SNR['eval'] = eig(SNR['mat'], left=False, right=False)

        # general real nonsymmetric problem
        GNR = DictWithRepr("gen-real-nonsym")
        GNR['mat'] = Ar
        GNR['bmat'] = M
        GNR['v0'] = v0
        GNR['eval'] = eig(GNR['mat'], GNR['bmat'], left=False, right=False)

        # standard complex nonsymmetric problem
        SNC = DictWithRepr("std-cmplx-nonsym")
        SNC['mat'] = Ac
        SNC['v0'] = v0
        SNC['eval'] = eig(SNC['mat'], left=False, right=False)

        # general complex nonsymmetric problem
        GNC = DictWithRepr("gen-cmplx-nonsym")
        GNC['mat'] = Ac
        GNC['bmat'] = M
        GNC['v0'] = v0
        GNC['eval'] = eig(GNC['mat'], GNC['bmat'], left=False, right=False)

        self.real_test_cases = [SNR, GNR]
        self.complex_test_cases = [SNC, GNC]


def test_symmetric_modes():
    params = SymmetricParams()
    k = 2
    symmetric = True
    for D in params.real_test_cases:
        for typ in 'fd':
            for which in params.which:
                for mattype in params.mattypes:
                    for (sigma, modes) in params.sigmas_modes.items():
                        for mode in modes:
                            eval_evec(symmetric, D, typ, k, which,
                                      None, sigma, mattype, None, mode)


def test_hermitian_modes():
    params = SymmetricParams()
    k = 2
    symmetric = True
    for D in params.complex_test_cases:
        for typ in 'FD':
            for which in params.which:
                if which == 'BE':
                    continue  # BE invalid for complex
                for mattype in params.mattypes:
                    for sigma in params.sigmas_modes:
                        eval_evec(symmetric, D, typ, k, which,
                                  None, sigma, mattype)


def test_symmetric_starting_vector():
    params = SymmetricParams()
    symmetric = True
    for k in [1, 2, 3, 4, 5]:
        for D in params.real_test_cases:
            for typ in 'fd':
                v0 = random.rand(len(D['v0'])).astype(typ)
                eval_evec(symmetric, D, typ, k, 'LM', v0)


def test_symmetric_no_convergence():
    np.random.seed(1234)
    m = generate_matrix(30, hermitian=True, pos_definite=True)
    tol, rtol, atol = _get_test_tolerance('d')
    try:
        w, v = eigsh(m, 4, which='LM', v0=m[:, 0], maxiter=5, tol=tol, ncv=9)
        raise AssertionError("Spurious no-error exit")
    except ArpackNoConvergence as err:
        k = len(err.eigenvalues)
        if k <= 0:
            raise AssertionError("Spurious no-eigenvalues-found case") from err
        w, v = err.eigenvalues, err.eigenvectors
        assert_allclose(dot(m, v), w * v, rtol=rtol, atol=atol)


def test_real_nonsymmetric_modes():
    params = NonSymmetricParams()
    k = 2
    symmetric = False
    for D in params.real_test_cases:
        for typ in 'fd':
            for which in params.which:
                for mattype in params.mattypes:
                    for sigma, OPparts in params.sigmas_OPparts.items():
                        for OPpart in OPparts:
                            eval_evec(symmetric, D, typ, k, which,
                                      None, sigma, mattype, OPpart)


def test_complex_nonsymmetric_modes():
    params = NonSymmetricParams()
    k = 2
    symmetric = False
    for D in params.complex_test_cases:
        for typ in 'DF':
            for which in params.which:
                for mattype in params.mattypes:
                    for sigma in params.sigmas_OPparts:
                        eval_evec(symmetric, D, typ, k, which,
                                  None, sigma, mattype)


def test_standard_nonsymmetric_starting_vector():
    params = NonSymmetricParams()
    sigma = None
    symmetric = False
    for k in [1, 2, 3, 4]:
        for d in params.complex_test_cases:
            for typ in 'FD':
                A = d['mat']
                n = A.shape[0]
                v0 = random.rand(n).astype(typ)
                eval_evec(symmetric, d, typ, k, "LM", v0, sigma)


def test_general_nonsymmetric_starting_vector():
    params = NonSymmetricParams()
    sigma = None
    symmetric = False
    for k in [1, 2, 3, 4]:
        for d in params.complex_test_cases:
            for typ in 'FD':
                A = d['mat']
                n = A.shape[0]
                v0 = random.rand(n).astype(typ)
                eval_evec(symmetric, d, typ, k, "LM", v0, sigma)


def test_standard_nonsymmetric_no_convergence():
    np.random.seed(1234)
    m = generate_matrix(30, complex_=True)
    tol, rtol, atol = _get_test_tolerance('d')
    try:
        w, v = eigs(m, 4, which='LM', v0=m[:, 0], maxiter=5, tol=tol)
        raise AssertionError("Spurious no-error exit")
    except ArpackNoConvergence as err:
        k = len(err.eigenvalues)
        if k <= 0:
            raise AssertionError("Spurious no-eigenvalues-found case") from err
        w, v = err.eigenvalues, err.eigenvectors
        for ww, vv in zip(w, v.T):
            assert_allclose(dot(m, vv), ww * vv, rtol=rtol, atol=atol)


def test_eigen_bad_shapes():
    # A is not square.
    A = csc_matrix(np.zeros((2, 3)))
    assert_raises(ValueError, eigs, A)


def test_eigen_bad_kwargs():
    # Test eigen on wrong keyword argument
    A = csc_matrix(np.zeros((8, 8)))
    assert_raises(ValueError, eigs, A, which='XX')


def test_ticket_1459_arpack_crash():
    for dtype in [np.float32, np.float64]:
        # This test does not seem to catch the issue for float32,
        # but we made the same fix there, just to be sure

        N = 6
        k = 2

        np.random.seed(2301)
        A = np.random.random((N, N)).astype(dtype)
        v0 = np.array([-0.71063568258907849895, -0.83185111795729227424,
                       -0.34365925382227402451, 0.46122533684552280420,
                       -0.58001341115969040629, -0.78844877570084292984e-01],
                      dtype=dtype)

        # Should not crash:
        evals, evecs = eigs(A, k, v0=v0)


@pytest.mark.skipif(IS_PYPY, reason="Test not meaningful on PyPy")
def test_linearoperator_deallocation():
    # Check that the linear operators used by the Arpack wrappers are
    # deallocatable by reference counting -- they are big objects, so
    # Python's cyclic GC may not collect them fast enough before
    # running out of memory if eigs/eigsh are called in a tight loop.

    M_d = np.eye(10)
    M_s = csc_matrix(M_d)
    M_o = aslinearoperator(M_d)

    with assert_deallocated(lambda: arpack.SpLuInv(M_s)):
        pass
    with assert_deallocated(lambda: arpack.LuInv(M_d)):
        pass
    with assert_deallocated(lambda: arpack.IterInv(M_s)):
        pass
    with assert_deallocated(lambda: arpack.IterOpInv(M_o, None, 0.3)):
        pass
    with assert_deallocated(lambda: arpack.IterOpInv(M_o, M_o, 0.3)):
        pass

def test_parallel_threads():
    results = []
    v0 = np.random.rand(50)

    def worker():
        x = diags([1, -2, 1], [-1, 0, 1], shape=(50, 50))
        w, v = eigs(x, k=3, v0=v0)
        results.append(w)

        w, v = eigsh(x, k=3, v0=v0)
        results.append(w)

    threads = [threading.Thread(target=worker) for k in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    worker()

    for r in results:
        assert_allclose(r, results[-1])


def test_reentering():
    # Just some linear operator that calls eigs recursively
    def A_matvec(x):
        x = diags([1, -2, 1], [-1, 0, 1], shape=(50, 50))
        w, v = eigs(x, k=1)
        return v / w[0]
    A = LinearOperator(matvec=A_matvec, dtype=float, shape=(50, 50))

    # The Fortran code is not reentrant, so this fails (gracefully, not crashing)
    assert_raises(RuntimeError, eigs, A, k=1)
    assert_raises(RuntimeError, eigsh, A, k=1)


def test_regression_arpackng_1315():
    # Check that issue arpack-ng/#1315 is not present.
    # Adapted from arpack-ng/TESTS/bug_1315_single.c
    # If this fails, then the installed ARPACK library is faulty.

    for dtype in [np.float32, np.float64]:
        np.random.seed(1234)

        w0 = np.arange(1, 1000+1).astype(dtype)
        A = diags([w0], [0], shape=(1000, 1000))

        v0 = np.random.rand(1000).astype(dtype)
        w, v = eigs(A, k=9, ncv=2*9+1, which="LM", v0=v0)

        assert_allclose(np.sort(w), np.sort(w0[-9:]),
                        rtol=1e-4)


def test_eigs_for_k_greater():
    # Test eigs() for k beyond limits.
    A_sparse = diags([1, -2, 1], [-1, 0, 1], shape=(4, 4))  # sparse
    A = generate_matrix(4, sparse=False)
    M_dense = np.random.random((4, 4))
    M_sparse = generate_matrix(4, sparse=True)
    M_linop = aslinearoperator(M_dense)
    eig_tuple1 = eig(A, b=M_dense)
    eig_tuple2 = eig(A, b=M_sparse)

    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning)

        assert_equal(eigs(A, M=M_dense, k=3), eig_tuple1)
        assert_equal(eigs(A, M=M_dense, k=4), eig_tuple1)
        assert_equal(eigs(A, M=M_dense, k=5), eig_tuple1)
        assert_equal(eigs(A, M=M_sparse, k=5), eig_tuple2)

        # M as LinearOperator
        assert_raises(TypeError, eigs, A, M=M_linop, k=3)

        # Test 'A' for different types
        assert_raises(TypeError, eigs, aslinearoperator(A), k=3)
        assert_raises(TypeError, eigs, A_sparse, k=3)


def test_eigsh_for_k_greater():
    # Test eigsh() for k beyond limits.
    A_sparse = diags([1, -2, 1], [-1, 0, 1], shape=(4, 4))  # sparse
    A = generate_matrix(4, sparse=False)
    M_dense = generate_matrix_symmetric(4, pos_definite=True)
    M_sparse = generate_matrix_symmetric(4, pos_definite=True, sparse=True)
    M_linop = aslinearoperator(M_dense)
    eig_tuple1 = eigh(A, b=M_dense)
    eig_tuple2 = eigh(A, b=M_sparse)

    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning)

        assert_equal(eigsh(A, M=M_dense, k=4), eig_tuple1)
        assert_equal(eigsh(A, M=M_dense, k=5), eig_tuple1)
        assert_equal(eigsh(A, M=M_sparse, k=5), eig_tuple2)

        # M as LinearOperator
        assert_raises(TypeError, eigsh, A, M=M_linop, k=4)

        # Test 'A' for different types
        assert_raises(TypeError, eigsh, aslinearoperator(A), k=4)
        assert_raises(TypeError, eigsh, A_sparse, M=M_dense, k=4)


def test_real_eigs_real_k_subset():
    np.random.seed(1)

    n = 10
    A = rand(n, n, density=0.5)
    A.data *= 2
    A.data -= 1

    v0 = np.ones(n)

    whichs = ['LM', 'SM', 'LR', 'SR', 'LI', 'SI']
    dtypes = [np.float32, np.float64]

    for which, sigma, dtype in itertools.product(whichs, [None, 0, 5], dtypes):
        prev_w = np.array([], dtype=dtype)
        eps = np.finfo(dtype).eps
        for k in range(1, 9):
            w, z = eigs(A.astype(dtype), k=k, which=which, sigma=sigma,
                        v0=v0.astype(dtype), tol=0)
            assert_allclose(np.linalg.norm(A.dot(z) - z * w), 0, atol=np.sqrt(eps))

            # Check that the set of eigenvalues for `k` is a subset of that for `k+1`
            dist = abs(prev_w[:,None] - w).min(axis=1)
            assert_allclose(dist, 0, atol=np.sqrt(eps))

            prev_w = w

            # Check sort order
            if sigma is None:
                d = w
            else:
                d = 1 / (w - sigma)

            if which == 'LM':
                # ARPACK is systematic for 'LM', but sort order
                # appears not well defined for other modes
                assert np.all(np.diff(abs(d)) <= 1e-6)
