import re
import sys
import warnings
from io import StringIO

import numpy as np
import pytest
import scipy.sparse as sp
from scipy import linalg
from scipy.sparse import csc_matrix

from sklearn.base import clone
from sklearn.decomposition import NMF, MiniBatchNMF, non_negative_factorization
from sklearn.decomposition import _nmf as nmf  # For testing internals
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
    ignore_warnings,
)
from sklearn.utils.extmath import squared_norm


@pytest.mark.parametrize(
    ["Estimator", "solver"],
    [[NMF, {"solver": "cd"}], [NMF, {"solver": "mu"}], [MiniBatchNMF, {}]],
)
def test_convergence_warning(Estimator, solver):
    convergence_warning = (
        "Maximum number of iterations 1 reached. Increase it to improve convergence."
    )
    A = np.ones((2, 2))
    with pytest.warns(ConvergenceWarning, match=convergence_warning):
        Estimator(max_iter=1, **solver).fit(A)


def test_initialize_nn_output():
    # Test that initialization does not return negative values
    rng = np.random.mtrand.RandomState(42)
    data = np.abs(rng.randn(10, 10))
    for init in ("random", "nndsvd", "nndsvda", "nndsvdar"):
        W, H = nmf._initialize_nmf(data, 10, init=init, random_state=0)
        assert not ((W < 0).any() or (H < 0).any())


@pytest.mark.filterwarnings(
    r"ignore:The multiplicative update \('mu'\) solver cannot update zeros present in"
    r" the initialization"
)
def test_parameter_checking():
    # Here we only check for invalid parameter values that are not already
    # automatically tested in the common tests.

    A = np.ones((2, 2))

    msg = "Invalid beta_loss parameter: solver 'cd' does not handle beta_loss = 1.0"
    with pytest.raises(ValueError, match=msg):
        NMF(solver="cd", beta_loss=1.0).fit(A)
    msg = "Negative values in data passed to"
    with pytest.raises(ValueError, match=msg):
        NMF().fit(-A)
    clf = NMF(2, tol=0.1).fit(A)
    with pytest.raises(ValueError, match=msg):
        clf.transform(-A)
    with pytest.raises(ValueError, match=msg):
        nmf._initialize_nmf(-A, 2, "nndsvd")

    for init in ["nndsvd", "nndsvda", "nndsvdar"]:
        msg = re.escape(
            "init = '{}' can only be used when "
            "n_components <= min(n_samples, n_features)".format(init)
        )
        with pytest.raises(ValueError, match=msg):
            NMF(3, init=init).fit(A)
        with pytest.raises(ValueError, match=msg):
            MiniBatchNMF(3, init=init).fit(A)
        with pytest.raises(ValueError, match=msg):
            nmf._initialize_nmf(A, 3, init)


def test_initialize_close():
    # Test NNDSVD error
    # Test that _initialize_nmf error is less than the standard deviation of
    # the entries in the matrix.
    rng = np.random.mtrand.RandomState(42)
    A = np.abs(rng.randn(10, 10))
    W, H = nmf._initialize_nmf(A, 10, init="nndsvd")
    error = linalg.norm(np.dot(W, H) - A)
    sdev = linalg.norm(A - A.mean())
    assert error <= sdev


def test_initialize_variants():
    # Test NNDSVD variants correctness
    # Test that the variants 'nndsvda' and 'nndsvdar' differ from basic
    # 'nndsvd' only where the basic version has zeros.
    rng = np.random.mtrand.RandomState(42)
    data = np.abs(rng.randn(10, 10))
    W0, H0 = nmf._initialize_nmf(data, 10, init="nndsvd")
    Wa, Ha = nmf._initialize_nmf(data, 10, init="nndsvda")
    War, Har = nmf._initialize_nmf(data, 10, init="nndsvdar", random_state=0)

    for ref, evl in ((W0, Wa), (W0, War), (H0, Ha), (H0, Har)):
        assert_almost_equal(evl[ref != 0], ref[ref != 0])


# ignore UserWarning raised when both solver='mu' and init='nndsvd'
@ignore_warnings(category=UserWarning)
@pytest.mark.parametrize(
    ["Estimator", "solver"],
    [[NMF, {"solver": "cd"}], [NMF, {"solver": "mu"}], [MiniBatchNMF, {}]],
)
@pytest.mark.parametrize("init", (None, "nndsvd", "nndsvda", "nndsvdar", "random"))
@pytest.mark.parametrize("alpha_W", (0.0, 1.0))
@pytest.mark.parametrize("alpha_H", (0.0, 1.0, "same"))
def test_nmf_fit_nn_output(Estimator, solver, init, alpha_W, alpha_H):
    # Test that the decomposition does not contain negative values
    A = np.c_[5.0 - np.arange(1, 6), 5.0 + np.arange(1, 6)]
    model = Estimator(
        n_components=2,
        init=init,
        alpha_W=alpha_W,
        alpha_H=alpha_H,
        random_state=0,
        **solver,
    )
    transf = model.fit_transform(A)
    assert not ((model.components_ < 0).any() or (transf < 0).any())


@pytest.mark.parametrize(
    ["Estimator", "solver"],
    [[NMF, {"solver": "cd"}], [NMF, {"solver": "mu"}], [MiniBatchNMF, {}]],
)
def test_nmf_fit_close(Estimator, solver):
    rng = np.random.mtrand.RandomState(42)
    # Test that the fit is not too far away
    pnmf = Estimator(
        5,
        init="nndsvdar",
        random_state=0,
        max_iter=600,
        **solver,
    )
    X = np.abs(rng.randn(6, 5))
    assert pnmf.fit(X).reconstruction_err_ < 0.1


def test_nmf_true_reconstruction():
    # Test that the fit is not too far away from an exact solution
    # (by construction)
    n_samples = 15
    n_features = 10
    n_components = 5
    beta_loss = 1
    batch_size = 3
    max_iter = 1000

    rng = np.random.mtrand.RandomState(42)
    W_true = np.zeros([n_samples, n_components])
    W_array = np.abs(rng.randn(n_samples))
    for j in range(n_components):
        W_true[j % n_samples, j] = W_array[j % n_samples]
    H_true = np.zeros([n_components, n_features])
    H_array = np.abs(rng.randn(n_components))
    for j in range(n_features):
        H_true[j % n_components, j] = H_array[j % n_components]
    X = np.dot(W_true, H_true)

    model = NMF(
        n_components=n_components,
        solver="mu",
        beta_loss=beta_loss,
        max_iter=max_iter,
        random_state=0,
    )
    transf = model.fit_transform(X)
    X_calc = np.dot(transf, model.components_)

    assert model.reconstruction_err_ < 0.1
    assert_allclose(X, X_calc)

    mbmodel = MiniBatchNMF(
        n_components=n_components,
        beta_loss=beta_loss,
        batch_size=batch_size,
        random_state=0,
        max_iter=max_iter,
    )
    transf = mbmodel.fit_transform(X)
    X_calc = np.dot(transf, mbmodel.components_)

    assert mbmodel.reconstruction_err_ < 0.1
    assert_allclose(X, X_calc, atol=1)


@pytest.mark.parametrize("solver", ["cd", "mu"])
def test_nmf_transform(solver):
    # Test that fit_transform is equivalent to fit.transform for NMF
    # Test that NMF.transform returns close values
    rng = np.random.mtrand.RandomState(42)
    A = np.abs(rng.randn(6, 5))
    m = NMF(
        solver=solver,
        n_components=3,
        init="random",
        random_state=0,
        tol=1e-6,
    )
    ft = m.fit_transform(A)
    t = m.transform(A)
    assert_allclose(ft, t, atol=1e-1)


def test_minibatch_nmf_transform():
    # Test that fit_transform is equivalent to fit.transform for MiniBatchNMF
    # Only guaranteed with fresh restarts
    rng = np.random.mtrand.RandomState(42)
    A = np.abs(rng.randn(6, 5))
    m = MiniBatchNMF(
        n_components=3,
        random_state=0,
        tol=1e-3,
        fresh_restarts=True,
    )
    ft = m.fit_transform(A)
    t = m.transform(A)
    assert_allclose(ft, t)


@pytest.mark.parametrize(
    ["Estimator", "solver"],
    [[NMF, {"solver": "cd"}], [NMF, {"solver": "mu"}], [MiniBatchNMF, {}]],
)
def test_nmf_transform_custom_init(Estimator, solver):
    # Smoke test that checks if NMF.transform works with custom initialization
    random_state = np.random.RandomState(0)
    A = np.abs(random_state.randn(6, 5))
    n_components = 4
    avg = np.sqrt(A.mean() / n_components)
    H_init = np.abs(avg * random_state.randn(n_components, 5))
    W_init = np.abs(avg * random_state.randn(6, n_components))

    m = Estimator(
        n_components=n_components, init="custom", random_state=0, tol=1e-3, **solver
    )
    m.fit_transform(A, W=W_init, H=H_init)
    m.transform(A)


@pytest.mark.parametrize("solver", ("cd", "mu"))
def test_nmf_inverse_transform(solver):
    # Test that NMF.inverse_transform returns close values
    random_state = np.random.RandomState(0)
    A = np.abs(random_state.randn(6, 4))
    m = NMF(
        solver=solver,
        n_components=4,
        init="random",
        random_state=0,
        max_iter=1000,
    )
    ft = m.fit_transform(A)
    A_new = m.inverse_transform(ft)
    assert_array_almost_equal(A, A_new, decimal=2)


def test_mbnmf_inverse_transform():
    # Test that MiniBatchNMF.transform followed by MiniBatchNMF.inverse_transform
    # is close to the identity
    rng = np.random.RandomState(0)
    A = np.abs(rng.randn(6, 4))
    nmf = MiniBatchNMF(
        random_state=rng,
        max_iter=500,
        init="nndsvdar",
        fresh_restarts=True,
    )
    ft = nmf.fit_transform(A)
    A_new = nmf.inverse_transform(ft)
    assert_allclose(A, A_new, rtol=1e-3, atol=1e-2)


@pytest.mark.parametrize("Estimator", [NMF, MiniBatchNMF])
def test_n_components_greater_n_features(Estimator):
    # Smoke test for the case of more components than features.
    rng = np.random.mtrand.RandomState(42)
    A = np.abs(rng.randn(30, 10))
    Estimator(n_components=15, random_state=0, tol=1e-2).fit(A)


@pytest.mark.parametrize(
    ["Estimator", "solver"],
    [[NMF, {"solver": "cd"}], [NMF, {"solver": "mu"}], [MiniBatchNMF, {}]],
)
@pytest.mark.parametrize("alpha_W", (0.0, 1.0))
@pytest.mark.parametrize("alpha_H", (0.0, 1.0, "same"))
def test_nmf_sparse_input(Estimator, solver, alpha_W, alpha_H):
    # Test that sparse matrices are accepted as input
    from scipy.sparse import csc_matrix

    rng = np.random.mtrand.RandomState(42)
    A = np.abs(rng.randn(10, 10))
    A[:, 2 * np.arange(5)] = 0
    A_sparse = csc_matrix(A)

    est1 = Estimator(
        n_components=5,
        init="random",
        alpha_W=alpha_W,
        alpha_H=alpha_H,
        random_state=0,
        tol=0,
        max_iter=100,
        **solver,
    )
    est2 = clone(est1)

    W1 = est1.fit_transform(A)
    W2 = est2.fit_transform(A_sparse)
    H1 = est1.components_
    H2 = est2.components_

    assert_allclose(W1, W2)
    assert_allclose(H1, H2)


@pytest.mark.parametrize(
    ["Estimator", "solver"],
    [[NMF, {"solver": "cd"}], [NMF, {"solver": "mu"}], [MiniBatchNMF, {}]],
)
def test_nmf_sparse_transform(Estimator, solver):
    # Test that transform works on sparse data.  Issue #2124
    rng = np.random.mtrand.RandomState(42)
    A = np.abs(rng.randn(3, 2))
    A[1, 1] = 0
    A = csc_matrix(A)

    model = Estimator(random_state=0, n_components=2, max_iter=400, **solver)
    A_fit_tr = model.fit_transform(A)
    A_tr = model.transform(A)
    assert_allclose(A_fit_tr, A_tr, atol=1e-1)


@pytest.mark.parametrize("init", ["random", "nndsvd"])
@pytest.mark.parametrize("solver", ("cd", "mu"))
@pytest.mark.parametrize("alpha_W", (0.0, 1.0))
@pytest.mark.parametrize("alpha_H", (0.0, 1.0, "same"))
def test_non_negative_factorization_consistency(init, solver, alpha_W, alpha_H):
    # Test that the function is called in the same way, either directly
    # or through the NMF class
    max_iter = 500
    rng = np.random.mtrand.RandomState(42)
    A = np.abs(rng.randn(10, 10))
    A[:, 2 * np.arange(5)] = 0

    W_nmf, H, _ = non_negative_factorization(
        A,
        init=init,
        solver=solver,
        max_iter=max_iter,
        alpha_W=alpha_W,
        alpha_H=alpha_H,
        random_state=1,
        tol=1e-2,
    )
    W_nmf_2, H, _ = non_negative_factorization(
        A,
        H=H,
        update_H=False,
        init=init,
        solver=solver,
        max_iter=max_iter,
        alpha_W=alpha_W,
        alpha_H=alpha_H,
        random_state=1,
        tol=1e-2,
    )

    model_class = NMF(
        init=init,
        solver=solver,
        max_iter=max_iter,
        alpha_W=alpha_W,
        alpha_H=alpha_H,
        random_state=1,
        tol=1e-2,
    )
    W_cls = model_class.fit_transform(A)
    W_cls_2 = model_class.transform(A)

    assert_allclose(W_nmf, W_cls)
    assert_allclose(W_nmf_2, W_cls_2)


def test_non_negative_factorization_checking():
    # Note that the validity of parameter types and range of possible values
    # for scalar numerical or str parameters is already checked in the common
    # tests. Here we only check for problems that cannot be captured by simple
    # declarative constraints on the valid parameter values.

    A = np.ones((2, 2))
    # Test parameters checking in public function
    nnmf = non_negative_factorization
    msg = re.escape("Negative values in data passed to NMF (input H)")
    with pytest.raises(ValueError, match=msg):
        nnmf(A, A, -A, 2, init="custom")
    msg = re.escape("Negative values in data passed to NMF (input W)")
    with pytest.raises(ValueError, match=msg):
        nnmf(A, -A, A, 2, init="custom")
    msg = re.escape("Array passed to NMF (input H) is full of zeros")
    with pytest.raises(ValueError, match=msg):
        nnmf(A, A, 0 * A, 2, init="custom")


def _beta_divergence_dense(X, W, H, beta):
    """Compute the beta-divergence of X and W.H for dense array only.

    Used as a reference for testing nmf._beta_divergence.
    """
    WH = np.dot(W, H)

    if beta == 2:
        return squared_norm(X - WH) / 2

    WH_Xnonzero = WH[X != 0]
    X_nonzero = X[X != 0]
    np.maximum(WH_Xnonzero, 1e-9, out=WH_Xnonzero)

    if beta == 1:
        res = np.sum(X_nonzero * np.log(X_nonzero / WH_Xnonzero))
        res += WH.sum() - X.sum()

    elif beta == 0:
        div = X_nonzero / WH_Xnonzero
        res = np.sum(div) - X.size - np.sum(np.log(div))
    else:
        res = (X_nonzero**beta).sum()
        res += (beta - 1) * (WH**beta).sum()
        res -= beta * (X_nonzero * (WH_Xnonzero ** (beta - 1))).sum()
        res /= beta * (beta - 1)

    return res


def test_beta_divergence():
    # Compare _beta_divergence with the reference _beta_divergence_dense
    n_samples = 20
    n_features = 10
    n_components = 5
    beta_losses = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]

    # initialization
    rng = np.random.mtrand.RandomState(42)
    X = rng.randn(n_samples, n_features)
    np.clip(X, 0, None, out=X)
    X_csr = sp.csr_matrix(X)
    W, H = nmf._initialize_nmf(X, n_components, init="random", random_state=42)

    for beta in beta_losses:
        ref = _beta_divergence_dense(X, W, H, beta)
        loss = nmf._beta_divergence(X, W, H, beta)
        loss_csr = nmf._beta_divergence(X_csr, W, H, beta)

        assert_almost_equal(ref, loss, decimal=7)
        assert_almost_equal(ref, loss_csr, decimal=7)


def test_special_sparse_dot():
    # Test the function that computes np.dot(W, H), only where X is non zero.
    n_samples = 10
    n_features = 5
    n_components = 3
    rng = np.random.mtrand.RandomState(42)
    X = rng.randn(n_samples, n_features)
    np.clip(X, 0, None, out=X)
    X_csr = sp.csr_matrix(X)

    W = np.abs(rng.randn(n_samples, n_components))
    H = np.abs(rng.randn(n_components, n_features))

    WH_safe = nmf._special_sparse_dot(W, H, X_csr)
    WH = nmf._special_sparse_dot(W, H, X)

    # test that both results have same values, in X_csr nonzero elements
    ii, jj = X_csr.nonzero()
    WH_safe_data = np.asarray(WH_safe[ii, jj]).ravel()
    assert_array_almost_equal(WH_safe_data, WH[ii, jj], decimal=10)

    # test that WH_safe and X_csr have the same sparse structure
    assert_array_equal(WH_safe.indices, X_csr.indices)
    assert_array_equal(WH_safe.indptr, X_csr.indptr)
    assert_array_equal(WH_safe.shape, X_csr.shape)


@ignore_warnings(category=ConvergenceWarning)
def test_nmf_multiplicative_update_sparse():
    # Compare sparse and dense input in multiplicative update NMF
    # Also test continuity of the results with respect to beta_loss parameter
    n_samples = 20
    n_features = 10
    n_components = 5
    alpha = 0.1
    l1_ratio = 0.5
    n_iter = 20

    # initialization
    rng = np.random.mtrand.RandomState(1337)
    X = rng.randn(n_samples, n_features)
    X = np.abs(X)
    X_csr = sp.csr_matrix(X)
    W0, H0 = nmf._initialize_nmf(X, n_components, init="random", random_state=42)

    for beta_loss in (-1.2, 0, 0.2, 1.0, 2.0, 2.5):
        # Reference with dense array X
        W, H = W0.copy(), H0.copy()
        W1, H1, _ = non_negative_factorization(
            X,
            W,
            H,
            n_components,
            init="custom",
            update_H=True,
            solver="mu",
            beta_loss=beta_loss,
            max_iter=n_iter,
            alpha_W=alpha,
            l1_ratio=l1_ratio,
            random_state=42,
        )

        # Compare with sparse X
        W, H = W0.copy(), H0.copy()
        W2, H2, _ = non_negative_factorization(
            X_csr,
            W,
            H,
            n_components,
            init="custom",
            update_H=True,
            solver="mu",
            beta_loss=beta_loss,
            max_iter=n_iter,
            alpha_W=alpha,
            l1_ratio=l1_ratio,
            random_state=42,
        )

        assert_allclose(W1, W2, atol=1e-7)
        assert_allclose(H1, H2, atol=1e-7)

        # Compare with almost same beta_loss, since some values have a specific
        # behavior, but the results should be continuous w.r.t beta_loss
        beta_loss -= 1.0e-5
        W, H = W0.copy(), H0.copy()
        W3, H3, _ = non_negative_factorization(
            X_csr,
            W,
            H,
            n_components,
            init="custom",
            update_H=True,
            solver="mu",
            beta_loss=beta_loss,
            max_iter=n_iter,
            alpha_W=alpha,
            l1_ratio=l1_ratio,
            random_state=42,
        )

        assert_allclose(W1, W3, atol=1e-4)
        assert_allclose(H1, H3, atol=1e-4)


def test_nmf_negative_beta_loss():
    # Test that an error is raised if beta_loss < 0 and X contains zeros.
    # Test that the output has not NaN values when the input contains zeros.
    n_samples = 6
    n_features = 5
    n_components = 3

    rng = np.random.mtrand.RandomState(42)
    X = rng.randn(n_samples, n_features)
    np.clip(X, 0, None, out=X)
    X_csr = sp.csr_matrix(X)

    def _assert_nmf_no_nan(X, beta_loss):
        W, H, _ = non_negative_factorization(
            X,
            init="random",
            n_components=n_components,
            solver="mu",
            beta_loss=beta_loss,
            random_state=0,
            max_iter=1000,
        )
        assert not np.any(np.isnan(W))
        assert not np.any(np.isnan(H))

    msg = "When beta_loss <= 0 and X contains zeros, the solver may diverge."
    for beta_loss in (-0.6, 0.0):
        with pytest.raises(ValueError, match=msg):
            _assert_nmf_no_nan(X, beta_loss)
        _assert_nmf_no_nan(X + 1e-9, beta_loss)

    for beta_loss in (0.2, 1.0, 1.2, 2.0, 2.5):
        _assert_nmf_no_nan(X, beta_loss)
        _assert_nmf_no_nan(X_csr, beta_loss)


@pytest.mark.parametrize("beta_loss", [-0.5, 0.0])
def test_minibatch_nmf_negative_beta_loss(beta_loss):
    """Check that an error is raised if beta_loss < 0 and X contains zeros."""
    rng = np.random.RandomState(0)
    X = rng.normal(size=(6, 5))
    X[X < 0] = 0

    nmf = MiniBatchNMF(beta_loss=beta_loss, random_state=0)

    msg = "When beta_loss <= 0 and X contains zeros, the solver may diverge."
    with pytest.raises(ValueError, match=msg):
        nmf.fit(X)


@pytest.mark.parametrize(
    ["Estimator", "solver"],
    [[NMF, {"solver": "cd"}], [NMF, {"solver": "mu"}], [MiniBatchNMF, {}]],
)
def test_nmf_regularization(Estimator, solver):
    # Test the effect of L1 and L2 regularizations
    n_samples = 6
    n_features = 5
    n_components = 3
    rng = np.random.mtrand.RandomState(42)
    X = np.abs(rng.randn(n_samples, n_features))

    # L1 regularization should increase the number of zeros
    l1_ratio = 1.0
    regul = Estimator(
        n_components=n_components,
        alpha_W=0.5,
        l1_ratio=l1_ratio,
        random_state=42,
        **solver,
    )
    model = Estimator(
        n_components=n_components,
        alpha_W=0.0,
        l1_ratio=l1_ratio,
        random_state=42,
        **solver,
    )

    W_regul = regul.fit_transform(X)
    W_model = model.fit_transform(X)

    H_regul = regul.components_
    H_model = model.components_

    eps = np.finfo(np.float64).eps
    W_regul_n_zeros = W_regul[W_regul <= eps].size
    W_model_n_zeros = W_model[W_model <= eps].size
    H_regul_n_zeros = H_regul[H_regul <= eps].size
    H_model_n_zeros = H_model[H_model <= eps].size

    assert W_regul_n_zeros > W_model_n_zeros
    assert H_regul_n_zeros > H_model_n_zeros

    # L2 regularization should decrease the sum of the squared norm
    # of the matrices W and H
    l1_ratio = 0.0
    regul = Estimator(
        n_components=n_components,
        alpha_W=0.5,
        l1_ratio=l1_ratio,
        random_state=42,
        **solver,
    )
    model = Estimator(
        n_components=n_components,
        alpha_W=0.0,
        l1_ratio=l1_ratio,
        random_state=42,
        **solver,
    )

    W_regul = regul.fit_transform(X)
    W_model = model.fit_transform(X)

    H_regul = regul.components_
    H_model = model.components_

    assert (linalg.norm(W_model)) ** 2.0 + (linalg.norm(H_model)) ** 2.0 > (
        linalg.norm(W_regul)
    ) ** 2.0 + (linalg.norm(H_regul)) ** 2.0


@ignore_warnings(category=ConvergenceWarning)
@pytest.mark.parametrize("solver", ("cd", "mu"))
def test_nmf_decreasing(solver):
    # test that the objective function is decreasing at each iteration
    n_samples = 20
    n_features = 15
    n_components = 10
    alpha = 0.1
    l1_ratio = 0.5
    tol = 0.0

    # initialization
    rng = np.random.mtrand.RandomState(42)
    X = rng.randn(n_samples, n_features)
    np.abs(X, X)
    W0, H0 = nmf._initialize_nmf(X, n_components, init="random", random_state=42)

    for beta_loss in (-1.2, 0, 0.2, 1.0, 2.0, 2.5):
        if solver != "mu" and beta_loss != 2:
            # not implemented
            continue
        W, H = W0.copy(), H0.copy()
        previous_loss = None
        for _ in range(30):
            # one more iteration starting from the previous results
            W, H, _ = non_negative_factorization(
                X,
                W,
                H,
                beta_loss=beta_loss,
                init="custom",
                n_components=n_components,
                max_iter=1,
                alpha_W=alpha,
                solver=solver,
                tol=tol,
                l1_ratio=l1_ratio,
                verbose=0,
                random_state=0,
                update_H=True,
            )

            loss = (
                nmf._beta_divergence(X, W, H, beta_loss)
                + alpha * l1_ratio * n_features * W.sum()
                + alpha * l1_ratio * n_samples * H.sum()
                + alpha * (1 - l1_ratio) * n_features * (W**2).sum()
                + alpha * (1 - l1_ratio) * n_samples * (H**2).sum()
            )
            if previous_loss is not None:
                assert previous_loss > loss
            previous_loss = loss


def test_nmf_underflow():
    # Regression test for an underflow issue in _beta_divergence
    rng = np.random.RandomState(0)
    n_samples, n_features, n_components = 10, 2, 2
    X = np.abs(rng.randn(n_samples, n_features)) * 10
    W = np.abs(rng.randn(n_samples, n_components)) * 10
    H = np.abs(rng.randn(n_components, n_features))

    X[0, 0] = 0
    ref = nmf._beta_divergence(X, W, H, beta=1.0)
    X[0, 0] = 1e-323
    res = nmf._beta_divergence(X, W, H, beta=1.0)
    assert_almost_equal(res, ref)


@pytest.mark.parametrize(
    "dtype_in, dtype_out",
    [
        (np.float32, np.float32),
        (np.float64, np.float64),
        (np.int32, np.float64),
        (np.int64, np.float64),
    ],
)
@pytest.mark.parametrize(
    ["Estimator", "solver"],
    [[NMF, {"solver": "cd"}], [NMF, {"solver": "mu"}], [MiniBatchNMF, {}]],
)
def test_nmf_dtype_match(Estimator, solver, dtype_in, dtype_out):
    # Check that NMF preserves dtype (float32 and float64)
    X = np.random.RandomState(0).randn(20, 15).astype(dtype_in, copy=False)
    np.abs(X, out=X)

    nmf = Estimator(alpha_W=1.0, alpha_H=1.0, tol=1e-2, random_state=0, **solver)

    assert nmf.fit(X).transform(X).dtype == dtype_out
    assert nmf.fit_transform(X).dtype == dtype_out
    assert nmf.components_.dtype == dtype_out


@pytest.mark.parametrize(
    ["Estimator", "solver"],
    [[NMF, {"solver": "cd"}], [NMF, {"solver": "mu"}], [MiniBatchNMF, {}]],
)
def test_nmf_float32_float64_consistency(Estimator, solver):
    # Check that the result of NMF is the same between float32 and float64
    X = np.random.RandomState(0).randn(50, 7)
    np.abs(X, out=X)
    nmf32 = Estimator(random_state=0, tol=1e-3, **solver)
    W32 = nmf32.fit_transform(X.astype(np.float32))
    nmf64 = Estimator(random_state=0, tol=1e-3, **solver)
    W64 = nmf64.fit_transform(X)

    assert_allclose(W32, W64, atol=1e-5)


@pytest.mark.parametrize("Estimator", [NMF, MiniBatchNMF])
def test_nmf_custom_init_dtype_error(Estimator):
    # Check that an error is raise if custom H and/or W don't have the same
    # dtype as X.
    rng = np.random.RandomState(0)
    X = rng.random_sample((20, 15))
    H = rng.random_sample((15, 15)).astype(np.float32)
    W = rng.random_sample((20, 15))

    with pytest.raises(TypeError, match="should have the same dtype as X"):
        Estimator(init="custom").fit(X, H=H, W=W)

    with pytest.raises(TypeError, match="should have the same dtype as X"):
        non_negative_factorization(X, H=H, update_H=False)


@pytest.mark.parametrize("beta_loss", [-0.5, 0, 0.5, 1, 1.5, 2, 2.5])
def test_nmf_minibatchnmf_equivalence(beta_loss):
    # Test that MiniBatchNMF is equivalent to NMF when batch_size = n_samples and
    # forget_factor 0.0 (stopping criterion put aside)
    rng = np.random.mtrand.RandomState(42)
    X = np.abs(rng.randn(48, 5))

    nmf = NMF(
        n_components=5,
        beta_loss=beta_loss,
        solver="mu",
        random_state=0,
        tol=0,
    )
    mbnmf = MiniBatchNMF(
        n_components=5,
        beta_loss=beta_loss,
        random_state=0,
        tol=0,
        max_no_improvement=None,
        batch_size=X.shape[0],
        forget_factor=0.0,
    )
    W = nmf.fit_transform(X)
    mbW = mbnmf.fit_transform(X)
    assert_allclose(W, mbW)


def test_minibatch_nmf_partial_fit():
    # Check fit / partial_fit equivalence. Applicable only with fresh restarts.
    rng = np.random.mtrand.RandomState(42)
    X = np.abs(rng.randn(100, 5))

    n_components = 5
    batch_size = 10
    max_iter = 2

    mbnmf1 = MiniBatchNMF(
        n_components=n_components,
        init="custom",
        random_state=0,
        max_iter=max_iter,
        batch_size=batch_size,
        tol=0,
        max_no_improvement=None,
        fresh_restarts=False,
    )
    mbnmf2 = MiniBatchNMF(n_components=n_components, init="custom", random_state=0)

    # Force the same init of H (W is recomputed anyway) to be able to compare results.
    W, H = nmf._initialize_nmf(
        X, n_components=n_components, init="random", random_state=0
    )

    mbnmf1.fit(X, W=W, H=H)
    for i in range(max_iter):
        for j in range(batch_size):
            mbnmf2.partial_fit(X[j : j + batch_size], W=W[:batch_size], H=H)

    assert mbnmf1.n_steps_ == mbnmf2.n_steps_
    assert_allclose(mbnmf1.components_, mbnmf2.components_)


def test_feature_names_out():
    """Check feature names out for NMF."""
    random_state = np.random.RandomState(0)
    X = np.abs(random_state.randn(10, 4))
    nmf = NMF(n_components=3).fit(X)

    names = nmf.get_feature_names_out()
    assert_array_equal([f"nmf{i}" for i in range(3)], names)


def test_minibatch_nmf_verbose():
    # Check verbose mode of MiniBatchNMF for better coverage.
    A = np.random.RandomState(0).random_sample((100, 10))
    nmf = MiniBatchNMF(tol=1e-2, random_state=0, verbose=1)
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        nmf.fit(A)
    finally:
        sys.stdout = old_stdout


# TODO(1.5): remove this test
def test_NMF_inverse_transform_W_deprecation():
    rng = np.random.mtrand.RandomState(42)
    A = np.abs(rng.randn(6, 5))
    est = NMF(
        n_components=3,
        init="random",
        random_state=0,
        tol=1e-6,
    )
    Xt = est.fit_transform(A)

    with pytest.raises(TypeError, match="Missing required positional argument"):
        est.inverse_transform()

    with pytest.raises(ValueError, match="Please provide only"):
        est.inverse_transform(Xt=Xt, W=Xt)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("error")
        est.inverse_transform(Xt)

    with pytest.warns(FutureWarning, match="Input argument `W` was renamed to `Xt`"):
        est.inverse_transform(W=Xt)
