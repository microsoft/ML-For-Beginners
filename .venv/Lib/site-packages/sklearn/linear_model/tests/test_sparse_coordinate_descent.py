import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose

from sklearn.datasets import make_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso, LassoCV
from sklearn.utils._testing import (
    assert_almost_equal,
    assert_array_almost_equal,
    create_memmap_backed_data,
    ignore_warnings,
)
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, LIL_CONTAINERS


def test_sparse_coef():
    # Check that the sparse_coef property works
    clf = ElasticNet()
    clf.coef_ = [1, 2, 3]

    assert sp.issparse(clf.sparse_coef_)
    assert clf.sparse_coef_.toarray().tolist()[0] == clf.coef_


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_lasso_zero(csc_container):
    # Check that the sparse lasso can handle zero data without crashing
    X = csc_container((3, 1))
    y = [0, 0, 0]
    T = np.array([[1], [2], [3]])
    clf = Lasso().fit(X, y)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0])
    assert_array_almost_equal(pred, [0, 0, 0])
    assert_almost_equal(clf.dual_gap_, 0)


@pytest.mark.parametrize("with_sample_weight", [True, False])
@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_enet_toy_list_input(with_sample_weight, csc_container):
    # Test ElasticNet for various values of alpha and l1_ratio with list X

    X = np.array([[-1], [0], [1]])
    X = csc_container(X)
    Y = [-1, 0, 1]  # just a straight line
    T = np.array([[2], [3], [4]])  # test sample
    if with_sample_weight:
        sw = np.array([2.0, 2, 2])
    else:
        sw = None

    # this should be the same as unregularized least squares
    clf = ElasticNet(alpha=0, l1_ratio=1.0)
    # catch warning about alpha=0.
    # this is discouraged but should work.
    ignore_warnings(clf.fit)(X, Y, sample_weight=sw)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [1])
    assert_array_almost_equal(pred, [2, 3, 4])
    assert_almost_equal(clf.dual_gap_, 0)

    clf = ElasticNet(alpha=0.5, l1_ratio=0.3)
    clf.fit(X, Y, sample_weight=sw)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.50819], decimal=3)
    assert_array_almost_equal(pred, [1.0163, 1.5245, 2.0327], decimal=3)
    assert_almost_equal(clf.dual_gap_, 0)

    clf = ElasticNet(alpha=0.5, l1_ratio=0.5)
    clf.fit(X, Y, sample_weight=sw)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.45454], 3)
    assert_array_almost_equal(pred, [0.9090, 1.3636, 1.8181], 3)
    assert_almost_equal(clf.dual_gap_, 0)


@pytest.mark.parametrize("lil_container", LIL_CONTAINERS)
def test_enet_toy_explicit_sparse_input(lil_container):
    # Test ElasticNet for various values of alpha and l1_ratio with sparse X
    f = ignore_warnings
    # training samples
    X = lil_container((3, 1))
    X[0, 0] = -1
    # X[1, 0] = 0
    X[2, 0] = 1
    Y = [-1, 0, 1]  # just a straight line (the identity function)

    # test samples
    T = lil_container((3, 1))
    T[0, 0] = 2
    T[1, 0] = 3
    T[2, 0] = 4

    # this should be the same as lasso
    clf = ElasticNet(alpha=0, l1_ratio=1.0)
    f(clf.fit)(X, Y)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [1])
    assert_array_almost_equal(pred, [2, 3, 4])
    assert_almost_equal(clf.dual_gap_, 0)

    clf = ElasticNet(alpha=0.5, l1_ratio=0.3)
    clf.fit(X, Y)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.50819], decimal=3)
    assert_array_almost_equal(pred, [1.0163, 1.5245, 2.0327], decimal=3)
    assert_almost_equal(clf.dual_gap_, 0)

    clf = ElasticNet(alpha=0.5, l1_ratio=0.5)
    clf.fit(X, Y)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.45454], 3)
    assert_array_almost_equal(pred, [0.9090, 1.3636, 1.8181], 3)
    assert_almost_equal(clf.dual_gap_, 0)


def make_sparse_data(
    sparse_container,
    n_samples=100,
    n_features=100,
    n_informative=10,
    seed=42,
    positive=False,
    n_targets=1,
):
    random_state = np.random.RandomState(seed)

    # build an ill-posed linear regression problem with many noisy features and
    # comparatively few samples

    # generate a ground truth model
    w = random_state.randn(n_features, n_targets)
    w[n_informative:] = 0.0  # only the top features are impacting the model
    if positive:
        w = np.abs(w)

    X = random_state.randn(n_samples, n_features)
    rnd = random_state.uniform(size=(n_samples, n_features))
    X[rnd > 0.5] = 0.0  # 50% of zeros in input signal

    # generate training ground truth labels
    y = np.dot(X, w)
    X = sparse_container(X)
    if n_targets == 1:
        y = np.ravel(y)
    return X, y


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
@pytest.mark.parametrize(
    "alpha, fit_intercept, positive",
    [(0.1, False, False), (0.1, True, False), (1e-3, False, True), (1e-3, True, True)],
)
def test_sparse_enet_not_as_toy_dataset(csc_container, alpha, fit_intercept, positive):
    n_samples, n_features, max_iter = 100, 100, 1000
    n_informative = 10

    X, y = make_sparse_data(
        csc_container, n_samples, n_features, n_informative, positive=positive
    )

    X_train, X_test = X[n_samples // 2 :], X[: n_samples // 2]
    y_train, y_test = y[n_samples // 2 :], y[: n_samples // 2]

    s_clf = ElasticNet(
        alpha=alpha,
        l1_ratio=0.8,
        fit_intercept=fit_intercept,
        max_iter=max_iter,
        tol=1e-7,
        positive=positive,
        warm_start=True,
    )
    s_clf.fit(X_train, y_train)

    assert_almost_equal(s_clf.dual_gap_, 0, 4)
    assert s_clf.score(X_test, y_test) > 0.85

    # check the convergence is the same as the dense version
    d_clf = ElasticNet(
        alpha=alpha,
        l1_ratio=0.8,
        fit_intercept=fit_intercept,
        max_iter=max_iter,
        tol=1e-7,
        positive=positive,
        warm_start=True,
    )
    d_clf.fit(X_train.toarray(), y_train)

    assert_almost_equal(d_clf.dual_gap_, 0, 4)
    assert d_clf.score(X_test, y_test) > 0.85

    assert_almost_equal(s_clf.coef_, d_clf.coef_, 5)
    assert_almost_equal(s_clf.intercept_, d_clf.intercept_, 5)

    # check that the coefs are sparse
    assert np.sum(s_clf.coef_ != 0.0) < 2 * n_informative


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_sparse_lasso_not_as_toy_dataset(csc_container):
    n_samples = 100
    max_iter = 1000
    n_informative = 10
    X, y = make_sparse_data(
        csc_container, n_samples=n_samples, n_informative=n_informative
    )

    X_train, X_test = X[n_samples // 2 :], X[: n_samples // 2]
    y_train, y_test = y[n_samples // 2 :], y[: n_samples // 2]

    s_clf = Lasso(alpha=0.1, fit_intercept=False, max_iter=max_iter, tol=1e-7)
    s_clf.fit(X_train, y_train)
    assert_almost_equal(s_clf.dual_gap_, 0, 4)
    assert s_clf.score(X_test, y_test) > 0.85

    # check the convergence is the same as the dense version
    d_clf = Lasso(alpha=0.1, fit_intercept=False, max_iter=max_iter, tol=1e-7)
    d_clf.fit(X_train.toarray(), y_train)
    assert_almost_equal(d_clf.dual_gap_, 0, 4)
    assert d_clf.score(X_test, y_test) > 0.85

    # check that the coefs are sparse
    assert np.sum(s_clf.coef_ != 0.0) == n_informative


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_enet_multitarget(csc_container):
    n_targets = 3
    X, y = make_sparse_data(csc_container, n_targets=n_targets)

    estimator = ElasticNet(alpha=0.01, precompute=False)
    # XXX: There is a bug when precompute is not False!
    estimator.fit(X, y)
    coef, intercept, dual_gap = (
        estimator.coef_,
        estimator.intercept_,
        estimator.dual_gap_,
    )

    for k in range(n_targets):
        estimator.fit(X, y[:, k])
        assert_array_almost_equal(coef[k, :], estimator.coef_)
        assert_array_almost_equal(intercept[k], estimator.intercept_)
        assert_array_almost_equal(dual_gap[k], estimator.dual_gap_)


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_path_parameters(csc_container):
    X, y = make_sparse_data(csc_container)
    max_iter = 50
    n_alphas = 10
    clf = ElasticNetCV(
        n_alphas=n_alphas,
        eps=1e-3,
        max_iter=max_iter,
        l1_ratio=0.5,
        fit_intercept=False,
    )
    ignore_warnings(clf.fit)(X, y)  # new params
    assert_almost_equal(0.5, clf.l1_ratio)
    assert n_alphas == clf.n_alphas
    assert n_alphas == len(clf.alphas_)
    sparse_mse_path = clf.mse_path_
    ignore_warnings(clf.fit)(X.toarray(), y)  # compare with dense data
    assert_almost_equal(clf.mse_path_, sparse_mse_path)


@pytest.mark.parametrize("Model", [Lasso, ElasticNet, LassoCV, ElasticNetCV])
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("n_samples, n_features", [(24, 6), (6, 24)])
@pytest.mark.parametrize("with_sample_weight", [True, False])
@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_sparse_dense_equality(
    Model, fit_intercept, n_samples, n_features, with_sample_weight, csc_container
):
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        effective_rank=n_features // 2,
        n_informative=n_features // 2,
        bias=4 * fit_intercept,
        noise=1,
        random_state=42,
    )
    if with_sample_weight:
        sw = np.abs(np.random.RandomState(42).normal(scale=10, size=y.shape))
    else:
        sw = None
    Xs = csc_container(X)
    params = {"fit_intercept": fit_intercept}
    reg_dense = Model(**params).fit(X, y, sample_weight=sw)
    reg_sparse = Model(**params).fit(Xs, y, sample_weight=sw)
    if fit_intercept:
        assert reg_sparse.intercept_ == pytest.approx(reg_dense.intercept_)
        # balance property
        assert np.average(reg_sparse.predict(X), weights=sw) == pytest.approx(
            np.average(y, weights=sw)
        )
    assert_allclose(reg_sparse.coef_, reg_dense.coef_)


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_same_output_sparse_dense_lasso_and_enet_cv(csc_container):
    X, y = make_sparse_data(csc_container, n_samples=40, n_features=10)
    clfs = ElasticNetCV(max_iter=100)
    clfs.fit(X, y)
    clfd = ElasticNetCV(max_iter=100)
    clfd.fit(X.toarray(), y)
    assert_almost_equal(clfs.alpha_, clfd.alpha_, 7)
    assert_almost_equal(clfs.intercept_, clfd.intercept_, 7)
    assert_array_almost_equal(clfs.mse_path_, clfd.mse_path_)
    assert_array_almost_equal(clfs.alphas_, clfd.alphas_)

    clfs = LassoCV(max_iter=100, cv=4)
    clfs.fit(X, y)
    clfd = LassoCV(max_iter=100, cv=4)
    clfd.fit(X.toarray(), y)
    assert_almost_equal(clfs.alpha_, clfd.alpha_, 7)
    assert_almost_equal(clfs.intercept_, clfd.intercept_, 7)
    assert_array_almost_equal(clfs.mse_path_, clfd.mse_path_)
    assert_array_almost_equal(clfs.alphas_, clfd.alphas_)


@pytest.mark.parametrize("coo_container", COO_CONTAINERS)
def test_same_multiple_output_sparse_dense(coo_container):
    l = ElasticNet()
    X = [
        [0, 1, 2, 3, 4],
        [0, 2, 5, 8, 11],
        [9, 10, 11, 12, 13],
        [10, 11, 12, 13, 14],
    ]
    y = [
        [1, 2, 3, 4, 5],
        [1, 3, 6, 9, 12],
        [10, 11, 12, 13, 14],
        [11, 12, 13, 14, 15],
    ]
    l.fit(X, y)
    sample = np.array([1, 2, 3, 4, 5]).reshape(1, -1)
    predict_dense = l.predict(sample)

    l_sp = ElasticNet()
    X_sp = coo_container(X)
    l_sp.fit(X_sp, y)
    sample_sparse = coo_container(sample)
    predict_sparse = l_sp.predict(sample_sparse)

    assert_array_almost_equal(predict_sparse, predict_dense)


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_sparse_enet_coordinate_descent(csc_container):
    """Test that a warning is issued if model does not converge"""
    clf = Lasso(max_iter=2)
    n_samples = 5
    n_features = 2
    X = csc_container((n_samples, n_features)) * 1e50
    y = np.ones(n_samples)
    warning_message = (
        "Objective did not converge. You might want "
        "to increase the number of iterations."
    )
    with pytest.warns(ConvergenceWarning, match=warning_message):
        clf.fit(X, y)


@pytest.mark.parametrize("copy_X", (True, False))
def test_sparse_read_only_buffer(copy_X):
    """Test that sparse coordinate descent works for read-only buffers"""
    rng = np.random.RandomState(0)

    clf = ElasticNet(alpha=0.1, copy_X=copy_X, random_state=rng)
    X = sp.random(100, 20, format="csc", random_state=rng)

    # Make X.data read-only
    X.data = create_memmap_backed_data(X.data)

    y = rng.rand(100)
    clf.fit(X, y)
