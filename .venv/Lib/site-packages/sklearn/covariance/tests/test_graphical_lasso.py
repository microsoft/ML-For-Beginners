""" Test the graphical_lasso module.
"""
import sys
from io import StringIO

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import linalg

from sklearn import datasets
from sklearn.covariance import (
    GraphicalLasso,
    GraphicalLassoCV,
    empirical_covariance,
    graphical_lasso,
)
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
    _convert_container,
    assert_array_almost_equal,
    assert_array_less,
)


def test_graphical_lasso(random_state=0):
    # Sample data from a sparse multivariate normal
    dim = 20
    n_samples = 100
    random_state = check_random_state(random_state)
    prec = make_sparse_spd_matrix(dim, alpha=0.95, random_state=random_state)
    cov = linalg.inv(prec)
    X = random_state.multivariate_normal(np.zeros(dim), cov, size=n_samples)
    emp_cov = empirical_covariance(X)

    for alpha in (0.0, 0.1, 0.25):
        covs = dict()
        icovs = dict()
        for method in ("cd", "lars"):
            cov_, icov_, costs = graphical_lasso(
                emp_cov, return_costs=True, alpha=alpha, mode=method
            )
            covs[method] = cov_
            icovs[method] = icov_
            costs, dual_gap = np.array(costs).T
            # Check that the costs always decrease (doesn't hold if alpha == 0)
            if not alpha == 0:
                assert_array_less(np.diff(costs), 0)
        # Check that the 2 approaches give similar results
        assert_array_almost_equal(covs["cd"], covs["lars"], decimal=4)
        assert_array_almost_equal(icovs["cd"], icovs["lars"], decimal=4)

    # Smoke test the estimator
    model = GraphicalLasso(alpha=0.25).fit(X)
    model.score(X)
    assert_array_almost_equal(model.covariance_, covs["cd"], decimal=4)
    assert_array_almost_equal(model.covariance_, covs["lars"], decimal=4)

    # For a centered matrix, assume_centered could be chosen True or False
    # Check that this returns indeed the same result for centered data
    Z = X - X.mean(0)
    precs = list()
    for assume_centered in (False, True):
        prec_ = GraphicalLasso(assume_centered=assume_centered).fit(Z).precision_
        precs.append(prec_)
    assert_array_almost_equal(precs[0], precs[1])


def test_graphical_lasso_when_alpha_equals_0():
    """Test graphical_lasso's early return condition when alpha=0."""
    X = np.random.randn(100, 10)
    emp_cov = empirical_covariance(X, assume_centered=True)

    model = GraphicalLasso(alpha=0, covariance="precomputed").fit(emp_cov)
    assert_allclose(model.precision_, np.linalg.inv(emp_cov))

    _, precision = graphical_lasso(emp_cov, alpha=0)
    assert_allclose(precision, np.linalg.inv(emp_cov))


@pytest.mark.parametrize("mode", ["cd", "lars"])
def test_graphical_lasso_n_iter(mode):
    X, _ = datasets.make_classification(n_samples=5_000, n_features=20, random_state=0)
    emp_cov = empirical_covariance(X)

    _, _, n_iter = graphical_lasso(
        emp_cov, 0.2, mode=mode, max_iter=2, return_n_iter=True
    )
    assert n_iter == 2


def test_graphical_lasso_iris():
    # Hard-coded solution from R glasso package for alpha=1.0
    # (need to set penalize.diagonal to FALSE)
    cov_R = np.array(
        [
            [0.68112222, 0.0000000, 0.265820, 0.02464314],
            [0.00000000, 0.1887129, 0.000000, 0.00000000],
            [0.26582000, 0.0000000, 3.095503, 0.28697200],
            [0.02464314, 0.0000000, 0.286972, 0.57713289],
        ]
    )
    icov_R = np.array(
        [
            [1.5190747, 0.000000, -0.1304475, 0.0000000],
            [0.0000000, 5.299055, 0.0000000, 0.0000000],
            [-0.1304475, 0.000000, 0.3498624, -0.1683946],
            [0.0000000, 0.000000, -0.1683946, 1.8164353],
        ]
    )
    X = datasets.load_iris().data
    emp_cov = empirical_covariance(X)
    for method in ("cd", "lars"):
        cov, icov = graphical_lasso(emp_cov, alpha=1.0, return_costs=False, mode=method)
        assert_array_almost_equal(cov, cov_R)
        assert_array_almost_equal(icov, icov_R)


def test_graph_lasso_2D():
    # Hard-coded solution from Python skggm package
    # obtained by calling `quic(emp_cov, lam=.1, tol=1e-8)`
    cov_skggm = np.array([[3.09550269, 1.186972], [1.186972, 0.57713289]])

    icov_skggm = np.array([[1.52836773, -3.14334831], [-3.14334831, 8.19753385]])
    X = datasets.load_iris().data[:, 2:]
    emp_cov = empirical_covariance(X)
    for method in ("cd", "lars"):
        cov, icov = graphical_lasso(emp_cov, alpha=0.1, return_costs=False, mode=method)
        assert_array_almost_equal(cov, cov_skggm)
        assert_array_almost_equal(icov, icov_skggm)


def test_graphical_lasso_iris_singular():
    # Small subset of rows to test the rank-deficient case
    # Need to choose samples such that none of the variances are zero
    indices = np.arange(10, 13)

    # Hard-coded solution from R glasso package for alpha=0.01
    cov_R = np.array(
        [
            [0.08, 0.056666662595, 0.00229729713223, 0.00153153142149],
            [0.056666662595, 0.082222222222, 0.00333333333333, 0.00222222222222],
            [0.002297297132, 0.003333333333, 0.00666666666667, 0.00009009009009],
            [0.001531531421, 0.002222222222, 0.00009009009009, 0.00222222222222],
        ]
    )
    icov_R = np.array(
        [
            [24.42244057, -16.831679593, 0.0, 0.0],
            [-16.83168201, 24.351841681, -6.206896552, -12.5],
            [0.0, -6.206896171, 153.103448276, 0.0],
            [0.0, -12.499999143, 0.0, 462.5],
        ]
    )
    X = datasets.load_iris().data[indices, :]
    emp_cov = empirical_covariance(X)
    for method in ("cd", "lars"):
        cov, icov = graphical_lasso(
            emp_cov, alpha=0.01, return_costs=False, mode=method
        )
        assert_array_almost_equal(cov, cov_R, decimal=5)
        assert_array_almost_equal(icov, icov_R, decimal=5)


def test_graphical_lasso_cv(random_state=1):
    # Sample data from a sparse multivariate normal
    dim = 5
    n_samples = 6
    random_state = check_random_state(random_state)
    prec = make_sparse_spd_matrix(dim, alpha=0.96, random_state=random_state)
    cov = linalg.inv(prec)
    X = random_state.multivariate_normal(np.zeros(dim), cov, size=n_samples)
    # Capture stdout, to smoke test the verbose mode
    orig_stdout = sys.stdout
    try:
        sys.stdout = StringIO()
        # We need verbose very high so that Parallel prints on stdout
        GraphicalLassoCV(verbose=100, alphas=5, tol=1e-1).fit(X)
    finally:
        sys.stdout = orig_stdout


@pytest.mark.parametrize("alphas_container_type", ["list", "tuple", "array"])
def test_graphical_lasso_cv_alphas_iterable(alphas_container_type):
    """Check that we can pass an array-like to `alphas`.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/22489
    """
    true_cov = np.array(
        [
            [0.8, 0.0, 0.2, 0.0],
            [0.0, 0.4, 0.0, 0.0],
            [0.2, 0.0, 0.3, 0.1],
            [0.0, 0.0, 0.1, 0.7],
        ]
    )
    rng = np.random.RandomState(0)
    X = rng.multivariate_normal(mean=[0, 0, 0, 0], cov=true_cov, size=200)
    alphas = _convert_container([0.02, 0.03], alphas_container_type)
    GraphicalLassoCV(alphas=alphas, tol=1e-1, n_jobs=1).fit(X)


@pytest.mark.parametrize(
    "alphas,err_type,err_msg",
    [
        ([-0.02, 0.03], ValueError, "must be > 0"),
        ([0, 0.03], ValueError, "must be > 0"),
        (["not_number", 0.03], TypeError, "must be an instance of float"),
    ],
)
def test_graphical_lasso_cv_alphas_invalid_array(alphas, err_type, err_msg):
    """Check that if an array-like containing a value
    outside of (0, inf] is passed to `alphas`, a ValueError is raised.
    Check if a string is passed, a TypeError is raised.
    """
    true_cov = np.array(
        [
            [0.8, 0.0, 0.2, 0.0],
            [0.0, 0.4, 0.0, 0.0],
            [0.2, 0.0, 0.3, 0.1],
            [0.0, 0.0, 0.1, 0.7],
        ]
    )
    rng = np.random.RandomState(0)
    X = rng.multivariate_normal(mean=[0, 0, 0, 0], cov=true_cov, size=200)

    with pytest.raises(err_type, match=err_msg):
        GraphicalLassoCV(alphas=alphas, tol=1e-1, n_jobs=1).fit(X)


def test_graphical_lasso_cv_scores():
    splits = 4
    n_alphas = 5
    n_refinements = 3
    true_cov = np.array(
        [
            [0.8, 0.0, 0.2, 0.0],
            [0.0, 0.4, 0.0, 0.0],
            [0.2, 0.0, 0.3, 0.1],
            [0.0, 0.0, 0.1, 0.7],
        ]
    )
    rng = np.random.RandomState(0)
    X = rng.multivariate_normal(mean=[0, 0, 0, 0], cov=true_cov, size=200)
    cov = GraphicalLassoCV(cv=splits, alphas=n_alphas, n_refinements=n_refinements).fit(
        X
    )

    cv_results = cov.cv_results_
    # alpha and one for each split

    total_alphas = n_refinements * n_alphas + 1
    keys = ["alphas"]
    split_keys = [f"split{i}_test_score" for i in range(splits)]
    for key in keys + split_keys:
        assert key in cv_results
        assert len(cv_results[key]) == total_alphas

    cv_scores = np.asarray([cov.cv_results_[key] for key in split_keys])
    expected_mean = cv_scores.mean(axis=0)
    expected_std = cv_scores.std(axis=0)

    assert_allclose(cov.cv_results_["mean_test_score"], expected_mean)
    assert_allclose(cov.cv_results_["std_test_score"], expected_std)


# TODO(1.5): remove in 1.5
def test_graphical_lasso_cov_init_deprecation():
    """Check that we raise a deprecation warning if providing `cov_init` in
    `graphical_lasso`."""
    rng, dim, n_samples = np.random.RandomState(0), 20, 100
    prec = make_sparse_spd_matrix(dim, alpha=0.95, random_state=0)
    cov = linalg.inv(prec)
    X = rng.multivariate_normal(np.zeros(dim), cov, size=n_samples)

    emp_cov = empirical_covariance(X)
    with pytest.warns(FutureWarning, match="cov_init parameter is deprecated"):
        graphical_lasso(emp_cov, alpha=0.1, cov_init=emp_cov)
