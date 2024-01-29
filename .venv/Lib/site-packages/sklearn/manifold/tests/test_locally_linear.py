from itertools import product

import numpy as np
import pytest
from scipy import linalg

from sklearn import manifold, neighbors
from sklearn.datasets import make_blobs
from sklearn.manifold._locally_linear import barycenter_kneighbors_graph
from sklearn.utils._testing import (
    assert_allclose,
    assert_array_equal,
    ignore_warnings,
)

eigen_solvers = ["dense", "arpack"]


# ----------------------------------------------------------------------
# Test utility routines
def test_barycenter_kneighbors_graph(global_dtype):
    X = np.array([[0, 1], [1.01, 1.0], [2, 0]], dtype=global_dtype)

    graph = barycenter_kneighbors_graph(X, 1)
    expected_graph = np.array(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=global_dtype
    )

    assert graph.dtype == global_dtype

    assert_allclose(graph.toarray(), expected_graph)

    graph = barycenter_kneighbors_graph(X, 2)
    # check that columns sum to one
    assert_allclose(np.sum(graph.toarray(), axis=1), np.ones(3))
    pred = np.dot(graph.toarray(), X)
    assert linalg.norm(pred - X) / X.shape[0] < 1


# ----------------------------------------------------------------------
# Test LLE by computing the reconstruction error on some manifolds.


def test_lle_simple_grid(global_dtype):
    # note: ARPACK is numerically unstable, so this test will fail for
    #       some random seeds.  We choose 42 because the tests pass.
    #       for arm64 platforms 2 makes the test fail.
    # TODO: rewrite this test to make less sensitive to the random seed,
    # irrespective of the platform.
    rng = np.random.RandomState(42)

    # grid of equidistant points in 2D, n_components = n_dim
    X = np.array(list(product(range(5), repeat=2)))
    X = X + 1e-10 * rng.uniform(size=X.shape)
    X = X.astype(global_dtype, copy=False)

    n_components = 2
    clf = manifold.LocallyLinearEmbedding(
        n_neighbors=5, n_components=n_components, random_state=rng
    )
    tol = 0.1

    N = barycenter_kneighbors_graph(X, clf.n_neighbors).toarray()
    reconstruction_error = linalg.norm(np.dot(N, X) - X, "fro")
    assert reconstruction_error < tol

    for solver in eigen_solvers:
        clf.set_params(eigen_solver=solver)
        clf.fit(X)
        assert clf.embedding_.shape[1] == n_components
        reconstruction_error = (
            linalg.norm(np.dot(N, clf.embedding_) - clf.embedding_, "fro") ** 2
        )

        assert reconstruction_error < tol
        assert_allclose(clf.reconstruction_error_, reconstruction_error, atol=1e-1)

    # re-embed a noisy version of X using the transform method
    noise = rng.randn(*X.shape).astype(global_dtype, copy=False) / 100
    X_reembedded = clf.transform(X + noise)
    assert linalg.norm(X_reembedded - clf.embedding_) < tol


@pytest.mark.parametrize("method", ["standard", "hessian", "modified", "ltsa"])
@pytest.mark.parametrize("solver", eigen_solvers)
def test_lle_manifold(global_dtype, method, solver):
    rng = np.random.RandomState(0)
    # similar test on a slightly more complex manifold
    X = np.array(list(product(np.arange(18), repeat=2)))
    X = np.c_[X, X[:, 0] ** 2 / 18]
    X = X + 1e-10 * rng.uniform(size=X.shape)
    X = X.astype(global_dtype, copy=False)
    n_components = 2

    clf = manifold.LocallyLinearEmbedding(
        n_neighbors=6, n_components=n_components, method=method, random_state=0
    )
    tol = 1.5 if method == "standard" else 3

    N = barycenter_kneighbors_graph(X, clf.n_neighbors).toarray()
    reconstruction_error = linalg.norm(np.dot(N, X) - X)
    assert reconstruction_error < tol

    clf.set_params(eigen_solver=solver)
    clf.fit(X)
    assert clf.embedding_.shape[1] == n_components
    reconstruction_error = (
        linalg.norm(np.dot(N, clf.embedding_) - clf.embedding_, "fro") ** 2
    )
    details = "solver: %s, method: %s" % (solver, method)
    assert reconstruction_error < tol, details
    assert (
        np.abs(clf.reconstruction_error_ - reconstruction_error)
        < tol * reconstruction_error
    ), details


def test_pipeline():
    # check that LocallyLinearEmbedding works fine as a Pipeline
    # only checks that no error is raised.
    # TODO check that it actually does something useful
    from sklearn import datasets, pipeline

    X, y = datasets.make_blobs(random_state=0)
    clf = pipeline.Pipeline(
        [
            ("filter", manifold.LocallyLinearEmbedding(random_state=0)),
            ("clf", neighbors.KNeighborsClassifier()),
        ]
    )
    clf.fit(X, y)
    assert 0.9 < clf.score(X, y)


# Test the error raised when the weight matrix is singular
def test_singular_matrix():
    M = np.ones((200, 3))
    f = ignore_warnings
    with pytest.raises(ValueError, match="Error in determining null-space with ARPACK"):
        f(
            manifold.locally_linear_embedding(
                M,
                n_neighbors=2,
                n_components=1,
                method="standard",
                eigen_solver="arpack",
            )
        )


# regression test for #6033
def test_integer_input():
    rand = np.random.RandomState(0)
    X = rand.randint(0, 100, size=(20, 3))

    for method in ["standard", "hessian", "modified", "ltsa"]:
        clf = manifold.LocallyLinearEmbedding(method=method, n_neighbors=10)
        clf.fit(X)  # this previously raised a TypeError


def test_get_feature_names_out():
    """Check get_feature_names_out for LocallyLinearEmbedding."""
    X, y = make_blobs(random_state=0, n_features=4)
    n_components = 2

    iso = manifold.LocallyLinearEmbedding(n_components=n_components)
    iso.fit(X)
    names = iso.get_feature_names_out()
    assert_array_equal(
        [f"locallylinearembedding{i}" for i in range(n_components)], names
    )
