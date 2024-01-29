import math
from itertools import product

import numpy as np
import pytest
from scipy.sparse import rand as sparse_rand

from sklearn import clone, datasets, manifold, neighbors, pipeline, preprocessing
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils._testing import (
    assert_allclose,
    assert_allclose_dense_sparse,
    assert_array_equal,
)
from sklearn.utils.fixes import CSR_CONTAINERS

eigen_solvers = ["auto", "dense", "arpack"]
path_methods = ["auto", "FW", "D"]


def create_sample_data(dtype, n_pts=25, add_noise=False):
    # grid of equidistant points in 2D, n_components = n_dim
    n_per_side = int(math.sqrt(n_pts))
    X = np.array(list(product(range(n_per_side), repeat=2))).astype(dtype, copy=False)
    if add_noise:
        # add noise in a third dimension
        rng = np.random.RandomState(0)
        noise = 0.1 * rng.randn(n_pts, 1).astype(dtype, copy=False)
        X = np.concatenate((X, noise), 1)
    return X


@pytest.mark.parametrize("n_neighbors, radius", [(24, None), (None, np.inf)])
@pytest.mark.parametrize("eigen_solver", eigen_solvers)
@pytest.mark.parametrize("path_method", path_methods)
def test_isomap_simple_grid(
    global_dtype, n_neighbors, radius, eigen_solver, path_method
):
    # Isomap should preserve distances when all neighbors are used
    n_pts = 25
    X = create_sample_data(global_dtype, n_pts=n_pts, add_noise=False)

    # distances from each point to all others
    if n_neighbors is not None:
        G = neighbors.kneighbors_graph(X, n_neighbors, mode="distance")
    else:
        G = neighbors.radius_neighbors_graph(X, radius, mode="distance")

    clf = manifold.Isomap(
        n_neighbors=n_neighbors,
        radius=radius,
        n_components=2,
        eigen_solver=eigen_solver,
        path_method=path_method,
    )
    clf.fit(X)

    if n_neighbors is not None:
        G_iso = neighbors.kneighbors_graph(clf.embedding_, n_neighbors, mode="distance")
    else:
        G_iso = neighbors.radius_neighbors_graph(
            clf.embedding_, radius, mode="distance"
        )
    atol = 1e-5 if global_dtype == np.float32 else 0
    assert_allclose_dense_sparse(G, G_iso, atol=atol)


@pytest.mark.parametrize("n_neighbors, radius", [(24, None), (None, np.inf)])
@pytest.mark.parametrize("eigen_solver", eigen_solvers)
@pytest.mark.parametrize("path_method", path_methods)
def test_isomap_reconstruction_error(
    global_dtype, n_neighbors, radius, eigen_solver, path_method
):
    if global_dtype is np.float32:
        pytest.skip(
            "Skipping test due to numerical instabilities on float32 data"
            "from KernelCenterer used in the reconstruction_error method"
        )

    # Same setup as in test_isomap_simple_grid, with an added dimension
    n_pts = 25
    X = create_sample_data(global_dtype, n_pts=n_pts, add_noise=True)

    # compute input kernel
    if n_neighbors is not None:
        G = neighbors.kneighbors_graph(X, n_neighbors, mode="distance").toarray()
    else:
        G = neighbors.radius_neighbors_graph(X, radius, mode="distance").toarray()
    centerer = preprocessing.KernelCenterer()
    K = centerer.fit_transform(-0.5 * G**2)

    clf = manifold.Isomap(
        n_neighbors=n_neighbors,
        radius=radius,
        n_components=2,
        eigen_solver=eigen_solver,
        path_method=path_method,
    )
    clf.fit(X)

    # compute output kernel
    if n_neighbors is not None:
        G_iso = neighbors.kneighbors_graph(clf.embedding_, n_neighbors, mode="distance")
    else:
        G_iso = neighbors.radius_neighbors_graph(
            clf.embedding_, radius, mode="distance"
        )
    G_iso = G_iso.toarray()
    K_iso = centerer.fit_transform(-0.5 * G_iso**2)

    # make sure error agrees
    reconstruction_error = np.linalg.norm(K - K_iso) / n_pts
    atol = 1e-5 if global_dtype == np.float32 else 0
    assert_allclose(reconstruction_error, clf.reconstruction_error(), atol=atol)


@pytest.mark.parametrize("n_neighbors, radius", [(2, None), (None, 0.5)])
def test_transform(global_dtype, n_neighbors, radius):
    n_samples = 200
    n_components = 10
    noise_scale = 0.01

    # Create S-curve dataset
    X, y = datasets.make_s_curve(n_samples, random_state=0)

    X = X.astype(global_dtype, copy=False)

    # Compute isomap embedding
    iso = manifold.Isomap(
        n_components=n_components, n_neighbors=n_neighbors, radius=radius
    )
    X_iso = iso.fit_transform(X)

    # Re-embed a noisy version of the points
    rng = np.random.RandomState(0)
    noise = noise_scale * rng.randn(*X.shape)
    X_iso2 = iso.transform(X + noise)

    # Make sure the rms error on re-embedding is comparable to noise_scale
    assert np.sqrt(np.mean((X_iso - X_iso2) ** 2)) < 2 * noise_scale


@pytest.mark.parametrize("n_neighbors, radius", [(2, None), (None, 10.0)])
def test_pipeline(n_neighbors, radius, global_dtype):
    # check that Isomap works fine as a transformer in a Pipeline
    # only checks that no error is raised.
    # TODO check that it actually does something useful
    X, y = datasets.make_blobs(random_state=0)
    X = X.astype(global_dtype, copy=False)
    clf = pipeline.Pipeline(
        [
            ("isomap", manifold.Isomap(n_neighbors=n_neighbors, radius=radius)),
            ("clf", neighbors.KNeighborsClassifier()),
        ]
    )
    clf.fit(X, y)
    assert 0.9 < clf.score(X, y)


def test_pipeline_with_nearest_neighbors_transformer(global_dtype):
    # Test chaining NearestNeighborsTransformer and Isomap with
    # neighbors_algorithm='precomputed'
    algorithm = "auto"
    n_neighbors = 10

    X, _ = datasets.make_blobs(random_state=0)
    X2, _ = datasets.make_blobs(random_state=1)

    X = X.astype(global_dtype, copy=False)
    X2 = X2.astype(global_dtype, copy=False)

    # compare the chained version and the compact version
    est_chain = pipeline.make_pipeline(
        neighbors.KNeighborsTransformer(
            n_neighbors=n_neighbors, algorithm=algorithm, mode="distance"
        ),
        manifold.Isomap(n_neighbors=n_neighbors, metric="precomputed"),
    )
    est_compact = manifold.Isomap(
        n_neighbors=n_neighbors, neighbors_algorithm=algorithm
    )

    Xt_chain = est_chain.fit_transform(X)
    Xt_compact = est_compact.fit_transform(X)
    assert_allclose(Xt_chain, Xt_compact)

    Xt_chain = est_chain.transform(X2)
    Xt_compact = est_compact.transform(X2)
    assert_allclose(Xt_chain, Xt_compact)


@pytest.mark.parametrize(
    "metric, p, is_euclidean",
    [
        ("euclidean", 2, True),
        ("manhattan", 1, False),
        ("minkowski", 1, False),
        ("minkowski", 2, True),
        (lambda x1, x2: np.sqrt(np.sum(x1**2 + x2**2)), 2, False),
    ],
)
def test_different_metric(global_dtype, metric, p, is_euclidean):
    # Isomap must work on various metric parameters work correctly
    # and must default to euclidean.
    X, _ = datasets.make_blobs(random_state=0)
    X = X.astype(global_dtype, copy=False)

    reference = manifold.Isomap().fit_transform(X)
    embedding = manifold.Isomap(metric=metric, p=p).fit_transform(X)

    if is_euclidean:
        assert_allclose(embedding, reference)
    else:
        with pytest.raises(AssertionError, match="Not equal to tolerance"):
            assert_allclose(embedding, reference)


def test_isomap_clone_bug():
    # regression test for bug reported in #6062
    model = manifold.Isomap()
    for n_neighbors in [10, 15, 20]:
        model.set_params(n_neighbors=n_neighbors)
        model.fit(np.random.rand(50, 2))
        assert model.nbrs_.n_neighbors == n_neighbors


@pytest.mark.parametrize("eigen_solver", eigen_solvers)
@pytest.mark.parametrize("path_method", path_methods)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_sparse_input(
    global_dtype, eigen_solver, path_method, global_random_seed, csr_container
):
    # TODO: compare results on dense and sparse data as proposed in:
    # https://github.com/scikit-learn/scikit-learn/pull/23585#discussion_r968388186
    X = csr_container(
        sparse_rand(
            100,
            3,
            density=0.1,
            format="csr",
            dtype=global_dtype,
            random_state=global_random_seed,
        )
    )

    iso_dense = manifold.Isomap(
        n_components=2,
        eigen_solver=eigen_solver,
        path_method=path_method,
        n_neighbors=8,
    )
    iso_sparse = clone(iso_dense)

    X_trans_dense = iso_dense.fit_transform(X.toarray())
    X_trans_sparse = iso_sparse.fit_transform(X)

    assert_allclose(X_trans_sparse, X_trans_dense, rtol=1e-4, atol=1e-4)


def test_isomap_fit_precomputed_radius_graph(global_dtype):
    # Isomap.fit_transform must yield similar result when using
    # a precomputed distance matrix.

    X, y = datasets.make_s_curve(200, random_state=0)
    X = X.astype(global_dtype, copy=False)
    radius = 10

    g = neighbors.radius_neighbors_graph(X, radius=radius, mode="distance")
    isomap = manifold.Isomap(n_neighbors=None, radius=radius, metric="precomputed")
    isomap.fit(g)
    precomputed_result = isomap.embedding_

    isomap = manifold.Isomap(n_neighbors=None, radius=radius, metric="minkowski")
    result = isomap.fit_transform(X)
    atol = 1e-5 if global_dtype == np.float32 else 0
    assert_allclose(precomputed_result, result, atol=atol)


def test_isomap_fitted_attributes_dtype(global_dtype):
    """Check that the fitted attributes are stored accordingly to the
    data type of X."""
    iso = manifold.Isomap(n_neighbors=2)

    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=global_dtype)

    iso.fit(X)

    assert iso.dist_matrix_.dtype == global_dtype
    assert iso.embedding_.dtype == global_dtype


def test_isomap_dtype_equivalence():
    """Check the equivalence of the results with 32 and 64 bits input."""
    iso_32 = manifold.Isomap(n_neighbors=2)
    X_32 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    iso_32.fit(X_32)

    iso_64 = manifold.Isomap(n_neighbors=2)
    X_64 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    iso_64.fit(X_64)

    assert_allclose(iso_32.dist_matrix_, iso_64.dist_matrix_)


def test_isomap_raise_error_when_neighbor_and_radius_both_set():
    # Isomap.fit_transform must raise a ValueError if
    # radius and n_neighbors are provided.

    X, _ = datasets.load_digits(return_X_y=True)
    isomap = manifold.Isomap(n_neighbors=3, radius=5.5)
    msg = "Both n_neighbors and radius are provided"
    with pytest.raises(ValueError, match=msg):
        isomap.fit_transform(X)


def test_multiple_connected_components():
    # Test that a warning is raised when the graph has multiple components
    X = np.array([0, 1, 2, 5, 6, 7])[:, None]
    with pytest.warns(UserWarning, match="number of connected components"):
        manifold.Isomap(n_neighbors=2).fit(X)


def test_multiple_connected_components_metric_precomputed(global_dtype):
    # Test that an error is raised when the graph has multiple components
    # and when X is a precomputed neighbors graph.
    X = np.array([0, 1, 2, 5, 6, 7])[:, None].astype(global_dtype, copy=False)

    # works with a precomputed distance matrix (dense)
    X_distances = pairwise_distances(X)
    with pytest.warns(UserWarning, match="number of connected components"):
        manifold.Isomap(n_neighbors=1, metric="precomputed").fit(X_distances)

    # does not work with a precomputed neighbors graph (sparse)
    X_graph = neighbors.kneighbors_graph(X, n_neighbors=2, mode="distance")
    with pytest.raises(RuntimeError, match="number of connected components"):
        manifold.Isomap(n_neighbors=1, metric="precomputed").fit(X_graph)


def test_get_feature_names_out():
    """Check get_feature_names_out for Isomap."""
    X, y = make_blobs(random_state=0, n_features=4)
    n_components = 2

    iso = manifold.Isomap(n_components=n_components)
    iso.fit_transform(X)
    names = iso.get_feature_names_out()
    assert_array_equal([f"isomap{i}" for i in range(n_components)], names)
