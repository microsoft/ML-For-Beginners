from unittest.mock import Mock

import numpy as np
import pytest
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh, lobpcg

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.manifold import SpectralEmbedding, _spectral_embedding, spectral_embedding
from sklearn.manifold._spectral_embedding import (
    _graph_connected_component,
    _graph_is_connected,
)
from sklearn.metrics import normalized_mutual_info_score, pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import NearestNeighbors
from sklearn.utils._testing import assert_array_almost_equal, assert_array_equal
from sklearn.utils.extmath import _deterministic_vector_sign_flip
from sklearn.utils.fixes import (
    COO_CONTAINERS,
    CSC_CONTAINERS,
    CSR_CONTAINERS,
    parse_version,
    sp_version,
)
from sklearn.utils.fixes import laplacian as csgraph_laplacian

try:
    from pyamg import smoothed_aggregation_solver  # noqa

    pyamg_available = True
except ImportError:
    pyamg_available = False
skip_if_no_pyamg = pytest.mark.skipif(
    not pyamg_available, reason="PyAMG is required for the tests in this function."
)

# non centered, sparse centers to check the
centers = np.array(
    [
        [0.0, 5.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 4.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 5.0, 1.0],
    ]
)
n_samples = 1000
n_clusters, n_features = centers.shape
S, true_labels = make_blobs(
    n_samples=n_samples, centers=centers, cluster_std=1.0, random_state=42
)


def _assert_equal_with_sign_flipping(A, B, tol=0.0):
    """Check array A and B are equal with possible sign flipping on
    each columns"""
    tol_squared = tol**2
    for A_col, B_col in zip(A.T, B.T):
        assert (
            np.max((A_col - B_col) ** 2) <= tol_squared
            or np.max((A_col + B_col) ** 2) <= tol_squared
        )


@pytest.mark.parametrize("coo_container", COO_CONTAINERS)
def test_sparse_graph_connected_component(coo_container):
    rng = np.random.RandomState(42)
    n_samples = 300
    boundaries = [0, 42, 121, 200, n_samples]
    p = rng.permutation(n_samples)
    connections = []

    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        group = p[start:stop]
        # Connect all elements within the group at least once via an
        # arbitrary path that spans the group.
        for i in range(len(group) - 1):
            connections.append((group[i], group[i + 1]))

        # Add some more random connections within the group
        min_idx, max_idx = 0, len(group) - 1
        n_random_connections = 1000
        source = rng.randint(min_idx, max_idx, size=n_random_connections)
        target = rng.randint(min_idx, max_idx, size=n_random_connections)
        connections.extend(zip(group[source], group[target]))

    # Build a symmetric affinity matrix
    row_idx, column_idx = tuple(np.array(connections).T)
    data = rng.uniform(0.1, 42, size=len(connections))
    affinity = coo_container((data, (row_idx, column_idx)))
    affinity = 0.5 * (affinity + affinity.T)

    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        component_1 = _graph_connected_component(affinity, p[start])
        component_size = stop - start
        assert component_1.sum() == component_size

        # We should retrieve the same component mask by starting by both ends
        # of the group
        component_2 = _graph_connected_component(affinity, p[stop - 1])
        assert component_2.sum() == component_size
        assert_array_equal(component_1, component_2)


# TODO: investigate why this test is seed-sensitive on 32-bit Python
# runtimes. Is this revealing a numerical stability problem ? Or is it
# expected from the test numerical design ? In the latter case the test
# should be made less seed-sensitive instead.
@pytest.mark.parametrize(
    "eigen_solver",
    [
        "arpack",
        "lobpcg",
        pytest.param("amg", marks=skip_if_no_pyamg),
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_spectral_embedding_two_components(eigen_solver, dtype, seed=0):
    # Test spectral embedding with two components
    random_state = np.random.RandomState(seed)
    n_sample = 100
    affinity = np.zeros(shape=[n_sample * 2, n_sample * 2])
    # first component
    affinity[0:n_sample, 0:n_sample] = (
        np.abs(random_state.randn(n_sample, n_sample)) + 2
    )
    # second component
    affinity[n_sample::, n_sample::] = (
        np.abs(random_state.randn(n_sample, n_sample)) + 2
    )

    # Test of internal _graph_connected_component before connection
    component = _graph_connected_component(affinity, 0)
    assert component[:n_sample].all()
    assert not component[n_sample:].any()
    component = _graph_connected_component(affinity, -1)
    assert not component[:n_sample].any()
    assert component[n_sample:].all()

    # connection
    affinity[0, n_sample + 1] = 1
    affinity[n_sample + 1, 0] = 1
    affinity.flat[:: 2 * n_sample + 1] = 0
    affinity = 0.5 * (affinity + affinity.T)

    true_label = np.zeros(shape=2 * n_sample)
    true_label[0:n_sample] = 1

    se_precomp = SpectralEmbedding(
        n_components=1,
        affinity="precomputed",
        random_state=np.random.RandomState(seed),
        eigen_solver=eigen_solver,
    )

    embedded_coordinate = se_precomp.fit_transform(affinity.astype(dtype))
    # thresholding on the first components using 0.
    label_ = np.array(embedded_coordinate.ravel() < 0, dtype=np.int64)
    assert normalized_mutual_info_score(true_label, label_) == pytest.approx(1.0)


@pytest.mark.parametrize("sparse_container", [None, *CSR_CONTAINERS])
@pytest.mark.parametrize(
    "eigen_solver",
    [
        "arpack",
        "lobpcg",
        pytest.param("amg", marks=skip_if_no_pyamg),
    ],
)
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_spectral_embedding_precomputed_affinity(
    sparse_container, eigen_solver, dtype, seed=36
):
    # Test spectral embedding with precomputed kernel
    gamma = 1.0
    X = S if sparse_container is None else sparse_container(S)

    se_precomp = SpectralEmbedding(
        n_components=2,
        affinity="precomputed",
        random_state=np.random.RandomState(seed),
        eigen_solver=eigen_solver,
    )
    se_rbf = SpectralEmbedding(
        n_components=2,
        affinity="rbf",
        gamma=gamma,
        random_state=np.random.RandomState(seed),
        eigen_solver=eigen_solver,
    )
    embed_precomp = se_precomp.fit_transform(rbf_kernel(X.astype(dtype), gamma=gamma))
    embed_rbf = se_rbf.fit_transform(X.astype(dtype))
    assert_array_almost_equal(se_precomp.affinity_matrix_, se_rbf.affinity_matrix_)
    _assert_equal_with_sign_flipping(embed_precomp, embed_rbf, 0.05)


def test_precomputed_nearest_neighbors_filtering():
    # Test precomputed graph filtering when containing too many neighbors
    n_neighbors = 2
    results = []
    for additional_neighbors in [0, 10]:
        nn = NearestNeighbors(n_neighbors=n_neighbors + additional_neighbors).fit(S)
        graph = nn.kneighbors_graph(S, mode="connectivity")
        embedding = (
            SpectralEmbedding(
                random_state=0,
                n_components=2,
                affinity="precomputed_nearest_neighbors",
                n_neighbors=n_neighbors,
            )
            .fit(graph)
            .embedding_
        )
        results.append(embedding)

    assert_array_equal(results[0], results[1])


@pytest.mark.parametrize("sparse_container", [None, *CSR_CONTAINERS])
def test_spectral_embedding_callable_affinity(sparse_container, seed=36):
    # Test spectral embedding with callable affinity
    gamma = 0.9
    kern = rbf_kernel(S, gamma=gamma)
    X = S if sparse_container is None else sparse_container(S)

    se_callable = SpectralEmbedding(
        n_components=2,
        affinity=(lambda x: rbf_kernel(x, gamma=gamma)),
        gamma=gamma,
        random_state=np.random.RandomState(seed),
    )
    se_rbf = SpectralEmbedding(
        n_components=2,
        affinity="rbf",
        gamma=gamma,
        random_state=np.random.RandomState(seed),
    )
    embed_rbf = se_rbf.fit_transform(X)
    embed_callable = se_callable.fit_transform(X)
    assert_array_almost_equal(se_callable.affinity_matrix_, se_rbf.affinity_matrix_)
    assert_array_almost_equal(kern, se_rbf.affinity_matrix_)
    _assert_equal_with_sign_flipping(embed_rbf, embed_callable, 0.05)


# TODO: Remove when pyamg does replaces sp.rand call with np.random.rand
# https://github.com/scikit-learn/scikit-learn/issues/15913
@pytest.mark.filterwarnings(
    "ignore:scipy.rand is deprecated:DeprecationWarning:pyamg.*"
)
# TODO: Remove when pyamg removes the use of np.float
@pytest.mark.filterwarnings(
    "ignore:`np.float` is a deprecated alias:DeprecationWarning:pyamg.*"
)
# TODO: Remove when pyamg removes the use of pinv2
@pytest.mark.filterwarnings(
    "ignore:scipy.linalg.pinv2 is deprecated:DeprecationWarning:pyamg.*"
)
@pytest.mark.filterwarnings(
    "ignore:np.find_common_type is deprecated:DeprecationWarning:pyamg.*"
)
@pytest.mark.skipif(
    not pyamg_available, reason="PyAMG is required for the tests in this function."
)
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("coo_container", COO_CONTAINERS)
def test_spectral_embedding_amg_solver(dtype, coo_container, seed=36):
    se_amg = SpectralEmbedding(
        n_components=2,
        affinity="nearest_neighbors",
        eigen_solver="amg",
        n_neighbors=5,
        random_state=np.random.RandomState(seed),
    )
    se_arpack = SpectralEmbedding(
        n_components=2,
        affinity="nearest_neighbors",
        eigen_solver="arpack",
        n_neighbors=5,
        random_state=np.random.RandomState(seed),
    )
    embed_amg = se_amg.fit_transform(S.astype(dtype))
    embed_arpack = se_arpack.fit_transform(S.astype(dtype))
    _assert_equal_with_sign_flipping(embed_amg, embed_arpack, 1e-5)

    # same with special case in which amg is not actually used
    # regression test for #10715
    # affinity between nodes
    row = np.array([0, 0, 1, 2, 3, 3, 4], dtype=np.int32)
    col = np.array([1, 2, 2, 3, 4, 5, 5], dtype=np.int32)
    val = np.array([100, 100, 100, 1, 100, 100, 100], dtype=np.int64)

    affinity = coo_container(
        (np.hstack([val, val]), (np.hstack([row, col]), np.hstack([col, row]))),
        shape=(6, 6),
    )
    se_amg.affinity = "precomputed"
    se_arpack.affinity = "precomputed"
    embed_amg = se_amg.fit_transform(affinity.astype(dtype))
    embed_arpack = se_arpack.fit_transform(affinity.astype(dtype))
    _assert_equal_with_sign_flipping(embed_amg, embed_arpack, 1e-5)

    # Check that passing a sparse matrix with `np.int64` indices dtype raises an error
    # or is successful based on the version of SciPy which is installed.
    # Use a CSR matrix to avoid any conversion during the validation
    affinity = affinity.tocsr()
    affinity.indptr = affinity.indptr.astype(np.int64)
    affinity.indices = affinity.indices.astype(np.int64)

    # PR: https://github.com/scipy/scipy/pull/18913
    # First integration in 1.11.3: https://github.com/scipy/scipy/pull/19279
    scipy_graph_traversal_supports_int64_index = sp_version >= parse_version("1.11.3")
    if scipy_graph_traversal_supports_int64_index:
        se_amg.fit_transform(affinity)
    else:
        err_msg = "Only sparse matrices with 32-bit integer indices are accepted"
        with pytest.raises(ValueError, match=err_msg):
            se_amg.fit_transform(affinity)


# TODO: Remove filterwarnings when pyamg does replaces sp.rand call with
# np.random.rand:
# https://github.com/scikit-learn/scikit-learn/issues/15913
@pytest.mark.filterwarnings(
    "ignore:scipy.rand is deprecated:DeprecationWarning:pyamg.*"
)
# TODO: Remove when pyamg removes the use of np.float
@pytest.mark.filterwarnings(
    "ignore:`np.float` is a deprecated alias:DeprecationWarning:pyamg.*"
)
# TODO: Remove when pyamg removes the use of pinv2
@pytest.mark.filterwarnings(
    "ignore:scipy.linalg.pinv2 is deprecated:DeprecationWarning:pyamg.*"
)
@pytest.mark.skipif(
    not pyamg_available, reason="PyAMG is required for the tests in this function."
)
# TODO: Remove when pyamg removes the use of np.find_common_type
@pytest.mark.filterwarnings(
    "ignore:np.find_common_type is deprecated:DeprecationWarning:pyamg.*"
)
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_spectral_embedding_amg_solver_failure(dtype, seed=36):
    # Non-regression test for amg solver failure (issue #13393 on github)
    num_nodes = 100
    X = sparse.rand(num_nodes, num_nodes, density=0.1, random_state=seed)
    X = X.astype(dtype)
    upper = sparse.triu(X) - sparse.diags(X.diagonal())
    sym_matrix = upper + upper.T
    embedding = spectral_embedding(
        sym_matrix, n_components=10, eigen_solver="amg", random_state=0
    )

    # Check that the learned embedding is stable w.r.t. random solver init:
    for i in range(3):
        new_embedding = spectral_embedding(
            sym_matrix, n_components=10, eigen_solver="amg", random_state=i + 1
        )
        _assert_equal_with_sign_flipping(embedding, new_embedding, tol=0.05)


@pytest.mark.filterwarnings("ignore:the behavior of nmi will change in version 0.22")
def test_pipeline_spectral_clustering(seed=36):
    # Test using pipeline to do spectral clustering
    random_state = np.random.RandomState(seed)
    se_rbf = SpectralEmbedding(
        n_components=n_clusters, affinity="rbf", random_state=random_state
    )
    se_knn = SpectralEmbedding(
        n_components=n_clusters,
        affinity="nearest_neighbors",
        n_neighbors=5,
        random_state=random_state,
    )
    for se in [se_rbf, se_knn]:
        km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        km.fit(se.fit_transform(S))
        assert_array_almost_equal(
            normalized_mutual_info_score(km.labels_, true_labels), 1.0, 2
        )


def test_connectivity(seed=36):
    # Test that graph connectivity test works as expected
    graph = np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1],
        ]
    )
    assert not _graph_is_connected(graph)
    for csr_container in CSR_CONTAINERS:
        assert not _graph_is_connected(csr_container(graph))
    for csc_container in CSC_CONTAINERS:
        assert not _graph_is_connected(csc_container(graph))

    graph = np.array(
        [
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1],
        ]
    )
    assert _graph_is_connected(graph)
    for csr_container in CSR_CONTAINERS:
        assert _graph_is_connected(csr_container(graph))
    for csc_container in CSC_CONTAINERS:
        assert _graph_is_connected(csc_container(graph))


def test_spectral_embedding_deterministic():
    # Test that Spectral Embedding is deterministic
    random_state = np.random.RandomState(36)
    data = random_state.randn(10, 30)
    sims = rbf_kernel(data)
    embedding_1 = spectral_embedding(sims)
    embedding_2 = spectral_embedding(sims)
    assert_array_almost_equal(embedding_1, embedding_2)


def test_spectral_embedding_unnormalized():
    # Test that spectral_embedding is also processing unnormalized laplacian
    # correctly
    random_state = np.random.RandomState(36)
    data = random_state.randn(10, 30)
    sims = rbf_kernel(data)
    n_components = 8
    embedding_1 = spectral_embedding(
        sims, norm_laplacian=False, n_components=n_components, drop_first=False
    )

    # Verify using manual computation with dense eigh
    laplacian, dd = csgraph_laplacian(sims, normed=False, return_diag=True)
    _, diffusion_map = eigh(laplacian)
    embedding_2 = diffusion_map.T[:n_components]
    embedding_2 = _deterministic_vector_sign_flip(embedding_2).T

    assert_array_almost_equal(embedding_1, embedding_2)


def test_spectral_embedding_first_eigen_vector():
    # Test that the first eigenvector of spectral_embedding
    # is constant and that the second is not (for a connected graph)
    random_state = np.random.RandomState(36)
    data = random_state.randn(10, 30)
    sims = rbf_kernel(data)
    n_components = 2

    for seed in range(10):
        embedding = spectral_embedding(
            sims,
            norm_laplacian=False,
            n_components=n_components,
            drop_first=False,
            random_state=seed,
        )

        assert np.std(embedding[:, 0]) == pytest.approx(0)
        assert np.std(embedding[:, 1]) > 1e-3


@pytest.mark.parametrize(
    "eigen_solver",
    [
        "arpack",
        "lobpcg",
        pytest.param("amg", marks=skip_if_no_pyamg),
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_spectral_embedding_preserves_dtype(eigen_solver, dtype):
    """Check that `SpectralEmbedding is preserving the dtype of the fitted
    attribute and transformed data.

    Ideally, this test should be covered by the common test
    `check_transformer_preserve_dtypes`. However, this test only run
    with transformers implementing `transform` while `SpectralEmbedding`
    implements only `fit_transform`.
    """
    X = S.astype(dtype)
    se = SpectralEmbedding(
        n_components=2, affinity="rbf", eigen_solver=eigen_solver, random_state=0
    )
    X_trans = se.fit_transform(X)

    assert X_trans.dtype == dtype
    assert se.embedding_.dtype == dtype
    assert se.affinity_matrix_.dtype == dtype


@pytest.mark.skipif(
    pyamg_available,
    reason="PyAMG is installed and we should not test for an error.",
)
def test_error_pyamg_not_available():
    se_precomp = SpectralEmbedding(
        n_components=2,
        affinity="rbf",
        eigen_solver="amg",
    )
    err_msg = "The eigen_solver was set to 'amg', but pyamg is not available."
    with pytest.raises(ValueError, match=err_msg):
        se_precomp.fit_transform(S)


# TODO: Remove when pyamg removes the use of np.find_common_type
@pytest.mark.filterwarnings(
    "ignore:np.find_common_type is deprecated:DeprecationWarning:pyamg.*"
)
@pytest.mark.parametrize("solver", ["arpack", "amg", "lobpcg"])
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_spectral_eigen_tol_auto(monkeypatch, solver, csr_container):
    """Test that `eigen_tol="auto"` is resolved correctly"""
    if solver == "amg" and not pyamg_available:
        pytest.skip("PyAMG is not available.")
    X, _ = make_blobs(
        n_samples=200, random_state=0, centers=[[1, 1], [-1, -1]], cluster_std=0.01
    )
    D = pairwise_distances(X)  # Distance matrix
    S = np.max(D) - D  # Similarity matrix

    solver_func = eigsh if solver == "arpack" else lobpcg
    default_value = 0 if solver == "arpack" else None
    if solver == "amg":
        S = csr_container(S)

    mocked_solver = Mock(side_effect=solver_func)

    monkeypatch.setattr(_spectral_embedding, solver_func.__qualname__, mocked_solver)

    spectral_embedding(S, random_state=42, eigen_solver=solver, eigen_tol="auto")
    mocked_solver.assert_called()

    _, kwargs = mocked_solver.call_args
    assert kwargs["tol"] == default_value
