import sys
from io import StringIO

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose
from scipy.optimize import check_grad
from scipy.spatial.distance import pdist, squareform

from sklearn import config_context
from sklearn.datasets import make_blobs
from sklearn.exceptions import EfficiencyWarning

# mypy error: Module 'sklearn.manifold' has no attribute '_barnes_hut_tsne'
from sklearn.manifold import (  # type: ignore
    TSNE,
    _barnes_hut_tsne,
)
from sklearn.manifold._t_sne import (
    _gradient_descent,
    _joint_probabilities,
    _joint_probabilities_nn,
    _kl_divergence,
    _kl_divergence_bh,
    trustworthiness,
)
from sklearn.manifold._utils import _binary_search_perplexity
from sklearn.metrics.pairwise import (
    cosine_distances,
    manhattan_distances,
    pairwise_distances,
)
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
    ignore_warnings,
    skip_if_32bit,
)

x = np.linspace(0, 1, 10)
xx, yy = np.meshgrid(x, x)
X_2d_grid = np.hstack(
    [
        xx.ravel().reshape(-1, 1),
        yy.ravel().reshape(-1, 1),
    ]
)


def test_gradient_descent_stops():
    # Test stopping conditions of gradient descent.
    class ObjectiveSmallGradient:
        def __init__(self):
            self.it = -1

        def __call__(self, _, compute_error=True):
            self.it += 1
            return (10 - self.it) / 10.0, np.array([1e-5])

    def flat_function(_, compute_error=True):
        return 0.0, np.ones(1)

    # Gradient norm
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        _, error, it = _gradient_descent(
            ObjectiveSmallGradient(),
            np.zeros(1),
            0,
            n_iter=100,
            n_iter_without_progress=100,
            momentum=0.0,
            learning_rate=0.0,
            min_gain=0.0,
            min_grad_norm=1e-5,
            verbose=2,
        )
    finally:
        out = sys.stdout.getvalue()
        sys.stdout.close()
        sys.stdout = old_stdout
    assert error == 1.0
    assert it == 0
    assert "gradient norm" in out

    # Maximum number of iterations without improvement
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        _, error, it = _gradient_descent(
            flat_function,
            np.zeros(1),
            0,
            n_iter=100,
            n_iter_without_progress=10,
            momentum=0.0,
            learning_rate=0.0,
            min_gain=0.0,
            min_grad_norm=0.0,
            verbose=2,
        )
    finally:
        out = sys.stdout.getvalue()
        sys.stdout.close()
        sys.stdout = old_stdout
    assert error == 0.0
    assert it == 11
    assert "did not make any progress" in out

    # Maximum number of iterations
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        _, error, it = _gradient_descent(
            ObjectiveSmallGradient(),
            np.zeros(1),
            0,
            n_iter=11,
            n_iter_without_progress=100,
            momentum=0.0,
            learning_rate=0.0,
            min_gain=0.0,
            min_grad_norm=0.0,
            verbose=2,
        )
    finally:
        out = sys.stdout.getvalue()
        sys.stdout.close()
        sys.stdout = old_stdout
    assert error == 0.0
    assert it == 10
    assert "Iteration 10" in out


def test_binary_search():
    # Test if the binary search finds Gaussians with desired perplexity.
    random_state = check_random_state(0)
    data = random_state.randn(50, 5)
    distances = pairwise_distances(data).astype(np.float32)
    desired_perplexity = 25.0
    P = _binary_search_perplexity(distances, desired_perplexity, verbose=0)
    P = np.maximum(P, np.finfo(np.double).eps)
    mean_perplexity = np.mean(
        [np.exp(-np.sum(P[i] * np.log(P[i]))) for i in range(P.shape[0])]
    )
    assert_almost_equal(mean_perplexity, desired_perplexity, decimal=3)


def test_binary_search_underflow():
    # Test if the binary search finds Gaussians with desired perplexity.
    # A more challenging case than the one above, producing numeric
    # underflow in float precision (see issue #19471 and PR #19472).
    random_state = check_random_state(42)
    data = random_state.randn(1, 90).astype(np.float32) + 100
    desired_perplexity = 30.0
    P = _binary_search_perplexity(data, desired_perplexity, verbose=0)
    perplexity = 2 ** -np.nansum(P[0, 1:] * np.log2(P[0, 1:]))
    assert_almost_equal(perplexity, desired_perplexity, decimal=3)


def test_binary_search_neighbors():
    # Binary perplexity search approximation.
    # Should be approximately equal to the slow method when we use
    # all points as neighbors.
    n_samples = 200
    desired_perplexity = 25.0
    random_state = check_random_state(0)
    data = random_state.randn(n_samples, 2).astype(np.float32, copy=False)
    distances = pairwise_distances(data)
    P1 = _binary_search_perplexity(distances, desired_perplexity, verbose=0)

    # Test that when we use all the neighbors the results are identical
    n_neighbors = n_samples - 1
    nn = NearestNeighbors().fit(data)
    distance_graph = nn.kneighbors_graph(n_neighbors=n_neighbors, mode="distance")
    distances_nn = distance_graph.data.astype(np.float32, copy=False)
    distances_nn = distances_nn.reshape(n_samples, n_neighbors)
    P2 = _binary_search_perplexity(distances_nn, desired_perplexity, verbose=0)

    indptr = distance_graph.indptr
    P1_nn = np.array(
        [
            P1[k, distance_graph.indices[indptr[k] : indptr[k + 1]]]
            for k in range(n_samples)
        ]
    )
    assert_array_almost_equal(P1_nn, P2, decimal=4)

    # Test that the highest P_ij are the same when fewer neighbors are used
    for k in np.linspace(150, n_samples - 1, 5):
        k = int(k)
        topn = k * 10  # check the top 10 * k entries out of k * k entries
        distance_graph = nn.kneighbors_graph(n_neighbors=k, mode="distance")
        distances_nn = distance_graph.data.astype(np.float32, copy=False)
        distances_nn = distances_nn.reshape(n_samples, k)
        P2k = _binary_search_perplexity(distances_nn, desired_perplexity, verbose=0)
        assert_array_almost_equal(P1_nn, P2, decimal=2)
        idx = np.argsort(P1.ravel())[::-1]
        P1top = P1.ravel()[idx][:topn]
        idx = np.argsort(P2k.ravel())[::-1]
        P2top = P2k.ravel()[idx][:topn]
        assert_array_almost_equal(P1top, P2top, decimal=2)


def test_binary_perplexity_stability():
    # Binary perplexity search should be stable.
    # The binary_search_perplexity had a bug wherein the P array
    # was uninitialized, leading to sporadically failing tests.
    n_neighbors = 10
    n_samples = 100
    random_state = check_random_state(0)
    data = random_state.randn(n_samples, 5)
    nn = NearestNeighbors().fit(data)
    distance_graph = nn.kneighbors_graph(n_neighbors=n_neighbors, mode="distance")
    distances = distance_graph.data.astype(np.float32, copy=False)
    distances = distances.reshape(n_samples, n_neighbors)
    last_P = None
    desired_perplexity = 3
    for _ in range(100):
        P = _binary_search_perplexity(distances.copy(), desired_perplexity, verbose=0)
        P1 = _joint_probabilities_nn(distance_graph, desired_perplexity, verbose=0)
        # Convert the sparse matrix to a dense one for testing
        P1 = P1.toarray()
        if last_P is None:
            last_P = P
            last_P1 = P1
        else:
            assert_array_almost_equal(P, last_P, decimal=4)
            assert_array_almost_equal(P1, last_P1, decimal=4)


def test_gradient():
    # Test gradient of Kullback-Leibler divergence.
    random_state = check_random_state(0)

    n_samples = 50
    n_features = 2
    n_components = 2
    alpha = 1.0

    distances = random_state.randn(n_samples, n_features).astype(np.float32)
    distances = np.abs(distances.dot(distances.T))
    np.fill_diagonal(distances, 0.0)
    X_embedded = random_state.randn(n_samples, n_components).astype(np.float32)

    P = _joint_probabilities(distances, desired_perplexity=25.0, verbose=0)

    def fun(params):
        return _kl_divergence(params, P, alpha, n_samples, n_components)[0]

    def grad(params):
        return _kl_divergence(params, P, alpha, n_samples, n_components)[1]

    assert_almost_equal(check_grad(fun, grad, X_embedded.ravel()), 0.0, decimal=5)


def test_trustworthiness():
    # Test trustworthiness score.
    random_state = check_random_state(0)

    # Affine transformation
    X = random_state.randn(100, 2)
    assert trustworthiness(X, 5.0 + X / 10.0) == 1.0

    # Randomly shuffled
    X = np.arange(100).reshape(-1, 1)
    X_embedded = X.copy()
    random_state.shuffle(X_embedded)
    assert trustworthiness(X, X_embedded) < 0.6

    # Completely different
    X = np.arange(5).reshape(-1, 1)
    X_embedded = np.array([[0], [2], [4], [1], [3]])
    assert_almost_equal(trustworthiness(X, X_embedded, n_neighbors=1), 0.2)


def test_trustworthiness_n_neighbors_error():
    """Raise an error when n_neighbors >= n_samples / 2.

    Non-regression test for #18567.
    """
    regex = "n_neighbors .+ should be less than .+"
    rng = np.random.RandomState(42)
    X = rng.rand(7, 4)
    X_embedded = rng.rand(7, 2)
    with pytest.raises(ValueError, match=regex):
        trustworthiness(X, X_embedded, n_neighbors=5)

    trust = trustworthiness(X, X_embedded, n_neighbors=3)
    assert 0 <= trust <= 1


@pytest.mark.parametrize("method", ["exact", "barnes_hut"])
@pytest.mark.parametrize("init", ("random", "pca"))
def test_preserve_trustworthiness_approximately(method, init):
    # Nearest neighbors should be preserved approximately.
    random_state = check_random_state(0)
    n_components = 2
    X = random_state.randn(50, n_components).astype(np.float32)
    tsne = TSNE(
        n_components=n_components,
        init=init,
        random_state=0,
        method=method,
        n_iter=700,
        learning_rate="auto",
    )
    X_embedded = tsne.fit_transform(X)
    t = trustworthiness(X, X_embedded, n_neighbors=1)
    assert t > 0.85


def test_optimization_minimizes_kl_divergence():
    """t-SNE should give a lower KL divergence with more iterations."""
    random_state = check_random_state(0)
    X, _ = make_blobs(n_features=3, random_state=random_state)
    kl_divergences = []
    for n_iter in [250, 300, 350]:
        tsne = TSNE(
            n_components=2,
            init="random",
            perplexity=10,
            learning_rate=100.0,
            n_iter=n_iter,
            random_state=0,
        )
        tsne.fit_transform(X)
        kl_divergences.append(tsne.kl_divergence_)
    assert kl_divergences[1] <= kl_divergences[0]
    assert kl_divergences[2] <= kl_divergences[1]


@pytest.mark.parametrize("method", ["exact", "barnes_hut"])
def test_fit_transform_csr_matrix(method):
    # TODO: compare results on dense and sparse data as proposed in:
    # https://github.com/scikit-learn/scikit-learn/pull/23585#discussion_r968388186
    # X can be a sparse matrix.
    rng = check_random_state(0)
    X = rng.randn(50, 2)
    X[(rng.randint(0, 50, 25), rng.randint(0, 2, 25))] = 0.0
    X_csr = sp.csr_matrix(X)
    tsne = TSNE(
        n_components=2,
        init="random",
        perplexity=10,
        learning_rate=100.0,
        random_state=0,
        method=method,
        n_iter=750,
    )
    X_embedded = tsne.fit_transform(X_csr)
    assert_allclose(trustworthiness(X_csr, X_embedded, n_neighbors=1), 1.0, rtol=1.1e-1)


def test_preserve_trustworthiness_approximately_with_precomputed_distances():
    # Nearest neighbors should be preserved approximately.
    random_state = check_random_state(0)
    for i in range(3):
        X = random_state.randn(80, 2)
        D = squareform(pdist(X), "sqeuclidean")
        tsne = TSNE(
            n_components=2,
            perplexity=2,
            learning_rate=100.0,
            early_exaggeration=2.0,
            metric="precomputed",
            random_state=i,
            verbose=0,
            n_iter=500,
            init="random",
        )
        X_embedded = tsne.fit_transform(D)
        t = trustworthiness(D, X_embedded, n_neighbors=1, metric="precomputed")
        assert t > 0.95


def test_trustworthiness_not_euclidean_metric():
    # Test trustworthiness with a metric different from 'euclidean' and
    # 'precomputed'
    random_state = check_random_state(0)
    X = random_state.randn(100, 2)
    assert trustworthiness(X, X, metric="cosine") == trustworthiness(
        pairwise_distances(X, metric="cosine"), X, metric="precomputed"
    )


@pytest.mark.parametrize(
    "method, retype",
    [
        ("exact", np.asarray),
        ("barnes_hut", np.asarray),
        ("barnes_hut", sp.csr_matrix),
    ],
)
@pytest.mark.parametrize(
    "D, message_regex",
    [
        ([[0.0], [1.0]], ".* square distance matrix"),
        ([[0.0, -1.0], [1.0, 0.0]], ".* positive.*"),
    ],
)
def test_bad_precomputed_distances(method, D, retype, message_regex):
    tsne = TSNE(
        metric="precomputed",
        method=method,
        init="random",
        random_state=42,
        perplexity=1,
    )
    with pytest.raises(ValueError, match=message_regex):
        tsne.fit_transform(retype(D))


def test_exact_no_precomputed_sparse():
    tsne = TSNE(
        metric="precomputed",
        method="exact",
        init="random",
        random_state=42,
        perplexity=1,
    )
    with pytest.raises(TypeError, match="sparse"):
        tsne.fit_transform(sp.csr_matrix([[0, 5], [5, 0]]))


def test_high_perplexity_precomputed_sparse_distances():
    # Perplexity should be less than 50
    dist = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    bad_dist = sp.csr_matrix(dist)
    tsne = TSNE(metric="precomputed", init="random", random_state=42, perplexity=1)
    msg = "3 neighbors per samples are required, but some samples have only 1"
    with pytest.raises(ValueError, match=msg):
        tsne.fit_transform(bad_dist)


@ignore_warnings(category=EfficiencyWarning)
def test_sparse_precomputed_distance():
    """Make sure that TSNE works identically for sparse and dense matrix"""
    random_state = check_random_state(0)
    X = random_state.randn(100, 2)

    D_sparse = kneighbors_graph(X, n_neighbors=100, mode="distance", include_self=True)
    D = pairwise_distances(X)
    assert sp.issparse(D_sparse)
    assert_almost_equal(D_sparse.A, D)

    tsne = TSNE(
        metric="precomputed", random_state=0, init="random", learning_rate="auto"
    )
    Xt_dense = tsne.fit_transform(D)

    for fmt in ["csr", "lil"]:
        Xt_sparse = tsne.fit_transform(D_sparse.asformat(fmt))
        assert_almost_equal(Xt_dense, Xt_sparse)


def test_non_positive_computed_distances():
    # Computed distance matrices must be positive.
    def metric(x, y):
        return -1

    # Negative computed distances should be caught even if result is squared
    tsne = TSNE(metric=metric, method="exact", perplexity=1)
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="All distances .*metric given.*"):
        tsne.fit_transform(X)


def test_init_ndarray():
    # Initialize TSNE with ndarray and test fit
    tsne = TSNE(init=np.zeros((100, 2)), learning_rate="auto")
    X_embedded = tsne.fit_transform(np.ones((100, 5)))
    assert_array_equal(np.zeros((100, 2)), X_embedded)


def test_init_ndarray_precomputed():
    # Initialize TSNE with ndarray and metric 'precomputed'
    # Make sure no FutureWarning is thrown from _fit
    tsne = TSNE(
        init=np.zeros((100, 2)),
        metric="precomputed",
        learning_rate=50.0,
    )
    tsne.fit(np.zeros((100, 100)))


def test_pca_initialization_not_compatible_with_precomputed_kernel():
    # Precomputed distance matrices cannot use PCA initialization.
    tsne = TSNE(metric="precomputed", init="pca", perplexity=1)
    with pytest.raises(
        ValueError,
        match='The parameter init="pca" cannot be used with metric="precomputed".',
    ):
        tsne.fit_transform(np.array([[0.0], [1.0]]))


def test_pca_initialization_not_compatible_with_sparse_input():
    # Sparse input matrices cannot use PCA initialization.
    tsne = TSNE(init="pca", learning_rate=100.0, perplexity=1)
    with pytest.raises(TypeError, match="PCA initialization.*"):
        tsne.fit_transform(sp.csr_matrix([[0, 5], [5, 0]]))


def test_n_components_range():
    # barnes_hut method should only be used with n_components <= 3
    tsne = TSNE(n_components=4, method="barnes_hut", perplexity=1)
    with pytest.raises(ValueError, match="'n_components' should be .*"):
        tsne.fit_transform(np.array([[0.0], [1.0]]))


def test_early_exaggeration_used():
    # check that the ``early_exaggeration`` parameter has an effect
    random_state = check_random_state(0)
    n_components = 2
    methods = ["exact", "barnes_hut"]
    X = random_state.randn(25, n_components).astype(np.float32)
    for method in methods:
        tsne = TSNE(
            n_components=n_components,
            perplexity=1,
            learning_rate=100.0,
            init="pca",
            random_state=0,
            method=method,
            early_exaggeration=1.0,
            n_iter=250,
        )
        X_embedded1 = tsne.fit_transform(X)
        tsne = TSNE(
            n_components=n_components,
            perplexity=1,
            learning_rate=100.0,
            init="pca",
            random_state=0,
            method=method,
            early_exaggeration=10.0,
            n_iter=250,
        )
        X_embedded2 = tsne.fit_transform(X)

        assert not np.allclose(X_embedded1, X_embedded2)


def test_n_iter_used():
    # check that the ``n_iter`` parameter has an effect
    random_state = check_random_state(0)
    n_components = 2
    methods = ["exact", "barnes_hut"]
    X = random_state.randn(25, n_components).astype(np.float32)
    for method in methods:
        for n_iter in [251, 500]:
            tsne = TSNE(
                n_components=n_components,
                perplexity=1,
                learning_rate=0.5,
                init="random",
                random_state=0,
                method=method,
                early_exaggeration=1.0,
                n_iter=n_iter,
            )
            tsne.fit_transform(X)

            assert tsne.n_iter_ == n_iter - 1


def test_answer_gradient_two_points():
    # Test the tree with only a single set of children.
    #
    # These tests & answers have been checked against the reference
    # implementation by LvdM.
    pos_input = np.array([[1.0, 0.0], [0.0, 1.0]])
    pos_output = np.array(
        [[-4.961291e-05, -1.072243e-04], [9.259460e-05, 2.702024e-04]]
    )
    neighbors = np.array([[1], [0]])
    grad_output = np.array(
        [[-2.37012478e-05, -6.29044398e-05], [2.37012478e-05, 6.29044398e-05]]
    )
    _run_answer_test(pos_input, pos_output, neighbors, grad_output)


def test_answer_gradient_four_points():
    # Four points tests the tree with multiple levels of children.
    #
    # These tests & answers have been checked against the reference
    # implementation by LvdM.
    pos_input = np.array([[1.0, 0.0], [0.0, 1.0], [5.0, 2.0], [7.3, 2.2]])
    pos_output = np.array(
        [
            [6.080564e-05, -7.120823e-05],
            [-1.718945e-04, -4.000536e-05],
            [-2.271720e-04, 8.663310e-05],
            [-1.032577e-04, -3.582033e-05],
        ]
    )
    neighbors = np.array([[1, 2, 3], [0, 2, 3], [1, 0, 3], [1, 2, 0]])
    grad_output = np.array(
        [
            [5.81128448e-05, -7.78033454e-06],
            [-5.81526851e-05, 7.80976444e-06],
            [4.24275173e-08, -3.69569698e-08],
            [-2.58720939e-09, 7.52706374e-09],
        ]
    )
    _run_answer_test(pos_input, pos_output, neighbors, grad_output)


def test_skip_num_points_gradient():
    # Test the kwargs option skip_num_points.
    #
    # Skip num points should make it such that the Barnes_hut gradient
    # is not calculated for indices below skip_num_point.
    # Aside from skip_num_points=2 and the first two gradient rows
    # being set to zero, these data points are the same as in
    # test_answer_gradient_four_points()
    pos_input = np.array([[1.0, 0.0], [0.0, 1.0], [5.0, 2.0], [7.3, 2.2]])
    pos_output = np.array(
        [
            [6.080564e-05, -7.120823e-05],
            [-1.718945e-04, -4.000536e-05],
            [-2.271720e-04, 8.663310e-05],
            [-1.032577e-04, -3.582033e-05],
        ]
    )
    neighbors = np.array([[1, 2, 3], [0, 2, 3], [1, 0, 3], [1, 2, 0]])
    grad_output = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [4.24275173e-08, -3.69569698e-08],
            [-2.58720939e-09, 7.52706374e-09],
        ]
    )
    _run_answer_test(pos_input, pos_output, neighbors, grad_output, False, 0.1, 2)


def _run_answer_test(
    pos_input,
    pos_output,
    neighbors,
    grad_output,
    verbose=False,
    perplexity=0.1,
    skip_num_points=0,
):
    distances = pairwise_distances(pos_input).astype(np.float32)
    args = distances, perplexity, verbose
    pos_output = pos_output.astype(np.float32)
    neighbors = neighbors.astype(np.int64, copy=False)
    pij_input = _joint_probabilities(*args)
    pij_input = squareform(pij_input).astype(np.float32)
    grad_bh = np.zeros(pos_output.shape, dtype=np.float32)

    from scipy.sparse import csr_matrix

    P = csr_matrix(pij_input)

    neighbors = P.indices.astype(np.int64)
    indptr = P.indptr.astype(np.int64)

    _barnes_hut_tsne.gradient(
        P.data, pos_output, neighbors, indptr, grad_bh, 0.5, 2, 1, skip_num_points=0
    )
    assert_array_almost_equal(grad_bh, grad_output, decimal=4)


def test_verbose():
    # Verbose options write to stdout.
    random_state = check_random_state(0)
    tsne = TSNE(verbose=2, perplexity=4)
    X = random_state.randn(5, 2)

    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        tsne.fit_transform(X)
    finally:
        out = sys.stdout.getvalue()
        sys.stdout.close()
        sys.stdout = old_stdout

    assert "[t-SNE]" in out
    assert "nearest neighbors..." in out
    assert "Computed conditional probabilities" in out
    assert "Mean sigma" in out
    assert "early exaggeration" in out


def test_chebyshev_metric():
    # t-SNE should allow metrics that cannot be squared (issue #3526).
    random_state = check_random_state(0)
    tsne = TSNE(metric="chebyshev", perplexity=4)
    X = random_state.randn(5, 2)
    tsne.fit_transform(X)


def test_reduction_to_one_component():
    # t-SNE should allow reduction to one component (issue #4154).
    random_state = check_random_state(0)
    tsne = TSNE(n_components=1, perplexity=4)
    X = random_state.randn(5, 2)
    X_embedded = tsne.fit(X).embedding_
    assert np.all(np.isfinite(X_embedded))


@pytest.mark.parametrize("method", ["barnes_hut", "exact"])
@pytest.mark.parametrize("dt", [np.float32, np.float64])
def test_64bit(method, dt):
    # Ensure 64bit arrays are handled correctly.
    random_state = check_random_state(0)

    X = random_state.randn(10, 2).astype(dt, copy=False)
    tsne = TSNE(
        n_components=2,
        perplexity=2,
        learning_rate=100.0,
        random_state=0,
        method=method,
        verbose=0,
        n_iter=300,
        init="random",
    )
    X_embedded = tsne.fit_transform(X)
    effective_type = X_embedded.dtype

    # tsne cython code is only single precision, so the output will
    # always be single precision, irrespectively of the input dtype
    assert effective_type == np.float32


@pytest.mark.parametrize("method", ["barnes_hut", "exact"])
def test_kl_divergence_not_nan(method):
    # Ensure kl_divergence_ is computed at last iteration
    # even though n_iter % n_iter_check != 0, i.e. 1003 % 50 != 0
    random_state = check_random_state(0)

    X = random_state.randn(50, 2)
    tsne = TSNE(
        n_components=2,
        perplexity=2,
        learning_rate=100.0,
        random_state=0,
        method=method,
        verbose=0,
        n_iter=503,
        init="random",
    )
    tsne.fit_transform(X)

    assert not np.isnan(tsne.kl_divergence_)


def test_barnes_hut_angle():
    # When Barnes-Hut's angle=0 this corresponds to the exact method.
    angle = 0.0
    perplexity = 10
    n_samples = 100
    for n_components in [2, 3]:
        n_features = 5
        degrees_of_freedom = float(n_components - 1.0)

        random_state = check_random_state(0)
        data = random_state.randn(n_samples, n_features)
        distances = pairwise_distances(data)
        params = random_state.randn(n_samples, n_components)
        P = _joint_probabilities(distances, perplexity, verbose=0)
        kl_exact, grad_exact = _kl_divergence(
            params, P, degrees_of_freedom, n_samples, n_components
        )

        n_neighbors = n_samples - 1
        distances_csr = (
            NearestNeighbors()
            .fit(data)
            .kneighbors_graph(n_neighbors=n_neighbors, mode="distance")
        )
        P_bh = _joint_probabilities_nn(distances_csr, perplexity, verbose=0)
        kl_bh, grad_bh = _kl_divergence_bh(
            params,
            P_bh,
            degrees_of_freedom,
            n_samples,
            n_components,
            angle=angle,
            skip_num_points=0,
            verbose=0,
        )

        P = squareform(P)
        P_bh = P_bh.toarray()
        assert_array_almost_equal(P_bh, P, decimal=5)
        assert_almost_equal(kl_exact, kl_bh, decimal=3)


@skip_if_32bit
def test_n_iter_without_progress():
    # Use a dummy negative n_iter_without_progress and check output on stdout
    random_state = check_random_state(0)
    X = random_state.randn(100, 10)
    for method in ["barnes_hut", "exact"]:
        tsne = TSNE(
            n_iter_without_progress=-1,
            verbose=2,
            learning_rate=1e8,
            random_state=0,
            method=method,
            n_iter=351,
            init="random",
        )
        tsne._N_ITER_CHECK = 1
        tsne._EXPLORATION_N_ITER = 0

        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            tsne.fit_transform(X)
        finally:
            out = sys.stdout.getvalue()
            sys.stdout.close()
            sys.stdout = old_stdout

        # The output needs to contain the value of n_iter_without_progress
        assert "did not make any progress during the last -1 episodes. Finished." in out


def test_min_grad_norm():
    # Make sure that the parameter min_grad_norm is used correctly
    random_state = check_random_state(0)
    X = random_state.randn(100, 2)
    min_grad_norm = 0.002
    tsne = TSNE(min_grad_norm=min_grad_norm, verbose=2, random_state=0, method="exact")

    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        tsne.fit_transform(X)
    finally:
        out = sys.stdout.getvalue()
        sys.stdout.close()
        sys.stdout = old_stdout

    lines_out = out.split("\n")

    # extract the gradient norm from the verbose output
    gradient_norm_values = []
    for line in lines_out:
        # When the computation is Finished just an old gradient norm value
        # is repeated that we do not need to store
        if "Finished" in line:
            break

        start_grad_norm = line.find("gradient norm")
        if start_grad_norm >= 0:
            line = line[start_grad_norm:]
            line = line.replace("gradient norm = ", "").split(" ")[0]
            gradient_norm_values.append(float(line))

    # Compute how often the gradient norm is smaller than min_grad_norm
    gradient_norm_values = np.array(gradient_norm_values)
    n_smaller_gradient_norms = len(
        gradient_norm_values[gradient_norm_values <= min_grad_norm]
    )

    # The gradient norm can be smaller than min_grad_norm at most once,
    # because in the moment it becomes smaller the optimization stops
    assert n_smaller_gradient_norms <= 1


def test_accessible_kl_divergence():
    # Ensures that the accessible kl_divergence matches the computed value
    random_state = check_random_state(0)
    X = random_state.randn(50, 2)
    tsne = TSNE(
        n_iter_without_progress=2, verbose=2, random_state=0, method="exact", n_iter=500
    )

    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        tsne.fit_transform(X)
    finally:
        out = sys.stdout.getvalue()
        sys.stdout.close()
        sys.stdout = old_stdout

    # The output needs to contain the accessible kl_divergence as the error at
    # the last iteration
    for line in out.split("\n")[::-1]:
        if "Iteration" in line:
            _, _, error = line.partition("error = ")
            if error:
                error, _, _ = error.partition(",")
                break
    assert_almost_equal(tsne.kl_divergence_, float(error), decimal=5)


@pytest.mark.parametrize("method", ["barnes_hut", "exact"])
def test_uniform_grid(method):
    """Make sure that TSNE can approximately recover a uniform 2D grid

    Due to ties in distances between point in X_2d_grid, this test is platform
    dependent for ``method='barnes_hut'`` due to numerical imprecision.

    Also, t-SNE is not assured to converge to the right solution because bad
    initialization can lead to convergence to bad local minimum (the
    optimization problem is non-convex). To avoid breaking the test too often,
    we re-run t-SNE from the final point when the convergence is not good
    enough.
    """
    seeds = range(3)
    n_iter = 500
    for seed in seeds:
        tsne = TSNE(
            n_components=2,
            init="random",
            random_state=seed,
            perplexity=50,
            n_iter=n_iter,
            method=method,
            learning_rate="auto",
        )
        Y = tsne.fit_transform(X_2d_grid)

        try_name = "{}_{}".format(method, seed)
        try:
            assert_uniform_grid(Y, try_name)
        except AssertionError:
            # If the test fails a first time, re-run with init=Y to see if
            # this was caused by a bad initialization. Note that this will
            # also run an early_exaggeration step.
            try_name += ":rerun"
            tsne.init = Y
            Y = tsne.fit_transform(X_2d_grid)
            assert_uniform_grid(Y, try_name)


def assert_uniform_grid(Y, try_name=None):
    # Ensure that the resulting embedding leads to approximately
    # uniformly spaced points: the distance to the closest neighbors
    # should be non-zero and approximately constant.
    nn = NearestNeighbors(n_neighbors=1).fit(Y)
    dist_to_nn = nn.kneighbors(return_distance=True)[0].ravel()
    assert dist_to_nn.min() > 0.1

    smallest_to_mean = dist_to_nn.min() / np.mean(dist_to_nn)
    largest_to_mean = dist_to_nn.max() / np.mean(dist_to_nn)

    assert smallest_to_mean > 0.5, try_name
    assert largest_to_mean < 2, try_name


def test_bh_match_exact():
    # check that the ``barnes_hut`` method match the exact one when
    # ``angle = 0`` and ``perplexity > n_samples / 3``
    random_state = check_random_state(0)
    n_features = 10
    X = random_state.randn(30, n_features).astype(np.float32)
    X_embeddeds = {}
    n_iter = {}
    for method in ["exact", "barnes_hut"]:
        tsne = TSNE(
            n_components=2,
            method=method,
            learning_rate=1.0,
            init="random",
            random_state=0,
            n_iter=251,
            perplexity=29.5,
            angle=0,
        )
        # Kill the early_exaggeration
        tsne._EXPLORATION_N_ITER = 0
        X_embeddeds[method] = tsne.fit_transform(X)
        n_iter[method] = tsne.n_iter_

    assert n_iter["exact"] == n_iter["barnes_hut"]
    assert_allclose(X_embeddeds["exact"], X_embeddeds["barnes_hut"], rtol=1e-4)


def test_gradient_bh_multithread_match_sequential():
    # check that the bh gradient with different num_threads gives the same
    # results

    n_features = 10
    n_samples = 30
    n_components = 2
    degrees_of_freedom = 1

    angle = 3
    perplexity = 5

    random_state = check_random_state(0)
    data = random_state.randn(n_samples, n_features).astype(np.float32)
    params = random_state.randn(n_samples, n_components)

    n_neighbors = n_samples - 1
    distances_csr = (
        NearestNeighbors()
        .fit(data)
        .kneighbors_graph(n_neighbors=n_neighbors, mode="distance")
    )
    P_bh = _joint_probabilities_nn(distances_csr, perplexity, verbose=0)
    kl_sequential, grad_sequential = _kl_divergence_bh(
        params,
        P_bh,
        degrees_of_freedom,
        n_samples,
        n_components,
        angle=angle,
        skip_num_points=0,
        verbose=0,
        num_threads=1,
    )
    for num_threads in [2, 4]:
        kl_multithread, grad_multithread = _kl_divergence_bh(
            params,
            P_bh,
            degrees_of_freedom,
            n_samples,
            n_components,
            angle=angle,
            skip_num_points=0,
            verbose=0,
            num_threads=num_threads,
        )

        assert_allclose(kl_multithread, kl_sequential, rtol=1e-6)
        assert_allclose(grad_multithread, grad_multithread)


@pytest.mark.parametrize(
    "metric, dist_func",
    [("manhattan", manhattan_distances), ("cosine", cosine_distances)],
)
@pytest.mark.parametrize("method", ["barnes_hut", "exact"])
def test_tsne_with_different_distance_metrics(metric, dist_func, method):
    """Make sure that TSNE works for different distance metrics"""

    if method == "barnes_hut" and metric == "manhattan":
        # The distances computed by `manhattan_distances` differ slightly from those
        # computed internally by NearestNeighbors via the PairwiseDistancesReduction
        # Cython code-based. This in turns causes T-SNE to converge to a different
        # solution but this should not impact the qualitative results as both
        # methods.
        # NOTE: it's probably not valid from a mathematical point of view to use the
        # Manhattan distance for T-SNE...
        # TODO: re-enable this test if/when `manhattan_distances` is refactored to
        # reuse the same underlying Cython code NearestNeighbors.
        # For reference, see:
        # https://github.com/scikit-learn/scikit-learn/pull/23865/files#r925721573
        pytest.xfail(
            "Distance computations are different for method == 'barnes_hut' and metric"
            " == 'manhattan', but this is expected."
        )

    random_state = check_random_state(0)
    n_components_original = 3
    n_components_embedding = 2
    X = random_state.randn(50, n_components_original).astype(np.float32)
    X_transformed_tsne = TSNE(
        metric=metric,
        method=method,
        n_components=n_components_embedding,
        random_state=0,
        n_iter=300,
        init="random",
        learning_rate="auto",
    ).fit_transform(X)
    X_transformed_tsne_precomputed = TSNE(
        metric="precomputed",
        method=method,
        n_components=n_components_embedding,
        random_state=0,
        n_iter=300,
        init="random",
        learning_rate="auto",
    ).fit_transform(dist_func(X))
    assert_array_equal(X_transformed_tsne, X_transformed_tsne_precomputed)


@pytest.mark.parametrize("method", ["exact", "barnes_hut"])
def test_tsne_n_jobs(method):
    """Make sure that the n_jobs parameter doesn't impact the output"""
    random_state = check_random_state(0)
    n_features = 10
    X = random_state.randn(30, n_features)
    X_tr_ref = TSNE(
        n_components=2,
        method=method,
        perplexity=25.0,
        angle=0,
        n_jobs=1,
        random_state=0,
        init="random",
        learning_rate="auto",
    ).fit_transform(X)
    X_tr = TSNE(
        n_components=2,
        method=method,
        perplexity=25.0,
        angle=0,
        n_jobs=2,
        random_state=0,
        init="random",
        learning_rate="auto",
    ).fit_transform(X)

    assert_allclose(X_tr_ref, X_tr)


def test_tsne_with_mahalanobis_distance():
    """Make sure that method_parameters works with mahalanobis distance."""
    random_state = check_random_state(0)
    n_samples, n_features = 300, 10
    X = random_state.randn(n_samples, n_features)
    default_params = {
        "perplexity": 40,
        "n_iter": 250,
        "learning_rate": "auto",
        "init": "random",
        "n_components": 3,
        "random_state": 0,
    }

    tsne = TSNE(metric="mahalanobis", **default_params)
    msg = "Must provide either V or VI for Mahalanobis distance"
    with pytest.raises(ValueError, match=msg):
        tsne.fit_transform(X)

    precomputed_X = squareform(pdist(X, metric="mahalanobis"), checks=True)
    X_trans_expected = TSNE(metric="precomputed", **default_params).fit_transform(
        precomputed_X
    )

    X_trans = TSNE(
        metric="mahalanobis", metric_params={"V": np.cov(X.T)}, **default_params
    ).fit_transform(X)
    assert_allclose(X_trans, X_trans_expected)


@pytest.mark.parametrize("perplexity", (20, 30))
def test_tsne_perplexity_validation(perplexity):
    """Make sure that perplexity > n_samples results in a ValueError"""

    random_state = check_random_state(0)
    X = random_state.randn(20, 2)
    est = TSNE(
        learning_rate="auto",
        init="pca",
        perplexity=perplexity,
        random_state=random_state,
    )
    msg = "perplexity must be less than n_samples"
    with pytest.raises(ValueError, match=msg):
        est.fit_transform(X)


def test_tsne_works_with_pandas_output():
    """Make sure that TSNE works when the output is set to "pandas".

    Non-regression test for gh-25365.
    """
    pytest.importorskip("pandas")
    with config_context(transform_output="pandas"):
        arr = np.arange(35 * 4).reshape(35, 4)
        TSNE(n_components=2).fit_transform(arr)
