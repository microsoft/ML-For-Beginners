"""
Testing for Clustering methods

"""

import warnings

import numpy as np
import pytest

from sklearn.cluster import AffinityPropagation, affinity_propagation
from sklearn.cluster._affinity_propagation import _equal_similarities_and_preferences
from sklearn.datasets import make_blobs
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.metrics import euclidean_distances
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import CSR_CONTAINERS

n_clusters = 3
centers = np.array([[1, 1], [-1, -1], [1, -1]]) + 10
X, _ = make_blobs(
    n_samples=60,
    n_features=2,
    centers=centers,
    cluster_std=0.4,
    shuffle=True,
    random_state=0,
)

# TODO: AffinityPropagation must preserve dtype for its fitted attributes
# and test must be created accordingly to this new behavior.
# For more details, see: https://github.com/scikit-learn/scikit-learn/issues/11000


def test_affinity_propagation(global_random_seed, global_dtype):
    """Test consistency of the affinity propagations."""
    S = -euclidean_distances(X.astype(global_dtype, copy=False), squared=True)
    preference = np.median(S) * 10
    cluster_centers_indices, labels = affinity_propagation(
        S, preference=preference, random_state=global_random_seed
    )

    n_clusters_ = len(cluster_centers_indices)

    assert n_clusters == n_clusters_


def test_affinity_propagation_precomputed():
    """Check equality of precomputed affinity matrix to internally computed affinity
    matrix.
    """
    S = -euclidean_distances(X, squared=True)
    preference = np.median(S) * 10
    af = AffinityPropagation(
        preference=preference, affinity="precomputed", random_state=28
    )
    labels_precomputed = af.fit(S).labels_

    af = AffinityPropagation(preference=preference, verbose=True, random_state=37)
    labels = af.fit(X).labels_

    assert_array_equal(labels, labels_precomputed)

    cluster_centers_indices = af.cluster_centers_indices_

    n_clusters_ = len(cluster_centers_indices)
    assert np.unique(labels).size == n_clusters_
    assert n_clusters == n_clusters_


def test_affinity_propagation_no_copy():
    """Check behaviour of not copying the input data."""
    S = -euclidean_distances(X, squared=True)
    S_original = S.copy()
    preference = np.median(S) * 10
    assert not np.allclose(S.diagonal(), preference)

    # with copy=True S should not be modified
    affinity_propagation(S, preference=preference, copy=True, random_state=0)
    assert_allclose(S, S_original)
    assert not np.allclose(S.diagonal(), preference)
    assert_allclose(S.diagonal(), np.zeros(S.shape[0]))

    # with copy=False S will be modified inplace
    affinity_propagation(S, preference=preference, copy=False, random_state=0)
    assert_allclose(S.diagonal(), preference)

    # test that copy=True and copy=False lead to the same result
    S = S_original.copy()
    af = AffinityPropagation(preference=preference, verbose=True, random_state=0)

    labels = af.fit(X).labels_
    _, labels_no_copy = affinity_propagation(
        S, preference=preference, copy=False, random_state=74
    )
    assert_array_equal(labels, labels_no_copy)


def test_affinity_propagation_affinity_shape():
    """Check the shape of the affinity matrix when using `affinity_propagation."""
    S = -euclidean_distances(X, squared=True)
    err_msg = "The matrix of similarities must be a square array"
    with pytest.raises(ValueError, match=err_msg):
        affinity_propagation(S[:, :-1])


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_affinity_propagation_precomputed_with_sparse_input(csr_container):
    err_msg = "Sparse data was passed for X, but dense data is required"
    with pytest.raises(TypeError, match=err_msg):
        AffinityPropagation(affinity="precomputed").fit(csr_container((3, 3)))


def test_affinity_propagation_predict(global_random_seed, global_dtype):
    # Test AffinityPropagation.predict
    af = AffinityPropagation(affinity="euclidean", random_state=global_random_seed)
    X_ = X.astype(global_dtype, copy=False)
    labels = af.fit_predict(X_)
    labels2 = af.predict(X_)
    assert_array_equal(labels, labels2)


def test_affinity_propagation_predict_error():
    # Test exception in AffinityPropagation.predict
    # Not fitted.
    af = AffinityPropagation(affinity="euclidean")
    with pytest.raises(NotFittedError):
        af.predict(X)

    # Predict not supported when affinity="precomputed".
    S = np.dot(X, X.T)
    af = AffinityPropagation(affinity="precomputed", random_state=57)
    af.fit(S)
    with pytest.raises(ValueError, match="expecting 60 features as input"):
        af.predict(X)


def test_affinity_propagation_fit_non_convergence(global_dtype):
    # In case of non-convergence of affinity_propagation(), the cluster
    # centers should be an empty array and training samples should be labelled
    # as noise (-1)
    X = np.array([[0, 0], [1, 1], [-2, -2]], dtype=global_dtype)

    # Force non-convergence by allowing only a single iteration
    af = AffinityPropagation(preference=-10, max_iter=1, random_state=82)

    with pytest.warns(ConvergenceWarning):
        af.fit(X)
    assert_allclose(np.empty((0, 2)), af.cluster_centers_)
    assert_array_equal(np.array([-1, -1, -1]), af.labels_)


def test_affinity_propagation_equal_mutual_similarities(global_dtype):
    X = np.array([[-1, 1], [1, -1]], dtype=global_dtype)
    S = -euclidean_distances(X, squared=True)

    # setting preference > similarity
    with pytest.warns(UserWarning, match="mutually equal"):
        cluster_center_indices, labels = affinity_propagation(S, preference=0)

    # expect every sample to become an exemplar
    assert_array_equal([0, 1], cluster_center_indices)
    assert_array_equal([0, 1], labels)

    # setting preference < similarity
    with pytest.warns(UserWarning, match="mutually equal"):
        cluster_center_indices, labels = affinity_propagation(S, preference=-10)

    # expect one cluster, with arbitrary (first) sample as exemplar
    assert_array_equal([0], cluster_center_indices)
    assert_array_equal([0, 0], labels)

    # setting different preferences
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        cluster_center_indices, labels = affinity_propagation(
            S, preference=[-20, -10], random_state=37
        )

    # expect one cluster, with highest-preference sample as exemplar
    assert_array_equal([1], cluster_center_indices)
    assert_array_equal([0, 0], labels)


def test_affinity_propagation_predict_non_convergence(global_dtype):
    # In case of non-convergence of affinity_propagation(), the cluster
    # centers should be an empty array
    X = np.array([[0, 0], [1, 1], [-2, -2]], dtype=global_dtype)

    # Force non-convergence by allowing only a single iteration
    with pytest.warns(ConvergenceWarning):
        af = AffinityPropagation(preference=-10, max_iter=1, random_state=75).fit(X)

    # At prediction time, consider new samples as noise since there are no
    # clusters
    to_predict = np.array([[2, 2], [3, 3], [4, 4]])
    with pytest.warns(ConvergenceWarning):
        y = af.predict(to_predict)
    assert_array_equal(np.array([-1, -1, -1]), y)


def test_affinity_propagation_non_convergence_regressiontest(global_dtype):
    X = np.array(
        [[1, 0, 0, 0, 0, 0], [0, 1, 1, 1, 0, 0], [0, 0, 1, 0, 0, 1]], dtype=global_dtype
    )
    af = AffinityPropagation(affinity="euclidean", max_iter=2, random_state=34)
    msg = (
        "Affinity propagation did not converge, this model may return degenerate"
        " cluster centers and labels."
    )
    with pytest.warns(ConvergenceWarning, match=msg):
        af.fit(X)

    assert_array_equal(np.array([0, 0, 0]), af.labels_)


def test_equal_similarities_and_preferences(global_dtype):
    # Unequal distances
    X = np.array([[0, 0], [1, 1], [-2, -2]], dtype=global_dtype)
    S = -euclidean_distances(X, squared=True)

    assert not _equal_similarities_and_preferences(S, np.array(0))
    assert not _equal_similarities_and_preferences(S, np.array([0, 0]))
    assert not _equal_similarities_and_preferences(S, np.array([0, 1]))

    # Equal distances
    X = np.array([[0, 0], [1, 1]], dtype=global_dtype)
    S = -euclidean_distances(X, squared=True)

    # Different preferences
    assert not _equal_similarities_and_preferences(S, np.array([0, 1]))

    # Same preferences
    assert _equal_similarities_and_preferences(S, np.array([0, 0]))
    assert _equal_similarities_and_preferences(S, np.array(0))


def test_affinity_propagation_random_state():
    """Check that different random states lead to different initialisations
    by looking at the center locations after two iterations.
    """
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(
        n_samples=300, centers=centers, cluster_std=0.5, random_state=0
    )
    # random_state = 0
    ap = AffinityPropagation(convergence_iter=1, max_iter=2, random_state=0)
    ap.fit(X)
    centers0 = ap.cluster_centers_

    # random_state = 76
    ap = AffinityPropagation(convergence_iter=1, max_iter=2, random_state=76)
    ap.fit(X)
    centers76 = ap.cluster_centers_
    # check that the centers have not yet converged to the same solution
    assert np.mean((centers0 - centers76) ** 2) > 1


@pytest.mark.parametrize("container", CSR_CONTAINERS + [np.array])
def test_affinity_propagation_convergence_warning_dense_sparse(container, global_dtype):
    """
    Check that having sparse or dense `centers` format should not
    influence the convergence.
    Non-regression test for gh-13334.
    """
    centers = container(np.zeros((1, 10)))
    rng = np.random.RandomState(42)
    X = rng.rand(40, 10).astype(global_dtype, copy=False)
    y = (4 * rng.rand(40)).astype(int)
    ap = AffinityPropagation(random_state=46)
    ap.fit(X, y)
    ap.cluster_centers_ = centers
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        assert_array_equal(ap.predict(X), np.zeros(X.shape[0], dtype=int))


# FIXME; this test is broken with different random states, needs to be revisited
def test_correct_clusters(global_dtype):
    # Test to fix incorrect clusters due to dtype change
    # (non-regression test for issue #10832)
    X = np.array(
        [[1, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 1]], dtype=global_dtype
    )
    afp = AffinityPropagation(preference=1, affinity="precomputed", random_state=0).fit(
        X
    )
    expected = np.array([0, 1, 1, 2])
    assert_array_equal(afp.labels_, expected)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_sparse_input_for_predict(csr_container):
    # Test to make sure sparse inputs are accepted for predict
    # (non-regression test for issue #20049)
    af = AffinityPropagation(affinity="euclidean", random_state=42)
    af.fit(X)
    labels = af.predict(csr_container((2, 2)))
    assert_array_equal(labels, (2, 2))


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_sparse_input_for_fit_predict(csr_container):
    # Test to make sure sparse inputs are accepted for fit_predict
    # (non-regression test for issue #20049)
    af = AffinityPropagation(affinity="euclidean", random_state=42)
    rng = np.random.RandomState(42)
    X = csr_container(rng.randint(0, 2, size=(5, 5)))
    labels = af.fit_predict(X)
    assert_array_equal(labels, (0, 1, 1, 2, 3))
