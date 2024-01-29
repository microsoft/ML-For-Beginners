"""
Testing for the nearest centroid module.
"""
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from sklearn import datasets
from sklearn.neighbors import NearestCentroid
from sklearn.utils.fixes import CSR_CONTAINERS

# toy sample
X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
y = [-1, -1, -1, 1, 1, 1]
T = [[-1, -1], [2, 2], [3, 2]]
true_result = [-1, 1, 1]

# also load the iris dataset
# and randomly permute it
iris = datasets.load_iris()
rng = np.random.RandomState(1)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_classification_toy(csr_container):
    # Check classification on a toy dataset, including sparse versions.
    X_csr = csr_container(X)
    T_csr = csr_container(T)

    clf = NearestCentroid()
    clf.fit(X, y)
    assert_array_equal(clf.predict(T), true_result)

    # Same test, but with a sparse matrix to fit and test.
    clf = NearestCentroid()
    clf.fit(X_csr, y)
    assert_array_equal(clf.predict(T_csr), true_result)

    # Fit with sparse, test with non-sparse
    clf = NearestCentroid()
    clf.fit(X_csr, y)
    assert_array_equal(clf.predict(T), true_result)

    # Fit with non-sparse, test with sparse
    clf = NearestCentroid()
    clf.fit(X, y)
    assert_array_equal(clf.predict(T_csr), true_result)

    # Fit and predict with non-CSR sparse matrices
    clf = NearestCentroid()
    clf.fit(X_csr.tocoo(), y)
    assert_array_equal(clf.predict(T_csr.tolil()), true_result)


# TODO(1.5): Remove filterwarnings when support for some metrics is removed
@pytest.mark.filterwarnings("ignore:Support for distance metrics:FutureWarning:sklearn")
def test_iris():
    # Check consistency on dataset iris.
    for metric in ("euclidean", "cosine"):
        clf = NearestCentroid(metric=metric).fit(iris.data, iris.target)
        score = np.mean(clf.predict(iris.data) == iris.target)
        assert score > 0.9, "Failed with score = " + str(score)


# TODO(1.5): Remove filterwarnings when support for some metrics is removed
@pytest.mark.filterwarnings("ignore:Support for distance metrics:FutureWarning:sklearn")
def test_iris_shrinkage():
    # Check consistency on dataset iris, when using shrinkage.
    for metric in ("euclidean", "cosine"):
        for shrink_threshold in [None, 0.1, 0.5]:
            clf = NearestCentroid(metric=metric, shrink_threshold=shrink_threshold)
            clf = clf.fit(iris.data, iris.target)
            score = np.mean(clf.predict(iris.data) == iris.target)
            assert score > 0.8, "Failed with score = " + str(score)


def test_pickle():
    import pickle

    # classification
    obj = NearestCentroid()
    obj.fit(iris.data, iris.target)
    score = obj.score(iris.data, iris.target)
    s = pickle.dumps(obj)

    obj2 = pickle.loads(s)
    assert type(obj2) == obj.__class__
    score2 = obj2.score(iris.data, iris.target)
    assert_array_equal(
        score,
        score2,
        "Failed to generate same score after pickling (classification).",
    )


def test_shrinkage_correct():
    # Ensure that the shrinking is correct.
    # The expected result is calculated by R (pamr),
    # which is implemented by the author of the original paper.
    # (One need to modify the code to output the new centroid in pamr.predict)

    X = np.array([[0, 1], [1, 0], [1, 1], [2, 0], [6, 8]])
    y = np.array([1, 1, 2, 2, 2])
    clf = NearestCentroid(shrink_threshold=0.1)
    clf.fit(X, y)
    expected_result = np.array([[0.7787310, 0.8545292], [2.814179, 2.763647]])
    np.testing.assert_array_almost_equal(clf.centroids_, expected_result)


def test_shrinkage_threshold_decoded_y():
    clf = NearestCentroid(shrink_threshold=0.01)
    y_ind = np.asarray(y)
    y_ind[y_ind == -1] = 0
    clf.fit(X, y_ind)
    centroid_encoded = clf.centroids_
    clf.fit(X, y)
    assert_array_equal(centroid_encoded, clf.centroids_)


def test_predict_translated_data():
    # Test that NearestCentroid gives same results on translated data

    rng = np.random.RandomState(0)
    X = rng.rand(50, 50)
    y = rng.randint(0, 3, 50)
    noise = rng.rand(50)
    clf = NearestCentroid(shrink_threshold=0.1)
    clf.fit(X, y)
    y_init = clf.predict(X)
    clf = NearestCentroid(shrink_threshold=0.1)
    X_noise = X + noise
    clf.fit(X_noise, y)
    y_translate = clf.predict(X_noise)
    assert_array_equal(y_init, y_translate)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_manhattan_metric(csr_container):
    # Test the manhattan metric.
    X_csr = csr_container(X)

    clf = NearestCentroid(metric="manhattan")
    clf.fit(X, y)
    dense_centroid = clf.centroids_
    clf.fit(X_csr, y)
    assert_array_equal(clf.centroids_, dense_centroid)
    assert_array_equal(dense_centroid, [[-1, -1], [1, 1]])


# TODO(1.5): remove this test
@pytest.mark.parametrize(
    "metric", sorted(list(NearestCentroid._valid_metrics - {"manhattan", "euclidean"}))
)
def test_deprecated_distance_metric_supports(metric):
    # Check that a warning is raised for all deprecated distance metric supports
    clf = NearestCentroid(metric=metric)
    with pytest.warns(
        FutureWarning,
        match="Support for distance metrics other than euclidean and manhattan",
    ):
        clf.fit(X, y)


def test_features_zero_var():
    # Test that features with 0 variance throw error

    X = np.empty((10, 2))
    X[:, 0] = -0.13725701
    X[:, 1] = -0.9853293
    y = np.zeros((10))
    y[0] = 1

    clf = NearestCentroid(shrink_threshold=0.1)
    with pytest.raises(ValueError):
        clf.fit(X, y)
