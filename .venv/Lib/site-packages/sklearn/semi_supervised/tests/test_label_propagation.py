""" test the label propagation module """

import warnings

import numpy as np
import pytest
from scipy.sparse import issparse

from sklearn.datasets import make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.semi_supervised import _label_propagation as label_propagation
from sklearn.utils._testing import (
    _convert_container,
    assert_allclose,
    assert_array_equal,
)

CONSTRUCTOR_TYPES = ("array", "sparse_csr", "sparse_csc")

ESTIMATORS = [
    (label_propagation.LabelPropagation, {"kernel": "rbf"}),
    (label_propagation.LabelPropagation, {"kernel": "knn", "n_neighbors": 2}),
    (
        label_propagation.LabelPropagation,
        {"kernel": lambda x, y: rbf_kernel(x, y, gamma=20)},
    ),
    (label_propagation.LabelSpreading, {"kernel": "rbf"}),
    (label_propagation.LabelSpreading, {"kernel": "knn", "n_neighbors": 2}),
    (
        label_propagation.LabelSpreading,
        {"kernel": lambda x, y: rbf_kernel(x, y, gamma=20)},
    ),
]


@pytest.mark.parametrize("Estimator, parameters", ESTIMATORS)
def test_fit_transduction(global_dtype, Estimator, parameters):
    samples = np.asarray([[1.0, 0.0], [0.0, 2.0], [1.0, 3.0]], dtype=global_dtype)
    labels = [0, 1, -1]
    clf = Estimator(**parameters).fit(samples, labels)
    assert clf.transduction_[2] == 1


@pytest.mark.parametrize("Estimator, parameters", ESTIMATORS)
def test_distribution(global_dtype, Estimator, parameters):
    if parameters["kernel"] == "knn":
        pytest.skip(
            "Unstable test for this configuration: changes in k-NN ordering break it."
        )
    samples = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=global_dtype)
    labels = [0, 1, -1]
    clf = Estimator(**parameters).fit(samples, labels)
    assert_allclose(clf.label_distributions_[2], [0.5, 0.5], atol=1e-2)


@pytest.mark.parametrize("Estimator, parameters", ESTIMATORS)
def test_predict(global_dtype, Estimator, parameters):
    samples = np.asarray([[1.0, 0.0], [0.0, 2.0], [1.0, 3.0]], dtype=global_dtype)
    labels = [0, 1, -1]
    clf = Estimator(**parameters).fit(samples, labels)
    assert_array_equal(clf.predict([[0.5, 2.5]]), np.array([1]))


@pytest.mark.parametrize("Estimator, parameters", ESTIMATORS)
def test_predict_proba(global_dtype, Estimator, parameters):
    samples = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 2.5]], dtype=global_dtype)
    labels = [0, 1, -1]
    clf = Estimator(**parameters).fit(samples, labels)
    assert_allclose(clf.predict_proba([[1.0, 1.0]]), np.array([[0.5, 0.5]]))


@pytest.mark.parametrize("alpha", [0.1, 0.3, 0.5, 0.7, 0.9])
@pytest.mark.parametrize("Estimator, parameters", ESTIMATORS)
def test_label_spreading_closed_form(global_dtype, Estimator, parameters, alpha):
    n_classes = 2
    X, y = make_classification(n_classes=n_classes, n_samples=200, random_state=0)
    X = X.astype(global_dtype, copy=False)
    y[::3] = -1

    gamma = 0.1
    clf = label_propagation.LabelSpreading(gamma=gamma).fit(X, y)
    # adopting notation from Zhou et al (2004):
    S = clf._build_graph()
    Y = np.zeros((len(y), n_classes + 1), dtype=X.dtype)
    Y[np.arange(len(y)), y] = 1
    Y = Y[:, :-1]

    expected = np.dot(np.linalg.inv(np.eye(len(S), dtype=S.dtype) - alpha * S), Y)
    expected /= expected.sum(axis=1)[:, np.newaxis]

    clf = label_propagation.LabelSpreading(
        max_iter=100, alpha=alpha, tol=1e-10, gamma=gamma
    )
    clf.fit(X, y)

    assert_allclose(expected, clf.label_distributions_)


def test_label_propagation_closed_form(global_dtype):
    n_classes = 2
    X, y = make_classification(n_classes=n_classes, n_samples=200, random_state=0)
    X = X.astype(global_dtype, copy=False)
    y[::3] = -1
    Y = np.zeros((len(y), n_classes + 1))
    Y[np.arange(len(y)), y] = 1
    unlabelled_idx = Y[:, (-1,)].nonzero()[0]
    labelled_idx = (Y[:, (-1,)] == 0).nonzero()[0]

    clf = label_propagation.LabelPropagation(max_iter=100, tol=1e-10, gamma=0.1)
    clf.fit(X, y)
    # adopting notation from Zhu et al 2002
    T_bar = clf._build_graph()
    Tuu = T_bar[tuple(np.meshgrid(unlabelled_idx, unlabelled_idx, indexing="ij"))]
    Tul = T_bar[tuple(np.meshgrid(unlabelled_idx, labelled_idx, indexing="ij"))]
    Y = Y[:, :-1]
    Y_l = Y[labelled_idx, :]
    Y_u = np.dot(np.dot(np.linalg.inv(np.eye(Tuu.shape[0]) - Tuu), Tul), Y_l)

    expected = Y.copy()
    expected[unlabelled_idx, :] = Y_u
    expected /= expected.sum(axis=1)[:, np.newaxis]

    assert_allclose(expected, clf.label_distributions_, atol=1e-4)


@pytest.mark.parametrize("accepted_sparse_type", ["sparse_csr", "sparse_csc"])
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("Estimator, parameters", ESTIMATORS)
def test_sparse_input_types(
    accepted_sparse_type, index_dtype, dtype, Estimator, parameters
):
    # This is non-regression test for #17085
    X = _convert_container([[1.0, 0.0], [0.0, 2.0], [1.0, 3.0]], accepted_sparse_type)
    X.data = X.data.astype(dtype, copy=False)
    X.indices = X.indices.astype(index_dtype, copy=False)
    X.indptr = X.indptr.astype(index_dtype, copy=False)
    labels = [0, 1, -1]
    clf = Estimator(**parameters).fit(X, labels)
    assert_array_equal(clf.predict([[0.5, 2.5]]), np.array([1]))


@pytest.mark.parametrize("constructor_type", CONSTRUCTOR_TYPES)
def test_convergence_speed(constructor_type):
    # This is a non-regression test for #5774
    X = _convert_container([[1.0, 0.0], [0.0, 1.0], [1.0, 2.5]], constructor_type)
    y = np.array([0, 1, -1])
    mdl = label_propagation.LabelSpreading(kernel="rbf", max_iter=5000)
    mdl.fit(X, y)

    # this should converge quickly:
    assert mdl.n_iter_ < 10
    assert_array_equal(mdl.predict(X), [0, 1, 1])


def test_convergence_warning():
    # This is a non-regression test for #5774
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 2.5]])
    y = np.array([0, 1, -1])
    mdl = label_propagation.LabelSpreading(kernel="rbf", max_iter=1)
    warn_msg = "max_iter=1 was reached without convergence."
    with pytest.warns(ConvergenceWarning, match=warn_msg):
        mdl.fit(X, y)
    assert mdl.n_iter_ == mdl.max_iter

    mdl = label_propagation.LabelPropagation(kernel="rbf", max_iter=1)
    with pytest.warns(ConvergenceWarning, match=warn_msg):
        mdl.fit(X, y)
    assert mdl.n_iter_ == mdl.max_iter

    mdl = label_propagation.LabelSpreading(kernel="rbf", max_iter=500)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        mdl.fit(X, y)

    mdl = label_propagation.LabelPropagation(kernel="rbf", max_iter=500)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        mdl.fit(X, y)


@pytest.mark.parametrize(
    "LabelPropagationCls",
    [label_propagation.LabelSpreading, label_propagation.LabelPropagation],
)
def test_label_propagation_non_zero_normalizer(LabelPropagationCls):
    # check that we don't divide by zero in case of null normalizer
    # non-regression test for
    # https://github.com/scikit-learn/scikit-learn/pull/15946
    # https://github.com/scikit-learn/scikit-learn/issues/9292
    X = np.array([[100.0, 100.0], [100.0, 100.0], [0.0, 0.0], [0.0, 0.0]])
    y = np.array([0, 1, -1, -1])
    mdl = LabelPropagationCls(kernel="knn", max_iter=100, n_neighbors=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        mdl.fit(X, y)


def test_predict_sparse_callable_kernel(global_dtype):
    # This is a non-regression test for #15866

    # Custom sparse kernel (top-K RBF)
    def topk_rbf(X, Y=None, n_neighbors=10, gamma=1e-5):
        nn = NearestNeighbors(n_neighbors=10, metric="euclidean", n_jobs=2)
        nn.fit(X)
        W = -1 * nn.kneighbors_graph(Y, mode="distance").power(2) * gamma
        np.exp(W.data, out=W.data)
        assert issparse(W)
        return W.T

    n_classes = 4
    n_samples = 500
    n_test = 10
    X, y = make_classification(
        n_classes=n_classes,
        n_samples=n_samples,
        n_features=20,
        n_informative=20,
        n_redundant=0,
        n_repeated=0,
        random_state=0,
    )
    X = X.astype(global_dtype)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=n_test, random_state=0
    )

    model = label_propagation.LabelSpreading(kernel=topk_rbf)
    model.fit(X_train, y_train)
    assert model.score(X_test, y_test) >= 0.9

    model = label_propagation.LabelPropagation(kernel=topk_rbf)
    model.fit(X_train, y_train)
    assert model.score(X_test, y_test) >= 0.9
