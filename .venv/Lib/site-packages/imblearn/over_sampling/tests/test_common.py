from collections import Counter

import numpy as np
import pytest
from sklearn.cluster import MiniBatchKMeans

from imblearn.over_sampling import (
    ADASYN,
    SMOTE,
    SMOTEN,
    SMOTENC,
    SVMSMOTE,
    BorderlineSMOTE,
    KMeansSMOTE,
)
from imblearn.utils.testing import _CustomNearestNeighbors


@pytest.fixture
def numerical_data():
    rng = np.random.RandomState(0)
    X = rng.randn(100, 2)
    y = np.repeat([0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0], 5)

    return X, y


@pytest.fixture
def categorical_data():
    rng = np.random.RandomState(0)

    feature_1 = ["A"] * 10 + ["B"] * 20 + ["C"] * 30
    feature_2 = ["A"] * 40 + ["B"] * 20
    feature_3 = ["A"] * 20 + ["B"] * 20 + ["C"] * 10 + ["D"] * 10
    X = np.array([feature_1, feature_2, feature_3], dtype=object).T
    rng.shuffle(X)
    y = np.array([0] * 20 + [1] * 40, dtype=np.int32)
    y_labels = np.array(["not apple", "apple"], dtype=object)
    y = y_labels[y]
    return X, y


@pytest.fixture
def heterogeneous_data():
    rng = np.random.RandomState(42)
    X = np.empty((30, 4), dtype=object)
    X[:, :2] = rng.randn(30, 2)
    X[:, 2] = rng.choice(["a", "b", "c"], size=30).astype(object)
    X[:, 3] = rng.randint(3, size=30)
    y = np.array([0] * 10 + [1] * 20)
    return X, y, [2, 3]


@pytest.mark.parametrize(
    "smote", [BorderlineSMOTE(), SVMSMOTE()], ids=["borderline", "svm"]
)
def test_smote_m_neighbors(numerical_data, smote):
    # check that m_neighbors is properly set. Regression test for:
    # https://github.com/scikit-learn-contrib/imbalanced-learn/issues/568
    X, y = numerical_data
    _ = smote.fit_resample(X, y)
    assert smote.nn_k_.n_neighbors == 6
    assert smote.nn_m_.n_neighbors == 11


@pytest.mark.parametrize(
    "smote, neighbor_estimator_name",
    [
        (ADASYN(random_state=0), "n_neighbors"),
        (BorderlineSMOTE(random_state=0), "k_neighbors"),
        (
            KMeansSMOTE(
                kmeans_estimator=MiniBatchKMeans(n_init=1, random_state=0),
                random_state=1,
            ),
            "k_neighbors",
        ),
        (SMOTE(random_state=0), "k_neighbors"),
        (SVMSMOTE(random_state=0), "k_neighbors"),
    ],
    ids=["adasyn", "borderline", "kmeans", "smote", "svm"],
)
def test_numerical_smote_custom_nn(numerical_data, smote, neighbor_estimator_name):
    X, y = numerical_data
    params = {
        neighbor_estimator_name: _CustomNearestNeighbors(n_neighbors=5),
    }
    smote.set_params(**params)
    X_res, _ = smote.fit_resample(X, y)

    assert X_res.shape[0] >= 120


def test_categorical_smote_k_custom_nn(categorical_data):
    X, y = categorical_data
    smote = SMOTEN(k_neighbors=_CustomNearestNeighbors(n_neighbors=5))
    X_res, y_res = smote.fit_resample(X, y)

    assert X_res.shape == (80, 3)
    assert Counter(y_res) == {"apple": 40, "not apple": 40}


def test_heterogeneous_smote_k_custom_nn(heterogeneous_data):
    X, y, categorical_features = heterogeneous_data
    smote = SMOTENC(
        categorical_features, k_neighbors=_CustomNearestNeighbors(n_neighbors=5)
    )
    X_res, y_res = smote.fit_resample(X, y)

    assert X_res.shape == (40, 4)
    assert Counter(y_res) == {0: 20, 1: 20}


@pytest.mark.parametrize(
    "smote",
    [BorderlineSMOTE(random_state=0), SVMSMOTE(random_state=0)],
    ids=["borderline", "svm"],
)
def test_numerical_smote_extra_custom_nn(numerical_data, smote):
    X, y = numerical_data
    smote.set_params(m_neighbors=_CustomNearestNeighbors(n_neighbors=5))
    X_res, y_res = smote.fit_resample(X, y)

    assert X_res.shape == (120, 2)
    assert Counter(y_res) == {0: 60, 1: 60}


# FIXME: to be removed in 0.12
@pytest.mark.parametrize(
    "sampler",
    [
        ADASYN(random_state=0),
        BorderlineSMOTE(random_state=0),
        SMOTE(random_state=0),
        SMOTEN(random_state=0),
        SMOTENC([0], random_state=0),
        SVMSMOTE(random_state=0),
    ],
)
def test_n_jobs_deprecation_warning(numerical_data, sampler):
    X, y = numerical_data
    sampler.set_params(n_jobs=2)
    warning_msg = "The parameter `n_jobs` has been deprecated"
    with pytest.warns(FutureWarning, match=warning_msg):
        sampler.fit_resample(X, y)
