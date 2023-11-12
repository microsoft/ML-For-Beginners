"""Test for the metrics that perform pairwise distance computation."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.utils._testing import _convert_container

from imblearn.metrics.pairwise import ValueDifferenceMetric


@pytest.fixture
def data():
    rng = np.random.RandomState(0)

    feature_1 = ["A"] * 10 + ["B"] * 20 + ["C"] * 30
    feature_2 = ["A"] * 40 + ["B"] * 20
    feature_3 = ["A"] * 20 + ["B"] * 20 + ["C"] * 10 + ["D"] * 10
    X = np.array([feature_1, feature_2, feature_3], dtype=object).T
    rng.shuffle(X)
    y = rng.randint(low=0, high=2, size=X.shape[0])
    y_labels = np.array(["not apple", "apple"], dtype=object)
    y = y_labels[y]
    return X, y


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
@pytest.mark.parametrize("k, r", [(1, 1), (1, 2), (2, 1), (2, 2)])
@pytest.mark.parametrize("y_type", ["list", "array"])
@pytest.mark.parametrize("encode_label", [True, False])
def test_value_difference_metric(data, dtype, k, r, y_type, encode_label):
    # Check basic feature of the metric:
    # * the shape of the distance matrix is (n_samples, n_samples)
    # * computing pairwise distance of X is the same than explicitely between
    #   X and X.
    X, y = data
    y = _convert_container(y, y_type)
    if encode_label:
        y = LabelEncoder().fit_transform(y)

    encoder = OrdinalEncoder(dtype=dtype)
    X_encoded = encoder.fit_transform(X)

    vdm = ValueDifferenceMetric(k=k, r=r)
    vdm.fit(X_encoded, y)

    dist_1 = vdm.pairwise(X_encoded)
    dist_2 = vdm.pairwise(X_encoded, X_encoded)

    np.testing.assert_allclose(dist_1, dist_2)
    assert dist_1.shape == (X.shape[0], X.shape[0])
    assert dist_2.shape == (X.shape[0], X.shape[0])


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
@pytest.mark.parametrize("k, r", [(1, 1), (1, 2), (2, 1), (2, 2)])
@pytest.mark.parametrize("y_type", ["list", "array"])
@pytest.mark.parametrize("encode_label", [True, False])
def test_value_difference_metric_property(dtype, k, r, y_type, encode_label):
    # Check the property of the vdm distance. Let's check the property
    # described in "Improved Heterogeneous Distance Functions", D.R. Wilson and
    # T.R. Martinez, Journal of Artificial Intelligence Research 6 (1997) 1-34
    # https://arxiv.org/pdf/cs/9701101.pdf
    #
    # "if an attribute color has three values red, green and blue, and the
    # application is to identify whether or not an object is an apple, red and
    # green would be considered closer than red and blue because the former two
    # both have similar correlations with the output class apple."

    # defined our feature
    X = np.array(["green"] * 10 + ["red"] * 10 + ["blue"] * 10).reshape(-1, 1)
    # 0 - not an apple / 1 - an apple
    y = np.array([1] * 8 + [0] * 5 + [1] * 7 + [0] * 9 + [1])
    y_labels = np.array(["not apple", "apple"], dtype=object)
    y = y_labels[y]
    y = _convert_container(y, y_type)
    if encode_label:
        y = LabelEncoder().fit_transform(y)

    encoder = OrdinalEncoder(dtype=dtype)
    X_encoded = encoder.fit_transform(X)

    vdm = ValueDifferenceMetric(k=k, r=r)
    vdm.fit(X_encoded, y)

    sample_green = encoder.transform([["green"]])
    sample_red = encoder.transform([["red"]])
    sample_blue = encoder.transform([["blue"]])

    for sample in (sample_green, sample_red, sample_blue):
        # computing the distance between a sample of the same category should
        # give a null distance
        dist = vdm.pairwise(sample).squeeze()
        assert dist == pytest.approx(0)

    # check the property explained in the introduction example
    dist_1 = vdm.pairwise(sample_green, sample_red).squeeze()
    dist_2 = vdm.pairwise(sample_blue, sample_red).squeeze()
    dist_3 = vdm.pairwise(sample_blue, sample_green).squeeze()

    # green and red are very close
    # blue is closer to red than green
    assert dist_1 < dist_2
    assert dist_1 < dist_3
    assert dist_2 < dist_3


def test_value_difference_metric_categories(data):
    # Check that "auto" is equivalent to provide the number categories
    # beforehand
    X, y = data

    encoder = OrdinalEncoder(dtype=np.int32)
    X_encoded = encoder.fit_transform(X)
    n_categories = np.array([len(cat) for cat in encoder.categories_])

    vdm_auto = ValueDifferenceMetric().fit(X_encoded, y)
    vdm_categories = ValueDifferenceMetric(n_categories=n_categories)
    vdm_categories.fit(X_encoded, y)

    np.testing.assert_array_equal(vdm_auto.n_categories_, n_categories)
    np.testing.assert_array_equal(vdm_auto.n_categories_, vdm_categories.n_categories_)


def test_value_difference_metric_categories_error(data):
    # Check that we raise an error if n_categories is inconsistent with the
    # number of features in X
    X, y = data

    encoder = OrdinalEncoder(dtype=np.int32)
    X_encoded = encoder.fit_transform(X)
    n_categories = [1, 2]

    vdm = ValueDifferenceMetric(n_categories=n_categories)
    err_msg = "The length of n_categories is not consistent with the number"
    with pytest.raises(ValueError, match=err_msg):
        vdm.fit(X_encoded, y)


def test_value_difference_metric_missing_categories(data):
    # Check that we don't get issue when a category is missing between 0
    # n_categories - 1
    X, y = data

    encoder = OrdinalEncoder(dtype=np.int32)
    X_encoded = encoder.fit_transform(X)
    n_categories = np.array([len(cat) for cat in encoder.categories_])

    # remove a categories that could be between 0 and n_categories
    X_encoded[X_encoded[:, -1] == 1] = 0
    np.testing.assert_array_equal(np.unique(X_encoded[:, -1]), [0, 2, 3])

    vdm = ValueDifferenceMetric(n_categories=n_categories)
    vdm.fit(X_encoded, y)

    for n_cats, proba in zip(n_categories, vdm.proba_per_class_):
        assert proba.shape == (n_cats, len(np.unique(y)))


def test_value_difference_value_unfitted(data):
    # Check that we raise a NotFittedError when `fit` is not not called before
    # pairwise.
    X, y = data

    encoder = OrdinalEncoder(dtype=np.int32)
    X_encoded = encoder.fit_transform(X)

    with pytest.raises(NotFittedError):
        ValueDifferenceMetric().pairwise(X_encoded)
