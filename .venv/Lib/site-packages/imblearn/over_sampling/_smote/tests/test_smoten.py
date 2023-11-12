import numpy as np
import pytest
from sklearn.exceptions import DataConversionWarning
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils._testing import _convert_container

from imblearn.over_sampling import SMOTEN


@pytest.fixture
def data():
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


def test_smoten(data):
    # overall check for SMOTEN
    X, y = data
    sampler = SMOTEN(random_state=0)
    X_res, y_res = sampler.fit_resample(X, y)

    assert X_res.shape == (80, 3)
    assert y_res.shape == (80,)
    assert isinstance(sampler.categorical_encoder_, OrdinalEncoder)


def test_smoten_resampling():
    # check if the SMOTEN resample data as expected
    # we generate data such that "not apple" will be the minority class and
    # samples from this class will be generated. We will force the "blue"
    # category to be associated with this class. Therefore, the new generated
    # samples should as well be from the "blue" category.
    X = np.array(["green"] * 5 + ["red"] * 10 + ["blue"] * 7, dtype=object).reshape(
        -1, 1
    )
    y = np.array(
        ["apple"] * 5
        + ["not apple"] * 3
        + ["apple"] * 7
        + ["not apple"] * 5
        + ["apple"] * 2,
        dtype=object,
    )
    sampler = SMOTEN(random_state=0)
    X_res, y_res = sampler.fit_resample(X, y)

    X_generated, y_generated = X_res[X.shape[0] :], y_res[X.shape[0] :]
    np.testing.assert_array_equal(X_generated, "blue")
    np.testing.assert_array_equal(y_generated, "not apple")


@pytest.mark.parametrize("sparse_format", ["sparse_csr", "sparse_csc"])
def test_smoten_sparse_input(data, sparse_format):
    """Check that we handle sparse input in SMOTEN even if it is not efficient.

    Non-regression test for:
    https://github.com/scikit-learn-contrib/imbalanced-learn/issues/971
    """
    X, y = data
    X = OneHotEncoder().fit_transform(X)
    X = _convert_container(X, sparse_format)

    with pytest.warns(DataConversionWarning, match="is not really efficient"):
        X_res, y_res = SMOTEN(random_state=0).fit_resample(X, y)

    assert X_res.format == X.format
    assert X_res.shape[0] == len(y_res)


def test_smoten_categorical_encoder(data):
    """Check that `categorical_encoder` is used when provided."""

    X, y = data
    sampler = SMOTEN(random_state=0)
    sampler.fit_resample(X, y)

    assert isinstance(sampler.categorical_encoder_, OrdinalEncoder)
    assert sampler.categorical_encoder_.dtype == np.int32

    encoder = OrdinalEncoder(dtype=np.int64)
    sampler.set_params(categorical_encoder=encoder).fit_resample(X, y)

    assert isinstance(sampler.categorical_encoder_, OrdinalEncoder)
    assert sampler.categorical_encoder is encoder
    assert sampler.categorical_encoder_ is not encoder
    assert sampler.categorical_encoder_.dtype == np.int64
