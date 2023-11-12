"""Test the module SMOTENC."""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
#          Dzianis Dudnik
# License: MIT

from collections import Counter

import numpy as np
import pytest
import sklearn
from scipy import sparse
from sklearn.datasets import make_classification
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import parse_version

from imblearn.over_sampling import SMOTENC
from imblearn.utils.estimator_checks import (
    _set_checking_parameters,
    check_param_validation,
)

sklearn_version = parse_version(sklearn.__version__)


def data_heterogneous_ordered():
    rng = np.random.RandomState(42)
    X = np.empty((30, 4), dtype=object)
    # create 2 random continuous feature
    X[:, :2] = rng.randn(30, 2)
    # create a categorical feature using some string
    X[:, 2] = rng.choice(["a", "b", "c"], size=30).astype(object)
    # create a categorical feature using some integer
    X[:, 3] = rng.randint(3, size=30)
    y = np.array([0] * 10 + [1] * 20)
    # return the categories
    return X, y, [2, 3]


def data_heterogneous_unordered():
    rng = np.random.RandomState(42)
    X = np.empty((30, 4), dtype=object)
    # create 2 random continuous feature
    X[:, [1, 2]] = rng.randn(30, 2)
    # create a categorical feature using some string
    X[:, 0] = rng.choice(["a", "b", "c"], size=30).astype(object)
    # create a categorical feature using some integer
    X[:, 3] = rng.randint(3, size=30)
    y = np.array([0] * 10 + [1] * 20)
    # return the categories
    return X, y, [0, 3]


def data_heterogneous_masked():
    rng = np.random.RandomState(42)
    X = np.empty((30, 4), dtype=object)
    # create 2 random continuous feature
    X[:, [1, 2]] = rng.randn(30, 2)
    # create a categorical feature using some string
    X[:, 0] = rng.choice(["a", "b", "c"], size=30).astype(object)
    # create a categorical feature using some integer
    X[:, 3] = rng.randint(3, size=30)
    y = np.array([0] * 10 + [1] * 20)
    # return the categories
    return X, y, [True, False, False, True]


def data_heterogneous_unordered_multiclass():
    rng = np.random.RandomState(42)
    X = np.empty((50, 4), dtype=object)
    # create 2 random continuous feature
    X[:, [1, 2]] = rng.randn(50, 2)
    # create a categorical feature using some string
    X[:, 0] = rng.choice(["a", "b", "c"], size=50).astype(object)
    # create a categorical feature using some integer
    X[:, 3] = rng.randint(3, size=50)
    y = np.array([0] * 10 + [1] * 15 + [2] * 25)
    # return the categories
    return X, y, [0, 3]


def data_sparse(format):
    rng = np.random.RandomState(42)
    X = np.empty((30, 4), dtype=np.float64)
    # create 2 random continuous feature
    X[:, [1, 2]] = rng.randn(30, 2)
    # create a categorical feature using some string
    X[:, 0] = rng.randint(3, size=30)
    # create a categorical feature using some integer
    X[:, 3] = rng.randint(3, size=30)
    y = np.array([0] * 10 + [1] * 20)
    X = sparse.csr_matrix(X) if format == "csr" else sparse.csc_matrix(X)
    return X, y, [0, 3]


def test_smotenc_error():
    X, y, _ = data_heterogneous_unordered()
    categorical_features = [0, 10]
    smote = SMOTENC(random_state=0, categorical_features=categorical_features)
    with pytest.raises(ValueError, match="all features must be in"):
        smote.fit_resample(X, y)


@pytest.mark.parametrize(
    "data",
    [
        data_heterogneous_ordered(),
        data_heterogneous_unordered(),
        data_heterogneous_masked(),
        data_sparse("csr"),
        data_sparse("csc"),
    ],
)
def test_smotenc(data):
    X, y, categorical_features = data
    smote = SMOTENC(random_state=0, categorical_features=categorical_features)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    assert X_resampled.dtype == X.dtype

    categorical_features = np.array(categorical_features)
    if categorical_features.dtype == bool:
        categorical_features = np.flatnonzero(categorical_features)
    for cat_idx in categorical_features:
        if sparse.issparse(X):
            assert set(X[:, cat_idx].data) == set(X_resampled[:, cat_idx].data)
            assert X[:, cat_idx].dtype == X_resampled[:, cat_idx].dtype
        else:
            assert set(X[:, cat_idx]) == set(X_resampled[:, cat_idx])
            assert X[:, cat_idx].dtype == X_resampled[:, cat_idx].dtype


# part of the common test which apply to SMOTE-NC even if it is not default
# constructible
def test_smotenc_check_target_type():
    X, _, categorical_features = data_heterogneous_unordered()
    y = np.linspace(0, 1, 30)
    smote = SMOTENC(categorical_features=categorical_features, random_state=0)
    with pytest.raises(ValueError, match="Unknown label type"):
        smote.fit_resample(X, y)
    rng = np.random.RandomState(42)
    y = rng.randint(2, size=(20, 3))
    msg = "Multilabel and multioutput targets are not supported."
    with pytest.raises(ValueError, match=msg):
        smote.fit_resample(X, y)


def test_smotenc_samplers_one_label():
    X, _, categorical_features = data_heterogneous_unordered()
    y = np.zeros(30)
    smote = SMOTENC(categorical_features=categorical_features, random_state=0)
    with pytest.raises(ValueError, match="needs to have more than 1 class"):
        smote.fit(X, y)


def test_smotenc_fit():
    X, y, categorical_features = data_heterogneous_unordered()
    smote = SMOTENC(categorical_features=categorical_features, random_state=0)
    smote.fit_resample(X, y)
    assert hasattr(
        smote, "sampling_strategy_"
    ), "No fitted attribute sampling_strategy_"


def test_smotenc_fit_resample():
    X, y, categorical_features = data_heterogneous_unordered()
    target_stats = Counter(y)
    smote = SMOTENC(categorical_features=categorical_features, random_state=0)
    _, y_res = smote.fit_resample(X, y)
    _ = Counter(y_res)
    n_samples = max(target_stats.values())
    assert all(value >= n_samples for value in Counter(y_res).values())


def test_smotenc_fit_resample_sampling_strategy():
    X, y, categorical_features = data_heterogneous_unordered_multiclass()
    expected_stat = Counter(y)[1]
    smote = SMOTENC(categorical_features=categorical_features, random_state=0)
    sampling_strategy = {2: 25, 0: 25}
    smote.set_params(sampling_strategy=sampling_strategy)
    X_res, y_res = smote.fit_resample(X, y)
    assert Counter(y_res)[1] == expected_stat


def test_smotenc_pandas():
    pd = pytest.importorskip("pandas")
    # Check that the samplers handle pandas dataframe and pandas series
    X, y, categorical_features = data_heterogneous_unordered_multiclass()
    X_pd = pd.DataFrame(X)
    smote = SMOTENC(categorical_features=categorical_features, random_state=0)
    X_res_pd, y_res_pd = smote.fit_resample(X_pd, y)
    X_res, y_res = smote.fit_resample(X, y)
    assert_array_equal(X_res_pd.to_numpy(), X_res)
    assert_allclose(y_res_pd, y_res)


def test_smotenc_preserve_dtype():
    X, y = make_classification(
        n_samples=50,
        n_classes=3,
        n_informative=4,
        weights=[0.2, 0.3, 0.5],
        random_state=0,
    )
    # Cast X and y to not default dtype
    X = X.astype(np.float32)
    y = y.astype(np.int32)
    smote = SMOTENC(categorical_features=[1], random_state=0)
    X_res, y_res = smote.fit_resample(X, y)
    assert X.dtype == X_res.dtype, "X dtype is not preserved"
    assert y.dtype == y_res.dtype, "y dtype is not preserved"


@pytest.mark.parametrize("categorical_features", [[True, True, True], [0, 1, 2]])
def test_smotenc_raising_error_all_categorical(categorical_features):
    X, y = make_classification(
        n_features=3,
        n_informative=1,
        n_redundant=1,
        n_repeated=0,
        n_clusters_per_class=1,
    )
    smote = SMOTENC(categorical_features=categorical_features)
    err_msg = "SMOTE-NC is not designed to work only with categorical features"
    with pytest.raises(ValueError, match=err_msg):
        smote.fit_resample(X, y)


def test_smote_nc_with_null_median_std():
    # Non-regression test for #662
    # https://github.com/scikit-learn-contrib/imbalanced-learn/issues/662
    data = np.array(
        [
            [1, 2, 1, "A"],
            [2, 1, 2, "A"],
            [1, 2, 3, "B"],
            [1, 2, 4, "C"],
            [1, 2, 5, "C"],
        ],
        dtype="object",
    )
    labels = np.array(
        ["class_1", "class_1", "class_1", "class_2", "class_2"], dtype=object
    )
    smote = SMOTENC(categorical_features=[3], k_neighbors=1, random_state=0)
    X_res, y_res = smote.fit_resample(data, labels)
    # check that the categorical feature is not random but correspond to the
    # categories seen in the minority class samples
    assert X_res[-1, -1] == "C"


def test_smotenc_categorical_encoder():
    """Check that we can pass our own categorical encoder."""

    # TODO: only use `sparse_output` when sklearn >= 1.2
    param = "sparse" if sklearn_version < parse_version("1.2") else "sparse_output"

    X, y, categorical_features = data_heterogneous_unordered()
    smote = SMOTENC(categorical_features=categorical_features, random_state=0)
    smote.fit_resample(X, y)

    assert getattr(smote.categorical_encoder_, param) is True

    encoder = OneHotEncoder()
    encoder.set_params(**{param: False})
    smote.set_params(categorical_encoder=encoder).fit_resample(X, y)
    assert smote.categorical_encoder is encoder
    assert smote.categorical_encoder_ is not encoder
    assert getattr(smote.categorical_encoder_, param) is False


# TODO(0.13): remove this test
def test_smotenc_deprecation_ohe_():
    """Check that we raise a deprecation warning when using `ohe_`."""
    X, y, categorical_features = data_heterogneous_unordered()
    smote = SMOTENC(categorical_features=categorical_features, random_state=0)
    smote.fit_resample(X, y)

    with pytest.warns(FutureWarning, match="'ohe_' attribute has been deprecated"):
        smote.ohe_


def test_smotenc_param_validation():
    """Check that we validate the parameters correctly since this estimator requires
    a specific parameter.
    """
    categorical_features = [0]
    smote = SMOTENC(categorical_features=categorical_features, random_state=0)
    name = smote.__class__.__name__
    _set_checking_parameters(smote)
    check_param_validation(name, smote)


def test_smotenc_bool_categorical():
    """Check that we don't try to early convert the full input data to numeric when
    handling a pandas dataframe.

    Non-regression test for:
    https://github.com/scikit-learn-contrib/imbalanced-learn/issues/974
    """
    pd = pytest.importorskip("pandas")

    X = pd.DataFrame(
        {
            "c": pd.Categorical([x for x in "abbacaba" * 3]),
            "f": [0.3, 0.5, 0.1, 0.2] * 6,
            "b": [False, False, True] * 8,
        }
    )
    y = pd.DataFrame({"out": [1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0] * 2})
    smote = SMOTENC(categorical_features=[0])

    X_res, y_res = smote.fit_resample(X, y)
    pd.testing.assert_series_equal(X_res.dtypes, X.dtypes)
    assert len(X_res) == len(y_res)

    smote.set_params(categorical_features=[0, 2])
    X_res, y_res = smote.fit_resample(X, y)
    pd.testing.assert_series_equal(X_res.dtypes, X.dtypes)
    assert len(X_res) == len(y_res)

    X = X.astype({"b": "category"})
    X_res, y_res = smote.fit_resample(X, y)
    pd.testing.assert_series_equal(X_res.dtypes, X.dtypes)
    assert len(X_res) == len(y_res)


def test_smotenc_categorical_features_str():
    """Check that we support array-like of strings for `categorical_features` using
    pandas dataframe.
    """
    pd = pytest.importorskip("pandas")

    X = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "B": ["a", "b"] * 5,
            "C": ["a", "b", "c"] * 3 + ["a"],
        }
    )
    X = pd.concat([X] * 10, ignore_index=True)
    y = np.array([0] * 70 + [1] * 30)
    smote = SMOTENC(categorical_features=["B", "C"], random_state=0)
    X_res, y_res = smote.fit_resample(X, y)
    assert X_res["B"].isin(["a", "b"]).all()
    assert X_res["C"].isin(["a", "b", "c"]).all()
    counter = Counter(y_res)
    assert counter[0] == counter[1] == 70
    assert_array_equal(smote.categorical_features_, [1, 2])
    assert_array_equal(smote.continuous_features_, [0])


def test_smotenc_categorical_features_auto():
    """Check that we can automatically detect categorical features based on pandas
    dataframe.
    """
    pd = pytest.importorskip("pandas")

    X = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "B": ["a", "b"] * 5,
            "C": ["a", "b", "c"] * 3 + ["a"],
        }
    )
    X = pd.concat([X] * 10, ignore_index=True)
    X["B"] = X["B"].astype("category")
    X["C"] = X["C"].astype("category")
    y = np.array([0] * 70 + [1] * 30)
    smote = SMOTENC(categorical_features="auto", random_state=0)
    X_res, y_res = smote.fit_resample(X, y)
    assert X_res["B"].isin(["a", "b"]).all()
    assert X_res["C"].isin(["a", "b", "c"]).all()
    counter = Counter(y_res)
    assert counter[0] == counter[1] == 70
    assert_array_equal(smote.categorical_features_, [1, 2])
    assert_array_equal(smote.continuous_features_, [0])


def test_smote_nc_categorical_features_auto_error():
    """Check that we raise a proper error when we cannot use the `'auto'` mode."""
    pd = pytest.importorskip("pandas")

    X = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "B": ["a", "b"] * 5,
            "C": ["a", "b", "c"] * 3 + ["a"],
        }
    )
    y = np.array([0] * 70 + [1] * 30)
    smote = SMOTENC(categorical_features="auto", random_state=0)

    with pytest.raises(ValueError, match="the input data should be a pandas.DataFrame"):
        smote.fit_resample(X.to_numpy(), y)

    err_msg = "SMOTE-NC is not designed to work only with numerical features"
    with pytest.raises(ValueError, match=err_msg):
        smote.fit_resample(X, y)
