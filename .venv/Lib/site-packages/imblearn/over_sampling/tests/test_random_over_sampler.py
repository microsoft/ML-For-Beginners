"""Test the module under sampler."""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

from collections import Counter
from datetime import datetime

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.utils._testing import (
    _convert_container,
    assert_allclose,
    assert_array_equal,
)

from imblearn.over_sampling import RandomOverSampler

RND_SEED = 0


@pytest.fixture
def data():
    X = np.array(
        [
            [0.04352327, -0.20515826],
            [0.92923648, 0.76103773],
            [0.20792588, 1.49407907],
            [0.47104475, 0.44386323],
            [0.22950086, 0.33367433],
            [0.15490546, 0.3130677],
            [0.09125309, -0.85409574],
            [0.12372842, 0.6536186],
            [0.13347175, 0.12167502],
            [0.094035, -2.55298982],
        ]
    )
    Y = np.array([1, 0, 1, 0, 1, 1, 1, 1, 0, 1])
    return X, Y


def test_ros_init():
    sampling_strategy = "auto"
    ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=RND_SEED)
    assert ros.random_state == RND_SEED


@pytest.mark.parametrize(
    "params", [{"shrinkage": None}, {"shrinkage": 0}, {"shrinkage": {0: 0}}]
)
@pytest.mark.parametrize("X_type", ["array", "dataframe"])
def test_ros_fit_resample(X_type, data, params):
    X, Y = data
    X_ = _convert_container(X, X_type)
    ros = RandomOverSampler(**params, random_state=RND_SEED)
    X_resampled, y_resampled = ros.fit_resample(X_, Y)
    X_gt = np.array(
        [
            [0.04352327, -0.20515826],
            [0.92923648, 0.76103773],
            [0.20792588, 1.49407907],
            [0.47104475, 0.44386323],
            [0.22950086, 0.33367433],
            [0.15490546, 0.3130677],
            [0.09125309, -0.85409574],
            [0.12372842, 0.6536186],
            [0.13347175, 0.12167502],
            [0.094035, -2.55298982],
            [0.92923648, 0.76103773],
            [0.47104475, 0.44386323],
            [0.92923648, 0.76103773],
            [0.47104475, 0.44386323],
        ]
    )
    y_gt = np.array([1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0])

    if X_type == "dataframe":
        assert hasattr(X_resampled, "loc")
        # FIXME: we should use to_numpy with pandas >= 0.25
        X_resampled = X_resampled.values

    assert_allclose(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)

    if params["shrinkage"] is None:
        assert ros.shrinkage_ is None
    else:
        assert ros.shrinkage_ == {0: 0}


@pytest.mark.parametrize("params", [{"shrinkage": None}, {"shrinkage": 0}])
def test_ros_fit_resample_half(data, params):
    X, Y = data
    sampling_strategy = {0: 3, 1: 7}
    ros = RandomOverSampler(
        **params, sampling_strategy=sampling_strategy, random_state=RND_SEED
    )
    X_resampled, y_resampled = ros.fit_resample(X, Y)
    X_gt = np.array(
        [
            [0.04352327, -0.20515826],
            [0.92923648, 0.76103773],
            [0.20792588, 1.49407907],
            [0.47104475, 0.44386323],
            [0.22950086, 0.33367433],
            [0.15490546, 0.3130677],
            [0.09125309, -0.85409574],
            [0.12372842, 0.6536186],
            [0.13347175, 0.12167502],
            [0.094035, -2.55298982],
        ]
    )
    y_gt = np.array([1, 0, 1, 0, 1, 1, 1, 1, 0, 1])
    assert_allclose(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)

    if params["shrinkage"] is None:
        assert ros.shrinkage_ is None
    else:
        assert ros.shrinkage_ == {0: 0, 1: 0}


@pytest.mark.parametrize("params", [{"shrinkage": None}, {"shrinkage": 0}])
def test_multiclass_fit_resample(data, params):
    # check the random over-sampling with a multiclass problem
    X, Y = data
    y = Y.copy()
    y[5] = 2
    y[6] = 2
    ros = RandomOverSampler(**params, random_state=RND_SEED)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    count_y_res = Counter(y_resampled)
    assert count_y_res[0] == 5
    assert count_y_res[1] == 5
    assert count_y_res[2] == 5

    if params["shrinkage"] is None:
        assert ros.shrinkage_ is None
    else:
        assert ros.shrinkage_ == {0: 0, 2: 0}


def test_random_over_sampling_heterogeneous_data():
    # check that resampling with heterogeneous dtype is working with basic
    # resampling
    X_hetero = np.array(
        [["xxx", 1, 1.0], ["yyy", 2, 2.0], ["zzz", 3, 3.0]], dtype=object
    )
    y = np.array([0, 0, 1])
    ros = RandomOverSampler(random_state=RND_SEED)
    X_res, y_res = ros.fit_resample(X_hetero, y)

    assert X_res.shape[0] == 4
    assert y_res.shape[0] == 4
    assert X_res.dtype == object
    assert X_res[-1, 0] in X_hetero[:, 0]


def test_random_over_sampling_nan_inf(data):
    # check that we can oversample even with missing or infinite data
    # regression tests for #605
    X, Y = data
    rng = np.random.RandomState(42)
    n_not_finite = X.shape[0] // 3
    row_indices = rng.choice(np.arange(X.shape[0]), size=n_not_finite)
    col_indices = rng.randint(0, X.shape[1], size=n_not_finite)
    not_finite_values = rng.choice([np.nan, np.inf], size=n_not_finite)

    X_ = X.copy()
    X_[row_indices, col_indices] = not_finite_values

    ros = RandomOverSampler(random_state=0)
    X_res, y_res = ros.fit_resample(X_, Y)

    assert y_res.shape == (14,)
    assert X_res.shape == (14, 2)
    assert np.any(~np.isfinite(X_res))


def test_random_over_sampling_heterogeneous_data_smoothed_bootstrap():
    # check that we raise an error when heterogeneous dtype data are given
    # and a smoothed bootstrap is requested
    X_hetero = np.array(
        [["xxx", 1, 1.0], ["yyy", 2, 2.0], ["zzz", 3, 3.0]], dtype=object
    )
    y = np.array([0, 0, 1])
    ros = RandomOverSampler(shrinkage=1, random_state=RND_SEED)
    err_msg = "When shrinkage is not None, X needs to contain only numerical"
    with pytest.raises(ValueError, match=err_msg):
        ros.fit_resample(X_hetero, y)


@pytest.mark.parametrize("X_type", ["dataframe", "array", "sparse_csr", "sparse_csc"])
def test_random_over_sampler_smoothed_bootstrap(X_type, data):
    # check that smoothed bootstrap is working for numerical array
    X, y = data
    sampler = RandomOverSampler(shrinkage=1)
    X = _convert_container(X, X_type)
    X_res, y_res = sampler.fit_resample(X, y)

    assert y_res.shape == (14,)
    assert X_res.shape == (14, 2)

    if X_type == "dataframe":
        assert hasattr(X_res, "loc")


def test_random_over_sampler_equivalence_shrinkage(data):
    # check that a shrinkage factor of 0 is equivalent to not create a smoothed
    # bootstrap
    X, y = data

    ros_not_shrink = RandomOverSampler(shrinkage=0, random_state=0)
    ros_hard_bootstrap = RandomOverSampler(shrinkage=None, random_state=0)

    X_res_not_shrink, y_res_not_shrink = ros_not_shrink.fit_resample(X, y)
    X_res, y_res = ros_hard_bootstrap.fit_resample(X, y)

    assert_allclose(X_res_not_shrink, X_res)
    assert_allclose(y_res_not_shrink, y_res)

    assert y_res.shape == (14,)
    assert X_res.shape == (14, 2)
    assert y_res_not_shrink.shape == (14,)
    assert X_res_not_shrink.shape == (14, 2)


def test_random_over_sampler_shrinkage_behaviour(data):
    # check the behaviour of the shrinkage parameter
    # the covariance of the data generated with the larger shrinkage factor
    # should also be larger.
    X, y = data

    ros = RandomOverSampler(shrinkage=1, random_state=0)
    X_res_shink_1, y_res_shrink_1 = ros.fit_resample(X, y)

    ros.set_params(shrinkage=5)
    X_res_shink_5, y_res_shrink_5 = ros.fit_resample(X, y)

    disperstion_shrink_1 = np.linalg.det(np.cov(X_res_shink_1[y_res_shrink_1 == 0].T))
    disperstion_shrink_5 = np.linalg.det(np.cov(X_res_shink_5[y_res_shrink_5 == 0].T))

    assert disperstion_shrink_1 < disperstion_shrink_5


@pytest.mark.parametrize(
    "shrinkage, err_msg",
    [
        ({}, "`shrinkage` should contain a shrinkage factor for each class"),
        ({0: -1}, "The shrinkage factor needs to be >= 0"),
    ],
)
def test_random_over_sampler_shrinkage_error(data, shrinkage, err_msg):
    # check the validation of the shrinkage parameter
    X, y = data
    ros = RandomOverSampler(shrinkage=shrinkage)
    with pytest.raises(ValueError, match=err_msg):
        ros.fit_resample(X, y)


@pytest.mark.parametrize(
    "sampling_strategy", ["auto", "minority", "not minority", "not majority", "all"]
)
def test_random_over_sampler_strings(sampling_strategy):
    """Check that we support all supposed strings as `sampling_strategy` in
    a sampler inheriting from `BaseOverSampler`."""

    X, y = make_classification(
        n_samples=100,
        n_clusters_per_class=1,
        n_classes=3,
        weights=[0.1, 0.3, 0.6],
        random_state=0,
    )
    RandomOverSampler(sampling_strategy=sampling_strategy).fit_resample(X, y)


def test_random_over_sampling_datetime():
    """Check that we don't convert input data and only sample from it."""
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame({"label": [0, 0, 0, 1], "td": [datetime.now()] * 4})
    y = X["label"]
    ros = RandomOverSampler(random_state=0)
    X_res, y_res = ros.fit_resample(X, y)

    pd.testing.assert_series_equal(X_res.dtypes, X.dtypes)
    pd.testing.assert_index_equal(X_res.index, y_res.index)
    assert_array_equal(y_res.to_numpy(), np.array([0, 0, 0, 1, 1, 1]))


def test_random_over_sampler_full_nat():
    """Check that we can return timedelta columns full of NaT.

    Non-regression test for:
    https://github.com/scikit-learn-contrib/imbalanced-learn/issues/1055
    """
    pd = pytest.importorskip("pandas")

    X = pd.DataFrame(
        {
            "col_str": ["abc", "def", "xyz"],
            "col_timedelta": pd.to_timedelta([np.nan, np.nan, np.nan]),
        }
    )
    y = np.array([0, 0, 1])

    X_res, y_res = RandomOverSampler().fit_resample(X, y)
    assert X_res.shape == (4, 2)
    assert y_res.shape == (4,)

    assert X_res["col_timedelta"].dtype == "timedelta64[ns]"
