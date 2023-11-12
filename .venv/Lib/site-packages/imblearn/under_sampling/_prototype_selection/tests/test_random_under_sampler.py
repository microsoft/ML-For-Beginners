"""Test the module random under sampler."""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

from collections import Counter
from datetime import datetime

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.utils._testing import assert_array_equal

from imblearn.under_sampling import RandomUnderSampler

RND_SEED = 0
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


@pytest.mark.parametrize("as_frame", [True, False], ids=["dataframe", "array"])
def test_rus_fit_resample(as_frame):
    if as_frame:
        pd = pytest.importorskip("pandas")
        X_ = pd.DataFrame(X)
    else:
        X_ = X
    rus = RandomUnderSampler(random_state=RND_SEED, replacement=True)
    X_resampled, y_resampled = rus.fit_resample(X_, Y)

    X_gt = np.array(
        [
            [0.92923648, 0.76103773],
            [0.47104475, 0.44386323],
            [0.13347175, 0.12167502],
            [0.09125309, -0.85409574],
            [0.12372842, 0.6536186],
            [0.04352327, -0.20515826],
        ]
    )
    y_gt = np.array([0, 0, 0, 1, 1, 1])

    if as_frame:
        assert hasattr(X_resampled, "loc")
        # FIXME: we should use to_numpy with pandas >= 0.25
        X_resampled = X_resampled.values

    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_rus_fit_resample_half():
    sampling_strategy = {0: 3, 1: 6}
    rus = RandomUnderSampler(
        sampling_strategy=sampling_strategy,
        random_state=RND_SEED,
        replacement=True,
    )
    X_resampled, y_resampled = rus.fit_resample(X, Y)

    X_gt = np.array(
        [
            [0.92923648, 0.76103773],
            [0.47104475, 0.44386323],
            [0.92923648, 0.76103773],
            [0.15490546, 0.3130677],
            [0.15490546, 0.3130677],
            [0.15490546, 0.3130677],
            [0.20792588, 1.49407907],
            [0.15490546, 0.3130677],
            [0.12372842, 0.6536186],
        ]
    )
    y_gt = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_multiclass_fit_resample():
    y = Y.copy()
    y[5] = 2
    y[6] = 2
    rus = RandomUnderSampler(random_state=RND_SEED)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    count_y_res = Counter(y_resampled)
    assert count_y_res[0] == 2
    assert count_y_res[1] == 2
    assert count_y_res[2] == 2


def test_random_under_sampling_heterogeneous_data():
    X_hetero = np.array(
        [["xxx", 1, 1.0], ["yyy", 2, 2.0], ["zzz", 3, 3.0]], dtype=object
    )
    y = np.array([0, 0, 1])
    rus = RandomUnderSampler(random_state=RND_SEED)
    X_res, y_res = rus.fit_resample(X_hetero, y)

    assert X_res.shape[0] == 2
    assert y_res.shape[0] == 2
    assert X_res.dtype == object


def test_random_under_sampling_nan_inf():
    # check that we can undersample even with missing or infinite data
    # regression tests for #605
    rng = np.random.RandomState(42)
    n_not_finite = X.shape[0] // 3
    row_indices = rng.choice(np.arange(X.shape[0]), size=n_not_finite)
    col_indices = rng.randint(0, X.shape[1], size=n_not_finite)
    not_finite_values = rng.choice([np.nan, np.inf], size=n_not_finite)

    X_ = X.copy()
    X_[row_indices, col_indices] = not_finite_values

    rus = RandomUnderSampler(random_state=0)
    X_res, y_res = rus.fit_resample(X_, Y)

    assert y_res.shape == (6,)
    assert X_res.shape == (6, 2)
    assert np.any(~np.isfinite(X_res))


@pytest.mark.parametrize(
    "sampling_strategy", ["auto", "majority", "not minority", "not majority", "all"]
)
def test_random_under_sampler_strings(sampling_strategy):
    """Check that we support all supposed strings as `sampling_strategy` in
    a sampler inheriting from `BaseUnderSampler`."""

    X, y = make_classification(
        n_samples=100,
        n_clusters_per_class=1,
        n_classes=3,
        weights=[0.1, 0.3, 0.6],
        random_state=0,
    )
    RandomUnderSampler(sampling_strategy=sampling_strategy).fit_resample(X, y)


def test_random_under_sampling_datetime():
    """Check that we don't convert input data and only sample from it."""
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame({"label": [0, 0, 0, 1], "td": [datetime.now()] * 4})
    y = X["label"]
    rus = RandomUnderSampler(random_state=0)
    X_res, y_res = rus.fit_resample(X, y)

    pd.testing.assert_series_equal(X_res.dtypes, X.dtypes)
    pd.testing.assert_index_equal(X_res.index, y_res.index)
    assert_array_equal(y_res.to_numpy(), np.array([0, 1]))
