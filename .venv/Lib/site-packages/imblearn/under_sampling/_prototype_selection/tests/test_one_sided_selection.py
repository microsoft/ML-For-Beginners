"""Test the module one-sided selection."""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils._testing import assert_array_equal

from imblearn.under_sampling import OneSidedSelection

RND_SEED = 0
X = np.array(
    [
        [-0.3879569, 0.6894251],
        [-0.09322739, 1.28177189],
        [-0.77740357, 0.74097941],
        [0.91542919, -0.65453327],
        [-0.03852113, 0.40910479],
        [-0.43877303, 1.07366684],
        [-0.85795321, 0.82980738],
        [-0.18430329, 0.52328473],
        [-0.30126957, -0.66268378],
        [-0.65571327, 0.42412021],
        [-0.28305528, 0.30284991],
        [0.20246714, -0.34727125],
        [1.06446472, -1.09279772],
        [0.30543283, -0.02589502],
        [-0.00717161, 0.00318087],
    ]
)
Y = np.array([0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0])


def test_oss_init():
    oss = OneSidedSelection(random_state=RND_SEED)

    assert oss.n_seeds_S == 1
    assert oss.n_jobs is None
    assert oss.random_state == RND_SEED


def test_oss_fit_resample():
    oss = OneSidedSelection(random_state=RND_SEED)
    X_resampled, y_resampled = oss.fit_resample(X, Y)

    X_gt = np.array(
        [
            [-0.3879569, 0.6894251],
            [0.91542919, -0.65453327],
            [-0.65571327, 0.42412021],
            [1.06446472, -1.09279772],
            [0.30543283, -0.02589502],
            [-0.00717161, 0.00318087],
            [-0.09322739, 1.28177189],
            [-0.77740357, 0.74097941],
            [-0.43877303, 1.07366684],
            [-0.85795321, 0.82980738],
            [-0.30126957, -0.66268378],
            [0.20246714, -0.34727125],
        ]
    )
    y_gt = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


@pytest.mark.parametrize("n_neighbors", [1, KNeighborsClassifier(n_neighbors=1)])
def test_oss_with_object(n_neighbors):
    oss = OneSidedSelection(random_state=RND_SEED, n_neighbors=n_neighbors)
    X_resampled, y_resampled = oss.fit_resample(X, Y)

    X_gt = np.array(
        [
            [-0.3879569, 0.6894251],
            [0.91542919, -0.65453327],
            [-0.65571327, 0.42412021],
            [1.06446472, -1.09279772],
            [0.30543283, -0.02589502],
            [-0.00717161, 0.00318087],
            [-0.09322739, 1.28177189],
            [-0.77740357, 0.74097941],
            [-0.43877303, 1.07366684],
            [-0.85795321, 0.82980738],
            [-0.30126957, -0.66268378],
            [0.20246714, -0.34727125],
        ]
    )
    y_gt = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)
    knn = 1
    oss = OneSidedSelection(random_state=RND_SEED, n_neighbors=knn)
    X_resampled, y_resampled = oss.fit_resample(X, Y)
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)
