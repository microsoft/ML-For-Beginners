"""Test the module SMOTE ENN."""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

import numpy as np
from sklearn.utils._testing import assert_allclose, assert_array_equal

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks

RND_SEED = 0
X = np.array(
    [
        [0.20622591, 0.0582794],
        [0.68481731, 0.51935141],
        [1.34192108, -0.13367336],
        [0.62366841, -0.21312976],
        [1.61091956, -0.40283504],
        [-0.37162401, -2.19400981],
        [0.74680821, 1.63827342],
        [0.2184254, 0.24299982],
        [0.61472253, -0.82309052],
        [0.19893132, -0.47761769],
        [1.06514042, -0.0770537],
        [0.97407872, 0.44454207],
        [1.40301027, -0.83648734],
        [-1.20515198, -1.02689695],
        [-0.27410027, -0.54194484],
        [0.8381014, 0.44085498],
        [-0.23374509, 0.18370049],
        [-0.32635887, -0.29299653],
        [-0.00288378, 0.84259929],
        [1.79580611, -0.02219234],
    ]
)
Y = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0])
R_TOL = 1e-4


def test_sample_regular():
    smote = SMOTETomek(random_state=RND_SEED)
    X_resampled, y_resampled = smote.fit_resample(X, Y)
    X_gt = np.array(
        [
            [0.68481731, 0.51935141],
            [1.34192108, -0.13367336],
            [0.62366841, -0.21312976],
            [1.61091956, -0.40283504],
            [-0.37162401, -2.19400981],
            [0.74680821, 1.63827342],
            [0.61472253, -0.82309052],
            [0.19893132, -0.47761769],
            [1.40301027, -0.83648734],
            [-1.20515198, -1.02689695],
            [-0.23374509, 0.18370049],
            [-0.00288378, 0.84259929],
            [1.79580611, -0.02219234],
            [0.38307743, -0.05670439],
            [0.70319159, -0.02571667],
            [0.75052536, -0.19246518],
        ]
    )
    y_gt = np.array([1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_sample_regular_half():
    sampling_strategy = {0: 9, 1: 12}
    smote = SMOTETomek(sampling_strategy=sampling_strategy, random_state=RND_SEED)
    X_resampled, y_resampled = smote.fit_resample(X, Y)
    X_gt = np.array(
        [
            [0.68481731, 0.51935141],
            [0.62366841, -0.21312976],
            [1.61091956, -0.40283504],
            [-0.37162401, -2.19400981],
            [0.74680821, 1.63827342],
            [0.61472253, -0.82309052],
            [0.19893132, -0.47761769],
            [1.40301027, -0.83648734],
            [-1.20515198, -1.02689695],
            [-0.23374509, 0.18370049],
            [-0.00288378, 0.84259929],
            [1.79580611, -0.02219234],
            [0.45784496, -0.1053161],
        ]
    )
    y_gt = np.array([1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_validate_estimator_init():
    smote = SMOTE(random_state=RND_SEED)
    tomek = TomekLinks(sampling_strategy="all")
    smt = SMOTETomek(smote=smote, tomek=tomek, random_state=RND_SEED)
    X_resampled, y_resampled = smt.fit_resample(X, Y)
    X_gt = np.array(
        [
            [0.68481731, 0.51935141],
            [1.34192108, -0.13367336],
            [0.62366841, -0.21312976],
            [1.61091956, -0.40283504],
            [-0.37162401, -2.19400981],
            [0.74680821, 1.63827342],
            [0.61472253, -0.82309052],
            [0.19893132, -0.47761769],
            [1.40301027, -0.83648734],
            [-1.20515198, -1.02689695],
            [-0.23374509, 0.18370049],
            [-0.00288378, 0.84259929],
            [1.79580611, -0.02219234],
            [0.38307743, -0.05670439],
            [0.70319159, -0.02571667],
            [0.75052536, -0.19246518],
        ]
    )
    y_gt = np.array([1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_validate_estimator_default():
    smt = SMOTETomek(random_state=RND_SEED)
    X_resampled, y_resampled = smt.fit_resample(X, Y)
    X_gt = np.array(
        [
            [0.68481731, 0.51935141],
            [1.34192108, -0.13367336],
            [0.62366841, -0.21312976],
            [1.61091956, -0.40283504],
            [-0.37162401, -2.19400981],
            [0.74680821, 1.63827342],
            [0.61472253, -0.82309052],
            [0.19893132, -0.47761769],
            [1.40301027, -0.83648734],
            [-1.20515198, -1.02689695],
            [-0.23374509, 0.18370049],
            [-0.00288378, 0.84259929],
            [1.79580611, -0.02219234],
            [0.38307743, -0.05670439],
            [0.70319159, -0.02571667],
            [0.75052536, -0.19246518],
        ]
    )
    y_gt = np.array([1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_parallelisation():
    # Check if default job count is None
    smt = SMOTETomek(random_state=RND_SEED)
    smt._validate_estimator()
    assert smt.n_jobs is None
    assert smt.smote_.n_jobs is None
    assert smt.tomek_.n_jobs is None

    # Check if job count is set
    smt = SMOTETomek(random_state=RND_SEED, n_jobs=8)
    smt._validate_estimator()
    assert smt.n_jobs == 8
    assert smt.smote_.n_jobs == 8
    assert smt.tomek_.n_jobs == 8
