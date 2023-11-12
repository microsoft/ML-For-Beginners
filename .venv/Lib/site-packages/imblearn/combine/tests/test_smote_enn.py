"""Test the module SMOTE ENN."""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

import numpy as np
from sklearn.utils._testing import assert_allclose, assert_array_equal

from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours

RND_SEED = 0
X = np.array(
    [
        [0.11622591, -0.0317206],
        [0.77481731, 0.60935141],
        [1.25192108, -0.22367336],
        [0.53366841, -0.30312976],
        [1.52091956, -0.49283504],
        [-0.28162401, -2.10400981],
        [0.83680821, 1.72827342],
        [0.3084254, 0.33299982],
        [0.70472253, -0.73309052],
        [0.28893132, -0.38761769],
        [1.15514042, 0.0129463],
        [0.88407872, 0.35454207],
        [1.31301027, -0.92648734],
        [-1.11515198, -0.93689695],
        [-0.18410027, -0.45194484],
        [0.9281014, 0.53085498],
        [-0.14374509, 0.27370049],
        [-0.41635887, -0.38299653],
        [0.08711622, 0.93259929],
        [1.70580611, -0.11219234],
    ]
)
Y = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0])
R_TOL = 1e-4


def test_sample_regular():
    smote = SMOTEENN(random_state=RND_SEED)
    X_resampled, y_resampled = smote.fit_resample(X, Y)

    X_gt = np.array(
        [
            [1.52091956, -0.49283504],
            [0.84976473, -0.15570176],
            [0.61319159, -0.11571667],
            [0.66052536, -0.28246518],
            [-0.28162401, -2.10400981],
            [0.83680821, 1.72827342],
            [0.08711622, 0.93259929],
        ]
    )
    y_gt = np.array([0, 0, 0, 0, 1, 1, 1])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_sample_regular_pass_smote_enn():
    smote = SMOTEENN(
        smote=SMOTE(sampling_strategy="auto", random_state=RND_SEED),
        enn=EditedNearestNeighbours(sampling_strategy="all"),
        random_state=RND_SEED,
    )
    X_resampled, y_resampled = smote.fit_resample(X, Y)

    X_gt = np.array(
        [
            [1.52091956, -0.49283504],
            [0.84976473, -0.15570176],
            [0.61319159, -0.11571667],
            [0.66052536, -0.28246518],
            [-0.28162401, -2.10400981],
            [0.83680821, 1.72827342],
            [0.08711622, 0.93259929],
        ]
    )
    y_gt = np.array([0, 0, 0, 0, 1, 1, 1])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_sample_regular_half():
    sampling_strategy = {0: 10, 1: 12}
    smote = SMOTEENN(sampling_strategy=sampling_strategy, random_state=RND_SEED)
    X_resampled, y_resampled = smote.fit_resample(X, Y)

    X_gt = np.array(
        [
            [1.52091956, -0.49283504],
            [-0.28162401, -2.10400981],
            [0.83680821, 1.72827342],
            [0.08711622, 0.93259929],
        ]
    )
    y_gt = np.array([0, 1, 1, 1])
    assert_allclose(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_validate_estimator_init():
    smote = SMOTE(random_state=RND_SEED)
    enn = EditedNearestNeighbours(sampling_strategy="all")
    smt = SMOTEENN(smote=smote, enn=enn, random_state=RND_SEED)
    X_resampled, y_resampled = smt.fit_resample(X, Y)
    X_gt = np.array(
        [
            [1.52091956, -0.49283504],
            [0.84976473, -0.15570176],
            [0.61319159, -0.11571667],
            [0.66052536, -0.28246518],
            [-0.28162401, -2.10400981],
            [0.83680821, 1.72827342],
            [0.08711622, 0.93259929],
        ]
    )
    y_gt = np.array([0, 0, 0, 0, 1, 1, 1])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_validate_estimator_default():
    smt = SMOTEENN(random_state=RND_SEED)
    X_resampled, y_resampled = smt.fit_resample(X, Y)
    X_gt = np.array(
        [
            [1.52091956, -0.49283504],
            [0.84976473, -0.15570176],
            [0.61319159, -0.11571667],
            [0.66052536, -0.28246518],
            [-0.28162401, -2.10400981],
            [0.83680821, 1.72827342],
            [0.08711622, 0.93259929],
        ]
    )
    y_gt = np.array([0, 0, 0, 0, 1, 1, 1])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_parallelisation():
    # Check if default job count is none
    smt = SMOTEENN(random_state=RND_SEED)
    smt._validate_estimator()
    assert smt.n_jobs is None
    assert smt.smote_.n_jobs is None
    assert smt.enn_.n_jobs is None

    # Check if job count is set
    smt = SMOTEENN(random_state=RND_SEED, n_jobs=8)
    smt._validate_estimator()
    assert smt.n_jobs == 8
    assert smt.smote_.n_jobs == 8
    assert smt.enn_.n_jobs == 8
