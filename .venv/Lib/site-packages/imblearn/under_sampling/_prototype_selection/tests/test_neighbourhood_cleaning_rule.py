"""Test the module neighbourhood cleaning rule."""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

import numpy as np
from sklearn.utils._testing import assert_array_equal

from imblearn.under_sampling import NeighbourhoodCleaningRule

X = np.array(
    [
        [1.57737838, 0.1997882],
        [0.8960075, 0.46130762],
        [0.34096173, 0.50947647],
        [-0.91735824, 0.93110278],
        [-0.14619583, 1.33009918],
        [-0.20413357, 0.64628718],
        [0.85713638, 0.91069295],
        [0.35967591, 2.61186964],
        [0.43142011, 0.52323596],
        [0.90701028, -0.57636928],
        [-1.20809175, -1.49917302],
        [-0.60497017, -0.66630228],
        [1.39272351, -0.51631728],
        [-1.55581933, 1.09609604],
        [1.55157493, -1.6981518],
    ]
)
Y = np.array([1, 2, 1, 1, 2, 1, 2, 2, 1, 2, 0, 0, 2, 1, 2])


def test_ncr_fit_resample():
    ncr = NeighbourhoodCleaningRule()
    X_resampled, y_resampled = ncr.fit_resample(X, Y)

    X_gt = np.array(
        [
            [0.34096173, 0.50947647],
            [-0.91735824, 0.93110278],
            [-0.20413357, 0.64628718],
            [0.35967591, 2.61186964],
            [0.90701028, -0.57636928],
            [-1.20809175, -1.49917302],
            [-0.60497017, -0.66630228],
            [1.39272351, -0.51631728],
            [-1.55581933, 1.09609604],
            [1.55157493, -1.6981518],
        ]
    )
    y_gt = np.array([1, 1, 1, 2, 2, 0, 0, 2, 1, 2])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_ncr_fit_resample_mode():
    ncr = NeighbourhoodCleaningRule(kind_sel="mode")
    X_resampled, y_resampled = ncr.fit_resample(X, Y)

    X_gt = np.array(
        [
            [0.34096173, 0.50947647],
            [-0.91735824, 0.93110278],
            [-0.20413357, 0.64628718],
            [0.35967591, 2.61186964],
            [0.90701028, -0.57636928],
            [-1.20809175, -1.49917302],
            [-0.60497017, -0.66630228],
            [1.39272351, -0.51631728],
            [-1.55581933, 1.09609604],
            [1.55157493, -1.6981518],
        ]
    )
    y_gt = np.array([1, 1, 1, 2, 2, 0, 0, 2, 1, 2])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)
