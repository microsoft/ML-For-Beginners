import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors
from sklearn.utils._testing import assert_allclose, assert_array_equal

from imblearn.over_sampling import BorderlineSMOTE


@pytest.fixture
def data():
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
    y = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0])
    return X, y


@pytest.mark.parametrize("kind", ["borderline-1", "borderline-2"])
def test_borderline_smote(kind, data):
    bsmote = BorderlineSMOTE(kind=kind, random_state=42)
    bsmote_nn = BorderlineSMOTE(
        kind=kind,
        random_state=42,
        k_neighbors=NearestNeighbors(n_neighbors=6),
        m_neighbors=NearestNeighbors(n_neighbors=11),
    )

    X_res_1, y_res_1 = bsmote.fit_resample(*data)
    X_res_2, y_res_2 = bsmote_nn.fit_resample(*data)

    assert_allclose(X_res_1, X_res_2)
    assert_array_equal(y_res_1, y_res_2)
