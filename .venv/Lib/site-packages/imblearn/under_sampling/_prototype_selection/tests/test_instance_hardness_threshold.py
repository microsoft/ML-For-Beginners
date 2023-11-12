"""Test the module ."""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB as NB
from sklearn.utils._testing import assert_array_equal

from imblearn.under_sampling import InstanceHardnessThreshold

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
ESTIMATOR = GradientBoostingClassifier(random_state=RND_SEED)


def test_iht_init():
    sampling_strategy = "auto"
    iht = InstanceHardnessThreshold(
        estimator=ESTIMATOR,
        sampling_strategy=sampling_strategy,
        random_state=RND_SEED,
    )

    assert iht.sampling_strategy == sampling_strategy
    assert iht.random_state == RND_SEED


def test_iht_fit_resample():
    iht = InstanceHardnessThreshold(estimator=ESTIMATOR, random_state=RND_SEED)
    X_resampled, y_resampled = iht.fit_resample(X, Y)
    assert X_resampled.shape == (12, 2)
    assert y_resampled.shape == (12,)


def test_iht_fit_resample_half():
    sampling_strategy = {0: 3, 1: 3}
    iht = InstanceHardnessThreshold(
        estimator=NB(),
        sampling_strategy=sampling_strategy,
        random_state=RND_SEED,
    )
    X_resampled, y_resampled = iht.fit_resample(X, Y)
    assert X_resampled.shape == (6, 2)
    assert y_resampled.shape == (6,)


def test_iht_fit_resample_class_obj():
    est = GradientBoostingClassifier(random_state=RND_SEED)
    iht = InstanceHardnessThreshold(estimator=est, random_state=RND_SEED)
    X_resampled, y_resampled = iht.fit_resample(X, Y)
    assert X_resampled.shape == (12, 2)
    assert y_resampled.shape == (12,)


def test_iht_reproducibility():
    from sklearn.datasets import load_digits

    X_digits, y_digits = load_digits(return_X_y=True)
    idx_sampled = []
    for seed in range(5):
        est = RandomForestClassifier(n_estimators=10, random_state=seed)
        iht = InstanceHardnessThreshold(estimator=est, random_state=RND_SEED)
        iht.fit_resample(X_digits, y_digits)
        idx_sampled.append(iht.sample_indices_.copy())
    for idx_1, idx_2 in zip(idx_sampled, idx_sampled[1:]):
        assert_array_equal(idx_1, idx_2)


def test_iht_fit_resample_default_estimator():
    iht = InstanceHardnessThreshold(estimator=None, random_state=RND_SEED)
    X_resampled, y_resampled = iht.fit_resample(X, Y)
    assert isinstance(iht.estimator_, RandomForestClassifier)
    assert X_resampled.shape == (12, 2)
    assert y_resampled.shape == (12,)
