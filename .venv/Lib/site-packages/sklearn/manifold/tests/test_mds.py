from unittest.mock import Mock

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal

from sklearn.manifold import _mds as mds
from sklearn.metrics import euclidean_distances


def test_smacof():
    # test metric smacof using the data of "Modern Multidimensional Scaling",
    # Borg & Groenen, p 154
    sim = np.array([[0, 5, 3, 4], [5, 0, 2, 2], [3, 2, 0, 1], [4, 2, 1, 0]])
    Z = np.array([[-0.266, -0.539], [0.451, 0.252], [0.016, -0.238], [-0.200, 0.524]])
    X, _ = mds.smacof(sim, init=Z, n_components=2, max_iter=1, n_init=1)
    X_true = np.array(
        [[-1.415, -2.471], [1.633, 1.107], [0.249, -0.067], [-0.468, 1.431]]
    )
    assert_array_almost_equal(X, X_true, decimal=3)


def test_smacof_error():
    # Not symmetric similarity matrix:
    sim = np.array([[0, 5, 9, 4], [5, 0, 2, 2], [3, 2, 0, 1], [4, 2, 1, 0]])

    with pytest.raises(ValueError):
        mds.smacof(sim)

    # Not squared similarity matrix:
    sim = np.array([[0, 5, 9, 4], [5, 0, 2, 2], [4, 2, 1, 0]])

    with pytest.raises(ValueError):
        mds.smacof(sim)

    # init not None and not correct format:
    sim = np.array([[0, 5, 3, 4], [5, 0, 2, 2], [3, 2, 0, 1], [4, 2, 1, 0]])

    Z = np.array([[-0.266, -0.539], [0.016, -0.238], [-0.200, 0.524]])
    with pytest.raises(ValueError):
        mds.smacof(sim, init=Z, n_init=1)


def test_MDS():
    sim = np.array([[0, 5, 3, 4], [5, 0, 2, 2], [3, 2, 0, 1], [4, 2, 1, 0]])
    mds_clf = mds.MDS(metric=False, n_jobs=3, dissimilarity="precomputed")
    mds_clf.fit(sim)


@pytest.mark.parametrize("k", [0.5, 1.5, 2])
def test_normed_stress(k):
    """Test that non-metric MDS normalized stress is scale-invariant."""
    sim = np.array([[0, 5, 3, 4], [5, 0, 2, 2], [3, 2, 0, 1], [4, 2, 1, 0]])

    X1, stress1 = mds.smacof(sim, metric=False, max_iter=5, random_state=0)
    X2, stress2 = mds.smacof(k * sim, metric=False, max_iter=5, random_state=0)

    assert_allclose(stress1, stress2, rtol=1e-5)
    assert_allclose(X1, X2, rtol=1e-5)


def test_normalize_metric_warning():
    """
    Test that a UserWarning is emitted when using normalized stress with
    metric-MDS.
    """
    msg = "Normalized stress is not supported"
    sim = np.array([[0, 5, 3, 4], [5, 0, 2, 2], [3, 2, 0, 1], [4, 2, 1, 0]])
    with pytest.raises(ValueError, match=msg):
        mds.smacof(sim, metric=True, normalized_stress=True)


@pytest.mark.parametrize("metric", [True, False])
def test_normalized_stress_auto(metric, monkeypatch):
    rng = np.random.RandomState(0)
    X = rng.randn(4, 3)
    dist = euclidean_distances(X)

    mock = Mock(side_effect=mds._smacof_single)
    monkeypatch.setattr("sklearn.manifold._mds._smacof_single", mock)

    est = mds.MDS(metric=metric, normalized_stress="auto", random_state=rng)
    est.fit_transform(X)
    assert mock.call_args[1]["normalized_stress"] != metric

    mds.smacof(dist, metric=metric, normalized_stress="auto", random_state=rng)
    assert mock.call_args[1]["normalized_stress"] != metric
