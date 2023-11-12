"""Test for the testing module"""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

import numpy as np
import pytest
from sklearn.neighbors._base import KNeighborsMixin

from imblearn.base import SamplerMixin
from imblearn.utils.testing import _CustomNearestNeighbors, all_estimators


def test_all_estimators():
    # check if the filtering is working with a list or a single string
    type_filter = "sampler"
    all_estimators(type_filter=type_filter)
    type_filter = ["sampler"]
    estimators = all_estimators(type_filter=type_filter)
    for estimator in estimators:
        # check that all estimators are sampler
        assert issubclass(estimator[1], SamplerMixin)

    # check that an error is raised when the type is unknown
    type_filter = "rnd"
    with pytest.raises(ValueError, match="Parameter type_filter must be 'sampler'"):
        all_estimators(type_filter=type_filter)


def test_custom_nearest_neighbors():
    """Check that our custom nearest neighbors can be used for our internal
    duck-typing."""

    neareat_neighbors = _CustomNearestNeighbors(n_neighbors=3)

    assert not isinstance(neareat_neighbors, KNeighborsMixin)
    assert hasattr(neareat_neighbors, "kneighbors")
    assert hasattr(neareat_neighbors, "kneighbors_graph")

    rng = np.random.RandomState(42)
    X = rng.randn(150, 3)
    y = rng.randint(0, 2, 150)
    neareat_neighbors.fit(X, y)

    distances, indices = neareat_neighbors.kneighbors(X)
    assert distances.shape == (150, 3)
    assert indices.shape == (150, 3)
    np.testing.assert_allclose(distances[:, 0], 0.0)
    np.testing.assert_allclose(indices[:, 0], np.arange(150))
