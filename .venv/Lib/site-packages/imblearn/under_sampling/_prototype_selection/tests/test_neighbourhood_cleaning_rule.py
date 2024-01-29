"""Test the module neighbourhood cleaning rule."""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

from collections import Counter

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.utils._testing import assert_array_equal

from imblearn.under_sampling import EditedNearestNeighbours, NeighbourhoodCleaningRule


@pytest.fixture(scope="module")
def data():
    return make_classification(
        n_samples=200,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_clusters_per_class=1,
        n_classes=3,
        weights=[0.1, 0.3, 0.6],
        random_state=0,
    )


def test_ncr_threshold_cleaning(data):
    """Test the effect of the `threshold_cleaning` parameter."""
    X, y = data
    # with a large `threshold_cleaning`, the algorithm is equivalent to ENN
    enn = EditedNearestNeighbours()
    ncr = NeighbourhoodCleaningRule(
        edited_nearest_neighbours=enn, n_neighbors=10, threshold_cleaning=10
    )

    enn.fit_resample(X, y)
    ncr.fit_resample(X, y)

    assert_array_equal(np.sort(enn.sample_indices_), np.sort(ncr.sample_indices_))
    assert ncr.classes_to_clean_ == []

    # set a threshold that we should consider only the class #2
    counter = Counter(y)
    threshold = counter[1] / counter[0]
    ncr.set_params(threshold_cleaning=threshold)
    ncr.fit_resample(X, y)

    assert set(ncr.classes_to_clean_) == {2}

    # making the threshold slightly smaller to take into account class #1
    ncr.set_params(threshold_cleaning=threshold - np.finfo(np.float32).eps)
    ncr.fit_resample(X, y)

    assert set(ncr.classes_to_clean_) == {1, 2}


def test_ncr_n_neighbors(data):
    """Check the effect of the NN on the cleaning of the second phase."""
    X, y = data

    enn = EditedNearestNeighbours()
    ncr = NeighbourhoodCleaningRule(edited_nearest_neighbours=enn, n_neighbors=3)

    ncr.fit_resample(X, y)
    sample_indices_3_nn = ncr.sample_indices_

    ncr.set_params(n_neighbors=10).fit_resample(X, y)
    sample_indices_10_nn = ncr.sample_indices_

    # we should have a more aggressive cleaning with n_neighbors is larger
    assert len(sample_indices_3_nn) > len(sample_indices_10_nn)


# TODO: remove in 0.14
@pytest.mark.parametrize("kind_sel", ["all", "mode"])
def test_ncr_deprecate_kind_sel(data, kind_sel):
    X, y = data

    with pytest.warns(FutureWarning, match="`kind_sel` is deprecated"):
        NeighbourhoodCleaningRule(kind_sel=kind_sel).fit_resample(X, y)
