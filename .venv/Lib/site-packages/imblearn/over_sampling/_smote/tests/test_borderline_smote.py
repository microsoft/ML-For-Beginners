from collections import Counter

import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.utils._testing import assert_allclose, assert_array_equal

from imblearn.over_sampling import BorderlineSMOTE


@pytest.mark.parametrize("kind", ["borderline-1", "borderline-2"])
def test_borderline_smote_no_in_danger_samples(kind):
    """Check that the algorithm behave properly even on a dataset without any sample
    in danger.
    """
    X, y = make_classification(
        n_samples=500,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_clusters_per_class=1,
        n_classes=3,
        weights=[0.1, 0.2, 0.7],
        class_sep=1.5,
        random_state=1,
    )
    smote = BorderlineSMOTE(kind=kind, m_neighbors=3, k_neighbors=5, random_state=0)
    X_res, y_res = smote.fit_resample(X, y)

    assert_allclose(X, X_res)
    assert_allclose(y, y_res)
    assert not smote.in_danger_indices


def test_borderline_smote_kind():
    """Check the behaviour of the `kind` parameter.

    In short, "borderline-2" generates sample closer to the boundary decision than
    "borderline-1". We generate an example where a logistic regression will perform
    worse on "borderline-2" than on "borderline-1".
    """
    X, y = make_classification(
        n_samples=500,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_clusters_per_class=1,
        n_classes=3,
        weights=[0.1, 0.2, 0.7],
        class_sep=1.0,
        random_state=1,
    )
    smote = BorderlineSMOTE(
        kind="borderline-1", m_neighbors=9, k_neighbors=5, random_state=0
    )
    X_res_borderline_1, y_res_borderline_1 = smote.fit_resample(X, y)
    smote.set_params(kind="borderline-2")
    X_res_borderline_2, y_res_borderline_2 = smote.fit_resample(X, y)

    score_borderline_1 = (
        LogisticRegression()
        .fit(X_res_borderline_1, y_res_borderline_1)
        .score(X_res_borderline_1, y_res_borderline_1)
    )
    score_borderline_2 = (
        LogisticRegression()
        .fit(X_res_borderline_2, y_res_borderline_2)
        .score(X_res_borderline_2, y_res_borderline_2)
    )
    assert score_borderline_1 > score_borderline_2


def test_borderline_smote_in_danger():
    X, y = make_classification(
        n_samples=500,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_clusters_per_class=1,
        n_classes=3,
        weights=[0.1, 0.2, 0.7],
        class_sep=0.8,
        random_state=1,
    )
    smote = BorderlineSMOTE(
        kind="borderline-1",
        m_neighbors=9,
        k_neighbors=5,
        random_state=0,
    )
    _, y_res_1 = smote.fit_resample(X, y)
    in_danger_indices_borderline_1 = smote.in_danger_indices
    smote.set_params(kind="borderline-2")
    _, y_res_2 = smote.fit_resample(X, y)
    in_danger_indices_borderline_2 = smote.in_danger_indices

    for key1, key2 in zip(
        in_danger_indices_borderline_1, in_danger_indices_borderline_2
    ):
        assert_array_equal(
            in_danger_indices_borderline_1[key1], in_danger_indices_borderline_2[key2]
        )
    assert len(in_danger_indices_borderline_1) == len(in_danger_indices_borderline_2)
    counter = Counter(y_res_1)
    assert counter[0] == counter[1] == counter[2]
    counter = Counter(y_res_2)
    assert counter[0] == counter[1] == counter[2]
