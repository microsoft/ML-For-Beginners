"""Test for score"""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

import pytest
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split

from imblearn.metrics import (
    geometric_mean_score,
    make_index_balanced_accuracy,
    sensitivity_score,
    specificity_score,
)

R_TOL = 1e-2


@pytest.fixture
def data():
    X, y = make_blobs(random_state=0, centers=2)
    return train_test_split(X, y, random_state=0)


@pytest.mark.parametrize(
    "score, expected_score",
    [
        (sensitivity_score, 0.90),
        (specificity_score, 0.90),
        (geometric_mean_score, 0.90),
        (make_index_balanced_accuracy()(geometric_mean_score), 0.82),
    ],
)
@pytest.mark.parametrize("average", ["macro", "weighted", "micro"])
def test_scorer_common_average(data, score, expected_score, average):
    X_train, X_test, y_train, _ = data

    scorer = make_scorer(score, pos_label=None, average=average)
    grid = GridSearchCV(
        LogisticRegression(),
        param_grid={"C": [1, 10]},
        scoring=scorer,
        cv=3,
    )
    grid.fit(X_train, y_train).predict(X_test)

    assert grid.best_score_ >= expected_score


@pytest.mark.parametrize(
    "score, average, expected_score",
    [
        (sensitivity_score, "binary", 0.94),
        (specificity_score, "binary", 0.89),
        (geometric_mean_score, "multiclass", 0.90),
        (
            make_index_balanced_accuracy()(geometric_mean_score),
            "multiclass",
            0.82,
        ),
    ],
)
def test_scorer_default_average(data, score, average, expected_score):
    X_train, X_test, y_train, _ = data

    scorer = make_scorer(score, pos_label=1, average=average)
    grid = GridSearchCV(
        LogisticRegression(),
        param_grid={"C": [1, 10]},
        scoring=scorer,
        cv=3,
    )
    grid.fit(X_train, y_train).predict(X_test)

    assert grid.best_score_ >= expected_score
