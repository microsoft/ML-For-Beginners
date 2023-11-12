"""Test the module easy ensemble."""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

import numpy as np
import pytest
import sklearn
from sklearn.datasets import load_iris, make_hastie_10_2
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import parse_version

from imblearn.datasets import make_imbalance
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler

sklearn_version = parse_version(sklearn.__version__)
iris = load_iris()

# Generate a global dataset to use
RND_SEED = 0
X = np.array(
    [
        [0.5220963, 0.11349303],
        [0.59091459, 0.40692742],
        [1.10915364, 0.05718352],
        [0.22039505, 0.26469445],
        [1.35269503, 0.44812421],
        [0.85117925, 1.0185556],
        [-2.10724436, 0.70263997],
        [-0.23627356, 0.30254174],
        [-1.23195149, 0.15427291],
        [-0.58539673, 0.62515052],
    ]
)
Y = np.array([1, 2, 2, 2, 1, 0, 1, 1, 1, 0])


@pytest.mark.parametrize("n_estimators", [10, 20])
@pytest.mark.parametrize(
    "estimator",
    [AdaBoostClassifier(n_estimators=5), AdaBoostClassifier(n_estimators=10)],
)
def test_easy_ensemble_classifier(n_estimators, estimator):
    # Check classification for various parameter settings.
    X, y = make_imbalance(
        iris.data,
        iris.target,
        sampling_strategy={0: 20, 1: 25, 2: 50},
        random_state=0,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    eec = EasyEnsembleClassifier(
        n_estimators=n_estimators,
        estimator=estimator,
        n_jobs=-1,
        random_state=RND_SEED,
    )
    eec.fit(X_train, y_train).score(X_test, y_test)
    assert len(eec.estimators_) == n_estimators
    for est in eec.estimators_:
        assert len(est.named_steps["classifier"]) == estimator.n_estimators
    # test the different prediction function
    eec.predict(X_test)
    eec.predict_proba(X_test)
    eec.predict_log_proba(X_test)
    eec.decision_function(X_test)


def test_estimator():
    # Check estimator and its default values.
    X, y = make_imbalance(
        iris.data,
        iris.target,
        sampling_strategy={0: 20, 1: 25, 2: 50},
        random_state=0,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    ensemble = EasyEnsembleClassifier(2, None, n_jobs=-1, random_state=0).fit(
        X_train, y_train
    )

    assert isinstance(ensemble.estimator_.steps[-1][1], AdaBoostClassifier)

    ensemble = EasyEnsembleClassifier(
        2, AdaBoostClassifier(), n_jobs=-1, random_state=0
    ).fit(X_train, y_train)

    assert isinstance(ensemble.estimator_.steps[-1][1], AdaBoostClassifier)


def test_bagging_with_pipeline():
    X, y = make_imbalance(
        iris.data,
        iris.target,
        sampling_strategy={0: 20, 1: 25, 2: 50},
        random_state=0,
    )
    estimator = EasyEnsembleClassifier(
        n_estimators=2,
        estimator=make_pipeline(SelectKBest(k=1), AdaBoostClassifier()),
    )
    estimator.fit(X, y).predict(X)


def test_warm_start(random_state=42):
    # Test if fitting incrementally with warm start gives a forest of the
    # right size and the same results as a normal fit.
    X, y = make_hastie_10_2(n_samples=20, random_state=1)

    clf_ws = None
    for n_estimators in [5, 10]:
        if clf_ws is None:
            clf_ws = EasyEnsembleClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                warm_start=True,
            )
        else:
            clf_ws.set_params(n_estimators=n_estimators)
        clf_ws.fit(X, y)
        assert len(clf_ws) == n_estimators

    clf_no_ws = EasyEnsembleClassifier(
        n_estimators=10, random_state=random_state, warm_start=False
    )
    clf_no_ws.fit(X, y)

    assert {pipe.steps[-1][1].random_state for pipe in clf_ws} == {
        pipe.steps[-1][1].random_state for pipe in clf_no_ws
    }


def test_warm_start_smaller_n_estimators():
    # Test if warm start'ed second fit with smaller n_estimators raises error.
    X, y = make_hastie_10_2(n_samples=20, random_state=1)
    clf = EasyEnsembleClassifier(n_estimators=5, warm_start=True)
    clf.fit(X, y)
    clf.set_params(n_estimators=4)
    with pytest.raises(ValueError):
        clf.fit(X, y)


def test_warm_start_equal_n_estimators():
    # Test that nothing happens when fitting without increasing n_estimators
    X, y = make_hastie_10_2(n_samples=20, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=43)

    clf = EasyEnsembleClassifier(n_estimators=5, warm_start=True, random_state=83)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    # modify X to nonsense values, this should not change anything
    X_train += 1.0

    warn_msg = "Warm-start fitting without increasing n_estimators"
    with pytest.warns(UserWarning, match=warn_msg):
        clf.fit(X_train, y_train)
    assert_array_equal(y_pred, clf.predict(X_test))


def test_warm_start_equivalence():
    # warm started classifier with 5+5 estimators should be equivalent to
    # one classifier with 10 estimators
    X, y = make_hastie_10_2(n_samples=20, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=43)

    clf_ws = EasyEnsembleClassifier(n_estimators=5, warm_start=True, random_state=3141)
    clf_ws.fit(X_train, y_train)
    clf_ws.set_params(n_estimators=10)
    clf_ws.fit(X_train, y_train)
    y1 = clf_ws.predict(X_test)

    clf = EasyEnsembleClassifier(n_estimators=10, warm_start=False, random_state=3141)
    clf.fit(X_train, y_train)
    y2 = clf.predict(X_test)

    assert_allclose(y1, y2)


def test_easy_ensemble_classifier_single_estimator():
    X, y = make_imbalance(
        iris.data,
        iris.target,
        sampling_strategy={0: 20, 1: 25, 2: 50},
        random_state=0,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    clf1 = EasyEnsembleClassifier(n_estimators=1, random_state=0).fit(X_train, y_train)
    clf2 = make_pipeline(
        RandomUnderSampler(random_state=0), AdaBoostClassifier(random_state=0)
    ).fit(X_train, y_train)

    assert_array_equal(clf1.predict(X_test), clf2.predict(X_test))


def test_easy_ensemble_classifier_grid_search():
    X, y = make_imbalance(
        iris.data,
        iris.target,
        sampling_strategy={0: 20, 1: 25, 2: 50},
        random_state=0,
    )

    parameters = {
        "n_estimators": [1, 2],
        "estimator__n_estimators": [3, 4],
    }
    grid_search = GridSearchCV(
        EasyEnsembleClassifier(estimator=AdaBoostClassifier()),
        parameters,
        cv=5,
    )
    grid_search.fit(X, y)


def test_easy_ensemble_classifier_n_features():
    """Check that we raise a FutureWarning when accessing `n_features_`."""
    X, y = load_iris(return_X_y=True)
    estimator = EasyEnsembleClassifier().fit(X, y)
    with pytest.warns(FutureWarning, match="`n_features_` was deprecated"):
        estimator.n_features_


@pytest.mark.skipif(
    sklearn_version < parse_version("1.2"), reason="warns for scikit-learn>=1.2"
)
def test_easy_ensemble_classifier_base_estimator():
    """Check that we raise a FutureWarning when accessing `base_estimator_`."""
    X, y = load_iris(return_X_y=True)
    estimator = EasyEnsembleClassifier().fit(X, y)
    with pytest.warns(FutureWarning, match="`base_estimator_` was deprecated"):
        estimator.base_estimator_


def test_easy_ensemble_classifier_set_both_estimator_and_base_estimator():
    """Check that we raise a ValueError when setting both `estimator` and
    `base_estimator`."""
    X, y = load_iris(return_X_y=True)
    err_msg = "Both `estimator` and `base_estimator` were set. Only set `estimator`."
    with pytest.raises(ValueError, match=err_msg):
        EasyEnsembleClassifier(
            estimator=AdaBoostClassifier(), base_estimator=AdaBoostClassifier()
        ).fit(X, y)
