"""
Testing for the base module (sklearn.ensemble.base).
"""

# Authors: Gilles Louppe
# License: BSD 3 clause

from collections import OrderedDict

import numpy as np
import pytest

from sklearn import ensemble
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble._base import _set_random_states
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, Perceptron, Ridge
from sklearn.pipeline import Pipeline


def test_base():
    # Check BaseEnsemble methods.
    ensemble = BaggingClassifier(
        estimator=Perceptron(random_state=None), n_estimators=3
    )

    iris = load_iris()
    ensemble.fit(iris.data, iris.target)
    ensemble.estimators_ = []  # empty the list and create estimators manually

    ensemble._make_estimator()
    random_state = np.random.RandomState(3)
    ensemble._make_estimator(random_state=random_state)
    ensemble._make_estimator(random_state=random_state)
    ensemble._make_estimator(append=False)

    assert 3 == len(ensemble)
    assert 3 == len(ensemble.estimators_)

    assert isinstance(ensemble[0], Perceptron)
    assert ensemble[0].random_state is None
    assert isinstance(ensemble[1].random_state, int)
    assert isinstance(ensemble[2].random_state, int)
    assert ensemble[1].random_state != ensemble[2].random_state

    np_int_ensemble = BaggingClassifier(
        estimator=Perceptron(), n_estimators=np.int32(3)
    )
    np_int_ensemble.fit(iris.data, iris.target)


def test_set_random_states():
    # Linear Discriminant Analysis doesn't have random state: smoke test
    _set_random_states(LinearDiscriminantAnalysis(), random_state=17)

    clf1 = Perceptron(random_state=None)
    assert clf1.random_state is None
    # check random_state is None still sets
    _set_random_states(clf1, None)
    assert isinstance(clf1.random_state, int)

    # check random_state fixes results in consistent initialisation
    _set_random_states(clf1, 3)
    assert isinstance(clf1.random_state, int)
    clf2 = Perceptron(random_state=None)
    _set_random_states(clf2, 3)
    assert clf1.random_state == clf2.random_state

    # nested random_state

    def make_steps():
        return [
            ("sel", SelectFromModel(Perceptron(random_state=None))),
            ("clf", Perceptron(random_state=None)),
        ]

    est1 = Pipeline(make_steps())
    _set_random_states(est1, 3)
    assert isinstance(est1.steps[0][1].estimator.random_state, int)
    assert isinstance(est1.steps[1][1].random_state, int)
    assert (
        est1.get_params()["sel__estimator__random_state"]
        != est1.get_params()["clf__random_state"]
    )

    # ensure multiple random_state parameters are invariant to get_params()
    # iteration order

    class AlphaParamPipeline(Pipeline):
        def get_params(self, *args, **kwargs):
            params = Pipeline.get_params(self, *args, **kwargs).items()
            return OrderedDict(sorted(params))

    class RevParamPipeline(Pipeline):
        def get_params(self, *args, **kwargs):
            params = Pipeline.get_params(self, *args, **kwargs).items()
            return OrderedDict(sorted(params, reverse=True))

    for cls in [AlphaParamPipeline, RevParamPipeline]:
        est2 = cls(make_steps())
        _set_random_states(est2, 3)
        assert (
            est1.get_params()["sel__estimator__random_state"]
            == est2.get_params()["sel__estimator__random_state"]
        )
        assert (
            est1.get_params()["clf__random_state"]
            == est2.get_params()["clf__random_state"]
        )


# TODO(1.4): remove
def test_validate_estimator_value_error():
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, 0])
    model = BaggingClassifier(estimator=Perceptron(), base_estimator=Perceptron())
    err_msg = "Both `estimator` and `base_estimator` were set. Only set `estimator`."
    with pytest.raises(ValueError, match=err_msg):
        model.fit(X, y)


# TODO(1.4): remove
@pytest.mark.parametrize(
    "model",
    [
        ensemble.GradientBoostingClassifier(),
        ensemble.GradientBoostingRegressor(),
        ensemble.HistGradientBoostingClassifier(),
        ensemble.HistGradientBoostingRegressor(),
        ensemble.VotingClassifier(
            [("a", LogisticRegression()), ("b", LogisticRegression())]
        ),
        ensemble.VotingRegressor([("a", Ridge()), ("b", Ridge())]),
    ],
)
def test_estimator_attribute_error(model):
    X = [[1], [2]]
    y = [0, 1]
    model.fit(X, y)

    assert not hasattr(model, "estimator_")
