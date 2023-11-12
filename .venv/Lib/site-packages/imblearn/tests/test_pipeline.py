"""
Test the pipeline module.
"""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

import itertools
import re
import shutil
import time
from tempfile import mkdtemp

import numpy as np
import pytest
import sklearn
from joblib import Memory
from pytest import raises
from sklearn.base import BaseEstimator, clone
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris, make_classification
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils._testing import (
    assert_allclose,
    assert_array_almost_equal,
    assert_array_equal,
)
from sklearn.utils.fixes import parse_version

from imblearn.datasets import make_imbalance
from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.under_sampling import EditedNearestNeighbours as ENN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.utils.estimator_checks import check_param_validation

sklearn_version = parse_version(sklearn.__version__)

JUNK_FOOD_DOCS = (
    "the pizza pizza beer copyright",
    "the pizza burger beer copyright",
    "the the pizza beer beer copyright",
    "the burger beer beer copyright",
    "the coke burger coke copyright",
    "the coke burger burger",
)

R_TOL = 1e-4


class NoFit:
    """Small class to test parameter dispatching."""

    def __init__(self, a=None, b=None):
        self.a = a
        self.b = b


class NoTrans(NoFit):
    def fit(self, X, y):
        return self

    def get_params(self, deep=False):
        return {"a": self.a, "b": self.b}

    def set_params(self, **params):
        self.a = params["a"]
        return self


class NoInvTransf(NoTrans):
    def transform(self, X, y=None):
        return X


class Transf(NoInvTransf):
    def transform(self, X, y=None):
        return X

    def inverse_transform(self, X):
        return X


class TransfFitParams(Transf):
    def fit(self, X, y, **fit_params):
        self.fit_params = fit_params
        return self


class Mult(BaseEstimator):
    def __init__(self, mult=1):
        self.mult = mult

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.asarray(X) * self.mult

    def inverse_transform(self, X):
        return np.asarray(X) / self.mult

    def predict(self, X):
        return (np.asarray(X) * self.mult).sum(axis=1)

    predict_proba = predict_log_proba = decision_function = predict

    def score(self, X, y=None):
        return np.sum(X)


class FitParamT(BaseEstimator):
    """Mock classifier"""

    def __init__(self):
        self.successful = False

    def fit(self, X, y, should_succeed=False):
        self.successful = should_succeed

    def predict(self, X):
        return self.successful

    def fit_predict(self, X, y, should_succeed=False):
        self.fit(X, y, should_succeed=should_succeed)
        return self.predict(X)

    def score(self, X, y=None, sample_weight=None):
        if sample_weight is not None:
            X = X * sample_weight
        return np.sum(X)


class DummyTransf(Transf):
    """Transformer which store the column means"""

    def fit(self, X, y):
        self.means_ = np.mean(X, axis=0)
        # store timestamp to figure out whether the result of 'fit' has been
        # cached or not
        self.timestamp_ = time.time()
        return self


class DummyEstimatorParams(BaseEstimator):
    """Mock classifier that takes params on predict"""

    def fit(self, X, y):
        return self

    def predict(self, X, got_attribute=False):
        self.got_attribute = got_attribute
        return self


class DummySampler(NoTrans):
    """Samplers which returns a balanced number of samples"""

    def fit_resample(self, X, y):
        self.means_ = np.mean(X, axis=0)
        # store timestamp to figure out whether the result of 'fit' has been
        # cached or not
        self.timestamp_ = time.time()
        return X, y


class FitTransformSample(NoTrans):
    """Estimator implementing both transform and sample"""

    def fit(self, X, y, should_succeed=False):
        pass

    def fit_resample(self, X, y=None):
        return X, y

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def transform(self, X, y=None):
        return X


def test_pipeline_init_tuple():
    # Pipeline accepts steps as tuple
    X = np.array([[1, 2]])
    pipe = Pipeline((("transf", Transf()), ("clf", FitParamT())))
    pipe.fit(X, y=None)
    pipe.score(X)
    pipe.set_params(transf="passthrough")
    pipe.fit(X, y=None)
    pipe.score(X)


def test_pipeline_init():
    # Test the various init parameters of the pipeline.
    with raises(TypeError):
        Pipeline()
    # Check that we can't instantiate pipelines with objects without fit
    # method
    X, y = load_iris(return_X_y=True)
    error_regex = (
        "Last step of Pipeline should implement fit or be the string 'passthrough'"
    )
    with raises(TypeError, match=error_regex):
        model = Pipeline([("clf", NoFit())])
        model.fit(X, y)
    # Smoke test with only an estimator
    clf = NoTrans()
    pipe = Pipeline([("svc", clf)])
    expected = dict(svc__a=None, svc__b=None, svc=clf, **pipe.get_params(deep=False))
    assert pipe.get_params(deep=True) == expected

    # Check that params are set
    pipe.set_params(svc__a=0.1)
    assert clf.a == 0.1
    assert clf.b is None
    # Smoke test the repr:
    repr(pipe)

    # Test with two objects
    clf = SVC(gamma="scale")
    filter1 = SelectKBest(f_classif)
    pipe = Pipeline([("anova", filter1), ("svc", clf)])

    # Check that we can't instantiate with non-transformers on the way
    # Note that NoTrans implements fit, but not transform
    error_regex = "implement fit and transform or fit_resample"
    with raises(TypeError, match=error_regex):
        model = Pipeline([("t", NoTrans()), ("svc", clf)])
        model.fit(X, y)

    # Check that params are set
    pipe.set_params(svc__C=0.1)
    assert clf.C == 0.1
    # Smoke test the repr:
    repr(pipe)

    # Check that params are not set when naming them wrong
    with raises(ValueError):
        pipe.set_params(anova__C=0.1)

    # Test clone
    pipe2 = clone(pipe)
    assert pipe.named_steps["svc"] is not pipe2.named_steps["svc"]

    # Check that apart from estimators, the parameters are the same
    params = pipe.get_params(deep=True)
    params2 = pipe2.get_params(deep=True)

    for x in pipe.get_params(deep=False):
        params.pop(x)

    for x in pipe2.get_params(deep=False):
        params2.pop(x)

    # Remove estimators that where copied
    params.pop("svc")
    params.pop("anova")
    params2.pop("svc")
    params2.pop("anova")
    assert params == params2


def test_pipeline_methods_anova():
    # Test the various methods of the pipeline (anova).
    iris = load_iris()
    X = iris.data
    y = iris.target
    # Test with Anova + LogisticRegression
    clf = LogisticRegression(solver="lbfgs", multi_class="auto")
    filter1 = SelectKBest(f_classif, k=2)
    pipe = Pipeline([("anova", filter1), ("logistic", clf)])
    pipe.fit(X, y)
    pipe.predict(X)
    pipe.predict_proba(X)
    pipe.predict_log_proba(X)
    pipe.score(X, y)


def test_pipeline_fit_params():
    # Test that the pipeline can take fit parameters
    pipe = Pipeline([("transf", Transf()), ("clf", FitParamT())])
    pipe.fit(X=None, y=None, clf__should_succeed=True)
    # classifier should return True
    assert pipe.predict(None)
    # and transformer params should not be changed
    assert pipe.named_steps["transf"].a is None
    assert pipe.named_steps["transf"].b is None
    # invalid parameters should raise an error message
    with raises(TypeError, match="unexpected keyword argument"):
        pipe.fit(None, None, clf__bad=True)


def test_pipeline_sample_weight_supported():
    # Pipeline should pass sample_weight
    X = np.array([[1, 2]])
    pipe = Pipeline([("transf", Transf()), ("clf", FitParamT())])
    pipe.fit(X, y=None)
    assert pipe.score(X) == 3
    assert pipe.score(X, y=None) == 3
    assert pipe.score(X, y=None, sample_weight=None) == 3
    assert pipe.score(X, sample_weight=np.array([2, 3])) == 8


def test_pipeline_sample_weight_unsupported():
    # When sample_weight is None it shouldn't be passed
    X = np.array([[1, 2]])
    pipe = Pipeline([("transf", Transf()), ("clf", Mult())])
    pipe.fit(X, y=None)
    assert pipe.score(X) == 3
    assert pipe.score(X, sample_weight=None) == 3
    with raises(TypeError, match="unexpected keyword argument"):
        pipe.score(X, sample_weight=np.array([2, 3]))


def test_pipeline_raise_set_params_error():
    # Test pipeline raises set params error message for nested models.
    pipe = Pipeline([("cls", LinearRegression())])
    with raises(ValueError, match="Invalid parameter"):
        pipe.set_params(fake="nope")

    # nested model check
    with raises(ValueError, match="Invalid parameter"):
        pipe.set_params(fake__estimator="nope")


def test_pipeline_methods_pca_svm():
    # Test the various methods of the pipeline (pca + svm).
    iris = load_iris()
    X = iris.data
    y = iris.target
    # Test with PCA + SVC
    clf = SVC(gamma="scale", probability=True, random_state=0)
    pca = PCA(svd_solver="full", n_components="mle", whiten=True)
    pipe = Pipeline([("pca", pca), ("svc", clf)])
    pipe.fit(X, y)
    pipe.predict(X)
    pipe.predict_proba(X)
    pipe.predict_log_proba(X)
    pipe.score(X, y)


def test_pipeline_methods_preprocessing_svm():
    # Test the various methods of the pipeline (preprocessing + svm).
    iris = load_iris()
    X = iris.data
    y = iris.target
    n_samples = X.shape[0]
    n_classes = len(np.unique(y))
    scaler = StandardScaler()
    pca = PCA(n_components=2, svd_solver="randomized", whiten=True)
    clf = SVC(
        gamma="scale",
        probability=True,
        random_state=0,
        decision_function_shape="ovr",
    )

    for preprocessing in [scaler, pca]:
        pipe = Pipeline([("preprocess", preprocessing), ("svc", clf)])
        pipe.fit(X, y)

        # check shapes of various prediction functions
        predict = pipe.predict(X)
        assert predict.shape == (n_samples,)

        proba = pipe.predict_proba(X)
        assert proba.shape == (n_samples, n_classes)

        log_proba = pipe.predict_log_proba(X)
        assert log_proba.shape == (n_samples, n_classes)

        decision_function = pipe.decision_function(X)
        assert decision_function.shape == (n_samples, n_classes)

        pipe.score(X, y)


def test_fit_predict_on_pipeline():
    # test that the fit_predict method is implemented on a pipeline
    # test that the fit_predict on pipeline yields same results as applying
    # transform and clustering steps separately
    iris = load_iris()
    scaler = StandardScaler()
    km = KMeans(random_state=0, n_init=10)
    # As pipeline doesn't clone estimators on construction,
    # it must have its own estimators
    scaler_for_pipeline = StandardScaler()
    km_for_pipeline = KMeans(random_state=0, n_init=10)

    # first compute the transform and clustering step separately
    scaled = scaler.fit_transform(iris.data)
    separate_pred = km.fit_predict(scaled)

    # use a pipeline to do the transform and clustering in one step
    pipe = Pipeline([("scaler", scaler_for_pipeline), ("Kmeans", km_for_pipeline)])
    pipeline_pred = pipe.fit_predict(iris.data)

    assert_array_almost_equal(pipeline_pred, separate_pred)


def test_fit_predict_on_pipeline_without_fit_predict():
    # tests that a pipeline does not have fit_predict method when final
    # step of pipeline does not have fit_predict defined
    scaler = StandardScaler()
    pca = PCA(svd_solver="full")
    pipe = Pipeline([("scaler", scaler), ("pca", pca)])
    error_regex = "'PCA' object has no attribute 'fit_predict'"
    with raises(AttributeError, match=error_regex):
        getattr(pipe, "fit_predict")


def test_fit_predict_with_intermediate_fit_params():
    # tests that Pipeline passes fit_params to intermediate steps
    # when fit_predict is invoked
    pipe = Pipeline([("transf", TransfFitParams()), ("clf", FitParamT())])
    pipe.fit_predict(
        X=None, y=None, transf__should_get_this=True, clf__should_succeed=True
    )
    assert pipe.named_steps["transf"].fit_params["should_get_this"]
    assert pipe.named_steps["clf"].successful
    assert "should_succeed" not in pipe.named_steps["transf"].fit_params


def test_pipeline_transform():
    # Test whether pipeline works with a transformer at the end.
    # Also test pipeline.transform and pipeline.inverse_transform
    iris = load_iris()
    X = iris.data
    pca = PCA(n_components=2, svd_solver="full")
    pipeline = Pipeline([("pca", pca)])

    # test transform and fit_transform:
    X_trans = pipeline.fit(X).transform(X)
    X_trans2 = pipeline.fit_transform(X)
    X_trans3 = pca.fit_transform(X)
    assert_array_almost_equal(X_trans, X_trans2)
    assert_array_almost_equal(X_trans, X_trans3)

    X_back = pipeline.inverse_transform(X_trans)
    X_back2 = pca.inverse_transform(X_trans)
    assert_array_almost_equal(X_back, X_back2)


def test_pipeline_fit_transform():
    # Test whether pipeline works with a transformer missing fit_transform
    iris = load_iris()
    X = iris.data
    y = iris.target
    transf = Transf()
    pipeline = Pipeline([("mock", transf)])

    # test fit_transform:
    X_trans = pipeline.fit_transform(X, y)
    X_trans2 = transf.fit(X, y).transform(X)
    assert_array_almost_equal(X_trans, X_trans2)


def test_set_pipeline_steps():
    transf1 = Transf()
    transf2 = Transf()
    pipeline = Pipeline([("mock", transf1)])
    assert pipeline.named_steps["mock"] is transf1

    # Directly setting attr
    pipeline.steps = [("mock2", transf2)]
    assert "mock" not in pipeline.named_steps
    assert pipeline.named_steps["mock2"] is transf2
    assert [("mock2", transf2)] == pipeline.steps

    # Using set_params
    pipeline.set_params(steps=[("mock", transf1)])
    assert [("mock", transf1)] == pipeline.steps

    # Using set_params to replace single step
    pipeline.set_params(mock=transf2)
    assert [("mock", transf2)] == pipeline.steps

    # With invalid data
    pipeline.set_params(steps=[("junk", ())])
    with raises(TypeError):
        pipeline.fit([[1]], [1])
    with raises(TypeError):
        pipeline.fit_transform([[1]], [1])


@pytest.mark.parametrize("passthrough", [None, "passthrough"])
def test_pipeline_correctly_adjusts_steps(passthrough):
    X = np.array([[1]])
    y = np.array([1])
    mult2 = Mult(mult=2)
    mult3 = Mult(mult=3)
    mult5 = Mult(mult=5)
    pipeline = Pipeline(
        [("m2", mult2), ("bad", passthrough), ("m3", mult3), ("m5", mult5)]
    )
    pipeline.fit(X, y)
    expected_names = ["m2", "bad", "m3", "m5"]
    actual_names = [name for name, _ in pipeline.steps]
    assert expected_names == actual_names


@pytest.mark.parametrize("passthrough", [None, "passthrough"])
def test_set_pipeline_step_passthrough(passthrough):
    # Test setting Pipeline steps to None
    X = np.array([[1]])
    y = np.array([1])
    mult2 = Mult(mult=2)
    mult3 = Mult(mult=3)
    mult5 = Mult(mult=5)

    def make():
        return Pipeline([("m2", mult2), ("m3", mult3), ("last", mult5)])

    pipeline = make()

    exp = 2 * 3 * 5
    assert_array_equal([[exp]], pipeline.fit_transform(X, y))
    assert_array_equal([exp], pipeline.fit(X).predict(X))
    assert_array_equal(X, pipeline.inverse_transform([[exp]]))

    pipeline.set_params(m3=passthrough)
    exp = 2 * 5
    assert_array_equal([[exp]], pipeline.fit_transform(X, y))
    assert_array_equal([exp], pipeline.fit(X).predict(X))
    assert_array_equal(X, pipeline.inverse_transform([[exp]]))
    expected_params = {
        "steps": pipeline.steps,
        "m2": mult2,
        "m3": passthrough,
        "last": mult5,
        "memory": None,
        "m2__mult": 2,
        "last__mult": 5,
        "verbose": False,
    }
    assert pipeline.get_params(deep=True) == expected_params

    pipeline.set_params(m2=passthrough)
    exp = 5
    assert_array_equal([[exp]], pipeline.fit_transform(X, y))
    assert_array_equal([exp], pipeline.fit(X).predict(X))
    assert_array_equal(X, pipeline.inverse_transform([[exp]]))

    # for other methods, ensure no AttributeErrors on None:
    other_methods = [
        "predict_proba",
        "predict_log_proba",
        "decision_function",
        "transform",
        "score",
    ]
    for method in other_methods:
        getattr(pipeline, method)(X)

    pipeline.set_params(m2=mult2)
    exp = 2 * 5
    assert_array_equal([[exp]], pipeline.fit_transform(X, y))
    assert_array_equal([exp], pipeline.fit(X).predict(X))
    assert_array_equal(X, pipeline.inverse_transform([[exp]]))

    pipeline = make()
    pipeline.set_params(last=passthrough)
    # mult2 and mult3 are active
    exp = 6
    pipeline.fit(X, y)
    pipeline.transform(X)
    assert_array_equal([[exp]], pipeline.fit(X, y).transform(X))
    assert_array_equal([[exp]], pipeline.fit_transform(X, y))
    assert_array_equal(X, pipeline.inverse_transform([[exp]]))
    with raises(AttributeError, match="has no attribute 'predict'"):
        getattr(pipeline, "predict")

    # Check 'passthrough' step at construction time
    exp = 2 * 5
    pipeline = Pipeline([("m2", mult2), ("m3", passthrough), ("last", mult5)])
    assert_array_equal([[exp]], pipeline.fit_transform(X, y))
    assert_array_equal([exp], pipeline.fit(X).predict(X))
    assert_array_equal(X, pipeline.inverse_transform([[exp]]))


def test_pipeline_ducktyping():
    pipeline = make_pipeline(Mult(5))
    pipeline.predict
    pipeline.transform
    pipeline.inverse_transform

    pipeline = make_pipeline(Transf())
    assert not hasattr(pipeline, "predict")
    pipeline.transform
    pipeline.inverse_transform

    pipeline = make_pipeline("passthrough")
    assert pipeline.steps[0] == ("passthrough", "passthrough")
    assert not hasattr(pipeline, "predict")
    pipeline.transform
    pipeline.inverse_transform

    pipeline = make_pipeline(Transf(), NoInvTransf())
    assert not hasattr(pipeline, "predict")
    pipeline.transform
    assert not hasattr(pipeline, "inverse_transform")

    pipeline = make_pipeline(NoInvTransf(), Transf())
    assert not hasattr(pipeline, "predict")
    pipeline.transform
    assert not hasattr(pipeline, "inverse_transform")


def test_make_pipeline():
    t1 = Transf()
    t2 = Transf()
    pipe = make_pipeline(t1, t2)
    assert isinstance(pipe, Pipeline)
    assert pipe.steps[0][0] == "transf-1"
    assert pipe.steps[1][0] == "transf-2"

    pipe = make_pipeline(t1, t2, FitParamT())
    assert isinstance(pipe, Pipeline)
    assert pipe.steps[0][0] == "transf-1"
    assert pipe.steps[1][0] == "transf-2"
    assert pipe.steps[2][0] == "fitparamt"


def test_classes_property():
    iris = load_iris()
    X = iris.data
    y = iris.target

    reg = make_pipeline(SelectKBest(k=1), LinearRegression())
    reg.fit(X, y)
    with raises(AttributeError):
        getattr(reg, "classes_")

    clf = make_pipeline(
        SelectKBest(k=1),
        LogisticRegression(solver="lbfgs", multi_class="auto", random_state=0),
    )
    with raises(AttributeError):
        getattr(clf, "classes_")
    clf.fit(X, y)
    assert_array_equal(clf.classes_, np.unique(y))


def test_pipeline_memory_transformer():
    iris = load_iris()
    X = iris.data
    y = iris.target
    cachedir = mkdtemp()
    try:
        memory = Memory(cachedir, verbose=10)
        # Test with Transformer + SVC
        clf = SVC(gamma="scale", probability=True, random_state=0)
        transf = DummyTransf()
        pipe = Pipeline([("transf", clone(transf)), ("svc", clf)])
        cached_pipe = Pipeline([("transf", transf), ("svc", clf)], memory=memory)

        # Memoize the transformer at the first fit
        cached_pipe.fit(X, y)
        pipe.fit(X, y)
        # Get the time stamp of the tranformer in the cached pipeline
        expected_ts = cached_pipe.named_steps["transf"].timestamp_
        # Check that cached_pipe and pipe yield identical results
        assert_array_equal(pipe.predict(X), cached_pipe.predict(X))
        assert_array_equal(pipe.predict_proba(X), cached_pipe.predict_proba(X))
        assert_array_equal(pipe.predict_log_proba(X), cached_pipe.predict_log_proba(X))
        assert_array_equal(pipe.score(X, y), cached_pipe.score(X, y))
        assert_array_equal(
            pipe.named_steps["transf"].means_,
            cached_pipe.named_steps["transf"].means_,
        )
        assert not hasattr(transf, "means_")
        # Check that we are reading the cache while fitting
        # a second time
        cached_pipe.fit(X, y)
        # Check that cached_pipe and pipe yield identical results
        assert_array_equal(pipe.predict(X), cached_pipe.predict(X))
        assert_array_equal(pipe.predict_proba(X), cached_pipe.predict_proba(X))
        assert_array_equal(pipe.predict_log_proba(X), cached_pipe.predict_log_proba(X))
        assert_array_equal(pipe.score(X, y), cached_pipe.score(X, y))
        assert_array_equal(
            pipe.named_steps["transf"].means_,
            cached_pipe.named_steps["transf"].means_,
        )
        assert cached_pipe.named_steps["transf"].timestamp_ == expected_ts
        # Create a new pipeline with cloned estimators
        # Check that even changing the name step does not affect the cache hit
        clf_2 = SVC(gamma="scale", probability=True, random_state=0)
        transf_2 = DummyTransf()
        cached_pipe_2 = Pipeline(
            [("transf_2", transf_2), ("svc", clf_2)], memory=memory
        )
        cached_pipe_2.fit(X, y)

        # Check that cached_pipe and pipe yield identical results
        assert_array_equal(pipe.predict(X), cached_pipe_2.predict(X))
        assert_array_equal(pipe.predict_proba(X), cached_pipe_2.predict_proba(X))
        assert_array_equal(
            pipe.predict_log_proba(X), cached_pipe_2.predict_log_proba(X)
        )
        assert_array_equal(pipe.score(X, y), cached_pipe_2.score(X, y))
        assert_array_equal(
            pipe.named_steps["transf"].means_,
            cached_pipe_2.named_steps["transf_2"].means_,
        )
        assert cached_pipe_2.named_steps["transf_2"].timestamp_ == expected_ts
    finally:
        shutil.rmtree(cachedir)


def test_pipeline_memory_sampler():
    X, y = make_classification(
        n_classes=2,
        class_sep=2,
        weights=[0.1, 0.9],
        n_informative=3,
        n_redundant=1,
        flip_y=0,
        n_features=20,
        n_clusters_per_class=1,
        n_samples=5000,
        random_state=0,
    )
    cachedir = mkdtemp()
    try:
        memory = Memory(cachedir, verbose=10)
        # Test with Transformer + SVC
        clf = SVC(gamma="scale", probability=True, random_state=0)
        transf = DummySampler()
        pipe = Pipeline([("transf", clone(transf)), ("svc", clf)])
        cached_pipe = Pipeline([("transf", transf), ("svc", clf)], memory=memory)

        # Memoize the transformer at the first fit
        cached_pipe.fit(X, y)
        pipe.fit(X, y)
        # Get the time stamp of the tranformer in the cached pipeline
        expected_ts = cached_pipe.named_steps["transf"].timestamp_
        # Check that cached_pipe and pipe yield identical results
        assert_array_equal(pipe.predict(X), cached_pipe.predict(X))
        assert_array_equal(pipe.predict_proba(X), cached_pipe.predict_proba(X))
        assert_array_equal(pipe.predict_log_proba(X), cached_pipe.predict_log_proba(X))
        assert_array_equal(pipe.score(X, y), cached_pipe.score(X, y))
        assert_array_equal(
            pipe.named_steps["transf"].means_,
            cached_pipe.named_steps["transf"].means_,
        )
        assert not hasattr(transf, "means_")
        # Check that we are reading the cache while fitting
        # a second time
        cached_pipe.fit(X, y)
        # Check that cached_pipe and pipe yield identical results
        assert_array_equal(pipe.predict(X), cached_pipe.predict(X))
        assert_array_equal(pipe.predict_proba(X), cached_pipe.predict_proba(X))
        assert_array_equal(pipe.predict_log_proba(X), cached_pipe.predict_log_proba(X))
        assert_array_equal(pipe.score(X, y), cached_pipe.score(X, y))
        assert_array_equal(
            pipe.named_steps["transf"].means_,
            cached_pipe.named_steps["transf"].means_,
        )
        assert cached_pipe.named_steps["transf"].timestamp_ == expected_ts
        # Create a new pipeline with cloned estimators
        # Check that even changing the name step does not affect the cache hit
        clf_2 = SVC(gamma="scale", probability=True, random_state=0)
        transf_2 = DummySampler()
        cached_pipe_2 = Pipeline(
            [("transf_2", transf_2), ("svc", clf_2)], memory=memory
        )
        cached_pipe_2.fit(X, y)

        # Check that cached_pipe and pipe yield identical results
        assert_array_equal(pipe.predict(X), cached_pipe_2.predict(X))
        assert_array_equal(pipe.predict_proba(X), cached_pipe_2.predict_proba(X))
        assert_array_equal(
            pipe.predict_log_proba(X), cached_pipe_2.predict_log_proba(X)
        )
        assert_array_equal(pipe.score(X, y), cached_pipe_2.score(X, y))
        assert_array_equal(
            pipe.named_steps["transf"].means_,
            cached_pipe_2.named_steps["transf_2"].means_,
        )
        assert cached_pipe_2.named_steps["transf_2"].timestamp_ == expected_ts
    finally:
        shutil.rmtree(cachedir)


def test_pipeline_methods_pca_rus_svm():
    # Test the various methods of the pipeline (pca + svm).
    X, y = make_classification(
        n_classes=2,
        class_sep=2,
        weights=[0.1, 0.9],
        n_informative=3,
        n_redundant=1,
        flip_y=0,
        n_features=20,
        n_clusters_per_class=1,
        n_samples=5000,
        random_state=0,
    )

    # Test with PCA + SVC
    clf = SVC(gamma="scale", probability=True, random_state=0)
    pca = PCA()
    rus = RandomUnderSampler(random_state=0)
    pipe = Pipeline([("pca", pca), ("rus", rus), ("svc", clf)])
    pipe.fit(X, y)
    pipe.predict(X)
    pipe.predict_proba(X)
    pipe.predict_log_proba(X)
    pipe.score(X, y)


def test_pipeline_methods_rus_pca_svm():
    # Test the various methods of the pipeline (pca + svm).
    X, y = make_classification(
        n_classes=2,
        class_sep=2,
        weights=[0.1, 0.9],
        n_informative=3,
        n_redundant=1,
        flip_y=0,
        n_features=20,
        n_clusters_per_class=1,
        n_samples=5000,
        random_state=0,
    )

    # Test with PCA + SVC
    clf = SVC(gamma="scale", probability=True, random_state=0)
    pca = PCA()
    rus = RandomUnderSampler(random_state=0)
    pipe = Pipeline([("rus", rus), ("pca", pca), ("svc", clf)])
    pipe.fit(X, y)
    pipe.predict(X)
    pipe.predict_proba(X)
    pipe.predict_log_proba(X)
    pipe.score(X, y)


def test_pipeline_sample():
    # Test whether pipeline works with a sampler at the end.
    # Also test pipeline.sampler
    X, y = make_classification(
        n_classes=2,
        class_sep=2,
        weights=[0.1, 0.9],
        n_informative=3,
        n_redundant=1,
        flip_y=0,
        n_features=20,
        n_clusters_per_class=1,
        n_samples=5000,
        random_state=0,
    )

    rus = RandomUnderSampler(random_state=0)
    pipeline = Pipeline([("rus", rus)])

    # test transform and fit_transform:
    X_trans, y_trans = pipeline.fit_resample(X, y)
    X_trans2, y_trans2 = rus.fit_resample(X, y)
    assert_allclose(X_trans, X_trans2, rtol=R_TOL)
    assert_allclose(y_trans, y_trans2, rtol=R_TOL)

    pca = PCA()
    pipeline = Pipeline([("pca", PCA()), ("rus", rus)])

    X_trans, y_trans = pipeline.fit_resample(X, y)
    X_pca = pca.fit_transform(X)
    X_trans2, y_trans2 = rus.fit_resample(X_pca, y)
    # We round the value near to zero. It seems that PCA has some issue
    # with that
    X_trans[np.bitwise_and(X_trans < R_TOL, X_trans > -R_TOL)] = 0
    X_trans2[np.bitwise_and(X_trans2 < R_TOL, X_trans2 > -R_TOL)] = 0
    assert_allclose(X_trans, X_trans2, rtol=R_TOL)
    assert_allclose(y_trans, y_trans2, rtol=R_TOL)


def test_pipeline_sample_transform():
    # Test whether pipeline works with a sampler at the end.
    # Also test pipeline.sampler
    X, y = make_classification(
        n_classes=2,
        class_sep=2,
        weights=[0.1, 0.9],
        n_informative=3,
        n_redundant=1,
        flip_y=0,
        n_features=20,
        n_clusters_per_class=1,
        n_samples=5000,
        random_state=0,
    )

    rus = RandomUnderSampler(random_state=0)
    pca = PCA()
    pca2 = PCA()
    pipeline = Pipeline([("pca", pca), ("rus", rus), ("pca2", pca2)])

    pipeline.fit(X, y).transform(X)


def test_pipeline_none_classifier():
    # Test pipeline using None as preprocessing step and a classifier
    X, y = make_classification(
        n_classes=2,
        class_sep=2,
        weights=[0.1, 0.9],
        n_informative=3,
        n_redundant=1,
        flip_y=0,
        n_features=20,
        n_clusters_per_class=1,
        n_samples=5000,
        random_state=0,
    )
    clf = LogisticRegression(solver="lbfgs", random_state=0)
    pipe = make_pipeline(None, clf)
    pipe.fit(X, y)
    pipe.predict(X)
    pipe.predict_proba(X)
    pipe.decision_function(X)
    pipe.score(X, y)


def test_pipeline_none_sampler_classifier():
    # Test pipeline using None, RUS and a classifier
    X, y = make_classification(
        n_classes=2,
        class_sep=2,
        weights=[0.1, 0.9],
        n_informative=3,
        n_redundant=1,
        flip_y=0,
        n_features=20,
        n_clusters_per_class=1,
        n_samples=5000,
        random_state=0,
    )
    clf = LogisticRegression(solver="lbfgs", random_state=0)
    rus = RandomUnderSampler(random_state=0)
    pipe = make_pipeline(None, rus, clf)
    pipe.fit(X, y)
    pipe.predict(X)
    pipe.predict_proba(X)
    pipe.decision_function(X)
    pipe.score(X, y)


def test_pipeline_sampler_none_classifier():
    # Test pipeline using RUS, None and a classifier
    X, y = make_classification(
        n_classes=2,
        class_sep=2,
        weights=[0.1, 0.9],
        n_informative=3,
        n_redundant=1,
        flip_y=0,
        n_features=20,
        n_clusters_per_class=1,
        n_samples=5000,
        random_state=0,
    )
    clf = LogisticRegression(solver="lbfgs", random_state=0)
    rus = RandomUnderSampler(random_state=0)
    pipe = make_pipeline(rus, None, clf)
    pipe.fit(X, y)
    pipe.predict(X)
    pipe.predict_proba(X)
    pipe.decision_function(X)
    pipe.score(X, y)


def test_pipeline_none_sampler_sample():
    # Test pipeline using None step and a sampler
    X, y = make_classification(
        n_classes=2,
        class_sep=2,
        weights=[0.1, 0.9],
        n_informative=3,
        n_redundant=1,
        flip_y=0,
        n_features=20,
        n_clusters_per_class=1,
        n_samples=5000,
        random_state=0,
    )

    rus = RandomUnderSampler(random_state=0)
    pipe = make_pipeline(None, rus)
    pipe.fit_resample(X, y)


def test_pipeline_none_transformer():
    # Test pipeline using None and a transformer that implements transform and
    # inverse_transform
    X, y = make_classification(
        n_classes=2,
        class_sep=2,
        weights=[0.1, 0.9],
        n_informative=3,
        n_redundant=1,
        flip_y=0,
        n_features=20,
        n_clusters_per_class=1,
        n_samples=5000,
        random_state=0,
    )

    pca = PCA(whiten=True)
    pipe = make_pipeline(None, pca)
    pipe.fit(X, y)
    X_trans = pipe.transform(X)
    X_inversed = pipe.inverse_transform(X_trans)
    assert_array_almost_equal(X, X_inversed)


def test_pipeline_methods_anova_rus():
    # Test the various methods of the pipeline (anova).
    X, y = make_classification(
        n_classes=2,
        class_sep=2,
        weights=[0.1, 0.9],
        n_informative=3,
        n_redundant=1,
        flip_y=0,
        n_features=20,
        n_clusters_per_class=1,
        n_samples=5000,
        random_state=0,
    )
    # Test with RandomUnderSampling + Anova + LogisticRegression
    clf = LogisticRegression(solver="lbfgs")
    rus = RandomUnderSampler(random_state=0)
    filter1 = SelectKBest(f_classif, k=2)
    pipe = Pipeline([("rus", rus), ("anova", filter1), ("logistic", clf)])
    pipe.fit(X, y)
    pipe.predict(X)
    pipe.predict_proba(X)
    pipe.predict_log_proba(X)
    pipe.score(X, y)


def test_pipeline_with_step_that_implements_both_sample_and_transform():
    # Test the various methods of the pipeline (anova).
    X, y = make_classification(
        n_classes=2,
        class_sep=2,
        weights=[0.1, 0.9],
        n_informative=3,
        n_redundant=1,
        flip_y=0,
        n_features=20,
        n_clusters_per_class=1,
        n_samples=5000,
        random_state=0,
    )

    clf = LogisticRegression(solver="lbfgs")
    with raises(TypeError):
        pipeline = Pipeline([("step", FitTransformSample()), ("logistic", clf)])
        pipeline.fit(X, y)


def test_pipeline_with_step_that_it_is_pipeline():
    # Test the various methods of the pipeline (anova).
    X, y = make_classification(
        n_classes=2,
        class_sep=2,
        weights=[0.1, 0.9],
        n_informative=3,
        n_redundant=1,
        flip_y=0,
        n_features=20,
        n_clusters_per_class=1,
        n_samples=5000,
        random_state=0,
    )
    # Test with RandomUnderSampling + Anova + LogisticRegression
    clf = LogisticRegression(solver="lbfgs")
    rus = RandomUnderSampler(random_state=0)
    filter1 = SelectKBest(f_classif, k=2)
    pipe1 = Pipeline([("rus", rus), ("anova", filter1)])
    with raises(TypeError):
        pipe2 = Pipeline([("pipe1", pipe1), ("logistic", clf)])
        pipe2.fit(X, y)


def test_pipeline_fit_then_sample_with_sampler_last_estimator():
    X, y = make_classification(
        n_classes=2,
        class_sep=2,
        weights=[0.1, 0.9],
        n_informative=3,
        n_redundant=1,
        flip_y=0,
        n_features=20,
        n_clusters_per_class=1,
        n_samples=50000,
        random_state=0,
    )

    rus = RandomUnderSampler(random_state=42)
    enn = ENN()
    pipeline = make_pipeline(rus, enn)
    X_fit_resample_resampled, y_fit_resample_resampled = pipeline.fit_resample(X, y)
    pipeline = make_pipeline(rus, enn)
    pipeline.fit(X, y)
    X_fit_then_sample_res, y_fit_then_sample_res = pipeline.fit_resample(X, y)
    assert_array_equal(X_fit_resample_resampled, X_fit_then_sample_res)
    assert_array_equal(y_fit_resample_resampled, y_fit_then_sample_res)


def test_pipeline_fit_then_sample_3_samplers_with_sampler_last_estimator():
    X, y = make_classification(
        n_classes=2,
        class_sep=2,
        weights=[0.1, 0.9],
        n_informative=3,
        n_redundant=1,
        flip_y=0,
        n_features=20,
        n_clusters_per_class=1,
        n_samples=50000,
        random_state=0,
    )

    rus = RandomUnderSampler(random_state=42)
    enn = ENN()
    pipeline = make_pipeline(rus, enn, rus)
    X_fit_resample_resampled, y_fit_resample_resampled = pipeline.fit_resample(X, y)
    pipeline = make_pipeline(rus, enn, rus)
    pipeline.fit(X, y)
    X_fit_then_sample_res, y_fit_then_sample_res = pipeline.fit_resample(X, y)
    assert_array_equal(X_fit_resample_resampled, X_fit_then_sample_res)
    assert_array_equal(y_fit_resample_resampled, y_fit_then_sample_res)


def test_make_pipeline_memory():
    cachedir = mkdtemp()
    try:
        memory = Memory(cachedir, verbose=10)
        pipeline = make_pipeline(DummyTransf(), SVC(gamma="scale"), memory=memory)
        assert pipeline.memory is memory
        pipeline = make_pipeline(DummyTransf(), SVC(gamma="scale"))
        assert pipeline.memory is None
    finally:
        shutil.rmtree(cachedir)


def test_predict_with_predict_params():
    # tests that Pipeline passes predict_params to the final estimator
    # when predict is invoked
    pipe = Pipeline([("transf", Transf()), ("clf", DummyEstimatorParams())])
    pipe.fit(None, None)
    pipe.predict(X=None, got_attribute=True)
    assert pipe.named_steps["clf"].got_attribute


def test_resampler_last_stage_passthrough():
    X, y = make_classification(
        n_classes=2,
        class_sep=2,
        weights=[0.1, 0.9],
        n_informative=3,
        n_redundant=1,
        flip_y=0,
        n_features=20,
        n_clusters_per_class=1,
        n_samples=50000,
        random_state=0,
    )

    rus = RandomUnderSampler(random_state=42)
    pipe = make_pipeline(rus, None)
    pipe.fit_resample(X, y)


def test_pipeline_score_samples_pca_lof_binary():
    X, y = make_classification(
        n_classes=2,
        class_sep=2,
        weights=[0.3, 0.7],
        n_informative=3,
        n_redundant=1,
        flip_y=0,
        n_features=20,
        n_clusters_per_class=1,
        n_samples=500,
        random_state=0,
    )
    # Test that the score_samples method is implemented on a pipeline.
    # Test that the score_samples method on pipeline yields same results as
    # applying transform and score_samples steps separately.
    rus = RandomUnderSampler(random_state=42)
    pca = PCA(svd_solver="full", n_components="mle", whiten=True)
    lof = LocalOutlierFactor(novelty=True)
    pipe = Pipeline([("rus", rus), ("pca", pca), ("lof", lof)])
    pipe.fit(X, y)
    # Check the shapes
    assert pipe.score_samples(X).shape == (X.shape[0],)
    # Check the values
    X_res, _ = rus.fit_resample(X, y)
    lof.fit(pca.fit_transform(X_res))
    assert_allclose(pipe.score_samples(X), lof.score_samples(pca.transform(X)))


def test_score_samples_on_pipeline_without_score_samples():
    X = np.array([[1], [2]])
    y = np.array([1, 2])
    # Test that a pipeline does not have score_samples method when the final
    # step of the pipeline does not have score_samples defined.
    pipe = make_pipeline(LogisticRegression())
    pipe.fit(X, y)
    with pytest.raises(
        AttributeError,
        match="'LogisticRegression' object has no attribute 'score_samples'",
    ):
        pipe.score_samples(X)


def test_pipeline_param_error():
    clf = make_pipeline(LogisticRegression())
    with pytest.raises(
        ValueError,
        match="Pipeline.fit does not accept the sample_weight parameter",
    ):
        clf.fit([[0], [0]], [0, 1], sample_weight=[1, 1])


parameter_grid_test_verbose = (
    (est, pattern, method)
    for (est, pattern), method in itertools.product(
        [
            (
                Pipeline([("transf", Transf()), ("clf", FitParamT())]),
                r"\[Pipeline\].*\(step 1 of 2\) Processing transf.* total=.*\n"
                r"\[Pipeline\].*\(step 2 of 2\) Processing clf.* total=.*\n$",
            ),
            (
                Pipeline([("transf", Transf()), ("noop", None), ("clf", FitParamT())]),
                r"\[Pipeline\].*\(step 1 of 3\) Processing transf.* total=.*\n"
                r"\[Pipeline\].*\(step 2 of 3\) Processing noop.* total=.*\n"
                r"\[Pipeline\].*\(step 3 of 3\) Processing clf.* total=.*\n$",
            ),
            (
                Pipeline(
                    [
                        ("transf", Transf()),
                        ("noop", "passthrough"),
                        ("clf", FitParamT()),
                    ]
                ),
                r"\[Pipeline\].*\(step 1 of 3\) Processing transf.* total=.*\n"
                r"\[Pipeline\].*\(step 2 of 3\) Processing noop.* total=.*\n"
                r"\[Pipeline\].*\(step 3 of 3\) Processing clf.* total=.*\n$",
            ),
            (
                Pipeline([("transf", Transf()), ("clf", None)]),
                r"\[Pipeline\].*\(step 1 of 2\) Processing transf.* total=.*\n"
                r"\[Pipeline\].*\(step 2 of 2\) Processing clf.* total=.*\n$",
            ),
            (
                Pipeline([("transf", None), ("mult", Mult())]),
                r"\[Pipeline\].*\(step 1 of 2\) Processing transf.* total=.*\n"
                r"\[Pipeline\].*\(step 2 of 2\) Processing mult.* total=.*\n$",
            ),
            (
                Pipeline([("transf", "passthrough"), ("mult", Mult())]),
                r"\[Pipeline\].*\(step 1 of 2\) Processing transf.* total=.*\n"
                r"\[Pipeline\].*\(step 2 of 2\) Processing mult.* total=.*\n$",
            ),
            (
                FeatureUnion([("mult1", Mult()), ("mult2", Mult())]),
                r"\[FeatureUnion\].*\(step 1 of 2\) Processing mult1.* total=.*\n"
                r"\[FeatureUnion\].*\(step 2 of 2\) Processing mult2.* total=.*\n$",
            ),
            (
                FeatureUnion([("mult1", "drop"), ("mult2", Mult()), ("mult3", "drop")]),
                r"\[FeatureUnion\].*\(step 1 of 1\) Processing mult2.* total=.*\n$",
            ),
        ],
        ["fit", "fit_transform", "fit_predict"],
    )
    if hasattr(est, method)
    and not (
        method == "fit_transform"
        and hasattr(est, "steps")
        and isinstance(est.steps[-1][1], FitParamT)
    )
)


@pytest.mark.parametrize("est, pattern, method", parameter_grid_test_verbose)
def test_verbose(est, method, pattern, capsys):
    func = getattr(est, method)

    X = [[1, 2, 3], [4, 5, 6]]
    y = [[7], [8]]

    est.set_params(verbose=False)
    func(X, y)
    assert not capsys.readouterr().out, "Got output for verbose=False"

    est.set_params(verbose=True)
    func(X, y)
    assert re.match(pattern, capsys.readouterr().out)


def test_pipeline_score_samples_pca_lof_multiclass():
    X, y = load_iris(return_X_y=True)
    sampling_strategy = {0: 50, 1: 30, 2: 20}
    X, y = make_imbalance(X, y, sampling_strategy=sampling_strategy)
    # Test that the score_samples method is implemented on a pipeline.
    # Test that the score_samples method on pipeline yields same results as
    # applying transform and score_samples steps separately.
    rus = RandomUnderSampler()
    pca = PCA(svd_solver="full", n_components="mle", whiten=True)
    lof = LocalOutlierFactor(novelty=True)
    pipe = Pipeline([("rus", rus), ("pca", pca), ("lof", lof)])
    pipe.fit(X, y)
    # Check the shapes
    assert pipe.score_samples(X).shape == (X.shape[0],)
    # Check the values
    lof.fit(pca.fit_transform(X))
    assert_allclose(pipe.score_samples(X), lof.score_samples(pca.transform(X)))


def test_pipeline_param_validation():
    model = Pipeline(
        [("sampler", RandomUnderSampler()), ("classifier", LogisticRegression())]
    )
    check_param_validation("Pipeline", model)


@pytest.mark.skipif(
    sklearn_version < parse_version("1.2"), reason="requires scikit-learn >= 1.2"
)
def test_pipeline_with_set_output():
    pd = pytest.importorskip("pandas")
    X, y = load_iris(return_X_y=True, as_frame=True)
    pipeline = make_pipeline(
        StandardScaler(), RandomUnderSampler(), LogisticRegression()
    ).set_output(transform="default")
    pipeline.fit(X, y)

    X_res, y_res = pipeline[:-1].fit_resample(X, y)
    assert isinstance(X_res, np.ndarray)
    # transformer will not change `y` and sampler will always preserve the type of `y`
    assert isinstance(y_res, type(y))

    pipeline.set_output(transform="pandas")
    X_res, y_res = pipeline[:-1].fit_resample(X, y)

    assert isinstance(X_res, pd.DataFrame)
    # transformer will not change `y` and sampler will always preserve the type of `y`
    assert isinstance(y_res, type(y))
