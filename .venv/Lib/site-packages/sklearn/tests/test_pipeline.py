"""
Test the pipeline module.
"""
import itertools
import re
import shutil
import time
from tempfile import mkdtemp

import joblib
import numpy as np
import pytest
from scipy import sparse

from sklearn.base import BaseEstimator, TransformerMixin, clone, is_classifier
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline, make_union
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils._testing import (
    MinimalClassifier,
    MinimalRegressor,
    MinimalTransformer,
    assert_allclose,
    assert_array_almost_equal,
    assert_array_equal,
)
from sklearn.utils.validation import check_is_fitted

iris = load_iris()

JUNK_FOOD_DOCS = (
    "the pizza pizza beer copyright",
    "the pizza burger beer copyright",
    "the the pizza beer beer copyright",
    "the burger beer beer copyright",
    "the coke burger coke copyright",
    "the coke burger burger",
)


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
    def transform(self, X):
        return X


class Transf(NoInvTransf):
    def transform(self, X):
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

    def predict_proba(self, X, got_attribute=False):
        self.got_attribute = got_attribute
        return self

    def predict_log_proba(self, X, got_attribute=False):
        self.got_attribute = got_attribute
        return self


def test_pipeline_invalid_parameters():
    # Test the various init parameters of the pipeline in fit
    # method
    pipeline = Pipeline([(1, 1)])
    with pytest.raises(TypeError):
        pipeline.fit([[1]], [1])

    # Check that we can't fit pipelines with objects without fit
    # method
    msg = (
        "Last step of Pipeline should implement fit "
        "or be the string 'passthrough'"
        ".*NoFit.*"
    )
    pipeline = Pipeline([("clf", NoFit())])
    with pytest.raises(TypeError, match=msg):
        pipeline.fit([[1]], [1])

    # Smoke test with only an estimator
    clf = NoTrans()
    pipe = Pipeline([("svc", clf)])
    assert pipe.get_params(deep=True) == dict(
        svc__a=None, svc__b=None, svc=clf, **pipe.get_params(deep=False)
    )

    # Check that params are set
    pipe.set_params(svc__a=0.1)
    assert clf.a == 0.1
    assert clf.b is None
    # Smoke test the repr:
    repr(pipe)

    # Test with two objects
    clf = SVC()
    filter1 = SelectKBest(f_classif)
    pipe = Pipeline([("anova", filter1), ("svc", clf)])

    # Check that estimators are not cloned on pipeline construction
    assert pipe.named_steps["anova"] is filter1
    assert pipe.named_steps["svc"] is clf

    # Check that we can't fit with non-transformers on the way
    # Note that NoTrans implements fit, but not transform
    msg = "All intermediate steps should be transformers.*\\bNoTrans\\b.*"
    pipeline = Pipeline([("t", NoTrans()), ("svc", clf)])
    with pytest.raises(TypeError, match=msg):
        pipeline.fit([[1]], [1])

    # Check that params are set
    pipe.set_params(svc__C=0.1)
    assert clf.C == 0.1
    # Smoke test the repr:
    repr(pipe)

    # Check that params are not set when naming them wrong
    msg = re.escape(
        "Invalid parameter 'C' for estimator SelectKBest(). Valid parameters are: ['k',"
        " 'score_func']."
    )
    with pytest.raises(ValueError, match=msg):
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


def test_pipeline_init_tuple():
    # Pipeline accepts steps as tuple
    X = np.array([[1, 2]])
    pipe = Pipeline((("transf", Transf()), ("clf", FitParamT())))
    pipe.fit(X, y=None)
    pipe.score(X)

    pipe.set_params(transf="passthrough")
    pipe.fit(X, y=None)
    pipe.score(X)


def test_pipeline_methods_anova():
    # Test the various methods of the pipeline (anova).
    X = iris.data
    y = iris.target
    # Test with Anova + LogisticRegression
    clf = LogisticRegression()
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

    msg = re.escape("fit() got an unexpected keyword argument 'bad'")
    with pytest.raises(TypeError, match=msg):
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

    msg = re.escape("score() got an unexpected keyword argument 'sample_weight'")
    with pytest.raises(TypeError, match=msg):
        pipe.score(X, sample_weight=np.array([2, 3]))


def test_pipeline_raise_set_params_error():
    # Test pipeline raises set params error message for nested models.
    pipe = Pipeline([("cls", LinearRegression())])

    # expected error message
    error_msg = re.escape(
        "Invalid parameter 'fake' for estimator Pipeline(steps=[('cls',"
        " LinearRegression())]). Valid parameters are: ['memory', 'steps', 'verbose']."
    )
    with pytest.raises(ValueError, match=error_msg):
        pipe.set_params(fake="nope")

    # invalid outer parameter name for compound parameter: the expected error message
    # is the same as above.
    with pytest.raises(ValueError, match=error_msg):
        pipe.set_params(fake__estimator="nope")

    # expected error message for invalid inner parameter
    error_msg = re.escape(
        "Invalid parameter 'invalid_param' for estimator LinearRegression(). Valid"
        " parameters are: ['copy_X', 'fit_intercept', 'n_jobs', 'positive']."
    )
    with pytest.raises(ValueError, match=error_msg):
        pipe.set_params(cls__invalid_param="nope")


def test_pipeline_methods_pca_svm():
    # Test the various methods of the pipeline (pca + svm).
    X = iris.data
    y = iris.target
    # Test with PCA + SVC
    clf = SVC(probability=True, random_state=0)
    pca = PCA(svd_solver="full", n_components="mle", whiten=True)
    pipe = Pipeline([("pca", pca), ("svc", clf)])
    pipe.fit(X, y)
    pipe.predict(X)
    pipe.predict_proba(X)
    pipe.predict_log_proba(X)
    pipe.score(X, y)


def test_pipeline_score_samples_pca_lof():
    X = iris.data
    # Test that the score_samples method is implemented on a pipeline.
    # Test that the score_samples method on pipeline yields same results as
    # applying transform and score_samples steps separately.
    pca = PCA(svd_solver="full", n_components="mle", whiten=True)
    lof = LocalOutlierFactor(novelty=True)
    pipe = Pipeline([("pca", pca), ("lof", lof)])
    pipe.fit(X)
    # Check the shapes
    assert pipe.score_samples(X).shape == (X.shape[0],)
    # Check the values
    lof.fit(pca.fit_transform(X))
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


def test_pipeline_methods_preprocessing_svm():
    # Test the various methods of the pipeline (preprocessing + svm).
    X = iris.data
    y = iris.target
    n_samples = X.shape[0]
    n_classes = len(np.unique(y))
    scaler = StandardScaler()
    pca = PCA(n_components=2, svd_solver="randomized", whiten=True)
    clf = SVC(probability=True, random_state=0, decision_function_shape="ovr")

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
    scaler = StandardScaler()
    km = KMeans(random_state=0, n_init="auto")
    # As pipeline doesn't clone estimators on construction,
    # it must have its own estimators
    scaler_for_pipeline = StandardScaler()
    km_for_pipeline = KMeans(random_state=0, n_init="auto")

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

    msg = "'PCA' object has no attribute 'fit_predict'"
    with pytest.raises(AttributeError, match=msg):
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


@pytest.mark.parametrize(
    "method_name", ["predict", "predict_proba", "predict_log_proba"]
)
def test_predict_methods_with_predict_params(method_name):
    # tests that Pipeline passes predict_* to the final estimator
    # when predict_* is invoked
    pipe = Pipeline([("transf", Transf()), ("clf", DummyEstimatorParams())])
    pipe.fit(None, None)
    method = getattr(pipe, method_name)
    method(X=None, got_attribute=True)

    assert pipe.named_steps["clf"].got_attribute


def test_feature_union():
    # basic sanity check for feature union
    X = iris.data
    X -= X.mean(axis=0)
    y = iris.target
    svd = TruncatedSVD(n_components=2, random_state=0)
    select = SelectKBest(k=1)
    fs = FeatureUnion([("svd", svd), ("select", select)])
    fs.fit(X, y)
    X_transformed = fs.transform(X)
    assert X_transformed.shape == (X.shape[0], 3)

    # check if it does the expected thing
    assert_array_almost_equal(X_transformed[:, :-1], svd.fit_transform(X))
    assert_array_equal(X_transformed[:, -1], select.fit_transform(X, y).ravel())

    # test if it also works for sparse input
    # We use a different svd object to control the random_state stream
    fs = FeatureUnion([("svd", svd), ("select", select)])
    X_sp = sparse.csr_matrix(X)
    X_sp_transformed = fs.fit_transform(X_sp, y)
    assert_array_almost_equal(X_transformed, X_sp_transformed.toarray())

    # Test clone
    fs2 = clone(fs)
    assert fs.transformer_list[0][1] is not fs2.transformer_list[0][1]

    # test setting parameters
    fs.set_params(select__k=2)
    assert fs.fit_transform(X, y).shape == (X.shape[0], 4)

    # test it works with transformers missing fit_transform
    fs = FeatureUnion([("mock", Transf()), ("svd", svd), ("select", select)])
    X_transformed = fs.fit_transform(X, y)
    assert X_transformed.shape == (X.shape[0], 8)

    # test error if some elements do not support transform
    msg = "All estimators should implement fit and transform.*\\bNoTrans\\b"
    fs = FeatureUnion([("transform", Transf()), ("no_transform", NoTrans())])
    with pytest.raises(TypeError, match=msg):
        fs.fit(X)

    # test that init accepts tuples
    fs = FeatureUnion((("svd", svd), ("select", select)))
    fs.fit(X, y)


def test_feature_union_named_transformers():
    """Check the behaviour of `named_transformers` attribute."""
    transf = Transf()
    noinvtransf = NoInvTransf()
    fs = FeatureUnion([("transf", transf), ("noinvtransf", noinvtransf)])
    assert fs.named_transformers["transf"] == transf
    assert fs.named_transformers["noinvtransf"] == noinvtransf

    # test named attribute
    assert fs.named_transformers.transf == transf
    assert fs.named_transformers.noinvtransf == noinvtransf


def test_make_union():
    pca = PCA(svd_solver="full")
    mock = Transf()
    fu = make_union(pca, mock)
    names, transformers = zip(*fu.transformer_list)
    assert names == ("pca", "transf")
    assert transformers == (pca, mock)


def test_make_union_kwargs():
    pca = PCA(svd_solver="full")
    mock = Transf()
    fu = make_union(pca, mock, n_jobs=3)
    assert fu.transformer_list == make_union(pca, mock).transformer_list
    assert 3 == fu.n_jobs

    # invalid keyword parameters should raise an error message
    msg = re.escape(
        "make_union() got an unexpected keyword argument 'transformer_weights'"
    )
    with pytest.raises(TypeError, match=msg):
        make_union(pca, mock, transformer_weights={"pca": 10, "Transf": 1})


def test_pipeline_transform():
    # Test whether pipeline works with a transformer at the end.
    # Also test pipeline.transform and pipeline.inverse_transform
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
    X = iris.data
    y = iris.target
    transf = Transf()
    pipeline = Pipeline([("mock", transf)])

    # test fit_transform:
    X_trans = pipeline.fit_transform(X, y)
    X_trans2 = transf.fit(X, y).transform(X)
    assert_array_almost_equal(X_trans, X_trans2)


@pytest.mark.parametrize(
    "start, end", [(0, 1), (0, 2), (1, 2), (1, 3), (None, 1), (1, None), (None, None)]
)
def test_pipeline_slice(start, end):
    pipe = Pipeline(
        [("transf1", Transf()), ("transf2", Transf()), ("clf", FitParamT())],
        memory="123",
        verbose=True,
    )
    pipe_slice = pipe[start:end]
    # Test class
    assert isinstance(pipe_slice, Pipeline)
    # Test steps
    assert pipe_slice.steps == pipe.steps[start:end]
    # Test named_steps attribute
    assert (
        list(pipe_slice.named_steps.items())
        == list(pipe.named_steps.items())[start:end]
    )
    # Test the rest of the parameters
    pipe_params = pipe.get_params(deep=False)
    pipe_slice_params = pipe_slice.get_params(deep=False)
    del pipe_params["steps"]
    del pipe_slice_params["steps"]
    assert pipe_params == pipe_slice_params
    # Test exception
    msg = "Pipeline slicing only supports a step of 1"
    with pytest.raises(ValueError, match=msg):
        pipe[start:end:-1]


def test_pipeline_index():
    transf = Transf()
    clf = FitParamT()
    pipe = Pipeline([("transf", transf), ("clf", clf)])
    assert pipe[0] == transf
    assert pipe["transf"] == transf
    assert pipe[-1] == clf
    assert pipe["clf"] == clf

    # should raise an error if slicing out of range
    with pytest.raises(IndexError):
        pipe[3]

    # should raise an error if indexing with wrong element name
    with pytest.raises(KeyError):
        pipe["foobar"]


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
    msg = re.escape(
        "Last step of Pipeline should implement fit or be the string 'passthrough'."
    )
    with pytest.raises(TypeError, match=msg):
        pipeline.fit([[1]], [1])

    msg = "This 'Pipeline' has no attribute 'fit_transform'"
    with pytest.raises(AttributeError, match=msg):
        pipeline.fit_transform([[1]], [1])


def test_pipeline_named_steps():
    transf = Transf()
    mult2 = Mult(mult=2)
    pipeline = Pipeline([("mock", transf), ("mult", mult2)])

    # Test access via named_steps bunch object
    assert "mock" in pipeline.named_steps
    assert "mock2" not in pipeline.named_steps
    assert pipeline.named_steps.mock is transf
    assert pipeline.named_steps.mult is mult2

    # Test bunch with conflict attribute of dict
    pipeline = Pipeline([("values", transf), ("mult", mult2)])
    assert pipeline.named_steps.values is not transf
    assert pipeline.named_steps.mult is mult2


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
    assert pipeline.get_params(deep=True) == {
        "steps": pipeline.steps,
        "m2": mult2,
        "m3": passthrough,
        "last": mult5,
        "memory": None,
        "m2__mult": 2,
        "last__mult": 5,
        "verbose": False,
    }

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
    assert_array_equal([[exp]], pipeline.fit(X, y).transform(X))
    assert_array_equal([[exp]], pipeline.fit_transform(X, y))
    assert_array_equal(X, pipeline.inverse_transform([[exp]]))

    msg = "'str' object has no attribute 'predict'"
    with pytest.raises(AttributeError, match=msg):
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


def test_feature_union_weights():
    # test feature union with transformer weights
    X = iris.data
    y = iris.target
    pca = PCA(n_components=2, svd_solver="randomized", random_state=0)
    select = SelectKBest(k=1)
    # test using fit followed by transform
    fs = FeatureUnion(
        [("pca", pca), ("select", select)], transformer_weights={"pca": 10}
    )
    fs.fit(X, y)
    X_transformed = fs.transform(X)
    # test using fit_transform
    fs = FeatureUnion(
        [("pca", pca), ("select", select)], transformer_weights={"pca": 10}
    )
    X_fit_transformed = fs.fit_transform(X, y)
    # test it works with transformers missing fit_transform
    fs = FeatureUnion(
        [("mock", Transf()), ("pca", pca), ("select", select)],
        transformer_weights={"mock": 10},
    )
    X_fit_transformed_wo_method = fs.fit_transform(X, y)
    # check against expected result

    # We use a different pca object to control the random_state stream
    assert_array_almost_equal(X_transformed[:, :-1], 10 * pca.fit_transform(X))
    assert_array_equal(X_transformed[:, -1], select.fit_transform(X, y).ravel())
    assert_array_almost_equal(X_fit_transformed[:, :-1], 10 * pca.fit_transform(X))
    assert_array_equal(X_fit_transformed[:, -1], select.fit_transform(X, y).ravel())
    assert X_fit_transformed_wo_method.shape == (X.shape[0], 7)


def test_feature_union_parallel():
    # test that n_jobs work for FeatureUnion
    X = JUNK_FOOD_DOCS

    fs = FeatureUnion(
        [
            ("words", CountVectorizer(analyzer="word")),
            ("chars", CountVectorizer(analyzer="char")),
        ]
    )

    fs_parallel = FeatureUnion(
        [
            ("words", CountVectorizer(analyzer="word")),
            ("chars", CountVectorizer(analyzer="char")),
        ],
        n_jobs=2,
    )

    fs_parallel2 = FeatureUnion(
        [
            ("words", CountVectorizer(analyzer="word")),
            ("chars", CountVectorizer(analyzer="char")),
        ],
        n_jobs=2,
    )

    fs.fit(X)
    X_transformed = fs.transform(X)
    assert X_transformed.shape[0] == len(X)

    fs_parallel.fit(X)
    X_transformed_parallel = fs_parallel.transform(X)
    assert X_transformed.shape == X_transformed_parallel.shape
    assert_array_equal(X_transformed.toarray(), X_transformed_parallel.toarray())

    # fit_transform should behave the same
    X_transformed_parallel2 = fs_parallel2.fit_transform(X)
    assert_array_equal(X_transformed.toarray(), X_transformed_parallel2.toarray())

    # transformers should stay fit after fit_transform
    X_transformed_parallel2 = fs_parallel2.transform(X)
    assert_array_equal(X_transformed.toarray(), X_transformed_parallel2.toarray())


def test_feature_union_feature_names():
    word_vect = CountVectorizer(analyzer="word")
    char_vect = CountVectorizer(analyzer="char_wb", ngram_range=(3, 3))
    ft = FeatureUnion([("chars", char_vect), ("words", word_vect)])
    ft.fit(JUNK_FOOD_DOCS)
    feature_names = ft.get_feature_names_out()
    for feat in feature_names:
        assert "chars__" in feat or "words__" in feat
    assert len(feature_names) == 35

    ft = FeatureUnion([("tr1", Transf())]).fit([[1]])

    msg = re.escape(
        "Transformer tr1 (type Transf) does not provide get_feature_names_out"
    )
    with pytest.raises(AttributeError, match=msg):
        ft.get_feature_names_out()


def test_classes_property():
    X = iris.data
    y = iris.target

    reg = make_pipeline(SelectKBest(k=1), LinearRegression())
    reg.fit(X, y)
    with pytest.raises(AttributeError):
        getattr(reg, "classes_")

    clf = make_pipeline(SelectKBest(k=1), LogisticRegression(random_state=0))
    with pytest.raises(AttributeError):
        getattr(clf, "classes_")
    clf.fit(X, y)
    assert_array_equal(clf.classes_, np.unique(y))


def test_set_feature_union_steps():
    mult2 = Mult(2)
    mult3 = Mult(3)
    mult5 = Mult(5)

    mult3.get_feature_names_out = lambda input_features: ["x3"]
    mult2.get_feature_names_out = lambda input_features: ["x2"]
    mult5.get_feature_names_out = lambda input_features: ["x5"]

    ft = FeatureUnion([("m2", mult2), ("m3", mult3)])
    assert_array_equal([[2, 3]], ft.transform(np.asarray([[1]])))
    assert_array_equal(["m2__x2", "m3__x3"], ft.get_feature_names_out())

    # Directly setting attr
    ft.transformer_list = [("m5", mult5)]
    assert_array_equal([[5]], ft.transform(np.asarray([[1]])))
    assert_array_equal(["m5__x5"], ft.get_feature_names_out())

    # Using set_params
    ft.set_params(transformer_list=[("mock", mult3)])
    assert_array_equal([[3]], ft.transform(np.asarray([[1]])))
    assert_array_equal(["mock__x3"], ft.get_feature_names_out())

    # Using set_params to replace single step
    ft.set_params(mock=mult5)
    assert_array_equal([[5]], ft.transform(np.asarray([[1]])))
    assert_array_equal(["mock__x5"], ft.get_feature_names_out())


def test_set_feature_union_step_drop():
    mult2 = Mult(2)
    mult3 = Mult(3)

    mult2.get_feature_names_out = lambda input_features: ["x2"]
    mult3.get_feature_names_out = lambda input_features: ["x3"]

    X = np.asarray([[1]])

    ft = FeatureUnion([("m2", mult2), ("m3", mult3)])
    assert_array_equal([[2, 3]], ft.fit(X).transform(X))
    assert_array_equal([[2, 3]], ft.fit_transform(X))
    assert_array_equal(["m2__x2", "m3__x3"], ft.get_feature_names_out())

    ft.set_params(m2="drop")
    assert_array_equal([[3]], ft.fit(X).transform(X))
    assert_array_equal([[3]], ft.fit_transform(X))
    assert_array_equal(["m3__x3"], ft.get_feature_names_out())

    ft.set_params(m3="drop")
    assert_array_equal([[]], ft.fit(X).transform(X))
    assert_array_equal([[]], ft.fit_transform(X))
    assert_array_equal([], ft.get_feature_names_out())

    # check we can change back
    ft.set_params(m3=mult3)
    assert_array_equal([[3]], ft.fit(X).transform(X))

    # Check 'drop' step at construction time
    ft = FeatureUnion([("m2", "drop"), ("m3", mult3)])
    assert_array_equal([[3]], ft.fit(X).transform(X))
    assert_array_equal([[3]], ft.fit_transform(X))
    assert_array_equal(["m3__x3"], ft.get_feature_names_out())


def test_set_feature_union_passthrough():
    """Check the behaviour of setting a transformer to `"passthrough"`."""
    mult2 = Mult(2)
    mult3 = Mult(3)

    # We only test get_features_names_out, as get_feature_names is unsupported by
    # FunctionTransformer, and hence unsupported by FeatureUnion passthrough.
    mult2.get_feature_names_out = lambda input_features: ["x2"]
    mult3.get_feature_names_out = lambda input_features: ["x3"]

    X = np.asarray([[1]])

    ft = FeatureUnion([("m2", mult2), ("m3", mult3)])
    assert_array_equal([[2, 3]], ft.fit(X).transform(X))
    assert_array_equal([[2, 3]], ft.fit_transform(X))
    assert_array_equal(["m2__x2", "m3__x3"], ft.get_feature_names_out())

    ft.set_params(m2="passthrough")
    assert_array_equal([[1, 3]], ft.fit(X).transform(X))
    assert_array_equal([[1, 3]], ft.fit_transform(X))
    assert_array_equal(["m2__myfeat", "m3__x3"], ft.get_feature_names_out(["myfeat"]))

    ft.set_params(m3="passthrough")
    assert_array_equal([[1, 1]], ft.fit(X).transform(X))
    assert_array_equal([[1, 1]], ft.fit_transform(X))
    assert_array_equal(
        ["m2__myfeat", "m3__myfeat"], ft.get_feature_names_out(["myfeat"])
    )

    # check we can change back
    ft.set_params(m3=mult3)
    assert_array_equal([[1, 3]], ft.fit(X).transform(X))
    assert_array_equal([[1, 3]], ft.fit_transform(X))
    assert_array_equal(["m2__myfeat", "m3__x3"], ft.get_feature_names_out(["myfeat"]))

    # Check 'passthrough' step at construction time
    ft = FeatureUnion([("m2", "passthrough"), ("m3", mult3)])
    assert_array_equal([[1, 3]], ft.fit(X).transform(X))
    assert_array_equal([[1, 3]], ft.fit_transform(X))
    assert_array_equal(["m2__myfeat", "m3__x3"], ft.get_feature_names_out(["myfeat"]))

    X = iris.data
    columns = X.shape[1]
    pca = PCA(n_components=2, svd_solver="randomized", random_state=0)

    ft = FeatureUnion([("passthrough", "passthrough"), ("pca", pca)])
    assert_array_equal(X, ft.fit(X).transform(X)[:, :columns])
    assert_array_equal(X, ft.fit_transform(X)[:, :columns])
    assert_array_equal(
        [
            "passthrough__f0",
            "passthrough__f1",
            "passthrough__f2",
            "passthrough__f3",
            "pca__pca0",
            "pca__pca1",
        ],
        ft.get_feature_names_out(["f0", "f1", "f2", "f3"]),
    )

    ft.set_params(pca="passthrough")
    X_ft = ft.fit(X).transform(X)
    assert_array_equal(X_ft, np.hstack([X, X]))
    X_ft = ft.fit_transform(X)
    assert_array_equal(X_ft, np.hstack([X, X]))
    assert_array_equal(
        [
            "passthrough__f0",
            "passthrough__f1",
            "passthrough__f2",
            "passthrough__f3",
            "pca__f0",
            "pca__f1",
            "pca__f2",
            "pca__f3",
        ],
        ft.get_feature_names_out(["f0", "f1", "f2", "f3"]),
    )

    ft.set_params(passthrough=pca)
    assert_array_equal(X, ft.fit(X).transform(X)[:, -columns:])
    assert_array_equal(X, ft.fit_transform(X)[:, -columns:])
    assert_array_equal(
        [
            "passthrough__pca0",
            "passthrough__pca1",
            "pca__f0",
            "pca__f1",
            "pca__f2",
            "pca__f3",
        ],
        ft.get_feature_names_out(["f0", "f1", "f2", "f3"]),
    )

    ft = FeatureUnion(
        [("passthrough", "passthrough"), ("pca", pca)],
        transformer_weights={"passthrough": 2},
    )
    assert_array_equal(X * 2, ft.fit(X).transform(X)[:, :columns])
    assert_array_equal(X * 2, ft.fit_transform(X)[:, :columns])
    assert_array_equal(
        [
            "passthrough__f0",
            "passthrough__f1",
            "passthrough__f2",
            "passthrough__f3",
            "pca__pca0",
            "pca__pca1",
        ],
        ft.get_feature_names_out(["f0", "f1", "f2", "f3"]),
    )


def test_feature_union_passthrough_get_feature_names_out():
    """Check that get_feature_names_out works with passthrough without
    passing input_features.
    """
    X = iris.data
    pca = PCA(n_components=2, svd_solver="randomized", random_state=0)

    ft = FeatureUnion([("pca", pca), ("passthrough", "passthrough")])
    ft.fit(X)
    assert_array_equal(
        [
            "pca__pca0",
            "pca__pca1",
            "passthrough__x0",
            "passthrough__x1",
            "passthrough__x2",
            "passthrough__x3",
        ],
        ft.get_feature_names_out(),
    )


def test_step_name_validation():
    error_message_1 = r"Estimator names must not contain __: got \['a__q'\]"
    error_message_2 = r"Names provided are not unique: \['a', 'a'\]"
    error_message_3 = r"Estimator names conflict with constructor arguments: \['%s'\]"
    bad_steps1 = [("a__q", Mult(2)), ("b", Mult(3))]
    bad_steps2 = [("a", Mult(2)), ("a", Mult(3))]
    for cls, param in [(Pipeline, "steps"), (FeatureUnion, "transformer_list")]:
        # we validate in construction (despite scikit-learn convention)
        bad_steps3 = [("a", Mult(2)), (param, Mult(3))]
        for bad_steps, message in [
            (bad_steps1, error_message_1),
            (bad_steps2, error_message_2),
            (bad_steps3, error_message_3 % param),
        ]:
            # three ways to make invalid:
            # - construction
            with pytest.raises(ValueError, match=message):
                cls(**{param: bad_steps}).fit([[1]], [1])

            # - setattr
            est = cls(**{param: [("a", Mult(1))]})
            setattr(est, param, bad_steps)
            with pytest.raises(ValueError, match=message):
                est.fit([[1]], [1])

            with pytest.raises(ValueError, match=message):
                est.fit_transform([[1]], [1])

            # - set_params
            est = cls(**{param: [("a", Mult(1))]})
            est.set_params(**{param: bad_steps})
            with pytest.raises(ValueError, match=message):
                est.fit([[1]], [1])

            with pytest.raises(ValueError, match=message):
                est.fit_transform([[1]], [1])


def test_set_params_nested_pipeline():
    estimator = Pipeline([("a", Pipeline([("b", DummyRegressor())]))])
    estimator.set_params(a__b__alpha=0.001, a__b=Lasso())
    estimator.set_params(a__steps=[("b", LogisticRegression())], a__b__C=5)


def test_pipeline_memory():
    X = iris.data
    y = iris.target
    cachedir = mkdtemp()
    try:
        memory = joblib.Memory(location=cachedir, verbose=10)
        # Test with Transformer + SVC
        clf = SVC(probability=True, random_state=0)
        transf = DummyTransf()
        pipe = Pipeline([("transf", clone(transf)), ("svc", clf)])
        cached_pipe = Pipeline([("transf", transf), ("svc", clf)], memory=memory)

        # Memoize the transformer at the first fit
        cached_pipe.fit(X, y)
        pipe.fit(X, y)
        # Get the time stamp of the transformer in the cached pipeline
        ts = cached_pipe.named_steps["transf"].timestamp_
        # Check that cached_pipe and pipe yield identical results
        assert_array_equal(pipe.predict(X), cached_pipe.predict(X))
        assert_array_equal(pipe.predict_proba(X), cached_pipe.predict_proba(X))
        assert_array_equal(pipe.predict_log_proba(X), cached_pipe.predict_log_proba(X))
        assert_array_equal(pipe.score(X, y), cached_pipe.score(X, y))
        assert_array_equal(
            pipe.named_steps["transf"].means_, cached_pipe.named_steps["transf"].means_
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
            pipe.named_steps["transf"].means_, cached_pipe.named_steps["transf"].means_
        )
        assert ts == cached_pipe.named_steps["transf"].timestamp_
        # Create a new pipeline with cloned estimators
        # Check that even changing the name step does not affect the cache hit
        clf_2 = SVC(probability=True, random_state=0)
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
        assert ts == cached_pipe_2.named_steps["transf_2"].timestamp_
    finally:
        shutil.rmtree(cachedir)


def test_make_pipeline_memory():
    cachedir = mkdtemp()
    memory = joblib.Memory(location=cachedir, verbose=10)
    pipeline = make_pipeline(DummyTransf(), SVC(), memory=memory)
    assert pipeline.memory is memory
    pipeline = make_pipeline(DummyTransf(), SVC())
    assert pipeline.memory is None
    assert len(pipeline) == 2

    shutil.rmtree(cachedir)


class FeatureNameSaver(BaseEstimator):
    def fit(self, X, y=None):
        self._check_feature_names(X, reset=True)
        return self

    def transform(self, X, y=None):
        return X

    def get_feature_names_out(self, input_features=None):
        return input_features


def test_features_names_passthrough():
    """Check pipeline.get_feature_names_out with passthrough"""
    pipe = Pipeline(
        steps=[
            ("names", FeatureNameSaver()),
            ("pass", "passthrough"),
            ("clf", LogisticRegression()),
        ]
    )
    iris = load_iris()
    pipe.fit(iris.data, iris.target)
    assert_array_equal(
        pipe[:-1].get_feature_names_out(iris.feature_names), iris.feature_names
    )


def test_feature_names_count_vectorizer():
    """Check pipeline.get_feature_names_out with vectorizers"""
    pipe = Pipeline(steps=[("vect", CountVectorizer()), ("clf", LogisticRegression())])
    y = ["pizza" in x for x in JUNK_FOOD_DOCS]
    pipe.fit(JUNK_FOOD_DOCS, y)
    assert_array_equal(
        pipe[:-1].get_feature_names_out(),
        ["beer", "burger", "coke", "copyright", "pizza", "the"],
    )
    assert_array_equal(
        pipe[:-1].get_feature_names_out("nonsense_is_ignored"),
        ["beer", "burger", "coke", "copyright", "pizza", "the"],
    )


def test_pipeline_feature_names_out_error_without_definition():
    """Check that error is raised when a transformer does not define
    `get_feature_names_out`."""
    pipe = Pipeline(steps=[("notrans", NoTrans())])
    iris = load_iris()
    pipe.fit(iris.data, iris.target)

    msg = "does not provide get_feature_names_out"
    with pytest.raises(AttributeError, match=msg):
        pipe.get_feature_names_out()


def test_pipeline_param_error():
    clf = make_pipeline(LogisticRegression())
    with pytest.raises(
        ValueError, match="Pipeline.fit does not accept the sample_weight parameter"
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


def test_n_features_in_pipeline():
    # make sure pipelines delegate n_features_in to the first step

    X = [[1, 2], [3, 4], [5, 6]]
    y = [0, 1, 2]

    ss = StandardScaler()
    gbdt = HistGradientBoostingClassifier()
    pipe = make_pipeline(ss, gbdt)
    assert not hasattr(pipe, "n_features_in_")
    pipe.fit(X, y)
    assert pipe.n_features_in_ == ss.n_features_in_ == 2

    # if the first step has the n_features_in attribute then the pipeline also
    # has it, even though it isn't fitted.
    ss = StandardScaler()
    gbdt = HistGradientBoostingClassifier()
    pipe = make_pipeline(ss, gbdt)
    ss.fit(X, y)
    assert pipe.n_features_in_ == ss.n_features_in_ == 2
    assert not hasattr(gbdt, "n_features_in_")


def test_n_features_in_feature_union():
    # make sure FeatureUnion delegates n_features_in to the first transformer

    X = [[1, 2], [3, 4], [5, 6]]
    y = [0, 1, 2]

    ss = StandardScaler()
    fu = make_union(ss)
    assert not hasattr(fu, "n_features_in_")
    fu.fit(X, y)
    assert fu.n_features_in_ == ss.n_features_in_ == 2

    # if the first step has the n_features_in attribute then the feature_union
    # also has it, even though it isn't fitted.
    ss = StandardScaler()
    fu = make_union(ss)
    ss.fit(X, y)
    assert fu.n_features_in_ == ss.n_features_in_ == 2


def test_feature_union_fit_params():
    # Regression test for issue: #15117
    class Dummy(TransformerMixin, BaseEstimator):
        def fit(self, X, y=None, **fit_params):
            if fit_params != {"a": 0}:
                raise ValueError
            return self

        def transform(self, X, y=None):
            return X

    X, y = iris.data, iris.target
    t = FeatureUnion([("dummy0", Dummy()), ("dummy1", Dummy())])
    with pytest.raises(ValueError):
        t.fit(X, y)

    with pytest.raises(ValueError):
        t.fit_transform(X, y)

    t.fit(X, y, a=0)
    t.fit_transform(X, y, a=0)


def test_pipeline_missing_values_leniency():
    # check that pipeline let the missing values validation to
    # the underlying transformers and predictors.
    X, y = iris.data, iris.target
    mask = np.random.choice([1, 0], X.shape, p=[0.1, 0.9]).astype(bool)
    X[mask] = np.nan
    pipe = make_pipeline(SimpleImputer(), LogisticRegression())
    assert pipe.fit(X, y).score(X, y) > 0.4


def test_feature_union_warns_unknown_transformer_weight():
    # Warn user when transformer_weights containers a key not present in
    # transformer_list
    X = [[1, 2], [3, 4], [5, 6]]
    y = [0, 1, 2]

    transformer_list = [("transf", Transf())]
    # Transformer weights dictionary with incorrect name
    weights = {"transformer": 1}
    expected_msg = (
        'Attempting to weight transformer "transformer", '
        "but it is not present in transformer_list."
    )
    union = FeatureUnion(transformer_list, transformer_weights=weights)
    with pytest.raises(ValueError, match=expected_msg):
        union.fit(X, y)


@pytest.mark.parametrize("passthrough", [None, "passthrough"])
def test_pipeline_get_tags_none(passthrough):
    # Checks that tags are set correctly when the first transformer is None or
    # 'passthrough'
    # Non-regression test for:
    # https://github.com/scikit-learn/scikit-learn/issues/18815
    pipe = make_pipeline(passthrough, SVC())
    assert not pipe._get_tags()["pairwise"]


# FIXME: Replace this test with a full `check_estimator` once we have API only
# checks.
@pytest.mark.parametrize("Predictor", [MinimalRegressor, MinimalClassifier])
def test_search_cv_using_minimal_compatible_estimator(Predictor):
    # Check that third-party library estimators can be part of a pipeline
    # and tuned by grid-search without inheriting from BaseEstimator.
    rng = np.random.RandomState(0)
    X, y = rng.randn(25, 2), np.array([0] * 5 + [1] * 20)

    model = Pipeline(
        [("transformer", MinimalTransformer()), ("predictor", Predictor())]
    )
    model.fit(X, y)

    y_pred = model.predict(X)
    if is_classifier(model):
        assert_array_equal(y_pred, 1)
        assert model.score(X, y) == pytest.approx(accuracy_score(y, y_pred))
    else:
        assert_allclose(y_pred, y.mean())
        assert model.score(X, y) == pytest.approx(r2_score(y, y_pred))


def test_pipeline_check_if_fitted():
    class Estimator(BaseEstimator):
        def fit(self, X, y):
            self.fitted_ = True
            return self

    pipeline = Pipeline([("clf", Estimator())])
    with pytest.raises(NotFittedError):
        check_is_fitted(pipeline)
    pipeline.fit(iris.data, iris.target)
    check_is_fitted(pipeline)


def test_feature_union_check_if_fitted():
    """Check __sklearn_is_fitted__ is defined correctly."""

    X = [[1, 2], [3, 4], [5, 6]]
    y = [0, 1, 2]

    union = FeatureUnion([("clf", MinimalTransformer())])
    with pytest.raises(NotFittedError):
        check_is_fitted(union)

    union.fit(X, y)
    check_is_fitted(union)

    # passthrough is stateless
    union = FeatureUnion([("pass", "passthrough")])
    check_is_fitted(union)

    union = FeatureUnion([("clf", MinimalTransformer()), ("pass", "passthrough")])
    with pytest.raises(NotFittedError):
        check_is_fitted(union)

    union.fit(X, y)
    check_is_fitted(union)


def test_pipeline_get_feature_names_out_passes_names_through():
    """Check that pipeline passes names through.

    Non-regresion test for #21349.
    """
    X, y = iris.data, iris.target

    class AddPrefixStandardScalar(StandardScaler):
        def get_feature_names_out(self, input_features=None):
            names = super().get_feature_names_out(input_features=input_features)
            return np.asarray([f"my_prefix_{name}" for name in names], dtype=object)

    pipe = make_pipeline(AddPrefixStandardScalar(), StandardScaler())
    pipe.fit(X, y)

    input_names = iris.feature_names
    feature_names_out = pipe.get_feature_names_out(input_names)

    assert_array_equal(feature_names_out, [f"my_prefix_{name}" for name in input_names])


def test_pipeline_set_output_integration():
    """Test pipeline's set_output with feature names."""
    pytest.importorskip("pandas")

    X, y = load_iris(as_frame=True, return_X_y=True)

    pipe = make_pipeline(StandardScaler(), LogisticRegression())
    pipe.set_output(transform="pandas")
    pipe.fit(X, y)

    feature_names_in_ = pipe[:-1].get_feature_names_out()
    log_reg_feature_names = pipe[-1].feature_names_in_

    assert_array_equal(feature_names_in_, log_reg_feature_names)


def test_feature_union_set_output():
    """Test feature union with set_output API."""
    pd = pytest.importorskip("pandas")

    X, _ = load_iris(as_frame=True, return_X_y=True)
    X_train, X_test = train_test_split(X, random_state=0)
    union = FeatureUnion([("scalar", StandardScaler()), ("pca", PCA())])
    union.set_output(transform="pandas")
    union.fit(X_train)

    X_trans = union.transform(X_test)
    assert isinstance(X_trans, pd.DataFrame)
    assert_array_equal(X_trans.columns, union.get_feature_names_out())
    assert_array_equal(X_trans.index, X_test.index)


def test_feature_union_getitem():
    """Check FeatureUnion.__getitem__ returns expected results."""
    scalar = StandardScaler()
    pca = PCA()
    union = FeatureUnion(
        [
            ("scalar", scalar),
            ("pca", pca),
            ("pass", "passthrough"),
            ("drop_me", "drop"),
        ]
    )
    assert union["scalar"] is scalar
    assert union["pca"] is pca
    assert union["pass"] == "passthrough"
    assert union["drop_me"] == "drop"


@pytest.mark.parametrize("key", [0, slice(0, 2)])
def test_feature_union_getitem_error(key):
    """Raise error when __getitem__ gets a non-string input."""

    union = FeatureUnion([("scalar", StandardScaler()), ("pca", PCA())])

    msg = "Only string keys are supported"
    with pytest.raises(KeyError, match=msg):
        union[key]


def test_feature_union_feature_names_in_():
    """Ensure feature union has `.feature_names_in_` attribute if `X` has a
    `columns` attribute.

    Test for #24754.
    """
    pytest.importorskip("pandas")

    X, _ = load_iris(as_frame=True, return_X_y=True)

    # FeatureUnion should have the feature_names_in_ attribute if the
    # first transformer also has it
    scaler = StandardScaler()
    scaler.fit(X)
    union = FeatureUnion([("scale", scaler)])
    assert hasattr(union, "feature_names_in_")
    assert_array_equal(X.columns, union.feature_names_in_)
    assert_array_equal(scaler.feature_names_in_, union.feature_names_in_)

    # fit with pandas.DataFrame
    union = FeatureUnion([("pass", "passthrough")])
    union.fit(X)
    assert hasattr(union, "feature_names_in_")
    assert_array_equal(X.columns, union.feature_names_in_)

    # fit with numpy array
    X_array = X.to_numpy()
    union = FeatureUnion([("pass", "passthrough")])
    union.fit(X_array)
    assert not hasattr(union, "feature_names_in_")
