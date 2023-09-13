"""
Metadata Routing Utility Tests
"""

# Author: Adrin Jalali <adrin.jalali@gmail.com>
# License: BSD 3 clause

import re

import numpy as np
import pytest

from sklearn import config_context
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    MetaEstimatorMixin,
    RegressorMixin,
    TransformerMixin,
    clone,
)
from sklearn.linear_model import LinearRegression
from sklearn.utils import metadata_routing
from sklearn.utils._metadata_requests import (
    METHODS,
    MethodMetadataRequest,
    _MetadataRequester,
    request_is_alias,
    request_is_valid,
)
from sklearn.utils.metadata_routing import (
    MetadataRequest,
    MetadataRouter,
    MethodMapping,
    get_routing_for_object,
    process_routing,
)
from sklearn.utils.validation import check_is_fitted

rng = np.random.RandomState(42)
N, M = 100, 4
X = rng.rand(N, M)
y = rng.randint(0, 2, size=N)
my_groups = rng.randint(0, 10, size=N)
my_weights = rng.rand(N)
my_other_weights = rng.rand(N)


@pytest.fixture(autouse=True)
def enable_slep006():
    """Enable SLEP006 for all tests."""
    with config_context(enable_metadata_routing=True):
        yield


def assert_request_is_empty(metadata_request, exclude=None):
    """Check if a metadata request dict is empty.

    One can exclude a method or a list of methods from the check using the
    ``exclude`` parameter.
    """
    if isinstance(metadata_request, MetadataRouter):
        for _, route_mapping in metadata_request:
            assert_request_is_empty(route_mapping.router)
        return

    exclude = [] if exclude is None else exclude
    for method in METHODS:
        if method in exclude:
            continue
        mmr = getattr(metadata_request, method)
        props = [
            prop
            for prop, alias in mmr.requests.items()
            if isinstance(alias, str) or alias is not None
        ]
        assert not len(props)


def assert_request_equal(request, dictionary):
    for method, requests in dictionary.items():
        mmr = getattr(request, method)
        assert mmr.requests == requests

    empty_methods = [method for method in METHODS if method not in dictionary]
    for method in empty_methods:
        assert not len(getattr(request, method).requests)


def record_metadata(obj, method, record_default=True, **kwargs):
    """Utility function to store passed metadata to a method.

    If record_default is False, kwargs whose values are "default" are skipped.
    This is so that checks on keyword arguments whose default was not changed
    are skipped.

    """
    if not hasattr(obj, "_records"):
        obj._records = {}
    if not record_default:
        kwargs = {
            key: val
            for key, val in kwargs.items()
            if not isinstance(val, str) or (val != "default")
        }
    obj._records[method] = kwargs


def check_recorded_metadata(obj, method, **kwargs):
    """Check whether the expected metadata is passed to the object's method."""
    records = getattr(obj, "_records", dict()).get(method, dict())
    assert set(kwargs.keys()) == set(records.keys())
    for key, value in kwargs.items():
        assert records[key] is value


class MetaRegressor(MetaEstimatorMixin, RegressorMixin, BaseEstimator):
    """A meta-regressor which is only a router."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, **fit_params):
        params = process_routing(self, "fit", fit_params)
        self.estimator_ = clone(self.estimator).fit(X, y, **params.estimator.fit)

    def get_metadata_routing(self):
        router = MetadataRouter(owner=self.__class__.__name__).add(
            estimator=self.estimator, method_mapping="one-to-one"
        )
        return router


class RegressorMetadata(RegressorMixin, BaseEstimator):
    """A regressor consuming a metadata."""

    def fit(self, X, y, sample_weight=None):
        record_metadata(self, "fit", sample_weight=sample_weight)
        return self

    def predict(self, X):
        return np.zeros(shape=(len(X)))


class WeightedMetaRegressor(MetaEstimatorMixin, RegressorMixin, BaseEstimator):
    """A meta-regressor which is also a consumer."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, sample_weight=None, **fit_params):
        record_metadata(self, "fit", sample_weight=sample_weight)
        params = process_routing(self, "fit", fit_params, sample_weight=sample_weight)
        self.estimator_ = clone(self.estimator).fit(X, y, **params.estimator.fit)
        return self

    def predict(self, X, **predict_params):
        params = process_routing(self, "predict", predict_params)
        return self.estimator_.predict(X, **params.estimator.predict)

    def get_metadata_routing(self):
        router = (
            MetadataRouter(owner=self.__class__.__name__)
            .add_self_request(self)
            .add(estimator=self.estimator, method_mapping="one-to-one")
        )
        return router


class ClassifierNoMetadata(ClassifierMixin, BaseEstimator):
    """An estimator which accepts no metadata on any method."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.ones(len(X))  # pragma: no cover


class ClassifierFitMetadata(ClassifierMixin, BaseEstimator):
    """An estimator accepting two metadata in its ``fit`` method."""

    def fit(self, X, y, sample_weight=None, brand=None):
        record_metadata(self, "fit", sample_weight=sample_weight, brand=brand)
        return self

    def predict(self, X):
        return np.ones(len(X))  # pragma: no cover


class SimpleMetaClassifier(MetaEstimatorMixin, ClassifierMixin, BaseEstimator):
    """A meta-estimator which also consumes sample_weight itself in ``fit``."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, sample_weight=None, **kwargs):
        record_metadata(self, "fit", sample_weight=sample_weight)
        params = process_routing(self, "fit", kwargs, sample_weight=sample_weight)
        self.estimator_ = clone(self.estimator).fit(X, y, **params.estimator.fit)
        return self

    def get_metadata_routing(self):
        router = (
            MetadataRouter(owner=self.__class__.__name__)
            .add_self_request(self)
            .add(estimator=self.estimator, method_mapping="fit")
        )
        return router


class TransformerMetadata(TransformerMixin, BaseEstimator):
    """A transformer which accepts metadata on fit and transform."""

    def fit(self, X, y=None, brand=None, sample_weight=None):
        record_metadata(self, "fit", brand=brand, sample_weight=sample_weight)
        return self

    def transform(self, X, sample_weight=None):
        record_metadata(self, "transform", sample_weight=sample_weight)
        return X


class MetaTransformer(MetaEstimatorMixin, TransformerMixin, BaseEstimator):
    """A simple meta-transformer."""

    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self, X, y=None, **fit_params):
        params = process_routing(self, "fit", fit_params)
        self.transformer_ = clone(self.transformer).fit(X, y, **params.transformer.fit)
        return self

    def transform(self, X, y=None, **transform_params):
        params = process_routing(self, "transform", transform_params)
        return self.transformer_.transform(X, **params.transformer.transform)

    def get_metadata_routing(self):
        return MetadataRouter(owner=self.__class__.__name__).add(
            transformer=self.transformer, method_mapping="one-to-one"
        )


class SimplePipeline(BaseEstimator):
    """A very simple pipeline, assuming the last step is always a predictor."""

    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, y, **fit_params):
        self.steps_ = []
        params = process_routing(self, "fit", fit_params)
        X_transformed = X
        for i, step in enumerate(self.steps[:-1]):
            transformer = clone(step).fit(
                X_transformed, y, **params.get(f"step_{i}").fit
            )
            self.steps_.append(transformer)
            X_transformed = transformer.transform(
                X_transformed, **params.get(f"step_{i}").transform
            )

        self.steps_.append(
            clone(self.steps[-1]).fit(X_transformed, y, **params.predictor.fit)
        )
        return self

    def predict(self, X, **predict_params):
        check_is_fitted(self)
        X_transformed = X
        params = process_routing(self, "predict", predict_params)
        for i, step in enumerate(self.steps_[:-1]):
            X_transformed = step.transform(X, **params.get(f"step_{i}").transform)

        return self.steps_[-1].predict(X_transformed, **params.predictor.predict)

    def get_metadata_routing(self):
        router = MetadataRouter(owner=self.__class__.__name__)
        for i, step in enumerate(self.steps[:-1]):
            router.add(
                **{f"step_{i}": step},
                method_mapping=MethodMapping()
                .add(callee="fit", caller="fit")
                .add(callee="transform", caller="fit")
                .add(callee="transform", caller="predict"),
            )
        router.add(predictor=self.steps[-1], method_mapping="one-to-one")
        return router


def test_assert_request_is_empty():
    requests = MetadataRequest(owner="test")
    assert_request_is_empty(requests)

    requests.fit.add_request(param="foo", alias=None)
    # this should still work, since None is the default value
    assert_request_is_empty(requests)

    requests.fit.add_request(param="bar", alias="value")
    with pytest.raises(AssertionError):
        # now requests is no more empty
        assert_request_is_empty(requests)

    # but one can exclude a method
    assert_request_is_empty(requests, exclude="fit")

    requests.score.add_request(param="carrot", alias=True)
    with pytest.raises(AssertionError):
        # excluding `fit` is not enough
        assert_request_is_empty(requests, exclude="fit")

    # and excluding both fit and score would avoid an exception
    assert_request_is_empty(requests, exclude=["fit", "score"])

    # test if a router is empty
    assert_request_is_empty(
        MetadataRouter(owner="test")
        .add_self_request(WeightedMetaRegressor(estimator=None))
        .add(method_mapping="fit", estimator=RegressorMetadata())
    )


@pytest.mark.parametrize(
    "val, res",
    [
        (False, False),
        (True, False),
        (None, False),
        ("$UNUSED$", False),
        ("$WARN$", False),
        ("invalid-input", False),
        ("valid_arg", True),
    ],
)
def test_request_type_is_alias(val, res):
    # Test request_is_alias
    assert request_is_alias(val) == res


@pytest.mark.parametrize(
    "val, res",
    [
        (False, True),
        (True, True),
        (None, True),
        ("$UNUSED$", True),
        ("$WARN$", True),
        ("invalid-input", False),
        ("alias_arg", False),
    ],
)
def test_request_type_is_valid(val, res):
    # Test request_is_valid
    assert request_is_valid(val) == res


def test_default_requests():
    class OddEstimator(BaseEstimator):
        __metadata_request__fit = {
            # set a different default request
            "sample_weight": True
        }  # type: ignore

    odd_request = get_routing_for_object(OddEstimator())
    assert odd_request.fit.requests == {"sample_weight": True}

    # check other test estimators
    assert not len(get_routing_for_object(ClassifierNoMetadata()).fit.requests)
    assert_request_is_empty(ClassifierNoMetadata().get_metadata_routing())

    trs_request = get_routing_for_object(TransformerMetadata())
    assert trs_request.fit.requests == {
        "sample_weight": None,
        "brand": None,
    }
    assert trs_request.transform.requests == {
        "sample_weight": None,
    }
    assert_request_is_empty(trs_request)

    est_request = get_routing_for_object(ClassifierFitMetadata())
    assert est_request.fit.requests == {
        "sample_weight": None,
        "brand": None,
    }
    assert_request_is_empty(est_request)


def test_process_routing_invalid_method():
    with pytest.raises(TypeError, match="Can only route and process input"):
        process_routing(ClassifierFitMetadata(), "invalid_method", {})


def test_process_routing_invalid_object():
    class InvalidObject:
        pass

    with pytest.raises(AttributeError, match="has not implemented the routing"):
        process_routing(InvalidObject(), "fit", {})


def test_simple_metadata_routing():
    # Tests that metadata is properly routed

    # The underlying estimator doesn't accept or request metadata
    clf = SimpleMetaClassifier(estimator=ClassifierNoMetadata())
    clf.fit(X, y)

    # Meta-estimator consumes sample_weight, but doesn't forward it to the underlying
    # estimator
    clf = SimpleMetaClassifier(estimator=ClassifierNoMetadata())
    clf.fit(X, y, sample_weight=my_weights)

    # If the estimator accepts the metadata but doesn't explicitly say it doesn't
    # need it, there's an error
    clf = SimpleMetaClassifier(estimator=ClassifierFitMetadata())
    err_message = (
        "[sample_weight] are passed but are not explicitly set as requested or"
        " not for ClassifierFitMetadata.fit"
    )
    with pytest.raises(ValueError, match=re.escape(err_message)):
        clf.fit(X, y, sample_weight=my_weights)

    # Explicitly saying the estimator doesn't need it, makes the error go away,
    # because in this case `SimpleMetaClassifier` consumes `sample_weight`. If
    # there was no consumer of sample_weight, passing it would result in an
    # error.
    clf = SimpleMetaClassifier(
        estimator=ClassifierFitMetadata().set_fit_request(sample_weight=False)
    )
    # this doesn't raise since SimpleMetaClassifier itself is a consumer,
    # and passing metadata to the consumer directly is fine regardless of its
    # metadata_request values.
    clf.fit(X, y, sample_weight=my_weights)
    check_recorded_metadata(clf.estimator_, "fit", sample_weight=None, brand=None)

    # Requesting a metadata will make the meta-estimator forward it correctly
    clf = SimpleMetaClassifier(
        estimator=ClassifierFitMetadata().set_fit_request(sample_weight=True)
    )
    clf.fit(X, y, sample_weight=my_weights)
    check_recorded_metadata(clf.estimator_, "fit", sample_weight=my_weights, brand=None)

    # And requesting it with an alias
    clf = SimpleMetaClassifier(
        estimator=ClassifierFitMetadata().set_fit_request(
            sample_weight="alternative_weight"
        )
    )
    clf.fit(X, y, alternative_weight=my_weights)
    check_recorded_metadata(clf.estimator_, "fit", sample_weight=my_weights, brand=None)


def test_nested_routing():
    # check if metadata is routed in a nested routing situation.
    pipeline = SimplePipeline(
        [
            MetaTransformer(
                transformer=TransformerMetadata()
                .set_fit_request(brand=True, sample_weight=False)
                .set_transform_request(sample_weight=True)
            ),
            WeightedMetaRegressor(
                estimator=RegressorMetadata().set_fit_request(
                    sample_weight="inner_weights"
                )
            ).set_fit_request(sample_weight="outer_weights"),
        ]
    )
    w1, w2, w3 = [1], [2], [3]
    pipeline.fit(
        X, y, brand=my_groups, sample_weight=w1, outer_weights=w2, inner_weights=w3
    )
    check_recorded_metadata(
        pipeline.steps_[0].transformer_, "fit", brand=my_groups, sample_weight=None
    )
    check_recorded_metadata(
        pipeline.steps_[0].transformer_, "transform", sample_weight=w1
    )
    check_recorded_metadata(pipeline.steps_[1], "fit", sample_weight=w2)
    check_recorded_metadata(pipeline.steps_[1].estimator_, "fit", sample_weight=w3)

    pipeline.predict(X, sample_weight=w3)
    check_recorded_metadata(
        pipeline.steps_[0].transformer_, "transform", sample_weight=w3
    )


def test_nested_routing_conflict():
    # check if an error is raised if there's a conflict between keys
    pipeline = SimplePipeline(
        [
            MetaTransformer(
                transformer=TransformerMetadata()
                .set_fit_request(brand=True, sample_weight=False)
                .set_transform_request(sample_weight=True)
            ),
            WeightedMetaRegressor(
                estimator=RegressorMetadata().set_fit_request(sample_weight=True)
            ).set_fit_request(sample_weight="outer_weights"),
        ]
    )
    w1, w2 = [1], [2]
    with pytest.raises(
        ValueError,
        match=(
            re.escape(
                "In WeightedMetaRegressor, there is a conflict on sample_weight between"
                " what is requested for this estimator and what is requested by its"
                " children. You can resolve this conflict by using an alias for the"
                " child estimator(s) requested metadata."
            )
        ),
    ):
        pipeline.fit(X, y, brand=my_groups, sample_weight=w1, outer_weights=w2)


def test_invalid_metadata():
    # check that passing wrong metadata raises an error
    trs = MetaTransformer(
        transformer=TransformerMetadata().set_transform_request(sample_weight=True)
    )
    with pytest.raises(
        TypeError,
        match=(re.escape("transform got unexpected argument(s) {'other_param'}")),
    ):
        trs.fit(X, y).transform(X, other_param=my_weights)

    # passing a metadata which is not requested by any estimator should also raise
    trs = MetaTransformer(
        transformer=TransformerMetadata().set_transform_request(sample_weight=False)
    )
    with pytest.raises(
        TypeError,
        match=(re.escape("transform got unexpected argument(s) {'sample_weight'}")),
    ):
        trs.fit(X, y).transform(X, sample_weight=my_weights)


def test_get_metadata_routing():
    class TestDefaultsBadMethodName(_MetadataRequester):
        __metadata_request__fit = {
            "sample_weight": None,
            "my_param": None,
        }
        __metadata_request__score = {
            "sample_weight": None,
            "my_param": True,
            "my_other_param": None,
        }
        # this will raise an error since we don't understand "other_method" as a method
        __metadata_request__other_method = {"my_param": True}

    class TestDefaults(_MetadataRequester):
        __metadata_request__fit = {
            "sample_weight": None,
            "my_other_param": None,
        }
        __metadata_request__score = {
            "sample_weight": None,
            "my_param": True,
            "my_other_param": None,
        }
        __metadata_request__predict = {"my_param": True}

    with pytest.raises(
        AttributeError, match="'MetadataRequest' object has no attribute 'other_method'"
    ):
        TestDefaultsBadMethodName().get_metadata_routing()

    expected = {
        "score": {
            "my_param": True,
            "my_other_param": None,
            "sample_weight": None,
        },
        "fit": {
            "my_other_param": None,
            "sample_weight": None,
        },
        "predict": {"my_param": True},
    }
    assert_request_equal(TestDefaults().get_metadata_routing(), expected)

    est = TestDefaults().set_score_request(my_param="other_param")
    expected = {
        "score": {
            "my_param": "other_param",
            "my_other_param": None,
            "sample_weight": None,
        },
        "fit": {
            "my_other_param": None,
            "sample_weight": None,
        },
        "predict": {"my_param": True},
    }
    assert_request_equal(est.get_metadata_routing(), expected)

    est = TestDefaults().set_fit_request(sample_weight=True)
    expected = {
        "score": {
            "my_param": True,
            "my_other_param": None,
            "sample_weight": None,
        },
        "fit": {
            "my_other_param": None,
            "sample_weight": True,
        },
        "predict": {"my_param": True},
    }
    assert_request_equal(est.get_metadata_routing(), expected)


def test_setting_default_requests():
    # Test _get_default_requests method
    test_cases = dict()

    class ExplicitRequest(BaseEstimator):
        # `fit` doesn't accept `props` explicitly, but we want to request it
        __metadata_request__fit = {"prop": None}

        def fit(self, X, y, **kwargs):
            return self

    test_cases[ExplicitRequest] = {"prop": None}

    class ExplicitRequestOverwrite(BaseEstimator):
        # `fit` explicitly accepts `props`, but we want to change the default
        # request value from None to True
        __metadata_request__fit = {"prop": True}

        def fit(self, X, y, prop=None, **kwargs):
            return self

    test_cases[ExplicitRequestOverwrite] = {"prop": True}

    class ImplicitRequest(BaseEstimator):
        # `fit` requests `prop` and the default None should be used
        def fit(self, X, y, prop=None, **kwargs):
            return self

    test_cases[ImplicitRequest] = {"prop": None}

    class ImplicitRequestRemoval(BaseEstimator):
        # `fit` (in this class or a parent) requests `prop`, but we don't want
        # it requested at all.
        __metadata_request__fit = {"prop": metadata_routing.UNUSED}

        def fit(self, X, y, prop=None, **kwargs):
            return self

    test_cases[ImplicitRequestRemoval] = {}

    for Klass, requests in test_cases.items():
        assert get_routing_for_object(Klass()).fit.requests == requests
        assert_request_is_empty(Klass().get_metadata_routing(), exclude="fit")
        Klass().fit(None, None)  # for coverage


def test_removing_non_existing_param_raises():
    """Test that removing a metadata using UNUSED which doesn't exist raises."""

    class InvalidRequestRemoval(BaseEstimator):
        # `fit` (in this class or a parent) requests `prop`, but we don't want
        # it requested at all.
        __metadata_request__fit = {"prop": metadata_routing.UNUSED}

        def fit(self, X, y, **kwargs):
            return self

    with pytest.raises(ValueError, match="Trying to remove parameter"):
        InvalidRequestRemoval().get_metadata_routing()


def test_method_metadata_request():
    mmr = MethodMetadataRequest(owner="test", method="fit")

    with pytest.raises(ValueError, match="The alias you're setting for"):
        mmr.add_request(param="foo", alias=1.4)

    mmr.add_request(param="foo", alias=None)
    assert mmr.requests == {"foo": None}
    mmr.add_request(param="foo", alias=False)
    assert mmr.requests == {"foo": False}
    mmr.add_request(param="foo", alias=True)
    assert mmr.requests == {"foo": True}
    mmr.add_request(param="foo", alias="foo")
    assert mmr.requests == {"foo": True}
    mmr.add_request(param="foo", alias="bar")
    assert mmr.requests == {"foo": "bar"}
    assert mmr._get_param_names(return_alias=False) == {"foo"}
    assert mmr._get_param_names(return_alias=True) == {"bar"}


def test_get_routing_for_object():
    class Consumer(BaseEstimator):
        __metadata_request__fit = {"prop": None}

    assert_request_is_empty(get_routing_for_object(None))
    assert_request_is_empty(get_routing_for_object(object()))

    mr = MetadataRequest(owner="test")
    mr.fit.add_request(param="foo", alias="bar")
    mr_factory = get_routing_for_object(mr)
    assert_request_is_empty(mr_factory, exclude="fit")
    assert mr_factory.fit.requests == {"foo": "bar"}

    mr = get_routing_for_object(Consumer())
    assert_request_is_empty(mr, exclude="fit")
    assert mr.fit.requests == {"prop": None}


def test_metaestimator_warnings():
    class WeightedMetaRegressorWarn(WeightedMetaRegressor):
        __metadata_request__fit = {"sample_weight": metadata_routing.WARN}

    with pytest.warns(
        UserWarning, match="Support for .* has recently been added to this class"
    ):
        WeightedMetaRegressorWarn(
            estimator=LinearRegression().set_fit_request(sample_weight=False)
        ).fit(X, y, sample_weight=my_weights)


def test_estimator_warnings():
    class RegressorMetadataWarn(RegressorMetadata):
        __metadata_request__fit = {"sample_weight": metadata_routing.WARN}

    with pytest.warns(
        UserWarning, match="Support for .* has recently been added to this class"
    ):
        MetaRegressor(estimator=RegressorMetadataWarn()).fit(
            X, y, sample_weight=my_weights
        )


@pytest.mark.parametrize(
    "obj, string",
    [
        (
            MethodMetadataRequest(owner="test", method="fit").add_request(
                param="foo", alias="bar"
            ),
            "{'foo': 'bar'}",
        ),
        (
            MetadataRequest(owner="test"),
            "{}",
        ),
        (MethodMapping.from_str("score"), "[{'callee': 'score', 'caller': 'score'}]"),
        (
            MetadataRouter(owner="test").add(
                method_mapping="predict", estimator=RegressorMetadata()
            ),
            (
                "{'estimator': {'mapping': [{'callee': 'predict', 'caller': "
                "'predict'}], 'router': {'fit': {'sample_weight': None}, "
                "'score': {'sample_weight': None}}}}"
            ),
        ),
    ],
)
def test_string_representations(obj, string):
    assert str(obj) == string


@pytest.mark.parametrize(
    "obj, method, inputs, err_cls, err_msg",
    [
        (
            MethodMapping(),
            "add",
            {"callee": "invalid", "caller": "fit"},
            ValueError,
            "Given callee",
        ),
        (
            MethodMapping(),
            "add",
            {"callee": "fit", "caller": "invalid"},
            ValueError,
            "Given caller",
        ),
        (
            MethodMapping,
            "from_str",
            {"route": "invalid"},
            ValueError,
            "route should be 'one-to-one' or a single method!",
        ),
        (
            MetadataRouter(owner="test"),
            "add_self_request",
            {"obj": MetadataRouter(owner="test")},
            ValueError,
            "Given `obj` is neither a `MetadataRequest` nor does it implement",
        ),
        (
            ClassifierFitMetadata(),
            "set_fit_request",
            {"invalid": True},
            TypeError,
            "Unexpected args",
        ),
    ],
)
def test_validations(obj, method, inputs, err_cls, err_msg):
    with pytest.raises(err_cls, match=err_msg):
        getattr(obj, method)(**inputs)


def test_methodmapping():
    mm = (
        MethodMapping()
        .add(caller="fit", callee="transform")
        .add(caller="fit", callee="fit")
    )

    mm_list = list(mm)
    assert mm_list[0] == ("transform", "fit")
    assert mm_list[1] == ("fit", "fit")

    mm = MethodMapping.from_str("one-to-one")
    assert (
        str(mm)
        == "[{'callee': 'fit', 'caller': 'fit'}, {'callee': 'partial_fit', 'caller':"
        " 'partial_fit'}, {'callee': 'predict', 'caller': 'predict'}, {'callee':"
        " 'predict_proba', 'caller': 'predict_proba'}, {'callee':"
        " 'predict_log_proba', 'caller': 'predict_log_proba'}, {'callee':"
        " 'decision_function', 'caller': 'decision_function'}, {'callee': 'score',"
        " 'caller': 'score'}, {'callee': 'split', 'caller': 'split'}, {'callee':"
        " 'transform', 'caller': 'transform'}, {'callee': 'inverse_transform',"
        " 'caller': 'inverse_transform'}]"
    )

    mm = MethodMapping.from_str("score")
    assert repr(mm) == "[{'callee': 'score', 'caller': 'score'}]"


def test_metadatarouter_add_self_request():
    # adding a MetadataRequest as `self` adds a copy
    request = MetadataRequest(owner="nested")
    request.fit.add_request(param="param", alias=True)
    router = MetadataRouter(owner="test").add_self_request(request)
    assert str(router._self_request) == str(request)
    # should be a copy, not the same object
    assert router._self_request is not request

    # one can add an estimator as self
    est = RegressorMetadata().set_fit_request(sample_weight="my_weights")
    router = MetadataRouter(owner="test").add_self_request(obj=est)
    assert str(router._self_request) == str(est.get_metadata_routing())
    assert router._self_request is not est.get_metadata_routing()

    # adding a consumer+router as self should only add the consumer part
    est = WeightedMetaRegressor(
        estimator=RegressorMetadata().set_fit_request(sample_weight="nested_weights")
    )
    router = MetadataRouter(owner="test").add_self_request(obj=est)
    # _get_metadata_request() returns the consumer part of the requests
    assert str(router._self_request) == str(est._get_metadata_request())
    # get_metadata_routing() returns the complete request set, consumer and
    # router included.
    assert str(router._self_request) != str(est.get_metadata_routing())
    # it should be a copy, not the same object
    assert router._self_request is not est._get_metadata_request()


def test_metadata_routing_add():
    # adding one with a string `method_mapping`
    router = MetadataRouter(owner="test").add(
        method_mapping="fit",
        est=RegressorMetadata().set_fit_request(sample_weight="weights"),
    )
    assert (
        str(router)
        == "{'est': {'mapping': [{'callee': 'fit', 'caller': 'fit'}], "
        "'router': {'fit': {'sample_weight': 'weights'}, 'score': "
        "{'sample_weight': None}}}}"
    )

    # adding one with an instance of MethodMapping
    router = MetadataRouter(owner="test").add(
        method_mapping=MethodMapping().add(callee="score", caller="fit"),
        est=RegressorMetadata().set_score_request(sample_weight=True),
    )
    assert (
        str(router)
        == "{'est': {'mapping': [{'callee': 'score', 'caller': 'fit'}], "
        "'router': {'fit': {'sample_weight': None}, 'score': "
        "{'sample_weight': True}}}}"
    )


def test_metadata_routing_get_param_names():
    router = (
        MetadataRouter(owner="test")
        .add_self_request(
            WeightedMetaRegressor(estimator=RegressorMetadata()).set_fit_request(
                sample_weight="self_weights"
            )
        )
        .add(
            method_mapping="fit",
            trs=TransformerMetadata().set_fit_request(
                sample_weight="transform_weights"
            ),
        )
    )

    assert (
        str(router)
        == "{'$self_request': {'fit': {'sample_weight': 'self_weights'}, 'score': "
        "{'sample_weight': None}}, 'trs': {'mapping': [{'callee': 'fit', "
        "'caller': 'fit'}], 'router': {'fit': {'brand': None, "
        "'sample_weight': 'transform_weights'}, 'transform': "
        "{'sample_weight': None}}}}"
    )

    assert router._get_param_names(
        method="fit", return_alias=True, ignore_self_request=False
    ) == {"transform_weights", "brand", "self_weights"}
    # return_alias=False will return original names for "self"
    assert router._get_param_names(
        method="fit", return_alias=False, ignore_self_request=False
    ) == {"sample_weight", "brand", "transform_weights"}
    # ignoring self would remove "sample_weight"
    assert router._get_param_names(
        method="fit", return_alias=False, ignore_self_request=True
    ) == {"brand", "transform_weights"}
    # return_alias is ignored when ignore_self_request=True
    assert router._get_param_names(
        method="fit", return_alias=True, ignore_self_request=True
    ) == router._get_param_names(
        method="fit", return_alias=False, ignore_self_request=True
    )


def test_method_generation():
    # Test if all required request methods are generated.

    # TODO: these test classes can be moved to sklearn.utils._testing once we
    # have a better idea of what the commonly used classes are.
    class SimpleEstimator(BaseEstimator):
        # This class should have no set_{method}_request
        def fit(self, X, y):
            pass  # pragma: no cover

        def partial_fit(self, X, y):
            pass  # pragma: no cover

        def predict(self, X):
            pass  # pragma: no cover

        def predict_proba(self, X):
            pass  # pragma: no cover

        def predict_log_proba(self, X):
            pass  # pragma: no cover

        def decision_function(self, X):
            pass  # pragma: no cover

        def score(self, X, y):
            pass  # pragma: no cover

        def split(self, X, y=None):
            pass  # pragma: no cover

        def transform(self, X):
            pass  # pragma: no cover

        def inverse_transform(self, X):
            pass  # pragma: no cover

    for method in METHODS:
        assert not hasattr(SimpleEstimator(), f"set_{method}_request")

    class SimpleEstimator(BaseEstimator):
        # This class should have every set_{method}_request
        def fit(self, X, y, sample_weight=None):
            pass  # pragma: no cover

        def partial_fit(self, X, y, sample_weight=None):
            pass  # pragma: no cover

        def predict(self, X, sample_weight=None):
            pass  # pragma: no cover

        def predict_proba(self, X, sample_weight=None):
            pass  # pragma: no cover

        def predict_log_proba(self, X, sample_weight=None):
            pass  # pragma: no cover

        def decision_function(self, X, sample_weight=None):
            pass  # pragma: no cover

        def score(self, X, y, sample_weight=None):
            pass  # pragma: no cover

        def split(self, X, y=None, sample_weight=None):
            pass  # pragma: no cover

        def transform(self, X, sample_weight=None):
            pass  # pragma: no cover

        def inverse_transform(self, X, sample_weight=None):
            pass  # pragma: no cover

    for method in METHODS:
        assert hasattr(SimpleEstimator(), f"set_{method}_request")


def test_no_feature_flag_raises_error():
    """Test that when feature flag disabled, set_{method}_requests raises."""
    with config_context(enable_metadata_routing=False):
        with pytest.raises(RuntimeError, match="This method is only available"):
            ClassifierFitMetadata().set_fit_request(sample_weight=True)


def test_none_metadata_passed():
    """Test that passing None as metadata when not requested doesn't raise"""
    MetaRegressor(estimator=RegressorMetadata()).fit(X, y, sample_weight=None)
