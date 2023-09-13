import copy
import re
from functools import partial

import numpy as np
import pytest

from sklearn import config_context
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import UnsetMetadataPassedError
from sklearn.multioutput import (
    ClassifierChain,
    MultiOutputClassifier,
    MultiOutputRegressor,
    RegressorChain,
)
from sklearn.tests.test_metadata_routing import (
    assert_request_is_empty,
    check_recorded_metadata,
    record_metadata,
)
from sklearn.utils.metadata_routing import MetadataRouter

rng = np.random.RandomState(42)
N, M = 100, 4
X = rng.rand(N, M)
y = rng.randint(0, 2, size=N)
y_multi = rng.randint(0, 2, size=(N, 3))
metadata = rng.randint(0, 10, size=N)
sample_weight = rng.rand(N)


@pytest.fixture(autouse=True)
def enable_slep006():
    """Enable SLEP006 for all tests."""
    with config_context(enable_metadata_routing=True):
        yield


record_metadata_not_default = partial(record_metadata, record_default=False)


class _Registry(list):
    # This list is used to get a reference to the sub-estimators, which are not
    # necessarily stored on the metaestimator. We need to override __deepcopy__
    # because the sub-estimators are probably cloned, which would result in a
    # new copy of the list, but we need copy and deep copy both to return the
    # same instance.
    def __deepcopy__(self, memo):
        return self

    def __copy__(self):
        return self


class ConsumingRegressor(RegressorMixin, BaseEstimator):
    """A regressor consuming metadata.

    Parameters
    ----------
    registry : list, default=None
        If a list, the estimator will append itself to the list in order to have
        a reference to the estimator later on. Since that reference is not
        required in all tests, registration can be skipped by leaving this value
        as None.

    """

    def __init__(self, registry=None):
        self.registry = registry

    def partial_fit(self, X, y, sample_weight="default", metadata="default"):
        if self.registry is not None:
            self.registry.append(self)

        record_metadata_not_default(
            self, "partial_fit", sample_weight=sample_weight, metadata=metadata
        )
        return self

    def fit(self, X, y, sample_weight="default", metadata="default"):
        if self.registry is not None:
            self.registry.append(self)

        record_metadata_not_default(
            self, "fit", sample_weight=sample_weight, metadata=metadata
        )
        return self

    def predict(self, X, sample_weight="default", metadata="default"):
        pass  # pragma: no cover

        # when needed, uncomment the implementation
        # if self.registry is not None:
        #     self.registry.append(self)

        # record_metadata_not_default(
        #     self, "predict", sample_weight=sample_weight, metadata=metadata
        # )
        # return np.zeros(shape=(len(X),))


class ConsumingClassifier(ClassifierMixin, BaseEstimator):
    """A classifier consuming metadata.

    Parameters
    ----------
    registry : list, default=None
        If a list, the estimator will append itself to the list in order to have
        a reference to the estimator later on. Since that reference is not
        required in all tests, registration can be skipped by leaving this value
        as None.

    """

    def __init__(self, registry=None):
        self.registry = registry

    def partial_fit(self, X, y, sample_weight="default", metadata="default"):
        if self.registry is not None:
            self.registry.append(self)

        record_metadata_not_default(
            self, "partial_fit", sample_weight=sample_weight, metadata=metadata
        )
        self.classes_ = [0, 1]
        return self

    def fit(self, X, y, sample_weight="default", metadata="default"):
        if self.registry is not None:
            self.registry.append(self)

        record_metadata_not_default(
            self, "fit", sample_weight=sample_weight, metadata=metadata
        )
        self.classes_ = [0, 1]
        return self

    def predict(self, X, sample_weight="default", metadata="default"):
        pass  # pragma: no cover

        # when needed, uncomment the implementation
        # if self.registry is not None:
        #     self.registry.append(self)

        # record_metadata_not_default(
        #     self, "predict", sample_weight=sample_weight, metadata=metadata
        # )
        # return np.zeros(shape=(len(X),))

    def predict_proba(self, X, sample_weight="default", metadata="default"):
        if self.registry is not None:
            self.registry.append(self)

        record_metadata_not_default(
            self, "predict_proba", sample_weight=sample_weight, metadata=metadata
        )
        return np.asarray([[0.0, 1.0]] * len(X))

    def predict_log_proba(self, X, sample_weight="default", metadata="default"):
        pass  # pragma: no cover

        # when needed, uncomment the implementation
        # if self.registry is not None:
        #     self.registry.append(self)

        # record_metadata_not_default(
        #     self, "predict_log_proba", sample_weight=sample_weight, metadata=metadata
        # )
        # return np.zeros(shape=(len(X), 2))


METAESTIMATORS: list = [
    {
        "metaestimator": MultiOutputRegressor,
        "estimator_name": "estimator",
        "estimator": ConsumingRegressor,
        "X": X,
        "y": y_multi,
        "routing_methods": ["fit", "partial_fit"],
    },
    {
        "metaestimator": MultiOutputClassifier,
        "estimator_name": "estimator",
        "estimator": ConsumingClassifier,
        "X": X,
        "y": y_multi,
        "routing_methods": ["fit", "partial_fit"],
    },
    {
        "metaestimator": CalibratedClassifierCV,
        "estimator_name": "estimator",
        "estimator": ConsumingClassifier,
        "X": X,
        "y": y,
        "routing_methods": ["fit"],
        "preserves_metadata": False,
    },
    {
        "metaestimator": ClassifierChain,
        "estimator_name": "base_estimator",
        "estimator": ConsumingClassifier,
        "X": X,
        "y": y_multi,
        "routing_methods": ["fit"],
    },
    {
        "metaestimator": RegressorChain,
        "estimator_name": "base_estimator",
        "estimator": ConsumingRegressor,
        "X": X,
        "y": y_multi,
        "routing_methods": ["fit"],
    },
]
"""List containing all metaestimators to be tested and their settings

The keys are as follows:

- metaestimator: The metaestmator to be tested
- estimator_name: The name of the argument for the sub-estimator
- estimator: The sub-estimator
- X: X-data to fit and predict
- y: y-data to fit
- routing_methods: list of all methods to check for routing
- preserves_metadata: Whether the metaestimator passes the metadata to the
  sub-estimator without modification or not. If it does, we check that the
  values are identical. If it doesn', no check is performed. TODO Maybe
  something smarter could be done if the data is modified.

"""

# ids used for pytest fixture
METAESTIMATOR_IDS = [str(row["metaestimator"].__name__) for row in METAESTIMATORS]


def test_registry_copy():
    # test that _Registry is not copied into a new instance.
    a = _Registry()
    b = _Registry()
    assert a is not b
    assert a is copy.copy(a)
    assert a is copy.deepcopy(a)


@pytest.mark.parametrize(
    "metaestimator",
    METAESTIMATORS,
    ids=METAESTIMATOR_IDS,
)
def test_default_request(metaestimator):
    # Check that by default request is empty and the right type
    cls = metaestimator["metaestimator"]
    estimator = metaestimator["estimator"]()
    estimator_name = metaestimator["estimator_name"]
    instance = cls(**{estimator_name: estimator})
    assert_request_is_empty(instance.get_metadata_routing())
    assert isinstance(instance.get_metadata_routing(), MetadataRouter)


@pytest.mark.parametrize(
    "metaestimator",
    METAESTIMATORS,
    ids=METAESTIMATOR_IDS,
)
def test_error_on_missing_requests(metaestimator):
    # Test that a UnsetMetadataPassedError is raised when it should.
    cls = metaestimator["metaestimator"]
    estimator = metaestimator["estimator"]()
    estimator_name = metaestimator["estimator_name"]
    X = metaestimator["X"]
    y = metaestimator["y"]
    routing_methods = metaestimator["routing_methods"]

    for method_name in routing_methods:
        for key in ["sample_weight", "metadata"]:
            val = {"sample_weight": sample_weight, "metadata": metadata}[key]
            kwargs = {key: val}
            msg = (
                f"[{key}] are passed but are not explicitly set as requested or not"
                f" for {estimator.__class__.__name__}.{method_name}"
            )

            instance = cls(**{estimator_name: estimator})
            if "fit" not in method_name:  # instance needs to be fitted first
                instance.fit(X, y)  # pragma: no cover
            with pytest.raises(UnsetMetadataPassedError, match=re.escape(msg)):
                method = getattr(instance, method_name)
                method(X, y, **kwargs)


@pytest.mark.parametrize(
    "metaestimator",
    METAESTIMATORS,
    ids=METAESTIMATOR_IDS,
)
def test_setting_request_removes_error(metaestimator):
    # When the metadata is explicitly requested, there should be no errors.
    def set_request(estimator, method_name):
        # e.g. call set_fit_request on estimator
        set_request_for_method = getattr(estimator, f"set_{method_name}_request")
        set_request_for_method(sample_weight=True, metadata=True)

    cls = metaestimator["metaestimator"]
    estimator_name = metaestimator["estimator_name"]
    X = metaestimator["X"]
    y = metaestimator["y"]
    routing_methods = metaestimator["routing_methods"]
    preserves_metadata = metaestimator.get("preserves_metadata", True)

    for method_name in routing_methods:
        for key in ["sample_weight", "metadata"]:
            val = {"sample_weight": sample_weight, "metadata": metadata}[key]
            kwargs = {key: val}

            registry = _Registry()
            estimator = metaestimator["estimator"](registry=registry)
            set_request(estimator, method_name)
            instance = cls(**{estimator_name: estimator})
            method = getattr(instance, method_name)
            method(X, y, **kwargs)

            if preserves_metadata:
                # sanity check that registry is not empty, or else the test
                # passes trivially
                assert registry
                for estimator in registry:
                    check_recorded_metadata(estimator, method_name, **kwargs)
