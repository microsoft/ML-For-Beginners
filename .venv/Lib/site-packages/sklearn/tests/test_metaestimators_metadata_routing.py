import copy
import re

import numpy as np
import pytest

from sklearn import config_context
from sklearn.base import is_classifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.covariance import GraphicalLassoCV
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.exceptions import UnsetMetadataPassedError
from sklearn.experimental import (
    enable_halving_search_cv,  # noqa
    enable_iterative_imputer,  # noqa
)
from sklearn.feature_selection import (
    RFE,
    RFECV,
    SelectFromModel,
    SequentialFeatureSelector,
)
from sklearn.impute import IterativeImputer
from sklearn.linear_model import (
    ElasticNetCV,
    LarsCV,
    LassoCV,
    LassoLarsCV,
    LogisticRegressionCV,
    MultiTaskElasticNetCV,
    MultiTaskLassoCV,
    OrthogonalMatchingPursuitCV,
    RANSACRegressor,
    RidgeClassifierCV,
    RidgeCV,
)
from sklearn.model_selection import (
    GridSearchCV,
    HalvingGridSearchCV,
    HalvingRandomSearchCV,
    RandomizedSearchCV,
)
from sklearn.multiclass import (
    OneVsOneClassifier,
    OneVsRestClassifier,
    OutputCodeClassifier,
)
from sklearn.multioutput import (
    ClassifierChain,
    MultiOutputClassifier,
    MultiOutputRegressor,
    RegressorChain,
)
from sklearn.pipeline import FeatureUnion
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.tests.metadata_routing_common import (
    ConsumingClassifier,
    ConsumingRegressor,
    ConsumingScorer,
    ConsumingSplitter,
    _Registry,
    assert_request_is_empty,
    check_recorded_metadata,
)
from sklearn.utils.metadata_routing import MetadataRouter

rng = np.random.RandomState(42)
N, M = 100, 4
X = rng.rand(N, M)
y = rng.randint(0, 3, size=N)
classes = np.unique(y)
y_multi = rng.randint(0, 3, size=(N, 3))
classes_multi = [np.unique(y_multi[:, i]) for i in range(y_multi.shape[1])]
metadata = rng.randint(0, 10, size=N)
sample_weight = rng.rand(N)
groups = np.array([0, 1] * (len(y) // 2))


@pytest.fixture(autouse=True)
def enable_slep006():
    """Enable SLEP006 for all tests."""
    with config_context(enable_metadata_routing=True):
        yield


METAESTIMATORS: list = [
    {
        "metaestimator": MultiOutputRegressor,
        "estimator_name": "estimator",
        "estimator": ConsumingRegressor,
        "X": X,
        "y": y_multi,
        "estimator_routing_methods": ["fit", "partial_fit"],
    },
    {
        "metaestimator": MultiOutputClassifier,
        "estimator_name": "estimator",
        "estimator": ConsumingClassifier,
        "X": X,
        "y": y_multi,
        "estimator_routing_methods": ["fit", "partial_fit"],
        "method_args": {"partial_fit": {"classes": classes_multi}},
    },
    {
        "metaestimator": CalibratedClassifierCV,
        "estimator_name": "estimator",
        "estimator": ConsumingClassifier,
        "X": X,
        "y": y,
        "estimator_routing_methods": ["fit"],
        "preserves_metadata": False,
    },
    {
        "metaestimator": ClassifierChain,
        "estimator_name": "base_estimator",
        "estimator": ConsumingClassifier,
        "X": X,
        "y": y_multi,
        "estimator_routing_methods": ["fit"],
    },
    {
        "metaestimator": RegressorChain,
        "estimator_name": "base_estimator",
        "estimator": ConsumingRegressor,
        "X": X,
        "y": y_multi,
        "estimator_routing_methods": ["fit"],
    },
    {
        "metaestimator": LogisticRegressionCV,
        "X": X,
        "y": y,
        "scorer_name": "scoring",
        "scorer_routing_methods": ["fit", "score"],
        "cv_name": "cv",
        "cv_routing_methods": ["fit"],
    },
    {
        "metaestimator": GridSearchCV,
        "estimator_name": "estimator",
        "estimator": ConsumingClassifier,
        "init_args": {"param_grid": {"alpha": [0.1, 0.2]}},
        "X": X,
        "y": y,
        "estimator_routing_methods": ["fit"],
        "preserves_metadata": "subset",
        "scorer_name": "scoring",
        "scorer_routing_methods": ["fit", "score"],
        "cv_name": "cv",
        "cv_routing_methods": ["fit"],
    },
    {
        "metaestimator": RandomizedSearchCV,
        "estimator_name": "estimator",
        "estimator": ConsumingClassifier,
        "init_args": {"param_distributions": {"alpha": [0.1, 0.2]}},
        "X": X,
        "y": y,
        "estimator_routing_methods": ["fit"],
        "preserves_metadata": "subset",
        "scorer_name": "scoring",
        "scorer_routing_methods": ["fit", "score"],
        "cv_name": "cv",
        "cv_routing_methods": ["fit"],
    },
    {
        "metaestimator": HalvingGridSearchCV,
        "estimator_name": "estimator",
        "estimator": ConsumingClassifier,
        "init_args": {"param_grid": {"alpha": [0.1, 0.2]}},
        "X": X,
        "y": y,
        "estimator_routing_methods": ["fit"],
        "preserves_metadata": "subset",
        "scorer_name": "scoring",
        "scorer_routing_methods": ["fit", "score"],
        "cv_name": "cv",
        "cv_routing_methods": ["fit"],
    },
    {
        "metaestimator": HalvingRandomSearchCV,
        "estimator_name": "estimator",
        "estimator": ConsumingClassifier,
        "init_args": {"param_distributions": {"alpha": [0.1, 0.2]}},
        "X": X,
        "y": y,
        "estimator_routing_methods": ["fit"],
        "preserves_metadata": "subset",
        "scorer_name": "scoring",
        "scorer_routing_methods": ["fit", "score"],
        "cv_name": "cv",
        "cv_routing_methods": ["fit"],
    },
    {
        "metaestimator": OneVsRestClassifier,
        "estimator_name": "estimator",
        "estimator": ConsumingClassifier,
        "X": X,
        "y": y,
        "estimator_routing_methods": ["fit", "partial_fit"],
        "method_args": {"partial_fit": {"classes": classes}},
    },
    {
        "metaestimator": OneVsOneClassifier,
        "estimator_name": "estimator",
        "estimator": ConsumingClassifier,
        "X": X,
        "y": y,
        "estimator_routing_methods": ["fit", "partial_fit"],
        "preserves_metadata": "subset",
        "method_args": {"partial_fit": {"classes": classes}},
    },
    {
        "metaestimator": OutputCodeClassifier,
        "estimator_name": "estimator",
        "estimator": ConsumingClassifier,
        "init_args": {"random_state": 42},
        "X": X,
        "y": y,
        "estimator_routing_methods": ["fit"],
    },
    {
        "metaestimator": SelectFromModel,
        "estimator_name": "estimator",
        "estimator": ConsumingClassifier,
        "X": X,
        "y": y,
        "estimator_routing_methods": ["fit", "partial_fit"],
        "method_args": {"partial_fit": {"classes": classes}},
    },
    {
        "metaestimator": OrthogonalMatchingPursuitCV,
        "X": X,
        "y": y,
        "cv_name": "cv",
        "cv_routing_methods": ["fit"],
    },
    {
        "metaestimator": ElasticNetCV,
        "X": X,
        "y": y,
        "cv_name": "cv",
        "cv_routing_methods": ["fit"],
    },
    {
        "metaestimator": LassoCV,
        "X": X,
        "y": y,
        "cv_name": "cv",
        "cv_routing_methods": ["fit"],
    },
    {
        "metaestimator": MultiTaskElasticNetCV,
        "X": X,
        "y": y_multi,
        "cv_name": "cv",
        "cv_routing_methods": ["fit"],
    },
    {
        "metaestimator": MultiTaskLassoCV,
        "X": X,
        "y": y_multi,
        "cv_name": "cv",
        "cv_routing_methods": ["fit"],
    },
    {
        "metaestimator": LarsCV,
        "X": X,
        "y": y,
        "cv_name": "cv",
        "cv_routing_methods": ["fit"],
    },
    {
        "metaestimator": LassoLarsCV,
        "X": X,
        "y": y,
        "cv_name": "cv",
        "cv_routing_methods": ["fit"],
    },
]
"""List containing all metaestimators to be tested and their settings

The keys are as follows:

- metaestimator: The metaestmator to be tested
- estimator_name: The name of the argument for the sub-estimator
- estimator: The sub-estimator
- init_args: The arguments to be passed to the metaestimator's constructor
- X: X-data to fit and predict
- y: y-data to fit
- estimator_routing_methods: list of all methods to check for routing metadata
  to the sub-estimator
- preserves_metadata:
    - True (default): the metaestimator passes the metadata to the
      sub-estimator without modification. We check that the values recorded by
      the sub-estimator are identical to what we've passed to the
      metaestimator.
    - False: no check is performed regarding values, we only check that a
      metadata with the expected names/keys are passed.
    - "subset": we check that the recorded metadata by the sub-estimator is a
      subset of what is passed to the metaestimator.
- scorer_name: The name of the argument for the scorer
- scorer_routing_methods: list of all methods to check for routing metadata
  to the scorer
- cv_name: The name of the argument for the CV splitter
- cv_routing_methods: list of all methods to check for routing metadata
  to the splitter
- method_args: a dict of dicts, defining extra arguments needed to be passed to
  methods, such as passing `classes` to `partial_fit`.
"""

# IDs used by pytest to get meaningful verbose messages when running the tests
METAESTIMATOR_IDS = [str(row["metaestimator"].__name__) for row in METAESTIMATORS]

UNSUPPORTED_ESTIMATORS = [
    AdaBoostClassifier(),
    AdaBoostRegressor(),
    BaggingClassifier(),
    BaggingRegressor(),
    FeatureUnion([]),
    GraphicalLassoCV(),
    IterativeImputer(),
    RANSACRegressor(),
    RFE(ConsumingClassifier()),
    RFECV(ConsumingClassifier()),
    RidgeCV(),
    RidgeClassifierCV(),
    SelfTrainingClassifier(ConsumingClassifier()),
    SequentialFeatureSelector(ConsumingClassifier()),
    StackingClassifier(ConsumingClassifier()),
    StackingRegressor(ConsumingRegressor()),
    TransformedTargetRegressor(),
    VotingClassifier(ConsumingClassifier()),
    VotingRegressor(ConsumingRegressor()),
]


def get_init_args(metaestimator_info):
    """Get the init args for a metaestimator

    This is a helper function to get the init args for a metaestimator from
    the METAESTIMATORS list. It returns an empty dict if no init args are
    required.

    Returns
    -------
    kwargs : dict
        The init args for the metaestimator.

    (estimator, estimator_registry) : (estimator, registry)
        The sub-estimator and the corresponding registry.

    (scorer, scorer_registry) : (scorer, registry)
        The scorer and the corresponding registry.

    (cv, cv_registry) : (CV splitter, registry)
        The CV splitter and the corresponding registry.
    """
    kwargs = metaestimator_info.get("init_args", {})
    estimator, estimator_registry = None, None
    scorer, scorer_registry = None, None
    cv, cv_registry = None, None
    if "estimator" in metaestimator_info:
        estimator_name = metaestimator_info["estimator_name"]
        estimator_registry = _Registry()
        estimator = metaestimator_info["estimator"](estimator_registry)
        kwargs[estimator_name] = estimator
    if "scorer_name" in metaestimator_info:
        scorer_name = metaestimator_info["scorer_name"]
        scorer_registry = _Registry()
        scorer = ConsumingScorer(registry=scorer_registry)
        kwargs[scorer_name] = scorer
    if "cv_name" in metaestimator_info:
        cv_name = metaestimator_info["cv_name"]
        cv_registry = _Registry()
        cv = ConsumingSplitter(registry=cv_registry)
        kwargs[cv_name] = cv

    return (
        kwargs,
        (estimator, estimator_registry),
        (scorer, scorer_registry),
        (cv, cv_registry),
    )


@pytest.mark.parametrize("estimator", UNSUPPORTED_ESTIMATORS)
def test_unsupported_estimators_get_metadata_routing(estimator):
    """Test that get_metadata_routing is not implemented on meta-estimators for
    which we haven't implemented routing yet."""
    with pytest.raises(NotImplementedError):
        estimator.get_metadata_routing()


@pytest.mark.parametrize("estimator", UNSUPPORTED_ESTIMATORS)
def test_unsupported_estimators_fit_with_metadata(estimator):
    """Test that fit raises NotImplementedError when metadata routing is
    enabled and a metadata is passed on meta-estimators for which we haven't
    implemented routing yet."""
    with pytest.raises(NotImplementedError):
        try:
            estimator.fit([[1]], [1], sample_weight=[1])
        except TypeError:
            # not all meta-estimators in the list support sample_weight,
            # and for those we skip this test.
            raise NotImplementedError


def test_registry_copy():
    # test that _Registry is not copied into a new instance.
    a = _Registry()
    b = _Registry()
    assert a is not b
    assert a is copy.copy(a)
    assert a is copy.deepcopy(a)


@pytest.mark.parametrize("metaestimator", METAESTIMATORS, ids=METAESTIMATOR_IDS)
def test_default_request(metaestimator):
    # Check that by default request is empty and the right type
    cls = metaestimator["metaestimator"]
    kwargs, *_ = get_init_args(metaestimator)
    instance = cls(**kwargs)
    if "cv_name" in metaestimator:
        # Our GroupCV splitters request groups by default, which we should
        # ignore in this test.
        exclude = {"splitter": ["split"]}
    else:
        exclude = None
    assert_request_is_empty(instance.get_metadata_routing(), exclude=exclude)
    assert isinstance(instance.get_metadata_routing(), MetadataRouter)


@pytest.mark.parametrize("metaestimator", METAESTIMATORS, ids=METAESTIMATOR_IDS)
def test_error_on_missing_requests_for_sub_estimator(metaestimator):
    # Test that a UnsetMetadataPassedError is raised when the sub-estimator's
    # requests are not set
    if "estimator" not in metaestimator:
        # This test only makes sense for metaestimators which have a
        # sub-estimator, e.g. MyMetaEstimator(estimator=MySubEstimator())
        return

    cls = metaestimator["metaestimator"]
    X = metaestimator["X"]
    y = metaestimator["y"]
    routing_methods = metaestimator["estimator_routing_methods"]

    for method_name in routing_methods:
        for key in ["sample_weight", "metadata"]:
            kwargs, (estimator, _), (scorer, _), *_ = get_init_args(metaestimator)
            if scorer:
                scorer.set_score_request(**{key: True})
            val = {"sample_weight": sample_weight, "metadata": metadata}[key]
            method_kwargs = {key: val}
            msg = (
                f"[{key}] are passed but are not explicitly set as requested or not"
                f" for {estimator.__class__.__name__}.{method_name}"
            )

            instance = cls(**kwargs)
            with pytest.raises(UnsetMetadataPassedError, match=re.escape(msg)):
                method = getattr(instance, method_name)
                method(X, y, **method_kwargs)


@pytest.mark.parametrize("metaestimator", METAESTIMATORS, ids=METAESTIMATOR_IDS)
def test_setting_request_on_sub_estimator_removes_error(metaestimator):
    # When the metadata is explicitly requested on the sub-estimator, there
    # should be no errors.
    if "estimator" not in metaestimator:
        # This test only makes sense for metaestimators which have a
        # sub-estimator, e.g. MyMetaEstimator(estimator=MySubEstimator())
        return

    def set_request(estimator, method_name):
        # e.g. call set_fit_request on estimator
        set_request_for_method = getattr(estimator, f"set_{method_name}_request")
        set_request_for_method(sample_weight=True, metadata=True)
        if is_classifier(estimator) and method_name == "partial_fit":
            set_request_for_method(classes=True)

    cls = metaestimator["metaestimator"]
    X = metaestimator["X"]
    y = metaestimator["y"]
    routing_methods = metaestimator["estimator_routing_methods"]
    preserves_metadata = metaestimator.get("preserves_metadata", True)

    for method_name in routing_methods:
        for key in ["sample_weight", "metadata"]:
            val = {"sample_weight": sample_weight, "metadata": metadata}[key]
            method_kwargs = {key: val}

            kwargs, (estimator, registry), (scorer, _), (cv, _) = get_init_args(
                metaestimator
            )
            if scorer:
                set_request(scorer, "score")
            if cv:
                cv.set_split_request(groups=True, metadata=True)
            set_request(estimator, method_name)
            instance = cls(**kwargs)
            method = getattr(instance, method_name)
            extra_method_args = metaestimator.get("method_args", {}).get(
                method_name, {}
            )
            method(X, y, **method_kwargs, **extra_method_args)
            # sanity check that registry is not empty, or else the test passes
            # trivially
            assert registry
            if preserves_metadata is True:
                for estimator in registry:
                    check_recorded_metadata(estimator, method_name, **method_kwargs)
            elif preserves_metadata == "subset":
                for estimator in registry:
                    check_recorded_metadata(
                        estimator,
                        method_name,
                        split_params=method_kwargs.keys(),
                        **method_kwargs,
                    )


@pytest.mark.parametrize("metaestimator", METAESTIMATORS, ids=METAESTIMATOR_IDS)
def test_metadata_is_routed_correctly_to_scorer(metaestimator):
    """Test that any requested metadata is correctly routed to the underlying
    scorers in CV estimators.
    """
    if "scorer_name" not in metaestimator:
        # This test only makes sense for CV estimators
        return

    cls = metaestimator["metaestimator"]
    routing_methods = metaestimator["scorer_routing_methods"]

    for method_name in routing_methods:
        kwargs, (estimator, _), (scorer, registry), (cv, _) = get_init_args(
            metaestimator
        )
        if estimator:
            estimator.set_fit_request(sample_weight=True, metadata=True)
        scorer.set_score_request(sample_weight=True)
        if cv:
            cv.set_split_request(groups=True, metadata=True)
        instance = cls(**kwargs)
        method = getattr(instance, method_name)
        method_kwargs = {"sample_weight": sample_weight}
        if "fit" not in method_name:
            instance.fit(X, y)
        method(X, y, **method_kwargs)

        assert registry
        for _scorer in registry:
            check_recorded_metadata(
                obj=_scorer,
                method="score",
                split_params=("sample_weight",),
                **method_kwargs,
            )


@pytest.mark.parametrize("metaestimator", METAESTIMATORS, ids=METAESTIMATOR_IDS)
def test_metadata_is_routed_correctly_to_splitter(metaestimator):
    """Test that any requested metadata is correctly routed to the underlying
    splitters in CV estimators.
    """
    if "cv_routing_methods" not in metaestimator:
        # This test is only for metaestimators accepting a CV splitter
        return

    cls = metaestimator["metaestimator"]
    routing_methods = metaestimator["cv_routing_methods"]
    X_ = metaestimator["X"]
    y_ = metaestimator["y"]

    for method_name in routing_methods:
        kwargs, (estimator, _), (scorer, _), (cv, registry) = get_init_args(
            metaestimator
        )
        if estimator:
            estimator.set_fit_request(sample_weight=False, metadata=False)
        if scorer:
            scorer.set_score_request(sample_weight=False, metadata=False)
        cv.set_split_request(groups=True, metadata=True)
        instance = cls(**kwargs)
        method_kwargs = {"groups": groups, "metadata": metadata}
        method = getattr(instance, method_name)
        method(X_, y_, **method_kwargs)
        assert registry
        for _splitter in registry:
            check_recorded_metadata(obj=_splitter, method="split", **method_kwargs)
