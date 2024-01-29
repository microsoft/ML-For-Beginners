"""
General tests for all estimators in sklearn.
"""

# Authors: Andreas Mueller <amueller@ais.uni-bonn.de>
#          Gael Varoquaux gael.varoquaux@normalesup.org
# License: BSD 3 clause

import os
import pkgutil
import re
import sys
import warnings
from functools import partial
from inspect import isgenerator, signature
from itertools import chain, product

import numpy as np
import pytest

import sklearn
from sklearn.cluster import (
    OPTICS,
    AffinityPropagation,
    Birch,
    MeanShift,
    SpectralClustering,
)
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning

# make it possible to discover experimental estimators when calling `all_estimators`
from sklearn.experimental import (
    enable_halving_search_cv,  # noqa
    enable_iterative_imputer,  # noqa
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.linear_model._base import LinearClassifierMixin
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from sklearn.model_selection import (
    GridSearchCV,
    HalvingGridSearchCV,
    HalvingRandomSearchCV,
    RandomizedSearchCV,
)
from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    LocalOutlierFactor,
    RadiusNeighborsClassifier,
    RadiusNeighborsRegressor,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.utils import _IS_WASM, IS_PYPY, all_estimators
from sklearn.utils._tags import _DEFAULT_TAGS, _safe_tags
from sklearn.utils._testing import (
    SkipTest,
    ignore_warnings,
    set_random_state,
)
from sklearn.utils.estimator_checks import (
    _construct_instance,
    _get_check_estimator_ids,
    _set_checking_parameters,
    check_class_weight_balanced_linear_classifier,
    check_dataframe_column_names_consistency,
    check_estimator,
    check_get_feature_names_out_error,
    check_global_output_transform_pandas,
    check_global_set_output_transform_polars,
    check_n_features_in_after_fitting,
    check_param_validation,
    check_set_output_transform,
    check_set_output_transform_pandas,
    check_set_output_transform_polars,
    check_transformer_get_feature_names_out,
    check_transformer_get_feature_names_out_pandas,
    parametrize_with_checks,
)


def test_all_estimator_no_base_class():
    # test that all_estimators doesn't find abstract classes.
    for name, Estimator in all_estimators():
        msg = (
            "Base estimators such as {0} should not be included in all_estimators"
        ).format(name)
        assert not name.lower().startswith("base"), msg


def _sample_func(x, y=1):
    pass


@pytest.mark.parametrize(
    "val, expected",
    [
        (partial(_sample_func, y=1), "_sample_func(y=1)"),
        (_sample_func, "_sample_func"),
        (partial(_sample_func, "world"), "_sample_func"),
        (LogisticRegression(C=2.0), "LogisticRegression(C=2.0)"),
        (
            LogisticRegression(
                random_state=1,
                solver="newton-cg",
                class_weight="balanced",
                warm_start=True,
            ),
            (
                "LogisticRegression(class_weight='balanced',random_state=1,"
                "solver='newton-cg',warm_start=True)"
            ),
        ),
    ],
)
def test_get_check_estimator_ids(val, expected):
    assert _get_check_estimator_ids(val) == expected


def _tested_estimators(type_filter=None):
    for name, Estimator in all_estimators(type_filter=type_filter):
        try:
            estimator = _construct_instance(Estimator)
        except SkipTest:
            continue

        yield estimator


def _generate_pipeline():
    for final_estimator in [Ridge(), LogisticRegression()]:
        yield Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("final_estimator", final_estimator),
            ]
        )


@parametrize_with_checks(list(chain(_tested_estimators(), _generate_pipeline())))
def test_estimators(estimator, check, request):
    # Common tests for estimator instances
    with ignore_warnings(category=(FutureWarning, ConvergenceWarning, UserWarning)):
        _set_checking_parameters(estimator)
        check(estimator)


def test_check_estimator_generate_only():
    all_instance_gen_checks = check_estimator(LogisticRegression(), generate_only=True)
    assert isgenerator(all_instance_gen_checks)


def test_configure():
    # Smoke test `python setup.py config` command run at the root of the
    # scikit-learn source tree.
    # This test requires Cython which is not necessarily there when running
    # the tests of an installed version of scikit-learn or when scikit-learn
    # is installed in editable mode by pip build isolation enabled.
    pytest.importorskip("Cython")
    cwd = os.getcwd()
    setup_path = os.path.abspath(os.path.join(sklearn.__path__[0], ".."))
    setup_filename = os.path.join(setup_path, "setup.py")
    if not os.path.exists(setup_filename):
        pytest.skip("setup.py not available")
    try:
        os.chdir(setup_path)
        old_argv = sys.argv
        sys.argv = ["setup.py", "config"]

        with warnings.catch_warnings():
            # The configuration spits out warnings when not finding
            # Blas/Atlas development headers
            warnings.simplefilter("ignore", UserWarning)
            with open("setup.py") as f:
                exec(f.read(), dict(__name__="__main__"))
    finally:
        sys.argv = old_argv
        os.chdir(cwd)


def _tested_linear_classifiers():
    classifiers = all_estimators(type_filter="classifier")

    with warnings.catch_warnings(record=True):
        for name, clazz in classifiers:
            required_parameters = getattr(clazz, "_required_parameters", [])
            if len(required_parameters):
                # FIXME
                continue

            if "class_weight" in clazz().get_params().keys() and issubclass(
                clazz, LinearClassifierMixin
            ):
                yield name, clazz


@pytest.mark.parametrize("name, Classifier", _tested_linear_classifiers())
def test_class_weight_balanced_linear_classifiers(name, Classifier):
    check_class_weight_balanced_linear_classifier(name, Classifier)


@pytest.mark.xfail(_IS_WASM, reason="importlib not supported for Pyodide packages")
@ignore_warnings
def test_import_all_consistency():
    # Smoke test to check that any name in a __all__ list is actually defined
    # in the namespace of the module or package.
    pkgs = pkgutil.walk_packages(
        path=sklearn.__path__, prefix="sklearn.", onerror=lambda _: None
    )
    submods = [modname for _, modname, _ in pkgs]
    for modname in submods + ["sklearn"]:
        if ".tests." in modname:
            continue
        # Avoid test suite depending on setuptools
        if "sklearn._build_utils" in modname:
            continue
        if IS_PYPY and (
            "_svmlight_format_io" in modname
            or "feature_extraction._hashing_fast" in modname
        ):
            continue
        package = __import__(modname, fromlist="dummy")
        for name in getattr(package, "__all__", ()):
            assert hasattr(package, name), "Module '{0}' has no attribute '{1}'".format(
                modname, name
            )


def test_root_import_all_completeness():
    EXCEPTIONS = ("utils", "tests", "base", "setup", "conftest")
    for _, modname, _ in pkgutil.walk_packages(
        path=sklearn.__path__, onerror=lambda _: None
    ):
        if "." in modname or modname.startswith("_") or modname in EXCEPTIONS:
            continue
        assert modname in sklearn.__all__


def test_all_tests_are_importable():
    # Ensure that for each contentful subpackage, there is a test directory
    # within it that is also a subpackage (i.e. a directory with __init__.py)

    HAS_TESTS_EXCEPTIONS = re.compile(r"""(?x)
                                      \.externals(\.|$)|
                                      \.tests(\.|$)|
                                      \._
                                      """)
    resource_modules = {
        "sklearn.datasets.data",
        "sklearn.datasets.descr",
        "sklearn.datasets.images",
    }
    lookup = {
        name: ispkg
        for _, name, ispkg in pkgutil.walk_packages(sklearn.__path__, prefix="sklearn.")
    }
    missing_tests = [
        name
        for name, ispkg in lookup.items()
        if ispkg
        and name not in resource_modules
        and not HAS_TESTS_EXCEPTIONS.search(name)
        and name + ".tests" not in lookup
    ]
    assert missing_tests == [], (
        "{0} do not have `tests` subpackages. "
        "Perhaps they require "
        "__init__.py or an add_subpackage directive "
        "in the parent "
        "setup.py".format(missing_tests)
    )


def test_class_support_removed():
    # Make sure passing classes to check_estimator or parametrize_with_checks
    # raises an error

    msg = "Passing a class was deprecated.* isn't supported anymore"
    with pytest.raises(TypeError, match=msg):
        check_estimator(LogisticRegression)

    with pytest.raises(TypeError, match=msg):
        parametrize_with_checks([LogisticRegression])


def _generate_column_transformer_instances():
    yield ColumnTransformer(
        transformers=[
            ("trans1", StandardScaler(), [0, 1]),
        ]
    )


def _generate_search_cv_instances():
    for SearchCV, (Estimator, param_grid) in product(
        [
            GridSearchCV,
            HalvingGridSearchCV,
            RandomizedSearchCV,
            HalvingGridSearchCV,
        ],
        [
            (Ridge, {"alpha": [0.1, 1.0]}),
            (LogisticRegression, {"C": [0.1, 1.0]}),
        ],
    ):
        init_params = signature(SearchCV).parameters
        extra_params = (
            {"min_resources": "smallest"} if "min_resources" in init_params else {}
        )
        search_cv = SearchCV(Estimator(), param_grid, cv=2, **extra_params)
        set_random_state(search_cv)
        yield search_cv

    for SearchCV, (Estimator, param_grid) in product(
        [
            GridSearchCV,
            HalvingGridSearchCV,
            RandomizedSearchCV,
            HalvingRandomSearchCV,
        ],
        [
            (Ridge, {"ridge__alpha": [0.1, 1.0]}),
            (LogisticRegression, {"logisticregression__C": [0.1, 1.0]}),
        ],
    ):
        init_params = signature(SearchCV).parameters
        extra_params = (
            {"min_resources": "smallest"} if "min_resources" in init_params else {}
        )
        search_cv = SearchCV(
            make_pipeline(PCA(), Estimator()), param_grid, cv=2, **extra_params
        ).set_params(error_score="raise")
        set_random_state(search_cv)
        yield search_cv


@parametrize_with_checks(list(_generate_search_cv_instances()))
def test_search_cv(estimator, check, request):
    # Common tests for SearchCV instances
    # We have a separate test because those meta-estimators can accept a
    # wide range of base estimators (classifiers, regressors, pipelines)
    with ignore_warnings(
        category=(
            FutureWarning,
            ConvergenceWarning,
            UserWarning,
            FitFailedWarning,
        )
    ):
        check(estimator)


@pytest.mark.parametrize(
    "estimator", _tested_estimators(), ids=_get_check_estimator_ids
)
def test_valid_tag_types(estimator):
    """Check that estimator tags are valid."""
    tags = _safe_tags(estimator)

    for name, tag in tags.items():
        correct_tags = type(_DEFAULT_TAGS[name])
        if name == "_xfail_checks":
            # _xfail_checks can be a dictionary
            correct_tags = (correct_tags, dict)
        assert isinstance(tag, correct_tags)


@pytest.mark.parametrize(
    "estimator", _tested_estimators(), ids=_get_check_estimator_ids
)
def test_check_n_features_in_after_fitting(estimator):
    _set_checking_parameters(estimator)
    check_n_features_in_after_fitting(estimator.__class__.__name__, estimator)


def _estimators_that_predict_in_fit():
    for estimator in _tested_estimators():
        est_params = set(estimator.get_params())
        if "oob_score" in est_params:
            yield estimator.set_params(oob_score=True, bootstrap=True)
        elif "early_stopping" in est_params:
            est = estimator.set_params(early_stopping=True, n_iter_no_change=1)
            if est.__class__.__name__ in {"MLPClassifier", "MLPRegressor"}:
                # TODO: FIX MLP to not check validation set during MLP
                yield pytest.param(
                    est, marks=pytest.mark.xfail(msg="MLP still validates in fit")
                )
            else:
                yield est
        elif "n_iter_no_change" in est_params:
            yield estimator.set_params(n_iter_no_change=1)


# NOTE: When running `check_dataframe_column_names_consistency` on a meta-estimator that
# delegates validation to a base estimator, the check is testing that the base estimator
# is checking for column name consistency.
column_name_estimators = list(
    chain(
        _tested_estimators(),
        [make_pipeline(LogisticRegression(C=1))],
        list(_generate_search_cv_instances()),
        _estimators_that_predict_in_fit(),
    )
)


@pytest.mark.parametrize(
    "estimator", column_name_estimators, ids=_get_check_estimator_ids
)
def test_pandas_column_name_consistency(estimator):
    _set_checking_parameters(estimator)
    with ignore_warnings(category=(FutureWarning)):
        with warnings.catch_warnings(record=True) as record:
            check_dataframe_column_names_consistency(
                estimator.__class__.__name__, estimator
            )
        for warning in record:
            assert "was fitted without feature names" not in str(warning.message)


# TODO: As more modules support get_feature_names_out they should be removed
# from this list to be tested
GET_FEATURES_OUT_MODULES_TO_IGNORE = [
    "ensemble",
    "kernel_approximation",
]


def _include_in_get_feature_names_out_check(transformer):
    if hasattr(transformer, "get_feature_names_out"):
        return True
    module = transformer.__module__.split(".")[1]
    return module not in GET_FEATURES_OUT_MODULES_TO_IGNORE


GET_FEATURES_OUT_ESTIMATORS = [
    est
    for est in _tested_estimators("transformer")
    if _include_in_get_feature_names_out_check(est)
]


@pytest.mark.parametrize(
    "transformer", GET_FEATURES_OUT_ESTIMATORS, ids=_get_check_estimator_ids
)
def test_transformers_get_feature_names_out(transformer):
    _set_checking_parameters(transformer)

    with ignore_warnings(category=(FutureWarning)):
        check_transformer_get_feature_names_out(
            transformer.__class__.__name__, transformer
        )
        check_transformer_get_feature_names_out_pandas(
            transformer.__class__.__name__, transformer
        )


ESTIMATORS_WITH_GET_FEATURE_NAMES_OUT = [
    est for est in _tested_estimators() if hasattr(est, "get_feature_names_out")
]


@pytest.mark.parametrize(
    "estimator", ESTIMATORS_WITH_GET_FEATURE_NAMES_OUT, ids=_get_check_estimator_ids
)
def test_estimators_get_feature_names_out_error(estimator):
    estimator_name = estimator.__class__.__name__
    _set_checking_parameters(estimator)
    check_get_feature_names_out_error(estimator_name, estimator)


@pytest.mark.parametrize(
    "Estimator",
    [est for name, est in all_estimators()],
)
def test_estimators_do_not_raise_errors_in_init_or_set_params(Estimator):
    """Check that init or set_param does not raise errors."""
    params = signature(Estimator).parameters

    smoke_test_values = [-1, 3.0, "helloworld", np.array([1.0, 4.0]), [1], {}, []]
    for value in smoke_test_values:
        new_params = {key: value for key in params}

        # Does not raise
        est = Estimator(**new_params)

        # Also do does not raise
        est.set_params(**new_params)


@pytest.mark.parametrize(
    "estimator",
    chain(
        _tested_estimators(),
        _generate_pipeline(),
        _generate_column_transformer_instances(),
        _generate_search_cv_instances(),
    ),
    ids=_get_check_estimator_ids,
)
def test_check_param_validation(estimator):
    name = estimator.__class__.__name__
    _set_checking_parameters(estimator)
    check_param_validation(name, estimator)


@pytest.mark.parametrize(
    "Estimator",
    [
        AffinityPropagation,
        Birch,
        MeanShift,
        KNeighborsClassifier,
        KNeighborsRegressor,
        RadiusNeighborsClassifier,
        RadiusNeighborsRegressor,
        LabelPropagation,
        LabelSpreading,
        OPTICS,
        SpectralClustering,
        LocalOutlierFactor,
        LocallyLinearEmbedding,
        Isomap,
        TSNE,
    ],
)
def test_f_contiguous_array_estimator(Estimator):
    # Non-regression test for:
    # https://github.com/scikit-learn/scikit-learn/issues/23988
    # https://github.com/scikit-learn/scikit-learn/issues/24013

    X, _ = make_blobs(n_samples=80, n_features=4, random_state=0)
    X = np.asfortranarray(X)
    y = np.round(X[:, 0])

    est = Estimator()
    est.fit(X, y)

    if hasattr(est, "transform"):
        est.transform(X)

    if hasattr(est, "predict"):
        est.predict(X)


SET_OUTPUT_ESTIMATORS = list(
    chain(
        _tested_estimators("transformer"),
        [
            make_pipeline(StandardScaler(), MinMaxScaler()),
            OneHotEncoder(sparse_output=False),
            FunctionTransformer(feature_names_out="one-to-one"),
        ],
    )
)


@pytest.mark.parametrize(
    "estimator", SET_OUTPUT_ESTIMATORS, ids=_get_check_estimator_ids
)
def test_set_output_transform(estimator):
    name = estimator.__class__.__name__
    if not hasattr(estimator, "set_output"):
        pytest.skip(
            f"Skipping check_set_output_transform for {name}: Does not support"
            " set_output API"
        )
    _set_checking_parameters(estimator)
    with ignore_warnings(category=(FutureWarning)):
        check_set_output_transform(estimator.__class__.__name__, estimator)


@pytest.mark.parametrize(
    "estimator", SET_OUTPUT_ESTIMATORS, ids=_get_check_estimator_ids
)
@pytest.mark.parametrize(
    "check_func",
    [
        check_set_output_transform_pandas,
        check_global_output_transform_pandas,
        check_set_output_transform_polars,
        check_global_set_output_transform_polars,
    ],
)
def test_set_output_transform_configured(estimator, check_func):
    name = estimator.__class__.__name__
    if not hasattr(estimator, "set_output"):
        pytest.skip(
            f"Skipping {check_func.__name__} for {name}: Does not support"
            " set_output API yet"
        )
    _set_checking_parameters(estimator)
    with ignore_warnings(category=(FutureWarning)):
        check_func(estimator.__class__.__name__, estimator)
