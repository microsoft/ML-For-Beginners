"""Utils to check the samplers and compatibility with scikit-learn"""

# Adapated from scikit-learn
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

import re
import sys
import traceback
import warnings
from collections import Counter
from functools import partial

import numpy as np
import pytest
import sklearn
from scipy import sparse
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.cluster import KMeans
from sklearn.datasets import (  # noqa
    load_iris,
    make_blobs,
    make_classification,
    make_multilabel_classification,
)
from sklearn.exceptions import SkipTestWarning
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.utils._tags import _safe_tags
from sklearn.utils._testing import (
    SkipTest,
    assert_allclose,
    assert_array_equal,
    assert_raises_regex,
    raises,
    set_random_state,
)
from sklearn.utils.estimator_checks import (
    _enforce_estimator_tags_y,
    _get_check_estimator_ids,
    _maybe_mark_xfail,
)

try:
    from sklearn.utils.estimator_checks import _enforce_estimator_tags_x
except ImportError:
    # scikit-learn >= 1.2
    from sklearn.utils.estimator_checks import (
        _enforce_estimator_tags_X as _enforce_estimator_tags_x,
    )

from sklearn.utils.fixes import parse_version
from sklearn.utils.multiclass import type_of_target

from imblearn.datasets import make_imbalance
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.under_sampling.base import BaseCleaningSampler, BaseUnderSampler
from imblearn.utils._param_validation import generate_invalid_param_val, make_constraint

sklearn_version = parse_version(sklearn.__version__)


def sample_dataset_generator():
    X, y = make_classification(
        n_samples=1000,
        n_classes=3,
        n_informative=4,
        weights=[0.2, 0.3, 0.5],
        random_state=0,
    )
    return X, y


@pytest.fixture(name="sample_dataset_generator")
def sample_dataset_generator_fixture():
    return sample_dataset_generator()


def _set_checking_parameters(estimator):
    params = estimator.get_params()
    name = estimator.__class__.__name__
    if "n_estimators" in params:
        estimator.set_params(n_estimators=min(5, estimator.n_estimators))
    if name == "ClusterCentroids":
        if sklearn_version < parse_version("1.1"):
            algorithm = "full"
        else:
            algorithm = "lloyd"
        estimator.set_params(
            voting="soft",
            estimator=KMeans(random_state=0, algorithm=algorithm, n_init=1),
        )
    if name == "KMeansSMOTE":
        estimator.set_params(kmeans_estimator=12)
    if name == "BalancedRandomForestClassifier":
        # TODO: remove in 0.13
        # future default in 0.13
        estimator.set_params(replacement=True, sampling_strategy="all", bootstrap=False)


def _yield_sampler_checks(sampler):
    tags = sampler._get_tags()
    yield check_target_type
    yield check_samplers_one_label
    yield check_samplers_fit
    yield check_samplers_fit_resample
    yield check_samplers_sampling_strategy_fit_resample
    if "sparse" in tags["X_types"]:
        yield check_samplers_sparse
    if "dataframe" in tags["X_types"]:
        yield check_samplers_pandas
        yield check_samplers_pandas_sparse
    if "string" in tags["X_types"]:
        yield check_samplers_string
    if tags["allow_nan"]:
        yield check_samplers_nan
    yield check_samplers_list
    yield check_samplers_multiclass_ova
    yield check_samplers_preserve_dtype
    # we don't filter samplers based on their tag here because we want to make
    # sure that the fitted attribute does not exist if the tag is not
    # stipulated
    yield check_samplers_sample_indices
    yield check_samplers_2d_target
    yield check_sampler_get_feature_names_out
    yield check_sampler_get_feature_names_out_pandas


def _yield_classifier_checks(classifier):
    yield check_classifier_on_multilabel_or_multioutput_targets
    yield check_classifiers_with_encoded_labels


def _yield_all_checks(estimator):
    name = estimator.__class__.__name__
    tags = estimator._get_tags()
    if tags["_skip_test"]:
        warnings.warn(
            f"Explicit SKIP via _skip_test tag for estimator {name}.",
            SkipTestWarning,
        )
        return
    # trigger our checks if this is a SamplerMixin
    if hasattr(estimator, "fit_resample"):
        for check in _yield_sampler_checks(estimator):
            yield check
    if hasattr(estimator, "predict"):
        for check in _yield_classifier_checks(estimator):
            yield check


def parametrize_with_checks(estimators):
    """Pytest specific decorator for parametrizing estimator checks.

    The `id` of each check is set to be a pprint version of the estimator
    and the name of the check with its keyword arguments.
    This allows to use `pytest -k` to specify which tests to run::

        pytest test_check_estimators.py -k check_estimators_fit_returns_self

    Parameters
    ----------
    estimators : list of estimators instances
        Estimators to generated checks for.

    Returns
    -------
    decorator : `pytest.mark.parametrize`

    Examples
    --------
    >>> from sklearn.utils.estimator_checks import parametrize_with_checks
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.tree import DecisionTreeRegressor

    >>> @parametrize_with_checks([LogisticRegression(),
    ...                           DecisionTreeRegressor()])
    ... def test_sklearn_compatible_estimator(estimator, check):
    ...     check(estimator)
    """

    def checks_generator():
        for estimator in estimators:
            name = type(estimator).__name__
            for check in _yield_all_checks(estimator):
                check = partial(check, name)
                yield _maybe_mark_xfail(estimator, check, pytest)

    return pytest.mark.parametrize(
        "estimator, check", checks_generator(), ids=_get_check_estimator_ids
    )


def check_target_type(name, estimator_orig):
    estimator = clone(estimator_orig)
    # should raise warning if the target is continuous (we cannot raise error)
    X = np.random.random((20, 2))
    y = np.linspace(0, 1, 20)
    msg = "Unknown label type:"
    assert_raises_regex(
        ValueError,
        msg,
        estimator.fit_resample,
        X,
        y,
    )
    # if the target is multilabel then we should raise an error
    rng = np.random.RandomState(42)
    y = rng.randint(2, size=(20, 3))
    msg = "Multilabel and multioutput targets are not supported."
    assert_raises_regex(
        ValueError,
        msg,
        estimator.fit_resample,
        X,
        y,
    )


def check_samplers_one_label(name, sampler_orig):
    sampler = clone(sampler_orig)
    error_string_fit = "Sampler can't balance when only one class is present."
    X = np.random.random((20, 2))
    y = np.zeros(20)
    try:
        sampler.fit_resample(X, y)
    except ValueError as e:
        if "class" not in repr(e):
            print(error_string_fit, sampler.__class__.__name__, e)
            traceback.print_exc(file=sys.stdout)
            raise e
        else:
            return
    except Exception as exc:
        print(error_string_fit, traceback, exc)
        traceback.print_exc(file=sys.stdout)
        raise exc
    raise AssertionError(error_string_fit)


def check_samplers_fit(name, sampler_orig):
    sampler = clone(sampler_orig)
    np.random.seed(42)  # Make this test reproducible
    X = np.random.random((30, 2))
    y = np.array([1] * 20 + [0] * 10)
    sampler.fit_resample(X, y)
    assert hasattr(
        sampler, "sampling_strategy_"
    ), "No fitted attribute sampling_strategy_"


def check_samplers_fit_resample(name, sampler_orig):
    sampler = clone(sampler_orig)
    X, y = sample_dataset_generator()
    target_stats = Counter(y)
    X_res, y_res = sampler.fit_resample(X, y)
    if isinstance(sampler, BaseOverSampler):
        target_stats_res = Counter(y_res)
        n_samples = max(target_stats.values())
        assert all(value >= n_samples for value in Counter(y_res).values())
    elif isinstance(sampler, BaseUnderSampler):
        n_samples = min(target_stats.values())
        if name == "InstanceHardnessThreshold":
            # IHT does not enforce the number of samples but provide a number
            # of samples the closest to the desired target.
            assert all(
                Counter(y_res)[k] <= target_stats[k] for k in target_stats.keys()
            )
        else:
            assert all(value == n_samples for value in Counter(y_res).values())
    elif isinstance(sampler, BaseCleaningSampler):
        target_stats_res = Counter(y_res)
        class_minority = min(target_stats, key=target_stats.get)
        assert all(
            target_stats[class_sample] > target_stats_res[class_sample]
            for class_sample in target_stats.keys()
            if class_sample != class_minority
        )


def check_samplers_sampling_strategy_fit_resample(name, sampler_orig):
    sampler = clone(sampler_orig)
    # in this test we will force all samplers to not change the class 1
    X, y = sample_dataset_generator()
    expected_stat = Counter(y)[1]
    if isinstance(sampler, BaseOverSampler):
        sampling_strategy = {2: 498, 0: 498}
        sampler.set_params(sampling_strategy=sampling_strategy)
        X_res, y_res = sampler.fit_resample(X, y)
        assert Counter(y_res)[1] == expected_stat
    elif isinstance(sampler, BaseUnderSampler):
        sampling_strategy = {2: 201, 0: 201}
        sampler.set_params(sampling_strategy=sampling_strategy)
        X_res, y_res = sampler.fit_resample(X, y)
        assert Counter(y_res)[1] == expected_stat
    elif isinstance(sampler, BaseCleaningSampler):
        sampling_strategy = [2, 0]
        sampler.set_params(sampling_strategy=sampling_strategy)
        X_res, y_res = sampler.fit_resample(X, y)
        assert Counter(y_res)[1] == expected_stat


def check_samplers_sparse(name, sampler_orig):
    sampler = clone(sampler_orig)
    # check that sparse matrices can be passed through the sampler leading to
    # the same results than dense
    X, y = sample_dataset_generator()
    X_sparse = sparse.csr_matrix(X)
    X_res_sparse, y_res_sparse = sampler.fit_resample(X_sparse, y)
    sampler = clone(sampler)
    X_res, y_res = sampler.fit_resample(X, y)
    assert sparse.issparse(X_res_sparse)
    assert_allclose(X_res_sparse.A, X_res, rtol=1e-5)
    assert_allclose(y_res_sparse, y_res)


def check_samplers_pandas_sparse(name, sampler_orig):
    pd = pytest.importorskip("pandas")
    sampler = clone(sampler_orig)
    # Check that the samplers handle pandas dataframe and pandas series
    X, y = sample_dataset_generator()
    X_df = pd.DataFrame(
        X, columns=[str(i) for i in range(X.shape[1])], dtype=pd.SparseDtype(float, 0)
    )
    y_s = pd.Series(y, name="class")

    X_res_df, y_res_s = sampler.fit_resample(X_df, y_s)
    X_res, y_res = sampler.fit_resample(X, y)

    # check that we return the same type for dataframes or series types
    assert isinstance(X_res_df, pd.DataFrame)
    assert isinstance(y_res_s, pd.Series)

    for column_dtype in X_res_df.dtypes:
        assert isinstance(column_dtype, pd.SparseDtype)

    assert X_df.columns.tolist() == X_res_df.columns.tolist()
    assert y_s.name == y_res_s.name

    # FIXME: we should use to_numpy with pandas >= 0.25
    assert_allclose(X_res_df.values, X_res)
    assert_allclose(y_res_s.values, y_res)


def check_samplers_pandas(name, sampler_orig):
    pd = pytest.importorskip("pandas")
    sampler = clone(sampler_orig)
    # Check that the samplers handle pandas dataframe and pandas series
    X, y = sample_dataset_generator()
    X_df = pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])])
    y_df = pd.DataFrame(y)
    y_s = pd.Series(y, name="class")

    X_res_df, y_res_s = sampler.fit_resample(X_df, y_s)
    X_res_df, y_res_df = sampler.fit_resample(X_df, y_df)
    X_res, y_res = sampler.fit_resample(X, y)

    # check that we return the same type for dataframes or series types
    assert isinstance(X_res_df, pd.DataFrame)
    assert isinstance(y_res_df, pd.DataFrame)
    assert isinstance(y_res_s, pd.Series)

    assert X_df.columns.tolist() == X_res_df.columns.tolist()
    assert y_df.columns.tolist() == y_res_df.columns.tolist()
    assert y_s.name == y_res_s.name

    # FIXME: we should use to_numpy with pandas >= 0.25
    assert_allclose(X_res_df.values, X_res)
    assert_allclose(y_res_df.values.ravel(), y_res)
    assert_allclose(y_res_s.values, y_res)


def check_samplers_list(name, sampler_orig):
    sampler = clone(sampler_orig)
    # Check that the can samplers handle simple lists
    X, y = sample_dataset_generator()
    X_list = X.tolist()
    y_list = y.tolist()

    X_res, y_res = sampler.fit_resample(X, y)
    X_res_list, y_res_list = sampler.fit_resample(X_list, y_list)

    assert isinstance(X_res_list, list)
    assert isinstance(y_res_list, list)

    assert_allclose(X_res, X_res_list)
    assert_allclose(y_res, y_res_list)


def check_samplers_multiclass_ova(name, sampler_orig):
    sampler = clone(sampler_orig)
    # Check that multiclass target lead to the same results than OVA encoding
    X, y = sample_dataset_generator()
    y_ova = label_binarize(y, classes=np.unique(y))
    X_res, y_res = sampler.fit_resample(X, y)
    X_res_ova, y_res_ova = sampler.fit_resample(X, y_ova)
    assert_allclose(X_res, X_res_ova)
    assert type_of_target(y_res_ova) == type_of_target(y_ova)
    assert_allclose(y_res, y_res_ova.argmax(axis=1))


def check_samplers_2d_target(name, sampler_orig):
    sampler = clone(sampler_orig)
    X, y = sample_dataset_generator()

    y = y.reshape(-1, 1)  # Make the target 2d
    sampler.fit_resample(X, y)


def check_samplers_preserve_dtype(name, sampler_orig):
    sampler = clone(sampler_orig)
    X, y = sample_dataset_generator()
    # Cast X and y to not default dtype
    X = X.astype(np.float32)
    y = y.astype(np.int32)
    X_res, y_res = sampler.fit_resample(X, y)
    assert X.dtype == X_res.dtype, "X dtype is not preserved"
    assert y.dtype == y_res.dtype, "y dtype is not preserved"


def check_samplers_sample_indices(name, sampler_orig):
    sampler = clone(sampler_orig)
    X, y = sample_dataset_generator()
    sampler.fit_resample(X, y)
    sample_indices = sampler._get_tags().get("sample_indices", None)
    if sample_indices:
        assert hasattr(sampler, "sample_indices_") is sample_indices
    else:
        assert not hasattr(sampler, "sample_indices_")


def check_samplers_string(name, sampler_orig):
    rng = np.random.RandomState(0)
    sampler = clone(sampler_orig)
    categories = np.array(["A", "B", "C"], dtype=object)
    n_samples = 30
    X = rng.randint(low=0, high=3, size=n_samples).reshape(-1, 1)
    X = categories[X]
    y = rng.permutation([0] * 10 + [1] * 20)

    X_res, y_res = sampler.fit_resample(X, y)
    assert X_res.dtype == object
    assert X_res.shape[0] == y_res.shape[0]
    assert_array_equal(np.unique(X_res.ravel()), categories)


def check_samplers_nan(name, sampler_orig):
    rng = np.random.RandomState(0)
    sampler = clone(sampler_orig)
    categories = np.array([0, 1, np.nan], dtype=np.float64)
    n_samples = 100
    X = rng.randint(low=0, high=3, size=n_samples).reshape(-1, 1)
    X = categories[X]
    y = rng.permutation([0] * 40 + [1] * 60)

    X_res, y_res = sampler.fit_resample(X, y)
    assert X_res.dtype == np.float64
    assert X_res.shape[0] == y_res.shape[0]
    assert np.any(np.isnan(X_res.ravel()))


def check_classifier_on_multilabel_or_multioutput_targets(name, estimator_orig):
    estimator = clone(estimator_orig)
    X, y = make_multilabel_classification(n_samples=30)
    msg = "Multilabel and multioutput targets are not supported."
    with pytest.raises(ValueError, match=msg):
        estimator.fit(X, y)


def check_classifiers_with_encoded_labels(name, classifier_orig):
    # Non-regression test for #709
    # https://github.com/scikit-learn-contrib/imbalanced-learn/issues/709
    pd = pytest.importorskip("pandas")
    classifier = clone(classifier_orig)
    iris = load_iris(as_frame=True)
    df, y = iris.data, iris.target
    y = pd.Series(iris.target_names[iris.target], dtype="category")
    df, y = make_imbalance(
        df,
        y,
        sampling_strategy={
            "setosa": 30,
            "versicolor": 20,
            "virginica": 50,
        },
    )
    classifier.set_params(sampling_strategy={"setosa": 20, "virginica": 20})
    classifier.fit(df, y)
    assert set(classifier.classes_) == set(y.cat.categories.tolist())
    y_pred = classifier.predict(df)
    assert set(y_pred) == set(y.cat.categories.tolist())


def check_param_validation(name, estimator_orig):
    # Check that an informative error is raised when the value of a constructor
    # parameter does not have an appropriate type or value.
    rng = np.random.RandomState(0)
    X = rng.uniform(size=(20, 5))
    y = rng.randint(0, 2, size=20)
    y = _enforce_estimator_tags_y(estimator_orig, y)

    estimator_params = estimator_orig.get_params(deep=False).keys()

    # check that there is a constraint for each parameter
    if estimator_params:
        validation_params = estimator_orig._parameter_constraints.keys()
        unexpected_params = set(validation_params) - set(estimator_params)
        missing_params = set(estimator_params) - set(validation_params)
        err_msg = (
            f"Mismatch between _parameter_constraints and the parameters of {name}."
            f"\nConsider the unexpected parameters {unexpected_params} and expected but"
            f" missing parameters {missing_params}"
        )
        assert validation_params == estimator_params, err_msg

    # this object does not have a valid type for sure for all params
    param_with_bad_type = type("BadType", (), {})()

    fit_methods = ["fit", "partial_fit", "fit_transform", "fit_predict", "fit_resample"]

    for param_name in estimator_params:
        constraints = estimator_orig._parameter_constraints[param_name]

        if constraints == "no_validation":
            # This parameter is not validated
            continue  # pragma: no cover

        match = rf"The '{param_name}' parameter of {name} must be .* Got .* instead."
        err_msg = (
            f"{name} does not raise an informative error message when the "
            f"parameter {param_name} does not have a valid type or value."
        )

        estimator = clone(estimator_orig)

        # First, check that the error is raised if param doesn't match any valid type.
        estimator.set_params(**{param_name: param_with_bad_type})

        for method in fit_methods:
            if not hasattr(estimator, method):
                # the method is not accessible with the current set of parameters
                continue

            with raises(ValueError, match=match, err_msg=err_msg):
                if any(
                    isinstance(X_type, str) and X_type.endswith("labels")
                    for X_type in _safe_tags(estimator, key="X_types")
                ):
                    # The estimator is a label transformer and take only `y`
                    getattr(estimator, method)(y)  # pragma: no cover
                else:
                    getattr(estimator, method)(X, y)

        # Then, for constraints that are more than a type constraint, check that the
        # error is raised if param does match a valid type but does not match any valid
        # value for this type.
        constraints = [make_constraint(constraint) for constraint in constraints]

        for constraint in constraints:
            try:
                bad_value = generate_invalid_param_val(constraint)
            except NotImplementedError:
                continue

            estimator.set_params(**{param_name: bad_value})

            for method in fit_methods:
                if not hasattr(estimator, method):
                    # the method is not accessible with the current set of parameters
                    continue

                with raises(ValueError, match=match, err_msg=err_msg):
                    if any(
                        X_type.endswith("labels")
                        for X_type in _safe_tags(estimator, key="X_types")
                    ):
                        # The estimator is a label transformer and take only `y`
                        getattr(estimator, method)(y)  # pragma: no cover
                    else:
                        getattr(estimator, method)(X, y)


def check_dataframe_column_names_consistency(name, estimator_orig):
    try:
        import pandas as pd
    except ImportError:
        raise SkipTest(
            "pandas is not installed: not checking column name consistency for pandas"
        )

    tags = _safe_tags(estimator_orig)
    is_supported_X_types = (
        "2darray" in tags["X_types"] or "categorical" in tags["X_types"]
    )

    if not is_supported_X_types or tags["no_validation"]:
        return

    rng = np.random.RandomState(0)

    estimator = clone(estimator_orig)
    set_random_state(estimator)

    X_orig = rng.normal(size=(150, 8))

    X_orig = _enforce_estimator_tags_x(estimator, X_orig)
    n_samples, n_features = X_orig.shape

    names = np.array([f"col_{i}" for i in range(n_features)])
    X = pd.DataFrame(X_orig, columns=names)

    if is_regressor(estimator):
        y = rng.normal(size=n_samples)
    else:
        y = rng.randint(low=0, high=2, size=n_samples)
    y = _enforce_estimator_tags_y(estimator, y)

    # Check that calling `fit` does not raise any warnings about feature names.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message="X does not have valid feature names",
            category=UserWarning,
            module="imblearn",
        )
        estimator.fit(X, y)

    if not hasattr(estimator, "feature_names_in_"):
        raise ValueError(
            "Estimator does not have a feature_names_in_ "
            "attribute after fitting with a dataframe"
        )
    assert isinstance(estimator.feature_names_in_, np.ndarray)
    assert estimator.feature_names_in_.dtype == object
    assert_array_equal(estimator.feature_names_in_, names)

    # Only check imblearn estimators for feature_names_in_ in docstring
    module_name = estimator_orig.__module__
    if (
        module_name.startswith("imblearn.")
        and not ("test_" in module_name or module_name.endswith("_testing"))
        and ("feature_names_in_" not in (estimator_orig.__doc__))
    ):
        raise ValueError(
            f"Estimator {name} does not document its feature_names_in_ attribute"
        )

    check_methods = []
    for method in (
        "predict",
        "transform",
        "decision_function",
        "predict_proba",
        "score",
        "score_samples",
        "predict_log_proba",
    ):
        if not hasattr(estimator, method):
            continue

        callable_method = getattr(estimator, method)
        if method == "score":
            callable_method = partial(callable_method, y=y)
        check_methods.append((method, callable_method))

    for _, method in check_methods:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error",
                message="X does not have valid feature names",
                category=UserWarning,
                module="sklearn",
            )
            method(X)  # works without UserWarning for valid features

    invalid_names = [
        (names[::-1], "Feature names must be in the same order as they were in fit."),
        (
            [f"another_prefix_{i}" for i in range(n_features)],
            "Feature names unseen at fit time:\n- another_prefix_0\n-"
            " another_prefix_1\n",
        ),
        (
            names[:3],
            f"Feature names seen at fit time, yet now missing:\n- {min(names[3:])}\n",
        ),
    ]
    params = {
        key: value
        for key, value in estimator.get_params().items()
        if "early_stopping" in key
    }
    early_stopping_enabled = any(value is True for value in params.values())

    for invalid_name, additional_message in invalid_names:
        X_bad = pd.DataFrame(X, columns=invalid_name)

        for name, method in check_methods:
            if sklearn_version >= parse_version("1.2"):
                expected_msg = re.escape(
                    "The feature names should match those that were passed during fit."
                    f"\n{additional_message}"
                )
                with raises(
                    ValueError, match=expected_msg, err_msg=f"{name} did not raise"
                ):
                    method(X_bad)
            else:
                expected_msg = re.escape(
                    "The feature names should match those that were passed "
                    "during fit. Starting version 1.2, an error will be raised.\n"
                    f"{additional_message}"
                )
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "error",
                        category=FutureWarning,
                        module="sklearn",
                    )
                    with raises(
                        FutureWarning,
                        match=expected_msg,
                        err_msg=f"{name} did not raise",
                    ):
                        method(X_bad)

        # partial_fit checks on second call
        # Do not call partial fit if early_stopping is on
        if not hasattr(estimator, "partial_fit") or early_stopping_enabled:
            continue

        estimator = clone(estimator_orig)
        if is_classifier(estimator):
            classes = np.unique(y)
            estimator.partial_fit(X, y, classes=classes)
        else:
            estimator.partial_fit(X, y)

        with raises(ValueError, match=expected_msg):
            estimator.partial_fit(X_bad, y)


def check_sampler_get_feature_names_out(name, sampler_orig):
    tags = sampler_orig._get_tags()
    if "2darray" not in tags["X_types"] or tags["no_validation"]:
        return

    X, y = make_blobs(
        n_samples=30,
        centers=[[0, 0, 0], [1, 1, 1]],
        random_state=0,
        n_features=2,
        cluster_std=0.1,
    )
    X = StandardScaler().fit_transform(X)

    sampler = clone(sampler_orig)
    X = _enforce_estimator_tags_x(sampler, X)

    n_features = X.shape[1]
    set_random_state(sampler)

    y_ = y
    X_res, y_res = sampler.fit_resample(X, y=y_)
    input_features = [f"feature{i}" for i in range(n_features)]

    # input_features names is not the same length as n_features_in_
    with raises(ValueError, match="input_features should have length equal"):
        sampler.get_feature_names_out(input_features[::2])

    feature_names_out = sampler.get_feature_names_out(input_features)
    assert feature_names_out is not None
    assert isinstance(feature_names_out, np.ndarray)
    assert feature_names_out.dtype == object
    assert all(isinstance(name, str) for name in feature_names_out)

    n_features_out = X_res.shape[1]

    assert (
        len(feature_names_out) == n_features_out
    ), f"Expected {n_features_out} feature names, got {len(feature_names_out)}"


def check_sampler_get_feature_names_out_pandas(name, sampler_orig):
    try:
        import pandas as pd
    except ImportError:
        raise SkipTest(
            "pandas is not installed: not checking column name consistency for pandas"
        )

    tags = sampler_orig._get_tags()
    if "2darray" not in tags["X_types"] or tags["no_validation"]:
        return

    X, y = make_blobs(
        n_samples=30,
        centers=[[0, 0, 0], [1, 1, 1]],
        random_state=0,
        n_features=2,
        cluster_std=0.1,
    )
    X = StandardScaler().fit_transform(X)

    sampler = clone(sampler_orig)
    X = _enforce_estimator_tags_x(sampler, X)

    n_features = X.shape[1]
    set_random_state(sampler)

    y_ = y
    feature_names_in = [f"col{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names_in)
    X_res, y_res = sampler.fit_resample(df, y=y_)

    # error is raised when `input_features` do not match feature_names_in
    invalid_feature_names = [f"bad{i}" for i in range(n_features)]
    with raises(ValueError, match="input_features is not equal to feature_names_in_"):
        sampler.get_feature_names_out(invalid_feature_names)

    feature_names_out_default = sampler.get_feature_names_out()
    feature_names_in_explicit_names = sampler.get_feature_names_out(feature_names_in)
    assert_array_equal(feature_names_out_default, feature_names_in_explicit_names)

    n_features_out = X_res.shape[1]

    assert (
        len(feature_names_out_default) == n_features_out
    ), f"Expected {n_features_out} feature names, got {len(feature_names_out_default)}"
