"""
The :mod:`sklearn.metrics.scorer` submodule implements a flexible
interface for model selection and evaluation using
arbitrary score functions.

A scorer object is a callable that can be passed to
:class:`~sklearn.model_selection.GridSearchCV` or
:func:`sklearn.model_selection.cross_val_score` as the ``scoring``
parameter, to specify how a model should be evaluated.

The signature of the call is ``(estimator, X, y)`` where ``estimator``
is the model to be evaluated, ``X`` is the test data and ``y`` is the
ground truth labeling (or ``None`` in the case of unsupervised models).
"""

# Authors: Andreas Mueller <amueller@ais.uni-bonn.de>
#          Lars Buitinck
#          Arnaud Joly <arnaud.v.joly@gmail.com>
# License: Simplified BSD

import copy
import warnings
from collections import Counter
from functools import partial
from inspect import signature
from traceback import format_exc

import numpy as np

from ..base import is_regressor
from ..utils import Bunch
from ..utils._param_validation import HasMethods, StrOptions, validate_params
from ..utils._response import _get_response_values
from ..utils.metadata_routing import (
    MetadataRequest,
    MetadataRouter,
    _MetadataRequester,
    _routing_enabled,
    get_routing_for_object,
    process_routing,
)
from ..utils.multiclass import type_of_target
from . import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    class_likelihood_ratios,
    explained_variance_score,
    f1_score,
    jaccard_score,
    log_loss,
    matthews_corrcoef,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_gamma_deviance,
    mean_poisson_deviance,
    mean_squared_error,
    mean_squared_log_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)
from .cluster import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    fowlkes_mallows_score,
    homogeneity_score,
    mutual_info_score,
    normalized_mutual_info_score,
    rand_score,
    v_measure_score,
)


def _cached_call(cache, estimator, response_method, *args, **kwargs):
    """Call estimator with method and args and kwargs."""
    if cache is not None and response_method in cache:
        return cache[response_method]

    result, _ = _get_response_values(
        estimator, *args, response_method=response_method, **kwargs
    )

    if cache is not None:
        cache[response_method] = result

    return result


class _MultimetricScorer:
    """Callable for multimetric scoring used to avoid repeated calls
    to `predict_proba`, `predict`, and `decision_function`.

    `_MultimetricScorer` will return a dictionary of scores corresponding to
    the scorers in the dictionary. Note that `_MultimetricScorer` can be
    created with a dictionary with one key  (i.e. only one actual scorer).

    Parameters
    ----------
    scorers : dict
        Dictionary mapping names to callable scorers.

    raise_exc : bool, default=True
        Whether to raise the exception in `__call__` or not. If set to `False`
        a formatted string of the exception details is passed as result of
        the failing scorer.
    """

    def __init__(self, *, scorers, raise_exc=True):
        self._scorers = scorers
        self._raise_exc = raise_exc

    def __call__(self, estimator, *args, **kwargs):
        """Evaluate predicted target values."""
        scores = {}
        cache = {} if self._use_cache(estimator) else None
        cached_call = partial(_cached_call, cache)

        if _routing_enabled():
            routed_params = process_routing(self, "score", kwargs)
        else:
            # they all get the same args, and they all get them all
            routed_params = Bunch(
                **{name: Bunch(score=kwargs) for name in self._scorers}
            )

        for name, scorer in self._scorers.items():
            try:
                if isinstance(scorer, _BaseScorer):
                    score = scorer._score(
                        cached_call, estimator, *args, **routed_params.get(name).score
                    )
                else:
                    score = scorer(estimator, *args, **routed_params.get(name).score)
                scores[name] = score
            except Exception as e:
                if self._raise_exc:
                    raise e
                else:
                    scores[name] = format_exc()
        return scores

    def _use_cache(self, estimator):
        """Return True if using a cache is beneficial.

        Caching may be beneficial when one of these conditions holds:
          - `_ProbaScorer` will be called twice.
          - `_PredictScorer` will be called twice.
          - `_ThresholdScorer` will be called twice.
          - `_ThresholdScorer` and `_PredictScorer` are called and
             estimator is a regressor.
          - `_ThresholdScorer` and `_ProbaScorer` are called and
             estimator does not have a `decision_function` attribute.

        """
        if len(self._scorers) == 1:  # Only one scorer
            return False

        counter = Counter([type(v) for v in self._scorers.values()])

        if any(
            counter[known_type] > 1
            for known_type in [_PredictScorer, _ProbaScorer, _ThresholdScorer]
        ):
            return True

        if counter[_ThresholdScorer]:
            if is_regressor(estimator) and counter[_PredictScorer]:
                return True
            elif counter[_ProbaScorer] and not hasattr(estimator, "decision_function"):
                return True
        return False

    def get_metadata_routing(self):
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.3

        Returns
        -------
        routing : MetadataRouter
            A :class:`~utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        return MetadataRouter(owner=self.__class__.__name__).add(
            **self._scorers, method_mapping="score"
        )


class _BaseScorer(_MetadataRequester):
    def __init__(self, score_func, sign, kwargs):
        self._kwargs = kwargs
        self._score_func = score_func
        self._sign = sign

    def _get_pos_label(self):
        if "pos_label" in self._kwargs:
            return self._kwargs["pos_label"]
        score_func_params = signature(self._score_func).parameters
        if "pos_label" in score_func_params:
            return score_func_params["pos_label"].default
        return None

    def __repr__(self):
        kwargs_string = "".join(
            [", %s=%s" % (str(k), str(v)) for k, v in self._kwargs.items()]
        )
        return "make_scorer(%s%s%s%s)" % (
            self._score_func.__name__,
            "" if self._sign > 0 else ", greater_is_better=False",
            self._factory_args(),
            kwargs_string,
        )

    def __call__(self, estimator, X, y_true, sample_weight=None, **kwargs):
        """Evaluate predicted target values for X relative to y_true.

        Parameters
        ----------
        estimator : object
            Trained estimator to use for scoring. Must have a predict_proba
            method; the output of that is used to compute the score.

        X : {array-like, sparse matrix}
            Test data that will be fed to estimator.predict.

        y_true : array-like
            Gold standard target values for X.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        **kwargs : dict
            Other parameters passed to the scorer. Refer to
            :func:`set_score_request` for more details.

            Only available if `enable_metadata_routing=True`. See the
            :ref:`User Guide <metadata_routing>`.

            .. versionadded:: 1.3

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        if kwargs and not _routing_enabled():
            raise ValueError(
                "kwargs is only supported if enable_metadata_routing=True. See"
                " the User Guide for more information."
            )

        _kwargs = copy.deepcopy(kwargs)
        if sample_weight is not None:
            _kwargs["sample_weight"] = sample_weight

        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)

    def _factory_args(self):
        """Return non-default make_scorer arguments for repr."""
        return ""

    def _warn_overlap(self, message, kwargs):
        """Warn if there is any overlap between ``self._kwargs`` and ``kwargs``.

        This method is intended to be used to check for overlap between
        ``self._kwargs`` and ``kwargs`` passed as metadata.
        """
        _kwargs = set() if self._kwargs is None else set(self._kwargs.keys())
        overlap = _kwargs.intersection(kwargs.keys())
        if overlap:
            warnings.warn(
                f"{message} Overlapping parameters are: {overlap}", UserWarning
            )

    def set_score_request(self, **kwargs):
        """Set requested parameters by the scorer.

        Please see :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.3

        Parameters
        ----------
        kwargs : dict
            Arguments should be of the form ``param_name=alias``, and `alias`
            can be one of ``{True, False, None, str}``.
        """
        self._warn_overlap(
            message=(
                "You are setting metadata request for parameters which are "
                "already set as kwargs for this metric. These set values will be "
                "overridden by passed metadata if provided. Please pass them either "
                "as metadata or kwargs to `make_scorer`."
            ),
            kwargs=kwargs,
        )
        self._metadata_request = MetadataRequest(owner=self.__class__.__name__)
        for param, alias in kwargs.items():
            self._metadata_request.score.add_request(param=param, alias=alias)
        return self


class _PredictScorer(_BaseScorer):
    def _score(self, method_caller, estimator, X, y_true, **kwargs):
        """Evaluate predicted target values for X relative to y_true.

        Parameters
        ----------
        method_caller : callable
            Returns predictions given an estimator, method name, and other
            arguments, potentially caching results.

        estimator : object
            Trained estimator to use for scoring. Must have a `predict`
            method; the output of that is used to compute the score.

        X : {array-like, sparse matrix}
            Test data that will be fed to estimator.predict.

        y_true : array-like
            Gold standard target values for X.

        **kwargs : dict
            Other parameters passed to the scorer. Refer to
            :func:`set_score_request` for more details.

            .. versionadded:: 1.3

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        self._warn_overlap(
            message=(
                "There is an overlap between set kwargs of this scorer instance and"
                " passed metadata. Please pass them either as kwargs to `make_scorer`"
                " or metadata, but not both."
            ),
            kwargs=kwargs,
        )
        y_pred = method_caller(estimator, "predict", X)
        scoring_kwargs = {**self._kwargs, **kwargs}
        return self._sign * self._score_func(y_true, y_pred, **scoring_kwargs)


class _ProbaScorer(_BaseScorer):
    def _score(self, method_caller, clf, X, y, **kwargs):
        """Evaluate predicted probabilities for X relative to y_true.

        Parameters
        ----------
        method_caller : callable
            Returns predictions given an estimator, method name, and other
            arguments, potentially caching results.

        clf : object
            Trained classifier to use for scoring. Must have a `predict_proba`
            method; the output of that is used to compute the score.

        X : {array-like, sparse matrix}
            Test data that will be fed to clf.predict_proba.

        y : array-like
            Gold standard target values for X. These must be class labels,
            not probabilities.

        **kwargs : dict
            Other parameters passed to the scorer. Refer to
            :func:`set_score_request` for more details.

            .. versionadded:: 1.3

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        self._warn_overlap(
            message=(
                "There is an overlap between set kwargs of this scorer instance and"
                " passed metadata. Please pass them either as kwargs to `make_scorer`"
                " or metadata, but not both."
            ),
            kwargs=kwargs,
        )

        y_pred = method_caller(clf, "predict_proba", X, pos_label=self._get_pos_label())
        scoring_kwargs = {**self._kwargs, **kwargs}
        return self._sign * self._score_func(y, y_pred, **scoring_kwargs)

    def _factory_args(self):
        return ", needs_proba=True"


class _ThresholdScorer(_BaseScorer):
    def _score(self, method_caller, clf, X, y, **kwargs):
        """Evaluate decision function output for X relative to y_true.

        Parameters
        ----------
        method_caller : callable
            Returns predictions given an estimator, method name, and other
            arguments, potentially caching results.

        clf : object
            Trained classifier to use for scoring. Must have either a
            decision_function method or a predict_proba method; the output of
            that is used to compute the score.

        X : {array-like, sparse matrix}
            Test data that will be fed to clf.decision_function or
            clf.predict_proba.

        y : array-like
            Gold standard target values for X. These must be class labels,
            not decision function values.

        **kwargs : dict
            Other parameters passed to the scorer. Refer to
            :func:`set_score_request` for more details.

            .. versionadded:: 1.3

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        self._warn_overlap(
            message=(
                "There is an overlap between set kwargs of this scorer instance and"
                " passed metadata. Please pass them either as kwargs to `make_scorer`"
                " or metadata, but not both."
            ),
            kwargs=kwargs,
        )

        y_type = type_of_target(y)
        if y_type not in ("binary", "multilabel-indicator"):
            raise ValueError("{0} format is not supported".format(y_type))

        if is_regressor(clf):
            y_pred = method_caller(clf, "predict", X)
        else:
            pos_label = self._get_pos_label()
            try:
                y_pred = method_caller(clf, "decision_function", X, pos_label=pos_label)

                if isinstance(y_pred, list):
                    # For multi-output multi-class estimator
                    y_pred = np.vstack([p for p in y_pred]).T

            except (NotImplementedError, AttributeError):
                y_pred = method_caller(clf, "predict_proba", X, pos_label=pos_label)
                if isinstance(y_pred, list):
                    y_pred = np.vstack([p[:, -1] for p in y_pred]).T

        scoring_kwargs = {**self._kwargs, **kwargs}
        return self._sign * self._score_func(y, y_pred, **scoring_kwargs)

    def _factory_args(self):
        return ", needs_threshold=True"


@validate_params(
    {
        "scoring": [str, callable, None],
    },
    prefer_skip_nested_validation=True,
)
def get_scorer(scoring):
    """Get a scorer from string.

    Read more in the :ref:`User Guide <scoring_parameter>`.
    :func:`~sklearn.metrics.get_scorer_names` can be used to retrieve the names
    of all available scorers.

    Parameters
    ----------
    scoring : str, callable or None
        Scoring method as string. If callable it is returned as is.
        If None, returns None.

    Returns
    -------
    scorer : callable
        The scorer.

    Notes
    -----
    When passed a string, this function always returns a copy of the scorer
    object. Calling `get_scorer` twice for the same scorer results in two
    separate scorer objects.
    """
    if isinstance(scoring, str):
        try:
            scorer = copy.deepcopy(_SCORERS[scoring])
        except KeyError:
            raise ValueError(
                "%r is not a valid scoring value. "
                "Use sklearn.metrics.get_scorer_names() "
                "to get valid options." % scoring
            )
    else:
        scorer = scoring
    return scorer


class _PassthroughScorer:
    def __init__(self, estimator):
        self._estimator = estimator

    def __call__(self, estimator, *args, **kwargs):
        """Method that wraps estimator.score"""
        return estimator.score(*args, **kwargs)

    def get_metadata_routing(self):
        """Get requested data properties.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.3

        Returns
        -------
        routing : MetadataRouter
            A :class:`~utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        # This scorer doesn't do any validation or routing, it only exposes the
        # score requests to the parent object. This object behaves as a
        # consumer rather than a router.
        res = MetadataRequest(owner=self._estimator.__class__.__name__)
        res.score = get_routing_for_object(self._estimator).score
        return res


def _check_multimetric_scoring(estimator, scoring):
    """Check the scoring parameter in cases when multiple metrics are allowed.

    Parameters
    ----------
    estimator : sklearn estimator instance
        The estimator for which the scoring will be applied.

    scoring : list, tuple or dict
        Strategy to evaluate the performance of the cross-validated model on
        the test set.

        The possibilities are:

        - a list or tuple of unique strings;
        - a callable returning a dictionary where they keys are the metric
          names and the values are the metric scores;
        - a dictionary with metric names as keys and callables a values.

        See :ref:`multimetric_grid_search` for an example.

    Returns
    -------
    scorers_dict : dict
        A dict mapping each scorer name to its validated scorer.
    """
    err_msg_generic = (
        f"scoring is invalid (got {scoring!r}). Refer to the "
        "scoring glossary for details: "
        "https://scikit-learn.org/stable/glossary.html#term-scoring"
    )

    if isinstance(scoring, (list, tuple, set)):
        err_msg = (
            "The list/tuple elements must be unique strings of predefined scorers. "
        )
        try:
            keys = set(scoring)
        except TypeError as e:
            raise ValueError(err_msg) from e

        if len(keys) != len(scoring):
            raise ValueError(
                f"{err_msg} Duplicate elements were found in"
                f" the given list. {scoring!r}"
            )
        elif len(keys) > 0:
            if not all(isinstance(k, str) for k in keys):
                if any(callable(k) for k in keys):
                    raise ValueError(
                        f"{err_msg} One or more of the elements "
                        "were callables. Use a dict of score "
                        "name mapped to the scorer callable. "
                        f"Got {scoring!r}"
                    )
                else:
                    raise ValueError(
                        f"{err_msg} Non-string types were found "
                        f"in the given list. Got {scoring!r}"
                    )
            scorers = {
                scorer: check_scoring(estimator, scoring=scorer) for scorer in scoring
            }
        else:
            raise ValueError(f"{err_msg} Empty list was given. {scoring!r}")

    elif isinstance(scoring, dict):
        keys = set(scoring)
        if not all(isinstance(k, str) for k in keys):
            raise ValueError(
                "Non-string types were found in the keys of "
                f"the given dict. scoring={scoring!r}"
            )
        if len(keys) == 0:
            raise ValueError(f"An empty dict was passed. {scoring!r}")
        scorers = {
            key: check_scoring(estimator, scoring=scorer)
            for key, scorer in scoring.items()
        }
    else:
        raise ValueError(err_msg_generic)
    return scorers


@validate_params(
    {
        "score_func": [callable],
        "greater_is_better": ["boolean"],
        "needs_proba": ["boolean"],
        "needs_threshold": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def make_scorer(
    score_func,
    *,
    greater_is_better=True,
    needs_proba=False,
    needs_threshold=False,
    **kwargs,
):
    """Make a scorer from a performance metric or loss function.

    This factory function wraps scoring functions for use in
    :class:`~sklearn.model_selection.GridSearchCV` and
    :func:`~sklearn.model_selection.cross_val_score`.
    It takes a score function, such as :func:`~sklearn.metrics.accuracy_score`,
    :func:`~sklearn.metrics.mean_squared_error`,
    :func:`~sklearn.metrics.adjusted_rand_score` or
    :func:`~sklearn.metrics.average_precision_score`
    and returns a callable that scores an estimator's output.
    The signature of the call is `(estimator, X, y)` where `estimator`
    is the model to be evaluated, `X` is the data and `y` is the
    ground truth labeling (or `None` in the case of unsupervised models).

    Read more in the :ref:`User Guide <scoring>`.

    Parameters
    ----------
    score_func : callable
        Score function (or loss function) with signature
        ``score_func(y, y_pred, **kwargs)``.

    greater_is_better : bool, default=True
        Whether `score_func` is a score function (default), meaning high is
        good, or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the `score_func`.

    needs_proba : bool, default=False
        Whether `score_func` requires `predict_proba` to get probability
        estimates out of a classifier.

        If True, for binary `y_true`, the score function is supposed to accept
        a 1D `y_pred` (i.e., probability of the positive class, shape
        `(n_samples,)`).

    needs_threshold : bool, default=False
        Whether `score_func` takes a continuous decision certainty.
        This only works for binary classification using estimators that
        have either a `decision_function` or `predict_proba` method.

        If True, for binary `y_true`, the score function is supposed to accept
        a 1D `y_pred` (i.e., probability of the positive class or the decision
        function, shape `(n_samples,)`).

        For example `average_precision` or the area under the roc curve
        can not be computed using discrete predictions alone.

    **kwargs : additional arguments
        Additional parameters to be passed to `score_func`.

    Returns
    -------
    scorer : callable
        Callable object that returns a scalar score; greater is better.

    Notes
    -----
    If `needs_proba=False` and `needs_threshold=False`, the score
    function is supposed to accept the output of :term:`predict`. If
    `needs_proba=True`, the score function is supposed to accept the
    output of :term:`predict_proba` (For binary `y_true`, the score function is
    supposed to accept probability of the positive class). If
    `needs_threshold=True`, the score function is supposed to accept the
    output of :term:`decision_function` or :term:`predict_proba` when
    :term:`decision_function` is not present.

    Examples
    --------
    >>> from sklearn.metrics import fbeta_score, make_scorer
    >>> ftwo_scorer = make_scorer(fbeta_score, beta=2)
    >>> ftwo_scorer
    make_scorer(fbeta_score, beta=2)
    >>> from sklearn.model_selection import GridSearchCV
    >>> from sklearn.svm import LinearSVC
    >>> grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]},
    ...                     scoring=ftwo_scorer)
    """
    sign = 1 if greater_is_better else -1
    if needs_proba and needs_threshold:
        raise ValueError(
            "Set either needs_proba or needs_threshold to True, but not both."
        )
    if needs_proba:
        cls = _ProbaScorer
    elif needs_threshold:
        cls = _ThresholdScorer
    else:
        cls = _PredictScorer
    return cls(score_func, sign, kwargs)


# Standard regression scores
explained_variance_scorer = make_scorer(explained_variance_score)
r2_scorer = make_scorer(r2_score)
max_error_scorer = make_scorer(max_error, greater_is_better=False)
neg_mean_squared_error_scorer = make_scorer(mean_squared_error, greater_is_better=False)
neg_mean_squared_log_error_scorer = make_scorer(
    mean_squared_log_error, greater_is_better=False
)
neg_mean_absolute_error_scorer = make_scorer(
    mean_absolute_error, greater_is_better=False
)
neg_mean_absolute_percentage_error_scorer = make_scorer(
    mean_absolute_percentage_error, greater_is_better=False
)
neg_median_absolute_error_scorer = make_scorer(
    median_absolute_error, greater_is_better=False
)
neg_root_mean_squared_error_scorer = make_scorer(
    mean_squared_error, greater_is_better=False, squared=False
)
neg_mean_poisson_deviance_scorer = make_scorer(
    mean_poisson_deviance, greater_is_better=False
)

neg_mean_gamma_deviance_scorer = make_scorer(
    mean_gamma_deviance, greater_is_better=False
)

# Standard Classification Scores
accuracy_scorer = make_scorer(accuracy_score)
balanced_accuracy_scorer = make_scorer(balanced_accuracy_score)
matthews_corrcoef_scorer = make_scorer(matthews_corrcoef)


def positive_likelihood_ratio(y_true, y_pred):
    return class_likelihood_ratios(y_true, y_pred)[0]


def negative_likelihood_ratio(y_true, y_pred):
    return class_likelihood_ratios(y_true, y_pred)[1]


positive_likelihood_ratio_scorer = make_scorer(positive_likelihood_ratio)
neg_negative_likelihood_ratio_scorer = make_scorer(
    negative_likelihood_ratio, greater_is_better=False
)

# Score functions that need decision values
top_k_accuracy_scorer = make_scorer(
    top_k_accuracy_score, greater_is_better=True, needs_threshold=True
)
roc_auc_scorer = make_scorer(
    roc_auc_score, greater_is_better=True, needs_threshold=True
)
average_precision_scorer = make_scorer(average_precision_score, needs_threshold=True)
roc_auc_ovo_scorer = make_scorer(roc_auc_score, needs_proba=True, multi_class="ovo")
roc_auc_ovo_weighted_scorer = make_scorer(
    roc_auc_score, needs_proba=True, multi_class="ovo", average="weighted"
)
roc_auc_ovr_scorer = make_scorer(roc_auc_score, needs_proba=True, multi_class="ovr")
roc_auc_ovr_weighted_scorer = make_scorer(
    roc_auc_score, needs_proba=True, multi_class="ovr", average="weighted"
)

# Score function for probabilistic classification
neg_log_loss_scorer = make_scorer(log_loss, greater_is_better=False, needs_proba=True)
neg_brier_score_scorer = make_scorer(
    brier_score_loss, greater_is_better=False, needs_proba=True
)
brier_score_loss_scorer = make_scorer(
    brier_score_loss, greater_is_better=False, needs_proba=True
)


# Clustering scores
adjusted_rand_scorer = make_scorer(adjusted_rand_score)
rand_scorer = make_scorer(rand_score)
homogeneity_scorer = make_scorer(homogeneity_score)
completeness_scorer = make_scorer(completeness_score)
v_measure_scorer = make_scorer(v_measure_score)
mutual_info_scorer = make_scorer(mutual_info_score)
adjusted_mutual_info_scorer = make_scorer(adjusted_mutual_info_score)
normalized_mutual_info_scorer = make_scorer(normalized_mutual_info_score)
fowlkes_mallows_scorer = make_scorer(fowlkes_mallows_score)


_SCORERS = dict(
    explained_variance=explained_variance_scorer,
    r2=r2_scorer,
    max_error=max_error_scorer,
    matthews_corrcoef=matthews_corrcoef_scorer,
    neg_median_absolute_error=neg_median_absolute_error_scorer,
    neg_mean_absolute_error=neg_mean_absolute_error_scorer,
    neg_mean_absolute_percentage_error=neg_mean_absolute_percentage_error_scorer,  # noqa
    neg_mean_squared_error=neg_mean_squared_error_scorer,
    neg_mean_squared_log_error=neg_mean_squared_log_error_scorer,
    neg_root_mean_squared_error=neg_root_mean_squared_error_scorer,
    neg_mean_poisson_deviance=neg_mean_poisson_deviance_scorer,
    neg_mean_gamma_deviance=neg_mean_gamma_deviance_scorer,
    accuracy=accuracy_scorer,
    top_k_accuracy=top_k_accuracy_scorer,
    roc_auc=roc_auc_scorer,
    roc_auc_ovr=roc_auc_ovr_scorer,
    roc_auc_ovo=roc_auc_ovo_scorer,
    roc_auc_ovr_weighted=roc_auc_ovr_weighted_scorer,
    roc_auc_ovo_weighted=roc_auc_ovo_weighted_scorer,
    balanced_accuracy=balanced_accuracy_scorer,
    average_precision=average_precision_scorer,
    neg_log_loss=neg_log_loss_scorer,
    neg_brier_score=neg_brier_score_scorer,
    positive_likelihood_ratio=positive_likelihood_ratio_scorer,
    neg_negative_likelihood_ratio=neg_negative_likelihood_ratio_scorer,
    # Cluster metrics that use supervised evaluation
    adjusted_rand_score=adjusted_rand_scorer,
    rand_score=rand_scorer,
    homogeneity_score=homogeneity_scorer,
    completeness_score=completeness_scorer,
    v_measure_score=v_measure_scorer,
    mutual_info_score=mutual_info_scorer,
    adjusted_mutual_info_score=adjusted_mutual_info_scorer,
    normalized_mutual_info_score=normalized_mutual_info_scorer,
    fowlkes_mallows_score=fowlkes_mallows_scorer,
)


def get_scorer_names():
    """Get the names of all available scorers.

    These names can be passed to :func:`~sklearn.metrics.get_scorer` to
    retrieve the scorer object.

    Returns
    -------
    list of str
        Names of all available scorers.
    """
    return sorted(_SCORERS.keys())


for name, metric in [
    ("precision", precision_score),
    ("recall", recall_score),
    ("f1", f1_score),
    ("jaccard", jaccard_score),
]:
    _SCORERS[name] = make_scorer(metric, average="binary")
    for average in ["macro", "micro", "samples", "weighted"]:
        qualified_name = "{0}_{1}".format(name, average)
        _SCORERS[qualified_name] = make_scorer(metric, pos_label=None, average=average)


@validate_params(
    {
        "estimator": [HasMethods("fit")],
        "scoring": [StrOptions(set(get_scorer_names())), callable, None],
        "allow_none": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def check_scoring(estimator, scoring=None, *, allow_none=False):
    """Determine scorer from user options.

    A TypeError will be thrown if the estimator cannot be scored.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    scoring : str or callable, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
        If None, the provided estimator object's `score` method is used.

    allow_none : bool, default=False
        If no scoring is specified and the estimator has no score function, we
        can either return None or raise an exception.

    Returns
    -------
    scoring : callable
        A scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
    """
    if isinstance(scoring, str):
        return get_scorer(scoring)
    if callable(scoring):
        # Heuristic to ensure user has not passed a metric
        module = getattr(scoring, "__module__", None)
        if (
            hasattr(module, "startswith")
            and module.startswith("sklearn.metrics.")
            and not module.startswith("sklearn.metrics._scorer")
            and not module.startswith("sklearn.metrics.tests.")
        ):
            raise ValueError(
                "scoring value %r looks like it is a metric "
                "function rather than a scorer. A scorer should "
                "require an estimator as its first parameter. "
                "Please use `make_scorer` to convert a metric "
                "to a scorer." % scoring
            )
        return get_scorer(scoring)
    if scoring is None:
        if hasattr(estimator, "score"):
            return _PassthroughScorer(estimator)
        elif allow_none:
            return None
        else:
            raise TypeError(
                "If no scoring is specified, the estimator passed should "
                "have a 'score' method. The estimator %r does not." % estimator
            )
