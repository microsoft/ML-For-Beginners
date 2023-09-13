# Authors: Gilles Louppe, Mathieu Blondel, Maheshakya Wijewardena
# License: BSD 3 clause

from copy import deepcopy
from numbers import Integral, Real

import numpy as np

from ..base import BaseEstimator, MetaEstimatorMixin, _fit_context, clone
from ..exceptions import NotFittedError
from ..utils._param_validation import HasMethods, Interval, Options
from ..utils._tags import _safe_tags
from ..utils.metaestimators import available_if
from ..utils.validation import _num_features, check_is_fitted, check_scalar
from ._base import SelectorMixin, _get_feature_importances


def _calculate_threshold(estimator, importances, threshold):
    """Interpret the threshold value"""

    if threshold is None:
        # determine default from estimator
        est_name = estimator.__class__.__name__
        is_l1_penalized = hasattr(estimator, "penalty") and estimator.penalty == "l1"
        is_lasso = "Lasso" in est_name
        is_elasticnet_l1_penalized = "ElasticNet" in est_name and (
            (hasattr(estimator, "l1_ratio_") and np.isclose(estimator.l1_ratio_, 1.0))
            or (hasattr(estimator, "l1_ratio") and np.isclose(estimator.l1_ratio, 1.0))
        )
        if is_l1_penalized or is_lasso or is_elasticnet_l1_penalized:
            # the natural default threshold is 0 when l1 penalty was used
            threshold = 1e-5
        else:
            threshold = "mean"

    if isinstance(threshold, str):
        if "*" in threshold:
            scale, reference = threshold.split("*")
            scale = float(scale.strip())
            reference = reference.strip()

            if reference == "median":
                reference = np.median(importances)
            elif reference == "mean":
                reference = np.mean(importances)
            else:
                raise ValueError("Unknown reference: " + reference)

            threshold = scale * reference

        elif threshold == "median":
            threshold = np.median(importances)

        elif threshold == "mean":
            threshold = np.mean(importances)

        else:
            raise ValueError(
                "Expected threshold='mean' or threshold='median' got %s" % threshold
            )

    else:
        threshold = float(threshold)

    return threshold


def _estimator_has(attr):
    """Check if we can delegate a method to the underlying estimator.

    First, we check the fitted estimator if available, otherwise we
    check the unfitted estimator.
    """
    return lambda self: (
        hasattr(self.estimator_, attr)
        if hasattr(self, "estimator_")
        else hasattr(self.estimator, attr)
    )


class SelectFromModel(MetaEstimatorMixin, SelectorMixin, BaseEstimator):
    """Meta-transformer for selecting features based on importance weights.

    .. versionadded:: 0.17

    Read more in the :ref:`User Guide <select_from_model>`.

    Parameters
    ----------
    estimator : object
        The base estimator from which the transformer is built.
        This can be both a fitted (if ``prefit`` is set to True)
        or a non-fitted estimator. The estimator should have a
        ``feature_importances_`` or ``coef_`` attribute after fitting.
        Otherwise, the ``importance_getter`` parameter should be used.

    threshold : str or float, default=None
        The threshold value to use for feature selection. Features whose
        absolute importance value is greater or equal are kept while the others
        are discarded. If "median" (resp. "mean"), then the ``threshold`` value
        is the median (resp. the mean) of the feature importances. A scaling
        factor (e.g., "1.25*mean") may also be used. If None and if the
        estimator has a parameter penalty set to l1, either explicitly
        or implicitly (e.g, Lasso), the threshold used is 1e-5.
        Otherwise, "mean" is used by default.

    prefit : bool, default=False
        Whether a prefit model is expected to be passed into the constructor
        directly or not.
        If `True`, `estimator` must be a fitted estimator.
        If `False`, `estimator` is fitted and updated by calling
        `fit` and `partial_fit`, respectively.

    norm_order : non-zero int, inf, -inf, default=1
        Order of the norm used to filter the vectors of coefficients below
        ``threshold`` in the case where the ``coef_`` attribute of the
        estimator is of dimension 2.

    max_features : int, callable, default=None
        The maximum number of features to select.

        - If an integer, then it specifies the maximum number of features to
          allow.
        - If a callable, then it specifies how to calculate the maximum number of
          features allowed by using the output of `max_features(X)`.
        - If `None`, then all features are kept.

        To only select based on ``max_features``, set ``threshold=-np.inf``.

        .. versionadded:: 0.20
        .. versionchanged:: 1.1
           `max_features` accepts a callable.

    importance_getter : str or callable, default='auto'
        If 'auto', uses the feature importance either through a ``coef_``
        attribute or ``feature_importances_`` attribute of estimator.

        Also accepts a string that specifies an attribute name/path
        for extracting feature importance (implemented with `attrgetter`).
        For example, give `regressor_.coef_` in case of
        :class:`~sklearn.compose.TransformedTargetRegressor`  or
        `named_steps.clf.feature_importances_` in case of
        :class:`~sklearn.pipeline.Pipeline` with its last step named `clf`.

        If `callable`, overrides the default feature importance getter.
        The callable is passed with the fitted estimator and it should
        return importance for each feature.

        .. versionadded:: 0.24

    Attributes
    ----------
    estimator_ : estimator
        The base estimator from which the transformer is built. This attribute
        exist only when `fit` has been called.

        - If `prefit=True`, it is a deep copy of `estimator`.
        - If `prefit=False`, it is a clone of `estimator` and fit on the data
          passed to `fit` or `partial_fit`.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

        .. versionadded:: 0.24

    max_features_ : int
        Maximum number of features calculated during :term:`fit`. Only defined
        if the ``max_features`` is not `None`.

        - If `max_features` is an `int`, then `max_features_ = max_features`.
        - If `max_features` is a callable, then `max_features_ = max_features(X)`.

        .. versionadded:: 1.1

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    threshold_ : float
        The threshold value used for feature selection.

    See Also
    --------
    RFE : Recursive feature elimination based on importance weights.
    RFECV : Recursive feature elimination with built-in cross-validated
        selection of the best number of features.
    SequentialFeatureSelector : Sequential cross-validation based feature
        selection. Does not rely on importance weights.

    Notes
    -----
    Allows NaN/Inf in the input if the underlying estimator does as well.

    Examples
    --------
    >>> from sklearn.feature_selection import SelectFromModel
    >>> from sklearn.linear_model import LogisticRegression
    >>> X = [[ 0.87, -1.34,  0.31 ],
    ...      [-2.79, -0.02, -0.85 ],
    ...      [-1.34, -0.48, -2.55 ],
    ...      [ 1.92,  1.48,  0.65 ]]
    >>> y = [0, 1, 0, 1]
    >>> selector = SelectFromModel(estimator=LogisticRegression()).fit(X, y)
    >>> selector.estimator_.coef_
    array([[-0.3252302 ,  0.83462377,  0.49750423]])
    >>> selector.threshold_
    0.55245...
    >>> selector.get_support()
    array([False,  True, False])
    >>> selector.transform(X)
    array([[-1.34],
           [-0.02],
           [-0.48],
           [ 1.48]])

    Using a callable to create a selector that can use no more than half
    of the input features.

    >>> def half_callable(X):
    ...     return round(len(X[0]) / 2)
    >>> half_selector = SelectFromModel(estimator=LogisticRegression(),
    ...                                 max_features=half_callable)
    >>> _ = half_selector.fit(X, y)
    >>> half_selector.max_features_
    2
    """

    _parameter_constraints: dict = {
        "estimator": [HasMethods("fit")],
        "threshold": [Interval(Real, None, None, closed="both"), str, None],
        "prefit": ["boolean"],
        "norm_order": [
            Interval(Integral, None, -1, closed="right"),
            Interval(Integral, 1, None, closed="left"),
            Options(Real, {np.inf, -np.inf}),
        ],
        "max_features": [Interval(Integral, 0, None, closed="left"), callable, None],
        "importance_getter": [str, callable],
    }

    def __init__(
        self,
        estimator,
        *,
        threshold=None,
        prefit=False,
        norm_order=1,
        max_features=None,
        importance_getter="auto",
    ):
        self.estimator = estimator
        self.threshold = threshold
        self.prefit = prefit
        self.importance_getter = importance_getter
        self.norm_order = norm_order
        self.max_features = max_features

    def _get_support_mask(self):
        estimator = getattr(self, "estimator_", self.estimator)
        max_features = getattr(self, "max_features_", self.max_features)

        if self.prefit:
            try:
                check_is_fitted(self.estimator)
            except NotFittedError as exc:
                raise NotFittedError(
                    "When `prefit=True`, `estimator` is expected to be a fitted "
                    "estimator."
                ) from exc
        if callable(max_features):
            # This branch is executed when `transform` is called directly and thus
            # `max_features_` is not set and we fallback using `self.max_features`
            # that is not validated
            raise NotFittedError(
                "When `prefit=True` and `max_features` is a callable, call `fit` "
                "before calling `transform`."
            )
        elif max_features is not None and not isinstance(max_features, Integral):
            raise ValueError(
                f"`max_features` must be an integer. Got `max_features={max_features}` "
                "instead."
            )

        scores = _get_feature_importances(
            estimator=estimator,
            getter=self.importance_getter,
            transform_func="norm",
            norm_order=self.norm_order,
        )
        threshold = _calculate_threshold(estimator, scores, self.threshold)
        if self.max_features is not None:
            mask = np.zeros_like(scores, dtype=bool)
            candidate_indices = np.argsort(-scores, kind="mergesort")[:max_features]
            mask[candidate_indices] = True
        else:
            mask = np.ones_like(scores, dtype=bool)
        mask[scores < threshold] = False
        return mask

    def _check_max_features(self, X):
        if self.max_features is not None:
            n_features = _num_features(X)

            if callable(self.max_features):
                max_features = self.max_features(X)
            else:  # int
                max_features = self.max_features

            check_scalar(
                max_features,
                "max_features",
                Integral,
                min_val=0,
                max_val=n_features,
            )
            self.max_features_ = max_features

    @_fit_context(
        # SelectFromModel.estimator is not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X, y=None, **fit_params):
        """Fit the SelectFromModel meta-transformer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,), default=None
            The target values (integers that correspond to classes in
            classification, real numbers in regression).

        **fit_params : dict
            Other estimator specific parameters.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self._check_max_features(X)

        if self.prefit:
            try:
                check_is_fitted(self.estimator)
            except NotFittedError as exc:
                raise NotFittedError(
                    "When `prefit=True`, `estimator` is expected to be a fitted "
                    "estimator."
                ) from exc
            self.estimator_ = deepcopy(self.estimator)
        else:
            self.estimator_ = clone(self.estimator)
            self.estimator_.fit(X, y, **fit_params)

        if hasattr(self.estimator_, "feature_names_in_"):
            self.feature_names_in_ = self.estimator_.feature_names_in_
        else:
            self._check_feature_names(X, reset=True)

        return self

    @property
    def threshold_(self):
        """Threshold value used for feature selection."""
        scores = _get_feature_importances(
            estimator=self.estimator_,
            getter=self.importance_getter,
            transform_func="norm",
            norm_order=self.norm_order,
        )
        return _calculate_threshold(self.estimator, scores, self.threshold)

    @available_if(_estimator_has("partial_fit"))
    @_fit_context(
        # SelectFromModel.estimator is not validated yet
        prefer_skip_nested_validation=False
    )
    def partial_fit(self, X, y=None, **fit_params):
        """Fit the SelectFromModel meta-transformer only once.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,), default=None
            The target values (integers that correspond to classes in
            classification, real numbers in regression).

        **fit_params : dict
            Other estimator specific parameters.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        first_call = not hasattr(self, "estimator_")

        if first_call:
            self._check_max_features(X)

        if self.prefit:
            if first_call:
                try:
                    check_is_fitted(self.estimator)
                except NotFittedError as exc:
                    raise NotFittedError(
                        "When `prefit=True`, `estimator` is expected to be a fitted "
                        "estimator."
                    ) from exc
                self.estimator_ = deepcopy(self.estimator)
            return self

        if first_call:
            self.estimator_ = clone(self.estimator)
        self.estimator_.partial_fit(X, y, **fit_params)

        if hasattr(self.estimator_, "feature_names_in_"):
            self.feature_names_in_ = self.estimator_.feature_names_in_
        else:
            self._check_feature_names(X, reset=first_call)

        return self

    @property
    def n_features_in_(self):
        """Number of features seen during `fit`."""
        # For consistency with other estimators we raise a AttributeError so
        # that hasattr() fails if the estimator isn't fitted.
        try:
            check_is_fitted(self)
        except NotFittedError as nfe:
            raise AttributeError(
                "{} object has no n_features_in_ attribute.".format(
                    self.__class__.__name__
                )
            ) from nfe

        return self.estimator_.n_features_in_

    def _more_tags(self):
        return {"allow_nan": _safe_tags(self.estimator, key="allow_nan")}
