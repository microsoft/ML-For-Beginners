"""Utilities to get the response values of a classifier or a regressor.

It allows to make uniform checks and validation.
"""
import numpy as np

from ..base import is_classifier
from .multiclass import type_of_target
from .validation import _check_response_method, check_is_fitted


def _process_predict_proba(*, y_pred, target_type, classes, pos_label):
    """Get the response values when the response method is `predict_proba`.

    This function process the `y_pred` array in the binary and multi-label cases.
    In the binary case, it selects the column corresponding to the positive
    class. In the multi-label case, it stacks the predictions if they are not
    in the "compressed" format `(n_samples, n_outputs)`.

    Parameters
    ----------
    y_pred : ndarray
        Output of `estimator.predict_proba`. The shape depends on the target type:

        - for binary classification, it is a 2d array of shape `(n_samples, 2)`;
        - for multiclass classification, it is a 2d array of shape
          `(n_samples, n_classes)`;
        - for multilabel classification, it is either a list of 2d arrays of shape
          `(n_samples, 2)` (e.g. `RandomForestClassifier` or `KNeighborsClassifier`) or
          an array of shape `(n_samples, n_outputs)` (e.g. `MLPClassifier` or
          `RidgeClassifier`).

    target_type : {"binary", "multiclass", "multilabel-indicator"}
        Type of the target.

    classes : ndarray of shape (n_classes,) or list of such arrays
        Class labels as reported by `estimator.classes_`.

    pos_label : int, float, bool or str
        Only used with binary and multiclass targets.

    Returns
    -------
    y_pred : ndarray of shape (n_samples,), (n_samples, n_classes) or \
            (n_samples, n_output)
        Compressed predictions format as requested by the metrics.
    """
    if target_type == "binary" and y_pred.shape[1] < 2:
        # We don't handle classifiers trained on a single class.
        raise ValueError(
            f"Got predict_proba of shape {y_pred.shape}, but need "
            "classifier with two classes."
        )

    if target_type == "binary":
        col_idx = np.flatnonzero(classes == pos_label)[0]
        return y_pred[:, col_idx]
    elif target_type == "multilabel-indicator":
        # Use a compress format of shape `(n_samples, n_output)`.
        # Only `MLPClassifier` and `RidgeClassifier` return an array of shape
        # `(n_samples, n_outputs)`.
        if isinstance(y_pred, list):
            # list of arrays of shape `(n_samples, 2)`
            return np.vstack([p[:, -1] for p in y_pred]).T
        else:
            # array of shape `(n_samples, n_outputs)`
            return y_pred

    return y_pred


def _process_decision_function(*, y_pred, target_type, classes, pos_label):
    """Get the response values when the response method is `decision_function`.

    This function process the `y_pred` array in the binary and multi-label cases.
    In the binary case, it inverts the sign of the score if the positive label
    is not `classes[1]`. In the multi-label case, it stacks the predictions if
    they are not in the "compressed" format `(n_samples, n_outputs)`.

    Parameters
    ----------
    y_pred : ndarray
        Output of `estimator.predict_proba`. The shape depends on the target type:

        - for binary classification, it is a 1d array of shape `(n_samples,)` where the
          sign is assuming that `classes[1]` is the positive class;
        - for multiclass classification, it is a 2d array of shape
          `(n_samples, n_classes)`;
        - for multilabel classification, it is a 2d array of shape `(n_samples,
          n_outputs)`.

    target_type : {"binary", "multiclass", "multilabel-indicator"}
        Type of the target.

    classes : ndarray of shape (n_classes,) or list of such arrays
        Class labels as reported by `estimator.classes_`.

    pos_label : int, float, bool or str
        Only used with binary and multiclass targets.

    Returns
    -------
    y_pred : ndarray of shape (n_samples,), (n_samples, n_classes) or \
            (n_samples, n_output)
        Compressed predictions format as requested by the metrics.
    """
    if target_type == "binary" and pos_label == classes[0]:
        return -1 * y_pred
    return y_pred


def _get_response_values(
    estimator,
    X,
    response_method,
    pos_label=None,
    return_response_method_used=False,
):
    """Compute the response values of a classifier, an outlier detector, or a regressor.

    The response values are predictions such that it follows the following shape:

    - for binary classification, it is a 1d array of shape `(n_samples,)`;
    - for multiclass classification, it is a 2d array of shape `(n_samples, n_classes)`;
    - for multilabel classification, it is a 2d array of shape `(n_samples, n_outputs)`;
    - for outlier detection, it is a 1d array of shape `(n_samples,)`;
    - for regression, it is a 1d array of shape `(n_samples,)`.

    If `estimator` is a binary classifier, also return the label for the
    effective positive class.

    This utility is used primarily in the displays and the scikit-learn scorers.

    .. versionadded:: 1.3

    Parameters
    ----------
    estimator : estimator instance
        Fitted classifier, outlier detector, or regressor or a
        fitted :class:`~sklearn.pipeline.Pipeline` in which the last estimator is a
        classifier, an outlier detector, or a regressor.

    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Input values.

    response_method : {"predict_proba", "predict_log_proba", "decision_function", \
            "predict"} or list of such str
        Specifies the response method to use get prediction from an estimator
        (i.e. :term:`predict_proba`, :term:`predict_log_proba`,
        :term:`decision_function` or :term:`predict`). Possible choices are:

        - if `str`, it corresponds to the name to the method to return;
        - if a list of `str`, it provides the method names in order of
          preference. The method returned corresponds to the first method in
          the list and which is implemented by `estimator`.

    pos_label : int, float, bool or str, default=None
        The class considered as the positive class when computing
        the metrics. If `None` and target is 'binary', `estimators.classes_[1]` is
        considered as the positive class.

    return_response_method_used : bool, default=False
        Whether to return the response method used to compute the response
        values.

        .. versionadded:: 1.4

    Returns
    -------
    y_pred : ndarray of shape (n_samples,), (n_samples, n_classes) or \
            (n_samples, n_outputs)
        Target scores calculated from the provided `response_method`
        and `pos_label`.

    pos_label : int, float, bool, str or None
        The class considered as the positive class when computing
        the metrics. Returns `None` if `estimator` is a regressor or an outlier
        detector.

    response_method_used : str
        The response method used to compute the response values. Only returned
        if `return_response_method_used` is `True`.

        .. versionadded:: 1.4

    Raises
    ------
    ValueError
        If `pos_label` is not a valid label.
        If the shape of `y_pred` is not consistent for binary classifier.
        If the response method can be applied to a classifier only and
        `estimator` is a regressor.
    """
    from sklearn.base import is_classifier, is_outlier_detector  # noqa

    if is_classifier(estimator):
        prediction_method = _check_response_method(estimator, response_method)
        classes = estimator.classes_
        target_type = type_of_target(classes)

        if target_type in ("binary", "multiclass"):
            if pos_label is not None and pos_label not in classes.tolist():
                raise ValueError(
                    f"pos_label={pos_label} is not a valid label: It should be "
                    f"one of {classes}"
                )
            elif pos_label is None and target_type == "binary":
                pos_label = classes[-1]

        y_pred = prediction_method(X)

        if prediction_method.__name__ in ("predict_proba", "predict_log_proba"):
            y_pred = _process_predict_proba(
                y_pred=y_pred,
                target_type=target_type,
                classes=classes,
                pos_label=pos_label,
            )
        elif prediction_method.__name__ == "decision_function":
            y_pred = _process_decision_function(
                y_pred=y_pred,
                target_type=target_type,
                classes=classes,
                pos_label=pos_label,
            )
    elif is_outlier_detector(estimator):
        prediction_method = _check_response_method(estimator, response_method)
        y_pred, pos_label = prediction_method(X), None
    else:  # estimator is a regressor
        if response_method != "predict":
            raise ValueError(
                f"{estimator.__class__.__name__} should either be a classifier to be "
                f"used with response_method={response_method} or the response_method "
                "should be 'predict'. Got a regressor with response_method="
                f"{response_method} instead."
            )
        prediction_method = estimator.predict
        y_pred, pos_label = prediction_method(X), None

    if return_response_method_used:
        return y_pred, pos_label, prediction_method.__name__
    return y_pred, pos_label


def _get_response_values_binary(estimator, X, response_method, pos_label=None):
    """Compute the response values of a binary classifier.

    Parameters
    ----------
    estimator : estimator instance
        Fitted classifier or a fitted :class:`~sklearn.pipeline.Pipeline`
        in which the last estimator is a binary classifier.

    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Input values.

    response_method : {'auto', 'predict_proba', 'decision_function'}
        Specifies whether to use :term:`predict_proba` or
        :term:`decision_function` as the target response. If set to 'auto',
        :term:`predict_proba` is tried first and if it does not exist
        :term:`decision_function` is tried next.

    pos_label : int, float, bool or str, default=None
        The class considered as the positive class when computing
        the metrics. By default, `estimators.classes_[1]` is
        considered as the positive class.

    Returns
    -------
    y_pred : ndarray of shape (n_samples,)
        Target scores calculated from the provided response_method
        and pos_label.

    pos_label : int, float, bool or str
        The class considered as the positive class when computing
        the metrics.
    """
    classification_error = "Expected 'estimator' to be a binary classifier."

    check_is_fitted(estimator)
    if not is_classifier(estimator):
        raise ValueError(
            classification_error + f" Got {estimator.__class__.__name__} instead."
        )
    elif len(estimator.classes_) != 2:
        raise ValueError(
            classification_error + f" Got {len(estimator.classes_)} classes instead."
        )

    if response_method == "auto":
        response_method = ["predict_proba", "decision_function"]

    return _get_response_values(
        estimator,
        X,
        response_method,
        pos_label=pos_label,
    )
