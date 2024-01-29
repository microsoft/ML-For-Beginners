import copy
import numbers
from copy import deepcopy

import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble._base import _set_random_states
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import _safe_indexing, parse_version
from sklearn.utils.validation import has_fit_parameter

from ..base import _ParamsValidationMixin
from ..pipeline import make_pipeline
from ..under_sampling import RandomUnderSampler
from ..under_sampling.base import BaseUnderSampler
from ..utils import Substitution, check_target_type
from ..utils._docstring import _random_state_docstring
from ..utils._param_validation import Interval, StrOptions
from ..utils.fixes import _fit_context
from ._common import _adaboost_classifier_parameter_constraints

sklearn_version = parse_version(sklearn.__version__)


@Substitution(
    sampling_strategy=BaseUnderSampler._sampling_strategy_docstring,
    random_state=_random_state_docstring,
)
class RUSBoostClassifier(_ParamsValidationMixin, AdaBoostClassifier):
    """Random under-sampling integrated in the learning of AdaBoost.

    During learning, the problem of class balancing is alleviated by random
    under-sampling the sample at each iteration of the boosting algorithm.

    Read more in the :ref:`User Guide <boosting>`.

    .. versionadded:: 0.4

    Parameters
    ----------
    estimator : estimator object, default=None
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required, as well as proper
        ``classes_`` and ``n_classes_`` attributes. If ``None``, then
        the base estimator is ``DecisionTreeClassifier(max_depth=1)``.

        .. versionadded:: 0.12

    n_estimators : int, default=50
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.

    learning_rate : float, default=1.0
        Learning rate shrinks the contribution of each classifier by
        ``learning_rate``. There is a trade-off between ``learning_rate`` and
        ``n_estimators``.

    algorithm : {{'SAMME', 'SAMME.R'}}, default='SAMME.R'
        If 'SAMME.R' then use the SAMME.R real boosting algorithm.
        ``base_estimator`` must support calculation of class probabilities.
        If 'SAMME' then use the SAMME discrete boosting algorithm.
        The SAMME.R algorithm typically converges faster than SAMME,
        achieving a lower test error with fewer boosting iterations.

        .. deprecated:: 0.12
            `"SAMME.R"` is deprecated and will be removed in version 0.14.
            '"SAMME"' will become the default.

    {sampling_strategy}

    replacement : bool, default=False
        Whether or not to sample randomly with replacement or not.

    {random_state}

    Attributes
    ----------
    estimator_ : estimator
        The base estimator from which the ensemble is grown.

        .. versionadded:: 0.10

    estimators_ : list of classifiers
        The collection of fitted sub-estimators.

    base_sampler_ : :class:`~imblearn.under_sampling.RandomUnderSampler`
        The base sampler used to generate the subsequent samplers.

    samplers_ : list of :class:`~imblearn.under_sampling.RandomUnderSampler`
        The collection of fitted samplers.

    pipelines_ : list of Pipeline
        The collection of fitted pipelines (samplers + trees).

    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    n_classes_ : int
        The number of classes.

    estimator_weights_ : ndarray of shape (n_estimator,)
        Weights for each estimator in the boosted ensemble.

    estimator_errors_ : ndarray of shape (n_estimator,)
        Classification error for each estimator in the boosted
        ensemble.

    feature_importances_ : ndarray of shape (n_features,)
        The feature importances if supported by the ``base_estimator``.

    n_features_in_ : int
        Number of features in the input dataset.

        .. versionadded:: 0.9

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during `fit`. Defined only when `X` has feature
        names that are all strings.

        .. versionadded:: 0.9

    See Also
    --------
    BalancedBaggingClassifier : Bagging classifier for which each base
        estimator is trained on a balanced bootstrap.

    BalancedRandomForestClassifier : Random forest applying random-under
        sampling to balance the different bootstraps.

    EasyEnsembleClassifier : Ensemble of AdaBoost classifier trained on
        balanced bootstraps.

    References
    ----------
    .. [1] Seiffert, C., Khoshgoftaar, T. M., Van Hulse, J., & Napolitano, A.
       "RUSBoost: A hybrid approach to alleviating class imbalance." IEEE
       Transactions on Systems, Man, and Cybernetics-Part A: Systems and Humans
       40.1 (2010): 185-197.

    Examples
    --------
    >>> from imblearn.ensemble import RUSBoostClassifier
    >>> from sklearn.datasets import make_classification
    >>>
    >>> X, y = make_classification(n_samples=1000, n_classes=3,
    ...                            n_informative=4, weights=[0.2, 0.3, 0.5],
    ...                            random_state=0)
    >>> clf = RUSBoostClassifier(random_state=0)
    >>> clf.fit(X, y)
    RUSBoostClassifier(...)
    >>> clf.predict(X)
    array([...])
    """

    # make a deepcopy to not modify the original dictionary
    if sklearn_version >= parse_version("1.4"):
        _parameter_constraints = copy.deepcopy(
            AdaBoostClassifier._parameter_constraints
        )
    else:
        _parameter_constraints = copy.deepcopy(
            _adaboost_classifier_parameter_constraints
        )

    _parameter_constraints.update(
        {
            "sampling_strategy": [
                Interval(numbers.Real, 0, 1, closed="right"),
                StrOptions({"auto", "majority", "not minority", "not majority", "all"}),
                dict,
                callable,
            ],
            "replacement": ["boolean"],
        }
    )
    # TODO: remove when minimum supported version of scikit-learn is 1.4
    if "base_estimator" in _parameter_constraints:
        del _parameter_constraints["base_estimator"]

    def __init__(
        self,
        estimator=None,
        *,
        n_estimators=50,
        learning_rate=1.0,
        algorithm="SAMME.R",
        sampling_strategy="auto",
        replacement=False,
        random_state=None,
    ):
        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            algorithm=algorithm,
            random_state=random_state,
        )
        self.estimator = estimator
        self.sampling_strategy = sampling_strategy
        self.replacement = replacement

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y, sample_weight=None):
        """Build a boosted classifier from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.

        y : array-like of shape (n_samples,)
            The target values (class labels).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, the sample weights are initialized to
            ``1 / n_samples``.

        Returns
        -------
        self : object
            Returns self.
        """
        self._validate_params()
        check_target_type(y)
        self.samplers_ = []
        self.pipelines_ = []
        super().fit(X, y, sample_weight)
        return self

    def _validate_estimator(self):
        """Check the estimator and the n_estimator attribute.

        Sets the `estimator_` attributes.
        """
        default = DecisionTreeClassifier(max_depth=1)
        if self.estimator is not None:
            self.estimator_ = clone(self.estimator)
        else:
            self.estimator_ = clone(default)

        #  SAMME-R requires predict_proba-enabled estimators
        if self.algorithm == "SAMME.R":
            if not hasattr(self.estimator_, "predict_proba"):
                raise TypeError(
                    "AdaBoostClassifier with algorithm='SAMME.R' requires "
                    "that the weak learner supports the calculation of class "
                    "probabilities with a predict_proba method.\n"
                    "Please change the base estimator or set "
                    "algorithm='SAMME' instead."
                )
        if not has_fit_parameter(self.estimator_, "sample_weight"):
            raise ValueError(
                f"{self.estimator_.__class__.__name__} doesn't support sample_weight."
            )

        self.base_sampler_ = RandomUnderSampler(
            sampling_strategy=self.sampling_strategy,
            replacement=self.replacement,
        )

    def _make_sampler_estimator(self, append=True, random_state=None):
        """Make and configure a copy of the `base_estimator_` attribute.
        Warning: This method should be used to properly instantiate new
        sub-estimators.
        """
        estimator = clone(self.estimator_)
        estimator.set_params(**{p: getattr(self, p) for p in self.estimator_params})
        sampler = clone(self.base_sampler_)

        if random_state is not None:
            _set_random_states(estimator, random_state)
            _set_random_states(sampler, random_state)

        if append:
            self.estimators_.append(estimator)
            self.samplers_.append(sampler)
            self.pipelines_.append(
                make_pipeline(deepcopy(sampler), deepcopy(estimator))
            )

        return estimator, sampler

    def _boost_real(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost using the SAMME.R real algorithm."""
        estimator, sampler = self._make_sampler_estimator(random_state=random_state)

        X_res, y_res = sampler.fit_resample(X, y)
        sample_weight_res = _safe_indexing(sample_weight, sampler.sample_indices_)
        estimator.fit(X_res, y_res, sample_weight=sample_weight_res)

        y_predict_proba = estimator.predict_proba(X)

        if iboost == 0:
            self.classes_ = getattr(estimator, "classes_", None)
            self.n_classes_ = len(self.classes_)

        y_predict = self.classes_.take(np.argmax(y_predict_proba, axis=1), axis=0)

        # Instances incorrectly classified
        incorrect = y_predict != y

        # Error fraction
        estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))

        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1.0, 0.0

        # Construct y coding as described in Zhu et al [2]:
        #
        #    y_k = 1 if c == k else -1 / (K - 1)
        #
        # where K == n_classes_ and c, k in [0, K) are indices along the second
        # axis of the y coding with c being the index corresponding to the true
        # class label.
        n_classes = self.n_classes_
        classes = self.classes_
        y_codes = np.array([-1.0 / (n_classes - 1), 1.0])
        y_coding = y_codes.take(classes == y[:, np.newaxis])

        # Displace zero probabilities so the log is defined.
        # Also fix negative elements which may occur with
        # negative sample weights.
        proba = y_predict_proba  # alias for readability
        np.clip(proba, np.finfo(proba.dtype).eps, None, out=proba)

        # Boost weight using multi-class AdaBoost SAMME.R alg
        estimator_weight = (
            -1.0
            * self.learning_rate
            * ((n_classes - 1.0) / n_classes)
            * (y_coding * np.log(y_predict_proba)).sum(axis=1)
        )

        # Only boost the weights if it will fit again
        if not iboost == self.n_estimators - 1:
            # Only boost positive weights
            sample_weight *= np.exp(
                estimator_weight * ((sample_weight > 0) | (estimator_weight < 0))
            )

        return sample_weight, 1.0, estimator_error

    def _boost_discrete(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost using the SAMME discrete algorithm."""
        estimator, sampler = self._make_sampler_estimator(random_state=random_state)

        X_res, y_res = sampler.fit_resample(X, y)
        sample_weight_res = _safe_indexing(sample_weight, sampler.sample_indices_)
        estimator.fit(X_res, y_res, sample_weight=sample_weight_res)

        y_predict = estimator.predict(X)

        if iboost == 0:
            self.classes_ = getattr(estimator, "classes_", None)
            self.n_classes_ = len(self.classes_)

        # Instances incorrectly classified
        incorrect = y_predict != y

        # Error fraction
        estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))

        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1.0, 0.0

        n_classes = self.n_classes_

        # Stop if the error is at least as bad as random guessing
        if estimator_error >= 1.0 - (1.0 / n_classes):
            self.estimators_.pop(-1)
            self.samplers_.pop(-1)
            self.pipelines_.pop(-1)
            if len(self.estimators_) == 0:
                raise ValueError(
                    "BaseClassifier in AdaBoostClassifier "
                    "ensemble is worse than random, ensemble "
                    "can not be fit."
                )
            return None, None, None

        # Boost weight using multi-class AdaBoost SAMME alg
        estimator_weight = self.learning_rate * (
            np.log((1.0 - estimator_error) / estimator_error) + np.log(n_classes - 1.0)
        )

        # Only boost the weights if I will fit again
        if not iboost == self.n_estimators - 1:
            # Only boost positive weights
            sample_weight *= np.exp(estimator_weight * incorrect * (sample_weight > 0))

        return sample_weight, estimator_weight, estimator_error
