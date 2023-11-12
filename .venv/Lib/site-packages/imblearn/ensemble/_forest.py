"""Forest classifiers trained on balanced boostrasp samples."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

import numbers
from copy import deepcopy
from warnings import warn

import numpy as np
import sklearn
from numpy import float32 as DTYPE
from numpy import float64 as DOUBLE
from scipy.sparse import issparse
from sklearn.base import clone, is_classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble._base import _set_random_states
from sklearn.ensemble._forest import (
    _generate_unsampled_indices,
    _get_n_samples_bootstrap,
    _parallel_build_trees,
)
from sklearn.exceptions import DataConversionWarning
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import _safe_indexing, check_random_state, parse_version
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _check_sample_weight

try:
    # scikit-learn >= 1.2
    from sklearn.utils.parallel import Parallel, delayed
except (ImportError, ModuleNotFoundError):
    from joblib import Parallel
    from sklearn.utils.fixes import delayed

from ..base import _ParamsValidationMixin
from ..pipeline import make_pipeline
from ..under_sampling import RandomUnderSampler
from ..utils import Substitution
from ..utils._docstring import _n_jobs_docstring, _random_state_docstring
from ..utils._param_validation import Hidden, Interval, StrOptions
from ..utils._validation import check_sampling_strategy
from ..utils.fixes import _fit_context
from ._common import _random_forest_classifier_parameter_constraints

MAX_INT = np.iinfo(np.int32).max
sklearn_version = parse_version(sklearn.__version__)


def _local_parallel_build_trees(
    sampler,
    tree,
    bootstrap,
    X,
    y,
    sample_weight,
    tree_idx,
    n_trees,
    verbose=0,
    class_weight=None,
    n_samples_bootstrap=None,
    forest=None,
):
    # resample before to fit the tree
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    if sample_weight is not None:
        sample_weight = _safe_indexing(sample_weight, sampler.sample_indices_)
    if _get_n_samples_bootstrap is not None:
        n_samples_bootstrap = min(n_samples_bootstrap, X_resampled.shape[0])

    if sklearn_version >= parse_version("1.1"):
        tree = _parallel_build_trees(
            tree,
            bootstrap,
            X_resampled,
            y_resampled,
            sample_weight,
            tree_idx,
            n_trees,
            verbose=verbose,
            class_weight=class_weight,
            n_samples_bootstrap=n_samples_bootstrap,
        )
    else:
        # TODO: remove when the minimum version of scikit-learn supported is 1.1
        tree = _parallel_build_trees(
            tree,
            forest,
            X_resampled,
            y_resampled,
            sample_weight,
            tree_idx,
            n_trees,
            verbose=verbose,
            class_weight=class_weight,
            n_samples_bootstrap=n_samples_bootstrap,
        )
    return sampler, tree


@Substitution(
    n_jobs=_n_jobs_docstring,
    random_state=_random_state_docstring,
)
class BalancedRandomForestClassifier(_ParamsValidationMixin, RandomForestClassifier):
    """A balanced random forest classifier.

    A balanced random forest randomly under-samples each bootstrap sample to
    balance it.

    Read more in the :ref:`User Guide <forest>`.

    .. versionadded:: 0.4

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.

    criterion : {{"gini", "entropy"}}, default="gini"
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.
        Note: this parameter is tree-specific.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a percentage and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node:

        - If int, then consider ``min_samples_leaf`` as the minimum number.
        - If float, then ``min_samples_leaf`` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : {{"auto", "sqrt", "log2"}}, int, float, or None, \
            default="sqrt"
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a percentage and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)` (same as "auto").
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_leaf_nodes : int, default=None
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.
        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees.

    oob_score : bool, default=False
        Whether to use out-of-bag samples to estimate
        the generalization accuracy.

    sampling_strategy : float, str, dict, callable, default="auto"
        Sampling information to sample the data set.

        - When ``float``, it corresponds to the desired ratio of the number of
          samples in the minority class over the number of samples in the
          majority class after resampling. Therefore, the ratio is expressed as
          :math:`\\alpha_{{us}} = N_{{m}} / N_{{rM}}` where :math:`N_{{m}}` is the
          number of samples in the minority class and
          :math:`N_{{rM}}` is the number of samples in the majority class
          after resampling.

          .. warning::
             ``float`` is only available for **binary** classification. An
             error is raised for multi-class classification.

        - When ``str``, specify the class targeted by the resampling. The
          number of samples in the different classes will be equalized.
          Possible choices are:

            ``'majority'``: resample only the majority class;

            ``'not minority'``: resample all classes but the minority class;

            ``'not majority'``: resample all classes but the majority class;

            ``'all'``: resample all classes;

            ``'auto'``: equivalent to ``'not minority'``.

        - When ``dict``, the keys correspond to the targeted classes. The
          values correspond to the desired number of samples for each targeted
          class.

        - When callable, function taking ``y`` and returns a ``dict``. The keys
          correspond to the targeted classes. The values correspond to the
          desired number of samples for each class.

        .. versionchanged:: 0.11
           The default of `sampling_strategy` will change from `"auto"` to
           `"all"` in version 0.13. This forces to use a bootstrap of the
           minority class as proposed in [1]_.

    replacement : bool, default=False
        Whether or not to sample randomly with replacement or not.

        .. versionchanged:: 0.11
           The default of `replacement` will change from `False` to `True` in
           version 0.13. This forces to use a bootstrap of the
           minority class and draw with replacement as proposed in [1]_.

    {n_jobs}

    {random_state}

    verbose : int, default=0
        Controls the verbosity of the tree building process.

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest.

    class_weight : dict, list of dicts, {{"balanced", "balanced_subsample"}}, \
            default=None
        Weights associated with classes in the form dictionary with the key
        being the class_label and the value the weight.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.
        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{{0: 1, 1: 1}}, {{0: 1, 1: 5}}, {{0: 1, 1: 1}}, {{0: 1, 1: 1}}]
        instead of [{{1:1}}, {{2:5}}, {{3:1}}, {{4:1}}].
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``
        The "balanced_subsample" mode is the same as "balanced" except that
        weights are computed based on the bootstrap sample for every tree
        grown.
        For multi-output, the weights of each column of y will be multiplied.
        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed.

        .. versionadded:: 0.6
           Added in `scikit-learn` in 0.22

    max_samples : int or float, default=None
        If bootstrap is True, the number of samples to draw from X
        to train each base estimator.
            - If None (default), then draw `X.shape[0]` samples.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples. Thus,
              `max_samples` should be in the interval `(0, 1)`.
        Be aware that the final number samples used will be the minimum between
        the number of samples given in `max_samples` and the number of samples
        obtained after resampling.

        .. versionadded:: 0.6
           Added in `scikit-learn` in 0.22

    Attributes
    ----------
    estimator_ : :class:`~sklearn.tree.DecisionTreeClassifier` instance
        The child estimator template used to create the collection of fitted
        sub-estimators.

        .. versionadded:: 0.10

    base_estimator_ : :class:`~sklearn.tree.DecisionTreeClassifier` instance
        The child estimator template used to create the collection of fitted
        sub-estimators.

        .. deprecated:: 1.2
           `base_estimator_` is deprecated in `scikit-learn` 1.2 and will be
           removed in 1.4. Use `estimator_` instead. When the minimum version
           of `scikit-learn` supported by `imbalanced-learn` will reach 1.4,
           this attribute will be removed.

    estimators_ : list of :class:`~sklearn.tree.DecisionTreeClassifier`
        The collection of fitted sub-estimators.

    base_sampler_ : :class:`~imblearn.under_sampling.RandomUnderSampler`
        The base sampler used to construct the subsequent list of samplers.

    samplers_ : list of :class:`~imblearn.under_sampling.RandomUnderSampler`
        The collection of fitted samplers.

    pipelines_ : list of Pipeline.
        The collection of fitted pipelines (samplers + trees).

    classes_ : ndarray of shape (n_classes,) or a list of such arrays
        The classes labels (single output problem), or a list of arrays of
        class labels (multi-output problem).

    n_classes_ : int or list
        The number of classes (single output problem), or a list containing the
        number of classes for each output (multi-output problem).

    n_features_ : int
        The number of features when `fit` is performed.

        .. deprecated:: 1.0
           `n_features_` is deprecated in `scikit-learn` 1.0 and will be removed
           in version 1.2. When the minimum version of `scikit-learn` supported
           by `imbalanced-learn` will reach 1.2, this attribute will be removed.

    n_features_in_ : int
        Number of features in the input dataset.

        .. versionadded:: 0.9

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during `fit`. Defined only when `X` has feature
        names that are all strings.

        .. versionadded:: 0.9

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    feature_importances_ : ndarray of shape (n_features,)
        The feature importances (the higher, the more important the feature).

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.

    oob_decision_function_ : ndarray of shape (n_samples, n_classes)
        Decision function computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_decision_function_` might contain NaN.

    See Also
    --------
    BalancedBaggingClassifier : Bagging classifier for which each base
        estimator is trained on a balanced bootstrap.

    EasyEnsembleClassifier : Ensemble of AdaBoost classifier trained on
        balanced bootstraps.

    RUSBoostClassifier : AdaBoost classifier were each bootstrap is balanced
        using random-under sampling at each round of boosting.

    References
    ----------
    .. [1] Chen, Chao, Andy Liaw, and Leo Breiman. "Using random forest to
       learn imbalanced data." University of California, Berkeley 110 (2004):
       1-12.

    Examples
    --------
    >>> from imblearn.ensemble import BalancedRandomForestClassifier
    >>> from sklearn.datasets import make_classification
    >>>
    >>> X, y = make_classification(n_samples=1000, n_classes=3,
    ...                            n_informative=4, weights=[0.2, 0.3, 0.5],
    ...                            random_state=0)
    >>> clf = BalancedRandomForestClassifier(
    ...     sampling_strategy="all", replacement=True, max_depth=2, random_state=0)
    >>> clf.fit(X, y)
    BalancedRandomForestClassifier(...)
    >>> print(clf.feature_importances_)
    [...]
    >>> print(clf.predict([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
    [1]
    """

    # make a deepcopy to not modify the original dictionary
    if sklearn_version >= parse_version("1.3"):
        _parameter_constraints = deepcopy(RandomForestClassifier._parameter_constraints)
    else:
        _parameter_constraints = deepcopy(
            _random_forest_classifier_parameter_constraints
        )

    _parameter_constraints.update(
        {
            "sampling_strategy": [
                Interval(numbers.Real, 0, 1, closed="right"),
                StrOptions({"auto", "majority", "not minority", "not majority", "all"}),
                dict,
                callable,
                Hidden(StrOptions({"warn"})),
            ],
            "replacement": ["boolean", Hidden(StrOptions({"warn"}))],
        }
    )

    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        sampling_strategy="warn",
        replacement="warn",
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
    ):
        super().__init__(
            criterion=criterion,
            max_depth=max_depth,
            n_estimators=n_estimators,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
        )

        self.sampling_strategy = sampling_strategy
        self.replacement = replacement

    def _validate_estimator(self, default=DecisionTreeClassifier()):
        """Check the estimator and the n_estimator attribute, set the
        `estimator_` attribute."""
        if hasattr(self, "estimator"):
            base_estimator = self.estimator
        else:
            base_estimator = self.base_estimator

        if base_estimator is not None:
            self._estimator = clone(base_estimator)
        else:
            self._estimator = clone(default)

        try:
            # scikit-learn < 1.2
            self.base_estimator_ = self._estimator
        except AttributeError:
            pass

        self.base_sampler_ = RandomUnderSampler(
            sampling_strategy=self._sampling_strategy,
            replacement=self._replacement,
        )

    def _make_sampler_estimator(self, random_state=None):
        """Make and configure a copy of the `base_estimator_` attribute.
        Warning: This method should be used to properly instantiate new
        sub-estimators.
        """
        estimator = clone(self._estimator)
        estimator.set_params(**{p: getattr(self, p) for p in self.estimator_params})
        sampler = clone(self.base_sampler_)

        if random_state is not None:
            _set_random_states(estimator, random_state)
            _set_random_states(sampler, random_state)

        return estimator, sampler

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        """Build a forest of trees from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples,)
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        Returns
        -------
        self : object
            The fitted instance.
        """
        self._validate_params()
        # TODO: remove in 0.13
        if self.sampling_strategy == "warn":
            warn(
                "The default of `sampling_strategy` will change from `'auto'` to "
                "`'all'` in version 0.13. This change will follow the implementation "
                "proposed in the original paper. Set to `'all'` to silence this "
                "warning and adopt the future behaviour.",
                FutureWarning,
            )
            self._sampling_strategy = "auto"
        else:
            self._sampling_strategy = self.sampling_strategy

        if self.replacement == "warn":
            warn(
                "The default of `replacement` will change from `False` to "
                "`True` in version 0.13. This change will follow the implementation "
                "proposed in the original paper. Set to `True` to silence this "
                "warning and adopt the future behaviour.",
                FutureWarning,
            )
            self._replacement = False
        else:
            self._replacement = self.replacement

        # Validate or convert input data
        if issparse(y):
            raise ValueError("sparse multilabel-indicator for y is not supported.")
        X, y = self._validate_data(
            X, y, multi_output=True, accept_sparse="csc", dtype=DTYPE
        )
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)
        self._n_features = X.shape[1]

        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warn(
                "A column-vector y was passed when a 1d array was"
                " expected. Please change the shape of y to "
                "(n_samples,), for example using ravel().",
                DataConversionWarning,
                stacklevel=2,
            )

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        y_encoded, expanded_class_weight = self._validate_y_class_weight(y)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y_encoded = np.ascontiguousarray(y_encoded, dtype=DOUBLE)

        if isinstance(self._sampling_strategy, dict):
            self._sampling_strategy = {
                np.where(self.classes_[0] == key)[0][0]: value
                for key, value in check_sampling_strategy(
                    self.sampling_strategy,
                    y,
                    "under-sampling",
                ).items()
            }
        else:
            self._sampling_strategy = self._sampling_strategy

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        # Get bootstrap sample size
        n_samples_bootstrap = _get_n_samples_bootstrap(
            n_samples=X.shape[0], max_samples=self.max_samples
        )

        # Check parameters
        self._validate_estimator()

        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available if bootstrap=True")

        random_state = check_random_state(self.random_state)

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []
            self.samplers_ = []
            self.pipelines_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError(
                "n_estimators=%d must be larger or equal to "
                "len(estimators_)=%d when warm_start==True"
                % (self.n_estimators, len(self.estimators_))
            )

        elif n_more_estimators == 0:
            warn(
                "Warm-start fitting without increasing n_estimators does not "
                "fit new trees."
            )
        else:
            if self.warm_start and len(self.estimators_) > 0:
                # We draw from the random state to get the random state we
                # would have got if we hadn't used a warm_start.
                random_state.randint(MAX_INT, size=len(self.estimators_))

            trees = []
            samplers = []
            for _ in range(n_more_estimators):
                tree, sampler = self._make_sampler_estimator(random_state=random_state)
                trees.append(tree)
                samplers.append(sampler)

            # Parallel loop: we prefer the threading backend as the Cython code
            # for fitting the trees is internally releasing the Python GIL
            # making threading more efficient than multiprocessing in
            # that case. However, we respect any parallel_backend contexts set
            # at a higher level, since correctness does not rely on using
            # threads.
            samplers_trees = Parallel(
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                prefer="threads",
            )(
                delayed(_local_parallel_build_trees)(
                    s,
                    t,
                    self.bootstrap,
                    X,
                    y_encoded,
                    sample_weight,
                    i,
                    len(trees),
                    verbose=self.verbose,
                    class_weight=self.class_weight,
                    n_samples_bootstrap=n_samples_bootstrap,
                    forest=self,
                )
                for i, (s, t) in enumerate(zip(samplers, trees))
            )
            samplers, trees = zip(*samplers_trees)

            # Collect newly grown trees
            self.estimators_.extend(trees)
            self.samplers_.extend(samplers)

            # Create pipeline with the fitted samplers and trees
            self.pipelines_.extend(
                [
                    make_pipeline(deepcopy(s), deepcopy(t))
                    for s, t in zip(samplers, trees)
                ]
            )

        if self.oob_score:
            y_type = type_of_target(y)
            if y_type in ("multiclass-multioutput", "unknown"):
                # FIXME: we could consider to support multiclass-multioutput if
                # we introduce or reuse a constructor parameter (e.g.
                # oob_score) allowing our user to pass a callable defining the
                # scoring strategy on OOB sample.
                raise ValueError(
                    "The type of target cannot be used to compute OOB "
                    f"estimates. Got {y_type} while only the following are "
                    "supported: continuous, continuous-multioutput, binary, "
                    "multiclass, multilabel-indicator."
                )
            self._set_oob_score_and_attributes(X, y_encoded)

        # Decapsulate classes_ attributes
        if hasattr(self, "classes_") and self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        return self

    def _set_oob_score_and_attributes(self, X, y):
        """Compute and set the OOB score and attributes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.
        y : ndarray of shape (n_samples, n_outputs)
            The target matrix.
        """
        self.oob_decision_function_ = self._compute_oob_predictions(X, y)
        if self.oob_decision_function_.shape[-1] == 1:
            # drop the n_outputs axis if there is a single output
            self.oob_decision_function_ = self.oob_decision_function_.squeeze(axis=-1)
        from sklearn.metrics import accuracy_score

        self.oob_score_ = accuracy_score(
            y, np.argmax(self.oob_decision_function_, axis=1)
        )

    def _compute_oob_predictions(self, X, y):
        """Compute and set the OOB score.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.
        y : ndarray of shape (n_samples, n_outputs)
            The target matrix.

        Returns
        -------
        oob_pred : ndarray of shape (n_samples, n_classes, n_outputs) or \
                (n_samples, 1, n_outputs)
            The OOB predictions.
        """
        # Prediction requires X to be in CSR format
        if issparse(X):
            X = X.tocsr()

        n_samples = y.shape[0]
        n_outputs = self.n_outputs_

        if is_classifier(self) and hasattr(self, "n_classes_"):
            # n_classes_ is a ndarray at this stage
            # all the supported type of target will have the same number of
            # classes in all outputs
            oob_pred_shape = (n_samples, self.n_classes_[0], n_outputs)
        else:
            # for regression, n_classes_ does not exist and we create an empty
            # axis to be consistent with the classification case and make
            # the array operations compatible with the 2 settings
            oob_pred_shape = (n_samples, 1, n_outputs)

        oob_pred = np.zeros(shape=oob_pred_shape, dtype=np.float64)
        n_oob_pred = np.zeros((n_samples, n_outputs), dtype=np.int64)

        for sampler, estimator in zip(self.samplers_, self.estimators_):
            X_resample = X[sampler.sample_indices_]
            y_resample = y[sampler.sample_indices_]

            n_sample_subset = y_resample.shape[0]
            n_samples_bootstrap = _get_n_samples_bootstrap(
                n_sample_subset, self.max_samples
            )

            unsampled_indices = _generate_unsampled_indices(
                estimator.random_state, n_sample_subset, n_samples_bootstrap
            )

            y_pred = self._get_oob_predictions(
                estimator, X_resample[unsampled_indices, :]
            )

            indices = sampler.sample_indices_[unsampled_indices]
            oob_pred[indices, ...] += y_pred
            n_oob_pred[indices, :] += 1

        for k in range(n_outputs):
            if (n_oob_pred == 0).any():
                warn(
                    "Some inputs do not have OOB scores. This probably means "
                    "too few trees were used to compute any reliable OOB "
                    "estimates.",
                    UserWarning,
                )
                n_oob_pred[n_oob_pred == 0] = 1
            oob_pred[..., k] /= n_oob_pred[..., [k]]

        return oob_pred

    # TODO: remove when supporting scikit-learn>=1.4
    @property
    def estimator_(self):
        """Estimator used to grow the ensemble."""
        return self._estimator

    # TODO: remove when supporting scikit-learn>=1.2
    @property
    def n_features_(self):
        """Number of features when ``fit`` is performed."""
        warn(
            "`n_features_` was deprecated in scikit-learn 1.0. This attribute will "
            "not be accessible when the minimum supported version of scikit-learn "
            "is 1.2.",
            FutureWarning,
        )
        return self.n_features_in_

    def _more_tags(self):
        return {
            "multioutput": False,
            "multilabel": False,
        }
