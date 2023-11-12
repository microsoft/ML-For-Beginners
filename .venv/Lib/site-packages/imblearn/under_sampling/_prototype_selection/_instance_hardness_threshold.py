"""Class to perform under-sampling based on the instance hardness
threshold."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Dayvid Oliveira
#          Christos Aridas
# License: MIT

import numbers
from collections import Counter

import numpy as np
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble._base import _set_random_states
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.utils import _safe_indexing, check_random_state

from ...utils import Substitution
from ...utils._docstring import _n_jobs_docstring, _random_state_docstring
from ...utils._param_validation import HasMethods
from ..base import BaseUnderSampler


@Substitution(
    sampling_strategy=BaseUnderSampler._sampling_strategy_docstring,
    n_jobs=_n_jobs_docstring,
    random_state=_random_state_docstring,
)
class InstanceHardnessThreshold(BaseUnderSampler):
    """Undersample based on the instance hardness threshold.

    Read more in the :ref:`User Guide <instance_hardness_threshold>`.

    Parameters
    ----------
    estimator : estimator object, default=None
        Classifier to be used to estimate instance hardness of the samples.
        This classifier should implement `predict_proba`.

    {sampling_strategy}

    {random_state}

    cv : int, default=5
        Number of folds to be used when estimating samples' instance hardness.

    {n_jobs}

    Attributes
    ----------
    sampling_strategy_ : dict
        Dictionary containing the information to sample the dataset. The keys
        corresponds to the class labels from which to sample and the values
        are the number of samples to sample.

    estimator_ : estimator object
        The validated classifier used to estimate the instance hardness of the samples.

    sample_indices_ : ndarray of shape (n_new_samples,)
        Indices of the samples selected.

        .. versionadded:: 0.4

    n_features_in_ : int
        Number of features in the input dataset.

        .. versionadded:: 0.9

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during `fit`. Defined only when `X` has feature
        names that are all strings.

        .. versionadded:: 0.10

    See Also
    --------
    NearMiss : Undersample based on near-miss search.

    RandomUnderSampler : Random under-sampling.

    Notes
    -----
    The method is based on [1]_.

    Supports multi-class resampling. A one-vs.-rest scheme is used when
    sampling a class as proposed in [1]_.

    References
    ----------
    .. [1] D. Smith, Michael R., Tony Martinez, and Christophe Giraud-Carrier.
       "An instance level analysis of data complexity." Machine learning
       95.2 (2014): 225-256.

    Examples
    --------
    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.under_sampling import InstanceHardnessThreshold
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> iht = InstanceHardnessThreshold(random_state=42)
    >>> X_res, y_res = iht.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{1: 5..., 0: 100}})
    """

    _parameter_constraints: dict = {
        **BaseUnderSampler._parameter_constraints,
        "estimator": [
            HasMethods(["fit", "predict_proba"]),
            None,
        ],
        "cv": ["cv_object"],
        "n_jobs": [numbers.Integral, None],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        *,
        estimator=None,
        sampling_strategy="auto",
        random_state=None,
        cv=5,
        n_jobs=None,
    ):
        super().__init__(sampling_strategy=sampling_strategy)
        self.random_state = random_state
        self.estimator = estimator
        self.cv = cv
        self.n_jobs = n_jobs

    def _validate_estimator(self, random_state):
        """Private function to create the classifier"""

        if (
            self.estimator is not None
            and isinstance(self.estimator, ClassifierMixin)
            and hasattr(self.estimator, "predict_proba")
        ):
            self.estimator_ = clone(self.estimator)
            _set_random_states(self.estimator_, random_state)

        elif self.estimator is None:
            self.estimator_ = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )

    def _fit_resample(self, X, y):
        random_state = check_random_state(self.random_state)
        self._validate_estimator(random_state)

        target_stats = Counter(y)
        skf = StratifiedKFold(
            n_splits=self.cv,
            shuffle=True,
            random_state=random_state,
        )
        probabilities = cross_val_predict(
            self.estimator_,
            X,
            y,
            cv=skf,
            n_jobs=self.n_jobs,
            method="predict_proba",
        )
        probabilities = probabilities[range(len(y)), y]

        idx_under = np.empty((0,), dtype=int)

        for target_class in np.unique(y):
            if target_class in self.sampling_strategy_.keys():
                n_samples = self.sampling_strategy_[target_class]
                threshold = np.percentile(
                    probabilities[y == target_class],
                    (1.0 - (n_samples / target_stats[target_class])) * 100.0,
                )
                index_target_class = np.flatnonzero(
                    probabilities[y == target_class] >= threshold
                )
            else:
                index_target_class = slice(None)

            idx_under = np.concatenate(
                (
                    idx_under,
                    np.flatnonzero(y == target_class)[index_target_class],
                ),
                axis=0,
            )

        self.sample_indices_ = idx_under

        return _safe_indexing(X, idx_under), _safe_indexing(y, idx_under)

    def _more_tags(self):
        return {"sample_indices": True}
