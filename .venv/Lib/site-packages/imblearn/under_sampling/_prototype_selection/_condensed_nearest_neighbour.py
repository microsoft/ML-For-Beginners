"""Class to perform under-sampling based on the condensed nearest neighbour
method."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

import numbers
from collections import Counter

import numpy as np
from scipy.sparse import issparse
from sklearn.base import clone
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import _safe_indexing, check_random_state

from ...utils import Substitution
from ...utils._docstring import _n_jobs_docstring, _random_state_docstring
from ...utils._param_validation import HasMethods, Interval
from ..base import BaseCleaningSampler


@Substitution(
    sampling_strategy=BaseCleaningSampler._sampling_strategy_docstring,
    n_jobs=_n_jobs_docstring,
    random_state=_random_state_docstring,
)
class CondensedNearestNeighbour(BaseCleaningSampler):
    """Undersample based on the condensed nearest neighbour method.

    Read more in the :ref:`User Guide <condensed_nearest_neighbors>`.

    Parameters
    ----------
    {sampling_strategy}

    {random_state}

    n_neighbors : int or estimator object, default=None
        If ``int``, size of the neighbourhood to consider to compute the
        nearest neighbors. If object, an estimator that inherits from
        :class:`~sklearn.neighbors.base.KNeighborsMixin` that will be used to
        find the nearest-neighbors.  If `None`, a
        :class:`~sklearn.neighbors.KNeighborsClassifier` with a 1-NN rules will
        be used.

    n_seeds_S : int, default=1
        Number of samples to extract in order to build the set S.

    {n_jobs}

    Attributes
    ----------
    sampling_strategy_ : dict
        Dictionary containing the information to sample the dataset. The keys
        corresponds to the class labels from which to sample and the values
        are the number of samples to sample.

    estimator_ : estimator object
        The validated K-nearest neighbor estimator created from `n_neighbors` parameter.

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
    EditedNearestNeighbours : Undersample by editing samples.

    RepeatedEditedNearestNeighbours : Undersample by repeating ENN algorithm.

    AllKNN : Undersample using ENN and various number of neighbours.

    Notes
    -----
    The method is based on [1]_.

    Supports multi-class resampling. A one-vs.-rest scheme is used when
    sampling a class as proposed in [1]_.

    References
    ----------
    .. [1] P. Hart, "The condensed nearest neighbor rule,"
       In Information Theory, IEEE Transactions on, vol. 14(3),
       pp. 515-516, 1968.

    Examples
    --------
    >>> from collections import Counter  # doctest: +SKIP
    >>> from sklearn.datasets import fetch_mldata  # doctest: +SKIP
    >>> from imblearn.under_sampling import \
CondensedNearestNeighbour  # doctest: +SKIP
    >>> pima = fetch_mldata('diabetes_scale')  # doctest: +SKIP
    >>> X, y = pima['data'], pima['target']  # doctest: +SKIP
    >>> print('Original dataset shape %s' % Counter(y))  # doctest: +SKIP
    Original dataset shape Counter({{1: 500, -1: 268}})  # doctest: +SKIP
    >>> cnn = CondensedNearestNeighbour(random_state=42)  # doctest: +SKIP
    >>> X_res, y_res = cnn.fit_resample(X, y)  #doctest: +SKIP
    >>> print('Resampled dataset shape %s' % Counter(y_res))  # doctest: +SKIP
    Resampled dataset shape Counter({{-1: 268, 1: 227}})  # doctest: +SKIP
    """

    _parameter_constraints: dict = {
        **BaseCleaningSampler._parameter_constraints,
        "n_neighbors": [
            Interval(numbers.Integral, 1, None, closed="left"),
            HasMethods(["kneighbors", "kneighbors_graph"]),
            None,
        ],
        "n_seeds_S": [Interval(numbers.Integral, 1, None, closed="left")],
        "n_jobs": [numbers.Integral, None],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        *,
        sampling_strategy="auto",
        random_state=None,
        n_neighbors=None,
        n_seeds_S=1,
        n_jobs=None,
    ):
        super().__init__(sampling_strategy=sampling_strategy)
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.n_seeds_S = n_seeds_S
        self.n_jobs = n_jobs

    def _validate_estimator(self):
        """Private function to create the NN estimator"""
        if self.n_neighbors is None:
            self.estimator_ = KNeighborsClassifier(n_neighbors=1, n_jobs=self.n_jobs)
        elif isinstance(self.n_neighbors, numbers.Integral):
            self.estimator_ = KNeighborsClassifier(
                n_neighbors=self.n_neighbors, n_jobs=self.n_jobs
            )
        elif isinstance(self.n_neighbors, KNeighborsClassifier):
            self.estimator_ = clone(self.n_neighbors)

    def _fit_resample(self, X, y):
        self._validate_estimator()

        random_state = check_random_state(self.random_state)
        target_stats = Counter(y)
        class_minority = min(target_stats, key=target_stats.get)
        idx_under = np.empty((0,), dtype=int)

        for target_class in np.unique(y):
            if target_class in self.sampling_strategy_.keys():
                # Randomly get one sample from the majority class
                # Generate the index to select
                idx_maj = np.flatnonzero(y == target_class)
                idx_maj_sample = idx_maj[
                    random_state.randint(
                        low=0,
                        high=target_stats[target_class],
                        size=self.n_seeds_S,
                    )
                ]

                # Create the set C - One majority samples and all minority
                C_indices = np.append(
                    np.flatnonzero(y == class_minority), idx_maj_sample
                )
                C_x = _safe_indexing(X, C_indices)
                C_y = _safe_indexing(y, C_indices)

                # Create the set S - all majority samples
                S_indices = np.flatnonzero(y == target_class)
                S_x = _safe_indexing(X, S_indices)
                S_y = _safe_indexing(y, S_indices)

                # fit knn on C
                self.estimator_.fit(C_x, C_y)

                good_classif_label = idx_maj_sample.copy()
                # Check each sample in S if we keep it or drop it
                for idx_sam, (x_sam, y_sam) in enumerate(zip(S_x, S_y)):
                    # Do not select sample which are already well classified
                    if idx_sam in good_classif_label:
                        continue

                    # Classify on S
                    if not issparse(x_sam):
                        x_sam = x_sam.reshape(1, -1)
                    pred_y = self.estimator_.predict(x_sam)

                    # If the prediction do not agree with the true label
                    # append it in C_x
                    if y_sam != pred_y:
                        # Keep the index for later
                        idx_maj_sample = np.append(idx_maj_sample, idx_maj[idx_sam])

                        # Update C
                        C_indices = np.append(C_indices, idx_maj[idx_sam])
                        C_x = _safe_indexing(X, C_indices)
                        C_y = _safe_indexing(y, C_indices)

                        # fit a knn on C
                        self.estimator_.fit(C_x, C_y)

                        # This experimental to speed up the search
                        # Classify all the element in S and avoid to test the
                        # well classified elements
                        pred_S_y = self.estimator_.predict(S_x)
                        good_classif_label = np.unique(
                            np.append(idx_maj_sample, np.flatnonzero(pred_S_y == S_y))
                        )

                idx_under = np.concatenate((idx_under, idx_maj_sample), axis=0)
            else:
                idx_under = np.concatenate(
                    (idx_under, np.flatnonzero(y == target_class)), axis=0
                )

        self.sample_indices_ = idx_under

        return _safe_indexing(X, idx_under), _safe_indexing(y, idx_under)

    def _more_tags(self):
        return {"sample_indices": True}
