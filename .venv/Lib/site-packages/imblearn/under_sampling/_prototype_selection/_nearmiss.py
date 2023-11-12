"""Class to perform under-sampling based on nearmiss methods."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

import numbers
import warnings
from collections import Counter

import numpy as np
from sklearn.utils import _safe_indexing

from ...utils import Substitution, check_neighbors_object
from ...utils._docstring import _n_jobs_docstring
from ...utils._param_validation import HasMethods, Interval
from ..base import BaseUnderSampler


@Substitution(
    sampling_strategy=BaseUnderSampler._sampling_strategy_docstring,
    n_jobs=_n_jobs_docstring,
)
class NearMiss(BaseUnderSampler):
    """Class to perform under-sampling based on NearMiss methods.

    Read more in the :ref:`User Guide <controlled_under_sampling>`.

    Parameters
    ----------
    {sampling_strategy}

    version : int, default=1
        Version of the NearMiss to use. Possible values are 1, 2 or 3.

    n_neighbors : int or estimator object, default=3
        If ``int``, size of the neighbourhood to consider to compute the
        average distance to the minority point samples.  If object, an
        estimator that inherits from
        :class:`~sklearn.neighbors.base.KNeighborsMixin` that will be used to
        find the k_neighbors.
        By default, it will be a 3-NN.

    n_neighbors_ver3 : int or estimator object, default=3
        If ``int``, NearMiss-3 algorithm start by a phase of re-sampling. This
        parameter correspond to the number of neighbours selected create the
        subset in which the selection will be performed.  If object, an
        estimator that inherits from
        :class:`~sklearn.neighbors.base.KNeighborsMixin` that will be used to
        find the k_neighbors.
        By default, it will be a 3-NN.

    {n_jobs}

    Attributes
    ----------
    sampling_strategy_ : dict
        Dictionary containing the information to sample the dataset. The keys
        corresponds to the class labels from which to sample and the values
        are the number of samples to sample.

    nn_ : estimator object
        Validated K-nearest Neighbours object created from `n_neighbors` parameter.

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
    RandomUnderSampler : Random undersample the dataset.

    InstanceHardnessThreshold : Use of classifier to undersample a dataset.

    Notes
    -----
    The methods are based on [1]_.

    Supports multi-class resampling.

    References
    ----------
    .. [1] I. Mani, I. Zhang. "kNN approach to unbalanced data distributions:
       a case study involving information extraction," In Proceedings of
       workshop on learning from imbalanced datasets, 2003.

    Examples
    --------
    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.under_sampling import NearMiss
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> nm = NearMiss()
    >>> X_res, y_res = nm.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{0: 100, 1: 100}})
    """

    _parameter_constraints: dict = {
        **BaseUnderSampler._parameter_constraints,
        "version": [Interval(numbers.Integral, 1, 3, closed="both")],
        "n_neighbors": [
            Interval(numbers.Integral, 1, None, closed="left"),
            HasMethods(["kneighbors", "kneighbors_graph"]),
        ],
        "n_neighbors_ver3": [
            Interval(numbers.Integral, 1, None, closed="left"),
            HasMethods(["kneighbors", "kneighbors_graph"]),
        ],
        "n_jobs": [numbers.Integral, None],
    }

    def __init__(
        self,
        *,
        sampling_strategy="auto",
        version=1,
        n_neighbors=3,
        n_neighbors_ver3=3,
        n_jobs=None,
    ):
        super().__init__(sampling_strategy=sampling_strategy)
        self.version = version
        self.n_neighbors = n_neighbors
        self.n_neighbors_ver3 = n_neighbors_ver3
        self.n_jobs = n_jobs

    def _selection_dist_based(
        self, X, y, dist_vec, num_samples, key, sel_strategy="nearest"
    ):
        """Select the appropriate samples depending of the strategy selected.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Original samples.

        y : array-like, shape (n_samples,)
            Associated label to X.

        dist_vec : ndarray, shape (n_samples, )
            The distance matrix to the nearest neigbour.

        num_samples: int
            The desired number of samples to select.

        key : str or int,
            The target class.

        sel_strategy : str, optional (default='nearest')
            Strategy to select the samples. Either 'nearest' or 'farthest'

        Returns
        -------
        idx_sel : ndarray, shape (num_samples,)
            The list of the indices of the selected samples.

        """

        # Compute the distance considering the farthest neighbour
        dist_avg_vec = np.sum(dist_vec[:, -self.nn_.n_neighbors :], axis=1)

        target_class_indices = np.flatnonzero(y == key)
        if dist_vec.shape[0] != _safe_indexing(X, target_class_indices).shape[0]:
            raise RuntimeError(
                "The samples to be selected do not correspond"
                " to the distance matrix given. Ensure that"
                " both `X[y == key]` and `dist_vec` are"
                " related."
            )

        # Sort the list of distance and get the index
        if sel_strategy == "nearest":
            sort_way = False
        else:  # sel_strategy == "farthest":
            sort_way = True

        sorted_idx = sorted(
            range(len(dist_avg_vec)),
            key=dist_avg_vec.__getitem__,
            reverse=sort_way,
        )

        # Throw a warning to tell the user that we did not have enough samples
        # to select and that we just select everything
        if len(sorted_idx) < num_samples:
            warnings.warn(
                "The number of the samples to be selected is larger"
                " than the number of samples available. The"
                " balancing ratio cannot be ensure and all samples"
                " will be returned."
            )

        # Select the desired number of samples
        return sorted_idx[:num_samples]

    def _validate_estimator(self):
        """Private function to create the NN estimator"""

        self.nn_ = check_neighbors_object("n_neighbors", self.n_neighbors)
        self.nn_.set_params(**{"n_jobs": self.n_jobs})

        if self.version == 3:
            self.nn_ver3_ = check_neighbors_object(
                "n_neighbors_ver3", self.n_neighbors_ver3
            )
            self.nn_ver3_.set_params(**{"n_jobs": self.n_jobs})

    def _fit_resample(self, X, y):
        self._validate_estimator()

        idx_under = np.empty((0,), dtype=int)

        target_stats = Counter(y)
        class_minority = min(target_stats, key=target_stats.get)
        minority_class_indices = np.flatnonzero(y == class_minority)

        self.nn_.fit(_safe_indexing(X, minority_class_indices))

        for target_class in np.unique(y):
            if target_class in self.sampling_strategy_.keys():
                n_samples = self.sampling_strategy_[target_class]
                target_class_indices = np.flatnonzero(y == target_class)
                X_class = _safe_indexing(X, target_class_indices)
                y_class = _safe_indexing(y, target_class_indices)

                if self.version == 1:
                    dist_vec, idx_vec = self.nn_.kneighbors(
                        X_class, n_neighbors=self.nn_.n_neighbors
                    )
                    index_target_class = self._selection_dist_based(
                        X,
                        y,
                        dist_vec,
                        n_samples,
                        target_class,
                        sel_strategy="nearest",
                    )
                elif self.version == 2:
                    dist_vec, idx_vec = self.nn_.kneighbors(
                        X_class, n_neighbors=target_stats[class_minority]
                    )
                    index_target_class = self._selection_dist_based(
                        X,
                        y,
                        dist_vec,
                        n_samples,
                        target_class,
                        sel_strategy="nearest",
                    )
                elif self.version == 3:
                    self.nn_ver3_.fit(X_class)
                    dist_vec, idx_vec = self.nn_ver3_.kneighbors(
                        _safe_indexing(X, minority_class_indices)
                    )
                    idx_vec_farthest = np.unique(idx_vec.reshape(-1))
                    X_class_selected = _safe_indexing(X_class, idx_vec_farthest)
                    y_class_selected = _safe_indexing(y_class, idx_vec_farthest)

                    dist_vec, idx_vec = self.nn_.kneighbors(
                        X_class_selected, n_neighbors=self.nn_.n_neighbors
                    )
                    index_target_class = self._selection_dist_based(
                        X_class_selected,
                        y_class_selected,
                        dist_vec,
                        n_samples,
                        target_class,
                        sel_strategy="farthest",
                    )
                    # idx_tmp is relative to the feature selected in the
                    # previous step and we need to find the indirection
                    index_target_class = idx_vec_farthest[index_target_class]
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

    # fmt: off
    def _more_tags(self):
        return {
            "sample_indices": True,
            "_xfail_checks": {
                "check_samplers_fit_resample":
                "Fails for NearMiss-3 with less samples than expected"
            }
        }
    # fmt: on
