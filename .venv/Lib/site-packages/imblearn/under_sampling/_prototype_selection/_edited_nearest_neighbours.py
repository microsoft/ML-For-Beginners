"""Class to perform under-sampling based on the edited nearest neighbour
method."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Dayvid Oliveira
#          Christos Aridas
# License: MIT

import numbers
from collections import Counter

import numpy as np
from sklearn.utils import _safe_indexing

from ...utils import Substitution, check_neighbors_object
from ...utils._docstring import _n_jobs_docstring
from ...utils._param_validation import HasMethods, Interval, StrOptions
from ...utils.fixes import _mode
from ..base import BaseCleaningSampler

SEL_KIND = ("all", "mode")


@Substitution(
    sampling_strategy=BaseCleaningSampler._sampling_strategy_docstring,
    n_jobs=_n_jobs_docstring,
)
class EditedNearestNeighbours(BaseCleaningSampler):
    """Undersample based on the edited nearest neighbour method.

    This method will clean the database by removing samples close to the
    decision boundary.

    Read more in the :ref:`User Guide <edited_nearest_neighbors>`.

    Parameters
    ----------
    {sampling_strategy}

    n_neighbors : int or object, default=3
        If ``int``, size of the neighbourhood to consider to compute the
        nearest neighbors. If object, an estimator that inherits from
        :class:`~sklearn.neighbors.base.KNeighborsMixin` that will be used to
        find the nearest-neighbors.

    kind_sel : {{'all', 'mode'}}, default='all'
        Strategy to use in order to exclude samples.

        - If ``'all'``, all neighbours will have to agree with the samples of
          interest to not be excluded.
        - If ``'mode'``, the majority vote of the neighbours will be used in
          order to exclude a sample.

        The strategy `"all"` will be less conservative than `'mode'`. Thus,
        more samples will be removed when `kind_sel="all"` generally.

    {n_jobs}

    Attributes
    ----------
    sampling_strategy_ : dict
        Dictionary containing the information to sample the dataset. The keys
        corresponds to the class labels from which to sample and the values
        are the number of samples to sample.

    nn_ : estimator object
        Validated K-nearest Neighbours instance created from `n_neighbors` parameter.

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
    CondensedNearestNeighbour : Undersample by condensing samples.

    RepeatedEditedNearestNeighbours : Undersample by repeating ENN algorithm.

    AllKNN : Undersample using ENN and various number of neighbours.

    Notes
    -----
    The method is based on [1]_.

    Supports multi-class resampling. A one-vs.-rest scheme is used when
    sampling a class as proposed in [1]_.

    References
    ----------
    .. [1] D. Wilson, Asymptotic" Properties of Nearest Neighbor Rules Using
       Edited Data," In IEEE Transactions on Systems, Man, and Cybernetrics,
       vol. 2 (3), pp. 408-421, 1972.

    Examples
    --------

    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.under_sampling import EditedNearestNeighbours
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> enn = EditedNearestNeighbours()
    >>> X_res, y_res = enn.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{1: 887, 0: 100}})
    """

    _parameter_constraints: dict = {
        **BaseCleaningSampler._parameter_constraints,
        "n_neighbors": [
            Interval(numbers.Integral, 1, None, closed="left"),
            HasMethods(["kneighbors", "kneighbors_graph"]),
        ],
        "kind_sel": [StrOptions({"all", "mode"})],
        "n_jobs": [numbers.Integral, None],
    }

    def __init__(
        self,
        *,
        sampling_strategy="auto",
        n_neighbors=3,
        kind_sel="all",
        n_jobs=None,
    ):
        super().__init__(sampling_strategy=sampling_strategy)
        self.n_neighbors = n_neighbors
        self.kind_sel = kind_sel
        self.n_jobs = n_jobs

    def _validate_estimator(self):
        """Validate the estimator created in the ENN."""
        self.nn_ = check_neighbors_object(
            "n_neighbors", self.n_neighbors, additional_neighbor=1
        )
        self.nn_.set_params(**{"n_jobs": self.n_jobs})

    def _fit_resample(self, X, y):
        self._validate_estimator()

        idx_under = np.empty((0,), dtype=int)

        self.nn_.fit(X)

        for target_class in np.unique(y):
            if target_class in self.sampling_strategy_.keys():
                target_class_indices = np.flatnonzero(y == target_class)
                X_class = _safe_indexing(X, target_class_indices)
                y_class = _safe_indexing(y, target_class_indices)
                nnhood_idx = self.nn_.kneighbors(X_class, return_distance=False)[:, 1:]
                nnhood_label = y[nnhood_idx]
                if self.kind_sel == "mode":
                    nnhood_label, _ = _mode(nnhood_label, axis=1)
                    nnhood_bool = np.ravel(nnhood_label) == y_class
                elif self.kind_sel == "all":
                    nnhood_label = nnhood_label == target_class
                    nnhood_bool = np.all(nnhood_label, axis=1)
                index_target_class = np.flatnonzero(nnhood_bool)
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


@Substitution(
    sampling_strategy=BaseCleaningSampler._sampling_strategy_docstring,
    n_jobs=_n_jobs_docstring,
)
class RepeatedEditedNearestNeighbours(BaseCleaningSampler):
    """Undersample based on the repeated edited nearest neighbour method.

    This method will repeat several time the ENN algorithm.

    Read more in the :ref:`User Guide <edited_nearest_neighbors>`.

    Parameters
    ----------
    {sampling_strategy}

    n_neighbors : int or object, default=3
        If ``int``, size of the neighbourhood to consider to compute the
        nearest neighbors. If object, an estimator that inherits from
        :class:`~sklearn.neighbors.base.KNeighborsMixin` that will be used to
        find the nearest-neighbors.

    max_iter : int, default=100
        Maximum number of iterations of the edited nearest neighbours
        algorithm for a single run.

    kind_sel : {{'all', 'mode'}}, default='all'
        Strategy to use in order to exclude samples.

        - If ``'all'``, all neighbours will have to agree with the samples of
          interest to not be excluded.
        - If ``'mode'``, the majority vote of the neighbours will be used in
          order to exclude a sample.

        The strategy `"all"` will be less conservative than `'mode'`. Thus,
        more samples will be removed when `kind_sel="all"` generally.

    {n_jobs}

    Attributes
    ----------
    sampling_strategy_ : dict
        Dictionary containing the information to sample the dataset. The keys
        corresponds to the class labels from which to sample and the values
        are the number of samples to sample.

    nn_ : estimator object
        Validated K-nearest Neighbours estimator linked to the parameter `n_neighbors`.

    enn_ : sampler object
        The validated :class:`~imblearn.under_sampling.EditedNearestNeighbours`
        instance.

    sample_indices_ : ndarray of shape (n_new_samples,)
        Indices of the samples selected.

        .. versionadded:: 0.4

    n_iter_ : int
        Number of iterations run.

        .. versionadded:: 0.6

    n_features_in_ : int
        Number of features in the input dataset.

        .. versionadded:: 0.9

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during `fit`. Defined only when `X` has feature
        names that are all strings.

        .. versionadded:: 0.10

    See Also
    --------
    CondensedNearestNeighbour : Undersample by condensing samples.

    EditedNearestNeighbours : Undersample by editing samples.

    AllKNN : Undersample using ENN and various number of neighbours.

    Notes
    -----
    The method is based on [1]_. A one-vs.-rest scheme is used when
    sampling a class as proposed in [1]_.

    Supports multi-class resampling.

    References
    ----------
    .. [1] I. Tomek, "An Experiment with the Edited Nearest-Neighbor
       Rule," IEEE Transactions on Systems, Man, and Cybernetics, vol. 6(6),
       pp. 448-452, June 1976.

    Examples
    --------
    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.under_sampling import RepeatedEditedNearestNeighbours
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> renn = RepeatedEditedNearestNeighbours()
    >>> X_res, y_res = renn.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{1: 887, 0: 100}})
    """

    _parameter_constraints: dict = {
        **BaseCleaningSampler._parameter_constraints,
        "n_neighbors": [
            Interval(numbers.Integral, 1, None, closed="left"),
            HasMethods(["kneighbors", "kneighbors_graph"]),
        ],
        "max_iter": [Interval(numbers.Integral, 1, None, closed="left")],
        "kind_sel": [StrOptions({"all", "mode"})],
        "n_jobs": [numbers.Integral, None],
    }

    def __init__(
        self,
        *,
        sampling_strategy="auto",
        n_neighbors=3,
        max_iter=100,
        kind_sel="all",
        n_jobs=None,
    ):
        super().__init__(sampling_strategy=sampling_strategy)
        self.n_neighbors = n_neighbors
        self.kind_sel = kind_sel
        self.n_jobs = n_jobs
        self.max_iter = max_iter

    def _validate_estimator(self):
        """Private function to create the NN estimator"""
        self.nn_ = check_neighbors_object(
            "n_neighbors", self.n_neighbors, additional_neighbor=1
        )

        self.enn_ = EditedNearestNeighbours(
            sampling_strategy=self.sampling_strategy,
            n_neighbors=self.nn_,
            kind_sel=self.kind_sel,
            n_jobs=self.n_jobs,
        )

    def _fit_resample(self, X, y):
        self._validate_estimator()

        X_, y_ = X, y
        self.sample_indices_ = np.arange(X.shape[0], dtype=int)
        target_stats = Counter(y)
        class_minority = min(target_stats, key=target_stats.get)

        for n_iter in range(self.max_iter):
            prev_len = y_.shape[0]
            X_enn, y_enn = self.enn_.fit_resample(X_, y_)

            # Check the stopping criterion
            # 1. If there is no changes for the vector y
            # 2. If the number of samples in the other class become inferior to
            # the number of samples in the majority class
            # 3. If one of the class is disappearing

            # Case 1
            b_conv = prev_len == y_enn.shape[0]

            # Case 2
            stats_enn = Counter(y_enn)
            count_non_min = np.array(
                [
                    val
                    for val, key in zip(stats_enn.values(), stats_enn.keys())
                    if key != class_minority
                ]
            )
            b_min_bec_maj = np.any(count_non_min < target_stats[class_minority])

            # Case 3
            b_remove_maj_class = len(stats_enn) < len(target_stats)

            (
                X_,
                y_,
            ) = (
                X_enn,
                y_enn,
            )
            self.sample_indices_ = self.sample_indices_[self.enn_.sample_indices_]

            if b_conv or b_min_bec_maj or b_remove_maj_class:
                if b_conv:
                    (
                        X_,
                        y_,
                    ) = (
                        X_enn,
                        y_enn,
                    )
                    self.sample_indices_ = self.sample_indices_[
                        self.enn_.sample_indices_
                    ]
                break

        self.n_iter_ = n_iter + 1
        X_resampled, y_resampled = X_, y_

        return X_resampled, y_resampled

    def _more_tags(self):
        return {"sample_indices": True}


@Substitution(
    sampling_strategy=BaseCleaningSampler._sampling_strategy_docstring,
    n_jobs=_n_jobs_docstring,
)
class AllKNN(BaseCleaningSampler):
    """Undersample based on the AllKNN method.

    This method will apply ENN several time and will vary the number of nearest
    neighbours.

    Read more in the :ref:`User Guide <edited_nearest_neighbors>`.

    Parameters
    ----------
    {sampling_strategy}

    n_neighbors : int or estimator object, default=3
        If ``int``, size of the neighbourhood to consider to compute the
        nearest neighbors. If object, an estimator that inherits from
        :class:`~sklearn.neighbors.base.KNeighborsMixin` that will be used to
        find the nearest-neighbors. By default, it will be a 3-NN.

    kind_sel : {{'all', 'mode'}}, default='all'
        Strategy to use in order to exclude samples.

        - If ``'all'``, all neighbours will have to agree with the samples of
          interest to not be excluded.
        - If ``'mode'``, the majority vote of the neighbours will be used in
          order to exclude a sample.

        The strategy `"all"` will be less conservative than `'mode'`. Thus,
        more samples will be removed when `kind_sel="all"` generally.

    allow_minority : bool, default=False
        If ``True``, it allows the majority classes to become the minority
        class without early stopping.

        .. versionadded:: 0.3

    {n_jobs}

    Attributes
    ----------
    sampling_strategy_ : dict
        Dictionary containing the information to sample the dataset. The keys
        corresponds to the class labels from which to sample and the values
        are the number of samples to sample.

    nn_ : estimator object
        Validated K-nearest Neighbours estimator linked to the parameter `n_neighbors`.

    enn_ : sampler object
        The validated :class:`~imblearn.under_sampling.EditedNearestNeighbours`
        instance.

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
    CondensedNearestNeighbour: Under-sampling by condensing samples.

    EditedNearestNeighbours: Under-sampling by editing samples.

    RepeatedEditedNearestNeighbours: Under-sampling by repeating ENN.

    Notes
    -----
    The method is based on [1]_.

    Supports multi-class resampling. A one-vs.-rest scheme is used when
    sampling a class as proposed in [1]_.

    References
    ----------
    .. [1] I. Tomek, "An Experiment with the Edited Nearest-Neighbor
       Rule," IEEE Transactions on Systems, Man, and Cybernetics, vol. 6(6),
       pp. 448-452, June 1976.

    Examples
    --------
    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.under_sampling import AllKNN
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> allknn = AllKNN()
    >>> X_res, y_res = allknn.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{1: 887, 0: 100}})
    """

    _parameter_constraints: dict = {
        **BaseCleaningSampler._parameter_constraints,
        "n_neighbors": [
            Interval(numbers.Integral, 1, None, closed="left"),
            HasMethods(["kneighbors", "kneighbors_graph"]),
        ],
        "kind_sel": [StrOptions({"all", "mode"})],
        "allow_minority": ["boolean"],
        "n_jobs": [numbers.Integral, None],
    }

    def __init__(
        self,
        *,
        sampling_strategy="auto",
        n_neighbors=3,
        kind_sel="all",
        allow_minority=False,
        n_jobs=None,
    ):
        super().__init__(sampling_strategy=sampling_strategy)
        self.n_neighbors = n_neighbors
        self.kind_sel = kind_sel
        self.allow_minority = allow_minority
        self.n_jobs = n_jobs

    def _validate_estimator(self):
        """Create objects required by AllKNN"""
        self.nn_ = check_neighbors_object(
            "n_neighbors", self.n_neighbors, additional_neighbor=1
        )

        self.enn_ = EditedNearestNeighbours(
            sampling_strategy=self.sampling_strategy,
            n_neighbors=self.nn_,
            kind_sel=self.kind_sel,
            n_jobs=self.n_jobs,
        )

    def _fit_resample(self, X, y):
        self._validate_estimator()

        X_, y_ = X, y
        target_stats = Counter(y)
        class_minority = min(target_stats, key=target_stats.get)

        self.sample_indices_ = np.arange(X.shape[0], dtype=int)

        for curr_size_ngh in range(1, self.nn_.n_neighbors):
            self.enn_.n_neighbors = curr_size_ngh

            X_enn, y_enn = self.enn_.fit_resample(X_, y_)

            # Check the stopping criterion
            # 1. If the number of samples in the other class become inferior to
            # the number of samples in the majority class
            # 2. If one of the class is disappearing
            # Case 1else:

            stats_enn = Counter(y_enn)
            count_non_min = np.array(
                [
                    val
                    for val, key in zip(stats_enn.values(), stats_enn.keys())
                    if key != class_minority
                ]
            )
            b_min_bec_maj = np.any(count_non_min < target_stats[class_minority])
            if self.allow_minority:
                # overwrite b_min_bec_maj
                b_min_bec_maj = False

            # Case 2
            b_remove_maj_class = len(stats_enn) < len(target_stats)

            (
                X_,
                y_,
            ) = (
                X_enn,
                y_enn,
            )
            self.sample_indices_ = self.sample_indices_[self.enn_.sample_indices_]

            if b_min_bec_maj or b_remove_maj_class:
                break

        X_resampled, y_resampled = X_, y_

        return X_resampled, y_resampled

    def _more_tags(self):
        return {"sample_indices": True}
