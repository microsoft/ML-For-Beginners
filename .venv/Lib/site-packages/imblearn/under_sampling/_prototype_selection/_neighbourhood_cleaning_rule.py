"""Class performing under-sampling based on the neighbourhood cleaning rule."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

import numbers
import warnings
from collections import Counter

import numpy as np
from sklearn.base import clone
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.utils import _safe_indexing

from ...utils import Substitution
from ...utils._docstring import _n_jobs_docstring
from ...utils._param_validation import HasMethods, Hidden, Interval, StrOptions
from ..base import BaseCleaningSampler
from ._edited_nearest_neighbours import EditedNearestNeighbours

SEL_KIND = ("all", "mode")


@Substitution(
    sampling_strategy=BaseCleaningSampler._sampling_strategy_docstring,
    n_jobs=_n_jobs_docstring,
)
class NeighbourhoodCleaningRule(BaseCleaningSampler):
    """Undersample based on the neighbourhood cleaning rule.

    This class uses ENN and a k-NN to remove noisy samples from the datasets.

    Read more in the :ref:`User Guide <condensed_nearest_neighbors>`.

    Parameters
    ----------
    {sampling_strategy}

    edited_nearest_neighbours : estimator object, default=None
        The :class:`~imblearn.under_sampling.EditedNearestNeighbours` (ENN)
        object to clean the dataset. If `None`, a default ENN is created with
        `kind_sel="mode"` and `n_neighbors=n_neighbors`.

    n_neighbors : int or estimator object, default=3
        If ``int``, size of the neighbourhood to consider to compute the
        K-nearest neighbors. If object, an estimator that inherits from
        :class:`~sklearn.neighbors.base.KNeighborsMixin` that will be used to
        find the nearest-neighbors. By default, it will be a 3-NN.

    kind_sel : {{"all", "mode"}}, default='all'
        Strategy to use in order to exclude samples in the ENN sampling.

        - If ``'all'``, all neighbours will have to agree with the samples of
          interest to not be excluded.
        - If ``'mode'``, the majority vote of the neighbours will be used in
          order to exclude a sample.

        The strategy `"all"` will be less conservative than `'mode'`. Thus,
        more samples will be removed when `kind_sel="all"` generally.

        .. deprecated:: 0.12
           `kind_sel` is deprecated in 0.12 and will be removed in 0.14.
           Currently the parameter has no effect and corresponds always to the
           `"all"` strategy.

    threshold_cleaning : float, default=0.5
        Threshold used to whether consider a class or not during the cleaning
        after applying ENN. A class will be considered during cleaning when:

        Ci > C x T ,

        where Ci and C is the number of samples in the class and the data set,
        respectively and theta is the threshold.

    {n_jobs}

    Attributes
    ----------
    sampling_strategy_ : dict
        Dictionary containing the information to sample the dataset. The keys
        corresponds to the class labels from which to sample and the values
        are the number of samples to sample.

    edited_nearest_neighbours_ : estimator object
        The edited nearest neighbour object used to make the first resampling.

    nn_ : estimator object
        Validated K-nearest Neighbours object created from `n_neighbors` parameter.

    classes_to_clean_ : list
        The classes considered with under-sampling by `nn_` in the second cleaning
        phase.

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
    EditedNearestNeighbours : Undersample by editing noisy samples.

    Notes
    -----
    See the original paper: [1]_.

    Supports multi-class resampling. A one-vs.-rest scheme is used when
    sampling a class as proposed in [1]_.

    References
    ----------
    .. [1] J. Laurikkala, "Improving identification of difficult small classes
       by balancing class distribution," Springer Berlin Heidelberg, 2001.

    Examples
    --------
    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.under_sampling import NeighbourhoodCleaningRule
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> ncr = NeighbourhoodCleaningRule()
    >>> X_res, y_res = ncr.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{1: 888, 0: 100}})
    """

    _parameter_constraints: dict = {
        **BaseCleaningSampler._parameter_constraints,
        "edited_nearest_neighbours": [
            HasMethods(["fit_resample"]),
            None,
        ],
        "n_neighbors": [
            Interval(numbers.Integral, 1, None, closed="left"),
            HasMethods(["kneighbors", "kneighbors_graph"]),
        ],
        "kind_sel": [StrOptions({"all", "mode"}), Hidden(StrOptions({"deprecated"}))],
        "threshold_cleaning": [Interval(numbers.Real, 0, None, closed="neither")],
        "n_jobs": [numbers.Integral, None],
    }

    def __init__(
        self,
        *,
        sampling_strategy="auto",
        edited_nearest_neighbours=None,
        n_neighbors=3,
        kind_sel="deprecated",
        threshold_cleaning=0.5,
        n_jobs=None,
    ):
        super().__init__(sampling_strategy=sampling_strategy)
        self.edited_nearest_neighbours = edited_nearest_neighbours
        self.n_neighbors = n_neighbors
        self.kind_sel = kind_sel
        self.threshold_cleaning = threshold_cleaning
        self.n_jobs = n_jobs

    def _validate_estimator(self):
        """Create the objects required by NCR."""
        if isinstance(self.n_neighbors, numbers.Integral):
            self.nn_ = KNeighborsClassifier(
                n_neighbors=self.n_neighbors, n_jobs=self.n_jobs
            )
        elif isinstance(self.n_neighbors, NearestNeighbors):
            # backward compatibility when passing a NearestNeighbors object
            self.nn_ = KNeighborsClassifier(
                n_neighbors=self.n_neighbors.n_neighbors - 1, n_jobs=self.n_jobs
            )
        else:
            self.nn_ = clone(self.n_neighbors)

        if self.edited_nearest_neighbours is None:
            self.edited_nearest_neighbours_ = EditedNearestNeighbours(
                sampling_strategy=self.sampling_strategy,
                n_neighbors=self.n_neighbors,
                kind_sel="mode",
                n_jobs=self.n_jobs,
            )
        else:
            self.edited_nearest_neighbours_ = clone(self.edited_nearest_neighbours)

    def _fit_resample(self, X, y):
        if self.kind_sel != "deprecated":
            warnings.warn(
                "`kind_sel` is deprecated in 0.12 and will be removed in 0.14. "
                "It already has not effect and corresponds to the `'all'` option.",
                FutureWarning,
            )
        self._validate_estimator()
        self.edited_nearest_neighbours_.fit_resample(X, y)
        index_not_a1 = self.edited_nearest_neighbours_.sample_indices_
        index_a1 = np.ones(y.shape, dtype=bool)
        index_a1[index_not_a1] = False
        index_a1 = np.flatnonzero(index_a1)

        # clean the neighborhood
        target_stats = Counter(y)
        class_minority = min(target_stats, key=target_stats.get)
        # compute which classes to consider for cleaning for the A2 group
        self.classes_to_clean_ = [
            c
            for c, n_samples in target_stats.items()
            if (
                c in self.sampling_strategy_.keys()
                and (n_samples > target_stats[class_minority] * self.threshold_cleaning)
            )
        ]
        self.nn_.fit(X, y)

        class_minority_indices = np.flatnonzero(y == class_minority)
        X_minority = _safe_indexing(X, class_minority_indices)
        y_minority = _safe_indexing(y, class_minority_indices)

        y_pred_minority = self.nn_.predict(X_minority)
        # add an additional sample since the query points contains the original dataset
        neighbors_to_minority_indices = self.nn_.kneighbors(
            X_minority, n_neighbors=self.nn_.n_neighbors + 1, return_distance=False
        )[:, 1:]

        mask_misclassified_minority = y_pred_minority != y_minority
        index_a2 = np.ravel(neighbors_to_minority_indices[mask_misclassified_minority])
        index_a2 = np.array(
            [
                index
                for index in np.unique(index_a2)
                if y[index] in self.classes_to_clean_
            ]
        )

        union_a1_a2 = np.union1d(index_a1, index_a2).astype(int)
        selected_samples = np.ones(y.shape, dtype=bool)
        selected_samples[union_a1_a2] = False
        self.sample_indices_ = np.flatnonzero(selected_samples)

        return (
            _safe_indexing(X, self.sample_indices_),
            _safe_indexing(y, self.sample_indices_),
        )

    def _more_tags(self):
        return {"sample_indices": True}
