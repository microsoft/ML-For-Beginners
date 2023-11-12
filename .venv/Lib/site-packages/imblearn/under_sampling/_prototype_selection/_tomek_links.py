"""Class to perform under-sampling by removing Tomek's links."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Fernando Nogueira
#          Christos Aridas
# License: MIT

import numbers

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import _safe_indexing

from ...utils import Substitution
from ...utils._docstring import _n_jobs_docstring
from ..base import BaseCleaningSampler


@Substitution(
    sampling_strategy=BaseCleaningSampler._sampling_strategy_docstring,
    n_jobs=_n_jobs_docstring,
)
class TomekLinks(BaseCleaningSampler):
    """Under-sampling by removing Tomek's links.

    Read more in the :ref:`User Guide <tomek_links>`.

    Parameters
    ----------
    {sampling_strategy}

    {n_jobs}

    Attributes
    ----------
    sampling_strategy_ : dict
        Dictionary containing the information to sample the dataset. The keys
        corresponds to the class labels from which to sample and the values
        are the number of samples to sample.

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
    EditedNearestNeighbours : Undersample by samples edition.

    CondensedNearestNeighbour : Undersample by samples condensation.

    RandomUnderSampler : Randomly under-sample the dataset.

    Notes
    -----
    This method is based on [1]_.

    Supports multi-class resampling. A one-vs.-rest scheme is used as
    originally proposed in [1]_.

    References
    ----------
    .. [1] I. Tomek, "Two modifications of CNN," In Systems, Man, and
       Cybernetics, IEEE Transactions on, vol. 6, pp 769-772, 1976.

    Examples
    --------
    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.under_sampling import TomekLinks
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> tl = TomekLinks()
    >>> X_res, y_res = tl.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{1: 897, 0: 100}})
    """

    _parameter_constraints: dict = {
        **BaseCleaningSampler._parameter_constraints,
        "n_jobs": [numbers.Integral, None],
    }

    def __init__(self, *, sampling_strategy="auto", n_jobs=None):
        super().__init__(sampling_strategy=sampling_strategy)
        self.n_jobs = n_jobs

    @staticmethod
    def is_tomek(y, nn_index, class_type):
        """Detect if samples are Tomek's link.

        More precisely, it uses the target vector and the first neighbour of
        every sample point and looks for Tomek pairs. Returning a boolean
        vector with True for majority Tomek links.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            Target vector of the data set, necessary to keep track of whether a
            sample belongs to minority or not.

        nn_index : ndarray of shape (len(y),)
            The index of the closes nearest neighbour to a sample point.

        class_type : int or str
            The label of the minority class.

        Returns
        -------
        is_tomek : ndarray of shape (len(y), )
            Boolean vector on len( # samples ), with True for majority samples
            that are Tomek links.
        """
        links = np.zeros(len(y), dtype=bool)

        # find which class to not consider
        class_excluded = [c for c in np.unique(y) if c not in class_type]

        # there is a Tomek link between two samples if they are both nearest
        # neighbors of each others.
        for index_sample, target_sample in enumerate(y):
            if target_sample in class_excluded:
                continue

            if y[nn_index[index_sample]] != target_sample:
                if nn_index[nn_index[index_sample]] == index_sample:
                    links[index_sample] = True

        return links

    def _fit_resample(self, X, y):
        # Find the nearest neighbour of every point
        nn = NearestNeighbors(n_neighbors=2, n_jobs=self.n_jobs)
        nn.fit(X)
        nns = nn.kneighbors(X, return_distance=False)[:, 1]

        links = self.is_tomek(y, nns, self.sampling_strategy_)
        self.sample_indices_ = np.flatnonzero(np.logical_not(links))

        return (
            _safe_indexing(X, self.sample_indices_),
            _safe_indexing(y, self.sample_indices_),
        )

    def _more_tags(self):
        return {"sample_indices": True}
