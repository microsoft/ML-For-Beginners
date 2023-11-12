"""SMOTE variant employing some clustering before the generation."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Fernando Nogueira
#          Christos Aridas
# License: MIT

import math
import numbers

import numpy as np
from scipy import sparse
from sklearn.base import clone
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances
from sklearn.utils import _safe_indexing

from ...utils import Substitution
from ...utils._docstring import _n_jobs_docstring, _random_state_docstring
from ...utils._param_validation import HasMethods, Interval, StrOptions
from ..base import BaseOverSampler
from .base import BaseSMOTE


@Substitution(
    sampling_strategy=BaseOverSampler._sampling_strategy_docstring,
    n_jobs=_n_jobs_docstring,
    random_state=_random_state_docstring,
)
class KMeansSMOTE(BaseSMOTE):
    """Apply a KMeans clustering before to over-sample using SMOTE.

    This is an implementation of the algorithm described in [1]_.

    Read more in the :ref:`User Guide <smote_adasyn>`.

    .. versionadded:: 0.5

    Parameters
    ----------
    {sampling_strategy}

    {random_state}

    k_neighbors : int or object, default=2
        The nearest neighbors used to define the neighborhood of samples to use
        to generate the synthetic samples. You can pass:

        - an `int` corresponding to the number of neighbors to use. A
          `~sklearn.neighbors.NearestNeighbors` instance will be fitted in this
          case.
        - an instance of a compatible nearest neighbors algorithm that should
          implement both methods `kneighbors` and `kneighbors_graph`. For
          instance, it could correspond to a
          :class:`~sklearn.neighbors.NearestNeighbors` but could be extended to
          any compatible class.

    {n_jobs}

    kmeans_estimator : int or object, default=None
        A KMeans instance or the number of clusters to be used. By default,
        we used a :class:`~sklearn.cluster.MiniBatchKMeans` which tend to be
        better with large number of samples.

    cluster_balance_threshold : "auto" or float, default="auto"
        The threshold at which a cluster is called balanced and where samples
        of the class selected for SMOTE will be oversampled. If "auto", this
        will be determined by the ratio for each class, or it can be set
        manually.

    density_exponent : "auto" or float, default="auto"
        This exponent is used to determine the density of a cluster. Leaving
        this to "auto" will use a feature-length based exponent.

    Attributes
    ----------
    sampling_strategy_ : dict
        Dictionary containing the information to sample the dataset. The keys
        corresponds to the class labels from which to sample and the values
        are the number of samples to sample.

    kmeans_estimator_ : estimator
        The fitted clustering method used before to apply SMOTE.

    nn_k_ : estimator
        The fitted k-NN estimator used in SMOTE.

    cluster_balance_threshold_ : float
        The threshold used during ``fit`` for calling a cluster balanced.

    n_features_in_ : int
        Number of features in the input dataset.

        .. versionadded:: 0.9

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during `fit`. Defined only when `X` has feature
        names that are all strings.

        .. versionadded:: 0.10

    See Also
    --------
    SMOTE : Over-sample using SMOTE.

    SMOTENC : Over-sample using SMOTE for continuous and categorical features.

    SMOTEN : Over-sample using the SMOTE variant specifically for categorical
        features only.

    SVMSMOTE : Over-sample using SVM-SMOTE variant.

    BorderlineSMOTE : Over-sample using Borderline-SMOTE variant.

    ADASYN : Over-sample using ADASYN.

    References
    ----------
    .. [1] Felix Last, Georgios Douzas, Fernando Bacao, "Oversampling for
       Imbalanced Learning Based on K-Means and SMOTE"
       https://arxiv.org/abs/1711.00837

    Examples
    --------
    >>> import numpy as np
    >>> from imblearn.over_sampling import KMeansSMOTE
    >>> from sklearn.datasets import make_blobs
    >>> blobs = [100, 800, 100]
    >>> X, y  = make_blobs(blobs, centers=[(-10, 0), (0,0), (10, 0)])
    >>> # Add a single 0 sample in the middle blob
    >>> X = np.concatenate([X, [[0, 0]]])
    >>> y = np.append(y, 0)
    >>> # Make this a binary classification problem
    >>> y = y == 1
    >>> sm = KMeansSMOTE(
    ...     kmeans_estimator=MiniBatchKMeans(n_init=1, random_state=0), random_state=42
    ... )
    >>> X_res, y_res = sm.fit_resample(X, y)
    >>> # Find the number of new samples in the middle blob
    >>> n_res_in_middle = ((X_res[:, 0] > -5) & (X_res[:, 0] < 5)).sum()
    >>> print("Samples in the middle blob: %s" % n_res_in_middle)
    Samples in the middle blob: 801
    >>> print("Middle blob unchanged: %s" % (n_res_in_middle == blobs[1] + 1))
    Middle blob unchanged: True
    >>> print("More 0 samples: %s" % ((y_res == 0).sum() > (y == 0).sum()))
    More 0 samples: True
    """

    _parameter_constraints: dict = {
        **BaseSMOTE._parameter_constraints,
        "kmeans_estimator": [
            HasMethods(["fit", "predict"]),
            Interval(numbers.Integral, 1, None, closed="left"),
            None,
        ],
        "cluster_balance_threshold": [StrOptions({"auto"}), numbers.Real],
        "density_exponent": [StrOptions({"auto"}), numbers.Real],
    }

    def __init__(
        self,
        *,
        sampling_strategy="auto",
        random_state=None,
        k_neighbors=2,
        n_jobs=None,
        kmeans_estimator=None,
        cluster_balance_threshold="auto",
        density_exponent="auto",
    ):
        super().__init__(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
            n_jobs=n_jobs,
        )
        self.kmeans_estimator = kmeans_estimator
        self.cluster_balance_threshold = cluster_balance_threshold
        self.density_exponent = density_exponent

    def _validate_estimator(self):
        super()._validate_estimator()
        if self.kmeans_estimator is None:
            self.kmeans_estimator_ = MiniBatchKMeans(random_state=self.random_state)
        elif isinstance(self.kmeans_estimator, int):
            self.kmeans_estimator_ = MiniBatchKMeans(
                n_clusters=self.kmeans_estimator,
                random_state=self.random_state,
            )
        else:
            self.kmeans_estimator_ = clone(self.kmeans_estimator)

        self.cluster_balance_threshold_ = (
            self.cluster_balance_threshold
            if self.kmeans_estimator_.n_clusters != 1
            else -np.inf
        )

    def _find_cluster_sparsity(self, X):
        """Compute the cluster sparsity."""
        euclidean_distances = pairwise_distances(
            X, metric="euclidean", n_jobs=self.n_jobs
        )
        # negate diagonal elements
        for ind in range(X.shape[0]):
            euclidean_distances[ind, ind] = 0

        non_diag_elements = (X.shape[0] ** 2) - X.shape[0]
        mean_distance = euclidean_distances.sum() / non_diag_elements
        exponent = (
            math.log(X.shape[0], 1.6) ** 1.8 * 0.16
            if self.density_exponent == "auto"
            else self.density_exponent
        )
        return (mean_distance**exponent) / X.shape[0]

    def _fit_resample(self, X, y):
        self._validate_estimator()
        X_resampled = X.copy()
        y_resampled = y.copy()
        total_inp_samples = sum(self.sampling_strategy_.values())

        for class_sample, n_samples in self.sampling_strategy_.items():
            if n_samples == 0:
                continue

            X_clusters = self.kmeans_estimator_.fit_predict(X)
            valid_clusters = []
            cluster_sparsities = []

            # identify cluster which are answering the requirements
            for cluster_idx in range(self.kmeans_estimator_.n_clusters):
                cluster_mask = np.flatnonzero(X_clusters == cluster_idx)

                if cluster_mask.size == 0:
                    # empty cluster
                    continue

                X_cluster = _safe_indexing(X, cluster_mask)
                y_cluster = _safe_indexing(y, cluster_mask)

                cluster_class_mean = (y_cluster == class_sample).mean()

                if self.cluster_balance_threshold_ == "auto":
                    balance_threshold = n_samples / total_inp_samples / 2
                else:
                    balance_threshold = self.cluster_balance_threshold_

                # the cluster is already considered balanced
                if cluster_class_mean < balance_threshold:
                    continue

                # not enough samples to apply SMOTE
                anticipated_samples = cluster_class_mean * X_cluster.shape[0]
                if anticipated_samples < self.nn_k_.n_neighbors:
                    continue

                X_cluster_class = _safe_indexing(
                    X_cluster, np.flatnonzero(y_cluster == class_sample)
                )

                valid_clusters.append(cluster_mask)
                cluster_sparsities.append(self._find_cluster_sparsity(X_cluster_class))

            cluster_sparsities = np.array(cluster_sparsities)
            cluster_weights = cluster_sparsities / cluster_sparsities.sum()

            if not valid_clusters:
                raise RuntimeError(
                    f"No clusters found with sufficient samples of "
                    f"class {class_sample}. Try lowering the "
                    f"cluster_balance_threshold or increasing the number of "
                    f"clusters."
                )

            for valid_cluster_idx, valid_cluster in enumerate(valid_clusters):
                X_cluster = _safe_indexing(X, valid_cluster)
                y_cluster = _safe_indexing(y, valid_cluster)

                X_cluster_class = _safe_indexing(
                    X_cluster, np.flatnonzero(y_cluster == class_sample)
                )

                self.nn_k_.fit(X_cluster_class)
                nns = self.nn_k_.kneighbors(X_cluster_class, return_distance=False)[
                    :, 1:
                ]

                cluster_n_samples = int(
                    math.ceil(n_samples * cluster_weights[valid_cluster_idx])
                )

                X_new, y_new = self._make_samples(
                    X_cluster_class,
                    y.dtype,
                    class_sample,
                    X_cluster_class,
                    nns,
                    cluster_n_samples,
                    1.0,
                )

                stack = [np.vstack, sparse.vstack][int(sparse.issparse(X_new))]
                X_resampled = stack((X_resampled, X_new))
                y_resampled = np.hstack((y_resampled, y_new))

        return X_resampled, y_resampled
