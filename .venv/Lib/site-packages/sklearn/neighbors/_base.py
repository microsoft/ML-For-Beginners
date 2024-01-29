"""Base and mixin classes for nearest neighbors."""
# Authors: Jake Vanderplas <vanderplas@astro.washington.edu>
#          Fabian Pedregosa <fabian.pedregosa@inria.fr>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Sparseness support by Lars Buitinck
#          Multi-output support by Arnaud Joly <a.joly@ulg.ac.be>
#
# License: BSD 3 clause (C) INRIA, University of Amsterdam
import itertools
import numbers
import warnings
from abc import ABCMeta, abstractmethod
from functools import partial
from numbers import Integral, Real

import numpy as np
from joblib import effective_n_jobs
from scipy.sparse import csr_matrix, issparse

from ..base import BaseEstimator, MultiOutputMixin, is_classifier
from ..exceptions import DataConversionWarning, EfficiencyWarning
from ..metrics import DistanceMetric, pairwise_distances_chunked
from ..metrics._pairwise_distances_reduction import (
    ArgKmin,
    RadiusNeighbors,
)
from ..metrics.pairwise import PAIRWISE_DISTANCE_FUNCTIONS
from ..utils import (
    _to_object_array,
    check_array,
    gen_even_slices,
)
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.fixes import parse_version, sp_base_version
from ..utils.multiclass import check_classification_targets
from ..utils.parallel import Parallel, delayed
from ..utils.validation import check_is_fitted, check_non_negative
from ._ball_tree import BallTree
from ._kd_tree import KDTree

SCIPY_METRICS = [
    "braycurtis",
    "canberra",
    "chebyshev",
    "correlation",
    "cosine",
    "dice",
    "hamming",
    "jaccard",
    "mahalanobis",
    "minkowski",
    "rogerstanimoto",
    "russellrao",
    "seuclidean",
    "sokalmichener",
    "sokalsneath",
    "sqeuclidean",
    "yule",
]
if sp_base_version < parse_version("1.11"):
    # Deprecated in SciPy 1.9 and removed in SciPy 1.11
    SCIPY_METRICS += ["kulsinski"]
if sp_base_version < parse_version("1.9"):
    # Deprecated in SciPy 1.0 and removed in SciPy 1.9
    SCIPY_METRICS += ["matching"]

VALID_METRICS = dict(
    ball_tree=BallTree.valid_metrics,
    kd_tree=KDTree.valid_metrics,
    # The following list comes from the
    # sklearn.metrics.pairwise doc string
    brute=sorted(set(PAIRWISE_DISTANCE_FUNCTIONS).union(SCIPY_METRICS)),
)

VALID_METRICS_SPARSE = dict(
    ball_tree=[],
    kd_tree=[],
    brute=(PAIRWISE_DISTANCE_FUNCTIONS.keys() - {"haversine", "nan_euclidean"}),
)


def _get_weights(dist, weights):
    """Get the weights from an array of distances and a parameter ``weights``.

    Assume weights have already been validated.

    Parameters
    ----------
    dist : ndarray
        The input distances.

    weights : {'uniform', 'distance'}, callable or None
        The kind of weighting used.

    Returns
    -------
    weights_arr : array of the same shape as ``dist``
        If ``weights == 'uniform'``, then returns None.
    """
    if weights in (None, "uniform"):
        return None

    if weights == "distance":
        # if user attempts to classify a point that was zero distance from one
        # or more training points, those training points are weighted as 1.0
        # and the other points as 0.0
        if dist.dtype is np.dtype(object):
            for point_dist_i, point_dist in enumerate(dist):
                # check if point_dist is iterable
                # (ex: RadiusNeighborClassifier.predict may set an element of
                # dist to 1e-6 to represent an 'outlier')
                if hasattr(point_dist, "__contains__") and 0.0 in point_dist:
                    dist[point_dist_i] = point_dist == 0.0
                else:
                    dist[point_dist_i] = 1.0 / point_dist
        else:
            with np.errstate(divide="ignore"):
                dist = 1.0 / dist
            inf_mask = np.isinf(dist)
            inf_row = np.any(inf_mask, axis=1)
            dist[inf_row] = inf_mask[inf_row]
        return dist

    if callable(weights):
        return weights(dist)


def _is_sorted_by_data(graph):
    """Return whether the graph's non-zero entries are sorted by data.

    The non-zero entries are stored in graph.data and graph.indices.
    For each row (or sample), the non-zero entries can be either:
        - sorted by indices, as after graph.sort_indices();
        - sorted by data, as after _check_precomputed(graph);
        - not sorted.

    Parameters
    ----------
    graph : sparse matrix of shape (n_samples, n_samples)
        Neighbors graph as given by `kneighbors_graph` or
        `radius_neighbors_graph`. Matrix should be of format CSR format.

    Returns
    -------
    res : bool
        Whether input graph is sorted by data.
    """
    assert graph.format == "csr"
    out_of_order = graph.data[:-1] > graph.data[1:]
    line_change = np.unique(graph.indptr[1:-1] - 1)
    line_change = line_change[line_change < out_of_order.shape[0]]
    return out_of_order.sum() == out_of_order[line_change].sum()


def _check_precomputed(X):
    """Check precomputed distance matrix.

    If the precomputed distance matrix is sparse, it checks that the non-zero
    entries are sorted by distances. If not, the matrix is copied and sorted.

    Parameters
    ----------
    X : {sparse matrix, array-like}, (n_samples, n_samples)
        Distance matrix to other samples. X may be a sparse matrix, in which
        case only non-zero elements may be considered neighbors.

    Returns
    -------
    X : {sparse matrix, array-like}, (n_samples, n_samples)
        Distance matrix to other samples. X may be a sparse matrix, in which
        case only non-zero elements may be considered neighbors.
    """
    if not issparse(X):
        X = check_array(X)
        check_non_negative(X, whom="precomputed distance matrix.")
        return X
    else:
        graph = X

    if graph.format not in ("csr", "csc", "coo", "lil"):
        raise TypeError(
            "Sparse matrix in {!r} format is not supported due to "
            "its handling of explicit zeros".format(graph.format)
        )
    copied = graph.format != "csr"
    graph = check_array(graph, accept_sparse="csr")
    check_non_negative(graph, whom="precomputed distance matrix.")
    graph = sort_graph_by_row_values(graph, copy=not copied, warn_when_not_sorted=True)

    return graph


@validate_params(
    {
        "graph": ["sparse matrix"],
        "copy": ["boolean"],
        "warn_when_not_sorted": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def sort_graph_by_row_values(graph, copy=False, warn_when_not_sorted=True):
    """Sort a sparse graph such that each row is stored with increasing values.

    .. versionadded:: 1.2

    Parameters
    ----------
    graph : sparse matrix of shape (n_samples, n_samples)
        Distance matrix to other samples, where only non-zero elements are
        considered neighbors. Matrix is converted to CSR format if not already.

    copy : bool, default=False
        If True, the graph is copied before sorting. If False, the sorting is
        performed inplace. If the graph is not of CSR format, `copy` must be
        True to allow the conversion to CSR format, otherwise an error is
        raised.

    warn_when_not_sorted : bool, default=True
        If True, a :class:`~sklearn.exceptions.EfficiencyWarning` is raised
        when the input graph is not sorted by row values.

    Returns
    -------
    graph : sparse matrix of shape (n_samples, n_samples)
        Distance matrix to other samples, where only non-zero elements are
        considered neighbors. Matrix is in CSR format.

    Examples
    --------
    >>> from scipy.sparse import csr_matrix
    >>> from sklearn.neighbors import sort_graph_by_row_values
    >>> X = csr_matrix(
    ...     [[0., 3., 1.],
    ...      [3., 0., 2.],
    ...      [1., 2., 0.]])
    >>> X.data
    array([3., 1., 3., 2., 1., 2.])
    >>> X_ = sort_graph_by_row_values(X)
    >>> X_.data
    array([1., 3., 2., 3., 1., 2.])
    """
    if graph.format == "csr" and _is_sorted_by_data(graph):
        return graph

    if warn_when_not_sorted:
        warnings.warn(
            (
                "Precomputed sparse input was not sorted by row values. Use the"
                " function sklearn.neighbors.sort_graph_by_row_values to sort the input"
                " by row values, with warn_when_not_sorted=False to remove this"
                " warning."
            ),
            EfficiencyWarning,
        )

    if graph.format not in ("csr", "csc", "coo", "lil"):
        raise TypeError(
            f"Sparse matrix in {graph.format!r} format is not supported due to "
            "its handling of explicit zeros"
        )
    elif graph.format != "csr":
        if not copy:
            raise ValueError(
                "The input graph is not in CSR format. Use copy=True to allow "
                "the conversion to CSR format."
            )
        graph = graph.asformat("csr")
    elif copy:  # csr format with copy=True
        graph = graph.copy()

    row_nnz = np.diff(graph.indptr)
    if row_nnz.max() == row_nnz.min():
        # if each sample has the same number of provided neighbors
        n_samples = graph.shape[0]
        distances = graph.data.reshape(n_samples, -1)

        order = np.argsort(distances, kind="mergesort")
        order += np.arange(n_samples)[:, None] * row_nnz[0]
        order = order.ravel()
        graph.data = graph.data[order]
        graph.indices = graph.indices[order]

    else:
        for start, stop in zip(graph.indptr, graph.indptr[1:]):
            order = np.argsort(graph.data[start:stop], kind="mergesort")
            graph.data[start:stop] = graph.data[start:stop][order]
            graph.indices[start:stop] = graph.indices[start:stop][order]

    return graph


def _kneighbors_from_graph(graph, n_neighbors, return_distance):
    """Decompose a nearest neighbors sparse graph into distances and indices.

    Parameters
    ----------
    graph : sparse matrix of shape (n_samples, n_samples)
        Neighbors graph as given by `kneighbors_graph` or
        `radius_neighbors_graph`. Matrix should be of format CSR format.

    n_neighbors : int
        Number of neighbors required for each sample.

    return_distance : bool
        Whether or not to return the distances.

    Returns
    -------
    neigh_dist : ndarray of shape (n_samples, n_neighbors)
        Distances to nearest neighbors. Only present if `return_distance=True`.

    neigh_ind : ndarray of shape (n_samples, n_neighbors)
        Indices of nearest neighbors.
    """
    n_samples = graph.shape[0]
    assert graph.format == "csr"

    # number of neighbors by samples
    row_nnz = np.diff(graph.indptr)
    row_nnz_min = row_nnz.min()
    if n_neighbors is not None and row_nnz_min < n_neighbors:
        raise ValueError(
            "%d neighbors per samples are required, but some samples have only"
            " %d neighbors in precomputed graph matrix. Decrease number of "
            "neighbors used or recompute the graph with more neighbors."
            % (n_neighbors, row_nnz_min)
        )

    def extract(a):
        # if each sample has the same number of provided neighbors
        if row_nnz.max() == row_nnz_min:
            return a.reshape(n_samples, -1)[:, :n_neighbors]
        else:
            idx = np.tile(np.arange(n_neighbors), (n_samples, 1))
            idx += graph.indptr[:-1, None]
            return a.take(idx, mode="clip").reshape(n_samples, n_neighbors)

    if return_distance:
        return extract(graph.data), extract(graph.indices)
    else:
        return extract(graph.indices)


def _radius_neighbors_from_graph(graph, radius, return_distance):
    """Decompose a nearest neighbors sparse graph into distances and indices.

    Parameters
    ----------
    graph : sparse matrix of shape (n_samples, n_samples)
        Neighbors graph as given by `kneighbors_graph` or
        `radius_neighbors_graph`. Matrix should be of format CSR format.

    radius : float
        Radius of neighborhoods which should be strictly positive.

    return_distance : bool
        Whether or not to return the distances.

    Returns
    -------
    neigh_dist : ndarray of shape (n_samples,) of arrays
        Distances to nearest neighbors. Only present if `return_distance=True`.

    neigh_ind : ndarray of shape (n_samples,) of arrays
        Indices of nearest neighbors.
    """
    assert graph.format == "csr"

    no_filter_needed = bool(graph.data.max() <= radius)

    if no_filter_needed:
        data, indices, indptr = graph.data, graph.indices, graph.indptr
    else:
        mask = graph.data <= radius
        if return_distance:
            data = np.compress(mask, graph.data)
        indices = np.compress(mask, graph.indices)
        indptr = np.concatenate(([0], np.cumsum(mask)))[graph.indptr]

    indices = indices.astype(np.intp, copy=no_filter_needed)

    if return_distance:
        neigh_dist = _to_object_array(np.split(data, indptr[1:-1]))
    neigh_ind = _to_object_array(np.split(indices, indptr[1:-1]))

    if return_distance:
        return neigh_dist, neigh_ind
    else:
        return neigh_ind


class NeighborsBase(MultiOutputMixin, BaseEstimator, metaclass=ABCMeta):
    """Base class for nearest neighbors estimators."""

    _parameter_constraints: dict = {
        "n_neighbors": [Interval(Integral, 1, None, closed="left"), None],
        "radius": [Interval(Real, 0, None, closed="both"), None],
        "algorithm": [StrOptions({"auto", "ball_tree", "kd_tree", "brute"})],
        "leaf_size": [Interval(Integral, 1, None, closed="left")],
        "p": [Interval(Real, 0, None, closed="right"), None],
        "metric": [StrOptions(set(itertools.chain(*VALID_METRICS.values()))), callable],
        "metric_params": [dict, None],
        "n_jobs": [Integral, None],
    }

    @abstractmethod
    def __init__(
        self,
        n_neighbors=None,
        radius=None,
        algorithm="auto",
        leaf_size=30,
        metric="minkowski",
        p=2,
        metric_params=None,
        n_jobs=None,
    ):
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.metric_params = metric_params
        self.p = p
        self.n_jobs = n_jobs

    def _check_algorithm_metric(self):
        if self.algorithm == "auto":
            if self.metric == "precomputed":
                alg_check = "brute"
            elif (
                callable(self.metric)
                or self.metric in VALID_METRICS["ball_tree"]
                or isinstance(self.metric, DistanceMetric)
            ):
                alg_check = "ball_tree"
            else:
                alg_check = "brute"
        else:
            alg_check = self.algorithm

        if callable(self.metric):
            if self.algorithm == "kd_tree":
                # callable metric is only valid for brute force and ball_tree
                raise ValueError(
                    "kd_tree does not support callable metric '%s'"
                    "Function call overhead will result"
                    "in very poor performance."
                    % self.metric
                )
        elif self.metric not in VALID_METRICS[alg_check] and not isinstance(
            self.metric, DistanceMetric
        ):
            raise ValueError(
                "Metric '%s' not valid. Use "
                "sorted(sklearn.neighbors.VALID_METRICS['%s']) "
                "to get valid options. "
                "Metric can also be a callable function." % (self.metric, alg_check)
            )

        if self.metric_params is not None and "p" in self.metric_params:
            if self.p is not None:
                warnings.warn(
                    (
                        "Parameter p is found in metric_params. "
                        "The corresponding parameter from __init__ "
                        "is ignored."
                    ),
                    SyntaxWarning,
                    stacklevel=3,
                )

    def _fit(self, X, y=None):
        if self._get_tags()["requires_y"]:
            if not isinstance(X, (KDTree, BallTree, NeighborsBase)):
                X, y = self._validate_data(
                    X, y, accept_sparse="csr", multi_output=True, order="C"
                )

            if is_classifier(self):
                # Classification targets require a specific format
                if y.ndim == 1 or y.ndim == 2 and y.shape[1] == 1:
                    if y.ndim != 1:
                        warnings.warn(
                            (
                                "A column-vector y was passed when a "
                                "1d array was expected. Please change "
                                "the shape of y to (n_samples,), for "
                                "example using ravel()."
                            ),
                            DataConversionWarning,
                            stacklevel=2,
                        )

                    self.outputs_2d_ = False
                    y = y.reshape((-1, 1))
                else:
                    self.outputs_2d_ = True

                check_classification_targets(y)
                self.classes_ = []
                # Using `dtype=np.intp` is necessary since `np.bincount`
                # (called in _classification.py) fails when dealing
                # with a float64 array on 32bit systems.
                self._y = np.empty(y.shape, dtype=np.intp)
                for k in range(self._y.shape[1]):
                    classes, self._y[:, k] = np.unique(y[:, k], return_inverse=True)
                    self.classes_.append(classes)

                if not self.outputs_2d_:
                    self.classes_ = self.classes_[0]
                    self._y = self._y.ravel()
            else:
                self._y = y

        else:
            if not isinstance(X, (KDTree, BallTree, NeighborsBase)):
                X = self._validate_data(X, accept_sparse="csr", order="C")

        self._check_algorithm_metric()
        if self.metric_params is None:
            self.effective_metric_params_ = {}
        else:
            self.effective_metric_params_ = self.metric_params.copy()

        effective_p = self.effective_metric_params_.get("p", self.p)
        if self.metric == "minkowski":
            self.effective_metric_params_["p"] = effective_p

        self.effective_metric_ = self.metric
        # For minkowski distance, use more efficient methods where available
        if self.metric == "minkowski":
            p = self.effective_metric_params_.pop("p", 2)
            w = self.effective_metric_params_.pop("w", None)

            if p == 1 and w is None:
                self.effective_metric_ = "manhattan"
            elif p == 2 and w is None:
                self.effective_metric_ = "euclidean"
            elif p == np.inf and w is None:
                self.effective_metric_ = "chebyshev"
            else:
                # Use the generic minkowski metric, possibly weighted.
                self.effective_metric_params_["p"] = p
                self.effective_metric_params_["w"] = w

        if isinstance(X, NeighborsBase):
            self._fit_X = X._fit_X
            self._tree = X._tree
            self._fit_method = X._fit_method
            self.n_samples_fit_ = X.n_samples_fit_
            return self

        elif isinstance(X, BallTree):
            self._fit_X = X.data
            self._tree = X
            self._fit_method = "ball_tree"
            self.n_samples_fit_ = X.data.shape[0]
            return self

        elif isinstance(X, KDTree):
            self._fit_X = X.data
            self._tree = X
            self._fit_method = "kd_tree"
            self.n_samples_fit_ = X.data.shape[0]
            return self

        if self.metric == "precomputed":
            X = _check_precomputed(X)
            # Precomputed matrix X must be squared
            if X.shape[0] != X.shape[1]:
                raise ValueError(
                    "Precomputed matrix must be square."
                    " Input is a {}x{} matrix.".format(X.shape[0], X.shape[1])
                )
            self.n_features_in_ = X.shape[1]

        n_samples = X.shape[0]
        if n_samples == 0:
            raise ValueError("n_samples must be greater than 0")

        if issparse(X):
            if self.algorithm not in ("auto", "brute"):
                warnings.warn("cannot use tree with sparse input: using brute force")

            if (
                self.effective_metric_ not in VALID_METRICS_SPARSE["brute"]
                and not callable(self.effective_metric_)
                and not isinstance(self.effective_metric_, DistanceMetric)
            ):
                raise ValueError(
                    "Metric '%s' not valid for sparse input. "
                    "Use sorted(sklearn.neighbors."
                    "VALID_METRICS_SPARSE['brute']) "
                    "to get valid options. "
                    "Metric can also be a callable function." % (self.effective_metric_)
                )
            self._fit_X = X.copy()
            self._tree = None
            self._fit_method = "brute"
            self.n_samples_fit_ = X.shape[0]
            return self

        self._fit_method = self.algorithm
        self._fit_X = X
        self.n_samples_fit_ = X.shape[0]

        if self._fit_method == "auto":
            # A tree approach is better for small number of neighbors or small
            # number of features, with KDTree generally faster when available
            if (
                self.metric == "precomputed"
                or self._fit_X.shape[1] > 15
                or (
                    self.n_neighbors is not None
                    and self.n_neighbors >= self._fit_X.shape[0] // 2
                )
            ):
                self._fit_method = "brute"
            else:
                if (
                    self.effective_metric_ == "minkowski"
                    and self.effective_metric_params_["p"] < 1
                ):
                    self._fit_method = "brute"
                elif (
                    self.effective_metric_ == "minkowski"
                    and self.effective_metric_params_.get("w") is not None
                ):
                    # 'minkowski' with weights is not supported by KDTree but is
                    # supported byBallTree.
                    self._fit_method = "ball_tree"
                elif self.effective_metric_ in VALID_METRICS["kd_tree"]:
                    self._fit_method = "kd_tree"
                elif (
                    callable(self.effective_metric_)
                    or self.effective_metric_ in VALID_METRICS["ball_tree"]
                ):
                    self._fit_method = "ball_tree"
                else:
                    self._fit_method = "brute"

        if (
            self.effective_metric_ == "minkowski"
            and self.effective_metric_params_["p"] < 1
        ):
            # For 0 < p < 1 Minkowski distances aren't valid distance
            # metric as they do not satisfy triangular inequality:
            # they are semi-metrics.
            # algorithm="kd_tree" and algorithm="ball_tree" can't be used because
            # KDTree and BallTree require a proper distance metric to work properly.
            # However, the brute-force algorithm supports semi-metrics.
            if self._fit_method == "brute":
                warnings.warn(
                    "Mind that for 0 < p < 1, Minkowski metrics are not distance"
                    " metrics. Continuing the execution with `algorithm='brute'`."
                )
            else:  # self._fit_method in ("kd_tree", "ball_tree")
                raise ValueError(
                    f'algorithm="{self._fit_method}" does not support 0 < p < 1 for '
                    "the Minkowski metric. To resolve this problem either "
                    'set p >= 1 or algorithm="brute".'
                )

        if self._fit_method == "ball_tree":
            self._tree = BallTree(
                X,
                self.leaf_size,
                metric=self.effective_metric_,
                **self.effective_metric_params_,
            )
        elif self._fit_method == "kd_tree":
            if (
                self.effective_metric_ == "minkowski"
                and self.effective_metric_params_.get("w") is not None
            ):
                raise ValueError(
                    "algorithm='kd_tree' is not valid for "
                    "metric='minkowski' with a weight parameter 'w': "
                    "try algorithm='ball_tree' "
                    "or algorithm='brute' instead."
                )
            self._tree = KDTree(
                X,
                self.leaf_size,
                metric=self.effective_metric_,
                **self.effective_metric_params_,
            )
        elif self._fit_method == "brute":
            self._tree = None

        return self

    def _more_tags(self):
        # For cross-validation routines to split data correctly
        return {"pairwise": self.metric == "precomputed"}


def _tree_query_parallel_helper(tree, *args, **kwargs):
    """Helper for the Parallel calls in KNeighborsMixin.kneighbors.

    The Cython method tree.query is not directly picklable by cloudpickle
    under PyPy.
    """
    return tree.query(*args, **kwargs)


class KNeighborsMixin:
    """Mixin for k-neighbors searches."""

    def _kneighbors_reduce_func(self, dist, start, n_neighbors, return_distance):
        """Reduce a chunk of distances to the nearest neighbors.

        Callback to :func:`sklearn.metrics.pairwise.pairwise_distances_chunked`

        Parameters
        ----------
        dist : ndarray of shape (n_samples_chunk, n_samples)
            The distance matrix.

        start : int
            The index in X which the first row of dist corresponds to.

        n_neighbors : int
            Number of neighbors required for each sample.

        return_distance : bool
            Whether or not to return the distances.

        Returns
        -------
        dist : array of shape (n_samples_chunk, n_neighbors)
            Returned only if `return_distance=True`.

        neigh : array of shape (n_samples_chunk, n_neighbors)
            The neighbors indices.
        """
        sample_range = np.arange(dist.shape[0])[:, None]
        neigh_ind = np.argpartition(dist, n_neighbors - 1, axis=1)
        neigh_ind = neigh_ind[:, :n_neighbors]
        # argpartition doesn't guarantee sorted order, so we sort again
        neigh_ind = neigh_ind[sample_range, np.argsort(dist[sample_range, neigh_ind])]
        if return_distance:
            if self.effective_metric_ == "euclidean":
                result = np.sqrt(dist[sample_range, neigh_ind]), neigh_ind
            else:
                result = dist[sample_range, neigh_ind], neigh_ind
        else:
            result = neigh_ind
        return result

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        """Find the K-neighbors of a point.

        Returns indices of and distances to the neighbors of each point.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_queries, n_features), \
            or (n_queries, n_indexed) if metric == 'precomputed', default=None
            The query point or points.
            If not provided, neighbors of each indexed point are returned.
            In this case, the query point is not considered its own neighbor.

        n_neighbors : int, default=None
            Number of neighbors required for each sample. The default is the
            value passed to the constructor.

        return_distance : bool, default=True
            Whether or not to return the distances.

        Returns
        -------
        neigh_dist : ndarray of shape (n_queries, n_neighbors)
            Array representing the lengths to points, only present if
            return_distance=True.

        neigh_ind : ndarray of shape (n_queries, n_neighbors)
            Indices of the nearest points in the population matrix.

        Examples
        --------
        In the following example, we construct a NearestNeighbors
        class from an array representing our data set and ask who's
        the closest point to [1,1,1]

        >>> samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]
        >>> from sklearn.neighbors import NearestNeighbors
        >>> neigh = NearestNeighbors(n_neighbors=1)
        >>> neigh.fit(samples)
        NearestNeighbors(n_neighbors=1)
        >>> print(neigh.kneighbors([[1., 1., 1.]]))
        (array([[0.5]]), array([[2]]))

        As you can see, it returns [[0.5]], and [[2]], which means that the
        element is at distance 0.5 and is the third element of samples
        (indexes start at 0). You can also query for multiple points:

        >>> X = [[0., 1., 0.], [1., 0., 1.]]
        >>> neigh.kneighbors(X, return_distance=False)
        array([[1],
               [2]]...)
        """
        check_is_fitted(self)

        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        elif n_neighbors <= 0:
            raise ValueError("Expected n_neighbors > 0. Got %d" % n_neighbors)
        elif not isinstance(n_neighbors, numbers.Integral):
            raise TypeError(
                "n_neighbors does not take %s value, enter integer value"
                % type(n_neighbors)
            )

        query_is_train = X is None
        if query_is_train:
            X = self._fit_X
            # Include an extra neighbor to account for the sample itself being
            # returned, which is removed later
            n_neighbors += 1
        else:
            if self.metric == "precomputed":
                X = _check_precomputed(X)
            else:
                X = self._validate_data(X, accept_sparse="csr", reset=False, order="C")

        n_samples_fit = self.n_samples_fit_
        if n_neighbors > n_samples_fit:
            if query_is_train:
                n_neighbors -= 1  # ok to modify inplace because an error is raised
                inequality_str = "n_neighbors < n_samples_fit"
            else:
                inequality_str = "n_neighbors <= n_samples_fit"
            raise ValueError(
                f"Expected {inequality_str}, but "
                f"n_neighbors = {n_neighbors}, n_samples_fit = {n_samples_fit}, "
                f"n_samples = {X.shape[0]}"  # include n_samples for common tests
            )

        n_jobs = effective_n_jobs(self.n_jobs)
        chunked_results = None
        use_pairwise_distances_reductions = (
            self._fit_method == "brute"
            and ArgKmin.is_usable_for(
                X if X is not None else self._fit_X, self._fit_X, self.effective_metric_
            )
        )
        if use_pairwise_distances_reductions:
            results = ArgKmin.compute(
                X=X,
                Y=self._fit_X,
                k=n_neighbors,
                metric=self.effective_metric_,
                metric_kwargs=self.effective_metric_params_,
                strategy="auto",
                return_distance=return_distance,
            )

        elif (
            self._fit_method == "brute" and self.metric == "precomputed" and issparse(X)
        ):
            results = _kneighbors_from_graph(
                X, n_neighbors=n_neighbors, return_distance=return_distance
            )

        elif self._fit_method == "brute":
            # Joblib-based backend, which is used when user-defined callable
            # are passed for metric.

            # This won't be used in the future once PairwiseDistancesReductions
            # support:
            #   - DistanceMetrics which work on supposedly binary data
            #   - CSR-dense and dense-CSR case if 'euclidean' in metric.
            reduce_func = partial(
                self._kneighbors_reduce_func,
                n_neighbors=n_neighbors,
                return_distance=return_distance,
            )

            # for efficiency, use squared euclidean distances
            if self.effective_metric_ == "euclidean":
                kwds = {"squared": True}
            else:
                kwds = self.effective_metric_params_

            chunked_results = list(
                pairwise_distances_chunked(
                    X,
                    self._fit_X,
                    reduce_func=reduce_func,
                    metric=self.effective_metric_,
                    n_jobs=n_jobs,
                    **kwds,
                )
            )

        elif self._fit_method in ["ball_tree", "kd_tree"]:
            if issparse(X):
                raise ValueError(
                    "%s does not work with sparse matrices. Densify the data, "
                    "or set algorithm='brute'"
                    % self._fit_method
                )
            chunked_results = Parallel(n_jobs, prefer="threads")(
                delayed(_tree_query_parallel_helper)(
                    self._tree, X[s], n_neighbors, return_distance
                )
                for s in gen_even_slices(X.shape[0], n_jobs)
            )
        else:
            raise ValueError("internal: _fit_method not recognized")

        if chunked_results is not None:
            if return_distance:
                neigh_dist, neigh_ind = zip(*chunked_results)
                results = np.vstack(neigh_dist), np.vstack(neigh_ind)
            else:
                results = np.vstack(chunked_results)

        if not query_is_train:
            return results
        else:
            # If the query data is the same as the indexed data, we would like
            # to ignore the first nearest neighbor of every sample, i.e
            # the sample itself.
            if return_distance:
                neigh_dist, neigh_ind = results
            else:
                neigh_ind = results

            n_queries, _ = X.shape
            sample_range = np.arange(n_queries)[:, None]
            sample_mask = neigh_ind != sample_range

            # Corner case: When the number of duplicates are more
            # than the number of neighbors, the first NN will not
            # be the sample, but a duplicate.
            # In that case mask the first duplicate.
            dup_gr_nbrs = np.all(sample_mask, axis=1)
            sample_mask[:, 0][dup_gr_nbrs] = False
            neigh_ind = np.reshape(neigh_ind[sample_mask], (n_queries, n_neighbors - 1))

            if return_distance:
                neigh_dist = np.reshape(
                    neigh_dist[sample_mask], (n_queries, n_neighbors - 1)
                )
                return neigh_dist, neigh_ind
            return neigh_ind

    def kneighbors_graph(self, X=None, n_neighbors=None, mode="connectivity"):
        """Compute the (weighted) graph of k-Neighbors for points in X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_queries, n_features), \
            or (n_queries, n_indexed) if metric == 'precomputed', default=None
            The query point or points.
            If not provided, neighbors of each indexed point are returned.
            In this case, the query point is not considered its own neighbor.
            For ``metric='precomputed'`` the shape should be
            (n_queries, n_indexed). Otherwise the shape should be
            (n_queries, n_features).

        n_neighbors : int, default=None
            Number of neighbors for each sample. The default is the value
            passed to the constructor.

        mode : {'connectivity', 'distance'}, default='connectivity'
            Type of returned matrix: 'connectivity' will return the
            connectivity matrix with ones and zeros, in 'distance' the
            edges are distances between points, type of distance
            depends on the selected metric parameter in
            NearestNeighbors class.

        Returns
        -------
        A : sparse-matrix of shape (n_queries, n_samples_fit)
            `n_samples_fit` is the number of samples in the fitted data.
            `A[i, j]` gives the weight of the edge connecting `i` to `j`.
            The matrix is of CSR format.

        See Also
        --------
        NearestNeighbors.radius_neighbors_graph : Compute the (weighted) graph
            of Neighbors for points in X.

        Examples
        --------
        >>> X = [[0], [3], [1]]
        >>> from sklearn.neighbors import NearestNeighbors
        >>> neigh = NearestNeighbors(n_neighbors=2)
        >>> neigh.fit(X)
        NearestNeighbors(n_neighbors=2)
        >>> A = neigh.kneighbors_graph(X)
        >>> A.toarray()
        array([[1., 0., 1.],
               [0., 1., 1.],
               [1., 0., 1.]])
        """
        check_is_fitted(self)
        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        # check the input only in self.kneighbors

        # construct CSR matrix representation of the k-NN graph
        if mode == "connectivity":
            A_ind = self.kneighbors(X, n_neighbors, return_distance=False)
            n_queries = A_ind.shape[0]
            A_data = np.ones(n_queries * n_neighbors)

        elif mode == "distance":
            A_data, A_ind = self.kneighbors(X, n_neighbors, return_distance=True)
            A_data = np.ravel(A_data)

        else:
            raise ValueError(
                'Unsupported mode, must be one of "connectivity", '
                f'or "distance" but got "{mode}" instead'
            )

        n_queries = A_ind.shape[0]
        n_samples_fit = self.n_samples_fit_
        n_nonzero = n_queries * n_neighbors
        A_indptr = np.arange(0, n_nonzero + 1, n_neighbors)

        kneighbors_graph = csr_matrix(
            (A_data, A_ind.ravel(), A_indptr), shape=(n_queries, n_samples_fit)
        )

        return kneighbors_graph


def _tree_query_radius_parallel_helper(tree, *args, **kwargs):
    """Helper for the Parallel calls in RadiusNeighborsMixin.radius_neighbors.

    The Cython method tree.query_radius is not directly picklable by
    cloudpickle under PyPy.
    """
    return tree.query_radius(*args, **kwargs)


class RadiusNeighborsMixin:
    """Mixin for radius-based neighbors searches."""

    def _radius_neighbors_reduce_func(self, dist, start, radius, return_distance):
        """Reduce a chunk of distances to the nearest neighbors.

        Callback to :func:`sklearn.metrics.pairwise.pairwise_distances_chunked`

        Parameters
        ----------
        dist : ndarray of shape (n_samples_chunk, n_samples)
            The distance matrix.

        start : int
            The index in X which the first row of dist corresponds to.

        radius : float
            The radius considered when making the nearest neighbors search.

        return_distance : bool
            Whether or not to return the distances.

        Returns
        -------
        dist : list of ndarray of shape (n_samples_chunk,)
            Returned only if `return_distance=True`.

        neigh : list of ndarray of shape (n_samples_chunk,)
            The neighbors indices.
        """
        neigh_ind = [np.where(d <= radius)[0] for d in dist]

        if return_distance:
            if self.effective_metric_ == "euclidean":
                dist = [np.sqrt(d[neigh_ind[i]]) for i, d in enumerate(dist)]
            else:
                dist = [d[neigh_ind[i]] for i, d in enumerate(dist)]
            results = dist, neigh_ind
        else:
            results = neigh_ind
        return results

    def radius_neighbors(
        self, X=None, radius=None, return_distance=True, sort_results=False
    ):
        """Find the neighbors within a given radius of a point or points.

        Return the indices and distances of each point from the dataset
        lying in a ball with size ``radius`` around the points of the query
        array. Points lying on the boundary are included in the results.

        The result points are *not* necessarily sorted by distance to their
        query point.

        Parameters
        ----------
        X : {array-like, sparse matrix} of (n_samples, n_features), default=None
            The query point or points.
            If not provided, neighbors of each indexed point are returned.
            In this case, the query point is not considered its own neighbor.

        radius : float, default=None
            Limiting distance of neighbors to return. The default is the value
            passed to the constructor.

        return_distance : bool, default=True
            Whether or not to return the distances.

        sort_results : bool, default=False
            If True, the distances and indices will be sorted by increasing
            distances before being returned. If False, the results may not
            be sorted. If `return_distance=False`, setting `sort_results=True`
            will result in an error.

            .. versionadded:: 0.22

        Returns
        -------
        neigh_dist : ndarray of shape (n_samples,) of arrays
            Array representing the distances to each point, only present if
            `return_distance=True`. The distance values are computed according
            to the ``metric`` constructor parameter.

        neigh_ind : ndarray of shape (n_samples,) of arrays
            An array of arrays of indices of the approximate nearest points
            from the population matrix that lie within a ball of size
            ``radius`` around the query points.

        Notes
        -----
        Because the number of neighbors of each point is not necessarily
        equal, the results for multiple query points cannot be fit in a
        standard data array.
        For efficiency, `radius_neighbors` returns arrays of objects, where
        each object is a 1D array of indices or distances.

        Examples
        --------
        In the following example, we construct a NeighborsClassifier
        class from an array representing our data set and ask who's
        the closest point to [1, 1, 1]:

        >>> import numpy as np
        >>> samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]
        >>> from sklearn.neighbors import NearestNeighbors
        >>> neigh = NearestNeighbors(radius=1.6)
        >>> neigh.fit(samples)
        NearestNeighbors(radius=1.6)
        >>> rng = neigh.radius_neighbors([[1., 1., 1.]])
        >>> print(np.asarray(rng[0][0]))
        [1.5 0.5]
        >>> print(np.asarray(rng[1][0]))
        [1 2]

        The first array returned contains the distances to all points which
        are closer than 1.6, while the second array returned contains their
        indices.  In general, multiple points can be queried at the same time.
        """
        check_is_fitted(self)

        if sort_results and not return_distance:
            raise ValueError("return_distance must be True if sort_results is True.")

        query_is_train = X is None
        if query_is_train:
            X = self._fit_X
        else:
            if self.metric == "precomputed":
                X = _check_precomputed(X)
            else:
                X = self._validate_data(X, accept_sparse="csr", reset=False, order="C")

        if radius is None:
            radius = self.radius

        use_pairwise_distances_reductions = (
            self._fit_method == "brute"
            and RadiusNeighbors.is_usable_for(
                X if X is not None else self._fit_X, self._fit_X, self.effective_metric_
            )
        )

        if use_pairwise_distances_reductions:
            results = RadiusNeighbors.compute(
                X=X,
                Y=self._fit_X,
                radius=radius,
                metric=self.effective_metric_,
                metric_kwargs=self.effective_metric_params_,
                strategy="auto",
                return_distance=return_distance,
                sort_results=sort_results,
            )

        elif (
            self._fit_method == "brute" and self.metric == "precomputed" and issparse(X)
        ):
            results = _radius_neighbors_from_graph(
                X, radius=radius, return_distance=return_distance
            )

        elif self._fit_method == "brute":
            # Joblib-based backend, which is used when user-defined callable
            # are passed for metric.

            # This won't be used in the future once PairwiseDistancesReductions
            # support:
            #   - DistanceMetrics which work on supposedly binary data
            #   - CSR-dense and dense-CSR case if 'euclidean' in metric.

            # for efficiency, use squared euclidean distances
            if self.effective_metric_ == "euclidean":
                radius *= radius
                kwds = {"squared": True}
            else:
                kwds = self.effective_metric_params_

            reduce_func = partial(
                self._radius_neighbors_reduce_func,
                radius=radius,
                return_distance=return_distance,
            )

            chunked_results = pairwise_distances_chunked(
                X,
                self._fit_X,
                reduce_func=reduce_func,
                metric=self.effective_metric_,
                n_jobs=self.n_jobs,
                **kwds,
            )
            if return_distance:
                neigh_dist_chunks, neigh_ind_chunks = zip(*chunked_results)
                neigh_dist_list = sum(neigh_dist_chunks, [])
                neigh_ind_list = sum(neigh_ind_chunks, [])
                neigh_dist = _to_object_array(neigh_dist_list)
                neigh_ind = _to_object_array(neigh_ind_list)
                results = neigh_dist, neigh_ind
            else:
                neigh_ind_list = sum(chunked_results, [])
                results = _to_object_array(neigh_ind_list)

            if sort_results:
                for ii in range(len(neigh_dist)):
                    order = np.argsort(neigh_dist[ii], kind="mergesort")
                    neigh_ind[ii] = neigh_ind[ii][order]
                    neigh_dist[ii] = neigh_dist[ii][order]
                results = neigh_dist, neigh_ind

        elif self._fit_method in ["ball_tree", "kd_tree"]:
            if issparse(X):
                raise ValueError(
                    "%s does not work with sparse matrices. Densify the data, "
                    "or set algorithm='brute'"
                    % self._fit_method
                )

            n_jobs = effective_n_jobs(self.n_jobs)
            delayed_query = delayed(_tree_query_radius_parallel_helper)
            chunked_results = Parallel(n_jobs, prefer="threads")(
                delayed_query(
                    self._tree, X[s], radius, return_distance, sort_results=sort_results
                )
                for s in gen_even_slices(X.shape[0], n_jobs)
            )
            if return_distance:
                neigh_ind, neigh_dist = tuple(zip(*chunked_results))
                results = np.hstack(neigh_dist), np.hstack(neigh_ind)
            else:
                results = np.hstack(chunked_results)
        else:
            raise ValueError("internal: _fit_method not recognized")

        if not query_is_train:
            return results
        else:
            # If the query data is the same as the indexed data, we would like
            # to ignore the first nearest neighbor of every sample, i.e
            # the sample itself.
            if return_distance:
                neigh_dist, neigh_ind = results
            else:
                neigh_ind = results

            for ind, ind_neighbor in enumerate(neigh_ind):
                mask = ind_neighbor != ind

                neigh_ind[ind] = ind_neighbor[mask]
                if return_distance:
                    neigh_dist[ind] = neigh_dist[ind][mask]

            if return_distance:
                return neigh_dist, neigh_ind
            return neigh_ind

    def radius_neighbors_graph(
        self, X=None, radius=None, mode="connectivity", sort_results=False
    ):
        """Compute the (weighted) graph of Neighbors for points in X.

        Neighborhoods are restricted the points at a distance lower than
        radius.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features), default=None
            The query point or points.
            If not provided, neighbors of each indexed point are returned.
            In this case, the query point is not considered its own neighbor.

        radius : float, default=None
            Radius of neighborhoods. The default is the value passed to the
            constructor.

        mode : {'connectivity', 'distance'}, default='connectivity'
            Type of returned matrix: 'connectivity' will return the
            connectivity matrix with ones and zeros, in 'distance' the
            edges are distances between points, type of distance
            depends on the selected metric parameter in
            NearestNeighbors class.

        sort_results : bool, default=False
            If True, in each row of the result, the non-zero entries will be
            sorted by increasing distances. If False, the non-zero entries may
            not be sorted. Only used with mode='distance'.

            .. versionadded:: 0.22

        Returns
        -------
        A : sparse-matrix of shape (n_queries, n_samples_fit)
            `n_samples_fit` is the number of samples in the fitted data.
            `A[i, j]` gives the weight of the edge connecting `i` to `j`.
            The matrix is of CSR format.

        See Also
        --------
        kneighbors_graph : Compute the (weighted) graph of k-Neighbors for
            points in X.

        Examples
        --------
        >>> X = [[0], [3], [1]]
        >>> from sklearn.neighbors import NearestNeighbors
        >>> neigh = NearestNeighbors(radius=1.5)
        >>> neigh.fit(X)
        NearestNeighbors(radius=1.5)
        >>> A = neigh.radius_neighbors_graph(X)
        >>> A.toarray()
        array([[1., 0., 1.],
               [0., 1., 0.],
               [1., 0., 1.]])
        """
        check_is_fitted(self)

        # check the input only in self.radius_neighbors

        if radius is None:
            radius = self.radius

        # construct CSR matrix representation of the NN graph
        if mode == "connectivity":
            A_ind = self.radius_neighbors(X, radius, return_distance=False)
            A_data = None
        elif mode == "distance":
            dist, A_ind = self.radius_neighbors(
                X, radius, return_distance=True, sort_results=sort_results
            )
            A_data = np.concatenate(list(dist))
        else:
            raise ValueError(
                'Unsupported mode, must be one of "connectivity", '
                f'or "distance" but got "{mode}" instead'
            )

        n_queries = A_ind.shape[0]
        n_samples_fit = self.n_samples_fit_
        n_neighbors = np.array([len(a) for a in A_ind])
        A_ind = np.concatenate(list(A_ind))
        if A_data is None:
            A_data = np.ones(len(A_ind))
        A_indptr = np.concatenate((np.zeros(1, dtype=int), np.cumsum(n_neighbors)))

        return csr_matrix((A_data, A_ind, A_indptr), shape=(n_queries, n_samples_fit))
