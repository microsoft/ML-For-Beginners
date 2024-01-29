"""
HDBSCAN: Hierarchical Density-Based Spatial Clustering
         of Applications with Noise
"""
# Authors: Leland McInnes <leland.mcinnes@gmail.com>
#          Steve Astels <sastels@gmail.com>
#          John Healy <jchealy@gmail.com>
#          Meekail Zain <zainmeekail@gmail.com>
# Copyright (c) 2015, Leland McInnes
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from numbers import Integral, Real
from warnings import warn

import numpy as np
from scipy.sparse import csgraph, issparse

from ...base import BaseEstimator, ClusterMixin, _fit_context
from ...metrics import pairwise_distances
from ...metrics._dist_metrics import DistanceMetric
from ...neighbors import BallTree, KDTree, NearestNeighbors
from ...utils._param_validation import Interval, StrOptions
from ...utils.validation import _allclose_dense_sparse, _assert_all_finite
from ._linkage import (
    MST_edge_dtype,
    make_single_linkage,
    mst_from_data_matrix,
    mst_from_mutual_reachability,
)
from ._reachability import mutual_reachability_graph
from ._tree import HIERARCHY_dtype, labelling_at_cut, tree_to_labels

FAST_METRICS = set(KDTree.valid_metrics + BallTree.valid_metrics)

# Encodings are arbitrary but must be strictly negative.
# The current encodings are chosen as extensions to the -1 noise label.
# Avoided enums so that the end user only deals with simple labels.
_OUTLIER_ENCODING: dict = {
    "infinite": {
        "label": -2,
        # The probability could also be 1, since infinite points are certainly
        # infinite outliers, however 0 is convention from the HDBSCAN library
        # implementation.
        "prob": 0,
    },
    "missing": {
        "label": -3,
        # A nan probability is chosen to emphasize the fact that the
        # corresponding data was not considered in the clustering problem.
        "prob": np.nan,
    },
}


def _brute_mst(mutual_reachability, min_samples):
    """
    Builds a minimum spanning tree (MST) from the provided mutual-reachability
    values. This function dispatches to a custom Cython implementation for
    dense arrays, and `scipy.sparse.csgraph.minimum_spanning_tree` for sparse
    arrays/matrices.

    Parameters
    ----------
    mututal_reachability_graph: {ndarray, sparse matrix} of shape \
            (n_samples, n_samples)
        Weighted adjacency matrix of the mutual reachability graph.

    min_samples : int, default=None
        The number of samples in a neighborhood for a point
        to be considered as a core point. This includes the point itself.

    Returns
    -------
    mst : ndarray of shape (n_samples - 1,), dtype=MST_edge_dtype
        The MST representation of the mutual-reachability graph. The MST is
        represented as a collection of edges.
    """
    if not issparse(mutual_reachability):
        return mst_from_mutual_reachability(mutual_reachability)

    # Check if the mutual reachability matrix has any rows which have
    # less than `min_samples` non-zero elements.
    indptr = mutual_reachability.indptr
    num_points = mutual_reachability.shape[0]
    if any((indptr[i + 1] - indptr[i]) < min_samples for i in range(num_points)):
        raise ValueError(
            f"There exists points with fewer than {min_samples} neighbors. Ensure"
            " your distance matrix has non-zero values for at least"
            f" `min_sample`={min_samples} neighbors for each points (i.e. K-nn"
            " graph), or specify a `max_distance` in `metric_params` to use when"
            " distances are missing."
        )
    # Check connected component on mutual reachability.
    # If more than one connected component is present,
    # it means that the graph is disconnected.
    n_components = csgraph.connected_components(
        mutual_reachability, directed=False, return_labels=False
    )
    if n_components > 1:
        raise ValueError(
            f"Sparse mutual reachability matrix has {n_components} connected"
            " components. HDBSCAN cannot be perfomed on a disconnected graph. Ensure"
            " that the sparse distance matrix has only one connected component."
        )

    # Compute the minimum spanning tree for the sparse graph
    sparse_min_spanning_tree = csgraph.minimum_spanning_tree(mutual_reachability)
    rows, cols = sparse_min_spanning_tree.nonzero()
    mst = np.rec.fromarrays(
        [rows, cols, sparse_min_spanning_tree.data],
        dtype=MST_edge_dtype,
    )
    return mst


def _process_mst(min_spanning_tree):
    """
    Builds a single-linkage tree (SLT) from the provided minimum spanning tree
    (MST). The MST is first sorted then processed by a custom Cython routine.

    Parameters
    ----------
    min_spanning_tree : ndarray of shape (n_samples - 1,), dtype=MST_edge_dtype
        The MST representation of the mutual-reachability graph. The MST is
        represented as a collection of edges.

    Returns
    -------
    single_linkage : ndarray of shape (n_samples - 1,), dtype=HIERARCHY_dtype
        The single-linkage tree tree (dendrogram) built from the MST.
    """
    # Sort edges of the min_spanning_tree by weight
    row_order = np.argsort(min_spanning_tree["distance"])
    min_spanning_tree = min_spanning_tree[row_order]
    # Convert edge list into standard hierarchical clustering format
    return make_single_linkage(min_spanning_tree)


def _hdbscan_brute(
    X,
    min_samples=5,
    alpha=None,
    metric="euclidean",
    n_jobs=None,
    copy=False,
    **metric_params,
):
    """
    Builds a single-linkage tree (SLT) from the input data `X`. If
    `metric="precomputed"` then `X` must be a symmetric array of distances.
    Otherwise, the pairwise distances are calculated directly and passed to
    `mutual_reachability_graph`.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features) or (n_samples, n_samples)
        Either the raw data from which to compute the pairwise distances,
        or the precomputed distances.

    min_samples : int, default=None
        The number of samples in a neighborhood for a point
        to be considered as a core point. This includes the point itself.

    alpha : float, default=1.0
        A distance scaling parameter as used in robust single linkage.

    metric : str or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array.

        - If metric is a string or callable, it must be one of
          the options allowed by :func:`~sklearn.metrics.pairwise_distances`
          for its metric parameter.

        - If metric is "precomputed", X is assumed to be a distance matrix and
          must be square.

    n_jobs : int, default=None
        The number of jobs to use for computing the pairwise distances. This
        works by breaking down the pairwise matrix into n_jobs even slices and
        computing them in parallel. This parameter is passed directly to
        :func:`~sklearn.metrics.pairwise_distances`.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    copy : bool, default=False
        If `copy=True` then any time an in-place modifications would be made
        that would overwrite `X`, a copy will first be made, guaranteeing that
        the original data will be unchanged. Currently, it only applies when
        `metric="precomputed"`, when passing a dense array or a CSR sparse
        array/matrix.

    metric_params : dict, default=None
        Arguments passed to the distance metric.

    Returns
    -------
    single_linkage : ndarray of shape (n_samples - 1,), dtype=HIERARCHY_dtype
        The single-linkage tree tree (dendrogram) built from the MST.
    """
    if metric == "precomputed":
        if X.shape[0] != X.shape[1]:
            raise ValueError(
                "The precomputed distance matrix is expected to be symmetric, however"
                f" it has shape {X.shape}. Please verify that the"
                " distance matrix was constructed correctly."
            )
        if not _allclose_dense_sparse(X, X.T):
            raise ValueError(
                "The precomputed distance matrix is expected to be symmetric, however"
                " its values appear to be asymmetric. Please verify that the distance"
                " matrix was constructed correctly."
            )

        distance_matrix = X.copy() if copy else X
    else:
        distance_matrix = pairwise_distances(
            X, metric=metric, n_jobs=n_jobs, **metric_params
        )
    distance_matrix /= alpha

    max_distance = metric_params.get("max_distance", 0.0)
    if issparse(distance_matrix) and distance_matrix.format != "csr":
        # we need CSR format to avoid a conversion in `_brute_mst` when calling
        # `csgraph.connected_components`
        distance_matrix = distance_matrix.tocsr()

    # Note that `distance_matrix` is manipulated in-place, however we do not
    # need it for anything else past this point, hence the operation is safe.
    mutual_reachability_ = mutual_reachability_graph(
        distance_matrix, min_samples=min_samples, max_distance=max_distance
    )
    min_spanning_tree = _brute_mst(mutual_reachability_, min_samples=min_samples)
    # Warn if the MST couldn't be constructed around the missing distances
    if np.isinf(min_spanning_tree["distance"]).any():
        warn(
            (
                "The minimum spanning tree contains edge weights with value "
                "infinity. Potentially, you are missing too many distances "
                "in the initial distance matrix for the given neighborhood "
                "size."
            ),
            UserWarning,
        )
    return _process_mst(min_spanning_tree)


def _hdbscan_prims(
    X,
    algo,
    min_samples=5,
    alpha=1.0,
    metric="euclidean",
    leaf_size=40,
    n_jobs=None,
    **metric_params,
):
    """
    Builds a single-linkage tree (SLT) from the input data `X`. If
    `metric="precomputed"` then `X` must be a symmetric array of distances.
    Otherwise, the pairwise distances are calculated directly and passed to
    `mutual_reachability_graph`.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The raw data.

    min_samples : int, default=None
        The number of samples in a neighborhood for a point
        to be considered as a core point. This includes the point itself.

    alpha : float, default=1.0
        A distance scaling parameter as used in robust single linkage.

    metric : str or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array. `metric` must be one of the options allowed by
        :func:`~sklearn.metrics.pairwise_distances` for its metric
        parameter.

    n_jobs : int, default=None
        The number of jobs to use for computing the pairwise distances. This
        works by breaking down the pairwise matrix into n_jobs even slices and
        computing them in parallel. This parameter is passed directly to
        :func:`~sklearn.metrics.pairwise_distances`.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    copy : bool, default=False
        If `copy=True` then any time an in-place modifications would be made
        that would overwrite `X`, a copy will first be made, guaranteeing that
        the original data will be unchanged. Currently, it only applies when
        `metric="precomputed"`, when passing a dense array or a CSR sparse
        array/matrix.

    metric_params : dict, default=None
        Arguments passed to the distance metric.

    Returns
    -------
    single_linkage : ndarray of shape (n_samples - 1,), dtype=HIERARCHY_dtype
        The single-linkage tree tree (dendrogram) built from the MST.
    """
    # The Cython routines used require contiguous arrays
    X = np.asarray(X, order="C")

    # Get distance to kth nearest neighbour
    nbrs = NearestNeighbors(
        n_neighbors=min_samples,
        algorithm=algo,
        leaf_size=leaf_size,
        metric=metric,
        metric_params=metric_params,
        n_jobs=n_jobs,
        p=None,
    ).fit(X)

    neighbors_distances, _ = nbrs.kneighbors(X, min_samples, return_distance=True)
    core_distances = np.ascontiguousarray(neighbors_distances[:, -1])
    dist_metric = DistanceMetric.get_metric(metric, **metric_params)

    # Mutual reachability distance is implicit in mst_from_data_matrix
    min_spanning_tree = mst_from_data_matrix(X, core_distances, dist_metric, alpha)
    return _process_mst(min_spanning_tree)


def remap_single_linkage_tree(tree, internal_to_raw, non_finite):
    """
    Takes an internal single_linkage_tree structure and adds back in a set of points
    that were initially detected as non-finite and returns that new tree.
    These points will all be merged into the final node at np.inf distance and
    considered noise points.

    Parameters
    ----------
    tree : ndarray of shape (n_samples - 1,), dtype=HIERARCHY_dtype
        The single-linkage tree tree (dendrogram) built from the MST.
    internal_to_raw: dict
        A mapping from internal integer index to the raw integer index
    non_finite : ndarray
        Boolean array of which entries in the raw data are non-finite
    """
    finite_count = len(internal_to_raw)

    outlier_count = len(non_finite)
    for i, _ in enumerate(tree):
        left = tree[i]["left_node"]
        right = tree[i]["right_node"]

        if left < finite_count:
            tree[i]["left_node"] = internal_to_raw[left]
        else:
            tree[i]["left_node"] = left + outlier_count
        if right < finite_count:
            tree[i]["right_node"] = internal_to_raw[right]
        else:
            tree[i]["right_node"] = right + outlier_count

    outlier_tree = np.zeros(len(non_finite), dtype=HIERARCHY_dtype)
    last_cluster_id = max(
        tree[tree.shape[0] - 1]["left_node"], tree[tree.shape[0] - 1]["right_node"]
    )
    last_cluster_size = tree[tree.shape[0] - 1]["cluster_size"]
    for i, outlier in enumerate(non_finite):
        outlier_tree[i] = (outlier, last_cluster_id + 1, np.inf, last_cluster_size + 1)
        last_cluster_id += 1
        last_cluster_size += 1
    tree = np.concatenate([tree, outlier_tree])
    return tree


def _get_finite_row_indices(matrix):
    """
    Returns the indices of the purely finite rows of a
    sparse matrix or dense ndarray
    """
    if issparse(matrix):
        row_indices = np.array(
            [i for i, row in enumerate(matrix.tolil().data) if np.all(np.isfinite(row))]
        )
    else:
        (row_indices,) = np.isfinite(matrix.sum(axis=1)).nonzero()
    return row_indices


class HDBSCAN(ClusterMixin, BaseEstimator):
    """Cluster data using hierarchical density-based clustering.

    HDBSCAN - Hierarchical Density-Based Spatial Clustering of Applications
    with Noise. Performs :class:`~sklearn.cluster.DBSCAN` over varying epsilon
    values and integrates the result to find a clustering that gives the best
    stability over epsilon.
    This allows HDBSCAN to find clusters of varying densities (unlike
    :class:`~sklearn.cluster.DBSCAN`), and be more robust to parameter selection.
    Read more in the :ref:`User Guide <hdbscan>`.

    For an example of how to use HDBSCAN, as well as a comparison to
    :class:`~sklearn.cluster.DBSCAN`, please see the :ref:`plotting demo
    <sphx_glr_auto_examples_cluster_plot_hdbscan.py>`.

    .. versionadded:: 1.3

    Parameters
    ----------
    min_cluster_size : int, default=5
        The minimum number of samples in a group for that group to be
        considered a cluster; groupings smaller than this size will be left
        as noise.

    min_samples : int, default=None
        The number of samples in a neighborhood for a point
        to be considered as a core point. This includes the point itself.
        When `None`, defaults to `min_cluster_size`.

    cluster_selection_epsilon : float, default=0.0
        A distance threshold. Clusters below this value will be merged.
        See [5]_ for more information.

    max_cluster_size : int, default=None
        A limit to the size of clusters returned by the `"eom"` cluster
        selection algorithm. There is no limit when `max_cluster_size=None`.
        Has no effect if `cluster_selection_method="leaf"`.

    metric : str or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array.

        - If metric is a string or callable, it must be one of
          the options allowed by :func:`~sklearn.metrics.pairwise_distances`
          for its metric parameter.

        - If metric is "precomputed", X is assumed to be a distance matrix and
          must be square.

    metric_params : dict, default=None
        Arguments passed to the distance metric.

    alpha : float, default=1.0
        A distance scaling parameter as used in robust single linkage.
        See [3]_ for more information.

    algorithm : {"auto", "brute", "kd_tree", "ball_tree"}, default="auto"
        Exactly which algorithm to use for computing core distances; By default
        this is set to `"auto"` which attempts to use a
        :class:`~sklearn.neighbors.KDTree` tree if possible, otherwise it uses
        a :class:`~sklearn.neighbors.BallTree` tree. Both `"kd_tree"` and
        `"ball_tree"` algorithms use the
        :class:`~sklearn.neighbors.NearestNeighbors` estimator.

        If the `X` passed during `fit` is sparse or `metric` is invalid for
        both :class:`~sklearn.neighbors.KDTree` and
        :class:`~sklearn.neighbors.BallTree`, then it resolves to use the
        `"brute"` algorithm.

        .. deprecated:: 1.4
           The `'kdtree'` option was deprecated in version 1.4,
           and will be renamed to `'kd_tree'` in 1.6.

        .. deprecated:: 1.4
           The `'balltree'` option was deprecated in version 1.4,
           and will be renamed to `'ball_tree'` in 1.6.

    leaf_size : int, default=40
        Leaf size for trees responsible for fast nearest neighbour queries when
        a KDTree or a BallTree are used as core-distance algorithms. A large
        dataset size and small `leaf_size` may induce excessive memory usage.
        If you are running out of memory consider increasing the `leaf_size`
        parameter. Ignored for `algorithm="brute"`.

    n_jobs : int, default=None
        Number of jobs to run in parallel to calculate distances.
        `None` means 1 unless in a :obj:`joblib.parallel_backend` context.
        `-1` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    cluster_selection_method : {"eom", "leaf"}, default="eom"
        The method used to select clusters from the condensed tree. The
        standard approach for HDBSCAN* is to use an Excess of Mass (`"eom"`)
        algorithm to find the most persistent clusters. Alternatively you can
        instead select the clusters at the leaves of the tree -- this provides
        the most fine grained and homogeneous clusters.

    allow_single_cluster : bool, default=False
        By default HDBSCAN* will not produce a single cluster, setting this
        to True will override this and allow single cluster results in
        the case that you feel this is a valid result for your dataset.

    store_centers : str, default=None
        Which, if any, cluster centers to compute and store. The options are:

        - `None` which does not compute nor store any centers.
        - `"centroid"` which calculates the center by taking the weighted
          average of their positions. Note that the algorithm uses the
          euclidean metric and does not guarantee that the output will be
          an observed data point.
        - `"medoid"` which calculates the center by taking the point in the
          fitted data which minimizes the distance to all other points in
          the cluster. This is slower than "centroid" since it requires
          computing additional pairwise distances between points of the
          same cluster but guarantees the output is an observed data point.
          The medoid is also well-defined for arbitrary metrics, and does not
          depend on a euclidean metric.
        - `"both"` which computes and stores both forms of centers.

    copy : bool, default=False
        If `copy=True` then any time an in-place modifications would be made
        that would overwrite data passed to :term:`fit`, a copy will first be
        made, guaranteeing that the original data will be unchanged.
        Currently, it only applies when `metric="precomputed"`, when passing
        a dense array or a CSR sparse matrix and when `algorithm="brute"`.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each point in the dataset given to :term:`fit`.
        Outliers are labeled as follows:

        - Noisy samples are given the label -1.
        - Samples with infinite elements (+/- np.inf) are given the label -2.
        - Samples with missing data are given the label -3, even if they
          also have infinite elements.

    probabilities_ : ndarray of shape (n_samples,)
        The strength with which each sample is a member of its assigned
        cluster.

        - Clustered samples have probabilities proportional to the degree that
          they persist as part of the cluster.
        - Noisy samples have probability zero.
        - Samples with infinite elements (+/- np.inf) have probability 0.
        - Samples with missing data have probability `np.nan`.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    centroids_ : ndarray of shape (n_clusters, n_features)
        A collection containing the centroid of each cluster calculated under
        the standard euclidean metric. The centroids may fall "outside" their
        respective clusters if the clusters themselves are non-convex.

        Note that `n_clusters` only counts non-outlier clusters. That is to
        say, the `-1, -2, -3` labels for the outlier clusters are excluded.

    medoids_ : ndarray of shape (n_clusters, n_features)
        A collection containing the medoid of each cluster calculated under
        the whichever metric was passed to the `metric` parameter. The
        medoids are points in the original cluster which minimize the average
        distance to all other points in that cluster under the chosen metric.
        These can be thought of as the result of projecting the `metric`-based
        centroid back onto the cluster.

        Note that `n_clusters` only counts non-outlier clusters. That is to
        say, the `-1, -2, -3` labels for the outlier clusters are excluded.

    See Also
    --------
    DBSCAN : Density-Based Spatial Clustering of Applications
        with Noise.
    OPTICS : Ordering Points To Identify the Clustering Structure.
    Birch : Memory-efficient, online-learning algorithm.

    References
    ----------

    .. [1] :doi:`Campello, R. J., Moulavi, D., & Sander, J. Density-based clustering
      based on hierarchical density estimates.
      <10.1007/978-3-642-37456-2_14>`
    .. [2] :doi:`Campello, R. J., Moulavi, D., Zimek, A., & Sander, J.
       Hierarchical density estimates for data clustering, visualization,
       and outlier detection.<10.1145/2733381>`

    .. [3] `Chaudhuri, K., & Dasgupta, S. Rates of convergence for the
       cluster tree.
       <https://papers.nips.cc/paper/2010/hash/
       b534ba68236ba543ae44b22bd110a1d6-Abstract.html>`_

    .. [4] `Moulavi, D., Jaskowiak, P.A., Campello, R.J., Zimek, A. and
       Sander, J. Density-Based Clustering Validation.
       <https://www.dbs.ifi.lmu.de/~zimek/publications/SDM2014/DBCV.pdf>`_

    .. [5] :arxiv:`Malzer, C., & Baum, M. "A Hybrid Approach To Hierarchical
       Density-based Cluster Selection."<1911.02282>`.

    Examples
    --------
    >>> from sklearn.cluster import HDBSCAN
    >>> from sklearn.datasets import load_digits
    >>> X, _ = load_digits(return_X_y=True)
    >>> hdb = HDBSCAN(min_cluster_size=20)
    >>> hdb.fit(X)
    HDBSCAN(min_cluster_size=20)
    >>> hdb.labels_
    array([ 2,  6, -1, ..., -1, -1, -1])
    """

    _parameter_constraints = {
        "min_cluster_size": [Interval(Integral, left=2, right=None, closed="left")],
        "min_samples": [Interval(Integral, left=1, right=None, closed="left"), None],
        "cluster_selection_epsilon": [
            Interval(Real, left=0, right=None, closed="left")
        ],
        "max_cluster_size": [
            None,
            Interval(Integral, left=1, right=None, closed="left"),
        ],
        "metric": [StrOptions(FAST_METRICS | {"precomputed"}), callable],
        "metric_params": [dict, None],
        "alpha": [Interval(Real, left=0, right=None, closed="neither")],
        # TODO(1.6): Remove "kdtree" and "balltree"  option
        "algorithm": [
            StrOptions(
                {"auto", "brute", "kd_tree", "ball_tree", "kdtree", "balltree"},
                deprecated={"kdtree", "balltree"},
            ),
        ],
        "leaf_size": [Interval(Integral, left=1, right=None, closed="left")],
        "n_jobs": [Integral, None],
        "cluster_selection_method": [StrOptions({"eom", "leaf"})],
        "allow_single_cluster": ["boolean"],
        "store_centers": [None, StrOptions({"centroid", "medoid", "both"})],
        "copy": ["boolean"],
    }

    def __init__(
        self,
        min_cluster_size=5,
        min_samples=None,
        cluster_selection_epsilon=0.0,
        max_cluster_size=None,
        metric="euclidean",
        metric_params=None,
        alpha=1.0,
        algorithm="auto",
        leaf_size=40,
        n_jobs=None,
        cluster_selection_method="eom",
        allow_single_cluster=False,
        store_centers=None,
        copy=False,
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.alpha = alpha
        self.max_cluster_size = max_cluster_size
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.metric = metric
        self.metric_params = metric_params
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.n_jobs = n_jobs
        self.cluster_selection_method = cluster_selection_method
        self.allow_single_cluster = allow_single_cluster
        self.store_centers = store_centers
        self.copy = copy

    @_fit_context(
        # HDBSCAN.metric is not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X, y=None):
        """Find clusters based on hierarchical density-based clustering.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features), or \
                ndarray of shape (n_samples, n_samples)
            A feature array, or array of distances between samples if
            `metric='precomputed'`.

        y : None
            Ignored.

        Returns
        -------
        self : object
            Returns self.
        """
        if self.metric == "precomputed" and self.store_centers is not None:
            raise ValueError(
                "Cannot store centers when using a precomputed distance matrix."
            )

        self._metric_params = self.metric_params or {}
        if self.metric != "precomputed":
            # Non-precomputed matrices may contain non-finite values.
            X = self._validate_data(
                X,
                accept_sparse=["csr", "lil"],
                force_all_finite=False,
                dtype=np.float64,
            )
            self._raw_data = X
            all_finite = True
            try:
                _assert_all_finite(X.data if issparse(X) else X)
            except ValueError:
                all_finite = False

            if not all_finite:
                # Pass only the purely finite indices into hdbscan
                # We will later assign all non-finite points their
                # corresponding labels, as specified in `_OUTLIER_ENCODING`

                # Reduce X to make the checks for missing/outlier samples more
                # convenient.
                reduced_X = X.sum(axis=1)

                # Samples with missing data are denoted by the presence of
                # `np.nan`
                missing_index = np.isnan(reduced_X).nonzero()[0]

                # Outlier samples are denoted by the presence of `np.inf`
                infinite_index = np.isinf(reduced_X).nonzero()[0]

                # Continue with only finite samples
                finite_index = _get_finite_row_indices(X)
                internal_to_raw = {x: y for x, y in enumerate(finite_index)}
                X = X[finite_index]
        elif issparse(X):
            # Handle sparse precomputed distance matrices separately
            X = self._validate_data(
                X,
                accept_sparse=["csr", "lil"],
                dtype=np.float64,
            )
        else:
            # Only non-sparse, precomputed distance matrices are handled here
            # and thereby allowed to contain numpy.inf for missing distances

            # Perform data validation after removing infinite values (numpy.inf)
            # from the given distance matrix.
            X = self._validate_data(X, force_all_finite=False, dtype=np.float64)
            if np.isnan(X).any():
                # TODO: Support np.nan in Cython implementation for precomputed
                # dense HDBSCAN
                raise ValueError("np.nan values found in precomputed-dense")
        if X.shape[0] == 1:
            raise ValueError("n_samples=1 while HDBSCAN requires more than one sample")
        self._min_samples = (
            self.min_cluster_size if self.min_samples is None else self.min_samples
        )

        if self._min_samples > X.shape[0]:
            raise ValueError(
                f"min_samples ({self._min_samples}) must be at most the number of"
                f" samples in X ({X.shape[0]})"
            )

        # TODO(1.6): Remove
        if self.algorithm == "kdtree":
            warn(
                (
                    "`algorithm='kdtree'`has been deprecated in 1.4 and will be renamed"
                    " to'kd_tree'`in 1.6. To keep the past behaviour, set"
                    " `algorithm='kd_tree'`."
                ),
                FutureWarning,
            )
            self.algorithm = "kd_tree"

        # TODO(1.6): Remove
        if self.algorithm == "balltree":
            warn(
                (
                    "`algorithm='balltree'`has been deprecated in 1.4 and will be"
                    " renamed to'ball_tree'`in 1.6. To keep the past behaviour, set"
                    " `algorithm='ball_tree'`."
                ),
                FutureWarning,
            )
            self.algorithm = "ball_tree"

        mst_func = None
        kwargs = dict(
            X=X,
            min_samples=self._min_samples,
            alpha=self.alpha,
            metric=self.metric,
            n_jobs=self.n_jobs,
            **self._metric_params,
        )
        if self.algorithm == "kd_tree" and self.metric not in KDTree.valid_metrics:
            raise ValueError(
                f"{self.metric} is not a valid metric for a KDTree-based algorithm."
                " Please select a different metric."
            )
        elif (
            self.algorithm == "ball_tree" and self.metric not in BallTree.valid_metrics
        ):
            raise ValueError(
                f"{self.metric} is not a valid metric for a BallTree-based algorithm."
                " Please select a different metric."
            )

        if self.algorithm != "auto":
            if (
                self.metric != "precomputed"
                and issparse(X)
                and self.algorithm != "brute"
            ):
                raise ValueError("Sparse data matrices only support algorithm `brute`.")

            if self.algorithm == "brute":
                mst_func = _hdbscan_brute
                kwargs["copy"] = self.copy
            elif self.algorithm == "kd_tree":
                mst_func = _hdbscan_prims
                kwargs["algo"] = "kd_tree"
                kwargs["leaf_size"] = self.leaf_size
            else:
                mst_func = _hdbscan_prims
                kwargs["algo"] = "ball_tree"
                kwargs["leaf_size"] = self.leaf_size
        else:
            if issparse(X) or self.metric not in FAST_METRICS:
                # We can't do much with sparse matrices ...
                mst_func = _hdbscan_brute
                kwargs["copy"] = self.copy
            elif self.metric in KDTree.valid_metrics:
                # TODO: Benchmark KD vs Ball Tree efficiency
                mst_func = _hdbscan_prims
                kwargs["algo"] = "kd_tree"
                kwargs["leaf_size"] = self.leaf_size
            else:
                # Metric is a valid BallTree metric
                mst_func = _hdbscan_prims
                kwargs["algo"] = "ball_tree"
                kwargs["leaf_size"] = self.leaf_size

        self._single_linkage_tree_ = mst_func(**kwargs)

        self.labels_, self.probabilities_ = tree_to_labels(
            self._single_linkage_tree_,
            self.min_cluster_size,
            self.cluster_selection_method,
            self.allow_single_cluster,
            self.cluster_selection_epsilon,
            self.max_cluster_size,
        )
        if self.metric != "precomputed" and not all_finite:
            # Remap indices to align with original data in the case of
            # non-finite entries. Samples with np.inf are mapped to -1 and
            # those with np.nan are mapped to -2.
            self._single_linkage_tree_ = remap_single_linkage_tree(
                self._single_linkage_tree_,
                internal_to_raw,
                # There may be overlap for points w/ both `np.inf` and `np.nan`
                non_finite=set(np.hstack([infinite_index, missing_index])),
            )
            new_labels = np.empty(self._raw_data.shape[0], dtype=np.int32)
            new_labels[finite_index] = self.labels_
            new_labels[infinite_index] = _OUTLIER_ENCODING["infinite"]["label"]
            new_labels[missing_index] = _OUTLIER_ENCODING["missing"]["label"]
            self.labels_ = new_labels

            new_probabilities = np.zeros(self._raw_data.shape[0], dtype=np.float64)
            new_probabilities[finite_index] = self.probabilities_
            # Infinite outliers have probability 0 by convention, though this
            # is arbitrary.
            new_probabilities[infinite_index] = _OUTLIER_ENCODING["infinite"]["prob"]
            new_probabilities[missing_index] = _OUTLIER_ENCODING["missing"]["prob"]
            self.probabilities_ = new_probabilities

        if self.store_centers:
            self._weighted_cluster_center(X)
        return self

    def fit_predict(self, X, y=None):
        """Cluster X and return the associated cluster labels.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features), or \
                ndarray of shape (n_samples, n_samples)
            A feature array, or array of distances between samples if
            `metric='precomputed'`.

        y : None
            Ignored.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            Cluster labels.
        """
        self.fit(X)
        return self.labels_

    def _weighted_cluster_center(self, X):
        """Calculate and store the centroids/medoids of each cluster.

        This requires `X` to be a raw feature array, not precomputed
        distances. Rather than return outputs directly, this helper method
        instead stores them in the `self.{centroids, medoids}_` attributes.
        The choice for which attributes are calculated and stored is mediated
        by the value of `self.store_centers`.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The feature array that the estimator was fit with.

        """
        # Number of non-noise clusters
        n_clusters = len(set(self.labels_) - {-1, -2})
        mask = np.empty((X.shape[0],), dtype=np.bool_)
        make_centroids = self.store_centers in ("centroid", "both")
        make_medoids = self.store_centers in ("medoid", "both")

        if make_centroids:
            self.centroids_ = np.empty((n_clusters, X.shape[1]), dtype=np.float64)
        if make_medoids:
            self.medoids_ = np.empty((n_clusters, X.shape[1]), dtype=np.float64)

        # Need to handle iteratively seen each cluster may have a different
        # number of samples, hence we can't create a homogeneous 3D array.
        for idx in range(n_clusters):
            mask = self.labels_ == idx
            data = X[mask]
            strength = self.probabilities_[mask]
            if make_centroids:
                self.centroids_[idx] = np.average(data, weights=strength, axis=0)
            if make_medoids:
                # TODO: Implement weighted argmin PWD backend
                dist_mat = pairwise_distances(
                    data, metric=self.metric, **self._metric_params
                )
                dist_mat = dist_mat * strength
                medoid_index = np.argmin(dist_mat.sum(axis=1))
                self.medoids_[idx] = data[medoid_index]
        return

    def dbscan_clustering(self, cut_distance, min_cluster_size=5):
        """Return clustering given by DBSCAN without border points.

        Return clustering that would be equivalent to running DBSCAN* for a
        particular cut_distance (or epsilon) DBSCAN* can be thought of as
        DBSCAN without the border points.  As such these results may differ
        slightly from `cluster.DBSCAN` due to the difference in implementation
        over the non-core points.

        This can also be thought of as a flat clustering derived from constant
        height cut through the single linkage tree.

        This represents the result of selecting a cut value for robust single linkage
        clustering. The `min_cluster_size` allows the flat clustering to declare noise
        points (and cluster smaller than `min_cluster_size`).

        Parameters
        ----------
        cut_distance : float
            The mutual reachability distance cut value to use to generate a
            flat clustering.

        min_cluster_size : int, default=5
            Clusters smaller than this value with be called 'noise' and remain
            unclustered in the resulting flat clustering.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            An array of cluster labels, one per datapoint.
            Outliers are labeled as follows:

            - Noisy samples are given the label -1.
            - Samples with infinite elements (+/- np.inf) are given the label -2.
            - Samples with missing data are given the label -3, even if they
              also have infinite elements.
        """
        labels = labelling_at_cut(
            self._single_linkage_tree_, cut_distance, min_cluster_size
        )
        # Infer indices from labels generated during `fit`
        infinite_index = self.labels_ == _OUTLIER_ENCODING["infinite"]["label"]
        missing_index = self.labels_ == _OUTLIER_ENCODING["missing"]["label"]

        # Overwrite infinite/missing outlier samples (otherwise simple noise)
        labels[infinite_index] = _OUTLIER_ENCODING["infinite"]["label"]
        labels[missing_index] = _OUTLIER_ENCODING["missing"]["label"]
        return labels

    def _more_tags(self):
        return {"allow_nan": self.metric != "precomputed"}
