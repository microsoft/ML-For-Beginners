"""
The :mod:`sklearn.utils.graph` module includes graph utilities and algorithms.
"""

# Authors: Aric Hagberg <hagberg@lanl.gov>
#          Gael Varoquaux <gael.varoquaux@normalesup.org>
#          Jake Vanderplas <vanderplas@astro.washington.edu>
# License: BSD 3 clause

import numpy as np
from scipy import sparse

from ..metrics.pairwise import pairwise_distances
from ._param_validation import Integral, Interval, validate_params


###############################################################################
# Path and connected component analysis.
# Code adapted from networkx
@validate_params(
    {
        "graph": ["array-like", "sparse matrix"],
        "source": [Interval(Integral, 0, None, closed="left")],
        "cutoff": [Interval(Integral, 0, None, closed="left"), None],
    },
    prefer_skip_nested_validation=True,
)
def single_source_shortest_path_length(graph, source, *, cutoff=None):
    """Return the length of the shortest path from source to all reachable nodes.

    Parameters
    ----------
    graph : {array-like, sparse matrix} of shape (n_nodes, n_nodes)
        Adjacency matrix of the graph. Sparse matrix of format LIL is
        preferred.

    source : int
       Start node for path.

    cutoff : int, default=None
        Depth to stop the search - only paths of length <= cutoff are returned.

    Returns
    -------
    paths : dict
        Reachable end nodes mapped to length of path from source,
        i.e. `{end: path_length}`.

    Examples
    --------
    >>> from sklearn.utils.graph import single_source_shortest_path_length
    >>> import numpy as np
    >>> graph = np.array([[ 0, 1, 0, 0],
    ...                   [ 1, 0, 1, 0],
    ...                   [ 0, 1, 0, 0],
    ...                   [ 0, 0, 0, 0]])
    >>> single_source_shortest_path_length(graph, 0)
    {0: 0, 1: 1, 2: 2}
    >>> graph = np.ones((6, 6))
    >>> sorted(single_source_shortest_path_length(graph, 2).items())
    [(0, 1), (1, 1), (2, 0), (3, 1), (4, 1), (5, 1)]
    """
    if sparse.issparse(graph):
        graph = graph.tolil()
    else:
        graph = sparse.lil_matrix(graph)
    seen = {}  # level (number of hops) when seen in BFS
    level = 0  # the current level
    next_level = [source]  # dict of nodes to check at next level
    while next_level:
        this_level = next_level  # advance to next level
        next_level = set()  # and start a new list (fringe)
        for v in this_level:
            if v not in seen:
                seen[v] = level  # set the level of vertex v
                next_level.update(graph.rows[v])
        if cutoff is not None and cutoff <= level:
            break
        level += 1
    return seen  # return all path lengths as dictionary


def _fix_connected_components(
    X,
    graph,
    n_connected_components,
    component_labels,
    mode="distance",
    metric="euclidean",
    **kwargs,
):
    """Add connections to sparse graph to connect unconnected components.

    For each pair of unconnected components, compute all pairwise distances
    from one component to the other, and add a connection on the closest pair
    of samples. This is a hacky way to get a graph with a single connected
    component, which is necessary for example to compute a shortest path
    between all pairs of samples in the graph.

    Parameters
    ----------
    X : array of shape (n_samples, n_features) or (n_samples, n_samples)
        Features to compute the pairwise distances. If `metric =
        "precomputed"`, X is the matrix of pairwise distances.

    graph : sparse matrix of shape (n_samples, n_samples)
        Graph of connection between samples.

    n_connected_components : int
        Number of connected components, as computed by
        `scipy.sparse.csgraph.connected_components`.

    component_labels : array of shape (n_samples)
        Labels of connected components, as computed by
        `scipy.sparse.csgraph.connected_components`.

    mode : {'connectivity', 'distance'}, default='distance'
        Type of graph matrix: 'connectivity' corresponds to the connectivity
        matrix with ones and zeros, and 'distance' corresponds to the distances
        between neighbors according to the given metric.

    metric : str
        Metric used in `sklearn.metrics.pairwise.pairwise_distances`.

    kwargs : kwargs
        Keyword arguments passed to
        `sklearn.metrics.pairwise.pairwise_distances`.

    Returns
    -------
    graph : sparse matrix of shape (n_samples, n_samples)
        Graph of connection between samples, with a single connected component.
    """
    if metric == "precomputed" and sparse.issparse(X):
        raise RuntimeError(
            "_fix_connected_components with metric='precomputed' requires the "
            "full distance matrix in X, and does not work with a sparse "
            "neighbors graph."
        )

    for i in range(n_connected_components):
        idx_i = np.flatnonzero(component_labels == i)
        Xi = X[idx_i]
        for j in range(i):
            idx_j = np.flatnonzero(component_labels == j)
            Xj = X[idx_j]

            if metric == "precomputed":
                D = X[np.ix_(idx_i, idx_j)]
            else:
                D = pairwise_distances(Xi, Xj, metric=metric, **kwargs)

            ii, jj = np.unravel_index(D.argmin(axis=None), D.shape)
            if mode == "connectivity":
                graph[idx_i[ii], idx_j[jj]] = 1
                graph[idx_j[jj], idx_i[ii]] = 1
            elif mode == "distance":
                graph[idx_i[ii], idx_j[jj]] = D[ii, jj]
                graph[idx_j[jj], idx_i[ii]] = D[ii, jj]
            else:
                raise ValueError(
                    "Unknown mode=%r, should be one of ['connectivity', 'distance']."
                    % mode
                )

    return graph
