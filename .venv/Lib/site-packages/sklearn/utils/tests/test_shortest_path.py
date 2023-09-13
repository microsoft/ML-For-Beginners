from collections import defaultdict

import numpy as np
from numpy.testing import assert_array_almost_equal

from sklearn.utils.graph import single_source_shortest_path_length


def floyd_warshall_slow(graph, directed=False):
    N = graph.shape[0]

    # set nonzero entries to infinity
    graph[np.where(graph == 0)] = np.inf

    # set diagonal to zero
    graph.flat[:: N + 1] = 0

    if not directed:
        graph = np.minimum(graph, graph.T)

    for k in range(N):
        for i in range(N):
            for j in range(N):
                graph[i, j] = min(graph[i, j], graph[i, k] + graph[k, j])

    graph[np.where(np.isinf(graph))] = 0

    return graph


def generate_graph(N=20):
    # sparse grid of distances
    rng = np.random.RandomState(0)
    dist_matrix = rng.random_sample((N, N))

    # make symmetric: distances are not direction-dependent
    dist_matrix = dist_matrix + dist_matrix.T

    # make graph sparse
    i = (rng.randint(N, size=N * N // 2), rng.randint(N, size=N * N // 2))
    dist_matrix[i] = 0

    # set diagonal to zero
    dist_matrix.flat[:: N + 1] = 0

    return dist_matrix


def test_shortest_path():
    dist_matrix = generate_graph(20)
    # We compare path length and not costs (-> set distances to 0 or 1)
    dist_matrix[dist_matrix != 0] = 1

    for directed in (True, False):
        if not directed:
            dist_matrix = np.minimum(dist_matrix, dist_matrix.T)

        graph_py = floyd_warshall_slow(dist_matrix.copy(), directed)
        for i in range(dist_matrix.shape[0]):
            # Non-reachable nodes have distance 0 in graph_py
            dist_dict = defaultdict(int)
            dist_dict.update(single_source_shortest_path_length(dist_matrix, i))

            for j in range(graph_py[i].shape[0]):
                assert_array_almost_equal(dist_dict[j], graph_py[i, j])
