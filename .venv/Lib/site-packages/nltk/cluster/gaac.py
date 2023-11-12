# Natural Language Toolkit: Group Average Agglomerative Clusterer
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Trevor Cohn <tacohn@cs.mu.oz.au>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

try:
    import numpy
except ImportError:
    pass

from nltk.cluster.util import Dendrogram, VectorSpaceClusterer, cosine_distance


class GAAClusterer(VectorSpaceClusterer):
    """
    The Group Average Agglomerative starts with each of the N vectors as singleton
    clusters. It then iteratively merges pairs of clusters which have the
    closest centroids.  This continues until there is only one cluster. The
    order of merges gives rise to a dendrogram: a tree with the earlier merges
    lower than later merges. The membership of a given number of clusters c, 1
    <= c <= N, can be found by cutting the dendrogram at depth c.

    This clusterer uses the cosine similarity metric only, which allows for
    efficient speed-up in the clustering process.
    """

    def __init__(self, num_clusters=1, normalise=True, svd_dimensions=None):
        VectorSpaceClusterer.__init__(self, normalise, svd_dimensions)
        self._num_clusters = num_clusters
        self._dendrogram = None
        self._groups_values = None

    def cluster(self, vectors, assign_clusters=False, trace=False):
        # stores the merge order
        self._dendrogram = Dendrogram(
            [numpy.array(vector, numpy.float64) for vector in vectors]
        )
        return VectorSpaceClusterer.cluster(self, vectors, assign_clusters, trace)

    def cluster_vectorspace(self, vectors, trace=False):
        # variables describing the initial situation
        N = len(vectors)
        cluster_len = [1] * N
        cluster_count = N
        index_map = numpy.arange(N)

        # construct the similarity matrix
        dims = (N, N)
        dist = numpy.ones(dims, dtype=float) * numpy.inf
        for i in range(N):
            for j in range(i + 1, N):
                dist[i, j] = cosine_distance(vectors[i], vectors[j])

        while cluster_count > max(self._num_clusters, 1):
            i, j = numpy.unravel_index(dist.argmin(), dims)
            if trace:
                print("merging %d and %d" % (i, j))

            # update similarities for merging i and j
            self._merge_similarities(dist, cluster_len, i, j)

            # remove j
            dist[:, j] = numpy.inf
            dist[j, :] = numpy.inf

            # merge the clusters
            cluster_len[i] = cluster_len[i] + cluster_len[j]
            self._dendrogram.merge(index_map[i], index_map[j])
            cluster_count -= 1

            # update the index map to reflect the indexes if we
            # had removed j
            index_map[j + 1 :] -= 1
            index_map[j] = N

        self.update_clusters(self._num_clusters)

    def _merge_similarities(self, dist, cluster_len, i, j):
        # the new cluster i merged from i and j adopts the average of
        # i and j's similarity to each other cluster, weighted by the
        # number of points in the clusters i and j
        i_weight = cluster_len[i]
        j_weight = cluster_len[j]
        weight_sum = i_weight + j_weight

        # update for x<i
        dist[:i, i] = dist[:i, i] * i_weight + dist[:i, j] * j_weight
        dist[:i, i] /= weight_sum
        # update for i<x<j
        dist[i, i + 1 : j] = (
            dist[i, i + 1 : j] * i_weight + dist[i + 1 : j, j] * j_weight
        )
        # update for i<j<x
        dist[i, j + 1 :] = dist[i, j + 1 :] * i_weight + dist[j, j + 1 :] * j_weight
        dist[i, i + 1 :] /= weight_sum

    def update_clusters(self, num_clusters):
        clusters = self._dendrogram.groups(num_clusters)
        self._centroids = []
        for cluster in clusters:
            assert len(cluster) > 0
            if self._should_normalise:
                centroid = self._normalise(cluster[0])
            else:
                centroid = numpy.array(cluster[0])
            for vector in cluster[1:]:
                if self._should_normalise:
                    centroid += self._normalise(vector)
                else:
                    centroid += vector
            centroid /= len(cluster)
            self._centroids.append(centroid)
        self._num_clusters = len(self._centroids)

    def classify_vectorspace(self, vector):
        best = None
        for i in range(self._num_clusters):
            centroid = self._centroids[i]
            dist = cosine_distance(vector, centroid)
            if not best or dist < best[0]:
                best = (dist, i)
        return best[1]

    def dendrogram(self):
        """
        :return: The dendrogram representing the current clustering
        :rtype:  Dendrogram
        """
        return self._dendrogram

    def num_clusters(self):
        return self._num_clusters

    def __repr__(self):
        return "<GroupAverageAgglomerative Clusterer n=%d>" % self._num_clusters


def demo():
    """
    Non-interactive demonstration of the clusterers with simple 2-D data.
    """

    from nltk.cluster import GAAClusterer

    # use a set of tokens with 2D indices
    vectors = [numpy.array(f) for f in [[3, 3], [1, 2], [4, 2], [4, 0], [2, 3], [3, 1]]]

    # test the GAAC clusterer with 4 clusters
    clusterer = GAAClusterer(4)
    clusters = clusterer.cluster(vectors, True)

    print("Clusterer:", clusterer)
    print("Clustered:", vectors)
    print("As:", clusters)
    print()

    # show the dendrogram
    clusterer.dendrogram().show()

    # classify a new vector
    vector = numpy.array([3, 3])
    print("classify(%s):" % vector, end=" ")
    print(clusterer.classify(vector))
    print()


if __name__ == "__main__":
    demo()
