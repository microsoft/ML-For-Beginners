# Natural Language Toolkit: Clusterers
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Trevor Cohn <tacohn@cs.mu.oz.au>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
This module contains a number of basic clustering algorithms. Clustering
describes the task of discovering groups of similar items with a large
collection. It is also describe as unsupervised machine learning, as the data
from which it learns is unannotated with class information, as is the case for
supervised learning.  Annotated data is difficult and expensive to obtain in
the quantities required for the majority of supervised learning algorithms.
This problem, the knowledge acquisition bottleneck, is common to most natural
language processing tasks, thus fueling the need for quality unsupervised
approaches.

This module contains a k-means clusterer, E-M clusterer and a group average
agglomerative clusterer (GAAC). All these clusterers involve finding good
cluster groupings for a set of vectors in multi-dimensional space.

The K-means clusterer starts with k arbitrary chosen means then allocates each
vector to the cluster with the closest mean. It then recalculates the means of
each cluster as the centroid of the vectors in the cluster. This process
repeats until the cluster memberships stabilise. This is a hill-climbing
algorithm which may converge to a local maximum. Hence the clustering is
often repeated with random initial means and the most commonly occurring
output means are chosen.

The GAAC clusterer starts with each of the *N* vectors as singleton clusters.
It then iteratively merges pairs of clusters which have the closest centroids.
This continues until there is only one cluster. The order of merges gives rise
to a dendrogram - a tree with the earlier merges lower than later merges. The
membership of a given number of clusters *c*, *1 <= c <= N*, can be found by
cutting the dendrogram at depth *c*.

The Gaussian EM clusterer models the vectors as being produced by a mixture
of k Gaussian sources. The parameters of these sources (prior probability,
mean and covariance matrix) are then found to maximise the likelihood of the
given data. This is done with the expectation maximisation algorithm. It
starts with k arbitrarily chosen means, priors and covariance matrices. It
then calculates the membership probabilities for each vector in each of the
clusters - this is the 'E' step. The cluster parameters are then updated in
the 'M' step using the maximum likelihood estimate from the cluster membership
probabilities. This process continues until the likelihood of the data does
not significantly increase.

They all extend the ClusterI interface which defines common operations
available with each clusterer. These operations include:

- cluster: clusters a sequence of vectors
- classify: assign a vector to a cluster
- classification_probdist: give the probability distribution over cluster memberships

The current existing classifiers also extend cluster.VectorSpace, an
abstract class which allows for singular value decomposition (SVD) and vector
normalisation. SVD is used to reduce the dimensionality of the vector space in
such a manner as to preserve as much of the variation as possible, by
reparameterising the axes in order of variability and discarding all bar the
first d dimensions. Normalisation ensures that vectors fall in the unit
hypersphere.

Usage example (see also demo())::

    from nltk import cluster
    from nltk.cluster import euclidean_distance
    from numpy import array

    vectors = [array(f) for f in [[3, 3], [1, 2], [4, 2], [4, 0]]]

    # initialise the clusterer (will also assign the vectors to clusters)
    clusterer = cluster.KMeansClusterer(2, euclidean_distance)
    clusterer.cluster(vectors, True)

    # classify a new vector
    print(clusterer.classify(array([3, 3])))

Note that the vectors must use numpy array-like
objects. nltk_contrib.unimelb.tacohn.SparseArrays may be used for
efficiency when required.
"""

from nltk.cluster.em import EMClusterer
from nltk.cluster.gaac import GAAClusterer
from nltk.cluster.kmeans import KMeansClusterer
from nltk.cluster.util import (
    Dendrogram,
    VectorSpaceClusterer,
    cosine_distance,
    euclidean_distance,
)
