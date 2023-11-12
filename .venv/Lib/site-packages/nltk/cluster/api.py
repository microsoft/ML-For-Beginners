# Natural Language Toolkit: Clusterer Interfaces
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Trevor Cohn <tacohn@cs.mu.oz.au>
# Porting: Steven Bird <stevenbird1@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

from abc import ABCMeta, abstractmethod

from nltk.probability import DictionaryProbDist


class ClusterI(metaclass=ABCMeta):
    """
    Interface covering basic clustering functionality.
    """

    @abstractmethod
    def cluster(self, vectors, assign_clusters=False):
        """
        Assigns the vectors to clusters, learning the clustering parameters
        from the data. Returns a cluster identifier for each vector.
        """

    @abstractmethod
    def classify(self, token):
        """
        Classifies the token into a cluster, setting the token's CLUSTER
        parameter to that cluster identifier.
        """

    def likelihood(self, vector, label):
        """
        Returns the likelihood (a float) of the token having the
        corresponding cluster.
        """
        if self.classify(vector) == label:
            return 1.0
        else:
            return 0.0

    def classification_probdist(self, vector):
        """
        Classifies the token into a cluster, returning
        a probability distribution over the cluster identifiers.
        """
        likelihoods = {}
        sum = 0.0
        for cluster in self.cluster_names():
            likelihoods[cluster] = self.likelihood(vector, cluster)
            sum += likelihoods[cluster]
        for cluster in self.cluster_names():
            likelihoods[cluster] /= sum
        return DictionaryProbDist(likelihoods)

    @abstractmethod
    def num_clusters(self):
        """
        Returns the number of clusters.
        """

    def cluster_names(self):
        """
        Returns the names of the clusters.
        :rtype: list
        """
        return list(range(self.num_clusters()))

    def cluster_name(self, index):
        """
        Returns the names of the cluster at index.
        """
        return index
