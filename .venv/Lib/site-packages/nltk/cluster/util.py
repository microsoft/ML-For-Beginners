# Natural Language Toolkit: Clusterer Utilities
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Trevor Cohn <tacohn@cs.mu.oz.au>
# Contributor: J Richard Snape
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT
import copy
from abc import abstractmethod
from math import sqrt
from sys import stdout

try:
    import numpy
except ImportError:
    pass

from nltk.cluster.api import ClusterI


class VectorSpaceClusterer(ClusterI):
    """
    Abstract clusterer which takes tokens and maps them into a vector space.
    Optionally performs singular value decomposition to reduce the
    dimensionality.
    """

    def __init__(self, normalise=False, svd_dimensions=None):
        """
        :param normalise:       should vectors be normalised to length 1
        :type normalise:        boolean
        :param svd_dimensions:  number of dimensions to use in reducing vector
                                dimensionsionality with SVD
        :type svd_dimensions:   int
        """
        self._Tt = None
        self._should_normalise = normalise
        self._svd_dimensions = svd_dimensions

    def cluster(self, vectors, assign_clusters=False, trace=False):
        assert len(vectors) > 0

        # normalise the vectors
        if self._should_normalise:
            vectors = list(map(self._normalise, vectors))

        # use SVD to reduce the dimensionality
        if self._svd_dimensions and self._svd_dimensions < len(vectors[0]):
            [u, d, vt] = numpy.linalg.svd(numpy.transpose(numpy.array(vectors)))
            S = d[: self._svd_dimensions] * numpy.identity(
                self._svd_dimensions, numpy.float64
            )
            T = u[:, : self._svd_dimensions]
            Dt = vt[: self._svd_dimensions, :]
            vectors = numpy.transpose(numpy.dot(S, Dt))
            self._Tt = numpy.transpose(T)

        # call abstract method to cluster the vectors
        self.cluster_vectorspace(vectors, trace)

        # assign the vectors to clusters
        if assign_clusters:
            return [self.classify(vector) for vector in vectors]

    @abstractmethod
    def cluster_vectorspace(self, vectors, trace):
        """
        Finds the clusters using the given set of vectors.
        """

    def classify(self, vector):
        if self._should_normalise:
            vector = self._normalise(vector)
        if self._Tt is not None:
            vector = numpy.dot(self._Tt, vector)
        cluster = self.classify_vectorspace(vector)
        return self.cluster_name(cluster)

    @abstractmethod
    def classify_vectorspace(self, vector):
        """
        Returns the index of the appropriate cluster for the vector.
        """

    def likelihood(self, vector, label):
        if self._should_normalise:
            vector = self._normalise(vector)
        if self._Tt is not None:
            vector = numpy.dot(self._Tt, vector)
        return self.likelihood_vectorspace(vector, label)

    def likelihood_vectorspace(self, vector, cluster):
        """
        Returns the likelihood of the vector belonging to the cluster.
        """
        predicted = self.classify_vectorspace(vector)
        return 1.0 if cluster == predicted else 0.0

    def vector(self, vector):
        """
        Returns the vector after normalisation and dimensionality reduction
        """
        if self._should_normalise:
            vector = self._normalise(vector)
        if self._Tt is not None:
            vector = numpy.dot(self._Tt, vector)
        return vector

    def _normalise(self, vector):
        """
        Normalises the vector to unit length.
        """
        return vector / sqrt(numpy.dot(vector, vector))


def euclidean_distance(u, v):
    """
    Returns the euclidean distance between vectors u and v. This is equivalent
    to the length of the vector (u - v).
    """
    diff = u - v
    return sqrt(numpy.dot(diff, diff))


def cosine_distance(u, v):
    """
    Returns 1 minus the cosine of the angle between vectors v and u. This is
    equal to ``1 - (u.v / |u||v|)``.
    """
    return 1 - (numpy.dot(u, v) / (sqrt(numpy.dot(u, u)) * sqrt(numpy.dot(v, v))))


class _DendrogramNode:
    """Tree node of a dendrogram."""

    def __init__(self, value, *children):
        self._value = value
        self._children = children

    def leaves(self, values=True):
        if self._children:
            leaves = []
            for child in self._children:
                leaves.extend(child.leaves(values))
            return leaves
        elif values:
            return [self._value]
        else:
            return [self]

    def groups(self, n):
        queue = [(self._value, self)]

        while len(queue) < n:
            priority, node = queue.pop()
            if not node._children:
                queue.push((priority, node))
                break
            for child in node._children:
                if child._children:
                    queue.append((child._value, child))
                else:
                    queue.append((0, child))
            # makes the earliest merges at the start, latest at the end
            queue.sort()

        groups = []
        for priority, node in queue:
            groups.append(node.leaves())
        return groups

    def __lt__(self, comparator):
        return cosine_distance(self._value, comparator._value) < 0


class Dendrogram:
    """
    Represents a dendrogram, a tree with a specified branching order.  This
    must be initialised with the leaf items, then iteratively call merge for
    each branch. This class constructs a tree representing the order of calls
    to the merge function.
    """

    def __init__(self, items=[]):
        """
        :param  items: the items at the leaves of the dendrogram
        :type   items: sequence of (any)
        """
        self._items = [_DendrogramNode(item) for item in items]
        self._original_items = copy.copy(self._items)
        self._merge = 1

    def merge(self, *indices):
        """
        Merges nodes at given indices in the dendrogram. The nodes will be
        combined which then replaces the first node specified. All other nodes
        involved in the merge will be removed.

        :param  indices: indices of the items to merge (at least two)
        :type   indices: seq of int
        """
        assert len(indices) >= 2
        node = _DendrogramNode(self._merge, *(self._items[i] for i in indices))
        self._merge += 1
        self._items[indices[0]] = node
        for i in indices[1:]:
            del self._items[i]

    def groups(self, n):
        """
        Finds the n-groups of items (leaves) reachable from a cut at depth n.
        :param  n: number of groups
        :type   n: int
        """
        if len(self._items) > 1:
            root = _DendrogramNode(self._merge, *self._items)
        else:
            root = self._items[0]
        return root.groups(n)

    def show(self, leaf_labels=[]):
        """
        Print the dendrogram in ASCII art to standard out.

        :param leaf_labels: an optional list of strings to use for labeling the
                            leaves
        :type leaf_labels: list
        """

        # ASCII rendering characters
        JOIN, HLINK, VLINK = "+", "-", "|"

        # find the root (or create one)
        if len(self._items) > 1:
            root = _DendrogramNode(self._merge, *self._items)
        else:
            root = self._items[0]
        leaves = self._original_items

        if leaf_labels:
            last_row = leaf_labels
        else:
            last_row = ["%s" % leaf._value for leaf in leaves]

        # find the bottom row and the best cell width
        width = max(map(len, last_row)) + 1
        lhalf = width // 2
        rhalf = int(width - lhalf - 1)

        # display functions
        def format(centre, left=" ", right=" "):
            return f"{lhalf * left}{centre}{right * rhalf}"

        def display(str):
            stdout.write(str)

        # for each merge, top down
        queue = [(root._value, root)]
        verticals = [format(" ") for leaf in leaves]
        while queue:
            priority, node = queue.pop()
            child_left_leaf = list(map(lambda c: c.leaves(False)[0], node._children))
            indices = list(map(leaves.index, child_left_leaf))
            if child_left_leaf:
                min_idx = min(indices)
                max_idx = max(indices)
            for i in range(len(leaves)):
                if leaves[i] in child_left_leaf:
                    if i == min_idx:
                        display(format(JOIN, " ", HLINK))
                    elif i == max_idx:
                        display(format(JOIN, HLINK, " "))
                    else:
                        display(format(JOIN, HLINK, HLINK))
                    verticals[i] = format(VLINK)
                elif min_idx <= i <= max_idx:
                    display(format(HLINK, HLINK, HLINK))
                else:
                    display(verticals[i])
            display("\n")
            for child in node._children:
                if child._children:
                    queue.append((child._value, child))
            queue.sort()

            for vertical in verticals:
                display(vertical)
            display("\n")

        # finally, display the last line
        display("".join(item.center(width) for item in last_row))
        display("\n")

    def __repr__(self):
        if len(self._items) > 1:
            root = _DendrogramNode(self._merge, *self._items)
        else:
            root = self._items[0]
        leaves = root.leaves(False)
        return "<Dendrogram with %d leaves>" % len(leaves)
