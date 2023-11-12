# Natural Language Toolkit: Expectation Maximization Clusterer
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Trevor Cohn <tacohn@cs.mu.oz.au>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

try:
    import numpy
except ImportError:
    pass

from nltk.cluster.util import VectorSpaceClusterer


class EMClusterer(VectorSpaceClusterer):
    """
    The Gaussian EM clusterer models the vectors as being produced by
    a mixture of k Gaussian sources. The parameters of these sources
    (prior probability, mean and covariance matrix) are then found to
    maximise the likelihood of the given data. This is done with the
    expectation maximisation algorithm. It starts with k arbitrarily
    chosen means, priors and covariance matrices. It then calculates
    the membership probabilities for each vector in each of the
    clusters; this is the 'E' step. The cluster parameters are then
    updated in the 'M' step using the maximum likelihood estimate from
    the cluster membership probabilities. This process continues until
    the likelihood of the data does not significantly increase.
    """

    def __init__(
        self,
        initial_means,
        priors=None,
        covariance_matrices=None,
        conv_threshold=1e-6,
        bias=0.1,
        normalise=False,
        svd_dimensions=None,
    ):
        """
        Creates an EM clusterer with the given starting parameters,
        convergence threshold and vector mangling parameters.

        :param  initial_means: the means of the gaussian cluster centers
        :type   initial_means: [seq of] numpy array or seq of SparseArray
        :param  priors: the prior probability for each cluster
        :type   priors: numpy array or seq of float
        :param  covariance_matrices: the covariance matrix for each cluster
        :type   covariance_matrices: [seq of] numpy array
        :param  conv_threshold: maximum change in likelihood before deemed
                    convergent
        :type   conv_threshold: int or float
        :param  bias: variance bias used to ensure non-singular covariance
                      matrices
        :type   bias: float
        :param  normalise:  should vectors be normalised to length 1
        :type   normalise:  boolean
        :param  svd_dimensions: number of dimensions to use in reducing vector
                               dimensionsionality with SVD
        :type   svd_dimensions: int
        """
        VectorSpaceClusterer.__init__(self, normalise, svd_dimensions)
        self._means = numpy.array(initial_means, numpy.float64)
        self._num_clusters = len(initial_means)
        self._conv_threshold = conv_threshold
        self._covariance_matrices = covariance_matrices
        self._priors = priors
        self._bias = bias

    def num_clusters(self):
        return self._num_clusters

    def cluster_vectorspace(self, vectors, trace=False):
        assert len(vectors) > 0

        # set the parameters to initial values
        dimensions = len(vectors[0])
        means = self._means
        priors = self._priors
        if not priors:
            priors = self._priors = (
                numpy.ones(self._num_clusters, numpy.float64) / self._num_clusters
            )
        covariances = self._covariance_matrices
        if not covariances:
            covariances = self._covariance_matrices = [
                numpy.identity(dimensions, numpy.float64)
                for i in range(self._num_clusters)
            ]

        # do the E and M steps until the likelihood plateaus
        lastl = self._loglikelihood(vectors, priors, means, covariances)
        converged = False

        while not converged:
            if trace:
                print("iteration; loglikelihood", lastl)
            # E-step, calculate hidden variables, h[i,j]
            h = numpy.zeros((len(vectors), self._num_clusters), numpy.float64)
            for i in range(len(vectors)):
                for j in range(self._num_clusters):
                    h[i, j] = priors[j] * self._gaussian(
                        means[j], covariances[j], vectors[i]
                    )
                h[i, :] /= sum(h[i, :])

            # M-step, update parameters - cvm, p, mean
            for j in range(self._num_clusters):
                covariance_before = covariances[j]
                new_covariance = numpy.zeros((dimensions, dimensions), numpy.float64)
                new_mean = numpy.zeros(dimensions, numpy.float64)
                sum_hj = 0.0
                for i in range(len(vectors)):
                    delta = vectors[i] - means[j]
                    new_covariance += h[i, j] * numpy.multiply.outer(delta, delta)
                    sum_hj += h[i, j]
                    new_mean += h[i, j] * vectors[i]
                covariances[j] = new_covariance / sum_hj
                means[j] = new_mean / sum_hj
                priors[j] = sum_hj / len(vectors)

                # bias term to stop covariance matrix being singular
                covariances[j] += self._bias * numpy.identity(dimensions, numpy.float64)

            # calculate likelihood - FIXME: may be broken
            l = self._loglikelihood(vectors, priors, means, covariances)

            # check for convergence
            if abs(lastl - l) < self._conv_threshold:
                converged = True
            lastl = l

    def classify_vectorspace(self, vector):
        best = None
        for j in range(self._num_clusters):
            p = self._priors[j] * self._gaussian(
                self._means[j], self._covariance_matrices[j], vector
            )
            if not best or p > best[0]:
                best = (p, j)
        return best[1]

    def likelihood_vectorspace(self, vector, cluster):
        cid = self.cluster_names().index(cluster)
        return self._priors[cluster] * self._gaussian(
            self._means[cluster], self._covariance_matrices[cluster], vector
        )

    def _gaussian(self, mean, cvm, x):
        m = len(mean)
        assert cvm.shape == (m, m), "bad sized covariance matrix, %s" % str(cvm.shape)
        try:
            det = numpy.linalg.det(cvm)
            inv = numpy.linalg.inv(cvm)
            a = det**-0.5 * (2 * numpy.pi) ** (-m / 2.0)
            dx = x - mean
            print(dx, inv)
            b = -0.5 * numpy.dot(numpy.dot(dx, inv), dx)
            return a * numpy.exp(b)
        except OverflowError:
            # happens when the exponent is negative infinity - i.e. b = 0
            # i.e. the inverse of cvm is huge (cvm is almost zero)
            return 0

    def _loglikelihood(self, vectors, priors, means, covariances):
        llh = 0.0
        for vector in vectors:
            p = 0
            for j in range(len(priors)):
                p += priors[j] * self._gaussian(means[j], covariances[j], vector)
            llh += numpy.log(p)
        return llh

    def __repr__(self):
        return "<EMClusterer means=%s>" % list(self._means)


def demo():
    """
    Non-interactive demonstration of the clusterers with simple 2-D data.
    """

    from nltk import cluster

    # example from figure 14.10, page 519, Manning and Schutze

    vectors = [numpy.array(f) for f in [[0.5, 0.5], [1.5, 0.5], [1, 3]]]
    means = [[4, 2], [4, 2.01]]

    clusterer = cluster.EMClusterer(means, bias=0.1)
    clusters = clusterer.cluster(vectors, True, trace=True)

    print("Clustered:", vectors)
    print("As:       ", clusters)
    print()

    for c in range(2):
        print("Cluster:", c)
        print("Prior:  ", clusterer._priors[c])
        print("Mean:   ", clusterer._means[c])
        print("Covar:  ", clusterer._covariance_matrices[c])
        print()

    # classify a new vector
    vector = numpy.array([2, 2])
    print("classify(%s):" % vector, end=" ")
    print(clusterer.classify(vector))

    # show the classification probabilities
    vector = numpy.array([2, 2])
    print("classification_probdist(%s):" % vector)
    pdist = clusterer.classification_probdist(vector)
    for sample in pdist.samples():
        print(f"{sample} => {pdist.prob(sample) * 100:.0f}%")


if __name__ == "__main__":
    demo()
