# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Joel Nothman <joel.nothman@gmail.com>
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#          Jacob Schreiber <jmschreiber91@gmail.com>
#
# License: BSD 3 clause

# See _criterion.pyx for implementation details.

from ._tree cimport DTYPE_t          # Type of X
from ._tree cimport DOUBLE_t         # Type of y, sample_weight
from ._tree cimport SIZE_t           # Type for indices and counters
from ._tree cimport INT32_t          # Signed 32 bit integer
from ._tree cimport UINT32_t         # Unsigned 32 bit integer

cdef class Criterion:
    # The criterion computes the impurity of a node and the reduction of
    # impurity of a split on that node. It also computes the output statistics
    # such as the mean in regression and class probabilities in classification.

    # Internal structures
    cdef const DOUBLE_t[:, ::1] y         # Values of y
    cdef const DOUBLE_t[:] sample_weight  # Sample weights

    cdef const SIZE_t[:] sample_indices   # Sample indices in X, y
    cdef SIZE_t start                     # samples[start:pos] are the samples in the left node
    cdef SIZE_t pos                       # samples[pos:end] are the samples in the right node
    cdef SIZE_t end
    cdef SIZE_t n_missing                # Number of missing values for the feature being evaluated
    cdef bint missing_go_to_left         # Whether missing values go to the left node

    cdef SIZE_t n_outputs                 # Number of outputs
    cdef SIZE_t n_samples                 # Number of samples
    cdef SIZE_t n_node_samples            # Number of samples in the node (end-start)
    cdef double weighted_n_samples        # Weighted number of samples (in total)
    cdef double weighted_n_node_samples   # Weighted number of samples in the node
    cdef double weighted_n_left           # Weighted number of samples in the left node
    cdef double weighted_n_right          # Weighted number of samples in the right node
    cdef double weighted_n_missing       # Weighted number of samples that are missing

    # The criterion object is maintained such that left and right collected
    # statistics correspond to samples[start:pos] and samples[pos:end].

    # Methods
    cdef int init(
        self,
        const DOUBLE_t[:, ::1] y,
        const DOUBLE_t[:] sample_weight,
        double weighted_n_samples,
        const SIZE_t[:] sample_indices,
        SIZE_t start,
        SIZE_t end
    ) except -1 nogil
    cdef void init_sum_missing(self)
    cdef void init_missing(self, SIZE_t n_missing) noexcept nogil
    cdef int reset(self) except -1 nogil
    cdef int reverse_reset(self) except -1 nogil
    cdef int update(self, SIZE_t new_pos) except -1 nogil
    cdef double node_impurity(self) noexcept nogil
    cdef void children_impurity(
        self,
        double* impurity_left,
        double* impurity_right
    ) noexcept nogil
    cdef void node_value(
        self,
        double* dest
    ) noexcept nogil
    cdef double impurity_improvement(
        self,
        double impurity_parent,
        double impurity_left,
        double impurity_right
    ) noexcept nogil
    cdef double proxy_impurity_improvement(self) noexcept nogil

cdef class ClassificationCriterion(Criterion):
    """Abstract criterion for classification."""

    cdef SIZE_t[::1] n_classes
    cdef SIZE_t max_n_classes

    cdef double[:, ::1] sum_total    # The sum of the weighted count of each label.
    cdef double[:, ::1] sum_left     # Same as above, but for the left side of the split
    cdef double[:, ::1] sum_right    # Same as above, but for the right side of the split
    cdef double[:, ::1] sum_missing  # Same as above, but for missing values in X

cdef class RegressionCriterion(Criterion):
    """Abstract regression criterion."""

    cdef double sq_sum_total

    cdef double[::1] sum_total    # The sum of w*y.
    cdef double[::1] sum_left     # Same as above, but for the left side of the split
    cdef double[::1] sum_right    # Same as above, but for the right side of the split
    cdef double[::1] sum_missing  # Same as above, but for missing values in X
