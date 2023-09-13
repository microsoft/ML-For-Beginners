# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#          Jacob Schreiber <jmschreiber91@gmail.com>
#          Nelson Liu <nelson@nelsonliu.me>
#
# License: BSD 3 clause

# See _utils.pyx for details.

cimport numpy as cnp
from ._tree cimport Node
from ..neighbors._quad_tree cimport Cell

ctypedef cnp.npy_float32 DTYPE_t          # Type of X
ctypedef cnp.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef cnp.npy_intp SIZE_t              # Type for indices and counters
ctypedef cnp.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef cnp.npy_uint32 UINT32_t          # Unsigned 32 bit integer


cdef enum:
    # Max value for our rand_r replacement (near the bottom).
    # We don't use RAND_MAX because it's different across platforms and
    # particularly tiny on Windows/MSVC.
    RAND_R_MAX = 0x7FFFFFFF


# safe_realloc(&p, n) resizes the allocation of p to n * sizeof(*p) bytes or
# raises a MemoryError. It never calls free, since that's __dealloc__'s job.
#   cdef DTYPE_t *p = NULL
#   safe_realloc(&p, n)
# is equivalent to p = malloc(n * sizeof(*p)) with error checking.
ctypedef fused realloc_ptr:
    # Add pointer types here as needed.
    (DTYPE_t*)
    (SIZE_t*)
    (unsigned char*)
    (WeightedPQueueRecord*)
    (DOUBLE_t*)
    (DOUBLE_t**)
    (Node*)
    (Cell*)
    (Node**)

cdef realloc_ptr safe_realloc(realloc_ptr* p, size_t nelems) except * nogil


cdef cnp.ndarray sizet_ptr_to_ndarray(SIZE_t* data, SIZE_t size)


cdef SIZE_t rand_int(SIZE_t low, SIZE_t high,
                     UINT32_t* random_state) noexcept nogil


cdef double rand_uniform(double low, double high,
                         UINT32_t* random_state) noexcept nogil


cdef double log(double x) noexcept nogil

# =============================================================================
# WeightedPQueue data structure
# =============================================================================

# A record stored in the WeightedPQueue
cdef struct WeightedPQueueRecord:
    DOUBLE_t data
    DOUBLE_t weight

cdef class WeightedPQueue:
    cdef SIZE_t capacity
    cdef SIZE_t array_ptr
    cdef WeightedPQueueRecord* array_

    cdef bint is_empty(self) noexcept nogil
    cdef int reset(self) except -1 nogil
    cdef SIZE_t size(self) noexcept nogil
    cdef int push(self, DOUBLE_t data, DOUBLE_t weight) except -1 nogil
    cdef int remove(self, DOUBLE_t data, DOUBLE_t weight) noexcept nogil
    cdef int pop(self, DOUBLE_t* data, DOUBLE_t* weight) noexcept nogil
    cdef int peek(self, DOUBLE_t* data, DOUBLE_t* weight) noexcept nogil
    cdef DOUBLE_t get_weight_from_index(self, SIZE_t index) noexcept nogil
    cdef DOUBLE_t get_value_from_index(self, SIZE_t index) noexcept nogil


# =============================================================================
# WeightedMedianCalculator data structure
# =============================================================================

cdef class WeightedMedianCalculator:
    cdef SIZE_t initial_capacity
    cdef WeightedPQueue samples
    cdef DOUBLE_t total_weight
    cdef SIZE_t k
    cdef DOUBLE_t sum_w_0_k  # represents sum(weights[0:k]) = w[0] + w[1] + ... + w[k-1]
    cdef SIZE_t size(self) noexcept nogil
    cdef int push(self, DOUBLE_t data, DOUBLE_t weight) except -1 nogil
    cdef int reset(self) except -1 nogil
    cdef int update_median_parameters_post_push(
        self, DOUBLE_t data, DOUBLE_t weight,
        DOUBLE_t original_median) noexcept nogil
    cdef int remove(self, DOUBLE_t data, DOUBLE_t weight) noexcept nogil
    cdef int pop(self, DOUBLE_t* data, DOUBLE_t* weight) noexcept nogil
    cdef int update_median_parameters_post_remove(
        self, DOUBLE_t data, DOUBLE_t weight,
        DOUBLE_t original_median) noexcept nogil
    cdef DOUBLE_t get_median(self) noexcept nogil
