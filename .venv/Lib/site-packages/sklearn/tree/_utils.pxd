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
from ..utils._typedefs cimport float32_t, float64_t, intp_t, int32_t, uint32_t

cdef enum:
    # Max value for our rand_r replacement (near the bottom).
    # We don't use RAND_MAX because it's different across platforms and
    # particularly tiny on Windows/MSVC.
    # It corresponds to the maximum representable value for
    # 32-bit signed integers (i.e. 2^31 - 1).
    RAND_R_MAX = 2147483647


# safe_realloc(&p, n) resizes the allocation of p to n * sizeof(*p) bytes or
# raises a MemoryError. It never calls free, since that's __dealloc__'s job.
#   cdef float32_t *p = NULL
#   safe_realloc(&p, n)
# is equivalent to p = malloc(n * sizeof(*p)) with error checking.
ctypedef fused realloc_ptr:
    # Add pointer types here as needed.
    (float32_t*)
    (intp_t*)
    (unsigned char*)
    (WeightedPQueueRecord*)
    (float64_t*)
    (float64_t**)
    (Node*)
    (Cell*)
    (Node**)

cdef int safe_realloc(realloc_ptr* p, size_t nelems) except -1 nogil


cdef cnp.ndarray sizet_ptr_to_ndarray(intp_t* data, intp_t size)


cdef intp_t rand_int(intp_t low, intp_t high,
                     uint32_t* random_state) noexcept nogil


cdef float64_t rand_uniform(float64_t low, float64_t high,
                            uint32_t* random_state) noexcept nogil


cdef float64_t log(float64_t x) noexcept nogil

# =============================================================================
# WeightedPQueue data structure
# =============================================================================

# A record stored in the WeightedPQueue
cdef struct WeightedPQueueRecord:
    float64_t data
    float64_t weight

cdef class WeightedPQueue:
    cdef intp_t capacity
    cdef intp_t array_ptr
    cdef WeightedPQueueRecord* array_

    cdef bint is_empty(self) noexcept nogil
    cdef int reset(self) except -1 nogil
    cdef intp_t size(self) noexcept nogil
    cdef int push(self, float64_t data, float64_t weight) except -1 nogil
    cdef int remove(self, float64_t data, float64_t weight) noexcept nogil
    cdef int pop(self, float64_t* data, float64_t* weight) noexcept nogil
    cdef int peek(self, float64_t* data, float64_t* weight) noexcept nogil
    cdef float64_t get_weight_from_index(self, intp_t index) noexcept nogil
    cdef float64_t get_value_from_index(self, intp_t index) noexcept nogil


# =============================================================================
# WeightedMedianCalculator data structure
# =============================================================================

cdef class WeightedMedianCalculator:
    cdef intp_t initial_capacity
    cdef WeightedPQueue samples
    cdef float64_t total_weight
    cdef intp_t k
    cdef float64_t sum_w_0_k  # represents sum(weights[0:k]) = w[0] + w[1] + ... + w[k-1]
    cdef intp_t size(self) noexcept nogil
    cdef int push(self, float64_t data, float64_t weight) except -1 nogil
    cdef int reset(self) except -1 nogil
    cdef int update_median_parameters_post_push(
        self, float64_t data, float64_t weight,
        float64_t original_median) noexcept nogil
    cdef int remove(self, float64_t data, float64_t weight) noexcept nogil
    cdef int pop(self, float64_t* data, float64_t* weight) noexcept nogil
    cdef int update_median_parameters_post_remove(
        self, float64_t data, float64_t weight,
        float64_t original_median) noexcept nogil
    cdef float64_t get_median(self) noexcept nogil
