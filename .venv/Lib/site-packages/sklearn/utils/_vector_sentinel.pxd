cimport numpy as cnp

from libcpp.vector cimport vector
from ..utils._typedefs cimport intp_t, float64_t, int32_t, int64_t

ctypedef fused vector_typed:
    vector[float64_t]
    vector[intp_t]
    vector[int32_t]
    vector[int64_t]

cdef cnp.ndarray vector_to_nd_array(vector_typed * vect_ptr)
