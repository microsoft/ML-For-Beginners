from cython cimport floating
from ..utils._typedefs cimport float64_t, intp_t

cdef int partition_node_indices(
        const floating *data,
        intp_t *node_indices,
        intp_t split_dim,
        intp_t split_index,
        intp_t n_features,
        intp_t n_points) except -1
