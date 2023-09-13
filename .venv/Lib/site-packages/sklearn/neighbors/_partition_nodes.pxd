from ..utils._typedefs cimport float64_t, intp_t

cdef int partition_node_indices(
        float64_t *data,
        intp_t *node_indices,
        intp_t split_dim,
        intp_t split_index,
        intp_t n_features,
        intp_t n_points) except -1
