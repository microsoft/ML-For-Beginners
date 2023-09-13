from cython cimport floating


cdef floating _euclidean_dense_dense(
    const floating*,
    const floating*,
    int,
    bint
) noexcept nogil

cdef floating _euclidean_sparse_dense(
    const floating[::1],
    const int[::1],
    const floating[::1],
    floating,
    bint
) noexcept nogil

cpdef void _relocate_empty_clusters_dense(
    const floating[:, ::1],
    const floating[::1],
    const floating[:, ::1],
    floating[:, ::1],
    floating[::1],
    const int[::1]
)

cpdef void _relocate_empty_clusters_sparse(
    const floating[::1],
    const int[::1],
    const int[::1],
    const floating[::1],
    const floating[:, ::1],
    floating[:, ::1],
    floating[::1],
    const int[::1]
)

cdef void _average_centers(
    floating[:, ::1],
    const floating[::1]
)

cdef void _center_shift(
    const floating[:, ::1],
    const floating[:, ::1],
    floating[::1]
)
