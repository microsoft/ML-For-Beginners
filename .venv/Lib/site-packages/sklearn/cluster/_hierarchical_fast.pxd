from ..utils._typedefs cimport intp_t

cdef class UnionFind:
    cdef intp_t next_label
    cdef intp_t[:] parent
    cdef intp_t[:] size

    cdef void union(self, intp_t m, intp_t n) noexcept
    cdef intp_t fast_find(self, intp_t n) noexcept
