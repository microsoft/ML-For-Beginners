# Heap routines, used in various Cython implementations.

from cython cimport floating

from ._typedefs cimport intp_t


cdef int heap_push(
    floating* values,
    intp_t* indices,
    intp_t size,
    floating val,
    intp_t val_idx,
) noexcept nogil
