from numpy cimport ndarray


cdef bint c_is_list_like(object, bint) except -1

cpdef ndarray eq_NA_compat(ndarray[object] arr, object key)
