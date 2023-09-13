from numpy cimport (
    ndarray,
    uint8_t,
)


cpdef bint is_matching_na(object left, object right, bint nan_matches_none=*)
cpdef bint check_na_tuples_nonequal(object left, object right)

cpdef bint checknull(object val, bint inf_as_na=*)
cpdef ndarray[uint8_t] isnaobj(ndarray arr, bint inf_as_na=*)

cdef bint is_null_datetime64(v)
cdef bint is_null_timedelta64(v)
cdef bint checknull_with_nat_and_na(object obj)

cdef class C_NAType:
    pass

cdef C_NAType C_NA
