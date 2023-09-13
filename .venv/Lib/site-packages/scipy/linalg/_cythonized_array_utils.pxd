cimport numpy as cnp

ctypedef fused lapack_t:
    float
    double
    (float complex)
    (double complex)

ctypedef fused lapack_cz_t:
    (float complex)
    (double complex)

ctypedef fused lapack_sd_t:
    float
    double

ctypedef fused np_numeric_t:
    cnp.int8_t
    cnp.int16_t
    cnp.int32_t
    cnp.int64_t
    cnp.uint8_t
    cnp.uint16_t
    cnp.uint32_t
    cnp.uint64_t
    cnp.float32_t
    cnp.float64_t
    cnp.longdouble_t
    cnp.complex64_t
    cnp.complex128_t

ctypedef fused np_complex_numeric_t:
    cnp.complex64_t
    cnp.complex128_t


cdef void swap_c_and_f_layout(lapack_t *a, lapack_t *b, int r, int c) noexcept nogil
cdef (int, int) band_check_internal_c(np_numeric_t[:, ::1]A) noexcept nogil
cdef bint is_sym_her_real_c_internal(np_numeric_t[:, ::1]A) noexcept nogil
cdef bint is_sym_her_complex_c_internal(np_complex_numeric_t[:, ::1]A) noexcept nogil
