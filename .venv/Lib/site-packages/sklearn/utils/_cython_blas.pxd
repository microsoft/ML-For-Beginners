from cython cimport floating


cpdef enum BLAS_Order:
    RowMajor  # C contiguous
    ColMajor  # Fortran contiguous


cpdef enum BLAS_Trans:
    NoTrans = 110  # correspond to 'n'
    Trans = 116    # correspond to 't'


# BLAS Level 1 ################################################################
cdef floating _dot(int, const floating*, int, const floating*, int) noexcept nogil

cdef floating _asum(int, const floating*, int) noexcept nogil

cdef void _axpy(int, floating, const floating*, int, floating*, int) noexcept nogil

cdef floating _nrm2(int, const floating*, int) noexcept nogil

cdef void _copy(int, const floating*, int, const floating*, int) noexcept nogil

cdef void _scal(int, floating, const floating*, int) noexcept nogil

cdef void _rotg(floating*, floating*, floating*, floating*) noexcept nogil

cdef void _rot(int, floating*, int, floating*, int, floating, floating) noexcept nogil

# BLAS Level 2 ################################################################
cdef void _gemv(BLAS_Order, BLAS_Trans, int, int, floating, const floating*, int,
                const floating*, int, floating, floating*, int) noexcept nogil

cdef void _ger(BLAS_Order, int, int, floating, const floating*, int, const floating*,
               int, floating*, int) noexcept nogil

# BLASLevel 3 ################################################################
cdef void _gemm(BLAS_Order, BLAS_Trans, BLAS_Trans, int, int, int, floating,
                const floating*, int, const floating*, int, floating, floating*,
                int) noexcept nogil
