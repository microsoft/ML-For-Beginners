# Helpers to safely access OpenMP routines
#
# no-op implementations are provided for the case where OpenMP is not available.
#
# All calls to OpenMP routines should be cimported from this module.

cdef extern from *:
    """
    #ifdef _OPENMP
        #include <omp.h>
        #define SKLEARN_OPENMP_PARALLELISM_ENABLED 1
    #else
        #define SKLEARN_OPENMP_PARALLELISM_ENABLED 0
        #define omp_lock_t int
        #define omp_init_lock(l) (void)0
        #define omp_destroy_lock(l) (void)0
        #define omp_set_lock(l) (void)0
        #define omp_unset_lock(l) (void)0
        #define omp_get_thread_num() 0
        #define omp_get_max_threads() 1
    #endif
    """
    bint SKLEARN_OPENMP_PARALLELISM_ENABLED

    ctypedef struct omp_lock_t:
        pass

    void omp_init_lock(omp_lock_t*) noexcept nogil
    void omp_destroy_lock(omp_lock_t*) noexcept nogil
    void omp_set_lock(omp_lock_t*) noexcept nogil
    void omp_unset_lock(omp_lock_t*) noexcept nogil
    int omp_get_thread_num() noexcept nogil
    int omp_get_max_threads() noexcept nogil
