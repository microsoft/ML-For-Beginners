# destined to be used in a LowLevelCallable
cdef double _geninvgauss_pdf(double x, void *user_data) except * nogil
cdef double _studentized_range_cdf(int n, double[2] x, void *user_data) noexcept nogil
cdef double _studentized_range_cdf_asymptotic(double z, void *user_data) noexcept nogil
cdef double _studentized_range_pdf(int n, double[2] x, void *user_data) noexcept nogil
cdef double _studentized_range_pdf_asymptotic(double z, void *user_data) noexcept nogil
cdef double _studentized_range_moment(int n, double[3] x_arg, void *user_data) noexcept nogil
cdef double _genhyperbolic_pdf(double x, void *user_data) except * nogil
cdef double _genhyperbolic_logpdf(double x, void *user_data) except * nogil
