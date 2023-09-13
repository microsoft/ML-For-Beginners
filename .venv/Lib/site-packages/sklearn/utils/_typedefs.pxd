# Commonly used types
# These are redefinitions of the ones defined by numpy in
# https://github.com/numpy/numpy/blob/main/numpy/__init__.pxd
# and exposed by cython in
# https://github.com/cython/cython/blob/master/Cython/Includes/numpy/__init__.pxd.
# It will eventually avoid having to always include the numpy headers even when we
# would only use it for the types.
#
# When used to declare variables that will receive values from numpy arrays, it
# should match the dtype of the array. For example, to declare a variable that will
# receive values from a numpy array of dtype np.float64, the type float64_t must be
# used.
#
# TODO: Stop defining custom types locally or globally like DTYPE_t and friends and
# use these consistently throughout the codebase.
# NOTE: Extend this list as needed when converting more cython extensions.
ctypedef unsigned char uint8_t
ctypedef Py_ssize_t intp_t
ctypedef float float32_t
ctypedef double float64_t
# Sparse matrices indices and indices' pointers arrays must use int32_t over
# intp_t because intp_t is platform dependent.
# When large sparse matrices are supported, indexing must use int64_t.
# See https://github.com/scikit-learn/scikit-learn/issues/23653 which tracks the
# ongoing work to support large sparse matrices.
ctypedef signed int int32_t
ctypedef signed long long int64_t
