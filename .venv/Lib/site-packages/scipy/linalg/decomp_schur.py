# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.linalg` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'schur', 'rsf2csf', 'asarray_chkfinite', 'single', 'array', 'norm',
    'LinAlgError', 'get_lapack_funcs', 'eigvals', 'eps', 'feps'
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="linalg", module="decomp_schur",
                                   private_modules=["_decomp_schur"], all=__all__,
                                   attribute=name)

