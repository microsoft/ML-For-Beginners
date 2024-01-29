# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.sparse` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'bsr_matmat',
    'bsr_matrix',
    'bsr_matvec',
    'bsr_matvecs',
    'bsr_sort_indices',
    'bsr_tocsr',
    'bsr_transpose',
    'check_shape',
    'csr_matmat_maxnnz',
    'getdata',
    'getdtype',
    'isshape',
    'isspmatrix_bsr',
    'spmatrix',
    'to_native',
    'upcast',
    'warn',
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="sparse", module="bsr",
                                   private_modules=["_bsr"], all=__all__,
                                   attribute=name)
