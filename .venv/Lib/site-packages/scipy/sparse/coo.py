# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.sparse` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'SparseEfficiencyWarning',
    'check_reshape_kwargs',
    'check_shape',
    'coo_matrix',
    'coo_matvec',
    'coo_tocsr',
    'coo_todense',
    'downcast_intp_index',
    'getdata',
    'getdtype',
    'isshape',
    'isspmatrix_coo',
    'operator',
    'spmatrix',
    'to_native',
    'upcast',
    'upcast_char',
    'warn',
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="sparse", module="coo",
                                   private_modules=["_coo"], all=__all__,
                                   attribute=name)
