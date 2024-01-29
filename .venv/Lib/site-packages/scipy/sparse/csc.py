# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.sparse` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'csc_matrix',
    'csc_tocsr',
    'expandptr',
    'isspmatrix_csc',
    'spmatrix',
    'upcast',
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="sparse", module="csc",
                                   private_modules=["_csc"], all=__all__,
                                   attribute=name)
