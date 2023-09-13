# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.sparse` namespace for importing the functions
# included below.

import warnings
from . import _bsr


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
    'get_index_dtype',
    'getdata',
    'getdtype',
    'isshape',
    'isspmatrix',
    'isspmatrix_bsr',
    'spmatrix',
    'to_native',
    'upcast',
    'warn',
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.sparse.bsr is deprecated and has no attribute "
            f"{name}. Try looking in scipy.sparse instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.sparse` namespace, "
                  "the `scipy.sparse.bsr` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_bsr, name)
