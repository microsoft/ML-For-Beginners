# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.sparse` namespace for importing the functions
# included below.

import warnings
from . import _csc


__all__ = [  # noqa: F822
    'csc_matrix',
    'csc_tocsr',
    'expandptr',
    'get_index_dtype',
    'isspmatrix_csc',
    'spmatrix',
    'upcast',
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.sparse.csc is deprecated and has no attribute "
            f"{name}. Try looking in scipy.sparse instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.sparse` namespace, "
                  "the `scipy.sparse.csc` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_csc, name)
