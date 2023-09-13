# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.sparse` namespace for importing the functions
# included below.

import warnings
from . import _lil


__all__ = [  # noqa: F822
    'INT_TYPES',
    'IndexMixin',
    'bisect_left',
    'check_reshape_kwargs',
    'check_shape',
    'get_index_dtype',
    'getdtype',
    'isscalarlike',
    'isshape',
    'isspmatrix',
    'isspmatrix_lil',
    'lil_matrix',
    'spmatrix',
    'upcast_scalar',
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.sparse.lil is deprecated and has no attribute "
            f"{name}. Try looking in scipy.sparse instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.sparse` namespace, "
                  "the `scipy.sparse.lil` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_lil, name)
