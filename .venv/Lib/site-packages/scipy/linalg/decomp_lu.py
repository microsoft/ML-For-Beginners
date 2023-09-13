# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.linalg` namespace for importing the functions
# included below.

import warnings
from . import _decomp_lu

__all__ = [  # noqa: F822
    'lu', 'lu_solve', 'lu_factor',
    'asarray_chkfinite', 'LinAlgWarning', 'get_lapack_funcs',
    'get_flinalg_funcs'

]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.linalg.decomp_lu is deprecated and has no attribute "
            f"{name}. Try looking in scipy.linalg instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.linalg` namespace, "
                  "the `scipy.linalg.decomp_lu` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_decomp_lu, name)
