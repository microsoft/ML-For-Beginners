# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.linalg` namespace for importing the functions
# included below.

import warnings
from . import _decomp_svd

__all__ = [  # noqa: F822
    'svd', 'svdvals', 'diagsvd', 'orth', 'subspace_angles', 'null_space',
    'LinAlgError', 'get_lapack_funcs'
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.linalg.decomp_svd is deprecated and has no attribute "
            f"{name}. Try looking in scipy.linalg instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.linalg` namespace, "
                  "the `scipy.linalg.decomp_svd` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_decomp_svd, name)
