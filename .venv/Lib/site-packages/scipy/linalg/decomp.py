# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.linalg` namespace for importing the functions
# included below.

import warnings
from . import _decomp

__all__ = [  # noqa: F822
    'eig', 'eigvals', 'eigh', 'eigvalsh',
    'eig_banded', 'eigvals_banded',
    'eigh_tridiagonal', 'eigvalsh_tridiagonal', 'hessenberg', 'cdf2rdf',
    'array', 'isfinite', 'inexact', 'nonzero', 'iscomplexobj', 'cast',
    'flatnonzero', 'argsort', 'iscomplex', 'einsum', 'eye', 'inf',
    'LinAlgError', 'norm', 'get_lapack_funcs'
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.linalg.decomp is deprecated and has no attribute "
            f"{name}. Try looking in scipy.linalg instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.linalg` namespace, "
                  "the `scipy.linalg.decomp` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_decomp, name)
