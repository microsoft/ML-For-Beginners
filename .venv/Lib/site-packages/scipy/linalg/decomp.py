# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.linalg` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'eig', 'eigvals', 'eigh', 'eigvalsh',
    'eig_banded', 'eigvals_banded',
    'eigh_tridiagonal', 'eigvalsh_tridiagonal', 'hessenberg', 'cdf2rdf',
    'array', 'isfinite', 'inexact', 'nonzero', 'iscomplexobj',
    'flatnonzero', 'argsort', 'iscomplex', 'einsum', 'eye', 'inf',
    'LinAlgError', 'norm', 'get_lapack_funcs'
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="linalg", module="decomp",
                                   private_modules=["_decomp"], all=__all__,
                                   attribute=name)
