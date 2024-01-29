# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.sparse.linalg` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'ArpackError', 'ArpackNoConvergence', 'ArpackError',
    'eigs', 'eigsh', 'lobpcg', 'svds', 'arpack', 'test'
]

eigen_modules = ['arpack']


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="sparse.linalg", module="eigen",
                                   private_modules=["_eigen"], all=__all__,
                                   attribute=name)
