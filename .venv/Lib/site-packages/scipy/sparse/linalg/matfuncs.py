# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.sparse.linalg` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'expm', 'inv', 'solve', 'solve_triangular',
    'spsolve', 'is_pydata_spmatrix', 'LinearOperator',
    'UPPER_TRIANGULAR', 'MatrixPowerOperator', 'ProductOperator'
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="sparse.linalg", module="matfuncs",
                                   private_modules=["_matfuncs"], all=__all__,
                                   attribute=name)
