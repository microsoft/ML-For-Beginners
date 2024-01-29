# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.linalg` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'solve', 'solve_triangular', 'solveh_banded', 'solve_banded',
    'solve_toeplitz', 'solve_circulant', 'inv', 'det', 'lstsq',
    'pinv', 'pinvh', 'matrix_balance', 'matmul_toeplitz',
    'atleast_1d', 'atleast_2d', 'get_flinalg_funcs', 'get_lapack_funcs',
    'LinAlgError', 'LinAlgWarning', 'levinson'
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="linalg", module="basic",
                                   private_modules=["_basic"], all=__all__,
                                   attribute=name)
