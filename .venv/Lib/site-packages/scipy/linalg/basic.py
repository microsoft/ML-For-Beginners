# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.linalg` namespace for importing the functions
# included below.

import warnings
from . import _basic

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
    if name not in __all__:
        raise AttributeError(
            "scipy.linalg.basic is deprecated and has no attribute "
            f"{name}. Try looking in scipy.linalg instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.linalg` namespace, "
                  "the `scipy.linalg.basic` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_basic, name)
