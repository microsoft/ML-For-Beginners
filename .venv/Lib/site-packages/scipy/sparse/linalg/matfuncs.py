# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.sparse.linalg` namespace for importing the functions
# included below.

import warnings
from . import _matfuncs


__all__ = [  # noqa: F822
    'expm', 'inv', 'solve', 'solve_triangular',
    'isspmatrix', 'spsolve', 'is_pydata_spmatrix', 'LinearOperator',
    'UPPER_TRIANGULAR', 'MatrixPowerOperator', 'ProductOperator'
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.sparse.linalg.matfuncs is deprecated and has no attribute "
            f"{name}. Try looking in scipy.sparse.linalg instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.sparse.linalg` namespace, "
                  "the `scipy.sparse.linalg.matfuncs` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_matfuncs, name)
