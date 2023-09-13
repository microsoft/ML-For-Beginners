# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.linalg` namespace for importing the functions
# included below.

import warnings
from . import _special_matrices

__all__ = [  # noqa: F822
    'tri', 'tril', 'triu', 'toeplitz', 'circulant', 'hankel',
    'hadamard', 'leslie', 'kron', 'block_diag', 'companion',
    'helmert', 'hilbert', 'invhilbert', 'pascal', 'invpascal', 'dft',
    'fiedler', 'fiedler_companion', 'convolution_matrix', 'as_strided'
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.linalg.special_matrices is deprecated and has no attribute "
            f"{name}. Try looking in scipy.linalg instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.linalg` namespace, the"
                  " `scipy.linalg.special_matrices` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_special_matrices, name)
