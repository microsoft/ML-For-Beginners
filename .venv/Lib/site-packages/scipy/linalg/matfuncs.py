# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.linalg` namespace for importing the functions
# included below.

import warnings
from . import _matfuncs

__all__ = [  # noqa: F822
    'expm', 'cosm', 'sinm', 'tanm', 'coshm', 'sinhm',
    'tanhm', 'logm', 'funm', 'signm', 'sqrtm',
    'expm_frechet', 'expm_cond', 'fractional_matrix_power',
    'khatri_rao', 'prod', 'logical_not', 'ravel', 'transpose',
    'conjugate', 'absolute', 'amax', 'sign', 'isfinite', 'single',
    'norm', 'solve', 'inv', 'triu', 'svd', 'schur', 'rsf2csf', 'eps', 'feps'
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.linalg.matfuncs is deprecated and has no attribute "
            f"{name}. Try looking in scipy.linalg instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.linalg` namespace, "
                  "the `scipy.linalg.matfuncs` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_matfuncs, name)
