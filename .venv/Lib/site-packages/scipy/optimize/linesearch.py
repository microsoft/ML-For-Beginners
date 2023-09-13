# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.optimize` namespace for importing the functions
# included below.

import warnings
from . import _linesearch


__all__ = [  # noqa: F822
    'LineSearchWarning',
    'line_search',
    'line_search_BFGS',
    'line_search_armijo',
    'line_search_wolfe1',
    'line_search_wolfe2',
    'minpack2',
    'scalar_search_armijo',
    'scalar_search_wolfe1',
    'scalar_search_wolfe2',
    'warn',
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.optimize.linesearch is deprecated and has no attribute "
            f"{name}. Try looking in scipy.optimize instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.optimize` namespace, "
                  "the `scipy.optimize.linesearch` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_linesearch, name)
