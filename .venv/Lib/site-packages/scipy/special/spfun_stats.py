# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.special` namespace for importing the functions
# included below.

import warnings
from . import _spfun_stats

__all__ = ['multigammaln', 'loggam']  # noqa: F822


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.special.spfun_stats is deprecated and has no attribute "
            f"{name}. Try looking in scipy.special instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.special` namespace, "
                  "the `scipy.special.spfun_stats` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_spfun_stats, name)
