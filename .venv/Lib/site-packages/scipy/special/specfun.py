# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.special` namespace for importing the functions
# included below.

import warnings
from . import _specfun  # type: ignore

__all__ = [  # noqa: F822
    'airyzo',
    'bernob',
    'cerzo',
    'clpmn',
    'clpn',
    'clqmn',
    'clqn',
    'cpbdn',
    'cyzo',
    'eulerb',
    'fcoef',
    'fcszo',
    'jdzo',
    'jyzo',
    'klvnzo',
    'lamn',
    'lamv',
    'lpmn',
    'lpn',
    'lqmn',
    'lqnb',
    'pbdv',
    'rctj',
    'rcty',
    'segv'
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.special.specfun is deprecated and has no attribute "
            f"{name}. Try looking in scipy.special instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.special` namespace, "
                  "the `scipy.special.specfun` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_specfun, name)
