# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.datasets` namespace for importing the dataset functions
# included below.

import warnings
from . import _common


__all__ = [  # noqa: F822
    'central_diff_weights', 'derivative', 'ascent', 'face',
    'electrocardiogram', 'arange', 'newaxis', 'hstack', 'prod', 'array', 'load'
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.misc.common is deprecated and has no attribute "
            f"{name}. Try looking in scipy.datasets instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.misc` namespace, "
                  "the `scipy.misc.common` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_common, name)
