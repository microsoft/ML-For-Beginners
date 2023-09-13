# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.io.matlab` namespace for importing the functions
# included below.

import warnings
from . import _streams


__all__ = [  # noqa: F822
    'BLOCK_SIZE', 'GenericStream', 'ZlibInputStream', 'make_stream'
]

def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.io.matlab.streams is deprecated and has no attribute "
            f"{name}. Try looking in scipy.io.matlab instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.io.matlab` namespace, "
                  "the `scipy.io.matlab.streams` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_streams, name)
