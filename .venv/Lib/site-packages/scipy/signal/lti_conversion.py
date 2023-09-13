# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.signal` namespace for importing the functions
# included below.

import warnings
from . import _lti_conversion


__all__ = [  # noqa: F822
    'tf2ss', 'abcd_normalize', 'ss2tf', 'zpk2ss', 'ss2zpk',
    'cont2discrete','eye', 'atleast_2d',
    'poly', 'prod', 'array', 'outer', 'linalg', 'tf2zpk', 'zpk2tf', 'normalize'
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.signal.lti_conversion is deprecated and has no attribute "
            f"{name}. Try looking in scipy.signal instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.signal` namespace, "
                  "the `scipy.signal.lti_conversion` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_lti_conversion, name)
