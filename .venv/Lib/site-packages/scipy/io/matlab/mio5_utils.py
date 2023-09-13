# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.io.matlab` namespace for importing the functions
# included below.

import warnings
from . import _mio5_utils


__all__ = [  # noqa: F822
    'VarHeader5', 'VarReader5', 'byteswap_u4', 'chars_to_strings',
    'csc_matrix', 'mio5p', 'miob', 'pycopy', 'swapped_code', 'squeeze_element'
]

def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.io.matlab.mio5_utils is deprecated and has no attribute "
            f"{name}. Try looking in scipy.io.matlab instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.io.matlab` namespace, "
                  "the `scipy.io.matlab.mio5_utils` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_mio5_utils, name)
