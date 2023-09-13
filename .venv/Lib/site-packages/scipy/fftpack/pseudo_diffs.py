# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.fftpack` namespace for importing the functions
# included below.

import warnings
from . import _pseudo_diffs

__all__ = [  # noqa: F822
    'diff',
    'tilbert', 'itilbert', 'hilbert', 'ihilbert',
    'cs_diff', 'cc_diff', 'sc_diff', 'ss_diff',
    'shift', 'iscomplexobj', 'convolve'
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.fftpack.pseudo_diffs is deprecated and has no attribute "
            f"{name}. Try looking in scipy.fftpack instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.fftpack` namespace, "
                  "the `scipy.fftpack.pseudo_diffs` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_pseudo_diffs, name)
