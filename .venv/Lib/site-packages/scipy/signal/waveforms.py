# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.signal` namespace for importing the functions
# included below.

import warnings
from . import _waveforms

__all__ = [  # noqa: F822
    'sawtooth', 'square', 'gausspulse', 'chirp', 'sweep_poly',
    'unit_impulse', 'place', 'nan', 'mod', 'extract', 'log', 'exp',
    'polyval', 'polyint'
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.signal.waveforms is deprecated and has no attribute "
            f"{name}. Try looking in scipy.signal instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.signal` namespace, "
                  "the `scipy.signal.waveforms` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_waveforms, name)
