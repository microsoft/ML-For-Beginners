# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.ndimage` namespace for importing the functions
# included below.

import warnings
from . import _measurements


__all__ = [  # noqa: F822
    'label', 'find_objects', 'labeled_comprehension',
    'sum', 'mean', 'variance', 'standard_deviation',
    'minimum', 'maximum', 'median', 'minimum_position',
    'maximum_position', 'extrema', 'center_of_mass',
    'histogram', 'watershed_ift', 'sum_labels'
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.ndimage.measurements is deprecated and has no attribute "
            f"{name}. Try looking in scipy.ndimage instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.ndimage` namespace, "
                  "the `scipy.ndimage.measurements` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_measurements, name)
