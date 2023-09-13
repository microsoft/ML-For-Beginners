# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.ndimage` namespace for importing the functions
# included below.

import warnings
from . import _morphology


__all__ = [  # noqa: F822
    'iterate_structure', 'generate_binary_structure',
    'binary_erosion', 'binary_dilation', 'binary_opening',
    'binary_closing', 'binary_hit_or_miss', 'binary_propagation',
    'binary_fill_holes', 'grey_erosion', 'grey_dilation',
    'grey_opening', 'grey_closing', 'morphological_gradient',
    'morphological_laplace', 'white_tophat', 'black_tophat',
    'distance_transform_bf', 'distance_transform_cdt',
    'distance_transform_edt'
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.ndimage.morphology is deprecated and has no attribute "
            f"{name}. Try looking in scipy.ndimage instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.ndimage` namespace, "
                  "the `scipy.ndimage.morphology` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_morphology, name)
