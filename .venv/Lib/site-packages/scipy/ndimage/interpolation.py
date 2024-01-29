# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.ndimage` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'spline_filter1d', 'spline_filter',
    'geometric_transform', 'map_coordinates',
    'affine_transform', 'shift', 'zoom', 'rotate',
    'docfiller'
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package='ndimage', module='interpolation',
                                   private_modules=['_interpolation'], all=__all__,
                                   attribute=name)
