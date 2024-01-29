# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.ndimage` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


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
    return _sub_module_deprecation(sub_package='ndimage', module='measurements',
                                   private_modules=['_measurements'], all=__all__,
                                   attribute=name)
