# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.ndimage` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


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
    return _sub_module_deprecation(sub_package='ndimage', module='morphology',
                                   private_modules=['_morphology'], all=__all__,
                                   attribute=name)
