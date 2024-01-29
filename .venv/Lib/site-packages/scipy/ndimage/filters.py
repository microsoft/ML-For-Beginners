# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.ndimage` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'correlate1d', 'convolve1d', 'gaussian_filter1d',
    'gaussian_filter', 'prewitt', 'sobel', 'generic_laplace',
    'laplace', 'gaussian_laplace', 'generic_gradient_magnitude',
    'gaussian_gradient_magnitude', 'correlate', 'convolve',
    'uniform_filter1d', 'uniform_filter', 'minimum_filter1d',
    'maximum_filter1d', 'minimum_filter', 'maximum_filter',
    'rank_filter', 'median_filter', 'percentile_filter',
    'generic_filter1d', 'generic_filter'
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package='ndimage', module='filters',
                                   private_modules=['_filters'], all=__all__,
                                   attribute=name)
