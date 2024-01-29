# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.datasets` namespace for importing the dataset functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation

__all__ = [  # noqa: F822
    'central_diff_weights', 'derivative', 'ascent', 'face',
    'electrocardiogram', 'array', 'load'
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="misc", module="common",
                                   private_modules=["_common"], all=__all__,
                                   attribute=name)
