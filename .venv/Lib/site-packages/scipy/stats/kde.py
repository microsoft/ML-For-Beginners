# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.stats` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'gaussian_kde', 'linalg', 'logsumexp', 'check_random_state',
    'atleast_2d', 'reshape', 'newaxis', 'exp', 'ravel', 'power',
    'atleast_1d', 'squeeze', 'sum', 'transpose', 'cov',
    'gaussian_kernel_estimate'
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="stats", module="kde",
                                   private_modules=["_kde"], all=__all__,
                                   attribute=name)
