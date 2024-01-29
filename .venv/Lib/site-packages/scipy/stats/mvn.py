# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.stats` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'mvnun',
    'mvnun_weighted',
    'mvndst',
    'dkblck'
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="stats", module="mvn",
                                   private_modules=["_mvn"], all=__all__,
                                   attribute=name)
