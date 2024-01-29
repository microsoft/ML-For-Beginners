# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.fftpack` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation

__all__ = [  # noqa: F822
    'dct', 'idct', 'dst', 'idst', 'dctn', 'idctn', 'dstn', 'idstn'
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="fftpack", module="realtransforms",
                                   private_modules=["_realtransforms"], all=__all__,
                                   attribute=name)
