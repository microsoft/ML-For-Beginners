# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.io` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation

__all__ = [  # noqa: F822
    'mminfo', 'mmread', 'mmwrite', 'MMFile',
    'coo_matrix', 'asstr'
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="io", module="mmio",
                                   private_modules=["_mmio"], all=__all__,
                                   attribute=name)
