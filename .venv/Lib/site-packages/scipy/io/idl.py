# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.io` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation

__all__ = [  # noqa: F822
    'readsav', 'DTYPE_DICT', 'RECTYPE_DICT', 'STRUCT_DICT',
    'Pointer', 'ObjectPointer', 'AttrDict'
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="io", module="idl",
                                   private_modules=["_idl"], all=__all__,
                                   attribute=name)
