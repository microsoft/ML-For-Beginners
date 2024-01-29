# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.io.matlab` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation

__all__ = [  # noqa: F822
    'VarHeader5', 'VarReader5', 'byteswap_u4', 'chars_to_strings',
    'csc_matrix', 'mio5p', 'pycopy', 'swapped_code', 'squeeze_element'
]

def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="io.matlab", module="mio5_utils",
                                   private_modules=["_mio5_utils"], all=__all__,
                                   attribute=name)
