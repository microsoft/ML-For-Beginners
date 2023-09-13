# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.io.matlab` namespace for importing the functions
# included below.

import warnings
from . import _mio4


__all__ = [  # noqa: F822
    'MatFile4Reader', 'MatFile4Writer', 'SYS_LITTLE_ENDIAN',
    'VarHeader4', 'VarReader4', 'VarWriter4', 'arr_to_2d', 'mclass_info',
    'mdtypes_template', 'miDOUBLE', 'miINT16', 'miINT32', 'miSINGLE',
    'miUINT16', 'miUINT8', 'mxCHAR_CLASS', 'mxFULL_CLASS', 'mxSPARSE_CLASS',
    'np_to_mtypes', 'order_codes', 'MatFileReader', 'docfiller',
    'matdims', 'read_dtype', 'convert_dtypes', 'arr_to_chars',
    'arr_dtype_number', 'squeeze_element', 'chars_to_strings'
]

def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.io.matlab.mio4 is deprecated and has no attribute "
            f"{name}. Try looking in scipy.io.matlab instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.io.matlab` namespace, "
                  "the `scipy.io.matlab.mio4` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_mio4, name)
