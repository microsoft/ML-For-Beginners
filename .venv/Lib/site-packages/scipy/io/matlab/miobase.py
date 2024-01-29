# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.io.matlab` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation

__all__ = [  # noqa: F822
    'MatFileReader', 'MatReadError', 'MatReadWarning',
    'MatVarReader', 'MatWriteError', 'arr_dtype_number',
    'arr_to_chars', 'convert_dtypes', 'doc_dict',
    'docfiller', 'get_matfile_version',
    'matdims', 'read_dtype', 'doccer', 'boc'
]

def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="io.matlab", module="miobase",
                                   private_modules=["_miobase"], all=__all__,
                                   attribute=name)
