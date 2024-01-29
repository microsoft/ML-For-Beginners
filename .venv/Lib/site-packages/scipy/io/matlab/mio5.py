# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.io.matlab` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation

__all__ = [  # noqa: F822
    'mclass_info', 'mxCHAR_CLASS', 'mxSPARSE_CLASS',
    'BytesIO', 'native_code',
    'swapped_code', 'MatFileReader', 'docfiller', 'matdims',
    'read_dtype', 'arr_to_chars', 'arr_dtype_number', 'MatWriteError',
    'MatReadError', 'MatReadWarning', 'VarReader5', 'MatlabObject',
    'MatlabFunction', 'MDTYPES', 'NP_TO_MTYPES', 'NP_TO_MXTYPES',
    'miCOMPRESSED', 'miMATRIX', 'miINT8', 'miUTF8', 'miUINT32',
    'mxCELL_CLASS', 'mxSTRUCT_CLASS', 'mxOBJECT_CLASS', 'mxDOUBLE_CLASS',
    'mat_struct', 'ZlibInputStream', 'MatFile5Reader', 'varmats_from_mat',
    'EmptyStructMarker', 'to_writeable', 'NDT_FILE_HDR', 'NDT_TAG_FULL',
    'NDT_TAG_SMALL', 'NDT_ARRAY_FLAGS', 'VarWriter5', 'MatFile5Writer'
]

def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="io.matlab", module="mio5",
                                   private_modules=["_mio5"], all=__all__,
                                   attribute=name)
