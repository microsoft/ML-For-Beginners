# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.io.matlab` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation

__all__ = [  # noqa: F822
    'MDTYPES', 'MatlabFunction', 'MatlabObject', 'MatlabOpaque',
    'NP_TO_MTYPES', 'NP_TO_MXTYPES', 'OPAQUE_DTYPE', 'codecs_template',
    'mat_struct', 'mclass_dtypes_template', 'mclass_info', 'mdtypes_template',
    'miCOMPRESSED', 'miDOUBLE', 'miINT16', 'miINT32', 'miINT64', 'miINT8',
    'miMATRIX', 'miSINGLE', 'miUINT16', 'miUINT32', 'miUINT64', 'miUINT8',
    'miUTF16', 'miUTF32', 'miUTF8', 'mxCELL_CLASS', 'mxCHAR_CLASS',
    'mxDOUBLE_CLASS', 'mxFUNCTION_CLASS', 'mxINT16_CLASS', 'mxINT32_CLASS',
    'mxINT64_CLASS', 'mxINT8_CLASS', 'mxOBJECT_CLASS',
    'mxOBJECT_CLASS_FROM_MATRIX_H', 'mxOPAQUE_CLASS', 'mxSINGLE_CLASS',
    'mxSPARSE_CLASS', 'mxSTRUCT_CLASS', 'mxUINT16_CLASS', 'mxUINT32_CLASS',
    'mxUINT64_CLASS', 'mxUINT8_CLASS', 'convert_dtypes'
]

def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="io.matlab", module="mio5_params",
                                   private_modules=["_mio5_params"], all=__all__,
                                   attribute=name)
