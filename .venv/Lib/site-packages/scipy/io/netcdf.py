# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.io` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation

__all__ = [  # noqa: F822
    'netcdf_file', 'netcdf_variable',
    'array', 'LITTLE_ENDIAN', 'IS_PYPY', 'ABSENT', 'ZERO',
    'NC_BYTE', 'NC_CHAR', 'NC_SHORT', 'NC_INT', 'NC_FLOAT',
    'NC_DOUBLE', 'NC_DIMENSION', 'NC_VARIABLE', 'NC_ATTRIBUTE',
    'FILL_BYTE', 'FILL_CHAR', 'FILL_SHORT', 'FILL_INT', 'FILL_FLOAT',
    'FILL_DOUBLE', 'TYPEMAP', 'FILLMAP', 'REVERSE', 'NetCDFFile',
    'NetCDFVariable'
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="io", module="netcdf",
                                   private_modules=["_netcdf"], all=__all__,
                                   attribute=name)
