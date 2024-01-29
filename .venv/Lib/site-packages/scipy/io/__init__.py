"""
==================================
Input and output (:mod:`scipy.io`)
==================================

.. currentmodule:: scipy.io

SciPy has many modules, classes, and functions available to read data
from and write data to a variety of file formats.

.. seealso:: `NumPy IO routines <https://www.numpy.org/devdocs/reference/routines.io.html>`__

MATLAB® files
=============

.. autosummary::
   :toctree: generated/

   loadmat - Read a MATLAB style mat file (version 4 through 7.1)
   savemat - Write a MATLAB style mat file (version 4 through 7.1)
   whosmat - List contents of a MATLAB style mat file (version 4 through 7.1)

For low-level MATLAB reading and writing utilities, see `scipy.io.matlab`.

IDL® files
==========

.. autosummary::
   :toctree: generated/

   readsav - Read an IDL 'save' file

Matrix Market files
===================

.. autosummary::
   :toctree: generated/

   mminfo - Query matrix info from Matrix Market formatted file
   mmread - Read matrix from Matrix Market formatted file
   mmwrite - Write matrix to Matrix Market formatted file

Unformatted Fortran files
===============================

.. autosummary::
   :toctree: generated/

   FortranFile - A file object for unformatted sequential Fortran files
   FortranEOFError - Exception indicating the end of a well-formed file
   FortranFormattingError - Exception indicating an inappropriate end

Netcdf
======

.. autosummary::
   :toctree: generated/

   netcdf_file - A file object for NetCDF data
   netcdf_variable - A data object for the netcdf module

Harwell-Boeing files
====================

.. autosummary::
   :toctree: generated/

   hb_read   -- read H-B file
   hb_write  -- write H-B file

Wav sound files (:mod:`scipy.io.wavfile`)
=========================================

.. module:: scipy.io.wavfile

.. autosummary::
   :toctree: generated/

   read
   write
   WavFileWarning

Arff files (:mod:`scipy.io.arff`)
=================================

.. module:: scipy.io.arff

.. autosummary::
   :toctree: generated/

   loadarff
   MetaData
   ArffError
   ParseArffError
"""
# matfile read and write
from .matlab import loadmat, savemat, whosmat

# netCDF file support
from ._netcdf import netcdf_file, netcdf_variable

# Fortran file support
from ._fortran import FortranFile, FortranEOFError, FortranFormattingError

from ._fast_matrix_market import mminfo, mmread, mmwrite
from ._idl import readsav
from ._harwell_boeing import hb_read, hb_write

# Deprecated namespaces, to be removed in v2.0.0
from . import arff, harwell_boeing, idl, mmio, netcdf, wavfile

__all__ = [s for s in dir() if not s.startswith('_')]

from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
