"""
Module to read ARFF files
=========================
ARFF is the standard data format for WEKA.
It is a text file format which support numerical, string and data values.
The format can also represent missing data and sparse data.

Notes
-----
The ARFF support in ``scipy.io`` provides file reading functionality only.
For more extensive ARFF functionality, see `liac-arff
<https://github.com/renatopp/liac-arff>`_.

See the `WEKA website <http://weka.wikispaces.com/ARFF>`_
for more details about the ARFF format and available datasets.

"""
from ._arffread import *
from . import _arffread

# Deprecated namespaces, to be removed in v2.0.0
from .import arffread

__all__ = _arffread.__all__ + ['arffread']

from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
