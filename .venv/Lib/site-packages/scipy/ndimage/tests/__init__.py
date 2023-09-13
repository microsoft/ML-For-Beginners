from __future__ import annotations
from typing import List, Type 
import numpy

# list of numarray data types
integer_types: list[type] = [
    numpy.int8, numpy.uint8, numpy.int16, numpy.uint16,
    numpy.int32, numpy.uint32, numpy.int64, numpy.uint64]

float_types: list[type] = [numpy.float32, numpy.float64]

complex_types: list[type] = [numpy.complex64, numpy.complex128]

types: list[type] = integer_types + float_types
