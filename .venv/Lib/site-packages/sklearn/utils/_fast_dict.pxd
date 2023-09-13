# Author: Gael Varoquaux
# License: BSD
"""
Uses C++ map containers for fast dict-like behavior with keys being
integers, and values float.
"""

from libcpp.map cimport map as cpp_map

from ._typedefs cimport float64_t, intp_t


###############################################################################
# An object to be used in Python

cdef class IntFloatDict:
    cdef cpp_map[intp_t, float64_t] my_map
    cdef _to_arrays(self, intp_t [:] keys, float64_t [:] values)
