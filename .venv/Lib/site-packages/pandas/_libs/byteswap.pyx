"""
The following are faster versions of struct.unpack that avoid the overhead of Python
function calls.

In the SAS7BDAT parser, they may be called up to (n_rows * n_cols) times.
"""
from cython cimport Py_ssize_t
from libc.stdint cimport (
    uint16_t,
    uint32_t,
    uint64_t,
)
from libc.string cimport memcpy


def read_float_with_byteswap(bytes data, Py_ssize_t offset, bint byteswap):
    cdef uint32_t value
    assert offset + sizeof(value) < len(data)
    cdef const void *ptr = <unsigned char*>(data) + offset
    memcpy(&value, ptr, sizeof(value))
    if byteswap:
        value = _byteswap4(value)

    cdef float res
    memcpy(&res, &value, sizeof(res))
    return res


def read_double_with_byteswap(bytes data, Py_ssize_t offset, bint byteswap):
    cdef uint64_t value
    assert offset + sizeof(value) < len(data)
    cdef const void *ptr = <unsigned char*>(data) + offset
    memcpy(&value, ptr, sizeof(value))
    if byteswap:
        value = _byteswap8(value)

    cdef double res
    memcpy(&res, &value, sizeof(res))
    return res


def read_uint16_with_byteswap(bytes data, Py_ssize_t offset, bint byteswap):
    cdef uint16_t res
    assert offset + sizeof(res) < len(data)
    memcpy(&res, <const unsigned char*>(data) + offset, sizeof(res))
    if byteswap:
        res = _byteswap2(res)
    return res


def read_uint32_with_byteswap(bytes data, Py_ssize_t offset, bint byteswap):
    cdef uint32_t res
    assert offset + sizeof(res) < len(data)
    memcpy(&res, <const unsigned char*>(data) + offset, sizeof(res))
    if byteswap:
        res = _byteswap4(res)
    return res


def read_uint64_with_byteswap(bytes data, Py_ssize_t offset, bint byteswap):
    cdef uint64_t res
    assert offset + sizeof(res) < len(data)
    memcpy(&res, <const unsigned char*>(data) + offset, sizeof(res))
    if byteswap:
        res = _byteswap8(res)
    return res


# Byteswapping

cdef extern from *:
    """
    #ifdef _MSC_VER
        #define _byteswap2 _byteswap_ushort
        #define _byteswap4 _byteswap_ulong
        #define _byteswap8 _byteswap_uint64
    #else
        #define _byteswap2 __builtin_bswap16
        #define _byteswap4 __builtin_bswap32
        #define _byteswap8 __builtin_bswap64
    #endif
    """
    uint16_t _byteswap2(uint16_t)
    uint32_t _byteswap4(uint32_t)
    uint64_t _byteswap8(uint64_t)
