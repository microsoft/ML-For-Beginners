"""Utils for interoperability with other libraries.

Just CFFI pointer casting for now.
"""

# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.

from typing import Any


def cast_int_addr(n: Any) -> int:
    """Cast an address to a Python int

    This could be a Python integer or a CFFI pointer
    """
    if isinstance(n, int):
        return n
    try:
        import cffi  # type: ignore
    except ImportError:
        pass
    else:
        # from pyzmq, this is an FFI void *
        ffi = cffi.FFI()
        if isinstance(n, ffi.CData):
            return int(ffi.cast("size_t", n))

    raise ValueError("Cannot cast %r to int" % n)
