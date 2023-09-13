"""zmq error functions"""

# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.

from ._cffi import ffi
from ._cffi import lib as C


def strerror(errno):
    s = ffi.string(C.zmq_strerror(errno))
    if not isinstance(s, str):
        # py3
        s = s.decode()
    return s


zmq_errno = C.zmq_errno

__all__ = ['strerror', 'zmq_errno']
