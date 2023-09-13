"""CFFI backend (for PyPy)"""

# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.

from zmq.backend.cffi import _poll, context, devices, error, message, socket, utils

from ._cffi import ffi
from ._cffi import lib as C


def zmq_version_info():
    """Get libzmq version as tuple of ints"""
    major = ffi.new('int*')
    minor = ffi.new('int*')
    patch = ffi.new('int*')

    C.zmq_version(major, minor, patch)

    return (int(major[0]), int(minor[0]), int(patch[0]))


__all__ = ["zmq_version_info"]
for submod in (error, message, context, socket, _poll, devices, utils):
    __all__.extend(submod.__all__)

from ._poll import *
from .context import *
from .devices import *
from .error import *
from .message import *
from .socket import *
from .utils import *
