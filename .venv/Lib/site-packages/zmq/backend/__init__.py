"""Import basic exposure of libzmq C API as a backend"""

# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.


import os
import platform

from .select import public_api, select_backend

if 'PYZMQ_BACKEND' in os.environ:
    backend = os.environ['PYZMQ_BACKEND']
    if backend in ('cython', 'cffi'):
        backend = 'zmq.backend.%s' % backend
    _ns = select_backend(backend)
else:
    # default to cython, fallback to cffi
    # (reverse on PyPy)
    if platform.python_implementation() == 'PyPy':
        first, second = ('zmq.backend.cffi', 'zmq.backend.cython')
    else:
        first, second = ('zmq.backend.cython', 'zmq.backend.cffi')

    try:
        _ns = select_backend(first)
    except Exception as original_error:
        try:
            _ns = select_backend(second)
        except ImportError:
            raise original_error from None

globals().update(_ns)

__all__ = public_api
