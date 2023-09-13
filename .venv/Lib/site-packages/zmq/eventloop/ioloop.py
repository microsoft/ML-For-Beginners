"""tornado IOLoop API with zmq compatibility

This module is deprecated in pyzmq 17.
To use zmq with tornado,
eventloop integration is no longer required
and tornado itself should be used.
"""

# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.


import warnings


def _deprecated():
    warnings.warn(
        "zmq.eventloop.ioloop is deprecated in pyzmq 17."
        " pyzmq now works with default tornado and asyncio eventloops.",
        DeprecationWarning,
        stacklevel=3,
    )


_deprecated()

from tornado.ioloop import *  # noqa
from tornado.ioloop import IOLoop

ZMQIOLoop = IOLoop


def install():
    """DEPRECATED

    pyzmq 17 no longer needs any special integration for tornado.
    """
    _deprecated()
