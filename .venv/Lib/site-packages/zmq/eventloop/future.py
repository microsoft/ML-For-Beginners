"""Future-returning APIs for tornado coroutines.

.. seealso::

    :mod:`zmq.asyncio`

"""

# Copyright (c) PyZMQ Developers.
# Distributed under the terms of the Modified BSD License.

import asyncio
import warnings
from typing import Any, Type

from tornado.concurrent import Future
from tornado.ioloop import IOLoop

import zmq as _zmq
from zmq._future import _AsyncPoller, _AsyncSocket


class CancelledError(Exception):
    pass


class _TornadoFuture(Future):
    """Subclass Tornado Future, reinstating cancellation."""

    def cancel(self):
        if self.done():
            return False
        self.set_exception(CancelledError())
        return True

    def cancelled(self):
        return self.done() and isinstance(self.exception(), CancelledError)


class _CancellableTornadoTimeout:
    def __init__(self, loop, timeout):
        self.loop = loop
        self.timeout = timeout

    def cancel(self):
        self.loop.remove_timeout(self.timeout)


# mixin for tornado/asyncio compatibility


class _AsyncTornado:
    _Future: Type[asyncio.Future] = _TornadoFuture
    _READ = IOLoop.READ
    _WRITE = IOLoop.WRITE

    def _default_loop(self):
        return IOLoop.current()

    def _call_later(self, delay, callback):
        io_loop = self._get_loop()
        timeout = io_loop.call_later(delay, callback)
        return _CancellableTornadoTimeout(io_loop, timeout)


class Poller(_AsyncTornado, _AsyncPoller):
    def _watch_raw_socket(self, loop, socket, evt, f):
        """Schedule callback for a raw socket"""
        loop.add_handler(socket, lambda *args: f(), evt)

    def _unwatch_raw_sockets(self, loop, *sockets):
        """Unschedule callback for a raw socket"""
        for socket in sockets:
            loop.remove_handler(socket)


class Socket(_AsyncTornado, _AsyncSocket):
    _poller_class = Poller


Poller._socket_class = Socket


class Context(_zmq.Context[Socket]):
    # avoid sharing instance with base Context class
    _instance = None

    io_loop = None

    @staticmethod
    def _socket_class(self, socket_type):
        return Socket(self, socket_type)

    def __init__(self: "Context", *args: Any, **kwargs: Any) -> None:
        io_loop = kwargs.pop('io_loop', None)
        if io_loop is not None:
            warnings.warn(
                f"{self.__class__.__name__}(io_loop) argument is deprecated in pyzmq 22.2."
                " The currently active loop will always be used.",
                DeprecationWarning,
                stacklevel=2,
            )
        super().__init__(*args, **kwargs)
