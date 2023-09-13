"""AsyncIO support for zmq

Requires asyncio and Python 3.
"""

# Copyright (c) PyZMQ Developers.
# Distributed under the terms of the Modified BSD License.

import asyncio
import selectors
import sys
import warnings
from asyncio import Future, SelectorEventLoop
from weakref import WeakKeyDictionary

import zmq as _zmq
from zmq import _future

# registry of asyncio loop : selector thread
_selectors: WeakKeyDictionary = WeakKeyDictionary()


class ProactorSelectorThreadWarning(RuntimeWarning):
    """Warning class for notifying about the extra thread spawned by tornado

    We automatically support proactor via tornado's AddThreadSelectorEventLoop"""


def _get_selector_windows(
    asyncio_loop,
) -> asyncio.AbstractEventLoop:
    """Get selector-compatible loop

    Returns an object with ``add_reader`` family of methods,
    either the loop itself or a SelectorThread instance.

    Workaround Windows proactor removal of
    *reader methods, which we need for zmq sockets.
    """

    if asyncio_loop in _selectors:
        return _selectors[asyncio_loop]

    # detect add_reader instead of checking for proactor?
    if hasattr(asyncio, "ProactorEventLoop") and isinstance(
        asyncio_loop, asyncio.ProactorEventLoop  # type: ignore
    ):
        try:
            from tornado.platform.asyncio import AddThreadSelectorEventLoop
        except ImportError:
            raise RuntimeError(
                "Proactor event loop does not implement add_reader family of methods required for zmq."
                " zmq will work with proactor if tornado >= 6.1 can be found."
                " Use `asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())`"
                " or install 'tornado>=6.1' to avoid this error."
            )

        warnings.warn(
            "Proactor event loop does not implement add_reader family of methods required for zmq."
            " Registering an additional selector thread for add_reader support via tornado."
            " Use `asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())`"
            " to avoid this warning.",
            RuntimeWarning,
            # stacklevel 5 matches most likely zmq.asyncio.Context().socket()
            stacklevel=5,
        )

        selector_loop = _selectors[asyncio_loop] = AddThreadSelectorEventLoop(asyncio_loop)  # type: ignore

        # patch loop.close to also close the selector thread
        loop_close = asyncio_loop.close

        def _close_selector_and_loop():
            # restore original before calling selector.close,
            # which in turn calls eventloop.close!
            asyncio_loop.close = loop_close
            _selectors.pop(asyncio_loop, None)
            selector_loop.close()

        asyncio_loop.close = _close_selector_and_loop  # type: ignore # mypy bug - assign a function to method
        return selector_loop
    else:
        return asyncio_loop


def _get_selector_noop(loop) -> asyncio.AbstractEventLoop:
    """no-op on non-Windows"""
    return loop


if sys.platform == "win32":
    _get_selector = _get_selector_windows
else:
    _get_selector = _get_selector_noop


class _AsyncIO:
    _Future = Future
    _WRITE = selectors.EVENT_WRITE
    _READ = selectors.EVENT_READ

    def _default_loop(self):
        if sys.version_info >= (3, 7):
            try:
                return asyncio.get_running_loop()
            except RuntimeError:
                warnings.warn(
                    "No running event loop. zmq.asyncio should be used from within an asyncio loop.",
                    RuntimeWarning,
                    stacklevel=4,
                )
        # get_event_loop deprecated in 3.10:
        return asyncio.get_event_loop()


class Poller(_AsyncIO, _future._AsyncPoller):
    """Poller returning asyncio.Future for poll results."""

    def _watch_raw_socket(self, loop, socket, evt, f):
        """Schedule callback for a raw socket"""
        selector = _get_selector(loop)
        if evt & self._READ:
            selector.add_reader(socket, lambda *args: f())
        if evt & self._WRITE:
            selector.add_writer(socket, lambda *args: f())

    def _unwatch_raw_sockets(self, loop, *sockets):
        """Unschedule callback for a raw socket"""
        selector = _get_selector(loop)
        for socket in sockets:
            selector.remove_reader(socket)
            selector.remove_writer(socket)


class Socket(_AsyncIO, _future._AsyncSocket):
    """Socket returning asyncio Futures for send/recv/poll methods."""

    _poller_class = Poller

    def _get_selector(self, io_loop=None):
        if io_loop is None:
            io_loop = self._get_loop()
        return _get_selector(io_loop)

    def _init_io_state(self, io_loop=None):
        """initialize the ioloop event handler"""
        self._get_selector(io_loop).add_reader(
            self._fd, lambda: self._handle_events(0, 0)
        )

    def _clear_io_state(self):
        """clear any ioloop event handler

        called once at close
        """
        loop = self._current_loop
        if loop and not loop.is_closed() and self._fd != -1:
            self._get_selector(loop).remove_reader(self._fd)


Poller._socket_class = Socket


class Context(_zmq.Context[Socket]):
    """Context for creating asyncio-compatible Sockets"""

    _socket_class = Socket

    # avoid sharing instance with base Context class
    _instance = None


class ZMQEventLoop(SelectorEventLoop):
    """DEPRECATED: AsyncIO eventloop using zmq_poll.

    pyzmq sockets should work with any asyncio event loop as of pyzmq 17.
    """

    def __init__(self, selector=None):
        _deprecated()
        return super().__init__(selector)


_loop = None


def _deprecated():
    if _deprecated.called:  # type: ignore
        return
    _deprecated.called = True  # type: ignore

    warnings.warn(
        "ZMQEventLoop and zmq.asyncio.install are deprecated in pyzmq 17. Special eventloop integration is no longer needed.",
        DeprecationWarning,
        stacklevel=3,
    )


_deprecated.called = False  # type: ignore


def install():
    """DEPRECATED: No longer needed in pyzmq 17"""
    _deprecated()


__all__ = [
    "Context",
    "Socket",
    "Poller",
    "ZMQEventLoop",
    "install",
]
