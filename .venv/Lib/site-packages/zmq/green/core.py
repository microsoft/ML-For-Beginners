# -----------------------------------------------------------------------------
#  Copyright (C) 2011-2012 Travis Cline
#
#  This file is part of pyzmq
#  It is adapted from upstream project zeromq_gevent under the New BSD License
#
#  Distributed under the terms of the New BSD License.  The full license is in
#  the file LICENSE.BSD, distributed as part of this software.
# -----------------------------------------------------------------------------

"""This module wraps the :class:`Socket` and :class:`Context` found in :mod:`pyzmq <zmq>` to be non blocking
"""


import sys
import time
import warnings
from typing import Tuple

import gevent
from gevent.event import AsyncResult
from gevent.hub import get_hub

import zmq
from zmq import Context as _original_Context
from zmq import Socket as _original_Socket

from .poll import _Poller

if hasattr(zmq, 'RCVTIMEO'):
    TIMEOS: Tuple = (zmq.RCVTIMEO, zmq.SNDTIMEO)
else:
    TIMEOS = ()


def _stop(evt):
    """simple wrapper for stopping an Event, allowing for method rename in gevent 1.0"""
    try:
        evt.stop()
    except AttributeError:
        # gevent<1.0 compat
        evt.cancel()


class _Socket(_original_Socket):
    """Green version of :class:`zmq.Socket`

    The following methods are overridden:

        * send
        * recv

    To ensure that the ``zmq.NOBLOCK`` flag is set and that sending or receiving
    is deferred to the hub if a ``zmq.EAGAIN`` (retry) error is raised.

    The `__state_changed` method is triggered when the zmq.FD for the socket is
    marked as readable and triggers the necessary read and write events (which
    are waited for in the recv and send methods).

    Some double underscore prefixes are used to minimize pollution of
    :class:`zmq.Socket`'s namespace.
    """

    __in_send_multipart = False
    __in_recv_multipart = False
    __writable = None
    __readable = None
    _state_event = None
    _gevent_bug_timeout = 11.6  # timeout for not trusting gevent
    _debug_gevent = False  # turn on if you think gevent is missing events
    _poller_class = _Poller
    _repr_cls = "zmq.green.Socket"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.__in_send_multipart = False
        self.__in_recv_multipart = False
        self.__setup_events()

    def __del__(self):
        self.close()

    def close(self, linger=None):
        super().close(linger)
        self.__cleanup_events()

    def __cleanup_events(self):
        # close the _state_event event, keeps the number of active file descriptors down
        if getattr(self, '_state_event', None):
            _stop(self._state_event)
            self._state_event = None
        # if the socket has entered a close state resume any waiting greenlets
        self.__writable.set()
        self.__readable.set()

    def __setup_events(self):
        self.__readable = AsyncResult()
        self.__writable = AsyncResult()
        self.__readable.set()
        self.__writable.set()

        try:
            self._state_event = get_hub().loop.io(
                self.getsockopt(zmq.FD), 1
            )  # read state watcher
            self._state_event.start(self.__state_changed)
        except AttributeError:
            # for gevent<1.0 compatibility
            from gevent.core import read_event

            self._state_event = read_event(
                self.getsockopt(zmq.FD), self.__state_changed, persist=True
            )

    def __state_changed(self, event=None, _evtype=None):
        if self.closed:
            self.__cleanup_events()
            return
        try:
            # avoid triggering __state_changed from inside __state_changed
            events = super().getsockopt(zmq.EVENTS)
        except zmq.ZMQError as exc:
            self.__writable.set_exception(exc)
            self.__readable.set_exception(exc)
        else:
            if events & zmq.POLLOUT:
                self.__writable.set()
            if events & zmq.POLLIN:
                self.__readable.set()

    def _wait_write(self):
        assert self.__writable.ready(), "Only one greenlet can be waiting on this event"
        self.__writable = AsyncResult()
        # timeout is because libzmq cannot be trusted to properly signal a new send event:
        # this is effectively a maximum poll interval of 1s
        tic = time.time()
        dt = self._gevent_bug_timeout
        if dt:
            timeout = gevent.Timeout(seconds=dt)
        else:
            timeout = None
        try:
            if timeout:
                timeout.start()
            self.__writable.get(block=True)
        except gevent.Timeout as t:
            if t is not timeout:
                raise
            toc = time.time()
            # gevent bug: get can raise timeout even on clean return
            # don't display zmq bug warning for gevent bug (this is getting ridiculous)
            if (
                self._debug_gevent
                and timeout
                and toc - tic > dt
                and self.getsockopt(zmq.EVENTS) & zmq.POLLOUT
            ):
                print(
                    "BUG: gevent may have missed a libzmq send event on %i!" % self.FD,
                    file=sys.stderr,
                )
        finally:
            if timeout:
                timeout.close()
            self.__writable.set()

    def _wait_read(self):
        assert self.__readable.ready(), "Only one greenlet can be waiting on this event"
        self.__readable = AsyncResult()
        # timeout is because libzmq cannot always be trusted to play nice with libevent.
        # I can only confirm that this actually happens for send, but lets be symmetrical
        # with our dirty hacks.
        # this is effectively a maximum poll interval of 1s
        tic = time.time()
        dt = self._gevent_bug_timeout
        if dt:
            timeout = gevent.Timeout(seconds=dt)
        else:
            timeout = None
        try:
            if timeout:
                timeout.start()
            self.__readable.get(block=True)
        except gevent.Timeout as t:
            if t is not timeout:
                raise
            toc = time.time()
            # gevent bug: get can raise timeout even on clean return
            # don't display zmq bug warning for gevent bug (this is getting ridiculous)
            if (
                self._debug_gevent
                and timeout
                and toc - tic > dt
                and self.getsockopt(zmq.EVENTS) & zmq.POLLIN
            ):
                print(
                    "BUG: gevent may have missed a libzmq recv event on %i!" % self.FD,
                    file=sys.stderr,
                )
        finally:
            if timeout:
                timeout.close()
            self.__readable.set()

    def send(self, data, flags=0, copy=True, track=False, **kwargs):
        """send, which will only block current greenlet

        state_changed always fires exactly once (success or fail) at the
        end of this method.
        """

        # if we're given the NOBLOCK flag act as normal and let the EAGAIN get raised
        if flags & zmq.NOBLOCK:
            try:
                msg = super().send(data, flags, copy, track, **kwargs)
            finally:
                if not self.__in_send_multipart:
                    self.__state_changed()
            return msg
        # ensure the zmq.NOBLOCK flag is part of flags
        flags |= zmq.NOBLOCK
        while (
            True
        ):  # Attempt to complete this operation indefinitely, blocking the current greenlet
            try:
                # attempt the actual call
                msg = super().send(data, flags, copy, track)
            except zmq.ZMQError as e:
                # if the raised ZMQError is not EAGAIN, reraise
                if e.errno != zmq.EAGAIN:
                    if not self.__in_send_multipart:
                        self.__state_changed()
                    raise
            else:
                if not self.__in_send_multipart:
                    self.__state_changed()
                return msg
            # defer to the event loop until we're notified the socket is writable
            self._wait_write()

    def recv(self, flags=0, copy=True, track=False):
        """recv, which will only block current greenlet

        state_changed always fires exactly once (success or fail) at the
        end of this method.
        """
        if flags & zmq.NOBLOCK:
            try:
                msg = super().recv(flags, copy, track)
            finally:
                if not self.__in_recv_multipart:
                    self.__state_changed()
            return msg

        flags |= zmq.NOBLOCK
        while True:
            try:
                msg = super().recv(flags, copy, track)
            except zmq.ZMQError as e:
                if e.errno != zmq.EAGAIN:
                    if not self.__in_recv_multipart:
                        self.__state_changed()
                    raise
            else:
                if not self.__in_recv_multipart:
                    self.__state_changed()
                return msg
            self._wait_read()

    def send_multipart(self, *args, **kwargs):
        """wrap send_multipart to prevent state_changed on each partial send"""
        self.__in_send_multipart = True
        try:
            msg = super().send_multipart(*args, **kwargs)
        finally:
            self.__in_send_multipart = False
            self.__state_changed()
        return msg

    def recv_multipart(self, *args, **kwargs):
        """wrap recv_multipart to prevent state_changed on each partial recv"""
        self.__in_recv_multipart = True
        try:
            msg = super().recv_multipart(*args, **kwargs)
        finally:
            self.__in_recv_multipart = False
            self.__state_changed()
        return msg

    def get(self, opt):
        """trigger state_changed on getsockopt(EVENTS)"""
        if opt in TIMEOS:
            warnings.warn(
                "TIMEO socket options have no effect in zmq.green", UserWarning
            )
        optval = super().get(opt)
        if opt == zmq.EVENTS:
            self.__state_changed()
        return optval

    def set(self, opt, val):
        """set socket option"""
        if opt in TIMEOS:
            warnings.warn(
                "TIMEO socket options have no effect in zmq.green", UserWarning
            )
        return super().set(opt, val)


class _Context(_original_Context[_Socket]):
    """Replacement for :class:`zmq.Context`

    Ensures that the greened Socket above is used in calls to `socket`.
    """

    _socket_class = _Socket
    _repr_cls = "zmq.green.Context"

    # avoid sharing instance with base Context class
    _instance = None
