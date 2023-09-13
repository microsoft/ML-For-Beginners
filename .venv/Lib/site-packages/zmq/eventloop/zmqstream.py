#
# Copyright 2009 Facebook
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

"""A utility class for event-based messaging on a zmq socket using tornado.

.. seealso::

    - :mod:`zmq.asyncio`
    - :mod:`zmq.eventloop.future`
"""

import asyncio
import pickle
import warnings
from queue import Queue
from typing import (
    Any,
    Awaitable,
    Callable,
    List,
    Optional,
    Sequence,
    Union,
    cast,
    overload,
)

from tornado.ioloop import IOLoop
from tornado.log import gen_log

import zmq
import zmq._future
from zmq import POLLIN, POLLOUT
from zmq._typing import Literal
from zmq.utils import jsonapi


class ZMQStream:
    """A utility class to register callbacks when a zmq socket sends and receives

    For use with tornado IOLoop.

    There are three main methods

    Methods:

    * **on_recv(callback, copy=True):**
        register a callback to be run every time the socket has something to receive
    * **on_send(callback):**
        register a callback to be run every time you call send
    * **send_multipart(self, msg, flags=0, copy=False, callback=None):**
        perform a send that will trigger the callback
        if callback is passed, on_send is also called.

        There are also send_multipart(), send_json(), send_pyobj()

    Three other methods for deactivating the callbacks:

    * **stop_on_recv():**
        turn off the recv callback
    * **stop_on_send():**
        turn off the send callback

    which simply call ``on_<evt>(None)``.

    The entire socket interface, excluding direct recv methods, is also
    provided, primarily through direct-linking the methods.
    e.g.

    >>> stream.bind is stream.socket.bind
    True


    .. versionadded:: 25

        send/recv callbacks can be coroutines.

    .. versionchanged:: 25

        ZMQStreams only support base zmq.Socket classes (this has always been true, but not enforced).
        If ZMQStreams are created with e.g. async Socket subclasses,
        a RuntimeWarning will be shown,
        and the socket cast back to the default zmq.Socket
        before connecting events.

        Previously, using async sockets (or any zmq.Socket subclass) would result in undefined behavior for the
        arguments passed to callback functions.
        Now, the callback functions reliably get the return value of the base `zmq.Socket` send/recv_multipart methods
        (the list of message frames).
    """

    socket: zmq.Socket
    io_loop: IOLoop
    poller: zmq.Poller
    _send_queue: Queue
    _recv_callback: Optional[Callable]
    _send_callback: Optional[Callable]
    _close_callback: Optional[Callable]
    _state: int = 0
    _flushed: bool = False
    _recv_copy: bool = False
    _fd: int

    def __init__(self, socket: "zmq.Socket", io_loop: Optional[IOLoop] = None):
        if isinstance(socket, zmq._future._AsyncSocket):
            warnings.warn(
                f"""ZMQStream only supports the base zmq.Socket class.

                Use zmq.Socket(shadow=other_socket)
                or `ctx.socket(zmq.{socket._type_name}, socket_class=zmq.Socket)`
                to create a base zmq.Socket object,
                no matter what other kind of socket your Context creates.
                """,
                RuntimeWarning,
                stacklevel=2,
            )
            # shadow back to base zmq.Socket,
            # otherwise callbacks like `on_recv` will get the wrong types.
            socket = zmq.Socket(shadow=socket)
        self.socket = socket

        # IOLoop.current() is deprecated if called outside the event loop
        # that means
        self.io_loop = io_loop or IOLoop.current()
        self.poller = zmq.Poller()
        self._fd = cast(int, self.socket.FD)

        self._send_queue = Queue()
        self._recv_callback = None
        self._send_callback = None
        self._close_callback = None
        self._recv_copy = False
        self._flushed = False

        self._state = 0
        self._init_io_state()

        # shortcircuit some socket methods
        self.bind = self.socket.bind
        self.bind_to_random_port = self.socket.bind_to_random_port
        self.connect = self.socket.connect
        self.setsockopt = self.socket.setsockopt
        self.getsockopt = self.socket.getsockopt
        self.setsockopt_string = self.socket.setsockopt_string
        self.getsockopt_string = self.socket.getsockopt_string
        self.setsockopt_unicode = self.socket.setsockopt_unicode
        self.getsockopt_unicode = self.socket.getsockopt_unicode

    def stop_on_recv(self):
        """Disable callback and automatic receiving."""
        return self.on_recv(None)

    def stop_on_send(self):
        """Disable callback on sending."""
        return self.on_send(None)

    def stop_on_err(self):
        """DEPRECATED, does nothing"""
        gen_log.warn("on_err does nothing, and will be removed")

    def on_err(self, callback: Callable):
        """DEPRECATED, does nothing"""
        gen_log.warn("on_err does nothing, and will be removed")

    @overload
    def on_recv(
        self,
        callback: Callable[[List[bytes]], Any],
    ) -> None:
        ...

    @overload
    def on_recv(
        self,
        callback: Callable[[List[bytes]], Any],
        copy: Literal[True],
    ) -> None:
        ...

    @overload
    def on_recv(
        self,
        callback: Callable[[List[zmq.Frame]], Any],
        copy: Literal[False],
    ) -> None:
        ...

    @overload
    def on_recv(
        self,
        callback: Union[
            Callable[[List[zmq.Frame]], Any],
            Callable[[List[bytes]], Any],
        ],
        copy: bool = ...,
    ):
        ...

    def on_recv(
        self,
        callback: Union[
            Callable[[List[zmq.Frame]], Any],
            Callable[[List[bytes]], Any],
        ],
        copy: bool = True,
    ) -> None:
        """Register a callback for when a message is ready to recv.

        There can be only one callback registered at a time, so each
        call to `on_recv` replaces previously registered callbacks.

        on_recv(None) disables recv event polling.

        Use on_recv_stream(callback) instead, to register a callback that will receive
        both this ZMQStream and the message, instead of just the message.

        Parameters
        ----------

        callback : callable
            callback must take exactly one argument, which will be a
            list, as returned by socket.recv_multipart()
            if callback is None, recv callbacks are disabled.
        copy : bool
            copy is passed directly to recv, so if copy is False,
            callback will receive Message objects. If copy is True,
            then callback will receive bytes/str objects.

        Returns : None
        """

        self._check_closed()
        assert callback is None or callable(callback)
        self._recv_callback = callback
        self._recv_copy = copy
        if callback is None:
            self._drop_io_state(zmq.POLLIN)
        else:
            self._add_io_state(zmq.POLLIN)

    @overload
    def on_recv_stream(
        self,
        callback: Callable[["ZMQStream", List[bytes]], Any],
    ) -> None:
        ...

    @overload
    def on_recv_stream(
        self,
        callback: Callable[["ZMQStream", List[bytes]], Any],
        copy: Literal[True],
    ) -> None:
        ...

    @overload
    def on_recv_stream(
        self,
        callback: Callable[["ZMQStream", List[zmq.Frame]], Any],
        copy: Literal[False],
    ) -> None:
        ...

    @overload
    def on_recv_stream(
        self,
        callback: Union[
            Callable[["ZMQStream", List[zmq.Frame]], Any],
            Callable[["ZMQStream", List[bytes]], Any],
        ],
        copy: bool = ...,
    ):
        ...

    def on_recv_stream(
        self,
        callback: Union[
            Callable[["ZMQStream", List[zmq.Frame]], Any],
            Callable[["ZMQStream", List[bytes]], Any],
        ],
        copy: bool = True,
    ):
        """Same as on_recv, but callback will get this stream as first argument

        callback must take exactly two arguments, as it will be called as::

            callback(stream, msg)

        Useful when a single callback should be used with multiple streams.
        """
        if callback is None:
            self.stop_on_recv()
        else:

            def stream_callback(msg):
                return callback(self, msg)

            self.on_recv(stream_callback, copy=copy)

    def on_send(
        self, callback: Callable[[Sequence[Any], Optional[zmq.MessageTracker]], Any]
    ):
        """Register a callback to be called on each send

        There will be two arguments::

            callback(msg, status)

        * `msg` will be the list of sendable objects that was just sent
        * `status` will be the return result of socket.send_multipart(msg) -
          MessageTracker or None.

        Non-copying sends return a MessageTracker object whose
        `done` attribute will be True when the send is complete.
        This allows users to track when an object is safe to write to
        again.

        The second argument will always be None if copy=True
        on the send.

        Use on_send_stream(callback) to register a callback that will be passed
        this ZMQStream as the first argument, in addition to the other two.

        on_send(None) disables recv event polling.

        Parameters
        ----------

        callback : callable
            callback must take exactly two arguments, which will be
            the message being sent (always a list),
            and the return result of socket.send_multipart(msg) -
            MessageTracker or None.

            if callback is None, send callbacks are disabled.
        """

        self._check_closed()
        assert callback is None or callable(callback)
        self._send_callback = callback

    def on_send_stream(
        self,
        callback: Callable[
            ["ZMQStream", Sequence[Any], Optional[zmq.MessageTracker]], Any
        ],
    ):
        """Same as on_send, but callback will get this stream as first argument

        Callback will be passed three arguments::

            callback(stream, msg, status)

        Useful when a single callback should be used with multiple streams.
        """
        if callback is None:
            self.stop_on_send()
        else:
            self.on_send(lambda msg, status: callback(self, msg, status))

    def send(self, msg, flags=0, copy=True, track=False, callback=None, **kwargs):
        """Send a message, optionally also register a new callback for sends.
        See zmq.socket.send for details.
        """
        return self.send_multipart(
            [msg], flags=flags, copy=copy, track=track, callback=callback, **kwargs
        )

    def send_multipart(
        self,
        msg: Sequence[Any],
        flags: int = 0,
        copy: bool = True,
        track: bool = False,
        callback: Optional[Callable] = None,
        **kwargs: Any,
    ) -> None:
        """Send a multipart message, optionally also register a new callback for sends.
        See zmq.socket.send_multipart for details.
        """
        kwargs.update(dict(flags=flags, copy=copy, track=track))
        self._send_queue.put((msg, kwargs))
        callback = callback or self._send_callback
        if callback is not None:
            self.on_send(callback)
        else:
            # noop callback
            self.on_send(lambda *args: None)
        self._add_io_state(zmq.POLLOUT)

    def send_string(
        self,
        u: str,
        flags: int = 0,
        encoding: str = 'utf-8',
        callback: Optional[Callable] = None,
        **kwargs: Any,
    ):
        """Send a unicode message with an encoding.
        See zmq.socket.send_unicode for details.
        """
        if not isinstance(u, str):
            raise TypeError("unicode/str objects only")
        return self.send(u.encode(encoding), flags=flags, callback=callback, **kwargs)

    send_unicode = send_string

    def send_json(
        self,
        obj: Any,
        flags: int = 0,
        callback: Optional[Callable] = None,
        **kwargs: Any,
    ):
        """Send json-serialized version of an object.
        See zmq.socket.send_json for details.
        """
        msg = jsonapi.dumps(obj)
        return self.send(msg, flags=flags, callback=callback, **kwargs)

    def send_pyobj(
        self,
        obj: Any,
        flags: int = 0,
        protocol: int = -1,
        callback: Optional[Callable] = None,
        **kwargs: Any,
    ):
        """Send a Python object as a message using pickle to serialize.

        See zmq.socket.send_json for details.
        """
        msg = pickle.dumps(obj, protocol)
        return self.send(msg, flags, callback=callback, **kwargs)

    def _finish_flush(self):
        """callback for unsetting _flushed flag."""
        self._flushed = False

    def flush(self, flag: int = zmq.POLLIN | zmq.POLLOUT, limit: Optional[int] = None):
        """Flush pending messages.

        This method safely handles all pending incoming and/or outgoing messages,
        bypassing the inner loop, passing them to the registered callbacks.

        A limit can be specified, to prevent blocking under high load.

        flush will return the first time ANY of these conditions are met:
            * No more events matching the flag are pending.
            * the total number of events handled reaches the limit.

        Note that if ``flag|POLLIN != 0``, recv events will be flushed even if no callback
        is registered, unlike normal IOLoop operation. This allows flush to be
        used to remove *and ignore* incoming messages.

        Parameters
        ----------
        flag : int, default=POLLIN|POLLOUT
                0MQ poll flags.
                If flag|POLLIN,  recv events will be flushed.
                If flag|POLLOUT, send events will be flushed.
                Both flags can be set at once, which is the default.
        limit : None or int, optional
                The maximum number of messages to send or receive.
                Both send and recv count against this limit.

        Returns
        -------
        int : count of events handled (both send and recv)
        """
        self._check_closed()
        # unset self._flushed, so callbacks will execute, in case flush has
        # already been called this iteration
        already_flushed = self._flushed
        self._flushed = False
        # initialize counters
        count = 0

        def update_flag():
            """Update the poll flag, to prevent registering POLLOUT events
            if we don't have pending sends."""
            return flag & zmq.POLLIN | (self.sending() and flag & zmq.POLLOUT)

        flag = update_flag()
        if not flag:
            # nothing to do
            return 0
        self.poller.register(self.socket, flag)
        events = self.poller.poll(0)
        while events and (not limit or count < limit):
            s, event = events[0]
            if event & POLLIN:  # receiving
                self._handle_recv()
                count += 1
                if self.socket is None:
                    # break if socket was closed during callback
                    break
            if event & POLLOUT and self.sending():
                self._handle_send()
                count += 1
                if self.socket is None:
                    # break if socket was closed during callback
                    break

            flag = update_flag()
            if flag:
                self.poller.register(self.socket, flag)
                events = self.poller.poll(0)
            else:
                events = []
        if count:  # only bypass loop if we actually flushed something
            # skip send/recv callbacks this iteration
            self._flushed = True
            # reregister them at the end of the loop
            if not already_flushed:  # don't need to do it again
                self.io_loop.add_callback(self._finish_flush)
        elif already_flushed:
            self._flushed = True

        # update ioloop poll state, which may have changed
        self._rebuild_io_state()
        return count

    def set_close_callback(self, callback: Optional[Callable]):
        """Call the given callback when the stream is closed."""
        self._close_callback = callback

    def close(self, linger: Optional[int] = None) -> None:
        """Close this stream."""
        if self.socket is not None:
            if self.socket.closed:
                # fallback on raw fd for closed sockets
                # hopefully this happened promptly after close,
                # otherwise somebody else may have the FD
                warnings.warn(
                    "Unregistering FD %s after closing socket. "
                    "This could result in unregistering handlers for the wrong socket. "
                    "Please use stream.close() instead of closing the socket directly."
                    % self._fd,
                    stacklevel=2,
                )
                self.io_loop.remove_handler(self._fd)
            else:
                self.io_loop.remove_handler(self.socket)
                self.socket.close(linger)
            self.socket = None  # type: ignore
            if self._close_callback:
                self._run_callback(self._close_callback)

    def receiving(self) -> bool:
        """Returns True if we are currently receiving from the stream."""
        return self._recv_callback is not None

    def sending(self) -> bool:
        """Returns True if we are currently sending to the stream."""
        return not self._send_queue.empty()

    def closed(self) -> bool:
        if self.socket is None:
            return True
        if self.socket.closed:
            # underlying socket has been closed, but not by us!
            # trigger our cleanup
            self.close()
            return True
        return False

    def _run_callback(self, callback, *args, **kwargs):
        """Wrap running callbacks in try/except to allow us to
        close our socket."""
        try:
            f = callback(*args, **kwargs)
            if isinstance(f, Awaitable):
                f = asyncio.ensure_future(f)
            else:
                f = None
        except Exception:
            gen_log.error("Uncaught exception in ZMQStream callback", exc_info=True)
            # Re-raise the exception so that IOLoop.handle_callback_exception
            # can see it and log the error
            raise

        if f is not None:
            # handle async callbacks
            def _log_error(f):
                try:
                    f.result()
                except Exception:
                    gen_log.error(
                        "Uncaught exception in ZMQStream callback", exc_info=True
                    )

            f.add_done_callback(_log_error)

    def _handle_events(self, fd, events):
        """This method is the actual handler for IOLoop, that gets called whenever
        an event on my socket is posted. It dispatches to _handle_recv, etc."""
        if not self.socket:
            gen_log.warning("Got events for closed stream %s", self)
            return
        try:
            zmq_events = self.socket.EVENTS
        except zmq.ContextTerminated:
            gen_log.warning("Got events for stream %s after terminating context", self)
            # trigger close check, this will unregister callbacks
            self.closed()
            return
        except zmq.ZMQError as e:
            # run close check
            # shadow sockets may have been closed elsewhere,
            # which should show up as ENOTSOCK here
            if self.closed():
                gen_log.warning(
                    "Got events for stream %s attached to closed socket: %s", self, e
                )
            else:
                gen_log.error("Error getting events for %s: %s", self, e)
            return
        try:
            # dispatch events:
            if zmq_events & zmq.POLLIN and self.receiving():
                self._handle_recv()
                if not self.socket:
                    return
            if zmq_events & zmq.POLLOUT and self.sending():
                self._handle_send()
                if not self.socket:
                    return

            # rebuild the poll state
            self._rebuild_io_state()
        except Exception:
            gen_log.error("Uncaught exception in zmqstream callback", exc_info=True)
            raise

    def _handle_recv(self):
        """Handle a recv event."""
        if self._flushed:
            return
        try:
            msg = self.socket.recv_multipart(zmq.NOBLOCK, copy=self._recv_copy)
        except zmq.ZMQError as e:
            if e.errno == zmq.EAGAIN:
                # state changed since poll event
                pass
            else:
                raise
        else:
            if self._recv_callback:
                callback = self._recv_callback
                self._run_callback(callback, msg)

    def _handle_send(self):
        """Handle a send event."""
        if self._flushed:
            return
        if not self.sending():
            gen_log.error("Shouldn't have handled a send event")
            return

        msg, kwargs = self._send_queue.get()
        try:
            status = self.socket.send_multipart(msg, **kwargs)
        except zmq.ZMQError as e:
            gen_log.error("SEND Error: %s", e)
            status = e
        if self._send_callback:
            callback = self._send_callback
            self._run_callback(callback, msg, status)

    def _check_closed(self):
        if not self.socket:
            raise OSError("Stream is closed")

    def _rebuild_io_state(self):
        """rebuild io state based on self.sending() and receiving()"""
        if self.socket is None:
            return
        state = 0
        if self.receiving():
            state |= zmq.POLLIN
        if self.sending():
            state |= zmq.POLLOUT

        self._state = state
        self._update_handler(state)

    def _add_io_state(self, state):
        """Add io_state to poller."""
        self._state = self._state | state
        self._update_handler(self._state)

    def _drop_io_state(self, state):
        """Stop poller from watching an io_state."""
        self._state = self._state & (~state)
        self._update_handler(self._state)

    def _update_handler(self, state):
        """Update IOLoop handler with state."""
        if self.socket is None:
            return

        if state & self.socket.events:
            # events still exist that haven't been processed
            # explicitly schedule handling to avoid missing events due to edge-triggered FDs
            self.io_loop.add_callback(lambda: self._handle_events(self.socket, 0))

    def _init_io_state(self):
        """initialize the ioloop event handler"""
        self.io_loop.add_handler(self.socket, self._handle_events, self.io_loop.READ)
