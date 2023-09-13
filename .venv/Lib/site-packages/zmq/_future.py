"""Future-returning APIs for coroutines."""

# Copyright (c) PyZMQ Developers.
# Distributed under the terms of the Modified BSD License.

import warnings
from asyncio import Future
from collections import deque
from itertools import chain
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)

import zmq as _zmq
from zmq import EVENTS, POLLIN, POLLOUT
from zmq._typing import Literal


class _FutureEvent(NamedTuple):
    future: Future
    kind: str
    kwargs: Dict
    msg: Any
    timer: Any


# These are incomplete classes and need a Mixin for compatibility with an eventloop
# defining the following attributes:
#
# _Future
# _READ
# _WRITE
# _default_loop()


class _Async:
    """Mixin for common async logic"""

    _current_loop: Any = None
    _Future: Type[Future]

    def _get_loop(self) -> Any:
        """Get event loop

        Notice if event loop has changed,
        and register init_io_state on activation of a new event loop
        """
        if self._current_loop is None:
            self._current_loop = self._default_loop()
            self._init_io_state(self._current_loop)
            return self._current_loop
        current_loop = self._default_loop()
        if current_loop is not self._current_loop:
            # warn? This means a socket is being used in multiple loops!
            self._current_loop = current_loop
            self._init_io_state(current_loop)
        return current_loop

    def _default_loop(self) -> Any:
        raise NotImplementedError("Must be implemented in a subclass")

    def _init_io_state(self, loop=None) -> None:
        pass


class _AsyncPoller(_Async, _zmq.Poller):
    """Poller that returns a Future on poll, instead of blocking."""

    _socket_class: Type["_AsyncSocket"]
    _READ: int
    _WRITE: int
    raw_sockets: List[Any]

    def _watch_raw_socket(self, loop: Any, socket: Any, evt: int, f: Callable) -> None:
        """Schedule callback for a raw socket"""
        raise NotImplementedError()

    def _unwatch_raw_sockets(self, loop: Any, *sockets: Any) -> None:
        """Unschedule callback for a raw socket"""
        raise NotImplementedError()

    def poll(self, timeout=-1) -> Awaitable[List[Tuple[Any, int]]]:  # type: ignore
        """Return a Future for a poll event"""
        future = self._Future()
        if timeout == 0:
            try:
                result = super().poll(0)
            except Exception as e:
                future.set_exception(e)
            else:
                future.set_result(result)
            return future

        loop = self._get_loop()

        # register Future to be called as soon as any event is available on any socket
        watcher = self._Future()

        # watch raw sockets:
        raw_sockets: List[Any] = []

        def wake_raw(*args):
            if not watcher.done():
                watcher.set_result(None)

        watcher.add_done_callback(
            lambda f: self._unwatch_raw_sockets(loop, *raw_sockets)
        )

        for socket, mask in self.sockets:
            if isinstance(socket, _zmq.Socket):
                if not isinstance(socket, self._socket_class):
                    # it's a blocking zmq.Socket, wrap it in async
                    socket = self._socket_class.from_socket(socket)
                if mask & _zmq.POLLIN:
                    socket._add_recv_event('poll', future=watcher)
                if mask & _zmq.POLLOUT:
                    socket._add_send_event('poll', future=watcher)
            else:
                raw_sockets.append(socket)
                evt = 0
                if mask & _zmq.POLLIN:
                    evt |= self._READ
                if mask & _zmq.POLLOUT:
                    evt |= self._WRITE
                self._watch_raw_socket(loop, socket, evt, wake_raw)

        def on_poll_ready(f):
            if future.done():
                return
            if watcher.cancelled():
                try:
                    future.cancel()
                except RuntimeError:
                    # RuntimeError may be called during teardown
                    pass
                return
            if watcher.exception():
                future.set_exception(watcher.exception())
            else:
                try:
                    result = super(_AsyncPoller, self).poll(0)
                except Exception as e:
                    future.set_exception(e)
                else:
                    future.set_result(result)

        watcher.add_done_callback(on_poll_ready)

        if timeout is not None and timeout > 0:
            # schedule cancel to fire on poll timeout, if any
            def trigger_timeout():
                if not watcher.done():
                    watcher.set_result(None)

            timeout_handle = loop.call_later(1e-3 * timeout, trigger_timeout)

            def cancel_timeout(f):
                if hasattr(timeout_handle, 'cancel'):
                    timeout_handle.cancel()
                else:
                    loop.remove_timeout(timeout_handle)

            future.add_done_callback(cancel_timeout)

        def cancel_watcher(f):
            if not watcher.done():
                watcher.cancel()

        future.add_done_callback(cancel_watcher)

        return future


class _NoTimer:
    @staticmethod
    def cancel():
        pass


T = TypeVar("T", bound="_AsyncSocket")


class _AsyncSocket(_Async, _zmq.Socket[Future]):
    # Warning : these class variables are only here to allow to call super().__setattr__.
    # They be overridden at instance initialization and not shared in the whole class
    _recv_futures = None
    _send_futures = None
    _state = 0
    _shadow_sock: "_zmq.Socket"
    _poller_class = _AsyncPoller
    _fd = None

    def __init__(
        self,
        context=None,
        socket_type=-1,
        io_loop=None,
        _from_socket: Optional["_zmq.Socket"] = None,
        **kwargs,
    ) -> None:
        if isinstance(context, _zmq.Socket):
            context, _from_socket = (None, context)
        if _from_socket is not None:
            super().__init__(shadow=_from_socket.underlying)  # type: ignore
            self._shadow_sock = _from_socket
        else:
            super().__init__(context, socket_type, **kwargs)  # type: ignore
            self._shadow_sock = _zmq.Socket.shadow(self.underlying)

        if io_loop is not None:
            warnings.warn(
                f"{self.__class__.__name__}(io_loop) argument is deprecated in pyzmq 22.2."
                " The currently active loop will always be used.",
                DeprecationWarning,
                stacklevel=3,
            )
        self._recv_futures = deque()
        self._send_futures = deque()
        self._state = 0
        self._fd = self._shadow_sock.FD

    @classmethod
    def from_socket(cls: Type[T], socket: "_zmq.Socket", io_loop: Any = None) -> T:
        """Create an async socket from an existing Socket"""
        return cls(_from_socket=socket, io_loop=io_loop)

    def close(self, linger: Optional[int] = None) -> None:
        if not self.closed and self._fd is not None:
            event_list: List[_FutureEvent] = list(
                chain(self._recv_futures or [], self._send_futures or [])
            )
            for event in event_list:
                if not event.future.done():
                    try:
                        event.future.cancel()
                    except RuntimeError:
                        # RuntimeError may be called during teardown
                        pass
            self._clear_io_state()
        super().close(linger=linger)

    close.__doc__ = _zmq.Socket.close.__doc__

    def get(self, key):
        result = super().get(key)
        if key == EVENTS:
            self._schedule_remaining_events(result)
        return result

    get.__doc__ = _zmq.Socket.get.__doc__

    @overload  # type: ignore
    def recv_multipart(
        self, flags: int = 0, *, track: bool = False
    ) -> Awaitable[List[bytes]]:
        ...

    @overload
    def recv_multipart(
        self, flags: int = 0, *, copy: Literal[True], track: bool = False
    ) -> Awaitable[List[bytes]]:
        ...

    @overload
    def recv_multipart(
        self, flags: int = 0, *, copy: Literal[False], track: bool = False
    ) -> Awaitable[List[_zmq.Frame]]:  # type: ignore
        ...

    @overload
    def recv_multipart(
        self, flags: int = 0, copy: bool = True, track: bool = False
    ) -> Awaitable[Union[List[bytes], List[_zmq.Frame]]]:
        ...

    def recv_multipart(
        self, flags: int = 0, copy: bool = True, track: bool = False
    ) -> Awaitable[Union[List[bytes], List[_zmq.Frame]]]:
        """Receive a complete multipart zmq message.

        Returns a Future whose result will be a multipart message.
        """
        return self._add_recv_event(
            'recv_multipart', dict(flags=flags, copy=copy, track=track)
        )

    def recv(  # type: ignore
        self, flags: int = 0, copy: bool = True, track: bool = False
    ) -> Awaitable[Union[bytes, _zmq.Frame]]:
        """Receive a single zmq frame.

        Returns a Future, whose result will be the received frame.

        Recommend using recv_multipart instead.
        """
        return self._add_recv_event('recv', dict(flags=flags, copy=copy, track=track))

    def send_multipart(  # type: ignore
        self, msg_parts: Any, flags: int = 0, copy: bool = True, track=False, **kwargs
    ) -> Awaitable[Optional[_zmq.MessageTracker]]:
        """Send a complete multipart zmq message.

        Returns a Future that resolves when sending is complete.
        """
        kwargs['flags'] = flags
        kwargs['copy'] = copy
        kwargs['track'] = track
        return self._add_send_event('send_multipart', msg=msg_parts, kwargs=kwargs)

    def send(  # type: ignore
        self,
        data: Any,
        flags: int = 0,
        copy: bool = True,
        track: bool = False,
        **kwargs: Any,
    ) -> Awaitable[Optional[_zmq.MessageTracker]]:
        """Send a single zmq frame.

        Returns a Future that resolves when sending is complete.

        Recommend using send_multipart instead.
        """
        kwargs['flags'] = flags
        kwargs['copy'] = copy
        kwargs['track'] = track
        kwargs.update(dict(flags=flags, copy=copy, track=track))
        return self._add_send_event('send', msg=data, kwargs=kwargs)

    def _deserialize(self, recvd, load):
        """Deserialize with Futures"""
        f = self._Future()

        def _chain(_):
            """Chain result through serialization to recvd"""
            if f.done():
                return
            if recvd.exception():
                f.set_exception(recvd.exception())
            else:
                buf = recvd.result()
                try:
                    loaded = load(buf)
                except Exception as e:
                    f.set_exception(e)
                else:
                    f.set_result(loaded)

        recvd.add_done_callback(_chain)

        def _chain_cancel(_):
            """Chain cancellation from f to recvd"""
            if recvd.done():
                return
            if f.cancelled():
                recvd.cancel()

        f.add_done_callback(_chain_cancel)

        return f

    def poll(self, timeout=None, flags=_zmq.POLLIN) -> Awaitable[int]:  # type: ignore
        """poll the socket for events

        returns a Future for the poll results.
        """

        if self.closed:
            raise _zmq.ZMQError(_zmq.ENOTSUP)

        p = self._poller_class()
        p.register(self, flags)
        poll_future = cast(Future, p.poll(timeout))

        future = self._Future()

        def unwrap_result(f):
            if future.done():
                return
            if poll_future.cancelled():
                try:
                    future.cancel()
                except RuntimeError:
                    # RuntimeError may be called during teardown
                    pass
                return
            if f.exception():
                future.set_exception(poll_future.exception())
            else:
                evts = dict(poll_future.result())
                future.set_result(evts.get(self, 0))

        if poll_future.done():
            # hook up result if already done
            unwrap_result(poll_future)
        else:
            poll_future.add_done_callback(unwrap_result)

        def cancel_poll(future):
            """Cancel underlying poll if request has been cancelled"""
            if not poll_future.done():
                try:
                    poll_future.cancel()
                except RuntimeError:
                    # RuntimeError may be called during teardown
                    pass

        future.add_done_callback(cancel_poll)

        return future

    # overrides only necessary for updated types
    def recv_string(self, *args, **kwargs) -> Awaitable[str]:  # type: ignore
        return super().recv_string(*args, **kwargs)  # type: ignore

    def send_string(self, s: str, flags: int = 0, encoding: str = 'utf-8') -> Awaitable[None]:  # type: ignore
        return super().send_string(s, flags=flags, encoding=encoding)  # type: ignore

    def _add_timeout(self, future, timeout):
        """Add a timeout for a send or recv Future"""

        def future_timeout():
            if future.done():
                # future already resolved, do nothing
                return

            # raise EAGAIN
            future.set_exception(_zmq.Again())

        return self._call_later(timeout, future_timeout)

    def _call_later(self, delay, callback):
        """Schedule a function to be called later

        Override for different IOLoop implementations

        Tornado and asyncio happen to both have ioloop.call_later
        with the same signature.
        """
        return self._get_loop().call_later(delay, callback)

    @staticmethod
    def _remove_finished_future(future, event_list):
        """Make sure that futures are removed from the event list when they resolve

        Avoids delaying cleanup until the next send/recv event,
        which may never come.
        """
        for f_idx, event in enumerate(event_list):
            if event.future is future:
                break
        else:
            return

        # "future" instance is shared between sockets, but each socket has its own event list.
        event_list.remove(event_list[f_idx])

    def _add_recv_event(self, kind, kwargs=None, future=None):
        """Add a recv event, returning the corresponding Future"""
        f = future or self._Future()
        if kind.startswith('recv') and kwargs.get('flags', 0) & _zmq.DONTWAIT:
            # short-circuit non-blocking calls
            recv = getattr(self._shadow_sock, kind)
            try:
                r = recv(**kwargs)
            except Exception as e:
                f.set_exception(e)
            else:
                f.set_result(r)
            return f

        timer = _NoTimer
        if hasattr(_zmq, 'RCVTIMEO'):
            timeout_ms = self._shadow_sock.rcvtimeo
            if timeout_ms >= 0:
                timer = self._add_timeout(f, timeout_ms * 1e-3)

        # we add it to the list of futures before we add the timeout as the
        # timeout will remove the future from recv_futures to avoid leaks
        self._recv_futures.append(_FutureEvent(f, kind, kwargs, msg=None, timer=timer))

        # Don't let the Future sit in _recv_events after it's done
        f.add_done_callback(
            lambda f: self._remove_finished_future(f, self._recv_futures)
        )

        if self._shadow_sock.get(EVENTS) & POLLIN:
            # recv immediately, if we can
            self._handle_recv()
        if self._recv_futures:
            self._add_io_state(POLLIN)
        return f

    def _add_send_event(self, kind, msg=None, kwargs=None, future=None):
        """Add a send event, returning the corresponding Future"""
        f = future or self._Future()
        # attempt send with DONTWAIT if no futures are waiting
        # short-circuit for sends that will resolve immediately
        # only call if no send Futures are waiting
        if kind in ('send', 'send_multipart') and not self._send_futures:
            flags = kwargs.get('flags', 0)
            nowait_kwargs = kwargs.copy()
            nowait_kwargs['flags'] = flags | _zmq.DONTWAIT

            # short-circuit non-blocking calls
            send = getattr(self._shadow_sock, kind)
            # track if the send resolved or not
            # (EAGAIN if DONTWAIT is not set should proceed with)
            finish_early = True
            try:
                r = send(msg, **nowait_kwargs)
            except _zmq.Again as e:
                if flags & _zmq.DONTWAIT:
                    f.set_exception(e)
                else:
                    # EAGAIN raised and DONTWAIT not requested,
                    # proceed with async send
                    finish_early = False
            except Exception as e:
                f.set_exception(e)
            else:
                f.set_result(r)

            if finish_early:
                # short-circuit resolved, return finished Future
                # schedule wake for recv if there are any receivers waiting
                if self._recv_futures:
                    self._schedule_remaining_events()
                return f

        timer = _NoTimer
        if hasattr(_zmq, 'SNDTIMEO'):
            timeout_ms = self._shadow_sock.get(_zmq.SNDTIMEO)
            if timeout_ms >= 0:
                timer = self._add_timeout(f, timeout_ms * 1e-3)

        # we add it to the list of futures before we add the timeout as the
        # timeout will remove the future from recv_futures to avoid leaks
        self._send_futures.append(
            _FutureEvent(f, kind, kwargs=kwargs, msg=msg, timer=timer)
        )
        # Don't let the Future sit in _send_futures after it's done
        f.add_done_callback(
            lambda f: self._remove_finished_future(f, self._send_futures)
        )

        self._add_io_state(POLLOUT)
        return f

    def _handle_recv(self):
        """Handle recv events"""
        if not self._shadow_sock.get(EVENTS) & POLLIN:
            # event triggered, but state may have been changed between trigger and callback
            return
        f = None
        while self._recv_futures:
            f, kind, kwargs, _, timer = self._recv_futures.popleft()
            # skip any cancelled futures
            if f.done():
                f = None
            else:
                break

        if not self._recv_futures:
            self._drop_io_state(POLLIN)

        if f is None:
            return

        timer.cancel()

        if kind == 'poll':
            # on poll event, just signal ready, nothing else.
            f.set_result(None)
            return
        elif kind == 'recv_multipart':
            recv = self._shadow_sock.recv_multipart
        elif kind == 'recv':
            recv = self._shadow_sock.recv
        else:
            raise ValueError("Unhandled recv event type: %r" % kind)

        kwargs['flags'] |= _zmq.DONTWAIT
        try:
            result = recv(**kwargs)
        except Exception as e:
            f.set_exception(e)
        else:
            f.set_result(result)

    def _handle_send(self):
        if not self._shadow_sock.get(EVENTS) & POLLOUT:
            # event triggered, but state may have been changed between trigger and callback
            return
        f = None
        while self._send_futures:
            f, kind, kwargs, msg, timer = self._send_futures.popleft()
            # skip any cancelled futures
            if f.done():
                f = None
            else:
                break

        if not self._send_futures:
            self._drop_io_state(POLLOUT)

        if f is None:
            return

        timer.cancel()

        if kind == 'poll':
            # on poll event, just signal ready, nothing else.
            f.set_result(None)
            return
        elif kind == 'send_multipart':
            send = self._shadow_sock.send_multipart
        elif kind == 'send':
            send = self._shadow_sock.send
        else:
            raise ValueError("Unhandled send event type: %r" % kind)

        kwargs['flags'] |= _zmq.DONTWAIT
        try:
            result = send(msg, **kwargs)
        except Exception as e:
            f.set_exception(e)
        else:
            f.set_result(result)

    # event masking from ZMQStream
    def _handle_events(self, fd=0, events=0):
        """Dispatch IO events to _handle_recv, etc."""
        if self._shadow_sock.closed:
            return

        zmq_events = self._shadow_sock.get(EVENTS)
        if zmq_events & _zmq.POLLIN:
            self._handle_recv()
        if zmq_events & _zmq.POLLOUT:
            self._handle_send()
        self._schedule_remaining_events()

    def _schedule_remaining_events(self, events=None):
        """Schedule a call to handle_events next loop iteration

        If there are still events to handle.
        """
        # edge-triggered handling
        # allow passing events in, in case this is triggered by retrieving events,
        # so we don't have to retrieve it twice.
        if self._state == 0:
            # not watching for anything, nothing to schedule
            return
        if events is None:
            events = self._shadow_sock.get(EVENTS)
        if events & self._state:
            self._call_later(0, self._handle_events)

    def _add_io_state(self, state):
        """Add io_state to poller."""
        if self._state != state:
            state = self._state = self._state | state
        self._update_handler(self._state)

    def _drop_io_state(self, state):
        """Stop poller from watching an io_state."""
        if self._state & state:
            self._state = self._state & (~state)
        self._update_handler(self._state)

    def _update_handler(self, state):
        """Update IOLoop handler with state.

        zmq FD is always read-only.
        """
        # ensure loop is registered and init_io has been called
        # if there are any events to watch for
        if state:
            self._get_loop()
        self._schedule_remaining_events()

    def _init_io_state(self, loop=None):
        """initialize the ioloop event handler"""
        if loop is None:
            loop = self._get_loop()
        loop.add_handler(self._shadow_sock, self._handle_events, self._READ)
        self._call_later(0, self._handle_events)

    def _clear_io_state(self):
        """unregister the ioloop event handler

        called once during close
        """
        fd = self._shadow_sock
        if self._shadow_sock.closed:
            fd = self._fd
        if self._current_loop is not None:
            self._current_loop.remove_handler(fd)
