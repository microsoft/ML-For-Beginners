"""0MQ Socket pure Python methods."""

# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.


import errno
import pickle
import random
import sys
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)
from warnings import warn

import zmq
from zmq._typing import Literal
from zmq.backend import Socket as SocketBase
from zmq.error import ZMQBindError, ZMQError
from zmq.utils import jsonapi
from zmq.utils.interop import cast_int_addr

from ..constants import SocketOption, SocketType, _OptType
from .attrsettr import AttributeSetter
from .poll import Poller

try:
    DEFAULT_PROTOCOL = pickle.DEFAULT_PROTOCOL
except AttributeError:
    DEFAULT_PROTOCOL = pickle.HIGHEST_PROTOCOL

T = TypeVar("T", bound="Socket")


class _SocketContext(Generic[T]):
    """Context Manager for socket bind/unbind"""

    socket: T
    kind: str
    addr: str

    def __repr__(self):
        return f"<SocketContext({self.kind}={self.addr!r})>"

    def __init__(self: "_SocketContext[T]", socket: T, kind: str, addr: str):
        assert kind in {"bind", "connect"}
        self.socket = socket
        self.kind = kind
        self.addr = addr

    def __enter__(self: "_SocketContext[T]") -> T:
        return self.socket

    def __exit__(self, *args):
        if self.socket.closed:
            return
        if self.kind == "bind":
            self.socket.unbind(self.addr)
        elif self.kind == "connect":
            self.socket.disconnect(self.addr)


ST = TypeVar("ST")


class Socket(SocketBase, AttributeSetter, Generic[ST]):
    """The ZMQ socket object

    To create a Socket, first create a Context::

        ctx = zmq.Context.instance()

    then call ``ctx.socket(socket_type)``::

        s = ctx.socket(zmq.ROUTER)

    .. versionadded:: 25

        Sockets can now be shadowed by passing another Socket.
        This helps in creating an async copy of a sync socket or vice versa::

            s = zmq.Socket(async_socket)

        Which previously had to be::

            s = zmq.Socket.shadow(async_socket.underlying)
    """

    _shadow = False
    _shadow_obj = None
    _monitor_socket = None
    _type_name = 'UNKNOWN'

    @overload
    def __init__(
        self: "Socket[bytes]",
        ctx_or_socket: "zmq.Context",
        socket_type: int,
        *,
        copy_threshold: Optional[int] = None,
    ):
        ...

    @overload
    def __init__(
        self: "Socket[bytes]",
        *,
        shadow: Union["Socket", int],
        copy_threshold: Optional[int] = None,
    ):
        ...

    @overload
    def __init__(
        self: "Socket[bytes]",
        ctx_or_socket: "Socket",
    ):
        ...

    def __init__(
        self: "Socket[bytes]",
        ctx_or_socket: Optional[Union["zmq.Context", "Socket"]] = None,
        socket_type: int = 0,
        *,
        shadow: Union["Socket", int] = 0,
        copy_threshold: Optional[int] = None,
    ):
        if isinstance(ctx_or_socket, zmq.Socket):
            # positional Socket(other_socket)
            shadow = ctx_or_socket
            ctx_or_socket = None

        shadow_address: int = 0

        if shadow:
            self._shadow = True
            # hold a reference to the shadow object
            self._shadow_obj = shadow
            if not isinstance(shadow, int):
                try:
                    shadow = cast(int, shadow.underlying)
                except AttributeError:
                    pass
            shadow_address = cast_int_addr(shadow)
        else:
            self._shadow = False

        super().__init__(
            ctx_or_socket,
            socket_type,
            shadow=shadow_address,
            copy_threshold=copy_threshold,
        )

        try:
            socket_type = cast(int, self.get(zmq.TYPE))
        except Exception:
            pass
        else:
            try:
                self.__dict__["type"] = stype = SocketType(socket_type)
            except ValueError:
                self._type_name = str(socket_type)
            else:
                self._type_name = stype.name

    def __del__(self):
        if not self._shadow and not self.closed:
            if warn is not None:
                # warn can be None during process teardown
                warn(
                    f"Unclosed socket {self}",
                    ResourceWarning,
                    stacklevel=2,
                    source=self,
                )
            self.close()

    _repr_cls = "zmq.Socket"

    def __repr__(self):
        cls = self.__class__
        # look up _repr_cls on exact class, not inherited
        _repr_cls = cls.__dict__.get("_repr_cls", None)
        if _repr_cls is None:
            _repr_cls = f"{cls.__module__}.{cls.__name__}"

        closed = ' closed' if self._closed else ''

        return f"<{_repr_cls}(zmq.{self._type_name}) at {hex(id(self))}{closed}>"

    # socket as context manager:
    def __enter__(self: T) -> T:
        """Sockets are context managers

        .. versionadded:: 14.4
        """
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    # -------------------------------------------------------------------------
    # Socket creation
    # -------------------------------------------------------------------------

    def __copy__(self: T, memo=None) -> T:
        """Copying a Socket creates a shadow copy"""
        return self.__class__.shadow(self.underlying)

    __deepcopy__ = __copy__

    @classmethod
    def shadow(cls: Type[T], address: Union[int, "zmq.Socket"]) -> T:
        """Shadow an existing libzmq socket

        address is a zmq.Socket or an integer (or FFI pointer)
        representing the address of the libzmq socket.

        .. versionadded:: 14.1

        .. versionadded:: 25
            Support for shadowing `zmq.Socket` objects,
            instead of just integer addresses.
        """
        return cls(shadow=address)

    def close(self, linger=None) -> None:
        """
        Close the socket.

        If linger is specified, LINGER sockopt will be set prior to closing.

        Note: closing a zmq Socket may not close the underlying sockets
        if there are undelivered messages.
        Only after all messages are delivered or discarded by reaching the socket's LINGER timeout
        (default: forever)
        will the underlying sockets be closed.

        This can be called to close the socket by hand. If this is not
        called, the socket will automatically be closed when it is
        garbage collected,
        in which case you may see a ResourceWarning about the unclosed socket.
        """
        if self.context:
            self.context._rm_socket(self)
        super().close(linger=linger)

    # -------------------------------------------------------------------------
    # Connect/Bind context managers
    # -------------------------------------------------------------------------

    def _connect_cm(self: T, addr: str) -> _SocketContext[T]:
        """Context manager to disconnect on exit

        .. versionadded:: 20.0
        """
        return _SocketContext(self, 'connect', addr)

    def _bind_cm(self: T, addr: str) -> _SocketContext[T]:
        """Context manager to unbind on exit

        .. versionadded:: 20.0
        """
        return _SocketContext(self, 'bind', addr)

    def bind(self: T, addr: str) -> _SocketContext[T]:
        """s.bind(addr)

        Bind the socket to an address.

        This causes the socket to listen on a network port. Sockets on the
        other side of this connection will use ``Socket.connect(addr)`` to
        connect to this socket.

        Returns a context manager which will call unbind on exit.

        .. versionadded:: 20.0
            Can be used as a context manager.

        Parameters
        ----------
        addr : str
            The address string. This has the form 'protocol://interface:port',
            for example 'tcp://127.0.0.1:5555'. Protocols supported include
            tcp, udp, pgm, epgm, inproc and ipc. If the address is unicode, it is
            encoded to utf-8 first.

        """
        try:
            super().bind(addr)
        except ZMQError as e:
            e.strerror += f" (addr={addr!r})"
            raise
        return self._bind_cm(addr)

    def connect(self: T, addr: str) -> _SocketContext[T]:
        """s.connect(addr)

        Connect to a remote 0MQ socket.

        Returns a context manager which will call disconnect on exit.

        .. versionadded:: 20.0
            Can be used as a context manager.

        Parameters
        ----------
        addr : str
            The address string. This has the form 'protocol://interface:port',
            for example 'tcp://127.0.0.1:5555'. Protocols supported are
            tcp, udp, pgm, inproc and ipc. If the address is unicode, it is
            encoded to utf-8 first.

        """
        try:
            super().connect(addr)
        except ZMQError as e:
            e.strerror += f" (addr={addr!r})"
            raise
        return self._connect_cm(addr)

    # -------------------------------------------------------------------------
    # Deprecated aliases
    # -------------------------------------------------------------------------

    @property
    def socket_type(self) -> int:
        warn("Socket.socket_type is deprecated, use Socket.type", DeprecationWarning)
        return cast(int, self.type)

    # -------------------------------------------------------------------------
    # Hooks for sockopt completion
    # -------------------------------------------------------------------------

    def __dir__(self):
        keys = dir(self.__class__)
        keys.extend(SocketOption.__members__)
        return keys

    # -------------------------------------------------------------------------
    # Getting/Setting options
    # -------------------------------------------------------------------------
    setsockopt = SocketBase.set
    getsockopt = SocketBase.get

    def __setattr__(self, key, value):
        """Override to allow setting zmq.[UN]SUBSCRIBE even though we have a subscribe method"""
        if key in self.__dict__:
            object.__setattr__(self, key, value)
            return
        _key = key.lower()
        if _key in ('subscribe', 'unsubscribe'):
            if isinstance(value, str):
                value = value.encode('utf8')
            if _key == 'subscribe':
                self.set(zmq.SUBSCRIBE, value)
            else:
                self.set(zmq.UNSUBSCRIBE, value)
            return
        super().__setattr__(key, value)

    def fileno(self) -> int:
        """Return edge-triggered file descriptor for this socket.

        This is a read-only edge-triggered file descriptor for both read and write events on this socket.
        It is important that all available events be consumed when an event is detected,
        otherwise the read event will not trigger again.

        .. versionadded:: 17.0
        """
        return self.FD

    def subscribe(self, topic: Union[str, bytes]) -> None:
        """Subscribe to a topic

        Only for SUB sockets.

        .. versionadded:: 15.3
        """
        if isinstance(topic, str):
            topic = topic.encode('utf8')
        self.set(zmq.SUBSCRIBE, topic)

    def unsubscribe(self, topic: Union[str, bytes]) -> None:
        """Unsubscribe from a topic

        Only for SUB sockets.

        .. versionadded:: 15.3
        """
        if isinstance(topic, str):
            topic = topic.encode('utf8')
        self.set(zmq.UNSUBSCRIBE, topic)

    def set_string(self, option: int, optval: str, encoding='utf-8') -> None:
        """Set socket options with a unicode object.

        This is simply a wrapper for setsockopt to protect from encoding ambiguity.

        See the 0MQ documentation for details on specific options.

        Parameters
        ----------
        option : int
            The name of the option to set. Can be any of: SUBSCRIBE,
            UNSUBSCRIBE, IDENTITY
        optval : str
            The value of the option to set.
        encoding : str
            The encoding to be used, default is utf8
        """
        if not isinstance(optval, str):
            raise TypeError(f"strings only, not {type(optval)}: {optval!r}")
        return self.set(option, optval.encode(encoding))

    setsockopt_unicode = setsockopt_string = set_string

    def get_string(self, option: int, encoding='utf-8') -> str:
        """Get the value of a socket option.

        See the 0MQ documentation for details on specific options.

        Parameters
        ----------
        option : int
            The option to retrieve.

        Returns
        -------
        optval : str
            The value of the option as a unicode string.
        """
        if SocketOption(option)._opt_type != _OptType.bytes:
            raise TypeError(f"option {option} will not return a string to be decoded")
        return cast(bytes, self.get(option)).decode(encoding)

    getsockopt_unicode = getsockopt_string = get_string

    def bind_to_random_port(
        self: T,
        addr: str,
        min_port: int = 49152,
        max_port: int = 65536,
        max_tries: int = 100,
    ) -> int:
        """Bind this socket to a random port in a range.

        If the port range is unspecified, the system will choose the port.

        Parameters
        ----------
        addr : str
            The address string without the port to pass to ``Socket.bind()``.
        min_port : int, optional
            The minimum port in the range of ports to try (inclusive).
        max_port : int, optional
            The maximum port in the range of ports to try (exclusive).
        max_tries : int, optional
            The maximum number of bind attempts to make.

        Returns
        -------
        port : int
            The port the socket was bound to.

        Raises
        ------
        ZMQBindError
            if `max_tries` reached before successful bind
        """
        if (
            (zmq.zmq_version_info() >= (3, 2))
            and min_port == 49152
            and max_port == 65536
        ):
            # if LAST_ENDPOINT is supported, and min_port / max_port weren't specified,
            # we can bind to port 0 and let the OS do the work
            self.bind("%s:*" % addr)
            url = cast(bytes, self.last_endpoint).decode('ascii', 'replace')
            _, port_s = url.rsplit(':', 1)
            return int(port_s)

        for i in range(max_tries):
            try:
                port = random.randrange(min_port, max_port)
                self.bind(f'{addr}:{port}')
            except ZMQError as exception:
                en = exception.errno
                if en == zmq.EADDRINUSE:
                    continue
                elif sys.platform == 'win32' and en == errno.EACCES:
                    continue
                else:
                    raise
            else:
                return port
        raise ZMQBindError("Could not bind socket to random port.")

    def get_hwm(self) -> int:
        """Get the High Water Mark.

        On libzmq ≥ 3, this gets SNDHWM if available, otherwise RCVHWM
        """
        major = zmq.zmq_version_info()[0]
        if major >= 3:
            # return sndhwm, fallback on rcvhwm
            try:
                return cast(int, self.get(zmq.SNDHWM))
            except zmq.ZMQError:
                pass

            return cast(int, self.get(zmq.RCVHWM))
        else:
            return cast(int, self.get(zmq.HWM))

    def set_hwm(self, value: int) -> None:
        """Set the High Water Mark.

        On libzmq ≥ 3, this sets both SNDHWM and RCVHWM


        .. warning::

            New values only take effect for subsequent socket
            bind/connects.
        """
        major = zmq.zmq_version_info()[0]
        if major >= 3:
            raised = None
            try:
                self.sndhwm = value
            except Exception as e:
                raised = e
            try:
                self.rcvhwm = value
            except Exception as e:
                raised = e

            if raised:
                raise raised
        else:
            self.set(zmq.HWM, value)

    hwm = property(
        get_hwm,
        set_hwm,
        None,
        """Property for High Water Mark.

        Setting hwm sets both SNDHWM and RCVHWM as appropriate.
        It gets SNDHWM if available, otherwise RCVHWM.
        """,
    )

    # -------------------------------------------------------------------------
    # Sending and receiving messages
    # -------------------------------------------------------------------------

    @overload
    def send(
        self,
        data: Any,
        flags: int = ...,
        copy: bool = ...,
        *,
        track: Literal[True],
        routing_id: Optional[int] = ...,
        group: Optional[str] = ...,
    ) -> "zmq.MessageTracker":
        ...

    @overload
    def send(
        self,
        data: Any,
        flags: int = ...,
        copy: bool = ...,
        *,
        track: Literal[False],
        routing_id: Optional[int] = ...,
        group: Optional[str] = ...,
    ) -> None:
        ...

    @overload
    def send(
        self,
        data: Any,
        flags: int = ...,
        *,
        copy: bool = ...,
        routing_id: Optional[int] = ...,
        group: Optional[str] = ...,
    ) -> None:
        ...

    @overload
    def send(
        self,
        data: Any,
        flags: int = ...,
        copy: bool = ...,
        track: bool = ...,
        routing_id: Optional[int] = ...,
        group: Optional[str] = ...,
    ) -> Optional["zmq.MessageTracker"]:
        ...

    def send(
        self,
        data: Any,
        flags: int = 0,
        copy: bool = True,
        track: bool = False,
        routing_id: Optional[int] = None,
        group: Optional[str] = None,
    ) -> Optional["zmq.MessageTracker"]:
        """Send a single zmq message frame on this socket.

        This queues the message to be sent by the IO thread at a later time.

        With flags=NOBLOCK, this raises :class:`ZMQError` if the queue is full;
        otherwise, this waits until space is available.
        See :class:`Poller` for more general non-blocking I/O.

        Parameters
        ----------
        data : bytes, Frame, memoryview
            The content of the message. This can be any object that provides
            the Python buffer API (i.e. `memoryview(data)` can be called).
        flags : int
            0, NOBLOCK, SNDMORE, or NOBLOCK|SNDMORE.
        copy : bool
            Should the message be sent in a copying or non-copying manner.
        track : bool
            Should the message be tracked for notification that ZMQ has
            finished with it? (ignored if copy=True)
        routing_id : int
            For use with SERVER sockets
        group : str
            For use with RADIO sockets

        Returns
        -------
        None : if `copy` or not track
            None if message was sent, raises an exception otherwise.
        MessageTracker : if track and not copy
            a MessageTracker object, whose `pending` property will
            be True until the send is completed.

        Raises
        ------
        TypeError
            If a unicode object is passed
        ValueError
            If `track=True`, but an untracked Frame is passed.
        ZMQError
            If the send does not succeed for any reason (including
            if NOBLOCK is set and the outgoing queue is full).


        .. versionchanged:: 17.0

            DRAFT support for routing_id and group arguments.
        """
        if routing_id is not None:
            if not isinstance(data, zmq.Frame):
                data = zmq.Frame(
                    data,
                    track=track,
                    copy=copy or None,
                    copy_threshold=self.copy_threshold,
                )
            data.routing_id = routing_id
        if group is not None:
            if not isinstance(data, zmq.Frame):
                data = zmq.Frame(
                    data,
                    track=track,
                    copy=copy or None,
                    copy_threshold=self.copy_threshold,
                )
            data.group = group
        return super().send(data, flags=flags, copy=copy, track=track)

    def send_multipart(
        self,
        msg_parts: Sequence,
        flags: int = 0,
        copy: bool = True,
        track: bool = False,
        **kwargs,
    ):
        """Send a sequence of buffers as a multipart message.

        The zmq.SNDMORE flag is added to all msg parts before the last.

        Parameters
        ----------
        msg_parts : iterable
            A sequence of objects to send as a multipart message. Each element
            can be any sendable object (Frame, bytes, buffer-providers)
        flags : int, optional
            Any valid flags for :func:`Socket.send`.
            SNDMORE is added automatically for frames before the last.
        copy : bool, optional
            Should the frame(s) be sent in a copying or non-copying manner.
            If copy=False, frames smaller than self.copy_threshold bytes
            will be copied anyway.
        track : bool, optional
            Should the frame(s) be tracked for notification that ZMQ has
            finished with it (ignored if copy=True).

        Returns
        -------
        None : if copy or not track
        MessageTracker : if track and not copy
            a MessageTracker object, whose `pending` property will
            be True until the last send is completed.
        """
        # typecheck parts before sending:
        for i, msg in enumerate(msg_parts):
            if isinstance(msg, (zmq.Frame, bytes, memoryview)):
                continue
            try:
                memoryview(msg)
            except Exception:
                rmsg = repr(msg)
                if len(rmsg) > 32:
                    rmsg = rmsg[:32] + '...'
                raise TypeError(
                    "Frame %i (%s) does not support the buffer interface."
                    % (
                        i,
                        rmsg,
                    )
                )
        for msg in msg_parts[:-1]:
            self.send(msg, zmq.SNDMORE | flags, copy=copy, track=track)
        # Send the last part without the extra SNDMORE flag.
        return self.send(msg_parts[-1], flags, copy=copy, track=track)

    @overload
    def recv_multipart(
        self, flags: int = ..., *, copy: Literal[True], track: bool = ...
    ) -> List[bytes]:
        ...

    @overload
    def recv_multipart(
        self, flags: int = ..., *, copy: Literal[False], track: bool = ...
    ) -> List[zmq.Frame]:
        ...

    @overload
    def recv_multipart(self, flags: int = ..., *, track: bool = ...) -> List[bytes]:
        ...

    @overload
    def recv_multipart(
        self, flags: int = 0, copy: bool = True, track: bool = False
    ) -> Union[List[zmq.Frame], List[bytes]]:
        ...

    def recv_multipart(
        self, flags: int = 0, copy: bool = True, track: bool = False
    ) -> Union[List[zmq.Frame], List[bytes]]:
        """Receive a multipart message as a list of bytes or Frame objects

        Parameters
        ----------
        flags : int, optional
            Any valid flags for :func:`Socket.recv`.
        copy : bool, optional
            Should the message frame(s) be received in a copying or non-copying manner?
            If False a Frame object is returned for each part, if True a copy of
            the bytes is made for each frame.
        track : bool, optional
            Should the message frame(s) be tracked for notification that ZMQ has
            finished with it? (ignored if copy=True)

        Returns
        -------
        msg_parts : list
            A list of frames in the multipart message; either Frames or bytes,
            depending on `copy`.

        Raises
        ------
        ZMQError
            for any of the reasons :func:`~Socket.recv` might fail
        """
        parts = [self.recv(flags, copy=copy, track=track)]
        # have first part already, only loop while more to receive
        while self.getsockopt(zmq.RCVMORE):
            part = self.recv(flags, copy=copy, track=track)
            parts.append(part)
        # cast List[Union] to Union[List]
        # how do we get mypy to recognize that return type is invariant on `copy`?
        return cast(Union[List[zmq.Frame], List[bytes]], parts)

    def _deserialize(
        self,
        recvd: bytes,
        load: Callable[[bytes], Any],
    ) -> Any:
        """Deserialize a received message

        Override in subclass (e.g. Futures) if recvd is not the raw bytes.

        The default implementation expects bytes and returns the deserialized message immediately.

        Parameters
        ----------

        load: callable
            Callable that deserializes bytes
        recvd:
            The object returned by self.recv

        """
        return load(recvd)

    def send_serialized(self, msg, serialize, flags=0, copy=True, **kwargs):
        """Send a message with a custom serialization function.

        .. versionadded:: 17

        Parameters
        ----------
        msg : The message to be sent. Can be any object serializable by `serialize`.
        serialize : callable
            The serialization function to use.
            serialize(msg) should return an iterable of sendable message frames
            (e.g. bytes objects), which will be passed to send_multipart.
        flags : int, optional
            Any valid flags for :func:`Socket.send`.
        copy : bool, optional
            Whether to copy the frames.

        """
        frames = serialize(msg)
        return self.send_multipart(frames, flags=flags, copy=copy, **kwargs)

    def recv_serialized(self, deserialize, flags=0, copy=True):
        """Receive a message with a custom deserialization function.

        .. versionadded:: 17

        Parameters
        ----------
        deserialize : callable
            The deserialization function to use.
            deserialize will be called with one argument: the list of frames
            returned by recv_multipart() and can return any object.
        flags : int, optional
            Any valid flags for :func:`Socket.recv`.
        copy : bool, optional
            Whether to recv bytes or Frame objects.

        Returns
        -------
        obj : object
            The object returned by the deserialization function.

        Raises
        ------
        ZMQError
            for any of the reasons :func:`~Socket.recv` might fail
        """
        frames = self.recv_multipart(flags=flags, copy=copy)
        return self._deserialize(frames, deserialize)

    def send_string(
        self,
        u: str,
        flags: int = 0,
        copy: bool = True,
        encoding: str = 'utf-8',
        **kwargs,
    ) -> Optional["zmq.Frame"]:
        """Send a Python unicode string as a message with an encoding.

        0MQ communicates with raw bytes, so you must encode/decode
        text (str) around 0MQ.

        Parameters
        ----------
        u : str
            The unicode string to send.
        flags : int, optional
            Any valid flags for :func:`Socket.send`.
        encoding : str [default: 'utf-8']
            The encoding to be used
        """
        if not isinstance(u, str):
            raise TypeError("str objects only")
        return self.send(u.encode(encoding), flags=flags, copy=copy, **kwargs)

    send_unicode = send_string

    def recv_string(self, flags: int = 0, encoding: str = 'utf-8') -> str:
        """Receive a unicode string, as sent by send_string.

        Parameters
        ----------
        flags : int
            Any valid flags for :func:`Socket.recv`.
        encoding : str [default: 'utf-8']
            The encoding to be used

        Returns
        -------
        s : str
            The Python unicode string that arrives as encoded bytes.

        Raises
        ------
        ZMQError
            for any of the reasons :func:`~Socket.recv` might fail
        """
        msg = self.recv(flags=flags)
        return self._deserialize(msg, lambda buf: buf.decode(encoding))

    recv_unicode = recv_string

    def send_pyobj(
        self, obj: Any, flags: int = 0, protocol: int = DEFAULT_PROTOCOL, **kwargs
    ) -> Optional[zmq.Frame]:
        """Send a Python object as a message using pickle to serialize.

        Parameters
        ----------
        obj : Python object
            The Python object to send.
        flags : int
            Any valid flags for :func:`Socket.send`.
        protocol : int
            The pickle protocol number to use. The default is pickle.DEFAULT_PROTOCOL
            where defined, and pickle.HIGHEST_PROTOCOL elsewhere.
        """
        msg = pickle.dumps(obj, protocol)
        return self.send(msg, flags=flags, **kwargs)

    def recv_pyobj(self, flags: int = 0) -> Any:
        """Receive a Python object as a message using pickle to serialize.

        Parameters
        ----------
        flags : int
            Any valid flags for :func:`Socket.recv`.

        Returns
        -------
        obj : Python object
            The Python object that arrives as a message.

        Raises
        ------
        ZMQError
            for any of the reasons :func:`~Socket.recv` might fail
        """
        msg = self.recv(flags)
        return self._deserialize(msg, pickle.loads)

    def send_json(self, obj: Any, flags: int = 0, **kwargs) -> None:
        """Send a Python object as a message using json to serialize.

        Keyword arguments are passed on to json.dumps

        Parameters
        ----------
        obj : Python object
            The Python object to send
        flags : int
            Any valid flags for :func:`Socket.send`
        """
        send_kwargs = {}
        for key in ('routing_id', 'group'):
            if key in kwargs:
                send_kwargs[key] = kwargs.pop(key)
        msg = jsonapi.dumps(obj, **kwargs)
        return self.send(msg, flags=flags, **send_kwargs)

    def recv_json(self, flags: int = 0, **kwargs) -> Union[List, str, int, float, Dict]:
        """Receive a Python object as a message using json to serialize.

        Keyword arguments are passed on to json.loads

        Parameters
        ----------
        flags : int
            Any valid flags for :func:`Socket.recv`.

        Returns
        -------
        obj : Python object
            The Python object that arrives as a message.

        Raises
        ------
        ZMQError
            for any of the reasons :func:`~Socket.recv` might fail
        """
        msg = self.recv(flags)
        return self._deserialize(msg, lambda buf: jsonapi.loads(buf, **kwargs))

    _poller_class = Poller

    def poll(self, timeout=None, flags=zmq.POLLIN) -> int:
        """Poll the socket for events.
        See :class:`Poller` to wait for multiple sockets at once.

        Parameters
        ----------
        timeout : int [default: None]
            The timeout (in milliseconds) to wait for an event. If unspecified
            (or specified None), will wait forever for an event.
        flags : int [default: POLLIN]
            POLLIN, POLLOUT, or POLLIN|POLLOUT. The event flags to poll for.

        Returns
        -------
        event_mask : int
            The poll event mask (POLLIN, POLLOUT),
            0 if the timeout was reached without an event.
        """

        if self.closed:
            raise ZMQError(zmq.ENOTSUP)

        p = self._poller_class()
        p.register(self, flags)
        evts = dict(p.poll(timeout))
        # return 0 if no events, otherwise return event bitfield
        return evts.get(self, 0)

    def get_monitor_socket(
        self: T, events: Optional[int] = None, addr: Optional[str] = None
    ) -> T:
        """Return a connected PAIR socket ready to receive the event notifications.

        .. versionadded:: libzmq-4.0
        .. versionadded:: 14.0

        Parameters
        ----------
        events : int [default: ZMQ_EVENT_ALL]
            The bitmask defining which events are wanted.
        addr :  string [default: None]
            The optional endpoint for the monitoring sockets.

        Returns
        -------
        socket :  (PAIR)
            The socket is already connected and ready to receive messages.
        """
        # safe-guard, method only available on libzmq >= 4
        if zmq.zmq_version_info() < (4,):
            raise NotImplementedError(
                "get_monitor_socket requires libzmq >= 4, have %s" % zmq.zmq_version()
            )

        # if already monitoring, return existing socket
        if self._monitor_socket:
            if self._monitor_socket.closed:
                self._monitor_socket = None
            else:
                return self._monitor_socket

        if addr is None:
            # create endpoint name from internal fd
            addr = f"inproc://monitor.s-{self.FD}"
        if events is None:
            # use all events
            events = zmq.EVENT_ALL
        # attach monitoring socket
        self.monitor(addr, events)
        # create new PAIR socket and connect it
        self._monitor_socket = self.context.socket(zmq.PAIR)
        self._monitor_socket.connect(addr)
        return self._monitor_socket

    def disable_monitor(self) -> None:
        """Shutdown the PAIR socket (created using get_monitor_socket)
        that is serving socket events.

        .. versionadded:: 14.4
        """
        self._monitor_socket = None
        self.monitor(None, 0)


__all__ = ['Socket']
