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

"""Utility classes to write to and read from non-blocking files and sockets.

Contents:

* `BaseIOStream`: Generic interface for reading and writing.
* `IOStream`: Implementation of BaseIOStream using non-blocking sockets.
* `SSLIOStream`: SSL-aware version of IOStream.
* `PipeIOStream`: Pipe-based IOStream implementation.
"""

import asyncio
import collections
import errno
import io
import numbers
import os
import socket
import ssl
import sys
import re

from tornado.concurrent import Future, future_set_result_unless_cancelled
from tornado import ioloop
from tornado.log import gen_log
from tornado.netutil import ssl_wrap_socket, _client_ssl_defaults, _server_ssl_defaults
from tornado.util import errno_from_exception

import typing
from typing import (
    Union,
    Optional,
    Awaitable,
    Callable,
    Pattern,
    Any,
    Dict,
    TypeVar,
    Tuple,
)
from types import TracebackType

if typing.TYPE_CHECKING:
    from typing import Deque, List, Type  # noqa: F401

_IOStreamType = TypeVar("_IOStreamType", bound="IOStream")

# These errnos indicate that a connection has been abruptly terminated.
# They should be caught and handled less noisily than other errors.
_ERRNO_CONNRESET = (errno.ECONNRESET, errno.ECONNABORTED, errno.EPIPE, errno.ETIMEDOUT)

if hasattr(errno, "WSAECONNRESET"):
    _ERRNO_CONNRESET += (  # type: ignore
        errno.WSAECONNRESET,  # type: ignore
        errno.WSAECONNABORTED,  # type: ignore
        errno.WSAETIMEDOUT,  # type: ignore
    )

if sys.platform == "darwin":
    # OSX appears to have a race condition that causes send(2) to return
    # EPROTOTYPE if called while a socket is being torn down:
    # http://erickt.github.io/blog/2014/11/19/adventures-in-debugging-a-potential-osx-kernel-bug/
    # Since the socket is being closed anyway, treat this as an ECONNRESET
    # instead of an unexpected error.
    _ERRNO_CONNRESET += (errno.EPROTOTYPE,)  # type: ignore

_WINDOWS = sys.platform.startswith("win")


class StreamClosedError(IOError):
    """Exception raised by `IOStream` methods when the stream is closed.

    Note that the close callback is scheduled to run *after* other
    callbacks on the stream (to allow for buffered data to be processed),
    so you may see this error before you see the close callback.

    The ``real_error`` attribute contains the underlying error that caused
    the stream to close (if any).

    .. versionchanged:: 4.3
       Added the ``real_error`` attribute.
    """

    def __init__(self, real_error: Optional[BaseException] = None) -> None:
        super().__init__("Stream is closed")
        self.real_error = real_error


class UnsatisfiableReadError(Exception):
    """Exception raised when a read cannot be satisfied.

    Raised by ``read_until`` and ``read_until_regex`` with a ``max_bytes``
    argument.
    """

    pass


class StreamBufferFullError(Exception):
    """Exception raised by `IOStream` methods when the buffer is full."""


class _StreamBuffer(object):
    """
    A specialized buffer that tries to avoid copies when large pieces
    of data are encountered.
    """

    def __init__(self) -> None:
        # A sequence of (False, bytearray) and (True, memoryview) objects
        self._buffers = (
            collections.deque()
        )  # type: Deque[Tuple[bool, Union[bytearray, memoryview]]]
        # Position in the first buffer
        self._first_pos = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    # Data above this size will be appended separately instead
    # of extending an existing bytearray
    _large_buf_threshold = 2048

    def append(self, data: Union[bytes, bytearray, memoryview]) -> None:
        """
        Append the given piece of data (should be a buffer-compatible object).
        """
        size = len(data)
        if size > self._large_buf_threshold:
            if not isinstance(data, memoryview):
                data = memoryview(data)
            self._buffers.append((True, data))
        elif size > 0:
            if self._buffers:
                is_memview, b = self._buffers[-1]
                new_buf = is_memview or len(b) >= self._large_buf_threshold
            else:
                new_buf = True
            if new_buf:
                self._buffers.append((False, bytearray(data)))
            else:
                b += data  # type: ignore

        self._size += size

    def peek(self, size: int) -> memoryview:
        """
        Get a view over at most ``size`` bytes (possibly fewer) at the
        current buffer position.
        """
        assert size > 0
        try:
            is_memview, b = self._buffers[0]
        except IndexError:
            return memoryview(b"")

        pos = self._first_pos
        if is_memview:
            return typing.cast(memoryview, b[pos : pos + size])
        else:
            return memoryview(b)[pos : pos + size]

    def advance(self, size: int) -> None:
        """
        Advance the current buffer position by ``size`` bytes.
        """
        assert 0 < size <= self._size
        self._size -= size
        pos = self._first_pos

        buffers = self._buffers
        while buffers and size > 0:
            is_large, b = buffers[0]
            b_remain = len(b) - size - pos
            if b_remain <= 0:
                buffers.popleft()
                size -= len(b) - pos
                pos = 0
            elif is_large:
                pos += size
                size = 0
            else:
                pos += size
                del typing.cast(bytearray, b)[:pos]
                pos = 0
                size = 0

        assert size == 0
        self._first_pos = pos


class BaseIOStream(object):
    """A utility class to write to and read from a non-blocking file or socket.

    We support a non-blocking ``write()`` and a family of ``read_*()``
    methods. When the operation completes, the ``Awaitable`` will resolve
    with the data read (or ``None`` for ``write()``). All outstanding
    ``Awaitables`` will resolve with a `StreamClosedError` when the
    stream is closed; `.BaseIOStream.set_close_callback` can also be used
    to be notified of a closed stream.

    When a stream is closed due to an error, the IOStream's ``error``
    attribute contains the exception object.

    Subclasses must implement `fileno`, `close_fd`, `write_to_fd`,
    `read_from_fd`, and optionally `get_fd_error`.

    """

    def __init__(
        self,
        max_buffer_size: Optional[int] = None,
        read_chunk_size: Optional[int] = None,
        max_write_buffer_size: Optional[int] = None,
    ) -> None:
        """`BaseIOStream` constructor.

        :arg max_buffer_size: Maximum amount of incoming data to buffer;
            defaults to 100MB.
        :arg read_chunk_size: Amount of data to read at one time from the
            underlying transport; defaults to 64KB.
        :arg max_write_buffer_size: Amount of outgoing data to buffer;
            defaults to unlimited.

        .. versionchanged:: 4.0
           Add the ``max_write_buffer_size`` parameter.  Changed default
           ``read_chunk_size`` to 64KB.
        .. versionchanged:: 5.0
           The ``io_loop`` argument (deprecated since version 4.1) has been
           removed.
        """
        self.io_loop = ioloop.IOLoop.current()
        self.max_buffer_size = max_buffer_size or 104857600
        # A chunk size that is too close to max_buffer_size can cause
        # spurious failures.
        self.read_chunk_size = min(read_chunk_size or 65536, self.max_buffer_size // 2)
        self.max_write_buffer_size = max_write_buffer_size
        self.error = None  # type: Optional[BaseException]
        self._read_buffer = bytearray()
        self._read_buffer_size = 0
        self._user_read_buffer = False
        self._after_user_read_buffer = None  # type: Optional[bytearray]
        self._write_buffer = _StreamBuffer()
        self._total_write_index = 0
        self._total_write_done_index = 0
        self._read_delimiter = None  # type: Optional[bytes]
        self._read_regex = None  # type: Optional[Pattern]
        self._read_max_bytes = None  # type: Optional[int]
        self._read_bytes = None  # type: Optional[int]
        self._read_partial = False
        self._read_until_close = False
        self._read_future = None  # type: Optional[Future]
        self._write_futures = (
            collections.deque()
        )  # type: Deque[Tuple[int, Future[None]]]
        self._close_callback = None  # type: Optional[Callable[[], None]]
        self._connect_future = None  # type: Optional[Future[IOStream]]
        # _ssl_connect_future should be defined in SSLIOStream
        # but it's here so we can clean it up in _signal_closed
        # TODO: refactor that so subclasses can add additional futures
        # to be cancelled.
        self._ssl_connect_future = None  # type: Optional[Future[SSLIOStream]]
        self._connecting = False
        self._state = None  # type: Optional[int]
        self._closed = False

    def fileno(self) -> Union[int, ioloop._Selectable]:
        """Returns the file descriptor for this stream."""
        raise NotImplementedError()

    def close_fd(self) -> None:
        """Closes the file underlying this stream.

        ``close_fd`` is called by `BaseIOStream` and should not be called
        elsewhere; other users should call `close` instead.
        """
        raise NotImplementedError()

    def write_to_fd(self, data: memoryview) -> int:
        """Attempts to write ``data`` to the underlying file.

        Returns the number of bytes written.
        """
        raise NotImplementedError()

    def read_from_fd(self, buf: Union[bytearray, memoryview]) -> Optional[int]:
        """Attempts to read from the underlying file.

        Reads up to ``len(buf)`` bytes, storing them in the buffer.
        Returns the number of bytes read. Returns None if there was
        nothing to read (the socket returned `~errno.EWOULDBLOCK` or
        equivalent), and zero on EOF.

        .. versionchanged:: 5.0

           Interface redesigned to take a buffer and return a number
           of bytes instead of a freshly-allocated object.
        """
        raise NotImplementedError()

    def get_fd_error(self) -> Optional[Exception]:
        """Returns information about any error on the underlying file.

        This method is called after the `.IOLoop` has signaled an error on the
        file descriptor, and should return an Exception (such as `socket.error`
        with additional information, or None if no such information is
        available.
        """
        return None

    def read_until_regex(
        self, regex: bytes, max_bytes: Optional[int] = None
    ) -> Awaitable[bytes]:
        """Asynchronously read until we have matched the given regex.

        The result includes the data that matches the regex and anything
        that came before it.

        If ``max_bytes`` is not None, the connection will be closed
        if more than ``max_bytes`` bytes have been read and the regex is
        not satisfied.

        .. versionchanged:: 4.0
            Added the ``max_bytes`` argument.  The ``callback`` argument is
            now optional and a `.Future` will be returned if it is omitted.

        .. versionchanged:: 6.0

           The ``callback`` argument was removed. Use the returned
           `.Future` instead.

        """
        future = self._start_read()
        self._read_regex = re.compile(regex)
        self._read_max_bytes = max_bytes
        try:
            self._try_inline_read()
        except UnsatisfiableReadError as e:
            # Handle this the same way as in _handle_events.
            gen_log.info("Unsatisfiable read, closing connection: %s" % e)
            self.close(exc_info=e)
            return future
        except:
            # Ensure that the future doesn't log an error because its
            # failure was never examined.
            future.add_done_callback(lambda f: f.exception())
            raise
        return future

    def read_until(
        self, delimiter: bytes, max_bytes: Optional[int] = None
    ) -> Awaitable[bytes]:
        """Asynchronously read until we have found the given delimiter.

        The result includes all the data read including the delimiter.

        If ``max_bytes`` is not None, the connection will be closed
        if more than ``max_bytes`` bytes have been read and the delimiter
        is not found.

        .. versionchanged:: 4.0
            Added the ``max_bytes`` argument.  The ``callback`` argument is
            now optional and a `.Future` will be returned if it is omitted.

        .. versionchanged:: 6.0

           The ``callback`` argument was removed. Use the returned
           `.Future` instead.
        """
        future = self._start_read()
        self._read_delimiter = delimiter
        self._read_max_bytes = max_bytes
        try:
            self._try_inline_read()
        except UnsatisfiableReadError as e:
            # Handle this the same way as in _handle_events.
            gen_log.info("Unsatisfiable read, closing connection: %s" % e)
            self.close(exc_info=e)
            return future
        except:
            future.add_done_callback(lambda f: f.exception())
            raise
        return future

    def read_bytes(self, num_bytes: int, partial: bool = False) -> Awaitable[bytes]:
        """Asynchronously read a number of bytes.

        If ``partial`` is true, data is returned as soon as we have
        any bytes to return (but never more than ``num_bytes``)

        .. versionchanged:: 4.0
            Added the ``partial`` argument.  The callback argument is now
            optional and a `.Future` will be returned if it is omitted.

        .. versionchanged:: 6.0

           The ``callback`` and ``streaming_callback`` arguments have
           been removed. Use the returned `.Future` (and
           ``partial=True`` for ``streaming_callback``) instead.

        """
        future = self._start_read()
        assert isinstance(num_bytes, numbers.Integral)
        self._read_bytes = num_bytes
        self._read_partial = partial
        try:
            self._try_inline_read()
        except:
            future.add_done_callback(lambda f: f.exception())
            raise
        return future

    def read_into(self, buf: bytearray, partial: bool = False) -> Awaitable[int]:
        """Asynchronously read a number of bytes.

        ``buf`` must be a writable buffer into which data will be read.

        If ``partial`` is true, the callback is run as soon as any bytes
        have been read.  Otherwise, it is run when the ``buf`` has been
        entirely filled with read data.

        .. versionadded:: 5.0

        .. versionchanged:: 6.0

           The ``callback`` argument was removed. Use the returned
           `.Future` instead.

        """
        future = self._start_read()

        # First copy data already in read buffer
        available_bytes = self._read_buffer_size
        n = len(buf)
        if available_bytes >= n:
            buf[:] = memoryview(self._read_buffer)[:n]
            del self._read_buffer[:n]
            self._after_user_read_buffer = self._read_buffer
        elif available_bytes > 0:
            buf[:available_bytes] = memoryview(self._read_buffer)[:]

        # Set up the supplied buffer as our temporary read buffer.
        # The original (if it had any data remaining) has been
        # saved for later.
        self._user_read_buffer = True
        self._read_buffer = buf
        self._read_buffer_size = available_bytes
        self._read_bytes = n
        self._read_partial = partial

        try:
            self._try_inline_read()
        except:
            future.add_done_callback(lambda f: f.exception())
            raise
        return future

    def read_until_close(self) -> Awaitable[bytes]:
        """Asynchronously reads all data from the socket until it is closed.

        This will buffer all available data until ``max_buffer_size``
        is reached. If flow control or cancellation are desired, use a
        loop with `read_bytes(partial=True) <.read_bytes>` instead.

        .. versionchanged:: 4.0
            The callback argument is now optional and a `.Future` will
            be returned if it is omitted.

        .. versionchanged:: 6.0

           The ``callback`` and ``streaming_callback`` arguments have
           been removed. Use the returned `.Future` (and `read_bytes`
           with ``partial=True`` for ``streaming_callback``) instead.

        """
        future = self._start_read()
        if self.closed():
            self._finish_read(self._read_buffer_size)
            return future
        self._read_until_close = True
        try:
            self._try_inline_read()
        except:
            future.add_done_callback(lambda f: f.exception())
            raise
        return future

    def write(self, data: Union[bytes, memoryview]) -> "Future[None]":
        """Asynchronously write the given data to this stream.

        This method returns a `.Future` that resolves (with a result
        of ``None``) when the write has been completed.

        The ``data`` argument may be of type `bytes` or `memoryview`.

        .. versionchanged:: 4.0
            Now returns a `.Future` if no callback is given.

        .. versionchanged:: 4.5
            Added support for `memoryview` arguments.

        .. versionchanged:: 6.0

           The ``callback`` argument was removed. Use the returned
           `.Future` instead.

        """
        self._check_closed()
        if data:
            if isinstance(data, memoryview):
                # Make sure that ``len(data) == data.nbytes``
                data = memoryview(data).cast("B")
            if (
                self.max_write_buffer_size is not None
                and len(self._write_buffer) + len(data) > self.max_write_buffer_size
            ):
                raise StreamBufferFullError("Reached maximum write buffer size")
            self._write_buffer.append(data)
            self._total_write_index += len(data)
        future = Future()  # type: Future[None]
        future.add_done_callback(lambda f: f.exception())
        self._write_futures.append((self._total_write_index, future))
        if not self._connecting:
            self._handle_write()
            if self._write_buffer:
                self._add_io_state(self.io_loop.WRITE)
            self._maybe_add_error_listener()
        return future

    def set_close_callback(self, callback: Optional[Callable[[], None]]) -> None:
        """Call the given callback when the stream is closed.

        This mostly is not necessary for applications that use the
        `.Future` interface; all outstanding ``Futures`` will resolve
        with a `StreamClosedError` when the stream is closed. However,
        it is still useful as a way to signal that the stream has been
        closed while no other read or write is in progress.

        Unlike other callback-based interfaces, ``set_close_callback``
        was not removed in Tornado 6.0.
        """
        self._close_callback = callback
        self._maybe_add_error_listener()

    def close(
        self,
        exc_info: Union[
            None,
            bool,
            BaseException,
            Tuple[
                "Optional[Type[BaseException]]",
                Optional[BaseException],
                Optional[TracebackType],
            ],
        ] = False,
    ) -> None:
        """Close this stream.

        If ``exc_info`` is true, set the ``error`` attribute to the current
        exception from `sys.exc_info` (or if ``exc_info`` is a tuple,
        use that instead of `sys.exc_info`).
        """
        if not self.closed():
            if exc_info:
                if isinstance(exc_info, tuple):
                    self.error = exc_info[1]
                elif isinstance(exc_info, BaseException):
                    self.error = exc_info
                else:
                    exc_info = sys.exc_info()
                    if any(exc_info):
                        self.error = exc_info[1]
            if self._read_until_close:
                self._read_until_close = False
                self._finish_read(self._read_buffer_size)
            elif self._read_future is not None:
                # resolve reads that are pending and ready to complete
                try:
                    pos = self._find_read_pos()
                except UnsatisfiableReadError:
                    pass
                else:
                    if pos is not None:
                        self._read_from_buffer(pos)
            if self._state is not None:
                self.io_loop.remove_handler(self.fileno())
                self._state = None
            self.close_fd()
            self._closed = True
        self._signal_closed()

    def _signal_closed(self) -> None:
        futures = []  # type: List[Future]
        if self._read_future is not None:
            futures.append(self._read_future)
            self._read_future = None
        futures += [future for _, future in self._write_futures]
        self._write_futures.clear()
        if self._connect_future is not None:
            futures.append(self._connect_future)
            self._connect_future = None
        for future in futures:
            if not future.done():
                future.set_exception(StreamClosedError(real_error=self.error))
            # Reference the exception to silence warnings. Annoyingly,
            # this raises if the future was cancelled, but just
            # returns any other error.
            try:
                future.exception()
            except asyncio.CancelledError:
                pass
        if self._ssl_connect_future is not None:
            # _ssl_connect_future expects to see the real exception (typically
            # an ssl.SSLError), not just StreamClosedError.
            if not self._ssl_connect_future.done():
                if self.error is not None:
                    self._ssl_connect_future.set_exception(self.error)
                else:
                    self._ssl_connect_future.set_exception(StreamClosedError())
            self._ssl_connect_future.exception()
            self._ssl_connect_future = None
        if self._close_callback is not None:
            cb = self._close_callback
            self._close_callback = None
            self.io_loop.add_callback(cb)
        # Clear the buffers so they can be cleared immediately even
        # if the IOStream object is kept alive by a reference cycle.
        # TODO: Clear the read buffer too; it currently breaks some tests.
        self._write_buffer = None  # type: ignore

    def reading(self) -> bool:
        """Returns ``True`` if we are currently reading from the stream."""
        return self._read_future is not None

    def writing(self) -> bool:
        """Returns ``True`` if we are currently writing to the stream."""
        return bool(self._write_buffer)

    def closed(self) -> bool:
        """Returns ``True`` if the stream has been closed."""
        return self._closed

    def set_nodelay(self, value: bool) -> None:
        """Sets the no-delay flag for this stream.

        By default, data written to TCP streams may be held for a time
        to make the most efficient use of bandwidth (according to
        Nagle's algorithm).  The no-delay flag requests that data be
        written as soon as possible, even if doing so would consume
        additional bandwidth.

        This flag is currently defined only for TCP-based ``IOStreams``.

        .. versionadded:: 3.1
        """
        pass

    def _handle_connect(self) -> None:
        raise NotImplementedError()

    def _handle_events(self, fd: Union[int, ioloop._Selectable], events: int) -> None:
        if self.closed():
            gen_log.warning("Got events for closed stream %s", fd)
            return
        try:
            if self._connecting:
                # Most IOLoops will report a write failed connect
                # with the WRITE event, but SelectIOLoop reports a
                # READ as well so we must check for connecting before
                # either.
                self._handle_connect()
            if self.closed():
                return
            if events & self.io_loop.READ:
                self._handle_read()
            if self.closed():
                return
            if events & self.io_loop.WRITE:
                self._handle_write()
            if self.closed():
                return
            if events & self.io_loop.ERROR:
                self.error = self.get_fd_error()
                # We may have queued up a user callback in _handle_read or
                # _handle_write, so don't close the IOStream until those
                # callbacks have had a chance to run.
                self.io_loop.add_callback(self.close)
                return
            state = self.io_loop.ERROR
            if self.reading():
                state |= self.io_loop.READ
            if self.writing():
                state |= self.io_loop.WRITE
            if state == self.io_loop.ERROR and self._read_buffer_size == 0:
                # If the connection is idle, listen for reads too so
                # we can tell if the connection is closed.  If there is
                # data in the read buffer we won't run the close callback
                # yet anyway, so we don't need to listen in this case.
                state |= self.io_loop.READ
            if state != self._state:
                assert (
                    self._state is not None
                ), "shouldn't happen: _handle_events without self._state"
                self._state = state
                self.io_loop.update_handler(self.fileno(), self._state)
        except UnsatisfiableReadError as e:
            gen_log.info("Unsatisfiable read, closing connection: %s" % e)
            self.close(exc_info=e)
        except Exception as e:
            gen_log.error("Uncaught exception, closing connection.", exc_info=True)
            self.close(exc_info=e)
            raise

    def _read_to_buffer_loop(self) -> Optional[int]:
        # This method is called from _handle_read and _try_inline_read.
        if self._read_bytes is not None:
            target_bytes = self._read_bytes  # type: Optional[int]
        elif self._read_max_bytes is not None:
            target_bytes = self._read_max_bytes
        elif self.reading():
            # For read_until without max_bytes, or
            # read_until_close, read as much as we can before
            # scanning for the delimiter.
            target_bytes = None
        else:
            target_bytes = 0
        next_find_pos = 0
        while not self.closed():
            # Read from the socket until we get EWOULDBLOCK or equivalent.
            # SSL sockets do some internal buffering, and if the data is
            # sitting in the SSL object's buffer select() and friends
            # can't see it; the only way to find out if it's there is to
            # try to read it.
            if self._read_to_buffer() == 0:
                break

            # If we've read all the bytes we can use, break out of
            # this loop.

            # If we've reached target_bytes, we know we're done.
            if target_bytes is not None and self._read_buffer_size >= target_bytes:
                break

            # Otherwise, we need to call the more expensive find_read_pos.
            # It's inefficient to do this on every read, so instead
            # do it on the first read and whenever the read buffer
            # size has doubled.
            if self._read_buffer_size >= next_find_pos:
                pos = self._find_read_pos()
                if pos is not None:
                    return pos
                next_find_pos = self._read_buffer_size * 2
        return self._find_read_pos()

    def _handle_read(self) -> None:
        try:
            pos = self._read_to_buffer_loop()
        except UnsatisfiableReadError:
            raise
        except asyncio.CancelledError:
            raise
        except Exception as e:
            gen_log.warning("error on read: %s" % e)
            self.close(exc_info=e)
            return
        if pos is not None:
            self._read_from_buffer(pos)

    def _start_read(self) -> Future:
        if self._read_future is not None:
            # It is an error to start a read while a prior read is unresolved.
            # However, if the prior read is unresolved because the stream was
            # closed without satisfying it, it's better to raise
            # StreamClosedError instead of AssertionError. In particular, this
            # situation occurs in harmless situations in http1connection.py and
            # an AssertionError would be logged noisily.
            #
            # On the other hand, it is legal to start a new read while the
            # stream is closed, in case the read can be satisfied from the
            # read buffer. So we only want to check the closed status of the
            # stream if we need to decide what kind of error to raise for
            # "already reading".
            #
            # These conditions have proven difficult to test; we have no
            # unittests that reliably verify this behavior so be careful
            # when making changes here. See #2651 and #2719.
            self._check_closed()
            assert self._read_future is None, "Already reading"
        self._read_future = Future()
        return self._read_future

    def _finish_read(self, size: int) -> None:
        if self._user_read_buffer:
            self._read_buffer = self._after_user_read_buffer or bytearray()
            self._after_user_read_buffer = None
            self._read_buffer_size = len(self._read_buffer)
            self._user_read_buffer = False
            result = size  # type: Union[int, bytes]
        else:
            result = self._consume(size)
        if self._read_future is not None:
            future = self._read_future
            self._read_future = None
            future_set_result_unless_cancelled(future, result)
        self._maybe_add_error_listener()

    def _try_inline_read(self) -> None:
        """Attempt to complete the current read operation from buffered data.

        If the read can be completed without blocking, schedules the
        read callback on the next IOLoop iteration; otherwise starts
        listening for reads on the socket.
        """
        # See if we've already got the data from a previous read
        pos = self._find_read_pos()
        if pos is not None:
            self._read_from_buffer(pos)
            return
        self._check_closed()
        pos = self._read_to_buffer_loop()
        if pos is not None:
            self._read_from_buffer(pos)
            return
        # We couldn't satisfy the read inline, so make sure we're
        # listening for new data unless the stream is closed.
        if not self.closed():
            self._add_io_state(ioloop.IOLoop.READ)

    def _read_to_buffer(self) -> Optional[int]:
        """Reads from the socket and appends the result to the read buffer.

        Returns the number of bytes read.  Returns 0 if there is nothing
        to read (i.e. the read returns EWOULDBLOCK or equivalent).  On
        error closes the socket and raises an exception.
        """
        try:
            while True:
                try:
                    if self._user_read_buffer:
                        buf = memoryview(self._read_buffer)[
                            self._read_buffer_size :
                        ]  # type: Union[memoryview, bytearray]
                    else:
                        buf = bytearray(self.read_chunk_size)
                    bytes_read = self.read_from_fd(buf)
                except (socket.error, IOError, OSError) as e:
                    # ssl.SSLError is a subclass of socket.error
                    if self._is_connreset(e):
                        # Treat ECONNRESET as a connection close rather than
                        # an error to minimize log spam  (the exception will
                        # be available on self.error for apps that care).
                        self.close(exc_info=e)
                        return None
                    self.close(exc_info=e)
                    raise
                break
            if bytes_read is None:
                return 0
            elif bytes_read == 0:
                self.close()
                return 0
            if not self._user_read_buffer:
                self._read_buffer += memoryview(buf)[:bytes_read]
            self._read_buffer_size += bytes_read
        finally:
            # Break the reference to buf so we don't waste a chunk's worth of
            # memory in case an exception hangs on to our stack frame.
            del buf
        if self._read_buffer_size > self.max_buffer_size:
            gen_log.error("Reached maximum read buffer size")
            self.close()
            raise StreamBufferFullError("Reached maximum read buffer size")
        return bytes_read

    def _read_from_buffer(self, pos: int) -> None:
        """Attempts to complete the currently-pending read from the buffer.

        The argument is either a position in the read buffer or None,
        as returned by _find_read_pos.
        """
        self._read_bytes = self._read_delimiter = self._read_regex = None
        self._read_partial = False
        self._finish_read(pos)

    def _find_read_pos(self) -> Optional[int]:
        """Attempts to find a position in the read buffer that satisfies
        the currently-pending read.

        Returns a position in the buffer if the current read can be satisfied,
        or None if it cannot.
        """
        if self._read_bytes is not None and (
            self._read_buffer_size >= self._read_bytes
            or (self._read_partial and self._read_buffer_size > 0)
        ):
            num_bytes = min(self._read_bytes, self._read_buffer_size)
            return num_bytes
        elif self._read_delimiter is not None:
            # Multi-byte delimiters (e.g. '\r\n') may straddle two
            # chunks in the read buffer, so we can't easily find them
            # without collapsing the buffer.  However, since protocols
            # using delimited reads (as opposed to reads of a known
            # length) tend to be "line" oriented, the delimiter is likely
            # to be in the first few chunks.  Merge the buffer gradually
            # since large merges are relatively expensive and get undone in
            # _consume().
            if self._read_buffer:
                loc = self._read_buffer.find(self._read_delimiter)
                if loc != -1:
                    delimiter_len = len(self._read_delimiter)
                    self._check_max_bytes(self._read_delimiter, loc + delimiter_len)
                    return loc + delimiter_len
                self._check_max_bytes(self._read_delimiter, self._read_buffer_size)
        elif self._read_regex is not None:
            if self._read_buffer:
                m = self._read_regex.search(self._read_buffer)
                if m is not None:
                    loc = m.end()
                    self._check_max_bytes(self._read_regex, loc)
                    return loc
                self._check_max_bytes(self._read_regex, self._read_buffer_size)
        return None

    def _check_max_bytes(self, delimiter: Union[bytes, Pattern], size: int) -> None:
        if self._read_max_bytes is not None and size > self._read_max_bytes:
            raise UnsatisfiableReadError(
                "delimiter %r not found within %d bytes"
                % (delimiter, self._read_max_bytes)
            )

    def _handle_write(self) -> None:
        while True:
            size = len(self._write_buffer)
            if not size:
                break
            assert size > 0
            try:
                if _WINDOWS:
                    # On windows, socket.send blows up if given a
                    # write buffer that's too large, instead of just
                    # returning the number of bytes it was able to
                    # process.  Therefore we must not call socket.send
                    # with more than 128KB at a time.
                    size = 128 * 1024

                num_bytes = self.write_to_fd(self._write_buffer.peek(size))
                if num_bytes == 0:
                    break
                self._write_buffer.advance(num_bytes)
                self._total_write_done_index += num_bytes
            except BlockingIOError:
                break
            except (socket.error, IOError, OSError) as e:
                if not self._is_connreset(e):
                    # Broken pipe errors are usually caused by connection
                    # reset, and its better to not log EPIPE errors to
                    # minimize log spam
                    gen_log.warning("Write error on %s: %s", self.fileno(), e)
                self.close(exc_info=e)
                return

        while self._write_futures:
            index, future = self._write_futures[0]
            if index > self._total_write_done_index:
                break
            self._write_futures.popleft()
            future_set_result_unless_cancelled(future, None)

    def _consume(self, loc: int) -> bytes:
        # Consume loc bytes from the read buffer and return them
        if loc == 0:
            return b""
        assert loc <= self._read_buffer_size
        # Slice the bytearray buffer into bytes, without intermediate copying
        b = (memoryview(self._read_buffer)[:loc]).tobytes()
        self._read_buffer_size -= loc
        del self._read_buffer[:loc]
        return b

    def _check_closed(self) -> None:
        if self.closed():
            raise StreamClosedError(real_error=self.error)

    def _maybe_add_error_listener(self) -> None:
        # This method is part of an optimization: to detect a connection that
        # is closed when we're not actively reading or writing, we must listen
        # for read events.  However, it is inefficient to do this when the
        # connection is first established because we are going to read or write
        # immediately anyway.  Instead, we insert checks at various times to
        # see if the connection is idle and add the read listener then.
        if self._state is None or self._state == ioloop.IOLoop.ERROR:
            if (
                not self.closed()
                and self._read_buffer_size == 0
                and self._close_callback is not None
            ):
                self._add_io_state(ioloop.IOLoop.READ)

    def _add_io_state(self, state: int) -> None:
        """Adds `state` (IOLoop.{READ,WRITE} flags) to our event handler.

        Implementation notes: Reads and writes have a fast path and a
        slow path.  The fast path reads synchronously from socket
        buffers, while the slow path uses `_add_io_state` to schedule
        an IOLoop callback.

        To detect closed connections, we must have called
        `_add_io_state` at some point, but we want to delay this as
        much as possible so we don't have to set an `IOLoop.ERROR`
        listener that will be overwritten by the next slow-path
        operation. If a sequence of fast-path ops do not end in a
        slow-path op, (e.g. for an @asynchronous long-poll request),
        we must add the error handler.

        TODO: reevaluate this now that callbacks are gone.

        """
        if self.closed():
            # connection has been closed, so there can be no future events
            return
        if self._state is None:
            self._state = ioloop.IOLoop.ERROR | state
            self.io_loop.add_handler(self.fileno(), self._handle_events, self._state)
        elif not self._state & state:
            self._state = self._state | state
            self.io_loop.update_handler(self.fileno(), self._state)

    def _is_connreset(self, exc: BaseException) -> bool:
        """Return ``True`` if exc is ECONNRESET or equivalent.

        May be overridden in subclasses.
        """
        return (
            isinstance(exc, (socket.error, IOError))
            and errno_from_exception(exc) in _ERRNO_CONNRESET
        )


class IOStream(BaseIOStream):
    r"""Socket-based `IOStream` implementation.

    This class supports the read and write methods from `BaseIOStream`
    plus a `connect` method.

    The ``socket`` parameter may either be connected or unconnected.
    For server operations the socket is the result of calling
    `socket.accept <socket.socket.accept>`.  For client operations the
    socket is created with `socket.socket`, and may either be
    connected before passing it to the `IOStream` or connected with
    `IOStream.connect`.

    A very simple (and broken) HTTP client using this class:

    .. testcode::

        import socket
        import tornado

        async def main():
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
            stream = tornado.iostream.IOStream(s)
            await stream.connect(("friendfeed.com", 80))
            await stream.write(b"GET / HTTP/1.0\r\nHost: friendfeed.com\r\n\r\n")
            header_data = await stream.read_until(b"\r\n\r\n")
            headers = {}
            for line in header_data.split(b"\r\n"):
                parts = line.split(b":")
                if len(parts) == 2:
                    headers[parts[0].strip()] = parts[1].strip()
            body_data = await stream.read_bytes(int(headers[b"Content-Length"]))
            print(body_data)
            stream.close()

        if __name__ == '__main__':
            asyncio.run(main())

    .. testoutput::
       :hide:

    """

    def __init__(self, socket: socket.socket, *args: Any, **kwargs: Any) -> None:
        self.socket = socket
        self.socket.setblocking(False)
        super().__init__(*args, **kwargs)

    def fileno(self) -> Union[int, ioloop._Selectable]:
        return self.socket

    def close_fd(self) -> None:
        self.socket.close()
        self.socket = None  # type: ignore

    def get_fd_error(self) -> Optional[Exception]:
        errno = self.socket.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
        return socket.error(errno, os.strerror(errno))

    def read_from_fd(self, buf: Union[bytearray, memoryview]) -> Optional[int]:
        try:
            return self.socket.recv_into(buf, len(buf))
        except BlockingIOError:
            return None
        finally:
            del buf

    def write_to_fd(self, data: memoryview) -> int:
        try:
            return self.socket.send(data)  # type: ignore
        finally:
            # Avoid keeping to data, which can be a memoryview.
            # See https://github.com/tornadoweb/tornado/pull/2008
            del data

    def connect(
        self: _IOStreamType, address: Any, server_hostname: Optional[str] = None
    ) -> "Future[_IOStreamType]":
        """Connects the socket to a remote address without blocking.

        May only be called if the socket passed to the constructor was
        not previously connected.  The address parameter is in the
        same format as for `socket.connect <socket.socket.connect>` for
        the type of socket passed to the IOStream constructor,
        e.g. an ``(ip, port)`` tuple.  Hostnames are accepted here,
        but will be resolved synchronously and block the IOLoop.
        If you have a hostname instead of an IP address, the `.TCPClient`
        class is recommended instead of calling this method directly.
        `.TCPClient` will do asynchronous DNS resolution and handle
        both IPv4 and IPv6.

        If ``callback`` is specified, it will be called with no
        arguments when the connection is completed; if not this method
        returns a `.Future` (whose result after a successful
        connection will be the stream itself).

        In SSL mode, the ``server_hostname`` parameter will be used
        for certificate validation (unless disabled in the
        ``ssl_options``) and SNI (if supported; requires Python
        2.7.9+).

        Note that it is safe to call `IOStream.write
        <BaseIOStream.write>` while the connection is pending, in
        which case the data will be written as soon as the connection
        is ready.  Calling `IOStream` read methods before the socket is
        connected works on some platforms but is non-portable.

        .. versionchanged:: 4.0
            If no callback is given, returns a `.Future`.

        .. versionchanged:: 4.2
           SSL certificates are validated by default; pass
           ``ssl_options=dict(cert_reqs=ssl.CERT_NONE)`` or a
           suitably-configured `ssl.SSLContext` to the
           `SSLIOStream` constructor to disable.

        .. versionchanged:: 6.0

           The ``callback`` argument was removed. Use the returned
           `.Future` instead.

        """
        self._connecting = True
        future = Future()  # type: Future[_IOStreamType]
        self._connect_future = typing.cast("Future[IOStream]", future)
        try:
            self.socket.connect(address)
        except BlockingIOError:
            # In non-blocking mode we expect connect() to raise an
            # exception with EINPROGRESS or EWOULDBLOCK.
            pass
        except socket.error as e:
            # On freebsd, other errors such as ECONNREFUSED may be
            # returned immediately when attempting to connect to
            # localhost, so handle them the same way as an error
            # reported later in _handle_connect.
            if future is None:
                gen_log.warning("Connect error on fd %s: %s", self.socket.fileno(), e)
            self.close(exc_info=e)
            return future
        self._add_io_state(self.io_loop.WRITE)
        return future

    def start_tls(
        self,
        server_side: bool,
        ssl_options: Optional[Union[Dict[str, Any], ssl.SSLContext]] = None,
        server_hostname: Optional[str] = None,
    ) -> Awaitable["SSLIOStream"]:
        """Convert this `IOStream` to an `SSLIOStream`.

        This enables protocols that begin in clear-text mode and
        switch to SSL after some initial negotiation (such as the
        ``STARTTLS`` extension to SMTP and IMAP).

        This method cannot be used if there are outstanding reads
        or writes on the stream, or if there is any data in the
        IOStream's buffer (data in the operating system's socket
        buffer is allowed).  This means it must generally be used
        immediately after reading or writing the last clear-text
        data.  It can also be used immediately after connecting,
        before any reads or writes.

        The ``ssl_options`` argument may be either an `ssl.SSLContext`
        object or a dictionary of keyword arguments for the
        `ssl.wrap_socket` function.  The ``server_hostname`` argument
        will be used for certificate validation unless disabled
        in the ``ssl_options``.

        This method returns a `.Future` whose result is the new
        `SSLIOStream`.  After this method has been called,
        any other operation on the original stream is undefined.

        If a close callback is defined on this stream, it will be
        transferred to the new stream.

        .. versionadded:: 4.0

        .. versionchanged:: 4.2
           SSL certificates are validated by default; pass
           ``ssl_options=dict(cert_reqs=ssl.CERT_NONE)`` or a
           suitably-configured `ssl.SSLContext` to disable.
        """
        if (
            self._read_future
            or self._write_futures
            or self._connect_future
            or self._closed
            or self._read_buffer
            or self._write_buffer
        ):
            raise ValueError("IOStream is not idle; cannot convert to SSL")
        if ssl_options is None:
            if server_side:
                ssl_options = _server_ssl_defaults
            else:
                ssl_options = _client_ssl_defaults

        socket = self.socket
        self.io_loop.remove_handler(socket)
        self.socket = None  # type: ignore
        socket = ssl_wrap_socket(
            socket,
            ssl_options,
            server_hostname=server_hostname,
            server_side=server_side,
            do_handshake_on_connect=False,
        )
        orig_close_callback = self._close_callback
        self._close_callback = None

        future = Future()  # type: Future[SSLIOStream]
        ssl_stream = SSLIOStream(socket, ssl_options=ssl_options)
        ssl_stream.set_close_callback(orig_close_callback)
        ssl_stream._ssl_connect_future = future
        ssl_stream.max_buffer_size = self.max_buffer_size
        ssl_stream.read_chunk_size = self.read_chunk_size
        return future

    def _handle_connect(self) -> None:
        try:
            err = self.socket.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
        except socket.error as e:
            # Hurd doesn't allow SO_ERROR for loopback sockets because all
            # errors for such sockets are reported synchronously.
            if errno_from_exception(e) == errno.ENOPROTOOPT:
                err = 0
        if err != 0:
            self.error = socket.error(err, os.strerror(err))
            # IOLoop implementations may vary: some of them return
            # an error state before the socket becomes writable, so
            # in that case a connection failure would be handled by the
            # error path in _handle_events instead of here.
            if self._connect_future is None:
                gen_log.warning(
                    "Connect error on fd %s: %s",
                    self.socket.fileno(),
                    errno.errorcode[err],
                )
            self.close()
            return
        if self._connect_future is not None:
            future = self._connect_future
            self._connect_future = None
            future_set_result_unless_cancelled(future, self)
        self._connecting = False

    def set_nodelay(self, value: bool) -> None:
        if self.socket is not None and self.socket.family in (
            socket.AF_INET,
            socket.AF_INET6,
        ):
            try:
                self.socket.setsockopt(
                    socket.IPPROTO_TCP, socket.TCP_NODELAY, 1 if value else 0
                )
            except socket.error as e:
                # Sometimes setsockopt will fail if the socket is closed
                # at the wrong time.  This can happen with HTTPServer
                # resetting the value to ``False`` between requests.
                if e.errno != errno.EINVAL and not self._is_connreset(e):
                    raise


class SSLIOStream(IOStream):
    """A utility class to write to and read from a non-blocking SSL socket.

    If the socket passed to the constructor is already connected,
    it should be wrapped with::

        ssl.wrap_socket(sock, do_handshake_on_connect=False, **kwargs)

    before constructing the `SSLIOStream`.  Unconnected sockets will be
    wrapped when `IOStream.connect` is finished.
    """

    socket = None  # type: ssl.SSLSocket

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """The ``ssl_options`` keyword argument may either be an
        `ssl.SSLContext` object or a dictionary of keywords arguments
        for `ssl.wrap_socket`
        """
        self._ssl_options = kwargs.pop("ssl_options", _client_ssl_defaults)
        super().__init__(*args, **kwargs)
        self._ssl_accepting = True
        self._handshake_reading = False
        self._handshake_writing = False
        self._server_hostname = None  # type: Optional[str]

        # If the socket is already connected, attempt to start the handshake.
        try:
            self.socket.getpeername()
        except socket.error:
            pass
        else:
            # Indirectly start the handshake, which will run on the next
            # IOLoop iteration and then the real IO state will be set in
            # _handle_events.
            self._add_io_state(self.io_loop.WRITE)

    def reading(self) -> bool:
        return self._handshake_reading or super().reading()

    def writing(self) -> bool:
        return self._handshake_writing or super().writing()

    def _do_ssl_handshake(self) -> None:
        # Based on code from test_ssl.py in the python stdlib
        try:
            self._handshake_reading = False
            self._handshake_writing = False
            self.socket.do_handshake()
        except ssl.SSLError as err:
            if err.args[0] == ssl.SSL_ERROR_WANT_READ:
                self._handshake_reading = True
                return
            elif err.args[0] == ssl.SSL_ERROR_WANT_WRITE:
                self._handshake_writing = True
                return
            elif err.args[0] in (ssl.SSL_ERROR_EOF, ssl.SSL_ERROR_ZERO_RETURN):
                return self.close(exc_info=err)
            elif err.args[0] == ssl.SSL_ERROR_SSL:
                try:
                    peer = self.socket.getpeername()
                except Exception:
                    peer = "(not connected)"
                gen_log.warning(
                    "SSL Error on %s %s: %s", self.socket.fileno(), peer, err
                )
                return self.close(exc_info=err)
            raise
        except ssl.CertificateError as err:
            # CertificateError can happen during handshake (hostname
            # verification) and should be passed to user. Starting
            # in Python 3.7, this error is a subclass of SSLError
            # and will be handled by the previous block instead.
            return self.close(exc_info=err)
        except socket.error as err:
            # Some port scans (e.g. nmap in -sT mode) have been known
            # to cause do_handshake to raise EBADF and ENOTCONN, so make
            # those errors quiet as well.
            # https://groups.google.com/forum/?fromgroups#!topic/python-tornado/ApucKJat1_0
            # Errno 0 is also possible in some cases (nc -z).
            # https://github.com/tornadoweb/tornado/issues/2504
            if self._is_connreset(err) or err.args[0] in (
                0,
                errno.EBADF,
                errno.ENOTCONN,
            ):
                return self.close(exc_info=err)
            raise
        except AttributeError as err:
            # On Linux, if the connection was reset before the call to
            # wrap_socket, do_handshake will fail with an
            # AttributeError.
            return self.close(exc_info=err)
        else:
            self._ssl_accepting = False
            if not self._verify_cert(self.socket.getpeercert()):
                self.close()
                return
            self._finish_ssl_connect()

    def _finish_ssl_connect(self) -> None:
        if self._ssl_connect_future is not None:
            future = self._ssl_connect_future
            self._ssl_connect_future = None
            future_set_result_unless_cancelled(future, self)

    def _verify_cert(self, peercert: Any) -> bool:
        """Returns ``True`` if peercert is valid according to the configured
        validation mode and hostname.

        The ssl handshake already tested the certificate for a valid
        CA signature; the only thing that remains is to check
        the hostname.
        """
        if isinstance(self._ssl_options, dict):
            verify_mode = self._ssl_options.get("cert_reqs", ssl.CERT_NONE)
        elif isinstance(self._ssl_options, ssl.SSLContext):
            verify_mode = self._ssl_options.verify_mode
        assert verify_mode in (ssl.CERT_NONE, ssl.CERT_REQUIRED, ssl.CERT_OPTIONAL)
        if verify_mode == ssl.CERT_NONE or self._server_hostname is None:
            return True
        cert = self.socket.getpeercert()
        if cert is None and verify_mode == ssl.CERT_REQUIRED:
            gen_log.warning("No SSL certificate given")
            return False
        try:
            ssl.match_hostname(peercert, self._server_hostname)
        except ssl.CertificateError as e:
            gen_log.warning("Invalid SSL certificate: %s" % e)
            return False
        else:
            return True

    def _handle_read(self) -> None:
        if self._ssl_accepting:
            self._do_ssl_handshake()
            return
        super()._handle_read()

    def _handle_write(self) -> None:
        if self._ssl_accepting:
            self._do_ssl_handshake()
            return
        super()._handle_write()

    def connect(
        self, address: Tuple, server_hostname: Optional[str] = None
    ) -> "Future[SSLIOStream]":
        self._server_hostname = server_hostname
        # Ignore the result of connect(). If it fails,
        # wait_for_handshake will raise an error too. This is
        # necessary for the old semantics of the connect callback
        # (which takes no arguments). In 6.0 this can be refactored to
        # be a regular coroutine.
        # TODO: This is trickier than it looks, since if write()
        # is called with a connect() pending, we want the connect
        # to resolve before the write. Or do we care about this?
        # (There's a test for it, but I think in practice users
        # either wait for the connect before performing a write or
        # they don't care about the connect Future at all)
        fut = super().connect(address)
        fut.add_done_callback(lambda f: f.exception())
        return self.wait_for_handshake()

    def _handle_connect(self) -> None:
        # Call the superclass method to check for errors.
        super()._handle_connect()
        if self.closed():
            return
        # When the connection is complete, wrap the socket for SSL
        # traffic.  Note that we do this by overriding _handle_connect
        # instead of by passing a callback to super().connect because
        # user callbacks are enqueued asynchronously on the IOLoop,
        # but since _handle_events calls _handle_connect immediately
        # followed by _handle_write we need this to be synchronous.
        #
        # The IOLoop will get confused if we swap out self.socket while the
        # fd is registered, so remove it now and re-register after
        # wrap_socket().
        self.io_loop.remove_handler(self.socket)
        old_state = self._state
        assert old_state is not None
        self._state = None
        self.socket = ssl_wrap_socket(
            self.socket,
            self._ssl_options,
            server_hostname=self._server_hostname,
            do_handshake_on_connect=False,
            server_side=False,
        )
        self._add_io_state(old_state)

    def wait_for_handshake(self) -> "Future[SSLIOStream]":
        """Wait for the initial SSL handshake to complete.

        If a ``callback`` is given, it will be called with no
        arguments once the handshake is complete; otherwise this
        method returns a `.Future` which will resolve to the
        stream itself after the handshake is complete.

        Once the handshake is complete, information such as
        the peer's certificate and NPN/ALPN selections may be
        accessed on ``self.socket``.

        This method is intended for use on server-side streams
        or after using `IOStream.start_tls`; it should not be used
        with `IOStream.connect` (which already waits for the
        handshake to complete). It may only be called once per stream.

        .. versionadded:: 4.2

        .. versionchanged:: 6.0

           The ``callback`` argument was removed. Use the returned
           `.Future` instead.

        """
        if self._ssl_connect_future is not None:
            raise RuntimeError("Already waiting")
        future = self._ssl_connect_future = Future()
        if not self._ssl_accepting:
            self._finish_ssl_connect()
        return future

    def write_to_fd(self, data: memoryview) -> int:
        # clip buffer size at 1GB since SSL sockets only support upto 2GB
        # this change in behaviour is transparent, since the function is
        # already expected to (possibly) write less than the provided buffer
        if len(data) >> 30:
            data = memoryview(data)[: 1 << 30]
        try:
            return self.socket.send(data)  # type: ignore
        except ssl.SSLError as e:
            if e.args[0] == ssl.SSL_ERROR_WANT_WRITE:
                # In Python 3.5+, SSLSocket.send raises a WANT_WRITE error if
                # the socket is not writeable; we need to transform this into
                # an EWOULDBLOCK socket.error or a zero return value,
                # either of which will be recognized by the caller of this
                # method. Prior to Python 3.5, an unwriteable socket would
                # simply return 0 bytes written.
                return 0
            raise
        finally:
            # Avoid keeping to data, which can be a memoryview.
            # See https://github.com/tornadoweb/tornado/pull/2008
            del data

    def read_from_fd(self, buf: Union[bytearray, memoryview]) -> Optional[int]:
        try:
            if self._ssl_accepting:
                # If the handshake hasn't finished yet, there can't be anything
                # to read (attempting to read may or may not raise an exception
                # depending on the SSL version)
                return None
            # clip buffer size at 1GB since SSL sockets only support upto 2GB
            # this change in behaviour is transparent, since the function is
            # already expected to (possibly) read less than the provided buffer
            if len(buf) >> 30:
                buf = memoryview(buf)[: 1 << 30]
            try:
                return self.socket.recv_into(buf, len(buf))
            except ssl.SSLError as e:
                # SSLError is a subclass of socket.error, so this except
                # block must come first.
                if e.args[0] == ssl.SSL_ERROR_WANT_READ:
                    return None
                else:
                    raise
            except BlockingIOError:
                return None
        finally:
            del buf

    def _is_connreset(self, e: BaseException) -> bool:
        if isinstance(e, ssl.SSLError) and e.args[0] == ssl.SSL_ERROR_EOF:
            return True
        return super()._is_connreset(e)


class PipeIOStream(BaseIOStream):
    """Pipe-based `IOStream` implementation.

    The constructor takes an integer file descriptor (such as one returned
    by `os.pipe`) rather than an open file object.  Pipes are generally
    one-way, so a `PipeIOStream` can be used for reading or writing but not
    both.

    ``PipeIOStream`` is only available on Unix-based platforms.
    """

    def __init__(self, fd: int, *args: Any, **kwargs: Any) -> None:
        self.fd = fd
        self._fio = io.FileIO(self.fd, "r+")
        if sys.platform == "win32":
            # The form and placement of this assertion is important to mypy.
            # A plain assert statement isn't recognized here. If the assertion
            # were earlier it would worry that the attributes of self aren't
            # set on windows. If it were missing it would complain about
            # the absence of the set_blocking function.
            raise AssertionError("PipeIOStream is not supported on Windows")
        os.set_blocking(fd, False)
        super().__init__(*args, **kwargs)

    def fileno(self) -> int:
        return self.fd

    def close_fd(self) -> None:
        self._fio.close()

    def write_to_fd(self, data: memoryview) -> int:
        try:
            return os.write(self.fd, data)  # type: ignore
        finally:
            # Avoid keeping to data, which can be a memoryview.
            # See https://github.com/tornadoweb/tornado/pull/2008
            del data

    def read_from_fd(self, buf: Union[bytearray, memoryview]) -> Optional[int]:
        try:
            return self._fio.readinto(buf)  # type: ignore
        except (IOError, OSError) as e:
            if errno_from_exception(e) == errno.EBADF:
                # If the writing half of a pipe is closed, select will
                # report it as readable but reads will fail with EBADF.
                self.close(exc_info=e)
                return None
            else:
                raise
        finally:
            del buf


def doctests() -> Any:
    import doctest

    return doctest.DocTestSuite()
