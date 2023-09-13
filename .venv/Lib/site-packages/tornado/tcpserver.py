#
# Copyright 2011 Facebook
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

"""A non-blocking, single-threaded TCP server."""

import errno
import os
import socket
import ssl

from tornado import gen
from tornado.log import app_log
from tornado.ioloop import IOLoop
from tornado.iostream import IOStream, SSLIOStream
from tornado.netutil import (
    bind_sockets,
    add_accept_handler,
    ssl_wrap_socket,
    _DEFAULT_BACKLOG,
)
from tornado import process
from tornado.util import errno_from_exception

import typing
from typing import Union, Dict, Any, Iterable, Optional, Awaitable

if typing.TYPE_CHECKING:
    from typing import Callable, List  # noqa: F401


class TCPServer(object):
    r"""A non-blocking, single-threaded TCP server.

    To use `TCPServer`, define a subclass which overrides the `handle_stream`
    method. For example, a simple echo server could be defined like this::

      from tornado.tcpserver import TCPServer
      from tornado.iostream import StreamClosedError

      class EchoServer(TCPServer):
          async def handle_stream(self, stream, address):
              while True:
                  try:
                      data = await stream.read_until(b"\n") await
                      stream.write(data)
                  except StreamClosedError:
                      break

    To make this server serve SSL traffic, send the ``ssl_options`` keyword
    argument with an `ssl.SSLContext` object. For compatibility with older
    versions of Python ``ssl_options`` may also be a dictionary of keyword
    arguments for the `ssl.wrap_socket` method.::

       ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
       ssl_ctx.load_cert_chain(os.path.join(data_dir, "mydomain.crt"),
                               os.path.join(data_dir, "mydomain.key"))
       TCPServer(ssl_options=ssl_ctx)

    `TCPServer` initialization follows one of three patterns:

    1. `listen`: single-process::

            async def main():
                server = TCPServer()
                server.listen(8888)
                await asyncio.Event.wait()

            asyncio.run(main())

       While this example does not create multiple processes on its own, when
       the ``reuse_port=True`` argument is passed to ``listen()`` you can run
       the program multiple times to create a multi-process service.

    2. `add_sockets`: multi-process::

            sockets = bind_sockets(8888)
            tornado.process.fork_processes(0)
            async def post_fork_main():
                server = TCPServer()
                server.add_sockets(sockets)
                await asyncio.Event().wait()
            asyncio.run(post_fork_main())

       The `add_sockets` interface is more complicated, but it can be used with
       `tornado.process.fork_processes` to run a multi-process service with all
       worker processes forked from a single parent.  `add_sockets` can also be
       used in single-process servers if you want to create your listening
       sockets in some way other than `~tornado.netutil.bind_sockets`.

       Note that when using this pattern, nothing that touches the event loop
       can be run before ``fork_processes``.

    3. `bind`/`start`: simple **deprecated** multi-process::

            server = TCPServer()
            server.bind(8888)
            server.start(0)  # Forks multiple sub-processes
            IOLoop.current().start()

       This pattern is deprecated because it requires interfaces in the
       `asyncio` module that have been deprecated since Python 3.10. Support for
       creating multiple processes in the ``start`` method will be removed in a
       future version of Tornado.

    .. versionadded:: 3.1
       The ``max_buffer_size`` argument.

    .. versionchanged:: 5.0
       The ``io_loop`` argument has been removed.
    """

    def __init__(
        self,
        ssl_options: Optional[Union[Dict[str, Any], ssl.SSLContext]] = None,
        max_buffer_size: Optional[int] = None,
        read_chunk_size: Optional[int] = None,
    ) -> None:
        self.ssl_options = ssl_options
        self._sockets = {}  # type: Dict[int, socket.socket]
        self._handlers = {}  # type: Dict[int, Callable[[], None]]
        self._pending_sockets = []  # type: List[socket.socket]
        self._started = False
        self._stopped = False
        self.max_buffer_size = max_buffer_size
        self.read_chunk_size = read_chunk_size

        # Verify the SSL options. Otherwise we don't get errors until clients
        # connect. This doesn't verify that the keys are legitimate, but
        # the SSL module doesn't do that until there is a connected socket
        # which seems like too much work
        if self.ssl_options is not None and isinstance(self.ssl_options, dict):
            # Only certfile is required: it can contain both keys
            if "certfile" not in self.ssl_options:
                raise KeyError('missing key "certfile" in ssl_options')

            if not os.path.exists(self.ssl_options["certfile"]):
                raise ValueError(
                    'certfile "%s" does not exist' % self.ssl_options["certfile"]
                )
            if "keyfile" in self.ssl_options and not os.path.exists(
                self.ssl_options["keyfile"]
            ):
                raise ValueError(
                    'keyfile "%s" does not exist' % self.ssl_options["keyfile"]
                )

    def listen(
        self,
        port: int,
        address: Optional[str] = None,
        family: socket.AddressFamily = socket.AF_UNSPEC,
        backlog: int = _DEFAULT_BACKLOG,
        flags: Optional[int] = None,
        reuse_port: bool = False,
    ) -> None:
        """Starts accepting connections on the given port.

        This method may be called more than once to listen on multiple ports.
        `listen` takes effect immediately; it is not necessary to call
        `TCPServer.start` afterwards.  It is, however, necessary to start the
        event loop if it is not already running.

        All arguments have the same meaning as in
        `tornado.netutil.bind_sockets`.

        .. versionchanged:: 6.2

           Added ``family``, ``backlog``, ``flags``, and ``reuse_port``
           arguments to match `tornado.netutil.bind_sockets`.
        """
        sockets = bind_sockets(
            port,
            address=address,
            family=family,
            backlog=backlog,
            flags=flags,
            reuse_port=reuse_port,
        )
        self.add_sockets(sockets)

    def add_sockets(self, sockets: Iterable[socket.socket]) -> None:
        """Makes this server start accepting connections on the given sockets.

        The ``sockets`` parameter is a list of socket objects such as
        those returned by `~tornado.netutil.bind_sockets`.
        `add_sockets` is typically used in combination with that
        method and `tornado.process.fork_processes` to provide greater
        control over the initialization of a multi-process server.
        """
        for sock in sockets:
            self._sockets[sock.fileno()] = sock
            self._handlers[sock.fileno()] = add_accept_handler(
                sock, self._handle_connection
            )

    def add_socket(self, socket: socket.socket) -> None:
        """Singular version of `add_sockets`.  Takes a single socket object."""
        self.add_sockets([socket])

    def bind(
        self,
        port: int,
        address: Optional[str] = None,
        family: socket.AddressFamily = socket.AF_UNSPEC,
        backlog: int = _DEFAULT_BACKLOG,
        flags: Optional[int] = None,
        reuse_port: bool = False,
    ) -> None:
        """Binds this server to the given port on the given address.

        To start the server, call `start`. If you want to run this server in a
        single process, you can call `listen` as a shortcut to the sequence of
        `bind` and `start` calls.

        Address may be either an IP address or hostname.  If it's a hostname,
        the server will listen on all IP addresses associated with the name.
        Address may be an empty string or None to listen on all available
        interfaces.  Family may be set to either `socket.AF_INET` or
        `socket.AF_INET6` to restrict to IPv4 or IPv6 addresses, otherwise both
        will be used if available.

        The ``backlog`` argument has the same meaning as for `socket.listen
        <socket.socket.listen>`. The ``reuse_port`` argument has the same
        meaning as for `.bind_sockets`.

        This method may be called multiple times prior to `start` to listen on
        multiple ports or interfaces.

        .. versionchanged:: 4.4
           Added the ``reuse_port`` argument.

        .. versionchanged:: 6.2
           Added the ``flags`` argument to match `.bind_sockets`.

        .. deprecated:: 6.2
           Use either ``listen()`` or ``add_sockets()`` instead of ``bind()``
           and ``start()``.
        """
        sockets = bind_sockets(
            port,
            address=address,
            family=family,
            backlog=backlog,
            flags=flags,
            reuse_port=reuse_port,
        )
        if self._started:
            self.add_sockets(sockets)
        else:
            self._pending_sockets.extend(sockets)

    def start(
        self, num_processes: Optional[int] = 1, max_restarts: Optional[int] = None
    ) -> None:
        """Starts this server in the `.IOLoop`.

        By default, we run the server in this process and do not fork any
        additional child process.

        If num_processes is ``None`` or <= 0, we detect the number of cores
        available on this machine and fork that number of child
        processes. If num_processes is given and > 1, we fork that
        specific number of sub-processes.

        Since we use processes and not threads, there is no shared memory
        between any server code.

        Note that multiple processes are not compatible with the autoreload
        module (or the ``autoreload=True`` option to `tornado.web.Application`
        which defaults to True when ``debug=True``).
        When using multiple processes, no IOLoops can be created or
        referenced until after the call to ``TCPServer.start(n)``.

        Values of ``num_processes`` other than 1 are not supported on Windows.

        The ``max_restarts`` argument is passed to `.fork_processes`.

        .. versionchanged:: 6.0

           Added ``max_restarts`` argument.

        .. deprecated:: 6.2
           Use either ``listen()`` or ``add_sockets()`` instead of ``bind()``
           and ``start()``.
        """
        assert not self._started
        self._started = True
        if num_processes != 1:
            process.fork_processes(num_processes, max_restarts)
        sockets = self._pending_sockets
        self._pending_sockets = []
        self.add_sockets(sockets)

    def stop(self) -> None:
        """Stops listening for new connections.

        Requests currently in progress may still continue after the
        server is stopped.
        """
        if self._stopped:
            return
        self._stopped = True
        for fd, sock in self._sockets.items():
            assert sock.fileno() == fd
            # Unregister socket from IOLoop
            self._handlers.pop(fd)()
            sock.close()

    def handle_stream(
        self, stream: IOStream, address: tuple
    ) -> Optional[Awaitable[None]]:
        """Override to handle a new `.IOStream` from an incoming connection.

        This method may be a coroutine; if so any exceptions it raises
        asynchronously will be logged. Accepting of incoming connections
        will not be blocked by this coroutine.

        If this `TCPServer` is configured for SSL, ``handle_stream``
        may be called before the SSL handshake has completed. Use
        `.SSLIOStream.wait_for_handshake` if you need to verify the client's
        certificate or use NPN/ALPN.

        .. versionchanged:: 4.2
           Added the option for this method to be a coroutine.
        """
        raise NotImplementedError()

    def _handle_connection(self, connection: socket.socket, address: Any) -> None:
        if self.ssl_options is not None:
            assert ssl, "Python 2.6+ and OpenSSL required for SSL"
            try:
                connection = ssl_wrap_socket(
                    connection,
                    self.ssl_options,
                    server_side=True,
                    do_handshake_on_connect=False,
                )
            except ssl.SSLError as err:
                if err.args[0] == ssl.SSL_ERROR_EOF:
                    return connection.close()
                else:
                    raise
            except socket.error as err:
                # If the connection is closed immediately after it is created
                # (as in a port scan), we can get one of several errors.
                # wrap_socket makes an internal call to getpeername,
                # which may return either EINVAL (Mac OS X) or ENOTCONN
                # (Linux).  If it returns ENOTCONN, this error is
                # silently swallowed by the ssl module, so we need to
                # catch another error later on (AttributeError in
                # SSLIOStream._do_ssl_handshake).
                # To test this behavior, try nmap with the -sT flag.
                # https://github.com/tornadoweb/tornado/pull/750
                if errno_from_exception(err) in (errno.ECONNABORTED, errno.EINVAL):
                    return connection.close()
                else:
                    raise
        try:
            if self.ssl_options is not None:
                stream = SSLIOStream(
                    connection,
                    max_buffer_size=self.max_buffer_size,
                    read_chunk_size=self.read_chunk_size,
                )  # type: IOStream
            else:
                stream = IOStream(
                    connection,
                    max_buffer_size=self.max_buffer_size,
                    read_chunk_size=self.read_chunk_size,
                )

            future = self.handle_stream(stream, address)
            if future is not None:
                IOLoop.current().add_future(
                    gen.convert_yielded(future), lambda f: f.result()
                )
        except Exception:
            app_log.error("Error in connection callback", exc_info=True)
