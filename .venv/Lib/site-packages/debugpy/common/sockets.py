# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root
# for license information.

import socket
import sys
import threading

from debugpy.common import log
from debugpy.common.util import hide_thread_from_debugger


def create_server(host, port=0, backlog=socket.SOMAXCONN, timeout=None):
    """Return a local server socket listening on the given port."""

    assert backlog > 0
    if host is None:
        host = "127.0.0.1"
    if port is None:
        port = 0

    try:
        server = _new_sock()
        if port != 0:
            # If binding to a specific port, make sure that the user doesn't have
            # to wait until the OS times out the socket to be able to use that port
            # again.if the server or the adapter crash or are force-killed.
            if sys.platform == "win32":
                server.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            else:
                try:
                    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                except (AttributeError, OSError):  # pragma: no cover
                    pass  # Not available everywhere
        server.bind((host, port))
        if timeout is not None:
            server.settimeout(timeout)
        server.listen(backlog)
    except Exception:  # pragma: no cover
        server.close()
        raise
    return server


def create_client():
    """Return a client socket that may be connected to a remote address."""
    return _new_sock()


def _new_sock():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP)

    # Set TCP keepalive on an open socket.
    # It activates after 1 second (TCP_KEEPIDLE,) of idleness,
    # then sends a keepalive ping once every 3 seconds (TCP_KEEPINTVL),
    # and closes the connection after 5 failed ping (TCP_KEEPCNT), or 15 seconds
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    except (AttributeError, OSError):  # pragma: no cover
        pass  # May not be available everywhere.
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 1)
    except (AttributeError, OSError):  # pragma: no cover
        pass  # May not be available everywhere.
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 3)
    except (AttributeError, OSError):  # pragma: no cover
        pass  # May not be available everywhere.
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 5)
    except (AttributeError, OSError):  # pragma: no cover
        pass  # May not be available everywhere.
    return sock


def shut_down(sock, how=socket.SHUT_RDWR):
    """Shut down the given socket."""
    sock.shutdown(how)


def close_socket(sock):
    """Shutdown and close the socket."""
    try:
        shut_down(sock)
    except Exception:  # pragma: no cover
        pass
    sock.close()


def serve(name, handler, host, port=0, backlog=socket.SOMAXCONN, timeout=None):
    """Accepts TCP connections on the specified host and port, and invokes the
    provided handler function for every new connection.

    Returns the created server socket.
    """

    assert backlog > 0

    try:
        listener = create_server(host, port, backlog, timeout)
    except Exception:  # pragma: no cover
        log.reraise_exception(
            "Error listening for incoming {0} connections on {1}:{2}:", name, host, port
        )
    host, port = listener.getsockname()
    log.info("Listening for incoming {0} connections on {1}:{2}...", name, host, port)

    def accept_worker():
        while True:
            try:
                sock, (other_host, other_port) = listener.accept()
            except (OSError, socket.error):
                # Listener socket has been closed.
                break

            log.info(
                "Accepted incoming {0} connection from {1}:{2}.",
                name,
                other_host,
                other_port,
            )
            handler(sock)

    thread = threading.Thread(target=accept_worker)
    thread.daemon = True
    hide_thread_from_debugger(thread)
    thread.start()

    return listener
