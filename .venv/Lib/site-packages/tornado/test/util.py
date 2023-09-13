import contextlib
import os
import platform
import socket
import sys
import textwrap
import typing  # noqa: F401
import unittest
import warnings

from tornado.testing import bind_unused_port

skipIfNonUnix = unittest.skipIf(
    os.name != "posix" or sys.platform == "cygwin", "non-unix platform"
)

# travis-ci.org runs our tests in an overworked virtual machine, which makes
# timing-related tests unreliable.
skipOnTravis = unittest.skipIf(
    "TRAVIS" in os.environ, "timing tests unreliable on travis"
)

# Set the environment variable NO_NETWORK=1 to disable any tests that
# depend on an external network.
skipIfNoNetwork = unittest.skipIf("NO_NETWORK" in os.environ, "network access disabled")

skipNotCPython = unittest.skipIf(
    platform.python_implementation() != "CPython", "Not CPython implementation"
)

# Used for tests affected by
# https://bitbucket.org/pypy/pypy/issues/2616/incomplete-error-handling-in
# TODO: remove this after pypy3 5.8 is obsolete.
skipPypy3V58 = unittest.skipIf(
    platform.python_implementation() == "PyPy"
    and sys.version_info > (3,)
    and sys.pypy_version_info < (5, 9),  # type: ignore
    "pypy3 5.8 has buggy ssl module",
)


def _detect_ipv6():
    if not socket.has_ipv6:
        # socket.has_ipv6 check reports whether ipv6 was present at compile
        # time. It's usually true even when ipv6 doesn't work for other reasons.
        return False
    sock = None
    try:
        sock = socket.socket(socket.AF_INET6)
        sock.bind(("::1", 0))
    except socket.error:
        return False
    finally:
        if sock is not None:
            sock.close()
    return True


skipIfNoIPv6 = unittest.skipIf(not _detect_ipv6(), "ipv6 support not present")


def refusing_port():
    """Returns a local port number that will refuse all connections.

    Return value is (cleanup_func, port); the cleanup function
    must be called to free the port to be reused.
    """
    # On travis-ci, port numbers are reassigned frequently. To avoid
    # collisions with other tests, we use an open client-side socket's
    # ephemeral port number to ensure that nothing can listen on that
    # port.
    server_socket, port = bind_unused_port()
    server_socket.setblocking(True)
    client_socket = socket.socket()
    client_socket.connect(("127.0.0.1", port))
    conn, client_addr = server_socket.accept()
    conn.close()
    server_socket.close()
    return (client_socket.close, client_addr[1])


def exec_test(caller_globals, caller_locals, s):
    """Execute ``s`` in a given context and return the result namespace.

    Used to define functions for tests in particular python
    versions that would be syntax errors in older versions.
    """
    # Flatten the real global and local namespace into our fake
    # globals: it's all global from the perspective of code defined
    # in s.
    global_namespace = dict(caller_globals, **caller_locals)  # type: ignore
    local_namespace = {}  # type: typing.Dict[str, typing.Any]
    exec(textwrap.dedent(s), global_namespace, local_namespace)
    return local_namespace


def subTest(test, *args, **kwargs):
    """Compatibility shim for unittest.TestCase.subTest.

    Usage: ``with tornado.test.util.subTest(self, x=x):``
    """
    try:
        subTest = test.subTest  # py34+
    except AttributeError:
        subTest = contextlib.contextmanager(lambda *a, **kw: (yield))
    return subTest(*args, **kwargs)


@contextlib.contextmanager
def ignore_deprecation():
    """Context manager to ignore deprecation warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        yield
