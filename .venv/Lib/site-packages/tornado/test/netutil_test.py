import errno
import os
import signal
import socket
from subprocess import Popen
import sys
import time
import unittest

from tornado.netutil import (
    BlockingResolver,
    OverrideResolver,
    ThreadedResolver,
    is_valid_ip,
    bind_sockets,
)
from tornado.testing import AsyncTestCase, gen_test, bind_unused_port
from tornado.test.util import skipIfNoNetwork

import typing

if typing.TYPE_CHECKING:
    from typing import List  # noqa: F401

try:
    import pycares  # type: ignore
except ImportError:
    pycares = None
else:
    from tornado.platform.caresresolver import CaresResolver

try:
    import twisted  # type: ignore
    import twisted.names  # type: ignore
except ImportError:
    twisted = None
else:
    from tornado.platform.twisted import TwistedResolver


class _ResolverTestMixin(object):
    resolver = None  # type: typing.Any

    @gen_test
    def test_localhost(self: typing.Any):
        addrinfo = yield self.resolver.resolve("localhost", 80, socket.AF_UNSPEC)
        # Most of the time localhost resolves to either the ipv4 loopback
        # address alone, or ipv4+ipv6. But some versions of pycares will only
        # return the ipv6 version, so we have to check for either one alone.
        self.assertTrue(
            ((socket.AF_INET, ("127.0.0.1", 80)) in addrinfo)
            or ((socket.AF_INET6, ("::1", 80)) in addrinfo),
            f"loopback address not found in {addrinfo}",
        )


# It is impossible to quickly and consistently generate an error in name
# resolution, so test this case separately, using mocks as needed.
class _ResolverErrorTestMixin(object):
    resolver = None  # type: typing.Any

    @gen_test
    def test_bad_host(self: typing.Any):
        with self.assertRaises(IOError):
            yield self.resolver.resolve("an invalid domain", 80, socket.AF_UNSPEC)


def _failing_getaddrinfo(*args):
    """Dummy implementation of getaddrinfo for use in mocks"""
    raise socket.gaierror(errno.EIO, "mock: lookup failed")


@skipIfNoNetwork
class BlockingResolverTest(AsyncTestCase, _ResolverTestMixin):
    def setUp(self):
        super().setUp()
        self.resolver = BlockingResolver()


# getaddrinfo-based tests need mocking to reliably generate errors;
# some configurations are slow to produce errors and take longer than
# our default timeout.
class BlockingResolverErrorTest(AsyncTestCase, _ResolverErrorTestMixin):
    def setUp(self):
        super().setUp()
        self.resolver = BlockingResolver()
        self.real_getaddrinfo = socket.getaddrinfo
        socket.getaddrinfo = _failing_getaddrinfo

    def tearDown(self):
        socket.getaddrinfo = self.real_getaddrinfo
        super().tearDown()


class OverrideResolverTest(AsyncTestCase, _ResolverTestMixin):
    def setUp(self):
        super().setUp()
        mapping = {
            ("google.com", 80): ("1.2.3.4", 80),
            ("google.com", 80, socket.AF_INET): ("1.2.3.4", 80),
            ("google.com", 80, socket.AF_INET6): (
                "2a02:6b8:7c:40c:c51e:495f:e23a:3",
                80,
            ),
        }
        self.resolver = OverrideResolver(BlockingResolver(), mapping)

    @gen_test
    def test_resolve_multiaddr(self):
        result = yield self.resolver.resolve("google.com", 80, socket.AF_INET)
        self.assertIn((socket.AF_INET, ("1.2.3.4", 80)), result)

        result = yield self.resolver.resolve("google.com", 80, socket.AF_INET6)
        self.assertIn(
            (socket.AF_INET6, ("2a02:6b8:7c:40c:c51e:495f:e23a:3", 80, 0, 0)), result
        )


@skipIfNoNetwork
class ThreadedResolverTest(AsyncTestCase, _ResolverTestMixin):
    def setUp(self):
        super().setUp()
        self.resolver = ThreadedResolver()

    def tearDown(self):
        self.resolver.close()
        super().tearDown()


class ThreadedResolverErrorTest(AsyncTestCase, _ResolverErrorTestMixin):
    def setUp(self):
        super().setUp()
        self.resolver = BlockingResolver()
        self.real_getaddrinfo = socket.getaddrinfo
        socket.getaddrinfo = _failing_getaddrinfo

    def tearDown(self):
        socket.getaddrinfo = self.real_getaddrinfo
        super().tearDown()


@skipIfNoNetwork
@unittest.skipIf(sys.platform == "win32", "preexec_fn not available on win32")
class ThreadedResolverImportTest(unittest.TestCase):
    def test_import(self):
        TIMEOUT = 5

        # Test for a deadlock when importing a module that runs the
        # ThreadedResolver at import-time. See resolve_test.py for
        # full explanation.
        command = [sys.executable, "-c", "import tornado.test.resolve_test_helper"]

        start = time.time()
        popen = Popen(command, preexec_fn=lambda: signal.alarm(TIMEOUT))
        while time.time() - start < TIMEOUT:
            return_code = popen.poll()
            if return_code is not None:
                self.assertEqual(0, return_code)
                return  # Success.
            time.sleep(0.05)

        self.fail("import timed out")


# We do not test errors with CaresResolver:
# Some DNS-hijacking ISPs (e.g. Time Warner) return non-empty results
# with an NXDOMAIN status code.  Most resolvers treat this as an error;
# C-ares returns the results, making the "bad_host" tests unreliable.
# C-ares will try to resolve even malformed names, such as the
# name with spaces used in this test.
@skipIfNoNetwork
@unittest.skipIf(pycares is None, "pycares module not present")
@unittest.skipIf(sys.platform == "win32", "pycares doesn't return loopback on windows")
@unittest.skipIf(sys.platform == "darwin", "pycares doesn't return 127.0.0.1 on darwin")
class CaresResolverTest(AsyncTestCase, _ResolverTestMixin):
    def setUp(self):
        super().setUp()
        self.resolver = CaresResolver()


# TwistedResolver produces consistent errors in our test cases so we
# could test the regular and error cases in the same class. However,
# in the error cases it appears that cleanup of socket objects is
# handled asynchronously and occasionally results in "unclosed socket"
# warnings if not given time to shut down (and there is no way to
# explicitly shut it down). This makes the test flaky, so we do not
# test error cases here.
@skipIfNoNetwork
@unittest.skipIf(twisted is None, "twisted module not present")
@unittest.skipIf(
    getattr(twisted, "__version__", "0.0") < "12.1", "old version of twisted"
)
@unittest.skipIf(sys.platform == "win32", "twisted resolver hangs on windows")
class TwistedResolverTest(AsyncTestCase, _ResolverTestMixin):
    def setUp(self):
        super().setUp()
        self.resolver = TwistedResolver()


class IsValidIPTest(unittest.TestCase):
    def test_is_valid_ip(self):
        self.assertTrue(is_valid_ip("127.0.0.1"))
        self.assertTrue(is_valid_ip("4.4.4.4"))
        self.assertTrue(is_valid_ip("::1"))
        self.assertTrue(is_valid_ip("2620:0:1cfe:face:b00c::3"))
        self.assertTrue(not is_valid_ip("www.google.com"))
        self.assertTrue(not is_valid_ip("localhost"))
        self.assertTrue(not is_valid_ip("4.4.4.4<"))
        self.assertTrue(not is_valid_ip(" 127.0.0.1"))
        self.assertTrue(not is_valid_ip(""))
        self.assertTrue(not is_valid_ip(" "))
        self.assertTrue(not is_valid_ip("\n"))
        self.assertTrue(not is_valid_ip("\x00"))
        self.assertTrue(not is_valid_ip("a" * 100))


class TestPortAllocation(unittest.TestCase):
    def test_same_port_allocation(self):
        if "TRAVIS" in os.environ:
            self.skipTest("dual-stack servers often have port conflicts on travis")
        sockets = bind_sockets(0, "localhost")
        try:
            port = sockets[0].getsockname()[1]
            self.assertTrue(all(s.getsockname()[1] == port for s in sockets[1:]))
        finally:
            for sock in sockets:
                sock.close()

    @unittest.skipIf(
        not hasattr(socket, "SO_REUSEPORT"), "SO_REUSEPORT is not supported"
    )
    def test_reuse_port(self):
        sockets = []  # type: List[socket.socket]
        socket, port = bind_unused_port(reuse_port=True)
        try:
            sockets = bind_sockets(port, "127.0.0.1", reuse_port=True)
            self.assertTrue(all(s.getsockname()[1] == port for s in sockets))
        finally:
            socket.close()
            for sock in sockets:
                sock.close()
