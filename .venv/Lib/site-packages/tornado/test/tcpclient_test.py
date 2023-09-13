#
# Copyright 2014 Facebook
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
from contextlib import closing
import getpass
import os
import socket
import unittest

from tornado.concurrent import Future
from tornado.netutil import bind_sockets, Resolver
from tornado.queues import Queue
from tornado.tcpclient import TCPClient, _Connector
from tornado.tcpserver import TCPServer
from tornado.testing import AsyncTestCase, gen_test
from tornado.test.util import skipIfNoIPv6, refusing_port, skipIfNonUnix
from tornado.gen import TimeoutError

import typing

if typing.TYPE_CHECKING:
    from tornado.iostream import IOStream  # noqa: F401
    from typing import List, Dict, Tuple  # noqa: F401

# Fake address families for testing.  Used in place of AF_INET
# and AF_INET6 because some installations do not have AF_INET6.
AF1, AF2 = 1, 2


class TestTCPServer(TCPServer):
    def __init__(self, family):
        super().__init__()
        self.streams = []  # type: List[IOStream]
        self.queue = Queue()  # type: Queue[IOStream]
        sockets = bind_sockets(0, "localhost", family)
        self.add_sockets(sockets)
        self.port = sockets[0].getsockname()[1]

    def handle_stream(self, stream, address):
        self.streams.append(stream)
        self.queue.put(stream)

    def stop(self):
        super().stop()
        for stream in self.streams:
            stream.close()


class TCPClientTest(AsyncTestCase):
    def setUp(self):
        super().setUp()
        self.server = None
        self.client = TCPClient()

    def start_server(self, family):
        if family == socket.AF_UNSPEC and "TRAVIS" in os.environ:
            self.skipTest("dual-stack servers often have port conflicts on travis")
        self.server = TestTCPServer(family)
        return self.server.port

    def stop_server(self):
        if self.server is not None:
            self.server.stop()
            self.server = None

    def tearDown(self):
        self.client.close()
        self.stop_server()
        super().tearDown()

    def skipIfLocalhostV4(self):
        # The port used here doesn't matter, but some systems require it
        # to be non-zero if we do not also pass AI_PASSIVE.
        addrinfo = self.io_loop.run_sync(lambda: Resolver().resolve("localhost", 80))
        families = set(addr[0] for addr in addrinfo)
        if socket.AF_INET6 not in families:
            self.skipTest("localhost does not resolve to ipv6")

    @gen_test
    def do_test_connect(self, family, host, source_ip=None, source_port=None):
        port = self.start_server(family)
        stream = yield self.client.connect(
            host,
            port,
            source_ip=source_ip,
            source_port=source_port,
            af=family,
        )
        assert self.server is not None
        server_stream = yield self.server.queue.get()
        with closing(stream):
            stream.write(b"hello")
            data = yield server_stream.read_bytes(5)
            self.assertEqual(data, b"hello")

    def test_connect_ipv4_ipv4(self):
        self.do_test_connect(socket.AF_INET, "127.0.0.1")

    def test_connect_ipv4_dual(self):
        self.do_test_connect(socket.AF_INET, "localhost")

    @skipIfNoIPv6
    def test_connect_ipv6_ipv6(self):
        self.skipIfLocalhostV4()
        self.do_test_connect(socket.AF_INET6, "::1")

    @skipIfNoIPv6
    def test_connect_ipv6_dual(self):
        self.skipIfLocalhostV4()
        if Resolver.configured_class().__name__.endswith("TwistedResolver"):
            self.skipTest("TwistedResolver does not support multiple addresses")
        self.do_test_connect(socket.AF_INET6, "localhost")

    def test_connect_unspec_ipv4(self):
        self.do_test_connect(socket.AF_UNSPEC, "127.0.0.1")

    @skipIfNoIPv6
    def test_connect_unspec_ipv6(self):
        self.skipIfLocalhostV4()
        self.do_test_connect(socket.AF_UNSPEC, "::1")

    def test_connect_unspec_dual(self):
        self.do_test_connect(socket.AF_UNSPEC, "localhost")

    @gen_test
    def test_refused_ipv4(self):
        cleanup_func, port = refusing_port()
        self.addCleanup(cleanup_func)
        with self.assertRaises(IOError):
            yield self.client.connect("127.0.0.1", port)

    def test_source_ip_fail(self):
        """Fail when trying to use the source IP Address '8.8.8.8'."""
        self.assertRaises(
            socket.error,
            self.do_test_connect,
            socket.AF_INET,
            "127.0.0.1",
            source_ip="8.8.8.8",
        )

    def test_source_ip_success(self):
        """Success when trying to use the source IP Address '127.0.0.1'."""
        self.do_test_connect(socket.AF_INET, "127.0.0.1", source_ip="127.0.0.1")

    @skipIfNonUnix
    def test_source_port_fail(self):
        """Fail when trying to use source port 1."""
        if getpass.getuser() == "root":
            # Root can use any port so we can't easily force this to fail.
            # This is mainly relevant for docker.
            self.skipTest("running as root")
        self.assertRaises(
            socket.error,
            self.do_test_connect,
            socket.AF_INET,
            "127.0.0.1",
            source_port=1,
        )

    @gen_test
    def test_connect_timeout(self):
        timeout = 0.05

        class TimeoutResolver(Resolver):
            def resolve(self, *args, **kwargs):
                return Future()  # never completes

        with self.assertRaises(TimeoutError):
            yield TCPClient(resolver=TimeoutResolver()).connect(
                "1.2.3.4", 12345, timeout=timeout
            )


class TestConnectorSplit(unittest.TestCase):
    def test_one_family(self):
        # These addresses aren't in the right format, but split doesn't care.
        primary, secondary = _Connector.split([(AF1, "a"), (AF1, "b")])
        self.assertEqual(primary, [(AF1, "a"), (AF1, "b")])
        self.assertEqual(secondary, [])

    def test_mixed(self):
        primary, secondary = _Connector.split(
            [(AF1, "a"), (AF2, "b"), (AF1, "c"), (AF2, "d")]
        )
        self.assertEqual(primary, [(AF1, "a"), (AF1, "c")])
        self.assertEqual(secondary, [(AF2, "b"), (AF2, "d")])


class ConnectorTest(AsyncTestCase):
    class FakeStream(object):
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    def setUp(self):
        super().setUp()
        self.connect_futures = (
            {}
        )  # type: Dict[Tuple[int, typing.Any], Future[ConnectorTest.FakeStream]]
        self.streams = {}  # type: Dict[typing.Any, ConnectorTest.FakeStream]
        self.addrinfo = [(AF1, "a"), (AF1, "b"), (AF2, "c"), (AF2, "d")]

    def tearDown(self):
        # Unless explicitly checked (and popped) in the test, we shouldn't
        # be closing any streams
        for stream in self.streams.values():
            self.assertFalse(stream.closed)
        super().tearDown()

    def create_stream(self, af, addr):
        stream = ConnectorTest.FakeStream()
        self.streams[addr] = stream
        future = Future()  # type: Future[ConnectorTest.FakeStream]
        self.connect_futures[(af, addr)] = future
        return stream, future

    def assert_pending(self, *keys):
        self.assertEqual(sorted(self.connect_futures.keys()), sorted(keys))

    def resolve_connect(self, af, addr, success):
        future = self.connect_futures.pop((af, addr))
        if success:
            future.set_result(self.streams[addr])
        else:
            self.streams.pop(addr)
            future.set_exception(IOError())
        # Run the loop to allow callbacks to be run.
        self.io_loop.add_callback(self.stop)
        self.wait()

    def assert_connector_streams_closed(self, conn):
        for stream in conn.streams:
            self.assertTrue(stream.closed)

    def start_connect(self, addrinfo):
        conn = _Connector(addrinfo, self.create_stream)
        # Give it a huge timeout; we'll trigger timeouts manually.
        future = conn.start(3600, connect_timeout=self.io_loop.time() + 3600)
        return conn, future

    def test_immediate_success(self):
        conn, future = self.start_connect(self.addrinfo)
        self.assertEqual(list(self.connect_futures.keys()), [(AF1, "a")])
        self.resolve_connect(AF1, "a", True)
        self.assertEqual(future.result(), (AF1, "a", self.streams["a"]))

    def test_immediate_failure(self):
        # Fail with just one address.
        conn, future = self.start_connect([(AF1, "a")])
        self.assert_pending((AF1, "a"))
        self.resolve_connect(AF1, "a", False)
        self.assertRaises(IOError, future.result)

    def test_one_family_second_try(self):
        conn, future = self.start_connect([(AF1, "a"), (AF1, "b")])
        self.assert_pending((AF1, "a"))
        self.resolve_connect(AF1, "a", False)
        self.assert_pending((AF1, "b"))
        self.resolve_connect(AF1, "b", True)
        self.assertEqual(future.result(), (AF1, "b", self.streams["b"]))

    def test_one_family_second_try_failure(self):
        conn, future = self.start_connect([(AF1, "a"), (AF1, "b")])
        self.assert_pending((AF1, "a"))
        self.resolve_connect(AF1, "a", False)
        self.assert_pending((AF1, "b"))
        self.resolve_connect(AF1, "b", False)
        self.assertRaises(IOError, future.result)

    def test_one_family_second_try_timeout(self):
        conn, future = self.start_connect([(AF1, "a"), (AF1, "b")])
        self.assert_pending((AF1, "a"))
        # trigger the timeout while the first lookup is pending;
        # nothing happens.
        conn.on_timeout()
        self.assert_pending((AF1, "a"))
        self.resolve_connect(AF1, "a", False)
        self.assert_pending((AF1, "b"))
        self.resolve_connect(AF1, "b", True)
        self.assertEqual(future.result(), (AF1, "b", self.streams["b"]))

    def test_two_families_immediate_failure(self):
        conn, future = self.start_connect(self.addrinfo)
        self.assert_pending((AF1, "a"))
        self.resolve_connect(AF1, "a", False)
        self.assert_pending((AF1, "b"), (AF2, "c"))
        self.resolve_connect(AF1, "b", False)
        self.resolve_connect(AF2, "c", True)
        self.assertEqual(future.result(), (AF2, "c", self.streams["c"]))

    def test_two_families_timeout(self):
        conn, future = self.start_connect(self.addrinfo)
        self.assert_pending((AF1, "a"))
        conn.on_timeout()
        self.assert_pending((AF1, "a"), (AF2, "c"))
        self.resolve_connect(AF2, "c", True)
        self.assertEqual(future.result(), (AF2, "c", self.streams["c"]))
        # resolving 'a' after the connection has completed doesn't start 'b'
        self.resolve_connect(AF1, "a", False)
        self.assert_pending()

    def test_success_after_timeout(self):
        conn, future = self.start_connect(self.addrinfo)
        self.assert_pending((AF1, "a"))
        conn.on_timeout()
        self.assert_pending((AF1, "a"), (AF2, "c"))
        self.resolve_connect(AF1, "a", True)
        self.assertEqual(future.result(), (AF1, "a", self.streams["a"]))
        # resolving 'c' after completion closes the connection.
        self.resolve_connect(AF2, "c", True)
        self.assertTrue(self.streams.pop("c").closed)

    def test_all_fail(self):
        conn, future = self.start_connect(self.addrinfo)
        self.assert_pending((AF1, "a"))
        conn.on_timeout()
        self.assert_pending((AF1, "a"), (AF2, "c"))
        self.resolve_connect(AF2, "c", False)
        self.assert_pending((AF1, "a"), (AF2, "d"))
        self.resolve_connect(AF2, "d", False)
        # one queue is now empty
        self.assert_pending((AF1, "a"))
        self.resolve_connect(AF1, "a", False)
        self.assert_pending((AF1, "b"))
        self.assertFalse(future.done())
        self.resolve_connect(AF1, "b", False)
        self.assertRaises(IOError, future.result)

    def test_one_family_timeout_after_connect_timeout(self):
        conn, future = self.start_connect([(AF1, "a"), (AF1, "b")])
        self.assert_pending((AF1, "a"))
        conn.on_connect_timeout()
        # the connector will close all streams on connect timeout, we
        # should explicitly pop the connect_future.
        self.connect_futures.pop((AF1, "a"))
        self.assertTrue(self.streams.pop("a").closed)
        conn.on_timeout()
        # if the future is set with TimeoutError, we will not iterate next
        # possible address.
        self.assert_pending()
        self.assertEqual(len(conn.streams), 1)
        self.assert_connector_streams_closed(conn)
        self.assertRaises(TimeoutError, future.result)

    def test_one_family_success_before_connect_timeout(self):
        conn, future = self.start_connect([(AF1, "a"), (AF1, "b")])
        self.assert_pending((AF1, "a"))
        self.resolve_connect(AF1, "a", True)
        conn.on_connect_timeout()
        self.assert_pending()
        self.assertEqual(self.streams["a"].closed, False)
        # success stream will be pop
        self.assertEqual(len(conn.streams), 0)
        # streams in connector should be closed after connect timeout
        self.assert_connector_streams_closed(conn)
        self.assertEqual(future.result(), (AF1, "a", self.streams["a"]))

    def test_one_family_second_try_after_connect_timeout(self):
        conn, future = self.start_connect([(AF1, "a"), (AF1, "b")])
        self.assert_pending((AF1, "a"))
        self.resolve_connect(AF1, "a", False)
        self.assert_pending((AF1, "b"))
        conn.on_connect_timeout()
        self.connect_futures.pop((AF1, "b"))
        self.assertTrue(self.streams.pop("b").closed)
        self.assert_pending()
        self.assertEqual(len(conn.streams), 2)
        self.assert_connector_streams_closed(conn)
        self.assertRaises(TimeoutError, future.result)

    def test_one_family_second_try_failure_before_connect_timeout(self):
        conn, future = self.start_connect([(AF1, "a"), (AF1, "b")])
        self.assert_pending((AF1, "a"))
        self.resolve_connect(AF1, "a", False)
        self.assert_pending((AF1, "b"))
        self.resolve_connect(AF1, "b", False)
        conn.on_connect_timeout()
        self.assert_pending()
        self.assertEqual(len(conn.streams), 2)
        self.assert_connector_streams_closed(conn)
        self.assertRaises(IOError, future.result)

    def test_two_family_timeout_before_connect_timeout(self):
        conn, future = self.start_connect(self.addrinfo)
        self.assert_pending((AF1, "a"))
        conn.on_timeout()
        self.assert_pending((AF1, "a"), (AF2, "c"))
        conn.on_connect_timeout()
        self.connect_futures.pop((AF1, "a"))
        self.assertTrue(self.streams.pop("a").closed)
        self.connect_futures.pop((AF2, "c"))
        self.assertTrue(self.streams.pop("c").closed)
        self.assert_pending()
        self.assertEqual(len(conn.streams), 2)
        self.assert_connector_streams_closed(conn)
        self.assertRaises(TimeoutError, future.result)

    def test_two_family_success_after_timeout(self):
        conn, future = self.start_connect(self.addrinfo)
        self.assert_pending((AF1, "a"))
        conn.on_timeout()
        self.assert_pending((AF1, "a"), (AF2, "c"))
        self.resolve_connect(AF1, "a", True)
        # if one of streams succeed, connector will close all other streams
        self.connect_futures.pop((AF2, "c"))
        self.assertTrue(self.streams.pop("c").closed)
        self.assert_pending()
        self.assertEqual(len(conn.streams), 1)
        self.assert_connector_streams_closed(conn)
        self.assertEqual(future.result(), (AF1, "a", self.streams["a"]))

    def test_two_family_timeout_after_connect_timeout(self):
        conn, future = self.start_connect(self.addrinfo)
        self.assert_pending((AF1, "a"))
        conn.on_connect_timeout()
        self.connect_futures.pop((AF1, "a"))
        self.assertTrue(self.streams.pop("a").closed)
        self.assert_pending()
        conn.on_timeout()
        # if the future is set with TimeoutError, connector will not
        # trigger secondary address.
        self.assert_pending()
        self.assertEqual(len(conn.streams), 1)
        self.assert_connector_streams_closed(conn)
        self.assertRaises(TimeoutError, future.result)
