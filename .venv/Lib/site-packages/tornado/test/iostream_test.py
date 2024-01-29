from tornado.concurrent import Future
from tornado import gen
from tornado import netutil
from tornado.ioloop import IOLoop
from tornado.iostream import (
    IOStream,
    SSLIOStream,
    PipeIOStream,
    StreamClosedError,
    _StreamBuffer,
)
from tornado.httputil import HTTPHeaders
from tornado.locks import Condition, Event
from tornado.log import gen_log
from tornado.netutil import ssl_options_to_context, ssl_wrap_socket
from tornado.platform.asyncio import AddThreadSelectorEventLoop
from tornado.tcpserver import TCPServer
from tornado.testing import (
    AsyncHTTPTestCase,
    AsyncHTTPSTestCase,
    AsyncTestCase,
    bind_unused_port,
    ExpectLog,
    gen_test,
)
from tornado.test.util import (
    skipIfNonUnix,
    refusing_port,
    skipPypy3V58,
    ignore_deprecation,
)
from tornado.web import RequestHandler, Application
import asyncio
import errno
import hashlib
import logging
import os
import platform
import random
import socket
import ssl
import typing
from unittest import mock
import unittest


def _server_ssl_options():
    return dict(
        certfile=os.path.join(os.path.dirname(__file__), "test.crt"),
        keyfile=os.path.join(os.path.dirname(__file__), "test.key"),
    )


class HelloHandler(RequestHandler):
    def get(self):
        self.write("Hello")


class TestIOStreamWebMixin(object):
    def _make_client_iostream(self):
        raise NotImplementedError()

    def get_app(self):
        return Application([("/", HelloHandler)])

    def test_connection_closed(self: typing.Any):
        # When a server sends a response and then closes the connection,
        # the client must be allowed to read the data before the IOStream
        # closes itself.  Epoll reports closed connections with a separate
        # EPOLLRDHUP event delivered at the same time as the read event,
        # while kqueue reports them as a second read/write event with an EOF
        # flag.
        response = self.fetch("/", headers={"Connection": "close"})
        response.rethrow()

    @gen_test
    def test_read_until_close(self: typing.Any):
        stream = self._make_client_iostream()
        yield stream.connect(("127.0.0.1", self.get_http_port()))
        stream.write(b"GET / HTTP/1.0\r\n\r\n")

        data = yield stream.read_until_close()
        self.assertTrue(data.startswith(b"HTTP/1.1 200"))
        self.assertTrue(data.endswith(b"Hello"))

    @gen_test
    def test_read_zero_bytes(self: typing.Any):
        self.stream = self._make_client_iostream()
        yield self.stream.connect(("127.0.0.1", self.get_http_port()))
        self.stream.write(b"GET / HTTP/1.0\r\n\r\n")

        # normal read
        data = yield self.stream.read_bytes(9)
        self.assertEqual(data, b"HTTP/1.1 ")

        # zero bytes
        data = yield self.stream.read_bytes(0)
        self.assertEqual(data, b"")

        # another normal read
        data = yield self.stream.read_bytes(3)
        self.assertEqual(data, b"200")

        self.stream.close()

    @gen_test
    def test_write_while_connecting(self: typing.Any):
        stream = self._make_client_iostream()
        connect_fut = stream.connect(("127.0.0.1", self.get_http_port()))
        # unlike the previous tests, try to write before the connection
        # is complete.
        write_fut = stream.write(b"GET / HTTP/1.0\r\nConnection: close\r\n\r\n")
        self.assertFalse(connect_fut.done())

        # connect will always complete before write.
        it = gen.WaitIterator(connect_fut, write_fut)
        resolved_order = []
        while not it.done():
            yield it.next()
            resolved_order.append(it.current_future)
        self.assertEqual(resolved_order, [connect_fut, write_fut])

        data = yield stream.read_until_close()
        self.assertTrue(data.endswith(b"Hello"))

        stream.close()

    @gen_test
    def test_future_interface(self: typing.Any):
        """Basic test of IOStream's ability to return Futures."""
        stream = self._make_client_iostream()
        connect_result = yield stream.connect(("127.0.0.1", self.get_http_port()))
        self.assertIs(connect_result, stream)
        yield stream.write(b"GET / HTTP/1.0\r\n\r\n")
        first_line = yield stream.read_until(b"\r\n")
        self.assertEqual(first_line, b"HTTP/1.1 200 OK\r\n")
        # callback=None is equivalent to no callback.
        header_data = yield stream.read_until(b"\r\n\r\n")
        headers = HTTPHeaders.parse(header_data.decode("latin1"))
        content_length = int(headers["Content-Length"])
        body = yield stream.read_bytes(content_length)
        self.assertEqual(body, b"Hello")
        stream.close()

    @gen_test
    def test_future_close_while_reading(self: typing.Any):
        stream = self._make_client_iostream()
        yield stream.connect(("127.0.0.1", self.get_http_port()))
        yield stream.write(b"GET / HTTP/1.0\r\n\r\n")
        with self.assertRaises(StreamClosedError):
            yield stream.read_bytes(1024 * 1024)
        stream.close()

    @gen_test
    def test_future_read_until_close(self: typing.Any):
        # Ensure that the data comes through before the StreamClosedError.
        stream = self._make_client_iostream()
        yield stream.connect(("127.0.0.1", self.get_http_port()))
        yield stream.write(b"GET / HTTP/1.0\r\nConnection: close\r\n\r\n")
        yield stream.read_until(b"\r\n\r\n")
        body = yield stream.read_until_close()
        self.assertEqual(body, b"Hello")

        # Nothing else to read; the error comes immediately without waiting
        # for yield.
        with self.assertRaises(StreamClosedError):
            stream.read_bytes(1)


class TestReadWriteMixin(object):
    # Tests where one stream reads and the other writes.
    # These should work for BaseIOStream implementations.

    def make_iostream_pair(self, **kwargs):
        raise NotImplementedError

    def iostream_pair(self, **kwargs):
        """Like make_iostream_pair, but called by ``async with``.

        In py37 this becomes simpler with contextlib.asynccontextmanager.
        """

        class IOStreamPairContext:
            def __init__(self, test, kwargs):
                self.test = test
                self.kwargs = kwargs

            async def __aenter__(self):
                self.pair = await self.test.make_iostream_pair(**self.kwargs)
                return self.pair

            async def __aexit__(self, typ, value, tb):
                for s in self.pair:
                    s.close()

        return IOStreamPairContext(self, kwargs)

    @gen_test
    def test_write_zero_bytes(self):
        # Attempting to write zero bytes should run the callback without
        # going into an infinite loop.
        rs, ws = yield self.make_iostream_pair()
        yield ws.write(b"")
        ws.close()
        rs.close()

    @gen_test
    def test_future_delayed_close_callback(self: typing.Any):
        # Same as test_delayed_close_callback, but with the future interface.
        rs, ws = yield self.make_iostream_pair()

        try:
            ws.write(b"12")
            chunks = []
            chunks.append((yield rs.read_bytes(1)))
            ws.close()
            chunks.append((yield rs.read_bytes(1)))
            self.assertEqual(chunks, [b"1", b"2"])
        finally:
            ws.close()
            rs.close()

    @gen_test
    def test_close_buffered_data(self: typing.Any):
        # Similar to the previous test, but with data stored in the OS's
        # socket buffers instead of the IOStream's read buffer.  Out-of-band
        # close notifications must be delayed until all data has been
        # drained into the IOStream buffer. (epoll used to use out-of-band
        # close events with EPOLLRDHUP, but no longer)
        #
        # This depends on the read_chunk_size being smaller than the
        # OS socket buffer, so make it small.
        rs, ws = yield self.make_iostream_pair(read_chunk_size=256)
        try:
            ws.write(b"A" * 512)
            data = yield rs.read_bytes(256)
            self.assertEqual(b"A" * 256, data)
            ws.close()
            # Allow the close to propagate to the `rs` side of the
            # connection.  Using add_callback instead of add_timeout
            # doesn't seem to work, even with multiple iterations
            yield gen.sleep(0.01)
            data = yield rs.read_bytes(256)
            self.assertEqual(b"A" * 256, data)
        finally:
            ws.close()
            rs.close()

    @gen_test
    def test_read_until_close_after_close(self: typing.Any):
        # Similar to test_delayed_close_callback, but read_until_close takes
        # a separate code path so test it separately.
        rs, ws = yield self.make_iostream_pair()
        try:
            ws.write(b"1234")
            # Read one byte to make sure the client has received the data.
            # It won't run the close callback as long as there is more buffered
            # data that could satisfy a later read.
            data = yield rs.read_bytes(1)
            ws.close()
            self.assertEqual(data, b"1")
            data = yield rs.read_until_close()
            self.assertEqual(data, b"234")
        finally:
            ws.close()
            rs.close()

    @gen_test
    def test_large_read_until(self: typing.Any):
        # Performance test: read_until used to have a quadratic component
        # so a read_until of 4MB would take 8 seconds; now it takes 0.25
        # seconds.
        rs, ws = yield self.make_iostream_pair()
        try:
            # This test fails on pypy with ssl.  I think it's because
            # pypy's gc defeats moves objects, breaking the
            # "frozen write buffer" assumption.
            if (
                isinstance(rs, SSLIOStream)
                and platform.python_implementation() == "PyPy"
            ):
                raise unittest.SkipTest("pypy gc causes problems with openssl")
            NUM_KB = 4096
            for i in range(NUM_KB):
                ws.write(b"A" * 1024)
            ws.write(b"\r\n")
            data = yield rs.read_until(b"\r\n")
            self.assertEqual(len(data), NUM_KB * 1024 + 2)
        finally:
            ws.close()
            rs.close()

    @gen_test
    async def test_read_until_with_close_after_second_packet(self):
        # This is a regression test for a regression in Tornado 6.0
        # (maybe 6.0.3?) reported in
        # https://github.com/tornadoweb/tornado/issues/2717
        #
        # The data arrives in two chunks; the stream is closed at the
        # same time that the second chunk is received. If the second
        # chunk is larger than the first, it works, but when this bug
        # existed it would fail if the second chunk were smaller than
        # the first. This is due to the optimization that the
        # read_until condition is only checked when the buffer doubles
        # in size
        async with self.iostream_pair() as (rs, ws):
            rf = asyncio.ensure_future(rs.read_until(b"done"))
            # We need to wait for the read_until to actually start. On
            # windows that's tricky because the selector runs in
            # another thread; sleeping is the simplest way.
            await asyncio.sleep(0.1)
            await ws.write(b"x" * 2048)
            ws.write(b"done")
            ws.close()
            await rf

    @gen_test
    async def test_read_until_unsatisfied_after_close(self: typing.Any):
        # If a stream is closed while reading, it raises
        # StreamClosedError instead of UnsatisfiableReadError (the
        # latter should only be raised when byte limits are reached).
        # The particular scenario tested here comes from #2717.
        async with self.iostream_pair() as (rs, ws):
            rf = asyncio.ensure_future(rs.read_until(b"done"))
            await ws.write(b"x" * 2048)
            ws.write(b"foo")
            ws.close()
            with self.assertRaises(StreamClosedError):
                await rf

    @gen_test
    def test_close_callback_with_pending_read(self: typing.Any):
        # Regression test for a bug that was introduced in 2.3
        # where the IOStream._close_callback would never be called
        # if there were pending reads.
        OK = b"OK\r\n"
        rs, ws = yield self.make_iostream_pair()
        event = Event()
        rs.set_close_callback(event.set)
        try:
            ws.write(OK)
            res = yield rs.read_until(b"\r\n")
            self.assertEqual(res, OK)

            ws.close()
            rs.read_until(b"\r\n")
            # If _close_callback (self.stop) is not called,
            # an AssertionError: Async operation timed out after 5 seconds
            # will be raised.
            yield event.wait()
        finally:
            ws.close()
            rs.close()

    @gen_test
    def test_future_close_callback(self: typing.Any):
        # Regression test for interaction between the Future read interfaces
        # and IOStream._maybe_add_error_listener.
        rs, ws = yield self.make_iostream_pair()
        closed = [False]
        cond = Condition()

        def close_callback():
            closed[0] = True
            cond.notify()

        rs.set_close_callback(close_callback)
        try:
            ws.write(b"a")
            res = yield rs.read_bytes(1)
            self.assertEqual(res, b"a")
            self.assertFalse(closed[0])
            ws.close()
            yield cond.wait()
            self.assertTrue(closed[0])
        finally:
            rs.close()
            ws.close()

    @gen_test
    def test_write_memoryview(self: typing.Any):
        rs, ws = yield self.make_iostream_pair()
        try:
            fut = rs.read_bytes(4)
            ws.write(memoryview(b"hello"))
            data = yield fut
            self.assertEqual(data, b"hell")
        finally:
            ws.close()
            rs.close()

    @gen_test
    def test_read_bytes_partial(self: typing.Any):
        rs, ws = yield self.make_iostream_pair()
        try:
            # Ask for more than is available with partial=True
            fut = rs.read_bytes(50, partial=True)
            ws.write(b"hello")
            data = yield fut
            self.assertEqual(data, b"hello")

            # Ask for less than what is available; num_bytes is still
            # respected.
            fut = rs.read_bytes(3, partial=True)
            ws.write(b"world")
            data = yield fut
            self.assertEqual(data, b"wor")

            # Partial reads won't return an empty string, but read_bytes(0)
            # will.
            data = yield rs.read_bytes(0, partial=True)
            self.assertEqual(data, b"")
        finally:
            ws.close()
            rs.close()

    @gen_test
    def test_read_until_max_bytes(self: typing.Any):
        rs, ws = yield self.make_iostream_pair()
        closed = Event()
        rs.set_close_callback(closed.set)
        try:
            # Extra room under the limit
            fut = rs.read_until(b"def", max_bytes=50)
            ws.write(b"abcdef")
            data = yield fut
            self.assertEqual(data, b"abcdef")

            # Just enough space
            fut = rs.read_until(b"def", max_bytes=6)
            ws.write(b"abcdef")
            data = yield fut
            self.assertEqual(data, b"abcdef")

            # Not enough space, but we don't know it until all we can do is
            # log a warning and close the connection.
            with ExpectLog(gen_log, "Unsatisfiable read", level=logging.INFO):
                fut = rs.read_until(b"def", max_bytes=5)
                ws.write(b"123456")
                yield closed.wait()
        finally:
            ws.close()
            rs.close()

    @gen_test
    def test_read_until_max_bytes_inline(self: typing.Any):
        rs, ws = yield self.make_iostream_pair()
        closed = Event()
        rs.set_close_callback(closed.set)
        try:
            # Similar to the error case in the previous test, but the
            # ws writes first so rs reads are satisfied
            # inline.  For consistency with the out-of-line case, we
            # do not raise the error synchronously.
            ws.write(b"123456")
            with ExpectLog(gen_log, "Unsatisfiable read", level=logging.INFO):
                with self.assertRaises(StreamClosedError):
                    yield rs.read_until(b"def", max_bytes=5)
            yield closed.wait()
        finally:
            ws.close()
            rs.close()

    @gen_test
    def test_read_until_max_bytes_ignores_extra(self: typing.Any):
        rs, ws = yield self.make_iostream_pair()
        closed = Event()
        rs.set_close_callback(closed.set)
        try:
            # Even though data that matches arrives the same packet that
            # puts us over the limit, we fail the request because it was not
            # found within the limit.
            ws.write(b"abcdef")
            with ExpectLog(gen_log, "Unsatisfiable read", level=logging.INFO):
                rs.read_until(b"def", max_bytes=5)
                yield closed.wait()
        finally:
            ws.close()
            rs.close()

    @gen_test
    def test_read_until_regex_max_bytes(self: typing.Any):
        rs, ws = yield self.make_iostream_pair()
        closed = Event()
        rs.set_close_callback(closed.set)
        try:
            # Extra room under the limit
            fut = rs.read_until_regex(b"def", max_bytes=50)
            ws.write(b"abcdef")
            data = yield fut
            self.assertEqual(data, b"abcdef")

            # Just enough space
            fut = rs.read_until_regex(b"def", max_bytes=6)
            ws.write(b"abcdef")
            data = yield fut
            self.assertEqual(data, b"abcdef")

            # Not enough space, but we don't know it until all we can do is
            # log a warning and close the connection.
            with ExpectLog(gen_log, "Unsatisfiable read", level=logging.INFO):
                rs.read_until_regex(b"def", max_bytes=5)
                ws.write(b"123456")
                yield closed.wait()
        finally:
            ws.close()
            rs.close()

    @gen_test
    def test_read_until_regex_max_bytes_inline(self: typing.Any):
        rs, ws = yield self.make_iostream_pair()
        closed = Event()
        rs.set_close_callback(closed.set)
        try:
            # Similar to the error case in the previous test, but the
            # ws writes first so rs reads are satisfied
            # inline.  For consistency with the out-of-line case, we
            # do not raise the error synchronously.
            ws.write(b"123456")
            with ExpectLog(gen_log, "Unsatisfiable read", level=logging.INFO):
                rs.read_until_regex(b"def", max_bytes=5)
                yield closed.wait()
        finally:
            ws.close()
            rs.close()

    @gen_test
    def test_read_until_regex_max_bytes_ignores_extra(self):
        rs, ws = yield self.make_iostream_pair()
        closed = Event()
        rs.set_close_callback(closed.set)
        try:
            # Even though data that matches arrives the same packet that
            # puts us over the limit, we fail the request because it was not
            # found within the limit.
            ws.write(b"abcdef")
            with ExpectLog(gen_log, "Unsatisfiable read", level=logging.INFO):
                rs.read_until_regex(b"def", max_bytes=5)
                yield closed.wait()
        finally:
            ws.close()
            rs.close()

    @gen_test
    def test_small_reads_from_large_buffer(self: typing.Any):
        # 10KB buffer size, 100KB available to read.
        # Read 1KB at a time and make sure that the buffer is not eagerly
        # filled.
        rs, ws = yield self.make_iostream_pair(max_buffer_size=10 * 1024)
        try:
            ws.write(b"a" * 1024 * 100)
            for i in range(100):
                data = yield rs.read_bytes(1024)
                self.assertEqual(data, b"a" * 1024)
        finally:
            ws.close()
            rs.close()

    @gen_test
    def test_small_read_untils_from_large_buffer(self: typing.Any):
        # 10KB buffer size, 100KB available to read.
        # Read 1KB at a time and make sure that the buffer is not eagerly
        # filled.
        rs, ws = yield self.make_iostream_pair(max_buffer_size=10 * 1024)
        try:
            ws.write((b"a" * 1023 + b"\n") * 100)
            for i in range(100):
                data = yield rs.read_until(b"\n", max_bytes=4096)
                self.assertEqual(data, b"a" * 1023 + b"\n")
        finally:
            ws.close()
            rs.close()

    @gen_test
    def test_flow_control(self):
        MB = 1024 * 1024
        rs, ws = yield self.make_iostream_pair(max_buffer_size=5 * MB)
        try:
            # Client writes more than the rs will accept.
            ws.write(b"a" * 10 * MB)
            # The rs pauses while reading.
            yield rs.read_bytes(MB)
            yield gen.sleep(0.1)
            # The ws's writes have been blocked; the rs can
            # continue to read gradually.
            for i in range(9):
                yield rs.read_bytes(MB)
        finally:
            rs.close()
            ws.close()

    @gen_test
    def test_read_into(self: typing.Any):
        rs, ws = yield self.make_iostream_pair()

        def sleep_some():
            self.io_loop.run_sync(lambda: gen.sleep(0.05))

        try:
            buf = bytearray(10)
            fut = rs.read_into(buf)
            ws.write(b"hello")
            yield gen.sleep(0.05)
            self.assertTrue(rs.reading())
            ws.write(b"world!!")
            data = yield fut
            self.assertFalse(rs.reading())
            self.assertEqual(data, 10)
            self.assertEqual(bytes(buf), b"helloworld")

            # Existing buffer is fed into user buffer
            fut = rs.read_into(buf)
            yield gen.sleep(0.05)
            self.assertTrue(rs.reading())
            ws.write(b"1234567890")
            data = yield fut
            self.assertFalse(rs.reading())
            self.assertEqual(data, 10)
            self.assertEqual(bytes(buf), b"!!12345678")

            # Existing buffer can satisfy read immediately
            buf = bytearray(4)
            ws.write(b"abcdefghi")
            data = yield rs.read_into(buf)
            self.assertEqual(data, 4)
            self.assertEqual(bytes(buf), b"90ab")

            data = yield rs.read_bytes(7)
            self.assertEqual(data, b"cdefghi")
        finally:
            ws.close()
            rs.close()

    @gen_test
    def test_read_into_partial(self: typing.Any):
        rs, ws = yield self.make_iostream_pair()

        try:
            # Partial read
            buf = bytearray(10)
            fut = rs.read_into(buf, partial=True)
            ws.write(b"hello")
            data = yield fut
            self.assertFalse(rs.reading())
            self.assertEqual(data, 5)
            self.assertEqual(bytes(buf), b"hello\0\0\0\0\0")

            # Full read despite partial=True
            ws.write(b"world!1234567890")
            data = yield rs.read_into(buf, partial=True)
            self.assertEqual(data, 10)
            self.assertEqual(bytes(buf), b"world!1234")

            # Existing buffer can satisfy read immediately
            data = yield rs.read_into(buf, partial=True)
            self.assertEqual(data, 6)
            self.assertEqual(bytes(buf), b"5678901234")

        finally:
            ws.close()
            rs.close()

    @gen_test
    def test_read_into_zero_bytes(self: typing.Any):
        rs, ws = yield self.make_iostream_pair()
        try:
            buf = bytearray()
            fut = rs.read_into(buf)
            self.assertEqual(fut.result(), 0)
        finally:
            ws.close()
            rs.close()

    @gen_test
    def test_many_mixed_reads(self):
        # Stress buffer handling when going back and forth between
        # read_bytes() (using an internal buffer) and read_into()
        # (using a user-allocated buffer).
        r = random.Random(42)
        nbytes = 1000000
        rs, ws = yield self.make_iostream_pair()

        produce_hash = hashlib.sha1()
        consume_hash = hashlib.sha1()

        @gen.coroutine
        def produce():
            remaining = nbytes
            while remaining > 0:
                size = r.randint(1, min(1000, remaining))
                data = os.urandom(size)
                produce_hash.update(data)
                yield ws.write(data)
                remaining -= size
            assert remaining == 0

        @gen.coroutine
        def consume():
            remaining = nbytes
            while remaining > 0:
                if r.random() > 0.5:
                    # read_bytes()
                    size = r.randint(1, min(1000, remaining))
                    data = yield rs.read_bytes(size)
                    consume_hash.update(data)
                    remaining -= size
                else:
                    # read_into()
                    size = r.randint(1, min(1000, remaining))
                    buf = bytearray(size)
                    n = yield rs.read_into(buf)
                    assert n == size
                    consume_hash.update(buf)
                    remaining -= size
            assert remaining == 0

        try:
            yield [produce(), consume()]
            assert produce_hash.hexdigest() == consume_hash.hexdigest()
        finally:
            ws.close()
            rs.close()


class TestIOStreamMixin(TestReadWriteMixin):
    def _make_server_iostream(self, connection, **kwargs):
        raise NotImplementedError()

    def _make_client_iostream(self, connection, **kwargs):
        raise NotImplementedError()

    @gen.coroutine
    def make_iostream_pair(self: typing.Any, **kwargs):
        listener, port = bind_unused_port()
        server_stream_fut = Future()  # type: Future[IOStream]

        def accept_callback(connection, address):
            server_stream_fut.set_result(
                self._make_server_iostream(connection, **kwargs)
            )

        netutil.add_accept_handler(listener, accept_callback)
        client_stream = self._make_client_iostream(socket.socket(), **kwargs)
        connect_fut = client_stream.connect(("127.0.0.1", port))
        server_stream, client_stream = yield [server_stream_fut, connect_fut]
        self.io_loop.remove_handler(listener.fileno())
        listener.close()
        raise gen.Return((server_stream, client_stream))

    @gen_test
    def test_connection_refused(self: typing.Any):
        # When a connection is refused, the connect callback should not
        # be run.  (The kqueue IOLoop used to behave differently from the
        # epoll IOLoop in this respect)
        cleanup_func, port = refusing_port()
        self.addCleanup(cleanup_func)
        stream = IOStream(socket.socket())

        stream.set_close_callback(self.stop)
        # log messages vary by platform and ioloop implementation
        with ExpectLog(gen_log, ".*", required=False):
            with self.assertRaises(StreamClosedError):
                yield stream.connect(("127.0.0.1", port))

        self.assertTrue(isinstance(stream.error, ConnectionRefusedError), stream.error)

    @gen_test
    def test_gaierror(self: typing.Any):
        # Test that IOStream sets its exc_info on getaddrinfo error.
        # It's difficult to reliably trigger a getaddrinfo error;
        # some resolvers own't even return errors for malformed names,
        # so we mock it instead. If IOStream changes to call a Resolver
        # before sock.connect, the mock target will need to change too.
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
        stream = IOStream(s)
        stream.set_close_callback(self.stop)
        with mock.patch(
            "socket.socket.connect", side_effect=socket.gaierror(errno.EIO, "boom")
        ):
            with self.assertRaises(StreamClosedError):
                yield stream.connect(("localhost", 80))
            self.assertTrue(isinstance(stream.error, socket.gaierror))

    @gen_test
    def test_read_until_close_with_error(self: typing.Any):
        server, client = yield self.make_iostream_pair()
        try:
            with mock.patch(
                "tornado.iostream.BaseIOStream._try_inline_read",
                side_effect=IOError("boom"),
            ):
                with self.assertRaisesRegex(IOError, "boom"):
                    client.read_until_close()
        finally:
            server.close()
            client.close()

    @skipIfNonUnix
    @skipPypy3V58
    @gen_test
    def test_inline_read_error(self: typing.Any):
        # An error on an inline read is raised without logging (on the
        # assumption that it will eventually be noticed or logged further
        # up the stack).
        #
        # This test is posix-only because windows os.close() doesn't work
        # on socket FDs, but we can't close the socket object normally
        # because we won't get the error we want if the socket knows
        # it's closed.
        #
        # This test is also disabled when the
        # AddThreadSelectorEventLoop is used, because a race between
        # this thread closing the socket and the selector thread
        # calling the select system call can make this test flaky.
        # This event loop implementation is normally only used on
        # windows, making this check redundant with skipIfNonUnix, but
        # we sometimes enable it on other platforms for testing.
        io_loop = IOLoop.current()
        if isinstance(
            io_loop.selector_loop,  # type: ignore[attr-defined]
            AddThreadSelectorEventLoop,
        ):
            self.skipTest("AddThreadSelectorEventLoop not supported")
        server, client = yield self.make_iostream_pair()
        try:
            os.close(server.socket.fileno())
            with self.assertRaises(socket.error):
                server.read_bytes(1)
        finally:
            server.close()
            client.close()

    @skipPypy3V58
    @gen_test
    def test_async_read_error_logging(self):
        # Socket errors on asynchronous reads should be logged (but only
        # once).
        server, client = yield self.make_iostream_pair()
        closed = Event()
        server.set_close_callback(closed.set)
        try:
            # Start a read that will be fulfilled asynchronously.
            server.read_bytes(1)
            client.write(b"a")
            # Stub out read_from_fd to make it fail.

            def fake_read_from_fd():
                os.close(server.socket.fileno())
                server.__class__.read_from_fd(server)

            server.read_from_fd = fake_read_from_fd
            # This log message is from _handle_read (not read_from_fd).
            with ExpectLog(gen_log, "error on read"):
                yield closed.wait()
        finally:
            server.close()
            client.close()

    @gen_test
    def test_future_write(self):
        """
        Test that write() Futures are never orphaned.
        """
        # Run concurrent writers that will write enough bytes so as to
        # clog the socket buffer and accumulate bytes in our write buffer.
        m, n = 5000, 1000
        nproducers = 10
        total_bytes = m * n * nproducers
        server, client = yield self.make_iostream_pair(max_buffer_size=total_bytes)

        @gen.coroutine
        def produce():
            data = b"x" * m
            for i in range(n):
                yield server.write(data)

        @gen.coroutine
        def consume():
            nread = 0
            while nread < total_bytes:
                res = yield client.read_bytes(m)
                nread += len(res)

        try:
            yield [produce() for i in range(nproducers)] + [consume()]
        finally:
            server.close()
            client.close()


class TestIOStreamWebHTTP(TestIOStreamWebMixin, AsyncHTTPTestCase):
    def _make_client_iostream(self):
        return IOStream(socket.socket())


class TestIOStreamWebHTTPS(TestIOStreamWebMixin, AsyncHTTPSTestCase):
    def _make_client_iostream(self):
        return SSLIOStream(socket.socket(), ssl_options=dict(cert_reqs=ssl.CERT_NONE))


class TestIOStream(TestIOStreamMixin, AsyncTestCase):
    def _make_server_iostream(self, connection, **kwargs):
        return IOStream(connection, **kwargs)

    def _make_client_iostream(self, connection, **kwargs):
        return IOStream(connection, **kwargs)


class TestIOStreamSSL(TestIOStreamMixin, AsyncTestCase):
    def _make_server_iostream(self, connection, **kwargs):
        ssl_ctx = ssl_options_to_context(_server_ssl_options(), server_side=True)
        connection = ssl_ctx.wrap_socket(
            connection,
            server_side=True,
            do_handshake_on_connect=False,
        )
        return SSLIOStream(connection, **kwargs)

    def _make_client_iostream(self, connection, **kwargs):
        return SSLIOStream(
            connection, ssl_options=dict(cert_reqs=ssl.CERT_NONE), **kwargs
        )


# This will run some tests that are basically redundant but it's the
# simplest way to make sure that it works to pass an SSLContext
# instead of an ssl_options dict to the SSLIOStream constructor.
class TestIOStreamSSLContext(TestIOStreamMixin, AsyncTestCase):
    def _make_server_iostream(self, connection, **kwargs):
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.load_cert_chain(
            os.path.join(os.path.dirname(__file__), "test.crt"),
            os.path.join(os.path.dirname(__file__), "test.key"),
        )
        connection = ssl_wrap_socket(
            connection, context, server_side=True, do_handshake_on_connect=False
        )
        return SSLIOStream(connection, **kwargs)

    def _make_client_iostream(self, connection, **kwargs):
        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return SSLIOStream(connection, ssl_options=context, **kwargs)


class TestIOStreamStartTLS(AsyncTestCase):
    def setUp(self):
        try:
            super().setUp()
            self.listener, self.port = bind_unused_port()
            self.server_stream = None
            self.server_accepted = Future()  # type: Future[None]
            netutil.add_accept_handler(self.listener, self.accept)
            self.client_stream = IOStream(
                socket.socket()
            )  # type: typing.Optional[IOStream]
            self.io_loop.add_future(
                self.client_stream.connect(("127.0.0.1", self.port)), self.stop
            )
            self.wait()
            self.io_loop.add_future(self.server_accepted, self.stop)
            self.wait()
        except Exception as e:
            print(e)
            raise

    def tearDown(self):
        if self.server_stream is not None:
            self.server_stream.close()
        if self.client_stream is not None:
            self.client_stream.close()
        self.io_loop.remove_handler(self.listener.fileno())
        self.listener.close()
        super().tearDown()

    def accept(self, connection, address):
        if self.server_stream is not None:
            self.fail("should only get one connection")
        self.server_stream = IOStream(connection)
        self.server_accepted.set_result(None)

    @gen.coroutine
    def client_send_line(self, line):
        assert self.client_stream is not None
        self.client_stream.write(line)
        assert self.server_stream is not None
        recv_line = yield self.server_stream.read_until(b"\r\n")
        self.assertEqual(line, recv_line)

    @gen.coroutine
    def server_send_line(self, line):
        assert self.server_stream is not None
        self.server_stream.write(line)
        assert self.client_stream is not None
        recv_line = yield self.client_stream.read_until(b"\r\n")
        self.assertEqual(line, recv_line)

    def client_start_tls(self, ssl_options=None, server_hostname=None):
        assert self.client_stream is not None
        client_stream = self.client_stream
        self.client_stream = None
        return client_stream.start_tls(False, ssl_options, server_hostname)

    def server_start_tls(self, ssl_options=None):
        assert self.server_stream is not None
        server_stream = self.server_stream
        self.server_stream = None
        return server_stream.start_tls(True, ssl_options)

    @gen_test
    def test_start_tls_smtp(self):
        # This flow is simplified from RFC 3207 section 5.
        # We don't really need all of this, but it helps to make sure
        # that after realistic back-and-forth traffic the buffers end up
        # in a sane state.
        yield self.server_send_line(b"220 mail.example.com ready\r\n")
        yield self.client_send_line(b"EHLO mail.example.com\r\n")
        yield self.server_send_line(b"250-mail.example.com welcome\r\n")
        yield self.server_send_line(b"250 STARTTLS\r\n")
        yield self.client_send_line(b"STARTTLS\r\n")
        yield self.server_send_line(b"220 Go ahead\r\n")
        client_future = self.client_start_tls(dict(cert_reqs=ssl.CERT_NONE))
        server_future = self.server_start_tls(_server_ssl_options())
        self.client_stream = yield client_future
        self.server_stream = yield server_future
        self.assertTrue(isinstance(self.client_stream, SSLIOStream))
        self.assertTrue(isinstance(self.server_stream, SSLIOStream))
        yield self.client_send_line(b"EHLO mail.example.com\r\n")
        yield self.server_send_line(b"250 mail.example.com welcome\r\n")

    @gen_test
    def test_handshake_fail(self):
        server_future = self.server_start_tls(_server_ssl_options())
        # Certificates are verified with the default configuration.
        with ExpectLog(gen_log, "SSL Error"):
            client_future = self.client_start_tls(server_hostname="localhost")
            with self.assertRaises(ssl.SSLError):
                yield client_future
            with self.assertRaises((ssl.SSLError, socket.error)):
                yield server_future

    @gen_test
    def test_check_hostname(self):
        # Test that server_hostname parameter to start_tls is being used.
        # The check_hostname functionality is only available in python 2.7 and
        # up and in python 3.4 and up.
        server_future = self.server_start_tls(_server_ssl_options())
        with ExpectLog(gen_log, "SSL Error"):
            client_future = self.client_start_tls(
                ssl.create_default_context(), server_hostname="127.0.0.1"
            )
            with self.assertRaises(ssl.SSLError):
                # The client fails to connect with an SSL error.
                yield client_future
            with self.assertRaises(Exception):
                # The server fails to connect, but the exact error is unspecified.
                yield server_future

    @gen_test
    def test_typed_memoryview(self):
        # Test support of memoryviews with an item size greater than 1 byte.
        buf = memoryview(bytes(80)).cast("L")
        assert self.server_stream is not None
        yield self.server_stream.write(buf)
        assert self.client_stream is not None
        # This will timeout if the calculation of the buffer size is incorrect
        recv = yield self.client_stream.read_bytes(buf.nbytes)
        self.assertEqual(bytes(recv), bytes(buf))


class WaitForHandshakeTest(AsyncTestCase):
    @gen.coroutine
    def connect_to_server(self, server_cls):
        server = client = None
        try:
            sock, port = bind_unused_port()
            server = server_cls(ssl_options=_server_ssl_options())
            server.add_socket(sock)

            ssl_ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE
            # These tests fail with ConnectionAbortedErrors with TLS
            # 1.3 on windows python 3.7.4 (which includes an upgrade
            # to openssl 1.1.c. Other platforms might be affected with
            # newer openssl too). Disable it until we figure out
            # what's up.
            # Update 2021-12-28: Still happening with Python 3.10 on
            # Windows. OP_NO_TLSv1_3 now raises a DeprecationWarning.
            with ignore_deprecation():
                ssl_ctx.options |= getattr(ssl, "OP_NO_TLSv1_3", 0)
                client = SSLIOStream(socket.socket(), ssl_options=ssl_ctx)
            yield client.connect(("127.0.0.1", port))
            self.assertIsNotNone(client.socket.cipher())
        finally:
            if server is not None:
                server.stop()
            if client is not None:
                client.close()

    @gen_test
    def test_wait_for_handshake_future(self):
        test = self
        handshake_future = Future()  # type: Future[None]

        class TestServer(TCPServer):
            def handle_stream(self, stream, address):
                test.assertIsNone(stream.socket.cipher())
                test.io_loop.spawn_callback(self.handle_connection, stream)

            @gen.coroutine
            def handle_connection(self, stream):
                yield stream.wait_for_handshake()
                handshake_future.set_result(None)

        yield self.connect_to_server(TestServer)
        yield handshake_future

    @gen_test
    def test_wait_for_handshake_already_waiting_error(self):
        test = self
        handshake_future = Future()  # type: Future[None]

        class TestServer(TCPServer):
            @gen.coroutine
            def handle_stream(self, stream, address):
                fut = stream.wait_for_handshake()
                test.assertRaises(RuntimeError, stream.wait_for_handshake)
                yield fut

                handshake_future.set_result(None)

        yield self.connect_to_server(TestServer)
        yield handshake_future

    @gen_test
    def test_wait_for_handshake_already_connected(self):
        handshake_future = Future()  # type: Future[None]

        class TestServer(TCPServer):
            @gen.coroutine
            def handle_stream(self, stream, address):
                yield stream.wait_for_handshake()
                yield stream.wait_for_handshake()
                handshake_future.set_result(None)

        yield self.connect_to_server(TestServer)
        yield handshake_future


class TestIOStreamCheckHostname(AsyncTestCase):
    # This test ensures that hostname checks are working correctly after
    # #3337 revealed that we have no test coverage in this area, and we
    # removed a manual hostname check that was needed only for very old
    # versions of python.
    def setUp(self):
        super().setUp()
        self.listener, self.port = bind_unused_port()

        def accept_callback(connection, address):
            ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_ctx.load_cert_chain(
                os.path.join(os.path.dirname(__file__), "test.crt"),
                os.path.join(os.path.dirname(__file__), "test.key"),
            )
            connection = ssl_ctx.wrap_socket(
                connection,
                server_side=True,
                do_handshake_on_connect=False,
            )
            SSLIOStream(connection)

        netutil.add_accept_handler(self.listener, accept_callback)

        # Our self-signed cert is its own CA.  We have to pass the CA check before
        # the hostname check will be performed.
        self.client_ssl_ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        self.client_ssl_ctx.load_verify_locations(
            os.path.join(os.path.dirname(__file__), "test.crt")
        )

    def tearDown(self):
        self.io_loop.remove_handler(self.listener.fileno())
        self.listener.close()
        super().tearDown()

    @gen_test
    async def test_match(self):
        stream = SSLIOStream(socket.socket(), ssl_options=self.client_ssl_ctx)
        await stream.connect(
            ("127.0.0.1", self.port),
            server_hostname="foo.example.com",
        )
        stream.close()

    @gen_test
    async def test_no_match(self):
        stream = SSLIOStream(socket.socket(), ssl_options=self.client_ssl_ctx)
        with ExpectLog(
            gen_log,
            ".*alert bad certificate",
            level=logging.WARNING,
            required=platform.system() != "Windows",
        ):
            with self.assertRaises(ssl.SSLCertVerificationError):
                with ExpectLog(
                    gen_log,
                    ".*(certificate verify failed: Hostname mismatch)",
                    level=logging.WARNING,
                ):
                    await stream.connect(
                        ("127.0.0.1", self.port),
                        server_hostname="bar.example.com",
                    )
            # The server logs a warning while cleaning up the failed connection.
            # Unfortunately there's no good hook to wait for this logging.
            # It doesn't seem to happen on windows; I'm not sure why.
            if platform.system() != "Windows":
                await asyncio.sleep(0.1)

    @gen_test
    async def test_check_disabled(self):
        # check_hostname can be set to false and the connection will succeed even though it doesn't
        # have the right hostname.
        self.client_ssl_ctx.check_hostname = False
        stream = SSLIOStream(socket.socket(), ssl_options=self.client_ssl_ctx)
        await stream.connect(
            ("127.0.0.1", self.port),
            server_hostname="bar.example.com",
        )


@skipIfNonUnix
class TestPipeIOStream(TestReadWriteMixin, AsyncTestCase):
    @gen.coroutine
    def make_iostream_pair(self, **kwargs):
        r, w = os.pipe()

        return PipeIOStream(r, **kwargs), PipeIOStream(w, **kwargs)

    @gen_test
    def test_pipe_iostream(self):
        rs, ws = yield self.make_iostream_pair()

        ws.write(b"hel")
        ws.write(b"lo world")

        data = yield rs.read_until(b" ")
        self.assertEqual(data, b"hello ")

        data = yield rs.read_bytes(3)
        self.assertEqual(data, b"wor")

        ws.close()

        data = yield rs.read_until_close()
        self.assertEqual(data, b"ld")

        rs.close()

    @gen_test
    def test_pipe_iostream_big_write(self):
        rs, ws = yield self.make_iostream_pair()

        NUM_BYTES = 1048576

        # Write 1MB of data, which should fill the buffer
        ws.write(b"1" * NUM_BYTES)

        data = yield rs.read_bytes(NUM_BYTES)
        self.assertEqual(data, b"1" * NUM_BYTES)

        ws.close()
        rs.close()


class TestStreamBuffer(unittest.TestCase):
    """
    Unit tests for the private _StreamBuffer class.
    """

    def setUp(self):
        self.random = random.Random(42)

    def to_bytes(self, b):
        if isinstance(b, (bytes, bytearray)):
            return bytes(b)
        elif isinstance(b, memoryview):
            return b.tobytes()  # For py2
        else:
            raise TypeError(b)

    def make_streambuffer(self, large_buf_threshold=10):
        buf = _StreamBuffer()
        assert buf._large_buf_threshold
        buf._large_buf_threshold = large_buf_threshold
        return buf

    def check_peek(self, buf, expected):
        size = 1
        while size < 2 * len(expected):
            got = self.to_bytes(buf.peek(size))
            self.assertTrue(got)  # Not empty
            self.assertLessEqual(len(got), size)
            self.assertTrue(expected.startswith(got), (expected, got))
            size = (size * 3 + 1) // 2

    def check_append_all_then_skip_all(self, buf, objs, input_type):
        self.assertEqual(len(buf), 0)

        expected = b""

        for o in objs:
            expected += o
            buf.append(input_type(o))
            self.assertEqual(len(buf), len(expected))
            self.check_peek(buf, expected)

        while expected:
            n = self.random.randrange(1, len(expected) + 1)
            expected = expected[n:]
            buf.advance(n)
            self.assertEqual(len(buf), len(expected))
            self.check_peek(buf, expected)

        self.assertEqual(len(buf), 0)

    def test_small(self):
        objs = [b"12", b"345", b"67", b"89a", b"bcde", b"fgh", b"ijklmn"]

        buf = self.make_streambuffer()
        self.check_append_all_then_skip_all(buf, objs, bytes)

        buf = self.make_streambuffer()
        self.check_append_all_then_skip_all(buf, objs, bytearray)

        buf = self.make_streambuffer()
        self.check_append_all_then_skip_all(buf, objs, memoryview)

        # Test internal algorithm
        buf = self.make_streambuffer(10)
        for i in range(9):
            buf.append(b"x")
        self.assertEqual(len(buf._buffers), 1)
        for i in range(9):
            buf.append(b"x")
        self.assertEqual(len(buf._buffers), 2)
        buf.advance(10)
        self.assertEqual(len(buf._buffers), 1)
        buf.advance(8)
        self.assertEqual(len(buf._buffers), 0)
        self.assertEqual(len(buf), 0)

    def test_large(self):
        objs = [
            b"12" * 5,
            b"345" * 2,
            b"67" * 20,
            b"89a" * 12,
            b"bcde" * 1,
            b"fgh" * 7,
            b"ijklmn" * 2,
        ]

        buf = self.make_streambuffer()
        self.check_append_all_then_skip_all(buf, objs, bytes)

        buf = self.make_streambuffer()
        self.check_append_all_then_skip_all(buf, objs, bytearray)

        buf = self.make_streambuffer()
        self.check_append_all_then_skip_all(buf, objs, memoryview)

        # Test internal algorithm
        buf = self.make_streambuffer(10)
        for i in range(3):
            buf.append(b"x" * 11)
        self.assertEqual(len(buf._buffers), 3)
        buf.append(b"y")
        self.assertEqual(len(buf._buffers), 4)
        buf.append(b"z")
        self.assertEqual(len(buf._buffers), 4)
        buf.advance(33)
        self.assertEqual(len(buf._buffers), 1)
        buf.advance(2)
        self.assertEqual(len(buf._buffers), 0)
        self.assertEqual(len(buf), 0)
