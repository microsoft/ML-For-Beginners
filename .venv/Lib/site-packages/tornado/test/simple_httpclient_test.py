import collections
from contextlib import closing
import errno
import logging
import os
import re
import socket
import ssl
import sys
import typing  # noqa: F401

from tornado.escape import to_unicode, utf8
from tornado import gen, version
from tornado.httpclient import AsyncHTTPClient
from tornado.httputil import HTTPHeaders, ResponseStartLine
from tornado.ioloop import IOLoop
from tornado.iostream import UnsatisfiableReadError
from tornado.locks import Event
from tornado.log import gen_log
from tornado.netutil import Resolver, bind_sockets
from tornado.simple_httpclient import (
    SimpleAsyncHTTPClient,
    HTTPStreamClosedError,
    HTTPTimeoutError,
)
from tornado.test.httpclient_test import (
    ChunkHandler,
    CountdownHandler,
    HelloWorldHandler,
    RedirectHandler,
    UserAgentHandler,
)
from tornado.test import httpclient_test
from tornado.testing import (
    AsyncHTTPTestCase,
    AsyncHTTPSTestCase,
    AsyncTestCase,
    ExpectLog,
    gen_test,
)
from tornado.test.util import skipOnTravis, skipIfNoIPv6, refusing_port
from tornado.web import RequestHandler, Application, url, stream_request_body


class SimpleHTTPClientCommonTestCase(httpclient_test.HTTPClientCommonTestCase):
    def get_http_client(self):
        client = SimpleAsyncHTTPClient(force_instance=True)
        self.assertTrue(isinstance(client, SimpleAsyncHTTPClient))
        return client


class TriggerHandler(RequestHandler):
    def initialize(self, queue, wake_callback):
        self.queue = queue
        self.wake_callback = wake_callback

    @gen.coroutine
    def get(self):
        logging.debug("queuing trigger")
        event = Event()
        self.queue.append(event.set)
        if self.get_argument("wake", "true") == "true":
            self.wake_callback()
        yield event.wait()


class ContentLengthHandler(RequestHandler):
    def get(self):
        self.stream = self.detach()
        IOLoop.current().spawn_callback(self.write_response)

    @gen.coroutine
    def write_response(self):
        yield self.stream.write(
            utf8(
                "HTTP/1.0 200 OK\r\nContent-Length: %s\r\n\r\nok"
                % self.get_argument("value")
            )
        )
        self.stream.close()


class HeadHandler(RequestHandler):
    def head(self):
        self.set_header("Content-Length", "7")


class OptionsHandler(RequestHandler):
    def options(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.write("ok")


class NoContentHandler(RequestHandler):
    def get(self):
        self.set_status(204)
        self.finish()


class SeeOtherPostHandler(RequestHandler):
    def post(self):
        redirect_code = int(self.request.body)
        assert redirect_code in (302, 303), "unexpected body %r" % self.request.body
        self.set_header("Location", "/see_other_get")
        self.set_status(redirect_code)


class SeeOtherGetHandler(RequestHandler):
    def get(self):
        if self.request.body:
            raise Exception("unexpected body %r" % self.request.body)
        self.write("ok")


class HostEchoHandler(RequestHandler):
    def get(self):
        self.write(self.request.headers["Host"])


class NoContentLengthHandler(RequestHandler):
    def get(self):
        if self.request.version.startswith("HTTP/1"):
            # Emulate the old HTTP/1.0 behavior of returning a body with no
            # content-length.  Tornado handles content-length at the framework
            # level so we have to go around it.
            stream = self.detach()
            stream.write(b"HTTP/1.0 200 OK\r\n\r\n" b"hello")
            stream.close()
        else:
            self.finish("HTTP/1 required")


class EchoPostHandler(RequestHandler):
    def post(self):
        self.write(self.request.body)


@stream_request_body
class RespondInPrepareHandler(RequestHandler):
    def prepare(self):
        self.set_status(403)
        self.finish("forbidden")


class SimpleHTTPClientTestMixin(object):
    def create_client(self, **kwargs):
        raise NotImplementedError()

    def get_app(self: typing.Any):
        # callable objects to finish pending /trigger requests
        self.triggers = (
            collections.deque()
        )  # type: typing.Deque[typing.Callable[[], None]]
        return Application(
            [
                url(
                    "/trigger",
                    TriggerHandler,
                    dict(queue=self.triggers, wake_callback=self.stop),
                ),
                url("/chunk", ChunkHandler),
                url("/countdown/([0-9]+)", CountdownHandler, name="countdown"),
                url("/hello", HelloWorldHandler),
                url("/content_length", ContentLengthHandler),
                url("/head", HeadHandler),
                url("/options", OptionsHandler),
                url("/no_content", NoContentHandler),
                url("/see_other_post", SeeOtherPostHandler),
                url("/see_other_get", SeeOtherGetHandler),
                url("/host_echo", HostEchoHandler),
                url("/no_content_length", NoContentLengthHandler),
                url("/echo_post", EchoPostHandler),
                url("/respond_in_prepare", RespondInPrepareHandler),
                url("/redirect", RedirectHandler),
                url("/user_agent", UserAgentHandler),
            ],
            gzip=True,
        )

    def test_singleton(self: typing.Any):
        # Class "constructor" reuses objects on the same IOLoop
        self.assertTrue(SimpleAsyncHTTPClient() is SimpleAsyncHTTPClient())
        # unless force_instance is used
        self.assertTrue(
            SimpleAsyncHTTPClient() is not SimpleAsyncHTTPClient(force_instance=True)
        )
        # different IOLoops use different objects
        with closing(IOLoop()) as io_loop2:

            async def make_client():
                await gen.sleep(0)
                return SimpleAsyncHTTPClient()

            client1 = self.io_loop.run_sync(make_client)
            client2 = io_loop2.run_sync(make_client)
            self.assertTrue(client1 is not client2)

    def test_connection_limit(self: typing.Any):
        with closing(self.create_client(max_clients=2)) as client:
            self.assertEqual(client.max_clients, 2)
            seen = []
            # Send 4 requests.  Two can be sent immediately, while the others
            # will be queued
            for i in range(4):

                def cb(fut, i=i):
                    seen.append(i)
                    self.stop()

                client.fetch(self.get_url("/trigger")).add_done_callback(cb)
            self.wait(condition=lambda: len(self.triggers) == 2)
            self.assertEqual(len(client.queue), 2)

            # Finish the first two requests and let the next two through
            self.triggers.popleft()()
            self.triggers.popleft()()
            self.wait(condition=lambda: (len(self.triggers) == 2 and len(seen) == 2))
            self.assertEqual(set(seen), set([0, 1]))
            self.assertEqual(len(client.queue), 0)

            # Finish all the pending requests
            self.triggers.popleft()()
            self.triggers.popleft()()
            self.wait(condition=lambda: len(seen) == 4)
            self.assertEqual(set(seen), set([0, 1, 2, 3]))
            self.assertEqual(len(self.triggers), 0)

    @gen_test
    def test_redirect_connection_limit(self: typing.Any):
        # following redirects should not consume additional connections
        with closing(self.create_client(max_clients=1)) as client:
            response = yield client.fetch(self.get_url("/countdown/3"), max_redirects=3)
            response.rethrow()

    def test_max_redirects(self: typing.Any):
        response = self.fetch("/countdown/5", max_redirects=3)
        self.assertEqual(302, response.code)
        # We requested 5, followed three redirects for 4, 3, 2, then the last
        # unfollowed redirect is to 1.
        self.assertTrue(response.request.url.endswith("/countdown/5"))
        self.assertTrue(response.effective_url.endswith("/countdown/2"))
        self.assertTrue(response.headers["Location"].endswith("/countdown/1"))

    def test_header_reuse(self: typing.Any):
        # Apps may reuse a headers object if they are only passing in constant
        # headers like user-agent.  The header object should not be modified.
        headers = HTTPHeaders({"User-Agent": "Foo"})
        self.fetch("/hello", headers=headers)
        self.assertEqual(list(headers.get_all()), [("User-Agent", "Foo")])

    def test_default_user_agent(self: typing.Any):
        response = self.fetch("/user_agent", method="GET")
        self.assertEqual(200, response.code)
        self.assertEqual(response.body.decode(), "Tornado/{}".format(version))

    def test_see_other_redirect(self: typing.Any):
        for code in (302, 303):
            response = self.fetch("/see_other_post", method="POST", body="%d" % code)
            self.assertEqual(200, response.code)
            self.assertTrue(response.request.url.endswith("/see_other_post"))
            self.assertTrue(response.effective_url.endswith("/see_other_get"))
            # request is the original request, is a POST still
            self.assertEqual("POST", response.request.method)

    @skipOnTravis
    @gen_test
    def test_connect_timeout(self: typing.Any):
        timeout = 0.1

        cleanup_event = Event()
        test = self

        class TimeoutResolver(Resolver):
            async def resolve(self, *args, **kwargs):
                await cleanup_event.wait()
                # Return something valid so the test doesn't raise during shutdown.
                return [(socket.AF_INET, ("127.0.0.1", test.get_http_port()))]

        with closing(self.create_client(resolver=TimeoutResolver())) as client:
            with self.assertRaises(HTTPTimeoutError):
                yield client.fetch(
                    self.get_url("/hello"),
                    connect_timeout=timeout,
                    request_timeout=3600,
                    raise_error=True,
                )

        # Let the hanging coroutine clean up after itself. We need to
        # wait more than a single IOLoop iteration for the SSL case,
        # which logs errors on unexpected EOF.
        cleanup_event.set()
        yield gen.sleep(0.2)

    @skipOnTravis
    def test_request_timeout(self: typing.Any):
        timeout = 0.1
        if os.name == "nt":
            timeout = 0.5

        with self.assertRaises(HTTPTimeoutError):
            self.fetch("/trigger?wake=false", request_timeout=timeout, raise_error=True)
        # trigger the hanging request to let it clean up after itself
        self.triggers.popleft()()
        self.io_loop.run_sync(lambda: gen.sleep(0))

    @skipIfNoIPv6
    def test_ipv6(self: typing.Any):
        [sock] = bind_sockets(0, "::1", family=socket.AF_INET6)
        port = sock.getsockname()[1]
        self.http_server.add_socket(sock)
        url = "%s://[::1]:%d/hello" % (self.get_protocol(), port)

        # ipv6 is currently enabled by default but can be disabled
        with self.assertRaises(Exception):
            self.fetch(url, allow_ipv6=False, raise_error=True)

        response = self.fetch(url)
        self.assertEqual(response.body, b"Hello world!")

    def test_multiple_content_length_accepted(self: typing.Any):
        response = self.fetch("/content_length?value=2,2")
        self.assertEqual(response.body, b"ok")
        response = self.fetch("/content_length?value=2,%202,2")
        self.assertEqual(response.body, b"ok")

        with ExpectLog(
            gen_log, ".*Multiple unequal Content-Lengths", level=logging.INFO
        ):
            with self.assertRaises(HTTPStreamClosedError):
                self.fetch("/content_length?value=2,4", raise_error=True)
            with self.assertRaises(HTTPStreamClosedError):
                self.fetch("/content_length?value=2,%202,3", raise_error=True)

    def test_head_request(self: typing.Any):
        response = self.fetch("/head", method="HEAD")
        self.assertEqual(response.code, 200)
        self.assertEqual(response.headers["content-length"], "7")
        self.assertFalse(response.body)

    def test_options_request(self: typing.Any):
        response = self.fetch("/options", method="OPTIONS")
        self.assertEqual(response.code, 200)
        self.assertEqual(response.headers["content-length"], "2")
        self.assertEqual(response.headers["access-control-allow-origin"], "*")
        self.assertEqual(response.body, b"ok")

    def test_no_content(self: typing.Any):
        response = self.fetch("/no_content")
        self.assertEqual(response.code, 204)
        # 204 status shouldn't have a content-length
        #
        # Tests with a content-length header are included below
        # in HTTP204NoContentTestCase.
        self.assertNotIn("Content-Length", response.headers)

    def test_host_header(self: typing.Any):
        host_re = re.compile(b"^127.0.0.1:[0-9]+$")
        response = self.fetch("/host_echo")
        self.assertTrue(host_re.match(response.body))

        url = self.get_url("/host_echo").replace("http://", "http://me:secret@")
        response = self.fetch(url)
        self.assertTrue(host_re.match(response.body), response.body)

    def test_connection_refused(self: typing.Any):
        cleanup_func, port = refusing_port()
        self.addCleanup(cleanup_func)
        with ExpectLog(gen_log, ".*", required=False):
            with self.assertRaises(socket.error) as cm:
                self.fetch("http://127.0.0.1:%d/" % port, raise_error=True)

        if sys.platform != "cygwin":
            # cygwin returns EPERM instead of ECONNREFUSED here
            contains_errno = str(errno.ECONNREFUSED) in str(cm.exception)
            if not contains_errno and hasattr(errno, "WSAECONNREFUSED"):
                contains_errno = str(errno.WSAECONNREFUSED) in str(  # type: ignore
                    cm.exception
                )
            self.assertTrue(contains_errno, cm.exception)
            # This is usually "Connection refused".
            # On windows, strerror is broken and returns "Unknown error".
            expected_message = os.strerror(errno.ECONNREFUSED)
            self.assertTrue(expected_message in str(cm.exception), cm.exception)

    def test_queue_timeout(self: typing.Any):
        with closing(self.create_client(max_clients=1)) as client:
            # Wait for the trigger request to block, not complete.
            fut1 = client.fetch(self.get_url("/trigger"), request_timeout=10)
            self.wait()
            with self.assertRaises(HTTPTimeoutError) as cm:
                self.io_loop.run_sync(
                    lambda: client.fetch(
                        self.get_url("/hello"), connect_timeout=0.1, raise_error=True
                    )
                )

            self.assertEqual(str(cm.exception), "Timeout in request queue")
            self.triggers.popleft()()
            self.io_loop.run_sync(lambda: fut1)

    def test_no_content_length(self: typing.Any):
        response = self.fetch("/no_content_length")
        if response.body == b"HTTP/1 required":
            self.skipTest("requires HTTP/1.x")
        else:
            self.assertEqual(b"hello", response.body)

    def sync_body_producer(self, write):
        write(b"1234")
        write(b"5678")

    @gen.coroutine
    def async_body_producer(self, write):
        yield write(b"1234")
        yield gen.moment
        yield write(b"5678")

    def test_sync_body_producer_chunked(self: typing.Any):
        response = self.fetch(
            "/echo_post", method="POST", body_producer=self.sync_body_producer
        )
        response.rethrow()
        self.assertEqual(response.body, b"12345678")

    def test_sync_body_producer_content_length(self: typing.Any):
        response = self.fetch(
            "/echo_post",
            method="POST",
            body_producer=self.sync_body_producer,
            headers={"Content-Length": "8"},
        )
        response.rethrow()
        self.assertEqual(response.body, b"12345678")

    def test_async_body_producer_chunked(self: typing.Any):
        response = self.fetch(
            "/echo_post", method="POST", body_producer=self.async_body_producer
        )
        response.rethrow()
        self.assertEqual(response.body, b"12345678")

    def test_async_body_producer_content_length(self: typing.Any):
        response = self.fetch(
            "/echo_post",
            method="POST",
            body_producer=self.async_body_producer,
            headers={"Content-Length": "8"},
        )
        response.rethrow()
        self.assertEqual(response.body, b"12345678")

    def test_native_body_producer_chunked(self: typing.Any):
        async def body_producer(write):
            await write(b"1234")
            import asyncio

            await asyncio.sleep(0)
            await write(b"5678")

        response = self.fetch("/echo_post", method="POST", body_producer=body_producer)
        response.rethrow()
        self.assertEqual(response.body, b"12345678")

    def test_native_body_producer_content_length(self: typing.Any):
        async def body_producer(write):
            await write(b"1234")
            import asyncio

            await asyncio.sleep(0)
            await write(b"5678")

        response = self.fetch(
            "/echo_post",
            method="POST",
            body_producer=body_producer,
            headers={"Content-Length": "8"},
        )
        response.rethrow()
        self.assertEqual(response.body, b"12345678")

    def test_100_continue(self: typing.Any):
        response = self.fetch(
            "/echo_post", method="POST", body=b"1234", expect_100_continue=True
        )
        self.assertEqual(response.body, b"1234")

    def test_100_continue_early_response(self: typing.Any):
        def body_producer(write):
            raise Exception("should not be called")

        response = self.fetch(
            "/respond_in_prepare",
            method="POST",
            body_producer=body_producer,
            expect_100_continue=True,
        )
        self.assertEqual(response.code, 403)

    def test_streaming_follow_redirects(self: typing.Any):
        # When following redirects, header and streaming callbacks
        # should only be called for the final result.
        # TODO(bdarnell): this test belongs in httpclient_test instead of
        # simple_httpclient_test, but it fails with the version of libcurl
        # available on travis-ci. Move it when that has been upgraded
        # or we have a better framework to skip tests based on curl version.
        headers = []  # type: typing.List[str]
        chunk_bytes = []  # type: typing.List[bytes]
        self.fetch(
            "/redirect?url=/hello",
            header_callback=headers.append,
            streaming_callback=chunk_bytes.append,
        )
        chunks = list(map(to_unicode, chunk_bytes))
        self.assertEqual(chunks, ["Hello world!"])
        # Make sure we only got one set of headers.
        num_start_lines = len([h for h in headers if h.startswith("HTTP/")])
        self.assertEqual(num_start_lines, 1)


class SimpleHTTPClientTestCase(SimpleHTTPClientTestMixin, AsyncHTTPTestCase):
    def setUp(self):
        super().setUp()
        self.http_client = self.create_client()

    def create_client(self, **kwargs):
        return SimpleAsyncHTTPClient(force_instance=True, **kwargs)


class SimpleHTTPSClientTestCase(SimpleHTTPClientTestMixin, AsyncHTTPSTestCase):
    def setUp(self):
        super().setUp()
        self.http_client = self.create_client()

    def create_client(self, **kwargs):
        return SimpleAsyncHTTPClient(
            force_instance=True, defaults=dict(validate_cert=False), **kwargs
        )

    def test_ssl_options(self):
        resp = self.fetch("/hello", ssl_options={"cert_reqs": ssl.CERT_NONE})
        self.assertEqual(resp.body, b"Hello world!")

    def test_ssl_context(self):
        ssl_ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE
        resp = self.fetch("/hello", ssl_options=ssl_ctx)
        self.assertEqual(resp.body, b"Hello world!")

    def test_ssl_options_handshake_fail(self):
        with ExpectLog(gen_log, "SSL Error|Uncaught exception", required=False):
            with self.assertRaises(ssl.SSLError):
                self.fetch(
                    "/hello",
                    ssl_options=dict(cert_reqs=ssl.CERT_REQUIRED),
                    raise_error=True,
                )

    def test_ssl_context_handshake_fail(self):
        with ExpectLog(gen_log, "SSL Error|Uncaught exception"):
            # CERT_REQUIRED is set by default.
            ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
            with self.assertRaises(ssl.SSLError):
                self.fetch("/hello", ssl_options=ctx, raise_error=True)

    def test_error_logging(self):
        # No stack traces are logged for SSL errors (in this case,
        # failure to validate the testing self-signed cert).
        # The SSLError is exposed through ssl.SSLError.
        with ExpectLog(gen_log, ".*") as expect_log:
            with self.assertRaises(ssl.SSLError):
                self.fetch("/", validate_cert=True, raise_error=True)
        self.assertFalse(expect_log.logged_stack)


class CreateAsyncHTTPClientTestCase(AsyncTestCase):
    def setUp(self):
        super().setUp()
        self.saved = AsyncHTTPClient._save_configuration()

    def tearDown(self):
        AsyncHTTPClient._restore_configuration(self.saved)
        super().tearDown()

    def test_max_clients(self):
        AsyncHTTPClient.configure(SimpleAsyncHTTPClient)
        with closing(AsyncHTTPClient(force_instance=True)) as client:
            self.assertEqual(client.max_clients, 10)  # type: ignore
        with closing(AsyncHTTPClient(max_clients=11, force_instance=True)) as client:
            self.assertEqual(client.max_clients, 11)  # type: ignore

        # Now configure max_clients statically and try overriding it
        # with each way max_clients can be passed
        AsyncHTTPClient.configure(SimpleAsyncHTTPClient, max_clients=12)
        with closing(AsyncHTTPClient(force_instance=True)) as client:
            self.assertEqual(client.max_clients, 12)  # type: ignore
        with closing(AsyncHTTPClient(max_clients=13, force_instance=True)) as client:
            self.assertEqual(client.max_clients, 13)  # type: ignore
        with closing(AsyncHTTPClient(max_clients=14, force_instance=True)) as client:
            self.assertEqual(client.max_clients, 14)  # type: ignore


class HTTP100ContinueTestCase(AsyncHTTPTestCase):
    def respond_100(self, request):
        self.http1 = request.version.startswith("HTTP/1.")
        if not self.http1:
            request.connection.write_headers(
                ResponseStartLine("", 200, "OK"), HTTPHeaders()
            )
            request.connection.finish()
            return
        self.request = request
        fut = self.request.connection.stream.write(b"HTTP/1.1 100 CONTINUE\r\n\r\n")
        fut.add_done_callback(self.respond_200)

    def respond_200(self, fut):
        fut.result()
        fut = self.request.connection.stream.write(
            b"HTTP/1.1 200 OK\r\nContent-Length: 1\r\n\r\nA"
        )
        fut.add_done_callback(lambda f: self.request.connection.stream.close())

    def get_app(self):
        # Not a full Application, but works as an HTTPServer callback
        return self.respond_100

    def test_100_continue(self):
        res = self.fetch("/")
        if not self.http1:
            self.skipTest("requires HTTP/1.x")
        self.assertEqual(res.body, b"A")


class HTTP204NoContentTestCase(AsyncHTTPTestCase):
    def respond_204(self, request):
        self.http1 = request.version.startswith("HTTP/1.")
        if not self.http1:
            # Close the request cleanly in HTTP/2; it will be skipped anyway.
            request.connection.write_headers(
                ResponseStartLine("", 200, "OK"), HTTPHeaders()
            )
            request.connection.finish()
            return

        # A 204 response never has a body, even if doesn't have a content-length
        # (which would otherwise mean read-until-close).  We simulate here a
        # server that sends no content length and does not close the connection.
        #
        # Tests of a 204 response with no Content-Length header are included
        # in SimpleHTTPClientTestMixin.
        stream = request.connection.detach()
        stream.write(b"HTTP/1.1 204 No content\r\n")
        if request.arguments.get("error", [False])[-1]:
            stream.write(b"Content-Length: 5\r\n")
        else:
            stream.write(b"Content-Length: 0\r\n")
        stream.write(b"\r\n")
        stream.close()

    def get_app(self):
        return self.respond_204

    def test_204_no_content(self):
        resp = self.fetch("/")
        if not self.http1:
            self.skipTest("requires HTTP/1.x")
        self.assertEqual(resp.code, 204)
        self.assertEqual(resp.body, b"")

    def test_204_invalid_content_length(self):
        # 204 status with non-zero content length is malformed
        with ExpectLog(
            gen_log, ".*Response with code 204 should not have body", level=logging.INFO
        ):
            with self.assertRaises(HTTPStreamClosedError):
                self.fetch("/?error=1", raise_error=True)
                if not self.http1:
                    self.skipTest("requires HTTP/1.x")
                if self.http_client.configured_class != SimpleAsyncHTTPClient:
                    self.skipTest("curl client accepts invalid headers")


class HostnameMappingTestCase(AsyncHTTPTestCase):
    def setUp(self):
        super().setUp()
        self.http_client = SimpleAsyncHTTPClient(
            hostname_mapping={
                "www.example.com": "127.0.0.1",
                ("foo.example.com", 8000): ("127.0.0.1", self.get_http_port()),
            }
        )

    def get_app(self):
        return Application([url("/hello", HelloWorldHandler)])

    def test_hostname_mapping(self):
        response = self.fetch("http://www.example.com:%d/hello" % self.get_http_port())
        response.rethrow()
        self.assertEqual(response.body, b"Hello world!")

    def test_port_mapping(self):
        response = self.fetch("http://foo.example.com:8000/hello")
        response.rethrow()
        self.assertEqual(response.body, b"Hello world!")


class ResolveTimeoutTestCase(AsyncHTTPTestCase):
    def setUp(self):
        self.cleanup_event = Event()
        test = self

        # Dummy Resolver subclass that never finishes.
        class BadResolver(Resolver):
            @gen.coroutine
            def resolve(self, *args, **kwargs):
                yield test.cleanup_event.wait()
                # Return something valid so the test doesn't raise during cleanup.
                return [(socket.AF_INET, ("127.0.0.1", test.get_http_port()))]

        super().setUp()
        self.http_client = SimpleAsyncHTTPClient(resolver=BadResolver())

    def get_app(self):
        return Application([url("/hello", HelloWorldHandler)])

    def test_resolve_timeout(self):
        with self.assertRaises(HTTPTimeoutError):
            self.fetch("/hello", connect_timeout=0.1, raise_error=True)

        # Let the hanging coroutine clean up after itself
        self.cleanup_event.set()
        self.io_loop.run_sync(lambda: gen.sleep(0))


class MaxHeaderSizeTest(AsyncHTTPTestCase):
    def get_app(self):
        class SmallHeaders(RequestHandler):
            def get(self):
                self.set_header("X-Filler", "a" * 100)
                self.write("ok")

        class LargeHeaders(RequestHandler):
            def get(self):
                self.set_header("X-Filler", "a" * 1000)
                self.write("ok")

        return Application([("/small", SmallHeaders), ("/large", LargeHeaders)])

    def get_http_client(self):
        return SimpleAsyncHTTPClient(max_header_size=1024)

    def test_small_headers(self):
        response = self.fetch("/small")
        response.rethrow()
        self.assertEqual(response.body, b"ok")

    def test_large_headers(self):
        with ExpectLog(gen_log, "Unsatisfiable read", level=logging.INFO):
            with self.assertRaises(UnsatisfiableReadError):
                self.fetch("/large", raise_error=True)


class MaxBodySizeTest(AsyncHTTPTestCase):
    def get_app(self):
        class SmallBody(RequestHandler):
            def get(self):
                self.write("a" * 1024 * 64)

        class LargeBody(RequestHandler):
            def get(self):
                self.write("a" * 1024 * 100)

        return Application([("/small", SmallBody), ("/large", LargeBody)])

    def get_http_client(self):
        return SimpleAsyncHTTPClient(max_body_size=1024 * 64)

    def test_small_body(self):
        response = self.fetch("/small")
        response.rethrow()
        self.assertEqual(response.body, b"a" * 1024 * 64)

    def test_large_body(self):
        with ExpectLog(
            gen_log,
            "Malformed HTTP message from None: Content-Length too long",
            level=logging.INFO,
        ):
            with self.assertRaises(HTTPStreamClosedError):
                self.fetch("/large", raise_error=True)


class MaxBufferSizeTest(AsyncHTTPTestCase):
    def get_app(self):
        class LargeBody(RequestHandler):
            def get(self):
                self.write("a" * 1024 * 100)

        return Application([("/large", LargeBody)])

    def get_http_client(self):
        # 100KB body with 64KB buffer
        return SimpleAsyncHTTPClient(
            max_body_size=1024 * 100, max_buffer_size=1024 * 64
        )

    def test_large_body(self):
        response = self.fetch("/large")
        response.rethrow()
        self.assertEqual(response.body, b"a" * 1024 * 100)


class ChunkedWithContentLengthTest(AsyncHTTPTestCase):
    def get_app(self):
        class ChunkedWithContentLength(RequestHandler):
            def get(self):
                # Add an invalid Transfer-Encoding to the response
                self.set_header("Transfer-Encoding", "chunked")
                self.write("Hello world")

        return Application([("/chunkwithcl", ChunkedWithContentLength)])

    def get_http_client(self):
        return SimpleAsyncHTTPClient()

    def test_chunked_with_content_length(self):
        # Make sure the invalid headers are detected
        with ExpectLog(
            gen_log,
            (
                "Malformed HTTP message from None: Response "
                "with both Transfer-Encoding and Content-Length"
            ),
            level=logging.INFO,
        ):
            with self.assertRaises(HTTPStreamClosedError):
                self.fetch("/chunkwithcl", raise_error=True)
