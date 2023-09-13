from tornado import gen, netutil
from tornado.escape import (
    json_decode,
    json_encode,
    utf8,
    _unicode,
    recursive_unicode,
    native_str,
)
from tornado.http1connection import HTTP1Connection
from tornado.httpclient import HTTPError
from tornado.httpserver import HTTPServer
from tornado.httputil import (
    HTTPHeaders,
    HTTPMessageDelegate,
    HTTPServerConnectionDelegate,
    ResponseStartLine,
)
from tornado.iostream import IOStream
from tornado.locks import Event
from tornado.log import gen_log, app_log
from tornado.netutil import ssl_options_to_context
from tornado.simple_httpclient import SimpleAsyncHTTPClient
from tornado.testing import (
    AsyncHTTPTestCase,
    AsyncHTTPSTestCase,
    AsyncTestCase,
    ExpectLog,
    gen_test,
)
from tornado.test.util import skipOnTravis
from tornado.web import Application, RequestHandler, stream_request_body

from contextlib import closing
import datetime
import gzip
import logging
import os
import shutil
import socket
import ssl
import sys
import tempfile
import textwrap
import unittest
import urllib.parse
from io import BytesIO

import typing

if typing.TYPE_CHECKING:
    from typing import Dict, List  # noqa: F401


async def read_stream_body(stream):
    """Reads an HTTP response from `stream` and returns a tuple of its
    start_line, headers and body."""
    chunks = []

    class Delegate(HTTPMessageDelegate):
        def headers_received(self, start_line, headers):
            self.headers = headers
            self.start_line = start_line

        def data_received(self, chunk):
            chunks.append(chunk)

        def finish(self):
            conn.detach()  # type: ignore

    conn = HTTP1Connection(stream, True)
    delegate = Delegate()
    await conn.read_response(delegate)
    return delegate.start_line, delegate.headers, b"".join(chunks)


class HandlerBaseTestCase(AsyncHTTPTestCase):
    Handler = None

    def get_app(self):
        return Application([("/", self.__class__.Handler)])

    def fetch_json(self, *args, **kwargs):
        response = self.fetch(*args, **kwargs)
        response.rethrow()
        return json_decode(response.body)


class HelloWorldRequestHandler(RequestHandler):
    def initialize(self, protocol="http"):
        self.expected_protocol = protocol

    def get(self):
        if self.request.protocol != self.expected_protocol:
            raise Exception("unexpected protocol")
        self.finish("Hello world")

    def post(self):
        self.finish("Got %d bytes in POST" % len(self.request.body))


# In pre-1.0 versions of openssl, SSLv23 clients always send SSLv2
# ClientHello messages, which are rejected by SSLv3 and TLSv1
# servers.  Note that while the OPENSSL_VERSION_INFO was formally
# introduced in python3.2, it was present but undocumented in
# python 2.7
skipIfOldSSL = unittest.skipIf(
    getattr(ssl, "OPENSSL_VERSION_INFO", (0, 0)) < (1, 0),
    "old version of ssl module and/or openssl",
)


class BaseSSLTest(AsyncHTTPSTestCase):
    def get_app(self):
        return Application([("/", HelloWorldRequestHandler, dict(protocol="https"))])


class SSLTestMixin(object):
    def get_ssl_options(self):
        return dict(
            ssl_version=self.get_ssl_version(),
            **AsyncHTTPSTestCase.default_ssl_options(),
        )

    def get_ssl_version(self):
        raise NotImplementedError()

    def test_ssl(self: typing.Any):
        response = self.fetch("/")
        self.assertEqual(response.body, b"Hello world")

    def test_large_post(self: typing.Any):
        response = self.fetch("/", method="POST", body="A" * 5000)
        self.assertEqual(response.body, b"Got 5000 bytes in POST")

    def test_non_ssl_request(self: typing.Any):
        # Make sure the server closes the connection when it gets a non-ssl
        # connection, rather than waiting for a timeout or otherwise
        # misbehaving.
        with ExpectLog(gen_log, "(SSL Error|uncaught exception)"):
            with ExpectLog(gen_log, "Uncaught exception", required=False):
                with self.assertRaises((IOError, HTTPError)):  # type: ignore
                    self.fetch(
                        self.get_url("/").replace("https:", "http:"),
                        request_timeout=3600,
                        connect_timeout=3600,
                        raise_error=True,
                    )

    def test_error_logging(self: typing.Any):
        # No stack traces are logged for SSL errors.
        with ExpectLog(gen_log, "SSL Error") as expect_log:
            with self.assertRaises((IOError, HTTPError)):  # type: ignore
                self.fetch(
                    self.get_url("/").replace("https:", "http:"), raise_error=True
                )
        self.assertFalse(expect_log.logged_stack)


# Python's SSL implementation differs significantly between versions.
# For example, SSLv3 and TLSv1 throw an exception if you try to read
# from the socket before the handshake is complete, but the default
# of SSLv23 allows it.


class SSLv23Test(BaseSSLTest, SSLTestMixin):
    def get_ssl_version(self):
        return ssl.PROTOCOL_SSLv23


@skipIfOldSSL
class SSLv3Test(BaseSSLTest, SSLTestMixin):
    def get_ssl_version(self):
        return ssl.PROTOCOL_SSLv3


@skipIfOldSSL
class TLSv1Test(BaseSSLTest, SSLTestMixin):
    def get_ssl_version(self):
        return ssl.PROTOCOL_TLSv1


class SSLContextTest(BaseSSLTest, SSLTestMixin):
    def get_ssl_options(self):
        context = ssl_options_to_context(
            AsyncHTTPSTestCase.get_ssl_options(self), server_side=True
        )
        assert isinstance(context, ssl.SSLContext)
        return context


class BadSSLOptionsTest(unittest.TestCase):
    def test_missing_arguments(self):
        application = Application()
        self.assertRaises(
            KeyError,
            HTTPServer,
            application,
            ssl_options={"keyfile": "/__missing__.crt"},
        )

    def test_missing_key(self):
        """A missing SSL key should cause an immediate exception."""

        application = Application()
        module_dir = os.path.dirname(__file__)
        existing_certificate = os.path.join(module_dir, "test.crt")
        existing_key = os.path.join(module_dir, "test.key")

        self.assertRaises(
            (ValueError, IOError),
            HTTPServer,
            application,
            ssl_options={"certfile": "/__mising__.crt"},
        )
        self.assertRaises(
            (ValueError, IOError),
            HTTPServer,
            application,
            ssl_options={
                "certfile": existing_certificate,
                "keyfile": "/__missing__.key",
            },
        )

        # This actually works because both files exist
        HTTPServer(
            application,
            ssl_options={"certfile": existing_certificate, "keyfile": existing_key},
        )


class MultipartTestHandler(RequestHandler):
    def post(self):
        self.finish(
            {
                "header": self.request.headers["X-Header-Encoding-Test"],
                "argument": self.get_argument("argument"),
                "filename": self.request.files["files"][0].filename,
                "filebody": _unicode(self.request.files["files"][0]["body"]),
            }
        )


# This test is also called from wsgi_test
class HTTPConnectionTest(AsyncHTTPTestCase):
    def get_handlers(self):
        return [
            ("/multipart", MultipartTestHandler),
            ("/hello", HelloWorldRequestHandler),
        ]

    def get_app(self):
        return Application(self.get_handlers())

    def raw_fetch(self, headers, body, newline=b"\r\n"):
        with closing(IOStream(socket.socket())) as stream:
            self.io_loop.run_sync(
                lambda: stream.connect(("127.0.0.1", self.get_http_port()))
            )
            stream.write(
                newline.join(headers + [utf8("Content-Length: %d" % len(body))])
                + newline
                + newline
                + body
            )
            start_line, headers, body = self.io_loop.run_sync(
                lambda: read_stream_body(stream)
            )
            return body

    def test_multipart_form(self):
        # Encodings here are tricky:  Headers are latin1, bodies can be
        # anything (we use utf8 by default).
        response = self.raw_fetch(
            [
                b"POST /multipart HTTP/1.0",
                b"Content-Type: multipart/form-data; boundary=1234567890",
                b"X-Header-encoding-test: \xe9",
            ],
            b"\r\n".join(
                [
                    b"Content-Disposition: form-data; name=argument",
                    b"",
                    "\u00e1".encode("utf-8"),
                    b"--1234567890",
                    'Content-Disposition: form-data; name="files"; filename="\u00f3"'.encode(
                        "utf8"
                    ),
                    b"",
                    "\u00fa".encode("utf-8"),
                    b"--1234567890--",
                    b"",
                ]
            ),
        )
        data = json_decode(response)
        self.assertEqual("\u00e9", data["header"])
        self.assertEqual("\u00e1", data["argument"])
        self.assertEqual("\u00f3", data["filename"])
        self.assertEqual("\u00fa", data["filebody"])

    def test_newlines(self):
        # We support both CRLF and bare LF as line separators.
        for newline in (b"\r\n", b"\n"):
            response = self.raw_fetch([b"GET /hello HTTP/1.0"], b"", newline=newline)
            self.assertEqual(response, b"Hello world")

    @gen_test
    def test_100_continue(self):
        # Run through a 100-continue interaction by hand:
        # When given Expect: 100-continue, we get a 100 response after the
        # headers, and then the real response after the body.
        stream = IOStream(socket.socket())
        yield stream.connect(("127.0.0.1", self.get_http_port()))
        yield stream.write(
            b"\r\n".join(
                [
                    b"POST /hello HTTP/1.1",
                    b"Content-Length: 1024",
                    b"Expect: 100-continue",
                    b"Connection: close",
                    b"\r\n",
                ]
            )
        )
        data = yield stream.read_until(b"\r\n\r\n")
        self.assertTrue(data.startswith(b"HTTP/1.1 100 "), data)
        stream.write(b"a" * 1024)
        first_line = yield stream.read_until(b"\r\n")
        self.assertTrue(first_line.startswith(b"HTTP/1.1 200"), first_line)
        header_data = yield stream.read_until(b"\r\n\r\n")
        headers = HTTPHeaders.parse(native_str(header_data.decode("latin1")))
        body = yield stream.read_bytes(int(headers["Content-Length"]))
        self.assertEqual(body, b"Got 1024 bytes in POST")
        stream.close()


class EchoHandler(RequestHandler):
    def get(self):
        self.write(recursive_unicode(self.request.arguments))

    def post(self):
        self.write(recursive_unicode(self.request.arguments))


class TypeCheckHandler(RequestHandler):
    def prepare(self):
        self.errors = {}  # type: Dict[str, str]
        fields = [
            ("method", str),
            ("uri", str),
            ("version", str),
            ("remote_ip", str),
            ("protocol", str),
            ("host", str),
            ("path", str),
            ("query", str),
        ]
        for field, expected_type in fields:
            self.check_type(field, getattr(self.request, field), expected_type)

        self.check_type("header_key", list(self.request.headers.keys())[0], str)
        self.check_type("header_value", list(self.request.headers.values())[0], str)

        self.check_type("cookie_key", list(self.request.cookies.keys())[0], str)
        self.check_type(
            "cookie_value", list(self.request.cookies.values())[0].value, str
        )
        # secure cookies

        self.check_type("arg_key", list(self.request.arguments.keys())[0], str)
        self.check_type("arg_value", list(self.request.arguments.values())[0][0], bytes)

    def post(self):
        self.check_type("body", self.request.body, bytes)
        self.write(self.errors)

    def get(self):
        self.write(self.errors)

    def check_type(self, name, obj, expected_type):
        actual_type = type(obj)
        if expected_type != actual_type:
            self.errors[name] = "expected %s, got %s" % (expected_type, actual_type)


class PostEchoHandler(RequestHandler):
    def post(self, *path_args):
        self.write(dict(echo=self.get_argument("data")))


class PostEchoGBKHandler(PostEchoHandler):
    def decode_argument(self, value, name=None):
        try:
            return value.decode("gbk")
        except Exception:
            raise HTTPError(400, "invalid gbk bytes: %r" % value)


class HTTPServerTest(AsyncHTTPTestCase):
    def get_app(self):
        return Application(
            [
                ("/echo", EchoHandler),
                ("/typecheck", TypeCheckHandler),
                ("//doubleslash", EchoHandler),
                ("/post_utf8", PostEchoHandler),
                ("/post_gbk", PostEchoGBKHandler),
            ]
        )

    def test_query_string_encoding(self):
        response = self.fetch("/echo?foo=%C3%A9")
        data = json_decode(response.body)
        self.assertEqual(data, {"foo": ["\u00e9"]})

    def test_empty_query_string(self):
        response = self.fetch("/echo?foo=&foo=")
        data = json_decode(response.body)
        self.assertEqual(data, {"foo": ["", ""]})

    def test_empty_post_parameters(self):
        response = self.fetch("/echo", method="POST", body="foo=&bar=")
        data = json_decode(response.body)
        self.assertEqual(data, {"foo": [""], "bar": [""]})

    def test_types(self):
        headers = {"Cookie": "foo=bar"}
        response = self.fetch("/typecheck?foo=bar", headers=headers)
        data = json_decode(response.body)
        self.assertEqual(data, {})

        response = self.fetch(
            "/typecheck", method="POST", body="foo=bar", headers=headers
        )
        data = json_decode(response.body)
        self.assertEqual(data, {})

    def test_double_slash(self):
        # urlparse.urlsplit (which tornado.httpserver used to use
        # incorrectly) would parse paths beginning with "//" as
        # protocol-relative urls.
        response = self.fetch("//doubleslash")
        self.assertEqual(200, response.code)
        self.assertEqual(json_decode(response.body), {})

    def test_post_encodings(self):
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        uni_text = "chinese: \u5f20\u4e09"
        for enc in ("utf8", "gbk"):
            for quote in (True, False):
                with self.subTest(enc=enc, quote=quote):
                    bin_text = uni_text.encode(enc)
                    if quote:
                        bin_text = urllib.parse.quote(bin_text).encode("ascii")
                    response = self.fetch(
                        "/post_" + enc,
                        method="POST",
                        headers=headers,
                        body=(b"data=" + bin_text),
                    )
                    self.assertEqual(json_decode(response.body), {"echo": uni_text})


class HTTPServerRawTest(AsyncHTTPTestCase):
    def get_app(self):
        return Application([("/echo", EchoHandler)])

    def setUp(self):
        super().setUp()
        self.stream = IOStream(socket.socket())
        self.io_loop.run_sync(
            lambda: self.stream.connect(("127.0.0.1", self.get_http_port()))
        )

    def tearDown(self):
        self.stream.close()
        super().tearDown()

    def test_empty_request(self):
        self.stream.close()
        self.io_loop.add_timeout(datetime.timedelta(seconds=0.001), self.stop)
        self.wait()

    def test_malformed_first_line_response(self):
        with ExpectLog(gen_log, ".*Malformed HTTP request line", level=logging.INFO):
            self.stream.write(b"asdf\r\n\r\n")
            start_line, headers, response = self.io_loop.run_sync(
                lambda: read_stream_body(self.stream)
            )
            self.assertEqual("HTTP/1.1", start_line.version)
            self.assertEqual(400, start_line.code)
            self.assertEqual("Bad Request", start_line.reason)

    def test_malformed_first_line_log(self):
        with ExpectLog(gen_log, ".*Malformed HTTP request line", level=logging.INFO):
            self.stream.write(b"asdf\r\n\r\n")
            # TODO: need an async version of ExpectLog so we don't need
            # hard-coded timeouts here.
            self.io_loop.add_timeout(datetime.timedelta(seconds=0.05), self.stop)
            self.wait()

    def test_malformed_headers(self):
        with ExpectLog(
            gen_log,
            ".*Malformed HTTP message.*no colon in header line",
            level=logging.INFO,
        ):
            self.stream.write(b"GET / HTTP/1.0\r\nasdf\r\n\r\n")
            self.io_loop.add_timeout(datetime.timedelta(seconds=0.05), self.stop)
            self.wait()

    def test_chunked_request_body(self):
        # Chunked requests are not widely supported and we don't have a way
        # to generate them in AsyncHTTPClient, but HTTPServer will read them.
        self.stream.write(
            b"""\
POST /echo HTTP/1.1
Transfer-Encoding: chunked
Content-Type: application/x-www-form-urlencoded

4
foo=
3
bar
0

""".replace(
                b"\n", b"\r\n"
            )
        )
        start_line, headers, response = self.io_loop.run_sync(
            lambda: read_stream_body(self.stream)
        )
        self.assertEqual(json_decode(response), {"foo": ["bar"]})

    def test_chunked_request_uppercase(self):
        # As per RFC 2616 section 3.6, "Transfer-Encoding" header's value is
        # case-insensitive.
        self.stream.write(
            b"""\
POST /echo HTTP/1.1
Transfer-Encoding: Chunked
Content-Type: application/x-www-form-urlencoded

4
foo=
3
bar
0

""".replace(
                b"\n", b"\r\n"
            )
        )
        start_line, headers, response = self.io_loop.run_sync(
            lambda: read_stream_body(self.stream)
        )
        self.assertEqual(json_decode(response), {"foo": ["bar"]})

    def test_chunked_request_body_invalid_size(self):
        # Only hex digits are allowed in chunk sizes. Python's int() function
        # also accepts underscores, so make sure we reject them here.
        self.stream.write(
            b"""\
POST /echo HTTP/1.1
Transfer-Encoding: chunked

1_a
1234567890abcdef1234567890
0

""".replace(
                b"\n", b"\r\n"
            )
        )
        with ExpectLog(gen_log, ".*invalid chunk size", level=logging.INFO):
            start_line, headers, response = self.io_loop.run_sync(
                lambda: read_stream_body(self.stream)
            )
        self.assertEqual(400, start_line.code)

    @gen_test
    def test_invalid_content_length(self):
        # HTTP only allows decimal digits in content-length. Make sure we don't
        # accept anything else, with special attention to things accepted by the
        # python int() function (leading plus signs and internal underscores).
        test_cases = [
            ("alphabetic", "foo"),
            ("leading plus", "+10"),
            ("internal underscore", "1_0"),
        ]
        for name, value in test_cases:
            with self.subTest(name=name), closing(IOStream(socket.socket())) as stream:
                with ExpectLog(
                    gen_log,
                    ".*Only integer Content-Length is allowed",
                    level=logging.INFO,
                ):
                    yield stream.connect(("127.0.0.1", self.get_http_port()))
                    stream.write(
                        utf8(
                            textwrap.dedent(
                                f"""\
                            POST /echo HTTP/1.1
                            Content-Length: {value}
                            Connection: close

                            1234567890
                            """
                            ).replace("\n", "\r\n")
                        )
                    )
                    yield stream.read_until_close()


class XHeaderTest(HandlerBaseTestCase):
    class Handler(RequestHandler):
        def get(self):
            self.set_header("request-version", self.request.version)
            self.write(
                dict(
                    remote_ip=self.request.remote_ip,
                    remote_protocol=self.request.protocol,
                )
            )

    def get_httpserver_options(self):
        return dict(xheaders=True, trusted_downstream=["5.5.5.5"])

    def test_ip_headers(self):
        self.assertEqual(self.fetch_json("/")["remote_ip"], "127.0.0.1")

        valid_ipv4 = {"X-Real-IP": "4.4.4.4"}
        self.assertEqual(
            self.fetch_json("/", headers=valid_ipv4)["remote_ip"], "4.4.4.4"
        )

        valid_ipv4_list = {"X-Forwarded-For": "127.0.0.1, 4.4.4.4"}
        self.assertEqual(
            self.fetch_json("/", headers=valid_ipv4_list)["remote_ip"], "4.4.4.4"
        )

        valid_ipv6 = {"X-Real-IP": "2620:0:1cfe:face:b00c::3"}
        self.assertEqual(
            self.fetch_json("/", headers=valid_ipv6)["remote_ip"],
            "2620:0:1cfe:face:b00c::3",
        )

        valid_ipv6_list = {"X-Forwarded-For": "::1, 2620:0:1cfe:face:b00c::3"}
        self.assertEqual(
            self.fetch_json("/", headers=valid_ipv6_list)["remote_ip"],
            "2620:0:1cfe:face:b00c::3",
        )

        invalid_chars = {"X-Real-IP": "4.4.4.4<script>"}
        self.assertEqual(
            self.fetch_json("/", headers=invalid_chars)["remote_ip"], "127.0.0.1"
        )

        invalid_chars_list = {"X-Forwarded-For": "4.4.4.4, 5.5.5.5<script>"}
        self.assertEqual(
            self.fetch_json("/", headers=invalid_chars_list)["remote_ip"], "127.0.0.1"
        )

        invalid_host = {"X-Real-IP": "www.google.com"}
        self.assertEqual(
            self.fetch_json("/", headers=invalid_host)["remote_ip"], "127.0.0.1"
        )

    def test_trusted_downstream(self):
        valid_ipv4_list = {"X-Forwarded-For": "127.0.0.1, 4.4.4.4, 5.5.5.5"}
        resp = self.fetch("/", headers=valid_ipv4_list)
        if resp.headers["request-version"].startswith("HTTP/2"):
            # This is a hack - there's nothing that fundamentally requires http/1
            # here but tornado_http2 doesn't support it yet.
            self.skipTest("requires HTTP/1.x")
        result = json_decode(resp.body)
        self.assertEqual(result["remote_ip"], "4.4.4.4")

    def test_scheme_headers(self):
        self.assertEqual(self.fetch_json("/")["remote_protocol"], "http")

        https_scheme = {"X-Scheme": "https"}
        self.assertEqual(
            self.fetch_json("/", headers=https_scheme)["remote_protocol"], "https"
        )

        https_forwarded = {"X-Forwarded-Proto": "https"}
        self.assertEqual(
            self.fetch_json("/", headers=https_forwarded)["remote_protocol"], "https"
        )

        https_multi_forwarded = {"X-Forwarded-Proto": "https , http"}
        self.assertEqual(
            self.fetch_json("/", headers=https_multi_forwarded)["remote_protocol"],
            "http",
        )

        http_multi_forwarded = {"X-Forwarded-Proto": "http,https"}
        self.assertEqual(
            self.fetch_json("/", headers=http_multi_forwarded)["remote_protocol"],
            "https",
        )

        bad_forwarded = {"X-Forwarded-Proto": "unknown"}
        self.assertEqual(
            self.fetch_json("/", headers=bad_forwarded)["remote_protocol"], "http"
        )


class SSLXHeaderTest(AsyncHTTPSTestCase, HandlerBaseTestCase):
    def get_app(self):
        return Application([("/", XHeaderTest.Handler)])

    def get_httpserver_options(self):
        output = super().get_httpserver_options()
        output["xheaders"] = True
        return output

    def test_request_without_xprotocol(self):
        self.assertEqual(self.fetch_json("/")["remote_protocol"], "https")

        http_scheme = {"X-Scheme": "http"}
        self.assertEqual(
            self.fetch_json("/", headers=http_scheme)["remote_protocol"], "http"
        )

        bad_scheme = {"X-Scheme": "unknown"}
        self.assertEqual(
            self.fetch_json("/", headers=bad_scheme)["remote_protocol"], "https"
        )


class ManualProtocolTest(HandlerBaseTestCase):
    class Handler(RequestHandler):
        def get(self):
            self.write(dict(protocol=self.request.protocol))

    def get_httpserver_options(self):
        return dict(protocol="https")

    def test_manual_protocol(self):
        self.assertEqual(self.fetch_json("/")["protocol"], "https")


@unittest.skipIf(
    not hasattr(socket, "AF_UNIX") or sys.platform == "cygwin",
    "unix sockets not supported on this platform",
)
class UnixSocketTest(AsyncTestCase):
    """HTTPServers can listen on Unix sockets too.

    Why would you want to do this?  Nginx can proxy to backends listening
    on unix sockets, for one thing (and managing a namespace for unix
    sockets can be easier than managing a bunch of TCP port numbers).

    Unfortunately, there's no way to specify a unix socket in a url for
    an HTTP client, so we have to test this by hand.
    """

    def setUp(self):
        super().setUp()
        self.tmpdir = tempfile.mkdtemp()
        self.sockfile = os.path.join(self.tmpdir, "test.sock")
        sock = netutil.bind_unix_socket(self.sockfile)
        app = Application([("/hello", HelloWorldRequestHandler)])
        self.server = HTTPServer(app)
        self.server.add_socket(sock)
        self.stream = IOStream(socket.socket(socket.AF_UNIX))
        self.io_loop.run_sync(lambda: self.stream.connect(self.sockfile))

    def tearDown(self):
        self.stream.close()
        self.io_loop.run_sync(self.server.close_all_connections)
        self.server.stop()
        shutil.rmtree(self.tmpdir)
        super().tearDown()

    @gen_test
    def test_unix_socket(self):
        self.stream.write(b"GET /hello HTTP/1.0\r\n\r\n")
        response = yield self.stream.read_until(b"\r\n")
        self.assertEqual(response, b"HTTP/1.1 200 OK\r\n")
        header_data = yield self.stream.read_until(b"\r\n\r\n")
        headers = HTTPHeaders.parse(header_data.decode("latin1"))
        body = yield self.stream.read_bytes(int(headers["Content-Length"]))
        self.assertEqual(body, b"Hello world")

    @gen_test
    def test_unix_socket_bad_request(self):
        # Unix sockets don't have remote addresses so they just return an
        # empty string.
        with ExpectLog(gen_log, "Malformed HTTP message from", level=logging.INFO):
            self.stream.write(b"garbage\r\n\r\n")
            response = yield self.stream.read_until_close()
        self.assertEqual(response, b"HTTP/1.1 400 Bad Request\r\n\r\n")


class KeepAliveTest(AsyncHTTPTestCase):
    """Tests various scenarios for HTTP 1.1 keep-alive support.

    These tests don't use AsyncHTTPClient because we want to control
    connection reuse and closing.
    """

    def get_app(self):
        class HelloHandler(RequestHandler):
            def get(self):
                self.finish("Hello world")

            def post(self):
                self.finish("Hello world")

        class LargeHandler(RequestHandler):
            def get(self):
                # 512KB should be bigger than the socket buffers so it will
                # be written out in chunks.
                self.write("".join(chr(i % 256) * 1024 for i in range(512)))

        class TransferEncodingChunkedHandler(RequestHandler):
            @gen.coroutine
            def head(self):
                self.write("Hello world")
                yield self.flush()

        class FinishOnCloseHandler(RequestHandler):
            def initialize(self, cleanup_event):
                self.cleanup_event = cleanup_event

            @gen.coroutine
            def get(self):
                self.flush()
                yield self.cleanup_event.wait()

            def on_connection_close(self):
                # This is not very realistic, but finishing the request
                # from the close callback has the right timing to mimic
                # some errors seen in the wild.
                self.finish("closed")

        self.cleanup_event = Event()
        return Application(
            [
                ("/", HelloHandler),
                ("/large", LargeHandler),
                ("/chunked", TransferEncodingChunkedHandler),
                (
                    "/finish_on_close",
                    FinishOnCloseHandler,
                    dict(cleanup_event=self.cleanup_event),
                ),
            ]
        )

    def setUp(self):
        super().setUp()
        self.http_version = b"HTTP/1.1"

    def tearDown(self):
        # We just closed the client side of the socket; let the IOLoop run
        # once to make sure the server side got the message.
        self.io_loop.add_timeout(datetime.timedelta(seconds=0.001), self.stop)
        self.wait()

        if hasattr(self, "stream"):
            self.stream.close()
        super().tearDown()

    # The next few methods are a crude manual http client
    @gen.coroutine
    def connect(self):
        self.stream = IOStream(socket.socket())
        yield self.stream.connect(("127.0.0.1", self.get_http_port()))

    @gen.coroutine
    def read_headers(self):
        first_line = yield self.stream.read_until(b"\r\n")
        self.assertTrue(first_line.startswith(b"HTTP/1.1 200"), first_line)
        header_bytes = yield self.stream.read_until(b"\r\n\r\n")
        headers = HTTPHeaders.parse(header_bytes.decode("latin1"))
        raise gen.Return(headers)

    @gen.coroutine
    def read_response(self):
        self.headers = yield self.read_headers()
        body = yield self.stream.read_bytes(int(self.headers["Content-Length"]))
        self.assertEqual(b"Hello world", body)

    def close(self):
        self.stream.close()
        del self.stream

    @gen_test
    def test_two_requests(self):
        yield self.connect()
        self.stream.write(b"GET / HTTP/1.1\r\n\r\n")
        yield self.read_response()
        self.stream.write(b"GET / HTTP/1.1\r\n\r\n")
        yield self.read_response()
        self.close()

    @gen_test
    def test_request_close(self):
        yield self.connect()
        self.stream.write(b"GET / HTTP/1.1\r\nConnection: close\r\n\r\n")
        yield self.read_response()
        data = yield self.stream.read_until_close()
        self.assertTrue(not data)
        self.assertEqual(self.headers["Connection"], "close")
        self.close()

    # keepalive is supported for http 1.0 too, but it's opt-in
    @gen_test
    def test_http10(self):
        self.http_version = b"HTTP/1.0"
        yield self.connect()
        self.stream.write(b"GET / HTTP/1.0\r\n\r\n")
        yield self.read_response()
        data = yield self.stream.read_until_close()
        self.assertTrue(not data)
        self.assertTrue("Connection" not in self.headers)
        self.close()

    @gen_test
    def test_http10_keepalive(self):
        self.http_version = b"HTTP/1.0"
        yield self.connect()
        self.stream.write(b"GET / HTTP/1.0\r\nConnection: keep-alive\r\n\r\n")
        yield self.read_response()
        self.assertEqual(self.headers["Connection"], "Keep-Alive")
        self.stream.write(b"GET / HTTP/1.0\r\nConnection: keep-alive\r\n\r\n")
        yield self.read_response()
        self.assertEqual(self.headers["Connection"], "Keep-Alive")
        self.close()

    @gen_test
    def test_http10_keepalive_extra_crlf(self):
        self.http_version = b"HTTP/1.0"
        yield self.connect()
        self.stream.write(b"GET / HTTP/1.0\r\nConnection: keep-alive\r\n\r\n\r\n")
        yield self.read_response()
        self.assertEqual(self.headers["Connection"], "Keep-Alive")
        self.stream.write(b"GET / HTTP/1.0\r\nConnection: keep-alive\r\n\r\n")
        yield self.read_response()
        self.assertEqual(self.headers["Connection"], "Keep-Alive")
        self.close()

    @gen_test
    def test_pipelined_requests(self):
        yield self.connect()
        self.stream.write(b"GET / HTTP/1.1\r\n\r\nGET / HTTP/1.1\r\n\r\n")
        yield self.read_response()
        yield self.read_response()
        self.close()

    @gen_test
    def test_pipelined_cancel(self):
        yield self.connect()
        self.stream.write(b"GET / HTTP/1.1\r\n\r\nGET / HTTP/1.1\r\n\r\n")
        # only read once
        yield self.read_response()
        self.close()

    @gen_test
    def test_cancel_during_download(self):
        yield self.connect()
        self.stream.write(b"GET /large HTTP/1.1\r\n\r\n")
        yield self.read_headers()
        yield self.stream.read_bytes(1024)
        self.close()

    @gen_test
    def test_finish_while_closed(self):
        yield self.connect()
        self.stream.write(b"GET /finish_on_close HTTP/1.1\r\n\r\n")
        yield self.read_headers()
        self.close()
        # Let the hanging coroutine clean up after itself
        self.cleanup_event.set()

    @gen_test
    def test_keepalive_chunked(self):
        self.http_version = b"HTTP/1.0"
        yield self.connect()
        self.stream.write(
            b"POST / HTTP/1.0\r\n"
            b"Connection: keep-alive\r\n"
            b"Transfer-Encoding: chunked\r\n"
            b"\r\n"
            b"0\r\n"
            b"\r\n"
        )
        yield self.read_response()
        self.assertEqual(self.headers["Connection"], "Keep-Alive")
        self.stream.write(b"GET / HTTP/1.0\r\nConnection: keep-alive\r\n\r\n")
        yield self.read_response()
        self.assertEqual(self.headers["Connection"], "Keep-Alive")
        self.close()

    @gen_test
    def test_keepalive_chunked_head_no_body(self):
        yield self.connect()
        self.stream.write(b"HEAD /chunked HTTP/1.1\r\n\r\n")
        yield self.read_headers()

        self.stream.write(b"HEAD /chunked HTTP/1.1\r\n\r\n")
        yield self.read_headers()
        self.close()


class GzipBaseTest(AsyncHTTPTestCase):
    def get_app(self):
        return Application([("/", EchoHandler)])

    def post_gzip(self, body):
        bytesio = BytesIO()
        gzip_file = gzip.GzipFile(mode="w", fileobj=bytesio)
        gzip_file.write(utf8(body))
        gzip_file.close()
        compressed_body = bytesio.getvalue()
        return self.fetch(
            "/",
            method="POST",
            body=compressed_body,
            headers={"Content-Encoding": "gzip"},
        )

    def test_uncompressed(self):
        response = self.fetch("/", method="POST", body="foo=bar")
        self.assertEqual(json_decode(response.body), {"foo": ["bar"]})


class GzipTest(GzipBaseTest, AsyncHTTPTestCase):
    def get_httpserver_options(self):
        return dict(decompress_request=True)

    def test_gzip(self):
        response = self.post_gzip("foo=bar")
        self.assertEqual(json_decode(response.body), {"foo": ["bar"]})

    def test_gzip_case_insensitive(self):
        # https://datatracker.ietf.org/doc/html/rfc7231#section-3.1.2.1
        bytesio = BytesIO()
        gzip_file = gzip.GzipFile(mode="w", fileobj=bytesio)
        gzip_file.write(utf8("foo=bar"))
        gzip_file.close()
        compressed_body = bytesio.getvalue()
        response = self.fetch(
            "/",
            method="POST",
            body=compressed_body,
            headers={"Content-Encoding": "GZIP"},
        )
        self.assertEqual(json_decode(response.body), {"foo": ["bar"]})


class GzipUnsupportedTest(GzipBaseTest, AsyncHTTPTestCase):
    def test_gzip_unsupported(self):
        # Gzip support is opt-in; without it the server fails to parse
        # the body (but parsing form bodies is currently just a log message,
        # not a fatal error).
        with ExpectLog(gen_log, "Unsupported Content-Encoding"):
            response = self.post_gzip("foo=bar")
        self.assertEqual(json_decode(response.body), {})


class StreamingChunkSizeTest(AsyncHTTPTestCase):
    # 50 characters long, and repetitive so it can be compressed.
    BODY = b"01234567890123456789012345678901234567890123456789"
    CHUNK_SIZE = 16

    def get_http_client(self):
        # body_producer doesn't work on curl_httpclient, so override the
        # configured AsyncHTTPClient implementation.
        return SimpleAsyncHTTPClient()

    def get_httpserver_options(self):
        return dict(chunk_size=self.CHUNK_SIZE, decompress_request=True)

    class MessageDelegate(HTTPMessageDelegate):
        def __init__(self, connection):
            self.connection = connection

        def headers_received(self, start_line, headers):
            self.chunk_lengths = []  # type: List[int]

        def data_received(self, chunk):
            self.chunk_lengths.append(len(chunk))

        def finish(self):
            response_body = utf8(json_encode(self.chunk_lengths))
            self.connection.write_headers(
                ResponseStartLine("HTTP/1.1", 200, "OK"),
                HTTPHeaders({"Content-Length": str(len(response_body))}),
            )
            self.connection.write(response_body)
            self.connection.finish()

    def get_app(self):
        class App(HTTPServerConnectionDelegate):
            def start_request(self, server_conn, request_conn):
                return StreamingChunkSizeTest.MessageDelegate(request_conn)

        return App()

    def fetch_chunk_sizes(self, **kwargs):
        response = self.fetch("/", method="POST", **kwargs)
        response.rethrow()
        chunks = json_decode(response.body)
        self.assertEqual(len(self.BODY), sum(chunks))
        for chunk_size in chunks:
            self.assertLessEqual(
                chunk_size, self.CHUNK_SIZE, "oversized chunk: " + str(chunks)
            )
            self.assertGreater(chunk_size, 0, "empty chunk: " + str(chunks))
        return chunks

    def compress(self, body):
        bytesio = BytesIO()
        gzfile = gzip.GzipFile(mode="w", fileobj=bytesio)
        gzfile.write(body)
        gzfile.close()
        compressed = bytesio.getvalue()
        if len(compressed) >= len(body):
            raise Exception("body did not shrink when compressed")
        return compressed

    def test_regular_body(self):
        chunks = self.fetch_chunk_sizes(body=self.BODY)
        # Without compression we know exactly what to expect.
        self.assertEqual([16, 16, 16, 2], chunks)

    def test_compressed_body(self):
        self.fetch_chunk_sizes(
            body=self.compress(self.BODY), headers={"Content-Encoding": "gzip"}
        )
        # Compression creates irregular boundaries so the assertions
        # in fetch_chunk_sizes are as specific as we can get.

    def test_chunked_body(self):
        def body_producer(write):
            write(self.BODY[:20])
            write(self.BODY[20:])

        chunks = self.fetch_chunk_sizes(body_producer=body_producer)
        # HTTP chunk boundaries translate to application-visible breaks
        self.assertEqual([16, 4, 16, 14], chunks)

    def test_chunked_compressed(self):
        compressed = self.compress(self.BODY)
        self.assertGreater(len(compressed), 20)

        def body_producer(write):
            write(compressed[:20])
            write(compressed[20:])

        self.fetch_chunk_sizes(
            body_producer=body_producer, headers={"Content-Encoding": "gzip"}
        )


class InvalidOutputContentLengthTest(AsyncHTTPTestCase):
    class MessageDelegate(HTTPMessageDelegate):
        def __init__(self, connection):
            self.connection = connection

        def headers_received(self, start_line, headers):
            content_lengths = {
                "normal": "10",
                "alphabetic": "foo",
                "leading plus": "+10",
                "underscore": "1_0",
            }
            self.connection.write_headers(
                ResponseStartLine("HTTP/1.1", 200, "OK"),
                HTTPHeaders({"Content-Length": content_lengths[headers["x-test"]]}),
            )
            self.connection.write(b"1234567890")
            self.connection.finish()

    def get_app(self):
        class App(HTTPServerConnectionDelegate):
            def start_request(self, server_conn, request_conn):
                return InvalidOutputContentLengthTest.MessageDelegate(request_conn)

        return App()

    def test_invalid_output_content_length(self):
        with self.subTest("normal"):
            response = self.fetch("/", method="GET", headers={"x-test": "normal"})
            response.rethrow()
            self.assertEqual(response.body, b"1234567890")
        for test in ["alphabetic", "leading plus", "underscore"]:
            with self.subTest(test):
                # This log matching could be tighter but I think I'm already
                # over-testing here.
                with ExpectLog(app_log, "Uncaught exception"):
                    with self.assertRaises(HTTPError):
                        self.fetch("/", method="GET", headers={"x-test": test})


class MaxHeaderSizeTest(AsyncHTTPTestCase):
    def get_app(self):
        return Application([("/", HelloWorldRequestHandler)])

    def get_httpserver_options(self):
        return dict(max_header_size=1024)

    def test_small_headers(self):
        response = self.fetch("/", headers={"X-Filler": "a" * 100})
        response.rethrow()
        self.assertEqual(response.body, b"Hello world")

    def test_large_headers(self):
        with ExpectLog(gen_log, "Unsatisfiable read", required=False):
            try:
                self.fetch("/", headers={"X-Filler": "a" * 1000}, raise_error=True)
                self.fail("did not raise expected exception")
            except HTTPError as e:
                # 431 is "Request Header Fields Too Large", defined in RFC
                # 6585. However, many implementations just close the
                # connection in this case, resulting in a missing response.
                if e.response is not None:
                    self.assertIn(e.response.code, (431, 599))


@skipOnTravis
class IdleTimeoutTest(AsyncHTTPTestCase):
    def get_app(self):
        return Application([("/", HelloWorldRequestHandler)])

    def get_httpserver_options(self):
        return dict(idle_connection_timeout=0.1)

    def setUp(self):
        super().setUp()
        self.streams = []  # type: List[IOStream]

    def tearDown(self):
        super().tearDown()
        for stream in self.streams:
            stream.close()

    @gen.coroutine
    def connect(self):
        stream = IOStream(socket.socket())
        yield stream.connect(("127.0.0.1", self.get_http_port()))
        self.streams.append(stream)
        raise gen.Return(stream)

    @gen_test
    def test_unused_connection(self):
        stream = yield self.connect()
        event = Event()
        stream.set_close_callback(event.set)
        yield event.wait()

    @gen_test
    def test_idle_after_use(self):
        stream = yield self.connect()
        event = Event()
        stream.set_close_callback(event.set)

        # Use the connection twice to make sure keep-alives are working
        for i in range(2):
            stream.write(b"GET / HTTP/1.1\r\n\r\n")
            yield stream.read_until(b"\r\n\r\n")
            data = yield stream.read_bytes(11)
            self.assertEqual(data, b"Hello world")

        # Now let the timeout trigger and close the connection.
        yield event.wait()


class BodyLimitsTest(AsyncHTTPTestCase):
    def get_app(self):
        class BufferedHandler(RequestHandler):
            def put(self):
                self.write(str(len(self.request.body)))

        @stream_request_body
        class StreamingHandler(RequestHandler):
            def initialize(self):
                self.bytes_read = 0

            def prepare(self):
                conn = typing.cast(HTTP1Connection, self.request.connection)
                if "expected_size" in self.request.arguments:
                    conn.set_max_body_size(int(self.get_argument("expected_size")))
                if "body_timeout" in self.request.arguments:
                    conn.set_body_timeout(float(self.get_argument("body_timeout")))

            def data_received(self, data):
                self.bytes_read += len(data)

            def put(self):
                self.write(str(self.bytes_read))

        return Application(
            [("/buffered", BufferedHandler), ("/streaming", StreamingHandler)]
        )

    def get_httpserver_options(self):
        return dict(body_timeout=3600, max_body_size=4096)

    def get_http_client(self):
        # body_producer doesn't work on curl_httpclient, so override the
        # configured AsyncHTTPClient implementation.
        return SimpleAsyncHTTPClient()

    def test_small_body(self):
        response = self.fetch("/buffered", method="PUT", body=b"a" * 4096)
        self.assertEqual(response.body, b"4096")
        response = self.fetch("/streaming", method="PUT", body=b"a" * 4096)
        self.assertEqual(response.body, b"4096")

    def test_large_body_buffered(self):
        with ExpectLog(gen_log, ".*Content-Length too long", level=logging.INFO):
            response = self.fetch("/buffered", method="PUT", body=b"a" * 10240)
        self.assertEqual(response.code, 400)

    @unittest.skipIf(os.name == "nt", "flaky on windows")
    def test_large_body_buffered_chunked(self):
        # This test is flaky on windows for unknown reasons.
        with ExpectLog(gen_log, ".*chunked body too large", level=logging.INFO):
            response = self.fetch(
                "/buffered",
                method="PUT",
                body_producer=lambda write: write(b"a" * 10240),
            )
        self.assertEqual(response.code, 400)

    def test_large_body_streaming(self):
        with ExpectLog(gen_log, ".*Content-Length too long", level=logging.INFO):
            response = self.fetch("/streaming", method="PUT", body=b"a" * 10240)
        self.assertEqual(response.code, 400)

    @unittest.skipIf(os.name == "nt", "flaky on windows")
    def test_large_body_streaming_chunked(self):
        with ExpectLog(gen_log, ".*chunked body too large", level=logging.INFO):
            response = self.fetch(
                "/streaming",
                method="PUT",
                body_producer=lambda write: write(b"a" * 10240),
            )
        self.assertEqual(response.code, 400)

    def test_large_body_streaming_override(self):
        response = self.fetch(
            "/streaming?expected_size=10240", method="PUT", body=b"a" * 10240
        )
        self.assertEqual(response.body, b"10240")

    def test_large_body_streaming_chunked_override(self):
        response = self.fetch(
            "/streaming?expected_size=10240",
            method="PUT",
            body_producer=lambda write: write(b"a" * 10240),
        )
        self.assertEqual(response.body, b"10240")

    @gen_test
    def test_timeout(self):
        stream = IOStream(socket.socket())
        try:
            yield stream.connect(("127.0.0.1", self.get_http_port()))
            # Use a raw stream because AsyncHTTPClient won't let us read a
            # response without finishing a body.
            stream.write(
                b"PUT /streaming?body_timeout=0.1 HTTP/1.0\r\n"
                b"Content-Length: 42\r\n\r\n"
            )
            with ExpectLog(gen_log, "Timeout reading body", level=logging.INFO):
                response = yield stream.read_until_close()
            self.assertEqual(response, b"")
        finally:
            stream.close()

    @gen_test
    def test_body_size_override_reset(self):
        # The max_body_size override is reset between requests.
        stream = IOStream(socket.socket())
        try:
            yield stream.connect(("127.0.0.1", self.get_http_port()))
            # Use a raw stream so we can make sure it's all on one connection.
            stream.write(
                b"PUT /streaming?expected_size=10240 HTTP/1.1\r\n"
                b"Content-Length: 10240\r\n\r\n"
            )
            stream.write(b"a" * 10240)
            start_line, headers, response = yield read_stream_body(stream)
            self.assertEqual(response, b"10240")
            # Without the ?expected_size parameter, we get the old default value
            stream.write(
                b"PUT /streaming HTTP/1.1\r\n" b"Content-Length: 10240\r\n\r\n"
            )
            with ExpectLog(gen_log, ".*Content-Length too long", level=logging.INFO):
                data = yield stream.read_until_close()
            self.assertEqual(data, b"HTTP/1.1 400 Bad Request\r\n\r\n")
        finally:
            stream.close()


class LegacyInterfaceTest(AsyncHTTPTestCase):
    def get_app(self):
        # The old request_callback interface does not implement the
        # delegate interface, and writes its response via request.write
        # instead of request.connection.write_headers.
        def handle_request(request):
            self.http1 = request.version.startswith("HTTP/1.")
            if not self.http1:
                # This test will be skipped if we're using HTTP/2,
                # so just close it out cleanly using the modern interface.
                request.connection.write_headers(
                    ResponseStartLine("", 200, "OK"), HTTPHeaders()
                )
                request.connection.finish()
                return
            message = b"Hello world"
            request.connection.write(
                utf8("HTTP/1.1 200 OK\r\n" "Content-Length: %d\r\n\r\n" % len(message))
            )
            request.connection.write(message)
            request.connection.finish()

        return handle_request

    def test_legacy_interface(self):
        response = self.fetch("/")
        if not self.http1:
            self.skipTest("requires HTTP/1.x")
        self.assertEqual(response.body, b"Hello world")
