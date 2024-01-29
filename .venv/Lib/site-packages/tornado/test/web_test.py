from tornado.concurrent import Future
from tornado import gen
from tornado.escape import (
    json_decode,
    utf8,
    to_unicode,
    recursive_unicode,
    native_str,
    to_basestring,
)
from tornado.httpclient import HTTPClientError
from tornado.httputil import format_timestamp
from tornado.iostream import IOStream
from tornado import locale
from tornado.locks import Event
from tornado.log import app_log, gen_log
from tornado.simple_httpclient import SimpleAsyncHTTPClient
from tornado.template import DictLoader
from tornado.testing import AsyncHTTPTestCase, AsyncTestCase, ExpectLog, gen_test
from tornado.test.util import ignore_deprecation
from tornado.util import ObjectDict, unicode_type
from tornado.web import (
    Application,
    RequestHandler,
    StaticFileHandler,
    RedirectHandler as WebRedirectHandler,
    HTTPError,
    MissingArgumentError,
    ErrorHandler,
    authenticated,
    url,
    _create_signature_v1,
    create_signed_value,
    decode_signed_value,
    get_signature_key_version,
    UIModule,
    Finish,
    stream_request_body,
    removeslash,
    addslash,
    GZipContentEncoding,
)

import binascii
import contextlib
import copy
import datetime
import email.utils
import gzip
from io import BytesIO
import itertools
import logging
import os
import re
import socket
import typing  # noqa: F401
import unittest
import urllib.parse


def relpath(*a):
    return os.path.join(os.path.dirname(__file__), *a)


class WebTestCase(AsyncHTTPTestCase):
    """Base class for web tests that also supports WSGI mode.

    Override get_handlers and get_app_kwargs instead of get_app.
    This class is deprecated since WSGI mode is no longer supported.
    """

    def get_app(self):
        self.app = Application(self.get_handlers(), **self.get_app_kwargs())
        return self.app

    def get_handlers(self):
        raise NotImplementedError()

    def get_app_kwargs(self):
        return {}


class SimpleHandlerTestCase(WebTestCase):
    """Simplified base class for tests that work with a single handler class.

    To use, define a nested class named ``Handler``.
    """

    Handler = None

    def get_handlers(self):
        return [("/", self.Handler)]


class HelloHandler(RequestHandler):
    def get(self):
        self.write("hello")


class CookieTestRequestHandler(RequestHandler):
    # stub out enough methods to make the signed_cookie functions work
    def __init__(self, cookie_secret="0123456789", key_version=None):
        # don't call super.__init__
        self._cookies = {}  # type: typing.Dict[str, bytes]
        if key_version is None:
            self.application = ObjectDict(  # type: ignore
                settings=dict(cookie_secret=cookie_secret)
            )
        else:
            self.application = ObjectDict(  # type: ignore
                settings=dict(cookie_secret=cookie_secret, key_version=key_version)
            )

    def get_cookie(self, name):
        return self._cookies.get(name)

    def set_cookie(self, name, value, expires_days=None):
        self._cookies[name] = value


# See SignedValueTest below for more.
class SecureCookieV1Test(unittest.TestCase):
    def test_round_trip(self):
        handler = CookieTestRequestHandler()
        handler.set_signed_cookie("foo", b"bar", version=1)
        self.assertEqual(handler.get_signed_cookie("foo", min_version=1), b"bar")

    def test_cookie_tampering_future_timestamp(self):
        handler = CookieTestRequestHandler()
        # this string base64-encodes to '12345678'
        handler.set_signed_cookie("foo", binascii.a2b_hex(b"d76df8e7aefc"), version=1)
        cookie = handler._cookies["foo"]
        match = re.match(rb"12345678\|([0-9]+)\|([0-9a-f]+)", cookie)
        assert match is not None
        timestamp = match.group(1)
        sig = match.group(2)
        self.assertEqual(
            _create_signature_v1(
                handler.application.settings["cookie_secret"],
                "foo",
                "12345678",
                timestamp,
            ),
            sig,
        )
        # shifting digits from payload to timestamp doesn't alter signature
        # (this is not desirable behavior, just confirming that that's how it
        # works)
        self.assertEqual(
            _create_signature_v1(
                handler.application.settings["cookie_secret"],
                "foo",
                "1234",
                b"5678" + timestamp,
            ),
            sig,
        )
        # tamper with the cookie
        handler._cookies["foo"] = utf8(
            "1234|5678%s|%s" % (to_basestring(timestamp), to_basestring(sig))
        )
        # it gets rejected
        with ExpectLog(gen_log, "Cookie timestamp in future"):
            self.assertTrue(handler.get_signed_cookie("foo", min_version=1) is None)

    def test_arbitrary_bytes(self):
        # Secure cookies accept arbitrary data (which is base64 encoded).
        # Note that normal cookies accept only a subset of ascii.
        handler = CookieTestRequestHandler()
        handler.set_signed_cookie("foo", b"\xe9", version=1)
        self.assertEqual(handler.get_signed_cookie("foo", min_version=1), b"\xe9")


# See SignedValueTest below for more.
class SecureCookieV2Test(unittest.TestCase):
    KEY_VERSIONS = {0: "ajklasdf0ojaisdf", 1: "aslkjasaolwkjsdf"}

    def test_round_trip(self):
        handler = CookieTestRequestHandler()
        handler.set_signed_cookie("foo", b"bar", version=2)
        self.assertEqual(handler.get_signed_cookie("foo", min_version=2), b"bar")

    def test_key_version_roundtrip(self):
        handler = CookieTestRequestHandler(
            cookie_secret=self.KEY_VERSIONS, key_version=0
        )
        handler.set_signed_cookie("foo", b"bar")
        self.assertEqual(handler.get_signed_cookie("foo"), b"bar")

    def test_key_version_roundtrip_differing_version(self):
        handler = CookieTestRequestHandler(
            cookie_secret=self.KEY_VERSIONS, key_version=1
        )
        handler.set_signed_cookie("foo", b"bar")
        self.assertEqual(handler.get_signed_cookie("foo"), b"bar")

    def test_key_version_increment_version(self):
        handler = CookieTestRequestHandler(
            cookie_secret=self.KEY_VERSIONS, key_version=0
        )
        handler.set_signed_cookie("foo", b"bar")
        new_handler = CookieTestRequestHandler(
            cookie_secret=self.KEY_VERSIONS, key_version=1
        )
        new_handler._cookies = handler._cookies
        self.assertEqual(new_handler.get_signed_cookie("foo"), b"bar")

    def test_key_version_invalidate_version(self):
        handler = CookieTestRequestHandler(
            cookie_secret=self.KEY_VERSIONS, key_version=0
        )
        handler.set_signed_cookie("foo", b"bar")
        new_key_versions = self.KEY_VERSIONS.copy()
        new_key_versions.pop(0)
        new_handler = CookieTestRequestHandler(
            cookie_secret=new_key_versions, key_version=1
        )
        new_handler._cookies = handler._cookies
        self.assertEqual(new_handler.get_signed_cookie("foo"), None)


class FinalReturnTest(WebTestCase):
    final_return = None  # type: Future

    def get_handlers(self):
        test = self

        class FinishHandler(RequestHandler):
            @gen.coroutine
            def get(self):
                test.final_return = self.finish()
                yield test.final_return

            @gen.coroutine
            def post(self):
                self.write("hello,")
                yield self.flush()
                test.final_return = self.finish("world")
                yield test.final_return

        class RenderHandler(RequestHandler):
            def create_template_loader(self, path):
                return DictLoader({"foo.html": "hi"})

            @gen.coroutine
            def get(self):
                test.final_return = self.render("foo.html")

        return [("/finish", FinishHandler), ("/render", RenderHandler)]

    def get_app_kwargs(self):
        return dict(template_path="FinalReturnTest")

    def test_finish_method_return_future(self):
        response = self.fetch(self.get_url("/finish"))
        self.assertEqual(response.code, 200)
        self.assertIsInstance(self.final_return, Future)
        self.assertTrue(self.final_return.done())

        response = self.fetch(self.get_url("/finish"), method="POST", body=b"")
        self.assertEqual(response.code, 200)
        self.assertIsInstance(self.final_return, Future)
        self.assertTrue(self.final_return.done())

    def test_render_method_return_future(self):
        response = self.fetch(self.get_url("/render"))
        self.assertEqual(response.code, 200)
        self.assertIsInstance(self.final_return, Future)


class CookieTest(WebTestCase):
    def get_handlers(self):
        class SetCookieHandler(RequestHandler):
            def get(self):
                # Try setting cookies with different argument types
                # to ensure that everything gets encoded correctly
                self.set_cookie("str", "asdf")
                self.set_cookie("unicode", "qwer")
                self.set_cookie("bytes", b"zxcv")

        class GetCookieHandler(RequestHandler):
            def get(self):
                cookie = self.get_cookie("foo", "default")
                assert cookie is not None
                self.write(cookie)

        class SetCookieDomainHandler(RequestHandler):
            def get(self):
                # unicode domain and path arguments shouldn't break things
                # either (see bug #285)
                self.set_cookie("unicode_args", "blah", domain="foo.com", path="/foo")

        class SetCookieSpecialCharHandler(RequestHandler):
            def get(self):
                self.set_cookie("equals", "a=b")
                self.set_cookie("semicolon", "a;b")
                self.set_cookie("quote", 'a"b')

        class SetCookieOverwriteHandler(RequestHandler):
            def get(self):
                self.set_cookie("a", "b", domain="example.com")
                self.set_cookie("c", "d", domain="example.com")
                # A second call with the same name clobbers the first.
                # Attributes from the first call are not carried over.
                self.set_cookie("a", "e")

        class SetCookieMaxAgeHandler(RequestHandler):
            def get(self):
                self.set_cookie("foo", "bar", max_age=10)

        class SetCookieExpiresDaysHandler(RequestHandler):
            def get(self):
                self.set_cookie("foo", "bar", expires_days=10)

        class SetCookieFalsyFlags(RequestHandler):
            def get(self):
                self.set_cookie("a", "1", secure=True)
                self.set_cookie("b", "1", secure=False)
                self.set_cookie("c", "1", httponly=True)
                self.set_cookie("d", "1", httponly=False)

        class SetCookieDeprecatedArgs(RequestHandler):
            def get(self):
                # Mixed case is supported, but deprecated
                self.set_cookie("a", "b", HttpOnly=True, pATH="/foo")

        return [
            ("/set", SetCookieHandler),
            ("/get", GetCookieHandler),
            ("/set_domain", SetCookieDomainHandler),
            ("/special_char", SetCookieSpecialCharHandler),
            ("/set_overwrite", SetCookieOverwriteHandler),
            ("/set_max_age", SetCookieMaxAgeHandler),
            ("/set_expires_days", SetCookieExpiresDaysHandler),
            ("/set_falsy_flags", SetCookieFalsyFlags),
            ("/set_deprecated", SetCookieDeprecatedArgs),
        ]

    def test_set_cookie(self):
        response = self.fetch("/set")
        self.assertEqual(
            sorted(response.headers.get_list("Set-Cookie")),
            ["bytes=zxcv; Path=/", "str=asdf; Path=/", "unicode=qwer; Path=/"],
        )

    def test_get_cookie(self):
        response = self.fetch("/get", headers={"Cookie": "foo=bar"})
        self.assertEqual(response.body, b"bar")

        response = self.fetch("/get", headers={"Cookie": 'foo="bar"'})
        self.assertEqual(response.body, b"bar")

        response = self.fetch("/get", headers={"Cookie": "/=exception;"})
        self.assertEqual(response.body, b"default")

    def test_set_cookie_domain(self):
        response = self.fetch("/set_domain")
        self.assertEqual(
            response.headers.get_list("Set-Cookie"),
            ["unicode_args=blah; Domain=foo.com; Path=/foo"],
        )

    def test_cookie_special_char(self):
        response = self.fetch("/special_char")
        headers = sorted(response.headers.get_list("Set-Cookie"))
        self.assertEqual(len(headers), 3)
        self.assertEqual(headers[0], 'equals="a=b"; Path=/')
        self.assertEqual(headers[1], 'quote="a\\"b"; Path=/')
        # python 2.7 octal-escapes the semicolon; older versions leave it alone
        self.assertTrue(
            headers[2] in ('semicolon="a;b"; Path=/', 'semicolon="a\\073b"; Path=/'),
            headers[2],
        )

        data = [
            ("foo=a=b", "a=b"),
            ('foo="a=b"', "a=b"),
            ('foo="a;b"', '"a'),  # even quoted, ";" is a delimiter
            ("foo=a\\073b", "a\\073b"),  # escapes only decoded in quotes
            ('foo="a\\073b"', "a;b"),
            ('foo="a\\"b"', 'a"b'),
        ]
        for header, expected in data:
            logging.debug("trying %r", header)
            response = self.fetch("/get", headers={"Cookie": header})
            self.assertEqual(response.body, utf8(expected))

    def test_set_cookie_overwrite(self):
        response = self.fetch("/set_overwrite")
        headers = response.headers.get_list("Set-Cookie")
        self.assertEqual(
            sorted(headers), ["a=e; Path=/", "c=d; Domain=example.com; Path=/"]
        )

    def test_set_cookie_max_age(self):
        response = self.fetch("/set_max_age")
        headers = response.headers.get_list("Set-Cookie")
        self.assertEqual(sorted(headers), ["foo=bar; Max-Age=10; Path=/"])

    def test_set_cookie_expires_days(self):
        response = self.fetch("/set_expires_days")
        header = response.headers.get("Set-Cookie")
        assert header is not None
        match = re.match("foo=bar; expires=(?P<expires>.+); Path=/", header)
        assert match is not None

        expires = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
            days=10
        )
        header_expires = email.utils.parsedate_to_datetime(match.groupdict()["expires"])
        self.assertTrue(abs((expires - header_expires).total_seconds()) < 10)

    def test_set_cookie_false_flags(self):
        response = self.fetch("/set_falsy_flags")
        headers = sorted(response.headers.get_list("Set-Cookie"))
        # The secure and httponly headers are capitalized in py35 and
        # lowercase in older versions.
        self.assertEqual(headers[0].lower(), "a=1; path=/; secure")
        self.assertEqual(headers[1].lower(), "b=1; path=/")
        self.assertEqual(headers[2].lower(), "c=1; httponly; path=/")
        self.assertEqual(headers[3].lower(), "d=1; path=/")

    def test_set_cookie_deprecated(self):
        with ignore_deprecation():
            response = self.fetch("/set_deprecated")
        header = response.headers.get("Set-Cookie")
        self.assertEqual(header, "a=b; HttpOnly; Path=/foo")


class AuthRedirectRequestHandler(RequestHandler):
    def initialize(self, login_url):
        self.login_url = login_url

    def get_login_url(self):
        return self.login_url

    @authenticated
    def get(self):
        # we'll never actually get here because the test doesn't follow redirects
        self.send_error(500)


class AuthRedirectTest(WebTestCase):
    def get_handlers(self):
        return [
            ("/relative", AuthRedirectRequestHandler, dict(login_url="/login")),
            (
                "/absolute",
                AuthRedirectRequestHandler,
                dict(login_url="http://example.com/login"),
            ),
        ]

    def test_relative_auth_redirect(self):
        response = self.fetch(self.get_url("/relative"), follow_redirects=False)
        self.assertEqual(response.code, 302)
        self.assertEqual(response.headers["Location"], "/login?next=%2Frelative")

    def test_absolute_auth_redirect(self):
        response = self.fetch(self.get_url("/absolute"), follow_redirects=False)
        self.assertEqual(response.code, 302)
        self.assertTrue(
            re.match(
                r"http://example.com/login\?next=http%3A%2F%2F127.0.0.1%3A[0-9]+%2Fabsolute",
                response.headers["Location"],
            ),
            response.headers["Location"],
        )


class ConnectionCloseHandler(RequestHandler):
    def initialize(self, test):
        self.test = test

    @gen.coroutine
    def get(self):
        self.test.on_handler_waiting()
        yield self.test.cleanup_event.wait()

    def on_connection_close(self):
        self.test.on_connection_close()


class ConnectionCloseTest(WebTestCase):
    def get_handlers(self):
        self.cleanup_event = Event()
        return [("/", ConnectionCloseHandler, dict(test=self))]

    def test_connection_close(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
        s.connect(("127.0.0.1", self.get_http_port()))
        self.stream = IOStream(s)
        self.stream.write(b"GET / HTTP/1.0\r\n\r\n")
        self.wait()
        # Let the hanging coroutine clean up after itself
        self.cleanup_event.set()
        self.io_loop.run_sync(lambda: gen.sleep(0))

    def on_handler_waiting(self):
        logging.debug("handler waiting")
        self.stream.close()

    def on_connection_close(self):
        logging.debug("connection closed")
        self.stop()


class EchoHandler(RequestHandler):
    def get(self, *path_args):
        # Type checks: web.py interfaces convert argument values to
        # unicode strings (by default, but see also decode_argument).
        # In httpserver.py (i.e. self.request.arguments), they're left
        # as bytes.  Keys are always native strings.
        for key in self.request.arguments:
            if type(key) != str:
                raise Exception("incorrect type for key: %r" % type(key))
            for bvalue in self.request.arguments[key]:
                if type(bvalue) != bytes:
                    raise Exception("incorrect type for value: %r" % type(bvalue))
            for svalue in self.get_arguments(key):
                if type(svalue) != unicode_type:
                    raise Exception("incorrect type for value: %r" % type(svalue))
        for arg in path_args:
            if type(arg) != unicode_type:
                raise Exception("incorrect type for path arg: %r" % type(arg))
        self.write(
            dict(
                path=self.request.path,
                path_args=path_args,
                args=recursive_unicode(self.request.arguments),
            )
        )


class RequestEncodingTest(WebTestCase):
    def get_handlers(self):
        return [("/group/(.*)", EchoHandler), ("/slashes/([^/]*)/([^/]*)", EchoHandler)]

    def fetch_json(self, path):
        return json_decode(self.fetch(path).body)

    def test_group_question_mark(self):
        # Ensure that url-encoded question marks are handled properly
        self.assertEqual(
            self.fetch_json("/group/%3F"),
            dict(path="/group/%3F", path_args=["?"], args={}),
        )
        self.assertEqual(
            self.fetch_json("/group/%3F?%3F=%3F"),
            dict(path="/group/%3F", path_args=["?"], args={"?": ["?"]}),
        )

    def test_group_encoding(self):
        # Path components and query arguments should be decoded the same way
        self.assertEqual(
            self.fetch_json("/group/%C3%A9?arg=%C3%A9"),
            {
                "path": "/group/%C3%A9",
                "path_args": ["\u00e9"],
                "args": {"arg": ["\u00e9"]},
            },
        )

    def test_slashes(self):
        # Slashes may be escaped to appear as a single "directory" in the path,
        # but they are then unescaped when passed to the get() method.
        self.assertEqual(
            self.fetch_json("/slashes/foo/bar"),
            dict(path="/slashes/foo/bar", path_args=["foo", "bar"], args={}),
        )
        self.assertEqual(
            self.fetch_json("/slashes/a%2Fb/c%2Fd"),
            dict(path="/slashes/a%2Fb/c%2Fd", path_args=["a/b", "c/d"], args={}),
        )

    def test_error(self):
        # Percent signs (encoded as %25) should not mess up printf-style
        # messages in logs
        with ExpectLog(gen_log, ".*Invalid unicode"):
            self.fetch("/group/?arg=%25%e9")


class TypeCheckHandler(RequestHandler):
    def prepare(self):
        self.errors = {}  # type: typing.Dict[str, str]

        self.check_type("status", self.get_status(), int)

        # get_argument is an exception from the general rule of using
        # type str for non-body data mainly for historical reasons.
        self.check_type("argument", self.get_argument("foo"), unicode_type)
        self.check_type("cookie_key", list(self.cookies.keys())[0], str)
        self.check_type("cookie_value", list(self.cookies.values())[0].value, str)

        # Secure cookies return bytes because they can contain arbitrary
        # data, but regular cookies are native strings.
        if list(self.cookies.keys()) != ["asdf"]:
            raise Exception(
                "unexpected values for cookie keys: %r" % self.cookies.keys()
            )
        self.check_type("get_signed_cookie", self.get_signed_cookie("asdf"), bytes)
        self.check_type("get_cookie", self.get_cookie("asdf"), str)

        self.check_type("xsrf_token", self.xsrf_token, bytes)
        self.check_type("xsrf_form_html", self.xsrf_form_html(), str)

        self.check_type("reverse_url", self.reverse_url("typecheck", "foo"), str)

        self.check_type("request_summary", self._request_summary(), str)

    def get(self, path_component):
        # path_component uses type unicode instead of str for consistency
        # with get_argument()
        self.check_type("path_component", path_component, unicode_type)
        self.write(self.errors)

    def post(self, path_component):
        self.check_type("path_component", path_component, unicode_type)
        self.write(self.errors)

    def check_type(self, name, obj, expected_type):
        actual_type = type(obj)
        if expected_type != actual_type:
            self.errors[name] = "expected %s, got %s" % (expected_type, actual_type)


class DecodeArgHandler(RequestHandler):
    def decode_argument(self, value, name=None):
        if type(value) != bytes:
            raise Exception("unexpected type for value: %r" % type(value))
        # use self.request.arguments directly to avoid recursion
        if "encoding" in self.request.arguments:
            return value.decode(to_unicode(self.request.arguments["encoding"][0]))
        else:
            return value

    def get(self, arg):
        def describe(s):
            if type(s) == bytes:
                return ["bytes", native_str(binascii.b2a_hex(s))]
            elif type(s) == unicode_type:
                return ["unicode", s]
            raise Exception("unknown type")

        self.write({"path": describe(arg), "query": describe(self.get_argument("foo"))})


class LinkifyHandler(RequestHandler):
    def get(self):
        self.render("linkify.html", message="http://example.com")


class UIModuleResourceHandler(RequestHandler):
    def get(self):
        self.render("page.html", entries=[1, 2])


class OptionalPathHandler(RequestHandler):
    def get(self, path):
        self.write({"path": path})


class MultiHeaderHandler(RequestHandler):
    def get(self):
        self.set_header("x-overwrite", "1")
        self.set_header("X-Overwrite", 2)
        self.add_header("x-multi", 3)
        self.add_header("X-Multi", "4")


class RedirectHandler(RequestHandler):
    def get(self):
        if self.get_argument("permanent", None) is not None:
            self.redirect("/", permanent=bool(int(self.get_argument("permanent"))))
        elif self.get_argument("status", None) is not None:
            self.redirect("/", status=int(self.get_argument("status")))
        else:
            raise Exception("didn't get permanent or status arguments")


class EmptyFlushCallbackHandler(RequestHandler):
    @gen.coroutine
    def get(self):
        # Ensure that the flush callback is run whether or not there
        # was any output.  The gen.Task and direct yield forms are
        # equivalent.
        yield self.flush()  # "empty" flush, but writes headers
        yield self.flush()  # empty flush
        self.write("o")
        yield self.flush()  # flushes the "o"
        yield self.flush()  # empty flush
        self.finish("k")


class HeaderInjectionHandler(RequestHandler):
    def get(self):
        try:
            self.set_header("X-Foo", "foo\r\nX-Bar: baz")
            raise Exception("Didn't get expected exception")
        except ValueError as e:
            if "Unsafe header value" in str(e):
                self.finish(b"ok")
            else:
                raise


class GetArgumentHandler(RequestHandler):
    def prepare(self):
        if self.get_argument("source", None) == "query":
            method = self.get_query_argument
        elif self.get_argument("source", None) == "body":
            method = self.get_body_argument
        else:
            method = self.get_argument  # type: ignore
        self.finish(method("foo", "default"))


class GetArgumentsHandler(RequestHandler):
    def prepare(self):
        self.finish(
            dict(
                default=self.get_arguments("foo"),
                query=self.get_query_arguments("foo"),
                body=self.get_body_arguments("foo"),
            )
        )


# This test was shared with wsgi_test.py; now the name is meaningless.
class WSGISafeWebTest(WebTestCase):
    COOKIE_SECRET = "WebTest.COOKIE_SECRET"

    def get_app_kwargs(self):
        loader = DictLoader(
            {
                "linkify.html": "{% module linkify(message) %}",
                "page.html": """\
<html><head></head><body>
{% for e in entries %}
{% module Template("entry.html", entry=e) %}
{% end %}
</body></html>""",
                "entry.html": """\
{{ set_resources(embedded_css=".entry { margin-bottom: 1em; }",
                 embedded_javascript="js_embed()",
                 css_files=["/base.css", "/foo.css"],
                 javascript_files="/common.js",
                 html_head="<meta>",
                 html_body='<script src="/analytics.js"/>') }}
<div class="entry">...</div>""",
            }
        )
        return dict(
            template_loader=loader,
            autoescape="xhtml_escape",
            cookie_secret=self.COOKIE_SECRET,
        )

    def tearDown(self):
        super().tearDown()
        RequestHandler._template_loaders.clear()

    def get_handlers(self):
        urls = [
            url("/typecheck/(.*)", TypeCheckHandler, name="typecheck"),
            url("/decode_arg/(.*)", DecodeArgHandler, name="decode_arg"),
            url("/decode_arg_kw/(?P<arg>.*)", DecodeArgHandler),
            url("/linkify", LinkifyHandler),
            url("/uimodule_resources", UIModuleResourceHandler),
            url("/optional_path/(.+)?", OptionalPathHandler),
            url("/multi_header", MultiHeaderHandler),
            url("/redirect", RedirectHandler),
            url(
                "/web_redirect_permanent",
                WebRedirectHandler,
                {"url": "/web_redirect_newpath"},
            ),
            url(
                "/web_redirect",
                WebRedirectHandler,
                {"url": "/web_redirect_newpath", "permanent": False},
            ),
            url(
                "//web_redirect_double_slash",
                WebRedirectHandler,
                {"url": "/web_redirect_newpath"},
            ),
            url("/header_injection", HeaderInjectionHandler),
            url("/get_argument", GetArgumentHandler),
            url("/get_arguments", GetArgumentsHandler),
        ]
        return urls

    def fetch_json(self, *args, **kwargs):
        response = self.fetch(*args, **kwargs)
        response.rethrow()
        return json_decode(response.body)

    def test_types(self):
        cookie_value = to_unicode(
            create_signed_value(self.COOKIE_SECRET, "asdf", "qwer")
        )
        response = self.fetch(
            "/typecheck/asdf?foo=bar", headers={"Cookie": "asdf=" + cookie_value}
        )
        data = json_decode(response.body)
        self.assertEqual(data, {})

        response = self.fetch(
            "/typecheck/asdf?foo=bar",
            method="POST",
            headers={"Cookie": "asdf=" + cookie_value},
            body="foo=bar",
        )

    def test_decode_argument(self):
        # These urls all decode to the same thing
        urls = [
            "/decode_arg/%C3%A9?foo=%C3%A9&encoding=utf-8",
            "/decode_arg/%E9?foo=%E9&encoding=latin1",
            "/decode_arg_kw/%E9?foo=%E9&encoding=latin1",
        ]
        for req_url in urls:
            response = self.fetch(req_url)
            response.rethrow()
            data = json_decode(response.body)
            self.assertEqual(
                data,
                {"path": ["unicode", "\u00e9"], "query": ["unicode", "\u00e9"]},
            )

        response = self.fetch("/decode_arg/%C3%A9?foo=%C3%A9")
        response.rethrow()
        data = json_decode(response.body)
        self.assertEqual(data, {"path": ["bytes", "c3a9"], "query": ["bytes", "c3a9"]})

    def test_decode_argument_invalid_unicode(self):
        # test that invalid unicode in URLs causes 400, not 500
        with ExpectLog(gen_log, ".*Invalid unicode.*"):
            response = self.fetch("/typecheck/invalid%FF")
            self.assertEqual(response.code, 400)
            response = self.fetch("/typecheck/invalid?foo=%FF")
            self.assertEqual(response.code, 400)

    def test_decode_argument_plus(self):
        # These urls are all equivalent.
        urls = [
            "/decode_arg/1%20%2B%201?foo=1%20%2B%201&encoding=utf-8",
            "/decode_arg/1%20+%201?foo=1+%2B+1&encoding=utf-8",
        ]
        for req_url in urls:
            response = self.fetch(req_url)
            response.rethrow()
            data = json_decode(response.body)
            self.assertEqual(
                data,
                {"path": ["unicode", "1 + 1"], "query": ["unicode", "1 + 1"]},
            )

    def test_reverse_url(self):
        self.assertEqual(self.app.reverse_url("decode_arg", "foo"), "/decode_arg/foo")
        self.assertEqual(self.app.reverse_url("decode_arg", 42), "/decode_arg/42")
        self.assertEqual(self.app.reverse_url("decode_arg", b"\xe9"), "/decode_arg/%E9")
        self.assertEqual(
            self.app.reverse_url("decode_arg", "\u00e9"), "/decode_arg/%C3%A9"
        )
        self.assertEqual(
            self.app.reverse_url("decode_arg", "1 + 1"), "/decode_arg/1%20%2B%201"
        )

    def test_uimodule_unescaped(self):
        response = self.fetch("/linkify")
        self.assertEqual(
            response.body, b'<a href="http://example.com">http://example.com</a>'
        )

    def test_uimodule_resources(self):
        response = self.fetch("/uimodule_resources")
        self.assertEqual(
            response.body,
            b"""\
<html><head><link href="/base.css" type="text/css" rel="stylesheet"/><link href="/foo.css" type="text/css" rel="stylesheet"/>
<style type="text/css">
.entry { margin-bottom: 1em; }
</style>
<meta>
</head><body>


<div class="entry">...</div>


<div class="entry">...</div>

<script src="/common.js" type="text/javascript"></script>
<script type="text/javascript">
//<![CDATA[
js_embed()
//]]>
</script>
<script src="/analytics.js"/>
</body></html>""",  # noqa: E501
        )

    def test_optional_path(self):
        self.assertEqual(self.fetch_json("/optional_path/foo"), {"path": "foo"})
        self.assertEqual(self.fetch_json("/optional_path/"), {"path": None})

    def test_multi_header(self):
        response = self.fetch("/multi_header")
        self.assertEqual(response.headers["x-overwrite"], "2")
        self.assertEqual(response.headers.get_list("x-multi"), ["3", "4"])

    def test_redirect(self):
        response = self.fetch("/redirect?permanent=1", follow_redirects=False)
        self.assertEqual(response.code, 301)
        response = self.fetch("/redirect?permanent=0", follow_redirects=False)
        self.assertEqual(response.code, 302)
        response = self.fetch("/redirect?status=307", follow_redirects=False)
        self.assertEqual(response.code, 307)

    def test_web_redirect(self):
        response = self.fetch("/web_redirect_permanent", follow_redirects=False)
        self.assertEqual(response.code, 301)
        self.assertEqual(response.headers["Location"], "/web_redirect_newpath")
        response = self.fetch("/web_redirect", follow_redirects=False)
        self.assertEqual(response.code, 302)
        self.assertEqual(response.headers["Location"], "/web_redirect_newpath")

    def test_web_redirect_double_slash(self):
        response = self.fetch("//web_redirect_double_slash", follow_redirects=False)
        self.assertEqual(response.code, 301)
        self.assertEqual(response.headers["Location"], "/web_redirect_newpath")

    def test_header_injection(self):
        response = self.fetch("/header_injection")
        self.assertEqual(response.body, b"ok")

    def test_get_argument(self):
        response = self.fetch("/get_argument?foo=bar")
        self.assertEqual(response.body, b"bar")
        response = self.fetch("/get_argument?foo=")
        self.assertEqual(response.body, b"")
        response = self.fetch("/get_argument")
        self.assertEqual(response.body, b"default")

        # Test merging of query and body arguments.
        # In singular form, body arguments take precedence over query arguments.
        body = urllib.parse.urlencode(dict(foo="hello"))
        response = self.fetch("/get_argument?foo=bar", method="POST", body=body)
        self.assertEqual(response.body, b"hello")
        # In plural methods they are merged.
        response = self.fetch("/get_arguments?foo=bar", method="POST", body=body)
        self.assertEqual(
            json_decode(response.body),
            dict(default=["bar", "hello"], query=["bar"], body=["hello"]),
        )

    def test_get_query_arguments(self):
        # send as a post so we can ensure the separation between query
        # string and body arguments.
        body = urllib.parse.urlencode(dict(foo="hello"))
        response = self.fetch(
            "/get_argument?source=query&foo=bar", method="POST", body=body
        )
        self.assertEqual(response.body, b"bar")
        response = self.fetch(
            "/get_argument?source=query&foo=", method="POST", body=body
        )
        self.assertEqual(response.body, b"")
        response = self.fetch("/get_argument?source=query", method="POST", body=body)
        self.assertEqual(response.body, b"default")

    def test_get_body_arguments(self):
        body = urllib.parse.urlencode(dict(foo="bar"))
        response = self.fetch(
            "/get_argument?source=body&foo=hello", method="POST", body=body
        )
        self.assertEqual(response.body, b"bar")

        body = urllib.parse.urlencode(dict(foo=""))
        response = self.fetch(
            "/get_argument?source=body&foo=hello", method="POST", body=body
        )
        self.assertEqual(response.body, b"")

        body = urllib.parse.urlencode(dict())
        response = self.fetch(
            "/get_argument?source=body&foo=hello", method="POST", body=body
        )
        self.assertEqual(response.body, b"default")

    def test_no_gzip(self):
        response = self.fetch("/get_argument")
        self.assertNotIn("Accept-Encoding", response.headers.get("Vary", ""))
        self.assertNotIn("gzip", response.headers.get("Content-Encoding", ""))


class NonWSGIWebTests(WebTestCase):
    def get_handlers(self):
        return [("/empty_flush", EmptyFlushCallbackHandler)]

    def test_empty_flush(self):
        response = self.fetch("/empty_flush")
        self.assertEqual(response.body, b"ok")


class ErrorResponseTest(WebTestCase):
    def get_handlers(self):
        class DefaultHandler(RequestHandler):
            def get(self):
                if self.get_argument("status", None):
                    raise HTTPError(int(self.get_argument("status")))
                1 / 0

        class WriteErrorHandler(RequestHandler):
            def get(self):
                if self.get_argument("status", None):
                    self.send_error(int(self.get_argument("status")))
                else:
                    1 / 0

            def write_error(self, status_code, **kwargs):
                self.set_header("Content-Type", "text/plain")
                if "exc_info" in kwargs:
                    self.write("Exception: %s" % kwargs["exc_info"][0].__name__)
                else:
                    self.write("Status: %d" % status_code)

        class FailedWriteErrorHandler(RequestHandler):
            def get(self):
                1 / 0

            def write_error(self, status_code, **kwargs):
                raise Exception("exception in write_error")

        return [
            url("/default", DefaultHandler),
            url("/write_error", WriteErrorHandler),
            url("/failed_write_error", FailedWriteErrorHandler),
        ]

    def test_default(self):
        with ExpectLog(app_log, "Uncaught exception"):
            response = self.fetch("/default")
            self.assertEqual(response.code, 500)
            self.assertTrue(b"500: Internal Server Error" in response.body)

            response = self.fetch("/default?status=503")
            self.assertEqual(response.code, 503)
            self.assertTrue(b"503: Service Unavailable" in response.body)

            response = self.fetch("/default?status=435")
            self.assertEqual(response.code, 435)
            self.assertTrue(b"435: Unknown" in response.body)

    def test_write_error(self):
        with ExpectLog(app_log, "Uncaught exception"):
            response = self.fetch("/write_error")
            self.assertEqual(response.code, 500)
            self.assertEqual(b"Exception: ZeroDivisionError", response.body)

            response = self.fetch("/write_error?status=503")
            self.assertEqual(response.code, 503)
            self.assertEqual(b"Status: 503", response.body)

    def test_failed_write_error(self):
        with ExpectLog(app_log, "Uncaught exception"):
            response = self.fetch("/failed_write_error")
            self.assertEqual(response.code, 500)
            self.assertEqual(b"", response.body)


class StaticFileTest(WebTestCase):
    # The expected SHA-512 hash of robots.txt, used in tests that call
    # StaticFileHandler.get_version
    robots_txt_hash = (
        b"63a36e950e134b5217e33c763e88840c10a07d80e6057d92b9ac97508de7fb1f"
        b"a6f0e9b7531e169657165ea764e8963399cb6d921ffe6078425aaafe54c04563"
    )
    static_dir = os.path.join(os.path.dirname(__file__), "static")

    def get_handlers(self):
        class StaticUrlHandler(RequestHandler):
            def get(self, path):
                with_v = int(self.get_argument("include_version", "1"))
                self.write(self.static_url(path, include_version=with_v))

        class AbsoluteStaticUrlHandler(StaticUrlHandler):
            include_host = True

        class OverrideStaticUrlHandler(RequestHandler):
            def get(self, path):
                do_include = bool(self.get_argument("include_host"))
                self.include_host = not do_include

                regular_url = self.static_url(path)
                override_url = self.static_url(path, include_host=do_include)
                if override_url == regular_url:
                    return self.write(str(False))

                protocol = self.request.protocol + "://"
                protocol_length = len(protocol)
                check_regular = regular_url.find(protocol, 0, protocol_length)
                check_override = override_url.find(protocol, 0, protocol_length)

                if do_include:
                    result = check_override == 0 and check_regular == -1
                else:
                    result = check_override == -1 and check_regular == 0
                self.write(str(result))

        return [
            ("/static_url/(.*)", StaticUrlHandler),
            ("/abs_static_url/(.*)", AbsoluteStaticUrlHandler),
            ("/override_static_url/(.*)", OverrideStaticUrlHandler),
            ("/root_static/(.*)", StaticFileHandler, dict(path="/")),
        ]

    def get_app_kwargs(self):
        return dict(static_path=relpath("static"))

    def test_static_files(self):
        response = self.fetch("/robots.txt")
        self.assertTrue(b"Disallow: /" in response.body)

        response = self.fetch("/static/robots.txt")
        self.assertTrue(b"Disallow: /" in response.body)
        self.assertEqual(response.headers.get("Content-Type"), "text/plain")

    def test_static_files_cacheable(self):
        # Test that the version parameter triggers cache-control headers. This
        # test is pretty weak but it gives us coverage of the code path which
        # was important for detecting the deprecation of datetime.utcnow.
        response = self.fetch("/robots.txt?v=12345")
        self.assertTrue(b"Disallow: /" in response.body)
        self.assertIn("Cache-Control", response.headers)
        self.assertIn("Expires", response.headers)

    def test_static_compressed_files(self):
        response = self.fetch("/static/sample.xml.gz")
        self.assertEqual(response.headers.get("Content-Type"), "application/gzip")
        response = self.fetch("/static/sample.xml.bz2")
        self.assertEqual(
            response.headers.get("Content-Type"), "application/octet-stream"
        )
        # make sure the uncompressed file still has the correct type
        response = self.fetch("/static/sample.xml")
        self.assertTrue(
            response.headers.get("Content-Type") in set(("text/xml", "application/xml"))
        )

    def test_static_url(self):
        response = self.fetch("/static_url/robots.txt")
        self.assertEqual(response.body, b"/static/robots.txt?v=" + self.robots_txt_hash)

    def test_absolute_static_url(self):
        response = self.fetch("/abs_static_url/robots.txt")
        self.assertEqual(
            response.body,
            (utf8(self.get_url("/")) + b"static/robots.txt?v=" + self.robots_txt_hash),
        )

    def test_relative_version_exclusion(self):
        response = self.fetch("/static_url/robots.txt?include_version=0")
        self.assertEqual(response.body, b"/static/robots.txt")

    def test_absolute_version_exclusion(self):
        response = self.fetch("/abs_static_url/robots.txt?include_version=0")
        self.assertEqual(response.body, utf8(self.get_url("/") + "static/robots.txt"))

    def test_include_host_override(self):
        self._trigger_include_host_check(False)
        self._trigger_include_host_check(True)

    def _trigger_include_host_check(self, include_host):
        path = "/override_static_url/robots.txt?include_host=%s"
        response = self.fetch(path % int(include_host))
        self.assertEqual(response.body, utf8(str(True)))

    def get_and_head(self, *args, **kwargs):
        """Performs a GET and HEAD request and returns the GET response.

        Fails if any ``Content-*`` headers returned by the two requests
        differ.
        """
        head_response = self.fetch(*args, method="HEAD", **kwargs)
        get_response = self.fetch(*args, method="GET", **kwargs)
        content_headers = set()
        for h in itertools.chain(head_response.headers, get_response.headers):
            if h.startswith("Content-"):
                content_headers.add(h)
        for h in content_headers:
            self.assertEqual(
                head_response.headers.get(h),
                get_response.headers.get(h),
                "%s differs between GET (%s) and HEAD (%s)"
                % (h, head_response.headers.get(h), get_response.headers.get(h)),
            )
        return get_response

    def test_static_304_if_modified_since(self):
        response1 = self.get_and_head("/static/robots.txt")
        response2 = self.get_and_head(
            "/static/robots.txt",
            headers={"If-Modified-Since": response1.headers["Last-Modified"]},
        )
        self.assertEqual(response2.code, 304)
        self.assertTrue("Content-Length" not in response2.headers)

    def test_static_304_if_none_match(self):
        response1 = self.get_and_head("/static/robots.txt")
        response2 = self.get_and_head(
            "/static/robots.txt", headers={"If-None-Match": response1.headers["Etag"]}
        )
        self.assertEqual(response2.code, 304)

    def test_static_304_etag_modified_bug(self):
        response1 = self.get_and_head("/static/robots.txt")
        response2 = self.get_and_head(
            "/static/robots.txt",
            headers={
                "If-None-Match": '"MISMATCH"',
                "If-Modified-Since": response1.headers["Last-Modified"],
            },
        )
        self.assertEqual(response2.code, 200)

    def test_static_if_modified_since_pre_epoch(self):
        # On windows, the functions that work with time_t do not accept
        # negative values, and at least one client (processing.js) seems
        # to use if-modified-since 1/1/1960 as a cache-busting technique.
        response = self.get_and_head(
            "/static/robots.txt",
            headers={"If-Modified-Since": "Fri, 01 Jan 1960 00:00:00 GMT"},
        )
        self.assertEqual(response.code, 200)

    def test_static_if_modified_since_time_zone(self):
        # Instead of the value from Last-Modified, make requests with times
        # chosen just before and after the known modification time
        # of the file to ensure that the right time zone is being used
        # when parsing If-Modified-Since.
        stat = os.stat(relpath("static/robots.txt"))

        response = self.get_and_head(
            "/static/robots.txt",
            headers={"If-Modified-Since": format_timestamp(stat.st_mtime - 1)},
        )
        self.assertEqual(response.code, 200)
        response = self.get_and_head(
            "/static/robots.txt",
            headers={"If-Modified-Since": format_timestamp(stat.st_mtime + 1)},
        )
        self.assertEqual(response.code, 304)

    def test_static_etag(self):
        response = self.get_and_head("/static/robots.txt")
        self.assertEqual(
            utf8(response.headers.get("Etag")), b'"' + self.robots_txt_hash + b'"'
        )

    def test_static_with_range(self):
        response = self.get_and_head(
            "/static/robots.txt", headers={"Range": "bytes=0-9"}
        )
        self.assertEqual(response.code, 206)
        self.assertEqual(response.body, b"User-agent")
        self.assertEqual(
            utf8(response.headers.get("Etag")), b'"' + self.robots_txt_hash + b'"'
        )
        self.assertEqual(response.headers.get("Content-Length"), "10")
        self.assertEqual(response.headers.get("Content-Range"), "bytes 0-9/26")

    def test_static_with_range_full_file(self):
        response = self.get_and_head(
            "/static/robots.txt", headers={"Range": "bytes=0-"}
        )
        # Note: Chrome refuses to play audio if it gets an HTTP 206 in response
        # to ``Range: bytes=0-`` :(
        self.assertEqual(response.code, 200)
        robots_file_path = os.path.join(self.static_dir, "robots.txt")
        with open(robots_file_path, encoding="utf-8") as f:
            self.assertEqual(response.body, utf8(f.read()))
        self.assertEqual(response.headers.get("Content-Length"), "26")
        self.assertEqual(response.headers.get("Content-Range"), None)

    def test_static_with_range_full_past_end(self):
        response = self.get_and_head(
            "/static/robots.txt", headers={"Range": "bytes=0-10000000"}
        )
        self.assertEqual(response.code, 200)
        robots_file_path = os.path.join(self.static_dir, "robots.txt")
        with open(robots_file_path, encoding="utf-8") as f:
            self.assertEqual(response.body, utf8(f.read()))
        self.assertEqual(response.headers.get("Content-Length"), "26")
        self.assertEqual(response.headers.get("Content-Range"), None)

    def test_static_with_range_partial_past_end(self):
        response = self.get_and_head(
            "/static/robots.txt", headers={"Range": "bytes=1-10000000"}
        )
        self.assertEqual(response.code, 206)
        robots_file_path = os.path.join(self.static_dir, "robots.txt")
        with open(robots_file_path, encoding="utf-8") as f:
            self.assertEqual(response.body, utf8(f.read()[1:]))
        self.assertEqual(response.headers.get("Content-Length"), "25")
        self.assertEqual(response.headers.get("Content-Range"), "bytes 1-25/26")

    def test_static_with_range_end_edge(self):
        response = self.get_and_head(
            "/static/robots.txt", headers={"Range": "bytes=22-"}
        )
        self.assertEqual(response.body, b": /\n")
        self.assertEqual(response.headers.get("Content-Length"), "4")
        self.assertEqual(response.headers.get("Content-Range"), "bytes 22-25/26")

    def test_static_with_range_neg_end(self):
        response = self.get_and_head(
            "/static/robots.txt", headers={"Range": "bytes=-4"}
        )
        self.assertEqual(response.body, b": /\n")
        self.assertEqual(response.headers.get("Content-Length"), "4")
        self.assertEqual(response.headers.get("Content-Range"), "bytes 22-25/26")

    def test_static_with_range_neg_past_start(self):
        response = self.get_and_head(
            "/static/robots.txt", headers={"Range": "bytes=-1000000"}
        )
        self.assertEqual(response.code, 200)
        robots_file_path = os.path.join(self.static_dir, "robots.txt")
        with open(robots_file_path, encoding="utf-8") as f:
            self.assertEqual(response.body, utf8(f.read()))
        self.assertEqual(response.headers.get("Content-Length"), "26")
        self.assertEqual(response.headers.get("Content-Range"), None)

    def test_static_invalid_range(self):
        response = self.get_and_head("/static/robots.txt", headers={"Range": "asdf"})
        self.assertEqual(response.code, 200)

    def test_static_unsatisfiable_range_zero_suffix(self):
        response = self.get_and_head(
            "/static/robots.txt", headers={"Range": "bytes=-0"}
        )
        self.assertEqual(response.headers.get("Content-Range"), "bytes */26")
        self.assertEqual(response.code, 416)

    def test_static_unsatisfiable_range_invalid_start(self):
        response = self.get_and_head(
            "/static/robots.txt", headers={"Range": "bytes=26"}
        )
        self.assertEqual(response.code, 416)
        self.assertEqual(response.headers.get("Content-Range"), "bytes */26")

    def test_static_unsatisfiable_range_end_less_than_start(self):
        response = self.get_and_head(
            "/static/robots.txt", headers={"Range": "bytes=10-3"}
        )
        self.assertEqual(response.code, 416)
        self.assertEqual(response.headers.get("Content-Range"), "bytes */26")

    def test_static_head(self):
        response = self.fetch("/static/robots.txt", method="HEAD")
        self.assertEqual(response.code, 200)
        # No body was returned, but we did get the right content length.
        self.assertEqual(response.body, b"")
        self.assertEqual(response.headers["Content-Length"], "26")
        self.assertEqual(
            utf8(response.headers["Etag"]), b'"' + self.robots_txt_hash + b'"'
        )

    def test_static_head_range(self):
        response = self.fetch(
            "/static/robots.txt", method="HEAD", headers={"Range": "bytes=1-4"}
        )
        self.assertEqual(response.code, 206)
        self.assertEqual(response.body, b"")
        self.assertEqual(response.headers["Content-Length"], "4")
        self.assertEqual(
            utf8(response.headers["Etag"]), b'"' + self.robots_txt_hash + b'"'
        )

    def test_static_range_if_none_match(self):
        response = self.get_and_head(
            "/static/robots.txt",
            headers={
                "Range": "bytes=1-4",
                "If-None-Match": b'"' + self.robots_txt_hash + b'"',
            },
        )
        self.assertEqual(response.code, 304)
        self.assertEqual(response.body, b"")
        self.assertTrue("Content-Length" not in response.headers)
        self.assertEqual(
            utf8(response.headers["Etag"]), b'"' + self.robots_txt_hash + b'"'
        )

    def test_static_404(self):
        response = self.get_and_head("/static/blarg")
        self.assertEqual(response.code, 404)

    def test_path_traversal_protection(self):
        # curl_httpclient processes ".." on the client side, so we
        # must test this with simple_httpclient.
        self.http_client.close()
        self.http_client = SimpleAsyncHTTPClient()
        with ExpectLog(gen_log, ".*not in root static directory"):
            response = self.get_and_head("/static/../static_foo.txt")
        # Attempted path traversal should result in 403, not 200
        # (which means the check failed and the file was served)
        # or 404 (which means that the file didn't exist and
        # is probably a packaging error).
        self.assertEqual(response.code, 403)

    @unittest.skipIf(os.name != "posix", "non-posix OS")
    def test_root_static_path(self):
        # Sometimes people set the StaticFileHandler's path to '/'
        # to disable Tornado's path validation (in conjunction with
        # their own validation in get_absolute_path). Make sure
        # that the stricter validation in 4.2.1 doesn't break them.
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "static/robots.txt"
        )
        response = self.get_and_head("/root_static" + urllib.parse.quote(path))
        self.assertEqual(response.code, 200)


class StaticDefaultFilenameTest(WebTestCase):
    def get_app_kwargs(self):
        return dict(
            static_path=relpath("static"),
            static_handler_args=dict(default_filename="index.html"),
        )

    def get_handlers(self):
        return []

    def test_static_default_filename(self):
        response = self.fetch("/static/dir/", follow_redirects=False)
        self.assertEqual(response.code, 200)
        self.assertEqual(b"this is the index\n", response.body)

    def test_static_default_redirect(self):
        response = self.fetch("/static/dir", follow_redirects=False)
        self.assertEqual(response.code, 301)
        self.assertTrue(response.headers["Location"].endswith("/static/dir/"))


class StaticDefaultFilenameRootTest(WebTestCase):
    def get_app_kwargs(self):
        return dict(
            static_path=os.path.abspath(relpath("static")),
            static_handler_args=dict(default_filename="index.html"),
            static_url_prefix="/",
        )

    def get_handlers(self):
        return []

    def get_http_client(self):
        # simple_httpclient only: curl doesn't let you send a request starting
        # with two slashes.
        return SimpleAsyncHTTPClient()

    def test_no_open_redirect(self):
        # This test verifies that the open redirect that affected some configurations
        # prior to Tornado 6.3.2 is no longer possible. The vulnerability required
        # a static_url_prefix of "/" and a default_filename (any value) to be set.
        # The absolute* server-side path to the static directory must also be known.
        #
        # * Almost absolute: On windows, the drive letter is stripped from the path.
        test_dir = os.path.dirname(__file__)
        drive, tail = os.path.splitdrive(test_dir)
        if os.name == "posix":
            self.assertEqual(tail, test_dir)
        else:
            test_dir = tail
        with ExpectLog(gen_log, ".*cannot redirect path with two initial slashes"):
            response = self.fetch(
                f"//evil.com/../{test_dir}/static/dir",
                follow_redirects=False,
            )
        self.assertEqual(response.code, 403)


class StaticFileWithPathTest(WebTestCase):
    def get_app_kwargs(self):
        return dict(
            static_path=relpath("static"),
            static_handler_args=dict(default_filename="index.html"),
        )

    def get_handlers(self):
        return [("/foo/(.*)", StaticFileHandler, {"path": relpath("templates/")})]

    def test_serve(self):
        response = self.fetch("/foo/utf8.html")
        self.assertEqual(response.body, b"H\xc3\xa9llo\n")


class CustomStaticFileTest(WebTestCase):
    def get_handlers(self):
        class MyStaticFileHandler(StaticFileHandler):
            @classmethod
            def make_static_url(cls, settings, path):
                version_hash = cls.get_version(settings, path)
                extension_index = path.rindex(".")
                before_version = path[:extension_index]
                after_version = path[(extension_index + 1) :]
                return "/static/%s.%s.%s" % (
                    before_version,
                    version_hash,
                    after_version,
                )

            def parse_url_path(self, url_path):
                extension_index = url_path.rindex(".")
                version_index = url_path.rindex(".", 0, extension_index)
                return "%s%s" % (url_path[:version_index], url_path[extension_index:])

            @classmethod
            def get_absolute_path(cls, settings, path):
                return "CustomStaticFileTest:" + path

            def validate_absolute_path(self, root, absolute_path):
                return absolute_path

            @classmethod
            def get_content(self, path, start=None, end=None):
                assert start is None and end is None
                if path == "CustomStaticFileTest:foo.txt":
                    return b"bar"
                raise Exception("unexpected path %r" % path)

            def get_content_size(self):
                if self.absolute_path == "CustomStaticFileTest:foo.txt":
                    return 3
                raise Exception("unexpected path %r" % self.absolute_path)

            def get_modified_time(self):
                return None

            @classmethod
            def get_version(cls, settings, path):
                return "42"

        class StaticUrlHandler(RequestHandler):
            def get(self, path):
                self.write(self.static_url(path))

        self.static_handler_class = MyStaticFileHandler

        return [("/static_url/(.*)", StaticUrlHandler)]

    def get_app_kwargs(self):
        return dict(static_path="dummy", static_handler_class=self.static_handler_class)

    def test_serve(self):
        response = self.fetch("/static/foo.42.txt")
        self.assertEqual(response.body, b"bar")

    def test_static_url(self):
        with ExpectLog(gen_log, "Could not open static file", required=False):
            response = self.fetch("/static_url/foo.txt")
            self.assertEqual(response.body, b"/static/foo.42.txt")


class HostMatchingTest(WebTestCase):
    class Handler(RequestHandler):
        def initialize(self, reply):
            self.reply = reply

        def get(self):
            self.write(self.reply)

    def get_handlers(self):
        return [("/foo", HostMatchingTest.Handler, {"reply": "wildcard"})]

    def test_host_matching(self):
        self.app.add_handlers(
            "www.example.com", [("/foo", HostMatchingTest.Handler, {"reply": "[0]"})]
        )
        self.app.add_handlers(
            r"www\.example\.com", [("/bar", HostMatchingTest.Handler, {"reply": "[1]"})]
        )
        self.app.add_handlers(
            "www.example.com", [("/baz", HostMatchingTest.Handler, {"reply": "[2]"})]
        )
        self.app.add_handlers(
            "www.e.*e.com", [("/baz", HostMatchingTest.Handler, {"reply": "[3]"})]
        )

        response = self.fetch("/foo")
        self.assertEqual(response.body, b"wildcard")
        response = self.fetch("/bar")
        self.assertEqual(response.code, 404)
        response = self.fetch("/baz")
        self.assertEqual(response.code, 404)

        response = self.fetch("/foo", headers={"Host": "www.example.com"})
        self.assertEqual(response.body, b"[0]")
        response = self.fetch("/bar", headers={"Host": "www.example.com"})
        self.assertEqual(response.body, b"[1]")
        response = self.fetch("/baz", headers={"Host": "www.example.com"})
        self.assertEqual(response.body, b"[2]")
        response = self.fetch("/baz", headers={"Host": "www.exe.com"})
        self.assertEqual(response.body, b"[3]")


class DefaultHostMatchingTest(WebTestCase):
    def get_handlers(self):
        return []

    def get_app_kwargs(self):
        return {"default_host": "www.example.com"}

    def test_default_host_matching(self):
        self.app.add_handlers(
            "www.example.com", [("/foo", HostMatchingTest.Handler, {"reply": "[0]"})]
        )
        self.app.add_handlers(
            r"www\.example\.com", [("/bar", HostMatchingTest.Handler, {"reply": "[1]"})]
        )
        self.app.add_handlers(
            "www.test.com", [("/baz", HostMatchingTest.Handler, {"reply": "[2]"})]
        )

        response = self.fetch("/foo")
        self.assertEqual(response.body, b"[0]")
        response = self.fetch("/bar")
        self.assertEqual(response.body, b"[1]")
        response = self.fetch("/baz")
        self.assertEqual(response.code, 404)

        response = self.fetch("/foo", headers={"X-Real-Ip": "127.0.0.1"})
        self.assertEqual(response.code, 404)

        self.app.default_host = "www.test.com"

        response = self.fetch("/baz")
        self.assertEqual(response.body, b"[2]")


class NamedURLSpecGroupsTest(WebTestCase):
    def get_handlers(self):
        class EchoHandler(RequestHandler):
            def get(self, path):
                self.write(path)

        return [
            ("/str/(?P<path>.*)", EchoHandler),
            ("/unicode/(?P<path>.*)", EchoHandler),
        ]

    def test_named_urlspec_groups(self):
        response = self.fetch("/str/foo")
        self.assertEqual(response.body, b"foo")

        response = self.fetch("/unicode/bar")
        self.assertEqual(response.body, b"bar")


class ClearHeaderTest(SimpleHandlerTestCase):
    class Handler(RequestHandler):
        def get(self):
            self.set_header("h1", "foo")
            self.set_header("h2", "bar")
            self.clear_header("h1")
            self.clear_header("nonexistent")

    def test_clear_header(self):
        response = self.fetch("/")
        self.assertTrue("h1" not in response.headers)
        self.assertEqual(response.headers["h2"], "bar")


class Header204Test(SimpleHandlerTestCase):
    class Handler(RequestHandler):
        def get(self):
            self.set_status(204)
            self.finish()

    def test_204_headers(self):
        response = self.fetch("/")
        self.assertEqual(response.code, 204)
        self.assertNotIn("Content-Length", response.headers)
        self.assertNotIn("Transfer-Encoding", response.headers)


class Header304Test(SimpleHandlerTestCase):
    class Handler(RequestHandler):
        def get(self):
            self.set_header("Content-Language", "en_US")
            self.write("hello")

    def test_304_headers(self):
        response1 = self.fetch("/")
        self.assertEqual(response1.headers["Content-Length"], "5")
        self.assertEqual(response1.headers["Content-Language"], "en_US")

        response2 = self.fetch(
            "/", headers={"If-None-Match": response1.headers["Etag"]}
        )
        self.assertEqual(response2.code, 304)
        self.assertTrue("Content-Length" not in response2.headers)
        self.assertTrue("Content-Language" not in response2.headers)
        # Not an entity header, but should not be added to 304s by chunking
        self.assertTrue("Transfer-Encoding" not in response2.headers)


class StatusReasonTest(SimpleHandlerTestCase):
    class Handler(RequestHandler):
        def get(self):
            reason = self.request.arguments.get("reason", [])
            self.set_status(
                int(self.get_argument("code")),
                reason=to_unicode(reason[0]) if reason else None,
            )

    def get_http_client(self):
        # simple_httpclient only: curl doesn't expose the reason string
        return SimpleAsyncHTTPClient()

    def test_status(self):
        response = self.fetch("/?code=304")
        self.assertEqual(response.code, 304)
        self.assertEqual(response.reason, "Not Modified")
        response = self.fetch("/?code=304&reason=Foo")
        self.assertEqual(response.code, 304)
        self.assertEqual(response.reason, "Foo")
        response = self.fetch("/?code=682&reason=Bar")
        self.assertEqual(response.code, 682)
        self.assertEqual(response.reason, "Bar")
        response = self.fetch("/?code=682")
        self.assertEqual(response.code, 682)
        self.assertEqual(response.reason, "Unknown")


class DateHeaderTest(SimpleHandlerTestCase):
    class Handler(RequestHandler):
        def get(self):
            self.write("hello")

    def test_date_header(self):
        response = self.fetch("/")
        header_date = email.utils.parsedate_to_datetime(response.headers["Date"])
        self.assertTrue(
            header_date - datetime.datetime.now(datetime.timezone.utc)
            < datetime.timedelta(seconds=2)
        )


class RaiseWithReasonTest(SimpleHandlerTestCase):
    class Handler(RequestHandler):
        def get(self):
            raise HTTPError(682, reason="Foo")

    def get_http_client(self):
        # simple_httpclient only: curl doesn't expose the reason string
        return SimpleAsyncHTTPClient()

    def test_raise_with_reason(self):
        response = self.fetch("/")
        self.assertEqual(response.code, 682)
        self.assertEqual(response.reason, "Foo")
        self.assertIn(b"682: Foo", response.body)

    def test_httperror_str(self):
        self.assertEqual(str(HTTPError(682, reason="Foo")), "HTTP 682: Foo")

    def test_httperror_str_from_httputil(self):
        self.assertEqual(str(HTTPError(682)), "HTTP 682: Unknown")


class ErrorHandlerXSRFTest(WebTestCase):
    def get_handlers(self):
        # note that if the handlers list is empty we get the default_host
        # redirect fallback instead of a 404, so test with both an
        # explicitly defined error handler and an implicit 404.
        return [("/error", ErrorHandler, dict(status_code=417))]

    def get_app_kwargs(self):
        return dict(xsrf_cookies=True)

    def test_error_xsrf(self):
        response = self.fetch("/error", method="POST", body="")
        self.assertEqual(response.code, 417)

    def test_404_xsrf(self):
        response = self.fetch("/404", method="POST", body="")
        self.assertEqual(response.code, 404)


class GzipTestCase(SimpleHandlerTestCase):
    class Handler(RequestHandler):
        def get(self):
            for v in self.get_arguments("vary"):
                self.add_header("Vary", v)
            # Must write at least MIN_LENGTH bytes to activate compression.
            self.write("hello world" + ("!" * GZipContentEncoding.MIN_LENGTH))

    def get_app_kwargs(self):
        return dict(
            gzip=True, static_path=os.path.join(os.path.dirname(__file__), "static")
        )

    def assert_compressed(self, response):
        # simple_httpclient renames the content-encoding header;
        # curl_httpclient doesn't.
        self.assertEqual(
            response.headers.get(
                "Content-Encoding", response.headers.get("X-Consumed-Content-Encoding")
            ),
            "gzip",
        )

    def test_gzip(self):
        response = self.fetch("/")
        self.assert_compressed(response)
        self.assertEqual(response.headers["Vary"], "Accept-Encoding")

    def test_gzip_static(self):
        # The streaming responses in StaticFileHandler have subtle
        # interactions with the gzip output so test this case separately.
        response = self.fetch("/robots.txt")
        self.assert_compressed(response)
        self.assertEqual(response.headers["Vary"], "Accept-Encoding")

    def test_gzip_not_requested(self):
        response = self.fetch("/", use_gzip=False)
        self.assertNotIn("Content-Encoding", response.headers)
        self.assertEqual(response.headers["Vary"], "Accept-Encoding")

    def test_vary_already_present(self):
        response = self.fetch("/?vary=Accept-Language")
        self.assert_compressed(response)
        self.assertEqual(
            [s.strip() for s in response.headers["Vary"].split(",")],
            ["Accept-Language", "Accept-Encoding"],
        )

    def test_vary_already_present_multiple(self):
        # Regression test for https://github.com/tornadoweb/tornado/issues/1670
        response = self.fetch("/?vary=Accept-Language&vary=Cookie")
        self.assert_compressed(response)
        self.assertEqual(
            [s.strip() for s in response.headers["Vary"].split(",")],
            ["Accept-Language", "Cookie", "Accept-Encoding"],
        )


class PathArgsInPrepareTest(WebTestCase):
    class Handler(RequestHandler):
        def prepare(self):
            self.write(dict(args=self.path_args, kwargs=self.path_kwargs))

        def get(self, path):
            assert path == "foo"
            self.finish()

    def get_handlers(self):
        return [("/pos/(.*)", self.Handler), ("/kw/(?P<path>.*)", self.Handler)]

    def test_pos(self):
        response = self.fetch("/pos/foo")
        response.rethrow()
        data = json_decode(response.body)
        self.assertEqual(data, {"args": ["foo"], "kwargs": {}})

    def test_kw(self):
        response = self.fetch("/kw/foo")
        response.rethrow()
        data = json_decode(response.body)
        self.assertEqual(data, {"args": [], "kwargs": {"path": "foo"}})


class ClearAllCookiesTest(SimpleHandlerTestCase):
    class Handler(RequestHandler):
        def get(self):
            self.clear_all_cookies()
            self.write("ok")

    def test_clear_all_cookies(self):
        response = self.fetch("/", headers={"Cookie": "foo=bar; baz=xyzzy"})
        set_cookies = sorted(response.headers.get_list("Set-Cookie"))
        # Python 3.5 sends 'baz="";'; older versions use 'baz=;'
        self.assertTrue(
            set_cookies[0].startswith("baz=;") or set_cookies[0].startswith('baz="";')
        )
        self.assertTrue(
            set_cookies[1].startswith("foo=;") or set_cookies[1].startswith('foo="";')
        )


class PermissionError(Exception):
    pass


class ExceptionHandlerTest(SimpleHandlerTestCase):
    class Handler(RequestHandler):
        def get(self):
            exc = self.get_argument("exc")
            if exc == "http":
                raise HTTPError(410, "no longer here")
            elif exc == "zero":
                1 / 0
            elif exc == "permission":
                raise PermissionError("not allowed")

        def write_error(self, status_code, **kwargs):
            if "exc_info" in kwargs:
                typ, value, tb = kwargs["exc_info"]
                if isinstance(value, PermissionError):
                    self.set_status(403)
                    self.write("PermissionError")
                    return
            RequestHandler.write_error(self, status_code, **kwargs)

        def log_exception(self, typ, value, tb):
            if isinstance(value, PermissionError):
                app_log.warning("custom logging for PermissionError: %s", value.args[0])
            else:
                RequestHandler.log_exception(self, typ, value, tb)

    def test_http_error(self):
        # HTTPErrors are logged as warnings with no stack trace.
        # TODO: extend ExpectLog to test this more precisely
        with ExpectLog(gen_log, ".*no longer here"):
            response = self.fetch("/?exc=http")
            self.assertEqual(response.code, 410)

    def test_unknown_error(self):
        # Unknown errors are logged as errors with a stack trace.
        with ExpectLog(app_log, "Uncaught exception"):
            response = self.fetch("/?exc=zero")
            self.assertEqual(response.code, 500)

    def test_known_error(self):
        # log_exception can override logging behavior, and write_error
        # can override the response.
        with ExpectLog(app_log, "custom logging for PermissionError: not allowed"):
            response = self.fetch("/?exc=permission")
            self.assertEqual(response.code, 403)


class BuggyLoggingTest(SimpleHandlerTestCase):
    class Handler(RequestHandler):
        def get(self):
            1 / 0

        def log_exception(self, typ, value, tb):
            1 / 0

    def test_buggy_log_exception(self):
        # Something gets logged even though the application's
        # logger is broken.
        with ExpectLog(app_log, ".*"):
            self.fetch("/")


class UIMethodUIModuleTest(SimpleHandlerTestCase):
    """Test that UI methods and modules are created correctly and
    associated with the handler.
    """

    class Handler(RequestHandler):
        def get(self):
            self.render("foo.html")

        def value(self):
            return self.get_argument("value")

    def get_app_kwargs(self):
        def my_ui_method(handler, x):
            return "In my_ui_method(%s) with handler value %s." % (x, handler.value())

        class MyModule(UIModule):
            def render(self, x):
                return "In MyModule(%s) with handler value %s." % (
                    x,
                    typing.cast(UIMethodUIModuleTest.Handler, self.handler).value(),
                )

        loader = DictLoader(
            {"foo.html": "{{ my_ui_method(42) }} {% module MyModule(123) %}"}
        )
        return dict(
            template_loader=loader,
            ui_methods={"my_ui_method": my_ui_method},
            ui_modules={"MyModule": MyModule},
        )

    def tearDown(self):
        super().tearDown()
        # TODO: fix template loader caching so this isn't necessary.
        RequestHandler._template_loaders.clear()

    def test_ui_method(self):
        response = self.fetch("/?value=asdf")
        self.assertEqual(
            response.body,
            b"In my_ui_method(42) with handler value asdf. "
            b"In MyModule(123) with handler value asdf.",
        )


class GetArgumentErrorTest(SimpleHandlerTestCase):
    class Handler(RequestHandler):
        def get(self):
            try:
                self.get_argument("foo")
                self.write({})
            except MissingArgumentError as e:
                self.write({"arg_name": e.arg_name, "log_message": e.log_message})

    def test_catch_error(self):
        response = self.fetch("/")
        self.assertEqual(
            json_decode(response.body),
            {"arg_name": "foo", "log_message": "Missing argument foo"},
        )


class SetLazyPropertiesTest(SimpleHandlerTestCase):
    class Handler(RequestHandler):
        def prepare(self):
            self.current_user = "Ben"
            self.locale = locale.get("en_US")

        def get_user_locale(self):
            raise NotImplementedError()

        def get_current_user(self):
            raise NotImplementedError()

        def get(self):
            self.write("Hello %s (%s)" % (self.current_user, self.locale.code))

    def test_set_properties(self):
        # Ensure that current_user can be assigned to normally for apps
        # that want to forgo the lazy get_current_user property
        response = self.fetch("/")
        self.assertEqual(response.body, b"Hello Ben (en_US)")


class GetCurrentUserTest(WebTestCase):
    def get_app_kwargs(self):
        class WithoutUserModule(UIModule):
            def render(self):
                return ""

        class WithUserModule(UIModule):
            def render(self):
                return str(self.current_user)

        loader = DictLoader(
            {
                "without_user.html": "",
                "with_user.html": "{{ current_user }}",
                "without_user_module.html": "{% module WithoutUserModule() %}",
                "with_user_module.html": "{% module WithUserModule() %}",
            }
        )
        return dict(
            template_loader=loader,
            ui_modules={
                "WithUserModule": WithUserModule,
                "WithoutUserModule": WithoutUserModule,
            },
        )

    def tearDown(self):
        super().tearDown()
        RequestHandler._template_loaders.clear()

    def get_handlers(self):
        class CurrentUserHandler(RequestHandler):
            def prepare(self):
                self.has_loaded_current_user = False

            def get_current_user(self):
                self.has_loaded_current_user = True
                return ""

        class WithoutUserHandler(CurrentUserHandler):
            def get(self):
                self.render_string("without_user.html")
                self.finish(str(self.has_loaded_current_user))

        class WithUserHandler(CurrentUserHandler):
            def get(self):
                self.render_string("with_user.html")
                self.finish(str(self.has_loaded_current_user))

        class CurrentUserModuleHandler(CurrentUserHandler):
            def get_template_namespace(self):
                # If RequestHandler.get_template_namespace is called, then
                # get_current_user is evaluated. Until #820 is fixed, this
                # is a small hack to circumvent the issue.
                return self.ui

        class WithoutUserModuleHandler(CurrentUserModuleHandler):
            def get(self):
                self.render_string("without_user_module.html")
                self.finish(str(self.has_loaded_current_user))

        class WithUserModuleHandler(CurrentUserModuleHandler):
            def get(self):
                self.render_string("with_user_module.html")
                self.finish(str(self.has_loaded_current_user))

        return [
            ("/without_user", WithoutUserHandler),
            ("/with_user", WithUserHandler),
            ("/without_user_module", WithoutUserModuleHandler),
            ("/with_user_module", WithUserModuleHandler),
        ]

    @unittest.skip("needs fix")
    def test_get_current_user_is_lazy(self):
        # TODO: Make this test pass. See #820.
        response = self.fetch("/without_user")
        self.assertEqual(response.body, b"False")

    def test_get_current_user_works(self):
        response = self.fetch("/with_user")
        self.assertEqual(response.body, b"True")

    def test_get_current_user_from_ui_module_is_lazy(self):
        response = self.fetch("/without_user_module")
        self.assertEqual(response.body, b"False")

    def test_get_current_user_from_ui_module_works(self):
        response = self.fetch("/with_user_module")
        self.assertEqual(response.body, b"True")


class UnimplementedHTTPMethodsTest(SimpleHandlerTestCase):
    class Handler(RequestHandler):
        pass

    def test_unimplemented_standard_methods(self):
        for method in ["HEAD", "GET", "DELETE", "OPTIONS"]:
            response = self.fetch("/", method=method)
            self.assertEqual(response.code, 405)
        for method in ["POST", "PUT"]:
            response = self.fetch("/", method=method, body=b"")
            self.assertEqual(response.code, 405)


class UnimplementedNonStandardMethodsTest(SimpleHandlerTestCase):
    class Handler(RequestHandler):
        def other(self):
            # Even though this method exists, it won't get called automatically
            # because it is not in SUPPORTED_METHODS.
            self.write("other")

    def test_unimplemented_patch(self):
        # PATCH is recently standardized; Tornado supports it by default
        # but wsgiref.validate doesn't like it.
        response = self.fetch("/", method="PATCH", body=b"")
        self.assertEqual(response.code, 405)

    def test_unimplemented_other(self):
        response = self.fetch("/", method="OTHER", allow_nonstandard_methods=True)
        self.assertEqual(response.code, 405)


class AllHTTPMethodsTest(SimpleHandlerTestCase):
    class Handler(RequestHandler):
        def method(self):
            assert self.request.method is not None
            self.write(self.request.method)

        get = delete = options = post = put = method  # type: ignore

    def test_standard_methods(self):
        response = self.fetch("/", method="HEAD")
        self.assertEqual(response.body, b"")
        for method in ["GET", "DELETE", "OPTIONS"]:
            response = self.fetch("/", method=method)
            self.assertEqual(response.body, utf8(method))
        for method in ["POST", "PUT"]:
            response = self.fetch("/", method=method, body=b"")
            self.assertEqual(response.body, utf8(method))


class PatchMethodTest(SimpleHandlerTestCase):
    class Handler(RequestHandler):
        SUPPORTED_METHODS = RequestHandler.SUPPORTED_METHODS + (  # type: ignore
            "OTHER",
        )

        def patch(self):
            self.write("patch")

        def other(self):
            self.write("other")

    def test_patch(self):
        response = self.fetch("/", method="PATCH", body=b"")
        self.assertEqual(response.body, b"patch")

    def test_other(self):
        response = self.fetch("/", method="OTHER", allow_nonstandard_methods=True)
        self.assertEqual(response.body, b"other")


class FinishInPrepareTest(SimpleHandlerTestCase):
    class Handler(RequestHandler):
        def prepare(self):
            self.finish("done")

        def get(self):
            # It's difficult to assert for certain that a method did not
            # or will not be called in an asynchronous context, but this
            # will be logged noisily if it is reached.
            raise Exception("should not reach this method")

    def test_finish_in_prepare(self):
        response = self.fetch("/")
        self.assertEqual(response.body, b"done")


class Default404Test(WebTestCase):
    def get_handlers(self):
        # If there are no handlers at all a default redirect handler gets added.
        return [("/foo", RequestHandler)]

    def test_404(self):
        response = self.fetch("/")
        self.assertEqual(response.code, 404)
        self.assertEqual(
            response.body,
            b"<html><title>404: Not Found</title>"
            b"<body>404: Not Found</body></html>",
        )


class Custom404Test(WebTestCase):
    def get_handlers(self):
        return [("/foo", RequestHandler)]

    def get_app_kwargs(self):
        class Custom404Handler(RequestHandler):
            def get(self):
                self.set_status(404)
                self.write("custom 404 response")

        return dict(default_handler_class=Custom404Handler)

    def test_404(self):
        response = self.fetch("/")
        self.assertEqual(response.code, 404)
        self.assertEqual(response.body, b"custom 404 response")


class DefaultHandlerArgumentsTest(WebTestCase):
    def get_handlers(self):
        return [("/foo", RequestHandler)]

    def get_app_kwargs(self):
        return dict(
            default_handler_class=ErrorHandler,
            default_handler_args=dict(status_code=403),
        )

    def test_403(self):
        response = self.fetch("/")
        self.assertEqual(response.code, 403)


class HandlerByNameTest(WebTestCase):
    def get_handlers(self):
        # All three are equivalent.
        return [
            ("/hello1", HelloHandler),
            ("/hello2", "tornado.test.web_test.HelloHandler"),
            url("/hello3", "tornado.test.web_test.HelloHandler"),
        ]

    def test_handler_by_name(self):
        resp = self.fetch("/hello1")
        self.assertEqual(resp.body, b"hello")
        resp = self.fetch("/hello2")
        self.assertEqual(resp.body, b"hello")
        resp = self.fetch("/hello3")
        self.assertEqual(resp.body, b"hello")


class StreamingRequestBodyTest(WebTestCase):
    def get_handlers(self):
        @stream_request_body
        class StreamingBodyHandler(RequestHandler):
            def initialize(self, test):
                self.test = test

            def prepare(self):
                self.test.prepared.set_result(None)

            def data_received(self, data):
                self.test.data.set_result(data)

            def get(self):
                self.test.finished.set_result(None)
                self.write({})

        @stream_request_body
        class EarlyReturnHandler(RequestHandler):
            def prepare(self):
                # If we finish the response in prepare, it won't continue to
                # the (non-existent) data_received.
                raise HTTPError(401)

        @stream_request_body
        class CloseDetectionHandler(RequestHandler):
            def initialize(self, test):
                self.test = test

            def on_connection_close(self):
                super().on_connection_close()
                self.test.close_future.set_result(None)

        return [
            ("/stream_body", StreamingBodyHandler, dict(test=self)),
            ("/early_return", EarlyReturnHandler),
            ("/close_detection", CloseDetectionHandler, dict(test=self)),
        ]

    def connect(self, url, connection_close):
        # Use a raw connection so we can control the sending of data.
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
        s.connect(("127.0.0.1", self.get_http_port()))
        stream = IOStream(s)
        stream.write(b"GET " + url + b" HTTP/1.1\r\n")
        if connection_close:
            stream.write(b"Connection: close\r\n")
        stream.write(b"Transfer-Encoding: chunked\r\n\r\n")
        return stream

    @gen_test
    def test_streaming_body(self):
        self.prepared = Future()  # type: Future[None]
        self.data = Future()  # type: Future[bytes]
        self.finished = Future()  # type: Future[None]

        stream = self.connect(b"/stream_body", connection_close=True)
        yield self.prepared
        stream.write(b"4\r\nasdf\r\n")
        # Ensure the first chunk is received before we send the second.
        data = yield self.data
        self.assertEqual(data, b"asdf")
        self.data = Future()
        stream.write(b"4\r\nqwer\r\n")
        data = yield self.data
        self.assertEqual(data, b"qwer")
        stream.write(b"0\r\n\r\n")
        yield self.finished
        data = yield stream.read_until_close()
        # This would ideally use an HTTP1Connection to read the response.
        self.assertTrue(data.endswith(b"{}"))
        stream.close()

    @gen_test
    def test_early_return(self):
        stream = self.connect(b"/early_return", connection_close=False)
        data = yield stream.read_until_close()
        self.assertTrue(data.startswith(b"HTTP/1.1 401"))

    @gen_test
    def test_early_return_with_data(self):
        stream = self.connect(b"/early_return", connection_close=False)
        stream.write(b"4\r\nasdf\r\n")
        data = yield stream.read_until_close()
        self.assertTrue(data.startswith(b"HTTP/1.1 401"))

    @gen_test
    def test_close_during_upload(self):
        self.close_future = Future()  # type: Future[None]
        stream = self.connect(b"/close_detection", connection_close=False)
        stream.close()
        yield self.close_future


# Each method in this handler returns a yieldable object and yields to the
# IOLoop so the future is not immediately ready.  Ensure that the
# yieldables are respected and no method is called before the previous
# one has completed.
@stream_request_body
class BaseFlowControlHandler(RequestHandler):
    def initialize(self, test):
        self.test = test
        self.method = None
        self.methods = []  # type: typing.List[str]

    @contextlib.contextmanager
    def in_method(self, method):
        if self.method is not None:
            self.test.fail("entered method %s while in %s" % (method, self.method))
        self.method = method
        self.methods.append(method)
        try:
            yield
        finally:
            self.method = None

    @gen.coroutine
    def prepare(self):
        # Note that asynchronous prepare() does not block data_received,
        # so we don't use in_method here.
        self.methods.append("prepare")
        yield gen.moment

    @gen.coroutine
    def post(self):
        with self.in_method("post"):
            yield gen.moment
        self.write(dict(methods=self.methods))


class BaseStreamingRequestFlowControlTest(object):
    def get_httpserver_options(self):
        # Use a small chunk size so flow control is relevant even though
        # all the data arrives at once.
        return dict(chunk_size=10, decompress_request=True)

    def get_http_client(self):
        # simple_httpclient only: curl doesn't support body_producer.
        return SimpleAsyncHTTPClient()

    # Test all the slightly different code paths for fixed, chunked, etc bodies.
    def test_flow_control_fixed_body(self: typing.Any):
        response = self.fetch("/", body="abcdefghijklmnopqrstuvwxyz", method="POST")
        response.rethrow()
        self.assertEqual(
            json_decode(response.body),
            dict(
                methods=[
                    "prepare",
                    "data_received",
                    "data_received",
                    "data_received",
                    "post",
                ]
            ),
        )

    def test_flow_control_chunked_body(self: typing.Any):
        chunks = [b"abcd", b"efgh", b"ijkl"]

        @gen.coroutine
        def body_producer(write):
            for i in chunks:
                yield write(i)

        response = self.fetch("/", body_producer=body_producer, method="POST")
        response.rethrow()
        self.assertEqual(
            json_decode(response.body),
            dict(
                methods=[
                    "prepare",
                    "data_received",
                    "data_received",
                    "data_received",
                    "post",
                ]
            ),
        )

    def test_flow_control_compressed_body(self: typing.Any):
        bytesio = BytesIO()
        gzip_file = gzip.GzipFile(mode="w", fileobj=bytesio)
        gzip_file.write(b"abcdefghijklmnopqrstuvwxyz")
        gzip_file.close()
        compressed_body = bytesio.getvalue()
        response = self.fetch(
            "/",
            body=compressed_body,
            method="POST",
            headers={"Content-Encoding": "gzip"},
        )
        response.rethrow()
        self.assertEqual(
            json_decode(response.body),
            dict(
                methods=[
                    "prepare",
                    "data_received",
                    "data_received",
                    "data_received",
                    "post",
                ]
            ),
        )


class DecoratedStreamingRequestFlowControlTest(
    BaseStreamingRequestFlowControlTest, WebTestCase
):
    def get_handlers(self):
        class DecoratedFlowControlHandler(BaseFlowControlHandler):
            @gen.coroutine
            def data_received(self, data):
                with self.in_method("data_received"):
                    yield gen.moment

        return [("/", DecoratedFlowControlHandler, dict(test=self))]


class NativeStreamingRequestFlowControlTest(
    BaseStreamingRequestFlowControlTest, WebTestCase
):
    def get_handlers(self):
        class NativeFlowControlHandler(BaseFlowControlHandler):
            async def data_received(self, data):
                with self.in_method("data_received"):
                    import asyncio

                    await asyncio.sleep(0)

        return [("/", NativeFlowControlHandler, dict(test=self))]


class IncorrectContentLengthTest(SimpleHandlerTestCase):
    def get_handlers(self):
        test = self
        self.server_error = None

        # Manually set a content-length that doesn't match the actual content.
        class TooHigh(RequestHandler):
            def get(self):
                self.set_header("Content-Length", "42")
                try:
                    self.finish("ok")
                except Exception as e:
                    test.server_error = e
                    raise

        class TooLow(RequestHandler):
            def get(self):
                self.set_header("Content-Length", "2")
                try:
                    self.finish("hello")
                except Exception as e:
                    test.server_error = e
                    raise

        return [("/high", TooHigh), ("/low", TooLow)]

    def test_content_length_too_high(self):
        # When the content-length is too high, the connection is simply
        # closed without completing the response.  An error is logged on
        # the server.
        with ExpectLog(app_log, "(Uncaught exception|Exception in callback)"):
            with ExpectLog(
                gen_log,
                "(Cannot send error response after headers written"
                "|Failed to flush partial response)",
            ):
                with self.assertRaises(HTTPClientError):
                    self.fetch("/high", raise_error=True)
        self.assertEqual(
            str(self.server_error), "Tried to write 40 bytes less than Content-Length"
        )

    def test_content_length_too_low(self):
        # When the content-length is too low, the connection is closed
        # without writing the last chunk, so the client never sees the request
        # complete (which would be a framing error).
        with ExpectLog(app_log, "(Uncaught exception|Exception in callback)"):
            with ExpectLog(
                gen_log,
                "(Cannot send error response after headers written"
                "|Failed to flush partial response)",
            ):
                with self.assertRaises(HTTPClientError):
                    self.fetch("/low", raise_error=True)
        self.assertEqual(
            str(self.server_error), "Tried to write more data than Content-Length"
        )


class ClientCloseTest(SimpleHandlerTestCase):
    class Handler(RequestHandler):
        def get(self):
            if self.request.version.startswith("HTTP/1"):
                # Simulate a connection closed by the client during
                # request processing.  The client will see an error, but the
                # server should respond gracefully (without logging errors
                # because we were unable to write out as many bytes as
                # Content-Length said we would)
                self.request.connection.stream.close()  # type: ignore
                self.write("hello")
            else:
                # TODO: add a HTTP2-compatible version of this test.
                self.write("requires HTTP/1.x")

    def test_client_close(self):
        with self.assertRaises((HTTPClientError, unittest.SkipTest)):  # type: ignore
            response = self.fetch("/", raise_error=True)
            if response.body == b"requires HTTP/1.x":
                self.skipTest("requires HTTP/1.x")
            self.assertEqual(response.code, 599)


class SignedValueTest(unittest.TestCase):
    SECRET = "It's a secret to everybody"
    SECRET_DICT = {0: "asdfbasdf", 1: "12312312", 2: "2342342"}

    def past(self):
        return self.present() - 86400 * 32

    def present(self):
        return 1300000000

    def test_known_values(self):
        signed_v1 = create_signed_value(
            SignedValueTest.SECRET, "key", "value", version=1, clock=self.present
        )
        self.assertEqual(
            signed_v1, b"dmFsdWU=|1300000000|31c934969f53e48164c50768b40cbd7e2daaaa4f"
        )

        signed_v2 = create_signed_value(
            SignedValueTest.SECRET, "key", "value", version=2, clock=self.present
        )
        self.assertEqual(
            signed_v2,
            b"2|1:0|10:1300000000|3:key|8:dmFsdWU=|"
            b"3d4e60b996ff9c5d5788e333a0cba6f238a22c6c0f94788870e1a9ecd482e152",
        )

        signed_default = create_signed_value(
            SignedValueTest.SECRET, "key", "value", clock=self.present
        )
        self.assertEqual(signed_default, signed_v2)

        decoded_v1 = decode_signed_value(
            SignedValueTest.SECRET, "key", signed_v1, min_version=1, clock=self.present
        )
        self.assertEqual(decoded_v1, b"value")

        decoded_v2 = decode_signed_value(
            SignedValueTest.SECRET, "key", signed_v2, min_version=2, clock=self.present
        )
        self.assertEqual(decoded_v2, b"value")

    def test_name_swap(self):
        signed1 = create_signed_value(
            SignedValueTest.SECRET, "key1", "value", clock=self.present
        )
        signed2 = create_signed_value(
            SignedValueTest.SECRET, "key2", "value", clock=self.present
        )
        # Try decoding each string with the other's "name"
        decoded1 = decode_signed_value(
            SignedValueTest.SECRET, "key2", signed1, clock=self.present
        )
        self.assertIs(decoded1, None)
        decoded2 = decode_signed_value(
            SignedValueTest.SECRET, "key1", signed2, clock=self.present
        )
        self.assertIs(decoded2, None)

    def test_expired(self):
        signed = create_signed_value(
            SignedValueTest.SECRET, "key1", "value", clock=self.past
        )
        decoded_past = decode_signed_value(
            SignedValueTest.SECRET, "key1", signed, clock=self.past
        )
        self.assertEqual(decoded_past, b"value")
        decoded_present = decode_signed_value(
            SignedValueTest.SECRET, "key1", signed, clock=self.present
        )
        self.assertIs(decoded_present, None)

    def test_payload_tampering(self):
        # These cookies are variants of the one in test_known_values.
        sig = "3d4e60b996ff9c5d5788e333a0cba6f238a22c6c0f94788870e1a9ecd482e152"

        def validate(prefix):
            return b"value" == decode_signed_value(
                SignedValueTest.SECRET, "key", prefix + sig, clock=self.present
            )

        self.assertTrue(validate("2|1:0|10:1300000000|3:key|8:dmFsdWU=|"))
        # Change key version
        self.assertFalse(validate("2|1:1|10:1300000000|3:key|8:dmFsdWU=|"))
        # length mismatch (field too short)
        self.assertFalse(validate("2|1:0|10:130000000|3:key|8:dmFsdWU=|"))
        # length mismatch (field too long)
        self.assertFalse(validate("2|1:0|10:1300000000|3:keey|8:dmFsdWU=|"))

    def test_signature_tampering(self):
        prefix = "2|1:0|10:1300000000|3:key|8:dmFsdWU=|"

        def validate(sig):
            return b"value" == decode_signed_value(
                SignedValueTest.SECRET, "key", prefix + sig, clock=self.present
            )

        self.assertTrue(
            validate("3d4e60b996ff9c5d5788e333a0cba6f238a22c6c0f94788870e1a9ecd482e152")
        )
        # All zeros
        self.assertFalse(validate("0" * 32))
        # Change one character
        self.assertFalse(
            validate("4d4e60b996ff9c5d5788e333a0cba6f238a22c6c0f94788870e1a9ecd482e152")
        )
        # Change another character
        self.assertFalse(
            validate("3d4e60b996ff9c5d5788e333a0cba6f238a22c6c0f94788870e1a9ecd482e153")
        )
        # Truncate
        self.assertFalse(
            validate("3d4e60b996ff9c5d5788e333a0cba6f238a22c6c0f94788870e1a9ecd482e15")
        )
        # Lengthen
        self.assertFalse(
            validate(
                "3d4e60b996ff9c5d5788e333a0cba6f238a22c6c0f94788870e1a9ecd482e1538"
            )
        )

    def test_non_ascii(self):
        value = b"\xe9"
        signed = create_signed_value(
            SignedValueTest.SECRET, "key", value, clock=self.present
        )
        decoded = decode_signed_value(
            SignedValueTest.SECRET, "key", signed, clock=self.present
        )
        self.assertEqual(value, decoded)

    def test_key_versioning_read_write_default_key(self):
        value = b"\xe9"
        signed = create_signed_value(
            SignedValueTest.SECRET_DICT, "key", value, clock=self.present, key_version=0
        )
        decoded = decode_signed_value(
            SignedValueTest.SECRET_DICT, "key", signed, clock=self.present
        )
        self.assertEqual(value, decoded)

    def test_key_versioning_read_write_non_default_key(self):
        value = b"\xe9"
        signed = create_signed_value(
            SignedValueTest.SECRET_DICT, "key", value, clock=self.present, key_version=1
        )
        decoded = decode_signed_value(
            SignedValueTest.SECRET_DICT, "key", signed, clock=self.present
        )
        self.assertEqual(value, decoded)

    def test_key_versioning_invalid_key(self):
        value = b"\xe9"
        signed = create_signed_value(
            SignedValueTest.SECRET_DICT, "key", value, clock=self.present, key_version=0
        )
        newkeys = SignedValueTest.SECRET_DICT.copy()
        newkeys.pop(0)
        decoded = decode_signed_value(newkeys, "key", signed, clock=self.present)
        self.assertEqual(None, decoded)

    def test_key_version_retrieval(self):
        value = b"\xe9"
        signed = create_signed_value(
            SignedValueTest.SECRET_DICT, "key", value, clock=self.present, key_version=1
        )
        key_version = get_signature_key_version(signed)
        self.assertEqual(1, key_version)


class XSRFTest(SimpleHandlerTestCase):
    class Handler(RequestHandler):
        def get(self):
            version = int(self.get_argument("version", "2"))
            # This would be a bad idea in a real app, but in this test
            # it's fine.
            self.settings["xsrf_cookie_version"] = version
            self.write(self.xsrf_token)

        def post(self):
            self.write("ok")

    def get_app_kwargs(self):
        return dict(xsrf_cookies=True)

    def setUp(self):
        super().setUp()
        self.xsrf_token = self.get_token()

    def get_token(self, old_token=None, version=None):
        if old_token is not None:
            headers = self.cookie_headers(old_token)
        else:
            headers = None
        response = self.fetch(
            "/" if version is None else ("/?version=%d" % version), headers=headers
        )
        response.rethrow()
        return native_str(response.body)

    def cookie_headers(self, token=None):
        if token is None:
            token = self.xsrf_token
        return {"Cookie": "_xsrf=" + token}

    def test_xsrf_fail_no_token(self):
        with ExpectLog(gen_log, ".*'_xsrf' argument missing"):
            response = self.fetch("/", method="POST", body=b"")
        self.assertEqual(response.code, 403)

    def test_xsrf_fail_body_no_cookie(self):
        with ExpectLog(gen_log, ".*XSRF cookie does not match POST"):
            response = self.fetch(
                "/",
                method="POST",
                body=urllib.parse.urlencode(dict(_xsrf=self.xsrf_token)),
            )
        self.assertEqual(response.code, 403)

    def test_xsrf_fail_argument_invalid_format(self):
        with ExpectLog(gen_log, ".*'_xsrf' argument has invalid format"):
            response = self.fetch(
                "/",
                method="POST",
                headers=self.cookie_headers(),
                body=urllib.parse.urlencode(dict(_xsrf="3|")),
            )
        self.assertEqual(response.code, 403)

    def test_xsrf_fail_cookie_invalid_format(self):
        with ExpectLog(gen_log, ".*XSRF cookie does not match POST"):
            response = self.fetch(
                "/",
                method="POST",
                headers=self.cookie_headers(token="3|"),
                body=urllib.parse.urlencode(dict(_xsrf=self.xsrf_token)),
            )
        self.assertEqual(response.code, 403)

    def test_xsrf_fail_cookie_no_body(self):
        with ExpectLog(gen_log, ".*'_xsrf' argument missing"):
            response = self.fetch(
                "/", method="POST", body=b"", headers=self.cookie_headers()
            )
        self.assertEqual(response.code, 403)

    def test_xsrf_success_short_token(self):
        response = self.fetch(
            "/",
            method="POST",
            body=urllib.parse.urlencode(dict(_xsrf="deadbeef")),
            headers=self.cookie_headers(token="deadbeef"),
        )
        self.assertEqual(response.code, 200)

    def test_xsrf_success_non_hex_token(self):
        response = self.fetch(
            "/",
            method="POST",
            body=urllib.parse.urlencode(dict(_xsrf="xoxo")),
            headers=self.cookie_headers(token="xoxo"),
        )
        self.assertEqual(response.code, 200)

    def test_xsrf_success_post_body(self):
        response = self.fetch(
            "/",
            method="POST",
            body=urllib.parse.urlencode(dict(_xsrf=self.xsrf_token)),
            headers=self.cookie_headers(),
        )
        self.assertEqual(response.code, 200)

    def test_xsrf_success_query_string(self):
        response = self.fetch(
            "/?" + urllib.parse.urlencode(dict(_xsrf=self.xsrf_token)),
            method="POST",
            body=b"",
            headers=self.cookie_headers(),
        )
        self.assertEqual(response.code, 200)

    def test_xsrf_success_header(self):
        response = self.fetch(
            "/",
            method="POST",
            body=b"",
            headers=dict(
                {"X-Xsrftoken": self.xsrf_token},  # type: ignore
                **self.cookie_headers(),
            ),
        )
        self.assertEqual(response.code, 200)

    def test_distinct_tokens(self):
        # Every request gets a distinct token.
        NUM_TOKENS = 10
        tokens = set()
        for i in range(NUM_TOKENS):
            tokens.add(self.get_token())
        self.assertEqual(len(tokens), NUM_TOKENS)

    def test_cross_user(self):
        token2 = self.get_token()
        # Each token can be used to authenticate its own request.
        for token in (self.xsrf_token, token2):
            response = self.fetch(
                "/",
                method="POST",
                body=urllib.parse.urlencode(dict(_xsrf=token)),
                headers=self.cookie_headers(token),
            )
            self.assertEqual(response.code, 200)
        # Sending one in the cookie and the other in the body is not allowed.
        for cookie_token, body_token in (
            (self.xsrf_token, token2),
            (token2, self.xsrf_token),
        ):
            with ExpectLog(gen_log, ".*XSRF cookie does not match POST"):
                response = self.fetch(
                    "/",
                    method="POST",
                    body=urllib.parse.urlencode(dict(_xsrf=body_token)),
                    headers=self.cookie_headers(cookie_token),
                )
            self.assertEqual(response.code, 403)

    def test_refresh_token(self):
        token = self.xsrf_token
        tokens_seen = set([token])
        # A user's token is stable over time.  Refreshing the page in one tab
        # might update the cookie while an older tab still has the old cookie
        # in its DOM.  Simulate this scenario by passing a constant token
        # in the body and re-querying for the token.
        for i in range(5):
            token = self.get_token(token)
            # Tokens are encoded uniquely each time
            tokens_seen.add(token)
            response = self.fetch(
                "/",
                method="POST",
                body=urllib.parse.urlencode(dict(_xsrf=self.xsrf_token)),
                headers=self.cookie_headers(token),
            )
            self.assertEqual(response.code, 200)
        self.assertEqual(len(tokens_seen), 6)

    def test_versioning(self):
        # Version 1 still produces distinct tokens per request.
        self.assertNotEqual(self.get_token(version=1), self.get_token(version=1))

        # Refreshed v1 tokens are all identical.
        v1_token = self.get_token(version=1)
        for i in range(5):
            self.assertEqual(self.get_token(v1_token, version=1), v1_token)

        # Upgrade to a v2 version of the same token
        v2_token = self.get_token(v1_token)
        self.assertNotEqual(v1_token, v2_token)
        # Each v1 token can map to many v2 tokens.
        self.assertNotEqual(v2_token, self.get_token(v1_token))

        # The tokens are cross-compatible.
        for cookie_token, body_token in ((v1_token, v2_token), (v2_token, v1_token)):
            response = self.fetch(
                "/",
                method="POST",
                body=urllib.parse.urlencode(dict(_xsrf=body_token)),
                headers=self.cookie_headers(cookie_token),
            )
            self.assertEqual(response.code, 200)


# A subset of the previous test with a different cookie name
class XSRFCookieNameTest(SimpleHandlerTestCase):
    class Handler(RequestHandler):
        def get(self):
            self.write(self.xsrf_token)

        def post(self):
            self.write("ok")

    def get_app_kwargs(self):
        return dict(
            xsrf_cookies=True,
            xsrf_cookie_name="__Host-xsrf",
            xsrf_cookie_kwargs={"secure": True},
        )

    def setUp(self):
        super().setUp()
        self.xsrf_token = self.get_token()

    def get_token(self, old_token=None):
        if old_token is not None:
            headers = self.cookie_headers(old_token)
        else:
            headers = None
        response = self.fetch("/", headers=headers)
        response.rethrow()
        return native_str(response.body)

    def cookie_headers(self, token=None):
        if token is None:
            token = self.xsrf_token
        return {"Cookie": "__Host-xsrf=" + token}

    def test_xsrf_fail_no_token(self):
        with ExpectLog(gen_log, ".*'_xsrf' argument missing"):
            response = self.fetch("/", method="POST", body=b"")
        self.assertEqual(response.code, 403)

    def test_xsrf_fail_body_no_cookie(self):
        with ExpectLog(gen_log, ".*XSRF cookie does not match POST"):
            response = self.fetch(
                "/",
                method="POST",
                body=urllib.parse.urlencode(dict(_xsrf=self.xsrf_token)),
            )
        self.assertEqual(response.code, 403)

    def test_xsrf_success_post_body(self):
        response = self.fetch(
            "/",
            method="POST",
            # Note that renaming the cookie doesn't rename the POST param
            body=urllib.parse.urlencode(dict(_xsrf=self.xsrf_token)),
            headers=self.cookie_headers(),
        )
        self.assertEqual(response.code, 200)


class XSRFCookieKwargsTest(SimpleHandlerTestCase):
    class Handler(RequestHandler):
        def get(self):
            self.write(self.xsrf_token)

    def get_app_kwargs(self):
        return dict(
            xsrf_cookies=True, xsrf_cookie_kwargs=dict(httponly=True, expires_days=2)
        )

    def test_xsrf_httponly(self):
        response = self.fetch("/")
        self.assertIn("httponly;", response.headers["Set-Cookie"].lower())
        self.assertIn("expires=", response.headers["Set-Cookie"].lower())
        header = response.headers.get("Set-Cookie")
        assert header is not None
        match = re.match(".*; expires=(?P<expires>.+);.*", header)
        assert match is not None

        expires = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
            days=2
        )
        header_expires = email.utils.parsedate_to_datetime(match.groupdict()["expires"])
        if header_expires.tzinfo is None:
            header_expires = header_expires.replace(tzinfo=datetime.timezone.utc)
        self.assertTrue(abs((expires - header_expires).total_seconds()) < 10)


class FinishExceptionTest(SimpleHandlerTestCase):
    class Handler(RequestHandler):
        def get(self):
            self.set_status(401)
            self.set_header("WWW-Authenticate", 'Basic realm="something"')
            if self.get_argument("finish_value", ""):
                raise Finish("authentication required")
            else:
                self.write("authentication required")
                raise Finish()

    def test_finish_exception(self):
        for u in ["/", "/?finish_value=1"]:
            response = self.fetch(u)
            self.assertEqual(response.code, 401)
            self.assertEqual(
                'Basic realm="something"', response.headers.get("WWW-Authenticate")
            )
            self.assertEqual(b"authentication required", response.body)


class DecoratorTest(WebTestCase):
    def get_handlers(self):
        class RemoveSlashHandler(RequestHandler):
            @removeslash
            def get(self):
                pass

        class AddSlashHandler(RequestHandler):
            @addslash
            def get(self):
                pass

        return [("/removeslash/", RemoveSlashHandler), ("/addslash", AddSlashHandler)]

    def test_removeslash(self):
        response = self.fetch("/removeslash/", follow_redirects=False)
        self.assertEqual(response.code, 301)
        self.assertEqual(response.headers["Location"], "/removeslash")

        response = self.fetch("/removeslash/?foo=bar", follow_redirects=False)
        self.assertEqual(response.code, 301)
        self.assertEqual(response.headers["Location"], "/removeslash?foo=bar")

    def test_addslash(self):
        response = self.fetch("/addslash", follow_redirects=False)
        self.assertEqual(response.code, 301)
        self.assertEqual(response.headers["Location"], "/addslash/")

        response = self.fetch("/addslash?foo=bar", follow_redirects=False)
        self.assertEqual(response.code, 301)
        self.assertEqual(response.headers["Location"], "/addslash/?foo=bar")


class CacheTest(WebTestCase):
    def get_handlers(self):
        class EtagHandler(RequestHandler):
            def get(self, computed_etag):
                self.write(computed_etag)

            def compute_etag(self):
                return self._write_buffer[0]

        return [("/etag/(.*)", EtagHandler)]

    def test_wildcard_etag(self):
        computed_etag = '"xyzzy"'
        etags = "*"
        self._test_etag(computed_etag, etags, 304)

    def test_strong_etag_match(self):
        computed_etag = '"xyzzy"'
        etags = '"xyzzy"'
        self._test_etag(computed_etag, etags, 304)

    def test_multiple_strong_etag_match(self):
        computed_etag = '"xyzzy1"'
        etags = '"xyzzy1", "xyzzy2"'
        self._test_etag(computed_etag, etags, 304)

    def test_strong_etag_not_match(self):
        computed_etag = '"xyzzy"'
        etags = '"xyzzy1"'
        self._test_etag(computed_etag, etags, 200)

    def test_multiple_strong_etag_not_match(self):
        computed_etag = '"xyzzy"'
        etags = '"xyzzy1", "xyzzy2"'
        self._test_etag(computed_etag, etags, 200)

    def test_weak_etag_match(self):
        computed_etag = '"xyzzy1"'
        etags = 'W/"xyzzy1"'
        self._test_etag(computed_etag, etags, 304)

    def test_multiple_weak_etag_match(self):
        computed_etag = '"xyzzy2"'
        etags = 'W/"xyzzy1", W/"xyzzy2"'
        self._test_etag(computed_etag, etags, 304)

    def test_weak_etag_not_match(self):
        computed_etag = '"xyzzy2"'
        etags = 'W/"xyzzy1"'
        self._test_etag(computed_etag, etags, 200)

    def test_multiple_weak_etag_not_match(self):
        computed_etag = '"xyzzy3"'
        etags = 'W/"xyzzy1", W/"xyzzy2"'
        self._test_etag(computed_etag, etags, 200)

    def _test_etag(self, computed_etag, etags, status_code):
        response = self.fetch(
            "/etag/" + computed_etag, headers={"If-None-Match": etags}
        )
        self.assertEqual(response.code, status_code)


class RequestSummaryTest(SimpleHandlerTestCase):
    class Handler(RequestHandler):
        def get(self):
            # remote_ip is optional, although it's set by
            # both HTTPServer and WSGIAdapter.
            # Clobber it to make sure it doesn't break logging.
            self.request.remote_ip = None
            self.finish(self._request_summary())

    def test_missing_remote_ip(self):
        resp = self.fetch("/")
        self.assertEqual(resp.body, b"GET / (None)")


class HTTPErrorTest(unittest.TestCase):
    def test_copy(self):
        e = HTTPError(403, reason="Go away")
        e2 = copy.copy(e)
        self.assertIsNot(e, e2)
        self.assertEqual(e.status_code, e2.status_code)
        self.assertEqual(e.reason, e2.reason)


class ApplicationTest(AsyncTestCase):
    def test_listen(self):
        app = Application([])
        server = app.listen(0, address="127.0.0.1")
        server.stop()


class URLSpecReverseTest(unittest.TestCase):
    def test_reverse(self):
        self.assertEqual("/favicon.ico", url(r"/favicon\.ico", None).reverse())
        self.assertEqual("/favicon.ico", url(r"^/favicon\.ico$", None).reverse())

    def test_non_reversible(self):
        # URLSpecs are non-reversible if they include non-constant
        # regex features outside capturing groups. Currently, this is
        # only strictly enforced for backslash-escaped character
        # classes.
        paths = [r"^/api/v\d+/foo/(\w+)$"]
        for path in paths:
            # A URLSpec can still be created even if it cannot be reversed.
            url_spec = url(path, None)
            try:
                result = url_spec.reverse()
                self.fail(
                    "did not get expected exception when reversing %s. "
                    "result: %s" % (path, result)
                )
            except ValueError:
                pass

    def test_reverse_arguments(self):
        self.assertEqual(
            "/api/v1/foo/bar", url(r"^/api/v1/foo/(\w+)$", None).reverse("bar")
        )
        self.assertEqual(
            "/api.v1/foo/5/icon.png",
            url(r"/api\.v1/foo/([0-9]+)/icon\.png", None).reverse(5),
        )


class RedirectHandlerTest(WebTestCase):
    def get_handlers(self):
        return [
            ("/src", WebRedirectHandler, {"url": "/dst"}),
            ("/src2", WebRedirectHandler, {"url": "/dst2?foo=bar"}),
            (r"/(.*?)/(.*?)/(.*)", WebRedirectHandler, {"url": "/{1}/{0}/{2}"}),
        ]

    def test_basic_redirect(self):
        response = self.fetch("/src", follow_redirects=False)
        self.assertEqual(response.code, 301)
        self.assertEqual(response.headers["Location"], "/dst")

    def test_redirect_with_argument(self):
        response = self.fetch("/src?foo=bar", follow_redirects=False)
        self.assertEqual(response.code, 301)
        self.assertEqual(response.headers["Location"], "/dst?foo=bar")

    def test_redirect_with_appending_argument(self):
        response = self.fetch("/src2?foo2=bar2", follow_redirects=False)
        self.assertEqual(response.code, 301)
        self.assertEqual(response.headers["Location"], "/dst2?foo=bar&foo2=bar2")

    def test_redirect_pattern(self):
        response = self.fetch("/a/b/c", follow_redirects=False)
        self.assertEqual(response.code, 301)
        self.assertEqual(response.headers["Location"], "/b/a/c")


class AcceptLanguageTest(WebTestCase):
    """Test evaluation of Accept-Language header"""

    def get_handlers(self):
        locale.load_gettext_translations(
            os.path.join(os.path.dirname(__file__), "gettext_translations"),
            "tornado_test",
        )

        class AcceptLanguageHandler(RequestHandler):
            def get(self):
                self.set_header(
                    "Content-Language", self.get_browser_locale().code.replace("_", "-")
                )
                self.finish(b"")

        return [
            ("/", AcceptLanguageHandler),
        ]

    def test_accept_language(self):
        response = self.fetch("/", headers={"Accept-Language": "fr-FR;q=0.9"})
        self.assertEqual(response.headers["Content-Language"], "fr-FR")

        response = self.fetch("/", headers={"Accept-Language": "fr-FR; q=0.9"})
        self.assertEqual(response.headers["Content-Language"], "fr-FR")

    def test_accept_language_ignore(self):
        response = self.fetch("/", headers={"Accept-Language": "fr-FR;q=0"})
        self.assertEqual(response.headers["Content-Language"], "en-US")

    def test_accept_language_invalid(self):
        response = self.fetch("/", headers={"Accept-Language": "fr-FR;q=-1"})
        self.assertEqual(response.headers["Content-Language"], "en-US")
