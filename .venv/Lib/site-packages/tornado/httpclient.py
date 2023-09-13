"""Blocking and non-blocking HTTP client interfaces.

This module defines a common interface shared by two implementations,
``simple_httpclient`` and ``curl_httpclient``.  Applications may either
instantiate their chosen implementation class directly or use the
`AsyncHTTPClient` class from this module, which selects an implementation
that can be overridden with the `AsyncHTTPClient.configure` method.

The default implementation is ``simple_httpclient``, and this is expected
to be suitable for most users' needs.  However, some applications may wish
to switch to ``curl_httpclient`` for reasons such as the following:

* ``curl_httpclient`` has some features not found in ``simple_httpclient``,
  including support for HTTP proxies and the ability to use a specified
  network interface.

* ``curl_httpclient`` is more likely to be compatible with sites that are
  not-quite-compliant with the HTTP spec, or sites that use little-exercised
  features of HTTP.

* ``curl_httpclient`` is faster.

Note that if you are using ``curl_httpclient``, it is highly
recommended that you use a recent version of ``libcurl`` and
``pycurl``.  Currently the minimum supported version of libcurl is
7.22.0, and the minimum version of pycurl is 7.18.2.  It is highly
recommended that your ``libcurl`` installation is built with
asynchronous DNS resolver (threaded or c-ares), otherwise you may
encounter various problems with request timeouts (for more
information, see
http://curl.haxx.se/libcurl/c/curl_easy_setopt.html#CURLOPTCONNECTTIMEOUTMS
and comments in curl_httpclient.py).

To select ``curl_httpclient``, call `AsyncHTTPClient.configure` at startup::

    AsyncHTTPClient.configure("tornado.curl_httpclient.CurlAsyncHTTPClient")
"""

import datetime
import functools
from io import BytesIO
import ssl
import time
import weakref

from tornado.concurrent import (
    Future,
    future_set_result_unless_cancelled,
    future_set_exception_unless_cancelled,
)
from tornado.escape import utf8, native_str
from tornado import gen, httputil
from tornado.ioloop import IOLoop
from tornado.util import Configurable

from typing import Type, Any, Union, Dict, Callable, Optional, cast


class HTTPClient(object):
    """A blocking HTTP client.

    This interface is provided to make it easier to share code between
    synchronous and asynchronous applications. Applications that are
    running an `.IOLoop` must use `AsyncHTTPClient` instead.

    Typical usage looks like this::

        http_client = httpclient.HTTPClient()
        try:
            response = http_client.fetch("http://www.google.com/")
            print(response.body)
        except httpclient.HTTPError as e:
            # HTTPError is raised for non-200 responses; the response
            # can be found in e.response.
            print("Error: " + str(e))
        except Exception as e:
            # Other errors are possible, such as IOError.
            print("Error: " + str(e))
        http_client.close()

    .. versionchanged:: 5.0

       Due to limitations in `asyncio`, it is no longer possible to
       use the synchronous ``HTTPClient`` while an `.IOLoop` is running.
       Use `AsyncHTTPClient` instead.

    """

    def __init__(
        self,
        async_client_class: "Optional[Type[AsyncHTTPClient]]" = None,
        **kwargs: Any
    ) -> None:
        # Initialize self._closed at the beginning of the constructor
        # so that an exception raised here doesn't lead to confusing
        # failures in __del__.
        self._closed = True
        self._io_loop = IOLoop(make_current=False)
        if async_client_class is None:
            async_client_class = AsyncHTTPClient

        # Create the client while our IOLoop is "current", without
        # clobbering the thread's real current IOLoop (if any).
        async def make_client() -> "AsyncHTTPClient":
            await gen.sleep(0)
            assert async_client_class is not None
            return async_client_class(**kwargs)

        self._async_client = self._io_loop.run_sync(make_client)
        self._closed = False

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        """Closes the HTTPClient, freeing any resources used."""
        if not self._closed:
            self._async_client.close()
            self._io_loop.close()
            self._closed = True

    def fetch(
        self, request: Union["HTTPRequest", str], **kwargs: Any
    ) -> "HTTPResponse":
        """Executes a request, returning an `HTTPResponse`.

        The request may be either a string URL or an `HTTPRequest` object.
        If it is a string, we construct an `HTTPRequest` using any additional
        kwargs: ``HTTPRequest(request, **kwargs)``

        If an error occurs during the fetch, we raise an `HTTPError` unless
        the ``raise_error`` keyword argument is set to False.
        """
        response = self._io_loop.run_sync(
            functools.partial(self._async_client.fetch, request, **kwargs)
        )
        return response


class AsyncHTTPClient(Configurable):
    """An non-blocking HTTP client.

    Example usage::

        async def f():
            http_client = AsyncHTTPClient()
            try:
                response = await http_client.fetch("http://www.google.com")
            except Exception as e:
                print("Error: %s" % e)
            else:
                print(response.body)

    The constructor for this class is magic in several respects: It
    actually creates an instance of an implementation-specific
    subclass, and instances are reused as a kind of pseudo-singleton
    (one per `.IOLoop`). The keyword argument ``force_instance=True``
    can be used to suppress this singleton behavior. Unless
    ``force_instance=True`` is used, no arguments should be passed to
    the `AsyncHTTPClient` constructor. The implementation subclass as
    well as arguments to its constructor can be set with the static
    method `configure()`

    All `AsyncHTTPClient` implementations support a ``defaults``
    keyword argument, which can be used to set default values for
    `HTTPRequest` attributes.  For example::

        AsyncHTTPClient.configure(
            None, defaults=dict(user_agent="MyUserAgent"))
        # or with force_instance:
        client = AsyncHTTPClient(force_instance=True,
            defaults=dict(user_agent="MyUserAgent"))

    .. versionchanged:: 5.0
       The ``io_loop`` argument (deprecated since version 4.1) has been removed.

    """

    _instance_cache = None  # type: Dict[IOLoop, AsyncHTTPClient]

    @classmethod
    def configurable_base(cls) -> Type[Configurable]:
        return AsyncHTTPClient

    @classmethod
    def configurable_default(cls) -> Type[Configurable]:
        from tornado.simple_httpclient import SimpleAsyncHTTPClient

        return SimpleAsyncHTTPClient

    @classmethod
    def _async_clients(cls) -> Dict[IOLoop, "AsyncHTTPClient"]:
        attr_name = "_async_client_dict_" + cls.__name__
        if not hasattr(cls, attr_name):
            setattr(cls, attr_name, weakref.WeakKeyDictionary())
        return getattr(cls, attr_name)

    def __new__(cls, force_instance: bool = False, **kwargs: Any) -> "AsyncHTTPClient":
        io_loop = IOLoop.current()
        if force_instance:
            instance_cache = None
        else:
            instance_cache = cls._async_clients()
        if instance_cache is not None and io_loop in instance_cache:
            return instance_cache[io_loop]
        instance = super(AsyncHTTPClient, cls).__new__(cls, **kwargs)  # type: ignore
        # Make sure the instance knows which cache to remove itself from.
        # It can't simply call _async_clients() because we may be in
        # __new__(AsyncHTTPClient) but instance.__class__ may be
        # SimpleAsyncHTTPClient.
        instance._instance_cache = instance_cache
        if instance_cache is not None:
            instance_cache[instance.io_loop] = instance
        return instance

    def initialize(self, defaults: Optional[Dict[str, Any]] = None) -> None:
        self.io_loop = IOLoop.current()
        self.defaults = dict(HTTPRequest._DEFAULTS)
        if defaults is not None:
            self.defaults.update(defaults)
        self._closed = False

    def close(self) -> None:
        """Destroys this HTTP client, freeing any file descriptors used.

        This method is **not needed in normal use** due to the way
        that `AsyncHTTPClient` objects are transparently reused.
        ``close()`` is generally only necessary when either the
        `.IOLoop` is also being closed, or the ``force_instance=True``
        argument was used when creating the `AsyncHTTPClient`.

        No other methods may be called on the `AsyncHTTPClient` after
        ``close()``.

        """
        if self._closed:
            return
        self._closed = True
        if self._instance_cache is not None:
            cached_val = self._instance_cache.pop(self.io_loop, None)
            # If there's an object other than self in the instance
            # cache for our IOLoop, something has gotten mixed up. A
            # value of None appears to be possible when this is called
            # from a destructor (HTTPClient.__del__) as the weakref
            # gets cleared before the destructor runs.
            if cached_val is not None and cached_val is not self:
                raise RuntimeError("inconsistent AsyncHTTPClient cache")

    def fetch(
        self,
        request: Union[str, "HTTPRequest"],
        raise_error: bool = True,
        **kwargs: Any
    ) -> "Future[HTTPResponse]":
        """Executes a request, asynchronously returning an `HTTPResponse`.

        The request may be either a string URL or an `HTTPRequest` object.
        If it is a string, we construct an `HTTPRequest` using any additional
        kwargs: ``HTTPRequest(request, **kwargs)``

        This method returns a `.Future` whose result is an
        `HTTPResponse`. By default, the ``Future`` will raise an
        `HTTPError` if the request returned a non-200 response code
        (other errors may also be raised if the server could not be
        contacted). Instead, if ``raise_error`` is set to False, the
        response will always be returned regardless of the response
        code.

        If a ``callback`` is given, it will be invoked with the `HTTPResponse`.
        In the callback interface, `HTTPError` is not automatically raised.
        Instead, you must check the response's ``error`` attribute or
        call its `~HTTPResponse.rethrow` method.

        .. versionchanged:: 6.0

           The ``callback`` argument was removed. Use the returned
           `.Future` instead.

           The ``raise_error=False`` argument only affects the
           `HTTPError` raised when a non-200 response code is used,
           instead of suppressing all errors.
        """
        if self._closed:
            raise RuntimeError("fetch() called on closed AsyncHTTPClient")
        if not isinstance(request, HTTPRequest):
            request = HTTPRequest(url=request, **kwargs)
        else:
            if kwargs:
                raise ValueError(
                    "kwargs can't be used if request is an HTTPRequest object"
                )
        # We may modify this (to add Host, Accept-Encoding, etc),
        # so make sure we don't modify the caller's object.  This is also
        # where normal dicts get converted to HTTPHeaders objects.
        request.headers = httputil.HTTPHeaders(request.headers)
        request_proxy = _RequestProxy(request, self.defaults)
        future = Future()  # type: Future[HTTPResponse]

        def handle_response(response: "HTTPResponse") -> None:
            if response.error:
                if raise_error or not response._error_is_response_code:
                    future_set_exception_unless_cancelled(future, response.error)
                    return
            future_set_result_unless_cancelled(future, response)

        self.fetch_impl(cast(HTTPRequest, request_proxy), handle_response)
        return future

    def fetch_impl(
        self, request: "HTTPRequest", callback: Callable[["HTTPResponse"], None]
    ) -> None:
        raise NotImplementedError()

    @classmethod
    def configure(
        cls, impl: "Union[None, str, Type[Configurable]]", **kwargs: Any
    ) -> None:
        """Configures the `AsyncHTTPClient` subclass to use.

        ``AsyncHTTPClient()`` actually creates an instance of a subclass.
        This method may be called with either a class object or the
        fully-qualified name of such a class (or ``None`` to use the default,
        ``SimpleAsyncHTTPClient``)

        If additional keyword arguments are given, they will be passed
        to the constructor of each subclass instance created.  The
        keyword argument ``max_clients`` determines the maximum number
        of simultaneous `~AsyncHTTPClient.fetch()` operations that can
        execute in parallel on each `.IOLoop`.  Additional arguments
        may be supported depending on the implementation class in use.

        Example::

           AsyncHTTPClient.configure("tornado.curl_httpclient.CurlAsyncHTTPClient")
        """
        super(AsyncHTTPClient, cls).configure(impl, **kwargs)


class HTTPRequest(object):
    """HTTP client request object."""

    _headers = None  # type: Union[Dict[str, str], httputil.HTTPHeaders]

    # Default values for HTTPRequest parameters.
    # Merged with the values on the request object by AsyncHTTPClient
    # implementations.
    _DEFAULTS = dict(
        connect_timeout=20.0,
        request_timeout=20.0,
        follow_redirects=True,
        max_redirects=5,
        decompress_response=True,
        proxy_password="",
        allow_nonstandard_methods=False,
        validate_cert=True,
    )

    def __init__(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[Union[Dict[str, str], httputil.HTTPHeaders]] = None,
        body: Optional[Union[bytes, str]] = None,
        auth_username: Optional[str] = None,
        auth_password: Optional[str] = None,
        auth_mode: Optional[str] = None,
        connect_timeout: Optional[float] = None,
        request_timeout: Optional[float] = None,
        if_modified_since: Optional[Union[float, datetime.datetime]] = None,
        follow_redirects: Optional[bool] = None,
        max_redirects: Optional[int] = None,
        user_agent: Optional[str] = None,
        use_gzip: Optional[bool] = None,
        network_interface: Optional[str] = None,
        streaming_callback: Optional[Callable[[bytes], None]] = None,
        header_callback: Optional[Callable[[str], None]] = None,
        prepare_curl_callback: Optional[Callable[[Any], None]] = None,
        proxy_host: Optional[str] = None,
        proxy_port: Optional[int] = None,
        proxy_username: Optional[str] = None,
        proxy_password: Optional[str] = None,
        proxy_auth_mode: Optional[str] = None,
        allow_nonstandard_methods: Optional[bool] = None,
        validate_cert: Optional[bool] = None,
        ca_certs: Optional[str] = None,
        allow_ipv6: Optional[bool] = None,
        client_key: Optional[str] = None,
        client_cert: Optional[str] = None,
        body_producer: Optional[
            Callable[[Callable[[bytes], None]], "Future[None]"]
        ] = None,
        expect_100_continue: bool = False,
        decompress_response: Optional[bool] = None,
        ssl_options: Optional[Union[Dict[str, Any], ssl.SSLContext]] = None,
    ) -> None:
        r"""All parameters except ``url`` are optional.

        :arg str url: URL to fetch
        :arg str method: HTTP method, e.g. "GET" or "POST"
        :arg headers: Additional HTTP headers to pass on the request
        :type headers: `~tornado.httputil.HTTPHeaders` or `dict`
        :arg body: HTTP request body as a string (byte or unicode; if unicode
           the utf-8 encoding will be used)
        :type body: `str` or `bytes`
        :arg collections.abc.Callable body_producer: Callable used for
           lazy/asynchronous request bodies.
           It is called with one argument, a ``write`` function, and should
           return a `.Future`.  It should call the write function with new
           data as it becomes available.  The write function returns a
           `.Future` which can be used for flow control.
           Only one of ``body`` and ``body_producer`` may
           be specified.  ``body_producer`` is not supported on
           ``curl_httpclient``.  When using ``body_producer`` it is recommended
           to pass a ``Content-Length`` in the headers as otherwise chunked
           encoding will be used, and many servers do not support chunked
           encoding on requests.  New in Tornado 4.0
        :arg str auth_username: Username for HTTP authentication
        :arg str auth_password: Password for HTTP authentication
        :arg str auth_mode: Authentication mode; default is "basic".
           Allowed values are implementation-defined; ``curl_httpclient``
           supports "basic" and "digest"; ``simple_httpclient`` only supports
           "basic"
        :arg float connect_timeout: Timeout for initial connection in seconds,
           default 20 seconds (0 means no timeout)
        :arg float request_timeout: Timeout for entire request in seconds,
           default 20 seconds (0 means no timeout)
        :arg if_modified_since: Timestamp for ``If-Modified-Since`` header
        :type if_modified_since: `datetime` or `float`
        :arg bool follow_redirects: Should redirects be followed automatically
           or return the 3xx response? Default True.
        :arg int max_redirects: Limit for ``follow_redirects``, default 5.
        :arg str user_agent: String to send as ``User-Agent`` header
        :arg bool decompress_response: Request a compressed response from
           the server and decompress it after downloading.  Default is True.
           New in Tornado 4.0.
        :arg bool use_gzip: Deprecated alias for ``decompress_response``
           since Tornado 4.0.
        :arg str network_interface: Network interface or source IP to use for request.
           See ``curl_httpclient`` note below.
        :arg collections.abc.Callable streaming_callback: If set, ``streaming_callback`` will
           be run with each chunk of data as it is received, and
           ``HTTPResponse.body`` and ``HTTPResponse.buffer`` will be empty in
           the final response.
        :arg collections.abc.Callable header_callback: If set, ``header_callback`` will
           be run with each header line as it is received (including the
           first line, e.g. ``HTTP/1.0 200 OK\r\n``, and a final line
           containing only ``\r\n``.  All lines include the trailing newline
           characters).  ``HTTPResponse.headers`` will be empty in the final
           response.  This is most useful in conjunction with
           ``streaming_callback``, because it's the only way to get access to
           header data while the request is in progress.
        :arg collections.abc.Callable prepare_curl_callback: If set, will be called with
           a ``pycurl.Curl`` object to allow the application to make additional
           ``setopt`` calls.
        :arg str proxy_host: HTTP proxy hostname.  To use proxies,
           ``proxy_host`` and ``proxy_port`` must be set; ``proxy_username``,
           ``proxy_pass`` and ``proxy_auth_mode`` are optional.  Proxies are
           currently only supported with ``curl_httpclient``.
        :arg int proxy_port: HTTP proxy port
        :arg str proxy_username: HTTP proxy username
        :arg str proxy_password: HTTP proxy password
        :arg str proxy_auth_mode: HTTP proxy Authentication mode;
           default is "basic". supports "basic" and "digest"
        :arg bool allow_nonstandard_methods: Allow unknown values for ``method``
           argument? Default is False.
        :arg bool validate_cert: For HTTPS requests, validate the server's
           certificate? Default is True.
        :arg str ca_certs: filename of CA certificates in PEM format,
           or None to use defaults.  See note below when used with
           ``curl_httpclient``.
        :arg str client_key: Filename for client SSL key, if any.  See
           note below when used with ``curl_httpclient``.
        :arg str client_cert: Filename for client SSL certificate, if any.
           See note below when used with ``curl_httpclient``.
        :arg ssl.SSLContext ssl_options: `ssl.SSLContext` object for use in
           ``simple_httpclient`` (unsupported by ``curl_httpclient``).
           Overrides ``validate_cert``, ``ca_certs``, ``client_key``,
           and ``client_cert``.
        :arg bool allow_ipv6: Use IPv6 when available?  Default is True.
        :arg bool expect_100_continue: If true, send the
           ``Expect: 100-continue`` header and wait for a continue response
           before sending the request body.  Only supported with
           ``simple_httpclient``.

        .. note::

            When using ``curl_httpclient`` certain options may be
            inherited by subsequent fetches because ``pycurl`` does
            not allow them to be cleanly reset.  This applies to the
            ``ca_certs``, ``client_key``, ``client_cert``, and
            ``network_interface`` arguments.  If you use these
            options, you should pass them on every request (you don't
            have to always use the same values, but it's not possible
            to mix requests that specify these options with ones that
            use the defaults).

        .. versionadded:: 3.1
           The ``auth_mode`` argument.

        .. versionadded:: 4.0
           The ``body_producer`` and ``expect_100_continue`` arguments.

        .. versionadded:: 4.2
           The ``ssl_options`` argument.

        .. versionadded:: 4.5
           The ``proxy_auth_mode`` argument.
        """
        # Note that some of these attributes go through property setters
        # defined below.
        self.headers = headers  # type: ignore
        if if_modified_since:
            self.headers["If-Modified-Since"] = httputil.format_timestamp(
                if_modified_since
            )
        self.proxy_host = proxy_host
        self.proxy_port = proxy_port
        self.proxy_username = proxy_username
        self.proxy_password = proxy_password
        self.proxy_auth_mode = proxy_auth_mode
        self.url = url
        self.method = method
        self.body = body  # type: ignore
        self.body_producer = body_producer
        self.auth_username = auth_username
        self.auth_password = auth_password
        self.auth_mode = auth_mode
        self.connect_timeout = connect_timeout
        self.request_timeout = request_timeout
        self.follow_redirects = follow_redirects
        self.max_redirects = max_redirects
        self.user_agent = user_agent
        if decompress_response is not None:
            self.decompress_response = decompress_response  # type: Optional[bool]
        else:
            self.decompress_response = use_gzip
        self.network_interface = network_interface
        self.streaming_callback = streaming_callback
        self.header_callback = header_callback
        self.prepare_curl_callback = prepare_curl_callback
        self.allow_nonstandard_methods = allow_nonstandard_methods
        self.validate_cert = validate_cert
        self.ca_certs = ca_certs
        self.allow_ipv6 = allow_ipv6
        self.client_key = client_key
        self.client_cert = client_cert
        self.ssl_options = ssl_options
        self.expect_100_continue = expect_100_continue
        self.start_time = time.time()

    @property
    def headers(self) -> httputil.HTTPHeaders:
        # TODO: headers may actually be a plain dict until fairly late in
        # the process (AsyncHTTPClient.fetch), but practically speaking,
        # whenever the property is used they're already HTTPHeaders.
        return self._headers  # type: ignore

    @headers.setter
    def headers(self, value: Union[Dict[str, str], httputil.HTTPHeaders]) -> None:
        if value is None:
            self._headers = httputil.HTTPHeaders()
        else:
            self._headers = value  # type: ignore

    @property
    def body(self) -> bytes:
        return self._body

    @body.setter
    def body(self, value: Union[bytes, str]) -> None:
        self._body = utf8(value)


class HTTPResponse(object):
    """HTTP Response object.

    Attributes:

    * ``request``: HTTPRequest object

    * ``code``: numeric HTTP status code, e.g. 200 or 404

    * ``reason``: human-readable reason phrase describing the status code

    * ``headers``: `tornado.httputil.HTTPHeaders` object

    * ``effective_url``: final location of the resource after following any
      redirects

    * ``buffer``: ``cStringIO`` object for response body

    * ``body``: response body as bytes (created on demand from ``self.buffer``)

    * ``error``: Exception object, if any

    * ``request_time``: seconds from request start to finish. Includes all
      network operations from DNS resolution to receiving the last byte of
      data. Does not include time spent in the queue (due to the
      ``max_clients`` option). If redirects were followed, only includes
      the final request.

    * ``start_time``: Time at which the HTTP operation started, based on
      `time.time` (not the monotonic clock used by `.IOLoop.time`). May
      be ``None`` if the request timed out while in the queue.

    * ``time_info``: dictionary of diagnostic timing information from the
      request. Available data are subject to change, but currently uses timings
      available from http://curl.haxx.se/libcurl/c/curl_easy_getinfo.html,
      plus ``queue``, which is the delay (if any) introduced by waiting for
      a slot under `AsyncHTTPClient`'s ``max_clients`` setting.

    .. versionadded:: 5.1

       Added the ``start_time`` attribute.

    .. versionchanged:: 5.1

       The ``request_time`` attribute previously included time spent in the queue
       for ``simple_httpclient``, but not in ``curl_httpclient``. Now queueing time
       is excluded in both implementations. ``request_time`` is now more accurate for
       ``curl_httpclient`` because it uses a monotonic clock when available.
    """

    # I'm not sure why these don't get type-inferred from the references in __init__.
    error = None  # type: Optional[BaseException]
    _error_is_response_code = False
    request = None  # type: HTTPRequest

    def __init__(
        self,
        request: HTTPRequest,
        code: int,
        headers: Optional[httputil.HTTPHeaders] = None,
        buffer: Optional[BytesIO] = None,
        effective_url: Optional[str] = None,
        error: Optional[BaseException] = None,
        request_time: Optional[float] = None,
        time_info: Optional[Dict[str, float]] = None,
        reason: Optional[str] = None,
        start_time: Optional[float] = None,
    ) -> None:
        if isinstance(request, _RequestProxy):
            self.request = request.request
        else:
            self.request = request
        self.code = code
        self.reason = reason or httputil.responses.get(code, "Unknown")
        if headers is not None:
            self.headers = headers
        else:
            self.headers = httputil.HTTPHeaders()
        self.buffer = buffer
        self._body = None  # type: Optional[bytes]
        if effective_url is None:
            self.effective_url = request.url
        else:
            self.effective_url = effective_url
        self._error_is_response_code = False
        if error is None:
            if self.code < 200 or self.code >= 300:
                self._error_is_response_code = True
                self.error = HTTPError(self.code, message=self.reason, response=self)
            else:
                self.error = None
        else:
            self.error = error
        self.start_time = start_time
        self.request_time = request_time
        self.time_info = time_info or {}

    @property
    def body(self) -> bytes:
        if self.buffer is None:
            return b""
        elif self._body is None:
            self._body = self.buffer.getvalue()

        return self._body

    def rethrow(self) -> None:
        """If there was an error on the request, raise an `HTTPError`."""
        if self.error:
            raise self.error

    def __repr__(self) -> str:
        args = ",".join("%s=%r" % i for i in sorted(self.__dict__.items()))
        return "%s(%s)" % (self.__class__.__name__, args)


class HTTPClientError(Exception):
    """Exception thrown for an unsuccessful HTTP request.

    Attributes:

    * ``code`` - HTTP error integer error code, e.g. 404.  Error code 599 is
      used when no HTTP response was received, e.g. for a timeout.

    * ``response`` - `HTTPResponse` object, if any.

    Note that if ``follow_redirects`` is False, redirects become HTTPErrors,
    and you can look at ``error.response.headers['Location']`` to see the
    destination of the redirect.

    .. versionchanged:: 5.1

       Renamed from ``HTTPError`` to ``HTTPClientError`` to avoid collisions with
       `tornado.web.HTTPError`. The name ``tornado.httpclient.HTTPError`` remains
       as an alias.
    """

    def __init__(
        self,
        code: int,
        message: Optional[str] = None,
        response: Optional[HTTPResponse] = None,
    ) -> None:
        self.code = code
        self.message = message or httputil.responses.get(code, "Unknown")
        self.response = response
        super().__init__(code, message, response)

    def __str__(self) -> str:
        return "HTTP %d: %s" % (self.code, self.message)

    # There is a cyclic reference between self and self.response,
    # which breaks the default __repr__ implementation.
    # (especially on pypy, which doesn't have the same recursion
    # detection as cpython).
    __repr__ = __str__


HTTPError = HTTPClientError


class _RequestProxy(object):
    """Combines an object with a dictionary of defaults.

    Used internally by AsyncHTTPClient implementations.
    """

    def __init__(
        self, request: HTTPRequest, defaults: Optional[Dict[str, Any]]
    ) -> None:
        self.request = request
        self.defaults = defaults

    def __getattr__(self, name: str) -> Any:
        request_attr = getattr(self.request, name)
        if request_attr is not None:
            return request_attr
        elif self.defaults is not None:
            return self.defaults.get(name, None)
        else:
            return None


def main() -> None:
    from tornado.options import define, options, parse_command_line

    define("print_headers", type=bool, default=False)
    define("print_body", type=bool, default=True)
    define("follow_redirects", type=bool, default=True)
    define("validate_cert", type=bool, default=True)
    define("proxy_host", type=str)
    define("proxy_port", type=int)
    args = parse_command_line()
    client = HTTPClient()
    for arg in args:
        try:
            response = client.fetch(
                arg,
                follow_redirects=options.follow_redirects,
                validate_cert=options.validate_cert,
                proxy_host=options.proxy_host,
                proxy_port=options.proxy_port,
            )
        except HTTPError as e:
            if e.response is not None:
                response = e.response
            else:
                raise
        if options.print_headers:
            print(response.headers)
        if options.print_body:
            print(native_str(response.body))
    client.close()


if __name__ == "__main__":
    main()
