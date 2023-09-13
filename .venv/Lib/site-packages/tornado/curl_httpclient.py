#
# Copyright 2009 Facebook
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

"""Non-blocking HTTP client implementation using pycurl."""

import collections
import functools
import logging
import pycurl
import threading
import time
from io import BytesIO

from tornado import httputil
from tornado import ioloop

from tornado.escape import utf8, native_str
from tornado.httpclient import (
    HTTPRequest,
    HTTPResponse,
    HTTPError,
    AsyncHTTPClient,
    main,
)
from tornado.log import app_log

from typing import Dict, Any, Callable, Union, Optional
import typing

if typing.TYPE_CHECKING:
    from typing import Deque, Tuple  # noqa: F401

curl_log = logging.getLogger("tornado.curl_httpclient")


class CurlAsyncHTTPClient(AsyncHTTPClient):
    def initialize(  # type: ignore
        self, max_clients: int = 10, defaults: Optional[Dict[str, Any]] = None
    ) -> None:
        super().initialize(defaults=defaults)
        # Typeshed is incomplete for CurlMulti, so just use Any for now.
        self._multi = pycurl.CurlMulti()  # type: Any
        self._multi.setopt(pycurl.M_TIMERFUNCTION, self._set_timeout)
        self._multi.setopt(pycurl.M_SOCKETFUNCTION, self._handle_socket)
        self._curls = [self._curl_create() for i in range(max_clients)]
        self._free_list = self._curls[:]
        self._requests = (
            collections.deque()
        )  # type: Deque[Tuple[HTTPRequest, Callable[[HTTPResponse], None], float]]
        self._fds = {}  # type: Dict[int, int]
        self._timeout = None  # type: Optional[object]

        # libcurl has bugs that sometimes cause it to not report all
        # relevant file descriptors and timeouts to TIMERFUNCTION/
        # SOCKETFUNCTION.  Mitigate the effects of such bugs by
        # forcing a periodic scan of all active requests.
        self._force_timeout_callback = ioloop.PeriodicCallback(
            self._handle_force_timeout, 1000
        )
        self._force_timeout_callback.start()

        # Work around a bug in libcurl 7.29.0: Some fields in the curl
        # multi object are initialized lazily, and its destructor will
        # segfault if it is destroyed without having been used.  Add
        # and remove a dummy handle to make sure everything is
        # initialized.
        dummy_curl_handle = pycurl.Curl()
        self._multi.add_handle(dummy_curl_handle)
        self._multi.remove_handle(dummy_curl_handle)

    def close(self) -> None:
        self._force_timeout_callback.stop()
        if self._timeout is not None:
            self.io_loop.remove_timeout(self._timeout)
        for curl in self._curls:
            curl.close()
        self._multi.close()
        super().close()

        # Set below properties to None to reduce the reference count of current
        # instance, because those properties hold some methods of current
        # instance that will case circular reference.
        self._force_timeout_callback = None  # type: ignore
        self._multi = None

    def fetch_impl(
        self, request: HTTPRequest, callback: Callable[[HTTPResponse], None]
    ) -> None:
        self._requests.append((request, callback, self.io_loop.time()))
        self._process_queue()
        self._set_timeout(0)

    def _handle_socket(self, event: int, fd: int, multi: Any, data: bytes) -> None:
        """Called by libcurl when it wants to change the file descriptors
        it cares about.
        """
        event_map = {
            pycurl.POLL_NONE: ioloop.IOLoop.NONE,
            pycurl.POLL_IN: ioloop.IOLoop.READ,
            pycurl.POLL_OUT: ioloop.IOLoop.WRITE,
            pycurl.POLL_INOUT: ioloop.IOLoop.READ | ioloop.IOLoop.WRITE,
        }
        if event == pycurl.POLL_REMOVE:
            if fd in self._fds:
                self.io_loop.remove_handler(fd)
                del self._fds[fd]
        else:
            ioloop_event = event_map[event]
            # libcurl sometimes closes a socket and then opens a new
            # one using the same FD without giving us a POLL_NONE in
            # between.  This is a problem with the epoll IOLoop,
            # because the kernel can tell when a socket is closed and
            # removes it from the epoll automatically, causing future
            # update_handler calls to fail.  Since we can't tell when
            # this has happened, always use remove and re-add
            # instead of update.
            if fd in self._fds:
                self.io_loop.remove_handler(fd)
            self.io_loop.add_handler(fd, self._handle_events, ioloop_event)
            self._fds[fd] = ioloop_event

    def _set_timeout(self, msecs: int) -> None:
        """Called by libcurl to schedule a timeout."""
        if self._timeout is not None:
            self.io_loop.remove_timeout(self._timeout)
        self._timeout = self.io_loop.add_timeout(
            self.io_loop.time() + msecs / 1000.0, self._handle_timeout
        )

    def _handle_events(self, fd: int, events: int) -> None:
        """Called by IOLoop when there is activity on one of our
        file descriptors.
        """
        action = 0
        if events & ioloop.IOLoop.READ:
            action |= pycurl.CSELECT_IN
        if events & ioloop.IOLoop.WRITE:
            action |= pycurl.CSELECT_OUT
        while True:
            try:
                ret, num_handles = self._multi.socket_action(fd, action)
            except pycurl.error as e:
                ret = e.args[0]
            if ret != pycurl.E_CALL_MULTI_PERFORM:
                break
        self._finish_pending_requests()

    def _handle_timeout(self) -> None:
        """Called by IOLoop when the requested timeout has passed."""
        self._timeout = None
        while True:
            try:
                ret, num_handles = self._multi.socket_action(pycurl.SOCKET_TIMEOUT, 0)
            except pycurl.error as e:
                ret = e.args[0]
            if ret != pycurl.E_CALL_MULTI_PERFORM:
                break
        self._finish_pending_requests()

        # In theory, we shouldn't have to do this because curl will
        # call _set_timeout whenever the timeout changes.  However,
        # sometimes after _handle_timeout we will need to reschedule
        # immediately even though nothing has changed from curl's
        # perspective.  This is because when socket_action is
        # called with SOCKET_TIMEOUT, libcurl decides internally which
        # timeouts need to be processed by using a monotonic clock
        # (where available) while tornado uses python's time.time()
        # to decide when timeouts have occurred.  When those clocks
        # disagree on elapsed time (as they will whenever there is an
        # NTP adjustment), tornado might call _handle_timeout before
        # libcurl is ready.  After each timeout, resync the scheduled
        # timeout with libcurl's current state.
        new_timeout = self._multi.timeout()
        if new_timeout >= 0:
            self._set_timeout(new_timeout)

    def _handle_force_timeout(self) -> None:
        """Called by IOLoop periodically to ask libcurl to process any
        events it may have forgotten about.
        """
        while True:
            try:
                ret, num_handles = self._multi.socket_all()
            except pycurl.error as e:
                ret = e.args[0]
            if ret != pycurl.E_CALL_MULTI_PERFORM:
                break
        self._finish_pending_requests()

    def _finish_pending_requests(self) -> None:
        """Process any requests that were completed by the last
        call to multi.socket_action.
        """
        while True:
            num_q, ok_list, err_list = self._multi.info_read()
            for curl in ok_list:
                self._finish(curl)
            for curl, errnum, errmsg in err_list:
                self._finish(curl, errnum, errmsg)
            if num_q == 0:
                break
        self._process_queue()

    def _process_queue(self) -> None:
        while True:
            started = 0
            while self._free_list and self._requests:
                started += 1
                curl = self._free_list.pop()
                (request, callback, queue_start_time) = self._requests.popleft()
                # TODO: Don't smuggle extra data on an attribute of the Curl object.
                curl.info = {  # type: ignore
                    "headers": httputil.HTTPHeaders(),
                    "buffer": BytesIO(),
                    "request": request,
                    "callback": callback,
                    "queue_start_time": queue_start_time,
                    "curl_start_time": time.time(),
                    "curl_start_ioloop_time": self.io_loop.current().time(),  # type: ignore
                }
                try:
                    self._curl_setup_request(
                        curl,
                        request,
                        curl.info["buffer"],  # type: ignore
                        curl.info["headers"],  # type: ignore
                    )
                except Exception as e:
                    # If there was an error in setup, pass it on
                    # to the callback. Note that allowing the
                    # error to escape here will appear to work
                    # most of the time since we are still in the
                    # caller's original stack frame, but when
                    # _process_queue() is called from
                    # _finish_pending_requests the exceptions have
                    # nowhere to go.
                    self._free_list.append(curl)
                    callback(HTTPResponse(request=request, code=599, error=e))
                else:
                    self._multi.add_handle(curl)

            if not started:
                break

    def _finish(
        self,
        curl: pycurl.Curl,
        curl_error: Optional[int] = None,
        curl_message: Optional[str] = None,
    ) -> None:
        info = curl.info  # type: ignore
        curl.info = None  # type: ignore
        self._multi.remove_handle(curl)
        self._free_list.append(curl)
        buffer = info["buffer"]
        if curl_error:
            assert curl_message is not None
            error = CurlError(curl_error, curl_message)  # type: Optional[CurlError]
            assert error is not None
            code = error.code
            effective_url = None
            buffer.close()
            buffer = None
        else:
            error = None
            code = curl.getinfo(pycurl.HTTP_CODE)
            effective_url = curl.getinfo(pycurl.EFFECTIVE_URL)
            buffer.seek(0)
        # the various curl timings are documented at
        # http://curl.haxx.se/libcurl/c/curl_easy_getinfo.html
        time_info = dict(
            queue=info["curl_start_ioloop_time"] - info["queue_start_time"],
            namelookup=curl.getinfo(pycurl.NAMELOOKUP_TIME),
            connect=curl.getinfo(pycurl.CONNECT_TIME),
            appconnect=curl.getinfo(pycurl.APPCONNECT_TIME),
            pretransfer=curl.getinfo(pycurl.PRETRANSFER_TIME),
            starttransfer=curl.getinfo(pycurl.STARTTRANSFER_TIME),
            total=curl.getinfo(pycurl.TOTAL_TIME),
            redirect=curl.getinfo(pycurl.REDIRECT_TIME),
        )
        try:
            info["callback"](
                HTTPResponse(
                    request=info["request"],
                    code=code,
                    headers=info["headers"],
                    buffer=buffer,
                    effective_url=effective_url,
                    error=error,
                    reason=info["headers"].get("X-Http-Reason", None),
                    request_time=self.io_loop.time() - info["curl_start_ioloop_time"],
                    start_time=info["curl_start_time"],
                    time_info=time_info,
                )
            )
        except Exception:
            self.handle_callback_exception(info["callback"])

    def handle_callback_exception(self, callback: Any) -> None:
        app_log.error("Exception in callback %r", callback, exc_info=True)

    def _curl_create(self) -> pycurl.Curl:
        curl = pycurl.Curl()
        if curl_log.isEnabledFor(logging.DEBUG):
            curl.setopt(pycurl.VERBOSE, 1)
            curl.setopt(pycurl.DEBUGFUNCTION, self._curl_debug)
        if hasattr(
            pycurl, "PROTOCOLS"
        ):  # PROTOCOLS first appeared in pycurl 7.19.5 (2014-07-12)
            curl.setopt(pycurl.PROTOCOLS, pycurl.PROTO_HTTP | pycurl.PROTO_HTTPS)
            curl.setopt(pycurl.REDIR_PROTOCOLS, pycurl.PROTO_HTTP | pycurl.PROTO_HTTPS)
        return curl

    def _curl_setup_request(
        self,
        curl: pycurl.Curl,
        request: HTTPRequest,
        buffer: BytesIO,
        headers: httputil.HTTPHeaders,
    ) -> None:
        curl.setopt(pycurl.URL, native_str(request.url))

        # libcurl's magic "Expect: 100-continue" behavior causes delays
        # with servers that don't support it (which include, among others,
        # Google's OpenID endpoint).  Additionally, this behavior has
        # a bug in conjunction with the curl_multi_socket_action API
        # (https://sourceforge.net/tracker/?func=detail&atid=100976&aid=3039744&group_id=976),
        # which increases the delays.  It's more trouble than it's worth,
        # so just turn off the feature (yes, setting Expect: to an empty
        # value is the official way to disable this)
        if "Expect" not in request.headers:
            request.headers["Expect"] = ""

        # libcurl adds Pragma: no-cache by default; disable that too
        if "Pragma" not in request.headers:
            request.headers["Pragma"] = ""

        curl.setopt(
            pycurl.HTTPHEADER,
            [
                b"%s: %s"
                % (native_str(k).encode("ASCII"), native_str(v).encode("ISO8859-1"))
                for k, v in request.headers.get_all()
            ],
        )

        curl.setopt(
            pycurl.HEADERFUNCTION,
            functools.partial(
                self._curl_header_callback, headers, request.header_callback
            ),
        )
        if request.streaming_callback:

            def write_function(b: Union[bytes, bytearray]) -> int:
                assert request.streaming_callback is not None
                self.io_loop.add_callback(request.streaming_callback, b)
                return len(b)

        else:
            write_function = buffer.write  # type: ignore
        curl.setopt(pycurl.WRITEFUNCTION, write_function)
        curl.setopt(pycurl.FOLLOWLOCATION, request.follow_redirects)
        curl.setopt(pycurl.MAXREDIRS, request.max_redirects)
        assert request.connect_timeout is not None
        curl.setopt(pycurl.CONNECTTIMEOUT_MS, int(1000 * request.connect_timeout))
        assert request.request_timeout is not None
        curl.setopt(pycurl.TIMEOUT_MS, int(1000 * request.request_timeout))
        if request.user_agent:
            curl.setopt(pycurl.USERAGENT, native_str(request.user_agent))
        else:
            curl.setopt(pycurl.USERAGENT, "Mozilla/5.0 (compatible; pycurl)")
        if request.network_interface:
            curl.setopt(pycurl.INTERFACE, request.network_interface)
        if request.decompress_response:
            curl.setopt(pycurl.ENCODING, "gzip,deflate")
        else:
            curl.setopt(pycurl.ENCODING, None)
        if request.proxy_host and request.proxy_port:
            curl.setopt(pycurl.PROXY, request.proxy_host)
            curl.setopt(pycurl.PROXYPORT, request.proxy_port)
            if request.proxy_username:
                assert request.proxy_password is not None
                credentials = httputil.encode_username_password(
                    request.proxy_username, request.proxy_password
                )
                curl.setopt(pycurl.PROXYUSERPWD, credentials)

            if request.proxy_auth_mode is None or request.proxy_auth_mode == "basic":
                curl.setopt(pycurl.PROXYAUTH, pycurl.HTTPAUTH_BASIC)
            elif request.proxy_auth_mode == "digest":
                curl.setopt(pycurl.PROXYAUTH, pycurl.HTTPAUTH_DIGEST)
            else:
                raise ValueError(
                    "Unsupported proxy_auth_mode %s" % request.proxy_auth_mode
                )
        else:
            try:
                curl.unsetopt(pycurl.PROXY)
            except TypeError:  # not supported, disable proxy
                curl.setopt(pycurl.PROXY, "")
            curl.unsetopt(pycurl.PROXYUSERPWD)
        if request.validate_cert:
            curl.setopt(pycurl.SSL_VERIFYPEER, 1)
            curl.setopt(pycurl.SSL_VERIFYHOST, 2)
        else:
            curl.setopt(pycurl.SSL_VERIFYPEER, 0)
            curl.setopt(pycurl.SSL_VERIFYHOST, 0)
        if request.ca_certs is not None:
            curl.setopt(pycurl.CAINFO, request.ca_certs)
        else:
            # There is no way to restore pycurl.CAINFO to its default value
            # (Using unsetopt makes it reject all certificates).
            # I don't see any way to read the default value from python so it
            # can be restored later.  We'll have to just leave CAINFO untouched
            # if no ca_certs file was specified, and require that if any
            # request uses a custom ca_certs file, they all must.
            pass

        if request.allow_ipv6 is False:
            # Curl behaves reasonably when DNS resolution gives an ipv6 address
            # that we can't reach, so allow ipv6 unless the user asks to disable.
            curl.setopt(pycurl.IPRESOLVE, pycurl.IPRESOLVE_V4)
        else:
            curl.setopt(pycurl.IPRESOLVE, pycurl.IPRESOLVE_WHATEVER)

        # Set the request method through curl's irritating interface which makes
        # up names for almost every single method
        curl_options = {
            "GET": pycurl.HTTPGET,
            "POST": pycurl.POST,
            "PUT": pycurl.UPLOAD,
            "HEAD": pycurl.NOBODY,
        }
        custom_methods = set(["DELETE", "OPTIONS", "PATCH"])
        for o in curl_options.values():
            curl.setopt(o, False)
        if request.method in curl_options:
            curl.unsetopt(pycurl.CUSTOMREQUEST)
            curl.setopt(curl_options[request.method], True)
        elif request.allow_nonstandard_methods or request.method in custom_methods:
            curl.setopt(pycurl.CUSTOMREQUEST, request.method)
        else:
            raise KeyError("unknown method " + request.method)

        body_expected = request.method in ("POST", "PATCH", "PUT")
        body_present = request.body is not None
        if not request.allow_nonstandard_methods:
            # Some HTTP methods nearly always have bodies while others
            # almost never do. Fail in this case unless the user has
            # opted out of sanity checks with allow_nonstandard_methods.
            if (body_expected and not body_present) or (
                body_present and not body_expected
            ):
                raise ValueError(
                    "Body must %sbe None for method %s (unless "
                    "allow_nonstandard_methods is true)"
                    % ("not " if body_expected else "", request.method)
                )

        if body_expected or body_present:
            if request.method == "GET":
                # Even with `allow_nonstandard_methods` we disallow
                # GET with a body (because libcurl doesn't allow it
                # unless we use CUSTOMREQUEST). While the spec doesn't
                # forbid clients from sending a body, it arguably
                # disallows the server from doing anything with them.
                raise ValueError("Body must be None for GET request")
            request_buffer = BytesIO(utf8(request.body or ""))

            def ioctl(cmd: int) -> None:
                if cmd == curl.IOCMD_RESTARTREAD:  # type: ignore
                    request_buffer.seek(0)

            curl.setopt(pycurl.READFUNCTION, request_buffer.read)
            curl.setopt(pycurl.IOCTLFUNCTION, ioctl)
            if request.method == "POST":
                curl.setopt(pycurl.POSTFIELDSIZE, len(request.body or ""))
            else:
                curl.setopt(pycurl.UPLOAD, True)
                curl.setopt(pycurl.INFILESIZE, len(request.body or ""))

        if request.auth_username is not None:
            assert request.auth_password is not None
            if request.auth_mode is None or request.auth_mode == "basic":
                curl.setopt(pycurl.HTTPAUTH, pycurl.HTTPAUTH_BASIC)
            elif request.auth_mode == "digest":
                curl.setopt(pycurl.HTTPAUTH, pycurl.HTTPAUTH_DIGEST)
            else:
                raise ValueError("Unsupported auth_mode %s" % request.auth_mode)

            userpwd = httputil.encode_username_password(
                request.auth_username, request.auth_password
            )
            curl.setopt(pycurl.USERPWD, userpwd)
            curl_log.debug(
                "%s %s (username: %r)",
                request.method,
                request.url,
                request.auth_username,
            )
        else:
            curl.unsetopt(pycurl.USERPWD)
            curl_log.debug("%s %s", request.method, request.url)

        if request.client_cert is not None:
            curl.setopt(pycurl.SSLCERT, request.client_cert)

        if request.client_key is not None:
            curl.setopt(pycurl.SSLKEY, request.client_key)

        if request.ssl_options is not None:
            raise ValueError("ssl_options not supported in curl_httpclient")

        if threading.active_count() > 1:
            # libcurl/pycurl is not thread-safe by default.  When multiple threads
            # are used, signals should be disabled.  This has the side effect
            # of disabling DNS timeouts in some environments (when libcurl is
            # not linked against ares), so we don't do it when there is only one
            # thread.  Applications that use many short-lived threads may need
            # to set NOSIGNAL manually in a prepare_curl_callback since
            # there may not be any other threads running at the time we call
            # threading.activeCount.
            curl.setopt(pycurl.NOSIGNAL, 1)
        if request.prepare_curl_callback is not None:
            request.prepare_curl_callback(curl)

    def _curl_header_callback(
        self,
        headers: httputil.HTTPHeaders,
        header_callback: Callable[[str], None],
        header_line_bytes: bytes,
    ) -> None:
        header_line = native_str(header_line_bytes.decode("latin1"))
        if header_callback is not None:
            self.io_loop.add_callback(header_callback, header_line)
        # header_line as returned by curl includes the end-of-line characters.
        # whitespace at the start should be preserved to allow multi-line headers
        header_line = header_line.rstrip()
        if header_line.startswith("HTTP/"):
            headers.clear()
            try:
                (__, __, reason) = httputil.parse_response_start_line(header_line)
                header_line = "X-Http-Reason: %s" % reason
            except httputil.HTTPInputError:
                return
        if not header_line:
            return
        headers.parse_line(header_line)

    def _curl_debug(self, debug_type: int, debug_msg: str) -> None:
        debug_types = ("I", "<", ">", "<", ">")
        if debug_type == 0:
            debug_msg = native_str(debug_msg)
            curl_log.debug("%s", debug_msg.strip())
        elif debug_type in (1, 2):
            debug_msg = native_str(debug_msg)
            for line in debug_msg.splitlines():
                curl_log.debug("%s %s", debug_types[debug_type], line)
        elif debug_type == 4:
            curl_log.debug("%s %r", debug_types[debug_type], debug_msg)


class CurlError(HTTPError):
    def __init__(self, errno: int, message: str) -> None:
        HTTPError.__init__(self, 599, message)
        self.errno = errno


if __name__ == "__main__":
    AsyncHTTPClient.configure(CurlAsyncHTTPClient)
    main()
