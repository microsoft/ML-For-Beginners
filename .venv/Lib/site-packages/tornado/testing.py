"""Support classes for automated testing.

* `AsyncTestCase` and `AsyncHTTPTestCase`:  Subclasses of unittest.TestCase
  with additional support for testing asynchronous (`.IOLoop`-based) code.

* `ExpectLog`: Make test logs less spammy.

* `main()`: A simple test runner (wrapper around unittest.main()) with support
  for the tornado.autoreload module to rerun the tests when code changes.
"""

import asyncio
from collections.abc import Generator
import functools
import inspect
import logging
import os
import re
import signal
import socket
import sys
import unittest
import warnings

from tornado import gen
from tornado.httpclient import AsyncHTTPClient, HTTPResponse
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop, TimeoutError
from tornado import netutil
from tornado.platform.asyncio import AsyncIOMainLoop
from tornado.process import Subprocess
from tornado.log import app_log
from tornado.util import raise_exc_info, basestring_type
from tornado.web import Application

import typing
from typing import Tuple, Any, Callable, Type, Dict, Union, Optional, Coroutine
from types import TracebackType

if typing.TYPE_CHECKING:
    _ExcInfoTuple = Tuple[
        Optional[Type[BaseException]], Optional[BaseException], Optional[TracebackType]
    ]


_NON_OWNED_IOLOOPS = AsyncIOMainLoop


def bind_unused_port(
    reuse_port: bool = False, address: str = "127.0.0.1"
) -> Tuple[socket.socket, int]:
    """Binds a server socket to an available port on localhost.

    Returns a tuple (socket, port).

    .. versionchanged:: 4.4
       Always binds to ``127.0.0.1`` without resolving the name
       ``localhost``.

    .. versionchanged:: 6.2
       Added optional ``address`` argument to
       override the default "127.0.0.1".
    """
    sock = netutil.bind_sockets(
        0, address, family=socket.AF_INET, reuse_port=reuse_port
    )[0]
    port = sock.getsockname()[1]
    return sock, port


def get_async_test_timeout() -> float:
    """Get the global timeout setting for async tests.

    Returns a float, the timeout in seconds.

    .. versionadded:: 3.1
    """
    env = os.environ.get("ASYNC_TEST_TIMEOUT")
    if env is not None:
        try:
            return float(env)
        except ValueError:
            pass
    return 5


class _TestMethodWrapper(object):
    """Wraps a test method to raise an error if it returns a value.

    This is mainly used to detect undecorated generators (if a test
    method yields it must use a decorator to consume the generator),
    but will also detect other kinds of return values (these are not
    necessarily errors, but we alert anyway since there is no good
    reason to return a value from a test).
    """

    def __init__(self, orig_method: Callable) -> None:
        self.orig_method = orig_method
        self.__wrapped__ = orig_method

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        result = self.orig_method(*args, **kwargs)
        if isinstance(result, Generator) or inspect.iscoroutine(result):
            raise TypeError(
                "Generator and coroutine test methods should be"
                " decorated with tornado.testing.gen_test"
            )
        elif result is not None:
            raise ValueError("Return value from test method ignored: %r" % result)

    def __getattr__(self, name: str) -> Any:
        """Proxy all unknown attributes to the original method.

        This is important for some of the decorators in the `unittest`
        module, such as `unittest.skipIf`.
        """
        return getattr(self.orig_method, name)


class AsyncTestCase(unittest.TestCase):
    """`~unittest.TestCase` subclass for testing `.IOLoop`-based
    asynchronous code.

    The unittest framework is synchronous, so the test must be
    complete by the time the test method returns. This means that
    asynchronous code cannot be used in quite the same way as usual
    and must be adapted to fit. To write your tests with coroutines,
    decorate your test methods with `tornado.testing.gen_test` instead
    of `tornado.gen.coroutine`.

    This class also provides the (deprecated) `stop()` and `wait()`
    methods for a more manual style of testing. The test method itself
    must call ``self.wait()``, and asynchronous callbacks should call
    ``self.stop()`` to signal completion.

    By default, a new `.IOLoop` is constructed for each test and is available
    as ``self.io_loop``.  If the code being tested requires a
    reused global `.IOLoop`, subclasses should override `get_new_ioloop` to return it,
    although this is deprecated as of Tornado 6.3.

    The `.IOLoop`'s ``start`` and ``stop`` methods should not be
    called directly.  Instead, use `self.stop <stop>` and `self.wait
    <wait>`.  Arguments passed to ``self.stop`` are returned from
    ``self.wait``.  It is possible to have multiple ``wait``/``stop``
    cycles in the same test.

    Example::

        # This test uses coroutine style.
        class MyTestCase(AsyncTestCase):
            @tornado.testing.gen_test
            def test_http_fetch(self):
                client = AsyncHTTPClient()
                response = yield client.fetch("http://www.tornadoweb.org")
                # Test contents of response
                self.assertIn("FriendFeed", response.body)

        # This test uses argument passing between self.stop and self.wait.
        class MyTestCase2(AsyncTestCase):
            def test_http_fetch(self):
                client = AsyncHTTPClient()
                client.fetch("http://www.tornadoweb.org/", self.stop)
                response = self.wait()
                # Test contents of response
                self.assertIn("FriendFeed", response.body)
    """

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.__stopped = False
        self.__running = False
        self.__failure = None  # type: Optional[_ExcInfoTuple]
        self.__stop_args = None  # type: Any
        self.__timeout = None  # type: Optional[object]

        # It's easy to forget the @gen_test decorator, but if you do
        # the test will silently be ignored because nothing will consume
        # the generator.  Replace the test method with a wrapper that will
        # make sure it's not an undecorated generator.
        setattr(self, methodName, _TestMethodWrapper(getattr(self, methodName)))

        # Not used in this class itself, but used by @gen_test
        self._test_generator = None  # type: Optional[Union[Generator, Coroutine]]

    def setUp(self) -> None:
        py_ver = sys.version_info
        if ((3, 10, 0) <= py_ver < (3, 10, 9)) or ((3, 11, 0) <= py_ver <= (3, 11, 1)):
            # Early releases in the Python 3.10 and 3.1 series had deprecation
            # warnings that were later reverted; we must suppress them here.
            setup_with_context_manager(self, warnings.catch_warnings())
            warnings.filterwarnings(
                "ignore",
                message="There is no current event loop",
                category=DeprecationWarning,
                module=r"tornado\..*",
            )
        super().setUp()
        if type(self).get_new_ioloop is not AsyncTestCase.get_new_ioloop:
            warnings.warn("get_new_ioloop is deprecated", DeprecationWarning)
        self.io_loop = self.get_new_ioloop()
        asyncio.set_event_loop(self.io_loop.asyncio_loop)  # type: ignore[attr-defined]

    def tearDown(self) -> None:
        # Native coroutines tend to produce warnings if they're not
        # allowed to run to completion. It's difficult to ensure that
        # this always happens in tests, so cancel any tasks that are
        # still pending by the time we get here.
        asyncio_loop = self.io_loop.asyncio_loop  # type: ignore
        if hasattr(asyncio, "all_tasks"):  # py37
            tasks = asyncio.all_tasks(asyncio_loop)  # type: ignore
        else:
            tasks = asyncio.Task.all_tasks(asyncio_loop)
        # Tasks that are done may still appear here and may contain
        # non-cancellation exceptions, so filter them out.
        tasks = [t for t in tasks if not t.done()]  # type: ignore
        for t in tasks:
            t.cancel()
        # Allow the tasks to run and finalize themselves (which means
        # raising a CancelledError inside the coroutine). This may
        # just transform the "task was destroyed but it is pending"
        # warning into a "uncaught CancelledError" warning, but
        # catching CancelledErrors in coroutines that may leak is
        # simpler than ensuring that no coroutines leak.
        if tasks:
            done, pending = self.io_loop.run_sync(lambda: asyncio.wait(tasks))
            assert not pending
            # If any task failed with anything but a CancelledError, raise it.
            for f in done:
                try:
                    f.result()
                except asyncio.CancelledError:
                    pass

        # Clean up Subprocess, so it can be used again with a new ioloop.
        Subprocess.uninitialize()
        asyncio.set_event_loop(None)
        if not isinstance(self.io_loop, _NON_OWNED_IOLOOPS):
            # Try to clean up any file descriptors left open in the ioloop.
            # This avoids leaks, especially when tests are run repeatedly
            # in the same process with autoreload (because curl does not
            # set FD_CLOEXEC on its file descriptors)
            self.io_loop.close(all_fds=True)
        super().tearDown()
        # In case an exception escaped or the StackContext caught an exception
        # when there wasn't a wait() to re-raise it, do so here.
        # This is our last chance to raise an exception in a way that the
        # unittest machinery understands.
        self.__rethrow()

    def get_new_ioloop(self) -> IOLoop:
        """Returns the `.IOLoop` to use for this test.

        By default, a new `.IOLoop` is created for each test.
        Subclasses may override this method to return
        `.IOLoop.current()` if it is not appropriate to use a new
        `.IOLoop` in each tests (for example, if there are global
        singletons using the default `.IOLoop`) or if a per-test event
        loop is being provided by another system (such as
        ``pytest-asyncio``).

        .. deprecated:: 6.3
           This method will be removed in Tornado 7.0.
        """
        return IOLoop(make_current=False)

    def _handle_exception(
        self, typ: Type[Exception], value: Exception, tb: TracebackType
    ) -> bool:
        if self.__failure is None:
            self.__failure = (typ, value, tb)
        else:
            app_log.error(
                "multiple unhandled exceptions in test", exc_info=(typ, value, tb)
            )
        self.stop()
        return True

    def __rethrow(self) -> None:
        if self.__failure is not None:
            failure = self.__failure
            self.__failure = None
            raise_exc_info(failure)

    def run(
        self, result: Optional[unittest.TestResult] = None
    ) -> Optional[unittest.TestResult]:
        ret = super().run(result)
        # As a last resort, if an exception escaped super.run() and wasn't
        # re-raised in tearDown, raise it here.  This will cause the
        # unittest run to fail messily, but that's better than silently
        # ignoring an error.
        self.__rethrow()
        return ret

    def stop(self, _arg: Any = None, **kwargs: Any) -> None:
        """Stops the `.IOLoop`, causing one pending (or future) call to `wait()`
        to return.

        Keyword arguments or a single positional argument passed to `stop()` are
        saved and will be returned by `wait()`.

        .. deprecated:: 5.1

           `stop` and `wait` are deprecated; use ``@gen_test`` instead.
        """
        assert _arg is None or not kwargs
        self.__stop_args = kwargs or _arg
        if self.__running:
            self.io_loop.stop()
            self.__running = False
        self.__stopped = True

    def wait(
        self,
        condition: Optional[Callable[..., bool]] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        """Runs the `.IOLoop` until stop is called or timeout has passed.

        In the event of a timeout, an exception will be thrown. The
        default timeout is 5 seconds; it may be overridden with a
        ``timeout`` keyword argument or globally with the
        ``ASYNC_TEST_TIMEOUT`` environment variable.

        If ``condition`` is not ``None``, the `.IOLoop` will be restarted
        after `stop()` until ``condition()`` returns ``True``.

        .. versionchanged:: 3.1
           Added the ``ASYNC_TEST_TIMEOUT`` environment variable.

        .. deprecated:: 5.1

           `stop` and `wait` are deprecated; use ``@gen_test`` instead.
        """
        if timeout is None:
            timeout = get_async_test_timeout()

        if not self.__stopped:
            if timeout:

                def timeout_func() -> None:
                    try:
                        raise self.failureException(
                            "Async operation timed out after %s seconds" % timeout
                        )
                    except Exception:
                        self.__failure = sys.exc_info()
                    self.stop()

                self.__timeout = self.io_loop.add_timeout(
                    self.io_loop.time() + timeout, timeout_func
                )
            while True:
                self.__running = True
                self.io_loop.start()
                if self.__failure is not None or condition is None or condition():
                    break
            if self.__timeout is not None:
                self.io_loop.remove_timeout(self.__timeout)
                self.__timeout = None
        assert self.__stopped
        self.__stopped = False
        self.__rethrow()
        result = self.__stop_args
        self.__stop_args = None
        return result


class AsyncHTTPTestCase(AsyncTestCase):
    """A test case that starts up an HTTP server.

    Subclasses must override `get_app()`, which returns the
    `tornado.web.Application` (or other `.HTTPServer` callback) to be tested.
    Tests will typically use the provided ``self.http_client`` to fetch
    URLs from this server.

    Example, assuming the "Hello, world" example from the user guide is in
    ``hello.py``::

        import hello

        class TestHelloApp(AsyncHTTPTestCase):
            def get_app(self):
                return hello.make_app()

            def test_homepage(self):
                response = self.fetch('/')
                self.assertEqual(response.code, 200)
                self.assertEqual(response.body, 'Hello, world')

    That call to ``self.fetch()`` is equivalent to ::

        self.http_client.fetch(self.get_url('/'), self.stop)
        response = self.wait()

    which illustrates how AsyncTestCase can turn an asynchronous operation,
    like ``http_client.fetch()``, into a synchronous operation. If you need
    to do other asynchronous operations in tests, you'll probably need to use
    ``stop()`` and ``wait()`` yourself.
    """

    def setUp(self) -> None:
        super().setUp()
        sock, port = bind_unused_port()
        self.__port = port

        self.http_client = self.get_http_client()
        self._app = self.get_app()
        self.http_server = self.get_http_server()
        self.http_server.add_sockets([sock])

    def get_http_client(self) -> AsyncHTTPClient:
        return AsyncHTTPClient()

    def get_http_server(self) -> HTTPServer:
        return HTTPServer(self._app, **self.get_httpserver_options())

    def get_app(self) -> Application:
        """Should be overridden by subclasses to return a
        `tornado.web.Application` or other `.HTTPServer` callback.
        """
        raise NotImplementedError()

    def fetch(
        self, path: str, raise_error: bool = False, **kwargs: Any
    ) -> HTTPResponse:
        """Convenience method to synchronously fetch a URL.

        The given path will be appended to the local server's host and
        port.  Any additional keyword arguments will be passed directly to
        `.AsyncHTTPClient.fetch` (and so could be used to pass
        ``method="POST"``, ``body="..."``, etc).

        If the path begins with http:// or https://, it will be treated as a
        full URL and will be fetched as-is.

        If ``raise_error`` is ``True``, a `tornado.httpclient.HTTPError` will
        be raised if the response code is not 200. This is the same behavior
        as the ``raise_error`` argument to `.AsyncHTTPClient.fetch`, but
        the default is ``False`` here (it's ``True`` in `.AsyncHTTPClient`)
        because tests often need to deal with non-200 response codes.

        .. versionchanged:: 5.0
           Added support for absolute URLs.

        .. versionchanged:: 5.1

           Added the ``raise_error`` argument.

        .. deprecated:: 5.1

           This method currently turns any exception into an
           `.HTTPResponse` with status code 599. In Tornado 6.0,
           errors other than `tornado.httpclient.HTTPError` will be
           passed through, and ``raise_error=False`` will only
           suppress errors that would be raised due to non-200
           response codes.

        """
        if path.lower().startswith(("http://", "https://")):
            url = path
        else:
            url = self.get_url(path)
        return self.io_loop.run_sync(
            lambda: self.http_client.fetch(url, raise_error=raise_error, **kwargs),
            timeout=get_async_test_timeout(),
        )

    def get_httpserver_options(self) -> Dict[str, Any]:
        """May be overridden by subclasses to return additional
        keyword arguments for the server.
        """
        return {}

    def get_http_port(self) -> int:
        """Returns the port used by the server.

        A new port is chosen for each test.
        """
        return self.__port

    def get_protocol(self) -> str:
        return "http"

    def get_url(self, path: str) -> str:
        """Returns an absolute url for the given path on the test server."""
        return "%s://127.0.0.1:%s%s" % (self.get_protocol(), self.get_http_port(), path)

    def tearDown(self) -> None:
        self.http_server.stop()
        self.io_loop.run_sync(
            self.http_server.close_all_connections, timeout=get_async_test_timeout()
        )
        self.http_client.close()
        del self.http_server
        del self._app
        super().tearDown()


class AsyncHTTPSTestCase(AsyncHTTPTestCase):
    """A test case that starts an HTTPS server.

    Interface is generally the same as `AsyncHTTPTestCase`.
    """

    def get_http_client(self) -> AsyncHTTPClient:
        return AsyncHTTPClient(force_instance=True, defaults=dict(validate_cert=False))

    def get_httpserver_options(self) -> Dict[str, Any]:
        return dict(ssl_options=self.get_ssl_options())

    def get_ssl_options(self) -> Dict[str, Any]:
        """May be overridden by subclasses to select SSL options.

        By default includes a self-signed testing certificate.
        """
        return AsyncHTTPSTestCase.default_ssl_options()

    @staticmethod
    def default_ssl_options() -> Dict[str, Any]:
        # Testing keys were generated with:
        # openssl req -new -keyout tornado/test/test.key \
        #                     -out tornado/test/test.crt -nodes -days 3650 -x509
        module_dir = os.path.dirname(__file__)
        return dict(
            certfile=os.path.join(module_dir, "test", "test.crt"),
            keyfile=os.path.join(module_dir, "test", "test.key"),
        )

    def get_protocol(self) -> str:
        return "https"


@typing.overload
def gen_test(
    *, timeout: Optional[float] = None
) -> Callable[[Callable[..., Union[Generator, "Coroutine"]]], Callable[..., None]]:
    pass


@typing.overload  # noqa: F811
def gen_test(func: Callable[..., Union[Generator, "Coroutine"]]) -> Callable[..., None]:
    pass


def gen_test(  # noqa: F811
    func: Optional[Callable[..., Union[Generator, "Coroutine"]]] = None,
    timeout: Optional[float] = None,
) -> Union[
    Callable[..., None],
    Callable[[Callable[..., Union[Generator, "Coroutine"]]], Callable[..., None]],
]:
    """Testing equivalent of ``@gen.coroutine``, to be applied to test methods.

    ``@gen.coroutine`` cannot be used on tests because the `.IOLoop` is not
    already running.  ``@gen_test`` should be applied to test methods
    on subclasses of `AsyncTestCase`.

    Example::

        class MyTest(AsyncHTTPTestCase):
            @gen_test
            def test_something(self):
                response = yield self.http_client.fetch(self.get_url('/'))

    By default, ``@gen_test`` times out after 5 seconds. The timeout may be
    overridden globally with the ``ASYNC_TEST_TIMEOUT`` environment variable,
    or for each test with the ``timeout`` keyword argument::

        class MyTest(AsyncHTTPTestCase):
            @gen_test(timeout=10)
            def test_something_slow(self):
                response = yield self.http_client.fetch(self.get_url('/'))

    Note that ``@gen_test`` is incompatible with `AsyncTestCase.stop`,
    `AsyncTestCase.wait`, and `AsyncHTTPTestCase.fetch`. Use ``yield
    self.http_client.fetch(self.get_url())`` as shown above instead.

    .. versionadded:: 3.1
       The ``timeout`` argument and ``ASYNC_TEST_TIMEOUT`` environment
       variable.

    .. versionchanged:: 4.0
       The wrapper now passes along ``*args, **kwargs`` so it can be used
       on functions with arguments.

    """
    if timeout is None:
        timeout = get_async_test_timeout()

    def wrap(f: Callable[..., Union[Generator, "Coroutine"]]) -> Callable[..., None]:
        # Stack up several decorators to allow us to access the generator
        # object itself.  In the innermost wrapper, we capture the generator
        # and save it in an attribute of self.  Next, we run the wrapped
        # function through @gen.coroutine.  Finally, the coroutine is
        # wrapped again to make it synchronous with run_sync.
        #
        # This is a good case study arguing for either some sort of
        # extensibility in the gen decorators or cancellation support.
        @functools.wraps(f)
        def pre_coroutine(self, *args, **kwargs):
            # type: (AsyncTestCase, *Any, **Any) -> Union[Generator, Coroutine]
            # Type comments used to avoid pypy3 bug.
            result = f(self, *args, **kwargs)
            if isinstance(result, Generator) or inspect.iscoroutine(result):
                self._test_generator = result
            else:
                self._test_generator = None
            return result

        if inspect.iscoroutinefunction(f):
            coro = pre_coroutine
        else:
            coro = gen.coroutine(pre_coroutine)  # type: ignore[assignment]

        @functools.wraps(coro)
        def post_coroutine(self, *args, **kwargs):
            # type: (AsyncTestCase, *Any, **Any) -> None
            try:
                return self.io_loop.run_sync(
                    functools.partial(coro, self, *args, **kwargs), timeout=timeout
                )
            except TimeoutError as e:
                # run_sync raises an error with an unhelpful traceback.
                # If the underlying generator is still running, we can throw the
                # exception back into it so the stack trace is replaced by the
                # point where the test is stopped. The only reason the generator
                # would not be running would be if it were cancelled, which means
                # a native coroutine, so we can rely on the cr_running attribute.
                if self._test_generator is not None and getattr(
                    self._test_generator, "cr_running", True
                ):
                    self._test_generator.throw(e)
                    # In case the test contains an overly broad except
                    # clause, we may get back here.
                # Coroutine was stopped or didn't raise a useful stack trace,
                # so re-raise the original exception which is better than nothing.
                raise

        return post_coroutine

    if func is not None:
        # Used like:
        #     @gen_test
        #     def f(self):
        #         pass
        return wrap(func)
    else:
        # Used like @gen_test(timeout=10)
        return wrap


# Without this attribute, nosetests will try to run gen_test as a test
# anywhere it is imported.
gen_test.__test__ = False  # type: ignore


class ExpectLog(logging.Filter):
    """Context manager to capture and suppress expected log output.

    Useful to make tests of error conditions less noisy, while still
    leaving unexpected log entries visible.  *Not thread safe.*

    The attribute ``logged_stack`` is set to ``True`` if any exception
    stack trace was logged.

    Usage::

        with ExpectLog('tornado.application', "Uncaught exception"):
            error_response = self.fetch("/some_page")

    .. versionchanged:: 4.3
       Added the ``logged_stack`` attribute.
    """

    def __init__(
        self,
        logger: Union[logging.Logger, basestring_type],
        regex: str,
        required: bool = True,
        level: Optional[int] = None,
    ) -> None:
        """Constructs an ExpectLog context manager.

        :param logger: Logger object (or name of logger) to watch.  Pass an
            empty string to watch the root logger.
        :param regex: Regular expression to match.  Any log entries on the
            specified logger that match this regex will be suppressed.
        :param required: If true, an exception will be raised if the end of the
            ``with`` statement is reached without matching any log entries.
        :param level: A constant from the ``logging`` module indicating the
            expected log level. If this parameter is provided, only log messages
            at this level will be considered to match. Additionally, the
            supplied ``logger`` will have its level adjusted if necessary (for
            the duration of the ``ExpectLog`` to enable the expected message.

        .. versionchanged:: 6.1
           Added the ``level`` parameter.

        .. deprecated:: 6.3
           In Tornado 7.0, only ``WARNING`` and higher logging levels will be
           matched by default. To match ``INFO`` and lower levels, the ``level``
           argument must be used. This is changing to minimize differences
           between ``tornado.testing.main`` (which enables ``INFO`` logs by
           default) and most other test runners (including those in IDEs)
           which have ``INFO`` logs disabled by default.
        """
        if isinstance(logger, basestring_type):
            logger = logging.getLogger(logger)
        self.logger = logger
        self.regex = re.compile(regex)
        self.required = required
        # matched and deprecated_level_matched are a counter for the respective event.
        self.matched = 0
        self.deprecated_level_matched = 0
        self.logged_stack = False
        self.level = level
        self.orig_level = None  # type: Optional[int]

    def filter(self, record: logging.LogRecord) -> bool:
        if record.exc_info:
            self.logged_stack = True
        message = record.getMessage()
        if self.regex.match(message):
            if self.level is None and record.levelno < logging.WARNING:
                # We're inside the logging machinery here so generating a DeprecationWarning
                # here won't be reported cleanly (if warnings-as-errors is enabled, the error
                # just gets swallowed by the logging module), and even if it were it would
                # have the wrong stack trace. Just remember this fact and report it in
                # __exit__ instead.
                self.deprecated_level_matched += 1
            if self.level is not None and record.levelno != self.level:
                app_log.warning(
                    "Got expected log message %r at unexpected level (%s vs %s)"
                    % (message, logging.getLevelName(self.level), record.levelname)
                )
                return True
            self.matched += 1
            return False
        return True

    def __enter__(self) -> "ExpectLog":
        if self.level is not None and self.level < self.logger.getEffectiveLevel():
            self.orig_level = self.logger.level
            self.logger.setLevel(self.level)
        self.logger.addFilter(self)
        return self

    def __exit__(
        self,
        typ: "Optional[Type[BaseException]]",
        value: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        if self.orig_level is not None:
            self.logger.setLevel(self.orig_level)
        self.logger.removeFilter(self)
        if not typ and self.required and not self.matched:
            raise Exception("did not get expected log message")
        if (
            not typ
            and self.required
            and (self.deprecated_level_matched >= self.matched)
        ):
            warnings.warn(
                "ExpectLog matched at INFO or below without level argument",
                DeprecationWarning,
            )


# From https://nedbatchelder.com/blog/201508/using_context_managers_in_test_setup.html
def setup_with_context_manager(testcase: unittest.TestCase, cm: Any) -> Any:
    """Use a contextmanager to setUp a test case."""
    val = cm.__enter__()
    testcase.addCleanup(cm.__exit__, None, None, None)
    return val


def main(**kwargs: Any) -> None:
    """A simple test runner.

    This test runner is essentially equivalent to `unittest.main` from
    the standard library, but adds support for Tornado-style option
    parsing and log formatting. It is *not* necessary to use this
    `main` function to run tests using `AsyncTestCase`; these tests
    are self-contained and can run with any test runner.

    The easiest way to run a test is via the command line::

        python -m tornado.testing tornado.test.web_test

    See the standard library ``unittest`` module for ways in which
    tests can be specified.

    Projects with many tests may wish to define a test script like
    ``tornado/test/runtests.py``.  This script should define a method
    ``all()`` which returns a test suite and then call
    `tornado.testing.main()`.  Note that even when a test script is
    used, the ``all()`` test suite may be overridden by naming a
    single test on the command line::

        # Runs all tests
        python -m tornado.test.runtests
        # Runs one test
        python -m tornado.test.runtests tornado.test.web_test

    Additional keyword arguments passed through to ``unittest.main()``.
    For example, use ``tornado.testing.main(verbosity=2)``
    to show many test details as they are run.
    See http://docs.python.org/library/unittest.html#unittest.main
    for full argument list.

    .. versionchanged:: 5.0

       This function produces no output of its own; only that produced
       by the `unittest` module (previously it would add a PASS or FAIL
       log message).
    """
    from tornado.options import define, options, parse_command_line

    define(
        "exception_on_interrupt",
        type=bool,
        default=True,
        help=(
            "If true (default), ctrl-c raises a KeyboardInterrupt "
            "exception.  This prints a stack trace but cannot interrupt "
            "certain operations.  If false, the process is more reliably "
            "killed, but does not print a stack trace."
        ),
    )

    # support the same options as unittest's command-line interface
    define("verbose", type=bool)
    define("quiet", type=bool)
    define("failfast", type=bool)
    define("catch", type=bool)
    define("buffer", type=bool)

    argv = [sys.argv[0]] + parse_command_line(sys.argv)

    if not options.exception_on_interrupt:
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    if options.verbose is not None:
        kwargs["verbosity"] = 2
    if options.quiet is not None:
        kwargs["verbosity"] = 0
    if options.failfast is not None:
        kwargs["failfast"] = True
    if options.catch is not None:
        kwargs["catchbreak"] = True
    if options.buffer is not None:
        kwargs["buffer"] = True

    if __name__ == "__main__" and len(argv) == 1:
        print("No tests specified", file=sys.stderr)
        sys.exit(1)
    # In order to be able to run tests by their fully-qualified name
    # on the command line without importing all tests here,
    # module must be set to None.  Python 3.2's unittest.main ignores
    # defaultTest if no module is given (it tries to do its own
    # test discovery, which is incompatible with auto2to3), so don't
    # set module if we're not asking for a specific test.
    if len(argv) > 1:
        unittest.main(module=None, argv=argv, **kwargs)  # type: ignore
    else:
        unittest.main(defaultTest="all", argv=argv, **kwargs)


if __name__ == "__main__":
    main()
