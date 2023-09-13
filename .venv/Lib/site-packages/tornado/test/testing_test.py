from tornado import gen, ioloop
from tornado.httpserver import HTTPServer
from tornado.locks import Event
from tornado.testing import AsyncHTTPTestCase, AsyncTestCase, bind_unused_port, gen_test
from tornado.web import Application
import asyncio
import contextlib
import inspect
import gc
import os
import platform
import sys
import traceback
import unittest
import warnings


@contextlib.contextmanager
def set_environ(name, value):
    old_value = os.environ.get(name)
    os.environ[name] = value

    try:
        yield
    finally:
        if old_value is None:
            del os.environ[name]
        else:
            os.environ[name] = old_value


class AsyncTestCaseTest(AsyncTestCase):
    def test_wait_timeout(self):
        time = self.io_loop.time

        # Accept default 5-second timeout, no error
        self.io_loop.add_timeout(time() + 0.01, self.stop)
        self.wait()

        # Timeout passed to wait()
        self.io_loop.add_timeout(time() + 1, self.stop)
        with self.assertRaises(self.failureException):
            self.wait(timeout=0.01)

        # Timeout set with environment variable
        self.io_loop.add_timeout(time() + 1, self.stop)
        with set_environ("ASYNC_TEST_TIMEOUT", "0.01"):
            with self.assertRaises(self.failureException):
                self.wait()

    def test_subsequent_wait_calls(self):
        """
        This test makes sure that a second call to wait()
        clears the first timeout.
        """
        # The first wait ends with time left on the clock
        self.io_loop.add_timeout(self.io_loop.time() + 0.00, self.stop)
        self.wait(timeout=0.1)
        # The second wait has enough time for itself but would fail if the
        # first wait's deadline were still in effect.
        self.io_loop.add_timeout(self.io_loop.time() + 0.2, self.stop)
        self.wait(timeout=0.4)


class LeakTest(AsyncTestCase):
    def tearDown(self):
        super().tearDown()
        # Trigger a gc to make warnings more deterministic.
        gc.collect()

    def test_leaked_coroutine(self):
        # This test verifies that "leaked" coroutines are shut down
        # without triggering warnings like "task was destroyed but it
        # is pending". If this test were to fail, it would fail
        # because runtests.py detected unexpected output to stderr.
        event = Event()

        async def callback():
            try:
                await event.wait()
            except asyncio.CancelledError:
                pass

        self.io_loop.add_callback(callback)
        self.io_loop.add_callback(self.stop)
        self.wait()


class AsyncHTTPTestCaseTest(AsyncHTTPTestCase):
    def setUp(self):
        super().setUp()
        # Bind a second port.
        sock, port = bind_unused_port()
        app = Application()
        server = HTTPServer(app, **self.get_httpserver_options())
        server.add_socket(sock)
        self.second_port = port
        self.second_server = server

    def get_app(self):
        return Application()

    def test_fetch_segment(self):
        path = "/path"
        response = self.fetch(path)
        self.assertEqual(response.request.url, self.get_url(path))

    def test_fetch_full_http_url(self):
        # Ensure that self.fetch() recognizes absolute urls and does
        # not transform them into references to our main test server.
        path = "http://127.0.0.1:%d/path" % self.second_port

        response = self.fetch(path)
        self.assertEqual(response.request.url, path)

    def tearDown(self):
        self.second_server.stop()
        super().tearDown()


class AsyncTestCaseWrapperTest(unittest.TestCase):
    def test_undecorated_generator(self):
        class Test(AsyncTestCase):
            def test_gen(self):
                yield

        test = Test("test_gen")
        result = unittest.TestResult()
        test.run(result)
        self.assertEqual(len(result.errors), 1)
        self.assertIn("should be decorated", result.errors[0][1])

    @unittest.skipIf(
        platform.python_implementation() == "PyPy",
        "pypy destructor warnings cannot be silenced",
    )
    @unittest.skipIf(
        sys.version_info >= (3, 12), "py312 has its own check for test case returns"
    )
    def test_undecorated_coroutine(self):
        class Test(AsyncTestCase):
            async def test_coro(self):
                pass

        test = Test("test_coro")
        result = unittest.TestResult()

        # Silence "RuntimeWarning: coroutine 'test_coro' was never awaited".
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test.run(result)

        self.assertEqual(len(result.errors), 1)
        self.assertIn("should be decorated", result.errors[0][1])

    def test_undecorated_generator_with_skip(self):
        class Test(AsyncTestCase):
            @unittest.skip("don't run this")
            def test_gen(self):
                yield

        test = Test("test_gen")
        result = unittest.TestResult()
        test.run(result)
        self.assertEqual(len(result.errors), 0)
        self.assertEqual(len(result.skipped), 1)

    def test_other_return(self):
        class Test(AsyncTestCase):
            def test_other_return(self):
                return 42

        test = Test("test_other_return")
        result = unittest.TestResult()
        test.run(result)
        self.assertEqual(len(result.errors), 1)
        self.assertIn("Return value from test method ignored", result.errors[0][1])

    def test_unwrap(self):
        class Test(AsyncTestCase):
            def test_foo(self):
                pass

        test = Test("test_foo")
        self.assertIs(
            inspect.unwrap(test.test_foo),
            test.test_foo.orig_method,  # type: ignore[attr-defined]
        )


class SetUpTearDownTest(unittest.TestCase):
    def test_set_up_tear_down(self):
        """
        This test makes sure that AsyncTestCase calls super methods for
        setUp and tearDown.

        InheritBoth is a subclass of both AsyncTestCase and
        SetUpTearDown, with the ordering so that the super of
        AsyncTestCase will be SetUpTearDown.
        """
        events = []
        result = unittest.TestResult()

        class SetUpTearDown(unittest.TestCase):
            def setUp(self):
                events.append("setUp")

            def tearDown(self):
                events.append("tearDown")

        class InheritBoth(AsyncTestCase, SetUpTearDown):
            def test(self):
                events.append("test")

        InheritBoth("test").run(result)
        expected = ["setUp", "test", "tearDown"]
        self.assertEqual(expected, events)


class AsyncHTTPTestCaseSetUpTearDownTest(unittest.TestCase):
    def test_tear_down_releases_app_and_http_server(self):
        result = unittest.TestResult()

        class SetUpTearDown(AsyncHTTPTestCase):
            def get_app(self):
                return Application()

            def test(self):
                self.assertTrue(hasattr(self, "_app"))
                self.assertTrue(hasattr(self, "http_server"))

        test = SetUpTearDown("test")
        test.run(result)
        self.assertFalse(hasattr(test, "_app"))
        self.assertFalse(hasattr(test, "http_server"))


class GenTest(AsyncTestCase):
    def setUp(self):
        super().setUp()
        self.finished = False

    def tearDown(self):
        self.assertTrue(self.finished)
        super().tearDown()

    @gen_test
    def test_sync(self):
        self.finished = True

    @gen_test
    def test_async(self):
        yield gen.moment
        self.finished = True

    def test_timeout(self):
        # Set a short timeout and exceed it.
        @gen_test(timeout=0.1)
        def test(self):
            yield gen.sleep(1)

        # This can't use assertRaises because we need to inspect the
        # exc_info triple (and not just the exception object)
        try:
            test(self)
            self.fail("did not get expected exception")
        except ioloop.TimeoutError:
            # The stack trace should blame the add_timeout line, not just
            # unrelated IOLoop/testing internals.
            self.assertIn("gen.sleep(1)", traceback.format_exc())

        self.finished = True

    def test_no_timeout(self):
        # A test that does not exceed its timeout should succeed.
        @gen_test(timeout=1)
        def test(self):
            yield gen.sleep(0.1)

        test(self)
        self.finished = True

    def test_timeout_environment_variable(self):
        @gen_test(timeout=0.5)
        def test_long_timeout(self):
            yield gen.sleep(0.25)

        # Uses provided timeout of 0.5 seconds, doesn't time out.
        with set_environ("ASYNC_TEST_TIMEOUT", "0.1"):
            test_long_timeout(self)

        self.finished = True

    def test_no_timeout_environment_variable(self):
        @gen_test(timeout=0.01)
        def test_short_timeout(self):
            yield gen.sleep(1)

        # Uses environment-variable timeout of 0.1, times out.
        with set_environ("ASYNC_TEST_TIMEOUT", "0.1"):
            with self.assertRaises(ioloop.TimeoutError):
                test_short_timeout(self)

        self.finished = True

    def test_with_method_args(self):
        @gen_test
        def test_with_args(self, *args):
            self.assertEqual(args, ("test",))
            yield gen.moment

        test_with_args(self, "test")
        self.finished = True

    def test_with_method_kwargs(self):
        @gen_test
        def test_with_kwargs(self, **kwargs):
            self.assertDictEqual(kwargs, {"test": "test"})
            yield gen.moment

        test_with_kwargs(self, test="test")
        self.finished = True

    def test_native_coroutine(self):
        @gen_test
        async def test(self):
            self.finished = True

        test(self)

    def test_native_coroutine_timeout(self):
        # Set a short timeout and exceed it.
        @gen_test(timeout=0.1)
        async def test(self):
            await gen.sleep(1)

        try:
            test(self)
            self.fail("did not get expected exception")
        except ioloop.TimeoutError:
            self.finished = True


if __name__ == "__main__":
    unittest.main()
