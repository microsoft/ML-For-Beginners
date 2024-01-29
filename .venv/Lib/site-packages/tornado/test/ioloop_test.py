import asyncio
from concurrent.futures import ThreadPoolExecutor
from concurrent import futures
from collections.abc import Generator
import contextlib
import datetime
import functools
import socket
import subprocess
import sys
import threading
import time
import types
from unittest import mock
import unittest

from tornado.escape import native_str
from tornado import gen
from tornado.ioloop import IOLoop, TimeoutError, PeriodicCallback
from tornado.log import app_log
from tornado.testing import (
    AsyncTestCase,
    bind_unused_port,
    ExpectLog,
    gen_test,
    setup_with_context_manager,
)
from tornado.test.util import (
    ignore_deprecation,
    skipIfNonUnix,
    skipOnTravis,
)
from tornado.concurrent import Future

import typing

if typing.TYPE_CHECKING:
    from typing import List  # noqa: F401


class TestIOLoop(AsyncTestCase):
    def test_add_callback_return_sequence(self):
        # A callback returning {} or [] shouldn't spin the CPU, see Issue #1803.
        self.calls = 0

        loop = self.io_loop
        test = self
        old_add_callback = loop.add_callback

        def add_callback(self, callback, *args, **kwargs):
            test.calls += 1
            old_add_callback(callback, *args, **kwargs)

        loop.add_callback = types.MethodType(add_callback, loop)  # type: ignore
        loop.add_callback(lambda: {})  # type: ignore
        loop.add_callback(lambda: [])  # type: ignore
        loop.add_timeout(datetime.timedelta(milliseconds=50), loop.stop)
        loop.start()
        self.assertLess(self.calls, 10)

    @skipOnTravis
    def test_add_callback_wakeup(self):
        # Make sure that add_callback from inside a running IOLoop
        # wakes up the IOLoop immediately instead of waiting for a timeout.
        def callback():
            self.called = True
            self.stop()

        def schedule_callback():
            self.called = False
            self.io_loop.add_callback(callback)
            # Store away the time so we can check if we woke up immediately
            self.start_time = time.time()

        self.io_loop.add_timeout(self.io_loop.time(), schedule_callback)
        self.wait()
        self.assertAlmostEqual(time.time(), self.start_time, places=2)
        self.assertTrue(self.called)

    @skipOnTravis
    def test_add_callback_wakeup_other_thread(self):
        def target():
            # sleep a bit to let the ioloop go into its poll loop
            time.sleep(0.01)
            self.stop_time = time.time()
            self.io_loop.add_callback(self.stop)

        thread = threading.Thread(target=target)
        self.io_loop.add_callback(thread.start)
        self.wait()
        delta = time.time() - self.stop_time
        self.assertLess(delta, 0.1)
        thread.join()

    def test_add_timeout_timedelta(self):
        self.io_loop.add_timeout(datetime.timedelta(microseconds=1), self.stop)
        self.wait()

    def test_multiple_add(self):
        sock, port = bind_unused_port()
        try:
            self.io_loop.add_handler(
                sock.fileno(), lambda fd, events: None, IOLoop.READ
            )
            # Attempting to add the same handler twice fails
            # (with a platform-dependent exception)
            self.assertRaises(
                Exception,
                self.io_loop.add_handler,
                sock.fileno(),
                lambda fd, events: None,
                IOLoop.READ,
            )
        finally:
            self.io_loop.remove_handler(sock.fileno())
            sock.close()

    def test_remove_without_add(self):
        # remove_handler should not throw an exception if called on an fd
        # was never added.
        sock, port = bind_unused_port()
        try:
            self.io_loop.remove_handler(sock.fileno())
        finally:
            sock.close()

    def test_add_callback_from_signal(self):
        # cheat a little bit and just run this normally, since we can't
        # easily simulate the races that happen with real signal handlers
        with ignore_deprecation():
            self.io_loop.add_callback_from_signal(self.stop)
        self.wait()

    def test_add_callback_from_signal_other_thread(self):
        # Very crude test, just to make sure that we cover this case.
        # This also happens to be the first test where we run an IOLoop in
        # a non-main thread.
        other_ioloop = IOLoop()
        thread = threading.Thread(target=other_ioloop.start)
        thread.start()
        with ignore_deprecation():
            other_ioloop.add_callback_from_signal(other_ioloop.stop)
        thread.join()
        other_ioloop.close()

    def test_add_callback_while_closing(self):
        # add_callback should not fail if it races with another thread
        # closing the IOLoop. The callbacks are dropped silently
        # without executing.
        closing = threading.Event()

        def target():
            other_ioloop.add_callback(other_ioloop.stop)
            other_ioloop.start()
            closing.set()
            other_ioloop.close(all_fds=True)

        other_ioloop = IOLoop()
        thread = threading.Thread(target=target)
        thread.start()
        closing.wait()
        for i in range(1000):
            other_ioloop.add_callback(lambda: None)

    @skipIfNonUnix  # just because socketpair is so convenient
    def test_read_while_writeable(self):
        # Ensure that write events don't come in while we're waiting for
        # a read and haven't asked for writeability. (the reverse is
        # difficult to test for)
        client, server = socket.socketpair()
        try:

            def handler(fd, events):
                self.assertEqual(events, IOLoop.READ)
                self.stop()

            self.io_loop.add_handler(client.fileno(), handler, IOLoop.READ)
            self.io_loop.add_timeout(
                self.io_loop.time() + 0.01, functools.partial(server.send, b"asdf")
            )
            self.wait()
            self.io_loop.remove_handler(client.fileno())
        finally:
            client.close()
            server.close()

    def test_remove_timeout_after_fire(self):
        # It is not an error to call remove_timeout after it has run.
        handle = self.io_loop.add_timeout(self.io_loop.time(), self.stop)
        self.wait()
        self.io_loop.remove_timeout(handle)

    def test_remove_timeout_cleanup(self):
        # Add and remove enough callbacks to trigger cleanup.
        # Not a very thorough test, but it ensures that the cleanup code
        # gets executed and doesn't blow up.  This test is only really useful
        # on PollIOLoop subclasses, but it should run silently on any
        # implementation.
        for i in range(2000):
            timeout = self.io_loop.add_timeout(self.io_loop.time() + 3600, lambda: None)
            self.io_loop.remove_timeout(timeout)
        # HACK: wait two IOLoop iterations for the GC to happen.
        self.io_loop.add_callback(lambda: self.io_loop.add_callback(self.stop))
        self.wait()

    def test_remove_timeout_from_timeout(self):
        calls = [False, False]

        # Schedule several callbacks and wait for them all to come due at once.
        # t2 should be cancelled by t1, even though it is already scheduled to
        # be run before the ioloop even looks at it.
        now = self.io_loop.time()

        def t1():
            calls[0] = True
            self.io_loop.remove_timeout(t2_handle)

        self.io_loop.add_timeout(now + 0.01, t1)

        def t2():
            calls[1] = True

        t2_handle = self.io_loop.add_timeout(now + 0.02, t2)
        self.io_loop.add_timeout(now + 0.03, self.stop)
        time.sleep(0.03)
        self.wait()
        self.assertEqual(calls, [True, False])

    def test_timeout_with_arguments(self):
        # This tests that all the timeout methods pass through *args correctly.
        results = []  # type: List[int]
        self.io_loop.add_timeout(self.io_loop.time(), results.append, 1)
        self.io_loop.add_timeout(datetime.timedelta(seconds=0), results.append, 2)
        self.io_loop.call_at(self.io_loop.time(), results.append, 3)
        self.io_loop.call_later(0, results.append, 4)
        self.io_loop.call_later(0, self.stop)
        self.wait()
        # The asyncio event loop does not guarantee the order of these
        # callbacks.
        self.assertEqual(sorted(results), [1, 2, 3, 4])

    def test_add_timeout_return(self):
        # All the timeout methods return non-None handles that can be
        # passed to remove_timeout.
        handle = self.io_loop.add_timeout(self.io_loop.time(), lambda: None)
        self.assertFalse(handle is None)
        self.io_loop.remove_timeout(handle)

    def test_call_at_return(self):
        handle = self.io_loop.call_at(self.io_loop.time(), lambda: None)
        self.assertFalse(handle is None)
        self.io_loop.remove_timeout(handle)

    def test_call_later_return(self):
        handle = self.io_loop.call_later(0, lambda: None)
        self.assertFalse(handle is None)
        self.io_loop.remove_timeout(handle)

    def test_close_file_object(self):
        """When a file object is used instead of a numeric file descriptor,
        the object should be closed (by IOLoop.close(all_fds=True),
        not just the fd.
        """
        # Use a socket since they are supported by IOLoop on all platforms.
        # Unfortunately, sockets don't support the .closed attribute for
        # inspecting their close status, so we must use a wrapper.
        class SocketWrapper(object):
            def __init__(self, sockobj):
                self.sockobj = sockobj
                self.closed = False

            def fileno(self):
                return self.sockobj.fileno()

            def close(self):
                self.closed = True
                self.sockobj.close()

        sockobj, port = bind_unused_port()
        socket_wrapper = SocketWrapper(sockobj)
        io_loop = IOLoop()
        io_loop.add_handler(socket_wrapper, lambda fd, events: None, IOLoop.READ)
        io_loop.close(all_fds=True)
        self.assertTrue(socket_wrapper.closed)

    def test_handler_callback_file_object(self):
        """The handler callback receives the same fd object it passed in."""
        server_sock, port = bind_unused_port()
        fds = []

        def handle_connection(fd, events):
            fds.append(fd)
            conn, addr = server_sock.accept()
            conn.close()
            self.stop()

        self.io_loop.add_handler(server_sock, handle_connection, IOLoop.READ)
        with contextlib.closing(socket.socket()) as client_sock:
            client_sock.connect(("127.0.0.1", port))
            self.wait()
        self.io_loop.remove_handler(server_sock)
        self.io_loop.add_handler(server_sock.fileno(), handle_connection, IOLoop.READ)
        with contextlib.closing(socket.socket()) as client_sock:
            client_sock.connect(("127.0.0.1", port))
            self.wait()
        self.assertIs(fds[0], server_sock)
        self.assertEqual(fds[1], server_sock.fileno())
        self.io_loop.remove_handler(server_sock.fileno())
        server_sock.close()

    def test_mixed_fd_fileobj(self):
        server_sock, port = bind_unused_port()

        def f(fd, events):
            pass

        self.io_loop.add_handler(server_sock, f, IOLoop.READ)
        with self.assertRaises(Exception):
            # The exact error is unspecified - some implementations use
            # IOError, others use ValueError.
            self.io_loop.add_handler(server_sock.fileno(), f, IOLoop.READ)
        self.io_loop.remove_handler(server_sock.fileno())
        server_sock.close()

    def test_reentrant(self):
        """Calling start() twice should raise an error, not deadlock."""
        returned_from_start = [False]
        got_exception = [False]

        def callback():
            try:
                self.io_loop.start()
                returned_from_start[0] = True
            except Exception:
                got_exception[0] = True
            self.stop()

        self.io_loop.add_callback(callback)
        self.wait()
        self.assertTrue(got_exception[0])
        self.assertFalse(returned_from_start[0])

    def test_exception_logging(self):
        """Uncaught exceptions get logged by the IOLoop."""
        self.io_loop.add_callback(lambda: 1 / 0)
        self.io_loop.add_callback(self.stop)
        with ExpectLog(app_log, "Exception in callback"):
            self.wait()

    def test_exception_logging_future(self):
        """The IOLoop examines exceptions from Futures and logs them."""

        @gen.coroutine
        def callback():
            self.io_loop.add_callback(self.stop)
            1 / 0

        self.io_loop.add_callback(callback)
        with ExpectLog(app_log, "Exception in callback"):
            self.wait()

    def test_exception_logging_native_coro(self):
        """The IOLoop examines exceptions from awaitables and logs them."""

        async def callback():
            # Stop the IOLoop two iterations after raising an exception
            # to give the exception time to be logged.
            self.io_loop.add_callback(self.io_loop.add_callback, self.stop)
            1 / 0

        self.io_loop.add_callback(callback)
        with ExpectLog(app_log, "Exception in callback"):
            self.wait()

    def test_spawn_callback(self):
        # Both add_callback and spawn_callback run directly on the IOLoop,
        # so their errors are logged without stopping the test.
        self.io_loop.add_callback(lambda: 1 / 0)
        self.io_loop.add_callback(self.stop)
        with ExpectLog(app_log, "Exception in callback"):
            self.wait()
        # A spawned callback is run directly on the IOLoop, so it will be
        # logged without stopping the test.
        self.io_loop.spawn_callback(lambda: 1 / 0)
        self.io_loop.add_callback(self.stop)
        with ExpectLog(app_log, "Exception in callback"):
            self.wait()

    @skipIfNonUnix
    def test_remove_handler_from_handler(self):
        # Create two sockets with simultaneous read events.
        client, server = socket.socketpair()
        try:
            client.send(b"abc")
            server.send(b"abc")

            # After reading from one fd, remove the other from the IOLoop.
            chunks = []

            def handle_read(fd, events):
                chunks.append(fd.recv(1024))
                if fd is client:
                    self.io_loop.remove_handler(server)
                else:
                    self.io_loop.remove_handler(client)

            self.io_loop.add_handler(client, handle_read, self.io_loop.READ)
            self.io_loop.add_handler(server, handle_read, self.io_loop.READ)
            self.io_loop.call_later(0.1, self.stop)
            self.wait()

            # Only one fd was read; the other was cleanly removed.
            self.assertEqual(chunks, [b"abc"])
        finally:
            client.close()
            server.close()

    @skipIfNonUnix
    @gen_test
    def test_init_close_race(self):
        # Regression test for #2367
        #
        # Skipped on windows because of what looks like a bug in the
        # proactor event loop when started and stopped on non-main
        # threads.
        def f():
            for i in range(10):
                loop = IOLoop(make_current=False)
                loop.close()

        yield gen.multi([self.io_loop.run_in_executor(None, f) for i in range(2)])

    def test_explicit_asyncio_loop(self):
        asyncio_loop = asyncio.new_event_loop()
        loop = IOLoop(asyncio_loop=asyncio_loop, make_current=False)
        assert loop.asyncio_loop is asyncio_loop  # type: ignore
        with self.assertRaises(RuntimeError):
            # Can't register two IOLoops with the same asyncio_loop
            IOLoop(asyncio_loop=asyncio_loop, make_current=False)
        loop.close()


# Deliberately not a subclass of AsyncTestCase so the IOLoop isn't
# automatically set as current.
class TestIOLoopCurrent(unittest.TestCase):
    def setUp(self):
        setup_with_context_manager(self, ignore_deprecation())
        self.io_loop = None  # type: typing.Optional[IOLoop]
        IOLoop.clear_current()

    def tearDown(self):
        if self.io_loop is not None:
            self.io_loop.close()

    def test_non_current(self):
        self.io_loop = IOLoop(make_current=False)
        # The new IOLoop is not initially made current.
        self.assertIsNone(IOLoop.current(instance=False))
        # Starting the IOLoop makes it current, and stopping the loop
        # makes it non-current. This process is repeatable.
        for i in range(3):

            def f():
                self.current_io_loop = IOLoop.current()
                assert self.io_loop is not None
                self.io_loop.stop()

            self.io_loop.add_callback(f)
            self.io_loop.start()
            self.assertIs(self.current_io_loop, self.io_loop)
            # Now that the loop is stopped, it is no longer current.
            self.assertIsNone(IOLoop.current(instance=False))

    def test_force_current(self):
        self.io_loop = IOLoop(make_current=True)
        self.assertIs(self.io_loop, IOLoop.current())


class TestIOLoopCurrentAsync(AsyncTestCase):
    def setUp(self):
        super().setUp()
        setup_with_context_manager(self, ignore_deprecation())

    @gen_test
    def test_clear_without_current(self):
        # If there is no current IOLoop, clear_current is a no-op (but
        # should not fail). Use a thread so we see the threading.Local
        # in a pristine state.
        with ThreadPoolExecutor(1) as e:
            yield e.submit(IOLoop.clear_current)


class TestIOLoopFutures(AsyncTestCase):
    def test_add_future_threads(self):
        with futures.ThreadPoolExecutor(1) as pool:

            def dummy():
                pass

            self.io_loop.add_future(
                pool.submit(dummy), lambda future: self.stop(future)
            )
            future = self.wait()
            self.assertTrue(future.done())
            self.assertTrue(future.result() is None)

    @gen_test
    def test_run_in_executor_gen(self):
        event1 = threading.Event()
        event2 = threading.Event()

        def sync_func(self_event, other_event):
            self_event.set()
            other_event.wait()
            # Note that return value doesn't actually do anything,
            # it is just passed through to our final assertion to
            # make sure it is passed through properly.
            return self_event

        # Run two synchronous functions, which would deadlock if not
        # run in parallel.
        res = yield [
            IOLoop.current().run_in_executor(None, sync_func, event1, event2),
            IOLoop.current().run_in_executor(None, sync_func, event2, event1),
        ]

        self.assertEqual([event1, event2], res)

    @gen_test
    def test_run_in_executor_native(self):
        event1 = threading.Event()
        event2 = threading.Event()

        def sync_func(self_event, other_event):
            self_event.set()
            other_event.wait()
            return self_event

        # Go through an async wrapper to ensure that the result of
        # run_in_executor works with await and not just gen.coroutine
        # (simply passing the underlying concurrent future would do that).
        async def async_wrapper(self_event, other_event):
            return await IOLoop.current().run_in_executor(
                None, sync_func, self_event, other_event
            )

        res = yield [async_wrapper(event1, event2), async_wrapper(event2, event1)]

        self.assertEqual([event1, event2], res)

    @gen_test
    def test_set_default_executor(self):
        count = [0]

        class MyExecutor(futures.ThreadPoolExecutor):
            def submit(self, func, *args):
                count[0] += 1
                return super().submit(func, *args)

        event = threading.Event()

        def sync_func():
            event.set()

        executor = MyExecutor(1)
        loop = IOLoop.current()
        loop.set_default_executor(executor)
        yield loop.run_in_executor(None, sync_func)
        self.assertEqual(1, count[0])
        self.assertTrue(event.is_set())


class TestIOLoopRunSync(unittest.TestCase):
    def setUp(self):
        self.io_loop = IOLoop(make_current=False)

    def tearDown(self):
        self.io_loop.close()

    def test_sync_result(self):
        with self.assertRaises(gen.BadYieldError):
            self.io_loop.run_sync(lambda: 42)

    def test_sync_exception(self):
        with self.assertRaises(ZeroDivisionError):
            self.io_loop.run_sync(lambda: 1 / 0)

    def test_async_result(self):
        @gen.coroutine
        def f():
            yield gen.moment
            raise gen.Return(42)

        self.assertEqual(self.io_loop.run_sync(f), 42)

    def test_async_exception(self):
        @gen.coroutine
        def f():
            yield gen.moment
            1 / 0

        with self.assertRaises(ZeroDivisionError):
            self.io_loop.run_sync(f)

    def test_current(self):
        def f():
            self.assertIs(IOLoop.current(), self.io_loop)

        self.io_loop.run_sync(f)

    def test_timeout(self):
        @gen.coroutine
        def f():
            yield gen.sleep(1)

        self.assertRaises(TimeoutError, self.io_loop.run_sync, f, timeout=0.01)

    def test_native_coroutine(self):
        @gen.coroutine
        def f1():
            yield gen.moment

        async def f2():
            await f1()

        self.io_loop.run_sync(f2)


class TestPeriodicCallbackMath(unittest.TestCase):
    def simulate_calls(self, pc, durations):
        """Simulate a series of calls to the PeriodicCallback.

        Pass a list of call durations in seconds (negative values
        work to simulate clock adjustments during the call, or more or
        less equivalently, between calls). This method returns the
        times at which each call would be made.
        """
        calls = []
        now = 1000
        pc._next_timeout = now
        for d in durations:
            pc._update_next(now)
            calls.append(pc._next_timeout)
            now = pc._next_timeout + d
        return calls

    def dummy(self):
        pass

    def test_basic(self):
        pc = PeriodicCallback(self.dummy, 10000)
        self.assertEqual(
            self.simulate_calls(pc, [0] * 5), [1010, 1020, 1030, 1040, 1050]
        )

    def test_overrun(self):
        # If a call runs for too long, we skip entire cycles to get
        # back on schedule.
        call_durations = [9, 9, 10, 11, 20, 20, 35, 35, 0, 0, 0]
        expected = [
            1010,
            1020,
            1030,  # first 3 calls on schedule
            1050,
            1070,  # next 2 delayed one cycle
            1100,
            1130,  # next 2 delayed 2 cycles
            1170,
            1210,  # next 2 delayed 3 cycles
            1220,
            1230,  # then back on schedule.
        ]

        pc = PeriodicCallback(self.dummy, 10000)
        self.assertEqual(self.simulate_calls(pc, call_durations), expected)

    def test_clock_backwards(self):
        pc = PeriodicCallback(self.dummy, 10000)
        # Backwards jumps are ignored, potentially resulting in a
        # slightly slow schedule (although we assume that when
        # time.time() and time.monotonic() are different, time.time()
        # is getting adjusted by NTP and is therefore more accurate)
        self.assertEqual(
            self.simulate_calls(pc, [-2, -1, -3, -2, 0]), [1010, 1020, 1030, 1040, 1050]
        )

        # For big jumps, we should perhaps alter the schedule, but we
        # don't currently. This trace shows that we run callbacks
        # every 10s of time.time(), but the first and second calls are
        # 110s of real time apart because the backwards jump is
        # ignored.
        self.assertEqual(self.simulate_calls(pc, [-100, 0, 0]), [1010, 1020, 1030])

    def test_jitter(self):
        random_times = [0.5, 1, 0, 0.75]
        expected = [1010, 1022.5, 1030, 1041.25]
        call_durations = [0] * len(random_times)
        pc = PeriodicCallback(self.dummy, 10000, jitter=0.5)

        def mock_random():
            return random_times.pop(0)

        with mock.patch("random.random", mock_random):
            self.assertEqual(self.simulate_calls(pc, call_durations), expected)

    def test_timedelta(self):
        pc = PeriodicCallback(lambda: None, datetime.timedelta(minutes=1, seconds=23))
        expected_callback_time = 83000
        self.assertEqual(pc.callback_time, expected_callback_time)


class TestPeriodicCallbackAsync(AsyncTestCase):
    def test_periodic_plain(self):
        count = 0

        def callback() -> None:
            nonlocal count
            count += 1
            if count == 3:
                self.stop()

        pc = PeriodicCallback(callback, 10)
        pc.start()
        self.wait()
        pc.stop()
        self.assertEqual(count, 3)

    def test_periodic_coro(self) -> None:
        counts = [0, 0]

        @gen.coroutine
        def callback() -> "Generator[Future[None], object, None]":
            counts[0] += 1
            yield gen.sleep(0.025)
            counts[1] += 1
            if counts[1] == 3:
                pc.stop()
                self.io_loop.add_callback(self.stop)

        pc = PeriodicCallback(callback, 10)
        pc.start()
        self.wait()
        self.assertEqual(counts[0], 3)
        self.assertEqual(counts[1], 3)

    def test_periodic_async(self) -> None:
        counts = [0, 0]

        async def callback() -> None:
            counts[0] += 1
            await gen.sleep(0.025)
            counts[1] += 1
            if counts[1] == 3:
                pc.stop()
                self.io_loop.add_callback(self.stop)

        pc = PeriodicCallback(callback, 10)
        pc.start()
        self.wait()
        self.assertEqual(counts[0], 3)
        self.assertEqual(counts[1], 3)


class TestIOLoopConfiguration(unittest.TestCase):
    def run_python(self, *statements):
        stmt_list = [
            "from tornado.ioloop import IOLoop",
            "classname = lambda x: x.__class__.__name__",
        ] + list(statements)
        args = [sys.executable, "-c", "; ".join(stmt_list)]
        return native_str(subprocess.check_output(args)).strip()

    def test_default(self):
        # When asyncio is available, it is used by default.
        cls = self.run_python("print(classname(IOLoop.current()))")
        self.assertEqual(cls, "AsyncIOMainLoop")
        cls = self.run_python("print(classname(IOLoop()))")
        self.assertEqual(cls, "AsyncIOLoop")

    def test_asyncio(self):
        cls = self.run_python(
            'IOLoop.configure("tornado.platform.asyncio.AsyncIOLoop")',
            "print(classname(IOLoop.current()))",
        )
        self.assertEqual(cls, "AsyncIOMainLoop")

    def test_asyncio_main(self):
        cls = self.run_python(
            "from tornado.platform.asyncio import AsyncIOMainLoop",
            "AsyncIOMainLoop().install()",
            "print(classname(IOLoop.current()))",
        )
        self.assertEqual(cls, "AsyncIOMainLoop")


if __name__ == "__main__":
    unittest.main()
