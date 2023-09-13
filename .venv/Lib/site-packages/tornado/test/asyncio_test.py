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

import asyncio
import unittest
import warnings

from concurrent.futures import ThreadPoolExecutor
from tornado import gen
from tornado.ioloop import IOLoop
from tornado.platform.asyncio import (
    AsyncIOLoop,
    to_asyncio_future,
    AnyThreadEventLoopPolicy,
)
from tornado.testing import AsyncTestCase, gen_test


class AsyncIOLoopTest(AsyncTestCase):
    @property
    def asyncio_loop(self):
        return self.io_loop.asyncio_loop  # type: ignore

    def test_asyncio_callback(self):
        # Basic test that the asyncio loop is set up correctly.
        async def add_callback():
            asyncio.get_event_loop().call_soon(self.stop)

        self.asyncio_loop.run_until_complete(add_callback())
        self.wait()

    @gen_test
    def test_asyncio_future(self):
        # Test that we can yield an asyncio future from a tornado coroutine.
        # Without 'yield from', we must wrap coroutines in ensure_future,
        # which was introduced during Python 3.4, deprecating the prior "async".
        if hasattr(asyncio, "ensure_future"):
            ensure_future = asyncio.ensure_future
        else:
            # async is a reserved word in Python 3.7
            ensure_future = getattr(asyncio, "async")

        x = yield ensure_future(
            asyncio.get_event_loop().run_in_executor(None, lambda: 42)
        )
        self.assertEqual(x, 42)

    @gen_test
    def test_asyncio_yield_from(self):
        @gen.coroutine
        def f():
            event_loop = asyncio.get_event_loop()
            x = yield from event_loop.run_in_executor(None, lambda: 42)
            return x

        result = yield f()
        self.assertEqual(result, 42)

    def test_asyncio_adapter(self):
        # This test demonstrates that when using the asyncio coroutine
        # runner (i.e. run_until_complete), the to_asyncio_future
        # adapter is needed. No adapter is needed in the other direction,
        # as demonstrated by other tests in the package.
        @gen.coroutine
        def tornado_coroutine():
            yield gen.moment
            raise gen.Return(42)

        async def native_coroutine_without_adapter():
            return await tornado_coroutine()

        async def native_coroutine_with_adapter():
            return await to_asyncio_future(tornado_coroutine())

        # Use the adapter, but two degrees from the tornado coroutine.
        async def native_coroutine_with_adapter2():
            return await to_asyncio_future(native_coroutine_without_adapter())

        # Tornado supports native coroutines both with and without adapters
        self.assertEqual(self.io_loop.run_sync(native_coroutine_without_adapter), 42)
        self.assertEqual(self.io_loop.run_sync(native_coroutine_with_adapter), 42)
        self.assertEqual(self.io_loop.run_sync(native_coroutine_with_adapter2), 42)

        # Asyncio only supports coroutines that yield asyncio-compatible
        # Futures (which our Future is since 5.0).
        self.assertEqual(
            self.asyncio_loop.run_until_complete(native_coroutine_without_adapter()),
            42,
        )
        self.assertEqual(
            self.asyncio_loop.run_until_complete(native_coroutine_with_adapter()),
            42,
        )
        self.assertEqual(
            self.asyncio_loop.run_until_complete(native_coroutine_with_adapter2()),
            42,
        )


class LeakTest(unittest.TestCase):
    def setUp(self):
        # Trigger a cleanup of the mapping so we start with a clean slate.
        AsyncIOLoop(make_current=False).close()
        # If we don't clean up after ourselves other tests may fail on
        # py34.
        self.orig_policy = asyncio.get_event_loop_policy()
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

    def tearDown(self):
        try:
            loop = asyncio.get_event_loop_policy().get_event_loop()
        except Exception:
            # We may not have a current event loop at this point.
            pass
        else:
            loop.close()
        asyncio.set_event_loop_policy(self.orig_policy)

    def test_ioloop_close_leak(self):
        orig_count = len(IOLoop._ioloop_for_asyncio)
        for i in range(10):
            # Create and close an AsyncIOLoop using Tornado interfaces.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                loop = AsyncIOLoop()
                loop.close()
        new_count = len(IOLoop._ioloop_for_asyncio) - orig_count
        self.assertEqual(new_count, 0)

    def test_asyncio_close_leak(self):
        orig_count = len(IOLoop._ioloop_for_asyncio)
        for i in range(10):
            # Create and close an AsyncIOMainLoop using asyncio interfaces.
            loop = asyncio.new_event_loop()
            loop.call_soon(IOLoop.current)
            loop.call_soon(loop.stop)
            loop.run_forever()
            loop.close()
        new_count = len(IOLoop._ioloop_for_asyncio) - orig_count
        # Because the cleanup is run on new loop creation, we have one
        # dangling entry in the map (but only one).
        self.assertEqual(new_count, 1)


class AnyThreadEventLoopPolicyTest(unittest.TestCase):
    def setUp(self):
        self.orig_policy = asyncio.get_event_loop_policy()
        self.executor = ThreadPoolExecutor(1)

    def tearDown(self):
        asyncio.set_event_loop_policy(self.orig_policy)
        self.executor.shutdown()

    def get_event_loop_on_thread(self):
        def get_and_close_event_loop():
            """Get the event loop. Close it if one is returned.

            Returns the (closed) event loop. This is a silly thing
            to do and leaves the thread in a broken state, but it's
            enough for this test. Closing the loop avoids resource
            leak warnings.
            """
            loop = asyncio.get_event_loop()
            loop.close()
            return loop

        future = self.executor.submit(get_and_close_event_loop)
        return future.result()

    def test_asyncio_accessor(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            # With the default policy, non-main threads don't get an event
            # loop.
            self.assertRaises(
                RuntimeError, self.executor.submit(asyncio.get_event_loop).result
            )
            # Set the policy and we can get a loop.
            asyncio.set_event_loop_policy(AnyThreadEventLoopPolicy())
            self.assertIsInstance(
                self.executor.submit(asyncio.get_event_loop).result(),
                asyncio.AbstractEventLoop,
            )
            # Clean up to silence leak warnings. Always use asyncio since
            # IOLoop doesn't (currently) close the underlying loop.
            self.executor.submit(lambda: asyncio.get_event_loop().close()).result()  # type: ignore

    def test_tornado_accessor(self):
        # Tornado's IOLoop.current() API can create a loop for any thread,
        # regardless of this event loop policy.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self.assertIsInstance(self.executor.submit(IOLoop.current).result(), IOLoop)
            # Clean up to silence leak warnings. Always use asyncio since
            # IOLoop doesn't (currently) close the underlying loop.
            self.executor.submit(lambda: asyncio.get_event_loop().close()).result()  # type: ignore

            asyncio.set_event_loop_policy(AnyThreadEventLoopPolicy())
            self.assertIsInstance(self.executor.submit(IOLoop.current).result(), IOLoop)
            self.executor.submit(lambda: asyncio.get_event_loop().close()).result()  # type: ignore
