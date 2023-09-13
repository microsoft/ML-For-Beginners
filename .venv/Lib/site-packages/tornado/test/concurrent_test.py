#
# Copyright 2012 Facebook
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
from concurrent import futures
import logging
import re
import socket
import typing
import unittest

from tornado.concurrent import (
    Future,
    run_on_executor,
    future_set_result_unless_cancelled,
)
from tornado.escape import utf8, to_unicode
from tornado import gen
from tornado.iostream import IOStream
from tornado.tcpserver import TCPServer
from tornado.testing import AsyncTestCase, bind_unused_port, gen_test


class MiscFutureTest(AsyncTestCase):
    def test_future_set_result_unless_cancelled(self):
        fut = Future()  # type: Future[int]
        future_set_result_unless_cancelled(fut, 42)
        self.assertEqual(fut.result(), 42)
        self.assertFalse(fut.cancelled())

        fut = Future()
        fut.cancel()
        is_cancelled = fut.cancelled()
        future_set_result_unless_cancelled(fut, 42)
        self.assertEqual(fut.cancelled(), is_cancelled)
        if not is_cancelled:
            self.assertEqual(fut.result(), 42)


# The following series of classes demonstrate and test various styles
# of use, with and without generators and futures.


class CapServer(TCPServer):
    @gen.coroutine
    def handle_stream(self, stream, address):
        data = yield stream.read_until(b"\n")
        data = to_unicode(data)
        if data == data.upper():
            stream.write(b"error\talready capitalized\n")
        else:
            # data already has \n
            stream.write(utf8("ok\t%s" % data.upper()))
        stream.close()


class CapError(Exception):
    pass


class BaseCapClient(object):
    def __init__(self, port):
        self.port = port

    def process_response(self, data):
        m = re.match("(.*)\t(.*)\n", to_unicode(data))
        if m is None:
            raise Exception("did not match")
        status, message = m.groups()
        if status == "ok":
            return message
        else:
            raise CapError(message)


class GeneratorCapClient(BaseCapClient):
    @gen.coroutine
    def capitalize(self, request_data):
        logging.debug("capitalize")
        stream = IOStream(socket.socket())
        logging.debug("connecting")
        yield stream.connect(("127.0.0.1", self.port))
        stream.write(utf8(request_data + "\n"))
        logging.debug("reading")
        data = yield stream.read_until(b"\n")
        logging.debug("returning")
        stream.close()
        raise gen.Return(self.process_response(data))


class ClientTestMixin(object):
    client_class = None  # type: typing.Callable

    def setUp(self):
        super().setUp()  # type: ignore
        self.server = CapServer()
        sock, port = bind_unused_port()
        self.server.add_sockets([sock])
        self.client = self.client_class(port=port)

    def tearDown(self):
        self.server.stop()
        super().tearDown()  # type: ignore

    def test_future(self: typing.Any):
        future = self.client.capitalize("hello")
        self.io_loop.add_future(future, self.stop)
        self.wait()
        self.assertEqual(future.result(), "HELLO")

    def test_future_error(self: typing.Any):
        future = self.client.capitalize("HELLO")
        self.io_loop.add_future(future, self.stop)
        self.wait()
        self.assertRaisesRegex(CapError, "already capitalized", future.result)  # type: ignore

    def test_generator(self: typing.Any):
        @gen.coroutine
        def f():
            result = yield self.client.capitalize("hello")
            self.assertEqual(result, "HELLO")

        self.io_loop.run_sync(f)

    def test_generator_error(self: typing.Any):
        @gen.coroutine
        def f():
            with self.assertRaisesRegex(CapError, "already capitalized"):
                yield self.client.capitalize("HELLO")

        self.io_loop.run_sync(f)


class GeneratorClientTest(ClientTestMixin, AsyncTestCase):
    client_class = GeneratorCapClient


class RunOnExecutorTest(AsyncTestCase):
    @gen_test
    def test_no_calling(self):
        class Object(object):
            def __init__(self):
                self.executor = futures.thread.ThreadPoolExecutor(1)

            @run_on_executor
            def f(self):
                return 42

        o = Object()
        answer = yield o.f()
        self.assertEqual(answer, 42)

    @gen_test
    def test_call_with_no_args(self):
        class Object(object):
            def __init__(self):
                self.executor = futures.thread.ThreadPoolExecutor(1)

            @run_on_executor()
            def f(self):
                return 42

        o = Object()
        answer = yield o.f()
        self.assertEqual(answer, 42)

    @gen_test
    def test_call_with_executor(self):
        class Object(object):
            def __init__(self):
                self.__executor = futures.thread.ThreadPoolExecutor(1)

            @run_on_executor(executor="_Object__executor")
            def f(self):
                return 42

        o = Object()
        answer = yield o.f()
        self.assertEqual(answer, 42)

    @gen_test
    def test_async_await(self):
        class Object(object):
            def __init__(self):
                self.executor = futures.thread.ThreadPoolExecutor(1)

            @run_on_executor()
            def f(self):
                return 42

        o = Object()

        async def f():
            answer = await o.f()
            return answer

        result = yield f()
        self.assertEqual(result, 42)


if __name__ == "__main__":
    unittest.main()
