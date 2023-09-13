import asyncio
import concurrent.futures
import threading

from wsgiref.validate import validator

from tornado.routing import RuleRouter
from tornado.testing import AsyncHTTPTestCase, gen_test
from tornado.wsgi import WSGIContainer


class WSGIAppMixin:
    # TODO: Now that WSGIAdapter is gone, this is a pretty weak test.
    def get_executor(self):
        raise NotImplementedError()

    def get_app(self):
        executor = self.get_executor()
        # The barrier test in DummyExecutorTest will always wait the full
        # value of this timeout, so we don't want it to be too high.
        self.barrier = threading.Barrier(2, timeout=0.3)

        def make_container(app):
            return WSGIContainer(validator(app), executor=executor)

        return RuleRouter(
            [
                ("/simple", make_container(self.simple_wsgi_app)),
                ("/barrier", make_container(self.barrier_wsgi_app)),
                ("/streaming_barrier", make_container(self.streaming_barrier_wsgi_app)),
            ]
        )

    def respond_plain(self, start_response):
        status = "200 OK"
        response_headers = [("Content-Type", "text/plain")]
        start_response(status, response_headers)

    def simple_wsgi_app(self, environ, start_response):
        self.respond_plain(start_response)
        return [b"Hello world!"]

    def barrier_wsgi_app(self, environ, start_response):
        self.respond_plain(start_response)
        try:
            n = self.barrier.wait()
        except threading.BrokenBarrierError:
            return [b"broken barrier"]
        else:
            return [b"ok %d" % n]

    def streaming_barrier_wsgi_app(self, environ, start_response):
        self.respond_plain(start_response)
        yield b"ok "
        try:
            n = self.barrier.wait()
        except threading.BrokenBarrierError:
            yield b"broken barrier"
        else:
            yield b"%d" % n


class WSGIContainerDummyExecutorTest(WSGIAppMixin, AsyncHTTPTestCase):
    def get_executor(self):
        return None

    def test_simple(self):
        response = self.fetch("/simple")
        self.assertEqual(response.body, b"Hello world!")

    @gen_test
    async def test_concurrent_barrier(self):
        self.barrier.reset()
        resps = await asyncio.gather(
            self.http_client.fetch(self.get_url("/barrier")),
            self.http_client.fetch(self.get_url("/barrier")),
        )
        for resp in resps:
            self.assertEqual(resp.body, b"broken barrier")

    @gen_test
    async def test_concurrent_streaming_barrier(self):
        self.barrier.reset()
        resps = await asyncio.gather(
            self.http_client.fetch(self.get_url("/streaming_barrier")),
            self.http_client.fetch(self.get_url("/streaming_barrier")),
        )
        for resp in resps:
            self.assertEqual(resp.body, b"ok broken barrier")


class WSGIContainerThreadPoolTest(WSGIAppMixin, AsyncHTTPTestCase):
    def get_executor(self):
        return concurrent.futures.ThreadPoolExecutor()

    def test_simple(self):
        response = self.fetch("/simple")
        self.assertEqual(response.body, b"Hello world!")

    @gen_test
    async def test_concurrent_barrier(self):
        self.barrier.reset()
        resps = await asyncio.gather(
            self.http_client.fetch(self.get_url("/barrier")),
            self.http_client.fetch(self.get_url("/barrier")),
        )
        self.assertEqual([b"ok 0", b"ok 1"], sorted([resp.body for resp in resps]))

    @gen_test
    async def test_concurrent_streaming_barrier(self):
        self.barrier.reset()
        resps = await asyncio.gather(
            self.http_client.fetch(self.get_url("/streaming_barrier")),
            self.http_client.fetch(self.get_url("/streaming_barrier")),
        )
        self.assertEqual([b"ok 0", b"ok 1"], sorted([resp.body for resp in resps]))
