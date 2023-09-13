import socket
import subprocess
import sys
import textwrap
import unittest

from tornado import gen
from tornado.iostream import IOStream
from tornado.log import app_log
from tornado.tcpserver import TCPServer
from tornado.test.util import skipIfNonUnix
from tornado.testing import AsyncTestCase, ExpectLog, bind_unused_port, gen_test

from typing import Tuple


class TCPServerTest(AsyncTestCase):
    @gen_test
    def test_handle_stream_coroutine_logging(self):
        # handle_stream may be a coroutine and any exception in its
        # Future will be logged.
        class TestServer(TCPServer):
            @gen.coroutine
            def handle_stream(self, stream, address):
                yield stream.read_bytes(len(b"hello"))
                stream.close()
                1 / 0

        server = client = None
        try:
            sock, port = bind_unused_port()
            server = TestServer()
            server.add_socket(sock)
            client = IOStream(socket.socket())
            with ExpectLog(app_log, "Exception in callback"):
                yield client.connect(("localhost", port))
                yield client.write(b"hello")
                yield client.read_until_close()
                yield gen.moment
        finally:
            if server is not None:
                server.stop()
            if client is not None:
                client.close()

    @gen_test
    def test_handle_stream_native_coroutine(self):
        # handle_stream may be a native coroutine.

        class TestServer(TCPServer):
            async def handle_stream(self, stream, address):
                stream.write(b"data")
                stream.close()

        sock, port = bind_unused_port()
        server = TestServer()
        server.add_socket(sock)
        client = IOStream(socket.socket())
        yield client.connect(("localhost", port))
        result = yield client.read_until_close()
        self.assertEqual(result, b"data")
        server.stop()
        client.close()

    def test_stop_twice(self):
        sock, port = bind_unused_port()
        server = TCPServer()
        server.add_socket(sock)
        server.stop()
        server.stop()

    @gen_test
    def test_stop_in_callback(self):
        # Issue #2069: calling server.stop() in a loop callback should not
        # raise EBADF when the loop handles other server connection
        # requests in the same loop iteration

        class TestServer(TCPServer):
            @gen.coroutine
            def handle_stream(self, stream, address):
                server.stop()  # type: ignore
                yield stream.read_until_close()

        sock, port = bind_unused_port()
        server = TestServer()
        server.add_socket(sock)
        server_addr = ("localhost", port)
        N = 40
        clients = [IOStream(socket.socket()) for i in range(N)]
        connected_clients = []

        @gen.coroutine
        def connect(c):
            try:
                yield c.connect(server_addr)
            except EnvironmentError:
                pass
            else:
                connected_clients.append(c)

        yield [connect(c) for c in clients]

        self.assertGreater(len(connected_clients), 0, "all clients failed connecting")
        try:
            if len(connected_clients) == N:
                # Ideally we'd make the test deterministic, but we're testing
                # for a race condition in combination with the system's TCP stack...
                self.skipTest(
                    "at least one client should fail connecting "
                    "for the test to be meaningful"
                )
        finally:
            for c in connected_clients:
                c.close()

        # Here tearDown() would re-raise the EBADF encountered in the IO loop


@skipIfNonUnix
class TestMultiprocess(unittest.TestCase):
    # These tests verify that the two multiprocess examples from the
    # TCPServer docs work. Both tests start a server with three worker
    # processes, each of which prints its task id to stdout (a single
    # byte, so we don't have to worry about atomicity of the shared
    # stdout stream) and then exits.
    def run_subproc(self, code: str) -> Tuple[str, str]:
        try:
            result = subprocess.run(
                [sys.executable, "-Werror::DeprecationWarning"],
                capture_output=True,
                input=code,
                encoding="utf8",
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Process returned {e.returncode} stdout={e.stdout} stderr={e.stderr}"
            ) from e
        return result.stdout, result.stderr

    def test_listen_single(self):
        # As a sanity check, run the single-process version through this test
        # harness too.
        code = textwrap.dedent(
            """
            import asyncio
            from tornado.tcpserver import TCPServer

            async def main():
                server = TCPServer()
                server.listen(0, address='127.0.0.1')

            asyncio.run(main())
            print('012', end='')
        """
        )
        out, err = self.run_subproc(code)
        self.assertEqual("".join(sorted(out)), "012")
        self.assertEqual(err, "")

    def test_bind_start(self):
        code = textwrap.dedent(
            """
            import warnings

            from tornado.ioloop import IOLoop
            from tornado.process import task_id
            from tornado.tcpserver import TCPServer

            warnings.simplefilter("ignore", DeprecationWarning)

            server = TCPServer()
            server.bind(0, address='127.0.0.1')
            server.start(3)
            IOLoop.current().run_sync(lambda: None)
            print(task_id(), end='')
        """
        )
        out, err = self.run_subproc(code)
        self.assertEqual("".join(sorted(out)), "012")
        self.assertEqual(err, "")

    def test_add_sockets(self):
        code = textwrap.dedent(
            """
            import asyncio
            from tornado.netutil import bind_sockets
            from tornado.process import fork_processes, task_id
            from tornado.ioloop import IOLoop
            from tornado.tcpserver import TCPServer

            sockets = bind_sockets(0, address='127.0.0.1')
            fork_processes(3)
            async def post_fork_main():
                server = TCPServer()
                server.add_sockets(sockets)
            asyncio.run(post_fork_main())
            print(task_id(), end='')
        """
        )
        out, err = self.run_subproc(code)
        self.assertEqual("".join(sorted(out)), "012")
        self.assertEqual(err, "")

    def test_listen_multi_reuse_port(self):
        code = textwrap.dedent(
            """
            import asyncio
            import socket
            from tornado.netutil import bind_sockets
            from tornado.process import task_id, fork_processes
            from tornado.tcpserver import TCPServer

            # Pick an unused port which we will be able to bind to multiple times.
            (sock,) = bind_sockets(0, address='127.0.0.1',
                family=socket.AF_INET, reuse_port=True)
            port = sock.getsockname()[1]

            fork_processes(3)

            async def main():
                server = TCPServer()
                server.listen(port, address='127.0.0.1', reuse_port=True)
            asyncio.run(main())
            print(task_id(), end='')
            """
        )
        out, err = self.run_subproc(code)
        self.assertEqual("".join(sorted(out)), "012")
        self.assertEqual(err, "")
