# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.

import copy
import gc
import os
import sys
import time
from queue import Queue
from threading import Event, Thread
from unittest import mock

import pytest
from pytest import mark

import zmq
from zmq.tests import PYPY, BaseZMQTestCase, GreenTest, SkipTest


class KwargTestSocket(zmq.Socket):
    test_kwarg_value = None

    def __init__(self, *args, **kwargs):
        self.test_kwarg_value = kwargs.pop('test_kwarg', None)
        super().__init__(*args, **kwargs)


class KwargTestContext(zmq.Context):
    _socket_class = KwargTestSocket


class TestContext(BaseZMQTestCase):
    def test_init(self):
        c1 = self.Context()
        assert isinstance(c1, self.Context)
        c1.term()
        c2 = self.Context()
        assert isinstance(c2, self.Context)
        c2.term()
        c3 = self.Context()
        assert isinstance(c3, self.Context)
        c3.term()

    _repr_cls = "zmq.Context"

    def test_repr(self):
        with self.Context() as ctx:
            assert f'{self._repr_cls}()' in repr(ctx)
            assert 'closed' not in repr(ctx)
            with ctx.socket(zmq.PUSH) as push:
                assert f'{self._repr_cls}(1 socket)' in repr(ctx)
                with ctx.socket(zmq.PULL) as pull:
                    assert f'{self._repr_cls}(2 sockets)' in repr(ctx)
        assert f'{self._repr_cls}()' in repr(ctx)
        assert 'closed' in repr(ctx)

    def test_dir(self):
        ctx = self.Context()
        assert 'socket' in dir(ctx)
        if zmq.zmq_version_info() > (3,):
            assert 'IO_THREADS' in dir(ctx)
        ctx.term()

    @mark.skipif(mock is None, reason="requires unittest.mock")
    def test_mockable(self):
        m = mock.Mock(spec=self.context)

    def test_term(self):
        c = self.Context()
        c.term()
        assert c.closed

    def test_context_manager(self):
        with pytest.warns(ResourceWarning):
            with self.Context() as ctx:
                s = ctx.socket(zmq.PUSH)
        # context exit destroys sockets
        assert s.closed
        assert ctx.closed

    def test_fail_init(self):
        self.assertRaisesErrno(zmq.EINVAL, self.Context, -1)

    def test_term_hang(self):
        rep, req = self.create_bound_pair(zmq.ROUTER, zmq.DEALER)
        req.setsockopt(zmq.LINGER, 0)
        req.send(b'hello', copy=False)
        req.close()
        rep.close()
        self.context.term()

    def test_instance(self):
        ctx = self.Context.instance()
        c2 = self.Context.instance(io_threads=2)
        assert c2 is ctx
        c2.term()
        c3 = self.Context.instance()
        c4 = self.Context.instance()
        assert not c3 is c2
        assert not c3.closed
        assert c3 is c4

    def test_instance_subclass_first(self):
        self.context.term()

        class SubContext(zmq.Context):
            pass

        sctx = SubContext.instance()
        ctx = zmq.Context.instance()
        ctx.term()
        sctx.term()
        assert type(ctx) is zmq.Context
        assert type(sctx) is SubContext

    def test_instance_subclass_second(self):
        self.context.term()

        class SubContextInherit(zmq.Context):
            pass

        class SubContextNoInherit(zmq.Context):
            _instance = None

        ctx = zmq.Context.instance()
        sctx = SubContextInherit.instance()
        sctx2 = SubContextNoInherit.instance()
        ctx.term()
        sctx.term()
        sctx2.term()
        assert type(ctx) is zmq.Context
        assert type(sctx) is zmq.Context
        assert type(sctx2) is SubContextNoInherit

    def test_instance_threadsafe(self):
        self.context.term()  # clear default context

        q = Queue()

        # slow context initialization,
        # to ensure that we are both trying to create one at the same time
        class SlowContext(self.Context):
            def __init__(self, *a, **kw):
                time.sleep(1)
                super().__init__(*a, **kw)

        def f():
            q.put(SlowContext.instance())

        # call ctx.instance() in several threads at once
        N = 16
        threads = [Thread(target=f) for i in range(N)]
        [t.start() for t in threads]
        # also call it in the main thread (not first)
        ctx = SlowContext.instance()
        assert isinstance(ctx, SlowContext)
        # check that all the threads got the same context
        for i in range(N):
            thread_ctx = q.get(timeout=5)
            assert thread_ctx is ctx
        # cleanup
        ctx.term()
        [t.join(timeout=5) for t in threads]

    def test_socket_passes_kwargs(self):
        test_kwarg_value = 'testing one two three'
        with KwargTestContext() as ctx:
            with ctx.socket(zmq.DEALER, test_kwarg=test_kwarg_value) as socket:
                assert socket.test_kwarg_value is test_kwarg_value

    def test_socket_class_arg(self):
        class CustomSocket(zmq.Socket):
            pass

        with self.Context() as ctx:
            with ctx.socket(zmq.PUSH, socket_class=CustomSocket) as s:
                assert isinstance(s, CustomSocket)

    def test_many_sockets(self):
        """opening and closing many sockets shouldn't cause problems"""
        ctx = self.Context()
        for i in range(16):
            sockets = [ctx.socket(zmq.REP) for i in range(65)]
            [s.close() for s in sockets]
            # give the reaper a chance
            time.sleep(1e-2)
        ctx.term()

    def test_sockopts(self):
        """setting socket options with ctx attributes"""
        ctx = self.Context()
        ctx.linger = 5
        assert ctx.linger == 5
        s = ctx.socket(zmq.REQ)
        assert s.linger == 5
        assert s.getsockopt(zmq.LINGER) == 5
        s.close()
        # check that subscribe doesn't get set on sockets that don't subscribe:
        ctx.subscribe = b''
        s = ctx.socket(zmq.REQ)
        s.close()

        ctx.term()

    @mark.skipif(sys.platform.startswith('win'), reason='Segfaults on Windows')
    def test_destroy(self):
        """Context.destroy should close sockets"""
        ctx = self.Context()
        sockets = [ctx.socket(zmq.REP) for i in range(65)]

        # close half of the sockets
        [s.close() for s in sockets[::2]]

        ctx.destroy()
        # reaper is not instantaneous
        time.sleep(1e-2)
        for s in sockets:
            assert s.closed

    def test_destroy_linger(self):
        """Context.destroy should set linger on closing sockets"""
        req, rep = self.create_bound_pair(zmq.REQ, zmq.REP)
        req.send(b'hi')
        time.sleep(1e-2)
        self.context.destroy(linger=0)
        # reaper is not instantaneous
        time.sleep(1e-2)
        for s in (req, rep):
            assert s.closed

    def test_term_noclose(self):
        """Context.term won't close sockets"""
        ctx = self.Context()
        s = ctx.socket(zmq.REQ)
        assert not s.closed
        t = Thread(target=ctx.term)
        t.start()
        t.join(timeout=0.1)
        assert t.is_alive(), "Context should be waiting"
        s.close()
        t.join(timeout=0.1)
        assert not t.is_alive(), "Context should have closed"

    def test_gc(self):
        """test close&term by garbage collection alone"""
        if PYPY:
            raise SkipTest("GC doesn't work ")

        # test credit @dln (GH #137):
        def gcf():
            def inner():
                ctx = self.Context()
                ctx.socket(zmq.PUSH)

            # can't seem to catch these with pytest.warns(ResourceWarning)
            inner()
            gc.collect()

        t = Thread(target=gcf)
        t.start()
        t.join(timeout=1)
        assert not t.is_alive(), "Garbage collection should have cleaned up context"

    def test_cyclic_destroy(self):
        """ctx.destroy should succeed when cyclic ref prevents gc"""

        # test credit @dln (GH #137):
        class CyclicReference:
            def __init__(self, parent=None):
                self.parent = parent

            def crash(self, sock):
                self.sock = sock
                self.child = CyclicReference(self)

        def crash_zmq():
            ctx = self.Context()
            sock = ctx.socket(zmq.PULL)
            c = CyclicReference()
            c.crash(sock)
            ctx.destroy()

        crash_zmq()

    def test_term_thread(self):
        """ctx.term should not crash active threads (#139)"""
        ctx = self.Context()
        evt = Event()
        evt.clear()

        def block():
            s = ctx.socket(zmq.REP)
            s.bind_to_random_port('tcp://127.0.0.1')
            evt.set()
            try:
                s.recv()
            except zmq.ZMQError as e:
                assert e.errno == zmq.ETERM
                return
            finally:
                s.close()
            self.fail("recv should have been interrupted with ETERM")

        t = Thread(target=block)
        t.start()

        evt.wait(1)
        assert evt.is_set(), "sync event never fired"
        time.sleep(0.01)
        ctx.term()
        t.join(timeout=1)
        assert not t.is_alive(), "term should have interrupted s.recv()"

    def test_destroy_no_sockets(self):
        ctx = self.Context()
        s = ctx.socket(zmq.PUB)
        s.bind_to_random_port('tcp://127.0.0.1')
        s.close()
        ctx.destroy()
        assert s.closed
        assert ctx.closed

    def test_ctx_opts(self):
        if zmq.zmq_version_info() < (3,):
            raise SkipTest("context options require libzmq 3")
        ctx = self.Context()
        ctx.set(zmq.MAX_SOCKETS, 2)
        assert ctx.get(zmq.MAX_SOCKETS) == 2
        ctx.max_sockets = 100
        assert ctx.max_sockets == 100
        assert ctx.get(zmq.MAX_SOCKETS) == 100

    def test_copy(self):
        c1 = self.Context()
        c2 = copy.copy(c1)
        c2b = copy.deepcopy(c1)
        c3 = copy.deepcopy(c2)
        assert c2._shadow
        assert c3._shadow
        assert c1.underlying == c2.underlying
        assert c1.underlying == c3.underlying
        assert c1.underlying == c2b.underlying
        s = c3.socket(zmq.PUB)
        s.close()
        c1.term()

    def test_shadow(self):
        ctx = self.Context()
        ctx2 = self.Context.shadow(ctx.underlying)
        assert ctx.underlying == ctx2.underlying
        s = ctx.socket(zmq.PUB)
        s.close()
        del ctx2
        assert not ctx.closed
        s = ctx.socket(zmq.PUB)
        ctx2 = self.Context.shadow(ctx)
        with ctx2.socket(zmq.PUB) as s2:
            pass

        assert s2.closed
        assert not s.closed
        s.close()

        ctx3 = self.Context(ctx)
        assert ctx3.underlying == ctx.underlying
        del ctx3
        assert not ctx.closed

        ctx.term()
        self.assertRaisesErrno(zmq.EFAULT, ctx2.socket, zmq.PUB)
        del ctx2

    def test_shadow_pyczmq(self):
        try:
            from pyczmq import zctx, zsocket, zstr
        except Exception:
            raise SkipTest("Requires pyczmq")

        ctx = zctx.new()
        a = zsocket.new(ctx, zmq.PUSH)
        zsocket.bind(a, "inproc://a")
        ctx2 = self.Context.shadow_pyczmq(ctx)
        b = ctx2.socket(zmq.PULL)
        b.connect("inproc://a")
        zstr.send(a, b'hi')
        rcvd = self.recv(b)
        assert rcvd == b'hi'
        b.close()

    @mark.skipif(sys.platform.startswith('win'), reason='No fork on Windows')
    def test_fork_instance(self):
        ctx = self.Context.instance()
        parent_ctx_id = id(ctx)
        r_fd, w_fd = os.pipe()
        reader = os.fdopen(r_fd, 'r')
        child_pid = os.fork()
        if child_pid == 0:
            ctx = self.Context.instance()
            writer = os.fdopen(w_fd, 'w')
            child_ctx_id = id(ctx)
            ctx.term()
            writer.write(str(child_ctx_id) + "\n")
            writer.flush()
            writer.close()
            os._exit(0)
        else:
            os.close(w_fd)

        child_id_s = reader.readline()
        reader.close()
        assert child_id_s
        assert int(child_id_s) != parent_ctx_id
        ctx.term()


if False:  # disable green context tests

    class TestContextGreen(GreenTest, TestContext):
        """gevent subclass of context tests"""

        # skip tests that use real threads:
        test_gc = GreenTest.skip_green
        test_term_thread = GreenTest.skip_green
        test_destroy_linger = GreenTest.skip_green
        _repr_cls = "zmq.green.Context"
