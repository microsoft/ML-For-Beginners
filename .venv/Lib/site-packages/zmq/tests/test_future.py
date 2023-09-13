# Copyright (c) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.

import json
import os
import sys
from datetime import timedelta

import pytest

gen = pytest.importorskip('tornado.gen')

from tornado.ioloop import IOLoop

import zmq
from zmq.eventloop import future
from zmq.tests import BaseZMQTestCase


class TestFutureSocket(BaseZMQTestCase):
    Context = future.Context

    def setUp(self):
        self.loop = IOLoop(make_current=False)
        super().setUp()

    def tearDown(self):
        super().tearDown()
        if self.loop:
            self.loop.close(all_fds=True)

    def test_socket_class(self):
        s = self.context.socket(zmq.PUSH)
        assert isinstance(s, future.Socket)
        s.close()

    def test_instance_subclass_first(self):
        actx = self.Context.instance()
        ctx = zmq.Context.instance()
        ctx.term()
        actx.term()
        assert type(ctx) is zmq.Context
        assert type(actx) is self.Context

    def test_instance_subclass_second(self):
        ctx = zmq.Context.instance()
        actx = self.Context.instance()
        ctx.term()
        actx.term()
        assert type(ctx) is zmq.Context
        assert type(actx) is self.Context

    def test_recv_multipart(self):
        async def test():
            a, b = self.create_bound_pair(zmq.PUSH, zmq.PULL)
            f = b.recv_multipart()
            assert not f.done()
            await a.send(b"hi")
            recvd = await f
            assert recvd == [b'hi']

        self.loop.run_sync(test)

    def test_recv(self):
        async def test():
            a, b = self.create_bound_pair(zmq.PUSH, zmq.PULL)
            f1 = b.recv()
            f2 = b.recv()
            assert not f1.done()
            assert not f2.done()
            await a.send_multipart([b"hi", b"there"])
            recvd = await f2
            assert f1.done()
            assert f1.result() == b'hi'
            assert recvd == b'there'

        self.loop.run_sync(test)

    def test_recv_cancel(self):
        async def test():
            a, b = self.create_bound_pair(zmq.PUSH, zmq.PULL)
            f1 = b.recv()
            f2 = b.recv_multipart()
            assert f1.cancel()
            assert f1.done()
            assert not f2.done()
            await a.send_multipart([b"hi", b"there"])
            recvd = await f2
            assert f1.cancelled()
            assert f2.done()
            assert recvd == [b'hi', b'there']

        self.loop.run_sync(test)

    @pytest.mark.skipif(not hasattr(zmq, 'RCVTIMEO'), reason="requires RCVTIMEO")
    def test_recv_timeout(self):
        async def test():
            a, b = self.create_bound_pair(zmq.PUSH, zmq.PULL)
            b.rcvtimeo = 100
            f1 = b.recv()
            b.rcvtimeo = 1000
            f2 = b.recv_multipart()
            with pytest.raises(zmq.Again):
                await f1
            await a.send_multipart([b"hi", b"there"])
            recvd = await f2
            assert f2.done()
            assert recvd == [b'hi', b'there']

        self.loop.run_sync(test)

    @pytest.mark.skipif(not hasattr(zmq, 'SNDTIMEO'), reason="requires SNDTIMEO")
    def test_send_timeout(self):
        async def test():
            s = self.socket(zmq.PUSH)
            s.sndtimeo = 100
            with pytest.raises(zmq.Again):
                await s.send(b"not going anywhere")

        self.loop.run_sync(test)

    def test_send_noblock(self):
        async def test():
            s = self.socket(zmq.PUSH)
            with pytest.raises(zmq.Again):
                await s.send(b"not going anywhere", flags=zmq.NOBLOCK)

        self.loop.run_sync(test)

    def test_send_multipart_noblock(self):
        async def test():
            s = self.socket(zmq.PUSH)
            with pytest.raises(zmq.Again):
                await s.send_multipart([b"not going anywhere"], flags=zmq.NOBLOCK)

        self.loop.run_sync(test)

    def test_recv_string(self):
        async def test():
            a, b = self.create_bound_pair(zmq.PUSH, zmq.PULL)
            f = b.recv_string()
            assert not f.done()
            msg = 'πøøπ'
            await a.send_string(msg)
            recvd = await f
            assert f.done()
            assert f.result() == msg
            assert recvd == msg

        self.loop.run_sync(test)

    def test_recv_json(self):
        async def test():
            a, b = self.create_bound_pair(zmq.PUSH, zmq.PULL)
            f = b.recv_json()
            assert not f.done()
            obj = dict(a=5)
            await a.send_json(obj)
            recvd = await f
            assert f.done()
            assert f.result() == obj
            assert recvd == obj

        self.loop.run_sync(test)

    def test_recv_json_cancelled(self):
        async def test():
            a, b = self.create_bound_pair(zmq.PUSH, zmq.PULL)
            f = b.recv_json()
            assert not f.done()
            f.cancel()
            # cycle eventloop to allow cancel events to fire
            await gen.sleep(0)
            obj = dict(a=5)
            await a.send_json(obj)
            with pytest.raises(future.CancelledError):
                recvd = await f
            assert f.done()
            # give it a chance to incorrectly consume the event
            events = await b.poll(timeout=5)
            assert events
            await gen.sleep(0)
            # make sure cancelled recv didn't eat up event
            recvd = await gen.with_timeout(timedelta(seconds=5), b.recv_json())
            assert recvd == obj

        self.loop.run_sync(test)

    def test_recv_pyobj(self):
        async def test():
            a, b = self.create_bound_pair(zmq.PUSH, zmq.PULL)
            f = b.recv_pyobj()
            assert not f.done()
            obj = dict(a=5)
            await a.send_pyobj(obj)
            recvd = await f
            assert f.done()
            assert f.result() == obj
            assert recvd == obj

        self.loop.run_sync(test)

    def test_custom_serialize(self):
        def serialize(msg):
            frames = []
            frames.extend(msg.get('identities', []))
            content = json.dumps(msg['content']).encode('utf8')
            frames.append(content)
            return frames

        def deserialize(frames):
            identities = frames[:-1]
            content = json.loads(frames[-1].decode('utf8'))
            return {
                'identities': identities,
                'content': content,
            }

        async def test():
            a, b = self.create_bound_pair(zmq.DEALER, zmq.ROUTER)

            msg = {
                'content': {
                    'a': 5,
                    'b': 'bee',
                }
            }
            await a.send_serialized(msg, serialize)
            recvd = await b.recv_serialized(deserialize)
            assert recvd['content'] == msg['content']
            assert recvd['identities']
            # bounce back, tests identities
            await b.send_serialized(recvd, serialize)
            r2 = await a.recv_serialized(deserialize)
            assert r2['content'] == msg['content']
            assert not r2['identities']

        self.loop.run_sync(test)

    def test_custom_serialize_error(self):
        async def test():
            a, b = self.create_bound_pair(zmq.DEALER, zmq.ROUTER)

            msg = {
                'content': {
                    'a': 5,
                    'b': 'bee',
                }
            }
            with pytest.raises(TypeError):
                await a.send_serialized(json, json.dumps)

            await a.send(b"not json")
            with pytest.raises(TypeError):
                await b.recv_serialized(json.loads)

        self.loop.run_sync(test)

    def test_poll(self):
        async def test():
            a, b = self.create_bound_pair(zmq.PUSH, zmq.PULL)
            f = b.poll(timeout=0)
            assert f.done()
            assert f.result() == 0

            f = b.poll(timeout=1)
            assert not f.done()
            evt = await f
            assert evt == 0

            f = b.poll(timeout=1000)
            assert not f.done()
            await a.send_multipart([b"hi", b"there"])
            evt = await f
            assert evt == zmq.POLLIN
            recvd = await b.recv_multipart()
            assert recvd == [b'hi', b'there']

        self.loop.run_sync(test)

    @pytest.mark.skipif(
        sys.platform.startswith('win'), reason='Windows unsupported socket type'
    )
    def test_poll_base_socket(self):
        async def test():
            ctx = zmq.Context()
            url = 'inproc://test'
            a = ctx.socket(zmq.PUSH)
            b = ctx.socket(zmq.PULL)
            self.sockets.extend([a, b])
            a.bind(url)
            b.connect(url)

            poller = future.Poller()
            poller.register(b, zmq.POLLIN)

            f = poller.poll(timeout=1000)
            assert not f.done()
            a.send_multipart([b'hi', b'there'])
            evt = await f
            assert evt == [(b, zmq.POLLIN)]
            recvd = b.recv_multipart()
            assert recvd == [b'hi', b'there']
            a.close()
            b.close()
            ctx.term()

        self.loop.run_sync(test)

    def test_close_all_fds(self):
        s = self.socket(zmq.PUB)

        async def attach():
            s._get_loop()

        self.loop.run_sync(attach)
        self.loop.close(all_fds=True)
        self.loop = None  # avoid second close later
        assert s.closed

    @pytest.mark.skipif(
        sys.platform.startswith('win'),
        reason='Windows does not support polling on files',
    )
    def test_poll_raw(self):
        async def test():
            p = future.Poller()
            # make a pipe
            r, w = os.pipe()
            r = os.fdopen(r, 'rb')
            w = os.fdopen(w, 'wb')

            # POLLOUT
            p.register(r, zmq.POLLIN)
            p.register(w, zmq.POLLOUT)
            evts = await p.poll(timeout=1)
            evts = dict(evts)
            assert r.fileno() not in evts
            assert w.fileno() in evts
            assert evts[w.fileno()] == zmq.POLLOUT

            # POLLIN
            p.unregister(w)
            w.write(b'x')
            w.flush()
            evts = await p.poll(timeout=1000)
            evts = dict(evts)
            assert r.fileno() in evts
            assert evts[r.fileno()] == zmq.POLLIN
            assert r.read(1) == b'x'
            r.close()
            w.close()

        self.loop.run_sync(test)
