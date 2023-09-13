"""Test asyncio support"""
# Copyright (c) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.

import asyncio
import json
import os
import sys
from concurrent.futures import CancelledError
from multiprocessing import Process

import pytest
from pytest import mark

import zmq
import zmq.asyncio as zaio


@pytest.fixture
def Context(event_loop):
    return zaio.Context


def test_socket_class(context):
    with context.socket(zmq.PUSH) as s:
        assert isinstance(s, zaio.Socket)


def test_instance_subclass_first(context):
    actx = zmq.asyncio.Context.instance()
    ctx = zmq.Context.instance()
    ctx.term()
    actx.term()
    assert type(ctx) is zmq.Context
    assert type(actx) is zmq.asyncio.Context


def test_instance_subclass_second(context):
    with zmq.Context.instance() as ctx:
        assert type(ctx) is zmq.Context
        with zmq.asyncio.Context.instance() as actx:
            assert type(actx) is zmq.asyncio.Context


async def test_recv_multipart(context, create_bound_pair):
    a, b = create_bound_pair(zmq.PUSH, zmq.PULL)
    f = b.recv_multipart()
    assert not f.done()
    await a.send(b"hi")
    recvd = await f
    assert recvd == [b"hi"]


async def test_recv(create_bound_pair):
    a, b = create_bound_pair(zmq.PUSH, zmq.PULL)
    f1 = b.recv()
    f2 = b.recv()
    assert not f1.done()
    assert not f2.done()
    await a.send_multipart([b"hi", b"there"])
    recvd = await f2
    assert f1.done()
    assert f1.result() == b"hi"
    assert recvd == b"there"


@mark.skipif(not hasattr(zmq, "RCVTIMEO"), reason="requires RCVTIMEO")
async def test_recv_timeout(push_pull):
    a, b = push_pull
    b.rcvtimeo = 100
    f1 = b.recv()
    b.rcvtimeo = 1000
    f2 = b.recv_multipart()
    with pytest.raises(zmq.Again):
        await f1
    await a.send_multipart([b"hi", b"there"])
    recvd = await f2
    assert f2.done()
    assert recvd == [b"hi", b"there"]


@mark.skipif(not hasattr(zmq, "SNDTIMEO"), reason="requires SNDTIMEO")
async def test_send_timeout(socket):
    s = socket(zmq.PUSH)
    s.sndtimeo = 100
    with pytest.raises(zmq.Again):
        await s.send(b"not going anywhere")


async def test_recv_string(push_pull):
    a, b = push_pull
    f = b.recv_string()
    assert not f.done()
    msg = "πøøπ"
    await a.send_string(msg)
    recvd = await f
    assert f.done()
    assert f.result() == msg
    assert recvd == msg


async def test_recv_json(push_pull):
    a, b = push_pull
    f = b.recv_json()
    assert not f.done()
    obj = dict(a=5)
    await a.send_json(obj)
    recvd = await f
    assert f.done()
    assert f.result() == obj
    assert recvd == obj


async def test_recv_json_cancelled(push_pull):
    a, b = push_pull
    f = b.recv_json()
    assert not f.done()
    f.cancel()
    # cycle eventloop to allow cancel events to fire
    await asyncio.sleep(0)
    obj = dict(a=5)
    await a.send_json(obj)
    # CancelledError change in 3.8 https://bugs.python.org/issue32528
    if sys.version_info < (3, 8):
        with pytest.raises(CancelledError):
            recvd = await f
    else:
        with pytest.raises(asyncio.exceptions.CancelledError):
            recvd = await f
    assert f.done()
    # give it a chance to incorrectly consume the event
    events = await b.poll(timeout=5)
    assert events
    await asyncio.sleep(0)
    # make sure cancelled recv didn't eat up event
    f = b.recv_json()
    recvd = await asyncio.wait_for(f, timeout=5)
    assert recvd == obj


async def test_recv_pyobj(push_pull):
    a, b = push_pull
    f = b.recv_pyobj()
    assert not f.done()
    obj = dict(a=5)
    await a.send_pyobj(obj)
    recvd = await f
    assert f.done()
    assert f.result() == obj
    assert recvd == obj


async def test_custom_serialize(create_bound_pair):
    def serialize(msg):
        frames = []
        frames.extend(msg.get("identities", []))
        content = json.dumps(msg["content"]).encode("utf8")
        frames.append(content)
        return frames

    def deserialize(frames):
        identities = frames[:-1]
        content = json.loads(frames[-1].decode("utf8"))
        return {
            "identities": identities,
            "content": content,
        }

    a, b = create_bound_pair(zmq.DEALER, zmq.ROUTER)

    msg = {
        "content": {
            "a": 5,
            "b": "bee",
        }
    }
    await a.send_serialized(msg, serialize)
    recvd = await b.recv_serialized(deserialize)
    assert recvd["content"] == msg["content"]
    assert recvd["identities"]
    # bounce back, tests identities
    await b.send_serialized(recvd, serialize)
    r2 = await a.recv_serialized(deserialize)
    assert r2["content"] == msg["content"]
    assert not r2["identities"]


async def test_custom_serialize_error(dealer_router):
    a, b = dealer_router

    msg = {
        "content": {
            "a": 5,
            "b": "bee",
        }
    }
    with pytest.raises(TypeError):
        await a.send_serialized(json, json.dumps)

    await a.send(b"not json")
    with pytest.raises(TypeError):
        await b.recv_serialized(json.loads)


async def test_recv_dontwait(push_pull):
    push, pull = push_pull
    f = pull.recv(zmq.DONTWAIT)
    with pytest.raises(zmq.Again):
        await f
    await push.send(b"ping")
    await pull.poll()  # ensure message will be waiting
    f = pull.recv(zmq.DONTWAIT)
    assert f.done()
    msg = await f
    assert msg == b"ping"


async def test_recv_cancel(push_pull):
    a, b = push_pull
    f1 = b.recv()
    f2 = b.recv_multipart()
    assert f1.cancel()
    assert f1.done()
    assert not f2.done()
    await a.send_multipart([b"hi", b"there"])
    recvd = await f2
    assert f1.cancelled()
    assert f2.done()
    assert recvd == [b"hi", b"there"]


async def test_poll(push_pull):
    a, b = push_pull
    f = b.poll(timeout=0)
    await asyncio.sleep(0)
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
    assert recvd == [b"hi", b"there"]


async def test_poll_base_socket(sockets):
    ctx = zmq.Context()
    url = "inproc://test"
    a = ctx.socket(zmq.PUSH)
    b = ctx.socket(zmq.PULL)
    sockets.extend([a, b])
    a.bind(url)
    b.connect(url)

    poller = zaio.Poller()
    poller.register(b, zmq.POLLIN)

    f = poller.poll(timeout=1000)
    assert not f.done()
    a.send_multipart([b"hi", b"there"])
    evt = await f
    assert evt == [(b, zmq.POLLIN)]
    recvd = b.recv_multipart()
    assert recvd == [b"hi", b"there"]


async def test_poll_on_closed_socket(push_pull):
    a, b = push_pull

    f = b.poll(timeout=1)
    b.close()

    # The test might stall if we try to await f directly so instead just make a few
    # passes through the event loop to schedule and execute all callbacks
    for _ in range(5):
        await asyncio.sleep(0)
        if f.cancelled():
            break
    assert f.cancelled()


@pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="Windows does not support polling on files",
)
async def test_poll_raw():
    p = zaio.Poller()
    # make a pipe
    r, w = os.pipe()
    r = os.fdopen(r, "rb")
    w = os.fdopen(w, "wb")

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
    w.write(b"x")
    w.flush()
    evts = await p.poll(timeout=1000)
    evts = dict(evts)
    assert r.fileno() in evts
    assert evts[r.fileno()] == zmq.POLLIN
    assert r.read(1) == b"x"
    r.close()
    w.close()


def test_multiple_loops(push_pull):
    a, b = push_pull

    async def test():
        await a.send(b'buf')
        msg = await b.recv()
        assert msg == b'buf'

    for i in range(3):
        loop = asyncio.new_event_loop()
        loop.run_until_complete(asyncio.wait_for(test(), timeout=10))
        loop.close()


def test_shadow():
    with zmq.Context() as ctx:
        s = ctx.socket(zmq.PULL)
        async_s = zaio.Socket(s)
        assert isinstance(async_s, zaio.Socket)
        assert async_s.underlying == s.underlying
        assert async_s.type == s.type


async def test_poll_leak():
    ctx = zmq.asyncio.Context()
    with ctx, ctx.socket(zmq.PULL) as s:
        assert len(s._recv_futures) == 0
        for i in range(10):
            f = asyncio.ensure_future(s.poll(timeout=1000, flags=zmq.PollEvent.POLLIN))
            f.cancel()
            await asyncio.sleep(0)
        # one more sleep allows further chained cleanup
        await asyncio.sleep(0.1)
        assert len(s._recv_futures) == 0


class ProcessForTeardownTest(Process):
    def run(self):
        """Leave context, socket and event loop upon implicit disposal"""

        actx = zaio.Context.instance()
        socket = actx.socket(zmq.PAIR)
        socket.bind_to_random_port("tcp://127.0.0.1")

        async def never_ending_task(socket):
            await socket.recv()  # never ever receive anything

        loop = asyncio.new_event_loop()
        coro = asyncio.wait_for(never_ending_task(socket), timeout=1)
        try:
            loop.run_until_complete(coro)
        except asyncio.TimeoutError:
            pass  # expected timeout
        else:
            assert False, "never_ending_task was completed unexpectedly"
        finally:
            loop.close()


def test_process_teardown(request):
    proc = ProcessForTeardownTest()
    proc.start()
    request.addfinalizer(proc.terminate)
    proc.join(10)  # starting new Python process may cost a lot
    assert proc.exitcode is not None, "process teardown hangs"
    assert proc.exitcode == 0, f"Python process died with code {proc.exitcode}"
