# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.


import asyncio
import logging
import warnings

import pytest

import zmq
import zmq.asyncio

try:
    import tornado

    from zmq.eventloop import zmqstream
except ImportError:
    tornado = None  # type: ignore


pytestmark = pytest.mark.usefixtures("io_loop")


@pytest.fixture
async def push_pull(socket):
    push = zmqstream.ZMQStream(socket(zmq.PUSH))
    pull = zmqstream.ZMQStream(socket(zmq.PULL))
    port = push.bind_to_random_port('tcp://127.0.0.1')
    pull.connect('tcp://127.0.0.1:%i' % port)
    return (push, pull)


@pytest.fixture
def push(push_pull):
    push, pull = push_pull
    return push


@pytest.fixture
def pull(push_pull):
    push, pull = push_pull
    return pull


async def test_callable_check(pull):
    """Ensure callable check works."""

    pull.on_send(lambda *args: None)
    pull.on_recv(lambda *args: None)
    with pytest.raises(AssertionError):
        pull.on_recv(1)
    with pytest.raises(AssertionError):
        pull.on_send(1)
    with pytest.raises(AssertionError):
        pull.on_recv(zmq)


async def test_on_recv_basic(push, pull):
    sent = [b'basic']
    push.send_multipart(sent)
    f = asyncio.Future()

    def callback(msg):
        f.set_result(msg)

    pull.on_recv(callback)
    recvd = await asyncio.wait_for(f, timeout=5)
    assert recvd == sent


async def test_on_recv_wake(push, pull):
    sent = [b'wake']

    f = asyncio.Future()
    pull.on_recv(f.set_result)
    await asyncio.sleep(0.5)
    push.send_multipart(sent)
    recvd = await asyncio.wait_for(f, timeout=5)
    assert recvd == sent


async def test_on_recv_async(push, pull):
    if tornado.version_info < (5,):
        pytest.skip()
    sent = [b'wake']

    f = asyncio.Future()

    async def callback(msg):
        await asyncio.sleep(0.1)
        f.set_result(msg)

    pull.on_recv(callback)
    await asyncio.sleep(0.5)
    push.send_multipart(sent)
    recvd = await asyncio.wait_for(f, timeout=5)
    assert recvd == sent


async def test_on_recv_async_error(push, pull, caplog):
    sent = [b'wake']

    f = asyncio.Future()

    async def callback(msg):
        f.set_result(msg)
        1 / 0

    pull.on_recv(callback)
    await asyncio.sleep(0.1)
    with caplog.at_level(logging.ERROR, logger=zmqstream.gen_log.name):
        push.send_multipart(sent)
        recvd = await asyncio.wait_for(f, timeout=5)
        assert recvd == sent
        # logging error takes a tick later
        await asyncio.sleep(0.5)

    messages = [
        x.message
        for x in caplog.get_records("call")
        if x.name == zmqstream.gen_log.name
    ]
    assert "Uncaught exception in ZMQStream callback" in "\n".join(messages)


async def test_shadow_socket(context):
    with context.socket(zmq.PUSH, socket_class=zmq.asyncio.Socket) as socket:
        with pytest.warns(RuntimeWarning):
            stream = zmqstream.ZMQStream(socket)
        assert type(stream.socket) is zmq.Socket
        assert stream.socket.underlying == socket.underlying
        stream.close()


async def test_shadow_socket_close(context, caplog):
    with context.socket(zmq.PUSH) as push, context.socket(zmq.PULL) as pull:
        push.linger = pull.linger = 0
        port = push.bind_to_random_port('tcp://127.0.0.1')
        pull.connect(f'tcp://127.0.0.1:{port}')
        shadow_pull = zmq.Socket.shadow(pull)
        stream = zmqstream.ZMQStream(shadow_pull)
        # send some messages
        for i in range(10):
            push.send_string(str(i))
        # make sure at least one message has been delivered
        pull.recv()
        # register callback
        # this should schedule event callback on the next tick
        stream.on_recv(print)
        # close the shadowed socket
        pull.close()
    # run the event loop, which should see some events on the shadow socket
    # but the socket has been closed!
    with warnings.catch_warnings(record=True) as records:
        await asyncio.sleep(0.2)
    warning_text = "\n".join(str(r.message) for r in records)
    assert "after closing socket" in warning_text
    assert "closed socket" in caplog.text
