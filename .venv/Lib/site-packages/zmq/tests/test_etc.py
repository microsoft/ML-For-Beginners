# Copyright (c) PyZMQ Developers.
# Distributed under the terms of the Modified BSD License.


from pytest import mark

import zmq

only_bundled = mark.skipif(not hasattr(zmq, '_libzmq'), reason="bundled libzmq")


@mark.skipif('zmq.zmq_version_info() < (4, 1)')
def test_has():
    assert not zmq.has('something weird')


@only_bundled
def test_has_curve():
    """bundled libzmq has curve support"""
    assert zmq.has('curve')


@only_bundled
def test_has_ipc():
    """bundled libzmq has ipc support"""
    assert zmq.has('ipc')
