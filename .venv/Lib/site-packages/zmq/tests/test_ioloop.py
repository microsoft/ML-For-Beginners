# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.


import pytest

try:
    import tornado.ioloop
except ImportError:
    _tornado = False
else:
    _tornado = True


def setup():
    if not _tornado:
        pytest.skip("requires tornado")


def test_ioloop():
    # may have been imported before,
    # can't capture the warning
    from zmq.eventloop import ioloop

    assert ioloop.IOLoop is tornado.ioloop.IOLoop
    assert ioloop.ZMQIOLoop is ioloop.IOLoop


def test_ioloop_install():
    from zmq.eventloop import ioloop

    with pytest.warns(DeprecationWarning):
        ioloop.install()
