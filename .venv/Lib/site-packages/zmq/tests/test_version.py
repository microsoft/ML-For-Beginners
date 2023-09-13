# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.


from unittest import TestCase

import zmq
from zmq.sugar import version


class TestVersion(TestCase):
    def test_pyzmq_version(self):
        vs = zmq.pyzmq_version()
        vs2 = zmq.__version__
        assert isinstance(vs, str)
        if zmq.__revision__:
            assert vs == '@'.join(vs2, zmq.__revision__)
        else:
            assert vs == vs2
        if version.VERSION_EXTRA:
            assert version.VERSION_EXTRA in vs
            assert version.VERSION_EXTRA in vs2

    def test_pyzmq_version_info(self):
        info = zmq.pyzmq_version_info()
        assert isinstance(info, tuple)
        for n in info[:3]:
            assert isinstance(n, int)
        if version.VERSION_EXTRA:
            assert len(info) == 4
            assert info[-1] == float('inf')
        else:
            assert len(info) == 3

    def test_zmq_version_info(self):
        info = zmq.zmq_version_info()
        assert isinstance(info, tuple)
        for n in info[:3]:
            assert isinstance(n, int)

    def test_zmq_version(self):
        v = zmq.zmq_version()
        assert isinstance(v, str)
