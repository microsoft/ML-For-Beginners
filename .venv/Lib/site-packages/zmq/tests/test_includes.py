# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.


from unittest import TestCase

import zmq


class TestIncludes(TestCase):
    def test_get_includes(self):
        from os.path import basename

        includes = zmq.get_includes()
        assert isinstance(includes, list)
        assert len(includes) >= 2
        parent = includes[0]
        assert isinstance(parent, str)
        utilsdir = includes[1]
        assert isinstance(utilsdir, str)
        utils = basename(utilsdir)
        assert utils == "utils"

    def test_get_library_dirs(self):
        from os.path import basename

        libdirs = zmq.get_library_dirs()
        assert isinstance(libdirs, list)
        assert len(libdirs) == 1
        parent = libdirs[0]
        assert isinstance(parent, str)
        libdir = basename(parent)
        assert libdir == "zmq"
