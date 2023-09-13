"""Tests for kernel connection utilities"""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

import errno
import json
import os
from tempfile import TemporaryDirectory
from typing import no_type_check
from unittest.mock import patch

import pytest
import zmq
from traitlets.config.loader import Config

from ipykernel import connect
from ipykernel.kernelapp import IPKernelApp

from .utils import TemporaryWorkingDirectory

sample_info: dict = {
    "ip": "1.2.3.4",
    "transport": "ipc",
    "shell_port": 1,
    "hb_port": 2,
    "iopub_port": 3,
    "stdin_port": 4,
    "control_port": 5,
    "key": b"abc123",
    "signature_scheme": "hmac-md5",
}


class DummyKernelApp(IPKernelApp):
    def _default_shell_port(self):
        return 0

    def initialize(self, argv=None):
        self.init_profile_dir()
        self.init_connection_file()


def test_get_connection_file():
    cfg = Config()
    with TemporaryWorkingDirectory() as d:
        cfg.ProfileDir.location = d
        cf = "kernel.json"
        app = DummyKernelApp(config=cfg, connection_file=cf)
        app.initialize()

        profile_cf = os.path.join(app.connection_dir, cf)
        assert profile_cf == app.abs_connection_file
        with open(profile_cf, "w") as f:
            f.write("{}")
        assert os.path.exists(profile_cf)
        assert connect.get_connection_file(app) == profile_cf

        app.connection_file = cf
        assert connect.get_connection_file(app) == profile_cf


def test_get_connection_info():
    with TemporaryDirectory() as d:
        cf = os.path.join(d, "kernel.json")
        connect.write_connection_file(cf, **sample_info)
        json_info = connect.get_connection_info(cf)
        info = connect.get_connection_info(cf, unpack=True)
    assert isinstance(json_info, str)

    sub_info = {k: v for k, v in info.items() if k in sample_info}
    assert sub_info == sample_info

    info2 = json.loads(json_info)
    info2["key"] = info2["key"].encode("utf-8")
    sub_info2 = {k: v for k, v in info.items() if k in sample_info}
    assert sub_info2 == sample_info


def test_port_bind_failure_raises(request):
    cfg = Config()
    with TemporaryWorkingDirectory() as d:
        cfg.ProfileDir.location = d
        cf = "kernel.json"
        app = DummyKernelApp(config=cfg, connection_file=cf)
        request.addfinalizer(app.close)
        app.initialize()
        with patch.object(app, "_try_bind_socket") as mock_try_bind:
            mock_try_bind.side_effect = zmq.ZMQError(-100, "fails for unknown error types")
            with pytest.raises(zmq.ZMQError):
                app.init_sockets()
            assert mock_try_bind.call_count == 1


@no_type_check
def test_port_bind_failure_recovery(request):
    try:
        errno.WSAEADDRINUSE
    except AttributeError:
        # Fake windows address in-use code
        p = patch.object(errno, "WSAEADDRINUSE", 12345, create=True)
        p.start()
        request.addfinalizer(p.stop)

    cfg = Config()
    with TemporaryWorkingDirectory() as d:
        cfg.ProfileDir.location = d
        cf = "kernel.json"
        app = DummyKernelApp(config=cfg, connection_file=cf)
        request.addfinalizer(app.close)
        app.initialize()
        with patch.object(app, "_try_bind_socket") as mock_try_bind:
            mock_try_bind.side_effect = [
                zmq.ZMQError(errno.EADDRINUSE, "fails for non-bind unix"),
                zmq.ZMQError(errno.WSAEADDRINUSE, "fails for non-bind windows"),
            ] + [0] * 100
            # Shouldn't raise anything as retries will kick in
            app.init_sockets()


def test_port_bind_failure_gives_up_retries(request):
    cfg = Config()
    with TemporaryWorkingDirectory() as d:
        cfg.ProfileDir.location = d
        cf = "kernel.json"
        app = DummyKernelApp(config=cfg, connection_file=cf)
        request.addfinalizer(app.close)
        app.initialize()
        with patch.object(app, "_try_bind_socket") as mock_try_bind:
            mock_try_bind.side_effect = zmq.ZMQError(errno.EADDRINUSE, "fails for non-bind")
            with pytest.raises(zmq.ZMQError):
                app.init_sockets()
            assert mock_try_bind.call_count == 100
