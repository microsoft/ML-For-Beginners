"""Tests for heartbeat thread"""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

import errno
from typing import no_type_check
from unittest.mock import patch

import pytest
import zmq

from ipykernel.heartbeat import Heartbeat


def test_port_bind_failure_raises():
    heart = Heartbeat(None)
    with patch.object(heart, "_try_bind_socket") as mock_try_bind:
        mock_try_bind.side_effect = zmq.ZMQError(-100, "fails for unknown error types")
        with pytest.raises(zmq.ZMQError):
            heart._bind_socket()
        assert mock_try_bind.call_count == 1


def test_port_bind_success():
    heart = Heartbeat(None)
    with patch.object(heart, "_try_bind_socket") as mock_try_bind:
        heart._bind_socket()
        assert mock_try_bind.call_count == 1


@no_type_check
def test_port_bind_failure_recovery():
    try:
        errno.WSAEADDRINUSE
    except AttributeError:
        # Fake windows address in-use code
        errno.WSAEADDRINUSE = 12345

    try:
        heart = Heartbeat(None)
        with patch.object(heart, "_try_bind_socket") as mock_try_bind:
            mock_try_bind.side_effect = [
                zmq.ZMQError(errno.EADDRINUSE, "fails for non-bind unix"),
                zmq.ZMQError(errno.WSAEADDRINUSE, "fails for non-bind windows"),
            ] + [0] * 100
            # Shouldn't raise anything as retries will kick in
            heart._bind_socket()
    finally:
        # Cleanup fake assignment
        if errno.WSAEADDRINUSE == 12345:
            del errno.WSAEADDRINUSE


def test_port_bind_failure_gives_up_retries():
    heart = Heartbeat(None)
    with patch.object(heart, "_try_bind_socket") as mock_try_bind:
        mock_try_bind.side_effect = zmq.ZMQError(errno.EADDRINUSE, "fails for non-bind")
        with pytest.raises(zmq.ZMQError):
            heart._bind_socket()
        assert mock_try_bind.call_count == 100
