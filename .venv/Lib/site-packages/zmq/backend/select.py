"""Import basic exposure of libzmq C API as a backend"""

# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.

from importlib import import_module
from typing import Dict

public_api = [
    'Context',
    'Socket',
    'Frame',
    'Message',
    'device',
    'proxy',
    'proxy_steerable',
    'zmq_poll',
    'strerror',
    'zmq_errno',
    'has',
    'curve_keypair',
    'curve_public',
    'zmq_version_info',
    'IPC_PATH_MAX_LEN',
]


def select_backend(name: str) -> Dict:
    """Select the pyzmq backend"""
    try:
        mod = import_module(name)
    except ImportError:
        raise
    except Exception as e:
        raise ImportError(f"Importing {name} failed with {e}") from e
    return {key: getattr(mod, key) for key in public_api}
