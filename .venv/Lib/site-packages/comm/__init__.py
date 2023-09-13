"""Comm package.

Copyright (c) IPython Development Team.
Distributed under the terms of the Modified BSD License.

This package provides a way to register a Kernel Comm implementation, as per
the Jupyter kernel protocol.
It also provides a base Comm implementation and a default CommManager for the IPython case.
"""

from .base_comm import BaseComm, CommManager

__version__ = "0.1.4"
__all__ = [
    "create_comm",
    "get_comm_manager",
    "__version__",
]

_comm_manager = None


class DummyComm(BaseComm):
    def publish_msg(self, msg_type, data=None, metadata=None, buffers=None, **keys):
        pass


def _create_comm(*args, **kwargs):
    """Create a Comm.

    This method is intended to be replaced, so that it returns your Comm instance.
    """
    return DummyComm(*args, **kwargs)


def _get_comm_manager():
    """Get the current Comm manager, creates one if there is none.

    This method is intended to be replaced if needed (if you want to manage multiple CommManagers).
    """
    global _comm_manager

    if _comm_manager is None:
        _comm_manager = CommManager()

    return _comm_manager


create_comm = _create_comm
get_comm_manager = _get_comm_manager
