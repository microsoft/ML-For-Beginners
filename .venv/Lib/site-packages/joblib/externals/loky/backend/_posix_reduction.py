###############################################################################
# Extra reducers for Unix based system and connections objects
#
# author: Thomas Moreau and Olivier Grisel
#
# adapted from multiprocessing/reduction.py (17/02/2017)
#  * Add adapted reduction for LokyProcesses and socket/Connection
#
import os
import socket
import _socket
from multiprocessing.connection import Connection
from multiprocessing.context import get_spawning_popen

from .reduction import register

HAVE_SEND_HANDLE = (
    hasattr(socket, "CMSG_LEN")
    and hasattr(socket, "SCM_RIGHTS")
    and hasattr(socket.socket, "sendmsg")
)


def _mk_inheritable(fd):
    os.set_inheritable(fd, True)
    return fd


def DupFd(fd):
    """Return a wrapper for an fd."""
    popen_obj = get_spawning_popen()
    if popen_obj is not None:
        return popen_obj.DupFd(popen_obj.duplicate_for_child(fd))
    elif HAVE_SEND_HANDLE:
        from multiprocessing import resource_sharer

        return resource_sharer.DupFd(fd)
    else:
        raise TypeError(
            "Cannot pickle connection object. This object can only be "
            "passed when spawning a new process"
        )


def _reduce_socket(s):
    df = DupFd(s.fileno())
    return _rebuild_socket, (df, s.family, s.type, s.proto)


def _rebuild_socket(df, family, type, proto):
    fd = df.detach()
    return socket.fromfd(fd, family, type, proto)


def rebuild_connection(df, readable, writable):
    fd = df.detach()
    return Connection(fd, readable, writable)


def reduce_connection(conn):
    df = DupFd(conn.fileno())
    return rebuild_connection, (df, conn.readable, conn.writable)


register(socket.socket, _reduce_socket)
register(_socket.socket, _reduce_socket)
register(Connection, reduce_connection)
