"""0MQ polling related functions and classes."""

# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.

from typing import Any, Dict, List, Optional, Tuple

from zmq.backend import zmq_poll
from zmq.constants import POLLERR, POLLIN, POLLOUT

# -----------------------------------------------------------------------------
# Polling related methods
# -----------------------------------------------------------------------------


class Poller:
    """A stateful poll interface that mirrors Python's built-in poll."""

    sockets: List[Tuple[Any, int]]
    _map: Dict

    def __init__(self) -> None:
        self.sockets = []
        self._map = {}

    def __contains__(self, socket: Any) -> bool:
        return socket in self._map

    def register(self, socket: Any, flags: int = POLLIN | POLLOUT):
        """p.register(socket, flags=POLLIN|POLLOUT)

        Register a 0MQ socket or native fd for I/O monitoring.

        register(s,0) is equivalent to unregister(s).

        Parameters
        ----------
        socket : zmq.Socket or native socket
            A zmq.Socket or any Python object having a ``fileno()``
            method that returns a valid file descriptor.
        flags : int
            The events to watch for.  Can be POLLIN, POLLOUT or POLLIN|POLLOUT.
            If `flags=0`, socket will be unregistered.
        """
        if flags:
            if socket in self._map:
                idx = self._map[socket]
                self.sockets[idx] = (socket, flags)
            else:
                idx = len(self.sockets)
                self.sockets.append((socket, flags))
                self._map[socket] = idx
        elif socket in self._map:
            # uregister sockets registered with no events
            self.unregister(socket)
        else:
            # ignore new sockets with no events
            pass

    def modify(self, socket, flags=POLLIN | POLLOUT):
        """Modify the flags for an already registered 0MQ socket or native fd."""
        self.register(socket, flags)

    def unregister(self, socket: Any):
        """Remove a 0MQ socket or native fd for I/O monitoring.

        Parameters
        ----------
        socket : Socket
            The socket instance to stop polling.
        """
        idx = self._map.pop(socket)
        self.sockets.pop(idx)
        # shift indices after deletion
        for socket, flags in self.sockets[idx:]:
            self._map[socket] -= 1

    def poll(self, timeout: Optional[int] = None) -> List[Tuple[Any, int]]:
        """Poll the registered 0MQ or native fds for I/O.

        If there are currently events ready to be processed, this function will return immediately.
        Otherwise, this function will return as soon the first event is available or after timeout
        milliseconds have elapsed.

        Parameters
        ----------
        timeout : int
            The timeout in milliseconds. If None, no `timeout` (infinite). This
            is in milliseconds to be compatible with ``select.poll()``.

        Returns
        -------
        events : list of tuples
            The list of events that are ready to be processed.
            This is a list of tuples of the form ``(socket, event_mask)``, where the 0MQ Socket
            or integer fd is the first element, and the poll event mask (POLLIN, POLLOUT) is the second.
            It is common to call ``events = dict(poller.poll())``,
            which turns the list of tuples into a mapping of ``socket : event_mask``.
        """
        if timeout is None or timeout < 0:
            timeout = -1
        elif isinstance(timeout, float):
            timeout = int(timeout)
        return zmq_poll(self.sockets, timeout=timeout)


def select(rlist: List, wlist: List, xlist: List, timeout: Optional[float] = None):
    """select(rlist, wlist, xlist, timeout=None) -> (rlist, wlist, xlist)

    Return the result of poll as a lists of sockets ready for r/w/exception.

    This has the same interface as Python's built-in ``select.select()`` function.

    Parameters
    ----------
    timeout : float, int, optional
        The timeout in seconds. If None, no timeout (infinite). This is in seconds to be
        compatible with ``select.select()``.
    rlist : list of sockets/FDs
        sockets/FDs to be polled for read events
    wlist : list of sockets/FDs
        sockets/FDs to be polled for write events
    xlist : list of sockets/FDs
        sockets/FDs to be polled for error events

    Returns
    -------
    (rlist, wlist, xlist) : tuple of lists of sockets (length 3)
        Lists correspond to sockets available for read/write/error events respectively.
    """
    if timeout is None:
        timeout = -1
    # Convert from sec -> ms for zmq_poll.
    # zmq_poll accepts 3.x style timeout in ms
    timeout = int(timeout * 1000.0)
    if timeout < 0:
        timeout = -1
    sockets = []
    for s in set(rlist + wlist + xlist):
        flags = 0
        if s in rlist:
            flags |= POLLIN
        if s in wlist:
            flags |= POLLOUT
        if s in xlist:
            flags |= POLLERR
        sockets.append((s, flags))
    return_sockets = zmq_poll(sockets, timeout)
    rlist, wlist, xlist = [], [], []
    for s, flags in return_sockets:
        if flags & POLLIN:
            rlist.append(s)
        if flags & POLLOUT:
            wlist.append(s)
        if flags & POLLERR:
            xlist.append(s)
    return rlist, wlist, xlist


# -----------------------------------------------------------------------------
# Symbols to export
# -----------------------------------------------------------------------------

__all__ = ['Poller', 'select']
