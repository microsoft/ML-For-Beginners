import gevent
from gevent import select

import zmq
from zmq import Poller as _original_Poller


class _Poller(_original_Poller):
    """Replacement for :class:`zmq.Poller`

    Ensures that the greened Poller below is used in calls to
    :meth:`zmq.Poller.poll`.
    """

    _gevent_bug_timeout = 1.33  # minimum poll interval, for working around gevent bug

    def _get_descriptors(self):
        """Returns three elements tuple with socket descriptors ready
        for gevent.select.select
        """
        rlist = []
        wlist = []
        xlist = []

        for socket, flags in self.sockets:
            if isinstance(socket, zmq.Socket):
                rlist.append(socket.getsockopt(zmq.FD))
                continue
            elif isinstance(socket, int):
                fd = socket
            elif hasattr(socket, 'fileno'):
                try:
                    fd = int(socket.fileno())
                except:
                    raise ValueError('fileno() must return an valid integer fd')
            else:
                raise TypeError(
                    'Socket must be a 0MQ socket, an integer fd '
                    'or have a fileno() method: %r' % socket
                )

            if flags & zmq.POLLIN:
                rlist.append(fd)
            if flags & zmq.POLLOUT:
                wlist.append(fd)
            if flags & zmq.POLLERR:
                xlist.append(fd)

        return (rlist, wlist, xlist)

    def poll(self, timeout=-1):
        """Overridden method to ensure that the green version of
        Poller is used.

        Behaves the same as :meth:`zmq.core.Poller.poll`
        """

        if timeout is None:
            timeout = -1

        if timeout < 0:
            timeout = -1

        rlist = None
        wlist = None
        xlist = None

        if timeout > 0:
            tout = gevent.Timeout.start_new(timeout / 1000.0)
        else:
            tout = None

        try:
            # Loop until timeout or events available
            rlist, wlist, xlist = self._get_descriptors()
            while True:
                events = super().poll(0)
                if events or timeout == 0:
                    return events

                # wait for activity on sockets in a green way
                # set a minimum poll frequency,
                # because gevent < 1.0 cannot be trusted to catch edge-triggered FD events
                _bug_timeout = gevent.Timeout.start_new(self._gevent_bug_timeout)
                try:
                    select.select(rlist, wlist, xlist)
                except gevent.Timeout as t:
                    if t is not _bug_timeout:
                        raise
                finally:
                    _bug_timeout.cancel()

        except gevent.Timeout as t:
            if t is not tout:
                raise
            return []
        finally:
            if timeout > 0:
                tout.cancel()
