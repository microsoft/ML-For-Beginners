""" Defines a dummy socket implementing (part of) the zmq.Socket interface. """

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

from queue import Queue

import zmq
from traitlets import HasTraits, Instance, Int

# -----------------------------------------------------------------------------
# Dummy socket class
# -----------------------------------------------------------------------------


class DummySocket(HasTraits):
    """A dummy socket implementing (part of) the zmq.Socket interface."""

    queue = Instance(Queue, ())
    message_sent = Int(0)  # Should be an Event
    context = Instance(zmq.Context)

    def _context_default(self):
        return zmq.Context()

    # -------------------------------------------------------------------------
    # Socket interface
    # -------------------------------------------------------------------------

    def recv_multipart(self, flags=0, copy=True, track=False):
        """Recv a multipart message."""
        return self.queue.get_nowait()

    def send_multipart(self, msg_parts, flags=0, copy=True, track=False):
        """Send a multipart message."""
        msg_parts = list(map(zmq.Message, msg_parts))
        self.queue.put_nowait(msg_parts)
        self.message_sent += 1

    def flush(self, timeout=1.0):
        """no-op to comply with stream API"""
        pass
