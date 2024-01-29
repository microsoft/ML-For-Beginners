"""A kernel client for in-process kernels."""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

from typing import List

from jupyter_client.channelsabc import HBChannelABC

# -----------------------------------------------------------------------------
# Channel classes
# -----------------------------------------------------------------------------


class InProcessChannel:
    """Base class for in-process channels."""

    proxy_methods: List[object] = []

    def __init__(self, client=None):
        """Initialize the channel."""
        super().__init__()
        self.client = client
        self._is_alive = False

    def is_alive(self):
        """Test if the channel is alive."""
        return self._is_alive

    def start(self):
        """Start the channel."""
        self._is_alive = True

    def stop(self):
        """Stop the channel."""
        self._is_alive = False

    def call_handlers(self, msg):
        """This method is called in the main thread when a message arrives.

        Subclasses should override this method to handle incoming messages.
        """
        msg = "call_handlers must be defined in a subclass."
        raise NotImplementedError(msg)

    def flush(self, timeout=1.0):
        """Flush the channel."""

    def call_handlers_later(self, *args, **kwds):
        """Call the message handlers later.

        The default implementation just calls the handlers immediately, but this
        method exists so that GUI toolkits can defer calling the handlers until
        after the event loop has run, as expected by GUI frontends.
        """
        self.call_handlers(*args, **kwds)

    def process_events(self):
        """Process any pending GUI events.

        This method will be never be called from a frontend without an event
        loop (e.g., a terminal frontend).
        """
        raise NotImplementedError


class InProcessHBChannel:
    """A dummy heartbeat channel interface for in-process kernels.

    Normally we use the heartbeat to check that the kernel process is alive.
    When the kernel is in-process, that doesn't make sense, but clients still
    expect this interface.
    """

    time_to_dead = 3.0

    def __init__(self, client=None):
        """Initialize the channel."""
        super().__init__()
        self.client = client
        self._is_alive = False
        self._pause = True

    def is_alive(self):
        """Test if the channel is alive."""
        return self._is_alive

    def start(self):
        """Start the channel."""
        self._is_alive = True

    def stop(self):
        """Stop the channel."""
        self._is_alive = False

    def pause(self):
        """Pause the channel."""
        self._pause = True

    def unpause(self):
        """Unpause the channel."""
        self._pause = False

    def is_beating(self):
        """Test if the channel is beating."""
        return not self._pause


HBChannelABC.register(InProcessHBChannel)
