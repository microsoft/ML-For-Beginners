""" Implements a fully blocking kernel client.

Useful for test suites and blocking terminal interfaces.
"""
import sys

# -----------------------------------------------------------------------------
#  Copyright (C) 2012  The IPython Development Team
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file LICENSE, distributed as part of this software.
# -----------------------------------------------------------------------------
from queue import Empty, Queue

# IPython imports
from traitlets import Type

# Local imports
from .channels import InProcessChannel
from .client import InProcessKernelClient


class BlockingInProcessChannel(InProcessChannel):
    """A blocking in-process channel."""

    def __init__(self, *args, **kwds):
        """Initialize the channel."""
        super().__init__(*args, **kwds)
        self._in_queue: Queue[object] = Queue()

    def call_handlers(self, msg):
        """Call the handlers for a message."""
        self._in_queue.put(msg)

    def get_msg(self, block=True, timeout=None):
        """Gets a message if there is one that is ready."""
        if timeout is None:
            # Queue.get(timeout=None) has stupid uninteruptible
            # behavior, so wait for a week instead
            timeout = 604800
        return self._in_queue.get(block, timeout)

    def get_msgs(self):
        """Get all messages that are currently ready."""
        msgs = []
        while True:
            try:
                msgs.append(self.get_msg(block=False))
            except Empty:
                break
        return msgs

    def msg_ready(self):
        """Is there a message that has been received?"""
        return not self._in_queue.empty()


class BlockingInProcessStdInChannel(BlockingInProcessChannel):
    """A blocking in-process stdin channel."""

    def call_handlers(self, msg):
        """Overridden for the in-process channel.

        This methods simply calls raw_input directly.
        """
        msg_type = msg["header"]["msg_type"]
        if msg_type == "input_request":
            _raw_input = self.client.kernel._sys_raw_input
            prompt = msg["content"]["prompt"]
            print(prompt, end="", file=sys.__stdout__)
            sys.__stdout__.flush()
            self.client.input(_raw_input())


class BlockingInProcessKernelClient(InProcessKernelClient):
    """A blocking in-process kernel client."""

    # The classes to use for the various channels.
    shell_channel_class = Type(BlockingInProcessChannel)  # type:ignore[arg-type]
    iopub_channel_class = Type(BlockingInProcessChannel)  # type:ignore[arg-type]
    stdin_channel_class = Type(BlockingInProcessStdInChannel)  # type:ignore[arg-type]

    def wait_for_ready(self):
        """Wait for kernel info reply on shell channel."""
        while True:
            self.kernel_info()
            try:
                msg = self.shell_channel.get_msg(block=True, timeout=1)
            except Empty:
                pass
            else:
                if msg["msg_type"] == "kernel_info_reply":
                    # Checking that IOPub is connected. If it is not connected, start over.
                    try:
                        self.iopub_channel.get_msg(block=True, timeout=0.2)
                    except Empty:
                        pass
                    else:
                        self._handle_kernel_info_reply(msg)
                        break

        # Flush IOPub channel
        while True:
            try:
                msg = self.iopub_channel.get_msg(block=True, timeout=0.2)
                print(msg["msg_type"])
            except Empty:
                break
