"""Implements an async kernel client"""
# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

import zmq.asyncio
from traitlets import Instance, Type

from ..channels import AsyncZMQSocketChannel, HBChannel
from ..client import KernelClient, reqrep


def wrapped(meth, channel):
    """Wrap a method on a channel and handle replies."""

    def _(self, *args, **kwargs):
        reply = kwargs.pop("reply", False)
        timeout = kwargs.pop("timeout", None)
        msg_id = meth(self, *args, **kwargs)
        if not reply:
            return msg_id
        return self._recv_reply(msg_id, timeout=timeout, channel=channel)

    return _


class AsyncKernelClient(KernelClient):
    """A KernelClient with async APIs

    ``get_[channel]_msg()`` methods wait for and return messages on channels,
    raising :exc:`queue.Empty` if no message arrives within ``timeout`` seconds.
    """

    context = Instance(zmq.asyncio.Context)

    def _context_default(self) -> zmq.asyncio.Context:
        self._created_context = True
        return zmq.asyncio.Context()

    # --------------------------------------------------------------------------
    # Channel proxy methods
    # --------------------------------------------------------------------------

    get_shell_msg = KernelClient._async_get_shell_msg
    get_iopub_msg = KernelClient._async_get_iopub_msg
    get_stdin_msg = KernelClient._async_get_stdin_msg
    get_control_msg = KernelClient._async_get_control_msg

    wait_for_ready = KernelClient._async_wait_for_ready

    # The classes to use for the various channels
    shell_channel_class = Type(AsyncZMQSocketChannel)
    iopub_channel_class = Type(AsyncZMQSocketChannel)
    stdin_channel_class = Type(AsyncZMQSocketChannel)
    hb_channel_class = Type(HBChannel)
    control_channel_class = Type(AsyncZMQSocketChannel)

    _recv_reply = KernelClient._async_recv_reply

    # replies come on the shell channel
    execute = reqrep(wrapped, KernelClient.execute)
    history = reqrep(wrapped, KernelClient.history)
    complete = reqrep(wrapped, KernelClient.complete)
    is_complete = reqrep(wrapped, KernelClient.is_complete)
    inspect = reqrep(wrapped, KernelClient.inspect)
    kernel_info = reqrep(wrapped, KernelClient.kernel_info)
    comm_info = reqrep(wrapped, KernelClient.comm_info)

    is_alive = KernelClient._async_is_alive
    execute_interactive = KernelClient._async_execute_interactive

    # replies come on the control channel
    shutdown = reqrep(wrapped, KernelClient.shutdown, channel="control")
