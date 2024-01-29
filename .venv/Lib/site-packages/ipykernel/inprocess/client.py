"""A client for in-process kernels."""

# -----------------------------------------------------------------------------
#  Copyright (C) 2012  The IPython Development Team
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file LICENSE, distributed as part of this software.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import asyncio

from jupyter_client.client import KernelClient
from jupyter_client.clientabc import KernelClientABC
from jupyter_core.utils import run_sync

# IPython imports
from traitlets import Instance, Type, default

# Local imports
from .channels import InProcessChannel, InProcessHBChannel

# -----------------------------------------------------------------------------
# Main kernel Client class
# -----------------------------------------------------------------------------


class InProcessKernelClient(KernelClient):
    """A client for an in-process kernel.

    This class implements the interface of
    `jupyter_client.clientabc.KernelClientABC` and allows
    (asynchronous) frontends to be used seamlessly with an in-process kernel.

    See `jupyter_client.client.KernelClient` for docstrings.
    """

    # The classes to use for the various channels.
    shell_channel_class = Type(InProcessChannel)  # type:ignore[arg-type]
    iopub_channel_class = Type(InProcessChannel)  # type:ignore[arg-type]
    stdin_channel_class = Type(InProcessChannel)  # type:ignore[arg-type]
    control_channel_class = Type(InProcessChannel)  # type:ignore[arg-type]
    hb_channel_class = Type(InProcessHBChannel)  # type:ignore[arg-type]

    kernel = Instance("ipykernel.inprocess.ipkernel.InProcessKernel", allow_none=True)

    # --------------------------------------------------------------------------
    # Channel management methods
    # --------------------------------------------------------------------------

    @default("blocking_class")
    def _default_blocking_class(self):
        from .blocking import BlockingInProcessKernelClient

        return BlockingInProcessKernelClient

    def get_connection_info(self):
        """Get the connection info for the client."""
        d = super().get_connection_info()
        d["kernel"] = self.kernel  # type:ignore[assignment]
        return d

    def start_channels(self, *args, **kwargs):
        """Start the channels on the client."""
        super().start_channels()
        if self.kernel:
            self.kernel.frontends.append(self)

    @property
    def shell_channel(self):
        if self._shell_channel is None:
            self._shell_channel = self.shell_channel_class(self)  # type:ignore[abstract,call-arg]
        return self._shell_channel

    @property
    def iopub_channel(self):
        if self._iopub_channel is None:
            self._iopub_channel = self.iopub_channel_class(self)  # type:ignore[abstract,call-arg]
        return self._iopub_channel

    @property
    def stdin_channel(self):
        if self._stdin_channel is None:
            self._stdin_channel = self.stdin_channel_class(self)  # type:ignore[abstract,call-arg]
        return self._stdin_channel

    @property
    def control_channel(self):
        if self._control_channel is None:
            self._control_channel = self.control_channel_class(self)  # type:ignore[abstract,call-arg]
        return self._control_channel

    @property
    def hb_channel(self):
        if self._hb_channel is None:
            self._hb_channel = self.hb_channel_class(self)  # type:ignore[abstract,call-arg]
        return self._hb_channel

    # Methods for sending specific messages
    # -------------------------------------

    def execute(
        self, code, silent=False, store_history=True, user_expressions=None, allow_stdin=None
    ):
        """Execute code on the client."""
        if allow_stdin is None:
            allow_stdin = self.allow_stdin
        content = dict(
            code=code,
            silent=silent,
            store_history=store_history,
            user_expressions=user_expressions or {},
            allow_stdin=allow_stdin,
        )
        msg = self.session.msg("execute_request", content)
        self._dispatch_to_kernel(msg)
        return msg["header"]["msg_id"]

    def complete(self, code, cursor_pos=None):
        """Get code completion."""
        if cursor_pos is None:
            cursor_pos = len(code)
        content = dict(code=code, cursor_pos=cursor_pos)
        msg = self.session.msg("complete_request", content)
        self._dispatch_to_kernel(msg)
        return msg["header"]["msg_id"]

    def inspect(self, code, cursor_pos=None, detail_level=0):
        """Get code inspection."""
        if cursor_pos is None:
            cursor_pos = len(code)
        content = dict(
            code=code,
            cursor_pos=cursor_pos,
            detail_level=detail_level,
        )
        msg = self.session.msg("inspect_request", content)
        self._dispatch_to_kernel(msg)
        return msg["header"]["msg_id"]

    def history(self, raw=True, output=False, hist_access_type="range", **kwds):
        """Get code history."""
        content = dict(raw=raw, output=output, hist_access_type=hist_access_type, **kwds)
        msg = self.session.msg("history_request", content)
        self._dispatch_to_kernel(msg)
        return msg["header"]["msg_id"]

    def shutdown(self, restart=False):
        """Handle shutdown."""
        # FIXME: What to do here?
        msg = "Cannot shutdown in-process kernel"
        raise NotImplementedError(msg)

    def kernel_info(self):
        """Request kernel info."""
        msg = self.session.msg("kernel_info_request")
        self._dispatch_to_kernel(msg)
        return msg["header"]["msg_id"]

    def comm_info(self, target_name=None):
        """Request a dictionary of valid comms and their targets."""
        content = {} if target_name is None else dict(target_name=target_name)
        msg = self.session.msg("comm_info_request", content)
        self._dispatch_to_kernel(msg)
        return msg["header"]["msg_id"]

    def input(self, string):
        """Handle kernel input."""
        if self.kernel is None:
            msg = "Cannot send input reply. No kernel exists."
            raise RuntimeError(msg)
        self.kernel.raw_input_str = string

    def is_complete(self, code):
        """Handle an is_complete request."""
        msg = self.session.msg("is_complete_request", {"code": code})
        self._dispatch_to_kernel(msg)
        return msg["header"]["msg_id"]

    def _dispatch_to_kernel(self, msg):
        """Send a message to the kernel and handle a reply."""
        kernel = self.kernel
        if kernel is None:
            msg = "Cannot send request. No kernel exists."
            raise RuntimeError(msg)

        stream = kernel.shell_stream
        self.session.send(stream, msg)
        msg_parts = stream.recv_multipart()
        if run_sync is not None:
            dispatch_shell = run_sync(kernel.dispatch_shell)
            dispatch_shell(msg_parts)
        else:
            loop = asyncio.get_event_loop()  # type:ignore[unreachable]
            loop.run_until_complete(kernel.dispatch_shell(msg_parts))
        idents, reply_msg = self.session.recv(stream, copy=False)
        self.shell_channel.call_handlers_later(reply_msg)

    def get_shell_msg(self, block=True, timeout=None):
        """Get a shell message."""
        return self.shell_channel.get_msg(block, timeout)

    def get_iopub_msg(self, block=True, timeout=None):
        """Get an iopub message."""
        return self.iopub_channel.get_msg(block, timeout)

    def get_stdin_msg(self, block=True, timeout=None):
        """Get a stdin message."""
        return self.stdin_channel.get_msg(block, timeout)

    def get_control_msg(self, block=True, timeout=None):
        """Get a control message."""
        return self.control_channel.get_msg(block, timeout)


# -----------------------------------------------------------------------------
# ABC Registration
# -----------------------------------------------------------------------------

KernelClientABC.register(InProcessKernelClient)
