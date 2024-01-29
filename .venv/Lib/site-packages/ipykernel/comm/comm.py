"""Base class for a Comm"""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

import uuid
from typing import Optional
from warnings import warn

import comm.base_comm
import traitlets.config
from traitlets import Bool, Bytes, Instance, Unicode, default

from ipykernel.jsonutil import json_clean
from ipykernel.kernelbase import Kernel


# this is the class that will be created if we do comm.create_comm
class BaseComm(comm.base_comm.BaseComm):  # type:ignore[misc]
    """The base class for comms."""

    kernel: Optional["Kernel"] = None

    def publish_msg(self, msg_type, data=None, metadata=None, buffers=None, **keys):
        """Helper for sending a comm message on IOPub"""
        if not Kernel.initialized():
            return

        data = {} if data is None else data
        metadata = {} if metadata is None else metadata
        content = json_clean(dict(data=data, comm_id=self.comm_id, **keys))

        if self.kernel is None:
            self.kernel = Kernel.instance()

        assert self.kernel.session is not None
        self.kernel.session.send(
            self.kernel.iopub_socket,
            msg_type,
            content,
            metadata=json_clean(metadata),
            parent=self.kernel.get_parent(),
            ident=self.topic,
            buffers=buffers,
        )


# but for backwards compatibility, we need to inherit from LoggingConfigurable
class Comm(BaseComm, traitlets.config.LoggingConfigurable):
    """Class for communicating between a Frontend and a Kernel"""

    kernel = Instance("ipykernel.kernelbase.Kernel", allow_none=True)  # type:ignore[assignment]
    comm_id = Unicode()
    primary = Bool(True, help="Am I the primary or secondary Comm?")

    target_name = Unicode("comm")
    target_module = Unicode(
        None,
        allow_none=True,
        help="""requirejs module from
        which to load comm target.""",
    )

    topic = Bytes()

    @default("kernel")
    def _default_kernel(self):
        if Kernel.initialized():
            return Kernel.instance()
        return None

    @default("comm_id")
    def _default_comm_id(self):
        return uuid.uuid4().hex

    def __init__(
        self, target_name="", data=None, metadata=None, buffers=None, show_warning=True, **kwargs
    ):
        """Initialize a comm."""
        if show_warning:
            warn(
                "The `ipykernel.comm.Comm` class has been deprecated. Please use the `comm` module instead."
                "For creating comms, use the function `from comm import create_comm`.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Handle differing arguments between base classes.
        had_kernel = "kernel" in kwargs
        kernel = kwargs.pop("kernel", None)
        if target_name:
            kwargs["target_name"] = target_name
        BaseComm.__init__(self, data=data, metadata=metadata, buffers=buffers, **kwargs)  # type:ignore[call-arg]
        # only re-add kernel if explicitly provided
        if had_kernel:
            kwargs["kernel"] = kernel
        traitlets.config.LoggingConfigurable.__init__(self, **kwargs)


__all__ = ["Comm"]
