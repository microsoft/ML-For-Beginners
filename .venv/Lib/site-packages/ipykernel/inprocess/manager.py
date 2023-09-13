"""A kernel manager for in-process kernels."""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

from jupyter_client.manager import KernelManager
from jupyter_client.managerabc import KernelManagerABC
from jupyter_client.session import Session
from traitlets import DottedObjectName, Instance, default

from .constants import INPROCESS_KEY


class InProcessKernelManager(KernelManager):
    """A manager for an in-process kernel.

    This class implements the interface of
    `jupyter_client.kernelmanagerabc.KernelManagerABC` and allows
    (asynchronous) frontends to be used seamlessly with an in-process kernel.

    See `jupyter_client.kernelmanager.KernelManager` for docstrings.
    """

    # The kernel process with which the KernelManager is communicating.
    kernel = Instance("ipykernel.inprocess.ipkernel.InProcessKernel", allow_none=True)
    # the client class for KM.client() shortcut
    client_class = DottedObjectName("ipykernel.inprocess.BlockingInProcessKernelClient")

    @default("blocking_class")
    def _default_blocking_class(self):
        from .blocking import BlockingInProcessKernelClient

        return BlockingInProcessKernelClient

    @default("session")
    def _default_session(self):
        # don't sign in-process messages
        return Session(key=INPROCESS_KEY, parent=self)

    # --------------------------------------------------------------------------
    # Kernel management methods
    # --------------------------------------------------------------------------

    def start_kernel(self, **kwds):
        """Start the kernel."""
        from ipykernel.inprocess.ipkernel import InProcessKernel

        self.kernel = InProcessKernel(parent=self, session=self.session)

    def shutdown_kernel(self):
        """Shutdown the kernel."""
        self.kernel.iopub_thread.stop()
        self._kill_kernel()

    def restart_kernel(self, now=False, **kwds):
        """Restart the kernel."""
        self.shutdown_kernel()
        self.start_kernel(**kwds)

    @property
    def has_kernel(self):
        return self.kernel is not None

    def _kill_kernel(self):
        self.kernel = None

    def interrupt_kernel(self):
        """Interrupt the kernel."""
        msg = "Cannot interrupt in-process kernel."
        raise NotImplementedError(msg)

    def signal_kernel(self, signum):
        """Send a signal to the kernel."""
        msg = "Cannot signal in-process kernel."
        raise NotImplementedError(msg)

    def is_alive(self):
        """Test if the kernel is alive."""
        return self.kernel is not None

    def client(self, **kwargs):
        """Get a client for the kernel."""
        kwargs["kernel"] = self.kernel
        return super().client(**kwargs)


# -----------------------------------------------------------------------------
# ABC Registration
# -----------------------------------------------------------------------------

KernelManagerABC.register(InProcessKernelManager)
