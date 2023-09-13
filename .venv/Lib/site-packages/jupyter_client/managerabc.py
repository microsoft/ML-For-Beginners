"""Abstract base class for kernel managers."""
# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
import abc


class KernelManagerABC(metaclass=abc.ABCMeta):
    """KernelManager ABC.

    The docstrings for this class can be found in the base implementation:

    `jupyter_client.kernelmanager.KernelManager`
    """

    @abc.abstractproperty
    def kernel(self):
        pass

    # --------------------------------------------------------------------------
    # Kernel management
    # --------------------------------------------------------------------------

    @abc.abstractmethod
    def start_kernel(self, **kw):
        """Start the kernel."""
        pass

    @abc.abstractmethod
    def shutdown_kernel(self, now=False, restart=False):
        """Shut down the kernel."""
        pass

    @abc.abstractmethod
    def restart_kernel(self, now=False, **kw):
        """Restart the kernel."""
        pass

    @abc.abstractproperty
    def has_kernel(self):
        pass

    @abc.abstractmethod
    def interrupt_kernel(self):
        """Interrupt the kernel."""
        pass

    @abc.abstractmethod
    def signal_kernel(self, signum):
        """Send a signal to the kernel."""
        pass

    @abc.abstractmethod
    def is_alive(self):
        """Test whether the kernel is alive."""
        pass
