"""Abstract base class for kernel managers."""
# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
import abc
from typing import Any


class KernelManagerABC(metaclass=abc.ABCMeta):
    """KernelManager ABC.

    The docstrings for this class can be found in the base implementation:

    `jupyter_client.manager.KernelManager`
    """

    @abc.abstractproperty
    def kernel(self) -> Any:
        pass

    # --------------------------------------------------------------------------
    # Kernel management
    # --------------------------------------------------------------------------

    @abc.abstractmethod
    def start_kernel(self, **kw: Any) -> None:
        """Start the kernel."""
        pass

    @abc.abstractmethod
    def shutdown_kernel(self, now: bool = False, restart: bool = False) -> None:
        """Shut down the kernel."""
        pass

    @abc.abstractmethod
    def restart_kernel(self, now: bool = False, **kw: Any) -> None:
        """Restart the kernel."""
        pass

    @abc.abstractproperty
    def has_kernel(self) -> bool:
        pass

    @abc.abstractmethod
    def interrupt_kernel(self) -> None:
        """Interrupt the kernel."""
        pass

    @abc.abstractmethod
    def signal_kernel(self, signum: int) -> None:
        """Send a signal to the kernel."""
        pass

    @abc.abstractmethod
    def is_alive(self) -> bool:
        """Test whether the kernel is alive."""
        pass
