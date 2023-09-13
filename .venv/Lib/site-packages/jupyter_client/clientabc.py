"""Abstract base class for kernel clients"""
# -----------------------------------------------------------------------------
#  Copyright (c) The Jupyter Development Team
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import abc

# -----------------------------------------------------------------------------
# Main kernel client class
# -----------------------------------------------------------------------------


class KernelClientABC(metaclass=abc.ABCMeta):
    """KernelManager ABC.

    The docstrings for this class can be found in the base implementation:

    `jupyter_client.client.KernelClient`
    """

    @abc.abstractproperty
    def kernel(self):
        pass

    @abc.abstractproperty
    def shell_channel_class(self):
        pass

    @abc.abstractproperty
    def iopub_channel_class(self):
        pass

    @abc.abstractproperty
    def hb_channel_class(self):
        pass

    @abc.abstractproperty
    def stdin_channel_class(self):
        pass

    @abc.abstractproperty
    def control_channel_class(self):
        pass

    # --------------------------------------------------------------------------
    # Channel management methods
    # --------------------------------------------------------------------------

    @abc.abstractmethod
    def start_channels(self, shell=True, iopub=True, stdin=True, hb=True, control=True):
        """Start the channels for the client."""
        pass

    @abc.abstractmethod
    def stop_channels(self):
        """Stop the channels for the client."""
        pass

    @abc.abstractproperty
    def channels_running(self):
        """Get whether the channels are running."""
        pass

    @abc.abstractproperty
    def shell_channel(self):
        pass

    @abc.abstractproperty
    def iopub_channel(self):
        pass

    @abc.abstractproperty
    def stdin_channel(self):
        pass

    @abc.abstractproperty
    def hb_channel(self):
        pass

    @abc.abstractproperty
    def control_channel(self):
        pass
