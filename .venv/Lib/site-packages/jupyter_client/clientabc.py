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
from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .channelsabc import ChannelABC

# -----------------------------------------------------------------------------
# Main kernel client class
# -----------------------------------------------------------------------------


class KernelClientABC(metaclass=abc.ABCMeta):
    """KernelManager ABC.

    The docstrings for this class can be found in the base implementation:

    `jupyter_client.client.KernelClient`
    """

    @abc.abstractproperty
    def kernel(self) -> Any:
        pass

    @abc.abstractproperty
    def shell_channel_class(self) -> type[ChannelABC]:
        pass

    @abc.abstractproperty
    def iopub_channel_class(self) -> type[ChannelABC]:
        pass

    @abc.abstractproperty
    def hb_channel_class(self) -> type[ChannelABC]:
        pass

    @abc.abstractproperty
    def stdin_channel_class(self) -> type[ChannelABC]:
        pass

    @abc.abstractproperty
    def control_channel_class(self) -> type[ChannelABC]:
        pass

    # --------------------------------------------------------------------------
    # Channel management methods
    # --------------------------------------------------------------------------

    @abc.abstractmethod
    def start_channels(
        self,
        shell: bool = True,
        iopub: bool = True,
        stdin: bool = True,
        hb: bool = True,
        control: bool = True,
    ) -> None:
        """Start the channels for the client."""
        pass

    @abc.abstractmethod
    def stop_channels(self) -> None:
        """Stop the channels for the client."""
        pass

    @abc.abstractproperty
    def channels_running(self) -> bool:
        """Get whether the channels are running."""
        pass

    @abc.abstractproperty
    def shell_channel(self) -> ChannelABC:
        pass

    @abc.abstractproperty
    def iopub_channel(self) -> ChannelABC:
        pass

    @abc.abstractproperty
    def stdin_channel(self) -> ChannelABC:
        pass

    @abc.abstractproperty
    def hb_channel(self) -> ChannelABC:
        pass

    @abc.abstractproperty
    def control_channel(self) -> ChannelABC:
        pass
