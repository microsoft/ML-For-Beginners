"""Base class to manage comms"""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

import logging

import comm.base_comm
import traitlets
import traitlets.config

from .comm import Comm

logger = logging.getLogger("ipykernel.comm")


class CommManager(comm.base_comm.CommManager, traitlets.config.LoggingConfigurable):  # type:ignore[misc]
    """A comm manager."""

    kernel = traitlets.Instance("ipykernel.kernelbase.Kernel")
    comms = traitlets.Dict()
    targets = traitlets.Dict()

    def __init__(self, **kwargs):
        """Initialize the manager."""
        # CommManager doesn't take arguments, so we explicitly forward arguments
        comm.base_comm.CommManager.__init__(self)
        traitlets.config.LoggingConfigurable.__init__(self, **kwargs)

    def comm_open(self, stream, ident, msg):
        """Handler for comm_open messages"""
        # This is for backward compatibility, the comm_open creates a a new ipykernel.comm.Comm
        # but we should let the base class create the comm with comm.create_comm in a major release
        content = msg["content"]
        comm_id = content["comm_id"]
        target_name = content["target_name"]
        f = self.targets.get(target_name, None)
        comm = Comm(
            comm_id=comm_id,
            primary=False,
            target_name=target_name,
            show_warning=False,
        )
        self.register_comm(comm)
        if f is None:
            logger.error("No such comm target registered: %s", target_name)
        else:
            try:
                f(comm, msg)
                return
            except Exception:
                logger.error("Exception opening comm with target: %s", target_name, exc_info=True)  # noqa: G201

        # Failure.
        try:
            comm.close()
        except Exception:
            logger.error(  # noqa: G201
                """Could not close comm during `comm_open` failure
                clean-up.  The comm may not have been opened yet.""",
                exc_info=True,
            )
