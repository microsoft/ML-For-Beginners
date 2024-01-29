"""Default classes for Comm and CommManager, for usage in IPython.
"""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.
from __future__ import annotations

import contextlib
import logging
import typing as t
import uuid

from traitlets.utils.importstring import import_item

import comm

if t.TYPE_CHECKING:
    from zmq.eventloop.zmqstream import ZMQStream

logger = logging.getLogger("Comm")

MessageType = t.Dict[str, t.Any]
MaybeDict = t.Optional[t.Dict[str, t.Any]]
BuffersType = t.Optional[t.List[bytes]]
CommCallback = t.Callable[[MessageType], None]
CommTargetCallback = t.Callable[["BaseComm", MessageType], None]


class BaseComm:
    """Class for communicating between a Frontend and a Kernel

    Must be subclassed with a publish_msg method implementation which
    sends comm messages through the iopub channel.
    """

    def __init__(
        self,
        target_name: str = "comm",
        data: MaybeDict = None,
        metadata: MaybeDict = None,
        buffers: BuffersType = None,
        comm_id: str | None = None,
        primary: bool = True,
        target_module: str | None = None,
        topic: bytes | None = None,
        _open_data: MaybeDict = None,
        _close_data: MaybeDict = None,
        **kwargs: t.Any,
    ) -> None:
        super().__init__(**kwargs)

        self.comm_id = comm_id if comm_id else uuid.uuid4().hex
        self.primary = primary
        self.target_name = target_name
        self.target_module = target_module
        self.topic = topic if topic else ("comm-%s" % self.comm_id).encode("ascii")

        self._open_data = _open_data if _open_data else {}
        self._close_data = _close_data if _close_data else {}

        self._msg_callback: CommCallback | None = None
        self._close_callback: CommCallback | None = None

        self._closed = True

        if self.primary:
            # I am primary, open my peer.
            self.open(data=data, metadata=metadata, buffers=buffers)
        else:
            self._closed = False

    def publish_msg(
        self,
        msg_type: str,  # noqa: ARG002
        data: MaybeDict = None,  # noqa: ARG002
        metadata: MaybeDict = None,  # noqa: ARG002
        buffers: BuffersType = None,  # noqa: ARG002
        **keys: t.Any,  # noqa: ARG002
    ) -> None:
        msg = "publish_msg Comm method is not implemented"
        raise NotImplementedError(msg)

    def __del__(self) -> None:
        """trigger close on gc"""
        with contextlib.suppress(Exception):
            # any number of things can have gone horribly wrong
            # when called during interpreter teardown
            self.close(deleting=True)

    # publishing messages

    def open(
        self, data: MaybeDict = None, metadata: MaybeDict = None, buffers: BuffersType = None
    ) -> None:
        """Open the frontend-side version of this comm"""

        if data is None:
            data = self._open_data
        comm_manager = comm.get_comm_manager()
        if comm_manager is None:
            msg = "Comms cannot be opened without a comm_manager."  # type:ignore[unreachable]
            raise RuntimeError(msg)

        comm_manager.register_comm(self)
        try:
            self.publish_msg(
                "comm_open",
                data=data,
                metadata=metadata,
                buffers=buffers,
                target_name=self.target_name,
                target_module=self.target_module,
            )
            self._closed = False
        except Exception:
            comm_manager.unregister_comm(self)
            raise

    def close(
        self,
        data: MaybeDict = None,
        metadata: MaybeDict = None,
        buffers: BuffersType = None,
        deleting: bool = False,
    ) -> None:
        """Close the frontend-side version of this comm"""
        if self._closed:
            # only close once
            return
        self._closed = True
        if data is None:
            data = self._close_data
        self.publish_msg(
            "comm_close",
            data=data,
            metadata=metadata,
            buffers=buffers,
        )
        if not deleting:
            # If deleting, the comm can't be registered
            comm.get_comm_manager().unregister_comm(self)

    def send(
        self, data: MaybeDict = None, metadata: MaybeDict = None, buffers: BuffersType = None
    ) -> None:
        """Send a message to the frontend-side version of this comm"""
        self.publish_msg(
            "comm_msg",
            data=data,
            metadata=metadata,
            buffers=buffers,
        )

    # registering callbacks

    def on_close(self, callback: CommCallback | None) -> None:
        """Register a callback for comm_close

        Will be called with the `data` of the close message.

        Call `on_close(None)` to disable an existing callback.
        """
        self._close_callback = callback

    def on_msg(self, callback: CommCallback | None) -> None:
        """Register a callback for comm_msg

        Will be called with the `data` of any comm_msg messages.

        Call `on_msg(None)` to disable an existing callback.
        """
        self._msg_callback = callback

    # handling of incoming messages

    def handle_close(self, msg: MessageType) -> None:
        """Handle a comm_close message"""
        logger.debug("handle_close[%s](%s)", self.comm_id, msg)
        if self._close_callback:
            self._close_callback(msg)

    def handle_msg(self, msg: MessageType) -> None:
        """Handle a comm_msg message"""
        logger.debug("handle_msg[%s](%s)", self.comm_id, msg)
        if self._msg_callback:
            from IPython import get_ipython

            shell = get_ipython()
            if shell:
                shell.events.trigger("pre_execute")
            self._msg_callback(msg)
            if shell:
                shell.events.trigger("post_execute")


class CommManager:
    """Default CommManager singleton implementation for Comms in the Kernel"""

    # Public APIs

    def __init__(self) -> None:
        self.comms: dict[str, BaseComm] = {}
        self.targets: dict[str, CommTargetCallback] = {}

    def register_target(self, target_name: str, f: CommTargetCallback | str) -> None:
        """Register a callable f for a given target name

        f will be called with two arguments when a comm_open message is received with `target`:

        - the Comm instance
        - the `comm_open` message itself.

        f can be a Python callable or an import string for one.
        """
        if isinstance(f, str):
            f = import_item(f)

        self.targets[target_name] = t.cast(CommTargetCallback, f)

    def unregister_target(self, target_name: str, f: CommTargetCallback) -> CommTargetCallback:  # noqa: ARG002
        """Unregister a callable registered with register_target"""
        return self.targets.pop(target_name)

    def register_comm(self, comm: BaseComm) -> str:
        """Register a new comm"""
        comm_id = comm.comm_id
        self.comms[comm_id] = comm
        return comm_id

    def unregister_comm(self, comm: BaseComm) -> None:
        """Unregister a comm, and close its counterpart"""
        # unlike get_comm, this should raise a KeyError
        comm = self.comms.pop(comm.comm_id)

    def get_comm(self, comm_id: str) -> BaseComm | None:
        """Get a comm with a particular id

        Returns the comm if found, otherwise None.

        This will not raise an error,
        it will log messages if the comm cannot be found.
        """
        try:
            return self.comms[comm_id]
        except KeyError:
            logger.warning("No such comm: %s", comm_id)
            if logger.isEnabledFor(logging.DEBUG):
                # don't create the list of keys if debug messages aren't enabled
                logger.debug("Current comms: %s", list(self.comms.keys()))
            return None

    # Message handlers

    def comm_open(self, stream: ZMQStream, ident: str, msg: MessageType) -> None:  # noqa: ARG002
        """Handler for comm_open messages"""
        from comm import create_comm

        content = msg["content"]
        comm_id = content["comm_id"]
        target_name = content["target_name"]
        f = self.targets.get(target_name, None)
        comm = create_comm(
            comm_id=comm_id,
            primary=False,
            target_name=target_name,
        )
        self.register_comm(comm)
        if f is None:
            logger.error("No such comm target registered: %s", target_name)
        else:
            try:
                f(comm, msg)
                return
            except Exception:
                logger.error("Exception opening comm with target: %s", target_name, exc_info=True)

        # Failure.
        try:
            comm.close()
        except Exception:
            logger.error(
                """Could not close comm during `comm_open` failure
                clean-up.  The comm may not have been opened yet.""",
                exc_info=True,
            )

    def comm_msg(self, stream: ZMQStream, ident: str, msg: MessageType) -> None:  # noqa: ARG002
        """Handler for comm_msg messages"""
        content = msg["content"]
        comm_id = content["comm_id"]
        comm = self.get_comm(comm_id)
        if comm is None:
            return

        try:
            comm.handle_msg(msg)
        except Exception:
            logger.error("Exception in comm_msg for %s", comm_id, exc_info=True)

    def comm_close(self, stream: ZMQStream, ident: str, msg: MessageType) -> None:  # noqa: ARG002
        """Handler for comm_close messages"""
        content = msg["content"]
        comm_id = content["comm_id"]
        comm = self.get_comm(comm_id)
        if comm is None:
            return

        self.comms[comm_id]._closed = True
        del self.comms[comm_id]

        try:
            comm.handle_close(msg)
        except Exception:
            logger.error("Exception in comm_close for %s", comm_id, exc_info=True)


__all__ = ["CommManager", "BaseComm"]
