"""A kernel manager for multiple kernels"""
# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
from __future__ import annotations

import asyncio
import json
import os
import socket
import typing as t
import uuid
from functools import wraps
from pathlib import Path

import zmq
from traitlets import Any, Bool, Dict, DottedObjectName, Instance, Unicode, default, observe
from traitlets.config.configurable import LoggingConfigurable
from traitlets.utils.importstring import import_item

from .connect import KernelConnectionInfo
from .kernelspec import NATIVE_KERNEL_NAME, KernelSpecManager
from .manager import KernelManager
from .utils import ensure_async, run_sync, utcnow


class DuplicateKernelError(Exception):
    pass


def kernel_method(f: t.Callable) -> t.Callable:
    """decorator for proxying MKM.method(kernel_id) to individual KMs by ID"""

    @wraps(f)
    def wrapped(
        self: t.Any, kernel_id: str, *args: t.Any, **kwargs: t.Any
    ) -> t.Callable | t.Awaitable:
        # get the kernel
        km = self.get_kernel(kernel_id)
        method = getattr(km, f.__name__)
        # call the kernel's method
        r = method(*args, **kwargs)
        # last thing, call anything defined in the actual class method
        # such as logging messages
        f(self, kernel_id, *args, **kwargs)
        # return the method result
        return r

    return wrapped


class MultiKernelManager(LoggingConfigurable):
    """A class for managing multiple kernels."""

    default_kernel_name = Unicode(
        NATIVE_KERNEL_NAME, help="The name of the default kernel to start"
    ).tag(config=True)

    kernel_spec_manager = Instance(KernelSpecManager, allow_none=True)

    kernel_manager_class = DottedObjectName(
        "jupyter_client.ioloop.IOLoopKernelManager",
        help="""The kernel manager class.  This is configurable to allow
        subclassing of the KernelManager for customized behavior.
        """,
    ).tag(config=True)

    @observe("kernel_manager_class")
    def _kernel_manager_class_changed(self, change: t.Any) -> None:
        self.kernel_manager_factory = self._create_kernel_manager_factory()

    kernel_manager_factory = Any(help="this is kernel_manager_class after import")

    @default("kernel_manager_factory")
    def _kernel_manager_factory_default(self) -> t.Callable:
        return self._create_kernel_manager_factory()

    def _create_kernel_manager_factory(self) -> t.Callable:
        kernel_manager_ctor = import_item(self.kernel_manager_class)

        def create_kernel_manager(*args: t.Any, **kwargs: t.Any) -> KernelManager:
            if self.shared_context:
                if self.context.closed:
                    # recreate context if closed
                    self.context = self._context_default()
                kwargs.setdefault("context", self.context)
            km = kernel_manager_ctor(*args, **kwargs)
            return km

        return create_kernel_manager

    shared_context = Bool(
        True,
        help="Share a single zmq.Context to talk to all my kernels",
    ).tag(config=True)

    context = Instance("zmq.Context")

    _created_context = Bool(False)

    _pending_kernels = Dict()

    @property
    def _starting_kernels(self) -> dict:
        """A shim for backwards compatibility."""
        return self._pending_kernels

    @default("context")
    def _context_default(self) -> zmq.Context:
        self._created_context = True
        return zmq.Context()

    connection_dir = Unicode("")
    external_connection_dir = Unicode(None, allow_none=True)

    _kernels = Dict()

    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        super().__init__(*args, **kwargs)
        self.kernel_id_to_connection_file: dict[str, Path] = {}

    def __del__(self) -> None:
        """Handle garbage collection.  Destroy context if applicable."""
        if self._created_context and self.context and not self.context.closed:
            if self.log:
                self.log.debug("Destroying zmq context for %s", self)
            self.context.destroy()
        try:
            super_del = super().__del__  # type:ignore[misc]
        except AttributeError:
            pass
        else:
            super_del()

    def list_kernel_ids(self) -> list[str]:
        """Return a list of the kernel ids of the active kernels."""
        if self.external_connection_dir is not None:
            external_connection_dir = Path(self.external_connection_dir)
            if external_connection_dir.is_dir():
                connection_files = [p for p in external_connection_dir.iterdir() if p.is_file()]

                # remove kernels (whose connection file has disappeared) from our list
                k = list(self.kernel_id_to_connection_file.keys())
                v = list(self.kernel_id_to_connection_file.values())
                for connection_file in list(self.kernel_id_to_connection_file.values()):
                    if connection_file not in connection_files:
                        kernel_id = k[v.index(connection_file)]
                        del self.kernel_id_to_connection_file[kernel_id]
                        del self._kernels[kernel_id]

                # add kernels (whose connection file appeared) to our list
                for connection_file in connection_files:
                    if connection_file in self.kernel_id_to_connection_file.values():
                        continue
                    try:
                        connection_info: KernelConnectionInfo = json.loads(
                            connection_file.read_text()
                        )
                    except Exception:  # noqa: S112
                        continue
                    self.log.debug("Loading connection file %s", connection_file)
                    if not ("kernel_name" in connection_info and "key" in connection_info):
                        continue
                    # it looks like a connection file
                    kernel_id = self.new_kernel_id()
                    self.kernel_id_to_connection_file[kernel_id] = connection_file
                    km = self.kernel_manager_factory(
                        parent=self,
                        log=self.log,
                        owns_kernel=False,
                    )
                    km.load_connection_info(connection_info)
                    km.last_activity = utcnow()
                    km.execution_state = "idle"
                    km.connections = 1
                    km.kernel_id = kernel_id
                    km.kernel_name = connection_info["kernel_name"]
                    km.ready.set_result(None)

                    self._kernels[kernel_id] = km

        # Create a copy so we can iterate over kernels in operations
        # that delete keys.
        return list(self._kernels.keys())

    def __len__(self) -> int:
        """Return the number of running kernels."""
        return len(self.list_kernel_ids())

    def __contains__(self, kernel_id: str) -> bool:
        return kernel_id in self._kernels

    def pre_start_kernel(
        self, kernel_name: str | None, kwargs: t.Any
    ) -> tuple[KernelManager, str, str]:
        # kwargs should be mutable, passing it as a dict argument.
        kernel_id = kwargs.pop("kernel_id", self.new_kernel_id(**kwargs))
        if kernel_id in self:
            raise DuplicateKernelError("Kernel already exists: %s" % kernel_id)

        if kernel_name is None:
            kernel_name = self.default_kernel_name
        # kernel_manager_factory is the constructor for the KernelManager
        # subclass we are using. It can be configured as any Configurable,
        # including things like its transport and ip.
        constructor_kwargs = {}
        if self.kernel_spec_manager:
            constructor_kwargs["kernel_spec_manager"] = self.kernel_spec_manager
        km = self.kernel_manager_factory(
            connection_file=os.path.join(self.connection_dir, "kernel-%s.json" % kernel_id),
            parent=self,
            log=self.log,
            kernel_name=kernel_name,
            **constructor_kwargs,
        )
        return km, kernel_name, kernel_id

    def update_env(self, *, kernel_id: str, env: t.Dict[str, str]) -> None:
        """
        Allow to update the environment of the given kernel.

        Forward the update env request to the corresponding kernel.

        .. version-added: 8.5
        """
        if kernel_id in self:
            self._kernels[kernel_id].update_env(env=env)

    async def _add_kernel_when_ready(
        self, kernel_id: str, km: KernelManager, kernel_awaitable: t.Awaitable
    ) -> None:
        try:
            await kernel_awaitable
            self._kernels[kernel_id] = km
            self._pending_kernels.pop(kernel_id, None)
        except Exception as e:
            self.log.exception(e)

    async def _remove_kernel_when_ready(
        self, kernel_id: str, kernel_awaitable: t.Awaitable
    ) -> None:
        try:
            await kernel_awaitable
            self.remove_kernel(kernel_id)
            self._pending_kernels.pop(kernel_id, None)
        except Exception as e:
            self.log.exception(e)

    def _using_pending_kernels(self) -> bool:
        """Returns a boolean; a clearer method for determining if
        this multikernelmanager is using pending kernels or not
        """
        return getattr(self, "use_pending_kernels", False)

    async def _async_start_kernel(self, *, kernel_name: str | None = None, **kwargs: t.Any) -> str:
        """Start a new kernel.

        The caller can pick a kernel_id by passing one in as a keyword arg,
        otherwise one will be generated using new_kernel_id().

        The kernel ID for the newly started kernel is returned.
        """
        km, kernel_name, kernel_id = self.pre_start_kernel(kernel_name, kwargs)
        if not isinstance(km, KernelManager):
            self.log.warning(  # type:ignore[unreachable]
                "Kernel manager class ({km_class}) is not an instance of 'KernelManager'!".format(
                    km_class=self.kernel_manager_class.__class__
                )
            )
        kwargs["kernel_id"] = kernel_id  # Make kernel_id available to manager and provisioner

        starter = ensure_async(km.start_kernel(**kwargs))
        task = asyncio.create_task(self._add_kernel_when_ready(kernel_id, km, starter))
        self._pending_kernels[kernel_id] = task
        # Handling a Pending Kernel
        if self._using_pending_kernels():
            # If using pending kernels, do not block
            # on the kernel start.
            self._kernels[kernel_id] = km
        else:
            await task
            # raise an exception if one occurred during kernel startup.
            if km.ready.exception():
                raise km.ready.exception()  # type: ignore[misc]

        return kernel_id

    start_kernel = run_sync(_async_start_kernel)

    async def _async_shutdown_kernel(
        self,
        kernel_id: str,
        now: bool | None = False,
        restart: bool | None = False,
    ) -> None:
        """Shutdown a kernel by its kernel uuid.

        Parameters
        ==========
        kernel_id : uuid
            The id of the kernel to shutdown.
        now : bool
            Should the kernel be shutdown forcibly using a signal.
        restart : bool
            Will the kernel be restarted?
        """
        self.log.info("Kernel shutdown: %s", kernel_id)
        # If the kernel is still starting, wait for it to be ready.
        if kernel_id in self._pending_kernels:
            task = self._pending_kernels[kernel_id]
            try:
                await task
                km = self.get_kernel(kernel_id)
                await t.cast(asyncio.Future, km.ready)
            except asyncio.CancelledError:
                pass
            except Exception:
                self.remove_kernel(kernel_id)
                return
        km = self.get_kernel(kernel_id)
        # If a pending kernel raised an exception, remove it.
        if not km.ready.cancelled() and km.ready.exception():
            self.remove_kernel(kernel_id)
            return
        stopper = ensure_async(km.shutdown_kernel(now, restart))
        fut = asyncio.ensure_future(self._remove_kernel_when_ready(kernel_id, stopper))
        self._pending_kernels[kernel_id] = fut
        # Await the kernel if not using pending kernels.
        if not self._using_pending_kernels():
            await fut
            # raise an exception if one occurred during kernel shutdown.
            if km.ready.exception():
                raise km.ready.exception()  # type: ignore[misc]

    shutdown_kernel = run_sync(_async_shutdown_kernel)

    @kernel_method
    def request_shutdown(self, kernel_id: str, restart: bool | None = False) -> None:
        """Ask a kernel to shut down by its kernel uuid"""

    @kernel_method
    def finish_shutdown(
        self,
        kernel_id: str,
        waittime: float | None = None,
        pollinterval: float | None = 0.1,
    ) -> None:
        """Wait for a kernel to finish shutting down, and kill it if it doesn't"""
        self.log.info("Kernel shutdown: %s", kernel_id)

    @kernel_method
    def cleanup_resources(self, kernel_id: str, restart: bool = False) -> None:
        """Clean up a kernel's resources"""

    def remove_kernel(self, kernel_id: str) -> KernelManager:
        """remove a kernel from our mapping.

        Mainly so that a kernel can be removed if it is already dead,
        without having to call shutdown_kernel.

        The kernel object is returned, or `None` if not found.
        """
        return self._kernels.pop(kernel_id, None)

    async def _async_shutdown_all(self, now: bool = False) -> None:
        """Shutdown all kernels."""
        kids = self.list_kernel_ids()
        kids += list(self._pending_kernels)
        kms = list(self._kernels.values())
        futs = [self._async_shutdown_kernel(kid, now=now) for kid in set(kids)]
        await asyncio.gather(*futs)
        # If using pending kernels, the kernels will not have been fully shut down.
        if self._using_pending_kernels():
            for km in kms:
                try:
                    await km.ready
                except asyncio.CancelledError:
                    self._pending_kernels[km.kernel_id].cancel()
                except Exception:
                    # Will have been logged in _add_kernel_when_ready
                    pass

    shutdown_all = run_sync(_async_shutdown_all)

    def interrupt_kernel(self, kernel_id: str) -> None:
        """Interrupt (SIGINT) the kernel by its uuid.

        Parameters
        ==========
        kernel_id : uuid
            The id of the kernel to interrupt.
        """
        kernel = self.get_kernel(kernel_id)
        if not kernel.ready.done():
            msg = "Kernel is in a pending state. Cannot interrupt."
            raise RuntimeError(msg)
        out = kernel.interrupt_kernel()
        self.log.info("Kernel interrupted: %s", kernel_id)
        return out

    @kernel_method
    def signal_kernel(self, kernel_id: str, signum: int) -> None:
        """Sends a signal to the kernel by its uuid.

        Note that since only SIGTERM is supported on Windows, this function
        is only useful on Unix systems.

        Parameters
        ==========
        kernel_id : uuid
            The id of the kernel to signal.
        signum : int
            Signal number to send kernel.
        """
        self.log.info("Signaled Kernel %s with %s", kernel_id, signum)

    async def _async_restart_kernel(self, kernel_id: str, now: bool = False) -> None:
        """Restart a kernel by its uuid, keeping the same ports.

        Parameters
        ==========
        kernel_id : uuid
            The id of the kernel to interrupt.
        now : bool, optional
            If True, the kernel is forcefully restarted *immediately*, without
            having a chance to do any cleanup action.  Otherwise the kernel is
            given 1s to clean up before a forceful restart is issued.

            In all cases the kernel is restarted, the only difference is whether
            it is given a chance to perform a clean shutdown or not.
        """
        kernel = self.get_kernel(kernel_id)
        if self._using_pending_kernels() and not kernel.ready.done():
            msg = "Kernel is in a pending state. Cannot restart."
            raise RuntimeError(msg)
        await ensure_async(kernel.restart_kernel(now=now))
        self.log.info("Kernel restarted: %s", kernel_id)

    restart_kernel = run_sync(_async_restart_kernel)

    @kernel_method
    def is_alive(self, kernel_id: str) -> bool:  # type:ignore[empty-body]
        """Is the kernel alive.

        This calls KernelManager.is_alive() which calls Popen.poll on the
        actual kernel subprocess.

        Parameters
        ==========
        kernel_id : uuid
            The id of the kernel.
        """

    def _check_kernel_id(self, kernel_id: str) -> None:
        """check that a kernel id is valid"""
        if kernel_id not in self:
            raise KeyError("Kernel with id not found: %s" % kernel_id)

    def get_kernel(self, kernel_id: str) -> KernelManager:
        """Get the single KernelManager object for a kernel by its uuid.

        Parameters
        ==========
        kernel_id : uuid
            The id of the kernel.
        """
        self._check_kernel_id(kernel_id)
        return self._kernels[kernel_id]

    @kernel_method
    def add_restart_callback(
        self, kernel_id: str, callback: t.Callable, event: str = "restart"
    ) -> None:
        """add a callback for the KernelRestarter"""

    @kernel_method
    def remove_restart_callback(
        self, kernel_id: str, callback: t.Callable, event: str = "restart"
    ) -> None:
        """remove a callback for the KernelRestarter"""

    @kernel_method
    def get_connection_info(self, kernel_id: str) -> dict[str, t.Any]:  # type:ignore[empty-body]
        """Return a dictionary of connection data for a kernel.

        Parameters
        ==========
        kernel_id : uuid
            The id of the kernel.

        Returns
        =======
        connection_dict : dict
            A dict of the information needed to connect to a kernel.
            This includes the ip address and the integer port
            numbers of the different channels (stdin_port, iopub_port,
            shell_port, hb_port).
        """

    @kernel_method
    def connect_iopub(  # type:ignore[empty-body]
        self, kernel_id: str, identity: bytes | None = None
    ) -> socket.socket:
        """Return a zmq Socket connected to the iopub channel.

        Parameters
        ==========
        kernel_id : uuid
            The id of the kernel
        identity : bytes (optional)
            The zmq identity of the socket

        Returns
        =======
        stream : zmq Socket or ZMQStream
        """

    @kernel_method
    def connect_shell(  # type:ignore[empty-body]
        self, kernel_id: str, identity: bytes | None = None
    ) -> socket.socket:
        """Return a zmq Socket connected to the shell channel.

        Parameters
        ==========
        kernel_id : uuid
            The id of the kernel
        identity : bytes (optional)
            The zmq identity of the socket

        Returns
        =======
        stream : zmq Socket or ZMQStream
        """

    @kernel_method
    def connect_control(  # type:ignore[empty-body]
        self, kernel_id: str, identity: bytes | None = None
    ) -> socket.socket:
        """Return a zmq Socket connected to the control channel.

        Parameters
        ==========
        kernel_id : uuid
            The id of the kernel
        identity : bytes (optional)
            The zmq identity of the socket

        Returns
        =======
        stream : zmq Socket or ZMQStream
        """

    @kernel_method
    def connect_stdin(  # type:ignore[empty-body]
        self, kernel_id: str, identity: bytes | None = None
    ) -> socket.socket:
        """Return a zmq Socket connected to the stdin channel.

        Parameters
        ==========
        kernel_id : uuid
            The id of the kernel
        identity : bytes (optional)
            The zmq identity of the socket

        Returns
        =======
        stream : zmq Socket or ZMQStream
        """

    @kernel_method
    def connect_hb(  # type:ignore[empty-body]
        self, kernel_id: str, identity: bytes | None = None
    ) -> socket.socket:
        """Return a zmq Socket connected to the hb channel.

        Parameters
        ==========
        kernel_id : uuid
            The id of the kernel
        identity : bytes (optional)
            The zmq identity of the socket

        Returns
        =======
        stream : zmq Socket or ZMQStream
        """

    def new_kernel_id(self, **kwargs: t.Any) -> str:
        """
        Returns the id to associate with the kernel for this request. Subclasses may override
        this method to substitute other sources of kernel ids.
        :param kwargs:
        :return: string-ized version 4 uuid
        """
        return str(uuid.uuid4())


class AsyncMultiKernelManager(MultiKernelManager):
    kernel_manager_class = DottedObjectName(
        "jupyter_client.ioloop.AsyncIOLoopKernelManager",
        config=True,
        help="""The kernel manager class.  This is configurable to allow
        subclassing of the AsyncKernelManager for customized behavior.
        """,
    )

    use_pending_kernels = Bool(
        False,
        help="""Whether to make kernels available before the process has started.  The
        kernel has a `.ready` future which can be awaited before connecting""",
    ).tag(config=True)

    context = Instance("zmq.asyncio.Context")

    @default("context")
    def _context_default(self) -> zmq.asyncio.Context:
        self._created_context = True
        return zmq.asyncio.Context()

    start_kernel: t.Callable[..., t.Awaitable] = MultiKernelManager._async_start_kernel  # type:ignore[assignment]
    restart_kernel: t.Callable[..., t.Awaitable] = MultiKernelManager._async_restart_kernel  # type:ignore[assignment]
    shutdown_kernel: t.Callable[..., t.Awaitable] = MultiKernelManager._async_shutdown_kernel  # type:ignore[assignment]
    shutdown_all: t.Callable[..., t.Awaitable] = MultiKernelManager._async_shutdown_all  # type:ignore[assignment]
