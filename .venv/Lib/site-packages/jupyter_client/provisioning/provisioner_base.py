"""Kernel Provisioner Classes"""
# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
import os
from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Union

from traitlets.config import Instance, LoggingConfigurable, Unicode

from ..connect import KernelConnectionInfo


class KernelProvisionerMeta(ABCMeta, type(LoggingConfigurable)):  # type: ignore[misc]
    pass


class KernelProvisionerBase(  # type:ignore[misc]
    ABC, LoggingConfigurable, metaclass=KernelProvisionerMeta
):
    """
    Abstract base class defining methods for KernelProvisioner classes.

    A majority of methods are abstract (requiring implementations via a subclass) while
    some are optional and others provide implementations common to all instances.
    Subclasses should be aware of which methods require a call to the superclass.

    Many of these methods model those of :class:`subprocess.Popen` for parity with
    previous versions where the kernel process was managed directly.
    """

    # The kernel specification associated with this provisioner
    kernel_spec: Any = Instance("jupyter_client.kernelspec.KernelSpec", allow_none=True)
    kernel_id: Union[str, Unicode] = Unicode(None, allow_none=True)
    connection_info: KernelConnectionInfo = {}

    @property
    @abstractmethod
    def has_process(self) -> bool:
        """
        Returns true if this provisioner is currently managing a process.

        This property is asserted to be True immediately following a call to
        the provisioner's :meth:`launch_kernel` method.
        """
        pass

    @abstractmethod
    async def poll(self) -> Optional[int]:
        """
        Checks if kernel process is still running.

        If running, None is returned, otherwise the process's integer-valued exit code is returned.
        This method is called from :meth:`KernelManager.is_alive`.
        """
        pass

    @abstractmethod
    async def wait(self) -> Optional[int]:
        """
        Waits for kernel process to terminate.

        This method is called from `KernelManager.finish_shutdown()` and
        `KernelManager.kill_kernel()` when terminating a kernel gracefully or
        immediately, respectively.
        """
        pass

    @abstractmethod
    async def send_signal(self, signum: int) -> None:
        """
        Sends signal identified by signum to the kernel process.

        This method is called from `KernelManager.signal_kernel()` to send the
        kernel process a signal.
        """
        pass

    @abstractmethod
    async def kill(self, restart: bool = False) -> None:
        """
        Kill the kernel process.

        This is typically accomplished via a SIGKILL signal, which cannot be caught.
        This method is called from `KernelManager.kill_kernel()` when terminating
        a kernel immediately.

        restart is True if this operation will precede a subsequent launch_kernel request.
        """
        pass

    @abstractmethod
    async def terminate(self, restart: bool = False) -> None:
        """
        Terminates the kernel process.

        This is typically accomplished via a SIGTERM signal, which can be caught, allowing
        the kernel provisioner to perform possible cleanup of resources.  This method is
        called indirectly from `KernelManager.finish_shutdown()` during a kernel's
        graceful termination.

        restart is True if this operation precedes a start launch_kernel request.
        """
        pass

    @abstractmethod
    async def launch_kernel(self, cmd: List[str], **kwargs: Any) -> KernelConnectionInfo:
        """
        Launch the kernel process and return its connection information.

        This method is called from `KernelManager.launch_kernel()` during the
        kernel manager's start kernel sequence.
        """
        pass

    @abstractmethod
    async def cleanup(self, restart: bool = False) -> None:
        """
        Cleanup any resources allocated on behalf of the kernel provisioner.

        This method is called from `KernelManager.cleanup_resources()` as part of
        its shutdown kernel sequence.

        restart is True if this operation precedes a start launch_kernel request.
        """
        pass

    async def shutdown_requested(self, restart: bool = False) -> None:
        """
        Allows the provisioner to determine if the kernel's shutdown has been requested.

        This method is called from `KernelManager.request_shutdown()` as part of
        its shutdown sequence.

        This method is optional and is primarily used in scenarios where the provisioner
        may need to perform other operations in preparation for a kernel's shutdown.
        """
        pass

    async def pre_launch(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Perform any steps in preparation for kernel process launch.

        This includes applying additional substitutions to the kernel launch command
        and environment. It also includes preparation of launch parameters.

        NOTE: Subclass implementations are advised to call this method as it applies
        environment variable substitutions from the local environment and calls the
        provisioner's :meth:`_finalize_env()` method to allow each provisioner the
        ability to cleanup the environment variables that will be used by the kernel.

        This method is called from `KernelManager.pre_start_kernel()` as part of its
        start kernel sequence.

        Returns the (potentially updated) keyword arguments that are passed to
        :meth:`launch_kernel()`.
        """
        env = kwargs.pop("env", os.environ).copy()
        env.update(self.__apply_env_substitutions(env))
        self._finalize_env(env)
        kwargs["env"] = env

        return kwargs

    async def post_launch(self, **kwargs: Any) -> None:
        """
        Perform any steps following the kernel process launch.

        This method is called from `KernelManager.post_start_kernel()` as part of its
        start kernel sequence.
        """
        pass

    async def get_provisioner_info(self) -> Dict[str, Any]:
        """
        Captures the base information necessary for persistence relative to this instance.

        This enables applications that subclass `KernelManager` to persist a kernel provisioner's
        relevant information to accomplish functionality like disaster recovery or high availability
        by calling this method via the kernel manager's `provisioner` attribute.

        NOTE: The superclass method must always be called first to ensure proper serialization.
        """
        provisioner_info: Dict[str, Any] = {}
        provisioner_info["kernel_id"] = self.kernel_id
        provisioner_info["connection_info"] = self.connection_info
        return provisioner_info

    async def load_provisioner_info(self, provisioner_info: Dict) -> None:
        """
        Loads the base information necessary for persistence relative to this instance.

        The inverse of `get_provisioner_info()`, this enables applications that subclass
        `KernelManager` to re-establish communication with a provisioner that is managing
        a (presumably) remote kernel from an entirely different process that the original
        provisioner.

        NOTE: The superclass method must always be called first to ensure proper deserialization.
        """
        self.kernel_id = provisioner_info["kernel_id"]
        self.connection_info = provisioner_info["connection_info"]

    def get_shutdown_wait_time(self, recommended: float = 5.0) -> float:
        """
        Returns the time allowed for a complete shutdown. This may vary by provisioner.

        This method is called from `KernelManager.finish_shutdown()` during the graceful
        phase of its kernel shutdown sequence.

        The recommended value will typically be what is configured in the kernel manager.
        """
        return recommended

    def get_stable_start_time(self, recommended: float = 10.0) -> float:
        """
        Returns the expected upper bound for a kernel (re-)start to complete.
        This may vary by provisioner.

        The recommended value will typically be what is configured in the kernel restarter.
        """
        return recommended

    def _finalize_env(self, env: Dict[str, str]) -> None:
        """
        Ensures env is appropriate prior to launch.

        This method is called from `KernelProvisionerBase.pre_launch()` during the kernel's
        start sequence.

        NOTE: Subclasses should be sure to call super()._finalize_env(env)
        """
        if self.kernel_spec.language and self.kernel_spec.language.lower().startswith("python"):
            # Don't allow PYTHONEXECUTABLE to be passed to kernel process.
            # If set, it can bork all the things.
            env.pop("PYTHONEXECUTABLE", None)

    def __apply_env_substitutions(self, substitution_values: Dict[str, str]) -> Dict[str, str]:
        """
        Walks entries in the kernelspec's env stanza and applies substitutions from current env.

        This method is called from `KernelProvisionerBase.pre_launch()` during the kernel's
        start sequence.

        Returns the substituted list of env entries.

        NOTE: This method is private and is not intended to be overridden by provisioners.
        """
        substituted_env = {}
        if self.kernel_spec:
            from string import Template

            # For each templated env entry, fill any templated references
            # matching names of env variables with those values and build
            # new dict with substitutions.
            templated_env = self.kernel_spec.env
            for k, v in templated_env.items():
                substituted_env.update({k: Template(v).safe_substitute(substitution_values)})
        return substituted_env
