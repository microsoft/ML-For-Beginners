"""A basic in process kernel monitor with autorestarting.

This watches a kernel's state using KernelManager.is_alive and auto
restarts the kernel if it dies.
"""
# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
import time
import warnings

from traitlets import Instance

from ..restarter import KernelRestarter


class IOLoopKernelRestarter(KernelRestarter):
    """Monitor and autorestart a kernel."""

    loop = Instance("tornado.ioloop.IOLoop")

    def _loop_default(self):
        warnings.warn(
            "IOLoopKernelRestarter.loop is deprecated in jupyter-client 5.2",
            DeprecationWarning,
            stacklevel=4,
        )
        from tornado import ioloop

        return ioloop.IOLoop.current()

    _pcallback = None

    def start(self):
        """Start the polling of the kernel."""
        if self._pcallback is None:
            from tornado.ioloop import PeriodicCallback

            self._pcallback = PeriodicCallback(
                self.poll,
                1000 * self.time_to_dead,
            )
            self._pcallback.start()

    def stop(self):
        """Stop the kernel polling."""
        if self._pcallback is not None:
            self._pcallback.stop()
            self._pcallback = None


class AsyncIOLoopKernelRestarter(IOLoopKernelRestarter):
    """An async io loop kernel restarter."""

    async def poll(self):
        """Poll the kernel."""
        if self.debug:
            self.log.debug("Polling kernel...")
        is_alive = await self.kernel_manager.is_alive()
        now = time.time()
        if not is_alive:
            self._last_dead = now
            if self._restarting:
                self._restart_count += 1
            else:
                self._restart_count = 1

            if self._restart_count > self.restart_limit:
                self.log.warning("AsyncIOLoopKernelRestarter: restart failed")
                self._fire_callbacks("dead")
                self._restarting = False
                self._restart_count = 0
                self.stop()
            else:
                newports = self.random_ports_until_alive and self._initial_startup
                self.log.info(
                    "AsyncIOLoopKernelRestarter: restarting kernel (%i/%i), %s random ports",
                    self._restart_count,
                    self.restart_limit,
                    "new" if newports else "keep",
                )
                self._fire_callbacks("restart")
                await self.kernel_manager.restart_kernel(now=True, newports=newports)
                self._restarting = True
        else:
            # Since `is_alive` only tests that the kernel process is alive, it does not
            # indicate that the kernel has successfully completed startup. To solve this
            # correctly, we would need to wait for a kernel info reply, but it is not
            # necessarily appropriate to start a kernel client + channels in the
            # restarter. Therefore, we use "has been alive continuously for X time" as a
            # heuristic for a stable start up.
            # See https://github.com/jupyter/jupyter_client/pull/717 for details.
            stable_start_time = self.stable_start_time
            if self.kernel_manager.provisioner:
                stable_start_time = self.kernel_manager.provisioner.get_stable_start_time(
                    recommended=stable_start_time
                )
            if self._initial_startup and now - self._last_dead >= stable_start_time:
                self._initial_startup = False
            if self._restarting and now - self._last_dead >= stable_start_time:
                self.log.debug("AsyncIOLoopKernelRestarter: restart apparently succeeded")
                self._restarting = False
