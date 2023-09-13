"""A trio loop runner."""
import builtins
import logging
import signal
import threading
import traceback
import warnings

import trio


class TrioRunner:
    """A trio loop runner."""

    def __init__(self):
        """Initialize the runner."""
        self._cell_cancel_scope = None
        self._trio_token = None

    def initialize(self, kernel, io_loop):
        """Initialize the runner."""
        kernel.shell.set_trio_runner(self)
        kernel.shell.run_line_magic("autoawait", "trio")
        kernel.shell.magics_manager.magics["line"]["autoawait"] = lambda _: warnings.warn(
            "Autoawait isn't allowed in Trio background loop mode.", stacklevel=2
        )
        self._interrupted = False
        bg_thread = threading.Thread(target=io_loop.start, daemon=True, name="TornadoBackground")
        bg_thread.start()

    def interrupt(self, signum, frame):
        """Interuppt the runner."""
        if self._cell_cancel_scope:
            self._cell_cancel_scope.cancel()
        else:
            msg = "Kernel interrupted but no cell is running"
            raise Exception(msg)

    def run(self):
        """Run the loop."""
        old_sig = signal.signal(signal.SIGINT, self.interrupt)

        def log_nursery_exc(exc):
            exc = "\n".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            logging.error("An exception occurred in a global nursery task.\n%s", exc)

        async def trio_main():
            """Run the main loop."""
            self._trio_token = trio.lowlevel.current_trio_token()
            async with trio.open_nursery() as nursery:
                # TODO This hack prevents the nursery from cancelling all child
                # tasks when an uncaught exception occurs, but it's ugly.
                nursery._add_exc = log_nursery_exc
                builtins.GLOBAL_NURSERY = nursery  # type:ignore[attr-defined]
                await trio.sleep_forever()

        trio.run(trio_main)
        signal.signal(signal.SIGINT, old_sig)

    def __call__(self, async_fn):
        """Handle a function call."""

        async def loc(coro):
            """A thread runner context."""
            self._cell_cancel_scope = trio.CancelScope()
            with self._cell_cancel_scope:
                return await coro
            self._cell_cancel_scope = None

        return trio.from_thread.run(loc, async_fn, trio_token=self._trio_token)
