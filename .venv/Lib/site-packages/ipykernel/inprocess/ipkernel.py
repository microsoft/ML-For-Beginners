"""An in-process kernel"""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

import logging
import sys
from contextlib import contextmanager

from IPython.core.interactiveshell import InteractiveShellABC
from traitlets import Any, Enum, Instance, List, Type, default

from ipykernel.ipkernel import IPythonKernel
from ipykernel.jsonutil import json_clean
from ipykernel.zmqshell import ZMQInteractiveShell

from ..iostream import BackgroundSocket, IOPubThread, OutStream
from .constants import INPROCESS_KEY
from .socket import DummySocket

# -----------------------------------------------------------------------------
# Main kernel class
# -----------------------------------------------------------------------------


class InProcessKernel(IPythonKernel):
    """An in-process kernel."""

    # -------------------------------------------------------------------------
    # InProcessKernel interface
    # -------------------------------------------------------------------------

    # The frontends connected to this kernel.
    frontends = List(Instance("ipykernel.inprocess.client.InProcessKernelClient", allow_none=True))

    # The GUI environment that the kernel is running under. This need not be
    # specified for the normal operation for the kernel, but is required for
    # IPython's GUI support (including pylab). The default is 'inline' because
    # it is safe under all GUI toolkits.
    gui = Enum(("tk", "gtk", "wx", "qt", "qt4", "inline"), default_value="inline")

    raw_input_str = Any()
    stdout = Any()
    stderr = Any()

    # -------------------------------------------------------------------------
    # Kernel interface
    # -------------------------------------------------------------------------

    shell_class = Type(allow_none=True)  # type:ignore[assignment]
    _underlying_iopub_socket = Instance(DummySocket, ())
    iopub_thread: IOPubThread = Instance(IOPubThread)  # type:ignore[assignment]

    shell_stream = Instance(DummySocket, ())  # type:ignore[arg-type]

    @default("iopub_thread")
    def _default_iopub_thread(self):
        thread = IOPubThread(self._underlying_iopub_socket)
        thread.start()
        return thread

    iopub_socket: BackgroundSocket = Instance(BackgroundSocket)  # type:ignore[assignment]

    @default("iopub_socket")
    def _default_iopub_socket(self):
        return self.iopub_thread.background_socket

    stdin_socket = Instance(DummySocket, ())  # type:ignore[assignment]

    def __init__(self, **traits):
        """Initialize the kernel."""
        super().__init__(**traits)

        self._underlying_iopub_socket.observe(self._io_dispatch, names=["message_sent"])
        if self.shell:
            self.shell.kernel = self

    async def execute_request(self, stream, ident, parent):
        """Override for temporary IO redirection."""
        with self._redirected_io():
            await super().execute_request(stream, ident, parent)

    def start(self):
        """Override registration of dispatchers for streams."""
        if self.shell:
            self.shell.exit_now = False

    def _abort_queues(self):
        """The in-process kernel doesn't abort requests."""

    async def _flush_control_queue(self):
        """No need to flush control queues for in-process"""

    def _input_request(self, prompt, ident, parent, password=False):
        # Flush output before making the request.
        self.raw_input_str = None
        sys.stderr.flush()
        sys.stdout.flush()

        # Send the input request.
        content = json_clean(dict(prompt=prompt, password=password))
        assert self.session is not None
        msg = self.session.msg("input_request", content, parent)
        for frontend in self.frontends:
            assert frontend is not None
            if frontend.session.session == parent["header"]["session"]:
                frontend.stdin_channel.call_handlers(msg)
                break
        else:
            logging.error("No frontend found for raw_input request")
            return ""

        # Await a response.
        while self.raw_input_str is None:
            frontend.stdin_channel.process_events()
        return self.raw_input_str  # type:ignore[unreachable]

    # -------------------------------------------------------------------------
    # Protected interface
    # -------------------------------------------------------------------------

    @contextmanager
    def _redirected_io(self):
        """Temporarily redirect IO to the kernel."""
        sys_stdout, sys_stderr = sys.stdout, sys.stderr
        try:
            sys.stdout, sys.stderr = self.stdout, self.stderr
            yield
        finally:
            sys.stdout, sys.stderr = sys_stdout, sys_stderr

    # ------ Trait change handlers --------------------------------------------

    def _io_dispatch(self, change):
        """Called when a message is sent to the IO socket."""
        assert self.iopub_socket.io_thread is not None
        assert self.session is not None
        ident, msg = self.session.recv(self.iopub_socket.io_thread.socket, copy=False)
        for frontend in self.frontends:
            assert frontend is not None
            frontend.iopub_channel.call_handlers(msg)

    # ------ Trait initializers -----------------------------------------------

    @default("log")
    def _default_log(self):
        return logging.getLogger(__name__)

    @default("session")
    def _default_session(self):
        from jupyter_client.session import Session

        return Session(parent=self, key=INPROCESS_KEY)

    @default("shell_class")
    def _default_shell_class(self):
        return InProcessInteractiveShell

    @default("stdout")
    def _default_stdout(self):
        return OutStream(self.session, self.iopub_thread, "stdout", watchfd=False)

    @default("stderr")
    def _default_stderr(self):
        return OutStream(self.session, self.iopub_thread, "stderr", watchfd=False)


# -----------------------------------------------------------------------------
# Interactive shell subclass
# -----------------------------------------------------------------------------


class InProcessInteractiveShell(ZMQInteractiveShell):
    """An in-process interactive shell."""

    kernel: InProcessKernel = Instance(
        "ipykernel.inprocess.ipkernel.InProcessKernel", allow_none=True
    )  # type:ignore[assignment]

    # -------------------------------------------------------------------------
    # InteractiveShell interface
    # -------------------------------------------------------------------------

    def enable_gui(self, gui=None):
        """Enable GUI integration for the kernel."""
        if not gui:
            gui = self.kernel.gui
        self.active_eventloop = gui

    def enable_matplotlib(self, gui=None):
        """Enable matplotlib integration for the kernel."""
        if not gui:
            gui = self.kernel.gui
        return super().enable_matplotlib(gui)

    def enable_pylab(self, gui=None, import_all=True, welcome_message=False):
        """Activate pylab support at runtime."""
        if not gui:
            gui = self.kernel.gui
        return super().enable_pylab(gui, import_all, welcome_message)


InteractiveShellABC.register(InProcessInteractiveShell)
