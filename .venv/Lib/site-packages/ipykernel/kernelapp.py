"""An Application for launching a kernel"""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.
from __future__ import annotations

import atexit
import errno
import logging
import os
import signal
import sys
import traceback
import typing as t
from functools import partial
from io import FileIO, TextIOWrapper
from logging import StreamHandler
from pathlib import Path

import zmq
from IPython.core.application import (  # type:ignore[attr-defined]
    BaseIPythonApplication,
    base_aliases,
    base_flags,
    catch_config_error,
)
from IPython.core.profiledir import ProfileDir
from IPython.core.shellapp import InteractiveShellApp, shell_aliases, shell_flags
from jupyter_client.connect import ConnectionFileMixin
from jupyter_client.session import Session, session_aliases, session_flags
from jupyter_core.paths import jupyter_runtime_dir
from tornado import ioloop
from traitlets.traitlets import (
    Any,
    Bool,
    Dict,
    DottedObjectName,
    Instance,
    Integer,
    Type,
    Unicode,
    default,
)
from traitlets.utils import filefind
from traitlets.utils.importstring import import_item
from zmq.eventloop.zmqstream import ZMQStream

from .connect import get_connection_info, write_connection_file

# local imports
from .control import ControlThread
from .heartbeat import Heartbeat
from .iostream import IOPubThread
from .ipkernel import IPythonKernel
from .parentpoller import ParentPollerUnix, ParentPollerWindows
from .zmqshell import ZMQInteractiveShell

# -----------------------------------------------------------------------------
# Flags and Aliases
# -----------------------------------------------------------------------------

kernel_aliases = dict(base_aliases)
kernel_aliases.update(
    {
        "ip": "IPKernelApp.ip",
        "hb": "IPKernelApp.hb_port",
        "shell": "IPKernelApp.shell_port",
        "iopub": "IPKernelApp.iopub_port",
        "stdin": "IPKernelApp.stdin_port",
        "control": "IPKernelApp.control_port",
        "f": "IPKernelApp.connection_file",
        "transport": "IPKernelApp.transport",
    }
)

kernel_flags = dict(base_flags)
kernel_flags.update(
    {
        "no-stdout": ({"IPKernelApp": {"no_stdout": True}}, "redirect stdout to the null device"),
        "no-stderr": ({"IPKernelApp": {"no_stderr": True}}, "redirect stderr to the null device"),
        "pylab": (
            {"IPKernelApp": {"pylab": "auto"}},
            """Pre-load matplotlib and numpy for interactive use with
        the default matplotlib backend.""",
        ),
        "trio-loop": (
            {"InteractiveShell": {"trio_loop": False}},
            "Enable Trio as main event loop.",
        ),
    }
)

# inherit flags&aliases for any IPython shell apps
kernel_aliases.update(shell_aliases)
kernel_flags.update(shell_flags)

# inherit flags&aliases for Sessions
kernel_aliases.update(session_aliases)
kernel_flags.update(session_flags)

_ctrl_c_message = """\
NOTE: When using the `ipython kernel` entry point, Ctrl-C will not work.

To exit, you will have to explicitly quit this process, by either sending
"quit" from a client, or using Ctrl-\\ in UNIX-like environments.

To read more about this, see https://github.com/ipython/ipython/issues/2049

"""

# -----------------------------------------------------------------------------
# Application class for starting an IPython Kernel
# -----------------------------------------------------------------------------


class IPKernelApp(BaseIPythonApplication, InteractiveShellApp, ConnectionFileMixin):
    """The IPYKernel application class."""

    name = "ipython-kernel"
    aliases = Dict(kernel_aliases)  # type:ignore[assignment]
    flags = Dict(kernel_flags)  # type:ignore[assignment]
    classes = [IPythonKernel, ZMQInteractiveShell, ProfileDir, Session]
    # the kernel class, as an importstring
    kernel_class = Type(
        "ipykernel.ipkernel.IPythonKernel",
        klass="ipykernel.kernelbase.Kernel",
        help="""The Kernel subclass to be used.

    This should allow easy reuse of the IPKernelApp entry point
    to configure and launch kernels other than IPython's own.
    """,
    ).tag(config=True)
    kernel = Any()
    poller = Any()  # don't restrict this even though current pollers are all Threads
    heartbeat = Instance(Heartbeat, allow_none=True)

    context: zmq.Context[t.Any] | None = Any()  # type:ignore[assignment]
    shell_socket = Any()
    control_socket = Any()
    debugpy_socket = Any()
    debug_shell_socket = Any()
    stdin_socket = Any()
    iopub_socket = Any()
    iopub_thread = Any()
    control_thread = Any()

    _ports = Dict()

    subcommands = {
        "install": (
            "ipykernel.kernelspec.InstallIPythonKernelSpecApp",
            "Install the IPython kernel",
        ),
    }

    # connection info:
    connection_dir = Unicode()

    @default("connection_dir")
    def _default_connection_dir(self):
        return jupyter_runtime_dir()

    @property
    def abs_connection_file(self):
        if Path(self.connection_file).name == self.connection_file and self.connection_dir:
            return str(Path(str(self.connection_dir)) / self.connection_file)
        return self.connection_file

    # streams, etc.
    no_stdout = Bool(False, help="redirect stdout to the null device").tag(config=True)
    no_stderr = Bool(False, help="redirect stderr to the null device").tag(config=True)
    trio_loop = Bool(False, help="Set main event loop.").tag(config=True)
    quiet = Bool(True, help="Only send stdout/stderr to output stream").tag(config=True)
    outstream_class = DottedObjectName(
        "ipykernel.iostream.OutStream",
        help="The importstring for the OutStream factory",
        allow_none=True,
    ).tag(config=True)
    displayhook_class = DottedObjectName(
        "ipykernel.displayhook.ZMQDisplayHook", help="The importstring for the DisplayHook factory"
    ).tag(config=True)

    capture_fd_output = Bool(
        True,
        help="""Attempt to capture and forward low-level output, e.g. produced by Extension libraries.
    """,
    ).tag(config=True)

    # polling
    parent_handle = Integer(
        int(os.environ.get("JPY_PARENT_PID") or 0),
        help="""kill this process if its parent dies.  On Windows, the argument
        specifies the HANDLE of the parent process, otherwise it is simply boolean.
        """,
    ).tag(config=True)
    interrupt = Integer(
        int(os.environ.get("JPY_INTERRUPT_EVENT") or 0),
        help="""ONLY USED ON WINDOWS
        Interrupt this process when the parent is signaled.
        """,
    ).tag(config=True)

    def init_crash_handler(self):
        """Initialize the crash handler."""
        sys.excepthook = self.excepthook

    def excepthook(self, etype, evalue, tb):
        """Handle an exception."""
        # write uncaught traceback to 'real' stderr, not zmq-forwarder
        traceback.print_exception(etype, evalue, tb, file=sys.__stderr__)

    def init_poller(self):
        """Initialize the poller."""
        if sys.platform == "win32":
            if self.interrupt or self.parent_handle:
                self.poller = ParentPollerWindows(self.interrupt, self.parent_handle)
        elif self.parent_handle and self.parent_handle != 1:
            # PID 1 (init) is special and will never go away,
            # only be reassigned.
            # Parent polling doesn't work if ppid == 1 to start with.
            self.poller = ParentPollerUnix()

    def _try_bind_socket(self, s, port):
        iface = f"{self.transport}://{self.ip}"
        if self.transport == "tcp":
            if port <= 0:
                port = s.bind_to_random_port(iface)
            else:
                s.bind("tcp://%s:%i" % (self.ip, port))
        elif self.transport == "ipc":
            if port <= 0:
                port = 1
                path = "%s-%i" % (self.ip, port)
                while Path(path).exists():
                    port = port + 1
                    path = "%s-%i" % (self.ip, port)
            else:
                path = "%s-%i" % (self.ip, port)
            s.bind("ipc://%s" % path)
        return port

    def _bind_socket(self, s, port):
        try:
            win_in_use = errno.WSAEADDRINUSE  # type:ignore[attr-defined]
        except AttributeError:
            win_in_use = None

        # Try up to 100 times to bind a port when in conflict to avoid
        # infinite attempts in bad setups
        max_attempts = 1 if port else 100
        for attempt in range(max_attempts):
            try:
                return self._try_bind_socket(s, port)
            except zmq.ZMQError as ze:
                # Raise if we have any error not related to socket binding
                if ze.errno != errno.EADDRINUSE and ze.errno != win_in_use:
                    raise
                if attempt == max_attempts - 1:
                    raise
        return None

    def write_connection_file(self):
        """write connection info to JSON file"""
        cf = self.abs_connection_file
        connection_info = dict(
            ip=self.ip,
            key=self.session.key,
            transport=self.transport,
            shell_port=self.shell_port,
            stdin_port=self.stdin_port,
            hb_port=self.hb_port,
            iopub_port=self.iopub_port,
            control_port=self.control_port,
        )
        if Path(cf).exists():
            # If the file exists, merge our info into it. For example, if the
            # original file had port number 0, we update with the actual port
            # used.
            existing_connection_info = get_connection_info(cf, unpack=True)
            assert isinstance(existing_connection_info, dict)
            connection_info = dict(existing_connection_info, **connection_info)
            if connection_info == existing_connection_info:
                self.log.debug("Connection file %s with current information already exists", cf)
                return

        self.log.debug("Writing connection file: %s", cf)

        write_connection_file(cf, **connection_info)

    def cleanup_connection_file(self):
        """Clean up our connection file."""
        cf = self.abs_connection_file
        self.log.debug("Cleaning up connection file: %s", cf)
        try:
            Path(cf).unlink()
        except OSError:
            pass

        self.cleanup_ipc_files()

    def init_connection_file(self):
        """Initialize our connection file."""
        if not self.connection_file:
            self.connection_file = "kernel-%s.json" % os.getpid()
        try:
            self.connection_file = filefind(self.connection_file, [".", self.connection_dir])
        except OSError:
            self.log.debug("Connection file not found: %s", self.connection_file)
            # This means I own it, and I'll create it in this directory:
            Path(self.abs_connection_file).parent.mkdir(mode=0o700, exist_ok=True, parents=True)
            # Also, I will clean it up:
            atexit.register(self.cleanup_connection_file)
            return
        try:
            self.load_connection_file()
        except Exception:
            self.log.error(  # noqa: G201
                "Failed to load connection file: %r", self.connection_file, exc_info=True
            )
            self.exit(1)

    def init_sockets(self):
        """Create a context, a session, and the kernel sockets."""
        self.log.info("Starting the kernel at pid: %i", os.getpid())
        assert self.context is None, "init_sockets cannot be called twice!"
        self.context = context = zmq.Context()
        atexit.register(self.close)

        self.shell_socket = context.socket(zmq.ROUTER)
        self.shell_socket.linger = 1000
        self.shell_port = self._bind_socket(self.shell_socket, self.shell_port)
        self.log.debug("shell ROUTER Channel on port: %i" % self.shell_port)

        self.stdin_socket = context.socket(zmq.ROUTER)
        self.stdin_socket.linger = 1000
        self.stdin_port = self._bind_socket(self.stdin_socket, self.stdin_port)
        self.log.debug("stdin ROUTER Channel on port: %i" % self.stdin_port)

        if hasattr(zmq, "ROUTER_HANDOVER"):
            # set router-handover to workaround zeromq reconnect problems
            # in certain rare circumstances
            # see ipython/ipykernel#270 and zeromq/libzmq#2892
            self.shell_socket.router_handover = self.stdin_socket.router_handover = 1

        self.init_control(context)
        self.init_iopub(context)

    def init_control(self, context):
        """Initialize the control channel."""
        self.control_socket = context.socket(zmq.ROUTER)
        self.control_socket.linger = 1000
        self.control_port = self._bind_socket(self.control_socket, self.control_port)
        self.log.debug("control ROUTER Channel on port: %i" % self.control_port)

        self.debugpy_socket = context.socket(zmq.STREAM)
        self.debugpy_socket.linger = 1000

        self.debug_shell_socket = context.socket(zmq.DEALER)
        self.debug_shell_socket.linger = 1000
        if self.shell_socket.getsockopt(zmq.LAST_ENDPOINT):
            self.debug_shell_socket.connect(self.shell_socket.getsockopt(zmq.LAST_ENDPOINT))

        if hasattr(zmq, "ROUTER_HANDOVER"):
            # set router-handover to workaround zeromq reconnect problems
            # in certain rare circumstances
            # see ipython/ipykernel#270 and zeromq/libzmq#2892
            self.control_socket.router_handover = 1

        self.control_thread = ControlThread(daemon=True)

    def init_iopub(self, context):
        """Initialize the iopub channel."""
        self.iopub_socket = context.socket(zmq.PUB)
        self.iopub_socket.linger = 1000
        self.iopub_port = self._bind_socket(self.iopub_socket, self.iopub_port)
        self.log.debug("iopub PUB Channel on port: %i" % self.iopub_port)
        self.configure_tornado_logger()
        self.iopub_thread = IOPubThread(self.iopub_socket, pipe=True)
        self.iopub_thread.start()
        # backward-compat: wrap iopub socket API in background thread
        self.iopub_socket = self.iopub_thread.background_socket

    def init_heartbeat(self):
        """start the heart beating"""
        # heartbeat doesn't share context, because it mustn't be blocked
        # by the GIL, which is accessed by libzmq when freeing zero-copy messages
        hb_ctx = zmq.Context()
        self.heartbeat = Heartbeat(hb_ctx, (self.transport, self.ip, self.hb_port))
        self.hb_port = self.heartbeat.port
        self.log.debug("Heartbeat REP Channel on port: %i" % self.hb_port)
        self.heartbeat.start()

    def close(self):
        """Close zmq sockets in an orderly fashion"""
        # un-capture IO before we start closing channels
        self.reset_io()
        self.log.info("Cleaning up sockets")
        if self.heartbeat:
            self.log.debug("Closing heartbeat channel")
            self.heartbeat.context.term()
        if self.iopub_thread:
            self.log.debug("Closing iopub channel")
            self.iopub_thread.stop()
            self.iopub_thread.close()
        if self.control_thread and self.control_thread.is_alive():
            self.log.debug("Closing control thread")
            self.control_thread.stop()
            self.control_thread.join()

        if self.debugpy_socket and not self.debugpy_socket.closed:
            self.debugpy_socket.close()
        if self.debug_shell_socket and not self.debug_shell_socket.closed:
            self.debug_shell_socket.close()

        for channel in ("shell", "control", "stdin"):
            self.log.debug("Closing %s channel", channel)
            socket = getattr(self, channel + "_socket", None)
            if socket and not socket.closed:
                socket.close()
        self.log.debug("Terminating zmq context")
        if self.context:
            self.context.term()
        self.log.debug("Terminated zmq context")

    def log_connection_info(self):
        """display connection info, and store ports"""
        basename = Path(self.connection_file).name
        if (
            basename == self.connection_file
            or str(Path(self.connection_file).parent) == self.connection_dir
        ):
            # use shortname
            tail = basename
        else:
            tail = self.connection_file
        lines = [
            "To connect another client to this kernel, use:",
            "    --existing %s" % tail,
        ]
        # log connection info
        # info-level, so often not shown.
        # frontends should use the %connect_info magic
        # to see the connection info
        for line in lines:
            self.log.info(line)
        # also raw print to the terminal if no parent_handle (`ipython kernel`)
        # unless log-level is CRITICAL (--quiet)
        if not self.parent_handle and int(self.log_level) < logging.CRITICAL:  # type:ignore[call-overload]
            print(_ctrl_c_message, file=sys.__stdout__)
            for line in lines:
                print(line, file=sys.__stdout__)

        self._ports = dict(
            shell=self.shell_port,
            iopub=self.iopub_port,
            stdin=self.stdin_port,
            hb=self.hb_port,
            control=self.control_port,
        )

    def init_blackhole(self):
        """redirects stdout/stderr to devnull if necessary"""
        if self.no_stdout or self.no_stderr:
            blackhole = open(os.devnull, "w")  # noqa: SIM115
            if self.no_stdout:
                sys.stdout = sys.__stdout__ = blackhole  # type:ignore[misc]
            if self.no_stderr:
                sys.stderr = sys.__stderr__ = blackhole  # type:ignore[misc]

    def init_io(self):
        """Redirect input streams and set a display hook."""
        if self.outstream_class:
            outstream_factory = import_item(str(self.outstream_class))
            if sys.stdout is not None:
                sys.stdout.flush()

            e_stdout = None if self.quiet else sys.__stdout__
            e_stderr = None if self.quiet else sys.__stderr__

            if not self.capture_fd_output:
                outstream_factory = partial(outstream_factory, watchfd=False)

            sys.stdout = outstream_factory(self.session, self.iopub_thread, "stdout", echo=e_stdout)
            if sys.stderr is not None:
                sys.stderr.flush()
            sys.stderr = outstream_factory(self.session, self.iopub_thread, "stderr", echo=e_stderr)
            if hasattr(sys.stderr, "_original_stdstream_copy"):
                for handler in self.log.handlers:
                    if isinstance(handler, StreamHandler) and (handler.stream.buffer.fileno() == 2):
                        self.log.debug("Seeing logger to stderr, rerouting to raw filedescriptor.")

                        handler.stream = TextIOWrapper(
                            FileIO(
                                sys.stderr._original_stdstream_copy,
                                "w",
                            )
                        )
        if self.displayhook_class:
            displayhook_factory = import_item(str(self.displayhook_class))
            self.displayhook = displayhook_factory(self.session, self.iopub_socket)
            sys.displayhook = self.displayhook

        self.patch_io()

    def reset_io(self):
        """restore original io

        restores state after init_io
        """
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        sys.displayhook = sys.__displayhook__

    def patch_io(self):
        """Patch important libraries that can't handle sys.stdout forwarding"""
        try:
            import faulthandler
        except ImportError:
            pass
        else:
            # Warning: this is a monkeypatch of `faulthandler.enable`, watch for possible
            # updates to the upstream API and update accordingly (up-to-date as of Python 3.5):
            # https://docs.python.org/3/library/faulthandler.html#faulthandler.enable

            # change default file to __stderr__ from forwarded stderr
            faulthandler_enable = faulthandler.enable

            def enable(file=sys.__stderr__, all_threads=True, **kwargs):
                return faulthandler_enable(file=file, all_threads=all_threads, **kwargs)

            faulthandler.enable = enable

            if hasattr(faulthandler, "register"):
                faulthandler_register = faulthandler.register

                def register(signum, file=sys.__stderr__, all_threads=True, chain=False, **kwargs):
                    return faulthandler_register(
                        signum, file=file, all_threads=all_threads, chain=chain, **kwargs
                    )

                faulthandler.register = register

    def init_signal(self):
        """Initialize the signal handler."""
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    def init_kernel(self):
        """Create the Kernel object itself"""
        shell_stream = ZMQStream(self.shell_socket)
        control_stream = ZMQStream(self.control_socket, self.control_thread.io_loop)
        debugpy_stream = ZMQStream(self.debugpy_socket, self.control_thread.io_loop)
        self.control_thread.start()
        kernel_factory = self.kernel_class.instance  # type:ignore[attr-defined]

        kernel = kernel_factory(
            parent=self,
            session=self.session,
            control_stream=control_stream,
            debugpy_stream=debugpy_stream,
            debug_shell_socket=self.debug_shell_socket,
            shell_stream=shell_stream,
            control_thread=self.control_thread,
            iopub_thread=self.iopub_thread,
            iopub_socket=self.iopub_socket,
            stdin_socket=self.stdin_socket,
            log=self.log,
            profile_dir=self.profile_dir,
            user_ns=self.user_ns,
        )
        kernel.record_ports({name + "_port": port for name, port in self._ports.items()})
        self.kernel = kernel

        # Allow the displayhook to get the execution count
        self.displayhook.get_execution_count = lambda: kernel.execution_count

    def init_gui_pylab(self):
        """Enable GUI event loop integration, taking pylab into account."""

        # Register inline backend as default
        # this is higher priority than matplotlibrc,
        # but lower priority than anything else (mpl.use() for instance).
        # This only affects matplotlib >= 1.5
        if not os.environ.get("MPLBACKEND"):
            os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"

        # Provide a wrapper for :meth:`InteractiveShellApp.init_gui_pylab`
        # to ensure that any exception is printed straight to stderr.
        # Normally _showtraceback associates the reply with an execution,
        # which means frontends will never draw it, as this exception
        # is not associated with any execute request.

        shell = self.shell
        assert shell is not None
        _showtraceback = shell._showtraceback
        try:
            # replace error-sending traceback with stderr
            def print_tb(etype, evalue, stb):
                print("GUI event loop or pylab initialization failed", file=sys.stderr)
                assert shell is not None
                print(shell.InteractiveTB.stb2text(stb), file=sys.stderr)

            shell._showtraceback = print_tb
            InteractiveShellApp.init_gui_pylab(self)
        finally:
            shell._showtraceback = _showtraceback

    def init_shell(self):
        """Initialize the shell channel."""
        self.shell = getattr(self.kernel, "shell", None)
        if self.shell:
            self.shell.configurables.append(self)

    def configure_tornado_logger(self):
        """Configure the tornado logging.Logger.

        Must set up the tornado logger or else tornado will call
        basicConfig for the root logger which makes the root logger
        go to the real sys.stderr instead of the capture streams.
        This function mimics the setup of logging.basicConfig.
        """
        logger = logging.getLogger("tornado")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(logging.BASIC_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    def _init_asyncio_patch(self):
        """set default asyncio policy to be compatible with tornado

        Tornado 6 (at least) is not compatible with the default
        asyncio implementation on Windows

        Pick the older SelectorEventLoopPolicy on Windows
        if the known-incompatible default policy is in use.

        Support for Proactor via a background thread is available in tornado 6.1,
        but it is still preferable to run the Selector in the main thread
        instead of the background.

        do this as early as possible to make it a low priority and overridable

        ref: https://github.com/tornadoweb/tornado/issues/2608

        FIXME: if/when tornado supports the defaults in asyncio without threads,
               remove and bump tornado requirement for py38.
               Most likely, this will mean a new Python version
               where asyncio.ProactorEventLoop supports add_reader and friends.

        """
        if sys.platform.startswith("win") and sys.version_info >= (3, 8):
            import asyncio

            try:
                from asyncio import WindowsProactorEventLoopPolicy, WindowsSelectorEventLoopPolicy
            except ImportError:
                pass
                # not affected
            else:
                if type(asyncio.get_event_loop_policy()) is WindowsProactorEventLoopPolicy:
                    # WindowsProactorEventLoopPolicy is not compatible with tornado 6
                    # fallback to the pre-3.8 default of Selector
                    asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())

    def init_pdb(self):
        """Replace pdb with IPython's version that is interruptible.

        With the non-interruptible version, stopping pdb() locks up the kernel in a
        non-recoverable state.
        """
        import pdb

        from IPython.core import debugger

        if hasattr(debugger, "InterruptiblePdb"):
            # Only available in newer IPython releases:
            debugger.Pdb = debugger.InterruptiblePdb  # type:ignore[misc]
            pdb.Pdb = debugger.Pdb  # type:ignore[assignment,misc]
            pdb.set_trace = debugger.set_trace  # type:ignore[assignment]

    @catch_config_error
    def initialize(self, argv=None):
        """Initialize the application."""
        self._init_asyncio_patch()
        super().initialize(argv)
        if self.subapp is not None:
            return

        self.init_pdb()
        self.init_blackhole()
        self.init_connection_file()
        self.init_poller()
        self.init_sockets()
        self.init_heartbeat()
        # writing/displaying connection info must be *after* init_sockets/heartbeat
        self.write_connection_file()
        # Log connection info after writing connection file, so that the connection
        # file is definitely available at the time someone reads the log.
        self.log_connection_info()
        self.init_io()
        try:
            self.init_signal()
        except Exception:
            # Catch exception when initializing signal fails, eg when running the
            # kernel on a separate thread
            if int(self.log_level) < logging.CRITICAL:  # type:ignore[call-overload]
                self.log.error("Unable to initialize signal:", exc_info=True)  # noqa: G201
        self.init_kernel()
        # shell init steps
        self.init_path()
        self.init_shell()
        if self.shell:
            self.init_gui_pylab()
            self.init_extensions()
            self.init_code()
        # flush stdout/stderr, so that anything written to these streams during
        # initialization do not get associated with the first execution request
        sys.stdout.flush()
        sys.stderr.flush()

    def start(self):
        """Start the application."""
        if self.subapp is not None:
            return self.subapp.start()
        if self.poller is not None:
            self.poller.start()
        self.kernel.start()
        self.io_loop = ioloop.IOLoop.current()
        if self.trio_loop:
            from ipykernel.trio_runner import TrioRunner

            tr = TrioRunner()
            tr.initialize(self.kernel, self.io_loop)
            try:
                tr.run()
            except KeyboardInterrupt:
                pass
        else:
            try:
                self.io_loop.start()
            except KeyboardInterrupt:
                pass


launch_new_instance = IPKernelApp.launch_instance


def main():  # pragma: no cover
    """Run an IPKernel as an application"""
    app = IPKernelApp.instance()
    app.initialize()
    app.start()


if __name__ == "__main__":
    main()
