"""A ZMQ-based subclass of InteractiveShell.

This code is meant to ease the refactoring of the base InteractiveShell into
something with a cleaner architecture for 2-process use, without actually
breaking InteractiveShell itself.  So we're doing something a bit ugly, where
we subclass and override what we want to fix.  Once this is working well, we
can go back to the base class and refactor the code for a cleaner inheritance
implementation that doesn't rely on so much monkeypatching.

But this lets us maintain a fully working IPython as we develop the new
machinery.  This should thus be thought of as scaffolding.
"""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

import os
import sys
import warnings
from threading import local

from IPython.core import page, payloadpage
from IPython.core.autocall import ZMQExitAutocall
from IPython.core.displaypub import DisplayPublisher
from IPython.core.error import UsageError
from IPython.core.interactiveshell import InteractiveShell, InteractiveShellABC
from IPython.core.magic import Magics, line_magic, magics_class
from IPython.core.magics import CodeMagics, MacroToEdit  # type:ignore[attr-defined]
from IPython.core.usage import default_banner
from IPython.display import Javascript, display  # type:ignore[attr-defined]
from IPython.utils import openpy
from IPython.utils.process import arg_split, system  # type:ignore[attr-defined]
from jupyter_client.session import Session, extract_header
from jupyter_core.paths import jupyter_runtime_dir
from traitlets import Any, CBool, CBytes, Dict, Instance, Type, default, observe

from ipykernel import connect_qtconsole, get_connection_file, get_connection_info
from ipykernel.displayhook import ZMQShellDisplayHook
from ipykernel.jsonutil import encode_images, json_clean

# -----------------------------------------------------------------------------
# Functions and classes
# -----------------------------------------------------------------------------


class ZMQDisplayPublisher(DisplayPublisher):
    """A display publisher that publishes data using a ZeroMQ PUB socket."""

    session = Instance(Session, allow_none=True)
    pub_socket = Any(allow_none=True)
    parent_header = Dict({})
    topic = CBytes(b"display_data")

    # thread_local:
    # An attribute used to ensure the correct output message
    # is processed. See ipykernel Issue 113 for a discussion.
    _thread_local = Any()

    def set_parent(self, parent):
        """Set the parent for outbound messages."""
        self.parent_header = extract_header(parent)

    def _flush_streams(self):
        """flush IO Streams prior to display"""
        sys.stdout.flush()
        sys.stderr.flush()

    @default("_thread_local")
    def _default_thread_local(self):
        """Initialize our thread local storage"""
        return local()

    @property
    def _hooks(self):
        if not hasattr(self._thread_local, "hooks"):
            # create new list for a new thread
            self._thread_local.hooks = []
        return self._thread_local.hooks

    def publish(
        self,
        data,
        metadata=None,
        transient=None,
        update=False,
    ):
        """Publish a display-data message

        Parameters
        ----------
        data : dict
            A mime-bundle dict, keyed by mime-type.
        metadata : dict, optional
            Metadata associated with the data.
        transient : dict, optional, keyword-only
            Transient data that may only be relevant during a live display,
            such as display_id.
            Transient data should not be persisted to documents.
        update : bool, optional, keyword-only
            If True, send an update_display_data message instead of display_data.
        """
        self._flush_streams()
        if metadata is None:
            metadata = {}
        if transient is None:
            transient = {}
        self._validate_data(data, metadata)
        content = {}
        content["data"] = encode_images(data)
        content["metadata"] = metadata
        content["transient"] = transient

        msg_type = "update_display_data" if update else "display_data"

        # Use 2-stage process to send a message,
        # in order to put it through the transform
        # hooks before potentially sending.
        msg = self.session.msg(msg_type, json_clean(content), parent=self.parent_header)

        # Each transform either returns a new
        # message or None. If None is returned,
        # the message has been 'used' and we return.
        for hook in self._hooks:
            msg = hook(msg)
            if msg is None:
                return

        self.session.send(
            self.pub_socket,
            msg,
            ident=self.topic,
        )

    def clear_output(self, wait=False):
        """Clear output associated with the current execution (cell).

        Parameters
        ----------
        wait : bool (default: False)
            If True, the output will not be cleared immediately,
            instead waiting for the next display before clearing.
            This reduces bounce during repeated clear & display loops.

        """
        content = dict(wait=wait)
        self._flush_streams()
        msg = self.session.msg("clear_output", json_clean(content), parent=self.parent_header)

        # see publish() for details on how this works
        for hook in self._hooks:
            msg = hook(msg)
            if msg is None:
                return

        self.session.send(
            self.pub_socket,
            msg,
            ident=self.topic,
        )

    def register_hook(self, hook):
        """
        Registers a hook with the thread-local storage.

        Parameters
        ----------
        hook : Any callable object

        Returns
        -------
        Either a publishable message, or `None`.
        The DisplayHook objects must return a message from
        the __call__ method if they still require the
        `session.send` method to be called after transformation.
        Returning `None` will halt that execution path, and
        session.send will not be called.
        """
        self._hooks.append(hook)

    def unregister_hook(self, hook):
        """
        Un-registers a hook with the thread-local storage.

        Parameters
        ----------
        hook : Any callable object which has previously been
            registered as a hook.

        Returns
        -------
        bool - `True` if the hook was removed, `False` if it wasn't
            found.
        """
        try:
            self._hooks.remove(hook)
            return True
        except ValueError:
            return False


@magics_class
class KernelMagics(Magics):
    """Kernel magics."""

    # ------------------------------------------------------------------------
    # Magic overrides
    # ------------------------------------------------------------------------
    # Once the base class stops inheriting from magic, this code needs to be
    # moved into a separate machinery as well.  For now, at least isolate here
    # the magics which this class needs to implement differently from the base
    # class, or that are unique to it.

    @line_magic
    def edit(self, parameter_s="", last_call=None):
        """Bring up an editor and execute the resulting code.

        Usage:
          %edit [options] [args]

        %edit runs an external text editor. You will need to set the command for
        this editor via the ``TerminalInteractiveShell.editor`` option in your
        configuration file before it will work.

        This command allows you to conveniently edit multi-line code right in
        your IPython session.

        If called without arguments, %edit opens up an empty editor with a
        temporary file and will execute the contents of this file when you
        close it (don't forget to save it!).

        Options:

        -n <number>
          Open the editor at a specified line number. By default, the IPython
          editor hook uses the unix syntax 'editor +N filename', but you can
          configure this by providing your own modified hook if your favorite
          editor supports line-number specifications with a different syntax.

        -p
          Call the editor with the same data as the previous time it was used,
          regardless of how long ago (in your current session) it was.

        -r
          Use 'raw' input. This option only applies to input taken from the
          user's history.  By default, the 'processed' history is used, so that
          magics are loaded in their transformed version to valid Python.  If
          this option is given, the raw input as typed as the command line is
          used instead.  When you exit the editor, it will be executed by
          IPython's own processor.

        Arguments:

        If arguments are given, the following possibilities exist:

        - The arguments are numbers or pairs of colon-separated numbers (like
          1 4:8 9). These are interpreted as lines of previous input to be
          loaded into the editor. The syntax is the same of the %macro command.

        - If the argument doesn't start with a number, it is evaluated as a
          variable and its contents loaded into the editor. You can thus edit
          any string which contains python code (including the result of
          previous edits).

        - If the argument is the name of an object (other than a string),
          IPython will try to locate the file where it was defined and open the
          editor at the point where it is defined. You can use ``%edit function``
          to load an editor exactly at the point where 'function' is defined,
          edit it and have the file be executed automatically.

          If the object is a macro (see %macro for details), this opens up your
          specified editor with a temporary file containing the macro's data.
          Upon exit, the macro is reloaded with the contents of the file.

          Note: opening at an exact line is only supported under Unix, and some
          editors (like kedit and gedit up to Gnome 2.8) do not understand the
          '+NUMBER' parameter necessary for this feature. Good editors like
          (X)Emacs, vi, jed, pico and joe all do.

        - If the argument is not found as a variable, IPython will look for a
          file with that name (adding .py if necessary) and load it into the
          editor. It will execute its contents with execfile() when you exit,
          loading any code in the file into your interactive namespace.

        Unlike in the terminal, this is designed to use a GUI editor, and we do
        not know when it has closed. So the file you edit will not be
        automatically executed or printed.

        Note that %edit is also available through the alias %ed.
        """
        last_call = last_call or ["", ""]
        opts, args = self.parse_options(parameter_s, "prn:")

        try:
            filename, lineno, _ = CodeMagics._find_edit_target(self.shell, args, opts, last_call)
        except MacroToEdit:
            # TODO: Implement macro editing over 2 processes.
            print("Macro editing not yet implemented in 2-process model.")
            return

        # Make sure we send to the client an absolute path, in case the working
        # directory of client and kernel don't match
        filename = os.path.abspath(filename)

        payload = {"source": "edit_magic", "filename": filename, "line_number": lineno}
        assert self.shell is not None
        self.shell.payload_manager.write_payload(payload)

    # A few magics that are adapted to the specifics of using pexpect and a
    # remote terminal

    @line_magic
    def clear(self, arg_s):
        """Clear the terminal."""
        assert self.shell is not None
        if os.name == "posix":
            self.shell.system("clear")
        else:
            self.shell.system("cls")

    if os.name == "nt":
        # This is the usual name in windows
        cls = line_magic("cls")(clear)

    # Terminal pagers won't work over pexpect, but we do have our own pager

    @line_magic
    def less(self, arg_s):
        """Show a file through the pager.

        Files ending in .py are syntax-highlighted."""
        if not arg_s:
            msg = "Missing filename."
            raise UsageError(msg)

        if arg_s.endswith(".py"):
            assert self.shell is not None
            cont = self.shell.pycolorize(openpy.read_py_file(arg_s, skip_encoding_cookie=False))
        else:
            with open(arg_s) as fid:
                cont = fid.read()
        page.page(cont)

    more = line_magic("more")(less)

    # Man calls a pager, so we also need to redefine it
    if os.name == "posix":

        @line_magic
        def man(self, arg_s):
            """Find the man page for the given command and display in pager."""
            assert self.shell is not None
            page.page(self.shell.getoutput("man %s | col -b" % arg_s, split=False))

    @line_magic
    def connect_info(self, arg_s):
        """Print information for connecting other clients to this kernel

        It will print the contents of this session's connection file, as well as
        shortcuts for local clients.

        In the simplest case, when called from the most recently launched kernel,
        secondary clients can be connected, simply with:

        $> jupyter <app> --existing

        """

        try:
            connection_file = get_connection_file()
            info = get_connection_info(unpack=False)
        except Exception as e:
            warnings.warn("Could not get connection info: %r" % e, stacklevel=2)
            return

        # if it's in the default dir, truncate to basename
        if jupyter_runtime_dir() == os.path.dirname(connection_file):
            connection_file = os.path.basename(connection_file)

        print(info + "\n")
        print(
            f"Paste the above JSON into a file, and connect with:\n"
            f"    $> jupyter <app> --existing <file>\n"
            f"or, if you are local, you can connect with just:\n"
            f"    $> jupyter <app> --existing {connection_file}\n"
            f"or even just:\n"
            f"    $> jupyter <app> --existing\n"
            f"if this is the most recent Jupyter kernel you have started."
        )

    @line_magic
    def qtconsole(self, arg_s):
        """Open a qtconsole connected to this kernel.

        Useful for connecting a qtconsole to running notebooks, for better
        debugging.
        """

        # %qtconsole should imply bind_kernel for engines:
        # FIXME: move to ipyparallel Kernel subclass
        if "ipyparallel" in sys.modules:
            from ipyparallel import bind_kernel

            bind_kernel()

        try:
            connect_qtconsole(argv=arg_split(arg_s, os.name == "posix"))
        except Exception as e:
            warnings.warn("Could not start qtconsole: %r" % e, stacklevel=2)
            return

    @line_magic
    def autosave(self, arg_s):
        """Set the autosave interval in the notebook (in seconds).

        The default value is 120, or two minutes.
        ``%autosave 0`` will disable autosave.

        This magic only has an effect when called from the notebook interface.
        It has no effect when called in a startup file.
        """

        try:
            interval = int(arg_s)
        except ValueError as e:
            raise UsageError("%%autosave requires an integer, got %r" % arg_s) from e

        # javascript wants milliseconds
        milliseconds = 1000 * interval
        display(
            Javascript("IPython.notebook.set_autosave_interval(%i)" % milliseconds),
            include=["application/javascript"],
        )
        if interval:
            print("Autosaving every %i seconds" % interval)
        else:
            print("Autosave disabled")


class ZMQInteractiveShell(InteractiveShell):
    """A subclass of InteractiveShell for ZMQ."""

    displayhook_class = Type(ZMQShellDisplayHook)
    display_pub_class = Type(ZMQDisplayPublisher)
    data_pub_class = Any()  # type:ignore[assignment]
    kernel = Any()
    parent_header = Any()

    @default("banner1")
    def _default_banner1(self):
        return default_banner

    # Override the traitlet in the parent class, because there's no point using
    # readline for the kernel. Can be removed when the readline code is moved
    # to the terminal frontend.
    readline_use = CBool(False)
    # autoindent has no meaning in a zmqshell, and attempting to enable it
    # will print a warning in the absence of readline.
    autoindent = CBool(False)

    exiter = Instance(ZMQExitAutocall)

    @default("exiter")
    def _default_exiter(self):
        return ZMQExitAutocall(self)

    @observe("exit_now")
    def _update_exit_now(self, change):
        """stop eventloop when exit_now fires"""
        if change["new"]:
            if hasattr(self.kernel, "io_loop"):
                loop = self.kernel.io_loop
                loop.call_later(0.1, loop.stop)
            if self.kernel.eventloop:
                exit_hook = getattr(self.kernel.eventloop, "exit_hook", None)
                if exit_hook:
                    exit_hook(self.kernel)

    keepkernel_on_exit = None

    # Over ZeroMQ, GUI control isn't done with PyOS_InputHook as there is no
    # interactive input being read; we provide event loop support in ipkernel
    def enable_gui(self, gui):
        """Enable a given guil."""
        from .eventloops import enable_gui as real_enable_gui

        try:
            real_enable_gui(gui)
            self.active_eventloop = gui
        except ValueError as e:
            raise UsageError("%s" % e) from e

    def init_environment(self):
        """Configure the user's environment."""
        env = os.environ
        # These two ensure 'ls' produces nice coloring on BSD-derived systems
        env["TERM"] = "xterm-color"
        env["CLICOLOR"] = "1"
        # These two add terminal color in tools that support it.
        env["FORCE_COLOR"] = "1"
        env["CLICOLOR_FORCE"] = "1"
        # Since normal pagers don't work at all (over pexpect we don't have
        # single-key control of the subprocess), try to disable paging in
        # subprocesses as much as possible.
        env["PAGER"] = "cat"
        env["GIT_PAGER"] = "cat"

    def init_hooks(self):
        """Initialize hooks."""
        super().init_hooks()
        self.set_hook("show_in_pager", page.as_hook(payloadpage.page), 99)

    def init_data_pub(self):
        """Delay datapub init until request, for deprecation warnings"""
        pass

    @property
    def data_pub(self):
        if not hasattr(self, "_data_pub"):
            warnings.warn(
                "InteractiveShell.data_pub is deprecated outside IPython parallel.",
                DeprecationWarning,
                stacklevel=2,
            )

            self._data_pub = self.data_pub_class(parent=self)  # type:ignore[has-type]
            self._data_pub.session = self.display_pub.session
            self._data_pub.pub_socket = self.display_pub.pub_socket
        return self._data_pub

    @data_pub.setter
    def data_pub(self, pub):
        self._data_pub = pub

    def ask_exit(self):
        """Engage the exit actions."""
        self.exit_now = not self.keepkernel_on_exit
        payload = dict(
            source="ask_exit",
            keepkernel=self.keepkernel_on_exit,
        )
        self.payload_manager.write_payload(payload)

    def run_cell(self, *args, **kwargs):
        """Run a cell."""
        self._last_traceback = None
        return super().run_cell(*args, **kwargs)

    def _showtraceback(self, etype, evalue, stb):
        # try to preserve ordering of tracebacks and print statements
        sys.stdout.flush()
        sys.stderr.flush()

        exc_content = {
            "traceback": stb,
            "ename": str(etype.__name__),
            "evalue": str(evalue),
        }

        dh = self.displayhook
        # Send exception info over pub socket for other clients than the caller
        # to pick up
        topic = None
        if dh.topic:
            topic = dh.topic.replace(b"execute_result", b"error")

        dh.session.send(
            dh.pub_socket,
            "error",
            json_clean(exc_content),
            dh.parent_header,
            ident=topic,
        )

        # FIXME - Once we rely on Python 3, the traceback is stored on the
        # exception object, so we shouldn't need to store it here.
        self._last_traceback = stb

    def set_next_input(self, text, replace=False):
        """Send the specified text to the frontend to be presented at the next
        input cell."""
        payload = dict(
            source="set_next_input",
            text=text,
            replace=replace,
        )
        self.payload_manager.write_payload(payload)

    def set_parent(self, parent):
        """Set the parent header for associating output with its triggering input"""
        self.parent_header = parent
        self.displayhook.set_parent(parent)
        self.display_pub.set_parent(parent)
        if hasattr(self, "_data_pub"):
            self.data_pub.set_parent(parent)
        try:
            sys.stdout.set_parent(parent)  # type:ignore[attr-defined]
        except AttributeError:
            pass
        try:
            sys.stderr.set_parent(parent)  # type:ignore[attr-defined]
        except AttributeError:
            pass

    def get_parent(self):
        """Get the parent header."""
        return self.parent_header

    def init_magics(self):
        """Initialize magics."""
        super().init_magics()
        self.register_magics(KernelMagics)
        self.magics_manager.register_alias("ed", "edit")

    def init_virtualenv(self):
        """Initialize virtual environment."""
        # Overridden not to do virtualenv detection, because it's probably
        # not appropriate in a kernel. To use a kernel in a virtualenv, install
        # it inside the virtualenv.
        # https://ipython.readthedocs.io/en/latest/install/kernel_install.html
        pass

    def system_piped(self, cmd):
        """Call the given cmd in a subprocess, piping stdout/err

        Parameters
        ----------
        cmd : str
            Command to execute (can not end in '&', as background processes are
            not supported.  Should not be a command that expects input
            other than simple text.
        """
        if cmd.rstrip().endswith("&"):
            # this is *far* from a rigorous test
            # We do not support backgrounding processes because we either use
            # pexpect or pipes to read from.  Users can always just call
            # os.system() or use ip.system=ip.system_raw
            # if they really want a background process.
            msg = "Background processes not supported."
            raise OSError(msg)

        # we explicitly do NOT return the subprocess status code, because
        # a non-None value would trigger :func:`sys.displayhook` calls.
        # Instead, we store the exit_code in user_ns.
        # Also, protect system call from UNC paths on Windows here too
        # as is done in InteractiveShell.system_raw
        if sys.platform == "win32":
            cmd = self.var_expand(cmd, depth=1)
            from IPython.utils._process_win32 import AvoidUNCPath

            with AvoidUNCPath() as path:
                if path is not None:
                    cmd = f"pushd {path} &&{cmd}"
                self.user_ns["_exit_code"] = system(cmd)
        else:
            self.user_ns["_exit_code"] = system(self.var_expand(cmd, depth=1))

    # Ensure new system_piped implementation is used
    system = system_piped


InteractiveShellABC.register(ZMQInteractiveShell)
