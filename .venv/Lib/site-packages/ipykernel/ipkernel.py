"""The IPython kernel implementation"""

import asyncio
import builtins
import getpass
import os
import signal
import sys
import threading
import typing as t
from contextlib import contextmanager
from functools import partial

import comm
from IPython.core import release
from IPython.utils.tokenutil import line_at_cursor, token_at_cursor
from traitlets import Any, Bool, HasTraits, Instance, List, Type, observe, observe_compat
from zmq.eventloop.zmqstream import ZMQStream

from .comm.comm import BaseComm
from .comm.manager import CommManager
from .compiler import XCachingCompiler
from .debugger import Debugger, _is_debugpy_available
from .eventloops import _use_appnope
from .kernelbase import Kernel as KernelBase
from .kernelbase import _accepts_cell_id
from .zmqshell import ZMQInteractiveShell

try:
    from IPython.core.interactiveshell import _asyncio_runner  # type:ignore[attr-defined]
except ImportError:
    _asyncio_runner = None  # type:ignore

try:
    from IPython.core.completer import provisionalcompleter as _provisionalcompleter
    from IPython.core.completer import rectify_completions as _rectify_completions

    _use_experimental_60_completion = True
except ImportError:
    _use_experimental_60_completion = False


_EXPERIMENTAL_KEY_NAME = "_jupyter_types_experimental"


def _create_comm(*args, **kwargs):
    """Create a new Comm."""
    return BaseComm(*args, **kwargs)


# there can only be one comm manager in a ipykernel process
_comm_lock = threading.Lock()
_comm_manager: t.Optional[CommManager] = None


def _get_comm_manager(*args, **kwargs):
    """Create a new CommManager."""
    global _comm_manager  # noqa
    if _comm_manager is None:
        with _comm_lock:
            if _comm_manager is None:
                _comm_manager = CommManager(*args, **kwargs)
    return _comm_manager


comm.create_comm = _create_comm
comm.get_comm_manager = _get_comm_manager


class IPythonKernel(KernelBase):
    """The IPython Kernel class."""

    shell = Instance("IPython.core.interactiveshell.InteractiveShellABC", allow_none=True)
    shell_class = Type(ZMQInteractiveShell)

    use_experimental_completions = Bool(
        True,
        help="Set this flag to False to deactivate the use of experimental IPython completion APIs.",
    ).tag(config=True)

    debugpy_stream = Instance(ZMQStream, allow_none=True) if _is_debugpy_available else None

    user_module = Any()

    @observe("user_module")
    @observe_compat
    def _user_module_changed(self, change):
        if self.shell is not None:
            self.shell.user_module = change["new"]

    user_ns = Instance(dict, args=None, allow_none=True)

    @observe("user_ns")
    @observe_compat
    def _user_ns_changed(self, change):
        if self.shell is not None:
            self.shell.user_ns = change["new"]
            self.shell.init_user_ns()

    # A reference to the Python builtin 'raw_input' function.
    # (i.e., __builtin__.raw_input for Python 2.7, builtins.input for Python 3)
    _sys_raw_input = Any()
    _sys_eval_input = Any()

    def __init__(self, **kwargs):
        """Initialize the kernel."""
        super().__init__(**kwargs)

        # Initialize the Debugger
        if _is_debugpy_available:
            self.debugger = Debugger(
                self.log,
                self.debugpy_stream,
                self._publish_debug_event,
                self.debug_shell_socket,
                self.session,
                self.debug_just_my_code,
            )

        # Initialize the InteractiveShell subclass
        self.shell = self.shell_class.instance(
            parent=self,
            profile_dir=self.profile_dir,
            user_module=self.user_module,
            user_ns=self.user_ns,
            kernel=self,
            compiler_class=XCachingCompiler,
        )
        self.shell.displayhook.session = self.session

        jupyter_session_name = os.environ.get('JPY_SESSION_NAME')
        if jupyter_session_name:
            self.shell.user_ns['__session__'] = jupyter_session_name

        self.shell.displayhook.pub_socket = self.iopub_socket
        self.shell.displayhook.topic = self._topic("execute_result")
        self.shell.display_pub.session = self.session
        self.shell.display_pub.pub_socket = self.iopub_socket

        self.comm_manager = comm.get_comm_manager()

        assert isinstance(self.comm_manager, HasTraits)
        self.shell.configurables.append(self.comm_manager)
        comm_msg_types = ["comm_open", "comm_msg", "comm_close"]
        for msg_type in comm_msg_types:
            self.shell_handlers[msg_type] = getattr(self.comm_manager, msg_type)

        if _use_appnope() and self._darwin_app_nap:
            # Disable app-nap as the kernel is not a gui but can have guis
            import appnope

            appnope.nope()

    help_links = List(
        [
            {
                "text": "Python Reference",
                "url": "https://docs.python.org/%i.%i" % sys.version_info[:2],
            },
            {
                "text": "IPython Reference",
                "url": "https://ipython.org/documentation.html",
            },
            {
                "text": "NumPy Reference",
                "url": "https://docs.scipy.org/doc/numpy/reference/",
            },
            {
                "text": "SciPy Reference",
                "url": "https://docs.scipy.org/doc/scipy/reference/",
            },
            {
                "text": "Matplotlib Reference",
                "url": "https://matplotlib.org/contents.html",
            },
            {
                "text": "SymPy Reference",
                "url": "http://docs.sympy.org/latest/index.html",
            },
            {
                "text": "pandas Reference",
                "url": "https://pandas.pydata.org/pandas-docs/stable/",
            },
        ]
    ).tag(config=True)

    # Kernel info fields
    implementation = "ipython"
    implementation_version = release.version
    language_info = {
        "name": "python",
        "version": sys.version.split()[0],
        "mimetype": "text/x-python",
        "codemirror_mode": {"name": "ipython", "version": sys.version_info[0]},
        "pygments_lexer": "ipython%d" % 3,
        "nbconvert_exporter": "python",
        "file_extension": ".py",
    }

    def dispatch_debugpy(self, msg):
        if _is_debugpy_available:
            # The first frame is the socket id, we can drop it
            frame = msg[1].bytes.decode("utf-8")
            self.log.debug("Debugpy received: %s", frame)
            self.debugger.tcp_client.receive_dap_frame(frame)

    @property
    def banner(self):
        return self.shell.banner

    async def poll_stopped_queue(self):
        """Poll the stopped queue."""
        while True:
            await self.debugger.handle_stopped_event()

    def start(self):
        """Start the kernel."""
        self.shell.exit_now = False
        if self.debugpy_stream is None:
            self.log.warning("debugpy_stream undefined, debugging will not be enabled")
        else:
            self.debugpy_stream.on_recv(self.dispatch_debugpy, copy=False)
        super().start()
        if self.debugpy_stream:
            asyncio.run_coroutine_threadsafe(
                self.poll_stopped_queue(), self.control_thread.io_loop.asyncio_loop
            )

    def set_parent(self, ident, parent, channel="shell"):
        """Overridden from parent to tell the display hook and output streams
        about the parent message.
        """
        super().set_parent(ident, parent, channel)
        if channel == "shell":
            self.shell.set_parent(parent)

    def init_metadata(self, parent):
        """Initialize metadata.

        Run at the beginning of each execution request.
        """
        md = super().init_metadata(parent)
        # FIXME: remove deprecated ipyparallel-specific code
        # This is required for ipyparallel < 5.0
        md.update(
            {
                "dependencies_met": True,
                "engine": self.ident,
            }
        )
        return md

    def finish_metadata(self, parent, metadata, reply_content):
        """Finish populating metadata.

        Run after completing an execution request.
        """
        # FIXME: remove deprecated ipyparallel-specific code
        # This is required by ipyparallel < 5.0
        metadata["status"] = reply_content["status"]
        if reply_content["status"] == "error" and reply_content["ename"] == "UnmetDependency":
            metadata["dependencies_met"] = False

        return metadata

    def _forward_input(self, allow_stdin=False):
        """Forward raw_input and getpass to the current frontend.

        via input_request
        """
        self._allow_stdin = allow_stdin

        self._sys_raw_input = builtins.input
        builtins.input = self.raw_input

        self._save_getpass = getpass.getpass
        getpass.getpass = self.getpass

    def _restore_input(self):
        """Restore raw_input, getpass"""
        builtins.input = self._sys_raw_input

        getpass.getpass = self._save_getpass

    @property
    def execution_count(self):
        return self.shell.execution_count

    @execution_count.setter
    def execution_count(self, value):
        # Ignore the incrementing done by KernelBase, in favour of our shell's
        # execution counter.
        pass

    @contextmanager
    def _cancel_on_sigint(self, future):
        """ContextManager for capturing SIGINT and cancelling a future

        SIGINT raises in the event loop when running async code,
        but we want it to halt a coroutine.

        Ideally, it would raise KeyboardInterrupt,
        but this turns it into a CancelledError.
        At least it gets a decent traceback to the user.
        """
        sigint_future: asyncio.Future[int] = asyncio.Future()

        # whichever future finishes first,
        # cancel the other one
        def cancel_unless_done(f, _ignored):
            if f.cancelled() or f.done():
                return
            f.cancel()

        # when sigint finishes,
        # abort the coroutine with CancelledError
        sigint_future.add_done_callback(partial(cancel_unless_done, future))
        # when the main future finishes,
        # stop watching for SIGINT events
        future.add_done_callback(partial(cancel_unless_done, sigint_future))

        def handle_sigint(*args):
            def set_sigint_result():
                if sigint_future.cancelled() or sigint_future.done():
                    return
                sigint_future.set_result(1)

            # use add_callback for thread safety
            self.io_loop.add_callback(set_sigint_result)

        # set the custom sigint hander during this context
        save_sigint = signal.signal(signal.SIGINT, handle_sigint)
        try:
            yield
        finally:
            # restore the previous sigint handler
            signal.signal(signal.SIGINT, save_sigint)

    async def do_execute(
        self,
        code,
        silent,
        store_history=True,
        user_expressions=None,
        allow_stdin=False,
        *,
        cell_id=None,
    ):
        """Handle code execution."""
        shell = self.shell  # we'll need this a lot here

        self._forward_input(allow_stdin)

        reply_content: t.Dict[str, t.Any] = {}
        if hasattr(shell, "run_cell_async") and hasattr(shell, "should_run_async"):
            run_cell = shell.run_cell_async
            should_run_async = shell.should_run_async
            with_cell_id = _accepts_cell_id(run_cell)
        else:
            should_run_async = lambda cell: False  # noqa
            # older IPython,
            # use blocking run_cell and wrap it in coroutine

            async def run_cell(*args, **kwargs):
                return shell.run_cell(*args, **kwargs)

            with_cell_id = _accepts_cell_id(shell.run_cell)
        try:
            # default case: runner is asyncio and asyncio is already running
            # TODO: this should check every case for "are we inside the runner",
            # not just asyncio
            preprocessing_exc_tuple = None
            try:
                transformed_cell = self.shell.transform_cell(code)
            except Exception:
                transformed_cell = code
                preprocessing_exc_tuple = sys.exc_info()

            if (
                _asyncio_runner
                and shell.loop_runner is _asyncio_runner
                and asyncio.get_event_loop().is_running()
                and should_run_async(
                    code,
                    transformed_cell=transformed_cell,
                    preprocessing_exc_tuple=preprocessing_exc_tuple,
                )
            ):
                if with_cell_id:
                    coro = run_cell(
                        code,
                        store_history=store_history,
                        silent=silent,
                        transformed_cell=transformed_cell,
                        preprocessing_exc_tuple=preprocessing_exc_tuple,
                        cell_id=cell_id,
                    )
                else:
                    coro = run_cell(
                        code,
                        store_history=store_history,
                        silent=silent,
                        transformed_cell=transformed_cell,
                        preprocessing_exc_tuple=preprocessing_exc_tuple,
                    )

                coro_future = asyncio.ensure_future(coro)

                with self._cancel_on_sigint(coro_future):
                    res = None
                    try:
                        res = await coro_future
                    finally:
                        shell.events.trigger("post_execute")
                        if not silent:
                            shell.events.trigger("post_run_cell", res)
            else:
                # runner isn't already running,
                # make synchronous call,
                # letting shell dispatch to loop runners
                if with_cell_id:
                    res = shell.run_cell(
                        code,
                        store_history=store_history,
                        silent=silent,
                        cell_id=cell_id,
                    )
                else:
                    res = shell.run_cell(code, store_history=store_history, silent=silent)
        finally:
            self._restore_input()

        err = res.error_before_exec if res.error_before_exec is not None else res.error_in_exec

        if res.success:
            reply_content["status"] = "ok"
        else:
            reply_content["status"] = "error"

            reply_content.update(
                {
                    "traceback": shell._last_traceback or [],
                    "ename": str(type(err).__name__),
                    "evalue": str(err),
                }
            )

            # FIXME: deprecated piece for ipyparallel (remove in 5.0):
            e_info = dict(engine_uuid=self.ident, engine_id=self.int_id, method="execute")
            reply_content["engine_info"] = e_info

        # Return the execution counter so clients can display prompts
        reply_content["execution_count"] = shell.execution_count - 1

        if "traceback" in reply_content:
            self.log.info(
                "Exception in execute request:\n%s",
                "\n".join(reply_content["traceback"]),
            )

        # At this point, we can tell whether the main code execution succeeded
        # or not.  If it did, we proceed to evaluate user_expressions
        if reply_content["status"] == "ok":
            reply_content["user_expressions"] = shell.user_expressions(user_expressions or {})
        else:
            # If there was an error, don't even try to compute expressions
            reply_content["user_expressions"] = {}

        # Payloads should be retrieved regardless of outcome, so we can both
        # recover partial output (that could have been generated early in a
        # block, before an error) and always clear the payload system.
        reply_content["payload"] = shell.payload_manager.read_payload()
        # Be aggressive about clearing the payload because we don't want
        # it to sit in memory until the next execute_request comes in.
        shell.payload_manager.clear_payload()

        return reply_content

    def do_complete(self, code, cursor_pos):
        """Handle code completion."""
        if _use_experimental_60_completion and self.use_experimental_completions:
            return self._experimental_do_complete(code, cursor_pos)

        # FIXME: IPython completers currently assume single line,
        # but completion messages give multi-line context
        # For now, extract line from cell, based on cursor_pos:
        if cursor_pos is None:
            cursor_pos = len(code)
        line, offset = line_at_cursor(code, cursor_pos)
        line_cursor = cursor_pos - offset
        txt, matches = self.shell.complete("", line, line_cursor)
        return {
            "matches": matches,
            "cursor_end": cursor_pos,
            "cursor_start": cursor_pos - len(txt),
            "metadata": {},
            "status": "ok",
        }

    async def do_debug_request(self, msg):
        """Handle a debug request."""
        if _is_debugpy_available:
            return await self.debugger.process_request(msg)

    def _experimental_do_complete(self, code, cursor_pos):
        """
        Experimental completions from IPython, using Jedi.
        """
        if cursor_pos is None:
            cursor_pos = len(code)
        with _provisionalcompleter():
            raw_completions = self.shell.Completer.completions(code, cursor_pos)
            completions = list(_rectify_completions(code, raw_completions))

            comps = []
            for comp in completions:
                comps.append(
                    dict(
                        start=comp.start,
                        end=comp.end,
                        text=comp.text,
                        type=comp.type,
                        signature=comp.signature,
                    )
                )

        if completions:
            s = completions[0].start
            e = completions[0].end
            matches = [c.text for c in completions]
        else:
            s = cursor_pos
            e = cursor_pos
            matches = []

        return {
            "matches": matches,
            "cursor_end": e,
            "cursor_start": s,
            "metadata": {_EXPERIMENTAL_KEY_NAME: comps},
            "status": "ok",
        }

    def do_inspect(self, code, cursor_pos, detail_level=0, omit_sections=()):
        """Handle code inspection."""
        name = token_at_cursor(code, cursor_pos)

        reply_content: t.Dict[str, t.Any] = {"status": "ok"}
        reply_content["data"] = {}
        reply_content["metadata"] = {}
        try:
            if release.version_info >= (8,):
                # `omit_sections` keyword will be available in IPython 8, see
                # https://github.com/ipython/ipython/pull/13343
                bundle = self.shell.object_inspect_mime(
                    name,
                    detail_level=detail_level,
                    omit_sections=omit_sections,
                )
            else:
                bundle = self.shell.object_inspect_mime(name, detail_level=detail_level)
            reply_content["data"].update(bundle)
            if not self.shell.enable_html_pager:
                reply_content["data"].pop("text/html")
            reply_content["found"] = True
        except KeyError:
            reply_content["found"] = False

        return reply_content

    def do_history(
        self,
        hist_access_type,
        output,
        raw,
        session=0,
        start=0,
        stop=None,
        n=None,
        pattern=None,
        unique=False,
    ):
        """Handle code history."""
        if hist_access_type == "tail":
            hist = self.shell.history_manager.get_tail(
                n, raw=raw, output=output, include_latest=True
            )

        elif hist_access_type == "range":
            hist = self.shell.history_manager.get_range(
                session, start, stop, raw=raw, output=output
            )

        elif hist_access_type == "search":
            hist = self.shell.history_manager.search(
                pattern, raw=raw, output=output, n=n, unique=unique
            )
        else:
            hist = []

        return {
            "status": "ok",
            "history": list(hist),
        }

    def do_shutdown(self, restart):
        """Handle kernel shutdown."""
        self.shell.exit_now = True
        return dict(status="ok", restart=restart)

    def do_is_complete(self, code):
        """Handle an is_complete request."""
        transformer_manager = getattr(self.shell, "input_transformer_manager", None)
        if transformer_manager is None:
            # input_splitter attribute is deprecated
            transformer_manager = self.shell.input_splitter
        status, indent_spaces = transformer_manager.check_complete(code)
        r = {"status": status}
        if status == "incomplete":
            r["indent"] = " " * indent_spaces
        return r

    def do_apply(self, content, bufs, msg_id, reply_metadata):
        """Handle an apply request."""
        try:
            from ipyparallel.serialize import serialize_object, unpack_apply_message
        except ImportError:
            from .serialize import serialize_object, unpack_apply_message

        shell = self.shell
        try:
            working = shell.user_ns

            prefix = "_" + str(msg_id).replace("-", "") + "_"
            f, args, kwargs = unpack_apply_message(bufs, working, copy=False)

            fname = getattr(f, "__name__", "f")

            fname = prefix + "f"
            argname = prefix + "args"
            kwargname = prefix + "kwargs"
            resultname = prefix + "result"

            ns = {fname: f, argname: args, kwargname: kwargs, resultname: None}
            # print ns
            working.update(ns)
            code = f"{resultname} = {fname}(*{argname},**{kwargname})"
            try:
                exec(code, shell.user_global_ns, shell.user_ns)  # noqa
                result = working.get(resultname)
            finally:
                for key in ns:
                    working.pop(key)

            result_buf = serialize_object(
                result,
                buffer_threshold=self.session.buffer_threshold,
                item_threshold=self.session.item_threshold,
            )

        except BaseException as e:
            # invoke IPython traceback formatting
            shell.showtraceback()
            reply_content = {
                "traceback": shell._last_traceback or [],
                "ename": str(type(e).__name__),
                "evalue": str(e),
            }
            # FIXME: deprecated piece for ipyparallel (remove in 5.0):
            e_info = dict(engine_uuid=self.ident, engine_id=self.int_id, method="apply")
            reply_content["engine_info"] = e_info

            self.send_response(
                self.iopub_socket,
                "error",
                reply_content,
                ident=self._topic("error"),
            )
            self.log.info("Exception in apply request:\n%s", "\n".join(reply_content["traceback"]))
            result_buf = []
            reply_content["status"] = "error"
        else:
            reply_content = {"status": "ok"}

        return reply_content, result_buf

    def do_clear(self):
        """Clear the kernel."""
        self.shell.reset(False)
        return dict(status="ok")


# This exists only for backwards compatibility - use IPythonKernel instead


class Kernel(IPythonKernel):
    """DEPRECATED.  An alias for the IPython kernel class."""

    def __init__(self, *args, **kwargs):  # pragma: no cover
        """DEPRECATED."""
        import warnings

        warnings.warn(
            "Kernel is a deprecated alias of ipykernel.ipkernel.IPythonKernel",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
