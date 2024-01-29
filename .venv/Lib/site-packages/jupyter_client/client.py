"""Base class to manage the interaction with a running kernel"""
# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
import asyncio
import inspect
import sys
import time
import typing as t
from functools import partial
from getpass import getpass
from queue import Empty

import zmq.asyncio
from jupyter_core.utils import ensure_async
from traitlets import Any, Bool, Instance, Type

from .channels import major_protocol_version
from .channelsabc import ChannelABC, HBChannelABC
from .clientabc import KernelClientABC
from .connect import ConnectionFileMixin
from .session import Session

# some utilities to validate message structure, these might get moved elsewhere
# if they prove to have more generic utility


def validate_string_dict(dct: t.Dict[str, str]) -> None:
    """Validate that the input is a dict with string keys and values.

    Raises ValueError if not."""
    for k, v in dct.items():
        if not isinstance(k, str):
            raise ValueError("key %r in dict must be a string" % k)
        if not isinstance(v, str):
            raise ValueError("value %r in dict must be a string" % v)


def reqrep(wrapped: t.Callable, meth: t.Callable, channel: str = "shell") -> t.Callable:
    wrapped = wrapped(meth, channel)
    if not meth.__doc__:
        # python -OO removes docstrings,
        # so don't bother building the wrapped docstring
        return wrapped

    basedoc, _ = meth.__doc__.split("Returns\n", 1)
    parts = [basedoc.strip()]
    if "Parameters" not in basedoc:
        parts.append(
            """
        Parameters
        ----------
        """
        )
    parts.append(
        """
        reply: bool (default: False)
            Whether to wait for and return reply
        timeout: float or None (default: None)
            Timeout to use when waiting for a reply

        Returns
        -------
        msg_id: str
            The msg_id of the request sent, if reply=False (default)
        reply: dict
            The reply message for this request, if reply=True
    """
    )
    wrapped.__doc__ = "\n".join(parts)
    return wrapped


class KernelClient(ConnectionFileMixin):
    """Communicates with a single kernel on any host via zmq channels.

    There are five channels associated with each kernel:

    * shell: for request/reply calls to the kernel.
    * iopub: for the kernel to publish results to frontends.
    * hb: for monitoring the kernel's heartbeat.
    * stdin: for frontends to reply to raw_input calls in the kernel.
    * control: for kernel management calls to the kernel.

    The messages that can be sent on these channels are exposed as methods of the
    client (KernelClient.execute, complete, history, etc.). These methods only
    send the message, they don't wait for a reply. To get results, use e.g.
    :meth:`get_shell_msg` to fetch messages from the shell channel.
    """

    # The PyZMQ Context to use for communication with the kernel.
    context = Instance(zmq.Context)

    _created_context = Bool(False)

    def _context_default(self) -> zmq.Context:
        self._created_context = True
        return zmq.Context()

    # The classes to use for the various channels
    shell_channel_class = Type(ChannelABC)
    iopub_channel_class = Type(ChannelABC)
    stdin_channel_class = Type(ChannelABC)
    hb_channel_class = Type(HBChannelABC)
    control_channel_class = Type(ChannelABC)

    # Protected traits
    _shell_channel = Any()
    _iopub_channel = Any()
    _stdin_channel = Any()
    _hb_channel = Any()
    _control_channel = Any()

    # flag for whether execute requests should be allowed to call raw_input:
    allow_stdin: bool = True

    def __del__(self) -> None:
        """Handle garbage collection.  Destroy context if applicable."""
        if (
            self._created_context
            and self.context is not None  # type:ignore[redundant-expr]
            and not self.context.closed
        ):
            if self.channels_running:
                if self.log:
                    self.log.warning("Could not destroy zmq context for %s", self)
            else:
                if self.log:
                    self.log.debug("Destroying zmq context for %s", self)
                self.context.destroy()
        try:
            super_del = super().__del__  # type:ignore[misc]
        except AttributeError:
            pass
        else:
            super_del()

    # --------------------------------------------------------------------------
    # Channel proxy methods
    # --------------------------------------------------------------------------

    async def _async_get_shell_msg(self, *args: t.Any, **kwargs: t.Any) -> t.Dict[str, t.Any]:
        """Get a message from the shell channel"""
        return await ensure_async(self.shell_channel.get_msg(*args, **kwargs))

    async def _async_get_iopub_msg(self, *args: t.Any, **kwargs: t.Any) -> t.Dict[str, t.Any]:
        """Get a message from the iopub channel"""
        return await ensure_async(self.iopub_channel.get_msg(*args, **kwargs))

    async def _async_get_stdin_msg(self, *args: t.Any, **kwargs: t.Any) -> t.Dict[str, t.Any]:
        """Get a message from the stdin channel"""
        return await ensure_async(self.stdin_channel.get_msg(*args, **kwargs))

    async def _async_get_control_msg(self, *args: t.Any, **kwargs: t.Any) -> t.Dict[str, t.Any]:
        """Get a message from the control channel"""
        return await ensure_async(self.control_channel.get_msg(*args, **kwargs))

    async def _async_wait_for_ready(self, timeout: t.Optional[float] = None) -> None:
        """Waits for a response when a client is blocked

        - Sets future time for timeout
        - Blocks on shell channel until a message is received
        - Exit if the kernel has died
        - If client times out before receiving a message from the kernel, send RuntimeError
        - Flush the IOPub channel
        """
        if timeout is None:
            timeout = float("inf")
        abs_timeout = time.time() + timeout

        from .manager import KernelManager

        if not isinstance(self.parent, KernelManager):
            # This Client was not created by a KernelManager,
            # so wait for kernel to become responsive to heartbeats
            # before checking for kernel_info reply
            while not await self._async_is_alive():
                if time.time() > abs_timeout:
                    raise RuntimeError(
                        "Kernel didn't respond to heartbeats in %d seconds and timed out" % timeout
                    )
                await asyncio.sleep(0.2)

        # Wait for kernel info reply on shell channel
        while True:
            self.kernel_info()
            try:
                msg = await ensure_async(self.shell_channel.get_msg(timeout=1))
            except Empty:
                pass
            else:
                if msg["msg_type"] == "kernel_info_reply":
                    # Checking that IOPub is connected. If it is not connected, start over.
                    try:
                        await ensure_async(self.iopub_channel.get_msg(timeout=0.2))
                    except Empty:
                        pass
                    else:
                        self._handle_kernel_info_reply(msg)
                        break

            if not await self._async_is_alive():
                msg = "Kernel died before replying to kernel_info"
                raise RuntimeError(msg)

            # Check if current time is ready check time plus timeout
            if time.time() > abs_timeout:
                raise RuntimeError("Kernel didn't respond in %d seconds" % timeout)

        # Flush IOPub channel
        while True:
            try:
                msg = await ensure_async(self.iopub_channel.get_msg(timeout=0.2))
            except Empty:
                break

    async def _async_recv_reply(
        self, msg_id: str, timeout: t.Optional[float] = None, channel: str = "shell"
    ) -> t.Dict[str, t.Any]:
        """Receive and return the reply for a given request"""
        if timeout is not None:
            deadline = time.monotonic() + timeout
        while True:
            if timeout is not None:
                timeout = max(0, deadline - time.monotonic())
            try:
                if channel == "control":
                    reply = await self._async_get_control_msg(timeout=timeout)
                else:
                    reply = await self._async_get_shell_msg(timeout=timeout)
            except Empty as e:
                msg = "Timeout waiting for reply"
                raise TimeoutError(msg) from e
            if reply["parent_header"].get("msg_id") != msg_id:
                # not my reply, someone may have forgotten to retrieve theirs
                continue
            return reply

    async def _stdin_hook_default(self, msg: t.Dict[str, t.Any]) -> None:
        """Handle an input request"""
        content = msg["content"]
        prompt = getpass if content.get("password", False) else input

        try:
            raw_data = prompt(content["prompt"])  # type:ignore[operator]
        except EOFError:
            # turn EOFError into EOF character
            raw_data = "\x04"
        except KeyboardInterrupt:
            sys.stdout.write("\n")
            return

        # only send stdin reply if there *was not* another request
        # or execution finished while we were reading.
        if not (await self.stdin_channel.msg_ready() or await self.shell_channel.msg_ready()):
            self.input(raw_data)

    def _output_hook_default(self, msg: t.Dict[str, t.Any]) -> None:
        """Default hook for redisplaying plain-text output"""
        msg_type = msg["header"]["msg_type"]
        content = msg["content"]
        if msg_type == "stream":
            stream = getattr(sys, content["name"])
            stream.write(content["text"])
        elif msg_type in ("display_data", "execute_result"):
            sys.stdout.write(content["data"].get("text/plain", ""))
        elif msg_type == "error":
            sys.stderr.write("\n".join(content["traceback"]))

    def _output_hook_kernel(
        self,
        session: Session,
        socket: zmq.sugar.socket.Socket,
        parent_header: t.Any,
        msg: t.Dict[str, t.Any],
    ) -> None:
        """Output hook when running inside an IPython kernel

        adds rich output support.
        """
        msg_type = msg["header"]["msg_type"]
        if msg_type in ("display_data", "execute_result", "error"):
            session.send(socket, msg_type, msg["content"], parent=parent_header)
        else:
            self._output_hook_default(msg)

    # --------------------------------------------------------------------------
    # Channel management methods
    # --------------------------------------------------------------------------

    def start_channels(
        self,
        shell: bool = True,
        iopub: bool = True,
        stdin: bool = True,
        hb: bool = True,
        control: bool = True,
    ) -> None:
        """Starts the channels for this kernel.

        This will create the channels if they do not exist and then start
        them (their activity runs in a thread). If port numbers of 0 are
        being used (random ports) then you must first call
        :meth:`start_kernel`. If the channels have been stopped and you
        call this, :class:`RuntimeError` will be raised.
        """
        if iopub:
            self.iopub_channel.start()
        if shell:
            self.shell_channel.start()
        if stdin:
            self.stdin_channel.start()
            self.allow_stdin = True
        else:
            self.allow_stdin = False
        if hb:
            self.hb_channel.start()
        if control:
            self.control_channel.start()

    def stop_channels(self) -> None:
        """Stops all the running channels for this kernel.

        This stops their event loops and joins their threads.
        """
        if self.shell_channel.is_alive():
            self.shell_channel.stop()
        if self.iopub_channel.is_alive():
            self.iopub_channel.stop()
        if self.stdin_channel.is_alive():
            self.stdin_channel.stop()
        if self.hb_channel.is_alive():
            self.hb_channel.stop()
        if self.control_channel.is_alive():
            self.control_channel.stop()

    @property
    def channels_running(self) -> bool:
        """Are any of the channels created and running?"""
        return (
            (self._shell_channel and self.shell_channel.is_alive())
            or (self._iopub_channel and self.iopub_channel.is_alive())
            or (self._stdin_channel and self.stdin_channel.is_alive())
            or (self._hb_channel and self.hb_channel.is_alive())
            or (self._control_channel and self.control_channel.is_alive())
        )

    ioloop = None  # Overridden in subclasses that use pyzmq event loop

    @property
    def shell_channel(self) -> t.Any:
        """Get the shell channel object for this kernel."""
        if self._shell_channel is None:
            url = self._make_url("shell")
            self.log.debug("connecting shell channel to %s", url)
            socket = self.connect_shell(identity=self.session.bsession)
            self._shell_channel = self.shell_channel_class(  # type:ignore[call-arg,abstract]
                socket, self.session, self.ioloop
            )
        return self._shell_channel

    @property
    def iopub_channel(self) -> t.Any:
        """Get the iopub channel object for this kernel."""
        if self._iopub_channel is None:
            url = self._make_url("iopub")
            self.log.debug("connecting iopub channel to %s", url)
            socket = self.connect_iopub()
            self._iopub_channel = self.iopub_channel_class(  # type:ignore[call-arg,abstract]
                socket, self.session, self.ioloop
            )
        return self._iopub_channel

    @property
    def stdin_channel(self) -> t.Any:
        """Get the stdin channel object for this kernel."""
        if self._stdin_channel is None:
            url = self._make_url("stdin")
            self.log.debug("connecting stdin channel to %s", url)
            socket = self.connect_stdin(identity=self.session.bsession)
            self._stdin_channel = self.stdin_channel_class(  # type:ignore[call-arg,abstract]
                socket, self.session, self.ioloop
            )
        return self._stdin_channel

    @property
    def hb_channel(self) -> t.Any:
        """Get the hb channel object for this kernel."""
        if self._hb_channel is None:
            url = self._make_url("hb")
            self.log.debug("connecting heartbeat channel to %s", url)
            self._hb_channel = self.hb_channel_class(  # type:ignore[call-arg,abstract]
                self.context, self.session, url
            )
        return self._hb_channel

    @property
    def control_channel(self) -> t.Any:
        """Get the control channel object for this kernel."""
        if self._control_channel is None:
            url = self._make_url("control")
            self.log.debug("connecting control channel to %s", url)
            socket = self.connect_control(identity=self.session.bsession)
            self._control_channel = self.control_channel_class(  # type:ignore[call-arg,abstract]
                socket, self.session, self.ioloop
            )
        return self._control_channel

    async def _async_is_alive(self) -> bool:
        """Is the kernel process still running?"""
        from .manager import KernelManager

        if isinstance(self.parent, KernelManager):
            # This KernelClient was created by a KernelManager,
            # we can ask the parent KernelManager:
            return await self.parent._async_is_alive()
        if self._hb_channel is not None:
            # We don't have access to the KernelManager,
            # so we use the heartbeat.
            return self._hb_channel.is_beating()
        # no heartbeat and not local, we can't tell if it's running,
        # so naively return True
        return True

    async def _async_execute_interactive(
        self,
        code: str,
        silent: bool = False,
        store_history: bool = True,
        user_expressions: t.Optional[t.Dict[str, t.Any]] = None,
        allow_stdin: t.Optional[bool] = None,
        stop_on_error: bool = True,
        timeout: t.Optional[float] = None,
        output_hook: t.Optional[t.Callable] = None,
        stdin_hook: t.Optional[t.Callable] = None,
    ) -> t.Dict[str, t.Any]:
        """Execute code in the kernel interactively

        Output will be redisplayed, and stdin prompts will be relayed as well.
        If an IPython kernel is detected, rich output will be displayed.

        You can pass a custom output_hook callable that will be called
        with every IOPub message that is produced instead of the default redisplay.

        .. versionadded:: 5.0

        Parameters
        ----------
        code : str
            A string of code in the kernel's language.

        silent : bool, optional (default False)
            If set, the kernel will execute the code as quietly possible, and
            will force store_history to be False.

        store_history : bool, optional (default True)
            If set, the kernel will store command history.  This is forced
            to be False if silent is True.

        user_expressions : dict, optional
            A dict mapping names to expressions to be evaluated in the user's
            dict. The expression values are returned as strings formatted using
            :func:`repr`.

        allow_stdin : bool, optional (default self.allow_stdin)
            Flag for whether the kernel can send stdin requests to frontends.

            Some frontends (e.g. the Notebook) do not support stdin requests.
            If raw_input is called from code executed from such a frontend, a
            StdinNotImplementedError will be raised.

        stop_on_error: bool, optional (default True)
            Flag whether to abort the execution queue, if an exception is encountered.

        timeout: float or None (default: None)
            Timeout to use when waiting for a reply

        output_hook: callable(msg)
            Function to be called with output messages.
            If not specified, output will be redisplayed.

        stdin_hook: callable(msg)
            Function or awaitable to be called with stdin_request messages.
            If not specified, input/getpass will be called.

        Returns
        -------
        reply: dict
            The reply message for this request
        """
        if not self.iopub_channel.is_alive():
            emsg = "IOPub channel must be running to receive output"
            raise RuntimeError(emsg)
        if allow_stdin is None:
            allow_stdin = self.allow_stdin
        if allow_stdin and not self.stdin_channel.is_alive():
            emsg = "stdin channel must be running to allow input"
            raise RuntimeError(emsg)
        msg_id = await ensure_async(
            self.execute(
                code,
                silent=silent,
                store_history=store_history,
                user_expressions=user_expressions,
                allow_stdin=allow_stdin,
                stop_on_error=stop_on_error,
            )
        )
        if stdin_hook is None:
            stdin_hook = self._stdin_hook_default
        # detect IPython kernel
        if output_hook is None and "IPython" in sys.modules:
            from IPython import get_ipython

            ip = get_ipython()  # type:ignore[no-untyped-call]
            in_kernel = getattr(ip, "kernel", False)
            if in_kernel:
                output_hook = partial(
                    self._output_hook_kernel,
                    ip.display_pub.session,
                    ip.display_pub.pub_socket,
                    ip.display_pub.parent_header,
                )
        if output_hook is None:
            # default: redisplay plain-text outputs
            output_hook = self._output_hook_default

        # set deadline based on timeout
        if timeout is not None:
            deadline = time.monotonic() + timeout
        else:
            timeout_ms = None

        poller = zmq.Poller()
        iopub_socket = self.iopub_channel.socket
        poller.register(iopub_socket, zmq.POLLIN)
        if allow_stdin:
            stdin_socket = self.stdin_channel.socket
            poller.register(stdin_socket, zmq.POLLIN)
        else:
            stdin_socket = None

        # wait for output and redisplay it
        while True:
            if timeout is not None:
                timeout = max(0, deadline - time.monotonic())
                timeout_ms = int(1000 * timeout)
            events = dict(poller.poll(timeout_ms))
            if not events:
                emsg = "Timeout waiting for output"
                raise TimeoutError(emsg)
            if stdin_socket in events:
                req = await ensure_async(self.stdin_channel.get_msg(timeout=0))
                res = stdin_hook(req)
                if inspect.isawaitable(res):
                    await res
                continue
            if iopub_socket not in events:
                continue

            msg = await ensure_async(self.iopub_channel.get_msg(timeout=0))

            if msg["parent_header"].get("msg_id") != msg_id:
                # not from my request
                continue
            output_hook(msg)

            # stop on idle
            if (
                msg["header"]["msg_type"] == "status"
                and msg["content"]["execution_state"] == "idle"
            ):
                break

        # output is done, get the reply
        if timeout is not None:
            timeout = max(0, deadline - time.monotonic())
        return await self._async_recv_reply(msg_id, timeout=timeout)

    # Methods to send specific messages on channels
    def execute(
        self,
        code: str,
        silent: bool = False,
        store_history: bool = True,
        user_expressions: t.Optional[t.Dict[str, t.Any]] = None,
        allow_stdin: t.Optional[bool] = None,
        stop_on_error: bool = True,
    ) -> str:
        """Execute code in the kernel.

        Parameters
        ----------
        code : str
            A string of code in the kernel's language.

        silent : bool, optional (default False)
            If set, the kernel will execute the code as quietly possible, and
            will force store_history to be False.

        store_history : bool, optional (default True)
            If set, the kernel will store command history.  This is forced
            to be False if silent is True.

        user_expressions : dict, optional
            A dict mapping names to expressions to be evaluated in the user's
            dict. The expression values are returned as strings formatted using
            :func:`repr`.

        allow_stdin : bool, optional (default self.allow_stdin)
            Flag for whether the kernel can send stdin requests to frontends.

            Some frontends (e.g. the Notebook) do not support stdin requests.
            If raw_input is called from code executed from such a frontend, a
            StdinNotImplementedError will be raised.

        stop_on_error: bool, optional (default True)
            Flag whether to abort the execution queue, if an exception is encountered.

        Returns
        -------
        The msg_id of the message sent.
        """
        if user_expressions is None:
            user_expressions = {}
        if allow_stdin is None:
            allow_stdin = self.allow_stdin

        # Don't waste network traffic if inputs are invalid
        if not isinstance(code, str):
            raise ValueError("code %r must be a string" % code)
        validate_string_dict(user_expressions)

        # Create class for content/msg creation. Related to, but possibly
        # not in Session.
        content = {
            "code": code,
            "silent": silent,
            "store_history": store_history,
            "user_expressions": user_expressions,
            "allow_stdin": allow_stdin,
            "stop_on_error": stop_on_error,
        }
        msg = self.session.msg("execute_request", content)
        self.shell_channel.send(msg)
        return msg["header"]["msg_id"]

    def complete(self, code: str, cursor_pos: t.Optional[int] = None) -> str:
        """Tab complete text in the kernel's namespace.

        Parameters
        ----------
        code : str
            The context in which completion is requested.
            Can be anything between a variable name and an entire cell.
        cursor_pos : int, optional
            The position of the cursor in the block of code where the completion was requested.
            Default: ``len(code)``

        Returns
        -------
        The msg_id of the message sent.
        """
        if cursor_pos is None:
            cursor_pos = len(code)
        content = {"code": code, "cursor_pos": cursor_pos}
        msg = self.session.msg("complete_request", content)
        self.shell_channel.send(msg)
        return msg["header"]["msg_id"]

    def inspect(self, code: str, cursor_pos: t.Optional[int] = None, detail_level: int = 0) -> str:
        """Get metadata information about an object in the kernel's namespace.

        It is up to the kernel to determine the appropriate object to inspect.

        Parameters
        ----------
        code : str
            The context in which info is requested.
            Can be anything between a variable name and an entire cell.
        cursor_pos : int, optional
            The position of the cursor in the block of code where the info was requested.
            Default: ``len(code)``
        detail_level : int, optional
            The level of detail for the introspection (0-2)

        Returns
        -------
        The msg_id of the message sent.
        """
        if cursor_pos is None:
            cursor_pos = len(code)
        content = {
            "code": code,
            "cursor_pos": cursor_pos,
            "detail_level": detail_level,
        }
        msg = self.session.msg("inspect_request", content)
        self.shell_channel.send(msg)
        return msg["header"]["msg_id"]

    def history(
        self,
        raw: bool = True,
        output: bool = False,
        hist_access_type: str = "range",
        **kwargs: t.Any,
    ) -> str:
        """Get entries from the kernel's history list.

        Parameters
        ----------
        raw : bool
            If True, return the raw input.
        output : bool
            If True, then return the output as well.
        hist_access_type : str
            'range' (fill in session, start and stop params), 'tail' (fill in n)
             or 'search' (fill in pattern param).

        session : int
            For a range request, the session from which to get lines. Session
            numbers are positive integers; negative ones count back from the
            current session.
        start : int
            The first line number of a history range.
        stop : int
            The final (excluded) line number of a history range.

        n : int
            The number of lines of history to get for a tail request.

        pattern : str
            The glob-syntax pattern for a search request.

        Returns
        -------
        The ID of the message sent.
        """
        if hist_access_type == "range":
            kwargs.setdefault("session", 0)
            kwargs.setdefault("start", 0)
        content = dict(raw=raw, output=output, hist_access_type=hist_access_type, **kwargs)
        msg = self.session.msg("history_request", content)
        self.shell_channel.send(msg)
        return msg["header"]["msg_id"]

    def kernel_info(self) -> str:
        """Request kernel info

        Returns
        -------
        The msg_id of the message sent
        """
        msg = self.session.msg("kernel_info_request")
        self.shell_channel.send(msg)
        return msg["header"]["msg_id"]

    def comm_info(self, target_name: t.Optional[str] = None) -> str:
        """Request comm info

        Returns
        -------
        The msg_id of the message sent
        """
        content = {} if target_name is None else {"target_name": target_name}
        msg = self.session.msg("comm_info_request", content)
        self.shell_channel.send(msg)
        return msg["header"]["msg_id"]

    def _handle_kernel_info_reply(self, msg: t.Dict[str, t.Any]) -> None:
        """handle kernel info reply

        sets protocol adaptation version. This might
        be run from a separate thread.
        """
        adapt_version = int(msg["content"]["protocol_version"].split(".")[0])
        if adapt_version != major_protocol_version:
            self.session.adapt_version = adapt_version

    def is_complete(self, code: str) -> str:
        """Ask the kernel whether some code is complete and ready to execute.

        Returns
        -------
        The ID of the message sent.
        """
        msg = self.session.msg("is_complete_request", {"code": code})
        self.shell_channel.send(msg)
        return msg["header"]["msg_id"]

    def input(self, string: str) -> None:
        """Send a string of raw input to the kernel.

        This should only be called in response to the kernel sending an
        ``input_request`` message on the stdin channel.

        Returns
        -------
        The ID of the message sent.
        """
        content = {"value": string}
        msg = self.session.msg("input_reply", content)
        self.stdin_channel.send(msg)

    def shutdown(self, restart: bool = False) -> str:
        """Request an immediate kernel shutdown on the control channel.

        Upon receipt of the (empty) reply, client code can safely assume that
        the kernel has shut down and it's safe to forcefully terminate it if
        it's still alive.

        The kernel will send the reply via a function registered with Python's
        atexit module, ensuring it's truly done as the kernel is done with all
        normal operation.

        Returns
        -------
        The msg_id of the message sent
        """
        # Send quit message to kernel. Once we implement kernel-side setattr,
        # this should probably be done that way, but for now this will do.
        msg = self.session.msg("shutdown_request", {"restart": restart})
        self.control_channel.send(msg)
        return msg["header"]["msg_id"]


KernelClientABC.register(KernelClient)
