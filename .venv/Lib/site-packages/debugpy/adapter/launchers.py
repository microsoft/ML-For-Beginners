# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root
# for license information.

import os
import subprocess
import sys

from debugpy import adapter, common
from debugpy.common import log, messaging, sockets
from debugpy.adapter import components, servers


class Launcher(components.Component):
    """Handles the launcher side of a debug session."""

    message_handler = components.Component.message_handler

    def __init__(self, session, stream):
        with session:
            assert not session.launcher
            super().__init__(session, stream)

            self.pid = None
            """Process ID of the debuggee process, as reported by the launcher."""

            self.exit_code = None
            """Exit code of the debuggee process."""

            session.launcher = self

    @message_handler
    def process_event(self, event):
        self.pid = event("systemProcessId", int)
        self.client.propagate_after_start(event)

    @message_handler
    def output_event(self, event):
        self.client.propagate_after_start(event)

    @message_handler
    def exited_event(self, event):
        self.exit_code = event("exitCode", int)
        # We don't want to tell the client about this just yet, because it will then
        # want to disconnect, and the launcher might still be waiting for keypress
        # (if wait-on-exit was enabled). Instead, we'll report the event when we
        # receive "terminated" from the launcher, right before it exits.

    @message_handler
    def terminated_event(self, event):
        try:
            self.client.channel.send_event("exited", {"exitCode": self.exit_code})
        except Exception:
            pass
        self.channel.close()

    def terminate_debuggee(self):
        with self.session:
            if self.exit_code is None:
                try:
                    self.channel.request("terminate")
                except Exception:
                    pass


def spawn_debuggee(
    session,
    start_request,
    python,
    launcher_path,
    adapter_host,
    args,
    shell_expand_args,
    cwd,
    console,
    console_title,
    sudo,
):
    # -E tells sudo to propagate environment variables to the target process - this
    # is necessary for launcher to get DEBUGPY_LAUNCHER_PORT and DEBUGPY_LOG_DIR.
    cmdline = ["sudo", "-E"] if sudo else []
    cmdline += python
    cmdline += [launcher_path]
    env = {}

    arguments = dict(start_request.arguments)
    if not session.no_debug:
        _, arguments["port"] = servers.listener.getsockname()
        arguments["adapterAccessToken"] = adapter.access_token

    def on_launcher_connected(sock):
        listener.close()
        stream = messaging.JsonIOStream.from_socket(sock)
        Launcher(session, stream)

    try:
        listener = sockets.serve(
            "Launcher", on_launcher_connected, adapter_host, backlog=1
        )
    except Exception as exc:
        raise start_request.cant_handle(
            "{0} couldn't create listener socket for launcher: {1}", session, exc
        )

    try:
        launcher_host, launcher_port = listener.getsockname()
        launcher_addr = (
            launcher_port
            if launcher_host == "127.0.0.1"
            else f"{launcher_host}:{launcher_port}"
        )
        cmdline += [str(launcher_addr), "--"]
        cmdline += args

        if log.log_dir is not None:
            env[str("DEBUGPY_LOG_DIR")] = log.log_dir
        if log.stderr.levels != {"warning", "error"}:
            env[str("DEBUGPY_LOG_STDERR")] = str(" ".join(log.stderr.levels))

        if console == "internalConsole":
            log.info("{0} spawning launcher: {1!r}", session, cmdline)
            try:
                # If we are talking to the client over stdio, sys.stdin and sys.stdout
                # are redirected to avoid mangling the DAP message stream. Make sure
                # the launcher also respects that.
                subprocess.Popen(
                    cmdline,
                    cwd=cwd,
                    env=dict(list(os.environ.items()) + list(env.items())),
                    stdin=sys.stdin,
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                )
            except Exception as exc:
                raise start_request.cant_handle("Failed to spawn launcher: {0}", exc)
        else:
            log.info('{0} spawning launcher via "runInTerminal" request.', session)
            session.client.capabilities.require("supportsRunInTerminalRequest")
            kinds = {"integratedTerminal": "integrated", "externalTerminal": "external"}
            request_args = {
                "kind": kinds[console],
                "title": console_title,
                "args": cmdline,
                "env": env,
            }
            if cwd is not None:
                request_args["cwd"] = cwd
            if shell_expand_args:
                request_args["argsCanBeInterpretedByShell"] = True
            try:
                # It is unspecified whether this request receives a response immediately, or only
                # after the spawned command has completed running, so do not block waiting for it.
                session.client.channel.send_request("runInTerminal", request_args)
            except messaging.MessageHandlingError as exc:
                exc.propagate(start_request)

        # If using sudo, it might prompt for password, and launcher won't start running
        # until the user enters it, so don't apply timeout in that case.
        if not session.wait_for(
            lambda: session.launcher,
            timeout=(None if sudo else common.PROCESS_SPAWN_TIMEOUT),
        ):
            raise start_request.cant_handle("Timed out waiting for launcher to connect")

        try:
            session.launcher.channel.request(start_request.command, arguments)
        except messaging.MessageHandlingError as exc:
            exc.propagate(start_request)

        if not session.wait_for(
            lambda: session.launcher.pid is not None,
            timeout=common.PROCESS_SPAWN_TIMEOUT,
        ):
            raise start_request.cant_handle(
                'Timed out waiting for "process" event from launcher'
            )

        if session.no_debug:
            return

        # Wait for the first incoming connection regardless of the PID - it won't
        # necessarily match due to the use of stubs like py.exe or "conda run".
        conn = servers.wait_for_connection(
            session, lambda conn: True, timeout=common.PROCESS_SPAWN_TIMEOUT
        )
        if conn is None:
            raise start_request.cant_handle("Timed out waiting for debuggee to spawn")
        conn.attach_to_session(session)

    finally:
        listener.close()
