# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root
# for license information.

import argparse
import atexit
import codecs
import locale
import os
import sys

# WARNING: debugpy and submodules must not be imported on top level in this module,
# and should be imported locally inside main() instead.


def main(args):
    # If we're talking DAP over stdio, stderr is not guaranteed to be read from,
    # so disable it to avoid the pipe filling and locking up. This must be done
    # as early as possible, before the logging module starts writing to it.
    if args.port is None:
        sys.stderr = stderr = open(os.devnull, "w")
        atexit.register(stderr.close)

    from debugpy import adapter
    from debugpy.common import json, log, sockets
    from debugpy.adapter import clients, servers, sessions

    if args.for_server is not None:
        if os.name == "posix":
            # On POSIX, we need to leave the process group and its session, and then
            # daemonize properly by double-forking (first fork already happened when
            # this process was spawned).
            # NOTE: if process is already the session leader, then
            # setsid would fail with `operation not permitted`
            if os.getsid(os.getpid()) != os.getpid():
                os.setsid()
            if os.fork() != 0:
                sys.exit(0)

        for stdio in sys.stdin, sys.stdout, sys.stderr:
            if stdio is not None:
                stdio.close()

    if args.log_stderr:
        log.stderr.levels |= set(log.LEVELS)
    if args.log_dir is not None:
        log.log_dir = args.log_dir

    log.to_file(prefix="debugpy.adapter")
    log.describe_environment("debugpy.adapter startup environment:")

    servers.access_token = args.server_access_token
    if args.for_server is None:
        adapter.access_token = codecs.encode(os.urandom(32), "hex").decode("ascii")

    endpoints = {}
    try:
        client_host, client_port = clients.serve(args.host, args.port)
    except Exception as exc:
        if args.for_server is None:
            raise
        endpoints = {"error": "Can't listen for client connections: " + str(exc)}
    else:
        endpoints["client"] = {"host": client_host, "port": client_port}

    if args.for_server is not None:
        try:
            server_host, server_port = servers.serve()
        except Exception as exc:
            endpoints = {"error": "Can't listen for server connections: " + str(exc)}
        else:
            endpoints["server"] = {"host": server_host, "port": server_port}

        log.info(
            "Sending endpoints info to debug server at localhost:{0}:\n{1}",
            args.for_server,
            json.repr(endpoints),
        )

        try:
            sock = sockets.create_client()
            try:
                sock.settimeout(None)
                sock.connect(("127.0.0.1", args.for_server))
                sock_io = sock.makefile("wb", 0)
                try:
                    sock_io.write(json.dumps(endpoints).encode("utf-8"))
                finally:
                    sock_io.close()
            finally:
                sockets.close_socket(sock)
        except Exception:
            log.reraise_exception("Error sending endpoints info to debug server:")

        if "error" in endpoints:
            log.error("Couldn't set up endpoints; exiting.")
            sys.exit(1)

    listener_file = os.getenv("DEBUGPY_ADAPTER_ENDPOINTS")
    if listener_file is not None:
        log.info(
            "Writing endpoints info to {0!r}:\n{1}", listener_file, json.repr(endpoints)
        )

        def delete_listener_file():
            log.info("Listener ports closed; deleting {0!r}", listener_file)
            try:
                os.remove(listener_file)
            except Exception:
                log.swallow_exception(
                    "Failed to delete {0!r}", listener_file, level="warning"
                )

        try:
            with open(listener_file, "w") as f:
                atexit.register(delete_listener_file)
                print(json.dumps(endpoints), file=f)
        except Exception:
            log.reraise_exception("Error writing endpoints info to file:")

    if args.port is None:
        clients.Client("stdio")

    # These must be registered after the one above, to ensure that the listener sockets
    # are closed before the endpoint info file is deleted - this way, another process
    # can wait for the file to go away as a signal that the ports are no longer in use.
    atexit.register(servers.stop_serving)
    atexit.register(clients.stop_serving)

    servers.wait_until_disconnected()
    log.info("All debug servers disconnected; waiting for remaining sessions...")

    sessions.wait_until_ended()
    log.info("All debug sessions have ended; exiting.")


def _parse_argv(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--for-server", type=int, metavar="PORT", help=argparse.SUPPRESS
    )

    parser.add_argument(
        "--port",
        type=int,
        default=None,
        metavar="PORT",
        help="start the adapter in debugServer mode on the specified port",
    )

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        metavar="HOST",
        help="start the adapter in debugServer mode on the specified host",
    )

    parser.add_argument(
        "--access-token", type=str, help="access token expected from the server"
    )

    parser.add_argument(
        "--server-access-token", type=str, help="access token expected by the server"
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        metavar="DIR",
        help="enable logging and use DIR to save adapter logs",
    )

    parser.add_argument(
        "--log-stderr", action="store_true", help="enable logging to stderr"
    )

    args = parser.parse_args(argv[1:])

    if args.port is None:
        if args.log_stderr:
            parser.error("--log-stderr requires --port")
        if args.for_server is not None:
            parser.error("--for-server requires --port")

    return args


if __name__ == "__main__":
    # debugpy can also be invoked directly rather than via -m. In this case, the first
    # entry on sys.path is the one added automatically by Python for the directory
    # containing this file. This means that import debugpy will not work, since we need
    # the parent directory of debugpy/ to be in sys.path, rather than debugpy/adapter/.
    #
    # The other issue is that many other absolute imports will break, because they
    # will be resolved relative to debugpy/adapter/ - e.g. `import state` will then try
    # to import debugpy/adapter/state.py.
    #
    # To fix both, we need to replace the automatically added entry such that it points
    # at parent directory of debugpy/ instead of debugpy/adapter, import debugpy with that
    # in sys.path, and then remove the first entry entry altogether, so that it doesn't
    # affect any further imports we might do. For example, suppose the user did:
    #
    #   python /foo/bar/debugpy/adapter ...
    #
    # At the beginning of this script, sys.path will contain "/foo/bar/debugpy/adapter"
    # as the first entry. What we want is to replace it with "/foo/bar', then import
    # debugpy with that in effect, and then remove the replaced entry before any more
    # code runs. The imported debugpy module will remain in sys.modules, and thus all
    # future imports of it or its submodules will resolve accordingly.
    if "debugpy" not in sys.modules:
        # Do not use dirname() to walk up - this can be a relative path, e.g. ".".
        sys.path[0] = sys.path[0] + "/../../"
        __import__("debugpy")
        del sys.path[0]

    # Apply OS-global and user-specific locale settings.
    try:
        locale.setlocale(locale.LC_ALL, "")
    except Exception:
        # On POSIX, locale is set via environment variables, and this can fail if
        # those variables reference a non-existing locale. Ignore and continue using
        # the default "C" locale if so.
        pass

    main(_parse_argv(sys.argv))
