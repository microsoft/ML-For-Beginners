# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root
# for license information.

__all__ = ["main"]

import locale
import signal
import sys

# WARNING: debugpy and submodules must not be imported on top level in this module,
# and should be imported locally inside main() instead.


def main():
    from debugpy import launcher
    from debugpy.common import log
    from debugpy.launcher import debuggee

    log.to_file(prefix="debugpy.launcher")
    log.describe_environment("debugpy.launcher startup environment:")

    if sys.platform == "win32":
        # For windows, disable exceptions on Ctrl+C - we want to allow the debuggee
        # process to handle these, or not, as it sees fit. If the debuggee exits
        # on Ctrl+C, the launcher will also exit, so it doesn't need to observe
        # the signal directly.
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Everything before "--" is command line arguments for the launcher itself,
    # and everything after "--" is command line arguments for the debuggee.
    log.info("sys.argv before parsing: {0}", sys.argv)
    sep = sys.argv.index("--")
    launcher_argv = sys.argv[1:sep]
    sys.argv[:] = [sys.argv[0]] + sys.argv[sep + 1 :]
    log.info("sys.argv after patching: {0}", sys.argv)

    # The first argument specifies the host/port on which the adapter is waiting
    # for launcher to connect. It's either host:port, or just port.
    adapter = launcher_argv[0]
    host, sep, port = adapter.partition(":")
    if not sep:
        host = "127.0.0.1"
        port = adapter
    port = int(port)

    launcher.connect(host, port)
    launcher.channel.wait()

    if debuggee.process is not None:
        sys.exit(debuggee.process.returncode)


if __name__ == "__main__":
    # debugpy can also be invoked directly rather than via -m. In this case, the first
    # entry on sys.path is the one added automatically by Python for the directory
    # containing this file. This means that import debugpy will not work, since we need
    # the parent directory of debugpy/ to be in sys.path, rather than debugpy/launcher/.
    #
    # The other issue is that many other absolute imports will break, because they
    # will be resolved relative to debugpy/launcher/ - e.g. `import state` will then try
    # to import debugpy/launcher/state.py.
    #
    # To fix both, we need to replace the automatically added entry such that it points
    # at parent directory of debugpy/ instead of debugpy/launcher, import debugpy with that
    # in sys.path, and then remove the first entry entry altogether, so that it doesn't
    # affect any further imports we might do. For example, suppose the user did:
    #
    #   python /foo/bar/debugpy/launcher ...
    #
    # At the beginning of this script, sys.path will contain "/foo/bar/debugpy/launcher"
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

    main()
