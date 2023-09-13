"""Connection file-related utilities for the kernel
"""
# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

import json
import sys
from subprocess import PIPE, Popen
from typing import Any, Dict

import jupyter_client
from jupyter_client import write_connection_file


def get_connection_file(app=None):
    """Return the path to the connection file of an app

    Parameters
    ----------
    app : IPKernelApp instance [optional]
        If unspecified, the currently running app will be used
    """
    from traitlets.utils import filefind

    if app is None:
        from ipykernel.kernelapp import IPKernelApp

        if not IPKernelApp.initialized():
            msg = "app not specified, and not in a running Kernel"
            raise RuntimeError(msg)

        app = IPKernelApp.instance()
    return filefind(app.connection_file, [".", app.connection_dir])


def _find_connection_file(connection_file):
    """Return the absolute path for a connection file

    - If nothing specified, return current Kernel's connection file
    - Otherwise, call jupyter_client.find_connection_file
    """
    if connection_file is None:
        # get connection file from current kernel
        return get_connection_file()
    else:
        return jupyter_client.find_connection_file(connection_file)


def get_connection_info(connection_file=None, unpack=False):
    """Return the connection information for the current Kernel.

    Parameters
    ----------
    connection_file : str [optional]
        The connection file to be used. Can be given by absolute path, or
        IPython will search in the security directory.
        If run from IPython,

        If unspecified, the connection file for the currently running
        IPython Kernel will be used, which is only allowed from inside a kernel.

    unpack : bool [default: False]
        if True, return the unpacked dict, otherwise just the string contents
        of the file.

    Returns
    -------
    The connection dictionary of the current kernel, as string or dict,
    depending on `unpack`.
    """
    cf = _find_connection_file(connection_file)

    with open(cf) as f:
        info_str = f.read()

    if unpack:
        info = json.loads(info_str)
        # ensure key is bytes:
        info["key"] = info.get("key", "").encode()
        return info

    return info_str


def connect_qtconsole(connection_file=None, argv=None):
    """Connect a qtconsole to the current kernel.

    This is useful for connecting a second qtconsole to a kernel, or to a
    local notebook.

    Parameters
    ----------
    connection_file : str [optional]
        The connection file to be used. Can be given by absolute path, or
        IPython will search in the security directory.
        If run from IPython,

        If unspecified, the connection file for the currently running
        IPython Kernel will be used, which is only allowed from inside a kernel.

    argv : list [optional]
        Any extra args to be passed to the console.

    Returns
    -------
    :class:`subprocess.Popen` instance running the qtconsole frontend
    """
    argv = [] if argv is None else argv

    cf = _find_connection_file(connection_file)

    cmd = ";".join(["from qtconsole import qtconsoleapp", "qtconsoleapp.main()"])

    kwargs: Dict[str, Any] = {}
    # Launch the Qt console in a separate session & process group, so
    # interrupting the kernel doesn't kill it.
    kwargs["start_new_session"] = True

    return Popen(
        [sys.executable, "-c", cmd, "--existing", cf, *argv],  # noqa
        stdout=PIPE,
        stderr=PIPE,
        close_fds=(sys.platform != "win32"),
        **kwargs,
    )


__all__ = [
    "write_connection_file",
    "get_connection_file",
    "get_connection_info",
    "connect_qtconsole",
]
