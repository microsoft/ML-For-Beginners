"""Utilities for launching kernels"""
# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
import os
import sys
import warnings
from subprocess import PIPE, Popen
from typing import Any, Dict, List, Optional

from traitlets.log import get_logger


def launch_kernel(
    cmd: List[str],
    stdin: Optional[int] = None,
    stdout: Optional[int] = None,
    stderr: Optional[int] = None,
    env: Optional[Dict[str, str]] = None,
    independent: bool = False,
    cwd: Optional[str] = None,
    **kw: Any,
) -> Popen:
    """Launches a localhost kernel, binding to the specified ports.

    Parameters
    ----------
    cmd : Popen list,
        A string of Python code that imports and executes a kernel entry point.

    stdin, stdout, stderr : optional (default None)
        Standards streams, as defined in subprocess.Popen.

    env: dict, optional
        Environment variables passed to the kernel

    independent : bool, optional (default False)
        If set, the kernel process is guaranteed to survive if this process
        dies. If not set, an effort is made to ensure that the kernel is killed
        when this process dies. Note that in this case it is still good practice
        to kill kernels manually before exiting.

    cwd : path, optional
        The working dir of the kernel process (default: cwd of this process).

    **kw: optional
        Additional arguments for Popen

    Returns
    -------

    Popen instance for the kernel subprocess
    """

    # Popen will fail (sometimes with a deadlock) if stdin, stdout, and stderr
    # are invalid. Unfortunately, there is in general no way to detect whether
    # they are valid.  The following two blocks redirect them to (temporary)
    # pipes in certain important cases.

    # If this process has been backgrounded, our stdin is invalid. Since there
    # is no compelling reason for the kernel to inherit our stdin anyway, we'll
    # place this one safe and always redirect.
    redirect_in = True
    _stdin = PIPE if stdin is None else stdin

    # If this process in running on pythonw, we know that stdin, stdout, and
    # stderr are all invalid.
    redirect_out = sys.executable.endswith("pythonw.exe")
    if redirect_out:
        blackhole = open(os.devnull, "w")  # noqa
        _stdout = blackhole if stdout is None else stdout
        _stderr = blackhole if stderr is None else stderr
    else:
        _stdout, _stderr = stdout, stderr

    env = env if (env is not None) else os.environ.copy()

    kwargs = kw.copy()
    main_args = {
        "stdin": _stdin,
        "stdout": _stdout,
        "stderr": _stderr,
        "cwd": cwd,
        "env": env,
    }
    kwargs.update(main_args)

    # Spawn a kernel.
    if sys.platform == "win32":
        if cwd:
            kwargs["cwd"] = cwd

        from .win_interrupt import create_interrupt_event

        # Create a Win32 event for interrupting the kernel
        # and store it in an environment variable.
        interrupt_event = create_interrupt_event()
        env["JPY_INTERRUPT_EVENT"] = str(interrupt_event)
        # deprecated old env name:
        env["IPY_INTERRUPT_EVENT"] = env["JPY_INTERRUPT_EVENT"]

        try:
            from _winapi import (
                CREATE_NEW_PROCESS_GROUP,
                DUPLICATE_SAME_ACCESS,
                DuplicateHandle,
                GetCurrentProcess,
            )
        except:  # noqa
            from _subprocess import (
                CREATE_NEW_PROCESS_GROUP,
                DUPLICATE_SAME_ACCESS,
                DuplicateHandle,
                GetCurrentProcess,
            )

        # create a handle on the parent to be inherited
        if independent:
            kwargs["creationflags"] = CREATE_NEW_PROCESS_GROUP
        else:
            pid = GetCurrentProcess()
            handle = DuplicateHandle(
                pid,
                pid,
                pid,
                0,
                True,
                DUPLICATE_SAME_ACCESS,  # Inheritable by new processes.
            )
            env["JPY_PARENT_PID"] = str(int(handle))

        # Prevent creating new console window on pythonw
        if redirect_out:
            kwargs["creationflags"] = (
                kwargs.setdefault("creationflags", 0) | 0x08000000
            )  # CREATE_NO_WINDOW

        # Avoid closing the above parent and interrupt handles.
        # close_fds is True by default on Python >=3.7
        # or when no stream is captured on Python <3.7
        # (we always capture stdin, so this is already False by default on <3.7)
        kwargs["close_fds"] = False
    else:
        # Create a new session.
        # This makes it easier to interrupt the kernel,
        # because we want to interrupt the whole process group.
        # We don't use setpgrp, which is known to cause problems for kernels starting
        # certain interactive subprocesses, such as bash -i.
        kwargs["start_new_session"] = True
        if not independent:
            env["JPY_PARENT_PID"] = str(os.getpid())

    try:
        # Allow to use ~/ in the command or its arguments
        cmd = [os.path.expanduser(s) for s in cmd]
        proc = Popen(cmd, **kwargs)  # noqa
    except Exception as ex:
        try:
            msg = "Failed to run command:\n{}\n    PATH={!r}\n    with kwargs:\n{!r}\n"
            # exclude environment variables,
            # which may contain access tokens and the like.
            without_env = {key: value for key, value in kwargs.items() if key != "env"}
            msg = msg.format(cmd, env.get("PATH", os.defpath), without_env)
            get_logger().error(msg)
        except Exception as ex2:  # Don't let a formatting/logger issue lead to the wrong exception
            warnings.warn(f"Failed to run command: '{cmd}' due to exception: {ex}", stacklevel=2)
            warnings.warn(
                f"The following exception occurred handling the previous failure: {ex2}",
                stacklevel=2,
            )
        raise ex

    if sys.platform == "win32":
        # Attach the interrupt event to the Popen object so it can be used later.
        proc.win32_interrupt_event = interrupt_event

    # Clean up pipes created to work around Popen bug.
    if redirect_in and stdin is None:
        assert proc.stdin is not None
        proc.stdin.close()

    return proc


__all__ = [
    "launch_kernel",
]
