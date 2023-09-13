# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root
# for license information.

import atexit
import ctypes
import os
import signal
import struct
import subprocess
import sys
import threading

from debugpy import launcher
from debugpy.common import log, messaging
from debugpy.launcher import output

if sys.platform == "win32":
    from debugpy.launcher import winapi


process = None
"""subprocess.Popen instance for the debuggee process."""

job_handle = None
"""On Windows, the handle for the job object to which the debuggee is assigned."""

wait_on_exit_predicates = []
"""List of functions that determine whether to pause after debuggee process exits.

Every function is invoked with exit code as the argument. If any of the functions
returns True, the launcher pauses and waits for user input before exiting.
"""


def describe():
    return f"Debuggee[PID={process.pid}]"


def spawn(process_name, cmdline, env, redirect_output):
    log.info(
        "Spawning debuggee process:\n\n"
        "Command line: {0!r}\n\n"
        "Environment variables: {1!r}\n\n",
        cmdline,
        env,
    )

    close_fds = set()
    try:
        if redirect_output:
            # subprocess.PIPE behavior can vary substantially depending on Python version
            # and platform; using our own pipes keeps it simple, predictable, and fast.
            stdout_r, stdout_w = os.pipe()
            stderr_r, stderr_w = os.pipe()
            close_fds |= {stdout_r, stdout_w, stderr_r, stderr_w}
            kwargs = dict(stdout=stdout_w, stderr=stderr_w)
        else:
            kwargs = {}

        if sys.platform != "win32":

            def preexec_fn():
                try:
                    # Start the debuggee in a new process group, so that the launcher can
                    # kill the entire process tree later.
                    os.setpgrp()

                    # Make the new process group the foreground group in its session, so
                    # that it can interact with the terminal. The debuggee will receive
                    # SIGTTOU when tcsetpgrp() is called, and must ignore it.
                    old_handler = signal.signal(signal.SIGTTOU, signal.SIG_IGN)
                    try:
                        tty = os.open("/dev/tty", os.O_RDWR)
                        try:
                            os.tcsetpgrp(tty, os.getpgrp())
                        finally:
                            os.close(tty)
                    finally:
                        signal.signal(signal.SIGTTOU, old_handler)
                except Exception:
                    # Not an error - /dev/tty doesn't work when there's no terminal.
                    log.swallow_exception(
                        "Failed to set up process group", level="info"
                    )

            kwargs.update(preexec_fn=preexec_fn)

        try:
            global process
            process = subprocess.Popen(cmdline, env=env, bufsize=0, **kwargs)
        except Exception as exc:
            raise messaging.MessageHandlingError(
                "Couldn't spawn debuggee: {0}\n\nCommand line:{1!r}".format(
                    exc, cmdline
                )
            )

        log.info("Spawned {0}.", describe())

        if sys.platform == "win32":
            # Assign the debuggee to a new job object, so that the launcher can kill
            # the entire process tree later.
            try:
                global job_handle
                job_handle = winapi.kernel32.CreateJobObjectA(None, None)

                job_info = winapi.JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
                job_info_size = winapi.DWORD(ctypes.sizeof(job_info))
                winapi.kernel32.QueryInformationJobObject(
                    job_handle,
                    winapi.JobObjectExtendedLimitInformation,
                    ctypes.pointer(job_info),
                    job_info_size,
                    ctypes.pointer(job_info_size),
                )

                job_info.BasicLimitInformation.LimitFlags |= (
                    # Ensure that the job will be terminated by the OS once the
                    # launcher exits, even if it doesn't terminate the job explicitly.
                    winapi.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                    |
                    # Allow the debuggee to create its own jobs unrelated to ours.
                    winapi.JOB_OBJECT_LIMIT_BREAKAWAY_OK
                )
                winapi.kernel32.SetInformationJobObject(
                    job_handle,
                    winapi.JobObjectExtendedLimitInformation,
                    ctypes.pointer(job_info),
                    job_info_size,
                )

                process_handle = winapi.kernel32.OpenProcess(
                    winapi.PROCESS_TERMINATE | winapi.PROCESS_SET_QUOTA,
                    False,
                    process.pid,
                )

                winapi.kernel32.AssignProcessToJobObject(job_handle, process_handle)

            except Exception:
                log.swallow_exception("Failed to set up job object", level="warning")

        atexit.register(kill)

        launcher.channel.send_event(
            "process",
            {
                "startMethod": "launch",
                "isLocalProcess": True,
                "systemProcessId": process.pid,
                "name": process_name,
                "pointerSize": struct.calcsize("P") * 8,
            },
        )

        if redirect_output:
            for category, fd, tee in [
                ("stdout", stdout_r, sys.stdout),
                ("stderr", stderr_r, sys.stderr),
            ]:
                output.CaptureOutput(describe(), category, fd, tee)
                close_fds.remove(fd)

        wait_thread = threading.Thread(target=wait_for_exit, name="wait_for_exit()")
        wait_thread.daemon = True
        wait_thread.start()

    finally:
        for fd in close_fds:
            try:
                os.close(fd)
            except Exception:
                log.swallow_exception(level="warning")


def kill():
    if process is None:
        return

    try:
        if process.poll() is None:
            log.info("Killing {0}", describe())
            # Clean up the process tree
            if sys.platform == "win32":
                # On Windows, kill the job object.
                winapi.kernel32.TerminateJobObject(job_handle, 0)
            else:
                # On POSIX, kill the debuggee's process group.
                os.killpg(process.pid, signal.SIGKILL)
    except Exception:
        log.swallow_exception("Failed to kill {0}", describe())


def wait_for_exit():
    try:
        code = process.wait()
        if sys.platform != "win32" and code < 0:
            # On POSIX, if the process was terminated by a signal, Popen will use
            # a negative returncode to indicate that - but the actual exit code of
            # the process is always an unsigned number, and can be determined by
            # taking the lowest 8 bits of that negative returncode.
            code &= 0xFF
    except Exception:
        log.swallow_exception("Couldn't determine process exit code")
        code = -1

    log.info("{0} exited with code {1}", describe(), code)
    output.wait_for_remaining_output()

    # Determine whether we should wait or not before sending "exited", so that any
    # follow-up "terminate" requests don't affect the predicates.
    should_wait = any(pred(code) for pred in wait_on_exit_predicates)

    try:
        launcher.channel.send_event("exited", {"exitCode": code})
    except Exception:
        pass

    if should_wait:
        _wait_for_user_input()

    try:
        launcher.channel.send_event("terminated")
    except Exception:
        pass


def _wait_for_user_input():
    if sys.stdout and sys.stdin and sys.stdin.isatty():
        from debugpy.common import log

        try:
            import msvcrt
        except ImportError:
            can_getch = False
        else:
            can_getch = True

        if can_getch:
            log.debug("msvcrt available - waiting for user input via getch()")
            sys.stdout.write("Press any key to continue . . . ")
            sys.stdout.flush()
            msvcrt.getch()
        else:
            log.debug("msvcrt not available - waiting for user input via read()")
            sys.stdout.write("Press Enter to continue . . . ")
            sys.stdout.flush()
            sys.stdin.read(1)
