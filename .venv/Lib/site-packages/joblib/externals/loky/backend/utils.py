import os
import sys
import time
import errno
import signal
import warnings
import subprocess
import traceback

try:
    import psutil
except ImportError:
    psutil = None


def kill_process_tree(process, use_psutil=True):
    """Terminate process and its descendants with SIGKILL"""
    if use_psutil and psutil is not None:
        _kill_process_tree_with_psutil(process)
    else:
        _kill_process_tree_without_psutil(process)


def recursive_terminate(process, use_psutil=True):
    warnings.warn(
        "recursive_terminate is deprecated in loky 3.2, use kill_process_tree"
        "instead",
        DeprecationWarning,
    )
    kill_process_tree(process, use_psutil=use_psutil)


def _kill_process_tree_with_psutil(process):
    try:
        descendants = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        return

    # Kill the descendants in reverse order to avoid killing the parents before
    # the descendant in cases where there are more processes nested.
    for descendant in descendants[::-1]:
        try:
            descendant.kill()
        except psutil.NoSuchProcess:
            pass

    try:
        psutil.Process(process.pid).kill()
    except psutil.NoSuchProcess:
        pass
    process.join()


def _kill_process_tree_without_psutil(process):
    """Terminate a process and its descendants."""
    try:
        if sys.platform == "win32":
            _windows_taskkill_process_tree(process.pid)
        else:
            _posix_recursive_kill(process.pid)
    except Exception:  # pragma: no cover
        details = traceback.format_exc()
        warnings.warn(
            "Failed to kill subprocesses on this platform. Please install"
            "psutil: https://github.com/giampaolo/psutil\n"
            f"Details:\n{details}"
        )
        # In case we cannot introspect or kill the descendants, we fall back to
        # only killing the main process.
        #
        # Note: on Windows, process.kill() is an alias for process.terminate()
        # which in turns calls the Win32 API function TerminateProcess().
        process.kill()
    process.join()


def _windows_taskkill_process_tree(pid):
    # On windows, the taskkill function with option `/T` terminate a given
    # process pid and its children.
    try:
        subprocess.check_output(
            ["taskkill", "/F", "/T", "/PID", str(pid)], stderr=None
        )
    except subprocess.CalledProcessError as e:
        # In Windows, taskkill returns 128, 255 for no process found.
        if e.returncode not in [128, 255]:
            # Let's raise to let the caller log the error details in a
            # warning and only kill the root process.
            raise  # pragma: no cover


def _kill(pid):
    # Not all systems (e.g. Windows) have a SIGKILL, but the C specification
    # mandates a SIGTERM signal. While Windows is handled specifically above,
    # let's try to be safe for other hypothetic platforms that only have
    # SIGTERM without SIGKILL.
    kill_signal = getattr(signal, "SIGKILL", signal.SIGTERM)
    try:
        os.kill(pid, kill_signal)
    except OSError as e:
        # if OSError is raised with [Errno 3] no such process, the process
        # is already terminated, else, raise the error and let the top
        # level function raise a warning and retry to kill the process.
        if e.errno != errno.ESRCH:
            raise  # pragma: no cover


def _posix_recursive_kill(pid):
    """Recursively kill the descendants of a process before killing it."""
    try:
        children_pids = subprocess.check_output(
            ["pgrep", "-P", str(pid)], stderr=None, text=True
        )
    except subprocess.CalledProcessError as e:
        # `ps` returns 1 when no child process has been found
        if e.returncode == 1:
            children_pids = ""
        else:
            raise  # pragma: no cover

    # Decode the result, split the cpid and remove the trailing line
    for cpid in children_pids.splitlines():
        cpid = int(cpid)
        _posix_recursive_kill(cpid)

    _kill(pid)


def get_exitcodes_terminated_worker(processes):
    """Return a formatted string with the exitcodes of terminated workers.

    If necessary, wait (up to .25s) for the system to correctly set the
    exitcode of one terminated worker.
    """
    patience = 5

    # Catch the exitcode of the terminated workers. There should at least be
    # one. If not, wait a bit for the system to correctly set the exitcode of
    # the terminated worker.
    exitcodes = [
        p.exitcode for p in list(processes.values()) if p.exitcode is not None
    ]
    while not exitcodes and patience > 0:
        patience -= 1
        exitcodes = [
            p.exitcode
            for p in list(processes.values())
            if p.exitcode is not None
        ]
        time.sleep(0.05)

    return _format_exitcodes(exitcodes)


def _format_exitcodes(exitcodes):
    """Format a list of exit code with names of the signals if possible"""
    str_exitcodes = [
        f"{_get_exitcode_name(e)}({e})" for e in exitcodes if e is not None
    ]
    return "{" + ", ".join(str_exitcodes) + "}"


def _get_exitcode_name(exitcode):
    if sys.platform == "win32":
        # The exitcode are unreliable  on windows (see bpo-31863).
        # For this case, return UNKNOWN
        return "UNKNOWN"

    if exitcode < 0:
        try:
            import signal

            return signal.Signals(-exitcode).name
        except ValueError:
            return "UNKNOWN"
    elif exitcode != 255:
        # The exitcode are unreliable on forkserver were 255 is always returned
        # (see bpo-30589). For this case, return UNKNOWN
        return "EXIT"

    return "UNKNOWN"
