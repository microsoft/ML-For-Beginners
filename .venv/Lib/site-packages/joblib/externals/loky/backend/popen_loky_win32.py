import os
import sys
import msvcrt
import _winapi
from pickle import load
from multiprocessing import process, util
from multiprocessing.context import set_spawning_popen
from multiprocessing.popen_spawn_win32 import Popen as _Popen

from . import reduction, spawn


__all__ = ["Popen"]

#
#
#


def _path_eq(p1, p2):
    return p1 == p2 or os.path.normcase(p1) == os.path.normcase(p2)


WINENV = hasattr(sys, "_base_executable") and not _path_eq(
    sys.executable, sys._base_executable
)


def _close_handles(*handles):
    for handle in handles:
        _winapi.CloseHandle(handle)


#
# We define a Popen class similar to the one from subprocess, but
# whose constructor takes a process object as its argument.
#


class Popen(_Popen):
    """
    Start a subprocess to run the code of a process object.

    We differ from cpython implementation with the way we handle environment
    variables, in order to be able to modify then in the child processes before
    importing any library, in order to control the number of threads in C-level
    threadpools.

    We also use the loky preparation data, in particular to handle main_module
    inits and the loky resource tracker.
    """

    method = "loky"

    def __init__(self, process_obj):
        prep_data = spawn.get_preparation_data(
            process_obj._name, getattr(process_obj, "init_main_module", True)
        )

        # read end of pipe will be duplicated by the child process
        # -- see spawn_main() in spawn.py.
        #
        # bpo-33929: Previously, the read end of pipe was "stolen" by the child
        # process, but it leaked a handle if the child process had been
        # terminated before it could steal the handle from the parent process.
        rhandle, whandle = _winapi.CreatePipe(None, 0)
        wfd = msvcrt.open_osfhandle(whandle, 0)
        cmd = get_command_line(parent_pid=os.getpid(), pipe_handle=rhandle)

        python_exe = spawn.get_executable()

        # copy the environment variables to set in the child process
        child_env = {**os.environ, **process_obj.env}

        # bpo-35797: When running in a venv, we bypass the redirect
        # executor and launch our base Python.
        if WINENV and _path_eq(python_exe, sys.executable):
            cmd[0] = python_exe = sys._base_executable
            child_env["__PYVENV_LAUNCHER__"] = sys.executable

        cmd = " ".join(f'"{x}"' for x in cmd)

        with open(wfd, "wb") as to_child:
            # start process
            try:
                hp, ht, pid, _ = _winapi.CreateProcess(
                    python_exe,
                    cmd,
                    None,
                    None,
                    False,
                    0,
                    child_env,
                    None,
                    None,
                )
                _winapi.CloseHandle(ht)
            except BaseException:
                _winapi.CloseHandle(rhandle)
                raise

            # set attributes of self
            self.pid = pid
            self.returncode = None
            self._handle = hp
            self.sentinel = int(hp)
            self.finalizer = util.Finalize(
                self, _close_handles, (self.sentinel, int(rhandle))
            )

            # send information to child
            set_spawning_popen(self)
            try:
                reduction.dump(prep_data, to_child)
                reduction.dump(process_obj, to_child)
            finally:
                set_spawning_popen(None)


def get_command_line(pipe_handle, parent_pid, **kwds):
    """Returns prefix of command line used for spawning a child process."""
    if getattr(sys, "frozen", False):
        return [sys.executable, "--multiprocessing-fork", pipe_handle]
    else:
        prog = (
            "from joblib.externals.loky.backend.popen_loky_win32 import main; "
            f"main(pipe_handle={pipe_handle}, parent_pid={parent_pid})"
        )
        opts = util._args_from_interpreter_flags()
        return [
            spawn.get_executable(),
            *opts,
            "-c",
            prog,
            "--multiprocessing-fork",
        ]


def is_forking(argv):
    """Return whether commandline indicates we are forking."""
    if len(argv) >= 2 and argv[1] == "--multiprocessing-fork":
        return True
    else:
        return False


def main(pipe_handle, parent_pid=None):
    """Run code specified by data received over pipe."""
    assert is_forking(sys.argv), "Not forking"

    if parent_pid is not None:
        source_process = _winapi.OpenProcess(
            _winapi.SYNCHRONIZE | _winapi.PROCESS_DUP_HANDLE, False, parent_pid
        )
    else:
        source_process = None
    new_handle = reduction.duplicate(
        pipe_handle, source_process=source_process
    )
    fd = msvcrt.open_osfhandle(new_handle, os.O_RDONLY)
    parent_sentinel = source_process

    with os.fdopen(fd, "rb", closefd=True) as from_parent:
        process.current_process()._inheriting = True
        try:
            preparation_data = load(from_parent)
            spawn.prepare(preparation_data, parent_sentinel)
            self = load(from_parent)
        finally:
            del process.current_process()._inheriting

    exitcode = self._bootstrap(parent_sentinel)
    sys.exit(exitcode)
