#
# Copyright 2011 Facebook
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

"""Utilities for working with multiple processes, including both forking
the server into multiple processes and managing subprocesses.
"""

import os
import multiprocessing
import signal
import subprocess
import sys
import time

from binascii import hexlify

from tornado.concurrent import (
    Future,
    future_set_result_unless_cancelled,
    future_set_exception_unless_cancelled,
)
from tornado import ioloop
from tornado.iostream import PipeIOStream
from tornado.log import gen_log

import typing
from typing import Optional, Any, Callable

if typing.TYPE_CHECKING:
    from typing import List  # noqa: F401

# Re-export this exception for convenience.
CalledProcessError = subprocess.CalledProcessError


def cpu_count() -> int:
    """Returns the number of processors on this machine."""
    if multiprocessing is None:
        return 1
    try:
        return multiprocessing.cpu_count()
    except NotImplementedError:
        pass
    try:
        return os.sysconf("SC_NPROCESSORS_CONF")  # type: ignore
    except (AttributeError, ValueError):
        pass
    gen_log.error("Could not detect number of processors; assuming 1")
    return 1


def _reseed_random() -> None:
    if "random" not in sys.modules:
        return
    import random

    # If os.urandom is available, this method does the same thing as
    # random.seed (at least as of python 2.6).  If os.urandom is not
    # available, we mix in the pid in addition to a timestamp.
    try:
        seed = int(hexlify(os.urandom(16)), 16)
    except NotImplementedError:
        seed = int(time.time() * 1000) ^ os.getpid()
    random.seed(seed)


_task_id = None


def fork_processes(
    num_processes: Optional[int], max_restarts: Optional[int] = None
) -> int:
    """Starts multiple worker processes.

    If ``num_processes`` is None or <= 0, we detect the number of cores
    available on this machine and fork that number of child
    processes. If ``num_processes`` is given and > 0, we fork that
    specific number of sub-processes.

    Since we use processes and not threads, there is no shared memory
    between any server code.

    Note that multiple processes are not compatible with the autoreload
    module (or the ``autoreload=True`` option to `tornado.web.Application`
    which defaults to True when ``debug=True``).
    When using multiple processes, no IOLoops can be created or
    referenced until after the call to ``fork_processes``.

    In each child process, ``fork_processes`` returns its *task id*, a
    number between 0 and ``num_processes``.  Processes that exit
    abnormally (due to a signal or non-zero exit status) are restarted
    with the same id (up to ``max_restarts`` times).  In the parent
    process, ``fork_processes`` calls ``sys.exit(0)`` after all child
    processes have exited normally.

    max_restarts defaults to 100.

    Availability: Unix
    """
    if sys.platform == "win32":
        # The exact form of this condition matters to mypy; it understands
        # if but not assert in this context.
        raise Exception("fork not available on windows")
    if max_restarts is None:
        max_restarts = 100

    global _task_id
    assert _task_id is None
    if num_processes is None or num_processes <= 0:
        num_processes = cpu_count()
    gen_log.info("Starting %d processes", num_processes)
    children = {}

    def start_child(i: int) -> Optional[int]:
        pid = os.fork()
        if pid == 0:
            # child process
            _reseed_random()
            global _task_id
            _task_id = i
            return i
        else:
            children[pid] = i
            return None

    for i in range(num_processes):
        id = start_child(i)
        if id is not None:
            return id
    num_restarts = 0
    while children:
        pid, status = os.wait()
        if pid not in children:
            continue
        id = children.pop(pid)
        if os.WIFSIGNALED(status):
            gen_log.warning(
                "child %d (pid %d) killed by signal %d, restarting",
                id,
                pid,
                os.WTERMSIG(status),
            )
        elif os.WEXITSTATUS(status) != 0:
            gen_log.warning(
                "child %d (pid %d) exited with status %d, restarting",
                id,
                pid,
                os.WEXITSTATUS(status),
            )
        else:
            gen_log.info("child %d (pid %d) exited normally", id, pid)
            continue
        num_restarts += 1
        if num_restarts > max_restarts:
            raise RuntimeError("Too many child restarts, giving up")
        new_id = start_child(id)
        if new_id is not None:
            return new_id
    # All child processes exited cleanly, so exit the master process
    # instead of just returning to right after the call to
    # fork_processes (which will probably just start up another IOLoop
    # unless the caller checks the return value).
    sys.exit(0)


def task_id() -> Optional[int]:
    """Returns the current task id, if any.

    Returns None if this process was not created by `fork_processes`.
    """
    global _task_id
    return _task_id


class Subprocess(object):
    """Wraps ``subprocess.Popen`` with IOStream support.

    The constructor is the same as ``subprocess.Popen`` with the following
    additions:

    * ``stdin``, ``stdout``, and ``stderr`` may have the value
      ``tornado.process.Subprocess.STREAM``, which will make the corresponding
      attribute of the resulting Subprocess a `.PipeIOStream`. If this option
      is used, the caller is responsible for closing the streams when done
      with them.

    The ``Subprocess.STREAM`` option and the ``set_exit_callback`` and
    ``wait_for_exit`` methods do not work on Windows. There is
    therefore no reason to use this class instead of
    ``subprocess.Popen`` on that platform.

    .. versionchanged:: 5.0
       The ``io_loop`` argument (deprecated since version 4.1) has been removed.

    """

    STREAM = object()

    _initialized = False
    _waiting = {}  # type: ignore
    _old_sigchld = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.io_loop = ioloop.IOLoop.current()
        # All FDs we create should be closed on error; those in to_close
        # should be closed in the parent process on success.
        pipe_fds = []  # type: List[int]
        to_close = []  # type: List[int]
        if kwargs.get("stdin") is Subprocess.STREAM:
            in_r, in_w = os.pipe()
            kwargs["stdin"] = in_r
            pipe_fds.extend((in_r, in_w))
            to_close.append(in_r)
            self.stdin = PipeIOStream(in_w)
        if kwargs.get("stdout") is Subprocess.STREAM:
            out_r, out_w = os.pipe()
            kwargs["stdout"] = out_w
            pipe_fds.extend((out_r, out_w))
            to_close.append(out_w)
            self.stdout = PipeIOStream(out_r)
        if kwargs.get("stderr") is Subprocess.STREAM:
            err_r, err_w = os.pipe()
            kwargs["stderr"] = err_w
            pipe_fds.extend((err_r, err_w))
            to_close.append(err_w)
            self.stderr = PipeIOStream(err_r)
        try:
            self.proc = subprocess.Popen(*args, **kwargs)
        except:
            for fd in pipe_fds:
                os.close(fd)
            raise
        for fd in to_close:
            os.close(fd)
        self.pid = self.proc.pid
        for attr in ["stdin", "stdout", "stderr"]:
            if not hasattr(self, attr):  # don't clobber streams set above
                setattr(self, attr, getattr(self.proc, attr))
        self._exit_callback = None  # type: Optional[Callable[[int], None]]
        self.returncode = None  # type: Optional[int]

    def set_exit_callback(self, callback: Callable[[int], None]) -> None:
        """Runs ``callback`` when this process exits.

        The callback takes one argument, the return code of the process.

        This method uses a ``SIGCHLD`` handler, which is a global setting
        and may conflict if you have other libraries trying to handle the
        same signal.  If you are using more than one ``IOLoop`` it may
        be necessary to call `Subprocess.initialize` first to designate
        one ``IOLoop`` to run the signal handlers.

        In many cases a close callback on the stdout or stderr streams
        can be used as an alternative to an exit callback if the
        signal handler is causing a problem.

        Availability: Unix
        """
        self._exit_callback = callback
        Subprocess.initialize()
        Subprocess._waiting[self.pid] = self
        Subprocess._try_cleanup_process(self.pid)

    def wait_for_exit(self, raise_error: bool = True) -> "Future[int]":
        """Returns a `.Future` which resolves when the process exits.

        Usage::

            ret = yield proc.wait_for_exit()

        This is a coroutine-friendly alternative to `set_exit_callback`
        (and a replacement for the blocking `subprocess.Popen.wait`).

        By default, raises `subprocess.CalledProcessError` if the process
        has a non-zero exit status. Use ``wait_for_exit(raise_error=False)``
        to suppress this behavior and return the exit status without raising.

        .. versionadded:: 4.2

        Availability: Unix
        """
        future = Future()  # type: Future[int]

        def callback(ret: int) -> None:
            if ret != 0 and raise_error:
                # Unfortunately we don't have the original args any more.
                future_set_exception_unless_cancelled(
                    future, CalledProcessError(ret, "unknown")
                )
            else:
                future_set_result_unless_cancelled(future, ret)

        self.set_exit_callback(callback)
        return future

    @classmethod
    def initialize(cls) -> None:
        """Initializes the ``SIGCHLD`` handler.

        The signal handler is run on an `.IOLoop` to avoid locking issues.
        Note that the `.IOLoop` used for signal handling need not be the
        same one used by individual Subprocess objects (as long as the
        ``IOLoops`` are each running in separate threads).

        .. versionchanged:: 5.0
           The ``io_loop`` argument (deprecated since version 4.1) has been
           removed.

        Availability: Unix
        """
        if cls._initialized:
            return
        io_loop = ioloop.IOLoop.current()
        cls._old_sigchld = signal.signal(
            signal.SIGCHLD,
            lambda sig, frame: io_loop.add_callback_from_signal(cls._cleanup),
        )
        cls._initialized = True

    @classmethod
    def uninitialize(cls) -> None:
        """Removes the ``SIGCHLD`` handler."""
        if not cls._initialized:
            return
        signal.signal(signal.SIGCHLD, cls._old_sigchld)
        cls._initialized = False

    @classmethod
    def _cleanup(cls) -> None:
        for pid in list(cls._waiting.keys()):  # make a copy
            cls._try_cleanup_process(pid)

    @classmethod
    def _try_cleanup_process(cls, pid: int) -> None:
        try:
            ret_pid, status = os.waitpid(pid, os.WNOHANG)  # type: ignore
        except ChildProcessError:
            return
        if ret_pid == 0:
            return
        assert ret_pid == pid
        subproc = cls._waiting.pop(pid)
        subproc.io_loop.add_callback_from_signal(subproc._set_returncode, status)

    def _set_returncode(self, status: int) -> None:
        if sys.platform == "win32":
            self.returncode = -1
        else:
            if os.WIFSIGNALED(status):
                self.returncode = -os.WTERMSIG(status)
            else:
                assert os.WIFEXITED(status)
                self.returncode = os.WEXITSTATUS(status)
        # We've taken over wait() duty from the subprocess.Popen
        # object. If we don't inform it of the process's return code,
        # it will log a warning at destruction in python 3.6+.
        self.proc.returncode = self.returncode
        if self._exit_callback:
            callback = self._exit_callback
            self._exit_callback = None
            callback(self.returncode)
