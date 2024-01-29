"""A parent poller for unix."""
# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

try:
    import ctypes
except ImportError:
    ctypes = None  # type:ignore[assignment]
import os
import platform
import signal
import time
import warnings
from _thread import interrupt_main  # Py 3
from threading import Thread

from traitlets.log import get_logger


class ParentPollerUnix(Thread):
    """A Unix-specific daemon thread that terminates the program immediately
    when the parent process no longer exists.
    """

    def __init__(self):
        """Initialize the poller."""
        super().__init__()
        self.daemon = True

    def run(self):
        """Run the poller."""
        # We cannot use os.waitpid because it works only for child processes.
        from errno import EINTR

        while True:
            try:
                if os.getppid() == 1:
                    get_logger().warning("Parent appears to have exited, shutting down.")
                    os._exit(1)
                time.sleep(1.0)
            except OSError as e:
                if e.errno == EINTR:
                    continue
                raise


class ParentPollerWindows(Thread):
    """A Windows-specific daemon thread that listens for a special event that
    signals an interrupt and, optionally, terminates the program immediately
    when the parent process no longer exists.
    """

    def __init__(self, interrupt_handle=None, parent_handle=None):
        """Create the poller. At least one of the optional parameters must be
        provided.

        Parameters
        ----------
        interrupt_handle : HANDLE (int), optional
            If provided, the program will generate a Ctrl+C event when this
            handle is signaled.
        parent_handle : HANDLE (int), optional
            If provided, the program will terminate immediately when this
            handle is signaled.
        """
        assert interrupt_handle or parent_handle
        super().__init__()
        if ctypes is None:
            msg = "ParentPollerWindows requires ctypes"  # type:ignore[unreachable]
            raise ImportError(msg)
        self.daemon = True
        self.interrupt_handle = interrupt_handle
        self.parent_handle = parent_handle

    def run(self):
        """Run the poll loop. This method never returns."""
        try:
            from _winapi import INFINITE, WAIT_OBJECT_0  # type:ignore[attr-defined]
        except ImportError:
            from _subprocess import INFINITE, WAIT_OBJECT_0

        # Build the list of handle to listen on.
        handles = []
        if self.interrupt_handle:
            handles.append(self.interrupt_handle)
        if self.parent_handle:
            handles.append(self.parent_handle)
        arch = platform.architecture()[0]
        c_int = ctypes.c_int64 if arch.startswith("64") else ctypes.c_int

        # Listen forever.
        while True:
            result = ctypes.windll.kernel32.WaitForMultipleObjects(  # type:ignore[attr-defined]
                len(handles),  # nCount
                (c_int * len(handles))(*handles),  # lpHandles
                False,  # bWaitAll
                INFINITE,
            )  # dwMilliseconds

            if WAIT_OBJECT_0 <= result < len(handles):
                handle = handles[result - WAIT_OBJECT_0]

                if handle == self.interrupt_handle:
                    # check if signal handler is callable
                    # to avoid 'int not callable' error (Python issue #23395)
                    if callable(signal.getsignal(signal.SIGINT)):
                        interrupt_main()

                elif handle == self.parent_handle:
                    get_logger().warning("Parent appears to have exited, shutting down.")
                    os._exit(1)
            elif result < 0:
                # wait failed, just give up and stop polling.
                warnings.warn(
                    """Parent poll failed.  If the frontend dies,
                the kernel may be left running.  Please let us know
                about your system (bitness, Python, etc.) at
                ipython-dev@scipy.org""",
                    stacklevel=2,
                )
                return
