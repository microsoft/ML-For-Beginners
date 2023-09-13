"""Win32 compatibility utilities."""

# -----------------------------------------------------------------------------
# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.
# -----------------------------------------------------------------------------

import os
from typing import Any, Callable, Optional


class allow_interrupt:
    """Utility for fixing CTRL-C events on Windows.

    On Windows, the Python interpreter intercepts CTRL-C events in order to
    translate them into ``KeyboardInterrupt`` exceptions.  It (presumably)
    does this by setting a flag in its "console control handler" and
    checking it later at a convenient location in the interpreter.

    However, when the Python interpreter is blocked waiting for the ZMQ
    poll operation to complete, it must wait for ZMQ's ``select()``
    operation to complete before translating the CTRL-C event into the
    ``KeyboardInterrupt`` exception.

    The only way to fix this seems to be to add our own "console control
    handler" and perform some application-defined operation that will
    unblock the ZMQ polling operation in order to force ZMQ to pass control
    back to the Python interpreter.

    This context manager performs all that Windows-y stuff, providing you
    with a hook that is called when a CTRL-C event is intercepted.  This
    hook allows you to unblock your ZMQ poll operation immediately, which
    will then result in the expected ``KeyboardInterrupt`` exception.

    Without this context manager, your ZMQ-based application will not
    respond normally to CTRL-C events on Windows.  If a CTRL-C event occurs
    while blocked on ZMQ socket polling, the translation to a
    ``KeyboardInterrupt`` exception will be delayed until the I/O completes
    and control returns to the Python interpreter (this may never happen if
    you use an infinite timeout).

    A no-op implementation is provided on non-Win32 systems to avoid the
    application from having to conditionally use it.

    Example usage:

    .. sourcecode:: python

       def stop_my_application():
           # ...

       with allow_interrupt(stop_my_application):
           # main polling loop.

    In a typical ZMQ application, you would use the "self pipe trick" to
    send message to a ``PAIR`` socket in order to interrupt your blocking
    socket polling operation.

    In a Tornado event loop, you can use the ``IOLoop.stop`` method to
    unblock your I/O loop.
    """

    def __init__(self, action: Optional[Callable[[], Any]] = None) -> None:
        """Translate ``action`` into a CTRL-C handler.

        ``action`` is a callable that takes no arguments and returns no
        value (returned value is ignored).  It must *NEVER* raise an
        exception.

        If unspecified, a no-op will be used.
        """
        if os.name != "nt":
            return
        self._init_action(action)

    def _init_action(self, action):
        from ctypes import WINFUNCTYPE, windll
        from ctypes.wintypes import BOOL, DWORD

        kernel32 = windll.LoadLibrary('kernel32')

        # <http://msdn.microsoft.com/en-us/library/ms686016.aspx>
        PHANDLER_ROUTINE = WINFUNCTYPE(BOOL, DWORD)
        SetConsoleCtrlHandler = (
            self._SetConsoleCtrlHandler
        ) = kernel32.SetConsoleCtrlHandler
        SetConsoleCtrlHandler.argtypes = (PHANDLER_ROUTINE, BOOL)
        SetConsoleCtrlHandler.restype = BOOL

        if action is None:
            action = lambda: None
        self.action = action

        @PHANDLER_ROUTINE
        def handle(event):
            if event == 0:  # CTRL_C_EVENT
                action()
                # Typical C implementations would return 1 to indicate that
                # the event was processed and other control handlers in the
                # stack should not be executed.  However, that would
                # prevent the Python interpreter's handler from translating
                # CTRL-C to a `KeyboardInterrupt` exception, so we pretend
                # that we didn't handle it.
            return 0

        self.handle = handle

    def __enter__(self):
        """Install the custom CTRL-C handler."""
        if os.name != "nt":
            return
        result = self._SetConsoleCtrlHandler(self.handle, 1)
        if result == 0:
            # Have standard library automatically call `GetLastError()` and
            # `FormatMessage()` into a nice exception object :-)
            raise OSError()

    def __exit__(self, *args):
        """Remove the custom CTRL-C handler."""
        if os.name != "nt":
            return
        result = self._SetConsoleCtrlHandler(self.handle, 0)
        if result == 0:
            # Have standard library automatically call `GetLastError()` and
            # `FormatMessage()` into a nice exception object :-)
            raise OSError()
