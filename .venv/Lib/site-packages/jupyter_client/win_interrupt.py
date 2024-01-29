"""Use a Windows event to interrupt a child process like SIGINT.

The child needs to explicitly listen for this - see
ipykernel.parentpoller.ParentPollerWindows for a Python implementation.
"""
import ctypes
from typing import Any


def create_interrupt_event() -> Any:
    """Create an interrupt event handle.

    The parent process should call this to create the
    interrupt event that is passed to the child process. It should store
    this handle and use it with ``send_interrupt`` to interrupt the child
    process.
    """

    # Create a security attributes struct that permits inheritance of the
    # handle by new processes.
    # FIXME: We can clean up this mess by requiring pywin32 for IPython.
    class SECURITY_ATTRIBUTES(ctypes.Structure):  # noqa
        _fields_ = [
            ("nLength", ctypes.c_int),
            ("lpSecurityDescriptor", ctypes.c_void_p),
            ("bInheritHandle", ctypes.c_int),
        ]

    sa = SECURITY_ATTRIBUTES()
    sa_p = ctypes.pointer(sa)
    sa.nLength = ctypes.sizeof(SECURITY_ATTRIBUTES)
    sa.lpSecurityDescriptor = 0
    sa.bInheritHandle = 1

    return ctypes.windll.kernel32.CreateEventA(  # type:ignore[attr-defined]
        sa_p,
        False,
        False,
        "",  # lpEventAttributes  # bManualReset  # bInitialState
    )  # lpName


def send_interrupt(interrupt_handle: Any) -> None:
    """Sends an interrupt event using the specified handle."""
    ctypes.windll.kernel32.SetEvent(interrupt_handle)  # type:ignore[attr-defined]
