# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

import asyncio
import atexit
import errno
import inspect
import os
import sys
import threading
import warnings
from pathlib import Path
from types import FrameType
from typing import Awaitable, Callable, List, Optional, TypeVar, Union, cast


def ensure_dir_exists(path, mode=0o777):
    """Ensure that a directory exists

    If it doesn't exist, try to create it, protecting against a race condition
    if another process is doing the same.
    The default permissions are determined by the current umask.
    """
    try:
        os.makedirs(path, mode=mode)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    if not os.path.isdir(path):
        raise OSError("%r exists but is not a directory" % path)


def _get_frame(level: int) -> Optional[FrameType]:
    """Get the frame at the given stack level."""
    # sys._getframe is much faster than inspect.stack, but isn't guaranteed to
    # exist in all python implementations, so we fall back to inspect.stack()

    # We need to add one to level to account for this get_frame call.
    if hasattr(sys, "_getframe"):
        frame = sys._getframe(level + 1)
    else:
        frame = inspect.stack(context=0)[level + 1].frame
    return frame


# This function is from https://github.com/python/cpython/issues/67998
# (https://bugs.python.org/file39550/deprecated_module_stacklevel.diff) and
# calculates the appropriate stacklevel for deprecations to target the
# deprecation for the caller, no matter how many internal stack frames we have
# added in the process. For example, with the deprecation warning in the
# __init__ below, the appropriate stacklevel will change depending on how deep
# the inheritance hierarchy is.
def _external_stacklevel(internal: List[str]) -> int:
    """Find the stacklevel of the first frame that doesn't contain any of the given internal strings

    The depth will be 1 at minimum in order to start checking at the caller of
    the function that called this utility method.
    """
    # Get the level of my caller's caller
    level = 2
    frame = _get_frame(level)

    # Normalize the path separators:
    normalized_internal = [str(Path(s)) for s in internal]

    # climb the stack frames while we see internal frames
    while frame and any(s in str(Path(frame.f_code.co_filename)) for s in normalized_internal):
        level += 1
        frame = frame.f_back

    # Return the stack level from the perspective of whoever called us (i.e., one level up)
    return level - 1


def deprecation(message: str, internal: Union[str, List[str]] = "jupyter_core/") -> None:
    """Generate a deprecation warning targeting the first frame that is not 'internal'

    internal is a string or list of strings, which if they appear in filenames in the
    frames, the frames will be considered internal. Changing this can be useful if, for examnple,
    we know that our internal code is calling out to another library.
    """
    _internal: List[str]
    _internal = [internal] if isinstance(internal, str) else internal

    # stack level of the first external frame from here
    stacklevel = _external_stacklevel(_internal)

    # The call to .warn adds one frame, so bump the stacklevel up by one
    warnings.warn(message, DeprecationWarning, stacklevel=stacklevel + 1)


T = TypeVar("T")


class _TaskRunner:
    """A task runner that runs an asyncio event loop on a background thread."""

    def __init__(self):
        self.__io_loop: Optional[asyncio.AbstractEventLoop] = None
        self.__runner_thread: Optional[threading.Thread] = None
        self.__lock = threading.Lock()
        atexit.register(self._close)

    def _close(self):
        if self.__io_loop:
            self.__io_loop.stop()

    def _runner(self):
        loop = self.__io_loop
        assert loop is not None  # noqa
        try:
            loop.run_forever()
        finally:
            loop.close()

    def run(self, coro):
        """Synchronously run a coroutine on a background thread."""
        with self.__lock:
            name = f"{threading.current_thread().name} - runner"
            if self.__io_loop is None:
                self.__io_loop = asyncio.new_event_loop()
                self.__runner_thread = threading.Thread(target=self._runner, daemon=True, name=name)
                self.__runner_thread.start()
        fut = asyncio.run_coroutine_threadsafe(coro, self.__io_loop)
        return fut.result(None)


_runner_map = {}
_loop_map = {}


def run_sync(coro: Callable[..., Awaitable[T]]) -> Callable[..., T]:
    """Wraps coroutine in a function that blocks until it has executed.

    Parameters
    ----------
    coro : coroutine-function
        The coroutine-function to be executed.

    Returns
    -------
    result :
        Whatever the coroutine-function returns.
    """

    if not inspect.iscoroutinefunction(coro):
        raise AssertionError

    def wrapped(*args, **kwargs):
        name = threading.current_thread().name
        inner = coro(*args, **kwargs)
        try:
            # If a loop is currently running in this thread,
            # use a task runner.
            asyncio.get_running_loop()
            if name not in _runner_map:
                _runner_map[name] = _TaskRunner()
            return _runner_map[name].run(inner)
        except RuntimeError:
            pass

        # Run the loop for this thread.
        if name not in _loop_map:
            _loop_map[name] = asyncio.new_event_loop()
        loop = _loop_map[name]
        return loop.run_until_complete(inner)

    wrapped.__doc__ = coro.__doc__
    return wrapped


async def ensure_async(obj: Union[Awaitable[T], T]) -> T:
    """Convert a non-awaitable object to a coroutine if needed,
    and await it if it was not already awaited.

    This function is meant to be called on the result of calling a function,
    when that function could either be asynchronous or not.
    """
    if inspect.isawaitable(obj):
        obj = cast(Awaitable[T], obj)
        try:
            result = await obj
        except RuntimeError as e:
            if str(e) == "cannot reuse already awaited coroutine":
                # obj is already the coroutine's result
                return cast(T, obj)
            raise
        return result
    # obj doesn't need to be awaited
    return cast(T, obj)
