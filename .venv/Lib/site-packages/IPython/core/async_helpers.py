"""
Async helper function that are invalid syntax on Python 3.5 and below.

This code is best effort, and may have edge cases not behaving as expected. In
particular it contain a number of heuristics to detect whether code is
effectively async and need to run in an event loop or not.

Some constructs (like top-level `return`, or `yield`) are taken care of
explicitly to actually raise a SyntaxError and stay as close as possible to
Python semantics.
"""


import ast
import asyncio
import inspect
from functools import wraps

_asyncio_event_loop = None


def get_asyncio_loop():
    """asyncio has deprecated get_event_loop

    Replicate it here, with our desired semantics:

    - always returns a valid, not-closed loop
    - not thread-local like asyncio's,
      because we only want one loop for IPython
    - if called from inside a coroutine (e.g. in ipykernel),
      return the running loop

    .. versionadded:: 8.0
    """
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        # not inside a coroutine,
        # track our own global
        pass

    # not thread-local like asyncio's,
    # because we only track one event loop to run for IPython itself,
    # always in the main thread.
    global _asyncio_event_loop
    if _asyncio_event_loop is None or _asyncio_event_loop.is_closed():
        _asyncio_event_loop = asyncio.new_event_loop()
    return _asyncio_event_loop


class _AsyncIORunner:
    def __call__(self, coro):
        """
        Handler for asyncio autoawait
        """
        return get_asyncio_loop().run_until_complete(coro)

    def __str__(self):
        return "asyncio"


_asyncio_runner = _AsyncIORunner()


class _AsyncIOProxy:
    """Proxy-object for an asyncio

    Any coroutine methods will be wrapped in event_loop.run_
    """

    def __init__(self, obj, event_loop):
        self._obj = obj
        self._event_loop = event_loop

    def __repr__(self):
        return f"<_AsyncIOProxy({self._obj!r})>"

    def __getattr__(self, key):
        attr = getattr(self._obj, key)
        if inspect.iscoroutinefunction(attr):
            # if it's a coroutine method,
            # return a threadsafe wrapper onto the _current_ asyncio loop
            @wraps(attr)
            def _wrapped(*args, **kwargs):
                concurrent_future = asyncio.run_coroutine_threadsafe(
                    attr(*args, **kwargs), self._event_loop
                )
                return asyncio.wrap_future(concurrent_future)

            return _wrapped
        else:
            return attr

    def __dir__(self):
        return dir(self._obj)


def _curio_runner(coroutine):
    """
    handler for curio autoawait
    """
    import curio

    return curio.run(coroutine)


def _trio_runner(async_fn):
    import trio

    async def loc(coro):
        """
        We need the dummy no-op async def to protect from
        trio's internal. See https://github.com/python-trio/trio/issues/89
        """
        return await coro

    return trio.run(loc, async_fn)


def _pseudo_sync_runner(coro):
    """
    A runner that does not really allow async execution, and just advance the coroutine.

    See discussion in https://github.com/python-trio/trio/issues/608,

    Credit to Nathaniel Smith
    """
    try:
        coro.send(None)
    except StopIteration as exc:
        return exc.value
    else:
        # TODO: do not raise but return an execution result with the right info.
        raise RuntimeError(
            "{coro_name!r} needs a real async loop".format(coro_name=coro.__name__)
        )


def _should_be_async(cell: str) -> bool:
    """Detect if a block of code need to be wrapped in an `async def`

    Attempt to parse the block of code, it it compile we're fine.
    Otherwise we  wrap if and try to compile.

    If it works, assume it should be async. Otherwise Return False.

    Not handled yet: If the block of code has a return statement as the top
    level, it will be seen as async. This is a know limitation.
    """
    try:
        code = compile(
            cell, "<>", "exec", flags=getattr(ast, "PyCF_ALLOW_TOP_LEVEL_AWAIT", 0x0)
        )
        return inspect.CO_COROUTINE & code.co_flags == inspect.CO_COROUTINE
    except (SyntaxError, MemoryError):
        return False
