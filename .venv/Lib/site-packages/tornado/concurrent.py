#
# Copyright 2012 Facebook
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
"""Utilities for working with ``Future`` objects.

Tornado previously provided its own ``Future`` class, but now uses
`asyncio.Future`. This module contains utility functions for working
with `asyncio.Future` in a way that is backwards-compatible with
Tornado's old ``Future`` implementation.

While this module is an important part of Tornado's internal
implementation, applications rarely need to interact with it
directly.

"""

import asyncio
from concurrent import futures
import functools
import sys
import types

from tornado.log import app_log

import typing
from typing import Any, Callable, Optional, Tuple, Union

_T = typing.TypeVar("_T")


class ReturnValueIgnoredError(Exception):
    # No longer used; was previously used by @return_future
    pass


Future = asyncio.Future

FUTURES = (futures.Future, Future)


def is_future(x: Any) -> bool:
    return isinstance(x, FUTURES)


class DummyExecutor(futures.Executor):
    def submit(
        self, fn: Callable[..., _T], *args: Any, **kwargs: Any
    ) -> "futures.Future[_T]":
        future = futures.Future()  # type: futures.Future[_T]
        try:
            future_set_result_unless_cancelled(future, fn(*args, **kwargs))
        except Exception:
            future_set_exc_info(future, sys.exc_info())
        return future

    def shutdown(self, wait: bool = True) -> None:
        pass


dummy_executor = DummyExecutor()


def run_on_executor(*args: Any, **kwargs: Any) -> Callable:
    """Decorator to run a synchronous method asynchronously on an executor.

    Returns a future.

    The executor to be used is determined by the ``executor``
    attributes of ``self``. To use a different attribute name, pass a
    keyword argument to the decorator::

        @run_on_executor(executor='_thread_pool')
        def foo(self):
            pass

    This decorator should not be confused with the similarly-named
    `.IOLoop.run_in_executor`. In general, using ``run_in_executor``
    when *calling* a blocking method is recommended instead of using
    this decorator when *defining* a method. If compatibility with older
    versions of Tornado is required, consider defining an executor
    and using ``executor.submit()`` at the call site.

    .. versionchanged:: 4.2
       Added keyword arguments to use alternative attributes.

    .. versionchanged:: 5.0
       Always uses the current IOLoop instead of ``self.io_loop``.

    .. versionchanged:: 5.1
       Returns a `.Future` compatible with ``await`` instead of a
       `concurrent.futures.Future`.

    .. deprecated:: 5.1

       The ``callback`` argument is deprecated and will be removed in
       6.0. The decorator itself is discouraged in new code but will
       not be removed in 6.0.

    .. versionchanged:: 6.0

       The ``callback`` argument was removed.
    """
    # Fully type-checking decorators is tricky, and this one is
    # discouraged anyway so it doesn't have all the generic magic.
    def run_on_executor_decorator(fn: Callable) -> Callable[..., Future]:
        executor = kwargs.get("executor", "executor")

        @functools.wraps(fn)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Future:
            async_future = Future()  # type: Future
            conc_future = getattr(self, executor).submit(fn, self, *args, **kwargs)
            chain_future(conc_future, async_future)
            return async_future

        return wrapper

    if args and kwargs:
        raise ValueError("cannot combine positional and keyword args")
    if len(args) == 1:
        return run_on_executor_decorator(args[0])
    elif len(args) != 0:
        raise ValueError("expected 1 argument, got %d", len(args))
    return run_on_executor_decorator


_NO_RESULT = object()


def chain_future(a: "Future[_T]", b: "Future[_T]") -> None:
    """Chain two futures together so that when one completes, so does the other.

    The result (success or failure) of ``a`` will be copied to ``b``, unless
    ``b`` has already been completed or cancelled by the time ``a`` finishes.

    .. versionchanged:: 5.0

       Now accepts both Tornado/asyncio `Future` objects and
       `concurrent.futures.Future`.

    """

    def copy(future: "Future[_T]") -> None:
        assert future is a
        if b.done():
            return
        if hasattr(a, "exc_info") and a.exc_info() is not None:  # type: ignore
            future_set_exc_info(b, a.exc_info())  # type: ignore
        else:
            a_exc = a.exception()
            if a_exc is not None:
                b.set_exception(a_exc)
            else:
                b.set_result(a.result())

    if isinstance(a, Future):
        future_add_done_callback(a, copy)
    else:
        # concurrent.futures.Future
        from tornado.ioloop import IOLoop

        IOLoop.current().add_future(a, copy)


def future_set_result_unless_cancelled(
    future: "Union[futures.Future[_T], Future[_T]]", value: _T
) -> None:
    """Set the given ``value`` as the `Future`'s result, if not cancelled.

    Avoids ``asyncio.InvalidStateError`` when calling ``set_result()`` on
    a cancelled `asyncio.Future`.

    .. versionadded:: 5.0
    """
    if not future.cancelled():
        future.set_result(value)


def future_set_exception_unless_cancelled(
    future: "Union[futures.Future[_T], Future[_T]]", exc: BaseException
) -> None:
    """Set the given ``exc`` as the `Future`'s exception.

    If the Future is already canceled, logs the exception instead. If
    this logging is not desired, the caller should explicitly check
    the state of the Future and call ``Future.set_exception`` instead of
    this wrapper.

    Avoids ``asyncio.InvalidStateError`` when calling ``set_exception()`` on
    a cancelled `asyncio.Future`.

    .. versionadded:: 6.0

    """
    if not future.cancelled():
        future.set_exception(exc)
    else:
        app_log.error("Exception after Future was cancelled", exc_info=exc)


def future_set_exc_info(
    future: "Union[futures.Future[_T], Future[_T]]",
    exc_info: Tuple[
        Optional[type], Optional[BaseException], Optional[types.TracebackType]
    ],
) -> None:
    """Set the given ``exc_info`` as the `Future`'s exception.

    Understands both `asyncio.Future` and the extensions in older
    versions of Tornado to enable better tracebacks on Python 2.

    .. versionadded:: 5.0

    .. versionchanged:: 6.0

       If the future is already cancelled, this function is a no-op.
       (previously ``asyncio.InvalidStateError`` would be raised)

    """
    if exc_info[1] is None:
        raise Exception("future_set_exc_info called with no exception")
    future_set_exception_unless_cancelled(future, exc_info[1])


@typing.overload
def future_add_done_callback(
    future: "futures.Future[_T]", callback: Callable[["futures.Future[_T]"], None]
) -> None:
    pass


@typing.overload  # noqa: F811
def future_add_done_callback(
    future: "Future[_T]", callback: Callable[["Future[_T]"], None]
) -> None:
    pass


def future_add_done_callback(  # noqa: F811
    future: "Union[futures.Future[_T], Future[_T]]", callback: Callable[..., None]
) -> None:
    """Arrange to call ``callback`` when ``future`` is complete.

    ``callback`` is invoked with one argument, the ``future``.

    If ``future`` is already done, ``callback`` is invoked immediately.
    This may differ from the behavior of ``Future.add_done_callback``,
    which makes no such guarantee.

    .. versionadded:: 5.0
    """
    if future.done():
        callback(future)
    else:
        future.add_done_callback(callback)
