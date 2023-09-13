"""ZAP Authenticator integrated with the tornado IOLoop.

.. versionadded:: 14.1
.. deprecated:: 25
    Use asyncio.AsyncioAuthenticator instead.
    Since tornado runs on asyncio, the asyncio authenticator
    offers the same functionality in tornado.
"""

import warnings

# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.
from typing import Any, Optional

import zmq

from .asyncio import AsyncioAuthenticator

warnings.warn(
    "zmq.auth.ioloop.IOLoopAuthenticator is deprecated. Use zmq.auth.asyncio.AsyncioAuthenticator",
    DeprecationWarning,
    stacklevel=2,
)


class IOLoopAuthenticator(AsyncioAuthenticator):
    """ZAP authentication for use in the tornado IOLoop"""

    def __init__(
        self,
        context: Optional["zmq.Context"] = None,
        encoding: str = 'utf-8',
        log: Any = None,
        io_loop: Any = None,
    ):
        loop = None
        if io_loop is not None:
            warnings.warn(
                f"{self.__class__.__name__}(io_loop) is deprecated and ignored",
                DeprecationWarning,
                stacklevel=2,
            )
            loop = io_loop.asyncio_loop
        super().__init__(context=context, encoding=encoding, log=log, loop=loop)


__all__ = ['IOLoopAuthenticator']
