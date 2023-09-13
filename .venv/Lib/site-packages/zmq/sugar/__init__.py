"""pure-Python sugar wrappers for core 0MQ objects."""

# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.


from zmq import error
from zmq.sugar import context, frame, poll, socket, tracker, version

__all__ = []
for submod in (context, error, frame, poll, socket, tracker, version):
    __all__.extend(submod.__all__)

from zmq.error import *  # noqa
from zmq.sugar.context import *  # noqa
from zmq.sugar.frame import *  # noqa
from zmq.sugar.poll import *  # noqa
from zmq.sugar.socket import *  # noqa

# deprecated:
from zmq.sugar.stopwatch import Stopwatch  # noqa
from zmq.sugar.tracker import *  # noqa
from zmq.sugar.version import *  # noqa

__all__.append('Stopwatch')
