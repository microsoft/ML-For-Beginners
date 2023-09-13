"""0MQ Device classes for running in background threads or processes."""

# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.

from zmq import device
from zmq.devices import (
    basedevice,
    monitoredqueue,
    monitoredqueuedevice,
    proxydevice,
    proxysteerabledevice,
)
from zmq.devices.basedevice import *
from zmq.devices.monitoredqueue import *
from zmq.devices.monitoredqueuedevice import *
from zmq.devices.proxydevice import *
from zmq.devices.proxysteerabledevice import *

__all__ = ['device']
for submod in (
    basedevice,
    proxydevice,
    proxysteerabledevice,
    monitoredqueue,
    monitoredqueuedevice,
):
    __all__.extend(submod.__all__)  # type: ignore
