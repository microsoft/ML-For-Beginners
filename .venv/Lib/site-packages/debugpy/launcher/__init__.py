# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root
# for license information.

__all__ = []


adapter_host = None
"""The host on which adapter is running and listening for incoming connections
from the launcher and the servers."""

channel = None
"""DAP message channel to the adapter."""


def connect(host, port):
    from debugpy.common import log, messaging, sockets
    from debugpy.launcher import handlers

    global channel, adapter_host
    assert channel is None
    assert adapter_host is None

    log.info("Connecting to adapter at {0}:{1}", host, port)

    sock = sockets.create_client()
    sock.connect((host, port))
    adapter_host = host

    stream = messaging.JsonIOStream.from_socket(sock, "Adapter")
    channel = messaging.JsonMessageChannel(stream, handlers=handlers)
    channel.start()
