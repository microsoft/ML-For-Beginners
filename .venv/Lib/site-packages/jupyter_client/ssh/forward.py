"""Sample script showing how to do local port forwarding over paramiko.

This script connects to the requested SSH server and sets up local port
forwarding (the openssh -L option) from a local port through a tunneled
connection to a destination reachable from the SSH server machine.
"""
#
# This file is adapted from a paramiko demo, and thus licensed under LGPL 2.1.
# Original Copyright (C) 2003-2007  Robey Pointer <robeypointer@gmail.com>
# Edits Copyright (C) 2010 The IPython Team
#
# Paramiko is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 2.1 of the License, or (at your option)
# any later version.
#
# Paramiko is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Paramiko; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02111-1301  USA.
import logging
import select
import socketserver
import typing as t

logger = logging.getLogger("ssh")


class ForwardServer(socketserver.ThreadingTCPServer):
    """A server to use for ssh forwarding."""

    daemon_threads = True
    allow_reuse_address = True


class Handler(socketserver.BaseRequestHandler):
    """A handle for server requests."""

    @t.no_type_check
    def handle(self):
        """Handle a request."""
        try:
            chan = self.ssh_transport.open_channel(
                "direct-tcpip",
                (self.chain_host, self.chain_port),
                self.request.getpeername(),
            )
        except Exception as e:
            logger.debug(
                "Incoming request to %s:%d failed: %s" % (self.chain_host, self.chain_port, repr(e))
            )
            return
        if chan is None:
            logger.debug(
                "Incoming request to %s:%d was rejected by the SSH server."
                % (self.chain_host, self.chain_port)
            )
            return

        logger.debug(
            "Connected!  Tunnel open {!r} -> {!r} -> {!r}".format(
                self.request.getpeername(),
                chan.getpeername(),
                (self.chain_host, self.chain_port),
            )
        )
        while True:
            r, w, x = select.select([self.request, chan], [], [])
            if self.request in r:
                data = self.request.recv(1024)
                if len(data) == 0:
                    break
                chan.send(data)
            if chan in r:
                data = chan.recv(1024)
                if len(data) == 0:
                    break
                self.request.send(data)
        chan.close()
        self.request.close()
        logger.debug("Tunnel closed ")


def forward_tunnel(local_port: int, remote_host: str, remote_port: int, transport: t.Any) -> None:
    """Forward an ssh tunnel."""

    # this is a little convoluted, but lets me configure things for the Handler
    # object.  (SocketServer doesn't give Handlers any way to access the outer
    # server normally.)
    class SubHander(Handler):
        chain_host = remote_host
        chain_port = remote_port
        ssh_transport = transport

    ForwardServer(("127.0.0.1", local_port), SubHander).serve_forever()


__all__ = ["forward_tunnel"]
