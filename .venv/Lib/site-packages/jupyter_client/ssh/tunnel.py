"""Basic ssh tunnel utilities, and convenience functions for tunneling
zeromq connections.
"""
# Copyright (C) 2010-2011  IPython Development Team
# Copyright (C) 2011- PyZMQ Developers
#
# Redistributed from IPython under the terms of the BSD License.
from __future__ import annotations

import atexit
import os
import re
import signal
import socket
import sys
import warnings
from getpass import getpass, getuser
from multiprocessing import Process
from typing import Any, cast

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import paramiko

        SSHException = paramiko.ssh_exception.SSHException
except ImportError:
    paramiko = None  # type:ignore[assignment]

    class SSHException(Exception):  # type:ignore[no-redef]  # noqa
        pass

else:
    from .forward import forward_tunnel

try:
    import pexpect  # type: ignore[import-untyped]
except ImportError:
    pexpect = None


def select_random_ports(n: int) -> list[int]:
    """Select and return n random ports that are available."""
    ports = []
    sockets = []
    for _ in range(n):
        sock = socket.socket()
        sock.bind(("", 0))
        ports.append(sock.getsockname()[1])
        sockets.append(sock)
    for sock in sockets:
        sock.close()
    return ports


# -----------------------------------------------------------------------------
# Check for passwordless login
# -----------------------------------------------------------------------------
_password_pat = re.compile((rb"pass(word|phrase):"), re.IGNORECASE)


def try_passwordless_ssh(server: str, keyfile: str | None, paramiko: Any = None) -> Any:
    """Attempt to make an ssh connection without a password.
    This is mainly used for requiring password input only once
    when many tunnels may be connected to the same server.

    If paramiko is None, the default for the platform is chosen.
    """
    if paramiko is None:
        paramiko = sys.platform == "win32"
    f = _try_passwordless_paramiko if paramiko else _try_passwordless_openssh
    return f(server, keyfile)


def _try_passwordless_openssh(server: str, keyfile: str | None) -> bool:
    """Try passwordless login with shell ssh command."""
    if pexpect is None:
        msg = "pexpect unavailable, use paramiko"
        raise ImportError(msg)
    cmd = "ssh -f " + server
    if keyfile:
        cmd += " -i " + keyfile
    cmd += " exit"

    # pop SSH_ASKPASS from env
    env = os.environ.copy()
    env.pop("SSH_ASKPASS", None)

    ssh_newkey = "Are you sure you want to continue connecting"
    p = pexpect.spawn(cmd, env=env)
    while True:
        try:
            i = p.expect([ssh_newkey, _password_pat], timeout=0.1)
            if i == 0:
                msg = "The authenticity of the host can't be established."
                raise SSHException(msg)
        except pexpect.TIMEOUT:
            continue
        except pexpect.EOF:
            return True
        else:
            return False


def _try_passwordless_paramiko(server: str, keyfile: str | None) -> bool:
    """Try passwordless login with paramiko."""
    if paramiko is None:
        msg = "Paramiko unavailable, "  # type:ignore[unreachable]
        if sys.platform == "win32":
            msg += "Paramiko is required for ssh tunneled connections on Windows."
        else:
            msg += "use OpenSSH."
        raise ImportError(msg)
    username, server, port = _split_server(server)
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.WarningPolicy())
    try:
        client.connect(server, port, username=username, key_filename=keyfile, look_for_keys=True)
    except paramiko.AuthenticationException:
        return False
    else:
        client.close()
        return True


def tunnel_connection(
    socket: socket.socket,
    addr: str,
    server: str,
    keyfile: str | None = None,
    password: str | None = None,
    paramiko: Any = None,
    timeout: int = 60,
) -> int:
    """Connect a socket to an address via an ssh tunnel.

    This is a wrapper for socket.connect(addr), when addr is not accessible
    from the local machine.  It simply creates an ssh tunnel using the remaining args,
    and calls socket.connect('tcp://localhost:lport') where lport is the randomly
    selected local port of the tunnel.

    """
    new_url, tunnel = open_tunnel(
        addr,
        server,
        keyfile=keyfile,
        password=password,
        paramiko=paramiko,
        timeout=timeout,
    )
    socket.connect(new_url)
    return tunnel


def open_tunnel(
    addr: str,
    server: str,
    keyfile: str | None = None,
    password: str | None = None,
    paramiko: Any = None,
    timeout: int = 60,
) -> tuple[str, int]:
    """Open a tunneled connection from a 0MQ url.

    For use inside tunnel_connection.

    Returns
    -------

    (url, tunnel) : (str, object)
        The 0MQ url that has been forwarded, and the tunnel object
    """

    lport = select_random_ports(1)[0]
    _, addr = addr.split("://")
    ip, rport = addr.split(":")
    rport_int = int(rport)
    paramiko = sys.platform == "win32" if paramiko is None else paramiko_tunnel
    tunnelf = paramiko_tunnel if paramiko else openssh_tunnel

    tunnel = tunnelf(
        lport,
        rport_int,
        server,
        remoteip=ip,
        keyfile=keyfile,
        password=password,
        timeout=timeout,
    )
    return "tcp://127.0.0.1:%i" % lport, cast(int, tunnel)


def openssh_tunnel(
    lport: int,
    rport: int,
    server: str,
    remoteip: str = "127.0.0.1",
    keyfile: str | None = None,
    password: str | None | bool = None,
    timeout: int = 60,
) -> int:
    """Create an ssh tunnel using command-line ssh that connects port lport
    on this machine to localhost:rport on server.  The tunnel
    will automatically close when not in use, remaining open
    for a minimum of timeout seconds for an initial connection.

    This creates a tunnel redirecting `localhost:lport` to `remoteip:rport`,
    as seen from `server`.

    keyfile and password may be specified, but ssh config is checked for defaults.

    Parameters
    ----------

    lport : int
        local port for connecting to the tunnel from this machine.
    rport : int
        port on the remote machine to connect to.
    server : str
        The ssh server to connect to. The full ssh server string will be parsed.
        user@server:port
    remoteip : str [Default: 127.0.0.1]
        The remote ip, specifying the destination of the tunnel.
        Default is localhost, which means that the tunnel would redirect
        localhost:lport on this machine to localhost:rport on the *server*.

    keyfile : str; path to public key file
        This specifies a key to be used in ssh login, default None.
        Regular default ssh keys will be used without specifying this argument.
    password : str;
        Your ssh password to the ssh server. Note that if this is left None,
        you will be prompted for it if passwordless key based login is unavailable.
    timeout : int [default: 60]
        The time (in seconds) after which no activity will result in the tunnel
        closing.  This prevents orphaned tunnels from running forever.
    """
    if pexpect is None:
        msg = "pexpect unavailable, use paramiko_tunnel"
        raise ImportError(msg)
    ssh = "ssh "
    if keyfile:
        ssh += "-i " + keyfile

    if ":" in server:
        server, port = server.split(":")
        ssh += " -p %s" % port

    cmd = f"{ssh} -O check {server}"
    (output, exitstatus) = pexpect.run(cmd, withexitstatus=True)
    if not exitstatus:
        pid = int(output[output.find(b"(pid=") + 5 : output.find(b")")])
        cmd = "%s -O forward -L 127.0.0.1:%i:%s:%i %s" % (
            ssh,
            lport,
            remoteip,
            rport,
            server,
        )
        (output, exitstatus) = pexpect.run(cmd, withexitstatus=True)
        if not exitstatus:
            atexit.register(_stop_tunnel, cmd.replace("-O forward", "-O cancel", 1))
            return pid
    cmd = "%s -f -S none -L 127.0.0.1:%i:%s:%i %s sleep %i" % (
        ssh,
        lport,
        remoteip,
        rport,
        server,
        timeout,
    )

    # pop SSH_ASKPASS from env
    env = os.environ.copy()
    env.pop("SSH_ASKPASS", None)

    ssh_newkey = "Are you sure you want to continue connecting"
    tunnel = pexpect.spawn(cmd, env=env)
    failed = False
    while True:
        try:
            i = tunnel.expect([ssh_newkey, _password_pat], timeout=0.1)
            if i == 0:
                msg = "The authenticity of the host can't be established."
                raise SSHException(msg)
        except pexpect.TIMEOUT:
            continue
        except pexpect.EOF as e:
            tunnel.wait()
            if tunnel.exitstatus:
                raise RuntimeError("tunnel '%s' failed to start" % (cmd)) from e
            else:
                return tunnel.pid
        else:
            if failed:
                warnings.warn("Password rejected, try again", stacklevel=2)
                password = None
            if password is None:
                password = getpass("%s's password: " % (server))
            tunnel.sendline(password)
            failed = True


def _stop_tunnel(cmd: Any) -> None:
    pexpect.run(cmd)


def _split_server(server: str) -> tuple[str, str, int]:
    if "@" in server:
        username, server = server.split("@", 1)
    else:
        username = getuser()
    if ":" in server:
        server, port_str = server.split(":")
        port = int(port_str)
    else:
        port = 22
    return username, server, port


def paramiko_tunnel(
    lport: int,
    rport: int,
    server: str,
    remoteip: str = "127.0.0.1",
    keyfile: str | None = None,
    password: str | None = None,
    timeout: float = 60,
) -> Process:
    """launch a tunner with paramiko in a subprocess. This should only be used
    when shell ssh is unavailable (e.g. Windows).

    This creates a tunnel redirecting `localhost:lport` to `remoteip:rport`,
    as seen from `server`.

    If you are familiar with ssh tunnels, this creates the tunnel:

    ssh server -L localhost:lport:remoteip:rport

    keyfile and password may be specified, but ssh config is checked for defaults.


    Parameters
    ----------

    lport : int
        local port for connecting to the tunnel from this machine.
    rport : int
        port on the remote machine to connect to.
    server : str
        The ssh server to connect to. The full ssh server string will be parsed.
        user@server:port
    remoteip : str [Default: 127.0.0.1]
        The remote ip, specifying the destination of the tunnel.
        Default is localhost, which means that the tunnel would redirect
        localhost:lport on this machine to localhost:rport on the *server*.

    keyfile : str; path to public key file
        This specifies a key to be used in ssh login, default None.
        Regular default ssh keys will be used without specifying this argument.
    password : str;
        Your ssh password to the ssh server. Note that if this is left None,
        you will be prompted for it if passwordless key based login is unavailable.
    timeout : int [default: 60]
        The time (in seconds) after which no activity will result in the tunnel
        closing.  This prevents orphaned tunnels from running forever.

    """
    if paramiko is None:
        msg = "Paramiko not available"  # type:ignore[unreachable]
        raise ImportError(msg)

    if password is None and not _try_passwordless_paramiko(server, keyfile):
        password = getpass("%s's password: " % (server))

    p = Process(
        target=_paramiko_tunnel,
        args=(lport, rport, server, remoteip),
        kwargs={"keyfile": keyfile, "password": password},
    )
    p.daemon = True
    p.start()
    return p


def _paramiko_tunnel(
    lport: int,
    rport: int,
    server: str,
    remoteip: str,
    keyfile: str | None = None,
    password: str | None = None,
) -> None:
    """Function for actually starting a paramiko tunnel, to be passed
    to multiprocessing.Process(target=this), and not called directly.
    """
    username, server, port = _split_server(server)
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.WarningPolicy())

    try:
        client.connect(
            server,
            port,
            username=username,
            key_filename=keyfile,
            look_for_keys=True,
            password=password,
        )
    #    except paramiko.AuthenticationException:
    #        if password is None:
    #            password = getpass("%s@%s's password: "%(username, server))
    #            client.connect(server, port, username=username, password=password)
    #        else:
    #            raise
    except Exception as e:
        warnings.warn("*** Failed to connect to %s:%d: %r" % (server, port, e), stacklevel=2)
        sys.exit(1)

    # Don't let SIGINT kill the tunnel subprocess
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    try:
        forward_tunnel(lport, remoteip, rport, client.get_transport())
    except KeyboardInterrupt:
        warnings.warn("SIGINT: Port forwarding stopped cleanly", stacklevel=2)
        sys.exit(0)
    except Exception as e:
        warnings.warn("Port forwarding stopped uncleanly: %s" % e, stacklevel=2)
        sys.exit(255)


if sys.platform == "win32":
    ssh_tunnel = paramiko_tunnel
else:
    ssh_tunnel = openssh_tunnel


__all__ = [
    "tunnel_connection",
    "ssh_tunnel",
    "openssh_tunnel",
    "paramiko_tunnel",
    "try_passwordless_ssh",
]
