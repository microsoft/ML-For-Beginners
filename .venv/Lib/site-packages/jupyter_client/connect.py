"""Utilities for connecting to jupyter kernels

The :class:`ConnectionFileMixin` class in this module encapsulates the logic
related to writing and reading connections files.
"""
# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
from __future__ import annotations

import errno
import glob
import json
import os
import socket
import stat
import tempfile
import warnings
from getpass import getpass
from typing import TYPE_CHECKING, Any, Dict, Union, cast

import zmq
from jupyter_core.paths import jupyter_data_dir, jupyter_runtime_dir, secure_write
from traitlets import Bool, CaselessStrEnum, Instance, Integer, Type, Unicode, observe
from traitlets.config import LoggingConfigurable, SingletonConfigurable

from .localinterfaces import localhost
from .utils import _filefind

if TYPE_CHECKING:
    from jupyter_client import BlockingKernelClient

    from .session import Session

# Define custom type for kernel connection info
KernelConnectionInfo = Dict[str, Union[int, str, bytes]]


def write_connection_file(
    fname: str | None = None,
    shell_port: int = 0,
    iopub_port: int = 0,
    stdin_port: int = 0,
    hb_port: int = 0,
    control_port: int = 0,
    ip: str = "",
    key: bytes = b"",
    transport: str = "tcp",
    signature_scheme: str = "hmac-sha256",
    kernel_name: str = "",
    **kwargs: Any,
) -> tuple[str, KernelConnectionInfo]:
    """Generates a JSON config file, including the selection of random ports.

    Parameters
    ----------

    fname : unicode
        The path to the file to write

    shell_port : int, optional
        The port to use for ROUTER (shell) channel.

    iopub_port : int, optional
        The port to use for the SUB channel.

    stdin_port : int, optional
        The port to use for the ROUTER (raw input) channel.

    control_port : int, optional
        The port to use for the ROUTER (control) channel.

    hb_port : int, optional
        The port to use for the heartbeat REP channel.

    ip  : str, optional
        The ip address the kernel will bind to.

    key : str, optional
        The Session key used for message authentication.

    signature_scheme : str, optional
        The scheme used for message authentication.
        This has the form 'digest-hash', where 'digest'
        is the scheme used for digests, and 'hash' is the name of the hash function
        used by the digest scheme.
        Currently, 'hmac' is the only supported digest scheme,
        and 'sha256' is the default hash function.

    kernel_name : str, optional
        The name of the kernel currently connected to.
    """
    if not ip:
        ip = localhost()
    # default to temporary connector file
    if not fname:
        fd, fname = tempfile.mkstemp(".json")
        os.close(fd)

    # Find open ports as necessary.

    ports: list[int] = []
    sockets: list[socket.socket] = []
    ports_needed = (
        int(shell_port <= 0)
        + int(iopub_port <= 0)
        + int(stdin_port <= 0)
        + int(control_port <= 0)
        + int(hb_port <= 0)
    )
    if transport == "tcp":
        for _ in range(ports_needed):
            sock = socket.socket()
            # struct.pack('ii', (0,0)) is 8 null bytes
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, b"\0" * 8)
            sock.bind((ip, 0))
            sockets.append(sock)
        for sock in sockets:
            port = sock.getsockname()[1]
            sock.close()
            ports.append(port)
    else:
        N = 1
        for _ in range(ports_needed):
            while os.path.exists(f"{ip}-{N!s}"):
                N += 1
            ports.append(N)
            N += 1
    if shell_port <= 0:
        shell_port = ports.pop(0)
    if iopub_port <= 0:
        iopub_port = ports.pop(0)
    if stdin_port <= 0:
        stdin_port = ports.pop(0)
    if control_port <= 0:
        control_port = ports.pop(0)
    if hb_port <= 0:
        hb_port = ports.pop(0)

    cfg: KernelConnectionInfo = {
        "shell_port": shell_port,
        "iopub_port": iopub_port,
        "stdin_port": stdin_port,
        "control_port": control_port,
        "hb_port": hb_port,
    }
    cfg["ip"] = ip
    cfg["key"] = key.decode()
    cfg["transport"] = transport
    cfg["signature_scheme"] = signature_scheme
    cfg["kernel_name"] = kernel_name
    cfg.update(kwargs)

    # Only ever write this file as user read/writeable
    # This would otherwise introduce a vulnerability as a file has secrets
    # which would let others execute arbitrary code as you
    with secure_write(fname) as f:
        f.write(json.dumps(cfg, indent=2))

    if hasattr(stat, "S_ISVTX"):
        # set the sticky bit on the parent directory of the file
        # to ensure only owner can remove it
        runtime_dir = os.path.dirname(fname)
        if runtime_dir:
            permissions = os.stat(runtime_dir).st_mode
            new_permissions = permissions | stat.S_ISVTX
            if new_permissions != permissions:
                try:
                    os.chmod(runtime_dir, new_permissions)
                except OSError as e:
                    if e.errno == errno.EPERM:
                        # suppress permission errors setting sticky bit on runtime_dir,
                        # which we may not own.
                        pass
    return fname, cfg


def find_connection_file(
    filename: str = "kernel-*.json",
    path: str | list[str] | None = None,
    profile: str | None = None,
) -> str:
    """find a connection file, and return its absolute path.

    The current working directory and optional search path
    will be searched for the file if it is not given by absolute path.

    If the argument does not match an existing file, it will be interpreted as a
    fileglob, and the matching file in the profile's security dir with
    the latest access time will be used.

    Parameters
    ----------
    filename : str
        The connection file or fileglob to search for.
    path : str or list of strs[optional]
        Paths in which to search for connection files.

    Returns
    -------
    str : The absolute path of the connection file.
    """
    if profile is not None:
        warnings.warn(
            "Jupyter has no profiles. profile=%s has been ignored." % profile, stacklevel=2
        )
    if path is None:
        path = [".", jupyter_runtime_dir()]
    if isinstance(path, str):
        path = [path]

    try:
        # first, try explicit name
        return _filefind(filename, path)
    except OSError:
        pass

    # not found by full name

    if "*" in filename:
        # given as a glob already
        pat = filename
    else:
        # accept any substring match
        pat = "*%s*" % filename

    matches = []
    for p in path:
        matches.extend(glob.glob(os.path.join(p, pat)))

    matches = [os.path.abspath(m) for m in matches]
    if not matches:
        msg = f"Could not find {filename!r} in {path!r}"
        raise OSError(msg)
    elif len(matches) == 1:
        return matches[0]
    else:
        # get most recent match, by access time:
        return sorted(matches, key=lambda f: os.stat(f).st_atime)[-1]


def tunnel_to_kernel(
    connection_info: str | KernelConnectionInfo,
    sshserver: str,
    sshkey: str | None = None,
) -> tuple[Any, ...]:
    """tunnel connections to a kernel via ssh

    This will open five SSH tunnels from localhost on this machine to the
    ports associated with the kernel.  They can be either direct
    localhost-localhost tunnels, or if an intermediate server is necessary,
    the kernel must be listening on a public IP.

    Parameters
    ----------
    connection_info : dict or str (path)
        Either a connection dict, or the path to a JSON connection file
    sshserver : str
        The ssh sever to use to tunnel to the kernel. Can be a full
        `user@server:port` string. ssh config aliases are respected.
    sshkey : str [optional]
        Path to file containing ssh key to use for authentication.
        Only necessary if your ssh config does not already associate
        a keyfile with the host.

    Returns
    -------

    (shell, iopub, stdin, hb, control) : ints
        The five ports on localhost that have been forwarded to the kernel.
    """
    from .ssh import tunnel

    if isinstance(connection_info, str):
        # it's a path, unpack it
        with open(connection_info) as f:
            connection_info = json.loads(f.read())

    cf = cast(Dict[str, Any], connection_info)

    lports = tunnel.select_random_ports(5)
    rports = (
        cf["shell_port"],
        cf["iopub_port"],
        cf["stdin_port"],
        cf["hb_port"],
        cf["control_port"],
    )

    remote_ip = cf["ip"]

    if tunnel.try_passwordless_ssh(sshserver, sshkey):
        password: bool | str = False
    else:
        password = getpass("SSH Password for %s: " % sshserver)

    for lp, rp in zip(lports, rports):
        tunnel.ssh_tunnel(lp, rp, sshserver, remote_ip, sshkey, password)

    return tuple(lports)


# -----------------------------------------------------------------------------
# Mixin for classes that work with connection files
# -----------------------------------------------------------------------------

channel_socket_types = {
    "hb": zmq.REQ,
    "shell": zmq.DEALER,
    "iopub": zmq.SUB,
    "stdin": zmq.DEALER,
    "control": zmq.DEALER,
}

port_names = ["%s_port" % channel for channel in ("shell", "stdin", "iopub", "hb", "control")]


class ConnectionFileMixin(LoggingConfigurable):
    """Mixin for configurable classes that work with connection files"""

    data_dir: str | Unicode = Unicode()

    def _data_dir_default(self) -> str:
        return jupyter_data_dir()

    # The addresses for the communication channels
    connection_file = Unicode(
        "",
        config=True,
        help="""JSON file in which to store connection info [default: kernel-<pid>.json]

    This file will contain the IP, ports, and authentication key needed to connect
    clients to this kernel. By default, this file will be created in the security dir
    of the current profile, but can be specified by absolute path.
    """,
    )
    _connection_file_written = Bool(False)

    transport = CaselessStrEnum(["tcp", "ipc"], default_value="tcp", config=True)
    kernel_name: str | Unicode = Unicode()

    context = Instance(zmq.Context)

    ip = Unicode(
        config=True,
        help="""Set the kernel\'s IP address [default localhost].
        If the IP address is something other than localhost, then
        Consoles on other machines will be able to connect
        to the Kernel, so be careful!""",
    )

    def _ip_default(self) -> str:
        if self.transport == "ipc":
            if self.connection_file:
                return os.path.splitext(self.connection_file)[0] + "-ipc"
            else:
                return "kernel-ipc"
        else:
            return localhost()

    @observe("ip")
    def _ip_changed(self, change: Any) -> None:
        if change["new"] == "*":
            self.ip = "0.0.0.0"  # noqa

    # protected traits

    hb_port = Integer(0, config=True, help="set the heartbeat port [default: random]")
    shell_port = Integer(0, config=True, help="set the shell (ROUTER) port [default: random]")
    iopub_port = Integer(0, config=True, help="set the iopub (PUB) port [default: random]")
    stdin_port = Integer(0, config=True, help="set the stdin (ROUTER) port [default: random]")
    control_port = Integer(0, config=True, help="set the control (ROUTER) port [default: random]")

    # names of the ports with random assignment
    _random_port_names: list[str] | None = None

    @property
    def ports(self) -> list[int]:
        return [getattr(self, name) for name in port_names]

    # The Session to use for communication with the kernel.
    session = Instance("jupyter_client.session.Session")

    def _session_default(self) -> Session:
        from .session import Session

        return Session(parent=self)

    # --------------------------------------------------------------------------
    # Connection and ipc file management
    # --------------------------------------------------------------------------

    def get_connection_info(self, session: bool = False) -> KernelConnectionInfo:
        """Return the connection info as a dict

        Parameters
        ----------
        session : bool [default: False]
            If True, return our session object will be included in the connection info.
            If False (default), the configuration parameters of our session object will be included,
            rather than the session object itself.

        Returns
        -------
        connect_info : dict
            dictionary of connection information.
        """
        info = {
            "transport": self.transport,
            "ip": self.ip,
            "shell_port": self.shell_port,
            "iopub_port": self.iopub_port,
            "stdin_port": self.stdin_port,
            "hb_port": self.hb_port,
            "control_port": self.control_port,
        }
        if session:
            # add *clone* of my session,
            # so that state such as digest_history is not shared.
            info["session"] = self.session.clone()
        else:
            # add session info
            info.update(
                {
                    "signature_scheme": self.session.signature_scheme,
                    "key": self.session.key,
                }
            )
        return info

    # factory for blocking clients
    blocking_class = Type(klass=object, default_value="jupyter_client.BlockingKernelClient")

    def blocking_client(self) -> BlockingKernelClient:
        """Make a blocking client connected to my kernel"""
        info = self.get_connection_info()
        bc = self.blocking_class(parent=self)  # type:ignore[operator]
        bc.load_connection_info(info)
        return bc

    def cleanup_connection_file(self) -> None:
        """Cleanup connection file *if we wrote it*

        Will not raise if the connection file was already removed somehow.
        """
        if self._connection_file_written:
            # cleanup connection files on full shutdown of kernel we started
            self._connection_file_written = False
            try:
                os.remove(self.connection_file)
            except (OSError, AttributeError):
                pass

    def cleanup_ipc_files(self) -> None:
        """Cleanup ipc files if we wrote them."""
        if self.transport != "ipc":
            return
        for port in self.ports:
            ipcfile = "%s-%i" % (self.ip, port)
            try:
                os.remove(ipcfile)
            except OSError:
                pass

    def _record_random_port_names(self) -> None:
        """Records which of the ports are randomly assigned.

        Records on first invocation, if the transport is tcp.
        Does nothing on later invocations."""

        if self.transport != "tcp":
            return
        if self._random_port_names is not None:
            return

        self._random_port_names = []
        for name in port_names:
            if getattr(self, name) <= 0:
                self._random_port_names.append(name)

    def cleanup_random_ports(self) -> None:
        """Forgets randomly assigned port numbers and cleans up the connection file.

        Does nothing if no port numbers have been randomly assigned.
        In particular, does nothing unless the transport is tcp.
        """

        if not self._random_port_names:
            return

        for name in self._random_port_names:
            setattr(self, name, 0)

        self.cleanup_connection_file()

    def write_connection_file(self, **kwargs: Any) -> None:
        """Write connection info to JSON dict in self.connection_file."""
        if self._connection_file_written and os.path.exists(self.connection_file):
            return

        self.connection_file, cfg = write_connection_file(
            self.connection_file,
            transport=self.transport,
            ip=self.ip,
            key=self.session.key,
            stdin_port=self.stdin_port,
            iopub_port=self.iopub_port,
            shell_port=self.shell_port,
            hb_port=self.hb_port,
            control_port=self.control_port,
            signature_scheme=self.session.signature_scheme,
            kernel_name=self.kernel_name,
            **kwargs,
        )
        # write_connection_file also sets default ports:
        self._record_random_port_names()
        for name in port_names:
            setattr(self, name, cfg[name])

        self._connection_file_written = True

    def load_connection_file(self, connection_file: str | None = None) -> None:
        """Load connection info from JSON dict in self.connection_file.

        Parameters
        ----------
        connection_file: unicode, optional
            Path to connection file to load.
            If unspecified, use self.connection_file
        """
        if connection_file is None:
            connection_file = self.connection_file
        self.log.debug("Loading connection file %s", connection_file)
        with open(connection_file) as f:
            info = json.load(f)
        self.load_connection_info(info)

    def load_connection_info(self, info: KernelConnectionInfo) -> None:
        """Load connection info from a dict containing connection info.

        Typically this data comes from a connection file
        and is called by load_connection_file.

        Parameters
        ----------
        info: dict
            Dictionary containing connection_info.
            See the connection_file spec for details.
        """
        self.transport = info.get("transport", self.transport)
        self.ip = info.get("ip", self._ip_default())  # type:ignore[assignment]

        self._record_random_port_names()
        for name in port_names:
            if getattr(self, name) == 0 and name in info:
                # not overridden by config or cl_args
                setattr(self, name, info[name])

        if "key" in info:
            key = info["key"]
            if isinstance(key, str):
                key = key.encode()
            assert isinstance(key, bytes)

            self.session.key = key
        if "signature_scheme" in info:
            self.session.signature_scheme = info["signature_scheme"]

    def _reconcile_connection_info(self, info: KernelConnectionInfo) -> None:
        """Reconciles the connection information returned from the Provisioner.

        Because some provisioners (like derivations of LocalProvisioner) may have already
        written the connection file, this method needs to ensure that, if the connection
        file exists, its contents match that of what was returned by the provisioner.  If
        the file does exist and its contents do not match, the file will be replaced with
        the provisioner information (which is considered the truth).

        If the file does not exist, the connection information in 'info' is loaded into the
        KernelManager and written to the file.
        """
        # Prevent over-writing a file that has already been written with the same
        # info.  This is to prevent a race condition where the process has
        # already been launched but has not yet read the connection file - as is
        # the case with LocalProvisioners.
        file_exists: bool = False
        if os.path.exists(self.connection_file):
            with open(self.connection_file) as f:
                file_info = json.load(f)
            # Prior to the following comparison, we need to adjust the value of "key" to
            # be bytes, otherwise the comparison below will fail.
            file_info["key"] = file_info["key"].encode()
            if not self._equal_connections(info, file_info):
                os.remove(self.connection_file)  # Contents mismatch - remove the file
                self._connection_file_written = False
            else:
                file_exists = True

        if not file_exists:
            # Load the connection info and write out file, clearing existing
            # port-based attributes so they will be reloaded
            for name in port_names:
                setattr(self, name, 0)
            self.load_connection_info(info)
            self.write_connection_file()

        # Ensure what is in KernelManager is what we expect.
        km_info = self.get_connection_info()
        if not self._equal_connections(info, km_info):
            msg = (
                "KernelManager's connection information already exists and does not match "
                "the expected values returned from provisioner!"
            )
            raise ValueError(msg)

    @staticmethod
    def _equal_connections(conn1: KernelConnectionInfo, conn2: KernelConnectionInfo) -> bool:
        """Compares pertinent keys of connection info data. Returns True if equivalent, False otherwise."""

        pertinent_keys = [
            "key",
            "ip",
            "stdin_port",
            "iopub_port",
            "shell_port",
            "control_port",
            "hb_port",
            "transport",
            "signature_scheme",
        ]

        return all(conn1.get(key) == conn2.get(key) for key in pertinent_keys)

    # --------------------------------------------------------------------------
    # Creating connected sockets
    # --------------------------------------------------------------------------

    def _make_url(self, channel: str) -> str:
        """Make a ZeroMQ URL for a given channel."""
        transport = self.transport
        ip = self.ip
        port = getattr(self, "%s_port" % channel)

        if transport == "tcp":
            return "tcp://%s:%i" % (ip, port)
        else:
            return f"{transport}://{ip}-{port}"

    def _create_connected_socket(
        self, channel: str, identity: bytes | None = None
    ) -> zmq.sugar.socket.Socket:
        """Create a zmq Socket and connect it to the kernel."""
        url = self._make_url(channel)
        socket_type = channel_socket_types[channel]
        self.log.debug("Connecting to: %s", url)
        sock = self.context.socket(socket_type)
        # set linger to 1s to prevent hangs at exit
        sock.linger = 1000
        if identity:
            sock.identity = identity
        sock.connect(url)
        return sock

    def connect_iopub(self, identity: bytes | None = None) -> zmq.sugar.socket.Socket:
        """return zmq Socket connected to the IOPub channel"""
        sock = self._create_connected_socket("iopub", identity=identity)
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        return sock

    def connect_shell(self, identity: bytes | None = None) -> zmq.sugar.socket.Socket:
        """return zmq Socket connected to the Shell channel"""
        return self._create_connected_socket("shell", identity=identity)

    def connect_stdin(self, identity: bytes | None = None) -> zmq.sugar.socket.Socket:
        """return zmq Socket connected to the StdIn channel"""
        return self._create_connected_socket("stdin", identity=identity)

    def connect_hb(self, identity: bytes | None = None) -> zmq.sugar.socket.Socket:
        """return zmq Socket connected to the Heartbeat channel"""
        return self._create_connected_socket("hb", identity=identity)

    def connect_control(self, identity: bytes | None = None) -> zmq.sugar.socket.Socket:
        """return zmq Socket connected to the Control channel"""
        return self._create_connected_socket("control", identity=identity)


class LocalPortCache(SingletonConfigurable):
    """
    Used to keep track of local ports in order to prevent race conditions that
    can occur between port acquisition and usage by the kernel.  All locally-
    provisioned kernels should use this mechanism to limit the possibility of
    race conditions.  Note that this does not preclude other applications from
    acquiring a cached but unused port, thereby re-introducing the issue this
    class is attempting to resolve (minimize).
    See: https://github.com/jupyter/jupyter_client/issues/487
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.currently_used_ports: set[int] = set()

    def find_available_port(self, ip: str) -> int:
        while True:
            tmp_sock = socket.socket()
            tmp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, b"\0" * 8)
            tmp_sock.bind((ip, 0))
            port = tmp_sock.getsockname()[1]
            tmp_sock.close()

            # This is a workaround for https://github.com/jupyter/jupyter_client/issues/487
            # We prevent two kernels to have the same ports.
            if port not in self.currently_used_ports:
                self.currently_used_ports.add(port)
                return port

    def return_port(self, port: int) -> None:
        if port in self.currently_used_ports:  # Tolerate uncached ports
            self.currently_used_ports.remove(port)


__all__ = [
    "write_connection_file",
    "find_connection_file",
    "tunnel_to_kernel",
    "KernelConnectionInfo",
    "LocalPortCache",
]
